import torch
import torch.utils.data as data
import json
import os
import os.path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import *
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from collections import namedtuple

DB_ROOT = '/home/jwkim/workspace/simple-faster-rcnn-pytorch/datasets/kaist-rgbt/'

JSON_GT_FILE = os.path.join( DB_ROOT, 'kaist_annotations_test20.json' )

DAY_NIGHT_CLS = {
    'set00': 1, 'set01': 1, 'set02': 1,
    'set03': 0, 'set04': 0, 'set05': 0,
    'set06': 1, 'set07': 1, 'set08': 1,
    'set09': 0, 'set10': 0, 'set11': 0,
}

OBJ_CLASSES = [ '__ignore__',   # Object with __backgroun__ label will be ignored.
                'person', 'cyclist', 'people', 'person?']
OBJ_IGNORE_CLASSES = [ 'cyclist', 'people', 'person?' ]
#OBJ_CLS_TO_IDX = { cls:(num-1) for num, cls in enumerate(OBJ_CLASSES)}
OBJ_CLS_TO_IDX = { cls:-1 if cls !='person' else 1 for num, cls in enumerate(OBJ_CLASSES)}

OBJ_LOAD_CONDITIONS = {    
    'train': {'hRng': (50, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}, 
}

#### General
IMAGE_MEAN = (0.3465,  0.3219,  0.2842)
IMAGE_STD = (0.2358, 0.2265, 0.2274)

LWIR_MEAN = (0.1598)
LWIR_STD = (0.0813)

classInfo = namedtuple('TASK', 'detection')
NUM_CLASSES = classInfo( len(set(OBJ_CLASSES)-set(OBJ_IGNORE_CLASSES)) ) # Including background


class KAISTPed(data.Dataset):
    """KAIST Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'KAIST')
        condition (string, optional): load condition
            (default: 'Reasonabel')
    """

    def __init__(self, image_set, img_transform=None, co_transform=None, condition='train'):

        assert condition in OBJ_LOAD_CONDITIONS
        
        self.image_set = image_set
        self.img_transform = img_transform
        self.co_transform = co_transform        
        self.cond = OBJ_LOAD_CONDITIONS[condition]

        self._parser = LoadBox()        
      
        self._annopath = os.path.join('%s', 'annotations-xml-15', '%s', '%s', '%s.xml')
        self._imgpath = os.path.join('%s', 'images', '%s', '%s', '%s', '%s.jpg')  

        self.ids = list()
        for line in open(os.path.join(DB_ROOT, 'imageSets', image_set)):
            self.ids.append((DB_ROOT, line.strip().split('/')))

    def __str__(self):
        return self.__class__.__name__ + '_' + self.image_set

    def __getitem__(self, index): 

        lwir, boxes, labels = self.pull_item(index)
        return lwir, boxes, labels, torch.ones(1,dtype=torch.int)*index  

    def pull_item(self, index):

        frame_id = self.ids[index]
        target = ET.parse(self._annopath % ( *frame_id[:-1], *frame_id[-1] ) ).getroot()
        
        set_id, vid_id, img_id = frame_id[-1]

        # isDay = DAY_NIGHT_CLS[set_id]        

        vis = Image.open( self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id ) )
        lwir = Image.open( self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'lwir', img_id ) ).convert('L')

       
        width, height = lwir.size

        boxes = self._parser(target, width, height)

        ## Apply transforms
        if self.img_transform is not None:
            vis, lwir, _ = self.img_transform(vis, lwir)
        
        if self.co_transform is not None:                    
            vis, lwir, boxes = self.co_transform(vis, lwir, boxes)

        for ii, box in enumerate(boxes):
                        
            y = box[0]
            x = box[1]
            h = box[2]-box[0]
            w = box[3]-box[1]

            if  x < self.cond['xRng'][0] or \
                y < self.cond['xRng'][0] or \
                x+w > self.cond['xRng'][1] or \
                y+h > self.cond['xRng'][1] or \
                w < self.cond['wRng'][0] or \
                w > self.cond['wRng'][1] or \
                h < self.cond['hRng'][0] or \
                h > self.cond['hRng'][1]:

                    boxes[ii, 4] = -1

        boxes_t = boxes[:,0:4] # ymin,xmin,ymax,ymin

        labels = boxes[:,4]
        difficults = boxes[:,5]

        return vis, lwir, boxes_t, labels, difficults

    def __len__(self):
        return len(self.ids)

class LoadBox(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, bbs_format='yxyx'):
        assert bbs_format in ['xyxy', 'xywh', 'yxyx']                
        self.bbs_format = bbs_format
        self.pts = ['x', 'y', 'w', 'h']

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """                
        res = [ [0, 0, 0, 0, -1, 0] ]

        for obj in target.iter('object'):           
            name = obj.find('name').text.lower().strip()            
            bbox = obj.find('bndbox')
            difficult = int(obj.find('difficult').text)
            occlusion = int(obj.find('occlusion').text)
            
            label_idx = OBJ_CLS_TO_IDX[name] if name not in OBJ_IGNORE_CLASSES else -1
            bndbox = [ int(bbox.find(pt).text) for pt in self.pts ]


            if self.bbs_format in ['xyxy']:
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )
            
            elif self.bbs_format in ['yxyx']:
                
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )
                
                bndbox_t = list()
                
                bndbox_t.append(bndbox[1])
                bndbox_t.append(bndbox[0])
                bndbox_t.append(bndbox[3])
                bndbox_t.append(bndbox[2])

                bndbox = bndbox_t

                
            bndbox.append(label_idx)
            bndbox.append(difficult)
            
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind, difficult]
            
        return np.array(res, dtype=np.float)  # [[xmin, ymin, xmax, ymax, label_ind, difficult], ... ]