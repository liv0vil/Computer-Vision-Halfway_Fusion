from __future__ import  absolute_import
import numpy as np
import os
import tqdm
print(tqdm.__version__)
from tqdm import tqdm
from utils.config import opt
from data.dataset import Dataset, TestDataset
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from datetime import datetime
import os.path
from torchcv.evaluations.coco import COCO
import torch
from torchcv.utils import write_coco_format as write_result_coco

annType = 'bbox'
DB_ROOT = './datasets/kaist-rgbt'
JSON_GT_FILE = os.path.join( DB_ROOT, 'kaist_annotations_test20.json' )
cocoGt = COCO(JSON_GT_FILE)

# Parameters
checkpoint_root = './checkpoints'
data_folder = './datasets/kaist-rgbt/'
input_size = [512., 640.]

# Load data
# load_path = './checkpoints/2019-08-30_16h41m_train_halfway_check_neglow01/fasterrcnn_1'
load_path = None

# random seed fix 
torch.manual_seed(12)
torch.cuda.manual_seed(12)
torch.backends.cudnn.deterministic=True
np.random.seed(12)

def eval(dataloader, faster_rcnn, epoch, jobs_dir):
    print('\nstart_evaluation\n')
    result_test = []
    for ii, (vis_test_image, lwir_test_iamge, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader), desc='Evaluating'):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(vis_test_image, lwir_test_iamge, [sizes])
        for box, label, score in zip(pred_bboxes_[0],pred_labels_[0],pred_scores_[0]) :
            bb = box.tolist()
            result_test.append( {\
                            'image_id': ii, \
                            'category_id': label.item()+1, \
                            'bbox': [bb[1],bb[0],bb[3]-bb[1],bb[2]-bb[0]], \
                            'score': score.item() } )
    print('\nCOCO_Missrate_test\n')
    rstFile = os.path.join(jobs_dir, 'COCO_TEST_det_{:d}.json'.format(epoch))            
    write_result_coco(result_test , rstFile)
    print("The results are saved, please check the checkpoints folder.")

def train(**kwargs):
    opt._parse(kwargs)
    exp_time  = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
    jobs_dir = os.path.join( 'checkpoints', exp_time + opt.exp_name )
    if not os.path.exists(jobs_dir): os.makedirs(jobs_dir)
    print('Setting..')
    print('load data')
    dataset = Dataset(opt)
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    pass_count = 0
    lr_ = opt.lr
    if load_path is not None :
        trainer.load(load_path)
        print('load pretrained model from %s' % load_path)
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        forward_count = 0
        pass_count = 0
        for ii, (vis_image, lwir_iamge, bbox_, label_, scale) in tqdm(enumerate(dataloader), desc='Training'):


            try :
                if len(bbox_[0]) == 1 :
                    pass_count += 1
                    continue
                else :
                    bbox_ = bbox_[label_==1][None,:,:]
                    label_ = label_[label_==1][None,:] -1
                    if len(bbox_[0]) < 1 :
                        pass_count += 1
                        continue
                    forward_count += 1
            except :
                import pdb; pdb.set_trace()
            scale = at.scalar(scale)
            vis_image, lwir_iamge, bbox, label = vis_image.cuda().float(), lwir_iamge.cuda().float(), bbox_.cuda(), label_.cuda()          
            trainer.train_step(vis_image, lwir_iamge, bbox, label, scale)

        print('\nPass Count  : {:d}   /// Training Image : {:d}\n'.format(pass_count,(len(dataloader)-pass_count)))

        eval(test_dataloader, faster_rcnn, epoch, jobs_dir)

        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']

        save_path = trainer.save(epoch=epoch,save_path=jobs_dir)
        
        if epoch > 1 :
            if epoch % 4 == 0:
                trainer.load(save_path)
                trainer.faster_rcnn.scale_lr(opt.lr_decay)
                lr_ = lr_ * opt.lr_decay


if __name__ == '__main__':
    import fire
    fire.Fire()
