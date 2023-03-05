import os
import os.path as osp
import sys
import numpy as np

class Config:
    
    ## dataset
    trainset = 'MSCOCO'
    testset = 'MSCOCO'

    ## model setting
    resnet_type = 50 # 50, 101, 152
    
    ## input, output
    #input_img_shape = (1333, 1333) 
    input_img_shape = (1024,1024)
    #output_hm_shape = (64, 64, 64)
    num_max_bbox = 20
    feature_map_resolutions = (16,32,64,128,256)
    anchor_strides = (64,32,16,8,4)
    anchor_aspect_ratios = (0.5,1.,2.)
    anchor_fm_scale = 8

    IoU_pos_thresh = 0.7
    IoU_neg_thresh = 0.3
    sampling_anchor_num = 64
    positive_fraction = 0.5
    
    sigma = 2.5

    ## training config
    lr_dec_epoch = [10,12]
    end_epoch = 13
    lr = 1e-4
    lr_dec_factor = 10
    train_batch_size = 5
    normal_loss_weight = 0.1

    ## testing config
    test_batch_size = 2
    use_gt_info = False

    ## others
    num_thread = 8
    gpu_ids = '1'
    num_gpus = 1
    stage = 'lixel' # lixel, param
    continue_train = False
    
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    mano_path = osp.join(root_dir, 'common', 'utils', 'manopth')
    smpl_path = osp.join(root_dir, 'common', 'utils', 'smplpytorch')
    
    def set_args(self, gpu_ids, stage='lixel', continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.stage = stage
        # extend training schedule
        if self.stage == 'param':
            self.lr_dec_epoch = [x+5 for x in self.lr_dec_epoch]
            self.end_epoch = self.end_epoch + 5
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))
        
        if self.testset == 'FreiHAND':
            assert self.trainset_3d[0] == 'FreiHAND'
            assert len(self.trainset_3d) == 1
            assert len(self.trainset_2d) == 0

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.trainset))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
# import yaml

# cfg = edict()
