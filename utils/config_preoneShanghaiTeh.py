#!/usr/bin/env python
# -*- coding:utf-8 -*-
from glob import glob
import os

if not os.path.exists('tensorboard_log'):
    os.mkdir('tensorboard_log')
if not os.path.exists('weights'):
    os.mkdir('weights')
if not os.path.exists('results'):
    os.mkdir('results')


class train_config():
    def __init__(self, dataset="shanghaitech"):
        self.dataset = dataset
        self.dataset_root = '/home/hxw/Python/TFUnet/datasets/'
        if dataset == "ped2":
            self.train_dataset_path = self.dataset_root + dataset + "/training/"
        elif dataset == "ped1":
            self.train_dataset_path = self.dataset_root + dataset + "/training/frames/"
        elif dataset == "avenue":
            self.train_dataset_path = self.dataset_root + dataset + "/training/training_frames/"
        elif dataset == "avenue_enhance":
            self.train_dataset_path = self.dataset_root + dataset + "/training/training_frames/"
        elif dataset == "shanghaitech":
            self.train_dataset_path = self.dataset_root + dataset + "/training/frames/"

        self.eval_dataset_path = self.dataset_root + dataset + "/testing/"
        self.logs_path = "./tensorboard_log/"
        self.wights_path = './weights/' + dataset
        self.img_size = (224, 224)  # 224-14 256-16

        # train config
        self.retrain = False
        self.retrain_epoch = 0
        self.retrain_path = "./weights/TFUnet/ped2/gauss0.0_ped2_l2grad_dp6dim256mlpd128h4ps14_96.8.pth"
        self.flownet = "lite1"  # flowsd2
        self.epochs = 5
        self.batchsize = 8
        self.clip = 16
        self.save_epochs = 1
        self.eval_epochs = 1
        self.eval_step = 1500
        self.stop_step = 15000
        self.g_learn_rate = 0.0002

        # loss
        self.loss = ['l2', 'grad']  # "l2+grad"
        self.eval_score = "psnr"
        self.use_GAN = False

        """"
        param: 
        MS: ms rfb aspp 
        Attention: att cbam eca
        1：ms_att                   2：early_ms_cbam_shared            3：early_ms_cbam_noshared   
        4：later_ms_cbam_conv_att   5：med_ms_cbam_noshared_conv_att   6：med_ms_cbam_shared_conv_att
        7：later_ms_cbam_att_conv   8：med_ms_cbam_noshared_att_conv   9：med_ms_cbam_shared_att_conv 
        """
        self.mod = 'later_mp_eca_conv_att'   #5   med_rfb_eca_shared_att_conv

    def print_cfg(self):
        print(f'Dataset:{self.dataset}')
        print(f'loss:{self.loss}')
        print(f'attention:{self.mod}')
        print(f'img_size:{self.img_size}')
        print(f'clip:{self.clip}')
        print(f'epochs:{self.epochs}')
        print(f'save_epochs:{self.save_epochs}')
        print(f'learn_rate: G {self.g_learn_rate}')
        print("=============================================\n")


class test_config():
    def __init__(self, dataset='shanghaitech'):
        self.dataset = dataset
        self.dataset_root = '/home/hxw/Python/TFUnet/datasets/'
        self.train = False
        self.img_size = (224, 224)
        if dataset == "ped2":
            self.test_dataset_path = self.dataset_root + dataset + "/testing/"
        elif dataset == "ped1":
            self.test_dataset_path = self.dataset_root + dataset + "/testing/frames/"
        elif dataset == "avenue":
            self.test_dataset_path = self.dataset_root + dataset + "/testing/testing_frames/"
        elif dataset == "avenue_enhance":
            self.test_dataset_path = self.dataset_root + dataset + "/testing/testing_frames/"
        elif dataset == "shanghaitech":
            self.test_dataset_path = self.dataset_root + dataset + "/testing/frames/"
        self.test_mat = self.dataset_root + dataset + "/"
        self.batchsize = 8
        self.clip = 16
