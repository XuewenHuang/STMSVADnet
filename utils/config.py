from glob import glob
import os

if not os.path.exists('tensorboard_log'):
    os.mkdir('tensorboard_log')
if not os.path.exists('weights'):
    os.mkdir('weights')
if not os.path.exists('results'):
    os.mkdir('results')


class train_config():
    def __init__(self, dataset="avenue"):
        self.dataset = dataset
        self.dataset_root = "./datasets/"
        if dataset == "ped2":
            self.train_dataset_path = self.dataset_root + dataset + "/training/"
        elif dataset == "ped1":
            self.train_dataset_path = self.dataset_root + dataset + "/training/frames/"
        elif dataset == "avenue":
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
        self.retrain_path = ""
        self.epochs = 60
        self.batchsize = 8
        self.clip = 16
        self.short_len = 2
        self.save_epochs = 1
        self.eval_epochs = 1
        self.eval_step = 1000
        self.stop_step = 30000
        self.g_learn_rate = 0.0002

        # loss
        self.loss = ['l2', 'grad'] 
        self.eval_score = "psnr"

        self.mod = ''

    def print_cfg(self):
        print(f'Dataset:{self.dataset}')
        print(f'loss:{self.loss}')
        print(f'attention:{self.mod}')
        print(f'img_size:{self.img_size}')
        print(f'clip:{self.clip}')
        print(f'short_len:{self.short_len}')
        print(f'epochs:{self.epochs}')
        print(f'save_epochs:{self.save_epochs}')
        print(f'learn_rate: G {self.g_learn_rate}')
        print("=============================================\n")


class test_config():
    def __init__(self, dataset='avenue'):
        self.dataset = dataset
        self.dataset_root = "./datasets/"
        self.train = False
        self.img_size = (224, 224)
        if dataset == "ped2":
            self.test_dataset_path = self.dataset_root + dataset + "/testing/"
        elif dataset == "ped1":
            self.test_dataset_path = self.dataset_root + dataset + "/testing/frames/"
        elif dataset == "avenue":
            self.test_dataset_path = self.dataset_root + dataset + "/testing/testing_frames/"
        elif dataset == "shanghaitech":
            self.test_dataset_path = self.dataset_root + dataset + "/testing/frames/"
        self.test_mat = self.dataset_root + dataset + "/"
        self.mod = ''
        self.batchsize = 8
        self.clip = 16
