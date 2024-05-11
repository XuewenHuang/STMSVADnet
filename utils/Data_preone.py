import random
import torch
import numpy as np
import cv2
import glob
import os
import scipy.io as scio
from torch.utils.data import Dataset


def save_npy_files(path, auc, fpr, tpr):
    data = {}
    data['auc'] = auc
    data["fpr"] = fpr
    data['tpr'] = tpr
    np.save(path, data)


def process_frame(frame_path, resize_w, resize_h):
    img = cv2.imread(frame_path)
    image_resized = cv2.resize(img, (resize_w, resize_h)).astype('float32')
    image_resized = (image_resized / 127.5) - 1.  # to -1 ~ 1

    image_resized = np.transpose(image_resized, [2, 0, 1])  # to (C, W, H)
    return image_resized


def np_load_frame(filename, resize_h, resize_w):
    img = cv2.imread(filename)
    image_resized = cv2.resize(img, (resize_w, resize_h)).astype('float32')
    image_resized = (image_resized / 127.5) - 1.0  # to -1 ~ 1
    image_resized = np.transpose(image_resized, [2, 0, 1])  # to (C, W, H)
    return image_resized


class train_datas(Dataset):
    def __init__(self, train_config):
        self.clip_length = train_config.clip
        self.img_size = train_config.img_size

        self.img_w = self.img_size[0]
        self.img_h = self.img_size[1]
        self.train_len = 0

        self.train_frame_imgs = []

        for folder in sorted(glob.glob(f'{train_config.train_dataset_path}/*')):
            one_folder_frame_imgs = glob.glob(f'{folder}/*.jpg')
            one_folder_frame_imgs.sort()
            for i in range(0, len(one_folder_frame_imgs) - 1 - self.clip_length):
                one_clip = []
                for index in range(i, i + self.clip_length+1):
                    # print(one_folder_frame_imgs[i])
                    one_clip.append(one_folder_frame_imgs[index])
                self.train_frame_imgs.append(one_clip)

    def __len__(self):
        self.train_len = len(self.train_frame_imgs)
        return self.train_len

    def __getitem__(self, item):
        frame_clips = []
        # i = 0
        for elem in self.train_frame_imgs[item]:
            frames = process_frame(elem, self.img_w, self.img_h)
            frame_clips.append(frames)
        frames_clip = torch.from_numpy(np.array(frame_clips).reshape((-1, self.img_w, self.img_h)))

        return frames_clip, item


class test_datas(Dataset):
    def __init__(self, test_config):
        super().__init__()
        self.clip_length = test_config.clip
        self.img_size = test_config.img_size
        self.img_w = self.img_size[0]
        self.img_h = self.img_size[1]
        self.test_len = 0
        self.one_folder_len = []

        self.test_frame_imgs = []

        for folder in sorted(glob.glob(f'{test_config.test_dataset_path}/*')):
            one_folder_frame_imgs = glob.glob(f'{folder}/*.jpg')
            one_folder_frame_imgs.sort()
            self.one_folder_len.append(len(one_folder_frame_imgs) - self.clip_length - 1)

            for i in range(0, len(one_folder_frame_imgs) - self.clip_length-1):
                one_clip = []
                for index in range(i, i + self.clip_length + 1):
                    one_clip.append(one_folder_frame_imgs[index])
                self.test_frame_imgs.append(one_clip)

    def __len__(self):
        self.test_len = len(self.test_frame_imgs)
        return self.test_len

    def __getitem__(self, item):
        frame_clips = []
        for elem in self.test_frame_imgs[item]:
            frame_clips.append(process_frame(elem, self.img_w, self.img_h))
        frames_clip = torch.from_numpy(np.array(frame_clips).reshape((-1, self.img_w, self.img_h)))

        return frames_clip, item


class Label_loader:
    def __init__(self, cfg):
        assert cfg.dataset in (
            'ped2', 'avenue', 'avenue_enhance', 'shanghaitech',
            'ped1'), f'Did not find the related gt for \'{cfg.dataset}\'.'
        self.cfg = cfg
        if cfg.dataset == "avenue_enhance":
            self.name = "avenue"
        else:
            self.name = cfg.dataset
        self.mat_path = f'{cfg.test_mat}/{self.name}.mat'
        self.video_folders = []

        for folder in sorted(glob.glob(f'{cfg.test_dataset_path}/*')):
            self.video_folders.append(folder)

    def __call__(self):
        if self.name == 'shanghaitech':
            gt = self.load_shanghaitech()
        else:
            gt = self.load_ucsd_avenue()
        return gt

    def load_ucsd_avenue(self):
        abnormal_events = scio.loadmat(self.mat_path, squeeze_me=True)['gt']

        all_gt = []
        for i in range(abnormal_events.shape[0]):
            length = len(os.listdir(self.video_folders[i]))
            sub_video_gt = np.zeros((length,), dtype=np.int8)

            one_abnormal = abnormal_events[i]
            if one_abnormal.ndim == 1:
                one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))

            for j in range(one_abnormal.shape[1]):
                start = one_abnormal[0, j] - 1
                end = one_abnormal[1, j]

                sub_video_gt[start: end] = 1

            all_gt.append(sub_video_gt)

        return all_gt

    def load_shanghaitech(self):
        np_list = glob.glob('')
        np_list.sort()

        gt = []
        for npy in np_list:
            gt.append(np.load(npy))

        return gt
