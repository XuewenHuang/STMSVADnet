import numpy as np
import os
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from models.SelectAttentionUnet import Later_SelectAttention_Net, Med_MS_Attention_Net, \
    Early_SelectAttention_Shared_Net, Early_SelectAttention_NoShared_Net
from utils.config_Train import test_config
from utils.Dataset import save_npy_files
from utils.utils import multi_patch_max_mse, psnr_v, multi_future_frames_to_scores
from utils import Dataset


def val_training(model=None, test_cfg=None):
    if model:
        model = model.eval()
    test_datas = Dataset.test_datas(test_cfg)
    one_folder_len = test_datas.one_folder_len
    test_data_loader = DataLoader(dataset=test_datas, batch_size=test_cfg.batchsize, num_workers=16, shuffle=False,
                                  drop_last=False)
    with torch.no_grad():
        test_len = test_datas.__len__()
        count = 0
        psnr_multi = np.array([], dtype=float)

        for frames, item in test_data_loader:
            count = count + len(item)
            input_frames = frames[:, :-3, :, :].cuda()
            real_frames = frames[:, -3:, :, :].cuda()

            pre_frames = model(input_frames)
            pre_frames = torch.chunk(pre_frames, 2, 1)[0]

            mse_pixel = (((pre_frames + 1) / 2) - ((real_frames + 1) / 2)) ** 2
            mse_32, mse_64, mse_128 = multi_patch_max_mse(mse_pixel.cpu().detach().numpy())
            mse_multi = mse_32 + mse_64 + mse_128
            psnr_multi = np.concatenate((psnr_multi, psnr_v(mse_multi).flatten()), axis=0)
            print(f'\rDetecting : {count}  |  {test_len}', end="")

    gt = Dataset.Label_loader(test_cfg)
    gt = gt()
    psnrs_ = []
    start = 0

    for l in one_folder_len:
        end = start + l
        psnrs_.append(psnr_multi[start: end])
        start = end

    psnr_scores = np.array([], dtype=float)
    labels = np.array([], dtype=int)
    for i in range(len(psnrs_)):
        min_d = min(psnrs_[i])
        max_d = max(psnrs_[i])
        distance = (psnrs_[i] - min_d) / (max_d - min_d)

        psnr_scores = np.concatenate((psnr_scores, distance), axis=0)
        labels = np.concatenate((labels, gt[i][test_cfg.clip + 1:]), axis=0)
    scores = np.concatenate((multi_future_frames_to_scores(psnr_scores)), axis=0)

    assert scores.shape == labels.shape, \
        f'Ground truth has {labels.shape[0]} frames, but got {scores.shape[0]} detected frames.'
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)
    print(f'\nAUC: {auc}\n')
    return auc, fpr, tpr


def val_training_pre1(model=None, test_cfg=None):
    if model:
        model = model.eval()
    test_datas = Dataset.test_datas(test_cfg)
    one_folder_len = test_datas.one_folder_len
    test_data_loader = DataLoader(dataset=test_datas, batch_size=test_cfg.batchsize, num_workers=12, shuffle=False,
                                  drop_last=False, pin_memory=True)
    with torch.no_grad():
        test_len = test_datas.__len__()
        count = 0
        psnr_multi = np.array([], dtype=float)

        for frames, item in test_data_loader:
            count = count + len(item)
            input_frames = frames[:, :-3, :, :].cuda()
            real_frames = frames[:, -3:, :, :].cuda()

            pre_frames = model(input_frames)

            mse_pixel = (((pre_frames + 1) / 2) - ((real_frames + 1) / 2)) ** 2
            mse_32, mse_64, mse_128 = multi_patch_max_mse(mse_pixel.cpu().detach().numpy())
            mse_multi = mse_32 + mse_64 + mse_128
            psnr_multi = np.concatenate((psnr_multi, psnr_v(mse_multi).flatten()), axis=0)
            print(f'\rDetecting : {count}  |  {test_len}', end="")

    gt = Dataset.Label_loader(test_cfg)
    gt = gt()
    psnrs_ = []
    start = 0

    for l in one_folder_len:
        end = start + l
        psnrs_.append(psnr_multi[start: end])
        start = end

    psnr_scores = np.array([], dtype=float)
    labels = np.array([], dtype=int)
    for i in range(len(psnrs_)):
        min_d = min(psnrs_[i])
        max_d = max(psnrs_[i])
        distance = (psnrs_[i] - min_d) / (max_d - min_d)

        psnr_scores = np.concatenate((psnr_scores, distance), axis=0)
        labels = np.concatenate((labels, gt[i][test_cfg.clip + 1:]), axis=0)
    scores = np.concatenate((multi_future_frames_to_scores(psnr_scores)), axis=0)

    assert scores.shape == labels.shape, \
        f'Ground truth has {labels.shape[0]} frames, but got {scores.shape[0]} detected frames.'
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)
    print(f'\nAUC: {auc}\n')
    return auc, fpr, tpr


def val_notraining(test_cfg):
    if (test_cfg.mod == 'mp_att') or (test_cfg.mod == 'rfb_att'):
        model = MS_Attention_Net(input_channels=test_cfg.clip * 3, output_channel=6,
                                     attention_type=test_cfg.mod).cuda()
        print(f"{test_cfg.mod} model")
    elif ('early' in test_cfg.mod) and ('noshared' in test_cfg.mod):
        model = Early_SelectAttention_NoShared_Net(input_channels=test_cfg.clip * 3, output_channel=6,
                                                       attention_type=test_cfg.mod).cuda()
        print(f"{test_cfg.mod} model")
    elif ('early' in test_cfg.mod) and ('shared' in test_cfg.mod):
        model = Early_SelectAttention_Shared_Net(input_channels=test_cfg.clip * 3, output_channel=6,
                                                     attention_type=test_cfg.mod).cuda()
        print(f"{test_cfg.mod} model")
    elif ('med' in test_cfg.mod) and ('noshared' in test_cfg.mod):
        model = Med_MS_Attention_Net(input_channels=test_cfg.clip * 3, output_channel=6,
                                         attention_type=test_cfg.mod).cuda()
        print(f"{test_cfg.mod} model")
    elif ('med' in test_cfg.mod) and ('shared' in test_cfg.mod):
        model = Med_MS_Attention_Net(input_channels=test_cfg.clip * 3, output_channel=6,
                                         attention_type=test_cfg.mod).cuda()
        print(f"{test_cfg.mod} model")
    elif ('later' in test_cfg.mod):
        model = Later_SelectAttention_Net(input_channels=test_cfg.clip * 3, output_channel=6,
                                              attention_type=test_cfg.mod).cuda()
        print(f"{test_cfg.mod} model")

    model.load_state_dict(torch.load(f'./weights/{test_cfg.dataset}/later_rfb_eca_conv_att_avenue_clip16sl4_bs8_14_0.855.pth')['net_g'])
    model = model.eval()
    test_datas = Dataset.test_datas(test_cfg)
    one_folder_len = test_datas.one_folder_len
    test_data_loader = DataLoader(dataset=test_datas, batch_size=test_cfg.batchsize, num_workers=12, shuffle=False,
                                  drop_last=False)
    with torch.no_grad():
        test_len = test_datas.__len__()

        count = 0
        psnr_multi = np.array([], dtype=float)
        for frames, item in test_data_loader:
            count = count + len(item)
            input_frames = frames[:, :-3, :, :].cuda()
            real_frames = frames[:, -3:, :, :].cuda()

            pre_frames = model(input_frames)
            pre_frames = torch.chunk(pre_frames, 2, 1)[0]

            mse_pixel = (((pre_frames + 1) / 2) - ((real_frames + 1) / 2)) ** 2
            mse_32, mse_64, mse_128 = multi_patch_max_mse(mse_pixel.cpu().detach().numpy())
            mse_multi = mse_32 + mse_64 + mse_128
            psnr_multi = np.concatenate((psnr_multi, psnr_v(mse_multi).flatten()), axis=0)
            print(f'\rDetecting : {count}  |  {test_len}', end="")

    gt = Dataset.Label_loader(test_cfg)
    gt = gt()
    psnrs_ = []
    start = 0
    for l in one_folder_len:
        end = start + l
        psnrs_.append(psnr_multi[start: end])
        start = end

    psnr_scores = np.array([], dtype=float)
    labels = np.array([], dtype=int)
    for i in range(len(psnrs_)):
        min_d = min(psnrs_[i])
        max_d = max(psnrs_[i])
        distance = (psnrs_[i] - min_d) / (max_d - min_d)

        psnr_scores = np.concatenate((psnr_scores, distance), axis=0)
        labels = np.concatenate((labels, gt[i][test_cfg.clip + 1:]), axis=0)

    scores = np.concatenate((multi_future_frames_to_scores(psnr_scores)), axis=0)
    assert scores.shape == labels.shape, \
        f'Ground truth has {labels.shape[0]} frames, but got {scores.shape[0]} detected frames.'
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)
    print(f'\nAUC: {auc}\n')
    if not os.path.exists(f'./npys/{test_cfg.dataset}/scores/'):
        os.makedirs(f'./npys/{test_cfg.dataset}/scores/')
    save_npy_files(path=f'./npys/{test_cfg.dataset}/scores/{test_cfg.dataset}_psnrl2_{auc:.4f}.npy', fpr=fpr, tpr=tpr,
                   auc=auc)
    return auc, fpr, tpr


if __name__ == "__main__":
    test_cfg = test_config()
    val_notraining(test_cfg)
