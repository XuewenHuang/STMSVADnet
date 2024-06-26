import os
from glob import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models.SelectAttentionUnet import *

from evaluate import val_training
from utils.utils import *
from utils.losses import *
from utils import Dataset
from utils import config


def train(train_cfg, test_cfg):
    train_cfg.print_cfg()
    ll = ""
    for l in train_cfg.loss:
        ll = ll + l

    save_files_path = f'{train_cfg.mod}_{train_cfg.dataset}_{ll}_ims{train_cfg.img_size[0]}clip{train_cfg.clip}sl{train_cfg.short_len}_bs{train_cfg.batchsize}_{train_cfg.epochs}'
    weights_path = f'./weights/{train_cfg.dataset}/{save_files_path}'
    npys_path = f'./npys/{train_cfg.dataset}/{save_files_path}'
    logs_path = f'./tensorboard_log/{train_cfg.dataset}'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    if not os.path.exists(npys_path):
        os.makedirs(npys_path)

    if (train_cfg.mod == 'ms_att') or (train_cfg.mod == 'rfb_att'):
        generator = MS_Attention_Net(input_channels=train_cfg.clip * 3, output_channel=6,
                                     attention_type=train_cfg.mod).cuda()
    elif ('early' in train_cfg.mod) and ('noshared' in train_cfg.mod):
        generator = Early_SelectAttention_NoShared_Net(input_channels=train_cfg.clip * 3, output_channel=6,
                                                       attention_type=train_cfg.mod).cuda()
    elif ('early' in train_cfg.mod) and ('shared' in train_cfg.mod):
        generator = Early_SelectAttention_Shared_Net(input_channels=train_cfg.clip * 3, output_channel=6,
                                                     attention_type=train_cfg.mod).cuda()
    elif ('med' in train_cfg.mod) and ('noshared' in train_cfg.mod):
        generator = Med_MS_Attention_Net(input_channels=train_cfg.clip * 3, output_channel=6,
                                         attention_type=train_cfg.mod).cuda()
    elif ('med' in train_cfg.mod) and ('shared' in train_cfg.mod):
        generator = Med_MS_Attention_Net(input_channels=train_cfg.clip * 3, output_channel=6,
                                         attention_type=train_cfg.mod).cuda()
    elif ('later' in train_cfg.mod):
        generator = Later_SelectAttention_Net(input_channels=train_cfg.clip * 3, output_channel=6,
                                              attention_type=train_cfg.mod).cuda()

    print(f"{train_cfg.mod} model")

    optimizer_G = torch.optim.AdamW(generator.parameters(), lr=train_cfg.g_learn_rate, betas=(0.9, 0.999),
                                    eps=1e-8)  # 生成器优化器
    # scheduler_l = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=train_cfg.epochs)

    generator.apply(weights_init_normal)
    print('Generator are going to be trained from scratch.')

    if 'grad' in train_cfg.loss:
        gradient_loss = Gradient_Loss(3).cuda()  # 梯度损失
    if 'l2' in train_cfg.loss:
        # intensity_loss = Intensity_Loss().cuda()  # 真实与预测帧之间的差异   强度损失
        intensity_loss = nn.MSELoss(reduction='none')

    train_dataset = Dataset.train_datas(train_cfg)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batchsize,
                                  shuffle=True, num_workers=8, drop_last=True)

    writer = SummaryWriter(f'{logs_path}/{save_files_path}')
    start_iter = int(train_cfg.retrain_epoch) if train_cfg.retrain_epoch else 0
    generator = generator.train()

    try:
        start = start_iter
        step = 0
        current_max_auc = 0

        for epoch in range(start, train_cfg.epochs):
            loop = tqdm(train_dataloader)
            for clips, item in loop:
                step = step + 1
                input_frames = clips[:, :-6, :, :].cuda()  # (n, 12, 256, 256)   [:, 0:18, :, :]
                target_frame1 = clips[:, -6:-3, :, :].cuda()  # [:, 18:21, :, :]
                target_frame2 = clips[:, -3:, :, :].cuda()  # [:, 21:24, :, :]

                G_frame = generator(input_frames)  # 使用Unet预测帧，预测帧为G_frame
                G_split = torch.chunk(G_frame, 2, 1)
                G_frame1 = G_split[0]
                G_frame2 = G_split[1]

                if 'l2' in train_cfg.loss:
                    inte_l1 = torch.mean(intensity_loss(G_frame1, target_frame1))
                    inte_l2 = torch.mean(intensity_loss(G_frame2, target_frame2))
                    inte_l = 0.8 * inte_l1 + 0.2 * inte_l2  # torch.mean(torch.abs((gen_frames - gt_frames) ** 2))
                if 'grad' in train_cfg.loss:
                    grad_l1 = gradient_loss(G_frame1, target_frame1)
                    grad_l2 = gradient_loss(G_frame2, target_frame2)
                    grad_l = 0.8 * grad_l1 + 0.2 * grad_l2  # 梯度损失

                G_l_t = 1. * inte_l + 1. * grad_l
                if use_flow:
                    G_l_t = G_l_t + 2. * fl_l

                optimizer_G.zero_grad()
                G_l_t.backward()
                optimizer_G.step()

                torch.cuda.synchronize()

                if step % 20 == 1:
                    psnr1 = torch.mean(psnr_error(G_frame1, target_frame1))
                    psnr2 = torch.mean(psnr_error(G_frame2, target_frame2))
                    psnr = psnr1 + psnr2
                    writer.add_scalar('psnr/train_psnr', psnr, global_step=step)
                    writer.add_scalar('psnr/train_psnr1', psnr1, global_step=step)
                    writer.add_scalar('psnr/train_psnr2', psnr2, global_step=step)

                loop.set_description(f'Epoch[{epoch}/{train_cfg.epochs}]')


                loop.set_postfix(
                    info=f'step:{step} | psnr:{psnr1:.3f},{psnr2:.3f} G:{G_l_t:.3f} | inte:{inte_l1:.3f},{inte_l2:.3f} | grad:{grad_l1:.3f},{grad_l2:.3f} ')

                writer.add_scalar('total_loss/g_loss_total', G_l_t, global_step=step)  # 总损失
                if 'l2' in train_cfg.loss:
                    writer.add_scalar('G_loss_total/inte_loss', inte_l, global_step=step)  # 烈度损失
                    writer.add_scalar('G_loss_total/inte_loss1', inte_l1, global_step=step)
                    writer.add_scalar('G_loss_total/inte_loss2', inte_l2, global_step=step)
                elif 'grad' in train_cfg.loss:
                    writer.add_scalar('G_loss_total/grad_loss', grad_l, global_step=step)  # 烈度损失
                    writer.add_scalar('G_loss_total/grad_loss1', grad_l1, global_step=step)
                    writer.add_scalar('G_loss_total/grad_loss2', grad_l2, global_step=step)

                if train_cfg.dataset == 'avenue' or train_cfg.dataset == 'shanghaitech':
                    if step % train_cfg.eval_step == 0:
                        auc, fpr, tpr = val_training(model=generator, test_cfg=test_cfg)
                        if auc >= current_max_auc:
                            current_max_auc = auc
                            model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict()}
                            torch.save(model_dict,
                                       f'{weights_path}/{train_cfg.mod}_{train_cfg.dataset}_clip{train_cfg.clip}sl{train_cfg.short_len}_bs{train_cfg.batchsize}_{step}_{auc:.3f}.pth')
                        print(f'Already saved: {train_cfg.dataset}_{step}.pth')
                        writer.add_scalar('results/auc', auc, global_step=step)

                        save_path = f'{npys_path}/{train_cfg.mod}_{train_cfg.dataset}_clip{train_cfg.clip}sl{train_cfg.short_len}_bs{train_cfg.batchsize}_{step}_{auc:.3f}.npy'
                        Dataset.save_npy_files(path=save_path, auc=auc, fpr=fpr, tpr=tpr)
                    if step * train_cfg.stop_step == 0:
                        break
            if train_cfg.dataset == 'ped2':
                if epoch % train_cfg.save_epochs == 0:
                    auc, fpr, tpr = val_training(model=generator, test_cfg=test_cfg)
                    if auc >= current_max_auc:
                        current_max_auc = auc
                        model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict()}
                        torch.save(model_dict,
                                   f'{weights_path}/{train_cfg.mod}_{train_cfg.dataset}_clip{train_cfg.clip}sl{train_cfg.short_len}_bs{train_cfg.batchsize}_{epoch}_{auc:.3f}.pth')
                    print(f'Already saved: {train_cfg.dataset}_{epoch}.pth')
                    writer.add_scalar('results/auc', auc, global_step=step)

                    save_path = f'{npys_path}/{train_cfg.mod}_{train_cfg.dataset}_clip{train_cfg.clip}sl{train_cfg.short_len}_bs{train_cfg.batchsize}_{epoch}_{auc:.3f}.npy'
                    Dataset.save_npy_files(path=save_path, auc=auc, fpr=fpr, tpr=tpr)
            if step % train_cfg.stop_step == 0:
                break

    except KeyboardInterrupt:
        print(f'\nStop early, model saved: \'latest_{train_cfg.dataset}_{epoch}.pth\'.\n')

        if glob(f'weights/latest*'):
            os.remove(glob(f'weights/{train_cfg.dataset}/latest*')[0])

        model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict()}

        torch.save(model_dict,
                   f'weights/{train_cfg.dataset}/latest_{train_cfg.mod}_{train_cfg.dataset}_{epoch}.pth')


if __name__ == "__main__":
    short_len = [2, 4, 8, 12, 16]
    ext_mod = ['', '']

    train_cfg = config.train_config()
    test_cfg = config.test_config()

    for m in ext_mod:
        train_cfg.mod = m
        for s in short_len:
            train_cfg.short_len = s
            train(train_cfg=train_cfg, test_cfg=test_cfg)


