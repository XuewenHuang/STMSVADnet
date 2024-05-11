import os
from glob import glob
from tqdm import tqdm

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models.Unet.SelectAttentionUnet import *
from models.liteFlownet import lite_flownet as lite_flow
from models.flownet2.models import FlowNet2SD

from evaluate import val_training_pre1
from utils.utils import *
from utils.losses import *
from utils import Data_preone as Dataset
from utils import config_preone as config

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def train(train_cfg, test_cfg):
    train_cfg.print_cfg()
    ll = ""
    for l in train_cfg.loss:
        ll = ll + l

    save_files_path = f'{train_cfg.mod}_pre1_{train_cfg.dataset}_{ll}_ims{train_cfg.img_size[0]}clip{train_cfg.clip}_bs{train_cfg.batchsize}_{train_cfg.epochs}'
    weights_path = f'./weights/{train_cfg.dataset}/{save_files_path}'
    npys_path = f'./npys/{train_cfg.dataset}/{save_files_path}'
    logs_path = f'./tensorboard_log/{train_cfg.dataset}'
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    if not os.path.exists(npys_path):
        os.makedirs(npys_path)

    if (train_cfg.mod == 'mp_att') or (train_cfg.mod == 'rfb_att') or (train_cfg.mod == 'aspp_att'):
        generator = MS_Attention_Net(input_channels=train_cfg.clip * 3, output_channel=3,
                                     attention_type=train_cfg.mod).cuda()
    elif ('early' in train_cfg.mod) and ('shared' in train_cfg.mod):
        generator = Early_SelectAttention_Shared_Net(input_channels=train_cfg.clip * 3, output_channel=3,
                                                     attention_type=train_cfg.mod).cuda()
    elif ('early' in train_cfg.mod) and ('noshared' in train_cfg.mod):
        generator = Early_SelectAttention_NoShared_Net(input_channels=train_cfg.clip * 3, output_channel=3,
                                                       attention_type=train_cfg.mod).cuda()

    elif ('med' in train_cfg.mod) and ('shared' in train_cfg.mod):
        generator = Med_MS_Attention_Net(input_channels=train_cfg.clip * 3, output_channel=3,
                                         attention_type=train_cfg.mod).cuda()
    elif ('med' in train_cfg.mod) and ('noshared' in train_cfg.mod):
        generator = Med_MS_Attention_Net(input_channels=train_cfg.clip * 3, output_channel=3,
                                         attention_type=train_cfg.mod).cuda()
    elif ('later' in train_cfg.mod):
        generator = Later_SelectAttention_Net(input_channels=train_cfg.clip * 3, output_channel=3,
                                              attention_type=train_cfg.mod).cuda()

    print(f"{train_cfg.mod} model")

    # discriminator = PixelDiscriminator(input_nc=6).cuda()  # 鉴别器
    optimizer_G = torch.optim.AdamW(generator.parameters(), lr=train_cfg.g_learn_rate, betas=(0.9, 0.999),
                                    eps=1e-8)  # 生成器优化器
    # scheduler_l=ReduceLROnPlateau(optimizer_G,mode="min",factor=0.1,patience=2)
    # optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=train_cfg.d_learn_rate)  # 鉴别器优化器

    # 初始化权重，接上次或重新初始化
    if train_cfg.retrain_epoch:  # 接上一次训练
        generator.load_state_dict(torch.load(train_cfg.retrain_path)['net_g'])
        # discriminator.load_state_dict(torch.load(train_cfg.retrain_path)['net_d'])
        optimizer_G.load_state_dict(torch.load(train_cfg.retrain_path)['optimizer_g'])
        # optimizer_D.load_state_dict(torch.load(train_cfg.retrain_path)['optimizer_d'])
        print(f'Pre-trained generator have been loaded.\n')
    else:
        generator.apply(weights_init_normal)  # 初始权重，生成器Unet
        # discriminator.apply(weights_init_normal)
        print('Generator are going to be trained from scratch.')

    # 选择光流网络
    use_flow = False
    if 'flow' in train_cfg.loss:
        use_flow = True
        assert train_cfg.flownet in ('lite1', '2sd'), 'Flow net only supports LiteFlownet or FlowNet2SD currently.'
        if train_cfg.flownet == '2sd':
            flow_net = FlowNet2SD()
            flow_net.load_state_dict(torch.load('models/flownet2/FlowNet2-SD.pth')['state_dict'])
        else:
            flow_net = lite_flow.Network()
            flow_net.load_state_dict(torch.load('models/liteFlownet/network-default.pytorch'))

        # 光流网络不做训练，迁移学习
        flow_net.cuda().eval()  # Use flow_net to generate optic flows, so set to eval mode.
        flow_loss = Flow_Loss().cuda()  # 光流损失
        print("FlowNet load finished")

    # if train_cfg.use_GAN:
    #     adversarial_loss = Adversarial_Loss().cuda()
    #     discriminate_loss = Discriminate_Loss().cuda()  # 鉴别器损失，比较真实与生成
    if 'grad' in train_cfg.loss:
        gradient_loss = Gradient_Loss(3).cuda()  # 梯度损失
    if 'l2' in train_cfg.loss:
        intensity_loss = Intensity_Loss().cuda()  # 真实与预测帧之间的差异   强度损失

    train_dataset = Dataset.train_datas(train_cfg)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batchsize,
                                  shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    writer = SummaryWriter(
        f'{logs_path}/{save_files_path}')
    start_iter = int(train_cfg.retrain_epoch) if train_cfg.retrain_epoch else 0
    generator = generator.train()
    # discriminator = discriminator.train()

    try:
        start = start_iter
        step = 0
        current_max_auc = 0

        for epoch in range(start, train_cfg.epochs):
            loop = tqdm(train_dataloader)
            for clips, item in loop:
                step = step + 1
                input_frames = clips[:, :-3, :, :].cuda()  # (n, 12, 256, 256)   [:, 0:18, :, :]
                pre_frame = clips[:, -6:-3, :, :].cuda()
                # target_frame = clips[:, -6:, :, :].cuda()  # (n, 3, 256, 256)   [:, 18:24, :, :]
                target_frame = clips[:, -3:, :, :].cuda()  # [:, 21:24, :, :]

                G_frame = generator(input_frames)  # 使用Unet预测帧，预测帧为G_frame

                # 计算光流
                if use_flow:
                    if train_cfg.flownet == 'lite1':
                        bound_flow_input1 = torch.cat([G_frame, pre_frame], 1)  # t-1时刻真实的和t真实的计算光流     输入    1横着拼
                        bound_flow_input2 = torch.cat([target_frame,pre_frame],1)
                        # No need to train flow_net, use .detach() to cut off gradients.
                        flow_bound1 = flow_net.batch_estimate(bound_flow_input1, flow_net).detach()  # 光流网络计算光流    真实
                        flow_bound2 = flow_net.batch_estimate(bound_flow_input2, flow_net).detach()
                    elif train_cfg.flownet == '2sd':
                        bound_flow_input1 = torch.cat([G_frame.unsqueeze(2), pre_frame.unsqueeze(2)], 2)
                        bound_flow_input2 = torch.cat([target_frame.unsqueeze(2), pre_frame.unsqueeze(2)], 2)
                        flow_bound1 = (flow_net(bound_flow_input1 * 255.) / 255.).detach()  # Input for flownet2sd is in (0, 255).
                        flow_bound2 = (flow_net(bound_flow_input2 * 255.) / 255.).detach()

                # criterion = MS_SSIM_L1_LOSS()
                if 'l2' in train_cfg.loss:
                    inte_l = intensity_loss(G_frame, target_frame)  # torch.mean(torch.abs((gen_frames - gt_frames) ** 2))
                if 'grad' in train_cfg.loss:
                    grad_l = gradient_loss(G_frame, target_frame)  # 梯度损失
                if use_flow:
                    fl_l = flow_loss(flow_bound1, flow_bound2)  # 光流损失

                # g_l = adversarial_loss(discriminator(G_frame))  # torch.mean((fake_outputs - 1) ** 2 / 2)
                G_l_t = 1. * inte_l + 1. * grad_l
                if use_flow:
                    G_l_t = G_l_t + 2. * fl_l

                optimizer_G.zero_grad()
                G_l_t.backward()
                optimizer_G.step()

                torch.cuda.synchronize()

                if step % 20 == 1:
                    psnr = torch.mean(psnr_error(G_frame, target_frame))
                    writer.add_scalar('psnr/train_psnr', psnr, global_step=step)

                loop.set_description(f'Epoch[{epoch}/{train_cfg.epochs}]')

                if use_flow:
                    loop.set_postfix(
                        info=f'step:{step} | psnr:{psnr:.3f} G:{G_l_t:.3f} | inte:{inte_l:.3f} | grad:{grad_l:.3f} | fl_l:{fl_l:.3f} ')
                else:
                    loop.set_postfix(
                        info=f'step:{step} | psnr:{psnr:.3f} G:{G_l_t:.3f} | inte:{inte_l:.3f} | grad:{grad_l:.3f} ')

                writer.add_scalar('total_loss/g_loss_total', G_l_t, global_step=step)  # 总损失
                if use_flow:
                    writer.add_scalar('G_loss_total/fl_loss', fl_l, global_step=step)  # 光流损失
                if 'l2' in train_cfg.loss:
                    writer.add_scalar('G_loss_total/inte_loss', inte_l, global_step=step)  # 烈度损失
                elif 'grad' in train_cfg.loss:
                    writer.add_scalar('G_loss_total/grad_loss', grad_l, global_step=step)  # 烈度损失

            if epoch % train_cfg.save_epochs == 0:
                auc, fpr, tpr = val_training_pre1(model=generator, test_cfg=test_cfg)
                if auc >= current_max_auc:
                    current_max_auc = auc
                    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict()}
                    torch.save(model_dict,
                               f'{weights_path}/{train_cfg.mod}_pre1_{train_cfg.dataset}_clip{train_cfg.clip}_bs{train_cfg.batchsize}_{epoch}_{auc:.3f}.pth')
                print(f'Already saved: {train_cfg.dataset}_{epoch}.pth')
                writer.add_scalar('results/auc', auc, global_step=step)

                save_path = f'{npys_path}/{train_cfg.mod}_pre1_{train_cfg.dataset}_clip{train_cfg.clip}_bs{train_cfg.batchsize}_{epoch}_{auc:.3f}.npy'
                Dataset.save_npy_files(path=save_path, auc=auc, fpr=fpr, tpr=tpr)

    except KeyboardInterrupt:
        print(f'\nStop early, model saved: \'latest_{train_cfg.dataset}_{epoch}.pth\'.\n')

        if glob(f'weights/latest*'):
            os.remove(glob(f'weights/{train_cfg.dataset}/latest*')[0])

        model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict()}

        torch.save(model_dict,
                   f'weights/{train_cfg.dataset}/latest_{train_cfg.mod}_pre1_{train_cfg.dataset}_{epoch}.pth')


if __name__ == "__main__":
    # ext_mod = []
    ext_mod=['early_mp_cbam_shared_att_conv', 'early_mp_cbam_noshared_att_conv', 'med_mp_cbam_shared_conv_att', 'med_mp_cbam_noshared_conv_att', 'med_mp_cbam_shared_att_conv', 'med_mp_cbam_noshared_att_conv']
    train_cfg = config.train_config()
    test_cfg = config.test_config()

    for m in ext_mod:
        train_cfg.mod = m
        train(train_cfg=train_cfg, test_cfg=test_cfg)


