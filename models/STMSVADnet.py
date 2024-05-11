import math
from torch import nn
import torch
import torch.nn.functional as F

"""Attention operate block"""


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x


class CBAM_Block(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=3):
        super(CBAM_Block, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x = self.relu(x)
        return x


class ECA_Block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_Block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class Attention_Block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_Block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


"""Multi scale block"""


class RFBBasic_Block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(RFBBasic_Block, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RFB_s_Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(RFB_s_Block, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            RFBBasic_Block(in_planes, inter_planes, kernel_size=1, stride=1),
            RFBBasic_Block(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            RFBBasic_Block(in_planes, inter_planes, kernel_size=1, stride=1),
            RFBBasic_Block(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            RFBBasic_Block(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            RFBBasic_Block(in_planes, inter_planes, kernel_size=1, stride=1),
            RFBBasic_Block(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            RFBBasic_Block(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch3 = nn.Sequential(
            RFBBasic_Block(in_planes, inter_planes // 2, kernel_size=1, stride=1),
            RFBBasic_Block(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            RFBBasic_Block((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            RFBBasic_Block(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = RFBBasic_Block(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = RFBBasic_Block(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class MyMsPlus_Block(nn.Module):
    def __init__(self, in_channels):
        super(MyMsPlus_Block, self).__init__()
        hidden_channels = in_channels // 4
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding=0, bias=False),
                                   nn.BatchNorm2d(hidden_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(hidden_channels),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=2, bias=False),
                                   nn.BatchNorm2d(hidden_channels),
                                   nn.ReLU(inplace=True))

        self.daliteconv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, dilation=3, padding=3, stride=1, groups=1),
            nn.BatchNorm2d(hidden_channels))
        self.daliteconv2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, dilation=3, padding=3, stride=1, groups=1),
            nn.BatchNorm2d(hidden_channels))
        self.daliteconv3 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, (hidden_channels // 4) * 3, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d((hidden_channels // 4) * 3),
            nn.ReLU(inplace=True),
            nn.Conv2d((hidden_channels // 4) * 3, hidden_channels, kernel_size=3, dilation=5, padding=5, stride=1,
                      groups=1),
            nn.BatchNorm2d(hidden_channels))

        self.fuse1 = nn.Sequential(nn.Conv2d(7 * (in_channels // 4), in_channels, kernel_size=1, padding=0),
                                   nn.BatchNorm2d(in_channels))
        self.fuse2 = nn.Sequential(nn.Conv2d(3 * hidden_channels + in_channels, in_channels, kernel_size=1, padding=0),
                                   nn.BatchNorm2d(in_channels))
        self.shortcut = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
                                      nn.BatchNorm2d(in_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        fuse_x = torch.cat([x, x1, x2, x3], dim=1)
        fuse_x = self.fuse1(fuse_x)

        x1 = self.daliteconv1(x)
        x2 = self.daliteconv2(x)
        x3 = self.daliteconv3(x)
        fuse_x2 = torch.cat([fuse_x, x1, x2, x3], dim=1)
        fuse_x2 = self.fuse2(fuse_x2)

        x = self.shortcut(x)
        x = fuse_x2 + x
        x = self.relu(x)
        return x


"""Multi attention select block"""


class MS_Attention_Block(nn.Module):
    def __init__(self, F_g, F_l, F_int, att_type='mp_att'):
        super(MS_Attention_Block, self).__init__()
        if 'rfb' in att_type:
            self.ms_block = RFB_s_Block(F_l, F_l)
        elif 'mp' in att_type:
            self.ms_block = MyMsPlus_Block(F_l)

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1, x1 = self.ms_block(g), self.ms_block(x)
        g1 = self.W_g(g1)
        x1 = self.W_x(x1)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class MS_AttentionSelect_Shared_Block(nn.Module):
    def __init__(self, in_channels, att_type='later_ms_cbam'):
        super(MS_AttentionSelect_Shared_Block, self).__init__()
        self.ms = False
        self.att = False
        if 'rfb' in att_type:
            self.ms = True
            self.ms_block = RFB_s_Block(in_channels, in_channels)
        elif 'mp' in att_type:
            self.ms = True
            self.ms_block = MyMsPlus_Block(in_channels)

        if 'cbam' in att_type:
            self.att = True
            self.attention = CBAM_Block(in_channels)
        elif 'eca' in att_type:
            self.att = True
            self.attention = ECA_Block(in_channels)

    def forward(self, x):
        if self.ms:
            x = self.ms_block(x)
        if self.att:
            x = self.attention(x)

        return x


class MS_AttentionSelect_NoShared_Block(nn.Module):
    def __init__(self, in_channels, att_type='early_ms_cbam_noshared'):
        super(MS_AttentionSelect_NoShared_Block, self).__init__()
        self.ms = False
        self.att = False
        if 'rfb' in att_type:
            self.ms = True
            self.ms_block_1 = RFB_s_Block(in_channels, in_channels)
            self.ms_block_2 = RFB_s_Block(in_channels, in_channels)
        elif 'mp' in att_type:
            self.ms = True
            self.ms_block_1 = MyMsPlus_Block(in_channels)
            self.ms_block_2 = MyMsPlus_Block(in_channels)

        if 'cbam' in att_type:
            self.att = True
            self.attention_1 = CBAM_Block(in_channels)
            self.attention_2 = CBAM_Block(in_channels)
        elif 'eca' in att_type:
            self.att = True
            self.attention_1 = ECA_Block(in_channels)
            self.attention_2 = ECA_Block(in_channels)

    def forward(self, x, y):
        if self.ms:
            x, y = self.ms_block_1(x), self.ms_block_2(y)
        if self.att:
            x, y = self.attention_1(x), self.attention_2(y)

        return x, y


"""base operate"""


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2),
                                    double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


"""different up layers"""


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class up_early_shared(nn.Module):
    def __init__(self, in_ch, out_ch, att_type='early_ms_cbam_shared'):
        super(up_early_shared, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

        self.attention = MS_AttentionSelect_Shared_Block(in_channels=in_ch // 2, att_type=att_type)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.attention(x1)
        x2 = self.attention(x2)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class up_early_noshared(nn.Module):
    def __init__(self, in_ch, out_ch, att_type='early_ms_cbam_noshared'):
        super(up_early_noshared, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

        self.att = MS_AttentionSelect_NoShared_Block(in_ch // 2, att_type=att_type)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1, x2 = self.att(x1, x2)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class up_later(nn.Module):
    def __init__(self, in_ch, out_ch, att_type='later_ms_cbam_conv_att'):
        super(up_later, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

        if 'conv_att' in att_type:
            self.conv_att = 'conv_att'
            self.attention = MS_AttentionSelect_Shared_Block(in_ch // 2, att_type=att_type)
        elif 'att_conv' in att_type:
            self.conv_att = 'att_conv'
            self.attention = MS_AttentionSelect_Shared_Block(in_ch, att_type=att_type)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)

        if self.conv_att == 'conv_att':
            x = self.conv(x)
            x = self.attention(x)
        elif self.conv_att == 'att_conv':
            x = self.attention(x)
            x = self.conv(x)

        return x


class up_ms_att(nn.Module):
    def __init__(self, in_ch, out_ch, att_type='mp_att'):
        super(up_ms_att, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

        if att_type == 'mp_att':
            self.attention = MS_Attention_Block(in_ch // 2, in_ch // 2, in_ch // 4, att_type='ms_att')
        elif att_type == 'rfb_att':
            self.attention = MS_Attention_Block(in_ch // 2, in_ch // 2, in_ch // 4, att_type='rfb_att')

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.attention(x1, x2)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x


class up_med(nn.Module):
    def __init__(self, in_ch, out_ch, att_type='med_ms_cbam_noshared_conv_att'):
        super(up_med, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)  # 512,256
        self.ms = False
        self.att = False
        if 'noshared' in att_type:
            self.type = 'noshared'
            if 'rfb' in att_type:
                self.ms = True
                self.ms_block_1 = RFB_s_Block(in_ch // 2, in_ch // 2)
                self.ms_block_2 = RFB_s_Block(in_ch // 2, in_ch // 2)
            elif 'mp' in att_type:
                self.ms = True
                self.ms_block_1 = MyMsPlus_Block(in_ch // 2)
                self.ms_block_2 = MyMsPlus_Block(in_ch // 2)
        elif 'shared' in att_type:
            self.type = 'shared'
            if 'rfb' in att_type:
                self.ms = True
                self.ms_block = RFB_s_Block(in_ch // 2, in_ch // 2)
            elif 'mp' in att_type:
                self.ms = True
                self.ms_block = MyMsPlus_Block(in_ch // 2)

        if 'conv_att' in att_type:
            self.conv_att = 'conv_att'
            if 'cbam' in att_type:
                self.att = True
                self.attention = CBAM_Block(in_ch // 2)
            elif 'eca' in att_type:
                self.att = True
                self.attention = ECA_Block(in_ch // 2)

        elif 'att_conv' in att_type:
            self.conv_att = 'att_conv'
            if 'cbam' in att_type:
                self.att = True
                self.attention = CBAM_Block(in_ch)
            elif 'eca' in att_type:
                self.att = True
                self.attention = ECA_Block(in_ch)

        self.conv = double_conv(in_ch, out_ch)  # 512,256

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.type == 'noshared':
            if self.ms:
                x1, x2 = self.ms_block_1(x1), self.ms_block_2(x2)
        elif self.type == 'shared':
            if self.ms:
                x1, x2 = self.ms_block(x1), self.ms_block(x2)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)

        if self.conv_att == 'conv_att':
            x = self.conv(x)
            if self.att:
                x = self.attention(x)
        elif self.conv_att == 'att_conv':
            if self.att:
                x = self.attention(x)
            x = self.conv(x)

        return x


"""model select"""


class Early_SelectAttention_Shared_Net(nn.Module):
    def __init__(self, input_channels, output_channel=6, attention_type='early_mp_cbam_shared'):  # output_channel=6
        super(Early_SelectAttention_Shared_Net, self).__init__()
        self.inc = inconv(input_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.up1 = up_early_shared(512, 256, attention_type)
        self.up2 = up_early_shared(256, 128, attention_type)
        self.up3 = up_early_shared(128, 64, attention_type)
        self.outc = nn.Conv2d(64, output_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return torch.tanh(x)


class Early_SelectAttention_NoShared_Net(nn.Module):
    def __init__(self, input_channels, output_channel=6, attention_type='early_mp_cbam_noshared'):  # output_channel=6
        super(Early_SelectAttention_NoShared_Net, self).__init__()
        self.inc = inconv(input_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.up1 = up_early_noshared(512, 256, attention_type)
        self.up2 = up_early_noshared(256, 128, attention_type)
        self.up3 = up_early_noshared(128, 64, attention_type)
        self.outc = nn.Conv2d(64, output_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return torch.tanh(x)


class Later_SelectAttention_Net(nn.Module):
    def __init__(self, input_channels, output_channel=6, attention_type='later_mp_cbam'):  # output_channel=6
        super(Later_SelectAttention_Net, self).__init__()
        self.inc = inconv(input_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.up1 = up_later(512, 256, attention_type)
        self.up2 = up_later(256, 128, attention_type)
        self.up3 = up_later(128, 64, attention_type)
        self.outc = nn.Conv2d(64, output_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return torch.tanh(x)


class MS_Attention_Net(nn.Module):
    def __init__(self, input_channels, output_channel=6, attention_type='mp_att'):
        super(MS_Attention_Net, self).__init__()
        self.inc = inconv(input_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.up1 = up_ms_att(512, 256, attention_type)
        self.up2 = up_ms_att(256, 128, attention_type)
        self.up3 = up_ms_att(128, 64, attention_type)
        self.outc = nn.Conv2d(64, output_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return torch.tanh(x)


class Med_MS_Attention_Net(nn.Module):
    def __init__(self, input_channels, output_channel=6, attention_type='med_mp_cbam_noshared'):
        super(Med_MS_Attention_Net, self).__init__()
        self.inc = inconv(input_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.up1 = up_med(512, 256, attention_type)
        self.up2 = up_med(256, 128, attention_type)
        self.up3 = up_med(128, 64, attention_type)
        self.outc = nn.Conv2d(64, output_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return torch.tanh(x)


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224).cuda()
    # t = Later_SelectAttention_Net(3, 6, attention_type='later_conv_att').cuda()  # later_ms_cbam
    # t = Med_MS_Attention_Net(3, 3, attention_type='med_mp_eca_noshared_att_conv').cuda()  # med_ms_cbam
    #
    # t = Early_SelectAttention_Shared_Net(3, 3, attention_type='early_shared').cuda()  # early_ms_cbam_shared
    t = Early_SelectAttention_NoShared_Net(3, 3, attention_type='early_noshared').cuda()
    # t = MS_Attention_Net(8, 3, attention_type='ms_att').cuda()
    out = t(x)
    print(out)
