import math
from torch import nn
import torch
import torch.nn.functional as F

"""Attention operate block"""

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


"""Multi scale block"""


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


class MS_AttentionSelect_Shared_Block(nn.Module):
    def __init__(self, in_channels, att_type=''):
        super(MS_AttentionSelect_Shared_Block, self).__init__()
        self.ms = False
        self.att = False

        if 'mp' in att_type:
            self.ms = True
            self.ms_block = MyMsPlus_Block(in_channels)

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
    def __init__(self, in_channels, att_type=''):
        super(MS_AttentionSelect_NoShared_Block, self).__init__()
        self.ms = False
        self.att = False

        if 'mp' in att_type:
            self.ms = True
            self.ms_block_1 = MyMsPlus_Block(in_channels)
            self.ms_block_2 = MyMsPlus_Block(in_channels)

        if 'eca' in att_type:
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
    def __init__(self, in_ch, out_ch, att_type=''):
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
    def __init__(self, in_ch, out_ch, att_type=''):
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
    def __init__(self, in_ch, out_ch, att_type=''):
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
    def __init__(self, in_ch, out_ch, att_type=''):
        super(up_ms_att, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

        if att_type == 'mp_att':
            self.attention = MS_Attention_Block(in_ch // 2, in_ch // 2, in_ch // 4, att_type='')

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
    def __init__(self, in_ch, out_ch, att_type=''):
        super(up_med, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2) 
        self.ms = False
        self.att = False
        if 'noshared' in att_type:
            self.type = 'noshared'
            if 'mp' in att_type:
                self.ms = True
                self.ms_block_1 = MyMsPlus_Block(in_ch // 2)
                self.ms_block_2 = MyMsPlus_Block(in_ch // 2)
        elif 'shared' in att_type:
            self.type = 'shared'
            if 'mp' in att_type:
                self.ms = True
                self.ms_block = MyMsPlus_Block(in_ch // 2)

        if 'conv_att' in att_type:
            self.conv_att = 'conv_att'
            elif 'eca' in att_type:
                self.att = True
                self.attention = ECA_Block(in_ch // 2)

        elif 'att_conv' in att_type:
            self.conv_att = 'att_conv'
            if 'eca' in att_type:
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
    def __init__(self, input_channels, output_channel=6, attention_type=''):
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
    def __init__(self, input_channels, output_channel=6, attention_type=''): 
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
    def __init__(self, input_channels, output_channel=6, attention_type=''):
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


class Med_MS_Attention_Net(nn.Module):
    def __init__(self, input_channels, output_channel=6, attention_type=''):
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
    # t = Later_SelectAttention_Net(3, 6, attention_type='').cuda() 
    # t = Med_MS_Attention_Net(3, 3, attention_type='').cuda() 

    # t = Early_SelectAttention_Shared_Net(3, 3, attention_type='').cuda()  
    t = Early_SelectAttention_NoShared_Net(3, 3, attention_type='').cuda()
    out = t(x)
    print(out)
