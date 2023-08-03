# PyTorch re-implementation using the sources from https://github.com/RobustFieldAutonomyLab/lidar_super_resolution

import torch
import torch.nn as nn

from models.model_utils import register_model


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(m.bias)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_block.apply(weight_init)

    def forward(self, x):
        return self.conv_block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.down = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(dropout_rate),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), output_padding=(0, 0)):
        super().__init__()
        self.up_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.up_block.apply(weight_init)

    def forward(self, x):
        return self.up_block(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.up = nn.Sequential(
            ConvBlock(in_channels * 2, in_channels),
            nn.Dropout2d(dropout_rate),
            UpBlock(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.up(x)


class InConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_h=4, scale_w=1):
        super(InConv, self).__init__()

        modules = []
        while scale_h > 1 or scale_w > 1:
            stride_h, output_padding_h, scale_h = (2, 1, (scale_h // 2)) if scale_h > 1 else (1, 0, scale_h)
            stride_w, output_padding_w, scale_w = (2, 1, (scale_w // 2)) if scale_w > 1 else (1, 0, scale_w)

            in_ch, out_ch = (out_channels, out_channels) if modules else (in_channels, out_channels)
            modules.append(UpBlock(in_ch, out_ch,
                                   kernel_size=(3, 3), stride=(stride_h, stride_w),
                                   padding=(1, 1), output_padding=(output_padding_h, output_padding_w)))

        in_ch, out_ch = (out_channels, out_channels) if modules else (in_channels, out_channels)
        modules.append(ConvBlock(in_ch, out_ch))

        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels * 2, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


@register_model('lsr')
class UNet(nn.Module):
    def __init__(self, scale_h=4, scale_w=1, dropout_rate=0.25):
        super(UNet, self).__init__()
        self.up_scale = InConv(1, 64, scale_h=scale_h, scale_w=scale_w)
        self.down1 = Down(64, 128, dropout_rate)
        self.down2 = Down(128, 256, dropout_rate)
        self.down3 = Down(256, 512, dropout_rate)
        self.downup = nn.Sequential(
            Down(512, 1024, dropout_rate),
            nn.Dropout2d(dropout_rate),
            UpBlock(1024, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        )
        self.up1 = Up(512, 256, dropout_rate)
        self.up2 = Up(256, 128, dropout_rate)
        self.up3 = Up(128, 64, dropout_rate)
        self.outc = OutConv(64, 1)

    def forward(self, x):
        x1 = self.up_scale(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        y = self.downup(x4)
        y = self.up1(x4, y)
        y = self.up2(x3, y)
        y = self.up3(x2, y)
        y = self.outc(x1, y)
        return y
