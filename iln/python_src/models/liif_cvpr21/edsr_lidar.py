# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLiDAR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(ConvLiDAR, self).__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)

    def forward(self, x):
        # A convolution for LiDAR range image; horizontal angle needs to be rotational.
        x = F.pad(x, (0, 0, self.pad, self.pad), mode='replicate')
        x = F.pad(x, (self.pad, self.pad, 0, 0), mode='circular')
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class EDSRLiDAR(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, res_scale=1, conv=ConvLiDAR):
        super(EDSRLiDAR, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(1, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        self.out_dim = n_feats

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        return res
