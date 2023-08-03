import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import register_model


@register_model('interpolation')
class Interpolation(nn.Module):
    def __init__(self, interpolation_mode='bilinear'):
        super(Interpolation, self).__init__()
        self.interpolation_mode = interpolation_mode

    def gen_feat(self, inp):
        self.feat = inp
        return self.feat

    def query_detection(self, coord):
        return F.grid_sample(self.feat, coord.flip(-1).unsqueeze(1), mode=self.interpolation_mode, align_corners=False, padding_mode='border')[:, :, 0, :].permute(0, 2, 1)

    def forward(self, inp, coord):
        self.gen_feat(inp)
        return self.query_detection(coord)

