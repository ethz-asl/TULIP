from functools import partial

# from swin_mae import SwinMAE

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from swin_unet import PatchEmbedding, BasicBlock, PatchExpanding, BasicBlockUp, PixelShuffleHead, BasicBlockUpV2, BasicBlockV2, PixelShuffleExpanding
from util.pos_embed import get_2d_sincos_pos_embed
from util.datasets import grid_reshape, grid_reshape_backward
import copy


class SwinMAEPerceptualLoss(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size = (224, 224), patch_size = (4, 4), in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True, circular_padding: bool = False, grid_reshape: bool = False, 
                 conv_projection: bool = False, swin_v2: bool = False, pixel_shuffle_expanding: bool = False, 
                 log_transform: bool = False, bottleneck_channel_reduction: bool = False):

        super().__init__()

        self.num_layers = len(depths)
        self.norm_pix_loss = norm_pix_loss
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer
        self.in_chans = in_chans
        self.conv_projection = conv_projection
        self.pixel_shuffle_expanding = pixel_shuffle_expanding
        self.log_transform = log_transform
        self.grid_reshape = grid_reshape
        self.bottleneck_channel_reduction = bottleneck_channel_reduction
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None, circular_padding=circular_padding)
        
        
        if self.bottleneck_channel_reduction:
            ## To Make sure the channel of latent space of high resolution image in pretraining stage match the low resolution one in the upsampling stage 
            self.bottleneck_reduction_layer = nn.Linear(decoder_embed_dim * 2, decoder_embed_dim, bias=True)

        if swin_v2:
            self.layers = self.build_layers_v2()
        else:
            self.layers = self.build_layers()

        if self.grid_reshape:
            H_in = (img_size[0] // in_chans) // patch_size[0]
            W_in = img_size[1] // patch_size[1]
            self.params_input = (H_in, 
                                 W_in,
                                 embed_dim,
                                 W_in // H_in,
                                 int((W_in // H_in)**0.5))

    def depth_wise_concate(self, x):
        downsample_factor = 4
        h_high_res = x.shape[2]
        low_res_indices = [range(i, h_high_res+i, downsample_factor) for i in range(downsample_factor)]
        x = torch.cat([x[:, :, low_res_indices[i], :] for i in range(len(low_res_indices))], dim = 1)

        return x

    def build_layers_v2(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlockV2(
                index=i,
                input_resolution=(int(self.patch_embed.num_patches**0.5) // (2**i),
                                  int(self.patch_embed.num_patches**0.5) // (2**i)),
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers


    def forward_encoder(self, x):
        x = self.patch_embed(x)

        x = grid_reshape(x, self.params_input, order = "bhwc")
        for layer in self.layers:
            x = layer(x)

        if self.bottleneck_channel_reduction:
            x = self.bottleneck_reduction_layer(x)

        return x
    
    def forward(self, x_out):
        # x_in = self.forward_encoder(x_in)
        x_out = self.forward_encoder(x_out)

        # perceptual_loss = (x_in - x_out).abs()

        return x_out


# def swin_mae(**kwargs):
#     model = SwinMAE(
#         img_size=224, patch_size=4, in_chans=3,
#         decoder_embed_dim=768,
#         depths=(2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
#         window_size=7, qkv_bias=True, mlp_ratio=4,
#         drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

def swin_mae_pacth4_ws4_dec768d(**kwargs):
    model = SwinMAEPerceptualLoss(
        patch_size=(4, 4),
        window_size=4,
        decoder_embed_dim=768,
        depths=(2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model


def swin_mae_pacth4_ws4_dec384d(**kwargs):
    model = SwinMAEPerceptualLoss(
        patch_size=(4, 4),
        window_size=4,
        decoder_embed_dim=384,
        depths=(2, 2, 2, 2), embed_dim=48, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model


def swin_mae_pacth4_ws4_dec192d(**kwargs):
    model = SwinMAEPerceptualLoss(
        patch_size=(4, 4),
        window_size=4,
        decoder_embed_dim=192,
        depths=(1, 1, 1, 1), embed_dim=24, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model

def swin_mae_pacth2_ws4_dec768d_depth4422(**kwargs):
    model = SwinMAEPerceptualLoss(
        patch_size=(2, 2),
        window_size=4,
        decoder_embed_dim=768,
        depths=(2, 2, 4, 4), embed_dim=96, num_heads=(6, 12, 16, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model

def swin_mae_pacth2_ws4_dec768d(**kwargs):
    model = SwinMAEPerceptualLoss(
        patch_size=(2, 2),
        window_size=4,
        decoder_embed_dim=768,
        depths=(2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model

def swin_mae_line2_ws4_dec768d(**kwargs):
    model = SwinMAEPerceptualLoss(
        patch_size=(1, 4),
        window_size=4,
        decoder_embed_dim=768,
        depths=(2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model

def swin_mae_moredepths_line2_ws4_dec768d(**kwargs):
    model = SwinMAEPerceptualLoss(
        patch_size=(1, 4),
        window_size=4,
        decoder_embed_dim=768,
        depths=(2, 2, 6, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model

def swin_mae_line2_ws8_dec768d(**kwargs):
    model = SwinMAEPerceptualLoss(
        patch_size=(1, 4),
        window_size=8,
        decoder_embed_dim=768,
        depths=(2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model

def swin_mae_pacth2_ws4_dec384d(**kwargs):
    model = SwinMAEPerceptualLoss(
        patch_size=(2, 2),
        window_size=4,
        decoder_embed_dim=384,
        depths=(2, 2, 2, 2), embed_dim=48, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model


def swin_mae_pacth2_ws4_dec192d(**kwargs):
    model = SwinMAEPerceptualLoss(
        patch_size=(4, 4),
        window_size=2,
        decoder_embed_dim=192,
        depths=(1, 1, 1, 1), embed_dim=24, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model

def swin_mae_line2_v2_ws4_dec768d(**kwargs):
    model = SwinMAEPerceptualLoss(
        patch_size=(1, 4),
        window_size=4,
        decoder_embed_dim=768,
        depths=(2, 2, 2, 2), embed_dim=96 ,num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), swin_v2=True, **kwargs)
        #  **kwargs)
    return model

def swin_mae_deepencoder_line2_ws4_dec768d(**kwargs):
    model = SwinMAEPerceptualLoss(
        patch_size=(1, 4),
        window_size=4,
        decoder_embed_dim=768,
        depths=(2, 2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24, 48),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        bottleneck_channel_reduction=True, **kwargs)
        #  **kwargs)
    return model


swin_mae_patch2_large = swin_mae_pacth2_ws4_dec768d_depth4422
swin_mae_patch2_base = swin_mae_pacth2_ws4_dec768d

swin_mae_patch2_small = swin_mae_pacth2_ws4_dec384d
swin_mae_patch2_tiny = swin_mae_pacth2_ws4_dec192d
swin_mae_patch4_base = swin_mae_pacth4_ws4_dec768d
swin_mae_patch4_small = swin_mae_pacth4_ws4_dec384d
swin_mae_patch4_tiny = swin_mae_pacth4_ws4_dec192d

swin_mae_v2_patch2_base_line = swin_mae_line2_v2_ws4_dec768d
swin_mae_patch2_base_line_ws4 = swin_mae_line2_ws4_dec768d
swin_mae_patch2_base_line_ws8 = swin_mae_line2_ws8_dec768d

swin_mae_moredepths_patch2_base_line_ws4 = swin_mae_moredepths_line2_ws4_dec768d

swin_mae_deepencoder_patch2_base_line_ws4 = swin_mae_deepencoder_line2_ws4_dec768d