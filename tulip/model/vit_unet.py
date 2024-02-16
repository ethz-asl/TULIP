# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from functools import partial

import torch
from einops import rearrange
import torch.nn as nn

from timm.models.vision_transformer import Block

from util.pos_embed import get_2d_sincos_pos_embed
import numpy as np
import copy

from model.swin_unet import PixelShuffleHead, PatchEmbedding

class ViTUnet(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(128, 2048), patch_size=(16, 16), in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, use_cls_token = False, 
                 log_transform = False, circular_padding = False, patch_norm = True):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        
        self.patch_embed = PatchEmbedding(img_size = img_size, patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                                norm_layer=norm_layer if patch_norm else None, circular_padding=circular_padding)
        num_patches = self.patch_embed.num_patches
        self.in_chans = in_chans
        self.use_cls_token = use_cls_token
        self.depth = depth
        self.embed_dim = embed_dim
        self.log_transform = log_transform
        self.input_size = img_size

        total_patches = num_patches + (1 if use_cls_token else 0)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            print('NO [CLS] TOKEN')
        # self.pos_embed = nn.Parameter(torch.zeros(1, total_patches, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.ps_head = PixelShuffleHead(decoder_embed_dim, upscale_factor=4)
        # --------------------------------------------------------------------------

        self.decoder_pred = nn.Conv2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=(1, 1), bias=False)
        self.norm_pix_loss = norm_pix_loss

        self.skip_connection_layers = self.skip_connection()

        self.initialize_weights()

    def skip_connection(self):
        skip_connection_layers = nn.ModuleList()
        for _ in range(self.depth - 1):
            layer = nn.Linear(self.embed_dim * 2, self.embed_dim)
            skip_connection_layers.append(layer)
        return skip_connection_layers
    
    def initialize_weights(self):
        # initialization
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # Allow non-square inputs
    def patch_size(self):
        return self.patch_embed.proj.kernel_size

    def grid_size(self):
        return self.patch_embed.grid_size
    
    def img_patch_dim(self):
        patch_size = self.patch_size()
        return patch_size[0] * patch_size[1] * self.in_chans
    
    	

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        x = rearrange(x, 'b h w c -> b (h w) c')

        # apply Transformer blocks
        x_save = []
        for blk in self.blocks:
            x_save.append(x)
            x = blk(x)
        x = self.norm(x)

        return x, x_save

    def forward_decoder(self, x, x_save):
        # embed tokens
        x = self.decoder_embed(x)

        # apply Transformer blocks

        for i, blk in enumerate(self.decoder_blocks):
            x = torch.cat([x, x_save[len(x_save) - i - 2]], -1)
            x = self.skip_connection_layers[i](x)
            x = blk(x)
        x = self.decoder_norm(x)

        x = rearrange(x, 'b (h w) c -> b h w c', h = self.input_size[0])
        x = rearrange(x, 'b h w c -> b c h w')

        # predictor projection
        x = self.ps_head(x.contiguous())
        x = self.decoder_pred(x)
        return x

    def forward_loss(self, target, pred):
        loss = (pred - target).abs()

        loss = loss.mean()

        if self.log_transform:
            pixel_loss = (torch.expm1(pred) - torch.expm1(target)).abs().mean()
        else:
            pixel_loss = loss.clone()
        return loss, pixel_loss

    def forward(self, imgs, target, eval = False, mc_drop = False):
        x, x_save = self.forward_encoder(imgs)
        pred = self.forward_decoder(x, x_save)  # [N, L, p*p*3]
        total_loss, pixel_loss = self.forward_loss(target, pred)
        if mc_drop:
            return pred
        else:
            return pred, total_loss, pixel_loss


def vit_unet(**kwargs):
    model = ViTUnet(
        embed_dim=192, depth=4, num_heads=6,
        decoder_embed_dim=192, decoder_depth=3, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# vit_unet = vit_unet_embd768_depth4
