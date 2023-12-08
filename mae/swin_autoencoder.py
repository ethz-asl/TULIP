from functools import partial

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from swin_unet import PatchEmbedding, BasicBlock, PatchExpanding, BasicBlockUp, PixelShuffleHead, BasicBlockUpV2, BasicBlockV2, PixelShuffleExpanding
from util.pos_embed import get_2d_sincos_pos_embed
from util.datasets import grid_reshape, grid_reshape_backward
import copy


class SwinAutoencoder(nn.Module):
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
        # self.mask_ratio = mask_ratio
        assert (img_size[0] % patch_size[0] == 0) and (img_size[1] % patch_size[1] == 0)
        # self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
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
        self.bottleneck_channel_reduction = bottleneck_channel_reduction
        self.img_size = img_size
        self.weighted_sum = True if self.in_chans > 1 else False
        
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None, circular_padding=circular_padding)
        self.num_patches = self.patch_embed.num_patches

        if self.pixel_shuffle_expanding:
            self.first_patch_expanding = PixelShuffleExpanding(dim = decoder_embed_dim, norm_layer=norm_layer)
        else:
            self.first_patch_expanding = PatchExpanding(dim=decoder_embed_dim, norm_layer=norm_layer)
        self.norm_up = norm_layer(embed_dim)

        final_upscale_factor = int((patch_size[0] * patch_size[1])**0.5)
        final_upscale_factor = final_upscale_factor*2 if self.weighted_sum else final_upscale_factor

        
        # Pixel Shuffle is conv based projection, maybe we should also match the projection in pretraining stage, I would use a pixel shuffle head for the projection as well
        if conv_projection:
            self.ps_head= PixelShuffleHead(dim = embed_dim, upscale_factor= final_upscale_factor)
            self.decoder_pred = nn.Conv2d(in_channels = embed_dim, out_channels = in_chans, kernel_size = (1, 1), bias=False)
        else:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 8, patch_size[0] * patch_size[1] * in_chans, bias=True)


        if swin_v2:
            self.layers = self.build_layers_v2()
            self.layers_up = self.build_layers_up_v2()
        else:
            self.layers = self.build_layers()
            self.layers_up = self.build_layers_up()

        # self.initialize_weights()
        self.apply(self.init_weights)
        self.initialize_weights_for_output_concatenation()
        self.pos_drop = nn.Dropout(p=drop_rate)
        # self.skip_connection_layers = self.skip_connection()

        self.grid_reshape = grid_reshape
        if self.grid_reshape:
            H_in = img_size[0]  // patch_size[0]
            W_in = img_size[1] // patch_size[1]

            if self.weighted_sum:
                H_out = img_size[0] * in_chans
                W_out = img_size[1] 
            else:
                H_out = img_size[0] 
                W_out = img_size[1] 
            self.params_input = (H_in, 
                                 W_in,
                                 embed_dim,
                                 W_in // H_in,
                                 int((W_in // H_in)**0.5))

            self.params_output = (H_out,
                                  W_out,
                                  embed_dim,
                                  W_out // H_out,
                                  int((W_out // H_out)**0.5))

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def depthwise_unconcate(self, img):
        b, c, h, w = img.shape
        new_img = torch.zeros(b, h * c, w).to(img.device)
        low_res_indices = [range(i, h*c+i, c) for i in range(c)]

        for i, indices in enumerate(low_res_indices):
            new_img[:, indices,:] = img[:, i, :, :]


        return new_img.reshape(b, 1, h*c, w)
            
    def initialize_weights_for_output_concatenation(self):
        if self.weighted_sum:
            self.output_weights = nn.Parameter(torch.ones((1, self.in_chans, 1, 1)) * (1/self.in_chans), requires_grad=True)
        else:
            self.output_weights = None

    def patch_size(self):
        return self.patch_embed.patch_size

    def grid_size(self):
        return self.patch_embed.grid_size
    
    def img_patch_dim(self):
        patch_size = self.patch_size()
        return patch_size[0] * patch_size[1] * self.in_chans

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
           
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
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
    
    def build_layers_up_v2(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUpV2(
                index=i,
                input_resolution=(int(self.patch_embed.num_patches**0.5)// (2**(self.num_layers-2-i)),
                                  int(self.patch_embed.num_patches**0.5)// (2**(self.num_layers-2-i))), 
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
                pixel_shuffle_expanding=self.pixel_shuffle_expanding)
            layers_up.append(layer)
        return layers_up

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

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        if self.bottleneck_channel_reduction:
            depths = self.depths[:4]
            num_heads = self.num_heads[:4]
            num_layers_up = 4
        else:
            depths = self.depths
            num_heads = self.num_heads
            num_layers_up = self.num_layers

        for i in range(num_layers_up - 1): 
            layer = BasicBlockUp(
                index=i,
                depths=depths,
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < num_layers_up - 2 else False,
                norm_layer=self.norm_layer,
                pixel_shuffle_expanding=self.pixel_shuffle_expanding)
            layers_up.append(layer)
        return layers_up



    def skip_connection(self):
        skip_connection_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            dim = self.embed_dim * 2 ** (self.num_layers - 2 - i)
            layer = nn.Linear(dim * 2, dim)
            skip_connection_layers.append(layer)
        return skip_connection_layers



    def forward_loss(self, pred, target):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        

        loss = (pred - target).abs()

        ## inverse huber loss
        # loss = inverse_huber_loss(pred, target)

        # Smooth L1 Loss
        # loss = func.smooth_l1_loss(pred, target, reduction='none')
        loss = loss.mean()

        if self.log_transform:
            pixel_loss = (torch.expm1(pred) - torch.expm1(target)).abs().mean()
        else:
            pixel_loss = loss.clone()


        return loss, pixel_loss
    
    def weighted_summation(self, x):
        assert self.output_weights is not None
        x = x*self.output_weights
        x = x.sum(1, keepdim=True)
        return x
            
    
    def forward(self, x, mask_ratio, mask_loss = False, eval = False, remove = False, loss_on_unmasked = False):
        
        
        target = x.clone()
        if self.weighted_sum:
            target = self.depthwise_unconcate(target)

        x = self.patch_embed(x) 
        # Have to rearrange to the shape with H * H * C, otherwise the shape won't match in transformer
        # (B, H, W, C)
        x = grid_reshape(x, self.params_input)
        x = self.pos_drop(x) 
        # x_save = []
        for i, layer in enumerate(self.layers):
            # x_save.append(x)
            x = layer(x)
            
        x = self.first_patch_expanding(x)


        for i, layer in enumerate(self.layers_up):
            # x = torch.cat([x, x_save[len(x_save) - i - 2]], -1)
            # x = self.skip_connection_layers[i](x)
            x = layer(x)

        
        x = self.norm_up(x)
        
        x = rearrange(x, 'B H W C -> B C H W')
        x = self.ps_head(x.contiguous())
        x = grid_reshape_backward(x, self.params_output, order="bchw")

        x = self.decoder_pred(x.contiguous())


        if self.weighted_sum:
            x = self.weighted_summation(x)

        total_loss, pixel_loss = self.forward_loss(x,target)
        return total_loss, x, pixel_loss



def swin_mae_line2_ws4_dec768d(**kwargs):
    model = SwinAutoencoder(
        patch_size=(1, 4),
        window_size=4,
        decoder_embed_dim=768,
        depths=(2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model

def swin_mae_small_line2_ws4_dec768d(**kwargs):
    model = SwinAutoencoder(
        patch_size=(1, 4),
        window_size=4,
        decoder_embed_dim=384,
        depths=(2, 2, 2), embed_dim=96, num_heads=(3, 6, 12),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model


def swin_mae_tiny_line2_ws4_dec768d(**kwargs):
    model = SwinAutoencoder(
        patch_size=(1, 4),
        window_size=4,
        decoder_embed_dim=192,
        depths=(2, 2), embed_dim=96, num_heads=(3, 6),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model


def swin_mae_deep_line2_ws4_dec768d(**kwargs):
    model = SwinAutoencoder(
        patch_size=(1, 4),
        window_size=4,
        decoder_embed_dim=1536,
        depths=(2, 2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24, 48),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model

def swin_mae_deeper_line2_ws4_dec768d(**kwargs):
    model = SwinAutoencoder(
        patch_size=(1, 4),
        window_size=4,
        decoder_embed_dim=3072,
        depths=(2, 2, 2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24, 48, 96),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model








swin_mae_patch2_base_line_ws4 = swin_mae_line2_ws4_dec768d
swin_mae_patch2_small_line_ws4 = swin_mae_small_line2_ws4_dec768d
swin_mae_patch2_tiny_line_ws4 = swin_mae_tiny_line2_ws4_dec768d
swin_mae_patch2_deep_line_ws4 = swin_mae_deep_line2_ws4_dec768d
swin_mae_patch2_deeper_line_ws4 = swin_mae_deeper_line2_ws4_dec768d

