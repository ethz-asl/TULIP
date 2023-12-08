from functools import partial

# from swin_mae import SwinMAE

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from swin_unet import PatchEmbedding, BasicBlock, PatchExpanding, BasicBlockUp, PixelShuffleHead, \
BasicBlockUpV2, BasicBlockV2, PixelShuffleExpanding, SwinTransformerBlock, PixelUnshuffleMerging, PixelUnshuffleHead
from util.pos_embed import get_2d_sincos_pos_embed
from util.datasets import grid_reshape, grid_reshape_backward
import copy

class BasicBlockUpReverse(nn.Module):
    def __init__(self, index: int, embed_dim: int = 96, window_size: int = 7, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 patch_merging: bool = True, norm_layer=nn.LayerNorm, pixel_shuffle_expanding: bool = False):
        super(BasicBlockUpReverse, self).__init__()
        index = len(depths) - index - 2
        depth = depths[index]
        dim = embed_dim * 2 ** index
        num_head = num_heads[index]

        dpr = [rate.item() for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_head,
                window_size=window_size,
                shift=False if (i % 2 == 0) else True,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        if patch_merging:
            self.upsample = PixelUnshuffleMerging(dim = embed_dim * 2 ** index)      
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        x = self.upsample(x)
        for layer in self.blocks:
            x = layer(x)
        return x

class SwinAutoEncoderReverseDecoder(nn.Module):
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

        self.upscale_factor = self.in_chans
        self.ps_head = PixelUnshuffleHead(dim = embed_dim, upscale_factor=self.upscale_factor)
        self.decoder_pred = nn.ConvTranspose2d(in_channels=self.in_chans, out_channels=self.embed_dim, kernel_size=(1, 1), bias=False)
        
        self.norm_up = norm_layer(embed_dim)

        self.layers_up = self.build_layers()

        # self.initialize_weights()
        self.apply(self.init_weights)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.grid_reshape = grid_reshape
        if self.grid_reshape:
            H_in = img_size[0] * self.in_chans
            W_in = img_size[1] 
            self.params_input = (H_in, 
                                 W_in,
                                 embed_dim,
                                 W_in // H_in,
                                 int((W_in // H_in)**0.5))

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def tile(self, img):
        # bchw
        return torch.tile(img, (1, self.in_chans, 1, 1))
        

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

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers-1):
            layer = BasicBlockUpReverse(
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
                patch_merging=False if i == self.num_layers-2 else True)
            layers.append(layer)

        return layers



    def forward_loss(self, pred_latentspace, target_latentspace):
        '''Compute the feature loss'''
        loss = (pred_latentspace - target_latentspace)**2
        loss = loss.mean()

        return loss
    
    
    # Reverse Version of the pretrain decoder
    def forward_encoder(self, x):
        x = self.decoder_pred(x)
        x = grid_reshape(x, self.params_input ,order="bchw")
        x = self.ps_head(x)

        x = rearrange(x, 'B C H W -> B H W C')


        x = self.norm_up(x)


        for i in range(len(self.layers_up)-1,-1,-1):

            x = self.layers_up[i](x)
        return x

    
    def forward(self, x, target):
        
        if self.weighted_sum:
            target = self.tile(target)
            x = self.tile(x)

        x = self.forward_encoder(x)
        target = self.forward_encoder(target)

        perceptual_loss = self.forward_loss(x, target)
        

        return perceptual_loss


class SwinAutoEncoderPerceptualLoss(nn.Module):
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
        self.norm_up = norm_layer(embed_dim)


        if swin_v2:
            self.layers = self.build_layers_v2()
        else:
            self.layers = self.build_layers()

        # self.initialize_weights()
        self.apply(self.init_weights)
        self.initialize_weights_for_output_concatenation()
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.grid_reshape = grid_reshape
        if self.grid_reshape:
            H_in = img_size[0]  // patch_size[0]
            W_in = img_size[1] // patch_size[1]
            self.params_input = (H_in, 
                                 W_in,
                                 embed_dim,
                                 W_in // H_in,
                                 int((W_in // H_in)**0.5))

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
    
    def depthwise_concate(self, x):
        downsample_factor = 4
        h_high_res = x.shape[2]
        low_res_indices = [range(i, h_high_res+i, downsample_factor) for i in range(downsample_factor)]
        x = torch.cat([x[:, :, low_res_indices[i], :] for i in range(len(low_res_indices))], dim = 1)
        return x
            
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



    def forward_loss(self, pred_latentspace, target_latentspace):
        '''Compute the feature loss'''
        loss = (pred_latentspace - target_latentspace)**2
        loss = loss.mean()

        return loss
    
    def weighted_summation(self, x):
        assert self.output_weights is not None
        x = x*self.output_weights
        x = x.sum(1, keepdim=True)
        return x
    
    def forward_encoder(self, x):
        x = self.patch_embed(x)
        
        # Have to rearrange to the shape with H * H * C, otherwise the shape won't match in transformer
        # (B, H, W, C)
        x = grid_reshape(x, self.params_input)

        x = self.pos_drop(x) 
        x_save = []
        for i, layer in enumerate(self.layers):
            x_save.append(x)
            x = layer(x)
        
        return x
    
    def forward(self, x, target):
        
        if self.weighted_sum:
            target = self.depthwise_concate(target)
            x = self.depthwise_concate(x)


        x = self.forward_encoder(x)
        target = self.forward_encoder(target)

        perceptual_loss = self.forward_loss(x, target)
        

        return perceptual_loss
    


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
            H_in = img_size[0] // patch_size[0]
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


def swin_autoencoder_line2_ws4_dec768d(**kwargs):
    model = SwinAutoEncoderPerceptualLoss(
        patch_size=(1, 4),
        window_size=4,
        decoder_embed_dim=768,
        depths=(2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model


def swin_reversedecoder_line2_ws4_dec768d(**kwargs):
    model = SwinAutoEncoderReverseDecoder(
        patch_size=(1, 4),
        window_size=4,
        decoder_embed_dim=768,
        depths=(2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model

swin_autoencoder_base_line_ws4 = swin_autoencoder_line2_ws4_dec768d
swin_mae_patch2_base_line_ws4 = swin_mae_line2_ws4_dec768d
swin_reversedecoder_base_line_ws4 = swin_reversedecoder_line2_ws4_dec768d
