
from swin_unet import SwinUnet
import torch.nn as nn
import torch
from einops import rearrange
from util.datasets import grid_reshape, grid_reshape_backward
import numpy as np
from functools import partial
from swin_mae_ploss import SwinMAEPerceptualLoss
from swin_mae import SwinMAE


class SwinMaeAnalysis(SwinMAEPerceptualLoss):
# class SwinMaeAnalysis(SwinMAE):
    def __init__(self, img_size = (224, 224), patch_size = (4, 4), in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True, circular_padding: bool = False, grid_reshape: bool = False, 
                 conv_projection: bool = False, swin_v2: bool = False, pixel_shuffle_expanding: bool = False, 
                 log_transform: bool = False, bottleneck_channel_reduction: bool = False):

        super().__init__(img_size = img_size, patch_size = patch_size, in_chans = in_chans,
                 decoder_embed_dim=decoder_embed_dim, norm_pix_loss=norm_pix_loss,
                 depths = depths, embed_dim = embed_dim, num_heads = num_heads,
                 window_size = window_size, qkv_bias = qkv_bias, mlp_ratio = mlp_ratio,
                 drop_path_rate = drop_path_rate, drop_rate = drop_rate, attn_drop_rate = attn_drop_rate,
                 norm_layer = norm_layer, patch_norm = patch_norm, circular_padding = circular_padding, grid_reshape = grid_reshape, 
                 conv_projection = conv_projection, swin_v2 = swin_v2, pixel_shuffle_expanding = pixel_shuffle_expanding, 
                 log_transform = log_transform, bottleneck_channel_reduction = bottleneck_channel_reduction)

    def forward_encoder(self, x):
        

        encoder_feature_map = []
        # x = self.depth_wise_concate(x)

        x = self.patch_embed(x)


        x = grid_reshape(x, self.params_input, order = "bhwc")

        encoder_feature_map.append(x)
        for layer in self.layers:
            x = layer(x)
            encoder_feature_map.append(x)

        if self.bottleneck_channel_reduction:
            x = self.bottleneck_reduction_layer(x)

        return x, encoder_feature_map
    
    def forward(self, x_out):
        # x_in = self.forward_encoder(x_in)
        x_out, encoder_feature_map = self.forward_encoder(x_out)

        # perceptual_loss = (x_in - x_out).abs()

        return x_out, encoder_feature_map



class SwinUnetAnalysis(SwinUnet):
    def __init__(self, img_size = (32, 2048), target_img_size = (128, 2048) ,patch_size = (4, 4), in_chans: int = 1, num_output_channel: int = 1, embed_dim: int = 96,
                 window_size: int = 4, depths: tuple = (2, 2, 6, 2), num_heads: tuple = (3, 6, 12, 24),
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1, norm_layer=nn.LayerNorm, patch_norm: bool = True, edge_loss: bool = False, pixel_shuffle: bool = False,
                 grid_reshape: bool = False, circular_padding: bool = False, swin_v2: bool = False, log_transform: bool = False, depth_scale_loss: bool = False,
                 pixel_shuffle_expanding: bool = False, relative_dist_loss: bool = False, perceptual_loss: bool = False, pretrain_mae: nn.Module = None
                 ):
        
        super().__init__(
            img_size=img_size,
            target_img_size=target_img_size,
            patch_size=patch_size,
            in_chans = in_chans, num_output_channel = num_output_channel, embed_dim = embed_dim,
            window_size = window_size, depths = depths, num_heads = num_heads,
            mlp_ratio = mlp_ratio, qkv_bias = qkv_bias, drop_rate = drop_rate, attn_drop_rate = attn_drop_rate,
            drop_path_rate = drop_path_rate, norm_layer=nn.LayerNorm, patch_norm = patch_norm, edge_loss = edge_loss, pixel_shuffle = pixel_shuffle,
            grid_reshape = grid_reshape, circular_padding = circular_padding, swin_v2 = swin_v2, log_transform = log_transform, depth_scale_loss = depth_scale_loss,
            pixel_shuffle_expanding = pixel_shuffle_expanding, relative_dist_loss = relative_dist_loss, perceptual_loss = perceptual_loss, pretrain_mae = pretrain_mae

        )
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def test_different_loss(self, pred, target):
        l1_loss = (pred-target).abs().mean()
        row_relative_loss_20 = self.row_relative_loss(pred, target, num_pixels=20, num_neighbors=8)
        row_relative_loss_500 = self.row_relative_loss(pred, target, num_pixels=500, num_neighbors=8)


        square_relative_loss_20_dilation0 = self.square_relative_loss(pred, target, num_pixels=20, dilation=0)
        square_relative_loss_500_dilation0 = self.square_relative_loss(pred, target, num_pixels=500, dilation=0)



        square_relative_loss_20_dilation1 = self.square_relative_loss(pred, target, num_pixels=20, dilation=1)
        square_relative_loss_500_dilation1 = self.square_relative_loss(pred, target, num_pixels=500, dilation=1)



        return torch.Tensor([l1_loss,
                row_relative_loss_20,
                row_relative_loss_500, 
                square_relative_loss_20_dilation0,
                square_relative_loss_500_dilation0,
                square_relative_loss_20_dilation1,
                square_relative_loss_500_dilation1])




    
    def row_relative_loss(self, pred, target, num_pixels = 20, num_neighbors = 8):
        
        B, C, H, W = pred.shape
        assert B == 1, "Only support batch size of 1"
        loss = 0
        for i in range(B):
        
            sampler_h = torch.randint(0, H, size = (num_pixels,),device=pred.device)
            sampler_w = torch.randint(0, W-num_neighbors, size=(num_pixels,), device=pred.device)

            patches_selected_pred = pred[i, :, sampler_h, sampler_w]
            patches_selected_target = target[i, :, sampler_h, sampler_w]
            relative_dist_pred = torch.zeros_like(patches_selected_pred, device=pred.device)
            relative_dist_target = torch.zeros_like(patches_selected_target, device=pred.device)


            for j in range(num_neighbors):
                relative_dist_pred += (patches_selected_pred - pred[i, :, sampler_h, sampler_w+j]).abs()
                relative_dist_target += (patches_selected_target - pred[i, :, sampler_h, sampler_w+j]).abs()

            loss += (relative_dist_pred - relative_dist_target).abs().mean()
        return loss / B


    def square_relative_loss(self, pred, target, num_pixels = 20, dilation = 0):
        B, C, H, W = pred.shape
        assert B == 1, "Only support batch size of 1"
        loss = 0

        neighborhood = [(1+dilation, 0), (-1-dilation, 0), (0, 1+dilation), (0, -1-dilation),
                        (1+dilation, 1+dilation), (-1-dilation, -1-dilation), (1+dilation, -1-dilation), (-1-dilation, 1+dilation)]

        for i in range(B):
        
            sampler_h = torch.randint(0 + (dilation + 1), H - (dilation + 1), size = (num_pixels,), device=pred.device)
            sampler_w = torch.randint(0 + (dilation + 1), W - (dilation + 1), size = (num_pixels,), device=pred.device)

            patches_selected_pred = pred[i, :, sampler_h, sampler_w]
            patches_selected_target = target[i, :, sampler_h, sampler_w]
            relative_dist_pred = torch.zeros_like(patches_selected_pred, device=pred.device)
            relative_dist_target = torch.zeros_like(patches_selected_target, device=pred.device)


            for nn in neighborhood:
                relative_dist_pred += (patches_selected_pred - pred[i, :, sampler_h + nn[0], sampler_w + nn[1]]).abs()
                relative_dist_target += (patches_selected_target - pred[i, :, sampler_h + nn[0], sampler_w + nn[1]]).abs()

            loss += (relative_dist_pred - relative_dist_target).abs().mean()
        return loss / B


    def forward(self, x, target, img_size_high_res, eval = False, mc_drop = False):
        feature_map_downsample = []
        feature_map_upsample = []

        x = self.patch_embed(x) 
        if self.window_size[0] == self.window_size[1]:
            x = x.contiguous().view((x.shape[0], int((x.shape[1] * x.shape[2])**0.5), int((x.shape[1] * x.shape[2])**0.5), x.shape[3]))
        # Have to rearrange to the shape with H * H * C, otherwise the shape won't match in transformer
        # (B, H, W, C)
        # x = grid_reshape(x, self.params_input)
        x = self.pos_drop(x) 
        x_save = []

        feature_map_downsample.append(x)
        for i, layer in enumerate(self.layers):
            x_save.append(x)
            x = layer(x)

        # print(layer.)

        feature_map_backbone = x.clone()
        # if self.perceptual_loss:
        #     target_feature_backbone, pretrain_downsample_features = self.pretrain_mae(target)
            
        x = self.first_patch_expanding(x)


        for i, layer in enumerate(self.layers_up):                
            x = torch.cat([x, x_save[len(x_save) - i - 2]], -1)
            x = self.skip_connection_layers[i](x)
            
            x = layer(x)

        x = self.norm_up(x)

        # x = rearrange(x, 'B H W C -> B C H W')
        if self.pixel_shuffle:
            x = rearrange(x, 'B H W C -> B C H W')
            x = self.ps_head(x.contiguous())
            # Grid Reshape
            # (B, C, H, W)
            feature_map_upsample.append(rearrange(x, 'B C H W -> B H W C'))
            if self.window_size[0] == self.window_size[1]:
                x = x.view((x.shape[0], x.shape[1], img_size_high_res[0], img_size_high_res[1]))
            # Reshape
            x = self.decoder_pred(x.contiguous())
        else:
            x = self.final_patch_expanding(x)
            # x = grid_reshape_backward(x, img_size_high_res)
            x = rearrange(x, 'B H W C -> B C H W')
            
            # Please consider reshape the image here again, as in transformer we always have the input with shape of B, H, H, C
            # for example, if 32*2048 range map as input
            # 32*2048 -> 16*1024 -> 128 * 128 -> 512*512 -> 128*2048
            # Grid Reshape  
            feature_map_upsample.append(rearrange(x, 'B C H W -> B H W C'))
            if self.window_size[0] == self.window_size[1]:
                x = x.view((x.shape[0], x.shape[1], img_size_high_res[0], img_size_high_res[1]))

            
            x = self.decoder_pred(x.contiguous())

        # feature_map_upsample.append(rearrange(x, 'B C H W -> B H W C'))
    

        # x = grid_reshape_backward(x, self.params_output, order="bchw")


        
        
        # x = self.head(x.contiguous())

        # feature_map_upsample.append(rearrange(x, 'B C H W -> B H W C'))


        # losses = self.test_different_loss(x, target)
        # cosine_similarity_map = self.cosine_similarity(feature_map_backbone, target_feature_backbone)

        # if self.perceptual_loss:
        #     return {'pred': x,
        #             'features_downsample': feature_map_downsample, 
        #             'features_upsample': feature_map_upsample,
        #             'cosine_similarity_map': cosine_similarity_map.unsqueeze(-1),
        #             'feature_backbone': feature_map_backbone,
        #             'target_feature_backbone': target_feature_backbone,
        #             'pretrain_downsample_features': pretrain_downsample_features}

        return {'pred': x,
                    'features_downsample': feature_map_downsample, 
                    'features_upsample': feature_map_upsample,
                    'feature_backbone': feature_map_backbone,}

        
        
    
    # def forward(self, x, target, img_size_high_res, eval = False, mc_drop = False):
    #     feature_map_downsample = []
    #     feature_map_upsample_skipcon = []
    #     feature_map_upsample_afterlayerup = []
    #     feature_map_upsample_beforelayerup = []

    #     x = self.patch_embed(x) 
    #     # Have to rearrange to the shape with H * H * C, otherwise the shape won't match in transformer
    #     # (B, H, W, C)
    #     x = grid_reshape(x, self.params_input)
    #     x = self.pos_drop(x) 
    #     x_save = []

    #     feature_map_downsample.append(x)
    #     for i, layer in enumerate(self.layers):
    #         x_save.append(x)
    #         x = layer(x)
    #         feature_map_downsample.append(x)

            
    #     x = self.first_patch_expanding(x)

        

    #     for i, layer in enumerate(self.layers_up):                
    #         x = torch.cat([x, x_save[len(x_save) - i - 2]], -1)
    #         feature_map_upsample_skipcon.append(x)
    #         x = self.skip_connection_layers[i](x)
    #         feature_map_upsample_skipcon.append(x)
    #         x = layer(x)
    #         if i == len(self.layers_up) - 2:
    #             feature_map_upsample_beforelayerup.append(x)
    #         else:
    #             feature_map_upsample_afterlayerup.append(x)

    #     x = self.norm_up(x)

    #     x = rearrange(x, 'B H W C -> B C H W')
    #     x = self.pixel_shuffle_layer(x.contiguous())

    #     feature_map_upsample_afterlayerup.append(rearrange(x, 'B C H W -> B H W C'))
    

    #     x = grid_reshape_backward(x, self.params_output, order="bchw")


        
        
    #     x = self.head(x.contiguous())

    #     feature_map_upsample_afterlayerup.append(rearrange(x, 'B C H W -> B H W C'))


    #     losses = self.test_different_loss(x, target)
    #     return x, losses, feature_map_downsample, feature_map_upsample_skipcon, feature_map_upsample_afterlayerup, feature_map_upsample_beforelayerup

def swin_mae_analysis_for_pretrain(**kwargs):
    pretrain_mae_model = SwinMaeAnalysis(
            in_chans=4,
            patch_size=(1, 4),
            window_size=4,
            decoder_embed_dim=768,
            depths=(2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
            qkv_bias=True, mlp_ratio=4,
            drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            bottleneck_channel_reduction=False, **kwargs)

    return pretrain_mae_model
    
def swin_unet(**kwargs):
    model = SwinUnetAnalysis(
        # patch_size=(4, 4),
        depths=(2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model

# RTX 2080_Ti
def swin_unet_moredepths(**kwargs):
    model = SwinUnetAnalysis(
        # patch_size=(4, 4),
        depths=(2, 2, 6, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model

# RTX 2080_Ti
def swin_unet_deep(**kwargs):
    model = SwinUnetAnalysis(
        # patch_size=(4, 4),
        depths=(2, 2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24, 48),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model


# RTX 3090
def swin_unet_deeper(**kwargs):
    model = SwinUnetAnalysis(
        # patch_size=(4, 4),
        depths=(2, 2, 2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24, 48, 96),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model


def swin_unet_pretrain(**kwargs):
    model = SwinUnetAnalysis(
        # patch_size=(4, 4),
        depths=(2, 2, 2, 2), embed_dim=128, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model

def swin_unet_v2(**kwargs):
    model = SwinUnetAnalysis(
        # patch_size=(4, 4),
        depths=(2, 2, 2, 2), embed_dim=96 ,num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), swin_v2=True, **kwargs)
        #  **kwargs)
    return model


def swin_unet_v2_moredepths(**kwargs):
    model = SwinUnetAnalysis(
        # patch_size=(4, 4),
        depths=(2, 2, 6, 2), embed_dim=96 ,num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), swin_v2=True, **kwargs)
        #  **kwargs)
    return model

def swin_unet_v2_deep(**kwargs):
    model = SwinUnetAnalysis(
        # patch_size=(4, 4),
        depths=(2, 2, 2, 2, 2), embed_dim=96 ,num_heads=(3, 6, 12, 24, 48),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), swin_v2=True, **kwargs)
        #  **kwargs)
    return model

def swin_unet_v2_deeper(**kwargs):
    model = SwinUnetAnalysis(
        # patch_size=(4, 4),
        depths=(2, 2, 2, 2, 2, 2), embed_dim=96 ,num_heads=(3, 6, 12, 24, 48, 96),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), swin_v2=True, **kwargs)
        #  **kwargs)
    return model
