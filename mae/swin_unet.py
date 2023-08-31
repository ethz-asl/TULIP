import torch
import torch.nn as nn
import torch.nn.functional as func

from einops import rearrange
from typing import Optional, Tuple

from functools import partial
from util.filter import *

from util.evaluation import inverse_huber_loss
from swin_transformer_v2 import SwinTransformerBlockV2, PatchMergingV2
from util.datasets import grid_reshape, grid_reshape_backward

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        x = x.div(keep_prob) * random_tensor
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(4, 4), in_c: int = 3, embed_dim: int = 96, norm_layer: nn.Module = None, circular_padding: bool = False):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        self.circular_padding = circular_padding
        if circular_padding:
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(1, 8), stride=patch_size)
            # self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(8, 1), stride=patch_size)
            # New Projection # Pixel Wise Embedding
            # self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=(1, 1))
        else:
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        # Have to change the grid size, now It should be the same size as the input resolution
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def padding(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        if H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0:
            x = func.pad(x, (0, self.patch_size[0] - W % self.patch_size[1],
                             0, self.patch_size[1] - H % self.patch_size[0],
                             0, 0))
        return x
    
    def circularpadding(self, x: torch.Tensor) -> torch.Tensor:

        # Left side 1 padidng and right side 2 padding
        # Pixel Wise Embedding
        # x = func.pad(x, (1, 2, 0, 0), "circular")

        x = func.pad(x, (2, 2, 0, 0), "circular")
        # x = func.pad(x, (0, 0, 2, 2), "circular")
        return x

    def forward(self, x):
        x = self.padding(x)

        if self.circular_padding:
            # Circular Padding
            x = self.circularpadding(x)

        x = self.proj(x)
        x = rearrange(x, 'B C H W -> B H W C')
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    @staticmethod
    def padding(x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        if H % 2 == 1 or W % 2 == 1:
            x = func.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        return x
    

    @staticmethod
    def merging(x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return x

    def forward(self, x):
        x = self.padding(x)
        x = self.merging(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PixelShuffleExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super(PixelShuffleExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=(1, 1))
        self.upsample = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x: torch.Tensor):
        x = rearrange(x, 'B H W C -> B C H W')
        x = self.expand(x.contiguous())
        x = self.upsample(x)
        # x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=1, P2=4)
        x = rearrange(x, 'B C H W -> B H W C')
        return x


class PatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)
        # self.patch_size = patch_size

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        # x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=1, P2=4)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=2, P2=2)
        x = self.norm(x)
        return x


# TODO: Adjust the input and output size for new input shape

class FinalPatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm, upscale_factor = 4):
        super(FinalPatchExpanding, self).__init__()
        self.dim = dim
        # self.additional_factor = patch_size[0] * patch_size[1]
        # Have to change for loading pretrain weights of ImageNet (input is 256x256, we have 128x128 after patch embedding)
        # self.expand = nn.Linear(dim, 64 * dim, bias=False)
        self.expand = nn.Linear(dim, (upscale_factor**2) * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        # x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=1, 
        #                                                         P2=16, 
        #                                                         C = self.dim)
    
        # Have to change for loading pretrain weights of ImageNet (input is 256x256, we have 128x128 after patch embedding)
        
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=4,
                                                                P2=4,
                                                                C = self.dim)
        # x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=2,
        #                                                         P2=2,
        #                                                         C = self.dim)
        
        x = self.norm(x)
        return x
    
class PixelShuffleHead(nn.Module):
    def __init__(self, dim: int, upscale_factor: int):
        super(PixelShuffleHead, self).__init__()
        self.dim = dim

        self.conv_expand = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim*(upscale_factor**2), kernel_size=(1, 1)),
                                         nn.LeakyReLU(inplace=True))


        # self.conv_expand = nn.Conv2d(in_channels=dim, out_channels=dim*(upscale_factor**2), kernel_size=(1, 1))
        self.upsample = nn.PixelShuffle(upscale_factor=upscale_factor)
        

    def forward(self, x: torch.Tensor):
        x = self.conv_expand(x)
        x = self.upsample(x)
     
        return x
    

class HybridPixelShuffleHead(nn.Module):
    def __init__(self, dim: int, upscale_factor: int, downscale_factor: int):
        super(HybridPixelShuffleHead, self).__init__()
        self.dim = dim

        self.downsample = nn.PixelUnshuffle(downscale_factor=downscale_factor)

        self.conv_expand = nn.Sequential(nn.Conv2d(in_channels=dim*(downscale_factor**2), out_channels=dim*(downscale_factor**2*upscale_factor**2), kernel_size=(1, 1)),
                                         nn.LeakyReLU(inplace=True))
        # self.conv_expand = nn.Conv2d(in_channels=dim, out_channels=dim*(upscale_factor**2), kernel_size=(1, 1))
        self.upsample = nn.PixelShuffle(upscale_factor=upscale_factor*downscale_factor)

    def forward(self, x: torch.Tensor):
        x = self.downsample(x)
        x = self.conv_expand(x)
        x = self.upsample(x)
        return x
        
        


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None,
                 act_layer=nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: Optional[bool] = True,
                 attn_drop: Optional[float] = 0., proj_drop: Optional[float] = 0., shift: bool = False):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        if shift:
            self.shift_size = window_size // 2
        else:
            self.shift_size = 0

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        coords_size = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_size, coords_size], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        x = rearrange(x, 'B (Nh Mh) (Nw Mw) C -> (B Nh Nw) Mh Mw C', Nh=H // self.window_size, Nw=W // self.window_size)
        return x

    def create_mask(self, x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        assert H % self.window_size == 0 and W % self.window_size == 0, "H or W is not divisible by window_size"

        img_mask = torch.zeros((1, H, W, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self.window_partition(img_mask)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        _, H, W, _ = x.shape

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            mask = self.create_mask(x)
        else:
            mask = None

        x = self.window_partition(x)
        Bn, Mh, Mw, _ = x.shape
        x = rearrange(x, 'Bn Mh Mw C -> Bn (Mh Mw) C')
        qkv = rearrange(self.qkv(x), 'Bn L (T Nh P) -> T Bn Nh L P', T=3, Nh=self.num_heads)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bn // nW, nW, self.num_heads, Mh * Mw, Mh * Mw) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, Mh * Mw, Mh * Mw)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = rearrange(x, 'Bn Nh (Mh Mw) C -> Bn Mh Mw (Nh C)', Mh=Mh)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, '(B Nh Nw) Mh Mw C -> B (Nh Mh) (Nw Mw) C', Nh=H // Mh, Nw=H // Mw)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift=False, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop, shift=shift)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_copy = x
        x = self.norm1(x)

        x = self.attn(x)
        x = self.drop_path(x)
        x = x + x_copy

        x_copy = x
        x = self.norm2(x)

        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + x_copy
        return x
    

class BasicBlockV2(nn.Module):
    def __init__(self, index: int, embed_dim: int = 96,input_resolution: tuple=(128, 128), window_size: int = 7, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 norm_layer=nn.LayerNorm, patch_merging: bool = True):
        super(BasicBlockV2, self).__init__()
        depth = depths[index]
        dim = embed_dim * 2 ** index
        num_head = num_heads[index]

        dpr = [rate.item() for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]

        self.blocks = nn.ModuleList([
            SwinTransformerBlockV2(
                dim=dim,
                # input_resolution = (input_resolution[0] // (2 ** i),
                #                     input_resolution[1] // (2 ** i)), 
                input_resolution = input_resolution,
                num_heads=num_head,
                window_size=window_size,
                shift_size= 0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        
        if patch_merging:
            self.downsample = PatchMergingV2(input_resolution=input_resolution,
                                             dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
    

class BasicBlock(nn.Module):
    def __init__(self, index: int, embed_dim: int = 96, window_size: int = 7, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 norm_layer=nn.LayerNorm, patch_merging: bool = True):
        super(BasicBlock, self).__init__()
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
            self.downsample = PatchMerging(dim=embed_dim * 2 ** index, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x




class BasicBlockUp(nn.Module):
    def __init__(self, index: int, embed_dim: int = 96, window_size: int = 7, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 patch_expanding: bool = True, norm_layer=nn.LayerNorm, pixel_shuffle_expanding: bool = False):
        super(BasicBlockUp, self).__init__()
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
        if patch_expanding:
            if pixel_shuffle_expanding:
                self.upsample = PixelShuffleExpanding(dim = embed_dim * 2 ** index)
            else:
                self.upsample = PatchExpanding(dim=embed_dim * 2 ** index, norm_layer=norm_layer)
            
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.upsample(x)
        return x
    
class BasicBlockUpV2(nn.Module):
    def __init__(self, index: int, embed_dim: int = 96, input_resolution: tuple=(128, 128),  window_size: int = 7, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 patch_expanding: bool = True, norm_layer=nn.LayerNorm, pixel_shuffle_expanding: bool = False):
        super(BasicBlockUpV2, self).__init__()
        index = len(depths) - index - 2
        depth = depths[index]
        dim = embed_dim * 2 ** index
        num_head = num_heads[index]

        dpr = [rate.item() for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]

        self.blocks = nn.ModuleList([
            SwinTransformerBlockV2(
                dim=dim,
                # input_resolution = (input_resolution[0] * (2 ** i),
                #                     input_resolution[1] * (2 ** i)), 
                input_resolution = input_resolution,
                num_heads=num_head,
                window_size=window_size,
                shift_size= 0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        if patch_expanding:
            if pixel_shuffle_expanding:
                self.upsample = PixelShuffleExpanding(dim = embed_dim * 2 ** index)
            else:
                self.upsample = PatchExpanding(dim=embed_dim * 2 ** index, norm_layer=norm_layer)
            

            # # Remove Skip Connection
            # self.upsample = nn.PixelShuffle(upscale_factor=2)
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)

        # PixelShuffle have to rearrange
        # x = rearrange(x, 'B H W C -> B C H W')
        x = self.upsample(x)
        # x = rearrange(x, 'B C H W -> B H W C')
        return x


class SwinUnet(nn.Module):
    def __init__(self, img_size = (32, 2048), target_img_size = (128, 2048) ,patch_size = (4, 4), in_chans: int = 1, num_output_channel: int = 1, embed_dim: int = 96,
                 window_size: int = 4, depths: tuple = (2, 2, 6, 2), num_heads: tuple = (3, 6, 12, 24),
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1, norm_layer=nn.LayerNorm, patch_norm: bool = True, edge_loss: bool = False, pixel_shuffle: bool = False,
                 grid_reshape: bool = False, circular_padding: bool = False, swin_v2: bool = False, log_transform: bool = False, depth_scale_loss: bool = False,
                 pixel_shuffle_expanding: bool = False):
        super().__init__()

        self.window_size = window_size
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path = drop_path_rate
        self.norm_layer = norm_layer
        self.img_size = img_size
        self.target_img_size = target_img_size

        self.patch_embed = PatchEmbedding(
            img_size = img_size, patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None, circular_padding=circular_padding)

        
        # TODO: This method not really working, because it compresses the channels of input embedding, which can potentially lead to information loss
        # self.pixel_unshuffle_layer = nn.Sequential(
        #             nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim//4, kernel_size=(1, 1)),
        #             # nn.LeakyReLU(inplace=True),
        #             nn.PixelUnshuffle(downscale_factor=2))
            
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pixel_shuffle_expanding = pixel_shuffle_expanding
        if swin_v2:
            self.layers = self.build_layers_v2()
            self.layers_up = self.build_layers_up_v2()
        else:
            self.layers = self.build_layers()
            self.layers_up = self.build_layers_up()

        if self.pixel_shuffle_expanding:
            self.first_patch_expanding = PixelShuffleExpanding(dim=embed_dim * 2 ** (len(depths) - 1))
        else:
            self.first_patch_expanding = PatchExpanding(dim=embed_dim * 2 ** (len(depths) - 1), norm_layer=norm_layer)
        

        self.skip_connection_layers = self.skip_connection()
        self.norm_up = norm_layer(embed_dim)
        self.head = nn.Conv2d(in_channels=embed_dim, out_channels=num_output_channel, kernel_size=(1, 1), bias=False)
        self.apply(self.init_weights)

        # egde detector and loss
        # Loss functions
        self.edge_loss = edge_loss
        self.log_transform = log_transform
        self.depth_scale_loss = depth_scale_loss



        self.pixel_shuffle = pixel_shuffle
        self.upscale_factor = int(((target_img_size[0]*target_img_size[1]) / (img_size[0]*img_size[1]))**0.5) * 2 
        
        if self.pixel_shuffle:
            # Pixel Wise Embedding : upscale_factor = 2, else 4
            self.pixel_shuffle_layer = PixelShuffleHead(dim = embed_dim, upscale_factor=self.upscale_factor)
            # self.pixel_shuffle_layer = HybridPixelShuffleHead(dim = embed_dim, downscale_factor=window_size, upscale_factor=4)
        else:
            self.final_patch_expanding = FinalPatchExpanding(dim=embed_dim, norm_layer=norm_layer, upscale_factor=self.upscale_factor)

        self.grid_reshape = grid_reshape
        if self.grid_reshape:
            #H, W, C, num_grids, grid_size = params
            # Pixel Wise Embedding
            # H_in = self.img_size[0] 
            # W_in = self.img_size[1]

            H_in = self.img_size[0] // patch_size[0]
            W_in = self.img_size[1] // patch_size[1]

            H_out = self.target_img_size[0]
            W_out = self.target_img_size[1]
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
        if self.edge_loss:
            self.vertical_edge_detector = VerticalEdgeDetectionCNN()
            self.horizontal_edge_detector = HorizontalEdgeDetectionCNN()

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
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
                # Skip Connection via concatenation
                # embed_dim = self.embed_dim * 2, 
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
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
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
                patch_expanding=True if i < self.num_layers - 2 else False,
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
        
        ## l2
        # loss = (pred - target) ** 2  
        ## l1
        pixel_loss = (pred - target).abs()

        ## inverse huber loss
        # loss = inverse_huber_loss(pred, target)

        # Smooth L1 Loss
        # loss = func.smooth_l1_loss(pred, target, reduction='none')
        pixel_loss = pixel_loss.mean()

        if self.log_transform:
            loss = (torch.log1p(pred) - torch.log1p(target)).abs()
            loss = loss.mean()
        elif self.depth_scale_loss:
            loss = (pred - target).abs() * target
            loss = loss.mean()
        else:
            loss = pixel_loss.clone()


        return loss, pixel_loss

    def forward(self, x, target, img_size_high_res, eval = False, mc_drop = False):

        x = self.patch_embed(x) 
        # Have to rearrange to the shape with H * H * C, otherwise the shape won't match in transformer
        # (B, H, W, C)
        if self.grid_reshape:
             # Grid Reshape
            x = grid_reshape(x, self.params_input)
        else:
            x = x.contiguous().view((x.shape[0], int((x.shape[1] * x.shape[2])**0.5), int((x.shape[1] * x.shape[2])**0.5), x.shape[3]))
        x = self.pos_drop(x) 
        x_save = []
        for i, layer in enumerate(self.layers):
            x_save.append(x)
            x = layer(x)

            
        x = self.first_patch_expanding(x)


        for i, layer in enumerate(self.layers_up):
            x = torch.cat([x, x_save[len(x_save) - i - 2]], -1)
            x = self.skip_connection_layers[i](x)
            x = layer(x)

        # # Skip connection with concatenation
        # for i, layer in enumerate(self.layers_up):
        #     x = torch.cat([x, x_save[len(x_save) - i - 2]], -1)
        #     print(x.shape)
        #     x = layer(x.contiguous())

        
        x = self.norm_up(x)
        

        if self.pixel_shuffle:
            x = rearrange(x, 'B H W C -> B C H W')
            x = self.pixel_shuffle_layer(x.contiguous())
            # Grid Reshape
            # (B, C, H, W)
            if self.grid_reshape:
                x = grid_reshape_backward(x, self.params_output, order="bchw")
            # Reshape
            else:
                x = x.view((x.shape[0], x.shape[1], img_size_high_res[0], img_size_high_res[1]))
            x = self.head(x.contiguous())

        else:
            x = self.final_patch_expanding(x)
            # x = grid_reshape_backward(x, img_size_high_res)
            x = rearrange(x, 'B H W C -> B C H W')
            
            # Please consider reshape the image here again, as in transformer we always have the input with shape of B, H, H, C
            # for example, if 32*2048 range map as input
            # 32*2048 -> 16*1024 -> 128 * 128 -> 512*512 -> 128*2048
            if self.grid_reshape:
                x = grid_reshape_backward(x, self.params_output, order="bchw")
            # Reshape
            else: 
                x = x.view((x.shape[0], x.shape[1], img_size_high_res[0], img_size_high_res[1]))
            # Grid Reshape  
            x = self.head(x.contiguous())

        if eval:
            total_loss, pixel_loss = self.forward_loss(x, target)
            return x, total_loss, pixel_loss
        elif mc_drop:
            return x
        else:
            total_loss, pixel_loss = self.forward_loss(x, target)
            return x, total_loss, pixel_loss


def swin_unet(**kwargs):
    model = SwinUnet(
        # patch_size=(4, 4),
        depths=(2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model

# RTX 2080_Ti
def swin_unet_moredepths(**kwargs):
    model = SwinUnet(
        # patch_size=(4, 4),
        depths=(2, 2, 6, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model

# RTX 2080_Ti
def swin_unet_deep(**kwargs):
    model = SwinUnet(
        # patch_size=(4, 4),
        depths=(2, 2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24, 48),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model


# RTX 3090
def swin_unet_deeper(**kwargs):
    model = SwinUnet(
        # patch_size=(4, 4),
        depths=(2, 2, 2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24, 48, 96),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model


def swin_unet_pretrain(**kwargs):
    model = SwinUnet(
        # patch_size=(4, 4),
        depths=(2, 2, 2, 2), embed_dim=128, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        #  **kwargs)
    return model

def swin_unet_v2(**kwargs):
    model = SwinUnet(
        # patch_size=(4, 4),
        depths=(2, 2, 2, 2), embed_dim=96 ,num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), swin_v2=True, **kwargs)
        #  **kwargs)
    return model


def swin_unet_v2_moredepths(**kwargs):
    model = SwinUnet(
        # patch_size=(4, 4),
        depths=(2, 2, 6, 2), embed_dim=96 ,num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), swin_v2=True, **kwargs)
        #  **kwargs)
    return model

def swin_unet_v2_deep(**kwargs):
    model = SwinUnet(
        # patch_size=(4, 4),
        depths=(2, 2, 2, 2, 2), embed_dim=96 ,num_heads=(3, 6, 12, 24, 48),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), swin_v2=True, **kwargs)
        #  **kwargs)
    return model

def swin_unet_v2_deeper(**kwargs):
    model = SwinUnet(
        # patch_size=(4, 4),
        depths=(2, 2, 2, 2, 2, 2), embed_dim=96 ,num_heads=(3, 6, 12, 24, 48, 96),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), swin_v2=True, **kwargs)
        #  **kwargs)
    return model


