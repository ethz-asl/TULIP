from mae.swin_unet import *


class SwinResidual(nn.Module):
    def __init__(self, img_size = (32, 2048), target_img_size = (128, 2048) ,patch_size = (4, 4), in_chans: int = 1, num_output_channel: int = 1, embed_dim: int = 96,
                 window_size: int = 4, depths: tuple = (2, 2, 6, 2), num_heads: tuple = (3, 6, 12, 24),
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1, norm_layer=nn.LayerNorm, patch_norm: bool = True, edge_loss: bool = False, pixel_shuffle: bool = False,
                 grid_reshape: bool = False, circular_padding: bool = False, swin_v2: bool = False, log_transform: bool = False, depth_scale_loss: bool = False,
                 pixel_shuffle_expanding: bool = False, relative_dist_loss: bool = False, perceptual_loss: bool = False, pretrain_mae: nn.Module = None,
                 output_multidims: bool = False, delta_pixel_loss: bool = False, shift_only_leftright: bool = False):
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
        self.relative_dist_loss = relative_dist_loss
        self.perceptual_loss = perceptual_loss
        self.delta_pixel_loss = delta_pixel_loss
        self.shift_only_leftright = shift_only_leftright

        if self.delta_pixel_loss:
            self.downsample_factor = target_img_size[0] // img_size[0]
            self.low_res_indices = [range(i, target_img_size[0]+i, self.downsample_factor) for i in range(self.downsample_factor)]

        if self.perceptual_loss:
            assert pretrain_mae is not None, "Pretrain MAE model is not provided"
            self.pretrain_mae = pretrain_mae
            # Target is already logtransfromed
            # self.perceptual_loss_criterion = nn.KLDivLoss(reduction='batchmean', log_target=False)
            # self.perceptual_loss_criterion = nn.CosineSimilarity(reduction='batchmean')
            # self.perceptual_loss_criterion = nn.MSELoss()
            # downsampling_factor = target_img_size[0] // img_size[0]
            # self.low_res_index = range(0, target_img_size[0], downsampling_factor)


        
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
        self.output_multidims = output_multidims

        if self.output_multidims:
            self.num_output_channel = 4
            self.initialize_weights_for_output_concatenation(output_dim=self.num_output_channel)
            self.patch_embed = PatchEmbedding(img_size = img_size, patch_size=patch_size, in_c=self.num_output_channel, embed_dim=embed_dim,
                                                norm_layer=norm_layer if patch_norm else None, circular_padding=circular_padding)
            self.decoder_pred = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_output_channel, kernel_size=(1, 1), bias=False)
        else:
            self.patch_embed = PatchEmbedding(img_size = img_size, patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                                norm_layer=norm_layer if patch_norm else None, circular_padding=circular_padding)

            self.decoder_pred = nn.Conv2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=(1, 1), bias=False)
        
        

        self.apply(self.init_weights)



        # egde detector and loss
        # Loss functions
        self.edge_loss = edge_loss
        self.log_transform = log_transform
        self.depth_scale_loss = depth_scale_loss



        self.pixel_shuffle = pixel_shuffle
        self.upscale_factor = int(((target_img_size[0]*target_img_size[1]) / (img_size[0]*img_size[1]))**0.5) * 2 * int(((patch_size[0]*patch_size[1])//4)**0.5)
        
        if self.pixel_shuffle:
            # Pixel Wise Embedding : upscale_factor = 2, else 4
            self.ps_head = PixelShuffleHead(dim = embed_dim, upscale_factor=self.upscale_factor)
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

    def initialize_weights_for_output_concatenation(self, output_dim: int = 4):
        # We initialize with weighted probability, as other 3 channels input are pseudo-input with added noise
        self.output_weights = nn.Parameter(torch.Tensor([0.7, 0.1, 0.1, 0.1]).reshape(1, output_dim, 1, 1), requires_grad=True)
        # self.output_weights = nn.Parameter(torch.ones((1, output_dim, 1, 1)) * (1/output_dim), requires_grad=True)

    def create_multidims_input(self, x, is_train = False):
        
        x = torch.tile(x, (1, self.num_output_channel, 1, 1))

        # if is_train:
        noise = torch.randn_like(x) * 0.05
        # add random noise to the input
        x[:, 1:, :, :] = x[:, 1:, :, :] + noise[:, 1:, :, :]

        return x

    def weighted_summation(self, x):
        assert self.output_weights is not None
        x = x*self.output_weights
        x = x.sum(1, keepdim=True)
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
                patch_merging=False if i == self.num_layers - 1 else True,
                shift_only_leftright=self.shift_only_leftright)
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
                pixel_shuffle_expanding=self.pixel_shuffle_expanding,
                shift_only_leftright=self.shift_only_leftright)
            layers_up.append(layer)
        return layers_up

    def skip_connection(self):
        skip_connection_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            dim = self.embed_dim * 2 ** (self.num_layers - 2 - i)
            layer = nn.Linear(dim * 2, dim)
            skip_connection_layers.append(layer)
        return skip_connection_layers
    
    
    # TODO: Add relative loss function:
    def row_relative_loss(self, pred, target, num_pixels = 500, num_neighbors = 8):
        
        B, C, H, W = pred.shape
        loss = 0
        for i in range(B):
        
            sampler_h = torch.randint(0, H, size = (num_pixels,),device=pred.device)
            sampler_w = torch.randint(0, W-num_neighbors, size=(num_pixels,), device=pred.device)

            patches_selected_pred = pred[i, :, sampler_h, sampler_w]
            patches_selected_target = target[i, :, sampler_h, sampler_w]
            relative_dist_pred = torch.zeros_like(patches_selected_pred, device=pred.device)
            relative_dist_target = torch.zeros_like(patches_selected_target, device=pred.device)

            # Found a Bug Here (iterator of B and neighbor is the same)
            for j in range(num_neighbors):
                relative_dist_pred += (patches_selected_pred - pred[i, :, sampler_h, sampler_w+j]).abs()
                relative_dist_target += (patches_selected_target - pred[i, :, sampler_h, sampler_w+j]).abs()

            loss += (relative_dist_pred - relative_dist_target).abs().mean()
        return torch.log1p(loss / B)

    def square_relative_loss(self, pred, target, num_pixels = 500, dilation = 0):
        B, C, H, W = pred.shape
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
        return torch.log1p(loss / B)
    
    def compute_delta_pixel(self, input, pred, target):
        delta_pixel = 0
        for i in range(self.downsample_factor - 1):
            delta_ = (input - pred[:, :, self.low_res_indices[i+1], :]) - (input - target[:, :, self.low_res_indices[i+1], :])
            delta_pixel += delta_.abs().mean()

        return delta_pixel

    # def depth_wise_concate(self, x):
    #     downsample_factor = 4
    #     h_high_res = x.shape[2]
    #     low_res_indices = [range(i, h_high_res+i, downsample_factor) for i in range(downsample_factor)]
    #     x = torch.cat([x[:, :, low_res_indices[i], :] for i in range(len(low_res_indices))], dim = 1)
    #     return x
    
    def forward_loss(self, pred, target, input):
        
        ## l2
        # loss = (pred - target) ** 2  
        ## l1
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


        if self.delta_pixel_loss:
            
            dp_loss = 0.2*self.compute_delta_pixel(input, pred, target)
            loss += dp_loss

        if self.relative_dist_loss:
            # Give a lambda weight to the additional loss
            loss += 0.1*self.row_relative_loss(pred, target, num_pixels=50, num_neighbors=8)
            # loss += 0.1* self.square_relative_loss(pred, target, num_pixels=500, dilation=1)

        if self.perceptual_loss:
            # TODO: Force low resolution feature map close to gt high resolution feature map
            feature_loss = self.pretrain_mae(pred, target)
            # feature_loss = (pred_feature_map_backbone - target_feature_map)**2
            # feature_loss = feature_loss.mean()
            loss += 0.1*feature_loss

        return loss, pixel_loss

    def forward(self, x, target, img_size_high_res, eval = False, mc_drop = False):


        input = x.clone().detach()

        if self.output_multidims:
            x = self.create_multidims_input(x, is_train = not (eval or mc_drop))


        x = self.patch_embed(x) 
        # Have to rearrange to the shape with H * H * C, otherwise the shape won't match in transformer
        # (B, H, W, C)
        if self.grid_reshape:
             # Grid Reshape
            x = grid_reshape(x, self.params_input)
        # elif self.window_size[0] == self.window_size[1]:
        #     x = x.contiguous().view((x.shape[0], int((x.shape[1] * x.shape[2])**0.5), int((x.shape[1] * x.shape[2])**0.5), x.shape[3]))
        x = self.pos_drop(x) 
        x_save = []
        for i, layer in enumerate(self.layers):
            x_save.append(x)
            x = layer(x)


        # feature_map_back_bone = x.clone().detach()
            
        x = self.first_patch_expanding(x)


        for i, layer in enumerate(self.layers_up):
            x = torch.cat([x, x_save[len(x_save) - i - 2]], -1)
            x = self.skip_connection_layers[i](x)
            x = layer(x)

        
        x = self.norm_up(x)
        

        if self.pixel_shuffle:
            x = rearrange(x, 'B H W C -> B C H W')
            x = self.ps_head(x.contiguous())
            # Grid Reshape
            # (B, C, H, W)
            if self.grid_reshape:
                x = grid_reshape_backward(x, self.params_output, order="bchw")
            # Reshape
            x = self.decoder_pred(x.contiguous())
        else:
            x = self.final_patch_expanding(x)
            # x = grid_reshape_backward(x, img_size_high_res)
            x = rearrange(x, 'B H W C -> B C H W')
            
            # Please consider reshape the image here again, as in transformer we always have the input with shape of B, H, H, C
            # for example, if 32*2048 range map as input
            # 32*2048 -> 16*1024 -> 128 * 128 -> 512*512 -> 128*2048
            if self.grid_reshape:
                x = grid_reshape_backward(x, self.params_output, order="bchw")
            # Grid Reshape  
            x = self.decoder_pred(x.contiguous())

        if self.output_multidims:
            x = self.weighted_summation(x)

        if eval:
            total_loss, pixel_loss = self.forward_loss(x, target, input)
            return x, total_loss, pixel_loss
        elif mc_drop:
            return x
        else:
            total_loss, pixel_loss = self.forward_loss(x, target, input)
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
        depths=(2, 2, 2, 6, 2), embed_dim=96, num_heads=(3, 6, 12, 24, 48),
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


