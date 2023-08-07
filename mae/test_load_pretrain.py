import torch
import os

# import swin_mae
from swin_unet import swin_unet, swin_unet_pretrain, swin_unet_v2_pretrain



# model = swin_unet_pretrain(img_size = (32, 2048),
#                                 patch_size = (4, 4), 
#                                 in_chans = 1,
#                                 window_size = 8,
#                                 edge_loss = False)


model = swin_unet_v2_pretrain(img_size = (32, 2048),
                                patch_size = (2, 2), 
                                in_chans = 1,
                                window_size = 8,
                                edge_loss = False)


# print(model)


# pretrain_path = r"D:\Documents\ETH\Master_Thesis\swin_transformer_pretrain\swinv2_base_patch4_window8_256.pth"
# pretrain_path = r"D:\Documents\ETH\Master_Thesis\swin_transformer_pretrain\swin_base_patch4_window7_224.pth"
pretrain_path = r"D:\Documents\ETH\Master_Thesis\swin_transformer_pretrain\swinv2_small_patch4_window8_256.pth"


pretrain = torch.load(pretrain_path, map_location = torch.device('cpu'))
# print(pretrain)

# print(pretrain['model'].keys())


pretrain_model = pretrain['model']
        
msg = model.load_state_dict(pretrain_model, strict=False)

print(msg)