#!/usr/bin/env python
import os
import datetime

from data import *
import torch
import torch.nn as nn



# Create Model 
model_name = 'UNet'
# case_name = model_name + '-mae-' + data_set + '-from-' + str(image_rows_low) + '-to-' + str(image_rows_high)
case_name = "your_prediction"

# automatically generate log and weight path
log_path = os.path.join(root_dir, 'logs', case_name)
weight_path = os.path.join(root_dir, 'weights', case_name)
weight_name = os.path.join(weight_path, 'weights.h5')
print('#'*50)
print('Using model:              {}'.format(model_name))
print('Trainig case:             {}'.format(case_name))
print('Log directory:            {}'.format(log_path))
print('Weight directory:         {}'.format(weight_path))
print('Weight name:              {}'.format(weight_name))
path_lists = [log_path, weight_path]
for folder_name in path_lists:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

class ConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # TODO: Check the padding
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, stride = (1, 1)):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride)
        # self.conv = DoubleConv(in_channels, out_channels)
        self.norm = nn.BatchNorm2d(out_channels),
        self.activation = nn.ReLU(inplace=True),

    def forward(self, x):
        x = self.up(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class UNet(nn.Module):
    def __init__(self, upscaling_factor, in_chans = 1, dropout_rate = 0.25):
        super(UNet, self).__init__()
        self.upscailing_layer = nn.ModuleList()
        self.mid_channels_base = 64

        self.upscaling_factor = upscaling_factor
        self.in_chans = in_chans
        self.dropout_rate = dropout_rate

        self.final_layer = nn.Sequential([nn.Conv2d(self.mid_channels_base, 1, (1, 1)),
                                        nn.ReLU(inplace=True)])
        
        self.upscaling_layers = self.build_upscaling_layers()
        self.upsampling_layers = self.build_upsampling_layers()
        self.downsampling_layers = self.build_downsampling_layers()


    def build_upscaling_layers(self):
        layers = nn.ModuleList()
        for _ in range(int(np.log(upscaling_factor) / np.log(2))):
            layers.append(UpBlock(self.in_chans, self.mid_channels_base, stride = (2, 1)))
        
        return layers

    def build_upsampling_layers(self):
        layers = nn.ModuleList()
        layers.append(ConvBlock(in_channels=self.mid_channels_base, 
                                out_channels=self.mid_channels_base))
        for i in range(1, 3):
            UpsamplingBlock = nn.Sequential([nn.AvgPool2d(2),
                                             nn.Dropout(self.dropout_rate),
                                             ConvBlock(in_channels=self.mid_channels_base * (2**(i-1)),
                                                       out_channels=self.mid_channels_base * (2**i))])
            layers.append(UpsamplingBlock)

        return layers

    def build_downsampling_layers(self):
        layers = nn.ModuleList()
        layers.append(nn.Sequential([nn.AvgPool2d(2),
                                    nn.Dropout(self.dropout_rate),
                                    ConvBlock(in_channels=self.mid_channels_base * (2**3),
                                            out_channels=self.mid_channels_base * (2**4)),
                                    nn.Dropout(self.dropout_rate),
                                    UpBlock(in_channels=self.mid_channels_base * (2**4), 
                                            out_channels=self.mid_channels_base * (2**3),
                                            stride = (2, 2))]))
        
        for i in range(3, 1, -1):
            DownsamplingBlock = nn.Sequential([ConvBlock(in_channels=self.mid_channels_base * (2**i),
                                                       out_channels=self.mid_channels_base * (2**(i-1))),
                                                nn.Dropout(self.dropout_rate),
                                                UpBlock(in_channels=self.mid_channels_base * (2**i),
                                                       out_channels=self.mid_channels_base * (2**(i-1)),
                                                       stride = (2, 2))
                                             ])
            layers.append(DownsamplingBlock)
            
        layers.append(ConvBlock(in_channels=self.mid_channels_base,
                                out_channels=self.mid_channels_base))


        return layers
    
        


    def forward(self, x):
        x = self.upscaling_layers(x)

        x_save = []
        
        for layer in self.upsampling_layers:
            x = layer(x)
            x_save.append(x)
        
        for i,  layer in enumerate(self.downsampling_layers):
            x = layer(x)
            x = torch.cat([x, x_save[len(x_save) - i - 1]], -1)
        
        x = self.final_layer(x)

        return 

###########################################################################################
#################################   some functions    #####################################
###########################################################################################

# def create_case_dir(type_name):
#     # tensorboard
#     model_checkpoint = None
#     tensorboard = None
#     os.system('killall tensorboard')
#     # create tensorboard checkpoint
#     if type_name == 'training':
#         model_checkpoint = ModelCheckpoint(weight_name, save_best_only=True, period=1)
#         tensorboard = TensorBoard(log_dir=log_path)
#         # run tensorboard
#         command = 'tensorboard --logdir=' + os.path.join(root_dir, 'logs') + ' &'
#         os.system(command)
#         # delete old log files
#         for the_file in os.listdir(log_path):
#             file_path = os.path.join(log_path, the_file)
#             if os.path.isfile(file_path):
#                 os.unlink(file_path)

#     return model_checkpoint, tensorboard


# def get_model(type_name='training'):
#     # create case dir
#     model_checkpoint, tensorboard = create_case_dir(type_name)
#     # create default model
#     model = None
#     # Choose Model
#     if model_name == 'UNet':
#         model = UNet()

#     return model, model_checkpoint, tensorboard