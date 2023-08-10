import torch
from PIL import Image
import numpy as np
from torchvision.transforms.functional import pil_to_tensor, to_grayscale, to_pil_image
from einops import rearrange

def grid_reshape(img):

    B, C, H, W = img.shape

    num_grids = W // H 
    grid_size = num_grids ** 0.5
    assert grid_size == int(grid_size)

    grid_size = int(grid_size)

    new_img = torch.empty((B, C, grid_size * H, grid_size * H))

    for i in range(num_grids):
        u = i // grid_size
        v = i % grid_size
        new_img[:,:, u*H:(u+1)*H, v*H:(v+1)*H] = img[:, :, 0:H, i*H:(i+1)*H]

    return new_img

def grid_reshape_backward(img, target_img_size = (128, 2048)):

    B, C, _, _ = img.shape
    H, W = target_img_size

    num_grids = W // H
    grid_size = num_grids ** 0.5
    assert grid_size == int(grid_size)

    grid_size = int(grid_size)

    new_img = torch.empty((B, C, H, W))

    for i in range(num_grids):
        u = i // grid_size
        v = i % grid_size
        new_img[:, :, 0:H, i*H:(i+1)*H] = img[:,:, u*H:(u+1)*H, v*H:(v+1)*H]

    return new_img

img_path = r"D:\Documents\ETH\Master_Thesis\test.jpg"
img = Image.open(img_path)

img = to_grayscale(img)
img = pil_to_tensor(img)

img = img.unsqueeze(0)

new_img = grid_reshape(img)

# new_img_2 = rearrange(img, 'b c h1 w1 -> b c h2 w2', h1 = 200, w1 = 800, w2 = 400,  h2 = 400)


img_reshape = img.reshape(1, 1, 400, 400)


img_backward = grid_reshape_backward(new_img, (200, 800))

new_img_pil = to_pil_image(new_img.squeeze(), mode = 'L')
reshape_img_pil = to_pil_image(img_reshape.squeeze(), mode = 'L')
img_backward_pil = to_pil_image(img_backward.squeeze(), mode = 'L')
new_img_2_pil = to_pil_image(new_img_2.squeeze(), mode = 'L')

new_img_2_pil.show()
new_img_pil.show()
img_backward_pil.show()
reshape_img_pil.show()




