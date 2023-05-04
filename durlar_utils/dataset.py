from timm.data.dataset import ImageDataset
from torchvision import datasets, transforms
import os


# if __name__ ==  "main":
path = "/cluster/work/riner/users/biyang/dataset/depth_test/"
is_train = False
root = os.path.join(path, 'train' if is_train else 'val')
dataset = ImageDataset(root)
# is_train = False
# 
#dataset = datasets.ImageFolder(root, transform=None)

print(dataset[0])

