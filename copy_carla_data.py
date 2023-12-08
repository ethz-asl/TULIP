import numpy as np
import matplotlib.pyplot as plt


def rimg_loader(path: str) -> np.ndarray:
        """
        Read a range image from a binary file.

        :param filename: filename of range image
        :param dtype: encoding type of binary data (default: float16)
        :param lidar: LiDAR specification for crop the invalid detection distances
        :return: range image encoded by float32 type
        """
        with open(path, 'rb') as f:
            size =  np.fromfile(f, dtype=np.uint, count=2)
            range_image = np.fromfile(f, dtype=np.float16)
        
        range_image = range_image.reshape(size[1], size[0])
        range_image = range_image.transpose()


        return range_image.astype(np.float32)


data_path = "/cluster/work/riner/users/biyang/dataset/Carla/Town06/128_2048"

data = "/cluster/work/riner/users/biyang/dataset/Carla/Town06/128_2048/996.rimg"

img = rimg_loader(data)

plt.imshow(img)
plt.savefig("carla.png")