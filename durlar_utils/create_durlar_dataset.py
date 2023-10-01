import os
import shutil
import argparse
from glob import glob
import pathlib

'''Create the real world lidar dataset from the raw data
Training Data:
    - 1. DurLAR_20210716 41994
    - 2. DurLAR_20211012 28643
    - 3. DurLAR 20211208 26851
We sample one data from each 4 frames, #training data = (41994 + 28642 + 26850) / 4 = 24459

Test Data:
    - 1. DurLAR 20211209 25080
We sample one data from each 10 frames #test data = 25080 / 10 = 2508'''




def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str , default=None)
    parser.add_argument("--output_path", type=str , default=None)
    parser.add_argument("--train_data_per_frame", type = int, default = 4, help = "skip rate of training data")
    parser.add_argument("--test_data_per_frame", type = int, default = 10, help = "skip rate of test data")
   
    return parser.parse_args()


def main(args):
    train_data_folder = ['DurLAR_20210716', 'DurLAR_20211012', 'DurLAR_20211208']
    test_data_folder = ['DurLAR_20211209']

    train_data_per_frame = args.train_data_per_frame
    test_data_per_frame = args.test_data_per_frame


    # Load all test data (fullpath name)
    train_data = {'depth': [], 'intensity': []}
    for folder in train_data_folder:
        depth_files = glob(os.path.join(args.data_path, folder, 'depth', '*.npy'))
        intensity_files = glob(os.path.join(args.data_path, folder, 'intensity', '*.npy'))

        depth_files.sort()
        intensity_files.sort()

        train_data['depth'].extend(depth_files)
        train_data['intensity'].extend(intensity_files)


    # Load all test data (fullpath name)
    test_data = {'depth': [], 'intensity': []}
    for folder in test_data_folder:
        depth_files = glob(os.path.join(args.data_path, folder, 'depth', '*.npy'))
        intensity_files = glob(os.path.join(args.data_path, folder, 'intensity', '*.npy'))

        depth_files.sort()
        intensity_files.sort()

        test_data['depth'].extend(depth_files)
        test_data['intensity'].extend(intensity_files)

    outputdir_fullpath = args.output_path


    # Create folder and directory for saving the data
    for data_split in ['train', 'val']:
        output_image_dir = os.path.join(outputdir_fullpath, data_split)

        for modality in ['depth', 'intensity']:
            outputdir_modality = os.path.join(os.path.join(output_image_dir, modality), 'all')
            outputdir_modality_fullpath = pathlib.Path(outputdir_modality)
            outputdir_modality_fullpath.parent.mkdir(parents = True, exist_ok = True)

            if os.path.exists(outputdir_modality):
                pass
            else:
                os.mkdir(outputdir_modality)
    


    assert len(train_data['depth']) == len(train_data['intensity'])
    assert len(test_data['depth']) == len(test_data['intensity'])


    # Copy the data to the output folder and rename it
    print("There are totally {} data for training, we skip with rate {}".format(len(train_data['depth']), train_data_per_frame))
    print("There are totally {} data for testing, we skip with rate {}".format(len(test_data['depth']), test_data_per_frame))



    # Saving Training data
    for i in range(len(train_data['depth'])):
        if i % train_data_per_frame == 0:
            # Copy the data to the output folder
            output_image_dir = os.path.join(outputdir_fullpath, 'train')

            # Copy the depth data
            outputdir_modality = os.path.join(os.path.join(output_image_dir, 'depth'), 'all')
            shutil.copy(train_data['depth'][i], os.path.join(outputdir_modality, '{:08d}.npy'.format(i)))

            # Copy the intensity data
            outputdir_modality = os.path.join(os.path.join(output_image_dir, 'intensity'), 'all')
            shutil.copy(train_data['intensity'][i], os.path.join(outputdir_modality, '{:08d}.npy'.format(i)))


    print("Training Data saved!")


    for i in range(len(test_data['depth'])):
        if i % test_data_per_frame == 0:
            # Copy the data to the output folder
            output_image_dir = os.path.join(outputdir_fullpath, 'val')

            # Copy the depth data
            outputdir_modality = os.path.join(os.path.join(output_image_dir, 'depth'), 'all')
            shutil.copy(test_data['depth'][i], os.path.join(outputdir_modality, '{:08d}.npy'.format(i)))

            # Copy the intensity data
            outputdir_modality = os.path.join(os.path.join(output_image_dir, 'intensity'), 'all')
            shutil.copy(test_data['intensity'][i], os.path.join(outputdir_modality, '{:08d}.npy'.format(i)))


    print("Test Data saved!")
    






if __name__ == "__main__":
    args = read_args()
    main(args)

