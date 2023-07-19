#!/usr/bin/env python
from model_pytorch import *
from data_durlar import *
import argparse
import wandb

def get_args_parser():
    parser = argparse.ArgumentParser('LiDAR Super Resolution Network', add_help=False)

    # Model Parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='optimizer for training')
    
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--crop', action="store_true", help='crop the image to 128 x 128 (default)')
    parser.add_argument('--upscaling_factor', default=4, type=int, help="The factor of superresolution")

    # Training parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    parser.add_argument('--wandb_disabled', action='store_true', help="disable wandb")
    parser.add_argument('--entity', type = str, default = "biyang")
    parser.add_argument('--project_name', type = str, default = "Ouster_MAE")
    parser.add_argument('--run_name', type = str, default = None)

    parser.add_argument('--eval', action='store_true', help="evaluation")

def get_optimizer(args, param_groups):
    if args.optimizer == 'adamw':
        return torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    elif args.optimizer == 'adam':
        return torch.optim.Adam(param_groups, lr=args.lr, betas=(0.9, 0.95))

def main(args):
    
    if args.wandb_disabled:
            mode = "disabled"
    else:
        mode = "online"
    wandb.init(project=args.project_name,
                entity=args.entity,
                name = args.run_name, 
                mode=mode,
                sync_tensorboard=True)
    wandb.config.update(args)

    dataset_train = build_durlar_upsampling_dataset(is_train = True, args = args)
    dataset_val = build_durlar_upsampling_dataset(is_train = False, args = args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, shuffle = True, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, shuffle = False, 
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = UNet(upscaling_factor=args.upscaling_factor)

    # TODO  
    if args.eval and os.path.exists(args.output_dir):
        pass
    
    

# TODO
def train_one_epoch():
    pass
# TODO
def evaluate():
    pass

def train():
    
    # print('Load training data...  ')
    training_data_input, training_data_pred_ground_truth = load_train_data()

    # print('Compiling model...     ')
    model, model_checkpoint, tensorboard = get_model('training')

    # print('Training model...      ')
    model.fit(
              training_data_input,
              training_data_pred_ground_truth,
              batch_size=32,
              validation_split=0.1,
              epochs=100,
              verbose=1,
              shuffle=True,
              callbacks=[model_checkpoint, tensorboard]
             )

    model.save(weight_name)


def MC_drop(iterate_count=50):

    test_data_input, _ = load_test_data()
    # load model
    model, _, _ = get_model('testing')
    model.load_weights(weight_name)

    this_test = np.empty([iterate_count, image_rows_low, image_cols, channel_num], dtype=np.float32)
    test_data_prediction = np.empty([test_data_input.shape[0], image_rows_high, image_cols, 2], dtype=np.float32)

    for i in range(test_data_prediction.shape[0]):

        print('Processing {} th of {} images ... '.format(i, test_data_prediction.shape[0]))
        
        for j in range(iterate_count):
            this_test[j] = test_data_input[i]

        this_prediction = model.predict(this_test, verbose=1)

        this_prediction_mean = np.mean(this_prediction, axis=0)
        this_prediction_var = np.std(this_prediction, axis=0)
        test_data_prediction[i,:,:,0:1] = this_prediction_mean
        test_data_prediction[i,:,:,1:2] = this_prediction_var

    np.save(os.path.join(home_dir, 'Documents', project_name, test_set + '-' + model_name + '-from-' + str(image_rows_low) + '-to-' + str(image_rows_high) + '_prediction.npy'), test_data_prediction)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # -> train network
    train()

    # -> Monte-Carlo Dropout Test
    MC_drop()
    
