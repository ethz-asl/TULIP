#!/usr/bin/env python
from model import *
from data_durlar import *
import wandb
import matplotlib.pyplot as plt
import trimesh
from mae.util.evaluation import *


wandb_disabled = False
evaluation = True
save_pcd = True
# project_name = "lidar_sr_network"
project_name = "experiment_upsampling"
entity = "biyang"
run_name = "baseline_lidarSR"
batch_size = 4
epochs = 15



if wandb_disabled:
    mode = "disabled"
else:
    mode = "online"
wandb.init(project=project_name,
            entity=entity,
            name = run_name, 
            mode=mode,
            sync_tensorboard=True)
# wandb.config.update(tf.flags.FLAGS)

def train():
    
    # print('Load training data...  ')
    training_data_input, training_data_pred_ground_truth = load_train_data()

    # print('Compiling model...     ')
    model, model_checkpoint, tensorboard = get_model('training')

    # print('Training model...      ')
    model.fit(
              training_data_input,
              training_data_pred_ground_truth,
              batch_size=batch_size,
              validation_split=0.1,
              epochs=epochs,
              verbose=1,
              shuffle=True,
              callbacks=[model_checkpoint, tensorboard]
             )

    
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # model.save(weight_name)
    model.save_weights(checkpoint_path)


def MC_drop(iterate_count=10):

    test_data_input, test_data_gt = load_test_data()
    # load model
    model, _, _ = get_model('testing')
    print(checkpoint_path)
    # model.load_weights(checkpoint_path)
    model  = tf.keras.models.load_model(checkpoint_path)

    # this_test = np.empty([iterate_count, image_rows_low, image_cols, channel_num], dtype=np.float32)
    this_test = np.empty([iterate_count, image_rows_low, image_cols], dtype=np.float32)
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

        # print(this_prediction.shape)
        # print(this_prediction_mean.shape)
        # print(this_prediction_var.shape)

        # this_prediction[this_prediction_var > this_prediction_mean * 0.03] = 0
        # test_data_prediction[i, :, :, :] = this_prediction
    local_step = 0
    grid_size = 0.1
    for i, (gt, pred) in enumerate(zip(test_data_gt, test_data_prediction)):
        if i % 4 != 0:
            continue
        pred = pred[..., 0:1]
        noise_variance = pred[..., 1:2]
        pred[noise_variance > pred * 0.03] = 0
        pred = np.squeeze(pred)

        low_res_index = range(0, image_rows_high, upscaling_factor)

        # Evaluate the loss of low resolution part
        loss_low_res_part = (pred[low_res_index, :] - gt[low_res_index, :]) ** 2
        loss_low_res_part = loss_low_res_part.mean()

        pred[low_res_index, :] = gt[low_res_index, :]

        mse_all = ((pred - gt)**2).mean()

        # 3D evaluation

        pcd_pred = img_to_pcd(pred)
        pcd_gt = img_to_pcd(gt)

        # print(pcd_pred.shape, pcd_gt.shape)

        pcd_all = np.vstack((pcd_pred, pcd_gt))

        chamfer_dist = chamfer_distance(pcd_gt, pcd_pred)
        min_coord = np.min(pcd_all, axis=0)
        max_coord = np.max(pcd_all, axis=0)
        
        # print("chamfer_dist: ", chamfer_dist)
        # Voxelize the ground truth and prediction point clouds
        voxel_grid_predicted = voxelize_point_cloud(pcd_pred, grid_size, min_coord, max_coord)
        voxel_grid_ground_truth = voxelize_point_cloud(pcd_gt, grid_size, min_coord, max_coord)
        # print("Voxelize")
        # print(voxel_grid_ground_truth.shape, voxel_grid_predicted.shape)
        # Calculate metrics
        iou, precision, recall = calculate_metrics(voxel_grid_predicted, voxel_grid_ground_truth)
        # print("Evaluation")
        # print(iou, precision, recall)

        if save_pcd:
            
            if not os.path.exists(pcd_eval_path):
                os.mkdir(pcd_eval_path)
            pcd_pred_color = np.zeros_like(pcd_pred)
            pcd_pred_color[:, 0] = 255
            pcd_gt_color = np.zeros_like(pcd_gt)
            pcd_gt_color[:, 2] = 255
            
            pcd_all_color = np.vstack((pcd_pred_color, pcd_gt_color))

            point_cloud = trimesh.PointCloud(
                vertices=pcd_all,
                colors=pcd_all_color)
            
            point_cloud.export(os.path.join(pcd_eval_path, f"pred_gt_{local_step}.ply"))
            local_step += 1
            

        wandb.log({"Test/mse_all": mse_all, 
                   "Test/mse_low_res": loss_low_res_part,
                   "Test/chamfer_dist": chamfer_dist, 
                   "Test/iou": iou,
                   "Test/precision": precision,
                   "Test/recall": recall}, local_step)

        gt_vis = scalarMap.to_rgba(gt)[..., :3]
        pred_vis = scalarMap.to_rgba(pred)[..., :3]

        f, ax = plt.subplots(2,1, figsize=(16, 2))
        ax[0].imshow(gt_vis, interpolation='none', aspect="auto")
        ax[0].axis("off")
        ax[1].imshow(pred_vis, interpolation='none', aspect="auto")
        ax[1].axis("off")
        # plt.suptitle("gt-prediction")
        plt.subplots_adjust(wspace=.05, hspace=.05)
        logger = wandb.Image(f)
        plt.close()      
        wandb.log({"gt - pred":
                logger})
        
        


    

    # np.save(os.path.join(home_dir, 'Documents', project_name, test_set + '-' + model_name + '-from-' + str(image_rows_low) + '-to-' + str(image_rows_high) + '_prediction.npy'), test_data_prediction)


if __name__ == '__main__':

    if evaluation:
        MC_drop()
    else:
        # -> train network
        train()

        # -> Monte-Carlo Dropout Test
        MC_drop()
        
