import os
import numpy as np
import argparse
import cv2
import math

offset_lut = np.array([48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0])

azimuth_lut = np.array([4.23,1.43,-1.38,-4.18,4.23,1.43,-1.38,-4.18,4.24,1.43,-1.38,-4.18,4.24,1.42,-1.38,-4.19,4.23,1.43,-1.38,-4.19,4.23,1.43,-1.39,-4.19,4.23,1.42,-1.39,-4.2,4.23,1.43,-1.39,-4.19,4.23,1.42,-1.4,-4.2,4.23,1.42,-1.4,-4.2,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.39,-4.2,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.41,-4.21,4.22,1.41,-1.41,-4.21,4.21,1.4,-1.41,-4.21,4.21,1.41,-1.41,-4.21,4.22,1.41,-1.42,-4.22,4.22,1.4,-1.41,-4.22,4.21,1.41,-1.42,-4.22,4.22,1.4,-1.41,-4.22,4.21,1.4,-1.41,-4.23,4.21,1.4,-1.42,-4.23,4.21,1.4,-1.42,-4.22,4.21,1.39,-1.42,-4.22,4.21,1.4,-1.42,-4.21,4.21,1.4,-1.42,-4.22,4.2,1.4,-1.41,-4.22,4.2,1.4,-1.42,-4.22,4.2,1.4,-1.42,-4.22])

elevation_lut = np.array([21.42,21.12,20.81,20.5,20.2,19.9,19.58,19.26,18.95,18.65,18.33,18.02,17.68,17.37,17.05,16.73,16.4,16.08,15.76,15.43,15.1,14.77,14.45,14.11,13.78,13.45,13.13,12.79,12.44,12.12,11.77,11.45,11.1,10.77,10.43,10.1,9.74,9.4,9.06,8.72,8.36,8.02,7.68,7.34,6.98,6.63,6.29,5.95,5.6,5.25,4.9,4.55,4.19,3.85,3.49,3.15,2.79,2.44,2.1,1.75,1.38,1.03,0.68,0.33,-0.03,-0.38,-0.73,-1.07,-1.45,-1.8,-2.14,-2.49,-2.85,-3.19,-3.54,-3.88,-4.26,-4.6,-4.95,-5.29,-5.66,-6.01,-6.34,-6.69,-7.05,-7.39,-7.73,-8.08,-8.44,-8.78,-9.12,-9.45,-9.82,-10.16,-10.5,-10.82,-11.19,-11.52,-11.85,-12.18,-12.54,-12.87,-13.2,-13.52,-13.88,-14.21,-14.53,-14.85,-15.2,-15.53,-15.84,-16.16,-16.5,-16.83,-17.14,-17.45,-17.8,-18.11,-18.42,-18.72,-19.06,-19.37,-19.68,-19.97,-20.31,-20.61,-20.92,-21.22])

origin_offset = 0.015806

lidar_to_sensor_z_offset = 0.03618

angle_off = math.pi * 4.2285/180.

def idx_from_px(px, cols):
    vv = (int(px[0]) + cols - offset_lut[int(px[1])]) % cols
    idx = px[1] * cols + vv
    return idx

def px_to_xyz(px, p_range, cols):
    u = (cols + px[0]) % cols
    azimuth_radians = math.pi * 2.0 / cols 
    encoder = 2.0 * math.pi - (u * azimuth_radians) 
    azimuth = angle_off
    elevation = math.pi * elevation_lut[int(px[1])] / 180.
    x_lidar = (p_range - origin_offset) * math.cos(encoder+azimuth)*math.cos(elevation) + origin_offset*math.cos(encoder)
    y_lidar = (p_range - origin_offset) * math.sin(encoder+azimuth)*math.cos(elevation) + origin_offset*math.sin(encoder)
    z_lidar = (p_range - origin_offset) * math.sin(elevation) 
    x_sensor = -x_lidar
    y_sensor = -y_lidar
    z_sensor = z_lidar + lidar_to_sensor_z_offset
    return [x_sensor, y_sensor, z_sensor]


def pcd_to_img(scan, rows = 128, cols = 2048):
    img_data = np.zeros((rows,cols))
    img_range = np.zeros((rows,cols))
    # max_diff = -0.1
    # avg_err = 0
    # n_val = 0
    for u in range(cols):
        for v in range(rows):

            idx = idx_from_px((u,v), cols)

            # Ouster has a kinda weird reprojection model, see page 12:
            # https://data.ouster.io/downloads/software-user-manual/software-user-manual-v2p0.pdf

            # Compensate beam to center offset
            xy_range = np.sqrt(scan[idx,0]**2 + scan[idx,1]**2) - origin_offset

            # Compensate beam to sensor bottom offset
            z = scan[idx,2] - lidar_to_sensor_z_offset

            # Calculate range as it's defined in the ouster manual
            img_range[v,u] = np.sqrt(xy_range**2 + z**2) + origin_offset

            # # Reproject pixel with range to 3D point
            # point_repro = px_to_xyz((u,v), img_range[v,u], cols)

            # # Check if point is valid
            # if (img_range[v,u] > 0.1):
            #     p_diff = np.sqrt((point_repro[0]-scan[idx,0])**2 + (point_repro[1]-scan[idx,1])**2 + (point_repro[2]-scan[idx,2])**2)
            #     avg_err += p_diff
            #     n_val += 1
            #     if (p_diff > max_diff):
            #         max_diff = p_diff
            #         v_max_diff = v
            #         u_max_diff = u
            img_data[v,u] = scan[idx,3]

    
    intensity_map = img_data
    range_map = img_range
    # max_diff_px = (v_max_diff, u_max_diff)


    return range_map, intensity_map  #, avg_err/n_val, max_diff, max_diff_px




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument('--rows', nargs='?', default=128, type=int)
    parser.add_argument('--cols', nargs='?', default=2048, type=int)
    args = parser.parse_args()
    rows = args.rows
    cols = args.cols

    print("Loading PCD from {}".format(args.path), "with shape", rows, cols)
    
    # data is: x, y, z, intensity
    scan = (np.fromfile(args.path, dtype=np.float32)).reshape(-1, 4)
    
    img_data = np.zeros((rows,cols))
    img_range = np.zeros((rows,cols))
    max_diff = -0.1
    avg_err = 0
    n_val = 0
    for u in range(cols):
        for v in range(rows):

            idx = idx_from_px((u,v), cols)

            # Ouster has a kinda weird reprojection model, see page 12:
            # https://data.ouster.io/downloads/software-user-manual/software-user-manual-v2p0.pdf

            # Compensate beam to center offset
            xy_range = np.sqrt(scan[idx,0]**2 + scan[idx,1]**2) - origin_offset

            # Compensate beam to sensor bottom offset
            z = scan[idx,2] - lidar_to_sensor_z_offset

            # Calculate range as it's defined in the ouster manual
            img_range[v,u] = np.sqrt(xy_range**2 + z**2) + origin_offset

            # Reproject pixel with range to 3D point
            point_repro = px_to_xyz((u,v), img_range[v,u], cols)
            point_raw = [scan[idx,0], scan[idx,1], scan[idx,2]]

            # Check if point is valid
            if (img_range[v,u] > 0.1):
                p_diff = np.sqrt((point_repro[0]-scan[idx,0])**2 + (point_repro[1]-scan[idx,1])**2 + (point_repro[2]-scan[idx,2])**2)
                avg_err += p_diff
                n_val += 1
                if (p_diff > max_diff):
                    max_diff = p_diff
            img_data[v,u] = scan[idx,3]
            
    print("avg_err", avg_err/n_val)
    print("max_diff", max_diff)
    # this conversion is to scale the intensity for a visibile range        
    viz_img = img_range/50.
    # viz_img = img_data / 300.
    cv2.imshow("image", viz_img)
    while cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) > 0:
        keyCode = cv2.waitKey(50)
    exit()