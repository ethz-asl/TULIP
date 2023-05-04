import os
import numpy as np
import argparse
import cv2

offset_lut = np.array([63,42,21,1,61,41,22,2,60,41,22,3,59,41,22,4,59,40,22,4,58,40,23,5,57,40,23,6,57,40,23,6,56,40,23,6,56,39,23,7,56,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,56,39,23,6,56,39,23,6,56,40,23,6,57,40,22,5,57,40,22,5,58,40,22,4,58,40,22,3,59,41,22,2,60,41,21,1,61,41,21,0])

def idx_from_px(px, cols):
    vv = (int(px[0]) + cols - offset_lut[int(px[1])]) % cols
    idx = px[1] * cols + vv
    return idx

def bin_to_img(scan, rows, cols): # scan = [x, y, z, intensity]
    intensity  = np.zeros((rows,cols))
    range_map = np.zeros((rows,cols))
    for u in range(cols):
        for v in range(rows):
            idx = idx_from_px((u,v), cols)
            intensity[v,u] = scan[idx,3]
            range_map[v,u] = np.linalg.norm(scan[idx, :3])
    return range_map, intensity



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
    for u in range(cols):
        for v in range(rows):
            idx = idx_from_px((u,v), cols)
            img_data[v,u] = scan[idx,3]
            
    # this conversion is to scale the intensity for a visibile range        
    viz_img = img_data / 300.
    cv2.imshow("image", viz_img)
    while cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) > 0:
        keyCode = cv2.waitKey(50)
    exit()