import glob
import os

import cv2
import numpy as np


def load_images(datapath, calib_fn, fisheye):
    calib = np.loadtxt(calib_fn, delimiter=" ")
    fx, fy, cx, cy = calib[:4]
    d1, d2, d3, d4 = calib[4:8]
    w, h = calib[8:10].astype(int)

    K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    new_K = K_l.copy()
    
    if fisheye:
        d_l = np.array([[d1], [d2], [d3], [d4]])
        map_l = cv2.fisheye.initUndistortRectifyMap(K_l, d_l, np.eye(3), new_K, (w, h), cv2.CV_32F)
    else:
        d_l = np.array([d1, d2, d3, d4, 0.0])
        map_l = cv2.initUndistortRectifyMap(K_l, d_l, np.eye(3), new_K, (w, h), cv2.CV_32F)

    # read all png images in folder
    images_fns = sorted(glob.glob(os.path.join(datapath, '*.png')))
    timestamps = []
    images = []

    for t, img in enumerate(images_fns):
        timestamps.append(float(img.split('/')[-1][:-4])*1e-9)
        images.append(cv2.remap(cv2.imread(img), map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR))

    return images, timestamps