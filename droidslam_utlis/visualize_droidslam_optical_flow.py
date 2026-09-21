import argparse
import glob
import os
import random

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np

from utils import load_images


def match_orb(img0, img1, u0, v0):
    orb = cv2.ORB_create()

    kp0 = cv2.KeyPoint(u0, v0, size=20)
    kp0, desc0 = orb.compute(img0, [kp0])
    if desc0 is None or len(desc0) == 0:
        print(f"No ORB descriptor found at the ({u0}, {v0}).")
        return u0, v0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    h, w = img0.shape[0], img0.shape[1]

    # match keypoint
    # search in a window
    x, y = int(kp0[0].pt[0]), int(kp0[0].pt[1])
    window = 140
    half_win = int(window/2)
    x_min = max(x - half_win, 0)
    y_min = max(y - half_win, 0)
    x_max = min(x + half_win, w - 1)
    y_max = min(y + half_win, h - 1)
    search_kps = []
    for yy in range(y_min, y_max):
        for xx in range(x_min, x_max):
            search_kps.append(cv2.KeyPoint(xx, yy, size=20))
    # search the entire image
    # search_kps = []
    # for yy in range(0, h):
    #     for xx in range(0, w):
    #         search_kps.append(cv2.KeyPoint(xx, yy, size=20))

    search_kps, local_desc = orb.compute(img1, search_kps)

    local_matches = bf.match(desc0, local_desc)
    best_match = min(local_matches, key=lambda m: m.distance)
    matched_kp = search_kps[best_match.trainIdx]
    return int(matched_kp.pt[0]), int(matched_kp.pt[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--opflow_fn", help="path to h5 file", required=True)
    parser.add_argument("--img_folder", help="path to image folder", required=True)
    parser.add_argument('--calib', required=True, help="calibration file in droidslam format")
    parser.add_argument('--fisheye', action="store_true")
    parser.add_argument('--original_res', action="store_true")
    args = parser.parse_args()

    with h5py.File(args.opflow_fn, "r") as f:
        opticalflow_timestamps = f["timestamps"][:]
        opticalflow = f["flow"][:]
        weights = f["weights"][:]
        weights = np.mean(weights, axis=3)
    print(f"Loaded optical flow with shape {opticalflow.shape}")

    images, timestamps = load_images(args.img_folder, args.calib, args.fisheye)
    print(f"Loaded {len(images)} images from {args.img_folder}")

    n = len(images)
    h, w, _ = images[0].shape

    # timestamps
    t_src = 1001700000000
    t_des = 1001750000000
    
    idxs_src = np.argwhere(opticalflow_timestamps[:, 0] == t_src).flatten().tolist()
    idxs_des = np.argwhere(opticalflow_timestamps[:, 1] == t_des).flatten().tolist()

    idx_flow = -1
    for i_src in idxs_src:
        for i_des in idxs_des:
            if i_src == i_des:
                idx_flow = i_src
    
    assert idx_flow >= 0, "Could not find matching optical flow for the desired timestamps"

    eps = 0.001
    idx_img0 = -1
    idx_img1 = -1
    for i, ts in enumerate(timestamps):
        if abs(ts - (t_src*1e-9)) < eps:
            idx_img0 = i
        if abs(ts - (t_des*1e-9)) < eps:
            idx_img1 = i
            break
    
    print(f"Visualizing optical flow between images {idx_img0} ({t_src}) and {idx_img1} ({t_des})")
    img0 = images[idx_img0]
    img1 = images[idx_img1]
    flow01 = opticalflow[idx_flow].astype('float32')
    weights01 = weights[idx_flow].astype('float32')

    # upsample to original image size using scipy (bilinear interpolation)
    # upsampled_flow_u = zoom(flow01[:,:,0], zoom=8, order=1)
    # upsampled_flow_v = zoom(flow01[:,:,1], zoom=8, order=1)

    if args.original_res:
        flow = flow01.astype('float32')
        weights = weights01.astype('float32')

        w, h = flow.shape[1], flow.shape[0]
        
        # Resize to 1/8 resolution
        img0 = cv2.resize(img0, (w, h), interpolation=cv2.INTER_AREA)
        img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)

        flow_norms = np.linalg.norm(flow, axis=-1)
        flow_avg_norm = flow_norms.mean()
        print(f"Flow average norm {flow_avg_norm}")

        print(f"Weights average {weights.mean()}")

        # pick some random points in the image
        u_list = [random.randint(2, w - 2) for _ in range(50)]
        v_list = [random.randint(2, h - 2) for _ in range(50)]

        i = 1
        for u0, v0 in zip(u_list, v_list):
            flow_u = flow[v0, u0, 0]
            flow_v = flow[v0, u0, 1]

            w = weights[v0, u0]

            u1 = int(u0 + flow_u)
            v1 = int(v0 + flow_v)
            # print(f"{i}) [DROIDFlow] Point ({u0}, {v0} on image {idx_img0}) moves to ({u1}, {v1} on image {idx_img1}) with weight {w}")

            # draw the point on the image
            cv2.drawMarker(img0, (u0, v0), (0, 0, 255), cv2.MARKER_CROSS, 1, 1)
            cv2.drawMarker(img1, (u1, v1), (0, 0, 255), cv2.MARKER_CROSS, 1, 1)

            text0 = f"{i}"
            # text1 = f"{i} ({w:.2f})"
            text1 = f"{i}"
            cv2.putText(img0, text0, (u0 + 1, v0 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.01, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(img1, text1, (u1 + 1, v1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.01, (0, 0, 255), 1, cv2.LINE_AA)

            i += 1
        
        img_combined = np.hstack((img0, img1))

        # Display the result
        cv2.imshow('Visualizing Optical Flow', img_combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        # upsample w/ opencv2 (bilinear interpolation)
        upsampled_flow_u = cv2.resize(flow01[:,:,0], (w, h), interpolation=cv2.INTER_LINEAR)
        upsampled_flow_v = cv2.resize(flow01[:,:,1], (w, h), interpolation=cv2.INTER_LINEAR)

        upsampled_weights = cv2.resize(weights01, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # scale flow values by 8 and combine into flow array
        flow = np.zeros((img0.shape[0], img0.shape[1], 2), dtype=np.float32)
        flow[:,:,0] = (upsampled_flow_u * 8).astype('float32')
        flow[:,:,1] = (upsampled_flow_v * 8).astype('float32')
        weights = upsampled_weights.astype('float32')

        assert flow.shape[0] == img0.shape[0] and flow.shape[1] == img0.shape[1], "Optical flow size does not match image size"
        
        flow_norms = np.linalg.norm(flow, axis=-1)
        flow_avg_norm = flow_norms.mean()
        print(f"Flow average norm {flow_avg_norm}")

        print(f"Weights average {weights.mean()}")

        # pick some random points in the image
        u_list = [random.randint(30, w - 30) for _ in range(50)]
        v_list = [random.randint(30, h - 30) for _ in range(50)]
        i = 1
        for u0, v0 in zip(u_list, v_list):
            flow_u = flow[v0, u0, 0]
            flow_v = flow[v0, u0, 1]

            w = weights[v0, u0]

            u1 = int(u0 + flow_u)
            v1 = int(v0 + flow_v)
            print(f"{i}) [DROIDFlow] Point ({u0}, {v0} on image {idx_img0}) moves to ({u1}, {v1} on image {idx_img1}) with weight {w}")

            # draw the point on the image
            # cv2.circle(img0, (u0, v0), 5, (0, 0, 255), -1)
            # cv2.circle(img1, (u1, v1), 5, (0, 0, 255), -1)
            cv2.drawMarker(img0, (u0, v0), (0, 0, 255), cv2.MARKER_CROSS, 5, 1)
            cv2.drawMarker(img1, (u1, v1), (0, 0, 255), cv2.MARKER_CROSS, 5, 1)

            # compute matching with ORB descriptor
            u1_orb, v1_orb = match_orb(img0, img1, u0, v0)

            print(f"{i}) [ORB] Point ({u0}, {v0} on image {idx_img0}) moves to ({u1_orb}, {v1_orb} on image {idx_img1})")

            cv2.circle(img1, (u1_orb, v1_orb), 5, (0, 255, 0), -1)
            cv2.drawMarker(img1, (u1_orb, v1_orb), (0, 255, 0), cv2.MARKER_CROSS, 5, 1)

            text0 = f"{i}"
            text1 = f"{i} ({w:.2f})"
            cv2.putText(img0, text0, (u0 + 10, v0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(img1, text1, (u1 + 10, v1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(img1, text0, (u1_orb + 10, v1_orb - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            i += 1
        
        img_combined = np.hstack((img0, img1))

        # Display the result
        cv2.imshow('Visualizing Optical Flow', img_combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()