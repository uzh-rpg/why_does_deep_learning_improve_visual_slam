import sys
sys.path.append('droid_slam')

import numpy as np
import torch
import cv2
import os
import glob
import time
import argparse

from droid_slam.droid_net import DroidNet
import droid_slam.geom.projective_ops as pops
from droid_slam.modules.corr import CorrBlock
from collections import OrderedDict
import h5py

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


def get_image(datapath, idx, calib_fn, image_size, fisheye):
    """ image generator """
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

    intrinsics_vec = [fx, fy, cx, cy]
    ht0, wd0 = [h, w]

    # read all png images in folder
    images_left = sorted(glob.glob(os.path.join(datapath, '*.png')))

    imgL = images_left[idx]

    # TartanAir timestamp convention
    t0 = 1000000000000
    dt = 50000000
    tstamp = t0 + idx * dt

    image = [cv2.remap(cv2.imread(imgL), map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)]
    image_np = image[0]
    image = torch.from_numpy(np.stack(image, 0))
    image = image.permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)
    image = F.interpolate(image, image_size, mode="bilinear", align_corners=False)
    
    intrinsics = torch.as_tensor(intrinsics_vec).cuda()
    intrinsics[0] *= image_size[1] / wd0
    intrinsics[1] *= image_size[0] / ht0
    intrinsics[2] *= image_size[1] / wd0
    intrinsics[3] *= image_size[0] / ht0

    return tstamp, image, intrinsics, image_np


# copy of Droid.load_weights
def load_weights(weights):
    """ load trained model weights """

    net = DroidNet()
    state_dict = OrderedDict([
        (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

    state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
    state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
    state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
    state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

    net.load_state_dict(state_dict)
    net.to("cuda:0").eval()

    return net


@torch.cuda.amp.autocast(enabled=True)
@torch.no_grad()
def compute_opticalflow(net, src_img, dest_img, interpolate=False):

    device = "cuda:0"

    mean = torch.as_tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    stdv = torch.as_tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    ht = src_img.shape[-2] // 8
    wd = src_img.shape[-1] // 8

    # normalize images
    src_inputs = src_img[None, :, [2,1,0]].to(device) / 255.0
    src_inputs = src_inputs.sub_(mean).div_(stdv)

    dest_inputs = dest_img[None, :, [2,1,0]].to(device) / 255.0
    dest_inputs = dest_inputs.sub_(mean).div_(stdv)

    # extract features
    # src
    fmap_t0 = net.fnet(src_inputs).squeeze(0)
    ne, inp = net.cnet(src_inputs[:,[0]]).split([128,128], dim=2)
    net_state, inp_state = ne.tanh().squeeze(0), inp.relu().squeeze(0)

    # dest
    gmap = net.fnet(dest_inputs).squeeze(0)
    
    # index correlation volume
    coords0 = pops.coords_grid(ht, wd, device=device)[None,None]  # (1, 1, ht, wd, 2)
    coords1 = pops.coords_grid(ht, wd, device=device)[None,None]  # (1, 1, ht, wd, 2)

    # Build correlation function (indexes the 4D correlation volume at arbitrary locations)
    corr_fn = CorrBlock(fmap_t0[None,[0]], gmap[None,[0]])

    # Add hidden state batch dim: (1, 1, 128, ht, wd)
    net_state = net_state[None]
    inp_state = inp_state[None]

    # params
    max_it = 20
    it = 0
    delta = None
    delta_norm_thr = 0.0625  # in 1/8 resolution, corresponds to 0.5 pixel in original resolution

    # compute optical flow 
    resd = torch.zeros((1, 1, ht, wd, 2), device=device)  # (1, 1, ht, wd, 2)

    while ((it < max_it) and (delta is None or delta.norm(dim=-1).mean().item() > delta_norm_thr)):
        coords1 = coords1.detach()

        # Recompute correlation at current coords1
        corr = corr_fn(coords1)

        # Motion features: [current_flow, residual]
        # In the factor graph, residual = target - coords1
        # Here we use zero residual since we have no geometric target
        flow = coords1 - coords0
        motion = torch.cat([flow, resd], dim=-1)
        motion = motion.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

        # GRU update — net_state is updated in-place across iterations
        net_state, delta, weights = net.update(net_state, inp_state, corr, motion)

        # Accumulate flow
        coords1 = coords1 + delta.to(dtype=torch.float)
        
        it += 1

    flow = coords1 - coords0

    if interpolate:
        # Bilinear upsampling
        flow_np = np.zeros((src_img.shape[-2], src_img.shape[-1], 2), dtype=np.float16)
        upsampled_flow_u = F.interpolate(flow[:,:,:,:,0], scale_factor=8, mode='bilinear', align_corners=False)
        upsampled_flow_v = F.interpolate(flow[:,:,:,:,1], scale_factor=8, mode='bilinear', align_corners=False)
        flow_np[:,:,0] = (upsampled_flow_u[0,0] * 8).detach().cpu().numpy().astype('float16')
        flow_np[:,:,1] = (upsampled_flow_v[0,0] * 8).detach().cpu().numpy().astype('float16')

        weights_np = np.zeros((src_img.shape[-2], src_img.shape[-1], 2), dtype=np.float16)
        upsampled_weights_u = F.interpolate(weights[:,:,:,:,0], scale_factor=8, mode='bilinear', align_corners=False)
        upsampled_weights_v = F.interpolate(weights[:,:,:,:,1], scale_factor=8, mode='bilinear', align_corners=False)
        weights_np[:,:,0] = (upsampled_weights_u[0,0]).detach().cpu().numpy().astype('float16')
        weights_np[:,:,1] = (upsampled_weights_v[0,0]).detach().cpu().numpy().astype('float16')

    else:
        flow_np = flow[0,0].detach().cpu().numpy().astype('float16')
        weights_np = weights[0,0].detach().cpu().numpy().astype('float16')
    
    return flow_np, weights_np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="path to dataset", required=True)
    parser.add_argument('--scene', help="path to image folder", required=True)
    parser.add_argument('--calib', required=True)
    parser.add_argument('--fisheye', action="store_true")
    parser.add_argument("--weights", default="models/droid.pth")
    parser.add_argument("--max_stepsize", required=True, type=int)
    parser.add_argument("--image_size", default=[480,640])
    parser.add_argument("--out_traj_prefix", help="path to saved estimated trajectory")

    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    imagedir = os.path.join(args.datapath, args.scene)

    calib_fn = os.path.join("calib", args.calib)

    print("Computing optical flow on {}".format(imagedir))
    print(args)

    max_stepsize = args.max_stepsize
    print(f"Computing Optical flow with max step size {max_stepsize}")

    # load network
    net = load_weights(args.weights)
    print("Network Loaded")
    time.sleep(5)

    N = len(glob.glob(os.path.join(imagedir, '*.png')))

    print(f"Going to process {N} images")

    srct_destt_map = []
    srcflow_destflow_map = []
    srcweight_destweight_map = []

    print("Computing Optical flow ...")
    
    for i in range(N-max_stepsize):
        if i % 100 == 0:
            print(f"Processed image {i}/{N}")

        src_t, src_img, _, src_img_np = get_image(imagedir, i, calib_fn, args.image_size, args.fisheye)

        for j in range(1, max_stepsize+1):
            dest_idx = i + j
            dest_t, dest_img, _, dest_img_np = get_image(imagedir, dest_idx, calib_fn, args.image_size, args.fisheye)

            flow, weights = compute_opticalflow(net, src_img, dest_img)

            srct_destt_map.append(np.array([src_t, dest_t], dtype=np.int64))
            srcflow_destflow_map.append(flow)
            srcweight_destweight_map.append(weights)
    # save
    outfn = args.out_traj_prefix + f'_opticalflow_maxstepsize_{max_stepsize}.h5'

    print(f"Saving optical flow to {outfn}")
    with h5py.File(outfn, "w") as f:
        n = len(srcflow_destflow_map)
        h, w, c = srcflow_destflow_map[0].shape
        dsetflow = f.create_dataset("flow", shape=(n, h, w, c), dtype='float16')
        dsetts = f.create_dataset("timestamps", shape=(n, 2), dtype='int64')
        dsetweights = f.create_dataset("weights", shape=(n, h, w, c), dtype='float16')

        for i in range(n):
            dsetflow[i] = srcflow_destflow_map[i]
            dsetts[i] = srct_destt_map[i]
            dsetweights[i] = srcweight_destweight_map[i]
