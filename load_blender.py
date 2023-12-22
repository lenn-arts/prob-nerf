import os
import torch
import numpy as np
import random
import imageio 
import json
import torch.nn.functional as F
import cv2

random.seed(0)
np.random.seed(3)
torch.random.seed()
torch.manual_seed(0) 

# only apply for train -->
# A1) K=1
#use_ids = [0, 85, 12, 16, 11]
use_ids = [3, 66, 74, 85, 61, 90, 12, 69, 57, 94]
use_ids = ['86', '90', '11', '94', '74', '12', '69', '61', '85', '47', '5', '95', '79', '40', '22', '34', '57', '32', '27', '16']
use_ids = [36, 8]

# A2) K=5
use_ids = [94, 57]
use_ids = [28, 94, 11, 52, 12]
use_ids = [35, 11, 95, 82, 93, 69, 15, 27, 34, 18]
use_ids = [9, 57, 95, 52, 66, 82, 71, 22, 27, 83, 87, 85, 56, 50, 34, 12, 17, 2, 74, 64]
use_ids = [int(use_id) for use_id in use_ids]

# B4) c=25, K=16
use_ids = [0, 86]
#use_ids = [20, 5, 61, 21, 97]
use_ids = [90, 61, 21, 86, 5, 79, 27, 97, 66, 83]
use_ids = [57, 14, 46, 5, 61, 21, 97, 27, 83, 9, 86, 79, 34, 39, 90, 12, 11, 66, 50, 16]

# random uniform
#use_ids = np.random.choice(100, 2, replace=False)

# worst case
#use_ids = list(range(2))
#use_ids = [5,9] + [10,16,23] + [32,34,35,38,40] + [14,24,27,29,31,45,44,51,52,53]

print("use_ids", use_ids)

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for j_frame, frame in enumerate(meta['frames'][::skip]):
            print("frame", j_frame)
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            if s=="train":
                if j_frame not in use_ids: # assumes json has image files in order of their ids
                    continue
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,20+1)[:-1]]+
                                [pose_spherical(90.0, angle, 4.0) for angle in np.linspace(-180,180,20+1)[:-1]]+
                               [pose_spherical(0.0, angle, 4.0) for angle in np.linspace(-180,180,20+1)[:-1]], 0)


    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split


