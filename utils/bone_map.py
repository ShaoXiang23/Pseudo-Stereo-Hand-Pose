import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

opts = {}

opts['batch_size'] = 16
opts['gpu_ids'] = ["0"]
opts['log_dir'] = 'log_dir'
opts['n_summary'] = 40  # number of iterations after which to run the summary-op
opts['n_test'] = 2000
opts['n_checkpoint'] = 1000 # number of iteration after which to save the model
opts['image_size'] = 256
opts['channel_wise_fc'] = False
opts['preds_2d'] = False
opts['sk_3d'] = True
opts['z_embed'] = False
opts['n_joints'] = 17
opts['num_cam_angles'] = 3  # 3 angles
opts['num_cam_params'] = 3  # 0 2d translation, 3 translation
opts['translation_param'] = 0
opts['flip_prob'] = 0.4
opts['tilt_prob'] = 0.3
opts['tilt_limit'] = 10
opts['jitter_prob'] = 0.3
opts['procustes'] = True

def get_rot_gaussian_maps_np(mu, shape_hw, inv_std1, inv_std2, angles, mode='rot'):
    """
    Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
    given the gaussian centers: MU [B, NMAPS, 2] tensor.

    STD: is the fixed standard dev.
    """
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

    y = np.linspace(-1.0, 1.0, shape_hw[0]).astype(float)
    x = np.linspace(-1.0, 1.0, shape_hw[1]).astype(float)

    y = np.reshape(np.repeat(y, [shape_hw[0]]), (-1, shape_hw[0], shape_hw[0]))
    y = np.expand_dims(y, 0) * np.ones((mu.shape[1], shape_hw[0], shape_hw[0]))

    x = np.reshape(np.repeat(x, [shape_hw[1]]), (-1, shape_hw[1], shape_hw[1]))
    x = np.expand_dims(x, 0) * np.ones((mu.shape[1], shape_hw[1], shape_hw[1]))
    x = np.transpose(x, [0, 1, 3, 2])
    mu_y, mu_x = np.expand_dims(mu_y, 3), np.expand_dims(mu_x, 3)

    y = y - mu_y
    x = x - mu_x  # Bx16x14x14

    if mode in ['rot', 'flat']:
        # apply rotation to the grid
        yx_stacked = np.stack([np.reshape(y, (-1, y.shape[1], y.shape[2] * y.shape[3])),
                               np.reshape(x, (-1, x.shape[1], x.shape[2] * x.shape[3]))], 2)  # (B, 16, 2, 196)
        rot_mat = np.stack([np.stack([np.cos(angles), np.sin(angles)], 2),
                            np.stack([-np.sin(angles), np.cos(angles)], 2)], 3)  # (B, 16, 2, 2)

        rotated = np.matmul(rot_mat, yx_stacked)  # (B, 16, 2, 196)

        y_rot = rotated[:, :, 0, :]  # (B, 16, 196)
        x_rot = rotated[:, :, 1, :]  # (B, 16, 196)

        y_rot = np.reshape(y_rot, (-1, mu.shape[1], shape_hw[0], shape_hw[0]))  # (B, 16, 14, 14)
        x_rot = np.reshape(x_rot, (-1, mu.shape[1], shape_hw[1], shape_hw[1]))  # (B, 16, 14, 14)

        g_y = np.square(y_rot)  # (B, 16, 14, 14)
        g_x = np.square(x_rot)  # (B, 16, 14, 14)

        inv_std1 = np.expand_dims(np.expand_dims(inv_std1, 2), 2)  # Bx16x1x1
        inv_std2 = np.expand_dims(np.expand_dims(inv_std2, 2), 2)  # Bx16x1x1

        dist = (g_y * inv_std1 ** 2 + g_x * (inv_std2).astype(float) ** 2)

        if mode == 'rot':
            g_yx = np.exp(-dist)

        else:
            g_yx = np.exp(-np.power(dist + 1e-5, 0.25))

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    g_yx = np.transpose(g_yx, [0, 3, 2, 1])

    return g_yx

def get_rot_gaussian_maps_torch(mu, shape_hw, inv_std1, inv_std2, angles, mode='rot'):
    """
    Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
    given the gaussian centers: MU [B, NMAPS, 2] tensor.

    STD: is the fixed standard dev.
    """
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

    y = torch.linspace(-1.0, 1.0, shape_hw[0]).float().to(device)
    x = torch.linspace(-1.0, 1.0, shape_hw[1]).float().to(device)

    y = torch.reshape(y.repeat([shape_hw[0]]), (-1, shape_hw[0], shape_hw[0]))
    y = torch.unsqueeze(y, 0) * torch.ones((mu.shape[1], shape_hw[0], shape_hw[0])).to(device)

    x = torch.reshape(x.repeat([shape_hw[1]]), (-1, shape_hw[1], shape_hw[1]))
    x = torch.unsqueeze(x, 0) * torch.ones((mu.shape[1], shape_hw[1], shape_hw[1])).to(device)
    # x = np.transpose(x, [0, 1, 3, 2])
    x = x.permute([0, 1, 3, 2])
    mu_y, mu_x = torch.unsqueeze(mu_y, 3), torch.unsqueeze(mu_x, 3)

    y = y - mu_y
    x = x - mu_x  # Bx16x14x14

    if mode in ['rot', 'flat']:
        # apply rotation to the grid
        yx_stacked = torch.stack([torch.reshape(y, (-1, y.shape[1], y.shape[2] * y.shape[3])),
                               torch.reshape(x, (-1, x.shape[1], x.shape[2] * x.shape[3]))], 2)  # (B, 16, 2, 196)
        rot_mat = torch.stack([torch.stack([torch.cos(angles), torch.sin(angles)], 2),
                            torch.stack([-torch.sin(angles), torch.cos(angles)], 2)], 3)  # (B, 16, 2, 2)

        rotated = torch.matmul(rot_mat, yx_stacked)  # (B, 16, 2, 196)

        y_rot = rotated[:, :, 0, :]  # (B, 16, 196)
        x_rot = rotated[:, :, 1, :]  # (B, 16, 196)

        y_rot = torch.reshape(y_rot, (-1, mu.shape[1], shape_hw[0], shape_hw[0]))  # (B, 16, 14, 14)
        x_rot = torch.reshape(x_rot, (-1, mu.shape[1], shape_hw[1], shape_hw[1]))  # (B, 16, 14, 14)

        g_y = torch.square(y_rot)  # (B, 16, 14, 14)
        g_x = torch.square(x_rot)  # (B, 16, 14, 14)

        inv_std1 = torch.unsqueeze(torch.unsqueeze(inv_std1, 2), 2)  # Bx16x1x1
        inv_std2 = torch.unsqueeze(torch.unsqueeze(inv_std2, 2), 2)  # Bx16x1x1

        dist = (g_y * inv_std1 ** 2 + g_x * (inv_std2) ** 2)

        if mode == 'rot':
            g_yx = torch.exp(-dist)

        else:
            g_yx = torch.exp(-torch.pow(dist + 1e-5, 0.25))

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # g_yx = np.transpose(g_yx, [0, 3, 2, 1])
    # g_yx = g_yx.permute([0, 3, 2, 1])

    return g_yx

def get_limb_centers_np(joints_2d):
    # limb_parents = [0, 0, 1, 2, 3, 1, 5, 6, 1, 0, 9, 10, 11, 0, 13, 14, 15]
    limb_parents = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 9, 11, 12, 9, 14, 15]
    angles_x = []
    angles_y = []
    limbs_x = []
    limbs_y = []
    limb_length = []
    for i in range(1, joints_2d.shape[1]):
        x_pair = [joints_2d[:, i, 0], joints_2d[:, limb_parents[i], 0]]
        y_pair = [joints_2d[:, i, 1], joints_2d[:, limb_parents[i], 1]]
        limbs_x.append((x_pair[0] + x_pair[1]) / 2.)
        limbs_y.append((y_pair[0] + y_pair[1]) / 2.)
        limb_length.append(np.sqrt((x_pair[0] - x_pair[1]) ** 2 + (y_pair[0] - y_pair[1]) ** 2))
        angles_x.append(x_pair[0] - x_pair[1])  # because y is represented as x
        angles_y.append(y_pair[0] - y_pair[1])
    angles_x = np.stack(angles_x, 1)
    angles_y = np.stack(angles_y, 1)

    angles = np.arctan2(angles_x, angles_y + 1e-7)

    limbs_x = np.stack(limbs_x, 1)
    limbs_y = np.stack(limbs_y, 1)
    limbs = np.stack([limbs_x, limbs_y], 2)
    limb_length = np.stack(limb_length, 1)

    return limbs, angles, limb_length

def get_limb_centers_torch(joints_2d):
    # limb_parents = [0, 0, 1, 2, 3, 1, 5, 6, 1, 0, 9, 10, 11, 0, 13, 14, 15]
    # limb_parents = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 9, 11, 12, 9, 14, 15]
    limb_parents = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    angles_x = []
    angles_y = []
    limbs_x = []
    limbs_y = []
    limb_length = []
    for i in range(1, joints_2d.shape[1]):
        x_pair = [joints_2d[:, i, 0], joints_2d[:, limb_parents[i], 0]]
        y_pair = [joints_2d[:, i, 1], joints_2d[:, limb_parents[i], 1]]
        limbs_x.append((x_pair[0] + x_pair[1]) / 2.)
        limbs_y.append((y_pair[0] + y_pair[1]) / 2.)
        limb_length.append(torch.sqrt((x_pair[0] - x_pair[1]) ** 2 + (y_pair[0] - y_pair[1]) ** 2))
        angles_x.append(x_pair[0] - x_pair[1])  # because y is represented as x
        angles_y.append(y_pair[0] - y_pair[1])
    angles_x = torch.stack(angles_x, 1)
    angles_y = torch.stack(angles_y, 1)

    angles = torch.atan2(angles_x, angles_y + 1e-7)

    limbs_x = torch.stack(limbs_x, 1)
    limbs_y = torch.stack(limbs_y, 1)
    limbs = torch.stack([limbs_x, limbs_y], 2)
    limb_length = torch.stack(limb_length, 1)

    return limbs, angles, limb_length

def limb_maps_np(pose_points):
    pose_points = pose_points.astype(int)
    points_exchanged = np.stack([pose_points[:, :, 1], pose_points[:, :, 0]], 2) / 128. - 1
    limb_centers_yx, angles, limb_length = get_limb_centers_np(points_exchanged)
    # decreasing the value of ratio increases the length of the gaussian
    length_ratios = np.ones(17 - 1) * 2.
    # decreasing the value of ratio increases the width of the gaussian
    # width_ratios = np.asarray([8., 25., 20., 25., 25., 20., 25., 12., 20., 15., 20., 20., 20., 15., 20., 20.]) * np.ones_like(limb_length)
    width_ratios = np.asarray([20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.]) * np.ones_like(limb_length)
    map_sizes = [opts['image_size'] // 16, opts['image_size'] // 4, opts['image_size'] // 2, opts['image_size']]
    gauss_xy = []

    for map_size in map_sizes:
        rot_gauss_map = get_rot_gaussian_maps_np(limb_centers_yx, [map_size, map_size], width_ratios,
                                              length_ratios / limb_length, angles, mode='rot')
        gauss_xy.append(rot_gauss_map)
    return gauss_xy

def add_amap(am):
    # assert am.shape == (224, 224, 16)
    am_copy = am[:, :, 0]
    for i in range(1, 20):
        am_copy += am[:, :, i]

    return am_copy

def limb_maps_torch(pose_points):
    pose_points_adjust = torch.zeros_like(pose_points).to(device)
    pose_points_adjust[:, :, 0], pose_points_adjust[:, :, 1] = pose_points[:, :, 1], pose_points[:, :, 0]
    points_exchanged = torch.stack([pose_points_adjust[:, :, 1], pose_points_adjust[:, :, 0]], 2) / 128. - 1
    limb_centers_yx, angles, limb_length = get_limb_centers_torch(points_exchanged)
    # decreasing the value of ratio increases the length of the gaussian
    length_ratios = (torch.ones(21 - 1) * 2.).to(device)
    # decreasing the value of ratio increases the width of the gaussian
    # width_ratios = np.asarray([8., 25., 20., 25., 25., 20., 25., 12., 20., 15., 20., 20., 20., 15., 20., 20.]) * np.ones_like(limb_length)
    # width_ratios = torch.from_numpy(
    #     np.asarray([30., 30., 30., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.])).to(device) \
    #                * torch.ones_like(limb_length)
    width_ratios = torch.ones_like(limb_length) * 12
    # map_sizes = [opts['image_size'] // 16, opts['image_size'] // 4, opts['image_size'] // 2, opts['image_size']]
    # gauss_xy = []
    # for map_size in map_sizes:
    #     rot_gauss_map = get_rot_gaussian_maps_torch(limb_centers_yx, [map_size, map_size], width_ratios,
    #                                           length_ratios / limb_length, angles, mode='rot')
    #     gauss_xy.append(rot_gauss_map)
    map_size = 64
    rot_gauss_map = get_rot_gaussian_maps_torch(limb_centers_yx, [map_size, map_size], width_ratios,
                                             length_ratios / limb_length, angles, mode='rot')

    return rot_gauss_map