# Copyright (c) Lixin YANG, Jiasen Li. All Rights Reserved.
import torch
import numpy as np

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

    # g_yx = np.transpose(g_yx, [0, 3, 2, 1])

    return g_yx

def get_limb_centers_np(joints_2d):
    # limb_parents = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 9, 11, 12, 9, 14, 15] # for human pose
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

def limb_maps_np(pose_points):
    pose_points = pose_points.astype(int)
    pose_points = np.expand_dims(pose_points, 0)
    points_exchanged = np.stack([pose_points[:, :, 1], pose_points[:, :, 0]], 2) / 128. - 1
    limb_centers_yx, angles, limb_length = get_limb_centers_np(points_exchanged)
    # decreasing the value of ratio increases the length of the gaussian
    length_ratios = np.ones(21 - 1) * 2.
    # decreasing the value of ratio increases the width of the gaussian
    # width_ratios = np.asarray([8., 25., 20., 25., 25., 20., 25., 12., 20., 15., 20., 20., 20., 15., 20., 20.]) * np.ones_like(limb_length)
    width_ratios = np.asarray([20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.])\
                   * np.ones_like(limb_length)
    # map_sizes = [256 // 16, 256 // 4, 256 // 2, 256]
    map_size = 64
    # gauss_xy = []

    # for map_size in map_sizes:
    #     rot_gauss_map = get_rot_gaussian_maps_np(limb_centers_yx, [map_size, map_size], width_ratios,
    #                                           length_ratios / limb_length, angles, mode='rot')
    #     gauss_xy.append(rot_gauss_map)
    # return gauss_xy
    rot_gauss_map = get_rot_gaussian_maps_np(limb_centers_yx, [map_size, map_size], width_ratios,
                                             length_ratios / limb_length, angles, mode='rot')

    return rot_gauss_map

def gen_heatmap(img, pt, sigma):
    """generate heatmap based on pt coord.

    :param img: original heatmap, zeros
    :type img: np (H,W) float32
    :param pt: keypoint coord.
    :type pt: np (2,) int32
    :param sigma: guassian sigma
    :type sigma: float
    :return
    - generated heatmap, np (H, W) each pixel values id a probability
    - flag 0 or 1: indicate wheather this heatmap is valid(1)

    """

    pt = pt.astype(np.int32)
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (
            ul[0] >= img.shape[1]
            or ul[1] >= img.shape[0]
            or br[0] < 0
            or br[1] < 0
    ):
        # If not, just return the image as is
        return img, 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return img, 1


def get_heatmap_pred(heatmaps):
    """ get predictions from heatmaps in torch Tensor
        return type: torch.LongTensor
    """
    assert heatmaps.dim() == 4, 'Score maps should be 4-dim (B, nJoints, H, W)'
    maxval, idx = torch.max(heatmaps.view(heatmaps.size(0), heatmaps.size(1), -1), 2)

    maxval = maxval.view(heatmaps.size(0), heatmaps.size(1), 1)
    idx = idx.view(heatmaps.size(0), heatmaps.size(1), 1)

    preds = idx.repeat(1, 1, 2).float()  # (B, njoint, 2)

    preds[:, :, 0] = (preds[:, :, 0]) % heatmaps.size(3)  # + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / heatmaps.size(3))  # + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def gen_heatmap_tensor(img, pt, sigma):
    """generate heatmap based on pt coord.

    :param img: original heatmap, zeros
    :type img: np (H,W) float32
    :param pt: keypoint coord.
    :type pt: np (2,) int32
    :param sigma: guassian sigma
    :type sigma: float
    :return
    - generated heatmap, np (H, W) each pixel values id a probability
    - flag 0 or 1: indicate wheather this heatmap is valid(1)

    """

    pt = pt.int()
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (
            ul[0] >= img.shape[1]
            or ul[1] >= img.shape[0]
            or br[0] < 0
            or br[1] < 0
    ):
        # If not, just return the image as is
        return img, 0

    # Generate gaussian
    size = 6 * sigma + 1
    # x = np.arange(0, size, 1, float)
    x = torch.arange(0, size, 1)
    # y = x[:, np.newaxis]
    y = x[:, None]
    x0 = y0 = size // 2
    # g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g = torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img, 1

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_dists(preds, target, normalize, mask):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))  # (njoint, B)
    for b in range(preds.size(0)):
        for j in range(preds.size(1)):
            if mask[b][j] == 0:
                dists[j, b] = -1
            elif target[b, j, 0] < 1 or target[b, j, 1] < 1:
                dists[j, b] = -1
            else:
                dists[j, b] = torch.dist(preds[b, j, :], target[b, j, :]) / normalize[b]

    return dists

def dist_acc(dist, thr=0.5):
    """ Return percentage below threshold while ignoring values with a -1 """
    dist = dist[dist != -1]
    if len(dist) > 0:
        return 1.0 * (dist < thr).sum().item() / len(dist)
    else:
        return -1

def accuracy_pose_2d(output, target, mask, thr=0.5):
    preds = output.float() # [B, 21, 2]
    gts = target.float() # [B, 21, 2]
    norm = torch.ones(preds.size(0)) * 32 / 10.0
    dists = calc_dists(preds, gts, norm, mask)

    acc = torch.zeros(mask.size(1))
    avg_acc = 0
    cnt = 0

    for i in range(mask.size(1)):  # njoint
        acc[i] = dist_acc(dists[i], thr)
        if acc[i] >= 0:
            avg_acc += acc[i]
            cnt += 1

    if cnt != 0:
        avg_acc /= cnt

    return avg_acc, acc

def accuracy_heatmap(output, target, mask, thr=0.5):
    """ Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First to be returned is average accuracy across 'idxs', Second is individual accuracies
    """
    preds = get_heatmap_pred(output).float()  # (B, njoint, 2)
    gts = get_heatmap_pred(target).float()
    norm = torch.ones(preds.size(0)) * output.size(3) / 10.0  # (B, ), all 6.4:(1/10 of heatmap side)
    dists = calc_dists(preds, gts, norm, mask)  # (njoint, B)

    acc = torch.zeros(mask.size(1))
    avg_acc = 0
    cnt = 0

    for i in range(mask.size(1)):  # njoint
        acc[i] = dist_acc(dists[i], thr)
        if acc[i] >= 0:
            avg_acc += acc[i]
            cnt += 1

    if cnt != 0:
        avg_acc /= cnt

    return avg_acc, acc