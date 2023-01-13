# Written by Guo.S.X 2020.12.15
# Vision Lab in Ocean University of China, Department of Information Science, B302.
# This code aims to processing STB hands (SK) datasets.
from __future__ import print_function
from scipy.io import loadmat
# from data.generate_hps import _generate_target, vis_image_with_heatmaps
# from Visualization.Visualization_3D_Pose import plot3D
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.mplot3d import Axes3D
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from pylab import *
import numpy as np, torch, cv2, os
import utils.func as func
from scipy.optimize import minimize

def perspective(points, calibrations):
    """Compute the perspective projections of 3D points into the image plane by given projection matrix
    Args:
        points (tensot): [Bx3xN] tensor of 3D points
        calibrations (tensor): [Bx4x4] Tensor of projection matrix
    Returns:
        tensor: [Bx3xN] Tensor of uvz coordinates in the image plane
    """
    if points.shape[1] == 2:
        points = torch.cat([points, torch.ones([points.shape[0], 1, points.shape[2]]).to(points.device)], 1)
    z = points[:, 2:3].clone()
    points[:, :3] = points[:, :3] / z
    points1 = torch.cat([points, torch.ones([points.shape[0], 1, points.shape[2]]).to(points.device)], 1)
    points_img = torch.bmm(calibrations, points1)
    points_img = torch.cat([points_img[:, :2], z], 1)

    return points_img


def perspective_np(points, calibrations):
    """Compute the perspective projections of 3D points into the image plane by given projection matrix
    Args:
        points (array): [BxNx3] array of 3D points
        calibrations (array): [Bx4x4] Tensor of projection matrix
    Returns:
        array: [BxNx3] Tensor of uvz coordinates in the image plane
    """
    if points.shape[1] == 2:
        points = np.concatenate([points, np.ones([points.shape[0], 1])], -1)
    z = points[:, 2:3].copy()
    points[:, :3] /= z
    points1 = np.concatenate([points, np.ones([points.shape[0], 1])], -1)
    points_img = np.dot(calibrations, points1.T).T
    points_img = np.concatenate([points_img[:, :2], z], -1)

    return points_img

def compute_iou(pred, gt):
    """Mask IoU
    Args:
        pred (array): prediction mask
        gt (array): ground-truth mask
    Returns:
        float: IoU
    """
    area_pred = pred.sum()
    area_gt = gt.sum()
    if area_pred == area_gt == 0:
        return 1
    union_area = (pred + gt).clip(max=1)
    union_area = union_area.sum()
    inter_area = area_pred + area_gt - union_area
    IoU = inter_area / union_area

    return IoU


def cnt_area(cnt):
    """Compute area of a contour
    Args:
        cnt (array): contour
    Returns:
        float: area
    """
    area = cv2.contourArea(cnt)
    return area

def tensor_to_image(tsr):
    tsr = func.batch_denormalize(tsr, [0.5, 0.5, 0.5], [1, 1, 1])
    tsr = func.bchw_2_bhwc(tsr)
    nmy = func.to_numpy(tsr.detach().cpu())
    if nmy.dtype is not np.uint8:
        nmy = (nmy * 255).astype(np.uint8)
    image = nmy.astype(np.uint8)

    return image

# draw_2dimg
def draw_2Dimg(kps_2d, image):
    color_hand_joints = [[1.0, 0.0, 0.0],
                         [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                         [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                         [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                         [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                         [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

    skeleton_overlay = image.copy()
    pose_uv = np.copy(kps_2d)

    assert pose_uv.shape[0] == 21

    marker_sz = 3
    line_wd = 2
    root_ind = 0

    for joint_ind in range(pose_uv.shape[0]):
        joint = pose_uv[joint_ind, 0].astype('int32'), pose_uv[joint_ind, 1].astype('int32')
        cv2.circle(
            skeleton_overlay, joint,
            radius=marker_sz, color=color_hand_joints[joint_ind] * np.array(255), thickness=-1,
            lineType=cv2.LINE_AA)

        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            root_joint = pose_uv[root_ind, 0].astype('int32'), pose_uv[root_ind, 1].astype('int32')
            cv2.line(skeleton_overlay, root_joint, joint,
                color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                lineType=cv2.LINE_AA)
        else:
            joint_2 = pose_uv[joint_ind - 1, 0].astype('int32'), pose_uv[joint_ind - 1, 1].astype('int32')
            cv2.line(skeleton_overlay, joint_2, joint,
                color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                lineType=cv2.LINE_AA)

    return skeleton_overlay

# generate heatmaps.
def _generate_target(joints, nof_joints=21, heatmap_sigma=2):
    """
    :param joints:  [nof_joints, 2]
    :param joints_vis: [nof_joints, 2]
    :return: target, target_weight(1: visible, 0: invisible)
    """
    nof_joints = nof_joints
    heatmap_type = 'gaussian'
    image_size, heatmap_size = [256, 256], [64, 64]
    heatmap_sigma = heatmap_sigma
    # use_different_joints_weight = True

    target_weight = np.ones((nof_joints, 1), dtype=np.float32)

    if heatmap_type == 'gaussian':
        target = np.zeros((nof_joints,
                           heatmap_size[1],
                           heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = heatmap_sigma * 3

        for joint_id in range(nof_joints):
            feat_stride = np.asarray(image_size) / np.asarray(heatmap_size)
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)] # up-left
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)] # down-right
            # control the range
            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * heatmap_sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    else:
        raise NotImplementedError

    # if use_different_joints_weight:
    #     target_weight = np.multiply(target_weight, joints_weight)

    return target

# draw 48 sphere model
def plot3D_sphere(ax, pos, mesh=0, r_lists=0, azim=0, elev=0):
    color_hand_joints = [[1.0, 0.0, 0.0],
                         [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                         [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                         [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                         [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                         [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.view_init(elev=-85, azim=-75)
    # ax.set_aspect('equal')
    position = np.copy(pos)
    uvd_pt = np.reshape(position, [21, 3])
    x_min, x_max = min(uvd_pt[:, 0]), max(uvd_pt[:, 0])
    y_min, y_max = min(uvd_pt[:, 1]), max(uvd_pt[:, 1])
    z_min, z_max = min(uvd_pt[:, 2]), max(uvd_pt[:, 2])

    center = [(x_min + x_max) / 2,
              (y_min + y_max) / 2,
              (z_min + z_max) / 2]

    # STB
    ax.set_xlim(center[0] - 85, center[0] + 85)
    ax.set_ylim(center[1] - 85, center[1] + 85)
    ax.set_zlim(center[2] - 85, center[2] + 85)
    # ax.dist = 8
    # ax.grid(True)

    marker_sz = 15
    line_wd = 2

    # for joint_ind in range(uvd_pt.shape[0]):
    #     if joint_ind > 0:
    #         ax.plot(uvd_pt[joint_ind:joint_ind + 1, 0], uvd_pt[joint_ind:joint_ind + 1, 1],
    #                 uvd_pt[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        # elif joint_ind % 4 == 1:
        #     ax.plot(uvd_pt[[0, joint_ind], 0], uvd_pt[[0, joint_ind], 1], uvd_pt[[0, joint_ind], 2],
        #             color=color_hand_joints[joint_ind], lineWidth=line_wd)
        # else:
        #     ax.plot(uvd_pt[[joint_ind - 1, joint_ind], 0], uvd_pt[[joint_ind - 1, joint_ind], 1],
        #             uvd_pt[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
        #             linewidth=line_wd)


    pai_2 = 3.14*1.2
    mesh = np.asarray(mesh) # [16, 3]
    palm = mesh[:16]
    r_palm = r_lists[:16]
    for joint_ind in range(palm.shape[0]):
        ax.plot(palm[joint_ind:joint_ind + 1, 0], palm[joint_ind:joint_ind + 1, 1],
                palm[joint_ind:joint_ind + 1, 2], '.', c='lightgrey', markersize=r_palm[joint_ind]*pai_2)
        # if joint_ind == 0:
        #     continue

    index = mesh[16:22]
    r_index = r_lists[16:22]
    for joint_ind in range(index.shape[0]):
        ax.plot(index[joint_ind:joint_ind + 1, 0], index[joint_ind:joint_ind + 1, 1],
                index[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[8], markersize=r_index[joint_ind]*pai_2)
    middle = mesh[22:28]
    r_middle = r_lists[22:28]
    for joint_ind in range(middle.shape[0]):
        ax.plot(middle[joint_ind:joint_ind + 1, 0], middle[joint_ind:joint_ind + 1, 1],
                middle[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[12], markersize=r_middle[joint_ind]*pai_2)
    ring = mesh[28:34]
    r_ring = r_lists[28:34]
    for joint_ind in range(ring.shape[0]):
        ax.plot(ring[joint_ind:joint_ind + 1, 0], ring[joint_ind:joint_ind + 1, 1],
                ring[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[16], markersize=r_ring[joint_ind]*pai_2)
    little = mesh[34:40]
    r_little = r_lists[34:40]
    for joint_ind in range(little.shape[0]):
        ax.plot(little[joint_ind:joint_ind + 1, 0], little[joint_ind:joint_ind + 1, 1],
                little[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[20], markersize=r_little[joint_ind]*pai_2)
    thumb = mesh[40:48]
    r_thumb = r_lists[40:48]
    for joint_ind in range(thumb.shape[0]):
        ax.plot(thumb[joint_ind:joint_ind + 1, 0], thumb[joint_ind:joint_ind + 1, 1],
                thumb[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[4], markersize=r_thumb[joint_ind]*pai_2)

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    return ax

def plot3D_sphere_adjust(ax, mesh=0, r_lists=0, azim=0, elev=0):
    color_hand_joints = [[1.0, 0.0, 0.0],
                         [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                         [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                         [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                         [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                         [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.view_init(elev=-85, azim=-75)
    # ax.set_aspect('equal')
    # uvd_pt = np.reshape(position, [21, 3])

    # for joint_ind in range(uvd_pt.shape[0]):
    #     if joint_ind > 0:
    #         ax.plot(uvd_pt[joint_ind:joint_ind + 1, 0], uvd_pt[joint_ind:joint_ind + 1, 1],
    #                 uvd_pt[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        # elif joint_ind % 4 == 1:
        #     ax.plot(uvd_pt[[0, joint_ind], 0], uvd_pt[[0, joint_ind], 1], uvd_pt[[0, joint_ind], 2],
        #             color=color_hand_joints[joint_ind], lineWidth=line_wd)
        # else:
        #     ax.plot(uvd_pt[[joint_ind - 1, joint_ind], 0], uvd_pt[[joint_ind - 1, joint_ind], 1],
        #             uvd_pt[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
        #             linewidth=line_wd)


    pai_2 = 3.14*1.5
    mesh = np.asarray(mesh) # [16, 3]
    palm = mesh[:16]
    r_palm = r_lists[:16]
    for joint_ind in range(palm.shape[0]):
        ax.plot(palm[joint_ind:joint_ind + 1, 0], palm[joint_ind:joint_ind + 1, 1],
                palm[joint_ind:joint_ind + 1, 2], '.', c='lightgrey', markersize=r_palm[joint_ind]*pai_2)
        # if joint_ind == 0:
        #     continue

    index = mesh[16:22]
    r_index = r_lists[16:22]
    for joint_ind in range(index.shape[0]):
        ax.plot(index[joint_ind:joint_ind + 1, 0], index[joint_ind:joint_ind + 1, 1],
                index[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[8], markersize=r_index[joint_ind]*pai_2)
    middle = mesh[22:28]
    r_middle = r_lists[22:28]
    for joint_ind in range(middle.shape[0]):
        ax.plot(middle[joint_ind:joint_ind + 1, 0], middle[joint_ind:joint_ind + 1, 1],
                middle[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[12], markersize=r_middle[joint_ind]*pai_2)
    ring = mesh[28:34]
    r_ring = r_lists[28:34]
    for joint_ind in range(ring.shape[0]):
        ax.plot(ring[joint_ind:joint_ind + 1, 0], ring[joint_ind:joint_ind + 1, 1],
                ring[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[16], markersize=r_ring[joint_ind]*pai_2)
    little = mesh[34:40]
    r_little = r_lists[34:40]
    for joint_ind in range(little.shape[0]):
        ax.plot(little[joint_ind:joint_ind + 1, 0], little[joint_ind:joint_ind + 1, 1],
                little[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[20], markersize=r_little[joint_ind]*pai_2)
    thumb = mesh[40:48]
    r_thumb = r_lists[40:48]
    for joint_ind in range(thumb.shape[0]):
        ax.plot(thumb[joint_ind:joint_ind + 1, 0], thumb[joint_ind:joint_ind + 1, 1],
                thumb[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[4], markersize=r_thumb[joint_ind]*pai_2)

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    return ax

def plot_pc(ax, pc, pos, azim=0, elev=0):
    ax.view_init(azim=azim, elev=elev)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # axis_bounds = np.array([np.min(hand_points[:, 0]), np.max(hand_points[:, 0]),
    #                         np.min(hand_points[:, 1]), np.max(hand_points[:, 1]),
    #                         np.min(hand_points[:, 2]), np.max(hand_points[:, 2])])
    # mask = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
    # axis_bounds[mask] -= 20
    # axis_bounds[~mask] += 20
    # mask1 = (pc[:, 0] >= axis_bounds[0]) & (pc[:, 0] <= axis_bounds[1])
    # mask2 = (pc[:, 1] >= axis_bounds[2]) & (pc[:, 1] <= axis_bounds[3])
    # mask3 = (pc[:, 2] >= axis_bounds[4]) & (pc[:, 2] <= axis_bounds[5])
    # inumpyuts = mask1 & mask2 & mask3
    # pc0 = pc[inumpyuts]
    #
    # ax.scatter(pc0[:, 0], pc0[:, 1], pc0[:, 2])
    color_hand_joints = [[1.0, 0.0, 0.0],
                         [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                         [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                         [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                         [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                         [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.view_init(elev=-85, azim=-75)
    # ax.set_aspect('equal')
    position = np.copy(pos)
    uvd_pt = np.reshape(position, [21, 3])

    marker_sz = 30
    line_wd = 5

    for joint_ind in range(uvd_pt.shape[0]):
        ax.plot(uvd_pt[joint_ind:joint_ind + 1, 0], uvd_pt[joint_ind:joint_ind + 1, 1],
                uvd_pt[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(uvd_pt[[0, joint_ind], 0], uvd_pt[[0, joint_ind], 1], uvd_pt[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lineWidth=line_wd)
        else:
            ax.plot(uvd_pt[[joint_ind - 1, joint_ind], 0], uvd_pt[[joint_ind - 1, joint_ind], 1],
                    uvd_pt[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                    linewidth=line_wd)

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='lightsteelblue')

    return ax

def plot_pc_sphere(ax, pc, mesh, r_lists, azim=0, elev=0):
    ax.view_init(azim=azim, elev=elev)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    color_hand_joints = [[1.0, 0.0, 0.0],
                         [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                         [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                         [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                         [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                         [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.view_init(elev=-85, azim=-75)
    # ax.set_aspect('equal')
    # uvd_pt = np.reshape(position, [21, 3])

    # for joint_ind in range(uvd_pt.shape[0]):
    #     if joint_ind > 0:
    #         ax.plot(uvd_pt[joint_ind:joint_ind + 1, 0], uvd_pt[joint_ind:joint_ind + 1, 1],
    #                 uvd_pt[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
    # elif joint_ind % 4 == 1:
    #     ax.plot(uvd_pt[[0, joint_ind], 0], uvd_pt[[0, joint_ind], 1], uvd_pt[[0, joint_ind], 2],
    #             color=color_hand_joints[joint_ind], lineWidth=line_wd)
    # else:
    #     ax.plot(uvd_pt[[joint_ind - 1, joint_ind], 0], uvd_pt[[joint_ind - 1, joint_ind], 1],
    #             uvd_pt[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
    #             linewidth=line_wd)

    pai_2 = 3.14 * 1.5
    mesh = np.asarray(mesh)  # [16, 3]
    palm = mesh[:16]
    r_palm = r_lists[:16]
    for joint_ind in range(palm.shape[0]):
        ax.plot(palm[joint_ind:joint_ind + 1, 0], palm[joint_ind:joint_ind + 1, 1],
                palm[joint_ind:joint_ind + 1, 2], '.', c='lightgrey', markersize=r_palm[joint_ind] * pai_2)
        # if joint_ind == 0:
        #     continue

    index = mesh[16:22]
    r_index = r_lists[16:22]
    for joint_ind in range(index.shape[0]):
        ax.plot(index[joint_ind:joint_ind + 1, 0], index[joint_ind:joint_ind + 1, 1],
                index[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[8], markersize=r_index[joint_ind] * pai_2)
    middle = mesh[22:28]
    r_middle = r_lists[22:28]
    for joint_ind in range(middle.shape[0]):
        ax.plot(middle[joint_ind:joint_ind + 1, 0], middle[joint_ind:joint_ind + 1, 1],
                middle[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[12],
                markersize=r_middle[joint_ind] * pai_2)
    ring = mesh[28:34]
    r_ring = r_lists[28:34]
    for joint_ind in range(ring.shape[0]):
        ax.plot(ring[joint_ind:joint_ind + 1, 0], ring[joint_ind:joint_ind + 1, 1],
                ring[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[16], markersize=r_ring[joint_ind] * pai_2)
    little = mesh[34:40]
    r_little = r_lists[34:40]
    for joint_ind in range(little.shape[0]):
        ax.plot(little[joint_ind:joint_ind + 1, 0], little[joint_ind:joint_ind + 1, 1],
                little[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[20],
                markersize=r_little[joint_ind] * pai_2)
    thumb = mesh[40:48]
    r_thumb = r_lists[40:48]
    for joint_ind in range(thumb.shape[0]):
        ax.plot(thumb[joint_ind:joint_ind + 1, 0], thumb[joint_ind:joint_ind + 1, 1],
                thumb[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[4], markersize=r_thumb[joint_ind] * pai_2)

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))


    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='lightsteelblue')

    return ax

# draw 3d pose
def plot3D(ax, pos, azim=0, elev=0):
    color_hand_joints = [[1.0, 0.0, 0.0],
                         [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                         [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                         [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                         [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                         [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1.0, 1.0, 1.0])
    position = np.copy(pos)
    uvd_pt = np.reshape(position, [21, 3])
    x_min, x_max = min(uvd_pt[:, 0]), max(uvd_pt[:, 0])
    y_min, y_max = min(uvd_pt[:, 1]), max(uvd_pt[:, 1])
    z_min, z_max = min(uvd_pt[:, 2]), max(uvd_pt[:, 2])

    center = [(x_min + x_max) / 2,
              (y_min + y_max) / 2,
              (z_min + z_max) / 2]

    # STB
    ax.set_xlim(center[0] - 0.1, center[0] + 0.1)
    ax.set_ylim(center[1] - 0.1, center[1] + 0.1)
    ax.set_zlim(center[2] - 0.1, center[2] + 0.1)
    ax.dist = 8
    ax.grid(True)

    # marker_sz = 30
    # line_wd = 6
    marker_sz = 20
    line_wd = 5

    for joint_ind in range(uvd_pt.shape[0]):
        ax.plot(uvd_pt[joint_ind:joint_ind + 1, 0], uvd_pt[joint_ind:joint_ind + 1, 1],
                uvd_pt[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(uvd_pt[[0, joint_ind], 0], uvd_pt[[0, joint_ind], 1], uvd_pt[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lw=line_wd)
        else:
            ax.plot(uvd_pt[[joint_ind - 1, joint_ind], 0], uvd_pt[[joint_ind - 1, joint_ind], 1],
                    uvd_pt[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                    lw=line_wd)

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    return ax

from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def plot3D_original_coord(ax, pos, z, azim=-90., elev=-90.):
    color_hand_joints = [[1.0, 0.0, 0.0],
                         [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                         [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                         [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                         [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                         [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

    # ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.view_init(elev=-85, azim=-75)
    # ax.set_aspect('equal')
    ax.set_box_aspect([0.75, 0.75, 1.0])
    # ax.set_xlim(-0.045, 0.045)
    # ax.set_ylim(-0.06, 0.06)
    ax.set_zlim(0, z)
    uvd_pt = np.copy(pos)

    marker_sz = 20
    line_wd = 5

    for joint_ind in range(uvd_pt.shape[0]):
        ax.plot(uvd_pt[joint_ind:joint_ind + 1, 0], uvd_pt[joint_ind:joint_ind + 1, 1],
                uvd_pt[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(uvd_pt[[0, joint_ind], 0], uvd_pt[[0, joint_ind], 1], uvd_pt[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lw=line_wd)
        else:
            ax.plot(uvd_pt[[joint_ind - 1, joint_ind], 0], uvd_pt[[joint_ind - 1, joint_ind], 1],
                    uvd_pt[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                    lw=line_wd)

    # ax.scatter(0, 0, 0)
    # ax.plot(
    #     [[0], [0]],
    #     [[0], [0]],
    #     [[uvd_pt[9, 2]], [0]],
    #     linestyle='-.',
    #     lw=2
    # )
    # ax.arrow([[0], [0]], [[0], [0]], [[uvd_pt[9, 2]], [0]], linestyle='-', lw=2)
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='<|-|>', color='r', shrinkA=0, shrinkB=0, lw=3, linestyle='--')
    arw = Arrow3D([uvd_pt[9, 0], 0], [uvd_pt[9, 1], 0], [uvd_pt[9, 2], 0], **arrow_prop_dict)
    ax.add_artist(arw)

    length = np.sqrt(uvd_pt[9, 0]**2 + uvd_pt[9, 1]**2 + uvd_pt[9, 2]**2)
    length = str(length)[:4] + 'm'
    ax.text(uvd_pt[9, 0]*0.5, uvd_pt[9, 1]*0.5, 0.5*uvd_pt[9, 2], length, color='red', size=50)

    ''' draw camera position '''
    scale = 0.02
    depth = 0.05
    ax.plot([-scale, -scale], [scale, -scale], [0, 0], color='red', lw=2) # AB
    ax.plot([-scale, scale], [scale, scale], [0, 0], color='red', lw=2) # AC
    ax.plot([scale, scale], [scale, -scale], [0, 0], color='red', lw=2) # CD
    ax.plot([-scale, scale], [-scale, -scale], [0, 0], color='red', lw=2) # BD

    ax.plot([-scale, 0], [scale, 0], [0, -depth], color='red', lw=2) # AE
    ax.plot([-scale, 0], [-scale, 0], [0, -depth], color='red', lw=2)  # BE
    ax.plot([scale, 0], [scale, 0], [0, -depth], color='red', lw=2)  # CE
    ax.plot([scale, 0], [-scale, 0], [0, -depth], color='red', lw=2)  # DE

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    return ax

def plot3D_(ax, pos, azim=0, elev=0):
    color_hand_joints = [[1.0, 0.0, 0.0],
                         [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                         [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                         [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                         [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                         [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.view_init(elev=-85, azim=-75)
    # ax.set_aspect('equal')
    position = np.copy(pos)
    uvd_pt = np.reshape(position, [21, 3])
    x_min, x_max = min(uvd_pt[:, 0]), max(uvd_pt[:, 0])
    y_min, y_max = min(uvd_pt[:, 1]), max(uvd_pt[:, 1])
    z_min, z_max = min(uvd_pt[:, 2]), max(uvd_pt[:, 2])

    center = [(x_min + x_max) / 2,
              (y_min + y_max) / 2,
              (z_min + z_max) / 2]

    # STB
    ax.set_xlim(center[0] - 85, center[0] + 85)
    ax.set_ylim(center[1] - 85, center[1] + 85)
    ax.set_zlim(center[2] - 85, center[2] + 85)
    ax.dist = 7.5
    # ax.grid(True)

    marker_sz = 15
    line_wd = 2

    for joint_ind in range(uvd_pt.shape[0]):
        ax.plot(uvd_pt[joint_ind:joint_ind + 1, 0], uvd_pt[joint_ind:joint_ind + 1, 1],
                uvd_pt[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(uvd_pt[[0, joint_ind], 0], uvd_pt[[0, joint_ind], 1], uvd_pt[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lineWidth=line_wd)
        else:
            ax.plot(uvd_pt[[joint_ind - 1, joint_ind], 0], uvd_pt[[joint_ind - 1, joint_ind], 1],
                    uvd_pt[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                    linewidth=line_wd)



    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    return ax, center

def plot3D_no_adjust(ax, pos, azim=0, elev=0):
    color_hand_joints = [[1.0, 0.0, 0.0],
                         [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                         [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                         [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                         [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                         [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.view_init(elev=-85, azim=-75)
    # ax.set_aspect('equal')
    position = np.copy(pos)
    uvd_pt = np.reshape(position, [21, 3])

    marker_sz = 15
    line_wd = 2

    for joint_ind in range(uvd_pt.shape[0]):
        ax.plot(uvd_pt[joint_ind:joint_ind + 1, 0], uvd_pt[joint_ind:joint_ind + 1, 1],
                uvd_pt[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(uvd_pt[[0, joint_ind], 0], uvd_pt[[0, joint_ind], 1], uvd_pt[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lineWidth=line_wd)
        else:
            ax.plot(uvd_pt[[joint_ind - 1, joint_ind], 0], uvd_pt[[joint_ind - 1, joint_ind], 1],
                    uvd_pt[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                    linewidth=line_wd)

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    return ax

# draw 3d pose
def plot3D_ball(ax, pos, azim=0, elev=0):
    color_hand_joints = [[1.0, 0.0, 0.0],
                         [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                         [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                         [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                         [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                         [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.view_init(elev=-85, azim=-75)
    # ax.set_aspect('equal')
    position = np.copy(pos)
    uvd_pt = np.reshape(position, [21, 3])
    x_min, x_max = min(uvd_pt[:, 0]), max(uvd_pt[:, 0])
    y_min, y_max = min(uvd_pt[:, 1]), max(uvd_pt[:, 1])
    z_min, z_max = min(uvd_pt[:, 2]), max(uvd_pt[:, 2])

    center = [(x_min + x_max) / 2,
              (y_min + y_max) / 2,
              (z_min + z_max) / 2]

    # STB
    ax.set_xlim(center[0] - 85, center[0] + 85)
    ax.set_ylim(center[1] - 85, center[1] + 85)
    ax.set_zlim(center[2] - 85, center[2] + 85)
    # ax.dist = 8
    # ax.grid(True)

    marker_sz = 60
    line_wd = 2

    for joint_ind in range(uvd_pt.shape[0]):
        ax.plot(uvd_pt[joint_ind:joint_ind + 1, 0], uvd_pt[joint_ind:joint_ind + 1, 1],
                uvd_pt[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(uvd_pt[[0, joint_ind], 0], uvd_pt[[0, joint_ind], 1], uvd_pt[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lineWidth=line_wd)
        else:
            ax.plot(uvd_pt[[joint_ind - 1, joint_ind], 0], uvd_pt[[joint_ind - 1, joint_ind], 1],
                    uvd_pt[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                    linewidth=line_wd)

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    return ax

def plot3D_adjust(ax, pos, azim=0, elev=0):
    color_hand_joints = [[1.0, 0.0, 0.0],
                         [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                         [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                         [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                         [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                         [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little
    ax.grid(True)
    # ax.grid(color='r')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.view_init(elev=-85, azim=-75)
    # ax.set_aspect('auto')
    position = np.copy(pos)
    uvd_pt = np.reshape(position, [21, 3])
    x_min, x_max = min(uvd_pt[:, 0]), max(uvd_pt[:, 0])
    y_min, y_max = min(uvd_pt[:, 1]), max(uvd_pt[:, 1])
    z_min, z_max = min(uvd_pt[:, 2]), max(uvd_pt[:, 2])

    center = [(x_min + x_max) / 2,
              (y_min + y_max) / 2,
              (z_min + z_max) / 2]

    # STB
    # ax.set_xlim(x_min - 5, x_max + 5)
    # ax.set_ylim(y_max - 5, y_min + 5)
    # ax.set_zlim(z_min - 8, z_max + 8)
    ax.set_xlim(center[0] - 60, center[0] + 60)
    ax.set_ylim(center[1] - 60, center[1] + 60)
    ax.set_zlim(center[2] - 60, center[2] + 60)
    ax.dist = 9
    # ax.grid(True)

    marker_sz = 15
    line_wd = 2

    for joint_ind in range(uvd_pt.shape[0]):
        ax.plot(uvd_pt[joint_ind:joint_ind + 1, 0], uvd_pt[joint_ind:joint_ind + 1, 1],
                uvd_pt[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(uvd_pt[[0, joint_ind], 0], uvd_pt[[0, joint_ind], 1], uvd_pt[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lineWidth=line_wd)
        else:
            ax.plot(uvd_pt[[joint_ind - 1, joint_ind], 0], uvd_pt[[joint_ind - 1, joint_ind], 1],
                    uvd_pt[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                    linewidth=line_wd)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    return ax

def plot3D_adjust_points(ax, pos, azim=0, elev=0):
    color_hand_joints = [[1.0, 0.0, 0.0],
                         [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                         [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                         [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                         [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                         [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little
    ax.grid(True)
    # ax.grid(color='r')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.view_init(elev=-85, azim=-75)
    # ax.set_aspect('auto')
    position = np.copy(pos)
    uvd_pt = np.reshape(position, [21, 3])
    x_min, x_max = min(uvd_pt[:, 0]), max(uvd_pt[:, 0])
    y_min, y_max = min(uvd_pt[:, 1]), max(uvd_pt[:, 1])
    z_min, z_max = min(uvd_pt[:, 2]), max(uvd_pt[:, 2])

    center = [(x_min + x_max) / 2,
              (y_min + y_max) / 2,
              (z_min + z_max) / 2]

    # STB
    ax.set_xlim(x_min - 5, x_max + 5)
    ax.set_ylim(y_max + 5, y_min - 5)
    ax.set_zlim(z_min - 8, z_max + 8)
    # ax.set_xlim(center[0] - 70, center[0] + 70)
    # ax.set_ylim(center[1] + 70, center[1] - 70)
    # ax.set_zlim(center[2] - 70, center[2] + 70)
    # ax.dist = 8
    # ax.grid(True)

    marker_sz = 15
    line_wd = 2

    for joint_ind in range(uvd_pt.shape[0]):
        ax.plot(uvd_pt[joint_ind:joint_ind + 1, 0], uvd_pt[joint_ind:joint_ind + 1, 1],
                uvd_pt[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        # if joint_ind == 0:
        #     continue
        # elif joint_ind % 4 == 1:
        #     ax.plot(uvd_pt[[0, joint_ind], 0], uvd_pt[[0, joint_ind], 1], uvd_pt[[0, joint_ind], 2],
        #             color=color_hand_joints[joint_ind], lineWidth=line_wd)
        # else:
        #     ax.plot(uvd_pt[[joint_ind - 1, joint_ind], 0], uvd_pt[[joint_ind - 1, joint_ind], 1],
        #             uvd_pt[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
        #             linewidth=line_wd)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    return ax

def plot3D_adjust_line(ax, pos, azim=0, elev=0):
    color_hand_joints = [[1.0, 0.0, 0.0],
                         [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                         [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                         [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                         [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                         [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little
    ax.grid(True)
    # ax.grid(color='r')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.view_init(elev=-85, azim=-75)
    # ax.set_aspect('auto')
    position = np.copy(pos)
    uvd_pt = np.reshape(position, [21, 3])
    x_min, x_max = min(uvd_pt[:, 0]), max(uvd_pt[:, 0])
    y_min, y_max = min(uvd_pt[:, 1]), max(uvd_pt[:, 1])
    z_min, z_max = min(uvd_pt[:, 2]), max(uvd_pt[:, 2])

    center = [(x_min + x_max) / 2,
              (y_min + y_max) / 2,
              (z_min + z_max) / 2]

    # STB
    ax.set_xlim(x_min - 5, x_max + 5)
    ax.set_ylim(y_max + 5, y_min - 5)
    ax.set_zlim(z_min - 8, z_max + 8)
    # ax.set_xlim(center[0] - 70, center[0] + 70)
    # ax.set_ylim(center[1] + 70, center[1] - 70)
    # ax.set_zlim(center[2] - 70, center[2] + 70)
    # ax.dist = 8
    # ax.grid(True)

    marker_sz = 15
    line_wd = 2

    for joint_ind in range(uvd_pt.shape[0]):
        # ax.plot(uvd_pt[joint_ind:joint_ind + 1, 0], uvd_pt[joint_ind:joint_ind + 1, 1],
        #         uvd_pt[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(uvd_pt[[0, joint_ind], 0], uvd_pt[[0, joint_ind], 1], uvd_pt[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lineWidth=line_wd)
        else:
            ax.plot(uvd_pt[[joint_ind - 1, joint_ind], 0], uvd_pt[[joint_ind - 1, joint_ind], 1],
                    uvd_pt[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                    linewidth=line_wd)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    return ax



def visulization_3d_pose(image, pose_3d):
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.imshow(image)
    ax2 = plot3D(ax2, pose_3d, azim=-75, elev=-105)

    plt.show()
    plt.close()

class MANOHandJoints:
  n_keypoints = 21

  n_joints = 21

  center = 4

  root = 0

  labels = [
    'W', #0
    'I0', 'I1', 'I2', #3
    'M0', 'M1', 'M2', #6
    'L0', 'L1', 'L2', #9
    'R0', 'R1', 'R2', #12
    'T0', 'T1', 'T2', #15
    'I3', 'M3', 'L3', 'R3', 'T3' #20, tips are manually added (not in MANO)
  ]

  # finger tips are not joints in MANO, we label them on the mesh manually
  mesh_mapping = {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}
  # mesh_mapping = {16: 317, 17: 444, 18: 673, 19: 556, 20: 745}

  parents = [
    None,
    0, 1, 2,
    0, 4, 5,
    0, 7, 8,
    0, 10, 11,
    0, 13, 14,
    3, 6, 9, 12, 15
  ]

  end_points = [0, 16, 17, 18, 19, 20]


class MPIIHandJoints:
  n_keypoints = 21

  n_joints = 21

  center = 9

  root = 0

  labels = [
    'W', #0
    'T0', 'T1', 'T2', 'T3', #4
    'I0', 'I1', 'I2', 'I3', #8
    'M0', 'M1', 'M2', 'M3', #12
    'R0', 'R1', 'R2', 'R3', #16
    'L0', 'L1', 'L2', 'L3', #20
  ]

  parents = [
    None,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19
  ]

  colors = [
    (0, 0, 0),
    (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
    (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
    (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255),
    (255, 255, 0), (255, 255, 0), (255, 255, 0), (255, 255, 0),
    (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255),
  ]


def mpii_to_mano(mpii):
  """
  Map data from MPIIHandJoints order to MANOHandJoints order.
  Parameters
  ----------
  mpii : np.ndarray, [21, ...]
    Data in MPIIHandJoints order. Note that the joints are along axis 0.
  Returns
  -------
  np.ndarray
    Data in MANOHandJoints order.
  """
  mano = []
  for j in range(MANOHandJoints.n_joints):
    mano.append(
      mpii[MPIIHandJoints.labels.index(MANOHandJoints.labels[j])]
    )
  mano = np.stack(mano, 0)
  return mano


def mano_to_mpii(mano):
  """
  Map data from MANOHandJoints order to MPIIHandJoints order.
  Parameters
  ----------
  mano : np.ndarray, [21, ...]
    Data in MANOHandJoints order. Note that the joints are along axis 0.
  Returns
  -------
  np.ndarray
    Data in MPIIHandJoints order.
  """
  mpii = []
  for j in range(MPIIHandJoints.n_joints):
    mpii.append(
      mano[MANOHandJoints.labels.index(MPIIHandJoints.labels[j])]
    )
  mpii = np.stack(mpii, 0)
  return mpii


def xyz_to_delta(xyz, joints_def):
  """
  Compute bone orientations from joint coordinates (child joint - parent joint).
  The returned vectors are normalized.
  For the root joint, it will be a zero vector.
  Parameters
  ----------
  xyz : np.ndarray, shape [J, 3]
    Joint coordinates.
  joints_def : object
    An object that defines the kinematic skeleton, e.g. MPIIHandJoints.
  Returns
  -------
  np.ndarray, shape [J, 3]
    The **unit** vectors from each child joint to its parent joint.
    For the root joint, it's are zero vector.
  np.ndarray, shape [J, 1]
    The length of each bone (from child joint to parent joint).
    For the root joint, it's zero.
  """
  delta = []
  for j in range(joints_def.n_joints):
    p = joints_def.parents[j]
    if p is None:
      delta.append(np.zeros(3))
    else:
      delta.append(xyz[j] - xyz[p])
  delta = np.stack(delta, 0)
  lengths = np.linalg.norm(delta, axis=-1, keepdims=True)
  delta /= np.maximum(lengths, np.finfo(xyz.dtype).eps)
  return delta, lengths


def base_transform(img, size, mean=0.5, std=0.5):
    x = cv2.resize(img, (size, size)).astype(np.float32) / 255
    x -= mean
    x /= std
    x = x.transpose(2, 0, 1)

    return x


def inv_base_tranmsform(x, mean=0.5, std=0.5):
    x = x.transpose(1, 2, 0)
    image = (x * std + mean) * 255
    return image.astype(np.uint8)


def crop_roi(img, bbox, out_sz, padding=(0, 0, 0)):
    bbox = [float(x) for x in bbox]
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(img, mapping, (out_sz, out_sz),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=padding)
    return crop


def registration(vertex, uv, j_regressor, K, size, uv_conf=None, poly=None):
    """
    Adaptive 2D-1D registration
    :param vertex: 3D mesh xyz
    :param uv: 2D pose
    :param j_regressor: matrix for vertex -> joint
    :param K: camera parameters
    :param size: image size
    :param uv_conf: 2D pose confidence
    :param poly: contours from silhouette
    :return: camera-space vertex
    """
    t = np.array([0, 0, 0.6])
    bounds = ((None, None), (None, None), (0.3, 2))
    poly_protect = [0.06, 0.02]

    vertex2xyz = np.matmul(j_regressor, vertex)
    if vertex2xyz.shape[0] == 21:
        vertex2xyz = mano_to_mpii(vertex2xyz)
    try_poly = True
    if uv_conf is None:
        uv_conf = np.ones([uv.shape[0], 1])
    uv_select = uv_conf > 0.1
    if uv_select.sum() == 0:
        success = False
    else:
        loss = np.array([5, ])
        attempt = 5
        while loss.mean() > 2 and attempt:
            attempt -= 1
            uv = uv[uv_select.repeat(2, axis=1)].reshape(-1, 2)
            uv_conf = uv_conf[uv_select].reshape(-1, 1)
            vertex2xyz = vertex2xyz[uv_select.repeat(3, axis=1)].reshape(-1, 3)
            sol = minimize(align_uv, t, method='SLSQP', bounds=bounds, args=(uv, vertex2xyz, K))
            t = sol.x
            success = sol.success
            xyz = vertex2xyz + t
            proj = np.matmul(K, xyz.T).T
            uvz = np.concatenate((uv, np.ones([uv.shape[0], 1])), axis=1) * xyz[:, 2:]
            loss = abs((proj - uvz).sum(axis=1))
            uv_select = loss < loss.mean() + loss.std()
            if uv_select.sum() < 13:
                break
            uv_select = uv_select[:, np.newaxis]

    if poly is not None and try_poly:
        poly = find_1Dproj(poly[0]) / size
        sol = minimize(align_poly, np.array([0, 0, 0.6]), method='SLSQP', bounds=bounds, args=(poly, vertex, K, size))
        if sol.success:
            t2 = sol.x
            d = distance(t, t2)
            if d > poly_protect[0]:
                t = t2
            elif d > poly_protect[1]:
                t = t * (1 - (d - poly_protect[1]) / (poly_protect[0] - poly_protect[1])) + t2 * ((d - poly_protect[1]) / (poly_protect[0] - poly_protect[1]))

    return vertex + t, success


def distance(x, y):
    return np.sqrt(((x - y)**2).sum())


def find_1Dproj(points):
    angles = [(0, 90), (-15, 75), (-30, 60), (-45, 45), (-60, 30), (-75, 15)]
    axs = [(np.array([[np.cos(x/180*np.pi), np.sin(x/180*np.pi)]]), np.array([np.cos(y/180*np.pi), np.sin(y/180*np.pi)])) for x, y in angles]
    proj = []
    for ax in axs:
        x = (points * ax[0]).sum(axis=1)
        y = (points * ax[1]).sum(axis=1)
        proj.append([x.min(), x.max(), y.min(), y.max()])

    return np.array(proj)


def align_poly(t, poly, vertex, K, size):
    proj = np.matmul(K, (vertex + t).T).T
    proj = (proj / proj[:, 2:])[:, :2]
    proj = find_1Dproj(proj) / size
    loss = (proj - poly)**2

    return loss.mean()


def align_uv(t, uv, vertex2xyz, K):
    xyz = vertex2xyz + t
    proj = np.matmul(K, xyz.T).T
    uvz = np.concatenate((uv, np.ones([uv.shape[0], 1])), axis=1) * xyz[:, 2:]
    loss = (proj - uvz)**2

    return loss.mean()


def map2uv(map, size=(224, 224)):
    if map.ndim == 4:
        uv = np.zeros((map.shape[0], map.shape[1], 2))
        uv_conf = np.zeros((map.shape[0], map.shape[1], 1))
        map_size = map.shape[2:]
        for j in range(map.shape[0]):
            for i in range(map.shape[1]):
                uv_conf[j][i] = map[j, i].max()
                max_pos = map[j, i].argmax()
                uv[j][i][1] = (max_pos // map_size[1]) / map_size[0] * size[0]
                uv[j][i][0] = (max_pos % map_size[1]) / map_size[1] * size[1]
    else:
        uv = np.zeros((map.shape[0], 2))
        uv_conf = np.zeros((map.shape[0], 1))
        map_size = map.shape[1:]
        for i in range(map.shape[0]):
            uv_conf[i] = map[i].max()
            max_pos = map[i].argmax()
            uv[i][1] = (max_pos // map_size[1]) / map_size[0] * size[0]
            uv[i][0] = (max_pos % map_size[1]) / map_size[1] * size[1]

    return uv, uv_conf


def uv2map(uv, size=(224, 224)):
    kernel_size = (size[0] * 13 // size[0] - 1) // 2
    gaussian_map = np.zeros((uv.shape[0], size[0], size[1]))
    size_transpose = np.array(size)
    gaussian_kernel = cv2.getGaussianKernel(2 * kernel_size + 1, (2 * kernel_size + 2)/4.)
    gaussian_kernel = np.dot(gaussian_kernel, gaussian_kernel.T)
    gaussian_kernel = gaussian_kernel/gaussian_kernel.max()

    for i in range(gaussian_map.shape[0]):
        if (uv[i] >= 0).prod() == 1 and (uv[i][1] <= size_transpose[0]) and (uv[i][0] <= size_transpose[1]):
            s_pt = np.array((uv[i][1], uv[i][0]))
            p_start = s_pt - kernel_size
            p_end = s_pt + kernel_size
            p_start_fix = (p_start >= 0) * p_start + (p_start < 0) * 0
            k_start_fix = (p_start >= 0) * 0 + (p_start < 0) * (-p_start)
            p_end_fix = (p_end <= (size_transpose - 1)) * p_end + (p_end > (size_transpose - 1)) * (size_transpose - 1)
            k_end_fix = (p_end <= (size_transpose - 1)) * kernel_size * 2 + (p_end > (size_transpose - 1)) * (2*kernel_size - (p_end - (size_transpose - 1)))
            gaussian_map[i, p_start_fix[0]: p_end_fix[0] + 1, p_start_fix[1]: p_end_fix[1] + 1] = \
                gaussian_kernel[k_start_fix[0]: k_end_fix[0] + 1, k_start_fix[1]: k_end_fix[1] + 1]

    return gaussian_map


def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area


def tensor2array(tensor, max_value=None, colormap='jet', channel_first=True, mean=0.5, std=0.5):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            color_cvt = cv2.COLOR_BGR2RGB
            if colormap == 'jet':
                colormap = cv2.COLORMAP_JET
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255 * tensor.squeeze().numpy() / max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy() / max_value).clip(0, 1)
        if channel_first:
            array = array.transpose(2, 0, 1)
    elif tensor.ndimension() == 3:
        assert (tensor.size(0) == 3)
        array = ((mean + tensor.numpy() * std) * 255).astype(np.uint8)
        if not channel_first:
            array = array.transpose(1, 2, 0)

    return array