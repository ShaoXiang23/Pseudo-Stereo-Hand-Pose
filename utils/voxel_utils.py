import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def discretize_0_1(coord, cropped_size):
    min_normalized = 0.
    max_normalized = 1.
    scale = (max_normalized - min_normalized) / cropped_size
    return (coord - min_normalized) / scale

def scattering(coord, cropped_size):
    # coord: [0, cropped_size]
    # Assign range[0, 1) -> 0, [1, 2) -> 1, .. [cropped_size-1, cropped_size) -> cropped_size-1
    # That is, around center 0.5 -> 0, around center 1.5 -> 1 .. around center cropped_size-0.5 -> cropped_size-1
    coord = coord.astype(np.int32)

    mask = (coord[:, 0] >= 0) & (coord[:, 0] < cropped_size) & \
           (coord[:, 1] >= 0) & (coord[:, 1] < cropped_size) & \
           (coord[:, 2] >= 0) & (coord[:, 2] < cropped_size)

    coord = coord[mask, :]

    cubic = np.zeros((cropped_size, cropped_size, cropped_size))

    # Note, directly map point coordinate (x, y, z) to index (i, j, k), instead of (k, j, i)
    # Need to be consistent with heatmap generating and coordinates extration from heatmap
    cubic[coord[:, 0], coord[:, 1], coord[:, 2]] = 1

    return cubic

def scattering_tsr(coord, cropped_size):
    # coord: [0, cropped_size]
    # Assign range[0, 1) -> 0, [1, 2) -> 1, .. [cropped_size-1, cropped_size) -> cropped_size-1
    # That is, around center 0.5 -> 0, around center 1.5 -> 1 .. around center cropped_size-0.5 -> cropped_size-1
    # coord = coord.astype(np.int32)
    coord = coord.long()

    mask = (coord[:, 0] >= 0) & (coord[:, 0] < cropped_size) & \
           (coord[:, 1] >= 0) & (coord[:, 1] < cropped_size) & \
           (coord[:, 2] >= 0) & (coord[:, 2] < cropped_size)

    coord = coord[mask, :]

    cubic = torch.zeros((cropped_size, cropped_size, cropped_size)).to(device)

    # Note, directly map point coordinate (x, y, z) to index (i, j, k), instead of (k, j, i)
    # Need to be consistent with heatmap generating and coordinates extration from heatmap
    cubic[coord[:, 1], coord[:, 0], coord[:, 2]] = 1

    return cubic

def putmask_le(x, t1, t2):
    m = x.le(t1)
    t2 = t2 * torch.ones_like(x).to(x.device)
    m_ = ~m
    return t2 * m + x * m_

def dep_to_voxel(dep_norm_tsr):
    h, w = dep_norm_tsr.shape

    xx, yy = torch.meshgrid(torch.arange(w) + 1, torch.arange(h) + 1)
    points = torch.zeros((h, w, 3))
    points[:, :, 0], points[:, :, 1], points[:, :, 2] = xx, yy, dep_norm_tsr
    points = points.reshape((-1, 3))
    points[:, 2] = points[:, 2] * 50.  # meter -> mm
    points = putmask_le(points, 0, 101)

    cubic_size, cropped_size = 64, 64
    points = points / cubic_size  # xy -> [-0.32, 0.32] z -> [0, 1]
    coords = discretize_0_1(points, cropped_size)
    cubics = scattering_tsr(coords, cropped_size)

    return cubics

def batch_dep_to_voxel(dep_norm_tsr):
    dep_norm_tsr = torch.squeeze(dep_norm_tsr)
    bs, h, w = dep_norm_tsr.shape

    voxels = torch.zeros([bs, 64, 64, 64]).to(device)

    # Using ugly loop to equal mini-batch strategy
    for i in range(bs):
        xx, yy = torch.meshgrid(torch.arange(w) + 1, torch.arange(h) + 1)
        points = torch.zeros((h, w, 3)).to(device)
        points[:, :, 0], points[:, :, 1], points[:, :, 2] = xx, yy, dep_norm_tsr[i]
        points = points.reshape((-1, 3))
        points[:, 2] = points[:, 2] * 50.  # meter -> mm
        points = putmask_le(points, 0, 101)

        cubic_size, cropped_size = 64, 64
        points = points / cubic_size  # xy -> [-0.32, 0.32] z -> [0, 1]
        coords = discretize_0_1(points, cropped_size)
        cubics = scattering_tsr(coords, cropped_size)

        voxels[i] = cubics

    return voxels