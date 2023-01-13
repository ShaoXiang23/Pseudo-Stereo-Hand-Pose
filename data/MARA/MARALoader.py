from data.MARA.MARAHand import MARAHandDataset
from data.MARA.v2v_utils import V2VVoxelization
from torch.utils.data import Dataset, DataLoader
import torch

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

keypoints_num = 21
cubic_size = 200
batch_size = 1

voxelization_train = V2VVoxelization(cubic_size=200, augmentation=True)
voxelization_val = V2VVoxelization(cubic_size=200, augmentation=False)

def transform_train(sample):
    points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)
    input, heatmap = voxelization_train({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})
    return (torch.from_numpy(input), torch.from_numpy(heatmap))

def transform_val(sample):
    points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)
    input, heatmap = voxelization_val({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})
    return (torch.from_numpy(input), torch.from_numpy(heatmap))

def transform_val_copy(sample):
    points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)
    # input, heatmap = voxelization_val({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})
    # return (torch.from_numpy(input), torch.from_numpy(heatmap))
    return (torch.from_numpy(points), torch.from_numpy(keypoints), torch.from_numpy(refpoint))

def main():
    print('==> Preparing data ..')

    data_dir = "/home/robot/gsx/MSRA/cvpr15_MSRAHandGestureDB"
    center_dir = "/home/robot/gsx/MSRA/center"
    test_subject_id = 3

    train_set = MARAHandDataset(data_dir, center_dir, 'train', test_subject_id, transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6)

    val_set = MARAHandDataset(data_dir, center_dir, 'test', test_subject_id, transform_val_copy)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=6)

    # loop the loader
    for i, metas in enumerate(val_loader):

        points, joints, refpoint = metas[0], metas[1], metas[-1]
        print(points.shape, joints.shape, refpoint.shape)

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(131, projection='3d')
        this_points = points[0]
        ax1.scatter(this_points[:, 0], this_points[:, 1], this_points[:, 2])
        ax1.dist = 8.0

        ax2 = fig.add_subplot(132, projection='3d')
        this_joints = joints[0]
        this_refpoint = refpoint[0]
        ax2.scatter(this_joints[:, 0], this_joints[:, 1], this_joints[:, 2], color='b')
        ax2.scatter(this_refpoint[0], this_refpoint[1], this_refpoint[2], color='r')
        ax2.dist = 8.0
        ax2.view_init(-90, -90)

        voxels, heatmap, normalize_coord = voxelization_val({'points': points[0].numpy(),
                          'keypoints': joints[0].numpy(),
                          'refpoint': refpoint[0].numpy()})

        # print("points: ", points[0])
        # print("refpoint: ", refpoint[0])
        # print("joint: ", joints[0])

        ax3 = fig.add_subplot(133, projection='3d')
        # ax3.scatter(normalize_coord[:, 0], normalize_coord[:, 1], normalize_coord[:, 2], color='b')
        ax3.voxels(voxels[0])
        ax3.view_init(-90, -90)

        plt.show()
        exit()

if __name__ == '__main__':
    main()