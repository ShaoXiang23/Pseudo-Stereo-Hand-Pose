import numpy as np
import torch

def gen_cam_param(joint, kp2d, mode='ortho', root_index=0):
    if mode in ['persp', 'perspective']:
        kp2d = kp2d.reshape(-1)[:, np.newaxis]  # (42, 1)
        joint = joint / joint[:, 2:]
        joint = joint[:, :2]
        jM = np.zeros((42, 2), dtype="float32")
        for i in range(joint.shape[0]):  # 21
            jM[2 * i][0] = joint[i][0]
            jM[2 * i + 1][1] = joint[i][1]
        pad2 = np.array(range(42))
        pad2 = (pad2 % 2)[:, np.newaxis]
        pad1 = (1 - pad2)

        jM = np.concatenate([jM, pad1, pad2], axis=1)  # (42, 4)
        jMT = jM.transpose()  # (4, 42)
        jMTjM = np.matmul(jMT, jM)  # (4,4)
        jMTb = np.matmul(jMT, kp2d)
        cam_param = np.matmul(np.linalg.inv(jMTjM), jMTb)
        cam_param = cam_param.reshape(-1)
        return cam_param
    elif mode in ['ortho', 'orthogonal']:
        # ortho only when
        assert np.sum(np.abs(joint[root_index, :])) == 0
        joint = joint[:, :2]  # (21, 2)
        joint = joint.reshape(-1)[:, np.newaxis]
        kp2d = kp2d.reshape(-1)[:, np.newaxis]
        pad2 = np.array(range(42))
        pad2 = (pad2 % 2)[:, np.newaxis]
        pad1 = (1 - pad2)
        jM = np.concatenate([joint, pad1, pad2], axis=1)  # (42, 3)
        jMT = jM.transpose()  # (3, 42)
        jMTjM = np.matmul(jMT, jM)
        jMTb = np.matmul(jMT, kp2d)
        cam_param = np.matmul(np.linalg.inv(jMTjM), jMTb)
        cam_param = cam_param.reshape(-1)
        return cam_param
    else:
        raise Exception("Unkonwn mode type. should in ['persp', 'orth']")

def gen_cam_param_ortho_11(joint, kp2d, root_index=0):
    # ortho only when
    assert np.sum(np.abs(joint[root_index, :])) == 0
    joint = joint[:, :2]  # (11, 2)
    joint = joint.reshape(-1)[:, np.newaxis]
    kp2d = kp2d.reshape(-1)[:, np.newaxis]
    pad2 = np.array(range(22))
    pad2 = (pad2 % 2)[:, np.newaxis]
    pad1 = (1 - pad2)
    jM = np.concatenate([joint, pad1, pad2], axis=1)  # (22, 3)
    jMT = jM.transpose()  # (3, 22)
    jMTjM = np.matmul(jMT, jM)
    jMTb = np.matmul(jMT, kp2d)
    cam_param = np.matmul(np.linalg.inv(jMTjM), jMTb)
    cam_param = cam_param.reshape(-1)
    return cam_param

def  batch_orth_proj(X, camera):
    '''
        X is N x num_points x 3
        camera: [B, 3]
    '''
    camera = camera.reshape(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape

    return (camera[:, :, 0] * X_trans.reshape(shape[0], -1)).reshape(shape)

def orthographic_proj_withz(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, tz]
    Orth preserving the z.
    """

    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:].contiguous().view(cam.size(0), 1, -1)

    proj = scale * X

    proj_xy = proj[:, :, :2] + trans[:, :, :2] # [B, 21, 2]
    # proj_z = proj[:, :, 2] + trans[:, :, 2] + offset_z
    # proj_z = proj_z[:, :, None]  # unsqueeze last dimension effect

    # return torch.cat((proj_xy, proj_z), 2)
    return proj_xy

if __name__ == '__main__':
    SNAP_PARENT = [
        0,  # 0's parent
        0,  # 1's parent
        1,
        2,
        3,
        0,  # 5's parent
        5,
        6,
        7,
        0,  # 9's parent
        9,
        10,
        11,
        0,  # 13's parent
        13,
        14,
        15,
        0,  # 17's parent
        17,
        18,
        19,
    ]

    from data.handataset import HandDataset
    from torch.utils.data import DataLoader
    from vis import draw_2Dimg
    from imgutils import tensor_to_image

    index = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20]
    # index = [0, 4, 8, 12, 16, 20]

    root_index = 9
    val_dataset = HandDataset(
        data_split='train',
        train=True,
        subset_name=['stb', 'rhd'],
        data_root='/home/robot/gsx/bihand/data',
        sigma=2.0,
        inp_res=256, hm_res=64, dep_res=64, njoints=21,
        joint_root_idx=root_index
    )

    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=8)

    for i, metas in enumerate(val_loader):
        img = metas['clr']
        img_np = tensor_to_image(img)

        kp2d_gt = metas['kp2d']
        joint_R = metas['jointR']
        cam_ortho = metas['cam_ortho']

        # kp2d, kp3d = np.asarray(kp2d_gt[0]), np.asarray(joint_R[0] * 1000)
        # kp2d_chain = [
        #     kp2d[i] - kp2d[SNAP_PARENT[i]]
        #     for i in range(21)
        # ]
        # kin_chain_2d = np.array(kp2d_chain[1:])
        # # bone_2d_mean = np.mean(np.sqrt(kin_chain_2d[:, 0] ** 2 + kin_chain_2d[:, 1] ** 2))
        # # bone_2d_mean = np.mean(np.sort(np.sqrt(kin_chain_2d[:, 0] ** 2 + kin_chain_2d[:, 1] ** 2))[7:])
        # bone_2d_mean = np.mean(np.sqrt(kin_chain_2d[:, 0] ** 2 + kin_chain_2d[:, 1] ** 2))
        #
        # kp3d_chain = [
        #     kp3d[i] - kp3d[SNAP_PARENT[i]]
        #     for i in range(21)
        # ]
        # kin_chain_3d = np.array(kp3d_chain[1:])
        # # bone_3d_mean = np.mean((np.sqrt(
        # #     kin_chain_3d[:, 0] ** 2 + kin_chain_3d[:, 1] ** 2 + kin_chain_3d[:, 2] ** 2)))
        # # bone_3d_mean = np.mean(np.sort(np.sqrt(
        # #     kin_chain_3d[:, 0] ** 2 + kin_chain_3d[:, 1] ** 2))[7:])
        # bone_3d_mean = np.mean(np.sqrt(kin_chain_3d[:, 0] ** 2 + kin_chain_3d[:, 1] ** 2))
        # scale = bone_2d_mean / bone_3d_mean
        # offset_xy = (kp2d[:, :2] / scale) - kp3d[:, :2]
        # offset_x, offset_y = np.mean(offset_xy[:, 0]), np.mean(offset_xy[:, 1])
        # cam_param = np.asarray([scale, offset_x, offset_y])
        #
        # cam_param = torch.from_numpy(cam_param)[None, :]
        # print(cam_param)
        # # cam_param = torch.cat([cam_param[:, 0], cam_param[:, 2], cam_param[:, 1]])
        # r_kp2d = batch_orth_proj(torch.from_numpy(kp3d)[None, :, :], cam_param)

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.imshow(img_np[0])
        img_kp2d = draw_2Dimg(kps_2d=kp2d_gt[0], image=img_np[0])
        ax2.imshow(img_kp2d)

        # cam_ortho[:, 1:] = torch.ones_like(cam_ortho[:, 1:]) * 128. + cam_ortho[:, 1:]
        r_kp2d = orthographic_proj_withz(joint_R, cam_ortho, offset_z=0.)
        # r_kp2d = batch_orth_proj(joint_R, cam_ortho)
        print(joint_R[0])
        print(cam_ortho[0])
        print(r_kp2d[0])

        img_kp2d_ = draw_2Dimg(kps_2d=r_kp2d[0], image=img_np[0])
        ax3.imshow(img_kp2d_)

        import time
        time.sleep(0.1)

        plt.show()
        exit()
