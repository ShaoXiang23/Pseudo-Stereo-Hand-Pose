# Copyright (c) Lixin YANG. All Rights Reserved.
r"""
Hand dataset controll all sub dataset
"""

import torch
import torch.utils.data
import os
from PIL import Image, ImageFilter
import numpy as np
import cv2
import random

import utils.imgutils as imutils
import utils.handutils as handutils
import utils.heatmaputils as hmutils
import utils.func as func
import utils.config as cfg

snap_joint_name2id = {w: i for i, w in enumerate(cfg.snap_joint_names)}

class HandDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_split='train',
            sigma=2.0,
            inp_res=256,
            hm_res=64,
            dep_res=64,
            njoints=21,
            train=True,
            scale_jittering=0.1,
            center_jettering=0.1,
            max_rot=np.pi,
            hue=0.15,
            saturation=0.5,
            contrast=0.5,
            brightness=0.5,
            blur_radius=0.5,
    ):

        self.inp_res = inp_res  # 256 # network input resolution
        self.hm_res = hm_res  # 64  # out hm resolution
        self.dep_res = dep_res  # 64 # out depthmap resolution
        self.njoints = njoints  # total 21 hand parts
        self.sigma = sigma
        self.max_rot = max_rot

        self.parents = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
        self.childs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        # Training attributes
        self.train = train
        self.scale_jittering = scale_jittering
        self.center_jittering = center_jettering

        # Color jitter attributes
        self.hue = hue
        self.contrast = contrast
        self.brightness = brightness
        self.saturation = saturation
        self.blur_radius = blur_radius

        self.datasets = []
        self.ref_bone_link = (0, 9)  # mid mcp
        self.joint_root_idx = 9  # root


        # if 'rhd' in subset_name:
        #     self.rhd = RHDDataset(
        #         data_root=os.path.join(data_root, 'RHD/RHD_published_v2'),
        #         data_split=data_split,
        #         hand_side=hand_side,
        #         njoints=njoints,
        #     )
        #     print(self.rhd)
        #     self.datasets.append(self.rhd)
        from data.FPHA.fhbhands_guo import FHBHands
        self.fhb = FHBHands(data_split)

        self.datasets.append(self.fhb)

        self.total_data = 0
        for ds in self.datasets:
            self.total_data += len(ds)

    def __getitem__(self, index):
        rng = np.random.RandomState(seed=random.randint(0, 1024))
        sample, ds = None, None

        try:
            sample, ds = self._get_sample(index)
        except Exception:
            index = np.random.randint(0, len(self))
            sample, ds = self._get_sample(index)

        clr = sample['clr']
        # dep = sample['dep_norm']
        center = sample['center']
        scale = sample['scale']
        # center_dep = sample['center_dep']
        # scale_dep = sample['scale_dep']
        intr = np.array(
            [
                [1395.749023, 0, 935.732544],
                        [0, 1395.749268, 540.681030],
                        [0, 0, 1]
            ]
        )
        dep_intr = np.array(
            [
                [475.065948, 0, 315.944855],
                [0, 475.065857, 245.287079],
                [0, 0, 1],
            ]
        )


        # Data augmentation
        if self.train:
            ### for rgb ###
            center_offsets = (
                    self.center_jittering
                    * scale
                    * rng.uniform(low=-1, high=1, size=2)
            )
            center = center + center_offsets.astype(int)

            # Scale jittering
            scale_jittering = self.scale_jittering * rng.randn() + 1
            scale_jittering = np.clip(
                scale_jittering,
                1 - self.scale_jittering,
                1 + self.scale_jittering,
            )
            scale = scale * scale_jittering
            rot = rng.uniform(low=-self.max_rot, high=self.max_rot)
            ### ### ### ###
        else:
            rot = 0
        rot_mat = np.array([
            [np.cos(rot), -np.sin(rot), 0],
            [np.sin(rot), np.cos(rot), 0],
            [0, 0, 1],
        ]).astype(np.float32)

        affinetrans, post_rot_trans = handutils.get_affine_transform(
            center=center,
            scale=scale,
            optical_center=[intr[0, 2], intr[1, 2]],  # (cx, cy)print(intr[0, 0])
            out_res=[self.inp_res, self.inp_res],
            rot=rot
        )

        # affinetrans_dep, post_rot_trans_dep = handutils.get_affine_transform(
        #     center=center_dep,
        #     scale=scale_dep,
        #     optical_center=[dep_intr[0, 2], dep_intr[1, 2]],  # (cx, cy)print(intr[0, 0])
        #     out_res=[self.inp_res, self.inp_res],
        #     rot=rot
        # )

        ''' prepare kp2d '''
        kp2d = sample['kp2d']
        kp2d = handutils.transform_coords(kp2d, affinetrans)

        ''' prepare joint '''
        joint = sample['joint'] / 1000.
        if self.train:
            joint = rot_mat.dot(
                joint.transpose(1, 0)
            ).transpose()

        joint_bone = 0
        for jid, nextjid in zip(self.ref_bone_link[:-1], self.ref_bone_link[1:]):
            joint_bone += np.linalg.norm(joint[nextjid] - joint[jid])

        joint_root = joint[self.joint_root_idx]
        joint_bone = np.atleast_1d(joint_bone)
        jointR = joint - joint_root[np.newaxis, :]
        jointRS = jointR / joint_bone

        kin_chain = [
            jointRS[i] - jointRS[cfg.SNAP_PARENT[i]]
            for i in range(21)
        ]
        kin_chain = np.array(kin_chain[1:])  # id 0's parent is itself
        kin_len = np.linalg.norm(
            kin_chain, ord=2, axis=-1, keepdims=True
        )
        kin_chain = kin_chain / kin_len

        ''' prepare intr '''
        new_intr = post_rot_trans.dot(intr)

        ''' prepare clr image '''
        if self.train:
            blur_radius = random.random() * self.blur_radius
            clr = clr.filter(ImageFilter.GaussianBlur(blur_radius))
            clr = imutils.color_jitter(
                clr,
                brightness=self.brightness,
                saturation=self.saturation,
                hue=self.hue,
                contrast=self.contrast,
            )
        # Transform and crop
        clr = handutils.transform_img(
            clr, affinetrans, [self.inp_res, self.inp_res]
        )
        clr = clr.crop((0, 0, self.inp_res, self.inp_res))

        ''' implicit HWC -> CHW, 255 -> 1 '''
        clr = func.to_tensor(clr).float()
        ''' 0-mean, 1 std,  [0,1] -> [-0.5, 0.5] '''
        clr = func.normalize(clr, [0.5, 0.5, 0.5], [1, 1, 1])

        # ''' prepare dep image if has '''
        # if dep is not None:
        #     dep = Image.fromarray(dep)
        #     dep = handutils.transform_img(
        #         dep, affinetrans_dep, [self.inp_res, self.inp_res]
        #     )
        #     dep = dep.crop((0, 0, self.inp_res, self.inp_res))
        #     ''' to float array '''
        #     dep = np.asarray(dep).astype('float32')
        #     dep = cv2.resize(dep, (self.dep_res, self.dep_res))
        # else:
        #     dep = np.zeros(
        #         (self.dep_res, self.dep_res)
        #     ).astype('float32')

        # ''' prepare mask '''
        # mask = dep.copy()
        # np.putmask(mask, mask > 1e-2, 1.0)
        # np.putmask(mask, mask <= 1e-2, 0.0)

        ''' Generate GT Gussian hm and hm veil '''
        hm = np.zeros(
            (self.njoints, self.hm_res, self.hm_res),
            dtype='float32'
        )  # (CHW)
        hm_veil = np.ones(self.njoints, dtype='float32')
        for i in range(self.njoints):
            kp = (
                    (kp2d[i] / self.inp_res) * self.hm_res
            ).astype(np.int32)  # kp uv: [0~256] -> [0~64]
            hm[i], aval = hmutils.gen_heatmap(hm[i], kp, self.sigma)
            hm_veil[i] *= aval

        # ''' Generate GT Gussian hm_bone and hm veil '''
        # kp_bone = 0.5 * (kp2d[self.childs, :] + kp2d[self.parents, :])
        # hm_bone = np.zeros(
        #     (self.njoints - 1, self.hm_res, self.hm_res),
        #     dtype='float32'
        # )  # (CHW)
        # for i in range(self.njoints - 1):
        #     kb = (
        #             (kp_bone[i] / self.inp_res) * self.hm_res
        #     ).astype(np.int32)  # kp uv: [0~256] -> [0~64]
        #     hm_bone[i], aval = hmutils.gen_heatmap(hm_bone[i], kb, self.sigma + 0.5)
        ''' Generate GT Gussian hm_bone and hm veil '''
        kp_bone = 0.5 * (kp2d[self.childs, :] + kp2d[self.parents, :])
        # print(kp_bone.shape, np.expand_dims(kp2d[0], 0).shape)
        kp_bone = np.concatenate((np.expand_dims(kp2d[0], 0), kp_bone))
        hm_bone = np.zeros(
            (self.njoints, self.hm_res, self.hm_res),
            dtype='float32'
        )  # (CHW)
        bm_veil = np.ones(self.njoints, dtype='float32')
        for i in range(self.njoints):
            kb = (
                    (kp_bone[i] / self.inp_res) * self.hm_res
            ).astype(np.int32)  # kp uv: [0~256] -> [0~64]
            hm_bone[i], aval = hmutils.gen_heatmap(hm_bone[i], kb, self.sigma + 0.5)
            bm_veil[i] *= aval

        ''' prepare veil to selected zeros invalid item'''
        # dep_veil = np.array([valid_dep], dtype='float32')

        ## to torch tensor
        clr = clr
        # dep = torch.from_numpy(dep).float()
        # dep256 = torch.from_numpy(dep256).float()
        # mask = torch.from_numpy(mask).float()
        hm = torch.from_numpy(hm).float()
        hm_veil = torch.from_numpy(hm_veil).float()
        bm = torch.from_numpy(hm_bone).float()
        # intr = torch.from_numpy(new_intr).float()
        kp2d = torch.from_numpy(kp2d).float()
        joint = torch.from_numpy(joint).float()
        jointR = torch.from_numpy(jointR).float()
        jointRS = torch.from_numpy(jointRS).float()
        kin_chain = torch.from_numpy(kin_chain).float()
        kin_len = torch.from_numpy(kin_len).float()
        joint_root = torch.from_numpy(joint_root).float()
        joint_bone = torch.from_numpy(joint_bone).float()
        # dep_veil = torch.from_numpy(dep_veil).float()

        ## Meta info
        """
        index torch.Size([16])
        hms_mask torch.Size([16, 21])
        kp2d torch.Size([16, 21, 2])
        bbx torch.Size([16, 4])
        cam_param torch.Size([16, 4])
        joint torch.Size([16, 21, 3])
        joint_root torch.Size([16, 3])
        joint_bone torch.Size([16, 1])
        dep_mask torch.Size([16, 1])
        """
        metas = {
            'index': index,
            'clr': clr,
            # 'dep': dep,
            # 'dep256': dep256,
            # 'mask': mask,
            'hm': hm,
            'hm_veil': hm_veil,
            'bm': bm,
            'kp2d': kp2d,
            'intr': new_intr,
            'joint': joint,
            'jointR': jointR,
            'jointRS': jointRS,
            'kin_chain': kin_chain,
            'kin_len': kin_len,
            'joint_root': joint_root,
            'joint_bone': joint_bone,
            # 'dep_veil': dep_veil
        }

        return metas

    def _get_sample(self, index):
        base = 0
        dataset = None
        for ds in self.datasets:
            if index < base + len(ds):
                sample = ds.get_sample(index - base)
                dataset = ds
                break
            else:
                base += len(ds)
        return sample, dataset

    def __len__(self):
        return self.total_data

def show_sample():
    from torch.utils.data import DataLoader
    import time

    train_dataset = HandDataset(
        data_split='train', train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)

    for i, metas in enumerate(train_loader):
        clr = metas['clr']
        kp2d = metas['kp2d']
        kp3d = metas['joint'] * 1000.
        hm_gt = metas['hm']
        dep_gt = metas['dep']

        import matplotlib.pyplot as plt
        figure = plt.figure(figsize=(12, 3))
        ax1 = figure.add_subplot(151)
        clr = func.batch_denormalize(
            clr, [0.5, 0.5, 0.5], [1, 1, 1]
        )
        clr = func.bchw_2_bhwc(clr)
        clr = func.to_numpy(clr.detach().cpu())
        if clr.dtype is not np.uint8:
            clr = (clr * 255).astype(np.uint8)
        img_np = clr[0].astype(np.uint8)  # img_np
        ax1.imshow(img_np)

        from utils.vis import draw_2Dimg
        ax2 = figure.add_subplot(152)
        img_2d = draw_2Dimg(kp2d[0], img_np)
        ax2.imshow(img_2d)

        ax3 = figure.add_subplot(153)
        hm_gt = torch.sum(hm_gt, dim=1)
        hm_gt = hm_gt[0].numpy()
        ax3.imshow(hm_gt, cmap='jet')

        ax4 = figure.add_subplot(154)
        dep_gt = dep_gt[0].numpy()
        ax4.imshow(dep_gt, cmap='gray')

        from utils.vis import plot3D
        ax5 = figure.add_subplot(155, projection='3d')
        root = kp3d[:, 0]
        kp3d_gt = kp3d - root[:, None, :]
        plot3D(ax5, kp3d_gt, -90, -90, 15, 2)

        plt.show()

        time.sleep(0.5)



if __name__ == '__main__':
    show_sample()
#     rotation_matrix_test()
