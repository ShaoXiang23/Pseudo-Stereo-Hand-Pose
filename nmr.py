import numpy as np
import os
import torch
import torch.nn as nn
import neural_renderer as nr

from utils.manopth.manopth.manolayer import ManoLayer_PCA

class NMR(nn.Module):
    def __init__(self, image_size=256):
        super(NMR, self).__init__()

        self.image_size = image_size
        self.renderer = nr.Renderer(
            image_size=self.image_size,
            camera_mode='look',
            perspective=False
        )
        self.manolayer = ManoLayer_PCA(
            use_pca=True,
            ncomps=45,
            flat_hand_mean=False
        )

    def forward(self, verts, faces, depth=True):
        bz = verts.shape[0]

        ''' coordinate mismatch, invert y axis '''
        verts[:, :, 1] = -verts[:, :, 1]

        faces = faces.repeat(bz, 1, 1).int()

        silhouettes = self.renderer(verts, faces, mode='silhouettes')
        if depth:
            depth_image = self.renderer(verts, faces, mode='depth')
            return silhouettes, depth_image
        else:
            return silhouettes

