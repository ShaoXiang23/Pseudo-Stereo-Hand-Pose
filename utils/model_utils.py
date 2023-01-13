import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored, cprint

# from bihand.models.bases.bottleneck import BottleneckBlock
class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BottleneckBlock, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes

        mid_planes = (out_planes // 2 ) if out_planes >= in_planes else in_planes // 2
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, bias=True)

        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=1, padding=1, bias=True)

        self.bn3 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if in_planes != out_planes:
            self.conv4 = nn.Conv2d(in_planes, out_planes, bias=True, kernel_size=1)


    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.in_planes != self.out_planes:
            residual = self.conv4(x)

        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, block, nblocks, in_planes, depth=4):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.hg = self._make_hourglass(block, nblocks, in_planes, depth)

    def _make_hourglass(self, block, nblocks, in_planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, nblocks, in_planes))
            if i == 0:
                res.append(self._make_residual(block, nblocks, in_planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _make_residual(self, block, nblocks, in_planes):
        layers = []
        for i in range(0, nblocks):
            layers.append(block(in_planes, in_planes))
        return nn.Sequential(*layers)

    def _hourglass_foward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hourglass_foward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hourglass_foward(self.depth, x)


class HourglassBisected(nn.Module):
    def __init__(
            self,
            block,
            nblocks,
            in_planes,
            depth=4
    ):
        super(HourglassBisected, self).__init__()
        self.depth = depth
        self.hg = self._make_hourglass(block, nblocks, in_planes, depth)

    def _make_hourglass(self, block, nblocks, in_planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                _res = []
                if j == 1:
                    _res.append(self._make_residual(block, nblocks, in_planes))
                else:
                    _res.append(self._make_residual(block, nblocks, in_planes))
                    _res.append(self._make_residual(block, nblocks, in_planes))

                res.append(nn.ModuleList(_res))

            if i == 0:
                _res = []
                _res.append(self._make_residual(block, nblocks, in_planes))
                _res.append(self._make_residual(block, nblocks, in_planes))
                res.append(nn.ModuleList(_res))

            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _make_residual(self, block, nblocks, in_planes):
        layers = []
        for i in range(0, nblocks):
            layers.append(block(in_planes, in_planes))
        return nn.Sequential(*layers)

    def _hourglass_foward(self, n, x):
        up1_1 = self.hg[n - 1][0][0](x)
        up1_2 = self.hg[n - 1][0][1](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1][0](low1)

        if n > 1:
            low2_1, low2_2, latent = self._hourglass_foward(n - 1, low1)
        else:
            latent = low1
            low2_1 = self.hg[n - 1][3][0](low1)
            low2_2 = self.hg[n - 1][3][1](low1)

        low3_1 = self.hg[n - 1][2][0](low2_1)
        low3_2 = self.hg[n - 1][2][1](low2_2)

        up2_1 = F.interpolate(low3_1, scale_factor=2)
        up2_2 = F.interpolate(low3_2, scale_factor=2)
        out_1 = up1_1 + up2_1
        out_2 = up1_2 + up2_2

        return out_1, out_2, latent

    def forward(self, x):
        return self._hourglass_foward(self.depth, x)

class _Residual_Module(nn.Module):
    def __init__(self, numIn, numOut):
        super(_Residual_Module, self).__init__()
        self.numIn, self.numOut = numIn, numOut
        self.bn = nn.BatchNorm2d(self.numIn)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias=True, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.numOut // 2)
        self.conv2 = nn.Conv2d(self.numOut // 2, self.numOut // 2, bias=True, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.numOut // 2)
        self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias=True, kernel_size=1)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)

        return out + residual

class Hourglass_LiuhaoGe_CVPR2019(nn.Module):
    def __init__(self, n, nModules, nFeats):
        super(Hourglass_LiuhaoGe_CVPR2019, self).__init__()
        self.n, self.nModules, self.nFeats= n, nModules, nFeats

        _up1_, _low1_, _low2_, _low3_ = [], [], [], []
        for j in range(self.nModules):
            _up1_.append(_Residual_Module(self.nFeats, self.nFeats))
        self.low1 = nn.MaxPool2d(kernel_size=2, stride=2)
        for j in range(self.nModules):
            _low1_.append(_Residual_Module(self.nFeats, self.nFeats))
        # 似乎是递归的写法
        if self.n > 1:
            self.low2 = Hourglass_LiuhaoGe_CVPR2019(n - 1, self.nModules, self.nFeats)
        else:
            for j in range(self.nModules):
                _low2_.append(_Residual_Module(self.nFeats, self.nFeats))
            self.low2_ = nn.ModuleList(_low2_)

        for j in range(self.nModules):
            _low3_.append(_Residual_Module(self.nFeats, self.nFeats))

        self.up1_ = nn.ModuleList(_up1_)
        self.low1_ = nn.ModuleList(_low1_)
        self.low3_ = nn.ModuleList(_low3_)
    def forward(self, x):
        up1 = x
        for j in range(self.nModules):
            up1 = self.up1_[j](up1)

        low1 = self.low1(x)
        for j in range(self.nModules):
            low1 = self.low1_[j](low1)

        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for j in range(self.nModules):
                low2 = self.low2_[j](low2)

        low3 = low2
        for j in range(self.nModules):
            low3 = self.low3_[j](low3)
        # up2 = self.up2(low3)
        up2 = nn.functional.interpolate(low3,scale_factor=2)
        return up1 + up2


class Net_HG(nn.Module):
#     _Ge Liuhao_CVPR2019
    def __init__(self, num_joints, num_stages=2, num_modules=2, num_feats=256):
        super(Net_HG, self).__init__()

        self.numOutput, self.nStack, self.nModules, self.nFeats = num_joints, num_stages, num_modules, num_feats

        self.conv1_ = nn.Conv2d(3, 64, bias=True, kernel_size=7, stride=2, padding=3) ### 256 --> 128
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.r1 = _Residual_Module(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.r4 = _Residual_Module(128, 128)
        self.r5 = _Residual_Module(128, self.nFeats)

        _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_, _reg_ = [], [], [], [], [], [], []
        for i in range(self.nStack):
            _hourglass.append(Hourglass_LiuhaoGe_CVPR2019(4, self.nModules, self.nFeats))
            for j in range(self.nModules):
                _Residual.append(_Residual_Module(self.nFeats, self.nFeats))
            lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1),
                                nn.BatchNorm2d(self.nFeats), self.relu)
            _lin_.append(lin)
            _tmpOut.append(nn.Conv2d(self.nFeats, self.numOutput, bias=True, kernel_size=1, stride=1))
            if i < self.nStack - 1:
                _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1))
                _tmpOut_.append(nn.Conv2d(self.numOutput, self.nFeats, bias=True, kernel_size=1, stride=1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin_ = nn.ModuleList(_lin_)
        self.tmpOut = nn.ModuleList(_tmpOut)
        self.ll_ = nn.ModuleList(_ll_)
        self.tmpOut_ = nn.ModuleList(_tmpOut_)

    def forward(self, x):
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.r1(x)
        x = self.maxpool(x)
        x = self.r4(x)
        x = self.r5(x)

        out = []
        encoding = []
        for i in range(self.nStack):
            hg = self.hourglass[i](x) ##Bz, 256, 64, 64
            ll = hg
            for j in range(self.nModules):
                ll = self.Residual[i * self.nModules + j](ll)
            ll = self.lin_[i](ll)
            tmpOut = self.tmpOut[i](ll)

            out.append(tmpOut)
            if i < self.nStack - 1:
                ll_ = self.ll_[i](ll)
                tmpOut_ = self.tmpOut_[i](tmpOut)
                x = x + ll_ + tmpOut_
                encoding.append(x)
            else:
                encoding.append(ll)

        return out, encoding


class Net_HG_hm_dep(nn.Module):
#     _Ge Liuhao_CVPR2019
    def __init__(self, num_joints, num_stages=2, num_modules=2, num_feats=256):
        super(Net_HG_hm_dep, self).__init__()

        self.numOutput, self.nStack, self.nModules, self.nFeats = num_joints, num_stages, num_modules, num_feats

        self.conv1_ = nn.Conv2d(3, 64, bias=True, kernel_size=7, stride=2, padding=3) ### 256 --> 128
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.r1 = _Residual_Module(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.r4 = _Residual_Module(128, 128)
        self.r5 = _Residual_Module(128, self.nFeats)

        _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_, _reg_ = [], [], [], [], [], [], []
        for i in range(self.nStack):
            _hourglass.append(Hourglass_LiuhaoGe_CVPR2019(4, self.nModules, self.nFeats))
            for j in range(self.nModules):
                _Residual.append(_Residual_Module(self.nFeats, self.nFeats))
            lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1),
                                nn.BatchNorm2d(self.nFeats), self.relu)
            _lin_.append(lin)
            _tmpOut.append(nn.Conv2d(self.nFeats, self.numOutput, bias=True, kernel_size=1, stride=1))
            if i < self.nStack - 1:
                _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1))
                _tmpOut_.append(nn.Conv2d(self.numOutput, self.nFeats, bias=True, kernel_size=1, stride=1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin_ = nn.ModuleList(_lin_)
        self.tmpOut = nn.ModuleList(_tmpOut)
        self.ll_ = nn.ModuleList(_ll_)
        self.tmpOut_ = nn.ModuleList(_tmpOut_)

        self.out2dep = nn.Conv2d(21, 20, 1, 1, bias=True)

    def forward(self, x):
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.r1(x)
        x = self.maxpool(x)
        x = self.r4(x)
        x = self.r5(x)

        out = []
        encoding = []
        for i in range(self.nStack):
            hg = self.hourglass[i](x) ##Bz, 256, 64, 64
            ll = hg
            for j in range(self.nModules):
                ll = self.Residual[i * self.nModules + j](ll)
            ll = self.lin_[i](ll)
            tmpOut = self.tmpOut[i](ll)

            out.append(tmpOut)
            if i < self.nStack - 1:
                ll_ = self.ll_[i](ll)
                tmpOut_ = self.tmpOut_[i](tmpOut)
                x = x + ll_ + tmpOut_
                encoding.append(x)
            else:
                encoding.append(ll)

        hm, dep = out[0], self.out2dep(out[1])
        encoding_ = encoding[0] + encoding[1]

        return hm, dep, encoding_