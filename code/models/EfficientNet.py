"""
Ref. https://github.com/withAnewWorld/models_from_scratch/blob/main/EfficientNet.ipynb

EfficientNet-B0 baseline network

Stage           Operator                  Resolution        Channel     Layers
1               Conv3x3                   224 x 224         32          1
2               MBConv1, k3x3             112 x 112         16          1
3               MBConv6, k3x3             112 x 112         24          2
4               MBConv6, k5x5             56 x 56           40          2
5               MBConv6, k3x3             28 x 28           80          3
6               MBConv6, k5x5             14 x 14           112         3
7               MBConv6, k5x5             14 x 14           192         4
8               MBConv6, k3x3             7 x 7             320         1
9               Conv1x1 & Pooling & FC    7 x 7             1280        1

"""


import torch
import torch.nn as nn


class Swish(nn.Module): # search torch.nn.SiLU()
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class DepthWiseConv(nn.Module):

    def __init__(self, in_channels = 3, kernel_size = 3, stride = 2):
        super(DepthWiseConv, self).__init__()
        '''
        inputs
          - input_tensor: Tensor[N, C, H, W]
          - kernel_size(int)
          - stride(int)
        outputs
          - Tensor[N, C, H, W]
        '''
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = kernel_size, stride = stride, padding = padding, groups = in_channels)

    def forward(self ,x):
        return self.conv(x)


class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pointwise = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (1, 1), stride = 1)

    def forward(self, x):
        return self.pointwise(x)


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return self.GAP(x)


class Excitation(nn.Module):
    def __init__(self, in_channels, r=4):
        '''
        inputs
          - in_channels
          - r(int): channel reduction ratio
        '''
        super(Excitation, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels / r), kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(in_channels / r), out_channels=in_channels, kernel_size=(1, 1), stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)


class SENet(nn.Module):
    def __init__(self, in_channels, r = 4):
        super(SENet, self).__init__()
        self.SE_block = nn.Sequential(
            Squeeze(),
            Excitation(in_channels, r)
        )

    def forward(self, x):
        output = self.SE_block(x)
        return x * output


class MBConv1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, r = 4):

        '''
        inputs:
          - in_channels(int)
          - out_channels(int)
          - kernel_size(int): 3: will get padding 1, 5: will get padding 5 in depth wise conv
          - stride(int): 1: will retrun same resolution(H*W), 2: will return resolution/4 (H/2 * W/2)
          - r(int): channel reduction ratio of SENet
        returns:
          - MBConv1
        '''
        super(MBConv1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.block = nn.Sequential(
            DepthWiseConv(in_channels = in_channels, kernel_size = kernel_size, stride = stride),
            nn.BatchNorm2d(num_features = in_channels),
            Swish(),
            SENet(in_channels = in_channels, r = 4),
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (1, 1), stride = 1),
            nn.BatchNorm2d(num_features = out_channels)
        )

    def forward(self, x):
        '''
        input:
          -  x(Tensor[N, in_C, H, W])
        return:
          -  x(Tensor[N, out_C, H', W']): stride 1: H', W' = H, W | stride 2: H', W' = H/2, W/2
        '''
        # inverted residual block
        if self.in_channels == self.out_channels and self.stride == 1:
            identity = x
            x = self.block(x)
            x += identity
        else:
            x = self.block(x)

        return x


class MBConv6(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, r=4):
        '''
        inputs:
          - in_channels(int)
          - out_channels(int)
          - kernel_size(int): 3: will get padding 1, 5: will get padding 5 in depth wise conv
          - stride(int): 1: will retrun same resolution(H*W), 2: will return half resolution(H/2 * W/2)
          - r(int): channel reduction ratio of SENet
        returns:
          - MBConv6 (6 means channel expansion)
        '''

        super(MBConv6, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6 * in_channels, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(num_features=6 * in_channels),
            Swish(),
            DepthWiseConv(in_channels=6 * in_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(num_features=6 * in_channels),
            Swish(),
            SENet(in_channels=6 * in_channels, r=4),
            nn.Conv2d(in_channels=6 * in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        '''
        input:
          -  x(Tensor[N, in_C, H, W])
        return:
          -  x(Tensor[N, out_C, H', W']): stride 1: H', W' = H, W | stride 2: H', W' = H/2, W/2
        '''

        # inverted residual block
        if self.in_channels == self.out_channels and self.stride == 1:
            identity = x
            x = self.block(x)
            x += identity
        else:
            x = self.block(x)

        return x


class EfficientNet(nn.Module):
    def __init__(self, img_channels, num_classes):
        super(EfficientNet, self).__init__()
        self.stage_1 = nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)

        self.stage_2 = MBConv1(in_channels=32, out_channels=16, kernel_size=3, stride=2)

        self.stage_3 = [MBConv6(in_channels=16, out_channels=24, kernel_size=3, stride=1),
                        MBConv6(in_channels=24, out_channels=24, kernel_size=3, stride=1)]
        self.stage_3 = nn.Sequential(*self.stage_3)

        self.stage_4 = [MBConv6(in_channels=24, out_channels=40, kernel_size=5, stride=2),
                        MBConv6(in_channels=40, out_channels=40, kernel_size=5, stride=1)]

        self.stage_4 = nn.Sequential(*self.stage_4)

        self.stage_5 = [MBConv6(in_channels=40, out_channels=80, kernel_size=3, stride=2),
                        MBConv6(in_channels=80, out_channels=80, kernel_size=3, stride=1),
                        MBConv6(in_channels=80, out_channels=80, kernel_size=3, stride=1)]

        self.stage_5 = nn.Sequential(*self.stage_5)

        self.stage_6 = [MBConv6(in_channels=80, out_channels=112, kernel_size=5, stride=2),
                        MBConv6(in_channels=112, out_channels=112, kernel_size=5, stride=1),
                        MBConv6(in_channels=112, out_channels=112, kernel_size=5, stride=1)]

        self.stage_6 = nn.Sequential(*self.stage_6)

        self.stage_7 = [MBConv6(in_channels=112, out_channels=192, kernel_size=5, stride=1),
                        MBConv6(in_channels=192, out_channels=192, kernel_size=5, stride=1),
                        MBConv6(in_channels=192, out_channels=192, kernel_size=5, stride=1),
                        MBConv6(in_channels=192, out_channels=192, kernel_size=5, stride=1),
                        MBConv6(in_channels=192, out_channels=192, kernel_size=5, stride=1)]

        self.stage_7 = nn.Sequential(*self.stage_7)

        self.stage_8 = MBConv6(in_channels=192, out_channels=320, kernel_size=3, stride=2)

        self.stage_9 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=(1, 1), stride=1),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features=1280, out_features=num_classes)
        )

    def forward(self, x):
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.stage_5(x)
        x = self.stage_6(x)
        x = self.stage_7(x)
        x = self.stage_8(x)
        x = self.stage_9(x)
        return x