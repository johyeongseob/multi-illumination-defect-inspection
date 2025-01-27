"""
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
from efficientnet_pytorch import EfficientNet # pip install efficientnet_pytorch
from torchsummary import summary


class EfficientNetV1(nn.Module):
    def __init__(self, version='b0'):
        super(EfficientNetV1, self).__init__()

        # EfficientNet model and pre-trained weights with ImageNet
        # Ref. https://pytorch.org/vision/main/models/efficientnet.html
        # Choose one of 8 versions (B0 ~ B7)
        model_name = f'efficientnet-{version}'
        self.efficientnet = EfficientNet.from_pretrained(model_name)


    def forward(self, x):
        x = self.efficientnet.extract_features(x) # Start: (batch_size, 3, 100, 100)
        return x


if __name__ == "__main__":
    model = EfficientNetV1(version='b0').cuda()
    summary(model, input_size=(3, 100, 100))
