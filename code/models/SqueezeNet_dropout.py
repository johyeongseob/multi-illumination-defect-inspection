"""

SqueezeNet with Squeeze-and-Excitation Networks(SENet)

Paper 1: SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size [2016]
Paper 2: Squeeze-and-Excitation Networks [2018]
Paper 3: Fusion of multi-light source illuminated images for effective defect inspection on highly reflective surfaces [2022]

CNN Architecture

Layers          Output size         Fiter size
Input           100 x 100 x 3
Conv 1          47 x 47 x 96        96 x 7 x 7, stride 2
Pool 1          23 x 23 x 96        3 x 3 Pooling, stride 2
Fire 2          23 x 23 x 128       16 x 1 x 1, 64 x 1 x 1, 64 x 3 x 3
Fire 3          23 x 23 x 128       16 x 1 x 1, 64 x 1 x 1, 64 x 3 x 3
Fire 4          23 x 23 x 256       32 x 1 x 1, 128 x 1 x 1, 128 x 3 x 3
Pool 4          11 x 11 x 256       3 x 3 Pooling, stride 2
Fire 5          11 x 11 x 256       32 x 1 x 1, 128 x 1 x 1, 128 x 3 x 3
Fire 6          11 x 11 x 384       48 x 1 x 1, 192 x 1 x 1, 192 x 3 x 3
Fire 7          11 x 11 x 384       48 x 1 x 1, 192 x 1 x 1, 192 x 3 x 3
Fire 8          11 x 11 x 512       64 x 1 x 1, 256 x 1 x 1, 256 x 3 x 3
Pool 8          5 x 5 x 512         3 x 3 Pooling, stride 2
Fire 9          5 x 5 x 512         64 x 1 x 1, 256 x 1 x 1, 256 x 3 x 3
SENet           5 x 5 x 512         X^CA (c) = Sig(FC(Re(FC(z))))*X(c), z(c) = GAP(X)

"""


import torch
import torch.nn as nn
import torchvision.models as models
from models.SENet import *
from torchsummary import summary


class SqueezeNet_dropout(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super(SqueezeNet_dropout, self).__init__()

        # squeezenet1_0 model and pre-trained weights with ImageNet
        squeezenet1_0 = models.squeezenet1_0(pretrained=True)

        # Spatial dropout
        self.features = nn.Sequential(
            *list(squeezenet1_0.features.children())[:3],
            nn.Dropout2d(p=dropout_prob),
            *list(squeezenet1_0.features.children())[3:6],
            nn.Dropout2d(p=dropout_prob),
            *list(squeezenet1_0.features.children())[6:9],
            nn.Dropout2d(p=dropout_prob),
            *list(squeezenet1_0.features.children())[9:11],
            nn.Dropout2d(p=dropout_prob),
            *list(squeezenet1_0.features.children())[11:],
        )
        self.SENet = SENet(512)


    def forward(self, x):
        x = self.features(x) # Start: (batch_size, 3, 100, 100)
        x = self.SENet(x) # End: (batch_size, 512, 5, 5)
        return x


if __name__ == "__main__":
    model = SqueezeNet_dropout().cuda()
    summary(model, input_size=(3, 100, 100))
    # print(model)
