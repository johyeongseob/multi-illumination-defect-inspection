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

# Ref. (torchvision) models.squeezenet1_0 -> class Fire(nn.Module)
class FireModified(nn.Module):
    def __init__(self, input_channel, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModified, self).__init__()
        self.squeeze = nn.Conv2d(in_channels=input_channel, out_channels=squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(in_channels=squeeze_channels, out_channels=expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(in_channels=squeeze_channels, out_channels=expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )

class SqueezeNet_modified_fire(nn.Module):
    def __init__(self):
        super(SqueezeNet_modified_fire, self).__init__()

        # squeezenet1_0 model and pre-trained weights with ImageNet
        squeezenet1_0 = models.squeezenet1_0(pretrained=True)

        scale_factor = (2/4)
        input_channel = [96, 128, 128, 256, 256, 384, 384, 512]
        modified_channels = [input_channel[0]] + [int(channel * scale_factor) for channel in input_channel[1:]]
        fire_indices = [3, 4, 5, 7, 8, 9, 10, 12]

        for idx, fire in enumerate(fire_indices):
            original_fire = squeezenet1_0.features[fire]
            squeeze_channels = int(original_fire.squeeze.out_channels * scale_factor)
            expand_channels = squeeze_channels * 4

            modified_fire = FireModified(input_channel=modified_channels[idx], squeeze_channels=squeeze_channels, expand1x1_channels=expand_channels, expand3x3_channels=expand_channels)

            with torch.no_grad():
                modified_fire.squeeze.weight.data.copy_(original_fire.squeeze.weight.data[:squeeze_channels, :modified_channels[idx]])
                modified_fire.squeeze.bias.data.copy_(original_fire.squeeze.bias.data[:squeeze_channels])
                modified_fire.expand1x1.weight.data.copy_(original_fire.expand1x1.weight.data[:expand_channels, :squeeze_channels])
                modified_fire.expand1x1.bias.data.copy_(original_fire.expand1x1.bias.data[:expand_channels])
                modified_fire.expand3x3.weight.data.copy_(original_fire.expand3x3.weight.data[:expand_channels, :squeeze_channels])
                modified_fire.expand3x3.bias.data.copy_(original_fire.expand3x3.bias.data[:expand_channels])

            squeezenet1_0.features[fire] = modified_fire

            for param in squeezenet1_0.features[fire].parameters():
                param.requires_grad = True

        # features: no use fully connected layer of SqueezeNet
        self.features = nn.Sequential(*list(squeezenet1_0.features.children()))
        self.SENet = SENet(int(512 * scale_factor))


    def forward(self, x):
        x = self.features(x) # Start: (batch_size, 3, 100, 100)
        x = self.SENet(x) # End: (batch_size, 512, 5, 5)
        return x


if __name__ == "__main__":
    model = SqueezeNet_modified_fire().cuda()
    summary(model, input_size=(3, 100, 100))
    # print(model)
