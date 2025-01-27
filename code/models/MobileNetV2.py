"""

MobileNet V2 Architecture(~ 12th layer)

Input           Operator            t       c         n     s
224x224x3       Conv1               -       32        1     2
112x112x32      Bottleneck2         1       16        1     1
112x112x16      Bottleneck3         6       24        2     2
56x56x24        Bottleneck4         6       32        3     2
28x28x32        Bottleneck5         6       64        4     2
14x14x64        Bottleneck6         6       96        3     1
14x14x96        Bottleneck7         6       160       3     2
7x7x160         Bottleneck8         6       320       1     1
7x7x320         Conv9               -       1280      1     1

"""


import torch
import torch.nn as nn
import torchvision.models as models

from torchsummary import summary


class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()

        # squeezenet1_0 model and pre-trained weights with ImageNet
        mobilenet_v2 = models.mobilenet_v2(pretrained=True)

        # features: no use fully connected layer of SqueezeNet
        layers = list(mobilenet_v2.features.children())
        self.features = nn.Sequential(*layers[:18])


    def forward(self, x):
        x = self.features(x) # Start: (batch_size, 3, 100, 100)
        # x = self.SENet(x) # End: (batch_size, 320, 4, 4)
        return x


if __name__ == "__main__":
    model = MobileNetV2().cuda()
    summary(model, input_size=(3, 100, 100))