import torch.nn as nn
from torchvision import models
from models.SENet import *
from torchsummary import summary

class MobileNetV3(nn.Module):
    def __init__(self):
        super(MobileNetV3, self).__init__()

        # squeezenet1_0 model and pre-trained weights with ImageNet
        mobilenetv3 = models.mobilenet_v3_small(pretrained=True)

        # features: no use fully connected layer of SqueezeNet
        self.features = nn.Sequential(*list(mobilenetv3.features.children()))
        self.SENet = SENet(576)


    def forward(self, x):
        x = self.features(x) # Start: (batch_size, 3, 100, 100)
        x = self.SENet(x)
        return x


if __name__ == "__main__":
    model = MobileNetV3().cuda()
    summary(model, input_size=(3, 100, 100))