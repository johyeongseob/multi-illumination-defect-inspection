import torch.nn as nn
from torchvision import models
from torchsummary import summary

class ShuffleNetv2_x1_0(nn.Module):
    def __init__(self):
        super(ShuffleNetv2_x1_0, self).__init__()

        # Choose version: 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
        shufflenetv2 = models.shufflenet_v2_x1_0(pretrained=True)

        # features: no use fully connected layer of SqueezeNet
        self.features = nn.Sequential(*list(shufflenetv2.children())[:-2])


    def forward(self, x):
        x = self.features(x) # Start: (batch_size, 3, 100, 100)
        return x


if __name__ == "__main__":
    model = ShuffleNetv2_x1_0().cuda()
    summary(model, input_size=(3, 100, 100))