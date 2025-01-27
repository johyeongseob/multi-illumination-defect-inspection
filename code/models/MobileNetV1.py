"""
Ref. HuggingFace: MobileNet V1

"""


import torch
import torch.nn as nn
from transformers import MobileNetV1Config, MobileNetV1Model # pip install transformers
from torchsummary import summary



class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()

        # Initializing a "mobilenet_v1_1.0_224" style configuration
        configuration = MobileNetV1Config()
        # Initializing a model from the "mobilenet_v1_1.0_224" style configuration
        huggingfacemodel = MobileNetV1Model(configuration)

        layers = list(huggingfacemodel.children())

        self.features1 = nn.Sequential(layers[0])
        # features2 use layers (from second layer to twelve layer)
        self.features2 = nn.Sequential(*layers[1])

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        return x

if __name__ == "__main__":
    model = MobileNetV1().cuda()
    summary(model, input_size=(3, 100, 100))