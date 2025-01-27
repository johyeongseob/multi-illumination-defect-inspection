import torch
import torch.nn as nn
import torchvision.models as models
from models.SENet import *
from torchsummary import summary


class ResNet18(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super(ResNet18, self).__init__()

        # Pre-trained ResNet-18 모델 불러오기
        resnet = models.resnet18(pretrained=True)

        # FC 레이어를 제거하고 특징 추출기만 사용
        self.features = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.SENet = SENet(512)


    def forward(self, x):
        x = self.features(x) # Start: (batch_size, 3, 100, 100)
        # x = self.SENet(x) # End: (batch_size, 512, 5, 5)
        return x


if __name__ == "__main__":
    model = ResNet18().cuda()
    summary(model, input_size=(3, 100, 100))
    # print(model)
