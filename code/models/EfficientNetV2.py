import torch.nn as nn
import timm
from torchsummary import summary

class EfficientNetv2(nn.Module):
    def __init__(self):
        super(EfficientNetv2, self).__init__()

        # squeezenet1_0 model and pre-trained weights with ImageNet
        self.efficientnetv2 = timm.create_model('efficientnetv2_rw_s.ra2_in1k', pretrained=True, features_only=True)


    def forward(self, x):
        x = self.efficientnetv2(x) # Start: (batch_size, 3, 100, 100)
        return x


if __name__ == "__main__":
    model = EfficientNetv2().cuda()
    summary(model, input_size=(3, 100, 100))