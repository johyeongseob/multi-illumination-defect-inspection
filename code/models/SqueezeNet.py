import torch
import torch.nn as nn


class Fire(nn.Module):
    def __init__(self, inplanes, channel):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(inplanes, channel, kernel_size=1, stride=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(channel, channel*4, kernel_size=1, stride=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(channel, channel*4, kernel_size=3, stride=1, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.squeeze_activation(x)
        out1 = self.expand1x1(x)
        out1 = self.expand1x1_activation(out1)
        out2 = self.expand3x3(x)
        out2 = self.expand3x3_activation(out2)
        out = torch.cat([out1, out2], dim=1)
        return out


class SENet(nn.Module):
    def __init__(self, c, r=1):
        super(SENet, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_filter):
        batch, channel, _, _ = input_filter.size()
        se = (self.squeeze(input_filter)).view(batch, channel)
        ex = (self.excitation(se))
        return input_filter * ex.view(batch, channel, 1, 1)


class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2), # Start: (batch_size, RGB, 100, 100)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(96, 16),
            Fire(128, 16),
            Fire(128, 32),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(256, 32),
            Fire(256, 48),
            Fire(384, 48),
            Fire(384, 64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(512, 64)
        )
        self.SENet = SENet(512)

    def forward(self, x):
        x = self.features(x)
        x = self.SENet(x)
        return x

