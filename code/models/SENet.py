import torch
import torch.nn as nn

class SENet(nn.Module):
    def __init__(self, c, r=2):
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
        alpha = ex.view(batch, channel, 1, 1)
        return alpha * input_filter