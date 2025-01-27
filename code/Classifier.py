import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SENet import SENet
from torchvision import transforms


class MVClassifier(nn.Module):
    def __init__(self, model, num_classes=None, threshold=None):
        super(MVClassifier, self).__init__()
        self.model = model
        self.threshold = threshold

        self.SENet = SENet(512 * 6)
        self.reduce = nn.Conv2d(in_channels=512 * 6, out_channels=512, kernel_size=3, padding=1)

        self.fclayer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, views):
        fuse, logits = [], []

        # Process each view
        for view in views:
            output = self.model(view)
            logit = F.adaptive_avg_pool2d(output, 1).squeeze()  # Use F.adaptive_avg_pool2d
            logit = self.fclayer(logit)
            logits.append(logit)
            fuse.append(output)

        # Fuse views
        fusion = torch.cat(fuse, dim=1)
        fusion = self.SENet(fusion)

        # Reduce and pool for final fusion
        flatten = self.reduce(fusion)
        flatten = F.adaptive_avg_pool2d(flatten, 1).squeeze()  # Use F.adaptive_avg_pool2d
        fusion_logit = self.fclayer(flatten)

        # Apply threshold if specified
        for batch in fusion_logit:
            if self.threshold and F.softmax(batch, dim=0)[-1] < self.threshold:
                batch[-1] = -10000

        return logits, fusion_logit
