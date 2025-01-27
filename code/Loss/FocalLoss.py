import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # CrossEntropyLoss 계산
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = F.softmax(inputs, dim=1)[range(len(targets)), targets]
        at = self.alpha[targets.long()]

        # Focal Loss 계산
        FC_loss = at * (1 - pt) ** self.gamma * CE_loss

        # reduction 방식에 따라 반환
        if self.reduction == 'mean':
            return FC_loss.mean()
        elif self.reduction == 'sum':
            return FC_loss.sum()
        else:
            return FC_loss
