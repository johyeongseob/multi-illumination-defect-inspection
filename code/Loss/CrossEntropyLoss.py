"""

nn.CrossEntropyLoss()의 하이퍼파라미터에 대해 설명해 드리겠습니다.

weight: 클래스별 가중치를 지정할 수 있는 매개변수입니다.
이 매개변수를 사용하여 손실을 계산할 때 특정 클래스에 대한 손실 기여도를 조절할 수 있습니다. 클래스 불균형을 고려하여 학습할 때 유용하게 사용됩니다.

reduction: 손실의 계산 방법을 지정하는 매개변수입니다.
기본값은 'mean'으로, 모든 샘플의 손실을 평균하여 하나의 스칼라 값으로 반환합니다.
'none'을 지정하면 각 샘플에 대한 손실이 개별적으로 반환됩니다.
다른 옵션으로는 'sum'이 있으며, 모든 샘플의 손실을 합하여 반환합니다.

ignore_index: 특정 클래스를 무시하고 손실을 계산할 때 사용하는 매개변수입니다.
일반적으로는 패딩 또는 무시해야 하는 클래스를 지정할 때 사용됩니다. 예를 들어, 시퀀스에서 패딩 토큰에 대한 손실을 무시하려는 경우에 유용합니다.

"""



import torch
import torch.nn as nn

logits = torch.tensor([[-0.0961, 0.2857, -0.4847, -0.1407],
                       [-0.0104, -0.3461, -0.2624, -0.2261],
                       [0.0977, -0.3707, -0.2187, -0.1085],
                       [0.1545, -0.2383, -0.1475, -0.0642]])

labels = torch.tensor([0, 1, 2, 3])

CE = nn.CrossEntropyLoss()
weight_CE = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.5, 1.0]))
list_CE = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.5, 1.0]), reduction='none')


loss = CE(logits, labels)
weight_loss = weight_CE(logits, labels)
list_loss = list_CE(logits, labels)
print(f"loss: {loss.item()}")
print(f"weight_loss: {weight_loss.item()}")
print(f"list_loss: {list_loss}")

for loss in list_loss:
    print(loss)
