"""
Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        손실함수 값을 계산하기 위한 모델

        Args:
            features: embeddings 벡터값 [batch_size, embeddings_dim], ex) [32, 128].
            labels: 레이블값 [batch_size, label], ex) [32, 1].
            mask: 'positive'와 'negative'를 구분하는 마스크 [labels, labels]
            mask_{i,j}=1 만약 j가 i와 같은 클래스이면 1이다.

        Returns:
            손실함수 결과 스칼라 값
        """

        # 장치 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 마스크 생성
        labels = labels.view(-1, 1) # dim: [batch_size, 1]
        mask = torch.eq(labels, labels.T).float().to(device) # dim: [batch_size, batch_size]

        # 내적(dot product) 계산 및 스케일링
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature) # dim: [batch_size, batch_size]

        # 수치적 안정성 확보: logit 값들을 0 또는 음수로 만들어 softmax 계산을 원할하게 한다.
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # dim: [batch_size, batch_size]

        # 자기 자신과의 비교 제거: 대각 성분이 0이고 나머지 성분이 1인 논리_마스크
        batch_size = features.shape[0]
        logits_mask = ~torch.eye(batch_size, device=device).bool()
        mask = mask * logits_mask

        """
        로그 확률 계산
        # log 내부 값이 0이 되는 것을 방지하기 위해 epsilon을 추가
        # 자신을 제외한 나머지 features 간의 내적 값: sigma(a in A(i)) {z_i * z_a/ t}
        """
        epsilon = 1e-8
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + epsilon)

        # 개별 Loss^{sup}_{out,i} 계산
        mask_pos_pairs = mask.sum(1) # 앵커별 |P(i)|의 값을 계산: 각 행의 요소들 합을 계산
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs) # |P(i)| = 0 일시, 1로 변환
        log_prob = mask * log_prob # 최종 log_prob 계산
        mean_log_prob_pos = (log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -mean_log_prob_pos
        loss = loss.view(1, batch_size).mean() # 배치 내 anchor (i)에 대한 평균값

        return loss

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 4
    feat_dim = 128
    batch_size = 8

    embeddings = torch.randn(batch_size, feat_dim).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    print(f"embeddings: {embeddings.shape}")
    print(f"labels: {labels}")

    criterion = SupConLoss(temperature=0.5)

    loss = criterion(embeddings, labels)

    print(f'loss = {loss}')