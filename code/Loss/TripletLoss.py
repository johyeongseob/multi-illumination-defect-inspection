import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin  # 트리플렛 손실에서 사용할 마진 값

    def PairwiseDistance(self, x):
        '''
        주어진 임베딩 벡터들 사이의 쌍별 유클리드 거리 행렬을 계산합니다.
        x: n개의 샘플을 가진 n-by-p 행렬
        반환값: n x n 쌍별 제곱 유클리드 거리 행렬
        '''
        n_samples, _ = x.shape
        r = torch.sum(x ** 2, dim=1, keepdim=True)
        S = r - 2 * torch.mm(x, x.T) + r.T
        S.fill_diagonal_(0)  # 대각선 원소를 0으로 설정하여 자신과의 거리를 0으로 만듦
        S = torch.clamp(S, min=0)  # 부동 소수점 오류로 인한 음수 거리를 방지
        return S / n_samples

    def forward(self, embeddings, labels):
        # 임베딩 벡터들을 정규화합니다.
        Dw = self.PairwiseDistance(embeddings)

        # 긍정 및 부정 마스크를 생성합니다.
        labels = labels.unsqueeze(1)
        positive_mask = torch.eq(labels, labels.T).float()  # 같은 클래스면 1, 아니면 0
        negative_mask = torch.ne(labels, labels.T).float()  # 다른 클래스면 1, 아니면 0

        # 긍정 및 부정 거리를 추출합니다.
        pos_distances = Dw * positive_mask
        neg_distances = Dw * negative_mask

        # 가장 어려운 긍정 거리와 부정 거리를 계산합니다.
        hardest_positive_dist = torch.max(pos_distances, dim=1)[0]
        hardest_negative_dist = torch.min(neg_distances + 1e6 * (1 - negative_mask), dim=1)[0]

        # 트리플렛 손실을 계산합니다.
        loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)

        return torch.mean(loss)  # 손실 값의 평균을 반환합니다.

# TripletLoss 테스트
if __name__ == '__main__':
    margin = 1.0
    embeddings = torch.tensor([[1.0, 2.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    print(f"embeddings: {embeddings}")
    print(f"labels: {labels}")

    triplet_loss = TripletLoss(margin)
    loss = triplet_loss(embeddings, labels)
    print(f"Loss: {loss.item()}")