import torch

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=10):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin  # 대조 손실에서 사용할 마진 값

    def PairwiseDistance(self, x):
        '''
        주어진 임베딩 벡터들 사이의 쌍별 제곱 유클리드 거리 행렬을 계산합니다.
        X:              n x p 행렬, 여기서 n은 샘플 수, p는 차원 수입니다.
        반환값:        n x n 쌍별 제곱 유클리드 거리 행렬
        '''
        _, dimension = x.size()
        r = torch.sum(x ** 2, dim=1, keepdim=True)
        S = r - 2 * torch.mm(x, x.T) + r.T
        S.fill_diagonal_(0)  # 대각선 원소를 0으로 설정하여 자신과의 거리를 0으로 만듦
        S = torch.clamp(S, min=0)  # 부동 소수점 오류로 인한 음수 거리를 방지
        return S / dimension

    def forward(self, embeddings, labels):
        # 임베딩 벡터들을 정규화합니다.
        Dw = self.PairwiseDistance(embeddings)

        # 긍정 및 부정 마스크를 생성합니다.
        labels = labels.unsqueeze(1)
        positive_mask = torch.eq(labels, labels.T).float()  # 같은 클래스면 1, 아니면 0
        negative_mask = torch.ne(labels, labels.T).float()  # 다른 클래스면 1, 아니면 0

        # 손실을 계산합니다.
        # 손실: Y*Dw + (1-Y)*max(0, margin-Dw), 여기서 클래스가 같으면 Y = 1, 다르면 Y = 0
        loss = Dw * positive_mask + torch.relu(self.margin - Dw) * negative_mask
        loss = torch.triu(loss)  # 삼각 행렬로 변환하여 중복된 손실을 제거
        return torch.sum(loss) # 손실 값의 합을 반환

if __name__ == '__main__':
    num_classes = 4
    feat_dim = 512*7
    batch_size = 2**5

    embeddings = torch.randn(batch_size, feat_dim).cuda()
    labels = torch.randint(0, num_classes, (batch_size,)).cuda()
    print(f"embeddings: {embeddings.shape}")
    print(f"labels: {labels}")

    criterion = ContrastiveLoss(margin=4)
    Ctl_loss = criterion(embeddings, labels)
    print(f"Ctl_loss: {Ctl_loss}")