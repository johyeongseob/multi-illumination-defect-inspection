import torch
import torch.nn as nn
from Loss.TripletLoss import TripletLoss


class CentroidTripletLoss(nn.Module):
    def __init__(self, num_classes=4, feat_dim=512*7, margin=1):
        super(CentroidTripletLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device='cuda'))
        self.margin = margin
        self.triplet_criterion = TripletLoss(margin=self.margin)

    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        centers_batch = self.centers.index_select(0, labels)
        center_loss = (embeddings - centers_batch).pow(2).sum() / 2.0 / batch_size

        triplet_loss = self.triplet_criterion(embeddings, labels)

        anchor_indices = torch.arange(batch_size).cuda()
        anchor_embeddings = embeddings[anchor_indices]

        positive_centroids = self.centers[labels]
        negative_centroids = []
        for i in range(batch_size):
            other_labels = [j for j in range(len(self.centers)) if j != labels[i]]
            negative_centroids.append(torch.mean(self.centers[other_labels], dim=0))
        negative_centroids = torch.stack(negative_centroids).cuda()

        pos_dists = torch.sum((anchor_embeddings - positive_centroids) ** 2, dim=1)
        neg_dists = torch.sum((anchor_embeddings - negative_centroids) ** 2, dim=1)

        centroid_triplet_loss = torch.mean(torch.relu(pos_dists - neg_dists + self.margin))

        return (center_loss*0.01 + triplet_loss + centroid_triplet_loss*0.01)


class CenterRecalculator:
    def __init__(self, centroid_triplet_loss, embeddings, labels):
        self.centroid_triplet_loss = centroid_triplet_loss
        self.embeddings = embeddings
        self.labels = labels

    def update_centers_epoch(self):
        with torch.no_grad():
            new_centers = torch.zeros_like(self.centroid_triplet_loss.centers)
            print(f"before epoch update: {self.centroid_triplet_loss.centers}")

            for i in range(new_centers.size(0)):
                mask = (self.labels == i)
                if mask.sum() > 0:
                    new_centers[i] = self.embeddings[mask].mean(dim=0)  # 해당 클래스의 임베딩 벡터 평균 계산

            self.centroid_triplet_loss.centers.data = new_centers.data
            print(f"after epoch update: {self.centroid_triplet_loss.centers}")

if __name__ == '__main__':
    num_classes = 4
    feat_dim = 512*7
    batch_size = 2**5

    embeddings = torch.randn(batch_size, feat_dim).cuda()
    labels = torch.randint(0, num_classes, (batch_size,)).cuda()
    print(f"embeddings: {embeddings.shape}")
    print(f"labels: {labels}")

    centroid_triplet_loss = CentroidTripletLoss(num_classes=4, feat_dim=512*7, margin=1)
    loss = centroid_triplet_loss(embeddings, labels)

    print(f"centroid_triplet_loss: {loss}")

    center_recalculator = CenterRecalculator(centroid_triplet_loss, embeddings, labels)
    center_recalculator.update_centers_epoch()
