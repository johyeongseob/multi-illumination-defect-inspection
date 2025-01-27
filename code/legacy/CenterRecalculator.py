import torch
from Loss.CentroidTripletLoss import *


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

if __name__ == "__main__":
    num_samples = 1000
    num_features = 512
    num_classes = 4

    embeddings = torch.randn(num_samples, num_features).cuda()
    labels = torch.randint(0, num_classes, (num_samples,)).cuda()

    centroid_triplet_loss = CentroidTripletLoss(margin=1)
    centroid_triplet_loss(embeddings, labels)

    center_recalculator = CenterRecalculator(centroid_triplet_loss, embeddings, labels)
    center_recalculator.update_centers_epoch()