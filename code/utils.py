import numpy as np

class ConfusionMatrix:
    def __init__(self, conf_matrix, dataset):
        self.conf_matrix = conf_matrix
        self.dataset = dataset

    def analyze(self):
        for i in range(len(self.conf_matrix)):
            correct = self.conf_matrix[i, i]
            div = self.conf_matrix[i].sum()
            acc = correct / div if div > 0 else 0
            if i == 0:
                print(f"\nConfusion Matrix for the {self.dataset}")
            print(f"Class {i} Accuracy: {acc:.2f} ({correct}/{div}) result: {self.conf_matrix[i]}")
            if i == (len(self.conf_matrix) - 1):
                print(" ")

if __name__ == "__main__":
    conf_matrix = np.array([[10, 2, 3], [1, 20, 5], [6, 7, 30]])
    ConfusionMatrix(conf_matrix, 'dataset').analyze()
