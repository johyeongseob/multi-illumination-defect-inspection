import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from Classifier import *
from DataLoader import *
from models.PretrainedSqueezeNet import PretrainedSqueezeNet
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE


# Define dataSet, dataLoader
batch_size = 2**5
test_dataset = MVDataset('data_set/test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_dataset = MVDataset('data_set/train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

test_indices = test_dataset.indices
train_indices = train_dataset.indices

weight_path = 'weights/Squeeze0_NG.pth'

# Set up model and tools
baseline = PretrainedSqueezeNet()
model = MVClassifier(baseline, threshold=None).to("cuda")
model.load_state_dict(torch.load(weight_path))

path1 = "t-SNE/Squeeze0_NG_train.png"
path2 = "t-SNE/Squeeze0_NG_test.png"

print(f"\nTest model: {model.__class__.__name__}, baseline: {baseline.__class__.__name__}, "
      f"Weight: {weight_path}, Dataset: {test_dataset.base_dir}\n")

# test_model.fclayer = Identity()

tsne = TSNE(n_components=2, random_state=0)

# Test model
model.eval()
features, actual = [], []
with torch.no_grad():
    for images, labels in tqdm(train_loader, ncols=150):

        images = [image.to("cuda") for image in images]  # images = [views, batch_size, channel, height, width]
        labels = labels.to("cuda")  # labels = [batch_size, label]

        _, flatten, _ = model(images)

        features += flatten.cpu().numpy().tolist()
        actual += labels.cpu().numpy().tolist()


model.eval()
with torch.no_grad():
    for images, labels in tqdm(test_loader, ncols=150):

        images = [image.to("cuda") for image in images]  # images = [views, batch_size, channel, height, width]
        labels = labels.to("cuda")  # labels = [batch_size, label]

        _, flatten, _ = model(images)

        features += flatten.cpu().numpy().tolist()
        actual += labels.cpu().numpy().tolist()

print(f"\nThe number of train data: {len(actual[:-34])}\n")
print(f"The number of test data: {len(actual[-34:])}\n")

cluster = np.array(tsne.fit_transform(np.array(features)))
actual = np.array(actual)

plt.figure(figsize=(10, 10))
classes = ['NG1', 'NG2', 'NG3', 'OK']
markers = ['o', 'o', 'o', '*']
for i, label, marker in zip(range(4), classes, markers):
    idx = np.where(actual[:-34] == i)
    plt.scatter(cluster[:-34][idx, 0], cluster[:-34][idx, 1], marker=marker, label=label)

plt.legend()
plt.tight_layout()
plt.savefig(path1)
print(f"Saved figure: {path1}")


plt.figure(figsize=(10, 10))
classes = ['NG1', 'NG2', 'NG3', 'OK']
markers = ['o', 'o', 'o', '*']
for i, label, marker in zip(range(4), classes, markers):
    idx = np.where(actual[-34:] == i)
    plt.scatter(cluster[-34:][idx, 0], cluster[-34:][idx, 1], marker=marker, label=label)

plt.legend()
plt.tight_layout()
plt.savefig(path2)
print(f"Saved figure: {path2}")

