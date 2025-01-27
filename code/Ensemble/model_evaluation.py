from torch.utils.data import DataLoader
from Classifier import MVClassifier
from DataLoader import MVDataset
from models.PretrainedSqueezeNet import PretrainedSqueezeNet
from utils import *
from sklearn.metrics import confusion_matrix
from output_visualization import *
import torch
import torch.nn.functional as F
from torchvision import transforms


target = 'test'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataSet, dataLoader
batch_size = 2**5

transform = transforms.Compose([transforms.ToTensor()])

test_dataset = MVDataset(base_dir='../data_set/test', target=target)
valid_dataset = MVDataset(base_dir='../data_set/val', target=target)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

test_indices = test_dataset.indices

weight_path = '../weights/weight_test.pth'

baseline = PretrainedSqueezeNet()

model = MVClassifier(baseline, num_classes=4, threshold=None).to(device)

model.load_state_dict(torch.load(weight_path))

print(f"\nTraining model: {model.__class__.__name__}, Baseline: {baseline.__class__.__name__}, "
      f"Weight path: {weight_path}\n")

model.eval()

preds, targets = [], []

with torch.no_grad():
    for images, labels in valid_loader:
        images = [image.to(device) for image in images]
        labels = labels.to(device)

        _, fusion_logit = model(images)
        if fusion_logit.dim() == 1:
            fusion_logit = fusion_logit.unsqueeze(0)

        _, pred = torch.max(fusion_logit, 1)
        preds.extend(pred.cpu().numpy())
        targets.extend(labels.cpu().numpy())

    ConfusionMatrix(confusion_matrix(targets, preds), 'valid dataset').analyze()

preds, targets = [], []


with torch.no_grad():
    for images, labels in test_loader:
        images = [image.to(device) for image in images]
        labels = labels.to(device)

        _, fusion_logit = model(images)
        if fusion_logit.dim() == 1:
            fusion_logit = fusion_logit.unsqueeze(0)

        _, pred = torch.max(fusion_logit, 1)
        preds.extend(pred.cpu().numpy())
        targets.extend(labels.cpu().numpy())

    ConfusionMatrix(confusion_matrix(targets, preds), 'test dataset').analyze()
