from torch.utils.data import DataLoader
from models.PretrainedSqueezeNet import PretrainedSqueezeNet
from DataLoader import MVDataset
from Classifier import MVClassifier
from utils import *
from output_visualization import *
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
from torchvision import transforms


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define dataSet, dataLoader
batch_size = 2 ** 5
data_path = 'path'

test_dataset = MVDataset(data_path + 'test', target='model1_1', augmentation=False)
valid_dataset = MVDataset(data_path + 'valid', target='model1_1')

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Indices
test_indices = test_dataset.indices

# Model weights
weight_path1 = 'weight/model/modell_OK.pth'
weight_path2 = 'weight/model/model12_55.pth'


# Model initialization
model_stage1 = MVClassifier(PretrainedSqueezeNet(), num_classes=4, threshold=None).to(device)
model_stage1.load_state_dict(torch.load(weight_path1))

model_stage2 = MVClassifier(PretrainedSqueezeNet(), num_classes=4, threshold=None).to(device)
model_stage2.load_state_dict(torch.load(weight_path2))

# Set models to evaluation mode
model_stage1.eval()
model_stage2.eval()

final_preds = []
targets = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, ncols=200):
        # images = [views, batch_size, channel, height, width]
        images = [image.to(device) for image in images]
        labels = labels.to(device)  # labels = [batch_size, label]

        # Forward pass through both models
        _, fusion_logit1 = model_stage1(images)
        _, fusion_logit2 = model_stage2(images)

        # Get predictions from both models
        _, pred1 = torch.max(fusion_logit1, 1)
        _, pred2 = torch.max(fusion_logit2, 1)

        # Final prediction logic
        for p1, p2 in zip(pred1.cpu().numpy(), pred2.cpu().numpy()):
            if p1 == 3:
                final_preds.append(3)
            else:
                final_preds.append(p2)

        # Extend targets
        targets.extend(labels.cpu().numpy())

# Analyze results using confusion matrix
ConfusionMatrix(confusion_matrix(targets, final_preds), 'test dataset').analyze()
