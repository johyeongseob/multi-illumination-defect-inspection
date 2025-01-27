from Classifier import MVClassifier
from DataLoader import *
from models.PretrainedSqueezeNet import PretrainedSqueezeNet
from sklearn.metrics import confusion_matrix
from utils import *
import torch
import torch.optim as optim
import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define dataSet, dataLoader
batch_size = 2 ** 5

train_dataset = MVDataset(base_dir='../data_set/train', target='test', augmentation=True)
valid_dataset = MVDataset(base_dir='../data_set/val', target='test')

# train_sampler = BalancedBatchSampler(train_dataset.labels_list, batch_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

baseline = PretrainedSqueezeNet()
model = MVClassifier(baseline, num_classes=4, threshold=None).to(device)

# class_weight = torch.tensor([1.2, 1.0], dtype=torch.float)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)

weight_path = '../weights/weight_test.pth'

print(f"\nTraining model: {model.__class__.__name__}, Baseline: {baseline.__class__.__name__}, "
      f"Criterion: {criterion}, Optimizer: {optimizer.__class__.__name__}.\n")

i, best_NG = 0, 0.0
epochs = 150
total_time = time.time()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    preds, targets = [], []

    for images, labels in train_loader:
        images = [image.to(device) for image in images]  # images = [views, batch_size, channel, height, width]
        labels = labels.to(device)

        # Forward pass, losses = each view losses(7) + fusion view loss
        _, fusion_logit = model(images)

        fusion_logit = fusion_logit.unsqueeze(0) if fusion_logit.dim() == 1 else fusion_logit

        losses = criterion(fusion_logit, labels)
        epoch_loss += losses

        # Backward pass
        optimizer.zero_grad()
        losses.backward()

        # Update weights
        optimizer.step()

        _, pred = torch.max(fusion_logit, 1)
        preds.extend(pred.cpu().numpy())
        targets.extend(labels.cpu().numpy())

    conf_matrix = confusion_matrix(targets, preds)

    correct, total, pred_OK = [], [], []

    for i in range(len(conf_matrix)):
        correct.append(conf_matrix[i, i])
        pred_OK.append(conf_matrix[i, -1])
        total.append(conf_matrix[i].sum())

    train_accuracy = sum(correct[:]) / sum(total[:])
    NG_accuracy = (sum(total[:-1]) - sum(pred_OK[:-1])) / sum(total[:-1])
    OK_accuracy = correct[-1] / total[-1]

    print(f"Epoch: {epoch+1}, train accuracy: {train_accuracy:.2f}, "
          f"NG accuracy: {NG_accuracy:.2f}, OK accuracy: {OK_accuracy:.2f}, Loss: {epoch_loss:.2f}.")

    # Return confusion matrix for training dataset every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for images, labels in train_loader:
                images = [image.to(device) for image in images]
                labels = labels.to(device)

                _, fusion_logit = model(images)
                fusion_logit = fusion_logit.unsqueeze(0) if fusion_logit.dim() == 1 else fusion_logit

                _, pred = torch.max(fusion_logit, 1)
                preds.extend(pred.cpu().numpy())
                targets.extend(labels.cpu().numpy())

            ConfusionMatrix(confusion_matrix(targets, preds), 'train dataset').analyze()

    # Evaluate model
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in valid_loader:
            images = [image.to(device) for image in images]
            labels = labels.to(device)

            _, fusion_logit = model(images)
            fusion_logit = fusion_logit.unsqueeze(0) if fusion_logit.dim() == 1 else fusion_logit

            _, pred = torch.max(fusion_logit, 1)
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())

        conf_matrix = confusion_matrix(targets, preds)

        correct, total, pred_OK = [], [], []

        for i in range(len(conf_matrix)):
            correct.append(conf_matrix[i, i])
            pred_OK.append(conf_matrix[i, -1])
            total.append(conf_matrix[i].sum())

        valid_accuracy = sum(correct[:]) / sum(total[:])
        NG_accuracy = (sum(total[:-1]) - sum(pred_OK[:-1])) / sum(total[:-1])
        OK_accuracy = correct[-1] / total[-1]

        print(
            f"Epoch: {epoch + 1}, valid accuracy: {valid_accuracy:.2f}, NG accuracy: {NG_accuracy:.2f}, "
            f"OK accuracy: {OK_accuracy:.2f}.")

        if valid_accuracy > 0.5:
            if NG_accuracy > best_NG:
                best_NG = NG_accuracy
                torch.save(model.state_dict(), weight_path)
                print(f"Saved weight: {weight_path} for the best NG.")

        if train_accuracy > 0.99:
            break

print(f'\nTraining end. Total epoch: {epochs + 1}, Total training Time: {(time.time() - total_time) / 3600: .2f} hours\n')
