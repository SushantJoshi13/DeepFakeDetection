import glob
import numpy as np
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn


# Define video dataset class
class VideoDataset(Dataset):
    def __init__(self, video_folder, labels_csv, sequence_length=60, transform=None):
        self.video_folder = video_folder
        self.labels = pd.read_csv(labels_csv)
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.video_folder))

    def __getitem__(self, idx):
        video_name = os.listdir(self.video_folder)[idx]
        if video_name.endswith('.DS_Store'):
            # Skip .DS_Store files
            return torch.zeros(self.sequence_length, 3, im_size, im_size), 0  # Placeholder tensors and label
        video_path = os.path.join(self.video_folder, video_name)
        frames = []
        cap = cv2.VideoCapture(video_path)
        success = True
        while len(frames) < self.sequence_length and success:
            success, frame = cap.read()
            if success:
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
        cap.release()
        if len(frames) == 0:
            print(f"No frames extracted from video: {video_name}")
            # Handle this scenario by returning default tensors for frames and a default label value
            frames = torch.zeros(self.sequence_length, 3, im_size, im_size)
            label = 0  # Default label value
        else:
            frames = torch.stack(frames)
            label = self.labels.loc[self.labels["file"].apply(os.path.basename) == video_name, "label"].values
            if len(label) == 0:
                print(f"No label found for video: {video_name}")
                label = 0  # Default label value
            else:
                label = 0 if label[0] == 'FAKE' else 1

        return frames, label

# Define transformations
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

# Specify paths
video_folder = "/Users/avikshitkharkar/Documents/deepfake-detection-challenge/Real_faces"
labels_csv = "/Users/avikshitkharkar/Downloads/metadata.csv"

# Split dataset into train and validation sets
video_files = [file for file in glob.glob(os.path.join(video_folder, "*.mp4")) if not file.endswith('.DS_Store')]
train_videos, valid_videos = train_test_split(video_files, test_size=0.2, random_state=42)

# Load labels
header_list = ["file", "label", "split", "original"]
labels = pd.read_csv(labels_csv, names=header_list)

# Count real and fake videos
def number_of_real_and_fake_videos(data_list, labels_df):
    fake = sum(labels_df.loc[labels_df["file"].isin([os.path.basename(i) for i in data_list]), "label"] == 'FAKE')
    real = len(data_list) - fake
    return real, fake

train_real, train_fake = number_of_real_and_fake_videos(train_videos, labels)
valid_real, valid_fake = number_of_real_and_fake_videos(valid_videos, labels)

print("TRAIN: Real:", train_real, " Fake:", train_fake)
print("VALIDATION: Real:", valid_real, " Fake:", valid_fake)

# Create datasets and data loaders
train_dataset = VideoDataset(video_folder=video_folder, labels_csv=labels_csv, sequence_length=10, transform=train_transforms)
valid_dataset = VideoDataset(video_folder=video_folder, labels_csv=labels_csv, sequence_length=10, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=0)

# Define model
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
         batch_size, seq_length, c, h, w = x.shape
         x = x.view(batch_size * seq_length, c, h, w)
         fmap = self.model(x)
         x = self.avgpool(fmap)
         x = x.view(batch_size, seq_length, 2048)
         x_lstm, _ = self.lstm(x, None)
         x_lstm = self.relu(x_lstm)
         return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

model = Model(2)

# Training and testing functions
def train_epoch(epoch, num_epochs, data_loader, model, criterion, optimizer):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    t = []
    for i, (inputs, targets) in enumerate(data_loader):
        if inputs is None or targets is None:
            continue
        _, outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("\r[Epoch %d/%d] [Batch %d / %d] [Loss: %f, Acc: %.2f%%]" %
              (epoch, num_epochs, i, len(data_loader), losses.avg, accuracies.avg), end="")
    return losses.avg, accuracies.avg

def test(epoch, model, data_loader, criterion):
    print('\nTesting')
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pred = []
    true = []
    count = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if inputs is None or targets is None:
                continue
            _, outputs = model(inputs)
            loss = torch.mean(criterion(outputs, targets))
            acc = calculate_accuracy(outputs, targets)
            _, p = torch.max(outputs, 1)
            true += targets.numpy().reshape(len(targets)).tolist()
            pred += p.numpy().reshape(len(p)).tolist()
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            print("\r[Batch %d / %d]  [Loss: %f, Acc: %.2f%%]" %
                  (i, len(data_loader), losses.avg, accuracies.avg), end="")
        print('\nAccuracy {}'.format(accuracies.avg))
    return true, pred, losses.avg, accuracies.avg

# Utility functions
class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return 100 * n_correct_elems / batch_size

# Learning rate and number of epochs
lr = 1e-5
num_epochs = 10

# Optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
train_loss_avg = []
train_accuracy = []
test_loss_avg = []
test_accuracy = []

for epoch in range(1, num_epochs + 1):
    l, acc = train_epoch(epoch, num_epochs, train_loader, model, criterion, optimizer)
    train_loss_avg.append(l)
    train_accuracy.append(acc)
    true, pred, tl, t_acc = test(epoch, model, valid_loader, criterion)
    test_loss_avg.append(tl)
    test_accuracy.append(t_acc)

# Plot loss and accuracy
def plot_loss(train_loss_avg, test_loss_avg, num_epochs):
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_loss_avg, 'g', label='Training loss')
    plt.plot(epochs, test_loss_avg, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_accuracy(train_accuracy, test_accuracy, num_epochs):
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_accuracy, 'g', label='Training accuracy')
    plt.plot(epochs, test_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_loss(train_loss_avg, test_loss_avg, len(train_loss_avg))
plot_accuracy(train_accuracy, test_accuracy, len(train_accuracy))

# Output confusion matrix
def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print('True positive = ', cm[0][0])
    print('False positive = ', cm[0][1])
    print('False negative = ', cm[1][0])
    print('True negative = ', cm[1][1])
    print('\n')
    df_cm = pd.DataFrame(cm, range(2), range(2))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    plt.ylabel('Actual label', size=20)
    plt.xlabel('Predicted label', size=20)
    plt.xticks(np.arange(2), ['Fake', 'Real'], size=16)
    plt.yticks(np.arange(2), ['Fake', 'Real'], size=16)
    plt.ylim([2, 0])
    plt.show()
    calculated_acc = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    print("Calculated Accuracy", calculated_acc * 100)

# Example usage: visualize the first sample in the training dataset
for images, labels in train_loader:
    print(images.shape)
    print(labels)
    images = torch.clamp(images, 0, 1)
    plt.imshow(np.transpose(images[0][0].numpy(), (1, 2, 0)))
    plt.show()
    break

print(confusion_matrix(true, pred))
print_confusion_matrix(true, pred)
