#!/bin/bash

echo "🚀 Project structure 생성 시작"

# =========================
# 1. 폴더 생성
# =========================
mkdir -p data
mkdir -p models
mkdir -p train
mkdir -p utils
mkdir -p logs/tensorboard
mkdir -p results/confusion_matrix
mkdir -p results/plots

echo "✔ 폴더 생성 완료"

# =========================
# 2. utils/dataset.py
# =========================
cat << 'EOT' > utils/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class WaferDatasetCNN(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = torch.tensor(x)

        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


class WaferDatasetTL(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = np.repeat(x, 3, axis=0)

        x = torch.tensor(x)
        y = torch.tensor(self.y[idx], dtype=torch.long)

        return x, y
EOT

echo "✔ dataset.py 생성 완료"

# =========================
# 3. models/cnn.py
# =========================
cat << 'EOT' > models/cnn.py
import torch.nn as nn
import torch.nn.functional as F

class CNNBaseline(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(128*8*8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
EOT

echo "✔ cnn.py 생성 완료"

# =========================
# 4. models/resnet.py
# =========================
cat << 'EOT' > models/resnet.py
import torchvision.models as models
import torch.nn as nn

def get_resnet18(num_classes=9):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
EOT

echo "✔ resnet.py 생성 완료"

# =========================
# 5. models/efficientnet.py
# =========================
cat << 'EOT' > models/efficientnet.py
import torchvision.models as models
import torch.nn as nn

def get_efficientnet(num_classes=9):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
EOT

echo "✔ efficientnet.py 생성 완료"

# =========================
# 6. train/trainer.py
# =========================
cat << 'EOT' > train/trainer.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, model_name="model"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name

        self.best_f1 = 0

    def train_one_epoch(self):
        self.model.train()
        losses = []

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)

            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        return np.mean(losses)

    def evaluate(self):
        self.model.eval()

        preds, labels = [], []

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                out = self.model(x)

                pred = torch.argmax(out, dim=1).cpu().numpy()

                preds.extend(pred)
                labels.extend(y.numpy())

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')

        return acc, f1

    def run(self, epochs=20):
        for epoch in range(epochs):

            loss = self.train_one_epoch()
            acc, f1 = self.evaluate()

            print(f"[{self.model_name}] Epoch {epoch+1}")
            print(f"Loss: {loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

            if f1 > self.best_f1:
                self.best_f1 = f1
                torch.save(self.model.state_dict(), f"logs/{self.model_name}_best.pt")
                print("✔ Best model saved")
EOT

echo "✔ trainer.py 생성 완료"

# =========================
# 7. summary.py
# =========================
cat << 'EOT' > results/summary.py
import pandas as pd

df = pd.DataFrame({
    "Model": ["CNN", "ResNet18", "EfficientNet"],
    "Accuracy": [0,0,0],
    "F1-score": [0,0,0],
    "Inference Time(ms)": [0,0,0],
    "Params(M)": [0,0,0]
})

print(df)
df.to_csv("results/metrics.csv", index=False)
EOT

echo "✔ summary.py 생성 완료"

echo "🎉 모든 프로젝트 구조 생성 완료!"
