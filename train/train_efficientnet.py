
import sys
import os
import time

sys.path.append(
    os.path.dirname(
        os.path.abspath(
            os.path.dirname(__file__)
        )
    )
)

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0

from utils.preprocess import preprocess_p2


# ==================================================
# Dataset
# ==================================================
class WaferDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # preprocessing (현재 구조 유지)
        x = preprocess_p2(self.X[idx])

        # (H, W) -> (1, H, W)
        x = np.expand_dims(x, axis=0)

        # numpy → torch (copy 최소화)
        x = torch.from_numpy(x).to(torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)

        return x, y


# ==================================================
# Device
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ==================================================
# Data Load
# ==================================================
df = pd.read_pickle("LSWMD_clean_64.pkl")

classes = [
    "none", "Loc", "Edge-Loc", "Center",
    "Edge-Ring", "Scratch", "Random",
    "Near-full", "Donut"
]

label_map = {c: i for i, c in enumerate(classes)}
df["label_idx"] = df["clean_label"].map(label_map)

X = np.stack(df["waferMap_scaled"].values)
y = df["label_idx"].values


# ==================================================
# Train / Val Split
# ==================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)


# ==================================================
# DataLoader (핵심 최적화)
# ==================================================
train_loader = DataLoader(
    WaferDataset(X_train, y_train),
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    WaferDataset(X_val, y_val),
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# ==================================================
# Model (EfficientNet-B0)
# ==================================================
model = efficientnet_b0(weights=None)

# input channel = 1
model.features[0][0] = nn.Conv2d(
    1, 32,
    kernel_size=3,
    stride=2,
    padding=1,
    bias=False
)

# output class = 9
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    9
)

model = model.to(device)


# ==================================================
# Loss / Optimizer
# ==================================================
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001   # baseline 비교용 동일 세팅
)


# ==================================================
# Train Setting
# ==================================================
epochs = 50

best_f1 = 0
best_metrics = {}


# ==================================================
# Training Loop
# ==================================================
print("\nTraining Start\n")

for epoch in range(epochs):

    model.train()
    losses = []

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    avg_loss = np.mean(losses)


    # ==================================================
    # Validation
    # ==================================================
    model.eval()

    preds = []
    labels = []

    start_time = time.time()

    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            batch_x = batch_x.to(device, non_blocking=True)

            outputs = model(batch_x)
            pred = torch.argmax(outputs, dim=1)

            preds.extend(pred.cpu().numpy())
            # 🚨 수정: pin_memory 사용에 따른 안전한 cpu() 이동 후 변환
            labels.extend(batch_y.cpu().numpy())

    inference_time = (time.time() - start_time) / 60


    # metrics
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro")


    # best update
    if f1 > best_f1:
        best_f1 = f1
        best_metrics = {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "time": inference_time
        }

    print(
        f"Epoch [{epoch+1:02d}/{epochs}] "
        f"| Loss {avg_loss:.4f} "
        f"| Accuracy {acc*100:.2f}% "
        f"| Precision {precision:.4f} "
        f"| Recall {recall:.4f} "
        f"| F1-score {f1:.4f} "
        f"| Inference Time  {inference_time:.4f} min"
    )


# ==================================================
# Result
# ==================================================
print("\n===================================")
print("EfficientNet-B0 Finished")
print("===================================")

print(f"Accuracy  : {best_metrics['acc']*100:.2f}%")
print(f"Precision : {best_metrics['precision']:.4f}")
print(f"Recall    : {best_metrics['recall']:.4f}")
print(f"F1-score  : {best_metrics['f1']:.4f}")
print(f"Inference Time  : {best_metrics['time']:.4f} min")


# ==================================================
# Save Model
# ==================================================
os.makedirs("results/models", exist_ok=True)

torch.save(
    model.state_dict(),
    "results/models/efficientnet_best.pth"
)

print("Saved: results/models/efficientnet_best.pth")

