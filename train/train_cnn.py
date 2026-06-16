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

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from models.cnn import CNNBaseline
from utils.preprocess import preprocess_p2


class WaferDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = preprocess_p2(self.X[idx])

        x = np.expand_dims(
            x,
            axis=0
        )

        return (
            torch.tensor(
                x,
                dtype=torch.float32
            ),
            torch.tensor(
                self.y[idx],
                dtype=torch.long
            )
        )


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device :", device)

# ==================================================
# Dataset
# ==================================================

df = pd.read_pickle(
    "LSWMD_clean_64.pkl"
)

classes = [
    "none",
    "Loc",
    "Edge-Loc",
    "Center",
    "Edge-Ring",
    "Scratch",
    "Random",
    "Near-full",
    "Donut"
]

label_map = {
    c: i
    for i, c in enumerate(classes)
}

df["label_idx"] = df["clean_label"].map(
    label_map
)

X = np.stack(
    df["waferMap_scaled"].values
)

y = df["label_idx"].values

# ==================================================
# Split
# ==================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

train_loader = DataLoader(
    WaferDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    WaferDataset(X_val, y_val),
    batch_size=32,
    shuffle=False
)

# ==================================================
# Model
# ==================================================

model = CNNBaseline(
    num_classes=9
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001
)

epochs = 50

best_acc = 0
best_precision = 0
best_recall = 0
best_f1 = 0
best_inference_time = 0

print("\nTraining Start\n")

for epoch in range(epochs):

    model.train()

    losses = []

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        losses.append(
            loss.item()
        )

    avg_loss = np.mean(losses)

    model.eval()

    preds = []
    labels = []

    start_time = time.time()

    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            batch_x = batch_x.to(device)

            outputs = model(batch_x)

            pred = torch.argmax(
                outputs,
                dim=1
            )

            preds.extend(
                pred.cpu().numpy()
            )

            # 수정 사항 적용: 안전하게 cpu() 이동 후 numpy 변환
            labels.extend(
                batch_y.cpu().numpy()
            )

    inference_time = (
        time.time() - start_time
    ) / 60

    acc = accuracy_score(
        labels,
        preds
    )

    precision = precision_score(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        labels,
        preds,
        average="macro"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_acc = acc
        best_precision = precision
        best_recall = recall
        best_inference_time = inference_time

    print(
        f"Epoch [{epoch+1:02d}/{epochs}] "
        f"| Loss {avg_loss:.4f} "
        f"| Accuracy {acc*100:.2f}% "
        f"| Precision {precision:.4f} "
        f"| Recall {recall:.4f} "
        f"| F1-score {f1:.4f} "
        f"| Inference Time {inference_time:.4f} min"
    )

print("\n===================================")
print("CNN Training Finished")
print("===================================")
print(f"Accuracy : {best_acc*100:.2f}%")
print(f"Precision : {best_precision:.4f}")
print(f"Recall : {best_recall:.4f}")
print(f"F1-score : {best_f1:.4f}")
print(f"Inference Time : {best_inference_time:.4f} min")

os.makedirs(
    "results/models",
    exist_ok=True
)

torch.save(
    model.state_dict(),
    "results/models/cnn_best.pth"
)

print("Saved : results/models/cnn_best.pth")

