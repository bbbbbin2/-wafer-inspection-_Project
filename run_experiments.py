import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import os

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from models.cnn import CNNBaseline
from utils import preprocess as P


# Dataset
class ExperimentWaferDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = np.expand_dims(self.X[idx], axis=0)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long)
        )


# Environment
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n============================================================")
print("Wafer Defect Classification")
print("Preprocessing Method Comparison")
print("============================================================")
print("Device :", device)
print("============================================================")


# Load Dataset
df = pd.read_pickle("LSWMD_clean_64.pkl")

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

df["label_idx"] = df["clean_label"].map(label_map)

X = np.stack(df["waferMap_scaled"].values)
y = df["label_idx"].values


# Apply Preprocessing
def build_dataset(X_data, y_data, preprocess_fn):


    X_new = np.array(
        [preprocess_fn(img) for img in X_data],
        dtype=np.float32
    )

    return ExperimentWaferDataset(X_new, y_data)


# Training & Evaluation
def run_single_experiment(
    exp_name,
    preprocess_fn,
    use_class_weight=False,
    epochs=15
):

    print("==============================================================")
    print("Preprocessing Method")
    print(exp_name)
    print("============================================================")

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

    train_set = build_dataset(
        X_train,
        y_train,
        preprocess_fn
    )

    val_set = build_dataset(
        X_val,
        y_val,
        preprocess_fn
    )

    train_loader = DataLoader(
        train_set,
        batch_size=128,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=128,
        shuffle=False
    )

    model = CNNBaseline(
        num_classes=9
    ).to(device)

    if use_class_weight:

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train
        )

        class_weights = torch.tensor(
            class_weights,
            dtype=torch.float32
        ).to(device)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

    else:

        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    best_acc = 0
    best_f1 = 0

    total_start = time.time()

    for epoch in range(epochs):

        epoch_start = time.time()

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

            losses.append(loss.item())

        avg_loss = np.mean(losses)

        model.eval()

        preds = []
        labels = []

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

                labels.extend(
                    batch_y.numpy()
                )

        acc = accuracy_score(
            labels,
            preds
        )

        f1 = f1_score(
            labels,
            preds,
            average="macro"
        )

        best_acc = max(best_acc, acc)
        best_f1 = max(best_f1, f1)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch+1:02d}/15] "
            f"| Loss {avg_loss:.4f} "
            f"| Accuracy {acc*100:.2f}% "
            f"| F1-score {f1:.4f} "
            f"| Time {epoch_time:.1f}s"
        )

    total_time = (
        time.time() - total_start
    ) / 60

    print("\nPerformance Summary")
    print("--------------------------------------")
    print(f"Best Accuracy : {best_acc*100:.2f}%")
    print(f"Best F1-score : {best_f1:.4f}")
    print(f"Training Time : {total_time:.2f} min")
    print("--------------------------------------")

    return best_acc, best_f1, total_time


# Preprocessing Experiments
experiments = [

    (
        "P2 - Resize + Normalization (Baseline)",
        P.preprocess_p2,
        False
    ),

    (
        "P3 - Resize + Median Filter + Normalization (Median Filter)",
        P.preprocess_p3,
        False
    ),

    (
        "P4 - Resize + Data Augmentation + Normalization (Augmentation)",
        P.preprocess_p4,
        False
    ),

    (
        "P5 - Resize + Normalization + Balanced Class Weight (Class Weight)",
        P.preprocess_p5,
        True
    )
]


# Run Experiments
results = []

for name, preprocess_fn, use_weight in experiments:

    acc, f1, train_time = run_single_experiment(
        name,
        preprocess_fn,
        use_weight,
        epochs=15
    )

    results.append([
        name,
        acc,
        f1,
        train_time
    ])


# Performance Comparison
result_df = pd.DataFrame(
    results,
    columns=[
        "Preprocessing_Method",
        "Best_Accuracy",
        "Best_F1_Score",
        "Training_Time(min)"
    ]
)

print("\n============================================================")
print("Preprocessing Performance Comparison")
print("============================================================")

print(
    result_df.to_string(
        index=False
    )
)

os.makedirs(
    "results/exp_compare",
    exist_ok=True
)

result_df.to_csv(
    "results/exp_compare/p2_p3_p4_p5_results.csv",
    index=False
)

print(
    "\nSaved : results/exp_compare/p2_p3_p4_p5_results.csv"
)

best_idx = result_df[
    "Best_F1_Score"
].idxmax()

print("\n============================================================")
print("Best Preprocessing Strategy")
print("============================================================")

print(
    result_df.iloc[best_idx]
)

print("============================================================")
