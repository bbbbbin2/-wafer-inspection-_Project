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
