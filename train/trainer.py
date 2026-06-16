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
