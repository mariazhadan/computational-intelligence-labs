from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def load_data(csv_path: str, batch_size: int = 32, val_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(csv_path)

    X = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y
    )

    # Normalization fitted only on training data.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X.shape[1], len(le.classes_), le


class IrisNet(nn.Module):
    """
    Minimal MLP for Iris classification.
    Topology: input -> Linear(16) -> ReLU -> Linear(num_classes).
    Hidden ReLU introduces non-linearity, final layer returns logits for CrossEntropyLoss.
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(), #funkcja aktywacji
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc, np.array(all_targets), np.array(all_preds)


def train(model, train_loader, val_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad() #pythorcz prechowuje gradienty
            logits = model(X)
            loss = criterion(logits, y) # otrzymujemy wejście i loss
            loss.backward() #backpropagation
            optimizer.step() #zmiana wag

            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += X.size(0) # statystyki ile jest zgadano? 

        train_loss = total_loss / total
        train_acc = correct / total
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
        )

    return history

def plot_history(history, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    for train_key, val_key, name in [
        ("train_loss", "val_loss", "loss"),
        ("train_acc", "val_acc", "accuracy"),
    ]:
        plt.figure()
        plt.plot(history[train_key], label="train")
        plt.plot(history[val_key], label="val")
        plt.legend()
        plt.savefig(out_dir / f"{name}.png")
        plt.close()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, in_features, num_classes, _ = load_data("iris_big.csv", 16)

    model = IrisNet(in_features, num_classes).to(device)
    history = train(model, train_loader, val_loader, 50, 0.001, device)
    plot_history(history, Path("iris_outputs"))

    val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
    print(f"accuracy = {val_acc:.4f}")
    print(confusion_matrix(y_true, y_pred))
    print("good result" if val_acc > 0.8 else "result could be better")


if __name__ == "__main__":
    main()