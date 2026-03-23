import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Minimal MLP for tabular classification.
# Topology:
# input -> Linear(16 hidden units) -> ReLU -> Linear(output_classes)
# ReLU adds non-linearity, the last layer returns logits.
# CrossEntropyLoss applies the softmax logic internally, so we do not add softmax in the model.
class Net(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_data(csv_path: str, batch_size: int = 32):
    df = pd.read_csv(csv_path)

    # Assume last column is the target, all previous columns are numeric features.
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    # Convert non-numeric feature columns if they exist.
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.mean(numeric_only=True))

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X.values,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X.shape[1], len(le.classes_), le.classes_


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            preds_all.extend(preds.cpu().numpy())
            targets_all.extend(yb.cpu().numpy())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    acc = accuracy_score(targets_all, preds_all)
    return avg_loss, acc, np.array(targets_all), np.array(preds_all)


def train_model(train_loader, val_loader, input_size, num_classes, epochs=50, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.detach().cpu().numpy())
            train_targets.extend(yb.detach().cpu().numpy())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        train_acc = accuracy_score(train_targets, train_preds)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

    return model, history, device, criterion


def plot_history(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    for train_key, val_key, ylabel, filename in [
        ("train_loss", "val_loss", "Loss", "diagnosis_loss.png"),
        ("train_acc", "val_acc", "Accuracy", "diagnosis_accuracy.png"),]:
        plt.figure()
        plt.plot(epochs, history[train_key], label="train")
        plt.plot(epochs, history[val_key], label="val")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()

def main():
    csv_path = "diagnosis.csv"
    epochs = 50
    batch_size = 32
    lr = 0.001
    out_dir = "outputs_diagnosis"

    train_loader, val_loader, input_size, num_classes, class_names = load_data(csv_path, batch_size)
    model, history, device, criterion = train_model(
        train_loader, val_loader, input_size, num_classes, epochs=epochs, lr=lr
    )

    plot_history(history, out_dir)

    val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    cm = confusion_matrix(y_true, y_pred)

    average = "binary" if num_classes == 2 else "macro"
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)

    print(f"loss: {val_loss:.4f}")
    print(f"accuracy: {val_acc:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print("confusion matrix:")
    print(cm)

if __name__ == "__main__":
    main()
