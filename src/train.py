import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import joblib

from src.preprocess import get_dataloaders, count_classes
from src.cnn_model import CNNClassifier
from src.pca_qnn_model import PCAReducer, QNNClassifier


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, pred = logits.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            _, pred = logits.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    return running_loss / total, correct / total


def extract_features(model: CNNClassifier, loader: DataLoader, device: torch.device):
    model.eval()
    feats_list, labels_list = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extract", leave=False):
            x = x.to(device)
            feats = model.backbone(x)
            feats_list.append(feats.cpu())
            labels_list.append(y)
    X = torch.cat(feats_list, dim=0)
    y = torch.cat(labels_list, dim=0)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="./data/devanagari_dataset/Train")
    parser.add_argument("--test_dir", type=str, default="./data/devanagari_dataset/Test")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="./artifacts")

    # PCA + QNN params
    parser.add_argument("--pca_components", type=int, default=8)
    parser.add_argument("--qnn_layers", type=int, default=2)
    parser.add_argument("--qnn_epochs", type=int, default=5)
    parser.add_argument("--qnn_lr", type=float, default=1e-3)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    # Data
    train_loader, test_loader, class_to_idx, idx_to_class = get_dataloaders(
        train_dir=args.train_dir, test_dir=args.test_dir, batch_size=args.batch_size
    )
    num_classes = len(class_to_idx)

    # Stage 1: Train CNN
    model = CNNClassifier(num_classes=num_classes, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Stage 1: Training CNN on {num_classes} classes for {args.epochs} epochs")
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)
        print(f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | test_loss={te_loss:.4f} test_acc={te_acc:.4f}")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "cnn_best.pt"))

    torch.save(model.state_dict(), os.path.join(args.save_dir, "cnn_last.pt"))

    # Stage 2: PCA + QNN on CNN features (freeze CNN)
    print("Stage 2: PCA + QNN")
    for p in model.parameters():
        p.requires_grad = False

    X_train, y_train = extract_features(model, train_loader, device)
    X_test, y_test = extract_features(model, test_loader, device)

    pca = PCAReducer(n_components=args.pca_components)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)

    # QNN classifier
    qnn = QNNClassifier(n_qubits=args.pca_components, n_layers=args.qnn_layers, n_classes=num_classes, device=device).to(device)
    qnn_criterion = nn.CrossEntropyLoss()
    qnn_opt = optim.Adam(qnn.parameters(), lr=args.qnn_lr)

    train_ds_q = torch.utils.data.TensorDataset(Z_train, y_train)
    test_ds_q = torch.utils.data.TensorDataset(Z_test, y_test)
    train_loader_q = DataLoader(train_ds_q, batch_size=args.batch_size, shuffle=True)
    test_loader_q = DataLoader(test_ds_q, batch_size=args.batch_size, shuffle=False)

    for epoch in range(1, args.qnn_epochs + 1):
        qnn.train()
        tr_loss, correct, total = 0.0, 0, 0
        for xb, yb in tqdm(train_loader_q, desc=f"QNN Train {epoch:02d}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            qnn_opt.zero_grad()
            logits = qnn(xb)
            loss = qnn_criterion(logits, yb)
            loss.backward()
            qnn_opt.step()
            tr_loss += loss.item() * xb.size(0)
            _, pred = logits.max(1)
            correct += pred.eq(yb).sum().item()
            total += yb.size(0)
        tr_acc = correct / total

        # eval
        qnn.eval()
        te_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in tqdm(test_loader_q, desc=f"QNN Eval {epoch:02d}", leave=False):
                xb, yb = xb.to(device), yb.to(device)
                logits = qnn(xb)
                loss = qnn_criterion(logits, yb)
                te_loss += loss.item() * xb.size(0)
                _, pred = logits.max(1)
                correct += pred.eq(yb).sum().item()
                total += yb.size(0)
        te_acc = correct / total

        print(f"QNN Epoch {epoch:02d}: train_loss={tr_loss/len(train_ds_q):.4f} train_acc={tr_acc:.4f} | test_loss={te_loss/len(test_ds_q):.4f} test_acc={te_acc:.4f}")

    # Save artifacts
    joblib.dump({
        "pca_components": args.pca_components,
        "pca": pca.pca,
    }, os.path.join(args.save_dir, "pca.joblib"))
    torch.save(qnn.state_dict(), os.path.join(args.save_dir, "qnn_last.pt"))


if __name__ == "__main__":
    main()
