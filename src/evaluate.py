import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.preprocess import get_dataloaders
from src.cnn_model import CNNClassifier


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', save_path=None):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="./data/devanagari_dataset/Train")
    parser.add_argument("--test_dir", type=str, default="./data/devanagari_dataset/Test")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weights", type=str, default="./artifacts/cnn_best.pt")
    parser.add_argument("--save_cm", type=str, default="./artifacts/confusion_matrix.png")
    args = parser.parse_args()

    device = torch.device(args.device)

    train_loader, test_loader, class_to_idx, idx_to_class = get_dataloaders(
        train_dir=args.train_dir, test_dir=args.test_dir, batch_size=args.batch_size
    )
    num_classes = len(class_to_idx)

    model = CNNClassifier(num_classes=num_classes).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(yb.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    # Confusion matrix and report
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=[k for k,_ in sorted(class_to_idx.items(), key=lambda x: x[1])])
    print(report)

    os.makedirs(os.path.dirname(args.save_cm), exist_ok=True)
    plot_confusion_matrix(cm, classes=list(class_to_idx.keys()), save_path=args.save_cm)


if __name__ == "__main__":
    main()
