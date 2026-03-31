from pandas import DataFrame
from pathlib import Path
from src.Classical.create_docx import create_docx_CNN

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

import time
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy

def CNN(df:DataFrame, output_path:Path, name:str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = df_scaled.astype(np.float32)
    y_arr = np.array(y)                  

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_arr)
    num_classes = len(np.unique(y_encoded))
    print("Số lớp:", num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    print("Train shape:", X_train.shape)
    print("Test  shape:", X_test.shape)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(device)

    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long).to(device)

    input_dim = X_train.shape[1]
    print("input_dim (Features):", input_dim)

    class_counts = np.bincount(y_train)
    class_weights = class_counts.sum() / (class_counts + 1e-6)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Class weights:", class_weights.cpu().numpy())

    class ClassicalCNN(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 128, kernel_size=3, padding=1)
            self.bn1   = nn.BatchNorm1d(128)
            self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
            self.bn2   = nn.BatchNorm1d(256)
            self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
            self.bn3   = nn.BatchNorm1d(512)
            self.pool  = nn.MaxPool1d(2)
            self.dropout = nn.Dropout(0.5)

            conv_out_len = input_dim // 8
            if conv_out_len == 0: conv_out_len = 1

            self.fc1 = nn.Linear(512 * conv_out_len, 512)
            self.fc2 = nn.Linear(512, num_classes)

        def forward(self, x):
            x = x.unsqueeze(1)
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(F.relu(self.fc1(x)))
            return self.fc2(x)

    # --- 1. CNN ---
    classical_cnn = ClassicalCNN(input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer_cnn = Adam(classical_cnn.parameters(), lr=1e-4, weight_decay=1e-4)

    loss_fn = CrossEntropyLoss(weight=class_weights)

    batch_size = 512
    epochs = 80

    optimizer_cnn = torch.optim.Adam(classical_cnn.parameters(), lr=1e-4, weight_decay=1e-4)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset  = TensorDataset(X_test_t,  y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    scheduler_cnn   = StepLR(optimizer_cnn, step_size=20, gamma=0.5)

    metrics = {
        'cnn': { 'train_acc': [], 'train_acc_best': [] },
    }

    best_cnn_acc = 0.0

    best_cnn_state = None

    best_cnn_epoch = 0

    for epoch in range(epochs):
        t0 = time.time()

        classical_cnn.train()
        all_preds_cnn = []
        all_targets_cnn = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer_cnn.zero_grad()
            out = classical_cnn(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer_cnn.step()

            preds = out.argmax(dim=1)
            all_preds_cnn.append(preds.detach().cpu())
            all_targets_cnn.append(yb.detach().cpu())

        y_pred_cnn = torch.cat(all_preds_cnn)
        y_true_cnn = torch.cat(all_targets_cnn)
        train_acc_cnn = (y_pred_cnn == y_true_cnn).float().mean().item()

        # Lưu Best CNN
        best_cnn_acc = max(best_cnn_acc, train_acc_cnn)
        if train_acc_cnn == best_cnn_acc:
            best_cnn_state = deepcopy(classical_cnn.state_dict())
            best_cnn_epoch = epoch + 1

        metrics['cnn']['train_acc'].append(train_acc_cnn)
        metrics['cnn']['train_acc_best'].append(best_cnn_acc)

        scheduler_cnn.step()

        t1 = time.time()
        print(f"Epoch {epoch + 1}/{epochs} - {t1 - t0:.1f}s")
        print(f"  CNN   - Acc: {train_acc_cnn:.4f} | Best: {best_cnn_acc:.4f}")
        print("-" * 60)

    if best_cnn_state is not None:
        classical_cnn.load_state_dict(best_cnn_state)

    print(f"Best CNN Accuracy   : {best_cnn_acc:.4f} tại epoch {best_cnn_epoch}")

    create_docx_CNN()