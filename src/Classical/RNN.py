from src.Classical.create_docx import create_docx_RNN
from pandas import DataFrame
from pathlib import Path

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

def RNN(df:DataFrame, output_path:Path, name:str) -> None:
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

    class ClassicalRNN(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.hidden_size = 128
            self.num_layers = 2

            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=0.3
            )
            self.fc = nn.Linear(self.hidden_size, num_classes)

        def forward(self, x):
            x = x.unsqueeze(-1)
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out)

    # --- 2. RNN ---
    classical_rnn = ClassicalRNN(input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer_rnn = Adam(classical_rnn.parameters(), lr=1e-4, weight_decay=1e-4)

    # --- Loss Function ---
    loss_fn = CrossEntropyLoss(weight=class_weights)

    batch_size = 512
    epochs = 80

    optimizer_rnn = torch.optim.Adam(classical_rnn.parameters(), lr=1e-4, weight_decay=1e-4)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset  = TensorDataset(X_test_t,  y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    scheduler_rnn   = StepLR(optimizer_rnn, step_size=20, gamma=0.5)
    
    metrics = {
        'rnn': { 'train_acc': [], 'train_acc_best': [] },
    }

    best_rnn_acc = 0.0
    
    best_rnn_state = None
    
    best_rnn_epoch = 0
    
    for epoch in range(epochs):
        t0 = time.time()

        # 2. TRAIN RNN
        classical_rnn.train()
        all_preds_rnn = []
        all_targets_rnn = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer_rnn.zero_grad()
            out = classical_rnn(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer_rnn.step()

            preds = out.argmax(dim=1)
            all_preds_rnn.append(preds.detach().cpu())
            all_targets_rnn.append(yb.detach().cpu())

        y_pred_rnn = torch.cat(all_preds_rnn)
        y_true_rnn = torch.cat(all_targets_rnn)
        train_acc_rnn = (y_pred_rnn == y_true_rnn).float().mean().item()

        # Lưu Best RNN
        best_rnn_acc = max(best_rnn_acc, train_acc_rnn)
        if train_acc_rnn == best_rnn_acc:
            best_rnn_state = deepcopy(classical_rnn.state_dict())
            best_rnn_epoch = epoch + 1

        metrics['rnn']['train_acc'].append(train_acc_rnn)
        metrics['rnn']['train_acc_best'].append(best_rnn_acc)

        scheduler_rnn.step()

        t1 = time.time()
        print(f"Epoch {epoch + 1}/{epochs} - {t1 - t0:.1f}s")
        print(f"  RNN   - Acc: {train_acc_rnn:.4f} | Best: {best_rnn_acc:.4f}")
        print("-" * 60)

    if best_rnn_state is not None:
        classical_rnn.load_state_dict(best_rnn_state)

    print(f"Best RNN Accuracy   : {best_rnn_acc:.4f} tại epoch {best_rnn_epoch}")

    create_docx_RNN()