from pandas import DataFrame
from pathlib import Path
from src.Classical.create_docx import create_docx_transformer

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

def transformer(df:DataFrame, output_path:Path, name:str) -> None:
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

    class ClassicalTransformer(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.d_model = 64       
            self.nhead = 4          
            self.num_layers = 2      
            self.dim_feedforward = 128 

            self.input_embedding = nn.Linear(1, self.d_model)

            self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, self.d_model))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                batch_first=True,
                dropout=0.3
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

            self.fc = nn.Linear(self.d_model, num_classes)

        def forward(self, x):
            x = x.unsqueeze(-1)

            x = self.input_embedding(x)

            x = x + self.pos_embedding

            x = self.transformer_encoder(x)

            x = x.mean(dim=1)

            return self.fc(x)

    classical_transformer = ClassicalTransformer(input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer_transformer = Adam(classical_transformer.parameters(), lr=1e-4, weight_decay=1e-4)

    loss_fn = CrossEntropyLoss(weight=class_weights)
    
    batch_size = 512
    epochs = 80

    optimizer_trans = torch.optim.Adam(classical_transformer.parameters(), lr=1e-4, weight_decay=1e-4)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset  = TensorDataset(X_test_t,  y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    scheduler_trans = StepLR(optimizer_trans, step_size=20, gamma=0.5)

    metrics = {
        'trans': { 'train_acc': [], 'train_acc_best': [] }
    }

    best_trans_acc = 0.0

    best_trans_state = None

    best_trans_epoch = 0

    for epoch in range(epochs):
        t0 = time.time()
        
        classical_transformer.train()
        all_preds_trans = []
        all_targets_trans = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer_trans.zero_grad()
            out = classical_transformer(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer_trans.step()

            preds = out.argmax(dim=1)
            all_preds_trans.append(preds.detach().cpu())
            all_targets_trans.append(yb.detach().cpu())

        y_pred_trans = torch.cat(all_preds_trans)
        y_true_trans = torch.cat(all_targets_trans)
        train_acc_trans = (y_pred_trans == y_true_trans).float().mean().item()

        best_trans_acc = max(best_trans_acc, train_acc_trans)
        if train_acc_trans == best_trans_acc:
            best_trans_state = deepcopy(classical_transformer.state_dict())
            best_trans_epoch = epoch + 1

        metrics['trans']['train_acc'].append(train_acc_trans)
        metrics['trans']['train_acc_best'].append(best_trans_acc)

        scheduler_trans.step()

        t1 = time.time()
        print(f"Epoch {epoch + 1}/{epochs} - {t1 - t0:.1f}s")
        print(f"  Trans - Acc: {train_acc_trans:.4f} | Best: {best_trans_acc:.4f}")
        print("-" * 60)

    if best_trans_state is not None:
        classical_transformer.load_state_dict(best_trans_state)

    print(f"Best Trans Accuracy : {best_trans_acc:.4f} tại epoch {best_trans_epoch}")

    create_docx_transformer()