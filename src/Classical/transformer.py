from src.Classical.create_docx import create_docx_RNN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

import time
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy

# Chọn device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device đang dùng:", device)

# ===== 1. Lấy df_scaled & y từ Cell 3, encode nhãn =====
X = df_scaled.astype(np.float32)      # (N, D)
y_arr = np.array(y)                   # nhãn dạng string / số gốc

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_arr)
num_classes = len(np.unique(y_encoded))
print("Số lớp:", num_classes)

# ===== 2. Split train/test (không dùng val, giống repo MIoT) =====
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded
)

print("Train shape:", X_train.shape)
print("Test  shape:", X_test.shape)

# ===== 3. Chuyển sang Tensor, đưa lên device =====
# Dữ liệu này dùng chung cho cả 3 mô hình
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(device)

y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_t  = torch.tensor(y_test,  dtype=torch.long).to(device)

input_dim = X_train.shape[1]
print("input_dim (Features):", input_dim)

# ===== 4. Class weights trên train =====
class_counts = np.bincount(y_train)
class_weights = class_counts.sum() / (class_counts + 1e-6)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights:", class_weights.cpu().numpy())

# ==========================================
# 5.1 ĐỊNH NGHĨA CLASSICAL CNN (1D)
# ==========================================
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

# ==========================================
# 5.2 ĐỊNH NGHĨA CLASSICAL RNN (LSTM)
# ==========================================
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

# ==========================================
# 5.3 ĐỊNH NGHĨA TRANSFORMER (Encoder Only) [MỚI]
# ==========================================
class ClassicalTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Cấu hình Transformer
        self.d_model = 64        # Kích thước vector embedding (phải chia hết cho nhead)
        self.nhead = 4           # Số lượng attention heads
        self.num_layers = 2      # Số lớp Encoder
        self.dim_feedforward = 128 # Kích thước lớp ẩn trong FFN

        # 1. Embedding: Chiếu giá trị scalar (1 feature) lên vector d_model
        self.input_embedding = nn.Linear(1, self.d_model)

        # 2. Positional Encoding: Học vị trí của từng feature
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, self.d_model))

        # 3. Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            batch_first=True, # Quan trọng: input dạng (Batch, Seq, Feature)
            dropout=0.3
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # 4. Classifier
        self.fc = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        # x shape: (Batch, Input_Dim)

        # Tạo sequence: (Batch, Input_Dim, 1)
        x = x.unsqueeze(-1)

        # Embedding: (Batch, Input_Dim, d_model)
        x = self.input_embedding(x)

        # Cộng Positional Encoding
        x = x + self.pos_embedding

        # Qua Transformer Encoder
        # Output: (Batch, Input_Dim, d_model)
        x = self.transformer_encoder(x)

        # Global Average Pooling (Lấy trung bình tất cả các đặc trưng)
        # Output: (Batch, d_model)
        x = x.mean(dim=1)

        return self.fc(x)

# ==========================================
# 6. KHỞI TẠO 3 MODELS & OPTIMIZERS
# ==========================================

# --- 1. CNN ---
classical_cnn = ClassicalCNN(input_dim=input_dim, num_classes=num_classes).to(device)
optimizer_cnn = Adam(classical_cnn.parameters(), lr=1e-4, weight_decay=1e-4)

# --- 2. RNN ---
classical_rnn = ClassicalRNN(input_dim=input_dim, num_classes=num_classes).to(device)
optimizer_rnn = Adam(classical_rnn.parameters(), lr=1e-4, weight_decay=1e-4)

# --- 3. TRANSFORMER --- [MỚI]
classical_transformer = ClassicalTransformer(input_dim=input_dim, num_classes=num_classes).to(device)
optimizer_transformer = Adam(classical_transformer.parameters(), lr=1e-4, weight_decay=1e-4)

# --- Loss Function ---
loss_fn = CrossEntropyLoss(weight=class_weights)

print("Hoàn thành Cell 5 – Đã cấu hình CNN, RNN và Transformer.")



print("Training trên:", device)

# ===== 0. Cấu hình Hyperparameters =====
batch_size = 512
epochs = 80

# Re-init optimizer cho CẢ BA mô hình
# (Đảm bảo classical_cnn, classical_rnn, classical_transformer đã có từ Cell 5)
optimizer_cnn = torch.optim.Adam(classical_cnn.parameters(), lr=1e-4, weight_decay=1e-4)
optimizer_rnn = torch.optim.Adam(classical_rnn.parameters(), lr=1e-4, weight_decay=1e-4)
optimizer_trans = torch.optim.Adam(classical_transformer.parameters(), lr=1e-4, weight_decay=1e-4)

# Loss_fn dùng lại từ Cell 5
# loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# ===== 1. Tạo DataLoader =====
# Tất cả đều dùng chung dữ liệu đầu vào (X_train_t)
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# Scheduler cho cả ba
scheduler_cnn   = StepLR(optimizer_cnn, step_size=20, gamma=0.5)
scheduler_rnn   = StepLR(optimizer_rnn, step_size=20, gamma=0.5)
scheduler_trans = StepLR(optimizer_trans, step_size=20, gamma=0.5)

# ===== 2. Struct lưu metrics =====
metrics = {
    'cnn': { 'train_acc': [], 'train_acc_best': [] },
    'rnn': { 'train_acc': [], 'train_acc_best': [] },
    'trans': { 'train_acc': [], 'train_acc_best': [] }
}

best_cnn_acc = 0.0
best_rnn_acc = 0.0
best_trans_acc = 0.0

best_cnn_state = None
best_rnn_state = None
best_trans_state = None

best_cnn_epoch = 0
best_rnn_epoch = 0
best_trans_epoch = 0

# ===== VÒNG LẶP HUẤN LUYỆN =====
for epoch in range(epochs):
    t0 = time.time()

    # ==========================
    # 1. TRAIN CNN
    # ==========================
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

    # ==========================
    # 2. TRAIN RNN
    # ==========================
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

    # ==========================
    # 3. TRAIN TRANSFORMER
    # ==========================
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

    # Lưu Best Transformer
    best_trans_acc = max(best_trans_acc, train_acc_trans)
    if train_acc_trans == best_trans_acc:
        best_trans_state = deepcopy(classical_transformer.state_dict())
        best_trans_epoch = epoch + 1

    metrics['trans']['train_acc'].append(train_acc_trans)
    metrics['trans']['train_acc_best'].append(best_trans_acc)

    # ==========================
    # Cập nhật Scheduler & In kết quả
    # ==========================
    scheduler_cnn.step()
    scheduler_rnn.step()
    scheduler_trans.step()

    t1 = time.time()
    print(f"Epoch {epoch + 1}/{epochs} - {t1 - t0:.1f}s")
    print(f"  CNN   - Acc: {train_acc_cnn:.4f} | Best: {best_cnn_acc:.4f}")
    print(f"  RNN   - Acc: {train_acc_rnn:.4f} | Best: {best_rnn_acc:.4f}")
    print(f"  Trans - Acc: {train_acc_trans:.4f} | Best: {best_trans_acc:.4f}")
    print("-" * 60)

# ===== 3. Load lại best weights cho CẢ BA mô hình =====
if best_cnn_state is not None:
    classical_cnn.load_state_dict(best_cnn_state)
if best_rnn_state is not None:
    classical_rnn.load_state_dict(best_rnn_state)
if best_trans_state is not None:
    classical_transformer.load_state_dict(best_trans_state)

print("\nHOÀN THÀNH TRAINING!")
print(f"Best CNN Accuracy   : {best_cnn_acc:.4f} tại epoch {best_cnn_epoch}")
print(f"Best RNN Accuracy   : {best_rnn_acc:.4f} tại epoch {best_rnn_epoch}")
print(f"Best Trans Accuracy : {best_trans_acc:.4f} tại epoch {best_trans_epoch}")