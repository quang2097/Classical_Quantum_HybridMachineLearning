
import os
import json
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
except ImportError as e:
    raise ImportError(
        "This script needs torch_geometric. "
        "Install the matching PyG wheels for your exact PyTorch/CUDA version, "
        "then re-run."
    ) from e


# =========================
# Configuration
# =========================
CSV_FILES = {
    "backdoor": "backdoor.csv",
    "normal": "benign_tot.csv",
    "ddos": "ddos.csv",
    "dos": "dos.csv",
    "injection": "injection.csv",
    "mitm": "mitm.csv",
    "password": "password.csv",
    "ransomware": "ransomware.csv",
    "scanning": "scanning.csv",
    "xss": "xss.csv",
}

DATA_DIR = Path(".")
OUTPUT_DIR = Path("./gnn_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DROP_DUPLICATES = True

# "multiclass" -> target = type
# "binary"     -> target = label
TARGET_MODE = "multiclass"

# Graph construction:
# We do NOT use exact kNN over all rows because this dataset is large.
# Instead, we create edges between flows that share meaningful cyber entities.
EDGE_KEY_SPECS = {
    "src_ip": ["src_ip"],
    "dst_ip_port": ["dst_ip", "dst_port"],
    "service_state": ["service", "conn_state"],
    "dns_query": ["dns_query"],
    # You can uncomment this if HTTP activity matters more in your experiment:
    # "http_uri": ["http_uri"],
}
EDGE_WINDOW = 1  # link each row to the next row(s) inside the same group

HIDDEN_DIM = 128
DROPOUT = 0.30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Utilities
# =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def stable_hash_mod(x: str, mod: int = 1_000_003) -> int:
    return int(hashlib.md5(str(x).encode("utf-8")).hexdigest(), 16) % mod


def signed_log1p(X):
    X = np.asarray(X, dtype=np.float64)
    return np.sign(X) * np.log1p(np.abs(X))


def make_key(df: pd.DataFrame, cols):
    s = df[cols[0]].astype(str)
    for c in cols[1:]:
        s = s + "||" + df[c].astype(str)
    return s


# =========================
# 1) Load + concatenate
# =========================
def load_and_concat(data_dir: Path, csv_files: dict, drop_duplicates: bool = True):
    frames = []
    summary_rows = []

    for alias, filename in csv_files.items():
        path = data_dir / filename
        df = pd.read_csv(path)

        summary_rows.append({
            "file": filename,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "duplicate_rows_inside_file": int(df.duplicated().sum()),
            "null_cells": int(df.isna().sum().sum()),
            "unique_label_values": sorted(df["label"].unique().tolist()),
            "unique_type_values": sorted(df["type"].unique().tolist()),
        })

        df["source_file"] = filename
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    before = len(all_df)
    if drop_duplicates:
        all_df = all_df.drop_duplicates().reset_index(drop=True)
    after = len(all_df)

    summary_df = pd.DataFrame(summary_rows)
    return all_df, summary_df, before, after


# =========================
# 2) Compact feature engineering for tabular flows
# =========================
def engineer_features(df: pd.DataFrame):
    df = df.copy()

    numeric_cols_base = [
        "src_port", "dst_port", "duration", "src_bytes", "dst_bytes",
        "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts",
        "dst_ip_bytes", "dns_qclass", "dns_qtype", "dns_rcode",
        "http_request_body_len", "http_response_body_len",
        "http_status_code",
    ]

    ip_cols = ["src_ip", "dst_ip"]

    binary_flag_cols = [
        "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
        "ssl_resumed", "ssl_established", "weird_notice",
    ]

    low_card_cols = [
        "proto", "service", "conn_state", "ssl_version", "ssl_cipher",
        "http_trans_depth", "http_method", "http_version",
        "http_orig_mime_types", "http_resp_mime_types",
        "weird_name", "weird_addl",
    ]

    sparse_text_cols = [
        "dns_query", "http_uri", "http_user_agent", "ssl_subject", "ssl_issuer",
    ]

    # --- IP features ---
    # We avoid direct IP integer conversion because some rows contain IPv6.
    for c in ip_cols:
        s = df[c].astype(str)
        vc = s.value_counts(dropna=False)
        df[f"{c}_freq"] = s.map(vc).astype(np.int32)
        df[f"{c}_len"] = s.str.len().astype(np.int16)
        df[f"{c}_is_ipv6"] = s.str.contains(":").astype(np.int8)
        df[f"{c}_hash"] = s.map(stable_hash_mod).astype(np.int32)

    # --- T/F/- flags ---
    flag_map = {"T": 1, "F": 0, "-": -1}
    for c in binary_flag_cols:
        df[f"{c}_num"] = df[c].astype(str).map(flag_map).fillna(-1).astype(np.int8)

    # --- Sparse text meta-features ---
    # We do not explode them with huge one-hot vocabularies.
    # Instead we keep compact signals: length, placeholder flag, frequency.
    for c in sparse_text_cols:
        s = df[c].astype(str)
        vc = s.value_counts(dropna=False)
        df[f"{c}_len"] = s.where(s != "-", "").str.len().astype(np.int32)
        df[f"{c}_is_dash"] = (s == "-").astype(np.int8)
        df[f"{c}_freq"] = s.map(vc).astype(np.int32)

    engineered_numeric = (
        numeric_cols_base
        + [f"{c}_freq" for c in ip_cols]
        + [f"{c}_len" for c in ip_cols]
        + [f"{c}_is_ipv6" for c in ip_cols]
        + [f"{c}_hash" for c in ip_cols]
        + [f"{c}_num" for c in binary_flag_cols]
        + [f"{c}_len" for c in sparse_text_cols]
        + [f"{c}_is_dash" for c in sparse_text_cols]
        + [f"{c}_freq" for c in sparse_text_cols]
    )

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="-")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, engineered_numeric),
        ("cat", cat_pipe, low_card_cols),
    ])

    X = preprocessor.fit_transform(df[engineered_numeric + low_card_cols]).astype(np.float32)

    if TARGET_MODE == "multiclass":
        target_values = df["type"].astype(str).values
    else:
        target_values = df["label"].astype(int).values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(target_values).astype(np.int64)

    feature_info = {
        "num_engineered_numeric": len(engineered_numeric),
        "num_low_card_categorical": len(low_card_cols),
        "final_feature_dim": int(X.shape[1]),
        "classes": label_encoder.classes_.tolist(),
    }

    return df, X, y, label_encoder, preprocessor, feature_info


# =========================
# 3) Build homogeneous graph from shared cyber entities
# =========================
def build_edge_index(df: pd.DataFrame, edge_key_specs: dict, window: int = 1):
    src_parts = []
    dst_parts = []

    for relation_name, cols in edge_key_specs.items():
        key = make_key(df, cols)
        groups = key.groupby(key, sort=False).groups

        for key_value, idx in groups.items():
            if relation_name in {"dns_query", "http_uri"} and str(key_value) == "-":
                continue

            idx = np.asarray(list(idx), dtype=np.int64)
            n = idx.size
            if n < 2:
                continue

            for w in range(1, window + 1):
                if n > w:
                    src_parts.append(idx[:-w])
                    dst_parts.append(idx[w:])
                    src_parts.append(idx[w:])
                    dst_parts.append(idx[:-w])

    if not src_parts:
        raise ValueError("No edges were created. Check edge definitions or preprocessing.")

    edge_index = np.vstack([
        np.concatenate(src_parts),
        np.concatenate(dst_parts),
    ])

    # Remove self-loops and duplicate edges
    keep = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, keep]

    edge_df = pd.DataFrame({
        "src": edge_index[0],
        "dst": edge_index[1],
    }).drop_duplicates()

    edge_index = edge_df[["src", "dst"]].to_numpy(dtype=np.int64).T
    return edge_index


# =========================
# 4) Train/val/test split
# =========================
def build_masks(y: np.ndarray, seed: int = 42):
    indices = np.arange(len(y))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=2/3,   # 10% val, 20% test overall
        random_state=seed,
        stratify=y[temp_idx],
    )

    train_mask = np.zeros(len(y), dtype=bool)
    val_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask, train_idx, val_idx, test_idx


# =========================
# 5) GNN model
# =========================
class FlowGraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits[mask].argmax(dim=1).cpu().numpy()
    true = data.y[mask].cpu().numpy()

    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro")),
        "weighted_f1": float(f1_score(true, pred, average="weighted")),
        "y_true": true,
        "y_pred": pred,
    }


def train_model(data, num_classes: int, train_idx: np.ndarray):
    model = FlowGraphSAGE(
        in_channels=data.num_node_features,
        hidden_channels=HIDDEN_DIM,
        out_channels=num_classes,
        dropout=DROPOUT,
    ).to(DEVICE)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=data.y[data.train_mask].cpu().numpy(),
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_macro_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

        train_metrics = evaluate(model, data, data.train_mask)
        val_metrics = evaluate(model, data, data.val_mask)

        history.append({
            "epoch": epoch,
            "loss": float(loss.item()),
            "train_acc": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss.item():.4f} | "
                f"Train Macro-F1 {train_metrics['macro_f1']:.4f} | "
                f"Val Macro-F1 {val_metrics['macro_f1']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# =========================
# 6) Main
# =========================
def main():
    set_seed(SEED)

    print("Loading and concatenating CSV files...")
    df, dataset_summary, rows_before, rows_after = load_and_concat(
        data_dir=DATA_DIR,
        csv_files=CSV_FILES,
        drop_duplicates=DROP_DUPLICATES,
    )

    print(dataset_summary)
    print(f"Rows before dropping duplicates: {rows_before}")
    print(f"Rows after dropping duplicates : {rows_after}")

    print("\nEngineering node features...")
    df, X, y, label_encoder, preprocessor, feature_info = engineer_features(df)
    print(json.dumps(feature_info, indent=2))

    print("\nBuilding graph edges...")
    edge_index = build_edge_index(df, EDGE_KEY_SPECS, window=EDGE_WINDOW)
    print(f"edge_index shape = {edge_index.shape}")

    print("\nBuilding train/val/test masks...")
    train_mask, val_mask, test_mask, train_idx, val_idx, test_idx = build_masks(y, seed=SEED)

    data = Data(
        x=torch.from_numpy(X),
        edge_index=torch.from_numpy(edge_index).long(),
        y=torch.from_numpy(y).long(),
    )
    data.train_mask = torch.from_numpy(train_mask)
    data.val_mask = torch.from_numpy(val_mask)
    data.test_mask = torch.from_numpy(test_mask)
    data = data.to(DEVICE)

    print("\nTraining GNN...")
    model, history = train_model(data, num_classes=len(label_encoder.classes_), train_idx=train_idx)

    print("\nEvaluating best model...")
    test_metrics = evaluate(model, data, data.test_mask)

    report = classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        target_names=label_encoder.classes_,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"])

    print(f"Test accuracy   : {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1   : {test_metrics['macro_f1']:.4f}")
    print(f"Test weighted F1: {test_metrics['weighted_f1']:.4f}")

    # Save outputs
    dataset_summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    pd.DataFrame(conf_mat, index=label_encoder.classes_, columns=label_encoder.classes_).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    with open(OUTPUT_DIR / "feature_info.json", "w", encoding="utf-8") as f:
        json.dump(feature_info, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    torch.save(model.state_dict(), OUTPUT_DIR / "flow_graphsage_model.pt")
    print(f"\nSaved artifacts to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
