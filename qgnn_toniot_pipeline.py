
import json
import math
import os
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------- CONFIG ---------------------------------

DATA_DIR = Path(r".")
CSV_FILES = [
    "backdoor.csv",
    "benign_tot.csv",
    "ddos.csv",
    "dos.csv",
    "injection.csv",
    "mitm.csv",
    "password.csv",
    "ransomware.csv",
    "scanning.csv",
    "xss.csv",
]

TARGET_MODE = "multiclass"   # "multiclass" -> use `type`, "binary" -> use `label`
DROP_DUPLICATES = True

# QGNN is expensive. Keep a per-class cap by default for a practical run.
# Set to None to use all rows.
MAX_SAMPLES_PER_CLASS = 4000

TRAIN_CLASSICAL_BASELINE = True
TRAIN_HYBRID_QGNN = True

SEED = 42
RESULTS_DIR = Path("./results_qgnn_toniot")

# Feature / graph settings
MAX_DNS_QUERY_CARDINALITY = 300
MAX_HTTP_URI_CARDINALITY = 120
MAX_HTTP_UA_CARDINALITY = 60
MAX_GROUP_SIZE_FOR_EDGES = 5000

# Subgraph mini-batching with k-hop extraction
NUM_HOPS = 2
BATCH_SIZE_BASELINE = 256
BATCH_SIZE_QGNN = 48

# Model sizes
HIDDEN_DIM = 128
EMBED_DIM = 64
DROPOUT = 0.30

# Quantum head
N_QUBITS = 8
Q_DEPTH = 2

# Training
EPOCHS_BASELINE = 20
EPOCHS_QGNN = 12
LR_BASELINE = 2e-3
LR_QGNN = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
USE_WEIGHTED_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0

# For compatibility and simplicity, run the hybrid QGNN on CPU by default.
# Classical baseline can be run on CUDA if you change FORCE_CPU to False.
FORCE_CPU = True

# --------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def choose_torch_device(force_cpu: bool = True) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_import_graph_libs():
    try:
        from torch_geometric.data import Data
        from torch_geometric.nn import SAGEConv
        from torch_geometric.utils import k_hop_subgraph
    except Exception as e:
        raise ImportError(
            "This script requires torch_geometric. Install PyTorch Geometric first. "
            f"Original import error: {e}"
        ) from e
    return Data, SAGEConv, k_hop_subgraph


def safe_import_quantum_lib():
    try:
        import pennylane as qml
    except Exception as e:
        raise ImportError(
            "This script requires PennyLane for the quantum layer. "
            f"Original import error: {e}"
        ) from e
    return qml


def load_and_concat(
    data_dir: Path,
    csv_files: List[str],
    drop_duplicates: bool = True,
) -> Tuple[pd.DataFrame, Dict, int, int]:
    frames = []
    for name in csv_files:
        path = data_dir / name
        print(f"Reading: {path.resolve()}")
        df = pd.read_csv(path)
        df["source_file"] = name
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    rows_before = len(merged)

    if drop_duplicates:
        merged = merged.drop_duplicates(ignore_index=True)

    rows_after = len(merged)

    summary = {
        "rows_before_concat": int(rows_before),
        "rows_after_drop_duplicates": int(rows_after),
        "duplicates_removed": int(rows_before - rows_after),
        "n_columns": int(merged.shape[1]),
        "columns": merged.columns.tolist(),
        "type_distribution": {str(k): int(v) for k, v in merged["type"].value_counts().to_dict().items()},
        "label_distribution": {str(k): int(v) for k, v in merged["label"].value_counts().to_dict().items()},
    }
    return merged, summary, rows_before, rows_after


def sample_per_class(
    df: pd.DataFrame,
    target_col: str,
    max_samples_per_class: Optional[int],
    seed: int = 42,
) -> pd.DataFrame:
    if max_samples_per_class is None:
        return df.reset_index(drop=True)

    sampled = []
    for cls, group in df.groupby(target_col):
        if len(group) <= max_samples_per_class:
            sampled.append(group)
        else:
            sampled.append(group.sample(n=max_samples_per_class, random_state=seed))
    out = pd.concat(sampled, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def ip_to_octets(ip: object) -> List[float]:
    if not isinstance(ip, str):
        return [0.0, 0.0, 0.0, 0.0]
    parts = ip.split(".")
    if len(parts) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    vals = []
    for p in parts:
        try:
            vals.append(float(int(p)) / 255.0)
        except Exception:
            vals.append(0.0)
    return vals


def maybe_missing_str(x: object) -> bool:
    if pd.isna(x):
        return True
    if isinstance(x, str) and x.strip() in {"", "-"}:
        return True
    return False


def normalize_object_column(col: pd.Series) -> pd.Series:
    return col.fillna("-").astype(str).replace({"nan": "-", "None": "-"})


def frequency_encode(col: pd.Series) -> pd.Series:
    freq = col.value_counts(normalize=True)
    return col.map(freq).fillna(0.0).astype(np.float32)


def preprocess_features(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str], Dict]:
    work = df.copy()

    object_cols = [c for c in work.columns if work[c].dtype == "object"]
    for c in object_cols:
        work[c] = normalize_object_column(work[c])

    feature_blocks = []
    feature_names = []

    # ---------- engineered numeric ----------
    numeric_block = pd.DataFrame(index=work.index)

    # Base numeric columns present in your data
    raw_numeric_cols = [
        "src_port",
        "dst_port",
        "duration",
        "src_bytes",
        "dst_bytes",
        "missed_bytes",
        "src_pkts",
        "src_ip_bytes",
        "dst_pkts",
        "dst_ip_bytes",
        "dns_qclass",
        "dns_qtype",
        "dns_rcode",
        "http_request_body_len",
        "http_response_body_len",
        "http_status_code",
    ]
    for col in raw_numeric_cols:
        numeric_block[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    # Log-scaled traffic features
    for col in [
        "duration",
        "src_bytes",
        "dst_bytes",
        "missed_bytes",
        "src_pkts",
        "src_ip_bytes",
        "dst_pkts",
        "dst_ip_bytes",
        "http_request_body_len",
        "http_response_body_len",
    ]:
        numeric_block[f"{col}_log1p"] = np.log1p(numeric_block[col].clip(lower=0.0))

    numeric_block["total_bytes"] = numeric_block["src_bytes"] + numeric_block["dst_bytes"]
    numeric_block["total_pkts"] = numeric_block["src_pkts"] + numeric_block["dst_pkts"]
    numeric_block["bytes_ratio"] = numeric_block["src_bytes"] / (numeric_block["dst_bytes"] + 1.0)
    numeric_block["pkts_ratio"] = numeric_block["src_pkts"] / (numeric_block["dst_pkts"] + 1.0)

    # Presence flags
    numeric_block["has_dns_query"] = (~work["dns_query"].isin(["-", ""])).astype(float)
    numeric_block["has_http"] = (~work["http_uri"].isin(["-", ""])).astype(float)
    numeric_block["has_ssl"] = (~work["ssl_version"].isin(["-", ""])).astype(float)
    numeric_block["has_weird"] = (~work["weird_name"].isin(["-", ""])).astype(float)

    # IP octets
    src_oct = np.array([ip_to_octets(v) for v in work["src_ip"]], dtype=np.float32)
    dst_oct = np.array([ip_to_octets(v) for v in work["dst_ip"]], dtype=np.float32)
    for i in range(4):
        numeric_block[f"src_ip_octet_{i+1}"] = src_oct[:, i]
        numeric_block[f"dst_ip_octet_{i+1}"] = dst_oct[:, i]

    # Frequency encodings for higher-cardinality string columns
    freq_cols = [
        "src_ip",
        "dst_ip",
        "dns_query",
        "http_uri",
        "http_user_agent",
    ]
    for col in freq_cols:
        numeric_block[f"{col}_freq"] = frequency_encode(work[col])

    # Scaled numeric
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_block.astype(np.float32))
    feature_blocks.append(numeric_scaled.astype(np.float32))
    feature_names.extend(numeric_block.columns.tolist())

    # ---------- one-hot categorical ----------
    low_card_cols = [
        "proto",
        "service",
        "conn_state",
        "dns_AA",
        "dns_RD",
        "dns_RA",
        "dns_rejected",
        "ssl_version",
        "ssl_cipher",
        "ssl_resumed",
        "ssl_established",
        "ssl_subject",
        "ssl_issuer",
        "http_trans_depth",
        "http_method",
        "http_version",
        "http_orig_mime_types",
        "http_resp_mime_types",
        "weird_name",
        "weird_addl",
        "weird_notice",
    ]
    dummies = pd.get_dummies(
        work[low_card_cols],
        prefix=low_card_cols,
        dummy_na=False,
        sparse=False,
        dtype=np.uint8,
    )
    feature_blocks.append(dummies.to_numpy(dtype=np.float32))
    feature_names.extend(dummies.columns.tolist())

    # ---------- capped one-hot for informative medium-cardinality cols ----------
    medium_card_specs = [
        ("dns_query", MAX_DNS_QUERY_CARDINALITY),
        ("http_uri", MAX_HTTP_URI_CARDINALITY),
        ("http_user_agent", MAX_HTTP_UA_CARDINALITY),
    ]
    for col, top_k in medium_card_specs:
        top_values = work[col].value_counts().head(top_k).index.tolist()
        reduced = work[col].where(work[col].isin(top_values), other="__OTHER__")
        tmp = pd.get_dummies(reduced, prefix=col, dtype=np.uint8)
        feature_blocks.append(tmp.to_numpy(dtype=np.float32))
        feature_names.extend(tmp.columns.tolist())

    X = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    artifacts = {"feature_names": feature_names}
    return X, feature_names, artifacts


def build_labels(
    df: pd.DataFrame,
    target_mode: str = "multiclass",
) -> Tuple[np.ndarray, Dict[int, str]]:
    if target_mode == "binary":
        y = df["label"].astype(int).to_numpy()
        idx_to_name = {0: "benign", 1: "attack"}
        return y, idx_to_name

    classes = sorted(df["type"].astype(str).unique().tolist())
    name_to_idx = {name: i for i, name in enumerate(classes)}
    idx_to_name = {i: name for name, i in name_to_idx.items()}
    y = df["type"].map(name_to_idx).astype(int).to_numpy()
    return y, idx_to_name


def add_chain_edges_from_values(
    values: pd.Series,
    edge_set: set,
    max_group_size: int = 5000,
) -> None:
    groups = defaultdict(list)
    for idx, val in enumerate(values.tolist()):
        if maybe_missing_str(val):
            continue
        groups[val].append(idx)

    for _, idxs in groups.items():
        if len(idxs) < 2:
            continue
        if len(idxs) > max_group_size:
            # Keep graph sparse and practical
            idxs = idxs[:max_group_size]

        idxs = sorted(idxs)
        for a, b in zip(idxs[:-1], idxs[1:]):
            edge_set.add((a, b))
            edge_set.add((b, a))


def build_graph(df: pd.DataFrame) -> torch.Tensor:
    edge_set = set()

    # Relation 1: same source IP
    add_chain_edges_from_values(df["src_ip"], edge_set, MAX_GROUP_SIZE_FOR_EDGES)

    # Relation 2: same destination endpoint
    dst_endpoint = df["dst_ip"].astype(str) + ":" + df["dst_port"].astype(str)
    add_chain_edges_from_values(dst_endpoint, edge_set, MAX_GROUP_SIZE_FOR_EDGES)

    # Relation 3: same network / application signature
    app_sig = (
        df["proto"].astype(str)
        + "|"
        + df["service"].astype(str)
        + "|"
        + df["conn_state"].astype(str)
    )
    add_chain_edges_from_values(app_sig, edge_set, MAX_GROUP_SIZE_FOR_EDGES)

    # Relation 4: same DNS query (if present)
    add_chain_edges_from_values(df["dns_query"], edge_set, MAX_GROUP_SIZE_FOR_EDGES)

    # Relation 5: same HTTP URI (if present)
    add_chain_edges_from_values(df["http_uri"], edge_set, MAX_GROUP_SIZE_FOR_EDGES)

    if not edge_set:
        raise RuntimeError("No graph edges were created. Check the graph-building logic.")

    edges = np.array(list(edge_set), dtype=np.int64)
    edge_index = torch.from_numpy(edges.T).long()
    return edge_index


def stratified_masks(
    y: np.ndarray,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8
    indices = np.arange(len(y))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=y,
    )
    temp_y = y[temp_idx]

    val_size_relative = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_size_relative),
        random_state=seed,
        stratify=temp_y,
    )

    return train_idx, val_idx, test_idx


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, a_min=1.0, a_max=None)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class GraphEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, embed_dim: int, dropout: float, SAGEConv):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class ClassicalGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, embed_dim: int, num_classes: int, dropout: float, SAGEConv):
        super().__init__()
        self.encoder = GraphEncoder(in_channels, hidden_dim, embed_dim, dropout, SAGEConv)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, edge_index)
        return self.classifier(z)


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits: int = 8, q_depth: int = 2, device_preference: str = "lightning.qubit"):
        super().__init__()
        qml = safe_import_quantum_lib()
        self.n_qubits = n_qubits
        self.q_depth = q_depth

        try:
            self.qdev = qml.device(device_preference, wires=n_qubits)
            self.q_device_name = device_preference
        except Exception:
            self.qdev = qml.device("default.qubit", wires=n_qubits)
            self.q_device_name = "default.qubit"

        self.q_weights = nn.Parameter(0.01 * torch.randn(q_depth, n_qubits, 3, dtype=torch.float64))

        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=list(range(n_qubits)), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=list(range(n_qubits)))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        x = x.to(dtype=torch.float64)
        for sample in x:
            result = self.circuit(sample, self.q_weights)
            if isinstance(result, (list, tuple)):
                result = torch.stack(list(result))
            outputs.append(result)
        out = torch.stack(outputs, dim=0)
        return out.to(dtype=torch.float32)


class HybridQGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        embed_dim: int,
        num_classes: int,
        dropout: float,
        n_qubits: int,
        q_depth: int,
        SAGEConv,
    ):
        super().__init__()
        self.encoder = GraphEncoder(in_channels, hidden_dim, embed_dim, dropout, SAGEConv)
        self.q_input = nn.Sequential(
            nn.Linear(embed_dim, n_qubits),
            nn.LayerNorm(n_qubits),
            nn.Tanh(),
        )
        self.q_layer = QuantumLayer(n_qubits=n_qubits, q_depth=q_depth)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + n_qubits, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, edge_index)
        q_in = self.q_input(z)
        q_out = self.q_layer(q_in)
        fused = torch.cat([z, q_out], dim=-1)
        return self.fusion(fused)


def get_batches(indices: np.ndarray, batch_size: int, shuffle: bool = True) -> List[np.ndarray]:
    idx = np.array(indices)
    if shuffle:
        np.random.shuffle(idx)
    batches = []
    for start in range(0, len(idx), batch_size):
        batches.append(idx[start:start + batch_size])
    return batches


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def train_one_epoch_subgraph(
    model: nn.Module,
    data,
    train_idx: np.ndarray,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
    num_hops: int,
    k_hop_subgraph,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    losses = []
    all_true = []
    all_pred = []

    batches = get_batches(train_idx, batch_size=batch_size, shuffle=True)

    for batch_nodes in batches:
        batch_nodes_t = torch.tensor(batch_nodes, dtype=torch.long, device=device)

        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            batch_nodes_t,
            num_hops=num_hops,
            edge_index=data.edge_index,
            relabel_nodes=True,
        )

        x_sub = data.x[subset]
        y_target = data.y[batch_nodes_t]

        optimizer.zero_grad()
        logits_sub = model(x_sub, sub_edge_index)
        logits_target = logits_sub[mapping]
        loss = criterion(logits_target, y_target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = logits_target.argmax(dim=1).detach().cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(y_target.detach().cpu().numpy().tolist())

    metrics = compute_metrics(np.array(all_true), np.array(all_pred))
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


@torch.no_grad()
def evaluate_subgraph(
    model: nn.Module,
    data,
    eval_idx: np.ndarray,
    criterion: nn.Module,
    batch_size: int,
    num_hops: int,
    k_hop_subgraph,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    all_true = []
    all_pred = []

    batches = get_batches(eval_idx, batch_size=batch_size, shuffle=False)

    for batch_nodes in batches:
        batch_nodes_t = torch.tensor(batch_nodes, dtype=torch.long, device=device)

        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            batch_nodes_t,
            num_hops=num_hops,
            edge_index=data.edge_index,
            relabel_nodes=True,
        )

        x_sub = data.x[subset]
        y_target = data.y[batch_nodes_t]

        logits_sub = model(x_sub, sub_edge_index)
        logits_target = logits_sub[mapping]
        loss = criterion(logits_target, y_target)

        losses.append(loss.item())
        preds = logits_target.argmax(dim=1).cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(y_target.cpu().numpy().tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    metrics = compute_metrics(all_true, all_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics, all_true, all_pred


def fit_model(
    model: nn.Module,
    model_name: str,
    data,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    class_weights: torch.Tensor,
    idx_to_name: Dict[int, str],
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    num_hops: int,
    k_hop_subgraph,
    device: torch.device,
    results_dir: Path,
) -> Dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = class_weights.to(device)

    if USE_WEIGHTED_FOCAL_LOSS:
        criterion = WeightedFocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val = -float("inf")
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch_subgraph(
            model=model,
            data=data,
            train_idx=train_idx,
            optimizer=optimizer,
            criterion=criterion,
            batch_size=batch_size,
            num_hops=num_hops,
            k_hop_subgraph=k_hop_subgraph,
            device=device,
        )

        val_metrics, _, _ = evaluate_subgraph(
            model=model,
            data=data,
            eval_idx=val_idx,
            criterion=criterion,
            batch_size=batch_size,
            num_hops=num_hops,
            k_hop_subgraph=k_hop_subgraph,
            device=device,
        )

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        print(
            f"[{model_name}] Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} train_macro_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"[{model_name}] Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    criterion_eval = nn.CrossEntropyLoss(weight=class_weights)

    val_metrics, val_true, val_pred = evaluate_subgraph(
        model=model,
        data=data,
        eval_idx=val_idx,
        criterion=criterion_eval,
        batch_size=batch_size,
        num_hops=num_hops,
        k_hop_subgraph=k_hop_subgraph,
        device=device,
    )
    test_metrics, test_true, test_pred = evaluate_subgraph(
        model=model,
        data=data,
        eval_idx=test_idx,
        criterion=criterion_eval,
        batch_size=batch_size,
        num_hops=num_hops,
        k_hop_subgraph=k_hop_subgraph,
        device=device,
    )

    # Save artifacts
    model_dir = results_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_dir / f"{model_name}_best.pt")
    pd.DataFrame(history).to_csv(model_dir / f"{model_name}_history.csv", index=False)

    labels = list(idx_to_name.keys())
    target_names = [idx_to_name[i] for i in labels]

    cm = confusion_matrix(test_true, test_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(model_dir / f"{model_name}_confusion_matrix.csv", index=True)

    report = classification_report(
        test_true,
        test_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    with open(model_dir / f"{model_name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    save_confusion_matrix_plot(cm, target_names, model_dir / f"{model_name}_confusion_matrix.png")

    summary = {
        "model_name": model_name,
        "best_val_macro_f1": float(best_val),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "classification_report_path": str((model_dir / f"{model_name}_classification_report.txt").resolve()),
        "history_path": str((model_dir / f"{model_name}_history.csv").resolve()),
        "weights_path": str((model_dir / f"{model_name}_best.pt").resolve()),
    }

    with open(model_dir / f"{model_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def save_confusion_matrix_plot(cm: np.ndarray, class_names: List[str], out_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    warnings.filterwarnings("ignore")
    set_seed(SEED)

    Data, SAGEConv, k_hop_subgraph = safe_import_graph_libs()
    device = choose_torch_device(force_cpu=FORCE_CPU)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and concatenating CSV files...")
    df, dataset_summary, rows_before, rows_after = load_and_concat(
        data_dir=DATA_DIR,
        csv_files=CSV_FILES,
        drop_duplicates=DROP_DUPLICATES,
    )

    target_col = "label" if TARGET_MODE == "binary" else "type"
    df = sample_per_class(df, target_col=target_col, max_samples_per_class=MAX_SAMPLES_PER_CLASS, seed=SEED)

    print(f"Rows before concat: {rows_before}")
    print(f"Rows after dedupe:  {rows_after}")
    print(f"Rows used for this QGNN run: {len(df)}")
    print(f"Target mode: {TARGET_MODE}")

    print("Preprocessing features...")
    X, feature_names, feature_artifacts = preprocess_features(df)
    y, idx_to_name = build_labels(df, target_mode=TARGET_MODE)

    print("Building graph...")
    edge_index = build_graph(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Edge index shape: {tuple(edge_index.shape)}")
    print(f"Number of classes: {len(idx_to_name)}")

    train_idx, val_idx, test_idx = stratified_masks(y, seed=SEED)
    class_weights = compute_class_weights(y[train_idx], num_classes=len(idx_to_name))

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index.long(),
        y=torch.tensor(y, dtype=torch.long),
    )
    data = data.to(device)

    run_summary = {
        "dataset_summary": dataset_summary,
        "rows_used_for_this_run": int(len(df)),
        "target_mode": TARGET_MODE,
        "max_samples_per_class": MAX_SAMPLES_PER_CLASS,
        "feature_dim": int(X.shape[1]),
        "num_edges": int(edge_index.shape[1]),
        "num_classes": int(len(idx_to_name)),
        "classes": idx_to_name,
        "device": str(device),
        "feature_names_count": int(len(feature_names)),
    }

    with open(RESULTS_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    if TRAIN_CLASSICAL_BASELINE:
        print("\nTraining classical baseline GNN...")
        classical = ClassicalGNN(
            in_channels=X.shape[1],
            hidden_dim=HIDDEN_DIM,
            embed_dim=EMBED_DIM,
            num_classes=len(idx_to_name),
            dropout=DROPOUT,
            SAGEConv=SAGEConv,
        ).to(device)

        classical_summary = fit_model(
            model=classical,
            model_name="classical_gnn",
            data=data,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            class_weights=class_weights,
            idx_to_name=idx_to_name,
            batch_size=BATCH_SIZE_BASELINE,
            epochs=EPOCHS_BASELINE,
            lr=LR_BASELINE,
            weight_decay=WEIGHT_DECAY,
            num_hops=NUM_HOPS,
            k_hop_subgraph=k_hop_subgraph,
            device=device,
            results_dir=RESULTS_DIR,
        )
        print("\nClassical baseline summary:")
        print(json.dumps(classical_summary, indent=2))

    if TRAIN_HYBRID_QGNN:
        print("\nTraining hybrid QGNN...")
        hybrid = HybridQGNN(
            in_channels=X.shape[1],
            hidden_dim=HIDDEN_DIM,
            embed_dim=EMBED_DIM,
            num_classes=len(idx_to_name),
            dropout=DROPOUT,
            n_qubits=N_QUBITS,
            q_depth=Q_DEPTH,
            SAGEConv=SAGEConv,
        ).to(device)

        qgnn_summary = fit_model(
            model=hybrid,
            model_name="hybrid_qgnn",
            data=data,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            class_weights=class_weights,
            idx_to_name=idx_to_name,
            batch_size=BATCH_SIZE_QGNN,
            epochs=EPOCHS_QGNN,
            lr=LR_QGNN,
            weight_decay=WEIGHT_DECAY,
            num_hops=NUM_HOPS,
            k_hop_subgraph=k_hop_subgraph,
            device=device,
            results_dir=RESULTS_DIR,
        )
        print("\nHybrid QGNN summary:")
        print(json.dumps(qgnn_summary, indent=2))

    print("\nDone. Check the results folder for weights, history, confusion matrices, and reports.")


if __name__ == "__main__":
    main()
