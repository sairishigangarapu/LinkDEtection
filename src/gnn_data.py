"""Temporal graph dataset for the GNN phase (pure torch, no PyG dependency).

Graph (INFERRED, not ground truth — stated openly):
  Nodes = the 7 monitored routers. Edges = tree inferred from hop counts:
    hop_to_core:  R1=0, {EDGE-MS,R2}=1, R3=2, {R4,R5,R6}=3
    hop_to_server: EDGE-MS=1, R1=2, R2=3, R3=4, {R4,R5,R6}=5
  Both hop metrics are consistent with exactly one chain:
    EDGE-MS — CORE-R1 — CORE-R2 — CORE-R3 — {CORE-R4, CORE-R5, CORE-R6}
  Undirected edges (message passing both ways) + self-loops via GCN
  renormalisation. R4/R5/R6 attach as a star at R3 (their relative order is
  not identifiable from hops; downstream counts 4/4/1 suggest R4,R5 each head
  small unmonitored subtrees).

Snapshots: one graph per timestamp containing the H/F-labelled routers
observed then (same rows, same temporal split, same 131 features as RF/XGB).
Each split becomes one disjoint-union graph so message passing NEVER crosses
timestamps (block-diagonal adjacency) and test features are never seen during
training (strictly inductive across the temporal split).
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from data_loader import PROJECT_ROOT, load_config
from feature_engineering import build_features, select_feature_columns
from preprocessing import apply_imputation, clean, fit_imputation

NODES = ["EDGE-MS", "CORE-R1", "CORE-R2", "CORE-R3", "CORE-R4", "CORE-R5", "CORE-R6"]
IDX = {d: i for i, d in enumerate(NODES)}
# Inferred undirected tree edges (see docstring).
EDGES = [("EDGE-MS", "CORE-R1"), ("CORE-R1", "CORE-R2"), ("CORE-R2", "CORE-R3"),
         ("CORE-R3", "CORE-R4"), ("CORE-R3", "CORE-R5"), ("CORE-R3", "CORE-R6")]

META_COLS = ["timestamp", "Device_name", "class", "label", "source_file"]


def _snapshot_block(devices):
    """Normalised adjacency block for one timestamp (renormalisation trick)."""
    present = [d for d in NODES if d in set(devices)]
    m = len(present)
    pos = {d: i for i, d in enumerate(present)}
    A = np.eye(m)
    for a, b in EDGES:
        if a in pos and b in pos:
            A[pos[a], pos[b]] = 1.0
            A[pos[b], pos[a]] = 1.0
    deg = A.sum(axis=1)
    D = np.diag(deg ** -0.5)
    return (D @ A @ D).astype(np.float32), present


def build_split_graph(part, feats):
    """Disjoint union of snapshot blocks. Returns sparse adj + tensors + meta."""
    blocks, Ns, metas = [], [], []
    for ts, grp in part.groupby("timestamp", sort=True):
        B, present = _snapshot_block(grp["Device_name"].tolist())
        sub = grp.set_index("Device_name").loc[present].reset_index()
        blocks.append(B)
        Ns.append(len(present))
        metas.append(sub[META_COLS + feats])
    n = sum(Ns)
    rows, cols, vals = [], [], []
    off = 0
    for B, m in zip(blocks, Ns):
        r, c = np.nonzero(B)
        rows.append(r + off)
        cols.append(c + off)
        vals.append(B[r, c])
        off += m
    idx = torch.tensor([np.concatenate(rows), np.concatenate(cols)], dtype=torch.long)
    val = torch.tensor(np.concatenate(vals), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()
    meta = pd.concat(metas, ignore_index=True)
    X = torch.tensor(meta[feats].to_numpy(dtype=np.float32))
    y = torch.tensor(meta["label"].to_numpy(dtype=np.float32))
    return adj, X, y, meta[META_COLS]


def load_splits():
    cfg = load_config()
    df = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "unified.parquet"))
    df = build_features(clean(df), cfg.get("rolling_windows", [5, 15]))
    df = df[~df["class"].isin(cfg.get("excluded_classes", []))].reset_index(drop=True)
    with open(os.path.join(PROJECT_ROOT, "results", "metrics", "split.json")) as f:
        split = json.load(f)
    feats = split["features"]
    d = df.sort_values("timestamp").reset_index(drop=True)
    i_tr = split["train_rows"] - split["val_rows"]
    i_va = split["train_rows"]
    tr, va, te = d.iloc[:i_tr], d.iloc[i_tr:i_va], d.iloc[i_va:]
    med = fit_imputation(tr, feats)
    return ((build_split_graph(apply_imputation(tr, med), feats)),
            (build_split_graph(apply_imputation(va, med), feats)),
            (build_split_graph(apply_imputation(te, med), feats)), feats, cfg)


if __name__ == "__main__":
    (atr, Xtr, ytr, mtr), (ava, Xva, yva, mva), (ate, Xte, yte, mte), feats, cfg = load_splits()
    for name, A, X, y in [("train", atr, Xtr, ytr), ("val", ava, Xva, yva), ("test", ate, Xte, yte)]:
        print(f"{name}: nodes={X.shape[0]} feats={X.shape[1]} edges={A._nnz()} "
              f"failures={int(y.sum())} ({100*y.mean():.2f}%)")
