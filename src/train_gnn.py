"""2-layer GCN for node-level failure prediction (pure torch, RTX 4060).

H(A,X) = softmax-free GCN: X1 = ReLU(A_hat X W0), logit = A_hat X1 W1.
Full-batch training on the train disjoint-union graph; early stopping on
VAL PR-AUC; operating threshold = val-F1 optimum (same rule as RF/XGB).
Outputs predictions in the identical parquet format so localisation and
comparison code is shared unchanged.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score

from data_loader import PROJECT_ROOT, load_config
from evaluation import evaluate
from gnn_data import load_splits

META_COLS = ["timestamp", "Device_name", "class", "label", "source_file"]


class GCN(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.5):
        super().__init__()
        self.lin0 = nn.Linear(in_dim, hidden)
        self.lin1 = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, A, X):
        H = torch.relu(A @ self.lin0(X))
        H = self.dropout(H)
        return (A @ self.lin1(H)).squeeze(1)


def main():
    cfg = load_config()
    gcfg = cfg.get("gnn", {})
    hidden = gcfg.get("hidden", 64)
    dropout = gcfg.get("dropout", 0.5)
    lr = gcfg.get("lr", 0.01)
    max_epochs = gcfg.get("max_epochs", 500)
    patience = gcfg.get("patience", 50)
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    assert torch.cuda.is_available(), "CUDA required for GNN training."
    dev = torch.device("cuda")
    t0 = time.time()
    (Atr, Xtr, ytr, mtr), (Ava, Xva, yva, mva), (Ate, Xte, yte, mte), feats, _ = load_splits()
    # Standardise with TRAIN statistics only (trees are scale-free; nets are not).
    mu, sd = Xtr.mean(dim=0, keepdim=True), Xtr.std(dim=0, keepdim=True).clamp_min(1e-8)
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    Atr, Xtr, ytr = Atr.to(dev), Xtr.to(dev), ytr.to(dev)
    Ava, Xva, yva = Ava.to(dev), Xva.to(dev), yva.to(dev)
    Ate, Xte = Ate.to(dev), Xte.to(dev)

    pos_weight = torch.tensor([(ytr == 0).sum() / max(ytr.sum(), 1)], device=dev)
    model = GCN(Xtr.shape[1], hidden, dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_ap, best_state, bad = -1.0, None, 0
    t_train = time.time()
    for ep in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Atr, Xtr), ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            ap = average_precision_score(
                yva.cpu().numpy(), torch.sigmoid(model(Ava, Xva)).cpu().numpy())
        if ap > best_ap:
            best_ap, best_state, bad = ap, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {ep} (best val PR-AUC={best_ap:.4f})")
                break
    train_time = time.time() - t_train
    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        va_proba = torch.sigmoid(model(Ava, Xva)).cpu().numpy()
        t_inf = time.time()
        te_proba = torch.sigmoid(model(Ate, Xte)).cpu().numpy()
        infer_time = time.time() - t_inf
    yva_np = yva.cpu().numpy()
    cand = sorted(set(np.quantile(va_proba, np.linspace(0, 1, 101)).tolist() + [0.5]))
    thr = max(cand, key=lambda t: f1_score(yva_np, (va_proba >= t).astype(int), zero_division=0))
    print(f"val PR-AUC={best_ap:.4f} val-tuned threshold={thr:.4f}")

    res = evaluate(yte.cpu().numpy(), te_proba, train_time, infer_time, threshold=thr)
    res.update({"model": "GCN", "params": {"hidden": hidden, "dropout": dropout,
                "lr": lr, "max_epochs": max_epochs, "patience": patience,
                "pos_weight": float(pos_weight.cpu())},
                "n_features": len(feats), "val_pr_auc": best_ap,
                "total_time_s": time.time() - t0, "device": "cuda",
                "test_rows": len(yte), "edges": "inferred tree (see gnn_data.py)"})

    torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, "models", "gcn.pt"))
    with open(os.path.join(PROJECT_ROOT, "results", "metrics", "gcn.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    pred = mte.copy()
    pred["proba"] = te_proba
    pred["pred"] = (te_proba >= thr).astype(int)
    pred.to_parquet(os.path.join(PROJECT_ROOT, "results", "predictions", "gcn.parquet"), index=False)
    print("GCN TEST:", {k: round(v, 4) for k, v in res.items()
          if k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")})


if __name__ == "__main__":
    main()
