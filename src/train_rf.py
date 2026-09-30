"""Random Forest baseline (CPU, all cores) with a small validation-based search.

Protocol:
  1. Build features -> global temporal split (70/15/15 by timestamp order).
  2. Imputation medians + constant-column list fitted on TRAIN only.
  3. Small grid over {class_weight x max_depth} scored by VAL PR-AUC
     (PR-AUC is the selection metric because failures are ~1.5% of rows).
  4. Refit the winner on train+val, evaluate once on the held-out TEST set.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from data_loader import PROJECT_ROOT, load_config, load_raw
from evaluation import evaluate
from feature_engineering import build_features, select_feature_columns
from preprocessing import apply_imputation, clean, constant_columns, fit_imputation

META_COLS = ["timestamp", "Device_name", "class", "label", "source_file"]


def temporal_split(df, train_frac=0.70, val_frac=0.15):
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    i_tr = int(n * train_frac)
    i_va = int(n * (train_frac + val_frac))
    tr, va, te = df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:]
    return tr, va, te, int(te.iloc[0]["timestamp"]), int(tr.iloc[-1]["timestamp"])


def main():
    cfg = load_config()
    t0 = time.time()
    raw_path = os.path.join(PROJECT_ROOT, "data", "unified.parquet")
    df = pd.read_parquet(raw_path) if os.path.exists(raw_path) else load_raw(cfg)
    n_all = len(df)
    # Features first (rolling history may use T-row telemetry; labels never do),
    # THEN restrict the binary task to healthy-vs-failure rows.
    df = build_features(clean(df), cfg.get("rolling_windows", [5, 15]))
    df = df[~df["class"].isin(cfg.get("excluded_classes", []))].reset_index(drop=True)
    print(f"binary task H-vs-F: kept {len(df)}/{n_all} rows "
          f"(excluded {cfg.get('excluded_classes')})")

    drop = list(cfg.get("drop_cols", []))
    feats = select_feature_columns(df, drop, cfg.get("drop_substrings", []))
    tr, va, te, test_start, train_end = temporal_split(
        df, cfg["split"]["train"], cfg["split"]["val"])
    medians = fit_imputation(tr, feats)
    tr, va, te = apply_imputation(tr, medians), apply_imputation(va, medians), apply_imputation(te, medians)
    consts = constant_columns(tr, feats)
    feats = [c for c in feats if c not in consts]
    print(f"features: {len(feats)} (dropped {len(consts)} constant): {consts}")

    base = cfg["random_forest"]
    grid = []
    for cw in ["balanced_subsample", None]:
        for md in [base.get("max_depth"), 20]:
            grid.append({**base, "class_weight": cw, "max_depth": md})
    best, best_ap = None, -1.0
    for i, params in enumerate(grid):
        clf = RandomForestClassifier(**{k: v for k, v in params.items()})
        clf.fit(tr[feats], tr["label"])
        ap = average_precision_score(va["label"], clf.predict_proba(va[feats])[:, 1])
        print(f"  candidate {i}: class_weight={params['class_weight']} "
              f"max_depth={params['max_depth']} val PR-AUC={ap:.4f}")
        if ap > best_ap:
            best_ap, best = ap, params
    print("winner:", {k: best[k] for k in ("class_weight", "max_depth")}, f"val PR-AUC={best_ap:.4f}")

    refit = pd.concat([tr, va], ignore_index=True)
    t_train = time.time()
    clf = RandomForestClassifier(**{k: v for k, v in best.items()})
    clf.fit(refit[feats], refit["label"])
    train_time = time.time() - t_train

    # Operating threshold: maximise F1 on validation (standard for 1:64
    # imbalance); ranking metrics (ROC/PR-AUC) are threshold-free.
    from sklearn.metrics import f1_score
    va_proba = clf.predict_proba(va[feats])[:, 1]
    cand = sorted(set(np.quantile(va_proba, np.linspace(0, 1, 101)).tolist() + [0.5]))
    thr = max(cand, key=lambda t: f1_score(va["label"], (va_proba >= t).astype(int),
                                           zero_division=0))
    print(f"val-tuned threshold={thr:.4f}")

    t_inf = time.time()
    proba = clf.predict_proba(te[feats])[:, 1]
    infer_time = time.time() - t_inf

    res = evaluate(te["label"], proba, train_time, infer_time, threshold=thr)
    res.update({"model": "Random Forest", "params": {k: v for k, v in best.items()},
                "n_features": len(feats), "test_start_ts": test_start,
                "train_end_ts": train_end, "train_rows": len(refit),
                "val_rows": len(va), "test_rows": len(te),
                "val_best_pr_auc": best_ap, "total_time_s": time.time() - t0,
                "dropped_constant_cols": consts, "feature_names": feats})

    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "metrics"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "predictions"), exist_ok=True)
    joblib.dump(clf, os.path.join(PROJECT_ROOT, "models", "random_forest.pkl"))
    with open(os.path.join(PROJECT_ROOT, "results", "metrics", "random_forest.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    pred = te[META_COLS].copy()
    pred["proba"] = proba
    pred["pred"] = (proba >= 0.5).astype(int)
    pred.to_parquet(os.path.join(PROJECT_ROOT, "results", "predictions", "random_forest.parquet"), index=False)
    with open(os.path.join(PROJECT_ROOT, "results", "metrics", "split.json"), "w") as f:
        json.dump({"train_end_ts": train_end, "test_start_ts": test_start,
                   "train_rows": len(refit), "val_rows": len(va),
                   "test_rows": len(te), "features": feats}, f, indent=2)
    print("Random Forest TEST:", {k: round(v, 4) for k, v in res.items()
          if k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")})


if __name__ == "__main__":
    main()
