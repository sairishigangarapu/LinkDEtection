"""Linear-kernel SVM baseline (same rows / split / features as RF/XGB/GCN).

Solver: liblinear `LinearSVC` (dual=False, class_weight='balanced') on
train-standardised features (SVMs are scale-sensitive), wrapped in
sigmoid calibration so outputs are probabilities compatible with the shared
ROC/PR/localisation code. Operating threshold = val-F1 optimum (same rule).
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from data_loader import PROJECT_ROOT, load_config
from evaluation import evaluate
from feature_engineering import build_features, select_feature_columns
from preprocessing import apply_imputation, clean, fit_imputation

META_COLS = ["timestamp", "Device_name", "class", "label", "source_file"]


def main():
    cfg = load_config()
    scfg = cfg.get("svm", {})
    t0 = time.time()
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
    tr, va, te = apply_imputation(tr, med), apply_imputation(va, med), apply_imputation(te, med)

    scaler = StandardScaler().fit(tr[feats])
    Xtr, Xva, Xte = scaler.transform(tr[feats]), scaler.transform(va[feats]), scaler.transform(te[feats])

    base = LinearSVC(C=scfg.get("C", 1.0), class_weight="balanced", dual=False,
                     max_iter=scfg.get("max_iter", 10000), random_state=cfg.get("seed", 42))
    t_train = time.time()
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(Xtr, tr["label"])
    train_time = time.time() - t_train

    va_proba = clf.predict_proba(Xva)[:, 1]
    cand = sorted(set(np.quantile(va_proba, np.linspace(0, 1, 101)).tolist() + [0.5]))
    thr = max(cand, key=lambda t: f1_score(va["label"], (va_proba >= t).astype(int), zero_division=0))
    print(f"val-tuned threshold={thr:.4f}")

    t_inf = time.time()
    proba = clf.predict_proba(Xte)[:, 1]
    infer_time = time.time() - t_inf

    res = evaluate(te["label"], proba, train_time, infer_time, threshold=thr)
    res.update({"model": "Linear SVM", "params": {"C": scfg.get("C", 1.0), "calibration": "sigmoid-cv3",
                "class_weight": "balanced", "dual": False},
                "n_features": len(feats), "test_rows": len(te)})

    joblib.dump({"scaler": scaler, "clf": clf, "features": feats},
                os.path.join(PROJECT_ROOT, "models", "svm.pkl"))
    with open(os.path.join(PROJECT_ROOT, "results", "metrics", "svm.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    pred = te[META_COLS].copy()
    pred["proba"] = proba
    pred["pred"] = (proba >= thr).astype(int)
    pred.to_parquet(os.path.join(PROJECT_ROOT, "results", "predictions", "svm.parquet"), index=False)
    print("SVM TEST:", {k: round(v, 4) for k, v in res.items()
          if k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")},
          f"train_time={train_time:.1f}s")


if __name__ == "__main__":
    main()
