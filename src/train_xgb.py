"""XGBoost baseline on the RTX 4060 (device='cuda', tree_method='hist').

GPU is a HARD requirement, not a preference:
  * xgboost is checked for a CUDA build (USE_CUDA) before training,
  * `device='cuda'` makes XGBoost raise (not silently fall back) if no GPU
    is usable,
  * a background thread polls `nvidia-smi` every second during fit and the
    observed peak utilisation / memory are stored in the metrics file as
    proof of GPU execution.
"""
import json
import os
import subprocess
import threading
import time

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from data_loader import PROJECT_ROOT, load_config
from evaluation import evaluate
from feature_engineering import build_features, select_feature_columns
from preprocessing import apply_imputation, clean, constant_columns, fit_imputation

META_COLS = ["timestamp", "Device_name", "class", "label", "source_file"]


class GPUMonitor(threading.Thread):
    """Poll nvidia-smi utilisation/memory until stopped."""

    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_util, self.peak_mem = 0.0, 0.0
        self.samples = 0
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                util, mem = (float(v) for v in out.split(","))
                self.peak_util = max(self.peak_util, util)
                self.peak_mem = max(self.peak_mem, mem)
                self.samples += 1
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()


def main():
    cfg = load_config()
    assert xgb.build_info().get("USE_CUDA"), "XGBoost lacks CUDA support: refusing CPU fallback."
    t0 = time.time()
    df = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "unified.parquet"))
    n_all = len(df)
    # Features first (rolling history may use T-row telemetry; labels never do),
    # THEN restrict the binary task to healthy-vs-failure rows.
    df = build_features(clean(df), cfg.get("rolling_windows", [5, 15]))
    df = df[~df["class"].isin(cfg.get("excluded_classes", []))].reset_index(drop=True)
    print(f"binary task H-vs-F: kept {len(df)}/{n_all} rows (excluded {cfg.get('excluded_classes')})")

    with open(os.path.join(PROJECT_ROOT, "results", "metrics", "split.json")) as f:
        split = json.load(f)
    feats = split["features"]
    medians = fit_imputation(
        df.sort_values("timestamp").iloc[:split["train_rows"] - split["val_rows"]], feats)
    # Reuse the EXACT split + imputation fitted for RF so both models share the test set.
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    i_tr = split["train_rows"] - split["val_rows"]
    i_va = split["train_rows"]
    tr, va, te = df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:]
    assert len(te) == split["test_rows"], "test-set mismatch between RF and XGB runs"
    tr = apply_imputation(tr, medians)
    va = apply_imputation(va, medians)
    te = apply_imputation(te, medians)

    neg, pos = (tr["label"] == 0).sum(), tr["label"].sum()
    p = dict(cfg["xgboost"])
    p["scale_pos_weight"] = float(neg / max(pos, 1))
    early = p.pop("early_stopping_rounds", None)
    print(f"train rows={len(tr)} failures={pos} scale_pos_weight={p['scale_pos_weight']:.1f}")

    mon = GPUMonitor()
    mon.start()
    t_train = time.time()
    try:
        clf = XGBClassifier(**p)
        clf.fit(tr[feats], tr["label"], eval_set=[(va[feats], va["label"])], verbose=False)
    finally:
        train_time = time.time() - t_train
        mon.stop()
        mon.join()
    print(f"fit done in {train_time:.1f}s | GPU peak util={mon.peak_util:.0f}% "
          f"peak mem={mon.peak_mem:.0f}MiB over {mon.samples} samples")
    assert mon.peak_util > 0 and mon.samples > 0, "no GPU activity observed: refusing CPU-fallback result."

    val_ap = average_precision_score(va["label"], clf.predict_proba(va[feats])[:, 1])
    # Operating threshold: maximise F1 on validation (same rule as RF).
    import numpy as np
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
    res.update({"model": "XGBoost", "params": {**p, "early_stopping_rounds": early},
                "n_features": len(feats), "val_pr_auc": val_ap,
                "total_time_s": time.time() - t0,
                "gpu_peak_util_pct": mon.peak_util, "gpu_peak_mem_mib": mon.peak_mem,
                "gpu_samples": mon.samples, "test_rows": len(te)})

    joblib.dump(clf, os.path.join(PROJECT_ROOT, "models", "xgboost.pkl"))
    with open(os.path.join(PROJECT_ROOT, "results", "metrics", "xgboost.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    pred = te[META_COLS].copy()
    pred["proba"] = proba
    pred["pred"] = (proba >= 0.5).astype(int)
    pred.to_parquet(os.path.join(PROJECT_ROOT, "results", "predictions", "xgboost.parquet"), index=False)
    print("XGBoost TEST:", {k: round(v, 4) for k, v in res.items()
          if k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")})


if __name__ == "__main__":
    main()
