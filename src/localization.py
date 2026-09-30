"""Failure localisation at router granularity.

Limitation (documented, not hidden): the LCORE-D release contains no
explicit link_id / source / destination columns and the fault log records
fault windows + domains, not per-link ground truth. Genuine link-level
localisation is therefore NOT supported by the data. What IS supported is
node-level localisation: at each test timestamp, rank the observed routers
(devices) by predicted P(failure) and check whether the truly failed
router(s) are ranked at the top.

Metrics (computed only on test timestamps with >= 1 actual failure):
  Top-1 / Top-3 accuracy and Mean Reciprocal Rank (MRR).
"""
import json
import os

import numpy as np
import pandas as pd

from data_loader import PROJECT_ROOT


def rank_at_timestamps(pred, k_list=(1, 3, 5)):
    """pred needs [timestamp, Device_name, label, proba]."""
    hits = {k: [] for k in k_list}
    rrs = []
    evaluable = 0
    for ts, grp in pred.groupby("timestamp"):
        failed = set(grp.loc[grp.label == 1, "Device_name"])
        if not failed:
            continue
        evaluable += 1
        ranked = grp.sort_values("proba", ascending=False)["Device_name"].tolist()
        best_rank = min(ranked.index(d) + 1 for d in failed if d in ranked)
        rrs.append(1.0 / best_rank)
        for k in k_list:
            hits[k].append(int(any(d in ranked[:k] for d in failed)))
    return {
        "evaluable_timestamps": evaluable,
        "total_timestamps": int(pred["timestamp"].nunique()),
        **{f"top{k}_acc": float(np.mean(hits[k])) if hits[k] else float("nan") for k in k_list},
        "mrr": float(np.mean(rrs)) if rrs else float("nan"),
    }


def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "metrics"), exist_ok=True)
    out = {}
    for name in ("random_forest", "xgboost", "gcn", "svm"):
        pred = pd.read_parquet(
            os.path.join(PROJECT_ROOT, "results", "predictions", f"{name}.parquet"))
        m = rank_at_timestamps(pred)
        # Stratified: only timestamps with >= 4 candidate devices (median),
        # where Top-3 is non-trivial.
        keep = pred.groupby("timestamp")["Device_name"].transform("size") >= 4
        m_ge4 = rank_at_timestamps(pred[keep])
        m["n_devices_median"] = float(pred.groupby("timestamp").size().median())
        m["frac_ts_le3_devices"] = float((pred.groupby("timestamp").size() <= 3).mean())
        for k in ("top1_acc", "top3_acc", "top5_acc", "mrr", "evaluable_timestamps"):
            m[k + "_ge4"] = m_ge4[k]
        out[name] = m
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
        # Example ranked snapshot: first evaluable timestamp.
        for ts, grp in pred.groupby("timestamp"):
            if (grp.label == 1).any():
                snap = grp.sort_values("proba", ascending=False)[
                    ["Device_name", "label", "proba"]].copy()
                snap.insert(0, "rank", range(1, len(snap) + 1))
                snap.to_csv(os.path.join(
                    PROJECT_ROOT, "results", "predictions", f"rank_snapshot_{name}.csv"),
                    index=False)
                break
    with open(os.path.join(PROJECT_ROOT, "results", "metrics", "localization.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
