"""Generate every required figure + the RF-vs-XGB comparison table."""
import json
import os

import joblib
import pandas as pd

from data_loader import PROJECT_ROOT, load_config
from evaluation import (comparison_table, plot_class_distribution,
                        plot_confusion, plot_importance, plot_proba_hist,
                        plot_proba_over_time, plot_roc_pr)


def main():
    cfg = load_config()
    figdir = os.path.join(PROJECT_ROOT, "results", "figures")
    os.makedirs(figdir, exist_ok=True)

    rf_m = json.load(open(os.path.join(PROJECT_ROOT, "results", "metrics", "random_forest.json")))
    xgb_m = json.load(open(os.path.join(PROJECT_ROOT, "results", "metrics", "xgboost.json")))
    rf_p = pd.read_parquet(os.path.join(PROJECT_ROOT, "results", "predictions", "random_forest.parquet"))
    xgb_p = pd.read_parquet(os.path.join(PROJECT_ROOT, "results", "predictions", "xgboost.parquet"))
    extra, curves = [], {"Random Forest": (rf_p["label"], rf_p["proba"]),
                         "XGBoost": (xgb_p["label"], xgb_p["proba"])}
    gcn_m_path = os.path.join(PROJECT_ROOT, "results", "metrics", "gcn.json")
    if os.path.exists(gcn_m_path):
        gcn_m = json.load(open(gcn_m_path))
        gcn_p = pd.read_parquet(os.path.join(PROJECT_ROOT, "results", "predictions", "gcn.parquet"))
        extra = [(gcn_m, gcn_p, "GCN")]
        curves["GCN"] = (gcn_p["label"], gcn_p["proba"])
    svm_m_path = os.path.join(PROJECT_ROOT, "results", "metrics", "svm.json")
    if os.path.exists(svm_m_path):
        svm_m = json.load(open(svm_m_path))
        svm_p = pd.read_parquet(os.path.join(PROJECT_ROOT, "results", "predictions", "svm.parquet"))
        extra.append((svm_m, svm_p, "Linear SVM"))
        curves["Linear SVM"] = (svm_p["label"], svm_p["proba"])

    full = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "unified.parquet"))
    plot_class_distribution(full)

    plot_confusion(rf_m, "rf")
    plot_confusion(xgb_m, "xgb")
    for gm, gp, gname in extra:
        plot_confusion(gm, gname.lower())
    plot_roc_pr(curves)
    plot_proba_hist(rf_p, xgb_p)
    plot_proba_over_time(rf_p, "rf")
    plot_proba_over_time(xgb_p, "xgb")

    feats = rf_m["feature_names"]
    rf = joblib.load(os.path.join(PROJECT_ROOT, "models", "random_forest.pkl"))
    plot_importance(feats, rf.feature_importances_,
                    "Random Forest — top-20 feature importances (test features)",
                    os.path.join(figdir, "06_importance_rf.png"))
    xb = joblib.load(os.path.join(PROJECT_ROOT, "models", "xgboost.pkl"))
    plot_importance(feats, xb.feature_importances_,
                    "XGBoost — top-20 feature importances (gain, test features)",
                    os.path.join(figdir, "07_importance_xgb.png"))

    df, md = comparison_table(
        [rf_m, xgb_m] + [gm for gm, _, _ in extra],
        os.path.join(PROJECT_ROOT, "results", "metrics", "comparison.md"))
    print(md)
    # Top-risk links snapshot: highest-XGB-proba test rows.
    top = xgb_p.sort_values("proba", ascending=False).head(20)
    top.to_csv(os.path.join(PROJECT_ROOT, "results", "predictions", "top_risk.csv"), index=False)
    print("figures + comparison written to results/")


if __name__ == "__main__":
    main()
