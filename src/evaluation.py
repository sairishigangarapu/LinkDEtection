"""Shared evaluation: metrics, comparison table and all figures."""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             precision_recall_curve, recall_score,
                             roc_auc_score, roc_curve)

from data_loader import PROJECT_ROOT


def evaluate(y_true, proba, train_time_s, infer_time_s, threshold=0.5):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, proba >= threshold).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": threshold,
        "train_time_s": float(train_time_s),
        "infer_time_s": float(infer_time_s),
        "test_rows": int(len(y_true)),
        "test_failures": int(y_true.sum()),
    }
    return out


def _figdir():
    d = os.path.join(PROJECT_ROOT, "results", "figures")
    os.makedirs(d, exist_ok=True)
    return d


def plot_class_distribution(df, path=None):
    path = path or os.path.join(_figdir(), "01_class_distribution.png")
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["healthy (0)", "failure (1)"], counts.values, color=["#2ca02c", "#d62728"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({100*v/counts.sum():.2f}%)", ha="center", va="bottom")
    ax.set_title("Class distribution (unified dataset, y=1 iff class=='F')")
    ax.set_ylabel("rows")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(res, name, path=None):
    path = path or os.path.join(_figdir(), f"02_confusion_{name}.png")
    cm = np.array([[res["tn"], res["fp"]], [res["fn"], res["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"])
    ax.set_title(f"Confusion matrix — {res['model']} (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_pr(curves, path_roc=None, path_pr=None):
    path_roc = path_roc or os.path.join(_figdir(), "04_roc_curve.png")
    path_pr = path_pr or os.path.join(_figdir(), "05_pr_curve.png")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC curve (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_roc, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y, p):.3f})")
    base = float(np.mean(list(curves.values())[0][0]))
    ax.axhline(base, color="k", ls="--", lw=1, label=f"chance ({base:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curve (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_pr, dpi=150)
    plt.close(fig)


def plot_importance(names, values, title, path):
    order = np.argsort(values)[::-1][:20]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(order))[::-1], np.asarray(values)[order[::-1]])
    ax.set_yticks(range(len(order))[::-1])
    ax.set_yticklabels([names[i] for i in order[::-1]], fontsize=8)
    ax.set_xlabel("importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_proba_hist(pred_rf, pred_xgb, path=None):
    path = path or os.path.join(_figdir(), "08_proba_failed_vs_healthy.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (name, pred) in zip(axes, [("Random Forest", pred_rf), ("XGBoost", pred_xgb)]):
        ax.hist(pred.loc[pred.label == 0, "proba"], bins=50, alpha=0.6, label="healthy", color="#2ca02c")
        ax.hist(pred.loc[pred.label == 1, "proba"], bins=50, alpha=0.6, label="failure", color="#d62728")
        ax.set_title(name)
        ax.set_xlabel("predicted P(failure)")
        ax.legend()
    axes[0].set_ylabel("count (log)")
    axes[0].set_yscale("log")
    fig.suptitle("Predicted-probability distribution: failed vs healthy (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_proba_over_time(pred, model_name, path=None, max_points=6000):
    path = path or os.path.join(_figdir(), f"09_proba_over_time_{model_name}.png")
    d = pred.sort_values("timestamp")
    fails = d[d.label == 1]
    healthy = d[d.label == 0]
    if len(healthy) > max_points:
        healthy = healthy.sample(max_points, random_state=42).sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(healthy["timestamp"], healthy["proba"], s=2, alpha=0.3,
               color="#2ca02c", label="healthy")
    ax.scatter(fails["timestamp"], fails["proba"], s=8, alpha=0.8,
               color="#d62728", label="failure")
    ax.set_xlabel("timestamp (unix)")
    ax.set_ylabel("predicted P(failure)")
    ax.set_title(f"Failure probability over time — {model_name} (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def comparison_table(results, path=None):
    rows = [{k: r.get(k) for k in
             ("model", "accuracy", "precision", "recall", "f1",
              "roc_auc", "pr_auc", "train_time_s")} for r in results]
    df = pd.DataFrame(rows)
    md = df.to_markdown(index=False, floatfmt=".4f")
    if path:
        with open(path, "w") as f:
            f.write(md + "\n")
        df.to_csv(path.replace(".md", ".csv"), index=False)
    return df, md
