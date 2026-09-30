# Link Failure Prediction and Localization (RF + XGBoost baseline)

Supervised-learning baseline on the **LCORE-D** core-network telemetry dataset:
predict whether a router's observation indicates a link/node failure and rank
routers per timestamp for failure localisation. Runs fully **locally**
(RTX 4060 for XGBoost, CPU for Random Forest). No Colab, no cloud, no GNN in
this phase (GNN is a documented future extension).

## Quickstart (local)

```bash
cd link-failure-prediction
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 1. Build unified dataset (reads ../LCORE-D* raw CSVs, writes data/unified.parquet)
./.venv/bin/python src/data_loader.py
# 2. Random Forest baseline (CPU, all cores; ~25 min incl. small val grid)
./.venv/bin/python src/train_rf.py
# 3. XGBoost on RTX 4060 (verifies GPU utilisation, refuses CPU fallback)
./.venv/bin/python src/train_xgb.py
# 4. GCN extension (pure-torch, RTX 4060; same rows/split/features as RF/XGB)
./.venv/bin/python src/train_gnn.py
# 5. Linear SVM baseline (liblinear + calibration; same rows/split/features)
./.venv/bin/python src/train_svm.py
# 6. Localisation + figures + comparison table
./.venv/bin/python src/localization.py
./.venv/bin/python src/make_figures.py
```

During XGBoost training, confirm GPU use in another terminal:

```bash
nvidia-smi   # expect >0% GPU util and ~200+ MiB used by python
```

## What the pipeline does

1. **Load** (`src/data_loader.py`): harmonise 7 per-router CSVs (R1–R6 CORE,
   R7 EDGE-MS) into one router-level table (`data/unified.parquet`, 169,682
   rows). Target `y = 1 iff class == 'F'`; `T/TH` rows excluded from the binary
   task (documented below); topology fields (device, hops, downstream,
   interfaces, timestamp) preserved for the future GNN phase.
2. **Clean** (`src/preprocessing.py`): `-1` sentinel → NaN + missing
   indicators; median imputation fitted on train only; constant columns
   dropped. One dirty cell (`` `1 `` in R1 `ICMP ping`) coerced to NaN.
3. **Features** (`src/feature_engineering.py`): instantaneous ICMP/SNMP
   telemetry + per-row interface aggregates + **strictly causal** per-device
   rolling stats (shift(1), windows 5/15) + deltas + device one-hots.
   `Uptime` clocks are dropped (they let trees memorise *when* faults
   happened). Raw `timestamp` is never a feature.
4. **Split**: global temporal 70/15/15 (no shuffling, no future leakage).
5. **Models**: RF small grid (`class_weight × max_depth`, val PR-AUC) refit on
   train+val; XGB (`hist/cuda`, `scale_pos_weight`) with early stopping.
   Operating threshold = val-F1 optimum; ROC/PR-AUC are threshold-free.
6. **Evaluate** (`src/evaluation.py`, `src/localization.py`): same test set,
   comparison table, confusion matrices, ROC/PR, importances, and node-level
   localisation (rank routers per timestamp; Top-1/3/5 + MRR).

## Key results (test set, same for both models)

| Model | Acc. | Prec. | Rec. | F1 | ROC-AUC | PR-AUC | Train |
|-------|-----:|------:|-----:|---:|--------:|-------:|------:|
| Random Forest | 0.903 | 0.463 | 0.079 | 0.135 | 0.776 | 0.276 | ~5 s refit (CPU) |
| XGBoost (RTX 4060) | 0.777 | 0.236 | 0.594 | 0.338 | 0.778 | 0.243 | ~4 s (GPU 68%, 209 MiB) |
| GCN (RTX 4060) | 0.817 | 0.268 | 0.525 | 0.355 | 0.790 | 0.305 | ~1 s (GPU) |
| Linear SVM | 0.811 | 0.240 | 0.447 | 0.312 | 0.767 | 0.210 | ~65 s (CPU) |

Localisation @ failure timestamps (642; median 4 candidate routers):
RF Top-1 0.69 / Top-3 0.89 / MRR 0.80 (≥4-router subset: 0.54/0.81/0.70);
XGB Top-1 0.56 / Top-3 0.90 / MRR 0.73;
GCN Top-1 0.66 / Top-3 0.86 / MRR 0.78 (≥4: 0.49/0.75/0.67);
SVM Top-1 0.65 / Top-3 0.89 / MRR 0.78 (≥4: 0.49/0.81/0.67).

## GNN extension (done)

`src/gnn_data.py` builds per-timestamp snapshot graphs over the inferred
7-node tree `EDGE-MS—R1—R2—R3—{R4,R5,R6}` (from hop counts; documented as
inferred, not ground truth), disjoint-union per split so no message passes
across timestamps or into the test set. `src/train_gnn.py` trains a 2-layer
GCN (131→64→1, pure torch, standardized features, `pos_weight`, early stop
on val PR-AUC) on the RTX 4060. Verdict: neighbour aggregation helps
detection (GCN leads ROC-AUC/PR-AUC/F1) while RF still ranks failures
slightly better for Top-1 localisation — an honest split result, not a
sweep.
Genuine *link*-level localisation is not supported by the data (no link IDs) —
reported as a limitation, node-level ranking used instead.

Two honest findings to mention in the viva:
* `T` (persistent degraded state, loss ≈ 100%) confounds naive F-vs-rest
  learning, so the binary task is H-vs-F; the trained models still score held
  out `T` rows ≈ 0, i.e. they learn more than "loss = 100 ⇒ failure".
* Same-instant telemetry overlaps between H and F near fault boundaries
  (15–32% of healthy rows also show loss = 100), which caps detection quality
  and motivates the GNN phase (neighbour/topology context).

## Layout

```
src/{data_loader,preprocessing,feature_engineering,train_rf,train_xgb,evaluation,localization,make_figures}.py
configs/config.yaml  requirements.txt
data/unified.parquet  models/{random_forest,xgboost}.pkl
results/{figures,metrics,predictions}
```

Reproducibility: seed 42 everywhere; `configs/config.yaml` holds splits,
hyperparameters and drop rules; metrics JSONs store params, timings, GPU
observations and test boundaries. Environment: Python 3.12.3,
scikit-learn 1.9.1, XGBoost 3.4.1 (CUDA build), driver 595.84 / CUDA 13.2,
NVIDIA GeForce RTX 4060 Laptop GPU.

GitHub: intentionally not created yet — submission repo comes at finalisation.
