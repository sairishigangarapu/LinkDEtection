"""Cleaning + missing-value handling for the unified LCORE-D table.

Key dataset quirks handled here:
  * `-1` is a sentinel for "measurement unavailable" in ICMP loss, all
    `Bits received/sent` counters and `Uptime (network)` -> converted to NaN.
  * A handful of NaNs in operational-status / uptime columns.
  * Zero-variance columns (path_up, all operational statuses, never-varying
    interface counters) carry no signal -> dropped (list fitted on train).

Imputation statistics (medians) and the drop list are fitted on the TRAIN
split only, then applied to validation/test, so no information leaks across
the temporal split.
"""
import numpy as np
import pandas as pd

SENTINEL_MINUS_ONE = ["ICMP loss", "Uptime (network)"]


def _bits_cols(df):
    return [c for c in df.columns if "Bits" in c]


def clean(df):
    """Sentinel -> NaN conversion + missing indicators. Returns a copy."""
    df = df.copy()
    for c in SENTINEL_MINUS_ONE:
        if c in df.columns:
            df[c] = df[c].replace(-1, np.nan).replace(-1.0, np.nan)
    for c in _bits_cols(df):
        df[c] = df[c].replace(-1, np.nan).replace(-1.0, np.nan)
    df["icmp_loss_missing"] = df["ICMP loss"].isna().astype(int) if "ICMP loss" in df.columns else 0
    df["uptime_missing"] = df["Uptime (network)"].isna().astype(int) if "Uptime (network)" in df.columns else 0
    # Operational status is constant == 1 wherever observed -> fill, drop later.
    for c in [c for c in df.columns if "Operational" in c]:
        df[c] = df[c].fillna(1)
    return df


def fit_imputation(train_df, feature_cols):
    """Median of each numeric feature column, computed on train only."""
    medians = {}
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[c]):
            medians[c] = train_df[c].median()
    return medians


def apply_imputation(df, medians):
    df = df.copy()
    for c, m in medians.items():
        if c in df.columns:
            df[c] = df[c].fillna(m)
    return df


def constant_columns(df, feature_cols):
    """Columns with a single unique value on train (no signal)."""
    return [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
