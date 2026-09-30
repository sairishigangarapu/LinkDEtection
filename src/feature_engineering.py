"""Feature engineering (strictly causal: no future information).

Two groups of features:
  1. Instantaneous telemetry available at the observation time t:
     ICMP loss / ping / response time, SNMP availability, uptime, hop /
     downstream topology fields, device one-hot, and per-row aggregates of
     the per-interface SNMP counters (total discards / errors, mean bits).
  2. Temporal context from the SAME device's past only (grouped rolling
     statistics with shift(1)): rolling mean / std / min / max and
     delta-from-previous for key signals.

Because every rolling computation is preceded by shift(1) within each
device's time order, features at time t only ever use observations <= t-1
plus the instantaneous measurement at t (the standard detection-baseline
setup: "given telemetry up to and including now, is the link failed now?").
"""
import numpy as np
import pandas as pd

ROLL_COLS = ["ICMP loss", "ICMP ping", "ICMP response time",
             "SNMP agent availability", "Uptime (network)",
             "if_total_discards", "if_total_errors", "if_bits_recv_mean"]


def _interface_aggregates(df):
    df = df.copy()
    recv = [c for c in df.columns if "Bits" in c and "received" in c]
    sent = [c for c in df.columns if "Bits" in c and "sent" in c]
    disc = [c for c in df.columns if "discard" in c.lower()]
    errs = [c for c in df.columns if "error" in c.lower() and "Operational" not in c]
    df["if_bits_recv_mean"] = df[recv].mean(axis=1, skipna=True) if recv else 0.0
    df["if_bits_sent_mean"] = df[sent].mean(axis=1, skipna=True) if sent else 0.0
    df["if_n_ifaces_observed"] = df[recv].notna().sum(axis=1) if recv else 0
    df["if_total_discards"] = df[disc].fillna(0).sum(axis=1) if disc else 0.0
    df["if_total_errors"] = df[errs].fillna(0).sum(axis=1) if errs else 0.0
    df["if_any_discard"] = (df["if_total_discards"] > 0).astype(int)
    df["if_any_error"] = (df["if_total_errors"] > 0).astype(int)
    return df


def _causal_rolling(df, windows):
    """Per-device causal rolling stats. df must be sorted by device, time."""
    df = df.copy()
    for col in [c for c in ROLL_COLS if c in df.columns]:
        g = df.groupby("Device_name")[col]
        prev = g.shift(1)
        df[f"{col}_delta"] = (df[col] - g.shift(1)).fillna(0)
        for w in windows:
            r = prev.rolling(window=w, min_periods=1)
            df[f"{col}_rollmean_{w}"] = r.mean().fillna(df[col])
            df[f"{col}_rollstd_{w}"] = r.std().fillna(0).fillna(0)
            df[f"{col}_rollmin_{w}"] = r.min().fillna(df[col])
            df[f"{col}_rollmax_{w}"] = r.max().fillna(df[col])
    return df


def build_features(df, windows=(5, 15)):
    df = df.copy().sort_values(["Device_name", "timestamp"]).reset_index(drop=True)
    df = _interface_aggregates(df)
    df = _causal_rolling(df, list(windows))
    # Device identity as one-hot (topology proxy until the GNN phase).
    dummies = pd.get_dummies(df["Device_name"], prefix="dev", dtype=int)
    df = pd.concat([df, dummies], axis=1)
    return df.sort_values(["timestamp", "Device_name"]).reset_index(drop=True)


def select_feature_columns(df, drop_cols, drop_substrings=()):
    """All numeric columns except metadata / identifiers / raw label."""
    meta = set(drop_cols) | {"label", "source_file"}
    feats = [c for c in df.columns
             if c not in meta and pd.api.types.is_numeric_dtype(df[c])
             and not any(s in c for s in drop_substrings)]
    return feats
