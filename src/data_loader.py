"""Load and harmonise the raw LCORE-D router files into one router-level table.

Each raw CSV holds one row per (router, timestamp) with per-interface SNMP
columns. There is no explicit link_id / src / dst in the dataset, so the
localisation unit preserved here is the router (device) observed at a
timestamp. Per-interface columns are kept for aggregate feature engineering,
and device / hop / timestamp columns are preserved for future GNN use
(graph nodes = devices, edges = interfaces between them).
"""
import glob
import os

import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    path = path or os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _standardise_device(df):
    """R1-R3,R7 use `Device_name`; R4-R6 use `router_name` (and case variants)."""
    rename = {}
    for c in df.columns:
        low = c.strip().lower()
        if low in ("device_name", "router_name"):
            rename[c] = "Device_name"
        elif low in ("hop_to_server", "hop_to_core"):
            rename[c] = "_".join(w.capitalize() for w in low.split("_to_"))
            # -> Hop_to_server / Hop_to_core to match R1-R3 naming
            rename[c] = "Hop_to_server" if "server" in low else "Hop_to_core"
    return df.rename(columns=rename)


def load_raw(config=None):
    """Return the unified router-level dataframe (one row per device x time)."""
    config = config or load_config()
    raw_dir = os.path.join(PROJECT_ROOT, config["raw_data_dir"])
    frames = []
    for fp in sorted(glob.glob(os.path.join(raw_dir, "LCORE-D R*.csv"))):
        df = pd.read_csv(fp, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df = _standardise_device(df)
        # Drop fully-empty artefact columns (e.g. R4 'Unnamed: 42'...).
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
        # Dirty-data guard: raw files contain stray strings (e.g. '`1' in
        # R1's ICMP ping). Coerce every non-numeric, non-identifier column.
        for c in df.columns:
            if c not in ("class", "Device_name", "timestamp") and not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["source_file"] = os.path.basename(fp)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True).copy()

    # Binary target: 1 = failure (class == 'F'), 0 = everything else (H/T/TH).
    # Rows with a missing class label cannot be supervised -> drop (30 in R7).
    full = full.dropna(subset=["class"]).reset_index(drop=True)
    full["label"] = (full["class"] == config["positive_label"]).astype(int)

    full["timestamp"] = full["timestamp"].astype("int64")
    full = full.sort_values(["timestamp", "Device_name"]).reset_index(drop=True)
    return full


def load_fault_log(config=None):
    """Parse the fault-injection log into a tidy frame (UTC datetimes)."""
    config = config or load_config()
    fp = os.path.join(PROJECT_ROOT, config["raw_data_dir"], config["fault_log_name"])
    fl = pd.read_csv(fp, usecols=lambda c: "Unnamed" not in str(c))
    fl = fl.dropna(how="all").reset_index(drop=True)
    fl.columns = [c.strip() for c in fl.columns]
    for c in ("Start  Time", "End Time"):
        fl[c] = pd.to_datetime(fl[c], format="%d-%m-%Y %H:%M", utc=True)
    return fl


if __name__ == "__main__":
    cfg = load_config()
    df = load_raw(cfg)
    print(f"rows={len(df)} cols={len(df.columns)} devices={sorted(df.Device_name.unique())}")
    print("label counts:", df.label.value_counts().to_dict(),
          "| failure %:", round(100 * df.label.mean(), 3))
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    df.to_parquet(os.path.join(PROJECT_ROOT, "data", "unified.parquet"), index=False)
    print("saved data/unified.parquet")
