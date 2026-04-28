"""
=============================================================================
SEZIONE 3.10 — FEATURE ENGINEERING UNIFICATA PER CROSS-DATASET IDS
=============================================================================
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# COSTANTI
# =============================================================================

EPS = 1e-9

UNIFIED_NUMERIC_FEATURES = [
    "bytes_total",
    "bytes_src",
    "bytes_dst",
    "pkts_total",
    "byte_asymmetry",
    "pkt_asymmetry",
    "payload_mean_fwd",
    "payload_mean_bwd",
    "flow_duration_sec",
    "flow_rate",
]

UNIFIED_CATEGORICAL_FEATURES = [
    "proto_unified",
    "service_unified",
    "conn_state_unified",
]

ALL_UNIFIED_FEATURES = UNIFIED_NUMERIC_FEATURES + UNIFIED_CATEGORICAL_FEATURES


# =============================================================================
# UTILS ROBUSTI
# =============================================================================

def _safe_series(df: pd.DataFrame, col: str, default=0):
    """Garantisce SEMPRE una Series (fix definitivo bug CIC)."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).fillna(0.0)


def _log1p_safe(s) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return np.log1p(s.fillna(0).clip(lower=0))


def _normalize_proto(x, n: int) -> pd.Series:
    mapping = {"6": "tcp", "17": "udp", "1": "icmp"}

    if isinstance(x, pd.Series):
        s = x
    else:
        s = pd.Series([x] * n)

    s = s.astype(str).str.lower().str.strip()
    return s.map(lambda v: mapping.get(v, "other")).fillna("other")


# =============================================================================
# TON FEATURE ENGINEERING
# =============================================================================

def build_unified_features_ton(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    n = len(d)

    src_bytes = _safe_series(d, "src_bytes", 0)
    dst_bytes = _safe_series(d, "dst_bytes", 0)
    src_pkts = _safe_series(d, "src_pkts", 1)
    dst_pkts = _safe_series(d, "dst_pkts", 1)
    duration = _safe_series(d, "duration", 1)

    total = src_bytes + dst_bytes
    pkts = src_pkts + dst_pkts

    return pd.DataFrame({
        "bytes_total": _log1p_safe(total),
        "bytes_src": _log1p_safe(src_bytes),
        "bytes_dst": _log1p_safe(dst_bytes),
        "pkts_total": _log1p_safe(pkts),

        "byte_asymmetry": _safe_div(src_bytes - dst_bytes, total + EPS),
        "pkt_asymmetry": _safe_div(src_pkts - dst_pkts, pkts + EPS),

        "payload_mean_fwd": _log1p_safe(_safe_div(src_bytes, src_pkts)),
        "payload_mean_bwd": _log1p_safe(_safe_div(dst_bytes, dst_pkts)),

        "flow_duration_sec": _log1p_safe(duration),
        "flow_rate": _log1p_safe(_safe_div(total, duration)),

        "proto_unified": _normalize_proto(d.get("proto", "other"), n),
        "service_unified": pd.Series(["other"] * n),
        "conn_state_unified": pd.Series(["other"] * n),
    })


# =============================================================================
# CIC FEATURE ENGINEERING 
# =============================================================================

def build_unified_features_cic(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = d.columns.str.lower().str.strip()

    n = len(d)

    # ✔ colonne REALI dal tuo dataset
    total_bytes = _safe_series(d, "tot sum", 0)
    min_val = _safe_series(d, "min", 0)
    max_val = _safe_series(d, "max", 0)
    avg_val = _safe_series(d, "avg", 0)
    std_val = _safe_series(d, "std", 0)

    # FIX CRITICO: nomi multipli (flow_duration vs duration)
    duration = (
        _safe_series(d, "flow_duration", np.nan)
        .replace(0, np.nan)
        .fillna(_safe_series(d, "duration", 1))
    )

    # ✔ proxy più stabile dei pacchetti
    pkts = _safe_series(d, "number", avg_val + 1)

    return pd.DataFrame({
        "bytes_total": _log1p_safe(total_bytes),
        "bytes_src": _log1p_safe(min_val),
        "bytes_dst": _log1p_safe(max_val),
        "pkts_total": _log1p_safe(pkts),

        "byte_asymmetry": _safe_div(max_val - min_val, total_bytes + EPS),
        "pkt_asymmetry": _safe_div(std_val, pkts + EPS),

        "payload_mean_fwd": _log1p_safe(avg_val),
        "payload_mean_bwd": _log1p_safe(std_val),

        "flow_duration_sec": _log1p_safe(duration),
        "flow_rate": _log1p_safe(_safe_div(total_bytes, duration)),

        "proto_unified": pd.Series(["other"] * n),
        "service_unified": pd.Series(["other"] * n),
        "conn_state_unified": pd.Series(["other"] * n),
    })


# =============================================================================
# PREPROCESSOR
# =============================================================================

def build_unified_preprocessor() -> ColumnTransformer:
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num, UNIFIED_NUMERIC_FEATURES),
        ("cat", cat, UNIFIED_CATEGORICAL_FEATURES),
    ])


# =============================================================================
# HELPERS
# =============================================================================

def prepare_ton_for_unified_space(df: pd.DataFrame, target_col="label"):
    return (
        build_unified_features_ton(df.drop(columns=[target_col])),
        df[target_col]
    )


def prepare_cic_for_unified_space(df: pd.DataFrame, target_col="label"):
    return (
        build_unified_features_cic(df.drop(columns=[target_col])),
        df[target_col]
    )


# =============================================================================
# DOMAIN SHIFT
# =============================================================================

def compute_distribution_shift(X_source: pd.DataFrame, X_target: pd.DataFrame):
    rows = []

    for col in UNIFIED_NUMERIC_FEATURES:
        if col not in X_source or col not in X_target:
            continue

        s = X_source[col].mean()
        t = X_target[col].mean()

        shift = abs(s - t) / (X_source[col].std() + EPS)

        rows.append({
            "feature": col,
            "source_mean": s,
            "target_mean": t,
            "norm_shift": shift,
        })

    return pd.DataFrame(rows).sort_values("norm_shift", ascending=False)