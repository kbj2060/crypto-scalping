from __future__ import annotations

import os
import sys
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENSEMBLE_DIR = os.path.dirname(_THIS_DIR)
_ROOT_DIR = os.path.dirname(_ENSEMBLE_DIR)
for _p in (_ROOT_DIR, _ENSEMBLE_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from features.engineering import ULTIMATE_FEATURE_COLS
from features.high_order_state import HIGH_ORDER_STATE_COLS
from features.high_order_state import add_high_order_state_features
from ensemble.supervised.train_trend_xgb import compute_atr, make_triple_barrier_label

logger = logging.getLogger(__name__)


DEFAULT_DATA_PATH = "data/splits/year_oos/training_features_2024.csv"
DEFAULT_RL_DATA_PATH = "data/splits/year_oos/rl_base_2024.csv"

PRED_CONF_MAP = {}

RL_SIG_COLS = [
    "sig_whale",
    "sig_orderblock",
    "sig_oi_divergence",
    "sig_ai_squeeze",
    "sig_garch_regime",
    "sig_ou_mean_rev",
    "sig_jump_rebound",
    "sig_evt_tail",
    "sig_volume_confirm",
    "sig_liquidity_trap",
    "sig_trend_health",
]

RL_ALPHA_COLS = [
    "garch_vol_z",
    "ou_funding_z",
    "ou_halflife",
    "jump_flag",
    "jump_z",
    "evt_tail_flag",
    "evt_excess_z",
    "mshd",
    "fvci",
    "wpad",
    "fdlv",
    "vsdi",
    "vebr",
    "tlad",
    "mtmb",
    "fcsz",
]

CATEGORICAL_HINTS = [
    "session_asia",
    "session_europe",
    "session_us",
    "cvp_regime",
    "regime_break",
]

FORBIDDEN_FEATURE_NAMES = {
    "regime_bull",
    "regime_bear",
    "regime_chop",
    "regime_whipsaw",
    "regime_normal",
}

FORBIDDEN_FEATURE_PREFIXES = (
    "m7_",
    "teacher_",
    "a5dir_",
    "clean_regime_2024_unsup_v4_",
    "clean_regime4_2024_unsup_v1_",
)

FORBIDDEN_FEATURE_FRAGMENTS = (
    "future",
    "target",
    "label",
    "trade_pnl",
    "cash_after",
    "_x_trend_prob",
    "_x_instability_prob",
    "_x_chop_prob",
)


def is_forbidden_feature(col: str) -> bool:
    c = str(col)
    lc = c.lower()
    if c in FORBIDDEN_FEATURE_NAMES:
        return True
    if any(c.startswith(prefix) for prefix in FORBIDDEN_FEATURE_PREFIXES):
        return True
    return any(fragment in lc for fragment in FORBIDDEN_FEATURE_FRAGMENTS)


def _combine_pred_conf(df: pd.DataFrame) -> pd.DataFrame:
    for pred_col, conf_col in PRED_CONF_MAP.items():
        sig_col = pred_col.replace("pred_", "signal_")
        if sig_col in df.columns:
            continue
        if pred_col in df.columns and conf_col in df.columns:
            df[sig_col] = df[pred_col] * df[conf_col]
        elif pred_col in df.columns:
            df[sig_col] = df[pred_col]
    return df


def load_feature_frame(
    data_path: str = DEFAULT_DATA_PATH,
    rl_path: str = DEFAULT_RL_DATA_PATH,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data file not found: {data_path}")

    df = pd.read_csv(data_path, parse_dates=[timestamp_col])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if os.path.exists(rl_path):
        df_rl = pd.read_csv(rl_path, parse_dates=[timestamp_col])
        df_rl.replace([np.inf, -np.inf], np.nan, inplace=True)
        extra_cols = [c for c in df_rl.columns if c not in (timestamp_col,) and c not in df.columns]
        if extra_cols:
            df = df.merge(df_rl[[timestamp_col] + extra_cols], on=timestamp_col, how="inner")

    df = _combine_pred_conf(df)
    if any(c not in df.columns for c in HIGH_ORDER_STATE_COLS):
        logger.info("auto-building missing high-order state features: %s", [c for c in HIGH_ORDER_STATE_COLS if c not in df.columns])
        df = add_high_order_state_features(df)
    df.sort_values(timestamp_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def select_feature_columns(
    df: pd.DataFrame,
    must_include: List[str] | None = None,
) -> List[str]:
    base_must_include = list(must_include or [])
    for c in HIGH_ORDER_STATE_COLS:
        if c in df.columns and c not in base_must_include:
            base_must_include.append(c)
    signal_cols = [p.replace("pred_", "signal_") for p in PRED_CONF_MAP]

    extra_cols = signal_cols + RL_SIG_COLS + RL_ALPHA_COLS + [
        "mtf_trend_1h",
        "mtf_trend_4h",
        "ret_12",
        "ret_24",
        "ret_48",
        "trend_accel",
        "hh_count_24",
        "hl_count_24",
    ]

    cols: List[str] = []
    for c in ULTIMATE_FEATURE_COLS:
        if c in df.columns and c not in cols:
            cols.append(c)
    for c in extra_cols + base_must_include:
        if c in df.columns and c not in cols:
            cols.append(c)
    cols = [c for c in cols if not is_forbidden_feature(c)]
    ordered = [c for c in base_must_include if c in cols]
    ordered.extend([c for c in cols if c not in ordered])
    return ordered


def make_triple_barrier_targets(
    df: pd.DataFrame,
    atr_mult: float = 0.8,
    max_hold: int = 12,
    atr_window: int = 14,
) -> np.ndarray:
    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64) if "high" in df.columns else closes
    lows = df["low"].values.astype(np.float64) if "low" in df.columns else closes
    atr = compute_atr(highs, lows, closes, window=atr_window)

    labels = np.full(len(df), -1, dtype=np.int64)
    for t in range(1, len(df) - max_hold - 1):
        lbl, _, _ = make_triple_barrier_label(
            closes=closes,
            atr=atr,
            t=t,
            highs=highs,
            lows=lows,
            atr_mult=atr_mult,
            max_hold=max_hold,
        )
        labels[t] = lbl
    return labels


def make_future_return(df: pd.DataFrame, horizon: int = 12) -> np.ndarray:
    close = df["close"].values.astype(np.float64)
    fwd = np.full(len(df), np.nan, dtype=np.float64)
    for i in range(len(df) - horizon):
        fwd[i] = close[i + horizon] / max(close[i], 1e-8) - 1.0
    return fwd


def time_split_indices(
    n: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n <= 10:
        raise ValueError(f"not enough rows for split: {n}")
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, n - n_train - n_val)
    if n_train + n_val + n_test > n:
        n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("train split is empty")

    idx = np.arange(n)
    tr = idx[:n_train]
    va = idx[n_train:n_train + n_val]
    te = idx[n_train + n_val:n_train + n_val + n_test]
    return tr, va, te


def median_fill_by_train(
    x_train: pd.DataFrame,
    x_other: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    med = x_train.median(numeric_only=True)
    return x_train.fillna(med), x_other.fillna(med)
