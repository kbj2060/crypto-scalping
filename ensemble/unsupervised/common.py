from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from ensemble.supervised.common import (
    load_feature_frame,
    select_feature_columns,
    DEFAULT_DATA_PATH,
    DEFAULT_RL_DATA_PATH,
)

logger = logging.getLogger(__name__)


VOLATILITY_FEATURE_HINTS = [
    "garch_vol_z",
    "bb_width_z",
    "realized_vol_ratio",
    "garman_klass_vol",
    "parkinson_vol",
    "rogers_satchell_vol",
]

ORDERFLOW_FEATURE_HINTS = [
    "net_taker_ratio",
    "oi_change_rate",
    "smart_money_flow",
    "whale_retail_ratio",
    "sig_whale",
    "sig_oi_divergence",
    "sig_volume_confirm",
]


def load_unsup_frame(data_path: str = DEFAULT_DATA_PATH, rl_path: str = DEFAULT_RL_DATA_PATH) -> pd.DataFrame:
    return load_feature_frame(data_path=data_path, rl_path=rl_path)


def select_numeric_features(df: pd.DataFrame, min_features: int = 16) -> List[str]:
    candidates = select_feature_columns(df)
    numeric = [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) < min_features:
        extra = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in numeric]
        numeric.extend(extra[: max(0, min_features - len(numeric))])
    return numeric


def rank_features_by_variance(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    valid_cols = [c for c in feature_cols if c in df.columns]
    if not valid_cols:
        return []
    variances = df[valid_cols].replace([np.inf, -np.inf], np.nan).var(axis=0, numeric_only=True).fillna(0.0)
    return variances.sort_values(ascending=False).index.tolist()


def zscore_fit_transform(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x_norm = (np.nan_to_num(x, nan=0.0) - mean) / std
    return x_norm, mean, std

