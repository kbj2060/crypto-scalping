"""Regenerate the quality-weighted 5m Layer B LightGBM (best Layer B variant this session), this
time saving predict_proba (not just argmax) for all bars, so the CASH/LONG/SHORT probabilities can
be merged into h48qual's feature panel as new columns (h48qual integration, zigzag half).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
ZIGZAG_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_labels_20260806.parquet"
QUALITY_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_quality_oracle_20260806.parquet"
DVOL_PATH = ROOT / "data/derivatives/deribit_dvol/BTC_dvol_hourly.csv"
OUT_PATH = ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerB_qualityweighted_proba.parquet"

DROP_RAW = {"timestamp", "open", "high", "low", "close", "close_btc", "volume_btc", "quote_volume_btc"}
VAL_START = "2025-09-01"
WEIGHT_SCALE = 40.0
WEIGHT_MIN, WEIGHT_MAX = 0.2, 3.0


def build_dvol_features() -> pd.DataFrame:
    df = pd.read_csv(DVOL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["available_at"] = df["timestamp"] + pd.Timedelta(hours=1)
    df = df[["available_at", "close"]].rename(columns={"available_at": "timestamp", "close": "dvol_btc"}).sort_values("timestamp")
    df["dvol_btc_roc_24h"] = df["dvol_btc"].pct_change(24)
    df["dvol_btc_roc_168h"] = df["dvol_btc"].pct_change(168)
    df["dvol_btc_pctrank_720h"] = df["dvol_btc"].rolling(720, min_periods=180).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)
    return df


def main() -> int:
    panel = pd.read_parquet(PANEL_PATH)
    labels = pd.read_parquet(ZIGZAG_PATH, columns=["timestamp", "zigzag_action"])
    qual = pd.read_parquet(QUALITY_PATH, columns=["timestamp", "net_ret_sim"])
    dvol = build_dvol_features()

    df = panel.merge(labels, on="timestamp", how="inner").merge(qual, on="timestamp", how="inner")
    df = pd.merge_asof(df.sort_values("timestamp"), dvol, on="timestamp", direction="backward")
    df = df.dropna(subset=["zigzag_action"]).reset_index(drop=True)

    feature_cols = [c for c in panel.columns if c not in DROP_RAW] + [
        "dvol_btc", "dvol_btc_roc_24h", "dvol_btc_roc_168h", "dvol_btc_pctrank_720h",
    ]
    X = df[feature_cols]
    y = df["zigzag_action"].astype(int)
    train_mask = (df["timestamp"] < VAL_START).to_numpy()

    sw = 1.0 + df["net_ret_sim"].fillna(0.0).to_numpy() * WEIGHT_SCALE
    sample_weight = np.clip(sw, WEIGHT_MIN, WEIGHT_MAX)[train_mask]

    clf = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, min_child_samples=100, verbosity=-1)
    clf.fit(X[train_mask], y[train_mask], sample_weight=sample_weight)
    proba = clf.predict_proba(X)  # columns ordered by class label 0,1,2 = CASH,LONG,SHORT

    out = df[["timestamp"]].copy()
    out["swing_prob_cash"] = proba[:, 0]
    out["swing_prob_long"] = proba[:, 1]
    out["swing_prob_short"] = proba[:, 2]
    out["swing_direction_score"] = proba[:, 1] - proba[:, 2]
    out.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}, shape={out.shape}")
    print(out.describe().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
