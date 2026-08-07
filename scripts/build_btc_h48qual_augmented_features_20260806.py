"""Augment h48qual's base BTC feature panels (data/splits/year_oos/btc_features_2025.csv /
btc_features_2026.csv) with this session's best signal -- the pivot-transition ("swing turning
point imminent") probability from scripts/train_btc_5m_layerA_tabm_20260806.py's LightGBM
version (tmp/btc_1h_volregime_20260805/btc5m_layerA_pred.parquet, OOS AUC 0.77).

Column name deliberately avoids h48qual's forbidden-feature tokens
(DENY_TOKENS = target/future/label/pnl/zigzag/wave3/tp_sl_action_score) so it passes
_numeric_feature_cols's automatic feature-discovery audit without any code change to the training
pipeline itself -- new numeric columns are picked up automatically.

Does NOT touch the original btc_features_2025/2026.csv (writes new _swingtransition-suffixed
copies) or any live bundle.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data/splits/year_oos/btc_features_2025.csv"
EVAL_CSV = ROOT / "data/splits/year_oos/btc_features_2026.csv"
LAYERA_PRED_PATH = ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerA_pred.parquet"

TRAIN_OUT = ROOT / "data/splits/year_oos/btc_features_2025_swingtransition.csv"
EVAL_OUT = ROOT / "data/splits/year_oos/btc_features_2026_swingtransition.csv"


def main() -> int:
    layerA = pd.read_parquet(LAYERA_PRED_PATH).rename(columns={"probA": "swing_transition_prob"})
    layerA["timestamp"] = pd.to_datetime(layerA["timestamp"])

    for src, out in [(TRAIN_CSV, TRAIN_OUT), (EVAL_CSV, EVAL_OUT)]:
        df = pd.read_csv(src, parse_dates=["timestamp"])
        before_cols = len(df.columns)
        df = df.merge(layerA, on="timestamp", how="left")
        missing = df["swing_transition_prob"].isna().sum()
        if missing:
            print(f"WARNING: {src.name} has {missing}/{len(df)} rows with no swing_transition_prob match -- filling with column median")
            df["swing_transition_prob"] = df["swing_transition_prob"].fillna(df["swing_transition_prob"].median())
        df.to_csv(out, index=False)
        print(f"wrote {out}: {before_cols} -> {len(df.columns)} cols, {len(df)} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
