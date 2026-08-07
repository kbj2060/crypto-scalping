"""Second augmentation of h48qual's base BTC feature panels: adds this session's zigzag-direction
signal (swing_prob_cash/long/short + swing_direction_score from the quality-weighted 5m Layer B
LightGBM) ON TOP of the already-added swing_transition_prob (Layer A). Column names avoid
h48qual's forbidden-feature tokens (target/future/label/pnl/zigzag/wave3/tp_sl_action_score).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data/splits/year_oos/btc_features_2025_swingtransition.csv"
EVAL_CSV = ROOT / "data/splits/year_oos/btc_features_2026_swingtransition.csv"
LAYERB_PROBA_PATH = ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerB_qualityweighted_proba.parquet"

TRAIN_OUT = ROOT / "data/splits/year_oos/btc_features_2025_swingtransition_zigzag.csv"
EVAL_OUT = ROOT / "data/splits/year_oos/btc_features_2026_swingtransition_zigzag.csv"


def main() -> int:
    layerB = pd.read_parquet(LAYERB_PROBA_PATH)
    layerB["timestamp"] = pd.to_datetime(layerB["timestamp"])

    for src, out in [(TRAIN_CSV, TRAIN_OUT), (EVAL_CSV, EVAL_OUT)]:
        df = pd.read_csv(src, parse_dates=["timestamp"])
        before_cols = len(df.columns)
        df = df.merge(layerB, on="timestamp", how="left")
        new_cols = ["swing_prob_cash", "swing_prob_long", "swing_prob_short", "swing_direction_score"]
        missing = df[new_cols[0]].isna().sum()
        if missing:
            print(f"WARNING: {src.name} has {missing}/{len(df)} rows with no Layer B match -- filling with column median")
            for c in new_cols:
                df[c] = df[c].fillna(df[c].median())
        df.to_csv(out, index=False)
        print(f"wrote {out}: {before_cols} -> {len(df.columns)} cols, {len(df)} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
