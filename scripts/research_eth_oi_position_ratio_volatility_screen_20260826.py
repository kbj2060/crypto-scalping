"""Volatility-framing screen for the raw OI/position-skew columns that failed DIRECTION
yesterday (research_eth_oi_position_ratio_raw_direction_screen_20260826.py, 0/8 survivors).

User (2026-08-26): "방향 말고 다른 축에서는 어떻게 쓰일 수 있을까?" -- this repo's own established
escape route for a data family that fails direction is the volatility-precursor reframing, already
proven for oi_delta_pct (1.21-1.45x realized-range lift, monotonic, direction-symmetric, now the
live "OI 급변" dashboard chip) and re-confirmed with TRAIN/VAL + autocorrelation-controlled partial
IC in research_eth_model_indicator_volatility_screen_20260825.py. That screen covered 9 microstructure_1m
features + whale_position_score + shadow_aftershock_prob, but never the RAW oi_z (level, not delta)
or the position-ratio features (top_pos_z/retail_pos_z) -- this script fills exactly that gap, on
data/TOTAL_ETHUSDT_metrics_2024_2026.csv (not microstructure_1m, since these columns don't live
there) rather than inventing new methodology.

Reused VERBATIM by import (not re-derived) from research_eth_model_indicator_volatility_screen_
20260825.py: shift_z (circular-shift permutation), fwd_range_pct/bwd_range_pct (target + the
autocorrelation-control confound), rolling_absz, partial_corr_check, screen_split. Only the input
frame (features + join) is new. oi_delta_z rides along as an in-script replication anchor -- exact
same formula as the live "OI 급변" chip and as oi_delta_pct, just sourced from the OI archive CSV
join instead of microstructure_1m/oi_lsratio_5m -- if it doesn't reproduce the known ~1.2-1.5x
lift shape, the join/pipeline is wrong and the other 3 features should not be trusted.

Candidates (all rolling-288/24h |z|, min_periods=259):
  - oi_delta_z: sum_open_interest.diff() -- replication anchor, matches live_oi_delta_signal_20260824.py
  - oi_z: sum_open_interest_value LEVEL (not delta) -- untested in this framing before now
  - top_pos_z: sum_toptrader_long_short_ratio -- untested in this framing before now
  - retail_pos_z: count_long_short_ratio -- untested in this framing before now

Same TRAIN (2026-05-03~07-31) / VAL (2026-08-01~08-16) split as every prior screen in this family,
for direct comparability. Nothing here is a promotion/deployment decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research_eth_model_indicator_volatility_screen_20260825 import (  # noqa: E402
    Z_MINP, Z_WIN, bwd_range_pct, fwd_range_pct, partial_corr_check, screen_split,
)

KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OI_ARCHIVE_CSV = ROOT / "data/TOTAL_ETHUSDT_metrics_2024_2026.csv"

TRAIN_START, TRAIN_END = "2026-05-03", "2026-07-31"
VAL_START, VAL_END = "2026-08-01", "2026-08-16"

FEATURES = ["oi_delta_z", "oi_z", "top_pos_z", "retail_pos_z"]  # oi_delta_z = replication anchor


def build_frame() -> pd.DataFrame:
    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"])
    klines = klines.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    klines = klines[(klines["timestamp"] >= "2026-04-28") & (klines["timestamp"] <= "2026-08-17")].reset_index(drop=True)
    klines["bar_close_time"] = klines["timestamp"] + pd.Timedelta(minutes=5)

    oi = pd.read_csv(OI_ARCHIVE_CSV, usecols=["create_time", "sum_open_interest", "sum_open_interest_value",
                                               "sum_toptrader_long_short_ratio", "count_long_short_ratio"])
    oi["ts"] = pd.to_datetime(oi["create_time"])
    oi = oi.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)

    def roll_z(s: pd.Series) -> pd.Series:
        # NOTE: this pre-computed z is what screen_split/partial_corr_check will ALSO re-z-score
        # internally via rolling_absz on the raw column -- so we pass the RAW (undetrended) level
        # here, matching how the reference script's own features (e.g. oi_delta_pct) are raw
        # columns, not pre-z-scored, before screen_split gets them.
        return s

    # raw candidate columns (screen_split/rolling_absz apply the rolling-288 z-score themselves)
    oi["oi_delta_z"] = oi["sum_open_interest"].diff()
    oi["oi_z"] = oi["sum_open_interest_value"]
    oi["top_pos_z"] = oi["sum_toptrader_long_short_ratio"]
    oi["retail_pos_z"] = oi["count_long_short_ratio"]

    m = pd.merge_asof(
        klines.sort_values("bar_close_time"), oi[["ts"] + FEATURES].sort_values("ts"),
        left_on="bar_close_time", right_on="ts", direction="backward", tolerance=pd.Timedelta("5min"),
    )
    for c in FEATURES:
        klines[c] = m[c].to_numpy()
    return klines


def main() -> None:
    klines = build_frame()
    close = klines["close"]
    klines["fwd_h12_1h"] = fwd_range_pct(klines["high"], klines["low"], close, 12)
    klines["fwd_h48_4h"] = fwd_range_pct(klines["high"], klines["low"], close, 48)
    klines["bwd_h12_1h"] = bwd_range_pct(klines["high"], klines["low"], close, 12)

    train_mask = (klines["timestamp"] >= TRAIN_START) & (klines["timestamp"] <= TRAIN_END)
    val_mask = (klines["timestamp"] >= VAL_START) & (klines["timestamp"] <= VAL_END)

    print(f"{'='*110}\nOI/POSITION-RATIO VOLATILITY-FRAMING SCREEN -- TRAIN {TRAIN_START}~{TRAIN_END}\n"
          f"(Z_WIN={Z_WIN}, Z_MINP={Z_MINP}, oi_delta_z = replication anchor)\n{'='*110}")
    screen_split(klines, train_mask, "TRAIN", FEATURES)
    partial_corr_check(klines, train_mask, "TRAIN", FEATURES)

    print(f"\n{'='*110}\nVAL {VAL_START}~{VAL_END} CONFIRMATION -- all 4 re-run (small battery, "
          f"cheap enough not to pre-filter)\n{'='*110}")
    screen_split(klines, val_mask, "VAL", FEATURES)
    partial_corr_check(klines, val_mask, "VAL", FEATURES)

    print(f"\n{'='*110}\nNothing above is a promotion/deployment decision.\n{'='*110}")


if __name__ == "__main__":
    main()
