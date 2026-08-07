"""Offline parity test for BtcSwingTransitionLiveFeature.

Simulates the live provider at several historical cutoffs: a 7000-bar raw 5m buffer (same source
CSVs the offline 1h builder consumed), a 600-bar decision frame carrying the 96 in-frame layerA
inputs (taken from the causalfix panel -- live parity of those 96 cols is already covered by the
existing h48qual live-parity audit), and a DVOL "fetcher" that serves the historical CSV rows the
real REST endpoint would have returned at that time. The provider's per-bar output must match the
offline training feature (tmp/btc_1h_volregime_20260805/btc5m_layerA_pred.parquet) that the
promoted candidate was trained/validated on.

Known tolerated divergence: mtf1h_rsi_14 is EWM-based, so a 7000-bar buffer truncates initial
conditions vs the full-2024 offline series -- decayed to ~1e-19 at the frame rows, far below any
LGBM split threshold in practice; the test still measures and reports the realized probability
difference and fails if it exceeds PROB_ATOL anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_bot_modules.btc_swing_transition_live import BtcSwingTransitionLiveFeature  # noqa: E402

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
PRED_PATH = ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerA_pred.parquet"
DVOL_CSV = ROOT / "data/derivatives/deribit_dvol/BTC_dvol_hourly.csv"
RAW_SOURCES = [ROOT / f"data/splits/year_oos/btc_features_{y}.csv" for y in (2024, 2025, 2026)]
RAW_COLS = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"]

CUTOFFS = ["2025-10-15 12:00:00", "2026-01-20 03:35:00", "2026-03-28 21:10:00"]
BUFFER_BARS = 7000
FRAME_BARS = 600
PROB_ATOL = 1e-6


def main() -> int:
    raw = pd.concat([pd.read_csv(p, usecols=RAW_COLS) for p in RAW_SOURCES], ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    panel = pd.read_parquet(PANEL_PATH)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    pred = pd.read_parquet(PRED_PATH)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    dvol_csv = pd.read_csv(DVOL_CSV)
    dvol_csv["timestamp"] = pd.to_datetime(dvol_csv["timestamp"])

    worst = 0.0
    for cutoff_s in CUTOFFS:
        cutoff = pd.Timestamp(cutoff_s)

        def dvol_fetcher(start_ms: int, end_ms: int, _cutoff=cutoff):
            sel = dvol_csv[(dvol_csv["timestamp"] >= pd.to_datetime(start_ms, unit="ms"))
                           & (dvol_csv["timestamp"] <= _cutoff)]
            return [[int(t.value // 10**6), c, c, c, c]
                    for t, c in zip(sel["timestamp"], sel["close"])]

        provider = BtcSwingTransitionLiveFeature(dvol_fetcher=dvol_fetcher)
        buf = raw[raw["timestamp"] <= cutoff].tail(BUFFER_BARS).reset_index(drop=True)
        in_frame = [c for c in provider.feature_columns
                    if not c.startswith("mtf1h_") and not c.startswith("dvol_btc")]
        frame = panel[panel["timestamp"] <= cutoff].tail(FRAME_BARS)[["timestamp"] + in_frame].reset_index(drop=True)

        out = provider.append(frame, raw_5m=buf)
        merged = out[["timestamp", "swing_transition_prob"]].merge(
            pred.rename(columns={"probA": "expected"}), on="timestamp", how="inner")
        if len(merged) != len(frame):
            raise SystemExit(f"cutoff {cutoff_s}: joined {len(merged)}/{len(frame)} rows with offline predictions")
        diff = np.abs(merged["swing_transition_prob"].to_numpy() - merged["expected"].to_numpy())
        print(f"cutoff {cutoff_s}: rows={len(merged)} max_abs_diff={diff.max():.3e} "
              f"mean_abs_diff={diff.mean():.3e} n_over_atol={(diff > PROB_ATOL).sum()}")
        worst = max(worst, float(diff.max()))

    if worst > PROB_ATOL:
        raise SystemExit(f"PARITY FAIL: worst max_abs_diff={worst:.3e} > {PROB_ATOL}")
    print(f"PARITY PASS: worst max_abs_diff={worst:.3e} <= {PROB_ATOL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
