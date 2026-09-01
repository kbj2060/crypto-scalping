#!/usr/bin/env python3
"""BTC Phase 1 -- scalping-scale regime LABEL geometry sweep, the BTC counterpart of
research_eth_regime_scalping_label_geometry_20260902.py. User goal 2026-09-02: "같은 논리로 btc
레짐도 최고 수준으로 만들어서 대시보드에 배포해줘".

WHY BTC IS A BIGGER JOB THAN ETH WAS. ETH already had a deployed regime classifier and this work
only swapped its label. BTC has NO dashboard regime classifier at all -- the BTC snapshot chart's
ribbon is a hard-coded grey "model not available" band (app.js renderCandleSvg, gated on
activeSnapshotAsset === "eth"; see memory eth-dashboard-btc-regime-classifier-not-trained-todo-
20260831 for why that guard exists -- it was stopping ETH's regime being drawn on BTC candles).
So BTC needs label + model + live scorer + dashboard wiring, not a one-line MODEL_PATH swap.

⭐PARAMETERS ARE RE-SCREENED ON BTC, NOT PORTED. ETH landed on S=12/K=3, but this repo has a
direct precedent for that not transferring: the DeMarker/Kalman BTC port
(btc_v_rebound_feeder_gap_threshold_screen_20260901) found ETH's GAP/threshold choices had never
been re-validated for the new objective and did not carry over -- Kalman actively hurt on BTC and
had to be dropped. So the full S x K grid is re-run here and BTC picks its own optimum.

Everything else mirrors the ETH study exactly: same scale-parameterized 3-class family, same
percentile-matched threshold calibration (anchored to RegimeEngine's own firing rates so class
shares stay comparable across scales), same transition-edge metric with a BLOCK bootstrap CI
(added on ETH after point estimates alone nearly produced a wrong call).

⚠️ OOS (2026-07-01~2026-08-01) IS NOT TOUCHED in this phase -- TRAIN only. BTC's canonical feature
file ends 2026-08-01 17:40, so BTC's OOS is 9,141 bars (~32d) vs ETH's 14,400 (~50d).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from features.elite import RegimeEngine  # noqa: E402
from research_eth_regime_scalping_label_geometry_20260902 import (  # noqa: E402
    DEBOUNCES, FWD_HORIZONS, SCALES, _debounce, _run_lengths, efficiency_ratio,
    scaled_label, transition_edge,
)

BTC_CSV = ROOT / "data/splits/year_oos/btc_features_2024_2026.csv"
TRAIN_START = pd.Timestamp("2024-01-01T00:00:00")
TRAIN_END = pd.Timestamp("2026-06-30T23:55:00")
OUT_DIR = ROOT / "tmp/btc_regime_scalping_label_geometry_20260902"


def load_btc_train() -> pd.DataFrame:
    df = pd.read_csv(BTC_CSV, usecols=["timestamp", "open", "high", "low", "close", "volume"],
                     parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return df[(df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)].reset_index(drop=True)


def main() -> None:
    df = load_btc_train()
    close = df["close"]
    close_np = close.to_numpy()
    print(f"BTC TRAIN {len(df):,} bars  {df['timestamp'].min().date()} ~ {df['timestamp'].max().date()}"
          "  (OOS deliberately NOT touched)")

    ref = df.copy()
    ref["mtf_trend_1h"] = close.ewm(span=12, adjust=False).mean().pct_change().fillna(0.0)
    lab = RegimeEngine().compute(ref)
    y_ref = np.full(len(df), 2, dtype=int)
    y_ref[lab["regime_bull"].to_numpy() > 0] = 0
    y_ref[lab["regime_bear"].to_numpy() > 0] = 1
    runs = _run_lengths(y_ref)
    print(f"\nREFERENCE (RegimeEngine on BTC, = ETH's deployed-label scale): "
          f"bull={np.mean(y_ref==0):.3f} bear={np.mean(y_ref==1):.3f} chop={np.mean(y_ref==2):.3f} "
          f"| flip={np.mean(y_ref[1:]!=y_ref[:-1]):.4f} | run median={np.median(runs):.0f} "
          f"mean={np.mean(runs):.1f}")
    for h in FWD_HORIZONS:
        te = transition_edge(y_ref, close_np, h)
        print(f"    h={h:2d} edge {te['edge_bp']:+.2f}bp [95% {te['edge_ci_lo_bp']:+.2f},"
              f"{te['edge_ci_hi_bp']:+.2f}] | baseline |move| {te['baseline_abs_bp']:.1f}bp")

    rate1 = float((efficiency_ratio(close, 24) >= 0.20).mean())
    rate2 = float((efficiency_ratio(close, 48) >= 0.16).mean())
    print(f"\nBTC calibration targets: P(er_24>=0.20)={rate1:.4f}, P(er_48>=0.16)={rate2:.4f}")

    rows = []
    for s in SCALES:
        t1 = float(efficiency_ratio(close, s).quantile(1.0 - rate1))
        t2 = float(efficiency_ratio(close, 2 * s).quantile(1.0 - rate2))
        y_raw = scaled_label(close, s, t1, t2)
        for k in DEBOUNCES:
            y = y_raw if k == 1 else _debounce(y_raw, k)
            r = _run_lengths(y)
            row = {"scale_bars": s, "scale_min": s * 5, "debounce_k": k,
                   "T1": round(t1, 4), "T2": round(t2, 4),
                   "bull": round(float(np.mean(y == 0)), 3), "bear": round(float(np.mean(y == 1)), 3),
                   "chop": round(float(np.mean(y == 2)), 3),
                   "flip_rate": round(float(np.mean(y[1:] != y[:-1])), 4),
                   "run_median": float(np.median(r))}
            for h in FWD_HORIZONS:
                te = transition_edge(y, close_np, h)
                row[f"edge_h{h}_bp"] = round(te["edge_bp"], 2)
                row[f"ci_h{h}"] = f"[{te['edge_ci_lo_bp']:+.2f},{te['edge_ci_hi_bp']:+.2f}]"
                row[f"sig_h{h}"] = bool(te["edge_ci_lo_bp"] > 0)
            rows.append(row)

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "label_geometry.csv", index=False)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 100)
    print("\n=== BTC label geometry sweep (TRAIN only) ===")
    print(out[["scale_bars", "scale_min", "debounce_k", "chop", "flip_rate", "run_median",
               "edge_h6_bp", "ci_h6", "sig_h6", "edge_h12_bp", "ci_h12", "sig_h12"]].to_string(index=False))
    sig = out[out["sig_h6"] | out["sig_h12"]]
    print(f"\ncells with 95% CI strictly above 0 at h=6 or h=12: {len(sig)} / {len(out)}")
    print(f"Wrote {OUT_DIR / 'label_geometry.csv'}")


if __name__ == "__main__":
    main()
