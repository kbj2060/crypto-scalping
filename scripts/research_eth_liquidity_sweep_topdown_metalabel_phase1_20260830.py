#!/usr/bin/env python3
"""Phase1 diagnostics for liquidity_sweep "top/down" metalabel -- Homer project, redo of signal
#2 using the STANDARD touch-based-MFE template (docs/homer/README.md "재사용 방법론 템플릿"),
replacing the specialized V_REBOUND design (giveback-ratio + confirmed-over-full-window +
excluded-middle-zone) that was never merged into the shared evidence-signal-chip-replacement
pattern taker_delta_z_climax/short_term_return_z use.

liquidity_sweep's raw trigger (bottom_liquidity_sweep/top_liquidity_sweep,
live_evidence_signal_dashboard_20260823.py::compute_signals() -- 48-bar causal swing-level
wick-poke + close-back-inside) is NOT changed by this work. This script only measures what
HORIZON/K(ATR multiple)/CLUSTER_GAP the touch-based MFE hit label (matching taker_delta_z_climax's
v4/short_term_return_z's v1 label shape exactly: entry=fire bar close, hit = intrabar MFE over a
fixed forward window >= K*atr_pct, no persistence/confirmed condition) should use for THIS signal
-- measured fresh per the project's own repeated warning ("방향이 신호마다 반대일 수 있어 절대
복붙 금지"), never copied from another signal's numbers.

Per methodology checklist (docs/homer/README.md "2) 라벨 설계 전 진단 체크리스트"):
  1. raw hit-rate horizon sensitivity (grid of HORIZON x K)
  2. size distribution (pred_dir_ret quantiles at a reference horizon)
  3. fire-bar <-> true local extreme lag/lead (+/-2h window argmax/argmin)
  4. consecutive-fire clustering (gap distribution -> CLUSTER_GAP_MERGE decision)
Chart-based visual verification (item 6) is a separate follow-up script once a first-pass label
is picked from these numbers.
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

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame  # noqa: E402

KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
START = pd.Timestamp("2024-01-01")
LAG_WINDOW = 24  # +/-2h, same window taker/short_term_return_z used for their own lag diagnostic
HORIZONS = [6, 12, 24, 36, 48, 72, 96]  # 30min..8h
K_GRID = [1.0, 1.5, 2.0, 2.5]


def log(msg: str) -> None:
    print(f"[liq_sweep_topdown_phase1] {msg}", flush=True)


def load_klines() -> pd.DataFrame:
    df = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=5)).all(), "gap/dup in klines"
    return df


def main() -> int:
    log("loading klines...")
    klines = load_klines()
    log(f"{len(klines)} bars loaded ({klines['timestamp'].min()} .. {klines['timestamp'].max()})")

    log("computing signals (compute_signals, live-dashboard formula, verbatim reuse)...")
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)

    log("building Tier0 indicator frame (verbatim reuse of taker_delta_z_climax's build_indicator_frame)...")
    ind = build_indicator_frame(klines)
    assert len(sig) == len(ind), "row count mismatch"
    assert (sig["timestamp"].to_numpy() == ind["timestamp"].to_numpy()).all(), "timestamp misalignment"

    high = sig["high"].to_numpy()
    low = sig["low"].to_numpy()
    close = sig["close"].to_numpy()
    atr_pct = ind["atr_pct"].to_numpy()
    ts = sig["timestamp"].to_numpy()
    n = len(sig)

    for side, col in [("bottom", "bottom_liquidity_sweep"), ("top", "top_liquidity_sweep")]:
        idx_all = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx_all = idx_all[ts[idx_all] >= np.datetime64(START)]
        log(f"\n=== {side} ({col}): {len(idx_all)} raw fires (>= {START.date()}) ===")

        # 1) clustering: gap (in bars) to the previous same-side fire
        gaps = np.diff(idx_all)
        log(f"  gap-to-prior-fire (bars): median={np.median(gaps):.1f} p25={np.percentile(gaps, 25):.1f} "
            f"p10={np.percentile(gaps, 10):.1f} frac<=3={np.mean(gaps <= 3):.3f} "
            f"frac<=6={np.mean(gaps <= 6):.3f} frac<=12={np.mean(gaps <= 12):.3f}")

        # 2) fire-bar vs true local extreme lag/lead, +/-LAG_WINDOW bars
        idx_lag = idx_all[(idx_all >= LAG_WINDOW) & (idx_all < n - LAG_WINDOW)]
        offsets = np.empty(len(idx_lag), dtype=int)
        for j, i in enumerate(idx_lag):
            if side == "bottom":
                window = low[i - LAG_WINDOW:i + LAG_WINDOW + 1]
                offsets[j] = window.argmin() - LAG_WINDOW
            else:
                window = high[i - LAG_WINDOW:i + LAG_WINDOW + 1]
                offsets[j] = window.argmax() - LAG_WINDOW
        at_fire = np.mean(offsets == 0)
        after_fire = offsets[offsets > 0]
        before_fire = offsets[offsets < 0]
        log(f"  true local extreme vs fire bar (+/-{LAG_WINDOW}bars={LAG_WINDOW * 5}min window, n={len(idx_lag)}): "
            f"AT fire={at_fire:.3f}, AFTER(lag)={len(after_fire) / len(idx_lag):.3f} "
            f"(median {np.median(after_fire) if len(after_fire) else float('nan'):.1f} bars / "
            f"{np.median(after_fire) * 5 if len(after_fire) else float('nan'):.0f}min, "
            f"p90 {np.percentile(after_fire, 90) if len(after_fire) else float('nan'):.1f} bars), "
            f"BEFORE(lead)={len(before_fire) / len(idx_lag):.3f} "
            f"(median {np.median(-before_fire) if len(before_fire) else float('nan'):.1f} bars)")

        # 3) hit-rate horizon x K sensitivity grid (touch-based MFE, no persistence)
        for horizon in HORIZONS:
            idx_h = idx_all[idx_all < n - horizon]
            entry_h = close[idx_h]
            if side == "bottom":
                fut_ext = np.array([high[i + 1:i + horizon + 1].max() for i in idx_h])
                pred_dir_ret = (fut_ext - entry_h) / entry_h
            else:
                fut_ext = np.array([low[i + 1:i + horizon + 1].min() for i in idx_h])
                pred_dir_ret = (entry_h - fut_ext) / entry_h
            atr_h = atr_pct[idx_h]
            valid = np.isfinite(atr_h) & (atr_h > 0)
            pred_dir_ret_v, atr_h_v = pred_dir_ret[valid], atr_h[valid]
            cells = [f"K={k}:{np.mean(pred_dir_ret_v >= k * atr_h_v):.3f}" for k in K_GRID]
            if horizon == 24:
                q = np.percentile(pred_dir_ret_v / atr_h_v, [10, 25, 50, 75, 90])
                log(f"  [ref] HORIZON=24 pred_dir_ret/atr_pct quantiles (10/25/50/75/90): "
                    f"{q[0]:.2f} {q[1]:.2f} {q[2]:.2f} {q[3]:.2f} {q[4]:.2f}")
            log(f"  HORIZON={horizon:>3d} ({horizon * 5:>4d}min): n={len(idx_h)} " + " ".join(cells))

    log("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
