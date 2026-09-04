#!/usr/bin/env python3
"""Design research for a 3-class V_REBOUND split (2026-08-30, user hand-drawn sketch): sustained
V (지속) / spike-then-plateau (횡보) / failed-or-fully-reverted (실패), replacing the current
binary label's collapse of "sustained" and "plateau" into one V_REBOUND=1 bucket.

Framework: keep the existing fast-window magnitude gate (>=1.5x pre-sweep ATR within the first
V_REBOUND_FAST_BARS) as the "did it even attempt a real reversal" filter -- events that never
reach it are automatically 실패 (this matches how label=0 already partly worked). For events
that DID reach the target, replace the current binary "confirmed" (all 6 closes beyond the
swept level) with a CONTINUOUS giveback_ratio = (peak_in_6bar_window - end_close) / (peak -
sweep_extreme), then cut it at two thresholds into 지속/횡보/실패 -- this also fixes the visual
problem found in research_eth_sweep_v_rebound_giveback_pattern_20260830.py's example chart,
where some very-high-giveback events that technically passed the old level-based "confirmed"
check actually looked like near-failures, not genuine plateaus.

This script computes giveback_ratio for the FULL 14,259-event population (not just the current
label=1 subset) so the "attempted but failed to hold at all" cases (currently buried inside
label=0) become visible too, and sweeps a small grid of (T_sustain, T_fail) threshold pairs to
report resulting class sizes -- research/diagnostic only, no label file written, no thresholds
picked as final here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
LABEL_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_labels.csv"
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_3class_design_20260830"
LOOKAHEAD_BARS = 6
FAST_BARS = 3
ATR_MULT = 1.5


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_3class_20260830", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    impl = load_impl()
    frame = impl.load_5m(SOURCE)
    events = pd.read_csv(LABEL_CSV)  # already has candidate_index/side/label/atr(pre-sweep, v4)/sweep_level

    attempted, giveback_ratio, peak_col, end_col = [], [], [], []
    for _, event in events.iterrows():
        idx = int(event["candidate_index"])
        row = frame.iloc[idx]
        future = frame.iloc[idx + 1: idx + LOOKAHEAD_BARS + 1]
        fast_future = future.iloc[:FAST_BARS]
        atr = event["atr"]  # already pre-sweep (v4)
        if event["side"] == "downside":
            sweep_extreme = row["low"]
            fast_move = fast_future["high"].max() - sweep_extreme
            peak = future["high"].max()
            end = future["close"].iloc[-1]
            giveback = peak - end
        else:
            sweep_extreme = row["high"]
            fast_move = sweep_extreme - fast_future["low"].min()
            peak = future["low"].min()
            end = future["close"].iloc[-1]
            giveback = end - peak
        total_move = abs(peak - sweep_extreme)
        attempted.append(bool(fast_move >= ATR_MULT * atr))
        giveback_ratio.append(float(giveback / total_move) if total_move > 1e-12 else np.nan)
        peak_col.append(peak)
        end_col.append(end)

    events["attempted"] = attempted
    events["giveback_ratio"] = giveback_ratio
    events["window_peak"] = peak_col
    events["window_end"] = end_col

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUT_DIR / "events_with_3class_inputs.csv", index=False)

    n = len(events)
    n_attempted = int(events["attempted"].sum())
    print(f"total events: {n}")
    print(f"never reached fast-window 1.5x-ATR target ('no attempt' -> auto 실패): "
          f"{n - n_attempted} ({(n-n_attempted)/n:.1%})")
    print(f"reached target within 15min ('attempted'): {n_attempted} ({n_attempted/n:.1%})")
    print(f"  -- of these, old v4 label==1 (passed the level-hold check too): "
          f"{int((events['attempted'] & (events['label']==1)).sum())}")
    print(f"  -- of these, old v4 label==0 (reached target FAST but still failed to hold the "
          f"level for all 30min): {int((events['attempted'] & (events['label']==0)).sum())}")

    att = events[events["attempted"]]
    print(f"\ngiveback_ratio distribution among the {len(att)} 'attempted' events:")
    print(att["giveback_ratio"].describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).to_string())

    print("\n=== threshold grid: resulting 3-class sizes (as % of the FULL 14,259 population) ===")
    print(f"{'T_sustain':>10} {'T_fail':>8} | {'지속':>7} {'횡보':>7} {'실패':>7}")
    for t_sustain in (0.10, 0.15, 0.20, 0.25):
        for t_fail in (0.50, 0.60, 0.70, 0.80):
            if t_fail <= t_sustain:
                continue
            sustained = int((att["giveback_ratio"] <= t_sustain).sum())
            plateau = int(((att["giveback_ratio"] > t_sustain) & (att["giveback_ratio"] <= t_fail)).sum())
            near_fail_within_attempted = int((att["giveback_ratio"] > t_fail).sum())
            failed = (n - n_attempted) + near_fail_within_attempted
            print(f"{t_sustain:>10.2f} {t_fail:>8.2f} | {sustained/n:>6.1%} {plateau/n:>6.1%} {failed/n:>6.1%}"
                  f"   (n={sustained}/{plateau}/{failed})")

    print("\nby side (downside vs upside), attempted-population giveback_ratio median (symmetry check):")
    print(att.groupby("side")["giveback_ratio"].median())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
