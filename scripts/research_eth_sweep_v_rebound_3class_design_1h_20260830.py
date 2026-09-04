#!/usr/bin/env python3
"""Same as research_eth_sweep_v_rebound_3class_design_20260830.py, but the outcome-observation
window (used only for peak/end/giveback_ratio -- NOT the trigger, NOT the 15min fast-target
check, both unchanged) is widened from 6 bars/30min to 12 bars/60min, per user request ("30분에서
1시간으로 늘리면 어떻게 되는지"). Same event population (same candidate_index list from the v4
label file) -- only how far forward we watch the outcome changes.
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
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_3class_design_1h_20260830"
LOOKAHEAD_BARS = 12   # 1h, was 6 (30min)
FAST_BARS = 3         # unchanged: still the first 15min for the "did it even attempt" gate
ATR_MULT = 1.5


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_3class_1h_20260830", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    impl = load_impl()
    frame = impl.load_5m(SOURCE)
    events = pd.read_csv(LABEL_CSV)

    n_before = len(events)
    events = events[events["candidate_index"] + LOOKAHEAD_BARS < len(frame)].reset_index(drop=True)
    print(f"events with full 60min of future data available: {len(events)}/{n_before} "
          f"(dropped {n_before - len(events)} too close to the end of history)")

    attempted, giveback_ratio = [], []
    for _, event in events.iterrows():
        idx = int(event["candidate_index"])
        row = frame.iloc[idx]
        future = frame.iloc[idx + 1: idx + LOOKAHEAD_BARS + 1]
        fast_future = future.iloc[:FAST_BARS]
        atr = event["atr"]
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

    events["attempted"] = attempted
    events["giveback_ratio"] = giveback_ratio

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUT_DIR / "events_with_3class_inputs_1h.csv", index=False)

    n = len(events)
    n_attempted = int(events["attempted"].sum())
    print(f"total events: {n}")
    print(f"never reached fast-window 1.5x-ATR target ('no attempt' -> auto 실패): "
          f"{n - n_attempted} ({(n-n_attempted)/n:.1%})")
    print(f"reached target within 15min ('attempted'): {n_attempted} ({n_attempted/n:.1%})")

    att = events[events["attempted"]]
    print(f"\ngiveback_ratio distribution among the {len(att)} 'attempted' events (60min horizon):")
    print(att["giveback_ratio"].describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).to_string())

    print("\n=== same threshold grid as the 30min version, now on the 60min giveback ===")
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

    # side-by-side comparison vs the 30min version at the SAME (0.15, 0.70) candidate
    t_sustain, t_fail = 0.15, 0.70
    sustained = int((att["giveback_ratio"] <= t_sustain).sum())
    plateau = int(((att["giveback_ratio"] > t_sustain) & (att["giveback_ratio"] <= t_fail)).sum())
    failed = (n - n_attempted) + int((att["giveback_ratio"] > t_fail).sum())
    print(f"\nat the SAME candidate (0.15, 0.70) used for the 30min version: "
          f"지속 {sustained/n:.1%} / 횡보 {plateau/n:.1%} / 실패 {failed/n:.1%}")
    print("(30min version at the same thresholds was: 지속 13.9% / 횡보 32.4% / 실패 53.7%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
