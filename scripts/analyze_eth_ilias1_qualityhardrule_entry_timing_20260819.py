#!/usr/bin/env python3
"""hard_rule quality_mode로 재학습한 zig075(seed260620, trial12 레시피)의 진입타이밍 rank를
analyze_eth_ilias1_zig075_trial12_entry_timing_20260819.py와 동일 방법론(±48bar 로컬 rank)
으로 same_as_direction(trial12) 대비 확인. q080(VAL-best) 사용."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818 as ev  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

HALF_WIDTH = 48
WINDOWS = ["oos_q1", "oos_q2"]
BUNDLE_PATH = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_trial12_qualityhardrule_20260819/true_3head_tabm_bundle.pt"
CFG = {"bundle": BUNDLE_PATH, "q_tag": "q080", "threshold": 0.80}


def main() -> int:
    frames = {}
    for wname in WINDOWS:
        wd = gate.WINDOW_DEFS[wname]
        frame = sweep.load_frame(wd["start"], wd["end"], base_csv=wd["base_csv"], wide24_csv=wd["wide24_csv"])
        frame, _ = gate._drop_route_nan(frame)
        frame = frame.reset_index(drop=True)
        low = pd.to_numeric(frame["low"], errors="raise")
        high = pd.to_numeric(frame["high"], errors="raise")
        close = pd.to_numeric(frame["close"], errors="raise")
        win = 2 * HALF_WIDTH + 1
        local_low = low.rolling(window=win, center=True, min_periods=1).min()
        local_high = high.rolling(window=win, center=True, min_periods=1).max()
        span = (local_high - local_low).replace(0, np.nan)
        frame["_rank"] = ((close - local_low) / span).to_numpy()
        frames[wname] = frame
        print(f"window={wname} rows={len(frame)}", flush=True)

    all_long_ranks, all_short_ranks = [], []
    for wname in WINDOWS:
        frame = frames[wname]
        oof = bool(gate.WINDOW_DEFS[wname]["oof"])
        preds = ev.generate_predictions("zig075", CFG, frame, oof=oof)
        action_col = [c for c in preds.columns if c.endswith("_final_action")][0]
        action = preds[action_col].to_numpy()
        rank = frame["_rank"].to_numpy()
        long_mask = action == 1
        short_mask = action == 2
        long_ranks = rank[long_mask]
        short_ranks = rank[short_mask]
        long_ranks = long_ranks[~np.isnan(long_ranks)]
        short_ranks = short_ranks[~np.isnan(short_ranks)]
        all_long_ranks.extend(long_ranks.tolist())
        all_short_ranks.extend(short_ranks.tolist())
        print(f"window={wname} n_long={len(long_ranks)} long_rank_mean={long_ranks.mean() if len(long_ranks) else float('nan'):.4f} "
              f"n_short={len(short_ranks)} short_rank_mean={short_ranks.mean() if len(short_ranks) else float('nan'):.4f}", flush=True)

    long_arr = np.array(all_long_ranks)
    short_arr = np.array(all_short_ranks)
    print()
    print("=== SUMMARY: hard_rule quality_mode (seed260620, q080) ===", flush=True)
    print(f"LONG:  n={len(long_arr)} mean={long_arr.mean():.4f} median={np.median(long_arr):.4f}  (trial12 same_as_direction was 0.636)", flush=True)
    print(f"SHORT: n={len(short_arr)} mean={short_arr.mean():.4f} median={np.median(short_arr):.4f}  (trial12 same_as_direction was 0.463)", flush=True)
    print("=== STUDY DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
