#!/usr/bin/env python3
"""h48qual이 zig075와 같은 진입타이밍 패턴(추세추종/사후확정 스윙인식, 바닥/꼭대기 아님)을
보이는지 확인 -- analyze_eth_ilias1_zig075_trial12_entry_timing_20260819.py와 완전히 동일한
방법론(±48bar 로컬 [low,high] 윈도우에서 종가 백분위 rank, 0=로컬바닥/1=로컬꼭대기)을
h48qual 번들에 적용. h48qual의 direction 라벨도 zig075와 동일한 zigzag_action 소스를 쓰므로
(quality 게이트만 다름 -- quality_label_action vs same_as_direction), 같은 패턴이 나올지가
관심사. h48qual은 오늘 저녁 Optuna 대상이 아니었으므로 원본(epochs=2)+5개 신규시드 variant
(전부 epochs=2) 6개 번들을 그대로 사용 -- trial12같은 epochs=40 재학습본은 존재하지 않음."""
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

WINDOWS = ["oos_q1", "oos_q2"]
HALF_WIDTH = 48

_ORIGINAL_DIR = "omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_pinned102_20260818"
SEED_BUNDLE_DIRS = {260620: _ORIGINAL_DIR}
for _seed in (121026, 337153, 390529, 640787, 794920):
    SEED_BUNDLE_DIRS[_seed] = f"{_ORIGINAL_DIR}_seedvariant_{_seed}"


def _bundle_cfg(seed: int) -> dict:
    bundle_dir = ROOT / f"tmp/causal_regen_20260516/{SEED_BUNDLE_DIRS[seed]}"
    return {"bundle": bundle_dir / "true_3head_tabm_bundle.pt", "q_tag": "q040", "threshold": 0.40}


def main() -> int:
    frames: dict[str, pd.DataFrame] = {}
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
        print(f"window={wname} rows={len(frame)} baseline_rank_mean={frame['_rank'].mean():.4f} "
              f"baseline_rank_median={frame['_rank'].median():.4f}", flush=True)

    all_long_ranks: list[float] = []
    all_short_ranks: list[float] = []

    for seed in SEED_BUNDLE_DIRS:
        cfg = _bundle_cfg(seed)
        for wname in WINDOWS:
            frame = frames[wname]
            oof = bool(gate.WINDOW_DEFS[wname]["oof"])
            preds = ev.generate_predictions("h48qual", cfg, frame, oof=oof)
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
            lm = float(np.mean(long_ranks)) if len(long_ranks) else float("nan")
            sm = float(np.mean(short_ranks)) if len(short_ranks) else float("nan")
            print(f"seed={seed} window={wname} n_long={len(long_ranks)} long_rank_mean={lm:.4f} "
                  f"n_short={len(short_ranks)} short_rank_mean={sm:.4f}", flush=True)

    print()
    print("=== SUMMARY (all 6 seeds x oos_q1+oos_q2 combined) ===", flush=True)
    long_arr = np.array(all_long_ranks)
    short_arr = np.array(all_short_ranks)
    baseline_all = pd.concat([frames[w]["_rank"] for w in WINDOWS]).dropna().to_numpy()
    print(f"baseline(all bars):  n={len(baseline_all)}  mean={baseline_all.mean():.4f}  median={np.median(baseline_all):.4f}")
    print(f"LONG entries:        n={len(long_arr)}  mean={long_arr.mean():.4f}  median={np.median(long_arr):.4f}  (0.0=local bottom, want LOW)")
    print(f"SHORT entries:       n={len(short_arr)}  mean={short_arr.mean():.4f}  median={np.median(short_arr):.4f}  (1.0=local top, want HIGH)")
    frac_long_bottom_half = float((long_arr < 0.5).mean()) if len(long_arr) else float("nan")
    frac_short_top_half = float((short_arr >= 0.5).mean()) if len(short_arr) else float("nan")
    print(f"LONG entries in bottom half (<0.5): {frac_long_bottom_half:.1%}  (50% = coin flip)")
    print(f"SHORT entries in top half (>=0.5):  {frac_short_top_half:.1%}  (50% = coin flip)")

    print()
    print("=== STUDY DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
