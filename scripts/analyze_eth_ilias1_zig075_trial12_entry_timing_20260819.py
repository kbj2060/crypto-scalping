#!/usr/bin/env python3
"""zig075 trial12(Optuna 우승 레시피)의 실제 진입이 바닥에서 LONG / 꼭대기에서 SHORT를 잡는지
직접 측정 -- 사용자 질문("바닥에서 롱을 잡는지 위에서 숏을 잡는지"). OOS-Q1/OOS-Q2(판정창) 프레임을
eval 엔진과 완전히 동일한 방식(sweep.load_frame + ev.generate_predictions, 진짜 fresh causal
추론 -- 저장된 예측 CSV 재사용 아님)으로 재생성해서 raw OHLC와 조인한다.

측정: 각 bar의 종가가 자기 자신 중심 [-48,+48]bar(8시간) 윈도우의 [low.min, high.max] 안에서
차지하는 백분위(rank = (close-local_low)/(local_high-local_low), 0=윈도우 최저가, 1=윈도우
최고가)를 프레임 전체 bar에 대해 계산(rolling, center=True) -- 이게 "이 bar가 로컬 바닥/
꼭대기였는가"의 사후(hindsight) 측정치다. 그 다음 LONG 진입 bar들의 rank 분포(0에 가까울수록
좋음)와 SHORT 진입 bar들의 rank 분포(1에 가까울수록 좋음)를, 같은 프레임의 "모든 bar" rank
분포(구조적 베이스라인 -- 가격계열 자체의 굴곡 때문에 uniform(0,1)이 아닐 수 있어서 0.5를
그냥 쓰지 않고 실측 baseline과 비교)와 대조한다."""
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

SEEDS = [260620, 121026, 337153, 390529, 640787, 794920]
WINDOWS = ["oos_q1", "oos_q2"]
HALF_WIDTH = 48  # bars each side = 4h, 8h total window -- matches this repo's h48 convention


def _bundle_cfg(seed: int) -> dict:
    bundle_dir = ROOT / f"tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_optuna_zig075_trial12_seed{seed}"
    return {"bundle": bundle_dir / "true_3head_tabm_bundle.pt", "q_tag": "q080", "threshold": 0.80}


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
    per_seed_rows = []

    for seed in SEEDS:
        cfg = _bundle_cfg(seed)
        for wname in WINDOWS:
            frame = frames[wname]
            oof = bool(gate.WINDOW_DEFS[wname]["oof"])
            preds = ev.generate_predictions("zig075", cfg, frame, oof=oof)
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
            per_seed_rows.append((seed, wname, len(long_ranks), lm, len(short_ranks), sm))
            print(f"seed={seed} window={wname} n_long={len(long_ranks)} long_rank_mean={lm:.4f} "
                  f"n_short={len(short_ranks)} short_rank_mean={sm:.4f}", flush=True)

    baseline_all = pd.concat([frames[w]["_rank"] for w in WINDOWS]).dropna().to_numpy()

    print()
    print("=== SUMMARY (all 6 seeds x oos_q1+oos_q2 combined) ===", flush=True)
    long_arr = np.array(all_long_ranks)
    short_arr = np.array(all_short_ranks)
    print(f"baseline(all bars):  n={len(baseline_all)}  mean={baseline_all.mean():.4f}  median={np.median(baseline_all):.4f}  "
          f"p10={np.percentile(baseline_all,10):.4f}  p50={np.percentile(baseline_all,50):.4f}  p90={np.percentile(baseline_all,90):.4f}")
    print(f"LONG entries:        n={len(long_arr)}  mean={long_arr.mean():.4f}  median={np.median(long_arr):.4f}  "
          f"p10={np.percentile(long_arr,10):.4f}  p50={np.percentile(long_arr,50):.4f}  p90={np.percentile(long_arr,90):.4f}  "
          f"(0.0=local bottom, want LOW)")
    print(f"SHORT entries:       n={len(short_arr)}  mean={short_arr.mean():.4f}  median={np.median(short_arr):.4f}  "
          f"p10={np.percentile(short_arr,10):.4f}  p50={np.percentile(short_arr,50):.4f}  p90={np.percentile(short_arr,90):.4f}  "
          f"(1.0=local top, want HIGH)")

    frac_long_bottom_half = float((long_arr < 0.5).mean()) if len(long_arr) else float("nan")
    frac_short_top_half = float((short_arr >= 0.5).mean()) if len(short_arr) else float("nan")
    print(f"LONG entries in bottom half (<0.5): {frac_long_bottom_half:.1%}  (50% = coin flip)")
    print(f"SHORT entries in top half (>=0.5):  {frac_short_top_half:.1%}  (50% = coin flip)")

    print()
    print("=== STUDY DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
