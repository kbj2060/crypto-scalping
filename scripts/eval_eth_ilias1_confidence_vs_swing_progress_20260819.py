#!/usr/bin/env python3
"""가설 검증: quality_for_action(모델 확신도)이 "진짜 피벗(스윙 초입)"이 아니라 "스윙이 이미
많이 진행된 시점"과 상관돼서, threshold 필터링이 결과적으로 늦은 진입을 선호하게 만드는가?

방법: raw price로 재구성한 zigzag_action 라벨(ground truth, eval_eth_ilias1_denoised_price_
zigzag_relabel_rank_20260819.py와 동일 재사용)의 zigzag_segment_id로 각 LONG/SHORT bar가
자기 세그먼트(confirmed wave) 안에서 몇 % 지점에 있는지(progress: 0=세그먼트 시작 직후,
1=세그먼트 끝) 계산. trial12의 6개 시드 번들 전체(threshold 필터 없이 매 bar에 대해)로
quality_for_action/dir_confidence를 fresh 생성해서, ground-truth LONG/SHORT bar들에서
confidence와 progress의 상관관계를 직접 측정한다. 추가로 "confidence 상위 20%만 골랐을 때
progress 평균이 전체 평균보다 유의하게 높아지는가"(threshold 필터링을 흉내낸 직접 재현)도
확인 -- 이게 사실이면 quality 필터링 메커니즘 자체가 문제라는 가설이 직접 확인된다."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_wave3_action_labels_20260531 as zigzag_builder  # noqa: E402
import eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818 as ev  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

ZIGZAG_PARAMS = dict(
    min_reversal_pct=0.010, min_wave_bars=8, transition_buffer=2,
    atr_window=14, atr_multiplier=1.0, mae_penalty=1.25,
    softmax_temperature=1.75, min_risk_floor=0.0010,
)
SEEDS = [260620, 121026, 337153, 390529, 640787, 794920]
JUDGED_WINDOWS = ("oos_q1", "oos_q2")


def _bundle_cfg(seed: int) -> dict:
    bundle_dir = ROOT / f"tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_optuna_zig075_trial12_seed{seed}"
    return {"bundle": bundle_dir / "true_3head_tabm_bundle.pt", "q_tag": "q080", "threshold": 0.80}


def _load_full_frame() -> pd.DataFrame:
    frames = []
    for wname, wd in gate.WINDOW_DEFS.items():
        frame = sweep.load_frame(wd["start"], wd["end"], base_csv=wd["base_csv"], wide24_csv=wd["wide24_csv"])
        frame, _ = gate._drop_route_nan(frame)
        frame["window"] = wname
        frames.append(frame)
    full = pd.concat(frames, ignore_index=True).drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return full


def _segment_progress(labels: pd.DataFrame) -> np.ndarray:
    seg_id = labels["zigzag_segment_id"].to_numpy()
    n = len(seg_id)
    progress = np.full(n, np.nan, dtype=np.float64)
    idx = np.arange(n)
    df = pd.DataFrame({"seg": seg_id, "idx": idx})
    valid = df[df["seg"] >= 0]
    grp = valid.groupby("seg")["idx"].agg(["min", "max"])
    span = (grp["max"] - grp["min"]).clip(lower=1)
    prog_by_seg = ((valid["idx"].to_numpy() - grp.loc[valid["seg"], "min"].to_numpy()) / span.loc[valid["seg"]].to_numpy())
    progress[valid["idx"].to_numpy()] = prog_by_seg
    return progress


HALF_WIDTH = 48


def _rank_series(low: pd.Series, high: pd.Series, close: pd.Series) -> np.ndarray:
    win = 2 * HALF_WIDTH + 1
    local_low = low.rolling(window=win, center=True, min_periods=1).min()
    local_high = high.rolling(window=win, center=True, min_periods=1).max()
    span = (local_high - local_low).replace(0, np.nan)
    return ((close - local_low) / span).to_numpy()


def main() -> int:
    print("stage=load_frame", flush=True)
    full = _load_full_frame()
    print(f"loaded rows={len(full)}", flush=True)

    print("stage=rebuild_ground_truth_label", flush=True)
    labels = zigzag_builder.build_zigzag_action_labels(full, **ZIGZAG_PARAMS)
    action = labels["zigzag_action"].to_numpy()
    progress = _segment_progress(labels)
    rank = _rank_series(full["low"], full["high"], full["close"])
    judged_mask = full["window"].isin(JUDGED_WINDOWS).to_numpy()

    print("stage=amplification_check (progress vs rank, no model -- pure label/price geometry)", flush=True)
    amp_mask = judged_mask & (action != 0) & ~np.isnan(progress) & ~np.isnan(rank)
    prog_amp = progress[amp_mask]
    rank_amp = rank[amp_mask]
    r_prog_rank = np.corrcoef(prog_amp, rank_amp)[0, 1]
    print(f"corr(swing_progress, local_price_rank) across all ground-truth LONG+SHORT bars: {r_prog_rank:.4f}", flush=True)
    print("decile breakdown -- if rank increments GROW in later deciles, that's convexity/acceleration:", flush=True)
    prev_rank = None
    for d in range(10):
        lo, hi = d / 10, (d + 1) / 10
        m = (prog_amp >= lo) & (prog_amp < hi if d < 9 else prog_amp <= hi)
        mean_rank = rank_amp[m].mean()
        step = f"  step={mean_rank - prev_rank:+.4f}" if prev_rank is not None else ""
        print(f"  progress[{lo:.1f},{hi:.1f}]: n={m.sum():6d} mean_rank={mean_rank:.4f}{step}", flush=True)
        prev_rank = mean_rank

    all_conf = []
    all_progress = []
    all_rank = []
    all_action = []
    for seed in SEEDS:
        cfg = _bundle_cfg(seed)
        oof = False  # OOS windows always oof=False per gate.WINDOW_DEFS
        preds = ev.generate_predictions("zig075", cfg, full, oof=oof)
        conf_col = [c for c in preds.columns if c.endswith("_quality_for_action")][0]
        conf = preds[conf_col].to_numpy()

        m = judged_mask & (action != 0) & ~np.isnan(progress) & ~np.isnan(rank)
        conf_m = conf[m]
        prog_m = progress[m]
        rank_m = rank[m]
        act_m = action[m]

        r_all = np.corrcoef(conf_m, prog_m)[0, 1]
        top20_thr = np.percentile(conf_m, 80)
        top20_mask = conf_m >= top20_thr
        prog_top20 = prog_m[top20_mask].mean()
        prog_all = prog_m.mean()
        rank_top20 = rank_m[top20_mask].mean()
        rank_all = rank_m.mean()
        real_thr_mask = conf_m >= 0.80
        prog_real_thr = prog_m[real_thr_mask].mean() if real_thr_mask.sum() else float("nan")
        rank_real_thr = rank_m[real_thr_mask].mean() if real_thr_mask.sum() else float("nan")
        print(f"seed={seed} n={m.sum()} corr(confidence,progress)={r_all:.4f}  "
              f"progress: all={prog_all:.4f} top20%={prog_top20:.4f}(d={prog_top20-prog_all:+.4f}) conf>=0.80={prog_real_thr:.4f}(d={prog_real_thr-prog_all:+.4f})  "
              f"|  rank: all={rank_all:.4f} top20%={rank_top20:.4f}(d={rank_top20-rank_all:+.4f}) conf>=0.80={rank_real_thr:.4f}(d={rank_real_thr-rank_all:+.4f})", flush=True)

        all_conf.append(conf_m)
        all_progress.append(prog_m)
        all_rank.append(rank_m)
        all_action.append(act_m)

    conf_pool = np.concatenate(all_conf)
    prog_pool = np.concatenate(all_progress)
    rank_pool = np.concatenate(all_rank)
    act_pool = np.concatenate(all_action)

    print()
    print("=== POOLED (6 seeds combined): progress-shift vs rank-shift from confidence filtering ===", flush=True)
    for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
        lo_thr, hi_thr = np.percentile(conf_pool, lo), np.percentile(conf_pool, hi)
        m = (conf_pool >= lo_thr) & (conf_pool <= hi_thr)
        print(f"  confidence percentile [{lo:3d},{hi:3d}]: n={m.sum():6d} mean_swing_progress={prog_pool[m].mean():.4f} mean_rank={rank_pool[m].mean():.4f}", flush=True)

    real_thr_mask_pool = conf_pool >= 0.80
    print(f"  conf>=0.80 (real threshold): n={real_thr_mask_pool.sum():6d} mean_swing_progress={prog_pool[real_thr_mask_pool].mean():.4f} mean_rank={rank_pool[real_thr_mask_pool].mean():.4f}", flush=True)
    print(f"  ALL (no filter):             n={len(conf_pool):6d} mean_swing_progress={prog_pool.mean():.4f} mean_rank={rank_pool.mean():.4f}", flush=True)

    for act, name in [(1, "LONG"), (2, "SHORT")]:
        m = act_pool == act
        thr = conf_pool[m] >= 0.80
        print(f"[{name}] n={m.sum()} rank: all={rank_pool[m].mean():.4f} conf>=0.80={rank_pool[m][thr].mean():.4f} "
              f"(d={rank_pool[m][thr].mean() - rank_pool[m].mean():+.4f})  vs earlier full-pipeline result for reference", flush=True)

    print()
    print("=== STUDY DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
