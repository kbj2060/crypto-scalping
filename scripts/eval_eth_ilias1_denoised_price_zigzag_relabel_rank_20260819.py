#!/usr/bin/env python3
"""Phase 1 cheap gate (모델 재학습 없음): train_eth_ilias1_price_denoiser_conv1d_ssl_20260819.py가
만든 denoised close로 zigzag_action 라벨을 재계산(scripts/build_wave3_action_labels_20260531.py의
build_zigzag_action_labels를 재사용, 재구현 아님 -- close 컬럼만 denoised로 교체, open/high/low는
원본 유지)한 뒤, 그 새 라벨 자체(모델 예측 아님, ground-truth 라벨)의 진입타이밍 rank를
analyze_eth_ilias1_zig075_trial12_entry_timing_20260819.py와 동일한 방법론(±48bar 로컬
[low,high] 백분위)으로 원본 라벨과 직접 비교한다.

목적: 모델을 재학습하기 전에 "라벨 자체가 denoised price로 다시 만들면 진입타이밍이 개선되는가"
부터 저렴하게 확인 -- 라벨(oracle)조차 안 좋아지면 그 라벨로 아무리 잘 학습시켜도 좋아질 수 없다."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_wave3_action_labels_20260531 as zigzag_builder  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

HALF_WIDTH = 48
ZIGZAG_PARAMS = dict(
    min_reversal_pct=0.010, min_wave_bars=8, transition_buffer=2,
    atr_window=14, atr_multiplier=1.0, mae_penalty=1.25,
    softmax_temperature=1.75, min_risk_floor=0.0010,
)
DENOISED_CSV = ROOT / "tmp/causal_regen_20260516/eth_ilias1_price_denoiser_conv1d_20260819/denoised_price_series.csv"
JUDGED_WINDOWS = ("oos_q1", "oos_q2")


def _load_full_ohlc() -> pd.DataFrame:
    frames = []
    for wname, wd in gate.WINDOW_DEFS.items():
        frame = sweep.load_frame(wd["start"], wd["end"], base_csv=wd["base_csv"], wide24_csv=wd["wide24_csv"])
        frame, _ = gate._drop_route_nan(frame)
        frame = frame[["timestamp", "open", "high", "low", "close"]].copy()
        frame["window"] = wname
        frames.append(frame)
    full = pd.concat(frames, ignore_index=True).drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return full


def _rank_series(low: pd.Series, high: pd.Series, close: pd.Series, *, half_width: int) -> np.ndarray:
    win = 2 * half_width + 1
    local_low = low.rolling(window=win, center=True, min_periods=1).min()
    local_high = high.rolling(window=win, center=True, min_periods=1).max()
    span = (local_high - local_low).replace(0, np.nan)
    return ((close - local_low) / span).to_numpy()


def _summarize(label_col: np.ndarray, rank: np.ndarray, window_mask: np.ndarray, tag: str) -> None:
    m = window_mask
    action = label_col[m]
    r = rank[m]
    long_r = r[(action == 1) & ~np.isnan(r)]
    short_r = r[(action == 2) & ~np.isnan(r)]
    baseline_r = r[~np.isnan(r)]
    print(f"[{tag}] n_long={len(long_r)} long_rank_mean={long_r.mean() if len(long_r) else float('nan'):.4f}  "
          f"n_short={len(short_r)} short_rank_mean={short_r.mean() if len(short_r) else float('nan'):.4f}  "
          f"baseline_rank_mean={baseline_r.mean():.4f}", flush=True)


def main() -> int:
    print("stage=load_ohlc", flush=True)
    full = _load_full_ohlc()
    print(f"loaded rows={len(full)}", flush=True)

    denoised = pd.read_csv(DENOISED_CSV, parse_dates=["timestamp"])
    full = full.merge(denoised[["timestamp", "close_denoised"]], on="timestamp", how="left")
    n_missing = int(full["close_denoised"].isna().sum())
    print(f"merged denoised close: missing={n_missing}/{len(full)} (expected {HALF_WIDTH+64} leading NaN from denoiser's own warmup window)", flush=True)
    full = full.dropna(subset=["close_denoised"]).reset_index(drop=True)

    print("stage=rebuild_raw_label (sanity check vs stored zigzag_action_labels_20260531)", flush=True)
    raw_labels = zigzag_builder.build_zigzag_action_labels(full, **ZIGZAG_PARAMS)
    print(f"raw rebuild action counts: {raw_labels['zigzag_action'].value_counts().sort_index().to_dict()}", flush=True)

    print("stage=build_denoised_label", flush=True)
    denoised_frame = full.copy()
    denoised_frame["close"] = denoised_frame["close_denoised"]
    denoised_labels = zigzag_builder.build_zigzag_action_labels(denoised_frame, **ZIGZAG_PARAMS)
    print(f"denoised action counts: {denoised_labels['zigzag_action'].value_counts().sort_index().to_dict()}", flush=True)

    agree = float((raw_labels["zigzag_action"].to_numpy() == denoised_labels["zigzag_action"].to_numpy()).mean())
    print(f"raw vs denoised label agreement (all bars, all windows): {agree:.4f}", flush=True)

    rank_raw = _rank_series(full["low"], full["high"], full["close"], half_width=HALF_WIDTH)
    rank_denoised = rank_raw  # local price geometry is the SAME (raw OHLC) -- only which bars get called LONG/SHORT changes

    judged_mask = full["window"].isin(JUDGED_WINDOWS).to_numpy()
    print()
    print("=== ENTRY-TIMING RANK: raw-price zigzag label vs denoised-price zigzag label (ground-truth labels, no model) ===", flush=True)
    _summarize(raw_labels["zigzag_action"].to_numpy(), rank_raw, judged_mask, "raw_label (rebuilt)")
    _summarize(denoised_labels["zigzag_action"].to_numpy(), rank_denoised, judged_mask, "denoised_label")

    print()
    print("=== STUDY DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
