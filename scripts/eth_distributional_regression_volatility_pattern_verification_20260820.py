#!/usr/bin/env python3
"""③분포적회귀 cheap-gate에서 나온 h48/h96 변동성클러스터링 패턴(3-split 부호일관+단조증가)
집중검증. 사용자 지시: "추세통제/하위기간 안정성 체크"로 진짜신호(변동성스파이크후 되돌림)와
가짜(VAL/OOS 단방향약세장에서 leverage effect로 vol이 하락과 동시발생)를 가른다.

**체크A(하위기간 안정성)**: TRAIN/VAL/OOS를 월별 블록으로 쪼개 블록별 IC를 개별 계산 --
"3개 대형split이 우연히 같은부호"가 아니라 "훨씬 잘게 쪼개도 계속 같은부호가 나오는가"를
직접 확인. 이 세션 기존 방법(3-split 일관성)보다 훨씬 엄격한 기준.

**체크B(후행추세 통제, partial correlation)**: 변동성피쳐가 예측하는 게 진짜 "미래" 정보인지,
아니면 "최근 하락추세가 계속되는 중"이라는 이미 아는 정보의 재탕인지 가르기 위해, 60bar(5시간,
158개 피쳐 어느 창과도 안 겹치는 window -- 12/24/48/96/288/2016 전부 회피, dual_momentum
윈도우 우연일치 재발 방지)짜리 후행(backward) 수익률을 새로 계산해 통제변수로 넣고, 변동성
피쳐와 forward return의 관계가 이 통제변수를 뺀 뒤에도 남는지 확인(랭크 잔차 상관)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "tmp/eth_distributional_regression_return_labels_20260819"
TREND_CONTROL_WINDOW = 60  # 5시간 -- 158피쳐/dual_momentum 어느 창과도 안 겹침(12/24/48/96/288/2016 회피)
VOL_FEATURES = ["volatility_z", "atr_pct_rank_288", "realized_vol_ratio", "garch_vol_z", "compression_score"]
HORIZONS = ["h48", "h96"]
SPLITS = {
    "TRAIN": ("2024-01-01", "2025-08-31"),
    "VAL": ("2025-09-01", "2025-12-31"),
    "OOS": ("2026-01-01", "2026-03-31"),
}

sys.path.insert(0, str(ROOT / "scripts"))
import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402
omega = canon.omega


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    if len(x) < 30:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """x,y의 상관에서 z(통제변수)의 선형효과를 뺀 partial correlation. 반환: (raw corr, partial corr)."""
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]
    if len(x) < 30:
        return float("nan"), float("nan")
    rx, ry, rz = rankdata(x).astype(np.float64), rankdata(y).astype(np.float64), rankdata(z).astype(np.float64)
    raw = float(np.corrcoef(rx, ry)[0, 1])
    rxy, rxz, ryz = raw, np.corrcoef(rx, rz)[0, 1], np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    partial = float((rxy - rxz * ryz) / denom) if denom > 1e-8 else float("nan")
    return raw, partial


def main() -> None:
    train, eval_df = omega._load_omega_frames()[:2]
    feat_frames = []
    for feat in (train, eval_df):
        f = feat[["timestamp", *VOL_FEATURES, "log_return"]].copy()
        f["timestamp"] = pd.to_datetime(f["timestamp"])
        feat_frames.append(f)
    f2024 = pd.read_csv(ROOT / "data/splits/year_oos/training_features_2024.csv",
                         usecols=["timestamp", *VOL_FEATURES, "log_return"])
    f2024["timestamp"] = pd.to_datetime(f2024["timestamp"])
    feat_frames.insert(0, f2024)
    feat_all = pd.concat(feat_frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    lbl_frames = [pd.read_csv(LABEL_DIR / f"fwd_return_labels_{y}.csv", parse_dates=["timestamp"]) for y in (2024, 2025, 2026)]
    lbl_all = pd.concat(lbl_frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    df = feat_all.merge(lbl_all, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    # 후행(backward) 60bar 수익률 -- 158피쳐 어느 창과도 안 겹치는 신규계산, 순수 통제변수 용도
    df["backward_ret_60"] = df["log_return"].rolling(TREND_CONTROL_WINDOW, min_periods=30).sum()
    df = df.set_index("timestamp")

    print("=== 체크B: 후행 5시간 추세 통제 partial correlation ===", flush=True)
    partial_report = {}
    for split_name, (start, end) in SPLITS.items():
        sub = df.loc[start:end]
        for h in HORIZONS:
            y = sub[f"fwd_logret_{h}"].to_numpy()
            z = sub["backward_ret_60"].to_numpy()
            for vf in VOL_FEATURES:
                x = pd.to_numeric(sub[vf], errors="coerce").to_numpy()
                raw, partial = _partial_spearman(x, y, z)
                partial_report[(split_name, h, vf)] = {"raw": raw, "partial": partial}
        print(f"  [{split_name}]", flush=True)
        for h in HORIZONS:
            row = []
            for vf in VOL_FEATURES:
                r = partial_report[(split_name, h, vf)]
                row.append(f"{vf}: raw={r['raw']:+.4f}->partial={r['partial']:+.4f}")
            print(f"    {h}: " + " | ".join(row), flush=True)

    print("\n=== 체크A: 월별 블록 하위기간 부호일관성 (h48/h96, 5개 변동성피쳐) ===", flush=True)
    df_reset = df.reset_index()
    df_reset["month"] = df_reset["timestamp"].dt.to_period("M")
    months = sorted(df_reset["month"].unique())
    print(f"총 {len(months)}개 월별 블록: {months[0]} ~ {months[-1]}", flush=True)

    block_report = {}
    for h in HORIZONS:
        for vf in VOL_FEATURES:
            ics = []
            for mo in months:
                block = df_reset[df_reset["month"] == mo]
                x = pd.to_numeric(block[vf], errors="coerce").to_numpy()
                y = block[f"fwd_logret_{h}"].to_numpy()
                ic = _spearman(x, y)
                if not np.isnan(ic):
                    ics.append(ic)
            ics = np.array(ics)
            agg_sign = 1 if np.nanmean(ics) > 0 else -1
            frac_agree = float((np.sign(ics) == agg_sign).mean())
            block_report[(h, vf)] = {"n_blocks": len(ics), "mean_ic": float(np.mean(ics)),
                                       "frac_same_sign_as_aggregate": frac_agree, "per_block_ic": ics.tolist()}
            print(f"  {h} {vf:20s}: {len(ics)}개 블록, 평균IC={np.mean(ics):+.4f}, "
                  f"집계부호와 같은 블록 비율={frac_agree:.1%}", flush=True)

    out = {
        "trend_control_window": TREND_CONTROL_WINDOW,
        "partial_correlation": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in partial_report.items()},
        "monthly_block_stability": {f"{k[0]}|{k[1]}": v for k, v in block_report.items()},
    }
    out_path = ROOT / "tmp/eth_distributional_regression_volatility_pattern_verification_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
