#!/usr/bin/env python3
"""CUSUM 거래량계열 신호(2026-08-20 pooled AUC 검증에서 p=0.000 유의, max_auc=0.5105,
top5 전부 trades/volume_btc/quote_volume/sum_open_interest_value/quote_volume_btc)가
2025/2026 두 기간 다 독립적으로 재현되는지 확인 -- pooled 검증 하나만으로는 "두 기간을
합쳐서 우연히 유의해진 것"과 "진짜 두 기간 다 있는 안정적 관계"를 구분 못 한다. 이 세션
전체에서 나온 첫 non-null 결과라 과대해석 방지를 위해 반드시 거쳐야 하는 체크."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402
omega = canon.omega

CUSUM_LABEL_DIR = ROOT / "tmp/eth_cusum_triple_barrier_labels_20260819"
TOP_FEATURES = ["trades", "volume_btc", "quote_volume", "sum_open_interest_value", "quote_volume_btc"]
RNG = np.random.default_rng(20260820)
N_PERM = 200


def auc_dir_agnostic(y: np.ndarray, x: np.ndarray) -> tuple[float, str]:
    valid = ~np.isnan(x)
    if valid.sum() < 30 or len(np.unique(y[valid])) < 2:
        return float("nan"), "?"
    auc = roc_auc_score(y[valid], x[valid])
    direction = "높을수록 LONG" if auc >= 0.5 else "높을수록 SHORT"
    return max(auc, 1.0 - auc), direction


def main() -> None:
    train, eval_df = omega._load_omega_frames()[:2]
    feat = {}
    for year, frame in ((2025, train), (2026, eval_df)):
        f = frame[["timestamp", *TOP_FEATURES]].copy()
        f["timestamp"] = pd.to_datetime(f["timestamp"])
        feat[year] = f

    for year in (2025, 2026):
        lbl = pd.read_csv(CUSUM_LABEL_DIR / f"zigzag_action_labels_{year}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
        data = feat[year].merge(lbl, on="timestamp", how="inner")
        events = data[data["zigzag_action"] != 0].reset_index(drop=True)
        y = (events["zigzag_action"] == 1).to_numpy().astype(np.int64)
        print(f"=== {year}년 단독: 이벤트 {len(events):,}개 (LONG={int(y.sum())} SHORT={int((1-y).sum())}) ===", flush=True)
        for c in TOP_FEATURES:
            x = pd.to_numeric(events[c], errors="coerce").to_numpy(dtype=np.float64)
            auc, direction = auc_dir_agnostic(y, x)
            print(f"  {c:30s} auc={auc:.4f} ({direction})", flush=True)
        print(flush=True)

    # --- pooled 재확인: 두 해 합쳐서 각 피쳐의 연도별 AUC가 같은 방향인지 표로 정리 ---
    print("=== 연도별 방향 일치 여부 (같은 방향이어야 '진짜 안정적 관계'에 가까움) ===", flush=True)
    summary = {}
    for c in TOP_FEATURES:
        row = {}
        for year in (2025, 2026):
            lbl = pd.read_csv(CUSUM_LABEL_DIR / f"zigzag_action_labels_{year}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
            data = feat[year].merge(lbl, on="timestamp", how="inner")
            events = data[data["zigzag_action"] != 0]
            y = (events["zigzag_action"] == 1).to_numpy().astype(np.int64)
            x = pd.to_numeric(events[c], errors="coerce").to_numpy(dtype=np.float64)
            valid = ~np.isnan(x)
            raw_auc = roc_auc_score(y[valid], x[valid])  # 방향 보존(0.5 기준 위/아래)
            row[str(year)] = float(raw_auc)
        same_sign = (row["2025"] - 0.5) * (row["2026"] - 0.5) > 0
        summary[c] = {**row, "same_direction_both_years": bool(same_sign)}
        print(f"  {c:30s} 2025_raw_auc={row['2025']:.4f} 2026_raw_auc={row['2026']:.4f} "
              f"{'[같은방향]' if same_sign else '[방향 반전!]'}", flush=True)

    out_path = ROOT / "tmp/eth_cusum_volume_signal_temporal_stability_20260820.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
