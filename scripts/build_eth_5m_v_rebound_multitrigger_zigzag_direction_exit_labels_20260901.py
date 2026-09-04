#!/usr/bin/env python3
"""ZDC(wick-앵커) 라벨에 "두 번째 피벗"(=확정된 스윙이 끝나는 지점) 정보를 추가 -- exit 후보.

entry(첫 피벗 확정, confirm_idx)까지는 build_eth_5m_v_rebound_multitrigger_zigzag_direction_
labels_20260901.py가 이미 계산해뒀다. 이 스크립트는 그 확정 시점부터 지그재그 상태머신을
계속 돌려(_zigzag_pivots()의 trend==1/trend==-1 분기 그대로 재현, 재구현 아님 -- 첫 피벗 이후의
"스윙이 반대로 꺾이는 지점"을 찾는 것뿐) 두 번째 피벗을 찾는다. 이게 "진입 방향으로 확정된
움직임이 끝나는 지점" = 자연스러운 exit 후보다.

두 번째 피벗은 hit(True/False) 무관하게 모든 해상된 이벤트에 대해 계산(분석용) -- 실제 exit
전략으로 쓸지는 이후 경제성게이트 단계(계획서 Step E)에서 별도 판단.

라벨/피쳐 자체는 변경하지 않는다(TabPFN 학습에 영향 없음) -- 순수 추가 컬럼.
research_diagnostic_not_live_wired.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import research_eth_v_rebound_multitrigger_zigzag_direction_wick_anchor_raw_lift_check_20260901 as wick  # noqa: E402

LABEL_DIR = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_zigzag_direction_20260901"
LABEL_CSV = LABEL_DIR / "eth_5m_v_rebound_multitrigger_zigzag_direction_labels.csv"
OUT_CSV = LABEL_DIR / "eth_5m_v_rebound_multitrigger_zigzag_direction_labels_with_exit.csv"

MIN_REVERSAL_PCT = wick.MIN_REVERSAL_PCT
ATR_MULTIPLIER = wick.ATR_MULTIPLIER
MAX_LOOKFORWARD_BARS = wick.MAX_LOOKFORWARD_BARS


def zdc_second_pivot(close: np.ndarray, atr_pct: np.ndarray, first_pivot_type: str, confirm_idx: int, *, max_lookforward: int):
    """_zigzag_pivots()의 trend==1/trend==-1 분기를 confirm_idx 직후 상태에서 그대로 재현(early-exit).
    첫 피벗="L"이면 이제 uptrend(다음은 "H" 탐색), 첫 피벗="H"면 downtrend(다음은 "L" 탐색) --
    실제 _zigzag_pivots()가 첫 피벗을 append한 직후 trend를 뒤집고 새 극값 추적을 시작하는 것과
    동일. subsequent 봉은 entry와 동일하게 종가로 추적.

    Returns (second_pivot_type, second_extreme_idx, second_confirm_idx) 또는 (None,None,None).
    """
    n = len(close)
    end = min(confirm_idx + 1 + max_lookforward, n)
    if first_pivot_type == "L":
        trend = 1
        high_idx, high_price = confirm_idx, close[confirm_idx]
        low_idx = low_price = None
    else:
        trend = -1
        low_idx, low_price = confirm_idx, close[confirm_idx]
        high_idx = high_price = None

    for i in range(confirm_idx + 1, end):
        price = close[i]
        if not np.isfinite(price):
            continue
        thr = max(MIN_REVERSAL_PCT, float(atr_pct[i]) * ATR_MULTIPLIER)
        if trend == 1:
            if price > high_price:
                high_idx, high_price = i, price
            drop = high_price / max(price, 1e-12) - 1.0
            if drop >= thr:
                return "H", high_idx, i
        else:
            if price < low_price:
                low_idx, low_price = i, price
            rise = price / max(low_price, 1e-12) - 1.0
            if rise >= thr:
                return "L", low_idx, i
    return None, None, None


def main() -> None:
    print("[1/3] klines+atr 로딩...", flush=True)
    df = wick.load_klines()
    close = df["close"].to_numpy(dtype=np.float64)
    atr_pct = wick.close_anchor.zz._atr_pct(df, wick.close_anchor.ATR_WINDOW)

    print("[2/3] 저장된 라벨 로딩 + 두 번째 피벗(exit 후보) 계산...", flush=True)
    labels = pd.read_csv(LABEL_CSV)
    labels["hit_bool"] = labels["hit"].astype(str).map({"True": True, "False": False})

    exit_pivot_type, exit_extreme_idx, exit_confirm_idx = [], [], []
    n_exit_resolved = n_exit_unresolved = n_skipped = 0
    for _, row in labels.iterrows():
        if row["hit_bool"] not in (True, False):
            exit_pivot_type.append(None); exit_extreme_idx.append(None); exit_confirm_idx.append(None)
            n_skipped += 1
            continue
        first_type = row["pivot_type"]
        confirm_idx = int(row["confirm_idx"])
        e_type, e_extreme, e_confirm = zdc_second_pivot(close, atr_pct, first_type, confirm_idx, max_lookforward=MAX_LOOKFORWARD_BARS)
        exit_pivot_type.append(e_type); exit_extreme_idx.append(e_extreme); exit_confirm_idx.append(e_confirm)
        if e_type is None:
            n_exit_unresolved += 1
        else:
            n_exit_resolved += 1

    labels["exit_pivot_type"] = exit_pivot_type
    labels["exit_extreme_idx"] = exit_extreme_idx
    labels["exit_confirm_idx"] = exit_confirm_idx
    labels["bars_entry_confirm_to_exit_confirm"] = labels["exit_confirm_idx"] - labels["confirm_idx"]
    labels = labels.drop(columns=["hit_bool"])

    print("[3/3] 저장...", flush=True)
    labels.to_csv(OUT_CSV, index=False)

    resolved_mask = labels["hit"].astype(str).isin(["True", "False"])
    bars_stat = labels.loc[resolved_mask & labels["exit_confirm_idx"].notna(), "bars_entry_confirm_to_exit_confirm"]
    report = {
        "n_total": len(labels),
        "n_entry_resolved": int(resolved_mask.sum()),
        "n_exit_resolved": n_exit_resolved, "n_exit_unresolved": n_exit_unresolved, "n_entry_unresolved_skipped": n_skipped,
        "bars_entry_to_exit": {
            "mean": float(bars_stat.mean()) if len(bars_stat) else None,
            "median": float(bars_stat.median()) if len(bars_stat) else None,
            "p10": float(bars_stat.quantile(0.10)) if len(bars_stat) else None,
            "p90": float(bars_stat.quantile(0.90)) if len(bars_stat) else None,
        },
        "output": str(OUT_CSV),
    }
    (LABEL_DIR / "exit_labels_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
