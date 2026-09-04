#!/usr/bin/env python3
"""ZDC(지그재그 방향확인, 완전wick앵커+고가/저가추적) 라벨을 9트리거 population 전체(TRAIN+VAL+OOS)에 부여.

`build_eth_5m_v_rebound_multitrigger_zigzag_direction_labels_20260901.py`(wick-앵커+종가추적판)와
동일 구조, 유일한 차이는 pivot 계산 함수(`zdc_first_pivot_full_wick` -- 시작앵커뿐 아니라 subsequent
봉 추적도 고가/저가 기준, `research_eth_v_rebound_multitrigger_zigzag_direction_full_wick_raw_lift_
check_20260901.py`에서 이미 raw-lift 검증됨: VAL/OOS 4칸 전부 lift>1.0, wick-앵커 대비 뚜렷한
추가개선은 없었음)과 출력 디렉터리뿐. research_diagnostic_not_live_wired.
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

import research_eth_v_rebound_multitrigger_zigzag_direction_full_wick_raw_lift_check_20260901 as fw  # noqa: E402

TRIGGER_LABELS = fw.close_anchor.TRIGGER_LABELS
OUT_DIR = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_zigzag_direction_full_wick_20260901"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/2] klines+atr 로딩...", flush=True)
    df = fw.load_klines()
    low = df["low"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    atr_pct = fw.close_anchor.zz._atr_pct(df, fw.close_anchor.ATR_WINDOW)
    print(f"  bars={len(df)}, {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}", flush=True)

    print("[2/2] 트리거 population 전체(TRAIN+VAL+OOS)에 라벨 부여...", flush=True)
    trig = pd.read_csv(TRIGGER_LABELS, usecols=["idx", "timestamp", "direction", "triggers", "n_triggers"])
    trig["timestamp"] = pd.to_datetime(trig["timestamp"], utc=True)

    rows = []
    n_hit = n_miss = n_unresolved = 0
    for _, row in trig.iterrows():
        idx = int(row["idx"])
        is_bottom = row["direction"] == "upside"
        pivot_type, extreme_idx, confirm_idx = fw.zdc_first_pivot_full_wick(
            low, high, atr_pct, idx, is_bottom, max_lookforward=fw.MAX_LOOKFORWARD_BARS
        )
        if pivot_type is None:
            n_unresolved += 1
            outcome = "미해결(제외)"
            hit = None
        else:
            matched = (pivot_type == "L") if is_bottom else (pivot_type == "H")
            hit = bool(matched)
            outcome = "확인(1)" if hit else "실패(0)"
            n_hit += int(hit)
            n_miss += int(not hit)
        rows.append({
            "idx": idx, "timestamp": row["timestamp"], "direction": row["direction"],
            "triggers": row["triggers"], "n_triggers": row["n_triggers"],
            "pivot_type": pivot_type, "extreme_idx": extreme_idx, "confirm_idx": confirm_idx,
            "hit": hit, "outcome": outcome,
        })

    labels = pd.DataFrame(rows)
    out_csv = OUT_DIR / "eth_5m_v_rebound_multitrigger_zigzag_direction_full_wick_labels.csv"
    labels.to_csv(out_csv, index=False)

    report = {
        "method": "zigzag_direction_confirmation_full_wick",
        "anchor": "wick (low[idx] for bottom, high[idx] for top), subsequent bars ALSO wick-tracked (high/low, not close)",
        "min_reversal_pct": fw.MIN_REVERSAL_PCT, "atr_multiplier": fw.ATR_MULTIPLIER,
        "max_lookforward_bars": fw.MAX_LOOKFORWARD_BARS,
        "n_total": len(labels), "n_hit": n_hit, "n_miss": n_miss, "n_unresolved": n_unresolved,
        "hit_rate_excl_unresolved": n_hit / (n_hit + n_miss) if (n_hit + n_miss) else None,
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nsaved: {out_csv}")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
