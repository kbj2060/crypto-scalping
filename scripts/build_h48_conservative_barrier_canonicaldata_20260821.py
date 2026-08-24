#!/usr/bin/env python3
"""h48_conservative triple-barrier 라벨을 canonical 데이터(`data/splits/year_oos/
training_features_*.csv`) 위에서 재계산 -- `build_omega1_2_triple_barrier_labels_20260619.py`의
로직/파라미터(BarrierConfig("h48_conservative", 48, 1.2, 0.8, 0.006, 0.004), ATR 계산,
_reason_and_return TP/SL/timeout 판정, quality 페널티 공식)는 그 모듈에서 그대로 import해서
씀 -- 재구현 아님. 유일한 차이는 입력 price 소스: 원본은 alpha6/7 계열
`trade_candidates_*_alpha6_current_tail111_exact.csv`(2026-08-10 커밋 4c46d20에서 생성
스크립트가 삭제돼 영구 재현불가, [[eth_omega4_quality_threshold_alpha67_pipeline_irreproducible_20260815]])를
썼는데, 이 스크립트는 timestamp/open/high/low/close 4개 컬럼만 있으면 되는 barrier 계산 자체엔
그 alpha6/7 피쳐가 전혀 필요 없다는 점(그냥 그 파일에 있던 OHLC를 썼을 뿐)에 착안해, 이미 이
세션에서 zigzag/cusum 재구축에 쓴 것과 동일한 표준 canonical 소스로 교체한 것.

⚠️ 이건 원본 h48_conservative의 "연장"이 아니라 **같은 레시피를 다른(더 표준적인) 가격소스
위에서 재계산한 것**이다 -- 겹치는 과거 구간(2025-01~2026-02-28)의 절대값이 원본과 완전히
일치한다는 보장은 없다(가격소스가 다르므로). 원본을 대체하지 않고 별도 산출물로 유지한다."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from build_omega1_2_triple_barrier_labels_20260619 import (  # noqa: E402
    CONFIGS,
    _build_config_labels,
    _prefixed,
    _read_market_frame,
)

FEE_RATE = 0.0005
SLIP_RATE = 0.0002
H48_CONS = next(c for c in CONFIGS if c.name == "h48_conservative")

OUT_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/h48_conservative_barrier_canonicaldata_20260821"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fee_cost = float(FEE_RATE + SLIP_RATE) * 2.0 * 3.0

    frames = {
        2024: ROOT / "data/splits/year_oos/training_features_2024.csv",
        2025: ROOT / "data/splits/year_oos/training_features_2025.csv",
        2026: ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
    }
    for year, path in frames.items():
        frame = _read_market_frame(path)
        labels = _build_config_labels(frame, H48_CONS, fee_cost=fee_cost)
        pref = _prefixed(labels, H48_CONS)
        # match downstream contract: column named tb_action_h48_conservative
        out_path = OUT_DIR / f"tb_h48_conservative_{year}.csv"
        pref.to_csv(out_path, index=False)
        counts = labels["tb_action"].value_counts().sort_index().to_dict()
        print(f"[{year}] rows={len(pref)} range={pref['timestamp'].min()}..{pref['timestamp'].max()} "
              f"action_counts={counts} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
