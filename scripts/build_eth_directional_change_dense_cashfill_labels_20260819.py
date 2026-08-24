#!/usr/bin/env python3
"""
ETH Directional-Change 라벨의 dense CASH-fill 재export.

`scripts/build_eth_directional_change_triple_barrier_labels_20260819.py`가 만든 sparse
이벤트 라벨(tmp/eth_directional_change_triple_barrier_labels_20260819/zigzag_action_labels_
{year}.csv, 전체 bar의 6~8%만 존재)을 TabM 학습 스크립트
(scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py)에 그대로 먹이면
안 되는 이유(둘 다 코드로 직접 확인됨):

1. `omega._align()`이 inner-join이라 라벨에 없는 feature row 자체가 통째로 사라져 학습/평가
   표본이 이벤트 수준(6~8%)으로 축소된다.
2. exit_head(`_continue_to_barrier_net`/`_build_exit_dataset_independent`)와
   `omega._metrics()`가 "인접 행 = 인접 5분 bar"를 가정하는데, sparse 라벨에서는 인접 행이
   실제로 몇 시간~며칠 떨어져 있을 수 있어 TP/SL 시뮬레이션이 크래시 없이 조용히 왜곡된다.

이 스크립트는 DC 이벤트 검출/배리어 로직 자체를 다시 계산하지 않는다 — sparse 산출물을
canonical 원시 bar 타임스탬프 그리드에 그대로 재색인(left-join)하고, DC 이벤트가 없는 bar는
zigzag_action=CASH(0)로 채운다. 이건 forward-fill(다음 이벤트까지 방향값을 미래로 미루는 것)과
다르다 — CASH는 "이 bar엔 신호가 없었다"는 사실 그 자체를 명시할 뿐, 어떤 방향 판정도
연장하지 않는다(/home/kbj20/.claude/plans/1-velvety-whistle.md 참고).

sparse 원본(tmp/eth_directional_change_triple_barrier_labels_20260819/)은 그대로 둔다 —
기존 실험 문서/OOS 차트/DC-vs-CUSUM 비교와의 정합성 유지. 이 dense 파일은 TabM 학습이라는
특정 소비자를 위한 재표현일 뿐이다.

⚠️ 이 dense 라벨을 실제로 소비할 때는 canonical 원시 bar(data/splits/year_oos/
training_features_{2025,2026_rebuilt}.csv)로 override된 omega.TRAIN_CSV/EVAL_CSV와 반드시
짝을 이뤄야 한다(scripts/eth_directional_change_tabm_training_canonicaldata_20260819.py) —
이 스크립트가 재색인 기준으로 쓴 grid와 다른 timestamp 그리드(예: legacy omega.TRAIN_CSV
기본값)로 소비하면 상위집합(superset) 보장이 깨질 수 있다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SPARSE_DIR = ROOT / "tmp/eth_directional_change_triple_barrier_labels_20260819"
DEFAULT_OUT = ROOT / "tmp/eth_directional_change_triple_barrier_labels_dense_cashfill_20260819"

# TabM 학습 스크립트(_prepare_frames())는 2025/2026만 읽지만, N-HiTS/ModernTCN 후보
# (train_eval_eth_direction_quality_nhits_moderntcn_20260816.py::load_panel_and_labels())는
# 2024도 읽는다 — 두 소비자를 모두 지원하려면 2024도 만들어야 함.
YEARS = (2024, 2025, 2026)
RAW_GRID_PATHS = {
    2024: ROOT / "data/splits/year_oos/training_features_2024.csv",
    2025: ROOT / "data/splits/year_oos/training_features_2025.csv",
    2026: ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
}

NUMERIC_DIAGNOSTIC_COLS = [
    "realized_ret", "vol_at_entry", "bars_held",
    "trend_tstat", "trend_slope", "trend_horizon",
    "weight_uniqueness", "weight_return_attr", "weight",
]


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sparse-dir", type=Path, default=DEFAULT_SPARSE_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = build_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_year_summary: dict[str, dict] = {}
    for year in YEARS:
        sparse_path = args.sparse_dir / f"zigzag_action_labels_{year}.csv"
        grid_path = RAW_GRID_PATHS[year]

        sparse = pd.read_csv(sparse_path, parse_dates=["timestamp"])
        grid = (
            pd.read_csv(grid_path, usecols=["timestamp"], parse_dates=["timestamp"])
            .dropna()
            .sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)
        )

        dense = grid.merge(sparse, on="timestamp", how="left")
        is_event = dense["zigzag_action"].notna()

        # --- fail-loud 검증: 재색인이 행을 잃거나 이벤트 값을 변형하지 않았는지 ---
        if len(dense) != len(grid):
            raise RuntimeError(f"{year}: merge 후 행수({len(dense)})가 grid({len(grid)})와 다름")
        if not dense["timestamp"].equals(grid["timestamp"]):
            raise RuntimeError(f"{year}: merge 후 timestamp 순서가 grid와 어긋남")
        roundtrip = dense.loc[is_event, ["timestamp", "zigzag_action"]].merge(
            sparse[["timestamp", "zigzag_action"]], on="timestamp", suffixes=("", "_orig")
        )
        if len(roundtrip) != int(is_event.sum()):
            raise RuntimeError(f"{year}: 이벤트 bar 매칭 개수 불일치 — merge 왕복 중 유실")
        if not (roundtrip["zigzag_action"] == roundtrip["zigzag_action_orig"]).all():
            raise RuntimeError(f"{year}: 이벤트 bar의 zigzag_action 값이 merge 왕복 중 변형됨")

        dense["zigzag_action"] = dense["zigzag_action"].fillna(0).astype("int64")
        dense["touch_type"] = dense["touch_type"].fillna("no_event")
        for c in NUMERIC_DIAGNOSTIC_COLS:
            dense[c] = dense[c].fillna(0.0)

        out_path = args.out_dir / f"zigzag_action_labels_{year}.csv"
        dense.to_csv(out_path, index=False)

        dist = dense["zigzag_action"].value_counts(normalize=True).sort_index()
        per_year_summary[str(year)] = {
            "n_rows": int(len(dense)),
            "n_events": int(is_event.sum()),
            "event_ratio": float(is_event.mean()),
            "label_dist": {str(k): float(v) for k, v in dist.items()},
            "output_path": str(out_path),
        }
        print(f"[{year}] n_rows={len(dense):,} n_events={int(is_event.sum()):,} "
              f"({is_event.mean()*100:.2f}%) dist={per_year_summary[str(year)]['label_dist']} -> {out_path}")

    report = {
        "type": "directional_change_dense_cashfill_relabel",
        "source_sparse_dir": str(args.sparse_dir),
        "note": "이 파일은 sparse 원본(source_sparse_dir)을 canonical bar 그리드에 재색인만 한 "
                "것 — DC 이벤트 검출/배리어 로직은 재계산하지 않음(이벤트 bar 값은 원본과 "
                "왕복검증으로 동일함을 확인). 비이벤트 bar는 zigzag_action=CASH(0)로 명시할 "
                "뿐 방향을 forward-fill하지 않는다.",
        "per_year_summary": per_year_summary,
    }
    import json
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[report] {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
