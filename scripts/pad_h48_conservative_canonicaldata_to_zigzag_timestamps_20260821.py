#!/usr/bin/env python3
"""h48_conservative(canonical-recomputed, `build_h48_conservative_barrier_canonicaldata_20260821.py`)의
tb_action을 이 세션에서 재구축한 zigzag(`zigzag_action_labels_rebuilt_20260821`, 2026-08-19까지)
timestamp 인덱스 위에 패딩한다(미매칭=CASH). 로직은
`pad_h48_quality_labels_to_zigzag_timestamps_eth_extended_20260809.py`와 동일(재구현 아님,
소스 경로만 이 세션의 신규 파일로 교체)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TB_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/h48_conservative_barrier_canonicaldata_20260821"
ZIGZAG_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/zigzag_action_labels_rebuilt_20260821"
OUT_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/h48_conservative_padded_to_zigzag_canonicaldata_20260821"
BARRIER_COL = "tb_action_h48_conservative"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {"barrier_source_col": BARRIER_COL, "artifacts": {}, "summaries": {}}
    for year in (2024, 2025, 2026):
        tb = pd.read_csv(TB_DIR / f"tb_h48_conservative_{year}.csv", parse_dates=["timestamp"], usecols=["timestamp", BARRIER_COL])
        zigzag_path = ZIGZAG_DIR / f"zigzag_action_labels_{year}.csv"
        zigzag = pd.read_csv(zigzag_path, parse_dates=["timestamp"], usecols=["timestamp"])
        merged = zigzag.merge(tb.rename(columns={BARRIER_COL: "zigzag_action"}), on="timestamp", how="left", validate="one_to_one")
        n_missing = int(merged["zigzag_action"].isna().sum())
        merged["zigzag_action"] = merged["zigzag_action"].fillna(0).astype(np.int64)
        out_path = OUT_DIR / f"zigzag_action_labels_{year}.csv"
        merged.to_csv(out_path, index=False)
        counts = merged["zigzag_action"].value_counts().sort_index().to_dict()
        audit["artifacts"][str(year)] = str(out_path)
        audit["summaries"][str(year)] = {
            "rows": int(len(merged)),
            "missing_timestamps_filled_as_cash": n_missing,
            "counts": {str(int(k)): int(v) for k, v in counts.items()},
        }
        print(f"[{year}] rows={len(merged)} missing_filled_as_cash={n_missing} counts={counts} -> {out_path}", flush=True)

    audit_path = OUT_DIR / "h48_conservative_padded_label_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
