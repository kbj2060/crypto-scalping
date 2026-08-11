"""h384_conservative(quality_head 재검토판, horizon=384bar) 배리어 action을 direction_head가 쓰는
zigzag 타임스탬프 그리드에 패딩(매칭 안 되는 지점은 CASH). h48qual --quality-label-dir 계약
(zigzag_action_labels_{year}.csv, 컬럼명 zigzag_action)을 그대로 따른다. ZIGZAG_DIR은 오늘자
zig075 final15 스크립트(train_eval_omega4_3head_parent72_eth_zig075_regime_jmredesign_final15_
20260811.py)가 쓰는 --direction-label-dir과 동일 -- 같은 타임스탬프 그리드에 조인되어야 하므로."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TB_DIR = ROOT / "tmp/eth_h384_conservative_triple_barrier_labels_20260811"
ZIGZAG_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
OUT_DIR = ROOT / "tmp/eth_h384_conservative_padded_to_zigzag_timestamps_20260811"
BARRIER_COL = "tb_action_h384_conservative"


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

    train_tb = pd.read_csv(TB_DIR / "train_triple_barrier_labels.csv", parse_dates=["timestamp"], usecols=["timestamp", BARRIER_COL])
    val_tb = pd.read_csv(TB_DIR / "validation_triple_barrier_labels.csv", parse_dates=["timestamp"], usecols=["timestamp", BARRIER_COL])
    oos_tb = pd.read_csv(TB_DIR / "oos_triple_barrier_labels.csv", parse_dates=["timestamp"], usecols=["timestamp", BARRIER_COL])
    tb_by_year = {
        2025: pd.concat([train_tb, val_tb], ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last"),
        2026: oos_tb.sort_values("timestamp").drop_duplicates("timestamp", keep="last"),
    }

    audit: dict[str, Any] = {"barrier_source_col": BARRIER_COL, "artifacts": {}, "summaries": {}}
    for year, tb in tb_by_year.items():
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

    audit_path = OUT_DIR / "eth_h384_padded_label_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
