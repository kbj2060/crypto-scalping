#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha5_29_hier_label_factory_20260519"
DEFAULT_BASE_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_27_label_factory_20260519"
DEFAULT_SPLIT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_28_label_factory_20260519"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_29_hier_label_factory_20260519"
SUBTYPE_MAP = {
    0: "none",
    1: "ambiguous_structural",
    2: "ambiguous_trade_like",
}


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(frame.get(col, default), errors="coerce").fillna(default).to_numpy(np.float64)


def _class_weights(y: np.ndarray, keep: np.ndarray) -> dict[int, float]:
    y = np.asarray(y, dtype=np.int64)
    keep = np.asarray(keep, dtype=bool)
    yy = y[keep]
    classes, counts = np.unique(yy, return_counts=True)
    total = float(len(yy))
    return {int(cls): float(total / (len(classes) * max(float(cnt), 1.0))) for cls, cnt in zip(classes, counts)}


def _augment(base: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    state4 = _num(split, "entry_state4", 1.0).astype(np.int8)
    subtype = np.where(state4 == 1, 1, np.where(state4 == 2, 2, 0)).astype(np.int8)
    out["ambiguous_subtype"] = subtype
    out["ambiguous_subtype_name"] = np.asarray([SUBTYPE_MAP[int(v)] for v in subtype], dtype=object)
    out["entry_band4"] = state4
    out["entry_band4_name"] = split["entry_state4_name"].astype(str).to_numpy()
    keep = (_num(out, "split_keep", 0.0).astype(np.int8) == 1) & (_num(out, "entry_state", 1.0).astype(np.int8) == 1)
    weight = np.clip(np.abs(_num(out, "quality_score", 0.0)) + 0.15, 1e-4, None)
    weight *= (0.85 + 0.20 * _num(out, "label_confidence", 0.0))
    weight *= (0.90 + 0.20 * np.clip(_num(out, "sample_uniqueness_weight", 0.0), 0.0, 1.0))
    cw = _class_weights(subtype, keep)
    if cw:
        weight *= np.asarray([cw.get(int(v), 0.0) for v in subtype], dtype=np.float64)
    weight *= np.where(subtype == 0, 0.0, 1.0)
    out["ambiguous_subtype_train_keep"] = keep.astype(np.int8)
    out["ambiguous_subtype_sample_weight"] = weight.astype(np.float32)
    return out


def _report(frame: pd.DataFrame) -> dict[str, Any]:
    work = frame[_num(frame, "split_keep", 0.0).astype(np.int8) == 1].copy()
    state = _num(work, "entry_state", 1.0).astype(np.int8)
    subtype = _num(work, "ambiguous_subtype", 0.0).astype(np.int8)
    amb = work[state == 1].copy()
    subtype_counts = {SUBTYPE_MAP[int(k)]: int(v) for k, v in pd.Series(subtype).value_counts().sort_index().to_dict().items()}
    return {
        "rows": int(len(work)),
        "entry_state_counts": {str(int(k)): int(v) for k, v in pd.Series(state).value_counts().sort_index().to_dict().items()},
        "ambiguous_subtype_counts": subtype_counts,
        "ambiguous_trade_like_ratio_within_ambiguous": float(np.mean(_num(amb, "ambiguous_subtype", 0.0) == 2)) if len(amb) else 0.0,
        "ambiguous_structural_ratio_within_ambiguous": float(np.mean(_num(amb, "ambiguous_subtype", 0.0) == 1)) if len(amb) else 0.0,
        "ambiguous_trade_like_event_return_mean": float(pd.to_numeric(amb.loc[_num(amb, "ambiguous_subtype", 0.0) == 2, "meta_event_return"], errors="coerce").fillna(0.0).mean()) if len(amb) else 0.0,
        "ambiguous_structural_event_return_mean": float(pd.to_numeric(amb.loc[_num(amb, "ambiguous_subtype", 0.0) == 1, "meta_event_return"], errors="coerce").fillna(0.0).mean()) if len(amb) else 0.0,
        "ambiguous_trade_like_quality_mean": float(pd.to_numeric(amb.loc[_num(amb, "ambiguous_subtype", 0.0) == 2, "quality_score"], errors="coerce").fillna(0.0).mean()) if len(amb) else 0.0,
        "ambiguous_structural_quality_mean": float(pd.to_numeric(amb.loc[_num(amb, "ambiguous_subtype", 0.0) == 1, "quality_score"], errors="coerce").fillna(0.0).mean()) if len(amb) else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build hierarchical alpha5_29 labels: hard 3-state entry + ambiguous subtype.")
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"model_id": MODEL_ID}
    for split in ("train", "val", "oos"):
        base = pd.read_parquet(args.base_dir / f"alpha5_27_label_factory_{split}.parquet")
        band = pd.read_parquet(args.split_dir / f"alpha5_28_label_factory_{split}.parquet")
        out = _augment(base, band)
        out.to_parquet(args.out_dir / f"alpha5_29_hier_label_factory_{split}.parquet", index=False)
        report[split] = _report(out)
    (args.out_dir / "alpha5_29_hier_label_factory_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": "alpha5_29_done",
        "report_path": str(args.out_dir / "alpha5_29_hier_label_factory_report.json"),
        "train_amb_trade_like_ratio_within_ambiguous": report["train"]["ambiguous_trade_like_ratio_within_ambiguous"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
