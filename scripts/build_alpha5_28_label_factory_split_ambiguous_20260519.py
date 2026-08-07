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


MODEL_ID = "alpha5_28_label_factory_split_ambiguous_20260519"
DEFAULT_IN_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_27_label_factory_20260519"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_28_label_factory_20260519"
ENTRY4_MAP = {
    0: "clean_wait",
    1: "ambiguous_structural",
    2: "ambiguous_trade_like",
    3: "trade",
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


def _augment(frame: pd.DataFrame, *, entry_event_ret_min: float, score_gate: float) -> pd.DataFrame:
    out = frame.copy()
    entry_state = _num(out, "entry_state", 1.0).astype(np.int64)
    action = _num(out, "label_action", 0.0).astype(np.int64)
    event_ret = _num(out, "meta_event_return", 0.0)
    raw_ret = _num(out, "meta_raw_terminal_return", 0.0)
    best_score = np.maximum(_num(out, "meta_long_score", 0.0), _num(out, "meta_short_score", 0.0))
    tp_first = _num(out, "meta_tp_first", 0.0).astype(np.int8)
    profitable = _num(out, "meta_is_profitable", 0.0).astype(np.int8)
    timeout = _num(out, "meta_timeout", 0.0).astype(np.int8)
    instability = _num(out, "regime_instability_flag", 0.0).astype(np.int8)
    regime = out["regime4_state"].astype(str).to_numpy()
    uniq = _num(out, "sample_uniqueness_weight", 0.0)
    conf = _num(out, "label_confidence", 0.0)
    quality = _num(out, "quality_score", 0.0)
    split_keep = _num(out, "split_keep", 0.0).astype(np.int8) == 1

    trade_like_core = (
        (action != 0)
        | (tp_first == 1)
        | (profitable == 1)
        | (best_score >= float(score_gate))
        | (event_ret >= float(entry_event_ret_min))
        | (np.abs(raw_ret) >= float(entry_event_ret_min))
    )
    structural_ambiguous = (
        (entry_state == 1)
        & (
            (regime == "whipsaw")
            | (instability == 1)
            | (timeout == 1)
            | (~trade_like_core)
        )
    )
    trade_like_ambiguous = (entry_state == 1) & (~structural_ambiguous)

    entry_state4 = np.select(
        [entry_state == 0, structural_ambiguous, trade_like_ambiguous, entry_state == 2],
        [0, 1, 2, 3],
        default=1,
    ).astype(np.int8)

    entry_keep4 = split_keep.astype(np.int8)
    entry_weight4 = np.clip(np.abs(quality) + 0.18, 1e-4, None)
    entry_weight4 *= (0.85 + 0.20 * conf)
    entry_weight4 *= (0.85 + 0.20 * np.clip(uniq, 0.0, 1.0))
    entry_weight4 *= np.where(entry_state4 == 1, 0.90, 1.0)
    entry_weight4 *= np.where(entry_state4 == 2, 1.05, 1.0)
    cw = _class_weights(entry_state4, entry_keep4 == 1)
    if cw:
        entry_weight4 *= np.asarray([cw.get(int(v), 0.0) for v in entry_state4], dtype=np.float64)

    out["entry_state4"] = entry_state4
    out["entry_state4_name"] = np.asarray([ENTRY4_MAP[int(v)] for v in entry_state4], dtype=object)
    out["ambiguous_structural_flag"] = structural_ambiguous.astype(np.int8)
    out["ambiguous_trade_like_flag"] = trade_like_ambiguous.astype(np.int8)
    out["entry_state4_train_keep"] = entry_keep4.astype(np.int8)
    out["entry_state4_sample_weight"] = entry_weight4.astype(np.float32)
    return out


def _report(frame: pd.DataFrame) -> dict[str, Any]:
    keep = _num(frame, "split_keep", 0.0).astype(np.int8) == 1
    work = frame.loc[keep].copy()
    state4 = _num(work, "entry_state4", 1.0).astype(np.int64)
    counts = {ENTRY4_MAP[int(k)]: int(v) for k, v in pd.Series(state4).value_counts().sort_index().to_dict().items()}
    month = pd.to_datetime(work["timestamp"], errors="coerce").dt.to_period("M").astype(str)
    monthly = []
    for key, grp in work.groupby(month):
        s = _num(grp, "entry_state4", 1.0).astype(np.int64)
        monthly.append({
            "month": key,
            "rows": int(len(grp)),
            "clean_wait_ratio": float(np.mean(s == 0)),
            "ambiguous_structural_ratio": float(np.mean(s == 1)),
            "ambiguous_trade_like_ratio": float(np.mean(s == 2)),
            "trade_ratio": float(np.mean(s == 3)),
        })
    return {
        "rows": int(len(work)),
        "entry_state4_counts": counts,
        "clean_wait_ratio": float(np.mean(state4 == 0)),
        "ambiguous_structural_ratio": float(np.mean(state4 == 1)),
        "ambiguous_trade_like_ratio": float(np.mean(state4 == 2)),
        "trade_ratio": float(np.mean(state4 == 3)),
        "ambiguous_trade_like_quality_mean": float(pd.to_numeric(work.loc[state4 == 2, "quality_score"], errors="coerce").fillna(0.0).mean()) if np.any(state4 == 2) else 0.0,
        "ambiguous_structural_quality_mean": float(pd.to_numeric(work.loc[state4 == 1, "quality_score"], errors="coerce").fillna(0.0).mean()) if np.any(state4 == 1) else 0.0,
        "ambiguous_trade_like_event_return_mean": float(pd.to_numeric(work.loc[state4 == 2, "meta_event_return"], errors="coerce").fillna(0.0).mean()) if np.any(state4 == 2) else 0.0,
        "ambiguous_structural_event_return_mean": float(pd.to_numeric(work.loc[state4 == 1, "meta_event_return"], errors="coerce").fillna(0.0).mean()) if np.any(state4 == 1) else 0.0,
        "monthly": monthly,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Split alpha5_27 ambiguous_wait into structural vs trade-like ambiguous states.")
    p.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--entry-event-ret-min", type=float, default=0.0045)
    p.add_argument("--score-gate", type=float, default=0.75)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"model_id": MODEL_ID}
    for split in ("train", "val", "oos"):
        src = args.in_dir / f"alpha5_27_label_factory_{split}.parquet"
        df = pd.read_parquet(src)
        out = _augment(df, entry_event_ret_min=args.entry_event_ret_min, score_gate=args.score_gate)
        out.to_parquet(args.out_dir / f"alpha5_28_label_factory_{split}.parquet", index=False)
        report[split] = _report(out)

    (args.out_dir / "alpha5_28_label_factory_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": "alpha5_28_done",
        "report_path": str(args.out_dir / "alpha5_28_label_factory_report.json"),
        "train_trade_ratio": report["train"]["trade_ratio"],
        "train_amb_trade_like_ratio": report["train"]["ambiguous_trade_like_ratio"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
