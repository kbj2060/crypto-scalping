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


MODEL_ID = "alpha5_30_direction_learnable_20260519"
DEFAULT_IN_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_29_hier_label_factory_20260519"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_30_direction_learnable_20260519"
DIR_FEATURE_SPECS: list[tuple[str, float]] = [
    ("clean_regime4_2024_unsup_v1_directional_bias", 1.00),
    ("whale_retail_ratio", 0.80),
    ("mtf_trend_4h", 0.70),
    ("rsi", 0.60),
    ("breakout_strength", 0.55),
    ("funding_pressure", -0.50),
    ("clean_regime4_2024_unsup_v1_trend_bias", 0.45),
    ("clean_regime4_2024_unsup_v1_factor_trend", 0.40),
    ("mtf_trend_1h", 0.30),
    ("ai_dir_edge", 0.25),
    ("smart_money_flow", 0.20),
]


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(frame.get(col, default), errors="coerce").fillna(default)


def _robust_z(train: pd.Series, other: pd.Series) -> pd.Series:
    med = float(train.median())
    mad = float((train - med).abs().median())
    scale = max(mad * 1.4826, 1e-6)
    return ((other - med) / scale).clip(-5.0, 5.0)


def _build_score(train_ref: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    score = pd.Series(np.zeros(len(frame), dtype=np.float64), index=frame.index)
    total_abs = 0.0
    for col, weight in DIR_FEATURE_SPECS:
        if col not in frame.columns or col not in train_ref.columns:
            continue
        z = _robust_z(_num(train_ref, col, 0.0), _num(frame, col, 0.0))
        score = score + float(weight) * z
        total_abs += abs(float(weight))
    if total_abs <= 0:
        return np.zeros(len(frame), dtype=np.float64)
    return (score / total_abs).to_numpy(np.float64)


def _class_weights(y: np.ndarray, keep: np.ndarray) -> dict[int, float]:
    yy = np.asarray(y, dtype=np.int64)[np.asarray(keep, dtype=bool)]
    cls, cnt = np.unique(yy, return_counts=True)
    total = float(len(yy))
    return {int(c): float(total / (len(cls) * max(float(n), 1.0))) for c, n in zip(cls, cnt)}


def _augment(train_ref: pd.DataFrame, frame: pd.DataFrame, *, abs_score_min: float) -> pd.DataFrame:
    out = frame.copy()
    score = _build_score(train_ref, frame)
    dir_label = _num(out, "direction_label", 0.0).astype(np.int64).to_numpy()
    sign_match = np.where(dir_label == 1, score > 0, np.where(dir_label == 2, score < 0, False))
    strong = np.abs(score) >= float(abs_score_min)
    keep = (_num(out, "direction_train_keep", 0.0).astype(np.int8).to_numpy() == 1) & sign_match & strong
    weight = np.clip(_num(out, "direction_sample_weight", 1.0).to_numpy(np.float64) * (1.0 + np.abs(score)), 1e-4, None)
    cw = _class_weights(dir_label, keep)
    if cw:
        weight *= np.asarray([cw.get(int(v), 0.0) for v in dir_label], dtype=np.float64)
    weight *= keep.astype(np.float64)
    out["current_direction_score"] = score.astype(np.float32)
    out["direction_learnable_flag"] = keep.astype(np.int8)
    out["direction_train_keep30"] = keep.astype(np.int8)
    out["direction_sample_weight30"] = weight.astype(np.float32)
    return out


def _report(frame: pd.DataFrame) -> dict[str, Any]:
    work = frame[_num(frame, "split_keep", 0.0).astype(np.int8) == 1].copy()
    dir_all = _num(work, "direction_label", 0.0).astype(np.int64)
    keep = _num(work, "direction_train_keep30", 0.0).astype(np.int8) == 1
    score = _num(work, "current_direction_score", 0.0)
    res = {
        "rows": int(len(work)),
        "direction_rows_all": int(np.sum(dir_all != 0)),
        "direction_rows_keep30": int(np.sum(keep)),
        "direction_keep30_ratio_within_all_direction": float(np.mean(keep[dir_all != 0])) if np.any(dir_all != 0) else 0.0,
        "score_abs_quantiles_all_direction": score[dir_all != 0].abs().quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(4).to_dict() if np.any(dir_all != 0) else {},
    }
    by_reg = {}
    for reg, grp in work[dir_all != 0].groupby("regime4_state"):
        kk = _num(grp, "direction_train_keep30", 0.0).astype(np.int8) == 1
        by_reg[str(reg)] = {
            "rows_all": int(len(grp)),
            "rows_keep30": int(np.sum(kk)),
            "keep_ratio": float(np.mean(kk)) if len(grp) else 0.0,
            "score_mean": float(_num(grp, "current_direction_score", 0.0).mean()) if len(grp) else 0.0,
        }
    res["by_regime"] = by_reg
    return res


def main() -> None:
    p = argparse.ArgumentParser(description="Keep only learnable direction rows based on current-feature directional alignment.")
    p.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--abs-score-min", type=float, default=0.10)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_ref = pd.read_parquet(args.in_dir / "alpha5_29_hier_label_factory_train.parquet")
    report: dict[str, Any] = {"model_id": MODEL_ID, "abs_score_min": float(args.abs_score_min)}
    for split in ("train", "val", "oos"):
        df = pd.read_parquet(args.in_dir / f"alpha5_29_hier_label_factory_{split}.parquet")
        out = _augment(train_ref, df, abs_score_min=args.abs_score_min)
        out.to_parquet(args.out_dir / f"alpha5_30_direction_learnable_{split}.parquet", index=False)
        report[split] = _report(out)
    (args.out_dir / "alpha5_30_direction_learnable_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": "alpha5_30_done",
        "report_path": str(args.out_dir / "alpha5_30_direction_learnable_report.json"),
        "train_keep_ratio": report["train"]["direction_keep30_ratio_within_all_direction"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
