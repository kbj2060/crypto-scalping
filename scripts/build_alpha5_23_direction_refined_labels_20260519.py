#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha5_23_direction_refined_labels_20260519"
DEFAULT_IN_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_high_quality_training_data_20260518"
DEFAULT_COMPARE_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_18_hgb_soft_labels_20260518"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_23_direction_refined_labels_20260519"

REGIME_AMBIGUITY_MIN = {
    "bull": 2.0,
    "bear": 2.0,
    "chop": 1.5,
    "whipsaw": 9e9,
}
REGIME_CONSENSUS_MIN = {
    "bull": 1.0,
    "bear": 1.0,
    "chop": 0.75,
    "whipsaw": 9e9,
}


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(frame.get(col, default), errors="coerce").fillna(default).to_numpy(np.float64)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _class_weights(labels: np.ndarray, valid_mask: np.ndarray) -> dict[int, float]:
    y = np.asarray(labels, dtype=np.int64)
    mask = np.asarray(valid_mask, dtype=bool)
    y = y[mask]
    if len(y) == 0:
        return {}
    mn = int(np.min(y))
    mx = int(np.max(y))
    cnt = np.bincount(y - mn).astype(np.float64)
    total = max(float(cnt.sum()), 1.0)
    out: dict[int, float] = {}
    for offset, count in enumerate(cnt):
        cls = mn + offset
        out[int(cls)] = float(total / (len(cnt) * max(count, 1.0)))
    return out


def _augment(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    action = _num(out, "label_action", 0.0).astype(np.int64)
    keep = _num(out, "label_train_keep", 0.0).astype(np.int8)
    consensus = _num(out, "label_consensus", 0.0)
    base_weight = _num(out, "label_sample_weight", 0.0)
    uniq = _num(out, "sample_uniqueness_weight", 0.0)
    edge_gap = _num(out, "meta_edge_gap", 0.0)
    tp_first = _num(out, "meta_tp_first", 0.0).astype(np.int8)
    profitable = _num(out, "meta_is_profitable", 0.0).astype(np.int8)
    event_ret = _num(out, "meta_event_return", 0.0)
    raw_ret = _num(out, "meta_raw_terminal_return", 0.0)
    long_score = _num(out, "meta_long_score", 0.0)
    short_score = _num(out, "meta_short_score", 0.0)
    selected = _num(out, "regime_trade_selected", 0.0).astype(np.int8)
    regime = out["regime4_state"].astype(str).to_numpy()

    best_score = np.maximum(long_score, short_score)
    cash_mask = action == 0
    trade_mask = action != 0

    cash_contam = cash_mask & ((best_score >= 1.0) | (event_ret >= 0.005))
    trade_contam = trade_mask & ((profitable != 1) | (tp_first != 1) | (event_ret < 0.005))

    direction_amb = np.zeros(len(out), dtype=np.int8)
    direction_valid = np.zeros(len(out), dtype=np.int8)

    for reg in ("bull", "bear", "chop", "whipsaw"):
        reg_mask = regime == reg
        amb_min = float(REGIME_AMBIGUITY_MIN[reg])
        con_min = float(REGIME_CONSENSUS_MIN[reg])
        if reg != "whipsaw":
            direction_amb[reg_mask & (edge_gap < amb_min)] = 1
            valid = (
                reg_mask
                & trade_mask
                & (tp_first == 1)
                & (profitable == 1)
                & (selected == 1)
                & (consensus >= con_min)
                & (edge_gap >= amb_min)
            )
            direction_valid[valid] = 1
        else:
            direction_amb[reg_mask] = 1

    entry_label = trade_mask.astype(np.int8)
    entry_keep = (keep == 1) & (~cash_contam) & (~trade_contam)
    entry_keep = entry_keep.astype(np.int8)

    amb_scale = np.where(edge_gap >= 2.0, 1.15, np.where(edge_gap >= 1.5, 1.05, 0.92))
    contam_scale = np.where(cash_contam | trade_contam, 0.0, 1.0)
    entry_weight = np.clip(base_weight, 1e-4, None)
    entry_weight *= (0.85 + 0.30 * _clip01(consensus))
    entry_weight *= (0.90 + 0.10 * tp_first + 0.10 * profitable)
    entry_weight *= amb_scale
    entry_weight *= contam_scale
    entry_weight = entry_weight.astype(np.float32)

    direction_label = np.where(direction_valid == 1, action, 0).astype(np.int8)
    direction_keep = direction_valid.astype(np.int8)
    direction_weight = np.clip(base_weight, 1e-4, None)
    direction_weight *= (0.80 + 0.40 * _clip01(consensus))
    direction_weight *= (1.0 + 0.08 * np.clip(edge_gap, 0.0, 8.0))
    direction_weight *= (1.0 + 0.15 * tp_first + 0.10 * profitable)
    direction_weight *= np.where(direction_valid == 1, 1.0, 0.0)
    direction_weight = direction_weight.astype(np.float32)

    entry_cw = _class_weights(entry_label, entry_keep == 1)
    if entry_cw:
        entry_weight *= np.asarray([entry_cw.get(int(y), 0.0) for y in entry_label], dtype=np.float32)
    direction_cw = _class_weights(direction_label, direction_keep == 1)
    if direction_cw:
        direction_weight *= np.asarray([direction_cw.get(int(y), 0.0) for y in direction_label], dtype=np.float32)

    out["cash_contamination_flag"] = cash_contam.astype(np.int8)
    out["trade_contamination_flag"] = trade_contam.astype(np.int8)
    out["direction_ambiguity_flag"] = direction_amb.astype(np.int8)
    out["entry_label"] = entry_label.astype(np.int8)
    out["entry_train_keep"] = entry_keep.astype(np.int8)
    out["entry_sample_weight"] = entry_weight
    out["direction_label"] = direction_label.astype(np.int8)
    out["direction_valid"] = direction_valid.astype(np.int8)
    out["direction_train_keep"] = direction_keep.astype(np.int8)
    out["direction_sample_weight"] = direction_weight
    return out


def _amb_rates(frame: pd.DataFrame, mask: np.ndarray | None = None) -> dict[str, float]:
    edge_gap = _num(frame, "meta_edge_gap", 0.0)
    if mask is None:
        mask = np.ones(len(frame), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    denom = max(float(np.sum(mask)), 1.0)
    return {
        "lt_1_0": float(np.sum(mask & (edge_gap < 1.0)) / denom),
        "lt_1_5": float(np.sum(mask & (edge_gap < 1.5)) / denom),
        "lt_2_0": float(np.sum(mask & (edge_gap < 2.0)) / denom),
        "median": float(np.median(edge_gap[mask])) if np.any(mask) else 0.0,
    }


def _regime_direction_rows(frame: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for reg, grp in frame.groupby("regime4_state"):
        out[str(reg)] = int(np.sum(_num(grp, "direction_valid", 0.0) == 1))
    return out


def _regime_purity(frame: pd.DataFrame) -> dict[str, Any]:
    valid = _num(frame, "direction_valid", 0.0) == 1
    action = _num(frame, "direction_label", 0.0).astype(np.int64)
    raw_ret = _num(frame, "meta_raw_terminal_return", 0.0)
    out: dict[str, Any] = {}
    for reg, grp_idx in pd.Series(frame["regime4_state"].astype(str)).groupby(frame["regime4_state"].astype(str)).groups.items():
        idx = np.asarray(list(grp_idx), dtype=np.int64)
        m = valid[idx]
        if not np.any(m):
            out[str(reg)] = {"rows": 0, "long_rows": 0, "short_rows": 0, "long_purity": 0.0, "short_purity": 0.0, "edge_gap_median": 0.0}
            continue
        ii = idx[m]
        a = action[ii]
        r = raw_ret[ii]
        e = _num(frame.iloc[ii], "meta_edge_gap", 0.0)
        long_mask = a == 1
        short_mask = a == 2
        out[str(reg)] = {
            "rows": int(len(ii)),
            "long_rows": int(np.sum(long_mask)),
            "short_rows": int(np.sum(short_mask)),
            "long_purity": float(np.mean(r[long_mask] > 0.0)) if np.any(long_mask) else 0.0,
            "short_purity": float(np.mean(r[short_mask] < 0.0)) if np.any(short_mask) else 0.0,
            "edge_gap_median": float(np.median(e)) if len(e) else 0.0,
        }
    return out


def _split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    action = _num(frame, "label_action", 0.0).astype(np.int64)
    entry = _num(frame, "entry_label", 0.0).astype(np.int64)
    direction_valid = _num(frame, "direction_valid", 0.0).astype(np.int64) == 1
    cash_mask = action == 0
    trade_mask = action != 0
    cash_contam = _num(frame, "cash_contamination_flag", 0.0).astype(np.int64) == 1
    trade_contam = _num(frame, "trade_contamination_flag", 0.0).astype(np.int64) == 1
    out = {
        "rows": int(len(frame)),
        "action_counts": {str(int(k)): int(v) for k, v in pd.Series(action).value_counts().sort_index().to_dict().items()},
        "entry_trade_ratio": float(np.mean(entry == 1)),
        "entry_keep_ratio": float(np.mean(_num(frame, "entry_train_keep", 0.0) == 1)),
        "direction_valid_ratio": float(np.mean(direction_valid)),
        "direction_valid_rows_by_regime": _regime_direction_rows(frame),
        "ambiguity_all": _amb_rates(frame),
        "ambiguity_direction_valid": _amb_rates(frame, direction_valid),
        "cash_contamination_rate": float(np.mean(cash_contam[cash_mask])) if np.any(cash_mask) else 0.0,
        "trade_contamination_rate": float(np.mean(trade_contam[trade_mask])) if np.any(trade_mask) else 0.0,
        "regime_purity": _regime_purity(frame),
        "whipsaw_direction_valid_rows": int(np.sum((frame["regime4_state"].astype(str) == "whipsaw") & direction_valid)),
    }
    return out


def _compare_soft(alpha5_23: pd.DataFrame, alpha5_18: pd.DataFrame) -> dict[str, Any]:
    trade_23 = _num(alpha5_23, "label_action", 0.0) != 0
    trade_18 = _num(alpha5_18, "label_action", 0.0) != 0
    dir_23 = _num(alpha5_23, "direction_valid", 0.0) == 1
    return {
        "alpha5_18_trade_ratio": float(np.mean(trade_18)),
        "alpha5_23_trade_ratio": float(np.mean(trade_23)),
        "alpha5_18_ambiguity_trade_lt_1_5": float(np.mean((_num(alpha5_18, "meta_edge_gap", 0.0) < 1.5)[trade_18])) if np.any(trade_18) else 0.0,
        "alpha5_23_ambiguity_trade_lt_1_5": float(np.mean((_num(alpha5_23, "meta_edge_gap", 0.0) < 1.5)[trade_23])) if np.any(trade_23) else 0.0,
        "alpha5_23_direction_valid_lt_1_5": float(np.mean((_num(alpha5_23, "meta_edge_gap", 0.0) < 1.5)[dir_23])) if np.any(dir_23) else 0.0,
        "alpha5_18_edge_gap_trade_median": float(np.median(_num(alpha5_18, "meta_edge_gap", 0.0)[trade_18])) if np.any(trade_18) else 0.0,
        "alpha5_23_edge_gap_trade_median": float(np.median(_num(alpha5_23, "meta_edge_gap", 0.0)[trade_23])) if np.any(trade_23) else 0.0,
        "alpha5_23_edge_gap_direction_valid_median": float(np.median(_num(alpha5_23, "meta_edge_gap", 0.0)[dir_23])) if np.any(dir_23) else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build direction-refined split labels from alpha5_13 HGB supervised data.")
    p.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    p.add_argument("--compare-dir", type=Path, default=DEFAULT_COMPARE_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_parquet(args.in_dir / "alpha5_13_hgb_atr_barrier_labels_train.parquet")
    val_df = pd.read_parquet(args.in_dir / "alpha5_13_hgb_atr_barrier_labels_val.parquet")
    oos_df = pd.read_parquet(args.in_dir / "alpha5_13_hgb_atr_barrier_labels_oos.parquet")

    print(json.dumps({
        "stage": "start",
        "model_id": MODEL_ID,
        "source": str(args.in_dir),
        "rows": {"train": int(len(train_df)), "validation": int(len(val_df)), "oos": int(len(oos_df))},
        "direction_filters": {
            "bull": {"consensus_min": 1.0, "edge_gap_min": 2.0},
            "bear": {"consensus_min": 1.0, "edge_gap_min": 2.0},
            "chop": {"consensus_min": 0.75, "edge_gap_min": 1.5},
            "whipsaw": "excluded",
        },
    }, ensure_ascii=False, default=_json_default), flush=True)

    train_out = _augment(train_df)
    val_out = _augment(val_df)
    oos_out = _augment(oos_df)

    train_path = args.out_dir / "alpha5_23_direction_refined_train.parquet"
    val_path = args.out_dir / "alpha5_23_direction_refined_val.parquet"
    oos_path = args.out_dir / "alpha5_23_direction_refined_oos.parquet"
    train_out.to_parquet(train_path, index=False)
    val_out.to_parquet(val_path, index=False)
    oos_out.to_parquet(oos_path, index=False)

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "source": str(args.in_dir),
        "train": _split_summary(train_out),
        "validation": _split_summary(val_out),
        "oos": _split_summary(oos_out),
    }
    if args.compare_dir.exists():
        try:
            train_soft = pd.read_parquet(args.compare_dir / "alpha5_18_hgb_soft_labels_train.parquet")
            report["comparison_vs_alpha5_18_train"] = _compare_soft(train_out, train_soft)
        except Exception as exc:  # pragma: no cover
            report["comparison_vs_alpha5_18_train_error"] = str(exc)

    report_path = args.out_dir / "alpha5_23_label_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    summary_csv = args.out_dir / "alpha5_23_label_quality_summary.csv"
    pd.DataFrame([
        {"split": "train", **{k: v for k, v in report["train"].items() if not isinstance(v, (dict, list))}},
        {"split": "validation", **{k: v for k, v in report["validation"].items() if not isinstance(v, (dict, list))}},
        {"split": "oos", **{k: v for k, v in report["oos"].items() if not isinstance(v, (dict, list))}},
    ]).to_csv(summary_csv, index=False)

    print(json.dumps({
        "stage": "complete",
        "train": str(train_path),
        "validation": str(val_path),
        "oos": str(oos_path),
        "report": str(report_path),
        "summary_csv": str(summary_csv),
        "train_summary": {
            "entry_trade_ratio": report["train"]["entry_trade_ratio"],
            "direction_valid_ratio": report["train"]["direction_valid_ratio"],
            "ambiguity_direction_valid": report["train"]["ambiguity_direction_valid"],
            "cash_contamination_rate": report["train"]["cash_contamination_rate"],
            "trade_contamination_rate": report["train"]["trade_contamination_rate"],
        },
    }, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
