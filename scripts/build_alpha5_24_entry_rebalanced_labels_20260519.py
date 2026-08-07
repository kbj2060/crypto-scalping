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

from scripts.build_alpha5_23_direction_refined_labels_20260519 import (  # noqa: E402
    DEFAULT_COMPARE_DIR,
    REGIME_AMBIGUITY_MIN,
    REGIME_CONSENSUS_MIN,
    _amb_rates,
    _class_weights,
    _compare_soft,
    _json_default,
    _num,
    _regime_direction_rows,
    _regime_purity,
)


MODEL_ID = "alpha5_24_entry_rebalanced_labels_20260519"
DEFAULT_IN_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_high_quality_training_data_20260518"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_24_entry_rebalanced_labels_20260519"


def _augment(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    action = _num(out, "label_action", 0.0).astype(np.int64)
    keep = _num(out, "label_train_keep", 0.0).astype(np.int8)
    consensus = _num(out, "label_consensus", 0.0)
    base_weight = _num(out, "label_sample_weight", 0.0)
    edge_gap = _num(out, "meta_edge_gap", 0.0)
    tp_first = _num(out, "meta_tp_first", 0.0).astype(np.int8)
    profitable = _num(out, "meta_is_profitable", 0.0).astype(np.int8)
    event_ret = _num(out, "meta_event_return", 0.0)
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
    clean_cash_keep = (keep == 1) & cash_mask & (~cash_contam)
    revived_cash_keep = cash_contam
    good_trade_keep = (keep == 1) & trade_mask & (~trade_contam)
    entry_keep = (clean_cash_keep | revived_cash_keep | good_trade_keep).astype(np.int8)

    entry_weight = np.clip(base_weight, 1e-4, None)
    entry_weight *= (0.90 + 0.20 * np.clip(consensus, 0.0, 1.0))
    entry_weight *= np.where(cash_mask, 1.00, 1.10 + 0.05 * profitable + 0.05 * tp_first)
    entry_weight *= np.where(cash_contam, 0.18, 1.0)
    entry_weight *= np.where(trade_contam, 0.0, 1.0)
    entry_weight *= np.where(edge_gap >= 2.0, 1.05, np.where(edge_gap >= 1.5, 1.0, 0.95))
    entry_weight = entry_weight.astype(np.float32)

    direction_label = np.where(direction_valid == 1, action, 0).astype(np.int8)
    direction_keep = direction_valid.astype(np.int8)
    direction_weight = np.clip(base_weight, 1e-4, None)
    direction_weight *= (0.80 + 0.40 * np.clip(consensus, 0.0, 1.0))
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


def _split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    action = _num(frame, "label_action", 0.0).astype(np.int64)
    entry = _num(frame, "entry_label", 0.0).astype(np.int64)
    entry_keep = _num(frame, "entry_train_keep", 0.0).astype(np.int64) == 1
    direction_valid = _num(frame, "direction_valid", 0.0).astype(np.int64) == 1
    cash_mask = action == 0
    trade_mask = action != 0
    cash_contam = _num(frame, "cash_contamination_flag", 0.0).astype(np.int64) == 1
    trade_contam = _num(frame, "trade_contamination_flag", 0.0).astype(np.int64) == 1
    kept_trade = entry_keep & (entry == 1)
    out = {
        "rows": int(len(frame)),
        "action_counts": {str(int(k)): int(v) for k, v in pd.Series(action).value_counts().sort_index().to_dict().items()},
        "entry_trade_ratio": float(np.mean(entry == 1)),
        "entry_keep_ratio": float(np.mean(entry_keep)),
        "entry_keep_trade_ratio": float(np.mean(entry[entry_keep] == 1)) if np.any(entry_keep) else 0.0,
        "direction_valid_ratio": float(np.mean(direction_valid)),
        "direction_valid_rows_by_regime": _regime_direction_rows(frame),
        "ambiguity_all": _amb_rates(frame),
        "ambiguity_direction_valid": _amb_rates(frame, direction_valid),
        "cash_contamination_rate": float(np.mean(cash_contam[cash_mask])) if np.any(cash_mask) else 0.0,
        "trade_contamination_rate": float(np.mean(trade_contam[trade_mask])) if np.any(trade_mask) else 0.0,
        "kept_trade_event_ret_mean": float(np.mean(_num(frame, "meta_event_return", 0.0)[kept_trade])) if np.any(kept_trade) else 0.0,
        "regime_purity": _regime_purity(frame),
        "whipsaw_direction_valid_rows": int(np.sum((frame["regime4_state"].astype(str) == "whipsaw") & direction_valid)),
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build alpha5_24 labels with entry rebalance and strong direction purity.")
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
        "entry_policy": "keep clean cash from original keep, revive contaminated cash as low-weight negatives, keep only non-contaminated trades",
    }, ensure_ascii=False, default=_json_default), flush=True)

    train_out = _augment(train_df)
    val_out = _augment(val_df)
    oos_out = _augment(oos_df)

    train_path = args.out_dir / "alpha5_24_entry_rebalanced_train.parquet"
    val_path = args.out_dir / "alpha5_24_entry_rebalanced_val.parquet"
    oos_path = args.out_dir / "alpha5_24_entry_rebalanced_oos.parquet"
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
        except Exception as exc:
            report["comparison_vs_alpha5_18_train_error"] = str(exc)

    report_path = args.out_dir / "alpha5_24_label_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    summary_csv = args.out_dir / "alpha5_24_label_quality_summary.csv"
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
        "train_summary": report["train"],
    }, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
