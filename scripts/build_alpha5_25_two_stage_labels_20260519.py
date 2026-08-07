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
    _amb_rates,
    _class_weights,
    _json_default,
    _num,
    _regime_direction_rows,
    _regime_purity,
)


MODEL_ID = "alpha5_25_two_stage_labels_20260519"
DEFAULT_IN_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_25_hgb_base_labels_20260519"
DEFAULT_COMPARE_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_24_entry_rebalanced_labels_20260519"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_25_two_stage_labels_20260519"

REGIME_AMBIGUITY_MIN = {
    "bull": 2.0,
    "bear": 2.0,
    "chop": 1.5,
    "whipsaw": 9e9,
}
REGIME_CONSENSUS_MIN = {
    "bull": 2.0 / 3.0,
    "bear": 2.0 / 3.0,
    "chop": 2.0 / 3.0,
    "whipsaw": 9e9,
}


def _entry_negative_rank(frame: pd.DataFrame) -> pd.Series:
    uniq = pd.to_numeric(frame["sample_uniqueness_weight"], errors="coerce").fillna(0.0)
    conf = pd.to_numeric(frame["label_confidence"], errors="coerce").fillna(0.0)
    raw_abs = pd.to_numeric(frame["meta_raw_terminal_return"], errors="coerce").fillna(0.0).abs()
    best_score = np.maximum(_num(frame, "meta_long_score", 0.0), _num(frame, "meta_short_score", 0.0))
    return pd.Series(
        list(zip(-uniq.to_numpy(), -conf.to_numpy(), raw_abs.to_numpy(), best_score.tolist())),
        index=frame.index,
        dtype=object,
    )


def _entry_keep_mask(out: pd.DataFrame, positive_keep: np.ndarray, clean_cash_keep: np.ndarray) -> np.ndarray:
    keep = np.zeros(len(out), dtype=np.int8)
    split = out["dataset_split"].astype(str).to_numpy()
    keep[positive_keep | clean_cash_keep] = 1

    train_mask = split == "train"
    pos_idx = np.flatnonzero(train_mask & positive_keep)
    neg_idx = np.flatnonzero(train_mask & clean_cash_keep)
    if len(pos_idx) > 0 and len(neg_idx) > 0:
        target = min(len(pos_idx), len(neg_idx))
        pos_ranked = out.iloc[pos_idx].copy()
        pos_ranked["__rank"] = list(
            zip(
                -pd.to_numeric(pos_ranked["label_confidence"], errors="coerce").fillna(0.0).to_numpy(),
                -pd.to_numeric(pos_ranked["meta_edge_gap"], errors="coerce").fillna(0.0).to_numpy(),
                -pd.to_numeric(pos_ranked["sample_uniqueness_weight"], errors="coerce").fillna(0.0).to_numpy(),
            )
        )
        pos_ranked = pos_ranked.sort_values("__rank", kind="mergesort")
        neg_ranked = out.iloc[neg_idx].copy()
        neg_ranked["__rank"] = _entry_negative_rank(neg_ranked)
        neg_ranked = neg_ranked.sort_values("__rank", kind="mergesort")
        keep[pos_idx] = 0
        keep[neg_idx] = 0
        keep[pos_ranked.index[:target].to_numpy(np.int64)] = 1
        keep[neg_ranked.index[:target].to_numpy(np.int64)] = 1
    return keep


def _augment(frame: pd.DataFrame, *, entry_event_ret_min: float) -> pd.DataFrame:
    out = frame.copy()
    action = _num(out, "label_action", 0.0).astype(np.int64)
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
    clean_cash = cash_mask & (best_score < 1.0) & (event_ret < 0.005) & (np.abs(raw_ret) < 0.005)
    ambiguous_cash = cash_mask & (~clean_cash)
    trade_contam = trade_mask & ((profitable != 1) | (tp_first != 1) | (event_ret < float(entry_event_ret_min)))

    positive_keep = trade_mask & (tp_first == 1) & (profitable == 1) & (event_ret >= float(entry_event_ret_min)) & (selected == 1) & (regime != "whipsaw")
    entry_label = trade_mask.astype(np.int8)
    entry_keep = _entry_keep_mask(out, positive_keep, clean_cash).astype(np.int8)

    direction_amb = np.zeros(len(out), dtype=np.int8)
    direction_valid = np.zeros(len(out), dtype=np.int8)
    for reg in ("bull", "bear", "chop", "whipsaw"):
        reg_mask = regime == reg
        amb_min = float(REGIME_AMBIGUITY_MIN[reg])
        con_min = float(REGIME_CONSENSUS_MIN[reg])
        if reg == "whipsaw":
            direction_amb[reg_mask] = 1
            continue
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

    edge_gap_clip = np.clip(edge_gap, 0.0, 8.0)
    entry_weight = np.clip(base_weight, 1e-4, None)
    entry_weight *= np.where(entry_label == 1, 1.0 + 0.10 * np.clip(consensus, 0.0, 1.0), 1.0 + 0.05 * np.clip(uniq, 0.0, 1.0))
    entry_weight *= np.where(entry_label == 1, 1.0 + 0.05 * edge_gap_clip, 1.0)
    entry_weight *= np.where(entry_keep == 1, 1.0, 0.0)
    entry_weight = entry_weight.astype(np.float32)

    direction_label = np.where(direction_valid == 1, action, 0).astype(np.int8)
    direction_weight = np.clip(base_weight, 1e-4, None)
    direction_weight *= (0.80 + 0.40 * np.clip(consensus, 0.0, 1.0))
    direction_weight *= (1.0 + 0.08 * edge_gap_clip)
    direction_weight *= (1.0 + 0.15 * tp_first + 0.10 * profitable)
    direction_weight *= np.where(direction_valid == 1, 1.0, 0.0)
    direction_weight = direction_weight.astype(np.float32)

    entry_cw = _class_weights(entry_label, entry_keep == 1)
    if entry_cw:
        entry_weight *= np.asarray([entry_cw.get(int(y), 0.0) for y in entry_label], dtype=np.float32)
    direction_cw = _class_weights(direction_label, direction_valid == 1)
    if direction_cw:
        direction_weight *= np.asarray([direction_cw.get(int(y), 0.0) for y in direction_label], dtype=np.float32)

    out["cash_contamination_flag"] = cash_contam.astype(np.int8)
    out["trade_contamination_flag"] = trade_contam.astype(np.int8)
    out["entry_ambiguous_cash_flag"] = ambiguous_cash.astype(np.int8)
    out["direction_ambiguity_flag"] = direction_amb.astype(np.int8)
    out["entry_label"] = entry_label.astype(np.int8)
    out["entry_train_keep"] = entry_keep.astype(np.int8)
    out["entry_sample_weight"] = entry_weight
    out["direction_label"] = direction_label.astype(np.int8)
    out["direction_valid"] = direction_valid.astype(np.int8)
    out["direction_train_keep"] = direction_valid.astype(np.int8)
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
    ambiguous_cash = _num(frame, "entry_ambiguous_cash_flag", 0.0).astype(np.int64) == 1
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
        "ambiguous_cash_rate": float(np.mean(ambiguous_cash[cash_mask])) if np.any(cash_mask) else 0.0,
        "kept_trade_event_ret_mean": float(np.mean(_num(frame, "meta_event_return", 0.0)[kept_trade])) if np.any(kept_trade) else 0.0,
        "regime_purity": _regime_purity(frame),
        "whipsaw_direction_valid_rows": int(np.sum((frame["regime4_state"].astype(str) == "whipsaw") & direction_valid)),
    }
    return out


def _compare_prev(alpha5_25: pd.DataFrame, alpha5_24: pd.DataFrame) -> dict[str, Any]:
    dir_25 = _num(alpha5_25, "direction_valid", 0.0) == 1
    dir_24 = _num(alpha5_24, "direction_valid", 0.0) == 1
    return {
        "alpha5_24_direction_valid_ratio": float(np.mean(dir_24)),
        "alpha5_25_direction_valid_ratio": float(np.mean(dir_25)),
        "alpha5_24_direction_valid_lt_1_5": float(np.mean((_num(alpha5_24, "meta_edge_gap", 0.0) < 1.5)[dir_24])) if np.any(dir_24) else 0.0,
        "alpha5_25_direction_valid_lt_1_5": float(np.mean((_num(alpha5_25, "meta_edge_gap", 0.0) < 1.5)[dir_25])) if np.any(dir_25) else 0.0,
        "alpha5_24_direction_valid_edge_gap_median": float(np.median(_num(alpha5_24, "meta_edge_gap", 0.0)[dir_24])) if np.any(dir_24) else 0.0,
        "alpha5_25_direction_valid_edge_gap_median": float(np.median(_num(alpha5_25, "meta_edge_gap", 0.0)[dir_25])) if np.any(dir_25) else 0.0,
        "alpha5_24_cash_contamination_rate": float(np.mean(_num(alpha5_24, "cash_contamination_flag", 0.0) == 1)),
        "alpha5_25_cash_contamination_rate": float(np.mean(_num(alpha5_25, "cash_contamination_flag", 0.0) == 1)),
        "alpha5_24_trade_contamination_rate": float(np.mean(_num(alpha5_24, "trade_contamination_flag", 0.0) == 1)),
        "alpha5_25_trade_contamination_rate": float(np.mean(_num(alpha5_25, "trade_contamination_flag", 0.0) == 1)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build alpha5_25 two-stage labels with clean entry negatives and relaxed direction purity.")
    p.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    p.add_argument("--compare-dir", type=Path, default=DEFAULT_COMPARE_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--entry-event-ret-min", type=float, default=0.005)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_parquet(args.in_dir / "alpha5_25_base_labels_train.parquet")
    val_df = pd.read_parquet(args.in_dir / "alpha5_25_base_labels_val.parquet")
    oos_df = pd.read_parquet(args.in_dir / "alpha5_25_base_labels_oos.parquet")

    print(json.dumps({
        "stage": "start",
        "model_id": MODEL_ID,
        "source": str(args.in_dir),
        "entry_policy": "good trade positives + clean cash negatives, ambiguous cash excluded, train negatives downsampled to near 50/50",
        "entry_event_ret_min": float(args.entry_event_ret_min),
        "direction_policy": {
            "bull_bear_consensus_min": REGIME_CONSENSUS_MIN["bull"],
            "bull_bear_edge_gap_min": 2.0,
            "chop_consensus_min": REGIME_CONSENSUS_MIN["chop"],
            "chop_edge_gap_min": 1.5,
            "whipsaw": "exclude",
        },
    }, ensure_ascii=False, default=_json_default), flush=True)

    train_out = _augment(train_df, entry_event_ret_min=float(args.entry_event_ret_min))
    val_out = _augment(val_df, entry_event_ret_min=float(args.entry_event_ret_min))
    oos_out = _augment(oos_df, entry_event_ret_min=float(args.entry_event_ret_min))

    train_path = args.out_dir / "alpha5_25_two_stage_labels_train.parquet"
    val_path = args.out_dir / "alpha5_25_two_stage_labels_val.parquet"
    oos_path = args.out_dir / "alpha5_25_two_stage_labels_oos.parquet"
    train_out.to_parquet(train_path, index=False)
    val_out.to_parquet(val_path, index=False)
    oos_out.to_parquet(oos_path, index=False)

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "source": str(args.in_dir),
        "entry_event_ret_min": float(args.entry_event_ret_min),
        "train": _split_summary(train_out),
        "validation": _split_summary(val_out),
        "oos": _split_summary(oos_out),
    }
    if args.compare_dir.exists():
        try:
            prev = pd.read_parquet(args.compare_dir / "alpha5_24_entry_rebalanced_train.parquet")
            report["comparison_vs_alpha5_24_train"] = _compare_prev(train_out, prev)
        except Exception as exc:
            report["comparison_vs_alpha5_24_train_error"] = str(exc)

    report_path = args.out_dir / "alpha5_25_label_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    summary_csv = args.out_dir / "alpha5_25_label_quality_summary.csv"
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
