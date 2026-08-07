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


MODEL_ID = "alpha5_18_hgb_soft_labels_20260518"
DEFAULT_IN_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_high_quality_training_data_20260518"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_18_hgb_soft_labels_20260518"
SOFT_REGIME_TRADE_RATIO = {
    "bull": 0.62,
    "bear": 0.62,
    "chop": 0.50,
    "whipsaw": 0.30,
}


def _class_weights(labels: np.ndarray) -> dict[int, float]:
    y = labels[(labels >= 0) & (labels <= 2)]
    cnt = np.bincount(y, minlength=3).astype(np.float64)
    total = max(float(cnt.sum()), 1.0)
    return {i: float(total / (3.0 * max(cnt[i], 1.0))) for i in range(3)}


def _soft_relabel(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    raw_vote = pd.to_numeric(out["label_vote_action"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    consensus = pd.to_numeric(out["label_consensus"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    conf = pd.to_numeric(out["label_confidence"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    edge_gap = pd.to_numeric(out["meta_edge_gap"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    tp_first = pd.to_numeric(out["meta_tp_first"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    profitable = (pd.to_numeric(out["meta_event_return"], errors="coerce").fillna(0.0).to_numpy(np.float64) > 0.0).astype(np.int8)
    regime = out["regime4_state"].astype(str).to_numpy()

    soft_action = raw_vote.copy()
    soft_selected = np.zeros(len(out), dtype=np.int8)

    for reg, target_trade_ratio in SOFT_REGIME_TRADE_RATIO.items():
        reg_idx = np.flatnonzero(regime == reg)
        if len(reg_idx) == 0:
            continue
        trade_idx = reg_idx[raw_vote[reg_idx] != 0]
        target_n = int(round(len(reg_idx) * float(target_trade_ratio)))
        target_n = max(0, min(target_n, len(trade_idx)))
        if target_n == 0:
            soft_action[trade_idx] = 0
            continue
        order = sorted(
            trade_idx.tolist(),
            key=lambda j: (
                float(consensus[j]),
                int(tp_first[j]),
                int(profitable[j]),
                float(edge_gap[j]) if np.isfinite(edge_gap[j]) else -1.0,
                float(conf[j]),
            ),
            reverse=True,
        )
        selected = np.asarray(order[:target_n], dtype=np.int32)
        dropped = np.asarray(order[target_n:], dtype=np.int32)
        soft_selected[selected] = 1
        soft_action[dropped] = 0

    keep = (
        (consensus >= 0.75)
        | (
            (raw_vote != 0)
            & (consensus >= 0.50)
            & (tp_first == 1)
            & (profitable == 1)
            & (edge_gap >= 2.0)
        )
    ).astype(np.int8)

    out["label_action"] = soft_action.astype(np.int16)
    out["regime_trade_selected"] = soft_selected.astype(np.int8)
    out["label_train_keep"] = keep.astype(np.int8)
    out["meta_is_profitable"] = ((soft_action != 0) & (profitable == 1)).astype(np.int8)

    uniq = pd.to_numeric(out["sample_uniqueness_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    class_w = _class_weights(pd.to_numeric(out["label_action"], errors="coerce").fillna(-1).to_numpy(np.int64))
    cw = np.asarray(
        [class_w.get(int(x), 0.0) if int(x) >= 0 else 0.0 for x in pd.to_numeric(out["label_action"], errors="coerce").fillna(-1).to_numpy(np.int64)],
        dtype=np.float64,
    )
    base_conf = conf.copy()
    soft_trade = (soft_action != 0) & (base_conf < 0.75)
    base_conf[soft_trade] = np.maximum(base_conf[soft_trade], 0.65)
    out["label_sample_weight"] = (base_conf * uniq * cw).astype(np.float32)
    return out


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    y = pd.to_numeric(frame["label_action"], errors="coerce").fillna(-1).to_numpy(np.int64)
    keep = pd.to_numeric(frame["label_train_keep"], errors="coerce").fillna(0).to_numpy(np.int8)
    return {
        "rows": int(len(frame)),
        "action_counts": {str(int(k)): int(v) for k, v in pd.Series(y).value_counts().sort_index().to_dict().items()},
        "trade_ratio": float(np.mean(y != 0)),
        "keep_ratio": float(np.mean(keep)),
        "trade_ratio_by_regime": {
            str(k): float(v) for k, v in frame.assign(is_trade=(pd.to_numeric(frame["label_action"], errors="coerce").fillna(-1) != 0).astype(float)).groupby("regime4_state")["is_trade"].mean().to_dict().items()
        },
        "tp_first_ratio": float(pd.to_numeric(frame["meta_tp_first"], errors="coerce").fillna(0).mean()),
        "profitable_ratio": float(pd.to_numeric(frame["meta_is_profitable"], errors="coerce").fillna(0).mean()),
        "consensus_mean": float(pd.to_numeric(frame["label_consensus"], errors="coerce").fillna(0.0).mean()),
        "weight_mean": float(pd.to_numeric(frame["label_sample_weight"], errors="coerce").fillna(0.0).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build slightly easier soft-label variant from alpha5_13 HGB supervised data.")
    p.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
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
        "soft_regime_trade_ratio": SOFT_REGIME_TRADE_RATIO,
        "soft_keep_logic": "consensus>=0.75 OR (trade & consensus>=0.50 & tp_first & profitable & edge_gap>=2.0)",
    }, ensure_ascii=False, default=_json_default), flush=True)

    train_soft = _soft_relabel(train_df)
    val_soft = _soft_relabel(val_df)
    oos_soft = _soft_relabel(oos_df)

    train_path = args.out_dir / "alpha5_18_hgb_soft_labels_train.parquet"
    val_path = args.out_dir / "alpha5_18_hgb_soft_labels_val.parquet"
    oos_path = args.out_dir / "alpha5_18_hgb_soft_labels_oos.parquet"
    train_soft.to_parquet(train_path, index=False)
    val_soft.to_parquet(val_path, index=False)
    oos_soft.to_parquet(oos_path, index=False)

    report = {
        "model_id": MODEL_ID,
        "soft_regime_trade_ratio": SOFT_REGIME_TRADE_RATIO,
        "soft_keep_logic": "consensus>=0.75 OR (trade & consensus>=0.50 & tp_first & profitable & edge_gap>=2.0)",
        "train": _summary(train_soft),
        "validation": _summary(val_soft),
        "oos": _summary(oos_soft),
    }
    report_path = args.out_dir / "alpha5_18_hgb_soft_labels_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    summary_csv = args.out_dir / "alpha5_18_hgb_soft_labels_summary.csv"
    pd.DataFrame([
        {"split": "train", **report["train"]},
        {"split": "validation", **report["validation"]},
        {"split": "oos", **report["oos"]},
    ]).to_csv(summary_csv, index=False)

    print(json.dumps({
        "stage": "complete",
        "train": str(train_path),
        "validation": str(val_path),
        "oos": str(oos_path),
        "report": str(report_path),
        "summary_csv": str(summary_csv),
        "train_summary": report["train"],
        "validation_summary": report["validation"],
        "oos_summary": report["oos"],
    }, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
