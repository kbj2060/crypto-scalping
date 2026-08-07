#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_alpha5_13_hgb_high_quality_training_data_20260518 import (  # noqa: E402
    ATR_WINDOW,
    DEFAULT_2025,
    DEFAULT_2026,
    REGIME_BARRIER,
    REGIME_TRADE_RATIO,
    VOL_Z_COL,
    _align_union,
    _apply_split_policy,
    _atr_pct,
    _class_weights,
    _json_default,
    _label_report,
    _read,
    _regime_name,
    _resolve_barrier_vote,
    _resolve_score_vote,
    _scan_event,
    _uniqueness_weights,
    _verify_state24_sticky090_inputs,
    _vol_override,
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_PREPROCESS_MANIFEST,
)


MODEL_ID = "alpha5_25_hgb_base_labels_20260519"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_25_hgb_base_labels_20260519"


def _label_frame(frame: pd.DataFrame, *, max_hold: int, atr_window: int, fixed_tp_pct: float, fixed_sl_pct: float) -> pd.DataFrame:
    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce").ffill().to_numpy(np.float64)
    high = pd.to_numeric(out["high"], errors="coerce").ffill().to_numpy(np.float64)
    low = pd.to_numeric(out["low"], errors="coerce").ffill().to_numpy(np.float64)
    vol_z = pd.to_numeric(out.get(VOL_Z_COL, 0.0), errors="coerce").fillna(0.0).to_numpy(np.float64)
    atr_pct = _atr_pct(out, window=atr_window)
    regime_name = _regime_name(out)
    n = len(out)

    primary = np.full(n, -1, dtype=np.int16)
    ensemble = np.full(n, -1, dtype=np.int16)
    vote_conf = np.zeros(n, dtype=np.float32)
    consensus = np.zeros(n, dtype=np.float32)
    event_end_idx = np.full(n, -1, dtype=np.int32)
    event_bars = np.full(n, -1, dtype=np.int16)
    tp_pct_used = np.full(n, np.nan, dtype=np.float32)
    sl_pct_used = np.full(n, np.nan, dtype=np.float32)
    meta_ret = np.full(n, np.nan, dtype=np.float32)
    meta_tp_first = np.zeros(n, dtype=np.int8)
    meta_adverse_first = np.zeros(n, dtype=np.int8)
    meta_timeout = np.zeros(n, dtype=np.int8)
    meta_primary_reason = np.full(n, "", dtype=object)
    meta_raw_terminal_ret = np.full(n, np.nan, dtype=np.float32)
    sharpe_vote = np.full(n, -1, dtype=np.int16)
    mfe_vote = np.full(n, -1, dtype=np.int16)
    fixed_vote = np.full(n, -1, dtype=np.int16)
    keep = np.zeros(n, dtype=np.int8)
    long_score = np.full(n, np.nan, dtype=np.float32)
    short_score = np.full(n, np.nan, dtype=np.float32)
    edge_gap = np.full(n, np.nan, dtype=np.float32)

    for i in range(atr_window, n - max_hold - 1):
        entry = float(close[i])
        if not np.isfinite(entry) or entry <= 0:
            continue
        future_close = close[i + 1 : i + 1 + max_hold]
        future_high = high[i + 1 : i + 1 + max_hold]
        future_low = low[i + 1 : i + 1 + max_hold]
        if len(future_close) < max_hold:
            continue

        base_tp_mult, base_sl_mult = REGIME_BARRIER[str(regime_name[i])]
        tp_mult, sl_mult = _vol_override(base_tp_mult, base_sl_mult, float(vol_z[i]))
        tp_pct = float(max(atr_pct[i] * tp_mult, 1e-4))
        sl_pct = float(max(atr_pct[i] * sl_mult, 1e-4))
        tp_pct_used[i] = tp_pct
        sl_pct_used[i] = sl_pct

        path = _scan_event(entry, future_close, future_high, future_low, tp_pct, sl_pct, fixed_tp_pct, fixed_sl_pct)
        pri_vote, pri_reason, pri_bar = _resolve_barrier_vote(
            path["long_primary_tp_bar"],
            path["long_primary_sl_bar"],
            path["short_primary_tp_bar"],
            path["short_primary_sl_bar"],
        )
        fx_vote, _, _ = _resolve_barrier_vote(
            path["long_fixed_tp_bar"],
            path["long_fixed_sl_bar"],
            path["short_fixed_tp_bar"],
            path["short_fixed_sl_bar"],
        )
        sh_vote = _resolve_score_vote(path["long_sharpe"], path["short_sharpe"], threshold=0.20, margin=0.05)
        mf_vote = _resolve_score_vote(path["long_mfe_mae"], path["short_mfe_mae"], threshold=0.0005, margin=0.0002)

        votes = np.asarray([pri_vote, sh_vote, mf_vote], dtype=np.int16)
        counts = np.bincount(np.clip(votes, 0, 2), minlength=3)
        top_count = int(np.max(counts))
        ensemble_vote = int(np.argmax(counts))
        if top_count < 2:
            ensemble_vote = int(pri_vote)

        primary[i] = int(pri_vote)
        fixed_vote[i] = int(fx_vote)
        sharpe_vote[i] = int(sh_vote)
        mfe_vote[i] = int(mf_vote)
        ensemble[i] = int(ensemble_vote)
        consensus[i] = float(top_count / 3.0)
        vote_conf[i] = float(top_count / 3.0)
        keep[i] = int(consensus[i] >= (2.0 / 3.0))

        if pri_vote == 1:
            ret = tp_pct if path["long_primary_tp_bar"] < path["long_primary_sl_bar"] else -sl_pct
            if path["long_primary_tp_bar"] >= 10**9 and path["long_primary_sl_bar"] >= 10**9:
                ret = float(path["long_timeout_ret"])
        elif pri_vote == 2:
            ret = tp_pct if path["short_primary_tp_bar"] < path["short_primary_sl_bar"] else -sl_pct
            if path["short_primary_tp_bar"] >= 10**9 and path["short_primary_sl_bar"] >= 10**9:
                ret = float(path["short_timeout_ret"])
        else:
            ret = 0.0

        if pri_bar <= 0 or pri_bar >= 10**9:
            pri_bar = max_hold
        event_bars[i] = int(pri_bar)
        event_end_idx[i] = int(i + pri_bar)
        meta_ret[i] = float(ret)
        meta_raw_terminal_ret[i] = float(path["raw_terminal_ret"])
        meta_primary_reason[i] = str(pri_reason)
        if "tp_first" in pri_reason:
            meta_tp_first[i] = 1
        elif pri_reason == "adverse_or_timeout":
            if (path["long_primary_sl_bar"] < 10**9) or (path["short_primary_sl_bar"] < 10**9):
                meta_adverse_first[i] = 1
            else:
                meta_timeout[i] = 1
        elif pri_reason == "tie_conflict":
            meta_timeout[i] = 1

        long_score[i] = float(path["long_sharpe"] + path["long_mfe_mae"])
        short_score[i] = float(path["short_sharpe"] + path["short_mfe_mae"])
        edge_gap[i] = float(abs(long_score[i] - short_score[i]))

        if (i % 10000) == 0 and i > 0:
            print(json.dumps({"stage": "label_progress", "done": int(i), "total": int(n)}, ensure_ascii=False), flush=True)

    raw_ensemble = ensemble.copy()
    regime_selected = np.zeros(n, dtype=np.int8)
    for regime, target_trade_ratio in REGIME_TRADE_RATIO.items():
        reg_mask = (regime_name == regime) & (raw_ensemble >= 0)
        reg_idx = np.flatnonzero(reg_mask)
        if len(reg_idx) == 0:
            continue
        cand_idx = reg_idx[raw_ensemble[reg_idx] != 0]
        target_n = int(round(len(reg_idx) * float(target_trade_ratio)))
        target_n = max(0, min(target_n, len(cand_idx)))
        if target_n == 0:
            ensemble[cand_idx] = 0
            continue
        if len(cand_idx) > target_n:
            order = sorted(
                cand_idx.tolist(),
                key=lambda j: (
                    float(consensus[j]),
                    float(edge_gap[j]) if np.isfinite(edge_gap[j]) else -1.0,
                    float(vote_conf[j]),
                ),
                reverse=True,
            )
            selected = np.asarray(order[:target_n], dtype=np.int32)
            dropped = np.asarray(order[target_n:], dtype=np.int32)
            ensemble[dropped] = 0
            regime_selected[selected] = 1
        else:
            regime_selected[cand_idx] = 1

    out["regime4_state"] = regime_name
    out["atr14_pct"] = atr_pct.astype(np.float32)
    out["label_primary_action"] = primary
    out["label_vote_action"] = raw_ensemble
    out["label_fixed_tp05_action"] = fixed_vote
    out["label_sharpe_action"] = sharpe_vote
    out["label_mfe_action"] = mfe_vote
    out["label_action"] = ensemble
    out["label_consensus"] = consensus
    out["label_confidence"] = vote_conf
    out["label_train_keep"] = keep
    out["label_event_end_idx"] = event_end_idx
    out["label_event_bars"] = event_bars
    out["label_tp_pct"] = tp_pct_used
    out["label_sl_pct"] = sl_pct_used
    out["meta_event_return"] = meta_ret
    out["meta_tp_first"] = meta_tp_first
    out["meta_adverse_first"] = meta_adverse_first
    out["meta_timeout"] = meta_timeout
    out["meta_primary_reason"] = meta_primary_reason
    out["meta_raw_terminal_return"] = meta_raw_terminal_ret
    out["meta_long_score"] = long_score
    out["meta_short_score"] = short_score
    out["meta_edge_gap"] = edge_gap
    out["regime_trade_selected"] = regime_selected
    out["meta_is_profitable"] = ((out["label_action"].isin([1, 2])) & (pd.to_numeric(out["meta_event_return"], errors="coerce").fillna(0.0) > 0.0)).astype(np.int8)
    out["meta_tp_ge_005"] = (pd.to_numeric(out["label_tp_pct"], errors="coerce").fillna(0.0) >= 0.005).astype(np.int8)
    out["label_valid"] = ((out["label_action"] >= 0) & (out["label_event_end_idx"] >= 0)).astype(np.int8)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build alpha5_25 base labels with 3-vote ensemble and fx_vote kept for diagnostics only.")
    p.add_argument("--train-2025-csv", type=Path, default=DEFAULT_2025)
    p.add_argument("--oos-2026-csv", type=Path, default=DEFAULT_2026)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--train-end", default="2025-10-01")
    p.add_argument("--val-start", default="2025-10-01")
    p.add_argument("--val-end", default="2026-01-01")
    p.add_argument("--oos-start", default="2026-01-01")
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--atr-window", type=int, default=ATR_WINDOW)
    p.add_argument("--fixed-tp-pct", type=float, default=0.005)
    p.add_argument("--fixed-sl-pct", type=float, default=0.005)
    p.add_argument("--embargo-bars", type=int, default=288)
    p.add_argument("--warmup-bars", type=int, default=288)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_2025 = _read(args.train_2025_csv)
    raw_2026 = _read(args.oos_2026_csv)
    audit = _verify_state24_sticky090_inputs(raw_2025, raw_2026, args.manifest, args.clean4_report)
    a, b, union_cols = _align_union(raw_2025, raw_2026)
    combined = pd.concat([a, b], axis=0, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    print(json.dumps({
        "stage": "start",
        "model_id": MODEL_ID,
        "rows_2025": int(len(raw_2025)),
        "rows_2026": int(len(raw_2026)),
        "union_cols": int(len(union_cols)),
        "audit_expected_model_found": audit.get("expected_model_found_in_manifest"),
        "audit_report_model_path": audit.get("report_model_path"),
        "ensemble_votes": ["pri_vote", "sh_vote", "mf_vote"],
    }, ensure_ascii=False, default=_json_default), flush=True)

    labeled = _label_frame(
        combined,
        max_hold=int(args.max_hold_bars),
        atr_window=int(args.atr_window),
        fixed_tp_pct=float(args.fixed_tp_pct),
        fixed_sl_pct=float(args.fixed_sl_pct),
    )
    labeled["sample_uniqueness_weight"] = _uniqueness_weights(labeled)
    base_weight = pd.to_numeric(labeled["label_confidence"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    uniq = pd.to_numeric(labeled["sample_uniqueness_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    class_w = _class_weights(pd.to_numeric(labeled["label_action"], errors="coerce").fillna(-1).to_numpy(np.int64))
    cw = np.asarray([class_w.get(int(x), 0.0) if int(x) >= 0 else 0.0 for x in pd.to_numeric(labeled["label_action"], errors="coerce").fillna(-1).to_numpy(np.int64)], dtype=np.float64)
    labeled["label_sample_weight"] = (base_weight * uniq * cw).astype(np.float32)

    labeled, split_meta = _apply_split_policy(
        labeled,
        train_end=str(args.train_end),
        val_start=str(args.val_start),
        val_end=str(args.val_end),
        oos_start=str(args.oos_start),
        embargo_bars=int(args.embargo_bars),
        warmup_bars=int(args.warmup_bars),
    )

    report = {
        "model_id": MODEL_ID,
        "config": {
            "train_end": str(args.train_end),
            "val_start": str(args.val_start),
            "val_end": str(args.val_end),
            "oos_start": str(args.oos_start),
            "max_hold_bars": int(args.max_hold_bars),
            "atr_window": int(args.atr_window),
            "fixed_tp_pct": float(args.fixed_tp_pct),
            "fixed_sl_pct": float(args.fixed_sl_pct),
            "embargo_bars": int(args.embargo_bars),
            "warmup_bars": int(args.warmup_bars),
            "regime_barrier": REGIME_BARRIER,
            "ensemble_votes": ["pri_vote", "sh_vote", "mf_vote"],
        },
        "audit": audit,
        "split_meta": split_meta,
        "label_quality": _label_report(labeled),
    }

    train_df = labeled[labeled["dataset_split"] == "train"].reset_index(drop=True)
    val_df = labeled[labeled["dataset_split"] == "validation"].reset_index(drop=True)
    oos_df = labeled[labeled["dataset_split"] == "oos"].reset_index(drop=True)

    train_path = args.out_dir / "alpha5_25_base_labels_train.parquet"
    val_path = args.out_dir / "alpha5_25_base_labels_val.parquet"
    oos_path = args.out_dir / "alpha5_25_base_labels_oos.parquet"
    report_path = args.out_dir / "alpha5_25_label_quality_report.json"
    summary_csv = args.out_dir / "alpha5_25_label_quality_summary.csv"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    oos_df.to_parquet(oos_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame([
        {"split": "train", "rows": len(train_df), "trade_ratio": float(np.mean(pd.to_numeric(train_df["label_action"], errors="coerce").fillna(-1).to_numpy() != 0)), "keep_ratio": float(pd.to_numeric(train_df["label_train_keep"], errors="coerce").fillna(0).mean())},
        {"split": "validation", "rows": len(val_df), "trade_ratio": float(np.mean(pd.to_numeric(val_df["label_action"], errors="coerce").fillna(-1).to_numpy() != 0)), "keep_ratio": float(pd.to_numeric(val_df["label_train_keep"], errors="coerce").fillna(0).mean())},
        {"split": "oos", "rows": len(oos_df), "trade_ratio": float(np.mean(pd.to_numeric(oos_df["label_action"], errors="coerce").fillna(-1).to_numpy() != 0)), "keep_ratio": float(pd.to_numeric(oos_df["label_train_keep"], errors="coerce").fillna(0).mean())},
    ]).to_csv(summary_csv, index=False)

    print(json.dumps({
        "stage": "complete",
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "oos_rows": int(len(oos_df)),
        "train_path": str(train_path),
        "validation_path": str(val_path),
        "oos_path": str(oos_path),
        "report_path": str(report_path),
        "summary_csv": str(summary_csv),
        "label_quality": {
            "trade_ratio": report["label_quality"]["trade_ratio"],
            "consensus_keep_ratio": report["label_quality"]["consensus_keep_ratio"],
            "tp_first_ratio": report["label_quality"]["tp_first_ratio"],
            "adverse_first_ratio": report["label_quality"]["adverse_first_ratio"],
            "ic_raw_terminal_return": report["label_quality"]["ic_raw_terminal_return"],
            "icir_raw_terminal_return": report["label_quality"]["icir_raw_terminal_return"],
        },
    }, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
