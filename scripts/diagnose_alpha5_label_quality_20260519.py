#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALPHA5_13 = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_high_quality_training_data_20260518"
DEFAULT_ALPHA5_18 = ROOT / "tmp/causal_regen_20260516/alpha5_18_hgb_soft_labels_20260518"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_label_quality_diagnostics_20260519"


DIAG_COLS = [
    "label_action",
    "label_train_keep",
    "label_consensus",
    "label_confidence",
    "regime4_state",
    "meta_tp_first",
    "meta_is_profitable",
    "meta_event_return",
    "meta_raw_terminal_return",
    "meta_long_score",
    "meta_short_score",
    "meta_edge_gap",
]


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _load_split(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, columns=DIAG_COLS)


def _direction_ambiguity(frame: pd.DataFrame) -> dict[str, Any]:
    action = _safe_num(frame["label_action"]).fillna(-1).astype(int)
    trade = action != 0
    edge_gap = _safe_num(frame["meta_edge_gap"]).fillna(0.0)
    long_score = _safe_num(frame["meta_long_score"]).fillna(0.0)
    short_score = _safe_num(frame["meta_short_score"]).fillna(0.0)
    signed_gap = (long_score - short_score).abs()
    thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]
    out: dict[str, Any] = {
        "trade_rows": int(trade.sum()),
        "trade_ratio": float(trade.mean()),
        "edge_gap_quantiles": {
            str(q): float(edge_gap.quantile(q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
        },
        "signed_gap_quantiles": {
            str(q): float(signed_gap.quantile(q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
        },
        "ambiguity_rate_by_threshold": {},
        "ambiguity_by_regime": {},
    }
    trade_edge = edge_gap[trade]
    for t in thresholds:
        out["ambiguity_rate_by_threshold"][str(t)] = float((trade_edge < float(t)).mean()) if len(trade_edge) else 0.0
    for regime, g in frame.assign(trade=trade.astype(int), edge_gap=edge_gap, signed_gap=signed_gap).groupby("regime4_state"):
        gm = g["trade"] == 1
        out["ambiguity_by_regime"][str(regime)] = {
            "trade_rows": int(gm.sum()),
            "trade_ratio": float(gm.mean()) if len(g) else 0.0,
            "edge_gap_p50": float(g.loc[gm, "edge_gap"].median()) if gm.any() else 0.0,
            "edge_gap_p90": float(g.loc[gm, "edge_gap"].quantile(0.9)) if gm.any() else 0.0,
            "signed_gap_p50": float(g.loc[gm, "signed_gap"].median()) if gm.any() else 0.0,
            "amb_lt_1.0": float((g.loc[gm, "edge_gap"] < 1.0).mean()) if gm.any() else 0.0,
            "amb_lt_1.5": float((g.loc[gm, "edge_gap"] < 1.5).mean()) if gm.any() else 0.0,
            "amb_lt_2.0": float((g.loc[gm, "edge_gap"] < 2.0).mean()) if gm.any() else 0.0,
        }
    return out


def _regime_purity(frame: pd.DataFrame) -> dict[str, Any]:
    action = _safe_num(frame["label_action"]).fillna(-1).astype(int)
    event_ret = _safe_num(frame["meta_event_return"]).fillna(0.0)
    raw_ret = _safe_num(frame["meta_raw_terminal_return"]).fillna(0.0)
    tp_first = _safe_num(frame["meta_tp_first"]).fillna(0.0)
    profitable = _safe_num(frame["meta_is_profitable"]).fillna(0.0)
    out: dict[str, Any] = {}
    for regime, g in frame.assign(action=action, event_ret=event_ret, raw_ret=raw_ret, tp_first=tp_first, profitable=profitable).groupby("regime4_state"):
        regime_out: dict[str, Any] = {}
        for a, name in ((0, "cash"), (1, "long"), (2, "short")):
            m = g["action"] == a
            if not m.any():
                regime_out[name] = {"rows": 0}
                continue
            regime_out[name] = {
                "rows": int(m.sum()),
                "ratio": float(m.mean()),
                "event_ret_mean": float(g.loc[m, "event_ret"].mean()),
                "event_ret_median": float(g.loc[m, "event_ret"].median()),
                "raw_ret_mean": float(g.loc[m, "raw_ret"].mean()),
                "raw_ret_median": float(g.loc[m, "raw_ret"].median()),
                "tp_first_ratio": float(g.loc[m, "tp_first"].mean()),
                "profitable_ratio": float(g.loc[m, "profitable"].mean()),
            }
        out[str(regime)] = regime_out
    return out


def _cash_contamination(frame: pd.DataFrame) -> dict[str, Any]:
    action = _safe_num(frame["label_action"]).fillna(-1).astype(int)
    cash = action == 0
    long_score = _safe_num(frame["meta_long_score"]).fillna(0.0)
    short_score = _safe_num(frame["meta_short_score"]).fillna(0.0)
    best_score = np.maximum(long_score, short_score)
    edge_gap = _safe_num(frame["meta_edge_gap"]).fillna(0.0)
    event_ret = _safe_num(frame["meta_event_return"]).fillna(0.0)
    raw_ret = _safe_num(frame["meta_raw_terminal_return"]).fillna(0.0).abs()
    thresholds = {
        "best_score_ge_1.0": best_score >= 1.0,
        "best_score_ge_2.0": best_score >= 2.0,
        "edge_gap_ge_1.5": edge_gap >= 1.5,
        "event_ret_ge_0.005": event_ret >= 0.005,
        "raw_ret_abs_ge_0.005": raw_ret >= 0.005,
    }
    out = {"cash_rows": int(cash.sum()), "rates": {}}
    for name, mask in thresholds.items():
        out["rates"][name] = float(mask[cash].mean()) if cash.any() else 0.0
    return out


def _trade_contamination(frame: pd.DataFrame) -> dict[str, Any]:
    action = _safe_num(frame["label_action"]).fillna(-1).astype(int)
    trade = action != 0
    tp_first = _safe_num(frame["meta_tp_first"]).fillna(0.0)
    profitable = _safe_num(frame["meta_is_profitable"]).fillna(0.0)
    event_ret = _safe_num(frame["meta_event_return"]).fillna(0.0)
    raw_ret = _safe_num(frame["meta_raw_terminal_return"]).fillna(0.0).abs()
    thresholds = {
        "tp_first_eq_0": tp_first <= 0.0,
        "profitable_eq_0": profitable <= 0.0,
        "event_ret_lt_0.003": event_ret < 0.003,
        "event_ret_lt_0.005": event_ret < 0.005,
        "raw_ret_abs_lt_0.003": raw_ret < 0.003,
        "raw_ret_abs_lt_0.005": raw_ret < 0.005,
    }
    out = {"trade_rows": int(trade.sum()), "rates": {}}
    for name, mask in thresholds.items():
        out["rates"][name] = float(mask[trade].mean()) if trade.any() else 0.0
    return out


def _label_summary(frame: pd.DataFrame) -> dict[str, Any]:
    action = _safe_num(frame["label_action"]).fillna(-1).astype(int)
    keep = _safe_num(frame["label_train_keep"]).fillna(0.0)
    return {
        "rows": int(len(frame)),
        "action_counts": {str(int(k)): int(v) for k, v in action.value_counts().sort_index().to_dict().items()},
        "trade_ratio": float((action != 0).mean()),
        "keep_ratio": float(keep.mean()),
        "consensus_mean": float(_safe_num(frame["label_consensus"]).fillna(0.0).mean()),
        "confidence_mean": float(_safe_num(frame["label_confidence"]).fillna(0.0).mean()),
    }


def _diagnose_dataset(name: str, root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"dataset": name, "root": str(root), "splits": {}}
    for split, fname in (
        ("train", root / f"{name}_train.parquet"),
        ("validation", root / f"{name}_val.parquet"),
        ("oos", root / f"{name}_oos.parquet"),
    ):
        frame = _load_split(fname)
        out["splits"][split] = {
            "summary": _label_summary(frame),
            "direction_ambiguity": _direction_ambiguity(frame),
            "regime_purity": _regime_purity(frame),
            "cash_contamination": _cash_contamination(frame),
            "trade_contamination": _trade_contamination(frame),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose Alpha5 label quality.")
    p.add_argument("--alpha5-13-dir", type=Path, default=DEFAULT_ALPHA5_13)
    p.add_argument("--alpha5-18-dir", type=Path, default=DEFAULT_ALPHA5_18)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    alpha5_13 = _diagnose_dataset("alpha5_13_hgb_atr_barrier_labels", args.alpha5_13_dir)
    alpha5_18 = _diagnose_dataset("alpha5_18_hgb_soft_labels", args.alpha5_18_dir)
    summary = {
        "model_id": "alpha5_label_quality_diagnostics_20260519",
        "datasets": {
            "alpha5_13": alpha5_13,
            "alpha5_18": alpha5_18,
        },
    }
    out_path = args.out_dir / "alpha5_label_quality_diagnostics.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "stage": "complete",
        "out": str(out_path),
        "alpha5_13_train_trade_ratio": alpha5_13["splits"]["train"]["summary"]["trade_ratio"],
        "alpha5_18_train_trade_ratio": alpha5_18["splits"]["train"]["summary"]["trade_ratio"],
        "alpha5_18_train_amb_lt_1.5": alpha5_18["splits"]["train"]["direction_ambiguity"]["ambiguity_rate_by_threshold"]["1.5"],
        "alpha5_18_train_cash_best_score_ge_1.0": alpha5_18["splits"]["train"]["cash_contamination"]["rates"]["best_score_ge_1.0"],
        "alpha5_18_train_trade_profitable_eq_0": alpha5_18["splits"]["train"]["trade_contamination"]["rates"]["profitable_eq_0"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
