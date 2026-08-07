#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import prepare_features  # noqa: E402
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_PREPROCESS_MANIFEST,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_alpha5_8_hgb_action_feature_contract_compare_20260518 import _alpha4_mapped_features  # noqa: E402
from scripts.train_eval_alpha5_13_hgb_single_20260518 import _backtest_barrier, _direction_metrics  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402
from scripts.tune_alpha5_9_hgb_action_master_20260518 import HGBSpec, _fit_hgb, _hgb_specs  # noqa: E402


MODEL_ID = "alpha5_14_hgb_hierarchical_core_20260518"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_high_quality_training_data_20260518"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_14_hgb_hierarchical_core_20260518"


def _x(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return prepare_features(frame, side_hint=0, close=_close(frame), feature_cols=cols)


def _feature_cols(train_raw: pd.DataFrame, eval_raw: pd.DataFrame, available: set[str]) -> list[str]:
    cols = _alpha4_mapped_features(train_raw, eval_raw, include_future=False)
    return [c for c in cols if c in available]


def _binary_proba(model: Any, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        idx = classes.index(1)
        return raw[:, idx]
    if raw.shape[1] == 2:
        return raw[:, 1]
    return raw[:, 0]


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    out = np.ones(len(y), dtype=np.float64)
    classes, counts = np.unique(y, return_counts=True)
    total = max(float(len(y)), 1.0)
    for cls, count in zip(classes, counts):
        out[y == int(cls)] = total / (float(len(classes)) * max(float(count), 1.0))
    return out


def _gate_target(frame: pd.DataFrame) -> np.ndarray:
    return (pd.to_numeric(frame["label_action"], errors="coerce").fillna(0).to_numpy(np.int64) != 0).astype(np.int64)


def _side_target(frame: pd.DataFrame) -> np.ndarray:
    return (pd.to_numeric(frame["label_action"], errors="coerce").fillna(0).to_numpy(np.int64) == 1).astype(np.int64)


def _quality_target(frame: pd.DataFrame) -> np.ndarray:
    tp_first = pd.to_numeric(frame["meta_tp_first"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
    profitable = pd.to_numeric(frame["meta_is_profitable"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
    return (tp_first & profitable).astype(np.int64)


def _gate_weights(frame: pd.DataFrame) -> np.ndarray:
    y = _gate_target(frame)
    base = pd.to_numeric(frame["label_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    consensus = pd.to_numeric(frame["label_consensus"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    w = np.clip(base, 1e-4, None) * (0.75 + 0.50 * np.clip(consensus, 0.0, 1.0))
    return w * _balanced_weights(y)


def _side_weights(frame: pd.DataFrame) -> np.ndarray:
    y = _side_target(frame)
    base = pd.to_numeric(frame["label_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    tp_first = pd.to_numeric(frame["meta_tp_first"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    edge_gap = pd.to_numeric(frame["meta_edge_gap"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    w = np.clip(base, 1e-4, None) * (1.0 + 0.20 * tp_first) * (1.0 + 0.02 * np.clip(edge_gap, 0.0, 10.0))
    return w * _balanced_weights(y)


def _quality_weights(frame: pd.DataFrame) -> np.ndarray:
    y = _quality_target(frame)
    base = pd.to_numeric(frame["label_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    edge_gap = pd.to_numeric(frame["meta_edge_gap"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    profitable = pd.to_numeric(frame["meta_is_profitable"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    w = np.clip(base, 1e-4, None) * (1.0 + 0.10 * profitable) * (1.0 + 0.03 * np.clip(edge_gap, 0.0, 10.0))
    return w * _balanced_weights(y)


def _make_stage_specs() -> list[dict[str, HGBSpec]]:
    specs = {spec.name: spec for spec in _hgb_specs()}
    return [
        {"name": "hier_v1", "gate": specs["regularized"], "side": specs["deeper"], "quality": specs["regularized"]},
        {"name": "hier_v2", "gate": specs["deeper"], "side": specs["deeper"], "quality": specs["regularized"]},
        {"name": "hier_v3", "gate": specs["regularized"], "side": specs["deeper"], "quality": specs["deeper"]},
    ]


def _compose_actions(
    gate_proba: np.ndarray,
    side_long_proba: np.ndarray,
    quality_proba: np.ndarray,
    *,
    trade_threshold: float,
    quality_threshold: float,
    margin_threshold: float,
    trade_score_threshold: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    p_trade = np.clip(gate_proba, 0.0, 1.0)
    p_quality = np.clip(quality_proba, 0.0, 1.0)
    p_long = np.clip(side_long_proba, 0.0, 1.0)
    p_short = 1.0 - p_long
    side_margin = np.abs(p_long - p_short)
    trade_score = p_trade * (0.55 + 0.45 * p_quality)
    long_score = trade_score * p_long
    short_score = trade_score * p_short
    actions = np.where(p_long >= p_short, 1, 2).astype(np.int64)
    actions = np.where(p_trade < float(trade_threshold), 0, actions)
    actions = np.where(p_quality < float(quality_threshold), 0, actions)
    actions = np.where(side_margin < float(margin_threshold), 0, actions)
    actions = np.where(trade_score < float(trade_score_threshold), 0, actions)
    return actions, {
        "p_trade": p_trade,
        "p_quality": p_quality,
        "p_long": p_long,
        "p_short": p_short,
        "side_margin": side_margin,
        "trade_score": trade_score,
        "long_score": long_score,
        "short_score": short_score,
    }


def _eval(frame: pd.DataFrame, actions: np.ndarray, labels: np.ndarray, *, fee: float, slip: float, exposure: float, max_hold: int) -> dict[str, Any]:
    bt = {
        f"cost{m}": _backtest_barrier(
            frame,
            actions,
            fee=float(fee) * float(m),
            slip=float(slip) * float(m),
            unit_exposure=float(exposure),
            max_hold_bars=int(max_hold),
        )
        for m in (1, 2, 3)
    }
    dm = _direction_metrics(actions, labels)
    c1, c2, c3 = bt["cost1"], bt["cost2"], bt["cost3"]
    if int(c1["trades"]) < 20:
        score = -1e6 + float(c1["pnl"])
    else:
        score = (
            float(c1["pnl"])
            + 0.50 * float(c2["pnl"])
            + 0.20 * float(c3["pnl"])
            + 14.0 * float(dm["balanced_trade_precision"])
            + 8.0 * float(dm["trade_precision"])
            - 0.30 * abs(float(c1["mdd"]))
            - max(0.0, 0.12 - float(dm["coverage"])) * 14.0
            - max(0.0, float(c1["trades_per_day"]) - 3.5) * 2.0
        )
    return {"backtest": bt, "direction": dm, "score": float(score)}


def _grid(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Train hierarchical HGB parent on Alpha5.13 regime4_core labels.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--raw-2025-csv", type=Path, default=DEFAULT_RAW_2025)
    p.add_argument("--raw-2026-csv", type=Path, default=DEFAULT_RAW_2026)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--trade-thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--quality-thresholds", default="0.45,0.50,0.55,0.60,0.65")
    p.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.08,0.12,0.16")
    p.add_argument("--trade-score-thresholds", default="0.25,0.30,0.35,0.40,0.45,0.50")
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=51421)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_2025 = _read(args.raw_2025_csv)
    raw_2026 = _read(args.raw_2026_csv)
    audit = _verify_state24_sticky090_inputs(raw_2025, raw_2026, args.manifest, args.clean4_report)

    train_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_train.parquet")
    val_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_val.parquet")
    oos_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_oos.parquet")

    train_fit = train_df[train_df["label_train_keep"] == 1].reset_index(drop=True)
    train_trade = train_fit[train_fit["label_action"] != 0].reset_index(drop=True)
    val_trade = val_df[val_df["label_action"] != 0].reset_index(drop=True)
    oos_trade = oos_df[oos_df["label_action"] != 0].reset_index(drop=True)

    cols = _feature_cols(raw_2025, raw_2026, set(train_df.columns))
    if not cols:
        raise ValueError("no usable regime4_core features found")

    x_train_gate = _x(train_fit, cols)
    x_train_side = _x(train_trade, cols)
    x_val = _x(val_df, cols)
    x_oos = _x(oos_df, cols)

    y_train_gate = _gate_target(train_fit)
    y_train_side = _side_target(train_trade)
    y_train_quality = _quality_target(train_trade)
    y_val = pd.to_numeric(val_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    y_oos = pd.to_numeric(oos_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    w_gate = _gate_weights(train_fit)
    w_side = _side_weights(train_trade)
    w_quality = _quality_weights(train_trade)

    stages = _make_stage_specs()
    rows: list[dict[str, Any]] = []
    total = len(stages)

    print(json.dumps({
        "stage": "start",
        "model_id": MODEL_ID,
        "baseline_fixed": {"track": "regime4_core", "classifier": "deeper_hgb_single"},
        "rows": {
            "train_fit": int(len(train_fit)),
            "train_trade": int(len(train_trade)),
            "validation": int(len(val_df)),
            "validation_trade": int(len(val_trade)),
            "oos": int(len(oos_df)),
            "oos_trade": int(len(oos_trade)),
        },
        "feature_count": int(len(cols)),
        "audit_expected_model_found": audit.get("expected_model_found_in_manifest"),
    }, ensure_ascii=False, default=_json_default), flush=True)

    for stage_i, stage in enumerate(stages, start=1):
        print(json.dumps({
            "stage": "fit",
            "done": stage_i,
            "total": total,
            "architecture": stage["name"],
            "gate": stage["gate"].name,
            "side": stage["side"].name,
            "quality": stage["quality"].name,
        }, ensure_ascii=False), flush=True)

        gate_model = _fit_hgb(x_train_gate, y_train_gate, w_gate, stage["gate"], int(args.seed + stage_i * 100 + 1))
        side_model = _fit_hgb(x_train_side, y_train_side, w_side, stage["side"], int(args.seed + stage_i * 100 + 2))
        quality_model = _fit_hgb(x_train_side, y_train_quality, w_quality, stage["quality"], int(args.seed + stage_i * 100 + 3))

        val_gate = _binary_proba(gate_model, x_val)
        val_side = _binary_proba(side_model, x_val)
        val_quality = _binary_proba(quality_model, x_val)

        best_val: dict[str, Any] | None = None
        for trade_threshold in _grid(args.trade_thresholds):
            for quality_threshold in _grid(args.quality_thresholds):
                for margin_threshold in _grid(args.margin_thresholds):
                    for trade_score_threshold in _grid(args.trade_score_thresholds):
                        val_actions, val_diag = _compose_actions(
                            val_gate,
                            val_side,
                            val_quality,
                            trade_threshold=trade_threshold,
                            quality_threshold=quality_threshold,
                            margin_threshold=margin_threshold,
                            trade_score_threshold=trade_score_threshold,
                        )
                        val_eval = _eval(
                            val_df,
                            val_actions,
                            y_val,
                            fee=float(args.fee),
                            slip=float(args.slip),
                            exposure=float(args.unit_exposure),
                            max_hold=int(args.max_hold_bars),
                        )
                        cand = {
                            "trade_threshold": float(trade_threshold),
                            "quality_threshold": float(quality_threshold),
                            "margin_threshold": float(margin_threshold),
                            "trade_score_threshold": float(trade_score_threshold),
                            "diag": {
                                "trade_score_mean": float(np.mean(val_diag["trade_score"])),
                                "p_trade_mean": float(np.mean(val_diag["p_trade"])),
                                "p_quality_mean": float(np.mean(val_diag["p_quality"])),
                                "side_margin_mean": float(np.mean(val_diag["side_margin"])),
                            },
                            **val_eval,
                        }
                        if best_val is None or float(cand["score"]) > float(best_val["score"]):
                            best_val = cand
        assert best_val is not None

        oos_gate = _binary_proba(gate_model, x_oos)
        oos_side = _binary_proba(side_model, x_oos)
        oos_quality = _binary_proba(quality_model, x_oos)
        oos_actions, oos_diag = _compose_actions(
            oos_gate,
            oos_side,
            oos_quality,
            trade_threshold=float(best_val["trade_threshold"]),
            quality_threshold=float(best_val["quality_threshold"]),
            margin_threshold=float(best_val["margin_threshold"]),
            trade_score_threshold=float(best_val["trade_score_threshold"]),
        )
        oos_eval = _eval(
            oos_df,
            oos_actions,
            y_oos,
            fee=float(args.fee),
            slip=float(args.slip),
            exposure=float(args.unit_exposure),
            max_hold=int(args.max_hold_bars),
        )
        oos_eval["diag"] = {
            "trade_score_mean": float(np.mean(oos_diag["trade_score"])),
            "p_trade_mean": float(np.mean(oos_diag["p_trade"])),
            "p_quality_mean": float(np.mean(oos_diag["p_quality"])),
            "side_margin_mean": float(np.mean(oos_diag["side_margin"])),
        }

        artifact = args.out_dir / f"{stage['name']}_alpha5_14_hgb_hierarchical_core.joblib"
        joblib.dump({
            "model_id": MODEL_ID,
            "architecture": stage["name"],
            "feature_cols": cols,
            "baseline_fixed": {"track": "regime4_core", "single_hgb": "deeper"},
            "models": {
                "trade_gate": gate_model,
                "side": side_model,
                "quality": quality_model,
            },
            "specs": {
                "gate": stage["gate"].name,
                "side": stage["side"].name,
                "quality": stage["quality"].name,
            },
            "decision": {
                "trade_threshold": float(best_val["trade_threshold"]),
                "quality_threshold": float(best_val["quality_threshold"]),
                "margin_threshold": float(best_val["margin_threshold"]),
                "trade_score_threshold": float(best_val["trade_score_threshold"]),
            },
        }, artifact)

        row = {
            "architecture": stage["name"],
            "baseline": {"track": "regime4_core", "single_hgb": "deeper"},
            "specs": {
                "gate": stage["gate"].name,
                "side": stage["side"].name,
                "quality": stage["quality"].name,
            },
            "feature_count": int(len(cols)),
            "train_rows": {"gate": int(len(train_fit)), "side": int(len(train_trade)), "quality": int(len(train_trade))},
            "validation": best_val,
            "oos": oos_eval,
            "artifact": str(artifact),
        }
        rows.append(row)
        print(json.dumps({
            "stage": "candidate",
            "architecture": stage["name"],
            "specs": row["specs"],
            "val_score": best_val["score"],
            "val_cost1": best_val["backtest"]["cost1"],
            "val_direction": best_val["direction"],
            "oos_score": oos_eval["score"],
            "oos_cost1": oos_eval["backtest"]["cost1"],
            "oos_direction": oos_eval["direction"],
        }, ensure_ascii=False, default=_json_default), flush=True)

    best = max(rows, key=lambda r: float(r["validation"]["score"]))
    summary = {
        "model_id": MODEL_ID,
        "design": "Hierarchical HGB parent: trade gate + side classifier + quality gate, fixed on regime4_core baseline.",
        "baseline_fixed": {"track": "regime4_core", "single_hgb": "deeper"},
        "experiments": rows,
        "best": best,
        "top10": sorted(rows, key=lambda r: float(r["validation"]["score"]), reverse=True)[:10],
    }
    summary_path = args.out_dir / "alpha5_14_hgb_hierarchical_core_summary.json"
    grid_path = args.out_dir / "alpha5_14_hgb_hierarchical_core_grid.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame([
        {
            "architecture": r["architecture"],
            "gate_hgb": r["specs"]["gate"],
            "side_hgb": r["specs"]["side"],
            "quality_hgb": r["specs"]["quality"],
            "feature_count": r["feature_count"],
            "val_score": r["validation"]["score"],
            "val_trade_threshold": r["validation"]["trade_threshold"],
            "val_quality_threshold": r["validation"]["quality_threshold"],
            "val_margin_threshold": r["validation"]["margin_threshold"],
            "val_trade_score_threshold": r["validation"]["trade_score_threshold"],
            "val_trade_precision": r["validation"]["direction"]["trade_precision"],
            "val_balanced_trade_precision": r["validation"]["direction"]["balanced_trade_precision"],
            "val_coverage": r["validation"]["direction"]["coverage"],
            "val_cost1_pnl": r["validation"]["backtest"]["cost1"]["pnl"],
            "val_cost1_mdd": r["validation"]["backtest"]["cost1"]["mdd"],
            "val_cost1_trades": r["validation"]["backtest"]["cost1"]["trades"],
            "oos_score": r["oos"]["score"],
            "oos_trade_precision": r["oos"]["direction"]["trade_precision"],
            "oos_balanced_trade_precision": r["oos"]["direction"]["balanced_trade_precision"],
            "oos_coverage": r["oos"]["direction"]["coverage"],
            "oos_cost1_pnl": r["oos"]["backtest"]["cost1"]["pnl"],
            "oos_cost1_mdd": r["oos"]["backtest"]["cost1"]["mdd"],
            "oos_cost1_trades": r["oos"]["backtest"]["cost1"]["trades"],
            "oos_cost2_pnl": r["oos"]["backtest"]["cost2"]["pnl"],
            "oos_cost3_pnl": r["oos"]["backtest"]["cost3"]["pnl"],
            "artifact": r["artifact"],
        }
        for r in rows
    ]).sort_values("val_score", ascending=False).to_csv(grid_path, index=False)
    print(json.dumps({
        "stage": "complete",
        "summary": str(summary_path),
        "grid": str(grid_path),
        "best": {
            "architecture": best["architecture"],
            "specs": best["specs"],
            "val_score": best["validation"]["score"],
            "oos_score": best["oos"]["score"],
            "oos_cost1": best["oos"]["backtest"]["cost1"],
            "oos_direction": best["oos"]["direction"],
        },
    }, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
