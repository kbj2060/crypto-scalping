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


MODEL_ID = "alpha5_15_hgb_ovr_core_20260518"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_high_quality_training_data_20260518"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_15_hgb_ovr_core_20260518"


def _x(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return prepare_features(frame, side_hint=0, close=_close(frame), feature_cols=cols)


def _feature_cols(train_raw: pd.DataFrame, eval_raw: pd.DataFrame, available: set[str]) -> list[str]:
    cols = _alpha4_mapped_features(train_raw, eval_raw, include_future=False)
    return [c for c in cols if c in available]


def _binary_proba(model: Any, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        return raw[:, classes.index(1)]
    return raw[:, -1]


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    out = np.ones(len(y), dtype=np.float64)
    classes, counts = np.unique(y, return_counts=True)
    total = max(float(len(y)), 1.0)
    for cls, count in zip(classes, counts):
        out[y == int(cls)] = total / (float(len(classes)) * max(float(count), 1.0))
    return out


def _strict_trade_mask(frame: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(frame["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    tp_first = pd.to_numeric(frame["meta_tp_first"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
    profitable = pd.to_numeric(frame["meta_is_profitable"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
    selected = pd.to_numeric(frame["regime_trade_selected"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
    return (action != 0) & tp_first & profitable & selected


def _long_target(frame: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(frame["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    return ((action == 1) & _strict_trade_mask(frame)).astype(np.int64)


def _short_target(frame: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(frame["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    return ((action == 2) & _strict_trade_mask(frame)).astype(np.int64)


def _trade_target(frame: pd.DataFrame) -> np.ndarray:
    return _strict_trade_mask(frame).astype(np.int64)


def _stage_weights(frame: pd.DataFrame, y: np.ndarray, *, boost_edge: float, cash_dampen: float) -> np.ndarray:
    action = pd.to_numeric(frame["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    base = pd.to_numeric(frame["label_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    consensus = pd.to_numeric(frame["label_consensus"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    edge_gap = pd.to_numeric(frame["meta_edge_gap"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    tp_first = pd.to_numeric(frame["meta_tp_first"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    profitable = pd.to_numeric(frame["meta_is_profitable"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    w = np.clip(base, 1e-4, None)
    w *= 0.75 + 0.45 * np.clip(consensus, 0.0, 1.0)
    w *= 1.0 + float(boost_edge) * np.clip(edge_gap, 0.0, 10.0) / 10.0
    w *= 1.0 + 0.20 * tp_first + 0.10 * profitable
    w[action == 0] *= float(cash_dampen)
    return w * _balanced_weights(y)


def _compose_actions(
    p_trade: np.ndarray,
    p_long: np.ndarray,
    p_short: np.ndarray,
    *,
    trade_threshold: float,
    side_threshold: float,
    margin_threshold: float,
    score_threshold: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    trade_score = np.clip(p_trade, 0.0, 1.0)
    long_score = trade_score * np.clip(p_long, 0.0, 1.0)
    short_score = trade_score * np.clip(p_short, 0.0, 1.0)
    best_side_score = np.maximum(long_score, short_score)
    side_margin = np.abs(long_score - short_score)
    actions = np.where(long_score >= short_score, 1, 2).astype(np.int64)
    actions = np.where(trade_score < float(trade_threshold), 0, actions)
    actions = np.where(np.maximum(p_long, p_short) < float(side_threshold), 0, actions)
    actions = np.where(side_margin < float(margin_threshold), 0, actions)
    actions = np.where(best_side_score < float(score_threshold), 0, actions)
    return actions, {
        "trade_score": trade_score,
        "long_score": long_score,
        "short_score": short_score,
        "best_side_score": best_side_score,
        "side_margin": side_margin,
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
            18.0 * float(dm["balanced_trade_precision"])
            + 10.0 * float(dm["trade_precision"])
            + float(c1["pnl"])
            + 0.35 * float(c2["pnl"])
            + 0.10 * float(c3["pnl"])
            - 0.22 * abs(float(c1["mdd"]))
            - max(0.0, 0.10 - float(dm["coverage"])) * 12.0
            - max(0.0, float(c1["trades_per_day"]) - 2.5) * 2.5
        )
    return {"backtest": bt, "direction": dm, "score": float(score)}


def _stage_specs() -> list[dict[str, Any]]:
    specs = {spec.name: spec for spec in _hgb_specs()}
    return [
        {"name": "ovr_v1", "trade": specs["regularized"], "long": specs["deeper"], "short": specs["deeper"], "boost_edge": 0.25, "cash_dampen": 1.00},
        {"name": "ovr_v2", "trade": specs["regularized"], "long": specs["regularized"], "short": specs["regularized"], "boost_edge": 0.35, "cash_dampen": 1.00},
        {"name": "ovr_v3", "trade": specs["deeper"], "long": specs["deeper"], "short": specs["regularized"], "boost_edge": 0.20, "cash_dampen": 0.90},
    ]


def _grid(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Train one-vs-rest HGB action selector on Alpha5.13 regime4_core.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--raw-2025-csv", type=Path, default=DEFAULT_RAW_2025)
    p.add_argument("--raw-2026-csv", type=Path, default=DEFAULT_RAW_2026)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--trade-thresholds", default="0.55,0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--side-thresholds", default="0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.08,0.12")
    p.add_argument("--score-thresholds", default="0.25,0.30,0.35,0.40,0.45")
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=51501)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_2025 = _read(args.raw_2025_csv)
    raw_2026 = _read(args.raw_2026_csv)
    audit = _verify_state24_sticky090_inputs(raw_2025, raw_2026, args.manifest, args.clean4_report)

    train_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_train.parquet")
    val_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_val.parquet")
    oos_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_oos.parquet")
    train_fit = train_df[train_df["label_train_keep"] == 1].reset_index(drop=True)

    cols = _feature_cols(raw_2025, raw_2026, set(train_df.columns))
    x_train = _x(train_fit, cols)
    x_val = _x(val_df, cols)
    x_oos = _x(oos_df, cols)

    y_trade = _trade_target(train_fit)
    y_long = _long_target(train_fit)
    y_short = _short_target(train_fit)
    y_val = pd.to_numeric(val_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    y_oos = pd.to_numeric(oos_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)

    stages = _stage_specs()
    rows: list[dict[str, Any]] = []
    print(json.dumps({
        "stage": "start",
        "model_id": MODEL_ID,
        "baseline_fixed": {"track": "regime4_core", "single_hgb": "deeper"},
        "rows": {"train_fit": int(len(train_fit)), "validation": int(len(val_df)), "oos": int(len(oos_df))},
        "feature_count": int(len(cols)),
        "strict_trade_ratio": float(np.mean(y_trade)),
        "strict_long_ratio": float(np.mean(y_long)),
        "strict_short_ratio": float(np.mean(y_short)),
        "audit_expected_model_found": audit.get("expected_model_found_in_manifest"),
    }, ensure_ascii=False, default=_json_default), flush=True)

    for i, stage in enumerate(stages, start=1):
        print(json.dumps({
            "stage": "fit",
            "done": i,
            "total": len(stages),
            "architecture": stage["name"],
            "trade_hgb": stage["trade"].name,
            "long_hgb": stage["long"].name,
            "short_hgb": stage["short"].name,
        }, ensure_ascii=False), flush=True)
        w_trade = _stage_weights(train_fit, y_trade, boost_edge=stage["boost_edge"], cash_dampen=stage["cash_dampen"])
        w_long = _stage_weights(train_fit, y_long, boost_edge=stage["boost_edge"], cash_dampen=stage["cash_dampen"])
        w_short = _stage_weights(train_fit, y_short, boost_edge=stage["boost_edge"], cash_dampen=stage["cash_dampen"])

        trade_model = _fit_hgb(x_train, y_trade, w_trade, stage["trade"], int(args.seed + i * 100 + 1))
        long_model = _fit_hgb(x_train, y_long, w_long, stage["long"], int(args.seed + i * 100 + 2))
        short_model = _fit_hgb(x_train, y_short, w_short, stage["short"], int(args.seed + i * 100 + 3))

        p_trade_val = _binary_proba(trade_model, x_val)
        p_long_val = _binary_proba(long_model, x_val)
        p_short_val = _binary_proba(short_model, x_val)

        best_val: dict[str, Any] | None = None
        for trade_threshold in _grid(args.trade_thresholds):
            for side_threshold in _grid(args.side_thresholds):
                for margin_threshold in _grid(args.margin_thresholds):
                    for score_threshold in _grid(args.score_thresholds):
                        val_actions, val_diag = _compose_actions(
                            p_trade_val,
                            p_long_val,
                            p_short_val,
                            trade_threshold=trade_threshold,
                            side_threshold=side_threshold,
                            margin_threshold=margin_threshold,
                            score_threshold=score_threshold,
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
                            "side_threshold": float(side_threshold),
                            "margin_threshold": float(margin_threshold),
                            "score_threshold": float(score_threshold),
                            "diag": {
                                "trade_score_mean": float(np.mean(val_diag["trade_score"])),
                                "long_score_mean": float(np.mean(val_diag["long_score"])),
                                "short_score_mean": float(np.mean(val_diag["short_score"])),
                                "side_margin_mean": float(np.mean(val_diag["side_margin"])),
                            },
                            **val_eval,
                        }
                        if best_val is None or float(cand["score"]) > float(best_val["score"]):
                            best_val = cand
        assert best_val is not None

        p_trade_oos = _binary_proba(trade_model, x_oos)
        p_long_oos = _binary_proba(long_model, x_oos)
        p_short_oos = _binary_proba(short_model, x_oos)
        oos_actions, oos_diag = _compose_actions(
            p_trade_oos,
            p_long_oos,
            p_short_oos,
            trade_threshold=float(best_val["trade_threshold"]),
            side_threshold=float(best_val["side_threshold"]),
            margin_threshold=float(best_val["margin_threshold"]),
            score_threshold=float(best_val["score_threshold"]),
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
            "long_score_mean": float(np.mean(oos_diag["long_score"])),
            "short_score_mean": float(np.mean(oos_diag["short_score"])),
            "side_margin_mean": float(np.mean(oos_diag["side_margin"])),
        }

        artifact = args.out_dir / f"{stage['name']}_alpha5_15_hgb_ovr_core.joblib"
        joblib.dump({
            "model_id": MODEL_ID,
            "feature_cols": cols,
            "baseline_fixed": {"track": "regime4_core", "single_hgb": "deeper"},
            "models": {"trade": trade_model, "long": long_model, "short": short_model},
            "specs": {
                "trade": stage["trade"].name,
                "long": stage["long"].name,
                "short": stage["short"].name,
                "boost_edge": stage["boost_edge"],
                "cash_dampen": stage["cash_dampen"],
            },
            "decision": {
                "trade_threshold": float(best_val["trade_threshold"]),
                "side_threshold": float(best_val["side_threshold"]),
                "margin_threshold": float(best_val["margin_threshold"]),
                "score_threshold": float(best_val["score_threshold"]),
            },
        }, artifact)
        row = {
            "architecture": stage["name"],
            "baseline": {"track": "regime4_core", "single_hgb": "deeper"},
            "specs": {
                "trade": stage["trade"].name,
                "long": stage["long"].name,
                "short": stage["short"].name,
                "boost_edge": stage["boost_edge"],
                "cash_dampen": stage["cash_dampen"],
            },
            "feature_count": int(len(cols)),
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
        "design": "Action-only OVR HGB on regime4_core: strict trade gate plus strict long/short scorers.",
        "baseline_fixed": {"track": "regime4_core", "single_hgb": "deeper"},
        "experiments": rows,
        "best": best,
        "top10": sorted(rows, key=lambda r: float(r["validation"]["score"]), reverse=True)[:10],
    }
    summary_path = args.out_dir / "alpha5_15_hgb_ovr_core_summary.json"
    grid_path = args.out_dir / "alpha5_15_hgb_ovr_core_grid.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame([
        {
            "architecture": r["architecture"],
            "trade_hgb": r["specs"]["trade"],
            "long_hgb": r["specs"]["long"],
            "short_hgb": r["specs"]["short"],
            "boost_edge": r["specs"]["boost_edge"],
            "cash_dampen": r["specs"]["cash_dampen"],
            "feature_count": r["feature_count"],
            "val_score": r["validation"]["score"],
            "val_trade_threshold": r["validation"]["trade_threshold"],
            "val_side_threshold": r["validation"]["side_threshold"],
            "val_margin_threshold": r["validation"]["margin_threshold"],
            "val_score_threshold": r["validation"]["score_threshold"],
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
