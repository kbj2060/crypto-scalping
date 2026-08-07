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


MODEL_ID = "alpha5_17_hgb_2stage_regime_direction_20260518"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_high_quality_training_data_20260518"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_17_hgb_2stage_regime_direction_20260518"
REGIMES = ("bull", "bear", "chop", "whipsaw")
REGIME_PROB_COLS = [f"clean_regime4_2024_unsup_v1_{r}_prob" for r in REGIMES]


def _x(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return prepare_features(frame, side_hint=0, close=_close(frame), feature_cols=cols)


def _all_cols(train_raw: pd.DataFrame, eval_raw: pd.DataFrame, available: set[str]) -> list[str]:
    cols = _alpha4_mapped_features(train_raw, eval_raw, include_future=False)
    return [c for c in cols if c in available]


def _entry_feature_cols(cols: list[str]) -> list[str]:
    exact = {
        "volatility_z","garch_vol_z","rogers_satchell_vol","liquidity_vacuum","execution_quality","jump_z","jump_flag",
        "evt_tail_flag","evt_excess_z","funding_abs","trade_intensity","squeeze_power","breakout_strength",
        "patchtst_regime_sim","ai_vol_regime_pct","tide_vol_raw","tide_vol_zscore","tp_sl_action_score",
    }
    prefixes = (
        "clean_regime4_2024_unsup_v1_bear_prob","clean_regime4_2024_unsup_v1_bull_prob","clean_regime4_2024_unsup_v1_chop_prob",
        "clean_regime4_2024_unsup_v1_whipsaw_prob","clean_regime4_2024_unsup_v1_trend_prob","clean_regime4_2024_unsup_v1_micro_prob",
        "clean_regime4_2024_unsup_v1_range_prob","clean_regime4_2024_unsup_v1_instability_prob","clean_regime4_2024_unsup_v1_confidence",
        "clean_regime4_2024_unsup_v1_entropy","clean_regime4_2024_unsup_v1_factor_vol","clean_regime4_2024_unsup_v1_factor_liquidity",
        "clean_regime4_2024_unsup_v1_factor_crowding","clean_regime4_2024_unsup_v1_risk_off_prob","clean_regime4_2024_unsup_v1_transition_risk",
        "clean_regime4_2024_unsup_v1_margin",
    )
    return list(dict.fromkeys([c for c in cols if c in exact or c.startswith(prefixes)]))


def _direction_feature_cols(cols: list[str]) -> list[str]:
    exact = {
        "mom_21d","mom_3d","mom_1d","log_return","funding_pressure","funding_price_divergence","crowding_pressure",
        "smart_money_flow","net_taker_ratio","taker_acceleration","ofi_acceleration","whale_conviction","m7_expected_ret",
        "m7_composite_score","m7_confidence","ai_dir_edge","ai_dir_p_up","ai_dir_p_down","ai_flow_pressure","ai_flow_exhaustion",
        "ai_flow_flip_prob","ai_flow_slope","dlinear_smf_ema","dlinear_smf_slope","mtf_trend_1h","mtf_trend_4h","rsi",
        "big_trade_ratio","whale_retail_ratio","breakout_strength","funding_abs",
    }
    prefixes = (
        "clean_regime4_2024_unsup_v1_bear_prob","clean_regime4_2024_unsup_v1_bull_prob","clean_regime4_2024_unsup_v1_directional_bias",
        "clean_regime4_2024_unsup_v1_trend_bias","clean_regime4_2024_unsup_v1_factor_flow","clean_regime4_2024_unsup_v1_factor_trend",
        "clean_regime4_2024_unsup_v1_trend_prob","clean_regime4_2024_unsup_v1_margin",
    )
    return list(dict.fromkeys([c for c in cols if c in exact or c.startswith(prefixes)]))


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


def _entry_target(frame: pd.DataFrame, *, consensus_min: float, edge_gap_min: float) -> np.ndarray:
    action = pd.to_numeric(frame["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    tp_first = pd.to_numeric(frame["meta_tp_first"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
    profitable = pd.to_numeric(frame["meta_is_profitable"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
    selected = pd.to_numeric(frame["regime_trade_selected"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
    consensus = pd.to_numeric(frame["label_consensus"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    edge_gap = pd.to_numeric(frame["meta_edge_gap"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    mask = (action != 0) & tp_first & profitable & selected & (consensus >= float(consensus_min)) & (edge_gap >= float(edge_gap_min))
    return mask.astype(np.int64)


def _direction_target(frame: pd.DataFrame) -> np.ndarray:
    return (pd.to_numeric(frame["label_action"], errors="coerce").fillna(0).to_numpy(np.int64) == 1).astype(np.int64)


def _regime_ids(frame: pd.DataFrame) -> np.ndarray:
    p = frame[REGIME_PROB_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float64)
    return np.argmax(p, axis=1).astype(np.int64)


def _entry_weights(frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    base = pd.to_numeric(frame["label_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    consensus = pd.to_numeric(frame["label_consensus"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    edge_gap = pd.to_numeric(frame["meta_edge_gap"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    tp_first = pd.to_numeric(frame["meta_tp_first"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    w = np.clip(base, 1e-4, None) * (0.80 + 0.40 * np.clip(consensus, 0.0, 1.0))
    w *= 1.0 + 0.02 * np.clip(edge_gap, 0.0, 10.0)
    w *= 1.0 + 0.25 * tp_first
    return w * _balanced_weights(y)


def _direction_weights(frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    base = pd.to_numeric(frame["label_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    edge_gap = pd.to_numeric(frame["meta_edge_gap"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    profitable = pd.to_numeric(frame["meta_is_profitable"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    w = np.clip(base, 1e-4, None) * (1.0 + 0.03 * np.clip(edge_gap, 0.0, 10.0))
    w *= 1.0 + 0.10 * profitable
    return w * _balanced_weights(y)


def _compose_actions(
    p_entry: np.ndarray,
    p_long_by_row: np.ndarray,
    *,
    entry_threshold: float,
    side_threshold: float,
    margin_threshold: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    p_entry = np.clip(p_entry, 0.0, 1.0)
    p_long = np.clip(p_long_by_row, 0.0, 1.0)
    p_short = 1.0 - p_long
    margin = np.abs(p_long - p_short)
    best_side = np.maximum(p_long, p_short)
    actions = np.where(p_long >= p_short, 1, 2).astype(np.int64)
    actions = np.where(p_entry < float(entry_threshold), 0, actions)
    actions = np.where(best_side < float(side_threshold), 0, actions)
    actions = np.where(margin < float(margin_threshold), 0, actions)
    return actions, {"p_entry": p_entry, "p_long": p_long, "p_short": p_short, "best_side": best_side, "margin": margin}


def _eval(frame: pd.DataFrame, actions: np.ndarray, labels: np.ndarray, *, fee: float, slip: float, exposure: float, max_hold: int) -> dict[str, Any]:
    bt = {
        f"cost{m}": _backtest_barrier(
            frame, actions, fee=float(fee) * float(m), slip=float(slip) * float(m), unit_exposure=float(exposure), max_hold_bars=int(max_hold)
        )
        for m in (1, 2, 3)
    }
    dm = _direction_metrics(actions, labels)
    c1, c2, c3 = bt["cost1"], bt["cost2"], bt["cost3"]
    if int(c1["trades"]) < 20:
        score = -1e6 + float(c1["pnl"])
    else:
        score = (
            20.0 * float(dm["balanced_trade_precision"])
            + 10.0 * float(dm["trade_precision"])
            + float(c1["pnl"])
            + 0.35 * float(c2["pnl"])
            + 0.10 * float(c3["pnl"])
            - 0.20 * abs(float(c1["mdd"]))
            - max(0.0, 0.10 - float(dm["coverage"])) * 12.0
            - max(0.0, float(c1["trades_per_day"]) - 2.0) * 3.0
        )
    return {"backtest": bt, "direction": dm, "score": float(score)}


def _stage_specs() -> list[dict[str, HGBSpec]]:
    specs = {spec.name: spec for spec in _hgb_specs()}
    return [
        {"name": "regdir_v1", "entry": specs["regularized"], "direction": specs["regularized"]},
        {"name": "regdir_v2", "entry": specs["regularized"], "direction": specs["deeper"]},
        {"name": "regdir_v3", "entry": specs["deeper"], "direction": specs["regularized"]},
    ]


def _grid(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Train global stage1 + regime-specific stage2 HGB action selector.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--raw-2025-csv", type=Path, default=DEFAULT_RAW_2025)
    p.add_argument("--raw-2026-csv", type=Path, default=DEFAULT_RAW_2026)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--entry-consensus-min", type=float, default=0.85)
    p.add_argument("--entry-edge-gap-min", type=float, default=2.0)
    p.add_argument("--entry-thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--side-thresholds", default="0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.08,0.12")
    p.add_argument("--min-regime-samples", type=int, default=1200)
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=51701)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_2025 = _read(args.raw_2025_csv)
    raw_2026 = _read(args.raw_2026_csv)
    audit = _verify_state24_sticky090_inputs(raw_2025, raw_2026, args.manifest, args.clean4_report)

    train_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_train.parquet")
    val_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_val.parquet")
    oos_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_oos.parquet")
    train_fit = train_df[train_df["label_train_keep"] == 1].reset_index(drop=True)

    all_cols = _all_cols(raw_2025, raw_2026, set(train_df.columns))
    entry_cols = _entry_feature_cols(all_cols)
    direction_cols = _direction_feature_cols(all_cols)
    target_kwargs = {"consensus_min": float(args.entry_consensus_min), "edge_gap_min": float(args.entry_edge_gap_min)}

    x_train_entry = _x(train_fit, entry_cols)
    x_val_entry = _x(val_df, entry_cols)
    x_oos_entry = _x(oos_df, entry_cols)
    x_train_direction = _x(train_fit, direction_cols)
    x_val_direction = _x(val_df, direction_cols)
    x_oos_direction = _x(oos_df, direction_cols)

    y_entry = _entry_target(train_fit, **target_kwargs)
    y_val = pd.to_numeric(val_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    y_oos = pd.to_numeric(oos_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    reg_train = _regime_ids(train_fit)
    reg_val = _regime_ids(val_df)
    reg_oos = _regime_ids(oos_df)
    direction_train_mask = y_entry == 1

    w_entry = _entry_weights(train_fit, y_entry)
    stages = _stage_specs()
    rows: list[dict[str, Any]] = []
    print(json.dumps({
        "stage": "start",
        "model_id": MODEL_ID,
        "baseline_fixed": {"track": "regime4_core", "single_hgb": "deeper"},
        "rows": {"train_fit": int(len(train_fit)), "validation": int(len(val_df)), "oos": int(len(oos_df))},
        "feature_counts": {"entry": int(len(entry_cols)), "direction": int(len(direction_cols))},
        "entry_target": target_kwargs,
        "entry_positive_ratio_train": float(np.mean(y_entry)),
        "min_regime_samples": int(args.min_regime_samples),
        "audit_expected_model_found": audit.get("expected_model_found_in_manifest"),
    }, ensure_ascii=False, default=_json_default), flush=True)

    for i, stage in enumerate(stages, start=1):
        print(json.dumps({
            "stage": "fit",
            "done": i,
            "total": len(stages),
            "architecture": stage["name"],
            "entry_hgb": stage["entry"].name,
            "direction_hgb": stage["direction"].name,
        }, ensure_ascii=False), flush=True)
        entry_model = _fit_hgb(x_train_entry, y_entry, w_entry, stage["entry"], int(args.seed + i * 100 + 1))

        global_dir_frame = train_fit.loc[direction_train_mask].reset_index(drop=True)
        global_dir_x = x_train_direction.loc[direction_train_mask].reset_index(drop=True)
        global_dir_y = _direction_target(global_dir_frame)
        global_dir_w = _direction_weights(global_dir_frame, global_dir_y)
        global_direction_model = _fit_hgb(global_dir_x, global_dir_y, global_dir_w, stage["direction"], int(args.seed + i * 100 + 2))

        regime_models: dict[str, Any] = {}
        regime_meta: dict[str, Any] = {}
        for ridx, rname in enumerate(REGIMES):
            mask = direction_train_mask & (reg_train == ridx)
            n = int(np.sum(mask))
            if n < int(args.min_regime_samples):
                regime_models[rname] = global_direction_model
                regime_meta[rname] = {"rows": n, "fallback": True}
                continue
            sub_frame = train_fit.loc[mask].reset_index(drop=True)
            sub_x = x_train_direction.loc[mask].reset_index(drop=True)
            sub_y = _direction_target(sub_frame)
            sub_w = _direction_weights(sub_frame, sub_y)
            if len(np.unique(sub_y)) < 2:
                regime_models[rname] = global_direction_model
                regime_meta[rname] = {"rows": n, "fallback": True, "single_class": True}
                continue
            regime_models[rname] = _fit_hgb(sub_x, sub_y, sub_w, stage["direction"], int(args.seed + i * 100 + 10 + ridx))
            regime_meta[rname] = {
                "rows": n,
                "fallback": False,
                "long_rows": int(np.sum(sub_y == 1)),
                "short_rows": int(np.sum(sub_y == 0)),
            }

        p_entry_val = _binary_proba(entry_model, x_val_entry)
        p_long_val = np.zeros(len(val_df), dtype=np.float64)
        for ridx, rname in enumerate(REGIMES):
            mask = reg_val == ridx
            if np.any(mask):
                p_long_val[mask] = _binary_proba(regime_models[rname], x_val_direction.loc[mask])

        best_val: dict[str, Any] | None = None
        for entry_threshold in _grid(args.entry_thresholds):
            for side_threshold in _grid(args.side_thresholds):
                for margin_threshold in _grid(args.margin_thresholds):
                    val_actions, val_diag = _compose_actions(
                        p_entry_val, p_long_val,
                        entry_threshold=entry_threshold, side_threshold=side_threshold, margin_threshold=margin_threshold,
                    )
                    val_eval = _eval(val_df, val_actions, y_val, fee=float(args.fee), slip=float(args.slip), exposure=float(args.unit_exposure), max_hold=int(args.max_hold_bars))
                    cand = {
                        "entry_threshold": float(entry_threshold),
                        "side_threshold": float(side_threshold),
                        "margin_threshold": float(margin_threshold),
                        "diag": {
                            "p_entry_mean": float(np.mean(val_diag["p_entry"])),
                            "best_side_mean": float(np.mean(val_diag["best_side"])),
                            "margin_mean": float(np.mean(val_diag["margin"])),
                        },
                        **val_eval,
                    }
                    if best_val is None or float(cand["score"]) > float(best_val["score"]):
                        best_val = cand
        assert best_val is not None

        p_entry_oos = _binary_proba(entry_model, x_oos_entry)
        p_long_oos = np.zeros(len(oos_df), dtype=np.float64)
        for ridx, rname in enumerate(REGIMES):
            mask = reg_oos == ridx
            if np.any(mask):
                p_long_oos[mask] = _binary_proba(regime_models[rname], x_oos_direction.loc[mask])
        oos_actions, oos_diag = _compose_actions(
            p_entry_oos, p_long_oos,
            entry_threshold=float(best_val["entry_threshold"]), side_threshold=float(best_val["side_threshold"]), margin_threshold=float(best_val["margin_threshold"]),
        )
        oos_eval = _eval(oos_df, oos_actions, y_oos, fee=float(args.fee), slip=float(args.slip), exposure=float(args.unit_exposure), max_hold=int(args.max_hold_bars))
        oos_eval["diag"] = {
            "p_entry_mean": float(np.mean(oos_diag["p_entry"])),
            "best_side_mean": float(np.mean(oos_diag["best_side"])),
            "margin_mean": float(np.mean(oos_diag["margin"])),
        }

        artifact = args.out_dir / f"{stage['name']}_alpha5_17_hgb_2stage_regime_direction.joblib"
        joblib.dump({
            "model_id": MODEL_ID,
            "baseline_fixed": {"track": "regime4_core", "single_hgb": "deeper"},
            "entry_feature_cols": entry_cols,
            "direction_feature_cols": direction_cols,
            "models": {"entry": entry_model, "direction_global": global_direction_model, "direction_by_regime": regime_models},
            "regime_meta": regime_meta,
            "specs": {"entry": stage["entry"].name, "direction": stage["direction"].name},
            "decision": {
                "entry_threshold": float(best_val["entry_threshold"]),
                "side_threshold": float(best_val["side_threshold"]),
                "margin_threshold": float(best_val["margin_threshold"]),
            },
        }, artifact)
        row = {
            "architecture": stage["name"],
            "baseline": {"track": "regime4_core", "single_hgb": "deeper"},
            "specs": {"entry": stage["entry"].name, "direction": stage["direction"].name},
            "feature_counts": {"entry": int(len(entry_cols)), "direction": int(len(direction_cols))},
            "regime_meta": regime_meta,
            "validation": best_val,
            "oos": oos_eval,
            "artifact": str(artifact),
        }
        rows.append(row)
        print(json.dumps({
            "stage": "candidate",
            "architecture": stage["name"],
            "specs": row["specs"],
            "regime_meta": regime_meta,
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
        "design": "Global stage1 entry HGB + regime-specific stage2 direction HGB on regime4_core.",
        "baseline_fixed": {"track": "regime4_core", "single_hgb": "deeper"},
        "experiments": rows,
        "best": best,
        "top10": sorted(rows, key=lambda r: float(r["validation"]["score"]), reverse=True)[:10],
    }
    summary_path = args.out_dir / "alpha5_17_hgb_2stage_regime_direction_summary.json"
    grid_path = args.out_dir / "alpha5_17_hgb_2stage_regime_direction_grid.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame([
        {
            "architecture": r["architecture"],
            "entry_hgb": r["specs"]["entry"],
            "direction_hgb": r["specs"]["direction"],
            "entry_feature_count": r["feature_counts"]["entry"],
            "direction_feature_count": r["feature_counts"]["direction"],
            "val_score": r["validation"]["score"],
            "val_entry_threshold": r["validation"]["entry_threshold"],
            "val_side_threshold": r["validation"]["side_threshold"],
            "val_margin_threshold": r["validation"]["margin_threshold"],
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
        } for r in rows
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
