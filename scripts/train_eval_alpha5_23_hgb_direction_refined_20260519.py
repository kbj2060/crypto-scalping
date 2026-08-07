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


MODEL_ID = "alpha5_23_hgb_direction_refined_20260519"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_23_direction_refined_labels_20260519"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_23_hgb_direction_refined_20260519"
BASELINE_MODEL = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_single_20260518/regime4_core_deeper_alpha5_13_hgb_parent.joblib"


def _x(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return prepare_features(frame, side_hint=0, close=_close(frame), feature_cols=cols)


def _feature_cols(train_raw: pd.DataFrame, eval_raw: pd.DataFrame, available: set[str]) -> list[str]:
    cols = _alpha4_mapped_features(train_raw, eval_raw, include_future=False)
    return [c for c in cols if c in available]


def _entry_feature_cols(cols: list[str]) -> list[str]:
    exact = {
        "volatility_z",
        "garch_vol_z",
        "rogers_satchell_vol",
        "liquidity_vacuum",
        "execution_quality",
        "jump_z",
        "jump_flag",
        "evt_tail_flag",
        "evt_excess_z",
        "funding_abs",
        "trade_intensity",
        "squeeze_power",
        "breakout_strength",
        "patchtst_regime_sim",
        "ai_vol_regime_pct",
        "tide_vol_raw",
        "tide_vol_zscore",
        "tp_sl_action_score",
    }
    prefixes = (
        "clean_regime4_2024_unsup_v1_bear_prob",
        "clean_regime4_2024_unsup_v1_bull_prob",
        "clean_regime4_2024_unsup_v1_chop_prob",
        "clean_regime4_2024_unsup_v1_whipsaw_prob",
        "clean_regime4_2024_unsup_v1_trend_prob",
        "clean_regime4_2024_unsup_v1_micro_prob",
        "clean_regime4_2024_unsup_v1_range_prob",
        "clean_regime4_2024_unsup_v1_instability_prob",
        "clean_regime4_2024_unsup_v1_confidence",
        "clean_regime4_2024_unsup_v1_entropy",
        "clean_regime4_2024_unsup_v1_factor_vol",
        "clean_regime4_2024_unsup_v1_factor_liquidity",
        "clean_regime4_2024_unsup_v1_factor_crowding",
        "clean_regime4_2024_unsup_v1_risk_off_prob",
        "clean_regime4_2024_unsup_v1_transition_risk",
        "clean_regime4_2024_unsup_v1_margin",
    )
    out = [c for c in cols if c in exact or c.startswith(prefixes)]
    return list(dict.fromkeys(out))


def _direction_feature_cols(cols: list[str]) -> list[str]:
    exact = {
        "mom_21d",
        "mom_3d",
        "mom_1d",
        "log_return",
        "funding_pressure",
        "funding_price_divergence",
        "crowding_pressure",
        "smart_money_flow",
        "net_taker_ratio",
        "taker_acceleration",
        "ofi_acceleration",
        "whale_conviction",
        "m7_expected_ret",
        "m7_composite_score",
        "m7_confidence",
        "ai_dir_edge",
        "ai_dir_p_up",
        "ai_dir_p_down",
        "ai_flow_pressure",
        "ai_flow_exhaustion",
        "ai_flow_flip_prob",
        "ai_flow_slope",
        "dlinear_smf_ema",
        "dlinear_smf_slope",
        "mtf_trend_1h",
        "mtf_trend_4h",
        "rsi",
        "big_trade_ratio",
        "whale_retail_ratio",
        "breakout_strength",
        "funding_abs",
    }
    prefixes = (
        "clean_regime4_2024_unsup_v1_bear_prob",
        "clean_regime4_2024_unsup_v1_bull_prob",
        "clean_regime4_2024_unsup_v1_directional_bias",
        "clean_regime4_2024_unsup_v1_trend_bias",
        "clean_regime4_2024_unsup_v1_factor_flow",
        "clean_regime4_2024_unsup_v1_factor_trend",
        "clean_regime4_2024_unsup_v1_trend_prob",
        "clean_regime4_2024_unsup_v1_margin",
    )
    out = [c for c in cols if c in exact or c.startswith(prefixes)]
    return list(dict.fromkeys(out))


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


def _compose_2stage(
    p_entry: np.ndarray,
    p_long: np.ndarray,
    *,
    entry_threshold: float,
    side_threshold: float,
    margin_threshold: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    p_entry = np.clip(p_entry, 0.0, 1.0)
    p_long = np.clip(p_long, 0.0, 1.0)
    p_short = 1.0 - p_long
    margin = np.abs(p_long - p_short)
    best_side = np.maximum(p_long, p_short)
    actions = np.where(p_long >= p_short, 1, 2).astype(np.int64)
    actions = np.where(p_entry < float(entry_threshold), 0, actions)
    actions = np.where(best_side < float(side_threshold), 0, actions)
    actions = np.where(margin < float(margin_threshold), 0, actions)
    return actions, {"p_entry": p_entry, "p_long": p_long, "p_short": p_short, "margin": margin, "best_side": best_side}


def _compose_ovr(
    p_entry: np.ndarray,
    p_long: np.ndarray,
    p_short: np.ndarray,
    *,
    trade_threshold: float,
    side_threshold: float,
    margin_threshold: float,
    score_threshold: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    trade_score = np.clip(p_entry, 0.0, 1.0)
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


def _grid(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _stage2_specs() -> list[dict[str, HGBSpec]]:
    specs = {spec.name: spec for spec in _hgb_specs()}
    return [
        {"name": "ref2_v1", "entry": specs["regularized"], "direction": specs["deeper"]},
        {"name": "ref2_v2", "entry": specs["deeper"], "direction": specs["regularized"]},
        {"name": "ref2_v3", "entry": specs["regularized"], "direction": specs["regularized"]},
    ]


def _ovr_specs() -> list[dict[str, Any]]:
    specs = {spec.name: spec for spec in _hgb_specs()}
    return [
        {"name": "refovr_v1", "entry": specs["regularized"], "long": specs["deeper"], "short": specs["deeper"]},
        {"name": "refovr_v2", "entry": specs["deeper"], "long": specs["regularized"], "short": specs["regularized"]},
        {"name": "refovr_v3", "entry": specs["regularized"], "long": specs["regularized"], "short": specs["deeper"]},
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Train and evaluate direction-refined HGB retries on alpha5_23 labels.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--train-file", default="alpha5_23_direction_refined_train.parquet")
    p.add_argument("--val-file", default="alpha5_23_direction_refined_val.parquet")
    p.add_argument("--oos-file", default="alpha5_23_direction_refined_oos.parquet")
    p.add_argument("--raw-2025-csv", type=Path, default=DEFAULT_RAW_2025)
    p.add_argument("--raw-2026-csv", type=Path, default=DEFAULT_RAW_2026)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--entry-thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--side-thresholds", default="0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.08,0.12")
    p.add_argument("--score-thresholds", default="0.25,0.30,0.35,0.40")
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=52301)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_2025 = _read(args.raw_2025_csv)
    raw_2026 = _read(args.raw_2026_csv)
    audit = _verify_state24_sticky090_inputs(raw_2025, raw_2026, args.manifest, args.clean4_report)

    train_df = pd.read_parquet(args.data_dir / str(args.train_file))
    val_df = pd.read_parquet(args.data_dir / str(args.val_file))
    oos_df = pd.read_parquet(args.data_dir / str(args.oos_file))

    cols = _feature_cols(raw_2025, raw_2026, set(train_df.columns))
    entry_cols = _entry_feature_cols(cols)
    direction_cols = _direction_feature_cols(cols)
    if not entry_cols or not direction_cols:
        raise ValueError("failed to select entry/direction regime4_core features")

    train_entry = train_df[train_df["entry_train_keep"] == 1].reset_index(drop=True)
    train_dir = train_df[train_df["direction_train_keep"] == 1].reset_index(drop=True)
    if int(np.sum(train_dir["regime4_state"].astype(str) == "whipsaw")) != 0:
        raise ValueError("whipsaw rows remain in direction training subset")

    x_train_entry = _x(train_entry, entry_cols)
    x_val_entry = _x(val_df, entry_cols)
    x_oos_entry = _x(oos_df, entry_cols)
    x_train_dir = _x(train_dir, direction_cols)
    x_val_dir = _x(val_df, direction_cols)
    x_oos_dir = _x(oos_df, direction_cols)

    y_entry = pd.to_numeric(train_entry["entry_label"], errors="coerce").fillna(0).to_numpy(np.int64)
    y_dir = (pd.to_numeric(train_dir["direction_label"], errors="coerce").fillna(0).to_numpy(np.int64) == 1).astype(np.int64)
    y_long = y_dir.copy()
    y_short = 1 - y_dir
    y_val = pd.to_numeric(val_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    y_oos = pd.to_numeric(oos_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)

    w_entry = np.clip(pd.to_numeric(train_entry["entry_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64), 1e-4, None)
    w_dir = np.clip(pd.to_numeric(train_dir["direction_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64), 1e-4, None)
    w_entry *= _balanced_weights(y_entry)
    w_dir *= _balanced_weights(y_dir)
    w_long = w_dir.copy()
    w_short = w_dir.copy()

    rows: list[dict[str, Any]] = []
    best_by_family: dict[str, dict[str, Any]] = {}

    print(json.dumps({
        "stage": "start",
        "model_id": MODEL_ID,
        "baseline_fixed": {"track": "regime4_core", "single_hgb": "deeper", "path": str(BASELINE_MODEL)},
        "rows": {"train_all": int(len(train_df)), "train_entry": int(len(train_entry)), "train_direction": int(len(train_dir)), "validation": int(len(val_df)), "oos": int(len(oos_df))},
        "feature_count": {"entry": int(len(entry_cols)), "direction": int(len(direction_cols))},
        "ratios": {
            "entry_positive": float(np.mean(y_entry)),
            "direction_long_positive": float(np.mean(y_dir)),
        },
        "audit_expected_model_found": audit.get("expected_model_found_in_manifest"),
    }, ensure_ascii=False, default=_json_default), flush=True)

    for i, spec in enumerate(_stage2_specs(), start=1):
        print(json.dumps({"stage": "fit_2stage", "done": i, "total": len(_stage2_specs()), "name": spec["name"], "entry_hgb": spec["entry"].name, "direction_hgb": spec["direction"].name}, ensure_ascii=False), flush=True)
        entry_model = _fit_hgb(x_train_entry, y_entry, w_entry, spec["entry"], int(args.seed + i * 100 + 1))
        direction_model = _fit_hgb(x_train_dir, y_dir, w_dir, spec["direction"], int(args.seed + i * 100 + 2))

        p_entry_val = _binary_proba(entry_model, x_val_entry)
        p_long_val = _binary_proba(direction_model, x_val_dir)

        best_val: dict[str, Any] | None = None
        for entry_threshold in _grid(args.entry_thresholds):
            for side_threshold in _grid(args.side_thresholds):
                for margin_threshold in _grid(args.margin_thresholds):
                    val_actions, val_diag = _compose_2stage(
                        p_entry_val,
                        p_long_val,
                        entry_threshold=entry_threshold,
                        side_threshold=side_threshold,
                        margin_threshold=margin_threshold,
                    )
                    metrics = _eval(val_df, val_actions, y_val, fee=float(args.fee), slip=float(args.slip), exposure=float(args.unit_exposure), max_hold=int(args.max_hold_bars))
                    candidate = {
                        "family": "two_stage",
                        "architecture": spec["name"],
                        "entry_hgb": spec["entry"].name,
                        "direction_hgb": spec["direction"].name,
                        "entry_threshold": float(entry_threshold),
                        "side_threshold": float(side_threshold),
                        "margin_threshold": float(margin_threshold),
                        "validation": metrics,
                        "validation_diag": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in val_diag.items()},
                    }
                    if best_val is None or float(candidate["validation"]["score"]) > float(best_val["validation"]["score"]):
                        best_val = candidate
        assert best_val is not None

        p_entry_oos = _binary_proba(entry_model, x_oos_entry)
        p_long_oos = _binary_proba(direction_model, x_oos_dir)
        oos_actions, oos_diag = _compose_2stage(
            p_entry_oos,
            p_long_oos,
            entry_threshold=float(best_val["entry_threshold"]),
            side_threshold=float(best_val["side_threshold"]),
            margin_threshold=float(best_val["margin_threshold"]),
        )
        best_val["oos"] = _eval(oos_df, oos_actions, y_oos, fee=float(args.fee), slip=float(args.slip), exposure=float(args.unit_exposure), max_hold=int(args.max_hold_bars))
        best_val["oos_diag"] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in oos_diag.items()}
        best_val["artifact_paths"] = {
            "entry_model": str(args.out_dir / f"{spec['name']}_entry_model.joblib"),
            "direction_model": str(args.out_dir / f"{spec['name']}_direction_model.joblib"),
        }
        joblib.dump({"model": entry_model, "feature_cols": entry_cols, "family": "two_stage", "head": "entry"}, best_val["artifact_paths"]["entry_model"])
        joblib.dump({"model": direction_model, "feature_cols": direction_cols, "family": "two_stage", "head": "direction"}, best_val["artifact_paths"]["direction_model"])
        rows.append(best_val)
        if ("two_stage" not in best_by_family) or (float(best_val["validation"]["score"]) > float(best_by_family["two_stage"]["validation"]["score"])):
            best_by_family["two_stage"] = best_val

    for i, spec in enumerate(_ovr_specs(), start=1):
        print(json.dumps({"stage": "fit_ovr", "done": i, "total": len(_ovr_specs()), "name": spec["name"], "entry_hgb": spec["entry"].name, "long_hgb": spec["long"].name, "short_hgb": spec["short"].name}, ensure_ascii=False), flush=True)
        entry_model = _fit_hgb(x_train_entry, y_entry, w_entry, spec["entry"], int(args.seed + 1000 + i * 100 + 1))
        long_model = _fit_hgb(x_train_dir, y_long, w_long, spec["long"], int(args.seed + 1000 + i * 100 + 2))
        short_model = _fit_hgb(x_train_dir, y_short, w_short, spec["short"], int(args.seed + 1000 + i * 100 + 3))

        p_entry_val = _binary_proba(entry_model, x_val_entry)
        p_long_val = _binary_proba(long_model, x_val_dir)
        p_short_val = _binary_proba(short_model, x_val_dir)

        best_val = None
        for entry_threshold in _grid(args.entry_thresholds):
            for side_threshold in _grid(args.side_thresholds):
                for margin_threshold in _grid(args.margin_thresholds):
                    for score_threshold in _grid(args.score_thresholds):
                        val_actions, val_diag = _compose_ovr(
                            p_entry_val,
                            p_long_val,
                            p_short_val,
                            trade_threshold=entry_threshold,
                            side_threshold=side_threshold,
                            margin_threshold=margin_threshold,
                            score_threshold=score_threshold,
                        )
                        metrics = _eval(val_df, val_actions, y_val, fee=float(args.fee), slip=float(args.slip), exposure=float(args.unit_exposure), max_hold=int(args.max_hold_bars))
                        candidate = {
                            "family": "ovr",
                            "architecture": spec["name"],
                            "entry_hgb": spec["entry"].name,
                            "long_hgb": spec["long"].name,
                            "short_hgb": spec["short"].name,
                            "entry_threshold": float(entry_threshold),
                            "side_threshold": float(side_threshold),
                            "margin_threshold": float(margin_threshold),
                            "score_threshold": float(score_threshold),
                            "validation": metrics,
                            "validation_diag": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in val_diag.items()},
                        }
                        if best_val is None or float(candidate["validation"]["score"]) > float(best_val["validation"]["score"]):
                            best_val = candidate
        assert best_val is not None

        p_entry_oos = _binary_proba(entry_model, x_oos_entry)
        p_long_oos = _binary_proba(long_model, x_oos_dir)
        p_short_oos = _binary_proba(short_model, x_oos_dir)
        oos_actions, oos_diag = _compose_ovr(
            p_entry_oos,
            p_long_oos,
            p_short_oos,
            trade_threshold=float(best_val["entry_threshold"]),
            side_threshold=float(best_val["side_threshold"]),
            margin_threshold=float(best_val["margin_threshold"]),
            score_threshold=float(best_val["score_threshold"]),
        )
        best_val["oos"] = _eval(oos_df, oos_actions, y_oos, fee=float(args.fee), slip=float(args.slip), exposure=float(args.unit_exposure), max_hold=int(args.max_hold_bars))
        best_val["oos_diag"] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in oos_diag.items()}
        best_val["artifact_paths"] = {
            "entry_model": str(args.out_dir / f"{spec['name']}_entry_model.joblib"),
            "long_model": str(args.out_dir / f"{spec['name']}_long_model.joblib"),
            "short_model": str(args.out_dir / f"{spec['name']}_short_model.joblib"),
        }
        joblib.dump({"model": entry_model, "feature_cols": entry_cols, "family": "ovr", "head": "entry"}, best_val["artifact_paths"]["entry_model"])
        joblib.dump({"model": long_model, "feature_cols": direction_cols, "family": "ovr", "head": "long"}, best_val["artifact_paths"]["long_model"])
        joblib.dump({"model": short_model, "feature_cols": direction_cols, "family": "ovr", "head": "short"}, best_val["artifact_paths"]["short_model"])
        rows.append(best_val)
        if ("ovr" not in best_by_family) or (float(best_val["validation"]["score"]) > float(best_by_family["ovr"]["validation"]["score"])):
            best_by_family["ovr"] = best_val

    summary = {
        "model_id": MODEL_ID,
        "baseline_fixed": {
            "track": "regime4_core",
            "single_hgb": "deeper",
            "oos_cost1_pnl": 1.22,
            "oos_cost1_mdd": -6.58,
            "oos_cost1_trades": 35,
            "path": str(BASELINE_MODEL),
        },
        "audit": audit,
        "feature_counts": {"entry": len(entry_cols), "direction": len(direction_cols)},
        "best_two_stage": best_by_family.get("two_stage"),
        "best_ovr": best_by_family.get("ovr"),
        "all_results": rows,
    }

    grid_csv = args.out_dir / "alpha5_23_hgb_direction_refined_grid.csv"
    pd.DataFrame([
        {
            "family": row["family"],
            "architecture": row["architecture"],
            "entry_hgb": row.get("entry_hgb"),
            "direction_hgb": row.get("direction_hgb"),
            "long_hgb": row.get("long_hgb"),
            "short_hgb": row.get("short_hgb"),
            "entry_threshold": row["entry_threshold"],
            "side_threshold": row["side_threshold"],
            "margin_threshold": row["margin_threshold"],
            "score_threshold": row.get("score_threshold"),
            "val_score": row["validation"]["score"],
            "val_cost1_pnl": row["validation"]["backtest"]["cost1"]["pnl"],
            "val_cost1_mdd": row["validation"]["backtest"]["cost1"]["mdd"],
            "val_cost1_trades": row["validation"]["backtest"]["cost1"]["trades"],
            "oos_score": row["oos"]["score"],
            "oos_cost1_pnl": row["oos"]["backtest"]["cost1"]["pnl"],
            "oos_cost1_mdd": row["oos"]["backtest"]["cost1"]["mdd"],
            "oos_cost1_trades": row["oos"]["backtest"]["cost1"]["trades"],
            "oos_trade_precision": row["oos"]["direction"]["trade_precision"],
            "oos_balanced_precision": row["oos"]["direction"]["balanced_trade_precision"],
        }
        for row in rows
    ]).sort_values(["family", "val_score"], ascending=[True, False]).to_csv(grid_csv, index=False)

    summary_path = args.out_dir / "alpha5_23_hgb_direction_refined_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    print(json.dumps({
        "stage": "complete",
        "summary": str(summary_path),
        "grid": str(grid_csv),
        "best_two_stage": {
            "architecture": summary["best_two_stage"]["architecture"] if summary.get("best_two_stage") else None,
            "validation_cost1_pnl": summary["best_two_stage"]["validation"]["backtest"]["cost1"]["pnl"] if summary.get("best_two_stage") else None,
            "oos_cost1_pnl": summary["best_two_stage"]["oos"]["backtest"]["cost1"]["pnl"] if summary.get("best_two_stage") else None,
            "oos_trades": summary["best_two_stage"]["oos"]["backtest"]["cost1"]["trades"] if summary.get("best_two_stage") else None,
        },
        "best_ovr": {
            "architecture": summary["best_ovr"]["architecture"] if summary.get("best_ovr") else None,
            "validation_cost1_pnl": summary["best_ovr"]["validation"]["backtest"]["cost1"]["pnl"] if summary.get("best_ovr") else None,
            "oos_cost1_pnl": summary["best_ovr"]["oos"]["backtest"]["cost1"]["pnl"] if summary.get("best_ovr") else None,
            "oos_trades": summary["best_ovr"]["oos"]["backtest"]["cost1"]["trades"] if summary.get("best_ovr") else None,
        },
    }, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
