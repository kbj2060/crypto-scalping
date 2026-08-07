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
from scripts.train_eval_alpha5_5_lgbm_supervised_parent_20260518 import _decide_actions, _predict_proba_3  # noqa: E402
from scripts.train_eval_alpha5_13_hgb_single_20260518 import _eval_candidate  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402
from scripts.tune_alpha5_9_hgb_action_master_20260518 import HGBSpec, _fit_hgb, _hgb_specs  # noqa: E402


MODEL_ID = "alpha5_19_hgb_meta_gate_20260518"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_high_quality_training_data_20260518"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_BASELINE = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_single_20260518/regime4_core_deeper_alpha5_13_hgb_parent.joblib"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_19_hgb_meta_gate_20260518"


def _x(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return prepare_features(frame, side_hint=0, close=_close(frame), feature_cols=cols)


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


def _meta_feature_matrix(frame: pd.DataFrame, proba: np.ndarray, actions: np.ndarray) -> pd.DataFrame:
    p_flat = proba[:, 0]
    p_long = proba[:, 1]
    p_short = proba[:, 2]
    p_trade = np.maximum(p_long, p_short)
    p_margin = np.abs(p_long - p_short)
    p_entropy = -(proba * np.log(np.clip(proba, 1e-12, None))).sum(axis=1)
    pred_is_long = (actions == 1).astype(np.float64)
    pred_is_short = (actions == 2).astype(np.float64)
    pred_is_trade = (actions != 0).astype(np.float64)

    cols = {
        "meta_p_flat": p_flat,
        "meta_p_long": p_long,
        "meta_p_short": p_short,
        "meta_p_trade": p_trade,
        "meta_p_margin": p_margin,
        "meta_p_entropy": p_entropy,
        "meta_pred_long": pred_is_long,
        "meta_pred_short": pred_is_short,
        "meta_pred_trade": pred_is_trade,
    }
    out = pd.DataFrame(cols, index=frame.index)

    passthrough = [
        "whale_retail_ratio",
        "smart_money_flow",
        "net_taker_ratio",
        "taker_acceleration",
        "trade_intensity",
        "volatility_z",
        "rsi",
        "mtf_trend_1h",
        "mtf_trend_4h",
        "breakout_strength",
        "ofi_acceleration",
        "garch_vol_z",
        "liquidity_vacuum",
        "execution_quality",
        "funding_pressure",
        "crowding_pressure",
        "ai_dir_edge",
        "ai_dir_entropy",
        "tp_sl_action_score",
        "clean_regime4_2024_unsup_v1_bull_prob",
        "clean_regime4_2024_unsup_v1_bear_prob",
        "clean_regime4_2024_unsup_v1_chop_prob",
        "clean_regime4_2024_unsup_v1_whipsaw_prob",
        "clean_regime4_2024_unsup_v1_confidence",
        "clean_regime4_2024_unsup_v1_entropy",
        "clean_regime4_2024_unsup_v1_margin",
        "clean_regime4_2024_unsup_v1_factor_trend",
        "clean_regime4_2024_unsup_v1_factor_flow",
        "clean_regime4_2024_unsup_v1_factor_vol",
        "clean_regime4_2024_unsup_v1_trend_bias",
        "clean_regime4_2024_unsup_v1_risk_off_prob",
        "clean_regime4_2024_unsup_v1_transition_risk",
    ]
    for col in passthrough:
        if col in frame.columns:
            out[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).to_numpy(np.float64)
    return out


def _meta_target(frame: pd.DataFrame, actions: np.ndarray) -> np.ndarray:
    label_action = pd.to_numeric(frame["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    tp_first = pd.to_numeric(frame["meta_tp_first"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
    profitable = pd.to_numeric(frame["meta_is_profitable"], errors="coerce").fillna(0).to_numpy(np.int64) == 1
    return ((actions != 0) & (actions == label_action) & tp_first & profitable).astype(np.int64)


def _meta_weights(frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    base = pd.to_numeric(frame["label_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    consensus = pd.to_numeric(frame["label_consensus"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    edge_gap = pd.to_numeric(frame["meta_edge_gap"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    w = np.clip(base, 1e-4, None) * (0.80 + 0.40 * np.clip(consensus, 0.0, 1.0))
    w *= 1.0 + 0.03 * np.clip(edge_gap, 0.0, 10.0)
    return w * _balanced_weights(y)


def _gate_actions(actions: np.ndarray, meta_proba: np.ndarray, threshold: float) -> np.ndarray:
    out = actions.copy()
    trade = out != 0
    out[trade & (meta_proba < float(threshold))] = 0
    return out


def _oof_slices(n: int) -> list[tuple[slice, slice]]:
    a = int(n * 0.50)
    b = int(n * 0.65)
    c = int(n * 0.80)
    return [
        (slice(0, a), slice(a, b)),
        (slice(0, b), slice(b, c)),
        (slice(0, c), slice(c, n)),
    ]


def _spec_by_name(name: str) -> HGBSpec:
    for spec in _hgb_specs():
        if spec.name == name:
            return spec
    raise ValueError(name)


def main() -> None:
    p = argparse.ArgumentParser(description="Train HGB meta gate on top of fixed regime4_core deeper HGB baseline.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--raw-2025-csv", type=Path, default=DEFAULT_RAW_2025)
    p.add_argument("--raw-2026-csv", type=Path, default=DEFAULT_RAW_2026)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--baseline-artifact", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--base-prob-thresholds", default="0.90,0.91,0.92,0.93")
    p.add_argument("--base-margin-thresholds", default="0.00")
    p.add_argument("--meta-thresholds", default="0.35,0.40,0.45,0.50,0.55")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--seed", type=int, default=51901)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_2025 = _read(args.raw_2025_csv)
    raw_2026 = _read(args.raw_2026_csv)
    audit = _verify_state24_sticky090_inputs(raw_2025, raw_2026, args.manifest, args.clean4_report)
    base_payload = joblib.load(args.baseline_artifact)
    base_spec = _spec_by_name(str(base_payload["hgb"]["name"]))
    base_prob = float(base_payload["decision"]["prob"])
    base_margin = float(base_payload["decision"]["margin"])
    feature_cols = list(base_payload["feature_cols"])

    train_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_train.parquet")
    val_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_val.parquet")
    oos_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_oos.parquet")
    train_fit = train_df[train_df["label_train_keep"] == 1].reset_index(drop=True)
    x_train = _x(train_fit, feature_cols)
    x_val = _x(val_df, feature_cols)
    x_oos = _x(oos_df, feature_cols)
    y_train = pd.to_numeric(train_fit["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    y_val = pd.to_numeric(val_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    y_oos = pd.to_numeric(oos_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    w_train = pd.to_numeric(train_fit["label_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)

    print(json.dumps({
        "stage": "start",
        "model_id": MODEL_ID,
        "baseline_artifact": str(args.baseline_artifact),
        "baseline_hgb": base_spec.name,
        "baseline_prob_default": base_prob,
        "baseline_margin_default": base_margin,
        "baseline_prob_grid": [float(x.strip()) for x in str(args.base_prob_thresholds).split(",") if x.strip()],
        "baseline_margin_grid": [float(x.strip()) for x in str(args.base_margin_thresholds).split(",") if x.strip()],
        "rows": {"train_fit": int(len(train_fit)), "validation": int(len(val_df)), "oos": int(len(oos_df))},
        "feature_count": int(len(feature_cols)),
        "audit_expected_model_found": audit.get("expected_model_found_in_manifest"),
    }, ensure_ascii=False, default=_json_default), flush=True)
    base_prob_grid = [float(x.strip()) for x in str(args.base_prob_thresholds).split(",") if x.strip()]
    base_margin_grid = [float(x.strip()) for x in str(args.base_margin_thresholds).split(",") if x.strip()]
    meta_threshold_grid = [float(x.strip()) for x in str(args.meta_thresholds).split(",") if x.strip()]

    fold_payloads: list[dict[str, Any]] = []
    slices = _oof_slices(len(train_fit))
    for fold_i, (tr_slice, te_slice) in enumerate(slices, start=1):
        x_tr = x_train.iloc[tr_slice].reset_index(drop=True)
        y_tr = y_train[tr_slice]
        w_tr = w_train[tr_slice]
        x_te = x_train.iloc[te_slice].reset_index(drop=True)
        f_te = train_fit.iloc[te_slice].reset_index(drop=True)
        model = _fit_hgb(x_tr, y_tr, w_tr, base_spec, int(args.seed + fold_i))
        proba = _predict_proba_3(model, x_te)
        fold_payloads.append({"frame": f_te, "proba": proba, "fold": fold_i, "pred_rows": len(x_te), "train_rows": len(x_tr)})

    full_base_model = _fit_hgb(x_train, y_train, w_train, base_spec, int(args.seed + 999))
    val_base_proba_full = _predict_proba_3(full_base_model, x_val)
    oos_base_proba_full = _predict_proba_3(full_base_model, x_oos)

    meta_rows: list[dict[str, Any]] = []
    for base_prob_try in base_prob_grid:
        for base_margin_try in base_margin_grid:
            oof_parts: list[pd.DataFrame] = []
            for fold_payload in fold_payloads:
                f_te = fold_payload["frame"]
                proba = fold_payload["proba"]
                actions = _decide_actions(proba, base_prob_try, base_margin_try)
                meta_x = _meta_feature_matrix(f_te, proba, actions)
                meta_y = _meta_target(f_te, actions)
                meta_x["meta_target"] = meta_y
                meta_x["meta_action"] = actions
                meta_x["meta_weight_base"] = pd.to_numeric(f_te["label_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
                meta_x["meta_label_consensus"] = pd.to_numeric(f_te["label_consensus"], errors="coerce").fillna(0.0).to_numpy(np.float64)
                meta_x["meta_edge_gap_raw"] = pd.to_numeric(f_te["meta_edge_gap"], errors="coerce").fillna(0.0).to_numpy(np.float64)
                meta_x = meta_x[meta_x["meta_action"] != 0].reset_index(drop=True)
                oof_parts.append(meta_x)
                print(json.dumps({
                    "stage": "oof_fold",
                    "fold": fold_payload["fold"],
                    "base_prob": base_prob_try,
                    "base_margin": base_margin_try,
                    "train_rows": int(fold_payload["train_rows"]),
                    "pred_rows": int(fold_payload["pred_rows"]),
                    "trade_rows": int(len(meta_x)),
                    "positive_ratio": float(meta_x["meta_target"].mean()) if len(meta_x) else 0.0,
                }, ensure_ascii=False), flush=True)

            meta_train = pd.concat(oof_parts, axis=0, ignore_index=True) if oof_parts else pd.DataFrame()
            if meta_train.empty:
                continue
            meta_y = meta_train.pop("meta_target").to_numpy(np.int64)
            meta_train.pop("meta_action")
            meta_weight_base = pd.to_numeric(meta_train.pop("meta_weight_base"), errors="coerce").fillna(0.0).to_numpy(np.float64)
            meta_consensus = pd.to_numeric(meta_train.pop("meta_label_consensus"), errors="coerce").fillna(0.0).to_numpy(np.float64)
            meta_edge_gap = pd.to_numeric(meta_train.pop("meta_edge_gap_raw"), errors="coerce").fillna(0.0).to_numpy(np.float64)
            meta_w = np.clip(meta_weight_base, 1e-4, None) * (0.80 + 0.40 * np.clip(meta_consensus, 0.0, 1.0))
            meta_w *= 1.0 + 0.03 * np.clip(meta_edge_gap, 0.0, 10.0)
            meta_w *= _balanced_weights(meta_y)

            val_base_actions = _decide_actions(val_base_proba_full, base_prob_try, base_margin_try)
            val_meta_x = _meta_feature_matrix(val_df, val_base_proba_full, val_base_actions)
            oos_base_actions = _decide_actions(oos_base_proba_full, base_prob_try, base_margin_try)
            oos_meta_x = _meta_feature_matrix(oos_df, oos_base_proba_full, oos_base_actions)

            for spec_i, meta_spec in enumerate([_spec_by_name("regularized"), _spec_by_name("deeper")], start=1):
                print(json.dumps({
                    "stage": "fit_meta",
                    "meta_hgb": meta_spec.name,
                    "base_prob": base_prob_try,
                    "base_margin": base_margin_try,
                    "meta_train_rows": int(len(meta_train)),
                    "meta_positive_ratio": float(np.mean(meta_y)),
                }, ensure_ascii=False), flush=True)
                meta_model = _fit_hgb(meta_train, meta_y, meta_w, meta_spec, int(args.seed + 2000 + int(base_prob_try * 100) + spec_i))
                val_meta_proba = _binary_proba(meta_model, val_meta_x)
                oos_meta_proba = _binary_proba(meta_model, oos_meta_x)

                best_val: dict[str, Any] | None = None
                for thr in meta_threshold_grid:
                    gated_val = _gate_actions(val_base_actions, val_meta_proba, thr)
                    val_eval = _eval_candidate(
                        val_df, gated_val, fee=float(args.fee), slip=float(args.slip),
                        exposure=float(args.unit_exposure), max_hold=int(args.max_hold_bars), labels=y_val
                    )
                    val_eval["meta_threshold"] = float(thr)
                    val_eval["base_prob"] = float(base_prob_try)
                    val_eval["base_margin"] = float(base_margin_try)
                    if best_val is None or float(val_eval["score"]) > float(best_val["score"]):
                        best_val = val_eval
                assert best_val is not None

                gated_oos = _gate_actions(oos_base_actions, oos_meta_proba, float(best_val["meta_threshold"]))
                oos_eval = _eval_candidate(
                    oos_df, gated_oos, fee=float(args.fee), slip=float(args.slip),
                    exposure=float(args.unit_exposure), max_hold=int(args.max_hold_bars), labels=y_oos
                )

                artifact = args.out_dir / f"bp{int(base_prob_try*1000):03d}_bm{int(base_margin_try*1000):03d}_{meta_spec.name}_alpha5_19_hgb_meta_gate.joblib"
                joblib.dump({
                    "model_id": MODEL_ID,
                    "baseline_artifact": str(args.baseline_artifact),
                    "baseline_hgb": base_spec.name,
                    "baseline_decision": {"prob": float(base_prob_try), "margin": float(base_margin_try)},
                    "base_model": full_base_model,
                    "meta_model": meta_model,
                    "feature_cols": feature_cols,
                    "meta_feature_cols": list(meta_train.columns),
                    "meta_hgb": meta_spec.name,
                    "meta_threshold": float(best_val["meta_threshold"]),
                }, artifact)

                row = {
                    "base_prob": float(base_prob_try),
                    "base_margin": float(base_margin_try),
                    "meta_hgb": meta_spec.name,
                    "meta_train_rows": int(len(meta_train)),
                    "meta_positive_ratio": float(np.mean(meta_y)),
                    "validation": best_val,
                    "oos": oos_eval,
                    "artifact": str(artifact),
                }
                meta_rows.append(row)
                print(json.dumps({
                    "stage": "candidate",
                    "base_prob": base_prob_try,
                    "base_margin": base_margin_try,
                    "meta_hgb": meta_spec.name,
                    "meta_train_rows": len(meta_train),
                    "meta_positive_ratio": float(np.mean(meta_y)),
                    "val_threshold": best_val["meta_threshold"],
                    "val_score": best_val["score"],
                    "val_cost1": best_val["backtest"]["cost1"],
                    "val_direction": best_val["direction"],
                    "oos_score": oos_eval["score"],
                    "oos_cost1": oos_eval["backtest"]["cost1"],
                    "oos_direction": oos_eval["direction"],
                }, ensure_ascii=False, default=_json_default), flush=True)

    best = max(meta_rows, key=lambda r: float(r["validation"]["score"]))
    summary = {
        "model_id": MODEL_ID,
        "design": "Fixed regime4_core deeper HGB baseline plus HGB meta gate trained on OOF trade predictions.",
        "baseline_artifact": str(args.baseline_artifact),
        "experiments": meta_rows,
        "best": best,
    }
    summary_path = args.out_dir / "alpha5_19_hgb_meta_gate_summary.json"
    grid_path = args.out_dir / "alpha5_19_hgb_meta_gate_grid.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame([
        {
            "base_prob": r["base_prob"],
            "base_margin": r["base_margin"],
            "meta_hgb": r["meta_hgb"],
            "meta_train_rows": r["meta_train_rows"],
            "meta_positive_ratio": r["meta_positive_ratio"],
            "val_score": r["validation"]["score"],
            "val_meta_threshold": r["validation"]["meta_threshold"],
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
        for r in meta_rows
    ]).sort_values("val_score", ascending=False).to_csv(grid_path, index=False)
    print(json.dumps({
        "stage": "complete",
        "summary": str(summary_path),
        "grid": str(grid_path),
        "best": {
            "meta_hgb": best["meta_hgb"],
            "val_score": best["validation"]["score"],
            "oos_score": best["oos"]["score"],
            "oos_cost1": best["oos"]["backtest"]["cost1"],
            "oos_direction": best["oos"]["direction"],
        },
    }, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
