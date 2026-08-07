#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    FullyLearnedGovernorConfig,
    build_training_set,
)
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    CLEAN4_PREFIX,
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_EVAL,
    DEFAULT_PREPROCESS_MANIFEST,
    DEFAULT_TRAIN,
    REGIMES,
    ROUTER_COLS,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_alpha5_4_single_conditioned_dqn_20260518 import (  # noqa: E402
    _feature_cols as _alpha5_feature_cols,
)
from scripts.train_eval_alpha5_5_lgbm_supervised_parent_20260518 import (  # noqa: E402
    _backtest_actions,
    _decide_actions,
    _predict_proba_3,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    _json_default,
    _read,
)


MODEL_ID = "alpha5_7_hgb_action_label_master_state24_sticky090_20260518"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_7_hgb_action_label_master_20260518"


def _split(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = frame.copy()
    if start:
        out = out[out["timestamp"] >= pd.Timestamp(start)]
    if end:
        out = out[out["timestamp"] < pd.Timestamp(end)]
    return out.reset_index(drop=True)


def _cfg_unit(cash_score: float, adverse: float, hold: float) -> FullyLearnedGovernorConfig:
    return FullyLearnedGovernorConfig(
        notional_buckets=(1.0,),
        leverage_buckets=(1.5,),
        take_profit_buckets=(0.006, 0.010, 0.018, 0.030, 0.050, 0.090, 0.180),
        stop_loss_buckets=(0.004, 0.006, 0.010, 0.016, 0.024, 0.035),
        max_hold_buckets=(6, 12, 24, 48, 96, 192, 288),
        cooldown_buckets=(0,),
        max_train_horizon_bars=288,
        adverse_penalty=float(adverse),
        size_penalty=0.0,
        hold_penalty=float(hold),
        turnover_bonus=0.0010,
        cash_score=float(cash_score),
    )


def _cfg_l1_size_search() -> FullyLearnedGovernorConfig:
    return FullyLearnedGovernorConfig(
        notional_buckets=(0.20, 0.32, 0.50, 0.75, 1.05, 1.45, 2.00, 2.70, 3.60),
        leverage_buckets=(1.5, 2.0, 3.0, 4.0, 5.0),
        take_profit_buckets=(0.007, 0.011, 0.018, 0.030, 0.050, 0.090, 0.180, 0.450, 0.900),
        stop_loss_buckets=(0.004, 0.006, 0.009, 0.014, 0.022, 0.035, 0.055),
        max_hold_buckets=(6, 12, 24, 48, 96, 192, 288),
        cooldown_buckets=(0, 1, 3, 6, 12, 24, 48),
        max_train_horizon_bars=288,
        adverse_penalty=2.45,
        size_penalty=0.180,
        hold_penalty=0.042,
        turnover_bonus=0.0012,
        cash_score=0.020,
    )


def _specs() -> dict[str, FullyLearnedGovernorConfig]:
    return {
        "A1_unit_cash012": _cfg_unit(0.012, 1.80, 0.020),
        "A2_unit_cash018": _cfg_unit(0.018, 2.20, 0.030),
        "A3_size_search_l1": _cfg_l1_size_search(),
    }


def _valid_indices(n_rows: int, horizon: int, stride: int) -> np.ndarray:
    return np.arange(0, max(0, int(n_rows) - int(horizon) - 1), max(1, int(stride)), dtype=np.int64)


def _feature_contract(train_df: pd.DataFrame, eval_df: pd.DataFrame, top_k: int, include_future: bool) -> list[str]:
    return _alpha5_feature_cols(
        train_df,
        eval_df,
        include_future_regime_pred=bool(include_future),
        feature_top_k=int(top_k),
        feature_select_horizon=48,
    )


def _clean_x(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return frame.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _fit_action_hgb(x: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray, seed: int) -> Any:
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=280,
            learning_rate=0.045,
            max_leaf_nodes=31,
            min_samples_leaf=60,
            l2_regularization=0.08,
            early_stopping=False,
            random_state=int(seed),
        ),
    )
    model.fit(x, y, histgradientboostingclassifier__sample_weight=sample_weight)
    return model


def _class_balanced_weight(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    out = np.ones(len(y), dtype=np.float64)
    classes, counts = np.unique(y, return_counts=True)
    total = float(max(len(y), 1))
    for cls, count in zip(classes, counts):
        out[y == int(cls)] = total / (float(len(classes)) * max(float(count), 1.0))
    return out


def _sample_weight(y: dict[str, np.ndarray], mode: str) -> np.ndarray:
    action = np.asarray(y["action"], dtype=np.int64)
    quality = np.asarray(y["quality"], dtype=np.float64)
    if mode == "none":
        return np.ones(len(action), dtype=np.float64)
    q = np.clip(np.abs(quality), 0.02, 1.0)
    trade = np.where(action == ACTION_CASH, 0.45, 1.0)
    w = np.maximum(q, trade)
    if mode == "balanced":
        w = w * _class_balanced_weight(action)
    return w.astype(np.float64)


def _label_report(y: dict[str, np.ndarray], frame: pd.DataFrame, valid_idx: np.ndarray) -> dict[str, Any]:
    action = np.asarray(y["action"], dtype=np.int64)
    quality = np.asarray(y["quality"], dtype=np.float64)
    probs = frame.iloc[valid_idx][ROUTER_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ridx = np.argmax(probs, axis=1) if len(probs) else np.zeros(0, dtype=np.int64)
    by_regime: dict[str, Any] = {}
    for i, name in enumerate(REGIMES):
        m = ridx == i
        by_regime[name] = {
            "n": int(m.sum()),
            "cash": int(np.sum(action[m] == ACTION_CASH)),
            "long": int(np.sum(action[m] == ACTION_LONG)),
            "short": int(np.sum(action[m] == ACTION_SHORT)),
            "quality_mean": float(np.mean(quality[m])) if np.any(m) else 0.0,
        }
    return {
        "rows": int(len(action)),
        "action_counts": {
            "cash": int(np.sum(action == ACTION_CASH)),
            "long": int(np.sum(action == ACTION_LONG)),
            "short": int(np.sum(action == ACTION_SHORT)),
        },
        "trade_ratio": float(np.mean(action != ACTION_CASH)),
        "quality": {
            "mean": float(np.mean(quality)),
            "p50": float(np.quantile(quality, 0.50)),
            "p90": float(np.quantile(quality, 0.90)),
            "p95": float(np.quantile(quality, 0.95)),
        },
        "by_regime": by_regime,
    }


def _metrics_for_proba(frame: pd.DataFrame, proba: np.ndarray, *, prob_threshold: float, margin_threshold: float, fee: float, slip: float, unit_exposure: float, max_hold_bars: int) -> dict[str, Any]:
    actions = _decide_actions(proba, prob_threshold, margin_threshold)
    return {
        f"cost{m}": _backtest_actions(
            frame,
            actions,
            fee=float(fee) * float(m),
            slip=float(slip) * float(m),
            unit_exposure=float(unit_exposure),
            max_hold_bars=int(max_hold_bars),
        )
        for m in (1, 2, 3)
    }


def _score(metrics: dict[str, Any]) -> float:
    c1, c2, c3 = metrics["cost1"], metrics["cost2"], metrics["cost3"]
    trades = int(c1.get("trades", 0))
    if trades < 12:
        return -1e6 + float(c1.get("pnl", 0.0))
    return (
        float(c1["pnl"])
        + 0.40 * float(c2["pnl"])
        + 0.20 * float(c3["pnl"])
        - 0.30 * abs(float(c1["mdd"]))
        - 1.5 * max(0.0, float(c1["trades_per_day"]) - 8.0)
    )


def _grid(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Alpha5.7 action-label-only HGB direction master.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--train-end", default="2025-10-01")
    p.add_argument("--val-start", default="2025-10-01")
    p.add_argument("--val-end", default="2026-01-01")
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--feature-top-k", type=int, default=64)
    p.add_argument("--include-future-regime-pred", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--variants", default="A1_unit_cash012,A2_unit_cash018,A3_size_search_l1")
    p.add_argument("--weight-modes", default="quality,balanced")
    p.add_argument("--prob-thresholds", default="0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    p.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.08,0.12,0.16,0.20,0.25")
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=5701)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train_df = _split(train_all, None, args.train_end)
    val_df = _split(train_all, args.val_start, args.val_end)
    audit = _verify_state24_sticky090_inputs(train_all, eval_df, args.manifest, args.clean4_report)
    feature_cols = _feature_contract(train_all, eval_df, int(args.feature_top_k), bool(args.include_future_regime_pred))
    legacy = [c for c in feature_cols if c.startswith("clean_regime_2024_unsup_v4_")]
    if legacy:
        raise RuntimeError("legacy clean v4 selected: " + ",".join(legacy[:20]))

    print(
        json.dumps(
            {
                "stage": "start",
                "model_id": MODEL_ID,
                "train_rows": len(train_df),
                "validation_rows": len(val_df),
                "oos_rows": len(eval_df),
                "feature_count": len(feature_cols),
                "clean4_feature_count": int(sum(c.startswith(CLEAN4_PREFIX) for c in feature_cols)),
                "future_regime4_feature_count": int(sum(c.startswith("regime4_pred_") for c in feature_cols)),
                "audit": {
                    "expected_model_found_in_manifest": audit.get("expected_model_found_in_manifest"),
                    "legacy_v4_count": audit.get("legacy_v4_count"),
                    "router_missing": audit.get("router_missing"),
                },
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )

    x_val = _clean_x(val_df, feature_cols)
    x_eval = _clean_x(eval_df, feature_cols)
    specs = _specs()
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    weight_modes = [v.strip() for v in str(args.weight_modes).split(",") if v.strip()]
    rows: list[dict[str, Any]] = []
    for vi, variant in enumerate(variants):
        cfg = specs[variant]
        print(json.dumps({"stage": "build_action_labels", "variant": variant, "cfg": asdict(cfg)}, ensure_ascii=False), flush=True)
        x_train, y_train, meta = build_training_set(
            train_df,
            cfg=cfg,
            stride_bars=int(args.stride),
            batch_size=512,
            feature_cols=feature_cols,
        )
        valid_idx = _valid_indices(len(train_df), int(cfg.max_train_horizon_bars), int(args.stride))
        label_report = _label_report(y_train, train_df, valid_idx)
        print(json.dumps({"stage": "label_report", "variant": variant, "label_report": label_report}, ensure_ascii=False, default=_json_default), flush=True)
        y_action = np.asarray(y_train["action"], dtype=np.int64)
        x_action = _clean_x(x_train, feature_cols)
        for wi, weight_mode in enumerate(weight_modes):
            print(json.dumps({"stage": "fit_action_hgb", "variant": variant, "weight_mode": weight_mode, "rows": len(x_action)}, ensure_ascii=False), flush=True)
            model = _fit_action_hgb(x_action, y_action, _sample_weight(y_train, weight_mode), int(args.seed) + vi * 100 + wi * 10)
            val_proba = _predict_proba_3(model, x_val)
            eval_proba = _predict_proba_3(model, x_eval)
            best_row: dict[str, Any] | None = None
            for prob_th in _grid(args.prob_thresholds):
                for margin_th in _grid(args.margin_thresholds):
                    val_metrics = _metrics_for_proba(
                        val_df,
                        val_proba,
                        prob_threshold=prob_th,
                        margin_threshold=margin_th,
                        fee=float(args.fee),
                        slip=float(args.slip),
                        unit_exposure=float(args.unit_exposure),
                        max_hold_bars=int(args.max_hold_bars),
                    )
                    score = _score(val_metrics)
                    candidate = {
                        "variant": variant,
                        "weight_mode": weight_mode,
                        "prob_threshold": float(prob_th),
                        "margin_threshold": float(margin_th),
                        "score": float(score),
                        "validation_metrics": val_metrics,
                    }
                    if best_row is None or score > float(best_row["score"]):
                        best_row = candidate
            assert best_row is not None
            oos_metrics = _metrics_for_proba(
                eval_df,
                eval_proba,
                prob_threshold=float(best_row["prob_threshold"]),
                margin_threshold=float(best_row["margin_threshold"]),
                fee=float(args.fee),
                slip=float(args.slip),
                unit_exposure=float(args.unit_exposure),
                max_hold_bars=int(args.max_hold_bars),
            )
            artifact = args.out_dir / f"{variant}_{weight_mode}_action_hgb.joblib"
            joblib.dump(
                {
                    "model_id": MODEL_ID,
                    "variant": variant,
                    "weight_mode": weight_mode,
                    "model": model,
                    "feature_cols": feature_cols,
                    "label_cfg": asdict(cfg),
                    "label_report": label_report,
                    "selected_thresholds": {
                        "prob_threshold": best_row["prob_threshold"],
                        "margin_threshold": best_row["margin_threshold"],
                        "max_hold_bars": int(args.max_hold_bars),
                    },
                },
                artifact,
            )
            row = {
                **best_row,
                "oos_metrics": oos_metrics,
                "label_report": label_report,
                "train_meta": meta,
                "artifact": str(artifact),
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "stage": "candidate_complete",
                        "variant": variant,
                        "weight_mode": weight_mode,
                        "selected": {
                            "prob_threshold": best_row["prob_threshold"],
                            "margin_threshold": best_row["margin_threshold"],
                        },
                        "validation_cost1": best_row["validation_metrics"]["cost1"],
                        "oos_cost1": oos_metrics["cost1"],
                        "artifact": str(artifact),
                    },
                    ensure_ascii=False,
                    default=_json_default,
                ),
                flush=True,
            )

    best = max(rows, key=lambda r: float(r["score"]))
    summary = {
        "model_id": MODEL_ID,
        "design": "HGB action direction master only. Labels are lifecycle-derived but only action cash/long/short is trained; TP/SL/notional/leverage/hold are intentionally excluded for DSAC.",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "split": {
            "train": [str(train_df["timestamp"].iloc[0]), str(train_df["timestamp"].iloc[-1])],
            "validation": [str(val_df["timestamp"].iloc[0]), str(val_df["timestamp"].iloc[-1])],
            "oos": [str(eval_df["timestamp"].iloc[0]), str(eval_df["timestamp"].iloc[-1])],
        },
        "feature_contract": {
            "feature_cols": feature_cols,
            "feature_count": len(feature_cols),
            "clean4_feature_count": int(sum(c.startswith(CLEAN4_PREFIX) for c in feature_cols)),
            "future_regime4_feature_count": int(sum(c.startswith("regime4_pred_") for c in feature_cols)),
            "legacy_clean_v4_count": int(sum(c.startswith("clean_regime_2024_unsup_v4_") for c in feature_cols)),
        },
        "state24_sticky090_audit": audit,
        "experiments": rows,
        "best": best,
        "artifacts": {
            "out_dir": str(args.out_dir),
            "summary": str(args.out_dir / "alpha5_7_hgb_action_label_master_summary.json"),
            "grid": str(args.out_dir / "alpha5_7_hgb_action_label_master_grid.csv"),
        },
    }
    summary_path = args.out_dir / "alpha5_7_hgb_action_label_master_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "variant": r["variant"],
                "weight_mode": r["weight_mode"],
                "score": r["score"],
                "prob_threshold": r["prob_threshold"],
                "margin_threshold": r["margin_threshold"],
                "val_cost1_pnl": r["validation_metrics"]["cost1"]["pnl"],
                "val_cost1_mdd": r["validation_metrics"]["cost1"]["mdd"],
                "val_cost1_trades_day": r["validation_metrics"]["cost1"]["trades_per_day"],
                "oos_cost1_pnl": r["oos_metrics"]["cost1"]["pnl"],
                "oos_cost1_mdd": r["oos_metrics"]["cost1"]["mdd"],
                "oos_cost1_trades_day": r["oos_metrics"]["cost1"]["trades_per_day"],
                "oos_cost2_pnl": r["oos_metrics"]["cost2"]["pnl"],
                "oos_cost3_pnl": r["oos_metrics"]["cost3"]["pnl"],
                "label_trade_ratio": r["label_report"]["trade_ratio"],
                "artifact": r["artifact"],
            }
            for r in rows
        ]
    ).to_csv(args.out_dir / "alpha5_7_hgb_action_label_master_grid.csv", index=False)
    print(json.dumps({"stage": "complete", "summary": str(summary_path), "best": {"variant": best["variant"], "weight_mode": best["weight_mode"]}}, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
