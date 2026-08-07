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
    prepare_features,
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
    _close,
    _json_default,
    _read,
)


MODEL_ID = "alpha5_8_hgb_action_feature_contract_compare_state24_sticky090_20260518"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_8_hgb_action_feature_contract_compare_20260518"
ALPHA4_PARENT = ROOT / "tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/artifacts/hgb/parent.pkl"


def _split(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = frame.copy()
    if start:
        out = out[out["timestamp"] >= pd.Timestamp(start)]
    if end:
        out = out[out["timestamp"] < pd.Timestamp(end)]
    return out.reset_index(drop=True)


def _cfg_action_l1() -> FullyLearnedGovernorConfig:
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


def _valid_indices(n_rows: int, horizon: int, stride: int) -> np.ndarray:
    return np.arange(0, max(0, int(n_rows) - int(horizon) - 1), max(1, int(stride)), dtype=np.int64)


def _mapped_clean4_col(old: str) -> str | None:
    prefix = "clean_regime_2024_unsup_v4_"
    if not old.startswith(prefix):
        return old
    suffix = old[len(prefix) :]
    direct = {
        "bear_prob": "bear_prob",
        "bull_prob": "bull_prob",
        "chop_prob": "chop_prob",
        "whipsaw_prob": "whipsaw_prob",
        "confidence": "confidence",
        "entropy": "entropy",
        "factor_crowding": "factor_crowding",
        "factor_flow": "factor_flow",
        "factor_liquidity": "factor_liquidity",
        "factor_trend": "factor_trend",
        "factor_vol": "factor_vol",
        "risk_off_prob": "risk_off_prob",
        "transition_risk": "transition_risk",
        "trend_bias": "trend_bias",
    }
    if suffix in direct:
        return f"{CLEAN4_PREFIX}{direct[suffix]}"
    if suffix == "normal_prob":
        return None
    if suffix.startswith("cluster") or suffix == "state_code":
        return None
    return None


def _alpha4_mapped_features(train: pd.DataFrame, eval_df: pd.DataFrame, *, include_future: bool) -> list[str]:
    parent = joblib.load(ALPHA4_PARENT)
    common = set(train.columns) & set(eval_df.columns)
    out: list[str] = []
    for old in parent["feature_cols"]:
        mapped = _mapped_clean4_col(str(old))
        if mapped is None:
            continue
        if mapped in out:
            continue
        if mapped in common or mapped == "side_hint" or mapped.startswith(("mom_", "abs_mom_")):
            out.append(mapped)
    extras = [
        f"{CLEAN4_PREFIX}trend_prob",
        f"{CLEAN4_PREFIX}micro_prob",
        f"{CLEAN4_PREFIX}directional_bias",
        f"{CLEAN4_PREFIX}range_prob",
        f"{CLEAN4_PREFIX}instability_prob",
        f"{CLEAN4_PREFIX}margin",
        "tp_sl_action_score",
    ]
    for col in extras:
        if col in common and col not in out:
            out.append(col)
    if include_future:
        for col in sorted(c for c in common if c.startswith("regime4_pred_")):
            if col not in out:
                out.append(col)
    return out


def _feature_contract(train: pd.DataFrame, eval_df: pd.DataFrame, mode: str, include_future: bool, top_k: int) -> list[str]:
    if mode == "alpha5_selected":
        return _alpha5_feature_cols(train, eval_df, include_future_regime_pred=include_future, feature_top_k=top_k, feature_select_horizon=48)
    if mode == "alpha4_mapped":
        return _alpha4_mapped_features(train, eval_df, include_future=include_future)
    if mode == "alpha4_mapped_no_future":
        return _alpha4_mapped_features(train, eval_df, include_future=False)
    raise ValueError(f"unknown feature mode: {mode}")


def _x(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return prepare_features(frame, side_hint=0, close=_close(frame), feature_cols=cols)


def _fit_hgb(x: pd.DataFrame, y: np.ndarray, w: np.ndarray, seed: int) -> Any:
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
    model.fit(x, y, histgradientboostingclassifier__sample_weight=w)
    return model


def _class_balanced_weight(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    out = np.ones(len(y), dtype=np.float64)
    classes, counts = np.unique(y, return_counts=True)
    total = float(max(len(y), 1))
    for cls, count in zip(classes, counts):
        out[y == int(cls)] = total / (float(len(classes)) * max(float(count), 1.0))
    return out


def _weights(y: dict[str, np.ndarray], mode: str) -> np.ndarray:
    action = np.asarray(y["action"], dtype=np.int64)
    quality = np.asarray(y["quality"], dtype=np.float64)
    if mode == "none":
        return np.ones(len(action), dtype=np.float64)
    base = np.maximum(np.clip(np.abs(quality), 0.02, 1.0), np.where(action == ACTION_CASH, 0.45, 1.0))
    if mode == "balanced":
        base = base * _class_balanced_weight(action)
    return base.astype(np.float64)


def _label_report(y: dict[str, np.ndarray], frame: pd.DataFrame, valid_idx: np.ndarray) -> dict[str, Any]:
    action = np.asarray(y["action"], dtype=np.int64)
    quality = np.asarray(y["quality"], dtype=np.float64)
    probs = frame.iloc[valid_idx][ROUTER_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ridx = np.argmax(probs, axis=1) if len(probs) else np.zeros(0, dtype=np.int64)
    by_regime = {}
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
        "quality_mean": float(np.mean(quality)),
        "quality_p95": float(np.quantile(quality, 0.95)),
        "by_regime": by_regime,
    }


def _metrics_for(frame: pd.DataFrame, proba: np.ndarray, prob: float, margin: float, fee: float, slip: float, exposure: float, max_hold: int) -> dict[str, Any]:
    actions = _decide_actions(proba, prob, margin)
    return {
        f"cost{m}": _backtest_actions(
            frame,
            actions,
            fee=float(fee) * float(m),
            slip=float(slip) * float(m),
            unit_exposure=float(exposure),
            max_hold_bars=int(max_hold),
        )
        for m in (1, 2, 3)
    }


def _score(metrics: dict[str, Any]) -> float:
    c1, c2, c3 = metrics["cost1"], metrics["cost2"], metrics["cost3"]
    if int(c1.get("trades", 0)) < 12:
        return -1e6 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"]) + 0.40 * float(c2["pnl"]) + 0.20 * float(c3["pnl"]) - 0.30 * abs(float(c1["mdd"]))


def _grid(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Compare Alpha4-mapped versus Alpha5-selected features for action-only HGB.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--train-end", default="2025-10-01")
    p.add_argument("--val-start", default="2025-10-01")
    p.add_argument("--val-end", default="2026-01-01")
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--feature-modes", default="alpha5_selected,alpha4_mapped_no_future,alpha4_mapped")
    p.add_argument("--weight-modes", default="balanced")
    p.add_argument("--feature-top-k", type=int, default=64)
    p.add_argument("--prob-thresholds", default="0.00,0.34,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    p.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.08,0.12,0.16,0.20")
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=5801)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train_df = _split(train_all, None, args.train_end)
    val_df = _split(train_all, args.val_start, args.val_end)
    audit = _verify_state24_sticky090_inputs(train_all, eval_df, args.manifest, args.clean4_report)
    cfg = _cfg_action_l1()

    # Build one action label set, matching the old Alpha4 action distribution.
    base_features = _feature_contract(train_all, eval_df, "alpha4_mapped", True, int(args.feature_top_k))
    x_label, y_label, train_meta = build_training_set(
        train_df,
        cfg=cfg,
        stride_bars=int(args.stride),
        batch_size=512,
        feature_cols=base_features,
    )
    valid_idx = _valid_indices(len(train_df), int(cfg.max_train_horizon_bars), int(args.stride))
    label_report = _label_report(y_label, train_df, valid_idx)
    y_action = np.asarray(y_label["action"], dtype=np.int64)
    print(
        json.dumps(
            {
                "stage": "start",
                "model_id": MODEL_ID,
                "train_rows": len(train_df),
                "validation_rows": len(val_df),
                "oos_rows": len(eval_df),
                "label_report": label_report,
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

    rows: list[dict[str, Any]] = []
    for mi, mode in enumerate([m.strip() for m in args.feature_modes.split(",") if m.strip()]):
        features = _feature_contract(train_all, eval_df, mode, True, int(args.feature_top_k))
        legacy = [c for c in features if c.startswith("clean_regime_2024_unsup_v4_")]
        if legacy:
            raise RuntimeError(f"{mode} selected legacy v4 features: {legacy[:10]}")
        x_train = _x(train_df, features).iloc[valid_idx].reset_index(drop=True)
        x_val = _x(val_df, features)
        x_eval = _x(eval_df, features)
        print(
            json.dumps(
                {
                    "stage": "feature_mode",
                    "mode": mode,
                    "feature_count": len(features),
                    "clean4_count": int(sum(c.startswith(CLEAN4_PREFIX) for c in features)),
                    "future_pred_count": int(sum(c.startswith("regime4_pred_") for c in features)),
                    "has_tp_sl_action_score": "tp_sl_action_score" in features,
                    "missing_train_all_nan": [c for c in features if x_train[c].isna().all()][:30],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        for wi, wmode in enumerate([w.strip() for w in args.weight_modes.split(",") if w.strip()]):
            model = _fit_hgb(x_train, y_action, _weights(y_label, wmode), int(args.seed) + mi * 100 + wi * 10)
            val_proba = _predict_proba_3(model, x_val)
            eval_proba = _predict_proba_3(model, x_eval)
            best: dict[str, Any] | None = None
            for prob in _grid(args.prob_thresholds):
                for margin in _grid(args.margin_thresholds):
                    val_metrics = _metrics_for(
                        val_df,
                        val_proba,
                        prob,
                        margin,
                        float(args.fee),
                        float(args.slip),
                        float(args.unit_exposure),
                        int(args.max_hold_bars),
                    )
                    score = _score(val_metrics)
                    if best is None or score > float(best["score"]):
                        best = {
                            "feature_mode": mode,
                            "weight_mode": wmode,
                            "prob_threshold": float(prob),
                            "margin_threshold": float(margin),
                            "score": float(score),
                            "validation_metrics": val_metrics,
                        }
            assert best is not None
            oos_metrics = _metrics_for(
                eval_df,
                eval_proba,
                float(best["prob_threshold"]),
                float(best["margin_threshold"]),
                float(args.fee),
                float(args.slip),
                float(args.unit_exposure),
                int(args.max_hold_bars),
            )
            artifact = args.out_dir / f"{mode}_{wmode}_action_hgb.joblib"
            joblib.dump(
                {
                    "model_id": MODEL_ID,
                    "feature_mode": mode,
                    "weight_mode": wmode,
                    "model": model,
                    "feature_cols": features,
                    "label_cfg": asdict(cfg),
                    "label_report": label_report,
                    "selected_thresholds": {
                        "prob_threshold": best["prob_threshold"],
                        "margin_threshold": best["margin_threshold"],
                        "max_hold_bars": int(args.max_hold_bars),
                    },
                },
                artifact,
            )
            row = {
                **best,
                "oos_metrics": oos_metrics,
                "feature_count": len(features),
                "clean4_count": int(sum(c.startswith(CLEAN4_PREFIX) for c in features)),
                "future_pred_count": int(sum(c.startswith("regime4_pred_") for c in features)),
                "artifact": str(artifact),
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "stage": "candidate_complete",
                        "feature_mode": mode,
                        "weight_mode": wmode,
                        "selected": {"prob": best["prob_threshold"], "margin": best["margin_threshold"]},
                        "validation_cost1": best["validation_metrics"]["cost1"],
                        "oos_cost1": oos_metrics["cost1"],
                    },
                    ensure_ascii=False,
                    default=_json_default,
                ),
                flush=True,
            )

    best = max(rows, key=lambda r: float(r["score"]))
    summary = {
        "model_id": MODEL_ID,
        "design": "Action-only HGB comparison between current Alpha5 selected features and Alpha4 feature contract mapped to clean_regime4. Labels are fixed to the Alpha4 lifecycle-derived action distribution; no TP/SL/notional/leverage heads are trained.",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "split": {
            "train": [str(train_df["timestamp"].iloc[0]), str(train_df["timestamp"].iloc[-1])],
            "validation": [str(val_df["timestamp"].iloc[0]), str(val_df["timestamp"].iloc[-1])],
            "oos": [str(eval_df["timestamp"].iloc[0]), str(eval_df["timestamp"].iloc[-1])],
        },
        "label_cfg": asdict(cfg),
        "label_report": label_report,
        "train_meta": train_meta,
        "state24_sticky090_audit": audit,
        "experiments": rows,
        "best": best,
        "artifacts": {
            "summary": str(args.out_dir / "alpha5_8_hgb_action_feature_contract_compare_summary.json"),
            "grid": str(args.out_dir / "alpha5_8_hgb_action_feature_contract_compare_grid.csv"),
        },
    }
    summary_path = args.out_dir / "alpha5_8_hgb_action_feature_contract_compare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "feature_mode": r["feature_mode"],
                "weight_mode": r["weight_mode"],
                "score": r["score"],
                "prob_threshold": r["prob_threshold"],
                "margin_threshold": r["margin_threshold"],
                "feature_count": r["feature_count"],
                "clean4_count": r["clean4_count"],
                "future_pred_count": r["future_pred_count"],
                "val_cost1_pnl": r["validation_metrics"]["cost1"]["pnl"],
                "val_cost1_mdd": r["validation_metrics"]["cost1"]["mdd"],
                "val_cost1_trades": r["validation_metrics"]["cost1"]["trades"],
                "oos_cost1_pnl": r["oos_metrics"]["cost1"]["pnl"],
                "oos_cost1_mdd": r["oos_metrics"]["cost1"]["mdd"],
                "oos_cost1_trades": r["oos_metrics"]["cost1"]["trades"],
                "oos_cost2_pnl": r["oos_metrics"]["cost2"]["pnl"],
                "oos_cost3_pnl": r["oos_metrics"]["cost3"]["pnl"],
                "artifact": r["artifact"],
            }
            for r in rows
        ]
    ).to_csv(args.out_dir / "alpha5_8_hgb_action_feature_contract_compare_grid.csv", index=False)
    print(json.dumps({"stage": "complete", "summary": str(summary_path), "best": {"feature_mode": best["feature_mode"], "weight_mode": best["weight_mode"]}}, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
