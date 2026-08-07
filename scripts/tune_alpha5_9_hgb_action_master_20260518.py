#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
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
from scripts.train_eval_alpha5_8_hgb_action_feature_contract_compare_20260518 import (  # noqa: E402
    ALPHA4_PARENT,
    _alpha4_mapped_features,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    _close,
    _json_default,
    _read,
)


MODEL_ID = "alpha5_9_hgb_action_master_tuned_state24_sticky090_20260518"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_9_hgb_action_master_tuned_20260518"


@dataclass(frozen=True)
class HGBSpec:
    name: str
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float


def _split(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = frame.copy()
    if start:
        out = out[out["timestamp"] >= pd.Timestamp(start)]
    if end:
        out = out[out["timestamp"] < pd.Timestamp(end)]
    return out.reset_index(drop=True)


def _cfgs() -> dict[str, FullyLearnedGovernorConfig]:
    def unit(cash: float, adverse: float, hold: float) -> FullyLearnedGovernorConfig:
        return FullyLearnedGovernorConfig(
            notional_buckets=(1.0,),
            leverage_buckets=(1.5,),
            take_profit_buckets=(0.006, 0.010, 0.018, 0.030, 0.050, 0.090, 0.180),
            stop_loss_buckets=(0.004, 0.006, 0.010, 0.016, 0.024, 0.035),
            max_hold_buckets=(6, 12, 24, 48, 96, 192, 288),
            cooldown_buckets=(0,),
            max_train_horizon_bars=288,
            adverse_penalty=adverse,
            size_penalty=0.0,
            hold_penalty=hold,
            turnover_bonus=0.0010,
            cash_score=cash,
        )

    return {
        "unit_c014_a20_h025": unit(0.014, 2.00, 0.025),
        "unit_c018_a22_h030": unit(0.018, 2.20, 0.030),
        "unit_c022_a25_h035": unit(0.022, 2.50, 0.035),
        "size_l1_c020": FullyLearnedGovernorConfig(
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
        ),
    }


def _hgb_specs() -> list[HGBSpec]:
    return [
        HGBSpec("base", 280, 0.045, 31, 60, 0.08),
        HGBSpec("regularized", 220, 0.040, 15, 110, 0.18),
        HGBSpec("deeper", 420, 0.035, 45, 35, 0.05),
    ]


def _valid_indices(n_rows: int, horizon: int, stride: int) -> np.ndarray:
    return np.arange(0, max(0, int(n_rows) - int(horizon) - 1), max(1, int(stride)), dtype=np.int64)


def _features(train: pd.DataFrame, eval_df: pd.DataFrame, mode: str, top_k: int) -> list[str]:
    if mode == "alpha5_selected":
        return _alpha5_feature_cols(train, eval_df, include_future_regime_pred=True, feature_top_k=top_k, feature_select_horizon=48)
    if mode == "alpha4_mapped":
        return _alpha4_mapped_features(train, eval_df, include_future=True)
    if mode == "alpha4_mapped_no_future":
        return _alpha4_mapped_features(train, eval_df, include_future=False)
    raise ValueError(f"unknown feature mode: {mode}")


def _x(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return prepare_features(frame, side_hint=0, close=_close(frame), feature_cols=cols)


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
    w = np.maximum(np.clip(np.abs(quality), 0.02, 1.0), np.where(action == ACTION_CASH, 0.45, 1.0))
    if mode == "balanced":
        w = w * _class_balanced_weight(action)
    return w.astype(np.float64)


def _fit_hgb(x: pd.DataFrame, y: np.ndarray, w: np.ndarray, spec: HGBSpec, seed: int) -> Any:
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=int(spec.max_iter),
            learning_rate=float(spec.learning_rate),
            max_leaf_nodes=int(spec.max_leaf_nodes),
            min_samples_leaf=int(spec.min_samples_leaf),
            l2_regularization=float(spec.l2_regularization),
            early_stopping=False,
            random_state=int(seed),
        ),
    )
    model.fit(x, y, histgradientboostingclassifier__sample_weight=w)
    return model


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
            "cash": int(np.sum(action[m] == 0)),
            "long": int(np.sum(action[m] == 1)),
            "short": int(np.sum(action[m] == 2)),
            "quality_mean": float(np.mean(quality[m])) if np.any(m) else 0.0,
        }
    return {
        "rows": int(len(action)),
        "action_counts": {
            "cash": int(np.sum(action == 0)),
            "long": int(np.sum(action == 1)),
            "short": int(np.sum(action == 2)),
        },
        "trade_ratio": float(np.mean(action != 0)),
        "quality_mean": float(np.mean(quality)),
        "quality_p90": float(np.quantile(quality, 0.90)),
        "quality_p95": float(np.quantile(quality, 0.95)),
        "by_regime": by_regime,
    }


def _metrics(frame: pd.DataFrame, proba: np.ndarray, prob: float, margin: float, fee: float, slip: float, exposure: float, max_hold: int) -> dict[str, Any]:
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
    trades = int(c1.get("trades", 0))
    if trades < 20:
        return -1e6 + float(c1.get("pnl", 0.0))
    undertrade_penalty = max(0.75 - float(c1["trades_per_day"]), 0.0) * 8.0
    return (
        float(c1["pnl"])
        + 0.45 * float(c2["pnl"])
        + 0.25 * float(c3["pnl"])
        - 0.30 * abs(float(c1["mdd"]))
        - undertrade_penalty
        - 1.5 * max(0.0, float(c1["trades_per_day"]) - 6.0)
    )


def _grid(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Tune Alpha5.9 HGB action master.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--train-end", default="2025-10-01")
    p.add_argument("--val-start", default="2025-10-01")
    p.add_argument("--val-end", default="2026-01-01")
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--feature-modes", default="alpha5_selected,alpha4_mapped")
    p.add_argument("--label-cfgs", default="unit_c014_a20_h025,unit_c018_a22_h030,unit_c022_a25_h035,size_l1_c020")
    p.add_argument("--weight-modes", default="balanced,quality")
    p.add_argument("--feature-top-k", type=int, default=64)
    p.add_argument("--prob-thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.93,0.95")
    p.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.08,0.12,0.16,0.20")
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=5901)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train_df = _split(train_all, None, args.train_end)
    val_df = _split(train_all, args.val_start, args.val_end)
    audit = _verify_state24_sticky090_inputs(train_all, eval_df, args.manifest, args.clean4_report)
    cfgs = _cfgs()
    selected_cfgs = [x.strip() for x in str(args.label_cfgs).split(",") if x.strip()]
    feature_modes = [x.strip() for x in str(args.feature_modes).split(",") if x.strip()]
    weight_modes = [x.strip() for x in str(args.weight_modes).split(",") if x.strip()]
    hgb_specs = _hgb_specs()

    print(
        json.dumps(
            {
                "stage": "start",
                "model_id": MODEL_ID,
                "train_rows": len(train_df),
                "validation_rows": len(val_df),
                "oos_rows": len(eval_df),
                "label_cfgs": selected_cfgs,
                "feature_modes": feature_modes,
                "hgb_specs": [asdict(s) for s in hgb_specs],
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
    label_payloads: dict[str, tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any], dict[str, Any]]] = {}
    for cfg_name in selected_cfgs:
        cfg = cfgs[cfg_name]
        # Any feature matrix can be used to build labels because labels only depend on price path and cfg.
        label_features = _features(train_all, eval_df, "alpha4_mapped", int(args.feature_top_k))
        _, y, train_meta = build_training_set(train_df, cfg=cfg, stride_bars=int(args.stride), batch_size=512, feature_cols=label_features)
        valid_idx = _valid_indices(len(train_df), int(cfg.max_train_horizon_bars), int(args.stride))
        report = _label_report(y, train_df, valid_idx)
        label_payloads[cfg_name] = (y, valid_idx, train_meta, report)
        print(json.dumps({"stage": "label_built", "label_cfg": cfg_name, "label_report": report}, ensure_ascii=False, default=_json_default), flush=True)

    feature_cache: dict[str, tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for mode in feature_modes:
        cols = _features(train_all, eval_df, mode, int(args.feature_top_k))
        legacy = [c for c in cols if c.startswith("clean_regime_2024_unsup_v4_")]
        if legacy:
            raise RuntimeError(f"legacy clean v4 features selected in {mode}: {legacy[:10]}")
        feature_cache[mode] = (cols, _x(train_df, cols), _x(val_df, cols), _x(eval_df, cols))
        print(
            json.dumps(
                {
                    "stage": "features_ready",
                    "mode": mode,
                    "feature_count": len(cols),
                    "clean4_count": int(sum(c.startswith(CLEAN4_PREFIX) for c in cols)),
                    "future_pred_count": int(sum(c.startswith("regime4_pred_") for c in cols)),
                    "has_tp_sl_action_score": "tp_sl_action_score" in cols,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    total = len(selected_cfgs) * len(feature_modes) * len(weight_modes) * len(hgb_specs)
    done = 0
    for cfg_i, cfg_name in enumerate(selected_cfgs):
        y, valid_idx, train_meta, label_report = label_payloads[cfg_name]
        y_action = np.asarray(y["action"], dtype=np.int64)
        for mode_i, mode in enumerate(feature_modes):
            cols, x_train_full, x_val, x_eval = feature_cache[mode]
            x_train = x_train_full.iloc[valid_idx].reset_index(drop=True)
            for weight_i, weight_mode in enumerate(weight_modes):
                sample_weight = _weights(y, weight_mode)
                for spec_i, spec in enumerate(hgb_specs):
                    done += 1
                    print(json.dumps({"stage": "fit", "done": done, "total": total, "label_cfg": cfg_name, "feature_mode": mode, "weight_mode": weight_mode, "hgb": spec.name}, ensure_ascii=False), flush=True)
                    model = _fit_hgb(x_train, y_action, sample_weight, spec, int(args.seed) + cfg_i * 1000 + mode_i * 200 + weight_i * 50 + spec_i)
                    val_proba = _predict_proba_3(model, x_val)
                    eval_proba = _predict_proba_3(model, x_eval)
                    best: dict[str, Any] | None = None
                    for prob in _grid(args.prob_thresholds):
                        for margin in _grid(args.margin_thresholds):
                            val_metrics = _metrics(val_df, val_proba, prob, margin, args.fee, args.slip, args.unit_exposure, args.max_hold_bars)
                            score = _score(val_metrics)
                            if best is None or score > float(best["score"]):
                                best = {
                                    "label_cfg": cfg_name,
                                    "feature_mode": mode,
                                    "weight_mode": weight_mode,
                                    "hgb": asdict(spec),
                                    "prob_threshold": float(prob),
                                    "margin_threshold": float(margin),
                                    "score": float(score),
                                    "validation_metrics": val_metrics,
                                }
                    assert best is not None
                    oos_metrics = _metrics(eval_df, eval_proba, best["prob_threshold"], best["margin_threshold"], args.fee, args.slip, args.unit_exposure, args.max_hold_bars)
                    artifact = args.out_dir / f"{cfg_name}_{mode}_{weight_mode}_{spec.name}_action_hgb.joblib"
                    joblib.dump(
                        {
                            "model_id": MODEL_ID,
                            "model": model,
                            "feature_cols": cols,
                            "label_cfg_name": cfg_name,
                            "label_cfg": asdict(cfgs[cfg_name]),
                            "feature_mode": mode,
                            "weight_mode": weight_mode,
                            "hgb": asdict(spec),
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
                        "label_report": label_report,
                        "train_meta": train_meta,
                        "feature_count": len(cols),
                        "clean4_count": int(sum(c.startswith(CLEAN4_PREFIX) for c in cols)),
                        "future_pred_count": int(sum(c.startswith("regime4_pred_") for c in cols)),
                        "artifact": str(artifact),
                    }
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                "stage": "candidate",
                                "label_cfg": cfg_name,
                                "feature_mode": mode,
                                "weight_mode": weight_mode,
                                "hgb": spec.name,
                                "score": best["score"],
                                "selected": {"prob": best["prob_threshold"], "margin": best["margin_threshold"]},
                                "val_cost1": best["validation_metrics"]["cost1"],
                                "oos_cost1": oos_metrics["cost1"],
                            },
                            ensure_ascii=False,
                            default=_json_default,
                        ),
                        flush=True,
                    )

    best = max(rows, key=lambda r: float(r["score"]))
    top = sorted(rows, key=lambda r: float(r["score"]), reverse=True)[:20]
    summary = {
        "model_id": MODEL_ID,
        "design": "Full tuning for HGB action direction master: label utility configs, feature contract, sample weighting, HGB hyperparameters, and thresholds. Output remains action probabilities only; DSAC is expected to own sizing/exit.",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "split": {
            "train": [str(train_df["timestamp"].iloc[0]), str(train_df["timestamp"].iloc[-1])],
            "validation": [str(val_df["timestamp"].iloc[0]), str(val_df["timestamp"].iloc[-1])],
            "oos": [str(eval_df["timestamp"].iloc[0]), str(eval_df["timestamp"].iloc[-1])],
        },
        "state24_sticky090_audit": audit,
        "experiments": rows,
        "best": best,
        "top20": top,
        "artifacts": {
            "summary": str(args.out_dir / "alpha5_9_hgb_action_master_tuned_summary.json"),
            "grid": str(args.out_dir / "alpha5_9_hgb_action_master_tuned_grid.csv"),
        },
    }
    summary_path = args.out_dir / "alpha5_9_hgb_action_master_tuned_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "label_cfg": r["label_cfg"],
                "feature_mode": r["feature_mode"],
                "weight_mode": r["weight_mode"],
                "hgb_name": r["hgb"]["name"],
                "score": r["score"],
                "prob_threshold": r["prob_threshold"],
                "margin_threshold": r["margin_threshold"],
                "feature_count": r["feature_count"],
                "clean4_count": r["clean4_count"],
                "future_pred_count": r["future_pred_count"],
                "label_trade_ratio": r["label_report"]["trade_ratio"],
                "val_cost1_pnl": r["validation_metrics"]["cost1"]["pnl"],
                "val_cost1_mdd": r["validation_metrics"]["cost1"]["mdd"],
                "val_cost1_trades": r["validation_metrics"]["cost1"]["trades"],
                "val_cost1_tpd": r["validation_metrics"]["cost1"]["trades_per_day"],
                "oos_cost1_pnl": r["oos_metrics"]["cost1"]["pnl"],
                "oos_cost1_mdd": r["oos_metrics"]["cost1"]["mdd"],
                "oos_cost1_trades": r["oos_metrics"]["cost1"]["trades"],
                "oos_cost1_tpd": r["oos_metrics"]["cost1"]["trades_per_day"],
                "oos_cost2_pnl": r["oos_metrics"]["cost2"]["pnl"],
                "oos_cost3_pnl": r["oos_metrics"]["cost3"]["pnl"],
                "artifact": r["artifact"],
            }
            for r in rows
        ]
    ).sort_values("score", ascending=False).to_csv(args.out_dir / "alpha5_9_hgb_action_master_tuned_grid.csv", index=False)
    print(
        json.dumps(
            {
                "stage": "complete",
                "summary": str(summary_path),
                "best": {
                    "label_cfg": best["label_cfg"],
                    "feature_mode": best["feature_mode"],
                    "weight_mode": best["weight_mode"],
                    "hgb": best["hgb"]["name"],
                    "score": best["score"],
                    "oos_cost1": best["oos_metrics"]["cost1"],
                },
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
