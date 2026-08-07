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
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.dueling_dqn_per_full_architecture import ActionSpace  # noqa: E402
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    CLEAN4_PREFIX,
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_EVAL,
    DEFAULT_PREPROCESS_MANIFEST,
    DEFAULT_TRAIN,
    FORBIDDEN_EXACT,
    FORBIDDEN_PREFIXES,
    ROUTER_COLS,
    ROUTER_PROB_SET,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_alpha5_4_single_conditioned_dqn_20260518 import (  # noqa: E402
    _ensemble_q_labels,
    _feature_cols,
    _fit_market_scaler,
    _parse_horizons,
    _regime_matrix,
    _transform_market,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    _days,
    _fill_price,
    _json_default,
    _read,
)


MODEL_ID = "alpha5_5_lgbm_supervised_parent_ensembleq_state24_sticky090_20260518"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_5_lgbm_supervised_parent_ensembleq_20260518"


def _seed(seed: int) -> None:
    np.random.seed(int(seed))


def _labels_from_q(q: np.ndarray) -> np.ndarray:
    return np.argmax(q[:, [ActionSpace.FLAT, ActionSpace.OPEN_LONG, ActionSpace.OPEN_SHORT]], axis=1).astype(np.int64)


def _load_or_build_ensemble_labels(
    df: pd.DataFrame,
    regime: np.ndarray,
    *,
    cache: Path | None,
    horizons: str,
    scale: float,
    entry_hurdle: float,
    clip: float,
    confidence_min: float,
    fee: float,
    slip: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if cache is not None and cache.exists():
        payload = joblib.load(cache)
        q = np.asarray(payload["q_labels"], dtype=np.float32)
        w = np.asarray(payload["q_weights"], dtype=np.float32)
        report = dict(payload.get("report", {}))
        if len(q) == len(df):
            report["cache_loaded"] = str(cache)
            return q, w, report
    q, w, report = _ensemble_q_labels(
        df,
        regime,
        horizons=_parse_horizons(horizons),
        fee=float(fee),
        slip=float(slip),
        label_scale=float(scale),
        entry_hurdle=float(entry_hurdle),
        clip_value=float(clip),
        confidence_min=float(confidence_min),
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"q_labels": q, "q_weights": w, "report": report}, cache)
        report["cache_saved"] = str(cache)
    return q, w, report


def _make_lgbm_model(seed: int, n_jobs: int, class_weight: str | None) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=1200,
        learning_rate=0.025,
        max_depth=-1,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.82,
        subsample_freq=1,
        colsample_bytree=0.82,
        reg_alpha=0.15,
        reg_lambda=0.30,
        class_weight=class_weight,
        random_state=int(seed),
        n_jobs=int(n_jobs),
        verbose=-1,
    )


def _make_hgb_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=320,
        learning_rate=0.035,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=0.10,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=35,
        tol=1e-5,
        random_state=int(seed),
    )


def _class_balanced_weight(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    out = np.ones(len(y), dtype=np.float64)
    classes, counts = np.unique(y, return_counts=True)
    total = max(float(len(y)), 1.0)
    for cls, count in zip(classes, counts):
        out[y == int(cls)] = total / (float(len(classes)) * max(float(count), 1.0))
    return out


def _predict_proba_3(model: Any, x: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    out = np.zeros((len(x), 3), dtype=np.float64)
    for i, cls in enumerate(model.classes_):
        if 0 <= int(cls) < 3:
            out[:, int(cls)] = raw[:, i]
    denom = np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out / denom


def _decide_actions(proba: np.ndarray, prob_threshold: float, margin_threshold: float) -> np.ndarray:
    p_flat = proba[:, 0]
    p_long = proba[:, 1]
    p_short = proba[:, 2]
    best_trade = np.maximum(p_long, p_short)
    margin = np.abs(p_long - p_short)
    action = np.where(p_long > p_short, 1, 2).astype(np.int64)
    action = np.where(best_trade < float(prob_threshold), 0, action)
    action = np.where(margin < float(margin_threshold), 0, action)
    action = np.where(p_flat >= best_trade, 0, action)
    return action.astype(np.int64)


def _backtest_actions(
    frame: pd.DataFrame,
    actions: np.ndarray,
    *,
    fee: float,
    slip: float,
    unit_exposure: float,
    max_hold_bars: int,
) -> dict[str, Any]:
    close = frame["close"].to_numpy(dtype=np.float64)
    cash = 1.0
    peak_equity = 1.0
    mdd = 0.0
    side = 0
    entry_price = 0.0
    entry_equity = 1.0
    peak_price = 0.0
    hold_bars = 0
    trades = wins = long_entries = short_entries = 0
    action_counts = {"flat": 0, "long": 0, "short": 0}
    exits: dict[str, int] = {}
    exposure = float(unit_exposure)

    def mark(i: int) -> float:
        if side == 0:
            return cash
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        return cash * (1.0 + raw * exposure)

    def enter(i: int, new_side: int) -> None:
        nonlocal side, entry_price, entry_equity, peak_price, cash, long_entries, short_entries, hold_bars
        fill_i = min(i + 1, len(frame) - 1)
        side = int(new_side)
        entry_price = _fill_price(frame, fill_i, side, slip, entry=True)
        peak_price = float(entry_price)
        entry_equity = cash
        hold_bars = 0
        cash -= cash * float(fee) * exposure
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def exit_position(i: int, reason: str) -> None:
        nonlocal side, entry_price, peak_price, cash, trades, wins, hold_bars
        fill_i = min(i + 1, len(frame) - 1)
        exit_px = _fill_price(frame, fill_i, side, slip, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * float(fee) * exposure
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        side = 0
        entry_price = 0.0
        peak_price = 0.0
        hold_bars = 0

    for i in range(len(frame) - 2):
        desired = int(actions[i])
        action_counts["flat" if desired == 0 else "long" if desired == 1 else "short"] += 1
        if side != 0:
            hold_bars += 1
            peak_price = max(peak_price, close[i]) if side > 0 else min(peak_price, close[i])
        eq = mark(i)
        peak_equity = max(peak_equity, eq)
        mdd = min(mdd, eq / max(peak_equity, 1e-12) - 1.0)
        desired_side = 0 if desired == 0 else 1 if desired == 1 else -1
        if side != 0 and int(max_hold_bars) > 0 and hold_bars >= int(max_hold_bars):
            exit_position(i, "max_hold")
        elif side == 0 and desired_side != 0:
            enter(i, desired_side)
        elif side != 0 and desired_side == 0:
            exit_position(i, "model_flat")
        elif side != 0 and desired_side != side:
            exit_position(i, "model_flip")
            # Do not reverse on the same bar; next bar must confirm the new side.

    if side != 0:
        exit_position(len(frame) - 2, "end_of_data")
    eq = mark(len(frame) - 1)
    peak_equity = max(peak_equity, eq)
    mdd = min(mdd, eq / max(peak_equity, 1e-12) - 1.0)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(frame)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float((long_entries + short_entries) * exposure / max(len(frame), 1)),
        "action_counts": action_counts,
        "exits": exits,
    }


def _metrics_for_proba(frame: pd.DataFrame, proba: np.ndarray, *, prob_threshold: float, margin_threshold: float, fee: float, slip: float, unit_exposure: float) -> dict[str, Any]:
    actions = _decide_actions(proba, prob_threshold, margin_threshold)
    return {
        f"cost{mult}": _backtest_actions(
            frame,
            actions,
            fee=float(fee) * float(mult),
            slip=float(slip) * float(mult),
            unit_exposure=float(unit_exposure),
            max_hold_bars=0,
        )
        for mult in (1, 2, 3)
    }


def _metrics_for_proba_lifecycle(
    frame: pd.DataFrame,
    proba: np.ndarray,
    *,
    prob_threshold: float,
    margin_threshold: float,
    fee: float,
    slip: float,
    unit_exposure: float,
    max_hold_bars: int,
) -> dict[str, Any]:
    actions = _decide_actions(proba, prob_threshold, margin_threshold)
    return {
        f"cost{mult}": _backtest_actions(
            frame,
            actions,
            fee=float(fee) * float(mult),
            slip=float(slip) * float(mult),
            unit_exposure=float(unit_exposure),
            max_hold_bars=int(max_hold_bars),
        )
        for mult in (1, 2, 3)
    }


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    undertrade_penalty = max(2.0 - float(c1["trades_per_day"]), 0.0) * 10.0
    overtrade_penalty = max(float(c1["trades_per_day"]) - 8.0, 0.0) * 2.0
    return float(c1["pnl"] + 0.45 * c2["pnl"] + 0.30 * c3["pnl"] - 0.35 * abs(c1["mdd"]) - undertrade_penalty - overtrade_penalty)


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {cost: {k: metrics[cost][k] for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "long_entries", "short_entries", "avg_notional")} for cost in ("cost1", "cost2", "cost3")}


def _importance(model: Any, cols: list[str], top_n: int = 40) -> list[dict[str, Any]]:
    if not hasattr(model, "booster_"):
        return []
    gain = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
    split = np.asarray(model.booster_.feature_importance(importance_type="split"), dtype=float)
    rows = [
        {
            "rank": i + 1,
            "feature": cols[idx],
            "gain": float(gain[idx]),
            "split": float(split[idx]),
        }
        for i, idx in enumerate(np.argsort(-gain)[: int(top_n)])
    ]
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate Alpha5.5 LightGBM supervised action parent.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=5518)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--include-future-regime-pred", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--feature-top-k", type=int, default=64)
    p.add_argument("--feature-select-horizon", type=int, default=48)
    p.add_argument("--label-cache", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha5_5_lgbm_ensemble_q_labels_h020_conf030_train_20260518.joblib")
    p.add_argument("--val-label-cache", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha5_5_lgbm_ensemble_q_labels_h020_conf030_val_20260518.joblib")
    p.add_argument("--label-horizons", type=str, default="12,24,48,96,288")
    p.add_argument("--label-scale", type=float, default=50.0)
    p.add_argument("--label-entry-hurdle", type=float, default=0.20)
    p.add_argument("--label-clip", type=float, default=3.0)
    p.add_argument("--label-confidence-min", type=float, default=0.30)
    p.add_argument("--min-label-weight", type=float, default=0.10)
    p.add_argument("--prob-thresholds", type=str, default="0.34,0.38,0.42,0.46,0.50,0.55,0.60")
    p.add_argument("--margin-thresholds", type=str, default="0.00,0.03,0.05,0.08,0.12,0.16")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--class-weight", choices=("balanced", "none"), default="balanced")
    p.add_argument("--model-type", choices=("lgbm", "hgb"), default="lgbm")
    return p.parse_args()


def _parse_grid(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _seed(int(args.seed))
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    audit = _verify_state24_sticky090_inputs(train_all, eval_df, DEFAULT_PREPROCESS_MANIFEST, DEFAULT_CLEAN4_REPORT)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    market_cols = _feature_cols(
        train_all,
        eval_df,
        include_future_regime_pred=bool(args.include_future_regime_pred),
        feature_top_k=int(args.feature_top_k),
        feature_select_horizon=int(args.feature_select_horizon),
    )
    bad = [
        c
        for c in market_cols
        if (c.startswith(FORBIDDEN_PREFIXES) and not (bool(args.include_future_regime_pred) and c.startswith("regime4_pred_")))
        or c in ROUTER_PROB_SET
        or c in FORBIDDEN_EXACT
    ]
    if bad:
        raise ValueError("invalid market feature leakage: " + ", ".join(bad[:20]))
    scaler = _fit_market_scaler(train_df, market_cols)
    x_train_market = _transform_market(train_df, market_cols, scaler)
    x_val_market = _transform_market(val_df, market_cols, scaler)
    x_eval_market = _transform_market(eval_df, market_cols, scaler)
    train_regime = _regime_matrix(train_df)
    val_regime = _regime_matrix(val_df)
    eval_regime = _regime_matrix(eval_df)
    x_train = np.concatenate([x_train_market, train_regime], axis=1)
    x_val = np.concatenate([x_val_market, val_regime], axis=1)
    x_eval = np.concatenate([x_eval_market, eval_regime], axis=1)
    feature_cols = market_cols + list(ROUTER_COLS)

    q_train, w_train, label_report = _load_or_build_ensemble_labels(
        train_df,
        train_regime,
        cache=args.label_cache,
        horizons=str(args.label_horizons),
        scale=float(args.label_scale),
        entry_hurdle=float(args.label_entry_hurdle),
        clip=float(args.label_clip),
        confidence_min=float(args.label_confidence_min),
        fee=float(args.fee),
        slip=float(args.slip),
    )
    q_val, w_val, val_label_report = _load_or_build_ensemble_labels(
        val_df,
        val_regime,
        cache=args.val_label_cache,
        horizons=str(args.label_horizons),
        scale=float(args.label_scale),
        entry_hurdle=float(args.label_entry_hurdle),
        clip=float(args.label_clip),
        confidence_min=float(args.label_confidence_min),
        fee=float(args.fee),
        slip=float(args.slip),
    )
    y_train = _labels_from_q(q_train)
    y_val = _labels_from_q(q_val)
    sample_weight = np.maximum(np.asarray(w_train, dtype=np.float64), float(args.min_label_weight))
    val_sample_weight = np.maximum(np.asarray(w_val, dtype=np.float64), float(args.min_label_weight))
    if str(args.model_type) == "hgb" and str(args.class_weight) == "balanced":
        sample_weight = sample_weight * _class_balanced_weight(y_train)
        val_sample_weight = val_sample_weight * _class_balanced_weight(y_val)
    print(
        json.dumps(
            {
                "stage": "start",
                "model_id": MODEL_ID,
                "model_type": str(args.model_type),
                "train_rows": len(train_df),
                "selection_rows": len(val_df),
                "oos_rows": len(eval_df),
                "feature_count": len(feature_cols),
                "market_feature_count": len(market_cols),
                "label_report": label_report,
                "val_label_report": val_label_report,
                "class_counts_train": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
                "class_counts_val": {str(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
                "audit": {
                    "expected_model": audit.get("expected_model"),
                    "expected_model_found_in_manifest": audit.get("expected_model_found_in_manifest"),
                    "legacy_v4_count": audit.get("legacy_v4_count"),
                    "future_regime4_common_count": audit.get("future_regime4_common_count"),
                },
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    if str(args.model_type) == "hgb":
        model = _make_hgb_model(int(args.seed))
        model.fit(x_train, y_train, sample_weight=sample_weight)
    else:
        model = _make_lgbm_model(int(args.seed), int(args.n_jobs), None if str(args.class_weight) == "none" else str(args.class_weight))
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(x_val, y_val)],
            eval_sample_weight=[val_sample_weight],
            callbacks=[early_stopping(80, verbose=False), log_evaluation(100)],
        )
    val_proba = _predict_proba_3(model, x_val)
    eval_proba = _predict_proba_3(model, x_eval)
    grid_rows = []
    best: dict[str, Any] | None = None
    for prob_th in _parse_grid(args.prob_thresholds):
        for margin_th in _parse_grid(args.margin_thresholds):
            val_metrics = _metrics_for_proba(
                val_df,
                val_proba,
                prob_threshold=float(prob_th),
                margin_threshold=float(margin_th),
                fee=float(args.fee),
                slip=float(args.slip),
                unit_exposure=float(args.unit_exposure),
            )
            score = _score(val_metrics["cost1"], val_metrics["cost2"], val_metrics["cost3"])
            row = {
                "prob_threshold": float(prob_th),
                "margin_threshold": float(margin_th),
                "selection_score": float(score),
                "selection_metrics": _compact(val_metrics),
            }
            grid_rows.append(row)
            if best is None or score > best["selection_score"]:
                best = row
    assert best is not None
    eval_metrics = _metrics_for_proba(
        eval_df,
        eval_proba,
        prob_threshold=float(best["prob_threshold"]),
        margin_threshold=float(best["margin_threshold"]),
        fee=float(args.fee),
        slip=float(args.slip),
        unit_exposure=float(args.unit_exposure),
    )
    report = {
        "model_id": MODEL_ID,
        "model_type": str(args.model_type),
        "design": "LightGBM supervised parent trained from ensemble Q labels. Outputs action probabilities; threshold and margin are selected on 2025Q4 and fixed for 2026 OOS.",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "split": {
            "train": [str(train_df["timestamp"].iloc[0]), str(train_df["timestamp"].iloc[-1])],
            "selection": [str(val_df["timestamp"].iloc[0]), str(val_df["timestamp"].iloc[-1])],
            "oos": [str(eval_df["timestamp"].iloc[0]), str(eval_df["timestamp"].iloc[-1])],
        },
        "feature_contract": {
            "market_cols": market_cols,
            "regime_cols": ROUTER_COLS,
            "feature_cols": feature_cols,
            "market_dim": len(market_cols),
            "regime_dim": len(ROUTER_COLS),
            "legacy_clean_v4_count": int(sum(c.startswith("clean_regime_2024_unsup_v4_") for c in market_cols)),
            "future_regime4_feature_count": int(sum(c.startswith("regime4_pred_") for c in market_cols)),
            "clean4_aux_count": int(sum(c.startswith(CLEAN4_PREFIX) for c in market_cols)),
        },
        "state24_sticky090_feature_audit": audit,
        "label_report": label_report,
        "val_label_report": val_label_report,
        "class_counts_train": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "class_counts_val": {str(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "selected_thresholds": {
            "prob_threshold": float(best["prob_threshold"]),
            "margin_threshold": float(best["margin_threshold"]),
        },
        "selection_score": float(best["selection_score"]),
        "selection_metrics": best["selection_metrics"],
        "grid": sorted(grid_rows, key=lambda x: x["selection_score"], reverse=True),
        "metrics": eval_metrics,
        "selected_metrics": _compact(eval_metrics),
        "feature_importance": _importance(model, feature_cols, top_n=60),
        "config": vars(args),
        "artifacts": {
            "model": str(args.out_dir / "alpha5_5_lgbm_supervised_parent.joblib"),
            "scaler": str(args.out_dir / "alpha5_5_lgbm_supervised_parent_scaler.joblib"),
            "summary": str(args.out_dir / "alpha5_5_lgbm_supervised_parent_summary.json"),
            "grid": str(args.out_dir / "alpha5_5_lgbm_supervised_parent_grid.csv"),
        },
    }
    joblib.dump({"model": model, "feature_cols": feature_cols, "market_cols": market_cols, "regime_cols": ROUTER_COLS}, args.out_dir / "alpha5_5_lgbm_supervised_parent.joblib")
    joblib.dump({"market_cols": market_cols, "regime_cols": ROUTER_COLS, "scaler": scaler}, args.out_dir / "alpha5_5_lgbm_supervised_parent_scaler.joblib")
    (args.out_dir / "alpha5_5_lgbm_supervised_parent_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame(grid_rows).to_csv(args.out_dir / "alpha5_5_lgbm_supervised_parent_grid.csv", index=False)
    print(json.dumps({"stage": "complete", "summary": report["artifacts"]["summary"], "selected_thresholds": report["selected_thresholds"], "selected_metrics": report["selected_metrics"], "top_features": report["feature_importance"][:15]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
