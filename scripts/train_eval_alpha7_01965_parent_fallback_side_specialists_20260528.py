#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    FullyLearnedGovernorConfig,
    _classifier,
    _regressor,
    build_training_set,
    prepare_features,
    predict_policy_frame,
)
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    TP_COL,
    _combine_primary_fallback,
    _combo_metrics,
    _close,
    _json_default,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.retrain_alpha7_1_01965_tp_sl_decontam_20260528 import (  # noqa: E402
    DERIVABLE_FEATURES,
    FORBIDDEN_PREFIXES,
    REQUIRED_PREFIX,
    EVAL_CSV,
    TRAIN_CSV,
)

MODEL_ID = "alpha7_01965_parent_fallback_side_specialists_20260528"
LIVE_DIR = ROOT / "data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

PRIMARY_PARENT = LIVE_DIR / "primary_parent.pkl"
PRIMARY_SUMMARY = LIVE_DIR / "primary_summary.json"
FALLBACK_PARENT = LIVE_DIR / "fallback_alpha43_no_legacy_parent.pkl"
FALLBACK_SUMMARY = LIVE_DIR / "fallback_alpha43_no_legacy_summary.json"

BUCKET_KEYS = ("notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown")


def _forbidden_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if c.startswith(FORBIDDEN_PREFIXES)]


def _assert_clean_frame(df: pd.DataFrame, *, name: str) -> None:
    bad = _forbidden_cols(list(df.columns))
    if bad:
        raise RuntimeError(f"{name} contains forbidden legacy regime columns: {bad[:20]}")
    if TP_COL not in df.columns:
        raise RuntimeError(f"{name} missing required {TP_COL}")
    if not any(c.startswith(REQUIRED_PREFIX) for c in df.columns):
        raise RuntimeError(f"{name} missing required {REQUIRED_PREFIX} columns")


def _assert_feature_cols(df: pd.DataFrame, cols: list[str], *, name: str) -> None:
    bad = _forbidden_cols(cols)
    if bad:
        raise RuntimeError(f"{name} feature contract contains forbidden legacy columns: {bad[:20]}")
    missing = [c for c in cols if c not in df.columns and c not in DERIVABLE_FEATURES]
    if missing:
        raise RuntimeError(f"{name} missing feature columns: {missing[:30]}")


def _pipeline_fit_classifier(model: Any, x: pd.DataFrame, y: np.ndarray, weights: np.ndarray) -> Any | None:
    if np.unique(y).size < 2:
        return None
    model.fit(x, y, histgradientboostingclassifier__sample_weight=weights)
    return model


def _pipeline_fit_regressor(model: Any, x: pd.DataFrame, y: np.ndarray, weights: np.ndarray) -> Any:
    model.fit(x, y, histgradientboostingregressor__sample_weight=weights)
    return model


def _binary_proba(model: Any | None, x: pd.DataFrame, default: float = 0.0) -> np.ndarray:
    if model is None:
        return np.full(len(x), float(default), dtype=np.float64)
    proba = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int)
    if 1 not in classes:
        return np.zeros(len(x), dtype=np.float64)
    return proba[:, int(np.flatnonzero(classes == 1)[0])].astype(np.float64)


def _bucket_predict(
    bundle: dict[str, Any],
    side_name: str,
    key: str,
    x: pd.DataFrame,
    buckets: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    models = dict(bundle[f"{side_name}_bucket_models"])
    defaults = dict(bundle[f"{side_name}_default_bucket_indexes"])
    model = models.get(key)
    if model is None:
        idx = int(defaults.get(key, 0))
        val = float(buckets[int(np.clip(idx, 0, len(buckets) - 1))])
        return np.full(len(x), val, dtype=np.float64), np.ones(len(x), dtype=np.float64)
    proba = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int)
    vals = np.asarray([buckets[int(c)] for c in classes], dtype=np.float64)
    return proba @ vals, np.max(proba, axis=1)


def _side_features(frame: pd.DataFrame, feature_cols: list[str], side: int) -> pd.DataFrame:
    return prepare_features(frame, side_hint=int(side), close=_close(frame), feature_cols=feature_cols, strict=True)


def _train_side_policy(
    *,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "side_policy.pkl"
    summary_path = out_dir / "summary.json"
    if bundle_path.exists() and summary_path.exists():
        return joblib.load(bundle_path), json.loads(summary_path.read_text(encoding="utf-8"))

    ref = joblib.load(PRIMARY_PARENT)
    cfg = FullyLearnedGovernorConfig(**dict(ref["config"]))
    x, y, meta = build_training_set(
        train_df,
        cfg=cfg,
        stride_bars=6,
        batch_size=512,
        feature_cols=feature_cols,
    )
    action = np.asarray(y["action"], dtype=np.int64)
    quality = np.asarray(y["quality"], dtype=np.float64)
    base_weights = np.maximum(
        np.where(action == ACTION_CASH, 0.35, 1.0),
        np.clip(np.abs(quality), 0.03, 1.0),
    )

    long_y = (action == ACTION_LONG).astype(np.int64)
    short_y = (action == ACTION_SHORT).astype(np.int64)
    long_w = base_weights * np.where(action == ACTION_SHORT, 1.15, 1.0)
    short_w = base_weights * np.where(action == ACTION_LONG, 1.15, 1.0)

    x_long = x.copy()
    x_short = x.copy()
    if "side_hint" in x_long.columns:
        x_long["side_hint"] = 1.0
        x_short["side_hint"] = -1.0

    bundle: dict[str, Any] = {
        "model_type": "alpha7_01965_side_specialized_governor_policy_v1",
        "feature_cols": list(feature_cols),
        "config": asdict(cfg),
        "long_model": _pipeline_fit_classifier(_classifier(seed), x_long, long_y, long_w),
        "short_model": _pipeline_fit_classifier(_classifier(seed + 1), x_short, short_y, short_w),
        "long_quality_model": _pipeline_fit_regressor(
            _regressor(seed + 101),
            x_long,
            np.where(action == ACTION_LONG, quality, 0.0),
            np.where(action == ACTION_LONG, base_weights, 0.25 * base_weights),
        ),
        "short_quality_model": _pipeline_fit_regressor(
            _regressor(seed + 102),
            x_short,
            np.where(action == ACTION_SHORT, quality, 0.0),
            np.where(action == ACTION_SHORT, base_weights, 0.25 * base_weights),
        ),
        "long_bucket_models": {},
        "short_bucket_models": {},
        "long_default_bucket_indexes": {},
        "short_default_bucket_indexes": {},
        "label_distribution": {
            "action": pd.Series(action).value_counts().sort_index().to_dict(),
            "long_positive": int(long_y.sum()),
            "short_positive": int(short_y.sum()),
            "quality_mean": float(np.mean(quality)),
            "quality_p95": float(np.quantile(quality, 0.95)),
        },
    }
    for side_name, side_action, x_side in (("long", ACTION_LONG, x_long), ("short", ACTION_SHORT, x_short)):
        mask = action == int(side_action)
        for offset, key in enumerate(BUCKET_KEYS, start=10):
            vals = np.asarray(y[key], dtype=np.int64)
            if np.any(mask):
                mode = int(pd.Series(vals[mask]).mode().iloc[0])
            else:
                mode = 0
            bundle[f"{side_name}_default_bucket_indexes"][key] = mode
            model = _pipeline_fit_classifier(
                _classifier(seed + offset + (0 if side_name == "long" else 100)),
                x_side.loc[mask].copy(),
                vals[mask],
                base_weights[mask],
            )
            if model is not None:
                bundle[f"{side_name}_bucket_models"][key] = model

    joblib.dump(bundle, bundle_path)
    summary = {
        "model_id": MODEL_ID,
        "feature_count": len(feature_cols),
        "feature_cols": list(feature_cols),
        "train_meta": meta,
        "label_distribution": bundle["label_distribution"],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return bundle, summary


def _predict_side_policy(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    prob_threshold: float,
    margin_threshold: float,
) -> pd.DataFrame:
    cfg = FullyLearnedGovernorConfig(**dict(bundle["config"]))
    feature_cols = list(bundle["feature_cols"])
    x_long = _side_features(frame, feature_cols, 1)
    x_short = _side_features(frame, feature_cols, -1)

    p_long = _binary_proba(bundle.get("long_model"), x_long)
    p_short = _binary_proba(bundle.get("short_model"), x_short)
    q_long = np.asarray(bundle["long_quality_model"].predict(x_long), dtype=np.float64)
    q_short = np.asarray(bundle["short_quality_model"].predict(x_short), dtype=np.float64)
    long_ok = (p_long >= float(prob_threshold)) & ((p_long - p_short) >= float(margin_threshold))
    short_ok = (p_short >= float(prob_threshold)) & ((p_short - p_long) >= float(margin_threshold))
    choose_long = long_ok & (~short_ok | (p_long >= p_short))
    choose_short = short_ok & (~long_ok | (p_short > p_long))

    n = len(frame)
    action = np.full(n, ACTION_CASH, dtype=np.int64)
    side = np.zeros(n, dtype=np.int64)
    action[choose_long] = ACTION_LONG
    action[choose_short] = ACTION_SHORT
    side[choose_long] = 1
    side[choose_short] = -1

    notional = np.zeros(n, dtype=np.float64)
    leverage = np.ones(n, dtype=np.float64)
    take_profit = np.zeros(n, dtype=np.float64)
    stop_loss = np.zeros(n, dtype=np.float64)
    max_hold = np.zeros(n, dtype=np.float64)
    cooldown = np.zeros(n, dtype=np.float64)
    bucket_conf = np.ones((6, n), dtype=np.float64)

    for mask, side_name, x_side in ((choose_long, "long", x_long), (choose_short, "short", x_short)):
        if not np.any(mask):
            continue
        idx = np.flatnonzero(mask)
        notional_v, bucket_conf[0, idx] = _bucket_predict(bundle, side_name, "notional", x_side.iloc[idx], cfg.notional_buckets)
        leverage_v, bucket_conf[1, idx] = _bucket_predict(bundle, side_name, "leverage", x_side.iloc[idx], cfg.leverage_buckets)
        tp_v, bucket_conf[2, idx] = _bucket_predict(bundle, side_name, "take_profit", x_side.iloc[idx], cfg.take_profit_buckets)
        sl_v, bucket_conf[3, idx] = _bucket_predict(bundle, side_name, "stop_loss", x_side.iloc[idx], cfg.stop_loss_buckets)
        hold_v, bucket_conf[4, idx] = _bucket_predict(bundle, side_name, "max_hold", x_side.iloc[idx], tuple(float(v) for v in cfg.max_hold_buckets))
        cd_v, bucket_conf[5, idx] = _bucket_predict(bundle, side_name, "cooldown", x_side.iloc[idx], tuple(float(v) for v in cfg.cooldown_buckets))
        notional[idx] = notional_v
        leverage[idx] = leverage_v
        take_profit[idx] = tp_v
        stop_loss[idx] = sl_v
        max_hold[idx] = hold_v
        cooldown[idx] = cd_v

    leverage = np.clip(leverage, min(cfg.leverage_buckets), max(cfg.leverage_buckets))
    notional = np.clip(notional, 0.0, max(cfg.notional_buckets))
    fraction = np.clip(notional / np.maximum(leverage, 1e-8), 0.0, cfg.max_margin_fraction)
    notional = fraction * leverage
    selected_p = np.where(choose_long, p_long, np.where(choose_short, p_short, np.maximum(1.0 - p_long, 1.0 - p_short)))
    quality = np.where(choose_long, q_long, np.where(choose_short, q_short, 0.0))
    confidence = np.mean(np.vstack([selected_p, bucket_conf]), axis=0)
    out = pd.DataFrame(
        {
            "action": action,
            "side": side,
            "notional_exposure": notional,
            "leverage": leverage,
            "position_fraction": fraction,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "max_hold_bars": np.rint(max_hold).astype(np.int64),
            "cooldown_bars": np.rint(cooldown).astype(np.int64),
            "quality_score": quality,
            "confidence": confidence,
        },
        index=frame.index,
    )
    cash = action == ACTION_CASH
    out.loc[cash, ["side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[cash, "leverage"] = 1.0
    return out


def _predict_side_raw(bundle: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    cfg = FullyLearnedGovernorConfig(**dict(bundle["config"]))
    feature_cols = list(bundle["feature_cols"])
    x_long = _side_features(frame, feature_cols, 1)
    x_short = _side_features(frame, feature_cols, -1)
    raw: dict[str, Any] = {
        "config": cfg,
        "p_long": _binary_proba(bundle.get("long_model"), x_long),
        "p_short": _binary_proba(bundle.get("short_model"), x_short),
        "q_long": np.asarray(bundle["long_quality_model"].predict(x_long), dtype=np.float64),
        "q_short": np.asarray(bundle["short_quality_model"].predict(x_short), dtype=np.float64),
        "long": {},
        "short": {},
    }
    bucket_map = {
        "notional": cfg.notional_buckets,
        "leverage": cfg.leverage_buckets,
        "take_profit": cfg.take_profit_buckets,
        "stop_loss": cfg.stop_loss_buckets,
        "max_hold": tuple(float(v) for v in cfg.max_hold_buckets),
        "cooldown": tuple(float(v) for v in cfg.cooldown_buckets),
    }
    for side_name, x_side in (("long", x_long), ("short", x_short)):
        for key, buckets in bucket_map.items():
            vals, conf = _bucket_predict(bundle, side_name, key, x_side, buckets)
            raw[side_name][key] = vals
            raw[side_name][f"{key}_conf"] = conf
    return raw


def _decision_from_side_raw(
    raw: dict[str, Any],
    *,
    frame: pd.DataFrame,
    prob_threshold: float,
    margin_threshold: float,
) -> pd.DataFrame:
    cfg: FullyLearnedGovernorConfig = raw["config"]
    p_long = np.asarray(raw["p_long"], dtype=np.float64)
    p_short = np.asarray(raw["p_short"], dtype=np.float64)
    q_long = np.asarray(raw["q_long"], dtype=np.float64)
    q_short = np.asarray(raw["q_short"], dtype=np.float64)
    long_ok = (p_long >= float(prob_threshold)) & ((p_long - p_short) >= float(margin_threshold))
    short_ok = (p_short >= float(prob_threshold)) & ((p_short - p_long) >= float(margin_threshold))
    choose_long = long_ok & (~short_ok | (p_long >= p_short))
    choose_short = short_ok & (~long_ok | (p_short > p_long))

    n = len(frame)
    action = np.full(n, ACTION_CASH, dtype=np.int64)
    side = np.zeros(n, dtype=np.int64)
    action[choose_long] = ACTION_LONG
    action[choose_short] = ACTION_SHORT
    side[choose_long] = 1
    side[choose_short] = -1

    notional = np.zeros(n, dtype=np.float64)
    leverage = np.ones(n, dtype=np.float64)
    take_profit = np.zeros(n, dtype=np.float64)
    stop_loss = np.zeros(n, dtype=np.float64)
    max_hold = np.zeros(n, dtype=np.float64)
    cooldown = np.zeros(n, dtype=np.float64)
    bucket_conf = np.ones((6, n), dtype=np.float64)
    for mask, side_name in ((choose_long, "long"), (choose_short, "short")):
        if not np.any(mask):
            continue
        notional[mask] = raw[side_name]["notional"][mask]
        leverage[mask] = raw[side_name]["leverage"][mask]
        take_profit[mask] = raw[side_name]["take_profit"][mask]
        stop_loss[mask] = raw[side_name]["stop_loss"][mask]
        max_hold[mask] = raw[side_name]["max_hold"][mask]
        cooldown[mask] = raw[side_name]["cooldown"][mask]
        for i, key in enumerate(BUCKET_KEYS):
            bucket_conf[i, mask] = raw[side_name][f"{key}_conf"][mask]

    leverage = np.clip(leverage, min(cfg.leverage_buckets), max(cfg.leverage_buckets))
    notional = np.clip(notional, 0.0, max(cfg.notional_buckets))
    fraction = np.clip(notional / np.maximum(leverage, 1e-8), 0.0, cfg.max_margin_fraction)
    notional = fraction * leverage
    selected_p = np.where(choose_long, p_long, np.where(choose_short, p_short, np.maximum(1.0 - p_long, 1.0 - p_short)))
    quality = np.where(choose_long, q_long, np.where(choose_short, q_short, 0.0))
    confidence = np.mean(np.vstack([selected_p, bucket_conf]), axis=0)
    out = pd.DataFrame(
        {
            "action": action,
            "side": side,
            "notional_exposure": notional,
            "leverage": leverage,
            "position_fraction": fraction,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "max_hold_bars": np.rint(max_hold).astype(np.int64),
            "cooldown_bars": np.rint(cooldown).astype(np.int64),
            "quality_score": quality,
            "confidence": confidence,
        },
        index=frame.index,
    )
    cash = action == ACTION_CASH
    out.loc[cash, ["side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[cash, "leverage"] = 1.0
    return out


def _active_count(dec: pd.DataFrame) -> int:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).astype(int)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).astype(int)
    return int(((action != ACTION_CASH) & (side != 0)).sum())


def _metrics_row(name: str, split: str, dec: pd.DataFrame, df: pd.DataFrame) -> dict[str, Any]:
    costs = _combo_metrics(df, dec)
    c3 = costs["cost3"]
    return {
        "variant": name,
        "split": split,
        "pnl": float(c3["pnl"]),
        "mdd": float(c3["mdd"]),
        "wr": float(c3["wr"]),
        "trades": int(c3["trades"]),
        "active": _active_count(dec),
        "score": float(c3["pnl"]) + 1.25 * float(c3["mdd"]),
        "costs": costs,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all = _read(TRAIN_CSV)
    eval_df = _read(EVAL_CSV)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    _assert_clean_frame(train_all, name="train")
    _assert_clean_frame(eval_df, name="eval")

    primary_base = joblib.load(PRIMARY_PARENT)
    fallback_base = joblib.load(FALLBACK_PARENT)
    primary_rt = _load_best_scale_runtime(PRIMARY_SUMMARY)
    fallback_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)
    primary_cols = list(primary_base["feature_cols"])
    fallback_cols = list(fallback_base["feature_cols"])
    _assert_feature_cols(train_all, primary_cols, name="primary")
    _assert_feature_cols(train_all, fallback_cols, name="fallback")
    _assert_feature_cols(eval_df, primary_cols, name="primary_eval")
    _assert_feature_cols(eval_df, fallback_cols, name="fallback_eval")

    primary_side, primary_summary = _train_side_policy(
        train_df=train_df,
        feature_cols=primary_cols,
        seed=5287101,
        out_dir=OUT_DIR / "primary_side_specialist",
    )
    fallback_side, fallback_summary = _train_side_policy(
        train_df=train_df,
        feature_cols=fallback_cols,
        seed=5287301,
        out_dir=OUT_DIR / "fallback_side_specialist",
    )

    val_primary_base = _predict_scaled(primary_base, val_df, primary_rt)
    val_fallback_base = _predict_scaled(fallback_base, val_df, fallback_rt)
    oos_primary_base = _predict_scaled(primary_base, eval_df, primary_rt)
    oos_fallback_base = _predict_scaled(fallback_base, eval_df, fallback_rt)
    val_primary_raw = _predict_side_raw(primary_side, val_df)
    val_fallback_raw = _predict_side_raw(fallback_side, val_df)
    oos_primary_raw = _predict_side_raw(primary_side, eval_df)
    oos_fallback_raw = _predict_side_raw(fallback_side, eval_df)

    rows: list[dict[str, Any]] = []
    grid_records: list[dict[str, Any]] = []
    best_val: dict[str, Any] | None = None
    best_payload: dict[str, Any] | None = None
    thresholds = [0.45, 0.55, 0.65]
    margins = [0.00, 0.06]

    base_val = _combine_primary_fallback(val_primary_base, val_fallback_base)
    base_oos = _combine_primary_fallback(oos_primary_base, oos_fallback_base)
    rows.append(_metrics_row("baseline_primary_fallback", "val", base_val, val_df))
    rows.append(_metrics_row("baseline_primary_fallback", "oos", base_oos, eval_df))

    for threshold in thresholds:
        for margin in margins:
            val_primary_side = _decision_from_side_raw(
                val_primary_raw,
                frame=val_df,
                prob_threshold=threshold,
                margin_threshold=margin,
            )
            val_fallback_side = _decision_from_side_raw(
                val_fallback_raw,
                frame=val_df,
                prob_threshold=threshold,
                margin_threshold=margin,
            )
            oos_primary_side = _decision_from_side_raw(
                oos_primary_raw,
                frame=eval_df,
                prob_threshold=threshold,
                margin_threshold=margin,
            )
            oos_fallback_side = _decision_from_side_raw(
                oos_fallback_raw,
                frame=eval_df,
                prob_threshold=threshold,
                margin_threshold=margin,
            )
            variants = {
                "primary_side_fallback_base": (
                    _combine_primary_fallback(val_primary_side, val_fallback_base),
                    _combine_primary_fallback(oos_primary_side, oos_fallback_base),
                ),
                "primary_base_fallback_side": (
                    _combine_primary_fallback(val_primary_base, val_fallback_side),
                    _combine_primary_fallback(oos_primary_base, oos_fallback_side),
                ),
                "both_side_specialized": (
                    _combine_primary_fallback(val_primary_side, val_fallback_side),
                    _combine_primary_fallback(oos_primary_side, oos_fallback_side),
                ),
            }
            for variant, (val_dec, oos_dec) in variants.items():
                val_row = _metrics_row(variant, "val", val_dec, val_df)
                oos_row = _metrics_row(variant, "oos", oos_dec, eval_df)
                for item in (val_row, oos_row):
                    item["prob_threshold"] = float(threshold)
                    item["margin_threshold"] = float(margin)
                rows.extend([val_row, oos_row])
                grid_records.append(
                    {
                        "variant": variant,
                        "prob_threshold": float(threshold),
                        "margin_threshold": float(margin),
                        "val_pnl": float(val_row["pnl"]),
                        "val_mdd": float(val_row["mdd"]),
                        "val_wr": float(val_row["wr"]),
                        "val_trades": int(val_row["trades"]),
                        "val_score": float(val_row["score"]),
                        "oos_pnl": float(oos_row["pnl"]),
                        "oos_mdd": float(oos_row["mdd"]),
                        "oos_wr": float(oos_row["wr"]),
                        "oos_trades": int(oos_row["trades"]),
                        "oos_score": float(oos_row["score"]),
                    }
                )
                if best_val is None or float(val_row["score"]) > float(best_val["score"]):
                    best_val = val_row
                    best_payload = {
                        "variant": variant,
                        "prob_threshold": float(threshold),
                        "margin_threshold": float(margin),
                        "validation": val_row,
                        "oos": oos_row,
                    }

    grid_df = pd.DataFrame(grid_records).sort_values(["val_score", "oos_score"], ascending=[False, False])
    grid_path = OUT_DIR / "grid.csv"
    grid_df.to_csv(grid_path, index=False)
    report = {
        "model_id": MODEL_ID,
        "scope": "Primary and fallback parent retrain with LONG/SHORT binary action heads and side-specific risk bucket heads. Active/live artifacts are unchanged.",
        "inputs": {"train_csv": str(TRAIN_CSV), "eval_csv": str(EVAL_CSV), "live_dir": str(LIVE_DIR)},
        "contract": {
            "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
            "required_prefix": REQUIRED_PREFIX,
            "primary_feature_count": len(primary_cols),
            "fallback_feature_count": len(fallback_cols),
        },
        "primary_side_summary": primary_summary,
        "fallback_side_summary": fallback_summary,
        "baseline": {
            "validation": rows[0],
            "oos": rows[1],
        },
        "best_by_validation_score": best_payload,
        "grid_csv": str(grid_path),
        "rows": rows,
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "grid": str(grid_path), "best": best_payload}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
