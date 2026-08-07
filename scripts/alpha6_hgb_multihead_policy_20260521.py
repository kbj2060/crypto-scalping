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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha6_catboost_multihead_policy_20260521 import (  # noqa: E402
    DEFAULT_FEATURE_CSV,
    DEFAULT_LABEL_DIR,
    DEFAULT_SPEC_DIR,
    PolicyConfig,
    _backtest,
    _build_lifecycle_labels,
    _feature_matrix,
    _json_default,
    _label_frame,
    _read_feature_frame,
    _read_spec,
    _score,
    _threshold_grid,
)


MODEL_ID = "alpha6_hgb_multihead_policy_20260521"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha6_hgb_multihead_policy_20260521"


def _clf(args: argparse.Namespace, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=int(args.max_iter),
        learning_rate=float(args.learning_rate),
        max_leaf_nodes=int(args.max_leaf_nodes),
        l2_regularization=float(args.l2_regularization),
        min_samples_leaf=int(args.min_samples_leaf),
        early_stopping=False,
        random_state=int(seed),
    )


def _reg(args: argparse.Namespace, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=int(args.max_iter),
        learning_rate=float(args.learning_rate),
        max_leaf_nodes=int(args.max_leaf_nodes),
        l2_regularization=float(args.l2_regularization),
        min_samples_leaf=int(args.min_samples_leaf),
        early_stopping=False,
        random_state=int(seed),
    )


def _fit_classifier(x: np.ndarray, y: np.ndarray, w: np.ndarray, args: argparse.Namespace, seed: int) -> HistGradientBoostingClassifier | None:
    if np.unique(y).size < 2:
        return None
    model = _clf(args, seed)
    model.fit(x, y, sample_weight=w)
    return model


def _fit_models(x: np.ndarray, y: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    trade = y["action"] != 0
    action_w = np.where(trade, 1.0, 0.35)
    q_w = np.clip(np.abs(y["quality"]), 0.03, 1.0)
    weight = np.maximum(action_w, q_w)
    models: dict[str, Any] = {
        "action_model": _fit_classifier(x, y["action"], weight, args, args.seed),
        "quality_model": _reg(args, args.seed + 99),
        "default_bucket_indexes": {},
        "label_distribution": {
            key: pd.Series(vals).value_counts().sort_index().to_dict() for key, vals in y.items() if key != "quality"
        },
    }
    models["quality_model"].fit(x, y["quality"], sample_weight=weight)
    x_trade = x[trade]
    w_trade = weight[trade]
    for offset, key in enumerate(("notional", "take_profit", "stop_loss", "max_hold", "cooldown"), start=1):
        vals = y[key][trade]
        models["default_bucket_indexes"][key] = int(pd.Series(vals).mode().iloc[0]) if len(vals) else 0
        model = _fit_classifier(x_trade, vals, w_trade, args, args.seed + offset)
        if model is not None:
            models[f"{key}_model"] = model
    models["label_distribution"]["quality_mean"] = float(np.mean(y["quality"]))
    models["label_distribution"]["quality_p95"] = float(np.quantile(y["quality"], 0.95))
    return models


def _predict_bucket(models: dict[str, Any], key: str, x: np.ndarray, bucket_count: int) -> tuple[np.ndarray, np.ndarray]:
    model = models.get(f"{key}_model")
    if model is None:
        default = int(models["default_bucket_indexes"].get(key, 0))
        return np.full(len(x), default, dtype=np.int64), np.ones(len(x), dtype=np.float64)
    proba = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int)
    idx = classes[np.argmax(proba, axis=1)]
    return np.clip(idx.astype(np.int64), 0, bucket_count - 1), np.max(proba, axis=1)


def _predict_policy(models: dict[str, Any], x: np.ndarray, frame: pd.DataFrame, cfg: PolicyConfig) -> pd.DataFrame:
    action_proba = models["action_model"].predict_proba(x)
    action_classes = np.asarray(models["action_model"].classes_, dtype=int)
    action = action_classes[np.argmax(action_proba, axis=1)].astype(np.int64)
    action_conf = np.max(action_proba, axis=1)
    quality = np.asarray(models["quality_model"].predict(x), dtype=np.float64)
    notional_i, c1 = _predict_bucket(models, "notional", x, len(cfg.notional_buckets))
    tp_i, c2 = _predict_bucket(models, "take_profit", x, len(cfg.tp_atr_buckets))
    sl_i, c3 = _predict_bucket(models, "stop_loss", x, len(cfg.sl_atr_buckets))
    hold_i, c4 = _predict_bucket(models, "max_hold", x, len(cfg.max_hold_buckets))
    cool_i, c5 = _predict_bucket(models, "cooldown", x, len(cfg.cooldown_buckets))
    atr = pd.to_numeric(frame.get("atr14_pct", 0.003), errors="coerce").fillna(0.003).to_numpy(dtype=np.float64)
    tp = np.clip(np.asarray(cfg.tp_atr_buckets, dtype=np.float64)[tp_i] * atr, cfg.tp_min, cfg.tp_max)
    sl = np.clip(np.asarray(cfg.sl_atr_buckets, dtype=np.float64)[sl_i] * atr, cfg.sl_min, cfg.sl_max)
    return pd.DataFrame(
        {
            "action": action,
            "quality_score": quality,
            "confidence": np.mean(np.vstack([action_conf, c1, c2, c3, c4, c5]), axis=0),
            "notional": np.asarray(cfg.notional_buckets, dtype=np.float64)[notional_i],
            "take_profit": tp,
            "stop_loss": sl,
            "max_hold_bars": np.asarray(cfg.max_hold_buckets, dtype=np.int64)[hold_i],
            "cooldown_bars": np.asarray(cfg.cooldown_buckets, dtype=np.int64)[cool_i],
        },
        index=frame.index,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Alpha6 HGB multi-head lifecycle policy, backend-only comparison against CatBoost v2.")
    ap.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    ap.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--variant", default="stable48_global_pca32")
    ap.add_argument("--max-iter", type=int, default=220)
    ap.add_argument("--learning-rate", type=float, default=0.040)
    ap.add_argument("--max-leaf-nodes", type=int, default=31)
    ap.add_argument("--l2-regularization", type=float, default=0.08)
    ap.add_argument("--min-samples-leaf", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stride-bars", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--thresholds", type=int, default=70)
    ap.add_argument("--no-pca", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PolicyConfig()
    spec = _read_spec(args.spec_dir, args.variant)
    use_pca = bool(spec.get("extra_pca_enable")) and not args.no_pca and int(spec.get("extra_pca_components") or 0) > 0
    feat, present, missing = _read_feature_frame(args.feature_csv, list(spec["features"]), [])
    frame = feat.merge(_label_frame(args.label_dir), on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    train = frame[frame["dataset_split"].astype(str).str.lower().eq("train")].copy()
    val = frame[frame["dataset_split"].astype(str).str.lower().ne("train")].copy()
    if args.smoke:
        train = train.iloc[: min(len(train), 5000)].copy()
        val = val.iloc[: min(len(val), 3000)].copy()
        args.max_iter = min(args.max_iter, 40)
        args.thresholds = min(args.thresholds, 8)
        args.stride_bars = max(args.stride_bars, 6)
    x_train_all, x_val, model_features, pipe = _feature_matrix(
        train,
        val,
        present,
        use_pca=use_pca,
        pca_components=int(spec.get("extra_pca_components") or 0),
    )
    valid, y, label_meta = _build_lifecycle_labels(train, cfg, stride_bars=args.stride_bars, batch_size=args.batch_size)
    x_train = x_train_all[valid]
    print(
        f"[alpha6-hgb] variant={args.variant} train_rows={len(train)} val_rows={len(val)} label_candidates={len(valid)} raw_features={len(present)} model_features={len(model_features)} use_pca={use_pca}",
        flush=True,
    )
    models = _fit_models(x_train, y, args)
    dec = _predict_policy(models, x_val, val, cfg)
    rows = []
    best: dict[str, Any] | None = None
    for th in _threshold_grid(dec, args.thresholds):
        bt = {
            f"cost{m}": _backtest(val, dec, threshold=float(th), fee=cfg.fee * m, slip=cfg.slip * m)
            for m in (1, 2, 3)
        }
        score = _score(bt["cost1"], bt["cost2"], bt["cost3"])
        row = {
            "threshold": float(th),
            "score": float(score),
            "pnl": float(bt["cost1"]["pnl"]),
            "mdd": float(bt["cost1"]["mdd"]),
            "trades": int(bt["cost1"]["trades"]),
            "trades_per_day": float(bt["cost1"]["trades_per_day"]),
            "wr": float(bt["cost1"]["wr"]),
            "long_entries": int(bt["cost1"]["long_entries"]),
            "short_entries": int(bt["cost1"]["short_entries"]),
            "avg_notional": float(bt["cost1"]["avg_notional"]),
            "exits": json.dumps(bt["cost1"]["exits"], sort_keys=True),
        }
        rows.append(row)
        if best is None or row["score"] > best["summary"]["score"]:
            best = {"summary": row, "backtest": bt}
    assert best is not None
    prefix = args.out_dir / args.variant
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(f"{prefix}_threshold_grid.csv", index=False)
    pred = val[["timestamp", "open", "high", "low", "close", "label_action"]].copy()
    for col in dec.columns:
        pred[col] = dec[col].to_numpy()
    pred.to_csv(f"{prefix}_val_predictions.csv", index=False)
    artifact = {
        "model_id": MODEL_ID,
        "variant": args.variant,
        "config": cfg,
        "feature_cols": present,
        "model_features": model_features,
        "missing_features": missing,
        "use_pca": use_pca,
        "pipeline": pipe,
        "models": models,
    }
    joblib.dump(artifact, f"{prefix}_bundle.joblib")
    summary = {
        "model_id": MODEL_ID,
        "variant": args.variant,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "label_meta": label_meta,
        "label_distribution": models["label_distribution"],
        "raw_feature_count": int(len(present)),
        "missing_features": missing,
        "model_feature_count": int(len(model_features)),
        "use_pca": bool(use_pca),
        "best": best["summary"],
        "best_backtest": best["backtest"],
        "params": vars(args),
    }
    Path(f"{prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    print(json.dumps(summary["best"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
