#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha6_catboost_entry_quality_exit_policy_20260522 import (  # noqa: E402
    EQEConfig,
    _apply_label_preset,
    _build_entry_labels,
    _bucket_horizon,
    _fit_entry_models,
    _target_horizon_bucket,
)
from scripts.alpha6_catboost_5head_policy_20260522 import _numeric_matrix  # noqa: E402
from scripts.train_alpha6_dsac_ensemble_router_20260523 import MODEL_SPECS  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


PRESETS = [
    ("primary_hreg", "short_horizon_robust", "horizon_reg"),
    ("coverage_current", "current_quality", "bucket5"),
    ("high_precision", "high_precision_robust", "bucket5"),
    ("perturbation", "perturbation_robust", "bucket5"),
    ("adverse_veto", "adverse_conformal", "bucket5"),
    ("sam_veto", "sam_conformal", "bucket5"),
]
ALPHA61_STICKY_PREFIX = "clean_regime4_state24_sticky090_v2_"
FIXED_STICKY_PREFIX = "clean_regime4_2024_unsup_v1_"


def _read_csv(path: Path) -> pd.DataFrame:
    out = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _add_sticky_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in list(out.columns):
        if not col.startswith(FIXED_STICKY_PREFIX):
            continue
        alias = ALPHA61_STICKY_PREFIX + col[len(FIXED_STICKY_PREFIX) :]
        if alias not in out.columns:
            out[alias] = out[col]
    return out


def _feature_cols_for(frame: pd.DataFrame, name: str) -> tuple[list[str], list[str]]:
    prefix = dict(MODEL_SPECS)[name]
    bundle_path = Path(f"{prefix}_bundle.joblib")
    import joblib

    bundle = joblib.load(bundle_path)
    wanted = list(bundle["feature_cols"])
    present = [c for c in wanted if c in frame.columns]
    missing = [c for c in wanted if c not in frame.columns]
    return present, missing


def _class_prob(model: Any, x: np.ndarray, cls: int) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int)
    if cls not in classes:
        return np.zeros(len(x), dtype=np.float64)
    return np.asarray(proba[:, int(np.flatnonzero(classes == cls)[0])], dtype=np.float64)


def _predict(models: dict[str, Any], train_frame: pd.DataFrame, pred_frame: pd.DataFrame, cols: list[str], target_head_mode: str, max_target_horizon: int) -> pd.DataFrame:
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    x_train = _numeric_matrix(train_frame, cols)
    x_pred = _numeric_matrix(pred_frame, cols)
    x_fit = imputer.fit_transform(x_train)
    _ = x_fit
    x = imputer.transform(x_pred)
    action_model = models["action_model"]
    cash_p = _class_prob(action_model, x, 0)
    long_p = _class_prob(action_model, x, 1)
    short_p = _class_prob(action_model, x, 2)
    proba = np.vstack([cash_p, long_p, short_p]).T
    action = np.argmax(proba, axis=1).astype(np.int64)
    quality = np.asarray(models["quality_model"].predict(x), dtype=np.float64)
    if target_head_mode == "horizon_reg":
        horizon_model = models.get("target_horizon_model") or models.get("target_model")
        pred_horizon = np.expm1(np.asarray(horizon_model.predict(x), dtype=np.float64))
        target_horizon = np.clip(np.rint(pred_horizon), 2, max(2, int(max_target_horizon))).astype(np.int64)
        target_horizon = np.where(action == 0, 0, target_horizon)
        target_bucket = np.where(action == 0, 0, _target_horizon_bucket(target_horizon)).astype(np.int64)
    else:
        bucket_model = models.get("target_bucket_model") or models.get("target_model")
        bucket_proba = bucket_model.predict_proba(x)
        bucket_classes = np.asarray(bucket_model.classes_, dtype=int)
        target_bucket = bucket_classes[np.argmax(bucket_proba, axis=1)].astype(np.int64)
        target_bucket = np.where(action == 0, 0, np.clip(target_bucket, 0, 4)).astype(np.int64)
        target_horizon = np.asarray([_bucket_horizon(int(v)) if a != 0 else 0 for v, a in zip(target_bucket, action)], dtype=np.int64)
    return pd.DataFrame(
        {
            "action": action,
            "cash_prob": cash_p,
            "long_prob": long_p,
            "short_prob": short_p,
            "confidence": np.max(proba, axis=1),
            "quality": quality,
            "target_bucket": target_bucket,
            "target_horizon": target_horizon,
        },
        index=pred_frame.index,
    )


def _args_for(seed: int, target_head_mode: str, max_target_horizon: int, iterations: int) -> SimpleNamespace:
    return SimpleNamespace(
        iterations=int(iterations),
        learning_rate=0.04,
        depth=5,
        l2_leaf_reg=8.0,
        seed=int(seed),
        verbose=0,
        task_type="CPU",
        target_head_mode=str(target_head_mode),
        max_target_horizon=int(max_target_horizon),
        fixed_target_horizon=0,
        cash_action_weight=0.35,
    )


def _fit_predict_one(
    *,
    fit_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    cols: list[str],
    preset: str,
    target_head_mode: str,
    seed: int,
    iterations: int,
    stride_bars: int,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = _apply_label_preset(replace(EQEConfig(), fixed_notional=0.25), preset)
    max_target_horizon = int(cfg.max_train_horizon_bars)
    valid, y, label_meta = _build_entry_labels(
        fit_frame,
        cfg,
        stride_bars=int(stride_bars),
        batch_size=int(batch_size),
        adaptive_sampling=False,
        label_preset=str(preset),
        session_topk=2,
    )
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    x_fit_raw = _numeric_matrix(fit_frame, cols)
    x_fit_all = imputer.fit_transform(x_fit_raw)
    args = _args_for(seed, target_head_mode, max_target_horizon, iterations)
    models = _fit_entry_models(x_fit_all[valid], y, args)

    # Reuse the exact imputer fitted on the fold train frame for prediction.
    x_pred = imputer.transform(_numeric_matrix(pred_frame, cols))
    action_model = models["action_model"]
    cash_p = _class_prob(action_model, x_pred, 0)
    long_p = _class_prob(action_model, x_pred, 1)
    short_p = _class_prob(action_model, x_pred, 2)
    proba = np.vstack([cash_p, long_p, short_p]).T
    action = np.argmax(proba, axis=1).astype(np.int64)
    quality = np.asarray(models["quality_model"].predict(x_pred), dtype=np.float64)
    if target_head_mode == "horizon_reg":
        horizon_model = models.get("target_horizon_model") or models.get("target_model")
        pred_horizon = np.expm1(np.asarray(horizon_model.predict(x_pred), dtype=np.float64))
        target_horizon = np.clip(np.rint(pred_horizon), 2, max(2, max_target_horizon)).astype(np.int64)
        target_horizon = np.where(action == 0, 0, target_horizon)
        target_bucket = np.where(action == 0, 0, _target_horizon_bucket(target_horizon)).astype(np.int64)
    else:
        bucket_model = models.get("target_bucket_model") or models.get("target_model")
        bucket_proba = bucket_model.predict_proba(x_pred)
        bucket_classes = np.asarray(bucket_model.classes_, dtype=int)
        target_bucket = bucket_classes[np.argmax(bucket_proba, axis=1)].astype(np.int64)
        target_bucket = np.where(action == 0, 0, np.clip(target_bucket, 0, 4)).astype(np.int64)
        target_horizon = np.asarray([_bucket_horizon(int(v)) if a != 0 else 0 for v, a in zip(target_bucket, action)], dtype=np.int64)
    pred = pd.DataFrame(
        {
            "action": action,
            "cash_prob": cash_p,
            "long_prob": long_p,
            "short_prob": short_p,
            "confidence": np.max(proba, axis=1),
            "quality": quality,
            "target_bucket": target_bucket,
            "target_horizon": target_horizon,
        },
        index=pred_frame.index,
    )
    meta = {
        "fit_rows": int(len(fit_frame)),
        "pred_rows": int(len(pred_frame)),
        "label_candidates": int(len(valid)),
        "label_meta": label_meta,
        "target_head_mode": str(target_head_mode),
        "max_target_horizon": int(max_target_horizon),
    }
    return pred, meta


def _empty_pred(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action": np.zeros(len(index), dtype=np.int64),
            "cash_prob": np.ones(len(index), dtype=np.float64),
            "long_prob": np.zeros(len(index), dtype=np.float64),
            "short_prob": np.zeros(len(index), dtype=np.float64),
            "confidence": np.ones(len(index), dtype=np.float64),
            "quality": np.zeros(len(index), dtype=np.float64),
            "target_bucket": np.zeros(len(index), dtype=np.int64),
            "target_horizon": np.zeros(len(index), dtype=np.int64),
        },
        index=index,
    )


def _append_prefixed(out: pd.DataFrame, name: str, pred: pd.DataFrame) -> None:
    for col in ("cash_prob", "long_prob", "short_prob", "confidence", "quality", "target_bucket", "target_horizon"):
        out[f"a61_{name}_{col}"] = pred[col].to_numpy()
    out[f"a61_{name}_long_edge"] = pred["long_prob"].to_numpy(dtype=np.float64) * pred["quality"].to_numpy(dtype=np.float64)
    out[f"a61_{name}_short_edge"] = pred["short_prob"].to_numpy(dtype=np.float64) * pred["quality"].to_numpy(dtype=np.float64)
    out[f"a61_{name}_active"] = (pred["action"].to_numpy(dtype=np.int64) != 0).astype(np.float64)


def _append_derived(out: pd.DataFrame, names: list[str]) -> None:
    actions = np.vstack([out[f"a61_{n}_long_prob"].to_numpy() < out[f"a61_{n}_short_prob"].to_numpy() for n in names]).T
    long_probs = np.vstack([out[f"a61_{n}_long_prob"].to_numpy(dtype=np.float64) for n in names]).T
    short_probs = np.vstack([out[f"a61_{n}_short_prob"].to_numpy(dtype=np.float64) for n in names]).T
    qualities = np.vstack([out[f"a61_{n}_quality"].to_numpy(dtype=np.float64) for n in names]).T
    horizons = np.vstack([out[f"a61_{n}_target_horizon"].to_numpy(dtype=np.float64) for n in names]).T
    active = np.vstack([out[f"a61_{n}_active"].to_numpy(dtype=np.float64) for n in names]).T > 0.5
    long_vote = long_probs > np.maximum(short_probs, 1.0 - long_probs - short_probs)
    short_vote = short_probs > np.maximum(long_probs, 1.0 - long_probs - short_probs)
    out["a61_consensus_long"] = long_vote.mean(axis=1)
    out["a61_consensus_short"] = short_vote.mean(axis=1)
    out["a61_quality_top"] = np.max(qualities, axis=1)
    out["a61_quality_mean"] = np.mean(qualities, axis=1)
    out["a61_quality_dispersion"] = np.std(qualities, axis=1)
    out["a61_long_edge_sum"] = np.sum(long_probs * qualities, axis=1)
    out["a61_short_edge_sum"] = np.sum(short_probs * qualities, axis=1)
    vote_p = np.stack([1.0 - long_vote.mean(axis=1) - short_vote.mean(axis=1), long_vote.mean(axis=1), short_vote.mean(axis=1)], axis=1)
    vote_p = np.clip(vote_p, 1e-9, 1.0)
    out["a61_disagreement_entropy"] = -(vote_p * np.log(vote_p)).sum(axis=1) / np.log(3.0)
    h = np.where(horizons > 0, horizons, np.nan)
    out["a61_horizon_mean"] = np.nan_to_num(np.nanmean(h, axis=1), nan=0.0) / 96.0
    out["a61_horizon_std"] = np.nan_to_num(np.nanstd(h, axis=1), nan=0.0) / 96.0
    for lo, hi, group in ((1, 12, "short"), (13, 48, "mid"), (49, 10_000, "long")):
        mask = (horizons >= lo) & (horizons <= hi)
        out[f"a61_{group}_long_edge"] = np.sum(np.where(mask, long_probs * qualities, 0.0), axis=1)
        out[f"a61_{group}_short_edge"] = np.sum(np.where(mask, short_probs * qualities, 0.0), axis=1)
    if {"primary_hreg", "adverse_veto", "sam_veto"}.issubset(set(names)):
        out["a61_primary_adverse_quality_gap"] = out["a61_primary_hreg_quality"] - out["a61_adverse_veto_quality"]
        out["a61_primary_sam_quality_gap"] = out["a61_primary_hreg_quality"] - out["a61_sam_veto_quality"]
        primary_side = np.sign(out["a61_primary_hreg_long_prob"] - out["a61_primary_hreg_short_prob"]).to_numpy(dtype=np.float64)
        adverse_side = np.sign(out["a61_adverse_veto_long_prob"] - out["a61_adverse_veto_short_prob"]).to_numpy(dtype=np.float64)
        sam_side = np.sign(out["a61_sam_veto_long_prob"] - out["a61_sam_veto_short_prob"]).to_numpy(dtype=np.float64)
        out["a61_risk_opposition"] = ((primary_side != 0) & ((adverse_side == -primary_side) | (sam_side == -primary_side))).astype(np.float64)
    out["a61_active_model_count"] = active.sum(axis=1).astype(np.float64)


def _build_train_meta(frame: pd.DataFrame, *, iterations: int, stride_bars: int, batch_size: int, min_train_rows: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame.copy()
    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    fold_specs = [
        ("2025Q1_zero", None, "2025-04-01"),
        ("2025Q2", "2025-04-01", "2025-07-01"),
        ("2025Q3", "2025-07-01", "2025-10-01"),
        ("2025Q4", "2025-10-01", None),
    ]
    names = [x[0] for x in PRESETS]
    audit: dict[str, Any] = {"folds": [], "experts": {}}
    for name, preset, target_mode in PRESETS:
        pred_all = _empty_pred(out.index)
        cols, missing = _feature_cols_for(out, name)
        audit["experts"][name] = {"preset": preset, "target_head_mode": target_mode, "feature_count": len(cols), "missing": missing}
        for fold_no, (fold_name, start, end) in enumerate(fold_specs, start=1):
            pred_mask = np.ones(len(out), dtype=bool)
            if start is not None:
                pred_mask &= ts >= pd.Timestamp(start)
            if end is not None:
                pred_mask &= ts < pd.Timestamp(end)
            pred_idx = np.flatnonzero(pred_mask)
            if pred_idx.size == 0:
                continue
            if start is None:
                audit["folds"].append({"expert": name, "fold": fold_name, "status": "zero_fill_initial", "pred_rows": int(pred_idx.size)})
                continue
            fit = out.loc[ts < pd.Timestamp(start)].reset_index(drop=True)
            pred = out.loc[pred_mask].reset_index(drop=True)
            if len(fit) < int(min_train_rows):
                audit["folds"].append({"expert": name, "fold": fold_name, "status": "zero_fill_insufficient_history", "fit_rows": int(len(fit)), "pred_rows": int(len(pred))})
                continue
            fold_pred, meta = _fit_predict_one(
                fit_frame=fit,
                pred_frame=pred,
                cols=cols,
                preset=preset,
                target_head_mode=target_mode,
                seed=6100 + fold_no * 101 + len(name),
                iterations=int(iterations),
                stride_bars=int(stride_bars),
                batch_size=int(batch_size),
            )
            pred_all.loc[pred_mask, :] = fold_pred.to_numpy()
            audit["folds"].append({"expert": name, "fold": fold_name, "status": "predicted", **meta})
        _append_prefixed(out, name, pred_all)
    _append_derived(out, names)
    return out, audit


def _build_eval_meta(train_frame: pd.DataFrame, eval_frame: pd.DataFrame, *, iterations: int, stride_bars: int, batch_size: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = eval_frame.copy()
    names = [x[0] for x in PRESETS]
    audit: dict[str, Any] = {"experts": {}}
    for name, preset, target_mode in PRESETS:
        cols, missing = _feature_cols_for(train_frame, name)
        fit_pred, meta = _fit_predict_one(
            fit_frame=train_frame,
            pred_frame=eval_frame,
            cols=cols,
            preset=preset,
            target_head_mode=target_mode,
            seed=7100 + len(name),
            iterations=int(iterations),
            stride_bars=int(stride_bars),
            batch_size=int(batch_size),
        )
        audit["experts"][name] = {"preset": preset, "target_head_mode": target_mode, "feature_count": len(cols), "missing": missing, **meta}
        _append_prefixed(out, name, fit_pred)
    _append_derived(out, names)
    return out, audit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--eval-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--stride-bars", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--min-train-rows", type=int, default=20000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = _add_sticky_aliases(_read_csv(args.train_csv))
    eval_df = _add_sticky_aliases(_read_csv(args.eval_csv))
    train_out, train_audit = _build_train_meta(
        train,
        iterations=int(args.iterations),
        stride_bars=int(args.stride_bars),
        batch_size=int(args.batch_size),
        min_train_rows=int(args.min_train_rows),
    )
    eval_out, eval_audit = _build_eval_meta(
        train,
        eval_df,
        iterations=int(args.iterations),
        stride_bars=int(args.stride_bars),
        batch_size=int(args.batch_size),
    )
    train_path = args.out_dir / args.train_csv.name
    eval_path = args.out_dir / args.eval_csv.name
    train_out.to_csv(train_path, index=False)
    eval_out.to_csv(eval_path, index=False)
    meta_cols = [c for c in train_out.columns if c.startswith("a61_")]
    audit = {
        "model_id": "alpha61_oof_meta_for_sticky_current_20260525",
        "selection_uses_2026": False,
        "train_generation": "walk-forward OOF: Q1 zero-filled; Q2/Q3/Q4 predicted by Alpha6.1 preset heads trained only on prior 2025 rows",
        "eval_generation": "fit Alpha6.1 preset heads on all 2025 rows, predict 2026 without 2026 labels",
        "iterations": int(args.iterations),
        "stride_bars": int(args.stride_bars),
        "meta_feature_count": int(len(meta_cols)),
        "meta_features": meta_cols,
        "train_audit": train_audit,
        "eval_audit": eval_audit,
        "artifacts": {"train_csv": str(train_path), "eval_csv": str(eval_path), "audit": str(args.out_dir / "alpha61_oof_meta_audit.json")},
    }
    (args.out_dir / "alpha61_oof_meta_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"train_csv": str(train_path), "eval_csv": str(eval_path), "meta_feature_count": len(meta_cols), "audit": str(args.out_dir / "alpha61_oof_meta_audit.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
