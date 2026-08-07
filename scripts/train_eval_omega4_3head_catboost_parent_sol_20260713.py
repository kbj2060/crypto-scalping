#!/usr/bin/env python3
"""Causal CatBoost replacement benchmark for the SOL Omega4 TabM parent.

This is a research-only parent benchmark.  It keeps the production feature,
label, regime-routing, and prediction-file contracts, but intentionally does
not make the resulting bundle loadable by the current TabM live adapter.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_sol_20260707 as omega  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707 as tabm  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "sol_omega4_3head_catboost_parent_20260713"
OUT_ROOT = ROOT / "tmp/causal_regen_20260516"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _class_probs(model: CatBoostClassifier, x: pd.DataFrame, classes: int) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    out = np.zeros((len(x), classes), dtype=np.float64)
    for col, cls in enumerate(np.asarray(model.classes_, dtype=np.int64)):
        out[:, int(cls)] = raw[:, col]
    if not np.isfinite(out).all() or not np.allclose(out.sum(axis=1), 1.0, atol=1.0e-6):
        raise RuntimeError("invalid CatBoost probability output")
    return out


def _fit_classifier(
    x: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    classes: int,
    seed: int,
    iterations: int,
) -> CatBoostClassifier:
    split = max(int(len(x) * 0.85), min(len(x) - 1, 512 if classes == 3 else 256))
    if split >= len(x):
        raise RuntimeError("insufficient rows for chronological CatBoost validation")
    model = CatBoostClassifier(
        loss_function="MultiClass" if classes == 3 else "Logloss",
        eval_metric="MultiClass" if classes == 3 else "Logloss",
        iterations=int(iterations),
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=8.0,
        random_seed=int(seed),
        random_strength=0.35,
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(
        x.iloc[:split],
        y[:split],
        sample_weight=weights[:split],
        eval_set=(x.iloc[split:], y[split:]),
        early_stopping_rounds=60,
        verbose=False,
    )
    return model


def _fit_expert(
    x_dir: pd.DataFrame,
    y_dir: np.ndarray,
    y_quality: np.ndarray,
    route_frame: pd.DataFrame,
    x_exit: pd.DataFrame,
    y_exit: np.ndarray,
    exit_route_frame: pd.DataFrame,
    *,
    expert_idx: int,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    route_weight = parent._route_probs(route_frame)[:, expert_idx].astype(np.float64)
    exit_route_weight = parent._route_probs(exit_route_frame)[:, expert_idx].astype(np.float64)
    dir_weight = compute_sample_weight(class_weight="balanced", y=y_dir).astype(np.float64) * route_weight
    quality_weight = compute_sample_weight(class_weight="balanced", y=y_quality).astype(np.float64) * route_weight
    exit_weight = compute_sample_weight(class_weight="balanced", y=y_exit).astype(np.float64) * exit_route_weight
    if min(dir_weight.sum(), quality_weight.sum(), exit_weight.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} has invalid sample weights")
    return {
        "direction": _fit_classifier(x_dir, y_dir, dir_weight, classes=3, seed=seed + expert_idx * 17 + 1, iterations=iterations),
        "quality": _fit_classifier(x_dir, y_quality, quality_weight, classes=3, seed=seed + expert_idx * 17 + 2, iterations=iterations),
        "exit": _fit_classifier(x_exit, y_exit, exit_weight, classes=2, seed=seed + expert_idx * 17 + 3, iterations=iterations),
        "direction_cols": list(x_dir.columns),
        "exit_cols": list(x_exit.columns),
    }


def _routed_probs(
    models: dict[str, dict[str, Any]], x: pd.DataFrame, route: np.ndarray, head: str, classes: int) -> np.ndarray:
    out = np.empty((len(x), classes), dtype=np.float64)
    for expert_idx, expert in enumerate(hard.EXPERT_NAMES):
        mask = np.asarray(route, dtype=np.int64) == expert_idx
        if mask.any():
            out[mask] = _class_probs(models[expert][head], x.loc[mask], classes)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--max-train-rows", type=int, default=30000)
    ap.add_argument("--max-exit-samples", type=int, default=12000)
    ap.add_argument("--quality-threshold", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=260713)
    ap.add_argument("--out-suffix", default="zig075_q070")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    out_dir = OUT_ROOT / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = tabm._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=tabm.LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    val_raw = frames["val_raw"]
    oos_raw = frames["oos_raw"]
    if int(args.max_train_rows) > 0:
        train_fit = train_raw.iloc[: int(args.max_train_rows)].reset_index(drop=True)
    else:
        train_fit = train_raw.copy().reset_index(drop=True)
    x_dir = parent._base_input(train_fit, base_cols)
    y_dir = train_fit["zigzag_action"].to_numpy(dtype=np.int64)
    y_quality = train_fit["omega4_quality_action"].to_numpy(dtype=np.int64)
    x_exit_raw, y_exit, exit_frame, exit_diag = tabm._build_exit_dataset_entry_label_terminal_giveback(
        frames["train_df"],
        frames["s_train_label"],
        fee=fee,
        slip=slip,
        cost_mult=3.0,
        max_samples=int(args.max_exit_samples),
    )
    x_exit = parent._exit_input_from_position_rows(x_exit_raw, base_cols)

    models: dict[str, dict[str, Any]] = {}
    for expert_idx, expert in enumerate(hard.EXPERT_NAMES):
        print(f"stage=train expert={expert}", flush=True)
        models[expert] = _fit_expert(
            x_dir, y_dir, y_quality, train_fit, x_exit, y_exit, exit_frame,
            expert_idx=expert_idx, seed=int(args.seed), iterations=int(args.iterations),
        )

    def predict(frame: pd.DataFrame, *, oof: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        x = parent._base_input(frame, base_cols)
        route = hard._route_id(frame)
        direction = _routed_probs(models, x, route, "direction", 3)
        quality = _routed_probs(models, x, route, "quality", 3)
        prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
        src = parent._prediction_output(frame, direction, quality, threshold=float(args.quality_threshold), prefix=prefix)
        return src, direction, quality

    outputs: dict[str, pd.DataFrame] = {}
    for split, frame, oof in (("train", train_raw, True), ("validation", val_raw, True), ("oos", oos_raw, False)):
        src, _direction, _quality = predict(frame, oof=oof)
        outputs[split] = src
        src.to_csv(out_dir / f"{split}_predictions_q{int(round(args.quality_threshold * 100)):03d}.csv", index=False)
    val_dec = parent._to_decisions(outputs["validation"], oof=True)
    oos_dec = parent._to_decisions(outputs["oos"], oof=False)
    report = {
        "model_id": MODEL_ID,
        "model_family": "catboost_3head_regime_experts",
        "baseline_model": "sol_omega4_3head_parent72_loose_entry_quality_20260707_zig075_20260707",
        "research_only": True,
        "live_adapter_compatible": False,
        "training_split": {
            "start": str(pd.to_datetime(train_raw["timestamp"].iloc[0])),
            "end": str(pd.to_datetime(train_raw["timestamp"].iloc[-1])),
        },
        "validation_split": {
            "start": str(pd.to_datetime(val_raw["timestamp"].iloc[0])),
            "end": str(pd.to_datetime(val_raw["timestamp"].iloc[-1])),
        },
        "oos_split": {
            "start": str(pd.to_datetime(oos_raw["timestamp"].iloc[0])),
            "end": str(pd.to_datetime(oos_raw["timestamp"].iloc[-1])),
            "promotion_score": False,
        },
        "quality_threshold": float(args.quality_threshold),
        "input_contract": {"base_feature_count": len(base_cols), "position_cols": parent.POS_COLS},
        "exit_label": {"mode": "entry_label_terminal_giveback", "diag": exit_diag},
        "prediction_artifacts": {
            "q%03d" % int(round(args.quality_threshold * 100)): {
                split: str(out_dir / f"{split}_predictions_q{int(round(args.quality_threshold * 100)):03d}.csv")
                for split in ("train", "validation", "oos")
            }
        },
        "parent_only_metrics": {
            "validation": omega._metrics(val_raw, val_dec, fee=fee, slip=slip, cost_mult=3.0),
            "oos": omega._metrics(oos_raw, oos_dec, fee=fee, slip=slip, cost_mult=3.0),
        },
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }
    with open(out_dir / "catboost_3head_bundle.pkl", "wb") as f:
        pickle.dump({"models": models, "base_cols": base_cols, "pos_cols": parent.POS_COLS}, f)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    print(json.dumps(report["parent_only_metrics"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
