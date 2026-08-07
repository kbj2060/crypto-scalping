#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_ai_patchmix_direction_core_20260530 import (  # noqa: E402
    _json_default,
    _matrix,
    _patch_embeddings,
    _read_frame,
)


DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/ai_patchmix_h6_label_sweep_20260530"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep h6 direction labels on fixed PatchTSMixer/local-regime features.")
    p.add_argument("--train-csv", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--score-csv", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--patch-model-id", default="ibm/patchtsmixer-etth1-pretrain")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--emb-dim", type=int, default=16)
    p.add_argument("--iterations", type=int, default=850)
    p.add_argument("--learning-rate", type=float, default=0.025)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--l2-leaf-reg", type=float, default=12.0)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--random-seed", type=int, default=20260530)
    p.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _future_extreme(s: pd.Series, horizon: int, mode: str) -> pd.Series:
    future = s.shift(-1)
    if mode == "max":
        return future[::-1].rolling(horizon, min_periods=1).max()[::-1]
    if mode == "min":
        return future[::-1].rolling(horizon, min_periods=1).min()[::-1]
    raise ValueError(mode)


def _labels(frame: pd.DataFrame, *, horizon: int, min_edge: float, atr_mult: float, mae_penalty: float, cost: float, margin: float) -> pd.DataFrame:
    close = _num(frame, "close").ffill().bfill().clip(lower=1e-12)
    high = _num(frame, "high").ffill().bfill()
    low = _num(frame, "low").ffill().bfill()
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_pct = (tr.rolling(14, min_periods=3).mean() / close).fillna(0.001)
    floor = np.maximum(float(min_edge), atr_pct.to_numpy(dtype=np.float64) * float(atr_mult))

    fut_high = _future_extreme(high, horizon, "max")
    fut_low = _future_extreme(low, horizon, "min")
    long_mfe = (fut_high / close - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    long_mae = (1.0 - fut_low / close).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    short_mfe = long_mae.copy()
    short_mae = long_mfe.copy()
    long_score = long_mfe - float(mae_penalty) * long_mae - float(cost)
    short_score = short_mfe - float(mae_penalty) * short_mae - float(cost)

    y = np.zeros(len(frame), dtype=np.int64)
    y[(short_score - long_score > float(margin)) & (short_score > floor)] = 1
    y[(long_score - short_score > float(margin)) & (long_score > floor)] = 2
    valid = np.ones(len(frame), dtype=bool)
    valid[-int(horizon) :] = False
    return pd.DataFrame({"label": y, "valid": valid.astype(np.int8)}, index=frame.index)


def _fit_values(frame: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in cols:
        x = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        val = float(x.median()) if x.notna().any() else 0.0
        out[col] = val if math.isfinite(val) else 0.0
    return out


def _class_weights(y: np.ndarray, power: float) -> list[float]:
    counts = np.maximum(np.bincount(y.astype(int), minlength=3).astype(float), 1.0)
    weights = (counts.sum() / (3.0 * counts)) ** float(power)
    return [float(v) for v in weights]


def _configs() -> list[dict[str, float]]:
    return [
        {"name": "base", "min_edge": 0.0012, "atr_mult": 0.22, "mae_penalty": 0.55, "cost": 0.00055, "margin": 0.00035},
        {"name": "clear_margin", "min_edge": 0.0012, "atr_mult": 0.22, "mae_penalty": 0.55, "cost": 0.00055, "margin": 0.00055},
        {"name": "higher_edge", "min_edge": 0.0017, "atr_mult": 0.26, "mae_penalty": 0.55, "cost": 0.00055, "margin": 0.00035},
        {"name": "mae_strict", "min_edge": 0.0012, "atr_mult": 0.22, "mae_penalty": 0.75, "cost": 0.00055, "margin": 0.00035},
        {"name": "mae_light", "min_edge": 0.0012, "atr_mult": 0.18, "mae_penalty": 0.40, "cost": 0.00055, "margin": 0.00025},
        {"name": "active_dense", "min_edge": 0.0009, "atr_mult": 0.16, "mae_penalty": 0.40, "cost": 0.00045, "margin": 0.00020},
        {"name": "cost_strict", "min_edge": 0.0014, "atr_mult": 0.22, "mae_penalty": 0.55, "cost": 0.00075, "margin": 0.00035},
    ]


def _fit_one(
    *,
    cfg: dict[str, float],
    train: pd.DataFrame,
    score: pd.DataFrame,
    train_x: pd.DataFrame,
    score_x: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cols = list(train_x.columns)
    fill = _fit_values(train_x, cols)
    lab_train = _labels(train, horizon=6, **{k: float(cfg[k]) for k in ("min_edge", "atr_mult", "mae_penalty", "cost", "margin")})
    lab_score = _labels(score, horizon=6, **{k: float(cfg[k]) for k in ("min_edge", "atr_mult", "mae_penalty", "cost", "margin")})
    data = pd.concat([train_x.reset_index(drop=True), lab_train.reset_index(drop=True)], axis=1)
    data = data[data["valid"] > 0].reset_index(drop=True)
    split = int(len(data) * 0.82)
    fit_df = data.iloc[:split].reset_index(drop=True)
    hold_df = data.iloc[split:].reset_index(drop=True)
    x_fit = _matrix(fit_df, cols, fill)
    y_fit = fit_df["label"].to_numpy(dtype=np.int64)
    x_hold = _matrix(hold_df, cols, fill)
    y_hold = hold_df["label"].to_numpy(dtype=np.int64)
    x_score = _matrix(score_x, cols, fill)
    y_score = lab_score.loc[lab_score["valid"] > 0, "label"].to_numpy(dtype=np.int64)
    x_score_valid = x_score.loc[lab_score["valid"] > 0].reset_index(drop=True)

    params = dict(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        iterations=int(args.iterations),
        learning_rate=float(args.learning_rate),
        depth=int(args.depth),
        l2_leaf_reg=float(args.l2_leaf_reg),
        random_seed=int(args.random_seed) + abs(hash(str(cfg["name"]))) % 10000,
        task_type=str(args.task_type),
        class_weights=_class_weights(y_fit, float(args.class_weight_power)),
        od_type="Iter",
        od_wait=80,
        verbose=False,
        allow_writing_files=False,
    )
    model = CatBoostClassifier(**params)
    try:
        model.fit(Pool(x_fit, y_fit), eval_set=Pool(x_hold, y_hold), use_best_model=True)
    except Exception as exc:
        if str(args.task_type) != "GPU":
            raise
        params["task_type"] = "CPU"
        model = CatBoostClassifier(**params)
        model.fit(Pool(x_fit, y_fit), eval_set=Pool(x_hold, y_hold), use_best_model=True)

    hold_p = np.asarray(model.predict_proba(x_hold), dtype=np.float64)
    score_p = np.asarray(model.predict_proba(x_score_valid), dtype=np.float64)
    pred = np.argmax(score_p, axis=1)
    model_path = args.out_dir / f"{cfg['name']}.cbm"
    model.save_model(model_path)
    result: dict[str, Any] = {
        "config": cfg,
        "model_path": str(model_path),
        "best_iteration": int(model.get_best_iteration() or 0),
        "train_label_counts": {str(k): int(v) for k, v in zip(*np.unique(lab_train["label"].to_numpy(), return_counts=True))},
        "score_label_counts": {str(k): int(v) for k, v in zip(*np.unique(lab_score["label"].to_numpy(), return_counts=True))},
        "hold_bacc": float(balanced_accuracy_score(y_hold, np.argmax(hold_p, axis=1))),
        "score_bacc": float(balanced_accuracy_score(y_score, pred)),
        "score_pred_counts": np.bincount(pred, minlength=3).astype(int).tolist(),
        "score_confusion": confusion_matrix(y_score, pred, labels=[0, 1, 2]).astype(int).tolist(),
    }
    try:
        result["hold_auc"] = float(roc_auc_score(y_hold, hold_p, multi_class="ovr"))
        result["score_auc"] = float(roc_auc_score(y_score, score_p, multi_class="ovr"))
    except Exception:
        result["hold_auc"] = None
        result["score_auc"] = None
    return result


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and args.task_type == "GPU" else "cpu")
    train = _read_frame(args.train_csv, int(args.limit))
    score = _read_frame(args.score_csv, int(args.limit))
    train_x = _patch_embeddings(
        train,
        model_id=str(args.patch_model_id),
        context_length=int(args.context_length),
        stride=int(args.stride),
        batch_size=int(args.batch_size),
        emb_dim=int(args.emb_dim),
        device=device,
    )
    score_x = _patch_embeddings(
        score,
        model_id=str(args.patch_model_id),
        context_length=int(args.context_length),
        stride=int(args.stride),
        batch_size=int(args.batch_size),
        emb_dim=int(args.emb_dim),
        device=device,
    )
    results = []
    for cfg in _configs():
        results.append(_fit_one(cfg=cfg, train=train, score=score, train_x=train_x, score_x=score_x, args=args))
        print(json.dumps(results[-1], ensure_ascii=False, default=_json_default), flush=True)
    summary = {
        "type": "ai_patchmix_h6_label_sweep_20260530",
        "train_csv": str(args.train_csv),
        "score_csv": str(args.score_csv),
        "feature_contract": "audit_compact_local_regime via build_ai_patchmix_direction_core_20260530",
        "results": sorted(results, key=lambda x: float(x["score_bacc"]), reverse=True),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
