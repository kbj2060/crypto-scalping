#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_ai_patchmix_direction_core_20260530 as patchmix  # noqa: E402


MODEL_ID = "ai_patchmix_binary_tradeable_20260530"
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PatchTSMixer embedding + CatBoost binary tradeable long/short heads."
    )
    p.add_argument("--train-2024", type=Path, default=ROOT / "data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--train-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--score-2026", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--patch-model-id", default="ibm/patchtsmixer-etth1-pretrain")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--emb-dim", type=int, default=16)
    p.add_argument("--iterations", type=int, default=900)
    p.add_argument("--learning-rate", type=float, default=0.025)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--l2-leaf-reg", type=float, default=12.0)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--random-seed", type=int, default=20260530)
    p.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--horizons", default="6,12")
    p.add_argument(
        "--input-profile",
        choices=("audit_compact_local_regime",),
        default="audit_compact_local_regime",
    )
    return p.parse_args()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        raise KeyError(col)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _future_extreme(s: pd.Series, horizon: int, mode: str) -> pd.Series:
    future = s.shift(-1)
    if mode == "max":
        return future[::-1].rolling(horizon, min_periods=1).max()[::-1]
    if mode == "min":
        return future[::-1].rolling(horizon, min_periods=1).min()[::-1]
    raise ValueError(mode)


def _configs() -> list[dict[str, float | str]]:
    return [
        {
            "name": "tradeable_dense",
            "min_edge": 0.0010,
            "atr_mult": 0.16,
            "mae_penalty": 0.40,
            "cost": 0.00055,
            "margin": 0.00015,
        },
        {
            "name": "tradeable_base",
            "min_edge": 0.0012,
            "atr_mult": 0.22,
            "mae_penalty": 0.55,
            "cost": 0.00065,
            "margin": 0.00025,
        },
        {
            "name": "tradeable_fee2",
            "min_edge": 0.0016,
            "atr_mult": 0.26,
            "mae_penalty": 0.65,
            "cost": 0.00085,
            "margin": 0.00035,
        },
        {
            "name": "tradeable_high_quality",
            "min_edge": 0.0020,
            "atr_mult": 0.30,
            "mae_penalty": 0.75,
            "cost": 0.00100,
            "margin": 0.00045,
        },
    ]


def _binary_labels(
    frame: pd.DataFrame,
    *,
    horizon: int,
    min_edge: float,
    atr_mult: float,
    mae_penalty: float,
    cost: float,
    margin: float,
) -> pd.DataFrame:
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

    label = np.full(len(frame), -1, dtype=np.int64)
    long_ok = (long_score - short_score > float(margin)) & (long_score > floor)
    short_ok = (short_score - long_score > float(margin)) & (short_score > floor)
    label[short_ok] = 0
    label[long_ok] = 1
    valid = label >= 0
    valid[-int(horizon) :] = False
    return pd.DataFrame(
        {
            "label": label,
            "valid": valid.astype(np.int8),
            "long_score": long_score.astype(np.float32),
            "short_score": short_score.astype(np.float32),
            "edge_floor": floor.astype(np.float32),
        },
        index=frame.index,
    )


def _fit_values(frame: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in cols:
        x = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        val = float(x.median()) if x.notna().any() else 0.0
        out[col] = val if math.isfinite(val) else 0.0
    return out


def _matrix(frame: pd.DataFrame, cols: list[str], fill: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            col: pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(fill[col])
            for col in cols
        },
        index=frame.index,
    )


def _class_weights(y: np.ndarray, power: float) -> list[float]:
    counts = np.maximum(np.bincount(y.astype(int), minlength=2).astype(float), 1.0)
    weights = (counts.sum() / (2.0 * counts)) ** float(power)
    return [float(v) for v in weights]


def _set_patchmix_profile() -> None:
    patchmix.CORE_FEATURES = (
        *patchmix.BASE_CORE_FEATURES,
        *patchmix.AUDITED_COMPACT_FEATURES,
        *patchmix.LOCAL_REGIME_FEATURES,
    )


def _embeddings_for(path: Path, args: argparse.Namespace, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = patchmix._read_frame(path, int(args.limit))
    emb = patchmix._patch_embeddings(
        frame,
        model_id=str(args.patch_model_id),
        context_length=int(args.context_length),
        stride=int(args.stride),
        batch_size=int(args.batch_size),
        emb_dim=int(args.emb_dim),
        device=device,
    )
    return frame, emb


def _fit_pair(
    *,
    pair_name: str,
    train: pd.DataFrame,
    score: pd.DataFrame,
    train_x: pd.DataFrame,
    score_x: pd.DataFrame,
    horizon: int,
    cfg: dict[str, float | str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    cols = list(train_x.columns)
    fill = _fit_values(train_x, cols)
    lab_train = _binary_labels(
        train,
        horizon=int(horizon),
        min_edge=float(cfg["min_edge"]),
        atr_mult=float(cfg["atr_mult"]),
        mae_penalty=float(cfg["mae_penalty"]),
        cost=float(cfg["cost"]),
        margin=float(cfg["margin"]),
    )
    lab_score = _binary_labels(
        score,
        horizon=int(horizon),
        min_edge=float(cfg["min_edge"]),
        atr_mult=float(cfg["atr_mult"]),
        mae_penalty=float(cfg["mae_penalty"]),
        cost=float(cfg["cost"]),
        margin=float(cfg["margin"]),
    )
    data = pd.concat([train_x.reset_index(drop=True), lab_train.reset_index(drop=True)], axis=1)
    data = data[data["valid"] > 0].reset_index(drop=True)
    if len(data) < 1000:
        raise RuntimeError(f"{pair_name} h{horizon} {cfg['name']} has too few train labels: {len(data)}")
    split = int(len(data) * 0.82)
    fit_df = data.iloc[:split].reset_index(drop=True)
    hold_df = data.iloc[split:].reset_index(drop=True)
    x_fit = _matrix(fit_df, cols, fill)
    y_fit = fit_df["label"].to_numpy(dtype=np.int64)
    x_hold = _matrix(hold_df, cols, fill)
    y_hold = hold_df["label"].to_numpy(dtype=np.int64)
    x_score_all = _matrix(score_x, cols, fill)
    score_valid = lab_score["valid"].to_numpy(dtype=bool)
    y_score = lab_score.loc[score_valid, "label"].to_numpy(dtype=np.int64)
    x_score = x_score_all.loc[score_valid].reset_index(drop=True)

    params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=int(args.iterations),
        learning_rate=float(args.learning_rate),
        depth=int(args.depth),
        l2_leaf_reg=float(args.l2_leaf_reg),
        random_seed=int(args.random_seed) + int(horizon) + abs(hash(f"{pair_name}:{cfg['name']}")) % 10000,
        task_type=str(args.task_type),
        class_weights=_class_weights(y_fit, float(args.class_weight_power)),
        od_type="Iter",
        od_wait=80,
        verbose=False,
        allow_writing_files=False,
    )
    model = CatBoostClassifier(**params)
    used_task_type = str(args.task_type)
    try:
        model.fit(Pool(x_fit, y_fit), eval_set=Pool(x_hold, y_hold), use_best_model=True)
    except Exception:
        if str(args.task_type) != "GPU":
            raise
        params["task_type"] = "CPU"
        used_task_type = "CPU"
        model = CatBoostClassifier(**params)
        model.fit(Pool(x_fit, y_fit), eval_set=Pool(x_hold, y_hold), use_best_model=True)

    hold_p = np.asarray(model.predict_proba(x_hold), dtype=np.float64)[:, 1]
    score_p = np.asarray(model.predict_proba(x_score), dtype=np.float64)[:, 1]
    score_all_p = np.asarray(model.predict_proba(x_score_all), dtype=np.float64)[:, 1]
    hold_pred = (hold_p >= 0.5).astype(np.int64)
    score_pred = (score_p >= 0.5).astype(np.int64)

    run_dir = args.out_dir / pair_name / f"h{horizon}" / str(cfg["name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.cbm"
    pred_path = run_dir / "pred.csv"
    model.save_model(model_path)
    pd.DataFrame(
        {
            "timestamp": score["timestamp"].astype(str).to_numpy(),
            f"ai_patch_bin_h{horizon}_p_long": score_all_p.astype(np.float32),
            f"ai_patch_bin_h{horizon}_p_short": (1.0 - score_all_p).astype(np.float32),
            f"ai_patch_bin_h{horizon}_edge": (2.0 * score_all_p - 1.0).astype(np.float32),
        }
    ).to_csv(pred_path, index=False)

    out: dict[str, Any] = {
        "pair": pair_name,
        "horizon": int(horizon),
        "config": cfg,
        "task_type": used_task_type,
        "model_path": str(model_path),
        "pred_path": str(pred_path),
        "best_iteration": int(model.get_best_iteration() or 0),
        "train_rows": int(len(data)),
        "score_rows": int(len(score)),
        "score_tradeable_rows": int(score_valid.sum()),
        "score_tradeable_coverage": float(score_valid.mean()),
        "train_label_counts": {str(k): int(v) for k, v in zip(*np.unique(lab_train.loc[lab_train["valid"] > 0, "label"], return_counts=True))},
        "score_label_counts": {str(k): int(v) for k, v in zip(*np.unique(y_score, return_counts=True))},
        "hold_accuracy": float(accuracy_score(y_hold, hold_pred)),
        "hold_bacc": float(balanced_accuracy_score(y_hold, hold_pred)),
        "score_accuracy": float(accuracy_score(y_score, score_pred)),
        "score_bacc": float(balanced_accuracy_score(y_score, score_pred)),
        "score_pred_counts": np.bincount(score_pred, minlength=2).astype(int).tolist(),
        "score_confusion": confusion_matrix(y_score, score_pred, labels=[0, 1]).astype(int).tolist(),
    }
    try:
        out["hold_auc"] = float(roc_auc_score(y_hold, hold_p))
        out["score_auc"] = float(roc_auc_score(y_score, score_p))
    except Exception:
        out["hold_auc"] = None
        out["score_auc"] = None
    return out


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    horizons = tuple(int(x.strip()) for x in str(args.horizons).split(",") if x.strip())
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError(f"invalid horizons: {args.horizons}")
    _set_patchmix_profile()
    device = torch.device("cuda" if torch.cuda.is_available() and args.task_type == "GPU" else "cpu")

    frames: dict[str, pd.DataFrame] = {}
    feats: dict[str, pd.DataFrame] = {}
    for name, path in (("2024", args.train_2024), ("2025", args.train_2025), ("2026", args.score_2026)):
        frames[name], feats[name] = _embeddings_for(path, args, device)

    pairs = (
        ("fit2024_score2025", "2024", "2025"),
        ("fit2025_score2026", "2025", "2026"),
        ("fit2024_score2026", "2024", "2026"),
    )
    results: list[dict[str, Any]] = []
    for pair_name, train_key, score_key in pairs:
        for horizon in horizons:
            for cfg in _configs():
                row = _fit_pair(
                    pair_name=pair_name,
                    train=frames[train_key],
                    score=frames[score_key],
                    train_x=feats[train_key],
                    score_x=feats[score_key],
                    horizon=int(horizon),
                    cfg=cfg,
                    args=args,
                )
                results.append(row)
                print(json.dumps(row, ensure_ascii=False, default=_json_default), flush=True)

    summary = {
        "type": MODEL_ID,
        "contract": "PatchTSMixer HF embeddings + CatBoost binary tradeable long/short heads; neutral/flat bars are excluded from the binary target.",
        "input_profile": str(args.input_profile),
        "core_features": list(patchmix.CORE_FEATURES),
        "patch_channels": list(patchmix.PATCH_CHANNELS),
        "horizons": list(horizons),
        "configs": _configs(),
        "paths": {
            "train_2024": str(args.train_2024),
            "train_2025": str(args.train_2025),
            "score_2026": str(args.score_2026),
        },
        "results": sorted(results, key=lambda x: (str(x["pair"]), int(x["horizon"]), -float(x["score_bacc"]))),
        "best_by_pair_horizon": {},
    }
    for pair_name, _, _ in pairs:
        for horizon in horizons:
            subset = [r for r in results if r["pair"] == pair_name and int(r["horizon"]) == int(horizon)]
            summary["best_by_pair_horizon"][f"{pair_name}_h{horizon}"] = max(subset, key=lambda x: float(x["score_bacc"]))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"summary": str(args.out_dir / "summary.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
