#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
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

from scripts.build_ai_patchmix_direction_core_20260530 import _core_features, _json_default, _matrix, _read_frame  # noqa: E402
from scripts.sweep_ai_patchmix_h6_label_params_20260530 import _class_weights, _fit_values, _labels  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/ai_chronos_h6_direction_20260530"
PATCH_H6_2025 = ROOT / "tmp/causal_regen_20260516/ai_patchmix_h6_classweight_sweep_20260530/p0.5/fit2024_score2025/pred2025.csv"
PATCH_H6_2026 = ROOT / "tmp/causal_regen_20260516/ai_patchmix_h6_classweight_sweep_20260530/p0.5/fit2025_score2026/pred2026.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Chronos h6 zero-shot features and test direction heads.")
    p.add_argument("--train-csv", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--score-csv", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model-id", default="amazon/chronos-t5-tiny")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--iterations", type=int, default=650)
    p.add_argument("--learning-rate", type=float, default=0.025)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--l2-leaf-reg", type=float, default=12.0)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--random-seed", type=int, default=20260530)
    p.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _chronos_features(frame: pd.DataFrame, *, out_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    if out_path.exists():
        got = pd.read_csv(out_path)
        got["timestamp"] = pd.to_datetime(got["timestamp"], errors="raise")
        return got
    from chronos import ChronosPipeline

    pipe = ChronosPipeline.from_pretrained(
        str(args.model_id),
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True,
    )
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().bfill().clip(lower=1e-12)
    x = np.log(close.to_numpy(dtype=np.float32))
    indices = np.arange(int(args.context_length), len(frame), max(1, int(args.stride)), dtype=np.int64)
    if indices.size == 0 or indices[-1] != len(frame) - 1:
        indices = np.append(indices, len(frame) - 1)
    cols = ["chronos_h6_q10", "chronos_h6_q50", "chronos_h6_q90", "chronos_h6_width", "chronos_h6_mean"]
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols, dtype="float32")
    qlevels = [0.1, 0.5, 0.9]
    with torch.no_grad():
        for start in range(0, len(indices), max(1, int(args.batch_size))):
            batch_idx = indices[start : start + int(args.batch_size)]
            windows = [torch.as_tensor(x[i - int(args.context_length) : i], dtype=torch.float32) for i in batch_idx]
            quantiles, mean = pipe.predict_quantiles(windows, prediction_length=6, quantile_levels=qlevels)
            q = quantiles[:, -1, :].detach().float().cpu().numpy()
            m = mean[:, -1].detach().float().cpu().numpy()
            cur = x[batch_idx]
            vals = np.column_stack([q[:, 0] - cur, q[:, 1] - cur, q[:, 2] - cur, q[:, 2] - q[:, 0], m - cur])
            out.loc[batch_idx, cols] = vals.astype("float32")
    out[cols] = out[cols].ffill().fillna(0.0)
    result = pd.concat([frame[["timestamp"]].reset_index(drop=True), out.reset_index(drop=True)], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    return result


def _merge_by_timestamp(left: pd.DataFrame, right: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    merged = left[["timestamp"]].merge(right[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    bad = [c for c in cols if merged[c].replace([np.inf, -np.inf], np.nan).isna().any()]
    if bad:
        raise RuntimeError(f"exact timestamp feature merge produced missing values: {bad}")
    return merged[cols].astype("float32")


def _fit_eval(name: str, train: pd.DataFrame, score: pd.DataFrame, x_train: pd.DataFrame, x_score: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    cfg = {"min_edge": 0.0012, "atr_mult": 0.18, "mae_penalty": 0.40, "cost": 0.00055, "margin": 0.00025}
    lab_train = _labels(train, horizon=6, **cfg)
    lab_score = _labels(score, horizon=6, **cfg)
    cols = list(x_train.columns)
    fill = _fit_values(x_train, cols)
    data = pd.concat([x_train.reset_index(drop=True), lab_train.reset_index(drop=True)], axis=1)
    data = data[data["valid"] > 0].reset_index(drop=True)
    split = int(len(data) * 0.82)
    fit_df = data.iloc[:split].reset_index(drop=True)
    hold_df = data.iloc[split:].reset_index(drop=True)
    x_fit = _matrix(fit_df, cols, fill)
    y_fit = fit_df["label"].to_numpy(dtype=np.int64)
    x_hold = _matrix(hold_df, cols, fill)
    y_hold = hold_df["label"].to_numpy(dtype=np.int64)
    score_valid = lab_score["valid"].to_numpy() > 0
    x_score_valid = _matrix(x_score.loc[score_valid].reset_index(drop=True), cols, fill)
    y_score = lab_score.loc[score_valid, "label"].to_numpy(dtype=np.int64)
    params = dict(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        iterations=int(args.iterations),
        learning_rate=float(args.learning_rate),
        depth=int(args.depth),
        l2_leaf_reg=float(args.l2_leaf_reg),
        random_seed=int(args.random_seed) + abs(hash(name)) % 10000,
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
    except Exception:
        params["task_type"] = "CPU"
        model = CatBoostClassifier(**params)
        model.fit(Pool(x_fit, y_fit), eval_set=Pool(x_hold, y_hold), use_best_model=True)
    p_hold = np.asarray(model.predict_proba(x_hold), dtype=np.float64)
    p_score = np.asarray(model.predict_proba(x_score_valid), dtype=np.float64)
    pred = np.argmax(p_score, axis=1)
    path = args.out_dir / f"{name}.cbm"
    model.save_model(path)
    out: dict[str, Any] = {
        "model": name,
        "features": cols,
        "model_path": str(path),
        "best_iteration": int(model.get_best_iteration() or 0),
        "hold_bacc": float(balanced_accuracy_score(y_hold, np.argmax(p_hold, axis=1))),
        "score_bacc": float(balanced_accuracy_score(y_score, pred)),
        "score_pred_counts": np.bincount(pred, minlength=3).astype(int).tolist(),
        "score_confusion": confusion_matrix(y_score, pred, labels=[0, 1, 2]).astype(int).tolist(),
    }
    try:
        out["hold_auc"] = float(roc_auc_score(y_hold, p_hold, multi_class="ovr"))
        out["score_auc"] = float(roc_auc_score(y_score, p_score, multi_class="ovr"))
    except Exception:
        out["hold_auc"] = None
        out["score_auc"] = None
    return out


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = _read_frame(args.train_csv, int(args.limit))
    score = _read_frame(args.score_csv, int(args.limit))
    chrono_train = _chronos_features(train, out_path=args.out_dir / "chronos_h6_2025.csv", args=args)
    chrono_score = _chronos_features(score, out_path=args.out_dir / "chronos_h6_2026.csv", args=args)
    chrono_cols = [c for c in chrono_train.columns if c.startswith("chronos_h6_")]
    x_ch_train = _merge_by_timestamp(train, chrono_train, chrono_cols)
    x_ch_score = _merge_by_timestamp(score, chrono_score, chrono_cols)
    core_cols = [
        "ret_1",
        "ret_3",
        "ret_6",
        "atr14_pct",
        "realized_vol_24",
        "funding_pressure",
        "oi_change_rate",
        "net_taker_ratio",
        "cvp_volume_imbalance",
    ]
    local_cols = [
        "cvp_regime",
        "regime_trending",
        "regime_persistence",
    ]
    x_core_train = pd.concat(
        [
            _core_features(train).loc[:, core_cols].reset_index(drop=True),
            train.loc[:, local_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).reset_index(drop=True),
        ],
        axis=1,
    )
    x_core_score = pd.concat(
        [
            _core_features(score).loc[:, core_cols].reset_index(drop=True),
            score.loc[:, local_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).reset_index(drop=True),
        ],
        axis=1,
    )
    patch_cols = [
        "ai_patch_h6_p_flat",
        "ai_patch_h6_p_down",
        "ai_patch_h6_p_up",
        "ai_patch_h6_edge",
        "ai_patch_h6_conf",
        "ai_patch_h6_entropy",
    ]
    patch_train = pd.read_csv(PATCH_H6_2025)
    patch_score = pd.read_csv(PATCH_H6_2026)
    patch_train["timestamp"] = pd.to_datetime(patch_train["timestamp"], errors="raise")
    patch_score["timestamp"] = pd.to_datetime(patch_score["timestamp"], errors="raise")
    x_patch_train = _merge_by_timestamp(train, patch_train, patch_cols)
    x_patch_score = _merge_by_timestamp(score, patch_score, patch_cols)
    variants = {
        "chronos_only": (x_ch_train, x_ch_score),
        "chronos_core": (pd.concat([x_ch_train, x_core_train], axis=1), pd.concat([x_ch_score, x_core_score], axis=1)),
        "chronos_patch": (pd.concat([x_ch_train, x_patch_train], axis=1), pd.concat([x_ch_score, x_patch_score], axis=1)),
        "chronos_patch_core": (
            pd.concat([x_ch_train, x_patch_train, x_core_train], axis=1),
            pd.concat([x_ch_score, x_patch_score, x_core_score], axis=1),
        ),
    }
    results = [_fit_eval(name, train, score, xt, xs, args) for name, (xt, xs) in variants.items()]
    summary = {
        "type": "ai_chronos_h6_direction_20260530",
        "label_config": {"min_edge": 0.0012, "atr_mult": 0.18, "mae_penalty": 0.40, "cost": 0.00055, "margin": 0.00025},
        "results": sorted(results, key=lambda x: float(x["score_bacc"]), reverse=True),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
