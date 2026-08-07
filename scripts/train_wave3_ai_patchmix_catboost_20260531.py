#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_ai_patchmix_direction_core_20260530 as patchmix  # noqa: E402


DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/zigzag_ai_patchmix_catboost_20260531"


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


def _set_profile() -> None:
    patchmix.CORE_FEATURES = (
        *patchmix.BASE_CORE_FEATURES,
        *patchmix.AUDITED_COMPACT_FEATURES,
        *patchmix.LOCAL_REGIME_FEATURES,
    )


def _read_labels(label_dir: Path, year: int) -> pd.DataFrame:
    path = label_dir / f"zigzag_action_labels_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    labels = pd.read_csv(path, parse_dates=["timestamp"])
    if "wave3_action" in labels.columns:
        raise ValueError(f"{path} contains removed active contract column: wave3_action")
    required = {"timestamp", "zigzag_action"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return labels[["timestamp", "zigzag_action"]].drop_duplicates("timestamp", keep="last")


def _join_labels(frame: pd.DataFrame, label_dir: Path, year: int) -> pd.DataFrame:
    before = len(frame)
    out = frame.merge(_read_labels(label_dir, year), on="timestamp", how="left", validate="one_to_one")
    if len(out) != before:
        raise RuntimeError(f"label join changed rows for {year}: {before}->{len(out)}")
    miss = int(out["zigzag_action"].isna().sum())
    if miss:
        raise RuntimeError(f"label join missing rows for {year}: {miss}")
    out["zigzag_action"] = out["zigzag_action"].astype(np.int64)
    return out


def _embeddings(frame: pd.DataFrame, args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    return patchmix._patch_embeddings(
        frame,
        model_id=str(args.patch_model_id),
        context_length=int(args.context_length),
        stride=int(args.stride),
        batch_size=int(args.batch_size),
        emb_dim=int(args.emb_dim),
        device=device,
    )


def _class_weights(y: np.ndarray, power: float) -> list[float]:
    counts = np.maximum(np.bincount(y.astype(np.int64), minlength=3).astype(np.float64), 1.0)
    weights = (counts.sum() / (3.0 * counts)) ** float(power)
    return [float(v) for v in weights]


def _fit_catboost(x: pd.DataFrame, y: np.ndarray, args: argparse.Namespace) -> tuple[CatBoostClassifier, str]:
    params = {
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1",
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_seed": int(args.random_seed),
        "verbose": False,
        "allow_writing_files": False,
        "class_weights": _class_weights(y, float(args.class_weight_power)),
    }
    last_error: Exception | None = None
    for task_type in [str(args.task_type), "CPU"]:
        try:
            model = CatBoostClassifier(**params, task_type=task_type)
            model.fit(Pool(x, y))
            return model, task_type
        except Exception as exc:  # GPU can fail on driver/runtime mismatch.
            last_error = exc
            if task_type == "CPU":
                raise
    raise RuntimeError(last_error)


def _metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(proba, axis=1).astype(np.int64)
    out: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y.astype(np.int64), minlength=3))},
        "pred_counts": {str(i): int(v) for i, v in enumerate(np.bincount(pred, minlength=3))},
    }
    try:
        out["ovr_auc"] = float(roc_auc_score(y, proba, multi_class="ovr", labels=[0, 1, 2]))
    except ValueError:
        out["ovr_auc"] = None
    return out


def _append_scores(frame: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    out = frame[["timestamp"]].copy()
    out["ai_zigzag_patch_p_cash"] = proba[:, 0].astype(np.float32)
    out["ai_zigzag_patch_p_long"] = proba[:, 1].astype(np.float32)
    out["ai_zigzag_patch_p_short"] = proba[:, 2].astype(np.float32)
    out["ai_zigzag_patch_action"] = np.argmax(proba, axis=1).astype(np.int8)
    out["ai_zigzag_patch_confidence"] = np.max(proba, axis=1).astype(np.float32)
    out["ai_zigzag_patch_side_edge"] = (proba[:, 1] - proba[:, 2]).astype(np.float32)
    out["ai_zigzag_patch_trade_prob"] = (1.0 - proba[:, 0]).astype(np.float32)
    return out


def _train_score_pair(
    *,
    train_year: int,
    score_year: int,
    train_path: Path,
    score_path: Path,
    label_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    train_raw = _join_labels(patchmix._read_frame(train_path, int(args.limit)), label_dir, train_year)
    score_raw = _join_labels(patchmix._read_frame(score_path, int(args.limit)), label_dir, score_year)
    train_x = _embeddings(train_raw, args, device)
    score_x = _embeddings(score_raw, args, device)
    cols = list(train_x.columns)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    y_score = score_raw["zigzag_action"].to_numpy(dtype=np.int64)
    model, task_type = _fit_catboost(train_x[cols], y_train, args)
    train_proba = model.predict_proba(train_x[cols])
    score_proba = model.predict_proba(score_x[cols])
    score_csv = out_dir / f"ai_zigzag_patch_scores_train{train_year}_score{score_year}.csv"
    model_path = out_dir / f"ai_zigzag_patch_catboost_train{train_year}_score{score_year}.cbm"
    model.save_model(str(model_path))
    _append_scores(score_raw, score_proba).to_csv(score_csv, index=False)
    return {
        "train_year": int(train_year),
        "score_year": int(score_year),
        "task_type_used": task_type,
        "feature_count": int(len(cols)),
        "model_path": str(model_path),
        "score_csv": str(score_csv),
        "train_metrics": _metrics(y_train, train_proba),
        "score_metrics": _metrics(y_score, score_proba),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-2024", type=Path, default=ROOT / "data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--score-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--train-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--score-2026", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--patch-model-id", default="ibm/patchtsmixer-etth1-pretrain")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--emb-dim", type=int, default=16)
    p.add_argument("--iterations", type=int, default=600)
    p.add_argument("--learning-rate", type=float, default=0.035)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--l2-leaf-reg", type=float, default=8.0)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--random-seed", type=int, default=20260531)
    p.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    _set_profile()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = [
        _train_score_pair(
            train_year=2024,
            score_year=2025,
            train_path=args.train_2024,
            score_path=args.score_2025,
            label_dir=args.label_dir,
            out_dir=args.out_dir,
            args=args,
            device=device,
        ),
        _train_score_pair(
            train_year=2025,
            score_year=2026,
            train_path=args.train_2025,
            score_path=args.score_2026,
            label_dir=args.label_dir,
            out_dir=args.out_dir,
            args=args,
            device=device,
        ),
    ]
    audit = {
        "type": "ai_zigzag_patchmix_catboost",
        "label_contract": str(args.label_dir / "zigzag_action_label_audit.json"),
        "device": str(device),
        "output_columns": [
            "ai_zigzag_patch_p_cash",
            "ai_zigzag_patch_p_long",
            "ai_zigzag_patch_p_short",
            "ai_zigzag_patch_action",
            "ai_zigzag_patch_confidence",
            "ai_zigzag_patch_side_edge",
            "ai_zigzag_patch_trade_prob",
        ],
        "pairs": pairs,
        "contract": {
            "pretrained_patchtsmixer_representation": str(args.patch_model_id),
            "catboost_head_label": "zigzag_action",
            "active_path_overwrite": False,
        },
    }
    audit_path = args.out_dir / "ai_zigzag_patchmix_catboost_audit.json"
    audit["audit"] = str(audit_path)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
