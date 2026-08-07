#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/zigzag_m7_action_hgb_20260531"

FORBIDDEN_PREFIXES = (
    "m7_",
    "teacher_",
    "a5dir_",
    "ai_",
    "pred_",
    "conf_",
    "clean_regime4_",
    "regime4_pred_",
    "regime3_pred_",
)
FORBIDDEN_TOKENS = ("label", "target", "future", "pnl", "action_score", "cash_after")
FORBIDDEN_NAMES = {"timestamp", "tp_sl_action_score"}


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


def _read_frame(path: Path, *, expected_year: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path} missing timestamp")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    years = sorted(frame["timestamp"].dt.year.dropna().astype(int).unique().tolist())
    if years != [int(expected_year)]:
        raise RuntimeError(f"{path} year guard failed: expected={[int(expected_year)]} actual={years}")
    return frame.replace([np.inf, -np.inf], np.nan)


def _read_labels(label_dir: Path, year: int) -> pd.DataFrame:
    path = label_dir / f"zigzag_action_labels_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    labels = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "zigzag_action"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    if "wave3_action" in labels.columns:
        raise ValueError(f"{path} contains removed active contract column: wave3_action")
    return labels[["timestamp", "zigzag_action"]].drop_duplicates("timestamp", keep="last")


def _join_labels(frame: pd.DataFrame, labels: pd.DataFrame, source: str) -> pd.DataFrame:
    before = len(frame)
    out = frame.merge(labels, on="timestamp", how="left", validate="one_to_one")
    if len(out) != before:
        raise RuntimeError(f"{source} label join changed rows: {before}->{len(out)}")
    miss = int(out["zigzag_action"].isna().sum())
    if miss:
        raise RuntimeError(f"{source} label join missing rows: {miss}")
    out["zigzag_action"] = out["zigzag_action"].astype(np.int64)
    return out


def _is_forbidden(col: str) -> bool:
    if col in FORBIDDEN_NAMES:
        return True
    if col.startswith(FORBIDDEN_PREFIXES):
        return True
    lower = col.lower()
    return any(tok in lower for tok in FORBIDDEN_TOKENS)


def _feature_cols(train: pd.DataFrame, score: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in train.columns:
        if col not in score.columns or _is_forbidden(col):
            continue
        if col == "zigzag_action":
            continue
        if pd.api.types.is_numeric_dtype(train[col]) and pd.api.types.is_numeric_dtype(score[col]):
            cols.append(col)
    if not cols:
        raise RuntimeError("no feature columns selected")
    return cols


def _prep(train: pd.DataFrame, score: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    x_train = train[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_score = score[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = x_train.median(axis=0).fillna(0.0)
    return x_train.fillna(med), x_score.fillna(med), {k: float(v) for k, v in med.to_dict().items()}


def _sample_weight(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=3).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return weights[y.astype(np.int64)]


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
    out["m7_zigzag_p_cash"] = proba[:, 0].astype(np.float32)
    out["m7_zigzag_p_long"] = proba[:, 1].astype(np.float32)
    out["m7_zigzag_p_short"] = proba[:, 2].astype(np.float32)
    out["m7_zigzag_action"] = np.argmax(proba, axis=1).astype(np.int8)
    out["m7_zigzag_confidence"] = np.max(proba, axis=1).astype(np.float32)
    out["m7_zigzag_side_edge"] = (proba[:, 1] - proba[:, 2]).astype(np.float32)
    out["m7_zigzag_trade_prob"] = (1.0 - proba[:, 0]).astype(np.float32)
    return out


def _train_score_pair(
    *,
    train_year: int,
    score_year: int,
    train_path: Path,
    score_path: Path,
    label_dir: Path,
    out_dir: Path,
    max_iter: int,
    learning_rate: float,
    max_leaf_nodes: int,
    l2_regularization: float,
    seed: int,
) -> dict[str, Any]:
    train = _join_labels(_read_frame(train_path, expected_year=train_year), _read_labels(label_dir, train_year), f"train{train_year}")
    score = _join_labels(_read_frame(score_path, expected_year=score_year), _read_labels(label_dir, score_year), f"score{score_year}")
    cols = _feature_cols(train, score)
    x_train, x_score, med = _prep(train, score, cols)
    y_train = train["zigzag_action"].to_numpy(dtype=np.int64)
    y_score = score["zigzag_action"].to_numpy(dtype=np.int64)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=float(learning_rate),
        max_iter=int(max_iter),
        max_leaf_nodes=int(max_leaf_nodes),
        l2_regularization=float(l2_regularization),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=int(seed),
    )
    model.fit(x_train, y_train, sample_weight=_sample_weight(y_train))
    train_proba = model.predict_proba(x_train)
    score_proba = model.predict_proba(x_score)
    if train_proba.shape[1] != 3 or score_proba.shape[1] != 3:
        raise RuntimeError("zigzag model did not learn all 3 classes")
    score_out = _append_scores(score, score_proba)
    score_csv = out_dir / f"m7_zigzag_scores_train{train_year}_score{score_year}.csv"
    model_path = out_dir / f"m7_zigzag_hgb_train{train_year}_score{score_year}.joblib"
    score_out.to_csv(score_csv, index=False)
    joblib.dump({"model": model, "feature_cols": cols, "median": med, "train_year": train_year, "score_year": score_year}, model_path)
    return {
        "train_year": int(train_year),
        "score_year": int(score_year),
        "feature_count": int(len(cols)),
        "model_path": str(model_path),
        "score_csv": str(score_csv),
        "train_metrics": _metrics(y_train, train_proba),
        "score_metrics": _metrics(y_score, score_proba),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--train-2024", type=Path, default=ROOT / "data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--score-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--train-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--score-2026", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--max-iter", type=int, default=600)
    p.add_argument("--learning-rate", type=float, default=0.035)
    p.add_argument("--max-leaf-nodes", type=int, default=31)
    p.add_argument("--l2-regularization", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260531)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        _train_score_pair(
            train_year=2024,
            score_year=2025,
            train_path=args.train_2024,
            score_path=args.score_2025,
            label_dir=args.label_dir,
            out_dir=args.out_dir,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2_regularization,
            seed=args.seed,
        ),
        _train_score_pair(
            train_year=2025,
            score_year=2026,
            train_path=args.train_2025,
            score_path=args.score_2026,
            label_dir=args.label_dir,
            out_dir=args.out_dir,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2_regularization,
            seed=args.seed + 1,
        ),
    ]
    audit = {
        "type": "m7_zigzag_action_hgb",
        "label_contract": str(args.label_dir / "zigzag_action_label_audit.json"),
        "output_columns": [
            "m7_zigzag_p_cash",
            "m7_zigzag_p_long",
            "m7_zigzag_p_short",
            "m7_zigzag_action",
            "m7_zigzag_confidence",
            "m7_zigzag_side_edge",
            "m7_zigzag_trade_prob",
        ],
        "pairs": pairs,
        "contract": {
            "no_m7_teacher_ai_a5dir_inputs": True,
            "label_mapping": {"0": "CASH", "1": "LONG", "2": "SHORT"},
            "active_path_overwrite": False,
        },
    }
    audit_path = args.out_dir / "m7_zigzag_action_hgb_audit.json"
    audit["audit"] = str(audit_path)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
