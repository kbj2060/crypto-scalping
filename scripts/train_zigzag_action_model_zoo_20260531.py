#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/zigzag_action_model_zoo_20260531"

ACTION_NAMES = {0: "cash", 1: "long", 2: "short"}

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
FORBIDDEN_TOKENS = (
    "label",
    "target",
    "future",
    "pnl",
    "action_score",
    "cash_after",
    "zigzag_",
    "wave3_",
)
FORBIDDEN_NAMES = {
    "timestamp",
    "tp_sl_action_score",
    "regime_bull",
    "regime_bear",
    "regime_chop",
    "regime_whipsaw",
    "regime_normal",
}

EXACT_ACTION_LABEL_OWNER_INVENTORY = [
    {
        "family": "m7_trend_xgb",
        "path": "ensemble/supervised/train_trend_xgb.py",
        "original_contract": "2-class DOWN/UP LightGBM over triple-barrier labels; excludes FLAT.",
        "zigzag_test": "trend_xgb_like_lgbm, trend_xgb_like_xgb",
    },
    {
        "family": "m7_multitarget_lgbm",
        "path": "ensemble/supervised/train_multitarget_lgbm.py",
        "original_contract": "direction classifier plus quality/hold regressors over triple-barrier labels.",
        "zigzag_test": "multitarget_lgbm_like",
    },
    {
        "family": "m7_quantile_forest",
        "path": "ensemble/supervised/train_quantile_forest.py",
        "original_contract": "future-return quantile regressors; direction is derived from q50 threshold, not a direct action head.",
        "zigzag_test": "quantile_feature_like_lgbm",
    },
    {
        "family": "alpha5_hgb_action_master",
        "path": "scripts/tune_alpha5_9_hgb_action_master_20260518.py",
        "original_contract": "3-class action master over lifecycle labels.",
        "zigzag_test": "alpha_hgb_action_master_like",
    },
    {
        "family": "alpha5_direction_master_lgbm_hgb",
        "path": "scripts/train_eval_alpha5_11_hgb_direction_master_20260518.py; scripts/train_eval_alpha5_24_lgbm_gpu_direction_refined_20260519.py",
        "original_contract": "direction/action parent variants over older alpha labels.",
        "zigzag_test": "alpha_lgbm_action_master_like",
    },
    {
        "family": "alpha6_catboost_policy",
        "path": "scripts/alpha6_catboost_fixed_barrier_policy_20260522.py",
        "original_contract": "CatBoost policy heads over fixed/triple-barrier alpha6 labels.",
        "zigzag_test": "alpha_catboost_action_master_like",
    },
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    feature_profile: str
    max_features: int


MODEL_SPECS = [
    ModelSpec("trend_xgb_like_lgbm", "lgbm", "trend_contract", 96),
    ModelSpec("trend_xgb_like_xgb", "xgb", "trend_contract", 96),
    ModelSpec("multitarget_lgbm_like", "lgbm", "m7_common_contract", 128),
    ModelSpec("quantile_feature_like_lgbm", "lgbm", "quantile_common_contract", 128),
    ModelSpec("alpha_hgb_action_master_like", "hgb", "all_sanitized", 220),
    ModelSpec("alpha_lgbm_action_master_like", "lgbm", "all_sanitized", 220),
    ModelSpec("alpha_catboost_action_master_like", "catboost", "all_sanitized", 220),
]


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
    labels = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if "wave3_action" in labels.columns:
        raise ValueError(f"{path} contains removed active contract column: wave3_action")
    required = {"timestamp", "zigzag_action"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    labels = labels[["timestamp", "zigzag_action"]].dropna(subset=["timestamp"]).drop_duplicates("timestamp", keep="last")
    values = sorted(pd.to_numeric(labels["zigzag_action"], errors="raise").astype(int).unique().tolist())
    if values != [0, 1, 2]:
        raise RuntimeError(f"{path} label value guard failed: expected=[0,1,2] actual={values}")
    return labels


def _join_labels(frame: pd.DataFrame, labels: pd.DataFrame, source: str) -> pd.DataFrame:
    before = len(frame)
    out = frame.merge(labels, on="timestamp", how="left", validate="one_to_one")
    if len(out) != before:
        raise RuntimeError(f"{source} label join changed rows: {before}->{len(out)}")
    miss = int(out["zigzag_action"].isna().sum())
    if miss:
        raise RuntimeError(f"{source} label join missing rows: {miss}")
    out["zigzag_action"] = pd.to_numeric(out["zigzag_action"], errors="raise").astype(np.int64)
    return out


def _is_forbidden(col: str) -> bool:
    if col in FORBIDDEN_NAMES:
        return True
    if col.startswith(FORBIDDEN_PREFIXES):
        return True
    lower = col.lower()
    return any(tok in lower for tok in FORBIDDEN_TOKENS)


def _numeric_common_cols(train: pd.DataFrame, score: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in train.columns:
        if col not in score.columns or _is_forbidden(col) or col == "zigzag_action":
            continue
        if pd.api.types.is_numeric_dtype(train[col]) and pd.api.types.is_numeric_dtype(score[col]):
            cols.append(col)
    if not cols:
        raise RuntimeError("no numeric feature columns selected")
    return cols


def _known_existing(cols: set[str], names: list[str]) -> list[str]:
    return [c for c in names if c in cols]


def _profile_candidates(train: pd.DataFrame, score: pd.DataFrame, profile: str) -> list[str]:
    cols = _numeric_common_cols(train, score)
    colset = set(cols)
    trend_cols = _known_existing(
        colset,
        [
            "ret_12",
            "ret_24",
            "ret_48",
            "trend_accel",
            "hh_count_24",
            "hl_count_24",
            "mtf_trend_1h",
            "mtf_trend_4h",
            "range_contraction_breakout_dir",
            "momentum",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
        ],
    )
    risk_cols = _known_existing(
        colset,
        [
            "atr",
            "atr_pct",
            "atr14_pct",
            "volatility",
            "bb_width",
            "garch_vol_z",
            "jump_flag",
            "jump_z",
            "evt_tail_flag",
            "evt_excess_z",
            "mshd",
            "fvci",
            "wpad",
            "fdlv",
            "vsdi",
            "vebr",
            "tlad",
            "mtmb",
            "fcsz",
        ],
    )
    micro_cols = [c for c in cols if any(tok in c.lower() for tok in ("funding", "oi", "flow", "taker", "whale", "cvd", "amihud", "spread", "liquid"))]
    high_order_cols = [c for c in cols if c.startswith(("state_", "hos_", "ho_", "regime_", "cvp_regime", "regime_trending", "regime_persistence"))]

    if profile == "trend_contract":
        selected = trend_cols + risk_cols + micro_cols + high_order_cols
    elif profile == "m7_common_contract":
        selected = trend_cols + risk_cols + micro_cols + high_order_cols
        selected += [c for c in cols if c.startswith(("sig_",))]
    elif profile == "quantile_common_contract":
        selected = trend_cols + risk_cols + micro_cols + high_order_cols
        selected += [c for c in cols if any(tok in c.lower() for tok in ("quant", "q10", "q50", "q90", "uncert", "width"))]
    elif profile == "all_sanitized":
        selected = cols
    else:
        raise ValueError(f"unknown feature profile: {profile}")

    deduped: list[str] = []
    for col in selected:
        if col in colset and col not in deduped:
            deduped.append(col)
    if not deduped:
        raise RuntimeError(f"profile selected no features: {profile}")
    return deduped


def _rank_features(train: pd.DataFrame, cols: list[str], y: np.ndarray, max_features: int, sample_rows: int, seed: int) -> list[str]:
    if len(cols) <= max_features:
        return list(cols)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(train))
    if len(idx) > sample_rows:
        idx = np.sort(rng.choice(idx, size=sample_rows, replace=False))
    x = train.iloc[idx][cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_s = y[idx].astype(np.int64)
    med = x.median(axis=0).fillna(0.0)
    x = x.fillna(med)
    scores: list[tuple[float, str]] = []
    for col in cols:
        arr = x[col].to_numpy(dtype=np.float64)
        if not np.isfinite(arr).any() or float(np.nanstd(arr)) <= 1e-12:
            continue
        s = 0.0
        for cls in (0, 1, 2):
            yy = (y_s == cls).astype(np.float64)
            if yy.std() <= 1e-12:
                continue
            cc = np.corrcoef(arr, yy)[0, 1]
            if np.isfinite(cc):
                s = max(s, abs(float(cc)))
        miss_penalty = float(x[col].isna().mean()) if x[col].isna().any() else 0.0
        scores.append((s - 0.02 * miss_penalty, col))
    scores.sort(reverse=True)
    ranked = [col for _, col in scores[:max_features]]
    if len(ranked) < min(max_features, len(cols)):
        for col in cols:
            if col not in ranked:
                ranked.append(col)
            if len(ranked) >= max_features:
                break
    return ranked


def _prep(train: pd.DataFrame, score: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    x_train = train[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_score = score[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = x_train.median(axis=0).fillna(0.0)
    return x_train.fillna(med), x_score.fillna(med), {k: float(v) for k, v in med.to_dict().items()}


def _class_weight(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=3).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / max(float(weights.mean()), 1e-12)
    return weights[y.astype(np.int64)]


def _align_proba(model: Any, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = np.asarray(getattr(model, "classes_", [0, 1, 2]), dtype=np.int64)
    out = np.zeros((len(x), 3), dtype=np.float64)
    for j, cls in enumerate(classes):
        if int(cls) in (0, 1, 2):
            out[:, int(cls)] = raw[:, j]
    row_sum = out.sum(axis=1, keepdims=True)
    out = np.divide(out, np.maximum(row_sum, 1e-12))
    return out


def _fit_model(spec: ModelSpec, x_train: pd.DataFrame, y_train: np.ndarray, seed: int, n_jobs: int, gpu: bool) -> Any:
    if spec.backend == "hgb":
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.035,
            max_iter=450,
            max_leaf_nodes=31,
            l2_regularization=0.08,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=seed,
        )
        model.fit(x_train, y_train, sample_weight=_class_weight(y_train))
        return model

    if spec.backend == "lgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=650,
            learning_rate=0.035,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.82,
            min_child_samples=35,
            reg_alpha=0.02,
            reg_lambda=0.08,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
        model.fit(x_train, y_train, sample_weight=_class_weight(y_train))
        return model

    if spec.backend == "xgb":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=500,
            learning_rate=0.035,
            max_depth=5,
            min_child_weight=30,
            subsample=0.85,
            colsample_bytree=0.82,
            reg_alpha=0.02,
            reg_lambda=0.12,
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
            eval_metric="mlogloss",
        )
        model.fit(x_train, y_train, sample_weight=_class_weight(y_train))
        return model

    if spec.backend == "catboost":
        from catboost import CatBoostClassifier

        params = {
            "loss_function": "MultiClass",
            "iterations": 650,
            "depth": 6,
            "learning_rate": 0.035,
            "l2_leaf_reg": 5.0,
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": n_jobs,
        }
        if gpu:
            try:
                model = CatBoostClassifier(**params, task_type="GPU", devices="0")
                model.fit(x_train, y_train, sample_weight=_class_weight(y_train))
                return model
            except Exception as exc:
                print(f"[WARN] catboost GPU failed, falling back to CPU: {exc}", flush=True)
        model = CatBoostClassifier(**params, task_type="CPU")
        model.fit(x_train, y_train, sample_weight=_class_weight(y_train))
        return model

    raise ValueError(f"unknown backend: {spec.backend}")


def _metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(proba, axis=1).astype(np.int64)
    out: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "label_counts": {ACTION_NAMES[i]: int(v) for i, v in enumerate(np.bincount(y.astype(np.int64), minlength=3))},
        "pred_counts": {ACTION_NAMES[i]: int(v) for i, v in enumerate(np.bincount(pred, minlength=3))},
    }
    try:
        out["ovr_auc"] = float(roc_auc_score(y, proba, multi_class="ovr", labels=[0, 1, 2]))
    except ValueError:
        out["ovr_auc"] = None
    return out


def _score_frame(frame: pd.DataFrame, proba: np.ndarray, prefix: str) -> pd.DataFrame:
    out = frame[["timestamp"]].copy()
    out[f"{prefix}_p_cash"] = proba[:, 0].astype(np.float32)
    out[f"{prefix}_p_long"] = proba[:, 1].astype(np.float32)
    out[f"{prefix}_p_short"] = proba[:, 2].astype(np.float32)
    out[f"{prefix}_action"] = np.argmax(proba, axis=1).astype(np.int8)
    out[f"{prefix}_confidence"] = np.max(proba, axis=1).astype(np.float32)
    out[f"{prefix}_side_edge"] = (proba[:, 1] - proba[:, 2]).astype(np.float32)
    out[f"{prefix}_trade_prob"] = (1.0 - proba[:, 0]).astype(np.float32)
    return out


def _train_score_pair(
    spec: ModelSpec,
    *,
    train: pd.DataFrame,
    score: pd.DataFrame,
    train_year: int,
    score_year: int,
    out_dir: Path,
    seed: int,
    n_jobs: int,
    rank_sample_rows: int,
    gpu: bool,
) -> dict[str, Any]:
    y_train = train["zigzag_action"].to_numpy(dtype=np.int64)
    y_score = score["zigzag_action"].to_numpy(dtype=np.int64)
    candidates = _profile_candidates(train, score, spec.feature_profile)
    selected = _rank_features(train, candidates, y_train, spec.max_features, rank_sample_rows, seed)
    x_train, x_score, med = _prep(train, score, selected)

    model = _fit_model(spec, x_train, y_train, seed, n_jobs, gpu)
    train_proba = _align_proba(model, x_train)
    score_proba = _align_proba(model, x_score)

    pair_dir = out_dir / spec.name
    pair_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"zigzag_{spec.name}"
    scores_path = pair_dir / f"{spec.name}_train{train_year}_score{score_year}.csv"
    model_path = pair_dir / f"{spec.name}_train{train_year}_score{score_year}.joblib"
    _score_frame(score, score_proba, prefix).to_csv(scores_path, index=False)
    joblib.dump(
        {
            "model": model,
            "spec": spec.__dict__,
            "feature_cols": selected,
            "median": med,
            "train_year": train_year,
            "score_year": score_year,
            "label_contract": "zigzag_action_3class",
            "forbidden_prefixes": FORBIDDEN_PREFIXES,
            "forbidden_tokens": FORBIDDEN_TOKENS,
        },
        model_path,
    )
    return {
        "model": spec.name,
        "backend": spec.backend,
        "feature_profile": spec.feature_profile,
        "train_year": int(train_year),
        "score_year": int(score_year),
        "feature_count": int(len(selected)),
        "top_features": selected[:40],
        "model_path": str(model_path),
        "scores_path": str(scores_path),
        "train_metrics": _metrics(y_train, train_proba),
        "score_metrics": _metrics(y_score, score_proba),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Retrain direct action-label model families on the active ZigZag 3-class label contract.")
    p.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--train-2024", type=Path, default=ROOT / "data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--score-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--train-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--score-2026", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--models", nargs="*", default=[s.name for s in MODEL_SPECS])
    p.add_argument("--seed", type=int, default=20260531)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--rank-sample-rows", type=int, default=80000)
    p.add_argument("--no-gpu", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = [s for s in MODEL_SPECS if s.name in set(args.models)]
    if not specs:
        raise RuntimeError(f"no model specs selected: {args.models}")

    train2024 = _join_labels(_read_frame(args.train_2024, expected_year=2024), _read_labels(args.label_dir, 2024), "train2024")
    score2025 = _join_labels(_read_frame(args.score_2025, expected_year=2025), _read_labels(args.label_dir, 2025), "score2025")
    train2025 = _join_labels(_read_frame(args.train_2025, expected_year=2025), _read_labels(args.label_dir, 2025), "train2025")
    score2026 = _join_labels(_read_frame(args.score_2026, expected_year=2026), _read_labels(args.label_dir, 2026), "score2026")

    results: list[dict[str, Any]] = []
    for i, spec in enumerate(specs):
        print(f"[INFO] training {spec.name} 2024->2025", flush=True)
        results.append(
            _train_score_pair(
                spec,
                train=train2024,
                score=score2025,
                train_year=2024,
                score_year=2025,
                out_dir=args.out_dir,
                seed=args.seed + i * 101,
                n_jobs=args.n_jobs,
                rank_sample_rows=args.rank_sample_rows,
                gpu=not args.no_gpu,
            )
        )
        print(f"[INFO] training {spec.name} 2025->2026", flush=True)
        results.append(
            _train_score_pair(
                spec,
                train=train2025,
                score=score2026,
                train_year=2025,
                score_year=2026,
                out_dir=args.out_dir,
                seed=args.seed + i * 101 + 17,
                n_jobs=args.n_jobs,
                rank_sample_rows=args.rank_sample_rows,
                gpu=not args.no_gpu,
            )
        )

    summary = {
        "label_contract": {
            "name": "zigzag_action_3class",
            "label_dir": str(args.label_dir),
            "label_column": "zigzag_action",
            "classes": ACTION_NAMES,
            "removed_columns_guard": ["wave3_action"],
        },
        "input_contract": {
            "forbidden_prefixes": FORBIDDEN_PREFIXES,
            "forbidden_tokens": FORBIDDEN_TOKENS,
            "forbidden_names": sorted(FORBIDDEN_NAMES),
            "note": "No silent alias/fallback is used. Missing ZigZag labels or legacy wave3_action columns fail fast.",
        },
        "action_label_owner_inventory": EXACT_ACTION_LABEL_OWNER_INVENTORY,
        "model_specs": [s.__dict__ for s in specs],
        "results": results,
    }
    summary_path = args.out_dir / "zigzag_action_model_zoo_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    flat_rows = []
    for r in results:
        sm = r["score_metrics"]
        tm = r["train_metrics"]
        flat_rows.append(
            {
                "model": r["model"],
                "backend": r["backend"],
                "feature_profile": r["feature_profile"],
                "train_year": r["train_year"],
                "score_year": r["score_year"],
                "feature_count": r["feature_count"],
                "train_bacc": tm["balanced_accuracy"],
                "train_ovr_auc": tm["ovr_auc"],
                "score_bacc": sm["balanced_accuracy"],
                "score_ovr_auc": sm["ovr_auc"],
                "score_pred_cash": sm["pred_counts"]["cash"],
                "score_pred_long": sm["pred_counts"]["long"],
                "score_pred_short": sm["pred_counts"]["short"],
                "scores_path": r["scores_path"],
                "model_path": r["model_path"],
            }
        )
    pd.DataFrame(flat_rows).sort_values(["score_year", "score_bacc"], ascending=[True, False]).to_csv(
        args.out_dir / "zigzag_action_model_zoo_flat_metrics.csv",
        index=False,
    )
    print(json.dumps({"summary": str(summary_path), "rows": len(results)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
