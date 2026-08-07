#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha5_router_v5_train_20260520 import (  # noqa: E402
    DEFAULT_DATA_DIR,
    ROUTER_FEATURE_COLS,
    _num,
    _prepare_frame,
    _router3_label,
    _router3_weight,
)
from scripts.alpha5_router_v5_ablation_20260520 import _profit_proxy  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_router5_full_candidate_search_20260521"

EXCLUDE_EXACT = {
    "timestamp",
    "quality_score",
    "tp_sl_action_score",
    "pred_patchtst",
    "conf_patchtst",
    "regime_trade_selected",
    "clean_wait_contamination_flag",
    "trade_contamination_flag",
    "entry_train_keep",
    "direction_train_keep",
    "entry_binary_label",
}
EXCLUDE_PREFIX = (
    "label_",
    "meta_",
    "entry_",
    "direction_",
    "path_",
    "ambiguous_",
    "sample_",
    "dataset_",
    "split_",
    "__",
)
RAW_LEVEL_COLS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_base",
    "taker_buy_quote",
    "sum_open_interest_value",
    "close_btc",
    "volume_btc",
    "quote_volume_btc",
}


def _clean_x(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame[cols].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def _load_data(data_dir: Path) -> dict[str, Any]:
    raw = {
        "train": pd.read_parquet(data_dir / "alpha5_29_hier_label_factory_train.parquet"),
        "val": pd.read_parquet(data_dir / "alpha5_29_hier_label_factory_val.parquet"),
        "oos": pd.read_parquet(data_dir / "alpha5_29_hier_label_factory_oos.parquet"),
    }
    work = {k: _prepare_frame(v) for k, v in raw.items()}
    keep = {k: _num(v, "split_keep", 0.0).astype(np.int8) == 1 for k, v in raw.items()}
    y3 = {k: _router3_label(v)[keep[k]] for k, v in raw.items()}
    return {"raw": raw, "work": work, "y3": y3}


def _candidate_pool(train_df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    keep_cols: list[str] = []
    for col in train_df.columns:
        reason = ""
        dtype = str(train_df[col].dtype)
        if col in EXCLUDE_EXACT:
            reason = "excluded_exact"
        elif col.startswith(EXCLUDE_PREFIX):
            reason = "excluded_prefix"
        elif dtype in {"object", "string"}:
            reason = "excluded_object"
        else:
            keep_cols.append(col)
            reason = "candidate"
        rows.append({"feature": col, "dtype": dtype, "status": reason})
    return keep_cols, pd.DataFrame(rows)


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 1e-9, None)
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-9)


def _predict_proba_3(model: CatBoostClassifier, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = [int(c) for c in getattr(model, "classes_", [0, 1, 2])]
    idx = {c: i for i, c in enumerate(classes)}
    return _normalize(np.stack([raw[:, idx[0]], raw[:, idx[1]], raw[:, idx[2]]], axis=1))


def _metrics(y: np.ndarray, p: np.ndarray, work: pd.DataFrame) -> dict[str, Any]:
    pred = np.asarray(p, dtype=np.float64).argmax(axis=1).astype(np.int64)
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "pred_counts": {str(int(k)): int(v) for k, v in pd.Series(pred).value_counts().sort_index().to_dict().items()},
        "class_counts": {str(int(k)): int(v) for k, v in pd.Series(y).value_counts().sort_index().to_dict().items()},
    }
    out.update(_profit_proxy(work, pred))
    return out


def _quantile_edges(train: np.ndarray, bins: int) -> np.ndarray:
    base = train[np.isfinite(train)]
    if base.size == 0:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    q = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(base, q))
    if edges.size < 2:
        v = float(edges[0]) if edges.size == 1 else 0.0
        return np.array([v - 1.0, v + 1.0], dtype=np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.astype(np.float64)


def _psi(train: np.ndarray, other: np.ndarray, bins: int) -> float:
    edges = _quantile_edges(train, bins)
    a, _ = np.histogram(train[np.isfinite(train)], bins=edges)
    b, _ = np.histogram(other[np.isfinite(other)], bins=edges)
    ap = np.clip(a / max(a.sum(), 1), 1e-9, None)
    bp = np.clip(b / max(b.sum(), 1), 1e-9, None)
    return float(np.sum((bp - ap) * np.log(bp / ap)))


def _fit_probe(x_train: pd.DataFrame, y_train: np.ndarray, w_train: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray, seed: int, *, task_type: str, devices: str) -> CatBoostClassifier:
    kwargs = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": 320,
        "depth": 6,
        "learning_rate": 0.04,
        "l2_leaf_reg": 6.0,
        "random_strength": 1.0,
        "bagging_temperature": 0.0,
        "random_seed": seed,
        "allow_writing_files": False,
        "verbose": False,
    }
    if task_type.upper() == "GPU":
        kwargs["task_type"] = "GPU"
        kwargs["devices"] = devices
    else:
        kwargs["task_type"] = "CPU"
    model = CatBoostClassifier(**kwargs)
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        eval_set=(x_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    return model


def _feature_family(name: str) -> str:
    if name.startswith("clean_regime4_2024_unsup_v1_"):
        return "clean_regime4"
    if name.startswith("regime4_pred_"):
        return "regime4_pred"
    if name.startswith("m7_"):
        return "m7"
    if name.startswith("teacher_"):
        return "teacher"
    if name.startswith("ai_") or name.startswith("patchtst_") or name.startswith("tide_") or name.startswith("dlinear_"):
        return "ai_family"
    if "funding" in name or "taker" in name or "whale" in name or "crowding" in name or "flow" in name or "ofi" in name:
        return "micro_flow"
    if "trend" in name or "mom" in name or "rsi" in name or "breakout" in name:
        return "trend"
    if name in RAW_LEVEL_COLS:
        return "raw_level"
    return "other"


def _build_feature_stats(data: dict[str, Any], candidate_cols: list[str], bins: int, seed: int) -> pd.DataFrame:
    x_train = _clean_x(data["work"]["train"], candidate_cols)
    x_val = _clean_x(data["work"]["val"], candidate_cols)
    x_oos = _clean_x(data["work"]["oos"], candidate_cols)
    train_med = x_train.median(numeric_only=True)
    filled_train = x_train.fillna(train_med).fillna(0.0)
    mi = mutual_info_classif(filled_train.to_numpy(np.float64), data["y3"]["train"].astype(np.int64), discrete_features=False, random_state=seed)
    rows: list[dict[str, Any]] = []
    for j, col in enumerate(candidate_cols):
        tr = x_train[col].to_numpy(np.float64)
        va = x_val[col].to_numpy(np.float64)
        oo = x_oos[col].to_numpy(np.float64)
        rows.append(
            {
                "feature": col,
                "family": _feature_family(col),
                "train_missing_ratio": float(np.mean(~np.isfinite(tr))),
                "val_missing_ratio": float(np.mean(~np.isfinite(va))),
                "oos_missing_ratio": float(np.mean(~np.isfinite(oo))),
                "train_std": float(np.nanstd(tr)),
                "train_nunique": int(pd.Series(tr).nunique(dropna=True)),
                "mutual_info": float(mi[j]),
                "val_psi": _psi(tr, va, bins),
                "oos_psi": _psi(tr, oo, bins),
            }
        )
    out = pd.DataFrame(rows)
    out["max_psi"] = out[["val_psi", "oos_psi"]].max(axis=1)
    out["near_constant_flag"] = (out["train_nunique"] <= 2) | (out["train_std"].abs() < 1e-8)
    return out.sort_values(["mutual_info", "max_psi"], ascending=[False, True]).reset_index(drop=True)


def _fit_all_candidate_model(data: dict[str, Any], cols: list[str], seed: int, task_type: str, devices: str) -> CatBoostClassifier:
    x_train = _clean_x(data["work"]["train"], cols)
    x_val = _clean_x(data["work"]["val"], cols)
    y_train = data["y3"]["train"]
    y_val = data["y3"]["val"]
    w_train = _router3_weight(data["work"]["train"], y_train)
    return _fit_probe(x_train, y_train, w_train, x_val, y_val, seed, task_type=task_type, devices=devices)


def _importance_df(model: CatBoostClassifier, cols: list[str]) -> pd.DataFrame:
    imp = np.asarray(model.get_feature_importance(), dtype=np.float64)
    feat_names = list(getattr(model, "feature_names_", cols))
    n = min(len(imp), len(feat_names))
    out = pd.DataFrame({"feature": feat_names[:n], "probe_importance": imp[:n]})
    return out.sort_values("probe_importance", ascending=False).reset_index(drop=True)


def _rank_features(stats: pd.DataFrame, importance: pd.DataFrame) -> pd.DataFrame:
    work = stats.merge(importance, on="feature", how="left")
    work["probe_importance"] = work["probe_importance"].fillna(0.0)
    work["mi_rank"] = work["mutual_info"].rank(pct=True, ascending=True)
    work["imp_rank"] = work["probe_importance"].rank(pct=True, ascending=True)
    work["stability_rank"] = (-work["max_psi"]).rank(pct=True, ascending=True)
    work["missing_rank"] = (-work[["train_missing_ratio", "val_missing_ratio", "oos_missing_ratio"]].max(axis=1)).rank(pct=True, ascending=True)
    work["rank_score"] = (
        0.45 * work["imp_rank"]
        + 0.30 * work["mi_rank"]
        + 0.20 * work["stability_rank"]
        + 0.05 * work["missing_rank"]
    )
    return work.sort_values(["rank_score", "probe_importance", "mutual_info"], ascending=False).reset_index(drop=True)


def _corr_pruned_selection(x_train: pd.DataFrame, ranked_features: list[str], *, corr_threshold: float, topk: int | None = None) -> list[str]:
    corr = x_train[ranked_features].corr(numeric_only=True).abs()
    chosen: list[str] = []
    for col in ranked_features:
        if col not in corr.columns:
            continue
        keep = True
        for prev in chosen:
            if float(corr.loc[col, prev]) >= corr_threshold:
                keep = False
                break
        if keep:
            chosen.append(col)
        if topk is not None and len(chosen) >= topk:
            break
    return chosen


def _evaluate_variant(data: dict[str, Any], cols: list[str], seed: int, task_type: str, devices: str) -> dict[str, Any]:
    x_train = _clean_x(data["work"]["train"], cols)
    x_val = _clean_x(data["work"]["val"], cols)
    x_oos = _clean_x(data["work"]["oos"], cols)
    y_train = data["y3"]["train"]
    y_val = data["y3"]["val"]
    y_oos = data["y3"]["oos"]
    w_train = _router3_weight(data["work"]["train"], y_train)
    model = _fit_probe(x_train, y_train, w_train, x_val, y_val, seed, task_type=task_type, devices=devices)
    val_p = _predict_proba_3(model, x_val)
    oos_p = _predict_proba_3(model, x_oos)
    val_m = _metrics(y_val, val_p, data["work"]["val"])
    oos_m = _metrics(y_oos, oos_p, data["work"]["oos"])
    score = float(oos_m["balanced_accuracy"] + 0.40 * oos_m["macro_f1"] + 0.00005 * oos_m["pred_trade_quality_sum"])
    return {
        "feature_count": int(len(cols)),
        "features": cols,
        "families": {fam: int(sum(_feature_family(c) == fam for c in cols)) for fam in sorted({_feature_family(c) for c in cols})},
        "val": val_m,
        "oos": oos_m,
        "selection_score": score,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Broad router5 candidate-feature exploration and reselection.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task-type", choices=("CPU", "GPU"), default="CPU")
    ap.add_argument("--devices", default="0")
    ap.add_argument("--psi-bins", type=int, default=20)
    ap.add_argument("--corr-threshold", type=float, default=0.985)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_data(args.data_dir)
    candidate_cols, pool_df = _candidate_pool(data["raw"]["train"])
    stats = _build_feature_stats(data, candidate_cols, args.psi_bins, args.seed)
    usable = stats.loc[~stats["near_constant_flag"], "feature"].tolist()
    all_model = _fit_all_candidate_model(data, usable, args.seed, args.task_type, args.devices)
    imp = _importance_df(all_model, usable)
    ranked = _rank_features(stats[stats["feature"].isin(usable)].copy(), imp)
    x_train = _clean_x(data["work"]["train"], usable).fillna(0.0)
    ranked_features = ranked["feature"].tolist()
    pruned_all = _corr_pruned_selection(x_train, ranked_features, corr_threshold=float(args.corr_threshold), topk=None)
    high_stability = ranked.loc[ranked["max_psi"] <= 0.25, "feature"].tolist()
    pruned_stable = _corr_pruned_selection(x_train, high_stability, corr_threshold=float(args.corr_threshold), topk=None)
    baseline_set = [c for c in ROUTER_FEATURE_COLS if c in usable]
    new_ranked = [c for c in pruned_all if c not in baseline_set]
    no_raw_ranked = [c for c in pruned_all if c not in RAW_LEVEL_COLS]

    variant_sets: dict[str, list[str]] = {
        "baseline_current38": baseline_set,
        "all_sanitized_usable": usable,
        "rank_pruned_top32": pruned_all[:32],
        "rank_pruned_top48": pruned_all[:48],
        "rank_pruned_top64": pruned_all[:64],
        "rank_pruned_top96": pruned_all[:96],
        "rank_pruned_stable_top48": pruned_stable[:48],
        "rank_pruned_no_raw_top48": no_raw_ranked[:48],
        "baseline_plus_top16_new": baseline_set + new_ranked[:16],
        "baseline_plus_top32_new": baseline_set + new_ranked[:32],
    }
    seen: set[tuple[str, ...]] = set()
    deduped_variants: dict[str, list[str]] = {}
    for name, cols in variant_sets.items():
        uniq_cols = []
        used = set()
        for c in cols:
            if c not in used:
                uniq_cols.append(c)
                used.add(c)
        key = tuple(uniq_cols)
        if key in seen:
            continue
        seen.add(key)
        deduped_variants[name] = uniq_cols

    variant_results = {
        name: _evaluate_variant(data, cols, args.seed, args.task_type, args.devices)
        for name, cols in deduped_variants.items()
    }
    ordered_variants = sorted(variant_results.items(), key=lambda kv: kv[1]["selection_score"], reverse=True)
    best_name, best_payload = ordered_variants[0]

    pool_path = args.out_dir / "candidate_pool.csv"
    stats_path = args.out_dir / "candidate_feature_stats.csv"
    imp_path = args.out_dir / "all_candidate_probe_importance.csv"
    rank_path = args.out_dir / "candidate_feature_rank.csv"
    summary_path = args.out_dir / "router5_full_candidate_search_summary.json"
    pool_df.to_csv(pool_path, index=False)
    stats.to_csv(stats_path, index=False)
    imp.to_csv(imp_path, index=False)
    ranked.to_csv(rank_path, index=False)

    summary = {
        "model_id": "explore_alpha5_router5_full_candidate_search_20260521",
        "data_dir": str(args.data_dir),
        "task_type": args.task_type,
        "candidate_pool": {
            "all_columns": int(len(data["raw"]["train"].columns)),
            "numeric_candidates": int(len(candidate_cols)),
            "usable_after_near_constant_filter": int(len(usable)),
            "baseline_current38": int(len(baseline_set)),
        },
        "artifacts": {
            "candidate_pool_csv": str(pool_path),
            "candidate_feature_stats_csv": str(stats_path),
            "all_candidate_probe_importance_csv": str(imp_path),
            "candidate_feature_rank_csv": str(rank_path),
        },
        "top_ranked_features": ranked.head(40).to_dict(orient="records"),
        "top_probe_importance": imp.head(40).to_dict(orient="records"),
        "variant_results": {name: payload for name, payload in ordered_variants},
        "recommended_variant": {
            "name": best_name,
            **best_payload,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "numeric_candidates": len(candidate_cols),
                "usable_candidates": len(usable),
                "best_variant": best_name,
                "best_feature_count": best_payload["feature_count"],
                "best_oos_bal_acc": best_payload["oos"]["balanced_accuracy"],
                "best_oos_macro_f1": best_payload["oos"]["macro_f1"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
