#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
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


DEFAULT_BASE_META = ROOT / "tmp/causal_regen_20260516/alpha5_router_v5_train_singlefile_20260520/router_ensemble_meta.joblib"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_router5_feature_analysis_20260521"


FEATURE_GROUPS: dict[str, list[str]] = {
    "funding_flow_micro": [
        "funding_abs",
        "funding_pressure",
        "funding_price_divergence",
        "crowding_pressure",
        "smart_money_flow",
        "net_taker_ratio",
        "taker_acceleration",
        "ofi_acceleration",
        "whale_conviction",
        "big_trade_ratio",
        "whale_retail_ratio",
        "execution_quality",
    ],
    "m7_core": [
        "m7_expected_ret",
        "m7_composite_score",
        "m7_confidence",
    ],
    "ai_core": [
        "ai_dir_edge",
        "ai_dir_p_up",
        "ai_dir_p_down",
        "ai_flow_pressure",
        "ai_flow_exhaustion",
        "ai_flow_flip_prob",
        "ai_flow_slope",
        "dlinear_smf_ema",
        "dlinear_smf_slope",
    ],
    "trend_mtf": [
        "log_return",
        "mtf_trend_1h",
        "mtf_trend_4h",
        "rsi",
        "breakout_strength",
    ],
    "regime4_clean": [
        "clean_regime4_2024_unsup_v1_bear_prob",
        "clean_regime4_2024_unsup_v1_bull_prob",
        "clean_regime4_2024_unsup_v1_factor_flow",
        "clean_regime4_2024_unsup_v1_factor_trend",
        "clean_regime4_2024_unsup_v1_trend_bias",
        "clean_regime4_2024_unsup_v1_trend_prob",
        "clean_regime4_2024_unsup_v1_directional_bias",
        "clean_regime4_2024_unsup_v1_margin",
        "clean_regime4_2024_unsup_v1_whipsaw_prob",
    ],
}


def _clean_x(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
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
    x = {k: _clean_x(v[ROUTER_FEATURE_COLS]) for k, v in work.items()}
    keep = {k: _num(v, "split_keep", 0.0).astype(np.int8) == 1 for k, v in raw.items()}
    y3 = {k: _router3_label(v)[keep[k]] for k, v in raw.items()}
    return {"raw": raw, "work": work, "x": x, "y3": y3}


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 1e-9, None)
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-9)


def _probe_proba(model: CatBoostClassifier, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = [int(c) for c in getattr(model, "classes_", [0, 1, 2])]
    idx = {c: i for i, c in enumerate(classes)}
    return _normalize(np.stack([raw[:, idx[0]], raw[:, idx[1]], raw[:, idx[2]]], axis=1))


def _fit_probe_model(data: dict[str, Any], seed: int) -> CatBoostClassifier:
    x_train = data["x"]["train"]
    x_val = data["x"]["val"]
    y_train = data["y3"]["train"]
    y_val = data["y3"]["val"]
    w_train = _router3_weight(data["work"]["train"], y_train)
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=350,
        depth=6,
        learning_rate=0.04,
        l2_leaf_reg=6.0,
        random_strength=1.0,
        bagging_temperature=0.0,
        random_seed=seed,
        task_type="CPU",
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        eval_set=(x_val, y_val),
        early_stopping_rounds=60,
        verbose=False,
    )
    return model


def _metric_bundle(y: np.ndarray, p: np.ndarray, work: pd.DataFrame) -> dict[str, Any]:
    pred = np.asarray(p, dtype=np.float64).argmax(axis=1).astype(np.int64)
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "pred_counts": {str(int(k)): int(v) for k, v in pd.Series(pred).value_counts().sort_index().to_dict().items()},
        "class_counts": {str(int(k)): int(v) for k, v in pd.Series(y).value_counts().sort_index().to_dict().items()},
    }
    out.update(_profit_proxy(work, pred))
    return out


def _split_overview(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "val", "oos"):
        raw = data["raw"][split]
        work = data["work"][split]
        y = data["y3"][split]
        out[split] = {
            "raw_rows": int(len(raw)),
            "kept_rows": int(len(work)),
            "timestamp_min": str(pd.to_datetime(raw["timestamp"]).min()) if "timestamp" in raw.columns else None,
            "timestamp_max": str(pd.to_datetime(raw["timestamp"]).max()) if "timestamp" in raw.columns else None,
            "class_counts": {str(int(k)): int(v) for k, v in pd.Series(y).value_counts().sort_index().to_dict().items()},
        }
    return out


def _missing_health(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    train_x = data["x"]["train"]
    for col in ROUTER_FEATURE_COLS:
        train_ser = train_x[col]
        row: dict[str, Any] = {
            "feature": col,
            "train_missing_ratio": float(train_ser.isna().mean()),
            "train_nunique": int(train_ser.nunique(dropna=True)),
            "train_std": float(train_ser.std(skipna=True)) if train_ser.notna().any() else 0.0,
            "train_abs_mean": float(train_ser.abs().mean(skipna=True)) if train_ser.notna().any() else 0.0,
        }
        for split in ("val", "oos"):
            ser = data["x"][split][col]
            row[f"{split}_missing_ratio"] = float(ser.isna().mean())
            row[f"{split}_nunique"] = int(ser.nunique(dropna=True))
            row[f"{split}_std"] = float(ser.std(skipna=True)) if ser.notna().any() else 0.0
        row["near_constant_flag"] = bool((row["train_nunique"] <= 2) or (abs(row["train_std"]) < 1e-8))
        rows.append(row)
    return pd.DataFrame(rows)


def _top_corr_pairs(train_x: pd.DataFrame, topn: int) -> pd.DataFrame:
    corr = train_x.corr(numeric_only=True).abs()
    rows: list[dict[str, Any]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rows.append({"feature_a": cols[i], "feature_b": cols[j], "abs_corr": float(corr.iat[i, j])})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).head(topn).reset_index(drop=True)


def _quantile_edges(train: np.ndarray, bins: int) -> np.ndarray:
    base = train[np.isfinite(train)]
    if base.size == 0:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(base, qs))
    if edges.size < 2:
        val = float(edges[0]) if edges.size == 1 else 0.0
        return np.array([val - 1.0, val + 1.0], dtype=np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.astype(np.float64)


def _psi(train: np.ndarray, other: np.ndarray, bins: int) -> float:
    edges = _quantile_edges(train, bins)
    train_hist, _ = np.histogram(train[np.isfinite(train)], bins=edges)
    other_hist, _ = np.histogram(other[np.isfinite(other)], bins=edges)
    train_pct = np.clip(train_hist / max(train_hist.sum(), 1), 1e-9, None)
    other_pct = np.clip(other_hist / max(other_hist.sum(), 1), 1e-9, None)
    return float(np.sum((other_pct - train_pct) * np.log(other_pct / train_pct)))


def _ks(train: np.ndarray, other: np.ndarray) -> float:
    a = np.sort(train[np.isfinite(train)])
    b = np.sort(other[np.isfinite(other)])
    if a.size == 0 or b.size == 0:
        return 0.0
    grid = np.unique(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side="right") / float(a.size)
    cdf_b = np.searchsorted(b, grid, side="right") / float(b.size)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _drift_table(data: dict[str, Any], bins: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in ROUTER_FEATURE_COLS:
        train = data["x"]["train"][col].to_numpy(np.float64)
        row: dict[str, Any] = {"feature": col}
        for split in ("val", "oos"):
            other = data["x"][split][col].to_numpy(np.float64)
            row[f"{split}_psi"] = _psi(train, other, bins)
            row[f"{split}_ks"] = _ks(train, other)
            row[f"{split}_mean_shift"] = float(np.nanmean(other) - np.nanmean(train))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["max_psi"] = out[[c for c in out.columns if c.endswith("_psi")]].max(axis=1)
    out["max_ks"] = out[[c for c in out.columns if c.endswith("_ks")]].max(axis=1)
    return out.sort_values(["max_psi", "max_ks"], ascending=False).reset_index(drop=True)


def _mi_table(data: dict[str, Any], seed: int) -> pd.DataFrame:
    train_x = data["x"]["train"].copy()
    med = train_x.median(numeric_only=True)
    train_x = train_x.fillna(med).fillna(0.0)
    y = data["y3"]["train"]
    mi = mutual_info_classif(train_x.to_numpy(np.float64), y.astype(np.int64), discrete_features=False, random_state=seed)
    corr_rows: list[float] = []
    y_long = (y == 1).astype(np.float64)
    y_short = (y == 2).astype(np.float64)
    for col in ROUTER_FEATURE_COLS:
        arr = train_x[col].to_numpy(np.float64)
        if np.nanstd(arr) < 1e-12:
            corr_rows.append(0.0)
            continue
        c1 = np.corrcoef(arr, y_long)[0, 1] if np.nanstd(y_long) > 0 else 0.0
        c2 = np.corrcoef(arr, y_short)[0, 1] if np.nanstd(y_short) > 0 else 0.0
        vals = [0.0 if not np.isfinite(v) else float(v) for v in (c1, c2)]
        corr_rows.append(max(abs(vals[0]), abs(vals[1])))
    out = pd.DataFrame(
        {
            "feature": ROUTER_FEATURE_COLS,
            "mutual_info": np.asarray(mi, dtype=np.float64),
            "abs_pointbiserial_max": np.asarray(corr_rows, dtype=np.float64),
        }
    )
    out["mi_rank_pct"] = out["mutual_info"].rank(pct=True, ascending=True)
    out["corr_rank_pct"] = out["abs_pointbiserial_max"].rank(pct=True, ascending=True)
    out["combined_rank"] = out["mi_rank_pct"] + out["corr_rank_pct"]
    return out.sort_values(["combined_rank", "mutual_info", "abs_pointbiserial_max"], ascending=False).reset_index(drop=True)


def _probe_model_importance(model: CatBoostClassifier, cols: list[str]) -> pd.DataFrame:
    imp = np.asarray(model.get_feature_importance(), dtype=np.float64)
    feat_names = list(getattr(model, "feature_names_", cols))
    n = min(len(feat_names), len(imp))
    out = pd.DataFrame({"feature": feat_names[:n], "probe_importance": imp[:n]})
    return out.sort_values("probe_importance", ascending=False).reset_index(drop=True)


def _sample_indices(n: int, max_rows: int, seed: int) -> np.ndarray:
    if n <= max_rows:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_rows, replace=False))
    return idx.astype(np.int64)


def _permute_rows(
    model: CatBoostClassifier,
    x: pd.DataFrame,
    y: np.ndarray,
    work: pd.DataFrame,
    *,
    feature_groups: dict[str, list[str]],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    base_p = _probe_proba(model, x)
    base = _metric_bundle(y, base_p, work)
    feat_rows: list[dict[str, Any]] = []
    for col in ROUTER_FEATURE_COLS:
        xp = x.copy()
        xp[col] = xp[col].to_numpy()[rng.permutation(len(xp))]
        p = _probe_proba(model, xp)
        cur = _metric_bundle(y, p, work)
        feat_rows.append(
            {
                "feature": col,
                "delta_balanced_accuracy": float(base["balanced_accuracy"] - cur["balanced_accuracy"]),
                "delta_macro_f1": float(base["macro_f1"] - cur["macro_f1"]),
                "delta_pred_trade_quality_sum": float(base["pred_trade_quality_sum"] - cur["pred_trade_quality_sum"]),
                "delta_pred_trade_count": int(base["pred_trade_count"] - cur["pred_trade_count"]),
            }
        )
    group_rows: list[dict[str, Any]] = []
    for name, cols in feature_groups.items():
        use = [c for c in cols if c in x.columns]
        if not use:
            continue
        xp = x.copy()
        perm = rng.permutation(len(xp))
        for col in use:
            xp[col] = xp[col].to_numpy()[perm]
        p = _probe_proba(model, xp)
        cur = _metric_bundle(y, p, work)
        group_rows.append(
            {
                "group": name,
                "feature_count": int(len(use)),
                "delta_balanced_accuracy": float(base["balanced_accuracy"] - cur["balanced_accuracy"]),
                "delta_macro_f1": float(base["macro_f1"] - cur["macro_f1"]),
                "delta_pred_trade_quality_sum": float(base["pred_trade_quality_sum"] - cur["pred_trade_quality_sum"]),
                "delta_pred_trade_count": int(base["pred_trade_count"] - cur["pred_trade_count"]),
            }
        )
    feat_df = pd.DataFrame(feat_rows).sort_values(["delta_balanced_accuracy", "delta_macro_f1"], ascending=False).reset_index(drop=True)
    group_df = pd.DataFrame(group_rows).sort_values(["delta_balanced_accuracy", "delta_macro_f1"], ascending=False).reset_index(drop=True)
    return feat_df, group_df, base


def _feature_contract_summary(base_meta: Path, configured_cols: list[str]) -> dict[str, Any]:
    if not base_meta.exists():
        return {"base_meta_exists": False, "model_feature_cols": [], "missing_in_model": configured_cols, "extra_in_model": []}
    meta = joblib.load(base_meta)
    model_cols = list(meta.get("feature_cols", []))
    cur = set(configured_cols)
    old = set(model_cols)
    return {
        "base_meta_exists": True,
        "model_feature_cols": model_cols,
        "missing_in_model": sorted(cur - old),
        "extra_in_model": sorted(old - cur),
        "exact_match": bool(model_cols == configured_cols),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Router5 feature health, drift, and predictive importance analysis.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--base-meta", type=Path, default=DEFAULT_BASE_META)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--top-corr", type=int, default=30)
    ap.add_argument("--perm-max-rows", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_data(args.data_dir)
    split_overview = _split_overview(data)
    health = _missing_health(data)
    drift = _drift_table(data, args.bins)
    mi = _mi_table(data, args.seed)
    corr = _top_corr_pairs(data["x"]["train"], args.top_corr)
    contract = _feature_contract_summary(args.base_meta, ROUTER_FEATURE_COLS)
    probe = _fit_probe_model(data, args.seed)
    importance = _probe_model_importance(probe, ROUTER_FEATURE_COLS)

    perm_summary: dict[str, Any] = {}
    for split, bump in (("val", 1000), ("oos", 2000)):
        idx = _sample_indices(len(data["x"][split]), args.perm_max_rows, args.seed + bump)
        x = data["x"][split].iloc[idx].reset_index(drop=True)
        work = data["work"][split].iloc[idx].reset_index(drop=True)
        y = data["y3"][split][idx]
        feat_df, group_df, base = _permute_rows(
            probe,
            x,
            y,
            work,
            feature_groups=FEATURE_GROUPS,
            seed=args.seed + bump,
        )
        feat_path = args.out_dir / f"permutation_feature_importance_{split}.csv"
        group_path = args.out_dir / f"permutation_group_importance_{split}.csv"
        feat_df.to_csv(feat_path, index=False)
        group_df.to_csv(group_path, index=False)
        perm_summary[split] = {
            "rows_used": int(len(idx)),
            "baseline": base,
            "feature_csv": str(feat_path),
            "group_csv": str(group_path),
            "top_feature_impacts": feat_df.head(10).to_dict(orient="records"),
            "top_group_impacts": group_df.head(10).to_dict(orient="records"),
        }

    health_path = args.out_dir / "feature_health.csv"
    drift_path = args.out_dir / "feature_drift.csv"
    mi_path = args.out_dir / "feature_univariate_rank.csv"
    corr_path = args.out_dir / "feature_top_corr_pairs.csv"
    imp_path = args.out_dir / "feature_model_importance.csv"
    summary_path = args.out_dir / "router5_feature_analysis_summary.json"
    health.to_csv(health_path, index=False)
    drift.to_csv(drift_path, index=False)
    mi.to_csv(mi_path, index=False)
    corr.to_csv(corr_path, index=False)
    importance.to_csv(imp_path, index=False)

    summary = {
        "model_id": "analyze_alpha5_router5_features_20260521",
        "data_dir": str(args.data_dir),
        "base_meta": str(args.base_meta),
        "feature_contract": contract,
        "split_overview": split_overview,
        "artifacts": {
            "feature_health_csv": str(health_path),
            "feature_drift_csv": str(drift_path),
            "feature_univariate_rank_csv": str(mi_path),
            "feature_top_corr_pairs_csv": str(corr_path),
            "feature_model_importance_csv": str(imp_path),
        },
        "near_constant_features": health.loc[health["near_constant_flag"], "feature"].tolist(),
        "highest_drift_features": drift.head(12).to_dict(orient="records"),
        "top_univariate_features": mi.head(12).to_dict(orient="records"),
        "top_model_features": importance.head(12).to_dict(orient="records"),
        "top_corr_pairs": corr.head(12).to_dict(orient="records"),
        "probe_metrics": {
            split: _metric_bundle(data["y3"][split], _probe_proba(probe, data["x"][split]), data["work"][split])
            for split in ("val", "oos")
        },
        "permutation": perm_summary,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "near_constant_features": summary["near_constant_features"],
                "top_model_features": [r["feature"] for r in summary["top_model_features"][:8]],
                "top_oos_permutation_features": [r["feature"] for r in perm_summary["oos"]["top_feature_impacts"][:8]],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
