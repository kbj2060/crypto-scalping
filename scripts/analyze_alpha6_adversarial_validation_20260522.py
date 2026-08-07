#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_CSV = ROOT / "tmp/causal_regen_20260516/dsac_feature_inventory_regime_fixed_20260521/rl_training_2025_direction_router_feature_inventory_base_with_family_pca.csv"
DEFAULT_SPEC_DIR = ROOT / "tmp/causal_regen_20260516/dsac_feature_variant_specs_regime_fixed_20260521"
DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_high_quality_training_data_20260518"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha6_adversarial_validation_20260522"


def _read_spec(spec_dir: Path, variant: str) -> dict[str, Any]:
    path = spec_dir / f"{variant}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = json.loads(path.read_text())
    features = list(spec.get("features") or spec.get("feature_cols") or [])
    if not features:
        raise ValueError(f"empty feature list: {path}")
    spec["features"] = features
    return spec


def _label_frame(label_dir: Path) -> pd.DataFrame:
    frames = []
    wanted = ["timestamp", "dataset_split"]
    for name in ("alpha5_13_hgb_atr_barrier_labels_train.parquet", "alpha5_13_hgb_atr_barrier_labels_val.parquet"):
        path = label_dir / name
        available = set(pq.ParquetFile(path).schema.names)
        frame = pd.read_parquet(path, columns=[c for c in wanted if c in available])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True).dropna(subset=["timestamp"])
    return out.drop_duplicates("timestamp", keep="last")


def _read_feature_frame(feature_csv: Path, features: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    columns = pd.read_csv(feature_csv, nrows=0).columns.tolist()
    available = set(columns)
    present = [c for c in features if c in available]
    missing = [c for c in features if c not in available]
    keep = ["timestamp", *present]
    frame = pd.read_csv(feature_csv, usecols=keep, parse_dates=["timestamp"], low_memory=False)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return frame, present, missing


def _numeric_matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = frame[cols].copy()
    for col in cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan)


def _psi(train_vals: np.ndarray, val_vals: np.ndarray, bins: int = 10) -> float:
    t = train_vals[np.isfinite(train_vals)]
    v = val_vals[np.isfinite(val_vals)]
    if len(t) < 32 or len(v) < 32:
        return float("nan")
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(t, qs))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    t_hist, _ = np.histogram(t, bins=edges)
    v_hist, _ = np.histogram(v, bins=edges)
    t_pct = np.clip(t_hist / max(np.sum(t_hist), 1), 1e-6, None)
    v_pct = np.clip(v_hist / max(np.sum(v_hist), 1), 1e-6, None)
    return float(np.sum((v_pct - t_pct) * np.log(v_pct / t_pct)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run adversarial validation for an Alpha6 feature variant.")
    ap.add_argument("--variant", default="current_tail111")
    ap.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    ap.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--l2-leaf-reg", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-cap", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = _read_spec(args.spec_dir, args.variant)
    feat, present, missing = _read_feature_frame(args.feature_csv, spec["features"])
    frame = feat.merge(_label_frame(args.label_dir), on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    split = frame["dataset_split"].astype(str).str.lower()
    frame = frame[split.isin(["train", "val", "validation"])].copy()
    frame["is_val"] = frame["dataset_split"].astype(str).str.lower().isin(["val", "validation"]).astype(np.int32)

    train_rows = frame[frame["is_val"] == 0].copy()
    val_rows = frame[frame["is_val"] == 1].copy()
    if int(args.sample_cap) > 0:
        train_rows = train_rows.sample(min(len(train_rows), int(args.sample_cap)), random_state=args.seed)
        val_rows = val_rows.sample(min(len(val_rows), int(args.sample_cap)), random_state=args.seed)
        frame = pd.concat([train_rows, val_rows], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    x_df = _numeric_matrix(frame, present)
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(x_df)
    y = frame["is_val"].to_numpy(dtype=np.int32)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, random_state=args.seed, stratify=y)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=int(args.iterations),
        depth=int(args.depth),
        learning_rate=float(args.learning_rate),
        l2_leaf_reg=float(args.l2_leaf_reg),
        random_seed=int(args.seed),
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )
    model.fit(Pool(x_tr, y_tr))
    prob = model.predict_proba(x_te)[:, 1]
    auc = float(roc_auc_score(y_te, prob))

    importances = pd.DataFrame(
        {"feature": present, "importance": model.get_feature_importance(Pool(x_tr, y_tr))}
    ).sort_values("importance", ascending=False)

    drift_rows = []
    for feat_name in importances["feature"].head(40):
        t = pd.to_numeric(train_rows[feat_name], errors="coerce").to_numpy(dtype=np.float64)
        v = pd.to_numeric(val_rows[feat_name], errors="coerce").to_numpy(dtype=np.float64)
        drift_rows.append(
            {
                "feature": feat_name,
                "importance": float(importances.loc[importances["feature"] == feat_name, "importance"].iloc[0]),
                "psi": _psi(t, v),
                "train_mean": float(np.nanmean(t)) if np.isfinite(t).any() else float("nan"),
                "val_mean": float(np.nanmean(v)) if np.isfinite(v).any() else float("nan"),
                "train_std": float(np.nanstd(t)) if np.isfinite(t).any() else float("nan"),
                "val_std": float(np.nanstd(v)) if np.isfinite(v).any() else float("nan"),
            }
        )
    drift = pd.DataFrame(drift_rows).sort_values(["importance", "psi"], ascending=[False, False])

    summary = {
        "variant": args.variant,
        "rows": int(len(frame)),
        "train_rows": int(len(train_rows)),
        "val_rows": int(len(val_rows)),
        "feature_count": int(len(present)),
        "missing_features": missing,
        "adversarial_auc": auc,
        "top_features": drift.head(15).to_dict(orient="records"),
    }

    prefix = args.out_dir / args.variant
    importances.to_csv(f"{prefix}_feature_importance.csv", index=False)
    drift.to_csv(f"{prefix}_drift_top40.csv", index=False)
    Path(f"{prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
