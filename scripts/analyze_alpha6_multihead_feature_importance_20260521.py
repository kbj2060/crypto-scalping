#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import Pool


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "tmp/causal_regen_20260516/alpha6_catboost_multihead_policy_20260521/stable48_global_pca32_bundle.joblib"
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/alpha6_catboost_feature_importance_20260521"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _importance(model: Any, feature_names: list[str]) -> np.ndarray:
    if model is None:
        return np.zeros(len(feature_names), dtype=np.float64)
    try:
        vals = model.get_feature_importance(type="PredictionValuesChange")
    except TypeError:
        vals = model.get_feature_importance()
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size != len(feature_names):
        vals = np.resize(vals, len(feature_names))
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize Alpha6 CatBoost multi-head feature importance.")
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.bundle)
    models = bundle["models"]
    feature_names = list(bundle.get("model_features") or bundle.get("feature_cols") or [])
    heads = [
        ("action", "action_model"),
        ("quality", "quality_model"),
        ("notional", "notional_model"),
        ("take_profit", "take_profit_model"),
        ("stop_loss", "stop_loss_model"),
        ("max_hold", "max_hold_model"),
        ("cooldown", "cooldown_model"),
    ]
    table = pd.DataFrame({"feature": feature_names})
    for head, key in heads:
        table[f"{head}_importance"] = _importance(models.get(key), feature_names)
    imp_cols = [c for c in table.columns if c.endswith("_importance")]
    for col in imp_cols:
        total = float(table[col].sum())
        table[col.replace("_importance", "_share")] = table[col] / total if total > 0 else 0.0
    table["mean_importance"] = table[imp_cols].mean(axis=1)
    table["max_importance"] = table[imp_cols].max(axis=1)
    table = table.sort_values("mean_importance", ascending=False).reset_index(drop=True)

    stem = args.bundle.stem.replace("_bundle", "")
    out_csv = args.out_dir / f"{stem}_feature_importance.csv"
    out_json = args.out_dir / f"{stem}_feature_importance_summary.json"
    table.to_csv(out_csv, index=False)
    summary = {
        "bundle": args.bundle,
        "model_id": bundle.get("model_id"),
        "variant": bundle.get("variant"),
        "use_pca": bundle.get("use_pca"),
        "feature_count": len(feature_names),
        "top20_mean": table.head(20).to_dict(orient="records"),
        "top10_by_head": {
            head: table.sort_values(f"{head}_importance", ascending=False).head(10)[["feature", f"{head}_importance"]].to_dict(orient="records")
            for head, _ in heads
            if f"{head}_importance" in table
        },
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    print(json.dumps({"csv": str(out_csv), "json": str(out_json), "top10": table.head(10)["feature"].tolist()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
