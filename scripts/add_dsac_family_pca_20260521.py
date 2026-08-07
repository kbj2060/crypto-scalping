#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/llewyn/crypto-scalping")
DEFAULT_INPUT_CSV = ROOT / "tmp/causal_regen_20260516/dsac_feature_inventory_regime_fixed_20260521/rl_training_2025_direction_router_feature_inventory_base.csv"
DEFAULT_FAMILY_DIR = ROOT / "tmp/causal_regen_20260516/dsac_feature_inventory_regime_fixed_20260521/families"
DEFAULT_OUT_CSV = ROOT / "tmp/causal_regen_20260516/dsac_feature_inventory_regime_fixed_20260521/rl_training_2025_direction_router_feature_inventory_base_with_family_pca.csv"
DEFAULT_META_JSON = ROOT / "tmp/causal_regen_20260516/dsac_feature_inventory_regime_fixed_20260521/family_pca_meta.json"
FIT_CUTOFF = pd.Timestamp("2025-10-01")
FAMILY_SPECS = {
    "m7": 5,
    "clean_regime4_state24": 5,
    "regime4_pred": 5,
    "ai_family": 5,
    "market_state": 5,
}


def _load_family_cols(family_dir: Path, family: str) -> list[str]:
    path = family_dir / f"{family}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    cols = payload.get("features", [])
    out: list[str] = []
    seen: set[str] = set()
    for col in cols:
        c = str(col).strip()
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def main() -> None:
    df = pd.read_csv(DEFAULT_INPUT_CSV)
    if "timestamp" not in df.columns:
        raise ValueError(f"timestamp column missing in {DEFAULT_INPUT_CSV}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    fit_mask = df["timestamp"] < FIT_CUTOFF
    if int(fit_mask.sum()) < 256:
        raise RuntimeError("not enough fit rows before cutoff for family PCA")

    out = df.copy()
    meta: dict[str, dict[str, object]] = {}
    for family, n_components in FAMILY_SPECS.items():
        cols = [c for c in _load_family_cols(DEFAULT_FAMILY_DIR, family) if c in out.columns]
        if len(cols) < 2:
            continue
        x_fit = out.loc[fit_mask, cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
        x_all = out.loc[:, cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
        scaler = StandardScaler()
        x_fit_z = scaler.fit_transform(x_fit)
        x_all_z = scaler.transform(x_all)
        k = max(1, min(int(n_components), x_fit_z.shape[1], x_fit_z.shape[0]))
        pca = PCA(n_components=k, random_state=20260521, svd_solver="full")
        x_all_p = pca.fit(x_fit_z).transform(x_all_z)
        names: list[str] = []
        for i in range(k):
            col = f"pca_{family}_{i:02d}"
            out[col] = x_all_p[:, i].astype(np.float32)
            names.append(col)
        meta[family] = {
            "input_cols": cols,
            "output_cols": names,
            "components": int(k),
            "explained_variance_ratio": np.asarray(pca.explained_variance_ratio_, dtype=np.float64).tolist(),
            "explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
        }

    DEFAULT_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DEFAULT_OUT_CSV, index=False)
    DEFAULT_META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "input_csv": str(DEFAULT_INPUT_CSV),
                "output_csv": str(DEFAULT_OUT_CSV),
                "meta_json": str(DEFAULT_META_JSON),
                "families": {k: {"components": int(v["components"]), "explained_variance_sum": float(v["explained_variance_sum"])} for k, v in meta.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
