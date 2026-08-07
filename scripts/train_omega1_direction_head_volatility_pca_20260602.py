#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import train_omega1_direction_head_direction_only_20260602 as base
import train_omega1_direction_head_raw_context_groups_20260602 as ctx


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_direction_head_volatility_pca_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_direction_head_volatility_pca_20260602"

BASE_COLS = ctx.BASE_COLS
VOL_COLS = ctx.GROUP_CANDIDATES["volatility_context"]

PCA_COMPONENTS = [4, 6, 8, 12, 16]

BASELINE_CORE_TSFMC = {
    "variant": "core_plus_tsfm_chronos",
    "feature_count": 55,
    "oos_bacc": 0.5974048650,
    "oos_auc": 0.7907205158,
    "oos_proxy_wr": 0.6579421029,
    "oos_proxy_trades": 13334,
}

BASELINE_RAW_VOL = {
    "variant": "add_volatility_context",
    "feature_count": 79,
    "oos_bacc": 0.6040391298,
    "oos_auc": 0.7933178682,
    "oos_proxy_wr": 0.6589017032,
    "oos_proxy_trades": 13093,
}


def _json_default(obj: Any) -> Any:
    return base._json_default(obj)


def _assert_finite(frame: pd.DataFrame, cols: list[str], label: str) -> None:
    arr = frame[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        bad = {c: int((~np.isfinite(frame[c].to_numpy(dtype=np.float64))).sum()) for c in cols}
        bad = {k: v for k, v in bad.items() if v}
        raise ValueError(f"{label} contains non-finite values: {bad}")


class VolPca:
    def __init__(self, n_components: int):
        self.n_components = int(n_components)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components, svd_solver="full", random_state=20260602)
        self.output_cols = [f"pca_volatility_{idx + 1:02d}" for idx in range(self.n_components)]
        self.explained_variance: dict[str, Any] = {}

    def fit(self, frame: pd.DataFrame) -> "VolPca":
        _assert_finite(frame, VOL_COLS, "volatility PCA fit")
        x = frame[VOL_COLS].to_numpy(dtype=np.float64)
        x_scaled = self.scaler.fit_transform(x)
        self.pca.fit(x_scaled)
        ratio = self.pca.explained_variance_ratio_.astype(float)
        self.explained_variance = {
            "source_feature_count": int(len(VOL_COLS)),
            "n_components": int(self.n_components),
            "explained_variance_ratio": ratio.tolist(),
            "explained_variance_sum": float(ratio.sum()),
            "source_cols": VOL_COLS,
            "output_cols": self.output_cols,
        }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        _assert_finite(frame, VOL_COLS, "volatility PCA transform")
        x = frame[VOL_COLS].to_numpy(dtype=np.float64)
        values = self.pca.transform(self.scaler.transform(x))
        return pd.DataFrame(values, columns=self.output_cols, index=frame.index).reset_index(drop=True)


def _features_with_pca(frame: pd.DataFrame, pca: VolPca) -> pd.DataFrame:
    base_part = frame[BASE_COLS].reset_index(drop=True)
    pca_part = pca.transform(frame)
    return pd.concat([base_part, pca_part], axis=1)


def _oof_proba_pca(train: pd.DataFrame, n_components: int, *, seed: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    n = len(train)
    starts = [int(n * 0.35), int(n * 0.50), int(n * 0.65), int(n * 0.80)]
    ends = [int(n * 0.50), int(n * 0.65), int(n * 0.80), n]
    proba = np.full((n, 3), np.nan, dtype=np.float64)
    covered = np.zeros(n, dtype=bool)
    folds: list[dict[str, Any]] = []
    pca_folds: list[dict[str, Any]] = []
    y = train["zigzag_action"].to_numpy(dtype=np.int64)
    for fold, (start, end) in enumerate(zip(starts, ends), start=1):
        transformer = VolPca(n_components).fit(train.iloc[:start])
        x_train = _features_with_pca(train.iloc[:start], transformer)
        x_pred = _features_with_pca(train.iloc[start:end], transformer)
        model = base._fit_catboost(x_train, y[:start], seed=seed + fold, iterations=500)
        pred = base._proba3(model, x_pred)
        proba[start:end] = pred
        covered[start:end] = True
        folds.append(
            {
                "fold": fold,
                "train_rows": int(start),
                "predict_start": int(start),
                "predict_end": int(end),
                "metrics": base._metrics(y[start:end], pred),
            }
        )
        pca_folds.append({"fold": fold, "explained_variance": transformer.explained_variance})
    return proba, covered, folds, pca_folds


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.DROP_EVENTS.clear()
    train, groups_train, missing_train = ctx._build_frame(2025)
    oos, groups_oos, missing_oos = ctx._build_frame(2026)
    if groups_train.get("volatility_context") != groups_oos.get("volatility_context"):
        raise RuntimeError("volatility context contract mismatch between 2025 and 2026")
    if groups_train.get("volatility_context") != VOL_COLS:
        raise RuntimeError(f"volatility columns changed: {groups_train.get('volatility_context')} != {VOL_COLS}")

    y_train = train["zigzag_action"].to_numpy(dtype=np.int64)
    y_oos = oos["zigzag_action"].to_numpy(dtype=np.int64)
    base._validate_features(BASE_COLS, train)
    base._validate_features(BASE_COLS, oos)
    ctx._validate_context_cols(VOL_COLS, train)
    ctx._validate_context_cols(VOL_COLS, oos)
    _assert_finite(train, BASE_COLS + VOL_COLS, "train")
    _assert_finite(oos, BASE_COLS + VOL_COLS, "oos")

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "label_source": "zigzag_action",
        "base_direction_head": "core_plus_tsfm_chronos",
        "pca_policy": "BASE_COLS remain raw; only volatility_context is compressed. OOF PCA fit inside each fold only; final PCA fit on 2025 only.",
        "base_cols": BASE_COLS,
        "volatility_cols": VOL_COLS,
        "baselines": {
            "core_plus_tsfm_chronos": BASELINE_CORE_TSFMC,
            "add_volatility_context_raw": BASELINE_RAW_VOL,
        },
        "missing_candidates_2025": {"volatility_context": missing_train.get("volatility_context", [])},
        "missing_candidates_2026": {"volatility_context": missing_oos.get("volatility_context", [])},
        "variants": {},
        "drop_events": base.DROP_EVENTS,
        "artifacts": {"out_dir": str(OUT_DIR)},
    }
    ranking: list[dict[str, Any]] = []
    for idx, n_components in enumerate(PCA_COMPONENTS, start=1):
        variant = f"volatility_pca{n_components:02d}"
        variant_dir = OUT_DIR / variant
        variant_dir.mkdir(parents=True, exist_ok=True)

        oof, covered, folds, pca_folds = _oof_proba_pca(train, n_components, seed=20260602 + idx * 100)
        oof_metrics = base._metrics(y_train[covered], oof[covered])

        final_transformer = VolPca(n_components).fit(train)
        x_train = _features_with_pca(train, final_transformer)
        x_oos = _features_with_pca(oos, final_transformer)
        final_model = base._fit_catboost(x_train, y_train, seed=20260602 + idx, iterations=800)
        oos_proba = base._proba3(final_model, x_oos)
        oos_metrics = base._metrics(y_oos, oos_proba)

        oof_out = base._outputs(train.loc[covered].reset_index(drop=True), oof[covered], prefix="omega1_dir_volpca_oof")
        oos_out = base._outputs(oos, oos_proba, prefix="omega1_dir_volpca")
        oof_path = variant_dir / f"training_features_2025_{variant}_omega1_direction_volpca_oof_20260602.csv"
        oos_path = variant_dir / f"training_features_2026_rebuilt_{variant}_omega1_direction_volpca_20260602.csv"
        oof_out.to_csv(oof_path, index=False)
        oos_out.to_csv(oos_path, index=False)

        model_path = variant_dir / f"{variant}_omega1_direction_volpca.cbm"
        final_model.save_model(str(model_path))
        contract_path = variant_dir / f"{variant}_omega1_direction_volpca_contract.joblib"
        joblib.dump(
            {
                "variant": variant,
                "label_source": "zigzag_action",
                "base_cols": BASE_COLS,
                "volatility_cols": VOL_COLS,
                "pca_feature_cols": final_transformer.output_cols,
                "feature_cols": list(x_train.columns),
                "pca_transformer": final_transformer,
            },
            contract_path,
        )

        delta_core = {
            "oos_bacc": float(oos_metrics["balanced_accuracy"] - BASELINE_CORE_TSFMC["oos_bacc"]),
            "oos_auc": None if oos_metrics["ovr_auc"] is None else float(oos_metrics["ovr_auc"] - BASELINE_CORE_TSFMC["oos_auc"]),
            "oos_proxy_wr": None if oos_metrics["proxy_wr"] is None else float(oos_metrics["proxy_wr"] - BASELINE_CORE_TSFMC["oos_proxy_wr"]),
            "oos_proxy_trades": int(oos_metrics["proxy_trades"] - BASELINE_CORE_TSFMC["oos_proxy_trades"]),
        }
        delta_raw_vol = {
            "oos_bacc": float(oos_metrics["balanced_accuracy"] - BASELINE_RAW_VOL["oos_bacc"]),
            "oos_auc": None if oos_metrics["ovr_auc"] is None else float(oos_metrics["ovr_auc"] - BASELINE_RAW_VOL["oos_auc"]),
            "oos_proxy_wr": None if oos_metrics["proxy_wr"] is None else float(oos_metrics["proxy_wr"] - BASELINE_RAW_VOL["oos_proxy_wr"]),
            "oos_proxy_trades": int(oos_metrics["proxy_trades"] - BASELINE_RAW_VOL["oos_proxy_trades"]),
        }
        payload = {
            "variant": variant,
            "feature_count": int(x_train.shape[1]),
            "base_feature_count": int(len(BASE_COLS)),
            "volatility_raw_feature_count": int(len(VOL_COLS)),
            "volatility_pca_components": int(n_components),
            "feature_cols": list(x_train.columns),
            "oof_metrics": oof_metrics,
            "oos_metrics": oos_metrics,
            "delta_vs_core_plus_tsfm_chronos": delta_core,
            "delta_vs_raw_add_volatility_context": delta_raw_vol,
            "folds": folds,
            "pca_folds": pca_folds,
            "final_pca_explained_variance": final_transformer.explained_variance,
            "artifacts": {
                "oof_2025": str(oof_path),
                "oos_2026": str(oos_path),
                "model": str(model_path),
                "contract": str(contract_path),
            },
        }
        report["variants"][variant] = payload
        ranking.append(
            {
                "variant": variant,
                "feature_count": int(x_train.shape[1]),
                "volatility_pca_components": int(n_components),
                "volatility_explained_variance_sum": final_transformer.explained_variance["explained_variance_sum"],
                "oof_bacc": oof_metrics["balanced_accuracy"],
                "oof_auc": oof_metrics["ovr_auc"],
                "oof_proxy_wr": oof_metrics["proxy_wr"],
                "oos_bacc": oos_metrics["balanced_accuracy"],
                "oos_auc": oos_metrics["ovr_auc"],
                "oos_proxy_wr": oos_metrics["proxy_wr"],
                "oos_proxy_trades": oos_metrics["proxy_trades"],
                "delta_oos_bacc_vs_core": delta_core["oos_bacc"],
                "delta_oos_auc_vs_core": delta_core["oos_auc"],
                "delta_oos_proxy_wr_vs_core": delta_core["oos_proxy_wr"],
                "delta_oos_bacc_vs_raw_vol": delta_raw_vol["oos_bacc"],
                "delta_oos_auc_vs_raw_vol": delta_raw_vol["oos_auc"],
                "delta_oos_proxy_wr_vs_raw_vol": delta_raw_vol["oos_proxy_wr"],
            }
        )

    ranking.sort(key=lambda r: (float(r["oos_bacc"]), float(r["oos_auc"] or 0.0)), reverse=True)
    report["ranking"] = ranking
    report["selected_by_oos_bacc"] = ranking[0]["variant"]
    pd.DataFrame(ranking).to_csv(OUT_DIR / "ranking.csv", index=False)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "ranking": ranking}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
