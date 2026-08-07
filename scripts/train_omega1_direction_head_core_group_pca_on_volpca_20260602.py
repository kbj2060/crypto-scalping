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
import train_omega1_direction_head_tsfm_chronos_20260602 as confirmed


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_direction_head_core_group_pca_on_volpca_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_direction_head_core_group_pca_on_volpca_20260602"

CORE_GROUPS: dict[str, list[str]] = {
    "vsnlstm": base.DIR3_VSNLSTM,
    "patch": base.DIR3_PATCH,
    "tsfm_role": confirmed.TSFM_COLS,
    "chronos_h6": confirmed.CHRONOS_H6_COLS,
    "chronos_uncertainty": confirmed.CHRONOS_UNC_COLS,
}
VOLATILITY_COLS = ctx.GROUP_CANDIDATES["volatility_context"]

BASELINE_VOLPCA06 = {
    "variant": "volatility_pca06",
    "feature_count": 61,
    "oos_bacc": 0.6052110159,
    "oos_auc": 0.7916830103,
    "oos_proxy_wr": 0.6626651567,
    "oos_proxy_trades": 13245,
}

VARIANTS: dict[str, dict[str, int]] = {
    "pca_vsnlstm03": {"vsnlstm": 3},
    "pca_patch03": {"patch": 3},
    "pca_tsfm06": {"tsfm_role": 6},
    "pca_tsfm08": {"tsfm_role": 8},
    "pca_chronos_h603": {"chronos_h6": 3},
    "pca_chronos_unc06": {"chronos_uncertainty": 6},
    "pca_direction_core06": {"vsnlstm": 3, "patch": 3},
    "pca_tsfm_chronos14": {"tsfm_role": 8, "chronos_h6": 3, "chronos_uncertainty": 3},
    "pca_all_core_light": {"vsnlstm": 3, "patch": 3, "tsfm_role": 6, "chronos_h6": 3, "chronos_uncertainty": 6},
    "pca_all_core_mid": {"vsnlstm": 4, "patch": 4, "tsfm_role": 8, "chronos_h6": 4, "chronos_uncertainty": 8},
}


def _json_default(obj: Any) -> Any:
    return base._json_default(obj)


def _assert_finite(frame: pd.DataFrame, cols: list[str], label: str) -> None:
    arr = frame[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        bad = {c: int((~np.isfinite(frame[c].to_numpy(dtype=np.float64))).sum()) for c in cols}
        bad = {k: v for k, v in bad.items() if v}
        raise ValueError(f"{label} contains non-finite values: {bad}")


class GroupPca:
    def __init__(self, group_name: str, source_cols: list[str], n_components: int):
        if n_components >= len(source_cols):
            raise ValueError(f"{group_name} PCA components must be smaller than source columns")
        self.group_name = str(group_name)
        self.source_cols = list(source_cols)
        self.n_components = int(n_components)
        self.output_cols = [f"pca_{self.group_name}_{idx + 1:02d}" for idx in range(self.n_components)]
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components, svd_solver="full", random_state=20260602)
        self.explained_variance: dict[str, Any] = {}

    def fit(self, frame: pd.DataFrame) -> "GroupPca":
        _assert_finite(frame, self.source_cols, f"{self.group_name} PCA fit")
        x = frame[self.source_cols].to_numpy(dtype=np.float64)
        self.pca.fit(self.scaler.fit_transform(x))
        ratio = self.pca.explained_variance_ratio_.astype(float)
        self.explained_variance = {
            "group_name": self.group_name,
            "source_feature_count": int(len(self.source_cols)),
            "n_components": int(self.n_components),
            "explained_variance_ratio": ratio.tolist(),
            "explained_variance_sum": float(ratio.sum()),
            "source_cols": self.source_cols,
            "output_cols": self.output_cols,
        }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        _assert_finite(frame, self.source_cols, f"{self.group_name} PCA transform")
        x = frame[self.source_cols].to_numpy(dtype=np.float64)
        values = self.pca.transform(self.scaler.transform(x))
        return pd.DataFrame(values, columns=self.output_cols, index=frame.index).reset_index(drop=True)


def _fit_transforms(frame: pd.DataFrame, pca_spec: dict[str, int]) -> dict[str, GroupPca]:
    transforms: dict[str, GroupPca] = {}
    for group_name, n_components in pca_spec.items():
        transforms[group_name] = GroupPca(group_name, CORE_GROUPS[group_name], n_components).fit(frame)
    transforms["volatility"] = GroupPca("volatility", VOLATILITY_COLS, 6).fit(frame)
    return transforms


def _features(frame: pd.DataFrame, transforms: dict[str, GroupPca], pca_spec: dict[str, int]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for group_name, cols in CORE_GROUPS.items():
        if group_name in pca_spec:
            parts.append(transforms[group_name].transform(frame))
        else:
            parts.append(frame[cols].reset_index(drop=True))
    parts.append(transforms["volatility"].transform(frame))
    return pd.concat(parts, axis=1)


def _oof_proba_pca(
    train: pd.DataFrame,
    pca_spec: dict[str, int],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    n = len(train)
    starts = [int(n * 0.35), int(n * 0.50), int(n * 0.65), int(n * 0.80)]
    ends = [int(n * 0.50), int(n * 0.65), int(n * 0.80), n]
    proba = np.full((n, 3), np.nan, dtype=np.float64)
    covered = np.zeros(n, dtype=bool)
    folds: list[dict[str, Any]] = []
    pca_folds: list[dict[str, Any]] = []
    y = train["zigzag_action"].to_numpy(dtype=np.int64)
    for fold, (start, end) in enumerate(zip(starts, ends), start=1):
        transforms = _fit_transforms(train.iloc[:start], pca_spec)
        x_train = _features(train.iloc[:start], transforms, pca_spec)
        x_pred = _features(train.iloc[start:end], transforms, pca_spec)
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
        pca_folds.append(
            {
                "fold": fold,
                "explained_variance": {name: pca.explained_variance for name, pca in transforms.items()},
            }
        )
    return proba, covered, folds, pca_folds


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train, groups_train, missing_train = ctx._build_frame(2025)
    oos, groups_oos, missing_oos = ctx._build_frame(2026)
    if groups_train != groups_oos:
        raise RuntimeError(f"context group contract mismatch 2025 vs 2026: {groups_train} != {groups_oos}")
    if groups_train.get("volatility_context") != VOLATILITY_COLS:
        raise RuntimeError("volatility context contract changed")
    return train, oos, {
        "available_groups": groups_train,
        "missing_candidates_2025": missing_train,
        "missing_candidates_2026": missing_oos,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.DROP_EVENTS.clear()
    train, oos, group_report = _load_frames()
    y_train = train["zigzag_action"].to_numpy(dtype=np.int64)
    y_oos = oos["zigzag_action"].to_numpy(dtype=np.int64)

    all_required = [col for cols in CORE_GROUPS.values() for col in cols] + VOLATILITY_COLS
    base._validate_features([col for cols in CORE_GROUPS.values() for col in cols], train)
    base._validate_features([col for cols in CORE_GROUPS.values() for col in cols], oos)
    ctx._validate_context_cols(VOLATILITY_COLS, train)
    ctx._validate_context_cols(VOLATILITY_COLS, oos)
    _assert_finite(train, all_required, "train")
    _assert_finite(oos, all_required, "oos")

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "label_source": "zigzag_action",
        "baseline": BASELINE_VOLPCA06,
        "core_groups": CORE_GROUPS,
        "volatility_cols": VOLATILITY_COLS,
        "pca_policy": "volatility_context is always PCA06; selected core_plus_tsfm_chronos subgroups are replaced by split-local PCA components.",
        "variants": {},
        "drop_events": base.DROP_EVENTS,
        "artifacts": {"out_dir": str(OUT_DIR)},
        **group_report,
    }
    ranking: list[dict[str, Any]] = []
    for idx, (variant, pca_spec) in enumerate(VARIANTS.items(), start=1):
        variant_dir = OUT_DIR / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        oof, covered, folds, pca_folds = _oof_proba_pca(train, pca_spec, seed=20260602 + idx * 100)
        oof_metrics = base._metrics(y_train[covered], oof[covered])

        final_transforms = _fit_transforms(train, pca_spec)
        x_train = _features(train, final_transforms, pca_spec)
        x_oos = _features(oos, final_transforms, pca_spec)
        final_model = base._fit_catboost(x_train, y_train, seed=20260602 + idx, iterations=800)
        oos_proba = base._proba3(final_model, x_oos)
        oos_metrics = base._metrics(y_oos, oos_proba)

        oof_out = base._outputs(train.loc[covered].reset_index(drop=True), oof[covered], prefix="omega1_dir_corepca_oof")
        oos_out = base._outputs(oos, oos_proba, prefix="omega1_dir_corepca")
        oof_path = variant_dir / f"training_features_2025_{variant}_omega1_direction_corepca_oof_20260602.csv"
        oos_path = variant_dir / f"training_features_2026_rebuilt_{variant}_omega1_direction_corepca_20260602.csv"
        oof_out.to_csv(oof_path, index=False)
        oos_out.to_csv(oos_path, index=False)
        model_path = variant_dir / f"{variant}_omega1_direction_corepca.cbm"
        final_model.save_model(str(model_path))
        contract_path = variant_dir / f"{variant}_omega1_direction_corepca_contract.joblib"
        joblib.dump(
            {
                "variant": variant,
                "label_source": "zigzag_action",
                "core_groups": CORE_GROUPS,
                "volatility_cols": VOLATILITY_COLS,
                "pca_spec": pca_spec,
                "feature_cols": list(x_train.columns),
                "transforms": final_transforms,
            },
            contract_path,
        )

        ev = {name: pca.explained_variance for name, pca in final_transforms.items()}
        delta = {
            "oos_bacc": float(oos_metrics["balanced_accuracy"] - BASELINE_VOLPCA06["oos_bacc"]),
            "oos_auc": None if oos_metrics["ovr_auc"] is None else float(oos_metrics["ovr_auc"] - BASELINE_VOLPCA06["oos_auc"]),
            "oos_proxy_wr": None if oos_metrics["proxy_wr"] is None else float(oos_metrics["proxy_wr"] - BASELINE_VOLPCA06["oos_proxy_wr"]),
            "oos_proxy_trades": int(oos_metrics["proxy_trades"] - BASELINE_VOLPCA06["oos_proxy_trades"]),
        }
        payload = {
            "variant": variant,
            "pca_spec": pca_spec,
            "feature_count": int(x_train.shape[1]),
            "feature_cols": list(x_train.columns),
            "oof_metrics": oof_metrics,
            "oos_metrics": oos_metrics,
            "delta_vs_volatility_pca06": delta,
            "folds": folds,
            "pca_folds": pca_folds,
            "final_pca_explained_variance": ev,
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
                "pca_spec": json.dumps(pca_spec, sort_keys=True),
                "oof_bacc": oof_metrics["balanced_accuracy"],
                "oof_auc": oof_metrics["ovr_auc"],
                "oof_proxy_wr": oof_metrics["proxy_wr"],
                "oos_bacc": oos_metrics["balanced_accuracy"],
                "oos_auc": oos_metrics["ovr_auc"],
                "oos_proxy_wr": oos_metrics["proxy_wr"],
                "oos_proxy_trades": oos_metrics["proxy_trades"],
                "delta_oos_bacc_vs_volpca06": delta["oos_bacc"],
                "delta_oos_auc_vs_volpca06": delta["oos_auc"],
                "delta_oos_proxy_wr_vs_volpca06": delta["oos_proxy_wr"],
                "delta_oos_trades_vs_volpca06": delta["oos_proxy_trades"],
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
