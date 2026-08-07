#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import train_omega1_direction_head_direction_only_20260602 as base


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_direction_head_direction_pca_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_direction_head_direction_pca_20260602"


@dataclass(frozen=True)
class PcaGroup:
    name: str
    cols: list[str]
    n_components: int


M7_CAT = base.M7_ZIGZAG[:6]
M7_XGB = base.M7_ZIGZAG[6:]
DIR3_RETRIEVAL_PROB = base.DIR3_RETRIEVAL[:6]
DIR3_RETRIEVAL_CONTEXT = base.DIR3_RETRIEVAL[6:]

GROUPS: dict[str, PcaGroup] = {
    "vsnlstm": PcaGroup("vsnlstm", base.DIR3_VSNLSTM, 3),
    "patch": PcaGroup("patch", base.DIR3_PATCH, 3),
    "m7_cat": PcaGroup("m7_cat", M7_CAT, 3),
    "m7_xgb": PcaGroup("m7_xgb", M7_XGB, 3),
    "regime3_current": PcaGroup("regime3_current", base.REGIME3_CURRENT, 3),
    "regime3_cmamba": PcaGroup("regime3_cmamba", base.REGIME3_CMAMBA, 2),
    "duet": PcaGroup("duet", base.DIR3_DUET, 3),
    "cryptomamba": PcaGroup("cryptomamba", base.DIR3_CRYPTOMAMBA, 3),
    "retrieval_prob": PcaGroup("retrieval_prob", DIR3_RETRIEVAL_PROB, 3),
    "retrieval_context": PcaGroup("retrieval_context", DIR3_RETRIEVAL_CONTEXT, 3),
}

VARIANT_GROUPS: dict[str, list[str]] = {
    "core_pca": ["vsnlstm", "patch"],
    "expanded_pca": ["vsnlstm", "patch", "m7_cat", "m7_xgb", "regime3_current", "regime3_cmamba"],
    "all_direction_pca": [
        "vsnlstm",
        "patch",
        "m7_cat",
        "m7_xgb",
        "regime3_current",
        "regime3_cmamba",
        "duet",
        "cryptomamba",
        "retrieval_prob",
        "retrieval_context",
    ],
}

RAW_VARIANT_BASELINE = {
    "core_pca": {
        "raw_variant": "core",
        "raw_feature_count": 12,
        "raw_oos_bacc": 0.5937722229,
        "raw_oos_auc": 0.7835423133,
        "raw_oos_proxy_wr": 0.6443173150,
        "raw_oos_proxy_trades": 13110,
    },
    "expanded_pca": {
        "raw_variant": "expanded",
        "raw_feature_count": 33,
        "raw_oos_bacc": 0.5912757299,
        "raw_oos_auc": 0.7825443871,
        "raw_oos_proxy_wr": 0.6413680540,
        "raw_oos_proxy_trades": 13479,
    },
    "all_direction_pca": {
        "raw_variant": "all_direction",
        "raw_feature_count": 56,
        "raw_oos_bacc": 0.5898455674,
        "raw_oos_auc": 0.7797412395,
        "raw_oos_proxy_wr": 0.6359873570,
        "raw_oos_proxy_trades": 13288,
    },
}


def _feature_cols_for_groups(group_names: list[str]) -> list[str]:
    out: list[str] = []
    for name in group_names:
        out.extend(GROUPS[name].cols)
    return out


def _assert_finite(frame: pd.DataFrame, cols: list[str], label: str) -> None:
    arr = frame[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        bad = {}
        for col in cols:
            values = frame[col].to_numpy(dtype=np.float64)
            n_bad = int((~np.isfinite(values)).sum())
            if n_bad:
                bad[col] = n_bad
        raise ValueError(f"{label} contains non-finite PCA input values: {bad}")


class GroupPcaTransformer:
    def __init__(self, groups: list[PcaGroup]):
        self.groups = groups
        self.steps: dict[str, tuple[StandardScaler, PCA]] = {}
        self.output_cols: list[str] = []
        self.explained_variance: dict[str, dict[str, Any]] = {}

    def fit(self, frame: pd.DataFrame) -> "GroupPcaTransformer":
        self.steps = {}
        self.output_cols = []
        self.explained_variance = {}
        for group in self.groups:
            _assert_finite(frame, group.cols, f"{group.name} fit")
            n_components = min(int(group.n_components), len(group.cols))
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(frame[group.cols].to_numpy(dtype=np.float64))
            pca = PCA(n_components=n_components, svd_solver="full", random_state=20260602)
            pca.fit(x_scaled)
            self.steps[group.name] = (scaler, pca)
            cols = [f"pca_{group.name}_{idx + 1:02d}" for idx in range(n_components)]
            self.output_cols.extend(cols)
            ratio = pca.explained_variance_ratio_.astype(float)
            self.explained_variance[group.name] = {
                "source_feature_count": int(len(group.cols)),
                "n_components": int(n_components),
                "explained_variance_ratio": ratio.tolist(),
                "explained_variance_sum": float(ratio.sum()),
                "source_cols": group.cols,
                "output_cols": cols,
            }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for group in self.groups:
            _assert_finite(frame, group.cols, f"{group.name} transform")
            scaler, pca = self.steps[group.name]
            x_scaled = scaler.transform(frame[group.cols].to_numpy(dtype=np.float64))
            values = pca.transform(x_scaled)
            cols = [f"pca_{group.name}_{idx + 1:02d}" for idx in range(values.shape[1])]
            parts.append(pd.DataFrame(values, columns=cols, index=frame.index))
        return pd.concat(parts, axis=1).reset_index(drop=True)


def _make_transformer(group_names: list[str]) -> GroupPcaTransformer:
    return GroupPcaTransformer([GROUPS[name] for name in group_names])


def _oof_proba_pca(
    train: pd.DataFrame,
    group_names: list[str],
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
        if start <= 0 or end <= start:
            raise RuntimeError(f"invalid OOF fold: {fold} {start}->{end}")
        transformer = _make_transformer(group_names).fit(train.iloc[:start])
        x_train = transformer.transform(train.iloc[:start])
        x_pred = transformer.transform(train.iloc[start:end])
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
    train = base._build_frame(2025)
    oos = base._build_frame(2026)
    y_train = train["zigzag_action"].to_numpy(dtype=np.int64)
    y_oos = oos["zigzag_action"].to_numpy(dtype=np.int64)

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "label_source": "zigzag_action",
        "train_year": 2025,
        "oos_year": 2026,
        "pca_policy": "group-wise StandardScaler+PCA; OOF PCA fit inside each fold only; final PCA fit on 2025 only",
        "teacher_policy": "teacher_* retired and forbidden as inputs",
        "variants": {},
        "drop_events": base.DROP_EVENTS,
        "raw_baseline": RAW_VARIANT_BASELINE,
        "artifacts": {"out_dir": str(OUT_DIR)},
    }
    ranking: list[dict[str, Any]] = []
    for idx, (variant, group_names) in enumerate(VARIANT_GROUPS.items(), start=1):
        raw_cols = _feature_cols_for_groups(group_names)
        base._validate_features(raw_cols, train)
        base._validate_features(raw_cols, oos)
        _assert_finite(train, raw_cols, f"{variant} train raw")
        _assert_finite(oos, raw_cols, f"{variant} oos raw")

        variant_dir = OUT_DIR / variant
        variant_dir.mkdir(parents=True, exist_ok=True)

        oof, covered, folds, pca_folds = _oof_proba_pca(train, group_names, seed=20260602 + idx * 100)
        oof_metrics = base._metrics(y_train[covered], oof[covered])

        final_transformer = _make_transformer(group_names).fit(train)
        x_train = final_transformer.transform(train)
        x_oos = final_transformer.transform(oos)
        final_model = base._fit_catboost(x_train, y_train, seed=20260602 + idx, iterations=800)
        oos_proba = base._proba3(final_model, x_oos)
        oos_metrics = base._metrics(y_oos, oos_proba)

        oof_out = base._outputs(train.loc[covered].reset_index(drop=True), oof[covered], prefix="omega1_dir_pca_oof")
        oos_out = base._outputs(oos, oos_proba, prefix="omega1_dir_pca")
        oof_path = variant_dir / f"training_features_2025_{variant}_omega1_direction_head_pca_oof_20260602.csv"
        oos_path = variant_dir / f"training_features_2026_rebuilt_{variant}_omega1_direction_head_pca_20260602.csv"
        oof_out.to_csv(oof_path, index=False)
        oos_out.to_csv(oos_path, index=False)

        model_path = variant_dir / f"{variant}_omega1_direction_head_pca.cbm"
        final_model.save_model(str(model_path))
        contract_path = variant_dir / f"{variant}_omega1_direction_head_pca_contract.joblib"
        joblib.dump(
            {
                "variant": variant,
                "label_source": "zigzag_action",
                "group_names": group_names,
                "raw_feature_cols": raw_cols,
                "pca_feature_cols": list(x_train.columns),
                "pca_transformer": final_transformer,
            },
            contract_path,
        )

        baseline = RAW_VARIANT_BASELINE[variant]
        payload = {
            "variant": variant,
            "group_names": group_names,
            "raw_feature_count": int(len(raw_cols)),
            "pca_feature_count": int(x_train.shape[1]),
            "raw_feature_cols": raw_cols,
            "pca_feature_cols": list(x_train.columns),
            "oof_coverage": float(covered.mean()),
            "oof_rows": int(covered.sum()),
            "oof_metrics": oof_metrics,
            "oos_metrics": oos_metrics,
            "raw_baseline": baseline,
            "delta_vs_raw": {
                "oos_bacc": float(oos_metrics["balanced_accuracy"] - baseline["raw_oos_bacc"]),
                "oos_auc": None
                if oos_metrics["ovr_auc"] is None
                else float(oos_metrics["ovr_auc"] - baseline["raw_oos_auc"]),
                "oos_proxy_wr": None
                if oos_metrics["proxy_wr"] is None
                else float(oos_metrics["proxy_wr"] - baseline["raw_oos_proxy_wr"]),
                "oos_proxy_trades": int(oos_metrics["proxy_trades"] - baseline["raw_oos_proxy_trades"]),
            },
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
                "raw_feature_count": len(raw_cols),
                "pca_feature_count": int(x_train.shape[1]),
                "oof_bacc": oof_metrics["balanced_accuracy"],
                "oof_auc": oof_metrics["ovr_auc"],
                "oof_proxy_wr": oof_metrics["proxy_wr"],
                "oos_bacc": oos_metrics["balanced_accuracy"],
                "oos_auc": oos_metrics["ovr_auc"],
                "oos_proxy_wr": oos_metrics["proxy_wr"],
                "oos_proxy_trades": oos_metrics["proxy_trades"],
                "delta_oos_bacc_vs_raw": payload["delta_vs_raw"]["oos_bacc"],
                "delta_oos_auc_vs_raw": payload["delta_vs_raw"]["oos_auc"],
                "delta_oos_proxy_wr_vs_raw": payload["delta_vs_raw"]["oos_proxy_wr"],
                "delta_oos_proxy_trades_vs_raw": payload["delta_vs_raw"]["oos_proxy_trades"],
            }
        )

    ranking.sort(key=lambda r: (float(r["oos_bacc"]), float(r["oos_auc"] or 0.0)), reverse=True)
    report["ranking"] = ranking
    report["selected_by_oos_bacc"] = ranking[0]["variant"]
    pd.DataFrame(ranking).to_csv(OUT_DIR / "ranking.csv", index=False)
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=base._json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "ranking": ranking}, ensure_ascii=False, indent=2, default=base._json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
