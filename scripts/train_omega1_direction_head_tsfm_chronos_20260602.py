#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import train_omega1_direction_head_direction_only_20260602 as base


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_direction_head_tsfm_chronos_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_direction_head_tsfm_chronos_20260602"
TSFM_DIR = ROOT / "tmp/causal_regen_20260516/ai_role_specific_eval_20260530"
CHRONOS_DIR = ROOT / "tmp/causal_regen_20260516/ai_chronos_h6_direction_20260530"
CHRONOS_UNC_DIR = ROOT / "tmp/causal_regen_20260516/chronos_uncertainty_large_move_20260530"

TSFM_COLS = [
    "ai_dir_edge",
    "ai_dir_p_up",
    "ai_dir_p_down",
    "ai_dir_p_flat",
    "ai_dir_entropy",
    "patchtst_median",
    "patchtst_regime_sim",
    "ai_adverse_risk",
    "ai_reward_risk",
    "ai_vol_regime_pct",
    "tide_vol_raw",
    "tide_vol_zscore",
    "ai_flow_pressure",
    "ai_flow_exhaustion",
    "ai_flow_flip_prob",
    "ai_flow_slope",
    "dlinear_smf_ema",
    "dlinear_smf_slope",
    "ai_anchor_revert_prob",
    "ai_anchor_overheat",
    "ai_anchor_trend_escape_prob",
    "timesnet_cycle_sin",
    "timesnet_cycle_cos",
    "timesnet_cycle_delta",
]

CHRONOS_H6_COLS = [
    "chronos_h6_q10",
    "chronos_h6_q50",
    "chronos_h6_q90",
    "chronos_h6_width",
    "chronos_h6_mean",
]

CHRONOS_UNC_SERIES = {
    "atr14": "atr14_pct",
    "rv24": "realized_vol_24",
}
CHRONOS_UNC_COLS = [
    "chronos_unc_atr14_q10",
    "chronos_unc_atr14_q50",
    "chronos_unc_atr14_q90",
    "chronos_unc_atr14_width",
    "chronos_unc_atr14_mean",
    "chronos_unc_atr14_width_ewm3",
    "chronos_unc_atr14_width_ewm6",
    "chronos_unc_rv24_q10",
    "chronos_unc_rv24_q50",
    "chronos_unc_rv24_q90",
    "chronos_unc_rv24_width",
    "chronos_unc_rv24_mean",
    "chronos_unc_rv24_width_ewm3",
    "chronos_unc_rv24_width_ewm6",
]

VARIANTS = {
    "tsfm_role": TSFM_COLS,
    "chronos_h6": CHRONOS_H6_COLS,
    "chronos_uncertainty": CHRONOS_UNC_COLS,
    "chronos_all": CHRONOS_H6_COLS + CHRONOS_UNC_COLS,
    "tsfm_chronos": TSFM_COLS + CHRONOS_H6_COLS + CHRONOS_UNC_COLS,
    "core_plus_tsfm": base.DIR3_VSNLSTM + base.DIR3_PATCH + TSFM_COLS,
    "core_plus_chronos": base.DIR3_VSNLSTM + base.DIR3_PATCH + CHRONOS_H6_COLS + CHRONOS_UNC_COLS,
    "core_plus_tsfm_chronos": base.DIR3_VSNLSTM + base.DIR3_PATCH + TSFM_COLS + CHRONOS_H6_COLS + CHRONOS_UNC_COLS,
}

BASELINES = {
    "all_sanitized_catboost_141": {
        "feature_count": 141,
        "oos_bacc": 0.5654735627,
        "oos_auc": 0.7557143162,
        "oos_proxy_wr": 0.6138582427,
        "oos_proxy_trades": 12599,
    },
    "direction_core_raw_12": {
        "feature_count": 12,
        "oos_bacc": 0.5937722229,
        "oos_auc": 0.7835423133,
        "oos_proxy_wr": 0.6443173150,
        "oos_proxy_trades": 13110,
    },
    "direction_core_pca_6": {
        "feature_count": 6,
        "oos_bacc": 0.5961415116,
        "oos_auc": 0.7835561376,
        "oos_proxy_wr": 0.6478001833,
        "oos_proxy_trades": 13092,
    },
}


def _read_feature(path: Path) -> pd.DataFrame:
    return base._read_csv(path)


def _tsfm_path(year: int) -> Path:
    return TSFM_DIR / f"tsfm_role_features_{year}_exact.csv"


def _chronos_path(year: int) -> Path:
    return CHRONOS_DIR / f"chronos_h6_{year}.csv"


def _chronos_unc_path(series: str, year: int) -> Path:
    split = "val2025" if int(year) == 2025 else "oos2026"
    return CHRONOS_UNC_DIR / f"{series}_{split}_chronos.csv"


def _rename_chronos_unc(raw: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = raw[["timestamp", "q10", "q50", "q90", "width", "mean"]].copy()
    out = out.rename(
        columns={
            "q10": f"chronos_unc_{prefix}_q10",
            "q50": f"chronos_unc_{prefix}_q50",
            "q90": f"chronos_unc_{prefix}_q90",
            "width": f"chronos_unc_{prefix}_width",
            "mean": f"chronos_unc_{prefix}_mean",
        }
    )
    width_col = f"chronos_unc_{prefix}_width"
    out[f"{width_col}_ewm3"] = out[width_col].ewm(span=3, adjust=False).mean()
    out[f"{width_col}_ewm6"] = out[width_col].ewm(span=6, adjust=False).mean()
    return out


def _assert_finite(frame: pd.DataFrame, cols: list[str], label: str) -> None:
    arr = frame[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        bad = {c: int((~np.isfinite(frame[c].to_numpy(dtype=np.float64))).sum()) for c in cols}
        bad = {k: v for k, v in bad.items() if v}
        raise ValueError(f"{label} contains non-finite values: {bad}")


def _build_frame(year: int, *, include_core: bool) -> pd.DataFrame:
    frame = base._add_labels(year)
    if include_core:
        frame = base._exact_join(
            frame,
            _read_feature(base._feature_path("vsnlstm", year)),
            base.DIR3_VSNLSTM,
            f"vsnlstm {year}",
            allow_head_drop=True,
        )
        frame = base._exact_join(
            frame,
            _read_feature(base._feature_path("patch", year)),
            base.DIR3_PATCH,
            f"patch {year}",
            allow_head_drop=True,
        )
    frame = base._exact_join(frame, _read_feature(_tsfm_path(year)), TSFM_COLS, f"tsfm {year}")
    frame = base._exact_join(frame, _read_feature(_chronos_path(year)), CHRONOS_H6_COLS, f"chronos_h6 {year}")
    for prefix, series in CHRONOS_UNC_SERIES.items():
        unc = _rename_chronos_unc(_read_feature(_chronos_unc_path(series, year)), prefix)
        cols = [c for c in unc.columns if c != "timestamp"]
        frame = base._exact_join(frame, unc, cols, f"chronos_unc_{prefix} {year}")
    return frame


def _oof_proba(train: pd.DataFrame, feature_cols: list[str], *, seed: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    n = len(train)
    starts = [int(n * 0.35), int(n * 0.50), int(n * 0.65), int(n * 0.80)]
    ends = [int(n * 0.50), int(n * 0.65), int(n * 0.80), n]
    proba = np.full((n, 3), np.nan, dtype=np.float64)
    covered = np.zeros(n, dtype=bool)
    folds: list[dict[str, Any]] = []
    y = train["zigzag_action"].to_numpy(dtype=np.int64)
    for fold, (start, end) in enumerate(zip(starts, ends), start=1):
        model = base._fit_catboost(train.iloc[:start][feature_cols], y[:start], seed=seed + fold, iterations=500)
        pred = base._proba3(model, train.iloc[start:end][feature_cols])
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
    return proba, covered, folds


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.DROP_EVENTS.clear()
    train_plain = _build_frame(2025, include_core=False)
    oos_plain = _build_frame(2026, include_core=False)
    train_core = _build_frame(2025, include_core=True)
    oos_core = _build_frame(2026, include_core=True)

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "label_source": "zigzag_action",
        "train_year": 2025,
        "oos_year": 2026,
        "contract": "TSFM/Chronos research comparison. teacher_* retired and not used. No Regime4 inputs.",
        "baselines": BASELINES,
        "variants": {},
        "drop_events": base.DROP_EVENTS,
        "artifacts": {"out_dir": str(OUT_DIR)},
    }
    ranking: list[dict[str, Any]] = []
    for idx, (variant, feature_cols) in enumerate(VARIANTS.items(), start=1):
        use_core_frame = variant.startswith("core_plus")
        train = train_core if use_core_frame else train_plain
        oos = oos_core if use_core_frame else oos_plain
        y_train = train["zigzag_action"].to_numpy(dtype=np.int64)
        y_oos = oos["zigzag_action"].to_numpy(dtype=np.int64)
        base._validate_features(feature_cols, train)
        base._validate_features(feature_cols, oos)
        _assert_finite(train, feature_cols, f"{variant} train")
        _assert_finite(oos, feature_cols, f"{variant} oos")

        variant_dir = OUT_DIR / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        oof, covered, folds = _oof_proba(train, feature_cols, seed=20260602 + idx * 100)
        oof_metrics = base._metrics(y_train[covered], oof[covered])
        final_model = base._fit_catboost(train[feature_cols], y_train, seed=20260602 + idx, iterations=800)
        oos_proba = base._proba3(final_model, oos[feature_cols])
        oos_metrics = base._metrics(y_oos, oos_proba)

        oof_out = base._outputs(train.loc[covered].reset_index(drop=True), oof[covered], prefix="omega1_tsfm_chronos_oof")
        oos_out = base._outputs(oos, oos_proba, prefix="omega1_tsfm_chronos")
        oof_path = variant_dir / f"training_features_2025_{variant}_omega1_tsfm_chronos_oof_20260602.csv"
        oos_path = variant_dir / f"training_features_2026_rebuilt_{variant}_omega1_tsfm_chronos_20260602.csv"
        oof_out.to_csv(oof_path, index=False)
        oos_out.to_csv(oos_path, index=False)
        model_path = variant_dir / f"{variant}_omega1_tsfm_chronos.cbm"
        final_model.save_model(str(model_path))
        contract_path = variant_dir / f"{variant}_omega1_tsfm_chronos_contract.joblib"
        joblib.dump(
            {
                "variant": variant,
                "label_source": "zigzag_action",
                "feature_cols": feature_cols,
                "include_core": use_core_frame,
            },
            contract_path,
        )

        delta = {
            "vs_core_pca_oos_bacc": float(oos_metrics["balanced_accuracy"] - BASELINES["direction_core_pca_6"]["oos_bacc"]),
            "vs_core_pca_oos_auc": None
            if oos_metrics["ovr_auc"] is None
            else float(oos_metrics["ovr_auc"] - BASELINES["direction_core_pca_6"]["oos_auc"]),
            "vs_core_pca_oos_proxy_wr": None
            if oos_metrics["proxy_wr"] is None
            else float(oos_metrics["proxy_wr"] - BASELINES["direction_core_pca_6"]["oos_proxy_wr"]),
        }
        payload = {
            "variant": variant,
            "feature_count": int(len(feature_cols)),
            "feature_cols": feature_cols,
            "rows": {"train": int(len(train)), "oos": int(len(oos)), "oof": int(covered.sum())},
            "oof_metrics": oof_metrics,
            "oos_metrics": oos_metrics,
            "delta": delta,
            "folds": folds,
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
                "feature_count": int(len(feature_cols)),
                "oof_bacc": oof_metrics["balanced_accuracy"],
                "oof_auc": oof_metrics["ovr_auc"],
                "oof_proxy_wr": oof_metrics["proxy_wr"],
                "oos_bacc": oos_metrics["balanced_accuracy"],
                "oos_auc": oos_metrics["ovr_auc"],
                "oos_proxy_wr": oos_metrics["proxy_wr"],
                "oos_proxy_trades": oos_metrics["proxy_trades"],
                "delta_oos_bacc_vs_core_pca": delta["vs_core_pca_oos_bacc"],
                "delta_oos_auc_vs_core_pca": delta["vs_core_pca_oos_auc"],
                "delta_oos_proxy_wr_vs_core_pca": delta["vs_core_pca_oos_proxy_wr"],
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
