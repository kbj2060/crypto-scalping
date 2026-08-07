#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_hgb_teacher_features_20260531"
DEFAULT_CANDIDATE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529"
DEFAULT_REGIME3_DIR = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530"
DEFAULT_REGIME3_CURRENT_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
DEFAULT_REGIME3_CMAMBA_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531"
DEFAULT_CHRONOS_DIR = ROOT / "tmp/causal_regen_20260516/chronos_uncertainty_large_move_20260530"
DEFAULT_M7_ZIGZAG_DIR = ROOT / "data/splits/year_oos"
DEFAULT_ZIGZAG_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_01965_hgb_teacher_v1_candidates_20260531"

OMEGA1_CORE_INPUTS = [
    "cvp_regime",
    "regime_trending",
    "ai_adverse_risk",
    "ai_reward_risk",
    "ai_vol_regime_pct",
    "tide_vol_zscore",
    "chronos_atr14_upside_band_ewm3",
    "chronos_atr14_width_ewm6",
    "chronos_atr14_width",
    "chronos_atr14_large_move_score",
    "chronos_realized_vol24_width",
    "chronos_realized_vol24_large_move_score",
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
    "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_entropy",
    "regime3_current_sensitive_wide24_margin",
    "regime3_stability_h6_score",
    "regime3_transition_h6_risk_prob",
    "regime3_transition_h6_risk_pred",
    "regime3_churn_h6_risk_score",
    "regime3_cmamba_h6_future_bull_prob",
    "regime3_cmamba_h6_future_bear_prob",
    "regime3_cmamba_h6_future_chop_prob",
    "regime3_cmamba_h6_confidence",
    "regime3_cmamba_h6_transition_prob",
    "regime3_cmamba_h6_stability_score",
    "m7_q10",
    "m7_q90",
    "m7_qwidth",
    "m7_zigzag_cat_fl",
    "m7_zigzag_cat_up",
    "m7_zigzag_cat_dn",
    "m7_zigzag_cat_confidence",
    "m7_zigzag_cat_side_edge",
    "m7_zigzag_cat_trade_prob",
    "m7_zigzag_xgb_fl",
    "m7_zigzag_xgb_up",
    "m7_zigzag_xgb_dn",
    "m7_zigzag_xgb_confidence",
    "m7_zigzag_xgb_side_edge",
    "m7_zigzag_xgb_trade_prob",
]
FORBIDDEN_INPUT_PREFIXES = ("teacher_", "a5dir_", "clean_regime_", "clean_regime4_", "regime4_pred_")
FORBIDDEN_INPUT_SUBSTRINGS = ("label", "target", "future", "pnl")
TARGET_ALIAS_SUBSTRINGS = ("label", "target", "future", "pnl", "action_score")

REGIME3_REQUIRED_FEATURES = [
    "regime3_stability_h6_score",
    "regime3_transition_h6_risk_prob",
    "regime3_transition_h6_risk_pred",
    "regime3_churn_h6_risk_score",
]

REGIME3_CURRENT_REQUIRED_FEATURES = [
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
    "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_entropy",
    "regime3_current_sensitive_wide24_margin",
]

REGIME3_CMAMBA_FEATURES = [
    "regime3_cmamba_h6_future_bull_prob",
    "regime3_cmamba_h6_future_bear_prob",
    "regime3_cmamba_h6_future_chop_prob",
    "regime3_cmamba_h6_confidence",
    "regime3_cmamba_h6_transition_prob",
    "regime3_cmamba_h6_stability_score",
]

CHRONOS_REQUIRED_FEATURES = [
    "chronos_atr14_upside_band_ewm3",
    "chronos_atr14_width_ewm6",
    "chronos_atr14_width",
    "chronos_atr14_large_move_score",
    "chronos_realized_vol24_width",
    "chronos_realized_vol24_large_move_score",
]

M7_REQUIRED_FEATURES = [
    "m7_q10",
    "m7_q90",
    "m7_qwidth",
    "m7_zigzag_cat_fl",
    "m7_zigzag_cat_up",
    "m7_zigzag_cat_dn",
    "m7_zigzag_cat_confidence",
    "m7_zigzag_cat_side_edge",
    "m7_zigzag_cat_trade_prob",
    "m7_zigzag_xgb_fl",
    "m7_zigzag_xgb_up",
    "m7_zigzag_xgb_dn",
    "m7_zigzag_xgb_confidence",
    "m7_zigzag_xgb_side_edge",
    "m7_zigzag_xgb_trade_prob",
]

TEACHER_HGB_OUTPUTS = [
    "teacher_hgb_p_cash",
    "teacher_hgb_p_long",
    "teacher_hgb_p_short",
    "teacher_hgb_confidence",
    "teacher_hgb_side_edge",
    "teacher_hgb_uncertainty",
    "teacher_hgb_risk_veto_score",
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _read_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    required = {"timestamp"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _exact_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    cols: list[str],
    source: str,
    *,
    allow_tail_drop: bool = False,
    allow_edge_drop: bool = False,
) -> pd.DataFrame:
    before = len(left)
    merged = left.merge(right[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    if len(merged) != before:
        raise RuntimeError(f"{source} changed row count: {before} -> {len(merged)}")
    missing = {col: int(merged[col].isna().sum()) for col in cols if int(merged[col].isna().sum()) > 0}
    if missing:
        miss_any = merged[cols].isna().any(axis=1).to_numpy()
        miss_idx = np.flatnonzero(miss_any)
        tail_only = miss_idx.size > 0 and np.array_equal(miss_idx, np.arange(len(merged) - miss_idx.size, len(merged)))
        if allow_tail_drop and tail_only:
            return merged.iloc[: len(merged) - miss_idx.size].reset_index(drop=True)
        head_only = miss_idx.size > 0 and np.array_equal(miss_idx, np.arange(0, miss_idx.size))
        if allow_edge_drop and head_only:
            return merged.iloc[miss_idx.size :].reset_index(drop=True)
        if allow_edge_drop and tail_only:
            return merged.iloc[: len(merged) - miss_idx.size].reset_index(drop=True)
        raise RuntimeError(f"{source} exact timestamp join has missing values: {missing}")
    return merged


def _add_regime3(frame: pd.DataFrame, *, year: int, regime3_dir: Path) -> pd.DataFrame:
    name = "training_features_2025_regime3_stability_risk_h6.csv" if int(year) == 2025 else "training_features_2026_rebuilt_regime3_stability_risk_h6.csv"
    path = regime3_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    regime = pd.read_csv(path, parse_dates=["timestamp"])
    missing = sorted(set(REGIME3_REQUIRED_FEATURES) - set(regime.columns))
    if missing:
        raise ValueError(f"{path} missing required Regime3 columns: {missing}")
    return _exact_join(frame, regime, REGIME3_REQUIRED_FEATURES, f"Regime3 h6 {year}", allow_tail_drop=True)


def _add_regime3_current(frame: pd.DataFrame, *, year: int, regime3_current_dir: Path) -> pd.DataFrame:
    name = (
        "training_features_2025_regime3_current_sensitive_hmm_wide24.csv"
        if int(year) == 2025
        else "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"
    )
    path = regime3_current_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    current = pd.read_csv(path, parse_dates=["timestamp"])
    missing = sorted(set(REGIME3_CURRENT_REQUIRED_FEATURES) - set(current.columns))
    if missing:
        raise ValueError(f"{path} missing required Regime3 current columns: {missing}")
    return _exact_join(
        frame,
        current,
        REGIME3_CURRENT_REQUIRED_FEATURES,
        f"Regime3 current sensitive wide24 {year}",
        allow_tail_drop=True,
    )


def _add_regime3_cmamba(frame: pd.DataFrame, *, year: int, regime3_cmamba_dir: Path) -> pd.DataFrame:
    name = (
        "training_features_2025_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv"
        if int(year) == 2025
        else "training_features_2026_rebuilt_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv"
    )
    path = regime3_cmamba_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    side = pd.read_csv(path, parse_dates=["timestamp"])
    missing = sorted(set(REGIME3_CMAMBA_FEATURES) - set(side.columns))
    if missing:
        raise ValueError(f"{path} missing required Regime3 CryptoMamba columns: {missing}")
    return _exact_join(
        frame,
        side,
        REGIME3_CMAMBA_FEATURES,
        f"Regime3 CryptoMamba h6 {year}",
        allow_edge_drop=True,
    )


def _chronos_series_features(path: Path, prefix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "q10", "q50", "q90", "width"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{path} missing required Chronos columns: {missing}")
    q50 = pd.to_numeric(raw["q50"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    q90 = pd.to_numeric(raw["q90"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    width = pd.to_numeric(raw["width"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    out = pd.DataFrame({"timestamp": raw["timestamp"]})
    out[f"chronos_{prefix}_width"] = width.astype("float32")
    out[f"chronos_{prefix}_large_move_score"] = (width * (1.0 + q50.abs())).astype("float32")
    out[f"chronos_{prefix}_upside_band_ewm3"] = q90.clip(lower=0.0).ewm(span=3, adjust=False, min_periods=1).mean().astype("float32")
    out[f"chronos_{prefix}_width_ewm6"] = width.ewm(span=6, adjust=False, min_periods=1).mean().astype("float32")
    return out


def _add_chronos(frame: pd.DataFrame, *, year: int, chronos_dir: Path) -> pd.DataFrame:
    split = "val2025" if int(year) == 2025 else "oos2026"
    atr = _chronos_series_features(chronos_dir / f"atr14_pct_{split}_chronos.csv", "atr14")
    rv = _chronos_series_features(chronos_dir / f"realized_vol_24_{split}_chronos.csv", "realized_vol24")
    chronos = atr.merge(rv, on="timestamp", how="inner", validate="one_to_one")
    missing = sorted(set(CHRONOS_REQUIRED_FEATURES) - set(chronos.columns))
    if missing:
        raise ValueError(f"Chronos derived feature set missing required columns: {missing}")
    return _exact_join(frame, chronos, CHRONOS_REQUIRED_FEATURES, f"Chronos uncertainty {year}", allow_tail_drop=True)


def _add_m7_zigzag(frame: pd.DataFrame, *, year: int, m7_zigzag_dir: Path) -> pd.DataFrame:
    path = m7_zigzag_dir / f"rl_training_{int(year)}_m7_zigzag_direction.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    m7 = pd.read_csv(path, parse_dates=["timestamp"])
    missing = sorted(set(M7_REQUIRED_FEATURES) - set(m7.columns))
    if missing:
        raise ValueError(f"{path} missing required M7 teacher columns: {missing}")
    overlap = [col for col in M7_REQUIRED_FEATURES if col in frame.columns]
    if overlap:
        check = frame[["timestamp", *overlap]].merge(
            m7[["timestamp", *overlap]],
            on="timestamp",
            how="left",
            suffixes=("_existing", "_source"),
            validate="one_to_one",
        )
        missing_overlap = {
            col: int(check[f"{col}_source"].isna().sum())
            for col in overlap
            if int(check[f"{col}_source"].isna().sum()) > 0
        }
        if missing_overlap:
            raise RuntimeError(f"M7 ZigZag direction {year} overlap source missing values: {missing_overlap}")
        mismatched: list[str] = []
        for col in overlap:
            existing = pd.to_numeric(check[f"{col}_existing"], errors="coerce")
            source = pd.to_numeric(check[f"{col}_source"], errors="coerce")
            valid = existing.notna() & source.notna()
            if int(valid.sum()) != len(check) or float((existing[valid] - source[valid]).abs().max()) > 1e-10:
                mismatched.append(col)
        if mismatched:
            raise RuntimeError(f"M7 ZigZag direction {year} overlap columns differ from source: {mismatched}")
    join_cols = [col for col in M7_REQUIRED_FEATURES if col not in frame.columns]
    if not join_cols:
        return frame
    return _exact_join(frame, m7, join_cols, f"M7 ZigZag direction {year}", allow_tail_drop=False)


def _add_zigzag_label(frame: pd.DataFrame, *, year: int, zigzag_label_dir: Path) -> pd.DataFrame:
    path = zigzag_label_dir / f"zigzag_action_labels_{int(year)}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    labels = pd.read_csv(path, parse_dates=["timestamp"])
    required = ["zigzag_action"]
    missing = sorted(set(required) - set(labels.columns))
    if missing:
        raise ValueError(f"{path} missing required ZigZag label columns: {missing}")
    return _exact_join(frame, labels, required, f"ZigZag action labels {year}", allow_tail_drop=False)


def _zigzag_labels(frame: pd.DataFrame) -> np.ndarray:
    if "zigzag_action" not in frame.columns:
        raise ValueError("missing active Omega1 label column: zigzag_action")
    y = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    invalid = sorted(set(np.unique(y).tolist()) - {0, 1, 2})
    if invalid:
        raise ValueError(f"invalid zigzag_action class values: {invalid}")
    return y


def _teacher_input_cols(train: pd.DataFrame, oos: pd.DataFrame) -> list[str]:
    missing = [col for col in OMEGA1_CORE_INPUTS if col not in train.columns or col not in oos.columns]
    if missing:
        raise ValueError(f"Omega1 core teacher inputs are missing: {missing}")
    cols: list[str] = []
    for col in OMEGA1_CORE_INPUTS:
        if col.startswith(FORBIDDEN_INPUT_PREFIXES):
            raise ValueError(f"Omega1 forbidden prefix selected: {col}")
        lower = col.lower()
        if any(token in lower for token in FORBIDDEN_INPUT_SUBSTRINGS) and col not in REGIME3_CMAMBA_FEATURES:
            raise ValueError(f"Omega1 forbidden token selected: {col}")
        if not pd.api.types.is_numeric_dtype(train[col]) or not pd.api.types.is_numeric_dtype(oos[col]):
            raise TypeError(f"Omega1 teacher input must be numeric: {col}")
        cols.append(col)
    return cols


def _assert_no_target_aliases(frame: pd.DataFrame, feature_cols: list[str]) -> None:
    target_like_cols = [
        col
        for col in frame.columns
        if col not in feature_cols
        and any(token in col.lower() for token in TARGET_ALIAS_SUBSTRINGS)
        and pd.api.types.is_numeric_dtype(frame[col])
    ]
    violations: list[str] = []
    for feat in feature_cols:
        fx = pd.to_numeric(frame[feat], errors="coerce")
        for target_col in target_like_cols:
            ty = pd.to_numeric(frame[target_col], errors="coerce")
            valid = fx.notna() & ty.notna()
            if int(valid.sum()) == 0:
                continue
            if float((fx[valid] - ty[valid]).abs().max()) <= 1e-12:
                violations.append(f"{feat} == {target_col}")
    if violations:
        raise RuntimeError(f"selected feature aliases target-like column(s): {violations[:20]}")


def _fit_hgb(train: pd.DataFrame, feature_cols: list[str], y: np.ndarray, seed: int) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=260,
                    learning_rate=0.035,
                    max_leaf_nodes=31,
                    l2_regularization=0.05,
                    early_stopping=True,
                    validation_fraction=0.12,
                    n_iter_no_change=25,
                    random_state=int(seed),
                ),
            ),
        ]
    )
    weights = compute_sample_weight(class_weight="balanced", y=y)
    model.fit(train[feature_cols], y, model__sample_weight=weights)
    return model


def _append_outputs(frame: pd.DataFrame, model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    proba = model.predict_proba(frame[feature_cols])
    classes = list(model.named_steps["model"].classes_)
    full = np.zeros((len(frame), 3), dtype=np.float64)
    for idx, cls in enumerate(classes):
        full[:, int(cls)] = proba[:, idx]
    out = frame.copy()
    out["teacher_hgb_p_cash"] = full[:, 0]
    out["teacher_hgb_p_long"] = full[:, 1]
    out["teacher_hgb_p_short"] = full[:, 2]
    out["teacher_hgb_confidence"] = np.max(full, axis=1)
    out["teacher_hgb_side_edge"] = full[:, 1] - full[:, 2]
    out["teacher_hgb_uncertainty"] = 1.0 - np.max(full, axis=1)
    out["teacher_hgb_risk_veto_score"] = np.clip(full[:, 0] + out["teacher_hgb_uncertainty"], 0.0, 1.0)
    return out


def _metrics(frame: pd.DataFrame, y: np.ndarray, model: Pipeline, feature_cols: list[str]) -> dict[str, Any]:
    proba = model.predict_proba(frame[feature_cols])
    classes = list(model.named_steps["model"].classes_)
    full = np.zeros((len(frame), 3), dtype=np.float64)
    for idx, cls in enumerate(classes):
        full[:, int(cls)] = proba[:, idx]
    pred = np.argmax(full, axis=1).astype(np.int64)
    metrics: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y, minlength=3))},
        "pred_counts": {str(i): int(v) for i, v in enumerate(np.bincount(pred, minlength=3))},
    }
    try:
        metrics["ovr_auc"] = float(roc_auc_score(y, full, multi_class="ovr", labels=[0, 1, 2]))
    except ValueError:
        metrics["ovr_auc"] = None
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--regime3-dir", type=Path, default=DEFAULT_REGIME3_DIR)
    parser.add_argument("--regime3-current-dir", type=Path, default=DEFAULT_REGIME3_CURRENT_DIR)
    parser.add_argument("--regime3-cmamba-dir", type=Path, default=DEFAULT_REGIME3_CMAMBA_DIR)
    parser.add_argument("--chronos-dir", type=Path, default=DEFAULT_CHRONOS_DIR)
    parser.add_argument("--m7-zigzag-dir", type=Path, default=DEFAULT_M7_ZIGZAG_DIR)
    parser.add_argument("--zigzag-label-dir", type=Path, default=DEFAULT_ZIGZAG_LABEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260531)
    args = parser.parse_args()

    train_name = "trade_candidates_2025_alpha6_current_tail111_exact.csv"
    oos_name = "trade_candidates_2026_alpha6_current_tail111_exact.csv"
    train = _read_candidates(args.candidate_dir / train_name)
    train = _add_regime3(train, year=2025, regime3_dir=args.regime3_dir)
    train = _add_regime3_current(train, year=2025, regime3_current_dir=args.regime3_current_dir)
    train = _add_regime3_cmamba(train, year=2025, regime3_cmamba_dir=args.regime3_cmamba_dir)
    train = _add_chronos(train, year=2025, chronos_dir=args.chronos_dir)
    train = _add_m7_zigzag(train, year=2025, m7_zigzag_dir=args.m7_zigzag_dir)
    train = _add_zigzag_label(
        train,
        year=2025,
        zigzag_label_dir=args.zigzag_label_dir,
    )
    oos = _read_candidates(args.candidate_dir / oos_name)
    oos = _add_regime3(oos, year=2026, regime3_dir=args.regime3_dir)
    oos = _add_regime3_current(oos, year=2026, regime3_current_dir=args.regime3_current_dir)
    oos = _add_regime3_cmamba(oos, year=2026, regime3_cmamba_dir=args.regime3_cmamba_dir)
    oos = _add_chronos(oos, year=2026, chronos_dir=args.chronos_dir)
    oos = _add_m7_zigzag(oos, year=2026, m7_zigzag_dir=args.m7_zigzag_dir)
    oos = _add_zigzag_label(
        oos,
        year=2026,
        zigzag_label_dir=args.zigzag_label_dir,
    )

    feature_cols = _teacher_input_cols(train, oos)
    _assert_no_target_aliases(train, feature_cols)
    _assert_no_target_aliases(oos, feature_cols)
    y_train = _zigzag_labels(train)
    y_oos = _zigzag_labels(oos)
    model = _fit_hgb(train, feature_cols, y_train, args.seed)

    train_out = _append_outputs(train, model, feature_cols)
    oos_out = _append_outputs(oos, model, feature_cols)
    for col in TEACHER_HGB_OUTPUTS:
        if col not in train_out.columns or col not in oos_out.columns:
            raise RuntimeError(f"missing teacher HGB output: {col}")
        if train_out[col].isna().any() or oos_out[col].isna().any():
            raise RuntimeError(f"teacher HGB output has NaN: {col}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / train_name
    oos_path = args.out_dir / oos_name
    train_save = train_out.drop(columns=REGIME3_REQUIRED_FEATURES, errors="ignore")
    oos_save = oos_out.drop(columns=REGIME3_REQUIRED_FEATURES, errors="ignore")
    train_save.to_csv(train_path, index=False)
    oos_save.to_csv(oos_path, index=False)
    model_path = args.out_dir / "omega1_teacher_hgb_v1.joblib"
    joblib.dump({"model_id": MODEL_ID, "model": model, "feature_cols": feature_cols, "label_source": "zigzag_action"}, model_path)

    audit = {
        "model_id": MODEL_ID,
        "version": "omega1",
        "candidate_dir": str(args.candidate_dir),
        "regime3_current_dir": str(args.regime3_current_dir),
        "chronos_dir": str(args.chronos_dir),
        "m7_zigzag_dir": str(args.m7_zigzag_dir),
        "zigzag_label_dir": str(args.zigzag_label_dir),
        "out_dir": str(args.out_dir),
        "label_source": "zigzag_action",
        "feature_count": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "outputs": TEACHER_HGB_OUTPUTS,
        "train_metrics": _metrics(train, y_train, model, feature_cols),
        "oos_label_probe_metrics": _metrics(oos, y_oos, model, feature_cols),
        "artifacts": {"train_csv": str(train_path), "oos_csv": str(oos_path), "model": str(model_path)},
        "contract": {
            "allowed_context_families": [
                "tide_risk",
                "chronos_uncertainty",
                "regime3_current",
                "regime3_h6",
                "regime3_cmamba_h6_future_context",
                "m7_quantile_risk",
                "m7_zigzag_direction",
            ],
            "selection_policy": "explicit_pass_only_core_inputs_no_prefix_sweep",
            "regime4_policy": "forbidden_all_regime4_prefixes_in_omega1_teacher_inputs",
            "forbidden_input_prefixes": FORBIDDEN_INPUT_PREFIXES,
            "forbidden_input_substrings": FORBIDDEN_INPUT_SUBSTRINGS,
            "target_alias_guard": True,
            "regime3_exact_join": True,
            "regime3_current_exact_join": True,
            "regime3_cmamba_exact_join": True,
            "future_token_exact_column_exceptions": REGIME3_CMAMBA_FEATURES,
            "chronos_exact_join": True,
            "m7_zigzag_exact_join": True,
            "zigzag_label_exact_join": True,
            "tail_only_missing_drop": True,
        },
    }
    if any(col.startswith(("clean_regime_", "clean_regime4_", "regime4_pred_")) for col in feature_cols):
        raise RuntimeError("omega1 teacher input contract violation: Regime4 column selected")
    (args.out_dir / "omega1_teacher_hgb_v1_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default))
    print(json.dumps(audit, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
