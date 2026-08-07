#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_direction_head_direction_only_20260602"
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_direction_head_direction_only_20260602"
DROP_EVENTS: list[dict[str, Any]] = []

DIR3_VSNLSTM_DIR = ROOT / "data/ensemble/supervised/omega1_dir3_vsnlstm_full_20260531"
DIR3_PATCH_DIR = ROOT / "data/ensemble/supervised/omega1_dir3_patch_full_20260531"
DIR3_DUET_DIR = ROOT / "data/ensemble/supervised/omega1_dir3_duet_20260531"
DIR3_CRYPTOMAMBA_DIR = ROOT / "data/ensemble/supervised/omega1_dir3_cryptomamba_20260531"
DIR3_RETRIEVAL_DIR = ROOT / "data/ensemble/supervised/omega1_dir3_retrieval_20260531"
M7_DIR = ROOT / "data/splits/year_oos"
REGIME3_CURRENT_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
REGIME3_CMAMBA_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531"


DIR3_VSNLSTM = [
    "dir3_vsnlstm_h6_fl_prob",
    "dir3_vsnlstm_h6_up_prob",
    "dir3_vsnlstm_h6_dn_prob",
    "dir3_vsnlstm_h6_confidence",
    "dir3_vsnlstm_h6_side_edge",
    "dir3_vsnlstm_h6_trade_prob",
]
DIR3_PATCH = [
    "dir3_patch_h6_fl_prob",
    "dir3_patch_h6_up_prob",
    "dir3_patch_h6_dn_prob",
    "dir3_patch_h6_confidence",
    "dir3_patch_h6_side_edge",
    "dir3_patch_h6_trade_prob",
]
M7_ZIGZAG = [
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
REGIME3_CURRENT = [
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
    "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_margin",
]
REGIME3_CMAMBA = [
    "regime3_cmamba_h6_future_bull_prob",
    "regime3_cmamba_h6_future_bear_prob",
    "regime3_cmamba_h6_future_chop_prob",
    "regime3_cmamba_h6_confidence",
]
DIR3_DUET = [
    "dir3_duet_h6_fl_prob",
    "dir3_duet_h6_up_prob",
    "dir3_duet_h6_dn_prob",
    "dir3_duet_h6_confidence",
    "dir3_duet_h6_side_edge",
    "dir3_duet_h6_trade_prob",
]
DIR3_CRYPTOMAMBA = [
    "dir3_cryptomamba_h6_fl_prob",
    "dir3_cryptomamba_h6_up_prob",
    "dir3_cryptomamba_h6_dn_prob",
    "dir3_cryptomamba_h6_confidence",
    "dir3_cryptomamba_h6_side_edge",
    "dir3_cryptomamba_h6_trade_prob",
]
DIR3_RETRIEVAL = [
    "dir3_retrieval_h6_fl_prob",
    "dir3_retrieval_h6_up_prob",
    "dir3_retrieval_h6_dn_prob",
    "dir3_retrieval_h6_confidence",
    "dir3_retrieval_h6_side_edge",
    "dir3_retrieval_h6_trade_prob",
    "dir3_retrieval_h6_neighbor_edge_mean",
    "dir3_retrieval_h6_neighbor_edge_q25",
    "dir3_retrieval_h6_neighbor_edge_q75",
    "dir3_retrieval_h6_regime_consensus",
    "dir3_retrieval_h6_similarity_score",
]

VARIANTS = {
    "core": DIR3_VSNLSTM + DIR3_PATCH,
    "expanded": DIR3_VSNLSTM + DIR3_PATCH + M7_ZIGZAG + REGIME3_CURRENT + REGIME3_CMAMBA,
    "all_direction": (
        DIR3_VSNLSTM
        + DIR3_PATCH
        + M7_ZIGZAG
        + REGIME3_CURRENT
        + REGIME3_CMAMBA
        + DIR3_DUET
        + DIR3_CRYPTOMAMBA
        + DIR3_RETRIEVAL
    ),
}

FORBIDDEN_PREFIXES = ("teacher_", "teacher_oof_", "a5dir_", "clean_regime_", "clean_regime4_", "regime4_pred_", "regime3_pred_")
FORBIDDEN_TOKENS = ("label", "target", "pnl", "action_score", "wave3", "zigzag_soft")
FUTURE_EXCEPTIONS = set(REGIME3_CMAMBA)


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


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path} missing timestamp")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _exact_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    cols: list[str],
    source: str,
    *,
    allow_head_drop: bool = False,
    allow_sparse_drop: bool = False,
) -> pd.DataFrame:
    missing_cols = sorted(set(cols) - set(right.columns))
    if missing_cols:
        raise ValueError(f"{source} missing required columns: {missing_cols}")
    before = len(left)
    merged = left.merge(right[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    if len(merged) != before:
        raise RuntimeError(f"{source} changed row count: {before} -> {len(merged)}")
    miss_any = merged[cols].isna().any(axis=1).to_numpy()
    if miss_any.any():
        miss_idx = np.flatnonzero(miss_any)
        head_only = np.array_equal(miss_idx, np.arange(miss_idx.size))
        if allow_head_drop and head_only:
            DROP_EVENTS.append({"source": source, "drop_type": "head", "rows": int(miss_idx.size)})
            return merged.iloc[miss_idx.size :].reset_index(drop=True)
        if allow_sparse_drop:
            DROP_EVENTS.append({"source": source, "drop_type": "sparse", "rows": int(miss_idx.size)})
            return merged.loc[~miss_any].reset_index(drop=True)
        missing = {c: int(merged[c].isna().sum()) for c in cols if int(merged[c].isna().sum()) > 0}
        raise RuntimeError(f"{source} exact timestamp join has missing values: {missing}")
    return merged


def _add_labels(year: int) -> pd.DataFrame:
    labels = _read_csv(LABEL_DIR / f"zigzag_action_labels_{year}.csv")
    required = ["zigzag_action"]
    missing = sorted(set(required) - set(labels.columns))
    if missing:
        raise ValueError(f"zigzag labels {year} missing columns: {missing}")
    out = labels[["timestamp", "zigzag_action"]].copy()
    y = pd.to_numeric(out["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    invalid = sorted(set(np.unique(y).tolist()) - {0, 1, 2})
    if invalid:
        raise ValueError(f"zigzag labels {year} invalid classes: {invalid}")
    return out


def _feature_path(family: str, year: int) -> Path:
    if family == "vsnlstm":
        name = "training_features_2025_omega1_dir3_vsnlstm_full_20260531.csv" if year == 2025 else "training_features_2026_rebuilt_omega1_dir3_vsnlstm_full_20260531.csv"
        return DIR3_VSNLSTM_DIR / name
    if family == "patch":
        name = "training_features_2025_omega1_dir3_patch_full_20260531.csv" if year == 2025 else "training_features_2026_rebuilt_omega1_dir3_patch_full_20260531.csv"
        return DIR3_PATCH_DIR / name
    if family == "m7":
        name = "rl_training_2025_m7_zigzag_direction.csv" if year == 2025 else "rl_training_2026_m7_zigzag_direction.csv"
        return M7_DIR / name
    if family == "regime3_current":
        name = "training_features_2025_regime3_current_sensitive_hmm_wide24.csv" if year == 2025 else "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"
        return REGIME3_CURRENT_DIR / name
    if family == "regime3_cmamba":
        name = "training_features_2025_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv" if year == 2025 else "training_features_2026_rebuilt_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv"
        return REGIME3_CMAMBA_DIR / name
    if family == "duet":
        name = "training_features_2025_omega1_dir3_duet_20260531.csv" if year == 2025 else "training_features_2026_rebuilt_omega1_dir3_duet_20260531.csv"
        return DIR3_DUET_DIR / name
    if family == "cryptomamba":
        name = "training_features_2025_omega1_dir3_cryptomamba_20260531.csv" if year == 2025 else "training_features_2026_rebuilt_omega1_dir3_cryptomamba_20260531.csv"
        return DIR3_CRYPTOMAMBA_DIR / name
    if family == "retrieval":
        name = "training_features_2025_omega1_dir3_retrieval_20260531.csv" if year == 2025 else "training_features_2026_rebuilt_omega1_dir3_retrieval_20260531.csv"
        return DIR3_RETRIEVAL_DIR / name
    raise ValueError(f"unknown family: {family}")


def _build_frame(year: int) -> pd.DataFrame:
    frame = _add_labels(year)
    joins = [
        ("vsnlstm", DIR3_VSNLSTM, True),
        ("patch", DIR3_PATCH, True),
        ("m7", M7_ZIGZAG, True),
        ("regime3_current", REGIME3_CURRENT, False),
        ("regime3_cmamba", REGIME3_CMAMBA, True),
        ("duet", DIR3_DUET, False),
        ("cryptomamba", DIR3_CRYPTOMAMBA, False),
        ("retrieval", DIR3_RETRIEVAL, False),
    ]
    for family, cols, allow_head_drop in joins:
        frame = _exact_join(
            frame,
            _read_csv(_feature_path(family, year)),
            cols,
            f"{family} {year}",
            allow_head_drop=allow_head_drop,
            allow_sparse_drop=(family in {"m7", "retrieval"}),
        )
    return frame


def _validate_features(cols: list[str], frame: pd.DataFrame) -> None:
    seen: set[str] = set()
    duplicates = [c for c in cols if c in seen or seen.add(c)]
    if duplicates:
        raise ValueError(f"duplicate feature columns: {duplicates}")
    missing = sorted(c for c in cols if c not in frame.columns)
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    for col in cols:
        lower = col.lower()
        if any(col.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            raise ValueError(f"forbidden prefix selected: {col}")
        if "future" in lower and col not in FUTURE_EXCEPTIONS:
            raise ValueError(f"forbidden future token selected: {col}")
        if any(token in lower for token in FORBIDDEN_TOKENS):
            raise ValueError(f"forbidden token selected: {col}")
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise TypeError(f"feature must be numeric: {col}")


def _fit_catboost(x: pd.DataFrame, y: np.ndarray, *, seed: int, iterations: int) -> CatBoostClassifier:
    weights = compute_sample_weight(class_weight="balanced", y=y)
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        iterations=int(iterations),
        depth=5,
        learning_rate=0.035,
        l2_leaf_reg=6.0,
        random_seed=int(seed),
        od_type="Iter",
        od_wait=50,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    model.fit(x, y, sample_weight=weights)
    return model


def _proba3(model: CatBoostClassifier, x: pd.DataFrame) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = [int(c) for c in model.classes_]
    full = np.zeros((len(x), 3), dtype=np.float64)
    for j, cls in enumerate(classes):
        full[:, cls] = proba[:, j]
    return full


def _metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(proba, axis=1).astype(np.int64)
    trade = pred != 0
    out: dict[str, Any] = {
        "rows": int(len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "pred_counts": {str(i): int(v) for i, v in enumerate(np.bincount(pred, minlength=3))},
        "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y, minlength=3))},
        "proxy_trades": int(trade.sum()),
        "proxy_trade_rate": float(trade.mean()),
        "proxy_wr": float((pred[trade] == y[trade]).mean()) if trade.any() else None,
        "mean_confidence": float(np.max(proba, axis=1).mean()),
    }
    try:
        out["ovr_auc"] = float(roc_auc_score(y, proba, multi_class="ovr", labels=[0, 1, 2]))
    except ValueError:
        out["ovr_auc"] = None
    return out


def _outputs(frame: pd.DataFrame, proba: np.ndarray, *, prefix: str) -> pd.DataFrame:
    action = np.argmax(proba, axis=1).astype(np.int64)
    return pd.DataFrame(
        {
            "timestamp": frame["timestamp"].to_numpy(),
            f"{prefix}_p_cash": proba[:, 0],
            f"{prefix}_p_long": proba[:, 1],
            f"{prefix}_p_short": proba[:, 2],
            f"{prefix}_confidence": np.max(proba, axis=1),
            f"{prefix}_side_edge": proba[:, 1] - proba[:, 2],
            f"{prefix}_trade_prob": proba[:, 1] + proba[:, 2],
            f"{prefix}_action": action,
        }
    )


def _oof_proba(train: pd.DataFrame, feature_cols: list[str], *, seed: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    n = len(train)
    starts = [int(n * 0.35), int(n * 0.50), int(n * 0.65), int(n * 0.80)]
    ends = [int(n * 0.50), int(n * 0.65), int(n * 0.80), n]
    proba = np.full((n, 3), np.nan, dtype=np.float64)
    covered = np.zeros(n, dtype=bool)
    folds: list[dict[str, Any]] = []
    y = train["zigzag_action"].to_numpy(dtype=np.int64)
    for fold, (start, end) in enumerate(zip(starts, ends), start=1):
        if start <= 0 or end <= start:
            raise RuntimeError(f"invalid OOF fold: {fold} {start}->{end}")
        model = _fit_catboost(train.iloc[:start][feature_cols], y[:start], seed=seed + fold, iterations=500)
        pred = _proba3(model, train.iloc[start:end][feature_cols])
        proba[start:end] = pred
        covered[start:end] = True
        folds.append({"fold": fold, "train_rows": int(start), "predict_start": int(start), "predict_end": int(end), "metrics": _metrics(y[start:end], pred)})
    return proba, covered, folds


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = _build_frame(2025)
    oos = _build_frame(2026)
    y_train = train["zigzag_action"].to_numpy(dtype=np.int64)
    y_oos = oos["zigzag_action"].to_numpy(dtype=np.int64)

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "label_source": "zigzag_action",
        "train_year": 2025,
        "oos_year": 2026,
        "teacher_policy": "teacher_* retired and forbidden as inputs",
        "variants": {},
        "drop_events": DROP_EVENTS,
        "artifacts": {"out_dir": str(OUT_DIR)},
    }
    ranking: list[dict[str, Any]] = []
    for idx, (variant, feature_cols) in enumerate(VARIANTS.items(), start=1):
        _validate_features(feature_cols, train)
        _validate_features(feature_cols, oos)
        variant_dir = OUT_DIR / variant
        variant_dir.mkdir(parents=True, exist_ok=True)

        oof, covered, folds = _oof_proba(train, feature_cols, seed=20260602 + idx * 100)
        oof_metrics = _metrics(y_train[covered], oof[covered])
        final_model = _fit_catboost(train[feature_cols], y_train, seed=20260602 + idx, iterations=800)
        oos_proba = _proba3(final_model, oos[feature_cols])
        oos_metrics = _metrics(y_oos, oos_proba)

        oof_out = _outputs(train.loc[covered].reset_index(drop=True), oof[covered], prefix="omega1_dir_oof")
        oos_out = _outputs(oos, oos_proba, prefix="omega1_dir")
        oof_path = variant_dir / f"training_features_2025_{variant}_omega1_direction_head_oof_20260602.csv"
        oos_path = variant_dir / f"training_features_2026_rebuilt_{variant}_omega1_direction_head_20260602.csv"
        oof_out.to_csv(oof_path, index=False)
        oos_out.to_csv(oos_path, index=False)
        model_path = variant_dir / f"{variant}_omega1_direction_head.cbm"
        final_model.save_model(str(model_path))
        joblib.dump({"feature_cols": feature_cols, "variant": variant, "label_source": "zigzag_action"}, variant_dir / f"{variant}_omega1_direction_head_contract.joblib")

        payload = {
            "variant": variant,
            "feature_count": int(len(feature_cols)),
            "feature_cols": feature_cols,
            "oof_coverage": float(covered.mean()),
            "oof_rows": int(covered.sum()),
            "oof_metrics": oof_metrics,
            "oos_metrics": oos_metrics,
            "folds": folds,
            "artifacts": {"oof_2025": str(oof_path), "oos_2026": str(oos_path), "model": str(model_path)},
        }
        report["variants"][variant] = payload
        ranking.append(
            {
                "variant": variant,
                "feature_count": len(feature_cols),
                "oof_bacc": oof_metrics["balanced_accuracy"],
                "oof_auc": oof_metrics["ovr_auc"],
                "oof_proxy_wr": oof_metrics["proxy_wr"],
                "oos_bacc": oos_metrics["balanced_accuracy"],
                "oos_auc": oos_metrics["ovr_auc"],
                "oos_proxy_wr": oos_metrics["proxy_wr"],
                "oos_proxy_trades": oos_metrics["proxy_trades"],
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
