#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_label_family_20260606 as label_family  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_full_retrain_cash_alpha43_20260608 as full  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega2_1_label_atr1_h24_hgb_12seed_ensemble_thr055"
ARTIFACT_DIR = ROOT / "data/ensemble/supervised" / MODEL_ID
REPORT_DIR = ROOT / "tmp/causal_regen_20260516" / "omega2_1_cash_sleeve_freeze_verify_20260609"
BUNDLE_PATH = ARTIFACT_DIR / "omega2_1_hgb_12seed_cash_sleeve.joblib"
MANIFEST_PATH = ARTIFACT_DIR / "candidate_manifest.json"
SEEDS = (260000, 260001, 260002, 260003, 260004, 260005, 260006, 260007, 260008, 260009, 260608, 260780)
THRESHOLD = 0.55
RISK = sleeve.FallbackRisk("tp026_sl014_n0.30_h192", 0.026, 0.014, 0.30, 2.0, 192)
FORBIDDEN_PREFIXES = ("clean_regime4_", "regime4_pred_", "teacher_", "exit_head_")
FORBIDDEN_EXACT = {"tp_sl_action_score"}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _reject_forbidden(cols: list[str], tag: str) -> None:
    bad = [
        col
        for col in cols
        if col in FORBIDDEN_EXACT or any(str(col).startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
    ]
    if bad:
        raise RuntimeError(f"{tag} forbidden feature columns: {bad[:40]}")


def _model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.035,
        max_leaf_nodes=7,
        l2_regularization=2.0,
        random_state=int(seed),
    )


def _classes_to_proba(model: Any, proba: np.ndarray) -> np.ndarray:
    out = np.zeros((len(proba), 3), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    for j, cls in enumerate(classes):
        cls_i = int(cls)
        if 0 <= cls_i <= 2:
            out[:, cls_i] = proba[:, j]
    return out


def _fit_models(x: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray) -> list[HistGradientBoostingClassifier]:
    idx = np.flatnonzero(train_mask)
    if len(idx) < 500:
        raise RuntimeError(f"Omega2.1 train rows too small: {len(idx)}")
    if len(np.unique(y[idx])) < 2:
        raise RuntimeError("Omega2.1 labels are single-class")
    models: list[HistGradientBoostingClassifier] = []
    arr = x.iloc[idx].to_numpy(dtype=np.float64)
    yy = y[idx]
    for seed in SEEDS:
        model = _model(seed)
        model.fit(arr, yy)
        models.append(model)
    return models


def _predict_proba(models: list[Any], x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    arr = x.to_numpy(dtype=np.float64)
    probs = []
    preds = []
    for model in models:
        p = _classes_to_proba(model, model.predict_proba(arr))
        probs.append(p)
        preds.append(np.argmax(p, axis=1))
    return np.stack(probs, axis=0).mean(axis=0), np.stack(preds, axis=0)


def _action_conf(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action = np.argmax(proba, axis=1).astype(np.int64)
    conf = proba[np.arange(len(proba)), action].astype(np.float64)
    return action, conf


def _metric(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_fallback_entries": int(metrics.get("fallback_entries", 0)),
        f"{prefix}_primary_takeovers": int(metrics.get("primary_takeovers", 0)),
        f"{prefix}_exit_reasons": metrics.get("exit_reasons", {}),
    }


def _evaluate(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    proba: np.ndarray,
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    action, conf = _action_conf(proba)
    return sleeve._metrics_with_fallback(frame, dec, RISK, action, conf, THRESHOLD, fee=fee, slip=slip, cost_mult=3.0)


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_dec, val_features = full._build_split(frames, "validation")
    oos_frame, oos_dec, oos_features = full._build_split(frames, "oos")
    feature_cols = list(val_features.columns)
    if feature_cols != list(oos_features.columns):
        raise RuntimeError("Omega2.1 validation/OOS feature contract mismatch")
    _reject_forbidden(feature_cols, "Omega2.1")

    y, valid_mask, label_diag = label_family._triple_barrier_labels(
        val_frame,
        atr_mult=1.0,
        max_hold=24,
        min_barrier=0.0035,
    )
    val_cash = ~omega._active(val_dec)
    train_mask = val_cash & valid_mask
    models = _fit_models(val_features, y, train_mask)

    val_proba, val_pred_stack = _predict_proba(models, val_features)
    oos_proba, oos_pred_stack = _predict_proba(models, oos_features)
    val_metrics = _evaluate(val_frame, val_dec, val_proba, fee=fee, slip=slip)
    oos_metrics = _evaluate(oos_frame, oos_dec, oos_proba, fee=fee, slip=slip)

    val_pred = np.argmax(val_proba, axis=1)
    oos_pred = np.argmax(oos_proba, axis=1)
    bundle = {
        "model_id": MODEL_ID,
        "created_at": "2026-06-09",
        "models": models,
        "seeds": list(SEEDS),
        "feature_cols": feature_cols,
        "threshold": THRESHOLD,
        "risk": RISK.__dict__,
        "label": {"name": "label_atr1_h24", "atr_mult": 1.0, "max_hold": 24, "min_barrier": 0.0035},
        "train_rows": int(np.count_nonzero(train_mask)),
        "label_diag": label_diag,
        "forbidden_feature_audit": {"passed": True, "forbidden": []},
        "reference_selection": {
            "source_report": "tmp/causal_regen_20260516/omega2_architect_priority_experiments_20260609/report.json",
            "validation_oof_pnl": 111.959707,
            "oos_full_train_pnl": 102.611483,
            "oos_full_train_mdd": -8.108171,
            "oos_full_train_wr": 0.609756,
            "oos_full_train_trades": 41,
        },
    }
    joblib.dump(bundle, BUNDLE_PATH)

    manifest = {
        "model_id": MODEL_ID,
        "status": "frozen_research_candidate_not_live_promoted",
        "artifact": str(BUNDLE_PATH.relative_to(ROOT)),
        "contract": "docs/model_contracts/omega2_1_hgb_12seed_cash_sleeve_20260609_contract.md",
        "runtime_adapter": "trading_bot_modules/omega2_1_cash_sleeve.py",
        "parent_baseline": "omega1_2_1_aggressive_compensated_scale200_cap090",
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "forbidden_features": {
            "prefixes": list(FORBIDDEN_PREFIXES),
            "exact": sorted(FORBIDDEN_EXACT),
        },
        "threshold": THRESHOLD,
        "risk": RISK.__dict__,
        "seeds": list(SEEDS),
        "training": {
            "split": "2025 validation primary-cash rows",
            "train_rows": int(np.count_nonzero(train_mask)),
            "label_diag": label_diag,
        },
        "metrics": {
            "validation_full_train_sanity": _metric("validation", val_metrics),
            "oos_full_train_parity": _metric("oos", oos_metrics),
            "selection_reference": bundle["reference_selection"],
        },
        "notes": [
            "Validation full-train sanity is in-sample and must not be used as selection evidence.",
            "Selection evidence is the OOF validation and full-train OOS result from omega2_architect_priority_experiments_20260609.",
            "Active live path is unchanged until explicit live promotion.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")

    pred_report = pd.DataFrame(
        {
            "timestamp": oos_frame["timestamp"],
            "p_cash": oos_proba[:, 0],
            "p_long": oos_proba[:, 1],
            "p_short": oos_proba[:, 2],
            "action": oos_pred,
            "confidence": oos_proba[np.arange(len(oos_proba)), oos_pred],
            "agree_count": (oos_pred_stack == oos_pred[None, :]).sum(axis=0),
        }
    )
    pred_report.to_csv(REPORT_DIR / "oos_predictions.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "bundle": str(BUNDLE_PATH),
        "manifest": str(MANIFEST_PATH),
        "feature_count": len(feature_cols),
        "train_rows": int(np.count_nonzero(train_mask)),
        "label_diag": label_diag,
        "validation_full_train_sanity": val_metrics,
        "oos_full_train_parity": oos_metrics,
        "reference_selection": bundle["reference_selection"],
        "oos_prediction_file": str(REPORT_DIR / "oos_predictions.csv"),
    }
    (REPORT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
