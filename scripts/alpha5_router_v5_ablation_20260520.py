#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha5_router_v5_train_20260520 import (  # noqa: E402
    DEFAULT_DATA_DIR,
    ROUTER_FEATURE_COLS,
    _fit_router,
    _num,
    _prepare_frame,
    _router3_label,
    _router3_weight,
    _router4_label,
    _router4_weight,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


DEFAULT_BASE_META = ROOT / "tmp/causal_regen_20260516/alpha5_router_v5_train_singlefile_20260520/router_ensemble_meta.joblib"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_router_v5_ablation_20260520"
REGIME_PROB_COLS = [
    "clean_regime4_2024_unsup_v1_bear_prob",
    "clean_regime4_2024_unsup_v1_bull_prob",
    "clean_regime4_2024_unsup_v1_trend_prob",
    "clean_regime4_2024_unsup_v1_whipsaw_prob",
]


def _clean_x(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def _load_data(data_dir: Path) -> dict[str, Any]:
    raw = {
        "train": pd.read_parquet(data_dir / "alpha5_29_hier_label_factory_train.parquet"),
        "val": pd.read_parquet(data_dir / "alpha5_29_hier_label_factory_val.parquet"),
        "oos": pd.read_parquet(data_dir / "alpha5_29_hier_label_factory_oos.parquet"),
    }
    work = {k: _prepare_frame(v) for k, v in raw.items()}
    x = {k: _clean_x(v[ROUTER_FEATURE_COLS]) for k, v in work.items()}
    keep = {k: _num(v, "split_keep", 0.0).astype(np.int8) == 1 for k, v in raw.items()}
    y3 = {k: _router3_label(v)[keep[k]] for k, v in raw.items()}
    y4 = {k: _router4_label(v)[keep[k]] for k, v in raw.items()}
    return {"raw": raw, "work": work, "x": x, "y3": y3, "y4": y4}


def _load_model(path: Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def _component_probas(meta_path: Path, x: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    meta = joblib.load(meta_path)
    comps = list(meta["components"])
    if len(comps) != 2:
        raise ValueError(f"expected two router5 components, got {len(comps)}")
    model3 = _load_model(Path(comps[0]["model_path"]))
    model4 = _load_model(Path(comps[1]["model_path"]))
    out: dict[str, np.ndarray] = {}
    for split, xf in x.items():
        p3_raw = np.asarray(model3.predict_proba(xf), dtype=np.float64)
        p4_raw = np.asarray(model4.predict_proba(xf), dtype=np.float64)
        c3 = [int(c) for c in getattr(model3, "classes_", [0, 1, 2])]
        c4 = [int(c) for c in getattr(model4, "classes_", [0, 1, 2, 3])]
        i3 = {c: i for i, c in enumerate(c3)}
        i4 = {c: i for i, c in enumerate(c4)}
        p3 = np.stack([p3_raw[:, i3[0]], p3_raw[:, i3[1]], p3_raw[:, i3[2]]], axis=1)
        p4 = np.stack([p4_raw[:, i4[0]] + p4_raw[:, i4[1]], p4_raw[:, i4[2]], p4_raw[:, i4[3]]], axis=1)
        out[f"{split}_p3"] = p3
        out[f"{split}_p4"] = p4
    return out


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.nan_to_num(np.asarray(p, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 1e-9, None)
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-9)


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    ok = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    val = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if not np.any(mask):
            continue
        val += float(np.mean(mask)) * abs(float(ok[mask].mean()) - float(conf[mask].mean()))
    return float(val if total else 0.0)


def _multi_brier(y: np.ndarray, p: np.ndarray) -> float:
    y_one = np.eye(3, dtype=np.float64)[y.astype(np.int64)]
    return float(np.mean(np.sum((p - y_one) ** 2, axis=1)))


def _profit_proxy(work: pd.DataFrame, pred: np.ndarray) -> dict[str, Any]:
    quality = pd.to_numeric(work.get("quality_score", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float64)
    mask = pred > 0
    long_mask = pred == 1
    short_mask = pred == 2
    return {
        "pred_trade_count": int(mask.sum()),
        "pred_long_count": int(long_mask.sum()),
        "pred_short_count": int(short_mask.sum()),
        "pred_trade_quality_mean": float(quality[mask].mean()) if np.any(mask) else 0.0,
        "pred_long_quality_mean": float(quality[long_mask].mean()) if np.any(long_mask) else 0.0,
        "pred_short_quality_mean": float(quality[short_mask].mean()) if np.any(short_mask) else 0.0,
        "pred_trade_quality_sum": float(quality[mask].sum()) if np.any(mask) else 0.0,
    }


def _report(y: np.ndarray, p: np.ndarray, work: pd.DataFrame) -> dict[str, Any]:
    p = _normalize(p)
    pred = p.argmax(axis=1).astype(np.int64)
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "multi_brier": _multi_brier(y, p),
        "ece": _ece(y, p),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(y, pred, labels=[0, 1, 2], output_dict=True, zero_division=0),
        "pred_counts": {str(int(k)): int(v) for k, v in pd.Series(pred).value_counts().sort_index().to_dict().items()},
        "class_counts": {str(int(k)): int(v) for k, v in pd.Series(y).value_counts().sort_index().to_dict().items()},
    }
    out.update(_profit_proxy(work, pred))
    return out


def _regime_matrix(work: pd.DataFrame) -> np.ndarray:
    cols = []
    for col in REGIME_PROB_COLS:
        if col in work.columns:
            cols.append(pd.to_numeric(work[col], errors="coerce").fillna(0.0).to_numpy(np.float64))
        else:
            cols.append(np.zeros(len(work), dtype=np.float64))
    return np.stack(cols, axis=1)


def _fit_dynamic_stacker(data: dict[str, Any], probas: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    val_features = np.concatenate(
        [probas["val_p3"], probas["val_p4"], _regime_matrix(data["work"]["val"])],
        axis=1,
    )
    oos_features = np.concatenate(
        [probas["oos_p3"], probas["oos_p4"], _regime_matrix(data["work"]["oos"])],
        axis=1,
    )
    stacker = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        C=0.35,
        random_state=42,
    )
    stacker.fit(val_features, data["y3"]["val"])
    return {
        "val": stacker.predict_proba(val_features),
        "oos": stacker.predict_proba(oos_features),
        "coef": stacker.coef_,
        "intercept": stacker.intercept_,
    }


def _fit_isotonic_calibrator(y: np.ndarray, p: np.ndarray) -> list[IsotonicRegression]:
    calibrators = []
    for cls in range(3):
        target = (y == cls).astype(np.float64)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p[:, cls], target)
        calibrators.append(iso)
    return calibrators


def _apply_isotonic(calibrators: list[IsotonicRegression], p: np.ndarray) -> np.ndarray:
    cols = [cal.transform(p[:, cls]) for cls, cal in enumerate(calibrators)]
    return _normalize(np.stack(cols, axis=1))


def _mahalanobis_uncertainty(data: dict[str, Any]) -> dict[str, np.ndarray]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(data["x"]["train"].fillna(0.0).to_numpy(np.float64))
    # Diagonal covariance is intentionally used here: stable, fast, and enough for OOD tension.
    var = np.var(x_train, axis=0) + 1e-6
    mean = np.mean(x_train, axis=0)
    out: dict[str, np.ndarray] = {}
    for split in ("val", "oos"):
        x = scaler.transform(data["x"][split].fillna(0.0).to_numpy(np.float64))
        dist = np.sqrt(np.mean(((x - mean) ** 2) / var, axis=1))
        ref = np.percentile(np.sqrt(np.mean(((x_train - mean) ** 2) / var, axis=1)), 99.0)
        out[split] = np.asarray(dist / max(ref, 1e-9), dtype=np.float64)
    return out


def _uncertainty_gate(p: np.ndarray, uncertainty: np.ndarray, threshold: float) -> np.ndarray:
    out = np.asarray(p, dtype=np.float64).copy()
    block = uncertainty >= threshold
    out[block] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return _normalize(out)


def _router3_weight_exp(frame: pd.DataFrame, y: np.ndarray, gamma: float, max_boost: float) -> np.ndarray:
    w = _router3_weight(frame, y)
    quality = np.abs(_num(frame, "quality_score", 0.0))
    base_old = 1.0 + np.clip(quality, 0.0, 1.0)
    exp_new = np.minimum(np.exp(gamma * np.clip(quality, 0.0, 2.0)), max_boost)
    return np.where(y > 0, w / np.maximum(base_old, 1e-9) * exp_new, w)


def _router4_weight_exp(frame: pd.DataFrame, y4: np.ndarray, gamma: float, max_boost: float) -> np.ndarray:
    w = _router4_weight(frame, y4)
    quality = np.abs(_num(frame, "quality_score", 0.0))
    base_old = 1.20 + np.clip(quality, 0.0, 1.0)
    exp_new = np.minimum(np.exp(gamma * np.clip(quality, 0.0, 2.0)), max_boost)
    return np.where(y4 >= 2, w / np.maximum(base_old, 1e-9) * exp_new, w)


def _train_exp_weight_variant(data: dict[str, Any], out_dir: Path, devices: str, seed: int, gamma: float, max_boost: float) -> dict[str, np.ndarray]:
    x_train = data["x"]["train"]
    x_val = data["x"]["val"]
    x_oos = data["x"]["oos"]
    y3_train = data["y3"]["train"]
    y3_val = data["y3"]["val"]
    y4_train = data["y4"]["train"]
    y4_val = data["y4"]["val"]
    w3 = _router3_weight_exp(data["work"]["train"], y3_train, gamma, max_boost)
    w4 = _router4_weight_exp(data["work"]["train"], y4_train, gamma, max_boost)
    model3 = _fit_router(x_train, y3_train, w3, x_val, y3_val, seed=seed, devices=devices)
    model4 = _fit_router(x_train, y4_train, w4, x_val, y4_val, seed=seed, devices=devices)
    out_dir.mkdir(parents=True, exist_ok=True)
    model3_path = out_dir / "router3_exp_weight_catboost_gpu.cbm"
    model4_path = out_dir / "router4_exp_weight_catboost_gpu.cbm"
    model3.save_model(str(model3_path))
    model4.save_model(str(model4_path))

    def comp(split: str) -> tuple[np.ndarray, np.ndarray]:
        p3_raw = np.asarray(model3.predict_proba(data["x"][split]), dtype=np.float64)
        p4_raw = np.asarray(model4.predict_proba(data["x"][split]), dtype=np.float64)
        c3 = [int(c) for c in getattr(model3, "classes_", [0, 1, 2])]
        c4 = [int(c) for c in getattr(model4, "classes_", [0, 1, 2, 3])]
        i3 = {c: i for i, c in enumerate(c3)}
        i4 = {c: i for i, c in enumerate(c4)}
        p3 = np.stack([p3_raw[:, i3[0]], p3_raw[:, i3[1]], p3_raw[:, i3[2]]], axis=1)
        p4 = np.stack([p4_raw[:, i4[0]] + p4_raw[:, i4[1]], p4_raw[:, i4[2]], p4_raw[:, i4[3]]], axis=1)
        return p3, p4

    val_p3, val_p4 = comp("val")
    oos_p3, oos_p4 = comp("oos")
    return {
        "val": _normalize(0.8 * val_p3 + 0.2 * val_p4),
        "oos": _normalize(0.8 * oos_p3 + 0.2 * oos_p4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--base-meta", type=Path, default=DEFAULT_BASE_META)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--devices", default="0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-exp-weight", action="store_true")
    ap.add_argument("--exp-gamma", type=float, default=2.0)
    ap.add_argument("--exp-max-boost", type=float, default=8.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_data(args.data_dir)
    probas = _component_probas(args.base_meta, data["x"])

    fixed = {
        "val": _normalize(0.8 * probas["val_p3"] + 0.2 * probas["val_p4"]),
        "oos": _normalize(0.8 * probas["oos_p3"] + 0.2 * probas["oos_p4"]),
    }
    dyn = _fit_dynamic_stacker(data, probas)
    iso = _fit_isotonic_calibrator(data["y3"]["val"], fixed["val"])
    calibrated = {"val": _apply_isotonic(iso, fixed["val"]), "oos": _apply_isotonic(iso, fixed["oos"])}

    unc = _mahalanobis_uncertainty(data)
    unc_threshold = float(np.quantile(unc["val"], 0.95))
    ood_gated = {
        "val": _uncertainty_gate(fixed["val"], unc["val"], unc_threshold),
        "oos": _uncertainty_gate(fixed["oos"], unc["oos"], unc_threshold),
    }

    variants: dict[str, dict[str, np.ndarray]] = {
        "baseline_fixed_0p8_0p2": fixed,
        "dynamic_logistic_stacking_valfit": {"val": dyn["val"], "oos": dyn["oos"]},
        "isotonic_calibrated_valfit": calibrated,
        "mahalanobis_ood_gate_p95": ood_gated,
    }
    if args.run_exp_weight:
        variants["exp_quality_weight_retrain"] = _train_exp_weight_variant(
            data,
            args.out_dir / "exp_quality_weight_models",
            devices=args.devices,
            seed=args.seed,
            gamma=args.exp_gamma,
            max_boost=args.exp_max_boost,
        )

    summary: dict[str, Any] = {
        "model_id": "alpha5_router_v5_ablation_20260520",
        "base_meta": str(args.base_meta),
        "data_dir": str(args.data_dir),
        "notes": {
            "dynamic_logistic_stacking_valfit": "meta learner is fit on validation probabilities and assessed primarily on OOS",
            "isotonic_calibrated_valfit": "classwise isotonic calibration is fit on validation probabilities and assessed primarily on OOS",
            "mahalanobis_ood_gate_p95": "diagonal Mahalanobis uncertainty; validation 95th percentile gates predictions to NONE",
            "exp_quality_weight_retrain": "replaces clipped trade quality boost with exp(gamma*abs_quality), capped by exp_max_boost",
        },
        "uncertainty": {
            "threshold_val_p95": unc_threshold,
            "val_block_ratio": float(np.mean(unc["val"] >= unc_threshold)),
            "oos_block_ratio": float(np.mean(unc["oos"] >= unc_threshold)),
            "val_mean": float(np.mean(unc["val"])),
            "oos_mean": float(np.mean(unc["oos"])),
        },
        "variants": {},
    }
    for name, vp in variants.items():
        summary["variants"][name] = {
            "val": _report(data["y3"]["val"], vp["val"], data["work"]["val"]),
            "oos": _report(data["y3"]["oos"], vp["oos"], data["work"]["oos"]),
        }

    out_path = args.out_dir / "router5_ablation_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"summary": str(out_path), "variants": list(summary["variants"].keys())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
