#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as base  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_cash_fallback_model_zoo_20260606"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASE_RISK = sleeve.FallbackRisk("base_tp026_sl014_n0405_h192", 0.026, 0.014, 0.405, 2.0, 192)
AGGRESSIVE_VAL = sleeve.AGGRESSIVE_VAL
AGGRESSIVE_OOS = sleeve.AGGRESSIVE_OOS


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


def _make_zoo_model(name: str, seed: int):
    if name == "tree":
        return DecisionTreeClassifier(max_depth=5, min_samples_leaf=80, class_weight="balanced", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=220, max_depth=6, min_samples_leaf=35, class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "gb":
        return GradientBoostingClassifier(n_estimators=90, learning_rate=0.035, max_depth=2, min_samples_leaf=50, random_state=seed)
    if name == "logreg":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.15, class_weight="balanced", max_iter=1000, random_state=seed)),
            ]
        )
    if name == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", MLPClassifier(hidden_layer_sizes=(32,), alpha=0.02, learning_rate_init=0.001, max_iter=220, early_stopping=True, random_state=seed)),
            ]
        )
    if name == "svc_rbf":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(C=0.8, gamma="scale", class_weight="balanced", probability=True, random_state=seed)),
            ]
        )
    raise RuntimeError(f"unknown zoo model: {name}")


def _predict_oof(model_name: str, x: pd.DataFrame, y: np.ndarray, cash_mask: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    idx = np.flatnonzero(cash_mask)
    action = np.zeros(len(x), dtype=np.int64)
    conf = np.zeros(len(x), dtype=np.float64)
    folds = []
    n = len(idx)
    for train_frac, end_frac in ((0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.00)):
        train_end = int(n * train_frac)
        val_end = int(n * end_frac)
        if train_end < 100 or val_end <= train_end:
            continue
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        model = _make_zoo_model(model_name, seed + train_end)
        model.fit(x.iloc[train_idx].to_numpy(dtype=np.float64), y[train_idx])
        proba = model.predict_proba(x.iloc[val_idx].to_numpy(dtype=np.float64))
        classes = np.asarray(model.classes_, dtype=np.int64)
        best = np.argmax(proba, axis=1)
        action[val_idx] = classes[best]
        conf[val_idx] = proba[np.arange(len(val_idx)), best]
        folds.append({"train_rows": int(len(train_idx)), "val_rows": int(len(val_idx))})
    return action, conf, {"folds": folds, "oof_rows": int(np.count_nonzero(conf > 0.0))}


def _fit_predict(model_name: str, x_train: pd.DataFrame, y_train: np.ndarray, train_cash_mask: np.ndarray, x_eval: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(train_cash_mask)
    model = _make_zoo_model(model_name, seed)
    model.fit(x_train.iloc[idx].to_numpy(dtype=np.float64), y_train[idx])
    proba = model.predict_proba(x_eval.to_numpy(dtype=np.float64))
    classes = np.asarray(model.classes_, dtype=np.int64)
    best = np.argmax(proba, axis=1)
    return classes[best].astype(np.int64), proba[np.arange(len(x_eval)), best].astype(np.float64)


def _metric_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return sleeve._metric_row(prefix, metrics)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_src, val_dec0, val_prefix = base._build_split(frames, "validation")
    oos_frame, oos_src, oos_dec0, oos_prefix = base._build_split(frames, "oos")
    val_dec = sleeve._apply_aggressive(val_dec0)
    oos_dec = sleeve._apply_aggressive(oos_dec0)
    val_features = sleeve._extra_features(base._feature_frame(val_frame, val_src, val_dec0, val_prefix), val_dec)
    oos_features = sleeve._extra_features(base._feature_frame(oos_frame, oos_src, oos_dec0, oos_prefix), oos_dec)
    val_cash = ~omega._active(val_dec)
    y_val, label_diag = sleeve._build_labels(val_frame, val_dec, BASE_RISK, 0.006)
    rows: list[dict[str, Any]] = []
    baseline_val = omega._metrics(val_frame, val_dec, fee=fee, slip=slip, cost_mult=3.0)
    baseline_oos = omega._metrics(oos_frame, oos_dec, fee=fee, slip=slip, cost_mult=3.0)
    rows.append(
        {
            "model": "aggressive_primary_only",
            "threshold": 1.0,
            **_metric_row("val", {**baseline_val, "primary_entries": baseline_val["long_entries"] + baseline_val["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}),
            **_metric_row("oos", {**baseline_oos, "primary_entries": baseline_oos["long_entries"] + baseline_oos["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}),
        }
    )
    diagnostics: dict[str, Any] = {"risk": BASE_RISK.__dict__, "min_edge": 0.006, "label_diag": label_diag, "feature_count": int(val_features.shape[1]), "features": list(val_features.columns)}
    for model_name in ("tree", "rf", "gb", "logreg", "mlp", "svc_rbf"):
        val_action, val_conf, oof_diag = _predict_oof(model_name, val_features, y_val, val_cash, seed=260606)
        oos_action, oos_conf = _fit_predict(model_name, val_features, y_val, val_cash, oos_features, seed=260606)
        diagnostics[f"{model_name}_oof"] = oof_diag
        for threshold in (0.45, 0.55, 0.65, 0.75, 0.85, 0.90):
            val_m = sleeve._metrics_with_fallback(val_frame, val_dec, BASE_RISK, val_action, val_conf, threshold, fee=fee, slip=slip, cost_mult=3.0)
            oos_m = sleeve._metrics_with_fallback(oos_frame, oos_dec, BASE_RISK, oos_action, oos_conf, threshold, fee=fee, slip=slip, cost_mult=3.0)
            row = {"model": model_name, "threshold": float(threshold)}
            row.update(_metric_row("val", val_m))
            row.update(_metric_row("oos", oos_m))
            rows.append(row)
    ranking = pd.DataFrame(rows)
    ranking["val_delta_pnl"] = ranking["val_pnl"] - AGGRESSIVE_VAL["pnl"]
    ranking["oos_delta_pnl"] = ranking["oos_pnl"] - AGGRESSIVE_OOS["pnl"]
    ranking["val_delta_mdd"] = ranking["val_mdd"] - AGGRESSIVE_VAL["mdd"]
    ranking["oos_delta_mdd"] = ranking["oos_mdd"] - AGGRESSIVE_OOS["mdd"]
    ranking["score"] = ranking["oos_pnl"] + 0.75 * ranking["val_pnl"] + 0.35 * ranking["oos_mdd"] + 0.35 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl", "score"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "cash_fallback_model_zoo_ranking.csv", index=False)
    promotable = ranking[
        (ranking["model"] != "aggressive_primary_only")
        & (ranking["oos_pnl"] > 77.75020153310189)
        & (ranking["val_pnl"] > 101.92538518551784)
        & (ranking["oos_mdd"] >= -8.108170708968387 * 1.35)
        & (ranking["val_mdd"] >= -10.677652697162888 * 1.35)
    ].copy()
    promotable.to_csv(OUT_DIR / "cash_fallback_model_zoo_promotable.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "baseline": "omega1_2_1_cash_fallback_extra_base_edge006_thr055_20260606",
        "method": "Model zoo comparison on the selected cash fallback sleeve risk/label setup. Primary aggressive baseline is unchanged.",
        "diagnostics": diagnostics,
        "best": ranking.iloc[0].to_dict(),
        "promotable_count": int(len(promotable)),
        "top10": ranking.head(10).to_dict(orient="records"),
        "artifacts": {"out_dir": str(OUT_DIR), "ranking": str(OUT_DIR / "cash_fallback_model_zoo_ranking.csv"), "promotable": str(OUT_DIR / "cash_fallback_model_zoo_promotable.csv"), "report": str(OUT_DIR / "report.json")},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "best": report["best"], "promotable_count": int(len(promotable))}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
