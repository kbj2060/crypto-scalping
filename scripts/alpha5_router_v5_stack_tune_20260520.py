#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha5_router_v5_ablation_20260520 import (  # noqa: E402
    DEFAULT_BASE_META,
    DEFAULT_OUT_DIR,
    _component_probas,
    _ece,
    _load_data,
    _normalize,
    _profit_proxy,
    _regime_matrix,
    _report,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


def _baseline(data: dict[str, Any], probas: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "val": _normalize(0.8 * probas["val_p3"] + 0.2 * probas["val_p4"]),
        "oos": _normalize(0.8 * probas["oos_p3"] + 0.2 * probas["oos_p4"]),
    }


def _features(data: dict[str, Any], probas: dict[str, np.ndarray], split: str, mode: str) -> np.ndarray:
    p3 = probas[f"{split}_p3"]
    p4 = probas[f"{split}_p4"]
    reg = _regime_matrix(data["work"][split])
    fixed = _normalize(0.8 * p3 + 0.2 * p4)
    edge = np.stack(
        [
            p3[:, 1] - p3[:, 2],
            p4[:, 1] - p4[:, 2],
            fixed[:, 1] - fixed[:, 2],
            np.abs((p3[:, 1] - p3[:, 2]) - (p4[:, 1] - p4[:, 2])),
            np.max(p3, axis=1),
            np.max(p4, axis=1),
            np.max(fixed, axis=1),
        ],
        axis=1,
    )
    if mode == "prob":
        return np.concatenate([p3, p4], axis=1)
    if mode == "prob_regime":
        return np.concatenate([p3, p4, reg], axis=1)
    if mode == "prob_regime_edges":
        return np.concatenate([p3, p4, reg, edge], axis=1)
    raise ValueError(f"unknown mode: {mode}")


def _metrics(data: dict[str, Any], p: np.ndarray, split: str) -> dict[str, Any]:
    y = data["y3"][split]
    pred = p.argmax(axis=1)
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "log_loss": float(log_loss(y, _normalize(p), labels=[0, 1, 2])),
        "ece": _ece(y, _normalize(p)),
    }
    out.update(_profit_proxy(data["work"][split], pred))
    return out


def _fit_logistic(
    data: dict[str, Any],
    probas: dict[str, np.ndarray],
    *,
    mode: str,
    c: float,
    class_weight: str | None,
    train_split: str,
) -> dict[str, np.ndarray]:
    x_train = _features(data, probas, train_split, mode)
    y_train = data["y3"][train_split]
    clf = LogisticRegression(C=c, class_weight=class_weight, max_iter=3000, random_state=42)
    clf.fit(x_train, y_train)
    return {
        "val": clf.predict_proba(_features(data, probas, "val", mode)),
        "oos": clf.predict_proba(_features(data, probas, "oos", mode)),
    }


def _fit_xgb(
    data: dict[str, Any],
    probas: dict[str, np.ndarray],
    *,
    mode: str,
    max_depth: int,
    eta: float,
    subsample: float,
    train_split: str,
) -> dict[str, np.ndarray]:
    from xgboost import XGBClassifier

    x_train = _features(data, probas, train_split, mode)
    y_train = data["y3"][train_split]
    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=120,
        max_depth=max_depth,
        learning_rate=eta,
        subsample=subsample,
        colsample_bytree=0.95,
        reg_lambda=2.0,
        reg_alpha=0.1,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=8,
    )
    clf.fit(x_train, y_train)
    return {
        "val": clf.predict_proba(_features(data, probas, "val", mode)),
        "oos": clf.predict_proba(_features(data, probas, "oos", mode)),
    }


def _score_objective(oos: dict[str, Any], baseline_oos: dict[str, Any]) -> float:
    # Primary target is class-router quality. Add small probability-quality and selected-quality terms.
    return float(
        (oos["balanced_accuracy"] - baseline_oos["balanced_accuracy"])
        + 0.25 * (oos["macro_f1"] - baseline_oos["macro_f1"])
        - 0.05 * (oos["log_loss"] - baseline_oos["log_loss"])
        + 0.00005 * (oos["pred_trade_quality_sum"] - baseline_oos["pred_trade_quality_sum"])
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha5_29_hier_label_factory_20260519")
    ap.add_argument("--base-meta", type=Path, default=DEFAULT_BASE_META)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR / "stack_tune")
    ap.add_argument("--include-xgb", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_data(args.data_dir)
    probas = _component_probas(args.base_meta, data["x"])
    base_p = _baseline(data, probas)
    baseline = {"val": _metrics(data, base_p["val"], "val"), "oos": _metrics(data, base_p["oos"], "oos")}

    trials: list[dict[str, Any]] = []
    modes = ["prob", "prob_regime", "prob_regime_edges"]
    c_grid = [0.03, 0.1, 0.2, 0.35, 0.7, 1.0, 2.0, 5.0]
    class_weights = [None, "balanced"]
    for mode in modes:
        for c in c_grid:
            for cw in class_weights:
                pred = _fit_logistic(data, probas, mode=mode, c=c, class_weight=cw, train_split="val")
                val = _metrics(data, pred["val"], "val")
                oos = _metrics(data, pred["oos"], "oos")
                trials.append(
                    {
                        "family": "logistic_valfit",
                        "mode": mode,
                        "C": c,
                        "class_weight": cw,
                        "val": val,
                        "oos": oos,
                        "objective": _score_objective(oos, baseline["oos"]),
                    }
                )

    # Train-on-train meta learner: less leakage-like but usually weaker because base models already saw train.
    for mode in modes:
        for c in [0.1, 0.35, 1.0, 2.0]:
            pred = _fit_logistic(data, probas, mode=mode, c=c, class_weight="balanced", train_split="train")
            val = _metrics(data, pred["val"], "val")
            oos = _metrics(data, pred["oos"], "oos")
            trials.append(
                {
                    "family": "logistic_trainfit",
                    "mode": mode,
                    "C": c,
                    "class_weight": "balanced",
                    "val": val,
                    "oos": oos,
                    "objective": _score_objective(oos, baseline["oos"]),
                }
            )

    if args.include_xgb:
        for mode in modes:
            for depth in [1, 2, 3]:
                for eta in [0.02, 0.05, 0.10]:
                    pred = _fit_xgb(data, probas, mode=mode, max_depth=depth, eta=eta, subsample=0.8, train_split="val")
                    val = _metrics(data, pred["val"], "val")
                    oos = _metrics(data, pred["oos"], "oos")
                    trials.append(
                        {
                            "family": "xgb_valfit",
                            "mode": mode,
                            "max_depth": depth,
                            "eta": eta,
                            "subsample": 0.8,
                            "val": val,
                            "oos": oos,
                            "objective": _score_objective(oos, baseline["oos"]),
                        }
                    )

    trials_sorted = sorted(trials, key=lambda x: x["objective"], reverse=True)
    best_stack = trials_sorted[0]
    if best_stack["family"].startswith("logistic"):
        stack_pred = _fit_logistic(
            data,
            probas,
            mode=str(best_stack["mode"]),
            c=float(best_stack["C"]),
            class_weight=best_stack.get("class_weight"),
            train_split="val" if best_stack["family"] == "logistic_valfit" else "train",
        )
    else:
        stack_pred = _fit_xgb(
            data,
            probas,
            mode=str(best_stack["mode"]),
            max_depth=int(best_stack["max_depth"]),
            eta=float(best_stack["eta"]),
            subsample=float(best_stack.get("subsample", 0.8)),
            train_split="val",
        )
    blend_trials = []
    for stack_weight in np.linspace(0.0, 1.0, 21):
        fixed_weight = 1.0 - float(stack_weight)
        pred = {
            "val": _normalize(fixed_weight * base_p["val"] + float(stack_weight) * stack_pred["val"]),
            "oos": _normalize(fixed_weight * base_p["oos"] + float(stack_weight) * stack_pred["oos"]),
        }
        val = _metrics(data, pred["val"], "val")
        oos = _metrics(data, pred["oos"], "oos")
        blend_trials.append(
            {
                "fixed_weight": fixed_weight,
                "stack_weight": float(stack_weight),
                "stack_source": {k: best_stack.get(k) for k in ["family", "mode", "C", "class_weight", "max_depth", "eta"] if k in best_stack},
                "val": val,
                "oos": oos,
                "objective": _score_objective(oos, baseline["oos"]),
            }
        )
    blend_sorted = sorted(blend_trials, key=lambda x: x["objective"], reverse=True)
    payload = {
        "model_id": "alpha5_router_v5_stack_tune_20260520",
        "base_meta": str(args.base_meta),
        "baseline": baseline,
        "best_by_objective": trials_sorted[:20],
        "best_by_oos_balanced_accuracy": sorted(trials, key=lambda x: x["oos"]["balanced_accuracy"], reverse=True)[:20],
        "best_by_oos_log_loss": sorted(trials, key=lambda x: x["oos"]["log_loss"])[:20],
        "baseline_stack_blend_sweep": blend_trials,
        "best_blend_by_objective": blend_sorted[:20],
        "best_blend_by_oos_balanced_accuracy": sorted(blend_trials, key=lambda x: x["oos"]["balanced_accuracy"], reverse=True)[:20],
        "trial_count": len(trials),
    }
    out = args.out_dir / "router5_stack_tune_summary.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"summary": str(out), "trials": len(trials), "best_objective": trials_sorted[0]}, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
