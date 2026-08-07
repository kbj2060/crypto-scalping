#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

import train_omega1_direction_head_direction_only_20260602 as base
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_regime3_soft_expert_direction_head_volpca_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_regime3_soft_expert_direction_head_volpca_20260602"


def _json_default(obj: Any) -> Any:
    return base._json_default(obj)


def _fit_weighted_catboost(x: pd.DataFrame, y: np.ndarray, weight: np.ndarray, *, seed: int, iterations: int) -> CatBoostClassifier:
    class_weight = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)
    sample_weight = class_weight * np.asarray(weight, dtype=np.float64)
    if not np.isfinite(sample_weight).all() or float(sample_weight.sum()) <= 0.0:
        raise RuntimeError("invalid expert sample weights")
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
    model.fit(x, y, sample_weight=sample_weight)
    return model


def _fit_soft_expert_models(
    x: pd.DataFrame,
    y: np.ndarray,
    route_probs: np.ndarray,
    *,
    seed: int,
    iterations: int,
    model_dir: Path,
    floor: float,
) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        weight = float(floor) + route_probs[:, idx]
        model = _fit_weighted_catboost(x, y, weight, seed=seed + idx, iterations=iterations)
        model_path = model_dir / f"{expert}_soft_direction_head.cbm"
        model.save_model(str(model_path))
        models[expert] = model
        summaries[expert] = {
            "rows": int(len(y)),
            "effective_weight_sum": float(weight.sum()),
            "weight_mean": float(weight.mean()),
            "weight_q90": float(np.quantile(weight, 0.90)),
            "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y, minlength=3))},
            "model": str(model_path),
        }
    return {"models": models, "summaries": summaries}


def _oof_soft(train: pd.DataFrame, *, floor: float) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    n = len(train)
    starts = [int(n * 0.35), int(n * 0.50), int(n * 0.65), int(n * 0.80)]
    ends = [int(n * 0.50), int(n * 0.65), int(n * 0.80), n]
    y = train["zigzag_action"].to_numpy(dtype=np.int64)
    proba = np.full((n, 3), np.nan, dtype=np.float64)
    covered = np.zeros(n, dtype=bool)
    output_parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    pca_folds: list[dict[str, Any]] = []
    for fold, (start, end) in enumerate(zip(starts, ends), start=1):
        fit_frame = train.iloc[:start].reset_index(drop=True)
        pred_frame = train.iloc[start:end].reset_index(drop=True)
        transformer = hard.volpca.VolPca(6).fit(fit_frame)
        x_fit = hard._features_with_transform(fit_frame, transformer)
        x_pred = hard._features_with_transform(pred_frame, transformer)
        route_probs_fit = fit_frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64)
        route_pred = hard._route_id(pred_frame)
        bundle = _fit_soft_expert_models(
            x_fit,
            y[:start],
            route_probs_fit,
            seed=20260602 + fold * 100,
            iterations=500,
            model_dir=OUT_DIR / "oof_folds" / f"floor_{floor:.2f}" / f"fold_{fold}",
            floor=floor,
        )
        expert_pred = hard._predict_all_experts(bundle["models"], x_pred)
        routed = hard._routed_proba(expert_pred, route_pred)
        proba[start:end] = routed
        covered[start:end] = True
        output_parts.append(hard._outputs(pred_frame, expert_pred, routed, prefix=f"omega1_regime3_soft_expert_dir_oof_f{floor:.2f}".replace(".", "p")))
        folds.append(
            {
                "fold": fold,
                "train_rows": int(start),
                "predict_start": int(start),
                "predict_end": int(end),
                "expert_summaries": bundle["summaries"],
                "metrics": base._metrics(y[start:end], routed),
            }
        )
        pca_folds.append({"fold": fold, "explained_variance": transformer.explained_variance})
    return proba, covered, pd.concat(output_parts, ignore_index=True), folds, pca_folds


def _run_floor(train: pd.DataFrame, oos: pd.DataFrame, *, floor: float, y_train: np.ndarray, y_oos: np.ndarray) -> dict[str, Any]:
    variant = f"soft_floor_{floor:.2f}".replace(".", "p")
    variant_dir = OUT_DIR / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    oof_proba, covered, oof_out, folds, pca_folds = _oof_soft(train, floor=floor)
    oof_metrics = base._metrics(y_train[covered], oof_proba[covered])

    final_transformer = hard.volpca.VolPca(6).fit(train)
    x_train = hard._features_with_transform(train, final_transformer)
    x_oos = hard._features_with_transform(oos, final_transformer)
    train_probs = train[hard.ROUTE_COLS].to_numpy(dtype=np.float64)
    oos_route = hard._route_id(oos)
    final_bundle = _fit_soft_expert_models(
        x_train,
        y_train,
        train_probs,
        seed=20260602 + int(floor * 1000),
        iterations=800,
        model_dir=variant_dir / "final_experts",
        floor=floor,
    )
    oos_expert_pred = hard._predict_all_experts(final_bundle["models"], x_oos)
    oos_routed = hard._routed_proba(oos_expert_pred, oos_route)
    oos_metrics = base._metrics(y_oos, oos_routed)
    oof_path = variant_dir / f"training_features_2025_{variant}_regime3_soft_expert_direction_volpca_oof_20260602.csv"
    oos_path = variant_dir / f"training_features_2026_rebuilt_{variant}_regime3_soft_expert_direction_volpca_20260602.csv"
    hard._outputs(oos, oos_expert_pred, oos_routed, prefix="omega1_regime3_soft_expert_dir").to_csv(oos_path, index=False)
    oof_out.to_csv(oof_path, index=False)
    contract_path = variant_dir / f"{variant}_regime3_soft_expert_direction_volpca_contract.joblib"
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "variant": variant,
            "floor": float(floor),
            "label_source": "zigzag_action",
            "route_cols": hard.ROUTE_COLS,
            "expert_names": hard.EXPERT_NAMES,
            "feature_cols": list(x_train.columns),
            "pca_transformer": final_transformer,
            "expert_model_paths": {k: v["model"] for k, v in final_bundle["summaries"].items()},
        },
        contract_path,
    )
    delta = {
        "oos_bacc": float(oos_metrics["balanced_accuracy"] - hard.BASELINE_VOLPCA06["oos_bacc"]),
        "oos_auc": None if oos_metrics["ovr_auc"] is None else float(oos_metrics["ovr_auc"] - hard.BASELINE_VOLPCA06["oos_auc"]),
        "oos_proxy_wr": None if oos_metrics["proxy_wr"] is None else float(oos_metrics["proxy_wr"] - hard.BASELINE_VOLPCA06["oos_proxy_wr"]),
        "oos_proxy_trades": int(oos_metrics["proxy_trades"] - hard.BASELINE_VOLPCA06["oos_proxy_trades"]),
    }
    return {
        "variant": variant,
        "floor": float(floor),
        "feature_count": int(x_train.shape[1]),
        "oof_metrics": oof_metrics,
        "oos_metrics": oos_metrics,
        "delta_vs_global_volatility_pca06": delta,
        "expert_summaries": final_bundle["summaries"],
        "folds": folds,
        "pca_folds": pca_folds,
        "final_pca_explained_variance": final_transformer.explained_variance,
        "artifacts": {
            "oof_2025": str(oof_path),
            "oos_2026": str(oos_path),
            "contract": str(contract_path),
            "variant_dir": str(variant_dir),
        },
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = hard._build_frame(2025)
    oos = hard._build_frame(2026)
    y_train = train["zigzag_action"].to_numpy(dtype=np.int64)
    y_oos = oos["zigzag_action"].to_numpy(dtype=np.int64)
    rows = []
    variants = {}
    for floor in [0.00, 0.05, 0.10, 0.20]:
        payload = _run_floor(train, oos, floor=floor, y_train=y_train, y_oos=y_oos)
        variants[payload["variant"]] = payload
        delta = payload["delta_vs_global_volatility_pca06"]
        rows.append(
            {
                "variant": payload["variant"],
                "floor": payload["floor"],
                "feature_count": payload["feature_count"],
                "oof_bacc": payload["oof_metrics"]["balanced_accuracy"],
                "oof_auc": payload["oof_metrics"]["ovr_auc"],
                "oof_proxy_wr": payload["oof_metrics"]["proxy_wr"],
                "oos_bacc": payload["oos_metrics"]["balanced_accuracy"],
                "oos_auc": payload["oos_metrics"]["ovr_auc"],
                "oos_proxy_wr": payload["oos_metrics"]["proxy_wr"],
                "oos_proxy_trades": payload["oos_metrics"]["proxy_trades"],
                "delta_oos_bacc_vs_global_volpca06": delta["oos_bacc"],
                "delta_oos_auc_vs_global_volpca06": delta["oos_auc"],
                "delta_oos_proxy_wr_vs_global_volpca06": delta["oos_proxy_wr"],
                "delta_oos_trades_vs_global_volpca06": delta["oos_proxy_trades"],
            }
        )
    rows.sort(key=lambda r: (float(r["oos_bacc"]), float(r["oos_auc"] or 0.0)), reverse=True)
    report = {
        "model_id": MODEL_ID,
        "design": "Regime3 current router selects bull/bear/chop. Each expert owns a separate CatBoost Direction Head, trained on all 2025 rows with regime-probability sample weights instead of hard row partitioning.",
        "baseline": hard.BASELINE_VOLPCA06,
        "ranking": rows,
        "selected_by_oos_bacc": rows[0]["variant"],
        "variants": variants,
        "artifacts": {"out_dir": str(OUT_DIR), "ranking": str(OUT_DIR / "ranking.csv"), "report": str(OUT_DIR / "report.json")},
    }
    pd.DataFrame(rows).to_csv(OUT_DIR / "ranking.csv", index=False)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "ranking": rows}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
