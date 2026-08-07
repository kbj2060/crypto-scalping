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
MODEL_ID = "omega1_regime3_routed_expert_direction_quality_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_regime3_routed_expert_direction_quality_20260602"

QUALITY_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
SOFT_FLOORS = [0.00, 0.05, 0.10, 0.20]
DIR_AUX_COLS = [
    "direction_p_cash",
    "direction_p_long",
    "direction_p_short",
    "direction_confidence",
    "direction_side_edge",
    "direction_trade_prob",
    "direction_action",
]
ROUTE_QUALITY_COLS = [
    "router_confidence",
    "router_margin",
]


def _json_default(obj: Any) -> Any:
    return base._json_default(obj)


def _fit_multiclass(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int,
    iterations: int,
    sample_weight: np.ndarray | None = None,
) -> CatBoostClassifier:
    weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)
    if sample_weight is not None:
        weights = weights * np.asarray(sample_weight, dtype=np.float64)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise RuntimeError("invalid sample weights")
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


def _route_probs(frame: pd.DataFrame) -> np.ndarray:
    values = frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite Regime3 route probabilities")
    return values


def _quality_base_features(x: pd.DataFrame, frame: pd.DataFrame, dir_proba: np.ndarray) -> pd.DataFrame:
    action = np.argmax(dir_proba, axis=1).astype(np.int64)
    out = x.reset_index(drop=True).copy()
    out["direction_p_cash"] = dir_proba[:, 0]
    out["direction_p_long"] = dir_proba[:, 1]
    out["direction_p_short"] = dir_proba[:, 2]
    out["direction_confidence"] = np.max(dir_proba, axis=1)
    out["direction_side_edge"] = dir_proba[:, 1] - dir_proba[:, 2]
    out["direction_trade_prob"] = dir_proba[:, 1] + dir_proba[:, 2]
    out["direction_action"] = action.astype(np.float64)
    out["router_confidence"] = hard._route_conf(frame)
    out["router_margin"] = pd.to_numeric(frame["regime3_current_sensitive_wide24_margin"], errors="raise").to_numpy(dtype=np.float64)
    return out


def _fit_direction_models(
    x: pd.DataFrame,
    y: np.ndarray,
    frame: pd.DataFrame,
    *,
    mode: str,
    floor: float,
    seed: int,
    iterations: int,
    model_dir: Path,
) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, CatBoostClassifier] = {}
    summaries: dict[str, Any] = {}
    route = hard._route_id(frame)
    probs = _route_probs(frame)
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        if mode == "hard":
            mask = route == idx
            if int(mask.sum()) < 1000:
                raise RuntimeError(f"{expert}: too few hard-routed direction rows: {int(mask.sum())}")
            x_fit = x.loc[mask].reset_index(drop=True)
            y_fit = y[mask]
            sample_weight = None
            effective_rows = int(mask.sum())
            weight_sum = None
        elif mode == "soft":
            x_fit = x
            y_fit = y
            sample_weight = float(floor) + probs[:, idx]
            effective_rows = int(len(y))
            weight_sum = float(np.asarray(sample_weight, dtype=np.float64).sum())
        else:
            raise ValueError(f"unknown mode: {mode}")
        classes = sorted(np.unique(y_fit).astype(int).tolist())
        if classes != [0, 1, 2]:
            raise RuntimeError(f"{mode}/{expert}: missing zigzag_action classes: {classes}")
        model = _fit_multiclass(x_fit, y_fit, seed=seed + idx, iterations=iterations, sample_weight=sample_weight)
        model_path = model_dir / f"{expert}_direction_head.cbm"
        model.save_model(str(model_path))
        models[expert] = model
        summaries[expert] = {
            "rows": effective_rows,
            "weight_sum": weight_sum,
            "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y_fit, minlength=3))},
            "model": str(model_path),
        }
    return {"models": models, "summaries": summaries}


def _fit_quality_models(
    xq: pd.DataFrame,
    y: np.ndarray,
    frame: pd.DataFrame,
    *,
    mode: str,
    floor: float,
    seed: int,
    iterations: int,
    model_dir: Path,
) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, CatBoostClassifier] = {}
    summaries: dict[str, Any] = {}
    route = hard._route_id(frame)
    probs = _route_probs(frame)
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        if mode == "hard":
            mask = route == idx
            if int(mask.sum()) < 1000:
                raise RuntimeError(f"{expert}: too few hard-routed quality rows: {int(mask.sum())}")
            x_fit = xq.loc[mask].reset_index(drop=True)
            y_fit = y[mask]
            sample_weight = None
            effective_rows = int(mask.sum())
            weight_sum = None
        elif mode == "soft":
            x_fit = xq
            y_fit = y
            sample_weight = float(floor) + probs[:, idx]
            effective_rows = int(len(y))
            weight_sum = float(np.asarray(sample_weight, dtype=np.float64).sum())
        else:
            raise ValueError(f"unknown mode: {mode}")
        classes = sorted(np.unique(y_fit).astype(int).tolist())
        if classes != [0, 1, 2]:
            raise RuntimeError(f"{mode}/{expert}: missing quality zigzag_action classes: {classes}")
        model = _fit_multiclass(x_fit, y_fit, seed=seed + idx, iterations=iterations, sample_weight=sample_weight)
        model_path = model_dir / f"{expert}_quality_head.cbm"
        model.save_model(str(model_path))
        models[expert] = model
        summaries[expert] = {
            "rows": effective_rows,
            "weight_sum": weight_sum,
            "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y_fit, minlength=3))},
            "model": str(model_path),
        }
    return {"models": models, "summaries": summaries}


def _predict_expert_models(models: dict[str, CatBoostClassifier], x: pd.DataFrame) -> dict[str, np.ndarray]:
    return {expert: base._proba3(model, x) for expert, model in models.items()}


def _routed_proba(expert_proba: dict[str, np.ndarray], route: np.ndarray) -> np.ndarray:
    out = np.zeros((len(route), 3), dtype=np.float64)
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        mask = route == idx
        out[mask] = expert_proba[expert][mask]
    return out


def _quality_for_action(quality_proba: np.ndarray, direction_action: np.ndarray) -> np.ndarray:
    idx = np.arange(len(direction_action))
    return quality_proba[idx, direction_action.astype(np.int64)]


def _apply_quality_filter(direction_proba: np.ndarray, quality_proba: np.ndarray, threshold: float) -> np.ndarray:
    action = np.argmax(direction_proba, axis=1).astype(np.int64)
    q = _quality_for_action(quality_proba, action)
    final_action = action.copy()
    final_action[(action != 0) & (q < float(threshold))] = 0
    out = direction_proba.copy()
    veto = final_action == 0
    out[veto] = 0.0
    out[veto, 0] = 1.0
    keep = ~veto
    if keep.any():
        out[keep] = direction_proba[keep]
    return out


def _prediction_output(
    frame: pd.DataFrame,
    direction_proba: np.ndarray,
    quality_proba: np.ndarray,
    *,
    threshold: float,
    prefix: str,
) -> pd.DataFrame:
    route = hard._route_id(frame)
    direction_action = np.argmax(direction_proba, axis=1).astype(np.int64)
    quality_for_action = _quality_for_action(quality_proba, direction_action)
    final_action = direction_action.copy()
    final_action[(direction_action != 0) & (quality_for_action < float(threshold))] = 0
    return pd.DataFrame(
        {
            "timestamp": frame["timestamp"].to_numpy(),
            f"{prefix}_router_expert": np.asarray(hard.EXPERT_NAMES, dtype=object)[route],
            f"{prefix}_router_confidence": hard._route_conf(frame),
            f"{prefix}_router_margin": pd.to_numeric(frame["regime3_current_sensitive_wide24_margin"], errors="raise").to_numpy(dtype=np.float64),
            f"{prefix}_dir_p_cash": direction_proba[:, 0],
            f"{prefix}_dir_p_long": direction_proba[:, 1],
            f"{prefix}_dir_p_short": direction_proba[:, 2],
            f"{prefix}_dir_confidence": np.max(direction_proba, axis=1),
            f"{prefix}_dir_side_edge": direction_proba[:, 1] - direction_proba[:, 2],
            f"{prefix}_dir_trade_prob": direction_proba[:, 1] + direction_proba[:, 2],
            f"{prefix}_dir_action": direction_action,
            f"{prefix}_quality_p_cash": quality_proba[:, 0],
            f"{prefix}_quality_p_long": quality_proba[:, 1],
            f"{prefix}_quality_p_short": quality_proba[:, 2],
            f"{prefix}_quality_for_action": quality_for_action,
            f"{prefix}_quality_threshold": float(threshold),
            f"{prefix}_final_action": final_action,
        }
    )


def _oof_direction(
    train: pd.DataFrame,
    *,
    mode: str,
    floor: float,
    variant_dir: Path,
) -> dict[str, Any]:
    n = len(train)
    starts = [int(n * 0.35), int(n * 0.50), int(n * 0.65), int(n * 0.80)]
    ends = [int(n * 0.50), int(n * 0.65), int(n * 0.80), n]
    y = train["zigzag_action"].to_numpy(dtype=np.int64)
    direction_proba = np.full((n, 3), np.nan, dtype=np.float64)
    covered = np.zeros(n, dtype=bool)
    x_parts: list[pd.DataFrame] = []
    frame_parts: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    pca_reports: list[dict[str, Any]] = []
    for fold, (start, end) in enumerate(zip(starts, ends), start=1):
        fit_frame = train.iloc[:start].reset_index(drop=True)
        pred_frame = train.iloc[start:end].reset_index(drop=True)
        transformer = hard.volpca.VolPca(6).fit(fit_frame)
        x_fit = hard._features_with_transform(fit_frame, transformer)
        x_pred = hard._features_with_transform(pred_frame, transformer)
        bundle = _fit_direction_models(
            x_fit,
            y[:start],
            fit_frame,
            mode=mode,
            floor=floor,
            seed=20260602 + fold * 100,
            iterations=500,
            model_dir=variant_dir / "oof_direction" / f"fold_{fold}",
        )
        expert_pred = _predict_expert_models(bundle["models"], x_pred)
        routed = _routed_proba(expert_pred, hard._route_id(pred_frame))
        direction_proba[start:end] = routed
        covered[start:end] = True
        x_parts.append(x_pred)
        frame_parts.append(pred_frame)
        fold_reports.append(
            {
                "fold": fold,
                "train_rows": int(start),
                "predict_start": int(start),
                "predict_end": int(end),
                "direction_expert_summaries": bundle["summaries"],
                "metrics": base._metrics(y[start:end], routed),
            }
        )
        pca_reports.append({"fold": fold, "explained_variance": transformer.explained_variance})
    return {
        "direction_proba": direction_proba,
        "covered": covered,
        "x_oof": pd.concat(x_parts, ignore_index=True),
        "frame_oof": pd.concat(frame_parts, ignore_index=True),
        "folds": fold_reports,
        "pca_folds": pca_reports,
    }


def _train_final_direction(
    train: pd.DataFrame,
    oos: pd.DataFrame,
    *,
    mode: str,
    floor: float,
    variant_dir: Path,
) -> dict[str, Any]:
    y_train = train["zigzag_action"].to_numpy(dtype=np.int64)
    transformer = hard.volpca.VolPca(6).fit(train)
    x_train = hard._features_with_transform(train, transformer)
    x_oos = hard._features_with_transform(oos, transformer)
    bundle = _fit_direction_models(
        x_train,
        y_train,
        train,
        mode=mode,
        floor=floor,
        seed=20260602,
        iterations=800,
        model_dir=variant_dir / "final_direction",
    )
    expert_pred = _predict_expert_models(bundle["models"], x_oos)
    routed = _routed_proba(expert_pred, hard._route_id(oos))
    return {
        "transformer": transformer,
        "x_train": x_train,
        "x_oos": x_oos,
        "direction_models": bundle["models"],
        "direction_summaries": bundle["summaries"],
        "oos_direction_proba": routed,
        "oos_expert_direction_proba": expert_pred,
    }


def _train_quality_from_oof(
    oof: dict[str, Any],
    train_y: np.ndarray,
    *,
    mode: str,
    floor: float,
    variant_dir: Path,
) -> dict[str, Any]:
    covered = oof["covered"]
    frame_oof = oof["frame_oof"].reset_index(drop=True)
    direction_oof = oof["direction_proba"][covered]
    xq = _quality_base_features(oof["x_oof"], frame_oof, direction_oof)
    yq = train_y[covered]
    bundle = _fit_quality_models(
        xq,
        yq,
        frame_oof,
        mode=mode,
        floor=floor,
        seed=20260603,
        iterations=600,
        model_dir=variant_dir / "quality",
    )
    expert_quality = _predict_expert_models(bundle["models"], xq)
    quality_oof = _routed_proba(expert_quality, hard._route_id(frame_oof))
    return {
        "x_quality_oof": xq,
        "quality_models": bundle["models"],
        "quality_summaries": bundle["summaries"],
        "quality_oof_proba": quality_oof,
    }


def _select_threshold(y: np.ndarray, direction_proba: np.ndarray, quality_proba: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    direction_metrics = base._metrics(y, direction_proba)
    min_trades = max(1, int(direction_metrics["proxy_trades"] * 0.70))
    rows: list[dict[str, Any]] = []
    for threshold in QUALITY_THRESHOLDS:
        filtered = _apply_quality_filter(direction_proba, quality_proba, threshold)
        metrics = base._metrics(y, filtered)
        rows.append({"threshold": float(threshold), "metrics": metrics, "direction_only_proxy_trades": direction_metrics["proxy_trades"], "min_trades": int(min_trades)})
    eligible = [r for r in rows if int(r["metrics"]["proxy_trades"]) >= min_trades]
    if not eligible:
        eligible = rows
    eligible.sort(
        key=lambda r: (
            float(r["metrics"]["balanced_accuracy"]),
            float(r["metrics"]["proxy_wr"] or 0.0),
            int(r["metrics"]["proxy_trades"]),
        ),
        reverse=True,
    )
    return float(eligible[0]["threshold"]), rows


def _evaluate_variant(train: pd.DataFrame, oos: pd.DataFrame, *, mode: str, floor: float) -> dict[str, Any]:
    variant = f"{mode}_floor_{floor:.2f}".replace(".", "p")
    variant_dir = OUT_DIR / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    y_train = train["zigzag_action"].to_numpy(dtype=np.int64)
    y_oos = oos["zigzag_action"].to_numpy(dtype=np.int64)
    oof = _oof_direction(train, mode=mode, floor=floor, variant_dir=variant_dir)
    direction_oof = oof["direction_proba"][oof["covered"]]
    direction_oof_metrics = base._metrics(y_train[oof["covered"]], direction_oof)
    quality = _train_quality_from_oof(oof, y_train, mode=mode, floor=floor, variant_dir=variant_dir)
    selected_threshold, threshold_rows = _select_threshold(y_train[oof["covered"]], direction_oof, quality["quality_oof_proba"])
    filtered_oof = _apply_quality_filter(direction_oof, quality["quality_oof_proba"], selected_threshold)
    filtered_oof_metrics = base._metrics(y_train[oof["covered"]], filtered_oof)

    final_direction = _train_final_direction(train, oos, mode=mode, floor=floor, variant_dir=variant_dir)
    xq_oos = _quality_base_features(final_direction["x_oos"], oos, final_direction["oos_direction_proba"])
    expert_quality_oos = _predict_expert_models(quality["quality_models"], xq_oos)
    quality_oos = _routed_proba(expert_quality_oos, hard._route_id(oos))
    direction_oos_metrics = base._metrics(y_oos, final_direction["oos_direction_proba"])
    filtered_oos = _apply_quality_filter(final_direction["oos_direction_proba"], quality_oos, selected_threshold)
    filtered_oos_metrics = base._metrics(y_oos, filtered_oos)

    oof_out = _prediction_output(oof["frame_oof"], direction_oof, quality["quality_oof_proba"], threshold=selected_threshold, prefix="omega1_regime3_expertdq_oof")
    oos_out = _prediction_output(oos, final_direction["oos_direction_proba"], quality_oos, threshold=selected_threshold, prefix="omega1_regime3_expertdq")
    oof_path = variant_dir / f"training_features_2025_{variant}_omega1_regime3_expertdq_oof_20260602.csv"
    oos_path = variant_dir / f"training_features_2026_rebuilt_{variant}_omega1_regime3_expertdq_20260602.csv"
    oof_out.to_csv(oof_path, index=False)
    oos_out.to_csv(oos_path, index=False)
    contract_path = variant_dir / f"{variant}_omega1_regime3_expertdq_contract.joblib"
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "variant": variant,
            "mode": mode,
            "floor": float(floor),
            "label_source_direction": "zigzag_action",
            "label_source_quality": "zigzag_action",
            "route_cols": hard.ROUTE_COLS,
            "route_extra_cols": hard.ROUTE_EXTRA_COLS,
            "expert_names": hard.EXPERT_NAMES,
            "base_cols": hard.volpca.BASE_COLS,
            "volatility_cols": hard.volpca.VOL_COLS,
            "direction_feature_cols": list(final_direction["x_train"].columns),
            "quality_feature_cols": list(xq_oos.columns),
            "selected_quality_threshold": float(selected_threshold),
            "pca_transformer": final_direction["transformer"],
            "direction_model_paths": {k: v["model"] for k, v in final_direction["direction_summaries"].items()},
            "quality_model_paths": {k: v["model"] for k, v in quality["quality_summaries"].items()},
        },
        contract_path,
    )
    delta = {
        "delta_direction_oos_bacc": float(direction_oos_metrics["balanced_accuracy"] - hard.BASELINE_VOLPCA06["oos_bacc"]),
        "delta_filtered_oos_bacc": float(filtered_oos_metrics["balanced_accuracy"] - hard.BASELINE_VOLPCA06["oos_bacc"]),
        "delta_direction_oos_auc": None if direction_oos_metrics["ovr_auc"] is None else float(direction_oos_metrics["ovr_auc"] - hard.BASELINE_VOLPCA06["oos_auc"]),
        "delta_filtered_oos_auc": None if filtered_oos_metrics["ovr_auc"] is None else float(filtered_oos_metrics["ovr_auc"] - hard.BASELINE_VOLPCA06["oos_auc"]),
        "delta_direction_oos_proxy_wr": None if direction_oos_metrics["proxy_wr"] is None else float(direction_oos_metrics["proxy_wr"] - hard.BASELINE_VOLPCA06["oos_proxy_wr"]),
        "delta_filtered_oos_proxy_wr": None if filtered_oos_metrics["proxy_wr"] is None else float(filtered_oos_metrics["proxy_wr"] - hard.BASELINE_VOLPCA06["oos_proxy_wr"]),
        "delta_filtered_oos_proxy_trades": int(filtered_oos_metrics["proxy_trades"] - hard.BASELINE_VOLPCA06["oos_proxy_trades"]),
    }
    return {
        "variant": variant,
        "mode": mode,
        "floor": float(floor),
        "selected_quality_threshold": float(selected_threshold),
        "direction_oof_metrics": direction_oof_metrics,
        "filtered_oof_metrics": filtered_oof_metrics,
        "direction_oos_metrics": direction_oos_metrics,
        "filtered_oos_metrics": filtered_oos_metrics,
        "delta_vs_global_volatility_pca06": delta,
        "threshold_grid": threshold_rows,
        "direction_folds": oof["folds"],
        "pca_folds": oof["pca_folds"],
        "final_pca_explained_variance": final_direction["transformer"].explained_variance,
        "direction_summaries": final_direction["direction_summaries"],
        "quality_summaries": quality["quality_summaries"],
        "artifacts": {
            "variant_dir": str(variant_dir),
            "oof_2025": str(oof_path),
            "oos_2026": str(oos_path),
            "contract": str(contract_path),
        },
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = hard._build_frame(2025)
    oos = hard._build_frame(2026)
    required = [*hard.volpca.BASE_COLS, *hard.volpca.VOL_COLS, *hard.ROUTE_COLS, *hard.ROUTE_EXTRA_COLS]
    hard._assert_finite(train, required, "train")
    hard._assert_finite(oos, required, "oos")
    variants: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    specs = [("hard", 0.0)] + [("soft", f) for f in [0.00, 0.05, 0.10, 0.20]]
    for mode, floor in specs:
        payload = _evaluate_variant(train, oos, mode=mode, floor=float(floor))
        variants[payload["variant"]] = payload
        rows.append(
            {
                "variant": payload["variant"],
                "mode": payload["mode"],
                "floor": payload["floor"],
                "selected_quality_threshold": payload["selected_quality_threshold"],
                "direction_oof_bacc": payload["direction_oof_metrics"]["balanced_accuracy"],
                "direction_oof_auc": payload["direction_oof_metrics"]["ovr_auc"],
                "direction_oof_proxy_wr": payload["direction_oof_metrics"]["proxy_wr"],
                "filtered_oof_bacc": payload["filtered_oof_metrics"]["balanced_accuracy"],
                "filtered_oof_auc": payload["filtered_oof_metrics"]["ovr_auc"],
                "filtered_oof_proxy_wr": payload["filtered_oof_metrics"]["proxy_wr"],
                "filtered_oof_proxy_trades": payload["filtered_oof_metrics"]["proxy_trades"],
                "direction_oos_bacc": payload["direction_oos_metrics"]["balanced_accuracy"],
                "direction_oos_auc": payload["direction_oos_metrics"]["ovr_auc"],
                "direction_oos_proxy_wr": payload["direction_oos_metrics"]["proxy_wr"],
                "direction_oos_proxy_trades": payload["direction_oos_metrics"]["proxy_trades"],
                "filtered_oos_bacc": payload["filtered_oos_metrics"]["balanced_accuracy"],
                "filtered_oos_auc": payload["filtered_oos_metrics"]["ovr_auc"],
                "filtered_oos_proxy_wr": payload["filtered_oos_metrics"]["proxy_wr"],
                "filtered_oos_proxy_trades": payload["filtered_oos_metrics"]["proxy_trades"],
                **payload["delta_vs_global_volatility_pca06"],
            }
        )
    rows.sort(key=lambda r: (float(r["filtered_oos_bacc"]), float(r["filtered_oos_proxy_wr"] or 0.0)), reverse=True)
    report = {
        "model_id": MODEL_ID,
        "design": "Regime3 Current Router runs before direction. Each routed expert owns a local Direction Head and a local 3-class Quality Head. Both Direction and Quality targets are zigzag_action. No global omega1_dir_volpca_* prediction is used as input.",
        "baseline": hard.BASELINE_VOLPCA06,
        "ranking": rows,
        "selected_by_filtered_oos_bacc": rows[0]["variant"],
        "variants": variants,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    pd.DataFrame(rows).to_csv(OUT_DIR / "ranking.csv", index=False)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "ranking": rows}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
