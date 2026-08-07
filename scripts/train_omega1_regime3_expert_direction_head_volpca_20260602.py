#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import train_omega1_direction_head_direction_only_20260602 as base
import train_omega1_direction_head_volatility_pca_20260602 as volpca


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_regime3_expert_direction_head_volpca_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_regime3_expert_direction_head_volpca_20260602"

REGIME3_CURRENT_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
REGIME3_CURRENT_FILES = {
    2025: REGIME3_CURRENT_DIR / "training_features_2025_regime3_current_sensitive_hmm_wide24.csv",
    2026: REGIME3_CURRENT_DIR / "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv",
}
ROUTE_COLS = [
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
]
ROUTE_EXTRA_COLS = [
    "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_margin",
]
EXPERT_NAMES = ["bull", "bear", "chop"]

BASELINE_VOLPCA06 = {
    "variant": "volatility_pca06_global",
    "feature_count": 61,
    "oos_bacc": 0.6052110159,
    "oos_auc": 0.7916830103,
    "oos_proxy_wr": 0.6626651567,
    "oos_proxy_trades": 13245,
}


def _json_default(obj: Any) -> Any:
    return base._json_default(obj)


def _assert_finite(frame: pd.DataFrame, cols: list[str], label: str) -> None:
    arr = frame[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        bad = {c: int((~np.isfinite(frame[c].to_numpy(dtype=np.float64))).sum()) for c in cols}
        bad = {k: v for k, v in bad.items() if v}
        raise ValueError(f"{label} contains non-finite values: {bad}")


def _regime_frame(year: int) -> pd.DataFrame:
    path = REGIME3_CURRENT_FILES[int(year)]
    frame = base._read_csv(path)
    cols = ["timestamp", *ROUTE_COLS, *ROUTE_EXTRA_COLS]
    missing = sorted(set(cols) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing regime columns: {missing}")
    years = sorted(frame["timestamp"].dt.year.dropna().astype(int).unique().tolist())
    if years != [int(year)]:
        raise RuntimeError(f"regime3 current year guard failed for {year}: {years}")
    return frame[cols]


def _build_frame(year: int) -> pd.DataFrame:
    frame, groups, _missing = volpca.ctx._build_frame(year)
    if groups.get("volatility_context") != volpca.VOL_COLS:
        raise RuntimeError("volatility context contract changed")
    regime = _regime_frame(year)
    return base._exact_join(frame, regime, [*ROUTE_COLS, *ROUTE_EXTRA_COLS], f"regime3_current {year}")


def _route_id(frame: pd.DataFrame) -> np.ndarray:
    values = frame[ROUTE_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite Regime3 route probabilities")
    return np.argmax(values, axis=1).astype(np.int64)


def _route_conf(frame: pd.DataFrame) -> np.ndarray:
    return frame[ROUTE_COLS].to_numpy(dtype=np.float64).max(axis=1)


def _fit_expert_models(x: pd.DataFrame, y: np.ndarray, route: np.ndarray, *, seed: int, iterations: int, model_dir: Path) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(EXPERT_NAMES):
        mask = route == idx
        if int(mask.sum()) < 1000:
            raise RuntimeError(f"{expert}: too few rows for expert Direction Head training: {int(mask.sum())}")
        labels = y[mask]
        classes = sorted(np.unique(labels).astype(int).tolist())
        if classes != [0, 1, 2]:
            raise RuntimeError(f"{expert}: missing zigzag_action classes in training subset: {classes}")
        model = base._fit_catboost(x.loc[mask].reset_index(drop=True), labels, seed=seed + idx, iterations=iterations)
        model_path = model_dir / f"{expert}_direction_head.cbm"
        model.save_model(str(model_path))
        models[expert] = model
        summaries[expert] = {
            "rows": int(mask.sum()),
            "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(labels, minlength=3))},
            "model": str(model_path),
        }
    return {"models": models, "summaries": summaries}


def _predict_all_experts(models: dict[str, Any], x: pd.DataFrame) -> dict[str, np.ndarray]:
    return {expert: base._proba3(model, x) for expert, model in models.items()}


def _routed_proba(expert_proba: dict[str, np.ndarray], route: np.ndarray) -> np.ndarray:
    out = np.zeros((len(route), 3), dtype=np.float64)
    for idx, expert in enumerate(EXPERT_NAMES):
        mask = route == idx
        out[mask] = expert_proba[expert][mask]
    return out


def _outputs(frame: pd.DataFrame, expert_proba: dict[str, np.ndarray], routed: np.ndarray, *, prefix: str) -> pd.DataFrame:
    route = _route_id(frame)
    out = pd.DataFrame(
        {
            "timestamp": frame["timestamp"].to_numpy(),
            f"{prefix}_router_expert": np.asarray(EXPERT_NAMES, dtype=object)[route],
            f"{prefix}_router_confidence": _route_conf(frame),
        }
    )
    for expert in EXPERT_NAMES:
        p = expert_proba[expert]
        out[f"{prefix}_{expert}_p_cash"] = p[:, 0]
        out[f"{prefix}_{expert}_p_long"] = p[:, 1]
        out[f"{prefix}_{expert}_p_short"] = p[:, 2]
        out[f"{prefix}_{expert}_confidence"] = np.max(p, axis=1)
        out[f"{prefix}_{expert}_side_edge"] = p[:, 1] - p[:, 2]
        out[f"{prefix}_{expert}_trade_prob"] = p[:, 1] + p[:, 2]
        out[f"{prefix}_{expert}_action"] = np.argmax(p, axis=1).astype(np.int64)
    routed_out = base._outputs(frame, routed, prefix=prefix)
    for col in routed_out.columns:
        if col != "timestamp":
            out[col] = routed_out[col].to_numpy()
    return out


def _features_with_transform(frame: pd.DataFrame, transformer: volpca.VolPca) -> pd.DataFrame:
    return pd.concat(
        [
            frame[volpca.BASE_COLS].reset_index(drop=True),
            transformer.transform(frame),
        ],
        axis=1,
    )


def _oof(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
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
        transformer = volpca.VolPca(6).fit(fit_frame)
        x_fit = _features_with_transform(fit_frame, transformer)
        x_pred = _features_with_transform(pred_frame, transformer)
        route_fit = _route_id(fit_frame)
        route_pred = _route_id(pred_frame)
        bundle = _fit_expert_models(
            x_fit,
            y[:start],
            route_fit,
            seed=20260602 + fold * 100,
            iterations=500,
            model_dir=OUT_DIR / "oof_folds" / f"fold_{fold}",
        )
        expert_pred = _predict_all_experts(bundle["models"], x_pred)
        routed = _routed_proba(expert_pred, route_pred)
        proba[start:end] = routed
        covered[start:end] = True
        output_parts.append(_outputs(pred_frame, expert_pred, routed, prefix="omega1_regime3_expert_dir_oof"))
        folds.append(
            {
                "fold": fold,
                "train_rows": int(start),
                "predict_start": int(start),
                "predict_end": int(end),
                "expert_summaries": bundle["summaries"],
                "route_counts_fit": {EXPERT_NAMES[i]: int((route_fit == i).sum()) for i in range(3)},
                "route_counts_pred": {EXPERT_NAMES[i]: int((route_pred == i).sum()) for i in range(3)},
                "metrics": base._metrics(y[start:end], routed),
            }
        )
        pca_folds.append({"fold": fold, "explained_variance": transformer.explained_variance})
    return proba, covered, pd.concat(output_parts, ignore_index=True), folds, pca_folds


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.DROP_EVENTS.clear()
    train = _build_frame(2025)
    oos = _build_frame(2026)
    y_train = train["zigzag_action"].to_numpy(dtype=np.int64)
    y_oos = oos["zigzag_action"].to_numpy(dtype=np.int64)

    base._validate_features(volpca.BASE_COLS, train)
    base._validate_features(volpca.BASE_COLS, oos)
    volpca.ctx._validate_context_cols(volpca.VOL_COLS, train)
    volpca.ctx._validate_context_cols(volpca.VOL_COLS, oos)
    _assert_finite(train, [*volpca.BASE_COLS, *volpca.VOL_COLS, *ROUTE_COLS, *ROUTE_EXTRA_COLS], "train")
    _assert_finite(oos, [*volpca.BASE_COLS, *volpca.VOL_COLS, *ROUTE_COLS, *ROUTE_EXTRA_COLS], "oos")

    oof_proba, covered, oof_out, folds, pca_folds = _oof(train)
    oof_metrics = base._metrics(y_train[covered], oof_proba[covered])

    final_transformer = volpca.VolPca(6).fit(train)
    x_train = _features_with_transform(train, final_transformer)
    x_oos = _features_with_transform(oos, final_transformer)
    train_route = _route_id(train)
    oos_route = _route_id(oos)
    final_bundle = _fit_expert_models(
        x_train,
        y_train,
        train_route,
        seed=20260602,
        iterations=800,
        model_dir=OUT_DIR / "final_experts",
    )
    oos_expert_pred = _predict_all_experts(final_bundle["models"], x_oos)
    oos_routed = _routed_proba(oos_expert_pred, oos_route)
    oos_metrics = base._metrics(y_oos, oos_routed)
    oos_out = _outputs(oos, oos_expert_pred, oos_routed, prefix="omega1_regime3_expert_dir")

    oof_path = OUT_DIR / "training_features_2025_regime3_expert_direction_volpca_oof_20260602.csv"
    oos_path = OUT_DIR / "training_features_2026_rebuilt_regime3_expert_direction_volpca_20260602.csv"
    oof_out.to_csv(oof_path, index=False)
    oos_out.to_csv(oos_path, index=False)
    contract_path = OUT_DIR / "regime3_expert_direction_volpca_contract.joblib"
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "label_source": "zigzag_action",
            "route_cols": ROUTE_COLS,
            "route_extra_cols": ROUTE_EXTRA_COLS,
            "expert_names": EXPERT_NAMES,
            "base_cols": volpca.BASE_COLS,
            "volatility_cols": volpca.VOL_COLS,
            "feature_cols": list(x_train.columns),
            "pca_transformer": final_transformer,
            "expert_model_paths": {k: v["model"] for k, v in final_bundle["summaries"].items()},
        },
        contract_path,
    )
    delta = {
        "oos_bacc": float(oos_metrics["balanced_accuracy"] - BASELINE_VOLPCA06["oos_bacc"]),
        "oos_auc": None if oos_metrics["ovr_auc"] is None else float(oos_metrics["ovr_auc"] - BASELINE_VOLPCA06["oos_auc"]),
        "oos_proxy_wr": None if oos_metrics["proxy_wr"] is None else float(oos_metrics["proxy_wr"] - BASELINE_VOLPCA06["oos_proxy_wr"]),
        "oos_proxy_trades": int(oos_metrics["proxy_trades"] - BASELINE_VOLPCA06["oos_proxy_trades"]),
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Regime3 current router selects bull/bear/chop. Each expert owns its own CatBoost Direction Head trained on the same core_plus_tsfm_chronos + volatility_pca06 feature contract; there is no global Direction Head input.",
        "label_source": "zigzag_action",
        "baseline": BASELINE_VOLPCA06,
        "oof_metrics": oof_metrics,
        "oos_metrics": oos_metrics,
        "delta_vs_global_volatility_pca06": delta,
        "route_counts_train": {EXPERT_NAMES[i]: int((train_route == i).sum()) for i in range(3)},
        "route_counts_oos": {EXPERT_NAMES[i]: int((oos_route == i).sum()) for i in range(3)},
        "expert_summaries": final_bundle["summaries"],
        "folds": folds,
        "pca_folds": pca_folds,
        "final_pca_explained_variance": final_transformer.explained_variance,
        "drop_events": base.DROP_EVENTS,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "oof_2025": str(oof_path),
            "oos_2026": str(oos_path),
            "contract": str(contract_path),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "variant": "regime3_expert_direction_volpca",
                "feature_count_per_expert": int(x_train.shape[1]),
                "oof_bacc": oof_metrics["balanced_accuracy"],
                "oof_auc": oof_metrics["ovr_auc"],
                "oof_proxy_wr": oof_metrics["proxy_wr"],
                "oos_bacc": oos_metrics["balanced_accuracy"],
                "oos_auc": oos_metrics["ovr_auc"],
                "oos_proxy_wr": oos_metrics["proxy_wr"],
                "oos_proxy_trades": oos_metrics["proxy_trades"],
                "delta_oos_bacc_vs_global_volpca06": delta["oos_bacc"],
                "delta_oos_auc_vs_global_volpca06": delta["oos_auc"],
                "delta_oos_proxy_wr_vs_global_volpca06": delta["oos_proxy_wr"],
                "delta_oos_trades_vs_global_volpca06": delta["oos_proxy_trades"],
            }
        ]
    ).to_csv(OUT_DIR / "ranking.csv", index=False)
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "oos_metrics": oos_metrics, "delta": delta}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
