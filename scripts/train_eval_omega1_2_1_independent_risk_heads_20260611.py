#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_1_tabm_7head_risk_20260611 as seven  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega1_2_1_independent_risk_heads_20260611"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
PARENT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"
PARENT_BUNDLE = PARENT_DIR / "true_3head_tabm_bundle.pt"

BASE_NOTIONAL = 0.45
BASE_LEVERAGE = 2.0
BASE_TP = 0.026
BASE_SL = 0.014
COMPENSATED_SCALE = 2.0
MARGIN_CAP = 0.90
OVERLAY_SCALES = {"bull": 0.65, "bear": 0.90, "chop": 0.90}


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _load_parent_bundle(device: torch.device) -> dict[str, dict[str, Any]]:
    if not PARENT_BUNDLE.exists():
        raise FileNotFoundError(PARENT_BUNDLE)
    payload = torch.load(PARENT_BUNDLE, map_location="cpu", weights_only=False)
    models = dict(payload["models"])
    for expert, model_payload in models.items():
        model_payload["_device"] = str(device)
    return {"models": models, "base_cols": list(payload["base_cols"])}


def _base_x(frame: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    return threehead._base_input(frame, base_cols)


def _predict_parent(frame: pd.DataFrame, bundle: dict[str, Any], *, threshold: float, device: torch.device, prefix: str) -> pd.DataFrame:
    x = _base_x(frame, list(bundle["base_cols"]))
    preds = {expert: threehead._predict_payload(bundle["models"][expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = threehead._routed(preds, route, "direction", 3)
    quality = threehead._routed(preds, route, "quality", 3)
    return threehead._prediction_output(frame, direction, quality, threshold=float(threshold), prefix=prefix)


def _parent_to_decisions(src: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    action = pd.to_numeric(src[f"{prefix}_final_action"], errors="raise").astype(int)
    side = np.where(action == 1, 1, np.where(action == 2, -1, 0)).astype(np.int64)
    out = pd.DataFrame(
        {
            "timestamp": src["timestamp"],
            "action": action,
            "side": side,
            "quality_score": pd.to_numeric(src[f"{prefix}_quality_for_action"], errors="raise"),
            "confidence": pd.to_numeric(src[f"{prefix}_dir_confidence"], errors="raise"),
            "router_expert": src[f"{prefix}_router_expert"].astype(str),
        }
    )
    base_margin = np.zeros(len(out), dtype=np.float64)
    for expert, scale in OVERLAY_SCALES.items():
        mask = out["router_expert"].eq(expert).to_numpy() & (side != 0)
        base = BASE_NOTIONAL * float(scale)
        margin = min(base * COMPENSATED_SCALE, MARGIN_CAP)
        base_margin[mask] = margin
    ratio = base_margin / np.maximum(BASE_NOTIONAL * out["router_expert"].map(OVERLAY_SCALES).fillna(0.0).to_numpy(dtype=np.float64), 1e-12)
    exposure = base_margin * BASE_LEVERAGE
    active = side != 0
    out["position_fraction"] = np.where(active, base_margin, 0.0)
    out["leverage"] = np.where(active, BASE_LEVERAGE, 1.0)
    out["notional_exposure"] = np.where(active, exposure, 0.0)
    out["take_profit"] = np.where(active, BASE_TP * ratio * BASE_LEVERAGE, 0.0)
    out["stop_loss"] = np.where(active, BASE_SL * ratio * BASE_LEVERAGE, 0.0)
    out["max_hold_bars"] = 0
    out["cooldown_bars"] = 0
    return out


def _risk_feature_frame(frame: pd.DataFrame, base_cols: list[str], parent: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    x = _base_x(frame, base_cols).reset_index(drop=True)
    meta_cols = [
        f"{prefix}_router_confidence",
        f"{prefix}_router_margin",
        f"{prefix}_dir_p_cash",
        f"{prefix}_dir_p_long",
        f"{prefix}_dir_p_short",
        f"{prefix}_dir_confidence",
        f"{prefix}_dir_side_edge",
        f"{prefix}_dir_trade_prob",
        f"{prefix}_quality_p_cash",
        f"{prefix}_quality_p_long",
        f"{prefix}_quality_p_short",
        f"{prefix}_quality_for_action",
    ]
    for col in meta_cols:
        x[f"parent_{col.removeprefix(prefix + '_')}"] = pd.to_numeric(parent[col], errors="raise").to_numpy(dtype=np.float32)
    action = pd.to_numeric(parent[f"{prefix}_final_action"], errors="raise").to_numpy(dtype=np.int64)
    x["parent_action"] = action.astype(np.float32)
    x["parent_side"] = np.where(action == 1, 1.0, np.where(action == 2, -1.0, 0.0)).astype(np.float32)
    expert = parent[f"{prefix}_router_expert"].astype(str)
    for name in hard.EXPERT_NAMES:
        x[f"parent_expert_{name}"] = expert.eq(name).astype(np.float32).to_numpy()
    bad = [c for c in x.columns if omega._forbidden_feature(str(c))]
    if bad:
        raise RuntimeError(f"independent risk feature audit failed: {bad[:40]}")
    return x.astype(np.float32)


def _fit_hgb(x: pd.DataFrame, y: np.ndarray, w: np.ndarray, *, seed: int) -> HistGradientBoostingClassifier:
    y = np.asarray(y, dtype=np.int64)
    classes = np.unique(y).astype(int).tolist()
    if len(classes) < 2:
        raise RuntimeError(f"risk head has <2 classes: {classes}")
    weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64) * np.asarray(w, dtype=np.float64)
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.035,
        max_leaf_nodes=9,
        min_samples_leaf=45,
        l2_regularization=3.0,
        random_state=int(seed),
    )
    model.fit(x, y, sample_weight=weights)
    return model


def _apply_risk_models(dec: pd.DataFrame, x: pd.DataFrame, models: dict[str, Any]) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = pd.to_numeric(out["action"], errors="raise").to_numpy(dtype=np.int64) != 0
    if not bool(active.any()):
        return out
    xa = x.loc[active].reset_index(drop=True)
    tp_i = models["tp"].predict(xa).astype(np.int64)
    sl_i = models["sl"].predict(xa).astype(np.int64)
    margin_i = models["margin"].predict(xa).astype(np.int64)
    lev_i = models["leverage"].predict(xa).astype(np.int64)
    hold_i = models["max_hold"].predict(xa).astype(np.int64)
    margin = seven.MARGIN_BUCKETS[margin_i]
    lev = seven.LEVERAGE_BUCKETS[lev_i]
    idx = np.flatnonzero(active)
    out.loc[idx, "take_profit"] = seven.TP_BUCKETS[tp_i]
    out.loc[idx, "stop_loss"] = seven.SL_BUCKETS[sl_i]
    out.loc[idx, "position_fraction"] = margin
    out.loc[idx, "leverage"] = lev
    out.loc[idx, "notional_exposure"] = margin * lev
    out.loc[idx, "max_hold_bars"] = seven.MAX_HOLD_BUCKETS[hold_i]
    out.loc[idx, "cooldown_bars"] = 0
    return out


def _bucket_summary(dec: pd.DataFrame) -> dict[str, Any]:
    active = pd.to_numeric(dec["action"], errors="raise").to_numpy(dtype=np.int64) != 0
    if not bool(active.any()):
        return {}
    d = dec.loc[active]
    return {
        "entries": int(len(d)),
        "avg_margin": float(pd.to_numeric(d["position_fraction"], errors="raise").mean()),
        "avg_leverage": float(pd.to_numeric(d["leverage"], errors="raise").mean()),
        "avg_effective_exposure": float(pd.to_numeric(d["notional_exposure"], errors="raise").mean()),
        "tp_counts": {str(k): int(v) for k, v in d["take_profit"].round(6).value_counts().sort_index().items()},
        "sl_counts": {str(k): int(v) for k, v in d["stop_loss"].round(6).value_counts().sort_index().items()},
        "margin_counts": {str(k): int(v) for k, v in d["position_fraction"].round(6).value_counts().sort_index().items()},
        "leverage_counts": {str(k): int(v) for k, v in d["leverage"].round(6).value_counts().sort_index().items()},
        "max_hold_counts": {str(k): int(v) for k, v in d["max_hold_bars"].value_counts().sort_index().items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-threshold", type=float, default=0.80)
    ap.add_argument("--risk-label-max-rows", type=int, default=1200)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260611)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    device = _device(str(args.device))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = seven._prepare_frames()
    fee, slip = omega._load_fee_slip()
    bundle = _load_parent_bundle(device)
    base_cols = list(bundle["base_cols"])

    train = frames["train_raw"]
    val = frames["val_raw"]
    oos = frames["oos_raw"]
    prefix = "omega1_regime3_expertdq"
    train_parent = _predict_parent(train, bundle, threshold=float(args.parent_threshold), device=device, prefix=prefix)
    val_parent = _predict_parent(val, bundle, threshold=float(args.parent_threshold), device=device, prefix=prefix)
    oos_parent = _predict_parent(oos, bundle, threshold=float(args.parent_threshold), device=device, prefix=prefix)
    train_dec = _parent_to_decisions(train_parent, prefix=prefix)
    val_dec_base = _parent_to_decisions(val_parent, prefix=prefix)
    oos_dec_base = _parent_to_decisions(oos_parent, prefix=prefix)

    train_action = pd.to_numeric(train_dec["action"], errors="raise").to_numpy(dtype=np.int64)
    risk_labels, risk_diag = seven._risk_labels(
        train,
        train_action,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_rows=int(args.risk_label_max_rows),
    )
    x_train = _risk_feature_frame(train, base_cols, train_parent, prefix=prefix)
    labeled = np.asarray(risk_labels["risk_weight"], dtype=np.float32) > 0.0
    if int(labeled.sum()) < 100:
        raise RuntimeError(f"too few risk labels: {int(labeled.sum())}")
    x_fit = x_train.loc[labeled].reset_index(drop=True)
    w_fit = np.asarray(risk_labels["risk_weight"], dtype=np.float32)[labeled]
    models = {
        "tp": _fit_hgb(x_fit, np.asarray(risk_labels["tp"], dtype=np.int64)[labeled], w_fit, seed=int(args.seed) + 1),
        "sl": _fit_hgb(x_fit, np.asarray(risk_labels["sl"], dtype=np.int64)[labeled], w_fit, seed=int(args.seed) + 2),
        "margin": _fit_hgb(x_fit, np.asarray(risk_labels["margin"], dtype=np.int64)[labeled], w_fit, seed=int(args.seed) + 3),
        "leverage": _fit_hgb(x_fit, np.asarray(risk_labels["leverage"], dtype=np.int64)[labeled], w_fit, seed=int(args.seed) + 4),
        "max_hold": _fit_hgb(x_fit, np.asarray(risk_labels["max_hold"], dtype=np.int64)[labeled], w_fit, seed=int(args.seed) + 5),
    }
    x_val = _risk_feature_frame(val, base_cols, val_parent, prefix=prefix)
    x_oos = _risk_feature_frame(oos, base_cols, oos_parent, prefix=prefix)
    val_dec_risk = _apply_risk_models(val_dec_base, x_val, models)
    oos_dec_risk = _apply_risk_models(oos_dec_base, x_oos, models)

    base_val = omega._metrics(val, val_dec_base, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    base_oos = omega._metrics(oos, oos_dec_base, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    risk_val = omega._metrics(val, val_dec_risk, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    risk_oos = omega._metrics(oos, oos_dec_risk, fee=fee, slip=slip, cost_mult=float(args.cost_mult))

    train_parent.to_csv(OUT_DIR / "train_parent_predictions.csv", index=False)
    val_parent.to_csv(OUT_DIR / "validation_parent_predictions.csv", index=False)
    oos_parent.to_csv(OUT_DIR / "oos_parent_predictions.csv", index=False)
    val_dec_base.to_csv(OUT_DIR / "validation_base_decisions.csv", index=False)
    oos_dec_base.to_csv(OUT_DIR / "oos_base_decisions.csv", index=False)
    val_dec_risk.to_csv(OUT_DIR / "validation_independent_risk_decisions.csv", index=False)
    oos_dec_risk.to_csv(OUT_DIR / "oos_independent_risk_decisions.csv", index=False)
    joblib.dump({"models": models, "feature_cols": list(x_train.columns)}, OUT_DIR / "independent_risk_heads.joblib")

    ranking = pd.DataFrame(
        [
            {"variant": "fixed_true_leverage_template", "split": "validation", **base_val},
            {"variant": "fixed_true_leverage_template", "split": "oos", **base_oos},
            {"variant": "independent_risk_heads", "split": "validation", **risk_val},
            {"variant": "independent_risk_heads", "split": "oos", **risk_oos},
        ]
    )
    ranking.to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Frozen Omega1.2 true 3-head TabM Direction/Quality parent. TP/SL/margin/leverage/max_hold are independent HGB bucket heads trained only on parent-entry rows.",
        "parent": str(PARENT_BUNDLE),
        "risk_label_diag": risk_diag,
        "feature_count": int(len(x_train.columns)),
        "results": {
            "fixed_true_leverage_template": {"validation": base_val, "oos": base_oos},
            "independent_risk_heads": {"validation": risk_val, "oos": risk_oos},
        },
        "bucket_summary": {
            "validation_base": _bucket_summary(val_dec_base),
            "oos_base": _bucket_summary(oos_dec_base),
            "validation_independent": _bucket_summary(val_dec_risk),
            "oos_independent": _bucket_summary(oos_dec_risk),
        },
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "model": str(OUT_DIR / "independent_risk_heads.joblib"),
            "ranking": str(OUT_DIR / "ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "results": report["results"], "bucket_summary": report["bucket_summary"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
