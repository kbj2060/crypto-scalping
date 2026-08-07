#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega2_1_dsac_feature_sweep_20260609 as dsac  # noqa: E402
import train_eval_omega1_2_1_full_retrain_cash_alpha43_20260608 as full  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_label_family_20260606 as label_family  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
from freeze_omega2_1_hgb_12seed_cash_sleeve_20260609 import (  # noqa: E402
    BUNDLE_PATH,
    MODEL_ID as OMEGA21_MODEL_ID,
    RISK,
)


OUT_DIR = ROOT / "tmp/causal_regen_20260516" / "omega2_1_dsac_overlay_20260609"
BASELINE_OOS = dsac.BASELINE_OOS


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


def _hgb_proba(bundle: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    cols = list(bundle["feature_cols"])
    dsac._reject_forbidden(cols, "omega21_hgb")
    if list(features.columns) != cols:
        raise RuntimeError("Omega2.1 feature columns do not match frozen HGB bundle")
    arr = features[cols].to_numpy(dtype=np.float64)
    probs = [dsac._classes_to_proba(model, model.predict_proba(arr)) for model in bundle["models"]]
    return np.stack(probs, axis=0).mean(axis=0)


def _hgb_action_conf(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.argmax(proba, axis=1).astype(np.int64)
    conf = proba[np.arange(len(proba)), raw].astype(np.float64)
    action = np.where(conf >= 0.55, raw, sleeve.ACTION_CASH).astype(np.int64)
    return action, conf


def _metric(prefix: str, m: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(m["pnl"]),
        f"{prefix}_mdd": float(m["mdd"]),
        f"{prefix}_wr": float(m["wr"]),
        f"{prefix}_trades": int(m["trades"]),
        f"{prefix}_fallback_entries": int(m.get("fallback_entries", 0)),
        f"{prefix}_primary_takeovers": int(m.get("primary_takeovers", 0)),
        f"{prefix}_reasons": m.get("exit_reasons", {}),
    }


def _metrics(frame: pd.DataFrame, dec: pd.DataFrame, action: np.ndarray, conf: np.ndarray, fee: float, slip: float) -> dict[str, Any]:
    return sleeve._metrics_with_fallback(frame, dec, RISK, action, conf, 0.55, fee=fee, slip=slip, cost_mult=3.0)


def _state_from_ckpt(
    ckpt: dict[str, Any],
    val_all: pd.DataFrame,
    oos_all: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    cols = list(ckpt["feature_cols"])
    dsac._reject_forbidden(cols, "dsac_overlay")
    norm = ckpt["normalizer"]
    mu = np.asarray(norm["mean"], dtype=np.float64)
    sd = np.asarray(norm["std"], dtype=np.float64)
    if len(cols) != len(mu) or len(cols) != len(sd):
        raise RuntimeError("DSAC checkpoint normalizer does not match feature columns")
    val = val_all[cols].to_numpy(dtype=np.float64)
    oos = oos_all[cols].to_numpy(dtype=np.float64)
    return (
        np.nan_to_num((val - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        np.nan_to_num((oos - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
    )


def _load_dsac(path: Path) -> tuple[dsac.Omega21DSACAgent, dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("train_cfg", {})
    if int(cfg.get("hidden", 256)) != 256:
        raise RuntimeError(f"non-heavy DSAC checkpoint rejected: {path}")
    model = dsac.Omega21DSACAgent(len(ckpt["feature_cols"]), 256, n_quantiles=32)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval(), ckpt


def _overlay_actions(
    hgb_action: np.ndarray,
    hgb_conf: np.ndarray,
    d_action: np.ndarray,
    d_conf: np.ndarray,
    mode: str,
    dsac_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    out_action = hgb_action.copy()
    out_conf = hgb_conf.copy()
    active = hgb_action != sleeve.ACTION_CASH
    same = d_action == hgb_action
    opposite = active & (d_action != sleeve.ACTION_CASH) & (d_action != hgb_action)
    strong = d_conf >= float(dsac_thr)
    if mode == "confirm_same":
        veto = active & (~same | ~strong)
        out_action[veto] = sleeve.ACTION_CASH
        out_conf[veto] = 0.0
    elif mode == "veto_opposite":
        veto = opposite & strong
        out_action[veto] = sleeve.ACTION_CASH
        out_conf[veto] = 0.0
    elif mode == "replace_high":
        replace = strong & (d_action != sleeve.ACTION_CASH)
        out_action[replace] = d_action[replace]
        out_conf[replace] = np.maximum(hgb_conf[replace], d_conf[replace])
    else:
        raise RuntimeError(f"unknown overlay mode: {mode}")
    return out_action, out_conf


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(BUNDLE_PATH)
    if bundle.get("model_id") != OMEGA21_MODEL_ID:
        raise RuntimeError(f"unexpected Omega2.1 bundle: {bundle.get('model_id')}")
    base_cols = list(bundle["feature_cols"])
    dsac._reject_forbidden(base_cols, "omega21_base")

    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_dec, val_features = full._build_split(frames, "validation")
    oos_frame, oos_dec, oos_features = full._build_split(frames, "oos")
    if list(val_features.columns) != base_cols or list(oos_features.columns) != base_cols:
        raise RuntimeError("Omega2.1 feature columns do not match frozen manifest")

    y, valid_mask, _label_diag = label_family._triple_barrier_labels(val_frame, atr_mult=1.0, max_hold=24, min_barrier=0.0035)
    train_mask = (~omega._active(val_dec)) & valid_mask
    hgb_val_p, hgb_oos_p, hgb_val_stack, hgb_oos_stack, _hgb_diag = dsac._hgb_oof_and_full(val_features, y, train_mask, oos_features)
    hgb_val_features = dsac._hgb_feature_frame(hgb_val_p, hgb_val_stack)
    hgb_oos_features = dsac._hgb_feature_frame(hgb_oos_p, hgb_oos_stack)
    val_all = pd.concat([val_features.reset_index(drop=True), hgb_val_features], axis=1)
    oos_all = pd.concat([oos_features.reset_index(drop=True), hgb_oos_features], axis=1)
    dsac._reject_forbidden(list(val_all.columns), "overlay_state")

    val_hgb_action, val_hgb_conf = _hgb_action_conf(_hgb_proba(bundle, val_features))
    oos_hgb_action, oos_hgb_conf = _hgb_action_conf(_hgb_proba(bundle, oos_features))
    baseline_val = _metrics(val_frame, val_dec, val_hgb_action, val_hgb_conf, fee, slip)
    baseline_oos = _metrics(oos_frame, oos_dec, oos_hgb_action, oos_hgb_conf, fee, slip)
    rows: list[dict[str, Any]] = [
        {
            "candidate": "omega21_hgb_baseline",
            "source": "baseline",
            "mode": "none",
            "dsac_thr": 0.0,
            **_metric("val", baseline_val),
            **_metric("oos", baseline_oos),
        }
    ]
    ckpts = [
        ROOT / "tmp/causal_regen_20260516/omega2_1_dsac_feature_sweep_20260609/dsac_parent23_s260901.pt",
        ROOT / "tmp/causal_regen_20260516/omega2_1_dsac_feature_sweep_20260609_cost3/dsac_price19_hgb9_s260901.pt",
        ROOT / "tmp/causal_regen_20260516/omega2_1_dsac_feature_sweep_20260609_cost3_bc0_cvar25_top3/dsac_price19_hgb9_s260902.pt",
    ]
    for ckpt_path in ckpts:
        if not ckpt_path.exists():
            continue
        model, ckpt = _load_dsac(ckpt_path)
        x_val, x_oos = _state_from_ckpt(ckpt, val_all, oos_all)
        val_d_action, val_d_conf, _val_q = dsac._predict(model, x_val)
        oos_d_action, oos_d_conf, _oos_q = dsac._predict(model, x_oos)
        source = ckpt_path.parent.name + "/" + ckpt_path.stem
        for mode in ("confirm_same", "veto_opposite", "replace_high"):
            for dsac_thr in (0.25, 0.35, 0.45, 0.55, 0.65):
                va, vc = _overlay_actions(val_hgb_action, val_hgb_conf, val_d_action, val_d_conf, mode, dsac_thr)
                oa, oc = _overlay_actions(oos_hgb_action, oos_hgb_conf, oos_d_action, oos_d_conf, mode, dsac_thr)
                val_m = _metrics(val_frame, val_dec, va, vc, fee, slip)
                oos_m = _metrics(oos_frame, oos_dec, oa, oc, fee, slip)
                row = {
                    "candidate": f"{ckpt_path.stem}_{mode}_dthr{dsac_thr:.2f}",
                    "source": source,
                    "mode": mode,
                    "dsac_thr": float(dsac_thr),
                    **_metric("val", val_m),
                    **_metric("oos", oos_m),
                }
                row["oos_delta_vs_omega21"] = float(row["oos_pnl"] - BASELINE_OOS["pnl"])
                rows.append(row)
    ranking = pd.DataFrame(rows).sort_values(["oos_pnl", "val_pnl", "oos_mdd"], ascending=[False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": "omega2_1_dsac_overlay_20260609",
        "status": "research_not_live_promoted",
        "baseline_oos": BASELINE_OOS,
        "top": ranking.head(20).to_dict(orient="records"),
        "artifacts": {"out_dir": str(OUT_DIR), "ranking": str(OUT_DIR / "ranking.csv"), "report": str(OUT_DIR / "report.json")},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top": report["top"][:8]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
