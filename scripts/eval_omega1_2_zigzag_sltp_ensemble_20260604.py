#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_pathlabel_3head_20260604 as sltp  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega1_2_zigzag_sltp_ensemble_20260604"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
ZIGZAG_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"
SLTP_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_pathlabel_3head_20260604_sltp_practical_tp026_sl012_e36_seed260604"
PRACTICAL_EXPERT_THRESHOLDS = {"bull": 0.72, "bear": 0.64, "chop": 0.65}
THRESHOLD_DELTAS = [-0.05, 0.0, 0.05]
SLTP_THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
BLEND_WEIGHTS = [0.25, 0.50, 0.75]


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _read_predictions(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    pred = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if not pred["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError(f"timestamp contract mismatch: {path}")
    return pred


def _source_prefix(oof: bool) -> str:
    return "omega1_regime3_expertdq_oof_" if oof else "omega1_regime3_expertdq_"


def _side_quality(src: pd.DataFrame, prefix: str, side: int) -> np.ndarray:
    col = "quality_p_long" if int(side) == omega.ACTION_LONG else "quality_p_short"
    return pd.to_numeric(src[f"{prefix}{col}"], errors="raise").to_numpy(dtype=np.float64)


def _action_quality(src: pd.DataFrame, prefix: str, action: np.ndarray) -> np.ndarray:
    long_q = pd.to_numeric(src[f"{prefix}quality_p_long"], errors="raise").to_numpy(dtype=np.float64)
    short_q = pd.to_numeric(src[f"{prefix}quality_p_short"], errors="raise").to_numpy(dtype=np.float64)
    cash_q = pd.to_numeric(src[f"{prefix}quality_p_cash"], errors="raise").to_numpy(dtype=np.float64)
    return np.where(action == omega.ACTION_LONG, long_q, np.where(action == omega.ACTION_SHORT, short_q, cash_q))


def _threshold_action(src: pd.DataFrame, prefix: str, thresholds: dict[str, float] | float) -> np.ndarray:
    action = pd.to_numeric(src[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    q = pd.to_numeric(src[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    if isinstance(thresholds, dict):
        expert = src[f"{prefix}router_expert"].astype(str).to_numpy()
        thr = np.array([float(thresholds.get(str(x).replace("chop_expert", "chop"), 1.0)) for x in expert], dtype=np.float64)
    else:
        thr = float(thresholds)
    return np.where(q >= thr, action, omega.ACTION_CASH).astype(np.int64)


def _expert_thresholds(delta: float) -> dict[str, float]:
    return {k: float(np.clip(v + float(delta), 0.01, 0.99)) for k, v in PRACTICAL_EXPERT_THRESHOLDS.items()}


def _make_src(base: pd.DataFrame, prefix: str, final_action: np.ndarray, quality_score: np.ndarray, confidence: np.ndarray) -> pd.DataFrame:
    out = base.copy()
    out[f"{prefix}final_action"] = final_action.astype(np.int64)
    out[f"{prefix}quality_for_action"] = quality_score.astype(np.float64)
    out[f"{prefix}quality_threshold"] = 0.0
    out[f"{prefix}dir_confidence"] = confidence.astype(np.float64)
    return out


def _decision_from_src(src: pd.DataFrame, *, oof: bool, tp: float, sl: float) -> pd.DataFrame:
    return sltp._scale_dec(omega._to_fixed_decisions(src, oof=oof), take_profit=float(tp), stop_loss=float(sl))


def _metrics(frame: pd.DataFrame, src: pd.DataFrame, *, oof: bool, fee: float, slip: float, cost_mult: float, tp: float, sl: float) -> dict[str, Any]:
    dec = _decision_from_src(src, oof=oof, tp=tp, sl=sl)
    return omega._metrics(frame, dec, fee=fee, slip=slip, cost_mult=cost_mult)


@torch.no_grad()
def _predict_sltp(frame: pd.DataFrame, bundle: dict[str, Any], *, oof: bool, device: torch.device) -> pd.DataFrame:
    base_cols = list(bundle["base_cols"])
    x = tabm._base_input(frame, base_cols)
    models = bundle["models"]
    preds = {expert: tabm._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = tabm._routed(preds, route, "direction", 3)
    quality = tabm._routed(preds, route, "quality", 3)
    prefix = _source_prefix(oof).rstrip("_")
    return tabm._prediction_output(frame, direction, quality, threshold=0.50, prefix=prefix)


def _blend_source(zig: pd.DataFrame, path: pd.DataFrame, *, oof: bool, weight_zigzag: float, threshold: float) -> pd.DataFrame:
    prefix = _source_prefix(oof)
    out = zig.copy()
    for name in ("p_cash", "p_long", "p_short"):
        out[f"{prefix}dir_{name}"] = (
            float(weight_zigzag) * pd.to_numeric(zig[f"{prefix}dir_{name}"], errors="raise").to_numpy(dtype=np.float64)
            + (1.0 - float(weight_zigzag)) * pd.to_numeric(path[f"{prefix}dir_{name}"], errors="raise").to_numpy(dtype=np.float64)
        )
        out[f"{prefix}quality_{name}"] = (
            float(weight_zigzag) * pd.to_numeric(zig[f"{prefix}quality_{name}"], errors="raise").to_numpy(dtype=np.float64)
            + (1.0 - float(weight_zigzag)) * pd.to_numeric(path[f"{prefix}quality_{name}"], errors="raise").to_numpy(dtype=np.float64)
        )
    probs = out[[f"{prefix}dir_p_cash", f"{prefix}dir_p_long", f"{prefix}dir_p_short"]].to_numpy(dtype=np.float64)
    action = probs.argmax(axis=1).astype(np.int64)
    quality = _action_quality(out, prefix, action)
    final_action = np.where(quality >= float(threshold), action, omega.ACTION_CASH).astype(np.int64)
    confidence = probs.max(axis=1)
    return _make_src(out, prefix, final_action, quality, confidence)


def _ensemble_source(
    zig: pd.DataFrame,
    path: pd.DataFrame,
    *,
    oof: bool,
    mode: str,
    zigzag_delta: float,
    sltp_threshold: float,
) -> pd.DataFrame:
    prefix = _source_prefix(oof)
    z_action = _threshold_action(zig, prefix, _expert_thresholds(float(zigzag_delta)))
    p_action = _threshold_action(path, prefix, float(sltp_threshold))
    z_q = _action_quality(zig, prefix, z_action)
    p_q = _action_quality(path, prefix, p_action)
    z_conf = pd.to_numeric(zig[f"{prefix}dir_confidence"], errors="raise").to_numpy(dtype=np.float64)
    p_conf = pd.to_numeric(path[f"{prefix}dir_confidence"], errors="raise").to_numpy(dtype=np.float64)

    if mode == "agreement_same_side":
        active = (z_action == p_action) & (z_action != omega.ACTION_CASH)
        final_action = np.where(active, z_action, omega.ACTION_CASH).astype(np.int64)
        quality = np.minimum(z_q, p_q)
        confidence = np.minimum(z_conf, p_conf)
    elif mode == "zigzag_primary_sltp_veto":
        same = (z_action == p_action) & (z_action != omega.ACTION_CASH)
        final_action = np.where(same, z_action, omega.ACTION_CASH).astype(np.int64)
        quality = np.where(same, 0.5 * (z_q + p_q), 0.0)
        confidence = np.where(same, 0.5 * (z_conf + p_conf), 0.0)
    elif mode == "sltp_primary_zigzag_veto":
        same = (z_action == p_action) & (p_action != omega.ACTION_CASH)
        final_action = np.where(same, p_action, omega.ACTION_CASH).astype(np.int64)
        quality = np.where(same, 0.5 * (z_q + p_q), 0.0)
        confidence = np.where(same, 0.5 * (z_conf + p_conf), 0.0)
    else:
        raise RuntimeError(f"unknown ensemble mode: {mode}")
    return _make_src(zig, prefix, final_action, quality, confidence)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--tp", type=float, default=0.026)
    ap.add_argument("--sl", type=float, default=0.012)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    device = _device(args.device)
    out_dir = OUT_DIR if not args.out_suffix.strip() else OUT_DIR.parent / f"{MODEL_ID}_{args.out_suffix.strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = tabm._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    bundle = torch.load(SLTP_DIR / "pathlabel_3head_tabm_bundle.pt", map_location=device, weights_only=False)

    zig_val = _read_predictions(ZIGZAG_DIR / "validation_predictions_2025_true3head.csv", frames["val_raw"])
    zig_oos = _read_predictions(ZIGZAG_DIR / "oos_predictions_2026_true3head.csv", frames["oos_raw"])
    path_val = _predict_sltp(frames["val_raw"], bundle, oof=True, device=device)
    path_oos = _predict_sltp(frames["oos_raw"], bundle, oof=False, device=device)
    path_val.to_csv(out_dir / "validation_predictions_2025_sltp_pathlabel.csv", index=False)
    path_oos.to_csv(out_dir / "oos_predictions_2026_sltp_pathlabel.csv", index=False)

    rows: list[dict[str, Any]] = []
    modes = ["agreement_same_side", "zigzag_primary_sltp_veto", "sltp_primary_zigzag_veto"]
    for mode in modes:
        for delta in THRESHOLD_DELTAS:
            for pthr in SLTP_THRESHOLDS:
                val_src = _ensemble_source(zig_val, path_val, oof=True, mode=mode, zigzag_delta=delta, sltp_threshold=pthr)
                oos_src = _ensemble_source(zig_oos, path_oos, oof=False, mode=mode, zigzag_delta=delta, sltp_threshold=pthr)
                val = _metrics(frames["val_raw"], val_src, oof=True, fee=fee, slip=slip, cost_mult=float(args.cost_mult), tp=float(args.tp), sl=float(args.sl))
                oos = _metrics(frames["oos_raw"], oos_src, oof=False, fee=fee, slip=slip, cost_mult=float(args.cost_mult), tp=float(args.tp), sl=float(args.sl))
                rows.append({
                    "mode": mode,
                    "zigzag_threshold_delta": float(delta),
                    "sltp_threshold": float(pthr),
                    "blend_weight_zigzag": np.nan,
                    "val_pnl": val["pnl"],
                    "val_mdd": val["mdd"],
                    "val_wr": val["wr"],
                    "val_trades": val["trades"],
                    "oos_pnl": oos["pnl"],
                    "oos_mdd": oos["mdd"],
                    "oos_wr": oos["wr"],
                    "oos_trades": oos["trades"],
                    "val_exit_reasons": val.get("exit_reasons", {}),
                    "oos_exit_reasons": oos.get("exit_reasons", {}),
                })
    for weight in BLEND_WEIGHTS:
        for threshold in SLTP_THRESHOLDS:
            val_src = _blend_source(zig_val, path_val, oof=True, weight_zigzag=weight, threshold=threshold)
            oos_src = _blend_source(zig_oos, path_oos, oof=False, weight_zigzag=weight, threshold=threshold)
            val = _metrics(frames["val_raw"], val_src, oof=True, fee=fee, slip=slip, cost_mult=float(args.cost_mult), tp=float(args.tp), sl=float(args.sl))
            oos = _metrics(frames["oos_raw"], oos_src, oof=False, fee=fee, slip=slip, cost_mult=float(args.cost_mult), tp=float(args.tp), sl=float(args.sl))
            rows.append({
                "mode": "probability_blend",
                "zigzag_threshold_delta": np.nan,
                "sltp_threshold": float(threshold),
                "blend_weight_zigzag": float(weight),
                "val_pnl": val["pnl"],
                "val_mdd": val["mdd"],
                "val_wr": val["wr"],
                "val_trades": val["trades"],
                "oos_pnl": oos["pnl"],
                "oos_mdd": oos["mdd"],
                "oos_wr": oos["wr"],
                "oos_trades": oos["trades"],
                "val_exit_reasons": val.get("exit_reasons", {}),
                "oos_exit_reasons": oos.get("exit_reasons", {}),
            })

    ranking = pd.DataFrame(rows).sort_values(["val_pnl", "val_wr"], ascending=False)
    ranking.to_csv(out_dir / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Post-hoc ensemble of zigzag-trained Omega1.2 practical baseline and SLTP-path-label Omega1.2 model. No retraining; validation selects ensemble rule, 2026 is fixed evaluation.",
        "inputs": {
            "zigzag_dir": str(ZIGZAG_DIR),
            "sltp_dir": str(SLTP_DIR),
            "practical_thresholds": PRACTICAL_EXPERT_THRESHOLDS,
            "tp": float(args.tp),
            "sl": float(args.sl),
            "cost_mult": float(args.cost_mult),
        },
        "ranking": rows,
        "artifacts": {
            "out_dir": str(out_dir),
            "ranking": str(out_dir / "ranking.csv"),
            "sltp_val_predictions": str(out_dir / "validation_predictions_2025_sltp_pathlabel.csv"),
            "sltp_oos_predictions": str(out_dir / "oos_predictions_2026_sltp_pathlabel.csv"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(ranking.head(20).to_string(index=False))
    print(json.dumps({"report": str(out_dir / "report.json"), "ranking": str(out_dir / "ranking.csv")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
