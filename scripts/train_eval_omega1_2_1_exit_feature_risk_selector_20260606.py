#!/usr/bin/env python3
from __future__ import annotations

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

import train_eval_omega1_2_1_exposure_selector_20260606 as base  # noqa: E402
import train_eval_omega1_2_mamba_sac_3head_feature_coordinator_20260604 as coord  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_exit_feature_risk_selector_20260606"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AGGRESSIVE_VAL = {"pnl": 100.54272942091158, "mdd": -10.677652697162888, "wr": 0.6363636363636364, "trades": 33}
AGGRESSIVE_OOS = {"pnl": 72.76004148106665, "mdd": -8.108170708968387, "wr": 0.7222222222222222, "trades": 18}


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


def _exit_feature_source(frame: pd.DataFrame, *, oof: bool, device: torch.device) -> pd.DataFrame:
    bundle = coord._load_3head_payloads(base.BASE_DIR)
    base_cols = list(bundle["base_cols"])
    x = threehead._base_input(frame, base_cols)
    preds = {expert: threehead._predict_payload(bundle["models"][expert], x, device=device) for expert in ("bull", "bear", "chop")}
    route = coord.hard._route_id(frame)
    exit_prob = threehead._routed(preds, route, "exit", 2)
    prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"], errors="raise"),
            f"{prefix}_exit_p_hold_feature_only": exit_prob[:, 0],
            f"{prefix}_exit_p_exit_feature_only": exit_prob[:, 1],
            f"{prefix}_exit_edge_feature_only": exit_prob[:, 1] - exit_prob[:, 0],
        }
    )


def _feature_frame_with_exit(
    frame: pd.DataFrame,
    src: pd.DataFrame,
    dec: pd.DataFrame,
    prefix: str,
    *,
    oof: bool,
    device: torch.device,
) -> pd.DataFrame:
    out = base._feature_frame(frame, src, dec, prefix)
    exit_src = _exit_feature_source(frame, oof=oof, device=device)
    aligned = frame[["timestamp"]].merge(exit_src, on="timestamp", how="left", validate="one_to_one")
    if aligned.isna().any().any():
        bad = aligned.loc[aligned.isna().any(axis=1), "timestamp"].head(10).tolist()
        raise RuntimeError(f"exit feature alignment produced NaN: {bad}")
    out["threehead_exit_p_hold_feature_only"] = pd.to_numeric(aligned[f"{prefix}exit_p_hold_feature_only"], errors="raise").to_numpy(dtype=np.float64)
    out["threehead_exit_p_exit_feature_only"] = pd.to_numeric(aligned[f"{prefix}exit_p_exit_feature_only"], errors="raise").to_numpy(dtype=np.float64)
    out["threehead_exit_edge_feature_only"] = pd.to_numeric(aligned[f"{prefix}exit_edge_feature_only"], errors="raise").to_numpy(dtype=np.float64)
    bad = [c for c in out.columns if c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_") or c == "tp_sl_action_score"]
    if bad:
        raise RuntimeError(f"forbidden exit-feature selector columns: {bad}")
    return out


def _apply_compensated(dec: pd.DataFrame, active_idx: np.ndarray, *, scale: float, cap: float) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    if len(active_idx) == 0:
        return out
    base_notional = pd.to_numeric(out.loc[active_idx, "notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    new_notional = np.minimum(base_notional * float(scale), float(cap))
    ratio = new_notional / np.maximum(base_notional, 1.0e-12)
    out.loc[active_idx, "notional_exposure"] = new_notional
    out.loc[active_idx, "position_fraction"] = new_notional
    out.loc[active_idx, "take_profit"] = pd.to_numeric(out.loc[active_idx, "take_profit"], errors="raise").to_numpy(dtype=np.float64) * ratio
    out.loc[active_idx, "stop_loss"] = pd.to_numeric(out.loc[active_idx, "stop_loss"], errors="raise").to_numpy(dtype=np.float64) * ratio
    return out


def _apply_safe_selector(
    dec: pd.DataFrame,
    active_idx: np.ndarray,
    safe_scores: np.ndarray,
    threshold: float,
    *,
    safe_scale: float,
    safe_cap: float,
    risky_scale: float,
    risky_cap: float,
) -> tuple[pd.DataFrame, int]:
    out = _apply_compensated(dec, active_idx, scale=risky_scale, cap=risky_cap)
    safe = np.isfinite(safe_scores) & (safe_scores >= float(threshold))
    safe_idx = active_idx[safe]
    if len(safe_idx) == 0:
        return out, 0
    safe_dec = _apply_compensated(dec, safe_idx, scale=safe_scale, cap=safe_cap)
    for col in ("notional_exposure", "position_fraction", "take_profit", "stop_loss"):
        out.loc[safe_idx, col] = safe_dec.loc[safe_idx, col].to_numpy()
    return out, int(len(safe_idx))


def _fit_and_score(
    model_name: str,
    x_val_active: np.ndarray,
    y_win: np.ndarray,
    y_net: np.ndarray,
    val_active: np.ndarray,
    x_oos_active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    oof_scores, diag = base._fit_oof_scores(model_name, x_val_active, y_win, y_net, val_active, seed=260606)
    model = base._make_model(model_name, 260606)
    target = y_win if model_name.endswith("_win") else y_net
    model.fit(x_val_active, target)
    oos_scores = base._predict_score(model_name, model, x_oos_active)
    return oof_scores, oos_scores, diag


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_src, val_dec, val_prefix = base._build_split(frames, "validation")
    oos_frame, oos_src, oos_dec, oos_prefix = base._build_split(frames, "oos")
    val_x_all = _feature_frame_with_exit(val_frame, val_src, val_dec, val_prefix, oof=True, device=device)
    oos_x_all = _feature_frame_with_exit(oos_frame, oos_src, oos_dec, oos_prefix, oof=False, device=device)
    val_active = np.flatnonzero(omega._active(val_dec))
    oos_active = np.flatnonzero(omega._active(oos_dec))
    aggressive_val_dec = _apply_compensated(val_dec, val_active, scale=2.0, cap=0.90)
    y_win, y_net = base._candidate_labels(val_frame, aggressive_val_dec, val_active, fee=fee, slip=slip)
    x_val_active = val_x_all.iloc[val_active].to_numpy(dtype=np.float64)
    x_oos_active = oos_x_all.iloc[oos_active].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "device": str(device),
        "val_active_rows": int(len(val_active)),
        "oos_active_rows": int(len(oos_active)),
        "feature_count": int(val_x_all.shape[1]),
        "features": list(val_x_all.columns),
        "target": "aggressive baseline trade win/net labels; selector chooses which active signals stay aggressive",
    }
    score_sets: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {
        "exit_hold_raw": (
            pd.to_numeric(val_x_all.iloc[val_active]["threehead_exit_p_hold_feature_only"], errors="raise").to_numpy(dtype=np.float64),
            pd.to_numeric(oos_x_all.iloc[oos_active]["threehead_exit_p_hold_feature_only"], errors="raise").to_numpy(dtype=np.float64),
            {"oof_rows": int(len(val_active)), "oof_win_auc": None, "raw_feature": "threehead_exit_p_hold_feature_only"},
        ),
        "exit_edge_inverse_raw": (
            -pd.to_numeric(val_x_all.iloc[val_active]["threehead_exit_edge_feature_only"], errors="raise").to_numpy(dtype=np.float64),
            -pd.to_numeric(oos_x_all.iloc[oos_active]["threehead_exit_edge_feature_only"], errors="raise").to_numpy(dtype=np.float64),
            {"oof_rows": int(len(val_active)), "oof_win_auc": None, "raw_feature": "-threehead_exit_edge_feature_only"},
        ),
    }
    for model_name in ("hgb_win", "extra_win", "hgb_net"):
        score_sets[model_name] = _fit_and_score(model_name, x_val_active, y_win, y_net, val_active, x_oos_active)
    baseline_val_m = omega._metrics(val_frame, aggressive_val_dec, fee=fee, slip=slip, cost_mult=3.0)
    baseline_oos_m = omega._metrics(oos_frame, _apply_compensated(oos_dec, oos_active, scale=2.0, cap=0.90), fee=fee, slip=slip, cost_mult=3.0)
    rows.append(
        {
            "selector": "aggressive_baseline_all",
            "keep_frac": 1.0,
            "threshold": -np.inf,
            "safe_scale": 2.0,
            "safe_cap": 0.90,
            "risky_scale": 2.0,
            "risky_cap": 0.90,
            "val_safe_signals": int(len(val_active)),
            "oos_safe_signals": int(len(oos_active)),
            **base._metric_row("val", baseline_val_m),
            **base._metric_row("oos", baseline_oos_m),
        }
    )
    for selector, (val_scores, oos_scores, diag) in score_sets.items():
        valid_oof = val_scores[np.isfinite(val_scores)]
        diagnostics[selector] = diag
        if len(valid_oof) == 0:
            continue
        for keep_frac in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
            threshold = float(np.quantile(valid_oof, 1.0 - float(keep_frac)))
            for risky_scale, risky_cap in ((1.0, 0.405), (1.35, 0.55)):
                val_sel_dec, val_safe = _apply_safe_selector(
                    val_dec,
                    val_active,
                    val_scores,
                    threshold,
                    safe_scale=2.0,
                    safe_cap=0.90,
                    risky_scale=risky_scale,
                    risky_cap=risky_cap,
                )
                oos_sel_dec, oos_safe = _apply_safe_selector(
                    oos_dec,
                    oos_active,
                    oos_scores,
                    threshold,
                    safe_scale=2.0,
                    safe_cap=0.90,
                    risky_scale=risky_scale,
                    risky_cap=risky_cap,
                )
                val_m = omega._metrics(val_frame, val_sel_dec, fee=fee, slip=slip, cost_mult=3.0)
                oos_m = omega._metrics(oos_frame, oos_sel_dec, fee=fee, slip=slip, cost_mult=3.0)
                row = {
                    "selector": selector,
                    "keep_frac": float(keep_frac),
                    "threshold": threshold,
                    "safe_scale": 2.0,
                    "safe_cap": 0.90,
                    "risky_scale": float(risky_scale),
                    "risky_cap": float(risky_cap),
                    "val_safe_signals": int(val_safe),
                    "oos_safe_signals": int(oos_safe),
                }
                row.update(base._metric_row("val", val_m))
                row.update(base._metric_row("oos", oos_m))
                rows.append(row)
    ranking = pd.DataFrame(rows)
    ranking["val_delta_pnl"] = ranking["val_pnl"] - AGGRESSIVE_VAL["pnl"]
    ranking["oos_delta_pnl"] = ranking["oos_pnl"] - AGGRESSIVE_OOS["pnl"]
    ranking["val_delta_mdd"] = ranking["val_mdd"] - AGGRESSIVE_VAL["mdd"]
    ranking["oos_delta_mdd"] = ranking["oos_mdd"] - AGGRESSIVE_OOS["mdd"]
    ranking["score"] = ranking["oos_pnl"] + 0.75 * ranking["val_pnl"] + 0.35 * ranking["oos_mdd"] + 0.35 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl", "score"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "exit_feature_risk_selector_ranking.csv", index=False)
    promotable = ranking[
        (ranking["oos_pnl"] > AGGRESSIVE_OOS["pnl"])
        & (ranking["val_pnl"] > AGGRESSIVE_VAL["pnl"])
        & (ranking["oos_mdd"] >= AGGRESSIVE_OOS["mdd"] * 1.25)
        & (ranking["val_mdd"] >= AGGRESSIVE_VAL["mdd"] * 1.25)
    ].copy()
    promotable.to_csv(OUT_DIR / "exit_feature_risk_selector_promotable.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "baseline": {
            "model_id": "omega1_2_1_aggressive_compensated_scale200_cap090",
            "validation": AGGRESSIVE_VAL,
            "oos": AGGRESSIVE_OOS,
        },
        "method": "EXIT Head is feature-only. It never owns immediate exits. Selector uses exit risk signals to decide whether each parent active signal remains aggressive scale=2.0/cap=0.90 or is de-risked to base/balanced exposure.",
        "diagnostics": diagnostics,
        "best": ranking.iloc[0].to_dict(),
        "promotable": promotable.head(10).to_dict(orient="records"),
        "top10": ranking.head(10).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "exit_feature_risk_selector_ranking.csv"),
            "promotable": str(OUT_DIR / "exit_feature_risk_selector_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "best": report["best"], "promotable_count": int(len(promotable))}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
