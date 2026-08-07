#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_3head_parent_veto_overlay_20260618 as veto  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402


MODEL_ID = "omega1_2_quality_gate_alternatives_20260618"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _metric_row(
    candidate: str,
    family: str,
    val_m: dict[str, Any],
    oos_m: dict[str, Any],
    base_val: dict[str, Any],
    base_oos: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"candidate": candidate, "family": family, **(params or {})}
    row.update(sleeve._metric_row("val", {**val_m, "primary_entries": val_m["long_entries"] + val_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}))
    row.update(sleeve._metric_row("oos", {**oos_m, "primary_entries": oos_m["long_entries"] + oos_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}))
    row["val_delta_vs_current"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_vs_current"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    val_reasons = row["val_reasons"] if isinstance(row["val_reasons"], dict) else {}
    row["val_stop_loss"] = int(val_reasons.get("stop_loss", 0))
    row["val_take_profit"] = int(val_reasons.get("take_profit", 0))
    row["selection_score_val_only"] = (
        row["val_delta_vs_current"]
        + 10.0 * float(row["val_wr"])
        + 0.25 * float(row["val_mdd"])
        - 0.75 * float(row["val_stop_loss"])
        - 0.05 * max(0.0, float(row["val_trades"]) - 80.0)
    )
    return row


def _drop_rows(dec: pd.DataFrame, drop: np.ndarray) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    drop = np.asarray(drop, dtype=bool)
    out.loc[drop, "action"] = omega.ACTION_CASH
    out.loc[drop, "side"] = 0
    out.loc[drop, "notional_exposure"] = 0.0
    out.loc[drop, "position_fraction"] = 0.0
    out.loc[drop, "take_profit"] = 0.0
    out.loc[drop, "stop_loss"] = 0.0
    out.loc[drop, "max_hold_bars"] = 0
    out.loc[drop, "cooldown_bars"] = 0
    return out


def _quality_scaled_decision(
    dec: pd.DataFrame,
    x: pd.DataFrame,
    *,
    floor: float,
    min_scale: float,
    max_scale: float,
) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = omega._active(out)
    q = x["quality_for_action"].to_numpy(dtype=np.float64)
    keep = active & (q >= float(floor))
    scale = np.zeros(len(out), dtype=np.float64)
    denom = max(1.0 - float(floor), 1.0e-8)
    scale[keep] = float(min_scale) + np.clip((q[keep] - float(floor)) / denom, 0.0, 1.0) * (float(max_scale) - float(min_scale))
    drop = active & ~keep
    out = _drop_rows(out, drop)
    for col in ("notional_exposure", "position_fraction", "take_profit", "stop_loss"):
        vals = pd.to_numeric(out[col], errors="raise").to_numpy(dtype=np.float64)
        vals[keep] *= scale[keep]
        out[col] = vals
    out["quality_scaled_ratio"] = scale
    return out


def _combined_ev_decision(dec: pd.DataFrame, x: pd.DataFrame, *, ev_threshold: float) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = omega._active(out)
    side = pd.to_numeric(out["side"], errors="raise").to_numpy(dtype=np.int64)
    q = x["quality_for_action"].to_numpy(dtype=np.float64)
    p_long = x["dir_p_long"].to_numpy(dtype=np.float64)
    p_short = x["dir_p_short"].to_numpy(dtype=np.float64)
    p_side = np.where(side > 0, p_long, np.where(side < 0, p_short, 0.0))
    p_win = np.clip(p_side * q, 0.0, 1.0)
    tp = pd.to_numeric(out["take_profit"], errors="raise").to_numpy(dtype=np.float64)
    sl = np.abs(pd.to_numeric(out["stop_loss"], errors="raise").to_numpy(dtype=np.float64))
    ev = p_win * tp - (1.0 - p_win) * sl
    drop = active & (ev <= float(ev_threshold))
    out = _drop_rows(out, drop)
    out["combined_ev"] = ev
    return out


def _adaptive_threshold_decision(
    dec: pd.DataFrame,
    x: pd.DataFrame,
    *,
    percentile: float,
    window: int,
    initial_threshold: float,
) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = omega._active(out)
    q = pd.Series(x["quality_for_action"].to_numpy(dtype=np.float64))
    dyn = q.rolling(int(window), min_periods=50).quantile(float(percentile)).shift(1).fillna(float(initial_threshold)).to_numpy(dtype=np.float64)
    drop = active & (q.to_numpy(dtype=np.float64) < dyn)
    out = _drop_rows(out, drop)
    out["adaptive_quality_threshold"] = dyn
    return out


def _evaluate_decision(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    return omega._metrics(frame, dec, fee=fee, slip=slip, cost_mult=3.0)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "build_payloads", "model_id": MODEL_ID}, ensure_ascii=True), flush=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, _val_src, current_val_dec, raw_val_dec, val_x, *_ = veto._load_split(frames, "validation")
    oos_frame, _oos_src, current_oos_dec, raw_oos_dec, oos_x, *_ = veto._load_split(frames, "oos")
    if list(val_x.columns) != list(oos_x.columns):
        raise RuntimeError("feature column mismatch")

    current_val_m = _evaluate_decision(val_frame, current_val_dec, fee=fee, slip=slip)
    current_oos_m = _evaluate_decision(oos_frame, current_oos_dec, fee=fee, slip=slip)
    raw_val_m = _evaluate_decision(val_frame, raw_val_dec, fee=fee, slip=slip)
    raw_oos_m = _evaluate_decision(oos_frame, raw_oos_dec, fee=fee, slip=slip)
    rows: list[dict[str, Any]] = [
        _metric_row("current_quality_gate_parent", "baseline", current_val_m, current_oos_m, current_val_m, current_oos_m),
        _metric_row("raw_direction_no_quality_gate", "baseline", raw_val_m, raw_oos_m, current_val_m, current_oos_m),
    ]

    for floor in (0.50, 0.55, 0.60, 0.65, 0.70):
        for min_scale in (0.20, 0.35, 0.50):
            val_dec = _quality_scaled_decision(raw_val_dec, val_x, floor=floor, min_scale=min_scale, max_scale=1.0)
            oos_dec = _quality_scaled_decision(raw_oos_dec, oos_x, floor=floor, min_scale=min_scale, max_scale=1.0)
            val_m = _evaluate_decision(val_frame, val_dec, fee=fee, slip=slip)
            oos_m = _evaluate_decision(oos_frame, oos_dec, fee=fee, slip=slip)
            rows.append(_metric_row(
                f"quality_scaled_floor{floor:.2f}_min{min_scale:.2f}".replace(".", "p"),
                "quality_scaled_notional",
                val_m,
                oos_m,
                current_val_m,
                current_oos_m,
                {"quality_floor": floor, "min_scale": min_scale, "max_scale": 1.0},
            ))

    for ev_threshold in (-0.006, -0.004, -0.002, 0.0, 0.001, 0.002, 0.003):
        val_dec = _combined_ev_decision(raw_val_dec, val_x, ev_threshold=ev_threshold)
        oos_dec = _combined_ev_decision(raw_oos_dec, oos_x, ev_threshold=ev_threshold)
        val_m = _evaluate_decision(val_frame, val_dec, fee=fee, slip=slip)
        oos_m = _evaluate_decision(oos_frame, oos_dec, fee=fee, slip=slip)
        rows.append(_metric_row(
            f"combined_ev_thr{ev_threshold:.3f}".replace(".", "p").replace("-", "m"),
            "combined_ev",
            val_m,
            oos_m,
            current_val_m,
            current_oos_m,
            {"ev_threshold": ev_threshold},
        ))

    for percentile in (0.55, 0.60, 0.65, 0.70, 0.75):
        for window in (250, 500, 1000):
            val_dec = _adaptive_threshold_decision(raw_val_dec, val_x, percentile=percentile, window=window, initial_threshold=0.80)
            oos_dec = _adaptive_threshold_decision(raw_oos_dec, oos_x, percentile=percentile, window=window, initial_threshold=0.80)
            val_m = _evaluate_decision(val_frame, val_dec, fee=fee, slip=slip)
            oos_m = _evaluate_decision(oos_frame, oos_dec, fee=fee, slip=slip)
            rows.append(_metric_row(
                f"adaptive_quality_p{percentile:.2f}_w{window}".replace(".", "p"),
                "adaptive_threshold",
                val_m,
                oos_m,
                current_val_m,
                current_oos_m,
                {"percentile": percentile, "window": window},
            ))

    print(json.dumps({"stage": "label_winprob_candidates"}, ensure_ascii=True), flush=True)
    labels = veto._label_candidates(val_frame, raw_val_dec, fee=fee, slip=slip, cost_mult=3.0)
    win_y = labels["exit_reason"].astype(str).eq("take_profit").to_numpy(dtype=np.int64)
    win_val_prob, win_oos_prob, win_diag, win_model = veto._fit_predict_veto(val_x, oos_x, labels, win_y, seed=618900)
    for threshold in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        val_dec = veto._apply_veto(raw_val_dec, win_val_prob, threshold)
        oos_dec = veto._apply_veto(raw_oos_dec, win_oos_prob, threshold)
        val_m = _evaluate_decision(val_frame, val_dec, fee=fee, slip=slip)
        oos_m = _evaluate_decision(oos_frame, oos_dec, fee=fee, slip=slip)
        rows.append(_metric_row(
            f"winprob_meta_thr{threshold:.2f}".replace(".", "p"),
            "win_probability_meta",
            val_m,
            oos_m,
            current_val_m,
            current_oos_m,
            {"winprob_threshold": threshold},
        ))

    ranking = pd.DataFrame(rows).sort_values(["selection_score_val_only", "val_delta_vs_current", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "quality_gate_alternatives_ranking.csv", index=False)
    candidates = ranking[ranking["family"].ne("baseline")].copy()
    selected = candidates.iloc[0].to_dict()
    best_oos = candidates.sort_values(["oos_pnl", "oos_delta_vs_current"], ascending=False).iloc[0].to_dict()
    best_by_family = (
        candidates.sort_values(["family", "selection_score_val_only", "val_delta_vs_current"], ascending=[True, False, False])
        .groupby("family", as_index=False)
        .head(1)
        .to_dict(orient="records")
    )
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "win_probability_model": win_model,
            "feature_cols": list(val_x.columns),
            "win_probability_diag": win_diag,
        },
        OUT_DIR / "win_probability_meta_model.joblib",
    )
    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_quality_gate_alternative_eval",
        "method": "Evaluate quality-threshold alternatives over the existing 3-head parent predictions: quality-scaled notional, combined EV gate, adaptive threshold, and TP-hit win-probability meta gate. Current quality gate is the baseline; validation-only selection, OOS diagnostic only.",
        "current_quality_gate_parent": {"validation": current_val_m, "oos": current_oos_m},
        "raw_direction_no_quality_gate": {"validation": raw_val_m, "oos": raw_oos_m},
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "best_by_family_validation": best_by_family,
        "top30": ranking.head(30).to_dict(orient="records"),
        "diagnostics": {
            "feature_count": int(val_x.shape[1]),
            "features": list(val_x.columns),
            "win_probability": win_diag,
            "candidate_labels": {
                "rows": int(len(labels)),
                "tp_hit_rate": float(win_y.mean()),
                "exit_reason_counts": labels["exit_reason"].value_counts().sort_index().to_dict(),
            },
        },
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "quality_gate_alternatives_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
            "win_probability_model": str(OUT_DIR / "win_probability_meta_model.joblib"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected, "best_oos": best_oos}, indent=2, ensure_ascii=True, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
