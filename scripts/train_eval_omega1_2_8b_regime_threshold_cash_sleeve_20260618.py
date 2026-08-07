#!/usr/bin/env python3
from __future__ import annotations

import itertools
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

import train_eval_omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260616 as base8b  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_8b_regime_threshold_cash_sleeve_20260618"
BUNDLE_PATH = (
    ROOT
    / "data/ensemble/supervised/omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260618/numeric_cash_sleeve.joblib"
)
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
REGIMES = ("bull", "bear", "chop")
EV_GRID = (0.001, 0.002, 0.003, 0.004, 0.005, 0.006)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _load_bundle() -> dict[str, Any]:
    if not BUNDLE_PATH.exists():
        raise RuntimeError(f"missing live 8b cash sleeve bundle: {BUNDLE_PATH}")
    bundle = joblib.load(BUNDLE_PATH)
    required = {
        "long_model",
        "short_model",
        "utility_long_model",
        "utility_short_model",
        "feature_cols",
        "risk",
        "ev_min",
        "utility_min",
        "margin_min",
        "calibration",
    }
    missing = sorted(k for k in required if k not in bundle)
    if missing:
        raise RuntimeError(f"live 8b bundle missing keys: {missing}")
    return bundle


def _route_regime(x: pd.DataFrame) -> np.ndarray:
    cols = ["router_is_bull", "router_is_bear", "router_is_chop"]
    missing = [c for c in cols if c not in x.columns]
    if missing:
        raise RuntimeError(f"missing router columns for regime threshold test: {missing}")
    flags = x[cols].to_numpy(dtype=np.float64) > 0.5
    good = flags.sum(axis=1) == 1
    if not bool(good.all()):
        bad_idx = np.flatnonzero(~good)[:10].tolist()
        raise RuntimeError(f"router regime must be exactly one-hot; bad rows: {bad_idx}")
    out = np.empty(len(x), dtype=object)
    out[flags[:, 0]] = "bull"
    out[flags[:, 1]] = "bear"
    out[flags[:, 2]] = "chop"
    return out


def _support_pass(x: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:
    gate = dict(bundle.get("conservative_gate") or {})
    if not bool(gate.get("block_if_out_of_support", False)):
        return np.ones(len(x), dtype=bool)
    profile = dict(bundle.get("support_profile") or {})
    low = dict(profile.get("low") or {})
    high = dict(profile.get("high") or {})
    median = dict(profile.get("median") or {})
    iqr = dict(profile.get("iqr") or {})
    feature_cols = list(bundle["feature_cols"])
    missing = [c for c in feature_cols if c not in low or c not in high or c not in median or c not in iqr]
    if missing:
        raise RuntimeError(f"support profile missing feature bounds: {missing[:20]}")
    values = x[feature_cols].to_numpy(dtype=np.float64)
    lo = np.asarray([float(low[c]) for c in feature_cols], dtype=np.float64)
    hi = np.asarray([float(high[c]) for c in feature_cols], dtype=np.float64)
    med = np.asarray([float(median[c]) for c in feature_cols], dtype=np.float64)
    scale = np.maximum(np.asarray([float(iqr[c]) for c in feature_cols], dtype=np.float64), 1.0e-8)
    support_fraction = ((values >= lo) & (values <= hi)).mean(axis=1)
    max_robust_abs_z = np.abs((values - med) / scale).max(axis=1)
    return (support_fraction >= float(profile.get("min_fraction_in_support", 0.92))) & (
        max_robust_abs_z <= float(profile.get("max_robust_abs_z", 8.0))
    )


def _score_bundle(x: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    feature_cols = list(bundle["feature_cols"])
    if list(x.columns) != feature_cols:
        missing = [c for c in feature_cols if c not in x.columns]
        extra = [c for c in x.columns if c not in feature_cols]
        if missing or extra:
            raise RuntimeError(f"feature contract mismatch; missing={missing[:20]} extra={extra[:20]}")
        x = x[feature_cols]
    arr = x.to_numpy(dtype=np.float64)
    calibration = dict(bundle["calibration"])
    return {
        "long_ev": bundle["long_model"].predict(arr).astype(np.float64)
        - float(calibration.get("long_abs_residual_offset", 0.0) or 0.0),
        "short_ev": bundle["short_model"].predict(arr).astype(np.float64)
        - float(calibration.get("short_abs_residual_offset", 0.0) or 0.0),
        "long_utility": bundle["utility_long_model"].predict(arr).astype(np.float64)
        - float(calibration.get("long_utility_abs_residual_offset", 0.0) or 0.0),
        "short_utility": bundle["utility_short_model"].predict(arr).astype(np.float64)
        - float(calibration.get("short_utility_abs_residual_offset", 0.0) or 0.0),
        "support_pass": _support_pass(x, bundle),
    }


def _actions(
    scores: dict[str, np.ndarray],
    regimes: np.ndarray,
    ev_thresholds: dict[str, float],
    *,
    utility_min: float,
    margin_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    long_ev = scores["long_ev"]
    short_ev = scores["short_ev"]
    long_utility = scores["long_utility"]
    short_utility = scores["short_utility"]
    threshold = np.asarray([float(ev_thresholds[str(r)]) for r in regimes], dtype=np.float64)
    best_long = long_ev >= short_ev
    best = np.where(best_long, long_ev, short_ev)
    action = np.where(best > threshold, np.where(best_long, sleeve.ACTION_LONG, sleeve.ACTION_SHORT), sleeve.ACTION_CASH)
    long_ok = (action == sleeve.ACTION_LONG) & (long_utility > float(utility_min)) & (
        (long_utility - short_utility) >= float(margin_min)
    )
    short_ok = (action == sleeve.ACTION_SHORT) & (short_utility > float(utility_min)) & (
        (short_utility - long_utility) >= float(margin_min)
    )
    keep = (long_ok | short_ok) & scores["support_pass"]
    action = np.where(keep, action, sleeve.ACTION_CASH).astype(np.int64)
    conf = np.clip((best - threshold) / 0.02, 0.0, 1.0)
    conf = np.where(action == sleeve.ACTION_CASH, 0.0, conf).astype(np.float64)
    return action, conf


def _reason_count(reasons: Any, key: str) -> int:
    if not isinstance(reasons, dict):
        return 0
    return int(reasons.get(key, 0) or 0)


def _policy_counts(regimes: np.ndarray, action: np.ndarray) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for regime in REGIMES:
        mask = regimes == regime
        out[regime] = {
            "rows": int(mask.sum()),
            "long": int(((action == sleeve.ACTION_LONG) & mask).sum()),
            "short": int(((action == sleeve.ACTION_SHORT) & mask).sum()),
            "cash": int(((action == sleeve.ACTION_CASH) & mask).sum()),
        }
    return out


def _metric_row(
    candidate: str,
    thresholds: dict[str, float],
    val_m: dict[str, Any],
    oos_m: dict[str, Any],
    base_val: dict[str, Any],
    base_oos: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "candidate": candidate,
        "bull_ev_min": float(thresholds["bull"]),
        "bear_ev_min": float(thresholds["bear"]),
        "chop_ev_min": float(thresholds["chop"]),
    }
    row.update(sleeve._metric_row("val", val_m))
    row.update(sleeve._metric_row("oos", oos_m))
    row["val_delta_pnl"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_pnl"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    row["val_fallback_stop_loss"] = _reason_count(row["val_reasons"], "fallback_stop_loss")
    row["val_fallback_primary_takeover"] = _reason_count(row["val_reasons"], "fallback_primary_takeover")
    row["val_fallback_take_profit"] = _reason_count(row["val_reasons"], "fallback_take_profit")
    row["val_wr_drop_vs_baseline"] = max(float(base_val["wr"]) - float(row["val_wr"]), 0.0)
    row["val_fallback_stop_rate"] = float(row["val_fallback_stop_loss"] / max(int(row["val_fallback_entries"]), 1))
    row["selection_score_val_only"] = (
        float(row["val_delta_pnl"])
        + 0.04 * float(row["val_fallback_entries"])
        + 8.0 * float(row["val_wr"])
        + 0.20 * float(row["val_mdd"])
        - 1.50 * float(row["val_fallback_stop_loss"])
        - 0.50 * float(row["val_fallback_primary_takeover"])
        - 18.0 * float(row["val_wr_drop_vs_baseline"])
        - 6.0 * float(row["val_fallback_stop_rate"])
    )
    return row


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = _load_bundle()
    print(json.dumps({"stage": "build_payloads", "model_id": MODEL_ID}, ensure_ascii=True), flush=True)
    val_payload, oos_payload, meta = base8b._build_payloads()
    x_val = val_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_oos = oos_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_cols = list(bundle["feature_cols"])
    x_val = x_val[feature_cols]
    x_oos = x_oos[feature_cols]
    val_regime = _route_regime(x_val)
    oos_regime = _route_regime(x_oos)
    val_scores = _score_bundle(x_val, bundle)
    oos_scores = _score_bundle(x_oos, bundle)
    risk_payload = dict(bundle["risk"])
    risk = sleeve.FallbackRisk(
        str(risk_payload["name"]),
        float(risk_payload["take_profit"]),
        float(risk_payload["stop_loss"]),
        float(risk_payload["notional"]),
        float(risk_payload["leverage"]),
        int(risk_payload["max_hold_bars"]),
    )
    fee = float(meta["fee"])
    slip = float(meta["slip"])
    base_val_parent = omega._metrics(val_payload["frame"], val_payload["dec"], fee=fee, slip=slip, cost_mult=3.0)
    base_oos_parent = omega._metrics(oos_payload["frame"], oos_payload["dec"], fee=fee, slip=slip, cost_mult=3.0)
    base_val = {
        **base_val_parent,
        "primary_entries": base_val_parent["long_entries"] + base_val_parent["short_entries"],
        "fallback_entries": 0,
        "primary_takeovers": 0,
    }
    base_oos = {
        **base_oos_parent,
        "primary_entries": base_oos_parent["long_entries"] + base_oos_parent["short_entries"],
        "fallback_entries": 0,
        "primary_takeovers": 0,
    }
    utility_min = float(bundle["utility_min"])
    margin_min = float(bundle["margin_min"])
    rows: list[dict[str, Any]] = []
    telemetry: dict[str, Any] = {
        "validation_regime_rows": {r: int((val_regime == r).sum()) for r in REGIMES},
        "oos_regime_rows": {r: int((oos_regime == r).sum()) for r in REGIMES},
    }
    for combo in itertools.product(EV_GRID, repeat=3):
        thresholds = {regime: float(value) for regime, value in zip(REGIMES, combo)}
        val_a, val_c = _actions(val_scores, val_regime, thresholds, utility_min=utility_min, margin_min=margin_min)
        oos_a, oos_c = _actions(oos_scores, oos_regime, thresholds, utility_min=utility_min, margin_min=margin_min)
        val_m = sleeve._metrics_with_fallback(
            val_payload["frame"], val_payload["dec"], risk, val_a, val_c, 0.0, fee=fee, slip=slip, cost_mult=3.0
        )
        oos_m = sleeve._metrics_with_fallback(
            oos_payload["frame"], oos_payload["dec"], risk, oos_a, oos_c, 0.0, fee=fee, slip=slip, cost_mult=3.0
        )
        name = "regime_ev_b{bull:.3f}_r{bear:.3f}_c{chop:.3f}".format(**thresholds)
        rows.append(_metric_row(name, thresholds, val_m, oos_m, base_val, base_oos))

    ranking = pd.DataFrame(rows).sort_values(
        ["selection_score_val_only", "val_delta_pnl", "val_pnl"], ascending=False
    ).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "regime_threshold_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    best_oos = ranking.sort_values(["oos_pnl", "oos_delta_pnl"], ascending=False).iloc[0].to_dict()
    selected_thresholds = {
        "bull": float(selected["bull_ev_min"]),
        "bear": float(selected["bear_ev_min"]),
        "chop": float(selected["chop_ev_min"]),
    }
    val_selected_a, _ = _actions(val_scores, val_regime, selected_thresholds, utility_min=utility_min, margin_min=margin_min)
    oos_selected_a, _ = _actions(oos_scores, oos_regime, selected_thresholds, utility_min=utility_min, margin_min=margin_min)
    telemetry["selected_policy_counts"] = {
        "validation": _policy_counts(val_regime, val_selected_a),
        "oos": _policy_counts(oos_regime, oos_selected_a),
    }
    global_thresholds = {r: float(bundle["ev_min"]) for r in REGIMES}
    val_global_a, val_global_c = _actions(val_scores, val_regime, global_thresholds, utility_min=utility_min, margin_min=margin_min)
    oos_global_a, oos_global_c = _actions(oos_scores, oos_regime, global_thresholds, utility_min=utility_min, margin_min=margin_min)
    global_val = sleeve._metrics_with_fallback(
        val_payload["frame"], val_payload["dec"], risk, val_global_a, val_global_c, 0.0, fee=fee, slip=slip, cost_mult=3.0
    )
    global_oos = sleeve._metrics_with_fallback(
        oos_payload["frame"], oos_payload["dec"], risk, oos_global_a, oos_global_c, 0.0, fee=fee, slip=slip, cost_mult=3.0
    )
    global_control = _metric_row("live_8b_global_threshold_replay", global_thresholds, global_val, global_oos, base_val, base_oos)
    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_regime_threshold_eval",
        "method": "Replay the live 8b global EV/utility models and select only bull/bear/chop-specific EV entry thresholds on validation. No sleeve retraining and no OOS selection.",
        "bundle_path": str(BUNDLE_PATH),
        "risk": dict(bundle["risk"]),
        "utility_min": utility_min,
        "margin_min": margin_min,
        "ev_grid": list(EV_GRID),
        "baseline_parent_only": {"validation": base_val, "oos": base_oos},
        "global_threshold_control": global_control,
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "top20": ranking.head(20).to_dict(orient="records"),
        "telemetry": telemetry,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "regime_threshold_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected, "global_control": global_control}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
