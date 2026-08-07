#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260616 as exp  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_8b_paper_fixes_20260617"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BUNDLE_PATH = (
    ROOT
    / "data/ensemble/supervised/omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260616/numeric_cash_sleeve.joblib"
)


@dataclass(frozen=True)
class Variant:
    name: str
    paper_motive: str
    ev_extra_offset: float
    ev_min_delta: float
    utility_min: float | None
    margin_min: float | None
    support_min_fraction: float | None
    support_max_z: float | None
    router_conf_min: float
    fallback_entry_cap_ratio: float | None


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


def _reason_count(reasons: Any, key: str) -> int:
    if not isinstance(reasons, dict):
        return 0
    return int(reasons.get(key, 0) or 0)


def _support_pass(x: pd.DataFrame, profile: dict[str, Any], *, min_fraction: float, max_z: float) -> tuple[np.ndarray, dict[str, Any]]:
    low = pd.Series({str(k): float(v) for k, v in dict(profile["low"]).items()})
    high = pd.Series({str(k): float(v) for k, v in dict(profile["high"]).items()})
    median = pd.Series({str(k): float(v) for k, v in dict(profile["median"]).items()})
    iqr = pd.Series({str(k): max(float(v), 1.0e-8) for k, v in dict(profile["iqr"]).items()})
    missing = [c for c in x.columns if c not in low.index or c not in high.index or c not in median.index or c not in iqr.index]
    if missing:
        raise RuntimeError(f"support profile missing features: {missing[:20]}")
    low = low.reindex(x.columns)
    high = high.reindex(x.columns)
    median = median.reindex(x.columns)
    iqr = iqr.reindex(x.columns)
    in_band = x.ge(low, axis=1) & x.le(high, axis=1)
    fraction = in_band.mean(axis=1).to_numpy(dtype=np.float64)
    robust_z = ((x - median) / iqr).abs().max(axis=1).to_numpy(dtype=np.float64)
    passed = (fraction >= float(min_fraction)) & (robust_z <= float(max_z))
    return passed, {
        "pass_rows": int(passed.sum()),
        "total_rows": int(len(passed)),
        "pass_rate": float(passed.mean()) if len(passed) else 0.0,
        "mean_support_fraction": float(np.mean(fraction)) if len(fraction) else 0.0,
        "p01_support_fraction": float(np.quantile(fraction, 0.01)) if len(fraction) else 0.0,
        "p99_robust_abs_z": float(np.quantile(robust_z, 0.99)) if len(robust_z) else 0.0,
        "max_robust_abs_z": float(np.max(robust_z)) if len(robust_z) else 0.0,
    }


def _predict_actions(
    x: pd.DataFrame,
    bundle: dict[str, Any],
    variant: Variant,
    *,
    base_fallback_entries: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cols = list(bundle["feature_cols"])
    if list(x.columns) != cols:
        raise RuntimeError("feature columns mismatch against Omega1.2.8b bundle")
    arr = x.to_numpy(dtype=np.float64)
    long_ev = (
        np.asarray(bundle["long_model"].predict(arr), dtype=np.float64)
        - float(bundle["calibration"]["long_abs_residual_offset"])
        - float(variant.ev_extra_offset)
    )
    short_ev = (
        np.asarray(bundle["short_model"].predict(arr), dtype=np.float64)
        - float(bundle["calibration"]["short_abs_residual_offset"])
        - float(variant.ev_extra_offset)
    )
    long_utility = (
        np.asarray(bundle["utility_long_model"].predict(arr), dtype=np.float64)
        - float(bundle["calibration"]["long_utility_abs_residual_offset"])
    )
    short_utility = (
        np.asarray(bundle["utility_short_model"].predict(arr), dtype=np.float64)
        - float(bundle["calibration"]["short_utility_abs_residual_offset"])
    )
    ev_min = float(bundle["ev_min"]) + float(variant.ev_min_delta)
    best_long = long_ev >= short_ev
    best_ev = np.where(best_long, long_ev, short_ev)
    action = np.where(best_ev > ev_min, np.where(best_long, sleeve.ACTION_LONG, sleeve.ACTION_SHORT), sleeve.ACTION_CASH)
    utility_min = float(bundle["utility_min"] if variant.utility_min is None else variant.utility_min)
    margin_min = float(bundle["margin_min"] if variant.margin_min is None else variant.margin_min)
    long_ok = (action == sleeve.ACTION_LONG) & (long_utility > utility_min) & ((long_utility - short_utility) >= margin_min)
    short_ok = (action == sleeve.ACTION_SHORT) & (short_utility > utility_min) & ((short_utility - long_utility) >= margin_min)
    action = np.where(long_ok | short_ok, action, sleeve.ACTION_CASH).astype(np.int64)
    if variant.support_min_fraction is not None or variant.support_max_z is not None:
        support_min = float(bundle["support_profile"]["min_fraction_in_support"] if variant.support_min_fraction is None else variant.support_min_fraction)
        support_z = float(bundle["support_profile"]["max_robust_abs_z"] if variant.support_max_z is None else variant.support_max_z)
        support_pass, support_diag = _support_pass(x, dict(bundle["support_profile"]), min_fraction=support_min, max_z=support_z)
        action = np.where(support_pass, action, sleeve.ACTION_CASH).astype(np.int64)
    else:
        support_diag = {"enabled": False}
    if variant.router_conf_min > 0.0:
        router_conf = x["router_confidence"].to_numpy(dtype=np.float64)
        action = np.where(router_conf >= float(variant.router_conf_min), action, sleeve.ACTION_CASH).astype(np.int64)
    conf = np.clip((best_ev - ev_min) / 0.02, 0.0, 1.0)
    conf = np.where(action != sleeve.ACTION_CASH, conf, 0.0).astype(np.float64)
    if variant.fallback_entry_cap_ratio is not None and base_fallback_entries is not None:
        active_idx = np.flatnonzero(action != sleeve.ACTION_CASH)
        cap = int(np.floor(float(base_fallback_entries) * float(variant.fallback_entry_cap_ratio)))
        if cap >= 0 and len(active_idx) > cap:
            score = np.maximum(long_ev, short_ev)[active_idx]
            keep = active_idx[np.argsort(score)[::-1][:cap]]
            capped_action = np.zeros_like(action)
            capped_conf = np.zeros_like(conf)
            capped_action[keep] = action[keep]
            capped_conf[keep] = conf[keep]
            action, conf = capped_action, capped_conf
    diag = {
        "ev_min": ev_min,
        "utility_min": utility_min,
        "margin_min": margin_min,
        "candidate_action_rows": int(np.count_nonzero(action)),
        "long_action_rows": int(np.count_nonzero(action == sleeve.ACTION_LONG)),
        "short_action_rows": int(np.count_nonzero(action == sleeve.ACTION_SHORT)),
        "support": support_diag,
    }
    return action.astype(np.int64), conf.astype(np.float64), diag


def _row(name: str, motive: str, val_m: dict[str, Any], oos_m: dict[str, Any], base_val: dict[str, Any], base_oos: dict[str, Any], diag: dict[str, Any]) -> dict[str, Any]:
    row = {"variant": name, "paper_motive": motive}
    row.update(sleeve._metric_row("val", val_m))
    row.update(sleeve._metric_row("oos", oos_m))
    row["val_delta_pnl"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_pnl"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    row["val_fallback_stop_loss"] = _reason_count(row["val_reasons"], "fallback_stop_loss")
    row["oos_fallback_stop_loss"] = _reason_count(row["oos_reasons"], "fallback_stop_loss")
    row["val_fallback_primary_takeover"] = _reason_count(row["val_reasons"], "fallback_primary_takeover")
    row["oos_fallback_primary_takeover"] = _reason_count(row["oos_reasons"], "fallback_primary_takeover")
    row["score"] = (
        row["oos_delta_pnl"]
        + 0.5 * row["val_delta_pnl"]
        + 0.25 * row["oos_mdd"]
        - 1.5 * row["oos_fallback_stop_loss"]
        - 0.5 * row["oos_fallback_primary_takeover"]
    )
    row["diag"] = diag
    return row


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(BUNDLE_PATH)
    val_payload, oos_payload, meta = exp._build_payloads()
    x_val = val_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)[list(bundle["feature_cols"])]
    x_oos = oos_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)[list(bundle["feature_cols"])]
    fee = float(meta["fee"])
    slip = float(meta["slip"])
    base_val_raw = omega._metrics(val_payload["frame"], val_payload["dec"], fee=fee, slip=slip, cost_mult=3.0)
    base_oos_raw = omega._metrics(oos_payload["frame"], oos_payload["dec"], fee=fee, slip=slip, cost_mult=3.0)
    base_val = {**base_val_raw, "primary_entries": base_val_raw["long_entries"] + base_val_raw["short_entries"], "fallback_entries": 0, "primary_takeovers": 0, "exit_reasons": base_val_raw.get("exit_reasons", {})}
    base_oos = {**base_oos_raw, "primary_entries": base_oos_raw["long_entries"] + base_oos_raw["short_entries"], "fallback_entries": 0, "primary_takeovers": 0, "exit_reasons": base_oos_raw.get("exit_reasons", {})}
    variants = [
        Variant("live_contract_support_gate", "CQL/OOD support blocking + existing conformal lower-bound", 0.0, 0.0, None, None, 0.92, 8.0, 0.0, None),
        Variant("spci_stricter_lower_bound", "SPCI-style more conservative residual lower-bound", 0.0015, 0.0, None, None, 0.92, 8.0, 0.0, None),
        Variant("cql_strict_support", "CQL-style stricter behavior-support filter", 0.0, 0.0, None, None, 0.95, 6.0, 0.0, None),
        Variant("cql_very_strict_support", "CQL-style high-confidence in-support only", 0.0, 0.0, None, None, 0.98, 4.0, 0.0, None),
        Variant("utility_margin_conservative", "Conservative utility agreement and margin filter", 0.0, 0.0, 0.001, 0.001, 0.92, 8.0, 0.0, None),
        Variant("mmdrex_router_confidence", "MM-DREX-inspired dynamic router confidence gate", 0.0, 0.0, None, None, 0.92, 8.0, 0.55, None),
        Variant("combined_spci_cql_router", "Combined SPCI lower-bound + CQL support + router confidence", 0.001, 0.001, 0.001, 0.001, 0.95, 6.0, 0.50, None),
    ]
    rows: list[dict[str, Any]] = [
        _row("parent_only_baseline", "control: no cash fallback", base_val, base_oos, base_val, base_oos, {"candidate_action_rows": 0})
    ]
    base_fallback_entries = None
    for variant in variants:
        val_a, val_c, val_diag = _predict_actions(x_val, bundle, variant, base_fallback_entries=base_fallback_entries)
        val_m = sleeve._metrics_with_fallback(val_payload["frame"], val_payload["dec"], exp.RISK, val_a, val_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
        if variant.name == "live_contract_support_gate":
            base_fallback_entries = int(val_m["fallback_entries"])
        oos_a, oos_c, oos_diag = _predict_actions(x_oos, bundle, variant, base_fallback_entries=base_fallback_entries)
        oos_m = sleeve._metrics_with_fallback(oos_payload["frame"], oos_payload["dec"], exp.RISK, oos_a, oos_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
        rows.append(_row(variant.name, variant.paper_motive, val_m, oos_m, base_val, base_oos, {"validation": val_diag, "oos": oos_diag}))
    ranking = pd.DataFrame(rows).sort_values(["score", "oos_delta_pnl", "oos_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "paper_fix_ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "bundle": str(BUNDLE_PATH),
        "method": "Paper-inspired conservative gates evaluated on existing Omega1.2.8b live bundle without refitting.",
        "papers": {
            "SPCI": "https://huggingface.co/papers/2212.03463",
            "CopulaCPTS": "https://huggingface.co/papers/2212.03281",
            "C-CQL": "https://huggingface.co/papers/2301.01298",
            "MM-DREX": "https://huggingface.co/papers/2509.05080",
            "TimeRFT": "https://huggingface.co/papers/2605.00015",
        },
        "baseline": {"validation": base_val, "oos": base_oos},
        "ranking": ranking.to_dict(orient="records"),
        "artifacts": {"ranking": str(OUT_DIR / "paper_fix_ranking.csv"), "report": str(OUT_DIR / "report.json")},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top": ranking.head(5).to_dict(orient="records")}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
