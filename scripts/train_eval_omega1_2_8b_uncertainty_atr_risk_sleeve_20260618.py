#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_8b_regime_threshold_cash_sleeve_20260618 as threshold_exp  # noqa: E402
import train_eval_omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260616 as base8b  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_8b_uncertainty_atr_risk_sleeve_20260618"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


@dataclass(frozen=True)
class RiskCfg:
    name: str
    min_notional_scale: float
    max_notional_scale: float
    atr_low: float
    atr_high: float
    tp_atr_power: float
    sl_atr_power: float


RISK_CFGS = (
    RiskCfg("static_control", 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
    RiskCfg("unc_n040_100_atr075_125", 0.40, 1.00, 0.75, 1.25, 1.0, 1.0),
    RiskCfg("unc_n050_110_atr075_135", 0.50, 1.10, 0.75, 1.35, 1.0, 1.0),
    RiskCfg("unc_n060_120_atr080_140", 0.60, 1.20, 0.80, 1.40, 1.0, 1.0),
    RiskCfg("unc_n040_100_tp100_sl060", 0.40, 1.00, 0.75, 1.35, 1.0, 0.6),
    RiskCfg("unc_n050_100_tp080_sl120", 0.50, 1.00, 0.80, 1.40, 0.8, 1.2),
    RiskCfg("unc_n030_090_atr090_150", 0.30, 0.90, 0.90, 1.50, 1.0, 1.0),
)


def _risk_from_bundle(bundle: dict[str, Any]) -> sleeve.FallbackRisk:
    payload = dict(bundle["risk"])
    return sleeve.FallbackRisk(
        str(payload["name"]),
        float(payload["take_profit"]),
        float(payload["stop_loss"]),
        float(payload["notional"]),
        float(payload["leverage"]),
        int(payload["max_hold_bars"]),
    )


def _dynamic_risk_arrays(x: pd.DataFrame, base: sleeve.FallbackRisk, cfg: RiskCfg, train_atr_median: float) -> dict[str, np.ndarray]:
    cash_conf = x["dir_p_cash"].to_numpy(dtype=np.float64)
    regime_conf = x["router_confidence"].to_numpy(dtype=np.float64)
    uncertainty = np.clip(1.0 - cash_conf * regime_conf, 0.0, 1.0)
    notional_scale = float(cfg.min_notional_scale) + (float(cfg.max_notional_scale) - float(cfg.min_notional_scale)) * uncertainty
    atr_ratio = x["atr14_pct"].to_numpy(dtype=np.float64) / max(float(train_atr_median), 1.0e-8)
    atr_scalar = np.clip(atr_ratio, float(cfg.atr_low), float(cfg.atr_high))
    tp_scalar = np.power(atr_scalar, float(cfg.tp_atr_power))
    sl_scalar = np.power(atr_scalar, float(cfg.sl_atr_power))
    return {
        "take_profit": np.full(len(x), float(base.take_profit), dtype=np.float64) * tp_scalar,
        "stop_loss": np.full(len(x), float(base.stop_loss), dtype=np.float64) * sl_scalar,
        "notional": np.full(len(x), float(base.notional), dtype=np.float64) * notional_scale,
        "leverage": np.full(len(x), float(base.leverage), dtype=np.float64),
        "max_hold_bars": np.full(len(x), int(base.max_hold_bars), dtype=np.int64),
        "uncertainty": uncertainty,
        "atr_scalar": atr_scalar,
        "notional_scale": notional_scale,
    }


def _open_dynamic(
    cash: float,
    arrays: dict[str, np.ndarray],
    i: int,
    side: int,
    dyn: dict[str, np.ndarray],
    fee_eff: float,
    slip_eff: float,
) -> tuple[float, sleeve.Position, bool]:
    filled, entry_px, entry_fee, _route = omega._try_execution(arrays, int(i), int(side), entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled:
        return cash, sleeve.Position(), False
    notional = float(dyn["notional"][i])
    if notional <= 0.0:
        return cash, sleeve.Position(), False
    entry_equity = cash
    cash -= cash * float(entry_fee) * notional
    return (
        cash,
        sleeve.Position(
            sleeve="fallback",
            side=int(side),
            entry_price=float(entry_px),
            entry_i=int(i),
            entry_equity=float(entry_equity),
            notional=notional,
            leverage=float(dyn["leverage"][i]),
            take_profit=float(dyn["take_profit"][i]),
            stop_loss=abs(float(dyn["stop_loss"][i])),
            max_hold_bars=int(dyn["max_hold_bars"][i]),
        ),
        True,
    )


def _metrics_with_dynamic_fallback(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    base_risk: sleeve.FallbackRisk,
    dyn: dict[str, np.ndarray],
    fallback_action: np.ndarray,
    fallback_conf: np.ndarray,
    threshold: float,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> dict[str, Any]:
    arrays = sleeve._arrays(frame)
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = peak = 1.0
    mdd = 0.0
    pos = sleeve.Position()
    trades = wins = 0
    primary_entries = fallback_entries = long_entries = short_entries = 0
    primary_takeovers = 0
    reasons: dict[str, int] = {}
    fallback_notional: list[float] = []
    fallback_tp: list[float] = []
    fallback_sl: list[float] = []
    fallback_unc: list[float] = []

    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            px = float(arrays["close"][i])
            raw = (
                (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1.0e-12)
                if pos.side > 0
                else (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1.0e-12)
            )
            unreal = raw * pos.notional
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)
            reason = ""
            if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                reason = "take_profit"
            elif pos.stop_loss > 0.0 and unreal <= -abs(pos.stop_loss):
                reason = "stop_loss"
            elif pos.max_hold_bars > 0 and int(i) - int(pos.entry_i) >= pos.max_hold_bars:
                reason = "max_hold"
            elif pos.sleeve == "fallback" and bool(active[i]):
                reason = "primary_takeover"
                primary_takeovers += 1
            if reason:
                cash, win = sleeve._close_position(cash, arrays, pos, i, fee_eff, slip_eff)
                trades += 1
                wins += int(win)
                reasons[f"{pos.sleeve}_{reason}"] = reasons.get(f"{pos.sleeve}_{reason}", 0) + 1
                pos = sleeve.Position()
            else:
                continue

        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1.0e-12) - 1.0)
        if bool(active[i]):
            row = dec.iloc[int(i)]
            side = int(row.get("side", 0) or 0)
            if side != 0:
                cash, pos, entered = sleeve._open_position(cash, arrays, i, side, "primary", None, row, fee_eff, slip_eff)
                if entered:
                    primary_entries += 1
                    long_entries += int(side > 0)
                    short_entries += int(side < 0)
            continue

        action = int(fallback_action[int(i)]) if int(i) < len(fallback_action) else sleeve.ACTION_CASH
        conf = float(fallback_conf[int(i)]) if int(i) < len(fallback_conf) else 0.0
        if action not in (sleeve.ACTION_LONG, sleeve.ACTION_SHORT) or conf < float(threshold):
            continue
        side = 1 if action == sleeve.ACTION_LONG else -1
        cash, pos, entered = _open_dynamic(cash, arrays, i, side, dyn, fee_eff, slip_eff)
        if entered:
            fallback_entries += 1
            long_entries += int(side > 0)
            short_entries += int(side < 0)
            fallback_notional.append(float(pos.notional))
            fallback_tp.append(float(pos.take_profit))
            fallback_sl.append(float(pos.stop_loss))
            fallback_unc.append(float(dyn["uncertainty"][i]))

    if pos.side != 0:
        cash, win = sleeve._close_position(cash, arrays, pos, len(frame) - 1, fee_eff, slip_eff)
        trades += 1
        wins += int(win)
        reasons[f"{pos.sleeve}_forced_end"] = reasons.get(f"{pos.sleeve}_forced_end", 0) + 1

    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "primary_entries": int(primary_entries),
        "fallback_entries": int(fallback_entries),
        "primary_takeovers": int(primary_takeovers),
        "exit_reasons": reasons,
        "avg_fallback_notional": float(np.mean(fallback_notional)) if fallback_notional else 0.0,
        "avg_fallback_tp": float(np.mean(fallback_tp)) if fallback_tp else 0.0,
        "avg_fallback_sl": float(np.mean(fallback_sl)) if fallback_sl else 0.0,
        "avg_fallback_uncertainty": float(np.mean(fallback_unc)) if fallback_unc else 0.0,
    }


def _metric_row(candidate: str, cfg: RiskCfg, val_m: dict[str, Any], oos_m: dict[str, Any], base_val: dict[str, Any], base_oos: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, **asdict(cfg)}
    row.update(sleeve._metric_row("val", val_m))
    row.update(sleeve._metric_row("oos", oos_m))
    for prefix, metrics in (("val", val_m), ("oos", oos_m)):
        for key in ("avg_fallback_notional", "avg_fallback_tp", "avg_fallback_sl", "avg_fallback_uncertainty"):
            row[f"{prefix}_{key}"] = float(metrics.get(key, 0.0))
    row["val_delta_pnl"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_pnl"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    row["val_fallback_stop_loss"] = threshold_exp._reason_count(row["val_reasons"], "fallback_stop_loss")
    row["val_fallback_primary_takeover"] = threshold_exp._reason_count(row["val_reasons"], "fallback_primary_takeover")
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
    bundle = threshold_exp._load_bundle()
    print(json.dumps({"stage": "build_payloads", "model_id": MODEL_ID}, ensure_ascii=True), flush=True)
    val_payload, oos_payload, meta = base8b._build_payloads()
    feature_cols = list(bundle["feature_cols"])
    x_val = val_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)[feature_cols]
    x_oos = oos_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)[feature_cols]
    scores_val = threshold_exp._score_bundle(x_val, bundle)
    scores_oos = threshold_exp._score_bundle(x_oos, bundle)
    regimes_val = threshold_exp._route_regime(x_val)
    regimes_oos = threshold_exp._route_regime(x_oos)
    thresholds = {r: float(bundle["ev_min"]) for r in threshold_exp.REGIMES}
    val_action, val_conf = threshold_exp._actions(scores_val, regimes_val, thresholds, utility_min=float(bundle["utility_min"]), margin_min=float(bundle["margin_min"]))
    oos_action, oos_conf = threshold_exp._actions(scores_oos, regimes_oos, thresholds, utility_min=float(bundle["utility_min"]), margin_min=float(bundle["margin_min"]))
    base_risk = _risk_from_bundle(bundle)
    fee = float(meta["fee"])
    slip = float(meta["slip"])
    base_val_parent = omega._metrics(val_payload["frame"], val_payload["dec"], fee=fee, slip=slip, cost_mult=3.0)
    base_oos_parent = omega._metrics(oos_payload["frame"], oos_payload["dec"], fee=fee, slip=slip, cost_mult=3.0)
    base_val = {**base_val_parent, "primary_entries": base_val_parent["long_entries"] + base_val_parent["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}
    base_oos = {**base_oos_parent, "primary_entries": base_oos_parent["long_entries"] + base_oos_parent["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}
    train_atr_median = float(dict(bundle["support_profile"])["median"]["atr14_pct"])

    rows: list[dict[str, Any]] = []
    for cfg in RISK_CFGS:
        dyn_val = _dynamic_risk_arrays(x_val, base_risk, cfg, train_atr_median)
        dyn_oos = _dynamic_risk_arrays(x_oos, base_risk, cfg, train_atr_median)
        val_m = _metrics_with_dynamic_fallback(val_payload["frame"], val_payload["dec"], base_risk, dyn_val, val_action, val_conf, 0.0, fee=fee, slip=slip, cost_mult=3.0)
        oos_m = _metrics_with_dynamic_fallback(oos_payload["frame"], oos_payload["dec"], base_risk, dyn_oos, oos_action, oos_conf, 0.0, fee=fee, slip=slip, cost_mult=3.0)
        rows.append(_metric_row(cfg.name, cfg, val_m, oos_m, base_val, base_oos))

    ranking = pd.DataFrame(rows).sort_values(["selection_score_val_only", "val_delta_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "uncertainty_atr_risk_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    best_oos = ranking.sort_values(["oos_pnl", "oos_delta_pnl"], ascending=False).iloc[0].to_dict()
    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_uncertainty_atr_risk_eval",
        "method": "Replay live 8b fallback entries and change only fallback risk. Notional scales with 1 - direction_cash_probability * router_confidence. TP/SL scale with ATR ratio from validation support median. Validation-only selection; OOS diagnostic only.",
        "bundle_path": str(threshold_exp.BUNDLE_PATH),
        "base_risk": dict(bundle["risk"]),
        "train_atr_median": train_atr_median,
        "risk_configs": [asdict(c) for c in RISK_CFGS],
        "baseline_parent_only": {"validation": base_val, "oos": base_oos},
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "ranking": ranking.to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "uncertainty_atr_risk_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=threshold_exp._json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected, "best_oos": best_oos}, indent=2, ensure_ascii=True, default=threshold_exp._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
