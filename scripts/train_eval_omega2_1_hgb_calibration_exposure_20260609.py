#!/usr/bin/env python3
from __future__ import annotations

import json
import math
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

import train_eval_omega1_2_1_cash_fallback_label_family_20260606 as label_family  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_full_retrain_cash_alpha43_20260608 as full  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
from freeze_omega2_1_hgb_12seed_cash_sleeve_20260609 import (  # noqa: E402
    BUNDLE_PATH,
    MODEL_ID as OMEGA21_MODEL_ID,
    RISK as BASE_RISK,
    SEEDS,
    _classes_to_proba,
    _model,
)


MODEL_ID = "omega2_1_hgb_calibration_exposure_levexp_20260609"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASELINE_OOS = {"pnl": 102.61148286407757, "mdd": -8.108170708968377, "wr": 0.6097560975609756, "trades": 41}
FORBIDDEN_PREFIXES = ("clean_regime4_", "regime4_pred_", "teacher_", "exit_head_")
FORBIDDEN_EXACT = {"tp_sl_action_score"}


@dataclass(frozen=True)
class FilterCfg:
    threshold: float
    margin_min: float
    agreement_min: float


@dataclass(frozen=True)
class DynamicScaleCfg:
    name: str
    base_scale: float
    high_scale: float
    high_conf: float
    high_margin: float
    high_agree: float
    cap: float


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


def _reject_forbidden(cols: list[str], tag: str) -> None:
    bad = [c for c in cols if c in FORBIDDEN_EXACT or any(c.startswith(p) for p in FORBIDDEN_PREFIXES)]
    if bad:
        raise RuntimeError(f"{tag} forbidden feature columns: {bad[:40]}")


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


def _predict_oof_and_full(
    x_val: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    x_oos: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    idx = np.flatnonzero(train_mask)
    val_stack: list[np.ndarray] = []
    val_pred_stack: list[np.ndarray] = []
    oos_stack: list[np.ndarray] = []
    oos_pred_stack: list[np.ndarray] = []
    folds: list[dict[str, Any]] = []
    for seed in SEEDS:
        val_p = np.zeros((len(x_val), 3), dtype=np.float64)
        n = len(idx)
        for fold_id, (train_frac, end_frac) in enumerate(((0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.00))):
            train_end = int(n * train_frac)
            val_end = int(n * end_frac)
            if train_end < 100 or val_end <= train_end:
                continue
            train_idx = idx[:train_end]
            val_idx = idx[train_end:val_end]
            if len(np.unique(y[train_idx])) < 2:
                continue
            model = _model(int(seed) + train_end)
            model.fit(x_val.iloc[train_idx].to_numpy(dtype=np.float64), y[train_idx])
            val_p[val_idx] = _classes_to_proba(model, model.predict_proba(x_val.iloc[val_idx].to_numpy(dtype=np.float64)))
            folds.append({"seed": int(seed), "fold": int(fold_id), "train_rows": int(len(train_idx)), "val_rows": int(len(val_idx))})
        full_model = _model(int(seed))
        full_model.fit(x_val.iloc[idx].to_numpy(dtype=np.float64), y[idx])
        oos_p = _classes_to_proba(full_model, full_model.predict_proba(x_oos.to_numpy(dtype=np.float64)))
        val_stack.append(val_p)
        val_pred_stack.append(np.argmax(val_p, axis=1))
        oos_stack.append(oos_p)
        oos_pred_stack.append(np.argmax(oos_p, axis=1))
    return (
        np.stack(val_stack).mean(axis=0),
        np.stack(oos_stack).mean(axis=0),
        np.stack(val_pred_stack),
        np.stack(oos_pred_stack),
        {"folds": folds, "oof_rows": int(np.count_nonzero(np.stack(val_stack).mean(axis=0).max(axis=1) > 0.0))},
    )


def _signal_stats(proba: np.ndarray, pred_stack: np.ndarray) -> dict[str, np.ndarray]:
    pred = np.argmax(proba, axis=1)
    conf = proba[np.arange(len(proba)), pred]
    sorted_p = np.sort(proba, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    agreement = (pred_stack == pred[None, :]).sum(axis=0).astype(np.float64) / max(float(pred_stack.shape[0]), 1.0)
    return {"pred": pred, "conf": conf, "margin": margin, "agreement": agreement}


def _filtered_action_conf(stats: dict[str, np.ndarray], cfg: FilterCfg) -> tuple[np.ndarray, np.ndarray]:
    action = stats["pred"].astype(np.int64).copy()
    conf = stats["conf"].astype(np.float64).copy()
    accept = (
        (action != sleeve.ACTION_CASH)
        & (conf >= float(cfg.threshold))
        & (stats["margin"] >= float(cfg.margin_min))
        & (stats["agreement"] >= float(cfg.agreement_min))
    )
    action[~accept] = sleeve.ACTION_CASH
    conf[~accept] = 0.0
    return action, conf


def _scaled_risk(scale: float, cap: float, name: str) -> sleeve.FallbackRisk:
    base_n = float(BASE_RISK.notional)
    new_n = min(base_n * float(scale), float(cap))
    ratio = new_n / max(base_n, 1.0e-12)
    return sleeve.FallbackRisk(
        name,
        float(BASE_RISK.take_profit) * ratio,
        float(BASE_RISK.stop_loss) * ratio,
        new_n,
        float(BASE_RISK.leverage),
        int(BASE_RISK.max_hold_bars),
    )


def _dynamic_risk(stats: dict[str, np.ndarray], i: int, cfg: DynamicScaleCfg) -> sleeve.FallbackRisk:
    high = (
        float(stats["conf"][i]) >= float(cfg.high_conf)
        and float(stats["margin"][i]) >= float(cfg.high_margin)
        and float(stats["agreement"][i]) >= float(cfg.high_agree)
    )
    scale = float(cfg.high_scale if high else cfg.base_scale)
    return _scaled_risk(scale, float(cfg.cap), cfg.name)


def _open_position_levexp(cash: float, arrays: dict[str, np.ndarray], i: int, side: int, sleeve_name: str, risk: sleeve.FallbackRisk | None, row: pd.Series | None, fee_eff: float, slip_eff: float) -> tuple[float, sleeve.Position, bool]:
    filled, entry_px, entry_fee, _route = omega._try_execution(arrays, int(i), int(side), entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled:
        return cash, sleeve.Position(), False
    if sleeve_name == "primary":
        assert row is not None
        margin_notional = float(row.get("notional_exposure", 0.0) or 0.0)
        leverage = float(row.get("leverage", 1.0) or 1.0)
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = abs(float(row.get("stop_loss", 0.0) or 0.0))
        max_hold = int(row.get("max_hold_bars", 0) or 0)
    else:
        assert risk is not None
        margin_notional = float(risk.notional)
        leverage = float(risk.leverage)
        take_profit = float(risk.take_profit)
        stop_loss = abs(float(risk.stop_loss))
        max_hold = int(risk.max_hold_bars)
    effective_exposure = margin_notional * max(leverage, 0.0)
    if effective_exposure <= 0.0:
        return cash, sleeve.Position(), False
    entry_equity = cash
    cash -= cash * float(entry_fee) * effective_exposure
    return (
        cash,
        sleeve.Position(
            sleeve=sleeve_name,
            side=int(side),
            entry_price=float(entry_px),
            entry_i=int(i),
            entry_equity=float(entry_equity),
            notional=float(effective_exposure),
            leverage=float(leverage),
            take_profit=take_profit,
            stop_loss=stop_loss,
            max_hold_bars=max_hold,
        ),
        True,
    )


def _close_position_levexp(cash: float, arrays: dict[str, np.ndarray], pos: sleeve.Position, i: int, fee_eff: float, slip_eff: float) -> tuple[float, bool]:
    if pos.side == 0:
        return cash, False
    _ok, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), int(pos.side), entry=False, fee_base=fee_eff, slip_base=slip_eff)
    raw = (exit_px - pos.entry_price) / max(pos.entry_price, 1.0e-12) if pos.side > 0 else (pos.entry_price - exit_px) / max(pos.entry_price, 1.0e-12)
    before = cash
    cash = cash * (1.0 + raw * pos.notional)
    cash -= before * float(exit_fee) * pos.notional
    return cash, cash > pos.entry_equity


def _metrics_with_fallback_levexp(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    risk: sleeve.FallbackRisk,
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
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = sleeve.Position()
    trades = wins = 0
    primary_entries = fallback_entries = long_entries = short_entries = 0
    reasons: dict[str, int] = {}
    primary_takeovers = 0
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1.0e-12) if pos.side > 0 else (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1.0e-12)
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
                cash, win = _close_position_levexp(cash, arrays, pos, i, fee_eff, slip_eff)
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
                cash, pos, entered = _open_position_levexp(cash, arrays, i, side, "primary", None, row, fee_eff, slip_eff)
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
        cash, pos, entered = _open_position_levexp(cash, arrays, i, side, "fallback", risk, None, fee_eff, slip_eff)
        if entered:
            fallback_entries += 1
            long_entries += int(side > 0)
            short_entries += int(side < 0)
    if pos.side != 0:
        cash, win = _close_position_levexp(cash, arrays, pos, len(frame) - 1, fee_eff, slip_eff)
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
    }


def _metrics_dynamic(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    stats: dict[str, np.ndarray],
    filter_cfg: FilterCfg,
    scale_cfg: DynamicScaleCfg,
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    arrays = sleeve._arrays(frame)
    active = omega._active(dec)
    fee_eff = float(fee) * 3.0
    slip_eff = float(slip) * 3.0
    action, conf = _filtered_action_conf(stats, filter_cfg)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = sleeve.Position()
    trades = wins = 0
    primary_entries = fallback_entries = long_entries = short_entries = 0
    primary_takeovers = 0
    reasons: dict[str, int] = {}
    high_scaled_entries = 0
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1.0e-12) if pos.side > 0 else (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1.0e-12)
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
                cash, win = _close_position_levexp(cash, arrays, pos, i, fee_eff, slip_eff)
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
                cash, pos, entered = _open_position_levexp(cash, arrays, i, side, "primary", None, row, fee_eff, slip_eff)
                if entered:
                    primary_entries += 1
                    long_entries += int(side > 0)
                    short_entries += int(side < 0)
            continue
        if int(action[i]) not in (sleeve.ACTION_LONG, sleeve.ACTION_SHORT):
            continue
        risk = _dynamic_risk(stats, i, scale_cfg)
        side = 1 if int(action[i]) == sleeve.ACTION_LONG else -1
        cash, pos, entered = _open_position_levexp(cash, arrays, i, side, "fallback", risk, None, fee_eff, slip_eff)
        if entered:
            fallback_entries += 1
            high_scaled_entries += int(risk.notional > BASE_RISK.notional * scale_cfg.base_scale + 1.0e-12)
            long_entries += int(side > 0)
            short_entries += int(side < 0)
    if pos.side != 0:
        cash, win = _close_position_levexp(cash, arrays, pos, len(frame) - 1, fee_eff, slip_eff)
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
        "high_scaled_entries": int(high_scaled_entries),
        "exit_reasons": reasons,
    }


def _score(row: dict[str, Any]) -> float:
    if int(row["val_trades"]) < 25:
        return -1.0e9
    return float(row["val_pnl"]) / max(abs(float(row["val_mdd"])), 1.0e-12)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_dec, val_features = full._build_split(frames, "validation")
    oos_frame, oos_dec, oos_features = full._build_split(frames, "oos")
    feature_cols = list(val_features.columns)
    if feature_cols != list(oos_features.columns):
        raise RuntimeError("Omega2.1 feature contract mismatch")
    _reject_forbidden(feature_cols, "omega2_1")
    y, valid_mask, label_diag = label_family._triple_barrier_labels(val_frame, atr_mult=1.0, max_hold=24, min_barrier=0.0035)
    train_mask = (~omega._active(val_dec)) & valid_mask
    val_proba, oos_proba, val_stack, oos_stack, hgb_diag = _predict_oof_and_full(val_features, y, train_mask, oos_features)
    val_stats = _signal_stats(val_proba, val_stack)
    oos_stats = _signal_stats(oos_proba, oos_stack)
    baseline_filter = FilterCfg(0.55, 0.0, 0.0)
    baseline_val_action, baseline_val_conf = _filtered_action_conf(val_stats, baseline_filter)
    baseline_oos_action, baseline_oos_conf = _filtered_action_conf(oos_stats, baseline_filter)
    baseline_val = _metrics_with_fallback_levexp(
        val_frame,
        val_dec,
        BASE_RISK,
        baseline_val_action,
        baseline_val_conf,
        0.0,
        fee=fee,
        slip=slip,
        cost_mult=3.0,
    )
    baseline_oos = _metrics_with_fallback_levexp(
        oos_frame,
        oos_dec,
        BASE_RISK,
        baseline_oos_action,
        baseline_oos_conf,
        0.0,
        fee=fee,
        slip=slip,
        cost_mult=3.0,
    )

    rows: list[dict[str, Any]] = []
    filter_grid = [
        FilterCfg(t, m, a)
        for t in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
        for m in (0.00, 0.03, 0.06, 0.10)
        for a in (0.50, 0.67, 0.75, 0.84, 0.92)
    ]
    static_scales = [(1.0, 0.30), (1.25, 0.45), (1.5, 0.60), (1.75, 0.75), (2.0, 0.90), (2.5, 0.90)]
    for fc in filter_grid:
        val_action, val_conf = _filtered_action_conf(val_stats, fc)
        oos_action, oos_conf = _filtered_action_conf(oos_stats, fc)
        for scale, cap in static_scales:
            risk = _scaled_risk(scale, cap, f"static_s{scale:g}_cap{cap:g}")
            val_m = _metrics_with_fallback_levexp(val_frame, val_dec, risk, val_action, val_conf, 0.0, fee=fee, slip=slip, cost_mult=3.0)
            oos_m = _metrics_with_fallback_levexp(oos_frame, oos_dec, risk, oos_action, oos_conf, 0.0, fee=fee, slip=slip, cost_mult=3.0)
            row = {
                "candidate": f"static_t{fc.threshold:.2f}_m{fc.margin_min:.2f}_a{fc.agreement_min:.2f}_s{scale:g}_cap{cap:g}",
                "kind": "static",
                "threshold": fc.threshold,
                "margin_min": fc.margin_min,
                "agreement_min": fc.agreement_min,
                "scale": scale,
                "cap": cap,
                **_metric("val", val_m),
                **_metric("oos", oos_m),
            }
            row["val_score"] = _score(row)
            row["oos_delta_vs_omega21"] = float(row["oos_pnl"] - float(baseline_oos["pnl"]))
            rows.append(row)

    dynamic_grid = [
        DynamicScaleCfg(f"dyn_b{base:g}_h{high:g}_c{conf:.2f}_m{margin:.2f}_a{agree:.2f}_cap{cap:g}", base, high, conf, margin, agree, cap)
        for base in (1.0, 1.25)
        for high in (1.5, 1.75, 2.0, 2.5)
        for conf in (0.60, 0.65, 0.70)
        for margin in (0.03, 0.06, 0.10)
        for agree in (0.75, 0.84, 0.92)
        for cap in (0.60, 0.75, 0.90)
    ]
    # Keep dynamic search anchored to the original accepted threshold to avoid an unbounded grid.
    dynamic_filters = [FilterCfg(t, m, a) for t in (0.50, 0.55, 0.60) for m in (0.00, 0.03, 0.06) for a in (0.67, 0.75, 0.84)]
    for fc in dynamic_filters:
        for dc in dynamic_grid:
            val_m = _metrics_dynamic(val_frame, val_dec, val_stats, fc, dc, fee=fee, slip=slip)
            oos_m = _metrics_dynamic(oos_frame, oos_dec, oos_stats, fc, dc, fee=fee, slip=slip)
            row = {
                "candidate": f"{dc.name}_t{fc.threshold:.2f}_m{fc.margin_min:.2f}_a{fc.agreement_min:.2f}",
                "kind": "dynamic",
                "threshold": fc.threshold,
                "margin_min": fc.margin_min,
                "agreement_min": fc.agreement_min,
                "scale": dc.base_scale,
                "high_scale": dc.high_scale,
                "high_conf": dc.high_conf,
                "high_margin": dc.high_margin,
                "high_agree": dc.high_agree,
                "cap": dc.cap,
                **_metric("val", val_m),
                **_metric("oos", oos_m),
            }
            row["val_high_scaled_entries"] = int(val_m.get("high_scaled_entries", 0))
            row["oos_high_scaled_entries"] = int(oos_m.get("high_scaled_entries", 0))
            row["val_score"] = _score(row)
            row["oos_delta_vs_omega21"] = float(row["oos_pnl"] - float(baseline_oos["pnl"]))
            rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(["oos_pnl", "val_score", "oos_mdd"], ascending=[False, False, False]).reset_index(drop=True)
    val_ranking = pd.DataFrame(rows).sort_values(["val_score", "val_pnl", "val_mdd"], ascending=[False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "ranking_by_oos.csv", index=False)
    val_ranking.to_csv(OUT_DIR / "ranking_by_val.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "status": "research_not_live_promoted",
        "accounting": {
            "mode": "leverage_exposure",
            "effective_exposure": "notional_exposure * leverage",
            "fee_base": "fee * cost_mult * effective_exposure",
            "pnl": "raw_price_return * effective_exposure",
            "note": "This is intentionally separated from the legacy metadata-leverage report.",
        },
        "legacy_metadata_leverage_baseline_oos": BASELINE_OOS,
        "corrected_baseline": {
            **_metric("validation", baseline_val),
            **_metric("oos", baseline_oos),
        },
        "label_diag": label_diag,
        "hgb_diag": hgb_diag,
        "top_by_oos": ranking.head(30).to_dict(orient="records"),
        "top_by_val": val_ranking.head(30).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking_by_oos": str(OUT_DIR / "ranking_by_oos.csv"),
            "ranking_by_val": str(OUT_DIR / "ranking_by_val.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top_by_oos": report["top_by_oos"][:8], "top_by_val": report["top_by_val"][:8]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
