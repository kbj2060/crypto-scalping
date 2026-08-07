#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _days, _fill_price, _read
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner


MODEL_ID = "hf_v13_v31_frozen_parent_layer_ablation_v45_20260512"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v31_frozen_parent_layer_ablation_v45_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v31_frozen_parent_layer_ablation_v45_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v31_frozen_parent_layer_ablation_v45_20260512_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v31_frozen_parent_layer_ablation_v45_20260512_grid.csv"


@dataclass(frozen=True)
class LayerVariant:
    name: str
    layer: str
    overlay: v31.OverlayConfig
    smart_addon: bool = False
    addon_ofti_th: float = 0.0
    addon_trap_th: float = -999.0
    addon_taker_th: float = -999.0
    addon_ofi_th: float = -999.0
    dynamic_scout_gate: bool = False
    scout_vol_scale: float = 0.0
    scout_edge_vol_scale: float = 0.0
    scout_vol_cap: float = 3.0
    execution_sniper: bool = False
    sniper_flow_th: float = 0.20
    sniper_fee_mult: float = 0.70
    sniper_slip_mult: float = 0.50
    mdd_entry_guard: bool = False
    mdd_soft_start: float = 0.20
    mdd_hard_start: float = 0.30
    mdd_parent_scale_col: str = ""
    mdd_deep_scale: float = 0.50
    mdd_hard_scale_mult: float = 0.50


def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(col, default))
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _micro_flow_score(row: pd.Series, side: int) -> float:
    direction = float(np.sign(side))
    return float(
        direction
        * (
            0.55 * _safe(row, "net_taker_ratio", 0.0)
            + 0.25 * _safe(row, "taker_acceleration", 0.0)
            + 0.20 * _safe(row, "ofi_acceleration", 0.0)
        )
    )


def _smart_addon_pass(row: pd.Series, side: int, variant: LayerVariant) -> bool:
    direction = float(np.sign(side))
    ofti = direction * _safe(row, "ofti", 0.0)
    trap = direction * _safe(row, "sig_liquidity_trap", 0.0)
    taker = direction * _safe(row, "taker_acceleration", 0.0)
    ofi = direction * _safe(row, "ofi_acceleration", 0.0)
    if ofti < variant.addon_ofti_th:
        return False
    return bool(
        trap >= variant.addon_trap_th
        or taker >= variant.addon_taker_th
        or ofi >= variant.addon_ofi_th
    )


def _scout_thresholds(row: pd.Series, cfg: v31.OverlayConfig, variant: LayerVariant) -> tuple[float, float]:
    if not variant.dynamic_scout_gate:
        return float(cfg.edge_th), float(cfg.margin_th)
    volz = max(0.0, _safe(row, "garch_vol_z", _safe(row, "volatility_z", 0.0)))
    volz = min(volz, float(variant.scout_vol_cap))
    margin = max(float(cfg.margin_th), float(cfg.margin_th) * (1.0 + variant.scout_vol_scale * volz))
    edge = max(float(cfg.edge_th), float(cfg.edge_th) * (1.0 + variant.scout_edge_vol_scale * volz))
    return float(edge), float(margin)


def _route_cost(row: pd.Series, side: int, fee: float, slip: float, variant: LayerVariant) -> tuple[float, float, str]:
    if not variant.execution_sniper:
        return float(fee), float(slip), "taker"
    flow = _micro_flow_score(row, side)
    liq = _safe(row, "execution_quality", 0.0) - abs(_safe(row, "liquidity_vacuum", 0.0)) * 0.10
    if flow >= float(variant.sniper_flow_th) and liq > -0.35:
        return float(fee * variant.sniper_fee_mult), float(slip * variant.sniper_slip_mult), "sniper_maker_proxy"
    return float(fee), float(slip), "sniper_taker_fallback"


def _fill_with_route(df: pd.DataFrame, idx: int, side: int, fee: float, slip: float, variant: LayerVariant, *, entry: bool) -> tuple[float, float, float, str]:
    idx = int(np.clip(idx, 0, len(df) - 1))
    fee_eff, slip_eff, route = _route_cost(df.iloc[idx], side, fee, slip, variant)
    return _fill_price(df, idx, side, slip_eff, entry=entry), fee_eff, slip_eff, route


def _runtime_mdd_scale(variant: LayerVariant, dd_abs: float, dec: pd.Series | None = None) -> float:
    if not variant.mdd_entry_guard or dd_abs < float(variant.mdd_soft_start):
        return 1.0
    if dec is not None and variant.mdd_parent_scale_col:
        try:
            scale = float(dec.get(variant.mdd_parent_scale_col, variant.mdd_deep_scale))
        except Exception:
            scale = float(variant.mdd_deep_scale)
    else:
        scale = float(variant.mdd_deep_scale)
    if dd_abs >= float(variant.mdd_hard_start):
        scale *= float(variant.mdd_hard_scale_mult)
    return float(np.clip(scale, 0.0, 1.0))


def backtest_variant(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    deep_q: np.ndarray,
    variant: LayerVariant,
    *,
    fee: float,
    slip: float,
    cost_mult: float = 1.0,
    decisions: pd.DataFrame | None = None,
    record: bool = False,
) -> dict[str, Any]:
    cfg = variant.overlay
    close = _close(df)
    if decisions is None:
        decisions = predict_policy_frame(bundle, df, close=close)
    fee_base = fee * cost_mult
    slip_base = slip * cost_mult
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    owner = ""
    entry_price = entry_equity = 0.0
    entry_idx = 0
    parent_notional = notional = 0.0
    take_profit = stop_loss = 0.0
    max_hold = 0
    cooldown = next_cooldown = deep_cooldown = 0
    add_done = False
    mfe = mae = 0.0
    entry_edge = 0.0
    entry_margin = 0.0
    entry_vol_anchor = 0.0
    trades = wins = long_entries = short_entries = deep_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    actions: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        _, slip_eff, _ = _route_cost(df.iloc[int(np.clip(i, 0, len(df) - 1))], pos, fee_base, slip_base, variant)
        raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        dd_abs = max(0.0, 1.0 - eq / max(peak, 1e-12))
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            hold = i - entry_idx
            reason = ""
            effective_tp = take_profit
            effective_sl = stop_loss
            if owner == "deep_alpha":
                if cfg.tp_util_mult > 0.0:
                    util_gain = 1.0 + cfg.tp_util_mult * max(entry_edge - cfg.edge_th, 0.0) / max(0.02, cfg.edge_th)
                    effective_tp = v31._clip(cfg.base_tp * util_gain, cfg.base_tp * 0.8, cfg.tp_cap)
                if cfg.sl_vol_mult > 0.0:
                    vol_sl = v31._clip(entry_vol_anchor * cfg.sl_vol_mult, cfg.base_sl * 0.6, cfg.sl_cap)
                    effective_sl = vol_sl
                if mfe > 0.0 and cfg.trail_gap_mult > 0.0:
                    trail_gap = entry_vol_anchor * cfg.trail_gap_mult
                    if cfg.hold_decay_start < 999 and hold >= cfg.hold_decay_start:
                        decay_bars = hold - cfg.hold_decay_start
                        trail_gap = max(entry_vol_anchor * 0.35, trail_gap - cfg.hold_decay_rate * decay_bars * entry_vol_anchor)
                    trail_stop = max(-effective_sl, mfe - trail_gap)
                    effective_sl = min(effective_sl, max(0.001, trail_stop))
            if effective_tp > 0.0 and unreal >= effective_tp:
                reason = f"{owner}_take_profit"
            elif effective_sl > 0.0 and unreal <= -abs(effective_sl):
                reason = f"{owner}_stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = f"{owner}_max_hold"
            if owner == "v21_2" and not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and dd_abs <= add_cfg.dd_block:
                state = {"parent_notional": parent_notional, "notional": notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": dd_abs, "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
                x = _feature_frame(df, bundle, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x)
                micro_ok = True if not variant.smart_addon else _smart_addon_pass(df.iloc[i], pos, variant)
                if micro_ok and p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40:
                    fill_i = min(i + 1, len(df) - 1)
                    delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - notional))
                    add_px, add_fee, _, add_route = _fill_with_route(df, fill_i, pos, fee_base, slip_base, variant, entry=True)
                    new_notional = notional + delta
                    entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                    before = cash
                    cash -= before * add_fee * delta
                    notional = new_notional
                    actions["v21_add_on"] = actions.get("v21_add_on", 0) + 1
                    route_counts[add_route] = route_counts.get(add_route, 0) + 1
                else:
                    key = "v21_micro_reject" if not micro_ok else "v21_reject"
                    actions[key] = actions.get(key, 0) + 1
                add_done = True
            if reason:
                fill_i = min(i + 1, len(df) - 1)
                exit_px, exit_fee, _, exit_route = _fill_with_route(df, fill_i, pos, fee_base, slip_base, variant, entry=False)
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * exit_fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                route_counts[exit_route] = route_counts.get(exit_route, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update({"exit_signal_timestamp": str(df["timestamp"].iloc[i]), "exit_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "exit_reason": reason, "effective_tp": float(effective_tp), "effective_sl": float(effective_sl), "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0), "final_notional_exposure": float(notional), "mfe_pct": float(mfe * 100.0), "mae_pct": float(mae * 100.0), "fee_exit_pct": float(exit_fee * notional * 100.0), "cash_after": float(cash), "exit_route": exit_route})
                    records.append(out)
                pos = 0
                owner = ""
                cooldown = int(next_cooldown)
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(cfg.cooldown))
                add_done = False
                open_record = None
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if deep_cooldown > 0:
            deep_cooldown -= 1
        dec = decisions.iloc[i]
        if int(dec.action) != ACTION_CASH and int(dec.side) != 0:
            entry_scale = _runtime_mdd_scale(variant, dd_abs, dec)
            if entry_scale <= 1e-8:
                actions["mdd_parent_block"] = actions.get("mdd_parent_block", 0) + 1
                continue
            fill_i = min(i + 1, len(df) - 1)
            pos = int(dec.side)
            owner = "v21_2"
            entry_price, entry_fee, _, entry_route = _fill_with_route(df, fill_i, pos, fee_base, slip_base, variant, entry=True)
            entry_equity = cash
            entry_idx = i
            parent_notional = min(float(dec.notional_exposure) * entry_scale, add_cfg.max_entry_notional)
            notional = parent_notional
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            next_cooldown = int(dec.cooldown_bars)
            cash -= cash * entry_fee * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += float(dec.leverage)
            mfe = mae = 0.0
            add_done = False
            actions["v21_entry"] = actions.get("v21_entry", 0) + 1
            route_counts[entry_route] = route_counts.get(entry_route, 0) + 1
            if record:
                open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "owner": owner, "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "notional_exposure": float(notional), "entry_scale": float(entry_scale), "leverage": float(dec.leverage), "take_profit": float(take_profit), "stop_loss": float(stop_loss), "max_hold_bars": int(max_hold), "fee_entry_pct": float(entry_fee * notional * 100.0), "entry_route": entry_route}
            continue
        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            edge_th, margin_th = _scout_thresholds(df.iloc[i], cfg, variant)
            if edge >= edge_th and margin >= margin_th:
                entry_scale = _runtime_mdd_scale(variant, dd_abs, None)
                if entry_scale <= 1e-8:
                    actions["mdd_deep_block"] = actions.get("mdd_deep_block", 0) + 1
                    continue
                fill_i = min(i + 1, len(df) - 1)
                pos = side
                owner = "deep_alpha"
                entry_price, entry_fee, _, entry_route = _fill_with_route(df, fill_i, pos, fee_base, slip_base, variant, entry=True)
                entry_equity = cash
                entry_idx = i
                parent_notional = float(cfg.notional) * entry_scale
                notional = float(cfg.notional) * entry_scale
                take_profit = float(cfg.base_tp)
                stop_loss = float(cfg.base_sl)
                max_hold = int(cfg.base_hold)
                next_cooldown = int(cfg.cooldown)
                entry_edge = edge
                entry_margin = margin
                entry_vol_anchor = v31._vol_anchor(df.iloc[i]) * notional
                cash -= cash * entry_fee * notional
                long_entries += int(pos > 0)
                short_entries += int(pos < 0)
                deep_entries += 1
                notional_sum += notional
                leverage_sum += max(notional, 1.0)
                mfe = mae = 0.0
                add_done = True
                actions["deep_entry"] = actions.get("deep_entry", 0) + 1
                route_counts[entry_route] = route_counts.get(entry_route, 0) + 1
                if record:
                    open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "owner": owner, "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "notional_exposure": float(notional), "entry_scale": float(entry_scale), "deep_q_long": ql, "deep_q_short": qs, "deep_edge": float(edge), "deep_margin": float(margin), "deep_edge_th": float(edge_th), "deep_margin_th": float(margin_th), "deep_vol_anchor": float(entry_vol_anchor), "take_profit": float(take_profit), "stop_loss": float(stop_loss), "max_hold_bars": int(max_hold), "fee_entry_pct": float(entry_fee * notional * 100.0), "entry_route": entry_route}
    if pos != 0:
        fill_i = len(df) - 1
        exit_px, exit_fee, _, exit_route = _fill_with_route(df, fill_i, pos, fee_base, slip_base, variant, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * exit_fee * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
        route_counts[exit_route] = route_counts.get(exit_route, 0) + 1
    n = max(long_entries + short_entries, 1)
    out = {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades), "wr": float(wins / max(trades, 1)), "trades_per_day": float(trades / _days(df)), "deep_entries": int(deep_entries), "long_entries": int(long_entries), "short_entries": int(short_entries), "avg_notional": float(notional_sum / n), "avg_leverage": float(leverage_sum / n), "exits": exits, "runner_actions": actions, "route_counts": route_counts}
    if record:
        out["trade_records"] = records
    return out


def _base_overlay() -> v31.OverlayConfig:
    return v31.OverlayConfig("v31_notional1_time_decay", 0.010, 0.004, 1.0, 12, 0.040, 0.018, 48, 1.5, 2.5, 1.0, 0.50, 18, 0.025, 0.075, 0.036)


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(c1["pnl"] + 0.35 * c2["pnl"] + 0.20 * c3["pnl"] - 0.35 * abs(c1["mdd"]) + 0.20 * min(c1.get("deep_entries", 0), 90))


def _addon_variants(base: v31.OverlayConfig, train_df: pd.DataFrame) -> list[LayerVariant]:
    ofti = pd.to_numeric(train_df.get("ofti", 0.0), errors="coerce").abs().replace([np.inf, -np.inf], np.nan).dropna()
    q60 = float(ofti.quantile(0.60)) if len(ofti) else 0.0
    q75 = float(ofti.quantile(0.75)) if len(ofti) else 0.0
    return [
        LayerVariant("smart_addon_balanced", "smart_addon", base, smart_addon=True, addon_ofti_th=0.0, addon_trap_th=0.0, addon_taker_th=0.12, addon_ofi_th=0.09),
        LayerVariant("smart_addon_ofti_q60", "smart_addon", base, smart_addon=True, addon_ofti_th=q60, addon_trap_th=-0.10, addon_taker_th=0.08, addon_ofi_th=0.06),
        LayerVariant("smart_addon_ofti_q75", "smart_addon", base, smart_addon=True, addon_ofti_th=q75, addon_trap_th=-0.05, addon_taker_th=0.12, addon_ofi_th=0.09),
    ]


def _dynamic_gate_variants(base: v31.OverlayConfig) -> list[LayerVariant]:
    return [
        LayerVariant("dynamic_scout_margin_vol025", "dynamic_scout_gate", base, dynamic_scout_gate=True, scout_vol_scale=0.25, scout_edge_vol_scale=0.00),
        LayerVariant("dynamic_scout_margin_vol050", "dynamic_scout_gate", base, dynamic_scout_gate=True, scout_vol_scale=0.50, scout_edge_vol_scale=0.00),
        LayerVariant("dynamic_scout_edge_margin_vol025", "dynamic_scout_gate", base, dynamic_scout_gate=True, scout_vol_scale=0.35, scout_edge_vol_scale=0.12),
        LayerVariant("dynamic_scout_edge_margin_vol050", "dynamic_scout_gate", base, dynamic_scout_gate=True, scout_vol_scale=0.50, scout_edge_vol_scale=0.20),
    ]


def _sniper_variants(base: v31.OverlayConfig) -> list[LayerVariant]:
    return [
        LayerVariant("execution_sniper_conservative", "execution_sniper", base, execution_sniper=True, sniper_flow_th=0.28, sniper_fee_mult=0.80, sniper_slip_mult=0.65),
        LayerVariant("execution_sniper_balanced", "execution_sniper", base, execution_sniper=True, sniper_flow_th=0.20, sniper_fee_mult=0.70, sniper_slip_mult=0.50),
        LayerVariant("execution_sniper_aggressive", "execution_sniper", base, execution_sniper=True, sniper_flow_th=0.12, sniper_fee_mult=0.60, sniper_slip_mult=0.35),
    ]


def _overlay_from_vector(x: np.ndarray, base: v31.OverlayConfig) -> v31.OverlayConfig:
    vals = np.asarray(x, dtype=float)
    base_tp = float(np.clip(vals[0], 0.030, 0.055))
    base_sl = float(np.clip(vals[1], 0.012, 0.026))
    tp_util = float(np.clip(vals[2], 0.50, 2.80))
    sl_vol = float(np.clip(vals[3], 1.50, 3.60))
    trail_gap = float(np.clip(vals[4], 0.50, 1.60))
    hold_decay = float(np.clip(vals[5], 0.000, 0.070))
    tp_cap = float(np.clip(vals[6], 0.050, 0.100))
    sl_cap = float(np.clip(vals[7], 0.024, 0.050))
    return replace(base, name="cma_exit_candidate", base_tp=base_tp, base_sl=base_sl, tp_util_mult=tp_util, sl_vol_mult=sl_vol, trail_gap_mult=trail_gap, hold_decay_rate=hold_decay, tp_cap=tp_cap, sl_cap=sl_cap)


def _select_cma_exit(
    val: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    val_q: np.ndarray,
    val_dec: pd.DataFrame,
    base: v31.OverlayConfig,
    *,
    fee: float,
    slip: float,
    maxfevals: int,
) -> tuple[LayerVariant, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_variant: LayerVariant | None = None
    best_score = -1e18

    def eval_vec(x: np.ndarray) -> float:
        nonlocal best_variant, best_score
        overlay = _overlay_from_vector(x, base)
        variant = LayerVariant("cma_exit_optimized", "cma_exit", overlay)
        v1 = backtest_variant(val, bundle, jackpot_model, add_cfg, val_q, variant, fee=fee, slip=slip, cost_mult=1.0, decisions=val_dec)
        v2 = backtest_variant(val, bundle, jackpot_model, add_cfg, val_q, variant, fee=fee, slip=slip, cost_mult=2.0, decisions=val_dec)
        v3 = backtest_variant(val, bundle, jackpot_model, add_cfg, val_q, variant, fee=fee, slip=slip, cost_mult=3.0, decisions=val_dec)
        score = _score(v1, v2, v3)
        row = {"variant": asdict(variant), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": score}
        rows.append(row)
        if score > best_score:
            best_score = score
            best_variant = variant
            print(f"[{MODEL_ID}] cma best score={score:.4f} cfg={json.dumps(asdict(overlay), ensure_ascii=False)}", flush=True)
        return -score

    x0 = np.array([base.base_tp, base.base_sl, base.tp_util_mult, base.sl_vol_mult, base.trail_gap_mult, base.hold_decay_rate, base.tp_cap, base.sl_cap], dtype=float)
    try:
        import cma

        opts = {"maxfevals": int(maxfevals), "popsize": 8, "verb_disp": 1, "bounds": [[0.030, 0.012, 0.50, 1.50, 0.50, 0.0, 0.050, 0.024], [0.055, 0.026, 2.80, 3.60, 1.60, 0.070, 0.100, 0.050]]}
        es = cma.CMAEvolutionStrategy(x0, 0.18, opts)
        es.optimize(eval_vec)
    except Exception as exc:
        print(f"[{MODEL_ID}] cma fallback due to {exc!r}", flush=True)
        rng = np.random.default_rng(20260512)
        eval_vec(x0)
        lo = np.array([0.030, 0.012, 0.50, 1.50, 0.50, 0.0, 0.050, 0.024], dtype=float)
        hi = np.array([0.055, 0.026, 2.80, 3.60, 1.60, 0.070, 0.100, 0.050], dtype=float)
        for _ in range(max(1, int(maxfevals) - 1)):
            eval_vec(lo + rng.random(len(lo)) * (hi - lo))
    assert best_variant is not None
    return best_variant, rows


def _select_from_grid(
    variants: list[LayerVariant],
    val: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    val_q: np.ndarray,
    val_dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
) -> tuple[LayerVariant, list[dict[str, Any]]]:
    best: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for variant in variants:
        v1 = backtest_variant(val, bundle, jackpot_model, add_cfg, val_q, variant, fee=fee, slip=slip, cost_mult=1.0, decisions=val_dec)
        v2 = backtest_variant(val, bundle, jackpot_model, add_cfg, val_q, variant, fee=fee, slip=slip, cost_mult=2.0, decisions=val_dec)
        v3 = backtest_variant(val, bundle, jackpot_model, add_cfg, val_q, variant, fee=fee, slip=slip, cost_mult=3.0, decisions=val_dec)
        row = {"variant": asdict(variant), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    assert best is not None
    return LayerVariant(**{**best["variant"], "overlay": v31.OverlayConfig(**best["variant"]["overlay"])}), rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V31 frozen-parent surrounding layer ablations: smart add-on, CMA exit, dynamic scout gate, and execution sniper proxy.")
    p.add_argument("--parent-model", type=Path, default=v31.DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=v31.DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=v31.DEFAULT_V27)
    p.add_argument("--train-csv", type=Path, default=v31.DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=v31.DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--cma-maxfevals", type=int, default=48)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.parent_model)
    jackpot_payload = joblib.load(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(args.v27_model)
    base_cfg = dict(bundle["config"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train_fit = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    print(f"[{MODEL_ID}] predicting frozen parent and V27", flush=True)
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    fee = float(base_cfg["fee"])
    slip = float(base_cfg["slip"])
    base = _base_overlay()
    baseline = LayerVariant("baseline_v31", "baseline", base)
    print(f"[{MODEL_ID}] selecting layer variants on 2025 Q4", flush=True)
    selected: dict[str, LayerVariant] = {"baseline": baseline}
    selection_rows: list[dict[str, Any]] = []
    for layer, variants in [
        ("smart_addon", _addon_variants(base, train_fit)),
        ("dynamic_scout_gate", _dynamic_gate_variants(base)),
        ("execution_sniper", _sniper_variants(base)),
    ]:
        best, rows = _select_from_grid(variants, val, bundle, jackpot_model, add_cfg, val_q, val_dec, fee=fee, slip=slip)
        selected[layer] = best
        selection_rows.extend(rows)
        print(f"[{MODEL_ID}] selected {layer}: {best.name}", flush=True)
    cma_best, cma_rows = _select_cma_exit(val, bundle, jackpot_model, add_cfg, val_q, val_dec, base, fee=fee, slip=slip, maxfevals=args.cma_maxfevals)
    selected["cma_exit"] = cma_best
    selection_rows.extend(cma_rows)
    print(f"[{MODEL_ID}] evaluating selected variants on fixed 2026 OOS", flush=True)
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for layer, variant in selected.items():
        metrics[layer] = {}
        for mult in (1, 2, 3):
            result = backtest_variant(eval_df, bundle, jackpot_model, add_cfg, eval_q, variant, fee=fee, slip=slip, cost_mult=float(mult), decisions=eval_dec, record=(mult == 1))
            if mult == 1:
                ledger = pd.DataFrame(result.pop("trade_records", []))
                ledger_path = args.report_out.with_name(f"{args.report_out.stem}_{layer}_cost1_ledger.csv")
                ledger.to_csv(ledger_path, index=False)
                ledgers[layer] = str(ledger_path)
            metrics[layer][f"cost{mult}"] = result
    grid_rows: list[dict[str, Any]] = []
    for row in selection_rows:
        variant = row["variant"]
        overlay = variant.pop("overlay")
        grid_rows.append({
            "layer": variant.get("layer"),
            "name": variant.get("name"),
            "selection_score": row["selection_score"],
            "val_cost1_pnl": row["validation_cost1"]["pnl"],
            "val_cost1_mdd": row["validation_cost1"]["mdd"],
            "val_cost1_trades": row["validation_cost1"]["trades"],
            "val_cost2_pnl": row["validation_cost2"]["pnl"],
            "val_cost3_pnl": row["validation_cost3"]["pnl"],
            **{f"variant_{k}": v for k, v in variant.items() if k not in {"name", "layer"}},
            **{f"overlay_{k}": v for k, v in overlay.items()},
        })
    pd.DataFrame(grid_rows).to_csv(args.grid_out, index=False)
    manifest_path = args.out_dir / "v45_frozen_parent_layer_ablation_manifest.json"
    manifest = {"model_id": MODEL_ID, "parent_model": str(args.parent_model), "jackpot_model": str(args.jackpot_model), "v27_model": str(args.v27_model), "selected_variants": {k: asdict(v) for k, v in selected.items()}}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit.get("blocking", []))
    warnings.extend(feature_audit.get("warnings", []))
    warnings.append("execution_sniper_is_ohlcv_proxy_not_live_l2_dsac")
    baseline_pnl = float(metrics["baseline"]["cost1"]["pnl"])
    best_layer = max(metrics, key=lambda k: float(metrics[k]["cost1"]["pnl"]))
    verdict = "promote" if not blocking and best_layer != "baseline" and float(metrics[best_layer]["cost1"]["pnl"]) > baseline_pnl and float(metrics[best_layer]["cost2"]["pnl"]) > 0.0 and float(metrics[best_layer]["cost3"]["pnl"]) > 0.0 else "iterate"
    audit = {"status": "pass" if not blocking else "fail", "verdict": verdict, "blocking": blocking, "warnings": warnings, "selection_uses_2026": False, "selection_window": "2025-10-01..2025-12-31", "oos_window": "2026 fixed OOS only after selection", "policy": "v31_frozen_parent_surrounding_layer_ablation", "parent_frozen": True, "v27_entry_frozen": True, "v21_2_model_frozen": True, "feature_audit": feature_audit, "best_layer_by_cost1": best_layer, "selected_variants": {k: asdict(v) for k, v in selected.items()}, "metrics": metrics}
    report = {"model_id": MODEL_ID, "design": "Frozen original V31 parent stack. Tests four surrounding-layer changes one at a time: microstructure smart V21.2 add-on, CMA-ES V31 exit constants, volatility-adjusted V27 scout gate, and an OHLCV proxy for maker/taker execution sniper.", "metrics": metrics, "audit": audit, "artifacts": {"manifest": str(manifest_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers}}
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "best_layer": best_layer, "verdict": verdict, "metrics": metrics}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
