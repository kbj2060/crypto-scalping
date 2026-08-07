#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha6_catboost_5head_policy_20260522 import _days, _fill_price  # noqa: E402
from scripts.alpha6_catboost_entry_quality_exit_policy_20260522 import (  # noqa: E402
    TARGET_BUCKET_TO_HORIZON,
    _exit_close_prob,
    _exit_state_vec,
    _threshold_for_bucket,
)
from scripts.analyze_alpha6_sleeve_complementarity_20260523 import Expert, _load_experts  # noqa: E402
from scripts.backtest_alpha6_label_scoring_stack_20260524 import _splits  # noqa: E402


PRIMARY = 0
COVERAGE = 1
CONFIRMERS = (2, 3)
RISKS = (4, 5)


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _desired(e: Expert, i: int, threshold_mult: float) -> int:
    row = e.dec.iloc[i]
    threshold = float(e.entry_threshold) * float(threshold_mult)
    return int(row.action) if float(row.quality_score) >= threshold else 0


def _edge(e: Expert, i: int, threshold_mult: float) -> float:
    q = float(e.dec.iloc[i].quality_score)
    threshold = float(e.entry_threshold) * float(threshold_mult)
    return float(np.clip((q - threshold) / max(abs(threshold), 1e-9), -3.0, 3.0))


def _signal_meta(experts: list[Expert], i: int, side: int, threshold_mult: float) -> dict[str, Any]:
    desired = [_desired(e, i, threshold_mult) for e in experts]
    edges = [_edge(e, i, threshold_mult) for e in experts]
    same = [idx for idx, d in enumerate(desired) if d == side]
    opp = [idx for idx, d in enumerate(desired) if d not in (0, side)]
    risk_opp = sum(1 for idx in RISKS if desired[idx] not in (0, side))
    risk_same = sum(1 for idx in RISKS if desired[idx] == side)
    confirm_same = sum(1 for idx in CONFIRMERS if desired[idx] == side)
    return {
        "desired": desired,
        "edges": edges,
        "agreement": len(same),
        "opposition": len(opp),
        "risk_opp": risk_opp,
        "risk_same": risk_same,
        "confirm_same": confirm_same,
        "high_precision_same": desired[2] == side,
        "avg_same_edge": float(np.mean([max(edges[idx], 0.0) for idx in same])) if same else 0.0,
    }


def _stack_route(experts: list[Expert], i: int, cfg: dict[str, Any]) -> tuple[int, int, float, dict[str, Any]]:
    threshold_mult = float(cfg.get("threshold_mult", 1.0))
    route = str(cfg.get("route", "stack"))

    if route == "primary_only":
        d = _desired(experts[PRIMARY], i, threshold_mult)
        if d:
            score = _edge(experts[PRIMARY], i, threshold_mult)
            return PRIMARY, d, score, _signal_meta(experts, i, d, threshold_mult)
        return -1, 0, -999.0, {}

    if route == "primary_coverage":
        for idx in (PRIMARY, COVERAGE):
            d = _desired(experts[idx], i, threshold_mult)
            if d:
                score = _edge(experts[idx], i, threshold_mult)
                return idx, d, score, _signal_meta(experts, i, d, threshold_mult)
        return -1, 0, -999.0, {}

    primary = _desired(experts[PRIMARY], i, threshold_mult)
    coverage = _desired(experts[COVERAGE], i, threshold_mult)
    if primary:
        base_idx = PRIMARY
        side = primary
        required = float(cfg.get("min_score", 0.0))
    elif coverage:
        base_idx = COVERAGE
        side = coverage
        required = float(cfg.get("coverage_min_score", 0.0))
    else:
        return -1, 0, -999.0, {}

    score = _edge(experts[base_idx], i, threshold_mult)
    risk_opp = 0
    for idx in CONFIRMERS:
        d = _desired(experts[idx], i, threshold_mult)
        if d == side:
            score += float(cfg.get("confirm_same_w", 0.0)) * max(_edge(experts[idx], i, threshold_mult), 0.0)
        elif d:
            score -= float(cfg.get("confirm_opp_w", 0.0)) * max(_edge(experts[idx], i, threshold_mult), 0.0)
    for idx in RISKS:
        d = _desired(experts[idx], i, threshold_mult)
        if d == side:
            score += float(cfg.get("risk_same_credit", 0.0)) * max(_edge(experts[idx], i, threshold_mult), 0.0)
        elif d:
            risk_opp += 1
            score -= float(cfg.get("risk_opp_w", 0.0)) * max(_edge(experts[idx], i, threshold_mult), 0.0)
        else:
            score -= float(cfg.get("risk_not_same_w", 0.0))
    if bool(cfg.get("hard_double_risk_veto", False)) and risk_opp >= 2:
        score = -999.0
    if bool(cfg.get("protect_primary", False)) and base_idx == PRIMARY:
        score = max(score, required)
    if score < required:
        return -1, 0, score, _signal_meta(experts, i, side, threshold_mult)
    return base_idx, side, score, _signal_meta(experts, i, side, threshold_mult)


def _scaled_exposure(row: pd.Series, score: float, meta: dict[str, Any], cfg: dict[str, Any]) -> float:
    base = _as_float(row.notional, 0.25)
    policy = str(cfg.get("exposure_policy", "fixed_scale"))
    if policy != "conditional_alpha4":
        exposure = base * float(cfg.get("exposure_scale", 1.0))
        return float(np.clip(exposure, float(cfg.get("min_exposure", 0.01)), float(cfg.get("max_exposure", 2.0))))

    exposure = base
    risk_opp = int(meta.get("risk_opp", 0))
    agreement = int(meta.get("agreement", 0))
    avg_edge = float(meta.get("avg_same_edge", 0.0))
    high_precision_same = bool(meta.get("high_precision_same", False))
    confirm_same = int(meta.get("confirm_same", 0))

    tier1 = float(cfg.get("tier1_exposure", 0.75))
    tier2 = float(cfg.get("tier2_exposure", 1.25))
    tier3 = float(cfg.get("tier3_exposure", 2.25))
    tier1_agreement = int(cfg.get("tier1_agreement", 1))
    tier2_agreement = int(cfg.get("tier2_agreement", 3))
    tier3_agreement = int(cfg.get("tier3_agreement", 4))
    tier2_score = float(cfg.get("tier2_score", 0.35))
    tier3_score = float(cfg.get("tier3_score", 0.75))
    tier3_edge = float(cfg.get("tier3_edge", 0.35))

    if risk_opp > 0:
        exposure = base
    elif high_precision_same and agreement >= tier1_agreement and score >= 0.0:
        exposure = max(exposure, tier1)
    if risk_opp == 0 and agreement >= tier2_agreement and confirm_same >= 1 and score >= tier2_score:
        exposure = max(exposure, tier2)
    if risk_opp == 0 and agreement >= tier3_agreement and high_precision_same and score >= tier3_score and avg_edge >= tier3_edge:
        exposure = max(exposure, tier3)
    return float(np.clip(exposure, float(cfg.get("min_exposure", 0.01)), float(cfg.get("max_exposure", 2.75))))


def _empty() -> dict[str, Any]:
    return {
        "pnl": 0.0,
        "mdd": 0.0,
        "calmar": 0.0,
        "trades": 0,
        "trades_per_day": 0.0,
        "wr": 0.0,
        "long_entries": 0,
        "short_entries": 0,
        "avg_notional": 0.0,
        "exit_model_closes": 0,
        "exits": {},
    }


def _replay_arch(
    frame: pd.DataFrame,
    experts: list[Expert],
    *,
    fee: float,
    slip: float,
    min_exit_hold: int,
    state_horizon: int,
    start: int,
    end: int,
    cfg: dict[str, Any],
    save_trades: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    start = max(0, int(start))
    end = min(int(end), len(frame) - 2)

    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    active = -1
    entry = 0.0
    entry_idx = 0
    entry_equity = 1.0
    hold = 0
    mae = 0.0
    mfe = 0.0
    exposure = 0.0
    base_exposure = 0.0
    addon_used = False
    entry_meta: dict[str, Any] = {}
    target_horizon = int(state_horizon)
    target_bucket = 4
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    exit_model_closes = 0
    exposure_sum = 0.0
    exits: dict[str, int] = {}
    trade_rows: list[dict[str, Any]] = []

    exposure_scale = float(cfg.get("exposure_scale", 1.0))
    max_exposure = float(cfg.get("max_exposure", 2.0))
    min_exposure = float(cfg.get("min_exposure", 0.01))
    exit_mode = str(cfg.get("exit_mode", "exit_model"))
    exit_on_flip = bool(cfg.get("exit_on_flip", False))

    def equity(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * exposure)

    def scaled_notional(row: pd.Series) -> float:
        return float(np.clip(_as_float(row.notional, 0.25) * exposure_scale, min_exposure, max_exposure))

    def enter(i: int, expert_idx: int, new_side: int, score: float, meta: dict[str, Any]) -> None:
        nonlocal side, active, entry, entry_idx, entry_equity, hold, mae, mfe, exposure, base_exposure
        nonlocal target_horizon, target_bucket, cash, exposure_sum, long_entries, short_entries, addon_used
        nonlocal entry_meta
        e = experts[expert_idx]
        row = e.dec.iloc[i]
        fill_i = min(i + 1, len(frame) - 1)
        side = int(new_side)
        active = int(expert_idx)
        entry_idx = int(i)
        exposure = _scaled_exposure(row, score, meta, cfg)
        base_exposure = exposure
        target_horizon = int(np.clip(int(getattr(row, "target_horizon", state_horizon)), 2, state_horizon))
        target_bucket = int(np.clip(int(getattr(row, "target_bucket", 4)), 0, 4))
        entry = _fill_price(frame, fill_i, side, slip, entry=True)
        entry_equity = cash
        cash -= cash * fee * exposure
        exposure_sum += exposure
        hold = 0
        mae = 0.0
        mfe = 0.0
        addon_used = False
        entry_meta = dict(meta)
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def maybe_addon(i: int, raw: float) -> None:
        nonlocal entry, exposure, cash, exposure_sum, addon_used
        if addon_used or not bool(cfg.get("addon", False)):
            return
        if hold < int(cfg.get("addon_min_hold", 3)):
            return
        if raw < float(cfg.get("addon_min_raw", 0.004)):
            return
        target_total = min(max_exposure, base_exposure * float(cfg.get("addon_max_total_mult", 1.35)))
        add_exposure = min(base_exposure * float(cfg.get("addon_frac", 0.20)), target_total - exposure)
        if add_exposure <= 1e-9:
            return
        fill_i = min(i + 1, len(frame) - 1)
        add_px = _fill_price(frame, fill_i, side, slip, entry=True)
        entry = (entry * exposure + add_px * add_exposure) / max(exposure + add_exposure, 1e-12)
        cash -= cash * fee * add_exposure
        exposure += add_exposure
        exposure_sum += add_exposure
        addon_used = True

    def exit_pos(i: int, reason: str) -> None:
        nonlocal side, active, entry, cash, hold, mae, mfe, exposure, base_exposure, target_horizon, target_bucket
        nonlocal trades, wins, exit_model_closes
        nonlocal entry_meta
        fill_i = min(i + 1, len(frame) - 1)
        fill_px = _fill_price(frame, fill_i, side, slip, entry=False)
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * fee * exposure
        pnl_pct = (cash / max(entry_equity, 1e-12) - 1.0) * 100.0
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        if reason == "exit_model":
            exit_model_closes += 1
        if save_trades:
            trade_rows.append(
                {
                    "arch": cfg["name"],
                    "expert": experts[active].name,
                    "entry_idx": int(entry_idx),
                    "exit_idx": int(i),
                    "entry_time": str(frame.iloc[entry_idx]["timestamp"]),
                    "exit_time": str(frame.iloc[int(i)]["timestamp"]),
                    "side": "LONG" if side > 0 else "SHORT",
                    "hold_bars": int(hold),
                    "exposure": float(exposure),
                    "raw_ret": float(raw),
                    "pnl_pct_on_equity": float(pnl_pct),
                    "mae": float(mae),
                    "mfe": float(mfe),
                    "target_horizon": int(target_horizon),
                    "target_bucket": int(target_bucket),
                    "reason": reason,
                    "addon_used": bool(addon_used),
                    "entry_agreement": int(entry_meta.get("agreement", 0)),
                    "entry_risk_opp": int(entry_meta.get("risk_opp", 0)),
                    "entry_avg_same_edge": float(entry_meta.get("avg_same_edge", 0.0)),
                }
            )
        side = 0
        active = -1
        entry = 0.0
        hold = 0
        mae = 0.0
        mfe = 0.0
        exposure = 0.0
        base_exposure = 0.0
        entry_meta = {}
        target_horizon = int(state_horizon)
        target_bucket = 4

    for i in range(start, end):
        cand_idx, cand_side, cand_score, cand_meta = _stack_route(experts, i, cfg)
        if side != 0:
            hold += 1
            px = float(close[i])
            raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
            mae = max(mae, max(0.0, -raw * exposure))
            mfe = max(mfe, max(0.0, raw * exposure))
            maybe_addon(i, raw)

            fixed_exit = None
            if exit_mode in {"fixed_barrier", "hybrid"} and hold >= int(cfg.get("barrier_min_hold", 1)):
                if raw >= float(cfg.get("tp_pct", 0.015)):
                    fixed_exit = "tp"
                elif raw <= -float(cfg.get("sl_pct", 0.006)):
                    fixed_exit = "sl"
                elif hold >= int(cfg.get("max_hold", target_horizon)):
                    fixed_exit = "max_hold"

            if active == COVERAGE and bool(cfg.get("primary_preempt", True)) and cand_idx == PRIMARY:
                exit_pos(i, "primary_preempt")
                enter(i, PRIMARY, 1 if cand_side == 1 else -1, cand_score, cand_meta)
            elif (
                bool(cfg.get("stale_exit", False))
                and hold
                >= max(
                    int(min_exit_hold),
                    int(cfg.get("stale_min_hold", 0)),
                    int(target_horizon) * float(cfg.get("stale_target_mult", 1.0)),
                )
                and raw <= float(cfg.get("stale_min_raw", 0.0))
            ):
                exit_pos(i, "stale_exit")
            elif (
                bool(cfg.get("giveback_exit", False))
                and mfe >= float(cfg.get("giveback_min_mfe", 0.004))
                and (mfe - max(raw * exposure, 0.0)) / max(mfe, 1e-9) >= float(cfg.get("giveback_ratio", 0.65))
            ):
                exit_pos(i, "giveback_exit")
            elif fixed_exit:
                exit_pos(i, fixed_exit)
            elif exit_mode in {"exit_model", "hybrid"} and hold >= int(min_exit_hold):
                e = experts[active]
                expected = e.bundle.get("expected_return_by_bucket") or {k: 0.01 for k in TARGET_BUCKET_TO_HORIZON}
                exit_meta = e.bundle.get("exit_meta", {})
                state = _exit_state_vec(
                    frame,
                    side=side,
                    entry_idx=entry_idx,
                    current_idx=i,
                    entry_px=entry,
                    px=px,
                    hold=hold,
                    horizon=int(target_horizon),
                    mae=mae,
                    mfe=mfe,
                    target_bucket=target_bucket,
                    regime_drift=bool(exit_meta.get("regime_drift", False)),
                    capture_ratio=bool(exit_meta.get("capture_ratio", False)),
                    expected_return=float(expected.get(target_bucket, 0.01)),
                )
                close_prob = _exit_close_prob(e.bundle["exit_model"], e.x[i], state)
                if close_prob >= _threshold_for_bucket(e.exit_threshold, target_bucket):
                    exit_pos(i, "exit_model")
                elif exit_on_flip and cand_side and ((cand_side == 1 and side < 0) or (cand_side == 2 and side > 0)):
                    exit_pos(i, "model_flip")

        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side == 0 and cand_idx >= 0:
            enter(i, cand_idx, 1 if cand_side == 1 else -1, cand_score, cand_meta)

    if side != 0:
        exit_pos(end, "end")
    if trades == 0:
        return _empty(), pd.DataFrame(trade_rows)
    pnl = float((cash - 1.0) * 100.0)
    mdd_pct = float(mdd * 100.0)
    return (
        {
            "pnl": pnl,
            "mdd": mdd_pct,
            "calmar": float(pnl / max(abs(mdd_pct), 1e-12)),
            "trades": int(trades),
            "trades_per_day": float(trades / _days(frame.iloc[start : end + 1])),
            "wr": float(wins / max(trades, 1)),
            "long_entries": int(long_entries),
            "short_entries": int(short_entries),
            "avg_notional": float(exposure_sum / max(trades, 1)),
            "exit_model_closes": int(exit_model_closes),
            "exits": exits,
        },
        pd.DataFrame(trade_rows),
    )


def _arch_configs() -> list[dict[str, Any]]:
    base_stack = {
        "route": "stack",
        "threshold_mult": 1.0,
        "min_score": 0.0,
        "coverage_min_score": 0.0,
        "confirm_same_w": 0.0,
        "confirm_opp_w": 0.0,
        "risk_opp_w": 0.0,
        "risk_not_same_w": 0.2,
        "risk_same_credit": 0.05,
        "hard_double_risk_veto": True,
        "protect_primary": True,
        "primary_preempt": True,
        "exit_mode": "exit_model",
        "exposure_scale": 1.0,
        "max_exposure": 2.0,
    }
    return [
        {"name": "alpha6_1_baseline_stack", **base_stack},
        {
            "name": "alpha4_parent_direct_exposure",
            **base_stack,
            "exposure_scale": 9.0,
            "max_exposure": 2.75,
        },
        {
            "name": "alpha6_1_conditional_alpha4_exposure",
            **base_stack,
            "exposure_policy": "conditional_alpha4",
            "max_exposure": 2.75,
        },
        {
            "name": "alpha6_1_conditional_exposure_stale_exit",
            **base_stack,
            "exposure_policy": "conditional_alpha4",
            "max_exposure": 2.75,
            "stale_exit": True,
            "stale_target_mult": 1.0,
            "stale_min_raw": 0.0,
        },
        {
            "name": "alpha6_1_conditional_exposure_giveback_exit",
            **base_stack,
            "exposure_policy": "conditional_alpha4",
            "max_exposure": 2.75,
            "giveback_exit": True,
            "giveback_min_mfe": 0.004,
            "giveback_ratio": 0.65,
        },
        {
            "name": "alpha6_1_conditional_exposure_stale_giveback",
            **base_stack,
            "exposure_policy": "conditional_alpha4",
            "max_exposure": 2.75,
            "stale_exit": True,
            "stale_target_mult": 1.0,
            "stale_min_raw": 0.0,
            "giveback_exit": True,
            "giveback_min_mfe": 0.004,
            "giveback_ratio": 0.65,
        },
        {
            "name": "alpha6_1_strict_conditional_exposure",
            **base_stack,
            "exposure_policy": "conditional_alpha4",
            "max_exposure": 1.50,
            "tier1_exposure": 0.50,
            "tier2_exposure": 0.90,
            "tier3_exposure": 1.50,
            "tier1_agreement": 3,
            "tier2_agreement": 4,
            "tier3_agreement": 5,
            "tier2_score": 0.60,
            "tier3_score": 1.20,
            "tier3_edge": 0.55,
        },
        {
            "name": "alpha6_1_strict_conditional_late_stale",
            **base_stack,
            "exposure_policy": "conditional_alpha4",
            "max_exposure": 1.50,
            "tier1_exposure": 0.50,
            "tier2_exposure": 0.90,
            "tier3_exposure": 1.50,
            "tier1_agreement": 3,
            "tier2_agreement": 4,
            "tier3_agreement": 5,
            "tier2_score": 0.60,
            "tier3_score": 1.20,
            "tier3_edge": 0.55,
            "stale_exit": True,
            "stale_min_hold": 288,
            "stale_target_mult": 4.0,
            "stale_min_raw": 0.0,
        },
        {
            "name": "alpha6_1_sniper_conditional_exposure",
            **base_stack,
            "exposure_policy": "conditional_alpha4",
            "max_exposure": 2.25,
            "tier1_exposure": 0.25,
            "tier2_exposure": 1.00,
            "tier3_exposure": 2.25,
            "tier1_agreement": 4,
            "tier2_agreement": 4,
            "tier3_agreement": 5,
            "tier2_score": 0.85,
            "tier3_score": 1.50,
            "tier3_edge": 0.70,
        },
        {
            "name": "alpha4_no_teacher_primary_coverage",
            **base_stack,
            "route": "primary_coverage",
            "threshold_mult": 0.85,
            "exposure_scale": 9.0,
            "max_exposure": 2.75,
            "exit_on_flip": True,
        },
        {
            "name": "alpha4_primary_only_direct",
            **base_stack,
            "route": "primary_only",
            "exposure_scale": 9.0,
            "max_exposure": 2.75,
            "exit_on_flip": True,
        },
        {
            "name": "alpha4_legacy_mask_like",
            **base_stack,
            "risk_opp_w": 0.0,
            "risk_not_same_w": 0.0,
            "hard_double_risk_veto": False,
            "exposure_scale": 9.0,
            "max_exposure": 2.75,
        },
        {
            "name": "alpha4_high_trade_gate_like",
            **base_stack,
            "threshold_mult": 0.65,
            "risk_opp_w": 0.0,
            "risk_not_same_w": 0.0,
            "hard_double_risk_veto": False,
            "exposure_scale": 9.0,
            "max_exposure": 2.75,
            "exit_on_flip": True,
        },
        {
            "name": "alpha4_v212_runner_hybrid",
            **base_stack,
            "route": "primary_coverage",
            "threshold_mult": 0.85,
            "exposure_scale": 9.0,
            "max_exposure": 2.75,
            "exit_mode": "hybrid",
            "tp_pct": 0.015,
            "sl_pct": 0.006,
            "max_hold": 96,
            "addon": True,
            "addon_frac": 0.20,
            "addon_max_total_mult": 1.35,
            "addon_min_hold": 3,
            "addon_min_raw": 0.004,
        },
        {
            "name": "alpha4_v212_runner_barrier_only",
            **base_stack,
            "route": "primary_coverage",
            "threshold_mult": 0.85,
            "exposure_scale": 9.0,
            "max_exposure": 2.75,
            "exit_mode": "fixed_barrier",
            "tp_pct": 0.015,
            "sl_pct": 0.006,
            "max_hold": 96,
            "addon": True,
            "addon_frac": 0.20,
            "addon_max_total_mult": 1.35,
            "addon_min_hold": 3,
            "addon_min_raw": 0.004,
        },
        {
            "name": "alpha5_simple_action_score_like",
            **base_stack,
            "route": "stack",
            "threshold_mult": 0.80,
            "min_score": -0.25,
            "coverage_min_score": -0.25,
            "confirm_same_w": 0.15,
            "confirm_opp_w": 0.0,
            "risk_opp_w": 0.0,
            "risk_not_same_w": 0.0,
            "hard_double_risk_veto": False,
            "exposure_scale": 7.0,
            "max_exposure": 2.25,
            "exit_on_flip": True,
        },
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="current_tail111")
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--purge-bars", type=int, default=96)
    ap.add_argument("--min-exit-hold", type=int, default=2)
    ap.add_argument("--state-horizon", type=int, default=96)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "tmp/causal_regen_20260516/alpha6_1_scoring_stack_arch_sweep_20260524",
    )
    args = ap.parse_args()

    frame, experts = _load_experts(args.variant)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fee = 0.0004 * float(args.cost_mult)
    slip = 0.00015 * float(args.cost_mult)
    splits = _splits(len(frame), args.purge_bars)

    rows: list[dict[str, Any]] = []
    trade_paths: dict[str, str] = {}
    for cfg in _arch_configs():
        print(f"[arch] {cfg['name']}", flush=True)
        for split, (start, end) in splits.items():
            save = split in {"test", "full_val"}
            bt, trades = _replay_arch(
                frame,
                experts,
                fee=fee,
                slip=slip,
                min_exit_hold=args.min_exit_hold,
                state_horizon=args.state_horizon,
                start=start,
                end=end,
                cfg=cfg,
                save_trades=save,
            )
            row = {
                "arch": cfg["name"],
                "split": split,
                "route": cfg.get("route"),
                "exit_mode": cfg.get("exit_mode"),
                "threshold_mult": cfg.get("threshold_mult"),
                "exposure_policy": cfg.get("exposure_policy", "fixed_scale"),
                "exposure_scale": cfg.get("exposure_scale"),
                "max_exposure": cfg.get("max_exposure"),
                **bt,
            }
            rows.append(row)
            if save:
                path = args.out_dir / f"{cfg['name']}_{split}_trades.csv"
                trades.to_csv(path, index=False)
                trade_paths[f"{cfg['name']}:{split}"] = str(path)
            print(
                f"  [{split}] pnl={bt['pnl']:.4f} mdd={bt['mdd']:.4f} "
                f"trades={bt['trades']} wr={bt['wr']:.3f} avgN={bt['avg_notional']:.3f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "architecture_sweep.csv", index=False)
    ranked = df[df["split"].eq("full_val")].sort_values("pnl", ascending=False).reset_index(drop=True)
    ranked.to_csv(args.out_dir / "ranking_full_val.csv", index=False)
    ranked_test = df[df["split"].eq("test")].sort_values("pnl", ascending=False).reset_index(drop=True)
    ranked_test.to_csv(args.out_dir / "ranking_test.csv", index=False)
    summary = {
        "cost_mult": float(args.cost_mult),
        "fee": fee,
        "slip": slip,
        "splits": splits,
        "architectures": _arch_configs(),
        "ranking_full_val": ranked.to_dict(orient="records"),
        "ranking_test": ranked_test.to_dict(orient="records"),
        "trade_paths": trade_paths,
        "notes": [
            "This sweep keeps Alpha6.1 expert predictions fixed and only changes deployable architecture/routing/exposure/exit runner.",
            "legacy_mask_like is an architecture proxy: exact historical regime-feature bug is not reintroduced into CatBoost inference.",
            "Costs are charged on notional exposure, including add-on resize fees.",
        ],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"[out] {args.out_dir}", flush=True)
    print(ranked[["arch", "pnl", "mdd", "calmar", "trades", "wr", "avg_notional", "exits"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
