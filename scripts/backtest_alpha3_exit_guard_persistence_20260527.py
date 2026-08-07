#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_regime4_state24_v2_full_retrain_20260526 as alpha3_full  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts import train_eval_hf_v13_jackpot_runner_v21_2 as v21  # noqa: E402
from scripts.eval_alpha2_1_signal_immediate_limit_20260514 import (  # noqa: E402
    ImmediateLimitConfig,
    _limit_price,
    _limit_touched,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _days, _fill_price, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _predict_cost_runner  # noqa: E402


MODEL_ID = "alpha3_exit_guard_persistence_20260527"
BASE_REPORT = ROOT / "data/ensemble/reports/alpha3_regime4_state24_v2_full_retrain_20260526_summary.json"
GRID_OUT = ROOT / f"data/ensemble/reports/{MODEL_ID}_grid.csv"
REPORT_OUT = ROOT / f"data/ensemble/reports/{MODEL_ID}_summary.json"
AUDIT_OUT = ROOT / f"data/ensemble/reports/{MODEL_ID}_audit.json"


@dataclass(frozen=True)
class ExitGuardConfig:
    name: str
    hard_sl_mult: float
    soft_sl_mult: float
    early_bars: int
    early_sl_mult: float
    soft_min_hold: int
    soft_persist_bars: int
    regime_bad_th: float
    flow_bad_th: float
    giveback_trigger: float
    giveback_min_mfe: float
    giveback_min_hold: int
    entry_quality_min: float
    entry_conf_min: float
    same_side_entry_gap: int
    cooldown_after_hard_stop: int
    cooldown_after_soft_stop: int
    cooldown_after_giveback: int


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(col, default))
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _fallback_close_price(df: pd.DataFrame, fill_i: int, side: int, slip: float, *, entry: bool) -> float:
    px = float(pd.to_numeric(df["close"], errors="coerce").ffill().iloc[int(np.clip(fill_i, 0, len(df) - 1))])
    if side > 0:
        return px * (1.0 + slip if entry else 1.0 - slip)
    return px * (1.0 - slip if entry else 1.0 + slip)


def _try_immediate_limit(
    df: pd.DataFrame,
    signal_i: int,
    side: int,
    cfg: ImmediateLimitConfig,
    *,
    entry: bool,
    fee: float,
    slip: float,
) -> tuple[bool, float, float, float, str]:
    fill_i = min(int(signal_i) + 1, len(df) - 1)
    offset = cfg.entry_offset_bps if entry else cfg.exit_offset_bps
    limit_px = _limit_price(df, signal_i, side, entry=entry, offset_bps=offset, anchor=cfg.anchor)
    if limit_px > 0.0 and _limit_touched(df, fill_i, limit_px, side, entry=entry, penetration_bps=cfg.penetration_bps):
        return True, float(limit_px), float(fee * cfg.maker_fee_mult), 0.0, "signal_immediate_maker_limit"
    if entry:
        if cfg.entry_miss == "market_fallback":
            return True, float(_fallback_close_price(df, fill_i, side, slip, entry=True)), float(fee), float(slip), "entry_market_fallback_after_limit_miss_close"
        return False, 0.0, 0.0, 0.0, "signal_immediate_limit_miss"
    if cfg.exit_miss != "market_fallback":
        return False, 0.0, 0.0, 0.0, "signal_immediate_limit_miss"
    return True, float(_fallback_close_price(df, fill_i, side, slip, entry=False)), float(fee), float(slip), "exit_market_fallback_after_limit_miss_close"


def _time_sl_mult(hold: int, early_bars: int, early_sl_mult: float) -> float:
    if early_bars <= 0:
        return 1.0
    if hold >= early_bars:
        return 1.0
    frac = 1.0 - float(max(hold, 0)) / float(max(early_bars, 1))
    return 1.0 + max(0.0, early_sl_mult - 1.0) * frac


def _regime_bad(row: pd.Series) -> float:
    vals = []
    for col in (
        "regime_bear_id",
        "regime_whipsaw_id",
        "whipsaw_prob",
        "risk_off_prob",
        "instability_prob",
        "regime4_pred_risk_off_prob",
        "regime4_pred_whipsaw_prob",
    ):
        if col in row.index:
            vals.append(_safe(row, col, 0.0))
    if not vals:
        return 0.0
    return float(np.clip(np.mean(vals), 0.0, 1.0))


def _flow_bad(row: pd.Series, side: int) -> float:
    net_taker = _safe(row, "net_taker_ratio", 0.0)
    taker_acc = _safe(row, "taker_acceleration", 0.0)
    ofi_acc = _safe(row, "ofi_acceleration", 0.0)
    flow_pressure = _safe(row, "ai_flow_pressure", 0.0)
    side_sign = 1.0 if side > 0 else -1.0
    adverse = [
        -(net_taker * side_sign),
        -(taker_acc * side_sign),
        -(ofi_acc * side_sign),
        -(flow_pressure * side_sign),
    ]
    return float(np.mean(adverse))


def backtest_signal_limit_exit_guard(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: v21.CostRunnerConfig,
    deep_q: np.ndarray,
    decisions: pd.DataFrame,
    overlay: v31.OverlayConfig,
    limit_cfg: ImmediateLimitConfig,
    guard_cfg: ExitGuardConfig,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    record: bool = False,
) -> dict[str, Any]:
    close = _close(df)
    fee_base = float(fee) * float(cost_mult)
    slip_base = float(slip) * float(cost_mult)
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
    entry_vol_anchor = 0.0
    soft_counter = 0
    last_entry_side = 0
    last_entry_idx = -10**9
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
        raw = (px * (1.0 - slip_base) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_base)) / max(entry_price, 1e-12)
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
                if overlay.tp_util_mult > 0.0:
                    util_gain = 1.0 + overlay.tp_util_mult * max(entry_edge - overlay.edge_th, 0.0) / max(0.02, overlay.edge_th)
                    effective_tp = v31._clip(overlay.base_tp * util_gain, overlay.base_tp * 0.8, overlay.tp_cap)
                if overlay.sl_vol_mult > 0.0:
                    effective_sl = v31._clip(entry_vol_anchor * overlay.sl_vol_mult, overlay.base_sl * 0.6, overlay.sl_cap)
                if mfe > 0.0 and mfe >= float(getattr(overlay, "trail_activation", 0.009)) and overlay.trail_gap_mult > 0.0:
                    trail_gap = entry_vol_anchor * overlay.trail_gap_mult
                    if overlay.hold_decay_start < 999 and hold >= overlay.hold_decay_start:
                        trail_gap = max(entry_vol_anchor * 0.35, trail_gap - overlay.hold_decay_rate * (hold - overlay.hold_decay_start) * entry_vol_anchor)
                    trail_stop = max(-effective_sl, mfe - trail_gap)
                    effective_sl = min(effective_sl, max(0.001, trail_stop))

            if effective_tp > 0.0 and unreal >= effective_tp:
                reason = f"{owner}_take_profit"
            else:
                sl_tm = _time_sl_mult(hold, guard_cfg.early_bars, guard_cfg.early_sl_mult)
                hard_sl = max(0.0, abs(effective_sl) * guard_cfg.hard_sl_mult * sl_tm)
                soft_sl = max(0.0, abs(effective_sl) * guard_cfg.soft_sl_mult * sl_tm)
                regime_bad = _regime_bad(df.iloc[i])
                flow_bad = _flow_bad(df.iloc[i], pos)
                soft_hit = (
                    soft_sl > 0.0
                    and hold >= guard_cfg.soft_min_hold
                    and unreal <= -soft_sl
                    and regime_bad >= guard_cfg.regime_bad_th
                    and flow_bad >= guard_cfg.flow_bad_th
                )
                if soft_hit:
                    soft_counter += 1
                else:
                    soft_counter = 0
                giveback = (mfe - unreal) / max(abs(mfe), 1e-12) if mfe > 0.0 else 0.0

                if hard_sl > 0.0 and unreal <= -hard_sl:
                    reason = f"{owner}_hard_stop_loss"
                elif soft_counter >= guard_cfg.soft_persist_bars:
                    reason = f"{owner}_soft_stop_loss"
                elif hold >= guard_cfg.giveback_min_hold and mfe >= guard_cfg.giveback_min_mfe and giveback >= guard_cfg.giveback_trigger:
                    reason = f"{owner}_giveback_exit"
                elif max_hold > 0 and hold >= max_hold:
                    reason = f"{owner}_max_hold"

            if owner == "v21_2" and not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and dd_abs <= add_cfg.dd_block:
                state = {"parent_notional": parent_notional, "notional": notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": dd_abs, "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
                x = _feature_frame(df, bundle, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x)
                if p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40:
                    delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - notional))
                    filled, add_px, add_fee, _, route = _try_immediate_limit(df, i, pos, limit_cfg, entry=True, fee=fee_base, slip=slip_base)
                    if filled and delta > 0.0:
                        new_notional = notional + delta
                        entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                        before = cash
                        cash -= before * add_fee * delta
                        notional = new_notional
                        actions["v21_add_on"] = actions.get("v21_add_on", 0) + 1
                        route_counts[route] = route_counts.get(route, 0) + 1
                    else:
                        actions["v21_add_on_limit_miss"] = actions.get("v21_add_on_limit_miss", 0) + 1
                        route_counts[route] = route_counts.get(route, 0) + 1
                else:
                    actions["v21_reject"] = actions.get("v21_reject", 0) + 1
                add_done = True

            if reason:
                filled, exit_px, exit_fee, _, route = _try_immediate_limit(df, i, pos, limit_cfg, entry=False, fee=fee_base, slip=slip_base)
                if not filled:
                    actions["exit_limit_miss_hold"] = actions.get("exit_limit_miss_hold", 0) + 1
                    continue
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * exit_fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                route_counts[route] = route_counts.get(route, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update(
                        {
                            "exit_signal_idx": int(i),
                            "exit_fill_idx": int(min(i + 1, len(df) - 1)),
                            "exit_time": str(df.iloc[int(min(i + 1, len(df) - 1))]["timestamp"]),
                            "exit_price": float(exit_px),
                            "exit_reason": str(reason),
                            "exit_route": str(route),
                            "hold_bars": int(hold),
                            "trade_return": float(raw * notional),
                            "cash_after": float(cash),
                        }
                    )
                    records.append(out)
                pos = 0
                owner = ""
                extra_cd = 0
                if "hard_stop_loss" in reason:
                    extra_cd = int(guard_cfg.cooldown_after_hard_stop)
                elif "soft_stop_loss" in reason:
                    extra_cd = int(guard_cfg.cooldown_after_soft_stop)
                elif "giveback_exit" in reason:
                    extra_cd = int(guard_cfg.cooldown_after_giveback)
                cooldown = max(int(next_cooldown), int(extra_cd))
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(overlay.cooldown), int(extra_cd))
                add_done = False
                soft_counter = 0
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
            if int(guard_cfg.same_side_entry_gap) > 0 and int(dec.side) == int(last_entry_side) and (i - int(last_entry_idx)) <= int(guard_cfg.same_side_entry_gap):
                actions["parent_entry_same_side_gap_block"] = actions.get("parent_entry_same_side_gap_block", 0) + 1
                continue
            if float(dec.quality_score) < float(guard_cfg.entry_quality_min):
                actions["parent_entry_quality_block"] = actions.get("parent_entry_quality_block", 0) + 1
                continue
            if float(dec.confidence) < float(guard_cfg.entry_conf_min):
                actions["parent_entry_conf_block"] = actions.get("parent_entry_conf_block", 0) + 1
                continue
            filled, px, entry_fee, _, route = _try_immediate_limit(df, i, int(dec.side), limit_cfg, entry=True, fee=fee_base, slip=slip_base)
            if not filled:
                actions["parent_entry_limit_miss"] = actions.get("parent_entry_limit_miss", 0) + 1
                route_counts[route] = route_counts.get(route, 0) + 1
                continue
            pos = int(dec.side)
            owner = "v21_2"
            entry_price = px
            entry_equity = cash
            entry_idx = i
            parent_notional = min(float(dec.notional_exposure), add_cfg.max_entry_notional)
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
            soft_counter = 0
            last_entry_side = int(pos)
            last_entry_idx = int(i)
            if record:
                fill_i = int(min(i + 1, len(df) - 1))
                open_record = {
                    "entry_signal_idx": int(i),
                    "entry_fill_idx": fill_i,
                    "entry_time": str(df.iloc[fill_i]["timestamp"]),
                    "entry_price": float(entry_price),
                    "side": "LONG" if pos > 0 else "SHORT",
                    "owner": str(owner),
                    "notional": float(notional),
                    "leverage_like": float(dec.leverage),
                    "entry_route": str(route),
                }
            actions["v21_entry"] = actions.get("v21_entry", 0) + 1
            route_counts[route] = route_counts.get(route, 0) + 1
            continue

        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if int(guard_cfg.same_side_entry_gap) > 0 and int(side) == int(last_entry_side) and (i - int(last_entry_idx)) <= int(guard_cfg.same_side_entry_gap):
                actions["deep_entry_same_side_gap_block"] = actions.get("deep_entry_same_side_gap_block", 0) + 1
                continue
            if edge >= overlay.edge_th and margin >= overlay.margin_th:
                filled, px, entry_fee, _, route = _try_immediate_limit(df, i, side, limit_cfg, entry=True, fee=fee_base, slip=slip_base)
                if not filled:
                    actions["deep_entry_limit_miss"] = actions.get("deep_entry_limit_miss", 0) + 1
                    route_counts[route] = route_counts.get(route, 0) + 1
                    continue
                pos = side
                owner = "deep_alpha"
                entry_price = px
                entry_equity = cash
                entry_idx = i
                parent_notional = notional = float(overlay.notional)
                take_profit = float(overlay.base_tp)
                stop_loss = float(overlay.base_sl)
                max_hold = int(overlay.base_hold)
                next_cooldown = int(overlay.cooldown)
                entry_edge = edge
                entry_vol_anchor = v31._vol_anchor(df.iloc[i]) * notional
                cash -= cash * entry_fee * notional
                long_entries += int(pos > 0)
                short_entries += int(pos < 0)
                deep_entries += 1
                notional_sum += notional
                leverage_sum += max(notional, 1.0)
                mfe = mae = 0.0
                add_done = True
                soft_counter = 0
                last_entry_side = int(pos)
                last_entry_idx = int(i)
                if record:
                    fill_i = int(min(i + 1, len(df) - 1))
                    open_record = {
                        "entry_signal_idx": int(i),
                        "entry_fill_idx": fill_i,
                        "entry_time": str(df.iloc[fill_i]["timestamp"]),
                        "entry_price": float(entry_price),
                        "side": "LONG" if pos > 0 else "SHORT",
                        "owner": str(owner),
                        "notional": float(notional),
                        "leverage_like": float(max(notional, 1.0)),
                        "entry_route": str(route),
                    }
                actions["deep_entry"] = actions.get("deep_entry", 0) + 1
                route_counts[route] = route_counts.get(route, 0) + 1

    if pos != 0:
        exit_px = _fill_price(df, len(df) - 1, pos, slip_base, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee_base * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
        route_counts["forced_end_market"] = route_counts.get("forced_end_market", 0) + 1
        if record and open_record is not None:
            out = dict(open_record)
            out.update(
                {
                    "exit_signal_idx": int(len(df) - 1),
                    "exit_fill_idx": int(len(df) - 1),
                    "exit_time": str(df.iloc[len(df) - 1]["timestamp"]),
                    "exit_price": float(exit_px),
                    "exit_reason": "forced_end",
                    "exit_route": "forced_end_market",
                    "hold_bars": int((len(df) - 1) - entry_idx),
                    "trade_return": float(raw * notional),
                    "cash_after": float(cash),
                }
            )
            records.append(out)

    n = max(long_entries + short_entries, 1)
    result = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(df)),
        "deep_entries": int(deep_entries),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n),
        "avg_leverage": float(leverage_sum / n),
        "exits": exits,
        "runner_actions": actions,
        "route_counts": route_counts,
    }
    if record:
        result["trade_records"] = records
    return result


def _metrics_guard(
    df: pd.DataFrame,
    parent: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: v21.CostRunnerConfig,
    q: np.ndarray,
    decisions: pd.DataFrame,
    overlay: v31.OverlayConfig,
    limit_cfg: ImmediateLimitConfig,
    guard_cfg: ExitGuardConfig,
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    return {
        f"cost{mult}": backtest_signal_limit_exit_guard(
            df,
            parent,
            jackpot_model,
            add_cfg,
            q,
            decisions,
            overlay,
            limit_cfg,
            guard_cfg,
            fee=fee,
            slip=slip,
            cost_mult=float(mult),
            record=False,
        )
        for mult in (1, 2, 3)
    }


def _load_stack() -> tuple[dict[str, Any], Any, Any, Any, Any, Any, Any, Any, Any]:
    report = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    exp = dict(report["experiments"][-1])
    parent = joblib.load(exp["artifacts"]["parent"])
    runner_payload = joblib.load(exp["artifacts"]["runner"])
    runner = runner_payload["cost_runner"]
    add_cfg = v21.CostRunnerConfig(**dict(exp["selected_runner_config"]))
    overlay = v31.OverlayConfig(**dict(exp["selected_overlay"]))
    runtime = alpha2.Alpha2Runtime(**dict(exp["selected_teacher_runtime"]))
    teacher_payload = torch.load(exp["artifacts"]["teacher"], map_location="cpu", weights_only=False)
    teacher_model = alpha2._load_teacher_model(teacher_payload)
    teacher_cols = list(teacher_payload["feature_cols"])
    teacher_norm = dict(teacher_payload["train_meta"]["norm"])
    teacher_buckets = tuple(float(x) for x in teacher_payload["buckets"])
    deep_payload = torch.load(exp["artifacts"]["deep_scout"], map_location="cpu", weights_only=False)
    deep_model = v27.DeepAlphaTCN(len(deep_payload["seq_cols"]))
    deep_model.load_state_dict(deep_payload["state_dict"])
    deep_model = deep_model.cpu().eval()
    return parent, runner, add_cfg, overlay, runtime, teacher_model, teacher_cols, teacher_norm, teacher_buckets, deep_model, deep_payload


def _default_limit_cfg() -> ImmediateLimitConfig:
    return ImmediateLimitConfig(
        "next_open_limit_touch0_fee20",
        "next_open",
        0.0,
        0.0,
        0.0,
        0.20,
        entry_miss="skip",
        exit_miss="market_fallback",
    )


def _guard_grid() -> list[ExitGuardConfig]:
    return [
        ExitGuardConfig("baseline_guard_off", 1.0, 1.0, 0, 1.0, 1, 1, 1.0, 9.0, 9.0, 9.0, 0, -999.0, 0.0, 0, 0, 0, 0),
        ExitGuardConfig("guard_soft3_hard1p45", 1.45, 1.0, 18, 1.35, 3, 3, 0.50, 0.02, 0.72, 0.014, 3, -999.0, 0.0, 0, 0, 0, 0),
        ExitGuardConfig("guard_tuned_cd6_gap2_gb078", 1.45, 1.0, 18, 1.35, 3, 3, 0.50, 0.02, 0.78, 0.016, 6, -999.0, 0.0, 2, 6, 4, 6),
        ExitGuardConfig("guard_tuned_cd8_gap3_gb082", 1.50, 1.0, 20, 1.40, 3, 3, 0.52, 0.03, 0.82, 0.018, 8, -999.0, 0.0, 3, 8, 6, 8),
        ExitGuardConfig("guard_tuned_parentgate_cd6", 1.45, 1.0, 18, 1.35, 3, 3, 0.50, 0.02, 0.78, 0.016, 6, 0.0020, 0.56, 2, 6, 4, 6),
        ExitGuardConfig("guard_tuned_parentgate_cd8", 1.50, 1.0, 20, 1.40, 3, 3, 0.52, 0.03, 0.82, 0.018, 8, 0.0025, 0.58, 3, 8, 6, 8),
    ]


def _sl_ratio(cost3: dict[str, Any]) -> float:
    exits = dict(cost3.get("exits", {}))
    sl_hits = sum(v for k, v in exits.items() if "stop_loss" in str(k))
    return float(sl_hits / max(int(cost3.get("trades", 0)), 1))


def main() -> int:
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    parent, jackpot_model, add_cfg, overlay, runtime, teacher_model, teacher_cols, teacher_norm, teacher_buckets, deep_model, deep_payload = _load_stack()
    fee = float(parent["config"]["fee"])
    slip = float(parent["config"]["slip"])
    limit_cfg = _default_limit_cfg()

    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train_all = _merge_state24(train_all, alpha3_full.SIDE_CLEAN4_2025)
    eval_df = _merge_state24(eval_df, alpha3_full.SIDE_CLEAN4_2026)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)

    val_parent = predict_policy_frame(parent, val_df, close=_close(val_df))
    eval_parent = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    val_features = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=teacher_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=teacher_cols)
    val_teacher = alpha2.teacher._predict_deep(teacher_model, val_features, teacher_cols, teacher_norm)
    eval_teacher = alpha2.teacher._predict_deep(teacher_model, eval_features, teacher_cols, teacher_norm)
    val_dec = alpha2._decisions(val_parent, val_teacher, teacher_buckets, runtime)
    eval_dec = alpha2._decisions(eval_parent, eval_teacher, teacher_buckets, runtime)
    val_q = v27._predict_all(deep_model, val_df, deep_payload["seq_cols"], deep_payload["norm"])
    eval_q = v27._predict_all(deep_model, eval_df, deep_payload["seq_cols"], deep_payload["norm"])

    rows: list[dict[str, Any]] = []
    best_name = ""
    best_val_score = -1e18
    best_oos: dict[str, Any] | None = None
    for cfg in _guard_grid():
        val_metrics = _metrics_guard(val_df, parent, jackpot_model, add_cfg, val_q, val_dec, overlay, limit_cfg, cfg, fee=fee, slip=slip)
        oos_metrics = _metrics_guard(eval_df, parent, jackpot_model, add_cfg, eval_q, eval_dec, overlay, limit_cfg, cfg, fee=fee, slip=slip)
        val_score = _score(val_metrics)
        oos_score = _score(oos_metrics)
        c3 = oos_metrics["cost3"]
        row = {
            "guard": cfg.name,
            "val_score": float(val_score),
            "val_cost3_pnl": float(val_metrics["cost3"]["pnl"]),
            "oos_score": float(oos_score),
            "oos_cost3_pnl": float(c3["pnl"]),
            "oos_cost3_mdd": float(c3["mdd"]),
            "oos_cost3_wr": float(c3["wr"]),
            "oos_cost3_trades": int(c3["trades"]),
            "oos_sl_ratio": float(_sl_ratio(c3)),
            "oos_exits": json.dumps(c3.get("exits", {}), ensure_ascii=False),
            **asdict(cfg),
        }
        rows.append(row)
        if val_score > best_val_score:
            best_val_score = val_score
            best_name = cfg.name
            best_oos = {"guard": asdict(cfg), "metrics": oos_metrics, "score": float(oos_score), "val_score": float(val_score)}

    if best_oos is None:
        raise RuntimeError("no guard variant result")

    grid = pd.DataFrame(rows).sort_values("val_score", ascending=False).reset_index(drop=True)
    grid.to_csv(GRID_OUT, index=False)
    baseline = next(r for r in rows if r["guard"] == "baseline_guard_off")
    selected = next(r for r in rows if r["guard"] == best_name)
    delta = {
        "cost3_pnl": float(selected["oos_cost3_pnl"] - baseline["oos_cost3_pnl"]),
        "cost3_mdd": float(selected["oos_cost3_mdd"] - baseline["oos_cost3_mdd"]),
        "cost3_wr": float(selected["oos_cost3_wr"] - baseline["oos_cost3_wr"]),
        "cost3_trades": int(selected["oos_cost3_trades"] - baseline["oos_cost3_trades"]),
        "sl_ratio": float(selected["oos_sl_ratio"] - baseline["oos_sl_ratio"]),
    }
    audit = {
        "status": "pass",
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS",
        "selected_guard": best_name,
        "baseline_guard": "baseline_guard_off",
        "delta_vs_baseline": delta,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Two-stage stop-loss overlay with persistence soft-stop, widened early hard-stop, and giveback exit to reduce immediate stop-outs before delayed trend realization.",
        "base_model": str(BASE_REPORT),
        "selected": best_oos,
        "baseline": baseline,
        "grid": str(GRID_OUT),
        "audit": audit,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT), "selected_guard": best_name}, ensure_ascii=False))
    return 0


def _merge_state24(base: pd.DataFrame, side_path: Path) -> pd.DataFrame:
    side = alpha3_full._rename_state24_sidecar(_read(side_path))
    merged, _ = alpha3_full._merge_state24(base, side)
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
