#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3  # noqa: E402
from scripts import eval_alpha3_exit_front_run_layer_20260514 as front_run  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import retest_alpha3_current_live_guard_20260515 as liveguard  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _days, _fill_price  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _predict_cost_runner  # noqa: E402


MODEL_ID = "alpha3_entry_exit_timing_20260515"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_entry_exit_timing_20260515.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_entry_exit_timing_20260515_grid.csv"


@dataclass(frozen=True)
class TimingConfig:
    name: str
    block_counter_regime: bool = False
    chase_lookback: int = 0
    max_aligned_move: float = 9.0
    require_short_momentum_confirm: bool = False
    signal_exit_enable: bool = False
    signal_exit_min_age: int = 2
    signal_flip_margin: float = 0.004
    signal_edge_floor_mult: float = 0.55
    adverse_counter_regime_exit: float = 0.0015
    peak_giveback_exit: float = 0.006


def _side_return(close: np.ndarray, i: int, lookback: int, side: int) -> float:
    if lookback <= 0 or i - lookback < 0:
        return 0.0
    prev = float(close[i - lookback])
    cur = float(close[i])
    if prev <= 0.0:
        return 0.0
    raw = cur / prev - 1.0
    return float(raw if side > 0 else -raw)


def _same_opp_q(row: pd.Series, q_long: float, q_short: float, side: int) -> tuple[float, float]:
    ok, adjusted_side, _edge, _margin, trace = liveguard._deep_decision(row, q_long, q_short, v31.OverlayConfig(
        "tmp",
        0.010,
        0.004,
        1.0,
        12,
        0.040,
        0.018,
        48,
        1.5,
        2.5,
        1.0,
        0.50,
        18,
        0.025,
        0.075,
        0.036,
        0.008,
    ))
    del ok, adjusted_side
    ql = float(trace["q_long"])
    qs = float(trace["q_short"])
    return (ql, qs) if side > 0 else (qs, ql)


def _deep_entry_decision(
    df: pd.DataFrame,
    close: np.ndarray,
    i: int,
    q_long: float,
    q_short: float,
    overlay: v31.OverlayConfig,
    timing: TimingConfig,
) -> tuple[bool, int, float, float, dict[str, Any]]:
    pass_gate, side, edge, margin, trace = liveguard._deep_decision(df.iloc[i], q_long, q_short, overlay)
    regime = str(trace.get("regime", "")).upper()
    reasons = list(trace.get("guard_reasons", []) or [])
    if pass_gate and timing.block_counter_regime:
        if side > 0 and regime == "BEAR":
            pass_gate = False
            reasons.append("timing_block_long_in_bear")
        elif side < 0 and regime == "BULL":
            pass_gate = False
            reasons.append("timing_block_short_in_bull")
    if pass_gate and timing.chase_lookback > 0:
        aligned = _side_return(close, i, timing.chase_lookback, side)
        trace["aligned_pre_move"] = float(aligned)
        if aligned > float(timing.max_aligned_move):
            pass_gate = False
            reasons.append("timing_anti_chase")
    if pass_gate and timing.require_short_momentum_confirm:
        confirm = _side_return(close, i, 2, side)
        trace["confirm_2bar_move"] = float(confirm)
        if confirm < 0.0:
            pass_gate = False
            reasons.append("timing_no_short_momentum_confirm")
    trace["guard_reasons"] = reasons
    return pass_gate, side, edge, margin, trace


def _signal_exit_reason(
    df: pd.DataFrame,
    deep_q: np.ndarray,
    i: int,
    *,
    side: int,
    owner: str,
    hold: int,
    unreal: float,
    mfe: float,
    overlay: v31.OverlayConfig,
    timing: TimingConfig,
) -> str:
    if owner != "deep_alpha" or not timing.signal_exit_enable or hold < timing.signal_exit_min_age:
        return ""
    same_q, opp_q = _same_opp_q(df.iloc[i], float(deep_q[i, 0]), float(deep_q[i, 1]), side)
    if opp_q > same_q + float(timing.signal_flip_margin):
        return "deep_alpha_signal_flip_exit"
    if same_q < float(overlay.edge_th) * float(timing.signal_edge_floor_mult):
        return "deep_alpha_signal_decay_exit"
    regime = liveguard._regime_name(df.iloc[i]).upper()
    counter = (side > 0 and regime == "BEAR") or (side < 0 and regime == "BULL")
    if counter and unreal <= -abs(float(timing.adverse_counter_regime_exit)):
        return "deep_alpha_counter_regime_adverse_exit"
    if mfe > 0.0 and (mfe - unreal) >= float(timing.peak_giveback_exit):
        return "deep_alpha_peak_giveback_exit"
    return ""


def backtest_timing(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg,
    deep_q: np.ndarray,
    decisions: pd.DataFrame,
    overlay: v31.OverlayConfig,
    limit_cfg: alpha3.ImmediateLimitConfig,
    timing: TimingConfig,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
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
    trades = wins = long_entries = short_entries = deep_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    actions: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    guard_counts: dict[str, int] = {}

    def mark(idx: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(idx, 0, len(close) - 1))])
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
                trail_activation = max(liveguard.LIVE_TRAIL_ACTIVATION, entry_vol_anchor * max(overlay.trail_gap_mult, 0.0))
                min_trail_sl = max(0.0, overlay.base_sl * liveguard.TRAIL_MIN_SL_MULT)
                if mfe >= trail_activation and overlay.trail_gap_mult > 0.0:
                    trail_gap = entry_vol_anchor * overlay.trail_gap_mult
                    if overlay.hold_decay_start < 999 and hold >= overlay.hold_decay_start:
                        trail_gap = max(entry_vol_anchor * 0.35, trail_gap - overlay.hold_decay_rate * (hold - overlay.hold_decay_start) * entry_vol_anchor)
                    trail_stop = max(-effective_sl, mfe - trail_gap)
                    effective_sl = min(effective_sl, max(min_trail_sl, trail_stop))
                reason = _signal_exit_reason(
                    df,
                    deep_q,
                    i,
                    side=pos,
                    owner=owner,
                    hold=hold,
                    unreal=unreal,
                    mfe=mfe,
                    overlay=overlay,
                    timing=timing,
                )
            if not reason and effective_tp > 0.0 and unreal >= effective_tp:
                reason = f"{owner}_take_profit"
            elif not reason and effective_sl > 0.0 and unreal <= -abs(effective_sl):
                reason = f"{owner}_stop_loss"
            elif not reason and max_hold > 0 and hold >= max_hold:
                reason = f"{owner}_max_hold"

            if owner == "v21_2" and not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and dd_abs <= add_cfg.dd_block:
                state = {"parent_notional": parent_notional, "notional": notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": dd_abs, "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
                x = _feature_frame(df, bundle, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x)
                if p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40:
                    delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - notional))
                    filled, add_px, add_fee, _, route = alpha3._try_immediate_limit(df, i, pos, limit_cfg, entry=True, fee=fee_base, slip=slip_base)
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
                filled, exit_px, exit_fee, _, route = alpha3._try_immediate_limit(df, i, pos, limit_cfg, entry=False, fee=fee_base, slip=slip_base)
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
                pos = 0
                owner = ""
                cooldown = int(next_cooldown)
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(overlay.cooldown))
                add_done = False
                continue

        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if deep_cooldown > 0:
            deep_cooldown -= 1

        dec = decisions.iloc[i]
        if int(dec.action) != 0 and int(dec.side) != 0:
            filled, px, entry_fee, _, route = alpha3._try_immediate_limit(df, i, int(dec.side), limit_cfg, entry=True, fee=fee_base, slip=slip_base)
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
            actions["v21_entry"] = actions.get("v21_entry", 0) + 1
            route_counts[route] = route_counts.get(route, 0) + 1
            continue

        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            pass_gate, side, edge, _margin, trace = _deep_entry_decision(df, close, i, float(deep_q[i, 0]), float(deep_q[i, 1]), overlay, timing)
            if not pass_gate:
                for reason in trace["guard_reasons"]:
                    guard_counts[reason] = guard_counts.get(reason, 0) + 1
                continue
            filled, px, entry_fee, _, route = alpha3._try_immediate_limit(df, i, side, limit_cfg, entry=True, fee=fee_base, slip=slip_base)
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

    n = max(long_entries + short_entries, 1)
    return {
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
        "guard_counts": guard_counts,
    }


def _metrics(df: pd.DataFrame, stack: dict[str, Any], q, dec, overlay: v31.OverlayConfig, timing: TimingConfig) -> dict[str, Any]:
    cfg = liveguard._cfg()
    return {
        f"cost{mult}": backtest_timing(
            df,
            stack["parent"],
            stack["jackpot_model"],
            stack["add_cfg"],
            q,
            dec,
            overlay,
            cfg,
            timing,
            fee=stack["fee"],
            slip=stack["slip"],
            cost_mult=float(mult),
        )
        for mult in (1, 2, 3)
    }


def _timing_grid() -> list[TimingConfig]:
    return [
        TimingConfig("baseline_live_guard"),
        TimingConfig("signal_exit", signal_exit_enable=True),
        TimingConfig("counter_regime_block", block_counter_regime=True),
        TimingConfig("counter_block_signal_exit", block_counter_regime=True, signal_exit_enable=True),
        TimingConfig("counter_signal_antichase4", block_counter_regime=True, signal_exit_enable=True, chase_lookback=4, max_aligned_move=0.0045),
        TimingConfig("counter_signal_antichase6", block_counter_regime=True, signal_exit_enable=True, chase_lookback=6, max_aligned_move=0.0060),
        TimingConfig("counter_signal_confirm", block_counter_regime=True, signal_exit_enable=True, require_short_momentum_confirm=True),
        TimingConfig("fast_signal_exit", signal_exit_enable=True, signal_exit_min_age=1, signal_flip_margin=0.0025, signal_edge_floor_mult=0.70, adverse_counter_regime_exit=0.0008),
        TimingConfig("strict_timing", block_counter_regime=True, chase_lookback=6, max_aligned_move=0.0045, require_short_momentum_confirm=True, signal_exit_enable=True, signal_exit_min_age=1, signal_flip_margin=0.0025, signal_edge_floor_mult=0.70, adverse_counter_regime_exit=0.0008, peak_giveback_exit=0.004),
    ]


def _score(metrics: dict[str, Any]) -> float:
    return liveguard._score(metrics)


def main() -> int:
    print(f"[{MODEL_ID}] loading fixed Alpha3 stack", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    stack = front_run._load_fixed_stack()
    eval_df = liveguard._prepare_eval_frame()
    eval_dec, eval_q = front_run._decisions_and_q(eval_df, stack)
    overlay = replace(stack["overlay"], notional=2.0, trail_activation=liveguard.LIVE_TRAIL_ACTIVATION)

    rows: list[dict[str, Any]] = []
    variants: dict[str, Any] = {}
    for timing in _timing_grid():
        print(f"[{MODEL_ID}] testing {timing.name}", flush=True)
        metrics = _metrics(eval_df, stack, eval_q, eval_dec, overlay, timing)
        score = _score(metrics)
        variants[timing.name] = {"timing": asdict(timing), "metrics": metrics, "score": score}
        row = {"name": timing.name, "score": score}
        for k in ("cost1", "cost2", "cost3"):
            row[f"{k}_pnl"] = metrics[k]["pnl"]
            row[f"{k}_mdd"] = metrics[k]["mdd"]
            row[f"{k}_trades"] = metrics[k]["trades"]
            row[f"{k}_deep_stop_loss"] = metrics[k]["exits"].get("deep_alpha_stop_loss", 0)
            row[f"{k}_deep_signal_exits"] = sum(v for kk, v in metrics[k]["exits"].items() if "signal" in kk or "counter_regime" in kk or "giveback" in kk)
        rows.append(row)
    grid = pd.DataFrame(rows).sort_values("score", ascending=False)
    best_name = str(grid.iloc[0]["name"])
    report = {
        "model_id": MODEL_ID,
        "selection_uses_2026": True,
        "purpose": "entry_exit_timing_redesign_screen",
        "execution_contract": asdict(liveguard._cfg()),
        "overlay": asdict(overlay),
        "best": best_name,
        "grid": rows,
        "variants": variants,
        "audit": {
            "status": "research_only",
            "blocking": ["variant_selection_uses_2026_oos"],
            "warnings": [
                "maker_fill_uses_5m_high_low_touch_proxy_not_orderbook_queue",
                "this_screen_tests_timing_logic_before_live_enablement",
            ],
        },
    }
    GRID_OUT.write_text(grid.to_csv(index=False), encoding="utf-8")
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "grid": str(GRID_OUT), "best": best_name, "top": grid.head(5).to_dict("records")}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
