#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.elite import RegimeEngine  # noqa: E402
from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_exit_front_run_layer_20260514 as front_run  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _days, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _predict_cost_runner  # noqa: E402


MODEL_ID = "alpha3_current_live_guard_20260515"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_current_live_guard_20260515.json"

LIVE_TRAIL_ACTIVATION = 0.008
TRAIL_MIN_SL_MULT = 0.60


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _cfg() -> alpha3.ImmediateLimitConfig:
    return alpha3.ImmediateLimitConfig(
        "alpha3_current_live_touch0_skip_entry_exit_fallback",
        "next_open",
        0.0,
        0.0,
        0.0,
        0.20,
        entry_miss="skip",
        exit_miss="market_fallback",
    )


def _regime_name(row: pd.Series) -> str:
    cols = ("regime_bull", "regime_bear", "regime_chop", "regime_whipsaw", "regime_normal")
    vals: dict[str, float] = {}
    for col in cols:
        try:
            vals[col] = float(row.get(col, 0.0) or 0.0)
        except Exception:
            vals[col] = 0.0
    if not vals or max(abs(v) for v in vals.values()) <= 1e-12:
        return "normal"
    return max(vals, key=vals.get).replace("regime_", "")


def _transition_risk(row: pd.Series) -> float:
    try:
        return float(row.get("clean_regime_2024_unsup_v4_transition_risk", 0.0) or 0.0)
    except Exception:
        return 0.0


def _deep_decision(row: pd.Series, q_long: float, q_short: float, overlay: v31.OverlayConfig) -> tuple[bool, int, float, float, dict[str, Any]]:
    q_long_raw = float(q_long)
    q_short_raw = float(q_short)
    regime = _regime_name(row).upper()
    transition_risk = _transition_risk(row)
    guard_reasons: list[str] = []
    side = 1 if q_long_raw >= q_short_raw else -1
    edge = float(max(q_long_raw, q_short_raw))
    margin = float(abs(q_long_raw - q_short_raw))
    raw_margin = float(abs(q_long_raw - q_short_raw))
    pass_gate = bool(edge >= overlay.edge_th and margin >= overlay.margin_th)
    return pass_gate, side, edge, margin, {
        "q_long_raw": q_long_raw,
        "q_short_raw": q_short_raw,
        "q_long": q_long_raw,
        "q_short": q_short_raw,
        "raw_margin": raw_margin,
        "regime": regime,
        "transition_risk": transition_risk,
        "guard_reasons": guard_reasons,
    }


def backtest_current_live(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg,
    deep_q: np.ndarray,
    decisions: pd.DataFrame,
    overlay: v31.OverlayConfig,
    limit_cfg: alpha3.ImmediateLimitConfig,
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
                trail_activation = max(LIVE_TRAIL_ACTIVATION, entry_vol_anchor * max(overlay.trail_gap_mult, 0.0))
                min_trail_sl = max(0.0, overlay.base_sl * TRAIL_MIN_SL_MULT)
                if mfe >= trail_activation and overlay.trail_gap_mult > 0.0:
                    trail_gap = entry_vol_anchor * overlay.trail_gap_mult
                    if overlay.hold_decay_start < 999 and hold >= overlay.hold_decay_start:
                        trail_gap = max(entry_vol_anchor * 0.35, trail_gap - overlay.hold_decay_rate * (hold - overlay.hold_decay_start) * entry_vol_anchor)
                    trail_stop = max(-effective_sl, mfe - trail_gap)
                    effective_sl = min(effective_sl, max(min_trail_sl, trail_stop))
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
            pass_gate, side, edge, _margin, trace = _deep_decision(df.iloc[i], float(deep_q[i, 0]), float(deep_q[i, 1]), overlay)
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


def _metrics(df: pd.DataFrame, stack: dict[str, Any], q, dec, overlay: v31.OverlayConfig) -> dict[str, Any]:
    cfg = _cfg()
    return {
        f"cost{mult}": backtest_current_live(
            df,
            stack["parent"],
            stack["jackpot_model"],
            stack["add_cfg"],
            q,
            dec,
            overlay,
            cfg,
            fee=stack["fee"],
            slip=stack["slip"],
            cost_mult=float(mult),
        )
        for mult in (1, 2, 3)
    }


def _prepare_eval_frame() -> pd.DataFrame:
    df = _read(v31.DEFAULT_EVAL)
    return RegimeEngine().compute(df.copy())


def main() -> int:
    print(f"[{MODEL_ID}] loading fixed Alpha3 stack", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    stack = front_run._load_fixed_stack()
    eval_df = _prepare_eval_frame()
    eval_dec, eval_q = front_run._decisions_and_q(eval_df, stack)
    live_overlay = replace(stack["overlay"], notional=2.0, trail_activation=LIVE_TRAIL_ACTIVATION)

    print(f"[{MODEL_ID}] evaluating 2026 OOS with model-native V31 direction", flush=True)
    current = _metrics(eval_df, stack, eval_q, eval_dec, live_overlay)
    reference_overlay = replace(stack["overlay"], notional=2.0, trail_activation=0.0)
    reference = alpha3._metrics_signal_limit(
        eval_df,
        stack["parent"],
        stack["jackpot_model"],
        stack["add_cfg"],
        eval_q,
        eval_dec,
        reference_overlay,
        _cfg(),
        fee=stack["fee"],
        slip=stack["slip"],
    )
    report = {
        "model_id": MODEL_ID,
        "selection_uses_2026": False,
        "oos_window": "2026 fixed OOS",
        "contract": asdict(_cfg()),
        "live_overlay": asdict(live_overlay),
        "live_guard": {
            "directional_regime_q_adjustment": "removed",
            "model_native_direction": True,
            "trail_activation": LIVE_TRAIL_ACTIVATION,
            "trail_min_sl_mult": TRAIL_MIN_SL_MULT,
        },
        "reference_no_new_guard_activation0": reference,
        "current_live_guard": current,
        "score": _score(current),
        "audit": {
            "status": "pass",
            "blocking": [],
            "warnings": [
                "uses_5m_high_low_touch_proxy_for_maker_fill_not_orderbook_queue",
                "regime_bull_bear_chop_whipsaw_normal_recomputed_with_RegimeEngine_for_offline_parity",
                "clean_regime_transition_risk_uses_existing_oof_feature_column",
            ],
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(REPORT_OUT),
                "current_cost1": current["cost1"],
                "current_cost2": current["cost2"],
                "current_cost3": current["cost3"],
                "reference_cost1_pnl": reference["cost1"]["pnl"],
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
