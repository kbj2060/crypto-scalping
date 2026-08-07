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
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _days, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner  # noqa: E402


MODEL_ID = "alpha3_unified_exit_overlay_20260514"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_unified_exit_overlay_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_unified_exit_overlay_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_unified_exit_overlay_20260514_grid.csv"
LEDGER_OUT = ROOT / "data/ensemble/reports/alpha3_unified_exit_overlay_20260514_cost1_ledger.csv"


@dataclass(frozen=True)
class UnifiedExitConfig:
    name: str
    parent_opposite_conf: float
    parent_quality_floor: float
    cash_conf: float
    deep_opposite_edge: float
    deep_opposite_margin: float
    bail_min_hold: int
    bail_loss_floor: float
    bail_giveback: float
    soft_sl: float
    soft_sl_min_hold: int
    trail_start: float
    trail_gap_mult: float
    trail_min_gap: float
    trail_decay_start: int
    trail_decay_rate: float


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _selected_alpha3_runtime() -> alpha2.Alpha2Runtime:
    audit = json.loads(alpha3.ALPHA2_AUDIT.read_text(encoding="utf-8"))
    runtime = dict(audit.get("selected_runtime", {}) or {})
    return alpha2.Alpha2Runtime(
        name=str(runtime.get("name", "noflip_c0.56_parent_scale1.10")),
        confidence=float(runtime.get("confidence", 0.56)),
        parent_notional_scale=float(runtime.get("parent_notional_scale", 1.10)),
        max_notional=float(runtime.get("max_notional", 2.75)),
    )


def _selected_limit_cfg() -> alpha3.ImmediateLimitConfig:
    audit = json.loads(alpha3.AUDIT_OUT.read_text(encoding="utf-8"))
    cfg = dict(audit.get("selected_config", {}) or {})
    return alpha3.ImmediateLimitConfig(
        name=str(cfg.get("name", "next_open_limit_offset2_entry_fallback_fee20")),
        anchor=str(cfg.get("anchor", "next_open")),
        entry_offset_bps=float(cfg.get("entry_offset_bps", 2.0)),
        exit_offset_bps=float(cfg.get("exit_offset_bps", 2.0)),
        penetration_bps=float(cfg.get("penetration_bps", 0.5)),
        maker_fee_mult=float(cfg.get("maker_fee_mult", 0.20)),
        entry_miss=str(cfg.get("entry_miss", "market_fallback")),
        exit_miss=str(cfg.get("exit_miss", "market_fallback")),
    )


def _configs() -> list[UnifiedExitConfig]:
    disabled = UnifiedExitConfig(
        "noop_alpha3_baseline",
        2.0,
        99.0,
        2.0,
        99.0,
        99.0,
        999,
        -99.0,
        99.0,
        0.0,
        999,
        99.0,
        0.0,
        99.0,
        999,
        0.0,
    )
    rows = [disabled]
    for conf in (0.78, 0.86, 0.92):
        rows.append(
            UnifiedExitConfig(
                f"opposite_parent_loss_c{conf:.2f}",
                conf,
                -0.05,
                2.0,
                99.0,
                99.0,
                2,
                0.000,
                0.010,
                0.010,
                2,
                99.0,
                0.0,
                99.0,
                999,
                0.0,
            )
        )
        rows.append(
            UnifiedExitConfig(
                f"opposite_parent_cash_profitlock_c{conf:.2f}",
                conf,
                -0.05,
                0.94,
                99.0,
                99.0,
                3,
                0.006,
                0.007,
                0.012,
                2,
                0.030,
                1.00,
                0.010,
                18,
                0.025,
            )
        )
    for edge, margin in ((0.010, 0.004), (0.014, 0.006), (0.018, 0.008)):
        rows.append(
            UnifiedExitConfig(
                f"deep_cross_softsl_e{edge:.3f}_m{margin:.3f}",
                2.0,
                99.0,
                2.0,
                edge,
                margin,
                2,
                0.002,
                0.008,
                0.010,
                1,
                99.0,
                0.0,
                99.0,
                999,
                0.0,
            )
        )
        rows.append(
            UnifiedExitConfig(
                f"deep_cross_trail_e{edge:.3f}_m{margin:.3f}",
                0.86,
                -0.05,
                0.96,
                edge,
                margin,
                3,
                0.004,
                0.008,
                0.012,
                2,
                0.024,
                0.85,
                0.007,
                12,
                0.030,
            )
        )
    for start, gap_mult, min_gap in ((0.018, 0.60, 0.006), (0.024, 0.80, 0.008), (0.030, 1.00, 0.010)):
        rows.append(
            UnifiedExitConfig(
                f"global_trail_start{start:.3f}_gap{gap_mult:.2f}",
                2.0,
                99.0,
                2.0,
                99.0,
                99.0,
                999,
                -99.0,
                99.0,
                0.0,
                999,
                start,
                gap_mult,
                min_gap,
                18,
                0.025,
            )
        )
    rows.extend(
        [
            UnifiedExitConfig("unified_balanced_a", 0.86, -0.05, 0.95, 0.014, 0.006, 3, 0.004, 0.008, 0.012, 2, 0.026, 0.80, 0.008, 14, 0.025),
            UnifiedExitConfig("unified_balanced_b", 0.92, -0.02, 0.97, 0.018, 0.008, 4, 0.000, 0.010, 0.014, 2, 0.030, 1.00, 0.010, 18, 0.020),
            UnifiedExitConfig("unified_aggressive_lock", 0.78, -0.08, 0.93, 0.010, 0.004, 2, 0.006, 0.006, 0.010, 1, 0.020, 0.65, 0.006, 10, 0.035),
        ]
    )
    return rows


def _adverse_signal(
    dec: pd.Series,
    deep_q: np.ndarray,
    idx: int,
    side: int,
    cfg: UnifiedExitConfig,
) -> tuple[bool, dict[str, Any]]:
    action = int(dec.get("action", ACTION_CASH))
    dec_side = int(dec.get("side", 0))
    conf = float(dec.get("confidence", 0.0) or 0.0)
    quality = float(dec.get("quality_score", 0.0) or 0.0)
    parent_opposite = (
        action != ACTION_CASH
        and dec_side == -int(side)
        and conf >= float(cfg.parent_opposite_conf)
        and quality >= float(cfg.parent_quality_floor)
    )
    parent_cash = action == ACTION_CASH and conf >= float(cfg.cash_conf)
    same_q = float(deep_q[idx, 0] if side > 0 else deep_q[idx, 1])
    opp_q = float(deep_q[idx, 1] if side > 0 else deep_q[idx, 0])
    deep_opposite = (
        opp_q >= float(cfg.deep_opposite_edge)
        and (opp_q - same_q) >= float(cfg.deep_opposite_margin)
    )
    info = {
        "parent_opposite": bool(parent_opposite),
        "parent_cash": bool(parent_cash),
        "deep_opposite": bool(deep_opposite),
        "parent_conf": conf,
        "parent_quality": quality,
        "same_q": same_q,
        "opp_q": opp_q,
    }
    return bool(parent_opposite or parent_cash or deep_opposite), info


def _global_exit_reason(
    df: pd.DataFrame,
    dec: pd.Series,
    deep_q: np.ndarray,
    idx: int,
    side: int,
    owner: str,
    hold: int,
    unreal: float,
    mfe: float,
    entry_vol_anchor: float,
    cfg: UnifiedExitConfig,
) -> tuple[str, dict[str, Any]]:
    adverse, info = _adverse_signal(dec, deep_q, idx, side, cfg)
    if adverse and hold >= int(cfg.soft_sl_min_hold) and cfg.soft_sl > 0.0 and unreal <= -abs(float(cfg.soft_sl)):
        return f"{owner}_unified_soft_sl", info
    if adverse and hold >= int(cfg.bail_min_hold):
        gave_back = (mfe - unreal) >= float(cfg.bail_giveback)
        weak_or_loss = unreal <= float(cfg.bail_loss_floor)
        if weak_or_loss or gave_back:
            return f"{owner}_unified_signal_bailout", info
    if cfg.trail_start < 90.0 and mfe >= float(cfg.trail_start):
        vol_gap = max(float(cfg.trail_min_gap), float(entry_vol_anchor) * float(cfg.trail_gap_mult))
        if hold >= int(cfg.trail_decay_start):
            decay = float(cfg.trail_decay_rate) * float(hold - int(cfg.trail_decay_start)) * max(float(entry_vol_anchor), 1e-12)
            vol_gap = max(float(cfg.trail_min_gap) * 0.50, vol_gap - decay)
        if (mfe - unreal) >= vol_gap:
            return f"{owner}_unified_trailing_stop", {**info, "trail_gap": float(vol_gap)}
    return "", info


def backtest_unified_exit(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    deep_q: np.ndarray,
    decisions: pd.DataFrame,
    overlay: v31.OverlayConfig,
    limit_cfg: alpha3.ImmediateLimitConfig,
    exit_cfg: UnifiedExitConfig,
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
    trades = wins = long_entries = short_entries = deep_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    actions: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    signal_counts: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        if pos > 0:
            raw = (px * (1.0 - slip_base) - entry_price) / max(entry_price, 1e-12)
        else:
            raw = (entry_price - px * (1.0 + slip_base)) / max(entry_price, 1e-12)
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
                if mfe > 0.0 and overlay.trail_gap_mult > 0.0:
                    trail_gap = entry_vol_anchor * overlay.trail_gap_mult
                    if overlay.hold_decay_start < 999 and hold >= overlay.hold_decay_start:
                        trail_gap = max(
                            entry_vol_anchor * 0.35,
                            trail_gap - overlay.hold_decay_rate * (hold - overlay.hold_decay_start) * entry_vol_anchor,
                        )
                    trail_stop = max(-effective_sl, mfe - trail_gap)
                    effective_sl = min(effective_sl, max(0.001, trail_stop))
            if effective_tp > 0.0 and unreal >= effective_tp:
                reason = f"{owner}_take_profit"
            elif effective_sl > 0.0 and unreal <= -abs(effective_sl):
                reason = f"{owner}_stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = f"{owner}_max_hold"

            signal_info: dict[str, Any] = {}
            if not reason:
                reason, signal_info = _global_exit_reason(
                    df,
                    decisions.iloc[i],
                    deep_q,
                    i,
                    pos,
                    owner,
                    hold,
                    unreal,
                    mfe,
                    entry_vol_anchor,
                    exit_cfg,
                )
                for key in ("parent_opposite", "parent_cash", "deep_opposite"):
                    if bool(signal_info.get(key, False)):
                        signal_counts[key] = signal_counts.get(key, 0) + 1

            if owner == "v21_2" and not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and dd_abs <= add_cfg.dd_block:
                state = {
                    "parent_notional": parent_notional,
                    "notional": notional,
                    "bars_since_entry": hold,
                    "unrealized": unreal,
                    "mfe": mfe,
                    "mae": mae,
                    "drawdown_abs": dd_abs,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "max_hold": max_hold,
                }
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
                        entry_vol_anchor = max(entry_vol_anchor, v31._vol_anchor(df.iloc[i]) * notional)
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
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update(
                        {
                            "exit_signal_timestamp": str(df["timestamp"].iloc[i]),
                            "exit_fill_timestamp": str(df["timestamp"].iloc[min(i + 1, len(df) - 1)]),
                            "exit_reason": str(reason),
                            "exit_route": str(route),
                            "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0),
                            "mfe_pct": float(mfe * 100.0),
                            "mae_pct": float(mae * 100.0),
                            "effective_tp": float(effective_tp),
                            "effective_sl": float(effective_sl),
                            "final_notional_exposure": float(notional),
                            **{f"signal_{k}": v for k, v in signal_info.items() if isinstance(v, (int, float, bool))},
                        }
                    )
                    records.append(out)
                pos = 0
                owner = ""
                cooldown = int(next_cooldown)
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(overlay.cooldown))
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
            entry_vol_anchor = v31._vol_anchor(df.iloc[i]) * notional
            cash -= cash * entry_fee * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += float(dec.leverage)
            mfe = mae = 0.0
            add_done = False
            actions["v21_entry"] = actions.get("v21_entry", 0) + 1
            route_counts[route] = route_counts.get(route, 0) + 1
            if record:
                open_record = {
                    "entry_signal_timestamp": str(df["timestamp"].iloc[i]),
                    "entry_fill_timestamp": str(df["timestamp"].iloc[min(i + 1, len(df) - 1)]),
                    "owner": owner,
                    "side": "LONG" if pos > 0 else "SHORT",
                    "entry_price": float(entry_price),
                    "notional_exposure": float(notional),
                    "leverage": float(dec.leverage),
                    "take_profit": float(take_profit),
                    "stop_loss": float(stop_loss),
                    "max_hold_bars": int(max_hold),
                }
            continue

        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= overlay.edge_th and margin >= overlay.margin_th:
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
                if record:
                    open_record = {
                        "entry_signal_timestamp": str(df["timestamp"].iloc[i]),
                        "entry_fill_timestamp": str(df["timestamp"].iloc[min(i + 1, len(df) - 1)]),
                        "owner": owner,
                        "side": "LONG" if pos > 0 else "SHORT",
                        "entry_price": float(entry_price),
                        "notional_exposure": float(notional),
                        "deep_q_long": float(ql),
                        "deep_q_short": float(qs),
                        "deep_edge": float(edge),
                        "deep_margin": float(margin),
                        "take_profit": float(take_profit),
                        "stop_loss": float(stop_loss),
                        "max_hold_bars": int(max_hold),
                    }

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
    out: dict[str, Any] = {
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
        "signal_counts": signal_counts,
    }
    if record:
        out["trade_records"] = records
    return out


def _metrics_unified(
    df: pd.DataFrame,
    parent: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    q: np.ndarray,
    decisions: pd.DataFrame,
    overlay: v31.OverlayConfig,
    limit_cfg: alpha3.ImmediateLimitConfig,
    exit_cfg: UnifiedExitConfig,
    *,
    fee: float,
    slip: float,
    record_cost1: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mult in (1, 2, 3):
        r = backtest_unified_exit(
            df,
            parent,
            jackpot_model,
            add_cfg,
            q,
            decisions,
            overlay,
            limit_cfg,
            exit_cfg,
            fee=fee,
            slip=slip,
            cost_mult=float(mult),
            record=bool(record_cost1 and mult == 1),
        )
        out[f"cost{mult}"] = r
    return out


def main() -> int:
    print(f"[{MODEL_ID}] loading fixed Alpha3 stack", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    rt = _selected_alpha3_runtime()
    limit_cfg = _selected_limit_cfg()
    selected_variant = next(v for v in l2._variants() if v.name == "alpha1_l2_conservative_fee20")
    overlay = selected_variant.overlay
    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    teacher_payload = torch.load(alpha3.TEACHER_MODEL, map_location="cpu", weights_only=False)
    teacher_model = alpha2._load_teacher_model(teacher_payload)
    feature_cols = list(teacher_payload["feature_cols"])
    norm = dict(dict(teacher_payload["train_meta"])["norm"])
    buckets = tuple(float(x) for x in teacher_payload["buckets"])
    fee = float(dict(parent["config"])["fee"])
    slip = float(dict(parent["config"])["slip"])

    train_all = _read(v31.DEFAULT_TRAIN)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    eval_df = _read(v31.DEFAULT_EVAL)
    contract_features = list(teacher_payload["feature_cols"])

    print(f"[{MODEL_ID}] rebuilding parent, teacher and V27 signals", flush=True)
    val_base_dec = predict_policy_frame(parent, val_df, close=_close(val_df))
    eval_base_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    val_features = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=contract_features)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=contract_features)
    val_pred = teacher._predict_deep(teacher_model, val_features, feature_cols, norm)
    eval_pred = teacher._predict_deep(teacher_model, eval_features, feature_cols, norm)
    val_dec = alpha2._decisions(val_base_dec, val_pred, buckets, rt)
    eval_dec = alpha2._decisions(eval_base_dec, eval_pred, buckets, rt)
    val_q = v31._predict_all(v27_model, val_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    print(f"[{MODEL_ID}] selecting unified exit config on 2025Q4", flush=True)
    rows: list[dict[str, Any]] = []
    best_cfg: UnifiedExitConfig | None = None
    best_score = -1e18
    for cfg in _configs():
        metrics = _metrics_unified(val_df, parent, jackpot_model, add_cfg, val_q, val_dec, overlay, limit_cfg, cfg, fee=fee, slip=slip)
        score = _score(metrics)
        row = {
            **asdict(cfg),
            "selection_score": score,
            "val_cost1_pnl": metrics["cost1"]["pnl"],
            "val_cost1_mdd": metrics["cost1"]["mdd"],
            "val_cost1_trades": metrics["cost1"]["trades"],
            "val_cost2_pnl": metrics["cost2"]["pnl"],
            "val_cost3_pnl": metrics["cost3"]["pnl"],
        }
        rows.append(row)
        print(
            f"[{MODEL_ID}] {cfg.name} val c1={metrics['cost1']['pnl']:.2f} "
            f"mdd={metrics['cost1']['mdd']:.2f} c2={metrics['cost2']['pnl']:.2f} c3={metrics['cost3']['pnl']:.2f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_cfg = cfg
    assert best_cfg is not None
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)

    print(f"[{MODEL_ID}] fixed 2026 OOS", flush=True)
    baseline_metrics = alpha3._metrics_signal_limit(eval_df, parent, jackpot_model, add_cfg, eval_q, eval_dec, overlay, limit_cfg, fee=fee, slip=slip)
    candidate_metrics = _metrics_unified(
        eval_df,
        parent,
        jackpot_model,
        add_cfg,
        eval_q,
        eval_dec,
        overlay,
        limit_cfg,
        best_cfg,
        fee=fee,
        slip=slip,
        record_cost1=True,
    )
    ledger = pd.DataFrame(candidate_metrics["cost1"].pop("trade_records", []))
    ledger.to_csv(LEDGER_OUT, index=False)
    experiments = [
        {"name": "alpha3_baseline_fixed_exit", "metrics": baseline_metrics, "score": _score(baseline_metrics)},
        {"name": f"alpha3_unified_exit::{best_cfg.name}", "config": asdict(best_cfg), "metrics": candidate_metrics, "score": _score(candidate_metrics)},
    ]
    for e in experiments:
        m = e["metrics"]
        print(
            f"[{MODEL_ID}] {e['name']} c1={m['cost1']['pnl']:.2f} mdd={m['cost1']['mdd']:.2f} "
            f"c2={m['cost2']['pnl']:.2f} c3={m['cost3']['pnl']:.2f}",
            flush=True,
        )

    warnings = [
        "signal_limit_fill_uses_5m_high_low_touch_proxy_not_queue_fill",
        "real_l2_queue_and_partial_fill_require_forward_shadow_validation",
    ]
    if candidate_metrics["cost1"]["pnl"] <= baseline_metrics["cost1"]["pnl"]:
        warnings.append("unified_exit_did_not_improve_alpha3_cost1_pnl")
    if candidate_metrics["cost1"]["mdd"] < baseline_metrics["cost1"]["mdd"]:
        warnings.append("unified_exit_worsened_alpha3_cost1_mdd")
    audit = {
        "status": "pass",
        "verdict": "promote_shadow_candidate" if not any(w.startswith("unified_exit_did_not") or w.startswith("unified_exit_worsened") for w in warnings) else "iterate",
        "blocking": [],
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "frozen_layers": {
            "hgb_parent": str(v31.DEFAULT_PARENT),
            "teacher": str(alpha3.TEACHER_MODEL),
            "v27_deep_scout": str(v31.DEFAULT_V27),
            "v21_2_jackpot": str(v31.DEFAULT_JACKPOT),
            "execution": asdict(limit_cfg),
        },
        "selected_config": asdict(best_cfg),
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha3 with a unified model-wide exit overlay. The entry stack, V21.2 runner, V27 scout, V31 deep exit, and limit/fallback execution stay fixed; the new layer adds opposite-entry bailout, CASH/Deep adverse soft-SL, and global trailing/time-decay exits to both parent-owned and deep-owned positions.",
        "experiments": experiments,
        "selection_grid": str(GRID_OUT),
        "cost1_ledger": str(LEDGER_OUT),
        "audit": audit,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT), "ledger": str(LEDGER_OUT), "selected": best_cfg.name}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
