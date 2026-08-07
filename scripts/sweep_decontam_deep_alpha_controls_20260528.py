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

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts import precision_retest_01965_alpha7_combo_20260527 as precision  # noqa: E402
from scripts import runtime_retest_alpha7_1_01965_decontam_20260528 as decontam  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.backtest_alpha3_exit_guard_persistence_20260527 import (  # noqa: E402
    _flow_bad,
    _regime_bad,
    _time_sl_mult,
    _try_immediate_limit,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _days, _fill_price, _json_default  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _predict_cost_runner  # noqa: E402


OUT_DIR = ROOT / "tmp/causal_regen_20260516/decontam_deep_alpha_controls_20260528"
GRID_OUT = OUT_DIR / "grid.csv"
SUMMARY_OUT = OUT_DIR / "summary.json"


@dataclass(frozen=True)
class Variant:
    name: str
    deep_side: str = "both"
    deep_notional_mult: float = 1.0
    deep_edge_mult: float = 1.0
    deep_margin_mult: float = 1.0
    deep_long_edge_mult: float = 1.0
    deep_long_margin_mult: float = 1.0
    deep_short_edge_mult: float = 1.0
    deep_short_margin_mult: float = 1.0
    deep_stop_cooldown_extra: int = 0
    deep_any_loss_cooldown_extra: int = 0
    deep_block_long_in_bear_regime: bool = False
    deep_block_short_in_bull_regime: bool = False


def _sl_ratio(res: dict[str, Any]) -> float:
    exits = dict(res.get("exits", {}))
    return float(sum(int(v) for k, v in exits.items() if "stop_loss" in str(k)) / max(int(res.get("trades", 0)), 1))


def _score(res: dict[str, Any]) -> float:
    if int(res.get("trades", 0)) < 20:
        return -1e9 + float(res.get("pnl", 0.0))
    return float(res["pnl"]) + 2.0 * float(res["mdd"]) + 40.0 * float(res["wr"]) - 0.03 * float(res["trades"])


def _state24_dominant_regime(row: pd.Series) -> str:
    probs = {
        "bull": float(row.get("clean_regime4_state24_sticky090_v2_bull_prob", 0.0) or 0.0),
        "bear": float(row.get("clean_regime4_state24_sticky090_v2_bear_prob", 0.0) or 0.0),
        "chop": float(row.get("clean_regime4_state24_sticky090_v2_chop_prob", 0.0) or 0.0),
        "whipsaw": float(row.get("clean_regime4_state24_sticky090_v2_whipsaw_prob", 0.0) or 0.0),
    }
    if max(abs(v) for v in probs.values()) <= 1e-12:
        raise RuntimeError("missing clean_regime4_state24_sticky090_v2 regime probabilities")
    return max(probs, key=probs.get)


def _backtest_variant(
    *,
    df: pd.DataFrame,
    q: np.ndarray,
    dec: pd.DataFrame,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    variant: Variant,
    cost_mult: int,
    record: bool = False,
    deep_gate: Any | None = None,
) -> dict[str, Any]:
    close = _close(df)
    fee_base = float(stack["fee"]) * float(cost_mult)
    slip_base = float(stack["slip"]) * float(cost_mult)
    bundle = stack["parent"]
    jackpot_model = stack["runner"]
    add_cfg = stack["add_cfg"]
    base_overlay = precision._overlay(stack["overlay"], cfg)
    overlay = replace(
        base_overlay,
        name=f"{base_overlay.name}_{variant.name}",
        notional=float(base_overlay.notional * variant.deep_notional_mult),
        edge_th=float(base_overlay.edge_th * variant.deep_edge_mult),
        margin_th=float(base_overlay.margin_th * variant.deep_margin_mult),
    )
    limit_cfg = precision._default_limit_cfg()
    guard_cfg = precision._guard(cfg)
    decisions = precision._apply_decision_mods(dec, cfg).reset_index(drop=True)

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
        raw = (px * (1.0 - slip_base) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (
            entry_price - px * (1.0 + slip_base)
        ) / max(entry_price, 1e-12)
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
                    effective_tp = float(np.clip(overlay.base_tp * util_gain, overlay.base_tp * 0.8, overlay.tp_cap))
                if overlay.sl_vol_mult > 0.0:
                    effective_sl = float(np.clip(entry_vol_anchor * overlay.sl_vol_mult, overlay.base_sl * 0.6, overlay.sl_cap))
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
                soft_counter = soft_counter + 1 if soft_hit else 0
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
                    filled, add_px, add_fee, _, route = _try_immediate_limit(df, i, pos, limit_cfg, entry=True, fee=fee_base, slip=slip_base)
                    if filled and delta > 0.0:
                        new_notional = notional + delta
                        entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                        cash -= cash * add_fee * delta
                        notional = new_notional
                        actions["v21_add_on"] = actions.get("v21_add_on", 0) + 1
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
                was_deep = owner == "deep_alpha"
                was_deep_stop = was_deep and "stop_loss" in reason
                was_deep_loss = was_deep and cash <= entry_equity
                pos = 0
                owner = ""
                extra_cd = 0
                if "hard_stop_loss" in reason:
                    extra_cd = int(guard_cfg.cooldown_after_hard_stop)
                elif "soft_stop_loss" in reason:
                    extra_cd = int(guard_cfg.cooldown_after_soft_stop)
                elif "giveback_exit" in reason:
                    extra_cd = int(guard_cfg.cooldown_after_giveback)
                deep_extra = 0
                if was_deep_stop:
                    deep_extra = max(deep_extra, int(variant.deep_stop_cooldown_extra))
                if was_deep_loss:
                    deep_extra = max(deep_extra, int(variant.deep_any_loss_cooldown_extra))
                cooldown = max(int(next_cooldown), int(extra_cd))
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(overlay.cooldown), int(extra_cd), int(deep_extra))
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

        dec_row = decisions.iloc[i]
        if int(dec_row.action) != ACTION_CASH and int(dec_row.side) != 0:
            if int(guard_cfg.same_side_entry_gap) > 0 and int(dec_row.side) == int(last_entry_side) and (i - int(last_entry_idx)) <= int(guard_cfg.same_side_entry_gap):
                actions["parent_entry_same_side_gap_block"] = actions.get("parent_entry_same_side_gap_block", 0) + 1
                continue
            if float(dec_row.quality_score) < float(guard_cfg.entry_quality_min):
                actions["parent_entry_quality_block"] = actions.get("parent_entry_quality_block", 0) + 1
                continue
            if float(dec_row.confidence) < float(guard_cfg.entry_conf_min):
                actions["parent_entry_conf_block"] = actions.get("parent_entry_conf_block", 0) + 1
                continue
            filled, px, entry_fee, _, route = _try_immediate_limit(df, i, int(dec_row.side), limit_cfg, entry=True, fee=fee_base, slip=slip_base)
            if not filled:
                actions["parent_entry_limit_miss"] = actions.get("parent_entry_limit_miss", 0) + 1
                route_counts[route] = route_counts.get(route, 0) + 1
                continue
            pos = int(dec_row.side)
            owner = "v21_2"
            entry_price = px
            entry_equity = cash
            entry_idx = i
            parent_notional = min(float(dec_row.notional_exposure), add_cfg.max_entry_notional)
            notional = parent_notional
            take_profit = float(dec_row.take_profit)
            stop_loss = float(dec_row.stop_loss)
            max_hold = int(dec_row.max_hold_bars)
            next_cooldown = int(dec_row.cooldown_bars)
            cash -= cash * entry_fee * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += float(dec_row.leverage)
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
                    "leverage_like": float(dec_row.leverage),
                    "entry_route": str(route),
                }
            actions["v21_entry"] = actions.get("v21_entry", 0) + 1
            route_counts[route] = route_counts.get(route, 0) + 1
            continue

        if deep_cooldown <= 0 and i >= 60:
            ql, qs = float(q[i, 0]), float(q[i, 1])
            side = 1 if ql > qs else -1
            if variant.deep_side == "none":
                actions["deep_entry_side_block"] = actions.get("deep_entry_side_block", 0) + 1
                continue
            if variant.deep_side == "short_only" and side > 0:
                actions["deep_entry_long_block"] = actions.get("deep_entry_long_block", 0) + 1
                continue
            if variant.deep_block_long_in_bear_regime and side > 0 and _state24_dominant_regime(df.iloc[i]) == "bear":
                actions["deep_entry_bear_long_veto"] = actions.get("deep_entry_bear_long_veto", 0) + 1
                continue
            if variant.deep_block_short_in_bull_regime and side < 0 and _state24_dominant_regime(df.iloc[i]) == "bull":
                actions["deep_entry_bull_short_veto"] = actions.get("deep_entry_bull_short_veto", 0) + 1
                continue
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if deep_gate is not None:
                allowed, reason = deep_gate(i, side, ql, qs, df.iloc[i])
                if not bool(allowed):
                    key = f"deep_entry_{str(reason or 'meta_veto')}_block"
                    actions[key] = actions.get(key, 0) + 1
                    continue
            if int(guard_cfg.same_side_entry_gap) > 0 and int(side) == int(last_entry_side) and (i - int(last_entry_idx)) <= int(guard_cfg.same_side_entry_gap):
                actions["deep_entry_same_side_gap_block"] = actions.get("deep_entry_same_side_gap_block", 0) + 1
                continue
            side_edge_th = float(overlay.edge_th * (variant.deep_long_edge_mult if side > 0 else variant.deep_short_edge_mult))
            side_margin_th = float(overlay.margin_th * (variant.deep_long_margin_mult if side > 0 else variant.deep_short_margin_mult))
            if edge >= side_edge_th and margin >= side_margin_th:
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
                entry_vol_anchor = float(v31._vol_anchor(df.iloc[i]) * notional)
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
    res = {
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
        res["trade_records"] = records
    return res


def _ledger_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"rows": 0}
    df = pd.DataFrame(records)
    ret = pd.to_numeric(df["trade_return"], errors="coerce").fillna(0.0)
    by_owner = {}
    for owner, g in df.groupby("owner"):
        gr = pd.to_numeric(g["trade_return"], errors="coerce").fillna(0.0)
        by_owner[str(owner)] = {"trades": int(len(g)), "sum": float(gr.sum()), "wr_raw": float((gr > 0).mean())}
    return {
        "rows": int(len(df)),
        "raw_sum": float(ret.sum()),
        "raw_mean": float(ret.mean()),
        "raw_wr": float((ret > 0).mean()),
        "by_owner": by_owner,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decontam._assert_clean_frame(decontam.TRAIN_CSV, name="train")
    decontam._assert_clean_frame(decontam.EVAL_CSV, name="eval")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "primary_parent.pkl", name="primary")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "fallback_alpha43_no_legacy_parent.pkl", name="fallback")
    decontam._patch_runtime_sources()

    cfg = precision._cfg_from_results()
    stack = precision._load_stack()
    val_df, eval_df = precision._load_frames()
    sources = precision._decision_sources(val_df, eval_df, stack["parent"])
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    val_dec = sources[str(cfg["source"])][0]
    eval_dec = sources[str(cfg["source"])][1]

    variants = [
        Variant("baseline"),
        Variant("deep_short_only", deep_side="short_only"),
        Variant("deep_disabled", deep_side="none"),
        Variant("deep_notional_075", deep_notional_mult=0.75),
        Variant("deep_notional_050", deep_notional_mult=0.50),
        Variant("deep_short_only_notional_075", deep_side="short_only", deep_notional_mult=0.75),
        Variant("deep_short_only_notional_050", deep_side="short_only", deep_notional_mult=0.50),
        Variant("deep_threshold_110", deep_edge_mult=1.10, deep_margin_mult=1.10),
        Variant("deep_threshold_125", deep_edge_mult=1.25, deep_margin_mult=1.25),
        Variant("deep_stop_cd06", deep_stop_cooldown_extra=6),
        Variant("deep_stop_cd12", deep_stop_cooldown_extra=12),
        Variant("deep_stop_cd18", deep_stop_cooldown_extra=18),
        Variant("deep_stop_cd18_bear_long_veto", deep_stop_cooldown_extra=18, deep_block_long_in_bear_regime=True),
        Variant(
            "deep_stop_cd18_dual_regime_veto",
            deep_stop_cooldown_extra=18,
            deep_block_long_in_bear_regime=True,
            deep_block_short_in_bull_regime=True,
        ),
        Variant(
            "deep_stop_cd18_side_specialist_mild",
            deep_stop_cooldown_extra=18,
            deep_block_long_in_bear_regime=True,
            deep_long_edge_mult=1.05,
            deep_long_margin_mult=1.05,
            deep_short_edge_mult=1.00,
            deep_short_margin_mult=0.98,
        ),
        Variant(
            "deep_stop_cd18_side_specialist",
            deep_stop_cooldown_extra=18,
            deep_block_long_in_bear_regime=True,
            deep_block_short_in_bull_regime=True,
            deep_long_edge_mult=1.15,
            deep_long_margin_mult=1.20,
            deep_short_edge_mult=1.00,
            deep_short_margin_mult=0.95,
        ),
        Variant(
            "deep_stop_cd18_side_specialist_no_short_veto",
            deep_stop_cooldown_extra=18,
            deep_block_long_in_bear_regime=True,
            deep_long_edge_mult=1.15,
            deep_long_margin_mult=1.20,
            deep_short_edge_mult=1.00,
            deep_short_margin_mult=0.95,
        ),
        Variant(
            "deep_stop_cd18_long_defensive",
            deep_stop_cooldown_extra=18,
            deep_block_long_in_bear_regime=True,
            deep_long_edge_mult=1.25,
            deep_long_margin_mult=1.25,
        ),
        Variant("deep_stop_cd24", deep_stop_cooldown_extra=24),
        Variant("deep_stop_cd30", deep_stop_cooldown_extra=30),
        Variant("deep_stop_cd36", deep_stop_cooldown_extra=36),
        Variant("deep_stop_cd48", deep_stop_cooldown_extra=48),
        Variant("deep_loss_cd24", deep_any_loss_cooldown_extra=24),
        Variant("deep_stop_cd24_notional_075", deep_notional_mult=0.75, deep_stop_cooldown_extra=24),
        Variant("deep_stop_cd24_notional_050", deep_notional_mult=0.50, deep_stop_cooldown_extra=24),
        Variant("deep_short_only_cd24", deep_side="short_only", deep_stop_cooldown_extra=24),
        Variant("deep_short_only_notional_050_cd24", deep_side="short_only", deep_notional_mult=0.50, deep_stop_cooldown_extra=24),
    ]

    rows: list[dict[str, Any]] = []
    best_records: list[dict[str, Any]] = []
    best_name = ""
    best_oos_score = -1e18
    for variant in variants:
        val = _backtest_variant(df=val_df, q=val_q, dec=val_dec, stack=stack, cfg=cfg, variant=variant, cost_mult=3, record=False)
        oos = _backtest_variant(df=eval_df, q=eval_q, dec=eval_dec, stack=stack, cfg=cfg, variant=variant, cost_mult=3, record=True)
        records = list(oos.pop("trade_records", []))
        row = {
            **asdict(variant),
            "val_pnl": float(val["pnl"]),
            "val_mdd": float(val["mdd"]),
            "val_wr": float(val["wr"]),
            "val_trades": int(val["trades"]),
            "val_deep_entries": int(val.get("deep_entries", 0)),
            "val_sl_ratio": float(_sl_ratio(val)),
            "val_score": float(_score(val)),
            "oos_pnl": float(oos["pnl"]),
            "oos_mdd": float(oos["mdd"]),
            "oos_wr": float(oos["wr"]),
            "oos_trades": int(oos["trades"]),
            "oos_deep_entries": int(oos.get("deep_entries", 0)),
            "oos_long_entries": int(oos.get("long_entries", 0)),
            "oos_short_entries": int(oos.get("short_entries", 0)),
            "oos_sl_ratio": float(_sl_ratio(oos)),
            "oos_score": float(_score(oos)),
            "oos_exits": json.dumps(oos.get("exits", {}), ensure_ascii=False, sort_keys=True),
            "oos_ledger_stats": json.dumps(_ledger_stats(records), ensure_ascii=False, sort_keys=True),
        }
        rows.append(row)
        if row["oos_score"] > best_oos_score:
            best_oos_score = float(row["oos_score"])
            best_name = variant.name
            best_records = records

    grid = pd.DataFrame(rows).sort_values(["oos_pnl", "val_score"], ascending=[False, False])
    grid.to_csv(GRID_OUT, index=False)
    best_ledger_path = OUT_DIR / f"{best_name}_oos_cost3_ledger.csv"
    pd.DataFrame(best_records).to_csv(best_ledger_path, index=False)
    summary = {
        "model": "alpha7_submodel_01965_decontam_v2_tp_20260528",
        "scope": "deep_alpha control A/B only; parent/v21_2, decision source, costs, limit contract frozen",
        "grid": str(GRID_OUT),
        "best_by_oos_score": best_name,
        "best_oos_ledger": str(best_ledger_path),
        "rows": grid.to_dict(orient="records"),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "best": best_name}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
