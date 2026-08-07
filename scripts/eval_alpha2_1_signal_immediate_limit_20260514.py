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
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _days, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner  # noqa: E402


MODEL_ID = "alpha2_1_signal_immediate_limit_20260514"
TEACHER_MODEL = ROOT / "data/ensemble/supervised/alpha1_l2_teacher_deep_parent_20260514/teacher_deep_parent_l2_replay.pt"
ALPHA2_AUDIT = ROOT / "data/ensemble/reports/alpha2_teacher_l2_runtime_sweep_20260514_audit.json"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha2_1_signal_immediate_limit_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha2_1_signal_immediate_limit_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha2_1_signal_immediate_limit_20260514_grid.csv"


@dataclass(frozen=True)
class ImmediateLimitConfig:
    name: str
    anchor: str
    entry_offset_bps: float
    exit_offset_bps: float
    penetration_bps: float
    maker_fee_mult: float
    entry_miss: str = "skip"
    exit_miss: str = "market_fallback"


def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(col, default))
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _limit_price(df: pd.DataFrame, signal_i: int, side: int, *, entry: bool, offset_bps: float, anchor: str) -> float:
    signal_i = int(np.clip(signal_i, 0, len(df) - 1))
    if str(anchor) == "next_open":
        anchor_i = int(np.clip(signal_i + 1, 0, len(df) - 1))
        anchor_px = _safe(df.iloc[anchor_i], "open", _safe(df.iloc[anchor_i], "close", 0.0))
    else:
        anchor_px = _safe(df.iloc[signal_i], "close", _safe(df.iloc[signal_i], "open", 0.0))
    if anchor_px <= 0.0:
        return 0.0
    is_buy = (side > 0 and entry) or (side < 0 and not entry)
    if is_buy:
        return float(anchor_px * (1.0 - float(offset_bps) / 10000.0))
    return float(anchor_px * (1.0 + float(offset_bps) / 10000.0))


def _limit_touched(df: pd.DataFrame, fill_i: int, price: float, side: int, *, entry: bool, penetration_bps: float) -> bool:
    fill_i = int(np.clip(fill_i, 0, len(df) - 1))
    row = df.iloc[fill_i]
    high = _safe(row, "high", _safe(row, "open", 0.0))
    low = _safe(row, "low", _safe(row, "open", 0.0))
    is_buy = (side > 0 and entry) or (side < 0 and not entry)
    pen = float(penetration_bps) / 10000.0
    if is_buy:
        return bool(low <= price * (1.0 - pen))
    return bool(high >= price * (1.0 + pen))


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


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    return alpha2._score(c1, c2, c3)


def backtest_signal_limit(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    deep_q: np.ndarray,
    decisions: pd.DataFrame,
    overlay: v31.OverlayConfig,
    limit_cfg: ImmediateLimitConfig,
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
        if int(dec.action) != ACTION_CASH and int(dec.side) != 0:
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
            actions["v21_entry"] = actions.get("v21_entry", 0) + 1
            route_counts[route] = route_counts.get(route, 0) + 1
            continue

        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
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
    }


def _metrics_signal_limit(df, parent, jackpot_model, add_cfg, q, decisions, overlay, cfg, *, fee, slip) -> dict[str, Any]:
    return {
        f"cost{mult}": backtest_signal_limit(
            df,
            parent,
            jackpot_model,
            add_cfg,
            q,
            decisions,
            overlay,
            cfg,
            fee=fee,
            slip=slip,
            cost_mult=float(mult),
        )
        for mult in (1, 2, 3)
    }


def _configs() -> list[ImmediateLimitConfig]:
    return [
        ImmediateLimitConfig("signal_close_limit_touch0_fee20", "signal_close", 0.0, 0.0, 0.0, 0.20),
        ImmediateLimitConfig("signal_close_limit_penetrate05_fee20", "signal_close", 0.0, 0.0, 0.5, 0.20),
        ImmediateLimitConfig("signal_close_limit_offset1_penetrate05_fee20", "signal_close", 1.0, 1.0, 0.5, 0.20),
        ImmediateLimitConfig("signal_close_limit_offset2_penetrate05_fee20", "signal_close", 2.0, 2.0, 0.5, 0.20),
        ImmediateLimitConfig("signal_close_limit_touch0_fee35", "signal_close", 0.0, 0.0, 0.0, 0.35),
        ImmediateLimitConfig("next_open_limit_touch0_fee20", "next_open", 0.0, 0.0, 0.0, 0.20),
        ImmediateLimitConfig("next_open_limit_penetrate05_fee20", "next_open", 0.0, 0.0, 0.5, 0.20),
        ImmediateLimitConfig("next_open_limit_offset1_penetrate05_fee20", "next_open", 1.0, 1.0, 0.5, 0.20),
        ImmediateLimitConfig("next_open_limit_offset2_penetrate05_fee20", "next_open", 2.0, 2.0, 0.5, 0.20),
        ImmediateLimitConfig("next_open_limit_touch0_fee35", "next_open", 0.0, 0.0, 0.0, 0.35),
        ImmediateLimitConfig("signal_close_limit_offset2_entry_fallback_fee20", "signal_close", 2.0, 2.0, 0.5, 0.20, entry_miss="market_fallback"),
        ImmediateLimitConfig("next_open_limit_offset2_entry_fallback_fee20", "next_open", 2.0, 2.0, 0.5, 0.20, entry_miss="market_fallback"),
        ImmediateLimitConfig("next_open_limit_touch0_entry_fallback_fee20", "next_open", 0.0, 0.0, 0.0, 0.20, entry_miss="market_fallback"),
    ]


def main() -> int:
    print(f"[{MODEL_ID}] loading Alpha2.1 fixed stack", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    audit = json.loads(ALPHA2_AUDIT.read_text(encoding="utf-8"))
    runtime = dict(audit.get("selected_runtime", {}) or {})
    rt = alpha2.Alpha2Runtime(
        name=str(runtime.get("name", "noflip_c0.56_parent_scale1.10")),
        confidence=float(runtime.get("confidence", 0.56)),
        parent_notional_scale=float(runtime.get("parent_notional_scale", 1.10)),
        max_notional=float(runtime.get("max_notional", 2.75)),
    )
    selected_variant = next(v for v in l2._variants() if v.name == "alpha1_l2_conservative_fee20")
    overlay = selected_variant.overlay
    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    teacher_payload = torch.load(TEACHER_MODEL, map_location="cpu", weights_only=False)
    teacher_model = alpha2._load_teacher_model(teacher_payload)
    feature_cols = list(teacher_payload["feature_cols"])
    norm = dict(dict(teacher_payload["train_meta"])["norm"])
    buckets = tuple(float(x) for x in teacher_payload["buckets"])
    fee = float(dict(parent["config"])["fee"])
    slip = float(dict(parent["config"])["slip"])
    train_all = _read(v31.DEFAULT_TRAIN)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    eval_df = _read(v31.DEFAULT_EVAL)
    contract_features = list(teacher_payload["feature_cols"])

    print(f"[{MODEL_ID}] rebuilding decisions and V27 q", flush=True)
    val_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    val_features = prepare_features(val, side_hint=0, close=_close(val), feature_cols=contract_features)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=contract_features)
    val_pred = teacher._predict_deep(teacher_model, val_features, feature_cols, norm)
    eval_pred = teacher._predict_deep(teacher_model, eval_features, feature_cols, norm)
    val_dec = alpha2._decisions(val_dec, val_pred, buckets, rt)
    eval_dec = alpha2._decisions(eval_dec, eval_pred, buckets, rt)
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    rows: list[dict[str, Any]] = []
    best_cfg: ImmediateLimitConfig | None = None
    best_score = -1e18
    print(f"[{MODEL_ID}] selecting immediate-limit config on 2025Q4", flush=True)
    for cfg in _configs():
        m = _metrics_signal_limit(val, parent, jackpot_model, add_cfg, val_q, val_dec, overlay, cfg, fee=fee, slip=slip)
        score = _score(m["cost1"], m["cost2"], m["cost3"])
        rows.append(
            {
                **asdict(cfg),
                "selection_score": score,
                "val_cost1_pnl": m["cost1"]["pnl"],
                "val_cost1_mdd": m["cost1"]["mdd"],
                "val_cost1_trades": m["cost1"]["trades"],
                "val_cost2_pnl": m["cost2"]["pnl"],
                "val_cost3_pnl": m["cost3"]["pnl"],
            }
        )
        print(f"[{MODEL_ID}] {cfg.name} val c1={m['cost1']['pnl']:.2f} mdd={m['cost1']['mdd']:.2f} c2={m['cost2']['pnl']:.2f} c3={m['cost3']['pnl']:.2f}", flush=True)
        if score > best_score:
            best_score = score
            best_cfg = cfg
    assert best_cfg is not None
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)

    print(f"[{MODEL_ID}] fixed 2026 OOS", flush=True)
    taker = alpha2._metrics(eval_df, parent, jackpot_model, add_cfg, eval_q, eval_dec, l2._variants()[0], fee=fee, slip=slip)
    old_l2 = alpha2._metrics(eval_df, parent, jackpot_model, add_cfg, eval_q, eval_dec, selected_variant, fee=fee, slip=slip)
    new_limit = _metrics_signal_limit(eval_df, parent, jackpot_model, add_cfg, eval_q, eval_dec, overlay, best_cfg, fee=fee, slip=slip)
    experiments = [
        {"name": "alpha2_1_next_open_taker", "metrics": taker, "score": _score(taker["cost1"], taker["cost2"], taker["cost3"])},
        {"name": "alpha2_1_old_l2_replay_fee20", "metrics": old_l2, "score": _score(old_l2["cost1"], old_l2["cost2"], old_l2["cost3"])},
        {"name": f"alpha2_1_signal_immediate_limit::{best_cfg.name}", "config": asdict(best_cfg), "metrics": new_limit, "score": _score(new_limit["cost1"], new_limit["cost2"], new_limit["cost3"])},
    ]
    for e in experiments:
        m = e["metrics"]
        print(f"[{MODEL_ID}] {e['name']} c1={m['cost1']['pnl']:.2f} mdd={m['cost1']['mdd']:.2f} c2={m['cost2']['pnl']:.2f} c3={m['cost3']['pnl']:.2f}", flush=True)

    audit_out = {
        "status": "pass",
        "verdict": "candidate_retest_required_with_real_l2_ticks",
        "blocking": [],
        "warnings": [
            "signal_immediate_limit_uses_5m_high_low_touch_proxy_not_queue_fill",
            "market_fallback_after_limit_miss_uses_same_next_bar_close_not_next_bar_open",
            "live_post_only_reject_partial_fill_and_queue_position_not_modeled",
        ],
        "selection_uses_2026": False,
        "selected_config": asdict(best_cfg),
        "fallback_contract": "signal i -> maker touch check on i+1 high/low -> if miss, market fallback at i+1 close +/- slippage",
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha2.1 same decisions, but execution is changed from next-open taker/synthetic L2 to signal-immediate post-only limit. Next bar high/low touch is required for maker fill; missed maker entries/exits fall back at that same next bar close with taker fee/slippage.",
        "experiments": experiments,
        "selection_grid": str(GRID_OUT),
        "audit": audit_out,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit_out, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT), "selected": best_cfg.name}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
