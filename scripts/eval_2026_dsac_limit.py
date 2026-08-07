#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in [_ROOT_DIR, os.path.join(_ROOT_DIR, "ensemble")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.train_rl_dsac_agent import DSAC_STATE_DIM, DSACRouter, GaussianActor

CSV_PATH = "data/splits/year_oos/rl_meta_2026.csv"
CKPT = "data/ensemble/ckpt/best_dsac_agents.pth"
OUT_JSON = "data/ensemble/reports/eval_2026_dsac_limit.json"

TAKER_FEE = 0.0005
MAKER_FEE = 0.0002
TAKER_SLIP = 0.0002
ANNUAL_FACTOR_5M = math.sqrt(365 * 24 * 12)


@dataclass
class LimitConfig:
    name: str
    entry_scale: float
    volatility_mult: float
    liquidity_mult: float
    model_anchor: float
    exit_buffer_bps: float
    entry_ttl: int
    exit_ttl: int
    fallback_conf: float
    fallback_edge: float


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return x


def _clip(v: float, lo: float, hi: float) -> float:
    return float(np.clip(v, lo, hi))


def _sharpe(eq_curve: list[float]) -> float:
    eq = np.array(eq_curve, dtype=np.float64)
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    if len(rets) < 3 or np.std(rets) < 1e-12:
        return 0.0
    return float(np.mean(rets) / np.std(rets) * ANNUAL_FACTOR_5M)


def _mdd(eq_curve: list[float]) -> float:
    eq = np.array(eq_curve, dtype=np.float64)
    run_max = np.maximum.accumulate(eq)
    dd = eq / np.maximum(run_max, 1e-12) - 1.0
    return float(np.min(dd)) * 100.0


def _load_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _build_actor(device: str) -> GaussianActor:
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    actor = GaussianActor(state_dim=int(ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM)).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def _prob_pair(row: pd.Series) -> tuple[float, float]:
    dn = _safe_float(row.get("m7_trend_xgb_dn", row.get("prob_dn", 0.5)), 0.5)
    up = _safe_float(row.get("m7_trend_xgb_up", row.get("prob_up", 0.5)), 0.5)
    s = dn + up
    if s <= 1e-12:
        return (0.5, 0.5)
    return (dn / s, up / s)


def _flow_alignment(row: pd.Series, side: str) -> float:
    sign = 1.0 if side == "LONG" else -1.0
    smf = sign * np.tanh(_safe_float(row.get("smart_money_flow", 0.0)) / 0.0035)
    ofi = sign * np.tanh(_safe_float(row.get("ofi_acceleration", 0.0)) / 0.18)
    imbalance = sign * np.tanh(_safe_float(row.get("cvp_volume_imbalance", 0.0)) / 0.55)
    funding = -sign * np.tanh(_safe_float(row.get("funding_price_divergence", 0.0)) / 0.5)
    intensity = np.tanh((_safe_float(row.get("trade_intensity", 2.0)) - 1.8) / 0.8)
    return float(0.35 * smf + 0.25 * ofi + 0.20 * imbalance + 0.10 * funding + 0.10 * intensity)


def _compute_entry_plan(row: pd.Series, side: str, current_price: float, kelly: float, cfg: LimitConfig) -> dict:
    conf = _clip(_safe_float(row.get("m7_confidence", 0.5), 0.5), 0.0, 1.0)
    qwidth = max(_safe_float(row.get("m7_qwidth", 0.01), 0.01), 1e-4)
    amihud = max(_safe_float(row.get("amihud_illiquidity_z", 0.0), 0.0), 0.0)
    p_dn, p_up = _prob_pair(row)
    signal_edge = p_up - p_dn if side == "LONG" else p_dn - p_up
    flow = _flow_alignment(row, side)

    if side == "LONG":
        model_offset = abs(_safe_float(row.get("m7_entry_long_offset", -0.0016), -0.0016))
        model_price = _safe_float(row.get("m7_entry_long_price", 0.0), 0.0)
        cluster_bias = _clip((_safe_float(row.get("cvp_cluster_position", 0.5), 0.5) - 0.35) / 0.65, -0.5, 1.0)
        poc_bias = _clip(_safe_float(row.get("cvp_poc_dist", 0.0), 0.0), -1.0, 1.0)
    else:
        model_offset = abs(_safe_float(row.get("m7_entry_short_offset", 0.0016), 0.0016))
        model_price = _safe_float(row.get("m7_entry_short_price", 0.0), 0.0)
        cluster_bias = _clip((0.65 - _safe_float(row.get("cvp_cluster_position", 0.5), 0.5)) / 0.65, -0.5, 1.0)
        poc_bias = _clip(-_safe_float(row.get("cvp_poc_dist", 0.0), 0.0), -1.0, 1.0)

    aggression = _clip(0.55 * conf + 0.20 * _clip(signal_edge * 2.2, 0.0, 1.0) + 0.15 * _clip(flow, 0.0, 1.0) + 0.10 * _clip(kelly, 0.0, 1.0), 0.0, 1.0)
    offset = model_offset * cfg.entry_scale
    offset += qwidth * cfg.volatility_mult * 0.22
    offset += max(amihud - 0.25, 0.0) * cfg.liquidity_mult * 0.00035
    offset += max(cluster_bias, 0.0) * 0.00045
    offset += max(poc_bias, 0.0) * 0.00035
    offset -= aggression * 0.00085
    offset = _clip(offset, 0.00025, 0.00650)

    if side == "LONG":
        fallback_price = current_price * (1.0 - offset)
        anchor_price = model_price if model_price > 0.0 else fallback_price
        limit_price = cfg.model_anchor * anchor_price + (1.0 - cfg.model_anchor) * fallback_price
    else:
        fallback_price = current_price * (1.0 + offset)
        anchor_price = model_price if model_price > 0.0 else fallback_price
        limit_price = cfg.model_anchor * anchor_price + (1.0 - cfg.model_anchor) * fallback_price

    ttl = int(max(1, cfg.entry_ttl + (1 if conf < 0.45 else 0) + (1 if qwidth > 0.016 else 0)))
    allow_fallback = bool(conf >= cfg.fallback_conf and signal_edge >= cfg.fallback_edge and flow > -0.05)
    return {
        "limit_price": float(limit_price),
        "ttl": ttl,
        "allow_fallback": allow_fallback,
        "confidence": conf,
        "signal_edge": float(signal_edge),
        "offset_pct": float(offset),
    }


def _compute_exit_plan(row: pd.Series, side: str, current_price: float, cfg: LimitConfig) -> dict:
    buffer_pct = cfg.exit_buffer_bps / 10000.0
    tp_price = _safe_float(row.get("m7_tp_price", 0.0), 0.0)
    qwidth = max(_safe_float(row.get("m7_qwidth", 0.01), 0.01), 1e-4)
    conf = _clip(_safe_float(row.get("m7_confidence", 0.5), 0.5), 0.0, 1.0)
    dynamic_buffer = buffer_pct + qwidth * 0.08 - conf * 0.00018
    dynamic_buffer = _clip(dynamic_buffer, 0.00015, 0.00400)

    if side == "LONG":
        limit_price = current_price * (1.0 + dynamic_buffer)
        if tp_price > current_price:
            limit_price = min(limit_price, tp_price)
    else:
        limit_price = current_price * (1.0 - dynamic_buffer)
        if 0.0 < tp_price < current_price:
            limit_price = max(limit_price, tp_price)
    return {"limit_price": float(limit_price), "ttl": int(max(1, cfg.exit_ttl))}


def _realized_return(side: str, entry_price: float, exit_price: float, lev: float, entry_fee: float, exit_fee: float) -> float:
    if side == "LONG":
        gross = (exit_price - entry_price) / max(entry_price, 1e-12)
    else:
        gross = (entry_price - exit_price) / max(entry_price, 1e-12)
    return float(gross * lev - (entry_fee + exit_fee) * lev)


def simulate_market(df: pd.DataFrame, router: DSACRouter) -> dict:
    numeric_cols = [c for c in df.columns if c != "timestamp"]
    values = df[numeric_cols].to_numpy(dtype=np.float64)
    open_np = df["open"].to_numpy(dtype=np.float64)
    close_np = df["close"].to_numpy(dtype=np.float64)

    balance = 1.0
    pos: str | None = None
    entry_price = 0.0
    cur_lev = 0.0
    hold_count = 0
    trades = wins = 0
    eq_curve = [1.0]

    def _unrealized(current_price: float) -> float:
        if pos is None or entry_price <= 0.0 or cur_lev <= 0.0:
            return 0.0
        gross = (current_price * (1.0 - TAKER_SLIP) - entry_price) / entry_price if pos == "LONG" else (entry_price - current_price * (1.0 + TAKER_SLIP)) / entry_price
        return float(gross * cur_lev - (2.0 * TAKER_FEE * cur_lev))

    for i in range(len(df) - 1):
        cp = float(close_np[i])
        next_open = float(open_np[i + 1])
        next_close = float(close_np[i + 1])
        if pos is not None:
            hold_count += 1

        pos_dict = {
            "type": pos,
            "entry_price": float(entry_price),
            "unrealized": float(_unrealized(cp)),
            "mdd": 0.0,
            "hold_norm": float(min(hold_count / 96.0, 1.0)),
            "margin_usage": float(cur_lev if pos else 0.0),
            "hold_count": float(hold_count),
        }
        row = values[i]
        features = {k: float(v) for k, v in zip(numeric_cols, row)}
        action_int, lev, _ = router.decide(features, pos_dict)
        lev = float(np.clip(lev, 0.0, 1.0))

        if pos is None:
            if action_int == 1 and lev > 0.0:
                pos = "LONG"
                entry_price = next_open * (1.0 + TAKER_SLIP)
                cur_lev = lev
                hold_count = 0
                balance -= balance * TAKER_FEE * cur_lev
            elif action_int == 2 and lev > 0.0:
                pos = "SHORT"
                entry_price = next_open * (1.0 - TAKER_SLIP)
                cur_lev = lev
                hold_count = 0
                balance -= balance * TAKER_FEE * cur_lev
        else:
            should_close = action_int == 0 or (action_int == 1 and pos == "SHORT") or (action_int == 2 and pos == "LONG")
            if should_close:
                exit_price = next_open * (1.0 - TAKER_SLIP) if pos == "LONG" else next_open * (1.0 + TAKER_SLIP)
                realized = _realized_return(pos, entry_price, exit_price, cur_lev, TAKER_FEE, TAKER_FEE)
                balance *= 1.0 + realized
                trades += 1
                wins += int(realized > 0.0)
                pos = None
                entry_price = 0.0
                cur_lev = 0.0
                hold_count = 0
            elif abs(lev - cur_lev) > 0.05:
                balance -= balance * TAKER_FEE * abs(lev - cur_lev)
                cur_lev = lev

        eq_curve.append(max(balance * (1.0 + _unrealized(next_close)) if pos else balance, 1e-8))

    if pos is not None and entry_price > 0.0:
        exit_price = float(close_np[-1]) * (1.0 - TAKER_SLIP) if pos == "LONG" else float(close_np[-1]) * (1.0 + TAKER_SLIP)
        realized = _realized_return(pos, entry_price, exit_price, cur_lev, TAKER_FEE, TAKER_FEE)
        balance *= 1.0 + realized
        trades += 1
        wins += int(realized > 0.0)

    return {
        "method": "market_closed_loop",
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "wr_pct": round((100.0 * wins / trades) if trades else 0.0, 2),
        "trades": trades,
        "sharpe": round(_sharpe(eq_curve), 4),
        "mdd_pct": round(_mdd(eq_curve), 4),
    }


def simulate_limit(df: pd.DataFrame, router: DSACRouter, cfg: LimitConfig) -> dict:
    numeric_cols = [c for c in df.columns if c != "timestamp"]
    values = df[numeric_cols].to_numpy(dtype=np.float64)
    open_np = df["open"].to_numpy(dtype=np.float64)
    high_np = df["high"].to_numpy(dtype=np.float64)
    low_np = df["low"].to_numpy(dtype=np.float64)
    close_np = df["close"].to_numpy(dtype=np.float64)

    balance = 1.0
    pos: str | None = None
    entry_price = 0.0
    entry_fee = 0.0
    cur_lev = 0.0
    hold_count = 0
    trades = wins = 0
    maker_entries = fallback_entries = missed_entries = market_exits = 0
    eq_curve = [1.0]
    pending: dict | None = None

    def _unrealized(current_price: float) -> float:
        if pos is None or entry_price <= 0.0 or cur_lev <= 0.0:
            return 0.0
        if pos == "LONG":
            gross = (current_price - entry_price) / entry_price
        else:
            gross = (entry_price - current_price) / entry_price
        return float(gross * cur_lev - ((entry_fee + MAKER_FEE) * cur_lev))

    for i in range(1, len(df) - 1):
        bar_open = float(open_np[i])
        bar_high = float(high_np[i])
        bar_low = float(low_np[i])
        bar_close = float(close_np[i])

        if pos is not None:
            hold_count += 1

        if pending is not None:
            should_fill = False
            should_fill = (pending["side"] == "LONG" and bar_low <= pending["price"]) or (
                pending["side"] == "SHORT" and bar_high >= pending["price"]
            )
            if should_fill:
                pos = pending["side"]
                entry_price = float(pending["price"])
                entry_fee = MAKER_FEE
                cur_lev = float(pending["lev"])
                hold_count = 0
                balance -= balance * MAKER_FEE * cur_lev
                maker_entries += 1
                pending = None
            elif i > pending["expire_idx"]:
                if pending.get("allow_fallback", False):
                    pos = pending["side"]
                    entry_price = bar_open * (1.0 + TAKER_SLIP) if pos == "LONG" else bar_open * (1.0 - TAKER_SLIP)
                    entry_fee = TAKER_FEE
                    cur_lev = float(pending["lev"])
                    hold_count = 0
                    balance -= balance * TAKER_FEE * cur_lev
                    fallback_entries += 1
                else:
                    missed_entries += 1
                pending = None

        if pending is None:
            cp = float(close_np[i - 1])
            pos_dict = {
                "type": pos,
                "entry_price": float(entry_price),
                "unrealized": float(_unrealized(cp)),
                "mdd": 0.0,
                "hold_norm": float(min(hold_count / 96.0, 1.0)),
                "margin_usage": float(cur_lev if pos else 0.0),
                "hold_count": float(hold_count),
            }
            features = {k: float(v) for k, v in zip(numeric_cols, values[i - 1])}
            action_int, lev, _ = router.decide(features, pos_dict)
            lev = float(np.clip(lev, 0.0, 1.0))
            row = df.iloc[i - 1]

            if pos is None:
                if action_int == 1 and lev > 0.0:
                    plan = _compute_entry_plan(row, "LONG", cp, lev, cfg)
                    pending = {
                        "kind": "entry",
                        "side": "LONG",
                        "price": float(plan["limit_price"]),
                        "expire_idx": i - 1 + int(plan["ttl"]),
                        "lev": lev,
                        "allow_fallback": bool(plan["allow_fallback"]),
                    }
                elif action_int == 2 and lev > 0.0:
                    plan = _compute_entry_plan(row, "SHORT", cp, lev, cfg)
                    pending = {
                        "kind": "entry",
                        "side": "SHORT",
                        "price": float(plan["limit_price"]),
                        "expire_idx": i - 1 + int(plan["ttl"]),
                        "lev": lev,
                        "allow_fallback": bool(plan["allow_fallback"]),
                    }
            else:
                should_close = action_int == 0 or (action_int == 1 and pos == "SHORT") or (action_int == 2 and pos == "LONG")
                if should_close:
                    exit_price = bar_open * (1.0 - TAKER_SLIP) if pos == "LONG" else bar_open * (1.0 + TAKER_SLIP)
                    realized = _realized_return(pos, entry_price, exit_price, cur_lev, entry_fee, TAKER_FEE)
                    balance *= 1.0 + realized
                    trades += 1
                    wins += int(realized > 0.0)
                    market_exits += 1
                    pos = None
                    entry_price = 0.0
                    entry_fee = 0.0
                    cur_lev = 0.0
                    hold_count = 0

        eq_curve.append(max(balance * (1.0 + _unrealized(bar_close)) if pos else balance, 1e-8))

    if pending is not None and pos is not None:
        last_close = float(close_np[-1])
        exit_price = last_close * (1.0 - TAKER_SLIP) if pos == "LONG" else last_close * (1.0 + TAKER_SLIP)
        realized = _realized_return(pos, entry_price, exit_price, cur_lev, entry_fee, TAKER_FEE)
        balance *= 1.0 + realized
        trades += 1
        wins += int(realized > 0.0)
        fallback_exits += 1
        pending = None
        pos = None
    elif pos is not None:
        last_close = float(close_np[-1])
        exit_price = last_close * (1.0 - TAKER_SLIP) if pos == "LONG" else last_close * (1.0 + TAKER_SLIP)
        realized = _realized_return(pos, entry_price, exit_price, cur_lev, entry_fee, TAKER_FEE)
        balance *= 1.0 + realized
        trades += 1
        wins += int(realized > 0.0)

    return {
        "method": "adaptive_limit",
        "config": asdict(cfg),
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "wr_pct": round((100.0 * wins / trades) if trades else 0.0, 2),
        "trades": trades,
        "sharpe": round(_sharpe(eq_curve), 4),
        "mdd_pct": round(_mdd(eq_curve), 4),
        "maker_entry_ratio": round((maker_entries / max(maker_entries + fallback_entries, 1)), 4),
        "missed_entries": missed_entries,
        "maker_entries": maker_entries,
        "fallback_entries": fallback_entries,
        "market_exits": market_exits,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = _load_df()
    actor = _build_actor(device)
    router_market = DSACRouter(actor, device=device)

    baseline = simulate_market(df, router_market)
    print("[BASELINE]", baseline)

    configs = [
        LimitConfig("ultra_tight", 0.45, 0.25, 0.20, 0.88, 0.0, 1, 0, 0.70, 0.05),
        LimitConfig("tight_anchor", 0.60, 0.35, 0.30, 0.82, 0.0, 1, 0, 0.72, 0.05),
        LimitConfig("balanced_anchor", 0.75, 0.50, 0.45, 0.76, 0.0, 2, 0, 0.75, 0.06),
        LimitConfig("flow_aggressive", 0.65, 0.40, 0.25, 0.90, 0.0, 1, 0, 0.68, 0.04),
        LimitConfig("volatility_buffered", 0.85, 0.70, 0.60, 0.72, 0.0, 2, 0, 0.78, 0.08),
        LimitConfig("maker_bias", 1.00, 0.75, 0.80, 0.85, 0.0, 2, 0, 0.82, 0.10),
    ]

    results = []
    for cfg in configs:
        router_limit = DSACRouter(actor, device=device)
        result = simulate_limit(df, router_limit, cfg)
        result["delta_vs_market_pct"] = round(result["pnl_pct"] - baseline["pnl_pct"], 4)
        results.append(result)
        print(
            "[LIMIT]",
            cfg.name,
            f"pnl={result['pnl_pct']:.2f}%",
            f"delta={result['delta_vs_market_pct']:+.2f}%",
            f"wr={result['wr_pct']:.2f}%",
            f"trades={result['trades']}",
            f"maker_entry_ratio={result['maker_entry_ratio']:.2f}",
            f"missed={result['missed_entries']}",
        )

    best = max(results, key=lambda x: (x["pnl_pct"], -abs(x["mdd_pct"])))
    report = {
        "checkpoint": CKPT,
        "data_period": f"{df['timestamp'].min()} -> {df['timestamp'].max()}",
        "data_rows": int(len(df)),
        "baseline_market": baseline,
        "tested_configs": results,
        "best_limit": best,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("[BEST]", best["config"]["name"], best)
    print("[SAVED]", OUT_JSON)


if __name__ == "__main__":
    main()
