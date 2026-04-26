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
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from ensemble.train_rl_dsac_agent import DSACRouter, DSAC_STATE_DIM, GaussianActor

FILES = [
    "data/splits/year_oos/rl_meta_2026.csv",
]
CKPT = "data/ensemble/ckpt/best_dsac_agents.pth"
OUT_JSON = "data/ensemble/reports/backtest_dsac_execution_overlay_oos.json"
TAKER_FEE = 0.0005
MAKER_FEE = 0.0002
TAKER_SLIP = 0.0002
ANNUAL_FACTOR_5M = math.sqrt(365 * 24 * 12)


@dataclass
class OverlayConfig:
    name: str
    wait_enter_th: float
    wait_release_th: float
    max_wait_bars: int
    fallback_raw_th: float
    fallback_conf_th: float
    offset_scale: float
    adverse_mult: float


def _sharpe(eq_curve: list[float]) -> float:
    eq = np.array(eq_curve, dtype=np.float64)
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    if len(rets) < 3 or np.std(rets) < 1e-12:
        return 0.0
    return float(np.mean(rets) / np.std(rets) * ANNUAL_FACTOR_5M)


def _mdd(eq_curve: list[float]) -> float:
    eq = np.array(eq_curve, dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = eq / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(dd)) * 100.0


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return x


def _load_actor(device: str) -> GaussianActor:
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    actor = GaussianActor(state_dim=int(ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM)).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def _load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp", "open", "high", "low", "close"]).reset_index(drop=True)
    df["ret_1"] = df["close"].pct_change().fillna(0.0)
    df["ret_3"] = df["close"].pct_change(3).fillna(0.0)
    return df


def _entry_plan(row: pd.Series, side: str, cfg: OverlayConfig) -> dict:
    close = _safe_float(row.get("close", 0.0), 0.0)
    conf = np.clip(_safe_float(row.get("m7_confidence", 0.5), 0.5), 0.0, 1.0)
    quality = np.clip(_safe_float(row.get("m7_quality_pred", 0.005), 0.005), 0.0, 0.05)
    qwidth = max(_safe_float(row.get("m7_qwidth", 0.01), 0.01), 1e-4)
    if side == "LONG":
        offset = abs(_safe_float(row.get("m7_entry_long_offset", -0.0016), -0.0016))
        price = _safe_float(row.get("m7_entry_long_price", close * (1 - offset)), 0.0)
        fallback = close * (1.0 - offset * cfg.offset_scale)
        limit_price = 0.7 * price + 0.3 * fallback if price > 0 else fallback
    else:
        offset = abs(_safe_float(row.get("m7_entry_short_offset", 0.0016), 0.0016))
        price = _safe_float(row.get("m7_entry_short_price", close * (1 + offset)), 0.0)
        fallback = close * (1.0 + offset * cfg.offset_scale)
        limit_price = 0.7 * price + 0.3 * fallback if price > 0 else fallback
    ttl = 1 if conf > 0.85 and quality > 0.006 else 2
    return {"limit_price": float(limit_price), "ttl": ttl}


def _wait_score(row: pd.Series, side: str, cfg: OverlayConfig) -> float:
    sign = 1.0 if side == "LONG" else -1.0
    ret1 = sign * _safe_float(row.get("ret_1", 0.0), 0.0)
    ret3 = sign * _safe_float(row.get("ret_3", 0.0), 0.0)
    smf = sign * _safe_float(row.get("smart_money_flow", 0.0), 0.0)
    ofi = sign * _safe_float(row.get("ofi_acceleration", 0.0), 0.0)
    cvp = sign * _safe_float(row.get("cvp_volume_imbalance", 0.0), 0.0)
    qwidth = _safe_float(row.get("m7_qwidth", 0.01), 0.01)
    conf = _safe_float(row.get("m7_confidence", 0.5), 0.5)
    adverse = (
        0.34 * np.tanh(-ret1 / 0.0020)
        + 0.22 * np.tanh(-ret3 / 0.0040)
        + 0.14 * np.tanh(-smf / 0.0025)
        + 0.12 * np.tanh(-ofi / 0.14)
        + 0.10 * np.tanh(-cvp / 0.55)
        + 0.08 * np.tanh(qwidth / 0.012)
        - 0.10 * conf
    )
    return float(adverse * cfg.adverse_mult)


def _realized(side: str, ep: float, xp: float, lev: float, ef: float, xf: float) -> float:
    gross = (xp - ep) / ep if side == "LONG" else (ep - xp) / ep
    return float(gross * lev)


def _unrealized(side: str | None, ep: float, cp: float, lev: float, ef: float) -> float:
    if side is None or ep <= 0 or lev <= 0:
        return 0.0
    gross = (cp - ep) / ep if side == "LONG" else (ep - cp) / ep
    return float(gross * lev)


def simulate_market(df: pd.DataFrame, router: DSACRouter) -> dict:
    numeric_cols = [c for c in df.columns if c != "timestamp"]
    values = df[numeric_cols].to_numpy(dtype=np.float64)
    open_np = df["open"].to_numpy(dtype=np.float64)
    close_np = df["close"].to_numpy(dtype=np.float64)
    balance = 1.0
    eq_curve = [1.0]
    pos = None
    ep = 0.0
    ef = 0.0
    lev = 0.0
    hold = 0
    trades = wins = 0
    for i in range(len(df) - 1):
        cp = float(close_np[i])
        next_open = float(open_np[i + 1])
        next_close = float(close_np[i + 1])
        if pos is not None:
            hold += 1
        pos_dict = {
            "type": pos,
            "entry_price": float(ep),
            "unrealized": float(_unrealized(pos, ep, cp, lev, ef)),
            "mdd": 0.0,
            "hold_norm": float(min(hold / 96.0, 1.0)),
            "margin_usage": float(lev if pos else 0.0),
            "hold_count": float(hold),
        }
        features = {k: float(v) for k, v in zip(numeric_cols, values[i])}
        action, kelly, _ = router.decide(features, pos_dict)
        kelly = float(np.clip(kelly, 0.0, 1.0))
        if pos is None:
            if action == 1 and kelly > 0:
                pos, ep, ef, lev, hold = "LONG", next_open * (1.0 + TAKER_SLIP), TAKER_FEE, kelly, 0
                balance -= balance * TAKER_FEE * lev
            elif action == 2 and kelly > 0:
                pos, ep, ef, lev, hold = "SHORT", next_open * (1.0 - TAKER_SLIP), TAKER_FEE, kelly, 0
                balance -= balance * TAKER_FEE * lev
        else:
            should_close = action == 0 or (action == 1 and pos == "SHORT") or (action == 2 and pos == "LONG")
            if should_close:
                xp = next_open * (1.0 - TAKER_SLIP) if pos == "LONG" else next_open * (1.0 + TAKER_SLIP)
                r = _realized(pos, ep, xp, lev, ef, TAKER_FEE)
                balance = balance * (1.0 + r) - balance * TAKER_FEE * lev
                trades += 1
                wins += int(r > 0.0)
                pos, ep, ef, lev, hold = None, 0.0, 0.0, 0.0, 0
        eq_curve.append(max(balance * (1.0 + (_unrealized(pos, ep, next_close, lev, ef) if pos else 0.0)), 1e-8))
    return {
        "mode": "market",
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "trades": trades,
        "wr_pct": round((100.0 * wins / trades) if trades else 0.0, 2),
        "sharpe": round(_sharpe(eq_curve), 4),
        "mdd_pct": round(_mdd(eq_curve), 4),
    }


def simulate_overlay(df: pd.DataFrame, router: DSACRouter, cfg: OverlayConfig) -> dict:
    numeric_cols = [c for c in df.columns if c != "timestamp"]
    values = df[numeric_cols].to_numpy(dtype=np.float64)
    open_np = df["open"].to_numpy(dtype=np.float64)
    high_np = df["high"].to_numpy(dtype=np.float64)
    low_np = df["low"].to_numpy(dtype=np.float64)
    close_np = df["close"].to_numpy(dtype=np.float64)
    balance = 1.0
    eq_curve = [1.0]
    pos = None
    ep = 0.0
    ef = 0.0
    lev = 0.0
    hold = 0
    trades = wins = 0
    maker_entries = fallback_entries = missed_entries = 0
    wait_releases = wait_cancels = 0
    pending = None
    waiting = None
    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        cp = float(prev["close"])
        bar_open = float(open_np[i])
        bar_high = float(high_np[i])
        bar_low = float(low_np[i])
        bar_close = float(close_np[i])

        if pos is not None:
            hold += 1

        if waiting is not None:
            side = waiting["side"]
            score = _wait_score(prev, side, cfg)
            favorable = (_safe_float(prev.get("ret_1", 0.0), 0.0) > 0 if side == "LONG" else _safe_float(prev.get("ret_1", 0.0), 0.0) < 0)
            if i > waiting["expire_idx"] or score > cfg.wait_enter_th + 0.10:
                wait_cancels += 1
                waiting = None
            elif score <= cfg.wait_release_th or favorable:
                plan = _entry_plan(prev, side, cfg)
                pending = {
                    "side": side,
                    "price": float(plan["limit_price"]),
                    "expire_idx": i + int(plan["ttl"]),
                    "lev": float(waiting["lev"]),
                    "fallback": bool(waiting["raw_abs"] >= cfg.fallback_raw_th and waiting["conf"] >= cfg.fallback_conf_th),
                }
                wait_releases += 1
                waiting = None

        if pending is not None:
            fill = (pending["side"] == "LONG" and bar_low <= pending["price"]) or (pending["side"] == "SHORT" and bar_high >= pending["price"])
            if fill:
                pos, ep, ef, lev, hold = pending["side"], float(pending["price"]), MAKER_FEE, float(pending["lev"]), 0
                balance -= balance * MAKER_FEE * lev
                maker_entries += 1
                pending = None
            elif i > pending["expire_idx"]:
                if pending["fallback"]:
                    pos = pending["side"]
                    ep = bar_open * (1.0 + TAKER_SLIP) if pos == "LONG" else bar_open * (1.0 - TAKER_SLIP)
                    ef, lev, hold = TAKER_FEE, float(pending["lev"]), 0
                    balance -= balance * TAKER_FEE * lev
                    fallback_entries += 1
                else:
                    missed_entries += 1
                pending = None

        pos_dict = {
            "type": pos,
            "entry_price": float(ep),
            "unrealized": float(_unrealized(pos, ep, cp, lev, ef)),
            "mdd": 0.0,
            "hold_norm": float(min(hold / 96.0, 1.0)),
            "margin_usage": float(lev if pos else 0.0),
            "hold_count": float(hold),
        }
        features = {k: float(v) for k, v in zip(numeric_cols, values[i - 1])}
        action, kelly, info = router.decide(features, pos_dict)
        kelly = float(np.clip(kelly, 0.0, 1.0))
        raw_abs = abs(_safe_float(info.get("raw_action", 0.0), 0.0))
        conf = np.clip(_safe_float(prev.get("m7_confidence", 0.5), 0.5), 0.0, 1.0)

        if pos is None and pending is None and waiting is None:
            if action == 1 and kelly > 0:
                score = _wait_score(prev, "LONG", cfg)
                if score > cfg.wait_enter_th:
                    waiting = {"side": "LONG", "expire_idx": i + cfg.max_wait_bars, "lev": kelly, "raw_abs": raw_abs, "conf": conf}
                else:
                    plan = _entry_plan(prev, "LONG", cfg)
                    pending = {
                        "side": "LONG",
                        "price": float(plan["limit_price"]),
                        "expire_idx": i + int(plan["ttl"]),
                        "lev": kelly,
                        "fallback": bool(raw_abs >= cfg.fallback_raw_th and conf >= cfg.fallback_conf_th),
                    }
            elif action == 2 and kelly > 0:
                score = _wait_score(prev, "SHORT", cfg)
                if score > cfg.wait_enter_th:
                    waiting = {"side": "SHORT", "expire_idx": i + cfg.max_wait_bars, "lev": kelly, "raw_abs": raw_abs, "conf": conf}
                else:
                    plan = _entry_plan(prev, "SHORT", cfg)
                    pending = {
                        "side": "SHORT",
                        "price": float(plan["limit_price"]),
                        "expire_idx": i + int(plan["ttl"]),
                        "lev": kelly,
                        "fallback": bool(raw_abs >= cfg.fallback_raw_th and conf >= cfg.fallback_conf_th),
                    }
        elif pos is not None:
            should_close = action == 0 or (action == 1 and pos == "SHORT") or (action == 2 and pos == "LONG")
            if should_close:
                xp = float(open_np[i + 1]) * (1.0 - TAKER_SLIP) if pos == "LONG" else float(open_np[i + 1]) * (1.0 + TAKER_SLIP)
                r = _realized(pos, ep, xp, lev, ef, TAKER_FEE)
                balance = balance * (1.0 + r) - balance * TAKER_FEE * lev
                trades += 1
                wins += int(r > 0.0)
                pos, ep, ef, lev, hold = None, 0.0, 0.0, 0.0, 0

        eq_curve.append(max(balance * (1.0 + (_unrealized(pos, ep, bar_close, lev, ef) if pos else 0.0)), 1e-8))

    return {
        "mode": "wait_limit_entry_overlay",
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "trades": trades,
        "wr_pct": round((100.0 * wins / trades) if trades else 0.0, 2),
        "sharpe": round(_sharpe(eq_curve), 4),
        "mdd_pct": round(_mdd(eq_curve), 4),
        "maker_entry_ratio": round(maker_entries / max(maker_entries + fallback_entries, 1), 4),
        "missed_entries": missed_entries,
        "wait_releases": wait_releases,
        "wait_cancels": wait_cancels,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor = _load_actor(device)
    configs = [
        OverlayConfig("balanced", 0.12, 0.04, 2, 0.18, 0.70, 1.0, 1.0),
        OverlayConfig("conservative", 0.18, 0.06, 3, 0.24, 0.75, 0.9, 1.15),
        OverlayConfig("reactive", 0.10, 0.02, 1, 0.15, 0.65, 1.1, 0.9),
    ]
    reports = []
    for path in FILES:
        df = _load_df(path)
        file_results = []
        for cfg in configs:
            router_market = DSACRouter(actor, device=device)
            router_overlay = DSACRouter(actor, device=device)
            market = simulate_market(df, router_market)
            overlay = simulate_overlay(df, router_overlay, cfg)
            overlay["delta_vs_market_pct"] = round(overlay["pnl_pct"] - market["pnl_pct"], 4)
            file_results.append({"config": asdict(cfg), "market": market, "overlay": overlay})
            print(os.path.basename(path), cfg.name, market, overlay)
        best = max(file_results, key=lambda x: x["overlay"]["delta_vs_market_pct"])
        reports.append(
            {
                "file": path,
                "rows": int(len(df)),
                "period": f"{df['timestamp'].min()} -> {df['timestamp'].max()}",
                "results": file_results,
                "best": best,
            }
        )

    overall_best = max(reports, key=lambda x: x["best"]["overlay"]["delta_vs_market_pct"])
    out = {
        "checkpoint": CKPT,
        "notes": [
            "DSAC direction is unchanged; only entry execution changes from market to wait-limit overlay.",
            "Overlay uses M7 entry offsets/prices plus short-horizon adverse-move waiting logic.",
            "2026 OOS rl_meta file is used for the backtest.",
        ],
        "reports": reports,
        "overall_best": overall_best,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("SAVED", OUT_JSON)


if __name__ == "__main__":
    main()
