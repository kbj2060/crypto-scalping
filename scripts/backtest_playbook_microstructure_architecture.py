#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
MICRO_DB = ROOT / "data/live/microstructure.duckdb"
TAIL_DB = ROOT / "data/live/tail_risk.duckdb"
OUT_PATH = ROOT / "data/ensemble/metrics/playbook_microstructure_architecture_backtest.json"


import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playbook_router import PlaybookRouter


@dataclass
class BacktestStats:
    pnl_pct: float
    sharpe_1m: float
    mdd_pct: float
    win_rate_pct: float
    trades: int
    maker_ratio_pct: float
    avg_hold_min: float
    avg_kelly: float
    final_equity: float
    symbol: str
    leverage: float
    fee_taker_bps: float
    fee_maker_bps: float
    slip_taker_bps: float
    slip_maker_bps: float
    rows: int
    by_playbook: dict[str, Any]


@dataclass
class StrategyTuning:
    pb16_max_toxicity: float = 0.50
    pb16_min_eai_delta: float = 0.03
    pb16_min_abs_whale: float = 0.18
    pb16_kelly_mult: float = 0.70
    pb10_kelly_boost: float = 1.35
    pb8_kelly_boost: float = 1.25


def load_signals() -> pd.DataFrame:
    con_m = duckdb.connect(str(MICRO_DB))
    con_t = duckdb.connect(str(TAIL_DB))
    m = con_m.execute(
        """
        SELECT
          date_trunc('minute', ts) AS ts,
          obi, taker_buy_ratio, spoofing_score,
          nif_whale, nif_retail, eai, oi_delta_pct, funding_rate,
          shadow_toxicity_score, shadow_queue_collapse, shadow_absorption_score, shadow_queue_bias,
          shadow_regime_tag, shadow_regime_conf
        FROM microstructure_1m
        ORDER BY ts
        """
    ).df()
    t = con_t.execute(
        """
        SELECT
          date_trunc('minute', ts) AS ts,
          long_usd_1m, short_usd_1m, shadow_aftershock_prob, shadow_risk_bucket
        FROM tail_risk_1m
        ORDER BY ts
        """
    ).df()
    con_m.close()
    con_t.close()
    m["ts"] = pd.to_datetime(m["ts"], utc=True)
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    df = pd.merge(m, t, on="ts", how="inner")
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def fetch_binance_1m_ohlcv(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/klines"
    out: list[list[Any]] = []
    cur = start_ms
    while cur <= end_ms:
        params = {"symbol": symbol, "interval": "1m", "startTime": cur, "endTime": end_ms, "limit": 1500}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        out.extend(rows)
        cur = int(rows[-1][0]) + 60_000
        if len(rows) < 1500:
            break
    if not out:
        raise RuntimeError("No Binance klines fetched.")

    px = pd.DataFrame(
        out,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
        ],
    )
    px["ts"] = pd.to_datetime(px["open_time"].astype(np.int64), unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume", "quote_volume"):
        px[c] = pd.to_numeric(px[c], errors="coerce")
    return px[["ts", "open", "high", "low", "close", "volume", "quote_volume"]].dropna().drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def build_frame(symbol: str) -> pd.DataFrame:
    sig = load_signals()
    if len(sig) < 300:
        raise RuntimeError(f"Not enough signal rows: {len(sig)}")
    start_ms = int(sig["ts"].min().timestamp() * 1000) - 360_000
    end_ms = int(sig["ts"].max().timestamp() * 1000) + 360_000
    px = fetch_binance_1m_ohlcv(symbol=symbol, start_ms=start_ms, end_ms=end_ms)

    sig["ts"] = pd.to_datetime(sig["ts"], utc=True).astype("datetime64[ns, UTC]")
    px["ts"] = pd.to_datetime(px["ts"], utc=True).astype("datetime64[ns, UTC]")
    df = pd.merge_asof(sig.sort_values("ts"), px.sort_values("ts"), on="ts", direction="backward")
    df = df.dropna(subset=["close", "high", "low"]).reset_index(drop=True)

    df["price_change_30m"] = df["close"] / df["close"].shift(30) - 1.0
    roll30_max = df["close"].rolling(30, min_periods=10).max()
    roll30_min = df["close"].rolling(30, min_periods=10).min()
    df["price_volatility_30m"] = (roll30_max - roll30_min) / (df["close"] + 1e-12)
    prev55_max = df["close"].shift(5).rolling(55, min_periods=20).max()
    prev55_min = df["close"].shift(5).rolling(55, min_periods=20).min()
    df["price_breakout_60m"] = df["close"] > prev55_max
    df["price_breakdown_60m"] = df["close"] < prev55_min

    df["nif_whale_sum_30m"] = df["nif_whale"].rolling(30, min_periods=10).sum()
    df["nif_whale_avg_30m"] = df["nif_whale"].rolling(30, min_periods=10).mean()
    df["nif_whale_std_30m"] = df["nif_whale"].rolling(30, min_periods=10).std()
    df["absorption_avg_30m"] = df["shadow_absorption_score"].rolling(30, min_periods=10).mean()
    df["bias_avg_30m"] = df["shadow_queue_bias"].rolling(30, min_periods=10).mean()
    df["toxicity_avg_30m"] = df["shadow_toxicity_score"].rolling(30, min_periods=10).mean()
    df["eai_delta_15m"] = df["eai"] - df["eai"].shift(15)

    vwap_15 = (df["quote_volume"].rolling(15, min_periods=5).sum() / (df["volume"].rolling(15, min_periods=5).sum() + 1e-12))
    df["vwap_gap_15m"] = (df["close"] - vwap_15) / (vwap_15 + 1e-12)
    vwap_30 = (df["quote_volume"].rolling(30, min_periods=10).sum() / (df["volume"].rolling(30, min_periods=10).sum() + 1e-12))
    df["vwap_30m"] = vwap_30

    mu_l = df["long_usd_1m"].rolling(30, min_periods=10).mean()
    sd_l = df["long_usd_1m"].rolling(30, min_periods=10).std().clip(lower=1e-6)
    mu_s = df["short_usd_1m"].rolling(30, min_periods=10).mean()
    sd_s = df["short_usd_1m"].rolling(30, min_periods=10).std().clip(lower=1e-6)
    df["z_long"] = (df["long_usd_1m"] - mu_l) / sd_l
    df["z_short"] = (df["short_usd_1m"] - mu_s) / sd_s
    ret1 = (df["close"] / df["close"].shift(1) - 1.0).abs().clip(lower=1e-4)
    dominant = np.where(df["z_long"] >= df["z_short"], df["long_usd_1m"], df["short_usd_1m"])
    df["lai"] = dominant / ret1

    prev_high_60 = df["high"].shift(1).rolling(60, min_periods=20).max()
    prev_low_60 = df["low"].shift(1).rolling(60, min_periods=20).min()
    df["prev_high_60"] = prev_high_60
    df["prev_low_60"] = prev_low_60

    df["ret_fwd_1m"] = df["close"].shift(-1) / df["close"] - 1.0
    keep = [
        "ts", "open", "high", "low", "close", "ret_fwd_1m",
        "obi", "nif_whale", "nif_retail", "eai", "funding_rate",
        "shadow_toxicity_score", "shadow_queue_collapse", "shadow_absorption_score", "shadow_queue_bias",
        "shadow_aftershock_prob", "shadow_risk_bucket",
        "price_change_30m", "price_volatility_30m", "price_breakout_60m", "price_breakdown_60m",
        "nif_whale_sum_30m", "nif_whale_avg_30m", "nif_whale_std_30m",
        "absorption_avg_30m", "bias_avg_30m", "toxicity_avg_30m", "eai_delta_15m",
        "vwap_gap_15m", "vwap_30m", "prev_high_60", "prev_low_60",
        "z_long", "z_short", "lai",
    ]
    out = df[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(out) < 200:
        raise RuntimeError(f"Not enough rows after feature build: {len(out)}")
    return out


def pick_decision(eval_out: dict[str, Any]) -> tuple[str, int, float]:
    winner_hft = eval_out.get("winner_hft", {}) or {}
    winner_mft = eval_out.get("winner_mft", {}) or {}
    evals = eval_out.get("evaluations", []) or []
    matched = {str(e.get("name")): e for e in evals if bool(e.get("matched", False))}

    if "PB9_VACUUM_WHIPSAW" in matched:
        return "PB9_VACUUM_WHIPSAW", 0, 0.0
    if "PB5_MAMMOTH_SNIPER" in matched:
        e = matched["PB5_MAMMOTH_SNIPER"]
        return "PB5_MAMMOTH_SNIPER", int(e.get("action", 0)), float(e.get("kelly", 0.0))

    mft_name = str(winner_mft.get("name", ""))
    mft_matched = bool(winner_mft.get("matched", False))
    if mft_matched and mft_name:
        hft_name = str(winner_hft.get("name", ""))
        hft_matched = bool(winner_hft.get("matched", False))
        if hft_matched and hft_name == "PB8_HOLY_TRINITY_TRAP":
            if int(winner_hft.get("action", 0)) != int(winner_mft.get("action", 0)):
                return "CLASH_HOLD", 0, 0.0
        return mft_name, int(winner_mft.get("action", 0)), float(winner_mft.get("kelly", 0.0))

    hft_name = str(winner_hft.get("name", ""))
    hft_matched = bool(winner_hft.get("matched", False))
    if hft_matched and hft_name:
        return hft_name, int(winner_hft.get("action", 0)), float(winner_hft.get("kelly", 0.0))

    return "NONE", 0, 0.0


def playbook_profile(name: str) -> dict[str, Any]:
    # mode: maker/taker, ttl bars, sl/tp/trailing, path-dependent hints
    m = {
        "mode": "taker",
        "ttl": 30,
        "sl": 0.008,
        "tp1": 0.010,
        "tp2": 0.016,
        "trail_arm": 0.008,
        "trail_gap": 0.004,
    }
    if name in {"PB8_HOLY_TRINITY_TRAP", "PB13_BREAKOUT_TRAP"}:
        m.update({"mode": "maker", "ttl": 3 if name == "PB8_HOLY_TRINITY_TRAP" else 5, "sl": 0.0035, "tp1": 0.0025, "tp2": 0.0040, "trail_arm": 0.0020, "trail_gap": 0.0015})
    elif name == "PB9_VACUUM_WHIPSAW":
        m.update({"mode": "none", "ttl": 1})
    elif name == "PB5_MAMMOTH_SNIPER":
        m.update({"mode": "taker", "ttl": 12, "sl": 0.007, "tp1": 0.010, "tp2": 0.018, "trail_arm": 0.009, "trail_gap": 0.005})
    elif name == "PB2_SQUEEZE_IGNITION":
        m.update({"mode": "taker", "ttl": 15, "sl": 0.006, "tp1": 0.009, "tp2": 0.015, "trail_arm": 0.007, "trail_gap": 0.0035})
    elif name == "PB7_HOLY_TRINITY_TREND":
        m.update({"mode": "taker", "ttl": 25, "sl": 0.007, "tp1": 0.010, "tp2": 0.018, "trail_arm": 0.010, "trail_gap": 0.004})
    elif name in {"PB10_CVD_DIVERGENCE", "PB12_FUNDING_SNAPBACK"}:
        m.update({"mode": "maker", "ttl": 40 if name == "PB10_CVD_DIVERGENCE" else 30, "sl": 0.0075, "tp1": 0.008, "tp2": 0.014, "trail_arm": 0.009, "trail_gap": 0.0045})
    elif name == "PB11_TWAP_ABSORPTION":
        m.update({"mode": "taker", "ttl": 60, "sl": 0.009, "tp1": 0.012, "tp2": 0.020, "trail_arm": 0.012, "trail_gap": 0.006})
    elif name == "PB14_STEALTH_MOMENTUM":
        m.update({"mode": "taker", "ttl": 60, "sl": 0.009, "tp1": 0.013, "tp2": 0.022, "trail_arm": 0.013, "trail_gap": 0.006})
    elif name == "PB15_VWAP_MEAN_REVERSION":
        m.update({"mode": "maker", "ttl": 20, "sl": 0.0055, "tp1": 0.006, "tp2": 0.010, "trail_arm": 0.005, "trail_gap": 0.0025})
    elif name == "PB16_MICRO_TREND_SURFING":
        m.update({"mode": "taker", "ttl": 15, "sl": 0.006, "tp1": 0.008, "tp2": 0.012, "trail_arm": 0.007, "trail_gap": 0.003})
    return m


def simulate(
    df: pd.DataFrame,
    leverage: float,
    fee_taker_bps: float,
    fee_maker_bps: float,
    slip_taker_bps: float,
    slip_maker_bps: float,
    maker_entry_offset_bps: float,
    tuning: StrategyTuning,
) -> BacktestStats:
    r = PlaybookRouter()
    lev = float(max(leverage, 0.0))
    fee_taker = fee_taker_bps / 10_000.0
    fee_maker = fee_maker_bps / 10_000.0
    slip_taker = slip_taker_bps / 10_000.0
    slip_maker = slip_maker_bps / 10_000.0
    maker_offset = maker_entry_offset_bps / 10_000.0

    equity = 1.0
    pnl_series: list[float] = []
    eq_series: list[float] = []
    pos = None
    pending_order = None
    maker_entries = 0
    all_entries = 0
    hold_bars_total = 0
    kelly_hist: list[float] = []
    trade_pnls: list[float] = []
    stats_by_pb: dict[str, dict[str, float]] = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl_sum": 0.0})

    def close_position(px: float, fee: float, reason: str) -> float:
        nonlocal pos
        if pos is None:
            return 0.0
        side = pos["side"]
        size = pos["size"]
        entry = pos["entry"]
        ret = (px / entry - 1.0) if side == 1 else (entry / px - 1.0)
        gross = lev * size * ret
        net = gross - lev * size * fee
        pb = pos["pb"]
        stats_by_pb[pb]["trades"] += 1
        stats_by_pb[pb]["wins"] += 1 if net > 0 else 0
        stats_by_pb[pb]["pnl_sum"] += net
        trade_pnls.append(net)
        hold = int(pos["hold"])
        pos = None
        return net, hold

    for row in df.itertuples(index=False):
        ms = {
            "obi": float(row.obi),
            "nif_whale": float(row.nif_whale),
            "nif_retail": float(row.nif_retail),
            "eai": float(row.eai),
            "funding_rate": float(row.funding_rate),
            "shadow_absorption_score": float(row.shadow_absorption_score),
            "shadow_queue_collapse": float(row.shadow_queue_collapse),
            "shadow_toxicity_score": float(row.shadow_toxicity_score),
            "shadow_queue_bias": int(row.shadow_queue_bias),
            "price_change_30m": float(row.price_change_30m),
            "price_volatility_30m": float(row.price_volatility_30m),
            "price_breakout_60m": bool(row.price_breakout_60m),
            "price_breakdown_60m": bool(row.price_breakdown_60m),
            "nif_whale_sum_30m": float(row.nif_whale_sum_30m),
            "nif_whale_avg_30m": float(row.nif_whale_avg_30m),
            "nif_whale_std_30m": float(row.nif_whale_std_30m),
            "absorption_avg_30m": float(row.absorption_avg_30m),
            "bias_avg_30m": float(row.bias_avg_30m),
            "toxicity_avg_30m": float(row.toxicity_avg_30m),
            "eai_delta_15m": float(row.eai_delta_15m),
            "vwap_gap_15m": float(row.vwap_gap_15m),
        }
        tr = {
            "z_long": float(row.z_long),
            "z_short": float(row.z_short),
            "lai": float(row.lai),
            "shadow_aftershock_prob": float(row.shadow_aftershock_prob),
            "shadow_risk_bucket": str(row.shadow_risk_bucket),
        }

        out = r.evaluate_all(action=0, pos=None, kelly=1.0, ms=ms, tr=tr)
        pb_name, action, kelly = pick_decision(out)
        kelly = float(np.clip(kelly, 0.0, 1.0))

        # PB16 강화 필터: 미니모멘텀 오염구간 제거
        if pb_name == "PB16_MICRO_TREND_SURFING":
            if (
                float(row.shadow_toxicity_score) > tuning.pb16_max_toxicity
                or float(row.eai_delta_15m) < tuning.pb16_min_eai_delta
                or abs(float(row.nif_whale)) < tuning.pb16_min_abs_whale
            ):
                pb_name, action, kelly = "PB16_FILTERED_HOLD", 0, 0.0
            else:
                kelly = float(np.clip(kelly * tuning.pb16_kelly_mult, 0.0, 1.0))

        # PB10/PB8 우선 비중 상향
        if pb_name == "PB10_CVD_DIVERGENCE":
            kelly = float(np.clip(kelly * tuning.pb10_kelly_boost, 0.0, 1.0))
        elif pb_name == "PB8_HOLY_TRINITY_TRAP":
            kelly = float(np.clip(kelly * tuning.pb8_kelly_boost, 0.0, 1.0))

        kelly_hist.append(kelly)
        profile = playbook_profile(pb_name)

        # pending maker order fill/cancel
        if pending_order is not None and pos is None:
            pending_order["ttl"] -= 1
            p_side = pending_order["side"]
            p_px = pending_order["price"]
            if p_side == 1 and float(row.low) <= p_px:
                entry_px = p_px * (1.0 + slip_maker)
                pos = {
                    "side": 1,
                    "entry": entry_px,
                    "size": pending_order["size"],
                    "pb": pending_order["pb"],
                    "ttl": pending_order["profile"]["ttl"],
                    "hold": 0,
                    "remaining": 1.0,
                    "peak": entry_px,
                    "trough": entry_px,
                    "profile": pending_order["profile"],
                    "invalidation": pending_order.get("invalidation"),
                    "did_partial": False,
                }
                equity *= (1.0 - lev * pos["size"] * fee_maker)
                pending_order = None
                maker_entries += 1
                all_entries += 1
            elif p_side == -1 and float(row.high) >= p_px:
                entry_px = p_px * (1.0 - slip_maker)
                pos = {
                    "side": -1,
                    "entry": entry_px,
                    "size": pending_order["size"],
                    "pb": pending_order["pb"],
                    "ttl": pending_order["profile"]["ttl"],
                    "hold": 0,
                    "remaining": 1.0,
                    "peak": entry_px,
                    "trough": entry_px,
                    "profile": pending_order["profile"],
                    "invalidation": pending_order.get("invalidation"),
                    "did_partial": False,
                }
                equity *= (1.0 - lev * pos["size"] * fee_maker)
                pending_order = None
                maker_entries += 1
                all_entries += 1
            elif pending_order["ttl"] <= 0:
                pending_order = None

        # manage live position (path-dependent exits + ttl)
        pnl_step = 0.0
        hold_inc = 0
        if pos is not None:
            pos["hold"] += 1
            pos["ttl"] -= 1
            side = pos["side"]
            entry = pos["entry"]
            size = pos["size"] * pos["remaining"]
            prof = pos["profile"]

            if side == 1:
                pos["peak"] = max(pos["peak"], float(row.high))
                worst_ret = float(row.low) / entry - 1.0
                best_ret = float(row.high) / entry - 1.0
            else:
                pos["trough"] = min(pos["trough"], float(row.low))
                worst_ret = entry / float(row.high) - 1.0
                best_ret = entry / float(row.low) - 1.0

            # dynamic invalidation for PB13
            inv = pos.get("invalidation")
            if inv is not None:
                if side == 1 and float(row.low) < inv:
                    step, hold = close_position(inv * (1.0 - slip_taker), fee_taker, "invalidation")
                    pnl_step += step
                    hold_inc += hold
                elif side == -1 and float(row.high) > inv:
                    step, hold = close_position(inv * (1.0 + slip_taker), fee_taker, "invalidation")
                    pnl_step += step
                    hold_inc += hold

            if pos is not None:
                # stop
                if worst_ret <= -prof["sl"]:
                    px = entry * (1.0 - prof["sl"]) if side == 1 else entry * (1.0 + prof["sl"])
                    px = px * (1.0 - slip_taker) if side == 1 else px * (1.0 + slip_taker)
                    step, hold = close_position(px, fee_taker, "hard_stop")
                    pnl_step += step
                    hold_inc += hold

            if pos is not None:
                # partial take profit (PB10/PB12/mean-reversion)
                if (not pos["did_partial"]) and (best_ret >= prof["tp1"]):
                    half = pos["remaining"] * 0.5
                    px = entry * (1.0 + prof["tp1"]) if side == 1 else entry * (1.0 - prof["tp1"])
                    px = px * (1.0 - slip_taker) if side == 1 else px * (1.0 + slip_taker)
                    ret = (px / entry - 1.0) if side == 1 else (entry / px - 1.0)
                    net = lev * pos["size"] * half * ret - lev * pos["size"] * half * fee_taker
                    equity *= (1.0 + net)
                    pnl_step += net
                    pos["remaining"] -= half
                    pos["did_partial"] = True

                # VWAP snapback assist for PB10/PB12/PB15
                if pos is not None and pos["pb"] in {"PB10_CVD_DIVERGENCE", "PB12_FUNDING_SNAPBACK", "PB15_VWAP_MEAN_REVERSION"}:
                    vwap = float(row.vwap_30m)
                    if side == 1 and float(row.close) >= vwap and not pos["did_partial"]:
                        pos["did_partial"] = True
                        half = pos["remaining"] * 0.5
                        px = float(row.close) * (1.0 - slip_taker)
                        ret = px / entry - 1.0
                        net = lev * pos["size"] * half * ret - lev * pos["size"] * half * fee_taker
                        equity *= (1.0 + net)
                        pnl_step += net
                        pos["remaining"] -= half
                    elif side == -1 and float(row.close) <= vwap and not pos["did_partial"]:
                        pos["did_partial"] = True
                        half = pos["remaining"] * 0.5
                        px = float(row.close) * (1.0 + slip_taker)
                        ret = entry / px - 1.0
                        net = lev * pos["size"] * half * ret - lev * pos["size"] * half * fee_taker
                        equity *= (1.0 + net)
                        pnl_step += net
                        pos["remaining"] -= half

            if pos is not None:
                # trailing stop
                if side == 1:
                    runup = pos["peak"] / entry - 1.0
                    if runup >= prof["trail_arm"]:
                        trail_px = pos["peak"] * (1.0 - prof["trail_gap"])
                        if float(row.low) <= trail_px:
                            px = trail_px * (1.0 - slip_taker)
                            step, hold = close_position(px, fee_taker, "trail")
                            pnl_step += step
                            hold_inc += hold
                else:
                    runup = entry / pos["trough"] - 1.0
                    if runup >= prof["trail_arm"]:
                        trail_px = pos["trough"] * (1.0 + prof["trail_gap"])
                        if float(row.high) >= trail_px:
                            px = trail_px * (1.0 + slip_taker)
                            step, hold = close_position(px, fee_taker, "trail")
                            pnl_step += step
                            hold_inc += hold

            if pos is not None and pos["ttl"] <= 0:
                px = float(row.close) * (1.0 - slip_taker if side == 1 else 1.0 + slip_taker)
                step, hold = close_position(px, fee_taker, "ttl")
                pnl_step += step
                hold_inc += hold

        # entry decision (no position)
        if pos is None and action in (1, 2) and kelly > 0:
            side = 1 if action == 1 else -1
            size = kelly
            if profile["mode"] == "taker":
                px = float(row.close) * (1.0 + slip_taker if side == 1 else 1.0 - slip_taker)
                invalidation = None
                if pb_name == "PB13_BREAKOUT_TRAP":
                    if side == -1:
                        invalidation = float(row.prev_high_60) * 1.001
                    else:
                        invalidation = float(row.prev_low_60) * 0.999
                pos = {
                    "side": side,
                    "entry": px,
                    "size": size,
                    "pb": pb_name,
                    "ttl": profile["ttl"],
                    "hold": 0,
                    "remaining": 1.0,
                    "peak": px,
                    "trough": px,
                    "profile": profile,
                    "invalidation": invalidation,
                    "did_partial": False,
                }
                equity *= (1.0 - lev * size * fee_taker)
                all_entries += 1
            elif profile["mode"] == "maker":
                limit_px = float(row.close) * (1.0 - maker_offset if side == 1 else 1.0 + maker_offset)
                invalidation = None
                if pb_name == "PB13_BREAKOUT_TRAP":
                    if side == -1:
                        invalidation = float(row.prev_high_60) * 1.001
                    else:
                        invalidation = float(row.prev_low_60) * 0.999
                pending_order = {
                    "side": side,
                    "price": limit_px,
                    "ttl": 2,
                    "size": size,
                    "pb": pb_name,
                    "profile": profile,
                    "invalidation": invalidation,
                }

        pnl_series.append(pnl_step)
        eq_series.append(equity)
        hold_bars_total += hold_inc

    # force close at end
    if pos is not None:
        side = pos["side"]
        px = float(df["close"].iloc[-1]) * (1.0 - slip_taker if side == 1 else 1.0 + slip_taker)
        step, hold = close_position(px, fee_taker, "eod")
        equity *= (1.0 + step)
        pnl_series.append(step)
        eq_series.append(equity)
        hold_bars_total += hold

    pnl = np.array(pnl_series, dtype=np.float64)
    eq = np.array(eq_series, dtype=np.float64) if eq_series else np.array([1.0])
    peak = np.maximum.accumulate(eq)
    dd = eq / (peak + 1e-12) - 1.0
    mdd = float(-dd.min()) if len(dd) else 0.0
    mu = float(pnl.mean()) if len(pnl) else 0.0
    sd = float(pnl.std()) + 1e-12
    sharpe = mu / sd
    trades = len(trade_pnls)
    wr = float(np.mean(np.array(trade_pnls) > 0.0)) if trades else 0.0
    avg_hold = float(hold_bars_total / max(trades, 1))
    maker_ratio = float(maker_entries / max(all_entries, 1))
    avg_kelly = float(np.mean(kelly_hist)) if kelly_hist else 0.0
    by_pb = {}
    for k, v in stats_by_pb.items():
        trd = int(v["trades"])
        by_pb[k] = {
            "trades": trd,
            "win_rate_pct": float((v["wins"] / trd) * 100.0) if trd else 0.0,
            "pnl_pct_sum": float(v["pnl_sum"] * 100.0),
        }

    return BacktestStats(
        pnl_pct=float((eq[-1] - 1.0) * 100.0),
        sharpe_1m=float(sharpe),
        mdd_pct=float(mdd * 100.0),
        win_rate_pct=float(wr * 100.0),
        trades=int(trades),
        maker_ratio_pct=float(maker_ratio * 100.0),
        avg_hold_min=float(avg_hold),
        avg_kelly=float(avg_kelly),
        final_equity=float(eq[-1]),
        symbol="ETHUSDT",
        leverage=float(lev),
        fee_taker_bps=float(fee_taker_bps),
        fee_maker_bps=float(fee_maker_bps),
        slip_taker_bps=float(slip_taker_bps),
        slip_maker_bps=float(slip_maker_bps),
        rows=int(len(df)),
        by_playbook=by_pb,
    )


def objective_from_stats(s: BacktestStats) -> float:
    return float(s.pnl_pct - 0.40 * s.mdd_pct + 0.03 * s.sharpe_1m)


def random_search(
    df: pd.DataFrame,
    trials: int,
    leverage: float,
    fee_taker_bps: float,
    fee_maker_bps: float,
    slip_taker_bps: float,
    slip_maker_bps: float,
    maker_entry_offset_bps: float,
    seed: int,
) -> tuple[BacktestStats, StrategyTuning, float]:
    rng = np.random.default_rng(seed)
    best_stats: BacktestStats | None = None
    best_tune: StrategyTuning | None = None
    best_obj = -1e18
    for _ in range(int(trials)):
        tune = StrategyTuning(
            pb16_max_toxicity=float(rng.uniform(0.20, 0.50)),
            pb16_min_eai_delta=float(rng.uniform(0.03, 0.25)),
            pb16_min_abs_whale=float(rng.uniform(0.18, 0.45)),
            pb16_kelly_mult=float(rng.uniform(0.20, 0.90)),
            pb10_kelly_boost=float(rng.uniform(1.00, 1.90)),
            pb8_kelly_boost=float(rng.uniform(1.00, 1.70)),
        )
        stats = simulate(
            df=df,
            leverage=leverage,
            fee_taker_bps=fee_taker_bps,
            fee_maker_bps=fee_maker_bps,
            slip_taker_bps=slip_taker_bps,
            slip_maker_bps=slip_maker_bps,
            maker_entry_offset_bps=maker_entry_offset_bps,
            tuning=tune,
        )
        obj = objective_from_stats(stats)
        if obj > best_obj:
            best_obj = obj
            best_stats = stats
            best_tune = tune

    assert best_stats is not None and best_tune is not None
    return best_stats, best_tune, best_obj


def main() -> None:
    ap = argparse.ArgumentParser(description="Playbook microstructure backtest with maker/taker, TTL, Kelly, path-dependent exits")
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--leverage", type=float, default=10.0)
    ap.add_argument("--fee-taker-bps", type=float, default=5.0)
    ap.add_argument("--fee-maker-bps", type=float, default=-1.0)
    ap.add_argument("--slip-taker-bps", type=float, default=2.0)
    ap.add_argument("--slip-maker-bps", type=float, default=0.5)
    ap.add_argument("--maker-entry-offset-bps", type=float, default=15.0)
    ap.add_argument("--pb16-max-toxicity", type=float, default=0.50)
    ap.add_argument("--pb16-min-eai-delta", type=float, default=0.03)
    ap.add_argument("--pb16-min-abs-whale", type=float, default=0.18)
    ap.add_argument("--pb16-kelly-mult", type=float, default=0.70)
    ap.add_argument("--pb10-kelly-boost", type=float, default=1.35)
    ap.add_argument("--pb8-kelly-boost", type=float, default=1.25)
    ap.add_argument("--search-trials", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=str(OUT_PATH))
    args = ap.parse_args()

    df = build_frame(symbol=args.symbol)
    base_tuning = StrategyTuning(
        pb16_max_toxicity=float(args.pb16_max_toxicity),
        pb16_min_eai_delta=float(args.pb16_min_eai_delta),
        pb16_min_abs_whale=float(args.pb16_min_abs_whale),
        pb16_kelly_mult=float(args.pb16_kelly_mult),
        pb10_kelly_boost=float(args.pb10_kelly_boost),
        pb8_kelly_boost=float(args.pb8_kelly_boost),
    )

    if int(args.search_trials) > 0:
        stats, best_tune, obj = random_search(
            df=df,
            trials=int(args.search_trials),
            leverage=args.leverage,
            fee_taker_bps=args.fee_taker_bps,
            fee_maker_bps=args.fee_maker_bps,
            slip_taker_bps=args.slip_taker_bps,
            slip_maker_bps=args.slip_maker_bps,
            maker_entry_offset_bps=args.maker_entry_offset_bps,
            seed=int(args.seed),
        )
        tuning_used = asdict(best_tune)
        objective = obj
    else:
        stats = simulate(
            df=df,
            leverage=args.leverage,
            fee_taker_bps=args.fee_taker_bps,
            fee_maker_bps=args.fee_maker_bps,
            slip_taker_bps=args.slip_taker_bps,
            slip_maker_bps=args.slip_maker_bps,
            maker_entry_offset_bps=args.maker_entry_offset_bps,
            tuning=base_tuning,
        )
        tuning_used = asdict(base_tuning)
        objective = objective_from_stats(stats)

    out = asdict(stats)
    out["objective"] = float(objective)
    out["tuning"] = tuning_used
    out["output_path"] = str(args.output)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
