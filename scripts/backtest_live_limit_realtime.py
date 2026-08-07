#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
import urllib.request
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_JSON = "data/ensemble/reports/backtest_live_limit_realtime.json"
SYMBOL = "ETHUSDT"
TAKER_FEE = 0.0005
MAKER_FEE = 0.0002
TAKER_SLIP = 0.0002
ANNUAL_FACTOR_1M = math.sqrt(365 * 24 * 60)


@dataclass
class StrategyConfig:
    name: str
    entry_threshold: float
    exit_threshold: float
    pullback_bps: float
    queue_bonus_bps: float
    tox_penalty_bps: float
    max_hold_min: int
    stop_loss_pct: float
    take_profit_pct: float
    aftershock_cap: float
    toxicity_cap: float
    wait_max_min: int = 3
    release_threshold: float = 0.05


def _sharpe(eq_curve: list[float]) -> float:
    eq = np.array(eq_curve, dtype=np.float64)
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    if len(rets) < 3 or np.std(rets) < 1e-12:
        return 0.0
    return float(np.mean(rets) / np.std(rets) * ANNUAL_FACTOR_1M)


def _mdd(eq_curve: list[float]) -> float:
    eq = np.array(eq_curve, dtype=np.float64)
    run_max = np.maximum.accumulate(eq)
    dd = eq / np.maximum(run_max, 1e-12) - 1.0
    return float(np.min(dd)) * 100.0


def _parse_poly_center(label: str) -> float | None:
    nums = []
    for token in str(label).replace(">", "").replace("<", "").split("-"):
        token = token.strip().replace(",", "")
        try:
            nums.append(float(token))
        except Exception:
            pass
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def load_live_features() -> pd.DataFrame:
    con = duckdb.connect("data/live/microstructure.duckdb", read_only=True)
    ms = con.execute("select * from microstructure_1m order by ts").fetchdf()
    con.close()

    con = duckdb.connect("data/live/tail_risk.duckdb", read_only=True)
    tr = con.execute("select * from tail_risk_1m order by ts").fetchdf()
    con.close()

    con = duckdb.connect("data/live/polymarket.duckdb", read_only=True)
    poly = con.execute("select * from polymarket_markets_10s_json order by ts").fetchdf()
    con.close()

    def _poly_summary(markets_json: str) -> pd.Series:
        try:
            arr = json.loads(markets_json)
        except Exception:
            return pd.Series({"poly_exp": np.nan, "poly_top_prob": np.nan})
        centers = []
        probs = []
        for item in arr:
            prob = float(item.get("prob", 0.0) or 0.0)
            center = _parse_poly_center(str(item.get("label", "")))
            if center is None:
                continue
            centers.append(center)
            probs.append(prob)
        if not probs:
            return pd.Series({"poly_exp": np.nan, "poly_top_prob": np.nan})
        p = np.array(probs, dtype=np.float64)
        c = np.array(centers, dtype=np.float64)
        p_sum = float(p.sum())
        return pd.Series(
            {
                "poly_exp": float((p * c).sum() / max(p_sum, 1e-12)),
                "poly_top_prob": float(p.max()),
            }
        )

    poly = poly.join(poly["markets_json"].apply(_poly_summary))
    poly["ts"] = pd.to_datetime(poly["ts"]).dt.floor("min")
    poly_1m = poly.groupby("ts")[["poly_exp", "poly_top_prob"]].last().reset_index()

    feat = ms.merge(tr, on="ts", how="left").merge(poly_1m, on="ts", how="left")
    feat = feat.sort_values("ts").reset_index(drop=True)
    return feat


def fetch_1m_prices(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    start_utc = pd.Timestamp(start_ts).tz_convert("UTC")
    end_utc = pd.Timestamp(end_ts).tz_convert("UTC") + pd.Timedelta(minutes=1)
    cur = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)
    rows: list[list] = []
    while cur < end_ms:
        url = (
            "https://fapi.binance.com/fapi/v1/klines"
            f"?symbol={SYMBOL}&interval=1m&startTime={cur}&limit=1500"
        )
        raw = urllib.request.urlopen(url, timeout=20).read().decode("utf-8")
        arr = json.loads(raw)
        if not arr:
            break
        rows.extend(arr)
        nxt = int(arr[-1][0]) + 60_000
        if nxt <= cur:
            break
        cur = nxt
    px = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "tb_base",
            "tb_quote",
            "ignore",
        ],
    )
    px = px[["open_time", "open", "high", "low", "close"]].copy()
    px["ts"] = pd.to_datetime(px["open_time"], unit="ms", utc=True).dt.tz_convert("Asia/Seoul")
    for c in ("open", "high", "low", "close"):
        px[c] = pd.to_numeric(px[c], errors="coerce")
    px = px.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts").reset_index(drop=True)
    return px


def build_dataset() -> pd.DataFrame:
    feat = load_live_features()
    px = fetch_1m_prices(feat["ts"].min(), feat["ts"].max())
    df = feat.merge(px[["ts", "open", "high", "low", "close"]], on="ts", how="inner")
    df = df.sort_values("ts").reset_index(drop=True)
    df["poly_gap"] = (df["poly_exp"] - df["close"]) / df["close"]
    df["obi_norm"] = np.tanh(pd.to_numeric(df["obi"], errors="coerce").fillna(0.0))
    df["taker_edge"] = (pd.to_numeric(df["taker_buy_ratio"], errors="coerce").fillna(0.5) - 0.5) * 2.0
    df["whale_flow"] = np.tanh(pd.to_numeric(df["nif_whale"], errors="coerce").fillna(0.0) / 0.35)
    df["eai_norm"] = np.tanh(pd.to_numeric(df["eai"], errors="coerce").fillna(0.0))
    df["queue_penalty"] = pd.to_numeric(df["shadow_queue_collapse"], errors="coerce").fillna(0.0).clip(0.0, 1.5)
    df["toxicity"] = pd.to_numeric(df["shadow_toxicity_score"], errors="coerce").fillna(0.0).clip(0.0, 1.5)
    df["absorption"] = pd.to_numeric(df["shadow_absorption_score"], errors="coerce").fillna(0.0).clip(0.0, 1.5)
    df["aftershock"] = pd.to_numeric(df["shadow_aftershock_prob"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["kelly_mult"] = pd.to_numeric(df["kelly_mult"], errors="coerce").fillna(1.0).clip(0.5, 1.5)
    df["signal_bias"] = pd.to_numeric(df["signal_bias"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    df["ret_1m"] = df["close"].pct_change().fillna(0.0)
    df["ret_2m"] = df["close"].pct_change(2).fillna(0.0)
    df["ret_3m"] = df["close"].pct_change(3).fillna(0.0)
    df["d_whale_flow"] = df["whale_flow"].diff().fillna(0.0)
    df["d_obi_norm"] = df["obi_norm"].diff().fillna(0.0)
    df["d_taker_edge"] = df["taker_edge"].diff().fillna(0.0)
    df["d_queue_penalty"] = df["queue_penalty"].diff().fillna(0.0)
    df["d_toxicity"] = df["toxicity"].diff().fillna(0.0)
    df["d_poly_gap"] = df["poly_gap"].diff().fillna(0.0)
    state_cols = ["whale_flow", "obi_norm", "taker_edge", "poly_gap"]
    state = df[state_cols].fillna(0.0).to_numpy(dtype=np.float64)
    vel = np.zeros_like(state)
    vel[1:] = state[1:] - state[:-1]
    speed = np.linalg.norm(vel, axis=1)
    accel = np.zeros_like(state)
    accel[1:] = vel[1:] - vel[:-1]
    accel_norm = np.linalg.norm(accel, axis=1)
    cross_xy = vel[:, 0] * accel[:, 1] - vel[:, 1] * accel[:, 0]
    curvature = np.abs(cross_xy) / np.maximum(speed ** 3, 1e-8)
    circulation = state[:, 0] * vel[:, 1] - state[:, 1] * vel[:, 0]
    bullish_anchor = np.array([0.9, 0.7, 0.5, 0.004], dtype=np.float64)
    bearish_anchor = -bullish_anchor
    bull_potential = np.sum((state - bullish_anchor) ** 2, axis=1)
    bear_potential = np.sum((state - bearish_anchor) ** 2, axis=1)
    field = np.column_stack(
        [
            0.42 * state[:, 0] + 0.18 * vel[:, 0],
            0.28 * state[:, 1] + 0.15 * vel[:, 1],
            0.18 * state[:, 2] + 0.10 * vel[:, 2],
            18.0 * state[:, 3] + 4.0 * vel[:, 3],
        ]
    )
    bull_flux = np.sum(field * bullish_anchor, axis=1)
    bear_flux = -np.sum(field * bullish_anchor, axis=1)
    df["state_speed"] = speed
    df["state_accel"] = accel_norm
    df["state_curvature"] = curvature
    df["state_circulation"] = circulation
    df["bull_potential"] = bull_potential
    df["bear_potential"] = bear_potential
    df["bull_flux"] = bull_flux
    df["bear_flux"] = bear_flux
    df["bull_line_int_5"] = pd.Series(bull_flux).rolling(5, min_periods=1).sum().to_numpy()
    df["bear_line_int_5"] = pd.Series(bear_flux).rolling(5, min_periods=1).sum().to_numpy()
    df["bull_line_int_9"] = pd.Series(bull_flux).rolling(9, min_periods=1).sum().to_numpy()
    df["bear_line_int_9"] = pd.Series(bear_flux).rolling(9, min_periods=1).sum().to_numpy()
    df["curvature_mean_5"] = pd.Series(curvature).rolling(5, min_periods=1).mean().to_numpy()
    df["speed_mean_5"] = pd.Series(speed).rolling(5, min_periods=1).mean().to_numpy()
    df["potential_gap"] = df["bear_potential"] - df["bull_potential"]
    df["score_long"] = (
        0.28 * df["whale_flow"]
        + 0.20 * df["obi_norm"]
        + 0.16 * df["taker_edge"]
        + 0.14 * np.tanh(df["poly_gap"] / 0.004)
        + 0.08 * df["eai_norm"]
        + 0.06 * (df["kelly_mult"] - 1.0)
        + 0.08 * df["signal_bias"]
        - 0.10 * df["queue_penalty"]
        - 0.08 * df["toxicity"]
        - 0.05 * df["aftershock"]
        + 0.05 * df["absorption"]
    )
    df["score_short"] = (
        -0.28 * df["whale_flow"]
        - 0.20 * df["obi_norm"]
        - 0.16 * df["taker_edge"]
        - 0.14 * np.tanh(df["poly_gap"] / 0.004)
        - 0.08 * df["eai_norm"]
        + 0.06 * (df["kelly_mult"] - 1.0)
        - 0.08 * df["signal_bias"]
        - 0.10 * df["queue_penalty"]
        - 0.08 * df["toxicity"]
        - 0.05 * df["aftershock"]
        + 0.05 * df["absorption"]
    )
    df["wait_long"] = (
        0.35 * np.tanh(-df["ret_1m"] / 0.0015)
        + 0.20 * np.tanh(-df["ret_2m"] / 0.0025)
        + 0.15 * np.tanh(-df["ret_3m"] / 0.0035)
        + 0.12 * np.tanh(df["queue_penalty"] / 0.8)
        + 0.10 * np.tanh(df["toxicity"] / 0.8)
        - 0.18 * np.tanh(df["whale_flow"] / 0.5)
        - 0.15 * np.tanh(df["obi_norm"] / 0.5)
        - 0.10 * np.tanh(df["taker_edge"] / 0.5)
        - 0.10 * np.tanh(df["poly_gap"] / 0.004)
    )
    df["wait_short"] = (
        0.35 * np.tanh(df["ret_1m"] / 0.0015)
        + 0.20 * np.tanh(df["ret_2m"] / 0.0025)
        + 0.15 * np.tanh(df["ret_3m"] / 0.0035)
        + 0.12 * np.tanh(df["queue_penalty"] / 0.8)
        + 0.10 * np.tanh(df["toxicity"] / 0.8)
        + 0.18 * np.tanh(df["whale_flow"] / 0.5)
        + 0.15 * np.tanh(df["obi_norm"] / 0.5)
        + 0.10 * np.tanh(df["taker_edge"] / 0.5)
        + 0.10 * np.tanh(df["poly_gap"] / 0.004)
    )
    return df


def _realized(side: str, entry_price: float, exit_price: float, lev: float, entry_fee: float, exit_fee: float) -> float:
    gross = (exit_price - entry_price) / entry_price if side == "LONG" else (entry_price - exit_price) / entry_price
    return float(gross * lev - (entry_fee + exit_fee) * lev)


def _unrealized(side: str | None, entry_price: float, current_price: float, lev: float, entry_fee: float) -> float:
    if side is None or entry_price <= 0.0 or lev <= 0.0:
        return 0.0
    gross = (current_price - entry_price) / entry_price if side == "LONG" else (entry_price - current_price) / entry_price
    return float(gross * lev - (entry_fee + MAKER_FEE) * lev)


def simulate_market(df: pd.DataFrame, cfg: StrategyConfig) -> dict:
    balance = 1.0
    eq_curve = [1.0]
    pos: str | None = None
    entry_price = 0.0
    entry_fee = 0.0
    entry_idx = -1
    trades = wins = 0

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_open = float(df.iloc[i + 1]["open"])
        next_close = float(df.iloc[i + 1]["close"])
        long_score = float(row["score_long"])
        short_score = float(row["score_short"])
        lev = float(np.clip(0.08 + 0.22 * max(abs(long_score), abs(short_score)), 0.08, 0.30))

        if pos is None:
            if row["aftershock"] <= cfg.aftershock_cap and row["toxicity"] <= cfg.toxicity_cap:
                if long_score >= cfg.entry_threshold and long_score > short_score:
                    pos = "LONG"
                    entry_price = next_open * (1.0 + TAKER_SLIP)
                    entry_fee = TAKER_FEE
                    entry_idx = i + 1
                    balance -= balance * TAKER_FEE * lev
                    cur_lev = lev
                elif short_score >= cfg.entry_threshold and short_score > long_score:
                    pos = "SHORT"
                    entry_price = next_open * (1.0 - TAKER_SLIP)
                    entry_fee = TAKER_FEE
                    entry_idx = i + 1
                    balance -= balance * TAKER_FEE * lev
                    cur_lev = lev
        else:
            hold = i - entry_idx
            live_ret = _unrealized(pos, entry_price, float(row["close"]), cur_lev, entry_fee)
            exit_cond = False
            if pos == "LONG":
                exit_cond = short_score >= cfg.exit_threshold or live_ret <= -cfg.stop_loss_pct or live_ret >= cfg.take_profit_pct
            else:
                exit_cond = long_score >= cfg.exit_threshold or live_ret <= -cfg.stop_loss_pct or live_ret >= cfg.take_profit_pct
            exit_cond = exit_cond or hold >= cfg.max_hold_min or row["aftershock"] > cfg.aftershock_cap + 0.12
            if exit_cond:
                exit_price = next_open * (1.0 - TAKER_SLIP) if pos == "LONG" else next_open * (1.0 + TAKER_SLIP)
                realized = _realized(pos, entry_price, exit_price, cur_lev, entry_fee, TAKER_FEE)
                balance *= 1.0 + realized
                trades += 1
                wins += int(realized > 0.0)
                pos = None
                entry_price = 0.0
                entry_idx = -1
                entry_fee = 0.0

        eq_curve.append(max(balance * (1.0 + (_unrealized(pos, entry_price, next_close, cur_lev, entry_fee) if pos else 0.0)), 1e-8))

    return {
        "mode": "market",
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "trades": trades,
        "wr_pct": round((100.0 * wins / trades) if trades else 0.0, 2),
        "sharpe": round(_sharpe(eq_curve), 4),
        "mdd_pct": round(_mdd(eq_curve), 4),
    }


def simulate_limit(df: pd.DataFrame, cfg: StrategyConfig) -> dict:
    balance = 1.0
    eq_curve = [1.0]
    pos: str | None = None
    entry_price = 0.0
    entry_fee = 0.0
    entry_idx = -1
    cur_lev = 0.0
    trades = wins = 0
    maker_entries = fallback_entries = missed_entries = 0
    pending: dict | None = None

    for i in range(1, len(df) - 1):
        bar = df.iloc[i]
        prev = df.iloc[i - 1]
        bar_open = float(bar["open"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])

        if pending is not None:
            fill = (pending["side"] == "LONG" and bar_low <= pending["price"]) or (pending["side"] == "SHORT" and bar_high >= pending["price"])
            if fill:
                pos = pending["side"]
                entry_price = float(pending["price"])
                entry_fee = MAKER_FEE
                entry_idx = i
                cur_lev = float(pending["lev"])
                balance -= balance * MAKER_FEE * cur_lev
                maker_entries += 1
                pending = None
            elif i > pending["expire_idx"]:
                if pending["fallback"]:
                    pos = pending["side"]
                    entry_price = bar_open * (1.0 + TAKER_SLIP) if pos == "LONG" else bar_open * (1.0 - TAKER_SLIP)
                    entry_fee = TAKER_FEE
                    entry_idx = i
                    cur_lev = float(pending["lev"])
                    balance -= balance * TAKER_FEE * cur_lev
                    fallback_entries += 1
                else:
                    missed_entries += 1
                pending = None

        if pending is None and pos is None:
            long_score = float(prev["score_long"])
            short_score = float(prev["score_short"])
            if prev["aftershock"] <= cfg.aftershock_cap and prev["toxicity"] <= cfg.toxicity_cap:
                if long_score >= cfg.entry_threshold and long_score > short_score:
                    lev = float(np.clip(0.08 + 0.22 * long_score, 0.08, 0.30))
                    pullback = cfg.pullback_bps / 10000.0
                    pullback += max(float(prev["queue_penalty"]), 0.0) * cfg.queue_bonus_bps / 10000.0
                    pullback += max(float(prev["toxicity"]), 0.0) * cfg.tox_penalty_bps / 10000.0
                    pullback -= max(float(prev["whale_flow"]), 0.0) * 0.00035
                    pullback -= max(float(prev["poly_gap"]), 0.0) * 0.18
                    pullback = float(np.clip(pullback, 0.0001, 0.0030))
                    pending = {
                        "side": "LONG",
                        "price": float(prev["close"]) * (1.0 - pullback),
                        "expire_idx": i + 1,
                        "lev": lev,
                        "fallback": bool(long_score >= cfg.entry_threshold + 0.10 and prev["toxicity"] < cfg.toxicity_cap * 0.85),
                    }
                elif short_score >= cfg.entry_threshold and short_score > long_score:
                    lev = float(np.clip(0.08 + 0.22 * short_score, 0.08, 0.30))
                    pullback = cfg.pullback_bps / 10000.0
                    pullback += max(float(prev["queue_penalty"]), 0.0) * cfg.queue_bonus_bps / 10000.0
                    pullback += max(float(prev["toxicity"]), 0.0) * cfg.tox_penalty_bps / 10000.0
                    pullback -= max(float(-prev["whale_flow"]), 0.0) * 0.00035
                    pullback -= max(float(-prev["poly_gap"]), 0.0) * 0.18
                    pullback = float(np.clip(pullback, 0.0001, 0.0030))
                    pending = {
                        "side": "SHORT",
                        "price": float(prev["close"]) * (1.0 + pullback),
                        "expire_idx": i + 1,
                        "lev": lev,
                        "fallback": bool(short_score >= cfg.entry_threshold + 0.10 and prev["toxicity"] < cfg.toxicity_cap * 0.85),
                    }

        if pos is not None:
            hold = i - entry_idx
            live_ret = _unrealized(pos, entry_price, float(bar["close"]), cur_lev, entry_fee)
            long_score = float(bar["score_long"])
            short_score = float(bar["score_short"])
            exit_cond = False
            if pos == "LONG":
                exit_cond = short_score >= cfg.exit_threshold or live_ret <= -cfg.stop_loss_pct or live_ret >= cfg.take_profit_pct
            else:
                exit_cond = long_score >= cfg.exit_threshold or live_ret <= -cfg.stop_loss_pct or live_ret >= cfg.take_profit_pct
            exit_cond = exit_cond or hold >= cfg.max_hold_min or float(bar["aftershock"]) > cfg.aftershock_cap + 0.12 or float(bar["toxicity"]) > cfg.toxicity_cap + 0.20
            if exit_cond:
                exit_price = float(df.iloc[i + 1]["open"]) * (1.0 - TAKER_SLIP) if pos == "LONG" else float(df.iloc[i + 1]["open"]) * (1.0 + TAKER_SLIP)
                realized = _realized(pos, entry_price, exit_price, cur_lev, entry_fee, TAKER_FEE)
                balance *= 1.0 + realized
                trades += 1
                wins += int(realized > 0.0)
                pos = None
                entry_price = 0.0
                entry_idx = -1
                entry_fee = 0.0
                cur_lev = 0.0

        eq_curve.append(max(balance * (1.0 + (_unrealized(pos, entry_price, bar_close, cur_lev, entry_fee) if pos else 0.0)), 1e-8))

    return {
        "mode": "limit",
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "trades": trades,
        "wr_pct": round((100.0 * wins / trades) if trades else 0.0, 2),
        "sharpe": round(_sharpe(eq_curve), 4),
        "mdd_pct": round(_mdd(eq_curve), 4),
        "maker_entry_ratio": round(maker_entries / max(maker_entries + fallback_entries, 1), 4),
        "missed_entries": missed_entries,
        "maker_entries": maker_entries,
        "fallback_entries": fallback_entries,
    }


def simulate_wait_limit(df: pd.DataFrame, cfg: StrategyConfig) -> dict:
    balance = 1.0
    eq_curve = [1.0]
    pos: str | None = None
    entry_price = 0.0
    entry_fee = 0.0
    entry_idx = -1
    cur_lev = 0.0
    trades = wins = 0
    maker_entries = fallback_entries = missed_entries = wait_releases = wait_cancels = 0
    pending: dict | None = None
    waiting: dict | None = None

    for i in range(1, len(df) - 1):
        bar = df.iloc[i]
        prev = df.iloc[i - 1]
        bar_open = float(bar["open"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])

        if waiting is not None:
            side = str(waiting["side"])
            score = float(prev["score_long"] if side == "LONG" else prev["score_short"])
            opp_score = float(prev["score_short"] if side == "LONG" else prev["score_long"])
            wait_score = float(prev["wait_long"] if side == "LONG" else prev["wait_short"])
            better_flow = float(prev["whale_flow"] if side == "LONG" else -prev["whale_flow"])
            better_obi = float(prev["obi_norm"] if side == "LONG" else -prev["obi_norm"])
            better_taker = float(prev["taker_edge"] if side == "LONG" else -prev["taker_edge"])
            release_score = (
                0.45 * score
                - 0.35 * wait_score
                + 0.10 * better_flow
                + 0.05 * better_obi
                + 0.05 * better_taker
            )
            expired = i > int(waiting["expire_idx"])
            invalid = (
                score < cfg.entry_threshold - 0.03
                or opp_score > score
                or float(prev["aftershock"]) > cfg.aftershock_cap + 0.10
                or float(prev["toxicity"]) > cfg.toxicity_cap + 0.15
            )
            released = release_score >= cfg.release_threshold
            if invalid or expired:
                wait_cancels += 1
                waiting = None
            elif released:
                lev = float(waiting["lev"])
                pullback = cfg.pullback_bps / 10000.0
                pullback += max(float(prev["queue_penalty"]), 0.0) * cfg.queue_bonus_bps / 10000.0
                pullback += max(float(prev["toxicity"]), 0.0) * cfg.tox_penalty_bps / 10000.0
                if side == "LONG":
                    pullback -= max(float(prev["whale_flow"]), 0.0) * 0.00035
                    pullback -= max(float(prev["poly_gap"]), 0.0) * 0.18
                    limit_price = float(prev["close"]) * (1.0 - float(np.clip(pullback, 0.0001, 0.0030)))
                    fallback = bool(score >= cfg.entry_threshold + 0.10 and prev["toxicity"] < cfg.toxicity_cap * 0.85)
                else:
                    pullback -= max(float(-prev["whale_flow"]), 0.0) * 0.00035
                    pullback -= max(float(-prev["poly_gap"]), 0.0) * 0.18
                    limit_price = float(prev["close"]) * (1.0 + float(np.clip(pullback, 0.0001, 0.0030)))
                    fallback = bool(score >= cfg.entry_threshold + 0.10 and prev["toxicity"] < cfg.toxicity_cap * 0.85)
                pending = {
                    "side": side,
                    "price": limit_price,
                    "expire_idx": i + 1,
                    "lev": lev,
                    "fallback": fallback,
                }
                wait_releases += 1
                waiting = None

        if pending is not None:
            fill = (pending["side"] == "LONG" and bar_low <= pending["price"]) or (pending["side"] == "SHORT" and bar_high >= pending["price"])
            if fill:
                pos = pending["side"]
                entry_price = float(pending["price"])
                entry_fee = MAKER_FEE
                entry_idx = i
                cur_lev = float(pending["lev"])
                balance -= balance * MAKER_FEE * cur_lev
                maker_entries += 1
                pending = None
            elif i > pending["expire_idx"]:
                if pending["fallback"]:
                    pos = pending["side"]
                    entry_price = bar_open * (1.0 + TAKER_SLIP) if pos == "LONG" else bar_open * (1.0 - TAKER_SLIP)
                    entry_fee = TAKER_FEE
                    entry_idx = i
                    cur_lev = float(pending["lev"])
                    balance -= balance * TAKER_FEE * cur_lev
                    fallback_entries += 1
                else:
                    missed_entries += 1
                pending = None

        if pos is None and pending is None and waiting is None:
            long_score = float(prev["score_long"])
            short_score = float(prev["score_short"])
            if prev["aftershock"] <= cfg.aftershock_cap and prev["toxicity"] <= cfg.toxicity_cap:
                if long_score >= cfg.entry_threshold and long_score > short_score:
                    lev = float(np.clip(0.08 + 0.22 * long_score, 0.08, 0.30))
                    wait_score = float(prev["wait_long"])
                    if wait_score > cfg.release_threshold:
                        waiting = {
                            "side": "LONG",
                            "start_idx": i,
                            "expire_idx": i + cfg.wait_max_min,
                            "lev": lev,
                        }
                    else:
                        pullback = cfg.pullback_bps / 10000.0
                        pullback += max(float(prev["queue_penalty"]), 0.0) * cfg.queue_bonus_bps / 10000.0
                        pullback += max(float(prev["toxicity"]), 0.0) * cfg.tox_penalty_bps / 10000.0
                        pullback -= max(float(prev["whale_flow"]), 0.0) * 0.00035
                        pullback -= max(float(prev["poly_gap"]), 0.0) * 0.18
                        pullback = float(np.clip(pullback, 0.0001, 0.0030))
                        pending = {
                            "side": "LONG",
                            "price": float(prev["close"]) * (1.0 - pullback),
                            "expire_idx": i + 1,
                            "lev": lev,
                            "fallback": bool(long_score >= cfg.entry_threshold + 0.10 and prev["toxicity"] < cfg.toxicity_cap * 0.85),
                        }
                elif short_score >= cfg.entry_threshold and short_score > long_score:
                    lev = float(np.clip(0.08 + 0.22 * short_score, 0.08, 0.30))
                    wait_score = float(prev["wait_short"])
                    if wait_score > cfg.release_threshold:
                        waiting = {
                            "side": "SHORT",
                            "start_idx": i,
                            "expire_idx": i + cfg.wait_max_min,
                            "lev": lev,
                        }
                    else:
                        pullback = cfg.pullback_bps / 10000.0
                        pullback += max(float(prev["queue_penalty"]), 0.0) * cfg.queue_bonus_bps / 10000.0
                        pullback += max(float(prev["toxicity"]), 0.0) * cfg.tox_penalty_bps / 10000.0
                        pullback -= max(float(-prev["whale_flow"]), 0.0) * 0.00035
                        pullback -= max(float(-prev["poly_gap"]), 0.0) * 0.18
                        pullback = float(np.clip(pullback, 0.0001, 0.0030))
                        pending = {
                            "side": "SHORT",
                            "price": float(prev["close"]) * (1.0 + pullback),
                            "expire_idx": i + 1,
                            "lev": lev,
                            "fallback": bool(short_score >= cfg.entry_threshold + 0.10 and prev["toxicity"] < cfg.toxicity_cap * 0.85),
                        }

        if pos is not None:
            hold = i - entry_idx
            live_ret = _unrealized(pos, entry_price, float(bar["close"]), cur_lev, entry_fee)
            long_score = float(bar["score_long"])
            short_score = float(bar["score_short"])
            exit_cond = False
            if pos == "LONG":
                exit_cond = short_score >= cfg.exit_threshold or live_ret <= -cfg.stop_loss_pct or live_ret >= cfg.take_profit_pct
            else:
                exit_cond = long_score >= cfg.exit_threshold or live_ret <= -cfg.stop_loss_pct or live_ret >= cfg.take_profit_pct
            exit_cond = exit_cond or hold >= cfg.max_hold_min or float(bar["aftershock"]) > cfg.aftershock_cap + 0.12 or float(bar["toxicity"]) > cfg.toxicity_cap + 0.20
            if exit_cond:
                exit_price = float(df.iloc[i + 1]["open"]) * (1.0 - TAKER_SLIP) if pos == "LONG" else float(df.iloc[i + 1]["open"]) * (1.0 + TAKER_SLIP)
                realized = _realized(pos, entry_price, exit_price, cur_lev, entry_fee, TAKER_FEE)
                balance *= 1.0 + realized
                trades += 1
                wins += int(realized > 0.0)
                pos = None
                entry_price = 0.0
                entry_idx = -1
                entry_fee = 0.0
                cur_lev = 0.0

        eq_curve.append(max(balance * (1.0 + (_unrealized(pos, entry_price, bar_close, cur_lev, entry_fee) if pos else 0.0)), 1e-8))

    return {
        "mode": "wait_limit",
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "trades": trades,
        "wr_pct": round((100.0 * wins / trades) if trades else 0.0, 2),
        "sharpe": round(_sharpe(eq_curve), 4),
        "mdd_pct": round(_mdd(eq_curve), 4),
        "maker_entry_ratio": round(maker_entries / max(maker_entries + fallback_entries, 1), 4),
        "missed_entries": missed_entries,
        "maker_entries": maker_entries,
        "fallback_entries": fallback_entries,
        "wait_releases": wait_releases,
        "wait_cancels": wait_cancels,
    }


def simulate_creative_limit(df: pd.DataFrame) -> dict:
    balance = 1.0
    eq_curve = [1.0]
    pos: str | None = None
    entry_price = 0.0
    entry_fee = 0.0
    entry_idx = -1
    cur_lev = 0.0
    trades = wins = 0
    maker_entries = fallback_entries = missed_entries = 0
    wait_releases = wait_cancels = 0
    pending: dict | None = None
    waiting: dict | None = None
    archetype_counts: dict[str, int] = {}
    active_setup = ""
    active_stop = 0.0
    active_tp = 0.0
    active_hold = 0

    def _pick_setup(row: pd.Series) -> dict | None:
        long_cont = (
            0.28 * float(row["whale_flow"])
            + 0.18 * float(row["obi_norm"])
            + 0.16 * float(row["taker_edge"])
            + 0.12 * np.tanh(float(row["poly_gap"]) / 0.004)
            + 0.10 * float(row["d_whale_flow"])
            + 0.06 * float(row["d_obi_norm"])
            - 0.10 * float(row["toxicity"])
            - 0.08 * float(row["queue_penalty"])
            - 0.06 * float(row["aftershock"])
        )
        long_absorb = (
            0.24 * float(row["absorption"])
            + 0.18 * np.tanh(max(-float(row["ret_1m"]), 0.0) / 0.0015)
            + 0.14 * np.tanh(max(-float(row["ret_2m"]), 0.0) / 0.0025)
            + 0.14 * max(float(row["d_whale_flow"]), 0.0)
            + 0.10 * max(float(row["d_obi_norm"]), 0.0)
            + 0.10 * np.tanh(float(row["poly_gap"]) / 0.004)
            + 0.06 * max(float(row["whale_flow"]), -0.15)
            - 0.14 * float(row["toxicity"])
            - 0.06 * float(row["aftershock"])
        )
        short_cont = (
            -0.28 * float(row["whale_flow"])
            - 0.18 * float(row["obi_norm"])
            - 0.16 * float(row["taker_edge"])
            - 0.12 * np.tanh(float(row["poly_gap"]) / 0.004)
            - 0.10 * float(row["d_whale_flow"])
            - 0.06 * float(row["d_obi_norm"])
            - 0.10 * float(row["toxicity"])
            - 0.08 * float(row["queue_penalty"])
            - 0.06 * float(row["aftershock"])
        )
        setups = [
            {
                "name": "LONG_CONT",
                "side": "LONG",
                "entry_score": float(long_cont),
                "wait_score": float(
                    0.32 * np.tanh(max(float(row["ret_1m"]), 0.0) / 0.0015)
                    + 0.22 * np.tanh(max(float(row["ret_2m"]), 0.0) / 0.0025)
                    + 0.16 * float(row["queue_penalty"])
                    + 0.12 * float(row["toxicity"])
                    - 0.18 * max(float(row["d_whale_flow"]), 0.0)
                ),
                "pullback_bps": 2.2,
                "max_wait": 2,
                "threshold": 0.22,
                "take_profit_pct": 0.0054,
                "stop_loss_pct": 0.0038,
                "max_hold": 12,
            },
            {
                "name": "LONG_ABSORB",
                "side": "LONG",
                "entry_score": float(long_absorb),
                "wait_score": float(
                    0.30 * np.tanh(max(-float(row["ret_1m"]), 0.0) / 0.0015)
                    + 0.20 * np.tanh(max(-float(row["ret_2m"]), 0.0) / 0.0025)
                    + 0.18 * max(-float(row["d_whale_flow"]), 0.0)
                    + 0.12 * max(-float(row["d_obi_norm"]), 0.0)
                    - 0.20 * max(float(row["d_whale_flow"]), 0.0)
                    - 0.10 * max(float(row["d_obi_norm"]), 0.0)
                ),
                "pullback_bps": 4.8,
                "max_wait": 4,
                "threshold": 0.24,
                "take_profit_pct": 0.0062,
                "stop_loss_pct": 0.0035,
                "max_hold": 16,
            },
            {
                "name": "SHORT_CONT",
                "side": "SHORT",
                "entry_score": float(short_cont),
                "wait_score": float(
                    0.32 * np.tanh(max(-float(row["ret_1m"]), 0.0) / 0.0015)
                    + 0.22 * np.tanh(max(-float(row["ret_2m"]), 0.0) / 0.0025)
                    + 0.16 * float(row["queue_penalty"])
                    + 0.12 * float(row["toxicity"])
                    + 0.18 * min(float(row["d_whale_flow"]), 0.0) * -1.0
                ),
                "pullback_bps": 2.6,
                "max_wait": 2,
                "threshold": 0.28,
                "take_profit_pct": 0.0056,
                "stop_loss_pct": 0.0038,
                "max_hold": 10,
            },
        ]
        best = max(setups, key=lambda x: x["entry_score"])
        if best["entry_score"] < best["threshold"]:
            return None
        if float(row["aftershock"]) > 0.55 or float(row["toxicity"]) > 0.92:
            return None
        if best["side"] == "SHORT" and float(row["poly_gap"]) > 0.0015:
            return None
        return best

    for i in range(1, len(df) - 1):
        bar = df.iloc[i]
        prev = df.iloc[i - 1]
        bar_open = float(bar["open"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])

        if waiting is not None:
            side = str(waiting["side"])
            setup_name = str(waiting["setup_name"])
            score = float(waiting["entry_score_fn"](prev))
            wait_score = float(waiting["wait_score_fn"](prev))
            improving = False
            if side == "LONG":
                improving = (
                    float(prev["d_whale_flow"]) > -0.02
                    and float(prev["d_obi_norm"]) > -0.02
                    and float(prev["ret_1m"]) > -0.0012
                )
            else:
                improving = (
                    float(prev["d_whale_flow"]) < 0.02
                    and float(prev["d_obi_norm"]) < 0.02
                    and float(prev["ret_1m"]) < 0.0012
                )
            released = score >= float(waiting["threshold"]) and (wait_score <= float(waiting["release_threshold"]) or improving)
            invalid = score < float(waiting["threshold"]) - 0.03 or float(prev["aftershock"]) > 0.62 or float(prev["toxicity"]) > 1.00
            expired = i > int(waiting["expire_idx"])
            if invalid or expired:
                wait_cancels += 1
                waiting = None
            elif released:
                pullback = float(waiting["pullback_bps"]) / 10000.0
                pullback += max(float(prev["queue_penalty"]), 0.0) * 0.00018
                pullback += max(float(prev["toxicity"]), 0.0) * 0.00008
                if side == "LONG":
                    pullback -= max(float(prev["d_whale_flow"]), 0.0) * 0.0004
                    pullback -= max(float(prev["poly_gap"]), 0.0) * 0.18
                    limit_price = float(prev["close"]) * (1.0 - float(np.clip(pullback, 0.0001, 0.0035)))
                else:
                    pullback -= max(float(-prev["d_whale_flow"]), 0.0) * 0.0004
                    pullback -= max(float(-prev["poly_gap"]), 0.0) * 0.18
                    limit_price = float(prev["close"]) * (1.0 + float(np.clip(pullback, 0.0001, 0.0035)))
                pending = {
                    "side": side,
                    "price": limit_price,
                    "expire_idx": i + 1,
                    "lev": float(waiting["lev"]),
                    "fallback": bool(score >= float(waiting["threshold"]) + 0.10 and float(prev["toxicity"]) < 0.80),
                    "setup_name": setup_name,
                    "stop_loss_pct": float(waiting["stop_loss_pct"]),
                    "take_profit_pct": float(waiting["take_profit_pct"]),
                    "max_hold": int(waiting["max_hold"]),
                }
                wait_releases += 1
                waiting = None

        if pending is not None:
            fill = (pending["side"] == "LONG" and bar_low <= pending["price"]) or (pending["side"] == "SHORT" and bar_high >= pending["price"])
            if fill:
                pos = pending["side"]
                entry_price = float(pending["price"])
                entry_fee = MAKER_FEE
                entry_idx = i
                cur_lev = float(pending["lev"])
                active_setup = str(pending["setup_name"])
                active_stop = float(pending["stop_loss_pct"])
                active_tp = float(pending["take_profit_pct"])
                active_hold = int(pending["max_hold"])
                balance -= balance * MAKER_FEE * cur_lev
                maker_entries += 1
                pending = None
            elif i > pending["expire_idx"]:
                if pending["fallback"]:
                    pos = pending["side"]
                    entry_price = bar_open * (1.0 + TAKER_SLIP) if pos == "LONG" else bar_open * (1.0 - TAKER_SLIP)
                    entry_fee = TAKER_FEE
                    entry_idx = i
                    cur_lev = float(pending["lev"])
                    active_setup = str(pending["setup_name"])
                    active_stop = float(pending["stop_loss_pct"])
                    active_tp = float(pending["take_profit_pct"])
                    active_hold = int(pending["max_hold"])
                    balance -= balance * TAKER_FEE * cur_lev
                    fallback_entries += 1
                else:
                    missed_entries += 1
                pending = None

        if pos is None and pending is None and waiting is None:
            setup = _pick_setup(prev)
            if setup is not None:
                side = str(setup["side"])
                lev = float(np.clip(0.08 + 0.24 * float(setup["entry_score"]), 0.08, 0.28))
                archetype_counts[setup["name"]] = archetype_counts.get(setup["name"], 0) + 1
                if float(setup["wait_score"]) > 0.06:
                    waiting = {
                        "side": side,
                        "setup_name": str(setup["name"]),
                        "expire_idx": i + int(setup["max_wait"]),
                        "lev": lev,
                        "pullback_bps": float(setup["pullback_bps"]),
                        "threshold": float(setup["threshold"]),
                        "release_threshold": 0.03 if side == "LONG" else 0.05,
                        "stop_loss_pct": float(setup["stop_loss_pct"]),
                        "take_profit_pct": float(setup["take_profit_pct"]),
                        "max_hold": int(setup["max_hold"]),
                        "entry_score_fn": (lambda r, n=setup["name"]: _pick_setup(r)["entry_score"] if _pick_setup(r) and _pick_setup(r)["name"] == n else -1.0),
                        "wait_score_fn": (lambda r, s=side: float(r["wait_long"]) if s == "LONG" else float(r["wait_short"])),
                    }
                else:
                    pullback = float(setup["pullback_bps"]) / 10000.0
                    pullback += max(float(prev["queue_penalty"]), 0.0) * 0.00018
                    pullback += max(float(prev["toxicity"]), 0.0) * 0.00008
                    if side == "LONG":
                        pullback -= max(float(prev["d_whale_flow"]), 0.0) * 0.0004
                        pullback -= max(float(prev["poly_gap"]), 0.0) * 0.18
                        limit_price = float(prev["close"]) * (1.0 - float(np.clip(pullback, 0.0001, 0.0035)))
                    else:
                        pullback -= max(float(-prev["d_whale_flow"]), 0.0) * 0.0004
                        pullback -= max(float(-prev["poly_gap"]), 0.0) * 0.18
                        limit_price = float(prev["close"]) * (1.0 + float(np.clip(pullback, 0.0001, 0.0035)))
                    pending = {
                        "side": side,
                        "price": limit_price,
                        "expire_idx": i + 1,
                        "lev": lev,
                        "fallback": bool(float(setup["entry_score"]) >= float(setup["threshold"]) + 0.10 and float(prev["toxicity"]) < 0.80),
                        "setup_name": str(setup["name"]),
                        "stop_loss_pct": float(setup["stop_loss_pct"]),
                        "take_profit_pct": float(setup["take_profit_pct"]),
                        "max_hold": int(setup["max_hold"]),
                    }

        if pos is not None:
            hold = i - entry_idx
            live_ret = _unrealized(pos, entry_price, float(bar["close"]), cur_lev, entry_fee)
            setup_now = _pick_setup(bar)
            exit_cond = live_ret <= -active_stop or live_ret >= active_tp or hold >= active_hold
            exit_cond = exit_cond or float(bar["aftershock"]) > 0.62 or float(bar["toxicity"]) > 1.02
            if pos == "LONG":
                exit_cond = exit_cond or float(bar["score_short"]) > 0.20
                if active_setup == "LONG_CONT":
                    exit_cond = exit_cond or float(bar["d_whale_flow"]) < -0.10 or float(bar["poly_gap"]) < -0.0020
                else:
                    exit_cond = exit_cond or float(bar["ret_1m"]) > 0.0025 or (setup_now is not None and setup_now["name"] == "SHORT_CONT")
            else:
                exit_cond = exit_cond or float(bar["score_long"]) > 0.22
                exit_cond = exit_cond or float(bar["d_whale_flow"]) > 0.08 or float(bar["poly_gap"]) > 0.0020
            if exit_cond:
                exit_price = float(df.iloc[i + 1]["open"]) * (1.0 - TAKER_SLIP) if pos == "LONG" else float(df.iloc[i + 1]["open"]) * (1.0 + TAKER_SLIP)
                realized = _realized(pos, entry_price, exit_price, cur_lev, entry_fee, TAKER_FEE)
                balance *= 1.0 + realized
                trades += 1
                wins += int(realized > 0.0)
                pos = None
                entry_price = 0.0
                entry_idx = -1
                entry_fee = 0.0
                cur_lev = 0.0
                active_setup = ""
                active_stop = 0.0
                active_tp = 0.0
                active_hold = 0

        eq_curve.append(max(balance * (1.0 + (_unrealized(pos, entry_price, bar_close, cur_lev, entry_fee) if pos else 0.0)), 1e-8))

    return {
        "mode": "creative_limit",
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "trades": trades,
        "wr_pct": round((100.0 * wins / trades) if trades else 0.0, 2),
        "sharpe": round(_sharpe(eq_curve), 4),
        "mdd_pct": round(_mdd(eq_curve), 4),
        "maker_entry_ratio": round(maker_entries / max(maker_entries + fallback_entries, 1), 4),
        "missed_entries": missed_entries,
        "maker_entries": maker_entries,
        "fallback_entries": fallback_entries,
        "wait_releases": wait_releases,
        "wait_cancels": wait_cancels,
        "archetypes": archetype_counts,
    }


def simulate_geometric_limit(df: pd.DataFrame) -> dict:
    balance = 1.0
    eq_curve = [1.0]
    pos: str | None = None
    entry_price = 0.0
    entry_fee = 0.0
    entry_idx = -1
    cur_lev = 0.0
    active_stop = 0.0
    active_tp = 0.0
    active_hold = 0
    trades = wins = 0
    maker_entries = fallback_entries = missed_entries = 0
    wait_releases = wait_cancels = 0
    pending: dict | None = None
    waiting: dict | None = None

    def _geo_setup(row: pd.Series) -> dict | None:
        curve = float(np.tanh(float(row["state_curvature"]) * 0.08))
        speed = float(np.tanh(float(row["state_speed"]) / 0.22))
        accel = float(np.tanh(float(row["state_accel"]) / 0.18))
        curv_mean = float(np.tanh(float(row["curvature_mean_5"]) * 0.08))
        bull_energy = float(np.tanh(float(row["bull_line_int_5"]) / 1.2))
        bull_energy_slow = float(np.tanh(float(row["bull_line_int_9"]) / 2.0))
        bear_energy = float(np.tanh(float(row["bear_line_int_5"]) / 1.2))
        bear_energy_slow = float(np.tanh(float(row["bear_line_int_9"]) / 2.0))
        potential_bias = float(np.tanh(float(row["potential_gap"]) / 1.4))
        rotation = float(np.tanh(float(row["state_circulation"]) / 0.06))
        dissipation = float(np.tanh((float(row["toxicity"]) + float(row["queue_penalty"])) / 1.2))
        aftershock = float(row["aftershock"])

        long_geodesic = (
            0.30 * bull_energy
            + 0.18 * bull_energy_slow
            + 0.16 * potential_bias
            + 0.10 * max(float(row["d_whale_flow"]), -0.2)
            + 0.08 * max(float(row["d_obi_norm"]), -0.2)
            + 0.06 * max(float(row["d_taker_edge"]), -0.2)
            + 0.06 * rotation
            - 0.15 * curve
            - 0.10 * curv_mean
            - 0.10 * dissipation
            - 0.07 * aftershock
        )
        long_reversal = (
            0.24 * max(-float(row["ret_1m"]), 0.0) / 0.002
            + 0.14 * max(-float(row["ret_2m"]), 0.0) / 0.003
            + 0.16 * max(float(row["d_whale_flow"]), 0.0)
            + 0.12 * max(float(row["d_obi_norm"]), 0.0)
            + 0.10 * bull_energy
            + 0.08 * float(row["absorption"])
            + 0.08 * potential_bias
            - 0.14 * dissipation
            - 0.10 * accel
        )
        short_geodesic = (
            0.30 * bear_energy
            + 0.18 * bear_energy_slow
            - 0.16 * potential_bias
            + 0.10 * max(-float(row["d_whale_flow"]), -0.2)
            + 0.08 * max(-float(row["d_obi_norm"]), -0.2)
            + 0.06 * max(-float(row["d_taker_edge"]), -0.2)
            - 0.06 * rotation
            - 0.15 * curve
            - 0.10 * curv_mean
            - 0.10 * dissipation
            - 0.07 * aftershock
        )
        setups = [
            {
                "name": "GEO_LONG_PATH",
                "side": "LONG",
                "entry_score": float(long_geodesic),
                "wait_score": float(
                    0.28 * curve
                    + 0.18 * curv_mean
                    + 0.18 * dissipation
                    + 0.12 * max(float(row["ret_1m"]), 0.0) / 0.0018
                    - 0.14 * max(float(row["d_whale_flow"]), 0.0)
                    - 0.10 * max(float(row["d_obi_norm"]), 0.0)
                ),
                "pullback_bps": 2.4,
                "max_wait": 2,
                "threshold": 0.24,
                "tp": 0.0056,
                "sl": 0.0038,
                "hold": 12,
            },
            {
                "name": "GEO_LONG_CURVE",
                "side": "LONG",
                "entry_score": float(long_reversal),
                "wait_score": float(
                    0.22 * max(-float(row["ret_1m"]), 0.0) / 0.002
                    + 0.20 * max(-float(row["ret_2m"]), 0.0) / 0.003
                    + 0.16 * max(-float(row["d_whale_flow"]), 0.0)
                    + 0.10 * curve
                    - 0.22 * max(float(row["d_whale_flow"]), 0.0)
                    - 0.14 * max(float(row["d_obi_norm"]), 0.0)
                ),
                "pullback_bps": 5.2,
                "max_wait": 4,
                "threshold": 0.22,
                "tp": 0.0064,
                "sl": 0.0036,
                "hold": 18,
            },
            {
                "name": "GEO_SHORT_PATH",
                "side": "SHORT",
                "entry_score": float(short_geodesic),
                "wait_score": float(
                    0.28 * curve
                    + 0.18 * curv_mean
                    + 0.18 * dissipation
                    + 0.12 * max(-float(row["ret_1m"]), 0.0) / 0.0018
                    - 0.14 * max(-float(row["d_whale_flow"]), 0.0)
                    - 0.10 * max(-float(row["d_obi_norm"]), 0.0)
                ),
                "pullback_bps": 2.8,
                "max_wait": 2,
                "threshold": 0.32,
                "tp": 0.0052,
                "sl": 0.0038,
                "hold": 10,
            },
        ]
        best = max(setups, key=lambda x: x["entry_score"])
        if best["entry_score"] < best["threshold"]:
            return None
        if aftershock > 0.60 or dissipation > 0.78:
            return None
        if best["side"] == "SHORT" and potential_bias > 0.20:
            return None
        return best

    for i in range(1, len(df) - 1):
        bar = df.iloc[i]
        prev = df.iloc[i - 1]
        bar_open = float(bar["open"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])

        if waiting is not None:
            side = str(waiting["side"])
            wait_curve = float(np.tanh(float(prev["state_curvature"]) * 0.08))
            wait_diss = float(np.tanh((float(prev["toxicity"]) + float(prev["queue_penalty"])) / 1.2))
            if side == "LONG":
                release_force = (
                    0.30 * max(float(prev["d_whale_flow"]), 0.0)
                    + 0.22 * max(float(prev["d_obi_norm"]), 0.0)
                    + 0.14 * np.tanh(float(prev["bull_line_int_5"]) / 1.2)
                    + 0.10 * np.tanh(float(prev["potential_gap"]) / 1.4)
                    - 0.18 * wait_curve
                    - 0.10 * wait_diss
                )
            else:
                release_force = (
                    0.30 * max(-float(prev["d_whale_flow"]), 0.0)
                    + 0.22 * max(-float(prev["d_obi_norm"]), 0.0)
                    + 0.14 * np.tanh(float(prev["bear_line_int_5"]) / 1.2)
                    - 0.10 * np.tanh(float(prev["potential_gap"]) / 1.4)
                    - 0.18 * wait_curve
                    - 0.10 * wait_diss
                )
            expired = i > int(waiting["expire_idx"])
            invalid = float(prev["aftershock"]) > 0.68 or wait_diss > 0.86
            if invalid or expired:
                wait_cancels += 1
                waiting = None
            elif release_force >= float(waiting["release_threshold"]):
                pullback = float(waiting["pullback_bps"]) / 10000.0
                pullback += max(float(prev["state_curvature"]), 0.0) * 0.00002
                pullback += max(float(prev["toxicity"]), 0.0) * 0.00008
                pullback -= max(abs(float(prev["potential_gap"])) - 0.2, 0.0) * 0.00005
                if side == "LONG":
                    pullback -= max(float(prev["d_whale_flow"]), 0.0) * 0.00035
                    price = float(prev["close"]) * (1.0 - float(np.clip(pullback, 0.0001, 0.0032)))
                else:
                    pullback -= max(-float(prev["d_whale_flow"]), 0.0) * 0.00035
                    price = float(prev["close"]) * (1.0 + float(np.clip(pullback, 0.0001, 0.0032)))
                pending = {
                    "side": side,
                    "price": price,
                    "expire_idx": i + 1,
                    "lev": float(waiting["lev"]),
                    "fallback": bool(float(waiting["entry_score"]) >= float(waiting["threshold"]) + 0.10 and float(prev["toxicity"]) < 0.78),
                    "tp": float(waiting["tp"]),
                    "sl": float(waiting["sl"]),
                    "hold": int(waiting["hold"]),
                }
                wait_releases += 1
                waiting = None

        if pending is not None:
            fill = (pending["side"] == "LONG" and bar_low <= pending["price"]) or (pending["side"] == "SHORT" and bar_high >= pending["price"])
            if fill:
                pos = pending["side"]
                entry_price = float(pending["price"])
                entry_fee = MAKER_FEE
                entry_idx = i
                cur_lev = float(pending["lev"])
                active_tp = float(pending["tp"])
                active_stop = float(pending["sl"])
                active_hold = int(pending["hold"])
                balance -= balance * MAKER_FEE * cur_lev
                maker_entries += 1
                pending = None
            elif i > pending["expire_idx"]:
                if pending["fallback"]:
                    pos = pending["side"]
                    entry_price = bar_open * (1.0 + TAKER_SLIP) if pos == "LONG" else bar_open * (1.0 - TAKER_SLIP)
                    entry_fee = TAKER_FEE
                    entry_idx = i
                    cur_lev = float(pending["lev"])
                    active_tp = float(pending["tp"])
                    active_stop = float(pending["sl"])
                    active_hold = int(pending["hold"])
                    balance -= balance * TAKER_FEE * cur_lev
                    fallback_entries += 1
                else:
                    missed_entries += 1
                pending = None

        if pos is None and pending is None and waiting is None:
            setup = _geo_setup(prev)
            if setup is not None:
                lev = float(np.clip(0.08 + 0.22 * float(setup["entry_score"]), 0.08, 0.28))
                if float(setup["wait_score"]) > 0.05:
                    waiting = {
                        "side": str(setup["side"]),
                        "expire_idx": i + int(setup["max_wait"]),
                        "lev": lev,
                        "pullback_bps": float(setup["pullback_bps"]),
                        "threshold": float(setup["threshold"]),
                        "release_threshold": 0.03 if str(setup["side"]) == "LONG" else 0.05,
                        "entry_score": float(setup["entry_score"]),
                        "tp": float(setup["tp"]),
                        "sl": float(setup["sl"]),
                        "hold": int(setup["hold"]),
                    }
                else:
                    pullback = float(setup["pullback_bps"]) / 10000.0
                    pullback += max(float(prev["state_curvature"]), 0.0) * 0.00002
                    pullback += max(float(prev["toxicity"]), 0.0) * 0.00008
                    if str(setup["side"]) == "LONG":
                        pullback -= max(float(prev["d_whale_flow"]), 0.0) * 0.00035
                        price = float(prev["close"]) * (1.0 - float(np.clip(pullback, 0.0001, 0.0032)))
                    else:
                        pullback -= max(-float(prev["d_whale_flow"]), 0.0) * 0.00035
                        price = float(prev["close"]) * (1.0 + float(np.clip(pullback, 0.0001, 0.0032)))
                    pending = {
                        "side": str(setup["side"]),
                        "price": price,
                        "expire_idx": i + 1,
                        "lev": lev,
                        "fallback": bool(float(setup["entry_score"]) >= float(setup["threshold"]) + 0.10 and float(prev["toxicity"]) < 0.78),
                        "tp": float(setup["tp"]),
                        "sl": float(setup["sl"]),
                        "hold": int(setup["hold"]),
                    }

        if pos is not None:
            hold = i - entry_idx
            live_ret = _unrealized(pos, entry_price, float(bar["close"]), cur_lev, entry_fee)
            curve = float(np.tanh(float(bar["state_curvature"]) * 0.08))
            diss = float(np.tanh((float(bar["toxicity"]) + float(bar["queue_penalty"])) / 1.2))
            exit_cond = live_ret <= -active_stop or live_ret >= active_tp or hold >= active_hold
            exit_cond = exit_cond or curve > 0.72 or diss > 0.88 or float(bar["aftershock"]) > 0.66
            if pos == "LONG":
                exit_cond = exit_cond or float(np.tanh(float(bar["bear_line_int_5"]) / 1.2)) > 0.24 or float(bar["potential_gap"]) < -0.20
            else:
                exit_cond = exit_cond or float(np.tanh(float(bar["bull_line_int_5"]) / 1.2)) > 0.22 or float(bar["potential_gap"]) > 0.25
            if exit_cond:
                exit_price = float(df.iloc[i + 1]["open"]) * (1.0 - TAKER_SLIP) if pos == "LONG" else float(df.iloc[i + 1]["open"]) * (1.0 + TAKER_SLIP)
                realized = _realized(pos, entry_price, exit_price, cur_lev, entry_fee, TAKER_FEE)
                balance *= 1.0 + realized
                trades += 1
                wins += int(realized > 0.0)
                pos = None
                entry_price = 0.0
                entry_fee = 0.0
                entry_idx = -1
                cur_lev = 0.0
                active_stop = 0.0
                active_tp = 0.0
                active_hold = 0

        eq_curve.append(max(balance * (1.0 + (_unrealized(pos, entry_price, bar_close, cur_lev, entry_fee) if pos else 0.0)), 1e-8))

    return {
        "mode": "geometric_limit",
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "trades": trades,
        "wr_pct": round((100.0 * wins / trades) if trades else 0.0, 2),
        "sharpe": round(_sharpe(eq_curve), 4),
        "mdd_pct": round(_mdd(eq_curve), 4),
        "maker_entry_ratio": round(maker_entries / max(maker_entries + fallback_entries, 1), 4),
        "missed_entries": missed_entries,
        "maker_entries": maker_entries,
        "fallback_entries": fallback_entries,
        "wait_releases": wait_releases,
        "wait_cancels": wait_cancels,
    }


def main() -> None:
    df = build_dataset()
    configs = [
        StrategyConfig("balanced", 0.22, 0.18, 4.0, 3.0, 2.0, 18, 0.0045, 0.0060, 0.55, 0.90),
        StrategyConfig("strict_low_toxic", 0.26, 0.20, 4.5, 3.5, 2.5, 15, 0.0040, 0.0055, 0.45, 0.75),
        StrategyConfig("queue_aware", 0.24, 0.19, 3.5, 5.0, 2.5, 20, 0.0048, 0.0062, 0.55, 0.95),
        StrategyConfig("poly_aligned", 0.23, 0.18, 3.0, 2.5, 1.5, 16, 0.0042, 0.0058, 0.50, 0.85),
        StrategyConfig("high_conviction", 0.38, 0.18, 4.0, 2.5, 1.5, 16, 0.0038, 0.0050, 0.55, 0.85),
    ]

    results = []
    for cfg in configs:
        market = simulate_market(df, cfg)
        limit = simulate_limit(df, cfg)
        wait_limit = simulate_wait_limit(df, cfg)
        creative = simulate_creative_limit(df)
        geometric = simulate_geometric_limit(df)
        limit["delta_vs_market_pct"] = round(limit["pnl_pct"] - market["pnl_pct"], 4)
        wait_limit["delta_vs_market_pct"] = round(wait_limit["pnl_pct"] - market["pnl_pct"], 4)
        wait_limit["delta_vs_limit_pct"] = round(wait_limit["pnl_pct"] - limit["pnl_pct"], 4)
        creative["delta_vs_market_pct"] = round(creative["pnl_pct"] - market["pnl_pct"], 4)
        creative["delta_vs_limit_pct"] = round(creative["pnl_pct"] - limit["pnl_pct"], 4)
        creative["delta_vs_wait_limit_pct"] = round(creative["pnl_pct"] - wait_limit["pnl_pct"], 4)
        geometric["delta_vs_market_pct"] = round(geometric["pnl_pct"] - market["pnl_pct"], 4)
        geometric["delta_vs_limit_pct"] = round(geometric["pnl_pct"] - limit["pnl_pct"], 4)
        geometric["delta_vs_wait_limit_pct"] = round(geometric["pnl_pct"] - wait_limit["pnl_pct"], 4)
        geometric["delta_vs_creative_pct"] = round(geometric["pnl_pct"] - creative["pnl_pct"], 4)
        results.append(
            {
                "config": cfg.__dict__,
                "market": market,
                "limit": limit,
                "wait_limit": wait_limit,
                "creative_limit": creative,
                "geometric_limit": geometric,
            }
        )
        print(
            cfg.name,
            "market",
            market,
            "limit",
            limit,
            "wait_limit",
            wait_limit,
            "creative_limit",
            creative,
            "geometric_limit",
            geometric,
        )

    best = max(results, key=lambda x: x["geometric_limit"]["delta_vs_market_pct"])
    report = {
        "symbol": SYMBOL,
        "period": f"{df['ts'].min()} -> {df['ts'].max()}",
        "rows": int(len(df)),
        "notes": [
            "Uses persisted live microstructure/tail-risk/polymarket history plus matched Binance 1m futures candles.",
            "Compares market execution vs immediate maker vs wait-then-post maker vs creative archetype-based maker vs geometric/calculus-based maker execution.",
            "Historical whale_position_estimate is not stored minute-by-minute, so backtest uses persisted whale flow proxies instead.",
        ],
        "results": results,
        "best": best,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("BEST", best)
    print("SAVED", OUT_JSON)


if __name__ == "__main__":
    main()
