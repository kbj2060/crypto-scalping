#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

SYMBOL = "ETHUSDT"
INTERVAL = "5m"
OUT_CSV = "data/api_execution_30d_5m.csv"
OUT_JSON = "data/ensemble/reports/backtest_api_limit_30d.json"
TAKER_FEE = 0.0005
MAKER_FEE = 0.0002
TAKER_SLIP = 0.0002
ANNUAL_FACTOR_5M = math.sqrt(365 * 24 * 12)


@dataclass
class Config:
    name: str
    entry_th: float
    exit_th: float
    pullback_bps: float
    max_wait_bars: int
    max_hold_bars: int
    sl_pct: float
    tp_pct: float


def _fetch_json(url: str):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_klines_30d() -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    cur = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows = []
    while cur < end_ms:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={SYMBOL}&interval={INTERVAL}&startTime={cur}&limit=1500"
        arr = _fetch_json(url)
        if not arr:
            break
        rows.extend(arr)
        nxt = int(arr[-1][0]) + 5 * 60_000
        if nxt <= cur:
            break
        cur = nxt
    df = pd.DataFrame(
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
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert("Asia/Seoul")
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "tb_base", "tb_quote", "trades"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["ts", "open", "high", "low", "close", "volume", "quote_volume", "trades", "tb_base", "tb_quote"]]


def _fetch_metric(path: str, value_cols: list[str]) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=29, hours=23)
    cur = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows = []
    while cur < end_ms:
        url = f"https://fapi.binance.com/futures/data/{path}?symbol={SYMBOL}&period=5m&limit=500&startTime={cur}"
        arr = _fetch_json(url)
        if not arr:
            break
        rows.extend(arr)
        ts_field = "timestamp"
        nxt = int(arr[-1][ts_field]) + 5 * 60_000
        if nxt <= cur:
            break
        cur = nxt
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["ts"] + value_cols)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Seoul")
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = ["ts"] + value_cols
    return df[keep].drop_duplicates("ts").sort_values("ts")


def _fetch_funding_30d() -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    cur = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows = []
    while cur < end_ms:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={SYMBOL}&limit=1000&startTime={cur}"
        arr = _fetch_json(url)
        if not arr:
            break
        rows.extend(arr)
        nxt = int(arr[-1]["fundingTime"]) + 8 * 60 * 60_000
        if nxt <= cur:
            break
        cur = nxt
    df = pd.DataFrame(rows)
    df["ts_funding"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.tz_convert("Asia/Seoul")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df[["ts_funding", "fundingRate"]].drop_duplicates("ts_funding").sort_values("ts_funding")


def build_dataset() -> pd.DataFrame:
    px = _fetch_klines_30d().sort_values("ts").reset_index(drop=True)
    oi = _fetch_metric("openInterestHist", ["sumOpenInterestValue"])
    top = _fetch_metric("topLongShortAccountRatio", ["longShortRatio"])
    glob = _fetch_metric("globalLongShortAccountRatio", ["longShortRatio"]).rename(columns={"longShortRatio": "globalLongShortRatio"})
    taker = _fetch_metric("takerlongshortRatio", ["buySellRatio"])
    funding = _fetch_funding_30d()

    df = px.merge(oi, on="ts", how="left").merge(top, on="ts", how="left").merge(glob, on="ts", how="left").merge(taker, on="ts", how="left")
    df = pd.merge_asof(df.sort_values("ts"), funding.rename(columns={"ts_funding": "ts"}).sort_values("ts"), on="ts", direction="backward")
    df["has_oi"] = df["sumOpenInterestValue"].notna().astype(float)
    df["has_top_ratio"] = df["longShortRatio"].notna().astype(float)
    df["has_global_ratio"] = df["globalLongShortRatio"].notna().astype(float)
    df["has_taker_ratio"] = df["buySellRatio"].notna().astype(float)
    df["has_funding"] = df["fundingRate"].notna().astype(float)
    df = df.sort_values("ts").reset_index(drop=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce").ffill().fillna(0.0)
    df["sumOpenInterestValue"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
    df["longShortRatio"] = pd.to_numeric(df["longShortRatio"], errors="coerce")
    df["globalLongShortRatio"] = pd.to_numeric(df["globalLongShortRatio"], errors="coerce")
    df["buySellRatio"] = pd.to_numeric(df["buySellRatio"], errors="coerce")
    df = df.reset_index(drop=True)

    df["ret_1"] = df["close"].pct_change().fillna(0.0)
    df["ret_3"] = df["close"].pct_change(3).fillna(0.0)
    df["ret_6"] = df["close"].pct_change(6).fillna(0.0)
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["rv_12"] = df["ret_1"].rolling(12, min_periods=1).std().fillna(0.0)
    df["oi_chg"] = df["sumOpenInterestValue"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["taker_imb"] = df["buySellRatio"].fillna(1.0) - 1.0
    df["crowding"] = df["longShortRatio"].fillna(1.0) - df["globalLongShortRatio"].fillna(1.0)
    df["funding_z"] = (
        (pd.to_numeric(df["fundingRate"], errors="coerce").fillna(0.0) - pd.to_numeric(df["fundingRate"], errors="coerce").fillna(0.0).rolling(30, min_periods=5).mean())
        / (pd.to_numeric(df["fundingRate"], errors="coerce").fillna(0.0).rolling(30, min_periods=5).std() + 1e-8)
    ).fillna(0.0)

    df["pressure_long"] = (
        0.30 * np.tanh(df["taker_imb"] / 0.18) * df["has_taker_ratio"]
        + 0.22 * np.tanh(df["oi_chg"] / 0.003) * df["has_oi"]
        + 0.16 * np.tanh(df["ret_3"] / 0.01)
        - 0.12 * np.tanh(df["crowding"] / 0.35) * np.minimum(df["has_top_ratio"], df["has_global_ratio"])
        - 0.10 * np.tanh(df["funding_z"] / 1.5)
        - 0.10 * np.tanh(df["rv_12"] / 0.006)
    )
    df["pressure_short"] = (
        -0.30 * np.tanh(df["taker_imb"] / 0.18) * df["has_taker_ratio"]
        - 0.22 * np.tanh(df["oi_chg"] / 0.003) * df["has_oi"]
        - 0.16 * np.tanh(df["ret_3"] / 0.01)
        + 0.12 * np.tanh(df["crowding"] / 0.35) * np.minimum(df["has_top_ratio"], df["has_global_ratio"])
        + 0.10 * np.tanh(df["funding_z"] / 1.5)
        - 0.10 * np.tanh(df["rv_12"] / 0.006)
    )
    df["wait_long"] = (
        0.32 * np.tanh(-df["ret_1"] / 0.0035)
        + 0.18 * np.tanh(-df["ret_3"] / 0.009)
        + 0.16 * np.tanh(df["rv_12"] / 0.006)
        - 0.18 * np.tanh(df["taker_imb"] / 0.18) * df["has_taker_ratio"]
        - 0.10 * np.tanh(df["oi_chg"] / 0.003) * df["has_oi"]
    )
    df["wait_short"] = (
        0.32 * np.tanh(df["ret_1"] / 0.0035)
        + 0.18 * np.tanh(df["ret_3"] / 0.009)
        + 0.16 * np.tanh(df["rv_12"] / 0.006)
        + 0.18 * np.tanh(df["taker_imb"] / 0.18) * df["has_taker_ratio"]
        + 0.10 * np.tanh(df["oi_chg"] / 0.003) * df["has_oi"]
    )
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    return df


def _sharpe(eq: list[float]) -> float:
    arr = np.array(eq, dtype=np.float64)
    rets = np.diff(arr) / np.maximum(arr[:-1], 1e-12)
    if len(rets) < 3 or np.std(rets) < 1e-12:
        return 0.0
    return float(np.mean(rets) / np.std(rets) * ANNUAL_FACTOR_5M)


def _mdd(eq: list[float]) -> float:
    arr = np.array(eq, dtype=np.float64)
    run_max = np.maximum.accumulate(arr)
    dd = arr / np.maximum(run_max, 1e-12) - 1.0
    return float(np.min(dd)) * 100.0


def _realized(side: str, ep: float, xp: float, lev: float, ef: float, xf: float) -> float:
    gross = (xp - ep) / ep if side == "LONG" else (ep - xp) / ep
    return float(gross * lev - (ef + xf) * lev)


def _unrealized(side: str | None, ep: float, cp: float, lev: float, ef: float) -> float:
    if side is None or ep <= 0 or lev <= 0:
        return 0.0
    gross = (cp - ep) / ep if side == "LONG" else (ep - cp) / ep
    return float(gross * lev - (ef + MAKER_FEE) * lev)


def simulate_market(df: pd.DataFrame, cfg: Config) -> dict:
    balance = 1.0
    eq = [1.0]
    pos = None
    ep = 0.0
    ef = 0.0
    lev = 0.0
    entry_idx = -1
    trades = wins = 0
    for i in range(len(df) - 1):
        row = df.iloc[i]
        nxt_open = float(df.iloc[i + 1]["open"])
        nxt_close = float(df.iloc[i + 1]["close"])
        long_p = float(row["pressure_long"])
        short_p = float(row["pressure_short"])
        size = float(np.clip(0.10 + 0.20 * max(abs(long_p), abs(short_p)), 0.10, 0.30))
        if pos is None:
            if long_p >= cfg.entry_th and long_p > short_p:
                pos, ep, ef, lev, entry_idx = "LONG", nxt_open * (1.0 + TAKER_SLIP), TAKER_FEE, size, i + 1
                balance -= balance * TAKER_FEE * lev
            elif short_p >= cfg.entry_th and short_p > long_p:
                pos, ep, ef, lev, entry_idx = "SHORT", nxt_open * (1.0 - TAKER_SLIP), TAKER_FEE, size, i + 1
                balance -= balance * TAKER_FEE * lev
        else:
            hold = i - entry_idx
            live = _unrealized(pos, ep, float(row["close"]), lev, ef)
            exit_cond = live <= -cfg.sl_pct or live >= cfg.tp_pct or hold >= cfg.max_hold_bars
            exit_cond = exit_cond or (pos == "LONG" and short_p >= cfg.exit_th) or (pos == "SHORT" and long_p >= cfg.exit_th)
            if exit_cond:
                xp = nxt_open * (1.0 - TAKER_SLIP) if pos == "LONG" else nxt_open * (1.0 + TAKER_SLIP)
                r = _realized(pos, ep, xp, lev, ef, TAKER_FEE)
                balance *= 1.0 + r
                trades += 1
                wins += int(r > 0.0)
                pos, ep, ef, lev, entry_idx = None, 0.0, 0.0, 0.0, -1
        eq.append(max(balance * (1.0 + (_unrealized(pos, ep, nxt_close, lev, ef) if pos else 0.0)), 1e-8))
    return {
        "mode": "market",
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "trades": trades,
        "wr_pct": round((100.0 * wins / trades) if trades else 0.0, 2),
        "sharpe": round(_sharpe(eq), 4),
        "mdd_pct": round(_mdd(eq), 4),
    }


def simulate_wait_limit(df: pd.DataFrame, cfg: Config) -> dict:
    balance = 1.0
    eq = [1.0]
    pos = None
    ep = 0.0
    ef = 0.0
    lev = 0.0
    entry_idx = -1
    trades = wins = 0
    pending = None
    waiting = None
    maker_entries = fallback_entries = missed_entries = 0
    wait_releases = wait_cancels = 0
    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        bar = df.iloc[i]
        bopen, bhigh, blow, bclose = map(float, [bar["open"], bar["high"], bar["low"], bar["close"]])
        if waiting is not None:
            side = waiting["side"]
            score = float(prev["pressure_long"] if side == "LONG" else prev["pressure_short"])
            wait = float(prev["wait_long"] if side == "LONG" else prev["wait_short"])
            release = score >= waiting["entry_th"] and wait <= 0.02
            invalid = score < waiting["entry_th"] - 0.04
            if i > waiting["expire_idx"] or invalid:
                wait_cancels += 1
                waiting = None
            elif release:
                pull = cfg.pullback_bps / 10000.0
                pull += max(float(prev["rv_12"]) - 0.003, 0.0) * 0.08
                pull -= max(float(prev["taker_imb"]), 0.0) * 0.0007 if side == "LONG" else max(float(-prev["taker_imb"]), 0.0) * 0.0007
                price = float(prev["close"]) * (1.0 - pull) if side == "LONG" else float(prev["close"]) * (1.0 + pull)
                pending = {"side": side, "price": price, "expire_idx": i + 1, "lev": waiting["lev"], "fallback": waiting["fallback"]}
                wait_releases += 1
                waiting = None
        if pending is not None:
            fill = (pending["side"] == "LONG" and blow <= pending["price"]) or (pending["side"] == "SHORT" and bhigh >= pending["price"])
            if fill:
                pos, ep, ef, lev, entry_idx = pending["side"], float(pending["price"]), MAKER_FEE, float(pending["lev"]), i
                balance -= balance * MAKER_FEE * lev
                maker_entries += 1
                pending = None
            elif i > pending["expire_idx"]:
                if pending["fallback"]:
                    pos = pending["side"]
                    ep = bopen * (1.0 + TAKER_SLIP) if pos == "LONG" else bopen * (1.0 - TAKER_SLIP)
                    ef, lev, entry_idx = TAKER_FEE, float(pending["lev"]), i
                    balance -= balance * TAKER_FEE * lev
                    fallback_entries += 1
                else:
                    missed_entries += 1
                pending = None
        if pos is None and pending is None and waiting is None:
            long_p = float(prev["pressure_long"])
            short_p = float(prev["pressure_short"])
            size = float(np.clip(0.10 + 0.20 * max(abs(long_p), abs(short_p)), 0.10, 0.30))
            if long_p >= cfg.entry_th and long_p > short_p:
                if float(prev["wait_long"]) > 0.03:
                    waiting = {"side": "LONG", "expire_idx": i + cfg.max_wait_bars, "lev": size, "entry_th": cfg.entry_th, "fallback": long_p >= cfg.entry_th + 0.12}
                else:
                    pull = cfg.pullback_bps / 10000.0
                    pull += max(float(prev["rv_12"]) - 0.003, 0.0) * 0.08
                    pull -= max(float(prev["taker_imb"]), 0.0) * 0.0007
                    pending = {"side": "LONG", "price": float(prev["close"]) * (1.0 - pull), "expire_idx": i + 1, "lev": size, "fallback": long_p >= cfg.entry_th + 0.12}
            elif short_p >= cfg.entry_th and short_p > long_p:
                if float(prev["wait_short"]) > 0.03:
                    waiting = {"side": "SHORT", "expire_idx": i + cfg.max_wait_bars, "lev": size, "entry_th": cfg.entry_th, "fallback": short_p >= cfg.entry_th + 0.12}
                else:
                    pull = cfg.pullback_bps / 10000.0
                    pull += max(float(prev["rv_12"]) - 0.003, 0.0) * 0.08
                    pull -= max(float(-prev["taker_imb"]), 0.0) * 0.0007
                    pending = {"side": "SHORT", "price": float(prev["close"]) * (1.0 + pull), "expire_idx": i + 1, "lev": size, "fallback": short_p >= cfg.entry_th + 0.12}
        if pos is not None:
            hold = i - entry_idx
            long_p = float(bar["pressure_long"])
            short_p = float(bar["pressure_short"])
            live = _unrealized(pos, ep, bclose, lev, ef)
            exit_cond = live <= -cfg.sl_pct or live >= cfg.tp_pct or hold >= cfg.max_hold_bars
            exit_cond = exit_cond or (pos == "LONG" and short_p >= cfg.exit_th) or (pos == "SHORT" and long_p >= cfg.exit_th)
            if exit_cond:
                xp = float(df.iloc[i + 1]["open"]) * (1.0 - TAKER_SLIP) if pos == "LONG" else float(df.iloc[i + 1]["open"]) * (1.0 + TAKER_SLIP)
                r = _realized(pos, ep, xp, lev, ef, TAKER_FEE)
                balance *= 1.0 + r
                trades += 1
                wins += int(r > 0.0)
                pos, ep, ef, lev, entry_idx = None, 0.0, 0.0, 0.0, -1
        eq.append(max(balance * (1.0 + (_unrealized(pos, ep, bclose, lev, ef) if pos else 0.0)), 1e-8))
    return {
        "mode": "wait_limit",
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "trades": trades,
        "wr_pct": round((100.0 * wins / trades) if trades else 0.0, 2),
        "sharpe": round(_sharpe(eq), 4),
        "mdd_pct": round(_mdd(eq), 4),
        "maker_entry_ratio": round(maker_entries / max(maker_entries + fallback_entries, 1), 4),
        "missed_entries": missed_entries,
        "wait_releases": wait_releases,
        "wait_cancels": wait_cancels,
    }


def main():
    df = build_dataset()
    configs = [
        Config("balanced", 0.16, 0.14, 5.0, 2, 18, 0.005, 0.007),
        Config("tighter", 0.20, 0.16, 4.0, 2, 14, 0.0045, 0.0065),
        Config("high_conviction", 0.24, 0.18, 3.0, 3, 12, 0.004, 0.006),
    ]
    results = []
    for cfg in configs:
        market = simulate_market(df, cfg)
        wait_limit = simulate_wait_limit(df, cfg)
        wait_limit["delta_vs_market_pct"] = round(wait_limit["pnl_pct"] - market["pnl_pct"], 4)
        results.append({"config": asdict(cfg), "market": market, "wait_limit": wait_limit})
        print(cfg.name, market, wait_limit)
    best = max(results, key=lambda x: x["wait_limit"]["delta_vs_market_pct"])
    report = {
        "symbol": SYMBOL,
        "period": f"{df['ts'].min()} -> {df['ts'].max()}",
        "rows": int(len(df)),
        "dataset_csv": OUT_CSV,
        "notes": [
            "Built from Binance public APIs only.",
            "Klines and funding are available for long history; oi/top/global/taker ratio endpoints are limited to recent ~30 days, so this report uses a 30-day window.",
            "Polymarket is excluded from this expanded sample because historical public snapshot collection was not stable enough for long-range reconstruction.",
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
