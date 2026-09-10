#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def zscore(s: pd.Series, win: int = 96) -> pd.Series:
    mu = s.rolling(win, min_periods=max(12, win // 4)).mean()
    sd = s.rolling(win, min_periods=max(12, win // 4)).std().replace(0, np.nan)
    return ((s - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def sigmoid(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Result:
    pnl_pct: float
    mdd_pct: float
    sharpe: float
    trades: int
    win_rate: float
    equity: float
    params: dict


def calc_mdd(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    dd = eq / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min()) * 100.0


def calc_sharpe(eq: np.ndarray, bars_per_year: int = 365 * 24 * 12) -> float:
    if len(eq) < 10:
        return 0.0
    r = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    s = r.std()
    if s < 1e-12:
        return 0.0
    return float(r.mean() / s * math.sqrt(bars_per_year))


def load_merged(price_csv: str, days: int) -> pd.DataFrame:
    con_ms = duckdb.connect("data/live/microstructure.duckdb", read_only=True)
    con_tr = duckdb.connect("data/live/tail_risk.duckdb", read_only=True)
    ms = con_ms.execute("select * from microstructure_1m order by ts").fetchdf()
    tr = con_tr.execute("select * from tail_risk_1m order by ts").fetchdf()

    ms["ts"] = pd.to_datetime(ms["ts"]).dt.tz_convert("Asia/Seoul")
    tr["ts"] = pd.to_datetime(tr["ts"]).dt.tz_convert("Asia/Seoul")

    df = pd.merge_asof(
        ms.sort_values("ts"),
        tr.sort_values("ts"),
        on="ts",
        direction="backward",
        tolerance=pd.Timedelta("90s"),
    )

    # 1m -> 5m aggregate
    agg = {
        "obi": "mean",
        "taker_buy_ratio": "mean",
        "spoofing_score": "mean",
        "nif_whale": "mean",
        "nif_retail": "mean",
        "eai": "last",
        "oi_delta_pct": "mean",
        "funding_rate": "last",
        "shadow_toxicity_score": "mean",
        "shadow_queue_collapse": "mean",
        "shadow_absorption_score": "mean",
        "shadow_regime_conf": "mean",
        "long_usd_1m": "sum",
        "short_usd_1m": "sum",
        "shadow_aftershock_prob": "mean",
    }
    d5 = (
        df.set_index("ts")
        .resample("5min")
        .agg(agg)
        .dropna(subset=["nif_whale", "obi", "funding_rate", "shadow_toxicity_score"])
        .reset_index()
    )

    px = pd.read_csv(price_csv, usecols=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"])
    px["timestamp"] = pd.to_datetime(px["timestamp"]).dt.tz_localize("Asia/Seoul")
    px = px.rename(columns={"timestamp": "ts"}).sort_values("ts")

    m = pd.merge_asof(
        d5.sort_values("ts"),
        px,
        on="ts",
        direction="backward",
        tolerance=pd.Timedelta("7min"),
    )
    m = m.dropna(subset=["close"]).copy()

    if len(m) == 0:
        raise RuntimeError("No overlap between duckdb signals and price data")

    end = m["ts"].max()
    start = end - pd.Timedelta(days=days)
    m = m[m["ts"] >= start].reset_index(drop=True)

    # derived features
    m["flow_sign"] = 2.0 * m["taker_buy_ratio"].clip(0, 1) - 1.0
    m["liq_skew"] = (m["short_usd_1m"] - m["long_usd_1m"]) / (m["short_usd_1m"].abs() + m["long_usd_1m"].abs() + 1e-8)
    m["overheat"] = zscore(m["oi_delta_pct"], 96) + zscore(m["funding_rate"], 96)

    # vpin-lite from quote volume * imbalance
    imb = (m["quote_volume"].fillna(0.0) * m["flow_sign"].abs()).fillna(0.0)
    m["vpin_lite"] = (imb.rolling(60, min_periods=12).sum() / np.maximum(m["quote_volume"].rolling(60, min_periods=12).sum(), 1e-8)).fillna(0.0).clip(0, 1)

    # ATR and vol regime
    prev_close = m["close"].shift(1).fillna(m["close"])
    trr = np.maximum(
        (m["high"] - m["low"]).abs(),
        np.maximum((m["high"] - prev_close).abs(), (m["low"] - prev_close).abs()),
    )
    atr14 = trr.rolling(14, min_periods=5).mean().fillna(0.0)
    m["atr14_pct"] = (atr14 / np.maximum(m["close"], 1e-8)).fillna(0.0)
    v1h = m["volume"].rolling(12, min_periods=6).mean().fillna(0.0)
    v24h = m["volume"].rolling(288, min_periods=24).mean().ffill().fillna(1.0)
    m["vol1h_ratio"] = (v1h / np.maximum(v24h, 1e-8)).fillna(0.0)

    # session flags
    hh = m["ts"].dt.hour
    m["session_us"] = ((hh >= 22) | (hh <= 4)).astype(float)
    m["session_eu"] = ((hh >= 15) & (hh < 22)).astype(float)

    return m


def run_sim(m: pd.DataFrame, p: dict) -> Result:
    close = m["close"].to_numpy(np.float64)
    high = m["high"].to_numpy(np.float64)
    low = m["low"].to_numpy(np.float64)

    obi = m["obi"].to_numpy(np.float64)
    nif = m["nif_whale"].to_numpy(np.float64)
    flow = m["flow_sign"].to_numpy(np.float64)
    absb = m["shadow_absorption_score"].to_numpy(np.float64)
    tox = m["shadow_toxicity_score"].to_numpy(np.float64)
    qcol = m["shadow_queue_collapse"].to_numpy(np.float64)
    aft = m["shadow_aftershock_prob"].to_numpy(np.float64)
    eai = m["eai"].to_numpy(np.float64)
    over = m["overheat"].to_numpy(np.float64)
    liq = m["liq_skew"].to_numpy(np.float64)
    vpin = m["vpin_lite"].to_numpy(np.float64)
    atr = m["atr14_pct"].to_numpy(np.float64)
    volr = m["vol1h_ratio"].to_numpy(np.float64)
    us = m["session_us"].to_numpy(np.float64)
    eu = m["session_eu"].to_numpy(np.float64)

    # composite alpha score
    raw = (
        p["w_nif"] * nif
        + p["w_flow"] * flow
        + p["w_obi"] * (-obi)
        + p["w_abs"] * absb
        + p["w_liq"] * liq
        + p["w_eai"] * np.tanh(eai / 2.0)
        - p["w_tox"] * tox
        - p["w_aft"] * aft
        - p["w_vpin"] * np.clip(vpin - 0.7, 0, 1)
    )

    long_gate = (over < p["overheat_long_max"]).astype(float)
    short_boost = np.where(over > p["overheat_short_min"], p["short_boost"], 1.0)

    base_long = sigmoid((raw - p["bias"]) / max(p["temp"], 1e-4))
    base_short = sigmoid((-raw - p["bias"]) / max(p["temp"], 1e-4))

    # tail penalty
    tail_pen = np.clip(1.0 - (p["tail_tox"] * tox + p["tail_qc"] * qcol + p["tail_aft"] * aft), 0.0, 1.0)

    long_score = base_long * long_gate * tail_pen
    short_score = base_short * short_boost * tail_pen

    eq = 1.0
    eq_curve = [eq]
    pos = 0
    size = 0.0
    entry = 0.0
    peak_px = 0.0
    trough_px = 0.0
    trades = 0
    wins = 0
    cooldown = 0
    long_h = False
    short_h = False

    for i in range(1, len(close)):
        if cooldown > 0:
            cooldown -= 1

        # session-adaptive thresholds
        sess_mult = 0.93 if us[i] > 0.5 else (0.97 if eu[i] > 0.5 else 1.05)
        entry_th = p["entry"] * sess_mult
        exit_th = p["exit"] * (0.98 if us[i] > 0.5 else 1.0)

        if not long_h and long_score[i] >= entry_th:
            long_h = True
        elif long_h and long_score[i] <= exit_th:
            long_h = False

        if not short_h and short_score[i] >= entry_th:
            short_h = True
        elif short_h and short_score[i] <= exit_th:
            short_h = False

        sig = 0
        if long_h and (not short_h or long_score[i] >= short_score[i]):
            sig = 1
        elif short_h and (not long_h or short_score[i] > long_score[i]):
            sig = -1

        # regime filter
        tradable = (atr[i] >= p["atr_min"]) and (volr[i] >= p["volr_min"]) and (vpin[i] <= p["vpin_max"])
        if not tradable:
            sig = 0

        strength = long_score[i] if sig == 1 else (short_score[i] if sig == -1 else 0.0)
        target_size = np.clip((strength - exit_th) / max(entry_th - exit_th, 1e-6), 0, 1)

        slip = p["slip"]

        if pos == 0 and sig != 0 and target_size > 0 and cooldown == 0:
            # maker fill model + taker fallback
            maker_fill = abs((close[i] - close[i - 1]) / max(close[i - 1], 1e-8)) <= p["maker_move_max"]
            use_taker = (not maker_fill) and (abs(raw[i]) > p["fallback_raw_th"] or eai[i] > p["fallback_eai_th"])
            if maker_fill or use_taker:
                pos = sig
                size = target_size
                lev = p["lev"]
                entry = close[i] * (1 + slip if pos == 1 else 1 - slip)
                fee_in = p["taker_fee"] if use_taker else p["maker_fee"]
                eq *= (1.0 - fee_in * size * lev)
                trades += 1
                peak_px = entry
                trough_px = entry

        elif pos != 0:
            lev = p["lev"]
            if pos == 1:
                peak_px = max(peak_px, high[i])
            else:
                trough_px = min(trough_px, low[i])

            ret_m = (close[i] - entry) / max(entry, 1e-12)
            if pos == -1:
                ret_m = -ret_m

            # dynamic trailing
            dyn_gap = p["trail"] * min(1.0 + p["trail_tox_a"] * tox[i], p["trail_max_mult"]) * min(1.0 + p["trail_qc_b"] * qcol[i], p["trail_max_mult"])
            hit_trail = (close[i] <= peak_px * (1 - dyn_gap)) if pos == 1 else (close[i] >= trough_px * (1 + dyn_gap))
            hit_tp = ret_m >= p["tp"]
            hit_sl = ret_m <= -p["sl"]

            should_exit = hit_trail or hit_tp or hit_sl or sig == 0 or sig == -pos
            if should_exit:
                exit_px = close[i] * (1 - slip if pos == 1 else 1 + slip)
                rr = (exit_px - entry) / max(entry, 1e-12)
                if pos == -1:
                    rr = -rr
                pnl = rr * size * lev
                eq *= (1.0 + pnl)
                eq *= (1.0 - p["taker_fee"] * size * lev)
                wins += int(pnl > 0)
                pos = 0
                size = 0.0
                entry = 0.0
                cooldown = p["cooldown"]
            else:
                # maker rebalance
                delta = abs(target_size - size)
                if delta > 1e-3:
                    eq *= (1.0 - p["maker_fee"] * delta * lev)
                    size = target_size

        eq_curve.append(eq)

    eqa = np.array(eq_curve, dtype=np.float64)
    pnl_pct = (eqa[-1] - 1.0) * 100.0
    mdd = calc_mdd(eqa)
    shp = calc_sharpe(eqa)
    wr = (wins / trades * 100.0) if trades > 0 else 0.0
    return Result(
        pnl_pct=float(pnl_pct),
        mdd_pct=float(mdd),
        sharpe=float(shp),
        trades=int(trades),
        win_rate=float(wr),
        equity=float(eqa[-1]),
        params=p,
    )


def sample_params(rng: random.Random) -> dict:
    return {
        "w_nif": rng.uniform(0.2, 1.8),
        "w_flow": rng.uniform(0.1, 1.2),
        "w_obi": rng.uniform(0.1, 1.2),
        "w_abs": rng.uniform(0.1, 1.3),
        "w_liq": rng.uniform(0.0, 1.2),
        "w_eai": rng.uniform(0.0, 1.0),
        "w_tox": rng.uniform(0.2, 1.8),
        "w_aft": rng.uniform(0.1, 1.4),
        "w_vpin": rng.uniform(0.0, 1.2),
        "bias": rng.uniform(-0.15, 0.15),
        "temp": rng.uniform(0.08, 0.45),
        "overheat_long_max": rng.uniform(-0.2, 1.2),
        "overheat_short_min": rng.uniform(0.6, 2.0),
        "short_boost": rng.uniform(1.0, 2.2),
        "tail_tox": rng.uniform(0.15, 0.7),
        "tail_qc": rng.uniform(0.1, 0.6),
        "tail_aft": rng.uniform(0.1, 0.6),
        "entry": rng.uniform(0.58, 0.90),
        "exit": rng.uniform(0.30, 0.60),
        "atr_min": rng.uniform(0.0006, 0.0020),
        "volr_min": rng.uniform(0.35, 0.90),
        "vpin_max": rng.uniform(0.78, 0.98),
        "lev": rng.uniform(2.0, 12.0),
        "maker_fee": 0.0002,
        "taker_fee": 0.0005,
        "slip": rng.uniform(0.0001, 0.0004),
        "maker_move_max": rng.uniform(0.0010, 0.0035),
        "fallback_raw_th": rng.uniform(0.35, 1.4),
        "fallback_eai_th": rng.uniform(1.2, 3.2),
        "trail": rng.uniform(0.003, 0.010),
        "trail_tox_a": rng.uniform(0.2, 1.6),
        "trail_qc_b": rng.uniform(0.2, 1.6),
        "trail_max_mult": rng.uniform(1.6, 4.0),
        "tp": rng.uniform(0.015, 0.08),
        "sl": rng.uniform(0.004, 0.02),
        "cooldown": rng.randint(6, 72),
    }


def main():
    ap = argparse.ArgumentParser(description="Optimize quant formula on DuckDB micro/tail + price")
    ap.add_argument("--price-csv", default="binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv")
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--target-pnl", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/ensemble/metrics/duckdb_quant_opt_result.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    m = load_merged(args.price_csv, args.days)
    if len(m) < 300:
        raise SystemExit(f"not enough merged rows: {len(m)}")

    best: Result | None = None
    found: Result | None = None

    for i in range(1, args.iters + 1):
        p = sample_params(rng)
        # keep valid hysteresis
        if p["exit"] >= p["entry"]:
            p["exit"] = max(0.20, p["entry"] - 0.08)
        r = run_sim(m, p)
        if best is None or r.pnl_pct > best.pnl_pct:
            best = r
            print(f"[iter {i}] best pnl={r.pnl_pct:+.2f}% mdd={r.mdd_pct:.2f}% trades={r.trades} wr={r.win_rate:.1f}% sharpe={r.sharpe:.2f} lev={p['lev']:.2f}")
        if r.pnl_pct >= args.target_pnl:
            found = r
            print(f"[iter {i}] TARGET HIT pnl={r.pnl_pct:+.2f}%")
            break

    pick = found or best
    payload = {
        "target_pnl": args.target_pnl,
        "iters": args.iters,
        "days": args.days,
        "rows": int(len(m)),
        "start": str(m["ts"].min()),
        "end": str(m["ts"].max()),
        "hit_target": bool(found is not None),
        "best": {
            "pnl_pct": pick.pnl_pct,
            "mdd_pct": pick.mdd_pct,
            "sharpe": pick.sharpe,
            "trades": pick.trades,
            "win_rate": pick.win_rate,
            "equity": pick.equity,
            "params": pick.params,
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n=== Final ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
