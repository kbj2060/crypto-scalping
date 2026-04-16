#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
MICRO_DB = ROOT / "data/live/microstructure.duckdb"
TAIL_DB = ROOT / "data/live/tail_risk.duckdb"


@dataclass
class TrialResult:
    objective: float
    pnl_pct: float
    sharpe_1m: float
    mdd_pct: float
    win_rate_pct: float
    trades: int
    avg_pos: float
    fee_bps: float
    params: dict


def load_signals() -> pd.DataFrame:
    con_m = duckdb.connect(str(MICRO_DB))
    con_t = duckdb.connect(str(TAIL_DB))
    m = con_m.execute(
        """
        SELECT
          date_trunc('minute', ts) AS ts,
          obi, nif_whale, eai, funding_rate,
          shadow_toxicity_score, shadow_queue_collapse, shadow_absorption_score
        FROM microstructure_1m
        ORDER BY ts
        """
    ).df()
    t = con_t.execute(
        """
        SELECT
          date_trunc('minute', ts) AS ts,
          shadow_aftershock_prob
        FROM tail_risk_1m
        ORDER BY ts
        """
    ).df()
    con_m.close()
    con_t.close()

    m["ts"] = pd.to_datetime(m["ts"], utc=True)
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    df = pd.merge(m, t, on="ts", how="inner")
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df


def fetch_binance_1m(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/klines"
    out = []
    cur = start_ms
    while cur <= end_ms:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1500,
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        out.extend(rows)
        last_open = int(rows[-1][0])
        cur = last_open + 60_000
        if len(rows) < 1500:
            break
    if not out:
        raise RuntimeError("No Binance klines fetched.")
    px = pd.DataFrame(out, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"
    ])
    px["ts"] = pd.to_datetime(px["open_time"].astype(np.int64), unit="ms", utc=True)
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px[["ts", "close"]].dropna().drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return px


def compute_metrics(ret: np.ndarray, pos: np.ndarray, fee_bps: float, leverage: float) -> tuple[float, float, float, float, int]:
    fee = fee_bps / 10_000.0
    turn = np.abs(np.diff(pos, prepend=0.0))
    lev = max(float(leverage), 0.0)
    pnl = lev * pos * ret - fee * lev * turn
    eq = np.cumprod(1.0 + pnl)
    total = float(eq[-1] - 1.0) if len(eq) else 0.0
    mu = float(np.mean(pnl)) if len(pnl) else 0.0
    sd = float(np.std(pnl) + 1e-12) if len(pnl) else 1e-12
    sharpe = mu / sd
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([1.0])
    dd = (eq / peak - 1.0) if len(eq) else np.array([0.0])
    mdd = float(-dd.min()) if len(dd) else 0.0
    wr = float((pnl > 0).mean()) if len(pnl) else 0.0
    trades = int((turn > 1e-9).sum())
    return total, sharpe, mdd, wr, trades


def simulate(df: pd.DataFrame, p: dict, fee_bps: float, leverage: float) -> TrialResult:
    x = df.copy()
    for c in [
        "obi", "nif_whale", "shadow_absorption_score", "shadow_toxicity_score",
        "shadow_aftershock_prob", "funding_rate", "eai", "ret_fwd_1m", "shadow_queue_collapse"
    ]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)

    squeeze = np.sign(-x["funding_rate"].to_numpy()) * np.maximum(x["eai"].to_numpy() - p["eai_th"], 0.0)
    edge_raw = (
        p["w_obi"] * x["obi"].to_numpy()
        + p["w_nif"] * x["nif_whale"].to_numpy()
        + p["w_abs"] * x["shadow_absorption_score"].to_numpy()
        - p["w_tox"] * x["shadow_toxicity_score"].to_numpy()
        - p["w_aft"] * x["shadow_aftershock_prob"].to_numpy()
        + p["w_sqz"] * squeeze
    )
    edge = np.tanh(edge_raw)

    tox = x["shadow_toxicity_score"].to_numpy()
    col = x["shadow_queue_collapse"].to_numpy()
    aft = x["shadow_aftershock_prob"].to_numpy()
    obi = x["obi"].to_numpy()
    nif = x["nif_whale"].to_numpy()
    absb = x["shadow_absorption_score"].to_numpy()
    eai = x["eai"].to_numpy()
    fr = x["funding_rate"].to_numpy()

    pb9 = (col > 0.75) & (tox > 0.85)
    pb5 = (absb > 0.70) & (nif >= 0.40) & (aft < 0.55)
    pb8s = (obi > 0.35) & (nif < -0.35) & (tox > 0.75)
    pb8l = (obi < -0.35) & (nif > 0.35) & (tox > 0.75)
    pb2l = (eai > p["eai_th"]) & (fr < p["fund_th"]) & (absb > 0.70)
    pb2s = (eai > p["eai_th"]) & (fr > abs(p["fund_th"])) & (absb > 0.70)
    pb7l = (obi > 0.30) & (nif > 0.30) & (tox < 0.30)
    pb7s = (obi < -0.30) & (nif < -0.30) & (tox < 0.30)

    base_th = p["base_th"] + 0.20 * aft + 0.15 * tox
    sig = np.where(edge > base_th, 1, np.where(edge < -base_th, -1, 0))

    # Priority: PB5(95) > PB9(93) > PB8(92) > PB2(89) > PB7(88)
    pb_sig = np.full(len(x), 99, dtype=np.int8)  # 99 means no override
    pb_sig = np.where(pb7l, 1, np.where(pb7s, -1, pb_sig))
    pb_sig = np.where(pb2l, 1, np.where(pb2s, -1, pb_sig))
    pb_sig = np.where(pb8l, 1, np.where(pb8s, -1, pb_sig))
    pb_sig = np.where(pb9, 0, pb_sig)
    pb_sig = np.where(pb5, 1, pb_sig)
    desired = np.where(pb_sig != 99, pb_sig, sig)

    size = np.clip(np.abs(edge) * (1 - tox) * (1 - aft), 0.0, 1.0)
    size = np.where(pb7l | pb7s, np.clip(size * 1.25, 0.0, 1.0), size)
    size = np.where(pb2l | pb2s, np.clip(size * 1.40, 0.0, 1.0), size)
    size = np.where(pb8l | pb8s, np.clip(size * 1.50, 0.0, 1.0), size)
    size = np.where(pb5, np.clip(size * 1.50, 0.0, 1.0), size)
    size = np.where(pb9, 0.0, size)

    # Turnover suppression: min-hold + hysteresis + switch buffer
    enter_th = float(p["enter_th"])
    exit_th = float(p["exit_th"])
    switch_buf = float(p["switch_buf"])
    min_hold = int(p["min_hold"])
    pos_dir = np.zeros(len(x), dtype=np.int8)
    hold_bars = 0
    for i in range(len(x)):
        cur = pos_dir[i - 1] if i > 0 else 0
        d = int(desired[i])
        e = float(edge[i])
        if cur == 0:
            hold_bars = 0
            if d == 1 and e > enter_th:
                pos_dir[i] = 1
            elif d == -1 and e < -enter_th:
                pos_dir[i] = -1
            else:
                pos_dir[i] = 0
            continue

        hold_bars += 1
        if cur == 1:
            if d == 0 and hold_bars >= min_hold:
                pos_dir[i] = 0
                hold_bars = 0
            elif d == -1 and hold_bars >= min_hold and e < -switch_buf:
                pos_dir[i] = -1
                hold_bars = 0
            elif hold_bars >= min_hold and e < exit_th:
                pos_dir[i] = 0
                hold_bars = 0
            else:
                pos_dir[i] = 1
        else:  # cur == -1
            if d == 0 and hold_bars >= min_hold:
                pos_dir[i] = 0
                hold_bars = 0
            elif d == 1 and hold_bars >= min_hold and e > switch_buf:
                pos_dir[i] = 1
                hold_bars = 0
            elif hold_bars >= min_hold and e > -exit_th:
                pos_dir[i] = 0
                hold_bars = 0
            else:
                pos_dir[i] = -1

    pos = np.nan_to_num(pos_dir * size, nan=0.0, posinf=0.0, neginf=0.0)
    ret = np.nan_to_num(x["ret_fwd_1m"].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    pnl, sharpe, mdd, wr, trades = compute_metrics(ret, pos, fee_bps=fee_bps, leverage=leverage)
    objective = pnl - 0.35 * mdd + 0.02 * sharpe - 0.00001 * trades
    return TrialResult(
        objective=float(objective),
        pnl_pct=float(pnl * 100.0),
        sharpe_1m=float(sharpe),
        mdd_pct=float(mdd * 100.0),
        win_rate_pct=float(wr * 100.0),
        trades=int(trades),
        avg_pos=float(np.mean(np.abs(pos))),
        fee_bps=float(fee_bps),
        params=p,
    )


def random_search(df: pd.DataFrame, n_trials: int, fee_bps: float, leverage: float, seed: int) -> TrialResult:
    rng = np.random.default_rng(seed)
    best: TrialResult | None = None
    for _ in range(n_trials):
        p = {
            "w_obi": float(rng.uniform(0.6, 2.0)),
            "w_nif": float(rng.uniform(0.6, 2.0)),
            "w_abs": float(rng.uniform(0.4, 1.8)),
            "w_tox": float(rng.uniform(0.8, 2.5)),
            "w_aft": float(rng.uniform(0.8, 2.5)),
            "w_sqz": float(rng.uniform(0.2, 1.2)),
            "base_th": float(rng.uniform(0.05, 0.30)),
            "eai_th": float(rng.uniform(1.8, 2.6)),
            "fund_th": float(rng.uniform(-0.0015, -0.0006)),
            "enter_th": float(rng.uniform(0.12, 0.42)),
            "exit_th": float(rng.uniform(0.01, 0.18)),
            "switch_buf": float(rng.uniform(0.04, 0.22)),
            "min_hold": int(rng.integers(2, 18)),
        }
        r = simulate(df, p, fee_bps=fee_bps, leverage=leverage)
        if best is None or r.objective > best.objective:
            best = r
    assert best is not None
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--fee-bps", type=float, default=2.0)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--trials", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/ensemble/metrics/top5_playbook_formula_result.json")
    args = ap.parse_args()

    sig = load_signals()
    if len(sig) < 200:
        raise RuntimeError(f"Not enough signal rows: {len(sig)}")

    start_ms = int(sig["ts"].min().timestamp() * 1000) - 120_000
    end_ms = int(sig["ts"].max().timestamp() * 1000) + 120_000
    px = fetch_binance_1m(args.symbol, start_ms=start_ms, end_ms=end_ms)

    sig["ts"] = pd.to_datetime(sig["ts"], utc=True).astype("datetime64[ns, UTC]")
    px["ts"] = pd.to_datetime(px["ts"], utc=True).astype("datetime64[ns, UTC]")

    df = pd.merge_asof(
        sig.sort_values("ts"),
        px.sort_values("ts"),
        on="ts",
        direction="backward",
    ).dropna(subset=["close"])
    df["ret_fwd_1m"] = df["close"].pct_change().shift(-1)
    df = df.dropna(subset=["ret_fwd_1m"]).reset_index(drop=True)

    best = random_search(df, n_trials=args.trials, fee_bps=args.fee_bps, leverage=args.leverage, seed=args.seed)
    best_run = simulate(df, best.params, fee_bps=args.fee_bps, leverage=args.leverage)

    # Baseline: flat (always hold)
    baseline = TrialResult(
        objective=0.0, pnl_pct=0.0, sharpe_1m=0.0, mdd_pct=0.0, win_rate_pct=0.0,
        trades=0, avg_pos=0.0, fee_bps=args.fee_bps, params={}
    )

    result = {
        "symbol": args.symbol,
        "rows": int(len(df)),
        "ts_min": str(df["ts"].min()),
        "ts_max": str(df["ts"].max()),
        "leverage": float(args.leverage),
        "best": asdict(best_run),
        "baseline_hold": asdict(baseline),
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Top5 Playbook Quant Formula Backtest ===")
    print(f"symbol={args.symbol} rows={len(df)} range=[{df['ts'].min()} ~ {df['ts'].max()}]")
    print(f"BEST pnl={best_run.pnl_pct:.3f}% sharpe={best_run.sharpe_1m:.4f} mdd={best_run.mdd_pct:.3f}% win={best_run.win_rate_pct:.2f}% trades={best_run.trades} lev={args.leverage}")
    print("BEST params:", json.dumps(best_run.params, ensure_ascii=False))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
