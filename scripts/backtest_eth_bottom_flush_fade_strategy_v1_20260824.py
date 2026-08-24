#!/usr/bin/env python3
"""ETH bottom-flush-fade strategy v1 -- causal backtest of the surviving bottom combos.

Pre-registered contract (locked before running):
docs/experiments/eth_bottom_flush_fade_strategy_v1_20260824.md

Long-only. Trigger = OR of the three ~43%-precision bottom combos (orthogonal_combo,
sweep x flow, SMT x flow -- formulas verbatim from their validated scripts). Entry is a
resting limit delta below the trigger close (calibrated to the signals' measured adverse
excursion), NOT a market order at fire -- this is the previously-flagged untried path.
Quantile-only calibration on CAL 2024-01..2025-08; zero grids. Standard costs only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators  # noqa: E402

ETH_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC_PATH = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"

L_LIMIT = 24        # limit-order lifetime (bars), fixed a priori
H_HOLD = 48         # max hold (bars), fixed a priori
SL_FLOOR = 0.004
COSTS_RT = {"taker10bp": 10e-4, "maker6.2bp": 6.2e-4}
CAL = ("2024-01-01", "2025-08-31 23:59:59")
DEV = ("2025-09-01", "2026-02-17 23:59:59")
FRESH = ("2026-03-01", "2026-08-17 23:59:59")


def load_frames() -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]
    f = pd.read_csv(ETH_PATH, usecols=cols, parse_dates=["timestamp"])
    f = f.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    f = compute_indicators(f).reset_index(drop=True)

    delta = 2.0 * f["taker_buy_base"] - f["volume"]
    f["delta_z"] = (delta - delta.rolling(288, min_periods=288).mean()) / \
        delta.rolling(288, min_periods=288).std().replace(0.0, np.nan)

    swing_low_prior = f["low"].rolling(48, min_periods=48).min().shift(1)
    f["sweep_low"] = (f["low"] < swing_low_prior) & (f["close"] > swing_low_prior)
    eth_break_low = f["low"] < swing_low_prior

    btc = pd.read_csv(BTC_PATH, usecols=["timestamp", "low"], parse_dates=["timestamp"])
    btc = btc.sort_values("timestamp").drop_duplicates("timestamp", keep="last").rename(columns={"low": "btc_low"})
    f = f.merge(btc, on="timestamp", how="left")
    btc_swing_low = f["btc_low"].rolling(48, min_periods=48).min().shift(1)
    btc_holds = (f["btc_low"] > btc_swing_low).fillna(False)
    f["smt_raw_bottom"] = eth_break_low & btc_holds

    flow = f["delta_z"] <= -2.0
    t1 = (f["p_fast"] <= 0.10) & (f["p_slow"] <= 0.10) & flow
    t2 = f["sweep_low"] & flow
    t3 = f["smt_raw_bottom"] & flow
    f["trigger"] = (t1 | t2 | t3).fillna(False)
    for name, t in (("T1_orthogonal", t1), ("T2_sweep_flow", t2), ("T3_smt_flow", t3)):
        f[name] = t.fillna(False)
    return f


def run_engine(f: pd.DataFrame, start: str, end: str, delta_frac: float, tp: float, sl: float) -> pd.DataFrame:
    ts = f["timestamp"]
    idx = np.flatnonzero(((ts >= start) & (ts <= end)).to_numpy())
    o, h, l, c = (f[k].to_numpy() for k in ("open", "high", "low", "close"))
    trig = f["trigger"].to_numpy()
    trades = []
    i = idx[0]
    end_i = idx[-1]
    while i <= end_i:
        if trig[i]:
            limit = c[i] * (1.0 - delta_frac)
            fill_j = -1
            for j in range(i + 1, min(i + 1 + L_LIMIT, end_i + 1)):
                if l[j] <= limit:
                    fill_j = j
                    break
            if fill_j < 0:
                i = min(i + 1 + L_LIMIT, end_i + 1)
                continue
            e = min(o[fill_j], limit)
            tp_px, sl_px = e * (1.0 + tp), e * (1.0 - sl)
            exit_px, exit_j, reason = None, None, None
            for j in range(fill_j + 1, min(fill_j + 1 + H_HOLD, end_i + 1)):
                if l[j] <= sl_px:                      # SL first on ambiguous bars (conservative)
                    exit_px, exit_j, reason = sl_px if o[j] > sl_px else o[j], j, "SL"
                    break
                if h[j] >= tp_px:
                    exit_px, exit_j, reason = tp_px if o[j] < tp_px else o[j], j, "TP"
                    break
            if exit_px is None:
                exit_j = min(fill_j + H_HOLD, end_i)
                exit_px, reason = c[exit_j], "TIME"
            trades.append({"trig_i": i, "fill_i": fill_j, "exit_i": exit_j, "entry": e,
                           "exit": exit_px, "reason": reason, "gross": exit_px / e - 1.0,
                           "ts": ts.iloc[i]})
            i = exit_j + 1
        else:
            i += 1
    return pd.DataFrame(trades)


def report(name: str, f: pd.DataFrame, start: str, end: str, tr: pd.DataFrame) -> None:
    ts = f["timestamp"]
    w = f[(ts >= start) & (ts <= end)]
    bh = w["close"].iloc[-1] / w["close"].iloc[0] - 1.0
    bench = max(bh, -bh)
    n = len(tr)
    n_trig = int(f.loc[(ts >= start) & (ts <= end), "trigger"].sum())
    if n == 0:
        print(f"{name}: triggers={n_trig} trades=0")
        return
    exposure = (tr["exit_i"] - tr["fill_i"]).sum() / max(len(w), 1)
    line = (f"{name}: triggers={n_trig} trades={n} fill_rate~{n / max(n_trig, 1) * 100:.0f}%(engine-seq) "
            f"win={((tr['gross'] > 0).mean()) * 100:.1f}% gross_avg={tr['gross'].mean() * 1e4:+.1f}bp "
            f"exposure={exposure * 100:.1f}% reasons={tr['reason'].value_counts().to_dict()}")
    print(line)
    for cname, cost in COSTS_RT.items():
        net = tr["gross"] - cost
        tot = net.sum()
        print(f"   @{cname}: net_avg={net.mean() * 1e4:+.1f}bp/tr total={tot * 100:+.2f}% "
              f"vs bench(max_always={bench * 100:+.1f}%) inc={(tot - bench) * 100:+.2f}%")


def main() -> None:
    f = load_frames()
    assert f["timestamp"].max() >= pd.Timestamp("2026-08-18"), "klines stale"
    ts = f["timestamp"]

    # ---- calibration (quantiles only) ----
    cal_mask = (ts >= CAL[0]) & (ts <= CAL[1])
    trig_pos = np.flatnonzero((f["trigger"] & cal_mask).to_numpy())
    lows, closes = f["low"].to_numpy(), f["close"].to_numpy()
    mae24 = np.array([(closes[t] - lows[t + 1:t + 1 + L_LIMIT].min()) / closes[t]
                      for t in trig_pos if t + 1 + L_LIMIT < len(f)])
    delta_frac = float(np.median(mae24))

    cal_probe = run_engine(f, *CAL, delta_frac, tp=9.99, sl=9.99)  # no TP/SL: raw MFE/MAE via TIME exits
    highs = f["high"].to_numpy()
    mfe, mae = [], []
    for _, t in cal_probe.iterrows():
        j0, j1 = int(t["fill_i"]) + 1, int(t["fill_i"]) + H_HOLD
        if j1 >= len(f):
            continue
        mfe.append(highs[j0:j1 + 1].max() / t["entry"] - 1.0)
        mae.append(1.0 - lows[j0:j1 + 1].min() / t["entry"])
    tp = float(np.median(mfe))
    sl = max(float(np.percentile(mae, 75)), SL_FLOOR)
    print(f"CAL fires={len(trig_pos)} -> delta={delta_frac * 100:.3f}% | probe fills={len(cal_probe)} "
          f"tp=P50(MFE48)={tp * 100:.3f}% sl=P75(MAE48)={sl * 100:.3f}%\n")

    for name, (s, e) in (("CAL(in-sample)", CAL), ("DEV(consumed)", DEV), ("FRESH", FRESH)):
        tr = run_engine(f, s, e, delta_frac, tp, sl)
        report(name, f, s, e, tr)

    # pre-registered verdict
    tr_fresh = run_engine(f, *FRESH, delta_frac, tp, sl)
    net10 = (tr_fresh["gross"] - COSTS_RT["taker10bp"]).sum() if len(tr_fresh) else float("nan")
    ok = len(tr_fresh) >= 30 and net10 > 0
    print(f"\nVERDICT (pre-registered): FRESH n={len(tr_fresh)} (>=30) net@10bp={net10 * 100:+.2f}% (>0) "
          f"-> {'PASS(strategy survives)' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
