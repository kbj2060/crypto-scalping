#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Result:
    pnl_pct: float
    mdd_pct: float
    sharpe: float
    trades: int
    win_rate: float
    equity: float
    params: dict


def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def _z(s: pd.Series, w: int = 96) -> pd.Series:
    mu = s.rolling(w, min_periods=max(12, w // 4)).mean()
    sd = s.rolling(w, min_periods=max(12, w // 4)).std().replace(0, np.nan)
    return ((s - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _mdd(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    dd = eq / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min()) * 100.0


def _sharpe(eq: np.ndarray, bars_per_year: int = 365 * 24 * 12) -> float:
    if len(eq) < 10:
        return 0.0
    r = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    s = float(np.std(r))
    if s < 1e-12:
        return 0.0
    return float(np.mean(r) / s * math.sqrt(bars_per_year))


def load_2025_native(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    cols = [
        "open", "high", "low", "close", "volume", "quote_volume",
        "smart_money_flow", "net_taker_ratio", "oi_change_rate", "last_funding_rate",
        "ofti", "volatility_z", "trade_intensity", "big_trade_ratio",
        "amihud_illiquidity_z", "evt_tail_flag", "jump_z", "squeeze_power",
    ]
    for c in cols:
        df[c] = pd.to_numeric(df.get(c, 0.0), errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume", "quote_volume"]).copy()

    # Native feature engineering
    ofti_scale = float(df["ofti"].abs().quantile(0.99) + 1e-8)
    smf_scale = float(df["smart_money_flow"].abs().quantile(0.99) + 1e-8)
    oi_scale = float(df["oi_change_rate"].abs().quantile(0.99) + 1e-8)
    sq_scale = float(df["squeeze_power"].abs().quantile(0.95) + 1e-8)

    df["flow"] = _clip(df["net_taker_ratio"].fillna(0.0).to_numpy(), -1.0, 1.0)
    df["obi"] = np.tanh(df["ofti"].fillna(0.0) / ofti_scale)
    df["nif"] = np.tanh(df["smart_money_flow"].fillna(0.0) / smf_scale)

    ti_z = _z(df["trade_intensity"].fillna(0.0), 96)
    btr = df["big_trade_ratio"].fillna(0.0)
    volz = df["volatility_z"].fillna(0.0)
    amz = df["amihud_illiquidity_z"].fillna(0.0)
    jump = df["jump_z"].fillna(0.0).abs()
    tail = df["evt_tail_flag"].fillna(0.0)

    df["s_abs"] = _clip((0.9 * ti_z + 0.5 * btr - 0.3 * np.abs(df["flow"])) / 3.0 + 0.5, 0.0, 1.0)
    df["s_tox"] = _clip((0.8 * volz + 0.6 * amz + 0.4 * jump) / 6.0 + 0.5, 0.0, 1.0)
    df["s_qc"] = _clip((0.9 * tail + 0.5 * jump + 0.5 * np.maximum(volz, 0.0)) / 4.0, 0.0, 1.0)
    df["p_aft"] = _clip(0.4 * tail + 0.6 * _sigmoid((jump - 1.5).to_numpy()), 0.0, 1.0)

    df["liq"] = _clip(df["oi_change_rate"].fillna(0.0) / oi_scale, -1.0, 1.0)
    over = 0.6 * _z(df["oi_change_rate"].fillna(0.0), 96) + 0.4 * _z(df["last_funding_rate"].fillna(0.0), 96)
    df["overheat"] = over
    sq = np.tanh(df["squeeze_power"].fillna(0.0) / sq_scale)
    df["eai"] = _clip(0.6 * np.abs(sq) + 0.2 * _clip((ti_z + 3.0) / 6.0, 0.0, 1.0) + 0.2 * _clip((volz + 3.0) / 6.0, 0.0, 1.0), 0.0, 1.0)

    prev_close = df["close"].shift(1).fillna(df["close"])
    tr = np.maximum(
        (df["high"] - df["low"]).abs(),
        np.maximum((df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()),
    )
    atr = (tr.rolling(14, min_periods=5).mean() / np.maximum(df["close"], 1e-8)).fillna(0.0)
    v1h = df["volume"].rolling(12, min_periods=6).mean().fillna(0.0)
    v24h = df["volume"].rolling(288, min_periods=24).mean().ffill().fillna(1.0)
    volr = (v1h / np.maximum(v24h, 1e-8)).fillna(0.0)
    df["atr"] = atr
    df["volr"] = volr

    return df


def simulate(df: pd.DataFrame, p: dict, fee_bps: float, slip_bps: float) -> Result:
    flow = df["flow"].to_numpy(np.float64)
    obi = df["obi"].to_numpy(np.float64)
    nif = df["nif"].to_numpy(np.float64)
    s_abs = df["s_abs"].to_numpy(np.float64)
    s_tox = df["s_tox"].to_numpy(np.float64)
    s_qc = df["s_qc"].to_numpy(np.float64)
    p_aft = df["p_aft"].to_numpy(np.float64)
    liq = df["liq"].to_numpy(np.float64)
    over = df["overheat"].to_numpy(np.float64)
    eai = df["eai"].to_numpy(np.float64)
    atr = df["atr"].to_numpy(np.float64)
    volr = df["volr"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)

    # MSAF-inspired native score
    fas = p["w_flow"] * np.tanh(2.0 * flow) + p["w_obi"] * np.tanh(3.0 * obi) + p["w_nif"] * np.clip(nif, -1, 1)
    sgn = np.sign(fas)
    sis = p["w_abs"] * s_abs + p["w_tox"] * s_tox * sgn - p["w_vpin"] * s_qc
    lcs = p["w_liq"] * liq * (1.0 + p["w_aft"] * p_aft)
    roa = -p["gamma"] * np.tanh(p["beta"] * over) * sgn

    psi = (fas + sis + lcs + roa) * (1.0 + p["w_eai"] * eai)
    psi = np.tanh(psi)

    # dead-zone + risk gates
    tradable = (atr >= p["atr_min"]) & (volr >= p["volr_min"]) & (s_qc <= p["qc_gate"])
    psi = np.where(tradable, psi, 0.0)
    psi = np.where(s_tox >= p["tox_veto"], 0.0, psi)

    # state machine
    pos = np.zeros(len(df), dtype=np.float64)
    hold = 0
    last_nonzero = 0.0
    for i in range(1, len(df)):
        prev = pos[i - 1]
        x = float(psi[i])
        if prev == 0.0:
            hold = 0
            if abs(x) >= p["entry"]:
                pos[i] = np.sign(x)
                last_nonzero = pos[i]
            else:
                pos[i] = 0.0
        else:
            hold += 1
            if hold < p["min_hold"]:
                pos[i] = prev
                continue
            if abs(x) <= p["exit"]:
                pos[i] = 0.0
            elif np.sign(x) != np.sign(prev) and abs(x) >= p["entry_flip"]:
                pos[i] = np.sign(x)
                hold = 0
                last_nonzero = pos[i]
            else:
                pos[i] = prev

    size = np.clip(np.abs(psi) * p["lev"], 0.0, p["lev"])
    exposure = pos * size

    ret = np.zeros(len(df), dtype=np.float64)
    ret[:-1] = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0
    fee = fee_bps / 10000.0
    slip = slip_bps / 10000.0
    turn = np.abs(np.diff(exposure, prepend=0.0))
    pnl = exposure * ret - turn * (fee + slip)
    eq = np.cumprod(1.0 + pnl)

    # trades/win
    trades = int(np.sum((np.abs(np.diff(pos, prepend=0.0)) > 0) & (pos != 0)))
    wins = int(np.sum(pnl > 0))
    wr = float(np.mean(pnl > 0)) * 100.0

    return Result(
        pnl_pct=float((eq[-1] - 1.0) * 100.0),
        mdd_pct=_mdd(eq),
        sharpe=_sharpe(eq),
        trades=trades,
        win_rate=wr,
        equity=float(eq[-1]),
        params=dict(p),
    )


def sample_params(rng: np.random.Generator) -> dict:
    return {
        "w_flow": float(rng.uniform(0.10, 0.70)),
        "w_obi": float(rng.uniform(0.10, 0.70)),
        "w_nif": float(rng.uniform(0.10, 0.70)),
        "w_abs": float(rng.uniform(0.05, 0.50)),
        "w_tox": float(rng.uniform(0.05, 0.50)),
        "w_vpin": float(rng.uniform(0.02, 0.40)),
        "w_liq": float(rng.uniform(0.05, 0.50)),
        "w_aft": float(rng.uniform(0.10, 1.00)),
        "w_eai": float(rng.uniform(0.05, 0.50)),
        "gamma": float(rng.uniform(0.05, 0.40)),
        "beta": float(rng.uniform(0.6, 2.8)),
        "entry": float(rng.uniform(0.12, 0.45)),
        "exit": float(rng.uniform(0.03, 0.20)),
        "entry_flip": float(rng.uniform(0.18, 0.60)),
        "min_hold": int(rng.integers(2, 20)),
        "atr_min": float(rng.uniform(0.0004, 0.006)),
        "volr_min": float(rng.uniform(0.5, 1.3)),
        "qc_gate": float(rng.uniform(0.5, 0.95)),
        "tox_veto": float(rng.uniform(0.75, 0.98)),
        "lev": float(rng.uniform(0.4, 1.4)),
    }


def objective(r_train: Result, r_val: Result) -> float:
    gap = abs(r_train.pnl_pct - r_val.pnl_pct)
    return float(
        r_val.pnl_pct
        - 0.45 * abs(min(0.0, r_val.mdd_pct))
        + 0.06 * r_val.sharpe
        - 0.15 * gap
        - 0.02 * max(0, 12 - r_val.trades)
    )


def tune(df_train: pd.DataFrame, trials: int, fee_bps: float, slip_bps: float, seed: int) -> Result:
    n = len(df_train)
    cut = int(n * 0.7)
    dtr = df_train.iloc[:cut].reset_index(drop=True)
    dva = df_train.iloc[cut:].reset_index(drop=True)
    rng = np.random.default_rng(seed)

    best: Result | None = None
    best_val: Result | None = None
    best_score = -1e18
    for _ in range(max(1, trials)):
        p = sample_params(rng)
        rtr = simulate(dtr, p, fee_bps, slip_bps)
        rva = simulate(dva, p, fee_bps, slip_bps)
        sc = objective(rtr, rva)
        if sc > best_score:
            best_score = sc
            best = simulate(df_train, p, fee_bps, slip_bps)
            best_val = rva
    assert best is not None
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/splits/year_oos/rl_training_2025_m7.csv")
    ap.add_argument("--trials", type=int, default=240)
    ap.add_argument("--fee-bps", type=float, default=2.0)
    ap.add_argument("--slip-bps", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/ensemble/metrics/rl2025_native_formula_result.json")
    args = ap.parse_args()

    df = load_2025_native(args.csv)
    n = len(df)
    cut = int(n * 0.7)
    train = df.iloc[:cut].reset_index(drop=True)
    test = df.iloc[cut:].reset_index(drop=True)

    best_train = tune(train, args.trials, args.fee_bps, args.slip_bps, args.seed)
    test_res = simulate(test, best_train.params, args.fee_bps, args.slip_bps)

    result = {
        "meta": {
            "rows": n,
            "rows_train": len(train),
            "rows_test": len(test),
            "start": str(df["ts"].min()),
            "end": str(df["ts"].max()),
            "fee_bps": args.fee_bps,
            "slip_bps": args.slip_bps,
            "trials": args.trials,
            "source_csv": args.csv,
        },
        "train": asdict(best_train),
        "test": asdict(test_res),
        "overfit_gap_pnl_pct": float(best_train.pnl_pct - test_res.pnl_pct),
        "overfit_ratio": float((test_res.pnl_pct / best_train.pnl_pct) if abs(best_train.pnl_pct) > 1e-9 else 0.0),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
