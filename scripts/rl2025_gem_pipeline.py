#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CORE5 = [
    "smart_money_flow",
    "net_taker_ratio",
    "oi_change_rate",
    "last_funding_rate",
    "volatility_z",
]


@dataclass
class BTResult:
    pnl_pct: float
    mdd_pct: float
    sharpe: float
    trades: int
    win_rate: float
    equity_final: float


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


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    need = CORE5 + ["open", "high", "low", "close", "volume", "quote_volume"]
    for c in need:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"] + CORE5).reset_index(drop=True)
    return df


def build_labels(df: pd.DataFrame, horizon: int = 6, tp: float = 0.006, sl: float = 0.006) -> pd.DataFrame:
    d = df.copy()
    close = d["close"].to_numpy(np.float64)
    high = d["high"].to_numpy(np.float64)
    low = d["low"].to_numpy(np.float64)
    n = len(d)

    # L1: 30m forward return sign
    fwd = np.full(n, np.nan, dtype=np.float64)
    fwd[:-horizon] = close[horizon:] / np.maximum(close[:-horizon], 1e-12) - 1.0
    y1 = np.where(fwd > 0.0, 1.0, np.where(fwd < 0.0, -1.0, 0.0))

    # L2: MFE-MAE sign over next horizon
    y2 = np.zeros(n, dtype=np.float64)
    for i in range(n - horizon):
        c0 = close[i]
        h = float(np.max(high[i + 1 : i + 1 + horizon]))
        l = float(np.min(low[i + 1 : i + 1 + horizon]))
        mfe = h / max(c0, 1e-12) - 1.0
        mae = l / max(c0, 1e-12) - 1.0
        s = mfe + mae  # mae is negative
        y2[i] = 1.0 if s > 0 else (-1.0 if s < 0 else 0.0)

    # L3: triple barrier (first hit)
    y3 = np.zeros(n, dtype=np.float64)
    for i in range(n - horizon):
        c0 = close[i]
        up = c0 * (1.0 + tp)
        dn = c0 * (1.0 - sl)
        lab = 0.0
        for j in range(i + 1, i + 1 + horizon):
            if high[j] >= up:
                lab = 1.0
                break
            if low[j] <= dn:
                lab = -1.0
                break
        if lab == 0.0:
            r = close[i + horizon] / max(c0, 1e-12) - 1.0
            lab = 1.0 if r > 0 else (-1.0 if r < 0 else 0.0)
        y3[i] = lab

    d["y_l1"] = y1
    d["y_l2"] = y2
    d["y_l3"] = y3
    return d


def fit_linear_score(train: pd.DataFrame, label_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = train[CORE5].to_numpy(np.float64)
    y = train[label_col].to_numpy(np.float64)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    xz = (x - mu) / sd
    # robust directional weights: feature-label correlation
    w = np.array([np.corrcoef(xz[:, i], y)[0, 1] if np.std(xz[:, i]) > 1e-8 else 0.0 for i in range(xz.shape[1])], dtype=np.float64)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    if np.linalg.norm(w) < 1e-8:
        w = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    w = w / max(np.linalg.norm(w), 1e-8)
    return w, mu, sd


def make_score(df: pd.DataFrame, w: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    x = df[CORE5].to_numpy(np.float64)
    xz = (x - mu) / sd
    s = xz @ w
    return np.tanh(s)


def backtest_score(
    df: pd.DataFrame,
    score: np.ndarray,
    entry: float,
    exit_: float,
    min_hold: int,
    fee_bps: float,
    slip_bps: float,
    lev: float,
) -> BTResult:
    close = df["close"].to_numpy(np.float64)
    ret = np.zeros(len(df), dtype=np.float64)
    ret[:-1] = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0
    fee = fee_bps / 10000.0
    slip = slip_bps / 10000.0

    pos = np.zeros(len(df), dtype=np.float64)
    hold = 0
    for i in range(1, len(df)):
        prev = pos[i - 1]
        s = float(score[i])
        if prev == 0.0:
            hold = 0
            pos[i] = np.sign(s) if abs(s) >= entry else 0.0
        else:
            hold += 1
            if hold < min_hold:
                pos[i] = prev
            elif abs(s) <= exit_:
                pos[i] = 0.0
            elif np.sign(s) != np.sign(prev) and abs(s) >= entry:
                pos[i] = np.sign(s)
                hold = 0
            else:
                pos[i] = prev

    exposure = pos * lev
    turn = np.abs(np.diff(exposure, prepend=0.0))
    pnl = exposure * ret - turn * (fee + slip)
    eq = np.cumprod(1.0 + pnl)
    trades = int(np.sum((np.abs(np.diff(pos, prepend=0.0)) > 0) & (pos != 0)))
    win_rate = float(np.mean(pnl > 0.0)) * 100.0
    return BTResult(
        pnl_pct=float((eq[-1] - 1.0) * 100.0),
        mdd_pct=_mdd(eq),
        sharpe=_sharpe(eq),
        trades=trades,
        win_rate=win_rate,
        equity_final=float(eq[-1]),
    )


def choose_exec_on_val(df_val: pd.DataFrame, score: np.ndarray, fee_bps: float, slip_bps: float) -> tuple[float, float, int, float]:
    best = None
    for entry in [0.12, 0.16, 0.20, 0.24, 0.30]:
        for exit_ in [0.04, 0.06, 0.08, 0.10, 0.12]:
            if exit_ >= entry:
                continue
            for mh in [2, 4, 6, 8]:
                r = backtest_score(df_val, score, entry, exit_, mh, fee_bps, slip_bps, lev=1.0)
                obj = r.pnl_pct - 0.5 * abs(min(0.0, r.mdd_pct)) + 0.05 * r.sharpe - 0.02 * max(0, 10 - r.trades)
                if best is None or obj > best[0]:
                    best = (obj, entry, exit_, mh)
    assert best is not None
    return best[1], best[2], int(best[3]), float(best[0])


def evaluate_window(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    fee_bps: float,
    slip_bps: float,
) -> dict:
    label_cols = ["y_l1", "y_l2", "y_l3"]
    candidates = []
    for lc in label_cols:
        w, mu, sd = fit_linear_score(train, lc)
        s_val = make_score(val, w, mu, sd)
        entry, exit_, mh, obj = choose_exec_on_val(val, s_val, fee_bps, slip_bps)
        rv = backtest_score(val, s_val, entry, exit_, mh, fee_bps, slip_bps, lev=1.0)
        candidates.append((obj, lc, w, mu, sd, entry, exit_, mh, rv))
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_lc, w, mu, sd, entry, exit_, mh, rv = candidates[0]
    s_test = make_score(test, w, mu, sd)
    rt = backtest_score(test, s_test, entry, exit_, mh, fee_bps, slip_bps, lev=1.0)
    return {
        "best_label": best_lc,
        "entry": float(entry),
        "exit": float(exit_),
        "min_hold": int(mh),
        "val": rt.__class__(**rv.__dict__).__dict__,
        "test": rt.__dict__,
        "weights": {k: float(v) for k, v in zip(CORE5, w)},
    }


def run_walkforward(
    df: pd.DataFrame,
    fee_bps: float,
    slip_bps: float,
    months_train: int = 6,
    months_val: int = 3,
    months_test: int = 3,
) -> list[dict]:
    bars_per_month = 30 * 24 * 12
    bt = months_train * bars_per_month
    bv = months_val * bars_per_month
    bs = months_test * bars_per_month
    n = len(df)
    out = []
    start = 0
    fold = 0
    while start + bt + bv + bs <= n:
        fold += 1
        tr = df.iloc[start : start + bt].reset_index(drop=True)
        va = df.iloc[start + bt : start + bt + bv].reset_index(drop=True)
        te = df.iloc[start + bt + bv : start + bt + bv + bs].reset_index(drop=True)
        r = evaluate_window(tr, va, te, fee_bps, slip_bps)
        r["fold"] = fold
        r["range"] = {
            "train": [str(tr["ts"].iloc[0]), str(tr["ts"].iloc[-1])],
            "val": [str(va["ts"].iloc[0]), str(va["ts"].iloc[-1])],
            "test": [str(te["ts"].iloc[0]), str(te["ts"].iloc[-1])],
        }
        out.append(r)
        start += bs
    return out


def aggregate_folds(folds: list[dict]) -> dict:
    if not folds:
        return {}
    pnl = [f["test"]["pnl_pct"] for f in folds]
    mdd = [f["test"]["mdd_pct"] for f in folds]
    shp = [f["test"]["sharpe"] for f in folds]
    trd = [f["test"]["trades"] for f in folds]
    all_pos = all(x > 0 for x in pnl)
    return {
        "folds": len(folds),
        "mean_test_pnl_pct": float(np.mean(pnl)),
        "median_test_pnl_pct": float(np.median(pnl)),
        "mean_test_mdd_pct": float(np.mean(mdd)),
        "mean_test_sharpe": float(np.mean(shp)),
        "mean_test_trades": float(np.mean(trd)),
        "all_folds_positive_pnl": bool(all_pos),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/splits/year_oos/rl_training_2025_m7.csv")
    ap.add_argument("--fee-bps", type=float, default=4.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--out", default="data/ensemble/metrics/rl2025_gem_pipeline_result.json")
    args = ap.parse_args()

    df = load_data(args.csv)
    df = build_labels(df, horizon=6, tp=0.006, sl=0.006)

    folds = run_walkforward(df, args.fee_bps, args.slip_bps, 6, 3, 3)
    if not folds:
        # fallback for single-year data
        folds = run_walkforward(df, args.fee_bps, args.slip_bps, 4, 2, 2)

    summary = aggregate_folds(folds)

    # cost scenarios on same protocol
    cost_scenarios = []
    for fb, sb in [(2.0, 1.0), (4.0, 2.0), (5.0, 3.0), (6.0, 4.0)]:
        fs = run_walkforward(df, fb, sb, 6, 3, 3)
        if not fs:
            fs = run_walkforward(df, fb, sb, 4, 2, 2)
        ag = aggregate_folds(fs)
        cost_scenarios.append({"fee_bps": fb, "slip_bps": sb, **ag})

    # pass checklist
    checklist = {
        "oos_sharpe_gt_0p6_fee6bps": bool(any((c.get("fee_bps") == 6.0 and c.get("mean_test_sharpe", -999) > 0.6) for c in cost_scenarios)),
        "all_folds_positive": bool(summary.get("all_folds_positive_pnl", False)),
        "mdd_lt_8pct": bool(abs(summary.get("mean_test_mdd_pct", -999)) < 8.0),
        "stable_under_cost": bool(all(c.get("mean_test_pnl_pct", -999) > -3.0 for c in cost_scenarios)),
    }

    result = {
        "meta": {
            "source_csv": args.csv,
            "rows": int(len(df)),
            "start": str(df["ts"].min()),
            "end": str(df["ts"].max()),
            "core5": CORE5,
        },
        "walkforward": folds,
        "summary": summary,
        "cost_scenarios": cost_scenarios,
        "checklist": checklist,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
