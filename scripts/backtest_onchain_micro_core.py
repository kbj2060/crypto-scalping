#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from scripts.optimize_duckdb_quant_formula import load_merged, calc_mdd, calc_sharpe
except ModuleNotFoundError:
    from optimize_duckdb_quant_formula import load_merged, calc_mdd, calc_sharpe


@dataclass
class CoreResult:
    pnl_pct: float
    mdd_pct: float
    sharpe: float
    trades: int
    win_rate: float
    equity: float
    params: dict


def simulate_core(m, p: dict) -> CoreResult:
    close = m["close"].to_numpy(np.float64)
    ret = np.zeros(len(m), dtype=np.float64)
    ret[:-1] = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0

    # core features only (onchain proxy + microstructure)
    over = m["overheat"].to_numpy(np.float64)              # onchain proxy
    nif = m["nif_whale"].to_numpy(np.float64)              # micro
    obi = m["obi"].to_numpy(np.float64)                    # micro
    flow = m["flow_sign"].to_numpy(np.float64)             # micro
    tox = m["shadow_toxicity_score"].to_numpy(np.float64)  # micro risk
    atr = m["atr14_pct"].to_numpy(np.float64)
    volr = m["vol1h_ratio"].to_numpy(np.float64)
    us = m["session_us"].to_numpy(np.float64)
    eu = m["session_eu"].to_numpy(np.float64)

    # score
    raw = (
        p["w_nif"] * nif
        + p["w_obi"] * (-obi)
        + p["w_flow"] * flow
        - p["w_tox"] * tox
    )
    raw = np.tanh(raw)

    # onchain/overheat asymmetry
    long_gate = (over < p["overheat_long_max"]).astype(float)
    short_boost = np.where(over > p["overheat_short_min"], p["short_boost"], 1.0)

    long_score = np.clip((raw + 1.0) / 2.0, 0.0, 1.0) * long_gate
    short_score = np.clip((-raw + 1.0) / 2.0, 0.0, 1.0) * short_boost

    # toxicity penalty
    long_score *= np.clip(1.0 - p["tox_pen"] * tox, 0.0, 1.0)
    short_score *= np.clip(1.0 - p["tox_pen"] * tox, 0.0, 1.0)

    # state
    pos = 0
    size = 0.0
    eq = 1.0
    eq_curve = [eq]
    trades = 0
    wins = 0
    h_long = False
    h_short = False
    hold = 0

    fee = p["fee_bps"] / 10000.0
    slip = p["slip_bps"] / 10000.0

    for i in range(1, len(close)):
        # session-adjusted thresholds
        sm = 0.93 if us[i] > 0.5 else (0.97 if eu[i] > 0.5 else 1.05)
        entry = p["entry"] * sm
        exit_ = p["exit"] * (0.98 if us[i] > 0.5 else 1.0)

        if not h_long and long_score[i] >= entry:
            h_long = True
        elif h_long and long_score[i] <= exit_:
            h_long = False

        if not h_short and short_score[i] >= entry:
            h_short = True
        elif h_short and short_score[i] <= exit_:
            h_short = False

        sig = 0
        if h_long and (not h_short or long_score[i] >= short_score[i]):
            sig = 1
        elif h_short and (not h_long or short_score[i] > long_score[i]):
            sig = -1

        # regime filter
        if (atr[i] < p["atr_min"]) or (volr[i] < p["volr_min"]):
            sig = 0

        strength = long_score[i] if sig == 1 else (short_score[i] if sig == -1 else 0.0)
        target = float(np.clip((strength - exit_) / max(entry - exit_, 1e-6), 0.0, 1.0))

        prev_pos = pos
        prev_size = size
        turn = 0.0

        if pos == 0 and sig != 0 and target > 0.0:
            pos = sig
            size = target * p["lev"]
            trades += 1
            hold = 0
            turn = abs(size)
        elif pos != 0:
            hold += 1
            if hold < p["min_hold"]:
                pass
            elif sig == -pos or (p["exit_on_hold"] and sig == 0):
                pos = 0
                size = 0.0
                hold = 0
                turn = abs(prev_size)
            else:
                # same-side resize
                new_size = target * p["lev"] if sig == pos else prev_size
                turn = abs(new_size - prev_size)
                size = new_size

        pnl = prev_pos * prev_size * ret[i] - turn * (fee + slip)
        eq *= (1.0 + pnl)
        eq_curve.append(eq)
        wins += int(pnl > 0.0)

    eqa = np.asarray(eq_curve, dtype=np.float64)
    pnl_pct = float((eqa[-1] - 1.0) * 100.0)
    mdd_pct = float(calc_mdd(eqa) * 100.0)
    sharpe = float(calc_sharpe(eqa))
    wr = float(100.0 * wins / max(1, len(eq_curve) - 1))
    return CoreResult(
        pnl_pct=pnl_pct,
        mdd_pct=mdd_pct,
        sharpe=sharpe,
        trades=int(trades),
        win_rate=wr,
        equity=float(eqa[-1]),
        params=dict(p),
    )


def objective(r: CoreResult) -> float:
    # Encourage activity while keeping risk in check.
    return float(
        r.pnl_pct
        - 0.20 * abs(min(0.0, r.mdd_pct))
        + 0.05 * r.sharpe
        + 0.03 * min(r.trades, 60)
        - 0.10 * max(0, 15 - r.trades)
    )


def sample_params(rng: np.random.Generator, fee_bps: float, slip_bps: float) -> dict:
    return {
        "w_nif": float(rng.uniform(0.3, 2.0)),
        "w_obi": float(rng.uniform(0.2, 1.8)),
        "w_flow": float(rng.uniform(0.1, 1.6)),
        "w_tox": float(rng.uniform(0.2, 2.0)),
        "overheat_long_max": float(rng.uniform(-0.2, 1.0)),
        "overheat_short_min": float(rng.uniform(0.5, 2.2)),
        "short_boost": float(rng.uniform(1.0, 1.8)),
        "tox_pen": float(rng.uniform(0.2, 1.2)),
        # relaxed thresholds to increase trading frequency
        "entry": float(rng.uniform(0.06, 0.28)),
        "exit": float(rng.uniform(0.02, 0.10)),
        "min_hold": int(rng.integers(1, 10)),
        "atr_min": float(rng.uniform(0.0001, 0.0020)),
        "volr_min": float(rng.uniform(0.35, 0.95)),
        "lev": float(rng.uniform(0.5, 1.5)),
        "exit_on_hold": bool(rng.random() < 0.5),
        "fee_bps": float(fee_bps),
        "slip_bps": float(slip_bps),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-csv", default="data/training_features_5m.csv")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--fee-bps", type=float, default=2.0)
    ap.add_argument("--slip-bps", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/ensemble/metrics/onchain_micro_core_backtest.json")
    args = ap.parse_args()

    m = load_merged(args.price_csv, days=int(args.days))
    if len(m) < 300:
        raise RuntimeError(f"not enough rows: {len(m)}")

    cut = int(max(200, min(len(m) - 50, round(len(m) * args.train_ratio))))
    train = m.iloc[:cut].reset_index(drop=True)
    test = m.iloc[cut:].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    best = None
    best_obj = -1e18
    for _ in range(max(1, args.trials)):
        p = sample_params(rng, args.fee_bps, args.slip_bps)
        r_tr = simulate_core(train, p)
        if objective(r_tr) > best_obj:
            best_obj = objective(r_tr)
            best = r_tr

    assert best is not None
    test_res = simulate_core(test, best.params)

    result = {
        "meta": {
            "rows_total": int(len(m)),
            "rows_train": int(len(train)),
            "rows_test": int(len(test)),
            "start": str(m["ts"].min()),
            "end": str(m["ts"].max()),
            "core_features": ["overheat(onchain-proxy)", "nif_whale", "obi", "flow_sign", "shadow_toxicity_score"],
        },
        "train": asdict(best),
        "test": asdict(test_res),
        "overfit_gap_pnl_pct": float(best.pnl_pct - test_res.pnl_pct),
        "overfit_ratio": float((test_res.pnl_pct / best.pnl_pct) if abs(best.pnl_pct) > 1e-9 else 0.0),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
