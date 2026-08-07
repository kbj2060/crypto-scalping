"""Gate G0' -- does the triple-barrier transformer actually predict direction, or is it just
riding a regime prior?

Motivation: in the best model's VAL backtest
(tmp/btc_deepfeat_tripbarrier_backtest_cw09_20260806/backtest_summary.json) the continuous mode
took 972 SHORT trades out of 1028 (94.5%), while OOS was a much more balanced 263L/519S. A model
that is 94.5% one-sided over a four-month split is indistinguishable, on that split, from a fixed
directional bet -- so before building anything on top of it, the fixed directional bet has to be
measured through the EXACT same mechanics.

Baselines, all traded through core/causal_futures_backtest.simulate_single_position with the same
barrier basis the model used (12-bar cumret dispersion, 288-bar lookback, TP_MULT=2.5/SL_MULT=1.2,
horizon=288, margin_fraction=0.30/leverage=3/cost=10bps):

- always_long / always_short: a decision every bar, so the simulator's non-overlap rule makes trade
  cadence come out of hold duration exactly like it does for the model -- no artificial trade-count
  matching needed.
- random_side x 5 seeds: gives a null distribution for "a coin flip through these barriers", which
  is the honest reference for a 3-class model whose CASH class only gates timing.

Also reports gross (pre-cost) mean return per trade, since that is the quantity the current design
work is actually trying to move: gross = mean_ret + roundtrip_cost_rate * notional.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.causal_futures_backtest import simulate_single_position  # noqa: E402

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
MODEL_SUMMARY_PATH = ROOT / "tmp/btc_deepfeat_tripbarrier_backtest_cw09_20260806/backtest_summary.json"
OUT_DIR = ROOT / "tmp/btc_gate_g0prime_naive_baselines_20260807"

CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT, HORIZON_BARS = 12, 288, 2.5, 1.2, 288
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")
RANDOM_SEEDS = [11, 137, 2029, 40507, 918273]


def _run(row_idx, scores, tp_moves, sl_moves, panel):
    tp, sl = tp_moves[row_idx], sl_moves[row_idx]
    finite = np.isfinite(tp) & np.isfinite(sl)
    idx, sc, tp, sl = row_idx[finite], scores[finite], tp[finite], sl[finite]
    if len(idx) == 0:
        return None
    return simulate_single_position(
        timestamps=panel["timestamp"], open_px=panel["open"].to_numpy(dtype=np.float64),
        high=panel["high"].to_numpy(dtype=np.float64), low=panel["low"].to_numpy(dtype=np.float64),
        close=panel["close"].to_numpy(dtype=np.float64), decision_indices=idx, scores=sc,
        tp_moves=tp, sl_moves=sl, upper_threshold=0.0, lower_threshold=0.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )


def _summarize(result, name, split) -> dict:
    account_cost = ROUNDTRIP_COST_RATE * MARGIN_FRACTION * LEVERAGE
    if result is None or len(result.ledger) == 0:
        return {"baseline": name, "split": split, "n_trades": 0}
    ledger, equity = result.ledger, result.equity
    running_max = np.maximum.accumulate(equity)
    reasons = ledger["reason"].value_counts().to_dict()
    rets = ledger["trade_return"].to_numpy(dtype=np.float64)
    # cost is a per-trade constant, so gross and net share the same dispersion; this t-stat asks
    # whether the PRE-COST edge is distinguishable from zero at all, which is the quantity the
    # barrier-scaling / execution-cost design levers are trying to amplify.
    std_bps = float(rets.std(ddof=1) * 10000.0) if len(rets) > 1 else float("nan")
    gross_mean_bps = float((rets.mean() + account_cost) * 10000.0)
    return {
        "baseline": name, "split": split, "n_trades": int(len(ledger)),
        "win_rate": float((ledger["trade_return"] > 0).mean()),
        "tp_rate": float(reasons.get("tp", 0) / len(ledger)),
        "mean_ret_bps": float(ledger["trade_return"].mean() * 10000.0),
        "gross_mean_ret_bps": gross_mean_bps,
        "std_ret_bps": std_bps,
        "t_stat_gross": float(gross_mean_bps / (std_bps / np.sqrt(len(rets)))) if len(rets) > 1 else None,
        "sum_ret_pct": float(ledger["trade_return"].sum() * 100.0),
        "final_equity": float(equity[-1]),
        "mdd_pct": float(((equity - running_max) / running_max).min() * 100.0),
        "long_trades": int((ledger["side"] == 1).sum()), "short_trades": int((ledger["side"] == -1).sum()),
        "median_bars_held": float(ledger["bars_held"].median()),
        "exit_reasons": reasons,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    close = panel["close"].to_numpy(dtype=np.float64)

    log_ret_1bar = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret_1bar).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
    tp_moves_all, sl_moves_all = TP_MULT * vol, SL_MULT * vol

    ts = panel["timestamp"]
    splits = {
        "val": np.flatnonzero((ts >= VAL_START).to_numpy() & (ts <= VAL_END).to_numpy()),
        "oos": np.flatnonzero((ts >= OOS_START).to_numpy() & (ts <= OOS_END).to_numpy()),
    }

    results = []
    market = {}
    for split, row_idx in splits.items():
        market[split] = {
            "buy_and_hold_pct": float((close[row_idx[-1]] / close[row_idx[0]] - 1.0) * 100.0),
            "n_bars": int(len(row_idx)),
        }
        for name, side in (("always_long", 1.0), ("always_short", -1.0)):
            scores = np.full(len(row_idx), side, dtype=np.float64)
            results.append(_summarize(_run(row_idx, scores, tp_moves_all, sl_moves_all, panel), name, split))
        for seed in RANDOM_SEEDS:
            rng = np.random.default_rng(seed)
            scores = rng.choice(np.array([-1.0, 1.0]), size=len(row_idx))
            results.append(_summarize(_run(row_idx, scores, tp_moves_all, sl_moves_all, panel), f"random_seed{seed}", split))

    rand_agg = []
    for split in splits:
        rows = [r for r in results if r["baseline"].startswith("random_seed") and r["split"] == split and r["n_trades"]]
        rand_agg.append({
            "baseline": "random_side_mean±std", "split": split,
            "n_trades": float(np.mean([r["n_trades"] for r in rows])),
            "win_rate_mean": float(np.mean([r["win_rate"] for r in rows])),
            "sum_ret_pct_mean": float(np.mean([r["sum_ret_pct"] for r in rows])),
            "sum_ret_pct_std": float(np.std([r["sum_ret_pct"] for r in rows], ddof=1)),
            "gross_mean_ret_bps_mean": float(np.mean([r["gross_mean_ret_bps"] for r in rows])),
        })

    model_rows = json.loads(MODEL_SUMMARY_PATH.read_text(encoding="utf-8")) if MODEL_SUMMARY_PATH.exists() else []
    account_cost = ROUNDTRIP_COST_RATE * MARGIN_FRACTION * LEVERAGE
    model_ref = [
        {
            "baseline": f"MODEL_{m['mode']}", "split": m["split"], "n_trades": m["n_trades"],
            "win_rate": m["win_rate"],
            "mean_ret_bps": m["mean_ret_pct"] * 100.0,
            "gross_mean_ret_bps": m["mean_ret_pct"] * 100.0 + account_cost * 10000.0,
            "sum_ret_pct": m["sum_ret_pct"], "final_equity": m["final_equity"], "mdd_pct": m["mdd_pct"],
            "long_trades": m["long_trades"], "short_trades": m["short_trades"],
        }
        for m in model_rows
    ]

    header = (f"{'baseline':<22}{'split':<6}{'trades':>7}{'win%':>7}{'grossbps':>10}{'t':>7}"
              f"{'sum%':>9}{'equity':>8}{'mdd%':>8}{'L/S':>12}")
    print(header)
    print("-" * len(header))
    for r in model_ref + [x for x in results if not x["baseline"].startswith("random_seed")]:
        if not r.get("n_trades"):
            continue
        ls = f"{r['long_trades']}/{r['short_trades']}"
        t = f"{r['t_stat_gross']:.2f}" if r.get("t_stat_gross") is not None else "n/a"
        print(f"{r['baseline']:<22}{r['split']:<6}{r['n_trades']:>7}{r['win_rate']*100:>7.1f}"
              f"{r['gross_mean_ret_bps']:>10.2f}{t:>7}{r['sum_ret_pct']:>9.1f}{r['final_equity']:>8.3f}"
              f"{r['mdd_pct']:>8.1f}{ls:>12}")
    print()
    for r in rand_agg:
        print(f"{r['baseline']:<22}{r['split']:<6}{r['n_trades']:>7.0f}{r['win_rate_mean']*100:>7.1f}"
              f"{r['gross_mean_ret_bps_mean']:>10.2f}{r['sum_ret_pct_mean']:>9.1f}  (std {r['sum_ret_pct_std']:.1f})")
    print()
    print("market context:", json.dumps(market))

    payload = {
        "config": {
            "cumret_bars": CUMRET_BARS, "vol_lookback": VOL_LOOKBACK, "tp_mult": TP_MULT,
            "sl_mult": SL_MULT, "horizon_bars": HORIZON_BARS, "margin_fraction": MARGIN_FRACTION,
            "leverage": LEVERAGE, "roundtrip_cost_rate": ROUNDTRIP_COST_RATE,
            "account_cost_bps": account_cost * 10000.0, "random_seeds": RANDOM_SEEDS,
        },
        "market": market, "model_reference": model_ref, "baselines": results, "random_aggregate": rand_agg,
    }
    (OUT_DIR / "g0prime_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR}/g0prime_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
