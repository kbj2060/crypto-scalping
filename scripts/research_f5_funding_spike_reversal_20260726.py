#!/usr/bin/env python3
"""F5: funding-spike settlement-time contrarian reversal.

PRE-REGISTERED in docs/mechanical_trading_research_synthesis_20260726.md S7
(committed 54e5450, before this script produced any output). Grid, windows,
cost model and gates are fixed there. Do not edit them after seeing results.

Mechanism: when |funding rate| hits an extreme historical percentile at a
settlement (00:00/08:00/16:00 UTC), the crowded side's deleveraging pressure
into settlement produces a short-horizon price reversal. This is the mirror
image of F3-A's already-killed carry bet (research_f3a_funding_carry_20260719.py:
sign persists, ride the SAME direction, 8-24h hold, price+funding combined).
F5 bets the OPPOSITE direction, only at magnitude extremes, over a much
shorter single-event window, and deliberately excludes realized funding
accrual from PnL -- mixing it back in would recreate a carry test.

  eligibility  a settlement print is a candidate only once >=90 prior prints
               (~30 days) exist, so the percentile rank uses only prints
               strictly BEFORE the current one (expanding window, causal)
  signal       |funding_rate[i]| percentile rank (vs. all PRIOR prints) >= theta
  direction    contrarian to funding sign: positive-extreme -> short,
               negative-extreme -> long
  grid         theta in {90th, 95th} x holding in {1h, 4h, 8h} = 6 variants
  pnl          price log-return only (no funding component -- see docstring)
  cost         cost1 = 10bps round trip (primary), cost3 = 30bps (robustness)
  bootstrap    blocked by entry_ts (the settlement instant, shared across all
               3 assets) rather than calendar date, so cross-asset correlation
               at the same settlement is preserved under resampling
  gate         exploration net_cost1 mean > 0 AND event-blocked bootstrap t>3
  val/oos      2025-09..12 / 2026-01..03, entered only if the gate passes
  promotion    OOS DSR >= 0.95 AND PBO <= 0.25
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.selection_stats import (  # noqa: E402
    deflated_sharpe_ratio,
    expected_max_sharpe,
    pbo_cscv,
    sharpe,
)

FUNDING_DIR = ROOT / "data/research/funding_extracted"
KLINE_5M = {
    "ETHUSDT": ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
    "BTCUSDT": ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv",
    "SOLUSDT": ROOT / "binance_data/klines/SOLUSDT/SOLUSDT-5m-api.csv",
}
ASSETS = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]

THETA_GRID = [90, 95]  # percentile
HOLDING_GRID_H = [1, 4, 8]
MIN_PRIOR_PRINTS = 90  # ~30 days at 8h spacing
COST1_ROUNDTRIP_BPS = 10.0
COST3_ROUNDTRIP_BPS = 30.0

EXPL_START = pd.Timestamp("2024-01-01", tz="UTC")
EXPL_END = pd.Timestamp("2025-08-31", tz="UTC")
VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31", tz="UTC")
GATE_T = 3.0


def load_funding(asset: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(FUNDING_DIR / asset / "*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["calc_time"])
    df["ts"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)[["ts", "last_funding_rate"]]


def load_kline(asset: str) -> pd.DataFrame:
    df = pd.read_csv(KLINE_5M[asset], usecols=["timestamp", "close"])
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[["ts", "close"]].sort_values("ts").reset_index(drop=True)


def causal_percentile_rank(abs_funding: np.ndarray) -> np.ndarray:
    """rank[i] = percentile of abs_funding[i] among abs_funding[:i] only (no lookahead).
    NaN until MIN_PRIOR_PRINTS prior observations exist."""
    n = len(abs_funding)
    rank = np.full(n, np.nan)
    for i in range(MIN_PRIOR_PRINTS, n):
        prior = abs_funding[:i]
        rank[i] = 100.0 * np.mean(prior <= abs_funding[i])
    return rank


def price_at_or_after(kline_ts_i8: np.ndarray, kline_close: np.ndarray, t_i8: int) -> float:
    idx = np.searchsorted(kline_ts_i8, t_i8, side="left")
    return float(kline_close[idx]) if idx < len(kline_ts_i8) else np.nan


def build_events(asset: str) -> pd.DataFrame:
    funding = load_funding(asset)
    kline = load_kline(asset)
    # searchsorted needs a single consistent dtype; tz-aware .values gives object
    # arrays of Timestamp, so compare on int64 nanoseconds throughout instead.
    kline_ts, kline_close = kline["ts"].astype("int64").to_numpy(), kline["close"].to_numpy()
    funding_ts_i8 = funding["ts"].astype("int64").to_numpy()

    funding["abs_f"] = funding["last_funding_rate"].abs()
    funding["pct_rank"] = causal_percentile_rank(funding["abs_f"].to_numpy())
    funding["entry_price"] = [price_at_or_after(kline_ts, kline_close, t) for t in funding_ts_i8]

    rows = []
    for h in HOLDING_GRID_H:
        exit_ts = funding["ts"] + pd.Timedelta(hours=h)
        exit_ts_i8 = exit_ts.astype("int64").to_numpy()
        exit_price = [price_at_or_after(kline_ts, kline_close, t) for t in exit_ts_i8]
        sub = funding.copy()
        sub["holding_h"] = h
        sub["exit_ts"] = exit_ts
        sub["exit_price"] = exit_price
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["entry_price", "exit_price", "pct_rank"]).reset_index(drop=True)
    out["asset"] = asset
    direction = np.where(out["last_funding_rate"] > 0, -1.0, 1.0)
    price_ret = np.log(out["exit_price"] / out["entry_price"])
    out["direction"] = direction
    out["gross_return"] = direction * price_ret
    return out


def block_bootstrap_tstat(returns: np.ndarray, block_keys: np.ndarray, n_boot: int = 3000, seed: int = 20260726) -> dict | None:
    """Bootstrap over unique block_keys (here: settlement timestamps), resampling
    whole blocks so correlated same-instant events across assets move together."""
    df = pd.DataFrame({"ret": returns, "block": block_keys})
    by_block = df.groupby("block")["ret"].apply(list)
    keys = by_block.index.to_numpy()
    if len(keys) < 20:
        return None
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        vals = np.concatenate([by_block[k] for k in sampled])
        boot_means[b] = np.mean(vals)
    observed = float(np.mean(returns))
    se = float(np.std(boot_means))
    return {"observed_mean": observed, "boot_se": se, "t_stat": float(observed / se) if se > 1e-12 else None}


def variant_stats(events: pd.DataFrame, theta: int, holding_h: int, start, end) -> dict:
    sub = events[
        (events["holding_h"] == holding_h)
        & (events["pct_rank"] >= theta)
        & (events["ts"] >= start) & (events["ts"] <= end)
    ]
    if len(sub) < 20:
        return {"n_events": int(len(sub)), "insufficient": True}

    gross = sub["gross_return"].to_numpy()
    net_cost1 = gross - COST1_ROUNDTRIP_BPS / 1e4
    net_cost3 = gross - COST3_ROUNDTRIP_BPS / 1e4
    boot = block_bootstrap_tstat(net_cost1, sub["ts"].to_numpy())

    mu, sd = float(net_cost1.mean()), float(net_cost1.std(ddof=1))
    boot_t = boot.get("t_stat") if boot else None
    events_for_t3 = float((GATE_T * sd / mu) ** 2) if mu > 0 and sd > 0 else None

    n_pos_tail = int((sub["last_funding_rate"] > 0).sum())
    n_neg_tail = int((sub["last_funding_rate"] < 0).sum())
    pos_mean = float(net_cost1[(sub["last_funding_rate"] > 0).to_numpy()].mean()) if n_pos_tail else None
    neg_mean = float(net_cost1[(sub["last_funding_rate"] < 0).to_numpy()].mean()) if n_neg_tail else None

    return {
        "n_events": int(len(sub)),
        "n_events_per_asset": {a: int((sub["asset"] == a).sum()) for a in ASSETS},
        "gross_mean": float(gross.mean()),
        "net_cost1_mean": mu,
        "net_cost1_sd": sd,
        "net_cost3_mean": float(net_cost3.mean()),
        "annualized_net_cost1_pct": mu * (365 * 24 / holding_h) * 100.0,
        "events_needed_for_t3": events_for_t3,
        "sharpe_like": sharpe(net_cost1),
        "bootstrap_net_cost1": boot,
        "positive_funding_tail": {"n": n_pos_tail, "net_cost1_mean": pos_mean},
        "negative_funding_tail": {"n": n_neg_tail, "net_cost1_mean": neg_mean},
    }


def passes_gate(stats: dict) -> bool:
    if stats.get("insufficient"):
        return False
    boot = stats.get("bootstrap_net_cost1")
    t = boot.get("t_stat") if boot else None
    return bool(stats["net_cost1_mean"] > 0 and t is not None and t > GATE_T)


def main() -> None:
    report = {
        "stage": "F5",
        "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "preregistration": "docs/mechanical_trading_research_synthesis_20260726.md S7 (commit 54e5450)",
        "grid": {"theta_percentile": THETA_GRID, "holding_hours": HOLDING_GRID_H},
        "gate": f"exploration net_cost1_mean > 0 AND event-blocked bootstrap t > {GATE_T}",
    }

    print("Loading funding + kline data per asset, building events...", flush=True)
    events = pd.concat([build_events(a) for a in ASSETS], ignore_index=True)
    report["data_coverage"] = {
        a: {
            "n_prints_eligible": int((events[(events["asset"] == a) & (events["holding_h"] == HOLDING_GRID_H[0])]).shape[0]),
        }
        for a in ASSETS
    }

    rows = []
    for theta in THETA_GRID:
        for h in HOLDING_GRID_H:
            expl = variant_stats(events, theta, h, EXPL_START, EXPL_END)
            rows.append({"theta_pct": theta, "holding_h": h, "exploration": expl, "gate_pass": passes_gate(expl)})
    report["exploration_grid"] = rows

    expl_sharpes = np.array(
        [r["exploration"].get("sharpe_like", 0.0) for r in rows], dtype=np.float64
    )
    sr_std = float(np.std(expl_sharpes, ddof=1))
    report["noise_floor"] = {
        "n_variants": len(rows),
        "trial_sharpe_std": sr_std,
        "expected_max_sharpe": expected_max_sharpe(len(rows), sr_std),
        "best_observed_sharpe": float(np.max(expl_sharpes)),
    }

    passing = [r for r in rows if r["gate_pass"]]
    report["n_variants_passing_exploration"] = len(passing)

    if not passing:
        report["verdict"] = (
            f"KILLED at exploration -- 0/{len(rows)} variants clear net_cost1_mean > 0 with t > {GATE_T}. "
            "Per the pre-registration, val/OOS are not run."
        )
    else:
        best = max(passing, key=lambda r: r["exploration"]["net_cost1_mean"])
        theta, h = best["theta_pct"], best["holding_h"]
        val = variant_stats(events, theta, h, VAL_START, VAL_END)
        oos = variant_stats(events, theta, h, OOS_START, OOS_END)

        oos_sub = events[
            (events["holding_h"] == h) & (events["pct_rank"] >= theta)
            & (events["ts"] >= OOS_START) & (events["ts"] <= OOS_END)
        ]
        oos_net = oos_sub["gross_return"].to_numpy() - COST1_ROUNDTRIP_BPS / 1e4
        dsr = deflated_sharpe_ratio(oos_net, expl_sharpes)

        # PBO across all 6 variants, using OOS-window-truncated net returns per variant,
        # aligned to a common event-time index within the combined 3-asset event stream.
        variant_series = {}
        for th in THETA_GRID:
            for hh in HOLDING_GRID_H:
                s = events[(events["holding_h"] == hh) & (events["pct_rank"] >= th) & (events["ts"] <= OOS_END)]
                s = s.set_index("ts")["gross_return"] - COST1_ROUNDTRIP_BPS / 1e4
                variant_series[f"th{th}_h{hh}"] = s.groupby(level=0).mean()
        net_df = pd.DataFrame(variant_series).sort_index().dropna(how="all")
        net_df = net_df.fillna(0.0)  # a variant with no event that instant contributes 0, not missing
        pbo = pbo_cscv(net_df.to_numpy(), n_splits=10) if len(net_df) >= 30 else {"pbo": None, "note": "insufficient combined event timestamps"}

        report["selected_variant"] = {"theta_pct": theta, "holding_h": h}
        report["val"] = val
        report["oos"] = oos
        report["oos_deflated_sharpe"] = dsr
        report["pbo_cscv"] = pbo
        promoted = bool(dsr.get("passes_95") and pbo.get("pbo") is not None and pbo["pbo"] <= 0.25)
        report["promotion_pass"] = promoted
        report["verdict"] = (
            "PROMOTION CRITERIA MET" if promoted
            else "gate passed at exploration but promotion criteria (OOS DSR >= 0.95 AND PBO <= 0.25) not met"
        )

    out = Path("data/ensemble/metrics/f5_funding_spike_reversal_20260726.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
