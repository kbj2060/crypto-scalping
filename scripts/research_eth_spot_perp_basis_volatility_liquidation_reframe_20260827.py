#!/usr/bin/env python3
"""spot-perp basis, reframed in the direction the literature actually predicted (2026-08-20 audit
in research_eth_spot_perp_basis_direction_cheap_gate_20260820.py): Schmeling/Schrimpf/Todorov
"Crypto Carry" (BIS WP1087/Management Science) and He/Manela/Ross/von Wachter (arXiv:2212.06888)
find basis predicts NOT direction but (a) future realized VOLATILITY and (b) SHORT LIQUIDATIONS
specifically (crowded-long-carry-trade unwind mechanism) -- "standardized carry +10% -> next-month
short liquidations +22%, significantly predicts future volatility" per this repo's own literature
audit doc (docs/experiments/eth_candidate_spot_perp_basis_direction_cheap_gate_20260820.md).

Same basis construction (_load_basis_frame, basis_raw/basis_z48/basis_roc12) and same IC-scan
methodology (Spearman rank IC + vectorized permutation-null significance, N=2000) as the original
direction cheap-gate script -- reused verbatim, not re-derived. Only the TARGET changes.

PART A (VOLATILITY) -- full rigor, same 3-split (TRAIN/VAL/OOS) x 4-horizon design as the original
direction test. Target: forward realized volatility sqrt(sum of squared forward log returns over
the horizon) -- well-defined at every horizon including h=1 (collapses to |next-bar log return|).
Data: same full 2024-01-01..2026-03-31 coverage as the original basis frame.

PART B (LIQUIDATION CROWDING) -- EXPLORATORY, not confirmatory, and disclosed as such: target is
forward short_usd_1m (USD short-liquidation volume, data/live/tail_risk.duckdb::tail_risk_1m,
aggregated to this script's 5-min bar grid to match basis's granularity). This table's own
documented reliable window starts 2026-07-18 (reference_clean_data_locations_20260823 memory:
pre-07-18 is a flagged permanent-deficiency window) -- so this part covers at most ~1 month
(2026-07-18 through this local duckdb copy's last row), one combined window, NOT a 3-split test.
Too short a sample for a confirmatory verdict either way; reported honestly as such, not dressed up
to look like Part A's statistical power. This does NOT touch or pre-empt the separately pre-
registered 09-15 liquidation-crowding gate (eth_liquidation_crowding_conditional_fade_arm_
preregistration_20260823) -- that gate is about a specific fade-trade arm on liq_net_z_12, a
different signal/hypothesis entirely; this is a basis-predicts-liquidation-volume IC check.

This is diagnostic (IC-scan), NOT a cost-gate backtest -- matches this repo's dashboard-exposure
bar (statistical information content), not the live-trading economic bar. No promotion claim.
fresh_forward_bar_by_bar=true (targets are strictly forward of the signal bar), trade_ledgers_used_
as_input=false, saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(20260827)
N_PERM = 2000

Z_WINDOW = 48
ROC_WINDOW = 12
HORIZONS = [1, 3, 12, 48]  # 5m/15m/1h/4h, same as the original direction cheap-gate
SIGNALS = ["basis_raw", "basis_z48", "basis_roc12"]

SPLITS = {
    "TRAIN": ("2024-01-01", "2025-08-31"),
    "VAL": ("2025-09-01", "2025-12-31"),
    "OOS": ("2026-01-01", "2026-03-31"),
}

TAIL_RISK_DB = ROOT / "data" / "live" / "tail_risk.duckdb"
LIQ_CLEAN_START = pd.Timestamp("2026-07-18")  # reference_clean_data_locations_20260823: pre-this is flagged-unreliable


def _load_basis_frame() -> pd.DataFrame:
    """Verbatim from research_eth_spot_perp_basis_direction_cheap_gate_20260820.py."""
    spot = pd.read_csv(ROOT / "binance_data/klines_spot/ETHUSDT/ETHUSDT-5m-spot.csv",
                        usecols=["timestamp", "close"], parse_dates=["timestamp"]).rename(columns={"close": "spot_close"})
    perp_frames = []
    for f in ["data/splits/year_oos/training_features_2024.csv",
              "data/splits/year_oos/training_features_2025.csv",
              "data/splits/year_oos/training_features_2026_rebuilt.csv"]:
        p = pd.read_csv(ROOT / f, usecols=["timestamp", "close"], parse_dates=["timestamp"]).rename(columns={"close": "perp_close"})
        perp_frames.append(p)
    perp = pd.concat(perp_frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    df = perp.merge(spot, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    df["basis_raw"] = (df["perp_close"] - df["spot_close"]) / df["spot_close"]
    roll = df["basis_raw"].rolling(Z_WINDOW)
    df["basis_z48"] = (df["basis_raw"] - roll.mean()) / roll.std()
    df["basis_roc12"] = df["basis_raw"] - df["basis_raw"].shift(ROC_WINDOW)
    df["log_return"] = np.log(df["perp_close"] / df["perp_close"].shift(1))
    return df


def _permutation_null_ic(x: np.ndarray, y: np.ndarray, n_perm: int) -> tuple[float, float, float, int]:
    """Verbatim from the original script."""
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 30:
        return float("nan"), float("nan"), float("nan"), n
    rank_x = rankdata(x)
    rank_y = rankdata(y)
    mean_x, std_x = rank_x.mean(), rank_x.std()
    mean_y, std_y = rank_y.mean(), rank_y.std()
    real_ic = float((np.mean(rank_x * rank_y) - mean_x * mean_y) / (std_x * std_y))

    y_perm = np.column_stack([RNG.permutation(rank_y) for _ in range(n_perm)])
    sum_xy_perm = rank_x @ y_perm
    ic_perm = (sum_xy_perm / n - mean_x * mean_y) / (std_x * std_y)

    z = float((real_ic - ic_perm.mean()) / ic_perm.std())
    p = float((np.abs(ic_perm) >= abs(real_ic)).mean())
    return real_ic, z, p, n


def part_a_volatility(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 80)
    print("PART A: does basis predict forward realized VOLATILITY? (full 3-split IC scan)")
    print("=" * 80)
    d = df.set_index("timestamp")
    sq_ret = d["log_return"].shift(-1) ** 2

    results = {}
    for split_name, (start, end) in SPLITS.items():
        sub_idx = d.loc[start:end].index
        for h in HORIZONS:
            fwd_vol = np.sqrt(sq_ret.rolling(h).sum().shift(-(h - 1)))
            fwd_vol_sub = fwd_vol.loc[sub_idx]
            for sig in SIGNALS:
                sig_sub = d.loc[sub_idx, sig]
                ic, z, p, n = _permutation_null_ic(sig_sub.to_numpy(), fwd_vol_sub.to_numpy(), N_PERM)
                results[(split_name, sig, h)] = {"ic": ic, "z": z, "p": p, "n": n}

    for sig in SIGNALS:
        print(f"\n  [{sig}] vs forward realized vol")
        for split_name in SPLITS:
            row = []
            for h in HORIZONS:
                r = results[(split_name, sig, h)]
                flag = "**" if abs(r["z"]) >= 2.0 else "  "
                row.append(f"h{h}bar: ic={r['ic']:+.4f} z={r['z']:+.2f}{flag}")
            n0 = results[(split_name, sig, HORIZONS[0])]["n"]
            print(f"    {split_name:5s}(n={n0}): " + " | ".join(row))

    n_sig = sum(1 for v in results.values() if abs(v["z"]) >= 2.0)
    n_sig_consistent_sign = 0
    for sig in SIGNALS:
        for h in HORIZONS:
            signs = [np.sign(results[(s, sig, h)]["ic"]) for s in SPLITS if abs(results[(s, sig, h)]["z"]) >= 2.0]
            if len(signs) == len(SPLITS) and len(set(signs)) == 1:
                n_sig_consistent_sign += 1
    print(f"\n  significant cells (|z|>=2.0): {n_sig}/{len(results)}")
    print(f"  (signal,horizon) pairs significant AND same-signed in ALL 3 splits: {n_sig_consistent_sign}/{len(SIGNALS) * len(HORIZONS)}")
    return {f"{k[0]}|{k[1]}|h{k[2]}": v for k, v in results.items()}


def part_b_liquidation_crowding(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 80)
    print("PART B (EXPLORATORY, thin sample): does basis predict forward SHORT liquidation volume?")
    print("=" * 80)
    if not TAIL_RISK_DB.exists():
        print(f"  SKIPPED: {TAIL_RISK_DB} not found on this machine.")
        return {"skipped": True, "reason": "tail_risk.duckdb not found"}

    con = duckdb.connect(str(TAIL_RISK_DB), read_only=True)
    liq = con.sql("""
        SELECT ts, short_usd_1m, long_usd_1m FROM tail_risk_1m
        WHERE valid_liq_stream
    """).df()
    con.close()
    liq["ts"] = pd.to_datetime(liq["ts"], utc=True).dt.tz_convert(None)
    liq = liq[liq["ts"] >= LIQ_CLEAN_START].sort_values("ts")
    print(f"  liq rows (>=  {LIQ_CLEAN_START.date()}, valid_liq_stream only): {len(liq)}, "
          f"range {liq['ts'].min()}..{liq['ts'].max()}" if len(liq) else "  liq rows: 0")
    if len(liq) < 500:
        print("  SKIPPED: too few rows after the clean-window filter to say anything.")
        return {"skipped": True, "reason": "insufficient rows after 2026-07-18 filter", "n_rows": int(len(liq))}

    # aggregate 1m liquidation $ up to this script's 5-min bar grid (basis granularity), sum per bucket
    liq["bucket"] = liq["ts"].dt.floor("5min")
    liq_5m = liq.groupby("bucket", as_index=False)[["short_usd_1m", "long_usd_1m"]].sum()

    d = df[["timestamp"] + SIGNALS].merge(liq_5m, left_on="timestamp", right_on="bucket", how="inner").set_index("timestamp")
    print(f"  merged with basis frame: {len(d)} 5-min bars")
    if len(d) < 500:
        print("  SKIPPED: too few merged rows.")
        return {"skipped": True, "reason": "insufficient merged rows", "n_rows": int(len(d))}

    # horizons in 5-min bars: 1h=12, 4h=48, 1d=288 (drop the sub-hour 1/3-bar horizons here --
    # liquidation $ is much sparser/burstier per-bar than returns, needs more bars to be non-degenerate)
    liq_horizons = {"1h": 12, "4h": 48, "1d": 288}
    results = {}
    for h_name, h in liq_horizons.items():
        fwd_short = d["short_usd_1m"].shift(-1).rolling(h).sum().shift(-(h - 1))
        fwd_long = d["long_usd_1m"].shift(-1).rolling(h).sum().shift(-(h - 1))
        for sig in SIGNALS:
            for target_name, target in (("fwd_short_usd", fwd_short), ("fwd_long_usd", fwd_long)):
                ic, z, p, n = _permutation_null_ic(d[sig].to_numpy(), target.to_numpy(), N_PERM)
                results[(sig, target_name, h_name)] = {"ic": ic, "z": z, "p": p, "n": n}

    for sig in SIGNALS:
        print(f"\n  [{sig}]")
        for target_name in ("fwd_short_usd", "fwd_long_usd"):
            row = []
            for h_name in liq_horizons:
                r = results[(sig, target_name, h_name)]
                flag = "**" if abs(r["z"]) >= 2.0 else "  "
                row.append(f"{h_name}: ic={r['ic']:+.4f} z={r['z']:+.2f}{flag}")
            n0 = results[(sig, target_name, list(liq_horizons)[0])]["n"]
            print(f"    {target_name:14s}(n={n0}): " + " | ".join(row))

    n_sig = sum(1 for v in results.values() if abs(v["z"]) >= 2.0)
    print(f"\n  significant cells (|z|>=2.0): {n_sig}/{len(results)} -- EXPLORATORY, single window, "
          f"~1 month of data, NOT a confirmatory result either way.")
    return {"skipped": False, "window_start": str(LIQ_CLEAN_START.date()), "n_bars": int(len(d)),
            "results": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in results.items()}}


def main() -> None:
    df = _load_basis_frame()
    a = part_a_volatility(df)
    b = part_b_liquidation_crowding(df)

    out = {"part_a_volatility": a, "part_b_liquidation_crowding": b}
    out_path = ROOT / "tmp/eth_spot_perp_basis_volatility_liquidation_reframe_20260827.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
