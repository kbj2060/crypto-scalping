#!/usr/bin/env python3
"""DSR/PBO-CSCV/falsification_audit on the ETH live-promotion (Omega4.6.1 dual) N=3 seed
check, this time at trade-level -> daily-resampled granularity instead of the 6
window-level summary numbers used in analyze_eth_live_promotion_seed_dsr_pbo_20260819.py.

Rebuilds the exact "with_gate" (duration-gated) per-trade return series from each seed's
portfolio_ledger_{window}_posfix_canonicaldata.csv (trade_return column is already the
account-level fractional return per trade, ready to compound -- confirmed against
research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py::_ledger_metrics), applying
the same ou_halflife<=DURATION_THRESHOLD gate research_eth_omega461_live_sltp_mfe_width_20260813
.py::_duration_gated uses, then compounds trades landing on the same exit date and reindexes
onto the full daily calendar so all 3 seeds share the same period index (required for
pbo_cscv/falsification_audit's row-alignment assumption).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.selection_stats import deflated_sharpe_ratio, falsification_audit, pbo_cscv, sharpe  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402

WINDOWS = ["2025q1", "2025q2", "2025q3", "val", "oos_q1", "oos_q2"]
SEED_LABELS = ["seed260620_original", "94046540", "524707103"]
OUT_ROOT = ROOT / "tmp/causal_regen_20260516"


def gated_daily_returns(seed_label: str, wname: str) -> pd.Series:
    """Per-day compounded, duration-gated trade returns for one seed/window."""
    ledger_path = OUT_ROOT / f"eth_live_promotion_seed_robustness_20260819_{seed_label}" / f"portfolio_ledger_{wname}_posfix_canonicaldata.csv"
    ledger = pd.read_csv(ledger_path)
    if len(ledger) == 0:
        return pd.Series(dtype=np.float64)

    wd = gate.WINDOW_DEFS[wname]
    frame = sweep.load_frame(wd["start"], wd["end"], base_csv=wd["base_csv"], wide24_csv=wd["wide24_csv"])
    frame, _ = gate._drop_route_nan(frame)

    ledger["entry_timestamp_dt"] = pd.to_datetime(ledger["entry_timestamp"])
    market = frame[["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp_dt"})
    merged = ledger.merge(market, on="entry_timestamp_dt", how="left")
    hit = merged["ou_halflife"] <= greedy.DURATION_THRESHOLD
    gated_return = np.where(hit, 0.0, merged["trade_return"].to_numpy(dtype=np.float64))

    exit_date = pd.to_datetime(merged["exit_timestamp"]).dt.normalize()
    # compound same-day exits, exactly like the sequential cumprod _duration_gated itself uses
    per_day_log1p = np.log1p(gated_return)
    daily_log = pd.Series(per_day_log1p, index=exit_date).groupby(level=0).sum()
    return np.expm1(daily_log)


def build_matrix():
    per_seed_daily = {s: [] for s in SEED_LABELS}
    for wname in WINDOWS:
        for s in SEED_LABELS:
            per_seed_daily[s].append(gated_daily_returns(s, wname))
        print(f"window={wname} done", flush=True)

    full_index = pd.date_range(
        gate.WINDOW_DEFS[WINDOWS[0]]["start"], gate.WINDOW_DEFS[WINDOWS[-1]]["end"], freq="D"
    )
    cols = {}
    for s in SEED_LABELS:
        combined = pd.concat(per_seed_daily[s])
        combined = combined.groupby(level=0).sum()  # in case a window boundary duplicates a date
        cols[s] = combined.reindex(full_index, fill_value=0.0)
    matrix_df = pd.DataFrame(cols, index=full_index)
    return matrix_df


def main():
    matrix_df = build_matrix()
    out_csv = OUT_ROOT / "eth_live_promotion_seed_robustness_20260819_tradelevel_daily_matrix.csv"
    matrix_df.to_csv(out_csv)
    print(f"\ndaily returns matrix written: {out_csv}  shape={matrix_df.shape}")

    matrix = matrix_df.to_numpy()
    nonzero_days = (matrix != 0).sum(axis=0)
    print(f"nonzero (trade-exit) days per seed: {dict(zip(SEED_LABELS, nonzero_days.tolist()))}")

    per_seed_sharpe = [sharpe(matrix[:, i]) for i in range(matrix.shape[1])]
    print("\nper-seed daily Sharpe (n_obs = calendar days in the full 2025-01-01..2026-06-30 span):")
    for s, sr in zip(SEED_LABELS, per_seed_sharpe):
        print(f"  seed {s:20s} sharpe={sr:+.4f}")

    deployed_returns = matrix[:, 0]
    dsr = deflated_sharpe_ratio(deployed_returns, np.array(per_seed_sharpe))
    print(f"\nDSR of the actually-deployed seed ({SEED_LABELS[0]}):")
    for k, v in dsr.items():
        print(f"  {k:22s} {v}")

    pbo = pbo_cscv(matrix, n_splits=10)
    print(f"\nPBO-CSCV (n_splits=10, default -- {matrix.shape[0]} periods now comfortably clears "
          f"the >= {10*3} requirement):")
    for k, v in pbo.items():
        print(f"  {k:22s} {v}")

    fa = falsification_audit(matrix, n_null_draws=500, block_size=20)
    print(f"\nfalsification_audit (n_null_draws=500, block_size=20 days):")
    for k, v in fa.items():
        print(f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
