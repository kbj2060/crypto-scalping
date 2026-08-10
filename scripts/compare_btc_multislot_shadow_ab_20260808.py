"""A/B report for the two BTC multi-slot shadow records (2026-08-08).

  baseline  data/ensemble/omega4_6_1_btc_multislot_shadow_ledger_20260807.csv
  overlay   data/ensemble/omega4_6_1_btc_multislot_shadow_ledger_20260807_regime.csv

The overlay run is the same loop with BTC_MULTISLOT_SHADOW_REGIME_OVERLAY=1: sidecar
margin_fraction is multiplied by the czz_trend regime multiplier (bull 1.5 / chop 1.0 / bear 0.5)
from the causal 4% directional-change wave at the entry bar, before the /N_SLOTS split.

COLD-START CAVEAT, reported in the output rather than buried: the overlay process started flat
while the baseline was already holding slots from the previous day, so their earliest trades are
not paired. Comparison over a common window starting at the first bar where BOTH were flat is
also printed; until enough paired trades accumulate, treat every number here as provisional.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/ensemble/omega4_6_1_btc_multislot_shadow_ledger_20260807.csv"
OVER = ROOT / "data/ensemble/omega4_6_1_btc_multislot_shadow_ledger_20260807_regime.csv"


def metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0, "pnl_pct": 0.0, "mdd_pct": 0.0, "win_rate": None, "calmar": None}
    r = df["trade_return_net"].to_numpy(dtype=float)
    eq = np.cumprod(1.0 + r)
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min() * 100)
    pnl = float((eq[-1] - 1.0) * 100)
    return {"trades": int(len(r)), "pnl_pct": round(pnl, 2), "mdd_pct": round(mdd, 2),
            "win_rate": round(float((r > 0).mean()), 3),
            "calmar": round(pnl / abs(mdd), 2) if mdd < -1e-9 else None}


def load(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    for c in ("entry_timestamp", "exit_timestamp"):
        if c in d.columns:
            d[c] = pd.to_datetime(d[c])
    return d.sort_values("exit_timestamp").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="print machine-readable output only")
    args = ap.parse_args()
    base, over = load(BASE), load(OVER)
    out = {"baseline": {"path": str(BASE.relative_to(ROOT)), **metrics(base)},
           "overlay": {"path": str(OVER.relative_to(ROOT)), **metrics(over)}}

    if not base.empty and not over.empty:
        common = max(base["entry_timestamp"].min(), over["entry_timestamp"].min())
        out["common_window_start"] = str(common)
        out["baseline_common"] = metrics(base.loc[base["entry_timestamp"] >= common])
        out["overlay_common"] = metrics(over.loc[over["entry_timestamp"] >= common])
    if not over.empty and "regime_dir" in over.columns:
        out["overlay_regime_mix"] = {str(k): int(v) for k, v in
                                     over["regime_dir"].value_counts().sort_index().items()}
        out["overlay_mean_mult"] = round(float(over["regime_mult"].mean()), 3)
    out["status"] = ("no closed trades yet in one or both records -- keep accumulating"
                     if base.empty or over.empty else "both records have closed trades")

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if not base.empty and not over.empty:
        n = min(out["baseline"]["trades"], out["overlay"]["trades"])
        if n < 30:
            print(f"\nNOTE: only {n} closed trades on the shorter side. The backtest edge being "
                  f"tested (MDD -10.77 -> -9.24 on 124 OOS trades) is not resolvable at this "
                  f"sample size; this is a monitoring readout, not a verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
