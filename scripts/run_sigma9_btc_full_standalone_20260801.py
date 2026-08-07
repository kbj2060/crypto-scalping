#!/usr/bin/env python3
"""Standalone grid-sweep backtest of the full-feature (38-col) Sigma9 BTC ensemble, mirroring
run_sigma9_btc_standalone_20260706.py exactly (same grid, same VAL window, same Sigma6 barrier
mechanics, no regime filter) so the result is directly comparable to that script's
btc_standalone_val_frontier.csv (best: thr=0.60/lev=2/sl=1.5, VAL cost1=+16.6%/MDD-9.6%).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402
import run_sigma6_regime_trend_20260705 as s6  # noqa: E402

TAPE = ROOT / "tmp/causal_regen_20260516/sigma9_1h_btc_full_20260801/tape_btc_full_ensemble.parquet"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma9_1h_btc_full_20260801"
PFX = s6.PFX


def load_btc_tape() -> pd.DataFrame:
    t = pd.read_parquet(TAPE)
    t["timestamp"] = pd.to_datetime(t["timestamp"])
    t = t.sort_values("timestamp").reset_index(drop=True)
    t[f"{PFX}bull_prob"] = 0.0
    t[f"{PFX}bear_prob"] = 0.0
    t[f"{PFX}chop_prob"] = 0.0
    t["regime3_cmamba_h6_sidecar_stability_score"] = 1.0
    return t


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_btc_tape()
    tapes = {thr: v2.apply_quality_threshold(raw, thr) for thr in (0.45, 0.55, 0.60, 0.70)}
    base = dict(margin=0.30, trail_atr=5.0, min_profit_atr=2.0, max_hold=144, cooldown=3,
                reg_mode="none", reg_thr=0.0, stab_thr=0.0)
    grid = list(itertools.product([0.45, 0.55, 0.60, 0.70], [2.0, 3.0, 4.0], [1.5, 2.5]))
    rows = []
    for thr, lev, sl in grid:
        r = s6.backtest(tapes[thr], leverage=lev, sl_atr=sl, fee_mult=1.0,
                         start=s6.VAL_START, end=s6.VAL_END, **base)
        rows.append({"thr": thr, "lev": lev, "sl": sl, "c1": round(r["pnl"], 1),
                     "mdd": round(r["mdd"], 1), "tr": r["trades"], "wr": round(r["wr"], 3),
                     "mo": len(r["by_month"]),
                     "minmo": round(min(r["by_month"].values()) * 100, 1) if r["by_month"] else 0.0})
    df = pd.DataFrame(rows).sort_values("c1", ascending=False)
    df.to_csv(OUT_DIR / "btc_full_standalone_val_frontier.csv", index=False)
    print("=== BTC FULL-FEATURE standalone (no regime filter), VAL, by cost1 ===", flush=True)
    print(df.to_string(index=False), flush=True)
    print("\n=== Original 28-feature BTC standalone best (for reference) ===", flush=True)
    print("thr=0.60 lev=2.0 sl=1.5: c1=+16.6% mdd=-9.6% tr=56 wr=0.393", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
