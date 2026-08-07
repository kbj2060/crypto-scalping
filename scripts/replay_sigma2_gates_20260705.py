#!/usr/bin/env python3
"""Sigma2 pre-registered gate sweep, validation = 2025-07-01..12-31 (SIX months; Jul-Sep never
used for selection by any prior search round).

Pre-registered grid (18 configs/seed): threshold {0.45,0.55,0.65} x persistence {0,2,3} x
(tp,sl) {(15,5),(13,4)} x cooldown 12, fixed margin 0.30 x leverage 2.0.

Pre-registered gates (6-month window, scaled from the historical 3-month gates):
  cost1 AND cost3 PnL > 0; MDD >= -20% both tiers; trades >= 100; months-with-trades >= 5;
  AND the same config must have cost1 PnL > 0 on BOTH seeds (sign consistency).

If any config passes, freeze one from the CENTER of the passing region (seedA weights) and run
the one-shot on 2026-03-02..06-30 exactly once. 2026-01..02 is reported as soft context only.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402

VAL_START = pd.Timestamp("2025-07-01")
VAL_END = pd.Timestamp("2025-12-31 23:59:59")
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma2_gates_20260705"

GRID = list(itertools.product((0.45, 0.55, 0.65), (0, 2, 3), ((15.0, 5.0), (13.0, 4.0))))


def load_tape(suffix: str) -> pd.DataFrame:
    tape = pd.read_parquet(ROOT / "tmp/causal_regen_20260516" / f"sigma2_tape_{suffix}_20260705" / "tape.parquet")
    tape["timestamp"] = pd.to_datetime(tape["timestamp"])
    return tape.sort_values("i").reset_index(drop=True)


def passes_gates_6mo(result: dict) -> bool:
    c1, c3 = result["cost1"], result["cost3"]
    return (
        c1["pnl"] > 0 and c3["pnl"] > 0
        and c1["mdd"] >= -20.0 and c3["mdd"] >= -20.0
        and c1["trades"] >= 100
        and len(c1["trades_by_month"]) >= 5
    )


def run_seed(suffix: str) -> pd.DataFrame:
    tape = load_tape(suffix)
    tapes_by_thr = {thr: v2.apply_quality_threshold(tape, thr) for thr, _, _ in GRID}
    rows = []
    for thr, persist, (tp, sl) in GRID:
        cfg = v2.VariantConfig(
            name=f"s2_{suffix}_qt{thr}_p{persist}_tp{tp}_sl{sl}",
            tp_mode="atr_scaled",
            tp_atr_mult=tp,
            sl_atr_mult=sl,
            sizing_mode="fixed",
            fixed_margin=0.30,
            fixed_leverage=2.0,
            cooldown_bars=12,
            quality_threshold=thr,
            persistence_bars=persist,
            use_fallback=False,
        )
        result = v2.cost_stress(tapes_by_thr[thr], cfg, start=VAL_START, end=VAL_END)
        rows.append(
            {
                "seed": suffix,
                "quality_threshold": thr,
                "persistence_bars": persist,
                "tp_mult": tp,
                "sl_mult": sl,
                "cost1_pnl": result["cost1"]["pnl"],
                "cost1_mdd": result["cost1"]["mdd"],
                "cost1_trades": result["cost1"]["trades"],
                "cost1_wr": result["cost1"]["wr"],
                "cost3_pnl": result["cost3"]["pnl"],
                "cost3_mdd": result["cost3"]["mdd"],
                "months": len(result["cost1"]["trades_by_month"]),
                "gate_pass": passes_gates_6mo(result),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_a = run_seed("seedA")
    df_b = run_seed("seedB")
    merged = df_a.merge(df_b, on=["quality_threshold", "persistence_bars", "tp_mult", "sl_mult"], suffixes=("_A", "_B"))
    merged["joint_pass"] = merged["gate_pass_A"] & (merged["cost1_pnl_B"] > 0)
    merged.to_csv(OUT_DIR / "sigma2_gate_ranking.csv", index=False)
    passing = merged[merged["joint_pass"]]
    print(f"\nseedA gate_pass: {int(df_a['gate_pass'].sum())}/18, joint (seedB cost1>0 too): {len(passing)}/18", flush=True)
    if len(passing):
        print(passing[["quality_threshold", "persistence_bars", "tp_mult", "sl_mult", "cost1_pnl_A", "cost3_pnl_A", "cost1_mdd_A", "cost1_pnl_B"]].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
