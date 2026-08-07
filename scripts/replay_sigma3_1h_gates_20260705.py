#!/usr/bin/env python3
"""Sigma3 (1h HGB + trend-scanning) pre-registered gate sweep, validation = 2025-07-01..12-31.

1h-appropriate barriers (atr_pct median ~0.91%; run_variant's effective required move =
tp_atr_mult*atr/notional, so tp_atr_mult 1.5-2.5 -> ~2.3-3.8% TP, sl 0.9-1.2 -> ~1.4-1.8% SL).
max_hold 48 bars (2 days), cooldown 3 bars.

Pre-registered grid (27 configs/seed): threshold {0.50,0.60,0.70} x persistence {0,2,4} x
(tp,sl) {(1.5,1.0),(2.0,0.9),(2.5,1.2)} ; cooldown 3, max_hold 48, margin 0.30 x leverage 2.0.

Pre-registered gates (6-month window, 1h-scaled trade floor): cost1 AND cost3 PnL > 0;
MDD >= -20% both tiers; trades >= 40; months-with-trades >= 5; AND same config cost1 > 0 on
BOTH seeds. One-shot on untouched 2026-03-02..06-30 only if a config jointly passes.
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

TAPE_DIR = ROOT / "tmp/causal_regen_20260516/sigma3_1h_hgb_20260705"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma3_1h_gates_20260705"
VAL_START = pd.Timestamp("2025-07-01")
VAL_END = pd.Timestamp("2025-12-31 23:59:59")

THRESHOLDS = [0.50, 0.60, 0.70]
PERSISTS = [0, 2, 4]
TPSL = [(1.5, 1.0), (2.0, 0.9), (2.5, 1.2)]
COOLDOWN = 3
MAX_HOLD = 48


def load_tape(suffix: str) -> pd.DataFrame:
    t = pd.read_parquet(TAPE_DIR / f"tape_{suffix}.parquet")
    t["timestamp"] = pd.to_datetime(t["timestamp"])
    return t.sort_values("i").reset_index(drop=True)


def passes_gates_6mo(result: dict) -> bool:
    c1, c3 = result["cost1"], result["cost3"]
    return (
        c1["pnl"] > 0 and c3["pnl"] > 0
        and c1["mdd"] >= -20.0 and c3["mdd"] >= -20.0
        and c1["trades"] >= 40
        and len(c1["trades_by_month"]) >= 5
    )


def run_seed(suffix: str) -> pd.DataFrame:
    tape = load_tape(suffix)
    tapes = {thr: v2.apply_quality_threshold(tape, thr) for thr in THRESHOLDS}
    rows = []
    for thr, persist, (tp, sl) in itertools.product(THRESHOLDS, PERSISTS, TPSL):
        cfg = v2.VariantConfig(
            name=f"s3_{suffix}_qt{thr}_p{persist}_tp{tp}_sl{sl}",
            tp_mode="atr_scaled", tp_atr_mult=tp, sl_atr_mult=sl,
            sizing_mode="fixed", fixed_margin=0.30, fixed_leverage=2.0,
            cooldown_bars=COOLDOWN, quality_threshold=thr, persistence_bars=persist,
            max_hold_bars=MAX_HOLD, use_fallback=False,
        )
        r = v2.cost_stress(tapes[thr], cfg, start=VAL_START, end=VAL_END)
        rows.append({
            "seed": suffix, "quality_threshold": thr, "persistence_bars": persist,
            "tp_mult": tp, "sl_mult": sl,
            "cost1_pnl": r["cost1"]["pnl"], "cost1_mdd": r["cost1"]["mdd"],
            "cost1_trades": r["cost1"]["trades"], "cost1_wr": r["cost1"]["wr"],
            "cost3_pnl": r["cost3"]["pnl"], "cost3_mdd": r["cost3"]["mdd"],
            "months": len(r["cost1"]["trades_by_month"]),
            "gate_pass": passes_gates_6mo(r),
        })
        print(json.dumps(rows[-1]), flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = run_seed("seedA")
    b = run_seed("seedB")
    m = a.merge(b, on=["quality_threshold", "persistence_bars", "tp_mult", "sl_mult"], suffixes=("_A", "_B"))
    m["joint_pass"] = m["gate_pass_A"] & (m["cost1_pnl_B"] > 0)
    m.to_csv(OUT_DIR / "sigma3_gate_ranking.csv", index=False)
    passing = m[m["joint_pass"]].sort_values("cost3_pnl_A", ascending=False)
    print(f"\nseedA gate_pass: {int(a['gate_pass'].sum())}/27, joint (seedB cost1>0): {len(passing)}/27", flush=True)
    if len(passing):
        print(passing[["quality_threshold", "persistence_bars", "tp_mult", "sl_mult",
                       "cost1_pnl_A", "cost3_pnl_A", "cost1_mdd_A", "cost3_mdd_A", "cost1_trades_A",
                       "cost1_pnl_B", "cost3_pnl_B"]].to_string(index=False), flush=True)
    else:
        print("\ntop 8 seedA by cost3_pnl (none jointly pass):", flush=True)
        print(a.sort_values("cost3_pnl", ascending=False).head(8).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
