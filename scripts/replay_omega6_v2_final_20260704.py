#!/usr/bin/env python3
"""Final fine-grained pass around the closest near-miss:
ref_p3_qt0.6_tp14.0_sl5.0_cd12 (cost1 +6.29%, MDD -14.84%, cost3 -0.92%, MDD -17.82%).

Caveat logged deliberately: this is the 4th round of grid search on the SAME validation
window (v1 baseline -> ATR/vol-target round1/2/3 -> quality-threshold sweep -> persistence
sweep -> this refinement). Any variant that crosses zero here carries real overfitting risk
from repeated re-fitting to the same validation data, even though no future/OOS data was used
as a decision input at any point. The one-shot OOS check afterward is the real test of whether
this generalizes, not a formality.

Still validation-only (2025-10-01..12-31); OOS untouched until a variant is frozen.
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

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega6_v2_final_20260704"


def main() -> int:
    tape = v2.load_tape()
    thresholds = [0.55, 0.58, 0.60, 0.62, 0.65]
    tapes_by_threshold = {thr: v2.apply_quality_threshold(tape, thr) for thr in thresholds}

    rows = []
    for thr, persist, tp_mult, sl_mult, cooldown in itertools.product(
        thresholds,
        (3,),
        (13.0, 14.0, 15.0, 16.0),
        (4.5, 5.0, 5.5, 6.0),
        (9, 10, 11, 12, 13, 14),
    ):
        cfg = v2.VariantConfig(
            name=f"fin_p{persist}_qt{thr}_tp{tp_mult}_sl{sl_mult}_cd{cooldown}",
            tp_mode="atr_scaled",
            tp_atr_mult=tp_mult,
            sl_atr_mult=sl_mult,
            sizing_mode="fixed",
            fixed_margin=0.30,
            fixed_leverage=2.0,
            cooldown_bars=cooldown,
            quality_threshold=thr,
            persistence_bars=persist,
        )
        result = v2.cost_stress(tapes_by_threshold[thr], cfg, start=v2.VAL_START, end=v2.VAL_END)
        gate_pass = v2.passes_gates(result)
        row = {
            "name": cfg.name,
            "persistence_bars": persist,
            "quality_threshold": thr,
            "tp_mult": tp_mult,
            "sl_mult": sl_mult,
            "cooldown": cooldown,
            "cost1_pnl": result["cost1"]["pnl"],
            "cost1_mdd": result["cost1"]["mdd"],
            "cost1_trades": result["cost1"]["trades"],
            "cost1_wr": result["cost1"]["wr"],
            "cost3_pnl": result["cost3"]["pnl"],
            "cost3_mdd": result["cost3"]["mdd"],
            "cost3_trades": result["cost3"]["trades"],
            "months": len(result["cost1"]["trades_by_month"]),
            "gate_pass": gate_pass,
        }
        rows.append(row)
        if gate_pass:
            print("GATE PASS:", json.dumps(row), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["gate_pass", "cost3_pnl"], ascending=[False, False])
    df.to_csv(OUT_DIR / "final_variant_ranking.csv", index=False)
    print(f"\ntotal variants: {len(df)}  gate_pass: {int(df['gate_pass'].sum())}", flush=True)
    print("\n=== top 25 by cost3_pnl ===", flush=True)
    print(df.head(25).to_string(index=False), flush=True)
    print(f"\nfull ranking: {OUT_DIR / 'final_variant_ranking.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
