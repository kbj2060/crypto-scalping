#!/usr/bin/env python3
"""Combine quality_threshold sweep (no retraining, recomputed from cached raw probs) with the
best barrier/filter regions found in scripts/replay_omega6_v2_variants_20260704.py's round 2/3
search, to see if stricter conviction filtering closes the cost3 gap.

Still validation-only (2025-10-01..12-31); OOS untouched.
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

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega6_v2_qualitythreshold_20260704"


def main() -> int:
    tape = v2.load_tape()
    thresholds = [0.55, 0.65, 0.75]
    tapes_by_threshold = {thr: v2.apply_quality_threshold(tape, thr) for thr in thresholds}

    rows = []
    for thr, tp_mult, sl_mult, cooldown in itertools.product(
        thresholds,
        (8.0, 12.0),
        (4.0, 5.0),
        (0, 12),
    ):
        cfg = v2.VariantConfig(
            name=f"qt{thr}_tp{tp_mult}_sl{sl_mult}_cd{cooldown}",
            tp_mode="atr_scaled",
            tp_atr_mult=tp_mult,
            sl_atr_mult=sl_mult,
            sizing_mode="fixed",
            fixed_margin=0.30,
            fixed_leverage=2.0,
            cooldown_bars=cooldown,
            quality_threshold=thr,
        )
        result = v2.cost_stress(tapes_by_threshold[thr], cfg, start=v2.VAL_START, end=v2.VAL_END)
        gate_pass = v2.passes_gates(result)
        row = {
            "name": cfg.name,
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
    df.to_csv(OUT_DIR / "qt_variant_ranking.csv", index=False)
    print(f"\ntotal variants: {len(df)}  gate_pass: {int(df['gate_pass'].sum())}", flush=True)
    print("\n=== top 20 by cost3_pnl ===", flush=True)
    print(df.head(20).to_string(index=False), flush=True)
    print(f"\nfull ranking: {OUT_DIR / 'qt_variant_ranking.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
