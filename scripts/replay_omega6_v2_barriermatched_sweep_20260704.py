#!/usr/bin/env python3
"""Priority-1 test: sweep persistence/threshold/ATR-barrier/cooldown on the decision tape built
from the BARRIER-MATCHED-label-trained L2 bundles (scripts/precompute_omega6_barriermatched_tape_20260704.py),
same methodology as the earlier 5-round search on the original zigzag-swing-label bundles, to
see whether relabeling to match the deployed barrier improves win rate/PnL over the frozen
winner (fin_p3_qt0.58_tp15.0_sl5.0_cd12, cost1 +21.96%/cost3 +10.68%, WR 44.6%/40.5%, val).

Validation-only (2025-10-01..12-31); OOS untouched unless a variant beats the frozen winner.
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

TAPE_PATH = ROOT / "tmp/causal_regen_20260516/omega6_barriermatched_decision_tape_20260704/tape.parquet"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega6_v2_barriermatched_sweep_20260704"


def load_tape() -> pd.DataFrame:
    tape = pd.read_parquet(TAPE_PATH)
    tape["timestamp"] = pd.to_datetime(tape["timestamp"])
    return tape.sort_values("i").reset_index(drop=True)


def main() -> int:
    tape = load_tape()
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.58, 0.62, 0.65, 0.70, 0.75]
    tapes_by_threshold = {thr: v2.apply_quality_threshold(tape, thr) for thr in thresholds}

    rows = []
    for thr, persist, tp_mult, sl_mult, cooldown in itertools.product(
        thresholds,
        (1, 2, 3, 4),
        (8.0, 10.0, 13.0, 15.0, 18.0),
        (3.0, 4.0, 5.0, 6.0),
        (0, 4, 8, 12),
    ):
        cfg = v2.VariantConfig(
            name=f"bm_p{persist}_qt{thr}_tp{tp_mult}_sl{sl_mult}_cd{cooldown}",
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
            "cost3_wr": result["cost3"]["wr"],
            "months": len(result["cost1"]["trades_by_month"]),
            "gate_pass": gate_pass,
        }
        rows.append(row)
        if gate_pass:
            print("GATE PASS:", json.dumps(row), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["gate_pass", "cost3_pnl"], ascending=[False, False])
    df.to_csv(OUT_DIR / "barriermatched_variant_ranking.csv", index=False)
    print(f"\ntotal variants: {len(df)}  gate_pass: {int(df['gate_pass'].sum())}", flush=True)
    print("\n=== top 25 by cost3_pnl ===", flush=True)
    print(df.head(25).to_string(index=False), flush=True)
    print(f"\nfull ranking: {OUT_DIR / 'barriermatched_variant_ranking.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
