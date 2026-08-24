#!/usr/bin/env python3
"""RESEARCH ONLY -- evaluates the single-seed hard-regime-filter pilot bundle (trained with
--hard-regime-filter on top of the Phase-1-confirmed Variant B (same_as_direction quality)
recipe, seed=2559205075 -- the SAME seed already evaluated for the existing soft-weight Variant B
bundle in Phase 1) and prints it side by side with that already-computed soft-weight baseline.

Context: a prior single-seed pilot (docs/experiments/eth_h48qual_hard_regime_filter_pilot_20260812.md)
found hard-regime-filter made VAL/OOS PnL WORSE on the h48orig recipe (barrier-based quality
label, since confirmed inferior to same_as_direction in this session's Phase 1). This script
re-checks the same mechanism on the Variant B recipe using the exact evaluate_variant_seed
function already validated in eval_eth_candidate_unified_phase1_quality_ab_20260817.py (imported,
not copied) -- only the bundle directory name differs (variant string
"quality_B_samedir_hardregime_pilot" maps to the --out-suffix used when training the pilot).

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_
used=false, future_rows_used_for_entry=false.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_eth_candidate_unified_phase1_quality_ab_20260817 as phase1eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent3head  # noqa: E402

SEED = 2559205075
SOFT_VARIANT = "quality_B_samedir"
HARD_VARIANT = "quality_B_samedir_hardregime_pilot"


def main() -> int:
    device = parent3head._device("cpu")

    phase1_detail_csv = ROOT / "tmp/causal_regen_20260516/eth_candidate_unified_phase1_eval_20260817/per_seed_detail.csv"
    if phase1_detail_csv.exists():
        prior = pd.read_csv(phase1_detail_csv)
        soft_rows = prior[(prior["variant"] == SOFT_VARIANT) & (prior["seed"] == SEED)].to_dict("records")
        print(f"reused {len(soft_rows)} soft-weight rows from existing Phase 1 eval output ({phase1_detail_csv})")
    else:
        print("Phase 1 eval csv not found on this machine -- recomputing soft-weight baseline fresh", flush=True)
        soft_rows = phase1eval.evaluate_variant_seed(SOFT_VARIANT, SEED, device)

    print(f"=== evaluating hard-regime-filter pilot: {HARD_VARIANT} seed={SEED} ===", flush=True)
    hard_rows = phase1eval.evaluate_variant_seed(HARD_VARIANT, SEED, device)

    all_rows = list(soft_rows) + list(hard_rows)
    df = pd.DataFrame(all_rows)
    out_dir = ROOT / "tmp/causal_regen_20260516/eth_candidate_hardregime_pilot_eval_20260818"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "detail.csv", index=False)

    pivot = df.pivot_table(index=["window", "threshold"], columns="variant", values="pnl").round(2)
    pivot["delta_hard_minus_soft"] = (pivot[HARD_VARIANT] - pivot[SOFT_VARIANT]).round(2)
    pivot = pivot.reset_index()
    print("\n=== PnL: soft-weight (Phase 1) vs hard-regime-filter (pilot), seed=2559205075 ===")
    print(pivot.to_string(index=False))
    pivot.to_csv(out_dir / "pivot_pnl.csv", index=False)
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
