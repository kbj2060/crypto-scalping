#!/usr/bin/env python3
"""One-shot OOS check for the frozen Omega6 v2 winner, selected from the final 480-variant
fine-grained sweep (scripts/replay_omega6_v2_final_20260704.py).

Selection rule: among the 49/480 gate-passing variants, do NOT cherry-pick the single highest
cost3_pnl (that would be the most overfit choice after 5 rounds of grid search on the same
validation window). Instead freeze a variant near the CENTER of the passing region -- almost
all 49 passes cluster at quality_threshold=0.58, tp_mult in [13,16], sl_mult in [4.5,5.5],
cooldown in [9,14]. tp_mult=15.0 (mid of 13-16), sl_mult=5.0 (mid of 4.5-5.5), cooldown=12
(mid of 9-14, also matches cooldown chosen in the prior refinement round) is the frozen pick:
fin_p3_qt0.58_tp15.0_sl5.0_cd12 (val cost1 +21.96%, MDD -12.45%, 112 trades, WR 44.6%;
val cost3 +10.68%, MDD -13.49%, 111 trades).

This script scores that EXACT frozen config on the reserved OOS window ONLY. OOS data has not
been read by any prior script in this search (quality-threshold sweep, persistence sweep,
refinement sweep, final sweep all used VAL_START/VAL_END only). This is the one-shot OOS check.

Caveat for the record: the tape only extends through 2026-02-28 (not the default fresh-forward
OOS end of 2026-03-31), so this is a ~2-month OOS window (Jan-Feb 2026), not the full quarter.
"""

from __future__ import annotations

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

OOS_START = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-03-31 23:59:59")  # tape caps at 2026-02-28; documented above

FROZEN = v2.VariantConfig(
    name="frozen_fin_p3_qt0.58_tp15.0_sl5.0_cd12",
    tp_mode="atr_scaled",
    tp_atr_mult=15.0,
    sl_atr_mult=5.0,
    sizing_mode="fixed",
    fixed_margin=0.30,
    fixed_leverage=2.0,
    cooldown_bars=12,
    quality_threshold=0.58,
    persistence_bars=3,
)


def main() -> int:
    tape = v2.load_tape()
    tape_qt = v2.apply_quality_threshold(tape, FROZEN.quality_threshold)

    val_result = v2.cost_stress(tape_qt, FROZEN, start=v2.VAL_START, end=v2.VAL_END)
    oos_result = v2.cost_stress(tape_qt, FROZEN, start=OOS_START, end=OOS_END)

    print("=== FROZEN CONFIG ===")
    print(json.dumps({k: v for k, v in FROZEN.__dict__.items() if k != "extra"}, indent=2))

    print("\n=== VALIDATION (2025-10-01..12-31, already seen during search) ===")
    for tag in ("cost1", "cost3"):
        r = val_result[tag]
        print(f"{tag}: pnl={r['pnl']:.2f}% mdd={r['mdd']:.2f}% trades={r['trades']} wr={r['wr']:.3f} months={len(r['trades_by_month'])}")

    print("\n=== OOS one-shot (2026-01-01.., tape caps at 2026-02-28) ===")
    for tag in ("cost1", "cost3"):
        r = oos_result[tag]
        print(f"{tag}: pnl={r['pnl']:.2f}% mdd={r['mdd']:.2f}% trades={r['trades']} wr={r['wr']:.3f} months={len(r['trades_by_month'])} reasons={r['reasons']}")

    oos_pass = (
        oos_result["cost1"]["pnl"] > 0
        and oos_result["cost3"]["pnl"] > 0
        and oos_result["cost1"]["mdd"] >= -20.0
        and oos_result["cost3"]["mdd"] >= -20.0
    )
    print(f"\nOOS_PASS (pnl>0 both tiers, mdd>=-20% both tiers): {oos_pass}")

    out_dir = ROOT / "tmp/causal_regen_20260516/omega6_v2_oos_freeze_20260704"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "oos_freeze_report.json", "w") as f:
        json.dump(
            {
                "frozen_config": {k: v for k, v in FROZEN.__dict__.items() if k != "extra"},
                "validation": {tag: {k: v for k, v in val_result[tag].items() if k != "_trade_list"} for tag in ("cost1", "cost3")},
                "oos": {tag: {k: v for k, v in oos_result[tag].items() if k != "_trade_list"} for tag in ("cost1", "cost3")},
                "oos_pass": oos_pass,
                "oos_window_note": "tape caps at 2026-02-28, so this is Jan-Feb 2026 (~2 months), not the full 2026-01-01..03-31 default window",
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nreport: {out_dir / 'oos_freeze_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
