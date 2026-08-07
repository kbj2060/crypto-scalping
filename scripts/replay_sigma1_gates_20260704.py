#!/usr/bin/env python3
"""Sigma1 gate check on validation, with a DELIBERATELY SMALL pre-registered sweep to limit
the multiple-comparisons problem that plagued the 900+-variant Omega6 v2 searches:

  threshold in {0.45, 0.55, 0.65} x persistence in {0, 2} x frozen execution mechanics
  (ATR tp=15x / sl=5x / cooldown=12 / fixed margin 0.30 x leverage 2.0) = 18 configs total.

Gates (same pre-registered set as all prior rounds): val cost1 AND cost3 PnL > 0, MDD >= -20%
both tiers, trades >= 60, >= 3 months with trades. Validation only -- OOS is scored exactly once
by a separate step only if something passes here, with the explicit caveat that the Jan-Feb 2026
OOS window has already been examined twice this project (frozen-winner pass, barrier-matched
fail), so any Sigma1 OOS result is weaker evidence than a truly fresh window would give.
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

TAPE_PATH = ROOT / "tmp/causal_regen_20260516/sigma1_decision_tape_20260704/tape.parquet"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma1_gates_20260704"


def load_tape() -> pd.DataFrame:
    tape = pd.read_parquet(TAPE_PATH)
    tape["timestamp"] = pd.to_datetime(tape["timestamp"])
    return tape.sort_values("i").reset_index(drop=True)


def main() -> int:
    tape = load_tape()
    thresholds = [0.45, 0.55, 0.65]
    tapes_by_threshold = {thr: v2.apply_quality_threshold(tape, thr) for thr in thresholds}

    rows = []
    for thr, persist in itertools.product(thresholds, (0, 2, 3)):
        cfg = v2.VariantConfig(
            name=f"sigma1_qt{thr}_p{persist}",
            tp_mode="atr_scaled",
            tp_atr_mult=15.0,
            sl_atr_mult=5.0,
            sizing_mode="fixed",
            fixed_margin=0.30,
            fixed_leverage=2.0,
            cooldown_bars=12,
            quality_threshold=thr,
            persistence_bars=persist,
            use_fallback=False,
        )
        result = v2.cost_stress(tapes_by_threshold[thr], cfg, start=v2.VAL_START, end=v2.VAL_END)
        gate_pass = v2.passes_gates(result)
        row = {
            "name": cfg.name,
            "quality_threshold": thr,
            "persistence_bars": persist,
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
        print(json.dumps(row), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["gate_pass", "cost3_pnl"], ascending=[False, False])
    df.to_csv(OUT_DIR / "sigma1_gate_ranking.csv", index=False)
    print(f"\ntotal: {len(df)}  gate_pass: {int(df['gate_pass'].sum())}", flush=True)
    print(df.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
