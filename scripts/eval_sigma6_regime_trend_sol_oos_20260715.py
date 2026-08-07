"""One-shot OOS evaluation of the VAL-selected SOL Sigma6 config, per the pre-registered
discipline in run_sigma6_regime_trend_sol_20260715.py (VAL-only search, freeze, test once).

VAL-best config (from val_regime_frontier_sol.csv, top row by cost1):
quality_threshold=0.70, leverage=3.0, sl_atr=1.5, regime_mode=not_chop, reg_thr=0.34, stab_thr=0.0.

OOS window: CLAUDE.md's canonical frozen fresh-forward window (2026-01-01..2026-03-31) -- the
first and only use of this window for this exact SOL Sigma6 model/config combination.
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

import run_sigma6_regime_trend_sol_20260715 as sol6  # noqa: E402
import replay_omega6_v2_variants_20260704 as v2  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma6_regime_trend_sol_20260715"
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31 23:59:59")

FROZEN = dict(quality_threshold=0.70, leverage=3.0, sl_atr=1.5, reg_mode="not_chop", reg_thr=0.34, stab_thr=0.0)


def main() -> int:
    raw = sol6.load_tape_with_regime()
    tape = v2.apply_quality_threshold(raw, FROZEN["quality_threshold"])
    base = dict(margin=0.30, trail_atr=5.0, min_profit_atr=2.0, max_hold=144, cooldown=3)
    result = sol6.backtest(
        tape,
        leverage=FROZEN["leverage"], sl_atr=FROZEN["sl_atr"],
        reg_mode=FROZEN["reg_mode"], reg_thr=FROZEN["reg_thr"], stab_thr=FROZEN["stab_thr"],
        fee_mult=1.0, start=OOS_START, end=OOS_END, **base,
    )
    report = {
        "frozen_config": FROZEN,
        "oos_window": [str(OOS_START), str(OOS_END)],
        "oos_result": result,
        "val_result_for_reference": "cost1 +29.1%, mdd -23.5%, trades 54, wr 0.315 (val_regime_frontier_sol.csv top row)",
        "first_use_of_window": "true -- this exact SOL Sigma6 model/config has not scored this window before",
    }
    (OUT_DIR / "oos_report_sol.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
