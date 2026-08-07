"""OOS evaluation of the two trade-count-stabilized VAL-selected SOL Sigma6 configs from
search_sigma6_regime_trend_sol_stable_20260715.py (trades>=30 enforced during search).

Honesty note: this is the SECOND use of the 2026-01-01..03-31 OOS window for a SOL Sigma6
config (the first use tested thr=0.7/lev=3/sl=1.5/not_chop/rthr=0.34 from the narrow grid and
failed: pnl -15.88%). Its evidential value as "unseen" is correspondingly degraded for this
second look, same caveat the ETH Sigma6 doc makes about window reuse.

Candidates (both selected on VAL only, trades>=30 required):
- high_return: thr=0.7, lev=3.0, sl=2.5, trail=4.0, minp=2.5, maxh=72, cd=3, not_chop, rthr=0.30
  (VAL: pnl +47.3%, mdd -27.3%, trades 60, wr 0.450, calmar 1.73)
- low_risk:    thr=0.7, lev=1.5, sl=2.5, trail=4.0, minp=2.5, maxh=72, cd=3, not_chop, rthr=0.26
  (VAL: pnl +17.6%, mdd -12.5%, trades 48, wr 0.521, calmar 1.41)
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

CANDIDATES = {
    "high_return": dict(thr=0.70, lev=3.0, sl=2.5, trail=4.0, minp=2.5, maxh=72, cd=3, mode="not_chop", rthr=0.30,
                         val_ref="pnl+47.3%/mdd-27.3%/60tr/wr0.450/calmar1.73"),
    "low_risk": dict(thr=0.70, lev=1.5, sl=2.5, trail=4.0, minp=2.5, maxh=72, cd=3, mode="not_chop", rthr=0.26,
                      val_ref="pnl+17.6%/mdd-12.5%/48tr/wr0.521/calmar1.41"),
}


def main() -> int:
    raw = sol6.load_tape_with_regime()
    report = {"oos_window": [str(OOS_START), str(OOS_END)], "window_reuse_note": "second use of this OOS window for a SOL Sigma6 config; first use (thr0.7/lev3/sl1.5/not_chop/rthr0.34) failed pnl-15.88%", "results": {}}
    for name, cfg in CANDIDATES.items():
        tape = v2.apply_quality_threshold(raw, cfg["thr"])
        result = sol6.backtest(
            tape, leverage=cfg["lev"], margin=0.30, trail_atr=cfg["trail"], sl_atr=cfg["sl"],
            min_profit_atr=cfg["minp"], max_hold=cfg["maxh"], cooldown=cfg["cd"],
            reg_mode=cfg["mode"], reg_thr=cfg["rthr"], stab_thr=0.0, fee_mult=1.0,
            start=OOS_START, end=OOS_END,
        )
        report["results"][name] = {"config": cfg, "oos": result}
        print(f"{name}: pnl={result['pnl']:.2f}% mdd={result['mdd']:.2f}% trades={result['trades']} wr={result['wr']:.3f} by_month={result['by_month']}", flush=True)
    (OUT_DIR / "oos_report_sol_stable.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
