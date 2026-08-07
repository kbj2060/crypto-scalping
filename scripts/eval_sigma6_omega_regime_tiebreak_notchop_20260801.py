#!/usr/bin/env python3
"""Tests a chop-aware variant of ETH's regime_tiebreak rule -- NEVER tried before, on either asset.
The existing regime_tiebreak (eval_sigma6_omega_rule_and_meta_allocation_20260801.py, rule
'regime_tiebreak') picks regime_side = argmax(bull_prob, bear_prob) and ignores chop_prob entirely,
even though Sigma6's own trend-follower ENTRY gate (mode='not_chop') distrusts chop bars for that
different purpose. This script asks: on CONFLICT bars where chop_prob is actually the model's
dominant call (chop_prob > max(bull_prob, bear_prob)), does the regime model have anything
trustworthy to say about direction at all? If not, falling back to baseline (both legs kept at
weight 1.0, no bias) on those bars, instead of forcing a bull/bear pick, might do better.

Same selection discipline as the original rule sweep: selected on VAL_2025Q4 only (comparing
against the ALREADY-VALIDATED plain regime_tiebreak, not just raw baseline), then confirmed
(not re-picked) on OOS_2026H1. This is a genuinely new mechanism test, not a re-run of anything
already validated for ETH -- treat results accordingly (exploratory, not confirmatory).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from train_eval_sigma6_omega_rl_meta_controller_20260801 import build_bar_frame  # noqa: E402
from eval_sigma6_omega_rule_and_meta_allocation_20260801 import weighted_pnl, rule_weights  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260801/sigma6_omega_regime_tiebreak_notchop"


def notchop_tiebreak_weights(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = len(frame)
    w_om, w_s6 = np.ones(n), np.ones(n)
    conflict = frame["conflict"].to_numpy() > 0
    bull = frame["bull_prob"].to_numpy()
    bear = frame["bear_prob"].to_numpy()
    chop = frame["chop_prob"].to_numpy()
    om_side, s6_side = frame["omega_side"].to_numpy(), frame["sigma6_side"].to_numpy()
    regime_side = np.where(bull >= bear, 1, -1)
    chop_is_dominant = chop > np.maximum(bull, bear)
    trust = conflict & ~chop_is_dominant  # only apply the tiebreak pick when chop ISN'T the top call
    w_om[trust] = np.where(om_side[trust] == regime_side[trust], 1.0, 0.0)
    w_s6[trust] = np.where(s6_side[trust] == regime_side[trust], 1.0, 0.0)
    # conflict bars where chop dominates: left at baseline 1.0/1.0 (no bias applied)
    return w_om, w_s6


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_val = build_bar_frame("VAL_2025Q4")
    frame_oos = build_bar_frame("OOS_2026H1")

    for label, frame in (("VAL_2025Q4", frame_val), ("OOS_2026H1", frame_oos)):
        conflict = frame["conflict"].to_numpy() > 0
        chop = frame["chop_prob"].to_numpy()
        bull = frame["bull_prob"].to_numpy()
        bear = frame["bear_prob"].to_numpy()
        chop_dominant_frac = float((chop[conflict] > np.maximum(bull, bear)[conflict]).mean()) if conflict.sum() else 0.0

        w_om_base, w_s6_base = rule_weights(frame, "baseline")
        w_om_plain, w_s6_plain = rule_weights(frame, "regime_tiebreak")
        w_om_nc, w_s6_nc = notchop_tiebreak_weights(frame)

        base = weighted_pnl(frame, w_om_base, w_s6_base)
        plain = weighted_pnl(frame, w_om_plain, w_s6_plain)
        notchop = weighted_pnl(frame, w_om_nc, w_s6_nc)

        print(f"\n=== {label} (n_conflict_bars={int(conflict.sum())}, "
              f"chop-dominant among conflict bars={chop_dominant_frac*100:.1f}%) ===")
        print(f"baseline (1x-1x)          : pnl={base['pnl_pct']:+.2f}% mdd={base['mdd_pct']:.2f}%")
        print(f"regime_tiebreak (plain)   : pnl={plain['pnl_pct']:+.2f}% mdd={plain['mdd_pct']:.2f}%")
        print(f"regime_tiebreak (notchop) : pnl={notchop['pnl_pct']:+.2f}% mdd={notchop['mdd_pct']:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
