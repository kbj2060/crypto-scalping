#!/usr/bin/env python3
"""Extension of research_eth_smt_divergence_metalabel_tabpfn_20260831.py's HORIZON grid: the
original grid (6,8,12,16,20,24,30,36,48 bars x GAP 3,6,12) found min(VAL,OOS) STILL CLIMBING at
H=48 (the grid boundary, 0.5874->0.5414->...->0.6218 monotonic-ish increase at GAP=12) -- picking
a boundary value without checking further out would repeat the exact mistake this project's own
methodology (docs/homer/README.md 5.5) warns against. GAP=12 won at all 9/9 original horizons
tested (large, consistent margin) -- extending ONLY GAP=12 for the new horizons is justified by
that already-observed 9/9 pattern, not cherry-picking; GAP=3/6 are not re-tested here.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from live_evidence_signal_dashboard_20260823 import compute_signals
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame, load_klines
from research_eth_smt_divergence_metalabel_tabpfn_20260831 import (
    load_btc_klines, screen_one_combo,
)

HORIZON_GRID_EXT = [60, 72, 84, 96]
GAP = 12


def log(msg): print(f"[smt_divergence_horizon_extend] {msg}", flush=True)


def main() -> int:
    log("loading klines + BTC klines + building Tier0 indicator frame + compute_signals...")
    klines = load_klines()
    btc = load_btc_klines()
    indicator_frame = build_indicator_frame(klines)
    sig = compute_signals(klines, btc_df=btc, funding_df=None).reset_index(drop=True)
    assert len(sig) == len(indicator_frame)

    log(f"=== extended grid: HORIZON in {HORIZON_GRID_EXT} x GAP={GAP} only (GAP=12 won 9/9 in the original grid) ===")
    rows = []
    for horizon in HORIZON_GRID_EXT:
        row, _ = screen_one_combo(indicator_frame, sig, horizon, GAP)
        rows.append(row)

    log("\n=== extended results (min(VAL,OOS)) ===")
    for r in rows:
        log(f"  H={r['horizon']:>3d} K={r['k']:.2f} VAL={r['val_auc']:.4f} OOS={r['oos_auc']:.4f} "
            f"min={min(r['val_auc'], r['oos_auc']):.4f} gap={r['gap_val_oos']:.4f}")

    # reference: original H=36/48 @ GAP=12 for continuity
    log("\n(for comparison, original grid @ GAP=12: H=36 min=0.6079 gap=0.0560 | H=48 min=0.6218 gap=0.0415)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
