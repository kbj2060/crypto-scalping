#!/usr/bin/env python3
"""Boundary check (docs/homer/README.md 5.6) for fib_extension_exhaustion's grid screening winner
H=20/GAP=12: GAP=12 was the largest value tested in GAP_GRID=[3,6,12], and it won (by
max(min(VAL,OOS))) at H=20 and 5 of the other 8 horizons -- not as unanimous as smt_divergence's
9/9 pattern, but frequent enough at the winning horizon's neighborhood to warrant a direct check
rather than assuming H=20/GAP=12 is a true peak on the GAP axis too. Cheap (2 extra combos) since
median raw-fire gap is already 160-166 bars (this signal fires sparsely) -- GAP beyond ~24 would
start merging genuinely-independent fires rather than deduplicating near-simultaneous re-triggers,
so this only tests GAP in {18, 24}, not an open-ended extension.
"""
from __future__ import annotations

import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from live_evidence_signal_dashboard_20260823 import compute_signals
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame, load_klines
from research_eth_fib_extension_exhaustion_metalabel_tabpfn_20260831 import screen_one_combo, log

GAP_EXT = [18, 24]
HORIZONS_TO_CHECK = [16, 20, 24]  # the winner + its immediate H-neighbors, each re-tested at larger GAP


def main() -> int:
    log("loading klines + building Tier0 indicator frame + compute_signals...")
    klines = load_klines()
    indicator_frame = build_indicator_frame(klines)
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)

    log(f"=== GAP extension check: HORIZON in {HORIZONS_TO_CHECK} x GAP in {GAP_EXT} ===")
    rows = []
    for horizon in HORIZONS_TO_CHECK:
        for gap in GAP_EXT:
            row, _ = screen_one_combo(indicator_frame, sig, horizon, gap)
            rows.append(row)

    best = max(rows, key=lambda r: min(r["val_auc"], r["oos_auc"]))
    log(f"best among extension combos: H={best['horizon']} GAP={best['gap']} "
        f"VAL={best['val_auc']:.4f} OOS={best['oos_auc']:.4f} min={min(best['val_auc'],best['oos_auc']):.4f}")
    log("reference -- original winner H=20/GAP=12: VAL=0.6044 OOS=0.6157 min=0.6044")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
