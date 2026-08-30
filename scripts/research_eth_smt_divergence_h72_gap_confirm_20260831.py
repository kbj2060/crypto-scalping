#!/usr/bin/env python3
"""Confirms GAP=12's 9/9 dominance pattern still holds at H=72 (the new local-peak winner found
after fixing the K_GRID ceiling bug) by testing GAP=3/6 there too, rather than assuming."""
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
from research_eth_smt_divergence_metalabel_tabpfn_20260831 import load_btc_klines, screen_one_combo


def log(msg): print(f"[smt_h72_gap_confirm] {msg}", flush=True)


def main() -> int:
    klines = load_klines()
    btc = load_btc_klines()
    indicator_frame = build_indicator_frame(klines)
    sig = compute_signals(klines, btc_df=btc, funding_df=None).reset_index(drop=True)

    for gap in [3, 6, 12]:
        row, _ = screen_one_combo(indicator_frame, sig, 72, gap)
        log(f"  H=72 gap={gap:>2d}: min(VAL,OOS)={min(row['val_auc'], row['oos_auc']):.4f} gap_diff={row['gap_val_oos']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
