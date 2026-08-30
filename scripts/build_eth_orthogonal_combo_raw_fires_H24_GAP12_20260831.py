#!/usr/bin/env python3
"""Rebuild orthogonal_combo's RAW fires (H=24/GAP=12, the v2-final selected config) WITHOUT the
exclude-middle filter, for the trailing-stop cost-gate test.

data/labels/eth_5m_orthogonal_combo_metalabel_20260830/eth_5m_orthogonal_combo_metalabel_features.csv
(the classifier training CSV) already has apply_exclude_middle() applied (2334 raw fires -> 1493
kept, per tmp/eth_orthogonal_combo_metalabel_tabpfn_20260830/report.json) -- using it directly for
the cost-gate would silently drop the ~36% of real dashboard fires whose eventual MFE landed in the
ambiguous middle zone, which is NOT what the live dashboard's net_score/votes actually fire on
(unconditional on any label). Per this project's own liquidity_sweep precedent
(backtest_eth_liquidity_sweep_topdown_trailing_gridsearch_20260830.py docstring): the cost-gate must
test ALL raw fires of the underlying EVENT, unconditional on the trained model's own probability
(and, by the same logic, unconditional on the exclude-middle label decision, which is purely a
classifier-training device).

Reuses build_raw_fires() verbatim from research_eth_orthogonal_combo_metalabel_tabpfn_20260830.py
(same clustering/anchor logic, same dropna(FEATURE_COLUMNS) data-quality gate as the training
pipeline -- 2334 raw fires before AND after dropna at this config, so this changes nothing here)
minus apply_exclude_middle(). No TabPFN/CUDA needed (build_raw_fires is pure pandas/numpy) --
runs locally.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd

from research_eth_orthogonal_combo_metalabel_tabpfn_20260830 import (
    FEATURE_COLUMNS, build_raw_fires, load_funding_z,
)
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame, load_klines
from live_evidence_signal_dashboard_20260823 import compute_signals

OUT_PATH = ROOT / "data/labels/eth_5m_orthogonal_combo_metalabel_20260830/eth_5m_orthogonal_combo_metalabel_features_H24_GAP12_ALLFIRES.csv"
HORIZON, GAP = 24, 12


def log(msg: str) -> None:
    print(f"[orthogonal_combo_raw_fires] {msg}", flush=True)


def main() -> int:
    log("loading klines + building Tier0 indicator frame + funding_z + compute_signals...")
    klines = load_klines()
    indicator_frame = build_indicator_frame(klines)
    funding_df = load_funding_z()
    sig = compute_signals(klines, btc_df=None, funding_df=funding_df).reset_index(drop=True)
    assert len(sig) == len(indicator_frame) and (sig["timestamp"].to_numpy() == indicator_frame["timestamp"].to_numpy()).all()

    fires_raw = build_raw_fires(indicator_frame, sig, GAP, HORIZON)
    n_before = len(fires_raw)
    fires_raw = fires_raw.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    log(f"H={HORIZON} GAP={GAP}: raw fires before dropna={n_before}, after dropna={len(fires_raw)} "
        f"(bottom={int((fires_raw['side']=='bottom').sum())}, top={int((fires_raw['side']=='top').sum())})")
    assert len(fires_raw) == 2334, f"expected 2334 raw fires per report.json, got {len(fires_raw)}"

    fires_raw.to_csv(OUT_PATH, index=False)
    log(f"saved ALL raw fires (no exclude-middle) -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
