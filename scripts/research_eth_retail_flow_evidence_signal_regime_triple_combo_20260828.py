#!/usr/bin/env python3
"""User question 2026-08-28: does adding GBM3 regime (chop-gate) as a THIRD dimension on top of the
already-tested evidence-signal x nif_retail(리테일 수급) confirmation combo (research_eth_retail_
flow_evidence_signal_combo_20260825.py, 2026-08-25) show any directional edge the 2-way combo
didn't? That 2-way result was MIXED/null (roughly half the signal/side pairs improved, half got
worse, no consistent direction, n shrinking a lot -- e.g. orthogonal_combo:top 5.86->4.90, smt_
divergence:bottom 3.59->3.19) -- not previously re-tested with a regime dimension added.

Reuses build_frame()/load_nif_retail_5m() from that script verbatim (same window 2026-05-03~
2026-07-20, same nif_retail_z<=-0.5/>=0.5 confirmation threshold, same event_study() lift
methodology) and _regime_labels() from backtest_eth_evidence_signal_chop_gated_costgate_20260827.py
verbatim (same GBM3 model already used everywhere else this session) -- no new formulas, purely a
new stratification on top of two already-established pieces.

Diagnostic only (event_study lift vs zigzag pivots), NOT a cost-gated backtest -- matches the
2-way combo script's own scope. No pre-registered gate covers this specific 3-way combination (the
09-15 liquidation-crowding gate is unrelated), so this is legitimate exploratory research.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import K_HORIZONS, event_study, load_zigzag_pivots  # noqa: E402
from backtest_eth_evidence_signal_chop_gated_costgate_20260827 import _regime_labels  # noqa: E402
from research_eth_retail_flow_evidence_signal_combo_20260825 import DATA_PATH, WIN_END, WIN_START, build_frame  # noqa: E402

CANDIDATES = ["orthogonal_combo", "liquidity_sweep", "fib_extension_exhaustion", "smt_divergence", "dalton_rule2_balance_edge"]


def main() -> None:
    print("Building evidence-signal + nif_retail frame (2026-05-03..2026-07-20)...")
    frame = build_frame()

    print("Adding GBM3 regime_label (same model used all session)...")
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    regime = _regime_labels(raw)
    raw_regime = pd.DataFrame({"timestamp": raw["timestamp"], "regime_label": regime.to_numpy()})
    frame = frame.merge(raw_regime, on="timestamp", how="left")

    ts = frame["timestamp"]
    window_mask = ((ts >= WIN_START) & (ts <= WIN_END)).to_numpy()
    chop_mask = window_mask & (frame["regime_label"] == "chop").to_numpy()
    print(f"Window: {WIN_START.date()}..{WIN_END.date()}, {int(window_mask.sum())} bars, "
          f"chop share in window: {chop_mask.sum() / max(window_mask.sum(), 1) * 100:.1f}%")

    pivots = load_zigzag_pivots()
    all_pos = np.flatnonzero(window_mask)
    K = K_HORIZONS["K12_1h"]

    rows = []
    for name in CANDIDATES:
        for side in ("bottom", "top"):
            base_col = f"{side}_{name}"
            base_sig = frame[base_col].fillna(False).to_numpy()
            retail_agrees = ((frame["nif_retail_z"] <= -0.5) if side == "bottom" else (frame["nif_retail_z"] >= 0.5)).fillna(False).to_numpy()

            side_pivots = pivots.loc[pivots["pivot_type"] == side]
            pivot_pos = frame.index[frame["timestamp"].isin(side_pivots["timestamp"])].to_numpy()

            variants = {
                "base": base_sig & window_mask,
                "+retail": base_sig & retail_agrees & window_mask,
                "+chop": base_sig & chop_mask,
                "+retail+chop": base_sig & retail_agrees & chop_mask,
            }
            row = {"signal": name, "side": side}
            for vname, mask in variants.items():
                stats = event_study(np.flatnonzero(mask), pivot_pos, all_pos, K)
                row[f"{vname}_n"] = stats["n_triggers"]
                row[f"{vname}_lift"] = stats["lift"]
            rows.append(row)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    cols = ["signal", "side", "base_n", "base_lift", "+retail_n", "+retail_lift", "+chop_n", "+chop_lift", "+retail+chop_n", "+retail+chop_lift"]
    print("\n=== base vs +retail vs +chop vs +retail+chop, 1h lift ===")
    print(res[cols].round(3).to_string(index=False))

    out_dir = ROOT / "tmp" / "eth_retail_flow_evidence_signal_regime_triple_combo_20260828"
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "triple_combo_table.csv", index=False)
    print(f"\nWrote {out_dir / 'triple_combo_table.csv'}")


if __name__ == "__main__":
    main()
