#!/usr/bin/env python3
"""Economic gate for G2_fib_extension_exhaustion, the one signal from the geometric/Fibonacci
evidence scan (analyze_eth_fibonacci_harmonic_geometric_evidence_20260824.py) that showed
lift comparable to the plain sweep reference (bottom 3.27x/top 2.32x pooled, low ~10% overlap
with sweep, VAL/OOS same-direction). Per this repo's most-repeated lesson (lift is a probability
measure, price impact is a currency measure -- the 2026-08-24 evidence-signal economic-gate
round found the STRONGEST-lift signals had the WORST gross economics because "extreme condition
fires mid-move, not at the reversal"), this signal must clear a market-order economic check
before being treated as anything beyond a lift curiosity. Same convention as that round: forward
return from the firing bar's close to the K-bar-later close, non-overlapping, standard costs
only (10bp taker / 6.2bp maker realized), net>0 and |t|>=2 required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_eth_amt_vsa_footprint_ifvg_component_evidence_20260815 import add_sweep, load_frame  # noqa: E402
from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import OOS_END  # noqa: E402
from analyze_eth_fibonacci_harmonic_geometric_evidence_20260824 import add_fib_zones, add_leg_direction  # noqa: E402
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START,
    VAL_END,
    VAL_START,
    compute_indicators,
)

COSTS_RT = {"taker10bp": 10e-4, "maker6.2bp": 6.2e-4}
K_HORIZONS = {"K3_15m": 3, "K6_30m": 6, "K12_1h": 12, "K24_2h": 24}


def non_overlapping(trigger_pos: np.ndarray, K: int) -> np.ndarray:
    """Greedy left-to-right selection so consecutive fires within K bars don't double-count."""
    kept = []
    last_end = -1
    for p in trigger_pos:
        if p > last_end:
            kept.append(p)
            last_end = p + K
    return np.array(kept, dtype=np.int64)


def gate_cell(close: np.ndarray, trigger_pos: np.ndarray, K: int, side: str) -> dict:
    pos = non_overlapping(trigger_pos, K)
    pos = pos[pos + K < len(close)]
    if len(pos) == 0:
        return {"n": 0, "gross_bp": float("nan"), "t": float("nan")}
    fwd_ret = (close[pos + K] - close[pos]) / close[pos]
    signed = fwd_ret if side == "bottom" else -fwd_ret   # bottom expects reversal UP, top expects reversal DOWN
    t, p = stats.ttest_1samp(signed, 0.0) if len(signed) > 1 else (float("nan"), float("nan"))
    return {"n": int(len(signed)), "gross_bp": float(signed.mean() * 1e4), "t": float(t), "signed_ret": signed}


def main() -> None:
    raw = load_frame()
    f = compute_indicators(raw).reset_index(drop=True)
    f = add_sweep(f)
    f = add_leg_direction(f)
    f = add_fib_zones(f)
    close = f["close"].to_numpy()

    ts = f["timestamp"]
    pooled_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()

    rows = []
    for side, col in (("bottom", "fib_extension_exhaustion_bottom"), ("top", "fib_extension_exhaustion_top")):
        sig = f[col].to_numpy() & pooled_mask
        trigger_pos = np.flatnonzero(sig)
        for k_name, K in K_HORIZONS.items():
            cell = gate_cell(close, trigger_pos, K, side)
            for cost_name, cost in COSTS_RT.items():
                net_bp = cell["gross_bp"] - cost * 1e4 if cell["n"] else float("nan")
                rows.append({"side": side, "horizon": k_name, "cost": cost_name, "n": cell["n"],
                             "gross_bp": cell["gross_bp"], "net_bp": net_bp, "t": cell["t"],
                             "pass": bool(cell["n"] and net_bp > 0 and abs(cell["t"]) >= 2.0)})

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"Pooled window bars={int(pooled_mask.sum())}\n")
    for cost_name in COSTS_RT:
        print(f"\n===== {cost_name} =====")
        print(res[res["cost"] == cost_name].to_string(index=False))

    n_pass = int(res["pass"].sum())
    print(f"\nPASS cells (net>0 & |t|>=2): {n_pass}/{len(res)}")
    if n_pass:
        print(res[res["pass"]].to_string(index=False))

    out_dir = ROOT / "tmp" / "eth_fib_extension_exhaustion_economic_gate_20260824"
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "gate_table.csv", index=False)
    print(f"\nWrote {out_dir / 'gate_table.csv'}")


if __name__ == "__main__":
    main()
