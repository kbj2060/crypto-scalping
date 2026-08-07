#!/usr/bin/env python3
"""Precompute a per-bar decision tape (L2 primary/fallback outputs + ATR) for Omega6.

Every backtest run so far spent most of its wall-clock time re-running primary/fallback TabM
inference bar-by-bar. Since L2 (primary/fallback direction+quality+expert+route) doesn't change
across L3/L4/L5/L6 architecture experiments, this script walks the frame ONCE and caches those
outputs to a parquet file, so downstream sizing/barrier/filter iteration can run in seconds
instead of minutes -- this is what makes an actual architecture-search loop tractable.

No lookahead: each row's primary/fallback output is computed from a trailing CONTEXT_BARS
window ending at that row, identical to what scripts/backtest_omega6_synthesis_fresh_forward_20260703.py
does live. Covers CONTEXT_BARS before VAL_START through OOS_END so both windows are available
without recomputation.
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

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import backtest_omega6_synthesis_fresh_forward_20260703 as bt  # noqa: E402
from trading_bot_modules.omega6_live import Omega6LiveAdapter  # noqa: E402

OUT_PATH = ROOT / "tmp/causal_regen_20260516/omega6_decision_tape_20260704/tape.parquet"


def main() -> int:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = Omega6LiveAdapter(
        primary_bundle_path=str(bt.DEFAULT_PRIMARY_BUNDLE),
        fallback_bundle_path=str(bt.DEFAULT_FALLBACK_BUNDLE),
        tcn_gate_path=str(bt.DEFAULT_TCN_GATE),
        risk_sidecar_path=str(bt.DEFAULT_RISK_SIDECAR),
        device=device,
        enable_l3_gate=False,
    )

    frame = bt._load_combined_frame()
    val_start_idx, _ = bt._window_bounds(frame, bt.VAL_START, bt.VAL_END)
    _, oos_end_idx = bt._window_bounds(frame, bt.OOS_START, bt.OOS_END)
    start_idx = max(0, val_start_idx - bt.CONTEXT_BARS)
    end_idx = oos_end_idx

    rows: list[dict] = []
    total = end_idx - start_idx
    for n, i in enumerate(range(start_idx, end_idx)):
        window = frame.iloc[max(0, i - bt.CONTEXT_BARS + 1) : i + 1]
        p = adapter._predict_parent(adapter.primary, window)
        f = adapter._predict_parent(adapter.fallback, window)
        atr = adapter._atr_pct(window, adapter.atr_window)
        row = frame.iloc[i]
        rows.append(
            {
                "i": int(i),
                "timestamp": row["timestamp"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "jump_flag": float(row.get("jump_flag", 0.0) or 0.0),
                "evt_tail_flag": float(row.get("evt_tail_flag", 0.0) or 0.0),
                "jump_z": float(row.get("jump_z", 0.0) or 0.0),
                "atr_pct": float(atr),
                "primary_action": int(p["action"]),
                "primary_side": int(p["side"]),
                "primary_expert": p["expert"],
                "primary_route_confidence": float(p["route_confidence"]),
                "primary_route_margin": float(p["route_margin"]),
                "primary_dir_p_cash": float(p["direction"][0]),
                "primary_dir_p_long": float(p["direction"][1]),
                "primary_dir_p_short": float(p["direction"][2]),
                "primary_quality_p_cash": float(p["quality"][0]),
                "primary_quality_p_long": float(p["quality"][1]),
                "primary_quality_p_short": float(p["quality"][2]),
                "primary_quality_score": float(p["quality_score"]),
                "primary_confidence": float(p["confidence"]),
                "fallback_action": int(f["action"]),
                "fallback_side": int(f["side"]),
                "fallback_expert": f["expert"],
                "fallback_route_confidence": float(f["route_confidence"]),
                "fallback_route_margin": float(f["route_margin"]),
                "fallback_dir_p_cash": float(f["direction"][0]),
                "fallback_dir_p_long": float(f["direction"][1]),
                "fallback_dir_p_short": float(f["direction"][2]),
                "fallback_quality_p_cash": float(f["quality"][0]),
                "fallback_quality_p_long": float(f["quality"][1]),
                "fallback_quality_p_short": float(f["quality"][2]),
                "fallback_quality_score": float(f["quality_score"]),
                "fallback_confidence": float(f["confidence"]),
            }
        )
        if n % 5000 == 0:
            print(f"{n}/{total}", flush=True)

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"wrote {len(out)} rows to {OUT_PATH}", flush=True)
    print(f"val_start_idx={val_start_idx} start_idx={start_idx} end_idx={end_idx} context_bars={bt.CONTEXT_BARS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
