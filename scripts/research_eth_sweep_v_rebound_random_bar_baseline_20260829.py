#!/usr/bin/env python3
"""Unconditional ('any random bar') baseline for the V_REBOUND label -- user asked for a lift-
vs-random-bar number directly comparable to the 8 dashboard evidence-signal chips' event-study
methodology (scripts/research_eth_evidence_signal_scorecard_ci_20260825.py's precision/
baseline_rate/lift columns), since the model's own naive-majority-class baseline (54.6-56.1%,
see eth_liquidity_sweep_v_rebound_feature_plan_20260829.md) answers a different question --
it's already conditioned on a liquidity_sweep having fired.

Applies the EXACT SAME formula as build_eth_5m_liquidity_sweep_v_rebound_labels_20260829.py::
label_events (imported, not reimplemented) to EVERY bar in the 2024-01-01+ series, vectorized,
WITHOUT the sweep trigger gate (row.low < level and row.close > level) -- sweep_level_low/high
and atr already exist causally for every bar via add_causal_columns (rolling window, no
sweep-specific dependency), so dropping the gate is the only change; the move/confirmed/label
arithmetic is untouched.

Self-checks against the actual saved label file (14,259 real sweep events, row-for-row on
candidate_index) before trusting the resulting baseline/lift numbers -- if the vectorized
unconditional formula doesn't reduce EXACTLY to the known conditional labels when re-filtered
to real sweep bars, something is wrong and the script aborts rather than reporting numbers.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWEEP_IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
EXISTING_LABELS = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_labels.csv"
OUT_DIR = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829"
LOOKAHEAD_BARS = 6
V_REBOUND_ATR_MULT = 1.5


def load_sweep_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_20260829", SWEEP_IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    sweep_impl = load_sweep_impl()
    frame = sweep_impl.add_causal_columns(sweep_impl.load_5m(sweep_impl.ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"))
    n = len(frame)
    close, high, low = frame["close"], frame["high"], frame["low"]
    level_low, level_high, atr = frame["sweep_level_low"], frame["sweep_level_high"], frame["atr"]

    future_closes = [close.shift(-k) for k in range(1, LOOKAHEAD_BARS + 1)]
    future_highs = pd.concat([high.shift(-k) for k in range(1, LOOKAHEAD_BARS + 1)], axis=1)
    future_lows = pd.concat([low.shift(-k) for k in range(1, LOOKAHEAD_BARS + 1)], axis=1)
    future_high_max = future_highs.max(axis=1)
    future_low_min = future_lows.min(axis=1)

    atr_ok = atr.notna() & (atr > 0)

    # downside/"bottom": move off this bar's own low, confirmation vs the causal swept-low level
    move_down = future_high_max - low
    confirmed_down = pd.concat([future_closes[k] > level_low for k in range(LOOKAHEAD_BARS)], axis=1).all(axis=1)
    label_down = atr_ok & level_low.notna() & (move_down >= V_REBOUND_ATR_MULT * atr) & confirmed_down
    is_real_sweep_down = level_low.notna() & (low < level_low) & (close > level_low)

    # upside/"top"
    move_up = high - future_low_min
    confirmed_up = pd.concat([future_closes[k] < level_high for k in range(LOOKAHEAD_BARS)], axis=1).all(axis=1)
    label_up = atr_ok & level_high.notna() & (move_up >= V_REBOUND_ATR_MULT * atr) & confirmed_up
    is_real_sweep_up = level_high.notna() & (high > level_high) & (close < level_high)

    # match the original label_events() eligibility window exactly: index in
    # [SWEEP_LOOKBACK_BARS, len(frame) - LOOKAHEAD_BARS)
    eligible = np.zeros(n, dtype=bool)
    eligible[sweep_impl.SWEEP_LOOKBACK_BARS: n - LOOKAHEAD_BARS] = True

    # --- self-check vs the actual saved label file, row-for-row on candidate_index ---
    existing = pd.read_csv(EXISTING_LABELS)
    mismatches = 0
    for side, label_series, real_mask in (
        ("downside", label_down, is_real_sweep_down),
        ("upside", label_up, is_real_sweep_up),
    ):
        existing_side = existing.loc[existing["side"] == side].set_index("candidate_index")["label"]
        recomputed_real_idx = np.flatnonzero(eligible & real_mask.to_numpy())
        if set(recomputed_real_idx.tolist()) != set(existing_side.index.tolist()):
            raise RuntimeError(
                f"{side}: recomputed real-sweep index set doesn't match existing label file "
                f"({len(recomputed_real_idx)} vs {len(existing_side)}) -- aborting, do not trust output."
            )
        recomputed_labels = label_series.iloc[existing_side.index].astype(int)
        diff = (recomputed_labels.to_numpy() != existing_side.to_numpy()).sum()
        mismatches += diff
        if diff:
            raise RuntimeError(f"{side}: {diff}/{len(existing_side)} label mismatches vs existing file -- aborting.")
    print(f"self-check OK: 0 mismatches across {len(existing)} real sweep events (both sides), exact row-for-row match.")

    def rates(label_series: pd.Series, real_mask: pd.Series) -> dict:
        elig = eligible
        baseline = float(label_series[elig].mean())
        n_real = int((elig & real_mask.to_numpy()).sum())
        conditional = float(label_series[elig & real_mask.to_numpy()].mean())
        return {
            "n_eligible_bars": int(elig.sum()),
            "n_real_sweep_events": n_real,
            "baseline_rate_random_bar": round(baseline, 4),
            "conditional_rate_given_sweep": round(conditional, 4),
            "lift_vs_random_bar": round(conditional / baseline, 4) if baseline > 0 else None,
        }

    result = {
        "definition": (
            f"V_REBOUND: within {LOOKAHEAD_BARS*5} min, price moves >= {V_REBOUND_ATR_MULT}x ATR(14) "
            "in the reversal direction off this bar's own extreme, AND all 6 bars close beyond the "
            "causal 48-bar swept level -- IDENTICAL formula to the real label, just not gated on the "
            "sweep condition itself (row.low<level & row.close>level, or the upside mirror)"
        ),
        "source_period": {"start": str(frame["timestamp"].min()), "end": str(frame["timestamp"].max())},
        "downside_bottom": rates(label_down, is_real_sweep_down),
        "upside_top": rates(label_up, is_real_sweep_up),
        "self_check_mismatches": int(mismatches),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    (OUT_DIR / "random_bar_baseline_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
