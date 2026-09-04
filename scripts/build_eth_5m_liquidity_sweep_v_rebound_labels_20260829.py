#!/usr/bin/env python3
"""Binary label: did the dashboard liquidity_sweep event V-rebound within 30 minutes?

label 1 = V_REBOUND: within 6 bars (30 minutes) of the sweep bar, price moves at least
  1.5x ATR(14) away from the sweep extreme in the reversal direction, AND EVERY one of
  those 6 bars closes beyond the swept level in that same direction (the reclaim held for
  the whole window, not just at one sampled instant).
label 0 = NO_V_REBOUND: every other sweep (original-direction continuation, slow/partial
  reaction, chop). No events are excluded as ambiguous -- every raw sweep gets a label.

Three corrections on top of the first version, all verified as clean one-directional
tightenings (only remove weak/noisy positives, never introduce new ones) against the full
14,259-event population -- see docs/experiments/eth_liquidity_sweep_frequency_investigation_20260829.md
for the v1-v3 history:
  v2: confirmation compares to the swept level, not the sweep bar's own close (comparing to
      the sweep bar's close was too strict whenever that bar itself closed well beyond the
      level on a strong reclaim candle -- 53.3% -> 64.7% V_REBOUND, 1624 clean 0->1 flips).
  v3: confirmation requires ALL 6 future bars beyond the level (not just the last one), and
      the ATR multiple was raised 1.0x -> 1.5x -- a single final-bar snapshot let noisy chop
      that spent most of the window drifting back toward (or through) the level pass as
      "confirmed" purely by where the 30-minute mark happened to land, and weak ~1.0-1.3x
      moves rarely look like a real V by eye. Caught by the user visually inspecting
      render_eth_5m_sweep_v_rebound_label_examples_20260829.py's output a second time.
  v4 (2026-08-30, user code review): two fixes to the threshold itself, both label-only --
      neither touches add_causal_columns, so the Tier0 "atr"/"sweep_penetration_atr" FEATURES
      TabPFN sees are unchanged; only the ground-truth threshold moves.
      (a) ATR was self-inclusive: row["atr"] is a 14-bar mean of true range ENDING AT the sweep
          bar itself, so the sweep bar's own (typically outsized, since a sweep is by
          definition an extreme wick) range inflated the very yardstick used to judge it --
          systematically hardest for the most violent sweeps. Now uses frame["atr"].iloc[index-1],
          the last ATR reading computed entirely from the 14 bars BEFORE the sweep bar.
      (b) The 1.5x-ATR move had no timing requirement -- reaching it on bar 6 (minute 30)
          counted the same as reaching it on bar 1 (minute 5), so slow grinds counted as
          "V-shaped" equally with sharp spikes. Now the move is measured only over the first
          V_REBOUND_FAST_BARS bars (15 minutes, half the window) instead of all 6; the
          6-bar/30-minute hold check (confirmed) is unchanged -- "fast" and "held" are
          separate questions.

The sweep trigger itself is reused unmodified (via import, not reimplemented) from
build_eth_5m_sweep_followthrough_v2_labels_20260829.py::add_causal_columns, which restates
the live evidence-signal dashboard's `liquidity_sweep` definition: a bar's low/high pierces
the prior 48-bar (causal, shifted) swing low/high and the close reclaims back inside it.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829"
BAR_MINUTES = 5
LOOKAHEAD_BARS = 6
V_REBOUND_ATR_MULT = 1.5
V_REBOUND_FAST_BARS = 3  # v4: the ATR-move must arrive within the first 15 of the 30 minutes


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_20260829", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def label_events(frame: pd.DataFrame, impl) -> pd.DataFrame:
    timestamps = frame["timestamp"].to_numpy()
    atr_series = frame["atr"].to_numpy()
    rows = []
    for index in range(impl.SWEEP_LOOKBACK_BARS, len(frame) - LOOKAHEAD_BARS):
        row = frame.iloc[index]
        # v4: pre-sweep ATR (14 bars strictly before the sweep bar) -- the sweep bar's own
        # true range no longer inflates the yardstick used to judge its own aftermath.
        atr = atr_series[index - 1]
        if not np.isfinite(atr) or atr <= 0:
            continue
        future = frame.iloc[index + 1:index + LOOKAHEAD_BARS + 1]
        fast_future = future.iloc[:V_REBOUND_FAST_BARS]

        level = row["sweep_level_low"]
        if np.isfinite(level) and row["low"] < level and row["close"] > level:
            move = float(fast_future["high"].max() - row["low"])
            confirmed = bool((future["close"] > level).all())
            rows.append(_row(timestamps, index, "downside", level, atr, move, confirmed))

        level = row["sweep_level_high"]
        if np.isfinite(level) and row["high"] > level and row["close"] < level:
            move = float(row["high"] - fast_future["low"].min())
            confirmed = bool((future["close"] < level).all())
            rows.append(_row(timestamps, index, "upside", level, atr, move, confirmed))
    return pd.DataFrame(rows)


def _row(timestamps, index, side, level, atr, move, confirmed) -> dict:
    label = int(move >= V_REBOUND_ATR_MULT * atr and confirmed)
    return {
        "candidate_index": index,
        "timestamp": pd.Timestamp(timestamps[index]).isoformat(),
        "side": side,
        "label": label,
        "sweep_level": float(level),
        "atr": float(atr),
        "rebound_move": move,
        "rebound_atr_multiple": move / float(atr),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    impl = load_impl()
    frame = impl.add_causal_columns(impl.load_5m(SOURCE))
    labels = label_events(frame, impl)

    label_path = OUT_DIR / "eth_5m_sweep_v_rebound_labels.csv"
    labels.to_csv(label_path, index=False)

    by_side = {
        side: {
            "n": int((labels["side"] == side).sum()),
            "v_rebound_rate": (
                float(labels.loc[labels["side"] == side, "label"].mean())
                if (labels["side"] == side).any() else None
            ),
        }
        for side in ("downside", "upside")
    }
    report = {
        "label_contract": {"NO_V_REBOUND": 0, "V_REBOUND": 1},
        "sweep_definition": (
            "dashboard liquidity_sweep (48-bar causal swing wick + close reclaim), "
            "reused unmodified from build_eth_5m_sweep_followthrough_v2_labels_20260829.py::add_causal_columns"
        ),
        "v_rebound_definition": (
            f"within the first {V_REBOUND_FAST_BARS * BAR_MINUTES} of the {LOOKAHEAD_BARS * BAR_MINUTES} "
            f"minutes after the sweep bar, price moves >= {V_REBOUND_ATR_MULT}x pre-sweep ATR(14) "
            f"(14 bars strictly before the sweep bar, excludes the sweep bar's own range) away "
            f"from the sweep extreme in the reversal direction, AND ALL {LOOKAHEAD_BARS} bars in "
            "the full window close beyond the swept level in that direction "
            "(v4: see docstring for the v1->v2->v3->v4 history)"
        ),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_period": {
            "start": str(frame["timestamp"].min()),
            "end": str(frame["timestamp"].max()),
            "closed_5m_bars": int(len(frame)),
        },
        "total_events": int(len(labels)),
        "label_counts": {
            "NO_V_REBOUND": int((labels["label"] == 0).sum()),
            "V_REBOUND": int((labels["label"] == 1).sum()),
        },
        "label_rate": float(labels["label"].mean()) if len(labels) else None,
        "by_side": by_side,
        "parameters": {
            "bar_minutes": BAR_MINUTES,
            "sweep_lookback_bars": int(impl.SWEEP_LOOKBACK_BARS),
            "lookahead_bars": LOOKAHEAD_BARS,
            "lookahead_minutes": LOOKAHEAD_BARS * BAR_MINUTES,
            "atr_n": int(impl.ATR_N),
            "atr_reference": "pre-sweep (frame['atr'].iloc[index-1], v4)",
            "v_rebound_atr_multiple": V_REBOUND_ATR_MULT,
            "v_rebound_fast_bars": V_REBOUND_FAST_BARS,
            "v_rebound_fast_minutes": V_REBOUND_FAST_BARS * BAR_MINUTES,
        },
        "excluded_or_ambiguous_events": 0,
        "future_features_used_for_labels": True,
        "input_features": ["futures_5m_ohlcv"],
        "output_labels": str(label_path),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
