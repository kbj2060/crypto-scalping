#!/usr/bin/env python3
"""Bug-hunt mirror of today's confirmed BTC same-bar look-ahead bug
(scripts/diagnose_btc_cryptomamba_tiebreak_dumbmomentum_control_20260801.py), applied to the ETH
regime_tiebreak result in scripts/eval_sigma6_omega_rule_and_meta_allocation_20260801.py, whose
data comes from build_bar_frame() in scripts/train_eval_sigma6_omega_rl_meta_controller_20260801.py:

    curve = curve.set_index("timestamp").resample("1h").last().ffill()   # LEFT-labeled: bin t
                                                                          # holds data through t+55m
    curve["omega_delta"] = curve["omega_equity"].diff().fillna(0.0)
    ...
    raw = load_tape_with_regime()[[...regime3_current_sensitive_wide24_* cols...]]
    curve = pd.merge_asof(curve, raw, on="timestamp", direction="backward")

Traced timestamp semantics (see report printed at the end of this script for the written verdict):
  - equity_curve_{label}.csv (input to build_bar_frame) is NATIVE 5-MINUTE data (confirmed:
    diff() is a constant 5min). resample("1h").last() therefore performs a REAL granularity change:
    the value at label t is whatever the equity curve last held at bar t+55m (pandas default
    left-labeled, closed-left binning -- same empirically-verified convention as the BTC bug).
    So curve["omega_delta"] at label t = PnL realized over the window (t-1h+5m .. t+55m], i.e. a
    window that ENDS at t+55m, mostly in the future relative to label t.
  - The regime3 raw features (training_features_2025_regime3_current_sensitive_hmm_wide24.csv) are
    ALSO native 5-minute data, but scripts/experiment_regime3_current_hmm_wide24_20260529.py builds
    them using only backward-looking transforms (rolling().mean/std, .shift(N) with N>0, .ewm) --
    no resample, no negative shift, no forward-looking window found anywhere in that script. So a
    regime3 row timestamped rt represents information available AT/BEFORE rt (~rt+5m at the latest,
    the close of that one native 5m candle) -- NOT extended through a much later window the way
    BTC's close_1h was.
  - merge_asof(curve, raw, on="timestamp", direction="backward") picks the newest raw row with
    rt <= t. Given raw's ~5-minute cadence, that is almost always a regime row timestamped within
    5 minutes of t, i.e. regime info genuinely available AT/BEFORE t -- which PREDATES the PnL
    window (t, t+55m] that the equity delta at label t actually measures.

  This is the OPPOSITE causal relationship from the BTC bug: BTC's regime and equity delta were
  both built from the exact same closing price at the end of the same window (perfectly
  correlated, decided using the very outcome being gated). Here the regime signal is an
  independently-computed model built from strictly-earlier bars, aligned so it predates (not
  postdates or coincides with) the window whose PnL it gates.

  Still: this reasoning must be tested empirically, not just asserted -- that is what this script
  does, exactly mirroring the BTC methodology: rebuild the regime merge with an EXTRA +1h forward
  shift applied to the raw regime timestamp (making the alignment MORE conservative / staler by a
  full hour, guaranteeing the regime info used at label t is now >= 1h older than in the original),
  and compare the regime_tiebreak rule's VAL/OOS pnl/mdd before vs after.
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

from run_sigma6_regime_trend_20260705 import load_tape_with_regime  # noqa: E402
from train_eval_sigma6_omega_rl_meta_controller_20260801 import (  # noqa: E402
    CURVE_DIR, PFX, run_baseline,
)
from eval_sigma6_omega_rule_and_meta_allocation_20260801 import (  # noqa: E402
    rule_weights, weighted_pnl, RULES,
)

OUT_DIR = ROOT / "tmp/research_20260802/eth_lookahead_bug_check"


def _leg_state(trades: pd.DataFrame, ts: pd.Series) -> pd.DataFrame:
    side = np.zeros(len(ts), dtype=np.float64)
    hold_frac = np.zeros(len(ts), dtype=np.float64)
    max_hold_bars = 144 / 12.0
    for tr in trades.itertuples():
        mask = (ts >= tr.entry_timestamp) & (ts <= tr.exit_timestamp)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        side[idx] = tr.side
        span = tr.exit_timestamp - tr.entry_timestamp
        total_bars = max(span.total_seconds() / 3600.0, 1.0)
        hold_frac[idx] = np.clip((ts[idx].values - np.datetime64(tr.entry_timestamp)) /
                                  np.timedelta64(1, "h") / max(total_bars, max_hold_bars), 0, 1)
    return pd.DataFrame({"side": side, "hold_frac": hold_frac})


def build_bar_frame_variant(label: str, *, regime_shift_hours: float) -> pd.DataFrame:
    """Exact reimplementation of build_bar_frame(), with the raw regime timestamp optionally
    shifted forward by `regime_shift_hours` before the merge_asof (0.0 = reproduce the original
    leaky/current behavior bit-for-bit; >0 = causal-control, makes the regime info used strictly
    staler by that many hours)."""
    curve = pd.read_csv(CURVE_DIR / f"equity_curve_{label}.csv", parse_dates=["timestamp"])
    curve = curve.set_index("timestamp").resample("1h").last().ffill()
    curve["omega_delta"] = curve["omega_equity"].diff().fillna(0.0)
    curve["sigma6_delta"] = curve["sigma6_equity"].diff().fillna(0.0)
    curve = curve.iloc[1:].reset_index()

    om = pd.read_csv(CURVE_DIR / f"omega_trades_{label}.csv", parse_dates=["entry_timestamp", "exit_timestamp"])
    s6 = pd.read_csv(CURVE_DIR / f"sigma6_trades_{label}.csv", parse_dates=["entry_timestamp", "exit_timestamp"])

    ts = curve["timestamp"]
    om_state = _leg_state(om, ts)
    s6_state = _leg_state(s6, ts)
    curve["omega_side"] = om_state["side"]
    curve["omega_active"] = (om_state["side"] != 0).astype(np.float64)
    curve["omega_hold_frac"] = om_state["hold_frac"]
    curve["sigma6_side"] = s6_state["side"]
    curve["sigma6_active"] = (s6_state["side"] != 0).astype(np.float64)
    curve["sigma6_hold_frac"] = s6_state["hold_frac"]
    curve["conflict"] = ((curve["omega_active"] > 0) & (curve["sigma6_active"] > 0) &
                          (curve["omega_side"] != curve["sigma6_side"])).astype(np.float64)

    raw = load_tape_with_regime()[["timestamp", f"{PFX}bull_prob", f"{PFX}bear_prob", f"{PFX}chop_prob",
                                    "regime3_cmamba_h6_sidecar_stability_score"]].copy()
    raw = raw.sort_values("timestamp")
    raw["timestamp"] = raw["timestamp"].astype("datetime64[ns]") + pd.Timedelta(hours=regime_shift_hours)
    curve = curve.sort_values("timestamp")
    curve["timestamp"] = curve["timestamp"].astype("datetime64[ns]")
    curve = pd.merge_asof(curve, raw, on="timestamp", direction="backward")
    curve[["bull_prob", "bear_prob", "chop_prob", "stability"]] = curve[[
        f"{PFX}bull_prob", f"{PFX}bear_prob", f"{PFX}chop_prob", "regime3_cmamba_h6_sidecar_stability_score"]]
    curve["stability"] = curve["stability"].fillna(1.0)
    return curve


def eval_regime_tiebreak(frame_val: pd.DataFrame, frame_oos: pd.DataFrame) -> dict:
    w_om_v, w_s6_v = rule_weights(frame_val, "regime_tiebreak")
    w_om_o, w_s6_o = rule_weights(frame_oos, "regime_tiebreak")
    val_res = weighted_pnl(frame_val, w_om_v, w_s6_v)
    oos_res = weighted_pnl(frame_oos, w_om_o, w_s6_o)
    return {"val_pnl": val_res["pnl_pct"], "val_mdd": val_res["mdd_pct"],
            "oos_pnl": oos_res["pnl_pct"], "oos_mdd": oos_res["mdd_pct"]}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for shift in (0.0, 1.0, 2.0):
        frame_val = build_bar_frame_variant("VAL_2025Q4", regime_shift_hours=shift)
        frame_oos = build_bar_frame_variant("OOS_2026H1", regime_shift_hours=shift)
        base_val = run_baseline(frame_val)
        base_oos = run_baseline(frame_oos)
        res = eval_regime_tiebreak(frame_val, frame_oos)
        rows.append({"regime_shift_hours": shift,
                     "baseline_val_pnl": base_val["pnl_pct"], "baseline_val_mdd": base_val["mdd_pct"],
                     "baseline_oos_pnl": base_oos["pnl_pct"], "baseline_oos_mdd": base_oos["mdd_pct"],
                     **res})
        print(f"\n=== regime_shift_hours={shift} "
              f"({'ORIGINAL/leaky-as-shipped' if shift == 0.0 else 'causal control, staler by ' + str(shift) + 'h'}) ===")
        print(f"baseline fixed-1x1x: VAL pnl={base_val['pnl_pct']:+.2f}% mdd={base_val['mdd_pct']:.2f}% | "
              f"OOS pnl={base_oos['pnl_pct']:+.2f}% mdd={base_oos['mdd_pct']:.2f}%")
        print(f"regime_tiebreak:     VAL pnl={res['val_pnl']:+.2f}% mdd={res['val_mdd']:.2f}% | "
              f"OOS pnl={res['oos_pnl']:+.2f}% mdd={res['oos_mdd']:.2f}%")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "regime_tiebreak_shift_control.csv", index=False)
    print("\n=== full comparison table ===")
    print(df.to_string(index=False))
    print(f"\nsaved -> {OUT_DIR / 'regime_tiebreak_shift_control.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
