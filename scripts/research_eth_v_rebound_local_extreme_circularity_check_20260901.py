#!/usr/bin/env python3
"""Clean, from-scratch check of whether local_extreme's CENTERED-window definition
(low[i]==min(low[i-W:i+W+1]), W=6) has genuine mechanical overlap with the V_REBOUND outcome
label's own forward window (fast_move uses close[i+1:i+FAST_BARS+1], FAST_BARS=6 -- the exact same
span as local_extreme's forward half).

This does NOT assume the prior same-day "measurement-method mismatch" retraction (memory
project_v_rebound_local_extreme_lookahead_unresolved_20260901) is correct or incorrect -- it tests
the mechanical claim directly, independent of that reconciliation, independent of any live-27h
sample (n=39, too small to settle a mechanism question).

Key mathematical fact this verifies: local_extreme[i]=True IMPLIES low[i+1:i+W+1].min() >= low[i]
(bar i's low is never undercut in the next W bars) -- by construction, with ZERO exceptions, since
being the min of a window trivially implies being <= every element of any subset of that window.
If this "held_up" property ALONE (regardless of which trigger fired, regardless of local_extreme
specifically) already predicts elevated V자반등 rates among OTHER triggers' candidates, that shows
local_extreme's apparent edge is substantially a tautological consequence of its own definition
overlapping the label's window, not new information.

Four measurements, TRAIN+VAL population, self-checked outcome formula (compute_outcome_fields/
label_side, verbatim from research_btc_v_rebound_gridscreen_20260901.py, already self-check-
validated 1000/0 mismatches in research_eth_v_rebound_multitrigger_feeder_role_screen_20260901.py):

  A) backward_only:  low[i] == low[i-W:i+1].min()   (no forward peek at all -- causally clean)
  B) forward_only:   low[i+1:i+W+1].min() >= low[i]  ("held_up" -- the exact mechanical property
                      that overlaps fast_move's window; note this is a >=, matching what
                      local_extreme's forward half actually requires, not a second independent min)
  C) centered (true, deployed): backward_only AND forward_only (== live local_extreme exactly)
  D) among the OTHER 8 named triggers' candidates ONLY (local_extreme excluded), split by
     forward_only True/False -- the decisive test: does "held_up" alone explain elevated hit rates
     even for candidates local_extreme never touches?

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_eth_v_rebound_local_extreme_circularity_check_20260901.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from research_btc_v_rebound_gridscreen_20260901 import compute_outcome_fields, label_side  # noqa: E402

ETH_CSV = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC_CSV = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
OUT_JSON = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/local_extreme_circularity_check_report.json"

VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
W = 6  # LOCAL_EXTREME_W, == FAST_BARS -- the exact overlap window
NAMED8 = ["liquidity_sweep", "taker_delta_z_climax", "short_term_return_z", "orthogonal_combo",
          "smt_divergence", "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]


def log(msg: str) -> None:
    print(f"[local_extreme_circularity] {msg}", flush=True)


def load_klines(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_circularity_20260901", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rate(status: np.ndarray, mask: np.ndarray) -> dict:
    pool = status[mask]
    denom = int((pool != "invalid").sum())
    n_v = int((pool == "v_rebound").sum())
    return {"n": int(mask.sum()), "n_labeled": denom, "n_v": n_v,
            "rate": round(n_v / denom, 4) if denom else None}


def main() -> int:
    t0 = time.time()
    eth = load_klines(ETH_CSV)
    btc = load_klines(BTC_CSV)
    impl = load_impl()
    causal = impl.add_causal_columns(eth[["timestamp", "open", "high", "low", "close"]].copy())
    sig = compute_signals(eth, btc_df=btc, funding_df=None)
    sig["atr"] = causal["atr"].to_numpy()
    sig = sig.loc[sig["timestamp"] < VAL_END].reset_index(drop=True)
    log(f"TRAIN+VAL population: {len(sig)} rows, {sig['timestamp'].iloc[0]} .. {sig['timestamp'].iloc[-1]}")

    fields = compute_outcome_fields(sig)
    n = len(sig)
    low, high = sig["low"].to_numpy(), sig["high"].to_numpy()

    def status_for(is_down: bool) -> np.ndarray:
        lab = label_side(fields, is_down=is_down)
        valid, label_raw = lab["valid"].to_numpy(), lab["label_raw"].to_numpy()
        return np.where(~valid, "invalid", np.where(label_raw == 1, "v_rebound",
                        np.where(label_raw == 0, "chop", "ambiguous")))

    status_down = status_for(True)
    status_up = status_for(False)

    # --- build the 3 local_extreme variants, both sides, vectorized (matches live's W=6 exactly) ---
    def window_min(arr: np.ndarray, lo_off: int, hi_off: int) -> np.ndarray:
        """rolling min over [i+lo_off, i+hi_off] inclusive, NaN where the window falls outside [0,n)."""
        out = np.full(n, np.nan)
        for i in range(n):
            a, b = i + lo_off, i + hi_off
            if a < 0 or b >= n:
                continue
            out[i] = arr[a:b + 1].min()
        return out

    def window_max(arr: np.ndarray, lo_off: int, hi_off: int) -> np.ndarray:
        out = np.full(n, np.nan)
        for i in range(n):
            a, b = i + lo_off, i + hi_off
            if a < 0 or b >= n:
                continue
            out[i] = arr[a:b + 1].max()
        return out

    log("computing backward/forward/centered local_extreme variants (bottom + top)...")
    bwd_low_min = window_min(low, -W, 0)          # low[i-W..i].min()
    fwd_low_min = window_min(low, 1, W)           # low[i+1..i+W].min()
    bwd_high_max = window_max(high, -W, 0)
    fwd_high_max = window_max(high, 1, W)

    backward_only_bottom = low == bwd_low_min
    forward_only_bottom = fwd_low_min >= low       # "held_up": never undercut in the next W bars
    centered_bottom = backward_only_bottom & forward_only_bottom
    live_bottom = sig["bottom_local_extreme"].to_numpy() if "bottom_local_extreme" in sig.columns else None

    backward_only_top = high == bwd_high_max
    forward_only_top = fwd_high_max <= high        # "held_down" mirror for the top/short side
    centered_top = backward_only_top & forward_only_top

    # sanity: centered must equal the SAME formula _multitrigger_rows()/label-builder use
    # (for i in range(W, n-W): low[i]==low[i-W:i+W+1].min()) -- verify by direct recomputation
    check_bottom = np.zeros(n, dtype=bool)
    check_top = np.zeros(n, dtype=bool)
    for i in range(W, n - W):
        if low[i] == low[i - W:i + W + 1].min():
            check_bottom[i] = True
        if high[i] == high[i - W:i + W + 1].max():
            check_top[i] = True
    mismatch_bottom = int((centered_bottom.astype(bool) & ~np.isnan(fwd_low_min) & ~np.isnan(bwd_low_min) != check_bottom).sum())
    mismatch_top = int((centered_top.astype(bool) & ~np.isnan(fwd_high_max) & ~np.isnan(bwd_high_max) != check_top).sum())
    log(f"  self-check vs live-identical loop formula: bottom mismatches={mismatch_bottom}, top mismatches={mismatch_top}")

    results = {}
    for side, status, backward_only, forward_only, centered, live in (
        ("bottom", status_down, backward_only_bottom, forward_only_bottom, centered_bottom, live_bottom),
        ("top", status_up, backward_only_top, forward_only_top, centered_top, None),
    ):
        valid_window = ~np.isnan(fwd_low_min if side == "bottom" else fwd_high_max) & \
                       ~np.isnan(bwd_low_min if side == "bottom" else bwd_high_max)
        A = rate(status, backward_only & valid_window)
        B = rate(status, forward_only & valid_window)
        C = rate(status, centered & valid_window)
        results[side] = {
            "A_backward_only_no_forward_peek": A,
            "B_forward_only_held_up_property_ALONE": B,
            "C_centered_true_deployed_definition": C,
        }
        if live is not None:
            D_live = rate(status, live.astype(bool) & valid_window)
            results[side]["D_live_bottom_local_extreme_column_crosscheck"] = D_live
        log(f"  [{side}] A(backward-only)={A['rate']}(n={A['n']})  "
            f"B(forward-only='held_up')={B['rate']}(n={B['n']})  C(centered=true live formula)={C['rate']}(n={C['n']})")

    # --- decisive test: among the OTHER 8 named triggers' candidates (local_extreme EXCLUDED),
    # split by whether they incidentally satisfy forward_only ("held_up") ---
    log("=== DECISIVE TEST: other-8-trigger candidates split by incidental 'held_up' property ===")
    decisive = {}
    for side, status, forward_only, others_down in (
        ("bottom", status_down, forward_only_bottom,
         np.any([sig[f"bottom_{nm}"].fillna(False).to_numpy() for nm in NAMED8], axis=0)),
        ("top", status_up, forward_only_top,
         np.any([sig[f"top_{nm}"].fillna(False).to_numpy() for nm in NAMED8], axis=0)),
    ):
        valid_window = ~np.isnan(fwd_low_min if side == "bottom" else fwd_high_max)
        base = others_down & valid_window
        held_up_true = rate(status, base & forward_only)
        held_up_false = rate(status, base & ~forward_only)
        decisive[side] = {"other8_held_up_true": held_up_true, "other8_held_up_false": held_up_false}
        log(f"  [{side}] other-8-trigger candidates: held_up=True rate={held_up_true['rate']}(n={held_up_true['n']})  "
            f"vs held_up=False rate={held_up_false['rate']}(n={held_up_false['n']})")

    # --- base-rate context: what fraction of ALL bars (any/no trigger) incidentally satisfy held_up ---
    log("=== base-rate context: held_up incidence + rate across ALL bars (any trigger or none) ===")
    all_fields_valid_down = ~np.isnan(fwd_low_min)
    all_fields_valid_up = ~np.isnan(fwd_high_max)
    base_rate = {
        "bottom_held_up_incidence": round(float((forward_only_bottom & all_fields_valid_down).mean()), 4),
        "bottom_all_bars_rate_given_held_up": rate(status_down, forward_only_bottom & all_fields_valid_down),
        "bottom_all_bars_rate_given_not_held_up": rate(status_down, ~forward_only_bottom & all_fields_valid_down),
        "top_held_down_incidence": round(float((forward_only_top & all_fields_valid_up).mean()), 4),
        "top_all_bars_rate_given_held_down": rate(status_up, forward_only_top & all_fields_valid_up),
        "top_all_bars_rate_given_not_held_down": rate(status_up, ~forward_only_top & all_fields_valid_up),
    }
    log(f"  bottom: held_up incidence={base_rate['bottom_held_up_incidence']:.1%} of ALL bars, "
        f"rate|held_up={base_rate['bottom_all_bars_rate_given_held_up']['rate']} "
        f"vs rate|not_held_up={base_rate['bottom_all_bars_rate_given_held_up']['rate'] and base_rate['bottom_all_bars_rate_given_not_held_up']['rate']}")

    report = {
        "mechanical_fact": "local_extreme[i]=True implies low[i+1:i+W+1].min()>=low[i] by construction (W=6=FAST_BARS)",
        "self_check_vs_live_loop_formula": {"bottom_mismatches": mismatch_bottom, "top_mismatches": mismatch_top},
        "variant_rates": results,
        "decisive_test_other8_triggers_split_by_held_up": decisive,
        "base_rate_all_bars": base_rate,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
