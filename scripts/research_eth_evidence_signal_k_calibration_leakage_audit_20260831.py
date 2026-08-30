#!/usr/bin/env python3
"""Following the orthogonal_combo deep audit (research_eth_orthogonal_combo_deep_audit_20260831.py
Check 1), user asked whether the SAME K-threshold global-calibration pattern (hit/miss threshold
picked using a pooled ~50%-hit-rate search over the FULL fires population, which spans through
VAL/OOS/HOLDOUT, not just TRAIN) exists in the other 4 Homer signals that DON'T use exclude-middle
(taker_delta_z_climax, short_term_return_z, volume_wick_climax, dalton_rule2_balance_edge --
liquidity_sweep confirmed clean of exclude-middle too, checked separately by grep).

Code audit already confirmed: none of these 5 signals use exclude-middle (only orthogonal_combo
does) -- so the "evaluated on an easier subset" inflation issue is confirmed unique to
orthogonal_combo. This script checks the OTHER leakage vector: was each signal's single hit/miss
K threshold calibrated using data that includes VAL/OOS/HOLDOUT?

Findings from reading the scripts directly:
- dalton_rule2_balance_edge: HAS a calibrate_k() function, called fresh on the full fires_raw
  population (spans through HOLDOUT) every run -- same pattern as orthogonal_combo.
- volume_wick_climax (v2): K=1.65 is hardcoded, sourced from research_eth_volume_wick_climax_
  anchor_and_horizon_recheck_20260830.py, whose own calibration loop also used the FULL fires
  population (confirmed by reading that script directly, before its later train/val/oos split).
- taker_delta_z_climax (v5) / short_term_return_z: ATR_HIT_MULT is a bare hardcoded constant
  (2.0 / 1.75) with a comment claiming it was "calibrated"/"swept" at some point, but NO committed
  script contains that original sweep loop (likely done in an uncommitted scratchpad script,
  consistent with this project's documented pattern of leaving early exploratory diagnostics
  uncommitted) -- so the ORIGINAL calibration population can't be directly verified. This script
  instead does the best available check: recompute what a properly-causal (TRAIN-only) calibration
  would give TODAY, using each signal's own unchanged fire-building/clustering logic, and compares
  it against the currently-hardcoded K and a full-period recalibration.

No TabPFN/CUDA needed (pure pandas/numpy) -- runs locally.
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

import numpy as np
import pandas as pd

from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame, load_klines

VAL_START = pd.Timestamp("2025-09-01")
K_SEARCH_GRID = np.round(np.arange(0.30, 3.51, 0.05), 2)


def log(msg: str) -> None:
    print(f"[k_calibration_leakage_audit] {msg}", flush=True)


def calibrate_k_50pct(move_atr_mult: np.ndarray) -> float:
    best_k, best_diff = None, np.inf
    for K in K_SEARCH_GRID:
        diff = abs(float((move_atr_mult >= K).mean()) - 0.5)
        if diff < best_diff:
            best_diff, best_k = diff, float(K)
    return best_k


def report_signal(name: str, fires: pd.DataFrame, pred_col: str, atr_col: str, current_k: float) -> None:
    move = fires[pred_col].to_numpy() / fires[atr_col].to_numpy()
    train_mask = fires["timestamp"].to_numpy() < np.datetime64(VAL_START)
    n_train, n_total = int(train_mask.sum()), len(fires)

    k_train_only = calibrate_k_50pct(move[train_mask])
    k_full_period = calibrate_k_50pct(move)

    hit_current = (move >= current_k)
    hit_train_only = (move >= k_train_only)
    flip_frac = float((hit_current != hit_train_only).mean())

    log(f"\n=== {name} (n_train={n_train}/{n_total}) ===")
    log(f"  K currently used (hardcoded/last-calibrated): {current_k}")
    log(f"  K if recalibrated TRAIN-only (causal, right now):  {k_train_only}")
    log(f"  K if recalibrated full-period (incl. VAL/OOS/HOLDOUT): {k_full_period}")
    log(f"  hit-rate @ current K: {hit_current.mean():.4f}   hit-rate @ TRAIN-only K: {hit_train_only.mean():.4f}")
    log(f"  label flip fraction (current K vs TRAIN-only K): {flip_frac:.4f}")
    verdict = "MATCHES (no meaningful leakage effect)" if abs(current_k - k_train_only) < 1e-9 and flip_frac == 0.0 else \
              ("CLOSE (small flip fraction)" if flip_frac < 0.02 else "DIFFERS -- worth re-checking AUC/economics")
    log(f"  verdict: {verdict}")


def main() -> int:
    log("building klines + Tier0 indicator_frame (shared across all 4 signals)...")
    klines = load_klines()
    indicator_frame = build_indicator_frame(klines)

    # ---- taker_delta_z_climax v5 ----
    from research_eth_taker_delta_climax_metalabel_v5_gap12_20260830 import (
        ATR_HIT_MULT as taker_k, build_fires_and_features as taker_build,
    )
    taker_fires = taker_build(klines, indicator_frame)
    report_signal("taker_delta_z_climax (v5, HORIZON=24/GAP=12)", taker_fires, "pred_dir_ret", "atr_pct", taker_k)

    # ---- short_term_return_z ----
    from research_eth_short_term_return_z_metalabel_tabpfn_20260829 import (
        ATR_HIT_MULT as strz_k, build_fires_and_features as strz_build,
    )
    strz_fires = strz_build(klines, indicator_frame)
    report_signal("short_term_return_z (HORIZON=12/GAP=3)", strz_fires, "pred_dir_ret", "atr_pct", strz_k)

    # ---- volume_wick_climax v2 ----
    from research_eth_volume_wick_climax_metalabel_v2_horizon16_20260830 import (
        K as vwc_k, build_fires_and_features as vwc_build,
    )
    vwc_fires = vwc_build(klines, indicator_frame)
    report_signal("volume_wick_climax (v2, HORIZON=16/GAP=3)", vwc_fires, "pred_dir_ret", "atr_pct", vwc_k)

    # ---- dalton_rule2_balance_edge ---- (final adopted HORIZON=30/GAP=12, chosen dynamically inside
    # that script's own main() via grid screening -- not exported as bare module constants)
    from live_evidence_signal_dashboard_20260823 import compute_signals
    from research_eth_dalton_rule2_balance_edge_metalabel_tabpfn_20260830 import (
        build_raw_fires as dalton_build, calibrate_k as dalton_calibrate_k, compute_atr_pct_288,
    )
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    atr_pct_288 = compute_atr_pct_288(klines)
    dalton_fires = dalton_build(indicator_frame, sig, atr_pct_288, 12, 30)
    dalton_k_current, _ = dalton_calibrate_k(dalton_fires)  # what the script computes fresh (full-period) today
    report_signal("dalton_rule2_balance_edge (HORIZON=30/GAP=12)", dalton_fires, "pred_dir_ret", "atr_pct_288", dalton_k_current)

    # ---- liquidity_sweep (final top/down standard redo) ----
    from research_eth_liquidity_sweep_topdown_metalabel_final_20260830 import (
        K as ls_k, build_fires as ls_build,
    )
    ls_fires, _ = ls_build(klines, indicator_frame, sig)  # reuses `sig` computed above for dalton (funding_df=None either way -- liquidity_sweep doesn't use funding_z)
    report_signal("liquidity_sweep (HORIZON=30/GAP=12)", ls_fires, "pred_dir_ret", "atr_pct", ls_k)

    log("\n=== summary ===")
    log("orthogonal_combo's own check found K_center identical (2.5) whether calibrated TRAIN-only or")
    log("full-period -- see per-signal verdicts above for whether the same holds here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
