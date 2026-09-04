#!/usr/bin/env python3
"""Per-signal numeric-threshold sweep for the 8 live evidence signals
(scripts/live_evidence_signal_dashboard_20260823.py::SIGNAL_ORDER), aimed at raising lift.
Retunes only the existing numeric cutoffs already inside each signal's formula (percentile /
z-score thresholds) -- does NOT change signal *structure* (which conditions AND/OR together).
User explicitly asked to go one signal at a time in SIGNAL_ORDER, starting with orthogonal_combo
(2026-08-29 recheck-before-DL thread, see memory eth_evidence_signal_8_recheck_predl_20260829).

Methodology (this repo's standard anti-overfitting split, reused not reinvented):
  1. Sweep candidate thresholds on VAL ONLY (2025-09-01..2025-12-31).
  2. A candidate only qualifies if n_triggers>=30 AND its Wilson-CI LOWER bound on precision
     clears the current live default's VAL point-estimate precision -- a higher point estimate
     alone is not enough (small-n grid cells are noisy; this project has been burned before by
     point-estimate-only comparisons, e.g. eth_live_promotion_seed_robustness_5seed memories).
  3. The single best qualifying candidate per side is confirmed ONCE on OOS
     (2026-01-01..2026-02-17), untouched during selection. Only recommended if OOS also holds up.
  4. Full VAL grid (lift + n per cell) is written to CSV for charting -- per this project's
     "show chart before parameter decisions" convention (feedback_show_chart_before_parameter_
     decisions), do not decide off this script's printed table alone.

Horizon: 1h (K12_1h) only, this lineage's headline horizon.
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

from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import (  # noqa: E402
    K_HORIZONS,
    OOS_END,
    event_study,
    load_zigzag_pivots,
)
from analyze_eth_creative_reversal_evidence_signals_20260814 import load_frame_with_orderflow  # noqa: E402
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START,
    VAL_END,
    VAL_START,
)
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from research_eth_funding_crossasset_combo_signal_20260825 import load_funding_z  # noqa: E402

BTC_PATH = ROOT / "data" / "btc_5m_1year.csv"
OUT_DIR = ROOT / "tmp" / "eth_evidence_signal_threshold_sweep_20260829"
K = K_HORIZONS["K12_1h"]
Z_95 = 1.959963984540054

DEFAULT_PC, DEFAULT_ZC = 0.10, 2.0
PCS = [0.05, 0.075, 0.10, 0.15, 0.20]
ZCS = [1.5, 1.75, 2.0, 2.25, 2.5, 3.0]


def wilson_ci(hits: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


def evaluate(mask: pd.Series, pivot_pos: np.ndarray, all_pos: np.ndarray, window_mask: np.ndarray) -> dict:
    trigger_pos = np.flatnonzero(mask.fillna(False).to_numpy() & window_mask)
    stats = event_study(trigger_pos, pivot_pos, all_pos, K)
    n, prec = stats["n_triggers"], stats["precision"]
    hits = round(prec * n) if n and np.isfinite(prec) else 0
    lo, hi = wilson_ci(hits, n) if n else (float("nan"), float("nan"))
    return {"n": n, "p": prec, "lo": lo, "hi": hi, "base": stats["baseline_rate"], "lift": stats["lift"]}


def main() -> None:
    raw = load_frame_with_orderflow()
    btc_raw = pd.read_csv(BTC_PATH, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    funding_df = load_funding_z()
    sig = compute_signals(raw, btc_df=btc_raw, funding_df=funding_df)
    pivots = load_zigzag_pivots()
    ts = sig["timestamp"]

    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()
    oos_mask = ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()
    all_pos_val = np.flatnonzero(val_mask)
    all_pos_oos = np.flatnonzero(oos_mask)

    p_fast, p_slow = sig["p_fast"], sig["p_slow"]
    delta_z, funding_z = sig["delta_z"], sig["funding_z"]

    bottom_pivots = pivots.loc[pivots["pivot_type"] == "bottom"]
    top_pivots = pivots.loc[pivots["pivot_type"] == "top"]
    bottom_pivot_pos = sig.index[sig["timestamp"].isin(bottom_pivots["timestamp"])].to_numpy()
    top_pivot_pos = sig.index[sig["timestamp"].isin(top_pivots["timestamp"])].to_numpy()

    rows = []
    for pc in PCS:
        for zc in ZCS:
            bottom_mask = (p_fast <= pc) & (p_slow <= pc) & ((delta_z <= -zc) | (funding_z <= -zc))
            top_mask = (p_fast >= 1 - pc) & (p_slow >= 1 - pc) & (delta_z >= zc)
            val_b = evaluate(bottom_mask, bottom_pivot_pos, all_pos_val, val_mask)
            val_t = evaluate(top_mask, top_pivot_pos, all_pos_val, val_mask)
            rows.append({"pc": pc, "zc": zc, "side": "bottom", **val_b})
            rows.append({"pc": pc, "zc": zc, "side": "top", **val_t})

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "orthogonal_combo_val_sweep.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)

    summary_rows = []
    for side, pivot_pos in (("bottom", bottom_pivot_pos), ("top", top_pivot_pos)):
        sub = df[df["side"] == side].copy()
        default_row = sub[(sub["pc"] == DEFAULT_PC) & (sub["zc"] == DEFAULT_ZC)].iloc[0]
        print(f"\n=== {side}: VAL grid (rows=pc, cols=zc, cell=lift[n]) ===")
        pivot_lift = sub.pivot(index="pc", columns="zc", values="lift").round(2)
        pivot_n = sub.pivot(index="pc", columns="zc", values="n").astype(int)
        for pc_val in PCS:
            line = f"pc={pc_val:.3f}: " + "  ".join(
                f"zc={zc_val}:{pivot_lift.loc[pc_val, zc_val]:.2f}x(n={pivot_n.loc[pc_val, zc_val]})"
                for zc_val in ZCS
            )
            print(line)

        candidates = sub[(sub["n"] >= 30) & (sub["lo"] > default_row["p"])].copy()
        candidates = candidates.sort_values("lift", ascending=False)
        print(f"\n{side} default (pc={DEFAULT_PC}, zc={DEFAULT_ZC}): "
              f"precision={default_row['p']*100:.1f}% n={default_row['n']:.0f} lift={default_row['lift']:.2f}x")
        if candidates.empty:
            print(f"{side}: NO candidate cleared default's VAL precision with CI-lower-bound margin "
                  f"(n>=30 required) -- default already looks threshold-optimal in this grid.")
            continue

        best = candidates.iloc[0]
        print(f"{side} best VAL candidate: pc={best['pc']}, zc={best['zc']} -> "
              f"precision={best['p']*100:.1f}% [{best['lo']*100:.1f}-{best['hi']*100:.1f}%] "
              f"n={best['n']:.0f} lift={best['lift']:.2f}x")

        # OOS confirmation (untouched during selection)
        pc_c, zc_c = best["pc"], best["zc"]
        if side == "bottom":
            cand_mask_oos = (p_fast <= pc_c) & (p_slow <= pc_c) & ((delta_z <= -zc_c) | (funding_z <= -zc_c))
            default_mask_oos = (p_fast <= DEFAULT_PC) & (p_slow <= DEFAULT_PC) & ((delta_z <= -DEFAULT_ZC) | (funding_z <= -DEFAULT_ZC))
        else:
            cand_mask_oos = (p_fast >= 1 - pc_c) & (p_slow >= 1 - pc_c) & (delta_z >= zc_c)
            default_mask_oos = (p_fast >= 1 - DEFAULT_PC) & (p_slow >= 1 - DEFAULT_PC) & (delta_z >= DEFAULT_ZC)

        oos_cand = evaluate(cand_mask_oos, pivot_pos, all_pos_oos, oos_mask)
        oos_default = evaluate(default_mask_oos, pivot_pos, all_pos_oos, oos_mask)
        print(f"{side} OOS confirm -- default: {oos_default['p']*100:.1f}% "
              f"[{oos_default['lo']*100:.1f}-{oos_default['hi']*100:.1f}%] n={oos_default['n']} lift={oos_default['lift']:.2f}x | "
              f"candidate: {oos_cand['p']*100:.1f}% [{oos_cand['lo']*100:.1f}-{oos_cand['hi']*100:.1f}%] "
              f"n={oos_cand['n']} lift={oos_cand['lift']:.2f}x")
        holds = oos_cand["lo"] > oos_default["p"] or oos_cand["lift"] >= oos_default["lift"]
        print(f"{side} verdict: {'OOS SUPPORTS candidate' if holds else 'OOS DOES NOT clearly support candidate (default point estimate not beaten)'}")

        summary_rows.append({
            "side": side, "default_pc": DEFAULT_PC, "default_zc": DEFAULT_ZC,
            "cand_pc": pc_c, "cand_zc": zc_c,
            "val_default_p": default_row["p"], "val_default_n": default_row["n"], "val_default_lift": default_row["lift"],
            "val_cand_p": best["p"], "val_cand_lo": best["lo"], "val_cand_hi": best["hi"], "val_cand_n": best["n"], "val_cand_lift": best["lift"],
            "oos_default_p": oos_default["p"], "oos_default_lo": oos_default["lo"], "oos_default_hi": oos_default["hi"], "oos_default_n": oos_default["n"], "oos_default_lift": oos_default["lift"],
            "oos_cand_p": oos_cand["p"], "oos_cand_lo": oos_cand["lo"], "oos_cand_hi": oos_cand["hi"], "oos_cand_n": oos_cand["n"], "oos_cand_lift": oos_cand["lift"],
            "oos_holds": holds,
        })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(OUT_DIR / "orthogonal_combo_candidate_summary.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'orthogonal_combo_val_sweep.csv'}")

    # --- bottom leg only: decouple delta_z vs funding_z cutoffs (currently both hardcoded to the
    # same 2.0 -- never independently tuned, see live_evidence_signal_dashboard_20260823.py bottom
    # formula). pc held at the live default (0.10) to isolate this one new axis. ---
    print("\n\n=== bottom: decoupled delta_zc x funding_zc sweep (pc fixed at 0.10) ===")
    decouple_rows = []
    for dzc in ZCS:
        for fzc in ZCS:
            mask = (p_fast <= DEFAULT_PC) & (p_slow <= DEFAULT_PC) & ((delta_z <= -dzc) | (funding_z <= -fzc))
            val_stat = evaluate(mask, bottom_pivot_pos, all_pos_val, val_mask)
            decouple_rows.append({"delta_zc": dzc, "funding_zc": fzc, **val_stat})
    ddf = pd.DataFrame(decouple_rows)
    ddf.to_csv(OUT_DIR / "orthogonal_combo_bottom_decoupled_sweep.csv", index=False)
    dpivot_lift = ddf.pivot(index="delta_zc", columns="funding_zc", values="lift").round(2)
    dpivot_n = ddf.pivot(index="delta_zc", columns="funding_zc", values="n").astype(int)
    for dzc in ZCS:
        line = f"delta_zc={dzc}: " + "  ".join(
            f"funding_zc={fzc}:{dpivot_lift.loc[dzc, fzc]:.2f}x(n={dpivot_n.loc[dzc, fzc]})" for fzc in ZCS
        )
        print(line)

    default_decoupled = ddf[(ddf["delta_zc"] == DEFAULT_ZC) & (ddf["funding_zc"] == DEFAULT_ZC)].iloc[0]
    d_candidates = ddf[(ddf["n"] >= 30) & (ddf["lo"] > default_decoupled["p"])].sort_values("lift", ascending=False)
    print(f"\nbottom decoupled default (delta_zc=2.0, funding_zc=2.0): "
          f"precision={default_decoupled['p']*100:.1f}% n={default_decoupled['n']:.0f} lift={default_decoupled['lift']:.2f}x")
    if d_candidates.empty:
        print("bottom decoupled: NO candidate cleared default's VAL precision with CI-lower-bound margin.")
    else:
        best_d = d_candidates.iloc[0]
        print(f"bottom decoupled best VAL candidate: delta_zc={best_d['delta_zc']}, funding_zc={best_d['funding_zc']} -> "
              f"precision={best_d['p']*100:.1f}% [{best_d['lo']*100:.1f}-{best_d['hi']*100:.1f}%] "
              f"n={best_d['n']:.0f} lift={best_d['lift']:.2f}x")
        dzc_c, fzc_c = best_d["delta_zc"], best_d["funding_zc"]
        cand_mask_oos = (p_fast <= DEFAULT_PC) & (p_slow <= DEFAULT_PC) & ((delta_z <= -dzc_c) | (funding_z <= -fzc_c))
        default_mask_oos = (p_fast <= DEFAULT_PC) & (p_slow <= DEFAULT_PC) & ((delta_z <= -DEFAULT_ZC) | (funding_z <= -DEFAULT_ZC))
        oos_cand = evaluate(cand_mask_oos, bottom_pivot_pos, all_pos_oos, oos_mask)
        oos_default = evaluate(default_mask_oos, bottom_pivot_pos, all_pos_oos, oos_mask)
        print(f"bottom decoupled OOS confirm -- default: {oos_default['p']*100:.1f}% "
              f"[{oos_default['lo']*100:.1f}-{oos_default['hi']*100:.1f}%] n={oos_default['n']} lift={oos_default['lift']:.2f}x | "
              f"candidate: {oos_cand['p']*100:.1f}% [{oos_cand['lo']*100:.1f}-{oos_cand['hi']*100:.1f}%] "
              f"n={oos_cand['n']} lift={oos_cand['lift']:.2f}x")


if __name__ == "__main__":
    main()
