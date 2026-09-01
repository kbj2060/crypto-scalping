#!/usr/bin/env python3
"""Meta-labeling for fib_extension_exhaustion -- Project Homer signal #8 (last remaining). Follows
the reusable methodology template (docs/homer/README.md) for the Tier0 23-feature builder / TabPFN
panel / permutation-importance helpers / Fresh-Forward split, imported verbatim from the taker
script.

Signal definition (live_evidence_signal_dashboard_20260823.py::compute_signals, ported from
analyze_eth_fibonacci_harmonic_geometric_evidence_20260824.py::add_leg_direction/add_fib_zones):
    leg_up   = low extreme occurred BEFORE high extreme in the trailing 48-bar window [i-48,i-1]
               (most recent extreme = the high -> up-leg)
    leg_down = mirror (most recent extreme = the low -> down-leg)
    fib_rng  = swing_high_prior - swing_low_prior (same 48-bar swing_low_prior/swing_high_prior
               liquidity_sweep/smt_divergence already use)
    bottom = leg_down & low in [swing_low_prior - 0.618*rng, swing_low_prior - 0.272*rng]
             (down-leg pushed 27.2-61.8% of its own range BELOW its prior swing low --
             "extension exhaustion", betting on a bounce)
    top    = mirror

Phase1 findings (scratchpad research_eth_fib_extension_exhaustion_phase1_diagnostic_20260831.py):
- Full-history (>=2024-01-01) raw fires: bottom=1078, top=1072 -- MUCH more than the old n~183-193
  estimate (that was a 5.5-month VAL+OOS-only pooled window from the original 2026-08-24 lift
  study, not full history). Fresh-Forward per-split counts are workable: TRAIN 678/655,
  VAL 124/136, OOS 103/94, HOLDOUT 173/187.
- Naive lift decays monotonically with horizon, faster than any other signal in this project so
  far: ~1.46-1.57x @15min -> ~1.24-1.32x @30min -> ~1.01-1.02x @6h. Skews the useful HORIZON range
  toward the shorter end, but per docs/homer/README.md 5.5/5.6 the grid is still screened densely
  and boundary wins are still extended rather than assumed.
- Fire-bar IS true local extreme only 23.5-27.4% of the time (median lag +4/+5 bars = +20-25min,
  p90 +21/+22 bars, well inside a +-24bar/2h window) -- touch-based MFE labeling required, same as
  every other signal.
- ATR self-inclusion: fire-bar atr_pct percentile 52.8-59.0th (~50th = no material gating effect).
- Overlap vs liquidity_sweep (same swing_low_prior/swing_high_prior): only 6.0-9.5% -- the LEAST
  redundant of this project's swing-break family (smt_divergence was 56-61%).

**2026-08-31 user catch -- MAE cap added to the hit definition (new for this signal)**: the
original pure-MFE-touch design (identical to every other signal in this project) can label a fire
HIT even when a much larger ADVERSE move coexists elsewhere in the same horizon window (found via
a real example: MFE touches +2.54xATR at bar6, then the SAME 12-bar window sees a -6.58xATR crash
at bar10 -- a real position might not capture the fleeting touch, e.g. slippage/latency in a
violent move, and instead ride into the crash). Redefined hit as a whole-window, order-BLIND joint
condition: MFE>=K AND MAE<K_LOSS_MULT*K, both measured over the full [i+1,i+HORIZON] window
regardless of which happens first. Swept K_LOSS_MULT in {1.0,1.5,2.0,3.0} on the placeholder
H=12/GAP=6 config: 1.0x (fully symmetric) CANNOT reach 50/50 balance (caps at 43.8%); 2.0x reaches
50.3% at K=1.85 while disqualifying 2.8% of former hits (the most extreme whipsaw-into-crash
cases, confirmed via a regenerated 20-example visual-verification chart -- user approved).
K_LOSS_MULT=2.0 is now applied identically to every combo in this grid, not just the placeholder.

Anchor criterion: reuses the EXACT SAME penetration-depth definition liquidity_sweep/smt_divergence
use for the swing-break family (swing_low_prior - low for bottom, high - swing_high_prior for top).

K calibration: TRAIN-only (< VAL_START), proactively from the start (per the cross-signal
K-calibration audit lesson) -- now searches for the K that best balances the JOINT
(MFE>=K & MAE<K_LOSS_MULT*K) hit rate to ~50/50, not plain MFE>=K.

K_GRID ceiling widened to 8.0 from the start (per docs/homer/README.md 5.6 -- smt_divergence hit a
silent capped-K bug when the grid ceiling was too narrow for larger HORIZONs; applying that lesson
proactively here instead of waiting to rediscover it).

Runs on the GPU server (quant_ai env, CUDA required for TabPFN) for the actual TabPFN panel; the
raw-fires-building + K-calibration steps are pure pandas/numpy (no CUDA needed).
"""
from __future__ import annotations

import json
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
from sklearn.metrics import roc_auc_score

from live_evidence_signal_dashboard_20260823 import compute_signals
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (
    FEATURE_COLUMNS, build_indicator_frame, load_klines,
)

OUT_DIR = ROOT / "data/labels/eth_5m_fib_extension_exhaustion_metalabel_20260831"
REPORT_DIR = ROOT / "tmp/eth_fib_extension_exhaustion_metalabel_tabpfn_20260831"

START = pd.Timestamp("2024-01-01")
SWEEP_LOOKBACK = 48
K_GRID = np.round(np.arange(0.30, 8.01, 0.05), 2)
K_LOSS_MULT = 2.0  # see module docstring -- swept & chosen 2026-08-31, user-approved via chart

HORIZON_GRID = [6, 8, 12, 16, 20, 24, 30, 36, 48]
GAP_GRID = [3, 6, 12]

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

SEEDS = [20260829, 141592, 271828, 577215]
SCREEN_SEED = SEEDS[0]


def log(msg: str) -> None:
    print(f"[fib_ext_metalabel_tabpfn] {msg}", flush=True)


def cluster_dedup_by_penetration(idx: np.ndarray, penetration: np.ndarray, gap: int) -> np.ndarray:
    """Verbatim logic pattern from liquidity_sweep/smt_divergence's cluster_dedup_by_penetration --
    anchor = deepest swing-break penetration within each same-side consecutive-fire cluster."""
    order = np.argsort(idx)
    idx_sorted, pen_sorted = idx[order], penetration[order]
    cluster_id = np.zeros(len(idx_sorted), dtype=int)
    cid = 0
    for i in range(1, len(idx_sorted)):
        if idx_sorted[i] - idx_sorted[i - 1] > gap:
            cid += 1
        cluster_id[i] = cid
    df = pd.DataFrame({"idx": idx_sorted, "cluster": cluster_id, "pen": pen_sorted})
    return np.sort(df.loc[df.groupby("cluster")["pen"].idxmax()]["idx"].to_numpy())


def build_raw_fires(indicator_frame: pd.DataFrame, sig: pd.DataFrame, gap: int, horizon: int) -> pd.DataFrame:
    """Fires with BOTH move_atr_mult (MFE) and mae_atr_mult (MAE) computed over the same forward
    window, no hit decision applied yet -- hit is a joint function of both (see calibrate_k/apply_k)."""
    high = sig["high"].to_numpy(); low = sig["low"].to_numpy(); close = sig["close"].to_numpy()
    atr_pct = indicator_frame["atr_pct"].to_numpy()
    ts = sig["timestamp"].to_numpy()
    n = len(sig)
    swing_low_prior = pd.Series(low).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min().shift(1).to_numpy()
    swing_high_prior = pd.Series(high).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max().shift(1).to_numpy()
    rows = []
    for side, col in [("bottom", "bottom_fib_extension_exhaustion"), ("top", "top_fib_extension_exhaustion")]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[(idx < n - horizon) & (idx >= SWEEP_LOOKBACK) & (ts[idx] >= np.datetime64(START))]
        penetration = (swing_low_prior[idx] - low[idx]) if side == "bottom" else (high[idx] - swing_high_prior[idx])
        idx = cluster_dedup_by_penetration(idx, penetration, gap)
        entry = close[idx]
        a = atr_pct[idx]
        fut_hi = np.array([high[i + 1:i + horizon + 1].max() for i in idx])
        fut_lo = np.array([low[i + 1:i + horizon + 1].min() for i in idx])
        if side == "bottom":
            mfe = (fut_hi - entry) / entry / a
            mae = (entry - fut_lo) / entry / a
        else:
            mfe = (entry - fut_lo) / entry / a
            mae = (fut_hi - entry) / entry / a
        feat_rows = indicator_frame.iloc[idx]
        out = pd.DataFrame({
            "pos": idx, "timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
            "entry": entry, "atr_pct": a, "move_atr_mult": mfe, "mae_atr_mult": mae,
            "is_bottom": 1 if side == "bottom" else 0,
        })
        for col_name in FEATURE_COLUMNS:
            if col_name == "is_bottom":
                continue
            out[col_name] = feat_rows[col_name].to_numpy()
        rows.append(out)
    fires = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return fires


def calibrate_k_train_only(fires_raw: pd.DataFrame) -> float:
    """TRAIN-only (< VAL_START) 50/50 balance calibration under the JOINT hit rule
    (MFE>=K & MAE<K_LOSS_MULT*K) -- see module docstring.

    2026-08-31 fix: hit_rate(K) under this joint rule is NOT monotonic in K -- as K rises from 0,
    MFE>=K gets harder (hit_rate down) but MAE<K_LOSS_MULT*K gets easier (hit_rate up); empirically
    this project's data shows a single interior peak (hit_rate ~0.59-0.62 around K~0.9-1.55), so the
    curve crosses hit_rate=0.5 TWICE -- once rising (small K, "tiny move + very tight loss cap") and
    once falling (larger K, "meaningful move + proportionally larger loss tolerance"). A naive
    nearest-to-0.5 scan over the whole grid can land on EITHER crossing depending on per-combo
    noise, and the two are qualitatively different labels. Confirmed this caused a real problem:
    the original (unfixed) full-grid screening run picked the low branch for H=6/GAP=12 (K=0.55),
    which produced an OOS AUC of 0.3885 (worse than random) -- while H=20/GAP=12 (the eventual
    overall winner) happened to land on the high branch (K=2.35) and screened cleanly. Every OTHER
    signal in this project uses a K representing a genuine, non-trivial directional move (taker
    2.4, liquidity_sweep 4.0, smt_divergence 4.20) -- so we now deterministically restrict the
    search to the DECLINING branch (K >= the peak), matching that convention and removing the
    ambiguity, then re-ran the full grid clean."""
    train = fires_raw.loc[fires_raw["timestamp"] < VAL_START]
    mfe = train["move_atr_mult"].to_numpy()
    mae = train["mae_atr_mult"].to_numpy()
    curve = [(float(K), float(((mfe >= K) & (mae < K_LOSS_MULT * K)).mean())) for K in K_GRID]
    peak_k = max(curve, key=lambda t: t[1])[0]
    declining_branch = [c for c in curve if c[0] >= peak_k]
    best_k = min(declining_branch, key=lambda c: abs(c[1] - 0.5))[0]
    return best_k


def apply_k(fires_raw: pd.DataFrame, K: float) -> pd.DataFrame:
    fires = fires_raw.copy()
    fires["hit"] = ((fires["move_atr_mult"] >= K) & (fires["mae_atr_mult"] < K_LOSS_MULT * K)).astype(float)
    return fires


def split_train_val_oos(fires: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    return train, val, oos


def screen_one_combo(indicator_frame: pd.DataFrame, sig: pd.DataFrame, horizon: int, gap: int) -> tuple[dict, pd.DataFrame]:
    from tabpfn import TabPFNClassifier

    fires_raw = build_raw_fires(indicator_frame, sig, gap, horizon)
    fires_raw = fires_raw.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    k = calibrate_k_train_only(fires_raw)
    fires = apply_k(fires_raw, k)

    train, val, oos = split_train_val_oos(fires)
    clf = TabPFNClassifier(device="cuda", random_state=SCREEN_SEED)
    clf.fit(train[FEATURE_COLUMNS], train["hit"].to_numpy().astype(int))
    val_auc = roc_auc_score(val["hit"].to_numpy().astype(int), clf.predict_proba(val[FEATURE_COLUMNS])[:, 1])
    oos_auc = roc_auc_score(oos["hit"].to_numpy().astype(int), clf.predict_proba(oos[FEATURE_COLUMNS])[:, 1])

    row = {
        "horizon": horizon, "gap": gap, "k": k, "k_loss": round(K_LOSS_MULT * k, 3),
        "n_fires_after_dropna": len(fires_raw),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_oos": int(len(oos)),
        "hit_rate_train": round(float(train["hit"].mean()), 4),
        "val_auc": round(float(val_auc), 4), "oos_auc": round(float(oos_auc), 4),
        "gap_val_oos": round(abs(float(val_auc) - float(oos_auc)), 4),
    }
    log(f"[screen] H={horizon:>2d} gap={gap:>2d} K={k:.2f} K_loss={row['k_loss']:.2f}: "
        f"n={row['n_fires_after_dropna']} train={row['n_train']}/val={row['n_val']}/oos={row['n_oos']} "
        f"hit_rate(train)={row['hit_rate_train']:.3f} VAL_AUC={row['val_auc']:.4f} OOS_AUC={row['oos_auc']:.4f} "
        f"gap={row['gap_val_oos']:.4f}")
    return row, fires


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading klines + building Tier0 indicator frame + compute_signals (ETH-only, no BTC needed)...")
    klines = load_klines()
    indicator_frame = build_indicator_frame(klines)
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    assert len(sig) == len(indicator_frame) and (sig["timestamp"].to_numpy() == indicator_frame["timestamp"].to_numpy()).all()

    log(f"=== screening grid: HORIZON in {HORIZON_GRID} x CLUSTER_GAP_MERGE in {GAP_GRID} "
        f"({len(HORIZON_GRID)*len(GAP_GRID)} combos, single seed={SCREEN_SEED}, K calibrated TRAIN-only per "
        f"combo under joint MFE/MAE rule (K_LOSS_MULT={K_LOSS_MULT}), TRAIN-fit -> VAL+OOS AUC, HOLDOUT untouched) ===")
    screening_rows = []
    fires_cache: dict[tuple[int, int], pd.DataFrame] = {}
    for horizon in HORIZON_GRID:
        for gap in GAP_GRID:
            row, fires = screen_one_combo(indicator_frame, sig, horizon, gap)
            screening_rows.append(row)
            fires_cache[(horizon, gap)] = fires

    by_val_max = max(screening_rows, key=lambda r: r["val_auc"])
    by_min_auc = max(screening_rows, key=lambda r: min(r["val_auc"], r["oos_auc"]))
    log(f"if selected by raw VAL max: H={by_val_max['horizon']} GAP={by_val_max['gap']} "
        f"(VAL={by_val_max['val_auc']:.4f} OOS={by_val_max['oos_auc']:.4f} gap={by_val_max['gap_val_oos']:.4f})")
    log(f"if selected by max(min(VAL,OOS)): H={by_min_auc['horizon']} GAP={by_min_auc['gap']} "
        f"(VAL={by_min_auc['val_auc']:.4f} OOS={by_min_auc['oos_auc']:.4f} gap={by_min_auc['gap_val_oos']:.4f})")
    best = by_min_auc
    log(f"=== SELECTED (by max(min(VAL,OOS))): HORIZON={best['horizon']} GAP={best['gap']} "
        f"K={best['k']:.2f} K_loss={best['k_loss']:.2f} ===")

    # boundary check (docs/homer/README.md 5.6) -- warn (not auto-extend) if the winner sits at a
    # grid edge for its own GAP column and min(VAL,OOS) is still trending upward there
    same_gap = sorted([r for r in screening_rows if r["gap"] == best["gap"]], key=lambda r: r["horizon"])
    horizons_this_gap = [r["horizon"] for r in same_gap]
    if best["horizon"] in (min(horizons_this_gap), max(horizons_this_gap)):
        log(f"*** WARNING: selected HORIZON={best['horizon']} sits at a GRID BOUNDARY for GAP={best['gap']} "
            f"-- per 5.6) lesson (smt_divergence), extend the grid before trusting this as a true local peak. ***")

    horizon_f, gap_f = best["horizon"], best["gap"]
    fires = fires_cache[(horizon_f, gap_f)]
    log(f"candidate fire counts: total={len(fires)} (bottom={int((fires['side']=='bottom').sum())}, top={int((fires['side']=='top').sum())})")

    # ================= STOP HERE -- do NOT touch HOLDOUT yet =================
    mfe = fires["move_atr_mult"].to_numpy()
    mae = fires["mae_atr_mult"].to_numpy()
    hit = fires["hit"].to_numpy().astype(bool)
    k, k_loss = best["k"], best["k_loss"]
    # ambiguous-middle check, adapted for the joint rule: restrict to the "plain miss" (MFE<K)
    # subset of NO_HIT (excludes MAE-flipped cases, which are unambiguous by construction -- they
    # have a LARGE adverse move, not a borderline-small MFE) and measure the clear-miss fraction
    # within that subset, same definition as every other signal.
    no_hit_plain_miss = mfe[(~hit) & (mfe < k)]
    clear_miss_frac = float((no_hit_plain_miss < 0.3 * k).mean()) if len(no_hit_plain_miss) else float("nan")
    flipped_frac_of_no_hit = float(((~hit) & (mfe >= k)).sum() / (~hit).sum()) if (~hit).sum() else float("nan")
    log(f"\n=== ambiguous-middle check at WINNING combo (H={horizon_f}/GAP={gap_f}/K={k:.2f}) ===")
    log(f"  NO_HIT 'plain miss' (MFE<K) clear-miss fraction [0,0.3K): {clear_miss_frac*100:.1f}% "
        f"(healthy range from other signals: 18-32%)")
    log(f"  fraction of NO_HIT that are MAE-flipped (MFE>=K but disqualified): {flipped_frac_of_no_hit*100:.1f}%")

    fires.to_csv(OUT_DIR / "eth_5m_fib_extension_exhaustion_metalabel_CANDIDATE_features.csv", index=False)
    report = {
        "signal": "fib_extension_exhaustion", "stage": "screening_only_holdout_untouched",
        "k_loss_mult": K_LOSS_MULT,
        "screening_grid": screening_rows,
        "selected_by": "max(min(VAL,OOS))", "selection_alt_by_val_max": by_val_max,
        "selected_horizon": horizon_f, "selected_gap": gap_f, "selected_k": k, "selected_k_loss": k_loss,
        "ambiguous_middle_clear_miss_frac_plain_miss_only": clear_miss_frac,
        "flipped_frac_of_no_hit": flipped_frac_of_no_hit,
        "feature_columns": FEATURE_COLUMNS, "n_fires_candidate": int(len(fires)),
    }
    out_path = REPORT_DIR / "screening_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"screening report saved -> {out_path}")
    log("NEXT STEP (separate script, after label-design decision + any grid extension): "
        "research_eth_fib_extension_exhaustion_metalabel_final_20260831.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
