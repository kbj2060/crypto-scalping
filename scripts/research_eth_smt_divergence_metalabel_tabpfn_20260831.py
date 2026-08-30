#!/usr/bin/env python3
"""Meta-labeling for smt_divergence -- Project Homer signal #7. Follows the reusable methodology
template (docs/homer/README.md) for the Tier0 23-feature builder / TabPFN panel / permutation-
importance helpers / Fresh-Forward split, imported verbatim from the taker script.

Signal definition (live_evidence_signal_dashboard_20260823.py::compute_signals):
    bottom = ETH low < ETH's own 48-bar prior swing low  AND  BTC low > BTC's own 48-bar prior
             swing low   (ETH breaks down, BTC does NOT confirm -- cross-asset non-confirmation)
    top    = mirror (ETH breaks up, BTC does NOT confirm)
Structurally a sibling of liquidity_sweep (same swing_low_prior/swing_high_prior, SWEEP_LOOKBACK=48
bars) -- confirmed via phase1 (research_eth_smt_divergence_phase1_diagnostic_20260831.py, scratchpad):
60.6%/56.2% of smt fires exact-same-bar-overlap liquidity_sweep (~40% genuinely different, matching
the pre-existing "형제 신호, 약 40% 다름" finding from eth_evidence_signal_8_recheck_predl_20260829).

Phase1 findings:
- Raw fires (>=2024-01-01): bottom=5603, top=7354 (comparable order of magnitude to liquidity_sweep).
- Fire-bar == true local extreme (48bar fwd window): only 12.6%/13.1% -- fires DURING a move, not at
  exhaustion (median lag 15-16 bars/75-80min, p90 45 bars/225min) -- same touch-based-MFE rationale
  as every other signal in this project (point-in-time labeling would be wrong).
- Naive lift (fire hit-rate / random-bar hit-rate at >=1.0xATR) DECAYS monotonically as horizon
  widens: ~1.11-1.15x @15m -> ~1.05-1.09x @1h -> ~1.0-1.04x @4h -- same "8h raw accuracy is an
  illusion, lift decays with horizon" pattern documented for all 8 original signals. Informs a
  HORIZON grid skewed toward the shorter end (not extending past 4h/48bar).
- ATR self-inclusion check: fire-bar atr_pct sits at the 53-56th percentile of all bars (~50% =
  no gate effect) -- confirmed clean, this signal's condition doesn't gate on any volatility
  percentile (dalton-style contamination does not apply here).

Anchor criterion: swing-break PENETRATION DEPTH (swing_low_prior - low for bottom, high -
swing_high_prior for top) -- same definition-intrinsic, non-circular anchor liquidity_sweep uses
for the identical swing-break family (cluster_dedup_by_penetration, reused verbatim).

K calibration: TRAIN-only (< VAL_START) from the start this time -- proactive fix per the
cross-signal K-calibration audit (eth_evidence_signal_cross_signal_k_calibration_audit_20260831),
which found taker's K went stale after a clustering-parameter change carried an old calibration
forward uncorrected. Targets the pooled ~50/50 hit-rate balance point (majority convention in this
project) unless HORIZON/GAP screening surfaces a reason to treat K as a VAL/OOS-AUC-tuned
hyperparameter instead (liquidity_sweep's own precedent) -- decided empirically, not assumed.

Per the volume_wick_climax methodology lesson (docs/homer/README.md 5.5), HORIZON is screened with
a dense grid from the start, and selection uses max(min(VAL,OOS)) (not raw VAL-max).

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

OUT_DIR = ROOT / "data/labels/eth_5m_smt_divergence_metalabel_20260831"
REPORT_DIR = ROOT / "tmp/eth_smt_divergence_metalabel_tabpfn_20260831"
BTC_KLINES_PATH = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"

START = pd.Timestamp("2024-01-01")
SWEEP_LOOKBACK = 48
K_GRID = np.round(np.arange(0.30, 8.01, 0.05), 2)
# 2026-08-31 bugfix: original ceiling was 3.51 -- H=48/GAP=12 needed K=3.50 to hit ~50/50 (landed
# clean at hit_rate=0.500, so the original 9-point HORIZON grid (H=6-48) is unaffected), but the
# HORIZON grid EXTENSION (H=60-96, research_eth_smt_divergence_metalabel_horizon_extend_20260831.py)
# silently hit this ceiling -- all 4 extended horizons returned the same capped K=3.50 with
# hit_rate_train 55-64% (not ~50%), since calibrate_k_train_only() just picks whichever grid point
# minimizes |hit_rate-0.5| and 3.50 was the closest AVAILABLE point, not the true balance point.
# Widened with headroom; the extension script must be re-run against this fix before trusting its
# AUC numbers.

HORIZON_GRID = [6, 8, 12, 16, 20, 24, 30, 36, 48]
GAP_GRID = [3, 6, 12]

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

SEEDS = [20260829, 141592, 271828, 577215]
SCREEN_SEED = SEEDS[0]


def log(msg: str) -> None:
    print(f"[smt_divergence_metalabel_tabpfn] {msg}", flush=True)


def load_btc_klines() -> pd.DataFrame:
    return pd.read_csv(BTC_KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def cluster_dedup_by_penetration(idx: np.ndarray, penetration: np.ndarray, gap: int) -> np.ndarray:
    """Verbatim logic pattern from liquidity_sweep's own cluster_dedup_by_penetration -- anchor =
    deepest swing-break penetration within each same-side consecutive-fire cluster."""
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
    """Fires with move_atr_mult computed, no hit decision applied yet."""
    high = sig["high"].to_numpy(); low = sig["low"].to_numpy(); close = sig["close"].to_numpy()
    atr_pct = indicator_frame["atr_pct"].to_numpy()
    ts = sig["timestamp"].to_numpy()
    n = len(sig)
    swing_low_prior = pd.Series(low).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min().shift(1).to_numpy()
    swing_high_prior = pd.Series(high).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max().shift(1).to_numpy()
    rows = []
    for side, col in [("bottom", "bottom_smt_divergence"), ("top", "top_smt_divergence")]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[(idx < n - horizon) & (ts[idx] >= np.datetime64(START))]
        penetration = (swing_low_prior[idx] - low[idx]) if side == "bottom" else (high[idx] - swing_high_prior[idx])
        idx = cluster_dedup_by_penetration(idx, penetration, gap)
        entry = close[idx]
        a = atr_pct[idx]
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + horizon + 1].max() for i in idx])
            pred_dir_ret = (fut_ext - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + horizon + 1].min() for i in idx])
            pred_dir_ret = (entry - fut_ext) / entry
        feat_rows = indicator_frame.iloc[idx]
        out = pd.DataFrame({
            "pos": idx, "timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
            "entry": entry, "atr_pct": a, "pred_dir_ret": pred_dir_ret,
            "move_atr_mult": pred_dir_ret / a,
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
    """TRAIN-only (< VAL_START) 50/50 balance calibration -- proactive fix, see module docstring."""
    train = fires_raw.loc[fires_raw["timestamp"] < VAL_START]
    mam = train["move_atr_mult"].to_numpy()
    best_k, best_diff = None, np.inf
    for K in K_GRID:
        diff = abs(float((mam >= K).mean()) - 0.5)
        if diff < best_diff:
            best_diff, best_k = diff, float(K)
    return best_k


def apply_k(fires_raw: pd.DataFrame, K: float) -> pd.DataFrame:
    fires = fires_raw.copy()
    fires["hit"] = (fires["move_atr_mult"] >= K).astype(float)
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
    n_before_dropna = len(fires_raw)
    fires_raw = fires_raw.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    k = calibrate_k_train_only(fires_raw)
    fires = apply_k(fires_raw, k)

    train, val, oos = split_train_val_oos(fires)
    clf = TabPFNClassifier(device="cuda", random_state=SCREEN_SEED)
    clf.fit(train[FEATURE_COLUMNS], train["hit"].to_numpy().astype(int))
    val_auc = roc_auc_score(val["hit"].to_numpy().astype(int), clf.predict_proba(val[FEATURE_COLUMNS])[:, 1])
    oos_auc = roc_auc_score(oos["hit"].to_numpy().astype(int), clf.predict_proba(oos[FEATURE_COLUMNS])[:, 1])

    row = {
        "horizon": horizon, "gap": gap, "k": k,
        "n_fires_after_dropna": len(fires_raw),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_oos": int(len(oos)),
        "hit_rate_train": round(float(train["hit"].mean()), 4),
        "val_auc": round(float(val_auc), 4), "oos_auc": round(float(oos_auc), 4),
        "gap_val_oos": round(abs(float(val_auc) - float(oos_auc)), 4),
    }
    log(f"[screen] H={horizon:>2d} gap={gap:>2d} K={k:.2f}: n={row['n_fires_after_dropna']} "
        f"train={row['n_train']}/val={row['n_val']}/oos={row['n_oos']} hit_rate(train)={row['hit_rate_train']:.3f} "
        f"VAL_AUC={row['val_auc']:.4f} OOS_AUC={row['oos_auc']:.4f} gap={row['gap_val_oos']:.4f}")
    return row, fires


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading klines + BTC klines + building Tier0 indicator frame + compute_signals...")
    klines = load_klines()
    btc = load_btc_klines()
    log(f"BTC klines: {btc['timestamp'].min()} ~ {btc['timestamp'].max()}, n={len(btc)}")
    indicator_frame = build_indicator_frame(klines)
    sig = compute_signals(klines, btc_df=btc, funding_df=None).reset_index(drop=True)
    assert len(sig) == len(indicator_frame) and (sig["timestamp"].to_numpy() == indicator_frame["timestamp"].to_numpy()).all()

    log(f"=== screening grid: HORIZON in {HORIZON_GRID} x CLUSTER_GAP_MERGE in {GAP_GRID} "
        f"({len(HORIZON_GRID)*len(GAP_GRID)} combos, single seed={SCREEN_SEED}, K calibrated TRAIN-only per combo, "
        f"TRAIN-fit -> VAL+OOS AUC, HOLDOUT untouched) ===")
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
    log(f"=== SELECTED (by max(min(VAL,OOS))): HORIZON={best['horizon']} GAP={best['gap']} K={best['k']:.2f} ===")

    horizon_f, gap_f = best["horizon"], best["gap"]
    fires = fires_cache[(horizon_f, gap_f)]
    log(f"candidate fire counts: total={len(fires)} (bottom={int((fires['side']=='bottom').sum())}, top={int((fires['side']=='top').sum())})")

    # ================= STOP HERE -- do NOT touch HOLDOUT yet =================
    # Per user request (2026-08-31): re-run the exclude-middle ambiguous-middle-concentration
    # check (the diagnostic that found orthogonal_combo's problem) at the ACTUAL WINNING combo,
    # not just the placeholder used for the visual-verification chart. That decision (plain 50/50
    # K vs exclude-middle) must be locked in BEFORE the label design is considered final -- if
    # decided AFTER a HOLDOUT touch, redesigning and re-touching HOLDOUT would violate this
    # project's single-touch discipline (exactly the failure mode this staged split avoids).
    mam = fires["move_atr_mult"].to_numpy()
    k = best["k"]
    no_hit = mam[mam < k]
    clear_miss_frac = float((no_hit < 0.3 * k).mean()) if len(no_hit) else float("nan")
    log(f"\n=== ambiguous-middle check at WINNING combo (H={horizon_f}/GAP={gap_f}/K={k:.2f}) ===")
    log(f"  NO_HIT clear-miss fraction [0,0.3K): {clear_miss_frac*100:.1f}% "
        f"(healthy range from other 4 signals: 18-32%; orthogonal_combo outlier was 9.2%)")

    fires.to_csv(OUT_DIR / "eth_5m_smt_divergence_metalabel_CANDIDATE_features.csv", index=False)
    report = {
        "signal": "smt_divergence", "stage": "screening_only_holdout_untouched",
        "screening_grid": screening_rows,
        "selected_by": "max(min(VAL,OOS))", "selection_alt_by_val_max": by_val_max,
        "selected_horizon": horizon_f, "selected_gap": gap_f, "selected_k": k,
        "ambiguous_middle_clear_miss_frac": clear_miss_frac,
        "feature_columns": FEATURE_COLUMNS, "n_fires_candidate": int(len(fires)),
    }
    out_path = REPORT_DIR / "screening_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"screening report saved -> {out_path}")
    log("NEXT STEP (separate script, after label-design decision): "
        "research_eth_smt_divergence_metalabel_final_20260831.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
