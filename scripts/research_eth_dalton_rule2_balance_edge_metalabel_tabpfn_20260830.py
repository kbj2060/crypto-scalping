#!/usr/bin/env python3
"""Meta-labeling for dalton_rule2_balance_edge -- Project Homer signal #5, following the reusable
methodology template for the Tier0 23-feature builder / TabPFN panel / permutation-importance
helpers / Fresh-Forward split (all imported from the taker_delta_z_climax script).

Signal definition (live_evidence_signal_dashboard_20260823.py::compute_signals):
    dalton_low_vol_regime = (14-bar ATR%'s 288-bar rolling percentile-rank) <= 0.30
    dalton_range_low/high = low/high.rolling(48 bars, NOT shifted -- includes current bar).min()/max()
    dalton_tol = 0.15 * (range_high - range_low)
    bottom = low_vol_regime & |low - range_low| <= tol      top = low_vol_regime & |range_high - high| <= tol
i.e. "currently sitting near the edge of its own trailing 4h range, during a low-volatility
regime" -- a REGIME/STATE variable, not a discrete climax/exhaustion event like the other 3 signals.

Phase 1 diagnostics (scratchpad/research_eth_dalton_rule2_balance_edge_phase1_diagnostic_20260830.py,
not committed) confirmed and quantified the pre-existing 2026-08-25 rule-based recheck's warning
(weakest lift of the 8 signals, 1.6-1.74x, and fires extremely often):
  - Fires on 4.1%/6.7% of ALL bars (bottom/top) -- median gap between raw per-bar fires is 1 BAR
    (contiguous). Forms 3,400/4,611 distinct contiguous runs, median run length 2 bars (10min) but
    tail to 32-40 bars (160-200min) -- some genuine multi-hour range-bound episodes.
  - At-anchor-bar precision (using RUN START, the only convention that's actually live-actionable
    for a state variable) is 4.6-4.9% -- the WORST of the 4 signals investigated so far (taker 14%,
    volume_wick_climax 27%). 75.9-77.6% of runs have their true local extreme AFTER the start
    (median lag +8 bars/+40min).
  - Naive sign-only horizon-sensitivity is FLAT everywhere (50.1-54.4% across 15min-4h) -- the
    weakest/flattest of any signal checked, no horizon shows a clear edge.
User was shown these numbers and an explicit 3-way choice (apply template as-is / redesign around
regime-duration+breakout-direction / deprioritize for orthogonal_combo instead) -- chose to apply
the reusable template as-is despite the weak diagnostics.

Label (touch-based MFE, no persistence check, same principle as the other 3 signals): entry = RUN
START bar's own close (first bar of a same-side contiguous run, gap<=GAP bars merges nearby runs
into one) -- NOT the "most extreme" bar within the cluster (unlike taker/short_term_return_z/
volume_wick_climax's magnitude-based anchor), because a state variable has no single natural
"peak intensity" bar the way a discrete climax event does, and run-start is the only point that's
actually actionable in live use (you can't know you're at the deepest point of an ongoing regime
until after it's over). Visually verified (scratchpad/render_eth_dalton_rule2_balance_edge_
metalabel_v1_20260830.py, 20-example candlestick chart, HORIZON=12/K=1.7/gap=3 placeholder) --
user raised no label-logic objections (asked clarifying questions about bottom/top semantics and
tuning status only).

Per the volume_wick_climax methodology lesson (docs/homer/README.md 5.5), HORIZON is screened with
a DENSE grid from the start here (not a sparse 3-point grid), and GAP is swept too -- this signal's
own phase1 horizon-sensitivity was flatter than any prior signal's, so there is even less reason to
trust a narrow a-priori guess.

⚠️FIX (2026-08-30, user-flagged via visual chart review): dalton's OWN trigger requires the
instantaneous 14-bar `atr_pct` to already be in the bottom 30th percentile of its trailing 288-bar
history -- using that SAME self-selected-to-be-tiny atr_pct as the K-multiple's hit-threshold
denominator let noise-level absolute moves register as large ATR-multiples purely because the
yardstick was artificially suppressed (a flagged example: a 0.18% move was scored as "1.86x ATR"
because that fire's 14-bar atr_pct was at the 5th percentile of ALL bars in history, 0.094%). Fixed
by scaling the hit threshold with a 288-bar MEAN true range (`atr_pct_288`) instead -- the same
window dalton_atr_pctile already uses to define "low-vol regime" -- so the threshold reflects "how
large is this move relative to a typical recent day" rather than "relative to the specific instant
already known to be unusually quiet". The standard 14-bar `atr_pct` is kept UNCHANGED as a model
FEATURE (FEATURE_COLUMNS) -- only the label's own normalization changed.

Runs on the GPU server (quant_ai env, CUDA required for TabPFN).
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
    FEATURE_COLUMNS, build_indicator_frame, compute_permutation_importance, load_klines, run_tabpfn_panel,
)

OUT_DIR = ROOT / "data/labels/eth_5m_dalton_rule2_balance_edge_metalabel_20260830"
REPORT_DIR = ROOT / "tmp/eth_dalton_rule2_balance_edge_metalabel_tabpfn_20260830"

START = pd.Timestamp("2024-01-01")
DALTON_TOL_FRAC = 0.15  # matches compute_signals()'s own tolerance, only used for the baseline check

HORIZON_GRID = [6, 8, 12, 16, 20, 24, 30, 36, 48]  # dense from the start -- phase1's own sensitivity was flat/uninformative
GAP_GRID = [3, 6, 12]
K_GRID = np.round(np.arange(0.30, 3.01, 0.05), 2)

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

SEEDS = [20260829, 141592, 271828, 577215]
SCREEN_SEED = SEEDS[0]


def log(msg: str) -> None:
    print(f"[dalton_metalabel_tabpfn] {msg}", flush=True)


def cluster_run_start(idx: np.ndarray, gap: int) -> np.ndarray:
    """Same gap<=N-bars clustering mechanism as the other 3 signals, but anchor = FIRST bar in the
    cluster (run start), not the highest-scoring bar -- see module docstring."""
    idx = np.sort(idx)
    cluster_id = np.zeros(len(idx), dtype=int)
    cid = 0
    for i in range(1, len(idx)):
        if idx[i] - idx[i - 1] > gap:
            cid += 1
        cluster_id[i] = cid
    df = pd.DataFrame({"idx": idx, "cluster": cluster_id})
    return np.sort(df.groupby("cluster")["idx"].min().to_numpy())


def compute_atr_pct_288(klines: pd.DataFrame) -> np.ndarray:
    """288-bar (24h) mean true range, as a %-of-close -- the label-threshold denominator (see
    module docstring fix note). Distinct from indicator_frame['atr_pct'] (14-bar), which stays
    unchanged as a model feature."""
    high, low, close = klines["high"], klines["low"], klines["close"]
    prev_close = close.shift(1); prev_close.iloc[0] = close.iloc[0]
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return (tr.rolling(288, min_periods=288).mean() / close).to_numpy()


def build_raw_fires(indicator_frame: pd.DataFrame, sig: pd.DataFrame, atr_pct_288: np.ndarray, gap: int, horizon: int) -> pd.DataFrame:
    high = sig["high"].to_numpy(); low = sig["low"].to_numpy(); close = sig["close"].to_numpy()
    ts = sig["timestamp"].to_numpy()
    n = len(sig)
    rows = []
    for side, col in [("bottom", "bottom_dalton_rule2_balance_edge"), ("top", "top_dalton_rule2_balance_edge")]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[(idx < n - horizon) & (ts[idx] >= np.datetime64(START))]
        idx = cluster_run_start(idx, gap)
        entry = close[idx]
        a288 = atr_pct_288[idx]
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + horizon + 1].max() for i in idx])
            pred_dir_ret = (fut_ext - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + horizon + 1].min() for i in idx])
            pred_dir_ret = (entry - fut_ext) / entry
        feat_rows = indicator_frame.iloc[idx]
        out = pd.DataFrame({
            "pos": idx, "timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
            "entry": entry, "atr_pct_288": a288, "pred_dir_ret": pred_dir_ret,
            "is_bottom": 1 if side == "bottom" else 0,
        })
        for col_name in FEATURE_COLUMNS:
            if col_name == "is_bottom":
                continue
            out[col_name] = feat_rows[col_name].to_numpy()  # atr_pct here stays the 14-bar feature
        rows.append(out)
    fires = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return fires


def calibrate_k(fires_raw: pd.DataFrame) -> tuple[float, list[dict]]:
    pred = fires_raw["pred_dir_ret"].to_numpy()
    a = fires_raw["atr_pct_288"].to_numpy()
    table = []
    best_k, best_diff = None, np.inf
    for K in K_GRID:
        hit_rate = float((pred >= K * a).mean())
        table.append({"K": float(K), "hit_rate": round(hit_rate, 4)})
        diff = abs(hit_rate - 0.5)
        if diff < best_diff:
            best_diff, best_k = diff, float(K)
    return best_k, table


def apply_k(fires_raw: pd.DataFrame, K: float) -> pd.DataFrame:
    fires = fires_raw.copy()
    fires["hit"] = (fires["pred_dir_ret"] >= K * fires["atr_pct_288"]).astype(float)
    return fires


def random_bar_baseline_no_lowvol_gate(indicator_frame: pd.DataFrame, klines: pd.DataFrame, atr_pct_288: np.ndarray, horizon: int, K: float) -> dict:
    """Isolates the value of the LOW-VOL-REGIME gate specifically: keep the range-edge-proximity
    condition (it defines direction) but drop the low-vol-regime requirement, applying the SAME
    MFE/K/horizon hit rule (K*atr_pct_288, matching the fixed label) to every bar meeting the
    edge-proximity condition alone (not deduped, matching the other signals' baseline convention)."""
    high, low, close = klines["high"], klines["low"], klines["close"]
    n = len(klines)
    fwd_high_max = high[::-1].rolling(window=horizon, min_periods=horizon).max()[::-1].shift(-1)
    fwd_low_min = low[::-1].rolling(window=horizon, min_periods=horizon).min()[::-1].shift(-1)
    mfe_up_pct = ((fwd_high_max - close) / close).to_numpy()
    mfe_down_pct = ((close - fwd_low_min) / close).to_numpy()

    range_low = low.rolling(48, min_periods=48).min()
    range_high = high.rolling(48, min_periods=48).max()
    tol = DALTON_TOL_FRAC * (range_high - range_low)
    near_low = ((low - range_low).abs() <= tol).to_numpy()
    near_high = ((range_high - high).abs() <= tol).to_numpy()

    ts = indicator_frame["timestamp"].to_numpy()
    valid_base = (ts >= np.datetime64(START)) & (np.arange(n) < n - horizon) & np.isfinite(atr_pct_288)

    bottom_idx = np.flatnonzero(valid_base & near_low)
    top_idx = np.flatnonzero(valid_base & near_high)
    bottom_hit = mfe_up_pct[bottom_idx] >= K * atr_pct_288[bottom_idx]
    top_hit = mfe_down_pct[top_idx] >= K * atr_pct_288[top_idx]
    n_total = len(bottom_idx) + len(top_idx)
    hit_rate = (bottom_hit.sum() + top_hit.sum()) / n_total
    return {"n": int(n_total), "edge_only_no_lowvol_gate_hit_rate": float(hit_rate)}


def split_train_val_oos(fires: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    return train, val, oos


def screen_one_combo(indicator_frame: pd.DataFrame, sig: pd.DataFrame, atr_pct_288: np.ndarray, horizon: int, gap: int) -> tuple[dict, pd.DataFrame]:
    from tabpfn import TabPFNClassifier

    fires_raw = build_raw_fires(indicator_frame, sig, atr_pct_288, gap, horizon)
    n_before_dropna = len(fires_raw)
    fires_raw = fires_raw.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    K, _ = calibrate_k(fires_raw)
    fires = apply_k(fires_raw, K)

    train, val, oos = split_train_val_oos(fires)
    clf = TabPFNClassifier(device="cuda", random_state=SCREEN_SEED)
    clf.fit(train[FEATURE_COLUMNS], train["hit"].to_numpy().astype(int))
    val_auc = roc_auc_score(val["hit"].to_numpy().astype(int), clf.predict_proba(val[FEATURE_COLUMNS])[:, 1])
    oos_auc = roc_auc_score(oos["hit"].to_numpy().astype(int), clf.predict_proba(oos[FEATURE_COLUMNS])[:, 1])

    row = {
        "horizon": horizon, "gap": gap, "K": K,
        "n_fires_before_dropna": n_before_dropna, "n_fires": int(len(fires)),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_oos": int(len(oos)),
        "hit_rate": round(float(fires["hit"].mean()), 4),
        "val_auc": round(float(val_auc), 4), "oos_auc": round(float(oos_auc), 4),
        "gap_val_oos": round(abs(float(val_auc) - float(oos_auc)), 4),
    }
    log(f"[screen] H={horizon:>2d} gap={gap:>2d} K={K:.2f}: n={row['n_fires']} (train={row['n_train']}/val={row['n_val']}/oos={row['n_oos']}) "
        f"hit_rate={row['hit_rate']:.3f} VAL_AUC={row['val_auc']:.4f} OOS_AUC={row['oos_auc']:.4f} gap={row['gap_val_oos']:.4f}")
    return row, fires


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading klines + building Tier0 indicator frame + compute_signals...")
    klines = load_klines()
    indicator_frame = build_indicator_frame(klines)
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    assert len(sig) == len(indicator_frame) and (sig["timestamp"].to_numpy() == indicator_frame["timestamp"].to_numpy()).all()
    atr_pct_288 = compute_atr_pct_288(klines)

    log(f"=== screening grid: HORIZON in {HORIZON_GRID} x CLUSTER_GAP_MERGE in {GAP_GRID} "
        f"({len(HORIZON_GRID)*len(GAP_GRID)} combos, single seed={SCREEN_SEED}, TRAIN-fit -> VAL+OOS AUC, HOLDOUT untouched) ===")
    screening_rows = []
    fires_cache: dict[tuple[int, int], pd.DataFrame] = {}
    for horizon in HORIZON_GRID:
        for gap in GAP_GRID:
            row, fires = screen_one_combo(indicator_frame, sig, atr_pct_288, horizon, gap)
            screening_rows.append(row)
            fires_cache[(horizon, gap)] = fires

    # Selection: maximize min(VAL,OOS) among combos with a reasonable VAL-OOS gap -- volume_wick_
    # climax lesson (raw VAL-max alone can pick an overfit point). Report both for transparency.
    by_val_max = max(screening_rows, key=lambda r: r["val_auc"])
    by_min_auc = max(screening_rows, key=lambda r: min(r["val_auc"], r["oos_auc"]))
    log(f"if selected by raw VAL max: H={by_val_max['horizon']} GAP={by_val_max['gap']} "
        f"(VAL={by_val_max['val_auc']:.4f} OOS={by_val_max['oos_auc']:.4f} gap={by_val_max['gap_val_oos']:.4f})")
    log(f"if selected by max(min(VAL,OOS)): H={by_min_auc['horizon']} GAP={by_min_auc['gap']} "
        f"(VAL={by_min_auc['val_auc']:.4f} OOS={by_min_auc['oos_auc']:.4f} gap={by_min_auc['gap_val_oos']:.4f})")
    best = by_min_auc
    log(f"=== SELECTED (by max(min(VAL,OOS)), per volume_wick_climax lesson): HORIZON={best['horizon']} GAP={best['gap']} K={best['K']:.2f} ===")

    horizon_f, gap_f, K_f = best["horizon"], best["gap"], best["K"]
    fires = fires_cache[(horizon_f, gap_f)]
    log(f"final fire counts: total={len(fires)} (bottom={int((fires['side']=='bottom').sum())}, top={int((fires['side']=='top').sum())})")

    baseline = random_bar_baseline_no_lowvol_gate(indicator_frame, klines, atr_pct_288, horizon_f, K_f)
    fire_hit_rate = float(fires["hit"].mean())
    log(f"fired-signal hit rate: {fire_hit_rate:.4f} vs edge-only-no-lowvol-gate baseline: "
        f"{baseline['edge_only_no_lowvol_gate_hit_rate']:.4f} (lift {fire_hit_rate/baseline['edge_only_no_lowvol_gate_hit_rate']:.3f}x)")

    train, val, oos = split_train_val_oos(fires)
    holdout = fires.loc[fires["timestamp"] >= HOLDOUT_START].reset_index(drop=True)
    log(f"TRAIN(<{VAL_START.date()}) n={len(train)}, VAL n={len(val)}, OOS n={len(oos)}, HOLDOUT(>={HOLDOUT_START.date()}) n={len(holdout)}")

    fires.to_csv(OUT_DIR / "eth_5m_dalton_rule2_balance_edge_metalabel_features.csv", index=False)

    log("=== VAL evaluation (TRAIN-fit, 4 seeds) ===")
    val_result = run_tabpfn_panel(train, val, FEATURE_COLUMNS, "VAL")
    log(f"VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}")

    log("=== OOS evaluation (TRAIN-fit, 4 seeds) ===")
    oos_result = run_tabpfn_panel(train, oos, FEATURE_COLUMNS, "OOS")
    log(f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}")

    log("=== RESERVED HOLDOUT evaluation (single touch, TRAIN-fit, 4 seeds) ===")
    holdout_result = run_tabpfn_panel(train, holdout, FEATURE_COLUMNS, "HOLDOUT") if len(holdout) >= 30 else {"note": "too few holdout fires"}
    if "auc_mean" in holdout_result:
        log(f"HOLDOUT -> AUC {holdout_result['auc_mean']:.4f}+/-{holdout_result['auc_std']:.4f}")

    log("=== permutation feature importance (VAL, single seed, 5 repeats) ===")
    perm_importance = compute_permutation_importance(train, val, FEATURE_COLUMNS)
    for row in perm_importance["importances"][:10]:
        log(f"  {row['feature']:<22s} importance={row['importance_mean']:+.5f}")

    report = {
        "signal": "dalton_rule2_balance_edge", "adopted_version": "v1",
        "status": "exploratory_single_signal_below_promotion_bar",
        "screening_grid": screening_rows,
        "selected_by": "max(min(VAL,OOS))", "selection_alt_by_val_max": by_val_max,
        "selected_horizon": horizon_f, "selected_gap": gap_f, "selected_K": K_f,
        "feature_columns": FEATURE_COLUMNS, "n_fires_total": int(len(fires)),
        "random_bar_baseline_no_lowvol_gate": baseline,
        "fired_signal_hit_rate": fire_hit_rate,
        "lift_vs_edge_only_baseline": fire_hit_rate / baseline["edge_only_no_lowvol_gate_hit_rate"],
        "val": val_result, "oos": oos_result, "reserved_holdout": holdout_result,
        "permutation_importance_val": perm_importance,
    }
    out_path = REPORT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"report saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
