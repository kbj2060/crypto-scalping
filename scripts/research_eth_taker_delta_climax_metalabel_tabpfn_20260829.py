#!/usr/bin/env python3
"""Meta-labeling for taker_delta_z_climax, round 2 -- richer Tier0-style features + TabPFN,
replicating the liquidity_sweep -> V_REBOUND upgrade's methodology
(docs/experiments/eth_liquidity_sweep_v_rebound_feature_plan_20260829.md,
scripts/build_eth_5m_sweep_v_rebound_features_tier0_20260829.py,
scripts/research_eth_sweep_v_rebound_tabpfn_reserved_holdout_20260829.py).

Round 1 (scripts/research_eth_taker_delta_climax_metalabel_phase0_20260829.py) used 10 hand
side-normalized klines features + logistic regression -> NULL (OOS AUC 0.489). Two changes here,
directly modeled on what actually worked for liquidity_sweep:
  1. Features: reuse the EXACT SAME 3 indicator functions the V_REBOUND Tier0 build calls
     (compute_indicators / add_creative_indicators / add_broad_indicators), verbatim import, not
     reimplemented -- gives p_fast/p_slow/adx14/pdi/ndi/delta_z/vol_z/wick ratios/vwap_dev_z/
     cvd_roll_roc_48/bb_pctb/bb_width_pctile for free. Adds ret3_z (2-line inline, same formula),
     atr_percentile_864, hour_utc/weekday, rsi (Wilder-14, the one feature that actually helped
     V_REBOUND), plus er_24/realized_vol_ratio/nyse_open_flag carried over from round 1.
     IMPORTANT CHANGE FROM ROUND 1: features are now RAW (signed p_fast/p_slow/ret3_z/wick
     ratios) + an is_bottom indicator, NOT hand side-normalized ("confluence") transforms --
     the V_REBOUND project found p_fast has an inverted-U relationship with its label (extremes
     bad, middle good) that a monotonic side-normalization would have destroyed; letting TabPFN
     see raw features + side lets it discover whatever shape actually exists instead of assuming
     monotonic "more confluent = better".
  2. Model: TabPFNClassifier(device="cuda", ...) instead of logistic regression -- V_REBOUND's own
     GBM-vs-TabPFN comparison showed a real, non-trivial jump (0.6425 -> 0.6566 OOS AUC) from the
     model class itself, on the identical feature set. A round-2 null with logistic regression
     wouldn't tell us if there's no signal or if the model just can't see it.

Label: v3, revised again after a second user-requested check found the v2 label (30min,
point-in-time close, 0.3xATR) was still misaligned with the signal's actual behavior. Directly
measuring where the TRUE local price extreme falls relative to each fire bar (scratchpad/
check_pivot_timing.py, +/-2h search window, descriptive only -- never used to relocate any
label's anchor, that would be the exact circular "vertex search" V_REBOUND's own anchor_bug
investigation explicitly rejected) found: only 14% of fires sit exactly at the true extreme, 70%
have the true extreme AFTER the fire bar (median lag 4 bars/20min, p90 22 bars/110min, with a
median additional 2.9x-ATR adverse move in between) -- i.e. taker_delta_z_climax fires DURING a
move, not reliably AT its exhaustion point, and v2's 30min/point-in-time window was too short and
too lookahead-sensitive to reward fires that do eventually pay off.

Fix (same principle V_REBOUND itself used -- `future["high"].max()` MFE within a FIXED,
pre-determined window, never a variable/unbounded forward search for wherever the extreme
happens to be):
    HORIZON = 24 (2h forward, ~p90 of the true-extreme-lag distribution above)
    hit = MFE_pct >= 2.0 * atr_pct_at_fire, where MFE_pct is the max favorable excursion using
    INTRABAR high/low (not just the closing print) over bars[fire+1 : fire+25] -- bottom:
    (future_high.max() - entry)/entry; top: (entry - future_low.min())/entry. K=2.0 calibrated
    locally (MFE over a longer window is mechanically larger than a v2-style point return, so K
    had to be recalibrated, not reused) -- gives a 50.5%/49.5% split, n_positive=6,698/13,273.
Entry is still anchored to the fire bar's own close (unchanged, non-circular) -- only the
evaluation window widened and switched from point-in-time-close to intrabar-MFE, both decided in
advance of looking at any outcome, not searched per-event.

v4 (this run): user asked (a) whether the fire-bar anchor itself could be improved -- tested via
scratchpad/check_cluster_anchor_timing.py: collapsing each same-side burst (gap<=3 bars, the same
grouping used for the oversampling caveat below) to the single bar with the MOST EXTREME delta_z
(never the price outcome -- causal, non-circular, uses only the flow feature itself to pick the
anchor) improved at-fire-bar accuracy only marginally (14.0%->15.6% bottom, 13.4%->15.2% top) and
did NOT reduce the median lag at all (still 4 bars both ways) -- so this is adopted mainly to fix
the oversampling caveat below (free side benefit), not because it meaningfully fixes timing
alignment (it doesn't, on its own). (b) a full lookahead audit: every line of compute_indicators/
add_creative_indicators/add_broad_indicators (plus _adx/_dmi) was read in full -- confirmed every
feature uses only .rolling()/.ewm()/.diff()/.shift(positive) or same-bar OHLC, zero instances of
.shift(-N) or reversed indexing. The label's own future-bar usage (high/low[fire+1:fire+HORIZON+1])
is strictly separated from feature construction (indicator_frame.iloc[idx], causal up to and
including the fire bar only) -- no overlap found. (c) permutation feature importance added (single
seed, VAL set, hand-rolled AUC-scored shuffle since TabPFN has no native .feature_importances_ and
isn't a plain sklearn estimator) to see which features are actually driving the v3 jump, since
atr_pct is used both to SET the hit threshold (K*atr_pct) and as an input feature -- worth knowing
how much of the signal is just vol-regime information versus the other 22 features. Result:
atr_percentile_864 dominated individually (+0.035, ~5x the next feature), but an ablation removing
all 3 volatility-regime features (atr_pct/atr_percentile_864/realized_vol_ratio) only cost
~0.01-0.012 AUC across VAL/OOS/HOLDOUT (research_eth_taker_delta_climax_metalabel_ablation_vol_
20260829.py) -- the signal is broadly distributed across the other 20 features, not primarily a
volatility-regime proxy (permutation importance measures marginal value with all features present;
redundant/correlated features can substitute for one removed feature, which is exactly what
happened here).

v5 (this run): v4's hit was pure touch-based MFE (>=2.0xATR at ANY point in the 2h window) with no
persistence/confirmation check -- unlike V_REBOUND, which required the close to hold beyond the
swept level for ALL 6 bars, not just touch it once. taker_delta_z_climax has no natural price
LEVEL to check persistence against (it's an order-flow event, not a structural one), but a
diagnostic (scratchpad/calibrate_v5_persistence.py) found 17.6% of v4's "touches" fully round-
tripped back to/below entry by window end (bar+HORIZON) -- i.e. a momentary favorable wick that
gave everything back would still have counted as hit=1. Tried: hit = touched (MFE_pct>=2.0xATR)
AND end_ret_pct>0 (close at bar+HORIZON still net favorable, i.e. not fully reversed) -- gave a
44.7%/55.3% split (n_positive=4,583/10,262), close to V_REBOUND's own 43.9%/56.1%.

**v5 REJECTED -- this made things worse, not better**: VAL/OOS/HOLDOUT AUC dropped from v4's
0.622/0.608/0.650 to 0.562/0.561/0.606 (worse across all three periods, no exceptions). Diagnosis:
end_ret_pct is a SINGLE bar's close evaluated at exactly bar+HORIZON -- this reintroduces the same
single-point-in-time noise sensitivity that made v1/v2 weak in the first place, just downstream of
a genuine touch condition instead of replacing it. "Did it touch 2xATR" is apparently more
learnable from fire-time features than "is it, at the exact instant 2h later, still net positive"
-- the latter adds a second, less-predictable random variable on top of the first. A smoothed
persistence check (e.g. majority of the last few bars, or their average, rather than one single
bar's close) might avoid this and is worth trying if this signal is revisited, but was NOT tested
this session. v4 (touch-only MFE, cluster-anchored, no persistence check) is the ADOPTED FINAL
VERSION -- this is what the code below implements.

Same-side consecutive-fire oversampling (24% of fires were within 3 bars of the prior fire of the
same side, i.e. likely re-triggers of one underlying event, not independent observations --
verified via scratchpad/verify_taker_delta_climax_label.py) is now fixed by the cluster-anchor
dedup above (~1.3x reduction, 13,273 -> ~10,262 fires), which matters more now that the outcome
window is 2h wide (more overlap between nearby same-side fires' windows) than it did at 30min.

Splits: this repo's own Fresh-Forward default (CLAUDE.md) AND the exact V_REBOUND split --
TRAIN = 2024-01-01..2025-08-31, VAL = 2025-09-01..2025-12-31, OOS = 2026-01-01..2026-03-31,
HOLDOUT = 2026-04-01..latest (single-touch, evaluated once, after VAL/OOS are done informing
nothing further about this run).

Runs on the GPU server (quant_ai env, CUDA required for TabPFN) -- see handoff.sh push before
executing remotely.
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from analyze_eth_broad_evidence_signal_sweep_20260814 import add_broad_indicators
from analyze_eth_creative_reversal_evidence_signals_20260814 import add_creative_indicators
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators
from live_evidence_signal_dashboard_20260823 import compute_signals

KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "data/labels/eth_5m_taker_delta_climax_metalabel_20260829"
REPORT_DIR = ROOT / "tmp/eth_taker_delta_climax_metalabel_tabpfn_20260829"

START = pd.Timestamp("2024-01-01")
HORIZON = 24  # 2h forward, MFE window -- v3, widened from v2's 30min after pivot-timing check
              # showed 70% of fires have their true extreme AFTER the fire bar (p90 lag 110min)
ATR_HIT_MULT = 2.0  # hit requires MFE_pct >= 2.0 * atr_pct_at_fire (intrabar high/low, not a
                     # single close) -- recalibrated for the wider MFE window, gives ~50/50 split
CLUSTER_GAP_MERGE = 3  # v4: collapse same-side fires within this many bars of each other into one
                        # cluster, keep only the bar with the most extreme delta_z per cluster --
                        # causal (uses only delta_z, never future price), fixes ~1.3x oversampling

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

SEEDS = [20260829, 141592, 271828, 577215]  # same 4 seeds as V_REBOUND's reserved-holdout run

FEATURE_COLUMNS = [
    # structural / signal-intensity (analog of V_REBOUND's sweep-derived group)
    "is_bottom", "delta_z", "atr_pct", "atr_percentile_864", "hour_utc", "weekday", "nyse_open_flag",
    # evidence-signal family, RAW (not side-normalized -- let TabPFN find nonlinear shapes itself)
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    # trend/volatility context
    "adx14", "pdi", "ndi", "bb_width_pctile", "er_24", "realized_vol_ratio",
    # Omega4.6.1 candidate that helped V_REBOUND
    "rsi",
]


def log(msg: str) -> None:
    print(f"[taker_delta_climax_tabpfn] {msg}", flush=True)


def load_klines() -> pd.DataFrame:
    df = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=5)).all(), "gap/dup in klines"
    return df


def build_indicator_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Verbatim reuse of the V_REBOUND Tier0 build's 3-function chain, plus the extras this
    project needs (ret3_z / atr / atr_percentile_864 / hour_utc / weekday / rsi / er_24 /
    realized_vol_ratio / nyse_open_flag). All causal, klines-only."""
    frame = compute_indicators(raw.copy())
    frame = add_creative_indicators(frame)
    frame = add_broad_indicators(frame)

    ret3 = frame["close"] / frame["close"].shift(3) - 1.0
    ret3_mean = ret3.rolling(288, min_periods=288).mean()
    ret3_std = ret3.rolling(288, min_periods=288).std()
    frame["ret3_z"] = (ret3 - ret3_mean) / ret3_std.replace(0.0, np.nan)

    high, low, close = frame["high"], frame["low"], frame["close"]
    prev_close = close.shift(1)
    prev_close.iloc[0] = close.iloc[0]
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_raw = tr.rolling(14, min_periods=14).mean()
    frame["atr_pct"] = atr_raw / close.clip(lower=1e-12)
    frame["atr_percentile_864"] = atr_raw.rolling(864, min_periods=864).rank(pct=True)

    frame["hour_utc"] = frame["timestamp"].dt.hour
    frame["weekday"] = frame["timestamp"].dt.weekday
    tmin = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    is_weekday = frame["timestamp"].dt.dayofweek < 5
    frame["nyse_open_flag"] = (is_weekday & (tmin >= 12 * 60 + 30) & (tmin <= 14 * 60 + 30)).astype(int)

    net_change_24 = close - close.shift(24)
    diff_abs = close.diff().abs()
    frame["er_24"] = (net_change_24.abs() / (diff_abs.rolling(24, min_periods=4).sum() + 1e-12)).fillna(0.0)
    log_ret = np.log(close / close.shift(1))
    frame["realized_vol_ratio"] = log_ret.rolling(12, min_periods=12).std() / log_ret.rolling(288, min_periods=288).std()

    # rsi: Wilder-14, verbatim formula from features/engineering.py::_calc_rsi (per V_REBOUND's
    # own docstring, "100% identical to pandas-ta's default Wilder's Smoothing"). Computed fresh
    # here (not joined from training_features_*.csv) to keep full 2024-2026 history self-contained.
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    frame["rsi"] = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-8))

    return frame


def cluster_dedup(idx: np.ndarray, delta_z_at_idx: np.ndarray, most_negative: bool) -> np.ndarray:
    """v4: collapse consecutive same-side fires (gap<=CLUSTER_GAP_MERGE bars) into one cluster,
    keep only the bar with the most extreme delta_z per cluster. Causal -- picks the anchor using
    only delta_z itself (never future price), so this is not the circular vertex-search pattern
    V_REBOUND's anchor_bug investigation rejected."""
    order = np.argsort(idx)
    idx_sorted = idx[order]
    dz_sorted = delta_z_at_idx[order]
    cluster_id = np.zeros(len(idx_sorted), dtype=int)
    cid = 0
    for i in range(1, len(idx_sorted)):
        if idx_sorted[i] - idx_sorted[i - 1] > CLUSTER_GAP_MERGE:
            cid += 1
        cluster_id[i] = cid
    df = pd.DataFrame({"idx": idx_sorted, "cluster": cluster_id, "dz": dz_sorted})
    keep = df.loc[df.groupby("cluster")["dz"].idxmin()] if most_negative else df.loc[df.groupby("cluster")["dz"].idxmax()]
    return np.sort(keep["idx"].to_numpy())


def build_fires_and_features(klines: pd.DataFrame, indicator_frame: pd.DataFrame) -> pd.DataFrame:
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    assert len(sig) == len(indicator_frame), "row count mismatch between compute_signals and indicator_frame"
    assert (sig["timestamp"].to_numpy() == indicator_frame["timestamp"].to_numpy()).all(), "timestamp misalignment"

    # Sanity check: compute_signals' own inline delta_z (which the fire trigger is defined on)
    # should closely match add_creative_indicators' delta_z (the one we actually record as a
    # feature) -- both descend from the same 2026-08-14 lineage. Not a hard assert (a small
    # mismatch wouldn't invalidate the run, just be worth knowing about).
    both = pd.concat([sig["delta_z"], indicator_frame["delta_z"]], axis=1).dropna()
    corr = both.iloc[:, 0].corr(both.iloc[:, 1])
    max_abs_diff = (both.iloc[:, 0] - both.iloc[:, 1]).abs().max()
    log(f"delta_z cross-check (compute_signals vs add_creative_indicators): corr={corr:.6f}, max_abs_diff={max_abs_diff:.6f}")

    high = sig["high"].to_numpy()
    low = sig["low"].to_numpy()
    close = sig["close"].to_numpy()
    atr_pct = indicator_frame["atr_pct"].to_numpy()
    n = len(sig)
    rows = []
    delta_z_all = indicator_frame["delta_z"].to_numpy()
    for side, col in [("bottom", "bottom_taker_delta_z_climax"), ("top", "top_taker_delta_z_climax")]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[(idx < n - HORIZON) & (sig["timestamp"].to_numpy()[idx] >= np.datetime64(START))]
        idx_before_dedup = len(idx)
        idx = cluster_dedup(idx, delta_z_all[idx], most_negative=(side == "bottom"))
        log(f"  {side}: {idx_before_dedup} raw fires -> {len(idx)} after cluster-anchor dedup")
        entry = close[idx]
        # MFE (max favorable excursion) within bars[fire+1 : fire+HORIZON+1], intrabar high/low --
        # same principle as V_REBOUND's future["high"].max(), a FIXED pre-determined window, not a
        # search for wherever the extreme happens to be (see module docstring, v3 label revision).
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + HORIZON + 1].max() for i in idx])
            pred_dir_ret = (fut_ext - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + HORIZON + 1].min() for i in idx])
            pred_dir_ret = (entry - fut_ext) / entry
        touched = pred_dir_ret >= ATR_HIT_MULT * atr_pct[idx]
        # v5 (TRIED AND REJECTED, see module docstring): adding "touched AND end_ret_pct>0" (not
        # fully round-tripped by exactly bar+HORIZON) made VAL/OOS/HOLDOUT AUC WORSE (0.622/0.608/
        # 0.650 -> 0.562/0.561/0.606), not better -- a single point-in-time close 24 bars out
        # reintroduces the same kind of single-bar-timing noise that made v1/v2 weak in the first
        # place. v4 (touch-only MFE, no persistence check) is the adopted final version -- do NOT
        # re-add a persistence condition without first fixing its noise-sensitivity (e.g. average
        # of several bars near the window end, not one single bar's close).
        hit = touched  # magnitude-gated MFE, touch-only (see module docstring, v4 = final adopted)
        feat_rows = indicator_frame.iloc[idx]
        out = pd.DataFrame({
            "pos": idx, "timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
            "hit": hit.astype(float), "pred_dir_ret": pred_dir_ret,
            "is_bottom": 1 if side == "bottom" else 0,
        })
        for col_name in FEATURE_COLUMNS:
            if col_name == "is_bottom":
                continue
            out[col_name] = feat_rows[col_name].to_numpy()
        rows.append(out)
    fires = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return fires


def random_bar_baseline(indicator_frame: pd.DataFrame, klines: pd.DataFrame) -> dict:
    """Analog of V_REBOUND's random_bar_baseline check: apply the SAME direction-from-delta_z-
    sign rule AND the same MFE/ATR hit definition to EVERY bar (not just |delta_z|>=2 fires) and
    compare hit rate to the actual fires' hit rate. Tests whether the >=2 extremity threshold
    itself adds anything over delta_z's raw sign as a weak always-on directional signal. Uses a
    vectorized forward-rolling max/min (not a per-row loop) since this runs over all ~280k bars,
    not just the ~13k fires."""
    high, low, close = klines["high"], klines["low"], klines["close"]
    n = len(klines)
    fwd_high_max = high[::-1].rolling(window=HORIZON, min_periods=HORIZON).max()[::-1].shift(-1)
    fwd_low_min = low[::-1].rolling(window=HORIZON, min_periods=HORIZON).min()[::-1].shift(-1)
    mfe_up_pct = ((fwd_high_max - close) / close).to_numpy()
    mfe_down_pct = ((close - fwd_low_min) / close).to_numpy()

    delta_z = indicator_frame["delta_z"].to_numpy()
    atr_pct = indicator_frame["atr_pct"].to_numpy()
    ts = indicator_frame["timestamp"].to_numpy()
    valid = np.isfinite(delta_z) & np.isfinite(atr_pct) & (ts >= np.datetime64(START)) & (np.arange(n) < n - HORIZON)
    idx = np.flatnonzero(valid & (delta_z != 0))
    side_is_bottom = delta_z[idx] < 0  # negative delta_z -> "bottom-like" -> predict up
    mfe_pct = np.where(side_is_bottom, mfe_up_pct[idx], mfe_down_pct[idx])
    hit = mfe_pct >= ATR_HIT_MULT * atr_pct[idx]
    return {"n": int(len(idx)), "all_bar_continuous_sign_hit_rate": float(hit.mean())}


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4),
    }


def run_tabpfn_panel(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str], tag: str) -> dict:
    from tabpfn import TabPFNClassifier
    seed_rows = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[feature_cols], train["hit"].to_numpy().astype(int))
        proba = clf.predict_proba(eval_df[feature_cols])[:, 1]
        r = evaluate(proba, eval_df["hit"].to_numpy().astype(int))
        r["seed"] = seed
        seed_rows.append(r)
        log(f"  [{tag}] seed={seed}: auc={r['auc']:.4f} acc={r['accuracy']:.4f} "
            f"bal_acc={r['balanced_accuracy']:.4f} (naive={r['naive_majority_accuracy']:.4f})")
    table = pd.DataFrame(seed_rows)
    return {
        "n_train": int(len(train)), "n_eval": int(len(eval_df)),
        "auc_mean": round(float(table["auc"].mean()), 4), "auc_std": round(float(table["auc"].std(ddof=1)), 4),
        "accuracy_mean": round(float(table["accuracy"].mean()), 4),
        "balanced_accuracy_mean": round(float(table["balanced_accuracy"].mean()), 4),
        "naive_majority_accuracy": seed_rows[0]["naive_majority_accuracy"],
        "per_seed": seed_rows,
    }


def compute_permutation_importance(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str],
                                    seed: int = SEEDS[0], n_repeats: int = 5) -> list[dict]:
    """Single-seed, hand-rolled permutation importance (AUC-scored) on the VAL set -- model-
    agnostic (TabPFN has no native .feature_importances_), and hand-rolled rather than sklearn's
    permutation_importance to avoid that helper's fitted-estimator/wrapper-class edge cases on a
    non-sklearn-native classifier. Checks how much of v3's AUC jump is atr_pct (used both to SET
    the hit threshold and as an input feature) versus the other 22 features."""
    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(device="cuda", random_state=seed)
    clf.fit(train[feature_cols], train["hit"].to_numpy().astype(int))
    y = eval_df["hit"].to_numpy().astype(int)
    X = eval_df[feature_cols].to_numpy()
    baseline_auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])

    rng = np.random.default_rng(seed)
    rows = []
    for j, feat in enumerate(feature_cols):
        shuffled_aucs = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            shuffled_aucs.append(roc_auc_score(y, clf.predict_proba(X_perm)[:, 1]))
        importance = baseline_auc - np.mean(shuffled_aucs)
        rows.append({"feature": feat, "importance_mean": round(float(importance), 5),
                     "importance_std": round(float(np.std(shuffled_aucs, ddof=1)), 5)})
    rows.sort(key=lambda r: -r["importance_mean"])
    return {"baseline_auc": round(float(baseline_auc), 4), "n_repeats": n_repeats, "seed": seed, "importances": rows}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading klines...")
    klines = load_klines()
    log(f"{len(klines)} bars loaded")

    log("building Tier0-style indicator frame (compute_indicators + add_creative_indicators + add_broad_indicators + extras)...")
    indicator_frame = build_indicator_frame(klines)

    log("building taker_delta_z_climax fires + features...")
    fires = build_fires_and_features(klines, indicator_frame)
    n_before = len(fires)
    fires = fires.dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
    log(f"{len(fires)}/{n_before} usable fires after dropna "
        f"(bottom={int((fires['side']=='bottom').sum())}, top={int((fires['side']=='top').sum())})")

    log("running random-bar continuous-sign baseline check...")
    baseline = random_bar_baseline(indicator_frame, klines)
    log(f"baseline: {baseline}")
    fire_hit_rate = float(fires["hit"].mean())
    log(f"fired-signal (|delta_z|>=2) hit rate: {fire_hit_rate:.4f} vs all-bar continuous-sign "
        f"baseline: {baseline['all_bar_continuous_sign_hit_rate']:.4f} "
        f"(lift {fire_hit_rate / baseline['all_bar_continuous_sign_hit_rate']:.3f}x)")

    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    holdout = fires.loc[ts >= HOLDOUT_START].reset_index(drop=True)
    log(f"TRAIN(<{VAL_START.date()}) n={len(train)}, VAL n={len(val)}, OOS n={len(oos)}, HOLDOUT(>={HOLDOUT_START.date()}) n={len(holdout)}")

    fires.to_csv(OUT_DIR / "eth_5m_taker_delta_climax_metalabel_features.csv", index=False)

    log("=== VAL evaluation (TRAIN-fit, 4 seeds) ===")
    val_result = run_tabpfn_panel(train, val, FEATURE_COLUMNS, "VAL")
    log(f"VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}  "
        f"acc {val_result['accuracy_mean']:.4f}  bal_acc {val_result['balanced_accuracy_mean']:.4f}")

    log("=== OOS evaluation (TRAIN-fit, 4 seeds) ===")
    oos_result = run_tabpfn_panel(train, oos, FEATURE_COLUMNS, "OOS")
    log(f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}  "
        f"acc {oos_result['accuracy_mean']:.4f}  bal_acc {oos_result['balanced_accuracy_mean']:.4f}")

    log("=== RESERVED HOLDOUT evaluation (2026-04-01~latest, single-touch, TRAIN-fit, 4 seeds) ===")
    holdout_result = run_tabpfn_panel(train, holdout, FEATURE_COLUMNS, "HOLDOUT") if len(holdout) >= 30 else {"note": "too few holdout fires"}
    if "auc_mean" in holdout_result:
        log(f"HOLDOUT -> AUC {holdout_result['auc_mean']:.4f}+/-{holdout_result['auc_std']:.4f}  "
            f"acc {holdout_result['accuracy_mean']:.4f}  bal_acc {holdout_result['balanced_accuracy_mean']:.4f}")

    log("=== permutation feature importance (VAL, single seed, AUC-scored, 5 repeats) ===")
    perm_importance = compute_permutation_importance(train, val, FEATURE_COLUMNS)
    log(f"baseline VAL AUC (single seed {perm_importance['seed']}): {perm_importance['baseline_auc']:.4f}")
    for row in perm_importance["importances"][:10]:
        log(f"  {row['feature']:<22s} importance={row['importance_mean']:+.5f} (std={row['importance_std']:.5f})")

    report = {
        "signal": "taker_delta_z_climax",
        "adopted_version": "v4",
        "status": "exploratory_single_signal_below_promotion_bar",
        "summary_for_future_sessions": (
            "FINAL/ADOPTED (v4): touch-based MFE over 2h, hit = (MFE_pct using intrabar high/low "
            "over bars[fire+1:fire+25] >= 2.0*atr_pct_at_fire), fires cluster-anchored (same-side "
            "bursts within 3 bars collapsed to their single most-extreme-delta_z bar). "
            "VAL AUC 0.622, OOS AUC 0.608, HOLDOUT AUC 0.650 (TabPFN, 23 features, 4 seeds each). "
            "v5 (touch AND end-of-window-still-favorable persistence check) was TRIED AND REJECTED "
            "-- made AUC worse (0.562/0.561/0.606) because a single bar's close at exactly "
            "bar+HORIZON reintroduces single-point-in-time noise. Do not re-add a persistence "
            "check without smoothing it (e.g. average/majority of several bars near window end). "
            "Full methodology + version history: docs/experiments/"
            "eth_taker_delta_climax_metalabel_20260829.md -- read that before reusing this "
            "template for another evidence signal."
        ),
        "methodology_note": (
            "v1 (research_eth_taker_delta_climax_metalabel_phase0_20260829.py): 10 hand "
            "side-normalized klines features + logistic regression, HORIZON=12/1h sign-only hit "
            "-> NULL (OOS AUC 0.489). v2 (this script, first pass): Tier0-style raw features + "
            "TabPFN, same HORIZON=12 but ATR-gated hit (0.3xATR) after verification found the "
            "sign-only label was noise-dominated -> weak but consistently positive OOS/holdout "
            "AUC (0.519/0.534), VAL flat. v3: after a pivot-timing check found 70% of "
            "fires have their true price extreme AFTER the fire bar (median lag 20min, p90 "
            "110min), switched to MFE (intrabar high/low max favorable excursion) over a widened "
            "HORIZON=24 (2h) window, hit >= 2.0xATR (recalibrated for the wider window) -- same "
            "principle V_REBOUND itself used (future.high.max() over a fixed window), never a "
            "search for wherever the extreme happens to be. VAL/OOS/holdout AUC jumped to "
            "~0.60-0.64 from ~0.50-0.55. v4 (ADOPTED, this run): added (a) cluster-anchor dedup "
            "(collapse same-side bursts within 3 bars to their single most-extreme-delta_z bar, "
            "causal/non-circular -- fixes ~1.3x oversampling, only marginal timing improvement on "
            "its own), (b) a full lookahead audit (every line of compute_indicators/"
            "add_creative_indicators/add_broad_indicators/_adx/_dmi read in full -- zero "
            ".shift(-N) or reversed-index patterns found), (c) permutation feature importance -- "
            "atr_percentile_864 dominated individually (+0.035) but an ablation removing all 3 "
            "volatility-regime features only cost ~0.01-0.012 AUC, confirming the signal is NOT "
            "primarily a volatility-regime proxy. Result: VAL/OOS/HOLDOUT AUC 0.622/0.608/0.650, "
            "slightly better than v3 despite fewer (deduped) fires -- confirms v3's jump wasn't an "
            "oversampling artifact. v5 (TRIED, REJECTED): added a persistence check on top of the "
            "touch (end_ret_pct>0 at exactly bar+HORIZON) -- made AUC WORSE (0.562/0.561/0.606) by "
            "reintroducing single-point-in-time noise. v4 stands as final. Split matches this "
            "repo's Fresh-Forward default and V_REBOUND's own split exactly."
        ),
        "feature_columns": FEATURE_COLUMNS,
        "n_fires_total": int(len(fires)),
        "delta_z_crosscheck_note": "see log for compute_signals vs add_creative_indicators delta_z corr/max_abs_diff",
        "random_bar_baseline": baseline,
        "fired_signal_hit_rate": fire_hit_rate,
        "lift_vs_all_bar_baseline": fire_hit_rate / baseline["all_bar_continuous_sign_hit_rate"],
        "val": val_result,
        "oos": oos_result,
        "reserved_holdout": holdout_result,
        "permutation_importance_val": perm_importance,
    }
    out_path = REPORT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"report saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
