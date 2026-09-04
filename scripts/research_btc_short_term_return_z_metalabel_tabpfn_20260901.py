#!/usr/bin/env python3
"""Final TabPFN metalabel training for BTC short_term_return_z, porting round 2's winning label
recipe (docs/experiments/btc_5m_short_term_return_z_gridscreen_featureanalysis_20260901.md) onto
this project's established TabPFN methodology (research_eth_short_term_return_z_metalabel_tabpfn_
20260829.py -- this signal's own ETH template) via the shared run_tabpfn_panel/compute_permutation_
importance helpers (research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py -- the actual
implementation the ETH str_z script itself imports rather than reimplements; SEEDS is imported from
there too, single source of truth, and happens to equal this task's own spec value verbatim).

Label (round 2's winner, NOT re-searched here): HIT_TYPE=touch_mae_capped, HORIZON=6, K=2.0,
K_LOSS_MULT=2.0. Candidate at row i (is_down=bottom-side): entry=close[i], atr=atr[i]. touch_bar =
first bar in [i+1,i+6] where high>=entry+2.0*atr (bottom) / low<=entry-2.0*atr (top) -- no touch ->
hit=0. If touched: MAE = entry - low[i+1:touch_bar+1].min() (bottom) / high[i+1:touch_bar+1].max()
- entry (top); hit=1 iff MAE<=2.0*atr (disqualifies "touched the target but only after first giving
back more than 2x ATR against the position"). Ported verbatim from THIS exact recipe's own
validation script, research_btc_short_term_return_z_gridscreen_hittype_20260901.py::
hit_touch_mae_capped -- not reimplemented from the markdown description alone.

Cluster dedup: GAP=12 bars, collapse same-side fires within GAP into one cluster, keep the bar with
the most extreme ret3_z (this signal's own trigger variable) per cluster, causal (never uses future
price). This deviates from the originating task brief's GAP=6 fallback -- that fallback was offered
because no ETH-side convention could be found in the ETH TabPFN reference scripts, but a BTC-side,
THIS-SIGNAL-specific convention already exists and is not a free choice: it is literally the dedup
the round-2 grid screen used to validate the (H=6, K=2.0, touch_mae_capped) recipe being ported here
("GAP = 12 # fixed cluster-dedup convention (project-wide), NOT swept -- same as round 1", op cit).
Reusing GAP=6 instead would silently evaluate a different candidate pool than the one round 2
actually validated. Ported verbatim (cluster_dedup_gap) from the same round-2 script.

Features (24, verbatim from the task spec): is_bottom, atr_pct, atr_percentile_864, hour_utc,
weekday, nyse_open_flag, p_fast, p_slow, ret3_z, vwap_dev_z, cvd_roll_roc_48, vol_z,
lower_wick_ratio, upper_wick_ratio, bb_pctb, adx14, pdi, ndi, bb_width_pctile, er_24,
realized_vol_ratio, rsi, delta_z, range_width_pct. 19 of these already exist as columns in the BTC
Tier0 CSV (data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_
tier0.csv) and are read directly. atr_pct is ALSO already a CSV column -- verified byte-for-byte
equal to atr/close (max abs diff ~1e-17 across a 2000-row sample) -- so it is read directly too,
not recomputed. Only nyse_open_flag / er_24 / realized_vol_ratio are genuinely missing from the CSV
and are added here, ported verbatim from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.
py::build_indicator_frame (this project's shared Tier0-extras bank, per that script's own docstring
framing -- generic helpers meant to be reused across signals/assets, not hand-picked per signal).

Not reproduced here: the ETH template's all-bar continuous-sign baseline check. Round 2's own grid
screen already computed authoritative pooled TRAIN/VAL/OOS lift numbers for this EXACT recipe
(GAP=12, H=6, K=2.0, touch_mae_capped: TRAIN lift 1.487, VAL 1.480, OOS 1.494, per short_term_
return_z_gridscreen_report.json::recommended) using its own (different, random-same-count-draw)
baseline definition -- recomputing a second, differently-defined baseline here would risk two
inconsistent "lift" numbers for the same recipe without adding new information. This script instead
logs its own hit rates per split/side so they can be cross-checked against round 2's numbers
directly (same GAP=12/H=6/K=2.0 recipe -> should match almost exactly).

Splits: TRAIN < 2025-09-01, VAL 2025-09-01..2026-01-01, OOS 2026-01-01..2026-04-01, HOLDOUT
>= 2026-04-01 (2026-08-20 = the CSV's own last row). Round 1/round 2's grid screens truncated the
frame at HOLDOUT_START immediately and never read past it -- this script is the first to actually
touch HOLDOUT for this signal/asset, evaluated once, single-touch discipline (Fresh-Forward rule).

Runs on the GPU server (quant_ai env, CUDA required for TabPFN) under a system-wide flock
(.tabpfn_gpu.lock) -- see CLAUDE.md GPU-safety note, up to 6 concurrent signal ports share one
RTX 3070 Ti 8GB. Root path derived dynamically (Path(__file__).resolve().parents[1]), never
hardcoded -- dev and server use different usernames/paths.
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

from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (
    SEEDS,
    compute_permutation_importance,
    run_tabpfn_panel,
)

CSV_PATH = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv"
OUT_DIR = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901"
FEATURES_CSV = OUT_DIR / "short_term_return_z_tabpfn_features.csv"
REPORT_JSON = OUT_DIR / "short_term_return_z_tabpfn_report.json"

# --- round 2's winning label recipe (NOT re-searched here) ---
HORIZON = 6
ATR_HIT_MULT = 2.0     # K
K_LOSS_MULT = 2.0       # MAE-cap threshold multiplier, fixed
GAP = 12                # BTC short_term_return_z's own established cluster-dedup convention

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

TIER0_LOAD_COLUMNS = [
    "timestamp", "high", "low", "close",
    "atr", "atr_pct", "atr_percentile_864", "range_width_pct", "hour_utc", "weekday",
    "delta_z", "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi",
    "bb_width_pctile", "rsi",
    "bottom_short_term_return_z", "top_short_term_return_z",
]

FEATURE_COLUMNS = [
    "is_bottom", "atr_pct", "atr_percentile_864", "hour_utc", "weekday", "nyse_open_flag",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "er_24", "realized_vol_ratio",
    "rsi", "delta_z", "range_width_pct",
]


def log(msg: str) -> None:
    print(f"[btc_str_z_tabpfn] {msg}", flush=True)


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, usecols=TIER0_LOAD_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """nyse_open_flag / er_24 / realized_vol_ratio -- ported verbatim from
    research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py::build_indicator_frame."""
    close = df["close"]
    ts = df["timestamp"]

    tmin = ts.dt.hour * 60 + ts.dt.minute
    is_weekday = ts.dt.dayofweek < 5
    df["nyse_open_flag"] = (is_weekday & (tmin >= 12 * 60 + 30) & (tmin <= 14 * 60 + 30)).astype(int)

    net_change_24 = close - close.shift(24)
    diff_abs = close.diff().abs()
    df["er_24"] = (net_change_24.abs() / (diff_abs.rolling(24, min_periods=4).sum() + 1e-12)).fillna(0.0)

    log_ret = np.log(close / close.shift(1))
    df["realized_vol_ratio"] = log_ret.rolling(12, min_periods=12).std() / log_ret.rolling(288, min_periods=288).std()

    return df


def cluster_dedup_gap(idx: np.ndarray, anchor_val: np.ndarray, most_negative: bool, gap: int = GAP) -> np.ndarray:
    """Collapse same-side fires within `gap` bars into one cluster, keep only the bar with the most
    extreme anchor_val (ret3_z) per cluster. Causal. Ported verbatim from
    research_btc_short_term_return_z_gridscreen_hittype_20260901.py::cluster_dedup_gap -- the actual
    established BTC convention this signal's round-2 recipe was validated against (GAP=12)."""
    if len(idx) == 0:
        return idx
    order = np.argsort(idx)
    idx_sorted = idx[order]
    val_sorted = anchor_val[order]
    cluster_id = np.zeros(len(idx_sorted), dtype=int)
    cid = 0
    for i in range(1, len(idx_sorted)):
        if idx_sorted[i] - idx_sorted[i - 1] > gap:
            cid += 1
        cluster_id[i] = cid
    tmp = pd.DataFrame({"idx": idx_sorted, "cluster": cluster_id, "val": val_sorted})
    keep = (tmp.loc[tmp.groupby("cluster")["val"].idxmin()] if most_negative
            else tmp.loc[tmp.groupby("cluster")["val"].idxmax()])
    return np.sort(keep["idx"].to_numpy())


def hit_touch_mae_capped(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray,
                          idx: np.ndarray, horizon: int, k: float, side: str,
                          k_loss_mult: float = K_LOSS_MULT) -> np.ndarray:
    """Round 2's winning HIT_TYPE, ported verbatim from research_btc_short_term_return_z_
    gridscreen_hittype_20260901.py::hit_touch_mae_capped."""
    out = np.zeros(len(idx), dtype=int)
    for j, i in enumerate(idx):
        entry, a = close[i], atr[i]
        thresh = k * a
        if side == "bottom":
            cond = high[i + 1:i + horizon + 1] >= entry + thresh
            if not cond.any():
                continue
            touch_bar = i + 1 + int(cond.argmax())
            mae = entry - low[i + 1:touch_bar + 1].min()
        else:
            cond = low[i + 1:i + horizon + 1] <= entry - thresh
            if not cond.any():
                continue
            touch_bar = i + 1 + int(cond.argmax())
            mae = high[i + 1:touch_bar + 1].max() - entry
        if mae <= k_loss_mult * a:
            out[j] = 1
    return out


def build_fires_and_features(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr = df["atr"].to_numpy()
    ret3_z_all = df["ret3_z"].to_numpy()
    bottom_trig = df["bottom_short_term_return_z"].fillna(False).to_numpy()
    top_trig = df["top_short_term_return_z"].fillna(False).to_numpy()

    rows = []
    for side, trig, most_neg in [("bottom", bottom_trig, True), ("top", top_trig, False)]:
        idx = np.flatnonzero(trig)
        idx = idx[(idx < n - HORIZON) & np.isfinite(atr[idx]) & np.isfinite(ret3_z_all[idx])]
        idx_before_dedup = len(idx)
        idx = cluster_dedup_gap(idx, ret3_z_all[idx], most_negative=most_neg)
        log(f"  {side}: {idx_before_dedup} raw fires (bounded+finite) -> {len(idx)} after "
            f"cluster-anchor dedup (GAP={GAP})")

        hit = hit_touch_mae_capped(high, low, close, atr, idx, HORIZON, ATR_HIT_MULT, side)
        feat_rows = df.iloc[idx]
        out = pd.DataFrame({
            "pos": idx, "timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
            "hit": hit.astype(float), "is_bottom": 1 if side == "bottom" else 0,
        })
        for col_name in FEATURE_COLUMNS:
            if col_name == "is_bottom":
                continue
            out[col_name] = feat_rows[col_name].to_numpy()
        rows.append(out)
    fires = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return fires


def split_summary(fires: pd.DataFrame, ts_col: str = "timestamp") -> dict:
    ts = fires[ts_col]
    summary = {}
    for name, mask in [
        ("train", ts < VAL_START),
        ("val", (ts >= VAL_START) & (ts < OOS_START)),
        ("oos", (ts >= OOS_START) & (ts < HOLDOUT_START)),
        ("holdout", ts >= HOLDOUT_START),
    ]:
        sub = fires.loc[mask]
        row = {"n": int(len(sub))}
        for side in ("bottom", "top"):
            s = sub.loc[sub["side"] == side]
            row[f"n_{side}"] = int(len(s))
            row[f"hit_rate_{side}"] = round(float(s["hit"].mean()), 4) if len(s) else None
        row["hit_rate_pooled"] = round(float(sub["hit"].mean()), 4) if len(sub) else None
        summary[name] = row
    return summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading BTC Tier0 CSV...")
    df = load_frame()
    log(f"{len(df)} rows loaded, {df['timestamp'].min()} .. {df['timestamp'].max()}")

    log("adding derived features (nyse_open_flag / er_24 / realized_vol_ratio)...")
    df = add_derived_features(df)

    log("building short_term_return_z fires + features (touch_mae_capped, H=6, K=2.0)...")
    fires = build_fires_and_features(df)
    n_before = len(fires)
    fires = fires.dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
    log(f"{len(fires)}/{n_before} usable fires after dropna "
        f"(bottom={int((fires['side']=='bottom').sum())}, top={int((fires['side']=='top').sum())})")

    summary = split_summary(fires)
    for name in ("train", "val", "oos", "holdout"):
        s = summary[name]
        log(f"{name.upper()}: n={s['n']} (bottom={s['n_bottom']}/hit={s['hit_rate_bottom']}, "
            f"top={s['n_top']}/hit={s['hit_rate_top']}, pooled hit_rate={s['hit_rate_pooled']})")
    log("cross-check vs round 2 grid screen recommended cell (same GAP=12/H=6/K=2.0 recipe): "
        "TRAIN n=(1172,1212) hit_rate=(0.3276,0.3053); VAL n=(218,234) hit_rate=(0.3211,0.3462); "
        "OOS n=(190,176) hit_rate=(0.3158,0.3295) -- expect near-exact match.")

    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    holdout = fires.loc[ts >= HOLDOUT_START].reset_index(drop=True)

    fires.to_csv(FEATURES_CSV, index=False)
    log(f"features CSV saved -> {FEATURES_CSV}")

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
        "signal": "short_term_return_z",
        "asset": "BTCUSDT",
        "adopted_version": "v1 (round 2 winning recipe, ported)",
        "status": "exploratory_single_signal_below_promotion_bar",
        "summary_for_future_sessions": (
            "BTC port of short_term_return_z, final TabPFN stage. Label = round 2's winner: "
            "HIT_TYPE=touch_mae_capped, HORIZON=6, K=2.0, K_LOSS_MULT=2.0 (touch entry+/-K*atr "
            "within 6 bars, disqualified if MAE before the touch exceeded K_LOSS_MULT*atr). "
            "Cluster dedup GAP=12 (this signal's OWN established BTC convention from round 1/2's "
            "grid screens, used here instead of a generic GAP=6 fallback -- see module docstring). "
            "Features: 24, ported from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py "
            "for the 3 genuinely missing ones (nyse_open_flag/er_24/realized_vol_ratio); atr_pct "
            "reused directly from the Tier0 CSV's own column (verified == atr/close). Full grid "
            "search + recipe selection: docs/experiments/"
            "btc_5m_short_term_return_z_gridscreen_featureanalysis_20260901.md."
        ),
        "hit_type": "touch_mae_capped",
        "horizon": HORIZON,
        "atr_hit_mult_k": ATR_HIT_MULT,
        "k_loss_mult": K_LOSS_MULT,
        "cluster_gap": GAP,
        "cluster_gap_note": (
            "BTC short_term_return_z's own established convention (round 1/2 grid screens), used "
            "in place of the originating task brief's generic GAP=6 fallback -- see module docstring."
        ),
        "feature_columns": FEATURE_COLUMNS,
        "n_fires_total": int(len(fires)),
        "split_summary": summary,
        "round2_crosscheck_reference": {
            "note": "round 2 grid screen recommended cell (same GAP=12/H=6/K=2.0 recipe), for sanity comparison",
            "source": "data/labels/btc_5m_evidence_signal_candidates_20260901/short_term_return_z_gridscreen_report.json::recommended",
            "train_n_bottom": 1172, "train_n_top": 1212,
            "train_hit_rate_bottom": 0.3276, "train_hit_rate_top": 0.3053,
            "val_n_bottom": 218, "val_n_top": 234,
            "val_hit_rate_bottom": 0.3211, "val_hit_rate_top": 0.3462,
            "oos_n_bottom": 190, "oos_n_top": 176,
            "oos_hit_rate_bottom": 0.3158, "oos_hit_rate_top": 0.3295,
        },
        "splits": {
            "train": f"< {VAL_START.date()}",
            "val": f"{VAL_START.date()} ~ {OOS_START.date()}",
            "oos": f"{OOS_START.date()} ~ {HOLDOUT_START.date()}",
            "holdout": f">= {HOLDOUT_START.date()} (single-touch, first time evaluated this round)",
        },
        "seeds": SEEDS,
        "val": val_result,
        "oos": oos_result,
        "reserved_holdout": holdout_result,
        "permutation_importance_val": perm_importance,
        "eth_comparison_reference": {
            "note": "ETH's own short_term_return_z metalabel result, per project memory, for direct comparison",
            "val_auc": 0.674, "oos_auc": 0.649, "holdout_auc": 0.643,
        },
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str))
    log(f"report saved -> {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
