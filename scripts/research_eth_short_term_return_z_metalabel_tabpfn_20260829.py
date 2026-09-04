#!/usr/bin/env python3
"""Meta-labeling for short_term_return_z -- signal #3 in the established queue
(docs/experiments/eth_taker_delta_climax_metalabel_20260829.md "남은 6개 신호 진행 순서"), following
that document's reusable template verbatim: same Tier0-style 23-feature builder (imported, not
reimplemented), same TabPFN panel/permutation-importance helpers, same Fresh-Forward split.

Phase 1 (scratchpad diagnostics, 2026-08-29, not committed) found a DIFFERENT behavioral pattern
than taker_delta_z_climax, so the label design differs on purpose:
  - The true local price extreme is already AT OR BEFORE the fire bar 88-89% of the time (median
    lag -15 bars/-75min) -- opposite of taker_delta_z_climax (70% AFTER, median +20min). Expected:
    ret3_z is itself a backward-looking 3-bar return, so by the time it crosses +-2.5 the sharp
    move that triggered it has typically already finished.
  - Sign-only forward hit rate peaks at SHORT horizons (~56-57% bottom / ~53-54% top at 15min-1h)
    and DECAYS at 2h/4h (top even drops to 49.6% at 4h, below the naive floor) -- opposite of
    taker_delta_z_climax, which needed WIDENING to 2h to reveal its signal. This label therefore
    uses HORIZON=12 (1h), not taker's 24 (2h) -- copying taker's window would sit past this
    signal's own decay point.
  - Same-side consecutive fires cluster heavily (46-50% within 3-6 bars) -- cluster-anchoring
    (identical mechanism to taker_delta_z_climax v4: collapse same-side bursts within 3 bars to
    the single most-extreme ret3_z bar, causal/price-blind) applies here too.

Label (v1, ADOPTED after visual review -- user reviewed 20-example candlestick sanity-check chart,
scratchpad/render_eth_short_term_return_z_metalabel_v1_20260829.py, raised no objections):
  HORIZON = 12 (1h forward), hit = touched (intrabar MFE_pct, using high/low over
  bars[fire+1:fire+13], >= 1.75 * atr_pct_at_fire). K=1.75 was swept over {0.5..3.5} on the
  cluster-anchored fire set to find a roughly balanced split -- gives 51.1%/48.9% (bottom 52.2%,
  top 50.1%). No persistence check (taker_delta_z_climax's v5 already showed this makes things
  worse via single-point-in-time noise -- not re-tried here).

Features: identical 23-column Tier0-style set (compute_indicators + add_creative_indicators +
add_broad_indicators + ret3_z/atr_pct/atr_percentile_864/hour_utc/weekday/nyse_open_flag/
er_24/realized_vol_ratio/rsi), imported verbatim from the taker_delta_z_climax script rather than
reimplemented -- this is the shared "Tier0 bank", not something to hand-pick per signal.
`ret3_z` is short_term_return_z's OWN trigger variable but is kept as a feature anyway (exactly
like taker_delta_z_climax kept delta_z as both its trigger AND a feature) -- the magnitude beyond
the +-2.5 threshold may carry information the boolean fire condition alone doesn't.

Runs on the GPU server (quant_ai env, CUDA required for TabPFN) -- see handoff.sh push before
executing remotely. Root path is derived dynamically (Path(__file__).resolve().parents[1]), never
hardcoded -- dev and server use different usernames/paths (reference_dev_server_handoff gotcha).
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

from live_evidence_signal_dashboard_20260823 import compute_signals
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (
    FEATURE_COLUMNS,
    build_indicator_frame,
    cluster_dedup,
    compute_permutation_importance,
    load_klines,
    run_tabpfn_panel,
)

OUT_DIR = ROOT / "data/labels/eth_5m_short_term_return_z_metalabel_20260829"
REPORT_DIR = ROOT / "tmp/eth_short_term_return_z_metalabel_tabpfn_20260829"

START = pd.Timestamp("2024-01-01")
HORIZON = 12  # 1h forward MFE window -- phase1 found edge peaks 15m-1h, decays by 2h/4h
ATR_HIT_MULT = 1.75  # calibrated on the cluster-anchored fire set for a ~50/50 split
CLUSTER_GAP_MERGE = 3  # same mechanism/window as taker_delta_z_climax v4

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

SEEDS = [20260829, 141592, 271828, 577215]  # same 4 seeds as taker_delta_z_climax/V_REBOUND


def log(msg: str) -> None:
    print(f"[str_z_metalabel_tabpfn] {msg}", flush=True)


def build_fires_and_features(klines: pd.DataFrame, indicator_frame: pd.DataFrame) -> pd.DataFrame:
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    assert len(sig) == len(indicator_frame), "row count mismatch between compute_signals and indicator_frame"
    assert (sig["timestamp"].to_numpy() == indicator_frame["timestamp"].to_numpy()).all(), "timestamp misalignment"

    both = pd.concat([sig["ret3_z"], indicator_frame["ret3_z"]], axis=1).dropna()
    corr = both.iloc[:, 0].corr(both.iloc[:, 1])
    max_abs_diff = (both.iloc[:, 0] - both.iloc[:, 1]).abs().max()
    log(f"ret3_z cross-check (compute_signals vs build_indicator_frame): corr={corr:.6f}, max_abs_diff={max_abs_diff:.6f}")

    high = sig["high"].to_numpy()
    low = sig["low"].to_numpy()
    close = sig["close"].to_numpy()
    atr_pct = indicator_frame["atr_pct"].to_numpy()
    ret3_z_all = indicator_frame["ret3_z"].to_numpy()
    n = len(sig)
    rows = []
    for side, col in [("bottom", "bottom_short_term_return_z"), ("top", "top_short_term_return_z")]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[(idx < n - HORIZON) & (sig["timestamp"].to_numpy()[idx] >= np.datetime64(START))]
        idx_before_dedup = len(idx)
        idx = cluster_dedup(idx, ret3_z_all[idx], most_negative=(side == "bottom"))
        log(f"  {side}: {idx_before_dedup} raw fires -> {len(idx)} after cluster-anchor dedup")
        entry = close[idx]
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + HORIZON + 1].max() for i in idx])
            pred_dir_ret = (fut_ext - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + HORIZON + 1].min() for i in idx])
            pred_dir_ret = (entry - fut_ext) / entry
        hit = pred_dir_ret >= ATR_HIT_MULT * atr_pct[idx]
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
    """Analog of taker_delta_z_climax's check: apply the SAME direction-from-ret3_z-sign rule AND
    the same MFE/ATR hit definition to EVERY bar (not just |ret3_z|>=2.5 fires) to see whether the
    extremity threshold itself adds anything over ret3_z's raw sign as an always-on directional
    signal. Vectorized forward-rolling max/min (runs over all ~280k bars)."""
    high, low, close = klines["high"], klines["low"], klines["close"]
    n = len(klines)
    fwd_high_max = high[::-1].rolling(window=HORIZON, min_periods=HORIZON).max()[::-1].shift(-1)
    fwd_low_min = low[::-1].rolling(window=HORIZON, min_periods=HORIZON).min()[::-1].shift(-1)
    mfe_up_pct = ((fwd_high_max - close) / close).to_numpy()
    mfe_down_pct = ((close - fwd_low_min) / close).to_numpy()

    ret3_z = indicator_frame["ret3_z"].to_numpy()
    atr_pct = indicator_frame["atr_pct"].to_numpy()
    ts = indicator_frame["timestamp"].to_numpy()
    valid = np.isfinite(ret3_z) & np.isfinite(atr_pct) & (ts >= np.datetime64(START)) & (np.arange(n) < n - HORIZON)
    idx = np.flatnonzero(valid & (ret3_z != 0))
    side_is_bottom = ret3_z[idx] < 0
    mfe_pct = np.where(side_is_bottom, mfe_up_pct[idx], mfe_down_pct[idx])
    hit = mfe_pct >= ATR_HIT_MULT * atr_pct[idx]
    return {"n": int(len(idx)), "all_bar_continuous_sign_hit_rate": float(hit.mean())}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading klines...")
    klines = load_klines()
    log(f"{len(klines)} bars loaded")

    log("building Tier0-style indicator frame (compute_indicators + add_creative_indicators + add_broad_indicators + extras)...")
    indicator_frame = build_indicator_frame(klines)

    log("building short_term_return_z fires + features...")
    fires = build_fires_and_features(klines, indicator_frame)
    n_before = len(fires)
    fires = fires.dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
    log(f"{len(fires)}/{n_before} usable fires after dropna "
        f"(bottom={int((fires['side']=='bottom').sum())}, top={int((fires['side']=='top').sum())})")

    log("running random-bar continuous-sign baseline check...")
    baseline = random_bar_baseline(indicator_frame, klines)
    log(f"baseline: {baseline}")
    fire_hit_rate = float(fires["hit"].mean())
    log(f"fired-signal (|ret3_z|>=2.5) hit rate: {fire_hit_rate:.4f} vs all-bar continuous-sign "
        f"baseline: {baseline['all_bar_continuous_sign_hit_rate']:.4f} "
        f"(lift {fire_hit_rate / baseline['all_bar_continuous_sign_hit_rate']:.3f}x)")

    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    holdout = fires.loc[ts >= HOLDOUT_START].reset_index(drop=True)
    log(f"TRAIN(<{VAL_START.date()}) n={len(train)}, VAL n={len(val)}, OOS n={len(oos)}, HOLDOUT(>={HOLDOUT_START.date()}) n={len(holdout)}")

    fires.to_csv(OUT_DIR / "eth_5m_short_term_return_z_metalabel_features.csv", index=False)

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
        "adopted_version": "v1",
        "status": "exploratory_single_signal_below_promotion_bar",
        "summary_for_future_sessions": (
            "v1 (this run, ADOPTED after visual review): HORIZON=12(1h), hit = touched "
            "(intrabar MFE_pct over bars[fire+1:fire+13] >= 1.75*atr_pct_at_fire), cluster-anchored "
            "(same-side bursts within 3 bars collapsed to most-extreme-ret3_z bar). Chosen over "
            "taker_delta_z_climax's 2h window because phase-1 diagnostics found the OPPOSITE timing "
            "behavior: 88-89% of fires have their true local extreme AT OR BEFORE the fire bar "
            "(median lag -75min), and sign-only hit rate peaks at 15min-1h then DECAYS by 2h/4h "
            "(top side drops below the naive floor at 4h) -- widening the window would sit past "
            "this signal's own decay point, not before it. K=1.75 swept over {0.5..3.5} for a "
            "~50/50 split (51.1%/48.9%, bottom 52.2%/top 50.1%). Full methodology template: "
            "docs/experiments/eth_taker_delta_climax_metalabel_20260829.md."
        ),
        "feature_columns": FEATURE_COLUMNS,
        "n_fires_total": int(len(fires)),
        "ret3_z_crosscheck_note": "see log for compute_signals vs build_indicator_frame ret3_z corr/max_abs_diff",
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
