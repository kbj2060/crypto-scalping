#!/usr/bin/env python3
"""Does an evidence signal's 1h reversal lift improve when it fires WITH a nearby liquidation-map
support/resistance level on the confirming side (bottom signal + support below price; top signal +
resistance above price), vs firing with no such level nearby? User request 2026-08-27, prompted by
a 3-day chart artifact overlaying regime + evidence signals (S/B markers) + liquidation spliced
levels (support/resistance lines) together and asking how evidence signals could be used for trade
direction.

Checked before writing this: docs/ and scripts/ have no prior "evidence signal x liquidation level"
combination study. The closed liquidation-heatmap-magnet-signal line (eth_liquidation_heatmap_
magnet_signal_scoping_20260822) tested magnet levels as a STANDALONE signal, not as a confluence
filter on evidence signals. The closed evidence-signal-injection-axis combined evidence signals with
model-INTERNAL indicators (whale_position_score/obi/toxicity_score), not liquidation levels. This is
a genuinely untested axis.

Data-availability check: compute_spliced_levels() (scripts/live_liquidation_map_20260824.py,
confirmed+live 2026-08-26) is a PURE function of a 24-hourly-bar tail window + current_price -- no
dependency on the live liquidation event feed (known historical gaps, see eth_liquidation_feed_
epoch_defect memory) or the liq_cluster history collector (deployed 2026-08-25, too young for a
multi-month backtest). LIQUIDATION_MAP_INTERVAL="1h" / .tail(LIQUIDATION_MAP_LOOKBACK_HOURS=24) in
dashboard/server.py -- 24 HOURLY bars, not 24 5-min bars. This means the exact live computation can
be reconstructed causally at every historical hour directly from data/eth_5m_1year.csv (resampled to
1h, same file the whole evidence-signal lift lineage uses), with no data blocker.

Method: same VAL(2025-09~12)+OOS(2026-01~02-17) window, event_study()/zigzag-pivot methodology,
K12_1h horizon, SIGNAL_ORDER signals -- identical to every sibling script in this lineage
(research_eth_evidence_signal_regime_chop_conditional_20260827.py etc), so numbers are directly
comparable to what's already been reported. NEW axis: has_confluence = the live compute_spliced_
levels() itself already found a same-side level within its own filters (MAX_LEVEL_DISTANCE_PCT=5%,
MIN_LEVEL_SHARE=5%) at that bar -- the live system's OWN definition of "a level exists on the
dashboard right now", not a new threshold invented for this test (avoids the post-hoc-parameter-
search trap this repo has repeatedly flagged). Levels are computed at EVERY hourly bar in the
window (not just signal-firing bars), so each segment's baseline_rate uses the correct same-
population control (bars WITH a level vs bars WITHOUT) instead of being confounded with level-
availability itself.

Causality: an hourly resampled bar labeled T (pandas resample left-label, covers [T, T+1h)) is not
actually CLOSED/available until T+1h. Levels computed from the 24-bar window ending at bar T are
therefore stamped with timestamp T+1h before being merge_asof'd (backward) onto the 5-min evidence
frame -- this exactly matches the live dashboard's own behavior (current_price there is itself the
last FULLY CLOSED hourly candle's close, confirmed by reading dashboard/server.py's fetch code, not
a more-precise live tick), not an approximation.

Diagnostic only (event_study lift), like every predecessor in this lineage -- NOT a cost-gated
backtest. lift != tradeable edge. This repo's existing chop-gated cost-gate backtest (2026-08-27,
eth_evidence_signal_chop_regime_conditional_lift memory) already found 10/10 REJECTED against the
always_long/always_short benchmark for a STRUCTURAL reason (strong-trend windows dominate that
benchmark) -- this script's job is only to check whether liquidation-level confluence is a
genuinely new, non-redundant lever on top of the already-tested signals/regime axis, not to
re-litigate that benchmark problem.
"""
from __future__ import annotations

import sys
import time
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
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START,
    VAL_END,
    VAL_START,
)
from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER  # noqa: E402
from live_liquidation_map_20260824 import compute_spliced_levels  # noqa: E402
from research_eth_evidence_signal_regime_chop_conditional_20260827 import build_evidence_frame  # noqa: E402
from research_eth_evidence_signal_regime_model_comparison_20260827 import build_regime_frame_gbm2_label  # noqa: E402

LOOKBACK_HOURLY_BARS = 24  # matches dashboard/server.py LIQUIDATION_MAP_LOOKBACK_HOURS with 1h klines
K = K_HORIZONS["K12_1h"]


def resample_1h(frame: pd.DataFrame) -> pd.DataFrame:
    """high/low/close/volume only -- everything compute_spliced_levels()/_prepare_common() reads."""
    d = frame[["timestamp", "high", "low", "close", "volume"]].set_index("timestamp")
    return d.resample("1h").agg({"high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()


def compute_hourly_levels(hourly: pd.DataFrame) -> pd.DataFrame:
    """compute_spliced_levels() at every hourly bar h using ONLY hourly[h-23:h+1] (24-bar causal
    tail). Result timestamped h+1h (when that closed candle actually becomes available to a live
    caller) so it can be merge_asof'd (backward) onto a finer-grained frame without lookahead."""
    n = len(hourly)
    close = hourly["close"].to_numpy()
    ts = hourly["timestamp"]
    rows = {
        "timestamp": [], "has_support": [], "support_distance_pct": [], "support_weight_pct": [],
        "has_resistance": [], "resistance_distance_pct": [], "resistance_weight_pct": [],
    }
    t0 = time.time()
    for h in range(LOOKBACK_HOURLY_BARS - 1, n):
        window = hourly.iloc[h - LOOKBACK_HOURLY_BARS + 1: h + 1]
        levels = compute_spliced_levels(window, float(close[h]))
        sup = levels.get("support_levels") or []
        res = levels.get("resistance_levels") or []
        rows["timestamp"].append(ts.iloc[h] + pd.Timedelta(hours=1))
        rows["has_support"].append(bool(sup))
        rows["support_distance_pct"].append(sup[0]["distance_pct"] if sup else np.nan)
        rows["support_weight_pct"].append(sup[0]["weight_pct"] if sup else np.nan)
        rows["has_resistance"].append(bool(res))
        rows["resistance_distance_pct"].append(res[0]["distance_pct"] if res else np.nan)
        rows["resistance_weight_pct"].append(res[0]["weight_pct"] if res else np.nan)
        if (h + 1) % 1000 == 0:
            elapsed = time.time() - t0
            done = h + 1 - (LOOKBACK_HOURLY_BARS - 1)
            total = n - (LOOKBACK_HOURLY_BARS - 1)
            print(f"  ...{done}/{total} hourly bars ({elapsed:.0f}s elapsed)")
    return pd.DataFrame(rows)


def tertile_labels(distance_abs: pd.Series, window_mask: np.ndarray) -> pd.Series:
    """near/mid/far tertiles of |distance_pct| to the nearest same-side level, cutpoints fit on the
    in-window population only. NOT a hand-picked threshold: qcut splits at the data's own 33rd/67th
    percentiles, avoiding the post-hoc "search for the threshold that works" trap. Bars are always
    assigned (compute_spliced_levels' own MAX_LEVEL_DISTANCE_PCT=5% filter is nearly always non-
    empty in this dataset -- see run log: has_support/has_resistance = 100.0% in-window -- so a
    same-side level virtually always exists; the informative question is how CLOSE it is, hence
    tertiles instead of a binary has-a-level/doesn't split, which is degenerate here)."""
    fit_values = distance_abs[window_mask]
    _, edges = pd.qcut(fit_values, 3, retbins=True, duplicates="drop")
    return pd.cut(distance_abs, bins=edges, labels=["near", "mid", "far"][: len(edges) - 1], include_lowest=True)


def run_segments(frame: pd.DataFrame, pivots: pd.DataFrame, segments: dict) -> pd.DataFrame:
    """segments: {seg_name: (side -> bool_mask)} where side in ('bottom','top'); a seg_name maps to
    a dict of side->mask so bottom/top can use different confluence masks (support vs resistance)."""
    rows = []
    for name, _desc in SIGNAL_ORDER:
        for side in ("bottom", "top"):
            col = f"{side}_{name}"
            side_pivots = pivots.loc[pivots["pivot_type"] == side]
            pivot_pos = frame.index[frame["timestamp"].isin(side_pivots["timestamp"])].to_numpy()
            sig_bool = frame[col].fillna(False).to_numpy()
            for seg_name, side_masks in segments.items():
                seg_mask = side_masks[side]
                trigger_pos = np.flatnonzero(sig_bool & seg_mask)
                seg_all_pos = np.flatnonzero(seg_mask)
                stats = event_study(trigger_pos, pivot_pos, seg_all_pos, K)
                rows.append({"signal": name, "side": side, "segment": seg_name, **stats})
    return pd.DataFrame(rows)


def main() -> None:
    print("Building evidence-signal frame (data/eth_5m_1year.csv)...")
    frame = build_evidence_frame()
    print(f"  rows: {len(frame)}")

    ts = frame["timestamp"]
    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    print(f"Window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars")

    print("Resampling to 1h for the liquidation-level computation (24-hourly-bar causal lookback, "
          "matches dashboard/server.py LIQUIDATION_MAP_INTERVAL='1h' x LIQUIDATION_MAP_LOOKBACK_HOURS=24)...")
    hourly = resample_1h(frame)
    print(f"  hourly bars: {len(hourly)}")

    print("Computing compute_spliced_levels() at every hourly bar (causal, no lookahead)...")
    levels_hourly = compute_hourly_levels(hourly)

    frame_sorted = frame.sort_values("timestamp").reset_index(drop=True)
    merged = pd.merge_asof(frame_sorted, levels_hourly, on="timestamp", direction="backward")
    # re-align window_mask to the (already-sorted, should be a no-op) merged frame order
    ts2 = merged["timestamp"]
    window_mask = (((ts2 >= VAL_START) & (ts2 <= VAL_END)) | ((ts2 >= OOS_START) & (ts2 <= OOS_END))).to_numpy()

    has_support = merged["has_support"].fillna(False).to_numpy()
    has_resistance = merged["has_resistance"].fillna(False).to_numpy()
    print(f"  in-window has_support: {has_support[window_mask].mean()*100:.1f}%  "
          f"has_resistance: {has_resistance[window_mask].mean()*100:.1f}%  "
          f"(near-100% expected -- a same-side level almost always exists within the live system's "
          f"own 5% band; distance/strength tertiles below are the informative split)")

    support_dist_abs = merged["support_distance_pct"].abs()
    resistance_dist_abs = merged["resistance_distance_pct"].abs()
    support_tertile = tertile_labels(support_dist_abs, window_mask)
    resistance_tertile = tertile_labels(resistance_dist_abs, window_mask)
    print(f"  support distance tertile edges (|%|): "
          f"{pd.qcut(support_dist_abs[window_mask], 3, duplicates='drop').cat.categories}")
    print(f"  resistance distance tertile edges (|%|): "
          f"{pd.qcut(resistance_dist_abs[window_mask], 3, duplicates='drop').cat.categories}")

    pivots = load_zigzag_pivots()

    print("\n=== PART A: 1h lift by nearest-level distance tertile, near=closest (all regimes) ===")
    segments_a = {
        "near": {"bottom": window_mask & (support_tertile == "near").to_numpy(),
                 "top": window_mask & (resistance_tertile == "near").to_numpy()},
        "mid": {"bottom": window_mask & (support_tertile == "mid").to_numpy(),
                "top": window_mask & (resistance_tertile == "mid").to_numpy()},
        "far": {"bottom": window_mask & (support_tertile == "far").to_numpy(),
                "top": window_mask & (resistance_tertile == "far").to_numpy()},
        "overall": {"bottom": window_mask, "top": window_mask},
    }
    a = run_segments(merged, pivots, segments_a)
    pd.set_option("display.width", 200)
    piv_a = a.pivot_table(index=["signal", "side"], columns="segment", values=["n_triggers", "precision", "lift"])
    piv_a = piv_a.reindex(columns=["n_triggers", "precision", "lift"], level=0)
    piv_a = piv_a.reindex(columns=["near", "mid", "far", "overall"], level=1)
    print(piv_a.round(3).to_string())

    out_dir = ROOT / "tmp" / "eth_evidence_signal_liquidation_confluence_20260827"
    out_dir.mkdir(parents=True, exist_ok=True)
    a.to_csv(out_dir / "lift_by_confluence_all_regimes.csv", index=False)
    levels_hourly.to_csv(out_dir / "hourly_levels.csv", index=False)

    print("\nBuilding GBM2-label regime frame for a secondary chop-only cut "
          "(most-reliable regime source per today's 3-source cross-validation)...")
    regime = build_regime_frame_gbm2_label()
    rframe = merged.merge(regime[["timestamp", "regime_label"]], on="timestamp", how="left")
    chop_mask = (rframe["regime_label"] == "chop").to_numpy() & window_mask
    print(f"  chop bars in window: {int(chop_mask.sum())} ({chop_mask.sum() / max(window_mask.sum(), 1) * 100:.1f}%)")

    print("\n=== PART B: 1h lift by nearest-level distance tertile, chop-only ===")
    segments_b = {
        "chop_near": {"bottom": chop_mask & (support_tertile == "near").to_numpy(),
                      "top": chop_mask & (resistance_tertile == "near").to_numpy()},
        "chop_mid": {"bottom": chop_mask & (support_tertile == "mid").to_numpy(),
                     "top": chop_mask & (resistance_tertile == "mid").to_numpy()},
        "chop_far": {"bottom": chop_mask & (support_tertile == "far").to_numpy(),
                     "top": chop_mask & (resistance_tertile == "far").to_numpy()},
        "chop_overall": {"bottom": chop_mask, "top": chop_mask},
    }
    b = run_segments(merged, pivots, segments_b)
    piv_b = b.pivot_table(index=["signal", "side"], columns="segment", values=["n_triggers", "precision", "lift"])
    piv_b = piv_b.reindex(columns=["n_triggers", "precision", "lift"], level=0)
    piv_b = piv_b.reindex(columns=["chop_near", "chop_mid", "chop_far", "chop_overall"], level=1)
    print(piv_b.round(3).to_string())
    b.to_csv(out_dir / "lift_by_confluence_chop_only.csv", index=False)

    print(f"\nWrote outputs to {out_dir}/")


if __name__ == "__main__":
    main()
