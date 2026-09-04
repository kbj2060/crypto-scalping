#!/usr/bin/env python3
"""Does the live bull/bear/chop regime classifier (scripts/live_regime_gbm3_signal_20260826.py,
OOS bal_acc 0.9189, deployed to the dashboard 2026-08-26) say anything useful about the 8 live
evidence signals (scripts/live_evidence_signal_dashboard_20260823.py), specifically for chop?

User asked "chop 장에서 수익 낼 방법 없을까" (2026-08-27) after being told the 3 prior offense-angle
attempts at chop/range alpha in this repo all failed structurally (MM spread capture: fee arithmetic
underwater before adverse selection; discretionary S/R reversal rulebook: net -63% over 4.7y;
reversal evidence-signal automation: 4/4 lost to always_long/always_short). Told to research using
ONLY dashboard-visible indicators/signals -- this script tests two such angles:

  (A) OFFENSE: does any of the 8 evidence signals' 1h lift-vs-zigzag-pivot improve specifically
      when regime=chop vs bull/bear? (Would justify a chop-gated version of an existing signal.)
  (B) DEFENSE: is chop actually where simple trend continuation fails (supporting "reduce exposure
      when chop_prob is high" as a zero-new-alpha risk overlay, instead of searching for new alpha)?
      Measured as: hit-rate of a 12-bar-forward return matching the sign of the prior 12-bar return,
      by regime.

Methodology, reused verbatim from established scripts, not re-derived:
  - Regime probabilities: _with_raw_state12() (scripts/retrain_clean_regime_hmm_raw_state12_20260517.py)
    applied directly to the canonical data/splits/year_oos/training_features_{2025,2026_rebuilt}.csv
    (same feature derivation the live dashboard signal uses, just run once over history instead of a
    live Binance fetch -- these CSVs already carry every raw ingredient _with_raw_state7/12 need).
  - Evidence signals: live_evidence_signal_dashboard_20260823.compute_signals(), the EXACT currently-
    deployed formula (including the 2026-08-27 orthogonal_combo/funding_oscillator_combo merge),
    computed on data/eth_5m_1year.csv -- the same file the whole evidence-signal lift lineage
    (3.51x etc.) was measured on, so segment-vs-overall lift stays comparable to known baselines.
  - Lift: event_study()/load_zigzag_pivots() (analyze_eth_confluence_oscillator_bottom_top_evidence_
    20260814.py), same VAL+OOS window as every sibling script in this lineage.

CAVEAT (disclosed, not hidden): 2025-09~2026-02 (the VAL+OOS window) is INSIDE the regime model's own
TRAIN range (2024-01-01~2026-06-30) -- its bull/bear/chop split there is in-sample and may be
optimistic about classification accuracy. This does not leak into the evidence-signal lift numbers
themselves (a separate, independently-computed quantity per segment), but the chop/non-chop split
boundary itself should be read as "best available", not OOS-clean. This is retrospective diagnostic
research (event_study lift + a directional hit-rate stat), NOT a fresh-forward cost-gate backtest --
no promotion or live-signal claim is made either way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
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
from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_eth_funding_crossasset_combo_signal_20260825 import load_funding_z  # noqa: E402
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402

ETH_EVIDENCE_PATH = ROOT / "data" / "eth_5m_1year.csv"
BTC_PATH = ROOT / "data" / "btc_5m_1year.csv"
REGIME_TRAIN_PATHS = [
    ROOT / "data" / "splits" / "year_oos" / "training_features_2025.csv",
    ROOT / "data" / "splits" / "year_oos" / "training_features_2026_rebuilt.csv",
]
REGIME_MODEL_PATH = ROOT / "tmp" / "eth_regime_gbm3_independent_20260826" / "model.joblib"


def build_regime_frame() -> pd.DataFrame:
    frames = [pd.read_csv(p, parse_dates=["timestamp"]) for p in REGIME_TRAIN_PATHS]
    raw = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    feats = _with_raw_state12(raw)

    payload = joblib.load(REGIME_MODEL_PATH)
    cols = payload["feature_cols"]
    med = pd.Series(payload["feature_medians"])
    for c in cols:
        if c not in feats.columns:
            feats[c] = med.get(c, 0.0)
    x = feats[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    proba = payload["model"].predict_proba(x)
    classes = list(payload["classes"])

    out = pd.DataFrame({"timestamp": feats["timestamp"].reset_index(drop=True)})
    for i, name in enumerate(classes):
        out[f"{name}_prob"] = proba[:, i]
    prob_cols = [f"{c}_prob" for c in classes]
    out["regime_label"] = out[prob_cols].idxmax(axis=1).str.replace("_prob", "", regex=False)
    return out


def build_evidence_frame() -> pd.DataFrame:
    raw = pd.read_csv(ETH_EVIDENCE_PATH, parse_dates=["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    btc = pd.read_csv(BTC_PATH, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    funding = load_funding_z()
    return compute_signals(raw, btc_df=btc, funding_df=funding)


def part_a_offense(frame: pd.DataFrame, pivots: pd.DataFrame, window_mask: np.ndarray, chop_mask: np.ndarray, nonchop_mask: np.ndarray) -> pd.DataFrame:
    rows = []
    segments = {
        "chop": (chop_mask, np.flatnonzero(chop_mask)),
        "non_chop": (nonchop_mask, np.flatnonzero(nonchop_mask)),
        "overall": (window_mask, np.flatnonzero(window_mask)),
    }
    for name, _desc in SIGNAL_ORDER:
        for side in ("bottom", "top"):
            col = f"{side}_{name}"
            side_pivots = pivots.loc[pivots["pivot_type"] == side]
            pivot_pos = frame.index[frame["timestamp"].isin(side_pivots["timestamp"])].to_numpy()
            sig_bool = frame[col].fillna(False).to_numpy()
            for seg_name, (seg_mask, seg_all_pos) in segments.items():
                trigger_pos = np.flatnonzero(sig_bool & seg_mask)
                stats = event_study(trigger_pos, pivot_pos, seg_all_pos, K_HORIZONS["K12_1h"])
                rows.append({"signal": name, "side": side, "segment": seg_name, **stats})
    return pd.DataFrame(rows)


def part_b_defense(frame: pd.DataFrame, window_mask: np.ndarray, chop_mask: np.ndarray, nonchop_mask: np.ndarray) -> dict:
    close = frame["close"]
    past12 = close.pct_change(12)
    fwd12 = close.shift(-12) / close - 1.0
    mom_sign = np.sign(past12)
    valid = past12.notna() & fwd12.notna() & (mom_sign != 0)

    results = {}
    for seg_name, seg_mask in (("chop", chop_mask), ("non_chop", nonchop_mask), ("overall", window_mask)):
        m = valid.to_numpy() & seg_mask
        n = int(m.sum())
        if n == 0:
            results[seg_name] = {"n": 0}
            continue
        hit = (np.sign(fwd12.to_numpy()[m]) == mom_sign.to_numpy()[m]).mean()
        trend_dir_fwd_move = (fwd12.to_numpy()[m] * mom_sign.to_numpy()[m]).mean()  # >0 = continuation on average
        fwd_abs_move = fwd12.to_numpy()[m].abs().mean() if hasattr(fwd12.to_numpy()[m], "abs") else np.abs(fwd12.to_numpy()[m]).mean()
        results[seg_name] = {
            "n": n,
            "continuation_hit_rate": float(hit),
            "mean_signed_fwd_move_pct": float(trend_dir_fwd_move * 100),
            "mean_abs_fwd_move_pct": float(fwd_abs_move * 100),
        }
    return results


def main() -> None:
    print("Building regime frame (applying live GBM3 model to canonical training_features CSVs)...")
    regime = build_regime_frame()
    print(f"  regime rows: {len(regime)}, label counts: {regime['regime_label'].value_counts().to_dict()}")

    print("Building evidence-signal frame (live compute_signals() on data/eth_5m_1year.csv)...")
    sig = build_evidence_frame()
    print(f"  evidence rows: {len(sig)}")

    frame = sig.merge(regime[["timestamp", "bull_prob", "bear_prob", "chop_prob", "regime_label"]], on="timestamp", how="inner")
    print(f"  merged rows: {len(frame)} (sig={len(sig)}, regime={len(regime)})")

    pivots = load_zigzag_pivots()
    ts = frame["timestamp"]
    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    chop_mask = (frame["regime_label"] == "chop").to_numpy() & window_mask
    nonchop_mask = (frame["regime_label"] != "chop").to_numpy() & window_mask
    print(f"Window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}")
    print(f"  window bars: {int(window_mask.sum())}, chop: {int(chop_mask.sum())} "
          f"({chop_mask.sum() / max(window_mask.sum(), 1) * 100:.1f}%), non_chop: {int(nonchop_mask.sum())}")

    print("\n=== PART A (OFFENSE): evidence-signal 1h lift by regime segment ===")
    a = part_a_offense(frame, pivots, window_mask, chop_mask, nonchop_mask)
    pd.set_option("display.width", 200)
    piv = a.pivot_table(index=["signal", "side"], columns="segment", values=["n_triggers", "precision", "lift"])
    piv = piv.reindex(columns=["n_triggers", "precision", "lift"], level=0)
    print(piv.round(3).to_string())

    out_dir = ROOT / "tmp" / "eth_evidence_signal_regime_chop_conditional_20260827"
    out_dir.mkdir(parents=True, exist_ok=True)
    a.to_csv(out_dir / "part_a_offense_lift_by_regime.csv", index=False)

    print("\n=== PART B (DEFENSE): 12-bar momentum continuation by regime ===")
    b = part_b_defense(frame, window_mask, chop_mask, nonchop_mask)
    for seg, stats in b.items():
        print(f"  {seg}: {stats}")
    pd.DataFrame(b).T.to_csv(out_dir / "part_b_defense_momentum_by_regime.csv")

    print(f"\nWrote {out_dir}/part_a_offense_lift_by_regime.csv and part_b_defense_momentum_by_regime.csv")


if __name__ == "__main__":
    main()
