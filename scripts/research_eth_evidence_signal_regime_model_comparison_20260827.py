#!/usr/bin/env python3
"""Which regime MODEL do the 8 live evidence signals combine best with? Extends
research_eth_evidence_signal_regime_chop_conditional_20260827.py (GBM3-only) to compare 3 regime
sources on the identical evidence-signal-lift methodology, same evaluation window (VAL 2025-09~12 +
OOS 2026-01~02-17, the eth_5m_1year.csv coverage limit -- the evidence-signal lineage's own default
split, not the regime axis's 2026-07~08 split):

  1. gbm3        -- the OLD (now-replaced) bull/bear/chop model's own predictions, chop = idxmax=="chop".
                    Reused unchanged from the prior script (same numbers as already reported to the user).
  2. gbm2_model  -- the NEW trend/chop model's own predictions (raw argmax; the live server currently
                    serves this exact raw reading, k_bars=1 override, no serving-side smoothing).
  3. gbm2_label  -- the k_bars=12-debounced RegimeEngine ground-truth label GBM2 was TRAINED to predict
                    (not a model prediction at all -- an upper bound on what a perfect gbm2_model would
                    look like, useful to see how much of any lift difference is "model accuracy" vs
                    "the debounced label definition itself segments signals differently than GBM3's").

Same caveat as the parent script applies to all three: the 2025-09~2026-02 window sits inside both
regime models' TRAIN range (2024-01-01~2026-06-30), so classification there is in-sample. That
caveat is now IDENTICAL across all 3 columns (doesn't advantage one over another), so it does not
distort this specific comparison even though it still means no column here is OOS-clean.
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

from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import OOS_END, load_zigzag_pivots  # noqa: E402
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import OOS_START, VAL_END, VAL_START  # noqa: E402
from features.elite import RegimeEngine  # noqa: E402
from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER  # noqa: E402
from research_eth_evidence_signal_regime_chop_conditional_20260827 import (  # noqa: E402
    build_evidence_frame,
    build_regime_frame as build_regime_frame_gbm3,
    part_a_offense,
)
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402
from train_eth_regime_gbm2_trend_chop_20260827 import _debounce  # noqa: E402

REGIME_TRAIN_PATHS = [
    ROOT / "data" / "splits" / "year_oos" / "training_features_2025.csv",
    ROOT / "data" / "splits" / "year_oos" / "training_features_2026_rebuilt.csv",
]
GBM2_MODEL_PATH = ROOT / "tmp" / "eth_regime_gbm2_trend_chop_20260827" / "model.joblib"


def _load_raw_2025_2026() -> pd.DataFrame:
    frames = [pd.read_csv(p, parse_dates=["timestamp"]) for p in REGIME_TRAIN_PATHS]
    raw = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return _with_raw_state12(raw)


def build_regime_frame_gbm2_model() -> pd.DataFrame:
    """GBM2's own live predictions (raw argmax -- matches the currently-deployed k_bars=1 config)."""
    feats = _load_raw_2025_2026()
    payload = joblib.load(GBM2_MODEL_PATH)
    cols = payload["feature_cols"]
    med = pd.Series(payload["feature_medians"])
    for c in cols:
        if c not in feats.columns:
            feats[c] = med.get(c, 0.0)
    x = feats[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    proba = payload["model"].predict_proba(x)
    classes = list(payload["classes"])  # ["chop", "trend"]
    out = pd.DataFrame({"timestamp": feats["timestamp"].reset_index(drop=True)})
    for i, name in enumerate(classes):
        out[f"{name}_prob"] = proba[:, i]
    out["regime_label"] = np.where(out["trend_prob"] >= out["chop_prob"], "trend", "chop")
    return out


def build_regime_frame_gbm2_label() -> pd.DataFrame:
    """The k_bars=12-debounced RegimeEngine ground truth GBM2 was trained to predict -- not a model
    output, an upper bound for comparison."""
    feats = _load_raw_2025_2026()
    labeled = RegimeEngine().compute(feats.copy())
    is_trend_raw = ((labeled["regime_bull"] + labeled["regime_bear"]) > 0).astype(int).to_numpy()
    confirmed = _debounce(is_trend_raw, 12)
    out = pd.DataFrame({"timestamp": feats["timestamp"].reset_index(drop=True)})
    out["regime_label"] = np.where(confirmed == 1, "trend", "chop")
    return out


def masks_for(regime: pd.DataFrame, sig: pd.DataFrame, chop_value: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    frame = sig.merge(regime[["timestamp", "regime_label"]], on="timestamp", how="inner")
    ts = frame["timestamp"]
    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    chop_mask = (frame["regime_label"] == chop_value).to_numpy() & window_mask
    nonchop_mask = (frame["regime_label"] != chop_value).to_numpy() & window_mask
    return frame, window_mask, chop_mask, nonchop_mask


def main() -> None:
    print("Building evidence-signal frame (shared across all 3 regime sources)...")
    sig = build_evidence_frame()
    pivots = load_zigzag_pivots()

    sources = {
        "gbm3": (build_regime_frame_gbm3(), "chop"),
        "gbm2_model": (build_regime_frame_gbm2_model(), "chop"),
        "gbm2_label": (build_regime_frame_gbm2_label(), "chop"),
    }

    results = {}
    for src_name, (regime, chop_value) in sources.items():
        frame, window_mask, chop_mask, nonchop_mask = masks_for(regime, sig, chop_value)
        n_chop, n_nonchop = int(chop_mask.sum()), int(nonchop_mask.sum())
        print(f"\n[{src_name}] window bars={int(window_mask.sum())}  chop={n_chop} ({n_chop/max(window_mask.sum(),1)*100:.1f}%)  non_chop={n_nonchop}")
        a = part_a_offense(frame, pivots, window_mask, chop_mask, nonchop_mask)
        a["regime_source"] = src_name
        results[src_name] = a

    combined = pd.concat(results.values(), ignore_index=True)
    out_dir = ROOT / "tmp" / "eth_evidence_signal_regime_model_comparison_20260827"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "lift_by_regime_source.csv", index=False)

    # summary: chop-lift and the lift-gap (chop - non_chop) for each signal x side x regime source,
    # restricted to the 5 signals the prior GBM3-only analysis found a consistent chop lift for.
    print("\n=== Chop lift & chop-vs-non_chop gap, by regime source (bottom/top reversal signals) ===")
    focus = combined[combined["segment"].isin(["chop", "non_chop"])].copy()
    piv = focus.pivot_table(index=["signal", "side"], columns=["regime_source", "segment"], values="lift")
    piv = piv.reindex(columns=["gbm3", "gbm2_model", "gbm2_label"], level=0)
    print(piv.round(3).to_string())
    piv.to_csv(out_dir / "lift_pivot_summary.csv")

    print("\n=== Sample sizes (n_triggers in chop) by regime source ===")
    n_piv = focus[focus["segment"] == "chop"].pivot_table(index=["signal", "side"], columns="regime_source", values="n_triggers")
    n_piv = n_piv.reindex(columns=["gbm3", "gbm2_model", "gbm2_label"])
    print(n_piv.to_string())

    print(f"\nWrote {out_dir}/lift_by_regime_source.csv and lift_pivot_summary.csv")


if __name__ == "__main__":
    main()
