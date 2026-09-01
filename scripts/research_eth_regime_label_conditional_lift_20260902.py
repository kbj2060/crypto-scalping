#!/usr/bin/env python3
"""Phase 2 of the scalping regime LABEL redesign: which label DEFINITION carries the most
evidence-signal conditioning value? -- user directive 2026-09-02 ("증거신호 조건부 lift를 더 크게
개선하는 라벨로 진행해줘"), following Phase 1
(docs/experiments/eth_regime_scalping_label_geometry_20260902.md) which closed the
"faster transitions" framing: no scale/debounce combination showed a significant transition edge,
and shrinking the scale made it worse.

REFRAMED OBJECTIVE

  Not "does the regime predict direction" (Phase 1: no, at any scale) but "does the regime say WHEN
  the evidence signals work better". The deployed label demonstrably does -- the 2026-08-27 study
  found chop-conditional lift improvements of +29~37% on the bottom-reversal family. This script
  asks whether a different label SCALE does it better.

  Labels are computed as GROUND TRUTH, not as model predictions. That is deliberate and is the
  right order of operations: if a label definition does not carry conditioning value in the first
  place, no classifier trained on it can create any. (The 2026-08-27 baseline used the GBM3 model's
  PREDICTED regime, so its +29~37% is not numerically comparable to the reference row here -- the
  reference variant below, i.e. the deployed RegimeEngine label as ground truth, is this script's
  own apples-to-apples baseline.)

GBM3 BASIS -- no GBM2 lineage import; labels are the 3-class bull/bear/chop family GBM3 itself uses,
and the reference is RAW (no debounce), GBM3's own convention. Debounce K>1 is an optional axis.

CONTROL -- CIRCULAR-SHIFT NULL, NOT RANDOM SUBSAMPLING

  Conditioning on a segment mechanically changes both the signal's precision and the segment's own
  pivot baseline, so "conditional lift > overall lift" proves nothing by itself. But a random
  scatter of bars is the WRONG null for a regime, because regimes are contiguous blocks with strong
  autocorrelation. The null here circularly SHIFTS the regime mask by a random offset (B=200),
  preserving its block structure and duty cycle exactly while destroying its alignment with price --
  the same construction this repo's microstructure panel screen used. A label variant must beat the
  95th percentile of its OWN shifted null, not merely beat 1.0.

WINDOW SPLIT

  Reported for VAL (2025-09-01~12-31) and OOS (2026-01-01~02-17) SEPARATELY as well as pooled.
  Today's composite-filter study (README ss5.15) found 7 pooled survivors that all died on exactly
  this split -- pooled-only evidence is not evidence.

⚠️ Both windows sit INSIDE the regime model's TRAIN range (2024-01-01~2026-06-30), so this study
spends NO regime-OOS budget. Same disclosure as the 2026-08-27 baseline: the regime split boundary
is in-sample, "best available" rather than OOS-clean. The evidence-signal lift numbers themselves
are an independently computed quantity per segment and are not leaked into by that.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import (  # noqa: E402
    OOS_END,
    event_study,
    load_zigzag_pivots,
)
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START,
    VAL_END,
    VAL_START,
)
from features.elite import RegimeEngine  # noqa: E402
from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_eth_funding_crossasset_combo_signal_20260825 import load_funding_z  # noqa: E402
from research_eth_regime_scalping_label_geometry_20260902 import (  # noqa: E402
    _debounce,
    efficiency_ratio,
    scaled_label,
)

ETH_PATH = ROOT / "data/eth_5m_1year.csv"
BTC_PATH = ROOT / "data/btc_5m_1year.csv"
K_HORIZON = 12                      # 1h -- the horizon the +29~37% baseline was measured at
SCALES = (6, 12, 24, 48)
DEBOUNCES = (1, 3, 6)               # K=12 dropped: Phase 1 showed lock-up/instability at high K
N_NULL = 200
MIN_SEG_FIRES = 25                  # below this a conditional lift is not interpretable
RNG_SEED = 20260902
OUT_DIR = ROOT / "tmp/eth_regime_label_conditional_lift_20260902"


def build_evidence_frame() -> pd.DataFrame:
    raw = pd.read_csv(ETH_PATH, parse_dates=["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    btc = pd.read_csv(BTC_PATH, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    return compute_signals(raw, btc_df=btc, funding_df=load_funding_z())


def deployed_label(frame: pd.DataFrame) -> np.ndarray:
    df = frame.copy()
    df["mtf_trend_1h"] = df["close"].ewm(span=12, adjust=False).mean().pct_change().fillna(0.0)
    lab = RegimeEngine().compute(df)
    y = np.full(len(df), 2, dtype=int)
    y[lab["regime_bull"].to_numpy() > 0] = 0
    y[lab["regime_bear"].to_numpy() > 0] = 1
    return y


def seg_lift(sig: np.ndarray, pivot_pos: np.ndarray, seg: np.ndarray) -> tuple[float, int]:
    """Lift of `sig` inside segment `seg`, with the baseline computed INSIDE the same segment."""
    all_pos = np.flatnonzero(seg)
    trig = np.flatnonzero(sig & seg)
    if len(trig) < MIN_SEG_FIRES or len(all_pos) == 0:
        return float("nan"), len(trig)
    st = event_study(trig, pivot_pos, all_pos, K_HORIZON)
    return st["lift"], len(trig)


def main() -> None:
    frame = build_evidence_frame()
    pivots = load_zigzag_pivots()
    ts = frame["timestamp"]
    close = frame["close"]

    windows = {
        "VAL": ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy(),
        "OOS": ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy(),
    }
    windows["POOLED"] = windows["VAL"] | windows["OOS"]
    print(f"frame {len(frame):,} bars | VAL {windows['VAL'].sum():,} OOS {windows['OOS'].sum():,}")

    pivot_pos = {s: frame.index[frame["timestamp"].isin(
        pivots.loc[pivots["pivot_type"] == s, "timestamp"])].to_numpy() for s in ("bottom", "top")}

    # --- label variants -------------------------------------------------------------------
    rate1 = float((efficiency_ratio(close, 24) >= 0.20).mean())
    rate2 = float((efficiency_ratio(close, 48) >= 0.16).mean())
    variants: dict[str, np.ndarray] = {"REF_deployed": deployed_label(frame)}
    for s in SCALES:
        t1 = float(efficiency_ratio(close, s).quantile(1.0 - rate1))
        t2 = float(efficiency_ratio(close, 2 * s).quantile(1.0 - rate2))
        raw_y = scaled_label(close, s, t1, t2)
        for k in DEBOUNCES:
            variants[f"S{s}_K{k}"] = raw_y if k == 1 else _debounce(raw_y, k)
    print(f"{len(variants)} label variants (incl. deployed reference)")

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for vname, y in variants.items():
        for wname, wmask in windows.items():
            chop = (y == 2) & wmask
            n_w = int(wmask.sum())
            for sname, _ in SIGNAL_ORDER:
                for side in ("bottom", "top"):
                    sig = frame[f"{side}_{sname}"].fillna(False).to_numpy()
                    l_all, n_all = seg_lift(sig, pivot_pos[side], wmask)
                    l_chop, n_chop = seg_lift(sig, pivot_pos[side], chop)
                    if not (np.isfinite(l_all) and np.isfinite(l_chop)) or l_all <= 0:
                        continue
                    improvement = l_chop / l_all - 1.0
                    # circular-shift null: same block structure & duty cycle, alignment destroyed
                    null = np.empty(N_NULL)
                    chop_in_w = (y == 2)
                    for b in range(N_NULL):
                        shifted = np.roll(chop_in_w, int(rng.integers(1, len(y)))) & wmask
                        lb, _ = seg_lift(sig, pivot_pos[side], shifted)
                        null[b] = (lb / l_all - 1.0) if np.isfinite(lb) else np.nan
                    null = null[np.isfinite(null)]
                    p95 = float(np.percentile(null, 95)) if len(null) >= 50 else float("nan")
                    rows.append({
                        "variant": vname, "window": wname, "signal": sname, "side": side,
                        "n_all": n_all, "n_chop": n_chop,
                        "lift_all": round(l_all, 3), "lift_chop": round(l_chop, 3),
                        "improvement": round(improvement, 4),
                        "null_p95": round(p95, 4) if np.isfinite(p95) else np.nan,
                        "null_pctile": round(float((null < improvement).mean() * 100), 1) if len(null) else np.nan,
                        "beats_null95": bool(np.isfinite(p95) and improvement > p95),
                    })
        print(f"  {vname}: done")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "conditional_lift.csv", index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 200)
    print("\n=== per-variant summary: mean chop-conditional lift improvement ===")
    summ = (df.groupby(["variant", "window"])
              .agg(cells=("improvement", "size"),
                   mean_improvement=("improvement", "mean"),
                   median_improvement=("improvement", "median"),
                   beats_null=("beats_null95", "sum"))
              .reset_index())
    piv = summ.pivot(index="variant", columns="window",
                     values=["mean_improvement", "beats_null", "cells"])
    print(piv.round(4).to_string())

    print("\n=== VAL/OOS agreement (the ss5.15 gate): variants positive in BOTH windows ===")
    m = summ.pivot(index="variant", columns="window", values="mean_improvement")
    both = m[(m["VAL"] > 0) & (m["OOS"] > 0)].sort_values("OOS", ascending=False)
    print(both.round(4).to_string() if len(both) else "  NONE")
    print(f"\nWrote {OUT_DIR / 'conditional_lift.csv'}")


if __name__ == "__main__":
    main()
