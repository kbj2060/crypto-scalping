"""Stage 0d — oracle_dp vs TRIPLE-BARRIER, the two PROFIT-DEFINED oracle labels  (2026-08-08)

Both labels here have a trading contract INSIDE their definition, which is what separates them from
the descriptive family (zigzag, trend-scan):
  oracle_dp        max sum(pos_t * r_t) - fee * switches, solved globally by DP
  triple-barrier   which of TP / SL / time is touched first, under vol-scaled barriers

TAUTOLOGY WARNING, STATED FIRST.  oracle_dp is the GLOBAL OPTIMUM of exactly the metric this
script evaluates (return net of fees), so at zero lag it must win by construction.  That number
carries no information.  The informative comparison is the SHAPE of the lag curve — how fast each
label's value decays once a detector is late — because today's Stage 0c measurement showed lag
tolerance, not peak return, is what separates usable labels from unusable ones on 5m bars.

FAIR-COMPARISON CHOICE, MADE EXPLICIT.  A TB label is defined per bar CONDITIONAL ON ENTERING, and
TB spans overlap, so it is not a position path on its own.  It is converted to one the way a
TB-driven trader would actually operate: walk forward, take the first directional label on a free
bar, hold for exactly that bar's `label_span_bars` (the measured bars-to-barrier-touch), then
become free again.  Sequential, non-overlapping, one position at a time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from refine_btc_regime_classifier_theta005_20260808 import PANEL_PATH  # noqa: E402
from stage0_btc_regime_label_design_20260808 import BEAR, BULL, CHOP, COST, label_family, zigzag_waves  # noqa: E402
from stage0c_btc_regime_label_oracle_dp_20260808 import LAG_GRID, oracle_dp, trade  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import contiguous_runs  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    TRAIN_END, OOS_START, OOS_END,
)

OUT_DIR = ROOT / "tmp/btc_regime_label_design_20260808"
TB_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_regimeline_20260808.parquet"
SPAN_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_label_span_20260807.parquet"


def tb_position_path(action: np.ndarray, span: np.ndarray, bull_code: int, bear_code: int) -> np.ndarray:
    """Sequential non-overlapping TB trader: first directional label on a free bar, hold its span."""
    n = len(action)
    pos = np.full(n, CHOP, dtype=np.int8)
    i = 0
    while i < n:
        a = action[i]
        if a == bull_code or a == bear_code:
            h = int(span[i]) if np.isfinite(span[i]) and span[i] >= 1 else 1
            j = min(i + h, n)
            pos[i:j] = BULL if a == bull_code else BEAR
            i = j
        else:
            i += 1
    return pos


def describe(state: np.ndarray, logret: np.ndarray, idx: np.ndarray, fee: float) -> dict:
    runs = [e - s + 1 for s, e, _ in contiguous_runs(np.where(state == BULL, 2, np.where(state == BEAR, 0, 1)))]
    lag = {str(L): trade(np.roll(state, L), logret, idx, fee) for L in LAG_GRID}
    pos_lags = [int(L) for L in lag if lag[L] > 0]
    return {"chop_occupancy": round(float((state == CHOP).mean()), 3),
            "median_run_bars": float(np.median(runs)) if runs else None,
            "n_switches_full": int(np.sum(state[1:] != state[:-1])),
            "time_in_market_pct": round(float((state != CHOP).mean()) * 100, 1),
            "lag_curve_pct": lag,
            "max_lag_bars_still_positive": max(pos_lags, default=None),
            "retention_L3_over_L0": (round(lag["3"] / lag["0"], 4)
                                     if lag["0"] not in (0,) and lag["0"] > 0 else None),
            "retention_L5_over_L0": (round(lag["5"] / lag["0"], 4)
                                     if lag["0"] not in (0,) and lag["0"] > 0 else None)}


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts, close = panel["timestamp"], panel["close"].to_numpy(dtype=np.float64)
    n = len(close)
    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    logret[0] = 0.0
    tr = np.flatnonzero((ts <= TRAIN_END).to_numpy())
    oo = np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy())

    tb = pd.read_parquet(TB_PATH)
    tb["timestamp"] = pd.to_datetime(tb["timestamp"])
    sp = pd.read_parquet(SPAN_PATH)
    sp["timestamp"] = pd.to_datetime(sp["timestamp"])
    m = panel[["timestamp"]].merge(tb, on="timestamp", how="left").merge(sp, on="timestamp", how="left")
    act = m["trade_outcome_action"].to_numpy()
    span = m["label_span_bars"].to_numpy(dtype=np.float64)

    fwd1 = np.full(n, np.nan)
    fwd1[:-12] = close[12:] / close[:-12] - 1.0
    enc = {a: round(float(np.nanmean(fwd1[act == a])) * 100, 4) for a in (0, 1, 2) if (act == a).any()}
    bull_code = max(enc, key=lambda a: enc[a])
    bear_code = min(enc, key=lambda a: enc[a])
    print(json.dumps({"tb_mean_fwd12_by_action_pct": enc,
                      "decoded": {"bull": int(bull_code), "bear": int(bear_code)},
                      "span_median_bars": float(np.nanmedian(span)),
                      "action_counts": {int(a): int((act == a).sum()) for a in (0, 1, 2)}}), flush=True)

    labels: dict[str, np.ndarray] = {}
    labels["tripbarrier_seq"] = tb_position_path(act, span, bull_code, bear_code)
    always = np.full(n, CHOP, dtype=np.int8)
    always[act == bull_code] = BULL
    always[act == bear_code] = BEAR
    labels["tripbarrier_perbar"] = always      # not tradeable (overlapping); shown for reference
    for k in (1.0, 2.0, 4.0, 8.0):
        labels[f"oracle_dp|fee{k:g}x"] = oracle_dp(logret, COST * k)
    zz = label_family(close, {th: zigzag_waves(close, th) for th in (0.005, 0.010, 0.020)})
    labels["zigzag_pure|th0.5"] = zz["zigzag_pure|th0.5"]

    rows = {}
    for name, st in labels.items():
        r = describe(st, logret, oo, COST)
        r["perfect_return_train_pct"] = trade(st, logret, tr, COST)
        rows[name] = r
        lc = r["lag_curve_pct"]
        print(f"  {name:22} inMkt {r['time_in_market_pct']:5.1f}%  run {str(r['median_run_bars']):>6}  "
              f"sw {r['n_switches_full']:6d}  L0 {lc['0']:>12,.1f}  L3 {lc['3']:>11,.1f}  "
              f"L5 {lc['5']:>11,.1f}  L12 {lc['12']:>10,.1f}  maxlag+ {r['max_lag_bars_still_positive']}",
              flush=True)

    out = {"contract": "docs/experiments/btc_regime_label_design_bullbearchop_20260808.json",
           "scope": "both labels here are PROFIT-DEFINED (a trading contract sits inside the "
                    "definition), unlike the descriptive family (zigzag, trend-scan)",
           "tautology_warning": "oracle_dp is the global optimum of the very metric evaluated here, "
                                "so its L=0 win is by construction and carries no information; the "
                                "lag-curve SHAPE is the comparison that matters",
           "tb_conversion": "sequential non-overlapping: first directional label on a free bar, held "
                            "for that bar's measured label_span_bars",
           "eval_fee_pct": round(COST * 100, 3), "labels": rows}
    (OUT_DIR / "stage0d_oracle_dp_vs_tb.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT_DIR / 'stage0d_oracle_dp_vs_tb.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
