"""Stage 0c — COST-OPTIMAL oracle labelling + a robustness curve  (2026-08-08)

Contract: docs/experiments/btc_regime_label_design_bullbearchop_20260808.json

LITERATURE BASIS.  Kovacevic, Mercep, Begusic, Kostanjcar, "Optimal Trend Labeling in Financial
Time Series", IEEE Access 11:83822-83832 (2023).  Two things there beat what Stage 0/0b built:

  (1) ORACLE LABELLING — the segmentation that MAXIMISES CUMULATIVE RETURN NET OF TRANSACTION
      FEES.  Stage 0's `zigzag_net` was a heuristic approximation of this criterion (greedy zigzag
      + a post-hoc cost filter); this is the exact global optimum of the same criterion, by DP.
  (2) A NOISE MODEL that evaluates a LABEL's robustness WITHOUT TRAINING A CLASSIFIER — simulate a
      detector at accuracy p, trade the corrupted labels, and see how much return survives.  The
      paper reports oracle labelling as the most robust on this measure.

Why (2) matters here specifically: Stage 0's separation gate (G2) is partly CIRCULAR for a
retrospective label — a bar labelled bull sits on a leg that ends higher, so a positive forward
return is near-mechanical, and Stage 0b showed the most circular label (trend-scan, whose horizons
overlap the forward metric) posted the biggest spread while failing the non-circular gate.  A
robustness curve is not circular: it asks how much money survives IMPERFECT detection, which is
the question that actually decides whether a label is worth having.

THE HEADLINE NUMBER: BREAK-EVEN ACCURACY — the detection accuracy at which trading the label stops
losing money.  Lower is better.  Our frozen detectors achieve ~68%, so a label needing 80% is
useless to us however elegant it is.

Two corruption models, because they answer different questions:
  iid   flip a random (1-p) fraction of bars — the paper's model; errors are uncorrelated
  lag   shift the label forward by L bars — errors are CORRELATED and systematic, which is what
        our detectors actually do (measured detection lag 2-5 bars).  For a lagging detector this
        is the more honest degradation, so both are reported.
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
from stage0_btc_regime_label_design_20260808 import (  # noqa: E402
    BEAR, BULL, CHOP, COST, FWD_H, label_family, separation, zigzag_waves,
)
from stage0b_btc_regime_label_design_trendscan_20260808 import gross_net_by_state  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import contiguous_runs  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    TRAIN_END, OOS_START, OOS_END,
)

OUT_DIR = ROOT / "tmp/btc_regime_label_design_20260808"
TS_PATH = ROOT / "data/splits/year_oos/btc_5m_trendscan_oracle_labels_20260806.parquet"
FEE_MULTS = [0.5, 1.0, 2.0, 4.0]          # switch fee as a multiple of the round-trip cost
ACC_GRID = [0.55, 0.60, 0.65, 0.68, 0.70, 0.75, 0.80, 0.90, 1.00]
LAG_GRID = [0, 2, 3, 5, 8, 12]
N_REP = 20
POS = {BULL: 1.0, CHOP: 0.0, BEAR: -1.0}


def oracle_dp(logret: np.ndarray, fee: float) -> np.ndarray:
    """Globally optimal {long, flat, short} labelling maximising sum(position*r) - fee*switches.

    Exact DP: V_t(s) = pos(s)*r_t + max_{s'} [ V_{t-1}(s') - fee*(s' != s) ].  O(n*9), then
    backtrack.  This is the criterion Stage 0's zigzag_net approximated greedily.
    """
    n = len(logret)
    states = (BEAR, CHOP, BULL)
    V = np.zeros(3)
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(n):
        prev = V
        newV = np.empty(3)
        for i, s in enumerate(states):
            cand = prev - fee * np.array([0.0 if states[j] == s else 1.0 for j in range(3)])
            j = int(np.argmax(cand))
            back[t, i] = j
            newV[i] = cand[j] + POS[s] * logret[t]
        V = newV
    out = np.empty(n, dtype=np.int8)
    i = int(np.argmax(V))
    for t in range(n - 1, -1, -1):
        out[t] = states[i]
        i = int(back[t, i])
    return out


def trade(state: np.ndarray, logret: np.ndarray, idx: np.ndarray, fee: float) -> float:
    s = state[idx]
    pos = np.array([POS[int(v)] for v in s])
    gross = float(np.sum(pos * logret[idx]))
    switches = int(np.sum(s[1:] != s[:-1]))
    return round((np.exp(gross - fee * switches) - 1.0) * 100, 2)


def robustness(state: np.ndarray, logret: np.ndarray, idx: np.ndarray, fee: float,
               rng: np.random.Generator) -> dict:
    """Return vs detector quality, under iid-accuracy corruption and under pure lag."""
    others = {BULL: [CHOP, BEAR], CHOP: [BULL, BEAR], BEAR: [BULL, CHOP]}
    iid = {}
    for p in ACC_GRID:
        if p >= 1.0:
            iid[f"{p:.2f}"] = trade(state, logret, idx, fee)
            continue
        vals = []
        for _ in range(N_REP):
            c = state.copy()
            flip = rng.random(len(c)) > p
            for st, alts in others.items():
                m = flip & (c == st)
                if m.any():
                    c[m] = rng.choice(alts, size=int(m.sum()))
            vals.append(trade(c, logret, idx, fee))
        iid[f"{p:.2f}"] = round(float(np.mean(vals)), 2)
    lag = {str(L): trade(np.roll(state, L), logret, idx, fee) for L in LAG_GRID}

    be = None
    ks = sorted(float(k) for k in iid)
    for a, b in zip(ks, ks[1:]):
        if iid[f"{a:.2f}"] <= 0 < iid[f"{b:.2f}"]:
            be = round(a + (b - a) * (0 - iid[f"{a:.2f}"]) / (iid[f"{b:.2f}"] - iid[f"{a:.2f}"]), 3)
            break
    if be is None and iid[f"{ks[0]:.2f}"] > 0:
        be = ks[0]
    return {"iid_accuracy_curve_pct": iid, "lag_curve_pct": lag, "break_even_accuracy": be,
            "max_lag_bars_still_positive": max([int(L) for L in lag if lag[L] > 0], default=None)}


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
    fwd = np.full(n, np.nan)
    fwd[:-FWD_H] = close[FWD_H:] / close[:-FWD_H] - 1.0

    labels: dict[str, np.ndarray] = {}
    for k in FEE_MULTS:
        f = COST * k
        labels[f"oracle_dp|fee{k:g}x"] = oracle_dp(logret, f)
        print(json.dumps({f"oracle_dp|fee{k:g}x": "built", "fee_pct": round(f * 100, 3)}), flush=True)

    wave_cache = {th: zigzag_waves(close, th) for th in (0.005, 0.010, 0.020)}
    zz = label_family(close, wave_cache)
    for k in ("zigzag_pure|th0.5", "zigzag_net|th0.5|m0", "zigzag_net|th2|m0"):
        labels[k] = zz[k]
    tsl = pd.read_parquet(TS_PATH)
    tsl["timestamp"] = pd.to_datetime(tsl["timestamp"])
    m = panel[["timestamp"]].merge(tsl, on="timestamp", how="left")
    act = m["trendscan_action"].to_numpy()
    raw = np.full(n, CHOP, dtype=np.int8)
    raw[act == 1] = BULL
    raw[act == 2] = BEAR
    labels["trendscan_raw"] = raw

    rng = np.random.default_rng(20260808)
    rows: dict[str, dict] = {}
    fee_eval = COST
    for name, st in labels.items():
        runs = [e - s + 1 for s, e, _ in contiguous_runs(np.where(st == BULL, 2, np.where(st == BEAR, 0, 1)))]
        rec = {
            "chop_occupancy": round(float((st == CHOP).mean()), 3),
            "median_run_bars": float(np.median(runs)) if runs else None,
            "n_switches": int(np.sum(st[1:] != st[:-1])),
            "G1_common_basis_move_minus_cost_pct": gross_net_by_state(close, st),
            "G2_oos_spread_pct": separation(fwd, st, oo)["spread_bull_minus_bear_pct"],
            "perfect_return_train_pct": trade(st, logret, tr, fee_eval),
            "perfect_return_oos_pct": trade(st, logret, oo, fee_eval),
            "robustness_oos": robustness(st, logret, oo, fee_eval, rng),
        }
        rows[name] = rec
        r = rec["robustness_oos"]
        print(f"  {name:24} chop {rec['chop_occupancy']:5.2f} run {str(rec['median_run_bars']):>6} "
              f"sw {rec['n_switches']:6d}  perfectOOS {rec['perfect_return_oos_pct']:>10}%  "
              f"@0.68 {r['iid_accuracy_curve_pct']['0.68']:>9}%  BE-acc {r['break_even_accuracy']}  "
              f"maxlag+ {r['max_lag_bars_still_positive']}", flush=True)

    usable = {k: v for k, v in rows.items()
              if v["robustness_oos"]["break_even_accuracy"] is not None
              and v["robustness_oos"]["break_even_accuracy"] <= 0.68}
    out = {"contract": "docs/experiments/btc_regime_label_design_bullbearchop_20260808.json",
           "literature": "Kovacevic et al., Optimal Trend Labeling in Financial Time Series, "
                         "IEEE Access 11:83822-83832 (2023) — oracle labelling + a noise model for "
                         "label robustness without training a classifier",
           "note": "break-even accuracy is the headline: the detection accuracy at which trading the "
                   "label stops losing money. Our frozen detectors reach ~68%, so labels needing more "
                   "are unusable to us regardless of any other property.",
           "eval_fee_pct": round(fee_eval * 100, 3), "n_rep": N_REP,
           "labels": rows,
           "usable_at_achievable_accuracy": sorted(usable, key=lambda k: rows[k]["robustness_oos"]["break_even_accuracy"])}
    (OUT_DIR / "stage0c_oracle_dp.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"usable_at_<=0.68_accuracy": out["usable_at_achievable_accuracy"]}, indent=2), flush=True)
    print(f"wrote {OUT_DIR / 'stage0c_oracle_dp.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
