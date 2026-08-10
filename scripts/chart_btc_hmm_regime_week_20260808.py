"""BTC HMM regime chart, most recent week (2026-08-08, diagnostic).

Same construction as the SOL chart (scripts/chart_sol_hmm_regime_analysis_20260808.py): 3-state
sticky Gaussian HMM fit on TRAIN ONLY (<=2025-08-31; inputs standardized 24h log return + 24h
realized vol), full-series CAUSAL filtered decode with frozen params, states named bull/bear/chop
by train-mean 24h return. Plot: the panel's final 7 days at full 5m resolution, price + regime
shading + causal-HMM and D2-rule strips.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from retrain_clean_regime_hmm_20260517 import GaussianStateModel  # noqa: E402

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--panel", default="data/splits/year_oos/btc_features_2024_2026_regimeline.csv")
_ap.add_argument("--out", default="tmp/btc_regime_conditioned_20260808/btc_hmm_week.png")
_ap.add_argument("--symbol", default="BTCUSDT")
_args = _ap.parse_args()
PANEL_PATH = ROOT / _args.panel
OUT_PATH = ROOT / _args.out
SYMBOL = _args.symbol
TRAIN_END = pd.Timestamp("2025-08-31 23:59:59")
SEED = 903174
C_BULL, C_BEAR, C_CHOP = "#2563EB", "#D9542B", "#9AA0A6"
REGIME_COLORS = {"bull": C_BULL, "bear": C_BEAR, "chop": C_CHOP}
INK = "#1F2430"


def contiguous_runs(states):
    change = np.flatnonzero(np.diff(states) != 0)
    starts = np.concatenate([[0], change + 1])
    ends = np.concatenate([change, [len(states) - 1]])
    return list(zip(starts, ends, states[starts]))


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)
    logc = np.log(close)
    r288 = np.full(len(close), np.nan)
    r288[288:] = logc[288:] - logc[:-288]
    lr = np.diff(logc, prepend=logc[0])
    vol288 = pd.Series(lr).rolling(288, min_periods=288).std().to_numpy()
    feats = np.column_stack([r288, vol288])
    valid = np.isfinite(feats).all(axis=1)
    train_mask = (ts <= TRAIN_END).to_numpy() & valid
    scaler = RobustScaler().fit(feats[train_mask])
    z = np.zeros_like(feats)
    z[valid] = scaler.transform(feats[valid])
    hmm = GaussianStateModel(n_states=3, n_iter=50, seed=SEED)
    hmm.fit(z[train_mask])
    proba = np.full((len(close), 3), np.nan)
    proba[valid] = hmm.filter_proba(z[valid])
    state = np.full(len(close), -1, dtype=int)
    state[valid] = np.nanargmax(proba[valid], axis=1)
    means = [np.nanmean(r288[train_mask & (state == k)]) for k in range(3)]
    order = np.argsort(means)
    name_of = {int(order[0]): "bear", int(order[1]): "chop", int(order[2]): "bull"}
    state[~valid] = int(order[1])

    d2 = np.full(len(close), 1, dtype=int)
    r288s = np.full(len(close), np.nan)
    r288s[288:] = close[288:] / close[:-288] - 1.0
    d2[r288s > 0.04] = 2
    d2[r288s < -0.04] = 0
    d2_name = {0: "bear", 1: "chop", 2: "bull"}

    week_start = ts.iloc[-1] - pd.Timedelta(days=7)
    idx = np.flatnonzero((ts >= week_start).to_numpy())
    h_ts = ts.to_numpy()[idx]
    occ = {name_of[k]: round(float((state[idx] == k).mean() * 100), 1) for k in range(3)}
    print(json.dumps({"week": [str(week_start), str(ts.iloc[-1])], "hmm_occupancy_pct": occ}))

    fig, axes = plt.subplots(3, 1, figsize=(16, 7), sharex=True,
                             gridspec_kw={"height_ratios": [10, 0.7, 0.7], "hspace": 0.06})
    ax = axes[0]
    for s, e, st in contiguous_runs(state[idx]):
        ax.axvspan(h_ts[s], h_ts[e], color=REGIME_COLORS[name_of[st]], alpha=0.16, linewidth=0)
    ax.plot(h_ts, close[idx], color=INK, linewidth=1.0)
    ax.set_title(f"{SYMBOL[:-4]} 5m — causal HMM regimes, last 7 days ({pd.Timestamp(week_start).date()} .. {ts.iloc[-1].date()})",
                 loc="left", fontsize=13, color=INK)
    ax.set_ylabel(f"{SYMBOL} close", fontsize=10, color=INK)
    ax.grid(axis="y", color="#000000", alpha=0.08, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(facecolor=c, alpha=0.6, label=l) for l, c in
                       (("bull", C_BULL), ("chop", C_CHOP), ("bear", C_BEAR))],
              loc="upper left", frameon=False, fontsize=9, ncol=3)
    for strip_ax, states, namer, label in ((axes[1], state[idx], name_of, "HMM (causal)  "),
                                           (axes[2], d2[idx], d2_name, "D2 rule ±4%  ")):
        for s, e, st in contiguous_runs(states):
            strip_ax.axvspan(h_ts[s], h_ts[e], color=REGIME_COLORS[namer[st]], alpha=0.9, linewidth=0)
        strip_ax.set_yticks([])
        strip_ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, color=INK)
        for side in ("top", "right", "left", "bottom"):
            strip_ax.spines[side].set_visible(False)
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
