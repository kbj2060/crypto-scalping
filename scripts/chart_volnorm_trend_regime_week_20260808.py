"""Human-scale trend regime: vol-normalized multi-day trend z-score with hysteresis
(2026-08-08, diagnostic; answers "bull doesn't look like bull" about the 24h/±4% detectors).

z_t = (logC_t - logC_{t-N}) / (sigma_1bar,t * sqrt(N)),  N = 864 bars (3 days),
sigma_1bar = rolling 288-bar std of 1-bar log returns (causal).
Hysteresis: enter bull when z > +1.25, stay until z < +0.5 (bear symmetric) -- prevents flicker
and matches how a human labels a chart. All causal, training-free.

Renders the last-7-days chart for a given panel with three strips: the new vol-normalized
detector, the old causal HMM, and the old D2 ±4% rule.
"""
from __future__ import annotations

import argparse
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

TRAIN_END = pd.Timestamp("2025-08-31 23:59:59")
SEED = 903174
N_TREND = 864  # 3 days
ENTER_Z, EXIT_Z = 1.25, 0.5  # overridden by --enter/--exit
C_BULL, C_BEAR, C_CHOP = "#2563EB", "#D9542B", "#9AA0A6"
REGIME_COLORS = {"bull": C_BULL, "bear": C_BEAR, "chop": C_CHOP}
INK = "#1F2430"


def contiguous_runs(states):
    change = np.flatnonzero(np.diff(states) != 0)
    starts = np.concatenate([[0], change + 1])
    ends = np.concatenate([change, [len(states) - 1]])
    return list(zip(starts, ends, states[starts]))


def volnorm_regime(close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logc = np.log(close)
    lr = np.diff(logc, prepend=logc[0])
    sig1 = pd.Series(lr).rolling(288, min_periods=288).std().to_numpy()
    z = np.full(len(close), np.nan)
    z[N_TREND:] = (logc[N_TREND:] - logc[:-N_TREND]) / (sig1[N_TREND:] * np.sqrt(N_TREND))
    reg = np.full(len(close), 1, dtype=int)
    cur = 1
    for t in range(len(close)):
        zt = z[t]
        if not np.isfinite(zt):
            reg[t] = 1
            continue
        if cur == 1:
            if zt > ENTER_Z:
                cur = 2
            elif zt < -ENTER_Z:
                cur = 0
        elif cur == 2 and zt < EXIT_Z:
            cur = 1
            if zt < -ENTER_Z:
                cur = 0
        elif cur == 0 and zt > -EXIT_Z:
            cur = 1
            if zt > ENTER_Z:
                cur = 2
        reg[t] = cur
    return reg, z


def hmm_regime(close: np.ndarray, ts: pd.Series) -> np.ndarray:
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
    remap = {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}
    named = np.array([remap.get(s, 1) for s in state])
    named[~valid] = 1
    return named


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--enter", type=float, default=1.25)
    ap.add_argument("--exit", type=float, default=0.5)
    args = ap.parse_args()
    global ENTER_Z, EXIT_Z
    ENTER_Z, EXIT_Z = args.enter, args.exit
    panel = pd.read_csv(ROOT / args.panel, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)

    vn, zscore = volnorm_regime(close)
    hm = hmm_regime(close, ts)
    d2 = np.full(len(close), 1, dtype=int)
    r288s = np.full(len(close), np.nan)
    r288s[288:] = close[288:] / close[:-288] - 1.0
    d2[r288s > 0.04] = 2
    d2[r288s < -0.04] = 0
    name_of = {0: "bear", 1: "chop", 2: "bull"}

    week_start = ts.iloc[-1] - pd.Timedelta(days=7)
    idx = np.flatnonzero((ts >= week_start).to_numpy())
    h_ts = ts.to_numpy()[idx]
    occ = {name_of[k]: round(float((vn[idx] == k).mean() * 100), 1) for k in range(3)}
    print(json.dumps({"symbol": args.symbol, "volnorm_week_occupancy_pct": occ}))

    fig, axes = plt.subplots(4, 1, figsize=(16, 8), sharex=True,
                             gridspec_kw={"height_ratios": [10, 0.7, 0.7, 0.7], "hspace": 0.07})
    ax = axes[0]
    for s, e, st in contiguous_runs(vn[idx]):
        ax.axvspan(h_ts[s], h_ts[e], color=REGIME_COLORS[name_of[st]], alpha=0.18, linewidth=0)
    ax.plot(h_ts, close[idx], color=INK, linewidth=1.0)
    ax.set_title(f"{args.symbol[:-4]} 5m — VOL-NORMALIZED 3-day trend regime (z enter ±{ENTER_Z}, exit ±{EXIT_Z}), last 7 days",
                 loc="left", fontsize=13, color=INK)
    ax.set_ylabel(f"{args.symbol} close", fontsize=10, color=INK)
    ax.grid(axis="y", color="#000000", alpha=0.08, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(facecolor=c, alpha=0.6, label=l) for l, c in
                       (("bull", C_BULL), ("chop", C_CHOP), ("bear", C_BEAR))],
              loc="upper left", frameon=False, fontsize=9, ncol=3)
    for strip_ax, states, label in ((axes[1], vn[idx], "vol-norm 3d  "),
                                    (axes[2], hm[idx], "old HMM 24h  "),
                                    (axes[3], d2[idx], "old D2 ±4%  ")):
        for s, e, st in contiguous_runs(states):
            strip_ax.axvspan(h_ts[s], h_ts[e], color=REGIME_COLORS[name_of[st]], alpha=0.9, linewidth=0)
        strip_ax.set_yticks([])
        strip_ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, color=INK)
        for side in ("top", "right", "left", "bottom"):
            strip_ax.spines[side].set_visible(False)
    fig.savefig(ROOT / args.out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"wrote {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
