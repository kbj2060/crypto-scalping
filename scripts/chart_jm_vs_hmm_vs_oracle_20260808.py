"""Visual verification charts for the Jump Model regime detector (2026-08-08).

Three windows (full period / last 6 months / last 7 days), each with:
  - price shaded by the JM (K=3, lambda=128, causal decode) regime
  - three comparison strips: Jump Model (causal) / old HMM (causal) / ZIGZAG ORACLE (the
    retrospective 4% wave ground truth -- the "what a human would label" answer key).
Agreement is judged by eye: does the JM strip match the oracle strip's color blocks, and does
the HMM strip look like confetti against both.
"""
from __future__ import annotations

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
from test_statistical_jump_model_regimes_20260808 import (  # noqa: E402
    jm_features, fit_jm, causal_decode, zigzag_oracle, contiguous_runs,
)

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026_regimeline.csv"
OUT_DIR = ROOT / "tmp/jump_model_regimes_20260808"
TRAIN_END = pd.Timestamp("2025-08-31 23:59:59")
SEED = 903174
LAM = 128.0
C_BULL, C_BEAR, C_CHOP = "#2563EB", "#D9542B", "#9AA0A6"
COLORS3 = {0: C_BEAR, 1: C_CHOP, 2: C_BULL}
INK = "#1F2430"


def main() -> int:
    panel = pd.read_csv(PANEL_PATH, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)
    train_mask = (ts <= TRAIN_END).to_numpy()

    # JM k3 lam128
    xj = jm_features(close)
    valid = np.isfinite(xj).all(axis=1)
    scaler = RobustScaler().fit(xj[train_mask & valid])
    z = np.zeros_like(xj)
    z[valid] = scaler.transform(xj[valid])
    mu = fit_jm(z[train_mask & valid], 3, LAM, SEED)
    st = causal_decode(z[valid], mu, LAM)
    jm = np.full(len(close), -1, dtype=int)
    jm[valid] = st
    r288 = np.full(len(close), np.nan)
    r288[288:] = np.log(close[288:] / close[:-288])
    means = [np.nanmean(r288[train_mask & (jm == s)]) for s in range(3)]
    order = np.argsort(means)
    remap = {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}
    jm = np.array([remap.get(s, 1) for s in jm])

    # HMM baseline
    lr = np.diff(np.log(close), prepend=np.log(close[0]))
    vol288 = pd.Series(lr).rolling(288, min_periods=288).std().to_numpy()
    fh = np.column_stack([r288, vol288])
    validh = np.isfinite(fh).all(axis=1)
    sc = RobustScaler().fit(fh[train_mask & validh])
    zh = np.zeros_like(fh)
    zh[validh] = sc.transform(fh[validh])
    hmmm = GaussianStateModel(n_states=3, n_iter=50, seed=SEED)
    hmmm.fit(zh[train_mask & validh])
    proba = np.full((len(close), 3), np.nan)
    proba[validh] = hmmm.filter_proba(zh[validh])
    sth = np.full(len(close), -1, dtype=int)
    sth[validh] = np.nanargmax(proba[validh], axis=1)
    means_h = [np.nanmean(r288[train_mask & (sth == s)]) for s in range(3)]
    order_h = np.argsort(means_h)
    remap_h = {int(order_h[0]): 0, int(order_h[1]): 1, int(order_h[2]): 2}
    hmm = np.array([remap_h.get(s, 1) for s in sth])

    # zigzag oracle (+1 up / -1 down) -> 0 bear / 2 bull
    odir, _ = zigzag_oracle(close, 0.04)
    oracle = np.where(odir > 0, 2, np.where(odir < 0, 0, 1))

    windows = {
        "full": np.arange(0, len(close), 12),
        "6mo": np.flatnonzero((ts >= ts.iloc[-1] - pd.Timedelta(days=180)).to_numpy())[::3],
        "week": np.flatnonzero((ts >= ts.iloc[-1] - pd.Timedelta(days=7)).to_numpy()),
    }
    for tag, idx in windows.items():
        h_ts = ts.to_numpy()[idx]
        fig, axes = plt.subplots(4, 1, figsize=(16, 8.5), sharex=True,
                                 gridspec_kw={"height_ratios": [10, 0.7, 0.7, 0.7], "hspace": 0.07})
        ax = axes[0]
        for s, e, stt in contiguous_runs(jm[idx]):
            ax.axvspan(h_ts[s], h_ts[e], color=COLORS3[stt], alpha=0.17, linewidth=0)
        ax.plot(h_ts, close[idx], color=INK, linewidth=1.0)
        if tag == "full":
            ax.set_yscale("log")
        ax.set_title(f"BTC — Jump Model (K=3, λ=128, causal) vs old HMM vs ZIGZAG ORACLE — {tag}",
                     loc="left", fontsize=13, color=INK)
        ax.set_ylabel("BTCUSDT close", fontsize=10, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(handles=[Patch(facecolor=c, alpha=0.6, label=l) for l, c in
                           (("bull", C_BULL), ("chop", C_CHOP), ("bear", C_BEAR))],
                  loc="upper left", frameon=False, fontsize=9, ncol=3)
        for strip_ax, sts, label in ((axes[1], jm[idx], "Jump Model  "),
                                     (axes[2], hmm[idx], "old HMM  "),
                                     (axes[3], oracle[idx], "ZIGZAG ORACLE  ")):
            for s, e, stt in contiguous_runs(sts):
                strip_ax.axvspan(h_ts[s], h_ts[e], color=COLORS3[stt], alpha=0.9, linewidth=0)
            strip_ax.set_yticks([])
            strip_ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, color=INK)
            for side in ("top", "right", "left", "bottom"):
                strip_ax.spines[side].set_visible(False)
        fig.savefig(OUT_DIR / f"jm_hmm_oracle_{tag}.png", dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {OUT_DIR / f'jm_hmm_oracle_{tag}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
