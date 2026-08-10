"""SOL HMM regime analysis charts (2026-08-08, diagnostic).

Fits the project's own sticky diagonal-Gaussian HMM (GaussianStateModel from
scripts/retrain_clean_regime_hmm_20260517.py) with 3 states on TRAIN ONLY
(inputs: standardized 24h log return + 24h realized vol), then decodes the full
series CAUSALLY via filter_proba (frozen params, forward filtering -- no smoothing).
States are named bull/bear/chop by their mean 24h-return. The rev6 rule detector
D2 (288-bar return +/-4%) is drawn as a comparison strip.

NOTE: SOL HMM regime models were fresh-forward REJECTED for live trading use
(2026-07-21); these charts are diagnostic visualization, not a trading artifact.

Outputs: tmp/sol_dl_rl_survey_20260807/hmm_regime_charts/{full_period,val_oos_zoom}.png + stats json.
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

PANEL_PATH = ROOT / "data/splits/year_oos/sol_features_2024_2026.csv"
OUT_DIR = ROOT / "tmp/sol_dl_rl_survey_20260807/hmm_regime_charts"
TRAIN_END = pd.Timestamp("2025-08-31 23:59:59")
VAL_START, OOS_START, OOS_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31 23:59:59")
SEED = 903174

# diverging pair + neutral midpoint (colorblind-safe blue/vermillion/gray); identity is also
# carried by the labeled comparison strips and legend, never color alone
C_BULL, C_BEAR, C_CHOP = "#2563EB", "#D9542B", "#9AA0A6"
REGIME_COLORS = {"bull": C_BULL, "bear": C_BEAR, "chop": C_CHOP}
INK = "#1F2430"


def contiguous_runs(states: np.ndarray):
    change = np.flatnonzero(np.diff(states) != 0)
    starts = np.concatenate([[0], change + 1])
    ends = np.concatenate([change, [len(states) - 1]])
    return list(zip(starts, ends, states[starts]))


def shade(ax, ts, states, name_of):
    for s, e, st in contiguous_runs(states):
        ax.axvspan(ts[s], ts[e], color=REGIME_COLORS[name_of[st]], alpha=0.16, linewidth=0)


def strip(ax, ts, states, name_of, label):
    for s, e, st in contiguous_runs(states):
        ax.axvspan(ts[s], ts[e], color=REGIME_COLORS[name_of[st]], alpha=0.9, linewidth=0)
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, color=INK)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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

    # name states by mean 24h return on train
    means = [np.nanmean(r288[train_mask & (state == k)]) for k in range(3)]
    order = np.argsort(means)  # low -> high
    name_of = {int(order[0]): "bear", int(order[1]): "chop", int(order[2]): "bull"}
    state[~valid] = int(order[1])  # warmup -> chop for display

    # D2 rule regime for comparison
    r288_simple = np.full(len(close), np.nan)
    r288_simple[288:] = close[288:] / close[:-288] - 1.0
    d2 = np.full(len(close), 1, dtype=int)
    d2[r288_simple > 0.04] = 2
    d2[r288_simple < -0.04] = 0
    d2_name = {0: "bear", 1: "chop", 2: "bull"}

    # stats (causal states)
    fwd288 = np.full(len(close), np.nan)
    fwd288[:-288] = close[288:] / close[:-288] - 1.0
    stats = {}
    for k in range(3):
        nm = name_of[k]
        m = valid & (state == k)
        runs = [e - s + 1 for s, e, st in contiguous_runs(state[valid]) if st == k]
        stats[nm] = {
            "occupancy_pct": round(float(m.mean() * 100), 2),
            "mean_fwd_24h_ret_pct": round(float(np.nanmean(fwd288[m]) * 100), 3),
            "median_run_length_bars": int(np.median(runs)) if runs else 0,
            "hmm_state_mean_r288_train_pct": round(float(means[k]) * 100, 2),
        }
    agree = float(np.mean(np.vectorize(name_of.get)(state[valid]) == np.vectorize(d2_name.get)(d2[valid])))
    stats["hmm_vs_D2_agreement_pct"] = round(agree * 100, 1)
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    legend = [Patch(facecolor=C_BULL, alpha=0.6, label="bull"),
              Patch(facecolor=C_CHOP, alpha=0.6, label="chop"),
              Patch(facecolor=C_BEAR, alpha=0.6, label="bear")]

    # hourly downsample for drawing
    hourly = np.arange(0, len(close), 12)

    def draw(idx, title, path, extra_strip_smoothed=False):
        h_ts = ts.to_numpy()[idx]
        h_close = close[idx]
        h_state = state[idx]
        h_d2 = d2[idx]
        n_rows = 3
        fig, axes = plt.subplots(
            n_rows, 1, figsize=(16, 7.5), sharex=True,
            gridspec_kw={"height_ratios": [10, 0.7, 0.7], "hspace": 0.06},
        )
        ax = axes[0]
        shade(ax, h_ts, h_state, name_of)
        ax.plot(h_ts, h_close, color=INK, linewidth=1.1)
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=13, color=INK)
        ax.grid(axis="y", color="#000000", alpha=0.08, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for b, lab in ((TRAIN_END, "train end"), (OOS_START, "OOS start"), (OOS_END, "OOS end")):
            if h_ts[0] <= np.datetime64(b) <= h_ts[-1]:
                ax.axvline(b, color=INK, alpha=0.5, linewidth=1, linestyle="--")
                ax.text(b, ax.get_ylim()[1], f" {lab}", fontsize=8, color=INK, va="top")
        ax.legend(handles=legend, loc="upper right", frameon=False, fontsize=9, ncol=3)
        ax.set_ylabel("SOLUSDT close (log)", fontsize=10, color=INK)
        strip(axes[1], h_ts, h_state, name_of, "HMM (causal)  ")
        strip(axes[2], h_ts, h_d2, d2_name, "D2 rule ±4%  ")
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {path}")

    # full period at daily-majority regime resolution (5m-causal states are 4-8h runs -- at a
    # 2-year x-axis raw shading reads as stripes; the daily mode vote keeps the macro structure)
    days = ts.dt.floor("D")
    day_state = pd.Series(state).groupby(days.to_numpy()).agg(lambda s: s.value_counts().idxmax())
    day_map = dict(zip(day_state.index, day_state.to_numpy()))
    state_daily = np.array([day_map[d] for d in days.to_numpy()])
    day_d2 = pd.Series(d2).groupby(days.to_numpy()).agg(lambda s: s.value_counts().idxmax())
    d2_daily = np.array([dict(zip(day_d2.index, day_d2.to_numpy()))[d] for d in days.to_numpy()])
    state_bak, d2_bak = state.copy(), d2.copy()
    state[:], d2[:] = state_daily, d2_daily
    draw(hourly, "SOL 5m — causal HMM regimes, DAILY majority vote (3-state sticky Gaussian, train-fit 2024-06..2025-08)",
         OUT_DIR / "full_period.png")
    state[:], d2[:] = state_bak, d2_bak
    zoom = hourly[(ts.to_numpy()[hourly] >= np.datetime64(pd.Timestamp("2025-09-01"))) & (ts.to_numpy()[hourly] <= np.datetime64(pd.Timestamp("2026-03-31")))]
    draw(zoom, "SOL — VAL (2025-09..12) + OOS (2026-01..03) zoom, causal HMM vs D2 rule",
         OUT_DIR / "val_oos_zoom.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
