"""SOL HMM regime optimization for persistence + per-regime feature-distribution separation
(2026-08-08, user-directed; analysis-only — NO PnL OOS read in this cycle).

Motivation: the causal 3-state HMM's median regime run is only 4-9 hours (stripey charts,
regime-flip churn), and rev8 showed per-regime structure is real but needs a regime the trader
can actually stand in. This script optimizes the regime model with TRAIN-ONLY selection:

Grid (pre-registered, closed):
  features: F1=[r288, vol288]  F2=[r288, r72, vol288]  F3=[r864, vol864]
  decode-time transition stickiness override: {em (as fitted), 0.995, 0.9995}
  (EM fits emissions; the override rewrites A's diagonal before CAUSAL filtering — a dwell
   prior at decode time, params otherwise frozen)

Train-only metrics per config:
  - persistence: median/mean causal-state run length (bars)
  - separation: mean of the top-20 per-feature max-pairwise-KS across regime pairs
  - mechanism: within-regime top-20 direction-sign stability across 3 train sub-windows
  - occupancy balance (reject degenerate states, min occupancy >= 8%)

Selection: among configs with median run >= 288 bars (24h) and occupancy ok, maximize train
sign stability (tie-break: separation). The selected config gets ONE VAL check: within-regime
top-20 train->VAL sign agreement (Stage-R style), compared against D2's 85/60/35 baseline.
Charts: regime overlay for the selected config + run-length comparison vs the original HMM.
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
from scipy.stats import ks_2samp, rankdata  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from retrain_clean_regime_hmm_20260517 import GaussianStateModel  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    RAW_LEVEL_COLS, PANEL_PATH, LABEL_PATH, SEED, HORIZON_BARS, TRAIN_END, VAL_START, VAL_END,
)

OUT_DIR = ROOT / "tmp/sol_dl_rl_survey_20260807/hmm_optimize"
TOP_K = 20
MIN_RUN = 288
MIN_OCC = 0.08
C_BULL, C_BEAR, C_CHOP = "#2563EB", "#D9542B", "#9AA0A6"
REGIME_COLORS = {"bull": C_BULL, "bear": C_BEAR, "chop": C_CHOP}
INK = "#1F2430"


def auc_binary(x, y):
    m = np.isfinite(x)
    x, y = x[m], y[m]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 < 50 or n0 < 50:
        return np.nan
    r = rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def causal_filter(model: GaussianStateModel, x: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Forward filtering with a custom transition matrix (params otherwise frozen)."""
    log_emit = model._log_emission(x)
    n, k = log_emit.shape
    logA = np.log(A + 1e-300)
    out = np.zeros((n, k))
    log_alpha = np.log(model.pi_ + 1e-300) + log_emit[0]
    log_alpha -= model._logsumexp(log_alpha)
    out[0] = np.exp(log_alpha)
    for t in range(1, n):
        log_alpha = model._logsumexp(log_alpha[:, None] + logA, axis=0) + log_emit[t]
        log_alpha -= model._logsumexp(log_alpha)
        out[t] = np.exp(log_alpha)
    return out


def runs_of(states: np.ndarray) -> list[int]:
    change = np.flatnonzero(np.diff(states) != 0)
    starts = np.concatenate([[0], change + 1])
    ends = np.concatenate([change, [len(states) - 1]])
    return list(ends - starts + 1)


def contiguous_runs(states):
    change = np.flatnonzero(np.diff(states) != 0)
    starts = np.concatenate([[0], change + 1])
    ends = np.concatenate([change, [len(states) - 1]])
    return list(zip(starts, ends, states[starts]))


def sign_stability(x, action, idx_windows, regime, k_top=TOP_K):
    """Return (mean across regimes of top-K train-sign stability across sub-windows, per-regime val agreement if 'val' in windows)."""
    per_regime_sub, per_regime_val = [], {}
    for r in range(3):
        aucs = {}
        for wname, idx in idx_windows.items():
            sub = idx[regime[idx] == r]
            a = action[sub]
            nz = a != 0
            vals = np.full(x.shape[1], np.nan)
            if nz.sum() > 200:
                y = (a[nz] == 1).astype(int)
                for f in range(x.shape[1]):
                    vals[f] = auc_binary(x[sub, f][nz].astype(np.float64), y)
            aucs[wname] = vals
        dev = np.abs(np.nan_to_num(aucs["train"], nan=0.5) - 0.5)
        top = np.argsort(-dev)[:k_top]
        s_tr = np.sign(aucs["train"][top] - 0.5)
        subs = [np.sign(np.nan_to_num(aucs[f"tr_sub{j}"][top], nan=0.5) - 0.5) for j in (1, 2, 3)]
        ok = float(np.mean((np.stack(subs, axis=1) == s_tr[:, None]).all(axis=1)))
        per_regime_sub.append(ok)
        if "val" in aucs:
            s_val = np.sign(np.nan_to_num(aucs["val"][top], nan=0.5) - 0.5)
            per_regime_val[r] = float((s_val == s_tr).mean())
    return float(np.mean(per_regime_sub)), per_regime_val


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = pd.read_parquet(LABEL_PATH)
    action = labels["trade_outcome_action"].to_numpy()
    tp_moves = labels["tp_move"].to_numpy(dtype=np.float64)
    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    xfeat = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)
    logc = np.log(close)

    def roll_ret(n):
        r = np.full(len(close), np.nan)
        r[n:] = logc[n:] - logc[:-n]
        return r

    lr = np.diff(logc, prepend=logc[0])

    def roll_vol(n):
        return pd.Series(lr).rolling(n, min_periods=n).std().to_numpy()

    FEATSETS = {
        "F1_r288_vol288": np.column_stack([roll_ret(288), roll_vol(288)]),
        "F2_r288_r72_vol288": np.column_stack([roll_ret(288), roll_ret(72), roll_vol(288)]),
        "F3_r864_vol864": np.column_stack([roll_ret(864), roll_vol(864)]),
    }
    STICKY_DECODE = {"em": None, "s995": 0.995, "s9995": 0.9995}

    train_mask = (ts <= TRAIN_END).to_numpy()
    purge_cut = np.flatnonzero(train_mask)[-HORIZON_BARS:]
    train_mask[purge_cut] = False
    train_mask &= np.isfinite(tp_moves)
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()
    tr_idx = np.flatnonzero(train_mask)
    subs = np.array_split(tr_idx, 3)
    train_windows = {"train": tr_idx, "tr_sub1": subs[0], "tr_sub2": subs[1], "tr_sub3": subs[2]}

    rng = np.random.default_rng(SEED)
    results = []
    regimes_by_config = {}
    for fname, feats in FEATSETS.items():
        valid = np.isfinite(feats).all(axis=1)
        scaler = RobustScaler().fit(feats[train_mask & valid])
        z = np.zeros_like(feats)
        z[valid] = scaler.transform(feats[valid])
        model = GaussianStateModel(n_states=3, n_iter=50, seed=SEED)
        model.fit(z[train_mask & valid])
        for sname, sval in STICKY_DECODE.items():
            A = model.A_.copy()
            if sval is not None:
                off = A * (1 - np.eye(3))
                off_norm = off / off.sum(axis=1, keepdims=True)
                A = np.eye(3) * sval + off_norm * (1 - sval)
            proba = np.full((len(close), 3), np.nan)
            proba[valid] = causal_filter(model, z[valid], A)
            state = np.full(len(close), -1, dtype=int)
            state[valid] = np.nanargmax(proba[valid], axis=1)
            means = [np.nanmean(roll_ret(288)[train_mask & (state == k)]) for k in range(3)]
            order = np.argsort(means)
            remap = {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}  # 0=bear,1=chop,2=bull
            state_named = np.array([remap.get(s, 1) for s in state])
            state_named[~valid] = 1

            occ = [float((state_named[tr_idx] == k).mean()) for k in range(3)]
            run_lengths = runs_of(state_named[tr_idx])
            med_run = float(np.median(run_lengths))
            # separation: mean of top-20 max-pairwise KS on sampled train rows
            samp = {k: rng.choice(tr_idx[state_named[tr_idx] == k], size=min(6000, int((state_named[tr_idx] == k).sum())), replace=False) for k in range(3)}
            ks_vals = []
            for f in range(xfeat.shape[1]):
                cols = {k: xfeat[samp[k], f][np.isfinite(xfeat[samp[k], f])] for k in range(3)}
                pairs = [(0, 1), (0, 2), (1, 2)]
                ks_vals.append(max(ks_2samp(cols[a], cols[b]).statistic if len(cols[a]) > 200 and len(cols[b]) > 200 else 0.0 for a, b in pairs))
            separation = float(np.mean(sorted(ks_vals, reverse=True)[:20]))
            stab, _ = sign_stability(xfeat, action, train_windows, state_named)
            rec = {"features": fname, "sticky_decode": sname, "occupancy": [round(o, 3) for o in occ],
                   "median_run_bars": med_run, "mean_run_bars": round(float(np.mean(run_lengths)), 1),
                   "separation_top20_ks": round(separation, 3), "train_sign_stability": round(stab, 3),
                   "eligible": bool(med_run >= MIN_RUN and min(occ) >= MIN_OCC)}
            results.append(rec)
            regimes_by_config[(fname, sname)] = state_named
            print(json.dumps(rec), flush=True)

    eligible = [r for r in results if r["eligible"]]
    if not eligible:
        print(json.dumps({"selected": None, "note": "no config reaches median run >= 288 with occupancy >= 8%"}))
        (OUT_DIR / "optimize_results.json").write_text(json.dumps({"results": results, "selected": None}, indent=2))
        return 0
    best = max(eligible, key=lambda r: (r["train_sign_stability"], r["separation_top20_ks"]))
    sel_state = regimes_by_config[(best["features"], best["sticky_decode"])]

    # ONE VAL sign-agreement check for the selected config
    windows_with_val = dict(train_windows)
    windows_with_val["val"] = np.flatnonzero(val_mask)
    _, val_agree = sign_stability(xfeat, action, windows_with_val, sel_state)
    best["val_sign_agreement"] = {"bear": round(val_agree.get(0, np.nan), 3), "chop": round(val_agree.get(1, np.nan), 3), "bull": round(val_agree.get(2, np.nan), 3)}
    best["d2_baseline_val_agreement"] = {"bear": 0.60, "chop": 0.35, "bull": 0.85}
    (OUT_DIR / "optimize_results.json").write_text(json.dumps({"results": results, "selected": best}, indent=2))
    print(json.dumps({"selected": best}, indent=2))

    # chart: full-period overlay for the optimized regime (hourly, no daily vote needed if persistent)
    hourly = np.arange(0, len(close), 12)
    name_of = {0: "bear", 1: "chop", 2: "bull"}
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True, gridspec_kw={"height_ratios": [10, 0.8], "hspace": 0.05})
    ax = axes[0]
    h_ts = ts.to_numpy()[hourly]
    for s, e, st in contiguous_runs(sel_state[hourly]):
        ax.axvspan(h_ts[s], h_ts[e], color=REGIME_COLORS[name_of[st]], alpha=0.16, linewidth=0)
    ax.plot(h_ts, close[hourly], color=INK, linewidth=1.1)
    ax.set_yscale("log")
    ax.set_title(f"SOL — OPTIMIZED causal HMM regimes ({best['features']}, decode-sticky {best['sticky_decode']}, median run {best['median_run_bars']:.0f} bars = {best['median_run_bars']/12:.0f}h)",
                 loc="left", fontsize=12, color=INK)
    ax.legend(handles=[Patch(facecolor=c, alpha=0.6, label=l) for l, c in (("bull", C_BULL), ("chop", C_CHOP), ("bear", C_BEAR))],
              loc="upper right", frameon=False, fontsize=9, ncol=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for b, lab in ((pd.Timestamp("2025-09-01"), "VAL start"), (pd.Timestamp("2026-01-01"), "OOS start")):
        ax.axvline(b, color=INK, alpha=0.5, linewidth=1, linestyle="--")
        ax.text(b, ax.get_ylim()[1], f" {lab}", fontsize=8, color=INK, va="top")
    axs = axes[1]
    for s, e, st in contiguous_runs(sel_state[hourly]):
        axs.axvspan(h_ts[s], h_ts[e], color=REGIME_COLORS[name_of[st]], alpha=0.9, linewidth=0)
    axs.set_yticks([])
    axs.set_ylabel("regime  ", rotation=0, ha="right", va="center", fontsize=9)
    for side in ("top", "right", "left", "bottom"):
        axs.spines[side].set_visible(False)
    fig.savefig(OUT_DIR / "optimized_regime_full.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_DIR/'optimized_regime_full.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
