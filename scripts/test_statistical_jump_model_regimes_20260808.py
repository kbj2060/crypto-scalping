"""Statistical Jump Model (JM) regime detection test vs the HMM baseline (2026-08-08).

Literature: Bemporad et al. 2018 (JM); Nystrup, Lindstrom & Madsen 2020-21 (financial regimes);
Shu, Yu & Kolm 2024 (downside-risk reduction with JMs, incl. BTC/ETH -- arXiv:2402.05272);
Aydinhan, Kolm et al. 2024 (continuous JM). Key claim: the explicit per-transition jump penalty
lambda yields persistent, stable regimes where HMMs flicker and false-alarm.

Implementation (discrete JM, faithful):
  - features: EWM return and EWM downside deviation at halflives {72, 288, 864} bars
    (paper's return/downside-deviation feature family at 5m scale), standardized on TRAIN.
  - fit on TRAIN only by coordinate descent: optimal state sequence via offline DP given
    centers (jump-penalized), centers re-estimated as state means; k-means++ init, 3 restarts.
  - CAUSAL online decode of the full series: forward DP
    V_t(s) = ||x_t - mu_s||^2 + min_{s'}(V_{t-1}(s') + lambda*1{s'!=s}), state_t = argmin V_t.
  - grid: K in {2, 3} x lambda in {32, 128, 512}.
Evaluation per config (vs the 3-state sticky HMM baseline measured identically):
  persistence (median/mean run), economic separation (fwd-24h return per regime),
  within-regime top-20 direction sign stability train->VAL (the project's mechanism metric).
Charts: last-7-days and full-period (daily majority) for the best JM config on BTC.
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
from scipy.stats import rankdata  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from retrain_clean_regime_hmm_20260517 import GaussianStateModel  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import RAW_LEVEL_COLS, HORIZON_BARS, TRAIN_END, VAL_START, VAL_END  # noqa: E402

SEED = 903174
HALflIVES = [72, 288, 864]
K_GRID = [2, 3]
LAMBDA_GRID = [32.0, 128.0, 512.0]
TOP_K = 20
C_BULL, C_BEAR, C_CHOP = "#2563EB", "#D9542B", "#9AA0A6"
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


def jm_features(close: np.ndarray) -> np.ndarray:
    logc = np.log(close)
    lr = pd.Series(np.diff(logc, prepend=logc[0]))
    feats = []
    for hl in HALflIVES:
        feats.append(lr.ewm(halflife=hl).mean().to_numpy())
        downside = lr.clip(upper=0.0) ** 2
        feats.append(np.sqrt(downside.ewm(halflife=hl).mean().to_numpy()))
    return np.column_stack(feats)


def offline_dp(x: np.ndarray, mu: np.ndarray, lam: float) -> np.ndarray:
    n, k = len(x), len(mu)
    cost = ((x[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)
    V = np.zeros((n, k))
    back = np.zeros((n, k), dtype=np.int8)
    V[0] = cost[0]
    for t in range(1, n):
        prev = V[t - 1]
        stay = prev
        switch = prev.min() + lam
        arg_switch = int(prev.argmin())
        for s in range(k):
            if stay[s] <= switch:
                V[t, s] = cost[t, s] + stay[s]
                back[t, s] = s
            else:
                V[t, s] = cost[t, s] + switch
                back[t, s] = arg_switch
    states = np.zeros(n, dtype=np.int8)
    states[-1] = int(V[-1].argmin())
    for t in range(n - 2, -1, -1):
        states[t] = back[t + 1, states[t + 1]]
    return states


def fit_jm(x: np.ndarray, k: int, lam: float, seed: int, n_init: int = 3, n_iter: int = 10):
    rng = np.random.default_rng(seed)
    best_obj, best_mu = np.inf, None
    for init in range(n_init):
        # k-means++ style init
        mu = [x[rng.integers(len(x))]]
        while len(mu) < k:
            d2 = np.min(((x[:, None, :] - np.array(mu)[None, :, :]) ** 2).sum(axis=2), axis=1)
            p = d2 / d2.sum()
            mu.append(x[rng.choice(len(x), p=p)])
        mu = np.array(mu)
        prev_states = None
        for it in range(n_iter):
            states = offline_dp(x, mu, lam)
            for s in range(k):
                if (states == s).sum() > 10:
                    mu[s] = x[states == s].mean(axis=0)
            if prev_states is not None and (states == prev_states).all():
                break
            prev_states = states
        cost = ((x - mu[states]) ** 2).sum() + lam * (np.diff(states) != 0).sum()
        if cost < best_obj:
            best_obj, best_mu = cost, mu.copy()
    return best_mu


def causal_decode(x: np.ndarray, mu: np.ndarray, lam: float) -> np.ndarray:
    n, k = len(x), len(mu)
    cost = ((x[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)
    states = np.zeros(n, dtype=np.int8)
    V = cost[0].copy()
    states[0] = int(V.argmin())
    for t in range(1, n):
        switch = V.min() + lam
        V = cost[t] + np.minimum(V, switch)
        V -= V.min()  # numerical drift control
        states[t] = int(V.argmin())
    return states


def zigzag_oracle(close: np.ndarray, threshold: float = 0.04):
    """Retrospective zigzag wave segmentation (the human 'bull looks like bull' ground truth,
    user-suggested). Returns per-bar wave direction (+1 up / -1 down) and the list of wave-start
    indices. Uses future information BY DESIGN -- scoring reference only, never a live input."""
    n = len(close)
    hi_i = lo_i = 0
    up: bool | None = None
    ext_i = 0
    pivots: list[int] = []
    for t in range(1, n):
        if close[t] > close[hi_i]:
            hi_i = t
        if close[t] < close[lo_i]:
            lo_i = t
        if up is None:
            if close[t] >= close[lo_i] * (1 + threshold):
                up, ext_i = True, t
                pivots.append(lo_i)
            elif close[t] <= close[hi_i] * (1 - threshold):
                up, ext_i = False, t
                pivots.append(hi_i)
        elif up:
            if close[t] > close[ext_i]:
                ext_i = t
            elif close[t] <= close[ext_i] * (1 - threshold):
                pivots.append(ext_i)
                up, ext_i = False, t
        else:
            if close[t] < close[ext_i]:
                ext_i = t
            elif close[t] >= close[ext_i] * (1 + threshold):
                pivots.append(ext_i)
                up, ext_i = True, t
    direction = np.zeros(n, dtype=np.int8)
    if len(pivots) >= 2:
        first_up = close[pivots[1]] > close[pivots[0]]
        bounds = pivots + [n - 1]
        d = 1 if first_up else -1
        for i in range(len(bounds) - 1):
            direction[bounds[i]: bounds[i + 1] + 1] = d
            d = -d
    return direction, pivots


def causal_zigzag_regime(close: np.ndarray, threshold: float = 0.04) -> np.ndarray:
    """CAUSAL zigzag regime (user-proposed): at each bar, the direction of the current wave --
    the opposite of the last CONFIRMED pivot type. Identical state machine to zigzag_oracle but
    without retro-relabeling: when a reversal confirms a pivot at bar t, the new direction
    applies from t onward, so each new wave's first ~threshold move is (unavoidably) still
    painted with the old direction. Returns +1 bull / -1 bear (0 only during warmup)."""
    n = len(close)
    hi_i = lo_i = 0
    up: bool | None = None
    ext_i = 0
    out = np.zeros(n, dtype=np.int8)
    for t in range(1, n):
        if close[t] > close[hi_i]:
            hi_i = t
        if close[t] < close[lo_i]:
            lo_i = t
        if up is None:
            if close[t] >= close[lo_i] * (1 + threshold):
                up, ext_i = True, t
            elif close[t] <= close[hi_i] * (1 - threshold):
                up, ext_i = False, t
        elif up:
            if close[t] > close[ext_i]:
                ext_i = t
            elif close[t] <= close[ext_i] * (1 - threshold):
                up, ext_i = False, t
        else:
            if close[t] < close[ext_i]:
                ext_i = t
            elif close[t] >= close[ext_i] * (1 + threshold):
                up, ext_i = True, t
        out[t] = 0 if up is None else (1 if up else -1)
    return out


def oracle_scores(named: np.ndarray, names: list[str], oracle_dir: np.ndarray, pivots: list[int], idx: np.ndarray) -> dict:
    det_dir = np.zeros(len(named), dtype=np.int8)
    for si, nm in enumerate(names):
        if nm == "bull":
            det_dir[named == si] = 1
        elif nm == "bear":
            det_dir[named == si] = -1
    active = det_dir[idx] != 0
    agree = float(np.mean(det_dir[idx][active] == oracle_dir[idx][active])) if active.any() else np.nan
    lags = []
    idx_set_lo, idx_set_hi = idx[0], idx[-1]
    for j, p in enumerate(pivots[1:], start=1):
        if not (idx_set_lo <= p <= idx_set_hi - 288):
            continue
        d = oracle_dir[min(p + 1, len(oracle_dir) - 1)]
        window = det_dir[p: min(p + 864, idx_set_hi)]
        hits = np.flatnonzero(window == d)
        if len(hits):
            lags.append(int(hits[0]))
    return {"oracle_agreement_pct": round(agree * 100, 1) if np.isfinite(agree) else None,
            "coverage_pct": round(float(active.mean()) * 100, 1),
            "median_pivot_lag_bars": int(np.median(lags)) if lags else None,
            "n_pivots_scored": len(lags)}


def runs_of(states):
    change = np.flatnonzero(np.diff(states) != 0)
    starts = np.concatenate([[0], change + 1])
    ends = np.concatenate([change, [len(states) - 1]])
    return ends - starts + 1


def contiguous_runs(states):
    change = np.flatnonzero(np.diff(states) != 0)
    starts = np.concatenate([[0], change + 1])
    ends = np.concatenate([change, [len(states) - 1]])
    return list(zip(starts, ends, states[starts]))


def evaluate(states_named: np.ndarray, names: list[str], close, x_feat, action, tr_idx, v_idx) -> dict:
    fwd288 = np.full(len(close), np.nan)
    fwd288[:-288] = close[288:] / close[:-288] - 1.0
    out = {"median_run_bars": float(np.median(runs_of(states_named[tr_idx]))),
           "mean_run_bars": round(float(np.mean(runs_of(states_named[tr_idx]))), 1)}
    econ, stab = {}, {}
    for si, nm in enumerate(names):
        m_tr = tr_idx[states_named[tr_idx] == si]
        econ[nm] = {"occupancy_pct": round(float((states_named[tr_idx] == si).mean() * 100), 1),
                    "fwd24h_ret_pct": round(float(np.nanmean(fwd288[m_tr]) * 100), 3) if len(m_tr) else None}
        a = action[m_tr]
        nz = a != 0
        auc_tr = np.full(x_feat.shape[1], np.nan)
        if nz.sum() > 300:
            y = (a[nz] == 1).astype(int)
            for f in range(x_feat.shape[1]):
                auc_tr[f] = auc_binary(x_feat[m_tr, f][nz].astype(np.float64), y)
        dev = np.abs(np.nan_to_num(auc_tr, nan=0.5) - 0.5)
        top = np.argsort(-dev)[:TOP_K]
        m_v = v_idx[states_named[v_idx] == si]
        a_v = action[m_v]
        nz_v = a_v != 0
        auc_v = np.full(x_feat.shape[1], np.nan)
        if nz_v.sum() > 300:
            y_v = (a_v[nz_v] == 1).astype(int)
            for f in top:
                auc_v[f] = auc_binary(x_feat[m_v, f][nz_v].astype(np.float64), y_v)
        s_tr = np.sign(auc_tr[top] - 0.5)
        stab[nm] = round(float(np.mean(s_tr == np.sign(np.nan_to_num(auc_v[top], nan=0.5) - 0.5))), 2)
    out["econ"] = econ
    out["val_sign_agreement"] = stab
    if EVAL_ORACLE is not None:
        oracle_dir, pivots, eval_idx = EVAL_ORACLE
        out["vs_zigzag_oracle"] = oracle_scores(states_named, names, oracle_dir, pivots, eval_idx)
    return out


EVAL_ORACLE: tuple | None = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/splits/year_oos/btc_features_2024_2026_regimeline.csv")
    ap.add_argument("--labels", default="data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_regimeline_20260808.parquet")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--outdir", default="tmp/jump_model_regimes_20260808")
    args = ap.parse_args()
    out_dir = ROOT / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(ROOT / args.panel, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = pd.read_parquet(ROOT / args.labels)
    n = min(len(panel), len(labels))
    panel, labels = panel.iloc[:n].reset_index(drop=True), labels.iloc[:n]
    action = labels["trade_outcome_action"].to_numpy()
    tp_moves = labels["tp_move"].to_numpy(dtype=np.float64)
    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    x_feat = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)

    train_mask = (ts <= TRAIN_END).to_numpy()
    tr_all = np.flatnonzero(train_mask)
    train_mask[tr_all[-HORIZON_BARS:]] = False
    train_mask &= np.isfinite(tp_moves)
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()
    tr_idx = np.flatnonzero(train_mask)
    v_idx = np.flatnonzero(val_mask)

    xj = jm_features(close)
    valid = np.isfinite(xj).all(axis=1)
    scaler = RobustScaler().fit(xj[train_mask & valid])
    z = np.zeros_like(xj)
    z[valid] = scaler.transform(xj[valid])

    r288 = np.full(len(close), np.nan)
    r288[288:] = np.log(close[288:] / close[:-288])

    global EVAL_ORACLE
    oracle_dir, pivots = zigzag_oracle(close, threshold=0.04)
    eval_idx = np.concatenate([tr_idx, v_idx])
    EVAL_ORACLE = (oracle_dir, pivots, eval_idx)
    wave_lens = np.diff(pivots)
    print(json.dumps({"zigzag_oracle": {"threshold": 0.04, "n_waves": len(pivots) - 1,
                                        "median_wave_bars": int(np.median(wave_lens)) if len(wave_lens) else None}}), flush=True)

    results = {}
    states_by_config = {}
    for k in K_GRID:
        for lam in LAMBDA_GRID:
            mu = fit_jm(z[train_mask & valid], k, lam, SEED)
            st = causal_decode(z[valid], mu, lam)
            states = np.full(len(close), -1, dtype=int)
            states[valid] = st
            means = [np.nanmean(r288[train_mask & (states == s)]) for s in range(k)]
            order = np.argsort(means)
            if k == 2:
                remap = {int(order[0]): 0, int(order[1]): 1}
                names = ["bear", "bull"]
            else:
                remap = {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}
                names = ["bear", "chop", "bull"]
            named = np.array([remap.get(s, 0 if k == 2 else 1) for s in states])
            named[~valid] = 0 if k == 2 else 1
            key = f"JM_k{k}_lam{int(lam)}"
            results[key] = evaluate(named, names, close, x_feat, action, tr_idx, v_idx)
            states_by_config[key] = (named, names)
            print(json.dumps({key: results[key]}), flush=True)

    # HMM baseline, identical evaluation
    logc = np.log(close)
    r288h = np.full(len(close), np.nan)
    r288h[288:] = logc[288:] - logc[:-288]
    lr = np.diff(logc, prepend=logc[0])
    vol288 = pd.Series(lr).rolling(288, min_periods=288).std().to_numpy()
    fh = np.column_stack([r288h, vol288])
    validh = np.isfinite(fh).all(axis=1)
    sc = RobustScaler().fit(fh[train_mask & validh])
    zh = np.zeros_like(fh)
    zh[validh] = sc.transform(fh[validh])
    hmm = GaussianStateModel(n_states=3, n_iter=50, seed=SEED)
    hmm.fit(zh[train_mask & validh])
    proba = np.full((len(close), 3), np.nan)
    proba[validh] = hmm.filter_proba(zh[validh])
    sth = np.full(len(close), -1, dtype=int)
    sth[validh] = np.nanargmax(proba[validh], axis=1)
    means = [np.nanmean(r288h[train_mask & (sth == s)]) for s in range(3)]
    order = np.argsort(means)
    remap = {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}
    hmm_named = np.array([remap.get(s, 1) for s in sth])
    hmm_named[~validh] = 1
    results["HMM_baseline"] = evaluate(hmm_named, ["bear", "chop", "bull"], close, x_feat, action, tr_idx, v_idx)
    print(json.dumps({"HMM_baseline": results["HMM_baseline"]}), flush=True)
    (out_dir / f"jm_vs_hmm_{args.symbol}.json").write_text(json.dumps(results, indent=2))

    # charts for the most persistent 3-state JM (lam=512)
    key = "JM_k3_lam512"
    named, names = states_by_config[key]
    name_of = dict(enumerate(names))
    colors = {"bear": C_BEAR, "chop": C_CHOP, "bull": C_BULL}
    week_start = ts.iloc[-1] - pd.Timedelta(days=7)
    for tag, idx in (("week", np.flatnonzero((ts >= week_start).to_numpy())),
                     ("full", np.arange(0, len(close), 12))):
        h_ts = ts.to_numpy()[idx]
        fig, axes = plt.subplots(3, 1, figsize=(16, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [10, 0.7, 0.7], "hspace": 0.06})
        ax = axes[0]
        for s, e, stt in contiguous_runs(named[idx]):
            ax.axvspan(h_ts[s], h_ts[e], color=colors[name_of[stt]], alpha=0.17, linewidth=0)
        ax.plot(h_ts, close[idx], color=INK, linewidth=1.0)
        if tag == "full":
            ax.set_yscale("log")
        ax.set_title(f"{args.symbol[:-4]} — Statistical Jump Model regimes ({key}, causal decode) — {tag}",
                     loc="left", fontsize=13, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(handles=[Patch(facecolor=c, alpha=0.6, label=l) for l, c in
                           (("bull", C_BULL), ("chop", C_CHOP), ("bear", C_BEAR))],
                  loc="upper left", frameon=False, fontsize=9, ncol=3)
        for strip_ax, sts, label in ((axes[1], named[idx], "Jump Model  "),
                                     (axes[2], hmm_named[idx], "old HMM  ")):
            for s, e, stt in contiguous_runs(sts):
                strip_ax.axvspan(h_ts[s], h_ts[e], color=colors[["bear", "chop", "bull"][stt]], alpha=0.9, linewidth=0)
            strip_ax.set_yticks([])
            strip_ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, color=INK)
            for side in ("top", "right", "left", "bottom"):
                strip_ax.spines[side].set_visible(False)
        fig.savefig(out_dir / f"jm_{args.symbol}_{tag}.png", dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out_dir / f'jm_{args.symbol}_{tag}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
