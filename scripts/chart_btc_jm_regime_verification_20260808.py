"""Visual verification charts for the Statistical Jump Model regime detector (2026-08-08).

Purpose: the user-facing acceptance test for replacing the HMM — "does the detector paint
falling markets bear and rising markets bull when you LOOK at the chart?"  Renders, per
window (full / VAL / OOS / last 30d):
  - price line colored by the JM regime (causal decode, lambda=32),
  - alignment strips: JM lam32, JM lam128, zigzag 4% oracle (retrospective human ground
    truth, scoring only), old 3-state sticky HMM,
  - per-window bull/bear oracle agreement for each detector in the panel title.
Also persists the decoded per-bar states to data/research/btc_jm_regime_states_20260808.parquet
(timestamp, close, jm_lam32, jm_lam128, hmm, oracle_dir) for downstream expert-model work.
Fitted JM centers are cached to the same directory so downstream scripts can decode causally
without refitting.
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
from test_statistical_jump_model_regimes_20260808 import (  # noqa: E402
    SEED, contiguous_runs, causal_decode, fit_jm, jm_features, zigzag_oracle,
)
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026_regimeline.csv"
OUT_DIR = ROOT / "tmp/jm_regime_verification_20260808"
STATE_DIR = ROOT / "data/research"
LAMBDAS = [32.0, 128.0]
K = 3
C_BULL, C_BEAR, C_CHOP = "#2563EB", "#D9542B", "#9AA0A6"
INK = "#1F2430"
REGIME_COLORS = {0: C_BEAR, 1: C_CHOP, 2: C_BULL}


def fit_or_load_jm(z, train_valid_mask, lam, cache_path: Path):
    if cache_path.exists():
        mu = np.array(json.loads(cache_path.read_text())["mu"])
    else:
        mu = fit_jm(z[train_valid_mask], K, lam, SEED)
        cache_path.write_text(json.dumps({"k": K, "lam": lam, "seed": SEED, "mu": mu.tolist()}))
    return mu


def causal_zigzag(close: np.ndarray, threshold: float = 0.04) -> np.ndarray:
    """Per-bar CURRENT wave direction using confirmed pivots only (no future).
    Mirrors zigzag_oracle's online `up` state: flips when price retraces `threshold`
    from the running extreme.  Returns +1 up-wave, -1 down-wave, 0 before first pivot."""
    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    hi_i = lo_i = 0
    up: bool | None = None
    ext_i = 0
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


def name_states(states, r288, train_mask, k=3):
    means = [np.nanmean(r288[train_mask & (states == s)]) for s in range(k)]
    order = np.argsort(means)
    remap = {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}  # 0=bear 1=chop 2=bull
    named = np.array([remap.get(s, 1) for s in states])
    return named


def agreement(named, oracle_dir, idx):
    det_dir = np.where(named == 2, 1, np.where(named == 0, -1, 0))
    active = det_dir[idx] != 0
    if not active.any():
        return None, 0.0
    agree = float(np.mean(det_dir[idx][active] == oracle_dir[idx][active]))
    return round(agree * 100, 1), round(float(active.mean()) * 100, 1)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)
    train_mask = (ts <= TRAIN_END).to_numpy()

    xj = jm_features(close)
    valid = np.isfinite(xj).all(axis=1)
    scaler = RobustScaler().fit(xj[train_mask & valid])
    z = np.zeros_like(xj)
    z[valid] = scaler.transform(xj[valid])

    r288 = np.full(len(close), np.nan)
    r288[288:] = np.log(close[288:] / close[:-288])

    named_by_lam = {}
    for lam in LAMBDAS:
        mu = fit_or_load_jm(z, train_mask & valid, lam, STATE_DIR / f"btc_jm_k3_lam{int(lam)}_centers_20260808.json")
        st = causal_decode(z[valid], mu, lam)
        states = np.full(len(close), -1, dtype=int)
        states[valid] = st
        named = name_states(states, r288, train_mask)
        named[~valid] = 1
        named_by_lam[int(lam)] = named
        print(f"decoded JM k3 lam{int(lam)}", flush=True)

    # HMM baseline identical to the comparison script
    logc = np.log(close)
    lr = np.diff(logc, prepend=logc[0])
    vol288 = pd.Series(lr).rolling(288, min_periods=288).std().to_numpy()
    fh = np.column_stack([r288, vol288])
    validh = np.isfinite(fh).all(axis=1)
    sc = RobustScaler().fit(fh[train_mask & validh])
    zh = np.zeros_like(fh)
    zh[validh] = sc.transform(fh[validh])
    hmm = GaussianStateModel(n_states=3, n_iter=50, seed=SEED)
    hmm.fit(zh[train_mask & validh])
    sth = np.full(len(close), -1, dtype=int)
    sth[validh] = np.nanargmax(hmm.filter_proba(zh[validh]), axis=1)
    hmm_named = name_states(sth, r288, train_mask)
    hmm_named[~validh] = 1
    print("decoded HMM baseline", flush=True)

    oracle_dir, pivots = zigzag_oracle(close, threshold=0.04)
    oracle_named = np.where(oracle_dir == 1, 2, np.where(oracle_dir == -1, 0, 1))

    # causal zigzag: the oracle's own online direction state (which uses only the past) --
    # the human 4%-wave criterion made live-usable; lag = the 4% confirmation move.
    czz = causal_zigzag(close, threshold=0.04)
    czz_named = np.where(czz == 1, 2, np.where(czz == -1, 0, 1))

    pd.DataFrame({
        "timestamp": ts, "close": close,
        "jm_lam32": named_by_lam[32], "jm_lam128": named_by_lam[128],
        "hmm": hmm_named, "czz4": czz_named, "oracle_dir": oracle_dir,
    }).to_parquet(STATE_DIR / "btc_jm_regime_states_20260808.parquet", index=False)

    windows = [
        ("full", None, 12),
        ("val_2025Q4", (VAL_START, VAL_END), 3),
        ("oos_2026Q1", (OOS_START, OOS_END), 3),
        ("last30d", (ts.iloc[-1] - pd.Timedelta(days=30), ts.iloc[-1]), 1),
    ]
    stats = {}
    for tag, bounds, ds in windows:
        if bounds is None:
            idx_all = np.arange(len(close))
        else:
            idx_all = np.flatnonzero(((ts >= bounds[0]) & (ts <= bounds[1])).to_numpy())
        idx = idx_all[::ds]
        h_ts = ts.to_numpy()[idx]
        stats[tag] = {}
        for det_name, named in (("causal zigzag 4%", czz_named), ("JM lam32", named_by_lam[32]),
                                ("JM lam128", named_by_lam[128]), ("old HMM", hmm_named)):
            ag, cov = agreement(named, oracle_dir, idx_all)
            stats[tag][det_name] = {"oracle_agreement_pct": ag, "coverage_pct": cov}

        fig, axes = plt.subplots(6, 1, figsize=(16, 9.7), sharex=True,
                                 gridspec_kw={"height_ratios": [10, 0.7, 0.7, 0.7, 0.7, 0.7], "hspace": 0.08})
        ax = axes[0]
        named_main = czz_named
        for s, e, stt in contiguous_runs(named_main[idx]):
            seg = slice(s, min(e + 2, len(idx)))
            ax.plot(h_ts[seg], close[idx][seg], color=REGIME_COLORS[stt], linewidth=1.1)
            if stt != 1:
                ax.axvspan(h_ts[s], h_ts[min(e + 1, len(idx) - 1)], color=REGIME_COLORS[stt],
                           alpha=0.07, linewidth=0)
        if tag == "full":
            ax.set_yscale("log")
        aczz = stats[tag]["causal zigzag 4%"]["oracle_agreement_pct"]
        a32 = stats[tag]["JM lam32"]["oracle_agreement_pct"]
        ahmm = stats[tag]["old HMM"]["oracle_agreement_pct"]
        ax.set_title(f"BTC — causal-zigzag 4% regimes (line color) — {tag}   "
                     f"[bull/bear oracle agreement: causal-zz {aczz}%  JM lam32 {a32}%  old HMM {ahmm}%]",
                     loc="left", fontsize=12, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(handles=[Patch(facecolor=c, alpha=0.8, label=l) for l, c in
                           (("bull", C_BULL), ("chop", C_CHOP), ("bear", C_BEAR))],
                  loc="upper left", frameon=False, fontsize=9, ncol=3)
        for strip_ax, sts_arr, label in ((axes[1], czz_named[idx], "causal zigzag  "),
                                         (axes[2], named_by_lam[32][idx], "JM lam32  "),
                                         (axes[3], named_by_lam[128][idx], "JM lam128  "),
                                         (axes[4], oracle_named[idx], "zigzag oracle*  "),
                                         (axes[5], hmm_named[idx], "old HMM  ")):
            for s, e, stt in contiguous_runs(sts_arr):
                strip_ax.axvspan(h_ts[s], h_ts[min(e + 1, len(idx) - 1)], color=REGIME_COLORS[stt],
                                 alpha=0.9, linewidth=0)
            strip_ax.set_yticks([])
            strip_ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, color=INK)
            for side in ("top", "right", "left", "bottom"):
                strip_ax.spines[side].set_visible(False)
        axes[-1].annotate("*zigzag oracle is retrospective (uses future) — human ground truth for scoring only",
                          xy=(0.0, -1.6), xycoords="axes fraction", fontsize=8, color="#6B7280")
        fig.savefig(OUT_DIR / f"jm_verify_{tag}.png", dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {OUT_DIR / f'jm_verify_{tag}.png'}", flush=True)

    (OUT_DIR / "window_agreement.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
