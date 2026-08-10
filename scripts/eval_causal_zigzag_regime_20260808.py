"""Causal zigzag regime (user-proposed: up-wave = bull, down-wave = bear, no chop) evaluated on
the same scorecard as JM/HMM (2026-08-08). Charts: full / 6mo / week with four strips
(causal zigzag / Jump Model / old HMM / zigzag oracle)."""
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
from scipy.stats import rankdata  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from retrain_clean_regime_hmm_20260517 import GaussianStateModel  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import (  # noqa: E402
    jm_features, fit_jm, causal_decode, zigzag_oracle, causal_zigzag_regime,
    contiguous_runs, runs_of, auc_binary,
)
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import RAW_LEVEL_COLS, HORIZON_BARS, TRAIN_END, VAL_START, VAL_END  # noqa: E402

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026_regimeline.csv"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_regimeline_20260808.parquet"
OUT_DIR = ROOT / "tmp/jump_model_regimes_20260808"
SEED, LAM, THRESH, TOP_K = 903174, 128.0, 0.04, 20
C_BULL, C_BEAR, C_CHOP = "#2563EB", "#D9542B", "#9AA0A6"
COLORS3 = {0: C_BEAR, 1: C_CHOP, 2: C_BULL}
INK = "#1F2430"


def main() -> int:
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = pd.read_parquet(LABEL_PATH)
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

    # causal zigzag regime -> named 0 bear / 2 bull (1 warmup only)
    cz_dir = causal_zigzag_regime(close, THRESH)
    cz = np.where(cz_dir > 0, 2, np.where(cz_dir < 0, 0, 1))
    odir, pivots = zigzag_oracle(close, THRESH)
    oracle = np.where(odir > 0, 2, np.where(odir < 0, 0, 1))

    # metrics
    eval_idx = np.concatenate([tr_idx, v_idx])
    active = cz_dir[eval_idx] != 0
    agreement = float(np.mean(cz_dir[eval_idx][active] == odir[eval_idx][active]))
    lags = []
    for p in pivots[1:]:
        if not (eval_idx[0] <= p <= eval_idx[-1] - 288):
            continue
        d = odir[min(p + 1, n - 1)]
        window = cz_dir[p: min(p + 864, eval_idx[-1])]
        hits = np.flatnonzero(window == d)
        if len(hits):
            lags.append(int(hits[0]))
    fwd288 = np.full(n, np.nan)
    fwd288[:-288] = close[288:] / close[:-288] - 1.0
    stats = {"oracle_agreement_pct": round(agreement * 100, 1), "coverage_pct": 100.0,
             "median_pivot_lag_bars": int(np.median(lags)) if lags else None,
             "median_run_bars": float(np.median(runs_of(cz[tr_idx]))),
             "mean_run_bars": round(float(np.mean(runs_of(cz[tr_idx]))), 1)}
    for si, nm in ((0, "bear"), (2, "bull")):
        m_tr = tr_idx[cz[tr_idx] == si]
        a = action[m_tr]
        nz = a != 0
        auc_tr = np.full(x_feat.shape[1], np.nan)
        if nz.sum() > 300:
            y = (a[nz] == 1).astype(int)
            for f in range(x_feat.shape[1]):
                auc_tr[f] = auc_binary(x_feat[m_tr, f][nz].astype(np.float64), y)
        dev = np.abs(np.nan_to_num(auc_tr, nan=0.5) - 0.5)
        top = np.argsort(-dev)[:TOP_K]
        m_v = v_idx[cz[v_idx] == si]
        a_v = action[m_v]
        nz_v = a_v != 0
        auc_v = np.full(x_feat.shape[1], np.nan)
        if nz_v.sum() > 300:
            y_v = (a_v[nz_v] == 1).astype(int)
            for f in top:
                auc_v[f] = auc_binary(x_feat[m_v, f][nz_v].astype(np.float64), y_v)
        s_tr = np.sign(auc_tr[top] - 0.5)
        stats[f"{nm}_sign_agreement_val"] = round(float(np.mean(s_tr == np.sign(np.nan_to_num(auc_v[top], nan=0.5) - 0.5))), 2)
        stats[f"{nm}_occupancy_pct"] = round(float((cz[tr_idx] == si).mean() * 100), 1)
        stats[f"{nm}_fwd24h_ret_pct"] = round(float(np.nanmean(fwd288[m_tr]) * 100), 3)
    (OUT_DIR / "causal_zigzag_eval.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    # JM + HMM for strips (refit as in the chart script)
    xj = jm_features(close)
    valid = np.isfinite(xj).all(axis=1)
    scaler = RobustScaler().fit(xj[train_mask & valid])
    z = np.zeros_like(xj)
    z[valid] = scaler.transform(xj[valid])
    mu = fit_jm(z[train_mask & valid], 3, LAM, SEED)
    st = causal_decode(z[valid], mu, LAM)
    jm = np.full(n, -1, dtype=int)
    jm[valid] = st
    r288 = np.full(n, np.nan)
    r288[288:] = np.log(close[288:] / close[:-288])
    means = [np.nanmean(r288[train_mask & (jm == s)]) for s in range(3)]
    order = np.argsort(means)
    jm = np.array([{int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}.get(s, 1) for s in jm])

    lr = np.diff(np.log(close), prepend=np.log(close[0]))
    vol288 = pd.Series(lr).rolling(288, min_periods=288).std().to_numpy()
    fh = np.column_stack([r288, vol288])
    validh = np.isfinite(fh).all(axis=1)
    sc = RobustScaler().fit(fh[train_mask & validh])
    zh = np.zeros_like(fh)
    zh[validh] = sc.transform(fh[validh])
    hm_model = GaussianStateModel(n_states=3, n_iter=50, seed=SEED)
    hm_model.fit(zh[train_mask & validh])
    proba = np.full((n, 3), np.nan)
    proba[validh] = hm_model.filter_proba(zh[validh])
    sth = np.full(n, -1, dtype=int)
    sth[validh] = np.nanargmax(proba[validh], axis=1)
    means_h = [np.nanmean(r288[train_mask & (sth == s)]) for s in range(3)]
    order_h = np.argsort(means_h)
    hmm = np.array([{int(order_h[0]): 0, int(order_h[1]): 1, int(order_h[2]): 2}.get(s, 1) for s in sth])

    windows = {
        "full": np.arange(0, n, 12),
        "6mo": np.flatnonzero((ts >= ts.iloc[-1] - pd.Timedelta(days=180)).to_numpy())[::3],
        "week": np.flatnonzero((ts >= ts.iloc[-1] - pd.Timedelta(days=7)).to_numpy()),
    }
    for tag, idx in windows.items():
        h_ts = ts.to_numpy()[idx]
        fig, axes = plt.subplots(5, 1, figsize=(16, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [10, 0.7, 0.7, 0.7, 0.7], "hspace": 0.07})
        ax = axes[0]
        for s, e, stt in contiguous_runs(cz[idx]):
            ax.axvspan(h_ts[s], h_ts[e], color=COLORS3[stt], alpha=0.17, linewidth=0)
        ax.plot(h_ts, close[idx], color=INK, linewidth=1.0)
        if tag == "full":
            ax.set_yscale("log")
        ax.set_title(f"BTC — CAUSAL ZIGZAG regime (4%, up-wave=bull/down-wave=bear) — {tag}",
                     loc="left", fontsize=13, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(handles=[Patch(facecolor=c, alpha=0.6, label=l) for l, c in
                           (("bull", C_BULL), ("chop/warmup", C_CHOP), ("bear", C_BEAR))],
                  loc="upper left", frameon=False, fontsize=9, ncol=3)
        for strip_ax, sts, label in ((axes[1], cz[idx], "causal zigzag  "),
                                     (axes[2], jm[idx], "Jump Model  "),
                                     (axes[3], hmm[idx], "old HMM  "),
                                     (axes[4], oracle[idx], "ZIGZAG ORACLE  ")):
            for s, e, stt in contiguous_runs(sts):
                strip_ax.axvspan(h_ts[s], h_ts[e], color=COLORS3[stt], alpha=0.9, linewidth=0)
            strip_ax.set_yticks([])
            strip_ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, color=INK)
            for side in ("top", "right", "left", "bottom"):
                strip_ax.spines[side].set_visible(False)
        fig.savefig(OUT_DIR / f"causal_zigzag_{tag}.png", dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {OUT_DIR / f'causal_zigzag_{tag}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
