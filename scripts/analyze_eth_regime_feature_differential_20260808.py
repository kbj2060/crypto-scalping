"""ETH port of the BTC regime-differential diagnostic (2026-08-08).

Question: BTC's regime-conditioned entry axis died because the regime DIFFERENTIAL
(delta = AUC_bull - AUC_bear) survives train->VAL (84-96%) but dies train->OOS (36-52%,
indistinguishable from random feature subsets). ETH is the one asset with a working
regime-routed live model and 1.5-3.5x SOL's within-regime signal magnitude -- does ETH's
differential persist where BTC's does not?

Same methodology as scripts/analyze_btc_regime_feature_differential_20260808.py:
  gates: d2_rule (288-bar +-4%), jm_lam32 (Statistical Jump Model k3), czz4 (causal 4% zigzag)
  1 IDENTITY      standardized bull-vs-bear feature means on TRAIN (no labels)
  2 WITHIN        per-regime AUC vs the TB action label, sign kept on VAL / OOS (old Stage R view)
  3 DIFFERENTIAL  delta ranked by |delta| on TRAIN, sign re-measured on VAL and OOS, against a
                  20-draw random-subset baseline
  4 CARRIERS      does the positioning/funding/CVD family carry the differential
  5 PERMUTATION   the decisive BTC test: TRAIN-ONLY qualifier count (delta sign identical across
                  4 contiguous train folds AND |median delta| >= 0.01) versus a null built by
                  circularly shifting the regime vector (preserves feature autocorrelation and
                  regime run structure, destroys the alignment), R=10.

Scope: descriptive diagnostic. OOS is read as a persistence measurement, not a selection
criterion; nothing here is adopted, promoted, or wired anywhere.
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
from scipy.stats import rankdata  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from test_statistical_jump_model_regimes_20260808 import (  # noqa: E402
    jm_features, fit_jm, causal_decode, causal_zigzag_regime,
)
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    RAW_LEVEL_COLS, HORIZON_BARS, TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

PANEL_PATH = ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv"
LABEL_PATH = ROOT / "data/splits/year_oos/eth_5m_tripbarrier_tradeoutcome_labels_20260808.parquet"
OUT_DIR = ROOT / "tmp/regime_feature_analysis_eth_20260808"
TOP_N, N_RANDOM, SEED = 25, 20, 903174
N_FOLDS, R_PERM, MIN_ABS_DELTA = 4, 10, 0.01
CARRIER_PAT = ("toptrader", "long_short", "whale", "funding", "cvd", "oi_", "open_interest",
               "crowding", "positioning", "taker")
INK, C_BULL, C_BEAR, C_NEU = "#1F2430", "#2563EB", "#D9542B", "#9AA0A6"


def is_carrier(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in CARRIER_PAT)


def auc_binary(x, y):
    m = np.isfinite(x)
    x, y = x[m], y[m]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 < 50 or n0 < 50:
        return np.nan
    r = rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def regime_auc(x, action, idx, regime, r):
    sub = idx[regime[idx] == r]
    a = action[sub]
    nz = a != 0
    out = np.full(x.shape[1], np.nan)
    if nz.sum() < 200:
        return out
    yv = (a[nz] == 1).astype(int)
    for f in range(x.shape[1]):
        out[f] = auc_binary(x[sub, f][nz].astype(np.float64), yv)
    return out


def auc_matrix(xf: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Mann-Whitney AUC per column (NaN already median-filled). Used for the
    permutation null, where the per-feature loop would be prohibitively slow."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 < 50 or n0 < 50:
        return np.full(xf.shape[1], np.nan)
    r = rankdata(xf, axis=0)
    return (r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def delta_on(xf, action, idx, regime):
    """delta = AUC_bull - AUC_bear on the given index set (vectorized path)."""
    out = []
    for r in (2, 0):
        sub = idx[regime[idx] == r]
        a = action[sub]
        nz = a != 0
        if nz.sum() < 200:
            return None
        out.append(auc_matrix(xf[sub][nz], (a[nz] == 1).astype(int)))
    return out[0] - out[1]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = pd.read_parquet(LABEL_PATH)
    n = min(len(panel), len(labels))
    panel, labels = panel.iloc[:n].reset_index(drop=True), labels.iloc[:n]
    assert (labels["timestamp"].to_numpy() == panel["timestamp"].to_numpy()).all()
    action = labels["trade_outcome_action"].to_numpy()
    tp_moves = labels["tp_move"].to_numpy(dtype=np.float64)
    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    x = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)

    train_mask = (ts <= TRAIN_END).to_numpy()
    tr_all = np.flatnonzero(train_mask)
    train_mask[tr_all[-HORIZON_BARS:]] = False
    train_mask &= np.isfinite(tp_moves)
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()
    oos_mask = ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()
    tr_idx, v_idx, o_idx = (np.flatnonzero(m) for m in (train_mask, val_mask, oos_mask))
    print(f"ETH panel {n} rows, {len(feat_cols)} features; train {len(tr_idx)} / val {len(v_idx)} / oos {len(o_idx)}", flush=True)

    # ---- gates ----
    r288 = np.full(n, np.nan)
    r288[288:] = close[288:] / close[:-288] - 1.0
    d2 = np.full(n, 1, dtype=np.int8)
    d2[r288 > 0.04] = 2
    d2[r288 < -0.04] = 0

    xj = jm_features(close)
    valid = np.isfinite(xj).all(axis=1)
    z = np.zeros_like(xj)
    z[valid] = RobustScaler().fit(xj[train_mask & valid]).transform(xj[valid])
    print("fitting JM k3 lam32 on ETH train...", flush=True)
    mu = fit_jm(z[train_mask & valid], 3, 32.0, SEED)
    st = causal_decode(z[valid], mu, 32.0)
    jm = np.full(n, -1, dtype=int)
    jm[valid] = st
    lr288 = np.full(n, np.nan)
    lr288[288:] = np.log(close[288:] / close[:-288])
    means = [np.nanmean(lr288[train_mask & (jm == s)]) for s in range(3)]
    order = np.argsort(means)
    jm = np.array([{int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}.get(s, 1) for s in jm], dtype=np.int8)

    cdir = causal_zigzag_regime(close, 0.04)
    czz = np.where(cdir > 0, 2, np.where(cdir < 0, 0, 1)).astype(np.int8)

    gates = {"d2_rule": d2, "jm_lam32": jm, "czz4": czz}

    # median-filled matrix for the vectorized permutation path
    med = np.nanmedian(x[tr_idx], axis=0)
    xf = np.where(np.isfinite(x), x, med).astype(np.float64)

    rng = np.random.default_rng(SEED)
    report: dict = {"asset": "ETHUSDT", "n_features": len(feat_cols), "top_n": TOP_N,
                    "scope": "descriptive diagnostic of ETH's regime differential; OOS is a "
                             "persistence measurement, not a selection criterion"}

    for gname, regime in gates.items():
        print(f"--- {gname} ---", flush=True)
        occ = {nm: round(float((regime[tr_idx] == r).mean()), 3)
               for r, nm in ((0, "bear"), (1, "chop"), (2, "bull"))}
        runs = np.diff(np.flatnonzero(np.diff(regime) != 0))
        blk = {"occupancy_train": occ, "median_run_bars": float(np.median(runs)) if len(runs) else None}

        mu_z = x[tr_idx].mean(axis=0)
        sd_z = np.where(x[tr_idx].std(axis=0) > 0, x[tr_idx].std(axis=0), 1.0)
        zz = (x - mu_z) / sd_z
        sep = np.nan_to_num(np.nanmean(zz[tr_idx[regime[tr_idx] == 2]], axis=0)
                            - np.nanmean(zz[tr_idx[regime[tr_idx] == 0]], axis=0))
        ident = np.argsort(-np.abs(sep))[:TOP_N]
        blk["1_identity_top"] = [{"feature": feat_cols[i], "bull_minus_bear_sd": round(float(sep[i]), 3)}
                                 for i in ident[:10]]

        auc_tr = {r: regime_auc(x, action, tr_idx, regime, r) for r in (0, 2)}
        auc_v = {r: regime_auc(x, action, v_idx, regime, r) for r in (0, 2)}
        auc_o = {r: regime_auc(x, action, o_idx, regime, r) for r in (0, 2)}
        within = {}
        for r, nm in ((0, "bear"), (2, "bull")):
            dev = np.abs(np.nan_to_num(auc_tr[r], nan=0.5) - 0.5)
            top = np.argsort(-dev)[:TOP_N]
            s_tr = np.sign(auc_tr[r][top] - 0.5)
            within[nm] = {
                "sign_kept_val": round(float(np.mean(s_tr == np.sign(np.nan_to_num(auc_v[r][top], nan=0.5) - 0.5))), 3),
                "sign_kept_oos": round(float(np.mean(s_tr == np.sign(np.nan_to_num(auc_o[r][top], nan=0.5) - 0.5))), 3),
                "mean_abs_dev_train": round(float(dev[top].mean()), 4),
                "top5": [feat_cols[i] for i in top[:5]]}
        blk["2_within_regime"] = within

        d_tr = np.nan_to_num(auc_tr[2], nan=0.5) - np.nan_to_num(auc_tr[0], nan=0.5)
        d_v = np.nan_to_num(auc_v[2], nan=0.5) - np.nan_to_num(auc_v[0], nan=0.5)
        d_o = np.nan_to_num(auc_o[2], nan=0.5) - np.nan_to_num(auc_o[0], nan=0.5)
        top_d = np.argsort(-np.abs(d_tr))[:TOP_N]
        rand_v, rand_o = [], []
        for _ in range(N_RANDOM):
            sel = rng.choice(len(feat_cols), size=TOP_N, replace=False)
            rand_v.append(float(np.mean(np.sign(d_tr[sel]) == np.sign(d_v[sel]))))
            rand_o.append(float(np.mean(np.sign(d_tr[sel]) == np.sign(d_o[sel]))))
        blk["3_differential"] = {
            "max_abs_delta_auc_train": round(float(np.abs(d_tr).max()), 4),
            "median_abs_delta_top": round(float(np.median(np.abs(d_tr[top_d]))), 4),
            "sign_kept_val": round(float(np.mean(np.sign(d_tr[top_d]) == np.sign(d_v[top_d]))), 3),
            "sign_kept_oos": round(float(np.mean(np.sign(d_tr[top_d]) == np.sign(d_o[top_d]))), 3),
            "sign_kept_both": round(float(np.mean((np.sign(d_tr[top_d]) == np.sign(d_v[top_d]))
                                                  & (np.sign(d_tr[top_d]) == np.sign(d_o[top_d])))), 3),
            "random_subset_baseline": {"val_mean": round(float(np.mean(rand_v)), 3),
                                       "oos_mean": round(float(np.mean(rand_o)), 3)},
            "top10": [{"feature": feat_cols[i], "delta_train": round(float(d_tr[i]), 4),
                       "delta_val": round(float(d_v[i]), 4), "delta_oos": round(float(d_o[i]), 4),
                       "sign_kept": bool(np.sign(d_tr[i]) == np.sign(d_v[i]) == np.sign(d_o[i])),
                       "carrier": is_carrier(feat_cols[i])} for i in top_d[:10]],
        }
        carrier_mask = np.array([is_carrier(c) for c in feat_cols])
        blk["4_carriers"] = {
            "carrier_share_of_all_features": round(float(carrier_mask.mean()), 3),
            "carrier_share_of_top_within_bull": round(float(np.mean(
                carrier_mask[np.argsort(-np.abs(np.nan_to_num(auc_tr[2], nan=0.5) - 0.5))[:TOP_N]])), 3),
            "carrier_share_of_top_differential": round(float(np.mean(carrier_mask[top_d])), 3),
            "mean_abs_delta_carriers": round(float(np.abs(d_tr[carrier_mask]).mean()), 4),
            "mean_abs_delta_others": round(float(np.abs(d_tr[~carrier_mask]).mean()), 4),
        }

        # 5 PERMUTATION -- train-only qualifier count vs a circular-shift null
        folds = np.array_split(tr_idx, N_FOLDS)
        def qualifiers(reg_vec) -> tuple[int, np.ndarray]:
            deltas = []
            for fold in folds:
                dd = delta_on(xf, action, fold, reg_vec)
                if dd is None:
                    return -1, np.array([], dtype=int)
                deltas.append(dd)
            D = np.stack(deltas, axis=1)
            same_sign = (np.sign(D) == np.sign(D[:, [0]])).all(axis=1)
            big = np.abs(np.median(D, axis=1)) >= MIN_ABS_DELTA
            return int((same_sign & big).sum()), np.flatnonzero(same_sign & big)
        real_n, real_idx = qualifiers(regime)
        null_counts = []
        for _ in range(R_PERM):
            shift = int(rng.integers(288, n - 288))
            null_counts.append(qualifiers(np.roll(regime, shift))[0])
        null_counts = [c for c in null_counts if c >= 0]
        blk["5_permutation_null"] = {
            "real_qualifiers": real_n,
            "null_mean": round(float(np.mean(null_counts)), 1) if null_counts else None,
            "null_range": [int(min(null_counts)), int(max(null_counts))] if null_counts else None,
            "R": len(null_counts),
            "passes": bool(null_counts and real_n > max(null_counts)),
            "qualifier_features": [feat_cols[i] for i in real_idx[:12]],
        }
        report[gname] = blk
        print(json.dumps({"gate": gname,
                          "diff_sign_kept_val": blk["3_differential"]["sign_kept_val"],
                          "diff_sign_kept_oos": blk["3_differential"]["sign_kept_oos"],
                          "random_oos": blk["3_differential"]["random_subset_baseline"]["oos_mean"],
                          "perm_real": real_n, "perm_null_mean": blk["5_permutation_null"]["null_mean"],
                          "perm_passes": blk["5_permutation_null"]["passes"]}), flush=True)

    (OUT_DIR / "eth_regime_differential.json").write_text(json.dumps(report, indent=2))

    # chart: differential sign persistence, ETH vs the recorded BTC numbers
    btc = {"d2_rule": (0.96, 0.36, 0.444), "jm_lam32": (0.84, 0.52, 0.474), "czz4": (0.84, 0.48, 0.506)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, gname in zip(axes, gates):
        e = report[gname]["3_differential"]
        vals_e = [e["sign_kept_val"], e["sign_kept_oos"]]
        vals_b = list(btc[gname][:2])
        xpos = np.arange(2)
        ax.bar(xpos - 0.19, vals_e, width=0.36, color=C_BULL, label="ETH")
        ax.bar(xpos + 0.19, vals_b, width=0.36, color=C_NEU, label="BTC (recorded)")
        ax.axhline(e["random_subset_baseline"]["oos_mean"], color=C_BEAR, linestyle="--",
                   linewidth=1.2, label="ETH random baseline (OOS)")
        for xx, vv in zip(xpos - 0.19, vals_e):
            ax.text(xx, vv + 0.01, f"{vv:.2f}", ha="center", fontsize=9, color=INK)
        for xx, vv in zip(xpos + 0.19, vals_b):
            ax.text(xx, vv + 0.01, f"{vv:.2f}", ha="center", fontsize=9, color=INK)
        ax.set_xticks(xpos)
        ax.set_xticklabels(["train→VAL", "train→OOS"])
        ax.set_title(gname, fontsize=11, color=INK)
        ax.set_ylim(0, 1.05)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("top-25 differential sign kept", fontsize=10, color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Regime DIFFERENTIAL persistence — ETH vs BTC (does ΔAUC = AUC_bull − AUC_bear survive out of sample?)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eth_vs_btc_differential.png", dpi=130, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_DIR / 'eth_vs_btc_differential.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
