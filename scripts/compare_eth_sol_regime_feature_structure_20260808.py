"""ETH vs SOL regime-feature structure comparison (2026-08-08, analysis-only, no OOS PnL).

Runs the IDENTICAL measurements on both assets' identically-built panels:
  - D2 trend regime (288-bar return +/-4%): occupancy + median/mean run length (persistence)
  - unconditional top-20 direction-feature sign agreement train->VAL (SOL was 0/20)
  - per-regime top-20 direction AUC: train magnitude, train->VAL sign agreement, VAL magnitude
  - bull-vs-bear per-feature AUC Spearman (how disjoint the regimes' information is)
  - within-regime train sub-window sign stability (the rev6 Stage-R metric)

Splits: train <= 2025-08-31 (purged 288), VAL 2025-09-01..12-31 -- both panels cover these.
Output: side-by-side JSON + comparison chart.
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
from scipy.stats import rankdata, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import RAW_LEVEL_COLS, HORIZON_BARS, TRAIN_END, VAL_START, VAL_END  # noqa: E402

ASSETS = {
    "SOL": {"panel": ROOT / "data/splits/year_oos/sol_features_2024_2026.csv",
            "labels": ROOT / "data/splits/year_oos/sol_5m_tripbarrier_tradeoutcome_labels_20260807.parquet"},
    "ETH": {"panel": ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv",
            "labels": ROOT / "data/splits/year_oos/eth_5m_tripbarrier_tradeoutcome_labels_20260808.parquet"},
}
OUT_DIR = ROOT / "tmp/sol_dl_rl_survey_20260807/eth_sol_comparison"
TOP_K = 20
REGIME_NAMES = ["bear", "chop", "bull"]
C_SOL, C_ETH = "#D9542B", "#2563EB"
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


def all_feature_auc(x, action, idx):
    a = action[idx]
    nz = a != 0
    out = np.full(x.shape[1], np.nan)
    if nz.sum() < 200:
        return out
    y = (a[nz] == 1).astype(int)
    for f in range(x.shape[1]):
        out[f] = auc_binary(x[idx, f][nz].astype(np.float64), y)
    return out


def runs_of(states):
    change = np.flatnonzero(np.diff(states) != 0)
    starts = np.concatenate([[0], change + 1])
    ends = np.concatenate([change, [len(states) - 1]])
    return ends - starts + 1


def analyze(asset: str) -> dict:
    cfg = ASSETS[asset]
    panel = pd.read_csv(cfg["panel"], low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = pd.read_parquet(cfg["labels"])
    n = min(len(panel), len(labels))
    panel, labels = panel.iloc[:n], labels.iloc[:n]
    assert (labels["timestamp"].to_numpy()[:n] == panel["timestamp"].to_numpy()[:n]).all()
    action = labels["trade_outcome_action"].to_numpy()
    tp_moves = labels["tp_move"].to_numpy(dtype=np.float64)
    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    x = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)

    r288 = np.full(len(close), np.nan)
    r288[288:] = close[288:] / close[:-288] - 1.0
    regime = np.full(len(close), 1, dtype=np.int8)
    regime[r288 > 0.04] = 2
    regime[r288 < -0.04] = 0

    train_mask = (ts <= TRAIN_END).to_numpy()
    tr_all = np.flatnonzero(train_mask)
    train_mask[tr_all[-HORIZON_BARS:]] = False
    train_mask &= np.isfinite(tp_moves)
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()
    tr_idx = np.flatnonzero(train_mask)
    v_idx = np.flatnonzero(val_mask)
    subs = np.array_split(tr_idx, 3)

    res = {"n_rows": int(n), "n_features": len(feat_cols),
           "regime_occupancy_train": {REGIME_NAMES[r]: round(float((regime[tr_idx] == r).mean()), 3) for r in range(3)},
           "regime_median_run_bars": float(np.median(runs_of(regime[tr_idx]))),
           "regime_mean_run_bars": round(float(np.mean(runs_of(regime[tr_idx]))), 1)}

    # unconditional sign agreement
    auc_tr = all_feature_auc(x, action, tr_idx)
    auc_v = all_feature_auc(x, action, v_idx)
    dev = np.abs(np.nan_to_num(auc_tr, nan=0.5) - 0.5)
    top = np.argsort(-dev)[:TOP_K]
    res["uncond_top20_sign_agreement"] = round(float(np.mean(np.sign(auc_tr[top] - 0.5) == np.sign(np.nan_to_num(auc_v[top], nan=0.5) - 0.5))), 3)
    res["uncond_max_train_auc_dev"] = round(float(dev.max()), 4)
    res["uncond_auc_spearman_train_val"] = round(float(spearmanr(auc_tr, auc_v, nan_policy="omit").statistic), 3)

    # per-regime
    per_regime = {}
    aucs_tr_by_r = {}
    for r in range(3):
        auc_r_tr = all_feature_auc(x, action, tr_idx[regime[tr_idx] == r])
        auc_r_v = all_feature_auc(x, action, v_idx[regime[v_idx] == r])
        aucs_tr_by_r[r] = auc_r_tr
        dev_r = np.abs(np.nan_to_num(auc_r_tr, nan=0.5) - 0.5)
        top_r = np.argsort(-dev_r)[:TOP_K]
        sub_signs = []
        for s_idx in subs:
            auc_s = all_feature_auc(x, action, s_idx[regime[s_idx] == r])
            sub_signs.append(np.sign(np.nan_to_num(auc_s[top_r], nan=0.5) - 0.5))
        s_tr = np.sign(auc_r_tr[top_r] - 0.5)
        per_regime[REGIME_NAMES[r]] = {
            "mean_top20_train_auc_dev": round(float(dev_r[top_r].mean()), 4),
            "mean_top20_val_auc_dev_signed": round(float(np.nanmean((np.nan_to_num(auc_r_v[top_r], nan=0.5) - 0.5) * s_tr)), 4),
            "sign_agreement_val": round(float(np.mean(s_tr == np.sign(np.nan_to_num(auc_r_v[top_r], nan=0.5) - 0.5))), 3),
            "train_subwindow_stability": round(float(np.mean((np.stack(sub_signs, axis=1) == s_tr[:, None]).all(axis=1))), 3),
            "top3": [str(feat_cols[i]) for i in top_r[:3]],
        }
    res["per_regime"] = per_regime
    finite = np.isfinite(aucs_tr_by_r[0]) & np.isfinite(aucs_tr_by_r[2])
    res["bull_vs_bear_auc_spearman"] = round(float(spearmanr(aucs_tr_by_r[0][finite], aucs_tr_by_r[2][finite]).statistic), 3)
    return res


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {a: analyze(a) for a in ASSETS}
    (OUT_DIR / "comparison.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # comparison chart: the four decisive metrics side by side
    metrics = [
        ("regime_median_run_bars", "Regime median run (bars)", None),
        ("uncond_top20_sign_agreement", "Unconditional top-20\nsign agreement train→VAL", 0.5),
        (("per_regime", "bull", "sign_agreement_val"), "BULL-regime top-20\nsign agreement train→VAL", 0.5),
        (("per_regime", "bull", "mean_top20_val_auc_dev_signed"), "BULL-regime mean VAL AUC edge\n(signed, |AUC-0.5|)", 0.0),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (key, title, refline) in zip(axes, metrics):
        vals = []
        for a in ("SOL", "ETH"):
            v = out[a]
            if isinstance(key, tuple):
                for k in key:
                    v = v[k]
            else:
                v = v[key]
            vals.append(v)
        bars = ax.bar(["SOL", "ETH"], vals, color=[C_SOL, C_ETH], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f" {v:g}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=10, color=INK)
        if refline is not None:
            ax.axhline(refline, color=INK, alpha=0.4, linewidth=1, linestyle="--")
        ax.set_title(title, fontsize=10, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("ETH vs SOL — regime/feature structure on the identical pipeline", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eth_sol_comparison.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_DIR/'eth_sol_comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
