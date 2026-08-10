"""Rev8: per-regime feature distributions + per-regime oracle-matched feature filtering
(docs/experiments/sol_dl_rl_architecture_survey_rev8_regime_feature_filter_20260808.json).

Stage `analysis` (train-only, plus VAL sign check for reporting):
  - per-regime feature distribution separation: max pairwise two-sample KS statistic across the
    three D2 regimes (which features SHIFT with regime);
  - per-regime univariate direction AUC (oracle LONG vs SHORT) with train sub-window and VAL
    signs; bull-vs-bear train AUC correlation (how opposed the regimes are);
  - charts: per-regime densities of the 8 most regime-separated features, and a dot plot of
    per-regime direction AUC for each regime's top features.

Stage `val` / `oos`: per-regime LGBM experts on each regime's TRAIN-selected top-K features,
grid K in {10,20} x chop policy {expert, force-cash} x 6 entry rules; gates per the contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp, rankdata, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    RAW_LEVEL_COLS, PANEL_PATH, LABEL_PATH, SEED, HORIZON_BARS, ENTRY_RULES,
    TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END, replay, side_state_from_proba,
)

OUT_DIR = ROOT / "tmp/sol_dl_rl_survey_20260807/regime_feature_filter_rev8"
K_GRID = [10, 20]
CHOP_POLICIES = ["expert", "force_cash"]
CONTROL_VAL_PNL = -6.90
REGIME_NAMES = ["bear", "chop", "bull"]
C_BULL, C_BEAR, C_CHOP = "#2563EB", "#D9542B", "#9AA0A6"
RCOLOR = {"bear": C_BEAR, "chop": C_CHOP, "bull": C_BULL}
INK = "#1F2430"


def auc_binary(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x)
    x, y = x[m], y[m]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 < 50 or n0 < 50:
        return np.nan
    r = rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def d2_regime(panel: pd.DataFrame) -> np.ndarray:
    close = panel["close"].to_numpy(dtype=np.float64)
    r = np.full(len(close), np.nan)
    r[288:] = close[288:] / close[:-288] - 1.0
    reg = np.full(len(close), 1, dtype=np.int8)
    reg[r > 0.04] = 2
    reg[r < -0.04] = 0
    return reg


def load_all():
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = pd.read_parquet(LABEL_PATH)
    action = labels["trade_outcome_action"].to_numpy()
    tp_moves = labels["tp_move"].to_numpy(dtype=np.float64)
    sl_moves = labels["sl_move"].to_numpy(dtype=np.float64)
    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    x = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    ts = panel["timestamp"]
    train_mask = (ts <= TRAIN_END).to_numpy()
    purge_cut = np.flatnonzero(train_mask)[-HORIZON_BARS:]
    train_mask[purge_cut] = False
    train_mask &= np.isfinite(tp_moves)
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()
    oos_mask = ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()
    return panel, ts, x, feat_cols, action, tp_moves, sl_moves, train_mask, val_mask, oos_mask


def per_regime_auc(x, action, idx, regime, r):
    sub = idx[regime[idx] == r]
    a = action[sub]
    nz = a != 0
    out = np.full(x.shape[1], np.nan)
    if nz.sum() < 200:
        return out
    y = (a[nz] == 1).astype(int)
    for f in range(x.shape[1]):
        out[f] = auc_binary(x[sub, f][nz].astype(np.float64), y)
    return out


def select_topk(auc_train_by_regime, k):
    tops = {}
    for r in range(3):
        dev = np.abs(np.nan_to_num(auc_train_by_regime[r], nan=0.5) - 0.5)
        tops[r] = np.argsort(-dev)[:k]
    return tops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["analysis", "val", "oos"], required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel, ts, x, feat_cols, action, tp_moves, sl_moves, train_mask, val_mask, oos_mask = load_all()
    regime = d2_regime(panel)
    tr_idx = np.flatnonzero(train_mask)
    v_idx = np.flatnonzero(val_mask)

    auc_train = {r: per_regime_auc(x, action, tr_idx, regime, r) for r in range(3)}

    if args.stage == "analysis":
        # --- distribution separation across regimes (train) ---
        rng = np.random.default_rng(SEED)
        sample = {r: tr_idx[regime[tr_idx] == r] for r in range(3)}
        sample = {r: rng.choice(s, size=min(len(s), 8000), replace=False) for r, s in sample.items()}
        ks_max = np.zeros(len(feat_cols))
        for f in range(len(feat_cols)):
            cols = {r: x[sample[r], f][np.isfinite(x[sample[r], f])] for r in range(3)}
            pairs = [(0, 1), (0, 2), (1, 2)]
            ks_max[f] = max(ks_2samp(cols[a], cols[b]).statistic if len(cols[a]) > 200 and len(cols[b]) > 200 else 0.0 for a, b in pairs)
        order = np.argsort(-ks_max)
        dist_table = [{"feature": feat_cols[i], "max_pairwise_ks": round(float(ks_max[i]), 3)} for i in order[:15]]
        invariant = [{"feature": feat_cols[i], "max_pairwise_ks": round(float(ks_max[i]), 3)} for i in order[-5:]]

        # --- per-regime AUC relationships ---
        auc_val = {r: per_regime_auc(x, action, v_idx, regime, r) for r in range(3)}
        finite = np.isfinite(auc_train[0]) & np.isfinite(auc_train[2])
        bull_bear_rho = float(spearmanr(auc_train[0][finite], auc_train[2][finite]).statistic)
        tops = select_topk(auc_train, 10)
        overlap_bb = len(set(tops[0]) & set(tops[2]))
        top_lists = {}
        for r in range(3):
            rows = []
            for f in tops[r]:
                rows.append({
                    "feature": feat_cols[f],
                    "train_auc": round(float(auc_train[r][f]), 4),
                    "val_auc": round(float(auc_val[r][f]), 4) if np.isfinite(auc_val[r][f]) else None,
                    "sign_holds_val": bool(np.sign(auc_train[r][f] - 0.5) == np.sign((auc_val[r][f] or 0.5) - 0.5)) if np.isfinite(auc_val[r][f]) else None,
                })
            top_lists[REGIME_NAMES[r]] = rows

        summary = {
            "most_regime_separated_features": dist_table,
            "most_regime_invariant_features": invariant,
            "bull_vs_bear_train_auc_spearman": round(bull_bear_rho, 3),
            "top10_overlap_bull_bear": overlap_bb,
            "per_regime_top10_direction_features": top_lists,
        }
        (OUT_DIR / "analysis.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: summary[k] for k in ("bull_vs_bear_train_auc_spearman", "top10_overlap_bull_bear")}, indent=2))
        print(json.dumps(dist_table[:8], indent=1))

        # --- chart 1: densities of 8 most regime-separated features ---
        fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
        for ax, f in zip(axes.ravel(), order[:8]):
            for r in range(3):
                vals = x[sample[r], f]
                vals = vals[np.isfinite(vals)]
                lo, hi = np.nanpercentile(vals, [1, 99])
                vals = vals[(vals >= lo) & (vals <= hi)]
                ax.hist(vals, bins=60, density=True, histtype="step", linewidth=1.6,
                        color=RCOLOR[REGIME_NAMES[r]], label=REGIME_NAMES[r])
            ax.set_title(f"{feat_cols[f]}  (KS {ks_max[f]:.2f})", fontsize=9, color=INK)
            ax.set_yticks([])
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
        axes[0, 0].legend(frameon=False, fontsize=9)
        fig.suptitle("SOL train: most regime-separated feature distributions (D2 regimes)", x=0.01, ha="left", fontsize=13, color=INK)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "regime_feature_distributions.png", dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # --- chart 2: per-regime top-10 direction AUC, train vs VAL ---
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True)
        for ax, r in zip(axes, (2, 0, 1)):  # bull, bear, chop
            rows = top_lists[REGIME_NAMES[r]]
            ypos = np.arange(len(rows))[::-1]
            ax.axvline(0.5, color=INK, alpha=0.35, linewidth=1)
            for yy, rec in zip(ypos, rows):
                col = RCOLOR[REGIME_NAMES[r]]
                ax.plot([rec["train_auc"]], [yy], "o", color=col, markersize=8, label="_")
                if rec["val_auc"] is not None:
                    ax.plot([rec["val_auc"]], [yy], "o", markerfacecolor="white", markeredgecolor=col, markersize=8)
                    ax.plot([rec["train_auc"], rec["val_auc"]], [yy, yy], color=col, alpha=0.5, linewidth=1.4)
            ax.set_yticks(ypos)
            ax.set_yticklabels([r_["feature"] for r_ in rows], fontsize=8)
            ax.set_title(f"{REGIME_NAMES[r]} — filled=train, hollow=VAL", fontsize=11, color=INK)
            ax.set_xlim(0.38, 0.62)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        fig.suptitle("Per-regime top-10 direction features: LONG-vs-SHORT AUC, train vs VAL (0.5 = no signal)", x=0.01, ha="left", fontsize=13, color=INK)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "regime_direction_auc.png", dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote charts to {OUT_DIR}")
        return 0

    if args.stage == "val":
        table = []
        for k in K_GRID:
            tops = select_topk(auc_train, k)
            experts = {}
            for r in range(3):
                rows = tr_idx[regime[tr_idx] == r]
                cols = tops[r]
                clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=600, learning_rate=0.05,
                                         num_leaves=63, min_child_samples=200, feature_fraction=0.8,
                                         bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                                         random_state=SEED + r, n_jobs=-1, verbosity=-1)
                clf.fit(x[rows][:, cols], action[rows])
                clf.booster_.save_model(str(OUT_DIR / f"expert_k{k}_{REGIME_NAMES[r]}.txt"))
                experts[r] = (clf.booster_, cols)
            proba = np.zeros((len(panel), 3))
            for r in range(3):
                sub = v_idx[regime[v_idx] == r]
                if len(sub):
                    proba[sub] = experts[r][0].predict(x[sub][:, experts[r][1]])
            months = ts.dt.to_period("M").astype(str).to_numpy()
            for chop_policy in CHOP_POLICIES:
                for rule in ENTRY_RULES:
                    side_state = np.zeros(len(panel), dtype=np.int64)
                    side_state[v_idx] = side_state_from_proba(proba[v_idx], rule["threshold"])
                    if chop_policy == "force_cash":
                        side_state[v_idx[regime[v_idx] == 1]] = 0
                    rres = replay(panel, side_state, tp_moves, sl_moves, val_mask)
                    mon = {}
                    for m in sorted(set(months[v_idx])):
                        mon[m] = replay(panel, side_state, tp_moves, sl_moves, val_mask & (months == m)).get("pnl_pct", 0.0)
                    n_pos_m = sum(v_ > 0 for v_ in mon.values())
                    rec = {"k": k, "chop_policy": chop_policy, "rule": rule["name"], "threshold": rule["threshold"],
                           **{kk: rres.get(kk) for kk in ("n_trades", "pnl_pct", "win_rate", "mdd_pct")},
                           "monthly": mon, "n_pos_months": int(n_pos_m)}
                    table.append(rec)
                    print(json.dumps({kk: rec[kk] for kk in ("k", "chop_policy", "rule", "n_trades", "pnl_pct", "n_pos_months")}), flush=True)
        eligible = [r_ for r_ in table if (r_["n_trades"] or 0) >= 15 and (r_["pnl_pct"] or 0) > 0 and r_["n_pos_months"] >= 3 and (r_["pnl_pct"] or 0) > CONTROL_VAL_PNL]
        best = max(eligible, key=lambda r_: r_["pnl_pct"]) if eligible else None
        out = {"stage": "val", "table": table,
               "selected": None if best is None else {kk: best[kk] for kk in ("k", "chop_policy", "rule", "threshold", "pnl_pct", "n_trades", "mdd_pct", "n_pos_months")},
               "earns_oos_read": best is not None}
        (OUT_DIR / "val_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"selected": out["selected"], "earns_oos_read": out["earns_oos_read"]}, indent=2))
        return 0

    prior = json.loads((OUT_DIR / "val_results.json").read_text())
    if not prior.get("earns_oos_read"):
        print(json.dumps({"oos": "REFUSED -- rev8 VAL gate failed"}))
        return 1
    sel = prior["selected"]
    k = sel["k"]
    tops = select_topk(auc_train, k)
    o_idx = np.flatnonzero(oos_mask)
    proba = np.zeros((len(panel), 3))
    for r in range(3):
        booster = lgb.Booster(model_file=str(OUT_DIR / f"expert_k{k}_{REGIME_NAMES[r]}.txt"))
        sub = o_idx[regime[o_idx] == r]
        if len(sub):
            proba[sub] = booster.predict(x[sub][:, tops[r]])
    side_state = np.zeros(len(panel), dtype=np.int64)
    side_state[o_idx] = side_state_from_proba(proba[o_idx], sel["threshold"])
    if sel["chop_policy"] == "force_cash":
        side_state[o_idx[regime[o_idx] == 1]] = 0
    rres = replay(panel, side_state, tp_moves, sl_moves, oos_mask)
    out = {"stage": "oos", "selected": sel, **rres, "adopted": bool((rres.get("pnl_pct") or 0) > 0)}
    (OUT_DIR / "oos_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
