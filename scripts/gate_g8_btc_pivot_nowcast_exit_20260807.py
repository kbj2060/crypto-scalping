"""Gate G8 -- can a real (imperfect) model nowcast an adverse zigzag pivot to within <=3 bars, and
does it recover a positive fraction of G1's exit ceiling?

G1 measured a large exit ceiling (perfect pivot exit: VAL -22.5% -> +39.0%, OOS -9.5% -> +59.9%,
t=5.4/5.7, MDD roughly halved) but a brutal timing requirement: exiting k bars early collapses the
gain, and both splits are positive only up to k<=3 bars. G4c closed the entry axis, so this is the
last open hypothesis in the design doc.

One structural reason to expect this axis to behave differently from the entry problem: the exit
label's event window is a handful of bars, not the triple-barrier label's median 51 / p90 189, so
label overlap is mild and effective sample size is a large multiple of the ~4,076 that bounded
every entry experiment. Measured and reported below rather than assumed.

Setup:
- Target: 3-class {0 = no adverse pivot soon, 1 = H pivot within [t, t+k], 2 = L pivot within
  [t, t+k]}, from the same oracle pivots G1's P2/P3/P4 policies used. An H pivot ends an up-wave
  (adverse for a LONG); an L pivot ends a down-wave (adverse for a SHORT).
- Features: causalfix_final's 113 causal columns plus the 4 causal pivot-tracker features from
  scripts/build_btc_5m_zigzag_state_causal_features_20260806.py -- `zz_dist_to_threshold` in
  particular is the live "how close is a reversal to confirming" signal, which is exactly the
  quantity this head needs and which the entry models had no use for.
- Policy: exit when P(adverse pivot for the held side) > threshold, profit-gated exactly like G1's
  P4 (a position already in profit past the roundtrip cost), on the SAME entry ledger produced by
  the best triple-barrier checkpoint, through G1's own simulator (imported, not reimplemented, so
  the numbers are directly comparable).
- Threshold is fitted on VAL and applied unchanged to OOS.

Verdict metric -- fraction of the reachable ceiling captured:

    capture = (model_policy_sumret - baseline_sumret) / (perfect_k_sumret - baseline_sumret)

where perfect_k is G1's P4 profit-gated policy at the same k. G8 passes if OOS capture is
meaningfully positive with a VAL-fitted threshold, across seeds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ensemble.deep_features.btc_deepfeat_dataset_20260806 import build_dataset  # noqa: E402
from ensemble.deep_features.btc_deepfeat_encoders_20260806 import build_model  # noqa: E402
from gate_g1_btc_exit_oracle_pivot_rescue_20260807 import (  # noqa: E402
    _fresh_entry_mask, _simulate, BREAKEVEN_PRICE_MOVE,
)

CHECKPOINT = ROOT / "tmp/btc_deepfeat_tripbarrier_20260806/flatsmooth_cw_0.9/deepfeat_bundle.pt"
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_flatsmooth_20260806.parquet"
PIVOT_PATH = ROOT / "data/splits/year_oos/btc_5m_pivot_transition_labels_20260806.parquet"
ZZSTATE_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_state_causal_features_20260806.parquet"
OUT_DIR = ROOT / "tmp/btc_gate_g8_pivot_nowcast_20260807"

CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT, HORIZON_BARS = 12, 288, 2.5, 1.2, 288
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")
LEAD_KS = (1, 2, 3)
SEEDS = [11, 137, 2029, 40507, 918273]
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)

XGB_PARAMS = dict(objective="multi:softprob", num_class=3, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20, reg_lambda=2.0,
                  tree_method="hist", eval_metric="mlogloss")
NUM_ROUNDS, EARLY_STOP = 600, 40


@torch.no_grad()
def _predict_entry(model, ds, split, device, batch_size=1024):
    model.eval()
    row_idx = ds.end_idx[split]
    out = []
    for i in range(0, len(row_idx), batch_size):
        x = torch.from_numpy(ds.get_batch(row_idx[i : i + batch_size])).to(device)
        logits, _, _ = model(x)
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _pivot_soon_labels(is_h, is_l, k):
    """3-class nowcast target: nearest adverse pivot within [t, t+k]; 0 when neither side has one."""
    n = len(is_h)
    y = np.zeros(n, dtype=np.int64)
    h_idx = np.flatnonzero(is_h)
    l_idx = np.flatnonzero(is_l)
    dist_h = np.full(n, np.inf)
    dist_l = np.full(n, np.inf)
    for arr, dist in ((h_idx, dist_h), (l_idx, dist_l)):
        for p in arr:
            lo = max(0, p - k)
            d = p - np.arange(lo, p + 1)
            dist[lo : p + 1] = np.minimum(dist[lo : p + 1], d)
    y[np.isfinite(dist_h) & (dist_h <= dist_l)] = 1
    y[np.isfinite(dist_l) & (dist_l < dist_h)] = 2
    return y


def _summ(ledger, equity_marks):
    if len(ledger) == 0:
        return {"n_trades": 0, "sum_ret_pct": 0.0}
    rets = ledger["trade_return"].to_numpy(dtype=np.float64)
    equity = np.concatenate([[1.0], equity_marks])
    rm = np.maximum.accumulate(equity)
    return {"n_trades": int(len(rets)), "win_rate": float((rets > 0).mean()),
            "sum_ret_pct": float(rets.sum() * 100.0), "final_equity": float(equity[-1]),
            "trade_mdd_pct": float(((equity - rm) / rm).min() * 100.0),
            "median_bars_held": float(ledger["bars_held"].median())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    panel = pd.read_parquet(PANEL_PATH).sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"].to_numpy()
    n = len(panel)
    open_ = panel["open"].to_numpy(dtype=np.float64)
    high = panel["high"].to_numpy(dtype=np.float64)
    low = panel["low"].to_numpy(dtype=np.float64)
    close = panel["close"].to_numpy(dtype=np.float64)

    log_ret = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
    tp_all, sl_all = TP_MULT * vol, SL_MULT * vol

    piv = pd.read_parquet(PIVOT_PATH, columns=["timestamp", "is_pivot", "pivot_type"])
    piv = piv.sort_values("timestamp").reset_index(drop=True)
    if not (piv["timestamp"].to_numpy() == ts).all():
        raise RuntimeError("pivot timestamps don't match the panel")
    is_h = ((piv["is_pivot"] == 1) & (piv["pivot_type"] == "H")).to_numpy()
    is_l = ((piv["is_pivot"] == 1) & (piv["pivot_type"] == "L")).to_numpy()

    zz = pd.read_parquet(ZZSTATE_PATH).sort_values("timestamp").reset_index(drop=True)
    if not (zz["timestamp"].to_numpy() == ts).all():
        raise RuntimeError("zigzag state timestamps don't match the panel")
    zz_cols = [c for c in zz.columns if c != "timestamp"]

    base_cols = [c for c in panel.columns if c not in ("timestamp",)]
    X = np.concatenate([panel[base_cols].to_numpy(dtype=np.float32),
                        zz[zz_cols].to_numpy(dtype=np.float32)], axis=1)
    feat_names = base_cols + zz_cols

    # ---- entry ledger from the best triple-barrier checkpoint (same as G1) ----
    bundle = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = bundle["config"]
    ds = build_dataset(window=cfg["window"], label_path=LABEL_PATH, hard_col="trade_outcome_action",
                       soft_cols=["trade_outcome_soft_cash", "trade_outcome_soft_long",
                                  "trade_outcome_soft_short"])
    entry_model = build_model(
        cfg["arch"], cfg["n_features"], cfg["category_sizes"], embed_dim=cfg["embed_dim"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        ffn_mult=cfg["ffn_mult"], dropout=cfg["dropout"], quality_head=cfg["quality_head"],
        head_type=cfg.get("head_type", "linear")).to(device)
    entry_model.load_state_dict(bundle["model_state"])

    entries = {}
    for split in ("val", "oos"):
        probs = _predict_entry(entry_model, ds, split, device)
        pred = probs.argmax(axis=1)
        side_state = np.where(pred == 1, 1, np.where(pred == 2, -1, 0))
        row_idx = ds.end_idx[split]
        fresh = _fresh_entry_mask(side_state)
        entries[split] = (row_idx[fresh], side_state[fresh])

    no_exit = np.zeros((2, n), dtype=bool)
    train_mask = (ts < np.datetime64(VAL_START))
    val_mask = (ts >= np.datetime64(VAL_START)) & (ts <= np.datetime64(VAL_END))
    oos_mask = (ts >= np.datetime64(OOS_START)) & (ts <= np.datetime64(OOS_END))
    finite_row = np.isfinite(X).all(axis=1)

    results, label_stats = [], []
    for k in LEAD_KS:
        y = _pivot_soon_labels(is_h, is_l, k)
        # purge: the nowcast label at bar t is determined by bars up to t+k only
        tr = np.flatnonzero(train_mask & finite_row)
        tr = tr[tr + k < np.searchsorted(ts, np.datetime64(VAL_START))]
        va = np.flatnonzero(val_mask & finite_row)
        oo = np.flatnonzero(oos_mask & finite_row)
        label_stats.append({
            "k": k, "n_train": int(len(tr)),
            "positive_rate_H": float((y[tr] == 1).mean()), "positive_rate_L": float((y[tr] == 2).mean()),
            # label window is k+1 bars, so overlap is mild: effective_n ~ n_train/(k+1)
            "approx_effective_n": float(len(tr) / (k + 1)),
        })
        print(f"[k={k}] train={len(tr)} posH={label_stats[-1]['positive_rate_H']:.4f} "
              f"posL={label_stats[-1]['positive_rate_L']:.4f} "
              f"approx_effective_n={label_stats[-1]['approx_effective_n']:.0f}", flush=True)

        dtr = xgb.DMatrix(X[tr], label=y[tr], feature_names=feat_names)
        dva = xgb.DMatrix(X[va], label=y[va], feature_names=feat_names)
        doo = xgb.DMatrix(X[oo], label=y[oo], feature_names=feat_names)

        # reference points: baseline (no exit policy) and PERFECT profit-gated lead-k (G1's P4)
        perfect = np.zeros((2, n), dtype=bool)
        for r, arr in ((0, is_h), (1, is_l)):
            for p in np.flatnonzero(arr):
                perfect[r, max(0, p - k) : p + 1] = True

        refs = {}
        for split, (d_idx, d_side) in entries.items():
            led, eq = _simulate(d_idx, d_side, tp_all[d_idx], sl_all[d_idx], open_, high, low, close, no_exit)
            refs[(split, "baseline")] = _summ(led, eq)
            led, eq = _simulate(d_idx, d_side, tp_all[d_idx], sl_all[d_idx], open_, high, low, close,
                                perfect, profit_gated=True)
            refs[(split, "perfect")] = _summ(led, eq)

        for seed in seeds:
            bst = xgb.train(dict(XGB_PARAMS, seed=seed), dtr, NUM_ROUNDS, evals=[(dva, "val")],
                            early_stopping_rounds=EARLY_STOP, verbose_eval=False)
            it = (0, bst.best_iteration + 1)
            proba = {"val": bst.predict(dva, iteration_range=it), "oos": bst.predict(doo, iteration_range=it)}
            rows_idx = {"val": va, "oos": oo}

            # fit the exit threshold on VAL, then apply it unchanged to OOS
            best_thr, best_val = None, -np.inf
            val_curve = []
            for thr in THRESHOLDS:
                forced = np.zeros((2, n), dtype=bool)
                forced[0, rows_idx["val"]] = proba["val"][:, 1] > thr
                forced[1, rows_idx["val"]] = proba["val"][:, 2] > thr
                d_idx, d_side = entries["val"]
                led, eq = _simulate(d_idx, d_side, tp_all[d_idx], sl_all[d_idx], open_, high, low,
                                    close, forced, profit_gated=True)
                sr = _summ(led, eq)["sum_ret_pct"]
                val_curve.append({"threshold": float(thr), "val_sum_ret_pct": sr})
                if sr > best_val:
                    best_val, best_thr = sr, float(thr)

            row = {"k": k, "seed": seed, "best_iter": int(bst.best_iteration),
                   "fitted_threshold": best_thr, "val_curve": val_curve}
            # classification diagnostics: separates "the classifier learned nothing" from "it
            # learned something the exit policy cannot monetise"
            for split, ix in rows_idx.items():
                yt = y[ix]
                for cls, name in ((1, "H"), (2, "L")):
                    p_cls = proba[split][:, cls]
                    pos = yt == cls
                    if pos.any() and (~pos).any():
                        order = np.argsort(p_cls)
                        ranks = np.empty(len(p_cls), dtype=np.float64)
                        ranks[order] = np.arange(1, len(p_cls) + 1)
                        n1, n0 = int(pos.sum()), int((~pos).sum())
                        row[f"{split}_auc_{name}"] = float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
                        row[f"{split}_base_rate_{name}"] = float(pos.mean())
                        sel = p_cls > best_thr
                        row[f"{split}_prec_{name}"] = float(pos[sel].mean()) if sel.any() else float("nan")
                        row[f"{split}_flagrate_{name}"] = float(sel.mean())
            for split in ("val", "oos"):
                forced = np.zeros((2, n), dtype=bool)
                forced[0, rows_idx[split]] = proba[split][:, 1] > best_thr
                forced[1, rows_idx[split]] = proba[split][:, 2] > best_thr
                d_idx, d_side = entries[split]
                led, eq = _simulate(d_idx, d_side, tp_all[d_idx], sl_all[d_idx], open_, high, low,
                                    close, forced, profit_gated=True)
                m = _summ(led, eq)
                b = refs[(split, "baseline")]["sum_ret_pct"]
                p = refs[(split, "perfect")]["sum_ret_pct"]
                row[f"{split}_sum_ret_pct"] = m["sum_ret_pct"]
                row[f"{split}_win_rate"] = m.get("win_rate")
                row[f"{split}_mdd_pct"] = m.get("trade_mdd_pct")
                row[f"{split}_n_policy_exits"] = int(sum(r == "policy" for r in led["reason"])) if len(led) else 0
                row[f"{split}_baseline_sum_ret_pct"] = b
                row[f"{split}_perfect_sum_ret_pct"] = p
                row[f"{split}_capture"] = (m["sum_ret_pct"] - b) / (p - b) if abs(p - b) > 1e-9 else np.nan
            results.append(row)
            print(f"  k={k} seed={seed} thr={best_thr:.2f} "
                  f"VAL {row['val_sum_ret_pct']:+.1f}% (cap {row['val_capture']:+.3f}) | "
                  f"OOS {row['oos_sum_ret_pct']:+.1f}% (cap {row['oos_capture']:+.3f}) "
                  f"[base {row['oos_baseline_sum_ret_pct']:+.1f}, perfect {row['oos_perfect_sum_ret_pct']:+.1f}]",
                  flush=True)

    df = pd.DataFrame([{c: v for c, v in r.items() if c != "val_curve"} for r in results])
    df.to_csv(OUT_DIR / "g8_per_seed.csv", index=False)

    print("\n=== capture of the reachable exit ceiling (threshold fitted on VAL) ===")
    hdr = f"{'k':>3}{'OOS base%':>11}{'OOS perfect%':>14}{'OOS model% mean':>17}{'OOS capture mean':>19}{'sd':>8}{'>0 seeds':>10}"
    print(hdr)
    print("-" * len(hdr))
    for k in LEAD_KS:
        d = df[df["k"] == k]
        print(f"{k:>3}{d['oos_baseline_sum_ret_pct'].iloc[0]:>11.1f}"
              f"{d['oos_perfect_sum_ret_pct'].iloc[0]:>14.1f}{d['oos_sum_ret_pct'].mean():>17.1f}"
              f"{d['oos_capture'].mean():>19.3f}{d['oos_capture'].std():>8.3f}"
              f"{int((d['oos_capture'] > 0).sum()):>7}/{len(d)}")

    print("\n=== VAL (in-sample for the threshold) ===")
    for k in LEAD_KS:
        d = df[df["k"] == k]
        print(f"  k={k}: VAL model {d['val_sum_ret_pct'].mean():+.1f}% "
              f"(base {d['val_baseline_sum_ret_pct'].iloc[0]:+.1f}, "
              f"perfect {d['val_perfect_sum_ret_pct'].iloc[0]:+.1f}), capture {d['val_capture'].mean():+.3f}")

    print("\n=== classifier quality (did it learn the pivot at all?) ===")
    for k in LEAD_KS:
        d = df[df["k"] == k]
        print(f"  k={k}: OOS AUC H {d['oos_auc_H'].mean():.3f} / L {d['oos_auc_L'].mean():.3f}  |  "
              f"precision@thr H {d['oos_prec_H'].mean():.3f} / L {d['oos_prec_L'].mean():.3f}  "
              f"(base rate H {d['oos_base_rate_H'].mean():.3f} / L {d['oos_base_rate_L'].mean():.3f}, "
              f"flag rate H {d['oos_flagrate_H'].mean():.3f})")

    print("\n=== label sample-size contrast with the entry problem ===")
    for s in label_stats:
        print(f"  k={s['k']}: approx effective_n {s['approx_effective_n']:.0f} "
              f"(entry-label effective_n was ~4,076)")

    (OUT_DIR / "g8_summary.json").write_text(json.dumps(
        {"xgb_params": XGB_PARAMS, "seeds": seeds, "lead_ks": list(LEAD_KS),
         "thresholds": THRESHOLDS.tolist(), "label_stats": label_stats, "runs": results},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
