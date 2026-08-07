"""New execution primitives after G8 -- P1 (marginal-EV diagnostic) and P2 (pivot-triggered entry).

G8 produced a genuinely strong adverse-pivot nowcaster (OOS AUC 0.95, 7-20x precision lift) that
captured ~none of G1's exit ceiling because recall at any usable threshold is 2.5-14.5%. Before
building a position-sizing-aware simulator to exploit it with partial exits, note that partial
exits cannot rescue a bad operating point: exiting a fraction f at a flagged bar has

    EV = f * [ p * (loss avoided) - (1 - p) * (gain forfeited) ]

which is LINEAR in f, so f scales the magnitude and never the sign. The prior question is whether
ANY exit operating point has positive marginal EV at all. That is P1, and it is policy-independent
(full exit, partial exit and scale-out all inherit its sign).

P1 -- marginal-EV diagnostic. For every bar j inside an open position from the baseline
triple-barrier entry ledger, compare the counterfactual of closing at bar j's close against the
trade's actual realised outcome. The roundtrip cost is identical either way and cancels, so

    marginal(j) = (unrealised move at j - realised move of the trade) * notional

Bucketing marginal(j) by the nowcaster's probability for the held side's adverse pivot answers
directly: conditional on the model shouting "reversal", is closing better than holding? If no
probability bucket is positive, every exit primitive is dead and no simulator work is justified.

P2 -- pivot-triggered ENTRY. A flagged L pivot is a bottom (enter LONG); a flagged H pivot is a top
(enter SHORT). This is NOT the entry axis G4c closed: that axis predicted the triple-barrier label
("would a trade opened now win"), whereas this trades a detected turning point. Same barriers, same
cost, same simulator, threshold fitted on VAL and applied unchanged to OOS.

Both analyses reuse the G8 nowcaster (k=1, the best-AUC setting) retrained here over 5 seeds.
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

from core.causal_futures_backtest import simulate_single_position  # noqa: E402
from ensemble.deep_features.btc_deepfeat_dataset_20260806 import build_dataset  # noqa: E402
from ensemble.deep_features.btc_deepfeat_encoders_20260806 import build_model  # noqa: E402
from gate_g1_btc_exit_oracle_pivot_rescue_20260807 import _fresh_entry_mask, _simulate  # noqa: E402
from gate_g8_btc_pivot_nowcast_exit_20260807 import (  # noqa: E402
    CHECKPOINT, LABEL_PATH, PANEL_PATH, PIVOT_PATH, ZZSTATE_PATH, XGB_PARAMS,
    NUM_ROUNDS, EARLY_STOP, _pivot_soon_labels, _predict_entry,
)

OUT_DIR = ROOT / "tmp/btc_primitive_p1p2_20260807"
CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT, HORIZON_BARS = 12, 288, 2.5, 1.2, 288
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010
NOTIONAL = MARGIN_FRACTION * LEVERAGE
ACCOUNT_COST = ROUNDTRIP_COST_RATE * NOTIONAL
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")
K = 1
SEEDS = [11, 137, 2029, 40507, 918273]
PROB_BINS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.01]
ENTRY_THRESHOLDS = np.round(np.arange(0.10, 0.86, 0.05), 2)


def _marginal_ev(ledger, proba_row, side_col, close, open_, prob_h, prob_l):
    """P1: per-bar counterfactual of closing now vs holding, tagged with the adverse-pivot prob."""
    recs = []
    for _, tr in ledger.iterrows():
        e, x, side = int(tr["entry_i"]), int(tr["exit_i"]), int(tr["side"])
        realised = float(tr["price_move"])
        p_adverse = prob_h if side > 0 else prob_l
        for j in range(e, x):  # exclude the exit bar itself: closing there is the actual outcome
            if not np.isfinite(p_adverse[j]):
                continue
            entry_px = open_[e]
            unreal = (close[j] / entry_px - 1.0) if side > 0 else (1.0 - close[j] / entry_px)
            recs.append({"prob": float(p_adverse[j]),
                         "marginal": (unreal - realised) * NOTIONAL,
                         "in_profit": unreal > ROUNDTRIP_COST_RATE})
    return pd.DataFrame(recs)


def _summ_bt(result):
    if result is None or len(result.ledger) == 0:
        return {"n_trades": 0}
    rets = result.ledger["trade_return"].to_numpy(dtype=np.float64)
    eq = result.equity
    rm = np.maximum.accumulate(eq)
    gross = float((rets.mean() + ACCOUNT_COST) * 10000.0)
    sd = float(rets.std(ddof=1) * 10000.0) if len(rets) > 1 else float("nan")
    return {"n_trades": int(len(rets)), "win_rate": float((rets > 0).mean()),
            "gross_bps": gross, "t_gross": gross / (sd / np.sqrt(len(rets))) if len(rets) > 1 else np.nan,
            "sum_ret_pct": float(rets.sum() * 100.0), "final_equity": float(eq[-1]),
            "mdd_pct": float(((eq - rm) / rm).min() * 100.0),
            "long_trades": int((result.ledger["side"] == 1).sum()),
            "short_trades": int((result.ledger["side"] == -1).sum())}


def _entry_backtest(decision_idx, sides, tp_all, sl_all, panel):
    if len(decision_idx) == 0:
        return None
    tp, sl = tp_all[decision_idx], sl_all[decision_idx]
    ok = np.isfinite(tp) & np.isfinite(sl)
    idx, sd, tp, sl = decision_idx[ok], sides[ok], tp[ok], sl[ok]
    if len(idx) == 0:
        return None
    return simulate_single_position(
        timestamps=panel["timestamp"], open_px=panel["open"].to_numpy(dtype=np.float64),
        high=panel["high"].to_numpy(dtype=np.float64), low=panel["low"].to_numpy(dtype=np.float64),
        close=panel["close"].to_numpy(dtype=np.float64), decision_indices=idx,
        scores=sd.astype(np.float64), tp_moves=tp, sl_moves=sl, upper_threshold=0.0,
        lower_threshold=0.0, horizon_bars=HORIZON_BARS, margin_fraction=MARGIN_FRACTION,
        leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE)


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
    is_h = ((piv["is_pivot"] == 1) & (piv["pivot_type"] == "H")).to_numpy()
    is_l = ((piv["is_pivot"] == 1) & (piv["pivot_type"] == "L")).to_numpy()
    zz = pd.read_parquet(ZZSTATE_PATH).sort_values("timestamp").reset_index(drop=True)
    zz_cols = [c for c in zz.columns if c != "timestamp"]
    base_cols = [c for c in panel.columns if c != "timestamp"]
    X = np.concatenate([panel[base_cols].to_numpy(dtype=np.float32),
                        zz[zz_cols].to_numpy(dtype=np.float32)], axis=1)
    finite_row = np.isfinite(X).all(axis=1)

    y = _pivot_soon_labels(is_h, is_l, K)
    val_start_i = int(np.searchsorted(ts, np.datetime64(VAL_START)))
    tr = np.flatnonzero((ts < np.datetime64(VAL_START)) & finite_row)
    tr = tr[tr + K < val_start_i]
    va = np.flatnonzero((ts >= np.datetime64(VAL_START)) & (ts <= np.datetime64(VAL_END)) & finite_row)
    oo = np.flatnonzero((ts >= np.datetime64(OOS_START)) & (ts <= np.datetime64(OOS_END)) & finite_row)
    dtr, dva, doo = (xgb.DMatrix(X[i], label=y[i]) for i in (tr, va, oo))

    # baseline triple-barrier entry ledger (same object G1/G8 measured against)
    bundle = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = bundle["config"]
    ds = build_dataset(window=cfg["window"], label_path=LABEL_PATH, hard_col="trade_outcome_action",
                       soft_cols=["trade_outcome_soft_cash", "trade_outcome_soft_long",
                                  "trade_outcome_soft_short"])
    em = build_model(cfg["arch"], cfg["n_features"], cfg["category_sizes"], embed_dim=cfg["embed_dim"],
                     d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                     ffn_mult=cfg["ffn_mult"], dropout=cfg["dropout"], quality_head=cfg["quality_head"],
                     head_type=cfg.get("head_type", "linear")).to(device)
    em.load_state_dict(bundle["model_state"])
    no_exit = np.zeros((2, n), dtype=bool)
    base_ledgers = {}
    for split in ("val", "oos"):
        probs = _predict_entry(em, ds, split, device)
        pred = probs.argmax(axis=1)
        side_state = np.where(pred == 1, 1, np.where(pred == 2, -1, 0))
        ridx = ds.end_idx[split]
        fresh = _fresh_entry_mask(side_state)
        d_idx, d_side = ridx[fresh], side_state[fresh]
        led, _ = _simulate(d_idx, d_side, tp_all[d_idx], sl_all[d_idx], open_, high, low, close, no_exit)
        base_ledgers[split] = led

    p1_rows, p2_rows = [], []
    for seed in seeds:
        bst = xgb.train(dict(XGB_PARAMS, seed=seed), dtr, NUM_ROUNDS, evals=[(dva, "val")],
                        early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        it = (0, bst.best_iteration + 1)
        prob_h = np.full(n, np.nan)
        prob_l = np.full(n, np.nan)
        for ix, dm in ((va, dva), (oo, doo)):
            pr = bst.predict(dm, iteration_range=it)
            prob_h[ix], prob_l[ix] = pr[:, 1], pr[:, 2]

        # ---- P1: marginal EV of closing now, bucketed by adverse-pivot probability ----
        for split in ("val", "oos"):
            df = _marginal_ev(base_ledgers[split], None, None, close, open_, prob_h, prob_l)
            if df.empty:
                continue
            df["bucket"] = pd.cut(df["prob"], PROB_BINS, right=False)
            for gate_name, sub in (("all_bars", df), ("in_profit_only", df[df["in_profit"]])):
                g = sub.groupby("bucket", observed=True)["marginal"]
                for b, s in g:
                    if len(s) < 30:
                        continue
                    m = float(s.mean())
                    t = m / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan
                    p1_rows.append({"seed": seed, "split": split, "gate": gate_name,
                                    "prob_bucket": str(b), "n_bars": int(len(s)),
                                    "mean_marginal_bps": m * 10000.0, "t_stat": float(t)})

        # ---- P2: pivot-triggered entry, threshold fitted on VAL ----
        def _entries(thr, ix):
            sig_l = prob_l[ix] > thr   # bottom -> LONG
            sig_h = prob_h[ix] > thr   # top    -> SHORT
            side = np.where(sig_l & ~sig_h, 1, np.where(sig_h & ~sig_l, -1, 0))
            keep = side != 0
            return ix[keep], side[keep]

        best_thr, best_val = None, -np.inf
        for thr in ENTRY_THRESHOLDS:
            d_idx, d_side = _entries(thr, va)
            r = _summ_bt(_entry_backtest(d_idx, d_side, tp_all, sl_all, panel))
            if r.get("n_trades", 0) >= 20 and r["sum_ret_pct"] > best_val:
                best_val, best_thr = r["sum_ret_pct"], float(thr)
        if best_thr is None:
            best_thr = 0.35
        for split, ix in (("val", va), ("oos", oo)):
            d_idx, d_side = _entries(best_thr, ix)
            r = _summ_bt(_entry_backtest(d_idx, d_side, tp_all, sl_all, panel))
            p2_rows.append({"seed": seed, "split": split, "threshold": best_thr, **r})
        print(f"seed {seed}: P2 thr={best_thr:.2f} "
              f"VAL {p2_rows[-2].get('sum_ret_pct', float('nan')):+.1f}% "
              f"OOS {p2_rows[-1].get('sum_ret_pct', float('nan')):+.1f}% "
              f"(OOS trades {p2_rows[-1].get('n_trades', 0)})", flush=True)

    p1 = pd.DataFrame(p1_rows)
    p2 = pd.DataFrame(p2_rows)
    p1.to_csv(OUT_DIR / "p1_marginal_ev.csv", index=False)
    p2.to_csv(OUT_DIR / "p2_pivot_entry.csv", index=False)

    print("\n=== P1: marginal EV of closing now vs holding, by adverse-pivot probability ===")
    print("    (positive = closing beats holding; this is the sign every exit primitive inherits)")
    for gate in ("all_bars", "in_profit_only"):
        for split in ("val", "oos"):
            sub = p1[(p1["gate"] == gate) & (p1["split"] == split)]
            if sub.empty:
                continue
            agg = sub.groupby("prob_bucket").agg(
                n_bars=("n_bars", "mean"), marginal_bps=("mean_marginal_bps", "mean"),
                t=("t_stat", "mean")).round(2)
            print(f"\n  [{gate} / {split}]")
            print(agg.to_string())

    print("\n=== P2: pivot-triggered entry (threshold fitted on VAL) ===")
    hdr = f"{'split':<6}{'trades':>8}{'win%':>8}{'gross_bps':>11}{'t':>7}{'sum%':>9}{'mdd%':>8}{'L/S':>12}"
    print(hdr)
    print("-" * len(hdr))
    for split in ("val", "oos"):
        s = p2[p2["split"] == split]
        if s.empty or "n_trades" not in s or s["n_trades"].sum() == 0:
            print(f"{split:<6}   no trades")
            continue
        ls = f"{s['long_trades'].mean():.0f}/{s['short_trades'].mean():.0f}"
        print(f"{split:<6}{s['n_trades'].mean():>8.0f}{s['win_rate'].mean()*100:>8.1f}"
              f"{s['gross_bps'].mean():>11.2f}{s['t_gross'].mean():>7.2f}"
              f"{s['sum_ret_pct'].mean():>9.1f}{s['mdd_pct'].mean():>8.1f}{ls:>12}")
    oos = p2[p2["split"] == "oos"]
    if not oos.empty and "sum_ret_pct" in oos:
        print(f"  per-seed OOS sum_ret: {oos['sum_ret_pct'].round(1).tolist()}  "
              f"({int((oos['sum_ret_pct'] > 0).sum())}/{len(oos)} positive)")
        print(f"  per-seed OOS gross bps: {oos['gross_bps'].round(2).tolist()}")

    (OUT_DIR / "summary.json").write_text(json.dumps(
        {"k": K, "seeds": seeds, "entry_thresholds": ENTRY_THRESHOLDS.tolist(),
         "prob_bins": PROB_BINS, "p1": p1_rows, "p2": p2_rows},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
