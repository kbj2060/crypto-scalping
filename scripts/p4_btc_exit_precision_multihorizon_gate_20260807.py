"""P4 -- precision-improving exit design: seed-averaged, dual-horizon (k=1 AND k=2) agreement gate.

P3 showed the pivot-nowcast exit's negative pooled marginal EV is a precision problem (true
adverse-pivot bars are strongly profitable to exit on, t=2.2-8.2 in every bucket/split; false
positives dilute the average and get worse as confidence rises). Best-bucket precision needed to
roughly double (OOS [0.2,0.35): 25.9% actual vs 35.5% breakeven) to turn the pooled EV positive.

Two cheap, well-motivated precision levers, combined (not swept as an architecture search -- a
single fixed design, only the threshold pair is tuned):
  1. Average probability across the 5 nowcaster seeds (variance reduction; this repo has
     independently confirmed elsewhere that seed-averaging reduces variance without fixing a bad
     mean -- here the mean is genuinely positive on true positives, so averaging should help).
  2. Require k=1 AND k=2 nowcasts to BOTH exceed their threshold before firing (agreement gate --
     a true pivot should show up at nearby horizons; an isolated single-k spike is more likely
     noise).

Selection discipline (both VAL and the existing Jan-Mar OOS have been looked at across multiple
probability buckets in P1/P3 already, so neither is clean for further comparison):
  - The (thr1, thr2) grid is swept and the single best config is picked on VAL ONLY.
  - That ONE config is then evaluated on the Jan-Mar OOS split (already-spent, reported for
    continuity only, NOT used to pick anything) and on a genuinely untouched window
    (2026-04-01..2026-08-01, never used by any gate this session) as the real confirmation.
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
from gate_g1_btc_exit_oracle_pivot_rescue_20260807 import _fresh_entry_mask, _simulate  # noqa: E402
from gate_g8_btc_pivot_nowcast_exit_20260807 import (  # noqa: E402
    CHECKPOINT, LABEL_PATH, PANEL_PATH, PIVOT_PATH, ZZSTATE_PATH, XGB_PARAMS,
    NUM_ROUNDS, EARLY_STOP, _pivot_soon_labels, _predict_entry,
)

OUT_DIR = ROOT / "tmp/btc_p4_exit_precision_multihorizon_20260807"
CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT = 12, 288, 2.5, 1.2
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")
FRESH_START, FRESH_END = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-08-01")  # never touched this session
SEEDS = [11, 137, 2029, 40507, 918273]  # identical to G8/P1/P3
THR_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
MIN_POLICY_EXITS_VAL = 15  # matches this repo's standing minimum_trades_per_split convention


def _summ(ledger, equity_marks):
    if len(ledger) == 0:
        return {"n_trades": 0, "sum_ret_pct": 0.0}
    rets = ledger["trade_return"].to_numpy(dtype=np.float64)
    equity = np.concatenate([[1.0], equity_marks])
    rm = np.maximum.accumulate(equity)
    return {"n_trades": int(len(rets)), "win_rate": float((rets > 0).mean()),
            "sum_ret_pct": float(rets.sum() * 100.0), "final_equity": float(equity[-1]),
            "trade_mdd_pct": float(((equity - rm) / rm).min() * 100.0)}


def _train_mean_probs(k, tr, dva, doo, dfresh, seeds, n):
    """Train `seeds` XGBoost nowcasters at horizon k, return SEED-AVERAGED probabilities for
    val/oos/fresh (each shape (n_split, 3))."""
    dtr = tr
    accum = {"val": None, "oos": None, "fresh": None}
    for seed in seeds:
        bst = xgb.train(dict(XGB_PARAMS, seed=seed), dtr, NUM_ROUNDS, evals=[(dva, "val")],
                        early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        it = (0, bst.best_iteration + 1)
        for name, dm in (("val", dva), ("oos", doo), ("fresh", dfresh)):
            pr = bst.predict(dm, iteration_range=it)
            accum[name] = pr if accum[name] is None else accum[name] + pr
    return {name: (accum[name] / len(seeds)) for name in accum}


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

    val_start_i = int(np.searchsorted(ts, np.datetime64(VAL_START)))
    va = np.flatnonzero((ts >= np.datetime64(VAL_START)) & (ts <= np.datetime64(VAL_END)) & finite_row)
    oo = np.flatnonzero((ts >= np.datetime64(OOS_START)) & (ts <= np.datetime64(OOS_END)) & finite_row)
    fr = np.flatnonzero((ts >= np.datetime64(FRESH_START)) & (ts <= np.datetime64(FRESH_END)) & finite_row)
    print(f"VAL n={len(va)}  OOS(spent) n={len(oo)}  FRESH(untouched, 2026-04..08) n={len(fr)}")

    mean_probs = {}
    for k in (1, 2):
        y = _pivot_soon_labels(is_h, is_l, k)
        tr = np.flatnonzero((ts < np.datetime64(VAL_START)) & finite_row)
        tr = tr[tr + k < val_start_i]
        dtr = xgb.DMatrix(X[tr], label=y[tr])
        dva = xgb.DMatrix(X[va], label=y[va])  # needs labels: used as early-stopping eval_set
        doo, dfr = xgb.DMatrix(X[oo]), xgb.DMatrix(X[fr])  # prediction only, no labels needed
        mean_probs[k] = _train_mean_probs(k, dtr, dva, doo, dfr, seeds, n)
        print(f"k={k}: trained {len(seeds)} seeds, averaged probabilities computed", flush=True)

    # ---- baseline entry ledger (same frozen checkpoint G1/G8/P1/P3 used), extended to FRESH ----
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

    window = cfg["window"]
    fresh_row_idx = np.flatnonzero(
        (ds.timestamps_all >= np.datetime64(FRESH_START)) & (ds.timestamps_all <= np.datetime64(FRESH_END))
        & (np.arange(len(ds.timestamps_all)) >= window - 1)
    )

    @torch.no_grad()
    def _predict_rows(row_idx, batch_size=1024):
        em.eval()
        out = []
        for i in range(0, len(row_idx), batch_size):
            x = torch.from_numpy(ds.get_batch(row_idx[i : i + batch_size])).to(device)
            logits, _, _ = em(x)
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)

    no_exit = np.zeros((2, n), dtype=bool)
    entries, base_ledgers = {}, {}
    for split_name, row_idx in (("val", ds.end_idx["val"]), ("oos", ds.end_idx["oos"]), ("fresh", fresh_row_idx)):
        probs = _predict_rows(row_idx)
        pred = probs.argmax(axis=1)
        side_state = np.where(pred == 1, 1, np.where(pred == 2, -1, 0))
        fresh_mask = _fresh_entry_mask(side_state)
        d_idx, d_side = row_idx[fresh_mask], side_state[fresh_mask]
        entries[split_name] = (d_idx, d_side)
        led, eq = _simulate(d_idx, d_side, tp_all[d_idx], sl_all[d_idx], open_, high, low, close, no_exit)
        base_ledgers[split_name] = (led, eq)
        print(f"[{split_name}] baseline entry ledger: {len(led)} trades")

    row_of = {"val": va, "oos": oo, "fresh": fr}

    def _policy_result(split_name, thr1, thr2):
        idx = row_of[split_name]
        prob1, prob2 = mean_probs[1][split_name], mean_probs[2][split_name]
        forced = np.zeros((2, n), dtype=bool)
        fire_h = (prob1[:, 1] > thr1) & (prob2[:, 1] > thr2)
        fire_l = (prob1[:, 2] > thr1) & (prob2[:, 2] > thr2)
        forced[0, idx] = fire_h
        forced[1, idx] = fire_l
        d_idx, d_side = entries[split_name]
        led, eq = _simulate(d_idx, d_side, tp_all[d_idx], sl_all[d_idx], open_, high, low, close,
                            forced, profit_gated=True)
        m = _summ(led, eq)
        m["n_policy_exits"] = int((led["reason"] == "policy").sum()) if len(led) else 0
        m["flag_rate"] = float(fire_h.mean() + fire_l.mean())
        return m

    baseline_sum = {name: _summ(*base_ledgers[name])["sum_ret_pct"] for name in base_ledgers}

    grid_rows = []
    for thr1 in THR_GRID:
        for thr2 in THR_GRID:
            m = _policy_result("val", thr1, thr2)
            grid_rows.append({"thr1": thr1, "thr2": thr2, **{f"val_{k}": v for k, v in m.items()}})
    grid = pd.DataFrame(grid_rows)
    grid["val_lift_vs_baseline"] = grid["val_sum_ret_pct"] - baseline_sum["val"]
    eligible = grid[grid["val_n_policy_exits"] >= MIN_POLICY_EXITS_VAL]
    if eligible.empty:
        print(f"WARNING: no (thr1,thr2) reached {MIN_POLICY_EXITS_VAL} VAL policy exits; falling back to full grid")
        eligible = grid
    best = eligible.loc[eligible["val_sum_ret_pct"].idxmax()]
    thr1_sel, thr2_sel = float(best["thr1"]), float(best["thr2"])
    print(f"\nVAL-selected config: thr1={thr1_sel} thr2={thr2_sel} "
          f"(VAL sum_ret {best['val_sum_ret_pct']:+.1f}%, baseline {baseline_sum['val']:+.1f}%, "
          f"policy_exits={best['val_n_policy_exits']:.0f}, flag_rate={best['val_flag_rate']:.4f})")

    print("\n=== single VAL-selected config applied to each split (fresh = single confirmation) ===")
    final = {}
    for split_name in ("val", "oos", "fresh"):
        m = _policy_result(split_name, thr1_sel, thr2_sel)
        lift = m["sum_ret_pct"] - baseline_sum[split_name]
        final[split_name] = {**m, "baseline_sum_ret_pct": baseline_sum[split_name], "lift_pct": lift}
        tag = " <- ALREADY-SPENT OOS, informational only" if split_name == "oos" else (
              " <- SINGLE FRESH CONFIRMATION, never touched before" if split_name == "fresh" else "")
        print(f"  [{split_name:>5}] policy {m['sum_ret_pct']:+7.1f}%  baseline {baseline_sum[split_name]:+7.1f}%  "
              f"lift {lift:+7.1f}pp  n_policy_exits={m['n_policy_exits']:>4}  flag_rate={m['flag_rate']:.4f}{tag}")

    payload = {"seeds": seeds, "thr_grid": THR_GRID, "selected_thr1": thr1_sel, "selected_thr2": thr2_sel,
               "grid": grid_rows, "final_by_split": final, "baseline_sum_ret_pct": baseline_sum,
               "fresh_window": [str(FRESH_START), str(FRESH_END)],
               "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
               "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}
    (OUT_DIR / "p4_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    grid.to_csv(OUT_DIR / "p4_val_grid.csv", index=False)
    print(f"\nwrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
