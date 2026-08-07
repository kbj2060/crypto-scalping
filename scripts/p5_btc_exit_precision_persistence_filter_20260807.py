"""P5 -- second precision-improving attempt after P4's multi-horizon agreement gate failed (best
VAL-eligible config: ~0.0pp lift, no config found a genuine improvement).

Different mechanism: temporal PERSISTENCE instead of cross-horizon agreement. Require the k=1
seed-averaged adverse-pivot probability to stay above threshold for >=m CONSECUTIVE bars before
firing, instead of requiring k=1 AND k=2 to agree at the same bar. Rationale: P4's AND-gate may
have failed because k=1 and k=2 nowcasts are highly correlated but not independent evidence (both
trained on nearly the same features/window), so requiring both to fire didn't add real
discriminating information, it just shrank the sample. A persistence requirement filters
single-bar noise spikes differently: it keeps the same k=1 signal but demands it be sustained.

Selection discipline: explores thresholds x persistence-length on VAL ONLY. Does NOT touch the
2026-04-08 fresh window again (P4 already spent that window's one confirmation) or the Jan-Mar OOS
comparison beyond a single already-spent report. If nothing clears VAL, this closes the precision-
improvement thread for this session rather than hunting further configs.
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

OUT_DIR = ROOT / "tmp/btc_p5_exit_precision_persistence_20260807"
CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT = 12, 288, 2.5, 1.2
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")
K = 1
SEEDS = [11, 137, 2029, 40507, 918273]
THR_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
PERSIST_GRID = [1, 2, 3, 4]  # consecutive bars prob must stay above threshold; 1 == P1/G8's original rule
MIN_POLICY_EXITS_VAL = 15


def _summ(ledger, equity_marks):
    if len(ledger) == 0:
        return {"n_trades": 0, "sum_ret_pct": 0.0}
    rets = ledger["trade_return"].to_numpy(dtype=np.float64)
    equity = np.concatenate([[1.0], equity_marks])
    rm = np.maximum.accumulate(equity)
    return {"n_trades": int(len(rets)), "win_rate": float((rets > 0).mean()),
            "sum_ret_pct": float(rets.sum() * 100.0)}


def _persist_mask(above: np.ndarray, m: int) -> np.ndarray:
    """True at bar i iff `above` has been True for the last m consecutive bars including i."""
    if m <= 1:
        return above
    out = above.copy()
    for shift in range(1, m):
        shifted = np.zeros_like(above)
        shifted[shift:] = above[:-shift]
        out &= shifted
    return out


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
    y = _pivot_soon_labels(is_h, is_l, K)

    zz = pd.read_parquet(ZZSTATE_PATH).sort_values("timestamp").reset_index(drop=True)
    zz_cols = [c for c in zz.columns if c != "timestamp"]
    base_cols = [c for c in panel.columns if c != "timestamp"]
    X = np.concatenate([panel[base_cols].to_numpy(dtype=np.float32),
                        zz[zz_cols].to_numpy(dtype=np.float32)], axis=1)
    finite_row = np.isfinite(X).all(axis=1)

    val_start_i = int(np.searchsorted(ts, np.datetime64(VAL_START)))
    tr = np.flatnonzero((ts < np.datetime64(VAL_START)) & finite_row)
    tr = tr[tr + K < val_start_i]
    va = np.flatnonzero((ts >= np.datetime64(VAL_START)) & (ts <= np.datetime64(VAL_END)) & finite_row)
    oo = np.flatnonzero((ts >= np.datetime64(OOS_START)) & (ts <= np.datetime64(OOS_END)) & finite_row)
    dtr = xgb.DMatrix(X[tr], label=y[tr])
    dva = xgb.DMatrix(X[va], label=y[va])
    doo = xgb.DMatrix(X[oo])

    # seed-averaged probability, full-length arrays (nan outside val/oos so downstream masks are safe)
    prob_h = np.full(n, np.nan)
    prob_l = np.full(n, np.nan)
    accum_val, accum_oos = None, None
    for seed in seeds:
        bst = xgb.train(dict(XGB_PARAMS, seed=seed), dtr, NUM_ROUNDS, evals=[(dva, "val")],
                        early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        it = (0, bst.best_iteration + 1)
        pv, po = bst.predict(dva, iteration_range=it), bst.predict(doo, iteration_range=it)
        accum_val = pv if accum_val is None else accum_val + pv
        accum_oos = po if accum_oos is None else accum_oos + po
    prob_h[va], prob_l[va] = (accum_val / len(seeds))[:, 1], (accum_val / len(seeds))[:, 2]
    prob_h[oo], prob_l[oo] = (accum_oos / len(seeds))[:, 1], (accum_oos / len(seeds))[:, 2]
    print(f"trained {len(seeds)} seeds, seed-averaged k=1 probabilities computed", flush=True)

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
    entries, base_ledgers = {}, {}
    for split in ("val", "oos"):
        probs = _predict_entry(em, ds, split, device)
        pred = probs.argmax(axis=1)
        side_state = np.where(pred == 1, 1, np.where(pred == 2, -1, 0))
        ridx = ds.end_idx[split]
        fresh = _fresh_entry_mask(side_state)
        d_idx, d_side = ridx[fresh], side_state[fresh]
        entries[split] = (d_idx, d_side)
        led, eq = _simulate(d_idx, d_side, tp_all[d_idx], sl_all[d_idx], open_, high, low, close, no_exit)
        base_ledgers[split] = _summ(led, eq)
        print(f"[{split}] baseline entry ledger: {base_ledgers[split]['n_trades']} trades, "
              f"sum_ret {base_ledgers[split]['sum_ret_pct']:+.1f}%")

    row_of = {"val": va, "oos": oo}

    def _policy_result(split, thr, m):
        idx = row_of[split]
        above_h = prob_h[idx] > thr
        above_l = prob_l[idx] > thr
        fire_h = _persist_mask(above_h, m)
        fire_l = _persist_mask(above_l, m)
        forced = np.zeros((2, n), dtype=bool)
        forced[0, idx] = fire_h
        forced[1, idx] = fire_l
        d_idx, d_side = entries[split]
        led, eq = _simulate(d_idx, d_side, tp_all[d_idx], sl_all[d_idx], open_, high, low, close,
                            forced, profit_gated=True)
        r = _summ(led, eq)
        r["n_policy_exits"] = int((led["reason"] == "policy").sum()) if len(led) else 0
        return r

    grid_rows = []
    for thr in THR_GRID:
        for m in PERSIST_GRID:
            r = _policy_result("val", thr, m)
            grid_rows.append({"thr": thr, "persist": m, "val_sum_ret_pct": r["sum_ret_pct"],
                              "val_n_policy_exits": r["n_policy_exits"],
                              "val_lift": r["sum_ret_pct"] - base_ledgers["val"]["sum_ret_pct"]})
    grid = pd.DataFrame(grid_rows)
    eligible = grid[grid["val_n_policy_exits"] >= MIN_POLICY_EXITS_VAL]
    print(f"\n{len(eligible)}/{len(grid)} grid cells reach the {MIN_POLICY_EXITS_VAL}-trade VAL floor")
    if eligible.empty:
        print("No config reaches the minimum-trade floor on VAL -- persistence filter closed, "
              "not evaluating OOS.")
    else:
        best = eligible.loc[eligible["val_lift"].idxmax()]
        print(f"Best VAL-eligible: thr={best.thr} persist={best.persist} "
              f"VAL sum_ret {best.val_sum_ret_pct:+.1f}% (baseline {base_ledgers['val']['sum_ret_pct']:+.1f}%) "
              f"lift={best.val_lift:+.2f}pp n_exits={best.val_n_policy_exits:.0f}")
        if best.val_lift > 0.5:  # only bother checking OOS if VAL shows a real (not noise-floor) improvement
            r_oos = _policy_result("oos", float(best.thr), int(best.persist))
            lift_oos = r_oos["sum_ret_pct"] - base_ledgers["oos"]["sum_ret_pct"]
            print(f"  -> already-spent OOS at this config: {r_oos['sum_ret_pct']:+.1f}% "
                  f"(baseline {base_ledgers['oos']['sum_ret_pct']:+.1f}%) lift={lift_oos:+.2f}pp "
                  f"n_exits={r_oos['n_policy_exits']} [informational only, not a fresh confirmation]")
        else:
            print("  VAL lift is within noise range (<0.5pp) -- not spending the OOS/fresh check on it.")

    print("\ntop 10 VAL cells by lift (any trade count, for visibility):")
    print(grid.sort_values("val_lift", ascending=False).head(10).to_string(index=False))

    grid.to_csv(OUT_DIR / "p5_val_grid.csv", index=False)
    (OUT_DIR / "p5_summary.json").write_text(json.dumps(
        {"seeds": seeds, "thr_grid": THR_GRID, "persist_grid": PERSIST_GRID, "grid": grid_rows,
         "baseline": base_ledgers}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
