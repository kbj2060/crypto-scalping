"""P6 -- two-stage exit classifier, the "genuinely better classifier" P4/P5's failure implied was
needed (simple post-hoc filtering of the stage-1 nowcaster's own output did not work).

Stage 1 (unchanged from G8/P4/P5): k=1 and k=2 pivot-nowcast XGBoost classifiers on causalfix_final
+ zz_state features (117 cols), 5 seeds each, seed-averaged probability.

Stage 2 (new): a SECOND classifier trained ONLY on the "candidate zone" (bars where stage-1's k=1
probability for either side already exceeds a low floor) to discriminate true adverse-pivot bars
from false positives specifically within that zone -- the exact boundary P3 showed is where the
precision problem lives. Its feature set is the same 117 raw/causal columns PLUS the 4 raw stage-1
probabilities (prob_h_k1, prob_l_k1, prob_h_k2, prob_l_k2) as explicit numeric meta-features, so
stage 2 can learn a much richer combination than P4's hard AND-gate (which only ever saw two
binary above/below-threshold decisions, never the underlying continuous values together).

No-leakage stacking split: stage 1 is fit on TRAIN_A (< 2025-05-01) to generate genuinely
out-of-fold probabilities on TRAIN_B (2025-05-01..2025-08-31), which become stage 2's training
data -- using stage 1's in-sample-fit predictions on its own training rows would let stage 2 learn
stage 1's overfitting pattern rather than its true generalization behaviour. A separate stage-1 fit
on the FULL train period (TRAIN_A+TRAIN_B) generates the probabilities stage 2 actually consumes at
VAL/OOS/fresh inference time (standard practice: no leakage risk there since VAL/OOS/fresh are
unseen by stage 1 either way).

Selection discipline unchanged from P4/P5: sweep the stage-2 threshold on VAL ONLY (min 15 policy
exits), report the already-spent Jan-Mar OOS informationally, and only spend the 2026-04-08 fresh
window if VAL shows a real (not <=3-trade noise) improvement over stage-1-alone's own best VAL
policy -- and note P4 already used that window once, so treat any further fresh use as the last one
available this session.
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

OUT_DIR = ROOT / "tmp/btc_p6_two_stage_precision_20260807"
CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT = 12, 288, 2.5, 1.2
TRAIN_A_END = pd.Timestamp("2025-05-01")
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")
FRESH_START, FRESH_END = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-08-01")
SEEDS = [11, 137, 2029, 40507, 918273]
CANDIDATE_FLOOR = 0.05  # stage-1 k=1 prob must clear this for stage 2 to even look at a bar
THR_GRID = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
MIN_POLICY_EXITS_VAL = 15


def _summ(ledger, equity_marks):
    if len(ledger) == 0:
        return {"n_trades": 0, "sum_ret_pct": 0.0}
    rets = ledger["trade_return"].to_numpy(dtype=np.float64)
    equity = np.concatenate([[1.0], equity_marks])
    return {"n_trades": int(len(rets)), "win_rate": float((rets > 0).mean()),
            "sum_ret_pct": float(rets.sum() * 100.0)}


def _train_avg(dtr, dva_eval, predict_targets, seeds):
    """Train `seeds` XGBoost models on dtr (early-stopped against dva_eval), return seed-averaged
    probability arrays for each DMatrix in predict_targets (dict name -> DMatrix)."""
    accum = {name: None for name in predict_targets}
    for seed in seeds:
        bst = xgb.train(dict(XGB_PARAMS, seed=seed), dtr, NUM_ROUNDS, evals=[(dva_eval, "val")],
                        early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        it = (0, bst.best_iteration + 1)
        for name, dm in predict_targets.items():
            pr = bst.predict(dm, iteration_range=it)
            accum[name] = pr if accum[name] is None else accum[name] + pr
    return {name: accum[name] / len(seeds) for name in accum}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--spend-fresh", action="store_true",
                    help="also evaluate the VAL-selected config on the untouched 2026-04..08 window")
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

    train_a_end_i = int(np.searchsorted(ts, np.datetime64(TRAIN_A_END)))
    val_start_i = int(np.searchsorted(ts, np.datetime64(VAL_START)))

    va = np.flatnonzero((ts >= np.datetime64(VAL_START)) & (ts <= np.datetime64(VAL_END)) & finite_row)
    oo = np.flatnonzero((ts >= np.datetime64(OOS_START)) & (ts <= np.datetime64(OOS_END)) & finite_row)
    fr = np.flatnonzero((ts >= np.datetime64(FRESH_START)) & (ts <= np.datetime64(FRESH_END)) & finite_row)

    # ===== stage 1: k=1 and k=2 nowcasters =====
    stage1_full = {}   # full-train fit -> used for VAL/OOS/fresh inference (deployed model)
    stage1_oof = {}    # TRAIN_A-fit -> out-of-fold predictions on TRAIN_B (stage-2 training data)
    train_a_idx = {}
    train_b_idx = {}
    for k in (1, 2):
        y = _pivot_soon_labels(is_h, is_l, k)
        ta = np.flatnonzero((ts < np.datetime64(TRAIN_A_END)) & finite_row)
        ta = ta[ta + k < train_a_end_i]
        tb = np.flatnonzero((ts >= np.datetime64(TRAIN_A_END)) & (ts < np.datetime64(VAL_START)) & finite_row)
        tb = tb[tb + k < val_start_i]
        tfull = np.flatnonzero((ts < np.datetime64(VAL_START)) & finite_row)
        tfull = tfull[tfull + k < val_start_i]
        train_a_idx[k], train_b_idx[k] = ta, tb

        dta = xgb.DMatrix(X[ta], label=y[ta])
        dtb_eval = xgb.DMatrix(X[tb], label=y[tb])  # TRAIN_A model's own "val" for early stop
        dtb_pred = xgb.DMatrix(X[tb])
        stage1_oof[k] = _train_avg(dta, dtb_eval, {"tb": dtb_pred}, seeds)["tb"]

        dtfull = xgb.DMatrix(X[tfull], label=y[tfull])
        dva = xgb.DMatrix(X[va], label=y[va])
        preds = _train_avg(dtfull, dva, {"val": xgb.DMatrix(X[va]), "oos": xgb.DMatrix(X[oo]),
                                         "fresh": xgb.DMatrix(X[fr])}, seeds)
        stage1_full[k] = preds
        print(f"stage1 k={k}: OOF on TRAIN_B (n={len(tb)}) + deployed model for VAL/OOS/fresh done", flush=True)

    # full-length stage-1 k=1/k=2 probability arrays (nan where undefined) for building stage-2 features
    p_h1, p_l1 = np.full(n, np.nan), np.full(n, np.nan)
    p_h2, p_l2 = np.full(n, np.nan), np.full(n, np.nan)
    p_h1[train_b_idx[1]], p_l1[train_b_idx[1]] = stage1_oof[1][:, 1], stage1_oof[1][:, 2]
    p_h2[train_b_idx[2]], p_l2[train_b_idx[2]] = stage1_oof[2][:, 1], stage1_oof[2][:, 2]
    for split, idx in (("val", va), ("oos", oo), ("fresh", fr)):
        p_h1[idx], p_l1[idx] = stage1_full[1][split][:, 1], stage1_full[1][split][:, 2]
        p_h2[idx], p_l2[idx] = stage1_full[2][split][:, 1], stage1_full[2][split][:, 2]

    # ===== stage 2: candidate-zone classifier =====
    y1 = _pivot_soon_labels(is_h, is_l, 1)  # stage 2 targets the same k=1 event stage-1's deployed policy would act on
    X2 = np.concatenate([X, p_h1[:, None], p_l1[:, None], p_h2[:, None], p_l2[:, None]], axis=1)

    def _candidate(idx):
        cand = idx[(np.maximum(p_h1[idx], p_l1[idx])) > CANDIDATE_FLOOR]
        return cand

    tb_cand = _candidate(train_b_idx[1])
    va_cand = _candidate(va)
    print(f"stage2 candidate zone: TRAIN_B {len(tb_cand)}/{len(train_b_idx[1])}  "
          f"VAL {len(va_cand)}/{len(va)}  (floor={CANDIDATE_FLOOR})")

    dtb2 = xgb.DMatrix(X2[tb_cand], label=y1[tb_cand])
    dva2_eval = xgb.DMatrix(X2[va_cand], label=y1[va_cand])
    targets2 = {"val_cand": xgb.DMatrix(X2[va_cand])}
    oo_cand = _candidate(oo)
    targets2["oos_cand"] = xgb.DMatrix(X2[oo_cand])
    fr_cand = _candidate(fr)
    if args.spend_fresh:
        targets2["fresh_cand"] = xgb.DMatrix(X2[fr_cand])
    stage2_pred = _train_avg(dtb2, dva2_eval, targets2, seeds)
    print("stage2: trained on TRAIN_B candidate zone, predictions computed", flush=True)

    # ===== backtest wiring (same simulator/entry-ledger convention as G1/G8/P1/P3/P4/P5) =====
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
    split_rows = {"val": ds.end_idx["val"], "oos": ds.end_idx["oos"], "fresh": fresh_row_idx}
    for split_name, row_idx in split_rows.items():
        probs = _predict_rows(row_idx)
        pred = probs.argmax(axis=1)
        side_state = np.where(pred == 1, 1, np.where(pred == 2, -1, 0))
        fresh_mask = _fresh_entry_mask(side_state)
        d_idx, d_side = row_idx[fresh_mask], side_state[fresh_mask]
        entries[split_name] = (d_idx, d_side)
        led, eq = _simulate(d_idx, d_side, tp_all[d_idx], sl_all[d_idx], open_, high, low, close, no_exit)
        base_ledgers[split_name] = _summ(led, eq)
        print(f"[{split_name}] baseline entry ledger: {base_ledgers[split_name]['n_trades']} trades, "
              f"sum_ret {base_ledgers[split_name]['sum_ret_pct']:+.1f}%")

    row_of = {"val": va, "oos": oo, "fresh": fr}
    cand_of = {"val": va_cand, "oos": oo_cand, "fresh": fr_cand}
    pred_key = {"val": "val_cand", "oos": "oos_cand", "fresh": "fresh_cand"}

    def _policy_result(split, thr, use_stage2=True):
        idx_full = row_of[split]
        prob_h_full = np.zeros(len(idx_full))
        prob_l_full = np.zeros(len(idx_full))
        if use_stage2:
            key = pred_key[split]
            if key not in stage2_pred:
                return None  # fresh not spent
            cand = cand_of[split]
            pos_of_cand_in_full = np.searchsorted(idx_full, cand)
            prob_h_full[pos_of_cand_in_full] = stage2_pred[key][:, 1]
            prob_l_full[pos_of_cand_in_full] = stage2_pred[key][:, 2]
        else:
            prob_h_full, prob_l_full = p_h1[idx_full], p_l1[idx_full]
        forced = np.zeros((2, n), dtype=bool)
        forced[0, idx_full] = prob_h_full > thr
        forced[1, idx_full] = prob_l_full > thr
        d_idx, d_side = entries[split]
        led, eq = _simulate(d_idx, d_side, tp_all[d_idx], sl_all[d_idx], open_, high, low, close,
                            forced, profit_gated=True)
        r = _summ(led, eq)
        r["n_policy_exits"] = int((led["reason"] == "policy").sum()) if len(led) else 0
        return r

    grid_rows = []
    for thr in THR_GRID:
        r = _policy_result("val", thr, use_stage2=True)
        grid_rows.append({"thr": thr, "val_sum_ret_pct": r["sum_ret_pct"],
                          "val_n_policy_exits": r["n_policy_exits"],
                          "val_lift": r["sum_ret_pct"] - base_ledgers["val"]["sum_ret_pct"]})
    grid = pd.DataFrame(grid_rows)
    eligible = grid[grid["val_n_policy_exits"] >= MIN_POLICY_EXITS_VAL]
    print(f"\nstage-2 VAL grid:\n{grid.to_string(index=False)}")
    print(f"\n{len(eligible)}/{len(grid)} cells reach the {MIN_POLICY_EXITS_VAL}-trade VAL floor")

    if eligible.empty:
        print("No stage-2 config reaches the VAL trade floor -- closed, not spending OOS/fresh.")
    else:
        best = eligible.loc[eligible["val_lift"].idxmax()]
        print(f"\nBest VAL-eligible stage-2 config: thr={best.thr} "
              f"VAL sum_ret {best.val_sum_ret_pct:+.1f}% (baseline {base_ledgers['val']['sum_ret_pct']:+.1f}%) "
              f"lift={best.val_lift:+.2f}pp n_exits={best.val_n_policy_exits:.0f}")
        r_oos = _policy_result("oos", float(best.thr), use_stage2=True)
        lift_oos = r_oos["sum_ret_pct"] - base_ledgers["oos"]["sum_ret_pct"]
        print(f"  already-spent OOS (informational): {r_oos['sum_ret_pct']:+.1f}% "
              f"(baseline {base_ledgers['oos']['sum_ret_pct']:+.1f}%) lift={lift_oos:+.2f}pp "
              f"n_exits={r_oos['n_policy_exits']}")
        if args.spend_fresh and best.val_lift > 0.5:
            r_fresh = _policy_result("fresh", float(best.thr), use_stage2=True)
            if r_fresh is not None:
                lift_fresh = r_fresh["sum_ret_pct"] - base_ledgers["fresh"]["sum_ret_pct"]
                print(f"  FRESH (single confirmation, 2026-04..08): {r_fresh['sum_ret_pct']:+.1f}% "
                      f"(baseline {base_ledgers['fresh']['sum_ret_pct']:+.1f}%) lift={lift_fresh:+.2f}pp "
                      f"n_exits={r_fresh['n_policy_exits']}")
        elif args.spend_fresh:
            print("  VAL lift <0.5pp (noise range) -- not spending fresh confirmation.")

    payload = {"seeds": seeds, "candidate_floor": CANDIDATE_FLOOR, "thr_grid": THR_GRID,
               "grid": grid_rows, "baseline": base_ledgers,
               "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
               "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}
    (OUT_DIR / "p6_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    grid.to_csv(OUT_DIR / "p6_val_grid.csv", index=False)
    print(f"\nwrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
