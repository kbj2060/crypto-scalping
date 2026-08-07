"""P3 -- decompose P1's negative marginal-EV finding: is it false positives, or does it persist
even on the classifier's TRUE positives?

P1 (scripts/primitive_p1p2_btc_pivot_marginal_ev_and_entry_20260807.py) found that the pivot
nowcaster's own high-confidence bucket has the MOST negative marginal EV (worst OOS bucket
[0.05,0.10): -16.44bps, t=-2.94) -- closing "now" is worse than holding, exactly backwards from
what an exit signal needs, and because exit EV is linear in exit fraction this closes the entire
full/partial/scaled exit-primitive family regardless of policy. But P1 measured marginal EV over
ALL bars the classifier flagged in each probability bucket, mixing true adverse-pivot bars with
false positives. Two different explanations produce the same P1 result:

  (a) false positives dominate the bucket and drag the average negative -> raising PRECISION
      further could still salvage an exit built on top of this classifier.
  (b) even the TRUE adverse-pivot bars have negative marginal EV for THIS entry model's trades ->
      "the profitable pivots are structurally undetectable" from this feature set / entry
      combination, and no amount of precision tuning on this classifier can fix it.

This is the "not yet measured" item flagged in registry line btc_zigzag_pivot_exit_primitives.
Reuses P1's exact setup (K=1, same 5 seeds, same nowcaster training, same baseline entry ledger)
and the SAME oracle pivot ground truth (is_h/is_l, the same one _pivot_soon_labels trains against)
to tag each bar inside an open position as a genuine true-adverse-pivot bar or not, independent of
what the classifier predicts -- this is ground truth, not the model's own precision at the fitted
threshold.
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

OUT_DIR = ROOT / "tmp/btc_p3_true_positive_marginal_ev_20260807"
CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT = 12, 288, 2.5, 1.2
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010
NOTIONAL = MARGIN_FRACTION * LEVERAGE
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")
K = 1  # matches P1's own choice: the best-AUC nowcast setting
SEEDS = [11, 137, 2029, 40507, 918273]  # identical to G8/P1, for direct comparability
PROB_BINS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.01]


def _marginal_ev_labeled(ledger, close, open_, prob_h, prob_l, y_true):
    """Per-bar counterfactual of closing now vs holding, tagged with the classifier's probability
    AND the ground-truth label (is this bar actually within k bars of a genuine adverse pivot for
    the held side) -- independent of what the classifier predicts."""
    recs = []
    for _, tr in ledger.iterrows():
        e, x, side = int(tr["entry_i"]), int(tr["exit_i"]), int(tr["side"])
        realised = float(tr["price_move"])
        p_adverse = prob_h if side > 0 else prob_l
        true_class = 1 if side > 0 else 2  # H pivot adverse for LONG, L pivot adverse for SHORT
        entry_px = open_[e]
        for j in range(e, x):
            if not np.isfinite(p_adverse[j]):
                continue
            unreal = (close[j] / entry_px - 1.0) if side > 0 else (1.0 - close[j] / entry_px)
            recs.append({"prob": float(p_adverse[j]), "marginal": (unreal - realised) * NOTIONAL,
                         "in_profit": unreal > ROUNDTRIP_COST_RATE,
                         "is_true_adverse": bool(y_true[j] == true_class)})
    return pd.DataFrame(recs)


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
    close = panel["close"].to_numpy(dtype=np.float64)

    log_ret = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
    tp_all, sl_all = TP_MULT * vol, SL_MULT * vol

    piv = pd.read_parquet(PIVOT_PATH, columns=["timestamp", "is_pivot", "pivot_type"])
    piv = piv.sort_values("timestamp").reset_index(drop=True)
    is_h = ((piv["is_pivot"] == 1) & (piv["pivot_type"] == "H")).to_numpy()
    is_l = ((piv["is_pivot"] == 1) & (piv["pivot_type"] == "L")).to_numpy()
    y_true = _pivot_soon_labels(is_h, is_l, K)

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
    dtr, dva, doo = (xgb.DMatrix(X[i], label=y_true[i]) for i in (tr, va, oo))

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
        led, _ = _simulate(d_idx, d_side, tp_all[d_idx], sl_all[d_idx], open_,
                           panel["high"].to_numpy(dtype=np.float64), panel["low"].to_numpy(dtype=np.float64),
                           close, no_exit)
        base_ledgers[split] = led
        print(f"[{split}] baseline ledger: {len(led)} trades")

    rows = []
    for seed in seeds:
        bst = xgb.train(dict(XGB_PARAMS, seed=seed), dtr, NUM_ROUNDS, evals=[(dva, "val")],
                        early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        it = (0, bst.best_iteration + 1)
        prob_h = np.full(n, np.nan)
        prob_l = np.full(n, np.nan)
        for ix, dm in ((va, dva), (oo, doo)):
            pr = bst.predict(dm, iteration_range=it)
            prob_h[ix], prob_l[ix] = pr[:, 1], pr[:, 2]

        for split in ("val", "oos"):
            df = _marginal_ev_labeled(base_ledgers[split], close, open_, prob_h, prob_l, y_true)
            if df.empty:
                continue
            df["bucket"] = pd.cut(df["prob"], PROB_BINS, right=False)
            for tp_gate_name, tp_sub in (
                ("true_positive_only", df[df["is_true_adverse"]]),
                ("false_positive_only", df[~df["is_true_adverse"]]),
                ("all", df),
            ):
                for profit_gate_name, sub in (("all_bars", tp_sub), ("in_profit_only", tp_sub[tp_sub["in_profit"]])):
                    g = sub.groupby("bucket", observed=True)["marginal"]
                    for b, s in g:
                        if len(s) < 10:  # lower floor than P1's 30: true-positive bars are much rarer
                            continue
                        m = float(s.mean())
                        t = m / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan
                        rows.append({"seed": seed, "split": split, "tp_gate": tp_gate_name,
                                    "profit_gate": profit_gate_name, "prob_bucket": str(b),
                                    "n_bars": int(len(s)), "mean_marginal_bps": m * 10000.0, "t_stat": float(t)})
        print(f"seed {seed} done", flush=True)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "p3_true_positive_marginal_ev.csv", index=False)

    print("\n=== P3: marginal EV restricted to TRUE adverse-pivot bars only (ground truth, not model precision) ===")
    for split in ("val", "oos"):
        for profit_gate in ("all_bars", "in_profit_only"):
            sub = df_out[(df_out.split == split) & (df_out.tp_gate == "true_positive_only") & (df_out.profit_gate == profit_gate)]
            if sub.empty:
                continue
            agg = sub.groupby("prob_bucket").agg(n_bars=("n_bars", "sum"),
                                                  marginal_bps=("mean_marginal_bps", "mean"),
                                                  t=("t_stat", "mean")).round(2)
            print(f"\n  [{split} / {profit_gate} / TRUE POSITIVES ONLY]")
            print(agg.to_string())

    print("\n=== for comparison: false positives only ===")
    for split in ("val", "oos"):
        sub = df_out[(df_out.split == split) & (df_out.tp_gate == "false_positive_only") & (df_out.profit_gate == "in_profit_only")]
        if sub.empty:
            continue
        agg = sub.groupby("prob_bucket").agg(n_bars=("n_bars", "sum"),
                                              marginal_bps=("mean_marginal_bps", "mean"),
                                              t=("t_stat", "mean")).round(2)
        print(f"\n  [{split} / in_profit_only / FALSE POSITIVES ONLY]")
        print(agg.to_string())

    payload = {"k": K, "seeds": seeds, "prob_bins": PROB_BINS, "rows": rows}
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
