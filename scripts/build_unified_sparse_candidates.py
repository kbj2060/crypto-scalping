from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified_dircat_oof.csv"
DEFAULT_OUT_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified_sparse_candidates.csv"
DEFAULT_OUT_JSON = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_sparse_candidates.json"
FEE = 0.0005
SLIP = 0.0002


def _sup_side(df: pd.DataFrame) -> np.ndarray:
    lp = pd.to_numeric(df["ud_cat_long_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    sp = pd.to_numeric(df["ud_cat_short_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    fp = pd.to_numeric(df["ud_cat_flat_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    return np.where((lp >= sp) & (lp >= fp), 1, np.where((sp > lp) & (sp >= fp), -1, 0)).astype(np.int8)


def _build_candidates(df: pd.DataFrame, p: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    raw_side = np.sign(pd.to_numeric(out["m7_action"], errors="coerce").fillna(0.0).to_numpy(np.float64)).astype(np.int8)
    sup_side = _sup_side(out)
    q = pd.to_numeric(out["m7_target_quality"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    raw_edge = (
        pd.to_numeric(out["m7_prob_up"], errors="coerce").fillna(0.0)
        - pd.to_numeric(out["m7_prob_dn"], errors="coerce").fillna(0.0)
    ).abs().to_numpy(np.float64)
    sup_prob_max = out[["ud_cat_long_prob", "ud_cat_short_prob", "ud_cat_flat_prob"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).max(axis=1).to_numpy(np.float64)
    hold_pred = pd.to_numeric(out["m7_hold_pred"], errors="coerce").fillna(6.0).clip(4.0, 8.0).round().astype(np.int32).to_numpy()
    raw_change = np.r_[True, raw_side[1:] != raw_side[:-1]]
    agree = (raw_side == sup_side) & (sup_side != 0)

    candidate = np.zeros(len(out), dtype=np.int8)
    side = np.zeros(len(out), dtype=np.int8)
    last_idx = -10**9
    for i in range(len(out)):
        if sup_side[i] == 0:
            continue
        if q[i] < p["quality_min"]:
            continue
        if raw_edge[i] < p["raw_edge_min"]:
            continue
        if sup_prob_max[i] < p["sup_prob_min"]:
            continue
        if p["require_agreement"] and not agree[i]:
            continue
        if p["sign_change_only"] and not raw_change[i]:
            continue
        if i - last_idx < p["debounce_bars"]:
            continue
        candidate[i] = 1
        side[i] = sup_side[i]
        last_idx = i

    out["ud_cand_flag"] = candidate
    out["ud_cand_side"] = side
    out["ud_cand_hold"] = hold_pred
    out["ud_cand_raw_side"] = raw_side
    out["ud_cand_sup_side"] = sup_side
    out["ud_cand_quality"] = q
    out["ud_cand_raw_edge"] = raw_edge
    out["ud_cand_sup_prob_max"] = sup_prob_max
    out["ud_cand_agree"] = agree.astype(np.int8)
    out["ud_cand_raw_change"] = raw_change.astype(np.int8)
    return out


def _run_sparse_backtest(df: pd.DataFrame) -> dict[str, Any]:
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(np.float64)
    cand = pd.to_numeric(df["ud_cand_flag"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    side = pd.to_numeric(df["ud_cand_side"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    hold_lim = pd.to_numeric(df["ud_cand_hold"], errors="coerce").fillna(6).astype(np.int32).to_numpy()

    pos = 0
    entry = 0.0
    hold = 0
    target_hold = 0
    balance = 1.0
    peak = 1.0
    mdd = 0.0
    trades = 0
    wins = 0
    longs = 0
    shorts = 0
    for i in range(len(df)):
        if pos == 0:
            if cand[i] == 1 and side[i] != 0:
                pos = int(side[i])
                entry = close[i] * (1.0 + SLIP) if pos == 1 else close[i] * (1.0 - SLIP)
                balance *= (1.0 - FEE)
                hold = 0
                target_hold = int(hold_lim[i])
                longs += int(pos == 1)
                shorts += int(pos == -1)
        else:
            hold += 1
            reverse = cand[i] == 1 and side[i] == -pos
            if reverse or hold >= target_hold:
                fill = close[i] * (1.0 - SLIP) if pos == 1 else close[i] * (1.0 + SLIP)
                pnl = ((fill - entry) / entry) if pos == 1 else ((entry - fill) / entry)
                balance *= max(1e-8, (1.0 + pnl) * (1.0 - FEE))
                trades += 1
                wins += int(pnl > 0)
                pos = 0
                entry = 0.0
                hold = 0
                target_hold = 0
        if balance > peak:
            peak = balance
        mdd = min(mdd, balance / max(peak, 1e-8) - 1.0)
    if pos != 0:
        fill = close[-1] * (1.0 - SLIP) if pos == 1 else close[-1] * (1.0 + SLIP)
        pnl = ((fill - entry) / entry) if pos == 1 else ((entry - fill) / entry)
        balance *= max(1e-8, (1.0 + pnl) * (1.0 - FEE))
        trades += 1
        wins += int(pnl > 0)
    return {
        "pnl_pct": float((balance - 1.0) * 100.0),
        "trades": int(trades),
        "wr_pct": float(wins / max(trades, 1) * 100.0),
        "mdd_pct": float(mdd * 100.0),
        "longs": int(longs),
        "shorts": int(shorts),
    }


def _score(res: dict[str, Any]) -> float:
    s = res["pnl_pct"] - 0.35 * abs(min(res["mdd_pct"], 0.0))
    if res["trades"] < 20:
        s -= 10.0
    if res["trades"] > 300:
        s -= 0.04 * (res["trades"] - 300)
    return float(s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build sparse candidate events from OOF direction data")
    ap.add_argument("--csv-path", default=DEFAULT_CSV)
    ap.add_argument("--output-csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--output-json", default=DEFAULT_OUT_JSON)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    calib = df[df["ud_cat_is_holdout"] == 0].copy().reset_index(drop=True)
    fold_ids = sorted(int(x) for x in pd.to_numeric(calib["ud_cat_oof_fold"], errors="coerce").dropna().unique() if x >= 0)
    recent_folds = fold_ids[-4:]
    windows: list[tuple[int, int]] = []
    fold_col = pd.to_numeric(calib["ud_cat_oof_fold"], errors="coerce").fillna(-1).astype(np.int32).to_numpy()
    for fid in recent_folds:
        idx = np.flatnonzero(fold_col == fid)
        if len(idx):
            windows.append((int(idx[0]), int(idx[-1] + 1)))

    grid = itertools.product(
        [0.006, 0.007, 0.008],
        [0.55, 0.65, 0.75],
        [0.70, 0.80, 0.90],
        [4, 8, 12],
        [False, True],
        [False, True],
    )
    best: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for quality_min, raw_edge_min, sup_prob_min, debounce_bars, require_agreement, sign_change_only in grid:
        p = {
            "quality_min": quality_min,
            "raw_edge_min": raw_edge_min,
            "sup_prob_min": sup_prob_min,
            "debounce_bars": debounce_bars,
            "require_agreement": require_agreement,
            "sign_change_only": sign_change_only,
        }
        cand_df = _build_candidates(calib, p)
        window_scores = []
        window_results = []
        for ws, we in windows:
            res = _run_sparse_backtest(cand_df.iloc[ws:we].reset_index(drop=True))
            sc = _score(res)
            window_results.append({"start": ws, "end": we, "result": res, "score": sc})
            window_scores.append(sc)
        row = {
            "params": p,
            "avg_score": float(np.mean(window_scores)),
            "avg_pnl_pct": float(np.mean([w["result"]["pnl_pct"] for w in window_results])),
            "avg_trades": float(np.mean([w["result"]["trades"] for w in window_results])),
            "windows": window_results,
        }
        rows.append(row)
        if best is None or row["avg_score"] > best["avg_score"]:
            best = row

    assert best is not None
    full = _build_candidates(df, best["params"])
    test = full[full["ud_cat_is_holdout"] == 1].copy().reset_index(drop=True)
    best["test"] = _run_sparse_backtest(test) if len(test) else None

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    full.to_csv(args.output_csv, index=False)
    rows = sorted(rows, key=lambda x: x["avg_score"], reverse=True)
    out = {
        "csv_path": args.csv_path,
        "output_csv": args.output_csv,
        "best": best,
        "top10": rows[:10],
        "windows": windows,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
