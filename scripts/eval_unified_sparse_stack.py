from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd


DEFAULT_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified_sparse_candidates.csv"
DEFAULT_GATE_META = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_sparse_meta_gate_catboost.json"
DEFAULT_OUT = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_sparse_stack_eval.json"
FEE = 0.0005
SLIP = 0.0002


def load_gate(meta_path: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model_path = meta["model_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(meta_path), model_path)
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["feature_cols"], float(obj.get("threshold", meta.get("threshold", 0.5)))


def _predict_gate(df: pd.DataFrame, model, feature_cols: list[str]) -> np.ndarray:
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    med = x.median(numeric_only=True)
    x = x.fillna(med)
    return model.predict_proba(x)[:, 1]


def run(df: pd.DataFrame, gate_thr: float, hold_scale: float, close_on_opp: bool) -> dict[str, float | int]:
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(np.float64)
    cand = pd.to_numeric(df["ud_cand_flag"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    side = pd.to_numeric(df["ud_cand_side"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    hold_arr = pd.to_numeric(df["ud_cand_hold"], errors="coerce").fillna(6).astype(np.int32).to_numpy()
    gate = pd.to_numeric(df["ud_sparse_gate_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)

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
        gated = cand[i] == 1 and side[i] != 0 and gate[i] >= gate_thr
        if pos == 0:
            if gated:
                pos = int(side[i])
                entry = close[i] * (1.0 + SLIP) if pos == 1 else close[i] * (1.0 - SLIP)
                balance *= (1.0 - FEE)
                hold = 0
                target_hold = max(2, int(round(float(hold_arr[i]) * hold_scale)))
                longs += int(pos == 1)
                shorts += int(pos == -1)
        else:
            hold += 1
            reverse = gated and side[i] == -pos and close_on_opp
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
        peak = max(peak, balance)
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


def score(res: dict[str, float | int]) -> float:
    s = float(res["pnl_pct"]) - 0.35 * abs(min(float(res["mdd_pct"]), 0.0))
    if int(res["trades"]) < 15:
        s -= 8.0
    if int(res["trades"]) > 200:
        s -= 0.05 * (int(res["trades"]) - 200)
    return float(s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate sparse direction + sparse gate stack")
    ap.add_argument("--csv-path", default=DEFAULT_CSV)
    ap.add_argument("--gate-meta-path", default=DEFAULT_GATE_META)
    ap.add_argument("--output-path", default=DEFAULT_OUT)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
    model, feature_cols, base_thr = load_gate(args.gate_meta_path)
    df["ud_sparse_gate_prob"] = _predict_gate(df, model, feature_cols)

    calib = df[df["ud_cat_is_holdout"] == 0].copy().reset_index(drop=True)
    test = df[df["ud_cat_is_holdout"] == 1].copy().reset_index(drop=True)
    folds = sorted(int(x) for x in pd.to_numeric(calib["ud_cat_oof_fold"], errors="coerce").dropna().unique() if x >= 0)
    recent = folds[-4:]
    fold_col = pd.to_numeric(calib["ud_cat_oof_fold"], errors="coerce").fillna(-1).astype(np.int32).to_numpy()
    windows = []
    for fid in recent:
        idx = np.flatnonzero(fold_col == fid)
        if len(idx):
            windows.append((int(idx[0]), int(idx[-1] + 1)))

    best = None
    rows = []
    thr_grid = sorted(set([round(base_thr, 3), 0.45, 0.50, 0.55, 0.65]))
    for gate_thr, hold_scale, close_on_opp in [(x, y, z) for x in thr_grid for y in [0.8, 1.0, 1.2] for z in [False, True]]:
        window_res = []
        scores = []
        for ws, we in windows:
            res = run(calib.iloc[ws:we].reset_index(drop=True), gate_thr, hold_scale, close_on_opp)
            sc = score(res)
            window_res.append({"start": ws, "end": we, "result": res, "score": sc})
            scores.append(sc)
        row = {
            "params": {"gate_thr": gate_thr, "hold_scale": hold_scale, "close_on_opp": close_on_opp},
            "avg_score": float(np.mean(scores)),
            "avg_pnl_pct": float(np.mean([w["result"]["pnl_pct"] for w in window_res])),
            "avg_trades": float(np.mean([w["result"]["trades"] for w in window_res])),
            "windows": window_res,
        }
        rows.append(row)
        if best is None or row["avg_score"] > best["avg_score"]:
            best = row
    assert best is not None
    best["test"] = run(test, best["params"]["gate_thr"], best["params"]["hold_scale"], best["params"]["close_on_opp"]) if len(test) else None
    best["test_score"] = score(best["test"]) if best["test"] is not None else None
    rows = sorted(rows, key=lambda x: x["avg_score"], reverse=True)
    out = {"best": best, "top10": rows[:10]}
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
