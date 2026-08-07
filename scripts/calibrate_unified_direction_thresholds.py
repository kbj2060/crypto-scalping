from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified_dircat_oof.csv"
DEFAULT_OUT = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_direction_thresholds.json"
FEE = 0.0005
SLIP = 0.0002


def _prep(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for c in (
        "close",
        "ud_cat_long_prob",
        "ud_cat_flat_prob",
        "ud_cat_short_prob",
        "ud_cat_edge",
        "smart_money_flow",
        "taker_acceleration",
        "trade_intensity",
        "garch_vol_z",
        "regime_bull",
        "regime_bear",
        "regime_chop",
        "regime_whipsaw",
        "regime_normal",
    ):
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(np.float64)
    out["close"] = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(np.float64)
    out["ts"] = pd.to_datetime(df["timestamp"], errors="coerce").astype("int64", copy=False).to_numpy() if "timestamp" in df.columns else np.arange(len(df))
    out["rows"] = np.arange(len(df), dtype=np.int32)
    return out


def _side_signal(i: int, data: dict[str, np.ndarray], p: dict[str, float]) -> int:
    lp = data["ud_cat_long_prob"][i]
    sp = data["ud_cat_short_prob"][i]
    fp = data["ud_cat_flat_prob"][i]
    edge = data["ud_cat_edge"][i]
    long_prob_min = p["long_prob_min"]
    short_prob_min = p["short_prob_min"]
    long_gap = p["long_gap"]
    short_gap = p["short_gap"]

    if data.get("regime_bull", np.zeros_like(data["close"]))[i] > 0.5:
        short_prob_min += p["bull_short_penalty"]
    if data.get("regime_whipsaw", np.zeros_like(data["close"]))[i] > 0.5:
        short_prob_min += p["whipsaw_short_penalty"]

    if lp >= max(sp + long_gap, fp + p["flat_gap"], long_prob_min) and edge >= p["long_edge_min"]:
        return 1
    if sp >= max(lp + short_gap, fp + p["flat_gap"], short_prob_min) and edge <= -p["short_edge_min"]:
        return -1
    return 0


def run_strategy(df: pd.DataFrame, p: dict[str, float]) -> dict[str, Any]:
    data = _prep(df)
    pos = 0
    entry = 0.0
    hold = 0
    balance = 1.0
    peak = 1.0
    mdd = 0.0
    trades = 0
    wins = 0
    longs = 0
    shorts = 0
    trade_rows: list[dict[str, Any]] = []

    for i in range(len(df)):
        signal = _side_signal(i, data, p)
        lp = data["ud_cat_long_prob"][i]
        sp = data["ud_cat_short_prob"][i]
        fp = data["ud_cat_flat_prob"][i]

        if pos == 0:
            if signal == 1:
                pos = 1
                entry = data["close"][i] * (1.0 + SLIP)
                balance *= (1.0 - FEE)
                hold = 0
                longs += 1
            elif signal == -1:
                pos = -1
                entry = data["close"][i] * (1.0 - SLIP)
                balance *= (1.0 - FEE)
                hold = 0
                shorts += 1
        else:
            hold += 1
            reverse_signal = (pos == 1 and signal == -1) or (pos == -1 and signal == 1)
            weak_side = (pos == 1 and lp < p["hold_side_prob_min"]) or (pos == -1 and sp < p["hold_side_prob_min"])
            flat_exit = fp >= p["flat_exit_prob"]
            should_close = reverse_signal or weak_side or flat_exit or (hold >= int(p["max_hold"]))
            if should_close:
                fill = data["close"][i] * (1.0 - SLIP) if pos == 1 else data["close"][i] * (1.0 + SLIP)
                pnl = ((fill - entry) / entry) if pos == 1 else ((entry - fill) / entry)
                balance *= max(1e-8, (1.0 + pnl) * (1.0 - FEE))
                trades += 1
                if pnl > 0:
                    wins += 1
                trade_rows.append(
                    {
                        "idx": int(i),
                        "side": "LONG" if pos == 1 else "SHORT",
                        "pnl_pct": float(pnl * 100.0),
                        "hold": int(hold),
                        "lp": float(lp),
                        "sp": float(sp),
                        "fp": float(fp),
                    }
                )
                pos = 0
                entry = 0.0
                hold = 0
        if balance > peak:
            peak = balance
        dd = balance / max(peak, 1e-8) - 1.0
        if dd < mdd:
            mdd = dd

    if pos != 0:
        fill = data["close"][-1] * (1.0 - SLIP) if pos == 1 else data["close"][-1] * (1.0 + SLIP)
        pnl = ((fill - entry) / entry) if pos == 1 else ((entry - fill) / entry)
        balance *= max(1e-8, (1.0 + pnl) * (1.0 - FEE))
        trades += 1
        if pnl > 0:
            wins += 1

    return {
        "pnl_pct": float((balance - 1.0) * 100.0),
        "trades": int(trades),
        "wr_pct": float(wins / max(trades, 1) * 100.0),
        "mdd_pct": float(mdd * 100.0),
        "longs": int(longs),
        "shorts": int(shorts),
        "trade_rows_tail": trade_rows[-10:],
    }


def score_result(res: dict[str, Any]) -> float:
    score = res["pnl_pct"] - 0.40 * abs(min(res["mdd_pct"], 0.0))
    if res["trades"] < 40:
        score -= 12.0
    if res["trades"] > 450:
        score -= 0.05 * (res["trades"] - 450)
    return float(score)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate deterministic thresholds on OOF CatBoost direction probabilities")
    ap.add_argument("--csv-path", default=DEFAULT_CSV)
    ap.add_argument("--output-path", default=DEFAULT_OUT)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    calib = df[df["ud_cat_is_holdout"] == 0].copy().reset_index(drop=True)
    test = df[df["ud_cat_is_holdout"] == 1].copy().reset_index(drop=True)
    covered = np.isfinite(pd.to_numeric(calib["ud_cat_long_prob"], errors="coerce"))
    calib = calib.loc[covered].reset_index(drop=True)

    n = len(calib)
    if n < 1000:
        raise ValueError(f"not enough calibration rows: {n}")

    fold_ids = sorted(int(x) for x in pd.to_numeric(calib["ud_cat_oof_fold"], errors="coerce").dropna().unique() if x >= 0)
    if len(fold_ids) < 2:
        raise ValueError("not enough OOF folds for calibration")
    recent_folds = fold_ids[-4:]
    windows = []
    for fid in recent_folds:
        idx = np.flatnonzero(pd.to_numeric(calib["ud_cat_oof_fold"], errors="coerce").to_numpy(np.int32) == fid)
        if len(idx) == 0:
            continue
        windows.append((int(idx[0]), int(idx[-1] + 1)))

    grid = itertools.product(
        [0.50, 0.54],
        [0.50, 0.54],
        [0.03, 0.06],
        [0.05, 0.10],
        [0.03, 0.08],
        [0.05, 0.10],
        [0.01],
        [0.45, 0.55],
        [0.40, 0.48],
        [6, 8],
        [0.00, 0.04],
        [0.00],
    )

    best: dict[str, Any] | None = None
    top_rows: list[dict[str, Any]] = []
    for vals in grid:
        p = {
            "long_prob_min": vals[0],
            "short_prob_min": vals[1],
            "long_edge_min": vals[2],
            "short_edge_min": vals[3],
            "long_gap": vals[4],
            "short_gap": vals[5],
            "flat_gap": vals[6],
            "flat_exit_prob": vals[7],
            "hold_side_prob_min": vals[8],
            "max_hold": vals[9],
            "bull_short_penalty": vals[10],
            "whipsaw_short_penalty": vals[11],
        }
        window_results = []
        scores = []
        for ws, we in windows:
            res = run_strategy(calib.iloc[ws:we].reset_index(drop=True), p)
            sc = score_result(res)
            window_results.append({"start": int(ws), "end": int(we), "result": res, "score": sc})
            scores.append(sc)
        avg_score = float(np.mean(scores))
        avg_pnl = float(np.mean([w["result"]["pnl_pct"] for w in window_results]))
        avg_trades = float(np.mean([w["result"]["trades"] for w in window_results]))
        row = {
            "params": p,
            "avg_score": avg_score,
            "avg_pnl_pct": avg_pnl,
            "avg_trades": avg_trades,
            "windows": window_results,
        }
        top_rows.append(row)
        if best is None or avg_score > best["avg_score"]:
            best = row

    assert best is not None
    best["test"] = run_strategy(test, best["params"]) if len(test) > 0 else None
    best["test_score"] = score_result(best["test"]) if best["test"] is not None else None
    top_rows = sorted(top_rows, key=lambda x: x["avg_score"], reverse=True)

    out = {
        "csv_path": args.csv_path,
        "windows": windows,
        "best": best,
        "top10": top_rows[:10],
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
