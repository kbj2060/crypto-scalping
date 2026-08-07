from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified_sparse_gate_oof.csv"
DEFAULT_OUT = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_sparse_handcrafted_veto_eval.json"
FEE = 0.0005
SLIP = 0.0002
REGIMES = ("bull", "bear", "chop", "whipsaw", "normal")
MODES = ("skip", "long", "short", "both")


@dataclass
class EvalResult:
    pnl_pct: float
    trades: int
    wr_pct: float
    mdd_pct: float
    longs: int
    shorts: int


def _regime_name(df: pd.DataFrame) -> np.ndarray:
    return (
        df[["regime_bull", "regime_bear", "regime_chop", "regime_whipsaw", "regime_normal"]]
        .idxmax(axis=1)
        .str.replace("regime_", "", regex=False)
        .to_numpy()
    )


def _run(df: pd.DataFrame, rule: dict[str, str], hold_scale: float, close_on_opp: bool) -> EvalResult:
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(np.float64)
    cand = pd.to_numeric(df["ud_cand_flag"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    side = pd.to_numeric(df["ud_cand_side"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    hold_arr = pd.to_numeric(df["ud_cand_hold"], errors="coerce").fillna(6).astype(np.int32).to_numpy()
    regimes = _regime_name(df)

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
        allowed = False
        if cand[i] == 1 and side[i] != 0:
            mode = rule[regimes[i]]
            allowed = (
                mode == "both"
                or (mode == "long" and side[i] == 1)
                or (mode == "short" and side[i] == -1)
            )
        if pos == 0:
            if allowed:
                pos = int(side[i])
                entry = close[i] * (1.0 + SLIP) if pos == 1 else close[i] * (1.0 - SLIP)
                balance *= (1.0 - FEE)
                hold = 0
                target_hold = max(2, int(round(float(hold_arr[i]) * hold_scale)))
                longs += int(pos == 1)
                shorts += int(pos == -1)
        else:
            hold += 1
            reverse = allowed and side[i] == -pos and close_on_opp
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

    return EvalResult(
        pnl_pct=float((balance - 1.0) * 100.0),
        trades=int(trades),
        wr_pct=float(wins / max(trades, 1) * 100.0),
        mdd_pct=float(mdd * 100.0),
        longs=int(longs),
        shorts=int(shorts),
    )


def _avg_score(res: EvalResult) -> float:
    s = float(res.pnl_pct) - 0.35 * abs(min(float(res.mdd_pct), 0.0))
    if int(res.trades) < 8:
        s -= 6.0
    if int(res.trades) > 120:
        s -= 0.05 * (int(res.trades) - 120)
    return float(s)


def _robust_score_1(pnls: np.ndarray, trades: np.ndarray, mdds: np.ndarray) -> float:
    s = (
        float(np.mean(pnls))
        + 0.5 * float(np.median(pnls))
        + 0.25 * float(np.min(pnls))
        - 0.25 * float(np.mean(np.abs(np.minimum(mdds, 0.0))))
    )
    if float(np.mean(trades)) < 4.0:
        s -= 4.0
    return float(s)


def _selector_candidates(rows: list[dict[str, Any]], min_avg_trades: float) -> list[dict[str, Any]]:
    return [r for r in rows if float(r["avg_trades"]) >= min_avg_trades]


def _pick_best(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    if not rows:
        return None
    if key == "avg_score":
        return max(rows, key=lambda r: (float(r["avg_score"]), float(r["median_pnl"])))
    if key == "robust1":
        return max(rows, key=lambda r: (float(r["robust1"]), float(r["median_pnl"])))
    raise ValueError(f"unknown key: {key}")


def _json_ready(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out = dict(row)
    out["windows"] = [
        {
            "fold": int(w["fold"]),
            "start": int(w["start"]),
            "end": int(w["end"]),
            "result": w["result"],
            "avg_score": float(w["avg_score"]),
        }
        for w in row["windows"]
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Search handcrafted sparse regime/side veto rules")
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

    fold_col = pd.to_numeric(calib["ud_cat_oof_fold"], errors="coerce").fillna(-1).astype(np.int32).to_numpy()
    recent_folds = sorted(int(x) for x in np.unique(fold_col) if x >= 0)[-4:]
    windows: list[tuple[int, int, int]] = []
    for fid in recent_folds:
        idx = np.flatnonzero(fold_col == fid)
        if len(idx):
            windows.append((int(idx[0]), int(idx[-1] + 1), fid))

    rows: list[dict[str, Any]] = []
    for combo in itertools.product(MODES, repeat=len(REGIMES)):
        rule = dict(zip(REGIMES, combo))
        if all(v == "skip" for v in rule.values()):
            continue
        for hold_scale in (0.8, 1.0, 1.2, 1.4):
            for close_on_opp in (False, True):
                per_fold = []
                pnl_vals = []
                trade_vals = []
                mdd_vals = []
                for ws, we, fid in windows:
                    res = _run(calib.iloc[ws:we].reset_index(drop=True), rule, hold_scale, close_on_opp)
                    sc = _avg_score(res)
                    per_fold.append(
                        {
                            "fold": fid,
                            "start": ws,
                            "end": we,
                            "result": asdict(res),
                            "avg_score": sc,
                        }
                    )
                    pnl_vals.append(res.pnl_pct)
                    trade_vals.append(res.trades)
                    mdd_vals.append(res.mdd_pct)
                pnls = np.asarray(pnl_vals, dtype=np.float64)
                trades = np.asarray(trade_vals, dtype=np.float64)
                mdds = np.asarray(mdd_vals, dtype=np.float64)
                row = {
                    "rule": rule,
                    "hold_scale": float(hold_scale),
                    "close_on_opp": bool(close_on_opp),
                    "avg_score": float(np.mean([w["avg_score"] for w in per_fold])),
                    "robust1": _robust_score_1(pnls, trades, mdds),
                    "avg_pnl": float(np.mean(pnls)),
                    "median_pnl": float(np.median(pnls)),
                    "min_pnl": float(np.min(pnls)),
                    "avg_trades": float(np.mean(trades)),
                    "pos_folds": int((pnls > 0).sum()),
                    "windows": per_fold,
                }
                rows.append(row)

    rows_by_avg = sorted(rows, key=lambda r: (float(r["avg_score"]), float(r["median_pnl"])), reverse=True)
    rows_by_robust = sorted(rows, key=lambda r: (float(r["robust1"]), float(r["median_pnl"])), reverse=True)

    selectors = {
        "best_avg_score": _pick_best(rows, "avg_score"),
        "best_avg_score_min_trades_3": _pick_best(_selector_candidates(rows, 3.0), "avg_score"),
        "best_avg_score_min_trades_5": _pick_best(_selector_candidates(rows, 5.0), "avg_score"),
        "best_robust1": _pick_best(rows, "robust1"),
        "best_robust1_min_trades_3": _pick_best(_selector_candidates(rows, 3.0), "robust1"),
        "best_robust1_min_trades_5": _pick_best(_selector_candidates(rows, 5.0), "robust1"),
    }

    report: dict[str, Any] = {
        "csv_path": args.csv_path,
        "recent_folds": recent_folds,
        "windows": [{"start": s, "end": e, "fold": fid} for s, e, fid in windows],
        "selectors": {},
        "top10_by_avg_score": [_json_ready(r) for r in rows_by_avg[:10]],
        "top10_by_robust1": [_json_ready(r) for r in rows_by_robust[:10]],
    }
    for name, row in selectors.items():
        item = _json_ready(row)
        if item is not None:
            item["holdout"] = asdict(
                _run(test, item["rule"], float(item["hold_scale"]), bool(item["close_on_opp"]))
            )
        report["selectors"][name] = item

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
