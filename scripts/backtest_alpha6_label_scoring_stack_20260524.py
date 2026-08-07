#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha6_catboost_5head_policy_20260522 import _days, _fill_price  # noqa: E402
from scripts.alpha6_catboost_entry_quality_exit_policy_20260522 import (  # noqa: E402
    TARGET_BUCKET_TO_HORIZON,
    _exit_close_prob,
    _exit_state_vec,
    _threshold_for_bucket,
)
from scripts.analyze_alpha6_sleeve_complementarity_20260523 import Expert, _load_experts  # noqa: E402


PRIMARY = 0
COVERAGE = 1
CONFIRMERS = (2, 3)
RISKS = (4, 5)


def _desired(e: Expert, i: int) -> int:
    row = e.dec.iloc[i]
    return int(row.action) if float(row.quality_score) >= float(e.entry_threshold) else 0


def _edge(e: Expert, i: int) -> float:
    q = float(e.dec.iloc[i].quality_score)
    return float(np.clip((q - float(e.entry_threshold)) / max(abs(float(e.entry_threshold)), 1e-9), -3.0, 3.0))


def _stack_decision(
    experts: list[Expert],
    i: int,
    *,
    min_score: float,
    coverage_min_score: float,
    confirm_same_w: float,
    confirm_opp_w: float,
    risk_opp_w: float,
    risk_not_same_w: float,
    risk_same_credit: float,
    hard_double_risk_veto: bool,
    protect_primary: bool,
) -> tuple[int, int, float, dict[str, Any]]:
    primary_desired = _desired(experts[PRIMARY], i)
    coverage_desired = _desired(experts[COVERAGE], i)
    if primary_desired != 0:
        base_idx = PRIMARY
        side = primary_desired
        threshold = min_score
    elif coverage_desired != 0:
        base_idx = COVERAGE
        side = coverage_desired
        threshold = coverage_min_score
    else:
        return -1, 0, -999.0, {}

    base_edge = _edge(experts[base_idx], i)
    score = base_edge
    confirm_same = confirm_opp = risk_opp = risk_not_same = risk_same = 0
    details: dict[str, Any] = {
        "base": experts[base_idx].name,
        "base_edge": base_edge,
        "threshold": threshold,
    }
    for idx in CONFIRMERS:
        d = _desired(experts[idx], i)
        if d == side:
            confirm_same += 1
            score += confirm_same_w * max(_edge(experts[idx], i), 0.0)
        elif d != 0:
            confirm_opp += 1
            score -= confirm_opp_w * max(_edge(experts[idx], i), 0.0)
    for idx in RISKS:
        d = _desired(experts[idx], i)
        if d == side:
            risk_same += 1
            score += risk_same_credit * max(_edge(experts[idx], i), 0.0)
        elif d != 0:
            risk_opp += 1
            score -= risk_opp_w * max(_edge(experts[idx], i), 0.0)
        else:
            risk_not_same += 1
            score -= risk_not_same_w
    if hard_double_risk_veto and risk_opp >= 2:
        score = -999.0
    if protect_primary and base_idx == PRIMARY:
        score = max(score, threshold)
    details.update(
        {
            "confirm_same": confirm_same,
            "confirm_opp": confirm_opp,
            "risk_opp": risk_opp,
            "risk_not_same": risk_not_same,
            "risk_same": risk_same,
            "final_score": score,
        }
    )
    if score < threshold:
        return -1, 0, score, details
    return base_idx, side, score, details


def _backtest_stack(
    frame: pd.DataFrame,
    experts: list[Expert],
    *,
    fee: float,
    slip: float,
    min_exit_hold: int,
    state_horizon: int,
    start: int,
    end: int,
    params: dict[str, Any],
    save_trades: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    start = max(0, int(start))
    end = min(int(end), len(frame) - 2)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    active = -1
    entry = 0.0
    entry_idx = 0
    entry_equity = 1.0
    hold = 0
    mae = mfe = exposure = 0.0
    target_horizon = int(state_horizon)
    target_bucket = 4
    trades = wins = long_entries = short_entries = exit_model_closes = 0
    exits: dict[str, int] = {}
    trade_rows: list[dict[str, Any]] = []

    def equity(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * exposure)

    def enter(i: int, expert_idx: int, new_side: int, score: float, details: dict[str, Any]) -> None:
        nonlocal side, active, entry, entry_idx, entry_equity, hold, mae, mfe, exposure, target_horizon, target_bucket, cash
        e = experts[expert_idx]
        row = e.dec.iloc[i]
        fill_i = min(i + 1, len(frame) - 1)
        side = int(new_side)
        active = int(expert_idx)
        entry_idx = int(i)
        exposure = float(np.clip(float(row.notional), 0.01, 2.0))
        target_horizon = int(np.clip(int(getattr(row, "target_horizon", state_horizon)), 2, state_horizon))
        target_bucket = int(np.clip(int(getattr(row, "target_bucket", 4)), 0, 4))
        entry = _fill_price(frame, fill_i, side, slip, entry=True)
        entry_equity = cash
        cash -= cash * fee * exposure
        hold = 0
        mae = mfe = 0.0
        if save_trades:
            details["_entry_score"] = float(score)

    def exit_pos(i: int, reason: str, last_score: float = 0.0, last_details: dict[str, Any] | None = None) -> None:
        nonlocal side, active, entry, cash, hold, mae, mfe, exposure, target_horizon, target_bucket
        nonlocal trades, wins, long_entries, short_entries, exit_model_closes
        e = experts[active]
        fill_i = min(i + 1, len(frame) - 1)
        fill_px = _fill_price(frame, fill_i, side, slip, entry=False)
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * fee * exposure
        pnl_pct = (cash / max(entry_equity, 1e-12) - 1.0) * 100.0
        trades += 1
        wins += int(cash > entry_equity)
        long_entries += int(side > 0)
        short_entries += int(side < 0)
        exits[reason] = exits.get(reason, 0) + 1
        if save_trades:
            row = {
                "expert": e.name,
                "entry_idx": int(entry_idx),
                "exit_idx": int(i),
                "entry_time": str(frame.iloc[entry_idx]["timestamp"]),
                "exit_time": str(frame.iloc[int(i)]["timestamp"]),
                "side": "LONG" if side > 0 else "SHORT",
                "hold_bars": int(hold),
                "target_horizon": int(target_horizon),
                "target_bucket": int(target_bucket),
                "raw_ret": float(raw),
                "pnl_pct_on_equity": float(pnl_pct),
                "mae": float(mae),
                "mfe": float(mfe),
                "reason": reason,
                "last_score": float(last_score),
            }
            if last_details:
                row.update({f"last_{k}": v for k, v in last_details.items() if isinstance(v, (int, float, str))})
            trade_rows.append(row)
        side = 0
        active = -1
        entry = 0.0
        hold = 0
        mae = mfe = exposure = 0.0
        target_horizon = int(state_horizon)
        target_bucket = 4

    for i in range(start, end):
        cand_idx, cand_side, score, details = _stack_decision(experts, i, **params)
        if side != 0:
            hold += 1
            px = float(close[i])
            raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
            mae = max(mae, max(0.0, -raw * exposure))
            mfe = max(mfe, max(0.0, raw * exposure))
            if active == COVERAGE and cand_idx == PRIMARY:
                exit_pos(i, "primary_preempt", score, details)
                enter(i, PRIMARY, 1 if cand_side == 1 else -1, score, details)
            elif hold >= int(min_exit_hold):
                e = experts[active]
                expected = e.bundle.get("expected_return_by_bucket") or {k: 0.01 for k in TARGET_BUCKET_TO_HORIZON}
                exit_meta = e.bundle.get("exit_meta", {})
                state = _exit_state_vec(
                    frame,
                    side=side,
                    entry_idx=entry_idx,
                    current_idx=i,
                    entry_px=entry,
                    px=px,
                    hold=hold,
                    horizon=int(target_horizon),
                    mae=mae,
                    mfe=mfe,
                    target_bucket=target_bucket,
                    regime_drift=bool(exit_meta.get("regime_drift", False)),
                    capture_ratio=bool(exit_meta.get("capture_ratio", False)),
                    expected_return=float(expected.get(target_bucket, 0.01)),
                )
                close_prob = _exit_close_prob(e.bundle["exit_model"], e.x[i], state)
                if close_prob >= _threshold_for_bucket(e.exit_threshold, target_bucket):
                    exit_model_closes += 1
                    exit_pos(i, "exit_model", score, details)
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side == 0 and cand_idx >= 0:
            enter(i, cand_idx, 1 if cand_side == 1 else -1, score, details)
    if side != 0:
        exit_pos(end, "end")
    return (
        {
            "pnl": float((cash - 1.0) * 100.0),
            "mdd": float(mdd * 100.0),
            "calmar": float(((cash - 1.0) * 100.0) / max(abs(mdd * 100.0), 1e-12)),
            "trades": int(trades),
            "trades_per_day": float(trades / _days(frame.iloc[start : end + 1])),
            "wr": float(wins / max(trades, 1)),
            "long_entries": int(long_entries),
            "short_entries": int(short_entries),
            "exit_model_closes": int(exit_model_closes),
            "exits": exits,
        },
        pd.DataFrame(trade_rows),
    )


def _splits(n: int, purge: int) -> dict[str, tuple[int, int]]:
    train_end = int(n * 0.50)
    calib_end = int(n * 0.75)
    return {
        "meta_train": (0, max(0, train_end - purge)),
        "calib": (train_end, max(train_end, calib_end - purge)),
        "test": (calib_end, n - 2),
        "full_val": (0, n - 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="current_tail111")
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--purge-bars", type=int, default=96)
    ap.add_argument("--min-exit-hold", type=int, default=2)
    ap.add_argument("--state-horizon", type=int, default=96)
    ap.add_argument("--grid", choices=("coarse", "full"), default="coarse")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha6_label_scoring_stack_20260524")
    args = ap.parse_args()

    frame, experts = _load_experts(args.variant)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fee = 0.0004 * float(args.cost_mult)
    slip = 0.00015 * float(args.cost_mult)
    splits = _splits(len(frame), args.purge_bars)

    if args.grid == "full":
        min_scores = (0.0, 0.25, 0.5, 0.75, 1.0)
        coverage_min_scores = (0.0, 0.5, 1.0, 1.5)
        confirm_same_ws = (0.0, 0.15, 0.30)
        confirm_opp_ws = (0.0, 0.20, 0.40)
        risk_opp_ws = (0.0, 0.35, 0.70)
        risk_not_same_ws = (0.0, 0.10, 0.20)
    else:
        min_scores = (0.0, 0.5, 1.0)
        coverage_min_scores = (0.0, 1.0)
        confirm_same_ws = (0.0, 0.30)
        confirm_opp_ws = (0.0, 0.40)
        risk_opp_ws = (0.0, 0.70)
        risk_not_same_ws = (0.0, 0.20)

    grid: list[dict[str, Any]] = []
    for min_score in min_scores:
        for coverage_min_score in coverage_min_scores:
            for confirm_same_w in confirm_same_ws:
                for confirm_opp_w in confirm_opp_ws:
                    for risk_opp_w in risk_opp_ws:
                        for risk_not_same_w in risk_not_same_ws:
                            grid.append(
                                {
                                    "min_score": min_score,
                                    "coverage_min_score": coverage_min_score,
                                    "confirm_same_w": confirm_same_w,
                                    "confirm_opp_w": confirm_opp_w,
                                    "risk_opp_w": risk_opp_w,
                                    "risk_not_same_w": risk_not_same_w,
                                    "risk_same_credit": 0.05,
                                    "hard_double_risk_veto": True,
                                    "protect_primary": True,
                                }
                            )

    rows: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    calib_start, calib_end = splits["calib"]
    for idx, params in enumerate(grid, start=1):
        if idx == 1 or idx % 25 == 0 or idx == len(grid):
            print(f"[grid] {idx}/{len(grid)}", flush=True)
        bt, _ = _backtest_stack(
            frame,
            experts,
            fee=fee,
            slip=slip,
            min_exit_hold=args.min_exit_hold,
            state_horizon=args.state_horizon,
            start=calib_start,
            end=calib_end,
            params=params,
        )
        row = {"split": "calib", "grid_idx": idx, **params, **bt}
        rows.append(row)
        score = bt["calmar"] if bt["trades"] >= 8 else -1e6 + bt["pnl"]
        if best is None or score > best[0]:
            best = (score, params, bt)
    assert best is not None
    best_params = best[1]
    eval_rows: list[dict[str, Any]] = []
    trade_paths: dict[str, str] = {}
    for split, (start, end) in splits.items():
        bt, trades = _backtest_stack(
            frame,
            experts,
            fee=fee,
            slip=slip,
            min_exit_hold=args.min_exit_hold,
            state_horizon=args.state_horizon,
            start=start,
            end=end,
            params=best_params,
            save_trades=True,
        )
        eval_rows.append({"split": split, **best_params, **bt})
        path = args.out_dir / f"best_{split}_trades.csv"
        trades.to_csv(path, index=False)
        trade_paths[split] = str(path)
        print(f"[bt] split={split} {bt}", flush=True)

    pd.DataFrame(rows).sort_values("calmar", ascending=False).to_csv(args.out_dir / "calib_grid.csv", index=False)
    pd.DataFrame(eval_rows).to_csv(args.out_dir / "best_eval.csv", index=False)
    summary = {
        "best_params": best_params,
        "splits": splits,
        "cost_mult": float(args.cost_mult),
        "eval": eval_rows,
        "trade_paths": trade_paths,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"[out] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
