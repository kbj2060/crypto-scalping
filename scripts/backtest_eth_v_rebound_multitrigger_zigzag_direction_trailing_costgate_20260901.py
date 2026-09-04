#!/usr/bin/env python3
"""ZDC(wick-앵커) 9트리거 통합모델 trailing-stop cost-gate — 계획서 Step E 2/2.

`backtest_eth_sweep_v_rebound_v7b_trailing_costgate_20260830.py`(giveback v7b판)의
simulate_trailing/grid_search/split_report 로직을 그대로 재사용(재구현 아님) — SL x ARM x Trail
205셀 그리드, 왕복비용 10bp, optimistic/pessimistic 봉내순서 이중검증. 유일한 변경점은 CANDIDATES
경로와 결과를 report.json으로도 저장하는 것뿐.

계획서 Step E 중단기준: VAL+OOS 동시양수 조합이 0개면 HOLDOUT(Step F) 진행 안 함.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data/research/eth_v_rebound_multitrigger_zigzag_direction_costgate_20260901/zdc_costgate_candidates.pkl"
OUT_DIR = ROOT / "data/research/eth_v_rebound_multitrigger_zigzag_direction_costgate_20260901"
STANDARD_COST_BP = 10.0
SL_RACE_WINDOW_BARS = 11  # ~60min from entry, kept identical to v7b for cross-model diagnostic comparability


def sl_race_diagnostic(df: pd.DataFrame) -> list[dict]:
    winners = df[df["label"] == 1]
    print(f"\n=== SL-race diagnostic (label==1 winners within the called population, n={len(winners)}) ===")
    out = []
    for sl_mult in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        raced = 0
        for _, row in winners.iterrows():
            atr = row["atr"]
            entry = row["entry_price"]
            highs = np.array(row["fwd_high"][:SL_RACE_WINDOW_BARS])
            lows = np.array(row["fwd_low"][:SL_RACE_WINDOW_BARS])
            if row["side"] == "long":
                adverse = entry - sl_mult * atr
                hit = (lows <= adverse).any()
            else:
                adverse = entry + sl_mult * atr
                hit = (highs >= adverse).any()
            raced += int(hit)
        rate = raced / len(winners) if len(winners) else float("nan")
        print(f"  SL={sl_mult:.1f}x: race-loss rate {rate:.1%} ({raced}/{len(winners)})")
        out.append({"sl_mult": sl_mult, "race_loss_rate": rate, "raced": raced, "n_winners": len(winners)})
    return out


def simulate_trailing(row: pd.Series, sl_mult: float, arm_mult: float, trail_mult: float,
                       pessimistic: bool) -> float:
    """Returns exit price-move as a fraction of entry (signed so positive=profit for the trade's
    own side). pessimistic=True assumes, within any single bar where BOTH the stop level and a
    new favorable extreme could plausibly occur, the WORST-case ordering (stop hit first, no
    credit for that bar's favorable excursion) -- optimistic=False assumes the opposite (favorable
    move happens first, stop only checked after)."""
    atr = row["atr"]
    entry = row["entry_price"]
    side = row["side"]
    opens, highs, lows, closes = row["fwd_open"], row["fwd_high"], row["fwd_low"], row["fwd_close"]
    sign = 1.0 if side == "long" else -1.0

    stop = entry - sign * sl_mult * atr
    armed = False
    best = entry
    for o, h, l, c in zip(opens, highs, lows, closes):
        fav_extreme = h if side == "long" else l  # this bar's most-favorable price reached
        adv_extreme = l if side == "long" else h  # this bar's most-adverse price reached

        def stop_hit() -> bool:
            return (adv_extreme <= stop) if side == "long" else (adv_extreme >= stop)

        def update_trailing() -> None:
            nonlocal armed, stop, best
            if sign * (fav_extreme - best) > 0:
                best = fav_extreme
            if not armed and sign * (best - entry) >= arm_mult * atr:
                armed = True
            if armed:
                new_stop = best - sign * trail_mult * atr
                if sign * (new_stop - stop) > 0:
                    stop = new_stop

        if pessimistic:
            if stop_hit():
                return sign * (stop - entry) / entry
            update_trailing()
        else:
            update_trailing()
            if stop_hit():
                return sign * (stop - entry) / entry
    return sign * (closes[-1] - entry) / entry  # never stopped out in the buffer -- close at last available price


def buffer_exhaustion_rate(df: pd.DataFrame, sl_mult: float, arm_mult: float, trail_mult: float) -> float:
    """Fraction of candidates that never triggered the (pessimistic) stop within FORWARD_BARS --
    i.e. got force-closed at the buffer's last available close. Plan-mandated Step E check."""
    def never_stopped(row: pd.Series) -> bool:
        atr, entry, side = row["atr"], row["entry_price"], row["side"]
        sign = 1.0 if side == "long" else -1.0
        stop = entry - sign * sl_mult * atr
        armed, best = False, entry
        for h, l in zip(row["fwd_high"], row["fwd_low"]):
            adv_extreme = l if side == "long" else h
            fav_extreme = h if side == "long" else l
            if (adv_extreme <= stop) if side == "long" else (adv_extreme >= stop):
                return False
            if sign * (fav_extreme - best) > 0:
                best = fav_extreme
            if not armed and sign * (best - entry) >= arm_mult * atr:
                armed = True
            if armed:
                new_stop = best - sign * trail_mult * atr
                if sign * (new_stop - stop) > 0:
                    stop = new_stop
        return True
    return float(df.apply(never_stopped, axis=1).mean())


def grid_search(df: pd.DataFrame) -> list[tuple]:
    print("\n=== Trailing-stop grid (VAL+OOS combined, bp net of 10bp standard cost) ===")
    print(f"{'SL':>5} {'ARM':>5} {'Trail':>6} | {'opt(bp)':>9} {'pess(bp)':>9} {'diverge':>8}")
    best = []
    for sl in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        for arm in (0.10, 0.25, 0.5, 0.75, 1.0, 1.5):
            for trail in (0.10, 0.15, 0.2, 0.3, 0.5):
                if arm >= sl:
                    continue
                opt_moves = df.apply(lambda r: simulate_trailing(r, sl, arm, trail, False), axis=1)
                pess_moves = df.apply(lambda r: simulate_trailing(r, sl, arm, trail, True), axis=1)
                opt_bp = (opt_moves.mean() * 1e4) - STANDARD_COST_BP
                pess_bp = (pess_moves.mean() * 1e4) - STANDARD_COST_BP
                diverge = float((np.sign(opt_moves) != np.sign(pess_moves)).mean())
                best.append((sl, arm, trail, opt_bp, pess_bp, diverge))
    best.sort(key=lambda x: min(x[3], x[4]), reverse=True)
    for sl, arm, trail, opt_bp, pess_bp, diverge in best[:15]:
        print(f"{sl:>5.2f} {arm:>5.2f} {trail:>6.2f} | {opt_bp:>9.2f} {pess_bp:>9.2f} {diverge:>7.1%}")
    return best


def split_report(df: pd.DataFrame, sl: float, arm: float, trail: float) -> dict:
    out = {}
    for split_name in ("val", "oos"):
        sub = df[df["split"] == split_name]
        opt = sub.apply(lambda r: simulate_trailing(r, sl, arm, trail, False), axis=1)
        pess = sub.apply(lambda r: simulate_trailing(r, sl, arm, trail, True), axis=1)
        opt_bp = float(opt.mean() * 1e4 - STANDARD_COST_BP)
        pess_bp = float(pess.mean() * 1e4 - STANDARD_COST_BP)
        win_rate = float((opt > 0).mean())
        print(f"  {split_name}: n={len(sub)}  opt={opt_bp:+.2f}bp  pess={pess_bp:+.2f}bp  win_rate={win_rate:.1%}")
        out[split_name] = {"n": int(len(sub)), "opt_bp": opt_bp, "pess_bp": pess_bp, "win_rate": win_rate}
    return out


def main() -> int:
    df = pd.read_pickle(CANDIDATES)
    n_val, n_oos = int((df["split"] == "val").sum()), int((df["split"] == "oos").sum())
    print(f"candidates: {len(df)} (val={n_val}, oos={n_oos})")
    print(f"label==1 rate within called: {df['label'].mean():.4f}")

    sl_race = sl_race_diagnostic(df)
    best = grid_search(df)

    # Step E stop-criterion: for each grid cell, check VAL and OOS independently (both opt AND
    # pess must be positive in EACH split) -- combined-sort alone can hide a split that's negative.
    val_oos_positive_configs = []
    for sl, arm, trail, opt_bp, pess_bp, diverge in best:
        sub_val = df[df["split"] == "val"]
        sub_oos = df[df["split"] == "oos"]
        val_opt = sub_val.apply(lambda r: simulate_trailing(r, sl, arm, trail, False), axis=1)
        val_pess = sub_val.apply(lambda r: simulate_trailing(r, sl, arm, trail, True), axis=1)
        oos_opt = sub_oos.apply(lambda r: simulate_trailing(r, sl, arm, trail, False), axis=1)
        oos_pess = sub_oos.apply(lambda r: simulate_trailing(r, sl, arm, trail, True), axis=1)
        val_pos = (val_opt.mean() * 1e4 - STANDARD_COST_BP) > 0 and (val_pess.mean() * 1e4 - STANDARD_COST_BP) > 0
        oos_pos = (oos_opt.mean() * 1e4 - STANDARD_COST_BP) > 0 and (oos_pess.mean() * 1e4 - STANDARD_COST_BP) > 0
        if val_pos and oos_pos:
            val_oos_positive_configs.append((sl, arm, trail))
    n_val_oos_pos = len(val_oos_positive_configs)
    print(f"\n=== Step E stop-criterion check: configs with VAL AND OOS both opt+pess positive: {n_val_oos_pos}/{len(best)} ===")

    print("\n=== Best config (top of combined-sort), VAL/OOS independently ===")
    sl, arm, trail = best[0][0], best[0][1], best[0][2]
    print(f"config: SL={sl} ARM={arm} Trail={trail}")
    econ = split_report(df, sl, arm, trail)
    buf_rate = buffer_exhaustion_rate(df, sl, arm, trail)
    print(f"  buffer exhaustion rate (never stopped within FORWARD_BARS, pessimistic ordering): {buf_rate:.1%}")

    report = {
        "n_candidates": {"total": len(df), "val": n_val, "oos": n_oos},
        "label_1_rate_within_called": float(df["label"].mean()),
        "sl_race_diagnostic": sl_race,
        "grid_top15": [{"sl": s, "arm": a, "trail": t, "opt_bp": o, "pess_bp": p, "diverge": d}
                        for s, a, t, o, p, d in best[:15]],
        "n_val_and_oos_both_positive_configs": n_val_oos_pos,
        "n_grid_cells": len(best),
        "val_oos_positive_configs": val_oos_positive_configs,
        "selected_config": {"sl": sl, "arm": arm, "trail": trail, "selected_on": "val+oos combined-sort top1"},
        "economics": econ,
        "buffer_exhaustion_rate_selected_config": buf_rate,
        "note": "HOLDOUT untouched -- VAL+OOS only, per plan Step E.",
    }
    (OUT_DIR / "costgate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\nWrote {OUT_DIR / 'costgate_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
