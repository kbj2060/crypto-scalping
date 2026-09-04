#!/usr/bin/env python3
"""ZDC(완전wick) 9트리거 통합모델 trailing-stop cost-gate -- wick-앵커판 대비 두 단계(그리드서치
+방향뒤집기 대조군)를 한 스크립트로 통합(wick-앵커판에서 아티팩트를 사후발견해 별도 스크립트로
땜빵했던 것과 달리, 이번엔 처음부터 알고 있으므로 한 번에 처리).

`backtest_eth_v_rebound_multitrigger_zigzag_direction_trailing_costgate_20260901.py`(wick-앵커판)의
simulate_trailing() 로직 그대로 재사용(재구현 아님). CANDIDATES 경로만 다름.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data/research/eth_v_rebound_multitrigger_zigzag_direction_full_wick_costgate_20260901/zdc_fullwick_costgate_candidates.pkl"
OUT_DIR = ROOT / "data/research/eth_v_rebound_multitrigger_zigzag_direction_full_wick_costgate_20260901"
STANDARD_COST_BP = 10.0


def simulate_trailing(row: pd.Series, sl_mult: float, arm_mult: float, trail_mult: float,
                       pessimistic: bool) -> float:
    atr = row["atr"]
    entry = row["entry_price"]
    side = row["side"]
    opens, highs, lows, closes = row["fwd_open"], row["fwd_high"], row["fwd_low"], row["fwd_close"]
    sign = 1.0 if side == "long" else -1.0
    stop = entry - sign * sl_mult * atr
    armed = False
    best = entry
    for o, h, l, c in zip(opens, highs, lows, closes):
        fav_extreme = h if side == "long" else l
        adv_extreme = l if side == "long" else h

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
    return sign * (closes[-1] - entry) / entry


def split_metrics(df: pd.DataFrame, sl: float, arm: float, trail: float) -> dict:
    out = {}
    for split_name in ("val", "oos"):
        sub = df[df["split"] == split_name]
        opt = sub.apply(lambda r: simulate_trailing(r, sl, arm, trail, False), axis=1)
        pess = sub.apply(lambda r: simulate_trailing(r, sl, arm, trail, True), axis=1)
        out[split_name] = {
            "n": int(len(sub)),
            "opt_bp": float(opt.mean() * 1e4 - STANDARD_COST_BP),
            "pess_bp": float(pess.mean() * 1e4 - STANDARD_COST_BP),
            "win_rate": float((opt > 0).mean()),
        }
    return out


def main() -> int:
    df = pd.read_pickle(CANDIDATES)
    flipped = df.copy()
    flipped["side"] = flipped["side"].map({"long": "short", "short": "long"})
    n_val, n_oos = int((df["split"] == "val").sum()), int((df["split"] == "oos").sum())
    print(f"candidates: {len(df)} (val={n_val}, oos={n_oos})")
    print(f"label==1 rate within called: {df['label'].mean():.4f}")

    print("\n=== Trailing-stop grid (VAL+OOS combined, bp net of 10bp standard cost) ===")
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
                best.append((sl, arm, trail, opt_bp, pess_bp))
    best.sort(key=lambda x: min(x[3], x[4]), reverse=True)
    print(f"{'SL':>5} {'ARM':>5} {'Trail':>6} | {'opt(bp)':>9} {'pess(bp)':>9}")
    for sl, arm, trail, opt_bp, pess_bp in best[:15]:
        print(f"{sl:>5.2f} {arm:>5.2f} {trail:>6.2f} | {opt_bp:>9.2f} {pess_bp:>9.2f}")

    val_oos_positive = []
    for sl, arm, trail, _, _ in best:
        m = split_metrics(df, sl, arm, trail)
        if (min(m["val"]["opt_bp"], m["val"]["pess_bp"]) > 0
                and min(m["oos"]["opt_bp"], m["oos"]["pess_bp"]) > 0):
            val_oos_positive.append((sl, arm, trail))
    print(f"\n=== VAL AND OOS both opt+pess positive: {len(val_oos_positive)}/{len(best)} ===")

    print(f"\n=== Direction-flip control on {len(val_oos_positive)} VAL+OOS-positive configs ===")
    flip_results = []
    for sl, arm, trail in val_oos_positive:
        real = split_metrics(df, sl, arm, trail)
        flip = split_metrics(flipped, sl, arm, trail)
        real_val_min = min(real["val"]["opt_bp"], real["val"]["pess_bp"])
        real_oos_min = min(real["oos"]["opt_bp"], real["oos"]["pess_bp"])
        flip_val_min = min(flip["val"]["opt_bp"], flip["val"]["pess_bp"])
        flip_oos_min = min(flip["oos"]["opt_bp"], flip["oos"]["pess_bp"])
        genuine = (real_val_min > flip_val_min) and (real_oos_min > flip_oos_min) and real_val_min > 0 and real_oos_min > 0
        gap_val, gap_oos = real_val_min - flip_val_min, real_oos_min - flip_oos_min
        row = {"sl": sl, "arm": arm, "trail": trail, "real": real, "flipped": flip,
               "gap_val_bp": gap_val, "gap_oos_bp": gap_oos, "genuine": genuine}
        flip_results.append(row)
        tag = "GENUINE" if genuine else "artifact"
        print(f"  SL={sl:.2f} ARM={arm:.2f} Trail={trail:.2f} | real(val={real_val_min:+.2f} oos={real_oos_min:+.2f}) "
              f"flip(val={flip_val_min:+.2f} oos={flip_oos_min:+.2f}) gap(val={gap_val:+.2f} oos={gap_oos:+.2f}) [{tag}]")

    genuine = [r for r in flip_results if r["genuine"]]
    genuine.sort(key=lambda r: min(r["real"]["val"]["opt_bp"], r["real"]["val"]["pess_bp"],
                                    r["real"]["oos"]["opt_bp"], r["real"]["oos"]["pess_bp"]), reverse=True)
    print(f"\n=== {len(genuine)}/{len(flip_results)} GENUINE (survive direction-flip control) ===")
    for r in genuine[:20]:
        print(f"  SL={r['sl']:.2f} ARM={r['arm']:.2f} Trail={r['trail']:.2f} "
              f"real(val={r['real']['val']['opt_bp']:+.2f}/{r['real']['val']['win_rate']:.1%} "
              f"oos={r['real']['oos']['opt_bp']:+.2f}/{r['real']['oos']['win_rate']:.1%}) "
              f"gap(val={r['gap_val_bp']:+.2f} oos={r['gap_oos_bp']:+.2f})")

    # buffer exhaustion for the top genuine config
    if genuine:
        top = genuine[0]
        sl, arm, trail = top["sl"], top["arm"], top["trail"]

        def never_stopped(row: pd.Series) -> bool:
            atr, entry, side = row["atr"], row["entry_price"], row["side"]
            sign = 1.0 if side == "long" else -1.0
            stop = entry - sign * sl * atr
            armed_, best_ = False, entry
            for h, l in zip(row["fwd_high"], row["fwd_low"]):
                adv = l if side == "long" else h
                fav = h if side == "long" else l
                if (adv <= stop) if side == "long" else (adv >= stop):
                    return False
                if sign * (fav - best_) > 0:
                    best_ = fav
                if not armed_ and sign * (best_ - entry) >= arm * atr:
                    armed_ = True
                if armed_:
                    ns = best_ - sign * trail * atr
                    if sign * (ns - stop) > 0:
                        stop = ns
            return True
        buf_rate = float(df.apply(never_stopped, axis=1).mean())
        print(f"\nTop genuine config buffer exhaustion rate: {buf_rate:.1%}")
    else:
        buf_rate = None

    report = {
        "n_candidates": {"total": len(df), "val": n_val, "oos": n_oos},
        "label_1_rate_within_called": float(df["label"].mean()),
        "grid_top15": [{"sl": s, "arm": a, "trail": t, "opt_bp": o, "pess_bp": p} for s, a, t, o, p in best[:15]],
        "n_val_oos_positive": len(val_oos_positive),
        "n_grid_cells": len(best),
        "flip_control_results": flip_results,
        "n_genuine": len(genuine),
        "genuine_sorted": genuine,
        "top_genuine_buffer_exhaustion_rate": buf_rate,
        "note": "HOLDOUT untouched -- VAL+OOS only.",
    }
    (OUT_DIR / "costgate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\nWrote {OUT_DIR / 'costgate_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
