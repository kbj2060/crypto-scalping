#!/usr/bin/env python3
"""v7b(sweep-only V자반등, giveback 9트리거의 전신) 경제성게이트에 방향-뒤집기 대조군 소급 적용.

giveback 9트리거 감사(research_eth_v_rebound_multitrigger_giveback_costgate_flip_audit_20260901.py)
와 동일 구조 -- 기존 산출물 v7b_costgate_candidates.pkl(재계산 없음)을 그대로 읽는다. v7b는
giveback처럼 하드코딩된 "선택 config" 기록이 없어(원 스크립트가 combined-sort top1을 그때그때
출력만 함), 이번에 205셀 그리드를 다시 돌려 그 top1이 무엇이었는지부터 재현한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data/research/eth_sweep_v_rebound_v7b_costgate_20260830/v7b_costgate_candidates.pkl"
OUT_PATH = ROOT / "data/research/eth_sweep_v_rebound_v7b_costgate_20260830/flip_audit_report.json"
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
    print(f"v7b candidates: val={n_val} oos={n_oos}")
    print(f"label==1 rate within called: {df['label'].mean():.4f}")

    print("\n=== Full 205-cell grid (VAL+OOS combined, real direction) ===")
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

    top1 = (best[0][0], best[0][1], best[0][2])
    print(f"\n=== Original script's own combined-sort top1: SL={top1[0]} ARM={top1[1]} Trail={top1[2]} -- flip check ===")
    real_top1 = split_metrics(df, *top1)
    flip_top1 = split_metrics(flipped, *top1)
    for name in ("val", "oos"):
        r, f = real_top1[name], flip_top1[name]
        print(f"  {name}: real opt={r['opt_bp']:+.2f}bp win={r['win_rate']:.1%} | flip opt={f['opt_bp']:+.2f}bp win={f['win_rate']:.1%}")
    top1_genuine = (min(real_top1["val"]["opt_bp"], real_top1["val"]["pess_bp"]) >
                    min(flip_top1["val"]["opt_bp"], flip_top1["val"]["pess_bp"]) and
                    min(real_top1["oos"]["opt_bp"], real_top1["oos"]["pess_bp"]) >
                    min(flip_top1["oos"]["opt_bp"], flip_top1["oos"]["pess_bp"]))
    print(f"  => {'GENUINE' if top1_genuine else 'ARTIFACT-SUSPECT'}")

    val_oos_positive = []
    for sl, arm, trail, _, _ in best:
        m = split_metrics(df, sl, arm, trail)
        if (min(m["val"]["opt_bp"], m["val"]["pess_bp"]) > 0
                and min(m["oos"]["opt_bp"], m["oos"]["pess_bp"]) > 0):
            val_oos_positive.append((sl, arm, trail))
    print(f"\n=== VAL AND OOS both positive: {len(val_oos_positive)}/{len(best)} ===")

    flip_results = []
    for sl, arm, trail in val_oos_positive:
        real = split_metrics(df, sl, arm, trail)
        flip = split_metrics(flipped, sl, arm, trail)
        real_val_min = min(real["val"]["opt_bp"], real["val"]["pess_bp"])
        real_oos_min = min(real["oos"]["opt_bp"], real["oos"]["pess_bp"])
        flip_val_min = min(flip["val"]["opt_bp"], flip["val"]["pess_bp"])
        flip_oos_min = min(flip["oos"]["opt_bp"], flip["oos"]["pess_bp"])
        genuine = real_val_min > flip_val_min and real_oos_min > flip_oos_min and real_val_min > 0 and real_oos_min > 0
        flip_results.append({"sl": sl, "arm": arm, "trail": trail, "real": real, "flipped": flip,
                              "gap_val_bp": real_val_min - flip_val_min, "gap_oos_bp": real_oos_min - flip_oos_min,
                              "genuine": genuine, "is_top1": (sl, arm, trail) == top1})

    genuine = [r for r in flip_results if r["genuine"]]
    print(f"\n=== {len(genuine)}/{len(flip_results)} GENUINE (survive direction-flip control) ===")
    for r in sorted(flip_results, key=lambda r: -min(r["real"]["val"]["opt_bp"], r["real"]["val"]["pess_bp"],
                                                       r["real"]["oos"]["opt_bp"], r["real"]["oos"]["pess_bp"]))[:20]:
        tag = "GENUINE" if r["genuine"] else "artifact"
        star = " <== combined-sort top1" if r["is_top1"] else ""
        print(f"  SL={r['sl']:.2f} ARM={r['arm']:.2f} Trail={r['trail']:.2f} gap(val={r['gap_val_bp']:+.2f} "
              f"oos={r['gap_oos_bp']:+.2f}) [{tag}]{star}")

    report = {
        "top1_combined_sort": {"sl": top1[0], "arm": top1[1], "trail": top1[2]},
        "top1_real": real_top1, "top1_flipped": flip_top1, "top1_genuine": top1_genuine,
        "n_val_oos_positive": len(val_oos_positive), "n_grid_cells": len(best),
        "flip_control_results": flip_results, "n_genuine": len(genuine),
    }
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
