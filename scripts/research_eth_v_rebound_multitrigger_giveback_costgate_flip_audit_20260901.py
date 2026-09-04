#!/usr/bin/env python3
"""giveback 9트리거(기존 배포모델)의 기존 경제성게이트 결과에 방향-뒤집기 대조군을 소급 적용.

이 세션에서 ZDC(지그재그) 경제성게이트 그리드서치 중 ARM이 극소(0.1x ATR)인 조합이 방향예측
실력과 무관하게 노이즈만으로 발동하는 아티팩트임을 발견했다(`feedback_trailing_stop_low_arm_
noise_harvest_artifact_20260901`). giveback 자신의 과거 세션(2026-08-31, `research_eth_v_
rebound_multitrigger_holdout_20260831.py`)이 이 대조군을 실제로 돌렸는지 이 세션에서 확인하지
못한 채 남아있던 캐비어트를 지금 채운다.

`data/research/eth_v_rebound_multitrigger_holdout_20260831/candidates_{val,oos}.pkl`(과거 세션이
이미 저장해둔 산출물, 재계산 없음)을 그대로 읽어 (1) 205셀 그리드+VAL/OOS동시양수 조합 재확인,
(2) 그 조합들에 방향뒤집기 대조군 적용, (3) 특히 giveback이 실제 채택한 SL=4.0/ARM=1.5/Trail=0.1을
명시적으로 확인한다. candidates_holdout.pkl은 이미 소진된(보고완료) 산출물을 감사(audit)하는
용도로만 읽는다 -- 새 파라미터 선정에 쓰지 않음(HOLDOUT 1회성 원칙 위반 아님, 사후 방법론 검증).

simulate_trailing()은 이 계열 스크립트들과 동일(재구현 아님).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_DIR = ROOT / "data/research/eth_v_rebound_multitrigger_holdout_20260831"
OUT_PATH = HOLDOUT_DIR / "flip_audit_report.json"
STANDARD_COST_BP = 10.0
GIVEBACK_SELECTED = (4.0, 1.5, 0.1)  # holdout_report.json's selected_config


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


def split_metrics(df: pd.DataFrame, sl: float, arm: float, trail: float, splits=("val", "oos")) -> dict:
    out = {}
    for split_name in splits:
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
    val = pd.read_pickle(HOLDOUT_DIR / "candidates_val.pkl")
    oos = pd.read_pickle(HOLDOUT_DIR / "candidates_oos.pkl")
    df = pd.concat([val, oos], ignore_index=True)
    flipped = df.copy()
    flipped["side"] = flipped["side"].map({"long": "short", "short": "long"})
    print(f"giveback candidates: val={len(val)} oos={len(oos)}")

    print(f"\n=== giveback's OWN selected config SL={GIVEBACK_SELECTED[0]} ARM={GIVEBACK_SELECTED[1]} "
          f"Trail={GIVEBACK_SELECTED[2]} -- direction-flip check ===")
    real_sel = split_metrics(df, *GIVEBACK_SELECTED)
    flip_sel = split_metrics(flipped, *GIVEBACK_SELECTED)
    for name in ("val", "oos"):
        r, f = real_sel[name], flip_sel[name]
        print(f"  {name}: real opt={r['opt_bp']:+.2f}bp pess={r['pess_bp']:+.2f}bp win={r['win_rate']:.1%} | "
              f"flip opt={f['opt_bp']:+.2f}bp pess={f['pess_bp']:+.2f}bp win={f['win_rate']:.1%}")
    sel_genuine = (min(real_sel["val"]["opt_bp"], real_sel["val"]["pess_bp"]) >
                   min(flip_sel["val"]["opt_bp"], flip_sel["val"]["pess_bp"]) and
                   min(real_sel["oos"]["opt_bp"], real_sel["oos"]["pess_bp"]) >
                   min(flip_sel["oos"]["opt_bp"], flip_sel["oos"]["pess_bp"]))
    print(f"  => giveback's deployed config is {'GENUINE' if sel_genuine else 'ARTIFACT-SUSPECT'}")

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
        row = {"sl": sl, "arm": arm, "trail": trail, "real": real, "flipped": flip,
               "gap_val_bp": real_val_min - flip_val_min, "gap_oos_bp": real_oos_min - flip_oos_min,
               "genuine": genuine, "is_giveback_selected": (sl, arm, trail) == GIVEBACK_SELECTED}
        flip_results.append(row)

    genuine = [r for r in flip_results if r["genuine"]]
    print(f"\n=== {len(genuine)}/{len(flip_results)} GENUINE (survive direction-flip control) ===")
    for r in sorted(flip_results, key=lambda r: -min(r["real"]["val"]["opt_bp"], r["real"]["val"]["pess_bp"],
                                                       r["real"]["oos"]["opt_bp"], r["real"]["oos"]["pess_bp"])):
        tag = "GENUINE" if r["genuine"] else "artifact"
        star = " <== giveback's deployed config" if r["is_giveback_selected"] else ""
        print(f"  SL={r['sl']:.2f} ARM={r['arm']:.2f} Trail={r['trail']:.2f} gap(val={r['gap_val_bp']:+.2f} "
              f"oos={r['gap_oos_bp']:+.2f}) [{tag}]{star}")

    # audit the already-spent HOLDOUT number for giveback's selected config (diagnostic only, not re-selection)
    holdout = pd.read_pickle(HOLDOUT_DIR / "candidates_holdout.pkl")
    holdout["split"] = "holdout"
    holdout_flipped = holdout.copy()
    holdout_flipped["side"] = holdout_flipped["side"].map({"long": "short", "short": "long"})
    real_ho = split_metrics(holdout, *GIVEBACK_SELECTED, splits=("holdout",))
    flip_ho = split_metrics(holdout_flipped, *GIVEBACK_SELECTED, splits=("holdout",))
    print(f"\n=== [진단전용, 이미소진된 HOLDOUT 감사] SL=4.0/ARM=1.5/Trail=0.1 ===")
    print(f"  real: opt={real_ho['holdout']['opt_bp']:+.2f}bp pess={real_ho['holdout']['pess_bp']:+.2f}bp win={real_ho['holdout']['win_rate']:.1%}")
    print(f"  flip: opt={flip_ho['holdout']['opt_bp']:+.2f}bp pess={flip_ho['holdout']['pess_bp']:+.2f}bp win={flip_ho['holdout']['win_rate']:.1%}")

    report = {
        "giveback_selected_config": {"sl": GIVEBACK_SELECTED[0], "arm": GIVEBACK_SELECTED[1], "trail": GIVEBACK_SELECTED[2]},
        "giveback_selected_real": real_sel, "giveback_selected_flipped": flip_sel, "giveback_selected_genuine": sel_genuine,
        "grid_top15": [{"sl": s, "arm": a, "trail": t, "opt_bp": o, "pess_bp": p} for s, a, t, o, p in best[:15]],
        "n_val_oos_positive": len(val_oos_positive), "n_grid_cells": len(best),
        "flip_control_results": flip_results, "n_genuine": len(genuine),
        "holdout_audit_diagnostic_only": {"real": real_ho, "flipped": flip_ho},
        "note": "HOLDOUT read here is diagnostic audit of an ALREADY-reported/deployed number, not new parameter selection.",
    }
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
