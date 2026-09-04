#!/usr/bin/env python3
"""48개 VAL+OOS 동시양수 조합 전체에 방향-뒤집기 대조군을 체계적으로 적용해 진짜 방향예측
edge와 트레일링 메커니즘 아티팩트(노이즈수확)를 분리한다 -- direction_flip_control_20260901.py의
단일사례 발견(ARM=0.1은 아티팩트, ARM=1.5는 진짜)을 전체 48개로 확장.

판정: REAL이 FLIPPED보다 val/oos 양쪽 다, opt/pess 양쪽 다 명확히 우월해야("진짜") 함 -- 그렇지
않으면(FLIPPED도 REAL과 비슷하게 양수면) 노이즈수확 아티팩트로 보고 제외.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data/research/eth_v_rebound_multitrigger_zigzag_direction_costgate_20260901/zdc_costgate_candidates.pkl"
REPORT_IN = ROOT / "data/research/eth_v_rebound_multitrigger_zigzag_direction_costgate_20260901/costgate_report.json"
OUT_PATH = ROOT / "data/research/eth_v_rebound_multitrigger_zigzag_direction_costgate_20260901/flip_filter_report.json"
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
            "opt_bp": float(opt.mean() * 1e4 - STANDARD_COST_BP),
            "pess_bp": float(pess.mean() * 1e4 - STANDARD_COST_BP),
            "win_rate": float((opt > 0).mean()),
        }
    return out


def main() -> int:
    df = pd.read_pickle(CANDIDATES)
    flipped = df.copy()
    flipped["side"] = flipped["side"].map({"long": "short", "short": "long"})

    configs = json.loads(REPORT_IN.read_text())["val_oos_positive_configs"]
    print(f"testing {len(configs)} VAL+OOS-positive configs against direction-flip control")

    results = []
    for sl, arm, trail in configs:
        real = split_metrics(df, sl, arm, trail)
        flip = split_metrics(flipped, sl, arm, trail)
        # "genuine": REAL beats FLIPPED on min(opt,pess) bp, in BOTH val and oos
        real_val_min = min(real["val"]["opt_bp"], real["val"]["pess_bp"])
        real_oos_min = min(real["oos"]["opt_bp"], real["oos"]["pess_bp"])
        flip_val_min = min(flip["val"]["opt_bp"], flip["val"]["pess_bp"])
        flip_oos_min = min(flip["oos"]["opt_bp"], flip["oos"]["pess_bp"])
        genuine = (real_val_min > flip_val_min) and (real_oos_min > flip_oos_min) and (real_val_min > 0) and (real_oos_min > 0)
        gap_val = real_val_min - flip_val_min
        gap_oos = real_oos_min - flip_oos_min
        row = {"sl": sl, "arm": arm, "trail": trail, "real": real, "flipped": flip,
               "gap_val_bp": gap_val, "gap_oos_bp": gap_oos, "genuine": genuine}
        results.append(row)
        tag = "GENUINE" if genuine else "artifact"
        print(f"  SL={sl:.2f} ARM={arm:.2f} Trail={trail:.2f} | real(val={real_val_min:+.2f} oos={real_oos_min:+.2f}) "
              f"flip(val={flip_val_min:+.2f} oos={flip_oos_min:+.2f}) gap(val={gap_val:+.2f} oos={gap_oos:+.2f}) [{tag}]")

    genuine_results = [r for r in results if r["genuine"]]
    genuine_results.sort(key=lambda r: min(r["real"]["val"]["opt_bp"], r["real"]["val"]["pess_bp"],
                                            r["real"]["oos"]["opt_bp"], r["real"]["oos"]["pess_bp"]), reverse=True)

    print(f"\n=== {len(genuine_results)}/{len(results)} configs are GENUINE (real beats flip-control both splits) ===")
    for r in genuine_results:
        print(f"  SL={r['sl']:.2f} ARM={r['arm']:.2f} Trail={r['trail']:.2f} "
              f"gap_val={r['gap_val_bp']:+.2f}bp gap_oos={r['gap_oos_bp']:+.2f}bp")

    OUT_PATH.write_text(json.dumps({"all_results": results, "n_genuine": len(genuine_results),
                                     "n_tested": len(results), "genuine_sorted": genuine_results},
                                    ensure_ascii=False, indent=2, default=str))
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
