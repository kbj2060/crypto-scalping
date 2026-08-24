#!/usr/bin/env python3
"""[[eth_tabm_label_logic_5way_regime_expert_lrschedule_20260822]] N=6x3라벨 실행 결과의
report.json(`ranking_by_validation_pnl[0]`, VAL=2026Q2/OOS=2026-07-01~)을 모아 라벨별
VAL/OOS pnl 평균±표준편차, 부호일치 개수를 집계한다. dev에서 실행(torch불필요, report.json은
이미 서버 학습에서 산출완료돼 pull만 하면 됨). 부분완료(18개중 일부만) 상태에서도 그때까지
완료분만 집계하도록 설계 -- 매 재실행마다 그 시점 진행상황을 그대로 보여준다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "tmp/causal_regen_20260516"
MODEL_ID = "omega4_3head_parent72_loose_entry_quality_20260620"

LABELS = ["zigzag", "h48qual", "cusum"]
SEEDS = [133725056, 325805917, 775149439, 126593178, 286919795, 310216042]

# [[eth_tabm_label_logic_3label_split_convention_retest_20260822]]와 동일 split(TRAIN fit+VAL-Q2
# ending 2026-06-30, OOS=2026-07-01~)에서 확인된 always-long 벤치마크. 부호비교 기준선.
BENCHMARK_VAL_PNL = -23.34
BENCHMARK_OOS_PNL = 21.51


def report_path(label: str, seed: int) -> Path:
    return OUT_ROOT / f"{MODEL_ID}_label5way_{label}_154feat_regime_expert_lrschedule_seed{seed}_20260822" / "report.json"


def main() -> None:
    print(f"always-long 벤치마크: VAL(2026Q2)={BENCHMARK_VAL_PNL:+.2f}% OOS(2026-07~)={BENCHMARK_OOS_PNL:+.2f}%\n")

    for label in LABELS:
        rows = []
        for seed in SEEDS:
            p = report_path(label, seed)
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            top = d["ranking_by_validation_pnl"][0]
            rows.append({
                "seed": seed,
                "val_pnl": top["validation_pnl"],
                "oos_pnl": top["oos_pnl"],
                "variant": top["variant"],
            })

        n_done = len(rows)
        print(f"=== label={label}  ({n_done}/{len(SEEDS)} seed 완료) ===")
        if not rows:
            print("  (완료된 실행 없음)\n")
            continue

        val = np.array([r["val_pnl"] for r in rows])
        oos = np.array([r["oos_pnl"] for r in rows])
        sign_match = int(np.sum(np.sign(val) == np.sign(oos)))

        for r in rows:
            match = "일치" if np.sign(r["val_pnl"]) == np.sign(r["oos_pnl"]) else "불일치"
            print(f"  seed={r['seed']:>10}  variant={r['variant']:<6}  "
                  f"VAL={r['val_pnl']:+7.2f}%  OOS={r['oos_pnl']:+7.2f}%  ({match})")

        print(f"  --> VAL mean={val.mean():+.2f}% std={val.std(ddof=0):.2f}   "
              f"OOS mean={oos.mean():+.2f}% std={oos.std(ddof=0):.2f}   "
              f"부호일치={sign_match}/{n_done}")
        print()


if __name__ == "__main__":
    main()
