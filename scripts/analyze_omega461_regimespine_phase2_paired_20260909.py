"""Phase 2 판정 — 레짐 척추 balnobb vs wide24 **짝지은 시드별** 비교.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계약: docs/model_contracts/omega461_regimegbm_rebuild_contract.md
스윕: `scripts/ops/run_omega461_regimespine_phase2_sweep_20260909.sh` (2컴포넌트 × 5시드 × 2arm)

왜 짝지은 비교인가
-----------------
두 arm 은 **같은 트레이너·같은 시드·같은 라벨·같은 HP**로 돌았고 차이는 레짐 척추 하나뿐이다.
따라서 시드별로 짝을 지어 차이를 보면 시드 분산이 상쇄된다. 이 저장소의 반복 교훈
(`tabm_hp_low_signal_pattern`: 단일 시드 비교는 노이즈, std 0.0009 > 전형적 HP 효과)에 따라
**시드 평균이 아니라 시드별 부호 일치**를 1차 판정으로 쓴다.

판정 기준 (계약 `selection.validation_pass_criteria` 그대로)
----------------------------------------------------------
  P1. VAL 에서 전신(wide24) 대비 **PnL 과 MDD 가 둘 다 비악화**
  P2. **N=5 시드 전부에서 OOS 부호가 일치** (Seed-Diversity Gate)
후보 선택은 **validation 만** 본다(`candidate_selection_scope: validation_only`).
OOS 는 보고하되 선택 근거로 쓰지 않는다.

⚠️ 표본 한계를 항상 함께 보고한다 — 이 라인의 거래 수는 창당 10~35건 수준이라, 어떤 차이도
소수 트레이드가 만들 수 있다. 판정문에 거래 수를 반드시 병기한다.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "tmp/causal_regen_20260516"
STEM = "omega4_3head_parent72_loose_entry_quality_20260620_regimespine_{arm}_{comp}_s{seed}_20260909"
SEEDS = [615372041, 208844917, 933105268, 471926350, 862017594]
COMPS = ["zig075", "h48qual"]
ARMS = ["balnobb", "wide24"]
OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909"


def read_metrics(arm: str, comp: str, seed: int) -> dict | None:
    p = BASE / STEM.format(arm=arm, comp=comp, seed=seed) / "report.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    rk = r.get("ranking_by_validation_pnl") or []
    if not rk:
        return None
    t = rk[0]                       # 이 스윕은 컴포넌트당 threshold 1개만 돌린다
    return {k: t.get(k) for k in ("variant", "quality_threshold",
                                  "validation_pnl", "validation_mdd", "validation_wr", "validation_trades",
                                  "oos_pnl", "oos_mdd", "oos_wr", "oos_trades")}


def main() -> int:
    rep = {"seeds": SEEDS, "missing": [], "components": {}}
    for comp in COMPS:
        rows = {}
        for arm in ARMS:
            for s in SEEDS:
                m = read_metrics(arm, comp, s)
                if m is None:
                    rep["missing"].append(f"{arm}/{comp}/{s}")
                rows[(arm, s)] = m
        print(f"\n{'='*104}\n[{comp}]", flush=True)
        print(f"{'seed':>11s} | {'VAL PnL  balnobb / wide24  (Δ)':>40s} | {'VAL MDD  balnobb / wide24  (Δ)':>40s}",
              flush=True)
        dpnl, dmdd, doos, pairs = [], [], [], []
        for s in SEEDS:
            a, b = rows[("balnobb", s)], rows[("wide24", s)]
            if not a or not b:
                print(f"{s:>11d} |  (결측 — 스킵)", flush=True)
                continue
            dp = a["validation_pnl"] - b["validation_pnl"]
            dm = a["validation_mdd"] - b["validation_mdd"]      # MDD 는 음수, +면 개선
            do = a["oos_pnl"] - b["oos_pnl"]
            dpnl.append(dp); dmdd.append(dm); doos.append(do)
            pairs.append({"seed": s, "balnobb": a, "wide24": b,
                          "d_val_pnl": dp, "d_val_mdd": dm, "d_oos_pnl": do})
            print(f"{s:>11d} | {a['validation_pnl']:+8.2f} / {b['validation_pnl']:+8.2f}  "
                  f"({dp:+7.2f}pp) | {a['validation_mdd']:+8.2f} / {b['validation_mdd']:+8.2f}  "
                  f"({dm:+7.2f}pp)", flush=True)
        if not pairs:
            print("  판정 불가 — 완료된 짝이 없다", flush=True)
            rep["components"][comp] = {"pairs": [], "verdict": "no_pairs"}
            continue

        n = len(pairs)
        pnl_better = sum(1 for x in dpnl if x > 0)
        mdd_better = sum(1 for x in dmdd if x > 0)
        both = sum(1 for p, m in zip(dpnl, dmdd) if p > 0 and m > 0)
        oos_pos = sum(1 for x in doos if x > 0)
        p1 = both == n                                  # 계약: VAL PnL·MDD 둘 다 비악화
        # OOS 부호 일치: balnobb 자체 OOS PnL 의 부호가 5시드 전부 같은가
        oos_signs = [1 if p["balnobb"]["oos_pnl"] > 0 else -1 for p in pairs]
        p2 = len(set(oos_signs)) == 1
        tr = [p["balnobb"]["validation_trades"] for p in pairs]
        tro = [p["balnobb"]["oos_trades"] for p in pairs]

        print(f"\n  짝 {n}개 | VAL PnL 개선 {pnl_better}/{n} · MDD 개선 {mdd_better}/{n} · "
              f"**둘 다 {both}/{n}**", flush=True)
        print(f"  Δ VAL PnL 중앙 {statistics.median(dpnl):+.2f}pp (범위 {min(dpnl):+.2f}~{max(dpnl):+.2f})", flush=True)
        print(f"  Δ VAL MDD 중앙 {statistics.median(dmdd):+.2f}pp (범위 {min(dmdd):+.2f}~{max(dmdd):+.2f})", flush=True)
        print(f"  [참고] Δ OOS PnL 개선 {oos_pos}/{n}, 중앙 {statistics.median(doos):+.2f}pp "
              f"— **선택 근거로 쓰지 않음**", flush=True)
        print(f"  거래 수(balnobb): VAL {tr} · OOS {tro}  ← 이 크기에서는 소수 트레이드가 차이를 만든다",
              flush=True)
        print(f"\n  P1 (VAL PnL·MDD 둘 다 비악화, {n}/{n}): {'✅ 통과' if p1 else '❌ 실패'}", flush=True)
        print(f"  P2 (OOS 부호 5시드 일치): {'✅ 통과' if p2 else '❌ 실패'} "
              f"(부호 {oos_signs})", flush=True)
        rep["components"][comp] = {
            "pairs": pairs, "n": n,
            "val_pnl_better": pnl_better, "val_mdd_better": mdd_better, "val_both_better": both,
            "median_d_val_pnl": statistics.median(dpnl), "median_d_val_mdd": statistics.median(dmdd),
            "median_d_oos_pnl": statistics.median(doos), "oos_signs": oos_signs,
            "P1_val_both_nonworse": p1, "P2_oos_sign_consistent": p2,
            "verdict": "pass" if (p1 and p2) else "fail",
            "val_trades": tr, "oos_trades": tro}

    (OUT / "phase2_paired_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    print(f"\n{'='*104}")
    for comp, v in rep["components"].items():
        print(f"[{comp}] 판정: {v.get('verdict')}", flush=True)
    if rep["missing"]:
        print(f"⚠️ 결측 {len(rep['missing'])}건: {rep['missing'][:8]}", flush=True)
    print(f"\n산출물: {OUT}/phase2_paired_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
