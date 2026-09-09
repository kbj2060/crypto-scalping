"""TabICL direction head 5시드 집계 — 저장된 per-seed JSON 에서 게이트를 재판정한다.

본 실행(`research_omega461_tabicl_direction_head_20260909.py`)이 집계 단계에서 import 누락으로
죽었지만 모델 실행은 5시드 전부 끝났고 JSON 이 시드마다 증분 저장됐다. 재실행 없이 집계만 한다.
판정 규칙은 본 스크립트와 동일: 선택은 VAL 중앙값으로만(validation_only).
"""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/tabicl_direction/tabicl_direction_zig075_5seed.json"



def _binom_ge(k: int, n: int) -> float:
    """공정한 동전(p=0.5)에서 n 시드 중 k 개 이상이 양수일 확률.

    과반 규칙의 검정력을 그대로 드러내기 위해 찍는다 — 5시드에서 3/5 는 귀무에서도 50% 로
    나온다. **과반 통과는 증거가 아니다.** 정보는 Δ 중앙값과 그 일관성이 담는다.
    """
    from math import comb
    return sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)

def main() -> int:
    rep = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else P).read_text())
    seeds = rep["seeds"]
    tags = [("전량" if r == 0 else f"{r//1000}k") for r in rep["ladder"]]
    arms = sorted({a for s in rep["per_seed"].values() for a in s["arms"]})
    ctrl_best = max(v["bal_acc"] for k, v in rep["controls"].items() if k != "majority")

    inc = [rep["per_seed"][str(s)]["incumbent"]["bal_acc"] for s in seeds
           if rep["per_seed"][str(s)]["incumbent"]]
    print(f"[데이터] TRAIN {rep['n_train']:,} · VAL {rep['n_val']:,} · 질의 {rep['n_query']:,}"
          f" (시드 간 고정) · n_estimators={rep['n_estimators']} · 균형={rep['balance_context']}")
    print(f"[현직 TabM] 중앙 {statistics.median(inc):.4f}  [{min(inc):.4f}, {max(inc):.4f}]  n={len(inc)}")
    print(f"[대조군 최고] {ctrl_best:.4f}\n")

    cells = {}
    for arm in arms:
        for tag in tags:
            accs, deltas = [], []
            for s in seeds:
                one = rep["per_seed"][str(s)]
                d = one["arms"].get(arm, {}).get(tag)
                if not d or "bal_acc" not in d:
                    continue
                accs.append(d["bal_acc"])
                if one["incumbent"]:
                    deltas.append(d["bal_acc"] - one["incumbent"]["bal_acc"])
            if accs:
                cells[f"{arm}|{tag}"] = {
                    "n_seeds": len(accs), "median_bal_acc": statistics.median(accs),
                    "min": min(accs), "max": max(accs), "n_delta": len(deltas),
                    "median_delta": statistics.median(deltas) if deltas else None,
                    "n_better": sum(1 for x in deltas if x > 0),
                    "sign_agreement": bool(deltas) and (all(x > 0 for x in deltas) or all(x < 0 for x in deltas))}

    print(f"  {'셀':14s} {'중앙bal':>8s} {'[최소,최대]':>17s} {'Δ중앙':>9s} {'이김':>6s}  부호일치")
    for k, v in cells.items():
        dv = "n/a" if v["median_delta"] is None else f"{v['median_delta']:+.4f}"
        print(f"  {k:14s} {v['median_bal_acc']:8.4f} [{v['min']:.4f},{v['max']:.4f}] {dv:>9s} "
              f"{v['n_better']:>3d}/{v['n_delta']:<2d}  {'✅' if v['sign_agreement'] else '❌'}")

    best_k, best = max(cells.items(), key=lambda kv: kv[1]["median_bal_acc"])
    k1 = best["median_bal_acc"] > ctrl_best
    k2 = best["median_delta"] is not None and best["median_delta"] > 0
    n_d = best["n_delta"]
    k2b = n_d > 0 and best["n_better"] * 2 > n_d           # 과반 (사용자 2026-09-09)
    k2b_strict = best["n_better"] == n_d == len(seeds)     # 저장소 승격 게이트 원본
    seed_p = _binom_ge(best["n_better"], n_d) if n_d else None
    mono = {}
    for arm in arms:
        xs = [cells[f"{arm}|{t}"]["median_bal_acc"] for t in tags if f"{arm}|{t}" in cells]
        mono[arm] = round(xs[-1] - xs[0], 4) if len(xs) >= 2 else None
    k3 = any(v is not None and v > 0 for v in mono.values())
    allp = k1 and k2 and k2b and k3
    print(f"\n{'='*88}\n[킬 게이트]  최선 셀 {best_k}  중앙 {best['median_bal_acc']:.4f}")
    print(f"  K1  대조군 우위 (>{ctrl_best:.4f})              : {'✅' if k1 else '❌'}")
    print(f"  K2  현직 우위 (Δ중앙 {best['median_delta']:+.4f})       : {'✅' if k2 else '❌'}")
    print(f"  K2b 시드 과반 ({best['n_better']}/{n_d})                       : {'✅' if k2b else '❌'}"
          f"   [귀무 p={seed_p:.3f} · 엄격(전부일치) {'✅' if k2b_strict else '❌'}]")
    print(f"  K3  컨텍스트 단조 {mono}       : {'✅' if k3 else '❌'}")
    print(f"\n  → Stage 1(경제성) 진행 {'허가' if allp else '불가'}"
          f" — 통과해도 채택 근거는 아니다(정확도 개선이 경제 이득으로 이어지지 않은 사례 반복).")
    rep["cells"] = cells
    rep["verdict"] = {"best_cell": best_k, "median_bal_acc": best["median_bal_acc"],
                      "control_best_bal_acc": ctrl_best, "K1": bool(k1), "K2": bool(k2),
                      "K2b_seed_majority": bool(k2b), "K2b_seed_all_agree_strict": bool(k2b_strict),
                      "seed_rule": "majority (user 2026-09-09) — phase progression only, NOT a "
                                   "promotion basis; repo gate still requires all-agree",
                      "seed_rule_null_p": seed_p, "K3": bool(k3),
                      "context_monotonicity": mono, "stage1_allowed": bool(allp)}
    Path(sys.argv[1] if len(sys.argv) > 1 else P).write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
