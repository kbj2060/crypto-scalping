#!/usr/bin/env python3
"""XRP 증거신호 격자 **선택 안정성** -- rng 재시드에서 승자가 바뀌는가.

## 왜 이걸 하나 -- 격자 확장이 "경계 문제가 아니다"라고 답했다

2026-09-03 경계 확장 결과가 이상했다:

    taker  원본(K격자 [1.5..3.2])      -> giveback/9/**1.5**  lift 1.4532 / 1.4747
           확장(K격자 [0.8..3.2])      -> giveback/9/**2.0**  lift 1.4051 / 1.3307

⭐**상위집합 격자에서 argmax의 목적함수 값이 내려갈 수는 없다.** 그런데 내려갔다.
⇒ 셀의 lift가 **rng에 의존**한다는 뜻이다. 무작위 베이스라인(비발동 봉 표집)이 rng를 쓰고,
격자에 셀을 추가하면 **rng 소비 순서가 바뀌어** 같은 셀의 베이스라인이 달라진다.

⇒ 즉 이건 **경계 문제가 아니라 선택 불안정성**이다. 격자를 넓히기 전에 이것부터 재야 한다.

## 검정

  A. **rng 재시드 안정성** (B회) -- 격자·데이터 고정, `RNG_SEED`만 바꿔 재실행.
     승자 셀의 분포와, 원본 선택 셀이 몇 %에서 이기는지 본다.
     승자가 흩어지면 **"어느 셀이 최적인가"라는 질문 자체가 답이 없는 것**이고,
     현행 값을 바꿀 근거도 없어진다.

  B. **두께 제약 하 최선** (orthogonal) -- 확장 격자에서 `MIN_TRAIN_HITS`를 실제 배포 셀 수준
     (touch_mfe/8/2.0의 348/211)으로 올렸을 때 무엇이 뽑히는지.
     원래 XRP orthogonal은 자동선택(giveback/12/2.5, hits 79/34)을 **두께 감사로 기각**하고
     touch_mfe로 갈아탄 이력이 있다. 확장 격자의 승자(close_at_h/3/2.0, hits 73/44)도
     같은 이유로 기각되는지 확인한다.

⚠️OOS/HOLDOUT 미터치.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUTDIR = ROOT / "data/research/xrp_evidence_grid_stability_20260903"
OUT = ROOT / "data/research/xrp_evidence_grid_selection_stability_20260903.json"

B_SEEDS = 8
SEEDS = [20260901, 11, 271828, 141592, 577215, 31337, 90210, 8675309][:B_SEEDS]

TARGETS = {
    "taker_delta_climax": {
        "script": "research_xrp_taker_delta_climax_gridscreen_hittype_20260903.py",
        "deployed": ("touch_giveback_sustained", 9, 1.5),
        "grids": {},                       # 원본 격자 그대로 (경계확장 없이 안정성만)
    },
    "orthogonal_combo": {
        "script": "research_xrp_orthogonal_combo_gridscreen_hittype_20260903.py",
        "deployed": ("touch_mfe", 8, 2.0),
        "grids": {},
    },
}

# B) 두께 제약 -- 배포 셀(touch_mfe 348/211)에 견줄 수준으로 올린다
THICK_MIN_HITS = 150


def log(m): print(f"[stab] {m}", flush=True)


def load(rel):
    sp = importlib.util.spec_from_file_location(f"m_{Path(rel).stem}", ROOT / "scripts" / rel)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def chosen_of(rep):
    g = rep.get("global_selection") or rep.get("chosen") or {}
    t = g.get("train", g)
    return {"hit_type": g.get("hit_type"), "horizon": g.get("horizon"), "k": g.get("k"),
            "lift_bottom": t.get("lift_bottom", t.get("train_lift_bottom")),
            "lift_top": t.get("lift_top", t.get("train_lift_top")),
            "hits_bottom": t.get("n_hits_bottom", t.get("n_train_bottom_hits")),
            "hits_top": t.get("n_hits_top", t.get("n_train_top_hits"))}


def run_once(mod, out_path, seed=None, min_hits=None, grids=None):
    saves = {}
    if seed is not None:
        saves["RNG_SEED"] = mod.RNG_SEED; mod.RNG_SEED = seed
    if min_hits is not None and hasattr(mod, "MIN_TRAIN_HITS"):
        saves["MIN_TRAIN_HITS"] = mod.MIN_TRAIN_HITS; mod.MIN_TRAIN_HITS = min_hits
    for k, v in (grids or {}).items():
        saves[k] = list(getattr(mod, k)); setattr(mod, k, list(v))
    old_out = mod.OUT_JSON; mod.OUT_JSON = out_path
    try:
        rc = mod.main()
    finally:
        for k, v in saves.items():
            setattr(mod, k, v)
        mod.OUT_JSON = old_out
    if rc != 0:
        return None
    return chosen_of(json.loads(out_path.read_text()))


def main() -> int:
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rep = {"seeds": SEEDS, "thick_min_hits": THICK_MIN_HITS,
           "holdout_touched": False, "signals": {}}

    # ---------------- A) rng 재시드 안정성 ----------------
    for name, spec in TARGETS.items():
        log("")
        log("#" * 68)
        log(f"A) {name} -- rng 재시드 안정성 (B={len(SEEDS)})")
        log("#" * 68)
        mod = load(spec["script"])
        rows = []
        for s in SEEDS:
            c = run_once(mod, OUTDIR / f"{name}_seed{s}.json", seed=s, grids=spec["grids"])
            if c is None:
                log(f"  seed={s}: 실패"); continue
            rows.append({"seed": s, **c})
            log(f"  seed={s:<9} -> {c['hit_type']}/{c['horizon']}/{c['k']}  "
                f"lift {c['lift_bottom']}/{c['lift_top']}")
        cells = Counter((r["hit_type"], r["horizon"], r["k"]) for r in rows)
        dep = spec["deployed"]
        dep_wins = cells.get(dep, 0)
        log("")
        log(f"  승자 분포 ({len(rows)}회):")
        for c, n in cells.most_common():
            mark = "  ⭐배포셀" if c == dep else ""
            log(f"    {c[0]}/{c[1]}/{c[2]:<5} {n}회 ({n/len(rows)*100:.0f}%){mark}")
        log(f"  ⇒ 배포셀 {dep[0]}/{dep[1]}/{dep[2]}이 이긴 비율: "
            f"**{dep_wins}/{len(rows)} ({dep_wins/max(1,len(rows))*100:.0f}%)**")
        stable = len(cells) == 1
        log(f"  ⇒ {'✅선택 안정 -- 단일 셀' if stable else f'⚠️**선택 불안정 -- {len(cells)}개 셀이 번갈아 이긴다**'}")
        rep["signals"][name] = {
            "deployed": list(dep), "per_seed": rows,
            "distinct_winners": len(cells),
            "winner_counts": [{"cell": list(c), "n": n} for c, n in cells.most_common()],
            "deployed_win_rate": dep_wins / max(1, len(rows)), "stable": bool(stable)}

    # ---------------- B) 두께 제약 하 최선 (orthogonal) ----------------
    log("")
    log("#" * 68)
    log(f"B) orthogonal_combo -- 두께 제약 MIN_TRAIN_HITS={THICK_MIN_HITS} 하에서 무엇이 뽑히나")
    log("#" * 68)
    mod = load(TARGETS["orthogonal_combo"]["script"])
    c = run_once(mod, OUTDIR / "orthogonal_thick.json", min_hits=THICK_MIN_HITS,
                 grids={"HORIZON_GRID": [3, 4, 6, 8, 12, 18, 24, 30, 36],
                        "K_GRID": [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.57, 4.0, 4.5]})
    if c:
        dep = TARGETS["orthogonal_combo"]["deployed"]
        log(f"  두께제약 선택: {c['hit_type']}/{c['horizon']}/{c['k']}  "
            f"hits {c['hits_bottom']}/{c['hits_top']}  lift {c['lift_bottom']}/{c['lift_top']}")
        log(f"  현재 배포:     {dep[0]}/{dep[1]}/{dep[2]}  (hits 348/211 -- 두께로 선택된 셀)")
        same = (c["hit_type"], c["horizon"], c["k"]) == dep
        log(f"  ⇒ {'✅배포셀과 동일 -- 두께 제약을 걸면 확장 격자도 같은 답' if same else '⚠️다른 셀'}")
        rep["orthogonal_thick_constrained"] = {"chosen": c, "deployed": list(dep),
                                               "same_as_deployed": bool(same),
                                               "min_train_hits": THICK_MIN_HITS}
    else:
        log("  ⚠️실패 (두께 제약을 만족하는 셀이 없을 수 있다)")
        rep["orthogonal_thick_constrained"] = {"error": "no cell passed"}

    log("")
    log("=== 종합 ===")
    for n, v in rep["signals"].items():
        log(f"  {n:<24} 서로 다른 승자 {v['distinct_winners']}종  "
            f"배포셀 승률 {v['deployed_win_rate']*100:>3.0f}%  "
            f"{'✅안정' if v['stable'] else '⚠️불안정'}")
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
