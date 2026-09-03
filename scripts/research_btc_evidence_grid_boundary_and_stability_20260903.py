#!/usr/bin/env python3
"""BTC 증거신호 **격자 경계 확장 + rng 선택 안정성** -- XRP 과정을 BTC에 적용.

## 왜

2026-09-03 XRP에 적용한 튜닝 절차(포팅 프로토콜 §5-A/§5-E)를 BTC에도 건다.
BTC는 **경계 감사까지만 하고 확장은 안 했다**:

| 신호 | H격자 | 선택 H | K격자 | 선택 K | 경계 |
|---|---|---|---|---|---|
| `taker_delta_climax` | [**6**,9,12,18,24,30] | **6** | [1.5..3.2] | 2.0 | ⚠️H 하단 |
| `orthogonal_combo` | [**8**,12,18,24,30,36] | **8** | [**2.0**,2.5,...] | **2.0** | ⚠️⚠️H·K 둘 다 |
| `fib_extension_exhaustion` | [**10**,16,20,24,30] | **10** | [1.5..3.25] | 2.75 | ⚠️H 하단 |
| demarker/kalman/str_z/liquidity_sweep | — | — | — | — | ✅내부 |

## 설계 -- XRP와 동일 (재구현 금지)

각 그리드스크린 스크립트의 **`main()`을 그대로 재실행**하고 격자 상수·출력 경로만 patch한다.

**⭐순서가 중요하다** -- XRP에서 배운 것:
  1) **먼저 rng 안정성**을 잰다(원본 격자 고정, `RNG_SEED`만 8회 변경).
     XRP에서 taker 승자가 5종, orthogonal이 6종으로 흩어졌다.
     ⇒ 승자가 흔들리면 "확장했더니 바뀌었다"를 **경계 문제로 오독**하게 된다.
  2) 그 다음에 경계 확장. 상위집합 격자에서 목적함수가 **내려가면** 그건 노이즈다.

⚠️lift는 무작위 베이스라인 대비로 계산되고 그 표집이 rng를 쓴다. 격자에 셀을 추가하는
것만으로 기존 셀의 lift 측정치가 달라진다(XRP taker 실측: 확장 후 argmax의 lift가
1.4532→1.4051로 **내려갔다** -- 상위집합에서 불가능한 일).

⚠️OOS/HOLDOUT 미터치(원본 선택 규칙은 TRAIN lift argmax, VAL로 확인).
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

OUTDIR = ROOT / "data/research/btc_evidence_grid_boundary_20260903"
OUT = ROOT / "data/research/btc_evidence_grid_boundary_and_stability_20260903.json"

SEEDS = [20260901, 11, 271828, 141592, 577215, 31337, 90210, 8675309]

# 경계에 걸린 3종 (demarker/kalman/str_z/liquidity_sweep은 내부값이라 제외)
TARGETS = {
    "taker_delta_climax": {
        "script": "research_btc_taker_delta_climax_gridscreen_hittype_20260901.py",
        "deployed": ("close_at_h", 6, 2.0),
        "ext": {"HORIZON_GRID": [2, 3, 4, 6, 9, 12, 18, 24, 30]},
        "reason": "H=6이 격자 하단 경계"},
    "orthogonal_combo": {
        "script": "research_btc_orthogonal_combo_gridscreen_hittype_20260901.py",
        "deployed": ("touch", 8, 2.0),
        "ext": {"HORIZON_GRID": [3, 4, 6, 8, 12, 18, 24, 30, 36],
                "K_GRID": [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.57, 4.0, 4.5]},
        "reason": "H=8·K=2.0 둘 다 격자 하단 경계"},
    "fib_extension_exhaustion": {
        "script": "research_btc_fib_extension_exhaustion_gridscreen_hittype_20260901.py",
        "deployed": ("close_at_h", 10, 2.75),
        "ext": {"HORIZON_GRID": [4, 6, 8, 10, 16, 20, 24, 30]},
        "reason": "H=10이 격자 하단 경계"},
}


def log(m): print(f"[btc-grid] {m}", flush=True)


def load(rel):
    sp = importlib.util.spec_from_file_location(f"m_{Path(rel).stem}", ROOT / "scripts" / rel)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def chosen_of(rep):
    g = rep.get("global_selection") or rep.get("chosen") or {}
    t = g.get("train", g)
    return {"hit_type": g.get("hit_type"), "horizon": g.get("horizon"), "k": g.get("k"),
            "lift_bottom": t.get("lift_bottom", t.get("train_lift_bottom")),
            "lift_top": t.get("lift_top", t.get("train_lift_top")),
            "hits_bottom": t.get("n_hits_bottom", t.get("n_train_bottom_hits")),
            "hits_top": t.get("n_hits_top", t.get("n_train_top_hits"))}


def run_once(mod, out_path, seed=None, grids=None):
    saves = {}
    if seed is not None:
        saves["RNG_SEED"] = mod.RNG_SEED; mod.RNG_SEED = seed
    for k, v in (grids or {}).items():
        saves[k] = list(getattr(mod, k)); setattr(mod, k, list(v))
    old = mod.OUT_JSON; mod.OUT_JSON = out_path
    try:
        rc = mod.main()
    finally:
        for k, v in saves.items():
            setattr(mod, k, v)
        mod.OUT_JSON = old
    return chosen_of(json.loads(out_path.read_text())) if rc == 0 else None


def main() -> int:
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rep = {"seeds": SEEDS, "holdout_touched": False,
           "order": "① rng 안정성 → ② 경계 확장 (XRP에서 배운 순서)", "signals": {}}

    for name, spec in TARGETS.items():
        log(""); log("#" * 76); log(f"{name}  --  {spec['reason']}"); log("#" * 76)
        mod = load(spec["script"])
        res = {"deployed": list(spec["deployed"]), "reason": spec["reason"]}

        # ---------- ① rng 안정성 (원본 격자) ----------
        log("  ① rng 재시드 안정성 (원본 격자 고정)")
        rows = []
        for s in SEEDS:
            c = run_once(mod, OUTDIR / f"{name}_seed{s}.json", seed=s)
            if c is None:
                log(f"    seed={s}: 실패"); continue
            rows.append({"seed": s, **c})
            log(f"    seed={s:<9} {c['hit_type']}/{c['horizon']}/{c['k']:<5} "
                f"lift {c['lift_bottom']}/{c['lift_top']}")
        cells = Counter((r["hit_type"], r["horizon"], r["k"]) for r in rows)
        dep = spec["deployed"]
        dw = cells.get(dep, 0)
        log(f"    승자 분포({len(rows)}회): " +
            "  ".join(f"{c[0]}/{c[1]}/{c[2]}×{n}" for c, n in cells.most_common()))
        log(f"    ⇒ 배포셀 승률 **{dw}/{len(rows)}**  "
            f"{'✅안정' if len(cells) == 1 else f'⚠️{len(cells)}종으로 흔들림'}")
        res["stability"] = {"per_seed": rows, "distinct": len(cells),
                            "winner_counts": [{"cell": list(c), "n": n} for c, n in cells.most_common()],
                            "deployed_win_rate": dw / max(1, len(rows))}

        # ---------- ② 경계 확장 ----------
        log("  ② 격자 경계 확장")
        for k, v in spec["ext"].items():
            log(f"    {k}: {list(getattr(mod, k))}")
            log(f"        -> {v}")
        c = run_once(mod, OUTDIR / f"{name}_EXT.json", grids=spec["ext"])
        if c is None:
            log("    ⚠️확장 실행 실패"); res["extension"] = {"error": "main failed"}
        else:
            base = next((r for r in rows if r["seed"] == SEEDS[0]), None)
            log(f"    확장 선택: {c['hit_type']}/{c['horizon']}/{c['k']}  "
                f"lift {c['lift_bottom']}/{c['lift_top']}  hits {c['hits_bottom']}/{c['hits_top']}")
            if base:
                log(f"    원본(seed {SEEDS[0]}): {base['hit_type']}/{base['horizon']}/{base['k']}  "
                    f"lift {base['lift_bottom']}/{base['lift_top']}")
                # ⭐상위집합에서 목적함수가 내려가면 노이즈다
                try:
                    ob = min(float(base["lift_bottom"]), float(base["lift_top"]))
                    nb = min(float(c["lift_bottom"]), float(c["lift_top"]))
                    if nb < ob - 1e-9:
                        log(f"    ⚠️**상위집합인데 목적함수가 내려갔다**({ob:.4f} → {nb:.4f}) "
                            f"-- 경계 문제가 아니라 rng 측정 노이즈다")
                except (TypeError, ValueError):
                    pass
            edges = []
            for gk, gv in spec["ext"].items():
                val = c["horizon"] if "HORIZON" in gk else c["k"]
                if val == gv[0]: edges.append(f"{gk} 하단")
                if val == gv[-1]: edges.append(f"{gk} 상단")
            log(f"    경계 재점검: {'⚠️' + ', '.join(edges) if edges else '✅내부값'}")
            res["extension"] = {"chosen": c, "grid": spec["ext"], "still_at_edge": edges,
                                "changed": bool(base and (c["horizon"] != base["horizon"]
                                                          or abs(float(c["k"]) - float(base["k"])) > 1e-9
                                                          or c["hit_type"] != base["hit_type"]))}
        rep["signals"][name] = res

    log(""); log("=" * 80)
    log("종합 -- BTC 격자 경계 확장 / rng 안정성")
    log("=" * 80)
    log(f"{'신호':<26}{'배포셀 승률':>12}{'승자종류':>9}  확장 결과")
    for name, v in rep["signals"].items():
        st = v["stability"]; ex = v.get("extension") or {}
        c = ex.get("chosen")
        s = (f"{c['hit_type']}/{c['horizon']}/{c['k']}"
             f"{'  ⚠️변경' if ex.get('changed') else '  불변'}"
             f"{('  (여전히 ' + ', '.join(ex['still_at_edge']) + ')') if ex.get('still_at_edge') else ''}") \
            if c else "실패"
        log(f"{name:<26}{st['deployed_win_rate']*100:>11.0f}%{st['distinct']:>9}  {s}")
    log("")
    log("⚠️확장으로 선택이 바뀌어도 그 자체는 교체 근거가 아니다 -- "
        "격자 lift != 모델 품질(§5-A). TabPFN 대조 + 부트스트랩 CI가 필요하다.")
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
