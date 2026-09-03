#!/usr/bin/env python3
"""XRP 증거신호 5종 **격자 경계 감사 + 확장 재스크리닝**.

## 왜

2026-09-03 레짐 감사에서 "포팅 미튜닝의 최다 원인은 격자 경계"임이 확인됐다
(`docs/homer/evidence_signal_new_coin_port_protocol.md` §5-A).
같은 잣대를 증거신호 5종에 대면 **2종이 경계에 걸려 있다**:

| 신호 | 격자 H | 선택 H | 격자 K | 선택 K | 경계 |
|---|---|---|---|---|---|
| demarker | [2..20] | 2 | [0.4..4.0] | 1.5 | H 하단 -> **검증완료(H=1 미승, 부록)** |
| kalman | [4..7]+[8..24] | 5 | [1.5..4.0] | 2.0 | ✅내부 |
| short_term_return_z | [2,3,6,9,12,18] | 12 | [1.0..2.5] | 1.5 | ✅내부 |
| **taker_delta_climax** | [6..30] | 9 | **[1.5,2.0,2.4,2.8,3.2]** | **1.5** | ⚠️**K 하단** |
| **orthogonal_combo** | **[8,12,18,24,30,36]** | **8** | **[2.0,2.5,3.0,3.57,4.0,4.5]** | **2.0** | ⚠️⚠️**H·K 둘 다 하단** |

## 설계 -- 재구현 금지

각 그리드스크린 스크립트의 **`main()`을 그대로 재실행**하고 격자 상수와 출력 경로만
monkey-patch한다. lift 정의·베이스라인 추출·적격성 게이트·선택 규칙이 조용히 달라지는 걸 막는다.

    taker      K_GRID      [1.5,...]              -> [0.8, 1.0, 1.25, 1.5, ...]
    orthogonal HORIZON_GRID [8,12,...]            -> [3, 4, 6, 8, 12, ...]
               K_GRID       [2.0,2.5,...]         -> [1.0, 1.25, 1.5, 2.0, 2.5, ...]

⚠️**격자가 고르는 것 != 모델이 나은 것**(demarker H=1 사례, §5-A). 여기서 선택이 바뀌어도
그 자체로는 교체 근거가 아니다. 바뀐 신호는 **TabPFN 대조 + 평가표본 부트스트랩 CI**로
따로 판정한다(별도 스크립트).

⚠️OOS/HOLDOUT 미터치 -- 원본 스크립트의 선택 규칙은 TRAIN lift argmax이고 VAL로 확인한다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = ROOT / "data/research/xrp_evidence_grid_boundary_extension_20260903.json"
OUTDIR = ROOT / "data/research/xrp_evidence_grid_ext_20260903"

# 현재 배포 중인 확정 셀 (docs/experiments/xrp_evidence_signal_and_regime_20260903.md)
CURRENT = {
    "taker_delta_climax": {"hit_type": "touch_giveback_sustained", "horizon": 9, "k": 1.5},
    "orthogonal_combo": {"hit_type": "touch_mfe", "horizon": 8, "k": 2.0},
}

# 경계 방향으로만 넓힌다 (반대쪽은 이미 탐색돼 선택되지 않았다)
EXTENSIONS = {
    "taker_delta_climax": {
        "script": "research_xrp_taker_delta_climax_gridscreen_hittype_20260903.py",
        "patch": {"K_GRID": [0.8, 1.0, 1.25, 1.5, 2.0, 2.4, 2.8, 3.2]},
        "reason": "K=1.5가 격자 하단 경계",
    },
    "orthogonal_combo": {
        "script": "research_xrp_orthogonal_combo_gridscreen_hittype_20260903.py",
        "patch": {"HORIZON_GRID": [3, 4, 6, 8, 12, 18, 24, 30, 36],
                  "K_GRID": [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.57, 4.0, 4.5]},
        "reason": "H=8·K=2.0 둘 다 격자 하단 경계",
    },
}


def log(m): print(f"[gridext] {m}", flush=True)


def load(rel):
    sp = importlib.util.spec_from_file_location(f"m_{Path(rel).stem}", ROOT / "scripts" / rel)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def at_edge(val, grid):
    return "하단" if val == grid[0] else ("상단" if val == grid[-1] else None)


def main() -> int:
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rep = {"current": CURRENT, "extensions": {}, "holdout_touched": False}

    for name, spec in EXTENSIONS.items():
        log("")
        log("#" * 68)
        log(f"{name}  --  {spec['reason']}")
        log("#" * 68)
        mod = load(spec["script"])
        orig = {k: list(getattr(mod, k)) for k in spec["patch"]}
        for k, v in spec["patch"].items():
            log(f"  {k}: {orig[k]}")
            log(f"      -> {v}")
            setattr(mod, k, list(v))
        old_out = mod.OUT_JSON
        mod.OUT_JSON = OUTDIR / f"{name}_gridscreen_report_EXT.json"
        try:
            rc = mod.main()
        finally:
            for k, v in orig.items():
                setattr(mod, k, v)
            mod.OUT_JSON = old_out
        if rc != 0:
            log(f"  ⚠️main() 반환 {rc}")
            rep["extensions"][name] = {"error": f"main returned {rc}"}
            continue

        r = json.loads((OUTDIR / f"{name}_gridscreen_report_EXT.json").read_text())
        ch = r.get("chosen") or r.get("winner") or {}
        # 리포트 구조가 스크립트마다 다르다 -- 가능한 키를 폭넓게 훑는다
        def pick(*keys):
            for src in (ch, r):
                for k in keys:
                    if isinstance(src, dict) and k in src and src[k] is not None:
                        return src[k]
            return None
        h = pick("horizon", "chosen_horizon")
        kk = pick("k", "chosen_k")
        ht = pick("hit_type", "chosen_hit_type")
        cur = CURRENT[name]
        changed = (h != cur["horizon"]) or (kk is not None and abs(float(kk) - cur["k"]) > 1e-9) \
            or (ht is not None and ht != cur["hit_type"])
        log("")
        log(f"  ⭐확장 격자 선택: HIT={ht} H={h} K={kk}")
        log(f"     현재 배포:     HIT={cur['hit_type']} H={cur['horizon']} K={cur['k']}")
        log(f"     ⇒ {'⚠️**선택이 바뀐다** -- 모델 대조 필요' if changed else '✅선택 불변 -- 격자가 짧았어도 최적은 같았다'}")
        edges = []
        for gk in spec["patch"]:
            g = spec["patch"][gk]
            v = h if "HORIZON" in gk else kk
            if v is not None:
                e = at_edge(v, g)
                if e:
                    edges.append(f"{gk} {e}")
        log(f"     경계 재점검: {'⚠️여전히 경계(' + ', '.join(edges) + ')' if edges else '✅내부값'}")
        rep["extensions"][name] = {
            "reason": spec["reason"], "grid_before": orig, "grid_after": spec["patch"],
            "chosen": {"hit_type": ht, "horizon": h, "k": kk},
            "current": cur, "changed": bool(changed),
            "still_at_edge": edges, "report": str(OUTDIR / f"{name}_gridscreen_report_EXT.json")}

    log("")
    log("=== 종합 ===")
    for n, v in rep["extensions"].items():
        if "error" in v:
            log(f"  {n:<24} ⚠️{v['error']}"); continue
        c = v["chosen"]
        log(f"  {n:<24} {c['hit_type']}/{c['horizon']}/{c['k']}  "
            f"{'⚠️변경' if v['changed'] else '✅불변'}"
            f"{'  (여전히 경계: ' + ', '.join(v['still_at_edge']) + ')' if v['still_at_edge'] else ''}")
    ch = [n for n, v in rep["extensions"].items() if v.get("changed")]
    log(f"  ⇒ 선택이 바뀐 신호: {ch if ch else '없음'}")
    log("  ⚠️바뀌었다고 교체하는 게 아니다 -- TabPFN 대조 + 부트스트랩 CI로 따로 판정한다(§5-A)")
    rep["changed_signals"] = ch
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
