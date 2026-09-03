#!/usr/bin/env python3
"""XRP `orthogonal_combo` **두께 제약 하 셀 선택** -- 안정성 + 경계 재확장.

## 왜

2026-09-03 감사에서 드러난 것:

  · 자동 argmax는 8시드 **전부 `touch_giveback_sustained`** 를 고르고 셀은 **6종으로 흩어진다**
    (giveback/12/2.5가 3회로 최빈). 그리고 그 유형은 **표본이 얇아**(79/34) 원래 두께 감사에서
    기각됐고, 사람이 `touch_mfe/8/2.0`(348/211)으로 갈아탄 것이 현재 배포본이다.
  · **두께 제약(MIN_TRAIN_HITS=150)을 걸고 확장 격자를 돌리니 `close_at_h/8/1.0`이 나왔다** --
    hits **401/201**(배포본 348/211보다 두껍다), lift **1.7137/1.7328**(배포본 ~1.56보다 높다).

⇒ 진짜 후보다. 다만 두 가지가 남았다:
   (1) **K=1.0이 그 격자의 하단 경계**였다 -> 아래로 더 넓힌다.
   (2) 자동 argmax가 시드에 따라 6종으로 흔들렸으므로 **두께 제약 하에서도 흔들리는지** 봐야 한다.

## 설계

원본 그리드스크린 `main()`을 그대로 재실행하고 아래만 patch:

    MIN_TRAIN_HITS   30 -> 150            (배포본 348/211에 견줄 두께)
    HORIZON_GRID     [8,...] -> [3,4,6,8,12,18,24,30,36]
    K_GRID           [2.0,...] -> [0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.57, 4.0, 4.5]

시드 8종으로 돌려 승자 분포를 낸다.

⚠️**여기서 이겨도 교체가 아니다.** 격자 lift는 raw 발동 전체에서, 모델 AUC는 dedup된 작은
평가셋에서 계산되므로 같은 방향을 가리킬 이유가 없다(demarker H=1 사례, 포팅 프로토콜 §5-A).
안정 승자가 나오면 **TabPFN 대조 + 평가표본 부트스트랩 CI**로 따로 판정한다.

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

SCRIPT = "research_xrp_orthogonal_combo_gridscreen_hittype_20260903.py"
OUTDIR = ROOT / "data/research/xrp_orthogonal_thick_20260903"
OUT = ROOT / "data/research/xrp_orthogonal_thick_constrained_selection_20260903.json"

SEEDS = [20260901, 11, 271828, 141592, 577215, 31337, 90210, 8675309]
THICK_MIN_HITS = 150
HORIZON_GRID = [3, 4, 6, 8, 12, 18, 24, 30, 36]
K_GRID = [0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.57, 4.0, 4.5]
DEPLOYED = ("touch_mfe", 8, 2.0)          # hits 348/211, lift ~1.56


def log(m): print(f"[ortho-thick] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sp = importlib.util.spec_from_file_location("m_ortho", ROOT / "scripts" / SCRIPT)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)

    log(f"두께 제약 MIN_TRAIN_HITS {mod.MIN_TRAIN_HITS} -> {THICK_MIN_HITS}")
    log(f"HORIZON_GRID -> {HORIZON_GRID}")
    log(f"K_GRID -> {K_GRID}   (하단 0.4까지 재확장)")
    log(f"시드 {len(SEEDS)}종 | 배포본 {DEPLOYED[0]}/{DEPLOYED[1]}/{DEPLOYED[2]} (hits 348/211)")

    saves = {"MIN_TRAIN_HITS": mod.MIN_TRAIN_HITS,
             "HORIZON_GRID": list(mod.HORIZON_GRID), "K_GRID": list(mod.K_GRID),
             "RNG_SEED": mod.RNG_SEED, "OUT_JSON": mod.OUT_JSON}
    rows = []
    try:
        mod.MIN_TRAIN_HITS = THICK_MIN_HITS
        mod.HORIZON_GRID = list(HORIZON_GRID)
        mod.K_GRID = list(K_GRID)
        for s in SEEDS:
            mod.RNG_SEED = s
            mod.OUT_JSON = OUTDIR / f"seed{s}.json"
            rc = mod.main()
            if rc != 0:
                log(f"  seed={s}: main() {rc}"); continue
            r = json.loads((OUTDIR / f"seed{s}.json").read_text())
            g = r.get("global_selection") or {}
            t = g.get("train", g)
            row = {"seed": s, "hit_type": g.get("hit_type"), "horizon": g.get("horizon"),
                   "k": g.get("k"), "lift_bottom": t.get("lift_bottom"),
                   "lift_top": t.get("lift_top"),
                   "hits_bottom": t.get("n_hits_bottom"), "hits_top": t.get("n_hits_top")}
            rows.append(row)
            log(f"  seed={s:<9} {row['hit_type']}/{row['horizon']}/{row['k']:<5} "
                f"lift {row['lift_bottom']}/{row['lift_top']}  hits {row['hits_bottom']}/{row['hits_top']}")
    finally:
        for k, v in saves.items():
            setattr(mod, k, v)

    cells = Counter((r["hit_type"], r["horizon"], r["k"]) for r in rows)
    log("")
    log(f"승자 분포 ({len(rows)}회):")
    for c, n in cells.most_common():
        log(f"  {c[0]}/{c[1]}/{c[2]:<5} {n}회 ({n/max(1,len(rows))*100:.0f}%)")
    top = cells.most_common(1)[0] if cells else None
    stable = len(cells) == 1
    log(f"⇒ {'✅단일 승자' if stable else f'⚠️{len(cells)}종으로 흔들림'}"
        f"  최빈 {top[0][0]}/{top[0][1]}/{top[0][2]} {top[1]}/{len(rows)}회" if top else "  판정 불가")

    edges = []
    if top:
        ht, h, k = top[0]
        if h == HORIZON_GRID[0]: edges.append("H 하단")
        if h == HORIZON_GRID[-1]: edges.append("H 상단")
        if k == K_GRID[0]: edges.append("K 하단")
        if k == K_GRID[-1]: edges.append("K 상단")
        log(f"  경계 점검: {'⚠️' + ', '.join(edges) if edges else '✅내부값'}")

    rep = {"min_train_hits": THICK_MIN_HITS, "horizon_grid": HORIZON_GRID, "k_grid": K_GRID,
           "seeds": SEEDS, "deployed": list(DEPLOYED), "per_seed": rows,
           "winner_counts": [{"cell": list(c), "n": n} for c, n in cells.most_common()],
           "distinct_winners": len(cells), "stable": bool(stable),
           "top_cell": list(top[0]) if top else None,
           "top_share": (top[1] / len(rows)) if top and rows else None,
           "still_at_edge": edges, "holdout_touched": False,
           "next_step": "안정 승자는 TabPFN 대조 + 부트스트랩 CI로 판정 (격자 lift != 모델 품질)",
           "runtime_sec": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
