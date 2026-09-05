#!/usr/bin/env python3
"""업그레이드 격자 -- **기준 팔 대비 일별 짝비교** (2026-09-05).

§5.27 표준: 결정 알고리즘 비교는 건당 기대값이 아니라 기준 팔 대비 **일별 짝비교 CI**로 한다
(같은 날 같은 모집단이라 상관이 높고, 짝비교가 훨씬 좁다).

  비교 1  물타기 축: 같은 (컨테이너·트랜치·레버리지)에서 물타기 팔 - 물타기없음
          ⭐**크기 매칭 필수**: 물타기 팔은 평균 k배의 트랜치를 넣는다. EV>0이면 크기만 키워도 이긴다.
            고정 레버리지에서 계좌%는 트랜치에 **선형**이고 청산선은 트랜치와 무관하므로,
            물타기없음 팔의 일별 계열에 k(창별 평균 트랜치)를 곱한 것이 정확한 동일크기 비교군이다.
  비교 2  컨테이너 축: 같은 (물타기·트랜치·레버리지)에서 각 컨테이너 - 트레일(검증본)
  비교 3  레버리지 축: 같은 (컨테이너·물타기·트랜치)에서 각 레버리지 - 3배
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/research/eth_avgdown_upgrade_verified_edge_20260905"
B, SEED = 2000, 20260905
WINDOWS = ("TRAIN", "VAL", "OOS")


def log(m): print(m, flush=True)


def paired(a, b, sp_arr, rng):
    """일별 평균 차이의 일군집 부트스트랩. a,b = 날짜별 평균(nan 허용)."""
    out = {}
    for sp in WINDOWS:
        m = (sp_arr == sp) & np.isfinite(a) & np.isfinite(b)
        d = a[m] - b[m]
        if len(d) < 5:
            out[sp] = None; continue
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
        out[sp] = {"n_days": int(len(d)), "mean": round(float(d.mean()), 4),
                   "ci": [round(float(np.percentile(bs, 2.5)), 4),
                          round(float(np.percentile(bs, 97.5)), 4)],
                   "win_days": round(float((d > 0).mean()), 3)}
    return out


def fmt(r):
    if r is None:
        return "  n/a "
    s = f"{r['mean']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"
    return s + (" ✅" if r["ci"][0] > 0 else (" ❌" if r["ci"][1] < 0 else "  ")) 


def main() -> int:
    cost = sys.argv[1] if len(sys.argv) > 1 else "4"
    rep = json.loads((D / "report.json").read_text())
    z = np.load(D / "daily.npz", allow_pickle=True)
    combos = [c.split("|") for c in z["combos"]]
    key = {tuple(c): i for i, c in enumerate(combos)}
    M = z[f"cost{cost}"]
    sp_arr = z["split"]
    rng = np.random.default_rng(SEED)
    log(f"=== 일별 짝비교 (계좌 %/건, 비용 {cost}bp) · {M.shape[0]}셀 x {M.shape[1]}일 ===\n")

    def name(c):
        ct, tx, am, ax, ma, tr, lv = c
        cn = {"trail": "트레일", "hybrid": "부분익절+트레일", "tp": f"고정TP{float(tx):.1f}bp"}[ct]
        an = "물타기없음" if am == "none" else (f"신호물타기x{ma}" if am == "signal"
                                          else f"물타기{float(ax):.1f}ATRx{ma}")
        return f"{cn}·{an}·{float(tr):.0%}x{float(lv):.0f}배"

    res = {"cost_bp": float(cost), "add_axis": [], "container_axis": [], "leverage_axis": []}

    log("── 비교 1: 물타기 팔 − 물타기없음 ⭐**동일 크기로 맞춘 뒤**(평균 트랜치 배수 곱함) ──")
    log(f"{'팔':44s} {'k̄':14s} {'TRAIN':26s} {'VAL':26s} {'OOS':26s}")
    for c in combos:
        ct, tx, am, ax, ma, tr, lv = c
        if am == "none":
            continue
        base = key.get((ct, tx, "none", "0.0", "0", tr, lv))
        if base is None:
            continue
        ci_ = key[tuple(c)]
        kbar = {sp: rep["cells"][ci_]["splits"].get(f"{sp}@meta", {}).get("mean_tranches", 1.0)
                for sp in WINDOWS}
        scaled = M[base].copy()
        for sp in WINDOWS:                       # 창별 k̄로 기준 팔을 확대 = 동일 명목
            scaled[sp_arr == sp] *= kbar[sp]
        r = paired(M[ci_], scaled, sp_arr, rng)
        aipf = rep["cells"][ci_]["splits"].get("TRAIN@meta", {}).get("adds_in_profit_frac")
        res["add_axis"].append({"arm": name(c), "vs": name(combos[base]) + " x k̄", "kbar": kbar,
                                "adds_in_profit_frac_TRAIN": aipf, "paired": r})
        kt = f"{kbar['TRAIN']:.2f}/{kbar['VAL']:.2f}/{kbar['OOS']:.2f}"
        log(f"{name(c):44s} {kt:14s} {fmt(r['TRAIN']):26s} {fmt(r['VAL']):26s} {fmt(r['OOS']):26s}")

    log("\n── 비교 2: 컨테이너 − 트레일(검증본) (같은 물타기·트랜치·레버리지) ──")
    log(f"{'팔':46s} {'TRAIN':26s} {'VAL':26s} {'OOS':26s}")
    for c in combos:
        ct, tx, am, ax, ma, tr, lv = c
        if ct == "trail":
            continue
        base = key.get(("trail", "0.0", am, ax, ma, tr, lv))
        if base is None:
            continue
        r = paired(M[key[tuple(c)]], M[base], sp_arr, rng)
        res["container_axis"].append({"arm": name(c), "vs": name(combos[base]), "paired": r})
        log(f"{name(c):46s} {fmt(r['TRAIN']):26s} {fmt(r['VAL']):26s} {fmt(r['OOS']):26s}")

    log("\n── 비교 3: 레버리지 − 3배 (같은 컨테이너·물타기·트랜치) ──")
    log(f"{'팔':46s} {'TRAIN':26s} {'VAL':26s} {'OOS':26s}")
    for c in combos:
        ct, tx, am, ax, ma, tr, lv = c
        if lv == "3.0":
            continue
        base = key.get((ct, tx, am, ax, ma, tr, "3.0"))
        if base is None:
            continue
        r = paired(M[key[tuple(c)]], M[base], sp_arr, rng)
        res["leverage_axis"].append({"arm": name(c), "vs": name(combos[base]), "paired": r})
        log(f"{name(c):46s} {fmt(r['TRAIN']):26s} {fmt(r['VAL']):26s} {fmt(r['OOS']):26s}")

    (D / f"paired_cost{cost}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    log(f"\n산출: {D}/paired_cost{cost}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
