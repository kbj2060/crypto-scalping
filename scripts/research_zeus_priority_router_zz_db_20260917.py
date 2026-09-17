#!/usr/bin/env python3
"""Zeus — **우선순위 라우터**: zigzag 우선, zigzag 가 쉴 때 더블배리어 (2026-09-17, 학습 없음)

사용자 아이디어: 확률 평균(N8 +21.71)이 아니라 **greedy 우선순위**로 묶는다.
이건 Omega4.6.1 이 이미 쓰는 구조다 -- `PRIORITY = ("h48qual", "zig075")`.

    zigzag 가 게이트를 통과하면 그 방향을 쓴다.
    zigzag 가 CASH 면 더블배리어 부모가 대신 낸다.
    둘 다 CASH 면 쉰다.

## 대조군 (사전 지정)
  P_zz   zigzag 우선 → 더블배리어 보조     ← 사용자 아이디어
  P_db   **역순**(더블배리어 우선)          ← ⭐순서를 바꿔도 같으면 라우터가 일을 안 하는 것
  N0     zigzag 단독                        ← 참조
  N7     더블배리어 단독                    ← 참조
  N8     확률 평균                          ← 이미 아는 값, 같은 매칭으로 재확인

⭐**단일 q 를 두 부모에 같이 걸고, 총 건수가 3,700 이 되는 q 를 찾는다.** 그래야 팔 간
선별성이 같아진다(건수가 다르면 「서열」과 「선별성」이 섞인다 -- 이 세션에서 반복 확인).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402
import research_omega461_side_skill_decomposition_20260917 as K  # noqa: E402

SEEDS = [613042, 27851, 904377, 155690, 488213]
FOLDS = [f for f in K.FOLDS if f[0] in ("F1", "F2", "F3", "CAND")]
CACHE = E.OUT / "stageP_probs.npz"
TPB, SLB, COST = K.BASE_TP * 1e4, K.BASE_SL * 1e4, 1.02
TARGET = 3700
EN = ("bull", "bear", "chop")


def log(*a): print(*a, flush=True)


def gate(D, Q, q):
    da = D.argmax(1)
    qf = np.where(da > 0, Q[np.arange(len(Q)), da], Q[:, 0])
    return np.where((da == 1) & (qf >= q), 1.0, np.where((da == 2) & (qf >= q), -1.0, 0.0))


def main() -> int:
    cache = dict(np.load(CACHE, allow_pickle=True))
    df, _ = E.load()
    segs = []
    for name, _t0, _t1, v0, v1 in FOLDS:
        te = df[(df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")].reset_index(drop=True)
        ev = tabm._route_probs(te).argmax(1)
        slots = []
        for sd in SEEDS:
            Dz = np.zeros((len(te), 3)); Qz = np.zeros((len(te), 3))
            Dd = np.zeros((len(te), 3)); Qd = np.zeros((len(te), 3))
            for ei, en in enumerate(EN):
                z = dict(cache[f"{name}|N0{en}s{sd}"].item())
                d = dict(cache[f"{name}|N7{ei}s{sd}"].item())
                m = ev == ei
                Dz[m], Qz[m] = z["D"][m], z["Q"][m]
                Dd[m], Qd[m] = d["D"][m], d["Q"][m]
            slots.append((Dz, Qz, Dd, Qd))
        segs.append((name, te, slots))
        log(f"  {name}: {len(te):,}봉 · {len(slots)} 시드")

    def side_of(arm, Dz, Qz, Dd, Qd, q):
        sz, sd_ = gate(Dz, Qz, q), gate(Dd, Qd, q)
        if arm == "N0":   return sz
        if arm == "N7":   return sd_
        if arm == "P_zz": return np.where(sz != 0, sz, sd_)        # zigzag 우선
        if arm == "P_db": return np.where(sd_ != 0, sd_, sz)       # 역순
        if arm == "N8":   return gate((Dz + Dd) / 2, (Qz + Qd) / 2, q)
        raise ValueError(arm)

    def count(arm, q):
        return float(np.mean([sum(int((side_of(arm, *sl, q) != 0).sum()) for _n, _t, slots in segs
                                  for sl in [slots[i]]) for i in range(len(SEEDS))]))

    def evaluate(arm, q):
        per = []
        for i in range(len(SEEDS)):
            pnl, hold, days = [], [], []
            for name, te, slots in segs:
                side = side_of(arm, *slots[i], q)
                idx = np.where(side != 0)[0]
                if len(idx) < 20:
                    continue
                hi = pd.to_numeric(te["high"]).to_numpy(float)
                lo = pd.to_numeric(te["low"]).to_numpy(float)
                cl = pd.to_numeric(te["close"]).to_numpy(float)
                r, h, _a, _b, _c = K._first_touch_open(idx, side, hi, lo, cl,
                                                       K.BASE_TP, K.BASE_SL, K.MAXBARS)
                pnl.append(r * 1e4 - COST); hold.append(h.astype(float))
                days.append(te.timestamp.dt.floor("D").to_numpy()[idx])
            pnl = np.concatenate(pnl); hold = np.concatenate(hold); days = np.concatenate(days)
            lo_, hi_, _ = E.block_ci(pnl, days)
            per.append((len(pnl), float(pnl.mean()), lo_, hi_, 288.0 / max(hold.mean(), 1e-9),
                        float(np.median(hold))))
        a = np.array(per, float)
        g = a[:, 1]
        return dict(n=a[:, 0].mean(), gross_bp=g.mean(), ci=[a[:, 2].mean(), a[:, 3].mean()],
                    per_day=a[:, 4].mean(), median_hold=a[:, 5].mean(),
                    p=(g.mean() + COST + SLB) / (TPB + SLB), seed_spread=g.max() - g.min())

    log(f"\n{'팔':<8}{'q':>6}{'건수':>8}{'건/일':>7}{'중앙보유':>9}{'건당bp':>9}{'함축p':>8}"
        f"{'CI(건당)':>20}{'순/일':>8}{'시드폭':>8}")
    rows = []
    for arm in ("N0", "N7", "N8", "P_zz", "P_db"):
        # 총 건수가 TARGET 이 되는 q 를 이분탐색
        lo_q, hi_q = 0.30, 0.995
        for _ in range(40):
            mid = (lo_q + hi_q) / 2
            if count(arm, mid) > TARGET: lo_q = mid
            else: hi_q = mid
        q = (lo_q + hi_q) / 2
        r = evaluate(arm, q); r["arm"] = arm; r["q"] = q
        rows.append(r)
        log(f"{arm:<8}{q:>6.3f}{int(r['n']):>8,}{r['per_day']:>7.2f}{r['median_hold']:>9.0f}"
            f"{r['gross_bp']:>+9.2f}{r['p']*100:>7.2f}%  [{r['ci'][0]:+7.2f},{r['ci'][1]:+7.2f}]"
            f"{r['gross_bp']*r['per_day']:>8.1f}{r['seed_spread']:>8.2f}")
    d = {r["arm"]: r for r in rows}
    log(f"\n⭐P_zz − N0 = {d['P_zz']['gross_bp'] - d['N0']['gross_bp']:+.2f}bp "
        f"(순/일 {d['P_zz']['gross_bp']*d['P_zz']['per_day'] - d['N0']['gross_bp']*d['N0']['per_day']:+.1f})"
        f"  ← 더블배리어가 «쉴 때» 보태는 값")
    log(f"⭐P_zz − P_db = {d['P_zz']['gross_bp'] - d['P_db']['gross_bp']:+.2f}bp"
        f"  ← 0 에 가까우면 «순서»가 일을 안 하는 것(라우터 무의미)")
    log(f"⭐P_zz − N8  = {d['P_zz']['gross_bp'] - d['N8']['gross_bp']:+.2f}bp  ← 우선순위 vs 확률평균")
    log(f"⚠️N0 시드폭 {d['N0']['seed_spread']:.2f} -- 위 차이들이 이보다 작으면 잡음과 구분 불가")
    (E.OUT / "stageT_priority_router.json").write_text(json.dumps(rows, indent=2, default=float))
    log(f"저장: {E.OUT}/stageT_priority_router.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
