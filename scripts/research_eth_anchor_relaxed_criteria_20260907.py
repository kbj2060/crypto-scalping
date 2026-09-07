#!/usr/bin/env python3
"""완화된 판정 기준 사다리 — **기준을 낮추면 무엇이 통과하는가** (2026-09-07).

사용자: *"참의 기준을 조금 낮추는건 어때? 지금 너무 기준이 높은거 같은데"*

## 지적의 타당한 부분
현재 기준(세 창 **각각** 일군집 CI 하한 > 0.5)은 노이즈 검정 3개의 **논리곱**이다.
창당 444~612행이고 독립 일수는 그보다 훨씬 적어(호메로스 기록: 42~45일) CI 가 넓다.
**세 창을 하나로 합치면 검정력이 훨씬 높고 똑같이 정당하다** -- 셋 다 TRAIN 밖이므로.

## 그러나 기준을 낮추면 귀무도 같이 통과한다
그래서 각 기준마다 **같은 기준에서 귀무가 몇 개를 통과시키는지**를 함께 낸다.
비교 대상 없이 "완화 기준에서 통과!"는 의미가 없다.

## 입력 -- 이 저장소에서 가장 규약이 엄격한 예측
`tmp/eth_anchor_walkforward_20260907/preds_*.npy`
월 1회 재학습 walk-forward(부록 S)의 앵커별 예측. 각 예측이 **그 시점 이전 데이터만**
쓴다. 10개월 × 3팔. 세 창을 합치면 n≈1500, 독립 일수 ~300 으로 창별 검정보다 훨씬 넓다.

## 기준 사다리
  L0 현행    세 창 각각 CI 하한 > 0.5
  L1 점추정  세 창 각각 AUC > 0.5 (CI 무시)
  L2 풀링    세 창 합쳐 AUC 하나 + 일군집 CI 하한 > 0.5   ← 검정력 최고, 사용자 취지
  L3 선별    상위 30% 예측의 지속률 - 기저 > 0            ← AUC 대신 실사용 형태
  L4 경제    상위 K% 진입 시 net bp > 0 (비용 차감)       ← 최종 잣대

각 기준에서 **관측 vs 날블록 귀무 B=200** 을 나란히 낸다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "tmp/eth_anchor_walkforward_20260907"
SRC = ROOT / "tmp/eth_anchor_features154_20260907"
OUT = ROOT / "tmp/eth_anchor_relaxed_20260907"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
B_NULL = 200
SEED = 20260907

CELLS = [("hard", "dirtop20"), ("three", "perm20"), ("wbin", "perm20")]


def arm_spec(D, arm):
    y3 = D["y3"].to_numpy()
    if arm == "hard":
        return D["is_clean"].to_numpy(bool), (y3 == 2).astype(int), False
    if arm == "wbin":
        y = D["y_bin"].to_numpy()
        return np.isfinite(y), np.nan_to_num(y).astype(int), False
    return np.isfinite(y3), y3.astype(int), True


def day_ci(y, p, d, rng, B=1500):
    u = np.unique(d)
    if len(u) < 5:
        return (np.nan, np.nan)
    idx = {x: np.flatnonzero(d == x) for x in u}
    o = []
    for _ in range(B):
        ii = np.concatenate([idx[x] for x in rng.choice(u, len(u), replace=True)])
        if len(np.unique(y[ii])) > 1:
            o.append(roc_auc_score(y[ii], p[ii]))
    return (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))) if len(o) > B // 3 else (np.nan, np.nan)


def dayblock_perm(y, d, rng):
    """예측은 고정, 라벨만 날 블록 단위로 재배치 (일내 셔플 금지)."""
    u = np.unique(d); src = {x: np.flatnonzero(d == x) for x in u}
    perm = rng.permutation(u); yb = y.copy()
    for a, b in zip(u, perm):
        if len(src[a]):
            yb[src[a]] = np.resize(y[src[b]], len(src[a]))
    return yb


def select_lift(y, p, q=0.30):
    """상위 q 비율 예측의 지속률 - 전체 기저."""
    k = max(10, int(len(p) * q))
    top = np.argsort(-p)[:k]
    return float(y[top].mean() - y.mean())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    D = pd.read_parquet(SRC / "features154.parquet")
    sp = D["split"].to_numpy(); day = D["timestamp"].dt.floor("D").to_numpy()
    # net bp: 지속 방향 진입의 비용후 수익 (라벨셋에 있으면 사용)
    bp_col = next((c for c in ("net_cont_bp", "cont_net_bp") if c in D.columns), None)
    print(f"[입력] 앵커 {len(D):,} · net bp 컬럼: {bp_col or '없음(L4 생략)'}", flush=True)

    rows = []
    for arm, fs in CELLS:
        f = WF / f"preds_{arm}_{fs}.npy"
        if not f.exists():
            print(f"  {arm}/{fs}: 예측 없음"); continue
        pr = np.load(f)
        mask, y, multi = arm_spec(D, arm)
        ok = mask & np.isfinite(pr)
        if multi:
            ok = ok & (y != 1)
        yy_all = (y == 2).astype(int) if multi else y

        rec = {"arm": arm, "featset": fs}
        # ---- L0/L1 창별
        n_ci, n_pt = 0, 0
        for w in WINS:
            s = ok & (sp == w)
            if s.sum() < 30 or len(np.unique(yy_all[s])) < 2:
                continue
            a = roc_auc_score(yy_all[s], pr[s]); lo, hi = day_ci(yy_all[s], pr[s], day[s], rng)
            rec[f"{w}_auc"], rec[f"{w}_lo"] = a, lo
            n_ci += int(lo > 0.5); n_pt += int(a > 0.5)
        rec["L0_windows_ci"], rec["L1_windows_pt"] = n_ci, n_pt

        # ---- L2 풀링 (세 창 합쳐 하나)
        s = ok & np.isin(sp, WINS)
        yv, pv, dv = yy_all[s], pr[s], day[s]
        a = roc_auc_score(yv, pv); lo, hi = day_ci(yv, pv, dv, rng)
        rec.update(POOL_n=int(s.sum()), POOL_days=int(len(np.unique(dv))),
                   POOL_auc=a, POOL_lo=lo, POOL_hi=hi)

        # ---- L3 선별 리프트
        rec["L3_lift30"] = select_lift(yv, pv)

        # ---- L4 경제
        if bp_col:
            bp = D[bp_col].to_numpy(float)[s]
            k = max(10, int(len(pv) * 0.30))
            top = np.argsort(-pv)[:k]
            rec["L4_top30_bp"] = float(np.nanmean(bp[top]))
            rec["L4_all_bp"] = float(np.nanmean(bp))

        # ---- 귀무: 같은 기준에서 우연이 무엇을 주는가
        nl_auc, nl_lo, nl_lift, nl_bp = [], [], [], []
        for _ in range(B_NULL):
            yb = dayblock_perm(yv, dv, rng)
            if len(np.unique(yb)) < 2:
                continue
            nl_auc.append(roc_auc_score(yb, pv))
            nl_lift.append(select_lift(yb, pv))
        # 귀무 CI 하한은 비싸므로 축소 표본으로
        for _ in range(30):
            yb = dayblock_perm(yv, dv, rng)
            l, _h = day_ci(yb, pv, dv, rng, B=300)
            nl_lo.append(l)
        rec["NULL_auc_p95"] = float(np.percentile(nl_auc, 95))
        rec["NULL_auc_p50"] = float(np.percentile(nl_auc, 50))
        rec["NULL_lo_gt05_rate"] = float(np.mean(np.array(nl_lo) > 0.5))
        rec["NULL_lift_p95"] = float(np.percentile(nl_lift, 95))
        rec["p_auc"] = float(np.mean(np.array(nl_auc) >= a))
        rec["p_lift"] = float(np.mean(np.array(nl_lift) >= rec["L3_lift30"]))
        rows.append(rec)
        print(f"  {arm}/{fs} 완료", flush=True)

    A = pd.DataFrame(rows)
    A.to_csv(OUT / "relaxed.csv", index=False)

    print("\n" + "=" * 108, flush=True)
    print("기준 사다리 — 관측 vs 같은 기준에서의 귀무", flush=True)
    print("=" * 108, flush=True)
    for _, r in A.iterrows():
        print(f"\n[{r['arm']}/{r['featset']}]  풀링 n={r.POOL_n} · 독립일수 {r.POOL_days}")
        print(f"  L0 세 창 각각 CI>0.5      : {int(r.L0_windows_ci)}/3 창 통과")
        print(f"  L1 세 창 각각 점추정>0.5  : {int(r.L1_windows_pt)}/3 창 통과")
        print(f"  L2 세 창 풀링 AUC         : {r.POOL_auc:.4f} [{r.POOL_lo:.4f}, {r.POOL_hi:.4f}]"
              f"  · 귀무 중앙 {r.NULL_auc_p50:.4f} p95 {r.NULL_auc_p95:.4f} → p={r.p_auc:.3f}")
        print(f"     ⚠️귀무가 CI하한>0.5 를 내는 비율: {r.NULL_lo_gt05_rate:.1%}")
        print(f"  L3 상위30% 선별 리프트     : {r.L3_lift30:+.4f}"
              f"  · 귀무 p95 {r.NULL_lift_p95:+.4f} → p={r.p_lift:.3f}")
        if "L4_top30_bp" in r and pd.notna(r.get("L4_top30_bp")):
            print(f"  L4 상위30% net bp         : {r.L4_top30_bp:+.2f}bp (전체 {r.L4_all_bp:+.2f}bp)")
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
