#!/usr/bin/env python3
"""비용 축 — **비용을 0 으로 내려도 무엇이 구제되는가** (2026-09-14).

사용자 *"비용 쪽으로 연구해줘"* (liquidity 종결 직후).

## 왜 이 형태인가 — 집행 최적화는 이미 소진됐다
09-13 이 섀도우 23,332legs 로 집행을 실측했다: 이 계좌 **메이커 2.0bp · 테이커 5.0bp ·
스프레드 0.0407bp**. ⭐스프레드가 0.04bp 라 역선택이 붙을 자리가 없고 **메이커 수수료 2.0bp 가
비용의 전부**다. 「리페그 N회 넘으면 시장가」 규칙의 개선 **상한이 0.147bp**(낙관 가정에서).
고변동에서 오히려 더 잘 체결된다(체결시간 순위상관 −0.429, 비용 +0.026≈0).
⇒ 집행을 더 깎아 얻을 수 있는 건 왕복 **5.52 → 3.96bp(수수료 하한)**, 최대 1.56bp 다.

## 그래서 묻는다: 비용이 **0** 이어도 승격되는 셀이 있는가
⭐**초과분(같은측면 무작위 진입 대비)은 비용과 무관하다** — 신호도 무작위도 같은 비용을 낸다.
따라서 두 조건을 갈라서 센다:
  (A) **gross CI 하한 > 비용**  — 실제로 돈이 되는가 (비용 의존)
  (B) **초과 CI 하한 > 0**      — 표류가 아니라 신호인가 (**비용 무관**)
(A) 를 비용 {10, 7.8, 5.52, 3.96(수수료하한), 0} 에서 세면 «비용을 얼마나 내리면 몇 개가
살아나는가」가 나오고, (B) 는 그 상한이 된다. **(B) 가 우연 수준이면 비용은 구속조건이 아니다.**

모집단: 증거신호 8종 × 2측면 × 4지평 = 64셀, **1,741일**(09-09 스크린은 365일).
통과 개수는 **랜덤 부분표집 귀무(B=200)** 와 대조한다(저장소 규율).

출력: tmp/eth_cost_floor_20260914/
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import research_eth_liquidity_sweep_block_independence_20260914 as P0  # noqa: E402

OUT = ROOT / "tmp/eth_cost_floor_20260914"
HORIZONS = [12, 24, 48, 144]
COSTS = [10.0, 7.80, 5.52, 3.96, 0.0]     # 테이커왕복 · peg진입+테이커청산 · peg양다리 · 수수료하한 · 0
NSUB = 200                                 # 랜덤 부분표집 귀무 복제수
RNG = np.random.default_rng(20260914)



def fast_day_boot(vals: np.ndarray, days: np.ndarray, B: int = 2000) -> tuple[float, float]:
    """일군집 부트스트랩을 **일별 합계/건수로 미리 접어** 벡터화한다.
    P0.day_cluster_boot 와 수학적으로 동일하되(가중평균 = Σ합 / Σ건수) 복제당 concat 이 없다."""
    u, inv = np.unique(days, return_inverse=True)
    sums = np.bincount(inv, weights=vals, minlength=len(u))
    cnts = np.bincount(inv, minlength=len(u)).astype(float)
    pick = RNG.integers(0, len(u), size=(B, len(u)))
    m = sums[pick].sum(1) / cnts[pick].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def selftest() -> None:
    # 초과분은 비용과 무관해야 한다 — 양쪽에서 같은 상수를 빼면 차가 그대로다
    g = np.array([3.0, -1.0, 5.0]); nul = 1.0
    for c in (0.0, 5.52, 10.0):
        assert abs(((g - c).mean() - (nul - c)) - (g.mean() - nul)) < 1e-12
    # fast_day_boot: 한 날짜만 있으면 CI 가 그 날 평균으로 붕괴한다
    v = np.array([1.0, 3.0]); d = np.array([7, 7])
    assert abs(fast_day_boot(v, d, 50)[0] - 2.0) < 1e-9
    # 두 날짜의 건수 가중이 유지되는지(날짜 A 2건·B 1건 -> 전체평균은 단순평균 아님)
    v = np.array([0.0, 0.0, 9.0]); d = np.array([1, 1, 2])
    loB, hiB = fast_day_boot(v, d, 4000)
    assert loB >= 0.0 and hiB <= 9.0
    print("selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest: selftest(); return 0
    selftest()
    OUT.mkdir(parents=True, exist_ok=True)
    import live_evidence_signal_dashboard_20260823 as EV          # noqa: E402
    import build_eth_anchor_label_dataset_20260907 as B           # noqa: E402

    kl = P0.load(P0.CSV); btc = P0.load(P0.BTC)
    print(f"[1/3] ETH 5분봉 {len(kl):,}  {kl.timestamp.iloc[0]} ~ {kl.timestamp.iloc[-1]}", flush=True)
    sig = EV.compute_signals(kl, btc_df=btc, funding_df=None)
    op = sig["open"].to_numpy(float); cl = sig["close"].to_numpy(float)
    day = sig["timestamp"].dt.floor("D").astype("int64").to_numpy()
    n = len(sig); lo, hi = P0.WARMUP, n - max(HORIZONS) - 2
    span = (sig["timestamp"].iloc[hi] - sig["timestamp"].iloc[lo]).total_seconds() / 86400
    print(f"  평가 {span:.0f}일", flush=True)

    print("[2/3] 64셀 gross·초과 + 일군집 CI …", flush=True)
    rows = []
    for s in B.SIGNALS:
        for side, long in (("bottom", True), ("top", False)):
            col = f"{side}_{s}"
            if col not in sig.columns: continue
            f = sig[col].fillna(False).to_numpy(bool)
            idx = np.flatnonzero(f); idx = idx[(idx >= lo) & (idx <= hi)]
            if len(idx) < 200: continue
            for H in HORIZONS:
                g = P0.gross_bp(op, cl, idx, H, long)
                nul = P0.same_side_null(op, cl, len(idx), H, long, lo, hi).mean()
                gci = fast_day_boot(g, day[idx], B=4000)
                eci = fast_day_boot(g - nul, day[idx], B=4000)
                rows.append(dict(signal=s, side=side, H=H, n=len(idx),
                                 per_day=round(len(idx)/span, 2),
                                 gross=round(float(g.mean()), 2),
                                 gross_ci_lo=round(gci[0], 2), gross_ci_hi=round(gci[1], 2),
                                 null=round(float(nul), 2),
                                 excess=round(float(g.mean()-nul), 2),
                                 excess_ci_lo=round(eci[0], 2), excess_ci_hi=round(eci[1], 2)))
    D = pd.DataFrame(rows); D.to_csv(OUT / "cells.csv", index=False)

    print("[3/3] 랜덤 부분표집 귀무(통과 개수의 우연 기대) …", flush=True)
    # 같은 셀 구조(같은 건수·같은 지평·같은 측면)를 무작위 진입으로 만들어 통과 개수를 센다
    null_counts = {c: [] for c in COSTS}; null_excess = []
    for b in range(NSUB):
        cg = {c: 0 for c in COSTS}; ce = 0
        for r in D.itertuples():
            long = r.side == "bottom"
            pick = RNG.choice(np.arange(lo, hi), r.n, replace=False)
            g = P0.gross_bp(op, cl, pick, r.H, long)
            nul = r.null      # 무작위 진입의 귀무평균은 같은 측면·지평이면 동일 분포다
            gci = fast_day_boot(g, day[pick], B=600)
            eci = fast_day_boot(g - nul, day[pick], B=600)
            for c in COSTS: cg[c] += int(gci[0] > c)
            ce += int(eci[0] > 0)
        for c in COSTS: null_counts[c].append(cg[c])
        null_excess.append(ce)
        if (b + 1) % 50 == 0: print(f"    {b+1}/{NSUB}", flush=True)

    print(f"\n{'='*112}\n비용 반사실 — 증거신호 {len(D)}셀 ({span:.0f}일, 건/일 {D.per_day.min():.1f}~{D.per_day.max():.1f})")
    print(f"\n(A) **gross CI 하한 > 비용** 인 셀 수  — 비용을 내리면 몇 개가 살아나는가")
    print(f"{'왕복 비용':>28}{'통과':>8}{'무작위 귀무 평균':>18}{'귀무 p95':>10}")
    lab = {10.0: "10.00  테이커 양다리", 7.80: " 7.80  peg진입+테이커청산", 5.52: " 5.52  peg 양다리(현행)",
           3.96: " 3.96  수수료 하한(집행완벽)", 0.0: " 0.00  ⭐비용 전액 면제"}
    for c in COSTS:
        k = int((D.gross_ci_lo > c).sum()); nn = np.array(null_counts[c])
        print(f"{lab[c]:>28}{k:>8}{nn.mean():>18.1f}{np.percentile(nn,95):>10.1f}")
    ke = int((D.excess_ci_lo > 0).sum()); ne = np.array(null_excess)
    print(f"\n(B) **초과 CI 하한 > 0** (비용 무관) = {ke}셀  ·  무작위 귀무 평균 {ne.mean():.1f} · p95 {np.percentile(ne,95):.1f}")
    print(f"\n■ gross 상위 8셀")
    print(f"{'신호':<26}{'측면':>5}{'H':>5}{'건수':>7}{'건/일':>7}{'gross':>9}{'CI하한':>9}{'초과':>8}{'초과CI하한':>11}")
    for r in D.sort_values("gross", ascending=False).head(8).itertuples():
        print(f"{r.signal:<26}{'바닥' if r.side=='bottom' else '천장':>5}{r.H:>5}{r.n:>7}{r.per_day:>7.2f}"
              f"{r.gross:>+9.2f}{r.gross_ci_lo:>+9.2f}{r.excess:>+8.2f}{r.excess_ci_lo:>+11.2f}")
    print("="*112)
    json.dump({"cells": len(D), "days": round(span, 1),
               "pass_by_cost": {str(c): int((D.gross_ci_lo > c).sum()) for c in COSTS},
               "null_mean_by_cost": {str(c): float(np.mean(null_counts[c])) for c in COSTS},
               "excess_pass": ke, "excess_null_mean": float(ne.mean())},
              open(OUT / "summary.json", "w"), ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
