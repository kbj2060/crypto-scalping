"""**손절을 두면 더 나은가** — 물타기를 버리고 빨리 도망가는 쪽 (2026-09-13, 사용자 제안).

사용자: *"물타기 개념을 지우고 스탑로스를 둬서 빨리 도망가게 하는건 어때?"*

## 왜 이게 «안전 장치»보다 큰 제안인가
2026-09-06 감사의 정정 한 줄이 핵심이다: **손절선이 청산선 안쪽이면 가격이 손절을 먼저
지나므로 청산은 구조적으로 불가능하다**(갭 제외). 그러면 생존 제약이 통째로 바뀐다.
  · 손절 없음: 손실 = MAE × L → 파산은 «MAE 가 1/L 에 닿는가»
  · 손절 s:    손실 = s × L (상한) → 파산은 «s·L >= 1 인가» = 결정론적
⇒ 손절은 크기를 **줄이는** 장치가 아니라 **키울 수 있게** 하는 장치다.
   대신 드문 큰 손실이 **잦은 작은 손실**로 바뀐다. 그 교환이 남는지가 이 실험이다.

## 체결 규약 — 09-07 결함을 피한다
`sim_exit` 이 시장이 이미 떠난 가격에 스톱을 놓고 체결시켜 OOS +6.06 → −15.75 로 뒤집힌 적이
있다. 여기서는 **봉내 고저로 도달을 확인**하고, 봉이 손절 너머에서 열리면(갭) **시가로** 채운다.
이건 이 저장소의 라이브 배리어 컨벤션과 같다(`omega4_6_1_live.py::evaluate_exit` 의 고가/저가).

## 대조군
  · **무손절**: 지금 배포판. L = min(생존 29.8배, 상한 6) = 6배, 4시간 보유.
  · **손절 s**: 같은 4시간, 손절 s%. L 은 (a) 같은 6배 (b) 파산 예산이 허용하는 최대.
(b)가 사용자 제안의 진짜 값이다 -- 손절이 파산을 막아 주면 그만큼 크게 갈 수 있다.

⚠️한계: 진입 무작위(재량 미반영) · 1분봉(그 안 경로 모름) · 갭은 봉 시가로만 근사 ·
슬리피지는 비용에 포함된 왕복 5.88bp 뿐(손절은 보통 테이커라 더 비싸다 -- 아래 TAKER_EXTRA).
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAPE_CANDIDATES = (ROOT / "data/research/eth_tape_1m_20260906.parquet",
                   pathlib.Path("/home/kbj20/crypto-scalping/data/research/eth_tape_1m_20260906.parquet"))
TRIPS = ROOT / "data/live/account_round_trips.jsonl"
COST_BP = 5.88
# 🔴손절은 **시장가**라 수수료 차이(3bp)만이 아니라 슬리피지가 붙는다.
# 실측(869일 1분봉, 3% 손절 발동 2,046건): 트리거 봉 안에서 손절선을 지나친 폭이
#   중앙 14.0bp · 90% 65.6bp · 99% 227bp · 최악 1,207bp (평균 28.9bp).
#   트리거 봉 종가가 이미 손절선 너머인 경우가 44.3% 다.
# 봉내 최저가 기준이라 상단이고, 운영 추정은 **중앙 14bp**를 쓴다. 수수료 3 + 슬리피지 14.
TAKER_EXTRA_BP = 17.0
HOLD = 240                  # 사용자 고정 보유시간
SEED = 20260913


def load(path=None):
    tape = pathlib.Path(path) if path else next((p for p in TAPE_CANDIDATES if p.exists()),
                                                TAPE_CANDIDATES[0])
    d = pd.read_parquet(tape, columns=["px_last", "px_max", "px_min"])
    return d.px_last.to_numpy(float), d.px_max.to_numpy(float), d.px_min.to_numpy(float)


def unit_returns() -> np.ndarray:
    rows = [json.loads(l) for l in TRIPS.read_text().splitlines() if l.strip()]
    tr = [x for x in rows if x.get("closed")]
    return np.array([x["net_pnl"] / (x["max_qty"] * x["entry_price"]) for x in tr])


def simulate(px, hi, lo, rng, n, *, L, stop, acc, hold=HOLD):
    """손절 stop(비율, None 이면 없음)·배수 L 로 hold 분 보유. 파산·수익·손절률을 돌려준다."""
    m = len(px)
    start = rng.integers(0, m - hold - 2, n)
    right = rng.random(n) < acc
    ret = np.empty(n); ruin = np.zeros(n, bool); stopped = np.zeros(n, bool)
    for t in range(n):
        a = int(start[t]); end = a + hold
        truth = 1.0 if px[end] >= px[a] else -1.0
        s = truth if right[t] else -truth
        e = px[a]
        seg_lo, seg_hi = lo[a:end + 1], hi[a:end + 1]
        adverse = (e - seg_lo.min()) / e if s > 0 else (seg_hi.max() - e) / e
        cost = COST_BP
        if stop is not None and adverse >= stop:
            # 손절에 **닿았다**. 봉내 고저로 확인했으므로 체결 가능한 자리다.
            # 갭은 근사: 도달 봉의 종가가 손절 너머면 그 종가로 채운다(보수적).
            k = a + int(np.argmax((e - seg_lo) / e >= stop if s > 0 else (seg_hi - e) / e >= stop))
            fill = -stop
            gap = (px[k] / e - 1) * s
            move = min(fill, gap)            # 갭이면 더 나쁜 쪽
            stopped[t] = True
            cost += TAKER_EXTRA_BP
        else:
            move = s * (px[end] / e - 1)
            if adverse * L >= 1.0:           # 손절이 없으면 청산이 가능하다
                ruin[t] = True; ret[t] = -1.0; continue
        r = L * (move - cost / 1e4)
        if r <= -1.0:
            ruin[t] = True; r = -1.0
        ret[t] = r
    return {"ret": ret, "ruin": float(ruin.mean()), "stop_rate": float(stopped.mean())}


def compound(ret_pool, rng, trades=735, paths=3000):
    out = np.empty(paths); dead = 0
    for p in range(paths):
        w = 1.0
        for r in rng.choice(ret_pool, trades):
            if r <= -1.0:
                w = 0.0; break
            w *= (1 + r)
            if w <= 0:
                w = 0.0; break
        if w == 0.0:
            dead += 1
        out[p] = w
    return float(np.median(out)), dead / paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape"); ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--acc", type=float, default=0.60)
    a = ap.parse_args()
    px, hi, lo = load(a.tape)
    rng = np.random.default_rng(SEED)
    print(f"1분봉 {len(px):,} · 표본 {a.n:,}/칸 · 보유 {HOLD}분 고정 · 정확도 {a.acc}")
    print(f"비용 왕복 {COST_BP}bp (손절은 +{TAKER_EXTRA_BP}bp 테이커)\n")
    print(f"{'팔':>18} {'L':>5} {'손절률':>7} {'건당bp':>9} {'파산%':>7} {'1년중앙':>9} {'1년파산%':>8}")

    rows = {}
    base = simulate(px, hi, lo, rng, a.n, L=6.0, stop=None, acc=a.acc)
    med, pr = compound(base["ret"], rng)
    rows["무손절 6배"] = (6.0, base, med, pr)
    print(f"{'무손절(현행) 6배':>18} {6.0:>5.1f} {'-':>7} {base['ret'].mean()*1e4:>+9.1f} "
          f"{100*base['ruin']:>6.2f}% {med:>9.2f} {100*pr:>7.1f}%")

    for stop in (0.01, 0.02, 0.03, 0.05):
        for lab, L in (("같은 6배", 6.0), ("최대", min(0.9 / stop, 25.0))):
            r = simulate(px, hi, lo, rng, a.n, L=L, stop=stop, acc=a.acc)
            med, pr = compound(r["ret"], rng)
            rows[f"손절{stop} {lab}"] = (L, r, med, pr)
            print(f"{f'손절 {100*stop:.0f}% · {lab}':>18} {L:>5.1f} {100*r['stop_rate']:>6.1f}% "
                  f"{r['ret'].mean()*1e4:>+9.1f} {100*r['ruin']:>6.2f}% {med:>9.2f} {100*pr:>7.1f}%")

    best = max(rows.items(), key=lambda kv: kv[1][2])
    print(f"\n⇒ 1년 중앙 계좌배수 최대: **{best[0]}** ({best[1][2]:.2f}배)")
    # 구조 주장 고정: 손절이 청산선 안쪽이면 파산은 거의 사라져야 한다
    for k, (L, r, _, _) in rows.items():
        if k.startswith("손절") and L * float(k.split()[0][2:]) < 1.0:
            assert r["ruin"] < 0.002, f"{k}: 손절이 청산선 안인데 파산 {r['ruin']:.3%}"
    print("확인: 손절이 청산선 안쪽인 칸에서는 파산이 사실상 0 (봉내 도달 기준).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
