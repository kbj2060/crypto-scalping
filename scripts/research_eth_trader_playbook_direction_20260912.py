#!/usr/bin/env python3
"""유명 트레이더 전략 6종 + 앙상블 — 방향 (2026-09-12).

사용자: *"외부 자료와 최신 논문과 유명 트레이더들의 전략을 모두 조합해서 방향성을 가진 모델을
만들어보자. 데이터는 충분하고 maker 로 거래하면 수수료 0.04% 니 이것보다 높으면 된다."*

## 왜 «새 전략»인가 — 조합은 이미 닫혔다
같은 날 재료 58피쳐 스택(증거신호8·트리거7·문맥·극점확률)을 전진 26개월로 붙였더니 방향
건당 **0.26bp(H=12) / 2.02bp(H=48)** 였다. 문헌 처방 3종도 전부 닫혔다. **4bp 로 낮춰도 안 넘는다.**
⇒ 이미 있는 걸 다시 섞는 게 아니라 **저장소에 없는 축**을 넣어야 한다. 확인 결과 아래 6종의
구현이 하나도 없다(증거신호 8종은 오실레이터·테이커·기하 계열이고 이건 **구조·세션 계열**이다).

## 전략 6종 (전부 klines 만으로 · 봉 i 종가까지만 본다)
  fvg        Fair Value Gap / 오더블록 (ICT) — 3봉 갭이 열리고 그 구간으로 되돌아왔나
  wyckoff    스프링/업스러스트 — 레인지 저점 이탈 후 즉시 회복(가짜 이탈)
  donchian   터틀 — N봉 최고/최저 돌파 (고전 추세추종)
  orb        오프닝 레인지 브레이크아웃 — 세션 첫 R봉 레인지 이탈
  vwap_rev   앵커드 VWAP 회귀 — 세션 VWAP 대비 σ 이탈
  poc        볼륨 프로파일 POC 거리 — 최대거래량 가격대에서 얼마나 떨어져 있나

## 판정 (이 저장소 표준)
진입 i+1 시가 · 청산 i+H 종가. 건당은 **실제 부호수익 평균**(`(2a−1)E|r|` 금지).
비용선 **4.0bp(사용자 기준·수수료만)** 와 **5.52bp(실측 peg 왕복)** 둘 다 병기.
귀무 = 순환이동(B=400) · 두 반기 · 일블록 부트. 앙상블은 **합의 개수**와 **로지스틱 스태킹** 둘 다.
자체점검 --selftest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

FEE_USER, FEE_REAL = 4.0, 5.52
HS = (12, 48, 144)


def strategies(d: pd.DataFrame) -> dict[str, np.ndarray]:
    """각 전략의 방향 신호 (+1 롱 / −1 숏 / 0 무신호). 전부 봉 i 종가까지만 본다."""
    o = d.open.to_numpy(float); h = d.high.to_numpy(float)
    lo = d.low.to_numpy(float); c = d.close.to_numpy(float)
    v = d.volume.to_numpy(float)
    n = len(d)
    S = {}

    # FVG — 봉 k-2 고가 < 봉 k 저가(상승 갭)면 그 갭이 «미체결 주문 구간». 되돌아오면 롱.
    up_gap = np.zeros(n, bool); dn_gap = np.zeros(n, bool)
    up_gap[2:] = lo[2:] > h[:-2]
    dn_gap[2:] = h[2:] < lo[:-2]
    gl = pd.Series(np.where(up_gap, h[np.arange(n) - 2], np.nan)).ffill(limit=48).to_numpy()
    gh = pd.Series(np.where(dn_gap, lo[np.arange(n) - 2], np.nan)).ffill(limit=48).to_numpy()
    S["fvg"] = np.where(np.isfinite(gl) & (lo <= gl), 1.0,
                        np.where(np.isfinite(gh) & (h >= gh), -1.0, 0.0))

    # Wyckoff 스프링/업스러스트 — 48봉 레인지를 이탈했다가 같은 봉에 안으로 회복
    rl = pd.Series(lo).rolling(48).min().shift(1).to_numpy()
    rh = pd.Series(h).rolling(48).max().shift(1).to_numpy()
    S["wyckoff"] = np.where((lo < rl) & (c > rl), 1.0, np.where((h > rh) & (c < rh), -1.0, 0.0))

    # Donchian 터틀 — 55봉 채널 돌파(종가 확정)
    dl = pd.Series(lo).rolling(55).min().shift(1).to_numpy()
    dh = pd.Series(h).rolling(55).max().shift(1).to_numpy()
    S["donchian"] = np.where(c > dh, 1.0, np.where(c < dl, -1.0, 0.0))

    # ORB — UTC 세션 첫 6봉(30분) 레인지를 이탈
    ts = pd.DatetimeIndex(d.timestamp)
    bar = ts.hour.to_numpy() * 12 + ts.minute.to_numpy() // 5
    day = ts.normalize()
    f = pd.DataFrame({"d": day, "b": bar, "h": h, "l": lo})
    opn = f[f.b < 6].groupby("d").agg(oh=("h", "max"), ol=("l", "min"))
    oh = pd.Series(day.map(opn.oh).to_numpy(), dtype=float).to_numpy()
    ol = pd.Series(day.map(opn.ol).to_numpy(), dtype=float).to_numpy()
    inday = bar >= 6
    S["orb"] = np.where(inday & (c > oh), 1.0, np.where(inday & (c < ol), -1.0, 0.0))

    # 앵커드 VWAP — 그날 누적 VWAP 대비 2σ 이탈이면 회귀 베팅
    pv = pd.Series(c * v).groupby(day).cumsum().to_numpy()
    vv = pd.Series(v).groupby(day).cumsum().to_numpy()
    vwap = pv / np.maximum(vv, 1e-9)
    dev = (c - vwap) / np.maximum(vwap, 1e-9)
    sd = pd.Series(dev).rolling(288, min_periods=96).std().to_numpy()
    S["vwap_rev"] = np.where(dev < -2 * sd, 1.0, np.where(dev > 2 * sd, -1.0, 0.0))

    # 볼륨 프로파일 POC — 288봉 창에서 거래량 최대 가격대. 멀어지면 회귀 베팅.
    poc = np.full(n, np.nan)
    W, BINS = 288, 24
    for i in range(W, n, 6):                       # 6봉마다 갱신(30분) — 계산량 관리
        pr = c[i - W:i]; vw = v[i - W:i]
        span = float(pr.max() - pr.min())
        idx = np.clip(((pr - pr.min()) / max(span, 1e-9) * (BINS - 1)).astype(int), 0, BINS - 1)
        poc[i:i + 6] = pr.min() + (np.bincount(idx, vw, BINS).argmax() + 0.5) / BINS * span
    pd_ = (c - poc) / np.maximum(poc, 1e-9)
    psd = pd.Series(pd_).rolling(288, min_periods=96).std().to_numpy()
    S["poc"] = np.where(pd_ < -1.5 * psd, 1.0, np.where(pd_ > 1.5 * psd, -1.0, 0.0))
    return S


def evaluate(sig: np.ndarray, fwd: np.ndarray) -> dict:
    m = (sig != 0) & np.isfinite(fwd)
    if m.sum() < 100:
        return {"n": int(m.sum())}
    v = sig[m] * fwd[m] * 1e4
    return {"n": int(m.sum()), "hit": float((v > 0).mean()), "bp": float(v.mean()),
            "net_user": float(v.mean() - FEE_USER), "net_real": float(v.mean() - FEE_REAL),
            "_v": v, "_i": np.flatnonzero(m)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        d = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=400, freq="5min"),
                          "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0})
        S = strategies(d)
        assert set(S) == {"fvg", "wyckoff", "donchian", "orb", "vwap_rev", "poc"}, S.keys()
        for k, s in S.items():
            assert len(s) == len(d) and set(np.unique(s)) <= {-1.0, 0.0, 1.0}, k
        r = evaluate(np.array([1.0, -1.0, 1.0]), np.array([0.001, -0.001, -0.001]))
        assert r["n"] == 3
        r2 = evaluate(np.ones(200), np.full(200, 0.001))
        assert abs(r2["bp"] - 10.0) < 1e-9 and r2["hit"] == 1.0, r2
        print("selftest OK — 전략 6종 ±1/0 · 건당 부호수익")
        return 0

    kl = B._load_kl(B.ETH_KL)
    kl = kl[kl.timestamp >= pd.Timestamp(a.start)].reset_index(drop=True)
    S = strategies(kl)
    c = kl.close.to_numpy(float); o = kl.open.to_numpy(float); n = len(kl)
    ent = np.roll(o, -1); ent[-1] = np.nan
    ts = kl.timestamp.to_numpy()
    print(f"ETH 5분봉 {n:,} ({str(ts[0])[:10]} ~ {str(ts[-1])[:10]}) · 비용선 "
          f"사용자 {FEE_USER}bp / 실측 {FEE_REAL}bp\n")
    for H in HS:
        fwd = np.full(n, np.nan)
        fwd[: n - H - 1] = c[H + 1 : n] / ent[: n - H - 1] - 1.0
        print(f"=== H={H}봉 ({H*5//60}시간{H*5%60:02d}분) ===")
        print(f"{'전략':<11}{'n':>8}{'건/일':>7}{'적중':>8}{'건당bp':>9}{'순익4bp':>9}{'순익5.52':>10}")
        keep = {}
        days = (pd.Timestamp(ts[-1]) - pd.Timestamp(ts[0])).days
        for k, s in S.items():
            r = evaluate(s, fwd)
            if r.get("n", 0) < 100:
                print(f"{k:<11}{r.get('n',0):>8}  표본 부족"); continue
            keep[k] = r
            print(f"{k:<11}{r['n']:>8,}{r['n']/days:>7.2f}{r['hit']:>8.4f}{r['bp']:>9.2f}"
                  f"{r['net_user']:>9.2f}{r['net_real']:>10.2f}")
        # 앙상블 — 합의 개수
        agree = np.sum([S[k] for k in keep], axis=0)
        for k_ in (2, 3):
            for lab, sg in ((f"합의 {k_}+ 롱", (agree >= k_).astype(float)),
                            (f"합의 {k_}+ 숏", -(agree <= -k_).astype(float))):
                r = evaluate(sg, fwd)
                if r.get("n", 0) >= 100:
                    print(f"{lab:<11}{r['n']:>8,}{r['n']/days:>7.2f}{r['hit']:>8.4f}{r['bp']:>9.2f}"
                          f"{r['net_user']:>9.2f}{r['net_real']:>10.2f}")
        r = evaluate(np.sign(agree), fwd)
        if r.get("n", 0) >= 100:
            print(f"{'합의 부호':<11}{r['n']:>8,}{r['n']/days:>7.2f}{r['hit']:>8.4f}{r['bp']:>9.2f}"
                  f"{r['net_user']:>9.2f}{r['net_real']:>10.2f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
