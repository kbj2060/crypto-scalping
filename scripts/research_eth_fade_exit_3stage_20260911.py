"""3단 청산 구조 — 경보(선행) → 축소 → 방향확정(즉시) 전량청산.

앞선 단일 규칙 결과(기준 -71.30%): 고정손절0.01 +27.14%p · 방출0.1 +32.32%p ·
체결속도2.0 +20.74%p · 무작위 +15.24%p. 3단이 단일 최고(+32.32%p)를 넘어야 의미가 있다.

1단  체결속도 z >= T1 이 D봉 지속(VolExpand 의 K 디바운스)  → 포지션 50% 축소
3단  compression_release_{반대} >= T3                      → 잔량 전량 청산
(2단 호가는 패널이 2026-03 에서 끊겨 이 구간에선 못 쓴다 — 별도 하위구간 검정)

최악 트레이드 방어가 사용자의 실제 요구다 — 총합보다 그 지표를 같이 본다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
COMPRESS, EXTREME, MAXHOLD = 0.7, 0.85, 72
C_ENTRY, C_EXIT = 0.000276, 0.000503
SEEDS = [615372041, 208844917, 933105268, 471926350, 862017594]


def main() -> int:
    d = pd.read_csv(D / "eth_5m_2026_tradecount.csv", parse_dates=["timestamp"])
    n = len(d)
    c, h, l = d.c.to_numpy(float), d.h.to_numpy(float), d.l.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    volexp = (pd.Series(lr).rolling(12).std() / pd.Series(lr).rolling(288).std()).to_numpy()
    nz = ((d.n - d.n.rolling(288).mean()) / d.n.rolling(288).std()).to_numpy()
    atr = pd.Series(np.maximum.reduce([h - l, np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))])
                    ).rolling(96).mean().to_numpy() / np.maximum(c, 1e-9)
    bbw = pd.Series(lr).rolling(48).std().rolling(288).rank(pct=True).to_numpy()
    atrr = pd.Series(atr).rolling(288).rank(pct=True).to_numpy()
    cs = np.clip(1.0 - np.maximum(bbw, atrr), 0, 1)
    imp = np.r_[0, np.diff(c) / c[:-1]] / np.maximum(atr, 1e-9)
    pc = np.r_[0, cs[:-1]]
    rel_up = np.clip(pc * np.clip(imp, 0, None), 0, 3) / 3
    rel_dn = np.clip(pc * np.clip(-imp, 0, None), 0, 3) / 3

    hi = pd.Series(c).rolling(48).max().to_numpy()
    lo = pd.Series(c).rolling(48).min().to_numpy()
    posn = (c - lo) / np.maximum(hi - lo, 1e-9)
    compd = volexp < COMPRESS
    se, le = compd & (posn >= EXTREME), compd & (posn <= 1 - EXTREME)
    ent = np.flatnonzero((se | le) & np.isfinite(volexp) & np.isfinite(nz))
    ent = ent[(ent > 400) & (ent < n - MAXHOLD - 2)]
    keep, last = [], -10**9
    for i in ent:
        if i - last >= MAXHOLD:
            keep.append(i); last = i
    ent = np.asarray(keep)
    side = np.where(se[ent], -1, 1)
    print(f"[합성 페이드] {len(ent)}건 (숏 {int((side<0).sum())} / 롱 {int((side>0).sum())})", flush=True)

    zrun = pd.Series(nz).rolling(1).min().to_numpy()   # placeholder, D별로 갱신

    def run3(T1, T3, Dbnc, cut, stop):
        """1단 z 지속 → cut 비율 축소, 3단 반대방출 → 전량. stop 은 하드스톱(None 가능)."""
        zd = pd.Series(nz).rolling(Dbnc).min().to_numpy() if Dbnc > 1 else nz
        out, fired1, fired3 = [], 0, 0
        for j, i in enumerate(ent):
            s, e = side[j], c[i]
            end = min(i + MAXHOLD, n - 1)
            rem, pnl, f1, f3 = 1.0, 0.0, False, False
            t = i + 1
            while t <= end:
                mv = (c[t] - e) / e * s
                if stop is not None and mv <= -stop:
                    pnl += rem * mv; rem = 0.0; break
                if (rel_up[t] if s < 0 else rel_dn[t]) >= T3:
                    pnl += rem * mv; rem = 0.0; f3 = True; break
                if (not f1) and np.isfinite(zd[t]) and zd[t] >= T1:
                    pnl += cut * mv; rem -= cut; f1 = True
                t += 1
            if rem > 0:
                pnl += rem * (c[min(t, end)] - e) / e * s
            fired1 += f1; fired3 += f3
            out.append(pnl * 100 - (C_ENTRY + C_EXIT) * 100)
        return np.asarray(out), fired1 / len(ent) * 100, fired3 / len(ent) * 100

    base, _, _ = run3(99, 99, 1, 0.0, None)
    print(f"[기준] 합계 {base.sum():+8.2f}%  승률 {(base>0).mean()*100:4.1f}%  "
          f"최악 {base.min():+6.2f}%  하위5%합 {np.sort(base)[:len(base)//20].sum():+7.2f}%\n")
    print(f"{'T1':>4s} {'T3':>5s} {'D':>2s} {'축소':>5s} {'스톱':>6s} {'1단%':>6s} {'3단%':>6s} "
          f"{'합계':>9s} {'승률':>6s} {'최악':>7s} {'하위5%':>8s} {'vs기준':>9s}")
    rows = []
    for T1 in (1.0, 1.5, 2.0):
        for T3 in (0.05, 0.10, 0.20):
            for Dbnc in (1, 3, 6):
                for stop in (None, 0.010):
                    r, f1, f3 = run3(T1, T3, Dbnc, 0.5, stop)
                    lo5 = np.sort(r)[:len(r) // 20].sum()
                    rows.append({"T1": T1, "T3": T3, "D": Dbnc, "stop": stop or 0,
                                 "f1": f1, "f3": f3, "sum": r.sum(), "wr": (r > 0).mean() * 100,
                                 "worst": r.min(), "lo5": lo5, "vs": r.sum() - base.sum()})
                    print(f"{T1:4.1f} {T3:5.2f} {Dbnc:2d} {0.5:5.0%} "
                          f"{('없음' if stop is None else f'{stop:.3f}'):>6s} {f1:5.1f}% {f3:5.1f}% "
                          f"{r.sum():+8.2f}% {(r>0).mean()*100:5.1f}% {r.min():+6.2f}% "
                          f"{lo5:+7.2f}% {r.sum()-base.sum():+8.2f}%p", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(D / "fade_exit_3stage.csv", index=False)
    print(f"\n[대조군] 고정손절0.01 +27.14%p · 방출0.1 +32.32%p · 체결속도2.0 +20.74%p · 무작위 +15.24%p")
    b = df.nlargest(1, "vs").iloc[0]
    w = df.nlargest(1, "worst").iloc[0]
    print(f"[3단 최고 총합] T1={b.T1} T3={b.T3} D={int(b.D)} 스톱={b.stop}  "
          f"{b['sum']:+.2f}% ({b.vs:+.2f}%p)  최악 {b.worst:+.2f}%")
    print(f"[3단 최고 방어] T1={w.T1} T3={w.T3} D={int(w.D)} 스톱={w.stop}  "
          f"{w['sum']:+.2f}% ({w.vs:+.2f}%p)  최악 {w.worst:+.2f}%")
    print(f"\n판정: 단일 최고(+32.32%p)를 넘는가, 그리고 최악을 고정손절(-4.26%)만큼 막는가.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
