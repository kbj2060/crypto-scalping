"""횡보 페이드 청산 규칙 비교 — 감지기 vs 그냥 손절.

사용자 전략: 압축(횡보) 구간의 국소 극단에서 페이드 진입, 추세로 바뀌면 큰 손실.
규칙: "돌파 방출이 **내 포지션 반대 방향**이면 즉시 청산." 돌파 시점의 방향은 예측이
아니라 관측이므로 방향 축(사전 예측)을 건드리지 않는다.

⭐핵심 대조군: **같은 발동 빈도의 고정 손절**. 역행시 자르는 규칙은 자명하게 손실을 줄이므로,
손절을 못 이기면 감지기는 손절의 복잡한 버전일 뿐이다. 무작위 청산도 같이 둔다.

실계좌 12건으로는 판정 불가 → 전략을 모사한 합성 페이드로 표본을 늘린다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
COMPRESS, EXTREME, MAXHOLD = 0.7, 0.85, 72     # 압축 컷 · 국소극단 분위 · 보유 상한 6시간
C_EXIT = 0.000503                               # 강제청산 taker 레그 5.03bp(실측)
C_ENTRY = 0.000276                              # peg maker 진입 2.76bp(실측)
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
    # compression_release (features/engineering.py:728-732 재현)
    bbw = pd.Series(lr).rolling(48).std().rolling(288).rank(pct=True).to_numpy()
    atrr = pd.Series(atr).rolling(288).rank(pct=True).to_numpy()
    comp_score = np.clip(1.0 - np.maximum(bbw, atrr), 0, 1)
    imp = np.r_[0, np.diff(c) / c[:-1]] / np.maximum(atr, 1e-9)
    pc = np.r_[0, comp_score[:-1]]
    rel_up = np.clip(pc * np.clip(imp, 0, None), 0, 3) / 3
    rel_dn = np.clip(pc * np.clip(-imp, 0, None), 0, 3) / 3

    # ── 합성 페이드 진입: 압축 구간의 국소 극단
    hi = pd.Series(c).rolling(48).max().to_numpy()
    lo = pd.Series(c).rolling(48).min().to_numpy()
    posn = (c - lo) / np.maximum(hi - lo, 1e-9)
    compd = volexp < COMPRESS
    short_e = compd & (posn >= EXTREME)
    long_e = compd & (posn <= 1 - EXTREME)
    ent = np.flatnonzero((short_e | long_e) & np.isfinite(volexp) & np.isfinite(nz))
    ent = ent[(ent > 400) & (ent < n - MAXHOLD - 2)]
    # 겹침 제거: 진입 후 MAXHOLD 동안 새 진입 금지
    keep, last = [], -10**9
    for i in ent:
        if i - last >= MAXHOLD:
            keep.append(i); last = i
    ent = np.asarray(keep)
    side = np.where(short_e[ent], -1, 1)
    print(f"[합성 페이드] 진입 {len(ent)}건 (숏 {int((side<0).sum())} / 롱 {int((side>0).sum())}) "
          f"· 압축<{COMPRESS} · 국소극단 {EXTREME:.0%} · 보유상한 {MAXHOLD}봉(6h)", flush=True)

    def run(rule, param):
        out = []
        for j, i in enumerate(ent):
            s = side[j]
            e = c[i]
            end = min(i + MAXHOLD, n - 1)
            k = None
            for t in range(i + 1, end + 1):
                mv = (c[t] - e) / e * s
                if rule == "hold":
                    pass
                elif rule == "stop" and mv <= -param:
                    k = t; break
                elif rule == "speed" and nz[t] >= param and mv < 0:
                    k = t; break
                elif rule == "release" and (rel_up[t] if s < 0 else rel_dn[t]) >= param:
                    k = t; break
                elif rule == "both" and ((nz[t] >= param[0] and mv < 0)
                                         or (rel_up[t] if s < 0 else rel_dn[t]) >= param[1]):
                    k = t; break
                elif rule == "random" and param[j] == t:
                    k = t; break
            t = k if k is not None else end
            out.append((c[t] - e) / e * s * 100 - (C_ENTRY + C_EXIT) * 100)
        return np.asarray(out)

    base = run("hold", None)
    print(f"\n[기준: 청산 규칙 없음(6시간 보유)]  합계 {base.sum():+8.2f}%  평균 {base.mean():+6.3f}%  "
          f"승률 {(base>0).mean()*100:4.1f}%  최악 {base.min():+6.2f}%  "
          f"하위5% 합 {np.sort(base)[:max(1,len(base)//20)].sum():+7.2f}%")
    print(f"\n{'규칙':22s} {'발동률':>7s} {'합계':>9s} {'평균':>8s} {'승률':>6s} {'최악':>7s} "
          f"{'하위5%합':>9s} {'vs 기준':>9s}")
    rows = []
    arms = ([("고정손절", "stop", p) for p in (0.004, 0.006, 0.010, 0.015)]
            + [("체결속도", "speed", p) for p in (1.0, 1.5, 2.0, 3.0)]
            + [("방출(반대)", "release", p) for p in (0.05, 0.10, 0.20, 0.35)]
            + [("속도+방출", "both", (1.5, 0.10)), ("속도+방출", "both", (2.0, 0.20))])
    for nm, rule, p in arms:
        r = run(rule, p)
        fired = float(np.mean(r != base) * 100)
        lo5 = np.sort(r)[:max(1, len(r) // 20)].sum()
        rows.append({"규칙": nm, "param": str(p), "fired": fired, "sum": r.sum(),
                     "mean": r.mean(), "wr": (r > 0).mean() * 100, "worst": r.min(), "lo5": lo5})
        print(f"{nm+' '+str(p):22s} {fired:6.1f}% {r.sum():+8.2f}% {r.mean():+7.3f}% "
              f"{(r>0).mean()*100:5.1f}% {r.min():+6.2f}% {lo5:+8.2f}% {r.sum()-base.sum():+8.2f}%p",
              flush=True)
    # 무작위 대조군: 발동률을 체결속도 z>=1.5 에 맞춘다
    tgt = [x for x in rows if x["규칙"] == "체결속도" and x["param"] == "1.5"][0]["fired"] / 100
    rr = []
    for sd in SEEDS:
        g = np.random.default_rng(sd)
        pick = np.where(g.random(len(ent)) < tgt,
                        ent + g.integers(1, MAXHOLD, size=len(ent)), -1)
        rr.append(run("random", pick).sum())
    print(f"{'무작위(빈도매칭)':22s} {tgt*100:6.1f}% {np.median(rr):+8.2f}% "
          f"{'':7s} {'':5s} {'':6s} {'':8s} {np.median(rr)-base.sum():+8.2f}%p  "
          f"[{min(rr):+.1f},{max(rr):+.1f}]")
    pd.DataFrame(rows).to_csv(D / "fade_exit_rules.csv", index=False)
    print("\n판정: 감지기가 **같은 발동률의 고정손절**을 이기는가. 못 이기면 손절의 복잡한 버전이다.")
    print(f"비용: 진입 peg {C_ENTRY*1e4:.2f}bp + 청산 taker {C_EXIT*1e4:.2f}bp (실측)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
