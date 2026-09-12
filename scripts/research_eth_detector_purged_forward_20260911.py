"""탐지기 재검정 — 앞만 보는 타깃 + 피쳐 중복도.

앞선 검정의 결함: 타깃 volexp(t) 가 **뒤 12봉**을 보는 창이라, t-5봉의 급등이 t 시점
volexp 에 이미 들어간다. "예측"의 상당 부분이 '타깃에 곧 들어갈 것이 이미 일어났다'일 수 있다.

여기서는 타깃을 **앞만 보게** 바꾼다: fwd_rv(t) = std(수익률[t+1 : t+12]) / 장기변동성(t).
탐지 시점 t 의 정보와 타깃 구간이 **겹치지 않는다**. 그리고 후보 피쳐 간 상관도 같이 낸다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
COMPRESS, EXPAND = 0.7, 1.8
B_NULL, SEED = 300, 615372041


def main() -> int:
    d = pd.read_csv(D / "eth_5m_2026_tradecount.csv", parse_dates=["timestamp"])
    n = len(d)
    c, h, l = d.c.to_numpy(float), d.h.to_numpy(float), d.l.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    rv12b = pd.Series(lr).rolling(12).std().to_numpy()          # 뒤 1시간 (기존 타깃 재료)
    rv288 = pd.Series(lr).rolling(288).std().to_numpy()
    volexp_back = rv12b / rv288
    # 앞만 보는 타깃: [t+1, t+12] 의 실현변동성 / t 시점 장기변동성
    rv12f = pd.Series(lr).rolling(12).std().shift(-12).to_numpy()
    volexp_fwd = rv12f / rv288
    atr = pd.Series(np.maximum.reduce([h - l, np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))])
                    ).rolling(96).mean().to_numpy() / np.maximum(c, 1e-9)
    bbw = pd.Series(lr).rolling(48).std().rolling(288).rank(pct=True).to_numpy()
    atrr = pd.Series(atr).rolling(288).rank(pct=True).to_numpy()
    cs = np.clip(1.0 - np.maximum(bbw, atrr), 0, 1)
    imp = np.r_[0, np.diff(c) / c[:-1]] / np.maximum(atr, 1e-9)
    release = np.clip(np.r_[0, cs[:-1]] * np.abs(imp), 0, 3) / 3
    z = lambda s, w=288: ((s - s.rolling(w).mean()) / s.rolling(w).std()).to_numpy()
    tb = d.taker_buy_ratio
    F = {"체결속도 n": z(d.n), "거래대금 qv": z(d.qv), "평균체결크기": z(d.avg_trade_size),
         "테이커 |쏠림|": z((tb - 0.5).abs()), "compression_release": release,
         "|수익률|/ATR": np.abs(imp), "volexp 상승률": np.r_[0, np.diff(volexp_back)],
         "체결속도 3봉지속": pd.Series(z(d.n)).rolling(3).min().to_numpy()}

    print("=== 피쳐 간 상관 (스피어만, 압축 봉) ===")
    comp = (volexp_back < COMPRESS) & np.isfinite(volexp_back)
    M = pd.DataFrame({k: v for k, v in F.items()})[comp].corr(method="spearman")
    print(M.round(2).to_string())

    print(f"\n=== 타깃 두 종류 비교 ===")
    for tname, tgt in (("뒤를 보는 창 (기존)", volexp_back), ("앞만 보는 창 (수정)", volexp_fwd)):
        v = comp & np.isfinite(tgt)
        # t 시점 이후 1시간 안에 확장에 도달하는가
        if tname.startswith("뒤"):
            fut = pd.Series(tgt).rolling(12).max().shift(-12).to_numpy()
        else:
            fut = tgt                                    # 이미 앞 1시간이다
        vv = v & np.isfinite(fut)
        base = float(np.mean(fut[vv] >= EXPAND))
        rng = np.random.default_rng(SEED)
        shifts = rng.integers(300, n - 300, size=B_NULL)
        print(f"\n[{tname}]  압축 {int(vv.sum()):,}봉 · 기저 확장률 {base*100:.2f}%")
        print(f"  {'피쳐':18s} {'임계(q99)':>9s} {'발동':>7s} {'적중률':>7s} {'lift':>6s} "
              f"{'귀무q95':>8s} {'p':>6s}")
        for nm, x in F.items():
            thr = float(np.nanquantile(x[vv], 0.99))
            m = vv & np.isfinite(x) & (x >= thr)
            if m.sum() < 50:
                continue
            hit = float(np.mean(fut[m] >= EXPAND))
            null = []
            for s in shifts[:200]:
                mm = vv & (np.roll(x, int(s)) >= thr)
                if mm.sum() >= 30:
                    null.append(float(np.mean(fut[mm] >= EXPAND)))
            null = np.asarray(null)
            p = float((null >= hit).mean()) if len(null) else np.nan
            print(f"  {nm:18s} {thr:9.3f} {int(m.sum()):7,d} {hit*100:6.2f}% "
                  f"{hit/max(base,1e-9):5.2f}x {np.quantile(null,.95)*100:7.2f}% {p:6.3f}", flush=True)
    print("\n앞만 보는 창에서 lift 가 무너지면, 기존 결과는 창 겹침이 만든 순환이었다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
