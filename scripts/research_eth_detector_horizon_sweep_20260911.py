"""피쳐별 최적 지평 탐색 — 타깃 지평 × 피쳐 정규화 창.

1시간은 임의 고정이었다. 그리고 지평을 늘리면 기저율이 2.23%→80.14% 로 뛰어 lift 가 자동으로
1에 붙는다. 그래서 타깃을 **자기 분포 상위 5%**로 정의해 기저를 지평 무관하게 5% 로 고정한다.
타깃은 앞만 본다(순환 배제): fwd_rv(H) = std(수익률[t+1 : t+H]).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
COMPRESS, TOPQ, FIREQ = 0.7, 0.95, 0.99
HS = [(2, "10분"), (3, "15분"), (4, "20분"), (6, "30분"), (9, "45분"),
      (12, "1시간"), (24, "2시간"), (48, "4시간")]
WS = [48, 96, 288, 864]
B_NULL, SEED = 120, 615372041


def main() -> int:
    d = pd.read_csv(D / "eth_5m_2026_tradecount.csv", parse_dates=["timestamp"])
    n = len(d)
    c, h, l = d.c.to_numpy(float), d.h.to_numpy(float), d.l.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    rv288 = pd.Series(lr).rolling(288).std().to_numpy()
    volexp_back = pd.Series(lr).rolling(12).std().to_numpy() / rv288
    atr = pd.Series(np.maximum.reduce([h - l, np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))])
                    ).rolling(96).mean().to_numpy() / np.maximum(c, 1e-9)
    bbw = pd.Series(lr).rolling(48).std().rolling(288).rank(pct=True).to_numpy()
    atrr = pd.Series(atr).rolling(288).rank(pct=True).to_numpy()
    cs = np.clip(1.0 - np.maximum(bbw, atrr), 0, 1)
    imp = np.r_[0, np.diff(c) / c[:-1]] / np.maximum(atr, 1e-9)
    rel = np.clip(np.r_[0, cs[:-1]] * np.abs(imp), 0, 3) / 3
    tb = d.taker_buy_ratio
    RAW = {"체결속도 n": d.n, "거래대금 qv": d.qv, "평균체결크기": d.avg_trade_size,
           "테이커 |쏠림|": (tb - 0.5).abs()}
    FIXED = {"compression_release": rel, "|수익률|/ATR": np.abs(imp),
             "volexp 상승률": np.r_[0, np.diff(volexp_back)]}
    comp = (volexp_back < COMPRESS) & np.isfinite(volexp_back)
    rng = np.random.default_rng(SEED)
    shifts = rng.integers(900, n - 900, size=B_NULL)
    print(f"[설정] 압축 봉 {int(comp.sum()):,} · 타깃=앞 H봉 실현변동성 상위 {1-TOPQ:.0%} "
          f"(기저 지평 무관 고정) · 발동 컷 q{FIREQ:.0%} · 귀무 B={B_NULL}\n")

    rows = []
    for H, hn in HS:
        fwd = pd.Series(lr).rolling(H).std().shift(-H).to_numpy()
        v0 = comp & np.isfinite(fwd)
        tgt = fwd >= np.nanquantile(fwd[v0], TOPQ)
        base = float(np.mean(tgt[v0]))
        cands = {}
        for nm, s in RAW.items():
            for W in WS:
                cands[f"{nm} (z{W})"] = ((s - s.rolling(W).mean()) / s.rolling(W).std()).to_numpy()
        cands.update(FIXED)
        for nm, x in cands.items():
            vv = v0 & np.isfinite(x)
            if vv.sum() < 2000:
                continue
            thr = float(np.nanquantile(x[vv], FIREQ))
            m = vv & (x >= thr)
            if m.sum() < 50:
                continue
            hit = float(np.mean(tgt[m]))
            null = []
            for s2 in shifts:
                mm = vv & (np.roll(x, int(s2)) >= thr)
                if mm.sum() >= 30:
                    null.append(float(np.mean(tgt[mm])))
            null = np.asarray(null)
            rows.append({"H": H, "hn": hn, "feat": nm, "base": base, "hit": hit,
                         "lift": hit / max(base, 1e-9),
                         "p": float((null >= hit).mean()) if len(null) else np.nan,
                         "nfire": int(m.sum())})
        print(f"  {hn} 완료 (기저 {base*100:.2f}%)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(D / "detector_horizon_sweep.csv", index=False)

    print(f"\n=== 피쳐별 최적 셀 (lift 최대, p<=0.05 만) ===")
    print(f"{'피쳐':22s} {'최적 지평':>8s} {'적중률':>7s} {'lift':>6s} {'p':>6s} {'발동':>6s}")
    df["base_name"] = df.feat.str.replace(r" \(z\d+\)", "", regex=True)
    for bn in ["체결속도 n", "거래대금 qv", "평균체결크기", "테이커 |쏠림|",
               "compression_release", "|수익률|/ATR", "volexp 상승률"]:
        q = df[(df.base_name == bn) & (df.p <= 0.05)]
        if q.empty:
            q2 = df[df.base_name == bn]
            b = q2.nlargest(1, "lift").iloc[0] if not q2.empty else None
            if b is not None:
                print(f"{bn:22s} {'—':>8s} {b.hit*100:6.2f}% {b.lift:5.2f}x {b.p:6.3f} "
                      f"{int(b.nfire):6d}   유의 셀 없음 (최고 {b.hn}/{b.feat})")
            continue
        b = q.nlargest(1, "lift").iloc[0]
        print(f"{b.feat:22s} {b.hn:>8s} {b.hit*100:6.2f}% {b.lift:5.2f}x {b.p:6.3f} {int(b.nfire):6d}")

    print(f"\n=== 지평별 최고 피쳐 ===")
    print(f"{'지평':>8s} {'기저':>7s} {'최고 피쳐':24s} {'적중률':>7s} {'lift':>6s} {'p':>6s}")
    for H, hn in HS:
        q = df[(df.H == H) & (df.p <= 0.05)]
        if q.empty:
            print(f"{hn:>8s} {'':>7s} {'유의 셀 없음':24s}")
            continue
        b = q.nlargest(1, "lift").iloc[0]
        print(f"{hn:>8s} {b.base*100:6.2f}% {b.feat:24s} {b.hit*100:6.2f}% {b.lift:5.2f}x {b.p:6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
