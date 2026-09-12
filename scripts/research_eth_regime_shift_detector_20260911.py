"""횡보→추세 전환 탐지 — 손익이 아니라 탐지 성능 자체를 잰다.

정답: 압축(volexp<0.7, 직전 6시간 내) → 확장(volexp>=1.8) 교차 = 전환 시점.
탐지 창: [전환-60분, 전환+15분] 안에 발동하면 적중(사건당 1회로 중복 제거).

지표: 탐지율(recall) · 정밀도 · 중앙 선행시간 · 시간당 오탐. 변화점 탐지의 표준 평가다.
탐지율만 높이는 건 임계값을 낮추면 자명하므로 **시간당 오탐을 고정하고 비교**한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
BOOK = ROOT / "data/research/eth_trend_signals_v1_screen_20260904/bookdepth_wide.parquet"
COMPRESS, EXPAND, BACK = 0.7, 1.8, 72
LEAD, LAG = 12, 3           # 탐지 창 = [-60분, +15분]


def main() -> int:
    d = pd.read_csv(D / "eth_5m_2026_tradecount.csv", parse_dates=["timestamp"])
    n = len(d)
    c, h, l = d.c.to_numpy(float), d.h.to_numpy(float), d.l.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    volexp = (pd.Series(lr).rolling(12).std() / pd.Series(lr).rolling(288).std()).to_numpy()
    atr = pd.Series(np.maximum.reduce([h - l, np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))])
                    ).rolling(96).mean().to_numpy() / np.maximum(c, 1e-9)
    bbw = pd.Series(lr).rolling(48).std().rolling(288).rank(pct=True).to_numpy()
    atrr = pd.Series(atr).rolling(288).rank(pct=True).to_numpy()
    cs = np.clip(1.0 - np.maximum(bbw, atrr), 0, 1)
    imp = np.r_[0, np.diff(c) / c[:-1]] / np.maximum(atr, 1e-9)
    pc = np.r_[0, cs[:-1]]
    release = np.clip(pc * np.abs(imp), 0, 3) / 3
    z = lambda s, w=288: ((s - s.rolling(w).mean()) / s.rolling(w).std()).to_numpy()

    cross = (volexp >= EXPAND) & np.r_[False, volexp[:-1] < EXPAND]
    was = pd.Series(volexp < COMPRESS).rolling(BACK).max().shift(12).to_numpy() == 1
    ev = np.flatnonzero(cross & was)
    ev = ev[(ev > 400) & (ev < n - 20)]
    # 감시 대상: 직전 12봉(1시간) 내에 압축이었던 봉 — 사건 봉과 그 직후가 빠지면 안 된다
    watch = (pd.Series(volexp < COMPRESS).rolling(12, min_periods=1).max().to_numpy() == 1) \
        & np.isfinite(volexp)
    hours = int(watch.sum()) * 5 / 60
    print(f"[정답] 전환 {len(ev)}건 · 감시 대상(압축) {int(watch.sum()):,}봉 = {hours:,.0f}시간")

    det = {
        "체결속도 z": z(d.n),
        "체결속도 z (3봉지속)": pd.Series(z(d.n)).rolling(3).min().to_numpy(),
        "거래대금 z": z(d.qv),
        "평균체결크기 z": z(d.avg_trade_size),
        "compression_release": release,
        "|수익률|/ATR (trivial)": np.abs(imp),
        "volexp 상승률": np.r_[0, np.diff(volexp)],
        "체결속도 OR 방출": None,
    }
    print(f"\n{'탐지기':24s} {'임계':>7s} {'탐지율':>7s} {'정밀도':>7s} {'중앙 선행':>9s} "
          f"{'시간당오탐':>9s} {'발동':>7s}")
    rows = []
    for nm, v in det.items():
        if v is None:
            zz = z(d.n)
            v = np.maximum(np.nan_to_num(zz / 2.0, nan=-9), np.nan_to_num(release * 5, nan=-9))
        qs = np.nanquantile(v[watch], [0.90, 0.95, 0.98, 0.995, 0.999])
        for thr in qs:
            fire = np.flatnonzero(watch & np.isfinite(v) & (v >= thr))
            if len(fire) < 10:
                continue
            hit_ev, leads, used = 0, [], np.zeros(len(fire), dtype=bool)
            for e in ev:
                k = np.flatnonzero((fire >= e - LEAD) & (fire <= e + LAG))
                if len(k):
                    hit_ev += 1
                    leads.append((fire[k[0]] - e) * 5)
                    used[k] = True
            prec = float(used.mean())
            fa_per_h = float((~used).sum()) / max(hours, 1)
            rows.append({"det": nm, "thr": thr, "recall": hit_ev / len(ev), "prec": prec,
                         "lead": float(np.median(leads)) if leads else np.nan,
                         "fa_h": fa_per_h, "fires": len(fire)})
            r = rows[-1]
            print(f"{nm:24s} {thr:7.2f} {r['recall']*100:6.1f}% {prec*100:6.1f}% "
                  f"{r['lead']:+8.0f}분 {fa_per_h:8.3f} {len(fire):7,d}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(D / "regime_shift_detector.csv", index=False)

    # 무작위 기준선: 같은 발동 수를 감시 봉에 무작위 배치
    print("\n=== 무작위 기준선 (같은 발동 수) ===")
    wi = np.flatnonzero(watch)
    for target_fa in (0.05, 0.10, 0.30):
        nf = int(target_fa * hours)
        rec = []
        for sd in (1, 2, 3, 4, 5):
            g = np.random.default_rng(sd)
            fire = np.sort(g.choice(wi, size=min(nf, len(wi)), replace=False))
            hit = sum(1 for e in ev if np.any((fire >= e - LEAD) & (fire <= e + LAG)))
            rec.append(hit / len(ev))
        print(f"  시간당 {target_fa:.2f}회 ({nf:,}발동)  무작위 탐지율 {np.median(rec)*100:5.1f}%")

    print("\n=== 오탐 빈도 고정 비교 (시간당 0.10회 근처에서) ===")
    print(f"{'탐지기':24s} {'임계':>7s} {'탐지율':>7s} {'정밀도':>7s} {'중앙 선행':>9s} {'시간당오탐':>9s}")
    for nm in det:
        q = df[df.det == nm].copy()
        if q.empty:
            continue
        q["gap"] = (q.fa_h - 0.10).abs()
        r = q.nsmallest(1, "gap").iloc[0]
        print(f"{nm:24s} {r.thr:7.2f} {r.recall*100:6.1f}% {r.prec*100:6.1f}% "
              f"{r.lead:+8.0f}분 {r.fa_h:8.3f}")
    print("\n탐지율만 높이는 건 임계값을 낮추면 자명하다 — 같은 오탐 빈도에서 비교해야 한다.")
    print("선행이 음수면 전환보다 먼저 발동(예측), 양수면 뒤(즉시 발견).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
