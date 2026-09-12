"""돌파(압축→확장) 감지기 — 체결 속도가 변동성 확장을 선행하는가.

사용자 실계좌 원장에서 확인된 손실 패턴: 압축 구간 상단에서 숏 페이드 → 상방 돌파 →
물타기 → -535.96 USDT(전체 손실의 95.7%). 진입 30분 후 체결속도 z 가 +6.6σ→+9.5σ 였고
그때 손실은 -0.28%(최종 -1.92%의 1/7)였다.

방향을 묻지 않는다 — "횡보 가정이 깨지고 있다"만 묻는다. 방어 신호라 요구 정밀도가 낮다.
⭐선행성이 전부다: 확장과 **동시**면 이미 늦어서 쓸모없다.
귀무는 순환이동(사건의 시간 군집 보존). 저장소 전례 2건이 기각됐으므로 대조군을 엄히 둔다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
COMPRESS, EXPAND, BACK = 0.7, 1.8, 72      # 압축 임계 · 확장 임계 · 압축 탐색 소급 봉수(6시간)
LAGS = [-12, -6, -3, -2, -1, 0, 1, 3, 6]
B_NULL, SEED = 400, 615372041


def _z(s, w=288):
    return ((s - s.rolling(w).mean()) / s.rolling(w).std()).to_numpy()


def main() -> int:
    d = pd.read_csv(D / "eth_5m_2026_tradecount.csv", parse_dates=["timestamp"])
    c = d.c.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    rv12 = pd.Series(lr).rolling(12).std()
    rv288 = pd.Series(lr).rolling(288).std()
    volexp = (rv12 / rv288).to_numpy()
    n = len(d)
    cand = {"체결속도 n": _z(d.n), "거래대금 qv": _z(d.qv),
            "평균체결크기": _z(d.avg_trade_size), "테이커비율": _z(d.taker_buy_ratio),
            "테이커 |쏠림|": _z((d.taker_buy_ratio - 0.5).abs()),
            "체결속도 1h평활": _z(d.n.rolling(12).mean())}

    cross = (volexp >= EXPAND) & np.r_[False, volexp[:-1] < EXPAND]
    was = pd.Series(volexp < COMPRESS).rolling(BACK).max().shift(12).to_numpy() == 1
    ev = np.flatnonzero(cross & was)
    ev = ev[(ev > 300) & (ev < n - 20)]
    print(f"[데이터] {n:,}봉 {d.timestamp.min()} ~ {d.timestamp.max()}")
    print(f"[사건] 압축(<{COMPRESS}, 직전 {BACK}봉 내) → 확장(>={EXPAND}) 교차  **{len(ev)}건** "
          f"(전체의 {len(ev)/n*100:.2f}%)", flush=True)
    if len(ev) < 30:
        print("사건 부족 — 임계값 재조정 필요")
        return 1

    rng = np.random.default_rng(SEED)
    shifts = rng.integers(300, n - 300, size=B_NULL)
    print(f"\n{'지표':16s} " + "".join(f"{f't{l:+d}':>9s}" for l in LAGS) + "   판정")
    rows = []
    for nm, z in cand.items():
        line, sig = f"{nm:16s}", []
        for lag in LAGS:
            k = np.clip(ev + lag, 0, n - 1)
            obs = float(np.nanmean(z[k]))
            null = np.array([np.nanmean(z[np.clip((ev + lag + s) % n, 0, n - 1)]) for s in shifts])
            p = float((null >= obs).mean())
            line += f"{obs:+8.2f}{'*' if p <= 0.01 else ('.' if p <= 0.05 else ' ')}"
            rows.append({"지표": nm, "lag": lag, "obs": obs, "p": p})
            if lag < 0 and p <= 0.01:
                sig.append(lag)
        lead = f"선행 {min(sig)}봉({min(sig)*5}분)" if sig else "선행 없음"
        print(line + f"   {lead}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(D / "breakout_detector_leadlag.csv", index=False)

    print(f"\n(* p<=0.01, . p<=0.05 · 순환이동 귀무 B={B_NULL})")
    print("\n=== 실용 검정: **압축 상태 봉으로 한정** (변동성 지속성 배제) ===")
    z = cand["체결속도 n"]
    zs = cand["체결속도 1h평활"]
    fut = pd.Series(volexp).rolling(12).max().shift(-12).to_numpy()
    compressed = volexp < COMPRESS
    for znm, zz in (("체결속도 n", z), ("체결속도 1h평활", zs)):
        uni = compressed & np.isfinite(zz) & np.isfinite(fut)
        base = float(np.mean(fut[uni] >= EXPAND))
        print(f"\n  [{znm}] 압축 봉 {int(uni.sum()):,}개 · 기저(앞 1시간 내 확장 도달) {base*100:.2f}%")
        print(f"  {'임계 z':>7s} {'발동':>7s} {'발동률':>7s} {'적중률':>7s} {'lift':>6s} "
              f"{'귀무중앙':>8s} {'귀무q95':>8s} {'p':>6s}")
        for thr in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            m = uni & (zz >= thr)
            if m.sum() < 50:
                continue
            hit = float(np.mean(fut[m] >= EXPAND))
            null = []
            for sh in shifts[:300]:
                mm = np.roll(zz, int(sh)) >= thr
                k = uni & mm
                if k.sum() >= 30:
                    null.append(float(np.mean(fut[k] >= EXPAND)))
            null = np.asarray(null)
            p = float((null >= hit).mean()) if len(null) else np.nan
            print(f"  {thr:7.1f} {int(m.sum()):7,d} {m.sum()/max(uni.sum(),1)*100:6.2f}% "
                  f"{hit*100:6.2f}% {hit/max(base,1e-9):5.2f}x {np.median(null)*100:7.2f}% "
                  f"{np.quantile(null,.95)*100:7.2f}% {p:6.3f}")
    print("\n선행이 없으면(모두 t+0 이후에만 유의) 감지기로 못 쓴다 — 이미 벌어진 뒤다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
