#!/usr/bin/env python3
"""대시보드 신호의 **엣지 크기(b) 축** 전수 스크린 (2026-09-10).

배경 항등식:  net = (2a-1) * b - cost
  a = 방향 적중률, b = 배리어/움직임 크기, cost = 왕복비용(테이커 10bp / 메이커 7.8bp).
이 저장소가 지금까지 판 축은 전부 **a**(더 좋은 모델/피쳐/라벨)였고 전부 닫혔다.
cost 는 거래소가 정한다. 남은 미검정 축은 **b** 하나다 -- b 는 호라이즌 H 와 ATR 에
비례해 커지는데 cost 는 고정이므로, 충분히 큰 b 에서는 같은 a 로도 부호가 뒤집힌다.

⚠️2026-09-09 스크린의 `gross` 는 **귀무 대비 초과가 아니다**. H=144 top 셀들이 27bp 를
   찍은 건 평가창이 하락장이었기 때문일 수 있다(같은 측면 무작위 숏도 +5bp -- 09-05 교훈).
   그래서 여기서는 **초과분**만 본다. 귀무 둘을 동시에 쓴다:
     · 순환이동(circular shift): 발동 간격/군집/개수를 그대로 보존한 채 가격 정렬만 파괴
       -> 겹침 표본의 자기상관을 올바로 반영한다(무작위 추출은 군집이 없어 CI 가 너무 좁다)
     · ATR 매칭 무작위: 같은 측면·같은 ATR 십분위에서 뽑는다 -> "고변동 구간이라 벌었다"를 뺀다
   판정은 둘 중 **불리한 쪽**으로 한다.

측정 축:
  1. H in {12,24,48,144,288,576}  (1h/2h/4h/12h/24h/48h)  -- b 를 H 로 키운다
  2. ATR 삼분위 (인과적: 발동 봉 시점까지의 확장 분위)   -- b 를 변동성으로 키운다
  3. 두 반기 안정성 (한 창짜리 우위는 버린다)

⚠️전건을 센다. 결과로 부분집합을 고르지 않는다(분류학 A).
"""
from __future__ import annotations
import argparse, json, os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402

OUT = ROOT / "tmp/eth_edge_scaling_20260910"
HS = (12, 24, 48, 144, 288, 576)
NSHIFT = 400
RNG = np.random.default_rng(20260910)
WARM = 900          # 지표 워밍업
ATR_WIN = 864       # 인과 ATR 분위 창 (기존 atr_percentile_864 와 같은 창)


def load_frame(start: str) -> pd.DataFrame:
    kl = B._load_kl(B.ETH_KL)
    kl = kl[kl["timestamp"] >= pd.Timestamp(start)].reset_index(drop=True)
    btc = B._load_kl(B.BTC_KL)
    fund = B._load_funding()
    tmax = kl["timestamp"].max()
    sig = compute_signals(kl, btc_df=btc[btc["timestamp"] <= tmax],
                          funding_df=fund[fund["calc_time"] <= tmax])
    return sig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-06-01")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sig = load_frame(a.start)
    ts = pd.to_datetime(sig["timestamp"].to_numpy())
    op = sig["open"].to_numpy(float); cl = sig["close"].to_numpy(float)
    hi = sig["high"].to_numpy(float); lo = sig["low"].to_numpy(float)
    n = len(sig)
    # 인과 ATR (Wilder 14) 과 그 확장 분위
    tr = np.maximum(hi[1:] - lo[1:], np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])))
    atr = np.full(n, np.nan); atr[1:] = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().to_numpy()
    atr_pct = atr / cl * 100.0
    atr_rank = pd.Series(atr_pct).rolling(ATR_WIN, min_periods=200).rank(pct=True).to_numpy()

    lo_i, hi_i = WARM, n - max(HS) - 2
    days = (ts[hi_i] - ts[lo_i]).total_seconds() / 86400
    print(f"평가 {ts[lo_i]} ~ {ts[hi_i]} UTC ({days:.0f}일, {hi_i-lo_i:,}봉)")
    print(f"ATR% 중앙 {np.nanmedian(atr_pct[lo_i:hi_i]):.3f}%\n")

    valid = np.zeros(n, bool); valid[lo_i:hi_i] = True
    valid &= ~np.isnan(atr_rank)
    pool = np.flatnonzero(valid)
    mid_t = ts[lo_i] + (ts[hi_i] - ts[lo_i]) / 2

    def ret_bp(idx, H, long):
        r = (cl[idx + H] - op[idx + 1]) / op[idx + 1] * 1e4
        return r if long else -r

    # ATR 십분위 풀 (귀무 B 용)
    dec = np.clip((atr_rank * 10).astype(float), 0, 9.999)
    dec_pool = {}
    for d in range(10):
        m = valid & (np.floor(dec) == d)
        dec_pool[d] = np.flatnonzero(m)

    rows = []
    for s in B.SIGNALS:
        for side in ("bottom", "top"):
            long = side == "bottom"
            fire = sig[f"{side}_{s}"].fillna(False).to_numpy(bool) & valid
            idx = np.flatnonzero(fire)
            if len(idx) < 100:
                continue
            fdec = np.floor(dec[idx]).astype(int)
            for H in HS:
                r = ret_bp(idx, H, long)
                g = float(np.mean(r))
                # 귀무 A: 순환이동 (군집 보존)
                span = hi_i - lo_i
                nullA = np.empty(NSHIFT)
                for b in range(NSHIFT):
                    sh = RNG.integers(500, span - 500)
                    j = lo_i + ((idx - lo_i + sh) % span)
                    nullA[b] = np.mean(ret_bp(j, H, long))
                # 귀무 B: 같은 측면·같은 ATR 십분위 무작위
                nullB = np.empty(NSHIFT)
                for b in range(NSHIFT):
                    j = np.concatenate([RNG.choice(dec_pool[d], size=int((fdec == d).sum()), replace=True)
                                        for d in range(10) if (fdec == d).sum() > 0])
                    nullB[b] = np.mean(ret_bp(j, H, long))
                exA = g - float(nullA.mean()); exB = g - float(nullB.mean())
                pA = float((nullA >= g).mean()); pB = float((nullB >= g).mean())
                # 두 반기
                first = ts[idx] < mid_t
                h1 = float(np.mean(r[first]) - np.mean([np.mean(ret_bp(lo_i + ((idx[first] - lo_i + RNG.integers(500, span-500)) % span), H, long)) for _ in range(60)]))
                h2 = float(np.mean(r[~first]) - np.mean([np.mean(ret_bp(lo_i + ((idx[~first] - lo_i + RNG.integers(500, span-500)) % span), H, long)) for _ in range(60)]))
                # ATR 삼분위
                terc = {}
                for name, m in (("저", fdec <= 2), ("중", (fdec >= 3) & (fdec <= 6)), ("고", fdec >= 7)):
                    if m.sum() < 40: terc[name] = (np.nan, np.nan, 0); continue
                    gg = float(np.mean(r[m]))
                    nb = np.mean([np.mean(ret_bp(np.concatenate(
                        [RNG.choice(dec_pool[d], size=int((fdec[m] == d).sum()), replace=True)
                         for d in range(10) if (fdec[m] == d).sum() > 0]), H, long)) for _ in range(60)])
                    terc[name] = (gg, gg - float(nb), int(m.sum()))
                rows.append(dict(signal=s, side=side, H=H, n=len(idx), per_day=len(idx)/days,
                                 gross=g, nullA=float(nullA.mean()), nullB=float(nullB.mean()),
                                 exA=exA, exB=exB, ex=min(exA, exB), pA=pA, pB=pB,
                                 h1=h1, h2=h2,
                                 lo_g=terc["저"][0], lo_ex=terc["저"][1], lo_n=terc["저"][2],
                                 mid_g=terc["중"][0], mid_ex=terc["중"][1], mid_n=terc["중"][2],
                                 hi_g=terc["고"][0], hi_ex=terc["고"][1], hi_n=terc["고"][2]))
            print(f"  {s:<26} {side:<6} 완료 ({len(idx):,}건)", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "edge_scaling.csv", index=False)
    print(f"\n저장 {OUT/'edge_scaling.csv'}  ({len(d)}행)")

    print("\n" + "=" * 118)
    print("초과분(ex = min(순환이동, ATR매칭) 대비) 상위 20 -- 비용 10bp 와 비교")
    print("=" * 118)
    top = d.sort_values("ex", ascending=False).head(20)
    print(top[["signal","side","H","n","per_day","gross","nullA","nullB","ex","pA","pB","h1","h2"]]
          .to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

    print("\n" + "=" * 118)
    print("H 스케일링: 셀별 초과분이 H 에 따라 커지는가 (같은 신호·측면 내)")
    print("=" * 118)
    piv = d.pivot_table(index=["signal","side"], columns="H", values="ex")
    print(piv.to_string(float_format=lambda v: f"{v:7.2f}"))

    print("\n" + "=" * 118)
    print("ATR 삼분위별 초과분 (H=48/144/288) -- b 를 변동성으로 키우면 넘는가")
    print("=" * 118)
    sub = d[d.H.isin((48,144,288))].sort_values(["signal","side","H"])
    print(sub[["signal","side","H","lo_n","lo_ex","mid_n","mid_ex","hi_n","hi_ex"]]
          .to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

    ok = d[(d.ex > 10) & (d.h1 > 0) & (d.h2 > 0)]
    print(f"\n★ 초과분 > 10bp 이고 두 반기 모두 양수인 셀: {len(ok)}개")
    if len(ok): print(ok[["signal","side","H","n","per_day","ex","h1","h2","pA","pB"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
