#!/usr/bin/env python3
"""XRP 증거신호·레짐이 **XRP에 맞게 튜닝됐는가** -- 상속 상수 진단.

## 왜

XRP 파이프라인은 BTC 포팅본을 재사용한다(str_z/taker/orthogonal은 BTC 모듈을 그대로 import).
그 과정에서 **격자로 탐색된 것**과 **상수로 물려받은 것**이 섞였다. 물려받은 상수는
"BTC/ETH에 맞춰진 값"이지 XRP 값이 아니다.

## 진단 3종 (전부 읽기 전용, 모델 학습 없음)

  A. **트리거 임계값 이식 적합성** -- demarker `dem<=0.10/>=0.90`, kalman `|z|>=2.0`은
     세 자산에 하드코딩돼 있다. 같은 절대 임계가 자산마다 **몇 %의 봉을 고르는지** 본다.
     발동률이 크게 다르면 임계가 그 자산의 분포에 안 맞는 것이다.
     같이 출력: 그 임계가 각 자산에서 몇 번째 백분위인지(= 백분위로 맞추려면 얼마여야 하는지).

  B. **교차자산 슬롯 선택** -- `FeatureEngineer`가 `close_btc`를 하드코딩하므로 XRP는 BTC를
     넣었는데, 이건 "BTC 원본이 ETH를 넣었으니 XRP는 BTC"라는 기계적 상속이다.
     XRP 수익률과 BTC/ETH 수익률의 상관을 여러 지연에서 비교해 어느 쪽이 더 정보적인지 본다.

  C. **CLUSTER_GAP 이식 적합성** -- XRP demarker/kalman은 `CLUSTER_GAP=6`을
     "fixed by task instruction"으로 받았다(ETH 자신의 선택은 demarker GAP=12).
     GAP을 바꾸면 앵커 수가 어떻게 변하는지, 그리고 **연속발동 클러스터 길이 분포**가
     자산마다 다른지 본다 -- 클러스터가 길수록 큰 GAP이 맞다.

⚠️읽기 전용 진단이다. 여기서 "안 맞는다"가 나오면 그 축을 실제로 격자탐색해야 한다.
⚠️C의 GAP은 오늘(2026-09-03) 앵커 미래참조 판정이 난 `cluster_dedup`의 인자다.
   경제성에는 어차피 못 쓰지만, **분류 학습 모집단**을 결정하므로 여전히 의미가 있다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

KL = {
    "ETH": ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
    "BTC": ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv",
    "XRP": ROOT / "binance_data/klines/XRPUSDT/XRPUSDT-5m-api.csv",
}
OUT = ROOT / "data/research/xrp_tuning_inherited_constants_20260903.json"

DEM_WINDOW = 14
DEM_LO, DEM_HI = 0.10, 0.90        # 세 자산 하드코딩
KAL_Z = 2.0                        # 세 자산 하드코딩 (라벨 K와 별개인 트리거 임계)
GAP_GRID = [3, 6, 9, 12, 18, 24]


def log(m): print(f"[tuning] {m}", flush=True)


def load(path):
    kl = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close"])
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True).dt.tz_localize(None)
    return kl.sort_values("timestamp").reset_index(drop=True)


def demarker(h, l, w=DEM_WINDOW):
    dh = np.diff(h, prepend=h[0]); dl = np.diff(l, prepend=l[0])
    demax = np.where(dh > 0, dh, 0.0)
    demin = np.where(dl < 0, -dl, 0.0)
    a = pd.Series(demax).rolling(w).mean()
    b = pd.Series(demin).rolling(w).mean()
    return (a / (a + b)).to_numpy()


def kalman_dev_z(close, q=1e-5, r=1e-2, zw=288):
    n = len(close)
    x = np.empty(n); P = 1.0; xh = close[0]
    for i in range(n):
        P += q
        K = P / (P + r)
        xh = xh + K * (close[i] - xh)
        P = (1 - K) * P
        x[i] = xh
    dev = (close - x) / np.where(x == 0, np.nan, x)
    s = pd.Series(dev)
    return ((s - s.rolling(zw).mean()) / s.rolling(zw).std()).to_numpy()


def cluster_lengths(idx, gap):
    """`cluster_dedup`과 동일한 경계(diff > gap)로 묶은 클러스터의 길이 분포."""
    idx = np.sort(np.asarray(idx))
    if len(idx) == 0:
        return np.array([])
    brk = np.flatnonzero(np.diff(idx) > gap) + 1
    return np.array([len(g) for g in np.split(idx, brk)])


def main() -> int:
    rep = {"dem_thresholds": [DEM_LO, DEM_HI], "kalman_z": KAL_Z,
           "gap_grid": GAP_GRID, "assets": {}}
    frames, dems, kzs = {}, {}, {}

    log("=" * 74)
    log("A. 트리거 임계값 이식 적합성 -- 같은 절대 임계가 자산마다 몇 %를 고르나")
    log("=" * 74)
    log(f"{'자산':<5} {'dem<=0.10':>11} {'dem>=0.90':>11} {'|kal_z|>=2':>11}   "
        f"{'dem 10%분위':>11} {'dem 90%분위':>11}  {'kal z 97.7%':>11}")
    for a, path in KL.items():
        kl = load(path); frames[a] = kl
        d = demarker(kl["high"].to_numpy(), kl["low"].to_numpy()); dems[a] = d
        z = kalman_dev_z(kl["close"].to_numpy()); kzs[a] = z
        dv = d[np.isfinite(d)]; zv = z[np.isfinite(z)]
        lo_rate = float((dv <= DEM_LO).mean())
        hi_rate = float((dv >= DEM_HI).mean())
        kz_rate = float((np.abs(zv) >= KAL_Z).mean())
        # 백분위로 맞추려면 임계가 얼마여야 하나 (ETH 발동률 기준은 아래에서 별도)
        q10, q90 = float(np.quantile(dv, 0.10)), float(np.quantile(dv, 0.90))
        kq = float(np.quantile(np.abs(zv), 0.977))
        log(f"{a:<5} {lo_rate*100:>10.2f}% {hi_rate*100:>10.2f}% {kz_rate*100:>10.2f}%   "
            f"{q10:>11.4f} {q90:>11.4f}  {kq:>11.3f}")
        rep["assets"][a] = {"dem_lo_rate": lo_rate, "dem_hi_rate": hi_rate,
                            "kal_rate": kz_rate, "dem_q10": q10, "dem_q90": q90,
                            "kal_q977": kq, "n_bars": int(len(kl))}

    e = rep["assets"]["ETH"]
    log("")
    log("  ETH 대비 발동률 배수 (1.0에서 멀수록 임계가 그 자산 분포에 안 맞음):")
    for a in KL:
        v = rep["assets"][a]
        log(f"    {a:<5} dem_bottom {v['dem_lo_rate']/e['dem_lo_rate']:>5.2f}x  "
            f"dem_top {v['dem_hi_rate']/e['dem_hi_rate']:>5.2f}x  "
            f"kalman {v['kal_rate']/e['kal_rate']:>5.2f}x")
        rep["assets"][a]["ratio_vs_eth"] = {
            "dem_lo": v["dem_lo_rate"] / e["dem_lo_rate"],
            "dem_hi": v["dem_hi_rate"] / e["dem_hi_rate"],
            "kal": v["kal_rate"] / e["kal_rate"]}

    log("")
    log("=" * 74)
    log("B. 교차자산 슬롯 -- XRP의 파트너로 BTC와 ETH 중 어느 쪽이 더 정보적인가")
    log("=" * 74)
    m = frames["XRP"][["timestamp", "close"]].rename(columns={"close": "xrp"})
    for a in ("BTC", "ETH"):
        m = m.merge(frames[a][["timestamp", "close"]].rename(columns={"close": a.lower()}),
                    on="timestamp", how="inner")
    log(f"  공통 봉 {len(m):,}개")
    r = m[["xrp", "btc", "eth"]].pct_change().dropna()
    cross = {}
    for lag in (0, 1, 2, 3, 6, 12):
        cb = float(r["xrp"].corr(r["btc"].shift(lag)))
        ce = float(r["xrp"].corr(r["eth"].shift(lag)))
        cross[f"lag{lag}"] = {"btc": cb, "eth": ce, "winner": "BTC" if abs(cb) > abs(ce) else "ETH"}
        log(f"  lag {lag:>2}봉  XRP~BTC {cb:+.4f}   XRP~ETH {ce:+.4f}   "
            f"→ {'BTC' if abs(cb) > abs(ce) else 'ETH'}")
    nb = sum(1 for v in cross.values() if v["winner"] == "BTC")
    log(f"  ⇒ 6개 지연 중 BTC 우세 {nb} / ETH 우세 {6-nb}")
    rep["cross_asset"] = {"n_common_bars": int(len(m)), "by_lag": cross, "btc_wins": nb}

    log("")
    log("=" * 74)
    log("C. CLUSTER_GAP -- 연속발동 클러스터 길이가 자산마다 다른가")
    log("=" * 74)
    gapres = {}
    for a in KL:
        d = dems[a]
        bot = np.flatnonzero(np.isfinite(d) & (d <= DEM_LO))
        gapres[a] = {}
        row = []
        for g in GAP_GRID:
            cl = cluster_lengths(bot, g)
            gapres[a][g] = {"n_clusters": int(len(cl)), "mean_len": float(cl.mean()),
                            "p90_len": float(np.percentile(cl, 90)),
                            "keep_ratio": float(len(cl) / len(bot))}
            row.append(f"G{g}: {len(cl):>5} ({len(cl)/len(bot)*100:>4.1f}%)")
        log(f"  {a} demarker bottom 원발동 {len(bot):,} → 앵커수: " + "  ".join(row))
    log("")
    log("  GAP=6일 때 클러스터 평균/90분위 길이:")
    for a in KL:
        v = gapres[a][6]
        log(f"    {a:<5} 평균 {v['mean_len']:.2f}봉  90분위 {v['p90_len']:.0f}봉  "
            f"유지율 {v['keep_ratio']*100:.1f}%")
    rep["cluster_gap"] = gapres

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
