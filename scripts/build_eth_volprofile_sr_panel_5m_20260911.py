#!/usr/bin/env python3
"""**거래량 프로파일 지지/저항 패널** -- 5분봉마다, 청산맵과 같은 잣대 (2026-09-11).

사용자: *"지지·저항은 호가창이나 체결속도 같은 걸로 알 수 있을 것 같은데."*
호가는 이 저장소에 **가격 단위 역사가 없다**(bookDepth 는 ±0.2~5% 누적 밴드, raw depth20 은
2026-08-26~27 이틀). 체결로 «어느 가격에서» 를 만들 수 있는 형태가 **거래량 프로파일**이다:
1분봉 1,357,436개(2023-12~)를 가격빈에 쌓는다. 1분봉 고저폭 중앙 0.082% < 빈 폭 0.1% 라
«한 봉 = 한 빈» 근사가 성립한다.

## 청산맵과 같은 잣대로 맞춘 것 (그래야 «출처를 바꾸면 나아지나»를 답할 수 있다)
룩백 168h · 반감기 240h · 빈 폭 0.1% · 측면당 최대 6개 · 최대거리 5% · 최소지분 5% --
전부 `live_liquidation_map_20260824` 의 상수를 **import 해서** 쓴다. 바뀌는 건 **빈의 무게**뿐:
    청산맵 = 레버리지 티어로 역산한 가상 청산가의 부피
    여기   = 그 가격대에서 **실제로 체결된 양**(vol) 또는 **체결 건수**(cnt, ≈ 체결속도)

## 인과성
1분봉은 그 5분봉이 시작하기 전에 **이미 닫힌 것만**(`T+1m <= t_i`), 기준가는 그 봉 **종가**.
빈은 로그가격 격자(`floor(log(p)/log(1.001))`)라 기준가에 안 딸린다 -- 굴림 갱신이 가능하고,
청산맵처럼 봉마다 재비닝하지 않아도 폭이 0.1% 로 일정하다.
"""
from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT if (ROOT / "binance_data").exists() else Path(subprocess.run(
    ["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
    capture_output=True, text=True).stdout.strip()).parent
sys.path.insert(0, str(ROOT / "scripts"))
from live_liquidation_map_20260824 import (  # noqa: E402
    BIN_WIDTH_PCT, LOOKBACK_HOURS, MAX_LEVEL_DISTANCE_PCT, MAX_LEVELS_PER_SIDE,
    MIN_LEVEL_SHARE, RECENCY_HALFLIFE_HOURS,
)

K1 = DATA / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
K5 = DATA / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT = DATA / "tmp/eth_volprofile_sr_panel_20260911"
KEEP = 3
LOGB = np.log1p(BIN_WIDTH_PCT)          # 로그가격 격자 한 칸 = 0.1%


PEAK_HALF = 5      # 국소 봉우리 판정 반경(빈) -- +-0.5%. HVN 은 이 폭 안에서 최댓값이어야 한다.


def levels(acc: dict, cp: float, keep: int, peak: bool = False):
    """청산맵 `levels_from_bins` 와 같은 규칙 -- 최소지분·최대거리·측면당 상한·가까운 순.

    `peak=True` 면 **국소 봉우리(HVN)만** 남긴다. 이게 없으면 «전역 최대의 5% 이상» 조건이
    현재가 주변의 연속 덩어리를 통째로 통과시켜 레벨이 전부 가격에 달라붙는다(실측 중앙 0.13%).
    실제 거래량 프로파일이 지지/저항으로 쓰는 것은 그 덩어리가 아니라 봉우리다.
    """
    if not acc:
        return [], [], 0, 0
    mx = max(acc.values())
    if not (mx > 0):
        return [], [], 0, 0
    sup, res = [], []
    lo, hi = cp * (1 - MAX_LEVEL_DISTANCE_PCT), cp * (1 + MAX_LEVEL_DISTANCE_PCT)
    for b, w in acc.items():
        if w / mx < MIN_LEVEL_SHARE:
            continue
        if peak and any(acc.get(b + k, 0.0) > w for k in range(-PEAK_HALF, PEAK_HALF + 1) if k):
            continue
        px = float(np.exp((b + 0.5) * LOGB))
        if px < lo or px > hi:
            continue
        (sup if px < cp else res).append((px, w / mx))
    sup.sort(key=lambda x: -x[1]); res.sort(key=lambda x: -x[1])
    sup, res = sup[:MAX_LEVELS_PER_SIDE], res[:MAX_LEVELS_PER_SIDE]
    n_s, n_r = len(sup), len(res)
    sup.sort(key=lambda x: -x[0]); res.sort(key=lambda x: x[0])   # 가까운 순
    return sup[:keep], res[:keep], n_s, n_r


def main() -> int:
    m1 = (pd.read_csv(K1, usecols=["timestamp", "high", "low", "close", "quote_volume", "trades"],
                      parse_dates=["timestamp"])
          .sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    k5 = (pd.read_csv(K5, usecols=["timestamp", "close"], parse_dates=["timestamp"])
          .sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    k5 = k5[(k5.timestamp >= m1.timestamp.min() + pd.Timedelta(hours=LOOKBACK_HOURS))
            & (k5.timestamp <= m1.timestamp.max())].reset_index(drop=True)

    tp = ((m1.high + m1.low + m1.close) / 3.0).to_numpy(float)
    bins = np.floor(np.log(tp) / LOGB).astype(np.int64)
    t1 = m1.timestamp.to_numpy()
    hrs = (t1 - t1[0]) / np.timedelta64(1, "h")
    # 지수감쇠는 **덧셈적**이다: w*exp(+t/HL) 로 쌓고 질의 때 exp(-t/HL) 로 되돌린다.
    #   -> 창이 굴러도 재계산 없이 더하고 빼기만 하면 된다(2.6년에서 지수 최대 ~95, float64 안전).
    up = np.exp(hrs / RECENCY_HALFLIFE_HOURS)
    W = {"vol": m1.quote_volume.to_numpy(float) * up, "cnt": m1.trades.to_numpy(float) * up}

    t5 = k5.timestamp.to_numpy(); cl5 = k5.close.to_numpy(float)
    win = np.timedelta64(LOOKBACK_HOURS, "h")
    for tag in ("vol", "cnt", "volpeak", "cntpeak"):
        acc: dict[int, float] = defaultdict(float)
        peak = tag.endswith("peak")
        w = W[tag[:3]]
        add = drop = 0
        rows = []
        for i in range(len(t5)):
            while add < len(t1) and t1[add] < t5[i]:            # 이미 닫힌 1분봉만
                acc[bins[add]] += w[add]; add += 1
            while drop < add and t1[drop] < t5[i] - win:
                acc[bins[drop]] -= w[drop]
                if acc[bins[drop]] <= 0:
                    acc.pop(bins[drop], None)
                drop += 1
            if add - drop < 1000:
                rows.append(None); continue
            sup, res, ns, nr = levels(acc, cl5[i], KEEP, peak)
            r = {"n_sup": ns, "n_res": nr}
            for j in range(KEEP):
                r[f"s{j+1}_px"], r[f"s{j+1}_w"] = (sup[j] if j < len(sup) else (np.nan, np.nan))
                r[f"r{j+1}_px"], r[f"r{j+1}_w"] = (res[j] if j < len(res) else (np.nan, np.nan))
            rows.append(r)
            if i % 100000 == 0:
                print(f"  [{tag}] {i:,}/{len(t5):,}", flush=True)
        keepmask = [r is not None for r in rows]
        out = pd.DataFrame([r for r in rows if r is not None])
        out.insert(0, "close", cl5[keepmask]); out.insert(0, "timestamp", t5[keepmask])
        OUT.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT / f"sr_panel_5m_{tag}.parquet")
        both = out.s1_px.notna() & out.r1_px.notna()
        print(f"[{tag}] {len(out):,}행 · {out.timestamp.min()} ~ {out.timestamp.max()} · "
              f"양쪽 레벨 {both.mean():.3f} · 저항거리중앙 "
              f"{np.nanmedian((out.r1_px - out.close) / out.close) * 100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
