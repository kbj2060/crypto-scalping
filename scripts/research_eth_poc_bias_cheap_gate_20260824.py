#!/usr/bin/env python3
"""VPVR/POC 편향 cheap-gate (재량 룰북 §2.2 채우기용, 2026-08-24)

질문: 5m 재량 매매에서 "POC 대비 위/아래"를 방향 편향 필터로 쓸 때,
      부호는 순추세(위=롱)인가 회귀(위=숏)인가, 아니면 무정보인가?

방법: ETH 1h 봉 4.7년(2021-12~2026-02). 각 시점에서 직전 7일(168h) 볼륨프로파일의
      POC(0.25% 로그폭 bin, typical price 가중)를 계산하고, 종가>POC / 종가<POC 조건별
      전방 6h/24h/72h 로그수익 승률·평균을 비교. t-stat은 겹침 없는 서브샘플로 계산.

이건 경제성 게이트가 아니라 룰북에 넣을 부호 결정용 방향성 참고다.
결과: data/research/eth_poc_bias_cheap_gate_20260824.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
WINDOW_H = 168          # 7일 (청산맵 lookback과 정합)
BIN_PCT = 0.0025        # 0.25% (청산맵 bin폭과 정합)
HORIZONS_H = [6, 24, 72]


def load_5m() -> pd.DataFrame:
    a = pd.read_csv(REPO / "data" / "eth_5m_2021_2023_archive.csv")
    a["ts"] = pd.to_datetime(a["open_time"], unit="ms")
    b = pd.read_csv(REPO / "data" / "eth_5m_1year.csv")
    b["ts"] = pd.to_datetime(b["timestamp"])
    df = pd.concat([a, b], ignore_index=True)
    df = df.drop_duplicates(subset="ts").sort_values("ts").set_index("ts")
    return df[["high", "low", "close", "volume"]].astype(float)


def to_1h(df5: pd.DataFrame) -> pd.DataFrame:
    h = df5.resample("1h").agg(
        {"high": "max", "low": "min", "close": "last", "volume": "sum"})
    h = h.dropna()
    h["typical"] = (h["high"] + h["low"] + h["close"]) / 3.0
    return h


def rolling_poc(h: pd.DataFrame) -> np.ndarray:
    logp = np.log(h["typical"].to_numpy())
    vol = h["volume"].to_numpy()
    n = len(h)
    poc = np.full(n, np.nan)
    binw = np.log1p(BIN_PCT)
    for i in range(WINDOW_H, n):
        lp = logp[i - WINDOW_H:i]
        v = vol[i - WINDOW_H:i]
        lo = lp.min()
        idx = ((lp - lo) / binw).astype(int)
        counts = np.bincount(idx, weights=v)
        poc[i] = np.exp(lo + (counts.argmax() + 0.5) * binw)
    return poc


def tstat_nonoverlap(rets: np.ndarray, step: int) -> float:
    sub = rets[::step]
    if len(sub) < 3 or sub.std(ddof=1) == 0:
        return float("nan")
    return float(sub.mean() / (sub.std(ddof=1) / np.sqrt(len(sub))))


def main():
    h = to_1h(load_5m())
    poc = rolling_poc(h)
    close = h["close"].to_numpy()
    logc = np.log(close)

    out = {"window_h": WINDOW_H, "bin_pct": BIN_PCT, "n_bars_1h": len(h),
           "span": [str(h.index[0]), str(h.index[-1])], "results": {}}

    valid = ~np.isnan(poc)
    above = valid & (close > poc)
    below = valid & (close < poc)
    out["frac_above_poc"] = float(above.sum() / valid.sum())
    dist = np.abs(close[valid] / poc[valid] - 1.0)
    out["median_abs_dist_to_poc_pct"] = float(np.median(dist) * 100)

    for hz in HORIZONS_H:
        fwd = np.full(len(h), np.nan)
        fwd[:-hz] = logc[hz:] - logc[:-hz]
        ok = ~np.isnan(fwd)
        res = {}
        for name, mask in [("above_poc", above), ("below_poc", below),
                           ("all", valid)]:
            m = mask & ok
            r = fwd[m]
            res[name] = {
                "n": int(m.sum()),
                "win_rate": float((r > 0).mean()),
                "mean_ret_bp": float(r.mean() * 1e4),
                "t_nonoverlap": tstat_nonoverlap(r, hz),
            }
        out["results"][f"fwd_{hz}h"] = res

    dst = REPO / "data" / "research" / "eth_poc_bias_cheap_gate_20260824.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
