"""**테이커 흐름 불균형의 유효 지평** — 5분에서 왜 안 되는지를 곡선으로 (2026-09-14).

문헌 배경:
  · Cont·Kukanov·Stoikov(2013, J.Fin.Econometrics) — 단기 가격변화는 **주문흐름 불균형(OFI)** 이
    거의 선형으로 설명하고, 기울기는 **깊이에 반비례**한다.
  · Kolm·Turiel·Westray(2023, Mathematical Finance) — 최고 해상도 호가로 딥러닝을 해도
    **유효 지평은 «평균 가격변화 두 번»**(틱 몇 개). 그리고 **호가 «수준»이 아니라 «흐름»** 으로
    학습해야 한다(정상성).
우리는 이벤트 단위 L2 가 없으므로 OFI 를 못 만든다. 가장 가까운 대리물은 klines 의
**테이커 매수 비중**(taker_buy_base / volume)이다. 이걸로 **감쇠 곡선**을 그린다:

  IC(H) 가 지평 H 와 함께 어떻게 죽는가  vs  손익분기 IC(H) = 비용 / (E|r_H| · 0.798)

두 곡선이 만나는 H 가 있으면 거기가 이 신호의 자리이고, 없으면 **원리적으로 닫힌 것**이다.
🔴동시점(같은 봉) 상관도 같이 낸다 — 그게 Cont 의 효과가 우리 데이터에 **있기는 한가**의 확인이다.
   있는데 앞으로만 죽으면 「정보는 실재하되 이미 반영됨」이고, 애초에 없으면 데이터가 거친 것이다.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
KL = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
OUT = ROOT / "data/research/eth_direction_barrier_label_20260914"
HS = (1, 2, 3, 5, 10, 15, 30, 60, 120, 240)
ROUND_TRIP_BP = 5.88          # 배포 실측(지정가 진입 + peg 청산)


def block_t(x: np.ndarray, day: np.ndarray) -> float:
    """일 군집 t — 겹치는 지평에서 봉 단위 t 는 부풀려진다."""
    s = pd.Series(x).groupby(day).mean().to_numpy()
    s = s[np.isfinite(s)]
    if len(s) < 3 or s.std(ddof=1) == 0:
        return float("nan")
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="taker_flow_decay")
    a = ap.parse_args()
    d = pd.read_csv(KL, usecols=["timestamp", "close", "volume", "quote_volume", "trades",
                                 "taker_buy_base"], parse_dates=["timestamp"])
    d = d.sort_values("timestamp").reset_index(drop=True)
    c = d.close.to_numpy(float); v = np.maximum(d.volume.to_numpy(float), 1e-9)
    tb = d.taker_buy_base.to_numpy(float)
    lc = np.log(np.maximum(c, 1e-12))
    day = d.timestamp.dt.floor("D").to_numpy()
    print(f"1분봉 {len(d):,} · {d.timestamp.iloc[0]} ~ {d.timestamp.iloc[-1]}")

    # 예측자 — 테이커 매수 비중(−1..+1)과 그 누적. 전부 봉 t 마감까지의 정보.
    tfi = 2.0 * tb / v - 1.0
    preds = {"tfi1": tfi}
    for k in (5, 15, 60):
        preds[f"tfi{k}"] = pd.Series(tfi).rolling(k, min_periods=k).mean().to_numpy()
    # 거래대금 가중(큰 봉의 불균형에 무게) — Cont 의 «깊이로 나눈다»에 대응하는 거친 대리물
    sv = (2.0 * tb - v)
    preds["sv_z60"] = ((pd.Series(sv) - pd.Series(sv).rolling(60, min_periods=60).mean())
                       / pd.Series(sv).rolling(60, min_periods=60).std()).to_numpy()

    print(f"\n{'예측자':>8} {'지평':>5} {'동시 ρ':>8} {'전방 IC':>8} {'일군집 t':>8} "
          f"{'E|r_H|bp':>9} {'손익분기IC':>9} {'관측/필요':>8}")
    rep = {}
    for name, x in preds.items():
        rep[name] = {}
        for H in HS:
            fwd = np.full(len(c), np.nan)
            fwd[:-H] = lc[H:] - lc[:-H]                    # 봉 t 종가 → t+H 종가
            cur = np.full(len(c), np.nan)
            cur[1:] = lc[1:] - lc[:-1]                     # 같은 봉의 수익(동시점)
            m = np.isfinite(x) & np.isfinite(fwd) & np.isfinite(cur)
            ic = float(spearmanr(x[m], fwd[m]).statistic)
            co = float(spearmanr(x[m], cur[m]).statistic)
            e_abs = float(np.mean(np.abs(fwd[m])) * 1e4)
            be = ROUND_TRIP_BP / (e_abs * 0.7979) if e_abs > 0 else float("nan")
            # 신호 부호로 베팅했을 때의 일 군집 t
            s = np.sign(x[m]); s[s == 0] = 1.0
            t = block_t(s * fwd[m] * 1e4, day[m])
            rep[name][H] = {"ic": ic, "contemp": co, "e_abs_bp": e_abs, "breakeven_ic": be,
                            "ratio": ic / be if be > 0 else float("nan"), "block_t": t,
                            "n": int(m.sum())}
            print(f"{name:>8} {H:>4}분 {co:>+8.4f} {ic:>+8.4f} {t:>+8.2f} {e_abs:>9.1f} "
                  f"{be:>9.4f} {ic/be if be > 0 else float('nan'):>7.0%}", flush=True)
        print()

    # 감쇠 반감기 — 1분 IC 대비 절반이 되는 지평
    print("감쇠 반감기(1분 전방 IC 대비 50% 지점):")
    for name in preds:
        base = abs(rep[name][1]["ic"])
        half = next((H for H in HS if abs(rep[name][H]["ic"]) < base / 2), None)
        print(f"  {name:>8}: 1분 IC {rep[name][1]['ic']:+.4f} → "
              f"{'반감 지평 ' + str(half) + '분' if half else '240분 내 반감 안 됨'}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"\n저장: {OUT/(a.tag+'.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
