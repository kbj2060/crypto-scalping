#!/usr/bin/env python3
"""대시보드 /api/footprint 가 센 체결량이 바이낸스 5분봉과 맞는지 확인한다.

풋프린트는 「가격 행별 매수/매도」를 보여주는데, 그 숫자가 맞는지는 눈으로 알 수 없다.
봉 합계는 klines 로 검증 가능하다 -- 총합은 volume, 매수합은 taker_buy_base 와 같아야 한다.
후자가 맞으면 `m`(매수자가 메이커) 해석, 즉 «어느 쪽이 공격했는가» 분류가 맞다는 뜻이다.

사용:  python3 scripts/check_footprint_tape_20260915.py [--url http://127.0.0.1:8787]
새 봉/버킷/분류 로직을 건드렸으면 이걸 돌린다. 수집 중(ready=false)이면 아직 못 잰다고 말한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

KLINES = "https://fapi.binance.com/fapi/v1/klines"
TOL = 0.002          # 0.2%. WS/REST 경계 봉은 1초 미만 겹칠 수 있다(server.py 주석 참조)
MAX_OUTLIERS = 1     # 그 경계 봉 하나까지만 봐준다


def fetch(url: str, timeout: float = 15.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8787")
    args = ap.parse_args()

    payload = fetch(args.url.rstrip("/") + "/api/footprint")
    bars = payload.get("bars") or []
    if not payload.get("ready"):
        print(f"수집 중(ready=false, {len(bars)}봉) -- 백필이 끝난 뒤 다시 돌린다")
        return 2
    if len(bars) < 2:
        print("봉이 너무 적다")
        return 2

    symbol = payload["symbol"]
    bar_seconds = int(payload["barSeconds"])
    # 마지막 봉은 아직 진행 중이라 klines 와 시점이 어긋난다 -- 닫힌 봉만 본다.
    closed = bars[:-1]
    start_ms = closed[0]["time"] * 1000
    end_ms = closed[-1]["time"] * 1000 + 1
    raw = fetch(f"{KLINES}?symbol={symbol}&interval={bar_seconds // 60}m"
                f"&startTime={start_ms}&endTime={end_ms}&limit=500")
    kline = {int(k[0]) // 1000: (float(k[5]), float(k[9])) for k in raw}

    failures = 0
    for bar in closed:
        got = kline.get(bar["time"])
        if got is None:
            print(f"  {bar['time']}  klines 에 없음")
            failures += 1
            continue
        vol, taker_buy = got
        buy = sum(level[1] for level in bar["levels"])
        sell = sum(level[2] for level in bar["levels"])
        d_total = (buy + sell - vol) / vol if vol else 0.0
        d_buy = (buy - taker_buy) / taker_buy if taker_buy else 0.0
        bad = abs(d_total) > TOL or abs(d_buy) > TOL
        failures += bad
        print(f"  {bar['time']}  총합 {buy + sell:11.1f} vs {vol:11.1f} ({d_total:+.4%})"
              f" · 매수 {buy:10.1f} vs {taker_buy:10.1f} ({d_buy:+.4%}){'  ← 불일치' if bad else ''}")

    print(f"닫힌 봉 {len(closed)}개 중 불일치 {failures}개 (허용 {MAX_OUTLIERS})")
    if failures > MAX_OUTLIERS:
        print("FAIL -- 버킷 합산이나 매수/매도 분류가 틀렸다")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
