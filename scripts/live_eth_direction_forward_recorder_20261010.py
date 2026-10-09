"""ETH·BTC 방향 후보(일봉 추세 5표 + UTC 00:00 개장 되돌림) 전진 기록기 — 결정만 남기고 결과는 판정일 전까지 계산하지 않는다 (2026-10-10).
사전등록: docs/prereg/eth_4h_rolling_hgb_frozen_20261010.json «portfolio2_recommended» — 판정 = 2026-09-15 이후 ≥180일, 4bp, 일 평균 CI 하한 > 0.

실행(서버 cron 후보, 미등록): 매일 00:31 UTC `python scripts/live_eth_direction_forward_recorder_20261010.py --record`
  · 일봉 추세: 닫힌 일봉 91개 종가로 7·14·28·56·90일 부호 평균(votes), |votes| ≥ .5 면 그 방향으로 그날 하루 보유.
  · 개장 되돌림: 00:00~00:30 UTC 수익 f30 = ln(00:25 봉 종가 / 23:55 봉 종가), σ30 = 직전 288개 5분 수익 표준편차 × √6.
    |f30| > 0.5·σ30 이면 반대로, 진입 = 00:25 봉 종가, 청산 = 04:25 봉 종가(04:30 UTC).
  · 08:31·16:31 UTC 에도 `--record` (펀딩 시각 가설 H_fund, 10-10 08:15 고정): 시각 h∈{00,08,16} 첫 30분 수익 f30 에서 테이커 흐름 설명 몫을 뺀
    잔차 = f30 − (b0 + b1·imb30) (계수 동결 FLOW_COEF) 가 |잔차| > 0.5·σ30 이면 잔차 반대로 4h. 근거 = 2022~26 ETH·BTC 잔차 β −8bp/σ(00시)·−4~−7(08·16시)·그 외 0.
`--status` = 건수·날짜만(결과 비공개). 손익 정산은 판정일 이후 별도로 한 번만.
"""
import json, os, sys, time, urllib.request
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "live", "eth_direction_forward.jsonl")
SYMS = ("ETHUSDT", "BTCUSDT")
VERDICT_FROM = "2027-03-14"
FLOW_COEF = {"ETHUSDT": (2.547, 264.279), "BTCUSDT": (1.627, 175.668)}           # f30(bp) = b0 + b1·imb30, 2022-01~2026-09 펀딩 시각 적합


def klines(sym, interval, limit):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={interval}&limit={limit}"
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read())


def decide(daily, m5, now_ms, sym="ETHUSDT", hour=0):
    """daily = 닫힌 일봉 [[open_ms, o, h, l, c, ...]](마지막 = 어제), m5 = 닫힌 5분봉(00:25 봉 포함). 반환 결정 dict."""
    dc = np.array([float(k[4]) for k in daily])
    votes = float(np.mean([np.sign(dc[-1] - dc[-1 - L]) for L in (7, 14, 28, 56, 90)]))
    day0 = now_ms // 86_400_000 * 86_400_000 + hour * 3_600_000                     # 기준 시각(00·08·16 UTC)
    by_open = {int(k[0]): float(k[4]) for k in m5}
    p0, p30 = by_open[day0 - 300_000], by_open[day0 + 1_500_000]                    # 23:55 봉 종가 = 00:00 가격 · 00:25 봉 종가 = 00:30 가격
    opens = sorted(t for t in by_open if t <= day0 - 300_000)[-289:]
    lc = np.log([by_open[t] for t in opens])
    sig30 = float(np.std(np.diff(lc), ddof=1) * np.sqrt(6) * 1e4)                  # 백테스트(판다스 rolling std)와 같은 ddof
    f30 = float(np.log(p30 / p0) * 1e4)
    return {"date": time.strftime("%Y-%m-%d", time.gmtime(day0 / 1000)), "votes": votes, "trend_pos": int(np.sign(votes)) if abs(votes) >= .5 else 0,
            "trend_entry_px": p0, "f30_bp": round(f30, 2), "sigma30_bp": round(sig30, 2),
            "open_pos": int(-np.sign(f30)) if abs(f30) > .5 * sig30 else 0, "entry_px": p30, "exit_due_ms": day0 + 4 * 3_600_000 + 1_800_000,
            "hour": hour, **fund_resid(m5, day0, f30, sig30, sym)}


def fund_resid(m5, day0, f30, sig30, sym):
    """H_fund: 첫 30분 테이커 불균형 → 흐름 설명 몫을 뺀 잔차와 결정. 테이커 필드(k[5] 거래량, k[9] 테이커 매수) 없으면 생략."""
    rows = [k for k in m5 if day0 <= int(k[0]) < day0 + 1_800_000]
    if len(rows) != 6 or len(rows[0]) < 10:
        return {}
    v = sum(float(k[5]) for k in rows); tb = sum(float(k[9]) for k in rows)
    imb = (2 * tb - v) / v if v > 0 else 0.0
    b0, b1 = FLOW_COEF[sym]; res = f30 - (b0 + b1 * imb)
    return {"imb30": round(imb, 5), "resid_bp": round(res, 2), "fund_pos": int(-np.sign(res)) if abs(res) > .5 * sig30 else 0}


def record():
    now = int(time.time() * 1000)
    hour = (now % 86_400_000) // 3_600_000
    if hour not in (0, 8, 16) or now % 3_600_000 < 1_860_000:                      # 00·08·16 시 31분 이후 그 시간 안에서만
        sys.exit("00:31·08:31·16:31 UTC 이후 그 시간 안에서 실행")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for sym in SYMS:
        daily = klines(sym, "1d", 93)[:-1]                                          # 형성 중인 오늘 일봉 제외
        m5 = [k for k in klines(sym, "5m", 320) if int(k[0]) + 300_000 <= now]      # 닫힌 봉만
        r = {"sym": sym, "recorded_ms": now, **decide(daily, m5, now - hour * 3_600_000, sym, hour)}
        with open(OUT, "a") as f:
            f.write(json.dumps(r) + "\n")
    print("기록", time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now / 1000)))


def status():
    rows = [json.loads(l) for l in open(OUT)] if os.path.exists(OUT) else []
    days = sorted({r["date"] for r in rows})
    print(f"기록 {len(rows)}줄 · {len(days)}일({days[0] if days else '-'} ~ {days[-1] if days else '-'}) · 판정 가능 {VERDICT_FROM} 이후 · 결과는 비공개")


def selftest():
    day0 = 1_800_000_000_000 // 86_400_000 * 86_400_000
    daily = [[day0 - (91 - i) * 86_400_000, 0, 0, 0, 100 + i] for i in range(91)]     # 꾸준한 상승 → votes = 1
    m5 = [[day0 - 300_000 * (n + 1), 0, 0, 0, 200.0 + (n % 2) * 0.1] for n in range(300)][::-1]   # 어제: 200.0/200.1 진동, 23:55 봉 = 200.0
    m5 += [[day0 + 300_000 * i, 0, 0, 0, 200.0 * (1 + 0.004 * (i + 1) / 6)] for i in range(6)]    # 00:00~00:30 +0.4%
    d = decide(daily, m5, day0 + 1_860_000)
    assert d["votes"] == 1.0 and d["trend_pos"] == 1
    assert abs(d["f30_bp"] - np.log(1.004) * 1e4) < 0.01 and 10 < d["sigma30_bp"] < 15     # f30 ≈ 39.9bp · σ30 ≈ 12bp
    assert d["open_pos"] == -1 and d["exit_due_ms"] == day0 + 16_200_000                  # 크게 오른 개장 → 숏, 04:30 청산
    assert "resid_bp" not in d                                                            # 테이커 필드 없으면 H_fund 생략
    m5t = [k + [0, 0, 0, 0, 0] for k in m5]                                               # 필드 10개로
    for k in m5t[-6:]:
        k[5], k[9] = 10.0, 5.0                                                            # 불균형 0 → 잔차 = f30 − b0
    d2 = decide(daily, m5t, day0 + 1_860_000)
    assert abs(d2["resid_bp"] - (d2["f30_bp"] - 2.547)) < 0.02 and d2["fund_pos"] == -1 and d2["hour"] == 0
    print("selftest OK", d2)


if __name__ == "__main__":
    {"--record": record, "--status": status, "--selftest": selftest}[sys.argv[1]]()
