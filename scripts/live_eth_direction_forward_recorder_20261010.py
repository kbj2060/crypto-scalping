"""ETH·BTC 방향 후보(일봉 추세 5표 + UTC 00:00 개장 되돌림) 전진 기록기 — 결정만 남기고 결과는 판정일 전까지 계산하지 않는다 (2026-10-10).
사전등록: docs/prereg/eth_4h_rolling_hgb_frozen_20261010.json «portfolio2_recommended» — 판정 = 2026-09-15 이후 ≥180일, 4bp, 일 평균 CI 하한 > 0.

실행(서버 cron 후보, 미등록): 매일 00:31 UTC `python scripts/live_eth_direction_forward_recorder_20261010.py --record`
  · 일봉 추세: 닫힌 일봉 91개 종가로 7·14·28·56·90일 부호 평균(votes), |votes| ≥ .5 면 그 방향으로 그날 하루 보유.
  · 개장 되돌림: 00:00~00:30 UTC 수익 f30 = ln(00:25 봉 종가 / 23:55 봉 종가), σ30 = 직전 288개 5분 수익 표준편차 × √6.
    |f30| > 0.5·σ30 이면 반대로, 진입 = 00:25 봉 종가, 청산 = 04:25 봉 종가(04:30 UTC).
`--status` = 건수·날짜만(결과 비공개). 손익 정산은 판정일 이후 별도로 한 번만.
"""
import json, os, sys, time, urllib.request
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "live", "eth_direction_forward.jsonl")
SYMS = ("ETHUSDT", "BTCUSDT")
VERDICT_FROM = "2027-03-14"


def klines(sym, interval, limit):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={interval}&limit={limit}"
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read())


def decide(daily, m5, now_ms):
    """daily = 닫힌 일봉 [[open_ms, o, h, l, c, ...]](마지막 = 어제), m5 = 닫힌 5분봉(00:25 봉 포함). 반환 결정 dict."""
    dc = np.array([float(k[4]) for k in daily])
    votes = float(np.mean([np.sign(dc[-1] - dc[-1 - L]) for L in (7, 14, 28, 56, 90)]))
    day0 = now_ms // 86_400_000 * 86_400_000
    by_open = {int(k[0]): float(k[4]) for k in m5}
    p0, p30 = by_open[day0 - 300_000], by_open[day0 + 1_500_000]                    # 23:55 봉 종가 = 00:00 가격 · 00:25 봉 종가 = 00:30 가격
    opens = sorted(t for t in by_open if t <= day0 - 300_000)[-289:]
    lc = np.log([by_open[t] for t in opens])
    sig30 = float(np.std(np.diff(lc), ddof=1) * np.sqrt(6) * 1e4)                  # 백테스트(판다스 rolling std)와 같은 ddof
    f30 = float(np.log(p30 / p0) * 1e4)
    return {"date": time.strftime("%Y-%m-%d", time.gmtime(day0 / 1000)), "votes": votes, "trend_pos": int(np.sign(votes)) if abs(votes) >= .5 else 0,
            "trend_entry_px": p0, "f30_bp": round(f30, 2), "sigma30_bp": round(sig30, 2),
            "open_pos": int(-np.sign(f30)) if abs(f30) > .5 * sig30 else 0, "entry_px": p30, "exit_due_ms": day0 + 4 * 3_600_000 + 1_800_000}


def record():
    now = int(time.time() * 1000)
    if now % 86_400_000 < 1_860_000:                                                # 00:31 UTC 전이면 00:25 봉이 아직 안 닫힘
        sys.exit("00:31 UTC 이후에 실행")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for sym in SYMS:
        daily = klines(sym, "1d", 93)[:-1]                                          # 형성 중인 오늘 일봉 제외
        m5 = [k for k in klines(sym, "5m", 320) if int(k[0]) + 300_000 <= now]      # 닫힌 봉만
        r = {"sym": sym, "recorded_ms": now, **decide(daily, m5, now)}
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
    print("selftest OK", d)


if __name__ == "__main__":
    {"--record": record, "--status": status, "--selftest": selftest}[sys.argv[1]]()
