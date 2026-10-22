"""SOXLUSDT 주말 상승 전진 기록기 — 사전등록 docs/experiments/soxl_weekend_fwd_prereg_20261022.md (이 파일과 같은 커밋 = 동결, 2026-10-22).

가설(연구 scripts/research_soxl_trend_limit_sessions_20261022.py 사후 관찰): 바이낸스 SOXLUSDT 무기한은 ETF 휴장 주말(토 00:00 ET → 월 00:00 ET)에 오른다.
  표본 안 2026-05-16~10-17 주말 블록 23개 중 17개 +, 평균 +2.2% (휴장 연휴 포함 블록 기준).
한 주말 = 뉴욕 시각 토 00:00 → 월 00:00 (서머타임 자동). 가격 P(T) = T 직전에 닫힌 5분봉 종가.
  기록: SOXL·ETH 의 P(토 00:00)·P(월 00:00)·P(월 09:30)·P(월 16:00) + SOXL 펀딩(토 00:00 초과 ~ 월 00:00 이하) 합·횟수. 손익은 계산하지 않는다.
  ETH 는 대조(크립토 주말 흐름이 설명하는지 해석용, 판정에 안 씀). 월 09:30·16:00 은 «월요일에 되돌리나» 해석용.
순수익(롱) = P(월 00)/P(토 00) − 1 − 2×5bp(시장가 왕복) − 펀딩 합.
판정(동결): 1차 = 전진 26주말 → 평균 순수익 ≤ 0 이면 기각·종료 · 최종 = 52주말 → 평균 > 0 이고 단측 t ≥ 1.675(df 51, 5%) 이면 통과.
cron(서버, 매일 21:30 UTC): `python scripts/live_soxl_weekend_forward_recorder_20261022.py --record` — 끝난 주말 중 장부에 없는 것만 덧붙임(빠진 주는 다음 실행이 메움, 쓴 줄은 안 고침).
  --status 건수·날짜만 · --look 판정(26·52주말 전엔 거부, --force 는 looks 장부에 위반으로 남김) · --insample 상장~동결 주말을 같은 코드로(장부 안 씀) · --selftest
"""
import json, os, sys, time, urllib.request
from datetime import date, datetime, timedelta, timezone
from statistics import mean, stdev
from zoneinfo import ZoneInfo

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "data", "live", "soxl_weekend_forward.jsonl")
LOOKS = os.path.join(ROOT, "data", "live", "soxl_weekend_forward_looks.jsonl")
NY = ZoneInfo("America/New_York")
FWD_FROM, INS_FROM = date(2026, 10, 24), date(2026, 5, 16)          # 전진 첫 토요일 · 상장 뒤 첫 토요일
LOOK_N, T_CRIT, TAKER = (26, 52), 1.675, 0.0005
SYMS = ("SOXLUSDT", "ETHUSDT")


def get(path):
    with urllib.request.urlopen("https://fapi.binance.com/fapi/v1/" + path, timeout=15) as r:
        return json.loads(r.read())


def ms(dt):
    return int(dt.timestamp() * 1000)


def points(sat):
    """토요일 날짜 → 기록 시각 4개(UTC ms). 서머타임 바뀌는 주말도 뉴욕 벽시계 기준."""
    t = lambda d, h, m: ms(datetime(d.year, d.month, d.day, h, m, tzinfo=NY))
    mon = sat + timedelta(days=2)
    return {"sat00": t(sat, 0, 0), "mon00": t(mon, 0, 0), "mon0930": t(mon, 9, 30), "mon1600": t(mon, 16, 0)}


def price_at(sym, t_ms):
    """t 직전에 닫힌 5분봉 종가 · 그 봉이 t 에서 5분 넘게 떨어져 있으면 lag 로 남긴다."""
    k = get(f"klines?symbol={sym}&interval=5m&endTime={t_ms - 300_000}&limit=1")[-1]
    return float(k[4]), (t_ms - 300_000 - int(k[0])) // 60_000


def record_weekend(sat):
    pt = points(sat)
    row = {"weekend": sat.isoformat(), **{f"{k}_utc_ms": v for k, v in pt.items()}}
    for sym in SYMS:
        for k, t in pt.items():
            px, lag = price_at(sym, t)
            row[f"{sym}_{k}"] = px
            if lag:
                row[f"{sym}_{k}_lag_min"] = lag
    f = get(f"fundingRate?symbol=SOXLUSDT&startTime={pt['sat00'] + 1}&endTime={pt['mon00']}&limit=100")
    row["SOXLUSDT_funding_sum"] = sum(float(x["fundingRate"]) for x in f)
    row["SOXLUSDT_funding_n"] = len(f)
    row["recorded_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return row


def closed_weekends(since, now_ms):
    """since 부터 월 16:00 ET 봉이 닫히고 10분 지난 토요일들."""
    out, d = [], since
    while points(d)["mon1600"] + 600_000 <= now_ms:
        out.append(d)
        d += timedelta(days=7)
    return out


def read_rows(path=OUT):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def net(row):
    return row["SOXLUSDT_mon00"] / row["SOXLUSDT_sat00"] - 1 - 2 * TAKER - row["SOXLUSDT_funding_sum"]


def stats(rows):
    x = [net(r) for r in rows]
    eth = [r["ETHUSDT_mon00"] / r["ETHUSDT_sat00"] - 1 for r in rows]
    mon = [r["SOXLUSDT_mon1600"] / r["SOXLUSDT_mon0930"] - 1 for r in rows]
    n = len(x); m = mean(x); s = stdev(x) if n > 1 else float("nan")
    return {"n": n, "평균 순수익 %": round(m * 100, 3), "표준편차 %": round(s * 100, 3), "t": round(m / (s / n ** .5), 3) if n > 1 else None,
            "양수 비율": round(sum(v > 0 for v in x) / n, 3), "ETH 주말 평균 % (대조)": round(mean(eth) * 100, 3),
            "월 미국장 평균 % (해석)": round(mean(mon) * 100, 3), "펀딩 합 평균 bp": round(mean(r["SOXLUSDT_funding_sum"] for r in rows) * 1e4, 2)}


def decide(st, n_look):
    if n_look == LOOK_N[0]:
        return "기각(종료)" if st["평균 순수익 %"] <= 0 else "계속"
    return "통과" if st["평균 순수익 %"] > 0 and st["t"] >= T_CRIT else "불통과"


def record():
    done = {r["weekend"] for r in read_rows()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for sat in closed_weekends(FWD_FROM, int(time.time() * 1000)):
        if sat.isoformat() in done:
            continue
        row = record_weekend(sat)
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("기록", sat)


def look(force):
    rows = [r for r in read_rows() if r["weekend"] >= FWD_FROM.isoformat()]
    due = [k for k in LOOK_N if len(rows) >= k]
    if not due and not force:
        sys.exit(f"아직 {len(rows)}주말 — 1차 점검은 {LOOK_N[0]}주말부터(판정 전 결과 열람 금지)")
    k = due[-1] if due else len(rows)
    st = stats(rows[:k])
    st["판정"] = decide(st, k) if due else "판정 전 열람(위반)"
    with open(LOOKS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "n": k, "forced": not due, **st}, ensure_ascii=False) + "\n")
    print(json.dumps(st, ensure_ascii=False, indent=1))


def selftest():
    p = points(date(2026, 10, 31))                                      # 11-01(일) 서머타임 끝: 토 00 ET = 04 UTC, 월 00 ET = 05 UTC
    assert p["sat00"] == ms(datetime(2026, 10, 31, 4, tzinfo=timezone.utc)) and p["mon00"] == ms(datetime(2026, 11, 2, 5, tzinfo=timezone.utc))
    assert closed_weekends(date(2026, 10, 24), points(date(2026, 10, 24))["mon1600"] + 600_000) == [date(2026, 10, 24)]
    assert closed_weekends(date(2026, 10, 24), points(date(2026, 10, 24))["mon1600"] + 599_999) == []
    r = {"SOXLUSDT_sat00": 100.0, "SOXLUSDT_mon00": 103.0, "SOXLUSDT_funding_sum": 0.0006}
    assert abs(net(r) - (0.03 - 0.001 - 0.0006)) < 1e-12
    assert decide({"평균 순수익 %": 0.0, "t": 9}, 26) == "기각(종료)" and decide({"평균 순수익 %": 0.1, "t": -1}, 26) == "계속"
    assert decide({"평균 순수익 %": 0.5, "t": 1.675}, 52) == "통과" and decide({"평균 순수익 %": 0.5, "t": 1.6}, 52) == "불통과"
    print("selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        selftest()
    elif "--record" in a:
        record()
    elif "--status" in a:
        rows = [r for r in read_rows() if r["weekend"] >= FWD_FROM.isoformat()]
        print(f"전진 {len(rows)}주말 기록 · 마지막 {rows[-1]['weekend'] if rows else '-'} · 1차 점검 {LOOK_N[0]} · 최종 {LOOK_N[1]}")
    elif "--look" in a:
        look("--force" in a)
    elif "--insample" in a:
        rows = [record_weekend(d) for d in closed_weekends(INS_FROM, ms(datetime(FWD_FROM.year, FWD_FROM.month, FWD_FROM.day, tzinfo=NY)))]
        with open(os.path.join(ROOT, "tmp", "soxl_trend_limit_20261022", "weekend_insample.jsonl"), "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(json.dumps(stats(rows), ensure_ascii=False, indent=1))
    else:
        sys.exit(__doc__)
