"""ETH max pain 표본 밖 장부 — 서버 cron 자동 기록 (2026-10-07, 사용자 «남은 기능 진행»).

규칙 원문 = docs/experiments/eth_maxpain_1h_oos_ledger_20261003.md (바꾸지 않는다). 손으로 쓰던 장부가 10-04 뒤 멈춰 기계로 옮겼다.
  pre     02:01~02:09 · 07:01~07:09 UTC 매분: 수집기 상태 파일(deribit_gex_state.json)의 그날 만기 max pain 을 **진입 전에** 적는다.
          DB 를 안 연다(수집기 단일 writer 락). 스냅샷이 창(02:00~02:10 · 07:00~07:10) 밖이면 안 적고 다음 분에 다시 본다.
  settle  08:15·08:45 UTC: Deribit 공개 API(ETH-PERPETUAL 1분봉 · 정산가)로 1시간·6시간 규칙을 정산하고,
          다음 날 만기의 «이벤트일(사전)»을 거시 달력 상태 파일에서 적는다(만기 전 24h 안 «높음» 일정).
  backfill DATE  pre 를 놓친 날: DB 를 read_only 로 잠깐 열어 같은 창의 첫 스냅샷으로 채운다(source=db · 이벤트일 = 사후 미상).
  report  장부를 표로(판정 = 표본 밖 40개 뒤 -- 문서 규칙).
  selftest
가격·정산 = Deribit 공개 API(바이낸스 REST 아님). 장부 = data/live/maxpain_oos_ledger.jsonl(한 줄 = 한 사실, 덧붙이기만).
cron(서버): 1-9 2,7 * * * … pre · 15,45 8 * * * … settle
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
LEDGER = LIVE / "maxpain_oos_ledger.jsonl"
STATE = LIVE / "deribit_gex_state.json"
CAL = LIVE / "macro_calendar_state.json"
DB = LIVE / "deribit_options.duckdb"
API = "https://www.deribit.com/api/v2/public/"
WINDOWS = {"pre6h": 2, "pre1h": 7}          # 스냅샷 창 = 그 시 00~10분(UTC) -- 1시간 규칙 원 정의(exp−60분 ≤ t < exp−50분)


def max_pain(g: pd.DataFrame) -> float:
    """analyze_eth_expiry_day_20261003.max_pain 과 같은 식(수집기 options_summary 의 pain 과도 같다 -- selftest)."""
    ks = np.sort(g.strike.unique()); c = g[g.option_type == "call"]; q = g[g.option_type == "put"]
    return float(ks[int(np.argmin([(c.open_interest * np.maximum(K - c.strike, 0)).sum()
                                   + (q.open_interest * np.maximum(q.strike - K, 0)).sum() for K in ks]))])


def rows() -> list[dict]:
    return [json.loads(x) for x in LEDGER.read_text("utf-8").splitlines() if x.strip()] if LEDGER.exists() else []


def has(kind: str, day: str) -> bool:
    return any(r["kind"] == kind and r["day"] == day for r in rows())


def append(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**row, "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, ensure_ascii=False) + "\n")
    print("기록", row)


def exp_of(day: str) -> pd.Timestamp:
    return pd.Timestamp(day + " 08:00", tz="UTC")


def in_window(t: pd.Timestamp, day: str, hour: int) -> bool:
    t0 = pd.Timestamp(f"{day} {hour:02d}:00", tz="UTC")
    return t0 <= t < t0 + pd.Timedelta(minutes=10)


def pre(now: datetime | None = None) -> None:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    day = f"{now:%Y-%m-%d}"
    kind = next((k for k, h in WINDOWS.items() if now.hour == h), None)
    if kind is None or has(kind, day):
        return
    eth = (json.loads(STATE.read_text("utf-8")).get("currencies") or {}).get("ETH") or {}
    t = pd.to_datetime(eth.get("recorded_at_utc") or "1970-01-01", utc=True)
    if not in_window(t, day, WINDOWS[kind]):
        print(f"{kind}: 스냅샷 {t} 이 창 밖 -- 다음 분에 다시")
        return
    e = next((x for x in (eth.get("options") or {}).get("expiries") or [] if x.get("exp_ms") == int(exp_of(day).timestamp() * 1000)), None)
    if not e:
        print(f"{kind}: 그날 만기({day} 08:00) 요약 없음")
        return
    append({"kind": kind, "day": day, "snap_utc": t.isoformat(), "pain": e["pain"], "source": "state",
            "oi_usd": round(e.get("call_oi_usd", 0) + e.get("put_oi_usd", 0))})


def backfill(day: str) -> None:
    """놓친 날 -- DB read_only 로 잠깐(수집기는 쓰기 실패 시 30초 재시도·버퍼 보류, 8eefc85c)."""
    import duckdb
    exp = exp_of(day)
    con = duckdb.connect(str(DB), read_only=True)
    try:
        ch = con.execute("SELECT recorded_at_utc, option_type, strike, open_interest FROM option_chain_snapshot "
                         "WHERE currency='ETH' AND expiration_ts = ? AND recorded_at_utc BETWEEN ? AND ?",
                         [exp.to_pydatetime(), (exp - pd.Timedelta(hours=7)).to_pydatetime(), exp.to_pydatetime()]).df()
    finally:
        con.close()
    ch["t"] = pd.to_datetime(ch.recorded_at_utc, utc=True)
    for kind, h in WINDOWS.items():
        if has(kind, day):
            continue
        g = ch[[in_window(t, day, h) for t in ch.t]]
        if g.empty:
            print(f"{kind} {day}: 창 안 스냅샷 없음(수집기 결측)")
            continue
        g = g[g.t == g.t.min()]
        append({"kind": kind, "day": day, "snap_utc": g.t.iloc[0].isoformat(), "pain": max_pain(g), "source": "db"})


def _get(method: str, **params) -> dict:
    err = None
    for k in range(4):                         # get_tradingview_chart_data 는 4~55초까지 늘어진다(09-28 실측)
        try:
            return requests.get(API + method, params=params, timeout=90).json()["result"]
        except Exception as exc:               # 네트워크·빈 응답 -- 다음 cron 이 다시 돈다
            err = exc
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"{method} 실패: {err}")


def perp_1m(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    out = []
    for a in pd.date_range(lo, hi, freq="12h"):
        b = min(a + pd.Timedelta(hours=12), hi)
        r = _get("get_tradingview_chart_data", instrument_name="ETH-PERPETUAL", resolution="1",
                 start_timestamp=int(a.timestamp() * 1000), end_timestamp=int(b.timestamp() * 1000))
        out.append(pd.DataFrame({"o": r["open"], "c": r["close"]}, index=pd.to_datetime(r["ticks"], unit="ms", utc=True)))
    p = pd.concat(out)
    return p[~p.index.duplicated()].sort_index()


def leg(p: pd.DataFrame, snap: pd.Timestamp, pain: float) -> tuple[float, int, pd.Timestamp]:
    """진입 = 스냅샷 다음 분 시가(원 정의) → (진입가, 방향, 진입 시각). 방향 0 = max pain 과 같은 가격(매매 없음)."""
    a = snap.floor("min") + pd.Timedelta(minutes=1)
    P0 = float(p.o.asof(a))
    return P0, int(np.sign(pain - P0)), a


def settle(day: str | None = None, now: datetime | None = None) -> None:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    day = day or f"{now:%Y-%m-%d}"
    pres = {r["kind"]: r for r in rows() if r["day"] == day and r["kind"] in WINDOWS}
    if not has("settle", day) and pres:
        st = {x["date"]: x["delivery_price"] for x in _get("get_delivery_prices", index_name="eth_usd", count=10)["data"]}.get(day)
        if st is None:
            print(f"{day}: 정산가 아직 없음 -- 다음 cron")
        else:
            exp = exp_of(day)
            p = perp_1m(exp - pd.Timedelta(hours=31), exp)
            out = {"kind": "settle", "day": day, "delivery": st}
            if "pre1h" in pres:
                r = pres["pre1h"]; P0, s, _ = leg(p, pd.Timestamp(r["snap_utc"]), r["pain"])
                P1 = float(p.c.asof(exp - pd.Timedelta(minutes=1)))
                out.update(r1h_entry=P0, r1h_exit=P1, r1h_side=s, r1h_bp=round(s * (P1 / P0 - 1) * 1e4, 2),
                           delivery_vs_pain_pct=round((st / r["pain"] - 1) * 100, 3))
            if "pre6h" in pres:
                r = pres["pre6h"]; P0, s, a = leg(p, pd.Timestamp(r["snap_utc"]), r["pain"])
                avg = float(p.c[(p.index >= a - pd.Timedelta(hours=24)) & (p.index < a)].mean())
                sc = int(np.sign(avg - P0))         # 대조 = 24h 평균가 쪽
                out.update(r6h_entry=P0, r6h_side=s, r6h_pct=round(s * (st / P0 - 1) * 100, 3),
                           ctl_side=sc, ctl_pct=round(sc * (st / P0 - 1) * 100, 3))
            append(out)
    if day == f"{now:%Y-%m-%d}":                    # 지난 날 재정산 때는 사전 표시를 만들지 않는다
        event_flag(f"{now + pd.Timedelta(days=1):%Y-%m-%d}", now)


def event_flag(day: str, now: pd.Timestamp) -> None:
    """만기(day 08:00 UTC) 전 24h 안 «높음» 일정 -- 진입(07:00) 23시간 전에 적으므로 «사전» 표시다."""
    if has("event", day):
        return
    exp = exp_of(day)
    try:
        cal = json.loads(CAL.read_text("utf-8"))
    except (OSError, ValueError):
        cal = {}
    gen = pd.to_datetime(cal.get("generated_at") or "1970-01-01", utc=True)
    hi = [e["title_ko"] for e in cal.get("events") or [] if e.get("importance") == "high"
          and exp - pd.Timedelta(hours=24) <= pd.to_datetime(e["time_utc"], utc=True) < exp]
    stale = (now - gen) > pd.Timedelta(hours=12)
    append({"kind": "event", "day": day, "event": None if stale else bool(hi), "high": hi,
            "calendar_generated_utc": gen.isoformat(), "note": "달력 낡음 -- 미상" if stale else "사전(자동)"})


def report() -> None:
    by: dict[str, dict] = {}
    for r in rows():
        by.setdefault(r["day"], {}).update({f"{r['kind']}.{k}": v for k, v in r.items()})
    print("| 만기 | 이벤트일 | 1h pain·출처 | 1시간 규칙 bp | 6시간 규칙 % | 대조 % | 정산−pain % |")
    print("|---|---|---|---|---|---|---|")
    for d in sorted(by):
        x = by[d]
        ev = {True: "예(사전)", False: "아니오(사전)", None: "미상"}.get(x.get("event.event"), "사후 미상")
        print(f"| {d} | {ev} | {x.get('pre1h.pain', '-')}·{x.get('pre1h.source', '-')} | {x.get('settle.r1h_bp', '-')} | "
              f"{x.get('settle.r6h_pct', '-')} | {x.get('settle.ctl_pct', '-')} | {x.get('settle.delivery_vs_pain_pct', '-')} |")


def selftest() -> None:
    import importlib.util
    import tempfile
    rng = np.random.default_rng(7)
    g = None
    # ① 수집기 pain(options_summary 의 min-람다) == 이 max_pain -- 동률 포함 무작위 체인 300개
    for _ in range(300):
        ks = rng.choice(np.arange(2000, 3200, 25), size=rng.integers(3, 15), replace=False).astype(float)
        g = pd.DataFrame({"strike": np.r_[ks, ks], "option_type": ["call"] * len(ks) + ["put"] * len(ks),
                          "open_interest": rng.integers(0, 4, 2 * len(ks)).astype(float)})
        calls, puts = g[g.option_type == "call"], g[g.option_type == "put"]
        coll = min(sorted(g.strike.unique()), key=lambda P: float(((P - calls.strike).clip(lower=0) * calls.open_interest).sum()
                                                                 + ((puts.strike - P).clip(lower=0) * puts.open_interest).sum()))
        assert coll == max_pain(g), (coll, max_pain(g))
    # ② analyze 스크립트와 같은 함수 값
    spec = importlib.util.spec_from_file_location("ax", ROOT / "scripts/analyze_eth_expiry_day_20261003.py")
    ax = importlib.util.module_from_spec(spec); spec.loader.exec_module(ax)
    assert ax.max_pain(g) == max_pain(g)
    # ③ 진입·방향 산수(가짜 1분봉) + 창 판정
    idx = pd.date_range("2026-10-05 01:00", "2026-10-05 08:00", freq="1min", tz="UTC")
    p = pd.DataFrame({"o": np.arange(len(idx)) + 2500.0, "c": np.arange(len(idx)) + 2500.5}, index=idx)
    P0, s, a = leg(p, pd.Timestamp("2026-10-05 07:00:05", tz="UTC"), 2000.0)
    assert a == pd.Timestamp("2026-10-05 07:01", tz="UTC") and P0 == p.o[a] and s == -1
    assert in_window(pd.Timestamp("2026-10-05 07:09:59", tz="UTC"), "2026-10-05", 7)
    assert not in_window(pd.Timestamp("2026-10-05 07:10:00", tz="UTC"), "2026-10-05", 7)
    # ④ pre: 창 밖이면 안 적고, 창 안이면 한 번만(다음 분 재실행은 건너뜀) · 이벤트일 자동 표시(만기 뒤 일정 제외 · 한 번만)
    global LEDGER, STATE, CAL
    with tempfile.TemporaryDirectory() as td:
        LEDGER, STATE, CAL = Path(td) / "l.jsonl", Path(td) / "s.json", Path(td) / "c.json"
        exp_ms = int(exp_of("2026-10-05").timestamp() * 1000)
        mk = lambda t: STATE.write_text(json.dumps({"currencies": {"ETH": {"recorded_at_utc": t, "options": {"expiries": [  # noqa: E731
            {"exp_ms": exp_ms, "pain": 2650.0, "call_oi_usd": 1e6, "put_oi_usd": 2e6}]}}}}))
        mk("2026-10-05T06:50:04+00:00"); pre(datetime(2026, 10, 5, 7, 1, tzinfo=timezone.utc)); assert rows() == []
        mk("2026-10-05T07:00:04+00:00"); pre(datetime(2026, 10, 5, 7, 2, tzinfo=timezone.utc))
        pre(datetime(2026, 10, 5, 7, 3, tzinfo=timezone.utc))
        assert [r["kind"] for r in rows()] == ["pre1h"] and rows()[0]["pain"] == 2650.0
        CAL.write_text(json.dumps({"generated_at": "2026-10-05T06:00:00+00:00", "events": [
            {"time_utc": "2026-10-05T18:00:00+00:00", "importance": "high", "title_ko": "FOMC 의사록"},
            {"time_utc": "2026-10-06T09:00:00+00:00", "importance": "high", "title_ko": "만기 뒤라 제외"}]}))
        event_flag("2026-10-06", pd.Timestamp("2026-10-05 08:15", tz="UTC"))
        ev = rows()[-1]; assert ev["event"] is True and ev["high"] == ["FOMC 의사록"], ev
        event_flag("2026-10-06", pd.Timestamp("2026-10-05 08:45", tz="UTC")); assert len(rows()) == 2
    print("selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    {"pre": lambda: pre(), "settle": lambda: settle(a[1] if len(a) > 1 else None), "backfill": lambda: backfill(a[1]),
     "report": report, "selftest": selftest}[a[0] if a else "report"]()
