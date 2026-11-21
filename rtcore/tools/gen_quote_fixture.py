#!/usr/bin/env python3
"""rtcore 주문 경로 시세(quote.rs)·5분봉 청산 원 일치 시험의 입력·정답지 (2026-11-21). 작다 -- 커밋한다(tests/fixtures/quote_sol.json · quote_bn_ctx.sqlite).

① 1분봉: 합성 `@kline_1m`(형성 중 x=false 갱신 섞음 · 빈 분 몇 개 = 메시지 없음 · 한 번 재연결). 정답 = 마감 봉마다
   원본 `scripts/live_manual_peg_entry_20260912.realized_vol_bpm`(import)에 «REST klines limit 32 의 마지막(미완결)을 버린 31개»와 같은 종가 열
   (빈 분 = 직전 종가)을 넣은 값 · 손절 감시의 last_bar(orders.py bracket_tick 1222~1225 옮겨 적음: 5분봉 중 time + 300 ≤ 지금 인 마지막 봉의 (끝, 종가)).
② 최우선 호가: 합성 `@bookTicker` -- 정답 = REST bookTicker 꼴 {symbol, bidPrice, askPrice}(글자 그대로).
③ 마지막 체결가: 합성 `@trade` 두 종목 -- 정답 = 종목마다 마지막 체결 float(p)(REST ticker/price 가 주는 값 · server.py fetch_market_ticker).
④ 5분봉 청산 원: 바이낸스 맥락 DB(원본 수집기 init_db 로 표 · 행은 합성) + 합성 `@forceOrder` -- 정답 = server.py liq_events_load 의 5분봉 몫 +
   liq_5m_add + liq_5m_payload(옮겨 적음 · server.py 는 import 하지 않는다).
네트워크 없음(소켓 connect 를 막는다) · 실제 data/ 안 건드림.
    python3 rtcore/tools/gen_quote_fixture.py
"""
from __future__ import annotations

import json
import random
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "rtcore" / "tests" / "fixtures"


def _no_net(*_a, **_k):
    raise RuntimeError("gen_quote_fixture: 네트워크 금지")


socket.socket.connect = _no_net
socket.create_connection = _no_net
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from live_manual_peg_entry_20260912 import EXIT_VOL_WINDOW, realized_vol_bpm  # noqa: E402
import live_binance_ctx_collector_20261001 as bnctx  # noqa: E402

SEED = 20261125
T0 = 1_763_700_000_000          # 봉 경계(ms) -- 합성
BS = 300


# ── server.py 옮겨 적음(2026-11-20 판) ─────────────────────────────────────────
def liq_5m_add(store, ev, bar_s=300, keep_bars=288):
    """server.py liq_5m_add(517~535) 그대로."""
    bar = int(ev.get("ts_ms") or 0) // 1000 // bar_s * bar_s
    a = store.setdefault(bar, [0.0, 0.0, 0, None, None])
    is_long = ev.get("side") == "long"
    a[0 if is_long else 1] += float(ev.get("usd") or 0.0)
    a[2] += 1
    px = float(ev.get("price") or 0.0)
    if px > 0:
        if is_long:
            a[3] = px if a[3] is None else min(a[3], px)
        else:
            a[4] = px if a[4] is None else max(a[4], px)
    cut = bar - keep_bars * bar_s
    for old in [b for b in store if b < cut]:
        del store[old]


def liq_5m_payload(store, from_s, bars, now_s, bar_s=300):
    """server.py liq_5m_payload(538~556) 그대로."""
    now_bar = int(now_s) // bar_s * bar_s
    from_bar = int(from_s) // bar_s * bar_s
    out = []
    for k in range(bars - 1, -1, -1):
        b = now_bar - k * bar_s
        if b <= from_bar:
            continue
        a = store.get(b) or (0.0, 0.0, 0, None, None)
        out.append({"ts": datetime.fromtimestamp(b, timezone.utc).isoformat(), "bar": b,
                    "long_usd": round(a[0], 2), "short_usd": round(a[1], 2), "events": int(a[2]),
                    "long_min_px": a[3], "short_max_px": a[4], "partial": b == now_bar})
    return {"warmed_up": True, "bars": out, "error": None, "source": "dashboard-forceorder"}


def last_bar(closed_5m: list[tuple[int, float]], now_s: float):
    """orders.py bracket_tick 1222~1225: closed = [c for c in 5분봉 if time + 300 <= now] → (time + 300, close) 의 마지막."""
    closed = [c for c in closed_5m if c[0] + BS <= now_s]
    return (closed[-1][0] + BS, closed[-1][1]) if closed else None


def main() -> None:
    rng = random.Random(SEED)
    out: dict = {"about": "합성 시세 · 원본 함수 정답(rtcore/tools/gen_quote_fixture.py, 시드 %d)" % SEED, "t0": T0, "vol_window": EXIT_VOL_WINDOW}
    # ① 1분봉
    sym = "SOLUSDC"
    px = 140.0
    series: list[tuple[int, float, str]] = []      # 분마다(빈 분 포함) -- REST 가 줄 열
    msgs, checks = [], []
    empty = {7, 8, 23, 44, 64}                     # 메시지 없는 분(체결 없음) -- 64 는 5분봉 마지막 분(65분 끝)
    reconnect_before = 50                          # 이 분 앞에서 재연결(코어는 모은 종가를 버린다)
    for i in range(90):
        open_ms = T0 + i * 60_000
        if i in empty:
            series.append((open_ms, series[-1][1], series[-1][2]))
            continue
        px = round(px * (1 + rng.gauss(0, 0.0008)), 2)
        cs = f"{px:.2f}" if rng.random() < 0.8 else f"{px:.3f}"   # 글자 꼴 섞음(종가는 글자 그대로 실린다)
        series.append((open_ms, float(cs), cs))
        if i == reconnect_before:
            msgs.append({"kind": "reconnect"})
        for _ in range(rng.randint(0, 2)):          # 형성 중 갱신(x=false) -- 코어는 안 쓴다
            msgs.append({"kind": "kline", "rx_ms": open_ms + rng.randint(1000, 50_000),
                         "text": json.dumps({"e": "kline", "E": open_ms + 30_000, "s": sym, "k": {"t": open_ms, "T": open_ms + 59_999, "s": sym, "i": "1m",
                                                                                        "o": "140.00", "c": f"{px + 0.5:.2f}", "h": "1", "l": "1", "v": "1", "x": False}})})
        rx = open_ms + 60_000 + rng.randint(5, 400)
        msgs.append({"kind": "kline", "rx_ms": rx,
                     "text": json.dumps({"e": "kline", "E": rx - 3, "s": sym, "k": {"t": open_ms, "T": open_ms + 59_999, "s": sym, "i": "1m",
                                                                                    "o": "140.00", "c": cs, "h": "1", "l": "1", "v": "12.5", "x": True}})})
        # 정답: 이 봉이 막 마감된 순간(rx) REST limit 32 → [:-1] = 마감 31개
        closes = [c for _, c, _ in series[-(EXIT_VOL_WINDOW + 1):]]
        fives = [(t, c) for t, c, _ in series if (t + 60_000) % (BS * 1000) == 0]
        closed_5m = [((t + 60_000) // 1000 - BS, c) for t, c in fives]
        lb = last_bar(closed_5m, rx / 1000)
        checks.append({"after": len(msgs), "minute": i, "vol_py": realized_vol_bpm(closes) if len(series) >= EXIT_VOL_WINDOW + 1 else None,
                       "last_bar": list(lb) if lb else None})
    out["kline"] = {"symbol": sym.lower(), "msgs": msgs, "checks": checks, "empty": sorted(empty), "reconnect_before": reconnect_before}
    # ② 호가
    books = []
    for i in range(20):
        b = round(140 + rng.gauss(0, 0.3), 2)
        bs, as_ = f"{b:.2f}", (f"{b + 0.01:.2f}" if i % 3 else f"{b + 0.01:.3f}")
        books.append({"rx_ms": T0 + i * 137, "text": json.dumps({"e": "bookTicker", "u": 1000 + i, "s": sym, "b": bs, "B": "12.3", "a": as_, "A": "4.5",
                                                                  "T": T0 + i * 137 - 2, "E": T0 + i * 137 - 1}),
                      "want": {"symbol": sym, "bidPrice": bs, "askPrice": as_}})
    out["books"] = books
    # ③ 마지막 체결가
    trades, last = [], {}
    tid = 1
    for i in range(60):
        s = rng.choice(("SOLUSDT", "ETHUSDT"))
        p = "0" if i == 37 else f"{(140 if s == 'SOLUSDT' else 3000) * (1 + rng.gauss(0, 0.001)):.2f}"
        q = "0" if p == "0" else f"{rng.uniform(0.1, 5):.3f}"
        trades.append({"rx_ms": T0 + i * 50, "text": json.dumps({"e": "trade", "E": T0 + i * 50, "T": T0 + i * 50 - 3, "s": s, "t": tid, "p": p, "q": q,
                                                                  "X": "NA" if p == "0" else "MARKET", "m": rng.random() < 0.5})})
        tid += 1
        if float(p) > 0:     # 가격 0 체결(바이낸스 «NA»)은 ticker/price 에 안 잡힌다 -- 코어도 feed 에서 거른다
            last[s.lower()] = float(p)
        if i % 10 == 9:
            trades[-1]["want"] = dict(last)
    out["trades"] = trades
    # ④ 5분봉 청산 원: DB(최근 5,000건 · 재시작 때 읽음) + 실시간 사건
    db = FIX / "quote_bn_ctx.sqlite"
    for x in FIX.glob("quote_bn_ctx.sqlite*"):
        x.unlink()
    bnctx.init_db(db)
    import sqlite3
    now_s = T0 / 1000 + 6 * 3600                   # 재시작 = T0 + 6시간
    rows = []
    t = int((now_s - 30 * 3600) * 1000)             # 30시간 전부터(288봉 = 24시간 밖도 섞임)
    while t < now_s * 1000:
        t += rng.randint(20_000, 400_000)
        if t >= now_s * 1000:
            break
        q = round(rng.lognormvariate(1, 1), 3)
        p = round(140 * (1 + rng.gauss(0, 0.01)), 3) if rng.random() > 0.05 else 0.0
        rows.append((t, rng.choice(("long", "short")), q, p, q * p, rng.choice(("SOLUSDT", "SOLUSDT", "ETHUSDT"))))
    con = sqlite3.connect(db)
    con.executemany("INSERT INTO liquidations VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    con.execute("PRAGMA journal_mode=DELETE")
    con.close()
    # liq_events_load(server.py 1785~1815) 의 5분봉 몫 -- 옮겨 적음
    store: dict = {}
    from_s = now_s
    got = sorted([r for r in rows if r[5] == "SOLUSDT"], key=lambda r: -r[0])[:5000]
    cut_5m = now_s * 1000 - 288 * 300 * 1000
    first = None
    for ts, side, qty, price, usd, s in reversed(got):
        first = ts if first is None else first
        if ts >= cut_5m:
            liq_5m_add(store, {"ts_ms": ts, "side": side, "qty": qty, "price": price, "usd": usd})
    if first is not None:
        from_s = first / 1000
    live = []
    tl = int(now_s * 1000)
    for i in range(30):
        tl += rng.randint(500, 40_000)
        q = f"{rng.lognormvariate(1, 1):.2f}"
        ap = "0" if i == 11 else f"{140 * (1 + rng.gauss(0, 0.004)):.3f}"
        S = rng.choice(("SELL", "BUY"))
        live.append({"rx_ms": tl + 20, "text": json.dumps({"e": "forceOrder", "E": tl + 5, "o": {"s": "SOLUSDT", "S": S, "o": "LIMIT", "f": "IOC", "q": q, "p": "140.10",
                                                                                              "ap": ap, "X": "FILLED", "l": q, "z": q, "T": tl}})})
        qty = float(q)
        price = float(ap) if ap else 140.10     # 원본 float(o.get("ap") or o.get("p") or 0.0) -- "0" 은 참(글자)이라 ap = 0.0 이다(극값에 안 들어가고 usd 0)
        liq_5m_add(store, {"ts_ms": tl, "side": "long" if S == "SELL" else "short", "qty": qty, "price": price, "usd": qty * price})
    end_s = tl / 1000 + 1
    pay = liq_5m_payload(store, from_s, 288, end_s)
    out["liq5m"] = {"symbol": "solusdt", "restart_s": now_s, "live": live, "end_s": end_s, "from_s": from_s,
                    "bars": [[b["bar"], b["long_usd"], b["short_usd"], b["events"], b["long_min_px"], b["short_max_px"]] for b in pay["bars"] if b["events"]],
                    "payload_bars": len(pay["bars"])}
    p = FIX / "quote_sol.json"
    p.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    print(f"{p.relative_to(ROOT)}: 1분봉 메시지 {len(msgs)} · 대조 {len(checks)} · 호가 {len(books)} · 체결 {len(trades)} · "
          f"청산 DB {len(rows)} · 실시간 {len(live)} · 청산 원 봉 {len(out['liq5m']['bars'])}/{pay and len(pay['bars'])} · {p.stat().st_size // 1024}KB · DB {db.stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
