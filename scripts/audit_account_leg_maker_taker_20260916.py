"""실계좌 체결 단위 maker/taker 감사 — 남은 테이커 누수가 진입인가 청산인가 (2026-09-16, 읽기전용).

원장(`account_round_trips.jsonl`)의 `fills` 는 **개수**뿐이라 leg 분해가 안 된다.
바이낸스 `/fapi/v1/userTrades` 는 체결마다 `maker` 플래그를 주므로 이미 있는 데이터로 특정된다.
주문을 내지 않는다 — GET 만 한다.
"""
import asyncio, hashlib, hmac, json, os, time, sys
from pathlib import Path
from urllib.parse import urlencode
import aiohttp

ROOT = Path(__file__).resolve().parents[1]
FAPI = "https://fapi.binance.com"
SYM = "ETHUSDT"


def sign(p, secret, off=0):
    p = {**p, "timestamp": int(time.time() * 1000) + off, "recvWindow": 5000}
    q = urlencode(p)
    return f"{q}&signature={hmac.new(secret.encode(), q.encode(), hashlib.sha256).hexdigest()}"


async def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key, sec = os.getenv("BINANCE_API_KEY", ""), os.getenv("BINANCE_SECRET_KEY", "")
    assert key and sec, ".env 에 키가 없다"
    trips = [json.loads(l) for l in open(ROOT / "data/live/account_round_trips.jsonl")]
    t0 = min(x["entry_time"] for x in trips) - 3600_000
    t1 = max(x["exit_time"] for x in trips) + 3600_000
    out = []
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{FAPI}/fapi/v1/time") as r:
            off = (await r.json())["serverTime"] - int(time.time() * 1000)
        cur = t0
        while cur < t1:                       # 7일 창씩 (API 제한)
            end = min(cur + 6 * 86400_000, t1)
            url = f"{FAPI}/fapi/v1/userTrades?" + sign(
                {"symbol": SYM, "startTime": cur, "endTime": end, "limit": 1000}, sec, off)
            async with s.get(url, headers={"X-MBX-APIKEY": key}) as r:
                w = r.headers.get("x-mbx-used-weight-1m")
                d = await r.json()
            if isinstance(d, dict):
                print("ERR", d, file=sys.stderr); break
            out += d
            print(f"  {cur} ~ {end}: {len(d)}건 (weight {w})", flush=True)
            cur = end
            await asyncio.sleep(0.4)          # 봇과 REST 가중 공유 — 여유 있게
    seen, tr = set(), []
    for x in out:
        if x["id"] in seen: continue
        seen.add(x["id"]); tr.append(x)
    tr.sort(key=lambda x: x["time"])
    print(f"\n총 체결 {len(tr)}건 · {tr[0]['time']} ~ {tr[-1]['time']}")

    def agg(rows):
        q = sum(float(x["quoteQty"]) for x in rows)
        qm = sum(float(x["quoteQty"]) for x in rows if x["maker"])
        c = sum(float(x["commission"]) for x in rows)
        return len(rows), q, (qm / q if q else float("nan")), (c / q * 1e4 if q else float("nan"))

    op = [x for x in tr if float(x["realizedPnl"]) == 0.0]     # 진입(실현손익 0)
    cl = [x for x in tr if float(x["realizedPnl"]) != 0.0]     # 청산
    print(f"\n{'구분':<8}{'체결':>6}{'명목$':>12}{'메이커비중':>10}{'수수료bp':>9}")
    for nm, rows in (("진입", op), ("청산", cl)):
        n, q, m, b = agg(rows)
        print(f"{nm:<8}{n:>6}{q:>12,.0f}{m*100:>9.1f}%{b:>9.2f}")

    PEG = 1789171200000    # 2026-09-12 00:00 UTC (peg 진입 배포)
    PEGX = 1789257600000  # 2026-09-13 00:00 UTC (peg 청산 배포)
    print(f"\n■ peg 배포(09-12) 전후")
    print(f"{'구간':<8}{'다리':<6}{'체결':>6}{'명목$':>12}{'메이커비중':>10}{'수수료bp':>9}")
    for lab, lo, hi in (("~09-11", 0, PEG), ("09-12", PEG, PEGX), ("09-13~", PEGX, 10**14)):
        for nm, rows in (("진입", op), ("청산", cl)):
            r2 = [x for x in rows if lo <= x["time"] < hi]
            if not r2: continue
            n, q, m, b = agg(r2)
            print(f"{lab:<8}{nm:<6}{n:>6}{q:>12,.0f}{m*100:>9.1f}%{b:>9.2f}")

    import datetime as dt
    print(f"\n■ peg 배포 이후(09-12~) 테이커 체결 — 남은 누수 현장")
    late = [x for x in tr if x["time"] >= PEG and not x["maker"]]
    print(f"  테이커 {len(late)}건 · 명목 {sum(float(x['quoteQty']) for x in late):,.0f}$")
    for x in late:
        print(f"  {dt.datetime.utcfromtimestamp(x['time']/1000):%m-%d %H:%M:%S} "
              f"{'청산' if float(x['realizedPnl']) else '진입'} {x['side']:>4} "
              f"{float(x['quoteQty']):9,.0f}$ order {x.get('orderId')}")
    print(f"\n■ 09-12 이후 주문ID 단위 — 한 주문이 통째로 테이커면 폴백/시장가, 섞이면 부분체결")
    import collections
    byo = collections.defaultdict(list)
    for x in tr:
        if x["time"] >= PEG: byo[x["orderId"]].append(x)
    for oid, rows in sorted(byo.items(), key=lambda kv: kv[1][0]["time"]):
        q = sum(float(z["quoteQty"]) for z in rows)
        qm = sum(float(z["quoteQty"]) for z in rows if z["maker"])
        kind = "청산" if float(rows[0]["realizedPnl"]) else "진입"
        print(f"  {dt.datetime.utcfromtimestamp(rows[0]['time']/1000):%m-%d %H:%M} {kind} "
              f"{rows[0]['side']:>4} 체결 {len(rows):>2}건 {q:9,.0f}$ 메이커 {qm/q*100:5.1f}%")
    json.dump(tr, open(ROOT / "tmp/user_trades_20260916.json", "w"))
    print(f"\n원시 체결 저장: tmp/user_trades_20260916.json")

asyncio.run(main())
