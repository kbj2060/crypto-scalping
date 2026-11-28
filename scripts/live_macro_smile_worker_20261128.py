"""/api/macro-trend · /api/opt-smile 본문을 파일로 쓰는 워커(2026-11-28 · rtserve 전환용 -- rtserve/src/filefeed.rs 가 읽기만 한다).

원본 핸들러(dashboard/server.py api_macro_trend·opt_smile_payload)와 같은 함수·같은 실패 모양 · 주기 = 원본 swr 수명(3600초 · 60초).
  data/live/macro_trend.json · data/live/opt_smile_{eth,btc,xrp}.json  -- json.dumps 기본(= aiohttp json_response 본문 바이트)
FRED 키 = 원본처럼 env FRED_API_KEY(dashboard.server 를 들이면 그쪽 load_dotenv 가 읽는다).
  python scripts/live_macro_smile_worker_20261128.py              # 상시
  python scripts/live_macro_smile_worker_20261128.py --selftest   # 127.0.0.1 가짜 원천만
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data/live"
SMILE_CURS = ("ETH", "BTC", "XRP")   # 원본 asset 허용 목록 {"eth","btc","xrp"}


def write(live: Path, name: str, payload: dict) -> None:
    tmp = live / f".{name}.tmp"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, live / name)


def macro_body(srv) -> dict:
    """원본 macro_trend_payload 그대로."""
    try:
        return srv.macro_trend.payload()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}", "rows": [], "summary": {}, "errors": {}}


def smile_body(srv, cur: str) -> dict:
    """원본 opt_smile_payload 그대로(aiohttp 공용 세션 대신 requests · 시간 6초 · 200 아니면 None)."""
    lst, pre = srv.DERIBIT_SPECS[cur][:2]
    try:
        r = requests.get(srv.DERIBIT_BOOK_URL, params={"currency": lst, "kind": "option"}, timeout=6.0)
        j = r.json() if r.status_code == 200 else None
    except (requests.RequestException, ValueError):
        j = None
    if not j or not isinstance(j.get("result"), list):
        return {"ok": False, "reason": "deribit_unavailable"}
    return {**srv.front_smile(j["result"], time.time() * 1000, pre), "fetched_ts": time.time()}


def loop(every: float, fn) -> None:
    while True:
        t0 = time.time()
        try:
            fn()
        except Exception as e:  # noqa: BLE001 -- 디스크 오류 하나로 워커가 죽으면 안 된다(다음 주기에 다시)
            print(f"[macro_smile] {type(e).__name__}: {e}", flush=True)
        time.sleep(max(1.0, every - (time.time() - t0)))


def run() -> None:
    from dashboard import server as srv
    print(f"[macro_smile] start smile={SMILE_CURS}", flush=True)
    threading.Thread(target=loop, args=(3600.0, lambda: write(LIVE, "macro_trend.json", macro_body(srv))), daemon=True).start()
    loop(60.0, lambda: [write(LIVE, f"opt_smile_{c.lower()}.json", smile_body(srv, c)) for c in SMILE_CURS])


def selftest() -> None:
    """가짜 원천(127.0.0.1) → 워커 파일 바이트 == 원본 함수 본문의 json.dumps. 시각은 고정."""
    import http.server
    import tempfile
    from urllib.parse import parse_qs, urlparse
    os.environ["FRED_API_KEY"] = "selftest"   # 진짜 키 대신(가짜 서버로만 간다)
    day = 86_400_000
    t_end = 1_795_000_000_000
    kl = [[t_end - i * day, "0", "0", "0", str(2000 + (i * 37 % 300)), "0", t_end - i * day + day - 1] for i in range(1200, 0, -1)]
    obs = {"observations": [{"date": time.strftime("%Y-%m-%d", time.gmtime((t_end - i * day) / 1000)), "value": str(100 + i % 17)} for i in range(1400, 0, -1)]}
    llama = [{"date": str((t_end - i * day) // 1000), "totalCirculatingUSD": {"peggedUSD": 1e11 + i * 1e7}} for i in range(1400, 0, -1)]

    def book(lst: str) -> dict:
        names = {"ETH": "ETH-", "BTC": "BTC-", "USDC": "XRP_USDC-"}[lst]
        S = {"ETH": 3000.0, "BTC": 90000.0, "USDC": 2.0}[lst]
        rows = [{"instrument_name": f"{names}30NOV26-{str(round(S * (1 + d / 100), 4)).replace('.', 'd') if lst == 'USDC' else int(S * (1 + d / 100))}-{cp}",
                 "mark_iv": 50 + d * d * 0.5 + (1 if cp == "C" else 0), "underlying_price": S} for d in range(-6, 7, 2) for cp in "CP"]
        return {"result": rows + [{"instrument_name": "SOL_USDC-30NOV26-150-C", "mark_iv": 80, "underlying_price": 150}]}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            body = {"/deribit": lambda: book(q["currency"]) if q["currency"] != "BTC" else None, "/fred": lambda: obs,
                    "/llama": lambda: llama, "/klines": lambda: kl}[u.path]()
            self.send_response(200 if body is not None else 503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *a):
            pass

    hs = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=hs.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{hs.server_address[1]}"
    from dashboard import server as srv
    srv.DERIBIT_BOOK_URL = base + "/deribit"
    mt = srv.macro_trend
    mt.FRED_API, mt.LLAMA_URL, mt.KLINES_URL = base + "/fred", base + "/llama", base + "/klines"
    time.time = lambda: t_end / 1000 - 3 * 86400   # noqa: E731 -- 만기 30NOV26 이전 · 일봉 진행 중 봉 제외가 정해지게
    live = Path(tempfile.mkdtemp())
    write(live, "macro_trend.json", macro_body(srv))
    for c in SMILE_CURS:
        write(live, f"opt_smile_{c.lower()}.json", smile_body(srv, c))
    # 원본 핸들러가 내는 본문: json_response = json.dumps(payload) 기본 인자
    want = {"macro_trend.json": json.dumps(mt.payload())}
    for c in SMILE_CURS:
        lst, pre = srv.DERIBIT_SPECS[c][:2]
        j = book(lst) if c != "BTC" else None
        want[f"opt_smile_{c.lower()}.json"] = json.dumps({**srv.front_smile(j["result"], time.time() * 1000, pre), "fetched_ts": time.time()}
                                                         if j else {"ok": False, "reason": "deribit_unavailable"})
    for name, w in want.items():
        got = (live / name).read_text(encoding="utf-8")
        assert got == w, (name, got[:200], w[:200])
        print(f"{name}: {len(got)}B 같음 · {got[:90]}")
    m = json.loads(want["macro_trend.json"])
    assert m["ok"] and m["rows"] and not m["errors"], m.get("errors")
    assert json.loads(want["opt_smile_eth.json"])["ok"] and json.loads(want["opt_smile_xrp.json"])["ok"]
    assert not list(live.glob(".*.tmp"))
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else run()
