#!/usr/bin/env python3
"""rtserve 아홉째 조각 정답지(2026-11-22): /api/trend · /api/cycle-path (요청 때 바이낸스 klines 한 번).

원본 핸들러 본문을 옮겨 적었다(dashboard/server.py api_trend 2743~2760 · api_cycle_path 4820~4845 · fetch_binance_json 1531~1551 를 Read 로 읽은 줄).
계산 함수 trend_rule.trend_payload 는 원본 모듈을 그대로 불러 쓴다. 바깥 접속 없음: fetch_binance_json 자리에 «녹화된 응답 표»를 둔다.
aiohttp 예외 글자(f"{type}: {exc}" = "HTTPBadGateway: <reason>" · 본문 "400: <reason>")는 aiohttp 3.14 로 확인한 꼴을 작은 클래스로 흉내 낸다
(서버 판 venv 에 aiohttp 가 없다). 🔴파이썬 3.11 로 돌릴 것 -- sum() 이 3.12 부터 보정 합이라 vol_ann 끝자리가 달라진다:
    source <(bash rtmodel/tools/server_env.sh <venv> --env) && $PY rtserve/tools/gen_rest_fixture.py    # 저장소 루트에서
출력 rtserve/tests/fixtures/gen/rest/rest.json(커밋 안 함).
"""
from __future__ import annotations

import json
import random
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dashboard.trend_rule import trend_payload  # noqa: E402
from dashboard.swing_lines import swing_state  # noqa: E402  (원본 모듈 그대로 · api_swing_lines 2762~2794 는 아래에 옮겨 적음)

assert sys.version_info[:2] == (3, 11), sys.version
OUT = ROOT / "rtserve" / "tests" / "fixtures" / "gen" / "rest" / "rest.json"
NOW = 1_795_000_123.456789
FAPI = "/fapi/v1/klines"
MARKET_SYMBOLS = {"eth": "ETHUSDT", "sol": "SOLUSDT", "btc": "BTCUSDT", "xrp": "XRPUSDT", "hype": "HYPEUSDT", "sndk": "SNDKUSDT", "soxl": "SOXLUSDT",
                  "skhynix": "SKHYNIXUSDT", "mu": "MUUSDT"}                                    # orders.py:338
MANUAL_EXEC_SYMBOLS = {"eth": "ETHUSDT", "sol": "SOLUSDC", "xrp": "XRPUSDC", "btc": "BTCUSDC", "sndk": "SNDKUSDT", "soxl": "SOXLUSDT"}   # orders.py:347~353(env 없음)


class HTTPException(Exception):
    status = 0

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason

    def __str__(self):
        return self.reason


class HTTPBadRequest(HTTPException):
    status = 400


class HTTPBadGateway(HTTPException):
    status = 502


REST: dict[str, tuple[int, str]] = {}   # 요청 대상(경로?쿼리) → (상태, 본문)
SENT: list[str] = []
SWR: dict[str, object] = {}


def fetch_binance_json(path, params, *, error_reason=None):                     # 1531(밴 갈래 없음 -- Rust 시험이 따로 본다)
    target = path + "?" + urllib.parse.urlencode(params)
    SENT.append(target)
    status, body = REST[target]
    if status != 200:
        if error_reason is None:
            return None
        raise HTTPBadGateway(error_reason)
    return json.loads(body)


def swr_cached(key, produce):                                                    # 1482: 이 시나리오는 신선 구간 안(같은 키 = 캐시 값)
    if key not in SWR:
        SWR[key] = produce()
    return SWR[key]


def api_trend(q):                                                                # 2743
    asset = str(q.get("asset") or "eth").lower()
    if asset not in MARKET_SYMBOLS:
        raise HTTPBadRequest("unsupported_asset")

    def produce():
        raw = fetch_binance_json(FAPI, {"symbol": MARKET_SYMBOLS[asset], "interval": "1d", "limit": 120}, error_reason="trend_upstream_error")
        now_ms = NOW * 1000
        done = [r for r in raw if int(r[6]) < now_ms]
        return trend_payload([float(r[4]) for r in done], [int(r[6]) + 1 for r in done])
    try:
        return swr_cached(f"trend_{asset}_1d", produce)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def api_swing_lines(q):                                                          # 2762
    asset = str(q.get("asset") or "eth").lower()
    if asset not in MARKET_SYMBOLS:
        raise HTTPBadRequest("unsupported_asset")

    def produce():
        out = {"ok": True}
        now_ms = NOW * 1000
        errs = []
        for tf in ("1h", "30m"):
            try:
                raw = fetch_binance_json(FAPI, {"symbol": MARKET_SYMBOLS[asset], "interval": tf, "limit": 1000}, error_reason="swing_upstream_error")
            except Exception as exc:  # noqa: BLE001
                out[tf] = None; errs.append(f"{tf}: {type(exc).__name__}"); continue
            done = [r for r in raw if int(r[6]) < now_ms]
            out[tf] = swing_state([float(r[2]) for r in done], [float(r[3]) for r in done], [float(r[4]) for r in done],
                                  [int(r[6]) + 1 for r in done])
        if len(errs) == 2:
            raise RuntimeError("; ".join(errs))
        if errs:
            out["partial"] = errs
        return out
    try:
        return swr_cached(f"swing_{asset}", produce)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def api_cycle_path(q):                                                         # 4820
    symbol = str(q.get("symbol") or "").upper()
    if symbol not in {*MARKET_SYMBOLS.values(), *MANUAL_EXEC_SYMBOLS.values()}:
        raise HTTPBadRequest("unsupported_symbol")
    try:
        start = int(q.get("start") or 0)
    except ValueError:
        raise HTTPBadRequest("bad_start") from None
    now_ms = int(NOW * 1000)
    if not now_ms - 30 * 86400_000 <= start <= now_ms:
        raise HTTPBadRequest("bad_start")
    span_min = (now_ms - start) / 60_000
    interval = "1m" if span_min <= 1490 else "5m" if span_min <= 7450 else "15m"

    def produce():
        raw = fetch_binance_json(FAPI, {"symbol": symbol, "interval": interval, "startTime": start, "limit": 1500}, error_reason="cycle_path_upstream_error")
        return {"ok": True, "interval": interval, "rows": [[int(r[0]), float(r[4])] for r in raw]}
    try:
        return swr_cached(f"cycle_path:{symbol}:{start}", produce)
    except HTTPException as exc:
        return {"ok": False, "reason": exc.reason, "rows": []}


def kl(open_ms, step_ms, closes, dp=2):
    """klines 행(바이낸스 꼴: 시각은 정수 · 값은 글자)"""
    return [[open_ms + i * step_ms, f"{c:.{dp}f}", f"{c * 1.01:.{dp}f}", f"{c * 0.99:.{dp}f}", f"{c:.{dp}f}", "123.456", open_ms + (i + 1) * step_ms - 1,
             "1.0", 10, "1.0", "1.0", "0"] for i, c in enumerate(closes)]


def walk(rnd, n, c0, sd):
    out, c = [], c0
    for _ in range(n):
        c *= 1 + rnd.gauss(0, sd)
        out.append(c)
    return out


def main():
    rnd = random.Random(20261122)
    day = 86_400_000
    d0 = int(NOW * 1000) // day * day - 119 * day   # 120번째 봉 = 지금 형성 중
    rows = {
        "ETHUSDT": kl(d0, day, walk(rnd, 120, 2500, 0.03)),
        "SOLUSDT": kl(d0 + 60 * day, day, walk(rnd, 60, 150, 0.03)),
        "BTCUSDT": kl(d0, day, [0.0 if i == 40 else c for i, c in enumerate(walk(rnd, 120, 90000, 0.02))]),
        "HYPEUSDT": kl(d0, day, [50 * 0.999 ** i * (1 + 1e-5 * (i % 3)) for i in range(120)], dp=6),   # 꾸준한 하락 · 아주 작은 변동 → 크기 −2 상한
        "SNDKUSDT": kl(d0, day, [100.0] * 120),                                                         # 같은 종가 → f1 0표 · 신호 −1
        "SOXLUSDT": kl(d0, day, walk(rnd, 120, 30, 0.05), dp=3),
    }
    for sym, rs in rows.items():
        REST[FAPI + "?" + urllib.parse.urlencode({"symbol": sym, "interval": "1d", "limit": 120})] = (200, json.dumps(rs))
    REST[FAPI + "?" + urllib.parse.urlencode({"symbol": "XRPUSDT", "interval": "1d", "limit": 120})] = (500, '{"code": -1}')

    now_ms = int(NOW * 1000)
    starts = {
        "1m": now_ms - 600_000, "1m_edge": now_ms - 1490 * 60_000, "5m_edge": now_ms - 1490 * 60_000 - 1, "5m": now_ms - 3 * day,
        "5m_hi": now_ms - 7450 * 60_000, "15m": now_ms - 7450 * 60_000 - 1, "15m_far": now_ms - 30 * day,
    }
    paths = [f"/api/trend?asset={a}" for a in ("eth", "sol", "xrp", "btc", "hype", "SNDK", "soxl", "doge")] + ["/api/trend", "/api/trend?asset="]
    for i, (name, st) in enumerate(starts.items()):
        sym = ["ETHUSDC", "SOLUSDT", "XRPUSDC", "BTCUSDT", "SNDKUSDT", "SOXLUSDT", "HYPEUSDT"][i]
        step = {"1m": 60_000, "5m": 300_000, "15m": 900_000}[api_cycle_interval(st, now_ms)]
        REST[FAPI + "?" + urllib.parse.urlencode({"symbol": sym, "interval": api_cycle_interval(st, now_ms), "startTime": st, "limit": 1500})] = (
            200, json.dumps(kl(st // step * step, step, walk(rnd, 37, 100 + i, 0.002), dp=4)))
        paths.append(f"/api/cycle-path?symbol={sym.lower() if i % 2 else sym}&start={st}")
    bad = now_ms - 5 * day
    REST[FAPI + "?" + urllib.parse.urlencode({"symbol": "BTCUSDC", "interval": "5m", "startTime": bad, "limit": 1500})] = (502, "bad gateway")
    paths += [f"/api/cycle-path?symbol=BTCUSDC&start={bad}", "/api/cycle-path?symbol=DOGEUSDT&start=1", "/api/cycle-path?symbol=ETHUSDT&start=abc",
              "/api/cycle-path?symbol=ETHUSDT", f"/api/cycle-path?symbol=ETHUSDT&start={now_ms + 1}", f"/api/cycle-path?symbol=ETHUSDT&start={now_ms - 30 * day - 1}",
              f"/api/cycle-path?symbol=SOLUSDC&start={str(starts['1m'])[:4]}_{str(starts['1m'])[4:]}", f"/api/cycle-path?symbol=ETHUSDC&start={starts['1m']}"]
    REST[FAPI + "?" + urllib.parse.urlencode({"symbol": "SOLUSDC", "interval": "1m", "startTime": starts["1m"], "limit": 1500})] = (
        200, json.dumps(kl(starts["1m"] // 60_000 * 60_000, 60_000, walk(rnd, 11, 140, 0.001), dp=3)))

    for sym, n, fail in (("ETHUSDT", 1000, ()), ("SOLUSDT", 1000, ("1h",)), ("XRPUSDT", 1000, ("1h", "30m")), ("BTCUSDT", 60, ()), ("SOXLUSDT", 1000, ("30m",))):
        for tf, step in (("1h", 3_600_000), ("30m", 1_800_000)):
            t0 = int(NOW * 1000) // step * step - (n - 1) * step
            body = (500, "x") if tf in fail else (200, json.dumps(kl(t0, step, walk(rnd, n, 1000, 0.01), dp=2)))
            REST[FAPI + "?" + urllib.parse.urlencode({"symbol": sym, "interval": tf, "limit": 1000})] = body
    for tf, step in (("1h", 3_600_000), ("30m", 1_800_000)):   # 종가가 스윙 고점과 «꼭 같은» 봉(380) -- 깨짐은 > 이지 >= 가 아니다
        t0 = int(NOW * 1000) // step * step - 999 * step
        rs = kl(t0, step, walk(rnd, 1000, 1000, 0.0005), dp=2)
        x = f"{max(float(r[2]) for r in rs[250:351]) + 20:.2f}"
        rs[300][2] = rs[380][2] = rs[380][4] = x
        REST[FAPI + "?" + urllib.parse.urlencode({"symbol": "SNDKUSDT", "interval": tf, "limit": 1000})] = (200, json.dumps(rs))
    paths += [f"/api/swing-lines?asset={a}" for a in ("eth", "sol", "xrp", "btc", "soxl", "sndk", "doge")] + ["/api/swing-lines"]

    cases = []
    for p in paths:
        SENT.clear()
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(p).query, keep_blank_values=True))
        h = api_trend if p.startswith("/api/trend") else api_swing_lines if p.startswith("/api/swing") else api_cycle_path
        try:
            status, body = 200, json.dumps(h(q))
        except HTTPException as exc:
            status, body = exc.status, f"{exc.status}: {exc.reason}"
        cases.append({"path": p, "status": status, "body": body, "sent": list(SENT)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"now": NOW, "rest": {k: list(v) for k, v in REST.items()}, "cases": cases}, ensure_ascii=False, indent=1))
    print(f"{OUT.relative_to(ROOT)}: 요청 {len(cases)} · REST 표 {len(REST)} · 파이썬 {sys.version.split()[0]}")


def api_cycle_interval(start, now_ms):
    span_min = (now_ms - start) / 60_000
    return "1m" if span_min <= 1490 else "5m" if span_min <= 7450 else "15m"


if __name__ == "__main__":
    main()
