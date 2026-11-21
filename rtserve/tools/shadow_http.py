#!/usr/bin/env python3
"""rtserve 병행 대조(2026-11-22): 같은 GET 주소를 파이썬(8787)과 rtserve(8790)에 거의 같은 순간에 보내 상태·Content-Type·본문을 견준다.

- 읽기 전용(GET 만 · 쓰기·주문 주소 없음) · 루프백 주소만 받는다 · 표준 라이브러리만.
- 초당 요청 상한(기본 2/초 = 파이썬 서버에 보내는 수 · rtserve 에도 같은 수) -- 파이썬 서버를 괴롭히지 않게.
- 본문이 JSON 이면 칸별로 견주고, 시각에 달린 칸(IGNORE · 문서 §8 표)은 이름으로 뺀다. 아니면 바이트.
- 주소마다: «같음 n/N · 다른 칸(횟수) · 두 값 예 · 처리 시간(두 쪽 중앙·상위 1%)».
- 모델 주소 5(dir4h …)는 --models 를 줄 때만(병행 중 rtserve 는 REST 꺼짐이라 늘 RestOff -- 다른 것이 당연하다).
- 주문·계좌 14 는 안 보낸다(파이썬 쪽이 거래소 계좌 REST 를 부른다).

    python3 rtserve/tools/shadow_http.py [--py http://127.0.0.1:8787] [--rs http://127.0.0.1:8790] [--rounds 3] [--rate 2] [--coins eth,sol,xrp] [--only 부분글자] [--models]
    python3 rtserve/tools/shadow_http.py --selftest
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

# 시각·나이·지금에 달린 칸 -- 두 서버가 다른 순간에 만든 값이라 글자가 같을 수 없다(이름이 같으면 어느 깊이든 뺀다)
IGNORE = {
    "generated_at", "updated_at", "computed_at", "fetched_at", "now", "now_sec", "now_ms", "server_time",
    "age", "age_s", "age_sec", "age_ms", "tradeAge", "oiAge", "elapsed_ms",
    "asof_ms", "bar_elapsed_sec",   # 2026-11-22 서버 대조: pred-record asof_ms = 요청 순간 ms · liquidation-5m-signal bar_elapsed_sec = 지금 − 봉 시작
}

ROUTES = [  # (주소, 코인별로 ?asset= 붙임?)
    ("/dashboard/live/", False), ("/api/state", False), ("/api/acct-notes", False), ("/api/acct-wrong", False),
    ("/api/risk-radar", False), ("/api/sr-lean", False), ("/api/regime-band", False), ("/api/liq-tab", True), ("/api/liq-tab?asset=all", False),
    ("/api/iceberg", True), ("/api/ob5m", True), ("/api/treasury-futures", False), ("/api/treasury-futures?h=24", False), ("/api/xyz100", False),
    ("/api/macro-today", False), ("/api/macro-calendar", False), ("/api/macro-spikes", False), ("/api/news", False), ("/api/chip-score", True),
    ("/api/notify-center", False), ("/api/breakout-detector", True), ("/api/gex", False), ("/api/bubble-tape", True), ("/api/pred-record", False),
    ("/api/push/config", False), ("/api/push/devices", False), ("/api/session-alerts", False), ("/api/liquidation-5m-signal", True),
    ("/api/supply-1s", True), ("/api/footprint", True), ("/api/footprint?bars=48", True), ("/api/oi-5m", True), ("/api/oi-5m?bars=12", True),
    ("/api/liquidation-5m-history", True), ("/api/liquidation-5m-history?bars=144", True),
    ("/api/footprint-daily", True), ("/api/footprint-4h", True), ("/api/flow/heatmap?mode=rows", False), ("/api/flow/heatmap?mode=rows&symbol=solusdt", False),
]
MODEL_ROUTES = [("/api/dir4h", True), ("/api/dir-short", True), ("/api/vol-forecast", True), ("/api/open-rev", False), ("/api/dir-signals", False)]


def with_asset(path: str, coin: str) -> str:
    return path + ("&" if "?" in path else "?") + "asset=" + urllib.parse.quote(coin)


def get(base: str, path: str, timeout: float = 20.0):
    t0 = time.perf_counter()
    req = urllib.request.Request(base + path, headers={"User-Agent": "rtserve-shadow-http", "Accept-Encoding": "identity"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            st, ct, body = r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        st, ct, body = e.code, e.headers.get("Content-Type", ""), e.read()
    except OSError as e:
        st, ct, body = -1, "", repr(e).encode()
    return st, ct.split(";")[0].strip(), body, time.perf_counter() - t0


def diff(a, b, path: str = "", out: list | None = None) -> list:
    """(칸 경로, 파이썬 값, rtserve 값) 목록 · IGNORE 이름은 뺀다 · 목록은 길이가 다르면 그 자체를 한 칸으로."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            ka = [k for k in a if k not in IGNORE]
            kb = [k for k in b if k not in IGNORE]
            if ka != kb:
                out.append((path + "{키}", ka[:8], kb[:8]))
        for k in a:
            if k in IGNORE or k not in b:
                continue
            diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path + "[길이]", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[]", out)
            if len(out) > 200:
                break
    elif a != b or type(a) is not type(b):
        out.append((path or "(본문)", a, b))
    return out


def short(v) -> str:
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    return s if len(s) <= 80 else s[:77] + "..."


def pct(xs: list, q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def loopback(u: str) -> str:
    h = urllib.parse.urlparse(u).hostname or ""
    if h not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit(f"루프백 주소만: {u}")
    return u.rstrip("/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", default="http://127.0.0.1:8787")
    ap.add_argument("--rs", default="http://127.0.0.1:8790")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--rate", type=float, default=2.0, help="쪽마다 초당 요청 상한")
    ap.add_argument("--coins", default="eth,sol,xrp")
    ap.add_argument("--only", default="")
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    py, rs = loopback(a.py), loopback(a.rs)
    coins = [c for c in a.coins.split(",") if c]
    paths = []
    for p, per_coin in ROUTES + (MODEL_ROUTES if a.models else []):
        paths += [with_asset(p, c) for c in coins] if per_coin else [p]
    paths = [p for p in paths if a.only in p]
    stats = defaultdict(lambda: {"n": 0, "same": 0, "fields": Counter(), "ex": {}, "tp": [], "tr": []})
    gap = 1.0 / a.rate
    print(f"대조: {len(paths)} 주소 × {a.rounds} 회 · 쪽마다 {a.rate}/초 · 파이썬 {py} · rtserve {rs}", flush=True)
    for _ in range(a.rounds):
        for p in paths:
            t0 = time.monotonic()
            res = {}
            th = [threading.Thread(target=lambda k, b: res.__setitem__(k, get(b, p)), args=(k, b)) for k, b in (("py", py), ("rs", rs))]
            for t in th:
                t.start()
            for t in th:
                t.join()
            (sp, cp, bp, tp), (sr, cr, br, tr) = res["py"], res["rs"]
            s = stats[p]
            s["n"] += 1
            s["tp"].append(tp)
            s["tr"].append(tr)
            d = []
            if sp != sr:
                d.append(("(상태)", sp, sr))
            if cp != cr:
                d.append(("(Content-Type)", cp, cr))
            if cp == "application/json" and cr == "application/json":
                try:
                    d += diff(json.loads(bp), json.loads(br))
                except ValueError as e:
                    d.append(("(JSON 깨짐)", str(e), ""))
            elif bp != br:
                d.append(("(바이트)", f"{len(bp)}B", f"{len(br)}B"))
            if not d:
                s["same"] += 1
            for f, x, y in d:
                s["fields"][f] += 1
                s["ex"].setdefault(f, (x, y))
            time.sleep(max(0.0, gap - (time.monotonic() - t0)))
    tot_same = tot_n = 0
    for p, s in stats.items():
        tot_same += s["same"]
        tot_n += s["n"]
        ms = lambda xs, q: f"{pct(xs, q) * 1000:.1f}"  # noqa: E731
        print(f"\n{p}\n  같음 {s['same']}/{s['n']} · 시간 ms 파이썬 중앙 {statistics.median(s['tp']) * 1000:.1f}·상위1% {ms(s['tp'], .99)}"
              f" · rtserve 중앙 {statistics.median(s['tr']) * 1000:.1f}·상위1% {ms(s['tr'], .99)}")
        for f, k in s["fields"].most_common(6):
            x, y = s["ex"][f]
            print(f"  다른 칸 {f} ×{k} · 파이썬 {short(x)} · rtserve {short(y)}")
    print(f"\n합계: 같음 {tot_same}/{tot_n} (주소 {len(stats)})")
    return 0


def selftest() -> int:
    assert diff({"a": 1, "generated_at": 5}, {"a": 1, "generated_at": 6}) == []
    assert diff({"a": [1, 2]}, {"a": [1, 3]}) == [(".a[]", 2, 3)]
    assert diff({"a": [1]}, {"a": [1, 2]}) == [(".a[길이]", 1, 2)]
    assert diff({"a": 1, "b": 2}, {"b": 2, "a": 1}) == [("{키}", ["a", "b"], ["b", "a"])]
    assert diff({"x": {"age_s": 1, "v": 1.0}}, {"x": {"age_s": 2, "v": 1}}) == [(".x.v", 1.0, 1)]   # 1.0 대 1 = 글자가 다르다
    assert with_asset("/api/oi-5m?bars=12", "sol") == "/api/oi-5m?bars=12&asset=sol"
    try:
        loopback("http://10.0.0.1:8787")
        raise AssertionError("루프백 아님을 받았다")
    except SystemExit:
        pass
    print("selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
