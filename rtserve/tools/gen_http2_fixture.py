#!/usr/bin/env python3
"""rtserve 둘째 조각 정답지(2026-11-21): 나머지 파일·SQLite 엔드포인트 9 · 쓰기(메모·오답·알림 구독) · client-timing · order_request_log.

방식은 gen_http_fixture.py 와 같다(그 모듈의 가짜 전송·시계 고정·미들웨어를 가져다 쓴다): 진짜 web.Application(라우터·미들웨어)에
make_mocked_request 를 넣어 연결 바이트를 뜬다. 서버를 띄우지 않고 소켓을 열지 않는다. 대시보드 서버 모듈은 import 하지 않고 함수 본문을
옮겨 적었다(원본 행 주석). 별도 모듈(pred_record · acct_notes · push_webpush)은 파일 경로로 불러 쓴다 -- push_webpush 는 cryptography 만
끌어온다(.env·네트워크 없음 · 확인함). live_push_notifier 는 import 때 .env 를 읽어서 PUSH_KINDS 를 옮겨 적었다.
쓰기 시나리오는 요청마다 지켜보는 파일 바이트(메모·오답·구독·주문 기록)와 print 한 줄을 같이 떠 둔다. 파일 트리 = 합성.

    python3 rtserve/tools/gen_http2_fixture.py     # 저장소 루트에서 · rtserve/tests/fixtures/http2.json
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import io
import json
import math
import os
import random
import sqlite3
import statistics
import sys
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from html import unescape as html_unescape
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_http_fixture as g1  # noqa: E402  -- 시계 고정·가짜 전송·미들웨어·모듈 적재
from aiohttp import web  # noqa: E402

NOW, MTIME_NS, ROOT = g1.NOW, g1.MTIME_NS, g1.ROOT
OUT = ROOT / "rtserve" / "tests" / "fixtures" / "http2.json"
DASHBOARD_ASSETS = g1.DASHBOARD_ASSETS            # eth,sol,xrp → 흐름 엔진 = eth·sol·xrp
FLOW_SPECS = ["eth", "sol", "xrp", "btc", "sndk", "soxl", "skhynix", "mu"]   # dashboard/coin_flow.py:269 열쇠
MARKET_SYMBOLS = ["eth", "sol", "btc", "xrp", "hype", "sndk", "soxl", "skhynix", "mu"]   # orders.py:338 열쇠
TAPE_BUCKETS = {"ethusdt": 0.1, "btcusdt": 1.0, "solusdt": 0.01, "xrpusdt": 0.0001, "hypeusdt": 0.001,
                "zecusdt": 0.05, "sndkusdt": 0.05, "soxlusdt": 0.01, "skhynixusdt": 0.1, "muusdt": 0.05}   # 테이프 수집기 BUCKETS
_real_gmtime = time.gmtime
time.gmtime = lambda s=None: _real_gmtime(NOW if s is None else s)   # push_webpush 의 subscribed_utc
pred_record = g1.load("pred_record", "dashboard/pred_record.py")
acct_notes, worker_state, read_rows = g1.acct_notes, g1.worker_state, g1.read_rows
push = g1.load("push_webpush", "scripts/push_webpush_20260904.py")
from scripts.live_trade_tape_collector_20260916 import size_bands  # noqa: E402
# 원본 PUSH_KINDS(scripts/live_push_notifier_20260904.py:131) -- 그 모듈은 import 때 .env 를 읽는다
LIQ_WARN_PCT, EXPO_WARN_X = 5.0, 6.0
PUSH_KINDS = {
    "verdict": {"name": "판정 예정일", "when": "사전등록 판정일 아침 9시(KST)에 한 번"},
    "risk": {"name": "포지션 위험", "when": f"청산까지 {LIQ_WARN_PCT:g}% 미만 · 명목이 순자산의 {EXPO_WARN_X:g}배 초과로 들어설 때 한 번"},
    "burst_hold": {"name": "보유 중 청산 급증", "when": "ETH 보유 중 직전 60초 한쪽 청산이 학습 7일 상위 0.5%를 넘을 때 — 그 순간 시장가로 닫으면 평균 −17bp였다"},
}
NOCACHE = g1.NOCACHE


def make_app(R: Path, log: list) -> web.Application:
    LIVE_DIR = R / "data" / "live"
    HOT = R / "data" / "hot"
    _read_only_rows = read_rows
    flows = {a: None for a in FLOW_SPECS if a == "eth" or a in DASHBOARD_ASSETS}   # server.py:1721
    cctx = flows

    order_log = LIVE_DIR / "order_requests.jsonl"

    @web.middleware
    async def order_request_log(request, handler):                           # 1352
        if request.method != "POST" or not request.path.startswith("/api/manual-"):
            return await handler(request)
        t0, status, err = time.time(), 500, None
        try:
            response = await handler(request)
            status = response.status
            if isinstance(response, web.Response) and response.body is not None and len(response.body) < 20000:
                try:
                    j = json.loads(response.body)
                    err = None if j.get("ok", True) else (j.get("error") or "")
                except (ValueError, AttributeError):
                    pass
            return response
        except web.HTTPException as exc:
            status = exc.status
            raise
        finally:
            row = {"at": datetime.fromtimestamp(t0, ZoneInfo("Asia/Seoul")).isoformat(timespec="milliseconds"), "path": request.path,
                   "query": dict(request.query), "status": status, "error": err, "ms": round(1000 * (time.time() - t0)),
                   "ua": request.headers.get("User-Agent", "")[:200],
                   "ip": request.headers.get("CF-Connecting-IP") or request.headers.get("X-Forwarded-For") or request.remote}
            try:
                with order_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError as exc:
                print(f"order_request_log: 못 씀 {exc}", flush=True)

    app = web.Application(middlewares=[order_request_log, g1.static_asset_headers, g1.json_compress_etag])

    # ── server.py 모듈 함수(행) ──
    def file_signature(path):                                                # 1047
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def make_etag(prefix, *parts):                                           # 1055
        digest = hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:16]
        return f'W/"{prefix}-{digest}"'

    def json_response(request, payload, etag):                               # 1270
        headers = {"ETag": etag, "Cache-Control": "no-cache"}
        if g1.etag_matches(request, etag):
            return web.Response(status=web.HTTPNotModified.status_code, headers=headers)
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        response = web.Response(body=body, content_type="application/json", headers=headers)
        response.enable_compression()
        return response

    def chip_score_path(ts):                                                 # 299
        return LIVE_DIR / f"chip_score_{datetime.fromtimestamp(ts, timezone.utc):%Y%m}.jsonl"

    def chip_score_rows(now, asset):                                         # 303
        since = now - 14 * 86400
        rows = []
        for path in sorted({chip_score_path(since), chip_score_path(now)}):
            try:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        if r.get("a") == asset and float(r.get("t") or 0) >= since:
                            rows.append(r)
            except OSError:
                continue
        return rows

    def chip_score_summary(rows):                                            # 321
        out = {"days": 14}
        burst = [r for r in rows if r.get("k") == "burst"]
        if burst:
            out["burst"] = {"n": len(burst), "rev_mean_bp": sum(r["rev_bp"] for r in burst) / len(burst),
                            "rev_share": sum(r["rev_bp"] > 0 for r in burst) / len(burst)}
        return out

    VERDICT_CALENDAR_PATH = R / "dashboard" / "verdict_calendar.json"
    PUSH_SENT_LOG_PATH = LIVE_DIR / "push_sent_log.jsonl"

    def notify_center_payload():                                             # 336
        try:
            cal = json.loads(VERDICT_CALENDAR_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cal = {}
        recent = []
        try:
            with open(PUSH_SENT_LOG_PATH, encoding="utf-8") as fh:
                for line in deque(fh, maxlen=30):
                    try:
                        recent.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            pass
        items = cal.get("items") or []
        try:
            st = json.loads((LIVE_DIR / "liqmap_score_status.json").read_text(encoding="utf-8"))
            for it in items:
                if it.get("id") == "liqmap_fwd" and not it.get("done"):
                    it["what"] = f"{it.get('what', '')} · 지금 누적 {st.get('n_fwd', 0)}시간({st.get('days_fwd', 0)}일)"
        except (OSError, ValueError):
            pass
        return {"verdicts": sorted(items, key=lambda it: it.get("date") or ""),
                "kinds": [{"key": k, **v} for k, v in PUSH_KINDS.items()], "recent": recent[::-1]}

    import re

    def news_key(title):                                                     # 562
        t = re.sub(r"https?://\S+", " ", html_unescape(title or ""))
        t = re.sub(r"\[No Title\] - Post from [^\n]*", " ", t)
        t = re.sub(r"^[^:\n]{1,60}\(@\w+\):\s*", "", t.strip())
        t = re.sub(r"^(?:[A-Z0-9][A-Z0-9 .'&-]{0,30}|FinancialJuice|Tree News|AggrNews|BWEnews|Binance EN):\s*", "", t)
        t = re.sub(r"[\W_]+", "", t.lower())
        return t[:80] if len(t) >= 12 else ""

    def news_dedupe(items):                                                  # 575
        out = {}
        for it in sorted(items, key=lambda x: x["ts_ms"]):
            k = news_key(it["title"]) or f"{it['source']}|{it['link']}"
            hit = out.get(k)
            if hit is None:
                out[k] = {**it, "sources": [it["source"]]}
                continue
            if it["source"] not in hit["sources"]:
                hit["sources"].append(it["source"])
            if not hit.get("sentiment") and it.get("sentiment"):
                hit.update({f: it[f] for f in ("sentiment", "p_bull", "p_bear", "p_neu", "asset", "impact", "incident",
                                               "relevant", "judged_ms", "model")})
        return sorted(out.values(), key=lambda x: x["ts_ms"], reverse=True)

    NEWS_DB = HOT / "news_rss.sqlite"

    def news_payload(now, db=NEWS_DB):                                       # 729
        since = int((now - 12 * 3600) * 1000)
        try:
            rows = _read_only_rows(db, """SELECT i.source, i.link, i.title, coalesce(i.pub_ms, i.first_seen_ms), i.first_seen_ms, i.backfill,
                j.sentiment, j.p_bull, j.p_bear, j.p_neu, j.asset, j.impact, j.incident, j.relevant, j.judged_ms, j.model
                FROM items i LEFT JOIN judgments j ON j.source = i.source AND j.link = i.link
                WHERE coalesce(i.pub_ms, i.first_seen_ms) >= ? ORDER BY coalesce(i.pub_ms, i.first_seen_ms) DESC LIMIT 1500""", [since])
        except (sqlite3.Error, OSError):
            try:
                rows = [(*r, None, None, None, None, None, None, None, None, None, None) for r in _read_only_rows(db,
                    """SELECT source, link, title, coalesce(pub_ms, first_seen_ms), first_seen_ms, backfill FROM items
                    WHERE coalesce(pub_ms, first_seen_ms) >= ? ORDER BY coalesce(pub_ms, first_seen_ms) DESC LIMIT 1500""", [since])]
            except (sqlite3.Error, OSError):
                return {"ok": False, "items": [], "window_h": 12, "now_ms": int(now * 1000)}
        keys = ("source", "link", "title", "ts_ms", "seen_ms", "backfill", "sentiment", "p_bull", "p_bear", "p_neu",
                "asset", "impact", "incident", "relevant", "judged_ms", "model")
        items = news_dedupe([{**dict(zip(keys, r)), "title": html_unescape(r[2] or "")} for r in rows])
        return {"ok": True, "items": items, "window_h": 12, "now_ms": int(now * 1000), "model": "jevk5:4b",
                "last_judged_ms": max((it["judged_ms"] or 0 for it in items if it["sentiment"]), default=0) or None}

    TREASURY_DB, XYZ_DB, XYZ_COIN = HOT / "treasury_futures.sqlite", HOT / "hl_xyz.sqlite", "xyz:XYZ100"
    SPIKE_Z, SPIKE_BASE, SPIKE_MIN = 4.0, 5400, 288

    def spikes_5m(rows, keep_s=3600):                                        # 678
        if not rows:
            return {"events": [], "n": 0}
        last = {}
        for t, c in rows:
            last[int(t) // 300 * 300] = float(c)
        max_t = int(rows[-1][0])
        bs = sorted(b for b in last if b + 300 <= max_t)
        rets = [(b, math.log(last[b] / last[a]) * 1e4) for a, b in zip(bs, bs[1:]) if b - a == 300 and last[a] > 0 and last[b] > 0]
        events = []
        for i in range(len(rets) - 1, -1, -1):
            b, r = rets[i]
            if b + 300 < max_t - keep_s:
                break
            base = [x for _, x in rets[max(0, i - SPIKE_BASE):i]]
            sd = statistics.stdev(base) if len(base) >= SPIKE_MIN else 0.0
            if sd > 0 and abs(r / sd) >= SPIKE_Z:
                events.append([b, round(r / sd, 2), round(math.expm1(r / 1e4) * 100, 3)])
        return {"events": events[::-1], "n": len(rets)}

    _NY = ZoneInfo("America/New_York")

    def _cme_open(ts):                                                       # 706
        d = datetime.fromtimestamp(ts, _NY)
        return not (d.hour == 17 or d.weekday() == 5 or (d.weekday() == 6 and d.hour < 18) or (d.weekday() == 4 and d.hour >= 17))

    def macro_spikes_payload():                                              # 712
        out = {}
        q = """SELECT ts_sec, close FROM {t} WHERE {k} = ? AND close IS NOT NULL
               AND ts_sec >= (SELECT max(ts_sec) FROM {t} WHERE {k} = ?) - 8 * 86400 ORDER BY ts_sec"""
        for nm, db, t, k, key in (("ZF", TREASURY_DB, "treasury_fut_1m", "symbol", "ZF"), ("ZN", TREASURY_DB, "treasury_fut_1m", "symbol", "ZN"),
                                  ("NQ", XYZ_DB, "hl_xyz_1m", "coin", XYZ_COIN)):
            try:
                rows = _read_only_rows(db, q.format(t=t, k=k), [key, key])
                if nm == "NQ":
                    rows = [r for r in rows if _cme_open(int(r[0]))]
                out[nm] = spikes_5m(rows)
            except (sqlite3.Error, OSError):
                continue
        return {"ok": bool(out), "z": SPIKE_Z, "symbols": out, "now_sec": int(time.time())}

    HOT_TAPE_DB, OKX_TAPE_DB_PATH, BYBIT_TAPE_DB = HOT / "binance_tape.sqlite", HOT / "okx_tape.sqlite", HOT / "bybit_tape.sqlite"

    def bubble_tape_payload(symbol, sec, db=HOT_TAPE_DB, now=None, since=None, okx_db=None, bybit_db=None):   # 1071
        end = int(now if now is not None else time.time())
        lo = end - sec - 5
        if since is not None:
            lo = max(lo, since + 1)
        q = """SELECT ts_sec, price_bin, buy_qty, sell_qty, coalesce(whale_buy_qty, 0) + coalesce(whale_sell_qty, 0)
               FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ?"""
        base = symbol[:-4].upper()
        srcs = [("binance", db, symbol), ("okx", okx_db if okx_db is not None else OKX_TAPE_DB_PATH, f"{base}-USDT-SWAP"),
                ("bybit", bybit_db if bybit_db is not None else BYBIT_TAPE_DB, f"{base}USDT")]
        acc = {}
        venues = []
        for name, path, key in srcs:
            if name != "binance" and not Path(path).exists():
                continue
            try:
                got = _read_only_rows(path, q, [key, lo])
            except (sqlite3.Error, OSError):
                if name == "binance":
                    raise
                continue
            venues.append(name)
            for t, b, x, y, w in got:
                a = acc.setdefault((int(t), int(b)), [0.0, 0.0, 0.0])
                a[0] += float(x or 0); a[1] += float(y or 0); a[2] += float(w or 0)
        return {"symbol": symbol, "bucket": TAPE_BUCKETS[symbol], "lo": lo, "end": end, "venues": venues,
                "whale_usd": size_bands(symbol)[1],
                "rows": [[t, b, round(v[0], 6), round(v[1], 6), round(v[2], 6)] for (t, b), v in sorted(acc.items())]}

    async def api_bubble_tape(request):                                      # 1107 (1초 캐시 빼고)
        asset = str(request.query.get("asset") or "eth").lower()
        symbol = f"{asset}usdt"
        try:
            sec = int(request.query.get("sec", str(1800)))
            since = int(request.query["since"]) if request.query.get("since") else None
        except ValueError:
            raise web.HTTPBadRequest(reason="bad_sec_or_since") from None
        if symbol not in TAPE_BUCKETS or not 10 <= sec <= 1800:
            raise web.HTTPBadRequest(reason="unsupported_asset_or_window")
        try:
            return web.json_response(bubble_tape_payload(symbol, sec, since=since), headers=NOCACHE)
        except (sqlite3.Error, OSError) as exc:
            return web.json_response({"error": type(exc).__name__, "rows": []}, status=503, headers=NOCACHE)

    def gex_payload():                                                       # 1208
        out = worker_state.worker_payload(LIVE_DIR / "deribit_gex_state.json", 35.0, ts_field="generated_at",
                                          stamp_available=True, bare_missing=True, extra_missing={"currencies": {}})
        try:
            out.setdefault("currencies", {}).update(json.loads((LIVE_DIR / "cboe_option_state.json").read_text(encoding="utf-8"))["currencies"])
        except (OSError, ValueError, KeyError):
            pass
        blk = worker_state.worker_payload(LIVE_DIR / "deribit_block_trades_state.json", 3.0, ts_field="generated_at",
                                          stamp_available=True, bare_missing=True, extra_missing={"blocks": []})
        by_coin = {c: [] for c in ("ETH", "BTC", "SOL", "XRP")}
        by_coin.update(blk.get("blocks_by_coin") or {"ETH": blk.get("blocks") or []})
        return {**out, "block_trades": {"available": bool(blk.get("available")), "error": blk.get("error"),
                                        "by_coin": {c: (v or [])[:8] for c, v in by_coin.items()},
                                        "sum_by_coin": {c: sum(b.get("notional_usd") or 0 for b in (v or [])) for c, v in by_coin.items()},
                                        "max_by_coin": {c: max(v, key=lambda b: b.get("notional_usd") or 0) if v else None for c, v in by_coin.items()},
                                        "n_by_coin": blk.get("n_blocks_by_coin") or {c: len(v or []) for c, v in by_coin.items()},
                                        "flow_by_coin": blk.get("flow_by_coin") or {}}}

    def breakout_detector_payload(asset="eth"):                              # 1156
        return worker_state.worker_payload(LIVE_DIR / f"{asset}_breakout_detector_state.json", 15.0,
                                           require_ok=True, stamp_available=True, extra_missing={"history": [], "times": []})

    json_cache = {}

    def load_json_cached(path, signature=None):                              # 1534
        if signature is None:
            signature = file_signature(path)
        cached = json_cache.get(path)
        if cached and cached[0] == signature:
            return cached[1]
        payload = worker_state.load_json(path)
        json_cache[path] = (signature, payload)
        return payload

    async def api_state(request):                                            # 3647 · 1544
        state_path = LIVE_DIR / "dashboard_state.json"
        state_sig = file_signature(state_path)
        etag = make_etag("state", state_sig)
        return json_response(request, {"state": load_json_cached(state_path, state_sig)}, etag)

    async def api_breakout_detector(request):                                # 4456
        asset = str(request.query.get("asset") or "eth").lower()
        if asset not in ("eth", "sol", "xrp"):
            raise web.HTTPNotFound(reason="breakout_off")
        return web.json_response({**breakout_detector_payload(asset), "asset": asset}, headers=NOCACHE)

    async def api_gex(request):                                              # 4715
        return web.json_response(gex_payload(), headers=NOCACHE)

    async def api_chip_score(request):                                       # 2088
        asset = str(request.query.get("asset") or "eth").lower()
        if asset not in cctx:
            raise web.HTTPNotFound(reason="flow_off")
        return web.json_response(chip_score_summary(chip_score_rows(time.time(), asset)), headers=NOCACHE)

    async def api_notify_center(request):                                    # 2134
        return web.json_response(notify_center_payload(), headers=NOCACHE)

    async def api_news(request):                                             # 2096
        return web.json_response(news_payload(time.time()), headers=NOCACHE)

    async def api_macro_spikes(request):                                     # 2130
        return web.json_response(macro_spikes_payload(), headers=NOCACHE)

    async def api_pred_record(request):                                      # 2890
        try:
            payload = pred_record.summary(int(time.time() * 1000), LIVE_DIR / "pred_record", LIVE_DIR / "regime_band_forward_scored.jsonl",
                                          LIVE_DIR / "pred_record_backfill")
        except Exception as exc:  # noqa: BLE001
            payload = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
        return web.json_response(payload, headers=NOCACHE)

    ACCT_NOTES_PATH, ACCT_WRONG_PATH = LIVE_DIR / "acct_notes.json", LIVE_DIR / "acct_wrong.json"

    async def api_acct_notes_save(request):                                  # 2869
        try:
            body = await request.json()
            return web.json_response(acct_notes.save(ACCT_NOTES_PATH, body.get("date"), body.get("text")), headers=NOCACHE)
        except (ValueError, AttributeError) as exc:
            raise web.HTTPBadRequest(text=str(exc))

    async def api_acct_wrong_save(request):                                  # 2881
        try:
            body = await request.json()
            if "del" in body:
                return web.json_response(acct_notes.wrong_del(ACCT_WRONG_PATH, body["del"]), headers=NOCACHE)
            return web.json_response(acct_notes.wrong_add(ACCT_WRONG_PATH, body.get("date"), body.get("text"), body.get("tag", "")), headers=NOCACHE)
        except (ValueError, AttributeError, TypeError) as exc:
            raise web.HTTPBadRequest(text=str(exc))

    SUBS = LIVE_DIR / "push_subscriptions.json"
    VAPID = {"VAPID_PUBLIC_KEY": "BPUBLIC-test-key", "VAPID_PRIVATE_KEY": "set"}   # 가짜 값(시험 전용) -- 원본은 os.getenv

    async def api_push_config(request):                                      # 4856
        public_key = VAPID.get("VAPID_PUBLIC_KEY", "")
        return web.json_response({"enabled": bool(public_key and VAPID.get("VAPID_PRIVATE_KEY")), "vapid_public_key": public_key,
                                  "subscriber_count": len(push.load_subscriptions(SUBS))}, headers=NOCACHE)

    async def api_push_subscribe(request):                                   # 4868
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest(text="invalid json")
        sub = body.get("subscription") or {}
        if not sub.get("endpoint") or not (sub.get("keys") or {}).get("p256dh"):
            raise web.HTTPBadRequest(text="subscription must carry endpoint and keys.p256dh")
        sid = push.add_subscription(sub, label=str(body.get("label", ""))[:80], path=SUBS)
        return web.json_response({"ok": True, "id": sid}, headers=NOCACHE)

    async def api_push_unsubscribe(request):                                 # 4879
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest(text="invalid json")
        sub = body.get("subscription") or {}
        sid = body.get("id") or (push.subscription_id(sub) if sub.get("endpoint") else None)
        if not sid:
            raise web.HTTPBadRequest(text="id or subscription.endpoint required")
        return web.json_response({"ok": push.remove_subscription(sid, path=SUBS)}, headers=NOCACHE)

    async def api_push_devices(request):                                     # 4891
        rows = []
        for sid, sub in push.load_subscriptions(SUBS).items():
            endpoint = str(sub.get("endpoint", ""))
            rows.append({"id": sid, "label": sub.get("label") or "", "subscribed_utc": sub.get("subscribed_utc"), "endpoint_tail": endpoint[-12:]})
        rows.sort(key=lambda r: r.get("subscribed_utc") or "")
        return web.json_response({"devices": rows}, headers=NOCACHE)

    CT_NUM = {"order_rtt_ms", "order_srv_ms", "order_net_ms", "fp_n", "fp_med_ms", "fp_max_ms", "longtask_n", "px_n", "px_med_ms", "px_max_ms",
              # main 4ff90f88(화면 그림자 구독)이 더한 9칸
              "rt_lat_n", "rt_lat_med_ms", "rt_lat_p99_ms", "rt_lead_n", "rt_lead_med_ms", "rt_cmp_n", "rt_bad_n", "rt_fail_n", "rt_reconn_n"}
    _ct_at = [0.0]

    async def api_client_timing(request):                                    # 5102
        raw = await request.content.read(1025)
        if len(raw) > 1024:
            return web.json_response({"ok": False, "error": "too_large"}, status=413)
        try:
            body = json.loads(raw)
        except (ValueError, RecursionError):
            return web.json_response({"ok": False, "error": "bad_json"}, status=400)
        nums = {k: v for k, v in body.items() if k in CT_NUM} if isinstance(body, dict) else {}
        if (not nums or set(body) - CT_NUM - {"sym", "kind"} or str(body.get("sym")) not in MARKET_SYMBOLS
                or body.get("kind", "entry") not in ("entry", "exit")
                or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e7 for v in nums.values())):
            return web.json_response({"ok": False, "error": "bad_body"}, status=400)
        if time.monotonic() - _ct_at[0] < 0.2:
            return web.json_response({"ok": False, "error": "too_often"}, status=429)
        _ct_at[0] = time.monotonic()
        print(f"client_timing sym={body['sym']}" + (f" kind={body['kind']}" if "kind" in body else "")
              + "".join(f" {k}={round(v)}" for k, v in sorted(nums.items())), flush=True)
        return web.json_response({"ok": True}, headers=NOCACHE)

    for path, h in (("/api/state", api_state), ("/api/breakout-detector", api_breakout_detector), ("/api/gex", api_gex),
                    ("/api/bubble-tape", api_bubble_tape), ("/api/chip-score", api_chip_score), ("/api/notify-center", api_notify_center),
                    ("/api/news", api_news), ("/api/macro-spikes", api_macro_spikes), ("/api/pred-record", api_pred_record),
                    ("/api/push/config", api_push_config), ("/api/push/devices", api_push_devices)):
        app.router.add_get(path, h)
    async def api_acct_notes(request):                                       # 2866
        return web.json_response({"notes": acct_notes.load(ACCT_NOTES_PATH)}, headers=NOCACHE)

    async def api_acct_wrong(request):                                       # 2878
        return web.json_response({"rows": acct_notes.wrong_load(ACCT_WRONG_PATH)}, headers=NOCACHE)

    app.router.add_get("/api/acct-notes", api_acct_notes)
    app.router.add_post("/api/acct-notes", api_acct_notes_save)
    app.router.add_get("/api/acct-wrong", api_acct_wrong)
    app.router.add_post("/api/acct-wrong", api_acct_wrong_save)
    app.router.add_post("/api/push/subscribe", api_push_subscribe)
    app.router.add_post("/api/push/unsubscribe", api_push_unsubscribe)
    app.router.add_post("/api/client-timing", api_client_timing)
    app.router.add_static("/dashboard/live/", R / "dashboard" / "live", show_index=False)
    return app


# ───────────────────────── 파일 트리 ─────────────────────────
def j(o, **kw) -> bytes:
    return json.dumps(o, **kw).encode()


def utc(t):
    return datetime.fromtimestamp(t, timezone.utc)


def tree_api2() -> dict[str, bytes]:
    rnd = random.Random(7)
    m = int(NOW) // 60 * 60
    # 국채 ZF·ZN 1분봉 3일 + 마지막 1시간 안 급변 · NQ(xyz) 1분봉 -- 주말·CME 휴장 시간 포함(거르기 확인)
    tre, xyz = [], []
    for k, sym in enumerate(("ZF", "ZN")):
        px = 108.0 + k
        for i in range(3 * 1440, 0, -1):
            px *= math.exp(rnd.gauss(0, 0.0002))
            if i == 20 and sym == "ZF":
                px *= 1.004                         # 큰 급변(z ≥ 4)
            if i == 40 and sym == "ZN":
                px *= 1.0016                        # 중간 급변(3 ≤ z < 4 -- 문턱 4 를 지키는지)
            tre.append((sym, m - i * 60, round(px, 6), f"{sym}Z6"))
    px = 25000.0
    for i in range(6 * 1440, 0, -1):
        px *= math.exp(rnd.gauss(0, 0.0003))
        if i == 30:
            px *= 0.993
        xyz.append(("xyz:XYZ100", m - i * 60, round(px, 3)))
    sq = g1.sqlite_bytes
    tre_db = sq([("CREATE TABLE treasury_fut_1m (symbol TEXT, ts_sec INTEGER, close REAL, contract TEXT)", []),
                 ("INSERT INTO treasury_fut_1m VALUES (?,?,?,?)", tre)])
    xyz_db = sq([("CREATE TABLE hl_xyz_1m (coin TEXT, ts_sec INTEGER, close REAL)", []), ("INSERT INTO hl_xyz_1m VALUES (?,?,?)", xyz)])
    # 체결 테이프(바이낸스·OKX · Bybit 없음)
    tape_ddl = "CREATE TABLE trade_tape_1s (symbol TEXT, ts_sec INTEGER, price_bin INTEGER, buy_qty REAL, sell_qty REAL, whale_buy_qty REAL, whale_sell_qty REAL)"
    bn = [("ethusdt", int(NOW) - s, 30000 + s % 7, 0.1234567 * (s % 5) or None, 0.37 * (s % 3), 2.5 if s % 11 == 0 else None, None) for s in range(0, 2000, 3)]
    bn += [("xrpusdt", int(NOW) - 5, 21000, 1000.0, 3.0, None, 7.0), ("solusdt", int(NOW) - 4000, 1, 1.0, 1.0, 0.0, 0.0)]
    okx = [("ETH-USDT-SWAP", int(NOW) - s, 30000 + s % 7, 0.01 * s, None, None, 1.25) for s in range(0, 600, 5)]
    news_ddl = [("CREATE TABLE items (source TEXT, link TEXT, title TEXT, pub_ms INTEGER, first_seen_ms INTEGER, backfill INTEGER)", []),
                ("CREATE TABLE judgments (source TEXT, link TEXT, sentiment TEXT, p_bull REAL, p_bear REAL, p_neu REAL, asset TEXT, impact TEXT, incident TEXT, relevant INTEGER, judged_ms INTEGER, model TEXT)", [])]
    t0 = int(NOW * 1000)
    items = [("Tree News", "l1", "Tree News: Trump says &amp; tariffs on crypto coming soon https://t.co/x", t0 - 60_000, t0 - 50_000, 0),
             ("Truth", "l2", "Donald J. Trump (@realDonaldTrump): Trump says & tariffs on crypto coming soon", None, t0 - 70_000, 0),
             ("COINDESK", "l3", "COINDESK: Bitcoin ETF flows hit record &#8212; 3rd week &nbsp;in a row", t0 - 3_600_000, t0 - 3_500_000, 0),
             ("Decrypt", "l4", "DECRYPT: Bitcoin ETF flows hit record — 3rd week  in a row", t0 - 3_000_000, t0 - 2_900_000, 1),
             ("FinancialJuice", "l5", "FinancialJuice: Powell: rates stay &lt;higher&gt; for longer &notanentity; &amp", t0 - 100_000, t0 - 90_000, 0),
             ("BWEnews", "l6", "short", t0 - 200_000, t0 - 190_000, 0),
             ("Truth", "l7", "[No Title] - Post from 11/18/2026", t0 - 300_000, t0 - 290_000, 0),
             ("Old", "l8", "아주 오래된 소식 열세 글자 넘게 씀", t0 - 20 * 3600_000, t0 - 20 * 3600_000, 0),
             ("AggrNews", "l9", "AggrNews: 한국 거래소 «상장» 소식 &#x1F680; 열두 글자 넘김", t0 - 400_000, t0 - 400_000, None),
             ("Tree News", "l10", None, t0 - 500_000, t0 - 500_000, 0)]
    judg = [("Truth", "l2", "bullish", 0.7, 0.1, 0.2, "BTC", "high", None, 1, t0 - 40_000, "jevk5:4b"),
            ("COINDESK", "l3", None, None, None, None, None, None, None, None, t0 - 1000, "skip:empty"),
            ("Decrypt", "l4", "neutral", 0.3, 0.3, 0.4, "ETH", "low", "x", 0, t0 - 2_000_000, "jevk5:4b")]
    news_db = sq(news_ddl + [("INSERT INTO items VALUES (?,?,?,?,?,?)", items), ("INSERT INTO judgments VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", judg)])
    month = utc(NOW).strftime("%Y%m")
    chips = "".join(json.dumps(r) + "\n" for r in [
        {"k": "burst", "a": "eth", "t": NOW - 100, "rev_bp": 3.5}, {"k": "burst", "a": "eth", "t": NOW - 200, "rev_bp": -1},
        {"k": "burst", "a": "eth", "t": str(NOW - 300), "rev_bp": 2}, {"k": "bo", "a": "eth", "t": NOW - 5},
        {"k": "burst", "a": "eth", "t": NOW - 20 * 86400, "rev_bp": 9}, {"k": "burst", "a": "sol", "t": NOW - 5, "rev_bp": 1.25}]) + "{broken\r\n"
    cal = {"items": [{"id": "b", "date": "2026-12-01", "title": "나중"}, {"id": "liqmap_fwd", "date": "2026-11-30", "what": "청산지도 전진"},
                     {"id": "a", "title": "날짜 없음"}, {"id": "c", "date": "2026-11-20", "done": True}]}
    sent = "".join(json.dumps({"i": k, "t": "알림"}, ensure_ascii=False) + "\n" for k in range(40)) + "not json\n"
    gen_at = lambda m_ago: utc(NOW - m_ago * 60).strftime("%Y-%m-%dT%H:%M:%S")
    gex = {"generated_at": gen_at(5), "ok": True, "currencies": {"ETH": {"total": 1.5e9}, "BTC": {"total": 3}}}
    cboe = {"currencies": {"SNDK": {"total": 12.5}, "ETH": {"total": "cboe"}}}
    blocks = {"generated_at": gen_at(1) + "+00:00", "blocks_by_coin": {"ETH": [{"notional_usd": 1e6 + k, "id": k} for k in range(10)] + [{"notional_usd": None}],
                                                                       "BTC": [{"notional_usd": 5}, {"notional_usd": 5.0, "id": "second"}], "HYPE": []},
              "flow_by_coin": {"ETH": {"net": -3}}}
    state = {"session": {"name": "아시아", "open": True}, "microstructure": {"x": 1.5e-07}, "tail_risk": None, "other": [1, 2]}
    # 적중 기록: 실시간 + 재구성 + 상대 강도(9코인 4h 격자) + 띠 채점
    BAR = 300_000
    tb = (t0 - 3 * 86_400_000) // BAR * BAR
    pr = [{"k": "s", "t": tb + i * BAR, "px": 3000 + (i % 13) * 1.5 - (i % 7), "a": [95.0, 50.0, 10.0, 80][i % 4], "b": [15.0, 85.0, 50.0][i % 3]} for i in range(0, 700)]
    pr += [{"k": "4", "t": tb + i * BAR, "px": 3000.0 + i, "d": 3.2, "s": [1, -1, 0][i % 3]} for i in range(0, 700, 2)]
    tg = (t0 - 25 * 60_000 - 3 * BAR) // BAR * BAR      # 지평(15분)이 25분 전에 지났는데 종가가 없는 결정 = 대기(구멍 문턱 30분 안)
    pr += [{"k": "s", "t": tg, "px": 3001.0, "a": 95.0, "b": 50.0}]
    bfr = [{"k": "s", "t": tb - i * BAR, "px": 2990.0 + i, "a": 92.0, "b": 50.0, "bf": 1} for i in range(1, 40)] + [{"k": "s", "t": tb, "px": 1.0, "a": 5.0, "b": 5.0, "bf": 1}]
    step = 4 * 3600 * 1000
    xb = (t0 - 2 * 86_400_000) // step * step
    # 코인 둘씩 같은 가격 = 순위 동점(평균 순위)
    xs = [{"k": "x", "t": xb + i * step, "px": {c: 100 + (((k // 2) * 7 + i * 3) % 11) * (1 if i % 2 else -1) * 0.5 for k, c in enumerate(pred_record.X_COINS)}} for i in range(9)]
    xs += [{"k": "x", "t": xb + 9 * step, "px": {c: 100.0 for c in pred_record.X_COINS}}]
    scored = [{"coin": c, "ts": f"2026-11-{d:02d} {h:02d}:02:00", "correct": (d + h) % 2 == 0, "y": h % 3 == 0} for c in ("ETH", "sol") for d in (14, 15, 16) for h in (1, 9, 17)]
    return {
        "data/hot/treasury_futures.sqlite": tre_db, "data/hot/hl_xyz.sqlite": xyz_db,
        "data/hot/binance_tape.sqlite": sq([(tape_ddl, []), ("INSERT INTO trade_tape_1s VALUES (?,?,?,?,?,?,?)", bn)]),
        "data/hot/okx_tape.sqlite": sq([(tape_ddl, []), ("INSERT INTO trade_tape_1s VALUES (?,?,?,?,?,?,?)", okx)]),
        "data/hot/news_rss.sqlite": news_db,
        f"data/live/chip_score_{month}.jsonl": chips.encode(),
        "dashboard/verdict_calendar.json": j(cal, ensure_ascii=False),
        "data/live/push_sent_log.jsonl": sent.encode(),
        "data/live/liqmap_score_status.json": j({"n_fwd": 123, "days_fwd": 5.5}),
        "data/live/deribit_gex_state.json": j(gex), "data/live/cboe_option_state.json": j(cboe),
        "data/live/deribit_block_trades_state.json": j(blocks),
        "data/live/dashboard_state.json": j(state, ensure_ascii=False),
        "data/live/eth_breakout_detector_state.json": j({"updated_utc": utc(NOW - 3600).strftime("%Y-%m-%d %H:%M:%S"), "ok": True, "rvol": 1.2}),
        "data/live/xrp_breakout_detector_state.json": j({"updated_utc": utc(NOW - 60).isoformat(), "ok": False, "err": "x"}),
        "data/live/pred_record/ETH.jsonl": "".join(json.dumps(r) + "\n" for r in pr).encode() + b"{bad\n",
        "data/live/pred_record_backfill/ETH.jsonl": "".join(json.dumps(r) + "\n" for r in bfr).encode(),
        "data/live/pred_record/xsec.jsonl": "".join(json.dumps(r) + "\n" for r in xs).encode(),
        "data/live/regime_band_forward_scored.jsonl": "".join(json.dumps(r) + "\n" for r in scored).encode(),
    }


def tree_api2_variants() -> dict[str, bytes]:
    """없음·깨짐·낡음 갈래 -- 바이낸스 테이프 없음(503) · 판정 표 없는 뉴스(대신 기사만) · gex 낡음 · 블록 없음."""
    sq = g1.sqlite_bytes
    t0 = int(NOW * 1000)
    news_db = sq([("CREATE TABLE items (source TEXT, link TEXT, title TEXT, pub_ms INTEGER, first_seen_ms INTEGER, backfill INTEGER)", []),
                  ("INSERT INTO items VALUES (?,?,?,?,?,?)", [("S", "a", "같은 제목이 두 출처로 들어온 소식입니다", t0 - 1000, t0 - 1000, 0),
                                                              ("T", "b", "같은 제목이 두 출처로 들어온 소식입니다", t0 - 2000, t0 - 2000, 0)])])
    return {
        "data/hot/news_rss.sqlite": news_db,
        "data/live/deribit_gex_state.json": j({"generated_at": utc(NOW - 3600).strftime("%Y-%m-%d %H:%M:%S"), "currencies": {"ETH": 1}}),
        "data/live/cboe_option_state.json": b"{broken",
        "data/live/dashboard_state.json": b"{}",
        "data/live/sol_breakout_detector_state.json": b"[]",
        "dashboard/verdict_calendar.json": b"{broken",
        "data/live/pred_record/ETH.jsonl": b"",
    }


REQ_API2 = [("GET", p, {}) for p in (
    "/api/state", "/api/breakout-detector", "/api/breakout-detector?asset=SOL", "/api/breakout-detector?asset=xrp", "/api/breakout-detector?asset=btc",
    "/api/gex", "/api/chip-score", "/api/chip-score?asset=SOL", "/api/chip-score?asset=btc", "/api/notify-center", "/api/news", "/api/macro-spikes",
    "/api/bubble-tape", "/api/bubble-tape?asset=XRP&sec=60", f"/api/bubble-tape?since={int(NOW) - 40}", "/api/bubble-tape?asset=sol&sec=10",
    "/api/bubble-tape?sec=5", "/api/bubble-tape?sec=abc", "/api/bubble-tape?asset=doge", "/api/bubble-tape?since=x", "/api/bubble-tape?asset=btc",
    "/api/pred-record")] + [("GET", "/api/state", {"Accept-Encoding": "gzip, deflate"})]
REQ_API2_VAR = [("GET", p, {}) for p in ("/api/state", "/api/breakout-detector", "/api/breakout-detector?asset=sol", "/api/gex", "/api/news",
                                         "/api/macro-spikes", "/api/bubble-tape", "/api/notify-center", "/api/chip-score", "/api/pred-record")]

JS = {"Content-Type": "application/json"}
EP1, EP2 = "https://push.example.invalid/send/AAAA-device-one-token", "https://push.example.invalid/send/BBBB-device-two-token"
EP3 = "https://push.example.invalid/send/CCCC-device-old-token"
WRITES_TREE = {"data/live/push_subscriptions.json": json.dumps({hashlib.sha256(EP3.encode()).hexdigest()[:16]: {
    "endpoint": EP3, "keys": {"p256dh": "old"}, "label": "옛 기기", "subscribed_utc": "2026-10-01T00:00:00Z"}}, ensure_ascii=False, indent=2).encode()}
REQ_WRITES = [
    ("POST", "/api/acct-notes", JS, {"date": "2026-11-20", "text": "메모 «한글» \"따옴표\"\n둘째 줄"}),
    ("POST", "/api/acct-notes", JS, {"date": "2026-11-19", "text": "x"}),
    ("POST", "/api/acct-notes", JS, {"date": "2026-11-20", "text": "고친 메모"}),
    ("POST", "/api/acct-notes", JS, {"date": "2026-11-19", "text": "  \n "}),
    ("POST", "/api/acct-notes", JS, {"date": "2026-1-19", "text": "x"}),
    ("POST", "/api/acct-notes", JS, {"date": "2026-11-19", "text": 5}),
    ("POST", "/api/acct-notes", JS, [1, 2]),
    ("GET", "/api/acct-notes", {}, None),
    ("POST", "/api/acct-wrong", JS, {"date": "2026-11-20", "text": "  가짜   돌파\n따라 들어감 ", "tag": "가짜 돌파"}),
    ("POST", "/api/acct-wrong", JS, {"date": "2026-11-21", "text": "둘째"}),
    ("POST", "/api/acct-wrong", JS, {"date": "2026-11-21", "text": "x", "tag": "없는 꼬리표"}),
    ("POST", "/api/acct-wrong", JS, {"date": "2026-11-21", "text": "   "}),
    ("POST", "/api/acct-wrong", JS, {"del": int(NOW * 1000)}),
    ("POST", "/api/acct-wrong", JS, {"del": True}),
    ("POST", "/api/acct-wrong", JS, {"del": 12345}),
    ("GET", "/api/acct-wrong", {}, None),
    ("GET", "/api/push/config", {}, None),
    ("POST", "/api/push/subscribe", JS, {"subscription": {"endpoint": EP1, "keys": {"p256dh": "P1", "auth": "A1"}}, "label": "휴대폰 «주»"}),
    ("POST", "/api/push/subscribe", JS, {"subscription": {"endpoint": EP2, "keys": {"p256dh": "P2"}}}),
    ("POST", "/api/push/subscribe", JS, {"subscription": {"endpoint": EP1, "keys": {"p256dh": "P1b"}}}),   # 같은 기기 다시: 이름·시각은 그대로
    ("POST", "/api/push/subscribe", JS, {"subscription": {"endpoint": EP2}}),
    ("POST", "/api/push/subscribe", JS, {"subscription": {"endpoint": EP3, "keys": {"p256dh": "P3"}}}),   # 전에 있던 기기: 옛 시각·이름 유지
    ("POST", "/api/push/subscribe", {}, "not json"),
    ("GET", "/api/push/config", {}, None),
    ("GET", "/api/push/devices", {}, None),
    ("POST", "/api/push/unsubscribe", JS, {"subscription": {"endpoint": EP2}}),
    ("POST", "/api/push/unsubscribe", JS, {"id": "nope"}),
    ("POST", "/api/push/unsubscribe", JS, {}),
    ("POST", "/api/push/unsubscribe", JS, {"id": 5}),
    ("GET", "/api/push/devices", {}, None),
    ("POST", "/api/client-timing", JS, {"sym": "eth", "kind": "exit", "order_rtt_ms": 41.5, "fp_n": 230, "px_med_ms": 2.5}),
    ("POST", "/api/client-timing", JS, {"sym": "eth", "fp_n": 1}),
    ("POST", "/api/client-timing", JS, {"sym": "sol", "rt_lat_n": 7, "rt_lat_p99_ms": 31.5}),   # main 의 rt_* 칸 = 검사 통과 → 0.2초 상한(429) · 머지 대기 2칸은 넣지 않는다
    ("POST", "/api/client-timing", JS, {"sym": "doge", "fp_n": 1}),
    ("POST", "/api/client-timing", JS, {"sym": "eth", "fp_n": True}),
    ("POST", "/api/client-timing", JS, {"sym": "eth", "fp_n": 1e8}),
    ("POST", "/api/client-timing", JS, {"sym": "eth", "x": 1, "fp_n": 1}),
    ("POST", "/api/client-timing", JS, {"sym": "eth"}),
    ("POST", "/api/client-timing", JS, "{bad"),
    ("POST", "/api/client-timing", JS, "[" + "1," * 600 + "1]"),
    ("POST", "/api/manual-entry/submit?sym=eth&x=1&x=2&q=%ED%95%9C", {"User-Agent": "Mozilla/5.0 " + "x" * 250, "CF-Connecting-IP": "203.0.113.7"}, None),
    ("POST", "/api/manual-exit/submit", {"X-Forwarded-For": "198.51.100.2"}, None),
    ("POST", "/api/manual-exit/submit", {"X-Forwarded-For": "198.51.100.2", "CF-Connecting-IP": "203.0.113.9"}, None),
    ("POST", "/api/manual-bracket/tp", {"CF-Connecting-IP": ""}, b""),
    ("GET", "/api/manual-entry/status", {}, None),
    ("PUT", "/api/acct-notes", {}, None),
    ("GET", "/api/client-timing", {}, None),
]
WATCH = ["data/live/acct_notes.json", "data/live/acct_wrong.json", "data/live/push_subscriptions.json", "data/live/order_requests.jsonl"]


async def scenario(name, files, reqs, watch=()):
    with tempfile.TemporaryDirectory() as td:
        R = Path(td).resolve()
        (R / "dashboard" / "live").mkdir(parents=True)
        for rel, data in files.items():
            p = R / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            os.utime(p, ns=(MTIME_NS, MTIME_NS))
        log: list = []
        app = make_app(R, log)
        app.freeze()
        cases = []
        for req in reqs:
            method, path, headers = req[:3]
            body = req[3] if len(req) > 3 else None
            raw = None if body is None else body if isinstance(body, bytes) else body.encode() if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                out = await g1.run_one(app, method, path, headers, raw if raw is not None else (b"" if method == "POST" else None))
            case = {"method": method, "path": path, "req_headers": headers, "body_in": base64.b64encode(raw or b"").decode(), **out,
                    "log": [ln for ln in buf.getvalue().splitlines() if ln]}
            if watch:
                case["files_after"] = {w: (base64.b64encode((R / w).read_bytes()).decode() if (R / w).exists() else None) for w in watch}
                case["tmp_left"] = sorted(str(p.relative_to(R)) for p in R.rglob("*.tmp"))
            cases.append(case)
            etag = next((v for k, v in out["headers"] if k.lower() == "etag"), None)
            if etag and out["status"] == 200 and method == "GET":
                h2 = {**headers, "If-None-Match": etag}
                cases.append({"method": method, "path": path, "req_headers": h2, "body_in": "", **await g1.run_one(app, method, path, h2), "log": []})
    return {"name": name, "files": {k: base64.b64encode(v).decode() for k, v in files.items()}, "cases": cases}


async def main():
    out = {"now": NOW, "mtime_ns": MTIME_NS, "dashboard_assets": ",".join(DASHBOARD_ASSETS),
           "vapid_public": "BPUBLIC-test-key",
           "consts": {"flow_specs": FLOW_SPECS, "market_symbols": MARKET_SYMBOLS, "tape_buckets": TAPE_BUCKETS,
                      "whale_usd": {s: size_bands(s)[1] for s in TAPE_BUCKETS}},
           "scenarios": [await scenario("api2", tree_api2(), REQ_API2), await scenario("api2_variants", tree_api2_variants(), REQ_API2_VAR),
                         await scenario("writes", WRITES_TREE, REQ_WRITES, WATCH)]}
    # 정확한 stdev(statistics) 표본 -- 급변 칩의 기준
    rnd = random.Random(11)
    sets = [[rnd.gauss(0, 3) for _ in range(n)] for n in (2, 3, 288, 1000, 5400)] + [[1e-9 * k for k in range(300)], [5.0] * 300 + [5.000000001]]
    out["stdevs"] = [[s, statistics.stdev(s)] for s in sets]
    OUT.write_text(json.dumps(out, ensure_ascii=False) + "\n")
    n = sum(len(s["cases"]) for s in out["scenarios"])
    print(f"{OUT.relative_to(ROOT)}: 시나리오 {len(out['scenarios'])} · 요청 {n}")
    for s in out["scenarios"]:
        for c in s["cases"]:
            print(f"  {s['name']:14} {c['method']:4} {c['path'][:58]:58} {c['status']} {len(base64.b64decode(c['body_b64']))}B {c['log']}")


if __name__ == "__main__":
    asyncio.run(main())
