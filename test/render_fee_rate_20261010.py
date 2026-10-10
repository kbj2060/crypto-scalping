#!/usr/bin/env python3
"""10-10 주문 심볼 실수수료율 표시를 실제 브라우저로 -- 서버·거래소 **없이**(완전 오프라인, render_order_panel_20261010 패턴).

    python3 -u test/render_fee_rate_20261010.py [OUT_DIR]

🔴주문 0: 출처가 가짜 호스트(http://dash.local) · 모든 요청을 조건식으로 가로채 가짜 응답 · 바깥 호스트 abort · WS 가짜 ·
  누르기 전 가짜 제출로 가로채기 선검사(실패면 중단). 이 하네스는 확인(제출) 버튼을 누르지 않는다.
검사(1920·390): 진입 미리보기 «수수료 ETHUSDC 메이커 0bp · 테이커 4bp · 정상 왕복 약 1.88bp · USDC 프로모션 요율(기간 한정)» ·
  청산 미리보기 «지금 닫으면» peg 수수료 = 3.0 − 2.0 + 0 = 1bp · 조회 실패(fallback)면 «추정»·프로모션 표시 없음.
"""
import json, pathlib, sys
from urllib.parse import parse_qs, urlparse
from playwright.sync_api import sync_playwright

WT = pathlib.Path(__file__).resolve().parents[1]
DASH = WT / "dashboard/live"
ORIGIN = "http://dash.local"
URL = f"{ORIGIN}/dashboard/live/"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/render_fee_rate_20261010"); OUT.mkdir(parents=True, exist_ok=True)
PREVIEW = json.loads((WT / "test/fixtures/manual_entry_preview_20261005.json").read_text("utf-8"))
CT = {".js": "application/javascript", ".css": "text/css", ".html": "text/html", ".json": "application/json",
      ".webmanifest": "application/manifest+json", ".svg": "image/svg+xml", ".png": "image/png"}
FONT = pathlib.Path("/mnt/c/Windows/Fonts/malgun.ttf")
FEES = {"live": {"symbol": "ETHUSDC", "maker_bp": 0.0, "taker_bp": 4.0, "source": "live", "promo": True},
        "fallback": {"symbol": "ETHUSDC", "maker_bp": 2.0, "taker_bp": 5.0, "source": "fallback", "promo": False}}
S = {"fee": "live", "positions": []}
log, fails = [], []


def account():
    return {"ok": True, "generated_at": "2026-10-10T12:00:00+00:00", "exec_symbol": "ETHUSDC", "exec_symbols": {"eth": "ETHUSDC"},
            "balance": {"margin": 1000.0, "wallet": 1000.0, "unrealized": 0.0, "available": 900.0, "initial_margin": 100.0},
            "positions": S["positions"], "leverage_by_symbol": {"ETHUSDC": 20.0}, "trips": []}


def preview(q, kind):
    body = json.loads(json.dumps(PREVIEW)); body["exec_enabled"] = True
    p = body["plan"]; side = q.get("side", "LONG")
    p.update(symbol="ETHUSDC", positionSide=side, side="BUY" if (side == "LONG") == (kind == "entry") else "SELL", price=2499.90, blocked=None)
    f = FEES[S["fee"]]
    p["trade_plan"]["execution"]["cost"].update(round_trip_bp=round(5.88 - 2 * (2.0 - f["maker_bp"]), 2), fee=f)
    p["trade_plan"]["execution"]["entry"].update(maker_bp=f["maker_bp"], taker_bp=f["taker_bp"])
    if kind == "exit":
        p.update(position_side=side, quantity=2.563, position_qty=2.563, remaining_qty=0, fraction=1.0, reference_price=2499.90,
                 notional_usdt=round(2.563 * 2499.90, 2), unrealized_pnl=-3.2, exit_move_pct=-0.1, type="LIMIT", vol_bpm=10)
    return body


def handle(route):
    req = route.request; u = req.url; pr = urlparse(u)
    log.append((req.method, u))
    if not u.startswith(ORIGIN):
        return route.abort()
    path, q = pr.path, {k: v[0] for k, v in parse_qs(pr.query).items()}
    js = lambda obj: route.fulfill(body=json.dumps(obj), content_type="application/json")
    if "/api/manual-" in path:
        if req.method != "GET":
            return js({"ok": True, "state": {"phase": "submitting"}, "plan": PREVIEW["plan"]})
        if "/status" in path:
            return js({"ok": True, "state": {"phase": "idle"}, "wait": {"phase": "idle"}, "exec_enabled": True, "bracket_armed": {}})
        return js(preview(q, "exit" if "/manual-exit/" in path else "entry"))
    if path == "/api/binance-account":
        return js(account())
    if path == "/__font/kr.ttf" and FONT.is_file():
        return route.fulfill(body=FONT.read_bytes(), content_type="font/ttf")
    if path.startswith("/api/"):
        return route.fulfill(status=503, body="{}", content_type="application/json")
    name = "index.html" if path in ("/dashboard/live/", "/dashboard/live") else path.split("/dashboard/live/", 1)[-1]
    f = DASH / name
    if f.is_file():
        return route.fulfill(body=f.read_bytes(), content_type=CT.get(f.suffix, "application/octet-stream"))
    return route.fulfill(status=404, body="")


SETUP = lambda: f"() => {{ latestLivePriceByAsset.eth = 2500.40; renderBinanceAccount({json.dumps(account())}); manualExitSyncButtons(); }}"
RESULT = "() => document.getElementById('snapEntryResult').textContent"

with sync_playwright() as p:
    b = p.chromium.launch()
    for w, h in ((1920, 1080), (390, 844)):
        S.update(fee="live", positions=[])
        ctx = b.new_context(viewport={"width": w, "height": h}, device_scale_factor=2, has_touch=w < 500)
        pg = ctx.new_page()
        pg.route(lambda u: True, handle)
        pg.route_web_socket(lambda u: True, lambda ws: None)
        pg.add_init_script("try { localStorage.clear(); } catch (e) {}"
                           "document.addEventListener('DOMContentLoaded', () => { const st = document.createElement('style');"
                           "st.textContent = ['Noto Sans KR', 'Pretendard Variable', 'JetBrains Mono', 'Space Grotesk'].map((f) => `@font-face { font-family: '${f}';"
                           " src: url(/__font/kr.ttf); unicode-range: U+1100-11FF, U+3130-318F, U+AC00-D7AF; }`).join(' '); document.head.appendChild(st); });")
        pg.goto(URL, wait_until="load", timeout=60000); pg.wait_for_timeout(2500)
        n0 = len(log)
        pg.evaluate("() => fetch('/api/manual-entry/submit?side=LONG&confirm=1&pct=1&probe=1', {method:'POST'})"); pg.wait_for_timeout(300)
        if not [u for m, u in log[n0:] if m != "GET" and "probe=1" in u]:
            print("🔴가로채기 실패 -- 중단"); sys.exit(2)
        pg.evaluate(SETUP()); pg.wait_for_timeout(300)

        # 진입 미리보기(프로모션 실요율)
        pg.focus("#snapEntryLong"); pg.keyboard.press("Enter"); pg.wait_for_timeout(900)
        t = pg.evaluate(RESULT); print(w, "진입", [s for s in t.split("수수료")[1:2]])
        want = "수수료 ETHUSDC 메이커 0bp · 테이커 4bp · 정상 왕복 약 1.88bp · USDC 프로모션 요율(기간 한정)"
        if want not in t: fails.append(f"[{w}] 진입 수수료 줄 {t!r}")
        pg.locator("#snapEntryResult").screenshot(path=str(OUT / f"entry_fee_promo_{w}.png"))

        # 조회 실패(표준 추정) -- 프로모션 표시 없음
        S["fee"] = "fallback"
        pg.evaluate("() => manualEntryRefreshSize && manualEntryRefreshSize()"); pg.focus("#snapEntryLong"); pg.keyboard.press("Enter"); pg.wait_for_timeout(900)
        t = pg.evaluate(RESULT)
        if "수수료 추정(요율 조회 실패 · 표준) 메이커 2bp · 테이커 5bp · 정상 왕복 약 5.88bp" not in t or "프로모션" in t: fails.append(f"[{w}] 추정 줄 {t!r}")
        pg.locator("#snapEntryResult").screenshot(path=str(OUT / f"entry_fee_fallback_{w}.png"))

        # 청산 미리보기 -- «지금 닫으면» peg 수수료 1bp(실측 3.0 − 표준 메이커 2.0 + 실요율 0) · 자세히 안에 수수료 줄
        S.update(fee="live", positions=[{"symbol": "ETHUSDC", "side": "SHORT", "qty": 2.563, "entry_price": 2498.0, "mark_price": 2500.0, "leverage": 20.0,
                                         "notional": -6407.5, "unrealized_pnl": -3.2, "liquidation_price": 2900.0, "entry_at": "2026-10-10T10:00:00+00:00"}])
        pg.evaluate(SETUP()); pg.wait_for_timeout(300)
        pg.evaluate("() => manualExitPreview('SHORT')"); pg.wait_for_timeout(900)
        t = pg.evaluate(RESULT); print(w, "청산", t[:160])
        if "peg 수수료 ≈$0.64 (1bp)" not in t or want not in t: fails.append(f"[{w}] 청산 미리보기 {t!r}")
        pg.evaluate("() => { const b = document.querySelector('#snapEntryResult .detail-toggle'); if (b && b.getAttribute('aria-expanded') !== 'true') b.click(); }")
        pg.wait_for_timeout(200)
        pg.locator("#snapEntryResult").screenshot(path=str(OUT / f"exit_fee_promo_{w}.png"))
        ctx.close()
    b.close()

outside = [u for _, u in log if not u.startswith(ORIGIN)]
real = [u for m, u in log if m != "GET" and "/api/manual-" in u and "probe=1" not in u]
if real: fails.append(f"선검사 밖 POST {real}")
print("바깥 호스트 요청(전부 abort)", len(outside), "· 가로챈 요청", len(log))
print("FAIL" if fails else "PASS", *fails, sep="\n")
sys.exit(1 if fails else 0)
