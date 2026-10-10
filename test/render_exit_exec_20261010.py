#!/usr/bin/env python3
"""10-10 청산 체결 칩(지정가·즉시) + 스위칭 즉시 체결을 실제 브라우저로 -- 서버·거래소 **없이**(render_fee_rate_20261010 패턴).

    python3 -u test/render_exit_exec_20261010.py [OUT_DIR]

🔴주문 0: 출처가 가짜 호스트(http://dash.local) · 모든 요청을 조건식으로 가로채 가짜 응답 · 바깥 호스트 abort · WS 가짜 ·
  누르기 전 가짜 제출로 가로채기 선검사(실패면 중단). 확인 버튼을 누르지만 제출은 가로챈 가짜 응답만 받는다.
검사(1920·390): 기본 = 지정가 칩 켜짐 · 미리보기·제출에 exec 없음 · «즉시» 칩 → localStorage exitExec=1 · 미리보기·제출 &exec=market ·
  미리보기 «테이커 수수료» · 확인 버튼 «즉시 체결(시장가, 테이커 4bp)» · 스위칭은 칩이 지정가여도 &exec=market + switch_to.
"""
import json, pathlib, sys
from urllib.parse import parse_qs, urlparse
from playwright.sync_api import sync_playwright

WT = pathlib.Path(__file__).resolve().parents[1]
DASH = WT / "dashboard/live"
ORIGIN = "http://dash.local"
URL = f"{ORIGIN}/dashboard/live/"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/render_exit_exec_20261010"); OUT.mkdir(parents=True, exist_ok=True)
PREVIEW = json.loads((WT / "test/fixtures/manual_entry_preview_20261005.json").read_text("utf-8"))
CT = {".js": "application/javascript", ".css": "text/css", ".html": "text/html", ".json": "application/json",
      ".webmanifest": "application/manifest+json", ".svg": "image/svg+xml", ".png": "image/png"}
FONT = pathlib.Path("/mnt/c/Windows/Fonts/malgun.ttf")
FEE = {"symbol": "ETHUSDC", "maker_bp": 0.0, "taker_bp": 4.0, "source": "live", "promo": True}
POS = [{"symbol": "ETHUSDC", "side": "SHORT", "qty": 2.563, "entry_price": 2498.0, "mark_price": 2500.0, "leverage": 20.0,
        "notional": -6407.5, "unrealized_pnl": -3.2, "liquidation_price": 2900.0, "entry_at": "2026-10-10T10:00:00+00:00"}]
log, fails = [], []


def account():
    return {"ok": True, "generated_at": "2026-10-10T12:00:00+00:00", "exec_symbol": "ETHUSDC", "exec_symbols": {"eth": "ETHUSDC"},
            "balance": {"margin": 1000.0, "wallet": 1000.0, "unrealized": 0.0, "available": 900.0, "initial_margin": 100.0},
            "positions": POS, "leverage_by_symbol": {"ETHUSDC": 20.0}, "trips": []}


def exit_preview(q):
    """서버 assemble_exit_plan 흉내 -- exec=market 이면 MARKET(가격·GTX 없음), 아니면 LIMIT peg."""
    body = json.loads(json.dumps(PREVIEW)); body["exec_enabled"] = True
    p = body["plan"]
    p.update(symbol="ETHUSDC", positionSide="SHORT", side="BUY", price=2499.90, blocked=None, position_side="SHORT", quantity=2.563,
             position_qty=2.563, remaining_qty=0, fraction=1.0, reference_price=2499.90, notional_usdt=round(2.563 * 2499.90, 2),
             unrealized_pnl=-3.2, exit_move_pct=-0.1, type="LIMIT", vol_bpm=10, market_reason=None, fallback_after_sec=60)
    p["trade_plan"]["execution"]["cost"].update(fee=FEE)
    if q.get("exec") == "market":
        p.pop("price", None)
        p.update(type="MARKET", market_reason="즉시 체결 — 지정가를 걸지 않고 바로 시장가로 닫습니다(테이커 수수료)")
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
        return js(exit_preview(q) if "/manual-exit/" in path else PREVIEW)
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


def since(n0, frag, post):
    return [u for m, u in log[n0:] if (m != "GET") == post and frag in u]


SETUP = lambda: f"() => {{ latestLivePriceByAsset.eth = 2500.40; renderBinanceAccount({json.dumps(account())}); manualExitSyncButtons(); ofabSync(); renderOfab(); }}"
TXT = "(id) => document.getElementById(id).textContent"

with sync_playwright() as p:
    b = p.chromium.launch()
    for w, h in ((1920, 1080), (390, 844)):
        ctx = b.new_context(viewport={"width": w, "height": h}, device_scale_factor=2, has_touch=w < 500)
        pg = ctx.new_page()
        pg.route(lambda u: True, handle)
        pg.route_web_socket(lambda u: True, lambda ws: None)
        pg.add_init_script("try { if (!sessionStorage.getItem('h')) { localStorage.clear(); sessionStorage.setItem('h', '1'); } } catch (e) {}"
                           "document.addEventListener('DOMContentLoaded', () => { const st = document.createElement('style');"
                           "st.textContent = ['Noto Sans KR', 'Pretendard Variable', 'JetBrains Mono', 'Space Grotesk'].map((f) => `@font-face { font-family: '${f}';"
                           " src: url(/__font/kr.ttf); unicode-range: U+1100-11FF, U+3130-318F, U+AC00-D7AF; }`).join(' '); document.head.appendChild(st); });")
        pg.goto(URL, wait_until="load", timeout=60000); pg.wait_for_timeout(2500)
        n0 = len(log)
        pg.evaluate("() => fetch('/api/manual-exit/submit?side=SHORT&confirm=1&pct=1&probe=1', {method:'POST'})"); pg.wait_for_timeout(300)
        if not since(n0, "probe=1", True):
            print("🔴가로채기 실패 -- 중단"); sys.exit(2)
        pg.evaluate(SETUP()); pg.wait_for_timeout(400)

        # ① 기본 = 지정가: 칩 상태 · 미리보기·제출에 exec 없음
        on = pg.evaluate("() => [...document.querySelectorAll('#snapExitExecBox .chip')].map((c) => c.textContent + ':' + c.getAttribute('aria-pressed'))")
        if on != ["지정가:true", "즉시:false"]: fails.append(f"[{w}] 기본 칩 {on}")
        n1 = len(log)
        pg.evaluate("() => manualExitPreview('SHORT')"); pg.wait_for_timeout(900)
        t = pg.evaluate(TXT, "snapEntryResult")
        if "peg 수수료" not in t: fails.append(f"[{w}] 지정가 미리보기 {t[:120]!r}")
        cb = pg.evaluate(TXT, "snapEntryConfirm")
        if "즉시" in cb: fails.append(f"[{w}] 지정가 확인 버튼 {cb!r}")
        pg.click("#snapEntryConfirm"); pg.wait_for_timeout(600)
        pv, sb = since(n1, "/manual-exit/preview", False), since(n1, "/manual-exit/submit", True)
        if not pv or not sb or any("exec=" in u for u in pv + sb): fails.append(f"[{w}] 지정가 요청 {pv} {sb}")
        pg.wait_for_timeout(3300)   # 상태 폴링이 idle 을 받아 버튼을 푼다
        pg.locator("#snapExitRow").scroll_into_view_if_needed()
        pg.locator("#snapExitRow").screenshot(path=str(OUT / f"card_exit_row_limit_{w}.png"))

        # ② «즉시» 칩 → 기억 · 미리보기·제출 &exec=market · 테이커 수수료 · 확인 버튼 문구
        pg.click("#snapExitExecBox .chip[data-v='1']"); pg.wait_for_timeout(200)
        stored = pg.evaluate("() => localStorage.getItem('exitExec')")
        if stored != "1": fails.append(f"[{w}] 저장 {stored!r}")
        n2 = len(log)
        pg.evaluate("() => manualExitPreview('SHORT')"); pg.wait_for_timeout(900)
        t = pg.evaluate(TXT, "snapEntryResult"); cb = pg.evaluate(TXT, "snapEntryConfirm")
        print(w, "즉시 확인 버튼", cb)
        if "테이커 수수료" not in t or "시장가 즉시 체결" not in t: fails.append(f"[{w}] 즉시 미리보기 {t[:160]!r}")
        if "즉시 체결(시장가, 테이커 4bp)" not in cb: fails.append(f"[{w}] 즉시 확인 버튼 {cb!r}")
        pg.locator("#snapExitRow").screenshot(path=str(OUT / f"card_exit_row_market_{w}.png"))
        pg.locator("#snapEntryResult").screenshot(path=str(OUT / f"exit_preview_market_{w}.png"))
        pg.click("#snapEntryConfirm"); pg.wait_for_timeout(600)
        pv, sb = since(n2, "/manual-exit/preview", False), since(n2, "/manual-exit/submit", True)
        if not pv or not sb or not all("exec=market" in u for u in pv + sb) or any("switch_to" in u for u in sb): fails.append(f"[{w}] 즉시 요청 {pv} {sb}")
        pg.wait_for_timeout(3300)

        # 새로 열어도 «즉시» 유지
        pg.reload(wait_until="load"); pg.wait_for_timeout(2500); pg.evaluate(SETUP()); pg.wait_for_timeout(400)
        v = pg.evaluate("() => document.getElementById('snapExitExec').value")
        if v != "1": fails.append(f"[{w}] 새로 연 뒤 칩 {v!r}")

        # ③ 스위칭: 칩을 지정가로 돌려도 &exec=market + switch_to · 확인 버튼 «스위칭 · 손절 걸림 · 즉시 체결»
        pg.click("#snapExitExecBox .chip[data-v='0']"); pg.wait_for_timeout(200)
        pg.evaluate("() => ofabSetOpen(true)"); pg.wait_for_timeout(500)
        pg.locator("#ofabPanel").screenshot(path=str(OUT / f"ofab_panel_exit_chips_{w}.png"))
        n3 = len(log)
        pg.focus("#snapSwitch"); pg.keyboard.press("Enter"); pg.wait_for_timeout(1200)
        cb = pg.evaluate(TXT, "snapEntryConfirm")
        print(w, "스위칭 확인 버튼", cb)
        if "스위칭 · 손절 걸림" not in cb or "즉시 체결(시장가, 테이커 4bp)" not in cb: fails.append(f"[{w}] 스위칭 확인 버튼 {cb!r}")
        pg.locator("#ofabPanel").screenshot(path=str(OUT / f"ofab_switch_confirm_{w}.png"))
        pg.click("#snapEntryConfirm"); pg.wait_for_timeout(600)
        pv, sb = since(n3, "/manual-exit/preview", False), since(n3, "/manual-exit/submit", True)
        print(w, "스위칭 제출", sb[:1])
        if not pv or not sb or not all("exec=market" in u for u in pv + sb) or not all("switch_to=LONG" in u for u in sb): fails.append(f"[{w}] 스위칭 요청 {pv} {sb}")
        if since(n3, "/manual-entry/submit", True): fails.append(f"[{w}] 브라우저가 반대 진입을 직접 냈다")
        ctx.close()
    b.close()

outside = [u for _, u in log if not u.startswith(ORIGIN)]
print("바깥 호스트 요청(전부 abort)", len(outside), "· 가로챈 요청", len(log))
print("FAIL" if fails else "PASS", *fails, sep="\n")
sys.exit(1 if fails else 0)
