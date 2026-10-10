#!/usr/bin/env python3
"""«따라가기» 진입 대기 취소 버튼(2026-10-11)을 실제 브라우저로 -- 서버·거래소 **없이**(완전 오프라인).

    python3 -u test/render_entry_cancel_20261011.py [OUT_DIR]

🔴주문 0: 페이지 출처가 가짜 호스트(http://dash.local) · 모든 요청을 조건식으로 가로채 가짜 응답 · 바깥 호스트 abort · WebSocket 가짜 ·
  누르기 전 가짜 POST 로 가로채기 선검사(아니면 즉시 중단).
검사(1920·390): 따라가기 진입 working → 결과 칸에 «대기 취소» · 떠 있는 «대기» 표식 · 첫 클릭 = 무장(요청 0) · 둘째 클릭 = POST
  /api/manual-entry/cancel?confirm=1 한 번 · 서버가 cancelled 를 주면 버튼이 사라지고 «대기 취소 — 남은 수량은 사지 않았습니다» ·
  청산 working 엔 버튼 없음 · 가로 스크롤 0 · JS 오류 0.
"""
import json, pathlib, sys
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

WT = pathlib.Path(__file__).resolve().parents[1]
DASH = WT / "dashboard/live"
ORIGIN = "http://dash.local"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/render_entry_cancel_20261011"); OUT.mkdir(parents=True, exist_ok=True)
CT = {".js": "application/javascript", ".css": "text/css", ".html": "text/html", ".json": "application/json",
      ".webmanifest": "application/manifest+json", ".svg": "image/svg+xml", ".png": "image/png"}
FONT = pathlib.Path("/mnt/c/Windows/Fonts/malgun.ttf")
PLAN = {"symbol": "ETHUSDC", "side": "BUY", "positionSide": "LONG", "price": 2499.9, "quantity": 0.37}
S = {"state": {"phase": "idle"}}
log, fails, errs = [], [], []


def handle(route):
    req = route.request; u = req.url; pr = urlparse(u)
    log.append((req.method, u))
    if not u.startswith(ORIGIN):
        return route.abort()
    js = lambda obj, st=200: route.fulfill(status=st, body=json.dumps(obj), content_type="application/json")   # noqa: E731
    if pr.path.startswith("/api/manual-"):
        if req.method != "GET":
            if pr.path == "/api/manual-entry/cancel":
                S["state"].update(action="cancel")
                return js({"ok": True, "state": S["state"]})
            return js({"ok": False, "error": "harness"}, 400)
        if pr.path == "/api/manual-entry/status":
            return js({"ok": True, "state": S["state"], "wait": {"phase": "idle"}, "exec_enabled": True, "bracket_armed": {}})
        return route.fulfill(status=503, body="{}")
    if pr.path == "/__font/kr.ttf" and FONT.is_file():
        return route.fulfill(body=FONT.read_bytes(), content_type="font/ttf")
    if pr.path.startswith("/api/"):
        return route.fulfill(status=503, body="{}", content_type="application/json")
    name = "index.html" if pr.path in ("/dashboard/live/", "/dashboard/live") else pr.path.split("/dashboard/live/", 1)[-1]
    f = DASH / name
    return route.fulfill(body=f.read_bytes(), content_type=CT.get(f.suffix, "application/octet-stream")) if f.is_file() else route.fulfill(status=404, body="")


posts = lambda n0, frag: [u for m, u in log[n0:] if m != "GET" and frag in u]   # noqa: E731
BOX = "() => { const b = document.getElementById('snapEntryResult'), c = b.querySelector('[data-entry-cancel]'); return { text: b.textContent, btn: c && c.textContent, dis: c && c.disabled, pill: !document.getElementById('ofabWait').hidden, sw: document.documentElement.scrollWidth, vw: innerWidth }; }"

with sync_playwright() as p:
    b = p.chromium.launch()
    for w, h in ((1920, 1080), (390, 844)):
        S["state"] = {"phase": "idle"}
        ctx = b.new_context(viewport={"width": w, "height": h}, device_scale_factor=2, has_touch=w < 500)
        pg = ctx.new_page(); pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.route(lambda u: True, handle); pg.route_web_socket(lambda u: True, lambda ws: None)
        pg.add_init_script("try { localStorage.clear(); } catch (e) {}"
                           "document.addEventListener('DOMContentLoaded', () => { const st = document.createElement('style');"
                           "st.textContent = ['Noto Sans KR', 'Pretendard Variable', 'JetBrains Mono', 'Space Grotesk'].map((f) => `@font-face { font-family: '${f}';"
                           " src: url(/__font/kr.ttf); unicode-range: U+1100-11FF, U+3130-318F, U+AC00-D7AF; }`).join(' '); document.head.appendChild(st); });")
        pg.goto(f"{ORIGIN}/dashboard/live/", wait_until="load", timeout=60000); pg.wait_for_timeout(2000)
        n0 = len(log)
        pg.evaluate("() => fetch('/api/manual-entry/submit?side=LONG&confirm=1&probe=1', {method:'POST'})"); pg.wait_for_timeout(300)
        if not posts(n0, "probe=1"):
            print("🔴가로채기 실패 -- 중단"); sys.exit(2)
        # 따라가기 진입이 거래소에 걸려 있다(서버 manual_entry_state working)
        S["state"] = {"phase": "working", "side": "LONG", "asset": "eth", "plan": PLAN, "quantity": 0.37, "filled": 0.1, "limit_price": 2499.9, "repegs": 0}
        pg.evaluate("() => { const b = document.getElementById('snapEntryResult'); b.hidden = false; manualEntryPollStatus(); }"); pg.wait_for_timeout(700)
        s1 = pg.evaluate(BOX); print(w, "working", s1)
        if s1["btn"] != "대기 취소" or not s1["pill"]: fails.append(f"[{w}] working 인데 대기 취소 버튼·떠 있는 표식 {s1}")
        if s1["sw"] > s1["vw"]: fails.append(f"[{w}] 가로 스크롤 {s1}")
        pg.locator("#snapEntryResult").scroll_into_view_if_needed()
        pg.locator("#snapEntryResult").screenshot(path=str(OUT / f"entry_cancel_{w}.png"))
        pg.screenshot(path=str(OUT / f"ofab_pill_{w}.png"), clip=pg.evaluate(
            "() => { const r = document.getElementById('ofab').getBoundingClientRect(); return {x: Math.max(0, r.x - 20), y: Math.max(0, r.y - 20), width: Math.min(innerWidth - Math.max(0, r.x - 20), r.width + 40), height: r.height + 40}; }"))
        n1 = len(log)
        pg.click("#snapEntryResult [data-entry-cancel]"); pg.wait_for_timeout(200)
        s2 = pg.evaluate(BOX); print(w, "첫 클릭", s2["btn"])
        if posts(n1, "/manual-entry/cancel"): fails.append(f"[{w}] 첫 클릭에 취소가 나갔다")
        if "한 번 더" not in (s2["btn"] or ""): fails.append(f"[{w}] 무장 표시 없음 {s2}")
        pg.locator("#snapEntryResult").screenshot(path=str(OUT / f"entry_cancel_armed_{w}.png"))
        pg.click("#snapEntryResult [data-entry-cancel]"); pg.wait_for_timeout(400)
        c = posts(n1, "/manual-entry/cancel")
        if len(c) != 1 or "confirm=1" not in c[0]: fails.append(f"[{w}] 둘째 클릭 취소 요청 {c}")
        pg.wait_for_timeout(3300)                                                   # 다음 상태 조회(3초)가 «요청됨»을 그린다
        s3 = pg.evaluate(BOX); print(w, "요청 뒤", s3["btn"], s3["dis"])
        if not (s3["dis"] and "요청됨" in (s3["btn"] or "")): fails.append(f"[{w}] 요청 뒤 버튼 {s3}")
        S["state"] = {**S["state"], "phase": "cancelled", "done_at": "2099-01-01T00:00:00+00:00", "cancel_reason": "사용자 취소"}
        pg.wait_for_timeout(3300)
        s4 = pg.evaluate(BOX); print(w, "끝", s4["text"][:60], s4["btn"], s4["pill"])
        if s4["btn"] is not None or s4["pill"] or "대기 취소 — 남은 수량은 사지 않았습니다" not in s4["text"]: fails.append(f"[{w}] 끝난 뒤 {s4}")
        pg.locator("#snapEntryResult").screenshot(path=str(OUT / f"entry_cancelled_{w}.png"))
        # 청산 working 엔 버튼 없음(서버도 409)
        S["state"] = {"phase": "working", "kind": "exit", "side": "LONG", "asset": "eth", "plan": PLAN, "quantity": 0.37, "filled": 0.0}
        pg.evaluate("() => manualEntryPollStatus()"); pg.wait_for_timeout(600)
        s5 = pg.evaluate(BOX)
        if s5["btn"] is not None: fails.append(f"[{w}] 청산 working 에 대기 취소 버튼 {s5}")
        ctx.close()
    b.close()

outside = [u for _, u in log if not u.startswith(ORIGIN)]
print("바깥 호스트 요청(전부 abort)", len(outside), "· JS 오류", errs[:3])
if errs: fails.append(f"JS 오류 {errs[:3]}")
print("FAIL" if fails else "PASS", *fails, sep="\n")
sys.exit(1 if fails else 0)
