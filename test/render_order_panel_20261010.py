#!/usr/bin/env python3
"""주문 패널 10-10 변경 4건을 실제 브라우저로 -- 서버·거래소 **없이**(완전 오프라인).

    python3 -u test/render_order_panel_20261010.py [OUT_DIR]

🔴주문 0: 페이지 출처가 가짜 호스트(http://dash.local)라 실서버로 갈 길이 없다. 모든 요청을 조건식(predicate)으로 가로채
  로컬 파일·가짜 응답만 준다 · 바깥 호스트는 abort · WebSocket(바이낸스 직결)도 가짜로 막는다 · 누르기 전 가짜 제출로
  가로채기 선검사(아니면 즉시 중단). 터널(18787)이 없어 «실서버 마지막 주문 번호 전후 대조»는 할 수 없다 -- 대신 실서버로의
  경로 자체가 없다(요청 로그에 dash.local 밖 0건을 검사).
검사(1920·390): ② 자동 칩 옆·떠 있는 띠 «손절 시 순자산 −7.5%» ③ «60분 대기» 칩 → 제출에 mode=wait&wait_min=60 · 대기 상자·
  떠 있는 «대기» 표식 · 취소 = action=cancel 한 번 · 지금 시장가 = 두 번 눌러야 action=market ④ 진입·청산 미리보기와 떠 있는 띠에
  USDC 호가·USDT 대비 bp ① 스위칭 확인 → 청산 제출에 switch_to·sw_* · 서버 스위칭 상태(청산 중 → 완료)를 결과 칸이 말한다.
"""
import json, pathlib, sys
from urllib.parse import parse_qs, urlparse
from playwright.sync_api import sync_playwright

WT = pathlib.Path(__file__).resolve().parents[1]
DASH = WT / "dashboard/live"
ORIGIN = "http://dash.local"
URL = f"{ORIGIN}/dashboard/live/"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/render_order_panel_20261010"); OUT.mkdir(parents=True, exist_ok=True)
PREVIEW = json.loads((WT / "test/fixtures/manual_entry_preview_20261005.json").read_text("utf-8"))
USDT = 2500.40                      # 화면 시세(ETHUSDT)
BID, ASK = 2499.90, 2499.91         # 주문 심볼(ETHUSDC) -- USDT 보다 ≈2bp 싸다
CT = {".js": "application/javascript", ".css": "text/css", ".html": "text/html", ".json": "application/json",
      ".webmanifest": "application/manifest+json", ".svg": "image/svg+xml", ".png": "image/png"}

FONT = pathlib.Path("/mnt/c/Windows/Fonts/malgun.ttf")   # 캡처용 한글 글꼴(시험 환경에 CJK 글꼴이 없다 -- 바깥 글꼴 요청은 abort) -- 화면 글꼴 사슬의 Noto Sans KR 자리
S = {"positions": [], "status": {"phase": "idle"}, "wait": {"phase": "idle"}}
log, fails = [], []


def account():
    return {"ok": True, "generated_at": "2026-10-10T12:00:00+00:00", "exec_symbol": "ETHUSDC", "exec_symbols": {"eth": "ETHUSDC"},
            "balance": {"margin": 1000.0, "wallet": 1000.0, "unrealized": 0.0, "available": 900.0, "initial_margin": 100.0},
            "positions": S["positions"], "leverage_by_symbol": {"ETHUSDC": 20.0}, "trips": []}


def preview(q, kind):
    body = json.loads(json.dumps(PREVIEW)); body["exec_enabled"] = True
    p = body["plan"]; side = q.get("side", "LONG")
    buy = (side == "LONG") == (kind == "entry")
    p.update(symbol="ETHUSDC", positionSide=side, side="BUY" if buy else "SELL", price=BID if buy else ASK, blocked=None)
    if kind == "exit":
        p.update(position_side=side, quantity=2.563, position_qty=2.563, remaining_qty=0, fraction=1.0, reference_price=p["price"],
                 notional_usdt=round(2.563 * p["price"], 2), unrealized_pnl=-3.2, exit_move_pct=-0.1, fallback_after_sec=60, vol_bpm=10)
    if q.get("rule"):
        p["rule"] = {"l": 1.0, "binding": "동일위험", "cap_notional": 1000.0, "first": True, "room": 1000.0, "target_notional": 750.0,
                     "loss_at_stop_pct": 7.5, "sl_price": 2312.4, "sl_name": "첫 진입가 −7.5%(3σ)"}
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
            if "/manual-entry/wait-action" in path:
                return js({"ok": True, "wait": S["wait"]})
            if "/manual-entry/submit" in path and q.get("mode") == "wait":
                S["wait"] = {"phase": "working", "mode": "wait", "side": q.get("side"), "asset": "eth", "quantity": 0.37, "filled": 0.0,
                             "limit_price": BID, "wait_until": (__import__("datetime").datetime.now(__import__("datetime").timezone.utc) + __import__("datetime").timedelta(minutes=58)).isoformat()}
                return js({"ok": True, "wait": True, "state": S["wait"], "plan": PREVIEW["plan"]})
            if "/manual-exit/submit" in path and q.get("switch_to"):
                S["status"] = {"phase": "working", "kind": "exit", "side": q.get("side"), "quantity": 2.563, "filled": 0.8,
                               "switch": {"from": q.get("side"), "to": q.get("switch_to"), "phase": "exit"}}
                return js({"ok": True, "state": S["status"], "plan": PREVIEW["plan"]})
            return js({"ok": True, "state": {"phase": "submitting"}, "plan": PREVIEW["plan"]})
        if "/status" in path:
            return js({"ok": True, "state": S["status"], "wait": S["wait"], "exec_enabled": True, "bracket_armed": {}})
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


def submits(n0, frag):
    return [u for m, u in log[n0:] if m != "GET" and frag in u]


SETUP = lambda: f"""() => {{
  latestLivePriceByAsset.eth = {USDT};
  renderBinanceAccount({json.dumps(account())});
  manualExitSyncButtons(); ofabSync(); renderOfab();
}}"""

with sync_playwright() as p:
    b = p.chromium.launch()
    for w, h in ((1920, 1080), (390, 844)):
        S.update(positions=[], status={"phase": "idle"}, wait={"phase": "idle"})
        ctx = b.new_context(viewport={"width": w, "height": h}, device_scale_factor=2, has_touch=w < 500)
        pg = ctx.new_page()
        pg.route(lambda u: True, handle)
        pg.route_web_socket(lambda u: True, lambda ws: None)          # 바이낸스 직결 WS 를 열지 않는다(가짜)
        pg.add_init_script("try { localStorage.clear(); } catch (e) {}"
                           "document.addEventListener('DOMContentLoaded', () => { const st = document.createElement('style');"
                           "st.textContent = ['Noto Sans KR', 'Pretendard Variable', 'JetBrains Mono', 'Space Grotesk'].map((f) => `@font-face { font-family: '${f}';"
                           " src: url(/__font/kr.ttf); unicode-range: U+1100-11FF, U+3130-318F, U+AC00-D7AF; }`).join(' '); document.head.appendChild(st); });")
        pg.goto(URL, wait_until="load", timeout=60000); pg.wait_for_timeout(2500)
        n0 = len(log)
        pg.evaluate("() => fetch('/api/manual-entry/submit?side=LONG&confirm=1&pct=1&probe=1', {method:'POST'})"); pg.wait_for_timeout(300)
        if not submits(n0, "probe=1"):
            print("🔴가로채기 실패 -- 중단"); sys.exit(2)
        pg.evaluate(SETUP()); pg.evaluate("() => manualEntryRefreshSize()"); pg.wait_for_timeout(800)

        # ② 자동 규칙 «손절 시 순자산 −7.5%» + ④ 떠 있는 띠 USDC 호가
        pg.evaluate("() => { const i = document.getElementById('snapRule'); i.value = '1'; i.dispatchEvent(new Event('input', {bubbles: true})); renderOfab(); }")
        pg.wait_for_timeout(600)
        st = pg.evaluate("""() => { const v = (e) => !!e && getComputedStyle(e).display !== 'none' && e.getBoundingClientRect().height > 0;
            const card = document.querySelector('#snapRuleBox .rule-risk'), strip = document.querySelector('#ofabRule .rule-risk'), q = document.getElementById('ofabQuote');
            const sr = document.getElementById('ofabRule').getBoundingClientRect();
            return { card: v(card) && card.textContent, strip: v(strip) && strip.textContent, quote: v(q) && q.textContent, stripRight: sr.right, vw: innerWidth }; }""")
        print(w, "자동 규칙·띠", st)
        if st["card"] != "손절 시 순자산 −7.5%" or st["strip"] != "손절 시 순자산 −7.5%": fails.append(f"[{w}] 7.5% 문구 {st}")
        if not (st["quote"] or "").startswith("USDC −"): fails.append(f"[{w}] 띠 USDC 호가 {st}")
        if st["stripRight"] > st["vw"] + 0.5: fails.append(f"[{w}] 띠가 화면 밖 {st}")
        pg.screenshot(path=str(OUT / f"ofab_flat_auto_{w}.png"), clip=pg.evaluate(
            "() => { const r = document.getElementById('ofab').getBoundingClientRect(); return {x: Math.max(0, r.x - 20), y: Math.max(0, r.y - 60), width: Math.min(innerWidth - Math.max(0, r.x - 20), r.width + 40), height: r.height + 80}; }"))
        pg.locator("#snapRuleBox").scroll_into_view_if_needed()
        pg.locator(".manual-entry-row.entry-line").screenshot(path=str(OUT / f"card_entry_line_auto_{w}.png"))
        pg.evaluate("() => { const i = document.getElementById('snapRule'); i.value = '0'; i.dispatchEvent(new Event('input', {bubbles: true})); }")

        # ③ 60분 대기 → 진입 미리보기(④ USDC 줄) → 확인 → 제출에 mode=wait
        pg.click("#snapExecBox .chip[data-v='60']"); pg.wait_for_timeout(200)
        if pg.evaluate("() => document.querySelector(\"#snapExecBox .chip[data-v='60']\").classList.contains('on')") is not True: fails.append(f"[{w}] 60분 칩 켜짐 표시 없음")
        if pg.evaluate("() => localStorage.getItem('entryExec')") != "60": fails.append(f"[{w}] 대기 칩 저장 안 됨")
        pg.focus("#snapEntryLong"); pg.keyboard.press("Enter"); pg.wait_for_timeout(900)
        eq = pg.evaluate("() => (document.querySelector('#snapEntryResult .entry-quote') || {}).textContent || ''")
        note = pg.evaluate("() => document.getElementById('snapEntryResult').textContent")
        print(w, "진입 미리보기 호가 줄", eq)
        if "ETHUSDC 최우선 매수호가 2,499.90" not in eq or "−2.0bp" not in eq: fails.append(f"[{w}] 진입 미리보기 호가 {eq!r}")
        if "60분 뒤 남은 수량 시장가" not in note: fails.append(f"[{w}] 미리보기에 대기 약속 없음")
        pg.locator("#snapEntryResult").screenshot(path=str(OUT / f"entry_preview_quote_wait_{w}.png"))
        n1 = len(log)
        pg.click("#snapEntryConfirm"); pg.wait_for_timeout(1200)
        sb = submits(n1, "/manual-entry/submit")
        if not sb or "mode=wait" not in sb[0] or "wait_min=60" not in sb[0]: fails.append(f"[{w}] 대기 제출 {sb}")
        ws = pg.evaluate("""() => ({ box: !document.getElementById('snapWait').hidden, text: document.getElementById('snapWaitText').textContent,
            ofab: !document.getElementById('ofabWait').hidden, entryEnabled: !document.getElementById('snapEntryLong').disabled })""")
        print(w, "대기 상자", ws)
        if not (ws["box"] and ws["ofab"] and "기다리기 롱" in ws["text"] and ws["entryEnabled"]): fails.append(f"[{w}] 대기 상자 {ws}")
        pg.locator("#snapWait").scroll_into_view_if_needed()
        pg.locator("#snapWait").screenshot(path=str(OUT / f"wait_box_{w}.png"))
        pg.screenshot(path=str(OUT / f"ofab_wait_badge_{w}.png"), clip=pg.evaluate(
            "() => { const r = document.getElementById('ofab').getBoundingClientRect(); return {x: Math.max(0, r.x - 20), y: Math.max(0, r.y - 20), width: Math.min(innerWidth - Math.max(0, r.x - 20), r.width + 40), height: r.height + 40}; }"))
        n2 = len(log)
        pg.click("#snapWaitMarket"); pg.wait_for_timeout(300)
        if submits(n2, "wait-action"): fails.append(f"[{w}] 지금 시장가가 한 번에 나갔다")
        armed = pg.evaluate("() => document.getElementById('snapWaitMarket').textContent")
        pg.click("#snapWaitMarket"); pg.wait_for_timeout(400)
        mk = submits(n2, "wait-action")
        if armed != "한 번 더 = 남은 수량 시장가" or len(mk) != 1 or "action=market" not in mk[0] or "confirm=1" not in mk[0]: fails.append(f"[{w}] 시장가 두 번 {armed} {mk}")
        pg.wait_for_timeout(3300)                                               # 대기 조회가 버튼을 다시 연다
        n3 = len(log)
        pg.click("#snapWaitCancel"); pg.wait_for_timeout(400)
        cc = submits(n3, "wait-action")
        if len(cc) != 1 or "action=cancel" not in cc[0]: fails.append(f"[{w}] 취소 {cc}")
        S["wait"] = {"phase": "cancelled", "mode": "wait", "side": "LONG", "asset": "eth", "quantity": 0.37, "filled": 0.1,
                     "cancel_reason": "사용자 취소", "done_at": "2099-01-01T00:00:00+00:00"}
        pg.wait_for_timeout(3300)
        done = pg.evaluate("() => ({ text: document.getElementById('snapWaitText').textContent, acts: document.getElementById('snapWaitActs').hidden, ofab: document.getElementById('ofabWait').hidden })")
        print(w, "대기 끝", done)
        if "대기 취소" not in done["text"] or not done["acts"] or not done["ofab"]: fails.append(f"[{w}] 대기 끝 표시 {done}")
        pg.evaluate("() => { const i = document.getElementById('snapExec'); i.value = '0'; i.dispatchEvent(new Event('input', {bubbles: true})); }")

        # ① 스위칭: 숏 2.563 보유 → 패널 «스위칭 → 롱» → 확인 → 청산 제출에 switch_to·sw_*
        S["positions"] = [{"symbol": "ETHUSDC", "side": "SHORT", "qty": 2.563, "entry_price": 2498.0, "mark_price": 2500.0, "leverage": 20.0,
                           "notional": -6407.5, "unrealized_pnl": -5.1, "liquidation_price": 2900.0, "entry_at": "2026-10-10T10:00:00+00:00"}]
        pg.evaluate(SETUP()); pg.wait_for_timeout(300)
        pg.evaluate("() => ofabSetOpen(true)"); pg.wait_for_timeout(500)
        pg.focus("#snapSwitch"); pg.keyboard.press("Enter"); pg.wait_for_timeout(1200)
        xq = pg.evaluate("() => (document.querySelector('#snapEntryResult .entry-quote') || {}).textContent || ''")
        if "ETHUSDC 최우선 매수호가 2,499.90" not in xq: fails.append(f"[{w}] 청산 미리보기 호가 {xq!r}")
        n4 = len(log)
        pg.click("#snapEntryConfirm"); pg.wait_for_timeout(800)
        xs = submits(n4, "/manual-exit/submit")
        print(w, "스위칭 제출", xs[:1])
        if not xs or not all(k in xs[0] for k in ("switch_to=LONG", "sw_pct=", "sw_lev=20", "pct=100")): fails.append(f"[{w}] 스위칭 제출 {xs}")
        if submits(n4, "/manual-entry/submit"): fails.append(f"[{w}] 브라우저가 반대 진입을 직접 냈다")
        pg.wait_for_timeout(3300)
        t1 = pg.evaluate("() => document.getElementById('snapEntryResult').textContent")
        S["status"] = {"phase": "filled_maker", "side": "LONG", "quantity": 2.56, "filled": 2.56, "bracket": {"placed": False, "disabled": True},
                       "switch": {"from": "SHORT", "to": "LONG", "phase": "done", "exit": {"phase": "filled_maker"}}}
        pg.wait_for_timeout(3300)
        t2 = pg.evaluate("() => document.getElementById('snapEntryResult').textContent")
        print(w, "스위칭 상태", t1[:60], "→", t2[:60])
        if "스위칭 숏 → 롱 · 전량 청산 중" not in t1 or "스위칭 숏 → 롱 · 완료" not in t2: fails.append(f"[{w}] 스위칭 상태 {t1!r} {t2!r}")
        if submits(n4, "/manual-entry/submit"): fails.append(f"[{w}] 브라우저가 반대 진입을 직접 냈다(완료 뒤)")
        pg.locator("#ofabPanel").screenshot(path=str(OUT / f"switch_panel_{w}.png"))
        ctx.close()
    b.close()

outside = [u for _, u in log if not u.startswith(ORIGIN)]
real_orders = [u for m, u in log if m != "GET" and "/api/manual-" in u and not u.startswith(ORIGIN)]
print("바깥 호스트 요청(전부 abort)", len(outside), "· 실서버로 간 주문 0 =", not real_orders, "· 가로챈 요청", len(log))
print("FAIL" if fails else "PASS", *fails, sep="\n")
sys.exit(1 if fails else 0)
