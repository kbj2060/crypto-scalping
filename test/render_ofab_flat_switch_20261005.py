#!/usr/bin/env python3
"""떠 있는 주문 버튼 2026-10-05 변경 -- 포지션 없음 LONG·SHORT · 스위칭 · 말풍선 아래 -- 를 **실제 브라우저로** 누른다.

    python3 -u test/render_ofab_flat_switch_20261005.py      # 터널 127.0.0.1:18787 필요(ssh -L 18787:127.0.0.1:8787)

서버 페이지 위에 로컬 app.js·styles.css·index.html 을 덮는다. 🔴주문 API(`/api/manual-` 로 시작하는 **모든** 경로)와 계좌는
**브라우저 안에서 가짜로 답한다** -- 서버에 주문 요청이 하나도 안 간다(진입 미리보기도 고정 응답 fixtures/manual_entry_preview_20261005.json).
🔴2026-10-05 실사고: 첫 판이 glob `**/api/manual-*` 를 썼는데 `*` 는 `/` 를 못 넘어 submit 이 실서버로 가서 **실주문이 두 번 체결**됐다.
  그래서 ①조건식(predicate)으로 가로채고 ②누르기 전에 가짜 제출 요청으로 가로채기를 먼저 검사하고(아니면 즉시 중단)
  ③실행 전후 실서버의 마지막 주문 번호(읽기 전용 status)가 같은지 대조한다. 진입 비율은 게이지 최저(5%).
검사:
  ① 포지션 없음: «주문» 대신 LONG·SHORT · 마우스 1.5초 꾹 = 그 방향 진입(pct 5 · sltp 0) · 짧게 = 펼치기만(요청 0)
     · 터치 꾹 = 발주 없음 · 결과 말풍선이 버튼 **아래** · 글자 세로 가운데(±1.5px)
  ② 스위칭(롱 3.423 ×20 픽스처): 청산 게이지 25% 여도 **100%** 청산 → 전량 체결되면 숏 진입 pct=23.21(증거금 %)·lev=20·fresh=1
     · 청산이 거부되면 반대 진입 0 · 0.2초에 떼면 아무것도 안 나감 · 보통 «롱 청산»은 그대로 게이지 25%
  ③ 390px 에서 펼친 패널이 화면 안(옆 넘침 0) · JS 오류 0
"""
import json
import pathlib
import sys
import urllib.request
from urllib.parse import parse_qs, urlparse

HERE = pathlib.Path(__file__).resolve().parent
DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"
POS = (HERE / "fixtures" / "acct_position.js").read_text("utf-8")
PREVIEW = json.loads((HERE / "fixtures" / "manual_entry_preview_20261005.json").read_text("utf-8"))


def real_last_order():
    """실서버의 마지막 수동 주문 번호(읽기 전용 GET). 실행 전후가 같아야 한다."""
    with urllib.request.urlopen("http://127.0.0.1:18787/api/manual-entry/status", timeout=20) as r:
        s = json.load(r).get("state") or {}
    return s.get("order_id"), s.get("started_at")
EMPTY = """() => {
  for (let i = 1; i < 99999; i++) window.clearInterval(i);
  latestBinanceAccount = Object.assign({}, latestBinanceAccount || {}, { positions: [] });
  manualExitSyncButtons(); renderSnapshotAccount(); renderOfab();
}"""


def q(url):
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


def page(b, w, h, touch, exit_phase="filled_maker"):
    js, css, html = ((DASH / f).read_text("utf-8") for f in ("app.js", "styles.css", "index.html"))
    pg = b.new_page(viewport={"width": w, "height": h}, has_touch=touch, is_mobile=touch)
    log = {"errs": [], "calls": [], "acct": None, "exit_phase": exit_phase}
    pg.on("pageerror", lambda e: log["errs"].append(str(e)))
    # 🔴안전망: GET 이 아닌 요청은 서버에 닿기 전에 전부 끊는다(아래 주문 API 가짜 응답이 먼저 받는다 -- 뒤에 등록한 route 가 이긴다).
    pg.route("**/*", lambda r: r.continue_() if r.request.method == "GET" else (log["errs"].append(f"POST 차단 {r.request.url}"), r.abort()))
    serve = lambda body, ct: (lambda r: r.fulfill(body=body, content_type=ct))   # noqa: E731
    pg.route("**/app.js*", serve(js, "application/javascript"))
    pg.route("**/styles.css*", serve(css, "text/css"))
    pg.route(URL, serve(html, "text/html"))
    last = {}

    def api(route):
        u = route.request.url
        path = urlparse(u).path
        if path.startswith("/api/binance-account"):
            return route.fulfill(json={**log["acct"], "ok": True}) if log["acct"] else route.fulfill(json={"ok": False, "error": "test"})
        log["calls"].append(u)
        p = q(u)
        if p.get("sentinel") == "1":                        # 가로채기 선검사
            return route.fulfill(json={"sentinel": True})
        if path == "/api/manual-entry/preview":            # 고정 응답 -- 서버에 안 묻는다
            d = json.loads(json.dumps(PREVIEW))
            d["plan"].update(side="BUY" if p["side"] == "LONG" else "SELL", fraction=float(p["pct"]) / 100)
            return route.fulfill(json=d)
        if path == "/api/manual-exit/preview":
            qty = 3.423
            return route.fulfill(json={"ok": True, "exec_enabled": True, "plan": {
                "type": "LIMIT", "position_side": p["side"], "position_qty": qty, "quantity": qty, "fraction": float(p["pct"]) / 100,
                "price": 2739.4, "reference_price": 2739.4, "notional_usdt": qty * 2739.4, "remaining_qty": 0,
                "unrealized_pnl": 6.53, "fallback_after_sec": 120, "blocked": None, "risk": {}}})
        if path.endswith("/submit"):
            kind = "exit" if "manual-exit" in path else "entry"
            last.update(kind=kind, qty=3.423 if kind == "exit" else 0.85)
            return route.fulfill(json={"ok": True, "state": {"phase": "working", "kind": kind, "quantity": last["qty"], "filled": 0}})
        if path == "/api/manual-entry/status":
            if last.get("kind") == "exit":
                ph = log["exit_phase"]
                full = ph.startswith("filled_")
                return route.fulfill(json={"state": {"phase": ph, "kind": "exit", "quantity": last["qty"], "filled": last["qty"] if full else 0}})
            return route.fulfill(json={"state": {"phase": "filled_maker", "kind": "entry", "quantity": last.get("qty", 0),
                                                 "filled": last.get("qty", 0), "bracket": {"skipped": True}}})
        return route.fulfill(status=599, json={"ok": False, "error": "test_blocked"})   # 모르는 주문 경로도 서버로 안 보낸다
    pg.route(lambda url: "/api/manual-" in url or "/api/binance-account" in url, api)
    pg.goto(URL, wait_until="load", timeout=60000)
    pg.wait_for_timeout(5000)
    got = pg.evaluate("() => fetch('/api/manual-entry/submit?sentinel=1&side=LONG&confirm=1', {method: 'POST'}).then((r) => r.json())")
    if got != {"sentinel": True}:
        raise SystemExit(f"🔴가로채기 실패 -- 주문 요청이 서버로 갈 수 있다. 중단 ({got})")
    log["calls"].clear()
    pg.evaluate("() => localStorage.removeItem('ofabPos')")
    return pg, log


def calls(log, frag):
    return [q(u) | {"_path": urlparse(u).path} for u in log["calls"] if frag in u]


def run():
    from playwright.sync_api import sync_playwright
    fails = []
    ok = lambda c, m: None if c else fails.append(m)   # noqa: E731
    before = real_last_order()
    with sync_playwright() as p:
        b = p.chromium.launch()
        # ① 포지션 없음 -- 마우스
        pg, log = page(b, 1500, 900, False)
        pg.evaluate(EMPTY)
        log["acct"] = pg.evaluate("() => latestBinanceAccount")
        pg.evaluate("() => { renderOfab(); ofabSync(); ofabPlace(600, 200, false); }")
        pg.wait_for_timeout(300)
        ok(pg.is_hidden("#ofabToggle") and pg.is_visible("#ofabLong") and pg.is_visible("#ofabShort"), "① 포지션 없음인데 LONG·SHORT 가 안 보임")
        off = pg.evaluate("""() => [...document.querySelectorAll('.ofab-dir')].map((b) => { const r = b.getBoundingClientRect(),
            t = b.querySelector('span').getBoundingClientRect(); return (t.top + t.bottom) / 2 - (r.top + r.bottom) / 2; })""")
        ok(all(abs(o) <= 1.5 for o in off), f"① 글자 세로 어긋남 {off}")
        pg.click("#ofabLong")                                    # 짧게 = 펼치기만
        pg.wait_for_timeout(500)
        # (크기 표시용 미리보기 조회는 화면이 평소에도 보낸다 -- 판정은 «제출 0»)
        ok(pg.is_visible("#ofabPanel") and not calls(log, "/submit"), f"① 짧게 눌렀는데 제출/안 펼침 {calls(log, '/submit')}")
        pg.click("#ofabLong")                                    # 접기
        pg.wait_for_timeout(300)
        bx = pg.evaluate("() => { const r = document.getElementById('ofabShort').getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }")
        pg.mouse.move(*bx); pg.mouse.down(); pg.wait_for_timeout(1700); pg.mouse.up()
        pg.wait_for_timeout(2500)
        sub = calls(log, "manual-entry/submit")
        ok(len(sub) == 1 and sub[0]["side"] == "SHORT" and sub[0]["pct"] == "5" and sub[0].get("sltp") == "0" and "fresh" not in sub[0],
           f"① SHORT 1.5초 꾹 → 진입 5%·sltp 0 이어야 {sub}")
        say = pg.evaluate("() => { const s = document.getElementById('ofabSay'), b = document.querySelector('#ofab .ofab-bar');"
                          " return s.hidden ? null : [s.getBoundingClientRect().top, b.getBoundingClientRect().bottom]; }")
        ok(say and say[0] >= say[1], f"① 말풍선이 버튼 아래가 아님 {say}")
        ok(not log["errs"], f"① JS 오류 {log['errs'][:2]}")
        pg.close()
        # ① 포지션 없음 -- 터치 꾹은 발주 없음
        pg, log = page(b, 390, 844, True)
        pg.evaluate(EMPTY)
        pg.evaluate("() => { renderOfab(); ofabSync(); }")
        pg.wait_for_timeout(300)
        r = pg.evaluate("() => { const r = document.getElementById('ofabLong').getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }")
        cdp = pg.context.new_cdp_session(pg)
        cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": r[0], "y": r[1]}]})
        pg.wait_for_timeout(2000)
        cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        pg.wait_for_timeout(1500)
        ok(not calls(log, "/submit"), f"① 터치 꾹이 발주했다 {calls(log, '/submit')}")
        pg.close()
        # ② 스위칭 -- 전량 체결 / 청산 거부
        for phase, want_entry in (("filled_maker", True), ("rejected", False)):
            pg, log = page(b, 1500, 900, False, exit_phase=phase)
            pg.evaluate(POS)
            log["acct"] = pg.evaluate("() => latestBinanceAccount")
            pg.evaluate("() => { manualExitSyncButtons(); renderSnapshotAccount(); renderOfab(); ofabSync(); ofabPlace(600, 60, false); ofabSetOpen(true); manualExitSyncButtons(); }")
            pg.evaluate("() => { const g = document.getElementById('snapExitFrac'); g.value = '25'; g.dispatchEvent(new Event('input', {bubbles: true})); }")
            pg.wait_for_timeout(400)
            ok(pg.is_visible("#snapSwitch") and "숏" in pg.inner_text("#snapSwitchTo"), "② 스위칭 버튼이 안 보이거나 «→ 숏» 아님")
            if phase == "filled_maker":                           # 0.2초에 떼면 아무것도 안 나감
                pg.hover("#snapSwitch"); pg.mouse.down(); pg.wait_for_timeout(200); pg.mouse.up()
                pg.wait_for_timeout(1500)
                ok(not calls(log, "/submit"), f"② 0.2초에 뗐는데 제출 {calls(log, '/submit')}")
            pg.hover("#snapSwitch"); pg.mouse.down(); pg.wait_for_timeout(700); pg.mouse.up()
            pg.wait_for_timeout(9000)                             # 상태 폴링 3초 × 두 번 + 반대 진입 미리보기
            ex = calls(log, "manual-exit/submit")
            ok(len(ex) == 1 and ex[0]["side"] == "LONG" and ex[0]["pct"] == "100", f"② [{phase}] 청산이 100% 아님 {ex}")
            en = calls(log, "manual-entry/submit")
            if want_entry:
                ok(len(en) == 1 and en[0]["side"] == "SHORT" and en[0]["pct"] == "23.21" and en[0].get("lev") == "20" and en[0].get("fresh") == "1",
                   f"② 반대 진입이 숏 23.21%·20배·fresh 여야 {en}")
                pv = [c for c in calls(log, "manual-entry/preview") if c.get("fresh") == "1"]
                ok(len(pv) == 1 and pv[0]["side"] == "SHORT" and pv[0]["pct"] == "23.21", f"② 반대 진입 미리보기(fresh=1)가 하나가 아님 {pv}")
            else:
                ok(not en, f"② 청산 거부인데 반대 진입이 나갔다 {en}")
                ok("중단" in pg.inner_text("#ofabSay"), "② 청산 거부인데 «스위칭 중단» 말풍선이 없음")
            ok(not log["errs"], f"② [{phase}] JS 오류 {log['errs'][:2]}")
            pg.close()
        # ② 보통 «롱 청산»은 게이지 그대로(25%) · 뒤에 진입이 새지 않는다
        pg, log = page(b, 1500, 900, False)
        pg.evaluate(POS)
        log["acct"] = pg.evaluate("() => latestBinanceAccount")
        pg.evaluate("() => { manualExitSyncButtons(); renderSnapshotAccount(); renderOfab(); ofabSync(); ofabPlace(600, 60, false); ofabSetOpen(true); manualExitSyncButtons(); }")
        pg.evaluate("() => { const g = document.getElementById('snapExitFrac'); g.value = '25'; g.dispatchEvent(new Event('input', {bubbles: true})); }")
        pg.hover("#snapExitLong"); pg.mouse.down(); pg.wait_for_timeout(700); pg.mouse.up()
        pg.wait_for_timeout(2500)
        ex = calls(log, "manual-exit/submit")
        ok(len(ex) == 1 and ex[0]["pct"] == "25", f"② 보통 청산이 게이지 25% 가 아님 {ex}")
        pg.wait_for_timeout(4000)
        ok(not calls(log, "manual-entry/submit"), "② 보통 청산 뒤에 진입이 나갔다(스위칭이 샜다)")
        pg.close()
        # ③ 390px 패널 화면 안
        pg, log = page(b, 390, 844, True)
        pg.evaluate(POS)
        pg.evaluate("() => { manualExitSyncButtons(); renderOfab(); ofabSync(); ofabPlace(80, 60, false); ofabSetOpen(true); }")
        pg.wait_for_timeout(400)
        r = pg.evaluate("() => { const r = document.getElementById('ofabPanel').getBoundingClientRect(); return [r.left, r.right]; }")
        ok(r[0] >= 7.5 and r[1] <= 390 - 7.5, f"③ 패널이 화면 옆으로 넘침 {r}")
        ok(not log["errs"], f"③ JS 오류 {log['errs'][:2]}")
        pg.close()
        b.close()
    after = real_last_order()
    ok(before == after, f"🔴실서버 마지막 주문이 바뀌었다 {before} -> {after} -- 테스트가 실주문을 냈을 수 있다")
    print("\n".join(fails) or "ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(run())
