#!/usr/bin/env python3
"""터치는 «탭 → 확인», 마우스는 «길게 누르기» (2026-09-25 사용자 지시) -- 실제로 그려서 확인한다.

    python3 -u test/render_touch_confirm_20260925.py

서버 페이지(터널 127.0.0.1:18787) 위에 로컬 app.js·styles.css·index.html 을 덮는다.
🔴주문 제출은 브라우저 안에서 **막고 센다** -- 실주문은 원리적으로 못 나간다.
미리보기는 실서버(GET, 읽기 전용)를 받되 `blocked` 만 지운다(상한에 걸려도 확인 흐름을 볼 수 있게).
검사: 터치 탭 → 제출 0 · 확인 버튼 뜸 → 확인 탭 → 제출 1 · 마우스 0.6초 누르기 → 제출 1 · JS 오류 0.
"""
import pathlib
import sys

from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"
POS = (HERE / "fixtures" / "acct_position.js").read_text("utf-8")


def page(browser, touch, submits, errs):
    ctx = browser.new_context(viewport={"width": 390, "height": 844} if touch else {"width": 1500, "height": 1000},
                              has_touch=touch, is_mobile=touch)
    pg = ctx.new_page()
    js, css, html = ((DASH / f).read_text("utf-8") for f in ("app.js", "styles.css", "index.html"))
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.route("**/api/manual-*/submit*", lambda r: (submits.append(r.request.url), r.abort()))

    def unblock(route):
        resp = route.fetch()
        d = resp.json()
        (d.get("plan") or {})["blocked"] = None
        route.fulfill(response=resp, json=d)
    pg.route("**/api/manual-entry/preview*", unblock)
    pg.route("**/app.js*", lambda r: r.fulfill(body=js, content_type="application/javascript"))
    pg.route("**/styles.css*", lambda r: r.fulfill(body=css, content_type="text/css"))
    pg.route(URL, lambda r: r.fulfill(body=html, content_type="text/html"))
    pg.goto(URL, wait_until="load", timeout=60000)
    pg.wait_for_timeout(5000)
    pg.evaluate(POS)
    return ctx, pg


def main() -> int:
    fails = []
    ok = lambda c, m: None if c else fails.append(m)
    with sync_playwright() as p:
        b = p.chromium.launch()
        # ── 터치: 탭만으로는 안 나가고, 확인을 눌러야 나간다 ──────────────────────
        submits, errs = [], []
        ctx, pg = page(b, True, submits, errs)
        pg.locator("#snapEntryLong").scroll_into_view_if_needed()
        # 🔴위험은 «탭»이 아니라 **손가락이 0.4초 넘게 버튼 위에 머무는 것**이다(스크롤하다 멈춘 손가락).
        #   옛 코드는 탭이면 0.4초 전에 떼므로 원래 안 나갔다 -- 그래서 탭만 검사하면 음성 대조가 통과한다.
        box = pg.locator("#snapEntryLong").bounding_box()
        pt = [{"x": box["x"] + box["width"] / 2, "y": box["y"] + box["height"] / 2}]
        cdp = ctx.new_cdp_session(pg)
        cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": pt})
        pg.wait_for_timeout(1500)                       # 미리보기 왕복 + 0.4초를 넉넉히 넘긴다
        ok(not submits, f"터치로 누르고 있었더니 제출이 나갔다(길게 누르기 발주가 살아 있다) {submits}")
        cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        pg.wait_for_timeout(300)
        pg.tap("#snapEntryLong")
        pg.wait_for_timeout(2500)
        ok(not submits, f"터치 탭 한 번에 제출이 나갔다 {submits}")
        confirm_shown = pg.evaluate("() => !document.getElementById('snapEntryConfirm').hidden")
        ok(confirm_shown, "터치 탭 뒤 확인 버튼이 안 떴다")
        ok(pg.evaluate("() => getComputedStyle(document.getElementById('snapEntryLong')).touchAction") == "manipulation",
           "터치에서 버튼 위 스크롤이 여전히 막혀 있다(touch-action)")
        ok(pg.evaluate("() => getComputedStyle(document.querySelector('.lane-pact .pact-touch')).display") != "none",
           "터치 안내문이 안 보인다")
        if confirm_shown:
            pg.tap("#snapEntryConfirm")
            pg.wait_for_timeout(800)
            ok(len(submits) == 1, f"확인 탭 뒤 제출이 1건이 아니다 {submits}")
        ok(not errs, f"JS 오류(터치) {errs[:2]}")
        ctx.close()
        # ── 키보드(2026-09-26 비평 P1): Enter 도 «미리보기 → 확인» 경로. 예전엔 아무 반응이 없었다 ──
        submits, errs = [], []
        ctx, pg = page(b, False, submits, errs)
        pg.focus("#snapEntryLong")
        pg.keyboard.press("Enter")
        pg.wait_for_timeout(2500)
        ok(not submits, f"키보드 Enter 한 번에 제출이 나갔다 {submits}")
        kb_confirm = pg.evaluate("() => !document.getElementById('snapEntryConfirm').hidden")
        ok(kb_confirm, "키보드 Enter 뒤 확인 버튼이 안 떴다(키보드 주문 불가)")
        if kb_confirm:
            pg.focus("#snapEntryConfirm")
            pg.keyboard.press("Enter")
            pg.wait_for_timeout(800)
            ok(len(submits) == 1, f"키보드 확인 뒤 제출이 1건이 아니다 {submits}")
        ok(not errs, f"JS 오류(키보드) {errs[:2]}")
        ctx.close()
        # ── 마우스: 길게 누르기는 그대로 ────────────────────────────────────────
        submits, errs = [], []
        ctx, pg = page(b, False, submits, errs)
        box = pg.locator("#snapEntryLong").bounding_box()
        pg.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        pg.mouse.down()
        pg.wait_for_timeout(1500)       # 미리보기 왕복 + 0.4초
        pg.mouse.up()
        pg.wait_for_timeout(500)
        ok(len(submits) == 1, f"마우스 길게 누르기로 제출이 안 나갔다 {submits}")
        ok(not errs, f"JS 오류(마우스) {errs[:2]}")
        ctx.close()
        b.close()
    for f in fails:
        print("  ", f, flush=True)
    print(f"합계 {len(fails)}건", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
