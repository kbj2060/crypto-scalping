#!/usr/bin/env python3
"""«손절 해제»가 플로팅 버튼 안에 있다(2026-10-15 사용자 «데스크탑은 B · 모바일은 C»).

    python3 -u test/render_ofab_sl_20261015.py [--shot DIR]     # 터널 127.0.0.1:18787 필요

서버 페이지에 로컬 app.js·styles.css·index.html 을 덮는다. 🔴해제 POST 는 브라우저 안에서 **가짜 응답**으로 받고
그 밖의 GET 아닌 /api 요청은 전부 막는다(막힌 게 있으면 실패). 무장 손절은 픽스처로 심는다(라이브 포지션 유무와 무관).
검사: 카드에 손절 줄 없음 · 1500폭 = 알약 칩 보임·패널 칸 숨김 · 390폭 = 칩 숨김·패널 맨 위 칸 보임 ·
  한 번 누름 = POST 0 + «한 번 더» · 두 번째 = POST 1(side·symbol·confirm) · sl_off 면 «손절 없음» · 포지션 없으면 둘 다 숨김 · JS 오류 0.
"""
import argparse, json, pathlib, sys

DASH = pathlib.Path(__file__).resolve().parent.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"
FIX = """(off) => {
  for (let i = 1; i < 99999; i++) window.clearInterval(i);
  latestBinanceAccount = Object.assign({}, latestBinanceAccount || {}, {
    positions: [{ symbol: "ETHUSDC", side: "LONG", qty: 0.433, leverage: 10, entry_price: 2432.48, mark_price: 2432.34,
                  liquidation_price: 0, notional: 1053.2, unrealized_pnl: -0.06 }],
    bracket_armed: { "ETHUSDC:LONG": { sl_price: 2310.85, sl_name: "첫 진입가 −5%", sl_off: off ? { at: 1, cancelled: 1 } : null, rule: true } } });
  refreshBinanceAccount = () => {};   // 결과 뒤 다시 읽기가 픽스처를 덮지 않게
  renderSnapshotAccount(); renderOfab();
}"""
VIS = "(s) => { const e = document.querySelector(s); if (!e) return false; const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0 && getComputedStyle(e).visibility !== 'hidden'; }"


def run(shot):
    from playwright.sync_api import sync_playwright
    js, css, html = ((DASH / f).read_text("utf-8") for f in ("app.js", "styles.css", "index.html"))
    fails = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        for w, h in ((1500, 900), (390, 844)):
            mob = w <= 720
            pg = b.new_page(viewport={"width": w, "height": h})
            errs, blocked, disarm = [], [], []
            pg.on("pageerror", lambda e: errs.append(str(e)))

            def api(r):
                if r.request.method == "GET":
                    return r.continue_()
                if "/api/manual-bracket/disarm" in r.request.url:
                    disarm.append(r.request.url)
                    return r.fulfill(body=json.dumps({"ok": True}), content_type="application/json")
                blocked.append(r.request.url); r.abort()
            pg.route(lambda u: "/api/" in u, api)
            pg.route("**/app.js*", lambda r: r.fulfill(body=js, content_type="application/javascript"))
            pg.route("**/styles.css*", lambda r: r.fulfill(body=css, content_type="text/css"))
            pg.route(URL, lambda r: r.fulfill(body=html, content_type="text/html"))
            pg.goto(URL, wait_until="load", timeout=60000)
            pg.wait_for_timeout(5000)
            pg.evaluate("() => { localStorage.removeItem('ofabPos'); ofab.x = null; ofab.y = null; ofabPlace(innerWidth, innerHeight, false); }")
            pg.evaluate(FIX, False)
            pg.wait_for_timeout(200)
            ok = lambda c, m: None if c else fails.append(f"[{w}] {m}")
            ok(not pg.evaluate("() => /손절 해제|data-sl-off/.test(document.getElementById('snapAcctPosition').innerHTML)"), "계좌 카드에 손절 줄이 남았다")
            if not mob:
                ok(pg.evaluate(VIS, "#ofabSl"), "데스크탑 알약에 손절 칩이 없다")
                ok("2,310.85" in pg.inner_text("#ofabSl") or "2310.85" in pg.inner_text("#ofabSl"), f"칩 글자 {pg.inner_text('#ofabSl')!r}")
                bar = pg.evaluate("() => { const r = document.querySelector('#ofab .ofab-bar').getBoundingClientRect(); return [r.left, r.right]; }")
                ok(bar[0] >= 0 and bar[1] <= w, f"칩이 붙은 알약이 화면 밖 {bar}")
                tgt = "#ofabSl"
            else:
                ok(not pg.evaluate(VIS, "#ofabSl"), "휴대폰 알약에 칩이 보인다(패널 칸이어야)")
                pg.click("#ofabToggle"); pg.wait_for_timeout(300)
                ok(pg.evaluate(VIS, "#ofabSlPanel"), "휴대폰 패널에 손절 칸이 없다")
                ok(pg.evaluate("() => document.getElementById('ofabPanel').firstElementChild.id === 'ofabSlPanel'"), "손절 칸이 패널 맨 위가 아니다")
                tgt = "#ofabSlPanel [data-sl-off]"
            if not mob:
                ok(not pg.evaluate(VIS, "#ofabSlPanel"), "데스크탑 패널에 휴대폰용 칸이 보인다")
            if shot: pg.screenshot(path=f"{shot}/sl_{w}_idle.png")
            pg.click(tgt); pg.wait_for_timeout(150)
            ok(not disarm, "한 번 눌렀는데 해제 요청이 나갔다")
            ok("한 번 더" in pg.inner_text(tgt), f"첫 누름 뒤 글자 {pg.inner_text(tgt)!r}")
            if shot: pg.screenshot(path=f"{shot}/sl_{w}_arm.png")
            pg.click(tgt); pg.wait_for_timeout(400)
            ok(len(disarm) == 1 and "side=LONG" in disarm[0] and "symbol=ETHUSDC" in disarm[0] and "confirm=1" in disarm[0], f"해제 요청 {disarm}")
            pg.evaluate(FIX, True); pg.wait_for_timeout(150)
            off_txt = pg.inner_text("#ofabSl" if not mob else "#ofabSlPanel")
            ok("없음" in off_txt, f"해제 후 글자 {off_txt!r}")
            if not mob:
                pg.click("#ofabSl"); pg.wait_for_timeout(150)
                ok(len(disarm) == 1, "«손절 없음» 칩을 누르니 요청이 또 나갔다")
            if shot: pg.screenshot(path=f"{shot}/sl_{w}_off.png")
            pg.evaluate("() => { latestBinanceAccount.positions = []; renderOfab(); }")
            ok(not pg.evaluate(VIS, "#ofabSl") and not pg.evaluate(VIS, "#ofabSlPanel"), "포지션이 없는데 손절 표시가 남았다")
            ok(not errs, f"JS 오류 {errs[:2]}")
            ok(not blocked, f"🔴막힌 쓰기 요청 {blocked}")
            print(f"[{w}] {'ok' if not [f for f in fails if f.startswith(f'[{w}]')] else '✗'}", flush=True)
            pg.close()
        b.close()
    for f in fails:
        print("  ", f)
    print(f"합계 {len(fails)}건")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shot")
    a = ap.parse_args()
    if a.shot: pathlib.Path(a.shot).mkdir(parents=True, exist_ok=True)
    sys.exit(run(a.shot))
