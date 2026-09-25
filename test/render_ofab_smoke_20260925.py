#!/usr/bin/env python3
"""떠다니는 주문 버튼(#ofab)을 **실제로 그려서** 확인한다 (2026-09-25).

    python3 -u test/render_ofab_smoke_20260925.py [--shot DIR]

서버 페이지(터널 127.0.0.1:18787) 위에 **로컬** app.js·styles.css·index.html 을 덮어 배포 전에 본다
(render_dashboard_layout_check_20260922.py 와 같은 방식).
🔴주문 제출(`/api/manual-*/submit`)은 브라우저 안에서 **막는다** -- 이 검사는 주문 버튼을 누르지
  않지만, 누가 고쳐서 누르게 되더라도 실주문이 나갈 길을 원천 차단한다. 막힌 요청이 있으면 실패다.

상태 × 폭(09-22 교훈: 시안은 한 상태를 그리지만 화면은 모든 상태로 산다):
  상태  pos(보유 중 픽스처) · empty(포지션 없음) · 폭 1500×900 · 390×844
검사: 떠 있다 · 펼치면 조작부가 패널로 옮겨 온다 · 게이지가 보이고 칩은 숨는다 · 게이지를 밀면
  접은 뒤 카드의 칩이 같은 값이다 · 패널이 화면 안 · 손잡이로 끌면 옮겨지고 기억된다 ·
  접으면 조작부가 카드로 돌아온다 · 패널 안 겹침/잘림/넘침 0 · JS 오류 0 · 막힌 제출 0.
"""
import argparse, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
DASH = HERE.parent / "dashboard" / "live"
sys.path.insert(0, str(HERE))
from render_dashboard_layout_check_20260922 import SCAN  # noqa: E402  같은 레이아웃 검사기

URL = "http://127.0.0.1:18787/dashboard/live/"
POS = (HERE / "fixtures" / "acct_position.js").read_text("utf-8")
# 🔴포지션 목록만 비우면 안 된다 -- 진입 칸 접힘·청산 줄은 앱이 «포지션 유무가 바뀔 때» 맞춘다.
#   처음 판은 목록만 비워서 «추가 진입(물타기)»이 접힌 채 남은 가짜 상태를 검사했다(라이브에 롱이 있었다).
EMPTY = """() => {
  for (let i = 1; i < 99999; i++) window.clearInterval(i);
  latestBinanceAccount = Object.assign({}, latestBinanceAccount || {}, { positions: [] });
  manualExitSyncButtons();
  renderSnapshotAccount();
  renderOfab();
}"""
BOX = "(s) => { const r = document.querySelector(s)?.getBoundingClientRect(); return r && {x:r.x,y:r.y,w:r.width,h:r.height}; }"


def run(shot):
    from playwright.sync_api import sync_playwright
    js, css, html = ((DASH / f).read_text("utf-8") for f in ("app.js", "styles.css", "index.html"))
    fails = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        for state, fx in (("pos", POS), ("empty", EMPTY)):
            for w, h in ((1500, 900), (390, 844)):
                tag = f"{state}·{w}"
                pg = b.new_page(viewport={"width": w, "height": h})
                errs, blocked = [], []
                pg.on("pageerror", lambda e: errs.append(str(e)))
                pg.route("**/api/manual-*/submit*", lambda r: (blocked.append(r.request.url), r.abort()))
                pg.route("**/app.js*", lambda r: r.fulfill(body=js, content_type="application/javascript"))
                pg.route("**/styles.css*", lambda r: r.fulfill(body=css, content_type="text/css"))
                pg.route(URL, lambda r: r.fulfill(body=html, content_type="text/html"))
                # 🔴networkidle 은 못 쓴다 -- 수급 패널이 0.25초마다 폴링해서 영영 안 올 수 있다(두 번째 실행에서 60초 초과).
                pg.goto(URL, wait_until="load", timeout=60000)
                pg.wait_for_timeout(5000)
                pg.evaluate("() => localStorage.removeItem('ofabPos')")
                pg.evaluate(fx)
                pg.evaluate("() => { renderOfab(); ofabSync(); }")
                pg.wait_for_timeout(400)
                bad = lambda msg: fails.append(f"[{tag}] {msg}")
                ok = lambda c, msg: None if c else bad(msg)

                # 2026-09-25 접기 폐지: 포지션이 있어도 진입 칸은 열려 있고, 제목 줄을 눌러도 안 접힌다.
                pg.evaluate("() => { manualExitSyncButtons(); document.getElementById('snapEntrySummary').click(); }")
                ok(pg.evaluate("() => document.getElementById('snapEntryBox').open"), "진입 칸이 접혔다(상시 표시여야 한다)")
                bar = pg.evaluate(BOX, "#ofab .ofab-bar")
                ok(bar and 0 <= bar["x"] and bar["x"] + bar["w"] <= w and 0 <= bar["y"] and bar["y"] + bar["h"] <= h,
                   f"버튼이 화면 밖/없음 {bar}")
                label = pg.inner_text("#ofabPos")
                ok(("LONG" in label) if state != "empty" else (label == "주문"), f"버튼 글자 «{label}»")
                if shot: pg.screenshot(path=f"{shot}/ofab_{state}_{w}_closed.png")

                pg.click("#ofabToggle")
                pg.wait_for_timeout(300)
                ok(pg.evaluate("() => !!document.querySelector('#ofabPanel .acct-lanes')"), "펼쳐도 조작부가 안 옮겨 옴")
                ok(pg.evaluate("() => !document.getElementById('ofabAway').hidden"), "카드 자리 안내가 안 뜸")
                ok(pg.get_attribute("#ofabToggle", "aria-expanded") == "true", "aria-expanded")
                ok(pg.evaluate("() => !!document.querySelector('#ofabPanel #snapEntryLong')"), "진입 버튼이 패널에 없음")
                for gid in ("snapLevGauge", "snapEntryFrac", "snapExitFrac"):
                    g = pg.evaluate("""(id) => { const i = document.getElementById(id);
                        if (!i || i.hidden || i.closest('[hidden],details:not([open])')) return null;
                        const cs = getComputedStyle(i), r = i.getBoundingClientRect(),
                              chips = document.querySelector(`.chipset[data-for="${id}"]`);
                        return {op: cs.opacity, w: r.width, chips: chips ? getComputedStyle(chips).display : 'x'}; }""", gid)
                    if g is None:
                        continue       # 이 상태에서 안 보이는 게이지(예: 포지션 없을 때 청산 비율)
                    # 레버 «자동»이면 게이지는 disabled 라 0.55 로 흐리다 -- 의도된 «잠김» 표시지 안 보이는 게 아니다.
                    ok(float(g["op"]) >= 0.5 and g["w"] >= 90, f"{gid} 게이지가 안 보임 {g}")
                    ok(g["chips"] == "none", f"{gid} 칩이 패널에서 안 숨음 {g}")
                # «자동»은 레버를 푸는 토글이다 -- **비율 게이지 줄**에 붙으면 비율 설정으로 읽힌다(09-25 스크린샷에서 발견).
                lay = pg.evaluate("""() => { const q = (s) => { const e = document.querySelector('#ofabPanel ' + s);
                    if (!e || e.closest('[hidden],details:not([open])')) return null;
                    const r = e.getBoundingClientRect(); return r.height ? {t: r.top, b: r.bottom} : null; };
                    return {auto: q('.lev-auto'), lev: q('#snapLevGauge'), frac: q('#snapEntryFrac')}; }""")
                if lay["auto"] and lay["frac"]:
                    ok(lay["auto"]["b"] <= lay["frac"]["t"] + 1, f"«자동»이 비율 게이지 줄에 붙음 {lay}")
                if lay["auto"] and lay["lev"]:
                    ok(lay["auto"]["b"] <= lay["lev"]["t"] + 1, f"«자동»이 레버 게이지 위에 없음 {lay}")
                pan = pg.evaluate(BOX, "#ofabPanel")
                ok(pan and pan["x"] >= -1 and pan["x"] + pan["w"] <= w + 1 and pan["y"] >= -1 and pan["y"] + pan["h"] <= h + 1,
                   f"패널이 화면 밖 {pan}")
                # 🔴접힌 <details> 안은 Chromium 이 상자를 남겨 두어 검사기가 거짓 «잘림»을 낸다 --
                #   접힌 상태는 스크린샷으로 보고, 레이아웃 검사는 펼친 상태에서만 한다.
                if not pg.evaluate("() => !!document.querySelector('#ofabPanel details:not([open])')"):
                    scan = pg.evaluate(SCAN, ["#ofabPanel", True])
                    for x in scan.get("bad", [])[:6]:
                        bad(f"레이아웃 {x}")
                ok(pg.evaluate("() => { const b = document.getElementById('ofabBack'); return getComputedStyle(b).fontFamily === getComputedStyle(document.body).fontFamily; }"),
                   "«되돌리기» 버튼이 페이지 폰트를 안 씀(한글이 □ 로 깨진다)")
                # 게이지를 밀면(같은 input) 접은 뒤 카드 칩이 같은 값이어야 한다
                folded = pg.evaluate("() => !document.getElementById('snapEntryBox').open")
                pg.evaluate("""() => { const i = document.getElementById('snapEntryFrac');
                    i.value = 50; i.dispatchEvent(new Event('input', {bubbles: true})); }""")
                ok(pg.evaluate("() => document.getElementById('snapEntryFracVal').textContent.trim()") == "50%",
                   "게이지 값 표시가 안 따라옴")
                if shot: pg.screenshot(path=f"{shot}/ofab_{state}_{w}_open.png")

                pg.click("#ofabToggle")
                pg.wait_for_timeout(200)
                ok(pg.evaluate("() => !document.querySelector('#ofabPanel .acct-lanes')"), "접어도 조작부가 안 돌아감")
                ok(pg.evaluate("() => document.getElementById('ofabAway').hidden"), "카드 자리 안내가 안 사라짐")
                ok(pg.evaluate("() => document.querySelector('.chipset[data-for=snapEntryFrac] .chip[data-v=\"50\"]').classList.contains('on')"),
                   "카드 칩이 게이지 값(50%)을 안 따라감")

                g0 = pg.evaluate(BOX, "#ofabGrip")
                pg.mouse.move(g0["x"] + g0["w"] / 2, g0["y"] + g0["h"] / 2)
                pg.mouse.down()
                pg.mouse.move(g0["x"] - w * 0.4, g0["y"] - h * 0.6, steps=8)
                pg.mouse.up()
                pg.wait_for_timeout(150)
                g1 = pg.evaluate(BOX, "#ofab .ofab-bar")
                ok(g1["y"] < g0["y"] - 50 and 0 <= g1["x"] and g1["x"] + g1["w"] <= w and g1["y"] >= 0,
                   f"끌기가 안 됐거나 화면 밖 {g0} -> {g1}")
                ok(pg.evaluate("() => !!localStorage.getItem('ofabPos')"), "위치가 기억 안 됨")
                pg.click("#ofabToggle")
                pg.wait_for_timeout(300)
                pan2 = pg.evaluate(BOX, "#ofabPanel")
                ok(pan2 and pan2["y"] >= -1 and pan2["y"] + pan2["h"] <= h + 1, f"옮긴 뒤 연 패널이 화면 밖 {pan2}")
                if shot: pg.screenshot(path=f"{shot}/ofab_{state}_{w}_moved_open.png")
                pg.click("#ofabToggle")

                ok(not errs, f"JS 오류 {errs[:2]}")
                ok(not blocked, f"🔴주문 제출 시도가 있었다 {blocked}")
                print(f"[{tag}] {'ok' if not [f for f in fails if f.startswith(f'[{tag}]')] else '✗'}", flush=True)
                pg.close()
        b.close()
    for f in fails:
        print("  ", f, flush=True)
    print(f"\n합계 {len(fails)}건", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", help="스크린샷 디렉터리")
    a = ap.parse_args()
    if a.shot:
        pathlib.Path(a.shot).mkdir(parents=True, exist_ok=True)
    sys.exit(run(a.shot))
