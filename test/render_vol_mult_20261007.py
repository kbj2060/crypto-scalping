#!/usr/bin/env python3
"""권장 배수(#snapVolMult · #ofabVolMult)를 실제로 그려서 확인한다 (2026-10-07).

    python3 -u test/render_vol_mult_20261007.py [--shot DIR]

서버 페이지(터널 127.0.0.1:18787) 위에 로컬 app.js·styles.css·index.html 을 덮는다(render_ofab_smoke_20260925.py 와 같은 방식).
🔴주문 제출(`/api/manual-*/submit`)은 브라우저 안에서 막는다 -- 막힌 요청이 있으면 실패.
상태: 값 ×0.61(보통) · ×2.40(주황) · 값 없음(숨김) · 비ETH(숨김) × 포지션 없음(LONG·SHORT 막대) × 폭 1500·390.
검사: 보임/숨김 · 글자 · 화면 안 · 진입 줄 넘침 0 · 떠 있는 막대에서 LONG 과 안 겹침 · 호버 풀이 뜸 · JS 오류 0 · 막힌 제출 0.
"""
import argparse, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"
SETUP = """(vm) => {
  for (let i = 1; i < 99999; i++) window.clearInterval(i);
  latestBinanceAccount = Object.assign({}, latestBinanceAccount || {}, { positions: [] });
  manualExitSyncButtons(); renderSnapshotAccount(); renderOfab(); ofabSync();
  latestSituation = Object.assign({}, latestSituation || {}, { vol_mult: vm.v });
  activeSnapshotAsset = vm.asset;
  renderVolMult();
}"""
BOX = "(s) => { const n = document.querySelector(s); if (!n || n.hidden) return null; const r = n.getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}; }"
CASES = (("보통", {"mult": 0.6140, "sigma_bp": 74.0, "bar": 0}, "eth", "×0.6", False),
         ("고변동", {"mult": 2.40, "sigma_bp": 289.3, "bar": 0}, "eth", "×2.4", True),
         ("값없음", None, "eth", None, False),
         ("SOL", {"mult": 1.2, "sigma_bp": 145.0, "bar": 0}, "sol", None, False))


def run(shot):
    from playwright.sync_api import sync_playwright
    js, css, html = ((DASH / f).read_text("utf-8") for f in ("app.js", "styles.css", "index.html"))
    fails = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        for w, h in ((1500, 900), (390, 844)):
            pg = b.new_page(viewport={"width": w, "height": h})
            errs, blocked = [], []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.route(lambda u: "/api/manual-" in u and "/submit" in u, lambda r: (blocked.append(r.request.url), r.abort()))
            pg.route("**/app.js*", lambda r: r.fulfill(body=js, content_type="application/javascript"))
            pg.route("**/styles.css*", lambda r: r.fulfill(body=css, content_type="text/css"))
            pg.route(URL, lambda r: r.fulfill(body=html, content_type="text/html"))
            pg.goto(URL, wait_until="load", timeout=60000)
            pg.wait_for_timeout(5000)
            for name, v, asset, txt, hi in CASES:
                tag = f"{name}·{w}"
                bad = lambda msg: fails.append(f"[{tag}] {msg}")  # noqa: E731
                pg.evaluate(SETUP, {"v": v, "asset": asset})
                pg.wait_for_timeout(250)
                lane, fab = pg.evaluate(BOX, "#snapVolMult"), pg.evaluate(BOX, "#ofabVolMult")
                if txt is None:
                    if lane or fab:
                        bad("숨겨야 하는데 보인다")
                    continue
                if not lane or not fab:
                    bad(f"안 보인다 lane={lane} fab={fab}"); continue
                got = pg.evaluate("() => [document.querySelector('#snapVolMult b').textContent, document.getElementById('ofabVolMult').textContent]")
                if got != [txt, txt]:
                    bad(f"글자 {got} ≠ {txt}")
                if pg.evaluate("() => document.getElementById('ofabVolMult').classList.contains('hi')") != hi:
                    bad("주황 상태가 틀렸다")
                for nm, r in (("떠 있는 배수", fab), ("떠 있는 LONG", pg.evaluate(BOX, "#ofabLong")), ("떠 있는 SHORT", pg.evaluate(BOX, "#ofabShort"))):
                    if r and (r["x"] < 0 or r["x"] + r["w"] > w + 0.5):
                        bad(f"{nm} 화면 밖 {r}")
                lg = pg.evaluate(BOX, "#ofabLong")
                if lg and fab["x"] + fab["w"] > lg["x"] + 0.5:
                    bad(f"배수가 LONG 과 겹친다 {fab} {lg}")
                row = pg.evaluate("() => { const r = document.getElementById('snapVolMult').closest('.manual-entry-row'); return [r.scrollWidth, r.clientWidth]; }")
                if row[0] > row[1] + 1:
                    bad(f"진입 줄 넘침 {row}")
                pg.evaluate("() => document.getElementById('snapAcctPosition').closest('.panel').scrollIntoView()")
                pg.hover("#snapVolMult"); pg.wait_for_timeout(150)
                tip = pg.evaluate("() => { const t = document.getElementById('chartTooltip'); return t.classList.contains('visible') ? t.innerText : ''; }")
                if "권장 크기" not in tip or "전진 검정" not in tip:
                    bad(f"호버 풀이가 안 뜬다: {tip[:60]!r}")
                tl = pg.evaluate("() => { const r = document.getElementById('chartTooltip').getBoundingClientRect(); return [r.top, r.bottom, innerHeight]; }")
                if tl[0] < 0 or tl[1] > tl[2] + 0.5:
                    bad(f"진입 줄 풀이가 화면 밖 {tl}")
                pg.mouse.move(1, 1); pg.hover("#ofabVolMult"); pg.wait_for_timeout(150)
                tb = pg.evaluate("() => { const t = document.getElementById('chartTooltip'); if (!t.classList.contains('visible')) return null;"
                                 " const r = t.getBoundingClientRect(); return [r.top, r.bottom, innerHeight, r.left, r.right, innerWidth]; }")
                if not tb or tb[0] < 0 or tb[1] > tb[2] + 0.5 or tb[3] < 0 or tb[4] > tb[5] + 0.5:
                    bad(f"떠 있는 배수 풀이가 화면 밖/안 뜸 {tb}")
                if shot:
                    pathlib.Path(shot).mkdir(parents=True, exist_ok=True)
                    pg.screenshot(path=f"{shot}/volmult_{name}_{w}.png")
                    pg.locator("#ofab").screenshot(path=f"{shot}/volmult_ofab_{name}_{w}.png")
                pg.mouse.move(1, 1)
            if errs:
                fails.append(f"[{w}] JS 오류 {errs[:3]}")
            if blocked:
                fails.append(f"[{w}] 막힌 제출 {blocked}")
            pg.close()
        b.close()
    print("\n".join(fails) if fails else "OK 권장 배수 4상태 × 2폭")
    return not fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shot")
    sys.exit(0 if run(ap.parse_args().shot) else 1)
