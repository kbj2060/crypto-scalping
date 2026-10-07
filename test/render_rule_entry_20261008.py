#!/usr/bin/env python3
"""4분할 규칙 진입 화면(A안 «거리 자») 렌더 검사 (2026-10-08).

    python3 -u test/render_rule_entry_20261008.py [--shot DIR]

서버 페이지(터널 127.0.0.1:18787) 위에 로컬 app.js·styles.css·index.html. 게이지 응답은 서버 함수 build_rule_ladder 로 만든 가짜.
🔴실주문 차단 두 겹: ① 브라우저의 **GET 이 아닌 모든 요청** 을 abort ② 미리보기(/api/manual-entry/preview)도 가짜 응답으로 가로챈다(서버 미도달).
   GET 아닌 요청이 하나라도 시도되면 실패로 적는다.
검사: 모드 전환·비율 칩 숨김·게이지 숫자·버튼 이름표·떠 있는 버튼·미리보기 쿼리 mode=rule·직접으로 되돌림·가로 넘침·자 이름표 겹침·JS 오류.
"""
import argparse, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scripts.live_manual_peg_entry_20260912 import build_rule_ladder  # noqa: E402

DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"
F = {"tick": 0.01, "step": 0.001, "min_qty": 0.001, "min_notional": 20.0}


def gauge(equity=2600.0, price=2569.5, sigma4h=104.7, mult=0.87, lev=20, flat=True):
    s = sigma4h * 6 ** 0.5
    out = {side.lower(): build_rule_ladder(side=side, equity=equity, price=price, sigma24_bp=s, mult=mult,
                                           cap_notional=equity * lev * 0.5, filters=F) for side in ("LONG", "SHORT")}
    return {"ok": True, "flat": flat, "lev": lev, "price": price, "equity": equity, **out}


STATE_JS = """() => { const q = (s) => document.querySelector(s), vis = (n) => !!n && !n.hidden && n.getClientRects().length > 0;
  const chips = q('[data-for=snapEntryFrac]');
  return { gauge: vis(q('#snapRuleGauge')), chips: vis(chips) && getComputedStyle(chips).display !== 'none',
    cap: q('#snapEntryFracCap').textContent, head: q('#snapRuleGauge .rq-head')?.textContent || '', cut: q('#snapRuleGauge .rq-cut')?.textContent || '',
    long: q('#snapEntryLong').textContent, ofab: q('#ofabLong')?.textContent || '', stored: localStorage.getItem('entryMode'),
    over: [q('#snapRuleGauge').scrollWidth - q('#snapRuleGauge').clientWidth, document.documentElement.scrollWidth - innerWidth],
    labs: [...document.querySelectorAll('#snapRuleGauge .rq-lab')].map(n => n.getBoundingClientRect().toJSON()),
    box: q('#snapRuleGauge').getBoundingClientRect().toJSON() }; }"""


def run(shot):
    from playwright.sync_api import sync_playwright
    js, css, html = ((DASH / f).read_text("utf-8") for f in ("app.js", "styles.css", "index.html"))
    fails, G = [], gauge()
    with sync_playwright() as p:
        b = p.chromium.launch()
        for w, h in ((1500, 1000), (390, 844)):
            pg = b.new_page(viewport={"width": w, "height": h})
            errs, posts, previews = [], [], []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.route(lambda u: True, lambda r: (posts.append(f"{r.request.method} {r.request.url}"), r.abort())
                     if r.request.method != "GET" else r.fallback())
            pg.route("**/api/manual-entry/rule-gauge*", lambda r: r.fulfill(body=json.dumps(G), content_type="application/json"))
            pg.route(lambda u: "/api/manual-entry/preview" in u,
                     lambda r: (previews.append(r.request.url),
                                r.fulfill(body=json.dumps({"ok": False, "error": "harness", "detail": "시험 하네스 — 미리보기 가로챔"}),
                                          content_type="application/json")))
            pg.route("**/app.js*", lambda r: r.fulfill(body=js, content_type="application/javascript"))
            pg.route("**/styles.css*", lambda r: r.fulfill(body=css, content_type="text/css"))
            pg.route(URL, lambda r: r.fulfill(body=html, content_type="text/html"))
            pg.goto(URL, wait_until="load", timeout=60000)
            pg.wait_for_timeout(5000)
            pg.evaluate("() => { for (let i = 1; i < 99999; i++) window.clearInterval(i); try { localStorage.removeItem('entryMode'); } catch (_) {} "
                        "entryRuleMode = null; activeSnapshotAsset = 'eth'; renderRuleGauge(); }")
            tag = lambda m: f"[{w}] {m}"  # noqa: E731
            if pg.evaluate("() => document.getElementById('snapEntryMode').hidden"):
                fails.append(tag("ETH 인데 모드 전환이 숨어 있다"))
            pg.evaluate("() => document.querySelector('#snapEntryMode [data-mode=rule]').click()"); pg.wait_for_timeout(700)
            st = pg.evaluate(STATE_JS)
            L = G["long"]
            if not st["gauge"] or st["chips"]:
                fails.append(tag(f"규칙 모드 표시 틀림 gauge={st['gauge']} chips={st['chips']}"))
            if st["cap"] != "진입" or st["stored"] != "rule":
                fails.append(tag(f"이름표/기억 틀림 {st['cap']} {st['stored']}"))
            for want in (f"×{L['mult']:.2f}", f"{L['total_x']:.1f}배", f"{round(L['loss_pct'])}%"):
                if want not in st["head"]:
                    fails.append(tag(f"게이지 머리에 {want} 없음: {st['head'][:140]!r}"))
            if L["binding"] != "rule" and "줄임" not in st["cut"]:
                fails.append(tag("크기 축소 안내 없음"))
            if "4분할 진입" not in st["long"] or "손절" not in st["long"]:
                fails.append(tag(f"롱 버튼 이름표 {st['long']!r}"))
            if "4분할" not in st["ofab"]:
                fails.append(tag(f"떠 있는 버튼 {st['ofab']!r}"))
            if st["over"][0] > 1 or st["over"][1] > 0:
                fails.append(tag(f"가로 넘침 {st['over']}"))
            bx = st["box"]
            if any(r["x"] < bx["x"] - 0.5 or r["x"] + r["width"] > bx["x"] + bx["width"] + 0.5 for r in st["labs"]):
                fails.append(tag("자 이름표가 상자 밖"))
            ov = [(i, j) for i, a in enumerate(st["labs"]) for j, c in enumerate(st["labs"]) if i < j
                  and a["x"] < c["x"] + c["width"] and c["x"] < a["x"] + a["width"] and a["y"] < c["y"] + c["height"] and c["y"] < a["y"] + a["height"]]
            if ov:
                fails.append(tag(f"자 이름표 겹침 {ov}"))
            if shot:
                pathlib.Path(shot).mkdir(parents=True, exist_ok=True)
                pg.evaluate("() => document.getElementById('snapRuleGauge').scrollIntoView({block: 'center'})"); pg.wait_for_timeout(300)
                pg.locator("#snapEntryBox").screenshot(path=f"{shot}/rule_entry_{w}.png")
            pg.evaluate("() => manualEntryPreview('LONG', 'entry')"); pg.wait_for_timeout(500)
            if not previews or "mode=rule" not in previews[-1]:
                fails.append(tag(f"미리보기 쿼리에 mode=rule 없음 {previews[-1:]}"))
            pg.evaluate("() => document.querySelector('#snapEntryMode [data-mode=direct]').click()"); pg.wait_for_timeout(300)
            back = pg.evaluate("() => [document.getElementById('snapRuleGauge').hidden, document.getElementById('snapEntryLong').textContent,"
                               " document.getElementById('snapEntryFracCap').textContent]")
            if not back[0] or "4분할" in back[1] or back[2] != "진입비율":
                fails.append(tag(f"직접으로 안 돌아옴 {back}"))
            pg.evaluate("() => manualEntryPreview('LONG', 'entry')"); pg.wait_for_timeout(500)
            if previews and "mode=rule" in previews[-1]:
                fails.append(tag("직접 모드인데 mode=rule 이 실렸다"))
            if errs:
                fails.append(tag(f"JS 오류 {errs[:3]}"))
            if posts:
                fails.append(tag(f"🔴GET 아닌 요청 시도 {posts}"))
            pg.close()
        b.close()
    print("\n".join(fails) if fails else "OK 4분할 규칙 화면 × 2폭 · GET 아닌 요청 0 · 미리보기 mode=rule")
    return not fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shot")
    sys.exit(0 if run(ap.parse_args().shot) else 1)
