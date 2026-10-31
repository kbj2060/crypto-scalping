#!/usr/bin/env python3
"""Option 카드 XRP 이식(2026-10-31) -- 닿음 등고선 · 실현 감마 · ⑤ 다음 24시간을 ETH·XRP 둘 다 실제로 그려서 확인한다.

    python3 -u test/render_xrp_options_20261031.py [--shot DIR]

서버 페이지(터널 127.0.0.1:18787) 위에 로컬 app.js·styles.css·index.html 을 덮는다(render_opt_touch_contours_20261008.py 와 같은 방식).
/api/opt-smile 은 배포 전이라 로컬 front_smile(Deribit 공개 요약, XRP = USDC 목록)로 채운다.
🔴주문 제출(`/api/manual-*/submit|execute|cancel`)은 브라우저 안에서 막는다 -- 막힌 요청이 있으면 실패.
검사(1500·390 × ETH·XRP): 등고선 6줄 · 실현 감마 «12시간 자기상관»(XRP 는 rg_xrp 설명 키) · ⑤ 보이고 «옵션 만기» 이름표 · JS 오류 0.
"""
import argparse, json, pathlib, sys, time, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dashboard.server import front_smile  # noqa: E402

DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"


def smile(cur):
    lst, pre = ("USDC", "XRP_USDC-") if cur == "xrp" else (cur.upper(), "")
    j = json.loads(urllib.request.urlopen(f"https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={lst}&kind=option", timeout=20).read())
    return {**front_smile(j["result"], time.time() * 1000, pre), "fetched_ts": time.time()}


PROBE = """() => {
  const svg = document.querySelector('#optCard svg.opt-ladder');
  const lines = svg ? [...svg.querySelectorAll('line')].filter((l) => l.getAttribute('stroke') === 'var(--option)').length : -1;
  const rg = document.querySelector('#optCard .opt-sec:has(.opt-q[data-tip^="rg"])');
  const w = document.getElementById('mcWhen');
  return { lines, rgKey: rg ? rg.querySelector('.opt-q').dataset.tip : null, rgText: rg ? rg.textContent.replace(/\\s+/g, ' ') : null,
           whenShown: !!(w && w.offsetHeight > 0), whenText: w ? w.textContent.replace(/\\s+/g, ' ').slice(0, 300) : null,
           coin: document.body.dataset.coin };
}"""


def run(shot):
    from playwright.sync_api import sync_playwright
    js, css, html = ((DASH / f).read_text("utf-8") for f in ("app.js", "styles.css", "index.html"))
    sm = {c: smile(c) for c in ("eth", "xrp")}
    assert all(v.get("ok") for v in sm.values()), sm
    fails, blocked, errs = [], [], []
    with sync_playwright() as p:
        b = p.chromium.launch()
        for w, h in ((1500, 900), (390, 844)):
            for coin in ("eth", "xrp"):
                pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2 if w < 500 else 1)
                pg.on("pageerror", lambda e: errs.append(str(e)))
                pg.route(lambda u: "/api/manual-" in u and any(x in u for x in ("/submit", "/execute", "/cancel")),
                         lambda r: (blocked.append(r.request.url), r.abort()))
                pg.route("**/app.js*", lambda r: r.fulfill(body=js, content_type="application/javascript"))
                pg.route("**/styles.css*", lambda r: r.fulfill(body=css, content_type="text/css"))
                pg.route(URL, lambda r: r.fulfill(body=html, content_type="text/html"))
                pg.route("**/api/opt-smile*", lambda r: r.fulfill(
                    body=json.dumps(sm["xrp" if "asset=xrp" in r.request.url else "eth"]), content_type="application/json"))
                pg.goto(URL, wait_until="load", timeout=60000)
                pg.wait_for_timeout(8000)
                if coin != "eth":
                    pg.evaluate(f"() => setActiveSnapshotAsset('{coin}')")
                    pg.wait_for_timeout(15000)
                pg.evaluate("() => { gexLastFetchAt = 0; return refreshGex(); }")
                pg.wait_for_timeout(2500)
                pg.evaluate("() => { optLadderScope = 'front'; renderOptions(); renderMarketCtx(); }")
                pg.wait_for_timeout(500)
                got = pg.evaluate(PROBE)
                tag = f"{coin}·{w}"
                print(tag, json.dumps(got, ensure_ascii=False), flush=True)
                if got["lines"] != 6: fails.append(f"[{tag}] 등고선 {got['lines']}줄 (6 기대)")
                want_key = "rg_xrp" if coin == "xrp" else "rg"
                if got["rgKey"] != want_key or "12시간 자기상관" not in (got["rgText"] or ""):
                    fails.append(f"[{tag}] 실현 감마 {got['rgKey']} {got['rgText']!r}")
                if not got["whenShown"] or "옵션 만기" not in (got["whenText"] or ""):
                    fails.append(f"[{tag}] ⑤ 다음 24시간 {got['whenShown']} {got['whenText']!r}")
                if shot:
                    d = pathlib.Path(shot); d.mkdir(parents=True, exist_ok=True)
                    pg.evaluate("() => { const f = document.getElementById('ofab'); if (f) f.style.visibility = 'hidden'; }")
                    for nm, sel in (("ladder_rg", "#optCard"), ("when", "#mcWhen")):
                        box = pg.evaluate(f"""() => {{ const n = document.querySelector('{sel}'); if (!n || !n.offsetHeight) return null;
                          n.scrollIntoView(); const r = n.getBoundingClientRect(); return {{x: r.x, y: r.y, width: r.width, height: Math.min(r.height, innerHeight - r.y)}}; }}""")
                        if box:
                            pg.screenshot(path=str(d / f"{nm}_{coin}_{w}.png"), clip=box)
                pg.close()
        b.close()
    fails += [f"JS 오류: {e}" for e in errs] + [f"주문 요청 막힘: {u}" for u in blocked]
    print("FAIL" if fails else "PASS", *fails, sep="\n")
    return not fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shot")
    sys.exit(0 if run(ap.parse_args().shot) else 1)
