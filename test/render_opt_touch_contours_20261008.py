#!/usr/bin/env python3
"""Option 카드 행사가 사다리 «닿음 등고선»(2026-10-08)을 실제로 그려서 확인한다.

    python3 -u test/render_opt_touch_contours_20261008.py [--shot DIR]

서버 페이지(터널 127.0.0.1:18787) 위에 로컬 app.js·styles.css·index.html 을 덮는다(render_vol_mult_20261007.py 와 같은 방식).
/api/opt-smile 은 배포 전이라 로컬 front_smile(Deribit 공개 요약)로 채우고, /api/binance-account 응답엔 open_orders 를 덧붙인다.
🔴주문 제출(`/api/manual-*/submit|execute|cancel`)은 브라우저 안에서 막는다 -- 막힌 요청이 있으면 실패.
검사(1500·390): 등고선 6줄(50·25·10% 위아래) · 50% 띠 · 범례 · 내 칩(청산·매수 «닿음 %») · 스마일 없음이면 등고선·칩 0 · JS 오류 0.
"""
import argparse, json, pathlib, sys, time, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dashboard.server import front_smile  # noqa: E402

DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"
ORDERS = [{"symbol": "ETHUSDC", "side": "BUY", "price": 2522.56, "qty": 1.862, "reduce_only": False}]


def smile():
    j = json.loads(urllib.request.urlopen("https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=ETH&kind=option", timeout=20).read())
    return {**front_smile(j["result"], time.time() * 1000), "fetched_ts": time.time()}


PROBE = """() => {
  const svg = document.querySelector('#optCard svg.opt-ladder'); if (!svg) return null;
  const lines = [...svg.querySelectorAll('line')].filter((l) => l.getAttribute('stroke') === 'var(--option)').length;
  const band = [...svg.querySelectorAll('rect')].some((r) => r.getAttribute('fill') === 'var(--option)');
  const texts = [...svg.querySelectorAll('text')].map((t) => t.textContent);
  const leg = document.querySelector('#optCard .opt-touch-legend');
  return { lines, band, legend: leg ? leg.textContent : null, chip: texts.filter((t) => /^(청산|매수|매도) |^닿음 /.test(t)),
           pills: texts.filter((t) => /^(50|25|10)% /.test(t)) };
}"""


def run(shot):
    from playwright.sync_api import sync_playwright
    js, css, html = ((DASH / f).read_text("utf-8") for f in ("app.js", "styles.css", "index.html"))
    sm = smile()
    assert sm.get("ok"), sm
    fails, blocked, errs = [], [], []

    def account(route):
        r = route.fetch()
        body = r.json()
        if isinstance(body, dict) and body.get("ok"):
            body["open_orders"] = ORDERS
        route.fulfill(response=r, body=json.dumps(body), content_type="application/json")

    with sync_playwright() as p:
        b = p.chromium.launch()
        for w, h in ((1500, 900), (390, 844)):
            for case in ("smile", "nosmile"):
                pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2 if w < 500 else 1)
                pg.on("pageerror", lambda e: errs.append(str(e)))
                pg.route(lambda u: "/api/manual-" in u and any(x in u for x in ("/submit", "/execute", "/cancel")),
                         lambda r: (blocked.append(r.request.url), r.abort()))
                pg.route("**/app.js*", lambda r: r.fulfill(body=js, content_type="application/javascript"))
                pg.route("**/styles.css*", lambda r: r.fulfill(body=css, content_type="text/css"))
                pg.route(URL, lambda r: r.fulfill(body=html, content_type="text/html"))
                body = json.dumps(sm if case == "smile" else {"ok": False, "reason": "deribit_unavailable"})
                pg.route("**/api/opt-smile*", (lambda b_: lambda r: r.fulfill(body=b_, content_type="application/json"))(body))
                pg.route("**/api/binance-account*", account)
                pg.goto(URL, wait_until="load", timeout=60000)
                pg.wait_for_timeout(10000)
                pg.evaluate("() => { for (let i = 1; i < 99999; i++) window.clearInterval(i); optLadderScope = 'front'; renderOptions(); }")
                pg.wait_for_timeout(300)
                got = pg.evaluate(PROBE)
                tag = f"{case}·{w}"
                if got is None:
                    fails.append(f"[{tag}] 사다리 없음"); pg.close(); continue
                if case == "smile":
                    if got["lines"] != 6: fails.append(f"[{tag}] 등고선 {got['lines']}줄 (6 기대)")
                    if not got["band"]: fails.append(f"[{tag}] 50% 띠 없음")
                    if not (got["legend"] or "").startswith("선"): fails.append(f"[{tag}] 범례 {got['legend']!r}")
                    if not any(t.startswith("매수 2,522.6") for t in got["chip"]): fails.append(f"[{tag}] 매수 칩 없음 {got['chip']}")
                    if not any(t.startswith("닿음 ") for t in got["chip"]): fails.append(f"[{tag}] 닿음 % 없음")
                else:
                    if got["lines"] or got["band"] or got["legend"] or got["chip"]:
                        fails.append(f"[{tag}] 스마일 없는데 등고선이 그려졌다 {got}")
                print(tag, got, flush=True)
                if shot:
                    pathlib.Path(shot).mkdir(parents=True, exist_ok=True)
                    pg.evaluate("() => { const f = document.getElementById('ofab'); if (f) f.style.visibility = 'hidden'; }")
                    box = pg.evaluate("""() => { const n = document.querySelector('#optCard .opt-sec:has([data-tip="ladder"])');
                      n.scrollIntoView(); const r = n.getBoundingClientRect(); return {x: r.x, y: r.y, width: r.width, height: Math.min(r.height, innerHeight - r.y)}; }""")
                    pg.screenshot(path=str(pathlib.Path(shot) / f"touch_{case}_{w}.png"), clip=box)
                pg.close()
        b.close()
    if blocked: fails.append(f"막힌 주문 요청 {blocked}")
    if errs: fails.append(f"JS 오류 {errs[:3]}")
    print("FAIL" if fails else "PASS", *fails, sep="\n")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shot")
    sys.exit(run(ap.parse_args().shot))
