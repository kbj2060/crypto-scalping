#!/usr/bin/env python3
"""XRP 4h ±1σ 부채꼴(QX4, IV 없음)·⑤ 예측 줄 코인별 툴팁 (2026-10-31, 감사표 A-1 #2·B-3·B-4) 실화면 확인.

    python3 -u test/render_xrp_fan_tips_20261031.py --shot DIR

서버 페이지(터널 127.0.0.1:18787) 위에 로컬 app.js·styles.css·index.html 을 덮는다(render_box_hold_20261016.py 와 같은 방식). GET 아닌 요청은 막는다.
검사(1920·390, 미래 칸 열기): XRP = 부채꼴 글자 «4h 1σ ±x%» 1개 · «옵션 IV» 없음 · σ = /api/vol-forecast?asset=xrp 의 sig4h_pct ·
«지금 추세» 툴팁에 XRP 실측 · «7일 예측» 툴팁에 XRP 60% / ETH = 7일 툴팁 ETH 65% · 추세 툴팁에 XRP 없음 · 가로 스크롤 없음 · JS 오류 0.
"""
import argparse, json, pathlib, sys, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"

PROBE = """() => {
  const fan = [...document.querySelectorAll('svg text')].map((t) => t.textContent).filter((s) => /^4h 1σ ±/.test(s));
  const tip = (hz) => { const c = [...document.querySelectorAll('.fc-c')].find((e) => (e.querySelector('.hz') || {}).textContent === hz); return c ? c.title : null; };
  return { fan, trend: tip('지금 추세'), cb: tip('7일 예측'), hscroll: document.documentElement.scrollWidth > window.innerWidth + 1 };
}"""


def run(shot):
    from playwright.sync_api import sync_playwright
    js, css, html = ((DASH / f).read_text("utf-8") for f in ("app.js", "styles.css", "index.html"))
    sig = json.load(urllib.request.urlopen("http://127.0.0.1:18787/api/vol-forecast?asset=xrp", timeout=30))["sig4h_pct"]
    fails, blocked, errs = [], [], []
    shot = pathlib.Path(shot); shot.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        br = p.chromium.launch()
        for asset in ("xrp", "eth"):
            for w, h in ((1920, 1080), (390, 844)):
                tag = f"{asset}_{w}"
                pg = br.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
                pg.on("pageerror", lambda e, t=tag: errs.append(f"{t}: {e}"))
                pg.route("**/*", lambda r: (blocked.append(r.request.url), r.abort()) if r.request.method != "GET" else r.fallback())
                pg.route("**/app.js*", lambda r: r.fulfill(body=js, content_type="application/javascript"))
                pg.route("**/styles.css*", lambda r: r.fulfill(body=css, content_type="text/css"))
                pg.route(URL, lambda r: r.fulfill(body=html, content_type="text/html"))
                pg.goto(URL, wait_until="domcontentloaded", timeout=90000)
                pg.wait_for_timeout(8000)
                if asset != "eth":
                    pg.evaluate(f"() => setActiveSnapshotAsset('{asset}')")
                    pg.wait_for_timeout(12000)
                pg.evaluate("() => { chartPanRight = chartPanRightMax(); }")
                pg.wait_for_timeout(5000)
                pr = pg.evaluate(PROBE)
                if asset == "xrp":
                    ok = (len(pr["fan"]) == 1 and "옵션 IV" not in pr["fan"][0] and f"±{sig:.2f}%" in pr["fan"][0]
                          and bool(pr["trend"]) and "XRP 실측" in pr["trend"] and bool(pr["cb"]) and "XRP 적중 60%" in pr["cb"])
                else:
                    ok = bool(pr["cb"] and "ETH 적중 65%" in pr["cb"] and "XRP" not in (pr["trend"] or ""))
                ok = ok and not pr["hscroll"]
                print(tag, "OK" if ok else "FAIL", pr)
                if not ok:
                    fails.append(tag)
                pg.screenshot(path=str(shot / f"{tag}.png"), full_page=(w == 390))
                pg.close()
        br.close()
    if blocked:
        fails.append(f"GET 아닌 요청 {len(blocked)}건 막힘: {blocked[:3]}")
    if errs:
        fails.append(f"JS 오류 {errs[:3]}")
    print("PASS" if not fails else f"FAIL {fails}")
    return not fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shot", required=True)
    sys.exit(0 if run(ap.parse_args().shot) else 1)
