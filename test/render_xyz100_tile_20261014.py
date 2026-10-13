#!/usr/bin/env python3
"""나스닥100 실시간(XYZ100) «⑥ 나스닥 · 금리 · 1시간» 칸을 실제 화면에 그려 확인한다(2026-10-14, 사용자 시안 A 선택).

    python3 -u test/render_xyz100_tile_20261014.py --shot DIR

서버 페이지(터널 127.0.0.1:18787) 위에 로컬 app.js·styles.css·index.html 을 덮는다(render_opt_touch_contours_20261008.py 와 같은 방식).
/api/xyz100 은 배포 전이라 지금 Hyperliquid 공개 API(candleSnapshot 1분 24시간 + metaAndAssetCtxs 현재가)로 채운다.
🔴GET 이 아닌 요청(주문 포함)은 브라우저 안에서 전부 막는다 -- 막힌 요청이 있으면 실패.
세 상태(live = 실데이터 · stale = 초 가격 120초 묵음(수집기 끊김) · none = ok False) × 1920·390:
  1920 = ⑥ 칸 세 칸(나스닥 · 5년 · 10년) · 나스닥 값 live «±N.NN%» / stale·none «끊김» · 390 = 칸 없음(넓은 화면 전용) · JS 오류 0.
"""
import argparse, json, pathlib, time, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"


def hl(body):
    req = urllib.request.Request("https://api.hyperliquid.xyz/info", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def xyz_payload():
    now = int(time.time() * 1000)
    c = hl({"type": "candleSnapshot", "req": {"coin": "xyz:XYZ100", "interval": "1m", "startTime": now - 86_400_000, "endTime": now}})
    meta, ctx = hl({"type": "metaAndAssetCtxs", "dex": "xyz"})
    live = next(x for u, x in zip(meta["universe"], ctx) if u["name"] == "xyz:XYZ100")
    return [[int(x["t"]) // 1000, float(x["c"])] for x in c], {"px": float(live["midPx"] or live["markPx"]), "oracle": float(live["oraclePx"])}


PROBE = """() => {
  const cells = [...document.querySelectorAll('#mcBody .mc-rate-g.n3 .mc-gk')];
  return { n: cells.length, names: cells.map((c) => (c.querySelector('span') || {}).textContent), nq: cells[0] ? (cells[0].querySelector('b') || {}).textContent || '' : null,
           title: (document.querySelector('#mcBody .mc-rate') || {}).textContent?.slice(0, 20) || '' };
}"""


def run(shot):
    import re
    from playwright.sync_api import sync_playwright
    js, css, html = ((DASH / f).read_text("utf-8") for f in ("app.js", "styles.css", "index.html"))
    bars, live = xyz_payload()
    fails, blocked, errs = [], [], []

    def api_for(case):
        def api(route):
            if case == "none":
                return route.fulfill(body=json.dumps({"ok": False, "bars": [], "live": None}), content_type="application/json")
            since = int((route.request.url.split("since=") + ["0"])[1].split("&")[0] or 0)
            b = [x for x in bars if x[0] >= since] if since else bars
            age = 120 if case == "stale" else 0
            route.fulfill(body=json.dumps({"ok": True, "coin": "xyz:XYZ100", "full": not since, "bars": b,
                                           "live": {**live, "recv_sec": int(time.time()) - age}}), content_type="application/json")
        return api
    shot = pathlib.Path(shot); shot.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        br = p.chromium.launch()
        for case in ("live", "stale", "none"):
            for w, h in ((1920, 1080), (390, 844)):
                pg = br.new_page(viewport={"width": w, "height": h}, device_scale_factor=2 if w < 500 else 1)
                pg.on("pageerror", lambda e, c=case, w=w: errs.append(f"{c}{w}: {e}"))
                pg.route("**/*", lambda r: (blocked.append(r.request.url), r.abort()) if r.request.method != "GET" else r.fallback())
                pg.route("**/app.js*", lambda r: r.fulfill(body=js, content_type="application/javascript"))
                pg.route("**/styles.css*", lambda r: r.fulfill(body=css, content_type="text/css"))
                pg.route(URL, lambda r: r.fulfill(body=html, content_type="text/html"))
                pg.route("**/api/xyz100*", api_for(case))
                pg.goto(URL, wait_until="load", timeout=60000)
                pg.wait_for_timeout(14000)
                got = pg.evaluate(PROBE)
                tag = f"{case}·{w}"
                print(tag, json.dumps(got, ensure_ascii=False), flush=True)
                if w > 500:
                    if got["n"] != 3 or got["names"][:1] != ["나스닥"]: fails.append(f"[{tag}] ⑥ 칸 {got}")
                    ok_val = re.fullmatch(r"[+−]\d+\.\d\d%", got["nq"] or "") if case == "live" else got["nq"] == "끊김"
                    if not ok_val: fails.append(f"[{tag}] 나스닥 값 {got['nq']!r}")
                    if pg.locator("#mcBody").count():
                        pg.locator("#mcBody").screenshot(path=str(shot / f"tile_{case}.png"))
                elif got["n"]: fails.append(f"[{tag}] 휴대폰에 ⑥ 칸 {got['n']}")
                pg.close()
        br.close()
    if blocked: fails.append(f"막은 요청 {len(blocked)}: {blocked[:3]}")
    if errs: fails.append(f"JS 오류 {errs[:3]}")
    print("FAIL" if fails else "PASS", fails)
    return not fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default="tmp/xyz100_tile")
    raise SystemExit(0 if run(ap.parse_args().shot) else 1)
