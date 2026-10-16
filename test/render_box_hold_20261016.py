#!/usr/bin/env python3
"""«박스 유지 확률» 시안(2026-10-16 사용자 «①부터 재검증하고 시안»)을 실제 화면에 그려 확인한다.

    python3 -u test/render_box_hold_20261016.py --shot DIR

서버 페이지(터널 127.0.0.1:18787) 위에 로컬 app.js·styles.css·index.html 을 덮는다(render_dir_ladder_20261015.py 와 같은 방식).
/api/box-hold 는 로컬 dashboard/box_hold.py 로 지금 바이낸스 1분봉에서 계산한 값(live)과 «깨질 쪽» 상태(brk: 확률을 평소의 절반으로).
시안 A = 시장 맥락 판 ⑤ 아래 «박스 유지 확률» 줄만 · B(배포판, 10-16) = A + 풋프린트 가격판에 1시간 박스 두 선(BOX_ON_CHART = true) · A 는 끈 판.
🔴GET 이 아닌 요청(주문 포함)은 브라우저 안에서 전부 막는다 -- 막힌 요청이 있으면 실패.
검사(1920·390): 칸 셋(30분·1시간·4시간) · 값 = 응답 확률 · brk 이면 주황 칸(.brk) · 칸 글자 안 넘침 · 가로 스크롤 없음 · JS 오류 0 · B 1920 은 «1h 박스» 꼬리표 둘.
"""
import argparse, json, pathlib, sys, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"

PROBE = """() => {
  const sec = [...document.querySelectorAll('#mcBody .mc-sec')].find((s) => s.querySelector('.bx-row'));
  const cells = sec ? [...sec.querySelectorAll('.bx-c')] : [];
  return { found: !!sec, n: cells.length, hz: cells.map((c) => c.querySelector('.hz').textContent),
           val: cells.map((c) => (c.querySelector('b').firstChild || {}).textContent), brk: cells.map((c) => c.classList.contains('brk')),
           over: cells.some((c) => [...c.children].some((e) => e.scrollWidth > e.clientWidth + 1)),
           hscroll: document.documentElement.scrollWidth > window.innerWidth + 1,
           tags: [...document.querySelectorAll('svg text')].filter((t) => /^1h 박스/.test(t.textContent)).length,
           rect: sec ? (() => { const r = sec.getBoundingClientRect(); return [r.x, r.y + window.scrollY, r.width, r.height]; })() : null };
}"""


def live_payload():
    import pandas as pd
    from dashboard.box_hold import payload
    k = json.load(urllib.request.urlopen("https://fapi.binance.com/fapi/v1/klines?symbol=ETHUSDT&interval=1m&limit=1500", timeout=20))
    d = pd.DataFrame(k).iloc[:-1, :5]            # 형성 중 봉 제외
    d.columns = ["t", "o", "high", "low", "close"]
    d = d.astype(float); d["timestamp"] = pd.to_datetime(d.t, unit="ms", utc=True)
    return payload(d)


def fulfill(body, ctype):
    """🔴핸들러는 인자 1개여야 한다 -- 기본값 인자를 둔 2인자 lambda 는 playwright 가 (route, request) 로 불러 기본값이 덮인다."""
    return lambda r: r.fulfill(body=body, content_type=ctype)


def run(shot):
    from playwright.sync_api import sync_playwright
    js0, css, html = ((DASH / f).read_text("utf-8") for f in ("app.js", "styles.css", "index.html"))
    live = live_payload()
    assert live["ok"], live
    brk = json.loads(json.dumps(live))
    for s in brk["scales"].values():
        s["p"] = round(s["base"] * 0.5, 4)
    fails, blocked, errs = [], [], []
    shot = pathlib.Path(shot); shot.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        br = p.chromium.launch()
        for variant in ("A", "B"):
            js = js0 if variant == "B" else js0.replace("const BOX_ON_CHART = true;", "const BOX_ON_CHART = false;")   # 배포판 = B
            assert variant == "B" or js != js0
            for case, body in (("live", live), ("brk", brk)):
                for w, h in ((1920, 1080), (390, 844)):
                    tag = f"{variant}_{case}_{w}"
                    pg = br.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
                    pg.on("pageerror", lambda e, t=tag: errs.append(f"{t}: {e}"))
                    pg.route("**/*", lambda r: (blocked.append(r.request.url), r.abort()) if r.request.method != "GET" else r.fallback())
                    pg.route("**/app.js*", fulfill(js, "application/javascript"))
                    pg.route("**/styles.css*", lambda r: r.fulfill(body=css, content_type="text/css"))
                    pg.route("**/api/box-hold*", fulfill(json.dumps(body), "application/json"))
                    pg.route(URL, lambda r: r.fulfill(body=html, content_type="text/html"))
                    pg.goto(URL, wait_until="domcontentloaded", timeout=90000)
                    pg.wait_for_timeout(14000)
                    pr = pg.evaluate(PROBE)
                    want = [f"{round(body['scales'][k]['p'] * 100)}%" for k in ("30m", "1h", "4h")]
                    ok = pr["found"] and pr["n"] == 3 and pr["val"] == want and not pr["over"] and not pr["hscroll"] \
                        and (all(pr["brk"]) if case == "brk" else True) and (pr["tags"] == 2 if variant == "B" and w == 1920 else True)
                    print(tag, "OK" if ok else "FAIL", pr)
                    if not ok:
                        fails.append(tag)
                    pg.screenshot(path=str(shot / f"{tag}.png"), full_page=(w == 390))
                    if pr["rect"] and w == 1920:
                        x, y, ww, hh = pr["rect"]
                        pg.screenshot(path=str(shot / f"{tag}_row.png"), clip={"x": max(0, x - 8), "y": y - 120, "width": ww + 16, "height": hh + 128}, full_page=True)
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
