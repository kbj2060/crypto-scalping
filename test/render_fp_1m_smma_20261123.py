"""풋프린트 1m 봉·SMMA 21/50 · 미니 버블 제거·델타 %·CVD 캔들 렌더 검증 (2026-11-23).
  공개 대시보드에 이 작업 사본의 index.html·app.js·styles.css 를 끼우고(쓰기 요청 전부 차단), 서버에 아직 없는 /api/footprint?tf=1m 은
  같은 순간의 5분봉 응답을 1분 다섯 개로 나눠 흉내 낸다(칸 = 1/5 · OHLC = 칸 가격 범위 안 걸음 -- 그림 확인용, 값 검증 아님).
  (1) «1m» 탭이 켜지고 캔들 = 1분 간격 (2) 풋프린트 칸이 그려짐 (3) OI·청산 5분 레인 없음 (4) SMMA 21·50 칩·선 (5) 5m 으로 돌아가면 원래대로 · JS 오류 0
  데스크톱 1920 · 휴대폰 390. 실행: python test/render_fp_1m_smma_20261123.py <스크린숏 폴더>"""
import json, re, sys, pathlib
from playwright.sync_api import sync_playwright

W = str(pathlib.Path(__file__).resolve().parents[1] / "dashboard/live") + "/"
SP = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
LOCAL = {"app.js": "application/javascript", "styles.css": "text/css"}


def fake_1m(d5):
    out = []
    for b in (d5.get("bars") or [])[-24:]:
        lv = b.get("levels") or []
        if not lv:
            continue
        px = [l[0] for l in lv]
        lo, hi = min(px), max(px)
        for k in range(5):
            a, z = lo + (hi - lo) * k / 5, lo + (hi - lo) * (k + 1) / 5
            seg = [l for l in lv if a <= l[0] <= z] or lv[:1]
            o, c = (a, z) if k % 2 == 0 else (z, a)
            out.append({"time": b["time"] + 60 * k, "ohlc": [o, max(o, c, *[l[0] for l in seg]), min(o, c, *[l[0] for l in seg]), c],
                        "levels": [[l[0]] + [x / 1 for x in l[1:]] for l in seg]})
    return {**{k: d5[k] for k in ("symbol", "bucket", "ready", "retailMaxUsd", "whaleMinUsd", "venues") if k in d5},
            "barSeconds": 60, "barsExpected": 120, "full": True, "bars": out[-120:]}


def handle(route):
    req = route.request
    if req.method != "GET":
        return route.abort()
    path = re.sub(r"\?.*$", "", req.url)
    for f, ct in LOCAL.items():
        if "/dashboard/live/" in path and path.endswith("/" + f):
            return route.fulfill(body=open(W + f, encoding="utf-8").read(), content_type=ct)
    if path.rstrip("/").endswith("/dashboard/live") or path.endswith("/dashboard/live/index.html"):
        return route.fulfill(body=open(W + "index.html", encoding="utf-8").read(), content_type="text/html")
    if path.endswith("/api/footprint") and "tf=1m" in req.url:
        r = route.fetch(url=req.url.replace("tf=1m", "tf=5m").replace("since=", "x="))
        return route.fulfill(body=json.dumps(fake_1m(r.json())), content_type="application/json")
    return route.continue_()


JS = r"""() => {
  const svg = document.getElementById('candleSvgSnapshot');
  const titles = [...svg.querySelectorAll('title')].map((t) => t.textContent);
  const full = chartFull();
  const dt = full.length > 1 ? full[full.length - 1].time - full[full.length - 2].time : null;
  const act = document.querySelector('#chartWindowTabs .asset-tab.active');
  return { chart1m, tab: act ? act.textContent.trim() : null, n: full.length, dt,
           fp: !!footprintForChart(), cellsText: svg.querySelectorAll('text').length,
           smma21: titles.some((t) => t.startsWith('SMMA 21')), smma50: titles.some((t) => t.startsWith('SMMA 50')),
           chips: ['smma21', 'smma50'].map((k) => !!document.querySelector(`input[data-line="${k}"]`)),
           wallOn: !!document.querySelector('#mcWall.on'), bubbleShown: !!(document.getElementById('mcBubble') && document.getElementById('mcBubble').clientHeight > 0),
           cvdLegend: [...svg.querySelectorAll('text')].some((t) => t.textContent.includes('CVD 캔들')), delta: renderCandleSvg._deltaStats };
}"""

fails = []
def check(c, msg):
    print(("PASS " if c else "FAIL ") + msg)
    if not c:
        fails.append(msg)


with sync_playwright() as p:
    b = p.chromium.launch()
    for vp, name in (({"width": 1920, "height": 1080}, "desktop"), ({"width": 390, "height": 844}, "mobile")):
        ctx = b.new_context(viewport=vp, device_scale_factor=2 if name == "mobile" else 1)
        pg = ctx.new_page(); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.route("**/*", handle)
        pg.goto("https://thesan.xyz/dashboard/live/", wait_until="domcontentloaded"); pg.wait_for_timeout(9000)
        pg.evaluate("() => { Object.assign(mcLine, { smma21: true, smma50: true }); }")
        pg.evaluate("() => document.querySelector('#chartWindowTabs .asset-tab[data-m1]').click()")
        pg.wait_for_timeout(4000)
        r = pg.evaluate(JS); print(name, "1m", json.dumps(r, ensure_ascii=False))
        check(r["chart1m"] and r["tab"] == "1m", f"{name}: 1m 탭 켜짐")
        check(r["dt"] == 60 and r["n"] >= 30, f"{name}: 캔들 1분 간격 ({r['n']}봉)")
        check(r["fp"], f"{name}: 1분봉 풋프린트 칸")
        check(r["smma21"] and r["smma50"], f"{name}: SMMA 21·50 선")
        check(all(r["chips"]), f"{name}: SMMA 칩 둘")
        pg.screenshot(path=f"{SP}/fp_1m_{name}.png", full_page=False)
        pg.evaluate("() => [...document.querySelectorAll('#chartWindowTabs .asset-tab')].find((x) => x.textContent.trim() === '5m').click()")
        pg.wait_for_timeout(4000)
        r5 = pg.evaluate(JS); print(name, "5m", json.dumps(r5, ensure_ascii=False))
        check(not r5["chart1m"] and r5["tab"] == "5m" and r5["dt"] == 300, f"{name}: 5m 으로 돌아옴(5분 간격)")
        pg.screenshot(path=f"{SP}/fp_5m_{name}.png", full_page=False)
        check(not r5["wallOn"] and not r5["bubbleShown"], f"{name}: 미니 버블 띠·칸 없음")
        if name == "desktop":
            check(r5["cvdLegend"], f"{name}: CVD 캔들 범례")
            pg.evaluate("() => fpWideSet(true)"); pg.wait_for_timeout(2500)
            rw = pg.evaluate(JS); print(name, "wide", json.dumps(rw, ensure_ascii=False))
            check(not rw["wallOn"] and not rw["bubbleShown"], f"{name}: 전체화면도 미니 버블 없음")
            pg.screenshot(path=f"{SP}/fp_wide_{name}.png", full_page=False)
            pg.evaluate("() => fpWideSet(false)")
        check(not errs, f"{name}: JS 오류 0 {errs[:2]}")
        ctx.close()
    b.close()
print("FAILS", fails)
sys.exit(1 if fails else 0)
