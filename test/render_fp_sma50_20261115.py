"""풋프린트 «일봉 SMA50» 선 렌더 검증 (2026-11-15).
  공개 대시보드에 이 작업 사본의 app.js 만 끼우고(쓰기 요청 전부 차단), /api/trend 응답에 sma50 을 붙여(서버 배포 전이라 없음)
  (1) 지표 팝업에 «일봉 SMA50» 칩 (2) 화면 안 값 → 계단선(<title> 일봉 SMA50)·측면 색 (3) 화면 밖 값 → 왼쪽 끝 «↑/↓ 일봉 SMA50 … · 측면» 꼬리표
  를 데스크톱·휴대폰에서 확인하고 스크린숏을 남긴다. 실행: python test/render_fp_sma50_20261115.py <스크린숏 폴더>"""
import json, re, sys, time, pathlib
from playwright.sync_api import sync_playwright

W = str(pathlib.Path(__file__).resolve().parents[1] / "dashboard/live") + "/"
SP = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
MODE = {"v": None}   # 붙일 sma50 값(지금가 기준) -- None 이면 원본 그대로


def handle(route):
    req = route.request
    if req.method != "GET":
        return route.abort()                                   # 주문 등 쓰기 요청 전부 차단
    path = re.sub(r"\?.*$", "", req.url)
    if "/dashboard/live/" in path and path.endswith("/app.js"):
        return route.fulfill(body=open(W + "app.js", encoding="utf-8").read(), content_type="application/javascript")
    if path.endswith("/api/trend"):
        r = route.fetch(); d = r.json()
        if d.get("ok") and MODE["v"]:
            now = int(time.time() * 1000) // 86_400_000 * 86_400_000
            d["sma50"] = [{"t": now - 86_400_000, "v": MODE["v"] * 0.998}, {"t": now, "v": MODE["v"]}]
        return route.fulfill(response=r, body=json.dumps(d), content_type="application/json")
    return route.continue_()


JS = r"""() => {
  const svgs = [...document.querySelectorAll('svg')];
  const line = svgs.flatMap((s) => [...s.querySelectorAll('polyline')]).find((p) => (p.querySelector('title')?.textContent || '').startsWith('일봉 SMA50'));
  const tag = svgs.flatMap((s) => [...s.querySelectorAll('text')]).find((t) => t.textContent.includes('일봉 SMA50'));
  const chip = document.querySelector('input[data-line="sma50"]');
  return { line: line ? { stroke: line.getAttribute('stroke'), dash: line.getAttribute('stroke-dasharray'), n: line.getAttribute('points').split(' ').length } : null,
           tag: tag ? { text: tag.firstChild ? tag.firstChild.textContent : '', fill: tag.getAttribute('fill') } : null,
           chip: chip ? { checked: chip.checked, label: chip.parentElement.textContent } : null,
           price: Number(latestLivePriceByAsset[activeSnapshotAsset]), s50: latestTrend && latestTrend.sma50 };
}"""

fails = []
def check(c, msg):
    print(("PASS " if c else "FAIL ") + msg)
    if not c:
        fails.append(msg)


with sync_playwright() as p:
    b = p.chromium.launch()
    for vp, name in (({"width": 1720, "height": 1080}, "desktop"), ({"width": 412, "height": 915}, "mobile")):
        ctx = b.new_context(viewport=vp, device_scale_factor=2 if name == "mobile" else 1)
        pg = ctx.new_page(); pg.route("**/*", handle)
        MODE["v"] = None
        pg.goto("https://thesan.xyz/dashboard/live/", wait_until="domcontentloaded"); pg.wait_for_timeout(9000)
        price = pg.evaluate("Number(latestLivePriceByAsset[activeSnapshotAsset])")
        for mode, v in (("near", price * 0.9985), ("far", price * 1.15)):
            MODE["v"] = v
            pg.evaluate("() => { latestTrend = null; trendLastFetchAt = 0; }")   # 다음 주기에 /api/trend 다시 받게
            pg.wait_for_timeout(9000)
            r = pg.evaluate(JS)
            print(name, mode, json.dumps({k: r[k] for k in ("line", "tag", "chip", "price")}, ensure_ascii=False))
            check(r["s50"] is not None, f"{name}/{mode}: 응답에 sma50 실림")
            if name == "desktop":
                check(r["chip"] is not None and "일봉 SMA50" in r["chip"]["label"] and r["chip"]["checked"], f"{name}: 지표 칩 «일봉 SMA50» 기본 켬")
            check(r["line"] is not None and r["line"]["dash"] == "10 4", f"{name}/{mode}: 계단선(파선 10 4)")
            want = "var(--bad)" if mode == "far" else "var(--good)"            # far = SMA 가 위 → 지금가 아래 = 숏 추세(빨강)
            check(r["line"] is not None and r["line"]["stroke"] == want, f"{name}/{mode}: 측면 색 {want}")
            if mode == "far":
                check(r["tag"] is not None and r["tag"]["text"].startswith("↑ 일봉 SMA50") and "숏 추세" in r["tag"]["text"], f"{name}/far: 화면 밖 꼬리표 «↑ … 숏 추세»")
            pg.screenshot(path=f"{SP}/fp_sma50_{name}_{mode}.png", full_page=False)
        ctx.close()
    b.close()
print("FAILS", fails)
sys.exit(1 if fails else 0)
