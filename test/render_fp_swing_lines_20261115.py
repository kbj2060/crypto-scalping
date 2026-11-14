"""풋프린트 «1h·30m 스윙» 선 렌더 검증 (2026-11-15).
  공개 대시보드에 이 작업 사본의 app.js 만 끼우고(쓰기 요청 전부 차단), 아직 서버에 없는 /api/swing-lines 를 지금가 기준 가짜 응답으로 채워
  (1) 지표 팝업 칩 «1h 스윙»(기본 켬)·«30m 스윙»(기본 끔) (2) 화면 안 고점 선 + 화면 밖 저점 꼬리표(왼쪽 끝 «↓ 1h 스윙 저 …»)
  (3) 지금가가 고점을 넘었으면 «마감 대기» (4) 창 안 돌파 표식 «1h ↑ 돌파» (5) 30m 켜면 30m 선도
  를 데스크톱·휴대폰에서 확인하고 스크린숏을 남긴다. 실행: python test/render_fp_swing_lines_20261115.py <스크린숏 폴더>"""
import json, re, sys, time, pathlib
from playwright.sync_api import sync_playwright

W = str(pathlib.Path(__file__).resolve().parents[1] / "dashboard/live") + "/"
SP = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
MODE = {"px": None, "wait": False}


def payload():
    px, now = MODE["px"], int(time.time() * 1000)
    hi = px * (0.997 if MODE["wait"] else 1.002)
    kt = now + 864e5 if MODE.get("future") else now - 864e5
    st = lambda f: {"hi": {"price": hi * f, "pivot_t": now - 3 * 864e5, "known_t": kt}, "lo": {"price": px * 0.9 * f, "pivot_t": now - 5 * 864e5, "known_t": kt if MODE.get("future") else now - 3 * 864e5},
                    "last_break": {"dir": 1, "kind": "BOS", "level": px * 0.997, "t": now - 30 * 60e3}, "trend": 1, "asof_t": now}
    return {"ok": True, "1h": st(1.0), "30m": st(1.001)}


def handle(route):
    req = route.request
    if req.method != "GET":
        return route.abort()                                   # 주문 등 쓰기 요청 전부 차단
    path = re.sub(r"\?.*$", "", req.url)
    if "/dashboard/live/" in path and path.endswith("/app.js"):
        return route.fulfill(body=open(W + "app.js", encoding="utf-8").read(), content_type="application/javascript")
    if path.endswith("/api/swing-lines"):
        return route.fulfill(body=json.dumps(payload() if MODE["px"] else {"ok": False}), content_type="application/json")
    return route.continue_()


JS = r"""() => {
  const svgs = [...document.querySelectorAll('svg')];
  const lines = svgs.flatMap((s) => [...s.querySelectorAll('line')]).filter((p) => /큰 스윙/.test(p.querySelector('title')?.textContent || ''))
    .map((l) => ({ dash: l.getAttribute('stroke-dasharray'), title: l.querySelector('title').textContent.slice(0, 40) }));
  const texts = svgs.flatMap((s) => [...s.querySelectorAll('text')]).map((t) => (t.firstChild ? t.firstChild.textContent : '')).filter((x) => /스윙|돌파/.test(x));
  const notes = (window.fpNotesLast || []).map((n) => n.label);
  const chip = (k) => { const c = document.querySelector(`input[data-line="${k}"]`); return c ? c.checked : null; };
  return { lines, texts, chip1h: chip('sw1h'), chip30: chip('sw30m'), swingOk: !!(latestSwing && latestSwing.ok), gut: [...document.querySelectorAll('svg text')].map((t) => t.textContent).filter((x) => /스윙/.test(x)) };
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
        pg.goto("https://thesan.xyz/dashboard/live/", wait_until="domcontentloaded"); pg.wait_for_timeout(9000)
        MODE["px"] = pg.evaluate("Number(latestLivePriceByAsset[activeSnapshotAsset])")
        for case in ("near", "wait", "30m"):
            MODE["wait"] = case == "wait"
            MODE["px"] = pg.evaluate("Number(latestLivePriceByAsset[activeSnapshotAsset])")   # 지금가 기준으로 다시(시험 사이 가격이 움직인다)
            if case == "30m":
                pg.evaluate("() => { mcLine.sw30m = true; }")
            pg.evaluate("() => { latestSwing = null; swingLastFetchAt = 0; }")
            pg.wait_for_timeout(8000)
            r = pg.evaluate(JS)
            print(name, case, json.dumps(r, ensure_ascii=False)[:600])
            check(r["swingOk"], f"{name}/{case}: /api/swing-lines 받음")
            if name == "desktop" and case == "near":
                check(r["chip1h"] is True and r["chip30"] is False, f"{name}: 칩 «1h 스윙» 기본 켬 · «30m 스윙» 기본 끔")
            n1 = [l for l in r["lines"] if l["title"].startswith("1h")]
            check(len(n1) >= 1 and n1[0]["dash"] == "12 4", f"{name}/{case}: 1h 선(파선 12 4)")
            check(any(t.startswith("↓ 1h 스윙 저") for t in r["texts"]), f"{name}/{case}: 화면 밖 저점 꼬리표 «↓ 1h 스윙 저 …»")
            check(any("1h ↑ 돌파" in t for t in r["texts"]), f"{name}/{case}: 창 안 돌파 표식 «1h ↑ 돌파»")
            if case == "wait":
                check(any("마감 대기" in t for t in r["gut"] + r["texts"]), f"{name}/wait: «마감 대기» 표시")
            if case == "30m":
                check(any(l["title"].startswith("30m") for l in r["lines"]), f"{name}/30m: 30m 선도 그림")
            pg.screenshot(path=f"{SP}/fp_swing_{name}_{case}.png", full_page=False)
        # 11-15 감사 수정 확인: 창 끝보다 늦게 알려진 스윙은 그리지 않는다(지나간 화면의 미래 정보)
        MODE["future"] = True
        pg.evaluate("() => { latestSwing = null; swingLastFetchAt = 0; }"); pg.wait_for_timeout(8000)
        r = pg.evaluate(JS)
        check(not any(l["title"].startswith("1h") for l in r["lines"]) and not any(t.startswith("↓ 1h 스윙 저") or t.startswith("1h 스윙") for t in r["texts"] + r["gut"]),
              f"{name}/future: 창 끝 뒤에 알려진 스윙은 선·이름표 없음")
        MODE["future"] = False
        ctx.close()
    b.close()
print("FAILS", fails)
sys.exit(1 if fails else 0)
