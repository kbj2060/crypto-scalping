#!/usr/bin/env python3
"""⑤ 예측 판 시안 ②③④(2026-10-16 사용자 «모두 한꺼번에 시안») -- 제목 줄 «종합»(4h·추세·09시 칩) · 4h 칸 문턱 눈금.

    python3 -u test/render_fc_combo_20261016.py --shot DIR

서버 페이지(터널 127.0.0.1:18787) 위에 로컬 app.js·styles.css·index.html 을 덮는다. /api/dir4h·/api/trend 는 서버 실값.
/api/open-rev 는 로컬 dashboard/dir_signals.open_rev_from_klines(바이낸스 공개 klines)로 지금 값(live) ·
가짜 «진행 중 숏»(active) · 가짜 «4h 롱 문턱 근처 + 진행 중 롱»(near).
🔴GET 이 아닌 요청은 브라우저 안에서 전부 막는다 -- 막힌 요청이 있으면 실패.
검사(1920·1700·390): 종합 칩 셋(4h·추세·09시) · 4h 칸 눈금(지금 점 + 문턱 색 둘) · 종합 판정 = 칩 합의 부호 · active 면 09시 칩 ▼ ·
칸 글자 안 넘침 · 1700·1920 시장 맥락 칸 안 스크롤 없음 · 가로 스크롤 없음 · JS 오류 0.
"""
import argparse, json, pathlib, sys, time, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"

PROBE = """() => {
  const fc = document.querySelector('#mcBody .mc-fc'), sum = fc && fc.querySelector('.fc-sum'), b = document.getElementById('mcBody');
  const chips = sum ? [...sum.querySelectorAll('i')].map((i) => i.textContent) : [];
  const g = fc && fc.querySelector('.d4g');
  const cells = fc ? [...fc.querySelectorAll('.fc-c')] : [];
  return { sum: sum ? sum.querySelector('b').textContent : null, cls: sum ? sum.className : null, chips,
           gauge: g ? { dots: g.querySelectorAll('u').length, zones: g.querySelectorAll('.z').length } : null,
           d4sub: cells[2] ? (cells[2].querySelector('.sub') || {}).textContent : null,
           // 넘침 검사는 이번 시안이 바꾼 4h 칸만(«지금 추세» 칸 설명 줄은 배포판에서도 1700 폭에 줄임표 -- 10-16 별도 보고)
           over: cells[2] ? [...cells[2].querySelectorAll('.sub, b')].some((e) => e.scrollWidth > e.clientWidth + 1) : true,
           sumOver: sum ? sum.getBoundingClientRect().right > fc.getBoundingClientRect().right + 0.5 : null,
           clipped: !!(b && b.classList.contains('mc-cmp') && b.scrollHeight > b.clientHeight + 1),
           hscroll: document.documentElement.scrollWidth > window.innerWidth + 1,
           rect: fc ? (() => { const r = fc.getBoundingClientRect(); return [r.x, r.y + window.scrollY, r.width, r.height]; })() : null };
}"""


def fulfill(body, ctype):
    """🔴핸들러는 인자 1개 -- 기본값 인자를 둔 2인자 lambda 는 playwright 가 (route, request) 로 불러 기본값이 덮인다."""
    return lambda r: r.fulfill(body=body, content_type=ctype)


def kl(iv, n):
    return json.load(urllib.request.urlopen(f"https://fapi.binance.com/fapi/v1/klines?symbol=ETHUSDT&interval={iv}&limit={n}", timeout=20))


def run(shot):
    from playwright.sync_api import sync_playwright
    from dashboard.dir_signals import open_rev_from_klines
    js, css, html = ((DASH / f).read_text("utf-8") for f in ("app.js", "styles.css", "index.html"))
    now = time.time() * 1000
    live = open_rev_from_klines(kl("1d", 93), kl("5m", 600), now)
    assert live["ok"], live
    active = {"ok": True, "pending": False, "date": "2026-10-16", "f30_bp": 41.3, "sigma30_bp": 52.0, "open_pos": -1, "entry_px": 2470.0,
              "entry_ms": now - 3600e3, "exit_due_ms": now + 3 * 3600e3}
    near_rev = {**active, "open_pos": 1, "f30_bp": -41.3}
    d4 = json.load(urllib.request.urlopen("http://127.0.0.1:18787/api/dir4h", timeout=20))
    d4_near = {**d4, "pred_bp": round(d4["hi"] - 4, 1), "signal": 0, "decided_at_ms": now - 60e3,
               "hist": [[now - (48 - i) * 300e3, d4["hi"] * (0.3 + 0.6 * i / 47)] for i in range(48)]}
    cases = (("live", live, None), ("active", active, None), ("near", near_rev, d4_near))
    fails, blocked, errs = [], [], []
    shot = pathlib.Path(shot); shot.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        br = p.chromium.launch()
        for case, orv, d4x in cases:
            for w, h in ((1920, 1080), (1700, 950), (390, 844)):
                tag = f"{case}_{w}"
                pg = br.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
                pg.on("pageerror", lambda e, t=tag: errs.append(f"{t}: {e}"))
                pg.route("**/*", lambda r: (blocked.append(r.request.url), r.abort()) if r.request.method != "GET" else r.fallback())
                pg.route("**/app.js*", fulfill(js, "application/javascript"))
                pg.route("**/styles.css*", fulfill(css, "text/css"))
                pg.route("**/api/open-rev*", fulfill(json.dumps(orv), "application/json"))
                if d4x:
                    pg.route("**/api/dir4h*", fulfill(json.dumps(d4x), "application/json"))
                pg.route(URL, fulfill(html, "text/html"))
                pg.goto(URL, wait_until="domcontentloaded", timeout=90000)
                pg.wait_for_timeout(14000)
                pr = pg.evaluate(PROBE)
                chips_ok = len(pr["chips"]) == 3 and [c.split(" ")[0] for c in pr["chips"]] == ["4h", "추세", "09시"]
                sgn = sum((1 if "▲" in c else -1 if "▼" in c else 0) for c in pr["chips"])
                verdict_ok = pr["sum"] == ("▲ 롱 쪽" if sgn > 0 else "▼ 숏 쪽" if sgn < 0 else "중립")
                ok = (chips_ok and verdict_ok and pr["gauge"] and pr["gauge"]["zones"] == 2 and pr["gauge"]["dots"] >= 1
                      and not pr["over"] and not pr["sumOver"] and not pr["clipped"] and not pr["hscroll"]
                      and (case != "active" or pr["chips"][2] == "09시 ▼") and (case != "near" or "롱까지" in (pr["d4sub"] or "")) and "신호bp" not in (pr["d4sub"] or ""))
                print(tag, "OK" if ok else "FAIL", {k: v for k, v in pr.items() if k != "rect"})
                if not ok:
                    fails.append(tag)
                if pr["rect"]:
                    x, y, ww, hh = pr["rect"]
                    pg.screenshot(path=str(shot / f"{tag}_fc.png"), clip={"x": max(0, x - 8), "y": y - 8, "width": ww + 16, "height": hh + 130}, full_page=True)
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
