#!/usr/bin/env python3
"""Macro Trend 카드 S&P500·VIX «오늘 24h» 칸·수신 칸·상세 차트를 실제 화면에 그려 확인한다(2026-10-28 사용자 «S&P 는 xyz:SP500 실시간 ·
VIX 는 ^VIX 5분봉»).

    python3 -u test/render_macro_today_20261028.py --dump tmp/macro_today_dump_1028.json --shot DIR

서버 페이지(터널 127.0.0.1:18787) 위에 로컬 app.js·styles.css·index.html 을 덮는다(render_xyz100_tile_20261014.py 와 같은 방식).
/api/macro-today 는 배포 전이라 서버 hot DB 에서 뽑은 실데이터(--dump, 같은 모양)를 지금 시각으로 옮겨 준다.
🔴GET 이 아닌 요청(주문 포함)은 브라우저 안에서 전부 막는다 -- 막힌 요청이 있으면 실패.
두 상태 × 1920·390: live = S&P «±N.NN%»·수신 «실시간», VIX «±N.NN%»·수신 «n분 지연»/«휴장»(미 동부 산출 시간 밖) · 상세 차트 제목 ·
none = ok False → 두 칸 «…»·수신 «받는 중»(불 wait). VIX 색은 내리면 초록(값 부호와 반대). JS 오류 0.
"""
import argparse, json, pathlib, re, time

HERE = pathlib.Path(__file__).resolve().parent
DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"

PROBE = """() => {
  const row = (n) => [...document.querySelectorAll('#mtCard .mt-r')].find((r) => (r.querySelector('.mt-nm') || {}).textContent?.replace('*', '') === n);
  const one = (n) => { const r = row(n); if (!r) return null; const td = r.querySelector('.mt-td'), b = td && td.querySelector('b'), rt = r.querySelector('.mt-rt'), dot = rt && rt.querySelector('.mt-dot');
    return { today: b ? b.textContent : (td || {}).textContent, color: b ? b.style.color : null, chart: !!(td && td.querySelector('svg')), feed: rt ? rt.textContent : null,
             dot: dot ? [...dot.classList].filter((c) => c !== 'mt-dot')[0] : null, tip: td ? td.title : null }; };
  return { sp: one('S&P500'), vix: one('VIX'), nq: one('나스닥') };
}"""


def run(dump, shot):
    from playwright.sync_api import sync_playwright
    js, css, html = ((DASH / f).read_text("utf-8") for f in ("app.js", "styles.css", "index.html"))
    d = json.loads(pathlib.Path(dump).read_text())
    fails, blocked, errs = [], [], []

    def api_for(case):
        def api(route):
            if case == "none":
                return route.fulfill(body=json.dumps({"ok": False, "symbols": {}, "now_sec": int(time.time())}), content_type="application/json")
            now = int(time.time()); sh = now - d["now_sec"]
            sym = {}
            for k, v in d["symbols"].items():
                bars = [[t + sh, c] for t, c in v["bars"] if t + sh <= now]
                lv = v["live"] and {**v["live"], "recv_sec": v["live"]["recv_sec"] + sh}
                sym[k] = {"bars": bars, "live": lv}
            route.fulfill(body=json.dumps({"ok": True, "symbols": sym, "now_sec": now}), content_type="application/json")
        return api

    shot = pathlib.Path(shot); shot.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        br = p.chromium.launch()
        for case in ("live", "none"):
            for w, h in ((1920, 1080), (390, 844)):
                pg = br.new_page(viewport={"width": w, "height": h}, device_scale_factor=2 if w < 500 else 1)
                pg.on("pageerror", lambda e, c=case, w=w: errs.append(f"{c}{w}: {e}"))
                pg.route("**/*", lambda r: (blocked.append(r.request.url), r.abort()) if r.request.method != "GET" else r.fallback())
                pg.route("**/app.js*", lambda r: r.fulfill(body=js, content_type="application/javascript"))
                pg.route("**/styles.css*", lambda r: r.fulfill(body=css, content_type="text/css"))
                pg.route(URL, lambda r: r.fulfill(body=html, content_type="text/html"))
                pg.route("**/api/macro-today*", api_for(case))
                pg.goto(URL, wait_until="load", timeout=60000)
                pg.wait_for_function("() => document.querySelectorAll('#mtCard .mt-r').length > 10", timeout=90000)
                pg.wait_for_timeout(8000)
                got = pg.evaluate(PROBE)
                tag = f"{case}·{w}"
                print(tag, json.dumps(got, ensure_ascii=False), flush=True)
                sp, vx = got["sp"] or {}, got["vix"] or {}
                pct = re.compile(r"[+−]\d+\.\d\d%")
                if case == "live":
                    if not pct.fullmatch(sp.get("today") or "") or not sp.get("chart"): fails.append(f"[{tag}] S&P 오늘 {sp}")
                    if not pct.fullmatch(vx.get("today") or "") or not vx.get("chart"): fails.append(f"[{tag}] VIX 오늘 {vx}")
                    if w > 500:   # 수신 칸은 넓은 화면만(hx)
                        if sp.get("feed") != "실시간" or sp.get("dot") != "ok": fails.append(f"[{tag}] S&P 수신 {sp.get('feed')} {sp.get('dot')}")
                        if not re.fullmatch(r"\d+분 지연|휴장", vx.get("feed") or ""): fails.append(f"[{tag}] VIX 수신 {vx.get('feed')}")
                    v = float((vx.get("today") or "0%").replace("−", "-").rstrip("%"))
                    want = "var(--muted)" if abs(v) < 0.02 else ("var(--bad)" if v > 0 else "var(--good)")
                    if vx.get("color") != want: fails.append(f"[{tag}] VIX 색 {vx.get('color')} ≠ {want}(내리면 초록)")
                    pg.locator("#mtCard .mt-nm", has_text="S&P500").first.click(); pg.wait_for_timeout(500)
                    cap_sp = pg.locator("#mtCard .mt-ch figcaption").first.inner_text()
                    pg.locator("#mtCard .mt-nm", has_text="VIX").first.click(); pg.wait_for_timeout(500)
                    cap_vx = pg.locator("#mtCard .mt-ch figcaption").first.inner_text()
                    print(tag, "상세", json.dumps([cap_sp, cap_vx], ensure_ascii=False), flush=True)
                    if "xyz:SP500" not in cap_sp: fails.append(f"[{tag}] S&P 상세 {cap_sp!r}")
                    if "^VIX" not in cap_vx or "15분 지연" not in cap_vx: fails.append(f"[{tag}] VIX 상세 {cap_vx!r}")
                else:
                    if sp.get("today") != "…" or vx.get("today") != "…": fails.append(f"[{tag}] 원천 없음인데 값 {sp.get('today')} {vx.get('today')}")
                    if w > 500 and (sp.get("feed") != "받는 중" or sp.get("dot") == "ok"): fails.append(f"[{tag}] S&P 수신 {sp.get('feed')} {sp.get('dot')}")
                pg.locator("#mtCard").screenshot(path=str(shot / f"mt_{case}_{w}.png"))
                pg.close()
        br.close()
    if blocked: fails.append(f"막은 요청 {len(blocked)}: {blocked[:3]}")
    if errs: fails.append(f"JS 오류 {errs[:3]}")
    print("FAIL" if fails else "PASS", fails)
    return not fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--shot", default="tmp/macro_today_render")
    a = ap.parse_args()
    raise SystemExit(0 if run(a.dump, a.shot) else 1)
