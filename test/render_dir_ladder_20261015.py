#!/usr/bin/env python3
"""풋프린트 머리 «지평 사다리»(옵션 칩 자리, 2026-10-15 사용자 시안 C + 추세 칩 합침)를 실제 화면에 그려 확인한다.

    FRED_API_KEY=... python3 -u test/render_dir_ladder_20261015.py --shot DIR

서버 페이지(터널 127.0.0.1:18787) 위에 로컬 app.js·styles.css·index.html 을 덮는다(render_xyz100_tile_20261014.py 와 같은 방식).
/api/dir-signals 는 로컬 dashboard/dir_signals.py 로 지금 계산한 값(live)과 원천 실패(down: 두 신호 ok False)를 준다.
🔴GET 이 아닌 요청(주문 포함)은 브라우저 안에서 전부 막는다 -- 막힌 요청이 있으면 실패.
검사(1920·1700·390): #optChip 없음 · 1920 은 코인 탭과 같은 줄 · 알약 둘 = «지금 추세»(따로) | «7일 예측 · 4~13주 예측» · 일 단위 칸 판정이 응답과 같음
  · ⑤ 예측 판(10-15 시안 B): ⑤ 뉴스 판 없음 · 칸 «지금 추세 · 4h 예측 · 7일 예측 · 4~13주 예측» · 1920 판 줄 ①②④ 셋 · 4h 칸 변동성 줄
    (/api/vol-forecast 는 가짜 응답 -- 모델 아티팩트가 서버에만 있다) · 1920 이상 시장 맥락 칸 안 스크롤 없음
    (10-15 1초 수급 하한 30 → 22% 로 1700 도 스크롤 없음) · 4h 칸 = 방향(가짜 /api/dir4h) + 변동성 · ①②④ 제목 없음(«?» 는 상태 줄) · ⑥ 그래프 40px
  · 머리 줄 밖으로 안 넘침 · 페이지 가로 스크롤 없음 · 휴대폰 칸 높이 ≥40px · JS 오류 0 · down 이면 일 단위 칸은 «-».
"""
import argparse, json, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"

PROBE = """() => {
  const box = document.getElementById('sitChips'), head = box && box.closest('.ops-health-head');
  const segs = box ? [...box.querySelectorAll('.ds-seg')] : [];
  const hr = head ? head.getBoundingClientRect() : null;
  return { opt: !!document.getElementById('optChip'), ladders: box ? box.querySelectorAll('.ds-c').length : 0, nowPill: box ? box.querySelectorAll('.ds-now .ds-seg').length : 0,
           hz: segs.map((s) => (s.querySelector('.ds-hz') || {}).textContent),
           verdict: segs.map((s) => (s.querySelector(':scope > b') || {}).textContent),
           over: hr ? segs.some((s) => s.getBoundingClientRect().right > hr.right + 0.5) : null,
           row1: (() => { const a = document.getElementById('snapshotAssetTabs'); if (!a || !box) return null; const r = a.getBoundingClientRect(), b = box.getBoundingClientRect();
             return b.top < r.bottom && b.bottom > r.top; })(), wpx: box ? Math.round(box.getBoundingClientRect().width) : 0,
           hscroll: document.documentElement.scrollWidth > window.innerWidth + 1,
           fc: (() => { const b = document.getElementById('mcBody'); if (!b) return null; const fc = b.querySelector('.mc-fc');
             const titles = [...b.querySelectorAll('.mc-sec h4')].map((h) => h.innerText.replace(/\\?/g, '').trim());
             return { news: titles.some((t) => /⑤ 뉴스/.test(t)), strip: b.querySelectorAll('.mc-strip > .mc-sec').length,
                      hz: fc ? [...fc.querySelectorAll('.fc-c .hz')].map((h) => h.textContent) : null,
                      vol: fc ? [...fc.querySelectorAll('.fc-c')][1].innerText.split('\\n').slice(3).join(' ') : null,
                      clipped: b.classList.contains('mc-cmp') && b.scrollHeight > b.clientHeight + 2,
                      over: fc ? [...fc.querySelectorAll('.fc-c')].some((c) => c.scrollWidth > c.clientWidth + 1) : null,
                      d4: fc ? ((fc.querySelectorAll('.fc-c')[1] || {}).querySelector('b') || {}).textContent : null,
                      stripH4: b.querySelectorAll('.mc-strip > .mc-sec h4').length, q1: b.querySelectorAll('.mc-strip .mc-q1').length,
                      rateH: [...b.querySelectorAll('.mc-rate .mc-svg')].map((g) => Math.round(g.getBoundingClientRect().height)) }; })(),
           minH: segs.length ? Math.round(Math.min(...segs.map((s) => s.getBoundingClientRect().height))) : 0 };
}"""


D4 = {"ok": True, "decided_at_ms": 0, "pred_bp": -45.2, "lo": -41.4, "hi": 36.2, "signal": -1, "model_month": "2026-10-01",
      "train_window": ["2025-10-01 00:00:00", "2026-09-30 19:55:00"], "nan_feats": [], "hist": []}   # decided_at 는 요청 때 지금으로
VOL = {"ok": True, "asof_ms": 0, "px": 2365.65, "sig4h_pct": 1.324, "sig4h_usd": 31.33, "lvl": 1.492, "x_now": 1.35, "p_expand": 0.55, "ref_days": 30.0}   # 10-15 서버 실측


def run(shot):
    from playwright.sync_api import sync_playwright
    from dashboard.dir_signals import dir_signals_payload
    js, css, html = ((DASH / f).read_text("utf-8") for f in ("app.js", "styles.css", "index.html"))
    live = dir_signals_payload()
    if not (live["cb"].get("ok") and live["z91"].get("ok")):
        print("원천 실패(FRED_API_KEY?):", {k: v.get("reason") for k, v in live.items() if isinstance(v, dict)}); return False
    down = {"ok": True, "k": 1.0, "cb": {"ok": False, "reason": "HTTPError"}, "z91": {"ok": False, "reason": "FRED_API_KEY 없음"}}
    word = lambda s: "▲ 상승" if s > 0 else "▼ 하락" if s < 0 else "의견 없음"
    fails, blocked, errs = [], [], []
    shot = pathlib.Path(shot); shot.mkdir(parents=True, exist_ok=True)

    def api_for(txt):
        def api(r):
            r.fulfill(body=txt, content_type="application/json")
        return api
    with sync_playwright() as p:
        br = p.chromium.launch()
        for case, body in (("live", live), ("down", down)):
            want = [word(live["cb"]["state"]), word(live["z91"]["state"])] if case == "live" else ["-", "-"]
            for w, h in ((1920, 1080), (1700, 950), (390, 844)):
                pg = br.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
                pg.on("pageerror", lambda e, c=case, w=w: errs.append(f"{c}{w}: {e}"))
                pg.route("**/*", lambda r: (blocked.append(r.request.url), r.abort()) if r.request.method != "GET" else r.fallback())
                pg.route("**/app.js*", lambda r: r.fulfill(body=js, content_type="application/javascript"))
                pg.route("**/styles.css*", lambda r: r.fulfill(body=css, content_type="text/css"))
                pg.route(URL, lambda r: r.fulfill(body=html, content_type="text/html"))
                pg.route("**/api/dir-signals*", api_for(json.dumps(body)))
                pg.route("**/api/vol-forecast*", api_for(json.dumps(VOL if case == "live" else {"ok": False, "reason": "artifact_missing"})))
                pg.route("**/api/dir4h*", api_for(json.dumps({**D4, "decided_at_ms": int(time.time() * 1000) - 60_000} if case == "live" else {"ok": False, "reason": "artifact_missing"})))
                pg.goto(URL, wait_until="load", timeout=60000)
                pg.wait_for_timeout(12000)
                got = pg.evaluate(PROBE)
                tag = f"{case}·{w}"
                print(tag, json.dumps(got, ensure_ascii=False), flush=True)
                if got["opt"]: fails.append(f"[{tag}] 옵션 칩이 남아 있음")
                if got["ladders"] != 2 or got["nowPill"] != 1 or got["hz"] != ["지금 추세", "7일 예측", "4~13주 예측"]: fails.append(f"[{tag}] 칸 {got['hz']} 알약 {got['ladders']}")
                elif got["verdict"][1:] != want: fails.append(f"[{tag}] 판정 {got['verdict']} ≠ {want}")
                if got["over"]: fails.append(f"[{tag}] 머리 줄 밖으로 넘침")
                if w >= 1920 and not got["row1"]: fails.append(f"[{tag}] 1920 에서 사다리가 둘째 줄로 내려감 ({got['wpx']}px)")
                fc = got["fc"] or {}
                if fc.get("news"): fails.append(f"[{tag}] ⑤ 뉴스 판이 남아 있음")
                if fc.get("hz") != ["지금 추세", "4h 예측", "7일 예측", "4~13주 예측"]: fails.append(f"[{tag}] 예측 칸 {fc.get('hz')}")
                if w > 500 and fc.get("strip") != 3: fails.append(f"[{tag}] 판 줄 {fc.get('strip')}칸")
                want_vol = "변동 ±1.32% · 평소 1.49배" if case == "live" else "변동 불러오는 중"
                if fc.get("vol") != want_vol: fails.append(f"[{tag}] 변동성 줄 {fc.get('vol')!r}")
                if fc.get("d4") != ("▼ 하락" if case == "live" else "-"): fails.append(f"[{tag}] 4h 판정 {fc.get('d4')!r}")
                if w > 500 and (fc.get("stripH4") or fc.get("q1") != 3): fails.append(f"[{tag}] ①②④ 제목 {fc.get('stripH4')} · ? {fc.get('q1')}")
                if w > 500 and fc.get("rateH") and min(fc["rateH"]) < 40: fails.append(f"[{tag}] ⑥ 그래프 {fc.get('rateH')}")
                if fc.get("clipped") or fc.get("over"): fails.append(f"[{tag}] 예측 판 잘림 {fc}")
                if pg.locator(".mc-fc").count():
                    pg.add_style_tag(content="#ofab { visibility: hidden !important; }")
                    pg.locator(".mc-fc").screenshot(path=str(shot / f"fc_{case}_{w}.png"))
                if got["hscroll"]: fails.append(f"[{tag}] 가로 스크롤")
                if w < 500 and got["minH"] < 40: fails.append(f"[{tag}] 터치 칸 {got['minH']}px < 40")
                pg.locator("#sitChips").locator("xpath=ancestor::div[contains(@class,'ops-health-head')][1]").screenshot(path=str(shot / f"ladder_{case}_{w}.png"))
                pg.close()
        br.close()
    if blocked: fails.append(f"막은 요청 {len(blocked)}: {blocked[:3]}")
    if errs: fails.append(f"JS 오류 {errs[:3]}")
    print("FAIL" if fails else "PASS", fails)
    return not fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default="tmp/dir_ladder")
    raise SystemExit(0 if run(ap.parse_args().shot) else 1)
