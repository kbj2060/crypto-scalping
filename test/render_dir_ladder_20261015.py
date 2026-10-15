#!/usr/bin/env python3
"""풋프린트 머리 «지평 사다리»(옵션 칩 자리, 2026-10-15 사용자 시안 C + 추세 칩 합침)를 실제 화면에 그려 확인한다.

    FRED_API_KEY=... python3 -u test/render_dir_ladder_20261015.py --shot DIR

서버 페이지(터널 127.0.0.1:18787) 위에 로컬 app.js·styles.css·index.html 을 덮는다(render_xyz100_tile_20261014.py 와 같은 방식).
/api/dir-signals 는 로컬 dashboard/dir_signals.py 로 지금 계산한 값(live)과 원천 실패(down: 두 신호 ok False)를 준다.
🔴GET 이 아닌 요청(주문 포함)은 브라우저 안에서 전부 막는다 -- 막힌 요청이 있으면 실패.
검사(1920·1700·390): #optChip 없음 · 1920 은 코인 탭과 같은 줄 · 사다리 한 개에 칸 셋, 지평 «7일 · 1~13주 · 4~13주» 순서 · 일 단위 칸 판정이 응답과 같음
  · 머리 줄 밖으로 안 넘침 · 페이지 가로 스크롤 없음 · 휴대폰 칸 높이 ≥40px · JS 오류 0 · down 이면 일 단위 칸은 «-».
"""
import argparse, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"

PROBE = """() => {
  const box = document.getElementById('sitChips'), head = box && box.closest('.ops-health-head');
  const segs = box ? [...box.querySelectorAll('.ds-seg')] : [];
  const hr = head ? head.getBoundingClientRect() : null;
  return { opt: !!document.getElementById('optChip'), ladders: box ? box.querySelectorAll('.ds-c').length : 0,
           hz: segs.map((s) => (s.querySelector('.ds-hz') || {}).textContent),
           verdict: segs.map((s) => (s.querySelector(':scope > b') || {}).textContent),
           over: hr ? segs.some((s) => s.getBoundingClientRect().right > hr.right + 0.5) : null,
           row1: (() => { const a = document.getElementById('snapshotAssetTabs'); if (!a || !box) return null; const r = a.getBoundingClientRect(), b = box.getBoundingClientRect();
             return b.top < r.bottom && b.bottom > r.top; })(), wpx: box ? Math.round(box.getBoundingClientRect().width) : 0,
           hscroll: document.documentElement.scrollWidth > window.innerWidth + 1,
           minH: segs.length ? Math.round(Math.min(...segs.map((s) => s.getBoundingClientRect().height))) : 0 };
}"""


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
                pg.goto(URL, wait_until="load", timeout=60000)
                pg.wait_for_timeout(12000)
                got = pg.evaluate(PROBE)
                tag = f"{case}·{w}"
                print(tag, json.dumps(got, ensure_ascii=False), flush=True)
                if got["opt"]: fails.append(f"[{tag}] 옵션 칩이 남아 있음")
                if got["ladders"] != 1 or got["hz"] != ["7일", "1~13주", "4~13주"]: fails.append(f"[{tag}] 칸 {got['hz']}")
                elif [got["verdict"][0], got["verdict"][2]] != want: fails.append(f"[{tag}] 판정 {got['verdict']} ≠ {want}")
                if got["over"]: fails.append(f"[{tag}] 머리 줄 밖으로 넘침")
                if w >= 1920 and not got["row1"]: fails.append(f"[{tag}] 1920 에서 사다리가 둘째 줄로 내려감 ({got['wpx']}px)")
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
