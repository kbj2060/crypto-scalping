#!/usr/bin/env python3
"""대시보드 카드를 **실제로 렌더해서** 레이아웃 결함을 잰다 (2026-09-22).

왜 있는가: 2026-09-22 하루에 여섯 개를 잡았는데 **전부 코드를 읽어선 안 보였다**.
  · 전역 `.top { display:flex }` 가 `.sit-col top` 을 잡아 1순위 열만 가로로 누움
  · `.acct-pos` 의 overflow:hidden 이 레일 «위» 라벨을 통째로 잘라냄
  · `.acct-gauge-legend` 를 절대배치로 바꾸며 범위를 안 좁혀 라벨이 전부 0 에 쌓임
  · CSS 특이도 3연속(뒤 규칙이 이김 / 전역 클래스 / background:none 이 그라디언트를 덮음)
CSS 충돌은 소스에 «틀린 곳»이 없다 -- 두 파일이 각자 맞고 조합만 틀리다. 계산된 박스에만
나타나므로 정적으로는 원리적으로 못 찾는다.

쓰는 법:
  python3 -u test/render_dashboard_layout_check_20260922.py --anchor '#snapAcctBalance'
  python3 -u ... --anchor '#situationBody' --widths 1500,1024,390 --shot tmp/shots
  python3 -u ... --anchor '#snapAcctBalance' --inject test/fixtures/acct_position.js
  python3 -u ... --selftest          # 검사기 자신이 결함을 잡는지 증명

🔴`-u` 를 붙인다. 파이썬이 stdout 을 버퍼링해서 안 붙이면 끝날 때까지 아무것도 안 보인다.
🔴로컬 바이트를 서버 페이지 위에 덮어 라우팅하므로 **배포 전에** 실물로 볼 수 있다.
  글롭에 `*` 가 필수다 -- 페이지가 `app.js?v=...` 로 요청해서 쿼리 없는 글롭은 안 걸린다.
"""
import argparse, pathlib, sys

DASH = pathlib.Path(__file__).resolve().parent.parent / "dashboard" / "live"

# 오늘 실제로 버그를 잡은 다섯 가지만 잰다. 늘리기보다 **잡은 것만** 남긴다.
SCAN = r"""(anchor, exact) => {
  const a = document.querySelector(anchor);
  if (!a) return { error: "anchor not found: " + anchor };
  const root = exact ? a : (a.closest("section") || a);
  const all = [...root.querySelectorAll("*")];
  const bad = [], warn = [];
  const name = (e) => (e.textContent.trim().slice(0, 18) || e.className || e.tagName);
  // 일부러 숨긴 입력(칩이 위에 씌워진 range 등)은 «작은 버튼»이 아니다 -- 설계다.
  const hidden = (e) => e.tabIndex < 0 || getComputedStyle(e).opacity === "0"
                        || getComputedStyle(e).visibility === "hidden";
  const leaf = all.filter((e) => e.children.length === 0 && e.textContent.trim());

  // ① 겹침 -- 🔴줄 상자(getClientRects)로 잰다. getBoundingClientRect 는 줄바꿈된 인라인을
  //    여러 줄에 걸친 **하나의 큰 상자**로 주기 때문에 멀쩡한 문단이 전부 «겹침»으로 나온다.
  const lines = [];
  for (const e of leaf) for (const r of e.getClientRects())
    if (r.width > 0.5 && r.height > 0.5) lines.push({ e, t: name(e), r });
  for (let i = 0; i < lines.length; i++) for (let j = i + 1; j < lines.length; j++) {
    const p = lines[i], q = lines[j];
    if (p.e === q.e || p.e.contains(q.e) || q.e.contains(p.e)) continue;
    const A = p.r, B = q.r;
    const ox = Math.min(A.right, B.right) - Math.max(A.left, B.left);
    const oy = Math.min(A.bottom, B.bottom) - Math.max(A.top, B.top);
    if (ox > 1 && oy > Math.min(A.height, B.height) * 0.5) bad.push(["겹침", p.t, q.t]);
  }
  // ② 잘림 -- overflow:hidden 조상 밖으로 나간 **글자**(겹침 검사로는 안 잡히는 다른 축)
  for (const e of leaf) {
    const r = e.getBoundingClientRect();
    if (r.width < 0.5) continue;
    for (let p = e.parentElement; p && root.contains(p); p = p.parentElement) {
      if (getComputedStyle(p).overflow === "visible") continue;
      const b = p.getBoundingClientRect();
      if (r.y < b.y - 0.5 || r.y + r.height > b.y + b.height + 0.5)
        bad.push(["잘림", name(e), p.className || p.tagName]);
      break;
    }
  }
  // ③ 넘침 -- 제 상자를 넘는 **글자**만 본다. 절대배치 눈금 마커가 1~2px 삐져나오는 건
  //    설계이므로, 직접 글자를 든 요소로 한정한다.
  for (const e of all) {
    if (![...e.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim())) continue;
    const cs = getComputedStyle(e);
    if (cs.overflowX === "auto" || cs.overflowX === "scroll") continue;
    if (e.scrollWidth > e.clientWidth + 1) bad.push(["넘침", name(e), e.scrollWidth - e.clientWidth]);
  }
  // ④ 가로 스크롤
  const doc = document.documentElement;
  if (doc.scrollWidth > doc.clientWidth + 1) bad.push(["가로스크롤", doc.scrollWidth - doc.clientWidth]);
  // ⑤ 터치 타깃 -- 경고만. 이 저장소의 칩이 24px 라 실패로 두면 늘 빨갛다(그러면 안 본다).
  for (const e of root.querySelectorAll("button, a[href], input, select")) {
    if (hidden(e)) continue;
    const r = e.getBoundingClientRect();
    if (r.width > 0 && (r.height < 28 || r.width < 30))
      warn.push(["작은버튼", name(e), `${Math.round(r.width)}x${Math.round(r.height)}`]);
  }
  return { bad, warn, h: Math.round(root.getBoundingClientRect().height) };
}"""


def run(pg, anchor, exact):
    return pg.evaluate(SCAN, [anchor, exact])


def check(url, anchor, widths, inject, shot, settle, exact, html=None):
    from playwright.sync_api import sync_playwright
    js, css = (DASH / "app.js").read_text("utf-8"), (DASH / "styles.css").read_text("utf-8")
    page_html = html if html is not None else (DASH / "index.html").read_text("utf-8")
    inj = pathlib.Path(inject).read_text("utf-8") if inject else None
    fails = 0
    with sync_playwright() as p:
        b = p.chromium.launch()
        for w in widths:
            pg = b.new_page(viewport={"width": w, "height": 1500})
            # 🔴`*` 없이는 `app.js?v=...` 에 안 걸린다(2026-09-22 실제로 20분 헛돌았다)
            pg.route("**/app.js*", lambda r: r.fulfill(body=js, content_type="application/javascript"))
            pg.route("**/styles.css*", lambda r: r.fulfill(body=css, content_type="text/css"))
            pg.route(url, lambda r: r.fulfill(body=page_html, content_type="text/html"))
            pg.goto(url, wait_until="networkidle", timeout=60000)
            pg.wait_for_timeout(settle)
            if inj:
                pg.evaluate(inj)
                pg.wait_for_timeout(700)
            r = run(pg, anchor, exact)
            if r.get("error"):
                print(f"[{w:>5}px] ✗ {r['error']}", flush=True); fails += 1; pg.close(); continue
            bad, wn = r["bad"], r.get("warn", [])
            tag = "ok" if not bad else f"✗ {len(bad)}건"
            print(f"[{w:>5}px] {tag} h{r['h']}" + (f"  (경고 {len(wn)})" if wn else ""), flush=True)
            for x in bad[:8]:
                print(f"          {x}", flush=True)
            for x in wn[:3]:
                print(f"     경고 {x}", flush=True)
            fails += len(bad)
            if shot:
                d = pathlib.Path(shot); d.mkdir(parents=True, exist_ok=True)
                h = pg.evaluate_handle("(a) => { const e = document.querySelector(a);"
                                       " return e.closest('section') || e; }", anchor)
                h.as_element().screenshot(path=str(d / f"{anchor.strip('#.')}_{w}.png"))
            pg.close()
        b.close()
    return fails


# 검사기 자신이 맞는지 -- 결함을 **심어 놓고** 다섯 검사가 각각 잡는지 본다.
# 이게 없으면 «0건»이 «깨끗하다»인지 «검사기가 죽었다»인지 구분이 안 된다(09-22 교훈).
SELFTEST = """<!doctype html><meta charset="utf-8"><body style="margin:0">
<section id="probe" style="width:300px">
  <div style="position:relative">
    <span style="position:absolute;left:0;top:0">겹치는A</span>
    <span style="position:absolute;left:10px;top:0">겹치는B</span>
  </div>
  <div style="overflow:hidden;height:20px"><span style="position:relative;top:40px">잘린글자</span></div>
  <div style="width:40px;overflow:hidden;white-space:nowrap">넘치는아주긴글자입니다</div>
  <button style="width:10px;height:10px">x</button>
  <div style="width:900px">가로스크롤</div>
</section></body>"""


def selftest():
    from playwright.sync_api import sync_playwright
    want = {"겹침", "잘림", "넘침", "가로스크롤", "작은버튼"}
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 320, "height": 600})
        pg.set_content(SELFTEST); pg.wait_for_timeout(200)
        r = run(pg, "#probe", True)
        got = {x[0] for x in r["bad"]} | {x[0] for x in r["warn"]}
        b.close()
    miss = want - got
    print(f"자체점검: 잡음 {sorted(got)}", flush=True)
    if miss:
        print(f"🔴 못 잡은 검사: {sorted(miss)} — 검사기가 깨졌다", flush=True); return 1
    print("다섯 검사 전부 동작", flush=True); return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor", help="검사할 영역을 가리키는 CSS 선택자(그 요소의 closest('section'))")
    ap.add_argument("--exact", action="store_true", help="section 으로 안 올라가고 그 요소만")
    ap.add_argument("--url", default="http://127.0.0.1:18787/dashboard/live/")
    ap.add_argument("--widths", default="1500,1024,390")
    ap.add_argument("--inject", help="로드 뒤 평가할 JS 파일(상태 픽스처)")
    ap.add_argument("--shot", help="스크린샷을 저장할 디렉터리")
    ap.add_argument("--settle", type=int, default=6000, help="폴링이 한 바퀴 돌 때까지 기다리는 ms")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.anchor:
        ap.error("--anchor 가 필요합니다 (또는 --selftest)")
    n = check(a.url, a.anchor, [int(x) for x in a.widths.split(",")],
              a.inject, a.shot, a.settle, a.exact)
    print(f"\n합계 {n}건", flush=True)
    sys.exit(1 if n else 0)
