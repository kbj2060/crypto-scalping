#!/usr/bin/env python3
"""XRP 대시보드 이식(2026-10-31, 감사표 docs/xrp_dashboard_gap_audit_20261031.md) 실화면 확인 -- 세 검사가 한 하네스를 쓴다.

    python3 -u test/render_xrp_20261031.py fan options [--shot DIR]
    python3 -u test/render_xrp_20261031.py ob5m --db <xrpusdt.sqlite> [--shot DIR]

서버 페이지(터널 127.0.0.1:18787) 위에 로컬 app.js·styles.css·index.html 을 덮는다. 🔴GET 아닌 요청(주문 제출 등)은 전부 막고, 막힌 요청이 있으면 실패.
- fan (A-1 #2·B-3·B-4, 1920·390 × XRP·ETH): XRP = 부채꼴 글자 «4h 1σ ±x%» 1개 · «옵션 IV» 없음 · σ = /api/vol-forecast?asset=xrp 의 sig4h_pct ·
  «지금 추세» 툴팁에 XRP 실측 · «7일 예측» 툴팁 XRP 60% / ETH = 7일 툴팁 ETH 65% · 추세 툴팁에 XRP 없음 · 가로 스크롤 없음.
- options (A-1 #4·#6·A-2, 1500·390 × ETH·XRP): /api/opt-smile 은 로컬 front_smile(Deribit 공개 요약)로 채운다 · 등고선 6줄 ·
  실현 감마 «12시간 자기상관»(XRP 는 rg_xrp 설명 키) · ⑤ 다음 24시간 보이고 «옵션 만기» 이름표.
- ob5m (A-2 #14, 1920·390 × XRP): /api/ob5m?asset=xrp 를 <db>(워커 --symbol XRPUSDT 가 서버 실데이터 사본으로 쓴 sqlite)로 만든 응답으로 바꾼다 ·
  레인 범례 «호가 기울기 ±50bp»(준비 중 아님) · 응답 벽 ≥1. <db> 의 마지막 봉이 차트 창(~2시간)보다 오래되면 «준비 중»으로 실패한다 -- 새 사본을 쓴다.
모든 검사: JS 오류 0.
"""
import argparse, json, pathlib, sqlite3, sys, time, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dashboard.server import DERIBIT_BOOK_URL, DERIBIT_SPECS, front_smile  # noqa: E402

DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"
SRC = {f: (DASH / f).read_text("utf-8") for f in ("app.js", "styles.css", "index.html")}
EMPTY_OB5M = {"bars": [], "walls": [], "marks": []}


def open_page(br, w, h, coin, errs, blocked, routes=()):
    """로컬 화면 파일을 덮은 서버 페이지를 열고 coin 으로 바꿔 둔다. routes = [(패턴, 처리기)] -- 뒤에 등록한 것이 먼저 받는다."""
    pg = br.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
    pg.on("pageerror", lambda e: errs.append(f"{coin}·{w}: {e}"))
    pg.route("**/*", lambda r: (blocked.append(r.request.url), r.abort()) if r.request.method != "GET" else r.fallback())
    pg.route("**/app.js*", lambda r: r.fulfill(body=SRC["app.js"], content_type="application/javascript"))
    pg.route("**/styles.css*", lambda r: r.fulfill(body=SRC["styles.css"], content_type="text/css"))
    pg.route(URL, lambda r: r.fulfill(body=SRC["index.html"], content_type="text/html"))
    for pat, fn in routes:
        pg.route(pat, fn)
    pg.goto(URL, wait_until="domcontentloaded", timeout=90000)
    pg.wait_for_timeout(8000)
    if coin != "eth":
        pg.evaluate(f"() => setActiveSnapshotAsset('{coin}')")
        pg.wait_for_timeout(14000)
    return pg


def json_route(body_of):
    return lambda r: r.fulfill(body=json.dumps(body_of(r.request.url)), content_type="application/json")


FAN_PROBE = """() => {
  const fan = [...document.querySelectorAll('svg text')].map((t) => t.textContent).filter((s) => /^4h 1σ ±/.test(s));
  const tip = (hz) => { const c = [...document.querySelectorAll('.fc-c')].find((e) => (e.querySelector('.hz') || {}).textContent === hz); return c ? c.title : null; };
  return { fan, trend: tip('지금 추세'), cb: tip('7일 예측'), hscroll: document.documentElement.scrollWidth > window.innerWidth + 1 };
}"""


def check_fan(br, shot, errs, blocked, _db):
    fails = []
    for coin in ("xrp", "eth"):
        for w, h in ((1920, 1080), (390, 844)):
            pg = open_page(br, w, h, coin, errs, blocked)
            pg.evaluate("() => { chartPanRight = chartPanRightMax(); }")
            pg.wait_for_timeout(5000)
            pr = pg.evaluate(FAN_PROBE)
            if coin == "xrp":   # σ 는 서버가 5분마다 바꾼다 -- 화면 확인 직후에 읽는다
                sig = json.load(urllib.request.urlopen("http://127.0.0.1:18787/api/vol-forecast?asset=xrp", timeout=30))["sig4h_pct"]
                ok = (len(pr["fan"]) == 1 and "옵션 IV" not in pr["fan"][0] and f"±{sig:.2f}%" in pr["fan"][0]
                      and "XRP 실측" in (pr["trend"] or "") and "XRP 적중 60%" in (pr["cb"] or ""))
            else:
                ok = "ETH 적중 65%" in (pr["cb"] or "") and "XRP" not in (pr["trend"] or "")
            if not ok or pr["hscroll"]:
                fails.append(f"[fan {coin}·{w}] {pr}")
            if shot:
                pg.screenshot(path=str(shot / f"fan_{coin}_{w}.png"), full_page=(w == 390))
            pg.close()
    return fails


OPT_PROBE = """() => {
  const svg = document.querySelector('#optCard svg.opt-ladder');
  const lines = svg ? [...svg.querySelectorAll('line')].filter((l) => l.getAttribute('stroke') === 'var(--option)').length : -1;
  const rg = document.querySelector('#optCard .opt-sec:has(.opt-q[data-tip^="rg"])');
  const w = document.getElementById('mcWhen');
  return { lines, rgKey: rg ? rg.querySelector('.opt-q').dataset.tip : null, rgText: rg ? rg.textContent.replace(/\\s+/g, ' ') : null,
           whenShown: !!(w && w.offsetHeight > 0), whenText: w ? w.textContent.replace(/\\s+/g, ' ').slice(0, 300) : null };
}"""


def smile(coin):
    lst, pre = DERIBIT_SPECS[coin.upper()][:2]
    j = json.load(urllib.request.urlopen(f"{DERIBIT_BOOK_URL}?currency={lst}&kind=option", timeout=20))
    return {**front_smile(j["result"], time.time() * 1000, pre), "fetched_ts": time.time()}


def check_options(br, shot, errs, blocked, _db):
    sm = {c: smile(c) for c in ("eth", "xrp")}
    assert all(v.get("ok") for v in sm.values()), sm
    route = ("**/api/opt-smile*", json_route(lambda u: sm["xrp" if "asset=xrp" in u else "eth"]))
    fails = []
    for w, h in ((1500, 900), (390, 844)):
        for coin in ("eth", "xrp"):
            pg = open_page(br, w, h, coin, errs, blocked, [route])
            pg.evaluate("() => { gexLastFetchAt = 0; return refreshGex(); }")
            pg.wait_for_timeout(2500)
            pg.evaluate("() => { optLadderScope = 'front'; renderOptions(); renderMarketCtx(); }")
            pg.wait_for_timeout(500)
            got, tag = pg.evaluate(OPT_PROBE), f"options {coin}·{w}"
            print(tag, json.dumps(got, ensure_ascii=False), flush=True)
            if got["lines"] != 6:
                fails.append(f"[{tag}] 등고선 {got['lines']}줄 (6 기대)")
            if got["rgKey"] != ("rg_xrp" if coin == "xrp" else "rg") or "12시간 자기상관" not in (got["rgText"] or ""):
                fails.append(f"[{tag}] 실현 감마 {got['rgKey']} {got['rgText']!r}")
            if not got["whenShown"] or "옵션 만기" not in (got["whenText"] or ""):
                fails.append(f"[{tag}] ⑤ 다음 24시간 {got['whenShown']} {got['whenText']!r}")
            if shot:
                pg.evaluate("() => { const f = document.getElementById('ofab'); if (f) f.style.visibility = 'hidden'; }")
                for nm, sel in (("ladder_rg", "#optCard"), ("when", "#mcWhen")):
                    box = pg.evaluate(f"""() => {{ const n = document.querySelector('{sel}'); if (!n || !n.offsetHeight) return null;
                      n.scrollIntoView(); const r = n.getBoundingClientRect(); return {{x: r.x, y: r.y, width: r.width, height: Math.min(r.height, innerHeight - r.y)}}; }}""")
                    if box:
                        pg.screenshot(path=str(shot / f"{nm}_{coin}_{w}.png"), clip=box)
            pg.close()
    return fails


def ob5m_payload(db):
    """server.api_ob5m produce() 의 확정 봉 경로(진행 중 봉 live 표는 뺌).
    ponytail: produce() 가 make_app() 안 닫힘이라 import 못 해 벽 잇기를 옮겨 적었다 -- 모듈 함수로 빼면 이걸 지운다."""
    con = sqlite3.connect(db)
    rows = con.execute("SELECT t, warm, imb50, eat_n_a, wall_px_b, wall_usd_b, wall_px_a, wall_usd_a FROM bars ORDER BY t").fetchall()
    bars, walls, run = [], [], {}
    for t, warm, imb, ev, pb, ub, pa, ua in rows:
        bars.append({"t": t, "tilt_pct": None if imb is None else round(imb * 100, 2), "book_ok": not warm and imb is not None, "events_known": ev is not None})
        for side, px, usd in (("bid", pb, ub), ("ask", pa, ua)):
            r = run.get(side)
            if r and px is not None and r["px"] == px and r["t_end"] == t:
                r["t_end"], r["usd_max"] = t + 300_000, max(r["usd_max"], usd or 0.0)
                continue
            if r:
                walls.append(run.pop(side))
            if px is not None:
                run[side] = {"side": side, "px": px, "t_start": t, "t_end": t + 300_000, "usd_max": usd or 0.0}
    walls += run.values()
    marks = [{"t": t, "bar_t": bt, "side": s, "kind": k, "px": px, "usd": round(u or 0), "dist_bp": round(d, 2)}
             for t, bt, s, k, px, u, d in con.execute("SELECT t, bar_t, side, kind, px, usd, dist_bp FROM marks ORDER BY t")]
    return {"bars": bars, "walls": walls, "marks": marks}


def check_ob5m(br, shot, errs, blocked, db):
    assert db, "ob5m 검사는 --db 가 필요하다"
    body = ob5m_payload(db)
    print(f"응답: 봉 {len(body['bars'])} · 벽 {len(body['walls'])} · 표식 {len(body['marks'])}")
    route = ("**/api/ob5m*", json_route(lambda u: body if "asset=xrp" in u else EMPTY_OB5M))
    fails = []
    for w, h in ((1920, 1080), (390, 844)):
        pg = open_page(br, w, h, "xrp", errs, blocked, [route])
        legend = pg.evaluate("() => [...document.querySelectorAll('svg tspan')].map((t) => t.textContent).filter((s) => s.includes('호가 기울기'))")
        print(f"ob5m xrp·{w}", legend)
        if not legend or any("준비 중" in s for s in legend) or not body["walls"]:
            fails.append(f"[ob5m xrp·{w}] {legend}")
        if shot:
            pg.screenshot(path=str(shot / f"ob5m_xrp_{w}.png"), full_page=(w == 390))
        pg.close()
    return fails


CHECKS = {"fan": check_fan, "options": check_options, "ob5m": check_ob5m}


def run(names, shot, db):
    from playwright.sync_api import sync_playwright
    if shot:
        shot = pathlib.Path(shot); shot.mkdir(parents=True, exist_ok=True)
    fails, errs, blocked = [], [], []
    with sync_playwright() as p:
        br = p.chromium.launch()
        for n in names:
            fails += CHECKS[n](br, shot, errs, blocked, db)
        br.close()
    fails += [f"JS 오류: {e}" for e in errs] + [f"GET 아닌 요청 막힘: {u}" for u in blocked]
    print("FAIL" if fails else "PASS", *fails, sep="\n")
    return not fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checks", nargs="+", choices=sorted(CHECKS))
    ap.add_argument("--db"); ap.add_argument("--shot")
    a = ap.parse_args()
    sys.exit(0 if run(a.checks, a.shot, a.db) else 1)
