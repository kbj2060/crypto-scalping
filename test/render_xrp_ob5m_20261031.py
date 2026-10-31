#!/usr/bin/env python3
"""XRP 풋프린트 5분 호가 요약(호가 기울기 레인 · 벽 점선 · ▲▼◆) 실화면 확인 (2026-10-31, 사용자 «xrp 에 호가벽·가짜 호가벽이 안 보인다»).

    python3 -u test/render_xrp_ob5m_20261031.py --db <xrpusdt.sqlite> --shot DIR

<db> = 워커(live_ob5m_20261026.py --symbol XRPUSDT)가 서버 실데이터(depthdiff·hot 테이프 사본)로 쓴 sqlite.
/api/ob5m?asset=xrp 를 그 DB 로 만든 응답(server.api_ob5m 과 같은 모양: bars·walls·marks)으로 바꾸고, 나머지는 서버 페이지(터널 127.0.0.1:18787) + 로컬 app.js·styles.css·index.html.
검사(1920·390): XRP 탭 레인 범례가 «호가 기울기 ±50bp»(준비 중 아님) · 응답 벽 ≥1 · JS 오류 0 · GET 아닌 요청 0.
"""
import argparse, json, pathlib, sqlite3, sys

HERE = pathlib.Path(__file__).resolve().parent
DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"


def payload(db):
    """server.api_ob5m produce() 의 확정 봉 경로(진행 중 봉 live 표는 뺌)."""
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


def run(db, shot):
    from playwright.sync_api import sync_playwright
    js, css, html = ((DASH / f).read_text("utf-8") for f in ("app.js", "styles.css", "index.html"))
    body = payload(db)
    print(f"응답: 봉 {len(body['bars'])} · 벽 {len(body['walls'])} · 표식 {len(body['marks'])}")
    fails, blocked, errs = [], [], []
    shot = pathlib.Path(shot); shot.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        br = p.chromium.launch()
        for w, h in ((1920, 1080), (390, 844)):
            tag = f"xrp_{w}"
            pg = br.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
            pg.on("pageerror", lambda e, t=tag: errs.append(f"{t}: {e}"))
            pg.route("**/*", lambda r: (blocked.append(r.request.url), r.abort()) if r.request.method != "GET" else r.fallback())
            pg.route("**/app.js*", lambda r: r.fulfill(body=js, content_type="application/javascript"))
            pg.route("**/styles.css*", lambda r: r.fulfill(body=css, content_type="text/css"))
            pg.route("**/api/ob5m*", lambda r: r.fulfill(body=json.dumps(body if "asset=xrp" in r.request.url else {"bars": [], "walls": [], "marks": []}),
                                                         content_type="application/json"))
            pg.route(URL, lambda r: r.fulfill(body=html, content_type="text/html"))
            pg.goto(URL, wait_until="domcontentloaded", timeout=90000)
            pg.wait_for_timeout(8000)
            pg.evaluate("() => setActiveSnapshotAsset('xrp')")
            pg.wait_for_timeout(14000)
            legend = pg.evaluate("() => [...document.querySelectorAll('svg tspan')].map((t) => t.textContent).filter((s) => s.includes('호가 기울기'))")
            ok = bool(legend) and all("준비 중" not in s for s in legend) and len(body["walls"]) > 0
            print(tag, "OK" if ok else "FAIL", legend)
            if not ok:
                fails.append(tag)
            pg.screenshot(path=str(shot / f"{tag}.png"), full_page=(w == 390))
            pg.close()
        br.close()
    if blocked:
        fails.append(f"GET 아닌 요청 {len(blocked)}건 막힘: {blocked[:3]}")
    if errs:
        fails.append(f"JS 오류 {errs[:3]}")
    print("PASS" if not fails else f"FAIL {fails}")
    return not fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--db", required=True); ap.add_argument("--shot", required=True)
    a = ap.parse_args()
    sys.exit(0 if run(a.db, a.shot) else 1)
