#!/usr/bin/env python3
"""일봉 풋프린트(2026-10-08)를 실제로 그려서 확인한다.

    python3 -u test/render_fp_daily_20261008.py [--shot DIR]

서버 페이지(터널 127.0.0.1:18787) 위에 로컬 index.html·app.js·styles.css·fp_daily.js 를 덮는다(render_opt_touch_contours 와 같은 방식).
/api/footprint-daily* 는 배포 전이라 로컬 dashboard.footprint_daily 가 소급본(data/footprint_daily) + lake 백업으로 답한다.
🔴주문 제출(`/api/manual-*/submit|execute|cancel`)은 브라우저 안에서 막는다 -- 막힌 요청이 있으면 실패.
검사(1920·390): 켜면 교체 영역(window.fpRegion, 풋프린트~모의 판·지지/저항) 위에 겹치고(오차 ≤1px) 모의 판·지지/저항은 가려지며 시장 맥락은 남는다 · 기본 30일 · 넓은 화면 = 가격 칸 요청 · 휠 확대·끌기 이동이 그림을 바꾼다 ·
«전체» = 2020년부터 · 레인 호버 툴팁 · 끄면 5분 차트 복귀 · JS 오류 0.
같은 하네스로 «풋프린트/청산맵» 토글(liq_profile.js)도 본다 -- /api/liquidation-map 은 서버 응답에 tier_profile 을 로컬 계산으로 덧붙인다
(공개 아카이브 1시간봉 최근 24개, scripts.live_liquidation_map_20260824.compute_tier_profile). 토글 → 교체 영역 위·범례·툴팁 · 1·7·30일 버튼 · 휠 확대·끌기·더블클릭 복원이 그림을 바꿈 · 되돌리기 · 일봉이 이기기.
"""
import argparse, hashlib, json, pathlib, sys, time
from urllib.parse import parse_qs, urlparse

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dashboard import footprint_daily as fpd  # noqa: E402
from scripts import data_store  # noqa: E402
from scripts.live_liquidation_map_20260824 import compute_tier_profile  # noqa: E402

DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"
# 교체 영역(window.fpRegion = svg 왼쪽 위 (0,0)~(w,h)) 위에 겹치는가 + 모의 판·지지/저항이 가려졌나(넓은 화면 = #mcWall 덮음 ·
#   좁은 화면 = #liquidationMapList·#mcBody > .ppm 숨김) + 남아야 할 칸(시장 맥락 #mcBody)은 보이는가
OVERLAY = """(id) => { const o = document.getElementById(id), g = window.fpRegion, svg = document.getElementById('candleSvgSnapshot');
              if (!o || !g) return null; const r = svg.getBoundingClientRect(), q = o.getBoundingClientRect();
              const vis = (s) => { const e = document.querySelector(s); if (!e) return false; const c = getComputedStyle(e);
                                   return c.display !== 'none' && c.visibility !== 'hidden' && e.getBoundingClientRect().height > 0; };
              return { dx: q.left - r.left, dy: q.top - r.top, dw: q.width - g.w, dh: q.height - g.h, split: g.split,
                       mc: vis('#mcBody'), wall: vis('#mcWall'), sr: vis('#liquidationMapList'), ppm: vis('#mcBody > .ppm') }; }"""


def covered(ov) -> bool:
    """모의 판·지지/저항이 안 보이고 시장 맥락은 보인다."""
    return bool(ov) and ov["mc"] and not ov["wall"] and not (not ov["split"] and (ov["sr"] or ov["ppm"]))
TABS = """() => { const h = document.getElementById('chartWindowTabs');
              return { i: h.style.getPropertyValue('--i'), n: h.style.getPropertyValue('--seg-n'),
                       on: [...h.querySelectorAll('.asset-tab')].filter(b => b.classList.contains('active')).map(b => b.textContent.trim()) }; }"""
TIP = "() => { const t = document.getElementById('chartTooltip'); return t && t.classList.contains('visible') ? t.textContent : null; }"


def provider():
    def lake(stream, start, end, cols):
        try:
            return data_store.read("binance", stream, fpd.COIN, start and start.isoformat(), end and end.isoformat(), columns=cols)
        except Exception:  # noqa: BLE001
            return None
    # 소급본은 메인 체크아웃 data/ 에 생긴다(서버는 같은 저장소 루트라 기본 경로 그대로)
    f = fpd.DailyFootprint(lake, lambda s, p: [], lambda s, p: [], hist_dir=pathlib.Path("/home/kbj20/crypto-scalping/data/footprint_daily"))
    return f, f.days_payload(time.time())


def tier_profiles():
    """공개 아카이브(data.binance.vision, REST 아님) ETHUSDT 1시간봉 → {1·7·30일: tier_profile} (서버 /api/liquidation-map/tiers 와 같은 창)."""
    import io, urllib.request, zipfile
    import pandas as pd
    rows = []
    for back in range(1, 36):
        d = (pd.Timestamp.utcnow() - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
        try:
            raw = urllib.request.urlopen(f"https://data.binance.vision/data/futures/um/daily/klines/ETHUSDT/1h/ETHUSDT-1h-{d}.zip", timeout=30).read()
        except Exception:  # noqa: BLE001 -- 아직 안 올라온 날
            continue
        txt = zipfile.ZipFile(io.BytesIO(raw)).read(f"ETHUSDT-1h-{d}.csv").decode()
        rows += [r.split(",") for r in txt.splitlines() if r[:1].isdigit()]
        if len(rows) >= 30 * 24:
            break
    df = pd.DataFrame({"timestamp": pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True),
                       "high": [float(r[2]) for r in rows], "low": [float(r[3]) for r in rows],
                       "close": [float(r[4]) for r in rows], "volume": [float(r[5]) for r in rows]}).sort_values("timestamp")
    oi = data_store.read("binance", "oi_1s", "ETH", (pd.Timestamp.utcnow() - pd.Timedelta(days=3)).strftime("%Y-%m-%d"), None,
                         columns="ts_ms, open_interest").sort_values("ts_ms")
    cp = float(df.close.iloc[-1])
    return {d: compute_tier_profile(df.tail(d * 24).reset_index(drop=True), cp, float(oi.open_interest.iloc[-1]) * cp) for d in (1, 7, 30)}   # 서버처럼 OI 달러


def run(shot):
    from playwright.sync_api import sync_playwright
    files = {n: (DASH / n).read_text("utf-8") for n in ("app.js", "styles.css", "index.html", "fp_daily.js", "liq_profile.js")}
    f, days = provider()
    print("일 수", len(days["cols"]["d"]), days["cols"]["d"][0], "~", days["cols"]["d"][-1], "소급 끝", days["hist_until"], flush=True)
    fails, blocked, errs, cells_hits = [], [], [], []
    tps = tier_profiles(); tp = tps[1]
    print("청산맵 tier_profile", tp and (tp["current_price"], [t["name"] for t in tp["tiers"]], {d: len(t["tiers"][0]["values"]) for d, t in tps.items()}), flush=True)
    tier_hits, heat_hits = [], []

    def tiers(route):
        d = int(parse_qs(urlparse(route.request.url).query)["days"][0]); tier_hits.append(d)
        route.fulfill(body=json.dumps({"days": d, "tier_profile": tps[d]}), content_type="application/json")

    def liqmap(route):
        r = route.fetch()
        body = r.json()
        u = route.request.url.lower()
        if isinstance(body, dict) and ("asset=" not in u or "asset=eth" in u):
            body["tier_profile"] = tp
        route.fulfill(response=r, body=json.dumps(body), content_type="application/json")

    def api(route):
        u = urlparse(route.request.url)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path.endswith("/heat"):
            body = fpd.heat_payload(time.time(), q["from"], q["to"], fpd.parse_row(q.get("row")), None,
                                    hist_dir=pathlib.Path("/home/kbj20/crypto-scalping/data/footprint_daily"))
            heat_hits.append(q["from"])
        elif u.path.endswith("/cells"):
            body = f.cells_payload(time.time(), q["from"], q["to"], fpd.parse_row(q.get("row")))
            cells_hits.append((q["from"], q["to"], q.get("row"), len(body["cells"])))
        else:
            body = days
            if q.get("since"):
                keep = [i for i, d in enumerate(days["cols"]["d"]) if d >= q["since"]]
                body = {**days, "cols": {k: [v[i] for i in keep] for k, v in days["cols"].items()}}
        route.fulfill(body=json.dumps(body), content_type="application/json")

    with sync_playwright() as p:
        b = p.chromium.launch()
        for w, h in ((1920, 1080), (390, 844)):
            pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2 if w < 500 else 1)
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.route(lambda u: "/api/manual-" in u and any(x in u for x in ("/submit", "/execute", "/cancel")),
                     lambda r: (blocked.append(r.request.url), r.abort()))
            pg.route("**/app.js*", lambda r: r.fulfill(body=files["app.js"], content_type="application/javascript"))
            pg.route("**/fp_daily.js*", lambda r: r.fulfill(body=files["fp_daily.js"], content_type="application/javascript"))
            pg.route("**/styles.css*", lambda r: r.fulfill(body=files["styles.css"], content_type="text/css"))
            pg.route(URL, lambda r: r.fulfill(body=files["index.html"], content_type="text/html"))
            pg.route("**/api/footprint-daily**", api)
            pg.route("**/api/liquidation-map*", liqmap)
            pg.route("**/api/liquidation-map/tiers*", tiers)
            pg.route("**/liq_profile.js*", lambda r: r.fulfill(body=files["liq_profile.js"], content_type="application/javascript"))
            pg.add_init_script("try { localStorage.setItem('fpDailyOn', '0'); localStorage.setItem('fpView', 'fp'); localStorage.setItem('liqDays', '1'); } catch (e) {}")
            pg.goto(URL, wait_until="load", timeout=60000)
            pg.wait_for_timeout(8000)
            tag = str(w)
            pg.evaluate("() => { const f = document.getElementById('ofab'); if (f) f.style.visibility = 'hidden'; }")
            pg.locator("#fpDailyBtn").scroll_into_view_if_needed()
            pg.click("#fpDailyBtn")
            pg.wait_for_timeout(2500)
            st = pg.evaluate("""() => ({ on: window.fpDailyActive(), cls: document.getElementById('fpCard').classList.contains('fp-daily-on'),
              candle: getComputedStyle(document.querySelector('#fpCard .candle-container')).display,
              legend: document.getElementById('fpDailyLegend').textContent, status: document.getElementById('fpDailyStatus').textContent,
              pressed: document.getElementById('fpDailyBtn').getAttribute('aria-pressed') })""")
            pg.locator("#fpDaily").scroll_into_view_if_needed()
            box = pg.evaluate("() => document.getElementById('fpDailyCanvas').getBoundingClientRect().toJSON()")
            print(tag, st, "캔버스", round(box["width"]), "×", round(box["height"]), flush=True)
            tb = pg.evaluate(TABS)
            print(tag, "시간 탭", tb, flush=True)
            if tb != {"i": "4", "n": "5", "on": ["1d"]}: fails.append(f"[{tag}] 1d 를 눌렀는데 선택 칸이 1d 가 아님 {tb}")
            if shot:
                pathlib.Path(shot).mkdir(parents=True, exist_ok=True)
                pg.locator("#fpCard .chart-head-left").screenshot(path=str(pathlib.Path(shot) / f"tabs_1d_{w}.png"))
            if not (st["on"] and st["cls"] and st["pressed"] == "true"): fails.append(f"[{tag}] 켜짐 상태 {st}")
            if st["candle"] == "none": fails.append(f"[{tag}] 5분 차트 상자가 숨었다(플롯 위에만 겹쳐야 함)")
            ov = pg.evaluate(OVERLAY, "fpDaily")
            print(tag, "일봉 상자 vs 교체 영역", ov, flush=True)
            if not ov or max(abs(ov[k]) for k in ("dx", "dy", "dw", "dh")) > 1 or not covered(ov): fails.append(f"[{tag}] 일봉 상자 ≠ 교체 영역/모의 판·지지저항 노출/시장 맥락 숨음 {ov}")
            if box["width"] < (300 if w >= 1000 else 200) or box["height"] < 200: fails.append(f"[{tag}] 캔버스 크기 {box}")
            if days["cols"]["d"][-1] not in st["legend"]: fails.append(f"[{tag}] 범례에 마지막 날 없음 {st['legend']!r}")
            canvas_hash = lambda: hashlib.md5(pg.locator("#fpDailyCanvas").screenshot()).hexdigest()  # noqa: E731

            def shot_(name):
                if shot:
                    pathlib.Path(shot).mkdir(parents=True, exist_ok=True)
                    pg.locator("#fpDaily").screenshot(path=str(pathlib.Path(shot) / f"fpd_{name}_{w}.png"))
            n0 = len(cells_hits)
            if w >= 1000 and n0 == 0: fails.append(f"[{tag}] 30일 기본에서 가격 칸을 안 불렀다")
            shot_("default")
            cx, cy = box["x"] + box["width"] * 0.7, box["y"] + box["height"] * 0.3
            h0 = canvas_hash()
            pg.mouse.move(cx, cy)
            for _ in range(5 if w >= 1000 else 9):        # 좁은 화면은 봉 폭 24px(7~12일)까지 들어가야 칸이 된다
                pg.mouse.wheel(0, -240); pg.wait_for_timeout(80)
            pg.wait_for_timeout(1500)
            h1 = canvas_hash()
            if h1 == h0: fails.append(f"[{tag}] 휠 확대가 그림을 안 바꿨다")
            shot_("zoomin")
            if w < 1000 and len(cells_hits) == n0: fails.append(f"[{tag}] 확대해도 가격 칸을 안 불렀다")
            pg.mouse.move(cx, cy); pg.mouse.down(); pg.mouse.move(cx + 240, cy, steps=6); pg.mouse.up()
            pg.wait_for_timeout(600)
            if canvas_hash() == h1: fails.append(f"[{tag}] 끌기 이동이 그림을 안 바꿨다")
            pg.click("#fpDailyTools [data-view='all']"); pg.wait_for_timeout(800)
            shot_("all")
            pg.click("#fpDailyTools [data-view='30']"); pg.wait_for_timeout(1500)
            pg.mouse.move(box["x"] + box["width"] * 0.6, box["y"] + box["height"] * 0.80); pg.wait_for_timeout(300)
            tip = pg.evaluate(TIP)
            print(tag, "레인 툴팁", tip, flush=True)
            if not tip or "청산" not in tip or "OI" not in tip: fails.append(f"[{tag}] 레인 툴팁 {tip!r}")
            if w >= 1000:                                 # 가격 칸 위 = 행 툴팁
                pg.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.25); pg.wait_for_timeout(300)
                print(tag, "칸 툴팁", pg.evaluate(TIP), flush=True)
            shot_("hover")
            pg.mouse.move(5, 5)
            pg.click("#fpDailyBtn"); pg.wait_for_timeout(1500)
            back = pg.evaluate("() => [window.fpDailyActive(), getComputedStyle(document.querySelector('#fpCard .candle-container')).display, document.getElementById('fpDaily').hidden, document.getElementById('candleSvgSnapshot').style.clipPath]")
            if back[0] or back[1] == "none" or not back[2] or back[3]: fails.append(f"[{tag}] 끄고 5분 차트 복귀 실패 {back}")
            tb2 = pg.evaluate(TABS)
            if "1d" in tb2["on"] or len(tb2["on"]) != 1 or tb2["i"] == "4": fails.append(f"[{tag}] 1d 를 끈 뒤 선택 칸이 시간 칸으로 안 돌아옴 {tb2}")
            pg.click("#chartWindowTabs [data-bars='24']"); pg.wait_for_timeout(600)
            pg.click("#fpDailyBtn"); pg.wait_for_timeout(1500)
            pg.click("#chartWindowTabs [data-bars='24']"); pg.wait_for_timeout(1200)   # 지금 고른 시간 칸을 다시 눌러도 일봉이 꺼지고 그 칸으로
            tb3 = pg.evaluate(TABS)
            if pg.evaluate("() => window.fpDailyActive()") or tb3 != {"i": "1", "n": "5", "on": ["2h"]}: fails.append(f"[{tag}] 1d → 같은 시간 칸 복귀 실패 {tb3}")
            if shot:
                pg.locator("#fpCard .chart-head-left").screenshot(path=str(pathlib.Path(shot) / f"tabs_2h_{w}.png"))
            pg.click("#chartWindowTabs [data-bars='12']"); pg.wait_for_timeout(600)
            ovb = pg.evaluate(OVERLAY, "fpDaily")
            if ovb["split"] and not ovb["wall"] or not ovb["split"] and not (ovb["sr"] and ovb["ppm"]): fails.append(f"[{tag}] 끈 뒤 모의 판·지지저항이 안 돌아옴 {ovb}")
            # ── 풋프린트/청산맵 토글 ──
            pg.click("#fpViewTabs [data-view='liq']"); pg.wait_for_timeout(2500)
            lq = pg.evaluate("""() => ({ on: window.fpLiqActive(), candle: getComputedStyle(document.querySelector('#fpCard .candle-container')).display,
              hidden: document.getElementById('liqProfile').hidden, legend: document.getElementById('liqProfileLegend').textContent,
              status: document.getElementById('liqProfileStatus').textContent,
              box: document.getElementById('liqProfileCanvas').getBoundingClientRect().toJSON() })""")
            print(tag, "청산맵", {k: v for k, v in lq.items() if k != "box"}, round(lq["box"]["width"]), "×", round(lq["box"]["height"]), flush=True)
            if not lq["on"] or lq["candle"] == "none" or lq["hidden"]: fails.append(f"[{tag}] 청산맵 토글 상태 {lq}")
            ov2 = pg.evaluate(OVERLAY, "liqProfile")
            if not ov2 or max(abs(ov2[k]) for k in ("dx", "dy", "dw", "dh")) > 1 or not covered(ov2): fails.append(f"[{tag}] 청산맵 상자 ≠ 교체 영역/모의 판·지지저항 노출 {ov2}")
            for d in ("7", "30", "1"):                      # 기간 버튼: 7·30일은 /tiers 를 부르고 그림이 바뀐다
                hb = hashlib.md5(pg.locator("#liqProfileCanvas").screenshot()).hexdigest()
                pg.click(f"#liqProfileTools [data-days='{d}']"); pg.wait_for_timeout(1500)
                pressed = pg.get_attribute(f"#liqProfileTools [data-days='{d}']", "aria-pressed")
                if pressed != "true" or hashlib.md5(pg.locator("#liqProfileCanvas").screenshot()).hexdigest() == hb:
                    fails.append(f"[{tag}] 청산맵 {d}일 버튼이 그림을 안 바꿈(pressed={pressed})")
                if shot:
                    pg.locator("#liqProfile").screenshot(path=str(pathlib.Path(shot) / f"liq{d}d_{w}.png"))
            if "누적 롱 청산" not in lq["legend"] or "75~100배" not in lq["legend"]: fails.append(f"[{tag}] 청산맵 범례 {lq['legend']!r}")
            pg.locator("#liqProfile").scroll_into_view_if_needed()
            lb = pg.evaluate("() => document.getElementById('liqProfileCanvas').getBoundingClientRect().toJSON()")
            pg.mouse.move(lb["x"] + lb["width"] * 0.42, lb["y"] + lb["height"] * 0.6); pg.wait_for_timeout(300)
            ltip = pg.evaluate(TIP)
            print(tag, "청산맵 툴팁", ltip, flush=True)
            if not ltip or "추정" not in ltip: fails.append(f"[{tag}] 청산맵 툴팁 {ltip!r}")
            lhash = lambda: hashlib.md5(pg.locator("#liqProfileCanvas").screenshot()).hexdigest()  # noqa: E731
            lx, ly = lb["x"] + lb["width"] * 0.55, lb["y"] + lb["height"] * 0.5
            pg.mouse.move(lx, ly); pg.wait_for_timeout(200); l0 = lhash()
            for _ in range(4):
                pg.mouse.wheel(0, -240); pg.wait_for_timeout(80)
            pg.wait_for_timeout(400); l1 = lhash()
            if l1 == l0: fails.append(f"[{tag}] 청산맵 휠 확대가 그림을 안 바꿨다")
            if shot:
                pg.locator("#liqProfile").screenshot(path=str(pathlib.Path(shot) / f"liq_zoom_{w}.png"))
            pg.mouse.down(); pg.mouse.move(lx + 160, ly, steps=6); pg.mouse.up(); pg.wait_for_timeout(400)
            if lhash() == l1: fails.append(f"[{tag}] 청산맵 끌기 이동이 그림을 안 바꿨다")
            pg.mouse.dblclick(lx, ly); pg.wait_for_timeout(400)
            leg2 = pg.inner_text("#liqProfileLegend")
            print(tag, "청산맵 확대·이동·복원 후 범례", leg2.replace("\n", " "), flush=True)
            if shot:
                pg.locator("#liqProfile").screenshot(path=str(pathlib.Path(shot) / f"liq_{w}.png"))
            pg.mouse.move(5, 5)
            # 10-09 두 축 독립: 청산맵을 고른 채 1d → 청산맵 그대로 + 토글 보임 · 풋프린트 → 일봉
            VIS = """() => { const d = id => { const e = document.getElementById(id); return getComputedStyle(e).display !== 'none' && e.getBoundingClientRect().height > 0; };
                      return { liq: d('liqProfile'), daily: d('fpDaily'), toggle: d('fpViewTabs'), on: window.fpDailyActive() }; }"""
            pg.click("#fpDailyBtn"); pg.wait_for_timeout(1500)
            v1 = pg.evaluate(VIS)
            if v1 != {"liq": True, "daily": False, "toggle": True, "on": True}: fails.append(f"[{tag}] 청산맵 + 1d: 청산맵·토글이 보여야 함 {v1}")
            pg.click("#fpViewTabs [data-view='fp']"); pg.wait_for_timeout(1500)
            v2 = pg.evaluate(VIS)
            if v2 != {"liq": False, "daily": True, "toggle": True, "on": True}: fails.append(f"[{tag}] 1d + 풋프린트: 일봉이 보여야 함 {v2}")
            if shot:
                pg.locator("#fpCard .chart-head-left").screenshot(path=str(pathlib.Path(shot) / f"tabs_1d_toggle_{w}.png"))
            pg.click("#fpViewTabs [data-view='liq']"); pg.wait_for_timeout(800)
            pg.click("#fpDailyBtn"); pg.wait_for_timeout(800)
            pg.click("#fpViewTabs [data-view='fp']"); pg.wait_for_timeout(1500)
            if pg.evaluate("() => getComputedStyle(document.querySelector('#fpCard .candle-container')).display") == "none": fails.append(f"[{tag}] 풋프린트로 못 돌아옴")
            pg.close()
        b.close()
    print("가격 칸 요청", cells_hits[:6], "…", len(cells_hits), "회 · 히트맵 요청", len(heat_hits), "회 · 청산맵 기간 요청", tier_hits)
    if not heat_hits: fails.append("일봉 히트맵을 안 불렀다")
    if not {7, 30} <= set(tier_hits): fails.append(f"7·30일 청산맵을 안 불렀다 {tier_hits}")
    if blocked: fails.append(f"막힌 주문 요청 {blocked}")
    if errs: fails.append(f"JS 오류 {errs[:3]}")
    print("FAIL" if fails else "PASS", *fails, sep="\n")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shot")
    sys.exit(run(ap.parse_args().shot))
