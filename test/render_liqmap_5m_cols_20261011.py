#!/usr/bin/env python3
"""청산 밀도 층 = 5분봉마다 새 열(2026-10-11 사용자 «1시간마다가 아니라 5분봉이 닫힐 때마다») -- 서버·거래소 **없이** 실제 브라우저로 잰다.

    python3 -u test/render_liqmap_5m_cols_20261011.py [OUT_DIR] [--old-dash DIR]   # --old-dash = 고치기 전 dashboard/live(전후 비교 캡처)

원천 = 서버 실캡처 test/fixtures/liqmap_eth_5m_20261011.json(05:35:26 UTC): 옛 지도(1시간 스냅샷) 그대로 + 테이프 5분 OHLCV.
5분봉 고·저·종가는 차트 캔들(거래소 값)이 있으면 그것, 없으면 테이프(0.1 격자). 거래량은 테이프(= kline, 상대오차 0).
새 지도 = compute_spliced_heatmap_history_5m(이 모듈 그대로) 를 05:25 · 05:30 · 05:35 경계에서 -- 시계를 경계+26초에 고정.
청산맵 막대 = 서버와 같이 latest_window_5m 창 · 같은 기준가 · known_until = 경계(사용자 «둘 다 5분봉 기준»).
검사: 최근 1시간 봉마다 밀도 열이 따로 · 마지막 열 = 마지막 스냅샷 ts 봉(형성 봉) · 경계가 5분 밀리면 마지막 열도 5분 밀림 ·
막대 칸 = 마지막 열 칸 · 가짜 WS 체결로 가장 가까운 저항 칸 가운데를 넘기면 막대가 지우는 칸 = 밀도 층이 끊는 칸(같은 규칙) ·
JS 오류 0 · 바깥 호스트 요청은 전부 abort. 캡처: 풋프린트 12·48·144봉 + 청산맵 보기, 1920·390, 전후 나란히.
"""
import argparse, copy, json, pathlib, sys
from urllib.parse import urlparse
import pandas as pd
from playwright.sync_api import sync_playwright

WT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT))
from scripts.live_liquidation_map_20260824 import compute_spliced_heatmap_history_5m, compute_tier_profile, latest_window_5m  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("out", nargs="?", default="/tmp/render_liqmap_5m_cols_20261011"); ap.add_argument("--old-dash")
A = ap.parse_args()
OUT = pathlib.Path(A.out); OUT.mkdir(parents=True, exist_ok=True)
FIX = json.loads((WT / "test/fixtures/liqmap_eth_5m_20261011.json").read_text("utf-8"))
ORIGIN = "http://dash.local"
CT = {".js": "application/javascript", ".css": "text/css", ".html": "text/html"}
FONT = pathlib.Path("/mnt/c/Windows/Fonts/malgun.ttf")
LM = FIX["liqmap"]; CP = LM["current_price"]
N_SNAP = 144                                              # 서버 LIQUIDATION_MAP_HISTORY_SNAPSHOTS(12h 창 144봉)


def bars():
    by = {c["time"]: c for c in FIX["candles"] if not c.get("partial")}
    rows = []
    for t, lo, hi, v, cl in FIX["bars5"]:
        if t + 300 > FIX["captured_at"]: continue        # 닫힌 봉만
        c = by.get(t)
        rows.append((t, c["high"] if c else hi * 0.1, c["low"] if c else lo * 0.1, c["close"] if c else cl * 0.1, v))
    d5 = pd.DataFrame(rows, columns=["t", "high", "low", "close", "volume"])
    d5["timestamp"] = pd.to_datetime(d5.t, unit="s", utc=True)
    g = d5.groupby(d5.t // 3600 * 3600)
    d1 = pd.DataFrame({"high": g.high.max(), "low": g.low.min(), "close": g.close.last(), "volume": g.volume.sum(), "n": g.size()})
    d1 = d1[d1.n == 12].reset_index(names="t")
    d1["timestamp"] = pd.to_datetime(d1.t, unit="s", utc=True)
    return d1, d5


D1, D5 = bars()
CUTS = []                                                 # (t_last, 새 지도)
for back in (2, 1, 0):
    d5 = D5.iloc[: len(D5) - back]
    t_last = int(d5.t.iloc[-1]) + 300
    d1 = D1[D1.t + 3600 <= t_last]
    m = copy.deepcopy(LM); m["heatmap_history"] = compute_spliced_heatmap_history_5m(d1, d5, CP, 24, N_SNAP)
    win, t_end = latest_window_5m(d1, d5, 24)                 # 서버 load_liquidation_map 과 같은 막대
    tp = compute_tier_profile(win, CP, LM["tier_profile"]["oi_usd"])
    tp.update(current_price=float(win["close"].iloc[-1]), known_until=t_end.isoformat())
    m["tier_profile"] = tp
    m["generated_at"] = pd.Timestamp(t_last + 10, unit="s", tz="UTC").isoformat()
    CUTS.append((t_last, m))
fails, errs, log = [], [], []

DENS = """() => { const svg = document.getElementById('candleSvgSnapshot');
  const rs = [...svg.querySelectorAll('rect[fill-opacity="0.85"]')];
  return { n: rs.length, xs: [...new Set(rs.map((r) => +(+r.getAttribute('x')).toFixed(2)))].sort((a, b) => a - b) }; }"""


def shoot(dash, liqmap, t_last, w, h, tag, measure=False, days7=None):
    cs = [{k: v for k, v in c.items() if k != "partial"} for c in FIX["candles"] if c["time"] < t_last]
    fm = next((c for c in FIX["candles"] if c["time"] == t_last), None)    # 형성 봉 -- 캡처 순간(05:35:26) 차트엔 아직 없어 테이프로
    if fm is None:
        _, lo, hi, _, cl = next(r for r in FIX["bars5"] if r[0] == t_last)
        fm = {"time": t_last, "open": cs[-1]["close"], "high": hi * 0.1, "low": lo * 0.1, "close": cl * 0.1}
    cs.append({**fm, "partial": True})
    fp = copy.deepcopy(FIX["fp"]); fp["bars"] = [b for b in fp["bars"] if b["time"] <= t_last]

    def handle(route):
        u = route.request.url; pr = urlparse(u); log.append(u)
        if not u.startswith(ORIGIN): return route.abort()
        if "/api/manual-" in pr.path and route.request.method != "GET": return route.abort()
        if pr.path == "/api/liquidation-map": return route.fulfill(body=json.dumps(liqmap), content_type="application/json")
        if pr.path == "/api/liquidation-map/tiers" and days7: return route.fulfill(body=json.dumps(days7), content_type="application/json")
        if pr.path == "/api/market-history": return route.fulfill(body=json.dumps({"asset": "eth", "candles": cs}), content_type="application/json")
        if pr.path == "/api/footprint": return route.fulfill(body=json.dumps(fp), content_type="application/json")
        if pr.path == "/__font/kr.ttf" and FONT.is_file(): return route.fulfill(body=FONT.read_bytes(), content_type="font/ttf")
        if pr.path.startswith("/api/"): return route.fulfill(status=503, body="{}")
        name = "index.html" if pr.path.rstrip("/") == "/dashboard/live" else pr.path.split("/dashboard/live/", 1)[-1]
        f = dash / name
        return route.fulfill(body=f.read_bytes(), content_type=CT.get(f.suffix, "application/octet-stream")) if f.is_file() else route.fulfill(status=404, body="")

    ctx = B.new_context(viewport={"width": w, "height": h}, device_scale_factor=1 if w > 500 else 2)
    pg = ctx.new_page(); pg.on("pageerror", lambda e: errs.append(f"{tag}: {e}"))
    wss = []
    pg.route(lambda u: True, handle); pg.route_web_socket(lambda u: True, lambda ws: wss.append(ws))
    pg.clock.install(time=t_last + 26)
    pg.add_init_script("try { localStorage.clear(); } catch (e) {}"
                       "document.addEventListener('DOMContentLoaded', () => { const st = document.createElement('style');"
                       "st.textContent = ['Noto Sans KR', 'Pretendard Variable', 'JetBrains Mono', 'Space Grotesk'].map((f) => `@font-face { font-family: '${f}';"
                       " src: url(/__font/kr.ttf); unicode-range: U+1100-11FF, U+3130-318F, U+AC00-D7AF; }`).join(' '); document.head.appendChild(st); });")
    pg.goto(f"{ORIGIN}/dashboard/live/", wait_until="load"); pg.wait_for_timeout(2500)
    pg.evaluate("() => { const f = document.getElementById('ofab'); if (f) f.style.visibility = 'hidden'; }")
    got = {}
    for nb in (12, 48, 144):
        pg.click(f"#chartWindowTabs [data-bars='{nb}']"); pg.wait_for_timeout(900)
        pg.locator("#fpCard").screenshot(path=str(OUT / f"{tag}_fp{nb}.png"))
        if measure:
            d = pg.evaluate(DENS)
            # 봉 x(왼쪽 가장자리) -- 창의 봉들과 그 x. 열 시작 x 가 봉 x 와 같아야 한다.
            cx = pg.evaluate("(n) => { const c = candleHistoryByAsset.eth; return c.slice(-n).map((k) => k.time); }", nb)
            got[nb] = {**d, "times": cx}
    pg.click("#fpViewTabs [data-view='liq']"); pg.wait_for_timeout(1500)
    pg.locator("#fpCard").screenshot(path=str(OUT / f"{tag}_liq.png"))
    if measure and wss:
        last = liqmap["heatmap_history"][-1]; bw = liqmap["bin_width"]; px = liqmap["tier_profile"]["current_price"]
        k = min(round(b["price"] / bw) for b in last["bins"] if b["price"] > px and b["weight_pct"] > 0)
        hit = k * bw + 0.1                                    # 가장 가까운 저항 칸 가운데를 막 넘긴 체결
        for i, q in enumerate((hit, px)):
            for ws in wss: ws.send(json.dumps({"e": "trade", "p": f"{q:.2f}", "q": "0.1", "T": (t_last + 27 + i) * 1000, "m": False, "t": 900 + i}))
            pg.wait_for_timeout(1600)
        got["same"] = pg.evaluate("""() => { const m = latestLiquidationMap, h = m.heatmap_history, s = h[h.length - 1], bw = m.bin_width;
          const since = Math.floor(Date.parse(s.ts_utc) / 1000), cs = candleHistoryByAsset.eth, tp = m.tier_profile, live = window.fpLiqTp();
          const A = s.bins.filter((b) => b.weight_pct > 0 && liqSweepIdx(cs, since, b.price) >= 0).map((b) => Math.round(b.price / bw));
          const tot = (t, i) => t.tiers.reduce((a, x) => a + x.values[i], 0), n = tp.tiers[0].values.length, B = [], C = [];
          for (let i = 0; i < n; i++) { if (tot(tp, i) > 0) { C.push(tp.lo + i); if (!(tot(live, i) > 0)) B.push(tp.lo + i); } }
          return { A: A.sort((a, b) => a - b), B, C, col: s.bins.filter((b) => b.weight_pct > 0).map((b) => Math.round(b.price / bw)).sort((a, b) => a - b) }; }""")
        got["k"] = k
        pg.locator("#fpCard").screenshot(path=str(OUT / f"{tag}_liq_swept.png"))
    if days7 and wss:                                         # 7일 보기: 끝 시각 = generated_at(known_until 없음) -- 옛 동작 그대로여야
        pg.click("#liqDaysTabs [data-days='7']"); pg.wait_for_timeout(1500)
        tp7 = days7["tier_profile"]; bw7 = tp7["bin_width"]
        tot7 = [sum(t["values"][i] for t in tp7["tiers"]) for i in range(len(tp7["tiers"][0]["values"]))]
        k = min(tp7["lo"] + i for i, v in enumerate(tot7) if v > 0 and (tp7["lo"] + i) * bw7 > tp7["current_price"] + bw7)
        for i, q in enumerate((k * bw7 + 0.1, tp7["current_price"])):
            for ws in wss: ws.send(json.dumps({"e": "trade", "p": f"{q:.2f}", "q": "0.1", "T": (t_last + 30 + i) * 1000, "m": False, "t": 950 + i}))
            pg.wait_for_timeout(1600)
        got["d7"] = pg.evaluate("""() => { const tp = window.fpLiqTp(), n = tp.tiers[0].values.length, out = [];
          for (let i = 0; i < n; i++) if (!(tp.tiers.reduce((a, x) => a + x.values[i], 0) > 0)) out.push(tp.lo + i); return out; }""")
        got["d7_k"] = k
        pg.locator("#fpCard").screenshot(path=str(OUT / f"{tag}_liq7d_swept.png"))
    ctx.close()
    return got


def side_by_side(a, b, out, la, lb):
    from PIL import Image, ImageDraw, ImageFont
    A_, B_ = Image.open(a), Image.open(b)
    f = ImageFont.truetype(str(FONT), 26) if FONT.is_file() else None
    vertical = A_.width > 1000
    W = A_.width if vertical else A_.width + B_.width + 20
    H = A_.height + B_.height + 100 if vertical else max(A_.height, B_.height) + 50
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
    if vertical:
        d.text((10, 8), la, fill="black", font=f); im.paste(A_, (0, 45))
        d.text((10, A_.height + 55), lb, fill="black", font=f); im.paste(B_, (0, A_.height + 95))
    else:
        d.text((10, 8), la, fill="black", font=f); im.paste(A_, (0, 45))
        d.text((A_.width + 30, 8), lb, fill="black", font=f); im.paste(B_, (A_.width + 20, 45))
    im.save(out)


def stack(paths, labels, out):
    from PIL import Image, ImageDraw, ImageFont
    ims = [Image.open(p) for p in paths]; f = ImageFont.truetype(str(FONT), 26) if FONT.is_file() else None
    W = max(i.width for i in ims); H = sum(i.height + 45 for i in ims)
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im); y = 0
    for i, lab in zip(ims, labels):
        d.text((10, y + 8), lab, fill="black", font=f); im.paste(i, (0, y + 40)); y += i.height + 45
    im.save(out)


hhmm = lambda t: pd.Timestamp(t, unit="s", tz="UTC").strftime("%H:%M")   # noqa: E731
with sync_playwright() as p:
    B = p.chromium.launch()
    res = {}
    for t_last, m in CUTS:
        res[t_last] = shoot(WT / "dashboard/live", m, t_last, 1920, 1080, f"new_{hhmm(t_last).replace(':', '')}_1920", measure=True)
    t3, m3 = CUTS[-1]
    shoot(WT / "dashboard/live", m3, t3, 390, 844, "new_0535_390")
    # 7일 보기 회귀: 형성 1시간봉 포함 막대, generated_at = 형성 5분봉 시작 30초 전(그 봉 고·저가 지우기에 들어가야 한다)
    fm_close = next(r for r in FIX["bars5"] if r[0] == t3)[4] * 0.1
    tp7 = compute_tier_profile(D1, fm_close, LM["tier_profile"]["oi_usd"])
    D7 = {"days": 7, "tier_profile": tp7, "generated_at": pd.Timestamp(t3 - 30, unit="s", tz="UTC").isoformat()}
    d7 = {"new": shoot(WT / "dashboard/live", m3, t3, 1920, 1080, "new_0535_1920_d7", days7=D7)}
    if A.old_dash:
        d7["old"] = shoot(pathlib.Path(A.old_dash), m3, t3, 1920, 1080, "old_0535_1920_d7", days7=D7)
    if A.old_dash:
        old = copy.deepcopy(LM); old["generated_at"] = pd.Timestamp(t3 + 10, unit="s", tz="UTC").isoformat()
        for w, h in ((1920, 1080), (390, 844)):
            shoot(pathlib.Path(A.old_dash), old, t3, w, h, f"old_0535_{w}")
    B.close()

# 검사 -- 1h(12봉) 창: 봉마다 열 하나 · 마지막 열 = 마지막 스냅샷 봉(형성 봉) · 경계마다 밀림
for t_last, m in CUTS:
    g = res[t_last][12]
    snaps = [int(pd.Timestamp(s["ts_utc"]).timestamp()) for s in m["heatmap_history"]]
    print(f"[{hhmm(t_last)}] 스냅샷 {len(snaps)} ({hhmm(snaps[0])}~{hhmm(snaps[-1])}) · 1h 창 봉 {len(g['times'])} ({hhmm(g['times'][0])}~{hhmm(g['times'][-1])})"
          f" · 밀도 rect {g['n']} · 열 시작 x {len(g['xs'])}개 · 4h rect {res[t_last][48]['n']} · 12h rect {res[t_last][144]['n']}", flush=True)
    if snaps[-1] != t_last: fails.append(f"[{hhmm(t_last)}] 마지막 스냅샷 {hhmm(snaps[-1])} ≠ 경계 {hhmm(t_last)}")
    if g["times"][-1] != t_last: fails.append(f"[{hhmm(t_last)}] 형성 봉 {hhmm(g['times'][-1])} ≠ 경계")
    if len(g["xs"]) < len(g["times"]): fails.append(f"[{hhmm(t_last)}] 1h 창 열 시작 {len(g['xs'])} < 봉 {len(g['times'])} -- 봉마다 새 열이어야")
sm = res[CUTS[-1][0]].get("same")
if sm:
    print(f"[막대 vs 마지막 열] 칸 수 막대 {len(sm['C'])} · 열 {len(sm['col'])} · 같음 {sm['C'] == sm['col']} · 체결로 넘긴 칸 {res[CUTS[-1][0]]['k']}"
          f" · 밀도 층 끊음 {sm['A']} · 막대 지움 {sm['B']}", flush=True)
    if sm["C"] != sm["col"]: fails.append("막대 칸 ≠ 마지막 열 칸")
    if not sm["A"] or sm["A"] != sm["B"]: fails.append(f"지나간 칸 규칙 불일치/빈 대조 A={sm['A']} B={sm['B']}")
else:
    fails.append("막대·열 대조를 못 잼")
base7 = {tp7["lo"] + i for i in range(len(tp7["tiers"][0]["values"])) if not (sum(t["values"][i] for t in tp7["tiers"]) > 0)}
for who, g in d7.items():
    swept = sorted(set(g.get("d7", [])) - base7)
    print(f"[7일 회귀 {who}] 체결로 넘긴 칸 {g.get('d7_k')} · 지운 칸 {swept}", flush=True)
    if not swept or g.get("d7_k") not in swept: fails.append(f"[7일 {who}] 지우기 안 됨 {swept}")
if "old" in d7 and d7["old"].get("d7") != d7["new"].get("d7"): fails.append("7일 보기 지우기가 고치기 전과 다름")
if len(CUTS) == 3 and not (CUTS[1][0] - CUTS[0][0] == CUTS[2][0] - CUTS[1][0] == 300): fails.append("경계 간격 ≠ 5분")

# 비교 이미지
t3 = CUTS[-1][0]
if A.old_dash:
    for w in (1920, 390):
        for v in ("fp12", "fp48", "fp144", "liq"):
            nw = OUT / (f"new_0535_{w}_{v}.png")
            side_by_side(OUT / f"old_0535_{w}_{v}.png", nw, OUT / f"cmp_{w}_{v}.png", "전: 1시간마다(옛 지도)", "후: 5분봉마다(새 지도)")
for v in ("fp12", "fp48"):
    stack([OUT / f"new_{hhmm(t).replace(':', '')}_1920_{v}.png" for t, _ in CUTS],
          [f"후 · 시계 {hhmm(t)}:26 UTC (마지막 스냅샷 {hhmm(t)})" for t, _ in CUTS], OUT / f"seq_1920_{v}.png")

out = [u for u in log if not u.startswith(ORIGIN)]
print("바깥 호스트 요청(전부 abort)", len(out), "· JS 오류", errs[:3])
if errs: fails.append(f"JS 오류 {errs[:3]}")
print("FAIL" if fails else "PASS", *fails, sep="\n")
sys.exit(1 if fails else 0)
