#!/usr/bin/env python3
"""풋프린트 칸 세로 위치 = 캔들 고저(2026-10-11) -- 서버·거래소 **없이**(완전 오프라인) 실제 브라우저로 잰다.

    python3 -u test/render_fp_row_align_20261011.py [OUT_DIR] [--dash DIR]     # --dash = 다른 dashboard/live 로(음성 대조)

서버 칸 키 = round(가격/버킷) = 버킷 **중심**이다. 화면이 floor(중심/행)으로 행 [k, k+1] 에 넣어 칸이 반 버킷($0.25) 위로 그려졌다.
검사(1920 · 1h/2h/4h · 조용한 실데이터 사본 test/fixtures/fp_eth_quiet_20261010.json + 변동 큰 합성 봉 6개($15·31칸 → 행 > 버킷)):
  봉마다 캔들 심지 위끝(고가)이 맨 위 칸 안, 아래끝(저가)이 맨 아래 칸 안(±0.75px) · 형성 봉 포함(가짜 WS 체결로 그 봉 버킷 안 고저).
  체결 기둥 툴팁의 행 범위가 버킷 경계(.25/.75)에서 끊긴다 · 호가 띠(floor 격자 원천) 줄이 체결 기둥 줄과 같다. 390 폭 한 장. JS 오류 0 · 바깥 호스트 요청은 전부 abort.
"""
import argparse, copy, json, pathlib, random, re, sys, time
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

WT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("out", nargs="?", default="/tmp/render_fp_row_align_20261011"); ap.add_argument("--dash")
A = ap.parse_args()
DASH = pathlib.Path(A.dash) if A.dash else WT / "dashboard/live"
OUT = pathlib.Path(A.out); OUT.mkdir(parents=True, exist_ok=True)
FIX = json.loads((WT / "test/fixtures/fp_eth_quiet_20261010.json").read_text("utf-8"))
ORIGIN = "http://dash.local"
CT = {".js": "application/javascript", ".css": "text/css", ".html": "text/html"}
FONT = pathlib.Path("/mnt/c/Windows/Fonts/malgun.ttf")


def book():
    """호가 띠(마지막 1초): floor 0.5 칸 40개(+매수 아래 · −매도 위) -- 서버 수집기와 같은 격자."""
    import base64, struct
    mid = FIX["footprint"]["bars"][-1]["levels"][0][0]; lo = int((mid - 10) // 0.5); rnd = random.Random(3)
    q = [(1 if (lo + i) * 0.5 < mid else -1) * (5 + rnd.random() * 80) for i in range(40)]
    return {"bin_lo": lo, "bin_size": 0.5, "q_f4": base64.b64encode(struct.pack(f"<{len(q)}f", *q)).decode()}


BOOK = book()
MEASURE_BOOK = """() => { const rs = [...document.getElementById('candleSvgSnapshot').querySelectorAll('rect')];
  const tr = rs.filter((r) => r.getAttribute('fill') === 'var(--amber)').map((r) => +r.getAttribute('y'));
  const bk = rs.filter((r) => r.getAttribute('fill-opacity') === '0.3' && /--(good|bad)/.test(r.getAttribute('fill')));
  const x0 = Math.max(...bk.map((r) => +r.getAttribute('x')));
  return { tr, bk: bk.filter((r) => +r.getAttribute('x') === x0).map((r) => +r.getAttribute('y')) }; }"""


def scenario(volatile):
    """지금 봉 = 사본의 마지막(형성) 봉이 되게 시각을 민다. volatile 이면 마감봉 6개를 $15 폭·31칸으로 바꾼다."""
    fp, cs = copy.deepcopy(FIX["footprint"]), copy.deepcopy(FIX["candles"])
    d = int(time.time() // 300 * 300) - fp["bars"][-1]["time"]
    for b in fp["bars"]: b["time"] += d
    fp["okxLive"] = {}
    for c in cs: c["time"] += d
    if volatile:
        rnd, by = random.Random(11), {c["time"]: c for c in cs}
        for b in fp["bars"][-7:-1]:
            c = by[b["time"]]; lo = round(c["low"] - 7 + rnd.random() * 2, 2); hi = round(lo + 15 + rnd.random() * 0.5, 2)
            o, cl = round(lo + rnd.random() * 15, 2), round(lo + rnd.random() * 15, 2)
            c.update(open=o, high=hi, low=lo, close=cl)
            k0, k1 = round(lo / 0.5), round(hi / 0.5)       # 서버 규약(파이썬 round = 은행가 · 여기 값은 .5 아님)
            b["levels"] = [[k * 0.5, rnd.random() * 300, rnd.random() * 300, 0, 0, 0, 0] for k in range(k0, k1 + 1)]
    return fp, cs


MEASURE = """() => {
  const svg = document.getElementById('candleSvgSnapshot'), out = [];
  svg.querySelectorAll(':scope > g').forEach((g) => {
    const wk = [...g.children].find((e) => e.tagName === 'line' && e.getAttribute('stroke-width') === '1');
    const cells = [...g.children].filter((e) => e.tagName === 'rect' && /· 매(수|도) /.test(e.textContent));
    if (!wk || !cells.length) return;
    const ys = cells.map((r) => +r.getAttribute('y')), h = +cells[0].getAttribute('height');
    const top = Math.min(...ys), bot = Math.max(...ys) + h, yH = +wk.getAttribute('y1'), yL = +wk.getAttribute('y2');
    out.push({ rowPx: +h.toFixed(2), hiIn: yH >= top - 0.75 && yH <= top + h + 0.75, loIn: yL >= bot - h - 0.75 && yL <= bot + 0.75,
               offTop: +((top + h / 2) - yH).toFixed(2), offBot: +((bot - h / 2) - yL).toFixed(2) });
  });
  return out;
}"""
TIP = "() => { const t = document.getElementById('chartTooltip'); return t && t.classList.contains('visible') ? t.textContent : null; }"
fails, errs, log = [], [], []


def run(pg_w, pg_h, volatile, wins, tag, b):
    fp, cs = scenario(volatile)

    def handle(route):
        u = route.request.url; pr = urlparse(u); log.append(u)
        if not u.startswith(ORIGIN): return route.abort()
        if "/api/manual-" in pr.path and route.request.method != "GET": return route.abort()
        if pr.path == "/api/footprint": return route.fulfill(body=json.dumps(fp), content_type="application/json")
        if pr.path == "/api/market-history": return route.fulfill(body=json.dumps({"asset": "eth", "candles": cs}), content_type="application/json")
        if pr.path == "/api/flow/heatmap": return route.fulfill(body=json.dumps({"book": BOOK, "rows": None}), content_type="application/json")
        if pr.path == "/__font/kr.ttf" and FONT.is_file(): return route.fulfill(body=FONT.read_bytes(), content_type="font/ttf")
        if pr.path.startswith("/api/"): return route.fulfill(status=503, body="{}")
        name = "index.html" if pr.path.rstrip("/") == "/dashboard/live" else pr.path.split("/dashboard/live/", 1)[-1]
        f = DASH / name
        return route.fulfill(body=f.read_bytes(), content_type=CT.get(f.suffix, "application/octet-stream")) if f.is_file() else route.fulfill(status=404, body="")

    wss = []
    ctx = b.new_context(viewport={"width": pg_w, "height": pg_h}, device_scale_factor=1 if pg_w > 500 else 2)
    pg = ctx.new_page(); pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.route(lambda u: True, handle); pg.route_web_socket(lambda u: True, lambda ws: wss.append(ws))
    pg.add_init_script("try { localStorage.clear(); } catch (e) {}"
                       "document.addEventListener('DOMContentLoaded', () => { const st = document.createElement('style');"
                       "st.textContent = ['Noto Sans KR', 'Pretendard Variable', 'JetBrains Mono', 'Space Grotesk'].map((f) => `@font-face { font-family: '${f}';"
                       " src: url(/__font/kr.ttf); unicode-range: U+1100-11FF, U+3130-318F, U+AC00-D7AF; }`).join(' '); document.head.appendChild(st); });")
    pg.goto(f"{ORIGIN}/dashboard/live/", wait_until="load"); pg.wait_for_timeout(2500)
    lv = [l[0] for l in fp["bars"][-1]["levels"]]                # 형성 봉: 그 봉 버킷 범위 안의 고저로 체결을 흘린다
    for i, px in enumerate((lv[0] + 0.1, min(lv) - 0.19, max(lv) + 0.12, lv[0])):
        for w in wss: w.send(json.dumps({"e": "trade", "p": f"{px:.2f}", "q": "0.1", "T": int(time.time() * 1000), "m": False, "t": 10 + i}))
        pg.wait_for_timeout(150)
    pg.evaluate("() => { const f = document.getElementById('ofab'); if (f) f.style.visibility = 'hidden'; }")
    for bars in wins:
        pg.click(f"#chartWindowTabs [data-bars='{bars}']"); pg.wait_for_timeout(1200)
        m = pg.evaluate(MEASURE)
        # OKX 체결이 바이낸스 고저 밖이면 그 칸은 캔들 밖이 맞다(사본 144봉 중 31봉) -- 칸 범위 = round(고저/버킷) 인 봉만 잰다
        win = pg.evaluate("(n) => candleHistoryByAsset.eth.slice(-n).map((c) => [c.time, c.high, c.low])", bars)
        lv = {x["time"]: [l[0] for l in x["levels"]] for x in fp["bars"]}
        if len(win) != len(m): fails.append(f"[{tag} {bars}] 봉 수 {len(win)} ≠ 칸 그린 봉 {len(m)}")
        okx = sum(1 for t, hi, lo in win if not (t in lv and max(lv[t]) == round(hi / 0.5) * 0.5 and min(lv[t]) == round(lo / 0.5) * 0.5))
        m = [r for r, (t, hi, lo) in zip(m, win) if t in lv and max(lv[t]) == round(hi / 0.5) * 0.5 and min(lv[t]) == round(lo / 0.5) * 0.5]
        nbad = sum(not (r["hiIn"] and r["loIn"]) for r in m)
        mt = lambda k: round(sum(r[k] for r in m) / max(1, len(m)), 2)      # noqa: E731
        print(f"[{tag} {bars}봉] 잰 봉 {len(m)}(OKX 넘침 제외 {okx}) · 행 {sorted({r['rowPx'] for r in m})}px · 고저가 칸 밖 {nbad}봉 · "
              f"평균(칸 중심−심지) 위 {mt('offTop')} 아래 {mt('offBot')}px · 최대 |위| {max((abs(r['offTop']) for r in m), default=0)}", flush=True)
        if not m: fails.append(f"[{tag} {bars}] 잴 봉 없음")
        if nbad: fails.append(f"[{tag} {bars}] 고가·저가가 맨 위·아래 칸 밖 {nbad}/{len(m)}봉")
        pg.locator("#candleSvgSnapshot").screenshot(path=str(OUT / f"fp_{tag}_{bars}.png"))
    if pg_w > 500:                                                 # 호가 띠 줄 = 체결 기둥 줄(같은 가격 = 같은 줄, 09-28 사용자)
        mb = pg.evaluate(MEASURE_BOOK); lo_, hi_ = min(mb["tr"], default=0), max(mb["tr"], default=0)
        inside = [y for y in mb["bk"] if lo_ - 0.6 <= y <= hi_ + 0.6]
        off = [y for y in inside if min(abs(y - t) for t in mb["tr"]) > 0.6]
        print(f"[{tag}] 호가 띠 줄 {len(inside)}개(체결 기둥 범위 안) 중 체결 줄과 어긋남 {len(off)}", flush=True)
        if not inside: fails.append(f"[{tag}] 호가 띠를 못 그렸다 {len(mb['bk'])}")
        if off: fails.append(f"[{tag}] 호가 띠 줄이 체결 기둥 줄과 어긋남 {len(off)}/{len(inside)}")
    if pg_w > 500 and not volatile:                                # 체결 기둥 툴팁의 행 범위 = 버킷 경계
        spot = pg.evaluate("""() => { const svg = document.getElementById('candleSvgSnapshot'), m = svg.getScreenCTM();
            const g = [...svg.querySelectorAll(':scope > g')].reverse().find((g) => [...g.children].some((e) => /· 매수 /.test(e.textContent)));
            const r = [...g.children].find((e) => /· 매수 /.test(e.textContent)), y = +r.getAttribute('y') + +r.getAttribute('height') / 2;
            return { x: m.a * (fpProfileSpan[0] + 120) + m.e, y: m.d * y + m.f }; }""")
        pg.mouse.move(spot["x"], spot["y"]); pg.wait_for_timeout(300)
        tip = pg.evaluate(TIP) or ""
        print(f"[{tag}] 체결 기둥 툴팁", tip[:60])
        rng = (re.search(r"\d+\.\d+–\d+\.\d+", tip) or [""])[0]
        if not rng.endswith((".25", ".75")): fails.append(f"[{tag}] 체결 기둥 행 범위가 버킷 경계가 아님 {rng!r}")
    ctx.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    run(1920, 1080, False, (12, 24, 48), "1920_quiet", b)
    run(1920, 1080, True, (12, 24, 48), "1920_volatile", b)
    run(390, 844, False, (12,), "390_quiet", b)
    b.close()
out = [u for u in log if not u.startswith(ORIGIN)]
print("바깥 호스트 요청(전부 abort)", len(out), "· JS 오류", errs[:3])
if errs: fails.append(f"JS 오류 {errs[:3]}")
print("FAIL" if fails else "PASS", *fails, sep="\n")
sys.exit(1 if fails else 0)
