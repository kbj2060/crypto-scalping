"""마우스 조작 일치: 같은 입력 열을 지금 화면(dashboard/live -- 헤드리스 크로미움에 실제 마우스 이벤트)과 rtui(웹어셈블리)에 주고,
단계마다 창 상태(봉 수·tf·오른쪽 끝·소수·빈 칸·세로 이동·보이는 첫/끝 봉·세로 범위)와 그린 목록 좌표를 맞댄다.
네트워크 없음(harness.py): /api/footprint · market-history · footprint-week = compare/fixture(합성) · WebSocket = 죽은 가짜.
python -I rtui/compare/ops.py <worktree> [열 이름...]   (먼저 make_fixture.py · compare/build.sh)  →  rtui/compare/ops.md · _ops_<열>.json
페이지 하나씩(지금 화면 → rtui) · 열마다 새로 띄운다(처음 = 36봉 · 5m · 실시간)."""
import json, pathlib, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # -I 는 스크립트 폴더를 안 넣는다
from harness import INIT, LAUNCH, TZ, make_route

WT = pathlib.Path(sys.argv[1]).resolve()
HERE, FX = WT / "rtui/compare", WT / "rtui/compare/fixture"
API = {"/api/footprint": FX / "footprint.json", "/api/market-history": FX / "market-history.json", "/api/footprint-week": FX / "footprint-week.json"}
route = make_route(WT, API)
X, Y = 600, 300   # svg 좌표(가격 플롯 안)
W = lambda dy, x=X, y=Y: ("wheel", x, y, 0, dy)
SEQS = {   # 단계: ("wheel", x, y, dx, dy) · ("drag", x, y, dx, dy) · ("dbl", x, y) · ("tf", k)
    "A_기본열": [W(100), W(100), W(100), ("drag", 500, 300, -120, 0), ("tf", 12), W(-100), W(-100), W(-100), W(-100), W(-100)],
    "B_맨왼쪽끝": [("drag", 300, 300, 900, 0), ("drag", 300, 300, 900, 0), ("drag", 300, 300, 900, 0), ("drag", 300, 300, 900, 0), W(-100, 200)],
    "C_최소최대": [W(-400)] * 3 + [W(400)] * 8 + [W(-100, 150)],
    "D_벗어났다돌아오기": [("drag", 400, 300, 300, 0), ("drag", 400, 300, 0, 80), ("drag", 700, 300, -700, 0), ("drag", 700, 300, -400, 0), ("dbl", 500, 300)],
    "E_30m": [("tf", 6), W(100, 200), ("drag", 400, 300, 200, 0), W(-100, 300), ("tf", 1)],
    "F_DENSE": [W(400), W(300), W(400), W(400), ("tf", 12), W(400), W(-400)],
    "G_가로휠": [("wheel", X, Y, -150, 0), ("wheel", X, Y, -300, 20), ("wheel", X, Y, 900, 0)],
}
pick = [a for a in sys.argv[2:] if a in SEQS] or list(SEQS)

JS_STATE = r"""() => ({ bars: chartWindowBars, tfk: chartTfK, end: chartPanEnd, frac: chartPanFrac, right: chartPanRight, py: chartPanY, ypp: chartYPerPx })"""
JS_DUMP = r"""() => {
  Object.keys(pillWMax).forEach((k) => delete pillWMax[k]);
  window.__rec = []; renderSnapshotChart(); const rec = window.__rec; window.__rec = null;
  const svg = document.getElementById('candleSvgSnapshot'), b = svg.getBoundingClientRect();
  const vis = (e) => { const cs = getComputedStyle(e); return cs.display !== 'none' && cs.visibility !== 'hidden' && e.closest('svg') === svg && !e.closest('defs') && !e.closest('.hover-layer'); };
  const els = [...svg.querySelectorAll('rect,line,circle,text')].filter(vis).map((e) => ({ t: e.tagName, a: Object.fromEntries([...e.attributes].map((x) => [x.name, x.value])), s: e.tagName === 'text' ? e.textContent : undefined }));
  const g = chartGeo, c = g.bars;
  return { geo: { ml: g.ml, cw: g.cw, n: g.n, mt: g.mt, pb: g.pb, ymin: g.ymin, ymax: g.ymax, off: g.off, box: b.toJSON() }, t0: c[0].time, t1: c[c.length - 1].time, nc: c.length, rec, els };
}"""


def steps(pg, box, seq, js):
    out = []
    for st in seq:
        if st[0] == "wheel":
            pg.mouse.move(box["x"] + st[1], box["y"] + st[2]); pg.wait_for_timeout(60)
            pg.mouse.wheel(st[3], st[4])
        elif st[0] == "drag":
            pg.mouse.move(box["x"] + st[1], box["y"] + st[2]); pg.wait_for_timeout(60)
            pg.mouse.down(); pg.mouse.move(box["x"] + st[1] + st[3], box["y"] + st[2] + st[4], steps=12)
            pg.wait_for_timeout(180)   # 놓기 전 멈춤 = 관성 없음(flingFrom: 마지막 움직임이 40ms 넘게 지났다)
            pg.mouse.up()
        elif st[0] == "dbl":
            pg.mouse.dblclick(box["x"] + st[1], box["y"] + st[2])
        elif st[0] == "tf":
            pg.evaluate("(k) => document.querySelector(k === 1 ? '#chartWindowTabs .asset-tab[data-bars]:not([data-tfk])' : `#chartWindowTabs .asset-tab[data-tfk=\"${k}\"]`).click()" if js
                        else "(k) => window.rtui.tf(k)", st[1])
        pg.wait_for_timeout(450)   # 1주일·풋프린트 응답(가로챈 디스크) + 다음 프레임
        if js:
            d = pg.evaluate(JS_DUMP); d["state"] = pg.evaluate(JS_STATE)
        else:
            pg.evaluate("() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))")
            d = json.loads(pg.evaluate("window.rtui_scene()")); d["state"] = json.loads(pg.evaluate("window.rtui.state()"))
        out.append(d)
    return out


def run(name):
    res = {}
    with sync_playwright() as p:
        br = p.chromium.launch(args=LAUNCH)
        try:
            ctx = br.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1, service_workers="block", timezone_id=TZ, locale="ko-KR")
            ctx.route("**/*", route); ctx.add_init_script(INIT)
            pg = ctx.new_page()
            pg.goto("http://fx.invalid/dashboard/live/", wait_until="domcontentloaded")
            pg.wait_for_timeout(7000)
            box = pg.evaluate("() => document.getElementById('candleSvgSnapshot').getBoundingClientRect().toJSON()")
            wh = pg.evaluate("() => [chartGeo.wAll, chartGeo.hAll, new Date().getTimezoneOffset()]")
            res["js"] = steps(pg, box, SEQS[name], True)
            ctx.close()
            ctx = br.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1, timezone_id=TZ)
            ctx.route("**/*", route); ctx.add_init_script(INIT)
            pr = ctx.new_page()
            pr.on("pageerror", lambda e: print("rtui pageerror", str(e)[:300]))
            pr.goto(f"http://fx.invalid/rtui/index.html?mode=shot&x={box['x']}&y={box['y']}&w={wh[0]}&h={wh[1]}&tz={wh[2]}&bg=none&wk=0")
            pr.wait_for_function("window.RESULT", timeout=20_000, polling=100)
            res["rt"] = steps(pr, box, SEQS[name], False)
        finally:
            br.close()
    json.dump(res, open(HERE / f"_ops_{name}.json", "w"), ensure_ascii=False)
    return res


CN = {"#5dbf83": "Good", "#ef9387": "Bad", "#a6acc1": "Muted", "#171b23": "ChartBg", "rgba(203, 209, 227, 0.18)": "Line", "#5aa9f5": "Turnover",
      "#eef0f6": "Text", "#e99c57": "Amber", "#dfe4ee": "Accent", "#f5f6fb": "Ink", "#cbd1e3": "Line"}


def jnorm(r):
    """캔버스 기록 하나 → 그리기 목록(사각형 경로 여럿 = cvUnion 한 칠이면 여럿)"""
    c, cs = CN.get(r["s"], r["s"]), CN.get(r["st"], r["st"])
    if r["k"] == "fillRect":
        x, y, w, h = r["a"]; return [dict(k="rect", x=x, y=y, w=w, h=h, c=c, a=r["al"], rx=0)]
    if r["k"] == "fill":
        return [dict(k="rect", x=q[1], y=q[2], w=q[3], h=q[4], c=c, a=r["al"], rx=q[5] if q[0] == "roundRect" else 0) for q in r["p"] if q[0] in ("rect", "roundRect")] or \
               [dict(k="circle", x=q[1], y=q[2], r=q[3], c=c, a=r["al"]) for q in r["p"] if q[0] == "arc"]
    if r["k"] == "stroke":
        q = r["p"][0] if r["p"] else None
        if q and q[0] in ("rect", "roundRect"): return [dict(k="srect", x=q[1], y=q[2], w=q[3], h=q[4], c=cs, a=r["al"])]
        return [dict(k="line", pts=[q[1:3] for q in r["p"] if q[0] in ("moveTo", "lineTo")], c=cs, a=r["al"], lw=r["lw"])]
    if r["k"] == "fillText":
        return [dict(k="text", x=r["a"][1], y=r["a"][2], s=r["a"][0], c=c, a=r["al"])]
    return []


def rnorm(p):
    if p["k"] == "rect": return dict(k="rect", x=p["x"], y=p["y"], w=p["w"], h=p["h"], c=p["c"], a=p["a"], rx=min(p["rx"], p["w"] / 2, p["h"] / 2))
    if p["k"] == "srect": return dict(k="srect", x=p["x"], y=p["y"], w=p["w"], h=p["h"], c=p["c"], a=p["a"])
    if p["k"] == "line": return dict(k="line", pts=p["pts"], c=p["c"], a=p["a"], lw=p["lw"])
    if p["k"] == "text": return dict(k="text", x=p["x"], y=p["y"], s=p["s"], c=p["c"], a=p.get("a", 1.0))
    return dict(k="circle", x=p["x"], y=p["y"], r=p["r"], c=p["c"], a=p["a"])


def cmp_list(a, b):
    """같은 순서 목록 -- (개수 문자열, 좌표 최대 차, 어긋남 수, 첫 어긋남)"""
    n, dx, bad, first = min(len(a), len(b)), 0.0, abs(len(a) - len(b)), None
    for i in range(n):
        x, y = a[i], b[i]
        ok = x["k"] == y["k"] and x.get("c") == y.get("c") and abs(x.get("a", 1) - y.get("a", 1)) < 0.002 and x.get("s") == y.get("s")
        if ok and x["k"] == "line": ok = len(x["pts"]) == len(y["pts"])
        if ok:
            ks = [abs(u - v) for p, q in zip(x["pts"], y["pts"]) for u, v in zip(p, q)] + [abs(x["lw"] - y["lw"])] if x["k"] == "line" else \
                 [abs(x[k] - y[k]) for k in ("x", "y", "w", "h", "rx", "r") if k in x and k in y]
            dx = max([dx] + ks)
            ok = max(ks, default=0) < 0.01
        if not ok:
            bad += 1
            if first is None: first = (i, x, y)
    if first is None and len(a) != len(b): first = (n, a[n] if n < len(a) else None, b[n] if n < len(b) else None)
    return f"{len(a)}/{len(b)}", dx, bad, first


def svg_items(js):
    """지금 화면 svg: 시간축 글자(13px) · 프로파일(체결 기둥 왼쪽 오른쪽 · 알약 제외) · 레인 글자 · 알약"""
    px0 = js["geo"]["ml"] + js["geo"]["cw"]
    tax, prof = [], []
    for e in js["els"]:
        a = e["a"]
        if e["t"] == "text" and a.get("font-size") == "13": tax.append(dict(k="text", x=float(a["x"]), y=float(a["y"]), s=e["s"], c="Muted", a=1.0))
        elif e["t"] == "rect" and "data-pill" not in a and float(a.get("x", -1)) >= px0:
            f = lambda k: float(a[k])
            if a.get("fill") != "none": prof.append(dict(k="rect", x=f("x"), y=f("y"), w=f("width"), h=f("height"), c=CN.get(a["fill"], a["fill"].replace("var(--", "").rstrip(")").title()), a=float(a.get("fill-opacity", 1)), rx=0))
            if "stroke" in a: prof.append(dict(k="srect", x=f("x"), y=f("y"), w=f("width"), h=f("height"), c="Ink", a=float(a.get("stroke-opacity", 1))))
        elif e["t"] == "circle": prof.append(dict(k="circle", x=float(a["cx"]), y=float(a["cy"]), r=float(a["r"]), c="Accent", a=1.0))
    pill = next(((float(e["a"]["x"]), float(e["a"]["y"])) for e in js["els"] if e["t"] == "rect" and "data-pill" in e["a"]), None)
    return tax, prof, pill


def compare(name, res):
    rows = []
    for i, (j, r) in enumerate(zip(res["js"], res["rt"])):
        S, R = j["state"], r["state"]
        st_ok = all((S[k] is None and R[k] is None) or (S[k] is not None and R[k] is not None and abs(S[k] - R[k]) < 1e-6) for k in ("bars", "tfk", "end", "frac", "right", "py")) and abs(S["ypp"] - R["ypp"]) < 1e-9
        g, h = j["geo"], r["geo"]
        win_ok = (j["t0"], j["t1"], j["nc"]) == (r["t0"], r["t1"], r["nc"]) and abs(g["ymin"] - h["ymin"]) < 1e-6 and abs(g["ymax"] - h["ymax"]) < 1e-6 and abs(g["off"] - h["off"]) < 1e-9
        P = r["prims"]
        bg = cmp_list([q for x in j["rec"] if x["L"] == "bg" for q in jnorm(x)], [rnorm(p) for p in P if p["g"] == "bg"])
        fg = cmp_list([q for x in j["rec"] if x["L"] == "fg" for q in jnorm(x)], [rnorm(p) for p in P if p["g"] in ("fg", "stack", "lane", "ua")])
        tax, prof, pill = svg_items(j)
        ta = cmp_list(tax, [rnorm(p) for p in P if p["g"] == "svg" and p["k"] == "text"])
        pr = cmp_list(prof, [rnorm(p) for p in P if p["g"] == "profile" and p["k"] in ("rect", "srect", "circle")])
        rp = next(((p["x"], p["y"]) for p in P if p["g"] == "pill" and p["k"] == "rect"), None)
        pl = "—" if not (pill and rp) else f"{abs(pill[0] - rp[0]):.0f}·{abs(pill[1] - rp[1]):.3f}"
        f = lambda v: "실시간" if v is None else str(v)
        rows.append((i, f"{S['bars']:.4g}/{R['bars']:.4g} · tf {S['tfk']}/{R['tfk']} · 끝 {f(S['end'])}/{f(R['end'])} · 소수 {S['frac']:.3f}/{R['frac']:.3f} · 빈칸 {S['right']:.3f}/{R['right']:.3f} · 세로 {S['py']:.3f}/{R['py']:.3f}",
                     "같음" if st_ok else "다름", f"{j['nc']}/{r['nc']}봉 · {g['ymin']:.2f}~{g['ymax']:.2f} / {h['ymin']:.2f}~{h['ymax']:.2f}", "같음" if win_ok else "다름",
                     bg, fg, ta, pr, pl))
    return rows


def fmt(name, rows):
    out = [f"### {name} — {' → '.join(str(s[0]) + ('' if s[0] in ('tf', 'dbl') else '') + (str(s[1]) if s[0] == 'tf' else f'({s[3]:+d},{s[4]:+d})' if s[0] != 'dbl' else '') for s in SEQS[name])}", "",
           "| 단계 | 창 상태(지금/rtui) | 판정 | 보이는 봉·세로 범위(지금/rtui) | 판정 | bg 캔버스 | fg 캔버스(칸·캔들·델타·POC·스택·레인·꺾쇠) | 시간축 글자 | 프로파일 | 알약 x·y 차 |", "|---|---|---|---|---|---|---|---|---|---|"]
    c = lambda t: f"{t[0]} · {t[1]:.3f}px · 어긋남 {t[2]}" + (f" (첫 #{t[3][0]})" if t[3] else "")
    for i, s, a, w, b, bg, fg, ta, pr, pl in rows:
        out.append(f"| {i + 1} {SEQS[name][i][0]} | {s} | {a} | {w} | {b} | {c(bg)} | {c(fg)} | {c(ta)} | {c(pr)} | {pl} |")
    return out


if __name__ == "__main__":
    md = ["## 마우스 조작 일치 (같은 헤드리스 크로미움 · 1920×1080 · 같은 합성 자료 · 단계마다 창 상태·그린 목록 대조)", "",
          "좌표 = svg 상자 기준 px · 어긋남 = 종류·색·불투명·글자가 다르거나 좌표 차 0.01px 이상인 요소 수 · 알약 폭은 앞 조각의 글자 폭 차(4px)라 위치만", ""]
    for name in pick:
        res = run(name) if "cached" not in sys.argv else json.load(open(HERE / f"_ops_{name}.json"))
        rows = compare(name, res)
        md += fmt(name, rows) + [""]
        for i, *rest in rows:
            for t in rest[4:8]:
                if t[3]: print(name, i + 1, "첫 어긋남", t[3][0], json.dumps(t[3][1], ensure_ascii=False)[:220], "/", json.dumps(t[3][2], ensure_ascii=False)[:220])
        print("\n".join(fmt(name, rows)))
    (HERE / "ops.md").write_text("\n".join(md) + "\n")
