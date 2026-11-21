"""좌표 대조 표: 지금 화면이 실제로 낸 캔버스 명령·svg 글자(capture.py js 단계 기록) 대 rtui 그리기 목록(rtui 단계), 그리고 두 화면 PNG 에서 잰 글자 잉크 상자.
python -I rtui/compare/coords.py <bars>  →  표준출력에 마크다운 표 · coords_<bars>.md"""
import json, pathlib, sys
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
BARS = int(sys.argv[1]) if len(sys.argv) > 1 else 36
js, rt = json.load(open(HERE / f"_coords_js_{BARS}.json")), json.load(open(HERE / f"_coords_rtui_{BARS}.json"))
g, R = js["geo"], rt["geo"]
COL = {"#5dbf83": "Good", "#ef9387": "Bad", "#a6acc1": "Muted"}
rec = [r for r in js["rec"] if r["L"] in ("fg", "bg")]
P0 = rt["prims"]
P = [q for q in P0 if q["g"] in ("bg", "fg", "svg", "pill")]   # 첫 조각 요소(레인·프로파일·표식은 아래 따로)
rows = []


def cmp(name, a, b, keys):
    """a, b = 같은 순서의 목록(dict) -- 개수 · 칸별 최대 |차|"""
    if len(a) != len(b):
        rows.append((name, f"{len(a)} / {len(b)}", "개수 다름", ""))
        return
    m = {k: max((abs(x[k] - y[k]) for x, y in zip(a, b)), default=0.0) for k in keys}
    rows.append((name, f"{len(a)} / {len(b)}", " · ".join(f"{k} {v:.3f}" for k, v in m.items()), "같음" if max(m.values(), default=0) < 0.01 else "다름"))


# 1 격자(bg 1px 사각형)
ja = [dict(y=r["a"][1], x=r["a"][0], w=r["a"][2]) for r in rec if r["L"] == "bg" and r["k"] == "fillRect"]
rb = [dict(y=p["y"], x=p["x"], w=p["w"]) for p in P if p["k"] == "rect" and p["c"] == "Line" and p["g"] == "bg"]
cmp("가로 격자 y(가격 눈금)", ja, rb, ["x", "y", "w"])
# 2 옆 캔들 몸통 · 심지 · 칸 · 불균형 막대 (fg 사각형을 폭으로 가른다)
fw = 3 if R["slot"] * 0.14 < 3 else min(8, R["slot"] * 0.14)
half = max(2, (R["bw"] - fw - 2) / 2 - 0.5)
fg = [r for r in rec if r["L"] == "fg" and r["k"] == "fillRect" and r["s"] in COL]
def by_w(lst, wv, key):
    return [dict(x=v[0], y=v[1], w=v[2], h=v[3]) for v in lst if abs(v[2] - wv) < 1e-3 and key(v)]
jf = [(r["a"][0], r["a"][1], r["a"][2], r["a"][3], r["al"]) for r in fg]
# 몸통 폭 5px 이상이면 둥근 1px(roundRect 경로 = fill 기록) -- 12봉판
jf += [tuple(r["p"][0][1:5]) + (r["al"],) for r in rec if r["L"] == "fg" and r["k"] == "fill" and r["p"] and r["p"][0][0] == "roundRect" and r["p"][0][5] == 1 and abs(r["p"][0][3] - fw) < 1e-3 and r["s"] in COL]
rf = [(p["x"], p["y"], p["w"], p["h"], p["a"]) for p in P if p["k"] == "rect" and p["c"] in ("Good", "Bad", "Muted")]
for nm, wv, key in (("옆 캔들 몸통(x·y·높이)", fw, lambda v: v[4] == 1), ("심지(x·y·길이)", 1.0, lambda v: True),
                    ("칸 배경(매도·매수)", half, lambda v: True), ("불균형 막대", 2.0, lambda v: True)):
    cmp(nm, by_w(jf, wv, key), by_w(rf, wv, key), ["x", "y", "h"])
# 칸 농담(불투명) -- 같은 순서의 칸 배경끼리
ja = [dict(a=v[4]) for v in jf if abs(v[2] - half) < 1e-3]; rb = [dict(a=v[4]) for v in rf if abs(v[2] - half) < 1e-3]
cmp("칸 농담(4단계 불투명)", ja, rb, ["a"])
# 3 POC 테두리 · POC 선
js_sr = [r for r in rec if r["k"] == "stroke" and r["st"] == "#e99c57" and r["lw"] == 1]
rows.append(("POC 테두리", f"{len(js_sr)} / {sum(p['k'] == 'srect' for p in P)}", "strokeRect 는 경로라 좌표 기록 없음(개수만)", ""))
# 4 델타 글자(fg fillText)
ja = [dict(x=r["a"][1], y=r["a"][2], s=r["a"][0], fs=float(r["f"].split("px")[0].split()[-1])) for r in rec if r["k"] == "fillText"]
rb = [dict(x=p["x"], y=p["y"], s=p["s"], fs=p["size"]) for p in P if p["k"] == "text" and p["size"] < 13]
same_s = len(ja) == len(rb) and all(a["s"] == b["s"] for a, b in zip(ja, rb))
cmp("봉 델타 글자(x·바닥선 y·크기)", ja, rb, ["x", "y", "fs"]); rows[-1] = rows[-1][:3] + (rows[-1][3] + ("" if same_s else " · 글자 다름"),)
# 5 시간축(svg <text> 13px)
ja = [dict(x=t["x"], y=t["y"], fs=float(t["fs"]), s=t["s"]) for t in js["texts"] if t["fs"] == "13"]
rb = [dict(x=p["x"], y=p["y"], fs=p["size"], s=p["s"]) for p in P if p["k"] == "text" and p["size"] == 13]
cmp("시간축 글자(x·바닥선 y·크기)", ja, rb, ["x", "y", "fs"])
rows.append(("시간축 글자 내용", "", " ".join(t["s"] for t in ja) + " / " + " ".join(p["s"] for p in rb), "같음" if [t["s"] for t in ja] == [p["s"] for p in rb] else "다름"))
# 6 현재가 알약
pj = js["pill"]; pr = next(p for p in P if p["k"] == "rect" and p["c"] == "Ink")
cmp("현재가 알약 상자(x·y·폭)", [dict(x=pj["x"], y=pj["y"], w=pj["w"])], [dict(x=pr["x"], y=pr["y"], w=pr["w"])], ["x", "y", "w"])

# ── 둘째 조각: 레인(fg 캔버스) · 스택 · 미완 경매(fg 캔버스) · 프로파일 기둥·레인 글자(svg) ──
CN = {"#5dbf83": "Good", "#ef9387": "Bad", "#a6acc1": "Muted", "#171b23": "ChartBg", "rgba(203, 209, 227, 0.18)": "Line", "#5aa9f5": "Turnover",
      "#eef0f6": "Text", "#e99c57": "Amber", "#dfe4ee": "Accent", "#f5f6fb": "Ink",
      "var(--good)": "Good", "var(--bad)": "Bad", "var(--amber)": "Amber", "var(--accent)": "Accent", "var(--ink)": "Ink", "var(--muted)": "Muted", "var(--soft-line)": "SoftLine"}


def jnorm(r):
    """캔버스 기록 하나 → rect(x,y,w,h,색,불투명,rx) | line(점들,색,굵기)"""
    if r["k"] == "fillRect":
        x, y, w, h = r["a"]; return dict(k="rect", x=x, y=y, w=w, h=h, c=CN.get(r["s"], r["s"]), a=r["al"], rx=0)
    if r["k"] == "fill" and r["p"] and r["p"][0][0] == "roundRect":
        _, x, y, w, h, rx = r["p"][0]; return dict(k="rect", x=x, y=y, w=w, h=h, c=CN.get(r["s"], r["s"]), a=r["al"], rx=rx)
    if r["k"] == "stroke":
        return dict(k="line", pts=[q[1:3] for q in r["p"] if q[0] in ("moveTo", "lineTo")], c=CN.get(r["st"], r["st"]), a=r["al"], lw=r["lw"])
    if r["k"] == "fill" and r["p"] and r["p"][0][0] == "arc":
        _, x, y, rr = r["p"][0][:4]; return dict(k="circle", x=x, y=y, r=rr, c=CN.get(r["s"], r["s"]), a=r["al"])


def rnorm(p):
    if p["k"] == "rect":
        return dict(k="rect", x=p["x"], y=p["y"], w=p["w"], h=p["h"], c=p["c"], a=p["a"], rx=min(p["rx"], p["w"] / 2, p["h"] / 2))
    if p["k"] == "line":
        return dict(k="line", pts=p["pts"], c=p["c"], a=p["a"], lw=p["lw"])
    return dict(p)


def cmp_items(name, a, b):
    """같은 순서 그리기 목록 -- 개수 · 종류·색·불투명 어긋남 수 · 좌표 최대 |차|"""
    if len(a) != len(b):
        rows.append((name, f"{len(a)} / {len(b)}", "개수 다름", "다름")); return
    dx, bad = 0.0, 0
    for x, y in zip(a, b):
        if x["k"] != y["k"] or x.get("c") != y.get("c") or abs(x.get("a", 1) - y.get("a", 1)) > 0.002: bad += 1
        if x["k"] == "line" and y["k"] == "line":
            if len(x["pts"]) != len(y["pts"]): bad += 1; continue
            dx = max([dx] + [abs(u - v) for p, q in zip(x["pts"], y["pts"]) for u, v in zip(p, q)] + [abs(x["lw"] - y["lw"])])
        else:
            dx = max([dx] + [abs(x[k] - y[k]) for k in ("x", "y", "w", "h", "rx", "r") if k in x and k in y])
    rows.append((name, f"{len(a)} / {len(b)}", f"좌표 {dx:.3f} · 종류·색·불투명 어긋남 {bad}", "같음" if dx < 0.01 and not bad else "다름"))


fgr = [r for r in rec if r["L"] == "fg"]
i0 = next(n for n, r in enumerate(fgr) if r["k"] == "fillRect" and r["s"] == "#171b23")
lane_js = []
for r in fgr[i0:]:
    q = jnorm(r)
    if q is None or q["k"] == "circle" or (q["k"] == "rect" and q["y"] < R["quad_y"] - 9) or (q["k"] == "line" and q["c"] != "Turnover"): break   # 레인 뒤 = 청산 동그라미(11280)
    lane_js.append(q)
G = lambda g: [rnorm(p) for p in P0 if p["g"] == g]
cmp_items("레인: 바탕·경계선·리본·0선·델타 캔들(꼬리·몸통)·거래대금 선", lane_js, G("lane"))
cmp_items("스택 불균형 막대", [q for q in map(jnorm, fgr) if q and q["k"] == "rect" and q["w"] == 2 and q["rx"] == 1], G("stack"))
cmp_items("청산 동그라미(봉 위 · 다섯째 조각)", [q for q in map(jnorm, fgr) if q and q["k"] == "circle"], G("liq"))
cmp_items("미완 경매 꺾쇠(︿﹀)", [q for q in map(jnorm, fgr) if q and q["k"] == "line" and q["c"] == "Text"], G("ua"))

# 프로파일(svg 요소): 칠한 사각형 → rect · 테두리 → srect(같은 요소면 rect 다음) · 점 · 0선 · 머리 글자
px0 = R["ml"] + R["cw"]
prof_js = []
for e in js["els"]:
    a = e["a"]
    if e["t"] == "rect" and "data-pill" not in a and float(a["x"]) >= px0:
        f = lambda k: float(a[k])
        if a.get("fill") != "none": prof_js.append(dict(k="rect", x=f("x"), y=f("y"), w=f("width"), h=f("height"), c=CN.get(a["fill"], a["fill"]), a=float(a.get("fill-opacity", 1)), rx=0))
        if "stroke" in a: prof_js.append(dict(k="srect", x=f("x"), y=f("y"), w=f("width"), h=f("height"), c=CN.get(a["stroke"]), a=float(a.get("stroke-opacity", 1))))
    elif e["t"] == "circle":
        prof_js.append(dict(k="circle", x=float(a["cx"]), y=float(a["cy"]), r=float(a["r"]), c=CN.get(a["fill"]), a=1.0))
    elif e["t"] == "line" and float(a["x1"]) >= px0 and a.get("stroke") == "var(--soft-line)" and float(a["x1"]) > px0 + 66:   # 델타 0선(청산 기둥 경계선은 범위 밖)
        prof_js.append(dict(k="line", pts=[[float(a["x1"]), float(a["y1"])], [float(a["x2"]), float(a["y2"])]], c="SoftLine", a=1.0, lw=1.0))
    elif e["t"] == "text" and a.get("font-size") == "9":
        prof_js.append(dict(k="text", x=float(a["x"]), y=float(a["y"]), c="Muted", a=1.0, s=e["s"]))
prof_rt = [dict(k=p["k"], x=p["x"], y=p["y"], w=p["w"], h=p["h"], c=p["c"], a=p["a"]) if p["k"] == "srect" else
           dict(k="text", x=p["x"], y=p["y"], c=p["c"], a=1.0, s=p["s"]) if p["k"] == "text" else rnorm(p) for p in P0 if p["g"] == "profile"]
cmp_items("프로파일: 가치 영역·체결·델타·TPO 막대·테두리·한 번 지난 점·0선·머리 글자", prof_js, prof_rt)
rows.append(("프로파일 머리 글자 내용", "", " ".join(q["s"] for q in prof_js if q["k"] == "text") + " / " + " ".join(q["s"] for q in prof_rt if q["k"] == "text"),
             "같음" if [q["s"] for q in prof_js if q["k"] == "text"] == [q["s"] for q in prof_rt if q["k"] == "text"] else "다름"))
lg_js = next(e for e in js["els"] if e["t"] == "text" and e["a"].get("font-size") == "10" and e["s"].startswith("▮"))
lg_rt = [p for p in P0 if p["g"] == "lane_text" and p["s"].startswith("▮")]
rn_js = [dict(x=float(e["a"]["x"]), y=float(e["a"]["y"]), s=e["s"]) for e in js["els"] if e["t"] == "text" and e["a"].get("font-size") == "12" and e["s"] in ("매수 우위", "매도 우위")]
rn_rt = [dict(x=p["x"], y=p["y"], s=p["s"]) for p in P0 if p["g"] == "lane_text" and p["s"] in ("매수 우위", "매도 우위")]
cmp("레인 리본 이름(칸 52px 이상)", rn_js, rn_rt, ["x", "y"])
if rn_js and len(rn_js) == len(rn_rt) and [a["s"] for a in rn_js] != [b["s"] for b in rn_rt]: rows[-1] = rows[-1][:3] + ("글자 다름",)
cmp("레인 범례 글자(x·바닥선 y)", [dict(x=float(lg_js["a"]["x"]), y=float(lg_js["a"]["y"]))], [dict(x=lg_rt[0]["x"], y=lg_rt[0]["y"])], ["x", "y"])

# 7 그려진 픽셀에서 잰 글자 잉크 상자(화면 px) -- 글자 렌더링(글꼴 래스터·굵기·자간) 차이
A, B = Image.open(HERE / f"_js_full_{BARS}.png").convert("L"), Image.open(HERE / f"_rtui_full_{BARS}.png").convert("L")
bx, by = g["box"]["x"], g["box"]["y"]
def ink(im, cx, base, wd, ht, anchor, dark=False):
    x0 = int(bx + cx - (wd if anchor == "end" else wd / 2 if anchor == "middle" else 0)) - 3
    win = im.crop((x0, int(by + base - ht) - 2, x0 + int(wd) + 6, int(by + base) + 4))
    px = list(win.get_flattened_data()); W = win.size[0]
    bgv = sorted(px)[len(px) // 2] if not dark else max(px)
    on = [i for i, v in enumerate(px) if (abs(v - bgv) > 45)]
    if not on: return None
    xs = [i % W for i in on]; ys = [i // W for i in on]
    return (x0 + min(xs) - bx, max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, int(by + base - ht) - 2 + min(ys) - by)
ink_rows = []
for t in [t for t in js["texts"] if t["fs"] == "13"]:
    a, b = ink(A, t["x"], t["y"], 44, 14, "middle"), ink(B, t["x"], t["y"], 44, 14, "middle")
    ink_rows.append((f"시간 {t['s']}", a, b))
for d in ja[:0]: pass
for r in [r for r in rec if r["k"] == "fillText"][:6]:
    a, b = ink(A, r["a"][1], r["a"][2], 30, 9, "middle"), ink(B, r["a"][1], r["a"][2], 30, 9, "middle")
    ink_rows.append((f"델타 {r['a'][0]}", a, b))
for e in [e for e in js["els"] if e["t"] == "text" and e["a"].get("font-size") == "9" and e["s"] != "청산"]:
    a = e["a"]; ink_rows.append((f"기둥 머리 {e['s']}", ink(A, float(a["x"]), float(a["y"]), e["tl"], 9, "start"), ink(B, float(a["x"]), float(a["y"]), e["tl"], 9, "start")))
ink_rows.append(("레인 범례(두 조각)", ink(A, float(lg_js["a"]["x"]), float(lg_js["a"]["y"]), lg_js["tl"] + 4, 10, "start"), ink(B, float(lg_js["a"]["x"]), float(lg_js["a"]["y"]), lg_js["tl"] + 4, 10, "start")))
pt = next(t for t in js["texts"] if t["fs"] == "14")
ink_rows.append((f"알약 {pt['s']}", ink(A, pt["x"], pt["y"], 56, 12, "end"), ink(B, pt["x"], pt["y"], 56, 12, "end")))

out = [f"## 좌표 대조 ({BARS}봉 · 1920×1080 · svg 상자 ({bx},{by}) 기준 px)", "",
       f"기하: 지금 화면 ml {g['ml']} cw {g['cw']} mt {g['mt']} 플롯 바닥 {g['pb']} y {g['ymin']}~{g['ymax']} · rtui ml {R['ml']} cw {R['cw']} mt {R['mt']} 플롯 바닥 {R['pb']} y {R['ymin']}~{R['ymax']}", "",
       "| 요소 | 개수(지금/rtui) | 최대 차이(px) | 판정 |", "|---|---|---|---|"] + [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
out += ["", "### 그려진 글자 잉크 상자(PNG 에서 잼: 왼쪽 x · 폭 · 높이 · 위 y)", "", "| 글자 | 지금 화면 | rtui | 차이(x·폭·높이·y) |", "|---|---|---|---|"]
for nm, a, b in ink_rows:
    dd = " · ".join(f"{q - p:+d}" for p, q in zip(a, b)) if a and b else "?"
    out.append(f"| {nm} | {a} | {b} | {dd} |")
# 호버(capture.py hover 단계): svg 상자 기준 -- 십자선·가격 배지 · 툴팁 상자·줄
hp = HERE / f"_hover_{BARS}.json"
if hp.exists():
    h = json.load(open(hp)); jb, rb = h["jbox"], h["rbox"]
    out += ["", f"### 호버 {len(h['pts'])}점(svg 좌표에 같은 점 · 상자 기준 px · side_hover_{BARS}_<n>.png)", "",
            "| 점 | 십자선 가로 y · 세로 x(지금/rtui) | 가격 배지 x·y·폭(지금/rtui) | 툴팁 x·y·폭·높이(지금/rtui) | 툴팁 글자 |", "|---|---|---|---|---|"]
    for n, (a_, b_) in enumerate(zip(h["js"], h["rt"])):
        it, rp = a_["items"], (b_ or {}).get("prims", [])
        jh = next((float(i["a"]["y1"]) for i in it if i["t"] == "line" and i["a"]["y1"] == i["a"]["y2"]), None)
        jv = next((float(i["a"]["x1"]) for i in it if i["t"] == "line" and i["a"]["x1"] == i["a"]["x2"]), None)
        rh = next((q["pts"][0][1] for q in rp if q["k"] == "line" and q["pts"][0][1] == q["pts"][1][1]), None)
        rv = next((q["pts"][0][0] for q in rp if q["k"] == "line" and q["pts"][0][0] == q["pts"][1][0]), None)
        jr = next((i["a"] for i in it if i["t"] == "rect"), None); rr = next((q for q in rp if q["k"] == "rect"), None)
        f = lambda v: "—" if v is None else f"{v:.2f}"
        badge = f"{jr['x']}·{jr['y']}·{jr['width']} / {rr['x']:.2f}·{rr['y']:.0f}·{rr['w']:.2f}" if jr and rr else "—"
        tj, tr = a_["tip"], (b_ or {}).get("tip")
        tip = (f"{tj['box']['x'] - jb['x']:.0f}·{tj['box']['y'] - jb['y']:.0f}·{tj['box']['width']:.2f}·{tj['box']['height']:.0f} / {tr['x'] - rb['x']:.0f}·{tr['y'] - rb['y']:.0f}·{tr['w']:.2f}·{tr['h']:.0f}") if tj and tr else ("없음 / 없음" if not tj and not tr else "한쪽만")
        same = "—" if not (tj and tr) else ("같음" if [l for l in tj["text"].split("\n") if l.strip()] == [l for l in tr["lines"] if l.strip()] else "다름")
        out.append(f"| {n} | {f(jh)}·{f(jv)} / {f(rh)}·{f(rv)} | {badge} | {tip} | {same} |")
txt = "\n".join(out)
print(txt)
(HERE / f"coords_{BARS}.md").write_text(txt + "\n")
