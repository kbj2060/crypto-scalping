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
P = rt["prims"]
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
rb = [dict(y=p["y"], x=p["x"], w=p["w"]) for p in P if p["k"] == "rect" and p["c"] == "Line"]
cmp("가로 격자 y(가격 눈금)", ja, rb, ["x", "y", "w"])
# 2 옆 캔들 몸통 · 심지 · 칸 · 불균형 막대 (fg 사각형을 폭으로 가른다)
fw = 3 if R["slot"] * 0.14 < 3 else min(8, R["slot"] * 0.14)
half = max(2, (R["bw"] - fw - 2) / 2 - 0.5)
fg = [r for r in rec if r["L"] == "fg" and r["k"] == "fillRect" and r["s"] in COL]
def by_w(lst, wv, key):
    return [dict(x=v[0], y=v[1], w=v[2], h=v[3]) for v in lst if abs(v[2] - wv) < 1e-3 and key(v)]
jf = [(r["a"][0], r["a"][1], r["a"][2], r["a"][3], r["al"]) for r in fg]
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
pt = next(t for t in js["texts"] if t["fs"] == "14")
ink_rows.append((f"알약 {pt['s']}", ink(A, pt["x"], pt["y"], 56, 12, "end"), ink(B, pt["x"], pt["y"], 56, 12, "end")))

out = [f"## 좌표 대조 ({BARS}봉 · 1920×1080 · svg 상자 ({bx},{by}) 기준 px)", "",
       f"기하: 지금 화면 ml {g['ml']} cw {g['cw']} mt {g['mt']} 플롯 바닥 {g['pb']} y {g['ymin']}~{g['ymax']} · rtui ml {R['ml']} cw {R['cw']} mt {R['mt']} 플롯 바닥 {R['pb']} y {R['ymin']}~{R['ymax']}", "",
       "| 요소 | 개수(지금/rtui) | 최대 차이(px) | 판정 |", "|---|---|---|---|"] + [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
out += ["", "### 그려진 글자 잉크 상자(PNG 에서 잼: 왼쪽 x · 폭 · 높이 · 위 y)", "", "| 글자 | 지금 화면 | rtui | 차이(x·폭·높이·y) |", "|---|---|---|---|"]
for nm, a, b in ink_rows:
    dd = " · ".join(f"{q - p:+d}" for p, q in zip(a, b)) if a and b else "?"
    out.append(f"| {nm} | {a} | {b} | {dd} |")
txt = "\n".join(out)
print(txt)
(HERE / f"coords_{BARS}.md").write_text(txt + "\n")
