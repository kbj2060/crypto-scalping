"""풋프린트 머리 작은 칸 자리 정답지: 진짜 index.html(스크립트·바깥 링크만 뺌) + 원본 styles.css 를 헤드리스 크로미움 1920×1080(html.fit1 · 다크)에 띄우고,
원본 renderSnapshotAssetTabs·placeTopLeft 를 그대로 돌린 뒤(코인 이름 = 맨 위 가운데 · 세션 = 맨 위 왼쪽 · 코인 탭 = 다크 단추 왼쪽) head_js.mjs 가 지은 HTML 을 제자리에 넣고 잰다 → fixture/head_js.json.
덩이 넷(#topSession · #fpCoinName · .session-alert-center · #fpPos)마다 그 상자 왼쪽 위 기준 글·칠·테 + 페이지 자리 · 경보 칸이 쓸 수 있는 폭(아주 넓은 칩을 넣어 잰다).
맥박(statusPulse)은 끈다(불투명 1 -- rtui 는 그릴 때 곱한다). 네트워크 없음. 실행: unshare -rn bash -c 'ip link set lo up; python3 -I rtui/compare/head_layout.py <worktree>'"""
import json, os, pathlib, re, sys
from playwright.sync_api import sync_playwright

WT = pathlib.Path(sys.argv[1]).resolve()
HERE = pathlib.Path(__file__).resolve().parent
FX, FONTS = WT / "rtui/compare/fixture", WT / "rtui/fonts"
sys.path.insert(0, str(HERE))
from harness import FONT_CSS, MONO_CSS, LAUNCH

src = (HERE / "mt_layout.py").read_text()
DUMP = src[src.index('DUMP = r"""') + 11:src.index('"""', src.index('DUMP = r"""') + 11)]   # Macro Trend 하네스와 같은 재기
html = json.loads(pathlib.Path(os.environ.get("HEAD_HTML", FX / "head_html.json")).read_text())
inp = json.loads((FX / "head_in.json").read_text())
css = pathlib.Path(os.environ.get("STYLES_CSS", WT / "dashboard/live/styles.css")).read_text()
idx = (WT / "dashboard/live/index.html").read_text()
idx = re.sub(r"<script\b.*?</script>", "", idx, flags=re.S)
idx = re.sub(r"<link\b[^>]*>", "", idx)
idx = idx.replace("<html", '<html class="fit1"', 1).replace("</head>", f"<style>{FONT_CSS}{MONO_CSS}</style><style>{css}</style><style>*{{animation:none!important;transition:none!important}}</style></head>", 1)
ROOTS = {"sess": "#topSession", "coin": "#fpCoinName", "alert": ".session-alert-center", "pos": "#fpPos"}
PAGE = """(sel) => { const r = document.querySelector(sel).getBoundingClientRect(); return [r.left, r.top, r.width, r.height].map((v) => Math.round(v * 1000) / 1000); }"""
out = []
with sync_playwright() as p:
    b = p.chromium.launch(args=LAUNCH)
    try:
        pg = b.new_page(viewport={"width": 1920, "height": 1080})
        def route(r):
            n = r.request.url.rsplit("/", 1)[-1]
            if "fx.invalid/font/" in r.request.url: return r.fulfill(body=(FONTS / n).read_bytes(), content_type="font/ttf")
            r.abort()
        pg.route("**/*", route)
        for s, h in zip(inp, html["cases"]):
            pg.set_content(idx)
            pg.add_script_tag(content=f"(() => {{ const activeSnapshotAsset = {json.dumps(s['asset'])}, enabledAssets = null, el = (id) => document.getElementById(id);\n{html['page']}\n"
                              "renderSnapshotAssetTabs(); placeTopLeft(); document.getElementById('cardRail').hidden = false; })();")   # 함수 안(set_content 는 같은 창 -- 맨 위 const 는 둘째 판에서 다시 선언 오류)
            pg.evaluate("""(h) => { const E = (id) => document.getElementById(id);
                E("topSession").innerHTML = h.sess; E("macroEventAlertBadge").style.display = h.badge ? "" : "none"; E("macroSpikeChips").innerHTML = h.spk;
                if (h.pos != null) { E("fpPos").innerHTML = h.pos; E("fpPos").hidden = false; } }""", h)
            pg.evaluate("document.fonts.ready")
            pg.evaluate("() => Promise.all([...document.fonts].map((f) => f.load()))")
            d = {k: pg.evaluate(DUMP, v) for k, v in ROOTS.items()}
            for k, v in d.items():   # 안 보이는 것 빼기: 투명 칠·테 · 경보 칸(overflow hidden · 32px)에서 둘째 줄로 넘어간 칩
                v["boxes"] = [x for x in v["boxes"] if x["c"][3] > 0]
                v["lines"] = [x for x in v["lines"] if x["c"][3] > 0]
                if k == "alert":   # 넘친 칩의 글 폭은 따로 둔다(rtui 가 칸 폭을 낼 때 쓴다)
                    v["hid"] = {"texts": [x for x in v["texts"] if x["r"][1] >= v["h"]]}
                    for t in ("texts", "boxes", "lines", "rings"): v[t] = [x for x in v[t] if x["r"][1] < v["h"]]
            page = {k: pg.evaluate(PAGE, v) for k, v in ROOTS.items()}
            # 경보 칸 폭 = flex 줄임(코인 탭과 나눠 줄어든다): 비었을 때 탭 폭(기준) · 아주 넓은 칩을 넣었을 때 칸 자리와 탭 폭(최소)
            pg.evaluate("""() => { const E = (id) => document.getElementById(id); E("macroEventAlertBadge").style.display = "none"; E("macroSpikeChips").innerHTML = ""; }""")
            tabs = pg.evaluate(PAGE, "#snapshotAssetTabs")
            pg.evaluate("""() => { document.getElementById("macroSpikeChips").innerHTML = '<span class="ops-badge good" style="width:3000px">x</span>'; }""")
            room, tmin = pg.evaluate(PAGE, ".session-alert-center"), pg.evaluate(PAGE, "#snapshotAssetTabs")
            out.append({**d, "page": page, "room": {"l0": room[0], "s": room[2] + tmin[2], "tb": tabs[2], "tmin": tmin[2]}, "top": pg.evaluate(PAGE, ".topbar"), "head": pg.evaluate(PAGE, "#fpCard > .ops-health-head")})
        probe = pg.evaluate("""(ps) => ps.map(([s, z, w, mono]) => { const e = document.createElement("span");
            e.style.cssText = `font: ${w} ${z}px ${mono ? '"JetBrains Mono"' : '"Pretendard Variable"'}; white-space: pre; position: absolute`; e.textContent = s; document.body.append(e);
            const r = e.getBoundingClientRect().width; e.remove(); return [s, z, w, mono, Math.round(r * 1000) / 1000]; })""", [[" ", 11, 700, True]])
    finally:
        b.close()
pathlib.Path(os.environ.get("HEAD_OUT", FX / "head_js.json")).write_text(json.dumps({"cases": out, "probe": probe}, ensure_ascii=False) + "\n")
for k, o in enumerate(out):
    print(k, " · ".join(f'{n} {o[n]["w"]}x{o[n]["h"]} 글 {len(o[n]["texts"])} 칠 {len(o[n]["boxes"])} 테 {len(o[n]["lines"])} 고리 {len(o[n]["rings"])} @ {o["page"][n][:2]}' for n in ROOTS), "· 칸", o["room"], "· 위", o["top"], "· 머리", o["head"])
