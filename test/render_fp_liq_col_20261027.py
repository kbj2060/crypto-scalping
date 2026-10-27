"""청산 기둥 실데이터 검증 -- 화면이 그린 막대·툴팁을 브라우저 밖(파이썬)에서 지도 원본으로 다시 계산해 대조한다.
  (1) fpLiqColTp(화면 입력) vs 서버 원본 API: 칸 격자 동일 · 다른 칸은 전부 0(지나간 칸 지우기)이고 현재가를 품은 한 덩어리 안
  (2) 줄마다 툴팁 값(총·배수별·누적) vs 파이썬 재계산(칸 k 가운데 = k×bw, 줄과 겹친 길이로 나눔)
  (3) 막대 픽셀 폭 비 vs 값 비 · 쌓는 순서(바닥선부터 10→25→50→100배) · 툴팁이 가리킨 줄 = 그 막대 줄
  (4) 생존 규칙: 현재가 위 줄은 «숏», 아래는 «롱» 이름"""
import re, sys, pathlib
from playwright.sync_api import sync_playwright
# 실행: python test/render_fp_liq_col_20261027.py -- 공개 대시보드(thesan.xyz)에 이 작업 사본의 app.js·liq_profile.js·styles.css·index.html 변경만 끼운다.
#   쓰기 요청(POST 등)은 전부 막는다(실주문 경로가 같은 화면에 있다). 그림은 /tmp 에.
W = str(pathlib.Path(__file__).resolve().parents[1] / "dashboard/live") + "/"
SP = "/tmp"

LOCAL = {"app.js": "application/javascript", "liq_profile.js": "application/javascript", "styles.css": "text/css"}
def handle(route):
    req = route.request
    if req.method != "GET":
        return route.abort()                       # 주문 등 쓰기 요청 전부 차단
    path = re.sub(r"\?.*$", "", req.url)
    name = path.rsplit("/", 1)[-1]
    if "/dashboard/live/" in path and name in LOCAL:
        return route.fulfill(body=open(W + name, encoding="utf-8").read(), content_type=LOCAL[name])
    if req.resource_type == "document" and path.rstrip("/").endswith("/dashboard/live"):
        r = route.fetch(); html = r.text()
        html = re.sub(r'\s*<button type="button" class="asset-tab" data-view="liq"[^>]*>청산맵</button>', "", html)
        return route.fulfill(response=r, body=html)
    return route.continue_()

def money(s):
    m = re.fullmatch(r"\$([\d.]+)([BMk]?)", s.strip())
    return float(m.group(1)) * {"B": 1e9, "M": 1e6, "k": 1e3, "": 1}[m.group(2)]
def tol(v):   # 화면 표기 반올림 허용(fmtUsdCompact: M 2자리 · k 1자리 · B 2자리)
    return 0.006e9 if v >= 1e9 else 0.0051e6 if v >= 1e6 else 51 if v >= 1e3 else 0.51

JS = r"""(d) => {
  const svg = [...document.querySelectorAll('svg')].find((s) => s.onmousemove);
  const tp = window.fpLiqColTp(d), price = Number(latestLivePriceByAsset[activeSnapshotAsset]);
  const byY = new Map();
  [...svg.querySelectorAll('rect')].filter((r) => (r.getAttribute('fill') || '').startsWith('var(--lev')).forEach((r) => {
    const k = r.getAttribute('y'), a = byY.get(k) || [];
    a.push({ x: +r.getAttribute('x'), w: +r.getAttribute('width'), h: +r.getAttribute('height'), fill: r.getAttribute('fill'), bb: r.getBoundingClientRect().toJSON() });
    byY.set(k, a);
  });
  const tipEl = document.getElementById('chartTooltip'), groups = [];
  byY.forEach((rs, y) => {
    const r0 = rs.reduce((a, b) => (a.x > b.x ? a : b)), cx = r0.bb.left + r0.bb.width / 2, cy = r0.bb.top + r0.bb.height / 2;
    tipEl.classList.remove('visible'); tipEl.innerHTML = '';
    svg.onmousemove({ clientX: cx, clientY: cy, pageX: cx + scrollX, pageY: cy + scrollY });
    groups.push({ y: +y, rects: rs.map(({ x, w, h, fill }) => ({ x, w, h, fill })), tip: tipEl.classList.contains('visible') ? tipEl.innerText : '' });
  });
  const lr = svg.querySelector('rect[fill^="var(--lev"]'), G = lr && lr.parentNode;
  const base = G ? [...G.querySelectorAll('line')].pop() : null;
  const lines = G ? [...G.querySelectorAll('polyline')].map((l) => ({ col: l.getAttribute('stroke'), pts: l.getAttribute('points').split(' ').map((q) => q.split(',').map(Number)) })) : [];
  const texts = G ? [...G.querySelectorAll('text')].map((t) => t.textContent) : [];
  const src = d === 1 && latestLiquidationMap ? latestLiquidationMap.tier_profile : null;   // 화면이 실제로 쓰는 1일 지도(같은 순간)
  return { tp, src, price, groups, lines, texts, mt: base ? +base.getAttribute('y1') : null, pb: base ? +base.getAttribute('y2') : null,
           price2: Number(latestLivePriceByAsset[activeSnapshotAsset]) };
}"""

fails = []
def check(c, msg):
    if not c: fails.append(msg)

with sync_playwright() as p:
    b = p.chromium.launch()
    for d, pan in ((1, 0), (7, 0), (30, 0), (1, 0.02), (1, -0.02)):
        pg = b.new_page(viewport={"width": 1600, "height": 1100}, color_scheme="dark")
        pg.add_init_script(f"localStorage.setItem('fpCol','liq'); localStorage.setItem('liqDays','{d}');")
        pg.route("**/*", handle)
        errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto("https://thesan.xyz/dashboard/live/", wait_until="domcontentloaded")
        pg.wait_for_timeout(16000)
        if pan:
            pg.evaluate("(f) => { chartPanY = f * Number(latestLivePriceByAsset[activeSnapshotAsset]); scheduleSnapshotChartRender(); }", pan)
            pg.wait_for_timeout(2500)
        R = pg.evaluate(JS, d)
        tag = f"{d}d" + (f" 끌기{pan:+}" if pan else "")
        for l in R["lines"]:
            ys = [y for _, y in l["pts"]]
            check(R["mt"] - 0.5 <= min(ys) and max(ys) <= R["pb"] + 0.5, f"{tag} 누적선이 플롯 밖 {min(ys):.0f}..{max(ys):.0f} vs {R['mt']}..{R['pb']}")
        cols = sorted(l["col"] for l in R["lines"])
        if pan > 0:   # 현재가가 화면 아래 밖 = 보이는 건 전부 숏 쪽
            check(cols == ["var(--good)"], f"{tag} 선 색 {cols}")
            check(any(t.startswith("누적 롱 청산 ↓ $0") for t in R["texts"]), f"{tag} 롱 라벨 {R['texts']}")
            check(all(g["tip"].startswith("숏") for g in R["groups"] if g["tip"]), f"{tag} 롱 줄이 보임")
        if pan < 0:
            check(cols == ["var(--bad)"], f"{tag} 선 색 {cols}")
            check(any(t.startswith("누적 숏 청산 ↑ $0") for t in R["texts"]), f"{tag} 숏 라벨 {R['texts']}")
            check(all(g["tip"].startswith("롱") for g in R["groups"] if g["tip"]), f"{tag} 숏 줄이 보임")
        print(f"{tag}: 선 {cols} · 라벨 {[t for t in R['texts'] if '누적' in t]}")
        raw = pg.evaluate("(d) => fetch(d === 1 ? '/api/liquidation-map?asset=eth' : '/api/liquidation-map/tiers?asset=eth&days=' + d).then((r) => r.json())", d)["tier_profile"]
        pg.locator("#fpCard").screenshot(path=f"{SP}/fp_liq_col_{d}d.png")
        pg.close()
        tp, cp = R["tp"], R["price"]
        if R["src"]: raw = R["src"]
        check(not errs, f"{d}d 페이지 오류 {errs}")
        check(R["price"] == R["price2"], f"{d}d 측정 중 현재가 바뀜(재시도 필요)")
        bw, lo0, n = tp["bin_width"], tp["lo"], len(tp["tiers"][0]["values"])
        # (1) 화면 입력 vs 서버 원본 -- 1d 는 60초 갱신 사이 서버가 새 지도를 냈을 수 있어 격자가 같을 때만
        if (raw["bin_width"], raw["lo"], len(raw["tiers"][0]["values"])) == (bw, lo0, n):
            diff = [j for j in range(n) if any(t["values"][j] != r["values"][j] for t, r in zip(tp["tiers"], raw["tiers"]))]
            check(all(all(t["values"][j] == 0 for t in tp["tiers"]) for j in diff), f"{d}d 원본과 다른 칸이 0 이 아님")
            if diff:
                c = [(lo0 + j) * bw for j in diff]
                check(diff == list(range(diff[0], diff[-1] + 1)), f"{d}d 지운 칸이 한 덩어리가 아님")
                check(min(c) - bw <= cp <= max(c) + bw, f"{d}d 지운 칸 덩어리가 현재가를 안 품음 {min(c)}..{max(c)} cp {cp}")
            print(f"{d}d 원본 대조: 칸 {n} · 지운 칸 {len(diff)}" + (f" ({(lo0+diff[0])*bw:.1f}~{(lo0+diff[-1])*bw:.1f}, 현재가 {cp:.1f})" if diff else ""))
        else:
            print(f"{d}d 원본 대조 건너뜀: 그 사이 새 지도(격자 다름)")
        names = [t["name"] for t in tp["tiers"]]
        def row(lo, hi):   # 독립 재계산: 칸 j = [(lo0+j-.5)bw, (lo0+j+.5)bw), 줄과 겹친 몫
            out = [0.0] * len(names)
            for ti, t in enumerate(tp["tiers"]):
                for j, v in enumerate(t["values"]):
                    a, z = (lo0 + j - 0.5) * bw, (lo0 + j + 0.5) * bw
                    ov = min(z, hi) - max(a, lo)
                    if v > 0 and ov > 0: out[ti] += v * ov / bw
            return out
        def cum(px):
            s = 0.0
            for j in range(n):
                c = (lo0 + j) * bw; v = sum(max(0, t["values"][j]) for t in tp["tiers"])
                if (cp < c <= px) if px > cp else (px <= c < cp): s += v
            return s
        rows, nt = [], 0
        for gI in R["groups"]:
            m = re.match(r"(롱|숏) 청산 (\$[\d.]+[BMk]?) · ([\d.]+)–([\d.]+) \(현재가 (위|아래)", gI["tip"])
            if not m: check(False, f"{d}d 툴팁 못 읽음 y={gI['y']} '{gI['tip'][:60]}'"); continue
            side, tot, lo, hi, ud = m.group(1), money(m.group(2)), float(m.group(3)), float(m.group(4)), m.group(5)
            exp = row(lo, hi)
            check(abs(tot - sum(exp)) <= tol(sum(exp)), f"{d}d {lo}–{hi} 총 {tot} vs 재계산 {sum(exp):.0f}")
            check((side == "숏") == (ud == "위") == ((lo + hi) / 2 > cp), f"{d}d {lo}–{hi} 롱/숏 이름 틀림")
            tiers = re.findall(r"(\S+)배 (\$[\d.]+[BMk]?)", gI["tip"].split("\n")[1])
            check([t for t, _ in tiers] == names, f"{d}d 배수 이름 {tiers}")
            for (nm, v), e in zip(tiers, exp):
                check(abs(money(v) - e) <= tol(e), f"{d}d {lo}–{hi} {nm}배 {v} vs {e:.0f}")
            cm = re.search(r"누적 (\$[\d.]+[BMk]?)", gI["tip"]); ce = cum(hi if (lo + hi) / 2 > cp else lo)
            check(abs(money(cm.group(1)) - ce) <= tol(ce), f"{d}d {lo}–{hi} 누적 {cm.group(1)} vs {ce:.0f}")
            # (3) 쌓는 순서·폭: 바닥선(오른쪽)부터 10→25→50→100배, 폭 ∝ 값
            rs = sorted(gI["rects"], key=lambda r: -r["x"])
            order = [int(re.search(r"--lev(\d+)", r["fill"]).group(1)) for r in rs]
            check(order == sorted(order), f"{d}d {lo}–{hi} 쌓는 순서 {order}")
            nz = [e for e in exp if e > 0]
            if len(nz) == len(rs) and sum(nz) > 0:
                Ws = sum(r["w"] for r in rs)
                for r, e in zip(rs, nz): check(abs(r["w"] / Ws - e / sum(nz)) < 0.01, f"{d}d {lo}–{hi} 배수 폭 비")
            rows.append((sum(r["w"] for r in rs), sum(exp))); nt += 1
        if not rows:   # 화면 가격 범위 안에 청산 칸이 없을 수 있다(좁은 범위 · 1d) -- 끌어 둔 사례가 막대를 대조한다
            print(f"{d}d 화면 안 청산 줄 0 -- 막대 대조 건너뜀"); continue
        mxw, mxe = max(rows), max(rows, key=lambda x: x[1])
        check(mxw[1] == mxe[1], f"{d}d 가장 긴 막대 ≠ 가장 큰 값")
        bad = [(w, e) for w, e in rows if abs(w / mxw[0] - e / mxe[1]) > 0.01]
        check(not bad, f"{d}d 막대 길이 비 어긋남 {bad[:3]}")
        print(f"   줄 높이 {sorted(set(round(r['h'],1) for g in R['groups'] for r in g['rects']))[:5]} · 줄 수 {len(R['groups'])}")
        print(f"{d}d 줄 {nt}개 대조 · 현재가 {cp:.2f} · 단위 {tp['unit']}")
    b.close()
print("FAIL" if fails else "PASS", *fails[:20], sep="\n")
sys.exit(1 if fails else 0)
