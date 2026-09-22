#!/usr/bin/env python3
"""차트 요소가 **가격 플롯 밖으로 새지 않는지** 실제 렌더로 잰다 (2026-09-23).

왜 있는가: 2026-09-23 추세 밴드가 플롯 바닥(y=1032)을 넘어 **y=1202 까지** 내려와
  사분면 막대(1038~1170)를 통째로 덮었다. `yAt` 은 경계를 모르고, 레인 좌표는 **정상**이었다.
  🔴높이 계약 테스트는 통과했다 -- 예산은 맞았고 «그린 것»이 예산을 넘은 것이라 잡을 수 없다.
  🔴소스를 읽어서도 안 보인다. SMA144 가 보이는 가격 범위에서 멀어져야만 나타난다.
  => 계산된 박스에서만 보이므로 **렌더해서 재는 수밖에 없다**.

로컬 바이트를 서버 페이지 위에 덮어 라우팅하므로 **배포 전에** 검증된다
(test/render_dashboard_layout_check_20260922.py 와 같은 방식).

  python3 -u test/render_chart_lane_overlap_20260923.py
  python3 -u test/render_chart_lane_overlap_20260923.py --selftest   # 검사기가 결함을 잡는지 증명
"""
import argparse, asyncio, pathlib, sys

DASH = pathlib.Path(__file__).resolve().parent.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"

# 가격 플롯 안에만 있어야 하는 것들. 식별은 **그리는 쪽 규약**으로 한다(class 를 새로 안 만든다).
SCAN = r"""() => {
  const svg = document.querySelector('#candleSvgSnapshot');
  if (!svg) return { error: 'svg 없음' };
  const m = (svg.getAttribute('viewBox') || '').split(/\s+/).map(Number);
  const out = { vbH: m[3] || 0, items: [] };
  for (const e of svg.querySelectorAll('polygon,polyline,path,rect,line')) {
    let bb; try { bb = e.getBBox(); } catch (_) { continue; }
    if (!isFinite(bb.y) || bb.height < 0) continue;
    const fill = e.getAttribute('fill') || '', stroke = e.getAttribute('stroke') || '';
    // 🔴SVG 의 tagName 은 **소문자**다(HTML 과 다르다). 대문자로 비교하면 조용히 0건이 된다.
    const tn = e.tagName.toLowerCase();
    let kind = null;
    if (fill.includes('--muted) 14%')) kind = '추세밴드 면';
    else if (tn === 'polyline' && stroke.includes('--good')) kind = '추세밴드 상단';
    else if (tn === 'polyline' && stroke.includes('--bad')) kind = '추세밴드 하단';
    else if (tn === 'rect' && e.getAttribute('rx') === '3') kind = '사분면 막대';
    if (!kind) continue;
    // 🔴`getBBox()` 는 **클리핑 전** 기하다(클리핑은 렌더 효과). 그래서 «넘었나»만으로는
    //   판정할 수 없다 -- 클립 조상이 있는지와 그 클립 사각형 범위를 같이 본다.
    const holder = e.closest('[clip-path]');
    let clip = null;
    if (holder) {
      const id = (holder.getAttribute('clip-path') || '').replace(/^url\(#|\)$/g, '');
      const r = svg.querySelector('clipPath#' + id + ' rect');
      if (r) clip = { y: +r.getAttribute('y'), h: +r.getAttribute('height') };
    }
    out.items.push({ kind, y0: bb.y, y1: bb.y + bb.height, clip });
  }
  return out;
}"""


async def run(selftest: bool) -> int:
    from playwright.async_api import async_playwright
    js = (DASH / "app.js").read_text("utf-8")
    if selftest:
        # 검사기 증명: clip 을 떼면 반드시 잡혀야 한다(안 잡히면 이 테스트가 죽은 것이다).
        js = js.replace('gClip.setAttribute("clip-path", `url(#${clipId})`);', "")
    css = (DASH / "styles.css").read_text("utf-8")
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await (await b.new_context(viewport={"width": 1600, "height": 1000})).new_page()
        await pg.route("**/app.js*", lambda r: r.fulfill(body=js, content_type="application/javascript"))
        await pg.route("**/styles.css*", lambda r: r.fulfill(body=css, content_type="text/css"))
        await pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
        try:
            await pg.wait_for_selector("#candleSvgSnapshot polygon", timeout=45000)
        except Exception:
            pass
        await pg.wait_for_timeout(9000)
        r = await pg.evaluate(SCAN)
        await b.close()

    if r.get("error"):
        print("🔴", r["error"]); return 1
    items = r["items"]
    if not items:
        print("🔴 잴 요소가 없다 -- 차트가 안 그려졌거나 규약이 바뀌었다(검사기를 고쳐야 한다)")
        return 1
    # 가격 플롯 = [mt, mt+ch]. mt/ch 는 JS 안에 있으므로 **사분면 막대의 위 끝**으로 바닥을 잡는다
    # (quadY = plotBottom + LANE_GAP 라 그 위는 전부 플롯이어야 한다).
    quads = [i for i in items if i["kind"] == "사분면 막대"]
    if not quads:
        print("🔴 사분면 막대를 못 찾았다 -- 바닥 기준을 못 잡는다"); return 1
    bottom = min(q["y0"] for q in quads)
    # 판정: 바닥을 넘는 그림은 **클립되어 있어야** 하고, 그 클립 사각형도 바닥 안이어야 한다.
    def leaks(i):
        if i["kind"] == "사분면 막대" or i["y1"] <= bottom + 0.5:
            return None
        if not i["clip"]:
            return "클립 없음"
        if i["clip"]["y"] + i["clip"]["h"] > bottom + 0.5:
            return f"클립 사각형이 바닥을 넘음({i['clip']['y'] + i['clip']['h']:.0f})"
        return None
    bad = [i for i in items if leaks(i)]
    print(f"플롯 바닥(사분면 막대 위 끝) y = {bottom:.0f} · viewBox 높이 {r['vbH']:.0f}")
    for i in items:
        if i["kind"] == "사분면 막대":
            continue
        why = leaks(i)
        over = i["y1"] - bottom
        note = ("" if over <= 0.5 else
                (f" · 기하는 {over:.0f}px 넘지만 클립됨(y≤{i['clip']['y']+i['clip']['h']:.0f})"
                 if not why else f"   ← 바닥을 {over:.0f}px 넘어 아래 레인을 덮는다 [{why}]"))
        print(f"  {'ok ' if not why else '🔴 '}{i['kind']:<12} y {i['y0']:7.0f} ~ {i['y1']:7.0f}{note}")
    if selftest:
        print("\n[selftest] clip 을 떼고 돌렸다 -- 결함이 **잡혀야** 정상이다")
        return 0 if bad else 1
    return 1 if bad else 0


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--selftest", action="store_true")
    sys.exit(asyncio.run(run(a.parse_args().selftest)))
