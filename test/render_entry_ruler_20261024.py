#!/usr/bin/env python3
"""«지금 들어가면» 가격 자(2026-10-24 사용자 B 시안 · 체결 버블 미니 자리)를 실제 화면에 그려 확인한다.

    python3 -u test/render_entry_ruler_20261024.py [--shot DIR]

서버 페이지(터널 127.0.0.1:18787) 위에 로컬 app.js·styles.css·bubble_map.js 를 덮는다(render_box_hold_20261016.py 와 같은 방식).
🔴GET 이 아닌 요청(주문 포함)은 브라우저 안에서 전부 막는다 -- 막힌 요청이 있으면 실패.
검사(2133·1920 넓은 화면): #mcEntry 높이 = MC_EG_H + 12 · 손절 표시 둘 · 내용이 칸을 안 넘침 · 손절 손실 = 상한 × (5%/3 + 2/3 × 4.25%/0.9925)
· 체결 버블 미니 캔버스(#mcBubble) 없음 · «체결 버블» 보기는 그대로 그려진다 · JS 오류 0.
"""
import argparse, asyncio, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"
LOSS_K = 0.05 / 3 + (2 / 3) * (0.0425 / 0.9925)   # 첫 1/3 이 5% · 물타기 2/3 이 (5% − 0.75%)/(1 − 0.75%) 를 잃는다

PROBE = """() => {
  const e = document.getElementById('mcEntry'); if (!e) return null;
  const d = egData(), t = [...e.querySelectorAll('svg text')].map((x) => x.textContent);
  return { h: e.offsetHeight, ch: e.clientHeight, sh: e.scrollHeight, H: MC_EG_H, head: e.querySelector('.eg-head b').textContent, stops: t.filter((x) => /손절 [−+]5%/.test(x)).length,
           k: d.cap ? d.lossStop / d.cap : null, mini: !!document.getElementById('mcBubble') };
}"""


async def main(shot):
    from playwright.async_api import async_playwright
    fails = []
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for name, vp in (("2133", dict(width=2133, height=1200)), ("1920", dict(width=1920, height=1000))):
            pg = await b.new_page(viewport=vp)
            errs, blocked = [], []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            serve = lambda f, ct: (lambda r: r.fulfill(path=str(DASH / f), content_type=ct))   # 🔴인자 1개 -- 2개면 Playwright 가 둘째에 Request 를 넘긴다
            for f, ct in (("app.js", "text/javascript"), ("styles.css", "text/css"), ("bubble_map.js", "text/javascript")):
                await pg.route(f"**/dashboard/live/{f}*", serve(f, ct))
            async def guard(r):
                if r.request.method != "GET": blocked.append(r.request.url); await r.abort()
                else: await r.continue_()
            await pg.route("**/api/**", guard)
            await pg.goto(URL, wait_until="domcontentloaded")
            await pg.wait_for_timeout(14000)
            got = await pg.evaluate(PROBE)
            print(name, json.dumps(got, ensure_ascii=False))
            if not got: fails.append(f"{name}: #mcEntry 없음")
            else:
                if got["h"] != got["H"] + 12: fails.append(f"{name}: 높이 {got['h']} != {got['H'] + 12}")
                if got["sh"] > got["ch"] + 1: fails.append(f"{name}: 내용 넘침 {got['sh']} > {got['h']}")
                if got["head"] != "계좌·가격 불러오는 중" and got["stops"] != 2: fails.append(f"{name}: 손절 표시 {got['stops']}")
                if got["k"] is not None and abs(got["k"] - LOSS_K) > 1e-9: fails.append(f"{name}: 손절 손실 비율 {got['k']} != {LOSS_K}")
                if got["mini"]: fails.append(f"{name}: 체결 버블 미니가 남아 있다")
            if shot:
                await pg.screenshot(path=str(pathlib.Path(shot) / f"entry_ruler_{name}.png"), full_page=False)
            await pg.click("#fpViewTabs [data-view='bubble']")   # 큰 체결 버블 보기는 그대로
            await pg.wait_for_timeout(5000)
            cv = await pg.evaluate("() => { const c = document.getElementById('bubbleMapCanvas'); return c ? [c.clientWidth, c.clientHeight, c.width] : null; }")
            print(name, "bubble canvas", cv)
            if not cv or min(cv) <= 0: fails.append(f"{name}: 체결 버블 보기 캔버스가 비었다 {cv}")
            if errs: fails.append(f"{name}: JS 오류 {errs[:3]}")
            if blocked: fails.append(f"{name}: GET 아닌 요청 {blocked[:3]}")
            await pg.close()
        await b.close()
    print("FAIL" if fails else "PASS", *fails, sep="\n")
    return not fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shot")
    sys.exit(0 if asyncio.run(main(ap.parse_args().shot)) else 1)
