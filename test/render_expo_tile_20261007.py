#!/usr/bin/env python3
"""계좌 노출 타일 기준(4.1배 초록 · 6배 넘으면 빨강 · 레일 눈금)을 실제로 그려서 확인한다 (2026-10-07).

    python3 -u test/render_expo_tile_20261007.py [--shot DIR]

render_vol_mult_20261007.py 와 같은 방식(서버 터널 127.0.0.1:18787 위에 로컬 app.js·styles.css·index.html).
🔴주문 제출(`/api/manual-*/submit`)은 predicate 로 막는다 -- 막힌 요청이 있으면 실패. 버튼은 누르지 않는다.
가짜 ETHUSDC 롱 포지션(순자산 $2,000) × 노출 3·5·8·20배 × 폭 1500·390.
"""
import argparse, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
DASH = HERE.parent / "dashboard" / "live"
URL = "http://127.0.0.1:18787/dashboard/live/"
SETUP = """(x) => {
  for (let i = 1; i < 99999; i++) window.clearInterval(i);
  activeSnapshotAsset = "eth";
  const eq = 2000, px = 2500, notional = x * eq;
  latestBinanceAccount = Object.assign({}, latestBinanceAccount || {}, {
    balance: { wallet: eq, unrealized: 0, available: eq * 0.5, initial_margin: notional / 20 },
    positions: [{ symbol: "ETHUSDC", side: "LONG", qty: String(notional / px), entry_price: px, mark_price: px,
                  notional, leverage: 20, liquidation_price: px * (1 - 1 / x), unrealized: 0 }] });
  renderSnapshotAccount();
}"""
TILE = """() => { const t = document.querySelector('#snapAcctPosition .acct-tile[data-pv="expo"]'); if (!t) return null;
  const v = t.querySelector('.acct-tile-val'), i = t.querySelector('.acct-rail i'), r = t.querySelector('.acct-rail');
  const pct = (pe) => { const s = getComputedStyle(r, pe); return [s.left, s.content]; };
  const rw = r.getBoundingClientRect().width;
  return { text: v.textContent.trim(), cls: v.className, fill: i.getBoundingClientRect().width / rw,
           before: pct('::before'), after: pct('::after'), rw, title: t.getAttribute('title') || '',
           box: t.getBoundingClientRect().toJSON() }; }"""
CASES = ((3.0, "good"), (5.0, "warn"), (8.0, "bad"), (20.0, "bad"))


def run(shot):
    from playwright.sync_api import sync_playwright
    js, css, html = ((DASH / f).read_text("utf-8") for f in ("app.js", "styles.css", "index.html"))
    fails = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        for w, h in ((1500, 900), (390, 844)):
            pg = b.new_page(viewport={"width": w, "height": h})
            errs, blocked = [], []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.route(lambda u: "/api/manual-" in u and "/submit" in u, lambda r: (blocked.append(r.request.url), r.abort()))
            pg.route("**/app.js*", lambda r: r.fulfill(body=js, content_type="application/javascript"))
            pg.route("**/styles.css*", lambda r: r.fulfill(body=css, content_type="text/css"))
            pg.route(URL, lambda r: r.fulfill(body=html, content_type="text/html"))
            pg.goto(URL, wait_until="load", timeout=60000)
            pg.wait_for_timeout(5000)
            for x, tone in CASES:
                tag = f"{x}배·{w}"
                pg.evaluate(SETUP, x)
                pg.wait_for_timeout(250)
                t = pg.evaluate(TILE)
                if not t:
                    fails.append(f"[{tag}] 노출 타일 없음"); continue
                if t["text"] != f"{x:.1f}배":
                    fails.append(f"[{tag}] 글자 {t['text']!r}")
                if tone not in t["cls"].split():
                    fails.append(f"[{tag}] 색 {t['cls']} ≠ {tone}")
                want = min(1.0, x / 12)
                if abs(t["fill"] - want) > 0.02:
                    fails.append(f"[{tag}] 막대 {t['fill']:.3f} ≠ {want:.3f}")
                for nm, (left, content), frac in (("4.1 눈금", t["before"], 4.1 / 12), ("6 눈금", t["after"], 0.5)):
                    if content in ("none", "normal") or abs(float(left.rstrip("px")) - frac * t["rw"]) > 1.0:
                        fails.append(f"[{tag}] {nm} {left} {content} (폭 {t['rw']:.0f})")
                if "4.1배" not in t["title"] or "막지 않습니다" not in t["title"]:
                    fails.append(f"[{tag}] 툴팁 근거 없음")
                if t["box"]["x"] < 0 or t["box"]["x"] + t["box"]["width"] > w + 0.5:
                    fails.append(f"[{tag}] 화면 밖 {t['box']}")
                if shot:
                    pathlib.Path(shot).mkdir(parents=True, exist_ok=True)
                    pg.locator("#snapAcctPosition .acct-tiles").screenshot(path=f"{shot}/expo_{x:g}_{w}.png")
            if errs:
                fails.append(f"[{w}] JS 오류 {errs[:3]}")
            if blocked:
                fails.append(f"[{w}] 막힌 제출 {blocked}")
            pg.close()
        b.close()
    print("\n".join(fails) if fails else "OK 계좌 노출 4상태 × 2폭")
    return not fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shot")
    sys.exit(0 if run(ap.parse_args().shot) else 1)
