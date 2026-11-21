"""알약 숫자 폭 탐침: 같은 글꼴(Pretendard Bold 한 벌)로 svg <text> 14px 를 굵기·tnum 별로 재고 글자별 시작 x 를 낸다(오프라인).
python -I rtui/compare/probe_tnum.py <worktree>"""
import pathlib, sys
from playwright.sync_api import sync_playwright
FONT = pathlib.Path(sys.argv[1]).resolve() / "rtui/fonts/Pretendard-Bold.ttf"
HTML = """<style>@font-face{font-family:'Pretendard Variable';src:url(http://fx.invalid/p.ttf) format('truetype');font-weight:45 920;}</style>
<svg width=600 height=200 style="font-family:'Pretendard Variable'">%s</svg>"""
CASES = [(w, t) for w in ("700", "800") for t in ("", "font-variant-numeric: tabular-nums")]
with sync_playwright() as p:
    b = p.chromium.launch(args=["--host-resolver-rules=MAP * ~NOTFOUND", "--proxy-server=127.0.0.1:9"] + sys.argv[2:])
    try:
        pg = b.new_page()
        pg.route("**/*", lambda r: r.fulfill(body=FONT.read_bytes(), content_type="font/ttf") if r.request.url.endswith("p.ttf") else
                 r.fulfill(body=HTML % "".join(f'<text x="10" y="{30 + 30 * i}" font-size="14" font-weight="{w}" style="{t}">1777.5</text>' for i, (w, t) in enumerate(CASES)), content_type="text/html"))
        pg.goto("http://fx.invalid/")
        pg.wait_for_timeout(800)
        for i, c in enumerate(pg.evaluate("() => [...document.querySelectorAll('text')].map((t) => [t.getComputedTextLength(), [...Array(t.getNumberOfChars()).keys()].map((k) => +t.getStartPositionOfChar(k).x.toFixed(3))])")):
            print(CASES[i], c)
    finally:
        b.close()
