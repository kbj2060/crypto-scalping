"""웹어셈블리 측정: rtui/index.html?mode=bench 를 네트워크 없이 헤드리스 크로미움에 띄워 프레임당 가공 + egui 구성 + 테셀레이션(형성 봉 칸이 매 프레임 바뀜).
python -I rtui/compare/bench_web.py <worktree> <봉 수> [프레임=300]  (rtui/dist 먼저)"""
import pathlib, sys
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

WT, BARS = pathlib.Path(sys.argv[1]).resolve(), int(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
CT = {".html": "text/html", ".js": "application/javascript", ".wasm": "application/wasm", ".json": "application/json"}


def route(r):
    u = urlparse(r.request.url)
    f = WT / u.path.lstrip("/")
    if u.hostname != "fx.invalid" or not u.path.startswith("/rtui/") or not f.is_file():
        return r.abort()
    return r.fulfill(body=f.read_bytes(), content_type=CT.get(f.suffix, "application/octet-stream"))


def stats(v):
    s = sorted(v)
    return f"n={len(s)} med={s[len(s) // 2]:.3f} p99={s[min(len(s) - 1, int(0.99 * len(s)))]:.3f} max={s[-1]:.3f}"


with sync_playwright() as p:
    b = p.chromium.launch(args=["--host-resolver-rules=MAP * ~NOTFOUND", "--proxy-server=127.0.0.1:9"])
    try:
        pg = b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        pg.route("**/*", route)
        pg.on("pageerror", lambda e: print("pageerror", str(e)[:200]))
        pg.goto(f"http://fx.invalid/rtui/index.html?mode=bench&bars={BARS}&n={N}&w=1751&h=844&tz=-540")
        pg.wait_for_function("window.RESULT", timeout=120_000, polling=200)
        r = pg.evaluate("window.RESULT")
    finally:
        b.close()
v = r["v"]
bu, te = v[0::2], v[1::2]
print(f"wasm {BARS}봉 load {r['tInit']:.0f}ms")
print("build ", stats(bu)); print("tess  ", stats(te)); print("total ", stats([a + c for a, c in zip(bu, te)]))
