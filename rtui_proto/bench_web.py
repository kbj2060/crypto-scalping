"""웹어셈블리 측정: dist/ 를 네트워크 없이 헤드리스 크로미움에 띄워 프레임당 egui UI 구성 + 테셀레이션을 잰다.
python -I bench_web.py <bench|app> <0|1> <프레임> [churn] [shot.png]   (먼저 wasm-bindgen 으로 dist/kfont·dist/nofont 생성)"""
import gzip, json, pathlib, sys
from urllib.parse import urlparse, parse_qs
from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
mode, sc, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
churn = "1" if "churn" in sys.argv[4:] else "0"
shot = next((a for a in sys.argv[4:] if a.endswith(".png")), None)
font = "nofont" if "nofont" in sys.argv[4:] else "kfont"
CT = {".html": "text/html", ".js": "application/javascript", ".wasm": "application/wasm"}


def handle(route):
    u = urlparse(route.request.url)
    f = HERE / "index.html" if u.path in ("/", "/index.html") else HERE / "dist" / u.path.lstrip("/")
    if u.hostname != "fx.invalid" or not f.is_file():
        return route.abort()
    return route.fulfill(body=f.read_bytes(), content_type=CT.get(f.suffix, "application/octet-stream"))


def stats(v):
    s = sorted(v)
    return f"n={len(s)} med={s[len(s) // 2]:.3f} p99={s[min(len(s) - 1, int(0.99 * len(s)))]:.3f} max={s[-1]:.3f}"


with sync_playwright() as p:
    b = p.chromium.launch(args=["--host-resolver-rules=MAP * ~NOTFOUND", "--proxy-server=127.0.0.1:9"])
    pg = b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    pg.route("**/*", handle)
    pg.on("console", lambda m: m.type in ("error", "warning") and print("console", m.text[:200]))
    pg.goto(f"http://fx.invalid/index.html?font={font}&mode={mode}&sc={sc}&n={n}&churn={churn}")
    pg.wait_for_function("window.RESULT", timeout=600_000, polling=200)
    r = pg.evaluate("window.RESULT")
    if shot:
        pg.screenshot(path=shot)
    print("ua", pg.evaluate("navigator.userAgent"), "webgl", pg.evaluate("(() => { const g = document.createElement('canvas').getContext('webgl2'); const e = g && g.getExtension('WEBGL_debug_renderer_info'); return g ? g.getParameter(e ? e.UNMASKED_RENDERER_WEBGL : g.RENDERER) : null; })()"))
    print(f"wasm load+instantiate {r['tInit']:.1f} ms ({font}, local route, no network)")
    if mode == "bench":
        fr = r["frames"]
        print("first frame build/tess ms", [round(x, 2) for x in r["first"][:2]], "counts", fr[-1][3:], "verts", fr[-1][2])
        print("build ", stats([x[0] for x in fr]))
        print("tess  ", stats([x[1] for x in fr]))
        print("total ", stats([x[0] + x[1] for x in fr]))
    else:
        print(f"start() resolved {r['tStart']:.1f} ms · first frame on screen {r['nav']:.1f} ms after navigation")
        print("eframe cpu_usage(web, ui+tess+WebGL submit)", stats(r["cpu"]))
    b.close()
for d in ("kfont", "nofont"):
    w = HERE / "dist" / d / "rtui_proto_bg.wasm"
    if w.is_file():
        print(f"size {d}: wasm {w.stat().st_size} B · gzip -9 {len(gzip.compress(w.read_bytes(), 9))} B · js {(w.parent / 'rtui_proto.js').stat().st_size} B")
