"""Offline: serve dashboard/live from disk + /api/* from fixture copies, no network at all.
Measures renderCandleSvg (via renderSnapshotChart, same span as M0 fp) per rAF frame with the forming bar mutated each frame,
and counts canvas/SVG primitives per render. usage: python -I js_fp_bench.py <worktree> <fixture_dir> <frames> [shot.png]"""
import json, re, sys, pathlib
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

WT, FX, N = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
WB = int(sys.argv[4]) if len(sys.argv) > 4 else 0
SHOT = sys.argv[5] if len(sys.argv) > 5 else None
LIVE = WT / "dashboard/live"
BLOCK = {"binance-account", "acct-notes", "acct-wrong", "manual-entry_status"}   # account values stay out of the page
API = {}
for line in (FX / "index.tsv").read_text().splitlines():
    name, url = line.split("\t")[:2]
    if name not in BLOCK:
        API[urlparse(url).path] = FX / f"{name}.json"
T_FX = 1795156790   # fixture capture time (sse "now"), page clock is shifted there
SSE = (FX / "sse_stream.txt").read_text()

INIT = r"""(() => {
  const OFF = %d * 1000 - Date.now(); const RD = Date;
  class D extends RD { constructor(...a) { if (a.length) super(...a); else super(RD.now() + OFF); } static now() { return RD.now() + OFF; } }
  window.Date = D;
  window.WebSocket = class { constructor() { this.readyState = 3; } send() {} close() {} addEventListener() {} };
  const SSE = %s;
  window.EventSource = class { constructor(u) { this.url = u; this.readyState = 1; this.l = {};
      if (String(u).includes('/api/stream')) setTimeout(() => { for (const blk of SSE.split('\n\n')) { const ev = (blk.match(/^event: (.*)$/m) || [])[1] || 'message', d = (blk.match(/^data: (.*)$/m) || [])[1]; if (d) this._f(ev, d); } }, 300); }
    _f(ev, d) { const e = new MessageEvent(ev, { data: d }); (this.l[ev] || []).forEach((f) => f(e)); if (ev === 'message' && this.onmessage) this.onmessage(e); }
    addEventListener(t, f) { (this.l[t] = this.l[t] || []).push(f); } removeEventListener() {} close() { this.readyState = 2; } };
  // primitive counter
  const C = window.__cnt = {}; if (%s) return; const P = CanvasRenderingContext2D.prototype;
  for (const k of ['fillText', 'fillRect', 'fill', 'stroke', 'roundRect', 'clearRect', 'rect', 'lineTo', 'moveTo', 'measureText', 'arc']) { const o = P[k]; P[k] = function (...a) { C[k] = (C[k] || 0) + 1; if (k === 'fillText') C.chars = (C.chars || 0) + String(a[0]).length; if (k === 'fillRect' || k === 'fillText') (C['args_' + k] = C['args_' + k] || []).push(a.map((v) => typeof v === 'number' ? Math.round(v * 10) / 10 : v)); return o.apply(this, a); }; }
  const Q = Path2D.prototype; for (const k of ['rect', 'arc', 'lineTo', 'moveTo', 'roundRect', 'ellipse']) { const o = Q[k]; Q[k] = function (...a) { C['path2d.' + k] = (C['path2d.' + k] || 0) + 1; return o.apply(this, a); }; }
})();""" % (T_FX, json.dumps(SSE), "true" if __import__("os").environ.get("NOCNT") else "false")


def handle(route):
    req = route.request
    u = urlparse(req.url)
    if req.method != "GET" or u.hostname != "fx.invalid":
        return route.abort()
    if u.path.startswith("/dashboard/live/"):
        f = LIVE / u.path[len("/dashboard/live/"):]
        if u.path.endswith("/"):
            f = LIVE / "index.html"
        if f.is_file() and f.name != "sw.js":
            ct = {".js": "application/javascript", ".css": "text/css", ".html": "text/html", ".png": "image/png"}.get(f.suffix, "application/octet-stream")
            return route.fulfill(body=f.read_bytes(), content_type=ct)
        return route.fulfill(status=404, body="")
    if u.path in API:
        return route.fulfill(body=API[u.path].read_bytes(), content_type="application/json")
    return route.fulfill(status=404, body="{}", content_type="application/json")


BENCH = r"""async ([N, WB]) => {
  if (WB) { chartWindowBars = WB; renderSnapshotChart(); }
  const fp = latestFootprint; const bars = fp && (fp.bars || []);
  const last = bars[bars.length - 1];
  const svg = document.getElementById('candleSvgSnapshot');
  const out = [], cnt = [];
  for (let f = 0; f < N; f++) {
    await new Promise((r) => requestAnimationFrame(r));
    // forming bar: every cell's buy/sell nudged, like trades arriving
    if (last) last.levels.forEach((l, j) => { l[1] += 0.37 + ((f + j) % 5) * 0.11; l[2] += 0.29 + ((f * 3 + j) % 7) * 0.07; });
    Object.keys(__cnt).forEach((k) => delete __cnt[k]);
    m0.fp.length = 0;
    renderSnapshotChart();
    out.push(m0.fp.length ? m0.fp[0] : NaN);   // exactly the M0 span: one renderCandleSvg call
    if (f === N - 1) cnt.push({ ...__cnt });
  }
  const tags = {}; svg.querySelectorAll('*').forEach((e) => { tags[e.tagName] = (tags[e.tagName] || 0) + 1; });
  const polyPts = [...svg.querySelectorAll('polyline,polygon')].reduce((a, e) => a + (e.getAttribute('points') || '').trim().split(/\s+/).length, 0);
  const pathCmds = [...svg.querySelectorAll('path')].reduce((a, e) => a + ((e.getAttribute('d') || '').match(/[MLHVCQAZ]/gi) || []).length, 0);
  const els = svg.querySelectorAll('*').length, texts = svg.querySelectorAll('text').length;
  const tchars = [...svg.querySelectorAll('text')].reduce((a, t) => a + t.textContent.length, 0);
  return { ms: out, cnt: cnt[0], tags, polyPts, pathCmds, svgEls: els, svgTexts: texts, svgTextChars: tchars, nBarsFp: bars.length,
           candles: chartTfFull().length, windowBars: chartWindowBars, boxesOff: fpBoxesOff,
           box: svg.getBoundingClientRect().toJSON(), dpr: devicePixelRatio, ua: navigator.userAgent };
}"""

with sync_playwright() as p:
    b = p.chromium.launch(args=["--host-resolver-rules=MAP * ~NOTFOUND", "--proxy-server=127.0.0.1:9", "--disable-background-networking"])
    ctx = b.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1, service_workers="block")
    ctx.route("**/*", handle)
    ctx.add_init_script(INIT)
    pg = ctx.new_page()
    pg.on("console", lambda m: m.type == "error" and print("console.error", m.text[:200]))
    pg.goto("http://fx.invalid/dashboard/live/", wait_until="domcontentloaded")
    pg.wait_for_timeout(8000)
    print("fp m0 samples during load", pg.evaluate("m0.fp.length"), "flowOn", pg.evaluate("flowOn()"))
    cdp = ctx.new_cdp_session(pg) if __import__('os').environ.get('PROF') else None
    if cdp:
        cdp.send("Profiler.enable"); cdp.send("Profiler.setSamplingInterval", {"interval": 100}); cdp.send("Profiler.start")
    r = pg.evaluate(BENCH, [N, WB])
    if cdp:
        prof = cdp.send("Profiler.stop")["profile"]
        import collections
        dt = collections.Counter(); byid = {n["id"]: n for n in prof["nodes"]}
        par = {c: n["id"] for n in prof["nodes"] for c in n.get("children", [])}
        def inside(i):   # sample under renderCandleSvg?
            while i in byid:
                if byid[i]["callFrame"]["functionName"] == "renderCandleSvg": return True
                i = par.get(i)
            return False
        ts = prof["timeDeltas"]
        for sid, d in zip(prof["samples"], ts[1:] + [0]):
            n = byid[sid]; cf = n["callFrame"]
            if not inside(sid): continue
            cat = "app.js 로직" if cf["url"].endswith(("app.js", "app.js?v=20261120-m0")) or "app.js" in cf["url"] else ("GC" if "garbage" in cf["functionName"] else "브라우저 API: " + cf["functionName"])
            dt[cat] += d
        tot = sum(dt.values())
        top = collections.Counter(); 
        for k, v in dt.items(): top["브라우저 API(DOM·캔버스·스타일)" if k.startswith("브라우저") else k] += v
        print("profile share inside renderCandleSvg:", {k: f"{100*v/tot:.0f}%" for k, v in top.most_common()})
        print("top browser APIs:", [(k[8:], f"{100*v/tot:.1f}%") for k, v in dt.most_common(40) if k.startswith("브라우저")][:12])
    if SHOT:
        pg.screenshot(path=SHOT)
    ms = sorted(r.pop("ms"))
    q = lambda a: ms[min(len(ms) - 1, int(a * len(ms)))]
    import collections
    fr = r["cnt"].pop("args_fillRect", []) if r["cnt"] else []; ft = r["cnt"].pop("args_fillText", []) if r["cnt"] else []
    print("fillRect (w,h) top:", collections.Counter((a[2], a[3]) for a in fr).most_common(8))
    print("fillText sample:", ft[:12], "n", len(ft))
    print(json.dumps(r, ensure_ascii=False))
    print(f"JS renderSnapshotChart n={len(ms)} med={q(0.5):.2f} p99={q(0.99):.2f} max={ms[-1]:.2f} ms")
    b.close()
