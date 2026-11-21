"""실시간 조건 측정: 가짜 @trade 를 초당 수백 건 넣고(페이지 안에서 소켓 onmessage 를 직접 부른다 -- 바이낸스 접속 없음) 지금 화면과 rtui 를 같은 헤드리스 크로미움에서 잰다.
  · 프레임: requestAnimationFrame 콜백 시간(한 프레임 = 같은 타임스탬프의 콜백 합) -- 지금 화면은 페이지 전체(다른 카드 포함)와 풋프린트 몫(renderSnapshotChart · 형성 봉 부분 칠 fpCv.live)을 따로
  · 체결 도착 → 반영: 아직 안 그린 첫 체결의 도착 시각 → 그 체결을 담은 그리기가 끝난 시각(지금 화면 = fpCv.live 또는 renderSnapshotChart 끝 · rtui = egui ui() 끝)
  · 유휴: 체결 없이 6초 -- 프레임 수 · 풋프린트 그리기 수
시계는 형성 봉(1795156500) 열리기 12초 전에 시작 → 소켓이 봉 열리기 전부터 붙어 있어 형성 봉 칸이 브라우저 값(footprintMergeLive)으로 그려진다.
python -I rtui/compare/live_bench.py <worktree> [초당=300] [초=8]  →  표준출력"""
import json, pathlib, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from harness import INIT, LAUNCH, TZ, T_FX, make_route

WT = pathlib.Path(sys.argv[1]).resolve()
RATE = int(sys.argv[2]) if len(sys.argv) > 2 else 300
SECS = float(sys.argv[3]) if len(sys.argv) > 3 else 8
FX = WT / "rtui/compare/fixture"
route = make_route(WT, {"/api/footprint": FX / "footprint.json", "/api/market-history": FX / "market-history.json"})
B = 1795156500
INIT_B = INIT.replace(f"const OFF = {T_FX} * 1000", f"const OFF = {B - 12} * 1000")
HOOK = r"""(() => {
  const rq = window.requestAnimationFrame.bind(window);
  window.__fr = []; window.__cur = null; window.__fw = {}; window.__out = []; window.__ends = []; window.__arrs = []; window.__nfp = 0;
  window.requestAnimationFrame = (cb) => rq((ts) => { window.__cur = ts; const t0 = performance.now(); try { cb(ts); } finally { window.__fr.push([ts, performance.now() - t0]); window.__cur = null; } });
  window.__work = (dt) => { if (window.__cur == null) window.__out.push(dt); else window.__fw[window.__cur] = (window.__fw[window.__cur] || 0) + dt; window.__nfp++; };
  window.__mark = () => { window.__ends.push(performance.now()); };
})();"""
JS_WRAP = r"""() => {
  const orig = renderSnapshotChart;
  window.renderSnapshotChart = function () { const t0 = performance.now(); const r = orig.apply(this, arguments); window.__work(performance.now() - t0); window.__mark(); return r; };
  let L = fpCv.live;
  Object.defineProperty(fpCv, 'live', { configurable: true, get: () => L, set: (v) => { L = v && (() => { const t0 = performance.now(); const r = v(); window.__work(performance.now() - t0); if (r !== false) window.__mark(); return r; }); } });
}"""
INJECT = r"""([rate, secs, rt]) => new Promise((res) => {
  const ws = () => window.__wsAll[window.__wsAll.length - 1];
  const t0 = performance.now(); let n = 0, tid = 5_000_000, px = 1745.0, seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const iv = setInterval(() => {
    const due = Math.floor(((performance.now() - t0) / 1000) * rate);
    while (n < due) {
      n++; if (rnd() < 0.5) px = Math.round((px + (rnd() - 0.5) * 0.6) * 100) / 100;
      const T = Date.now(), msg = JSON.stringify({ e: 'trade', E: T, T, s: 'ETHUSDT', t: tid++, p: px.toFixed(2), q: (rnd() * 2).toFixed(3), m: rnd() < 0.5 });
      window.__arrs.push(t0 + (n * 1000) / rate);   // 정해진 도착 시각(메인 스레드가 막혀 늦게 처리된 몫까지 지연에 든다 -- 실제 소켓 메시지도 그동안 줄을 선다)
      ws().onmessage({ data: msg });
    }
    if (performance.now() - t0 > secs * 1000) { clearInterval(iv); res(n); }
  }, 4);
})"""


def q(v, a):
    s = sorted(v)
    return s[min(len(s) - 1, int(a * len(s)))] if s else float("nan")


def frames(fr):
    by = {}
    for ts, d in fr: by[ts] = by.get(ts, 0) + d
    return list(by.values())


def run(js):
    with sync_playwright() as p:
        br = p.chromium.launch(args=LAUNCH)
        try:
            ctx = br.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1, service_workers="block", timezone_id=TZ, locale="ko-KR")
            ctx.route("**/*", route); ctx.add_init_script(INIT_B); ctx.add_init_script(HOOK)
            pg = ctx.new_page()
            if js:
                pg.goto("http://fx.invalid/dashboard/live/", wait_until="domcontentloaded")
                pg.wait_for_function("typeof fpCv === 'object' && window.__wsAll && window.__wsAll.length > 0 && latestFootprint", timeout=30_000)
                pg.wait_for_timeout(1500); pg.evaluate(JS_WRAP)
            else:
                pg.goto("http://fx.invalid/rtui/index.html?mode=shot&x=113&y=131&w=1751&h=844&tz=-540&bg=1e2229&wk=0")
                pg.wait_for_function("window.RESULT", timeout=30_000, polling=100)
            # 봉이 열리기 전 체결 하나 = 소켓이 그 봉 전부터 붙어 있었다(since < 봉 시작 → 형성 봉 = 브라우저 값)
            pre = pg.evaluate("() => { const T = Date.now(); window.__wsAll[window.__wsAll.length - 1].onmessage({ data: JSON.stringify({ e: 'trade', E: T, T, t: 4999999, p: '1745.00', q: '0.1', m: false }) }); return T; }")
            assert pre < B * 1000, "페이지가 늦게 떴다(시계가 봉 시작을 넘김)"
            pg.wait_for_function(f"Date.now() > {B * 1000 + 800}", timeout=30_000, polling=100)   # 형성 봉이 열린 뒤부터
            pg.evaluate("() => { window.__fr = []; window.__fw = {}; window.__out = []; window.__ends = []; window.__arrs = []; window.__nfp = 0; }")
            if not js: pg.evaluate("() => window.rtui.take_lat()")
            n = pg.evaluate(INJECT, [RATE, SECS, not js])
            pg.wait_for_timeout(300)
            r = pg.evaluate("() => ({ fr: window.__fr, fw: Object.values(window.__fw), out: window.__out, ends: window.__ends, arrs: window.__arrs, nfp: window.__nfp })")
            if not js:
                v = pg.evaluate("() => [...window.rtui.take_lat()]"); k = next(i for i, x in enumerate(v) if x != x)
                r["ends"], r["ui"] = v[:k], v[k + 1:]
            # 체결마다: 도착 → 그 뒤 처음 끝난 그리기(지금 화면 = renderSnapshotChart · fpCv.live · rtui = ui())
            import bisect
            e = sorted(r["ends"]); r["lat"] = [e[i] - a for a in r["arrs"] if (i := bisect.bisect_left(e, a)) < len(e)]
            # 유휴 6초(체결 없음 · 묻기는 그대로)
            pg.evaluate("() => { window.__fr = []; window.__nfp = 0; }")
            pg.wait_for_timeout(6000)
            idle = pg.evaluate("() => ({ frames: new Set(window.__fr.map((x) => x[0])).size, cb: window.__fr.length, nfp: window.__nfp, busy: window.__fr.reduce((a, x) => a + x[1], 0) })")
            if not js:
                v = pg.evaluate("() => [...window.rtui.take_lat()]"); idle["ui"] = len(v) - 1 - next(i for i, x in enumerate(v) if x != x)
        finally:
            br.close()
    return n, r, idle


if __name__ == "__main__":
    out = {}
    for nm, js in (("지금 화면", True), ("rtui", False)):
        n, r, idle = run(js)
        f = frames(r["fr"])
        ts = sorted({x[0] for x in r["fr"]}); gaps = [b - a for a, b in zip(ts, ts[1:])]
        row = {"체결": n, "프레임 수": len(f), "프레임(rAF 합) 중앙": q(f, .5), "상위 1%": q(f, .99), "도착→반영 중앙": q(r["lat"], .5), "도착→반영 상위 1%": q(r["lat"], .99), "그리기 횟수": len(r["ends"]), "rAF 간격 중앙": q(gaps, .5), "rAF 간격 상위 1%": q(gaps, .99),
               "유휴 6초 rAF 콜백": idle["cb"], "유휴 6초 콜백 시간 합(ms)": idle["busy"]}
        if js:
            row |= {"풋프린트 몫/프레임 중앙": q(r["fw"], .5), "풋프린트 몫 상위 1%": q(r["fw"], .99), "프레임 밖(체결 중 동기 전체 그리기) 횟수": len(r["out"]), "그 중앙": q(r["out"], .5), "유휴 6초 풋프린트 그리기": idle["nfp"]}
        else:
            row |= {"egui ui() 횟수": len(r["ui"]), "egui ui() 중앙": q(r["ui"], .5), "ui() 상위 1%": q(r["ui"], .99), "유휴 6초 ui() 횟수": idle["ui"]}
        out[nm] = row
        print(nm, json.dumps(row, ensure_ascii=False))
    print(json.dumps({"rate": RATE, "secs": SECS, **out}, ensure_ascii=False))
