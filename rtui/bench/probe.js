// 실기기 측정(?bench=<초> 일 때만) -- rtui/index.html 이 맨 먼저 import 하고, 지금 화면은 serve.py 가 index.html 머리에 끼운다. 두 페이지 같은 코드.
// 주소 인자: bench=부하 초(8) · rate=체결/초(300) · idle=유휴 초(6) · lead=시계를 형성 봉 열리기 몇 초 전에서 시작(12)
// 하는 일: ① 시계를 합성 자료(형성 봉 1795156500) 앞으로 · @trade 소켓을 이 서버 /ws/trade 로 · EventSource 는 죽은 것 · 다른 출처 fetch 는 막고 기록
//          ② rAF·타이머·소켓 메시지 콜백 시간 · 프레임 시각 · 그리기 끝 시각(rtui = egui ui() 끝 · 지금 화면 = renderSnapshotChart · fpCv.live)
//          ③ 형성 봉이 열린 뒤 서버에 체결 rate×bench 를 시키고 → 유휴 idle 초 → /bench/result 로 보내고 화면에 표
(() => {
  const q = new URLSearchParams(location.search);
  if (!q.has("bench")) return;
  const N = (k, d) => (q.has(k) && q.get(k) !== "" ? +q.get(k) : d);
  const SECS = N("bench", 8) || 8, RATE = N("rate", 300), IDLE = N("idle", 6), LEAD = N("lead", 12);
  const PAGE = location.pathname.startsWith("/rtui/") ? "rtui" : "js";
  const LABEL = PAGE + (q.get("src") === "push" ? "-push" : "") + (q.get("aa") === "0" ? "-aa0" : "");   // 결과 파일 이름·표
  const B = 1795156500;   // rtui/compare/fixture 의 형성 봉 시작(live_bench.py 와 같다)
  const now = () => performance.now(), RT = { st: window.setTimeout.bind(window), rq: window.requestAnimationFrame.bind(window) };
  const sleep = (ms) => new Promise((r) => RT.st(r, ms));
  const S = { on: false, cur: null, fr: new Map(), wsms: 0, timer: 0, arr: [], draws: [], lt: 0, loaf: [], blocked: [], es: [], sockets: [], other: [] };
  window.__bench = S;

  // ① 시계: 페이지가 뜬 순간 = 형성 봉 LEAD 초 전(Date.now · new Date() 만 -- 인자 있는 Date 는 그대로)
  const RD = Date, OFF = (B - LEAD) * 1000 - RD.now();
  class D extends RD { constructor(...a) { if (a.length) super(...a); else super(RD.now() + OFF); } static now() { return RD.now() + OFF; } }
  window.Date = D;
  // 지금 화면: 겹침 선은 끄고 POC 만(compare/harness.py 와 같은 스위치)
  if (PAGE === "js") try { localStorage.setItem("mcLine", JSON.stringify({ poc: true, vwap: false, veto: false, sma50: false, sw1h: false, sw30m: false, sw30mOn: 1, ob: false, fvg: false, orb: false, cci: false, rsi2: false, rdiv: false, srl: false, bb: false, wvwap: false })); } catch (e) {}

  // ② rAF: 프레임 시각(ts)별 [첫 콜백 시작, 마지막 콜백 끝, 앱 콜백 시간 합, 앱 콜백 수]
  const note = (ts, s, e, cb, n) => { if (!S.on) return; const f = S.fr.get(ts); if (f) { f[0] = Math.min(f[0], s); f[1] = Math.max(f[1], e); f[2] += cb; f[3] += n; } else S.fr.set(ts, [s, e, cb, n]); };
  window.requestAnimationFrame = (cb) => RT.rq((ts) => { const t0 = now(); S.cur = ts; try { cb(ts); } finally { const t1 = now(); S.cur = null; note(ts, t0, t1, t1 - t0, 1); } });
  for (const k of ["setTimeout", "setInterval"]) {
    const o = window[k];
    window[k] = function (f, ...a) {
      if (typeof f !== "function") return o.call(window, f, ...a);
      return o.call(window, function (...x) { const t0 = now(); try { return f.apply(this, x); } finally { if (S.on) S.timer += now() - t0; } }, ...a);
    };
  }
  // 소켓: @trade → 이 서버 /ws/trade · 허브(경로 /rt · rtui ?src=push) → 그대로(이 서버의 가짜 허브) · 그 밖은 /ws/none(서버가 바로 닫고 주소는 기록)
  // onmessage 시간과 «보낸 시각 · 처리 시작 시각»: 보낸 시각 = 직결은 서버가 붙인 _s(ms) · 허브는 체결 사건의 tx_us(µs, 허브가 보낸 시각)
  const RWS = window.WebSocket;
  window.WebSocket = class extends RWS {
    constructor(u, p) {
      const hub = new URL(String(u), location.href).pathname === "/rt", trade = hub || /@trade$/.test(String(u));
      super(hub ? String(u) : `ws://${location.host}/ws/${trade ? "trade" : "none"}`, p);
      this.__trade = trade; S.sockets.push(this); if (!trade) S.other.push(String(u));
    }
    get onmessage() { return this.__om || null; }
    set onmessage(f) {
      this.__om = f;
      super.onmessage = f && ((e) => {
        const t0 = now(); let s = NaN;
        if (S.on && typeof e.data === "string") {
          const i = e.data.lastIndexOf('"_s":');
          if (i >= 0) s = parseFloat(e.data.slice(i + 5));
          else if (e.data.includes('"type":"trade"')) { const j = e.data.indexOf('"tx_us":'); if (j >= 0) s = parseFloat(e.data.slice(j + 8)) / 1000; }
        }
        try { return f.call(this, e); } finally { if (S.on) { S.wsms += now() - t0; if (s > 0) S.arr.push([s, t0]); } }
      });
    }
  };
  window.EventSource = class { constructor(u) { this.url = u; this.readyState = 1; S.es.push(String(u)); } addEventListener() {} removeEventListener() {} close() { this.readyState = 2; } };
  const RF = window.fetch.bind(window);
  window.fetch = (input, init) => {
    const u = new URL(typeof input === "string" || input instanceof URL ? input : input.url, location.href);
    if (u.origin !== location.origin) { S.blocked.push(u.href); return Promise.reject(new TypeError("bench: 바깥 주소 막음")); }
    return RF(input, init);
  };
  try { new PerformanceObserver((l) => { if (S.on) S.lt += l.getEntries().length; }).observe({ type: "longtask" }); } catch (e) {}
  try { new PerformanceObserver((l) => { if (S.on) for (const x of l.getEntries()) S.loaf.push([x.duration, x.blockingDuration]); }).observe({ type: "long-animation-frame" }); } catch (e) {}

  const qt = (v, a) => { const s = [...v].sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(a * s.length))] : null; };
  const r2 = (x) => (x == null || !Number.isFinite(x) ? null : Math.round(x * 100) / 100);
  const reset = () => { S.fr = new Map(); S.wsms = 0; S.timer = 0; S.arr = []; S.draws = []; S.lt = 0; S.loaf = []; };
  // rtui = 그리기 끝·ui() 시간(Rust LAT) · 지금 화면 = 아래 감싼 함수가 S.draws 에 넣는다
  const takeLat = () => { if (PAGE !== "rtui") return { ends: S.draws.slice(), ui: [] }; const v = [...window.rtui.take_lat()]; const k = v.findIndex((x) => x !== x); return { ends: v.slice(0, k), ui: v.slice(k + 1) }; };
  const wrapJs = () => {   // compare/live_bench.py JS_WRAP 과 같은 자리
    const orig = window.renderSnapshotChart;
    window.renderSnapshotChart = function () { const r = orig.apply(this, arguments); if (S.on) S.draws.push(now()); return r; };
    let L = fpCv.live;   // eslint-disable-line no-undef -- app.js 전역
    Object.defineProperty(fpCv, "live", { configurable: true, get: () => L, set: (v) => { L = v && (() => { const r = v(); if (r !== false && S.on) S.draws.push(now()); return r; }); } });
  };
  const ready = () => { try { return PAGE === "rtui" ? !!window.RESULT : !!(fpCv.cs && latestFootprint); } catch (e) { return false; } };   // eslint-disable-line no-undef
  const until = async (f, ms, what) => { const t = now(); while (!f()) { if (now() - t > ms) throw new Error(`기다림 넘김: ${what}`); await sleep(20); } };
  const renderer = () => {
    try { const c = document.createElement("canvas"), g = c.getContext("webgl2") || c.getContext("webgl"), e = g && g.getExtension("WEBGL_debug_renderer_info");
          return g ? String(g.getParameter(e ? e.UNMASKED_RENDERER_WEBGL : g.RENDERER)) : "WebGL 없음"; } catch (e) { return "알 수 없음: " + e; }
  };

  // 체결별: 도착(서버가 보낸 시각을 페이지 시계로 -- 최소 지연을 0 으로 맞춘다) → 그 뒤 처음 끝난 그리기 → 그 그리기가 들어간 프레임의 «다음» 프레임 시각(rAF 기준 표시)
  // 서버 시각 = 단조 시계(serve.py) · 페이지 = performance.now(단조) -- 두 시계의 차는 일정해서 최소 지연 하나로 맞춘다(벽시계를 쓰면 시각 맞추기 한 번에 전체가 수백 ms 밀린다)
  function latency(arr, ends, fr) {
    if (!arr.length) return {};
    const to = 0, off = Math.min(...arr.map(([s, h]) => h - (s - to)));
    const e = [...ends].sort((a, b) => a - b), F = [...fr.entries()].sort((a, b) => a[0] - b[0]);
    const lb = (v, x) => { let lo = 0, hi = v.length; while (lo < hi) { const m = (lo + hi) >> 1; if (v[m] < x) lo = m + 1; else hi = m; } return lo; };
    const shown = (d) => {
      let k = F.findIndex(([, f]) => f[0] <= d && d <= f[1]);   // 프레임 안(rAF 콜백 안)에서 그렸다
      if (k < 0) k = F.findIndex(([, f]) => f[0] > d);           // 밖(소켓 처리 중)이면 다음 프레임에 들어간다
      return k >= 0 && k + 1 < F.length ? F[k + 1][0] : null;
    };
    const drawn = [], pres = [], queue = [];
    const cache = new Map();
    for (const [s, h] of arr) {
      const a = s - to + off, i = lb(e, h);
      queue.push(h - a);
      if (i >= e.length) continue;
      drawn.push(e[i] - a);
      if (!cache.has(i)) cache.set(i, shown(e[i]));
      const p = cache.get(i); if (p != null) pres.push(p - a);
    }
    return { n: arr.length, queue_med: r2(qt(queue, 0.5)), queue_p99: r2(qt(queue, 0.99)), drawn_med: r2(qt(drawn, 0.5)), drawn_p99: r2(qt(drawn, 0.99)),
             shown_med: r2(qt(pres, 0.5)), shown_p99: r2(qt(pres, 0.99)), shown_n: pres.length };
  }

  function show(res) {
    const d = document.createElement("div");
    d.style.cssText = "position:fixed;right:8px;bottom:8px;z-index:99999;background:#111a;color:#eee;font:12px/1.4 monospace;padding:8px 10px;border:1px solid #888;white-space:pre;max-width:46vw;overflow:auto;max-height:90vh";
    d.id = "benchOut";
    d.textContent = `${res.page} · ${RATE}/초 × ${SECS}초 · ${res.env.renderer}\n` + Object.entries(res.summary).map(([k, v]) => `${k.padEnd(28)} ${v}`).join("\n");
    document.body.appendChild(d);
  }

  async function run() {
    const tNav = now();
    await until(ready, 120000, "첫 그림");
    const first = now();
    if (PAGE === "js") wrapJs();
    await until(() => S.sockets.some((w) => w.__trade && w.readyState === 1), 30000, "@trade 소켓");
    const ws = () => S.sockets.filter((w) => w.__trade && w.readyState === 1).pop();
    if (Date.now() >= B * 1000) throw new Error(`형성 봉이 이미 열렸다(첫 그림까지 ${Math.round(first)}ms) -- lead 를 늘려라`);
    ws().send(JSON.stringify({ cmd: "pre", T: Date.now() }));   // 봉 열리기 전 체결 1건 = 소켓이 봉 전부터 붙어 있었다(형성 봉 = 브라우저 값)
    await until(() => Date.now() > B * 1000 + 800, (LEAD + 5) * 1000, "형성 봉");
    // 부하: 이 동안만 probe 가 rAF 를 돌려 모든 프레임 시각을 본다(앱 콜백은 따로 센다)
    reset(); if (PAGE === "rtui") takeLat();
    S.on = true;
    let loop = true; const tick = (ts) => { const t = now(); note(ts, t, t, 0, 0); if (loop) RT.rq(tick); }; RT.rq(tick);
    const t0 = now();
    ws().send(JSON.stringify({ cmd: "start", rate: RATE, secs: SECS, off: OFF }));
    await sleep(SECS * 1000 + 300);
    loop = false; S.on = false;
    const el = now() - t0, L = takeLat(), fr = S.fr, ld = { wsms: S.wsms, timer: S.timer, arr: S.arr, lt: S.lt, loaf: S.loaf };
    // 유휴: 체결 없음(묻기 500ms 는 그대로) -- probe 의 rAF 는 안 돈다
    await sleep(1000);
    reset(); if (PAGE === "rtui") takeLat();
    S.on = true; const ti = now(); await sleep(IDLE * 1000); S.on = false;
    const iel = now() - ti, IL = takeLat();
    const iv = [...S.fr.values()], icb = iv.reduce((a, f) => a + f[2], 0);

    const F = [...fr.entries()].sort((a, b) => a[0] - b[0]), ts = F.map((x) => x[0]), gaps = ts.slice(1).map((t, i) => t - ts[i]);
    const appf = F.filter(([, f]) => f[3] > 0).map(([, f]) => f[2]), cbsum = appf.reduce((a, b) => a + b, 0);
    const lat = latency(ld.arr, L.ends, fr);
    // rtui: ui() 가 든 프레임의 (rAF 콜백 합 − ui()) = 테셀레이션 + WebGL 제출 + 입력 처리
    const rest = [];
    if (PAGE === "rtui") L.ends.forEach((e, i) => { const f = F.find(([, x]) => x[0] <= e && e <= x[1]); if (f) rest.push(f[1][2] - L.ui[i]); });
    const res = {
      page: LABEL, params: Object.fromEntries(q), when: new RD().toISOString(),
      env: { ua: navigator.userAgent, renderer: renderer(), dpr: devicePixelRatio, screen: [screen.width, screen.height], inner: [innerWidth, innerHeight],
             ctx: PAGE === "rtui" ? (() => { try { return document.getElementById("c").getContext("webgl2").getContextAttributes(); } catch (e) { return null; } })() : null },
      first_ms: r2(first - tNav), first_note: PAGE === "rtui" ? "window.RESULT(장면 + rAF 두 번) 시각 · 탐색 시작 기준" : "fpCv.cs·latestFootprint 가 생긴 시각(20ms 폴링) · 탐색 시작 기준",
      load: { secs: r2(el / 1000), sent: ld.arr.length, frames: F.length, gap_med: r2(qt(gaps, 0.5)), gap_p99: r2(qt(gaps, 0.99)), gap_max: r2(Math.max(...gaps)),
              app_frames: appf.length, app_cb_med: r2(qt(appf, 0.5)), app_cb_p99: r2(qt(appf, 0.99)),
              busy_pct: r2((100 * (cbsum + ld.wsms + ld.timer)) / el), rAF_ms: r2(cbsum), ws_ms: r2(ld.wsms), timer_ms: r2(ld.timer),
              draws: L.ends.length, ui_med: r2(qt(L.ui, 0.5)), ui_p99: r2(qt(L.ui, 0.99)), rest_med: r2(qt(rest, 0.5)), rest_p99: r2(qt(rest, 0.99)),
              longtask: ld.lt, loaf: ld.loaf.length, loaf_block_ms: r2(ld.loaf.reduce((a, x) => a + x[1], 0)), lat },
      idle: { secs: r2(iel / 1000), frames: S.fr.size, app_cb: iv.reduce((a, f) => a + f[3], 0), app_cb_ms: r2(icb), ws_ms: r2(S.wsms), timer_ms: r2(S.timer),
              busy_pct: r2((100 * (icb + S.wsms + S.timer)) / iel), draws: IL.ends.length },
      res: performance.getEntriesByType("resource").map((r) => [r.name.replace(location.origin, ""), r.decodedBodySize]),
      external: { blocked_fetch: S.blocked, event_source: S.es, other_ws: S.other, resources: performance.getEntriesByType("resource").map((r) => r.name).filter((u) => !u.startsWith(location.origin)) },
    };
    const l = res.load;
    res.summary = {
      "렌더러": res.env.renderer, "도착→표시(rAF 기준) 중앙": l.lat.shown_med, "도착→표시 상위 1%": l.lat.shown_p99,
      "도착→그리기 끝 중앙": l.lat.drawn_med, "도착→그리기 끝 상위 1%": l.lat.drawn_p99, "도착→처리 시작 상위 1%": l.lat.queue_p99,
      "프레임 간격 중앙": l.gap_med, "프레임 간격 상위 1%": l.gap_p99, "프레임당 앱 rAF 콜백 중앙": l.app_cb_med, "앱 rAF 콜백 상위 1%": l.app_cb_p99,
      "메인 스레드 바쁨 %(부하)": l.busy_pct, "egui ui() 중앙": l.ui_med, "rAF−ui() 중앙(테셀·제출)": l.rest_med, "긴 작업 수": l.longtask,
      "유휴 프레임 수": res.idle.frames, "유휴 바쁨 %": res.idle.busy_pct, "유휴 그리기 수": res.idle.draws, "첫 화면 ms": res.first_ms,
      "바깥 요청 수": res.external.blocked_fetch.length + res.external.resources.length + res.external.other_ws.length,
    };
    window.BENCH = res;
    show(res);
    try { const r = await RF("/bench/result", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(res) }); res.saved = (await r.json()).file; } catch (e) { res.saved = "보내기 실패: " + e; }
  }
  const go = () => run().catch((e) => { window.BENCH = { error: String(e) }; const d = document.createElement("div"); d.style.cssText = "position:fixed;right:8px;bottom:8px;z-index:99999;background:#a00;color:#fff;padding:8px"; d.textContent = "bench 실패: " + e; document.body.appendChild(d); });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", go); else go();
})();
