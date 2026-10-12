// 2026-10-13 체결 버블(사용자 «web.bookmap.com 같은 체결 버블 차트» · 시안 승인 «이 차트들을 청산맵 차트 사이즈로 해서 배포해줘»).
// 자리 = 청산맵과 같은 5분 차트 교체 영역(fp_daily.js fpPlaceOverlay · fpAltSet). 보기 토글은 liq_profile.js setView 가 «fpview» 이벤트로 알린다.
// 배경 = 호가 래스터(/api/flow/heatmap, 작음 파랑 → 노랑 → 큼 빨강) · 흰 선 = 현재가(래스터 mid) ·
// 버블 = 1초 체결 테이프(/api/bubble-tape, 수집기가 5초마다 써서 ~5초 늦다): $2.5만 미만 점 · 그 이상 원(넓이 ∝ 금액) · $10만+ 고리·금액.
// 아래 막대 = 구간별 거래대금(매수 | 매도).
// ponytail: 확대·이동·툴팁 없음(창 3/5/10분 탭만) -- 필요하면 liq_profile.js bindCanvas 를 옮겨 온다.
(function () {
  var $ = function (id) { return document.getElementById(id); };
  var css = function (n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); };
  var S = { on: false, sec: 180, tape: null, heat: null, heatT: 0, busyT: false, busyH: false, raf: 0 };
  var HEAT_Q = { 180: [180, 1], 300: [300, 1], 600: [600, 1] };   // 창 → 래스터 (cols, agg 초)
  var STOPS = [0, 0.2, 0.4, 0.55, 0.68, 0.84, 1];                   // --bbm-h0..6 위치

  function asset() { return typeof activeSnapshotAsset === "undefined" ? "eth" : activeSnapshotAsset; }
  function buf(s) { var r = atob(s), u = new Uint8Array(r.length); for (var i = 0; i < r.length; i++) u[i] = r.charCodeAt(i); return u.buffer; }
  function usd(v) { return "$" + (v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : Math.round(v / 1e3) + "k"); }
  function fmtPx(p) { return p.toFixed(p >= 10 ? 2 : 4); }
  function rgb(tok, a) { var c = css(tok).split(/\s+/).join(","); return a == null ? "rgb(" + c + ")" : "rgba(" + c + "," + a + ")"; }
  function lut() {                          // |v| 0..127 → 색 (값 = (|v|/127)^1.35, 시안 그대로)
    var C = STOPS.map(function (_, i) { return css("--bbm-h" + i).split(/\s+/).map(Number); }), out = [];
    for (var k = 0; k < 128; k++) {
      var v = Math.pow(k / 127, 1.35), i = 1;
      while (i < STOPS.length - 1 && v > STOPS[i]) i++;
      var f = (v - STOPS[i - 1]) / (STOPS[i] - STOPS[i - 1]), a = C[i - 1], b = C[i];
      out.push("rgb(" + [0, 1, 2].map(function (j) { return Math.round(a[j] + (b[j] - a[j]) * f); }).join(",") + ")");
    }
    return out;
  }

  function fetchTape() {
    var a = asset(), sec = S.sec;
    if (!S.on || S.busyT || document.hidden) return;
    S.busyT = true;
    fetch("/api/bubble-tape?asset=" + encodeURIComponent(a) + "&sec=" + sec, { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : r.status === 400 ? { rows: [], none: true } : null; })
      .then(function (j) { if (j) { j.asset = a; j.sec = sec; S.tape = j; redraw(); } })
      .catch(function () { /* 다음 초에 다시 */ })
      .then(function () { S.busyT = false; });
  }
  function fetchHeat() {
    var a = asset(), sec = S.sec, q = HEAT_Q[sec], now = Date.now();
    if (!S.on || S.busyH || document.hidden || now - S.heatT < (sec === 600 ? 5000 : sec === 300 ? 3000 : 2000)) return;   // 10분 래스터 = 1회 ~210KB
    S.busyH = true; S.heatT = now;
    fetch("/api/flow/heatmap?symbol=" + encodeURIComponent(a + "usdt") + "&cols=" + q[0] + "&agg=" + q[1], { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : r.status === 400 ? { none: true } : null; })
      .then(function (j) {
        if (!j) return;
        S.heat = j.none || !j.qty_i8 ? { asset: a, none: true }
          : { asset: a, sec: sec, t0: j.t0_ms / 1000, dt: j.dt_s, cols: j.cols, bs: j.bin_size, blo: j.bin_lo, nb: j.n_bins,
              q: new Int8Array(buf(j.qty_i8)), mid: new Float32Array(buf(j.mid_b64)) };
        redraw();
      })
      .catch(function () { S.heatT = 0; })
      .then(function () { S.busyH = false; });
  }

  function draw() {
    S.raf = 0;
    if (!S.on) return;
    if (typeof window.fpPlaceOverlay === "function") window.fpPlaceOverlay($("bubbleMap"));
    var cv = $("bubbleMapCanvas"), st = $("bubbleMapStatus");
    if (!cv || !cv.clientWidth || !cv.clientHeight) return;
    var W = cv.clientWidth, H = cv.clientHeight, dpr = window.devicePixelRatio || 1, narrow = W < 600;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    var g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    var a = asset(), T = S.tape && S.tape.asset === a && S.tape.sec === S.sec ? S.tape : null;
    var M = S.heat && S.heat.asset === a && !S.heat.none && S.heat.sec === S.sec ? S.heat : null;
    var WIN = S.sec, tEnd = T ? T.end : 0, t0 = tEnd - WIN, rows = [], mids = [], lo = Infinity, hi = -Infinity;
    if (T) T.rows.forEach(function (r) {
      if (r[0] < t0) return;
      var p = r[1] * T.bucket;
      rows.push({ t: r[0], p: p, b: r[2] * p, s: r[3] * p, u: (r[2] + r[3]) * p });
      if (p < lo) lo = p; if (p > hi) hi = p;
    });
    if (M) for (var c = 0; c < M.cols; c++) {
      var tc = M.t0 + c * M.dt, m = M.mid[c];
      if (tc + M.dt <= t0 || tc > tEnd || !(m > 0)) continue;
      mids.push([tc, m]); if (m < lo) lo = m; if (m > hi) hi = m;
    }
    if (st) st.textContent = !T ? (S.tape && S.tape.none ? "이 코인은 1초 체결 테이프를 모으지 않습니다" : "체결 버블을 기다리는 중…")
      : !rows.length ? "최근 " + WIN / 60 + "분 체결이 없습니다" : "";
    if (!rows.length) return;
    var last = mids.length ? mids[mids.length - 1][1] : rows[rows.length - 1].p;
    var pad = Math.max((hi - lo) * 0.35, last * 0.0002), minW = Math.max(M ? M.bs * 14 : 0, last * 0.0018);
    lo -= pad; hi += pad;
    if (hi - lo < minW) { var cm = (hi + lo) / 2; lo = cm - minW / 2; hi = cm + minW / 2; }   // 래스터 칸이 두꺼워지지 않게

    var R = narrow ? 74 : 76, B = 20, VH = Math.round((H - B) * 0.2), GAP = 6, PW = W - R, PH = H - B - VH - GAP, vy0 = PH + GAP;
    var X = function (t) { return (t - t0) / WIN * PW; }, Y = function (p) { return PH - (p - lo) / (hi - lo) * PH; };
    var bg = rgb("--bbm-bg"), price = rgb("--bbm-price"), good = css("--good"), bad = css("--bad"), muted = css("--muted");
    var sans = css("--font-sans") || "sans-serif", mono = css("--font-mono") || "monospace";
    g.fillStyle = bg; g.fillRect(0, 0, PW, PH); g.fillRect(0, vy0, PW, VH);

    g.save(); g.beginPath(); g.rect(0, 0, PW, PH); g.clip();
    if (M) {                                                             // 호가 래스터
      var L = lut(), colW = M.dt / WIN * PW, rowH = M.bs / (hi - lo) * PH;
      for (var c2 = 0; c2 < M.cols; c2++) {
        var tc2 = M.t0 + c2 * M.dt;
        if (tc2 + M.dt <= t0 || tc2 > tEnd) continue;
        for (var j = 0; j < M.nb; j++) {
          var v = M.q[c2 * M.nb + j]; if (!v) continue;
          var pj = (M.blo + j) * M.bs; if (pj < lo - M.bs || pj > hi + M.bs) continue;
          g.fillStyle = L[Math.min(127, Math.abs(v))]; g.fillRect(X(tc2), Y(pj + M.bs / 2), colW + 0.6, rowH + 0.6);
        }
      }
    }
    if (mids.length) {                                                   // 현재가 계단선
      var cw = M.dt / WIN * PW;
      g.strokeStyle = price; g.lineWidth = 2; g.beginPath();
      mids.forEach(function (d, i) { var x = X(d[0]), y = Y(d[1]); if (i) g.lineTo(x, y); else g.moveTo(x, y); g.lineTo(x + cw, y); });
      g.stroke();
    }
    var zs = Math.sqrt(180 / WIN), k = (narrow ? 5 : 10) * zs, labels = [];   // 체결 버블(작은 것부터 → 큰 게 위) · 긴 창은 원을 줄인다
    rows.sort(function (p, q) { return p.u - q.u; }).forEach(function (x) {
      var col = x.b >= x.s ? good : bad, cx = X(x.t + 0.5), cy = Y(x.p);
      g.fillStyle = col; g.strokeStyle = col;
      if (x.u < 25000) { g.globalAlpha = 0.7; g.beginPath(); g.arc(cx, cy, narrow ? 1.6 : 2, 0, Math.PI * 2); g.fill(); return; }
      var r = Math.min((narrow ? 13 : 28) * zs, 3 + k * Math.sqrt(x.u / 1e5));
      g.beginPath(); g.arc(cx, cy, r, 0, Math.PI * 2); g.globalAlpha = 0.6; g.fill(); g.globalAlpha = 1; g.lineWidth = 1.2; g.stroke();
      if (x.u >= (narrow ? 2e5 : 1e5) * WIN / 180) {                    // 고리 기준 = 3분 $10만 · 5분 $17만 · 10분 $33만
        g.lineWidth = 1.6; g.strokeStyle = price; g.beginPath(); g.arc(cx, cy, r + 2.5, 0, Math.PI * 2); g.stroke();
        labels.push({ x: cx, y: cy, r: r, u: x.u });
      }
    });
    g.globalAlpha = 1;
    g.font = "800 " + (narrow ? 10.5 : 12) + "px " + sans; g.textBaseline = "middle";
    var placed = [];
    labels.sort(function (p, q) { return q.u - p.u; }).slice(0, WIN > 180 ? 10 : 40).forEach(function (Lb) {   // 금액 글자 -- 긴 창은 상위 10 · 겹치면 작은 쪽 생략
      var txt = usd(Lb.u), w = g.measureText(txt).width, tx = Lb.x + Lb.r + 5;
      if (tx + w > PW - 4) tx = Lb.x - Lb.r - 5 - w;
      if (placed.some(function (p) { return Math.abs(p.y - Lb.y) < 14 && Math.abs(p.x - tx) < w + 6; })) return;
      placed.push({ x: tx, y: Lb.y });
      g.lineWidth = 4; g.strokeStyle = rgb("--bbm-bg", 0.9); g.strokeText(txt, tx, Lb.y); g.fillStyle = price; g.fillText(txt, tx, Lb.y);
    });
    g.restore();

    var ly = Y(last), raw = (hi - lo) / (narrow ? 6 : 9), e = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));   // 가격축
    var step = [1, 2, 5, 10].filter(function (s) { return s * e >= raw; })[0] * e, dec = Math.max(0, -Math.floor(Math.log(step) / Math.LN10 + 1e-9));
    g.fillStyle = muted; g.font = "600 " + (narrow ? 10.5 : 11.5) + "px " + mono; g.textBaseline = "middle";
    for (var p = Math.ceil(lo / step) * step; p <= hi; p += step) {
      var y = Y(p); if (y < 8 || y > PH - 6 || Math.abs(y - ly) < 14) continue;
      g.fillText(p.toFixed(dec), PW + 8, y);
    }
    var lab = fmtPx(last); g.font = "800 " + (narrow ? 11.5 : 14) + "px " + sans;
    var bw = Math.min(R - 4, g.measureText(lab).width + 12);
    g.fillStyle = price; g.beginPath();
    if (g.roundRect) g.roundRect(PW + 2, ly - 11, bw, 22, 6); else g.rect(PW + 2, ly - 11, bw, 22);
    g.fill(); g.fillStyle = bg; g.fillText(lab, PW + 8, ly + 1);

    var BK = Math.max(1, Math.round(WIN / (narrow ? 30 : 60))), nb = Math.ceil(WIN / BK), vb = [], vmax = 1;   // 거래 막대
    for (var i = 0; i < nb; i++) vb.push([0, 0]);
    rows.forEach(function (x) { var b = vb[Math.min(nb - 1, Math.floor((x.t - t0) / BK))]; b[0] += x.b; b[1] += x.s; });
    vb.forEach(function (b) { vmax = Math.max(vmax, b[0], b[1]); });
    var bw2 = PW / nb, hw = Math.max(1, bw2 / 2 - 1);
    vb.forEach(function (b, i2) {
      var x = i2 * bw2, hb = b[0] / vmax * (VH - 4), hs = b[1] / vmax * (VH - 4);
      g.fillStyle = good; g.fillRect(x + 1, vy0 + VH - hb, hw, hb); g.fillStyle = bad; g.fillRect(x + 1 + hw, vy0 + VH - hs, hw, hs);
    });
    g.fillStyle = muted; g.font = "600 10px " + mono; g.textBaseline = "top";
    g.fillText(usd(vmax), PW + 8, vy0); g.fillText(BK + "초", PW + 8, vy0 + 14);

    var ev = [30, 60, 120, 300, 600].filter(function (s) { return s >= WIN / (narrow ? 3 : 6); })[0] || 600;   // 시간축
    g.font = "600 10.5px " + mono; g.strokeStyle = rgb("--bbm-price", 0.18); g.lineWidth = 1;
    for (var s = Math.ceil(t0 / ev) * ev; s < tEnd; s += ev) {
      var xs = X(s); if (xs < 20 || xs > PW - 30) continue;
      g.setLineDash([3, 4]); g.beginPath(); g.moveTo(xs, 0); g.lineTo(xs, vy0 + VH); g.stroke(); g.setLineDash([]);
      var hm = new Date(s * 1000).toTimeString().slice(0, ev < 60 ? 8 : 5);
      g.fillText(hm, xs - g.measureText(hm).width / 2, vy0 + VH + 4);
    }
  }
  function redraw() { if (!S.raf) S.raf = requestAnimationFrame(draw); }

  function setSec(sec) {
    S.sec = HEAT_Q[sec] ? sec : 180; S.heatT = 0;
    document.querySelectorAll("#bubbleWinTabs [data-sec]").forEach(function (b) {
      var on = Number(b.dataset.sec) === S.sec;
      b.classList.toggle("active", on); b.setAttribute("aria-pressed", String(on));
    });
    try { localStorage.setItem("bubbleSec", String(S.sec)); } catch (e) { /* 저장 못 해도 동작 */ }
    fetchTape(); fetchHeat(); redraw();
  }
  function setOn(on) {
    S.on = on;
    if (typeof window.fpAltSet === "function") window.fpAltSet("bubble", on);
    var box = $("bubbleMap"), wt = $("bubbleWinTabs");
    if (box) box.hidden = !on;
    if (wt) wt.hidden = !on;
    if (on) { S.heatT = 0; fetchTape(); fetchHeat(); redraw(); }
  }

  window.addEventListener("fpview", function (e) { setOn(e.detail === "bubble"); });
  window.fpBubbleActive = function () { return S.on; };   // app.js 가 체결 기둥 토글을 숨길 때 읽는다
  function init() {
    var cv = $("bubbleMapCanvas");
    if (!cv) return;
    document.querySelectorAll("#bubbleWinTabs [data-sec]").forEach(function (b) {
      b.addEventListener("click", function () { setSec(Number(b.dataset.sec)); });
    });
    var sv = 180;
    try { sv = Number(localStorage.getItem("bubbleSec")) || 180; } catch (e) { /* 기본 3분 */ }
    setSec(sv);
    new ResizeObserver(redraw).observe(cv);
    setInterval(function () {                     // 테이프 1초 · 래스터 2초(5분 3초 · 10분 5초) · 코인 탭·5분 차트 배치가 바뀌어도 따라간다
      if (!S.on) return;
      fetchTape(); fetchHeat(); redraw();
    }, 1000);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
