// 풋프린트/청산맵 토글의 «청산맵» (2026-10-08, 사용자 «시간 탭에서 오른쪽 위 토글 → 풋프린트 자리에 코인글래스 청산맵처럼»).
// 원천 = app.js 가 60초마다 받는 /api/liquidation-map 의 tier_profile(같은 24시간 추정 모델을 레버리지 구간별로 남긴 것).
// x = 가격 · 막대 = 그 가격의 추정 청산 규모(레버리지 구간 쌓기, 왼쪽 축) · 선 = 현재가에서 그 가격까지 누적(빨강 롱 · 초록 숏, 오른쪽 축).
// 추정이지 실측 포지션이 아니다(scripts/live_liquidation_map_20260824.py 머리말). 🔴CI esprima: `?.(`·`?.[`·숫자 구분자 금지.
(function () {
  "use strict";
  var TIER_VARS = ["--lev10", "--lev25", "--lev50", "--lev100"];
  var MAX_RANGE = 0.15, BAR_PX = 4;
  var S = { on: false, map: null, price: 0, raf: 0, hover: null, view: null };
  var $ = function (id) { return document.getElementById(id); };
  var css = function (n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); };
  var tipShow = function (x, y, h) { if (typeof showTooltip === "function") showTooltip(x, y, h); };   // app.js 전역
  var tipHide = function () { if (typeof hideTooltip === "function") hideTooltip(); };

  var UNIT = "oi_usd";                       // 그릴 때 tier_profile.unit 으로 바뀐다(share = 전체 추정 포지션 중 비율)
  function usd(v) {
    if (UNIT === "share") return (v * 100).toFixed(v >= 0.1 ? 0 : 1) + "%";
    var a = Math.abs(v);
    return a >= 1e9 ? "$" + (a / 1e9).toFixed(2) + "B" : a >= 1e6 ? "$" + (a / 1e6).toFixed(1) + "M" : a >= 1e3 ? "$" + (a / 1e3).toFixed(0) + "k" : "$" + a.toFixed(0);
  }
  function fmtPx(p) { return p >= 1000 ? p.toFixed(0) : p >= 10 ? p.toFixed(2) : p.toFixed(4); }
  function curMap() { return typeof latestLiquidationMap === "undefined" ? null : latestLiquidationMap; }      // app.js 전역
  function livePrice(tp) {
    var a = typeof activeSnapshotAsset === "undefined" ? "eth" : activeSnapshotAsset;
    var p = typeof latestLivePriceByAsset === "undefined" ? 0 : Number(latestLivePriceByAsset[a] || 0);
    return p > 0 ? p : tp.current_price;
  }
  function niceStep(raw) {
    var p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
    return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
  }

  // 서버 칸(0.1%) → 화면 칸(막대 ≥ BAR_PX) · 현재가 기준 누적(롱 = 현재가에서 왼쪽으로 · 숏 = 오른쪽으로)
  function shape(tp, cp, plotW) {
    var n = tp.tiers[0].values.length, bw = tp.bin_width, i, t, a = n, b = -1;
    for (t = 0; t < tp.tiers.length; t++) for (i = 0; i < n; i++) if (tp.tiers[t].values[i] > 0) { a = Math.min(a, i); b = Math.max(b, i); }
    if (b < 0) return null;
    var lo = Math.max((tp.lo + a) * bw, cp * (1 - MAX_RANGE)), hi = Math.min((tp.lo + b + 1) * bw, cp * (1 + MAX_RANGE));
    var half = Math.max(cp - lo, hi - cp);                  // 현재가가 가운데 오게
    lo = cp - half; hi = cp + half;
    var span = Math.max(1, Math.round((hi - lo) / bw)), grp = Math.max(1, Math.ceil(span * BAR_PX / plotW));
    var gw = grp * bw, k0 = Math.floor(lo / gw), m = Math.ceil(hi / gw) - k0, bars = [];
    for (i = 0; i < m; i++) bars.push([0, 0, 0, 0]);
    for (t = 0; t < tp.tiers.length; t++) {
      var v = tp.tiers[t].values;
      for (i = 0; i < n; i++) {
        if (!(v[i] > 0)) continue;
        var g = Math.floor((tp.lo + i) * bw / gw) - k0;
        if (g >= 0 && g < m) bars[g][t] += v[i];
      }
    }
    var tot = bars.map(function (x) { return x[0] + x[1] + x[2] + x[3]; });
    var cum = new Array(m).fill(0), c0 = Math.floor(cp / gw) - k0, acc = 0;
    for (i = Math.min(c0, m - 1); i >= 0; i--) { acc += i === c0 ? 0 : tot[i]; cum[i] = acc; }
    acc = 0;
    for (i = Math.max(c0, 0); i < m; i++) { acc += i === c0 ? 0 : tot[i]; cum[i] = acc; }
    return { lo: k0 * gw, gw: gw, m: m, bars: bars, tot: tot, cum: cum, c0: c0 };
  }

  function draw() {
    S.raf = 0;
    var cv = $("liqProfileCanvas"), status = $("liqProfileStatus");
    if (!S.on || !cv || !cv.clientWidth) return;
    var map = curMap(), tp = map && map.tier_profile;
    var W = cv.clientWidth, H = cv.clientHeight, dpr = window.devicePixelRatio || 1, narrow = W < 560;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    var g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    if (!tp) {
      if (status) status.textContent = map && map.warmed_up === false ? "청산맵 데이터 수집 중" : "청산맵을 기다리는 중…";
      S.view = null; return;
    }
    var cp = livePrice(tp), L = narrow ? 44 : 58, R = narrow ? 48 : 62, top = 18, axisH = 22;
    var plotW = W - L - R, plotH = H - top - axisH, sh = shape(tp, cp, plotW);
    if (!sh) { if (status) status.textContent = "현재가 주변에 추정 청산 밀집이 없습니다"; S.view = null; return; }
    if (status) status.textContent = "";
    UNIT = tp.unit || "oi_usd";
    var C = { tiers: TIER_VARS.map(css), good: css("--good"), bad: css("--bad"), text: css("--text"), muted: css("--muted"),
              soft: css("--soft-line"), line: css("--line"), bg: css("--chart-bg"), mono: css("--font-mono") || "monospace" };
    var maxBar = Math.max.apply(null, sh.tot) || 1, maxCum = Math.max.apply(null, sh.cum) || 1;
    var bw = plotW / sh.m, base = top + plotH;
    var xOf = function (p) { return L + (p - sh.lo) / (sh.gw * sh.m) * plotW; };
    var yBar = function (v) { return base - v / maxBar * plotH * 0.92; };
    var yCum = function (v) { return base - v / maxCum * plotH * 0.92; };
    S.view = { L: L, plotW: plotW, sh: sh, bw: bw, cp: cp };
    // 눈금(가로선 · 왼쪽 막대 축 · 오른쪽 누적 축)
    var sb = niceStep(maxBar / 4), sc = niceStep(maxCum / 4), v, y;
    g.font = "10px " + C.mono; g.textBaseline = "middle"; g.lineWidth = 1;
    for (v = 0; v <= maxBar / 0.92; v += sb) {
      y = Math.round(yBar(v)) + 0.5;
      g.strokeStyle = C.soft; g.beginPath(); g.moveTo(L, y); g.lineTo(L + plotW, y); g.stroke();
      g.fillStyle = C.muted; g.textAlign = "right"; g.fillText(usd(v), L - 6, y);
    }
    g.textAlign = "left";
    for (v = 0; v <= maxCum / 0.92; v += sc) g.fillText(usd(v), L + plotW + 6, yCum(v));
    var c0 = Math.max(0, Math.min(sh.c0, sh.m - 1));
    var area = function (from, to, col) {                     // 누적 면(막대 뒤)
      g.beginPath(); g.moveTo(L + (from + 0.5) * bw, base);
      for (var i = from; i <= to; i++) g.lineTo(L + (i + 0.5) * bw, yCum(sh.cum[i]));
      g.lineTo(L + (to + 0.5) * bw, base); g.closePath();
      g.fillStyle = col; g.globalAlpha = 0.1; g.fill(); g.globalAlpha = 1;
    };
    area(0, c0, C.bad); area(c0, sh.m - 1, C.good);
    var w = Math.max(1, bw * 0.78), i, t;                    // 막대(레버리지 쌓기: 아래 10배 → 위 75~100배)
    for (i = 0; i < sh.m; i++) {
      var acc = 0;
      for (t = 0; t < 4; t++) {
        var val = sh.bars[i][t];
        if (!(val > 0)) continue;
        var y0 = yBar(acc), y1 = yBar(acc + val);
        g.fillStyle = C.tiers[t]; g.globalAlpha = S.hover === i ? 1 : 0.86;
        g.fillRect(L + i * bw + (bw - w) / 2, y1, w, Math.max(1, y0 - y1)); acc += val;
      }
    }
    g.globalAlpha = 1;
    var line = function (from, to, col) {                     // 누적 선(막대 앞)
      g.beginPath();
      for (var j = from; j <= to; j++) { var x = L + (j + 0.5) * bw, yy = yCum(sh.cum[j]); if (j === from) g.moveTo(x, yy); else g.lineTo(x, yy); }
      g.strokeStyle = col; g.lineWidth = 2; g.stroke();
    };
    line(0, c0, C.bad); line(c0, sh.m - 1, C.good);
    var xc = Math.round(xOf(cp)) + 0.5;                       // 현재가
    g.strokeStyle = C.text; g.setLineDash([4, 4]); g.lineWidth = 1;
    g.beginPath(); g.moveTo(xc, top); g.lineTo(xc, base); g.stroke(); g.setLineDash([]);
    var lab = "현재가 " + fmtPx(cp);
    g.font = "600 11px " + C.mono;
    var lw = g.measureText(lab).width + 12, lx = Math.max(L, Math.min(xc - lw / 2, L + plotW - lw));
    g.fillStyle = C.text; g.fillRect(lx, 1, lw, 15);
    g.fillStyle = C.bg; g.textAlign = "left"; g.fillText(lab, lx + 6, 9);
    g.font = "10px " + C.mono; g.fillStyle = C.muted; g.textAlign = "center"; g.textBaseline = "top";   // 가격 축
    var ps = niceStep(sh.gw * sh.m / Math.max(2, plotW / 90));
    for (var p = Math.ceil(sh.lo / ps) * ps; p <= sh.lo + sh.gw * sh.m; p += ps) {
      var x = xOf(p);
      if (x < L + 16 || x > L + plotW - 16 || Math.abs(x - xc) < 34) continue;
      g.fillText(fmtPx(p), x, base + 6);
      g.strokeStyle = C.soft; g.beginPath(); g.moveTo(Math.round(x) + 0.5, base); g.lineTo(Math.round(x) + 0.5, base + 4); g.stroke();
    }
    g.strokeStyle = C.line; g.beginPath(); g.moveTo(L, base + 0.5); g.lineTo(L + plotW, base + 0.5); g.stroke();
    if (S.hover != null && S.hover < sh.m) {
      g.strokeStyle = C.muted; g.globalAlpha = 0.6; g.setLineDash([2, 3]);
      var hx = Math.round(L + (S.hover + 0.5) * bw) + 0.5;
      g.beginPath(); g.moveTo(hx, top); g.lineTo(hx, base); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    }
    legend(tp, sh);
  }

  function legend(tp, sh) {
    var box = $("liqProfileLegend");
    if (!box) return;
    var item = function (cls, text) { return '<span class="lqp-item"><i class="lqp-sw ' + cls + '"></i>' + text + "</span>"; };
    box.innerHTML = item("lqp-long", "누적 롱 청산 " + usd(sh.cum[0] || 0)) + item("lqp-short", "누적 숏 청산 " + usd(sh.cum[sh.m - 1] || 0)) +
      tp.tiers.map(function (t, k) { return item("lqp-t" + k, t.name + "배"); }).join("");
  }

  function hoverAt(e) {
    var v = S.view, cv = $("liqProfileCanvas"), map = curMap();
    if (!v || !map || !map.tier_profile) return;
    var r = cv.getBoundingClientRect(), i = Math.floor((e.clientX - r.left - v.L) / v.bw);
    if (i < 0 || i >= v.sh.m) { S.hover = null; tipHide(); redraw(); return; }
    S.hover = i;
    var lo = v.sh.lo + i * v.sh.gw, names = map.tier_profile.tiers.map(function (t) { return t.name; });
    var rows = v.sh.bars[i].map(function (x, k) { return x > 0 ? names[k] + "배 " + usd(x) : null; }).filter(Boolean);
    var dist = ((lo + v.sh.gw / 2) / v.cp - 1) * 100;
    tipShow(e.pageX, e.pageY, "<b>$" + fmtPx(lo) + "–" + fmtPx(lo + v.sh.gw) + "</b> (" + (dist > 0 ? "+" : "") + dist.toFixed(2) + "%)<br>" +
      (rows.length ? rows.join(" · ") + "<br>합 " + usd(v.sh.tot[i]) : "추정 청산 없음") + "<br>" +
      (lo + v.sh.gw <= v.cp ? "여기까지 내려가면 누적 롱 청산 " : "여기까지 오르면 누적 숏 청산 ") + usd(v.sh.cum[i]) + " (추정)");
    redraw();
  }

  function redraw() { if (!S.raf) S.raf = requestAnimationFrame(draw); }

  function setView(view) {
    S.on = view === "liq";
    var card = $("fpCard"), box = $("liqProfile");
    if (card) card.classList.toggle("fp-liq-on", S.on);
    if (box) box.hidden = !S.on;
    document.querySelectorAll("#fpViewTabs .asset-tab").forEach(function (b) {
      var on = b.dataset.view === view;
      b.classList.toggle("active", on); b.setAttribute("aria-pressed", String(on));
    });
    try { localStorage.setItem("fpView", view); } catch (e) { /* 저장 못 해도 동작 */ }
    tipHide();
    if (S.on) redraw();
    else if (typeof scheduleSnapshotChartRender === "function") scheduleSnapshotChartRender();
  }

  function init() {
    var cv = $("liqProfileCanvas");
    if (!cv) return;
    document.querySelectorAll("#fpViewTabs .asset-tab").forEach(function (b) {
      b.addEventListener("click", function () { setView(b.dataset.view); });
    });
    cv.addEventListener("pointermove", hoverAt);
    cv.addEventListener("pointerleave", function () { S.hover = null; tipHide(); redraw(); });
    new ResizeObserver(redraw).observe(cv);
    setInterval(function () {                       // 새 청산맵(60초)·현재가(1초)가 오면 다시 그린다
      if (!S.on) return;
      var m = curMap(), tp = m && m.tier_profile, p = tp ? livePrice(tp) : 0;
      if (m !== S.map || p !== S.price) { S.map = m; S.price = p; redraw(); }
    }, 1000);
    var saved = null;
    try { saved = localStorage.getItem("fpView"); } catch (e) { /* 없음 */ }
    setView(saved === "liq" ? "liq" : "fp");
  }

  window.fpLiqActive = function () { return S.on; };   // app.js renderSnapshotChart 가 5분 차트 그리기를 건너뛴다
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
