// 풋프린트/청산맵 토글의 «청산맵» (2026-10-08, 사용자 «시간 탭에서 오른쪽 위 토글 → 풋프린트 자리에 코인글래스 청산맵처럼»
//   · «청산맵도 일봉처럼 확대/축소 · 풋프린트 차트만 바뀌게»).
// 원천 = 1일: app.js 가 60초마다 받는 /api/liquidation-map 의 tier_profile · 7·30일: /api/liquidation-map/tiers?days= (같은 추정 모델, 입력 1시간봉 창만 다름).
// x = 가격 · 막대 = 그 가격의 추정 청산 규모(레버리지 구간 쌓기, 왼쪽 축) · 선 = 현재가에서 그 가격까지 누적(빨강 롱 · 초록 숏, 오른쪽 축).
// 자리 = 5분 차트의 교체 영역 위, 배경 없이(fp_daily.js fpPlaceOverlay · fpAltSet). 확대 = 가격축(휠·두 손가락·키보드), 끌기 = 이동, 더블클릭 = 전체.
// 추정이지 실측 포지션이 아니다(scripts/live_liquidation_map_20260824.py 머리말). 🔴CI esprima: `?.(`·`?.[`·숫자 구분자 금지.
(function () {
  "use strict";
  var TIER_VARS = ["--lev10", "--lev25", "--lev50", "--lev100"];
  var MAX_RANGE = 0.15, BAR_PX = 4, MIN_BINS = 24;
  var S = { on: false, map: null, price: 0, raf: 0, hover: null, view: null, vlo: null, vhi: null, drag: null, pinch: null,
            days: 1, ext: {}, extT: {} };   // days = 보기 기간(1·7·30일). 1일은 app.js 가 받는 지도를 그대로, 7·30일은 /api/liquidation-map/tiers
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
  function curMap() {
    var a = typeof activeSnapshotAsset === "undefined" ? "eth" : activeSnapshotAsset;
    if (S.days > 1) return S.ext[a + "|" + S.days] || null;
    return typeof latestLiquidationMap === "undefined" ? null : latestLiquidationMap;      // app.js 전역
  }
  function fetchExt() {                       // 7·30일: 켜져 있을 때 60초마다(서버도 60초 캐시) -- 새 진입 반영 1분(10-10 사용자 유지)
    var a = typeof activeSnapshotAsset === "undefined" ? "eth" : activeSnapshotAsset, key = a + "|" + S.days, now = Date.now();
    if (!S.on || S.days === 1 || now - (S.extT[key] || 0) < 60000) return;
    S.extT[key] = now;
    fetch("/api/liquidation-map/tiers?asset=" + encodeURIComponent(a) + "&days=" + S.days, { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) { if (j && j.tier_profile) { S.ext[key] = j; redraw(); } })
      .catch(function () { S.extT[key] = 0; });
  }
  function setDays(d) {
    S.days = d; S.vlo = S.vhi = null;
    document.querySelectorAll("#liqDaysTabs [data-days]").forEach(function (b) {   // 2026-10-10 창 탭 자리로 옮김(사용자)
      var on = Number(b.dataset.days) === d;
      b.setAttribute("aria-pressed", String(on)); b.classList.toggle("active", on);
    });
    try { localStorage.setItem("liqDays", String(d)); } catch (e) { /* 저장 못 해도 동작 */ }
    fetchExt(); redraw();
  }
  // 2026-10-10 사용자 «현재가가 청산가를 넘어가면 바로 막대 제거 -- 매초»: 지도가 만들어진 뒤 가격이 지나간 범위(1초 현재가 + 그 뒤 5분봉 고·저)
  //   안의 칸을 0 으로 -- 서버 생존 필터(그 뒤 봉 고·저)와 같은 규칙을 지도 사이 시간에 브라우저에서 이어 붙인다(서버 호출 0).
  //   새 지도가 오면(1·7·30일 60초) 서버가 같은 범위를 이미 걸러 낸 상태라 범위를 그 기준가로 다시 시작한다.
  function sweep(m, tp) {
    var w = S.sweep, cp = tp.current_price, p = livePrice(tp);
    if (!w || w.tp !== tp) w = S.sweep = { tp: tp, lo: cp, hi: cp, out: null, key: "" };
    var lo = Math.min(w.lo, p), hi = Math.max(w.hi, p), a = typeof activeSnapshotAsset === "undefined" ? "eth" : activeSnapshotAsset;
    var gen = Date.parse(m.generated_at || "") / 1000, cs = typeof candleHistoryByAsset === "undefined" ? null : candleHistoryByAsset[a];
    if (gen > 0 && cs) for (var i = cs.length - 1; i >= 0 && cs[i].time + 300 > gen; i--) { lo = Math.min(lo, cs[i].low); hi = Math.max(hi, cs[i].high); }
    w.lo = lo; w.hi = hi;
    return w;
  }
  function liveTp(m) {
    var tp = m && m.tier_profile;
    if (!tp) return tp;
    var w = sweep(m, tp), bw = tp.bin_width, k0 = Math.ceil(w.lo / bw - 0.5), k1 = Math.floor(w.hi / bw + 0.5), key = k0 + "|" + k1;
    if (w.key === key) return w.out;
    w.key = key;
    w.out = Object.assign({}, tp, { tiers: tp.tiers.map(function (t) {
      return { name: t.name, values: t.values.map(function (v, i) { var k = tp.lo + i; return k >= k0 && k <= k1 ? 0 : v; }) };
    }) });
    return w.out;
  }
  function livePrice(tp) {
    var a = typeof activeSnapshotAsset === "undefined" ? "eth" : activeSnapshotAsset;
    var p = typeof latestLivePriceByAsset === "undefined" ? 0 : Number(latestLivePriceByAsset[a] || 0);
    return p > 0 ? p : tp.current_price;
  }
  function halo(g, C, txt, x, y, col) {   // 5분 차트 글자 규약: stroke var(--chart-bg) 3px · paint-order stroke
    g.lineWidth = 3; g.lineJoin = "round"; g.strokeStyle = C.bg; g.strokeText(txt, x, y);
    g.fillStyle = col; g.fillText(txt, x, y);
  }
  function niceStep(raw) {
    var p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
    return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
  }

  // 자동 범위 = 데이터가 있는 곳(현재가 ±15% 안), 현재가가 가운데 오게
  function autoRange(tp, cp) {
    var n = tp.tiers[0].values.length, bw = tp.bin_width, i, t, a = n, b = -1;
    for (t = 0; t < tp.tiers.length; t++) for (i = 0; i < n; i++) if (tp.tiers[t].values[i] > 0) { a = Math.min(a, i); b = Math.max(b, i); }
    if (b < 0) return null;
    var lo = Math.max((tp.lo + a) * bw, cp * (1 - MAX_RANGE)), hi = Math.min((tp.lo + b + 1) * bw, cp * (1 + MAX_RANGE));
    var half = Math.max(cp - lo, hi - cp, MIN_BINS * bw / 2);
    return [cp - half, cp + half];
  }

  // 서버 칸(0.1%) → 보이는 범위 [lo, hi] 의 화면 칸(막대 ≥ BAR_PX). 누적은 **서버 칸 전체**에서 현재가 기준으로 먼저 쌓고
  //   묶음의 «현재가에서 먼 끝» 값을 쓴다 -- 확대·이동해 현재가가 화면 밖이어도 누적이 맞다.
  function shape(tp, cp, plotW, lo, hi) {
    var n = tp.tiers[0].values.length, bw = tp.bin_width, lo0 = tp.lo, i, t;
    var tot = new Array(n).fill(0);
    for (t = 0; t < tp.tiers.length; t++) for (i = 0; i < n; i++) if (tp.tiers[t].values[i] > 0) tot[i] += tp.tiers[t].values[i];
    var cpAbs = Math.floor(cp / bw), c = cpAbs - lo0, cum = new Array(n).fill(0), acc = 0;
    for (i = Math.min(c - 1, n - 1); i >= 0; i--) { acc += tot[i]; cum[i] = acc; }      // 롱: 현재가 칸 아래로
    acc = 0;
    for (i = Math.max(c + 1, 0); i < n; i++) { acc += tot[i]; cum[i] = acc; }          // 숏: 위로
    var cumAt = function (abs) {                 // 데이터 밖은 끝 값(그 너머엔 칸이 없다)
      var k = abs - lo0;
      if (k < 0) return abs < cpAbs ? cum[0] || 0 : 0;
      if (k >= n) return abs > cpAbs ? cum[n - 1] || 0 : 0;
      return cum[k];
    };
    var grp = Math.max(1, Math.ceil((hi - lo) / bw * BAR_PX / plotW)), gw = grp * bw;
    var g0 = Math.floor(lo / gw), m = Math.max(1, Math.ceil(hi / gw) - g0), bars = [], gcum = [];
    for (i = 0; i < m; i++) bars.push([0, 0, 0, 0]);
    for (t = 0; t < tp.tiers.length; t++) {
      var v = tp.tiers[t].values;
      for (i = 0; i < n; i++) {
        if (!(v[i] > 0)) continue;
        var g = Math.floor((lo0 + i) / grp) - g0;
        if (g >= 0 && g < m) bars[g][t] += v[i];
      }
    }
    for (i = 0; i < m; i++) {
      var a = (g0 + i) * grp, b = a + grp - 1;        // 이 묶음의 서버 칸(절대 번호)
      gcum.push(b < cpAbs ? cumAt(a) : a > cpAbs ? cumAt(b) : 0);
    }
    var tsum = bars.map(function (x) { return x[0] + x[1] + x[2] + x[3]; });
    return { lo: g0 * gw, gw: gw, m: m, bars: bars, tot: tsum, cum: gcum, cpAbs: cpAbs, grp: grp,
             longAll: cumAt(lo0 - 1), shortAll: cumAt(lo0 + n) };
  }

  function draw() {
    S.raf = 0;
    if (!S.on) return;
    if (typeof window.fpPlaceOverlay === "function") window.fpPlaceOverlay($("liqProfile"));
    var cv = $("liqProfileCanvas"), status = $("liqProfileStatus");
    if (!cv || !cv.clientWidth) return;
    var map = curMap(), tp = liveTp(map);
    var W = cv.clientWidth, H = cv.clientHeight, dpr = window.devicePixelRatio || 1, narrow = W < 560;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    var g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    if (!tp) {
      if (status) status.textContent = map && map.warmed_up === false ? "청산맵 데이터 수집 중" : "청산맵(" + S.days + "일)을 기다리는 중…";
      S.view = null; return;
    }
    var cp = livePrice(tp), rng = autoRange(tp, cp);
    if (!rng) { if (status) status.textContent = "현재가 주변에 추정 청산 밀집이 없습니다"; S.view = null; return; }
    if (status) status.textContent = "";
    UNIT = tp.unit || "oi_usd";
    var lo = S.vlo != null ? S.vlo : rng[0], hi = S.vhi != null ? S.vhi : rng[1];
    var L = narrow ? 60 : 68, R = narrow ? 54 : 68, top = 22, axisH = 28;
    var plotW = W - L - R, plotH = H - top - axisH, sh = shape(tp, cp, plotW, lo, hi);
    var C = { tiers: TIER_VARS.map(css), good: css("--good"), bad: css("--bad"), text: css("--text"), muted: css("--muted"),
              soft: css("--soft-line"), line: css("--line"), bg: css("--chart-bg"), sans: css("--font-sans") || "sans-serif" };
    var maxBar = Math.max.apply(null, sh.tot) || 1, maxCum = Math.max.apply(null, sh.cum) || 1;
    var xOf = function (p) { return L + (p - lo) / (hi - lo) * plotW; };
    var gwPx = sh.gw / (hi - lo) * plotW, gx = function (k) { return xOf(sh.lo + k * sh.gw); };
    var base = top + plotH;
    var yBar = function (v) { return base - v / maxBar * plotH * 0.92; };
    var yCum = function (v) { return base - v / maxCum * plotH * 0.92; };
    S.view = { L: L, plotW: plotW, sh: sh, cp: cp, lo: lo, hi: hi };
    g.save(); g.beginPath(); g.rect(L, 0, plotW, H); g.clip();
    var sb = niceStep(maxBar / 4), sc = niceStep(maxCum / 4), v, y, i, t;
    g.lineWidth = 1;
    for (v = 0; v <= maxBar / 0.92; v += sb) { y = Math.round(yBar(v)) + 0.5; g.strokeStyle = C.line; g.globalAlpha = 0.82; g.beginPath(); g.moveTo(L, y); g.lineTo(L + plotW, y); g.stroke(); g.globalAlpha = 1; }   // 5분 차트 .chart-grid
    // 누적 면·선 -- 현재가 칸을 경계로 왼쪽(롱)·오른쪽(숏)
    var cIdx = sh.cpAbs / sh.grp - Math.floor(sh.lo / sh.gw);
    var side = function (left) { var out = []; for (var k = 0; k < sh.m; k++) if (left ? k <= cIdx : k >= Math.floor(cIdx)) out.push(k); return out; };
    var curve = function (ks, col) {
      if (!ks.length) return;
      g.beginPath(); g.moveTo(gx(ks[0]) + gwPx / 2, base);
      ks.forEach(function (k) { g.lineTo(gx(k) + gwPx / 2, yCum(sh.cum[k])); });
      g.lineTo(gx(ks[ks.length - 1]) + gwPx / 2, base); g.closePath();
      g.fillStyle = col; g.globalAlpha = 0.1; g.fill(); g.globalAlpha = 1;
      g.beginPath();
      ks.forEach(function (k, j) { var x = gx(k) + gwPx / 2, yy = yCum(sh.cum[k]); if (j === 0) g.moveTo(x, yy); else g.lineTo(x, yy); });
      g.strokeStyle = col; g.lineWidth = 2; g.stroke(); g.lineWidth = 1;
    };
    curve(side(true), C.bad); curve(side(false), C.good);
    var bwid = Math.max(1, gwPx * 0.78);                          // 막대(레버리지 쌓기: 아래 10배 → 위 75~100배)
    for (i = 0; i < sh.m; i++) {
      var acc = 0;
      for (t = 0; t < 4; t++) {
        var val = sh.bars[i][t];
        if (!(val > 0)) continue;
        var y0 = yBar(acc), y1 = yBar(acc + val);
        g.fillStyle = C.tiers[t]; g.globalAlpha = S.hover === i ? 1 : 0.86;
        g.fillRect(gx(i) + (gwPx - bwid) / 2, y1, bwid, Math.max(1, y0 - y1)); acc += val;
      }
    }
    g.globalAlpha = 1;
    var xc = Math.round(xOf(cp)) + 0.5;                           // 현재가
    if (xc >= L && xc <= L + plotW) {
      g.strokeStyle = C.text; g.setLineDash([4, 4]);
      g.beginPath(); g.moveTo(xc, top); g.lineTo(xc, base); g.stroke(); g.setLineDash([]);
    }
    if (S.hover != null && S.hover < sh.m) {
      g.strokeStyle = C.muted; g.globalAlpha = 0.6; g.setLineDash([2, 3]);
      var hx = Math.round(gx(S.hover) + gwPx / 2) + 0.5;
      g.beginPath(); g.moveTo(hx, top); g.lineTo(hx, base); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    }
    g.restore();
    g.font = "600 12px " + C.sans; g.textBaseline = "middle";      // 축 글자(플롯 밖) = 5분 차트 글자(굵게 · 바탕 외곽선)
    g.textAlign = "right";
    for (v = 0; v <= maxBar / 0.92; v += sb) halo(g, C, usd(v), L - 6, yBar(v), C.muted);
    g.textAlign = "left";
    for (v = 0; v <= maxCum / 0.92; v += sc) halo(g, C, usd(v), L + plotW + 6, yCum(v), C.muted);
    var lab = "현재가 " + fmtPx(cp) + (xc < L ? " (왼쪽 밖)" : xc > L + plotW ? " (오른쪽 밖)" : "");
    g.font = "700 13px " + C.sans; g.textAlign = "center";           // 현재가 = 상자 없이 굵은 숫자(5분 차트와 같은 규약)
    var lw = g.measureText(lab).width, lx = Math.max(L + lw / 2, Math.min(xc, L + plotW - lw / 2));
    halo(g, C, lab, lx, 10, C.text);
    g.font = "700 " + (narrow ? 12 : 13) + "px " + C.sans; g.textAlign = "center"; g.textBaseline = "top";   // 가격 축 = 5분 차트 x축 글자
    var ps = niceStep((hi - lo) / Math.max(2, plotW / 96));
    g.strokeStyle = C.line;
    for (var p = Math.ceil(lo / ps) * ps; p <= hi; p += ps) {
      var x = xOf(p);
      if (x < L + 18 || x > L + plotW - 18 || Math.abs(x - xc) < 40) continue;
      g.fillStyle = C.muted; g.fillText(fmtPx(p), x, base + 8);
      g.beginPath(); g.moveTo(Math.round(x) + 0.5, base); g.lineTo(Math.round(x) + 0.5, base + 5); g.stroke();
    }
    g.strokeStyle = C.line; g.beginPath(); g.moveTo(L, base + 0.5); g.lineTo(L + plotW, base + 0.5); g.stroke();
    legend(tp, sh);
  }

  function legend(tp, sh) {
    var box = $("liqProfileLegend");
    if (!box) return;
    var item = function (cls, text) { return '<span class="lqp-item"><i class="lqp-sw ' + cls + '"></i>' + text + "</span>"; };
    box.innerHTML = item("lqp-long", "누적 롱 청산 " + usd(sh.longAll)) + item("lqp-short", "누적 숏 청산 " + usd(sh.shortAll)) +
      tp.tiers.map(function (t, k) { return item("lqp-t" + k, t.name + "배"); }).join("");
  }

  // ── 조작 ───────────────────────────────────────────────────────────────────────────
  function redraw() { if (!S.raf) S.raf = requestAnimationFrame(draw); }
  function priceAt(clientX) {
    var v = S.view, r = $("liqProfileCanvas").getBoundingClientRect();
    return v.lo + (clientX - r.left - v.L) / v.plotW * (v.hi - v.lo);
  }
  function setRange(lo, hi) {
    var v = S.view, m = curMap(), tp = m && m.tier_profile;
    if (!v || !tp) return;
    var span = Math.max(MIN_BINS * tp.bin_width, Math.min(hi - lo, 2 * MAX_RANGE * v.cp));
    var mid = Math.max(v.cp * (1 - MAX_RANGE), Math.min((lo + hi) / 2, v.cp * (1 + MAX_RANGE)));
    S.vlo = mid - span / 2; S.vhi = mid + span / 2; redraw();
  }
  function zoomAt(price, factor) {
    var v = S.view;
    if (v) setRange(price - (price - v.lo) * factor, price + (v.hi - price) * factor);
  }
  function panPx(dx) { var v = S.view; if (v) { var d = dx / v.plotW * (v.hi - v.lo); setRange(v.lo - d, v.hi - d); } }
  function resetView() { S.vlo = S.vhi = null; redraw(); }

  function hoverAt(e) {
    var v = S.view, map = curMap();
    if (!v || !map || !map.tier_profile) return;
    var r = $("liqProfileCanvas").getBoundingClientRect(), x = e.clientX - r.left;
    var i = Math.floor((v.lo + (x - v.L) / v.plotW * (v.hi - v.lo) - v.sh.lo) / v.sh.gw);
    if (x < v.L || x > v.L + v.plotW || i < 0 || i >= v.sh.m) { S.hover = null; tipHide(); redraw(); return; }
    S.hover = i;
    var lo = v.sh.lo + i * v.sh.gw, names = map.tier_profile.tiers.map(function (t) { return t.name; });
    var rows = v.sh.bars[i].map(function (q, k) { return q > 0 ? names[k] + "배 " + usd(q) : null; }).filter(Boolean);
    var dist = ((lo + v.sh.gw / 2) / v.cp - 1) * 100;
    tipShow(e.pageX, e.pageY, "<b>$" + fmtPx(lo) + "–" + fmtPx(lo + v.sh.gw) + "</b> (" + (dist > 0 ? "+" : "") + dist.toFixed(2) + "%)<br>" +
      (rows.length ? rows.join(" · ") + "<br>합 " + usd(v.sh.tot[i]) : "추정 청산 없음") + "<br>" +
      (lo + v.sh.gw <= v.cp ? "여기까지 내려가면 누적 롱 청산 " : "여기까지 오르면 누적 숏 청산 ") + usd(v.sh.cum[i]) + " (추정)");
    redraw();
  }

  function bindCanvas(cv) {
    cv.addEventListener("wheel", function (e) {
      if (!S.view) return;
      e.preventDefault();
      if (Math.abs(e.deltaX) > Math.abs(e.deltaY) || e.shiftKey) panPx(-(e.deltaX || e.deltaY));
      else zoomAt(priceAt(e.clientX), Math.exp(e.deltaY * 0.0015));
    }, { passive: false });
    cv.addEventListener("pointerdown", function (e) {
      if (e.pointerType === "touch" || e.button !== 0 || !S.view) return;
      S.drag = { x: e.clientX, lo: S.view.lo, hi: S.view.hi, moved: false, id: e.pointerId };
    });
    cv.addEventListener("pointermove", function (e) {
      if (S.drag && (e.buttons & 1)) {
        var dx = e.clientX - S.drag.x;
        if (!S.drag.moved && Math.abs(dx) < 4) return;
        if (!S.drag.moved) { S.drag.moved = true; cv.setPointerCapture(S.drag.id); cv.classList.add("panning"); tipHide(); }
        var d = dx / S.view.plotW * (S.drag.hi - S.drag.lo);
        setRange(S.drag.lo - d, S.drag.hi - d);
        return;
      }
      if (e.pointerType !== "touch") hoverAt(e);
    });
    var stop = function () { S.drag = null; cv.classList.remove("panning"); };
    cv.addEventListener("pointerup", stop); cv.addEventListener("pointercancel", stop);
    cv.addEventListener("pointerleave", function () { S.hover = null; tipHide(); redraw(); });
    cv.addEventListener("dblclick", resetView);
    cv.addEventListener("touchstart", function (e) {         // 한 손가락 = 이동(짧게 탭 = 읽기) · 두 손가락 = 확대
      if (!S.view) return;
      if (e.touches.length === 2) {
        var p = e.touches[0], q = e.touches[1];
        S.pinch = { d: Math.max(20, Math.abs(p.clientX - q.clientX)), lo: S.view.lo, hi: S.view.hi, at: priceAt((p.clientX + q.clientX) / 2) };
        S.drag = null;
      } else if (e.touches.length === 1) S.drag = { x: e.touches[0].clientX, lo: S.view.lo, hi: S.view.hi, moved: false, t: e.touches[0] };
    }, { passive: true });
    cv.addEventListener("touchmove", function (e) {
      if (S.pinch && e.touches.length === 2) {
        e.preventDefault();
        var f = S.pinch.d / Math.max(20, Math.abs(e.touches[0].clientX - e.touches[1].clientX));
        setRange(S.pinch.at - (S.pinch.at - S.pinch.lo) * f, S.pinch.at + (S.pinch.hi - S.pinch.at) * f);
      } else if (S.drag && e.touches.length === 1) {
        var dx = e.touches[0].clientX - S.drag.x;
        if (!S.drag.moved && Math.abs(dx) < 6) return;
        e.preventDefault(); S.drag.moved = true;
        var d = dx / S.view.plotW * (S.drag.hi - S.drag.lo);
        setRange(S.drag.lo - d, S.drag.hi - d);
      }
    }, { passive: false });
    cv.addEventListener("touchend", function () {
      if (S.drag && !S.drag.moved && S.drag.t) hoverAt(S.drag.t);
      S.pinch = null; S.drag = null;
    });
    cv.addEventListener("keydown", function (e) {
      var v = S.view, k = e.key;
      if (!v) return;
      var mid = (v.lo + v.hi) / 2;
      if (k === "ArrowLeft" || k === "ArrowRight") panPx((k === "ArrowLeft" ? 0.2 : -0.2) * v.plotW);
      else if (k === "+" || k === "=") zoomAt(mid, 0.8);
      else if (k === "-" || k === "_") zoomAt(mid, 1.25);
      else if (k === "Home") resetView();
      else return;
      e.preventDefault();
    });
  }

  function setView(view) {
    S.on = view === "liq";
    if (typeof window.fpAltSet === "function") window.fpAltSet("liq", S.on);
    var card = $("fpCard"), box = $("liqProfile");
    if (card) card.classList.toggle("fp-liq-on", S.on);
    if (box) box.hidden = !S.on;
    var wt = $("chartWindowTabs"), dt = $("liqDaysTabs");   // 2026-10-10 보기에 따라 오른쪽 탭: 풋프린트 = 창(1h…1d) · 청산맵 = 기간(1d·7d·30d)
    if (wt) wt.hidden = S.on;
    if (dt) dt.hidden = !S.on;
    document.querySelectorAll("#fpViewTabs .asset-tab").forEach(function (b) {
      var on = b.dataset.view === view;
      b.classList.toggle("active", on); b.setAttribute("aria-pressed", String(on));
    });
    try { localStorage.setItem("fpView", view); } catch (e) { /* 저장 못 해도 동작 */ }
    tipHide();
    if (S.on) { fetchExt(); redraw(); }
  }

  function init() {
    var cv = $("liqProfileCanvas");
    if (!cv) return;
    document.querySelectorAll("#fpViewTabs .asset-tab").forEach(function (b) {
      b.addEventListener("click", function () { setView(b.dataset.view); });
    });
    document.querySelectorAll("#liqDaysTabs [data-days]").forEach(function (b) {
      b.addEventListener("click", function () { setDays(Number(b.dataset.days)); });
    });
    document.querySelectorAll("#liqProfileTools [data-view]").forEach(function (b) {
      b.addEventListener("click", function () {
        var v = S.view;
        if (b.dataset.view === "all") resetView();
        else if (v) zoomAt((v.lo + v.hi) / 2, b.dataset.view === "in" ? 0.7 : 1.45);
      });
    });
    bindCanvas(cv);
    new ResizeObserver(redraw).observe(cv);
    setInterval(function () {                       // 새 청산맵(60초)·현재가(1초)·5분 차트 배치가 바뀌면 다시 그린다(보던 범위는 유지)
      if (!S.on) return;
      fetchExt();
      var m = curMap(), tp = m && m.tier_profile, p = tp ? livePrice(tp) : 0, sw = liveTp(m);
      if (m !== S.map || p !== S.price || sw !== S.sw) { S.map = m; S.price = p; S.sw = sw; redraw(); }
      if (typeof window.fpPlaceOverlay === "function") window.fpPlaceOverlay($("liqProfile"));
    }, 1000);
    var saved = null;
    try { saved = localStorage.getItem("fpView"); } catch (e) { /* 없음 */ }
    var d = 1;
    try { d = Number(localStorage.getItem("liqDays")) || 1; } catch (e) { /* 기본 1일 */ }
    setDays(d === 7 || d === 30 ? d : 1);
    setView(saved === "liq" ? "liq" : "fp");
  }

  window.fpLiqActive = function () { return S.on; };   // 시험 하네스가 켜짐을 읽는다
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
