// 2026-11-23 펜 메모 «보기에 틱 -- 차트만 5분할로 100·500·1000·2000·5000틱 차트 · 지표 설정과 SLTP 는 살리고 · 마우스를 대면 다른 차트도 같은 시각에».
// 자리 = 체결 버블·일봉과 같은 5분 차트 교체 영역(fp_daily.js fpPlaceOverlay · fpAltSet «tick») · 보기 토글 = liq_profile.js setView 의 «fpview» 이벤트.
// 원천 = 서버 /api/tick-bars(바이낸스 선물 실시간 개별 체결 n 개마다 봉 하나 -- coin_flow tick_add) · 1초마다 새 봉만(since).
// 그림(칸마다): 캔들 · 아래 거래량(색 = 델타 부호) · 켠 이동평균(지표 팝업의 EMA 5·20 · SMMA 21·50 · 볼린저 -- 5분 차트와 같은 스위치·색) ·
//   가격선(지금가 · 진입가 · 무장된 익절·손절 -- 계좌 카드와 같은 값, 읽기 전용) · 커서 = 그 체결 시각을 다섯 칸이 함께 따라간다.
// 2026-11-24 사용자 «SLTP 주문도 틱 차트에서 · 청산맵·호가벽·지지/저항도 틱 차트에»:
//   주문 = «주문 칸» 하나(칸 제목을 누르면 바뀜 · 기본 2000틱)가 app.js 선 주문 층의 자리가 된다(window.fpTickFrame → ordFrame) -- 선 도구·들어간 선·초안·peg 꼬리표가
//   5분 차트와 똑같이(같은 함수·같은 서버 경로 = rtorder) 그 칸 위에 뜬다. 다른 칸은 그 선을 읽기 전용으로 그린다.
//   청산맵 = 최신 추정 청산 밀도(/api/liquidation-map heatmap_history 마지막 판)를 가격 줄 배경으로 · 호가벽 = 지금 호가 깊이(/api/flow/heatmap book)를 오른쪽 끝 막대로(큰 벽 둘은 가로선) ·
//   지지/저항 = 5분 차트·게이지와 같은 srLevelsLive(청산맵 밀집 · 쓸린 레벨 뺌) 점선. 셋 다 새 계산 없이 이미 받는 값을 그린다.
// 🔴CI esprima: `?.(`·`?.[`·숫자 구분자 금지.
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var SIZES = [100, 500, 1000, 2000, 5000];
  var S = { on: false, asset: "", d: {}, busy: false, raf: 0, hover: null, err: "", ord: 2000 };   // d[n] = { k, rows: [..닫힌 봉], cur } · ord = 주문 칸
  try { var o0 = Number(localStorage.getItem("tickOrd")); if (SIZES.indexOf(o0) >= 0) S.ord = o0; } catch (e) { /* 기억은 편의 */ }
  var asset = function () { return typeof activeSnapshotAsset === "undefined" ? "eth" : activeSnapshotAsset; };
  var css = function (v) {   // "var(--x)" → 실제 색(캔버스는 CSS 변수를 못 읽는다)
    var m = /^var\((--[\w-]+)\)$/.exec(String(v || "").trim());
    return m ? getComputedStyle(document.documentElement).getPropertyValue(m[1]).trim() : v;
  };
  var dp = function () { return typeof pxDp === "function" ? pxDp() : 2; };
  function hms(ms) { var d = new Date(ms); return [d.getHours(), d.getMinutes(), d.getSeconds()].map(function (x) { return String(x).padStart(2, "0"); }).join(":"); }
  function dur(ms) { var s = Math.max(0, Math.round(ms / 1000)); return s < 90 ? s + "초" : Math.round(s / 60) + "분"; }
  function qty(v) { var a = Math.abs(v); return (v < 0 ? "−" : "+") + (a >= 1000 ? (a / 1000).toFixed(1) + "k" : a.toFixed(a >= 10 ? 0 : 1)); }

  function reset(a) { S.asset = a; S.d = {}; S.hover = null; S.err = ""; }
  function poll() {
    var a = asset();
    if (a !== S.asset) reset(a);
    if (!S.on || S.busy || document.hidden) return;
    var since = SIZES.map(function (n) { var x = S.d[n]; return x && x.rows.length ? x.rows[x.rows.length - 1][0] + 1 : ""; }).join(",");
    S.busy = true;
    fetch("/api/tick-bars?asset=" + encodeURIComponent(a) + (since.replace(/,/g, "") ? "&since=" + since : ""), { cache: "no-store" })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function (j) {
        if (a !== S.asset) return;
        SIZES.forEach(function (n) {
          var t = (j.ticks || {})[n];
          if (!t) return;
          var x = S.d[n] || (S.d[n] = { rows: [] }), last = x.rows.length ? x.rows[x.rows.length - 1][0] : -1;
          t.bars.forEach(function (r) { if (r[0] > last) x.rows.push(r); });
          if (x.rows.length > 240) x.rows.splice(0, x.rows.length - 240);
          x.cur = t.cur; x.k = t.k;
        });
        S.err = ""; redraw();
      })
      .catch(function (e) { S.err = String(e && e.message || e); redraw(); })
      .then(function () { S.busy = false; });
  }
  function rowsOf(n) { var x = S.d[n]; return x ? (x.cur ? x.rows.concat([x.cur]) : x.rows) : []; }

  // 지표 = 5분 차트와 같은 계산(app.js mcEma·mcSmma·mcBollinger -- {time, close} 배열) · 같은 스위치(mcLine)·같은 색(lineSt)
  function indicators(rows) {
    if (typeof mcLine === "undefined" || typeof lineSt !== "function") return [];
    var cs = rows.map(function (r) { return { time: r[0], close: r[6] }; }), out = [];
    [["ema5", function () { return mcEma(cs, 5); }], ["ema20", function () { return mcEma(cs, 20); }],
     ["smma21", function () { return mcSmma(cs, 21); }], ["smma50", function () { return mcSmma(cs, 50); }]].forEach(function (d) {
      if (mcLine[d[0]] && typeof window[d[0].indexOf("smma") === 0 ? "mcSmma" : "mcEma"] !== "undefined") {
        var S0 = lineSt(d[0]); out.push({ m: d[1](), c: css(S0.c), w: S0.w, dash: S0.d });
      }
    });
    if (mcLine.bb && typeof mcBollinger === "function") {
      var bb = mcBollinger(cs), c = css(lineSt("bb").c);
      [0, 2].forEach(function (j) { var m = new Map(); bb.forEach(function (v, t) { m.set(t, v[j]); }); out.push({ m: m, c: c, w: 1, dash: "3 3", a: 0.55 }); });
    }
    return out;
  }
  // 가격선: 지금가 · 진입가 · 무장된 익절·손절(app.js 의 계좌·무장 값 그대로 -- 5분 차트 선과 같은 출처)
  function priceLines() {
    var L = [], p = typeof latestLivePriceByAsset === "undefined" ? 0 : Number(latestLivePriceByAsset[asset()] || 0);
    try {
      var pos = typeof snapshotAccountPosition === "function" ? snapshotAccountPosition() : null;
      var e = pos && typeof entryUsdt === "function" ? entryUsdt(pos, p) : 0;
      if (e > 0) L.push({ v: e, c: css("var(--amber)") || css("var(--warn)"), t: "진입", dash: [] });
      var a = typeof armedLinesNow === "function" ? armedLinesNow() : null;
      if (a && a.b) {
        if (a.b.tp_level > 0) L.push({ v: +a.b.tp_level, c: css("var(--good)"), t: "익절", dash: [6, 4] });
        if (a.b.sl_level > 0) L.push({ v: +a.b.sl_level, c: css("var(--bad)"), t: "손절", dash: [6, 4] });
      }
    } catch (err) { /* 계좌 없으면 선 없음 */ }
    return { now: p, lines: L };
  }

  function panelGeo(cv) {
    var W = cv.clientWidth, H = cv.clientHeight, narrow = W < 420;
    var R = narrow ? 50 : 84, top = 22, volH = Math.max(18, Math.round(H * 0.16)), axisH = 14;
    return { W: W, H: H, L: 6, R: R, top: top, plotW: W - 6 - R, priceH: H - top - volH - axisH - 6, volY: H - volH - axisH, volH: volH, axisY: H - axisH + 10, narrow: narrow };
  }
  // 보이는 봉 = 칸 폭에 맞춘 최근 봉(봉당 7px 안팎 · 최소 30봉) -- 오른쪽 끝 = 형성 봉
  function view(n, g) {
    var rows = rowsOf(n), cnt = Math.max(30, Math.floor(g.plotW / 7)), v = rows.slice(-cnt);
    var lo = Infinity, hi = -Infinity;
    v.forEach(function (r) { lo = Math.min(lo, r[5]); hi = Math.max(hi, r[4]); });
    var pad = (hi - lo) * 0.14 || 1;   // 2026-11-25 펜 메모 «위아래 여유 -- 고가·저가 자리가 없다»: 범위의 14% 씩(봉 정보 줄·가격 꼬리표가 고가·저가를 안 덮게)
    return { rows: v, cnt: cnt, lo: lo - pad, hi: hi + pad, slot: g.plotW / cnt };
  }
  // 그 시각(ms)을 품은 봉 = 첫 체결이 그 시각 이하인 마지막 봉 -- 다섯 칸 커서 맞추기(보이는 첫 봉보다 앞이면 없음)
  function idxAt(rows, t) {
    var a = 0, b = rows.length - 1, k = -1;
    while (a <= b) { var m = (a + b) >> 1; if (rows[m][1] <= t) { k = m; a = m + 1; } else b = m - 1; }
    return k;
  }

  // 청산맵(배경) · 호가벽(오른쪽 막대 · 큰 벽 가로선) · 지지/저항(점선) -- 가격만 보는 층이라 다섯 칸이 같은 값을 칸마다 제 축으로
  function overlays(ctx, g, V, Y, D) {
    var inY = function (p) { return p >= V.lo && p <= V.hi; }, good = css("var(--good)"), bad = css("var(--bad)");
    try {   // 청산맵: 최신 판의 칸 밀도 -- 보이는 범위 최대로 정규화 · 5분 차트 배경과 같은 색(densityColor)
      var hist = typeof liquidationDensityHistory === "function" ? liquidationDensityHistory() : [], last = hist.length ? hist[hist.length - 1] : null;
      if (last && last.bins && last.binWidth > 0 && typeof densityColor === "function") {
        var vis = last.bins.filter(function (b) { return b.price + last.binWidth / 2 >= V.lo && b.price - last.binWidth / 2 <= V.hi && b.weightPct > 0; });
        var mx = Math.max.apply(null, vis.map(function (b) { return b.weightPct; }).concat([1e-9]));
        ctx.save(); ctx.globalAlpha = 0.32;   // 배경이라 옅게(캔들이 먼저 읽히게)
        vis.forEach(function (b) {
          var y1 = Y(b.price + last.binWidth / 2), y2 = Y(b.price - last.binWidth / 2);
          ctx.fillStyle = densityColor(b.weightPct / mx); ctx.fillRect(g.L, Math.max(g.top, y1), g.plotW, Math.max(1, Math.min(g.top + g.priceH, y2) - Math.max(g.top, y1)));
        });
        ctx.restore();
      }
    } catch (e) { /* 청산맵 없으면 배경 없음 */ }
    try {   // 호가벽: 호가 깊이(매수 +·매도 −) -- 보이는 행 최대로 정규화 · 오른쪽 끝에서 안쪽으로(플롯의 22%) · 쪽마다 가장 큰 벽 하나는 가로선 + 수량
      var fh = typeof latestFlowHeatmap === "undefined" ? null : latestFlowHeatmap, bk = fh && (fh.asset == null || fh.asset === asset()) ? fh.book : null;   // 다른 코인 응답이면 벽 없음
      if (bk && bk.q && bk.bin_size > 0) {
        var rows = [], i, maxQ = 0;
        for (i = 0; i < bk.q.length; i++) { var v = bk.q[i], pr = (bk.bin_lo + i + 0.5) * bk.bin_size; if (v && inY(pr)) { rows.push([pr, v]); maxQ = Math.max(maxQ, Math.abs(v)); } }
        var W = g.plotW * 0.16, hh = Math.max(1, Math.abs(Y(0) - Y(bk.bin_size)) - 0.5), top2 = { b: null, a: null };
        rows.forEach(function (r) {
          var len = Math.abs(r[1]) / (maxQ || 1) * W, bid = r[1] > 0, key = bid ? "b" : "a";
          ctx.globalAlpha = 0.28; ctx.fillStyle = bid ? good : bad; ctx.fillRect(g.L + g.plotW - len, Y(r[0]) - hh / 2, len, hh);
          if (!top2[key] || Math.abs(r[1]) > Math.abs(top2[key][1])) top2[key] = r;
        });
        ctx.globalAlpha = 1;
        ["b", "a"].forEach(function (k) {
          var r = top2[k]; if (!r) return;
          var yy = Math.round(Y(r[0])) + 0.5, c = k === "b" ? good : bad;
          ctx.save(); ctx.strokeStyle = c; ctx.globalAlpha = 0.7; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(g.L, yy); ctx.lineTo(g.L + g.plotW, yy); ctx.stroke(); ctx.restore();
          ctx.fillStyle = c; ctx.textAlign = "right"; ctx.textBaseline = "bottom"; ctx.fillText("벽 " + qty(Math.abs(r[1])).slice(1), g.L + g.plotW - 2, yy - 2);
        });
      }
    } catch (e) { /* 호가 없으면 벽 없음 */ }
    try {   // 지지/저항: 가까운 셋씩
      var sr = typeof srLevelsLive === "function" ? srLevelsLive(3) : null;
      if (sr) [["res", bad, "R"], ["sup", good, "S"]].forEach(function (d) {
        (sr[d[0]] || []).forEach(function (lv, j) {
          var pv = Number(lv.price); if (!(pv > 0) || !inY(pv)) return;
          var yy = Math.round(Y(pv)) + 0.5;
          ctx.save(); ctx.strokeStyle = d[1]; ctx.globalAlpha = 0.75 - j * 0.15; ctx.setLineDash([4, 4]); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(g.L, yy); ctx.lineTo(g.L + g.plotW, yy); ctx.stroke(); ctx.restore();
          ctx.fillStyle = d[1]; ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.fillText(d[2] + (j + 1) + " " + pv.toFixed(D), g.L + 2, yy - 2);
        });
      });
    } catch (e) { /* 레벨 없음 */ }
  }

  // 2026-11-25 펜 메모 «틱 차트마다 보이는 전체 구간의 POC»: 보이는 봉(형성 봉 포함)의 가격별 체결량 합이 가장 큰 가격 -- rtcore 가 봉마다 싣는 [[가격, 수량]…](풋프린트와 같은 칸 폭)
  function pocOf(rows) {
    var vol = new Map(), best = null, bv = -1;
    rows.forEach(function (r) { (r[10] || []).forEach(function (l) { var k = l[0], v = (vol.get(k) || 0) + Number(l[1] || 0); vol.set(k, v); if (v > bv) { bv = v; best = k; } }); });
    return best == null ? null : { px: Number(best), vol: bv };
  }
  function pocLine(ctx, g, V, Y, D, rows) {
    var p = pocOf(rows);
    if (!p || p.px < V.lo || p.px > V.hi) return;
    var yy = Math.round(Y(p.px)) + 0.5, c = css("var(--amber)") || css("var(--warn)");
    ctx.save(); ctx.strokeStyle = c; ctx.lineWidth = 1.6; ctx.globalAlpha = 0.9; ctx.beginPath(); ctx.moveTo(g.L, yy); ctx.lineTo(g.L + g.plotW, yy); ctx.stroke(); ctx.restore();
    var s2 = "POC " + p.px.toFixed(D), w = ctx.measureText(s2).width + 8;
    ctx.fillStyle = c; ctx.fillRect(g.L + g.plotW + 2, yy - 8, Math.min(w, g.R - 2), 16);
    ctx.fillStyle = css("var(--on-fill)"); ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(s2, g.L + g.plotW + 5, yy);
  }

  function drawPanel(n, cell) {
    var cv = cell.querySelector("canvas");
    if (!cv || !cv.clientWidth || !cv.clientHeight) return;
    var g = panelGeo(cv), dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(g.W * dpr) || cv.height !== Math.round(g.H * dpr)) { cv.width = Math.round(g.W * dpr); cv.height = Math.round(g.H * dpr); }
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, g.W, g.H);
    var muted = css("var(--muted)"), ink = css("var(--text)"), good = css("var(--good)"), bad = css("var(--bad)"), line = css("var(--line)");
    var V = view(n, g), rows = V.rows, D = dp();
    var head = cell.querySelector(".tkc-sub");
    var x = S.d[n];
    if (!rows.length) {
      if (head) head.textContent = S.err ? "불러오지 못함" : "쌓는 중";
      ctx.fillStyle = muted; ctx.font = "12px " + getComputedStyle(cv).fontFamily; ctx.textAlign = "center";
      ctx.fillText(S.err ? "틱 봉을 불러오지 못했습니다 (" + S.err + ")" : "체결을 세는 중 -- " + n + "체결마다 봉 하나", g.W / 2, g.H / 2);
      cell._geo = null;
      return;
    }
    var spanMs = rows.length > 1 ? rows[rows.length - 1][1] - rows[0][1] : 0;
    if (head) head.textContent = rows.length + "봉 · " + dur(spanMs) + (x && x.cur ? " · 형성 " + x.cur[9] + "/" + n : "");
    var Y = function (p) { return g.top + (V.hi - p) / (V.hi - V.lo) * g.priceH; };
    var X = function (i) { return g.L + (i + 0.5) * V.slot + (V.cnt - rows.length) * V.slot; };
    var bw = Math.max(1, Math.min(V.slot * 0.68, 14));
    // 가격 눈금(오른쪽) -- 4줄
    ctx.font = (g.narrow ? 10 : 11) + "px " + getComputedStyle(cv).fontFamily; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    for (var q = 0; q <= 3; q++) {
      var pv = V.lo + (V.hi - V.lo) * (0.1 + q * 0.8 / 3), yy = Math.round(Y(pv)) + 0.5;
      ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(g.L, yy); ctx.lineTo(g.L + g.plotW, yy); ctx.stroke();
      ctx.fillStyle = muted; ctx.fillText(pv.toFixed(D), g.L + g.plotW + 6, yy);
    }
    overlays(ctx, g, V, Y, D);
    pocLine(ctx, g, V, Y, D, rows);
    // 거래량(색 = 델타 부호)
    var vMax = Math.max.apply(null, rows.map(function (r) { return r[7]; })) || 1;
    rows.forEach(function (r, i) {
      var h = Math.max(1, r[7] / vMax * (g.volH - 4));
      ctx.globalAlpha = 0.5; ctx.fillStyle = r[8] >= 0 ? good : bad; ctx.fillRect(X(i) - bw / 2, g.volY + g.volH - h, bw, h);
    });
    ctx.globalAlpha = 1;
    // 캔들
    rows.forEach(function (r, i) {
      var cx = Math.round(X(i)) + 0.5, up = r[6] >= r[3], col = up ? good : bad, y1 = Y(Math.max(r[3], r[6])), y2 = Y(Math.min(r[3], r[6]));
      ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx, Y(r[4])); ctx.lineTo(cx, Y(r[5])); ctx.stroke();
      ctx.fillStyle = col; ctx.fillRect(cx - bw / 2, y1, bw, Math.max(1, y2 - y1));
    });
    // 지표(5분 차트와 같은 스위치·색)
    indicators(rowsOf(n)).forEach(function (L) {
      ctx.save(); ctx.strokeStyle = L.c; ctx.lineWidth = L.w || 1.4; ctx.globalAlpha = L.a || 0.9;
      ctx.setLineDash(L.dash && L.dash !== "none" ? String(L.dash).split(/\s+/).map(Number) : []);
      ctx.beginPath(); var st = false;
      rows.forEach(function (r, i) { var v = L.m.get(r[0]); if (v == null) { st = false; return; } var px = X(i), py = Y(v); if (st) ctx.lineTo(px, py); else { ctx.moveTo(px, py); st = true; } });
      ctx.stroke(); ctx.restore();
    });
    // 가격선
    var PL = priceLines(), tag = function (v, c, t, dash) {
      if (!(v > 0)) return;
      var yy = Y(v), off = yy < g.top ? "↑" : yy > g.top + g.priceH ? "↓" : "";
      yy = Math.max(g.top, Math.min(g.top + g.priceH, yy));
      ctx.save(); ctx.strokeStyle = c; ctx.setLineDash(dash); ctx.lineWidth = 1.2; ctx.globalAlpha = off ? 0.45 : 0.9;
      ctx.beginPath(); ctx.moveTo(g.L, Math.round(yy) + 0.5); ctx.lineTo(g.L + g.plotW, Math.round(yy) + 0.5); ctx.stroke(); ctx.restore();
      var s = (t ? t + " " : "") + off + v.toFixed(D), w = ctx.measureText(s).width + 8;
      ctx.fillStyle = c; ctx.fillRect(g.L + g.plotW + 2, yy - 8, Math.min(w, g.R - 2), 16);
      ctx.fillStyle = css("var(--on-fill)"); ctx.textAlign = "left"; ctx.fillText(s, g.L + g.plotW + 5, yy);
    };
    if (n !== S.ord) PL.lines.forEach(function (l) { tag(l.v, l.c, g.narrow ? "" : l.t, l.dash); });   // 주문 칸은 app.js 선 층(끌 수 있는 선)이 그린다
    tag(PL.now, ink, "", [2, 3]);
    // 시각 눈금(아래) -- 처음·가운데·끝
    ctx.fillStyle = muted; ctx.textBaseline = "alphabetic";
    [[0, "left"], [Math.floor(rows.length / 2), "center"], [rows.length - 1, "right"]].forEach(function (p) {
      ctx.textAlign = p[1]; ctx.fillText(hms(rows[p[0]][1]), Math.max(g.L, Math.min(g.L + g.plotW, X(p[0]))), g.axisY);
    });
    cell._geo = { g: g, V: V, X: X, Y: Y };
    // 커서: 다섯 칸이 같은 시각
    var h = S.hover;
    if (h) {
      var k = idxAt(rows, h.t);
      if (k >= 0) {
        var cx2 = Math.round(X(k)) + 0.5, r = rows[k];
        ctx.save(); ctx.strokeStyle = ink; ctx.globalAlpha = 0.55; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(cx2, g.top); ctx.lineTo(cx2, g.volY + g.volH); ctx.stroke(); ctx.restore();
        if (h.n === n && h.y != null && h.y >= g.top && h.y <= g.top + g.priceH) {
          var pv2 = V.hi - (h.y - g.top) / g.priceH * (V.hi - V.lo);
          ctx.save(); ctx.strokeStyle = ink; ctx.globalAlpha = 0.45; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(g.L, h.y + 0.5); ctx.lineTo(g.L + g.plotW, h.y + 0.5); ctx.stroke(); ctx.restore();
          ctx.fillStyle = ink; ctx.fillRect(g.L + g.plotW + 2, h.y - 8, g.R - 4, 16); ctx.fillStyle = css("var(--on-fill)"); ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(pv2.toFixed(D), g.L + g.plotW + 5, h.y);
        }
        // 봉 정보 한 줄(칸 위)
        var info = hms(r[1]) + "  시 " + r[3].toFixed(D) + " 고 " + r[4].toFixed(D) + " 저 " + r[5].toFixed(D) + " 종 " + r[6].toFixed(D) + "  델타 " + qty(r[8]) + "  " + dur(r[2] - r[1]);
        ctx.font = (g.narrow ? 10 : 11) + "px " + getComputedStyle(cv).fontFamily; ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";
        ctx.fillStyle = r[6] >= r[3] ? good : bad; ctx.fillText(g.narrow ? hms(r[1]) + "  " + r[6].toFixed(D) + "  " + qty(r[8]) : info, g.L + 2, 14);
      }
    }
  }

  function redraw() { if (!S.raf) S.raf = requestAnimationFrame(draw); }
  function draw() {
    S.raf = 0;
    if (!S.on) return;
    var box = $("tickCharts");
    if (typeof window.fpPlaceOverlay === "function") window.fpPlaceOverlay(box);
    var st = $("tickChartsStatus");
    if (st) st.textContent = S.err ? "틱 봉 지연 (" + S.err + ")" : "";
    Array.prototype.forEach.call(document.querySelectorAll("#tickCharts .tkc-cell"), function (cell) {
      drawPanel(+cell.dataset.n, cell);
      cell.classList.toggle("tkc-ord", +cell.dataset.n === S.ord);
    });
    // 주문 층(app.js)을 이 칸의 새 가격 축에 다시 맞춘다
    ["renderLineTool", "renderArmedLines"].forEach(function (f) { if (typeof window[f] === "function") try { window[f](); } catch (e) { /* 층 하나가 실패해도 그림은 그대로 */ } });
  }
  // app.js ordFrame 이 읽는 주문 칸의 자리 -- chartGeo 와 같은 이름(G: 플롯 위·아래·가격 범위·왼쪽·폭 · m: 칸 좌표 → 화면 좌표)
  window.fpTickFrame = function () {
    if (!S.on) return null;
    var cell = document.querySelector('#tickCharts .tkc-cell[data-n="' + S.ord + '"]'), G0 = cell && cell._geo;
    if (!G0) return null;
    var r = cell.querySelector("canvas").getBoundingClientRect(), g = G0.g, V = G0.V;
    return { G: { mt: g.top, pb: g.top + g.priceH, ymax: V.hi, ymin: V.lo, ml: g.L, cw: g.plotW }, m: { a: 1, d: 1, e: r.left, f: r.top } };
  };

  function bind(cell) {
    var cv = cell.querySelector("canvas"), n = +cell.dataset.n;
    var at = function (e) {
      var G = cell._geo, b = cv.getBoundingClientRect();
      if (!G) return null;
      var xx = e.clientX - b.left, yy = e.clientY - b.top, rows = G.V.rows;
      var i = Math.round((xx - G.g.L) / G.V.slot - 0.5 - (G.V.cnt - rows.length));
      if (i < 0 || i >= rows.length) return null;
      return { t: rows[i][1], n: n, y: yy };
    };
    cv.addEventListener("pointermove", function (e) { S.hover = at(e); redraw(); });
    cv.addEventListener("pointerleave", function () { S.hover = null; redraw(); });
  }

  function setOn(on) {
    S.on = on;
    var card = $("fpCard");
    if (card) card.classList.toggle("fp-tick-on", on);
    if (typeof window.fpAltSet === "function") window.fpAltSet("tick", on);
    var box = $("tickCharts");
    if (box) box.hidden = !on;
    if (on) { poll(); redraw(); }
  }
  window.addEventListener("fpview", function (e) { setOn(e.detail === "tick"); });
  window.fpTickActive = function () { return S.on; };
  function init() {
    Array.prototype.forEach.call(document.querySelectorAll("#tickCharts .tkc-cell"), bind);
    setInterval(poll, 1000);
    window.addEventListener("resize", redraw);
    // 지표 스위치·계좌(무장선)·지금가가 바뀌면 다시 그린다 -- 5분 차트가 다시 그려질 때 같이(가볍다: 칸 다섯 · 봉 수백)
    var old = window.scheduleSnapshotChartRender;
    if (typeof old === "function") window.scheduleSnapshotChartRender = function () { if (S.on) redraw(); return old.apply(this, arguments); };
    // 주문 칸 고르기: 칸 제목(«2000틱 · 주문»)을 누르면 그 칸이 선 주문 자리가 된다(기억)
    Array.prototype.forEach.call(document.querySelectorAll("#tickCharts .tkc-t"), function (t) {
      t.setAttribute("role", "button"); t.tabIndex = 0;
      var pick = function () { var n = +t.parentElement.dataset.n; if (n === S.ord) return; S.ord = n; try { localStorage.setItem("tickOrd", String(n)); } catch (e) { /* 편의 */ } redraw(); };
      t.addEventListener("click", pick);
      t.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
