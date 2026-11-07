// 풋프린트 카드 «청산» 보기 -- 2026-11-07 사용자 시안 I: 남은 매물(전체) → 가격 사다리(±6% 확대) · 2026-11-08 누적 곡선은 풋프린트 청산 기둥으로 옮김(fpLiqCurve) ·
// 아래 줄 = 오늘·24시간 + 자기 흥분 지수. 보기 토글은 liq_profile.js setView, 이 파일은 그린다.
// 원천: 지도 = fpLiqColTp(1·7·30일, 지나간 칸 지움) · HL 고래·호가·60초 = latestMarketCtx · 5분봉 = /api/liquidation-5m-history ·
//   남은 매물·띠·실측 위치·탄력·자기 흥분 = /api/liq-tab(cron scripts/live_liq_tab_20261107.py, 매시).
// 지도·남은 매물은 가상 포지션 추정이다. 금액 눈금(OI×2)은 절대액이 아니다 -- 흰 선이 실제 터진 양. 전부 서술이고 신호가 아니다.
// 🔴CI esprima: `?.(`·`?.[`·숫자 구분자 금지.
(function () {
  "use strict";
  var LADDER_R = 6, FINE = 400, BAR_POLL_MS = 30000, DATA_POLL_MS = 600000, TICK_MS = 5000, WINS = [30, 60, 90, 180], STACK_MQ = "(max-width:1100px)";
  var S = { on: false, asset: null, win: 60, data: null, dataT: 0, bars: [], barsT: 0, built: "" };
  try { var saved = Number(localStorage.getItem("liqTabWin")); if (WINS.indexOf(saved) >= 0) S.win = saved; } catch (e) { /* 저장 못 해도 동작 */ }
  var $ = function (id) { return document.getElementById(id); };
  var asset = function () { return typeof activeSnapshotAsset === "undefined" ? "eth" : activeSnapshotAsset; };
  var ctx = function () { return (typeof latestMarketCtx !== "undefined" && latestMarketCtx && latestMarketCtx.asset === asset()) ? latestMarketCtx : null; };
  function price() {
    var p = typeof latestLivePriceByAsset === "undefined" ? 0 : Number(latestLivePriceByAsset[asset()] || 0);
    if (p > 0) return p;
    var c = ctx();
    return c && c.mid > 0 ? c.mid : S.data ? S.data.price : 0;
  }
  var usd = function (x) { if (x == null || x !== x) return "—"; var a = Math.abs(x), s = x < 0 ? "−" : ""; if (a >= 1e9) return s + "$" + (a / 1e9).toFixed(a >= 1e10 ? 0 : 1) + "B"; if (a >= 1e6) return s + "$" + (a / 1e6).toFixed(a >= 1e8 ? 0 : 1) + "M"; if (a >= 1e3) return s + "$" + (a / 1e3).toFixed(0) + "K"; return s + "$" + a.toFixed(0); };
  var sgn = function (x, d) { return x == null ? "—" : (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d || 0); };
  var pc = function (x, d) { return (x * 100).toFixed(d || 0) + "%"; };
  var px = function (p) { var d = p >= 1000 ? 0 : p >= 100 ? 1 : p >= 1 ? 2 : 4; return Number(p).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }); };
  var hhmm = function (t) { var k = ((t + 32400) % 86400 + 86400) % 86400; return String(Math.floor(k / 3600)).padStart(2, "0") + ":" + String(Math.floor(k % 3600 / 60)).padStart(2, "0"); };   // 화면 시각 = KST
  var niceStep = function (v, n) { var r = v / (n || 4), m = Math.pow(10, Math.floor(Math.log10(r || 1))); return [1, 2, 2.5, 5, 10].map(function (k) { return k * m; }).find(function (k) { return k >= r; }); };
  var ticks = function (mx, n) { var st = niceStep(mx, n), o = []; for (var v = st; v <= mx * 1.001; v += st) o.push(v); return o; };
  var svg = function (w, h, inner, label) { return '<svg width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + " " + h + '" role="img" aria-label="' + label + '">' + inner + "</svg>"; };
  var path = function (a) { return a.map(function (p, i) { return (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); }).join(""); };
  var line = function (cl, x1, x2, y1, y2) { return '<line class="' + cl + '" x1="' + x1 + '" x2="' + x2 + '" y1="' + y1 + '" y2="' + y2 + '"/>'; };
  var txt = function (cl, x, y, s, anc) { return '<text class="' + cl + '" x="' + x + '" y="' + y + '"' + (anc ? ' text-anchor="' + anc + '"' : "") + ">" + s + "</text>"; };

  // ── 데이터 ──────────────────────────────────────────────────────────────
  async function fetchData(force) {
    var a = asset(), now = Date.now();
    if (!force && now - S.dataT < DATA_POLL_MS) return;
    S.dataT = now;
    try {
      var r = await fetch("/api/liq-tab?asset=" + encodeURIComponent(a), { cache: "no-cache" });
      var j = r.ok ? await r.json() : null;
      if (asset() === a) { S.data = j && j.ok ? j : null; S.heatKey = ""; draw(); }
    } catch (e) { /* 다음 주기에 다시 -- 옛 값을 지우지 않는다 */ }
  }
  async function fetchBars(force) {
    var a = asset(), now = Date.now();
    if (!force && now - S.barsT < BAR_POLL_MS) return;
    S.barsT = now;
    try {
      var r = await fetch("/api/liquidation-5m-history?asset=" + encodeURIComponent(a) + "&bars=288", { cache: "no-cache" });
      var j = r.ok ? await r.json() : null;
      if (asset() === a && j && Array.isArray(j.bars)) { S.bars = j.bars.map(toBar); draw(); }
    } catch (e) { /* 위와 같다 */ }
  }
  function toBar(b) {
    var ok = b.okx_detail || {}, by = b.bybit || {}, hl = b.hl || {};
    return { t: Date.parse(b.ts) / 1000, l: b.long_usd || 0, s: b.short_usd || 0, n: b.events || 0,
             ok: (ok.long_usd || 0) + (ok.short_usd || 0), by: (by.long_usd || 0) + (by.short_usd || 0), hl: (hl.long_usd || 0) + (hl.short_usd || 0) };
  }
  // 오늘(UTC 일봉) 5분 누적. 비교 띠와 같은 범위(바이낸스+OKX)로 맞추고, 합계는 전 거래소로 따로 낸다.
  function today() {
    var d0 = Math.floor(Date.now() / 86400000) * 86400, cum = [], acc = 0, first = null, L = 0, Sh = 0, N = 0;
    S.bars.forEach(function (b) {
      if (b.t < d0) return;
      if (first == null) first = b.t;
      acc += b.l + b.s - b.by - b.hl; L += b.l; Sh += b.s; N += b.n;
      cum[Math.floor((b.t - d0) / 300)] = acc;
    });
    for (var i = 1; i < cum.length; i++) if (cum[i] == null) cum[i] = cum[i - 1];
    return { d0: d0, cum: cum, l: L, s: Sh, n: N, tot: L + Sh, first: first };
  }
  // 지도 → 0.25% 사다리·0.1% 누적(±40% -- 4h 창도 화면 끝까지). 지나간 칸은 fpLiqColTp 가 이미 지웠다.
  function mapOf(days) {
    var tp = typeof window.fpLiqColTp === "function" ? window.fpLiqColTp(days) : null, P = price();
    if (!tp || !tp.tiers || !(P > 0)) return null;
    var rows = {}, fineDn = new Float64Array(FINE + 1), fineUp = new Float64Array(FINE + 1);
    tp.tiers.forEach(function (t, ti) {
      t.values.forEach(function (v, k) {
        if (!(v > 0)) return;
        var d = ((tp.lo + k) * tp.bin_width / P - 1) * 100;
        if (Math.abs(d) > FINE / 10 + 0.1) return;
        var g = Math.round(d / 0.25);
        if (g !== 0) { var r = rows[g] || (rows[g] = [0, 0, 0, 0]); r[ti] += v; }
        var f = Math.min(FINE, Math.ceil(Math.abs(d) / 0.1));
        if (d < 0) fineDn[f] += v; else fineUp[f] += v;
      });
    });
    for (var i = 1; i <= FINE; i++) { fineDn[i] += fineDn[i - 1]; fineUp[i] += fineUp[i - 1]; }
    return { rows: rows, dn: fineDn, up: fineUp };
  }
  var cumAt = function (m, d) { var i = Math.min(FINE, Math.round(Math.abs(d) / 0.1)); return d < 0 ? m.dn[i] : m.up[i]; };
  function realRows(P) {   // 7일 실측을 지금 가격의 ±0.25% 칸으로
    var o = {};
    ((S.data && S.data.real) || []).forEach(function (r) { var g = Math.round((r[0] / P - 1) * 100 / 0.25), a = o[g] || (o[g] = [0, 0]); a[0] += r[1]; a[1] += r[2]; });
    return o;
  }

  // ── 차트. 각 함수는 (el, w, h, opt) → svg 문자열(없으면 null) · el._hit = 호버 글 ──────────
  var CH = {};
  CH.today = function (el, w, h) {
    var T = today(), B = S.data && S.data.band, L = 46, R = 8, Tp = 6, Bt = 15;
    if (!T.cum.length && !B) return null;
    var mx = Math.max.apply(null, (B ? B.p75 : [0]).concat(T.cum, [1])) * 1.06, X = function (i) { return L + (w - L - R) * (i + 1) / 288; }, Y = function (v) { return Tp + (h - Tp - Bt) * (1 - v / mx); }, s = "";
    ticks(mx, 3).forEach(function (v) { s += line("lq-gl", L, w - R, Y(v), Y(v)) + txt("lq-ax", L - 5, Y(v) + 3, usd(v), "end"); });
    if (B) {
      s += '<path class="lq-bd" d="' + path(B.p75.map(function (v, i) { return [X(i), Y(v)]; })) + "L" + B.p25.map(function (v, i) { return [X(i), Y(v)]; }).reverse().map(function (p) { return p[0].toFixed(1) + " " + p[1].toFixed(1); }).join("L") + 'Z"/>';
      s += '<path class="lq-ln2" d="' + path(B.p50.map(function (v, i) { return [X(i), Y(v)]; })) + '"/>';
    }
    if (T.cum.length) {
      var pts = []; T.cum.forEach(function (v, i) { if (v != null) pts.push([X(i), Y(v)]); });
      var e = pts[pts.length - 1], lst = T.cum[T.cum.length - 1];
      s += '<path class="lq-ln" d="' + path(pts) + '"/><circle class="lq-dt" cx="' + e[0] + '" cy="' + e[1] + '" r="3.5"/>' + txt("lq-ax t o", e[0] + 7, e[1] + 3, usd(lst));
    }
    [[0, "09시"], [72, "15시"], [144, "21시"], [216, "03시"], [288, "09시"]].forEach(function (q) { s += txt("lq-ax", L + (w - L - R) * q[0] / 288, h - 3, q[1], q[0] === 0 ? "start" : q[0] === 288 ? "end" : "middle"); });
    el._hit = function (x) { var i = Math.max(0, Math.min(287, Math.round((x - L) / (w - L - R) * 288) - 1)), v = T.cum[i]; return hhmm(T.d0 + (i + 1) * 300) + " KST" + (v != null ? "<br>오늘 " + usd(v) : "") + (B ? "<br>평소 " + usd(B.p50[i]) + " · 25~75% " + usd(B.p25[i]) + "~" + usd(B.p75[i]) : ""); };
    return svg(w, h, s, "오늘 5분 누적 청산과 지난 날들의 같은 시각 띠");
  };
  CH.bars24 = function (el, w, h) {
    var Rw = S.bars; if (!Rw.length) return null;
    var L = 46, R = 4, Tp = 4, Bt = 15, mx = Math.max.apply(null, Rw.map(function (b) { return Math.max(b.l, b.s); }).concat([1])), y0 = Tp + (h - Tp - Bt) / 2, sc = (y0 - Tp) / mx, sl = (w - L - R) / Rw.length, s = "";
    ticks(mx, 2).forEach(function (v) { s += line("lq-gl", L, w - R, y0 - v * sc, y0 - v * sc) + line("lq-gl", L, w - R, y0 + v * sc, y0 + v * sc) + txt("lq-ax", L - 5, y0 - v * sc + 3, usd(v), "end"); });
    Rw.forEach(function (b, i) {
      var x = L + i * sl, bw = Math.max(sl - 0.5, 1);
      if (b.s) s += '<rect class="lq-sh" x="' + x + '" y="' + (y0 - b.s * sc) + '" width="' + bw + '" height="' + Math.max(b.s * sc, 0.8) + '"/>';
      if (b.l) s += '<rect class="lq-lg" x="' + x + '" y="' + y0 + '" width="' + bw + '" height="' + Math.max(b.l * sc, 0.8) + '"/>';
      if (b.t % 21600 === 0 && x < w - 18) s += txt("lq-ax", x, h - 3, hhmm(b.t), "middle");
    });
    s += line("lq-z", L, w - R, y0, y0);
    el._hit = function (x) { var b = Rw[Math.max(0, Math.min(Rw.length - 1, Math.floor((x - L) / sl)))]; return b ? hhmm(b.t) + " KST<br>롱 " + usd(b.l) + " · 숏 " + usd(b.s) + " · " + b.n + "건<br>OKX " + usd(b.ok) + " · Bybit " + usd(b.by) + (b.hl ? " · HL " + usd(b.hl) : "") : ""; };
    return svg(w, h, s, "24시간 5분 청산, 위 숏 아래 롱");
  };
  CH.branch = function (el, w, h) {
    var a = S.data && S.data.branch; if (!a || a.length < 2) return null;
    var L = 38, R = 8, Tp = 6, Bt = 15, X = function (i) { return L + (w - L - R) * i / (a.length - 1); }, Y = function (v) { return Tp + (h - Tp - Bt) * (1 - Math.max(0, Math.min(1.2, v)) / 1.2); }, s = "";
    [0.25, 0.5, 0.75, 1].forEach(function (v) { s += line("lq-gl", L, w - R, Y(v), Y(v)) + txt("lq-ax", L - 5, Y(v) + 3, v.toFixed(2), "end"); });
    s += '<path class="lq-ln2" d="' + path(a.map(function (r, i) { return [X(i), Y(r[2])]; })) + '"/><path class="lq-ln" d="' + path(a.map(function (r, i) { return [X(i), Y(r[1])]; })) + '"/><circle class="lq-dt" cx="' + X(a.length - 1) + '" cy="' + Y(a[a.length - 1][1]) + '" r="3"/>';
    s += txt("lq-ax", L, h - 3, a[0][0].slice(5)) + txt("lq-ax", w - R, h - 3, a[a.length - 1][0].slice(5), "end");
    el._hit = function (x) { var r = a[Math.max(0, Math.min(a.length - 1, Math.round((x - L) / (w - L - R) * (a.length - 1))))]; return r[0] + "<br>자기 흥분 " + r[1].toFixed(2) + " · 셔플 대조 " + (Math.abs(r[2]) < 0.005 ? 0 : r[2]).toFixed(2) + "<br>" + r[3].toLocaleString("en-US") + "건(바이낸스+OKX)"; };
    return svg(w, h, s, "자기 흥분 지수 하루 추이와 셔플 대조");
  };
  // 남은 매물 히트맵 -- 그림 규칙은 일봉 히트맵(fp_daily.js drawHeat)과 같다: 투명도 .85×t^2.5 · densityColor. 칸값 t 는 서버가 98분위로 눌러 준다.
  function heatUrl(rm, w, h, cw, x0, Y, clip) {
    var key = [asset(), S.win, w, h, S.data.generated_ms, document.documentElement.dataset.theme || ""].join("|");
    if (S.heatKey === key) return S.heatUrl;
    var cv = document.createElement("canvas"), dp = 2; cv.width = w * dp; cv.height = h * dp;
    var g = cv.getContext("2d"); g.scale(dp, dp); g.beginPath(); g.rect(clip[0], clip[1], clip[2], clip[3]); g.clip();
    rm.cols.forEach(function (c, i) {
      if (!c.length) return;
      var xa = Math.floor(x0 + i * cw), xb = Math.floor(x0 + (i + 1) * cw);
      for (var k = 1; k < c.length; k++) {
        var t = c[k] / 1000; if (!(t > 0)) continue;
        var cell = c[0] + k - 1, top = Math.floor(Y(rm.lo + (cell + 1) * rm.row)), bot = Math.ceil(Y(rm.lo + cell * rm.row));
        g.globalAlpha = 0.85 * Math.pow(t, 2.5); g.fillStyle = densityColor(t); g.fillRect(xa, top, xb - xa, Math.max(1, bot - top));
      }
    });
    S.heatKey = key; S.heatUrl = cv.toDataURL();
    return S.heatUrl;
  }
  CH.remain = function (el, w, h) {
    var D = S.data, rm = D && D.remain[String(S.win)]; if (!rm || !rm.cols || !rm.cols.length) return null;
    var P = price(), days = D.days.slice(-rm.view), n = days.length, prof = w < 300 ? 36 : 52, sh = h < 240 ? 26 : 40, L = 44, R = prof + 6, Tp = 6, Bt = 15, xr = w - R;
    var lo = Math.min.apply(null, days.map(function (d) { return d[3]; })) * 0.985, hi = Math.max.apply(null, days.map(function (d) { return d[2]; })) * 1.015, cw = (xr - L) / days.length, X = function (i) { return L + (i + 0.5) * cw; }, ch = h - Bt - sh - Tp, Y = function (p) { return Tp + ch * (1 - (p - lo) / (hi - lo)); }, s = "";
    var st = niceStep(hi - lo, 5); for (var p = Math.ceil(lo / st) * st; p < hi; p += st) s += line("lq-gl", L, xr, Y(p), Y(p)) + txt("lq-ax", L - 5, Y(p) + 3, px(p), "end");
    s += '<image href="' + heatUrl({ lo: rm.lo, row: rm.row, cols: rm.cols.slice(-n) }, w, h, cw, L, Y, [L, Tp, xr - L, ch]) + '" x="0" y="0" width="' + w + '" height="' + h + '" preserveAspectRatio="none"/>';
    days.forEach(function (d, i) { var x = X(i), up = d[4] >= d[1]; s += line("lq-wk", x, x, Y(d[2]), Y(d[3])) + '<rect class="' + (up ? "lq-cu" : "lq-cd") + '" x="' + (x - cw * 0.3) + '" y="' + Y(Math.max(d[1], d[4])) + '" width="' + Math.max(cw * 0.6, 1) + '" height="' + Math.max(Math.abs(Y(d[1]) - Y(d[4])), 1) + '"/>'; });
    if (S.win < n - 1) { var xs = X(n - 1 - S.win); s += '<line x1="' + xs + '" x2="' + xs + '" y1="' + Tp + '" y2="' + (Tp + ch) + '" stroke="var(--accent)" stroke-dasharray="2 3" opacity=".7"/>' + txt("lq-ax o", xs + 4, Tp + 11, "오늘 지도 입력 시작 · " + S.win + "일 전"); }
    var ba = LADDER_R / 100, yT = Y(P * (1 + ba)), yB = Y(P * (1 - ba));
    s += '<rect x="' + L + '" y="' + yT + '" width="' + (xr - L) + '" height="' + (yB - yT) + '" fill="color-mix(in srgb,var(--accent) 9%,transparent)" stroke="color-mix(in srgb,var(--accent) 45%,transparent)" stroke-dasharray="4 3"/>' + txt("lq-ax t o", L + 5, yT + 12, "사다리 ±" + LADDER_R + "%");
    s += line("lq-px", L, xr, Y(P), Y(P)) + txt("lq-ax t o", xr + 7, Y(P) - 3, px(P));
    var pr = (rm.prof || []).filter(function (q) { return q[0] > lo && q[0] < hi; }), pm = Math.max.apply(null, pr.map(function (q) { return q[2]; }).concat([1e-9])), bh = Math.max(ch / ((hi - lo) / (P * 0.01)) - 1, 1.5);
    pr.forEach(function (q) { s += '<rect class="' + (q[1] ? "lq-sh" : "lq-lg") + '" opacity=".8" x="' + (xr + 5) + '" y="' + (Y(q[0]) - bh / 2) + '" width="' + (prof - 10) * q[2] / pm + '" height="' + bh + '"/>'; });
    var y0 = Tp + ch + 6 + (sh - 6) / 2, hh = (sh - 6) / 2 - 1, bw = Math.max(cw * 0.74, 0.8);
    s += line("lq-z", L, xr, y0, y0) + txt("lq-ax", L - 5, y0 - hh / 2 + 3, "위", "end") + txt("lq-ax", L - 5, y0 + hh / 2 + 3, "아래", "end");
    days.forEach(function (d, i) { var op = i < n - 1 - S.win ? 0.22 : 0.85; s += '<rect class="lq-sh" opacity="' + op + '" x="' + (X(i) - bw / 2) + '" y="' + (y0 - hh * d[6]) + '" width="' + bw + '" height="' + hh * d[6] + '"/><rect class="lq-lg" opacity="' + op + '" x="' + (X(i) - bw / 2) + '" y="' + y0 + '" width="' + bw + '" height="' + hh * d[5] + '"/>'; });
    [0, Math.floor(days.length / 3), Math.floor(2 * days.length / 3), days.length - 1].forEach(function (i) { s += txt("lq-ax", X(i), h - 3, days[i][0].slice(5), i === 0 ? "start" : i === days.length - 1 ? "end" : "middle"); });
    el._hit = function (x, y) {
      var i = Math.floor((x - L) / cw), d = days[i]; if (!d) return "";
      var head = d[0] + " · 고 " + px(d[2]) + " · 저 " + px(d[3]) + " · 종 " + px(d[4]);
      if (y > Tp + ch) return head + "<br>그날 만든 양 중 지금까지 남음: 위 " + pc(d[6]) + " · 아래 " + pc(d[5]);
      var pp = lo + (hi - lo) * (1 - (y - Tp) / ch), c = rm.cols[rm.cols.length - n + i], cell = Math.floor((pp - rm.lo) / rm.row), v = c && c.length && cell >= c[0] ? c[cell - c[0] + 1] : 0;
      return head + "<br>" + px(pp) + " 부근 진하기 " + ((v || 0) / 10).toFixed(0) + "%(이 탭 안 상대값)";
    };
    return svg(w, h, s, "일봉별 남은 청산 매물 히트맵");
  };
  CH.ladder = function (el, w, h) {
    var P = price(), M = { "30": mapOf(30), "7": mapOf(7), "1": mapOf(1) }, real = realRows(P), m = ctx(), hl = m && m.hl_liq, R_ = LADDER_R;
    if (!(P > 0) || !(M["1"] || M["7"] || M["30"])) return null;
    var cols = ["30", "7", "1", "real", "hl"], LAB = { "30": "지도 30일", "7": "지도 7일", "1": "지도 1일", real: "실측 7일", hl: "HL 고래" }, narrow = w < 420, L = 50, Tp = 18, Bt = 6, gap = narrow ? 5 : 10, cw = (w - L - gap * cols.length) / cols.length, Y = function (d) { return Tp + (h - Tp - Bt) * (1 - (d + R_) / (2 * R_)); }, rh = Math.max((h - Tp - Bt) / (2 * R_ / 0.25) - 0.6, 1.2), s = "", LEV = ["--lev10", "--lev25", "--lev50", "--lev100"];   // 풋프린트 «청산» 기둥과 같은 레버리지 색(app.js TV) -- 위·아래는 현재가 위치와 누적 곡선이 말한다
    for (var d = -R_; d <= R_; d += 1) if (d) s += line("lq-gl", L, w, Y(d), Y(d)) + txt("lq-ax", L - 5, Y(d) + 3, px(P * (1 + d / 100)), "end");
    cols.forEach(function (c, ci) {
      var x0 = L + gap + ci * (cw + gap); s += txt("lq-ax lq-axh", x0, 10, LAB[c]);
      if (c === "real") {
        var ks = Object.keys(real).filter(function (g) { return Math.abs(g * 0.25) <= R_; }), mr = Math.max.apply(null, ks.map(function (g) { return real[g][0] + real[g][1]; }).concat([1]));
        ks.forEach(function (g) { var r = real[g], t = r[0] + r[1], ww = cw * Math.sqrt(t / mr), wl = t ? ww * r[0] / t : 0; if (r[0]) s += '<rect class="lq-lg" x="' + x0 + '" y="' + (Y(g * 0.25) - rh / 2) + '" width="' + wl + '" height="' + rh + '"/>'; if (r[1]) s += '<rect class="lq-sh" x="' + (x0 + wl) + '" y="' + (Y(g * 0.25) - rh / 2) + '" width="' + (ww - wl) + '" height="' + rh + '"/>'; });
        if (cw > 120) s += txt("lq-ax", x0 + cw, 10, "최대 " + usd(mr), "end");
      } else if (c === "hl") {
        var all = hl ? (hl.below || []).map(function (l) { return [l, "--bad"]; }).concat((hl.above || []).map(function (l) { return [l, "--good"]; })) : [], hm = Math.max.apply(null, all.map(function (q) { return q[0].usd; }).concat([1]));
        all.forEach(function (q) { var dd = (q[0].px / P - 1) * 100; if (Math.abs(dd) > R_) return; var y = Y(dd), ww = Math.max(Math.min(cw * 0.55 * q[0].usd / hm, cw - 38), 2); s += '<line x1="' + x0 + '" x2="' + (x0 + ww) + '" y1="' + y + '" y2="' + y + '" stroke="var(' + q[1] + ')" stroke-width="3"/>' + txt("lq-ax t", x0 + ww + 4, y + 3, usd(q[0].usd)); });
      } else {
        var mp = M[c]; if (!mp) return;
        var ks2 = Object.keys(mp.rows).filter(function (g) { return Math.abs(g * 0.25) <= R_; }), sum = function (r) { return r[0] + r[1] + r[2] + r[3]; }, mx = Math.max.apply(null, ks2.map(function (g) { return sum(mp.rows[g]); }).concat([1]));
        ks2.forEach(function (g) { var xx = x0; mp.rows[g].forEach(function (v, ti) { if (!v) return; var ww = cw * v / mx; s += '<rect fill="var(' + LEV[ti] + ')" x="' + xx.toFixed(1) + '" y="' + (Y(g * 0.25) - rh / 2).toFixed(1) + '" width="' + ww.toFixed(1) + '" height="' + rh.toFixed(1) + '"/>'; xx += ww; }); });
        if (cw > 120) s += txt("lq-ax", x0 + cw, 10, "칸 최대 " + usd(mx), "end");
      }
    });
    var pos = typeof snapshotAccountPosition === "function" ? snapshotAccountPosition() : null, liq = pos && Number(pos.liquidation_price);
    if (liq > 0 && Math.abs(liq / P - 1) * 100 <= R_) s += '<line x1="' + L + '" x2="' + w + '" y1="' + Y((liq / P - 1) * 100) + '" y2="' + Y((liq / P - 1) * 100) + '" stroke="var(--position)" stroke-width="1.5"/>' + txt("lq-ax t o", w - 2, Y((liq / P - 1) * 100) - 4, "내 청산가 " + px(liq), "end");
    s += line("lq-px", 0, w, Y(0), Y(0)) + '<rect x="0" y="' + (Y(0) - 8) + '" width="' + (L - 4) + '" height="16" rx="3" fill="var(--accent)"/><text x="' + (L - 4) / 2 + '" y="' + (Y(0) + 3.5) + '" text-anchor="middle" class="lq-pxt">' + px(P) + "</text>";
    el._hit = function (x, y) {
      var dd = (1 - (y - Tp) / (h - Tp - Bt)) * 2 * R_ - R_, g = Math.round(dd / 0.25); if (!g) return "";
      var row = function (k) { var r = M[k] && M[k].rows[g]; return r ? usd(r[0] + r[1] + r[2] + r[3]) : "—"; }, rr = real[g];
      return px(P * (1 + g * 0.25 / 100)) + " · " + sgn(g * 0.25, 2) + "%<br>지도 30일 " + row("30") + " · 7일 " + row("7") + " · 1일 " + row("1") + (rr ? "<br>실측 7일 롱 " + usd(rr[0]) + " · 숏 " + usd(rr[1]) : "") + "<br>지도 금액은 OI×2 눈금 · 절대액 아님";
    };
    return svg(w, h, s, "가격 사다리: 지도 30일 7일 1일, 실측 7일, HL 고래 청산가");
  };
  // 2026-11-08 펜 메모 «이 곡선으로 풋프린트 청산 프로파일을 대체»(옛 «움직이면 무엇이 닿나» 칸) -- 가격축은 풋프린트 것(yAt) · 지도 1·7·30일을 한꺼번에.
  //   가로 = 로그 4자리, 코인마다 지도 크기에 맞춘다. app.js 청산 기둥이 부른다. 돌려주는 것 = { svg, hit(가격) → 툴팁 글 }.
  window.fpLiqCurve = function (x0, x1, top, bot, yAt, yMin, yMax, win) {
    var P = price(), M = { "1": mapOf(1), "7": mapOf(7), "30": mapOf(30) }, B = S.data && S.data.beta, m = ctx(), mx = 0;
    ["1", "7", "30"].forEach(function (k) { if (M[k]) mx = Math.max(mx, M[k].dn[FINE], M[k].up[FINE]); });
    if (!(P > 0) || !(mx > 0)) return null;
    var hi = Math.pow(10, Math.ceil(Math.log10(mx))), lo = hi / 1e4, L = x0 + 2, W = x1 - x0 - 8, s = "";
    var X = function (v) { return L + W * Math.log10(Math.max(v, lo) / lo) / 4; }, Y = function (p) { return Math.max(top, Math.min(bot, yAt(p))); }, inView = function (p) { return p >= yMin && p <= yMax; };
    [lo, lo * 100, hi].forEach(function (v) { s += line("lq-gl", X(v), X(v), top, bot); });
    s += txt("lq-ax o", L, top + 10, usd(lo)) + txt("lq-ax o", L + W, top + 10, usd(hi), "end");
    var sw = { "1": [2.2, ""], "7": [1.4, ""], "30": [1.2, "3 3"] };
    ["30", "7", "1"].forEach(function (k) {
      if (!M[k]) return;
      [[-1, "--bad"], [1, "--good"]].forEach(function (q) {
        var a = [];
        for (var i = 1; i <= FINE; i++) {
          var dd = q[0] * i / 10, p = P * (1 + dd / 100);
          if (q[0] > 0 ? p < yMin : p > yMax) continue;   // 현재가가 화면 밖이면 화면 끝부터
          a.push([X(cumAt(M[k], dd)), Y(p)]);
          if (!inView(p)) break;                          // 화면 끝까지 한 칸 더 긋고 멈춘다
        }
        while (a.length > 1 && a[1][0] <= L) a.shift();   // 눈금 아래(거의 0) 구간은 안 긋는다 -- 왼쪽 끝에 붙은 선이 칸 테두리처럼 보였다(5분 창은 대부분 이 구간)
        if (a.length > 1) s += '<path fill="none" stroke="var(' + q[1] + ')" stroke-width="' + sw[k][0] + '" stroke-dasharray="' + sw[k][1] + '" d="' + path(a) + '"/>';
      });
    });
    var k0 = M[win] ? String(win) : M["1"] ? "1" : M["7"] ? "7" : "30";   // 고른 막대 기간으로 화면 끝까지 얼마나 쌓였나 -- 5분 창처럼 좁으면 곡선보다 이 숫자가 먼저 읽힌다
    [[-1, B && B.dn, (1 - yMin / P) * 100, bot - 16, bot - 5, "↓"], [1, B && B.up, (yMax / P - 1) * 100, top + 34, top + 22, "↑"]].forEach(function (q) {
      if (!(q[2] > 0)) return;
      s += txt("lq-ax t o", L, q[4], q[5] + " " + k0 + "일 " + usd(cumAt(M[k0], q[0] * q[2])));
      if (!q[1]) return;
      if (q[2] > 0.5) s += '<path class="lq-ln" d="' + path([[X(q[1] * 0.5), Y(P * (1 + q[0] * 0.005))], [X(q[1] * q[2]), Y(P * (1 + q[0] * q[2] / 100))]]) + '"/>';
      s += txt("lq-ax o", L, q[3], "1%당 " + usd(q[1]));
    });
    var bk = m && m.book && m.book.sweep, bps = m && m.book && m.book.bps;
    if (bk && bps) bps.forEach(function (bp, j) { [[-1, bk.bid[j]], [1, bk.ask[j]]].forEach(function (q) { var pp = P * (1 + q[0] * bp / 1e4), x = X(q[1] * P), y = Y(pp); if (inView(pp)) s += '<rect x="' + (x - 2.5) + '" y="' + (y - 2.5) + '" width="5" height="5" fill="var(--warn)" transform="rotate(45 ' + x + " " + y + ')"/>'; }); });
    var hl = m && m.hl_liq; if (hl) (hl.below || []).concat(hl.above || []).forEach(function (l) { if (inView(l.px)) s += '<circle cx="' + X(l.usd) + '" cy="' + Y(l.px) + '" r="3.5" fill="none" stroke="var(--accent)" stroke-width="1.3"/>'; });
    return { svg: s, hit: function (p) {
      var dd = Math.round((p / P - 1) * 1000) / 10; if (!dd || Math.abs(dd) > FINE / 10) return "";
      var v = function (k) { return M[k] ? usd(cumAt(M[k], dd)) : "—"; };
      return px(p) + " · " + sgn(dd, 1) + "%까지 누적 청산<br>지도 1일 " + v("1") + " · 7일 " + v("7") + " · 30일 " + v("30") + (B ? "<br>실측 탄력으로는 " + usd((dd < 0 ? B.dn : B.up) * Math.abs(dd)) : "")
        + '<br><span style="opacity:.7">굵은 선 1일 · 가는 선 7일 · 점선 30일 · 흰 선 실측 탄력 · 마름모 호가 · 원 HL 고래<br>추정 지도(OI×2 눈금) · 절대액 아님 · 지도로만</span>';
    } };
  };

  // ── 조립 ────────────────────────────────────────────────────────────────
  var d_ = function (name, mh, empty) { return '<div class="lq-d" data-d="' + name + '" data-mh="' + mh + '" data-empty="' + empty + '"></div>'; };
  var kp = function (lab, val, sub, cl) { return '<div class="lq-kp"><span>' + lab + '</span><b class="' + (cl || "") + '">' + val + "</b>" + (sub ? "<small>" + sub + "</small>" : "") + "</div>"; };
  function remainKpi() {
    var r = S.data && S.data.remain[String(S.win)], P = price();
    if (!r || r.ll == null) return "";
    return '<div class="lq-rib">' + kp("아래(롱) 남음", pc(r.ll / r.lm), "만든 양 대비", "lg") + kp("위(숏) 남음", pc(r.ul / r.um), "만든 양 대비", "sh")
      + (r.lb ? kp("큰 아래 매물", px(r.lb), sgn((r.lb / P - 1) * 100, 1) + "%") : "") + (r.ub ? kp("큰 위 매물", px(r.ub), sgn((r.ub / P - 1) * 100, 1) + "%") : "")
      + kp("±3% 안", pc(r.near3), "7일보다 오래됨 " + pc(r.old7)) + "</div>";
  }
  function nowLine() {
    var T = today(), B = S.data && S.data.band, i = T.cum.length - 1, base = B && i >= 0 ? B.p50[i] : 0, bnOk = i >= 0 ? T.cum[i] : 0, bars = S.bars, big = asset() === "eth" || asset() === "btc" ? 1e5 : 1e4, tot24 = 0, n24 = 0, last = null;
    bars.forEach(function (b) { tot24 += b.l + b.s; n24 += b.n; if (b.l + b.s >= big) last = b; });
    var ago = last ? Math.max(0, (Date.now() / 1000 - last.t - 300) / 3600) : null, m = ctx(), bu = m && m.burst, side = bu && typeof mcLiqBurstSide === "function" && asset() === "eth" ? mcLiqBurstSide(bu) : null, burst = "";
    if (asset() === "eth") burst = side ? '<span class="lq-burst" title="직전 60초 청산이 큰 버스트입니다. 이런 순간 보유 포지션을 시장가로 닫으면 평균 −17bp 였습니다(35건, 95% −24~−11, 수수료 포함 모의 원장). 그 뒤 되돌림은 약합니다."><b>' + (side === "long" ? "롱" : "숏") + " 청산 급증 " + usd(side === "long" ? bu.long_usd_60s : bu.short_usd_60s) + "/60초</b> · 시장가 청산 주의</span>" : "<span>60초 급증 <b>없음</b></span>";
    return "<span>오늘(09시~) <b>" + usd(T.tot) + "</b></span>" + (base > 0 ? "<span>평소 같은 시각의 <b>" + pc(bnOk / base) + "</b></span>" : "") + (T.tot > 0 ? '<span>롱 쪽 <b class="lg">' + pc(T.l / T.tot) + "</b></span>" : "")
      + "<span>24시간 <b>" + usd(tot24) + "</b> · " + n24.toLocaleString("en-US") + "건</span>" + (ago != null ? "<span>마지막 " + usd(big) + "+ 5분봉 <b>" + ago.toFixed(1) + "시간 전</b></span>" : "") + burst
      + (T.first != null && T.first > T.d0 + 600 ? '<span class="lq-warn">오늘 기록 ' + hhmm(T.first) + "부터</span>" : "");
  }
  function build() {
    var host = $("lqTab"); if (!host) return;
    var seg = '<div class="lq-seg" role="group" aria-label="남은 매물 입력 기간">' + WINS.map(function (w) { return '<button type="button" data-win="' + w + '" aria-pressed="' + (S.win === w) + '">' + w + "일</button>"; }).join("") + "</div>";
    var H = function (t, tag, cls, right) { return "<header><h4>" + t + '</h4><span class="lq-tag ' + cls + '">' + tag + "</span>" + (right || "") + "</header>"; };
    host.innerHTML = '<div class="lq-now" id="lqNow"></div><div class="lq-cols">'
      + '<div class="lq-row lq-top">'
      + '<section class="lq-p lq-f-rem">' + H("남은 매물 일봉별", "파생·서술", "der", seg) + '<div class="lq-b"><div id="lqRemKpi"></div>' + d_("remain", 300, "남은 매물 계산 전입니다") + '<div class="lq-leg" id="lqRemLeg"></div></div></section>'
      + '<section class="lq-p lq-f-lad">' + H("가격 사다리 ±" + LADDER_R + "%", "추정", "est", "<small>칸 0.25%</small>") + '<div class="lq-b">' + d_("ladder", 560, "지도를 받는 중입니다") + '<div class="lq-leg" id="lqLadLeg"></div></div></section>'
      + "</div>"
      + '<div class="lq-row lq-bot">'
      + '<section class="lq-p lq-f-tod">' + H("오늘 · 24시간", "실측", "real") + '<div class="lq-b"><div class="lq-split"><div><h5 id="lqTodLab">오늘 5분 누적</h5>' + d_("today", 130, "오늘 기록이 아직 없습니다") + "</div><div><h5>24시간 5분 · 위 숏 · 아래 롱</h5>" + d_("bars24", 110, "5분봉을 받는 중입니다") + "</div></div></div></section>"
      + '<section class="lq-p lq-f-br">' + H("자기 흥분 지수 추이", "문헌·서술", "lit", '<small id="lqBrLab"></small>') + '<div class="lq-b">' + d_("branch", 150, "표본이 아직 부족합니다") + "</div></section>"
      + "</div></div>";
    S.built = asset();
  }
  function fill(id, html) { var e = $(id); if (e && e._h !== html) { e.innerHTML = html; e._h = html; } }
  function draw() {
    if (!S.on) return;
    var host = $("lqTab"); if (!host) return;
    if (S.built !== asset()) build();
    var P = price(), stack = matchMedia(STACK_MQ).matches;
    fill("lqNow", nowLine());
    fill("lqRemKpi", remainKpi());
    var age = S.data ? (Date.now() - S.data.generated_ms) / 3600000 : null;
    fill("lqRemLeg", "<span>칸 = 그날 마감 기준 지난 " + S.win + "일 입력 중 아직 안 닿은 청산가 · 진할수록 많음</span><span>오른쪽 = 오늘 남은 양</span><span>아래 띠 = 그 날 만든 양 중 지금까지 남은 비율</span>" + (age != null && age > 3 ? '<span class="lq-warn">' + age.toFixed(0) + "시간 전 계산</span>" : ""));
    var pos = typeof snapshotAccountPosition === "function" ? snapshotAccountPosition() : null, liq = pos && Number(pos.liquidation_price);
    fill("lqLadLeg", "<span><i class=\"lq-sw\" style=\"background:var(--lev10)\"></i>10배</span><span><i class=\"lq-sw\" style=\"background:var(--lev25)\"></i>20~25배</span><span><i class=\"lq-sw\" style=\"background:var(--lev50)\"></i>50배</span><span><i class=\"lq-sw\" style=\"background:var(--lev100)\"></i>75~100배</span><span>지도 눈금 = OI×2 · 절대액 아님</span><span>실측 = 바이낸스 7일 체결 위치</span>" + (liq > 0 && P > 0 ? "<span>내 청산가 " + px(liq) + "(" + sgn((liq / P - 1) * 100, 1) + "%)</span>" : ""));
    var B = S.data && S.data.band; fill("lqTodLab", "오늘 5분 누적" + (B ? " · 띠 = 지난 " + B.n + "일" : S.data ? " · 비교 띠는 기록 7일부터" : ""));
    var br = S.data && S.data.branch, bl = br && br.length ? br[br.length - 1] : null; fill("lqBrLab", bl ? bl[0].slice(5) + " " + bl[1].toFixed(2) + " · 점선 = 대조 " + (Math.abs(bl[2]) < 0.005 ? 0 : bl[2]).toFixed(2) + (br.length < 20 ? " · " + br.length + "일뿐" : "") : "");
    host.querySelectorAll(".lq-d").forEach(function (el) { el.style.height = stack ? el.dataset.mh + "px" : ""; });
    host.querySelectorAll(".lq-d").forEach(function (el) {
      var r = el.getBoundingClientRect(), w = Math.round(r.width), h = Math.round(r.height);
      if (!(w > 0 && h > 0)) return;
      el._hit = null;
      var out = CH[el.dataset.d](el, w, h);
      el.innerHTML = out || '<p class="lq-empty">' + el.dataset.empty + "</p>";
    });
  }
  function tick() {
    if (document.hidden) return;
    if (S.asset !== asset()) return watchAsset();
    fetchData(false);   // ponytail: 탭이 꺼져도 기둥의 실측 탄력(beta) 때문에 파일 통째(압축 전 ~1MB, 10분마다) -- 무거우면 서버에 beta 만 주는 길을 낸다
    if (!S.on) return;
    fetchBars(false);
    draw();
  }
  // 코인을 바꾼 순간: 옛 코인 숫자를 바로 지우고, 지도 7·30일은 응답이 늦으니 잠시 뒤 한두 번 더 그린다.
  function watchAsset() {
    if (S.asset === asset()) return;
    S.asset = asset(); S.data = null; S.bars = []; S.barsT = 0; S.heatKey = "";
    draw(); fetchData(true);
    if (S.on) { fetchBars(true); setTimeout(draw, 900); setTimeout(draw, 2200); }
  }
  function setOn(on) {
    S.on = on;
    var host = $("lqTab"); if (host) host.hidden = !on;
    if (on) { tick(); setTimeout(draw, 900); setTimeout(draw, 2200); }
    else if (typeof hideTooltip === "function") hideTooltip();
  }
  function init() {
    var host = $("lqTab"); if (!host) return;
    host.addEventListener("click", function (e) {
      var b = e.target.closest("[data-win]"); if (!b) return;
      S.win = Number(b.dataset.win); S.heatKey = "";
      try { localStorage.setItem("liqTabWin", String(S.win)); } catch (x) { /* 저장 못 해도 동작 */ }
      host.querySelectorAll("[data-win]").forEach(function (o) { o.setAttribute("aria-pressed", String(o === b)); });
      draw();
    });
    host.addEventListener("mousemove", function (e) {
      if (typeof showTooltip !== "function") return;
      var d = e.target.closest(".lq-d"), r = d && d.getBoundingClientRect(), t = d && d._hit ? d._hit(e.clientX - r.left, e.clientY - r.top) : "";
      if (t) showTooltip(e.pageX, e.pageY, t); else hideTooltip();
    });
    host.addEventListener("mouseleave", function () { if (typeof hideTooltip === "function") hideTooltip(); });
    if (typeof ResizeObserver === "function") { var rt = 0; new ResizeObserver(function () { clearTimeout(rt); rt = setTimeout(draw, 120); }).observe(host); }
    window.addEventListener("fpview", function (e) { setOn(e.detail === "liq"); });
    if (document.body && $("fpCard") && $("fpCard").classList.contains("lq-on")) setOn(true);
    setInterval(tick, TICK_MS);
    setInterval(watchAsset, 400);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
