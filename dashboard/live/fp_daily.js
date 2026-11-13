// 일봉 풋프린트 (2026-10-08, 사용자 «일봉 풋프린트를 최대치로 · 축소/확대 · 델타·거래대금·OI·청산 · 30일 이상 · 4시간봉 없이»).
// 서버 /api/footprint-daily(일 요약 전체) + /api/footprint-daily/cells(보이는 날의 가격 칸). ETH·XRP(2026-10-20, ?asset=).
// 2026-10-20 4시간봉(/api/footprint-4h)도 같은 그림·레인·히트맵으로 그린다(PERIODS -- 위 «4시간봉 없이»는 10-08 당시 요청).
// 캔버스로 그린다 -- 최대 ~2,500일을 끌고 확대해야 해서 SVG 요소 수만 개를 프레임마다 갈아 끼울 수 없다.
// 자리 = 5분 차트의 교체 영역(window.fpRegion) 위, 배경 없이(placeOverlay) -- 넓은 화면은 왼쪽 열(풋프린트~모의 판·지지/저항),
//   오른쪽 1초 수급·시장 맥락 칸은 그대로(사용자 «풋프린트부 차트부터 모의 판+지지·저항까지 대체 · 시간봉과 스타일 통일»).
// 확대 단계: 봉 폭 ≥ cellMinPx(G) 면 가격 칸(왼쪽 매도 · 오른쪽 매수), 그 아래는 캔들, 아주 좁으면 고저 선.
// 날짜 = UTC 자정(KST 09:00) -- 바이낸스 일봉 경계. 서술이지 신호가 아니다(볼륨 프로파일 규칙 5종 불합격, 10-04).
// 🔴CI 문법 검사(esprima)가 `?.(`·`?.[`·숫자 구분자를 못 읽는다 -- 쓰지 말 것(09-30 배포 워처 정지 사고).
(function () {
  "use strict";

  var HEAT_CLIP_PCT = 0.98;    // 양수 밀도의 98분위 = 가장 진한 색(진한 곳만 떠오르게)
  // 코인별 행 크기 단계($)·테두리 순위 칸·가격 자릿수 -- XRP = ETH 단계 × 서버 칸(footprint_daily.ASSETS CELL 0.0005 / $1)
  //   box = 2026-10-10 위·아래 2칸 순위는 늘 같은 칸(ETH $25)으로 고른다(사용자 A) -- 화면 칸 크기로 고르면 확대/축소마다 이웃 칸이 합쳐져 1·2등이 바뀌었다
  var ASSET_CFG = {
    eth: { coin: "ETH", rows: [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500], box: 25, dp: 2 },
    xrp: { coin: "XRP", rows: [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.0125, 0.025, 0.05, 0.1, 0.125, 0.25], box: 0.0125, dp: 4 },
    btc: { coin: "BTC", rows: [30, 60, 150, 300, 600, 750, 1500, 3000, 6000, 7500, 15000], box: 750, dp: 0 },   // 2026-10-31 ETH 단계 × 서버 칸 $30
    // 2026-11-03 주식 선물 = ETH 단계 × 서버 칸(SNDK $0.5 · SOXL $0.03)
    sndk: { coin: "SNDK", rows: [0.5, 1, 2.5, 5, 10, 12.5, 25, 50, 100, 125, 250], box: 12.5, dp: 2 },
    soxl: { coin: "SOXL", rows: [0.03, 0.06, 0.15, 0.3, 0.6, 0.75, 1.5, 3, 6, 7.5, 15], box: 0.75, dp: 2 },
    // 2026-11-13 ETH 단계 × 서버 칸(SKHYNIX $0.4 · MU $0.3)
    skhynix: { coin: "SKHYNIX", rows: [0.4, 0.8, 2, 4, 8, 10, 20, 40, 80, 100, 200], box: 10, dp: 2 },
    mu: { coin: "MU", rows: [0.3, 0.6, 1.5, 3, 6, 7.5, 15, 30, 60, 75, 150], box: 7.5, dp: 2 }
  };
  var HEAT_BOX_MIN_PX = 10;    // 봉 폭이 이보다 좁으면 테두리를 그리지 않는다(금액 글자는 칸 안에 들어갈 때만)
  var DEFAULT_SPAN = 30, ROW_PX = 11;
  // 이 폭부터 가격 칸(휴대폰은 7~12일 확대에서) · 2026-10-20 4시간봉은 1주(42봉 ≈ 24px)에서도 칸이 보이게 낮춘다(숫자는 칸에 들어갈 때만 -- drawCells)
  var cellMinPx = function (G) { return S.period === "4h" ? (G.narrow ? 6 : 16) : G.narrow ? 24 : 34; };
  var REFRESH_MS = 60000;
  // 2026-11-13 사용자 «1d 차트에 킹피셔처럼 현물/선물 CVD·OI 를 거래소별로»(시안 B) + «누적 레인은 제거»: 레인 = CVD 거래소별(5칸) · OI 거래소별(3칸) · 델타.
  //   거래대금·OI 막대·청산·추정 청산 레인은 화면에서 뺐다(값은 호버 툴팁에 그대로). 세 번째 칸 = 높이 배수.
  //   2026-11-13 사용자 «펀딩까지 해서 롱 우세·숏 우세를 더 정확히» → 펀딩 거래소별(2칸) 추가.
  var LANES = [["xcvd", "CVD · 거래소별", 5], ["xoi", "OI · 거래소별", 3], ["xfund", "펀딩 · 거래소별", 2], ["delta", "델타"]];
  //   검증 11-13: 4h·주식은 펀딩 자료가 없어 빈 레인 2칸이 가격 칸을 깎았다 → 펀딩 자료가 있을 때만 펀딩 레인.
  function lanesNow() { return LANES.filter(function (l) { return l[0] !== "xfund" || !!(S.X && S.X.bn_fund && Object.keys(S.X.bn_fund).length); }); }
  // 2026-10-20 사용자 «4h 버튼 = 1주일을 4시간봉 풋프린트로» → «1d 에서 데이터 양만 4h 로»: 같은 그림·레인·청산 히트맵을 봉 길이만 바꿔 쓴다
  //   (서버 /api/footprint-4h · 지난 7일 + 형성 중 · OI = 칸 마지막 값 · 청산 = 칸 합 · 히트맵 = 칸 마감 시점). 시각 = 칸 시작(UTC 00·04·…시 = KST 09·13·…시).
  var PERIODS = {
    "1d": { days: "/api/footprint-daily", cells: "/api/footprint-daily/cells", heatUrl: "/api/footprint-daily/heat", span: 30, minSpan: 7, name: "일봉",
            views: [["30", "30일"], ["180", "6개월"], ["all", "전체"]] },
    "4h": { days: "/api/footprint-4h", cells: "/api/footprint-4h/cells", heatUrl: "/api/footprint-4h/heat", span: 12, minSpan: 3, name: "4시간봉",
            views: [["12", "2일"], ["42", "1주"], ["all", "전체"]] }
  };
  var P = function () { return PERIODS[S.period]; };
  // 기본 범위(2026-10-20 사용자 «숫자가 보이게 · 휴대폰은 봉 수를 줄여»): 4시간봉 데스크탑 2일(12봉 ≈ 80px) · 휴대폰 12시간(3봉 ≈ 90px -- 4봉은 칸 숫자에 1~2px 모자람)
  var defSpan = function () { var cv = $("fpDailyCanvas"); return S.period === "4h" && cv && cv.clientWidth && cv.clientWidth < 560 ? 3 : P().span; };
  // 칸 키 → 사람 시각: 일봉 = 그대로(UTC 날짜) · 4시간봉 = 보는 사람 시계(KST)로 «10/19 13시»
  function lab(k) {
    if (S.period !== "4h") return k;
    var d = new Date(k + ":00:00Z");
    return (d.getMonth() + 1) + "/" + d.getDate() + " " + String(d.getHours()).padStart(2, "0") + "시";
  }
  // 2026-10-09 «추정 청산» 레인(사용자 «그 봉의 위·아래에 청산이 얼마나 쌓였나» · 시안 B): 그날 마감 7일 지도의 남은 청산 몫을
  //   고점 위(초록) · 봉 안(회색) · 저점 아래(빨강) 세 칸 100% 막대로 + 총액(2 × OI 달러) 가는 선. 🔴세 몫 합 = 늘 2 × OI -- 정보는 비율과 OI 크기.

  var S = { on: false, period: "1d", asset: "eth", D: null, n: 0, i0: 0, span: DEFAULT_SPAN, cells: new Map(), want: null, loading: false,
            heat: new Map(), stale: new Set(), heatWant: null, heatLoading: false, heatT: 0, heatOn: true,
            hover: null, timer: 0, raf: 0, drag: null, pinch: null, fetchT: 0, view: null, yOff: 0 };   // yOff = 세로 끌기(가격 단위, 10-09)
  var $ = function (id) { return document.getElementById(id); };
  var A = function () { return ASSET_CFG[S.asset]; };
  var Q = function () { return "asset=" + S.asset; };
  var css = function (n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); };
  var tipShow = function (x, y, h) { if (typeof showTooltip === "function") showTooltip(x, y, h); };   // app.js 전역
  var tipHide = function () { if (typeof hideTooltip === "function") hideTooltip(); };

  // ── 숫자 모양 ──────────────────────────────────────────────────────────────────────
  function usd(v) {
    var a = Math.abs(v), s = v < 0 ? "−" : "";
    return a >= 1e9 ? s + "$" + (a / 1e9).toFixed(2) + "B" : a >= 1e6 ? s + "$" + (a / 1e6).toFixed(1) + "M"
      : a >= 1e3 ? s + "$" + (a / 1e3).toFixed(0) + "k" : s + "$" + a.toFixed(0);
  }
  function sgnUsd(v) { return (v > 0 ? "+" : "") + usd(v); }
  // 칸 안 숫자는 더 짧게(2026-10-20 4시간봉 «숫자가 보이게»): 1만↑ «46k» · 1천↑ «4.6k» · 100↑ 정수 -- 툴팁·범례는 qty 그대로
  function cqty(v) { return v >= 9.95e5 ? (v / 1e6).toFixed(1) + "M" : v >= 9950 ? Math.round(v / 1e3) + "k" : v >= 1e3 ? (v / 1e3).toFixed(1) + "k" : v >= 99.95 ? v.toFixed(0) : v.toFixed(1); }   // 늘 4글자 이하
  function qty(v) { return v >= 1e6 ? (v / 1e6).toFixed(2) + "M" : v >= 1e3 ? (v / 1e3).toFixed(1) + "k" : v.toFixed(v >= 10 ? 0 : 1); }
  function px(v) { return v >= 1000 ? v.toFixed(0) : v.toFixed(A().dp); }
  function pct(v) { return (v > 0 ? "+" : "") + (v * 100).toFixed(2) + "%"; }
  function curAsset() { return typeof activeSnapshotAsset === "undefined" ? "eth" : activeSnapshotAsset; }   // app.js 전역
  function assetOn() { return Object.prototype.hasOwnProperty.call(ASSET_CFG, curAsset()); }

  // ── 데이터 ─────────────────────────────────────────────────────────────────────────
  async function loadDays(since) {
    var per = S.period, asset = S.asset;   // 리뷰 3: 받는 사이 1d ↔ 4h 를 바꾸면 늦게 온 응답은 버린다(다른 봉 길이 상태에 섞이면 4h 갱신이 일봉 키를 since 로 보내 400)
    var r = await fetch(P().days + "?" + Q() + (since ? "&since=" + since : ""), { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    var p = await r.json(), c = p.cols, oldN = S.n;
    if (per !== S.period || asset !== S.asset) return;
    var atEnd = oldN > 0 && S.i0 + S.span >= oldN - 0.5;          // 끝에 붙어 보던 중이면 새 날을 따라간다
    if (!since || !S.D) {
      S.D = c;
    } else {                                      // 같은 날은 덮고 새 날은 붙인다
      var at = S.D.d.indexOf(c.d[0]), keep = at < 0 ? S.D.d.length : at;
      Object.keys(c).forEach(function (k) { S.D[k] = S.D[k].slice(0, keep).concat(c[k]); });
      // 리뷰 10: since 창(직전 칸 + 형성 중)의 가격 칸·히트맵은 다시 묻는다 -- 형성 중 칸의 히트맵은 시간마다 바뀌고 막 닫힌 칸은 서버가 마감 뒤 다시 센다.
      //   지우지 않고 «묵음»만 표시(다시 받을 때까지 옛 그림) + want 를 비워 같은 화면이어도 다시 예약된다(전에는 오늘 칸만 지우고 다시 묻지 않아 캔들로 남았다).
      var fresh = new Set(c.d);
      [["c", S.cells], ["h", S.heat]].forEach(function (m) { m[1].forEach(function (_, k) { if (fresh.has(k.slice(k.indexOf("|") + 1))) S.stale.add(m[0] + k); }); });
      S.want = null; S.heatWant = null;
    }
    S.n = S.D.d.length;
    if (atEnd) S.i0 += S.n - oldN;
  }

  function rowFor(lo, hi, paneH) {
    var want = (hi - lo) / Math.max(1, paneH / ROW_PX);
    var steps = A().rows;
    for (var k = 0; k < steps.length; k++) if (steps[k] >= want) return steps[k];
    return steps[steps.length - 1];
  }

  // ── 청산 히트맵(하루 한 열 = 그날 마감의 추정 청산 밀도 · 7일 창 · 라이브 청산맵과 같은 계산) ──────────────
  function scheduleHeat(row, a, b) {
    var w = S.heatWant;
    if (w && w.row === row && w.a === a && w.b === b) return;
    S.heatWant = { row: row, a: a, b: b };
    clearTimeout(S.heatT);
    S.heatT = setTimeout(fetchHeat, 160);
  }
  // 리뷰 11: 4시간봉 히트맵은 한 요청에 60칸(10일)까지(footprint_daily.h4_heat_payload -- 넘으면 400). 창을 며칠 켜 두면 S.D 가 7일 넘게 쌓여
  //   «전체»·넓은 줌이 늘 400 이었다(같은 want 는 다시 안 묻는다) → 시각 폭 ≤ 60칸씩 나눠 묻는다. 일봉은 한 번에(서버 상한 4000일 > 소급본 ~2,500일).
  var H4_HEAT_SPAN_MS = 60 * 4 * 3600e3;
  function heatChunks(keys, h4) {
    if (!h4) return keys.length ? [keys] : [];
    var out = [], t = function (k) { return Date.parse(k + ":00:00Z"); };
    keys.forEach(function (k) { var c = out[out.length - 1]; if (c && t(k) - t(c[0]) <= H4_HEAT_SPAN_MS) c.push(k); else out.push([k]); });
    return out;
  }
  async function fetchHeat() {
    var w = S.heatWant;
    if (!w || S.heatLoading) return;
    S.heatLoading = true;
    var per = S.period, asset = S.asset, url = P().heatUrl, days = S.D.d.slice(w.a, w.b + 1);   // 키는 await 전에 -- 받는 사이 봉 길이가 바뀌면 버린다(리뷰 3)
    var rows = w.row === A().box ? [w.row] : [w.row, A().box];   // 화면 칸 + 테두리 순위용 고정 칸
    for (var q = 0; q < rows.length; q++) {
      var row = rows[q], parts = heatChunks(days.filter(function (k) { return !S.heat.has(row + "|" + k) || S.stale.has("h" + row + "|" + k); }), per === "4h");
      for (var c = 0; c < parts.length && per === S.period && asset === S.asset; c++) {
        var part = parts[c];
        try {
          var r = await fetch(url + "?" + Q() + "&from=" + part[0] + "&to=" + part[part.length - 1] + "&row=" + row, { cache: "no-store" });
          if (r.ok) {
            var p = await r.json();
            if (per === S.period && asset === S.asset) part.forEach(function (k) { S.heat.set(row + "|" + k, p.heat[k] || null); S.stale.delete("h" + row + "|" + k); });
          }
        } catch (e) { /* 다음 그리기 때 다시 묻는다 */ }
      }
    }
    S.heatLoading = false;
    redraw();
    if (S.heatWant !== w) fetchHeat();
  }
  // 표현 = 사용자 선택(10-08): 진하기에 비례한 투명도(약한 곳은 배경에 녹는다 · 98분위 = 가장 진한 색) +
  //   날마다 그날 종가 위(숏 청산)·아래(롱 청산)에서 몫이 가장 큰 칸 2개씩 굵은 테두리(초록 숏 · 빨강 롱 = 청산맵과 같은 색) +
  //   그 안에 금액 = 몫 × 2 × 그날 OI 달러(청산맵 tier_profile 의 2×OI 규모와 같은 뜻 · OI 없는 날은 %). 추정이지 실측 포지션이 아니다.
  function drawHeat(g, G, C, a, b, row, xOf, bw, yOf) {
    if (typeof densityColor !== "function") return;               // app.js 전역(5분 차트와 같은 색표·테마)
    var vals = [], i, k, h, t;
    for (i = a; i <= b; i++) { h = S.heat.get(row + "|" + S.D.d[i]); if (h) h[1].forEach(function (v) { if (v > 0) vals.push(v); }); }
    if (!vals.length) return;
    vals.sort(function (x, y) { return x - y; });
    var clip = vals[Math.min(vals.length - 1, Math.floor(vals.length * HEAT_CLIP_PCT))] || 1;
    g.save(); g.beginPath(); g.rect(G.L, G.top, G.plotW, G.priceH); g.clip();   // 가격 칸 안에서만(레인으로 안 샌다)
    for (i = a; i <= b; i++) {
      h = S.heat.get(row + "|" + S.D.d[i]);
      if (!h) continue;
      var x0 = Math.floor(xOf(i) - bw / 2), x1 = Math.ceil(xOf(i) + bw / 2);
      for (k = 0; k < h[1].length; k++) {
        t = Math.min(1, h[1][k] / clip);
        if (!(t > 0)) continue;                                  // 밀도 0 은 안 칠한다(배경이 비친다 -- 5분 차트 규약)
        var top = Math.floor(yOf(h[0] + (k + 1) * row)), bot = Math.ceil(yOf(h[0] + k * row));
        g.globalAlpha = 0.85 * Math.pow(t, 2.5); g.fillStyle = densityColor(t); g.fillRect(x0, top, x1 - x0, Math.max(1, bot - top));
      }
    }
    g.globalAlpha = 1;
    row = A().box;                                                // 테두리 = 고정 칸(화면 칸이 $10 이면 2~3칸을 묶은 높이로 그린다)
    if (bw >= HEAT_BOX_MIN_PX) for (i = a; i <= b; i++) {
      h = S.heat.get(row + "|" + S.D.d[i]);
      if (!h || !h[2]) continue;
      var cl = S.D.c[i], oiUsd = S.D.oi[i] != null ? S.D.oi[i] * cl : null, bx0 = Math.floor(xOf(i) - bw / 2) + 1, bxw = Math.ceil(bw) - 2;
      [true, false].forEach(function (up) {                       // 위(숏) 2 · 아래(롱) 2
        var ks = [];
        for (var j = 0; j < h[2].length; j++) if (h[2][j] > 0 && (h[0] + (j + 0.5) * row >= cl) === up) ks.push(j);
        ks.sort(function (x, y) { return h[2][y] - h[2][x]; });
        ks.slice(0, 2).forEach(function (j) {
          var yT = Math.floor(yOf(h[0] + (j + 1) * row)), yB = Math.ceil(yOf(h[0] + j * row));
          g.strokeStyle = up ? C.good : C.bad; g.lineWidth = 2;
          g.strokeRect(bx0 + 1, yT + 1, bxw - 2, Math.max(2, yB - yT - 2));
          var fs = Math.min(10, Math.floor(yB - yT - 2));          // 글자 크기 = 칸 높이에 맞춰 8~10px
          if (fs >= 8) {
            var v = h[2][j] * 2 * oiUsd, txt = !oiUsd ? (h[2][j] * 100).toFixed(1) + "%"     // 칸 폭이 좁아 «$»·소수 없이(툴팁 아님)
              : v >= 1e9 ? (v / 1e9).toFixed(1) + "B" : v >= 1e6 ? Math.round(v / 1e6) + "M" : Math.round(v / 1e3) + "k";
            g.font = "700 " + fs + "px " + C.sans; g.textAlign = "center"; g.textBaseline = "middle";
            if (g.measureText(txt).width <= bxw - 6) halo(g, C, txt, bx0 + bxw / 2, (yT + yB) / 2, C.text);
          }
        });
      });
    }
    g.restore();
  }

  function scheduleCells(row, a, b) {
    var w = S.want;
    if (w && w.row === row && w.a === a && w.b === b) return;
    S.want = { row: row, a: a, b: b };
    clearTimeout(S.fetchT);
    S.fetchT = setTimeout(fetchCells, 140);
  }

  async function fetchCells() {
    var w = S.want;
    if (!w || S.loading) return;
    var per = S.period, asset = S.asset, miss = S.D.d.slice(w.a, w.b + 1).filter(function (k) { return !S.cells.has(w.row + "|" + k) || S.stale.has("c" + w.row + "|" + k); });
    if (!miss.length) return;
    S.loading = true;
    try {
      var r = await fetch(P().cells + "?" + Q() + "&from=" + miss[0] + "&to=" + miss[miss.length - 1] + "&row=" + w.row, { cache: "no-store" });
      if (r.ok) {
        var p = await r.json();
        if (per === S.period && asset === S.asset) miss.forEach(function (k) { S.cells.set(w.row + "|" + k, p.cells[k] || null); S.stale.delete("c" + w.row + "|" + k); });   // 리뷰 3
      }
    } catch (e) { /* 다음 그리기 때 다시 묻는다 */ }
    S.loading = false;
    redraw();
    if (S.want !== w) fetchCells();
  }

  // ── 그리기 ─────────────────────────────────────────────────────────────────────────
  function layout(cv) {
    var W = cv.clientWidth, H = cv.clientHeight, narrow = W < 560;
    var L = 8, R = narrow ? 52 : 66, axisH = 26, gap = 8;
    var laneH = Math.max(30, Math.round((H - axisH) * (narrow ? 0.07 : 0.055)));   // 2026-11-04 펜 메모 «차트 높이가 너무 낮다 -- 아래 막대 높이를 줄이고 차트를 늘려줘»: 레인 0.1 → 0.055(휴대폰 0.095 → 0.07) · 가격 차트 329 → ~500px(776px 상자 실측)
    var LN = lanesNow(), units = LN.reduce(function (t, l) { return t + (l[2] || 1); }, 0), fixed = axisH + LN.length * gap + 6;
    // 검증 11-13: 레인 9칸이 가격 칸을 데스크톱 500 → 351px · 휴대폰 → 197px 로 눌렀다 -- 가격 칸이 높이의 55% 아래로 안 가게 레인 한 칸을 줄인다
    laneH = Math.max(14, Math.min(laneH, Math.floor((H - fixed - 0.55 * (H - axisH)) / units)));
    var priceH = H - fixed - units * laneH;
    var lanes = {}, y = 6 + priceH + gap;
    LN.forEach(function (l) { var hh = Math.round(laneH * (l[2] || 1)); lanes[l[0]] = { y: y, h: hh }; y += hh + gap; });
    // 2026-11-13 사용자 «1d 차트에도 오른쪽에 체결·델타 프로파일»: 플롯과 가격 눈금 사이 PW 칸(5분 차트 «체결 · 델타»와 같은 그림)
    var PW = narrow ? 56 : 160;   // 검증 11-13: 휴대폰 4h 3봉 칸 숫자가 1~2px 모자라던 폭 -- 70 → 56
    return { W: W, H: H, L: L, R: R, PW: PW, plotW: W - L - R - PW, top: 6, priceH: priceH, lanes: lanes, axisY: H - axisH, narrow: narrow };
  }

  function clampView() {
    S.span = Math.max(P().minSpan, Math.min(S.span, S.n + 4));
    S.i0 = Math.max(-S.span * 0.6, Math.min(S.i0, S.n - S.span * 0.4));
  }

  // 일봉·청산맵 = 5분 차트의 «교체 영역»(app.js renderCandleSvg 가 window.fpRegion 으로 남김 -- 넓은 화면은 왼쪽 열 전체:
  //   풋프린트·레인·모의 판·지지/저항, 좁은 화면은 풋프린트~5분 레인) 위에 **배경 없이** 놓는다(사용자 «시간봉과 스타일 통일 · 너무 검정»).
  //   아래 5분 그림은 svg 에 그 영역만큼 구멍(clip-path evenodd)을 내 숨긴다 -- 카드의 실제 배경이 그대로 비친다.
  //   viewBox = 상자 픽셀이라 좌표 그대로. 🔴SVG 에는 offsetTop 이 없어(undefined → NaN) 화면 좌표 차로 잰다.
  var ALT = { daily: false, bubble: false };   // 2026-10-13 bubble = 체결 버블(bubble_map.js)
  function syncHole() {
    var svg = $("candleSvgSnapshot"), card = $("fpCard"), g = window.fpRegion, on = ALT.daily || ALT.bubble;
    if (card) { card.classList.toggle("fp-alt-on", on); card.classList.toggle("fp-alt-wide", !!(on && g && g.split)); }
    if (!svg) return;
    var clip = "";
    if (on) {
      var W = svg.clientWidth || svg.getBoundingClientRect().width, H = svg.clientHeight || svg.getBoundingClientRect().height;
      var w = g ? g.w : W, h = g ? g.h : H, p = function (x, y) { return Math.round(x) + "px " + Math.round(y) + "px"; };
      clip = "polygon(evenodd, " + [p(0, 0), p(W, 0), p(W, H), p(0, H), p(0, 0), p(w, 0), p(w, h), p(0, h)].join(", ") + ")";
    }
    if (svg.style.clipPath !== clip) svg.style.clipPath = clip;
  }
  function placeOverlay(el) {
    var svg = $("candleSvgSnapshot"), g = window.fpRegion;
    syncHole();
    if (!el || !svg || !el.parentElement) return;
    var r = svg.getBoundingClientRect(), pr = el.parentElement.getBoundingClientRect();
    var x = r.left - pr.left - el.parentElement.clientLeft, y = r.top - pr.top - el.parentElement.clientTop;
    var w = g ? g.w : r.width, h = g ? g.h : r.height;
    var st = el.style, px_ = function (v) { return Math.round(v) + "px"; };
    if (st.left !== px_(x) || st.top !== px_(y) || st.width !== px_(w) || st.height !== px_(h)) {
      st.left = px_(x); st.top = px_(y); st.width = px_(w); st.height = px_(h);
    }
  }
  window.fpPlaceOverlay = placeOverlay;         // liq_profile.js 도 같은 자리에 놓는다
  window.fpAltSet = function (kind, on) { ALT[kind] = !!on; syncHole(); };

  function draw() {
    S.raf = 0;
    if (S.on) placeOverlay($("fpDaily"));
    var cv = $("fpDailyCanvas");
    if (!S.on || !cv || !S.D || !cv.clientWidth) return;
    var dpr = window.devicePixelRatio || 1, G = layout(cv);
    if (cv.width !== Math.round(G.W * dpr) || cv.height !== Math.round(G.H * dpr)) {
      cv.width = Math.round(G.W * dpr); cv.height = Math.round(G.H * dpr);
    }
    var g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, G.W, G.H);
    var C = { good: css("--good"), bad: css("--bad"), turn: css("--turnover"), warn: css("--warn"), text: css("--text"), amber: css("--amber"), accent: css("--accent"),
              muted: css("--muted"), line: css("--line"), soft: css("--soft-line"), bg: css("--chart-bg") || "#171b23",
              mono: css("--font-mono") || "monospace", sans: css("--font-sans") || "sans-serif" };
    var D = S.D, bw = G.plotW / S.span;
    var xOf = function (i) { return G.L + (i - S.i0 + 0.5) * bw; };
    var a = Math.max(0, Math.floor(S.i0)), b = Math.min(S.n - 1, Math.ceil(S.i0 + S.span));
    if (b < a) return;
    var lo = Infinity, hi = -Infinity, i;
    for (i = a; i <= b; i++) { lo = Math.min(lo, D.l[i]); hi = Math.max(hi, D.h[i]); }
    var pad = (hi - lo) * 0.05 || 1; lo = Math.max(0, lo - pad); hi += pad;
    lo += S.yOff; hi += S.yOff;
    var yOf = function (p) { return G.top + (hi - p) / (hi - lo) * G.priceH; };
    var cellMode = bw >= cellMinPx(G), row = rowFor(lo, hi, G.priceH);
    if (cellMode || b - a < 399) scheduleCells(row, a, b);   // 프로파일은 줌과 무관하게 칸이 필요하다 · 서버 상한(400일) 넘으면 프로파일 생략
    scheduleHeat(row, a, b);   // 히트맵을 꺼도 받는다 -- «추정 청산 위·아래» 레인이 같은 데이터를 쓴다(그리기만 켜기/끄기)
    S.view = { G: G, a: a, b: b, lo: lo, hi: hi, bw: bw, row: row, cellMode: cellMode };

    g.save();
    g.beginPath(); g.rect(G.L, 0, G.plotW, G.H); g.clip();
    g.save(); g.beginPath(); g.rect(G.L, G.top, G.plotW, G.priceH); g.clip();   // 가격 칸만 -- 세로로 끌면 봉이 레인으로 넘친다(10-09)
    var step = niceStep((hi - lo) / Math.max(3, G.priceH / 60)), p, y;
    g.strokeStyle = C.line; g.globalAlpha = 0.82; g.lineWidth = 1;      // 5분 차트 .chart-grid 와 같은 격자
    for (p = Math.ceil(lo / step) * step; p <= hi; p += step) { y = Math.round(yOf(p)) + 0.5; g.beginPath(); g.moveTo(G.L, y); g.lineTo(G.L + G.plotW, y); g.stroke(); }
    g.globalAlpha = 1;
    if (S.heatOn) drawHeat(g, G, C, a, b, row, xOf, bw, yOf);            // 봉 아래 배경
    // 검증 11-13: 그리기 범위 a..b 는 양끝에 반쯤 걸친 봉까지라 «보이는 N봉»·to_s 에 화면 밖 봉이 섞였다 -- 중심이 플롯 안인 봉만
    var va = Math.max(0, Math.ceil(S.i0 - 0.5)), vb = Math.min(S.n - 1, Math.floor(S.i0 + S.span - 0.5));
    if (vb < va) { va = a; vb = b; }
    drawLiqWall(g, C, G, row, yOf, va, vb);                                     // 2026-11-13 오른쪽 벽 반쪽 청산 프로파일(봉 뒤)
    for (i = a; i <= b; i++) {
      var x = xOf(i), up = D.c[i] >= D.o[i], col = up ? C.good : C.bad;
      var cell = cellMode ? S.cells.get(row + "|" + D.d[i]) : null;
      if (cell && cell[1].length) drawCells(g, C, cell, row, x, bw, yOf, D, i);
      else if (bw >= 4) {
        var w = Math.max(1, Math.min(bw * 0.62, 26)), y1 = yOf(Math.max(D.o[i], D.c[i])), y2 = yOf(Math.min(D.o[i], D.c[i]));
        g.strokeStyle = col; g.lineWidth = 1;
        g.beginPath(); g.moveTo(Math.round(x) + 0.5, yOf(D.h[i])); g.lineTo(Math.round(x) + 0.5, yOf(D.l[i])); g.stroke();
        g.fillStyle = col; g.globalAlpha = 0.85; g.fillRect(x - w / 2, y1, w, Math.max(1, y2 - y1)); g.globalAlpha = 1;
      } else {
        g.strokeStyle = col; g.lineWidth = Math.max(1, bw * 0.8);
        g.beginPath(); g.moveTo(x, yOf(D.h[i])); g.lineTo(x, yOf(D.l[i])); g.stroke();
      }
      if (D.part[i] === 1) {                       // 형성 중 = 점선 테두리
        g.setLineDash([3, 3]); g.strokeStyle = C.muted; g.lineWidth = 1;
        g.strokeRect(x - bw * 0.47, yOf(D.h[i]) - 2, bw * 0.94, yOf(D.l[i]) - yOf(D.h[i]) + 4); g.setLineDash([]);
      }
    }
    g.restore();
    loadX(); S.after = [];
    drawLanes(g, C, G, a, b, xOf, bw);
    if (S.hover && S.hover.i >= a && S.hover.i <= b) {          // 호버 십자선
      g.strokeStyle = C.muted; g.globalAlpha = 0.55; g.lineWidth = 1; g.setLineDash([2, 3]);
      var hx = Math.round(xOf(S.hover.i)) + 0.5;
      g.beginPath(); g.moveTo(hx, 0); g.lineTo(hx, G.axisY); g.stroke();
      if (S.hover.y < G.top + G.priceH) { y = Math.round(S.hover.y) + 0.5; g.beginPath(); g.moveTo(G.L, y); g.lineTo(G.L + G.plotW, y); g.stroke(); }
      g.setLineDash([]); g.globalAlpha = 1;
    }
    g.restore();
    S.after.forEach(function (f) { f(); });
    drawProfile(g, C, G, a, b, row, yOf);   // 플롯 잘라내기(G.L ~ G.L+plotW) 밖 칸이라 restore 뒤에
    drawLiqSummary(g, C, G, va, vb);
    var last = D.c[S.n - 1], lastY = last >= lo && last <= hi ? yOf(last) : -99;
    g.font = "600 12px " + C.sans; g.textAlign = "left"; g.textBaseline = "middle";
    for (p = Math.ceil(lo / step) * step; p <= hi; p += step) if (Math.abs(yOf(p) - lastY) > 14) halo(g, C, px(p), G.L + G.plotW + G.PW + 6, yOf(p), C.muted);
    if (last >= lo && last <= hi) {                // 현재가 = 5분 차트처럼 상자 없이 굵은 숫자 + 바탕 외곽선
      g.font = "700 13px " + C.sans;
      halo(g, C, px(last), G.L + G.plotW + G.PW + 6, yOf(last), C.text);
    }
    drawDates(g, C, G, a, b, xOf, bw);
    legend(S.hover ? S.hover.i : S.n - 1);
  }

  // 2026-11-13 사용자 «현재가 아래 어떤 포지션이 쌓였고 언제 청산되는지 -- 단위 맞춰 정확한 수치» → 방식 A(검증된 포지션 모델) · «차트 오른쪽 벽에 붙은 반쪽 청산 프로파일»:
  //   남은 계약 = 서버 «포지션 프로파일»(/api/position-profile 30일 창: 5분 OI·거래량 모델 · 하이퍼리퀴드 실측 대비 7일 평균 진입가 오차 27bp · 롱/숏 나눔은 1~3일만 믿을 만함)
  //   청산가 = 진입가 × (1 ∓ 1/레버리지 ± 0.5%) · 레버리지 10·20·25·50·75·100 같은 비중(서버 청산맵 가정) · 이미 지난 청산가 = 청산된 것으로 뺀다.
  //   단위 = 코인 수량(바이낸스 OI 와 같은 단위) · 달러 = 수량 × 지금가 · % = 지금가 대비. 30일 넘은 계약(«옛 묶음», 가격 모름)은 빠진다 -- 요약에 몫을 적는다.
  //   ponytail: 레버리지 비중·유지증거금은 고정 가정 -- 거래소 레버리지 분포를 받게 되면 그 비중으로.
  //   2026-11-13 사용자 «청산 프로파일도 보이는 봉으로 재계산(체결·델타처럼)»: 창 = 보이는 첫 봉 시작 ~ 마지막 봉 끝(서버 4시간 격자) 에 열려 지금 남은 계약.
  //   장부는 최근 30일만 안다 -- 그보다 앞의 보이는 봉은 «모델 밖»으로 요약에 적는다. 범위를 옮기면 다시 받는다(받는 동안은 옛 범위 그림).
  var PP_LEVS = [10, 20, 25, 50, 75, 100], PP_MMR = 0.005;
  function keySec(d) { return Date.parse(d.length > 10 ? d + ":00:00Z" : d + "T00:00:00Z") / 1000; }
  function loadPP(a, b) {
    var D = S.D, f = keySec(D.d[a]), t = keySec(D.d[b]) + (S.period === "4h" ? 14400 : 86400), key = S.asset + "|" + f + "|" + t;
    if (S.ppLoading || (S.PPK === key && Date.now() - S.PPT < (S.PP ? 60000 : 300000))) return;
    S.ppLoading = true; var asset = S.asset, per = S.period;
    // 검증 11-13: ①실패·400(보이는 봉이 31일보다 앞)에 옛 범위 그림을 «보이는 봉» 이름으로 남겼다 → null(요약이 «모델 밖» 줄)
    //   ②받는 동안 1d↔4h 를 바꾸면 늦은 응답이 새 봉 길이에 들어갔다 → 봉 길이도 확인 · 버린 뒤엔 다시 그려 새 범위를 받게
    var done = function (j) { S.ppLoading = false; if (asset !== S.asset || per !== S.period) { redraw(); return; }
      S.PP = j && j.win && j.win.range ? Object.assign(j, { asset: asset, from_s: f, to_s: t, nVis: b - a + 1 }) : null;
      S.PPK = key; S.PPT = Date.now(); redraw(); };
    fetch("/api/position-profile?asset=" + asset + "&from_s=" + f + "&to_s=" + t, { cache: "no-store" }).then(function (r) { return r.ok ? r.json() : null; })
      .then(done).catch(function () { done(null); });
  }
  function ppCalc(row) {
    var J = S.PP, W = J && J.win && J.win.range; if (!W || !W.long || !W.short || !(J.price > 0)) return null;   // 검증 11-13: 서버 1초 링이 비면 price = null
    var P = J.price, bw = J.bw, out = { P: P, liqL: new Map(), liqS: new Map(), lDn: 0, liqDn: 0, liqUp: 0, avgL: W.long.avg, tot: W.tot, oi: J.oi, row: row, lT: W.long.tot, sT: W.short.tot,
      nVis: J.nVis, outN: W.first_s && J.from_s < W.first_s ? Math.min(J.nVis, Math.floor((W.first_s - J.from_s) / (S.period === "4h" ? 14400 : 86400))) : 0 };
    var add = function (m, price, q) { var k = Math.floor(price / row + 1e-9); m.set(k, (m.get(k) || 0) + q); };
    [["long", true], ["short", false]].forEach(function (sd) {
      var D_ = W[sd[0]], isL = sd[1];
      D_.vals.forEach(function (q, i) {
        if (!(q > 0)) return;
        var e = (D_.lo + i + 0.5) * bw, w = q / PP_LEVS.length;
        if (isL && e < P) out.lDn += q;
        PP_LEVS.forEach(function (L) {
          var lp = isL ? e * (1 - 1 / L + PP_MMR) : e * (1 + 1 / L - PP_MMR);
          if (isL && lp < P) { add(out.liqL, lp, w); out.liqDn += w; }
          if (!isL && lp > P) { add(out.liqS, lp, w); out.liqUp += w; }
        });
      });
    });
    return out;
  }
  function ppTop(m, row, P) { var b = null; m.forEach(function (q, k) { if (!b || q > b[1]) b = [k, q]; }); return b ? { px: (b[0] + 0.5) * row, q: b[1], d: ((b[0] + 0.5) * row / P - 1) * 100 } : null; }
  // 오른쪽 벽에 붙은 반쪽 프로파일 -- 아래 빨강 = 롱 청산 대기 · 위 초록 = 숏 청산 대기 · 봉 뒤(가격 칸 잘라내기 안에서 봉보다 먼저)
  function drawLiqWall(g, C, G, row, yOf, a, b) {
    loadPP(a, b);
    var R = S.PP && S.PP.asset === S.asset ? ppCalc(row) : null; S.ppR = R; if (!R) return;
    var mx = 1e-9, xr = G.L + G.plotW, ww = Math.min(G.narrow ? 120 : 240, G.plotW * 0.22), top = G.top, bot = G.top + G.priceH;
    R.liqL.forEach(function (q) { mx = Math.max(mx, q); }); R.liqS.forEach(function (q) { mx = Math.max(mx, q); });
    [[R.liqL, C.bad], [R.liqS, C.good]].forEach(function (z) {
      z[0].forEach(function (q, k) {
        var yT = Math.max(top, yOf((k + 1) * row)), yB = Math.min(bot, yOf(k * row)); if (yB - yT < 1) return;
        var w = ww * q / mx; g.fillStyle = z[1]; g.globalAlpha = 0.42;
        g.fillRect(xr - w, yT + 0.5, w, Math.max(1, yB - yT - 1)); g.globalAlpha = 1;
      });
    });
    var tL = ppTop(R.liqL, row, R.P), tS = ppTop(R.liqS, row, R.P);   // 가장 큰 칸 = 이름표(화면 안일 때)
    g.font = "700 11px " + C.sans; g.textAlign = "right"; g.textBaseline = "middle";
    [[tL, C.bad, "롱 청산"], [tS, C.good, "숏 청산"]].forEach(function (z) {
      var t = z[0]; if (!t) return; var y = yOf(t.px); if (y < top + 8 || y > bot - 8) return;
      halo(g, C, z[2] + " " + px(t.px) + " · " + (t.d >= 0 ? "+" : "") + t.d.toFixed(1) + "% · " + usd(t.q * R.P), xr - ww * t.q / mx - 6, y, z[1]);
    });
    g.textAlign = "left"; g.textBaseline = "top";
  }
  // 2026-11-13 «롱·숏 우세» -- 근거 셋이 각자 롱/숏/중립을 낸다(합치지 않고 셋 다 보인다).
  //   선물은 롱 계약 수 = 숏 계약 수라 «우세» = 누가 더 급하게 열었고(사분면·포지션 모델) 누가 웃돈을 내나(펀딩).
  //   ① 펀딩: 최근 3일 거래소 평균 vs 기본 0.01%/8h -- 세 거래소 모두 «웃돈 + clamp(0.01% − 웃돈, ±0.05%)» 식이라 0.01% = 선물이 현물 근처,
  //      아래 = 선물 할인(숏이 급함) · 위 = 선물 웃돈(롱이 급함). 문턱 ±0.001%p. 2026 은 대부분 아래(ETH 바이낸스 중앙 0.0038%) -- 분위로 «평소 대비»를 같이 적는다.
  //   ② 가격×OI 사분면(보이는 봉, 바이낸스 선물 · 코인 수량): 그날 OI 변화 부호 × 그날 델타 부호 → 새 롱 · 새 숏 · 숏 커버 · 롱 정리.
  //      롱 순증 = 새 롱 − 롱 정리 · 숏 순증 = 새 숏 − 숏 커버 · 차이가 네 칸 합의 10% 미만이면 중립. ponytail: 일 단위라 그날 안 섞임은 못 본다.
  //   ③ 30일 포지션 모델: 남은 롱 몫 > 55% 롱 · < 45% 숏.
  //   🔴2026-11-13 사용자 «롱 우세가 한 번도 없다»: ①을 기본 0.01% 대비로 판정했더니 2026 ETH 167일 중 163일 «숏»(3일 평균이 0.01% 를 한 번도 안 넘음)
  //     → 롱 우세(근거 2개 롱)가 구조적으로 거의 불가능했다. ①을 «평소 대비» = 최근 90일 그날 평균 중 3일 평균의 분위(≥70 롱 · ≤30 숏)로 바꿨다.
  //     같은 167일 재계산: 펀딩 표 롱 41 · 중립 74 · 숏 23 → 판정 롱 쪽 약하게 34 · 숏 쪽 약하게 39 · 숏 우세 21 · 엇갈림 24 · 중립 18 · 롱 우세 2
  //     (롱 우세가 드문 건 그 기간 사분면이 실제로 숏 쪽이었기 때문). 0.01% 대비 절대 위치(선물 할인/웃돈)는 글로 같이 적는다.
  var F_BASE = 1e-4;
  function fmtF(v) { return (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(4) + "%"; }
  function fundDay(d) {   // 그날 거래소 평균(있는 것만)
    var X = S.X || {}, t = 0, n = 0;
    ["bn_fund", "okx_fund", "bybit_fund"].forEach(function (k) { var v = X[k] && X[k][d]; if (v != null) { t += v; n++; } });
    return n ? t / n : null;
  }
  function lsVerdict(R, a, b) {
    var D = S.D, X = S.X || {}, ev = [];
    var f3 = [], i, j;
    // 2026-11-13 사용자 (가) «보이는 범위로 통일»: 펀딩도 보이는 마지막 봉에서 거꾸로 3일 · 분위 이력 = 그날까지 90일(전엔 늘 오늘 기준이라 과거를 보면 시점이 섞였다)
    var fEnd = null;
    for (i = b; i >= 0 && f3.length < 3; i--) { var fd = fundDay(D.d[i]); if (fd != null) { f3.push(fd); if (fEnd == null) fEnd = D.d[i]; } }
    if (f3.length) {
      var fm = f3.reduce(function (t, v) { return t + v; }, 0) / f3.length, hist = [];
      Object.keys(X.bn_fund || {}).sort().filter(function (d) { return d <= fEnd; }).slice(-90).forEach(function (d) { var v = fundDay(d); if (v != null) hist.push(v); });
      var pr = hist.length >= 30 ? hist.filter(function (v) { return v <= fm; }).length / hist.length : null;
      var fv = pr != null ? (pr >= 0.7 ? 1 : pr <= 0.3 ? -1 : 0) : fm > F_BASE + 1e-5 ? 1 : fm < F_BASE - 1e-5 ? -1 : 0;   // 이력 30일 미만이면 0.01% 기준
      ev.push({ v: fv, s: "펀딩 " + (pr != null ? Math.round(pr * 100) + "분위" : fmtF(fm)),
                l: "펀딩 3일" + (fEnd !== D.d[S.n - 1] ? "(~" + fEnd.slice(5).replace("-", "/") + ")" : "") + " " + fmtF(fm) + "/8h(" + (pr != null ? "최근 " + hist.length + "일 중 " + Math.round(pr * 100) + "분위 · " : "") +
                   (fm < F_BASE - 1e-5 ? "0.01% 아래 = 선물 할인" : fm > F_BASE + 1e-5 ? "0.01% 위 = 선물 웃돈" : "0.01% = 현물 근처") + ")" });
    }
    var nl = 0, ns = 0, cv = 0, cl = 0;
    for (j = Math.max(a, 1); j <= b; j++) {
      if (D.oi[j] == null || D.oi[j - 1] == null) continue;
      var dOI = D.oi[j] - D.oi[j - 1], dl = D.bq[j] - D.sq[j];
      if (dOI > 0) { if (dl > 0) nl += dOI; else ns += dOI; } else if (dl > 0) cv -= dOI; else cl -= dOI;
    }
    var gross = nl + ns + cv + cl, netL = nl - cl, netS = ns - cv, c = A().coin, sq_ = function (v) { return (v >= 0 ? "+" : "−") + qty(Math.abs(v)); };
    if (gross > 0) ev.push({ v: Math.abs(netL - netS) < 0.1 * gross ? 0 : netL > netS ? 1 : -1, s: "사분면",
      l: "사분면 " + (b - Math.max(a, 1) + 1) + "봉(바이낸스 선물) 롱 순증 " + sq_(netL) + " · 숏 순증 " + sq_(netS) + " " + c });
    if (R && R.lT + R.sT > 0) { var sh = R.lT / (R.lT + R.sT);
      ev.push({ v: sh > 0.55 ? 1 : sh < 0.45 ? -1 : 0, s: "계약 롱 " + Math.round(sh * 100) + "%", l: "보이는 봉 남은 계약 롱 " + Math.round(sh * 100) + "%" }); }
    var up = ev.filter(function (e) { return e.v > 0; }).length, dn = ev.filter(function (e) { return e.v < 0; }).length, ne = ev.length - up - dn;
    var word = up && dn ? "엇갈림" : up >= 2 ? "롱 우세" : dn >= 2 ? "숏 우세" : up ? "롱 쪽 약하게" : dn ? "숏 쪽 약하게" : "중립";
    return { word: word, side: up && dn ? 0 : up - dn, cnt: "근거 " + ev.length + "개 중 롱 " + up + " · 숏 " + dn + " · 중립 " + ne, ev: ev };
  }
  function drawLiqSummary(g, C, G, a, b) {   // 왼쪽 위 상자 -- 단위 맞춘 숫자(수량 · 달러 · %)
    var R = S.ppR;   // 검증 11-13: 프로파일이 실패(503·404·모델 밖)해도 펀딩·사분면 판정은 그린다
    if (!R && !(S.X && S.X.bn_fund) && S.period !== "1d") return;
    var V = lsVerdict(R, a, b), vc = V.side > 0 ? C.good : V.side < 0 ? C.bad : C.text, tag = function (e) { return (e.v > 0 ? "롱" : e.v < 0 ? "숏" : "중립"); };
    var P = R ? R.P : 0, c = A().coin, tL = R ? ppTop(R.liqL, R.row, P) : null, tS = R ? ppTop(R.liqS, R.row, P) : null, nar = G.narrow;
    var vLines = nar ? [["롱·숏 " + V.word + " (" + V.cnt.replace("근거 ", "").replace("개 중", "중") + ")", vc], [V.ev.map(function (e) { return e.s + " " + tag(e); }).join(" · "), C.text]]
      : [["롱·숏 → " + V.word + " (" + V.cnt + ")", vc],
         [V.ev.map(function (e) { return e.l + " " + tag(e); }).join(" · "), C.text]];
    var lines = !R || !(R.tot > 0) || R.avgL == null ? [[(S.PP === undefined || S.ppLoading) && !R ? "청산 대기 받는 중" : (nar ? "청산 대기 없음(모델 밖·최근 30일만)" : "청산 대기 없음 -- 보이는 봉이 포지션 모델 밖(최근 30일만 앎)이거나 받지 못함"), C.muted]] : nar ? [["아래 롱 " + qty(R.lDn) + " " + c + " · 평균 진입 " + px(R.avgL) + " " + pct(P / R.avgL - 1), C.text],
                       ["롱 청산 대기 " + usd(R.liqDn * P) + (tL ? " · 최대 " + px(tL.px) + " " + tL.d.toFixed(1) + "%" : ""), C.bad],
                       ["숏 청산 대기 " + usd(R.liqUp * P) + (tS ? " · 최대 " + px(tS.px) + " +" + tS.d.toFixed(1) + "%" : ""), C.good]]
      : [["추정 · 보이는 " + R.nVis + "봉에 열려 남은 계약 " + qty(R.tot) + " " + c + "(OI 의 " + Math.round(100 * R.tot / R.oi) + "%)" + (R.outN ? " · 앞 " + R.outN + "봉은 모델 밖(최근 30일만 앎)" : "") + " · 청산가 = 레버리지 10~100배 같은 비중", C.muted],
         ["지금가 아래 롱 " + qty(R.lDn) + " " + c + " · " + usd(R.lDn * P) + " · 롱 평균 진입 " + px(R.avgL) + " (" + pct(P / R.avgL - 1) + ")", C.text],
         ["아래 롱 청산 대기 " + qty(R.liqDn) + " " + c + " · " + usd(R.liqDn * P) + (tL ? " · 최대 칸 " + px(tL.px) + " (" + tL.d.toFixed(1) + "%) " + usd(tL.q * P) : ""), C.bad],
         ["위 숏 청산 대기 " + qty(R.liqUp) + " " + c + " · " + usd(R.liqUp * P) + (tS ? " · 최대 칸 " + px(tS.px) + " (+" + tS.d.toFixed(1) + "%) " + usd(tS.q * P) : ""), C.good]];
    lines = vLines.concat(lines);
    g.font = "700 " + (nar ? 10 : 12) + "px " + C.sans; g.textAlign = "left"; g.textBaseline = "top";
    var lh = nar ? 13 : 16, bw_ = Math.min(G.plotW - 16, Math.max.apply(null, lines.map(function (t) { return g.measureText(t[0]).width; })) + 16), bx = G.L + 8, by = G.top + 8;
    g.fillStyle = css("--bg") || C.bg; g.globalAlpha = 0.9; g.fillRect(bx, by, bw_, lh * lines.length + 8); g.globalAlpha = 1;   // 펜 메모 11-13: 밝은 테마 --chart-bg 는 반투명(.42)이라 상자가 비쳤다 → 불투명 --bg
    g.strokeStyle = C.line; g.strokeRect(bx + 0.5, by + 0.5, bw_, lh * lines.length + 8);
    lines.forEach(function (t, i) { halo(g, C, t[0], bx + 8, by + 4 + lh * i, t[1]); });
  }

  // 보이는 봉 합산 가격대 프로파일 -- 가운데 선 왼쪽 = 체결(매수+매도, 주황 · 최다 행 테두리) · 오른쪽 = 델타(매수−매도, 초록 +/빨강 −) · 옅은 띠 = 체결 70% 가치 영역.
  //   5분 차트(app.js «체결 · 델타»)와 같은 규약 -- 두 막대는 각자 최대값 기준(길이끼리 비교 안 함). 수량 = 코인 단위. 서술이지 신호가 아님.
  function profAgg(a, b, row) {
    var D = S.D, agg = new Map(), got = 0, i, j;
    for (i = a; i <= b; i++) {
      var cell = S.cells.get(row + "|" + D.d[i]);
      if (cell === undefined) continue;          // 아직 안 받음
      got++;
      if (!cell) continue;
      var k0 = Math.round(cell[0] / row);
      for (j = 0; j < cell[1].length; j++) {
        var bq = cell[1][j], sq = cell[2][j];
        if (!(bq + sq)) continue;
        var e = agg.get(k0 + j) || [0, 0];
        e[0] += bq + sq; e[1] += bq - sq; agg.set(k0 + j, e);
      }
    }
    var tot = 0, dMax = 0, pocK = null, sum = 0, kLo = Infinity, kHi = -Infinity;
    agg.forEach(function (e, k) { sum += e[0]; if (e[0] > tot) { tot = e[0]; pocK = k; } dMax = Math.max(dMax, Math.abs(e[1])); kLo = Math.min(kLo, k); kHi = Math.max(kHi, k); });
    var up = pocK, dn = pocK, acc = tot;            // 가치 영역 = 최다 행에서 위·아래 중 큰 쪽을 붙여 가며 70%
    while (pocK != null && acc < 0.7 * sum && (up < kHi || dn > kLo)) {
      var u = up < kHi ? (agg.get(up + 1) || [0])[0] : -1, d = dn > kLo ? (agg.get(dn - 1) || [0])[0] : -1;
      if (u >= d) { up++; acc += Math.max(u, 0); } else { dn--; acc += Math.max(d, 0); }
    }
    return { agg: agg, got: got, n: b - a + 1, tot: tot, dMax: dMax, pocK: pocK, vaHi: up, vaLo: dn, row: row };
  }
  function drawProfile(g, C, G, a, b, row, yOf) {
    if (!G.PW) return;
    var x0 = G.L + G.plotW + 6, PW = G.PW - 12, mid = x0 + PW / 2, top = G.top, bot = G.top + G.priceH;
    g.font = "700 " + (G.narrow ? 10 : 11) + "px " + C.sans; g.textAlign = "left"; g.textBaseline = "top";
    S.prof = null;
    if (b - a >= 399) { halo(g, C, G.narrow ? "400봉↑" : "체결 · 델타 — 400봉 넘음", x0, top + 2, C.muted); return; }
    var P = profAgg(a, b, row);
    S.prof = { P: P, x0: x0, x1: x0 + PW };
    g.save(); g.beginPath(); g.rect(x0 - 2, top, PW + 4, G.priceH); g.clip();
    if (P.tot > 0) {
      var vaT = Math.max(top, yOf((P.vaHi + 1) * row)), vaB = Math.min(bot, yOf(P.vaLo * row));
      if (vaB > vaT) { g.fillStyle = C.accent; g.globalAlpha = 0.07; g.fillRect(x0, vaT, PW, vaB - vaT); g.globalAlpha = 1; }
      P.agg.forEach(function (e, k) {
        var yT = Math.max(top, yOf((k + 1) * row)), yB = Math.min(bot, yOf(k * row));
        if (yB - yT < 1) return;
        var hh = Math.max(1, yB - yT - 1), ow = (PW / 2 - 1) * e[0] / P.tot, dw = P.dMax > 0 ? (PW / 2 - 1) * Math.abs(e[1]) / P.dMax : 0;
        if (ow > 0.5) {
          g.fillStyle = C.amber; g.globalAlpha = k === P.pocK ? 0.7 : 0.5; g.fillRect(mid - 1 - ow, yT + 0.5, ow, hh); g.globalAlpha = 1;
          if (k === P.pocK) { g.strokeStyle = C.text; g.globalAlpha = 0.9; g.lineWidth = 1.2; g.strokeRect(mid - 1 - ow, yT + 0.5, ow, hh); g.globalAlpha = 1; }
        }
        if (dw > 0.5) { g.fillStyle = e[1] >= 0 ? C.good : C.bad; g.globalAlpha = 0.85; g.fillRect(mid + 1, yT + 0.5, dw, hh); g.globalAlpha = 1; }
      });
      g.strokeStyle = C.soft; g.lineWidth = 1; g.beginPath(); g.moveTo(Math.round(mid) + 0.5, top); g.lineTo(Math.round(mid) + 0.5, bot); g.stroke();
    }
    g.restore();
    halo(g, C, G.narrow ? "체결·델타" : "체결 · 델타", x0, top + 2, C.muted);
    if (P.got < P.n) { g.textBaseline = "bottom"; halo(g, C, "칸 받는 중 " + P.got + "/" + P.n, x0, bot - 2, C.muted); g.textBaseline = "top"; }
  }

  function drawCells(g, C, cell, row, x, bw, yOf, D, i) {
    var lo = cell[0], buy = cell[1], sell = cell[2], mx = 0, top = 0, poc = 0, k, t;
    for (k = 0; k < buy.length; k++) {
      t = buy[k] + sell[k]; mx = Math.max(mx, buy[k], sell[k]);
      if (t > top) { top = t; poc = k; }
    }
    var half = bw * 0.46, hRow = Math.max(1, yOf(lo) - yOf(lo + row));
    var bx = x - half + 3, cx = x + 3, room = half - 6;
    // 2026-10-20 숫자 = 칸에 들어갈 때만(옛 고정 기준 봉 폭 104px 를 대신) -- 가장 긴 숫자(cqty 4글자 «888k»)가 반쪽 칸에 들어가는가로
    //   화면 전체가 같이 켜지고 꺼진다(봉마다 따로 재면 봉끼리 들쭉날쭉). 축소해 겹치기 시작하면 사라지고 확대하면 돌아온다. 글자 9~10px · 행 9px 이상.
    var fs = bw < 70 ? 9 : 10;
    g.font = "600 " + fs + "px " + C.sans;
    var txt = hRow >= 9 && g.measureText("888k").width + 4 <= room;
    g.strokeStyle = D.c[i] >= D.o[i] ? C.good : C.bad; g.lineWidth = 1.5;   // 왼쪽 = OHLC 막대(색 = 양/음봉, 왼 눈금 시가 · 오른 눈금 종가)
    g.beginPath(); g.moveTo(bx, yOf(D.h[i])); g.lineTo(bx, yOf(D.l[i]));
    g.moveTo(bx - 3, yOf(D.o[i])); g.lineTo(bx, yOf(D.o[i])); g.moveTo(bx, yOf(D.c[i])); g.lineTo(bx + 3, yOf(D.c[i])); g.stroke();
    for (k = 0; k < buy.length; k++) {
      t = buy[k] + sell[k];
      if (t <= 0) continue;
      var yTop = yOf(lo + (k + 1) * row), h = Math.max(1, hRow - (hRow > 4 ? 1 : 0));
      var imb = (buy[k] - sell[k]) / t;                        // −1 매도 우세 · +1 매수 우세
      var ws = room * sell[k] / mx, wb = room * buy[k] / mx;
      g.globalAlpha = 0.42 + 0.5 * Math.pow(Math.abs(imb), 0.7);
      g.fillStyle = imb < 0 ? C.bad : imb > 0 ? C.good : C.soft;   // 2026-11-08 펜 메모: 좌우 모두 큰 쪽 색(5분 풋프린트 drawCell 과 같다)
      g.fillRect(cx - ws, yTop, ws, h); g.fillRect(cx, yTop, wb, h);
      g.globalAlpha = 1;
      if (k === poc) { g.strokeStyle = C.warn; g.lineWidth = 1; g.strokeRect(cx - room, yTop + 0.5, room * 2, h - 1); }
      if (txt) {
        g.font = "600 " + fs + "px " + C.sans; g.textBaseline = "middle"; g.fillStyle = C.text;
        g.textAlign = "right"; g.fillText(cqty(sell[k]), cx - 3, yTop + h / 2);
        g.textAlign = "left"; g.fillText(cqty(buy[k]), cx + 3, yTop + h / 2);
      }
    }
    g.strokeStyle = C.soft; g.lineWidth = 1;
    g.beginPath(); g.moveTo(cx + 0.5, yOf(D.h[i])); g.lineTo(cx + 0.5, yOf(D.l[i])); g.stroke();
  }

  // 다른 거래소 일봉(서버 /api/footprint-daily/xcvd · 1시간마다 갱신 · 주식은 빈 묶음) -- 1d 만(4h 는 바이낸스 선물 자체 열만)
  function loadX() {
    // 검증 11-13: 한 번만 받아 UTC 자정 뒤 합 선이 끊기고 오늘 값이 멈췄다 → 10분마다 다시(서버는 1시간 캐시) · 받는 동안은 옛 값 그대로
    if (S.period !== "1d" || S.xLoading || (S.X !== undefined && Date.now() - (S.XT || 0) < 600000)) return;
    S.xLoading = true;
    var asset = S.asset, done = function (j) { S.xLoading = false; if (asset !== S.asset) return; S.X = j || {}; S.XT = Date.now(); redraw(); };
    fetch("/api/footprint-daily/xcvd?" + Q(), { cache: "no-store" }).then(function (r) { return r.ok ? r.json() : {}; }).then(done).catch(function () { done(S.X || {}); });
  }
  var XC = { bn: "#f0b90b", okx: "#d65bd6", bybit: "#4c8dff" };
  function inkOn(col) {   // 바탕색(#rgb·#rrggbb·rgb()) 밝기 → 글자색. 못 읽으면 어두운 글자
    var c = String(col).trim(), m = c.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i), r, g_, b;
    if (m) { var hx = m[1].length === 3 ? m[1].replace(/./g, "$&$&") : m[1]; r = parseInt(hx.slice(0, 2), 16); g_ = parseInt(hx.slice(2, 4), 16); b = parseInt(hx.slice(4, 6), 16); }
    else { m = c.match(/rgba?\(\s*([\d.]+)[ ,]+([\d.]+)[ ,]+([\d.]+)/); if (!m) return "#111318"; r = +m[1]; g_ = +m[2]; b = +m[3]; }
    return 0.299 * r + 0.587 * g_ + 0.114 * b > 128 ? "#111318" : "#ffffff";
  }
  function xSeries(a, b) {
    var D = S.D, X = S.X || {}, i;
    var mk = function (fn) { var v = []; for (i = a; i <= b; i++) v.push(fn(i)); return v; };
    var m = function (key) { return function (i) { var o = X[key]; var v = o ? o[D.d[i]] : null; return v == null ? null : v; }; };
    return [
      { n: "바이낸스 선물", s: "BN 선", col: XC.bn, kind: "cvd", grp: "fut", v: mk(function (i) { return D.bv[i] - D.sv[i]; }) },
      { n: "OKX 선물", s: "OKX 선", col: XC.okx, kind: "cvd", grp: "fut", v: mk(m("okx_fut")) },
      { n: "바이낸스 현물", s: "BN 현", col: XC.bn, dash: true, kind: "cvd", grp: "spot", v: mk(m("bn_spot")) },
      { n: "OKX 현물", s: "OKX 현", col: XC.okx, dash: true, kind: "cvd", grp: "spot", v: mk(m("okx_spot")) },
      // 2026-11-13 사용자 «OI 차트도 달러 → 포지션 개수(코인), 종가 말고 평균가로»: 바이낸스 = 원래 코인 수량 · OKX(달러로만 옴) = ÷ 그날 평균가
      //   (바이낸스 선물 체결 VWAP = 거래대금 ÷ 수량) · Bybit = 서버가 곱한 바이낸스 현물 종가로 다시 나눠 원래 코인 수량.
      { n: "바이낸스", s: "BN", col: XC.bn, kind: "oi", v: mk(function (i) { return D.oi[i] != null ? D.oi[i] : null; }) },
      { n: "OKX", s: "OKX", col: XC.okx, kind: "oi", v: mk(function (i) { var o = X.okx_oi && X.okx_oi[D.d[i]], q = D.bq[i] + D.sq[i];
          return o != null && q > 0 ? o / ((D.bv[i] + D.sv[i]) / q) : null; }) },
      { n: "Bybit", s: "BYB", col: XC.bybit, kind: "oi", v: mk(function (i) { var o = X.bybit_oi && X.bybit_oi[D.d[i]], c0 = X.px && X.px[D.d[i]];
          return o != null && c0 ? o / c0 : null; }) },
      { n: "바이낸스", s: "BN", col: XC.bn, kind: "fund", v: mk(m("bn_fund")) },
      { n: "OKX", s: "OKX", col: XC.okx, kind: "fund", v: mk(m("okx_fund")) },
      { n: "Bybit", s: "BYB", col: XC.bybit, kind: "fund", v: mk(m("bybit_fund")) }
    ];
  }
  function xCum(se) {   // CVD = 보이는 첫 값부터 누적(달러) · OI = 보이는 첫 값 대비 변화(코인 수량)
    var out = [], acc = 0, base = null;
    se.v.forEach(function (v) {
      if (v == null) { out.push(null); return; }
      if (se.kind === "oi") { if (base == null) base = v; out.push(v - base); } else { acc += v; out.push(acc); }
    });
    return out;
  }
  function drawX(g, C, G, a, b, xOf, k, name, y, h) {
    var all = xSeries(a, b), pick = all.filter(function (se) {
      return se.kind === (k === "xoi" ? "oi" : k === "xfund" ? "fund" : "cvd");
    });
    var lines = pick.map(function (se) { return { se: se, c: se.kind === "fund" ? se.v : xCum(se) }; }).filter(function (l) { return l.c.some(function (v) { return v != null; }); });
    if (k === "xcvd") {   // 합 = 그날 모든 원천이 있는 날만(원천마다 시작일이 달라 섞으면 계단이 생긴다)
      [["fut", "선물 합", C.text], ["spot", "현물 합", css("--spot")]].forEach(function (gp) {
        var mem = pick.filter(function (se) { return se.grp === gp[0]; });
        if (mem.length < 2) return;
        var acc = 0, c = mem[0].v.map(function (_, j) {
          if (mem.some(function (se) { return se.v[j] == null; })) return null;
          acc += mem.reduce(function (t, se) { return t + se.v[j]; }, 0); return acc;
        });
        if (c.some(function (v) { return v != null; })) lines.push({ se: { n: gp[1], s: gp[1].replace(" 합", "합"), col: gp[2], sum: true }, c: c });
      });
    }
    var top = y + 18, hh = h - 22, lo = 0, hi = 0;
    lines.forEach(function (l) { l.c.forEach(function (v) { if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }); });
    if (k === "xfund") hi = Math.max(hi, F_BASE * 1.2);   // 기본 0.01% 선이 늘 보이게
    g.font = "700 12px " + C.sans; g.textAlign = "left"; g.textBaseline = "top";
    halo(g, C, name, G.L + 4, y + 4, C.muted);
    var tw = g.measureText(name).width + 8;
    halo(g, C, k === "xfund" ? (G.narrow ? "그날 평균 · 점선 0.01%" : "그날 정산 평균(8시간마다 · %) · 점선 = 기본 0.01%(선물 ≈ 현물) · 아래 = 선물 할인(숏이 급함) · 위 = 선물 웃돈(롱이 급함)")
      : k === "xoi" ? (G.narrow ? "첫 봉 대비 " + A().coin : "보이는 첫 봉 대비 변화(" + A().coin + " 수량 · 달러로만 오는 OKX 는 ÷ 그날 평균가) · 노랑 바이낸스 · 분홍 OKX · 파랑 Bybit")
      : (G.narrow ? (b - a + 1) + "봉 누적" : "보이는 " + (b - a + 1) + "봉 누적 · 실선 선물 · 점선 현물 · 굵은 선 = 합(모든 원천이 있는 날)") + (S.X === undefined && S.period === "1d" ? " · 불러오는 중" : ""), G.L + 4 + tw, y + 4, C.muted);
    if (!(hi > lo)) return;
    var Y = function (v) { return top + hh - (v - lo) / (hi - lo) * hh; };
    g.strokeStyle = C.line; g.setLineDash([2, 3]); g.lineWidth = 1; g.beginPath(); g.moveTo(G.L, Math.round(Y(0)) + 0.5); g.lineTo(G.L + G.plotW, Math.round(Y(0)) + 0.5); g.stroke(); g.setLineDash([]);
    if (k === "xfund") { g.strokeStyle = C.muted; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(G.L, Math.round(Y(F_BASE)) + 0.5); g.lineTo(G.L + G.plotW, Math.round(Y(F_BASE)) + 0.5); g.stroke(); g.setLineDash([]); }
    var tags = [];
    lines.forEach(function (l) {
      g.strokeStyle = l.se.col; g.lineWidth = l.se.sum ? 2.6 : 1.6; g.globalAlpha = l.se.sum ? 1 : 0.9; g.setLineDash(l.se.dash ? [5, 3] : []);
      g.beginPath(); var pen = false, last = null;
      l.c.forEach(function (v, j) { if (v == null) { pen = false; return; } var xx = xOf(a + j), yy = Y(v); if (pen) g.lineTo(xx, yy); else g.moveTo(xx, yy); pen = true; last = v; });
      g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
      if (last != null) tags.push({ y: Y(last), v: last, se: l.se });
    });
    // 오른쪽 이름표(킹피셔식 색 상자) -- 끝 값 높이에서 겹치면 아래로 민다
    tags.sort(function (p, q) { return p.y - q.y; });
    var th = 14, x0 = G.L + G.plotW + 6, w = G.PW - 12, yy0 = top - 2;
    tags.forEach(function (t) { t.ty = Math.max(yy0, t.y - th / 2); yy0 = t.ty + th + 1; });
    var over = yy0 - (top + hh + 4);
    if (over > 0) tags.forEach(function (t) { t.ty -= over; });
    S.after.push(function () {   // 이름표 칸은 플롯 잘라내기 밖 -- restore 뒤에 그린다
    g.font = "700 10px " + C.sans; g.textBaseline = "middle";
    tags.forEach(function (t) {
      // 2026-11-13 펜 메모 «글래스(밝은) 모드에서 이름표 배경이 안 보임»: 선물 합(--text)·현물 합(--spot)은 밝은 테마에서 어두운 색인데 글자도 어두운 색 고정이었다
      //   → 글자색 = 바탕 밝기로(어두운 바탕 = 흰 글자) · 점선(현물) 바탕 0.55 → 0.78(밝은 화면에서 흐려 바탕이 안 보였다).
      g.fillStyle = t.se.col; g.globalAlpha = t.se.dash ? 0.78 : 0.92; g.fillRect(x0, t.ty, w, th); g.globalAlpha = 1;
      g.fillStyle = inkOn(t.se.col); g.textAlign = "left";
      if (!G.narrow) g.fillText(t.se.n, x0 + 3, t.ty + th / 2 + 0.5);   // 휴대폰(칸 58px)은 이름과 값이 겹쳤다 -- 값만(색이 거래소)
      g.textAlign = "right"; g.fillText(k === "xfund" ? (G.narrow ? fmtF(t.v).replace("%", "") : fmtF(t.v)) : k === "xoi" ? (t.v >= 0 ? "+" : "−") + cqty(Math.abs(t.v)) : sgnUsd(t.v), x0 + w - 3, t.ty + th / 2 + 0.5);
    });
    g.textAlign = "left"; g.textBaseline = "top";
    });
  }

  function laneVal(k, i) {
    var D = S.D;
    if (k === "delta") return D.bv[i] - D.sv[i];
    if (k === "turn") return D.bv[i] + D.sv[i];
    if (k === "oi") return i > 0 && D.oi[i] != null && D.oi[i - 1] != null ? D.oi[i] / D.oi[i - 1] - 1 : null;
    if (k === "lmap") {   // [고점 위 몫, 저점 아래 몫, 2×OI 달러 | null, 봉 안 몫] -- 히트맵(보이는 날·지금 행 크기)이 있어야 안다
      //   10-09 사용자 «종가가 아니라 고점 위·저점 아래»: 그날 가격이 한 번도 안 닿은 레벨만 위/아래로 센다(봉 안 = 따로).
      //   봉 안 몫은 2025~ 중앙 6%(90분위 17%) -- 같은 1시간봉 안 청산·마지막 봉 같은 판정 빈틈과 그날 새로 생긴 포지션이 섞여 있다.
      var row = S.view && S.view.row, h = row ? S.heat.get(row + "|" + D.d[i]) : null;
      if (!h || !h[2]) return null;
      var up = 0, dn = 0, inr = 0;
      for (var j = 0; j < h[2].length; j++) {
        var pc = h[0] + (j + 0.5) * row;
        if (pc > D.h[i]) up += h[2][j]; else if (pc < D.l[i]) dn += h[2][j]; else inr += h[2][j];
      }
      return [up, dn, D.oi[i] != null ? 2 * D.oi[i] * D.c[i] : null, inr];
    }
    return D.ll[i] == null ? null : [D.ll[i], D.ls[i]];
  }

  function laneText(k, i) {
    var v = laneVal(k, i), D = S.D;
    if (v == null) return k === "liq" ? "모름(수집 전)" : k === "oi" ? (D.oi[i] == null ? (S.period === "4h" ? "모름" : "모름(2022-01 전)") : qty(D.oi[i]) + " " + A().coin) : "모름";
    if (k === "lmap") {
      var sh = (v[0] + v[1] + v[3]) || 1, pu = 100 * v[0] / sh, pd = 100 * v[1] / sh, pm = 100 * v[3] / sh;
      return "고점 위 " + pu.toFixed(0) + "% · 봉 안 " + pm.toFixed(0) + "% · 저점 아래 " + pd.toFixed(0) + "%" + (v[2] ? " · 합 " + usd(v[2]) : " (OI 모름)");
    }
    return k === "liq" ? "롱 " + usd(v[0]) + " · 숏 " + usd(v[1]) : k === "oi" ? pct(v) + " · " + qty(D.oi[i]) + " " + A().coin
      : k === "turn" ? usd(v) : sgnUsd(v);
  }

  function drawLanes(g, C, G, a, b, xOf, bw) {
    var D = S.D, w = Math.max(1, Math.min(bw * 0.62, 44)), hi = S.hover ? S.hover.i : S.n - 1;
    LANES.forEach(function (lane) {
      if (!G.lanes[lane[0]]) return;
      var k = lane[0], name = lane[1], y = G.lanes[k].y, h = G.lanes[k].h, i, v;
      g.strokeStyle = C.soft; g.lineWidth = 1;
      g.beginPath(); g.moveTo(G.L, y + 0.5); g.lineTo(G.L + G.plotW, y + 0.5); g.stroke();
      if (k.charAt(0) === "x") { drawX(g, C, G, a, b, xOf, k, name, y, h); return; }
      var mx = 0, oiLo = Infinity, oiHi = -Infinity;
      for (i = a; i <= b; i++) {
        v = laneVal(k, i);
        if (k === "oi" && D.oi[i] != null) { oiLo = Math.min(oiLo, D.oi[i]); oiHi = Math.max(oiHi, D.oi[i]); }
        if (v == null) continue;
        if (k === "lmap") { mx = 1; continue; }   // 100% 막대 -- 크기 축이 없다
        mx = Math.max(mx, k === "liq" ? Math.max(v[0], v[1]) : Math.abs(v));
      }
      var mid = k === "turn" ? y + h : y + h / 2, unk = null;
      for (i = a; i <= b; i++) {
        v = laneVal(k, i);
        var x = xOf(i);
        if (v == null) { if (unk == null) unk = i; continue; }       // 모름 = 빗금
        if (unk != null) { hatch(g, C, xOf(unk) - bw / 2, y + 1, (i - unk) * bw, h - 1); unk = null; }
        if (!mx) continue;
        if (k === "lmap") {
          var tt = (v[0] + v[1] + v[3]) || 1, fu = v[0] / tt, fm = v[3] / tt, fd = v[1] / tt, top = y + 2, hb = h - 4;
          g.globalAlpha = 0.85; g.fillStyle = C.good; g.fillRect(x - w / 2, top, w, hb * fu);                      // 고점 위 = 숏 청산
          g.fillStyle = C.muted; g.globalAlpha = 0.45; g.fillRect(x - w / 2, top + hb * fu, w, hb * fm);           // 봉 안
          g.globalAlpha = 0.85; g.fillStyle = C.bad; g.fillRect(x - w / 2, top + hb * (fu + fm), w, hb * fd);      // 저점 아래 = 롱 청산
          g.globalAlpha = 1;
          continue;
        }
        if (k === "liq") {
          var hs = (h / 2 - 1) * v[1] / mx, hl = (h / 2 - 1) * v[0] / mx;
          g.fillStyle = C.good; g.fillRect(x - w / 2, mid - hs, w, hs);      // 숏 청산(강제 매수) = 위
          g.fillStyle = C.bad; g.fillRect(x - w / 2, mid, w, hl);            // 롱 청산(강제 매도) = 아래
        } else {
          var hh = (k === "turn" ? h - 2 : h / 2 - 1) * Math.abs(v) / mx;
          g.fillStyle = k === "turn" ? C.turn : v >= 0 ? C.good : C.bad;
          g.globalAlpha = k === "turn" ? 0.75 : 0.9;
          g.fillRect(x - w / 2, k === "turn" || v >= 0 ? mid - hh : mid, w, Math.max(1, hh));
          g.globalAlpha = 1;
        }
      }
      if (unk != null) hatch(g, C, xOf(unk) - bw / 2, y + 1, (b - unk + 1) * bw, h - 1);
      if (k === "lmap") {                                                 // 총액(2×OI 달러) = 가는 선(레인 안 제 축)
        var tLo = Infinity, tHi = -Infinity, vv;
        for (i = a; i <= b; i++) { vv = laneVal(k, i); if (vv && vv[2]) { tLo = Math.min(tLo, vv[2]); tHi = Math.max(tHi, vv[2]); } }
        if (tHi > tLo) {
          g.strokeStyle = C.text; g.globalAlpha = 0.55; g.lineWidth = 1.2; g.beginPath(); var pn = false;
          for (i = a; i <= b; i++) { vv = laneVal(k, i); if (!vv || !vv[2]) { pn = false; continue; } var yt = y + h - 3 - (vv[2] - tLo) / (tHi - tLo) * (h - 6); if (pn) g.lineTo(xOf(i), yt); else g.moveTo(xOf(i), yt); pn = true; }
          g.stroke(); g.globalAlpha = 1;
        }
      }
      if (k === "oi" && oiHi > oiLo) {                  // OI 수준 = 가는 선(레인 안 제 축) · 막대 = 전일 대비
        g.strokeStyle = C.text; g.globalAlpha = 0.45; g.lineWidth = 1.2; g.beginPath();
        var pen = false;
        for (i = a; i <= b; i++) {
          if (D.oi[i] == null) { pen = false; continue; }
          var yy = y + h - 2 - (D.oi[i] - oiLo) / (oiHi - oiLo) * (h - 4);
          if (pen) g.lineTo(xOf(i), yy); else g.moveTo(xOf(i), yy);
          pen = true;
        }
        g.stroke(); g.globalAlpha = 1;
      }
      if (k !== "turn") { g.strokeStyle = C.line; g.beginPath(); g.moveTo(G.L, Math.round(mid) + 0.5); g.lineTo(G.L + G.plotW, Math.round(mid) + 0.5); g.stroke(); }
      g.font = "700 12px " + C.sans; g.textAlign = "left"; g.textBaseline = "top";      // 레인 이름표 = 5분 차트 글자(굵게 · 바탕 외곽선)
      var tw = g.measureText(name).width + 8;
      halo(g, C, name, G.L + 4, y + 4, C.muted);
      halo(g, C, laneText(k, hi), G.L + 4 + tw, y + 4, C.text);
    });
  }

  function hatch(g, C, x, y, w, h) {
    g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
    g.strokeStyle = C.soft; g.lineWidth = 1;
    for (var s = -h; s < w; s += 7) { g.beginPath(); g.moveTo(x + s, y + h); g.lineTo(x + s + h, y); g.stroke(); }
    g.restore();
  }

  function drawDates(g, C, G, a, b, xOf, bw) {
    if (S.period === "4h") return drawDates4h(g, C, G, a, b, xOf, bw);
    var D = S.D, every = bw >= 42 ? "d" : bw >= 7 ? "w" : bw >= 1.2 ? "m" : "y", lastX = -1e9;
    g.font = "700 " + (G.narrow ? 12 : 13) + "px " + C.sans; g.textAlign = "center"; g.textBaseline = "top";   // 5분 차트 x축과 같은 글자
    for (var i = a; i <= b; i++) {
      var d = D.d[i], prev = i > 0 ? D.d[i - 1] : "", lab = null;
      if (every === "d") lab = d.slice(5).replace("-", "/");
      else if (every === "w") lab = new Date(d + "T00:00:00Z").getUTCDay() === 1 ? d.slice(5).replace("-", "/") : null;
      else if (every === "m") lab = d.slice(0, 7) !== prev.slice(0, 7) ? (d.slice(5, 7) === "01" ? d.slice(0, 4) : +d.slice(5, 7) + "월") : null;
      else lab = d.slice(0, 4) !== prev.slice(0, 4) ? d.slice(0, 4) : null;
      if (!lab) continue;
      var x = xOf(i);
      if (x - lastX < 52 || x < G.L + 16 || x > G.L + G.plotW - 16) continue;
      g.fillStyle = C.muted; g.fillText(lab, x, G.axisY + 8); lastX = x;
      g.strokeStyle = C.line; g.lineWidth = 1;
      g.beginPath(); g.moveTo(Math.round(x) + 0.5, G.axisY); g.lineTo(Math.round(x) + 0.5, G.axisY + 5); g.stroke();
    }
  }

  // 4시간봉 x축: 날 시작(KST 09시 = UTC 00시) = «10/19», 봉이 넓으면(64px+) 나머지 칸 = «13시»
  function drawDates4h(g, C, G, a, b, xOf, bw) {
    var D = S.D, lastX = -1e9;
    g.font = "700 " + (G.narrow ? 12 : 13) + "px " + C.sans; g.textAlign = "center"; g.textBaseline = "top";
    for (var i = a; i <= b; i++) {
      var dayStart = D.d[i].slice(11, 13) === "00", d = new Date(D.d[i] + ":00:00Z");
      var t = dayStart ? (d.getMonth() + 1) + "/" + d.getDate() : bw >= 64 ? String(d.getHours()).padStart(2, "0") + "시" : null;
      if (!t) continue;
      var x = xOf(i) - bw / 2;                                   // 칸 시작 = 그 시각
      if (x - lastX < 52 || x < G.L + 16 || x > G.L + G.plotW - 16) continue;
      g.fillStyle = dayStart ? C.text : C.muted; g.fillText(t, x, G.axisY + 8); lastX = x;
      g.strokeStyle = C.line; g.lineWidth = 1;
      g.beginPath(); g.moveTo(Math.round(x) + 0.5, G.axisY); g.lineTo(Math.round(x) + 0.5, G.axisY + 5); g.stroke();
    }
  }

  function halo(g, C, txt, x, y, col) {   // 5분 차트 글자 규약: stroke var(--chart-bg) 3px · paint-order stroke
    g.lineWidth = 3; g.lineJoin = "round"; g.strokeStyle = C.bg; g.strokeText(txt, x, y);
    g.fillStyle = col; g.fillText(txt, x, y);
  }

  function niceStep(raw) {
    var p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
    return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
  }

  function legend(i) {
    var box = $("fpDailyLegend");
    if (!box || !S.D || i == null || i < 0 || i >= S.n) return;
    var D = S.D, chg = i > 0 ? D.c[i] / D.c[i - 1] - 1 : null, delta = D.bv[i] - D.sv[i];
    var tags = [D.part[i] === 1 ? "형성 중 · 1분마다 갱신" : "", D.src[i] === "t" ? "체결 테이프" : "aggTrades 아카이브",
                D.part[i] === 2 ? "청산 수집 첫날(반쪽)" : ""].filter(Boolean);
    var span = function (k, v, cls) { return '<span class="fpd-k">' + k + '</span><span class="fpd-v' + (cls ? " " + cls : "") + '">' + v + "</span>"; };
    box.innerHTML = '<b class="fpd-date">' + lab(D.d[i]) + "</b>" +
      span("시", px(D.o[i])) + span("고", px(D.h[i])) + span("저", px(D.l[i])) + span("종", px(D.c[i])) +
      (chg == null ? "" : '<span class="fpd-v ' + (chg >= 0 ? "fpd-pos" : "fpd-neg") + '">' + pct(chg) + "</span>") +
      span("델타", sgnUsd(delta), delta >= 0 ? "fpd-pos" : "fpd-neg") + span("거래대금", usd(D.bv[i] + D.sv[i])) +
      '<span class="fpd-tag">' + tags.join(" · ") + "</span>";
  }

  // ── 조작 ───────────────────────────────────────────────────────────────────────────
  function redraw() { if (!S.raf) S.raf = requestAnimationFrame(draw); }
  function idxAt(clientX) {
    var cv = $("fpDailyCanvas"), r = cv.getBoundingClientRect(), G = layout(cv);
    return S.i0 + (clientX - r.left - G.L) / (G.plotW / S.span) - 0.5;
  }
  function zoomAt(clientX, factor) {
    var at = idxAt(clientX) + 0.5, span = Math.max(P().minSpan, Math.min(S.span * factor, S.n + 4));
    S.i0 = at - (at - S.i0) * span / S.span;
    S.span = span; clampView(); redraw();
  }
  function setView(span) {
    S.span = span === "all" ? S.n + 2 : span;
    S.i0 = S.n - S.span + Math.min(1.5, S.span * 0.1); S.yOff = 0; clampView(); redraw();   // 최신 날 · 자동 축 · 오른쪽 빈 자리 = 1.5칸(봉이 적으면 10%, 10-20 휴대폰 3봉)
  }
  function panBy(dxPx) { S.i0 -= dxPx / (layout($("fpDailyCanvas")).plotW / S.span); clampView(); redraw(); }

  function bindCanvas(cv) {
    cv.addEventListener("wheel", function (e) {
      e.preventDefault();
      if (Math.abs(e.deltaX) > Math.abs(e.deltaY) || e.shiftKey) panBy(-(e.deltaX || e.deltaY));
      else zoomAt(e.clientX, Math.exp(e.deltaY * 0.0015));
    }, { passive: false });
    cv.addEventListener("pointerdown", function (e) {
      if (e.pointerType === "touch" || e.button !== 0) return;
      var v = S.view;
      S.drag = { x: e.clientX, y: e.clientY, i0: S.i0, yOff: S.yOff, k: v ? (v.hi - v.lo) / v.G.priceH : 0, moved: false, id: e.pointerId };
    });
    cv.addEventListener("pointermove", function (e) {
      if (S.drag && (e.buttons & 1)) {
        var dx = e.clientX - S.drag.x, dy = e.clientY - S.drag.y;
        if (!S.drag.moved && Math.abs(dx) < 4 && Math.abs(dy) < 4) return;
        if (!S.drag.moved) { S.drag.moved = true; cv.setPointerCapture(S.drag.id); cv.classList.add("panning"); tipHide(); }
        if (S.drag.k) S.yOff = S.drag.yOff + dy * S.drag.k;   // 아래로 끌면 위쪽 가격(5분 차트와 같다)
        S.i0 = S.drag.i0 - dx / (layout(cv).plotW / S.span); clampView(); redraw();
        return;
      }
      if (e.pointerType !== "touch") hoverAt(e);
    });
    var stop = function () { S.drag = null; cv.classList.remove("panning"); };
    cv.addEventListener("pointerup", stop); cv.addEventListener("pointercancel", stop);
    cv.addEventListener("pointerleave", function () { S.hover = null; tipHide(); redraw(); });
    cv.addEventListener("dblclick", function () { setView(defSpan()); });
    // 터치: 한 손가락 = 이동(짧게 탭 = 그날 읽기) · 두 손가락 = 확대
    cv.addEventListener("touchstart", function (e) {
      if (e.touches.length === 2) {
        var p = e.touches[0], q = e.touches[1];
        S.pinch = { d: Math.max(20, Math.abs(p.clientX - q.clientX)), span: S.span, i0: S.i0, at: idxAt((p.clientX + q.clientX) / 2) + 0.5 };
        S.drag = null;
      } else if (e.touches.length === 1) S.drag = { x: e.touches[0].clientX, i0: S.i0, moved: false, t: e.touches[0] };
    }, { passive: true });
    cv.addEventListener("touchmove", function (e) {
      if (S.pinch && e.touches.length === 2) {
        e.preventDefault();
        var d = Math.max(20, Math.abs(e.touches[0].clientX - e.touches[1].clientX));
        var span = Math.max(P().minSpan, Math.min(S.pinch.span * S.pinch.d / d, S.n + 4));
        S.i0 = S.pinch.at - (S.pinch.at - S.pinch.i0) * span / S.pinch.span; S.span = span; clampView(); redraw();
      } else if (S.drag && e.touches.length === 1) {
        var dx = e.touches[0].clientX - S.drag.x;
        if (!S.drag.moved && Math.abs(dx) < 6) return;
        e.preventDefault(); S.drag.moved = true;
        S.i0 = S.drag.i0 - dx / (layout(cv).plotW / S.span); clampView(); redraw();
      }
    }, { passive: false });
    cv.addEventListener("touchend", function (e) {
      if (S.drag && !S.drag.moved && S.drag.t) hoverAt(S.drag.t);
      S.pinch = null; S.drag = null;
    });
    cv.addEventListener("keydown", function (e) {
      var k = e.key;
      if (k === "ArrowLeft" || k === "ArrowRight") S.i0 += (k === "ArrowLeft" ? -0.2 : 0.2) * S.span;
      else if (k === "+" || k === "=") S.span /= 1.25;
      else if (k === "-" || k === "_") S.span *= 1.25;
      else if (k === "Home") { e.preventDefault(); setView("all"); return; }
      else if (k === "End") { e.preventDefault(); setView(defSpan()); return; }
      else return;
      e.preventDefault(); clampView(); redraw();
    });
  }

  function hoverAt(e) {
    var v = S.view;
    if (!v || !S.D) return;
    var cv = $("fpDailyCanvas"), r = cv.getBoundingClientRect();
    var i = Math.round(idxAt(e.clientX)), y = e.clientY - r.top, D = S.D, html = null, mx = e.clientX - r.left;
    if (S.prof && mx >= S.prof.x0 - 2 && mx <= S.prof.x1 + 2 && y >= v.G.top && y < v.G.top + v.G.priceH) {   // 프로파일 칸 = 보이는 봉 합계 그 행
      var pp = v.hi - (y - v.G.top) / v.G.priceH * (v.hi - v.lo), P = S.prof.P, kk = Math.floor(pp / P.row + 1e-9), ee = P.agg.get(kk);
      S.hover = null;
      if (ee) tipShow(e.pageX, e.pageY, "<b>$" + px(kk * P.row) + "–" + px((kk + 1) * P.row) + " · 보이는 " + P.n + (S.period === "4h" ? "칸" : "일") + " 합</b><br>체결 " + qty(ee[0]) + " " + A().coin +
        " (≈" + usd(ee[0] * (kk + 0.5) * P.row) + ")" + (kk === P.pocK ? " · 최다" : "") + '<br>델타 <span class="' + (ee[1] >= 0 ? "fpd-pos" : "fpd-neg") + '">' + (ee[1] >= 0 ? "+" : "−") + qty(Math.abs(ee[1])) + " " + A().coin + "</span>" +
        (kk >= P.vaLo && kk <= P.vaHi ? " · 가치 영역(70%) 안" : ""));
      else tipHide();
      redraw();
      return;
    }
    if (i < 0 || i >= S.n) { S.hover = null; tipHide(); redraw(); return; }
    S.hover = { i: i, y: y };
    if (y < v.G.top + v.G.priceH) {                    // 가격 칸 위면 그 행
      var p = v.hi - (y - v.G.top) / v.G.priceH * (v.hi - v.lo);
      var cell = v.cellMode ? S.cells.get(v.row + "|" + D.d[i]) : null;
      if (cell) {
        var k = Math.floor(p / v.row + 1e-9) - Math.round(cell[0] / v.row);   // 검증 11-13: floor 는 XRP 칸 하한에서 한 칸 어긋났다
        if (k >= 0 && k < cell[1].length && cell[1][k] + cell[2][k] > 0) {
          var bq = cell[1][k], sq = cell[2][k], lo = cell[0] + k * v.row;
          html = "<b>" + lab(D.d[i]) + " · $" + px(lo) + "–" + px(lo + v.row) + "</b><br>매수 " + qty(bq) + " " + A().coin + " · 매도 " + qty(sq) +
                 " " + A().coin + '<br>차 <span class="' + (bq >= sq ? "fpd-pos" : "fpd-neg") + '">' + (bq >= sq ? "+" : "−") + qty(Math.abs(bq - sq)) + "</span> · 행 $" + v.row;
        }
      }
      var hh = S.heatOn ? S.heat.get(v.row + "|" + D.d[i]) : null;      // 히트맵 칸 = 추정 청산 금액(테두리 칸이 좁아 글자가 안 들어갈 때도 여기서 읽힌다)
      if (hh && hh[2]) {
        var hk = Math.floor(p / v.row + 1e-9) - Math.round(hh[0] / v.row);
        if (hk >= 0 && hk < hh[2].length && hh[2][hk] > 0) {
          var hlo = hh[0] + hk * v.row, oiU = D.oi[i] != null ? D.oi[i] * D.c[i] : null, up = hlo + v.row / 2 >= D.c[i];
          html = (html ? html + "<br>" : "<b>" + lab(D.d[i]) + " · $" + px(hlo) + "–" + px(hlo + v.row) + "</b><br>") +
                 "추정 " + (up ? "숏" : "롱") + " 청산 " + (oiU ? usd(hh[2][hk] * 2 * oiU) : (hh[2][hk] * 100).toFixed(1) + "% (OI 없음)") + " · " + (S.period === "4h" ? "그 칸" : "그날") + " 마감 지도";
        }
      }
    } else {
      html = "<b>" + lab(D.d[i]) + "</b><br>델타 " + laneText("delta", i) + " · 거래대금 " + laneText("turn", i) +
             "<br>OI " + laneText("oi", i) + (S.X && S.X.bn_fund ? "<br>펀딩(그날 평균/8h) " + [["바이낸스", "bn_fund"], ["OKX", "okx_fund"], ["Bybit", "bybit_fund"]].map(function (z) {
               var v = S.X[z[1]] && S.X[z[1]][D.d[i]]; return z[0] + " " + (v == null ? "-" : fmtF(v)); }).join(" · ") : "") + "<br>청산 " + laneText("liq", i) + "<br>추정 청산(" + (S.period === "4h" ? "그 칸" : "그날") + " 마감 7일 지도) " + laneText("lmap", i);
    }
    if (html) tipShow(e.pageX, e.pageY, html); else tipHide();
    redraw();
  }

  // ── 켜고 끄기 ──────────────────────────────────────────────────────────────────────
  var NOTE_1D = null;
  var NOTE_4H = "시각 = 칸 시작(보는 시계 · UTC 00·04·08·12·16·20시 경계 = KST 09·13·17·21·01·05시) · 지난 7일 + 형성 중 칸 · 1d 와 같은 그림을 4시간 칸으로 -- " +
                "칸 = 공격자 매도(왼쪽) / 매수(오른쪽) · 색 = 큰 쪽(매도 빨강 · 매수 초록) · 주황 테두리 = 그 4시간 가장 많이 거래된 가격 · 파란 칸 = 그 칸 마감의 추정 청산 밀도(7일 1시간봉) · " +
                "OI = 막대 직전 칸 대비 · 선 수준 · 청산 = 그 4시간 강제주문 합 · 추정 청산 = 그 칸 마감 지도의 고점 위/봉 안/저점 아래 비율 · 원천 = 우리 체결 테이프·OI·청산 수집기 · 서술이지 신호가 아님";
  function syncPeriodUi() {   // 범위 버튼·설명 줄·aria 를 지금 봉 길이에 맞춘다
    var tools = document.querySelectorAll("#fpDailyTools [data-view]:not([data-view=in]):not([data-view=out])");
    P().views.forEach(function (v, k) { if (tools[k]) { tools[k].dataset.view = v[0]; tools[k].textContent = v[1]; } });
    var note = document.querySelector("#fpDaily .fpd-note");
    if (note) { if (NOTE_1D === null) NOTE_1D = note.innerHTML; note.innerHTML = S.period === "4h" ? NOTE_4H : NOTE_1D; }
    var cv = $("fpDailyCanvas");
    if (cv) cv.setAttribute("aria-label", A().coin + " " + P().name + " 풋프린트 — 가격 칸별 매수·매도와 델타·거래대금·OI·청산. 화살표 = 이동 · +/− = 확대/축소 · Home = 전체 · End = 기본 범위");
  }
  async function setOn(on, period) {
    on = on && assetOn();
    if ((period && period !== S.period) || (on && curAsset() !== S.asset)) {   // 봉 길이·코인이 바뀌면 받은 것·보던 자리를 비운다
      if (period) S.period = period;
      if (on) S.asset = curAsset();
      S.X = undefined; S.PP = undefined; S.ppR = null; S.PPK = null;
      S.D = null; S.n = 0; S.cells.clear(); S.heat.clear(); S.stale.clear(); S.want = null; S.heatWant = null; S.view = null; S.hover = null;
      syncPeriodUi();
    }
    S.on = on;
    var card = $("fpCard"), btn = $("fpDailyBtn"), b4 = $("fpH4Btn"), box = $("fpDaily"), status = $("fpDailyStatus");
    if (card) card.classList.toggle("fp-daily-on", on);
    if (btn) btn.setAttribute("aria-pressed", String(on && S.period === "1d"));
    if (b4) b4.setAttribute("aria-pressed", String(on && S.period === "4h"));
    if (typeof renderChartWindowTabs === "function") renderChartWindowTabs();   // 선택 칸 = 1d ↔ 시간 칸(app.js)
    window.fpAltSet("daily", on);
    if (box) { box.hidden = !on; if (on) placeOverlay(box); }
    try { localStorage.setItem("fpDailyOn", on ? S.period : "0"); } catch (e) { /* 저장 못 해도 동작 */ }
    clearInterval(S.timer);
    tipHide();
    if (!on) return;
    if (!S.D) {
      if (status) status.textContent = P().name + " 불러오는 중…";
      var per = S.period, asset = S.asset;
      try {
        await loadDays();
        if (per !== S.period || asset !== S.asset) return;      // 받는 사이 봉 길이를 또 바꿨다 -- 새 setOn 이 맡는다
        setView(defSpan());
        if (status) status.textContent = "";
      } catch (e) {
        if (per !== S.period || asset !== S.asset) return;      // 버린 쪽의 실패가 새 봉 길이 상태를 지우지 않게(리뷰 3)
        if (status) status.textContent = P().name + "을 못 받았습니다(" + e.message + ") · 버튼을 다시 누르면 재시도";
        S.D = null; S.n = 0;
        return;
      }
    } else redraw();
    if (!S.on) return;   // 첫 일봉을 받는 사이 껐다(켰다) -- 그 사이 setOn 이 이미 타이머를 맡았다
    clearInterval(S.timer);
    S.timer = setInterval(async function () {
      if (!S.on) return;
      try { await loadDays(S.D.d[Math.max(0, S.n - 2)]); redraw(); } catch (e) { /* 다음 주기 */ }
    }, REFRESH_MS);
  }

  function init() {
    var btn = $("fpDailyBtn"), b4 = $("fpH4Btn"), cv = $("fpDailyCanvas");
    if (!btn || !cv) return;
    btn.dataset.title = btn.title;
    btn.addEventListener("click", function () { setOn(!(S.on && S.period === "1d"), "1d"); });
    if (b4) { b4.dataset.title = b4.title; b4.addEventListener("click", function () { setOn(!(S.on && S.period === "4h"), "4h"); }); }
    // 5m 탭을 누르면 5분 화면으로 돌아간다 -- 🔴1d·4h 도 같은 묶음 안이라 [data-bars] 로만(안 그러면 켜자마자 꺼진다)
    document.querySelectorAll("#chartWindowTabs .asset-tab[data-bars]").forEach(function (b) {
      b.addEventListener("click", function () { if (S.on) setOn(false); });
    });
    var hb = document.querySelector("#fpDailyTools [data-heat]");
    try { S.heatOn = localStorage.getItem("fpDailyHeat") !== "0"; } catch (e) { /* 기본 켬 */ }
    if (hb) {
      hb.setAttribute("aria-pressed", String(S.heatOn)); hb.classList.toggle("active", S.heatOn);
      hb.addEventListener("click", function () {
        S.heatOn = !S.heatOn;
        hb.setAttribute("aria-pressed", String(S.heatOn)); hb.classList.toggle("active", S.heatOn);
        try { localStorage.setItem("fpDailyHeat", S.heatOn ? "1" : "0"); } catch (e) { /* 저장 못 해도 동작 */ }
        redraw();
      });
    }
    document.querySelectorAll("#fpDailyTools [data-view]").forEach(function (b) {
      b.addEventListener("click", function () {
        var v = b.dataset.view;
        if (!S.D) return;
        if (v === "in" || v === "out") { var r = cv.getBoundingClientRect(); zoomAt(r.left + r.width * 0.85, v === "in" ? 0.7 : 1.45); }
        else setView(v === "all" ? "all" : Number(b.dataset.view));
      });
    });
    bindCanvas(cv);
    new ResizeObserver(redraw).observe(cv);
    setInterval(function () { if (S.on) placeOverlay($("fpDaily")); }, 500);   // 5분 차트 배치가 바뀌면(창 탭·폭) 따라간다
    // ETH·XRP -- 다른 코인이면 끄고 버튼을 잠근다 · ETH ↔ XRP 면 같은 봉 길이로 새 코인을 다시 받는다(app.js 의 activeSnapshotAsset 을 1초마다 본다)
    setInterval(function () {
      var ok = assetOn();
      btn.disabled = !ok;
      btn.title = ok ? btn.dataset.title : "일봉 풋프린트는 ETH·BTC·XRP 만 있습니다";
      if (b4) { b4.disabled = !ok; b4.title = ok ? b4.dataset.title : "4시간봉 풋프린트는 ETH·BTC·XRP 만 있습니다"; }
      if (S.on && (!ok || curAsset() !== S.asset)) setOn(ok, S.period);
    }, 1000);
    var saved = null;
    try { saved = localStorage.getItem("fpDailyOn"); } catch (e) { /* 없음 */ }
    if (saved === "1" || saved === "1d") setOn(true, "1d");
    else if (saved === "4h") setOn(true, "4h");
  }

  // 2026-11-13 펜 메모 «글래스 모드에서 이름표 배경이 안 보임»: 색은 그릴 때 getComputedStyle 로 읽는다 -- 테마를 바꿔도 다음 갱신(최대 1분)까지
  //   옛 테마 색(다크의 흰 «선물 합» 등)이 밝은 바탕 위에 남았다 → data-theme 이 바뀌면 바로 다시 그린다.
  if (typeof MutationObserver === "function") new MutationObserver(function () { if (S.on) redraw(); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  window.fpDailyActive = function () { return S.on; };   // 시험 하네스가 켜짐을 읽는다
  window.fpDailyPeriod = function () { return S.period; };   // 2026-10-20 탭 선택 칸(app.js renderChartWindowTabs) -- 1d | 4h
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
