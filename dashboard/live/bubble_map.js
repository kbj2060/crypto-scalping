// 2026-10-13 체결 버블(사용자 «web.bookmap.com 같은 체결 버블 차트» · 시안 승인 «이 차트들을 청산맵 차트 사이즈로 해서 배포해줘»).
// 자리 = 청산맵과 같은 5분 차트 교체 영역(fp_daily.js fpPlaceOverlay · fpAltSet). 보기 토글은 liq_profile.js setView 가 «fpview» 이벤트로 알린다.
// 원천(2026-10-13 사용자 «합산으로 진행») = 체결은 바이낸스 + OKX + Bybit USDT 선물 합산(서버 bubble_tape_payload) · 배경 호가는 바이낸스만.
// 배경 = 호가 래스터(/api/flow/heatmap, 작음 파랑 → 노랑 → 큼 빨강) · 흰 선 = 현재가(래스터 mid) ·
// 버블 = 1초 체결 테이프(/api/bubble-tape, 수집기가 5초마다 써서 ~5초 늦다): $2.5만 미만 점 · 그 이상 원(넓이 ∝ 금액) · 큰 체결 고리·금액.
// 아래 막대 = 구간별 거래대금(매수 | 매도).
// 2026-10-13 사용자 «3·5·10분으로 나누지 말고 진짜 차트처럼 확대/축소해서 왔다 갔다»: 시간축 확대(20초~30분)·끌기 이동(최근 30분 안).
//   테이프 = 30분치 한 번 받고 매초 새 초만 덧붙임(since) · 래스터 = 최근 5분 안이면 1초 해상도(2초마다), 아니면 30분 2초 해상도(10초마다).
//   오른쪽 끝이 «지금»이면 따라간다(실시간), 끌어서 과거를 보면 그 자리에 멈춘다.
(function () {
  var $ = function (id) { return document.getElementById(id); };
  var css = function (n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); };
  var MAX = 1800, MIN_SPAN = 20, DEF_SPAN = 180, MINI_SPAN = 90;   // 미니 = 최근 90초(2026-10-13 사용자 «좀 더 확대»)
  // 보기 상태 -- 큰 오버레이(VM)와 시장 맥락 미니(VN, 2026-10-13 사용자 «미니에서도 확대 축소 실시간»)가 따로 확대·이동한다. 데이터(S)는 함께 쓴다.
  var VM = { span: DEF_SPAN, back: 0, def: DEF_SPAN, geo: null, drag: null, pinch: null, main: true };
  var VN = { span: MINI_SPAN, back: 0, def: MINI_SPAN, geo: null, drag: null, pinch: null, main: false };
  var S = { on: false, asset: "", rows: [], bucket: 0, end: 0, none: false, full: false,
            fine: null, coarse: null, fineT: 0, coarseT: 0, busyT: false, busyF: false, busyC: false, raf: 0 };
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
  function resetData(a) { S.asset = a; S.rows = []; S.end = 0; S.none = false; S.full = false; S.fine = S.coarse = null; S.fineT = S.coarseT = 0; }

  function fetchTape() {
    var a = asset();
    if (a !== S.asset) resetData(a);
    if (!live() || S.busyT || document.hidden || S.none) return;
    var last = S.rows.length ? S.rows[S.rows.length - 1][0] : 0, since = S.full && last ? last - 15 : null;   // 마지막 15초는 다시 받는다(거래소마다 늦게 쓰인 초)
    S.busyT = true;
    fetch("/api/bubble-tape?asset=" + encodeURIComponent(a) + "&sec=" + MAX + (since != null ? "&since=" + since : ""), { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : r.status === 400 ? { rows: [], none: true } : null; })
      .then(function (j) {
        if (!j || a !== S.asset) return;
        if (j.none) { S.none = true; redraw(); return; }
        var keep = since != null ? S.rows.filter(function (r) { return r[0] <= since; }) : [];
        var lo = j.end - MAX - 5;
        S.rows = keep.concat(j.rows).filter(function (r) { return r[0] >= lo; });
        S.bucket = j.bucket; S.end = j.end; S.full = true; redraw();
      })
      .catch(function () { /* 다음 초에 다시 */ })
      .then(function () { S.busyT = false; });
  }
  function fetchHeat(kind) {                // fine = 최근 300초 1초 해상도 · coarse = 최근 30분 2초 해상도
    var a = asset(), now = Date.now(), fine = kind === "fine", busy = fine ? "busyF" : "busyC", tk = fine ? "fineT" : "coarseT";
    if (!live() || S[busy] || document.hidden || now - S[tk] < (fine ? 2000 : 10000)) return;
    S[busy] = true; S[tk] = now;
    fetch("/api/flow/heatmap?symbol=" + encodeURIComponent(a + "usdt") + (fine ? "&cols=300&agg=1" : "&cols=900&agg=2"), { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : r.status === 400 ? { none: true } : null; })
      .then(function (j) {
        if (!j || a !== S.asset) return;
        S[kind] = j.none || !j.qty_i8 ? { none: true }
          : { t0: j.t0_ms / 1000, dt: j.dt_s, cols: j.cols, bs: j.bin_size, blo: j.bin_lo, nb: j.n_bins,
              q: new Int8Array(buf(j.qty_i8)), mid: new Float32Array(buf(j.mid_b64)) };
        redraw();
      })
      .catch(function () { S[tk] = 0; })
      .then(function () { S[busy] = false; });
  }
  // 2026-10-13 시장 맥락 아래 줄 미니(#mcBubble, app.js 가 넓은 화면에서 붙인다) -- 보이면 오버레이가 꺼져 있어도 받는다 · 최근 3분 고정
  function miniCv() { var c = $("mcBubble"); return c && c.offsetParent !== null && c.clientWidth ? c : null; }
  function live() { return S.on || !!miniCv(); }
  function poll() {   // 래스터: 최근 5분 안을 보면(오버레이) 또는 미니가 보이면 1초 해상도 · 오버레이가 그보다 넓으면 30분판도
    fetchTape();
    var m = !!miniCv(), fine = function (V) { return V.back + V.span <= 290; };
    if ((S.on && fine(VM)) || (m && fine(VN)) || (!S.on && !m)) fetchHeat("fine");
    if ((S.on && !fine(VM)) || (m && !fine(VN))) fetchHeat("coarse");
  }

  function draw() {
    S.raf = 0;
    if (S.on) {
      if (typeof window.fpPlaceOverlay === "function") window.fpPlaceOverlay($("bubbleMap"));
      drawInto($("bubbleMapCanvas"), $("bubbleMapStatus"), false);
    }
    var m = miniCv();
    if (m) { if (!m.__bbm) { m.__bbm = true; bindCanvas(m, VN); } drawInto(m, null, true); }
  }
  function drawInto(cv, st, mini) {   // mini = 시장 맥락 아래 줄: 최근 3분 실시간 · 휴대폰과 같은 좌우 꽉(라벨 없이 위 현재가만)
    if (!cv || !cv.clientWidth || !cv.clientHeight) return;
    var W = cv.clientWidth, H = cv.clientHeight, dpr = window.devicePixelRatio || 1, narrow = mini || W < 560;   // 청산맵과 같은 기준
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    var g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    var V = mini ? VN : VM;
    V.geo = null;
    var span = V.span, back = V.back;
    var WIN = span, tEnd = S.end - back, t0 = tEnd - WIN, rows = [], mids = [], lo = Infinity, hi = -Infinity, bk = S.bucket;
    var F = S.fine && !S.fine.none ? S.fine : null, Cc = S.coarse && !S.coarse.none ? S.coarse : null;
    var fineOk = F && F.t0 <= t0 && F.t0 + F.cols * F.dt >= tEnd - 10;   // 보는 창 양 끝을 다 덮어야(낡은 1초판이면 오른쪽 끝이 비고 현재가가 묵는다)
    var M = fineOk ? F : Cc || F;                                        // 보는 창을 다 덮는 래스터 하나만(해상도 섞으면 이음매가 보인다)
    S.rows.forEach(function (r) {
      if (r[0] < t0 || r[0] > tEnd) return;
      var p = r[1] * bk;
      rows.push({ t: r[0], p: p, b: r[2] * p, s: r[3] * p, u: (r[2] + r[3]) * p });
      if (p < lo) lo = p; if (p > hi) hi = p;
    });
    if (M) for (var c = 0; c < M.cols; c++) {
      var tc = M.t0 + c * M.dt, m = M.mid[c];
      if (tc + M.dt <= t0 || tc > tEnd || !(m > 0)) continue;
      mids.push([tc, m]); if (m < lo) lo = m; if (m > hi) hi = m;
    }
    if (st) st.textContent = S.none ? "이 코인은 1초 체결 테이프를 모으지 않습니다" : !S.full ? "체결 버블을 기다리는 중…"
      : !rows.length ? "이 구간에 체결이 없습니다" : "";
    if (!rows.length && !mids.length) return;
    var liveRow = S.rows[S.rows.length - 1], last = (M && M.mid[M.cols - 1] > 0) ? M.mid[M.cols - 1] : liveRow ? liveRow[1] * bk : (lo + hi) / 2;
    var pad = Math.max((hi - lo) * 0.35, last * 0.0002), minW = Math.max(M ? M.bs * 14 : 0, last * 0.0018);
    lo -= pad; hi += pad;
    if (hi - lo < minW) { var cm = (hi + lo) / 2; lo = cm - minW / 2; hi = cm + minW / 2; }   // 래스터 칸이 두꺼워지지 않게
    // 2026-10-13 사용자 «청산맵과 같은 디자인 규칙 · 청산맵 막대 차트 가로·세로에 맞게 채워»: 플롯 상자 = liq_profile.js 와 같은 여백
    //   (좌 68/60 · 우 68/54 · 위 22 · 아래 축 28). 상자 안 위 80% = 호가 히트맵 + 버블, 아래 20% = 거래 막대. 축 글자 = 플롯 밖 · 굵게 · 바탕 외곽선.
    // 2026-10-15 아티팩트 댓글 «체결 버블도 풋프린트처럼 배치 통일»: 왼쪽 여백 68 → 1 -- 플롯 왼쪽 끝 = 범례 첫 글자(카드 왼쪽 18px) = 풋프린트 플롯 왼쪽 끝. 왼쪽 여백엔 그리는 게 없었다(가격·거래대금 축은 오른쪽)
    var L0 = narrow ? 0 : 1, R0 = narrow ? 0 : 68, top = 22, axisH = 28, GAP = 6;   // 2026-10-13 휴대폰 = 좌우 꽉(사용자)
    var PW = W - L0 - R0, IH = H - top - axisH, VH = Math.round(IH * 0.2), PH = IH - VH - GAP, base = top + IH, vy0 = base - VH;
    var X = function (t) { return L0 + (t - t0) / WIN * PW; }, Y = function (p) { return top + PH - (p - lo) / (hi - lo) * PH; };
    V.geo = { t0: t0, L: L0, PW: PW, cv: cv };
    var C = { bg: css("--chart-bg"), text: css("--text"), muted: css("--muted"), line: css("--line"), good: css("--good"), bad: css("--bad") };
    var heatBg = rgb("--bbm-bg"), price = rgb("--bbm-price"), good = C.good, bad = C.bad;
    var sans = css("--font-sans") || "sans-serif";
    var halo = function (txt, x, y, col) { g.lineWidth = 3; g.lineJoin = "round"; g.strokeStyle = C.bg; g.strokeText(txt, x, y); g.fillStyle = col; g.fillText(txt, x, y); };
    var nice = function (raw) { var e = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10)), m = raw / e; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * e; };

    g.save(); g.beginPath(); g.rect(L0, top, PW, PH); g.clip();
    g.fillStyle = heatBg; g.fillRect(L0, top, PW, PH);                // 히트맵 램프가 읽히는 어두운 바탕(가격 칸만)
    if (M) {                                                             // 호가 래스터
      var LUT = lut(), colW = M.dt / WIN * PW, rowH = M.bs / (hi - lo) * PH;
      for (var c2 = 0; c2 < M.cols; c2++) {
        var tc2 = M.t0 + c2 * M.dt;
        if (tc2 + M.dt <= t0 || tc2 > tEnd) continue;
        for (var j = 0; j < M.nb; j++) {
          var v = M.q[c2 * M.nb + j]; if (!v) continue;
          var pj = (M.blo + j) * M.bs; if (pj < lo - M.bs || pj > hi + M.bs) continue;
          g.fillStyle = LUT[Math.min(127, Math.abs(v))]; g.fillRect(X(tc2), Y(pj + M.bs / 2), colW + 0.6, rowH + 0.6);
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
    g.font = "800 " + (narrow ? 10.5 : 12) + "px " + sans; g.textBaseline = "middle"; g.textAlign = "left";
    var placed = [];
    labels.sort(function (p, q) { return q.u - p.u; }).slice(0, WIN > 180 ? 10 : 40).forEach(function (Lb) {   // 금액 글자 -- 긴 창은 상위 10 · 겹치면 작은 쪽 생략
      var txt = usd(Lb.u), w = g.measureText(txt).width, tx = Lb.x + Lb.r + 5;
      if (tx + w > L0 + PW - 4) tx = Lb.x - Lb.r - 5 - w;
      if (placed.some(function (p) { return Math.abs(p.y - Lb.y) < 14 && Math.abs(p.x - tx) < w + 6; })) return;
      placed.push({ x: tx, y: Lb.y });
      g.lineWidth = 4; g.strokeStyle = rgb("--bbm-bg", 0.9); g.strokeText(txt, tx, Lb.y); g.fillStyle = price; g.fillText(txt, tx, Lb.y);
    });
    g.restore();

    // 2026-10-13 사용자 «지지와 저항 라인을 체결 버블 차트에 라인선 추가»: 풋프린트·게이지와 같은 목록(app.js srLevelsLive: 청산맵 레벨 ·
    //   아직 안 쓸린 · 가까운 순 3개씩) · 같은 규약(점선 · 지지 --liq-support / 저항 --liq-resistance · 굵기 ∝ weight_pct · 이름표 왼쪽).
    //   세로 범위가 좁아 밖에 있는 지지1·저항1은 플롯 아래·위 끝에 «이름 가격 ↓/↑ 거리%» 만 붙인다. 🔴서술 -- 반등을 예측하지 않는다.
    var sr = typeof srLevelsLive === "function" ? srLevelsLive(3) : null;
    if (sr) {
      var srCol = { sup: css("--liq-support"), res: css("--liq-resistance") }, edge = [];
      var srTxt = function (txt, x, y, col) { g.lineWidth = 4; g.lineJoin = "round"; g.strokeStyle = rgb("--bbm-bg", 0.9); g.strokeText(txt, x, y); g.fillStyle = col; g.fillText(txt, x, y); };   // 히트맵 위 글자 = 버블 금액과 같은 어두운 외곽선
      var srLine = function (yy, col, w) {
        g.save(); g.beginPath(); g.rect(L0, top, PW, PH); g.clip();
        g.strokeStyle = col; g.lineWidth = w; g.setLineDash([6, 4]); g.beginPath(); g.moveTo(L0, yy); g.lineTo(L0 + PW, yy); g.stroke();
        g.restore();
      };
      g.font = "700 11px " + sans; g.textBaseline = "middle"; g.textAlign = "left";
      ["res", "sup"].forEach(function (side) {
        sr[side].forEach(function (lv, i) {
          var pv = Number(lv.price), tag = (side === "sup" ? "지지" : "저항") + (i + 1);
          if (!(pv > 0)) return;
          if (pv < lo || pv > hi) { if (!i) edge.push({ side: side, tag: tag, p: pv }); return; }
          var yy = Math.round(Y(pv)) + 0.5;
          srLine(yy, srCol[side], Math.max(1, Math.min(4, Math.round(1 + (lv.weight_pct || 0) * 3))));
          srTxt(tag + " " + fmtPx(pv), L0 + 6, yy - 9, srCol[side]);
        });
      });
      edge.forEach(function (e) {
        var d = (e.p / last - 1) * 100, txt = e.tag + " " + fmtPx(e.p) + (e.side === "res" ? " ↑ " : " ↓ ") + (d > 0 ? "+" : "") + d.toFixed(2) + "%";
        var ye = e.side === "res" ? top + 1.5 : top + PH - 1.5;                // 화면 밖 = 그 쪽 끝에 같은 색 점선 + 거리
        srLine(ye, srCol[e.side], 2);
        srTxt(txt, L0 + 6, e.side === "res" ? top + 13 : top + PH - 13, srCol[e.side]);
      });
    }

    // 2026-10-15 사용자 «지난 거래량 막대가 매초 위아래로 흔들림»: 묶음 경계를 화면 왼쪽 끝(t0, 매초 움직임) 대신 시계 정각(BK 배수)에 고정 -- 지난 막대 값은 그대로, 왼쪽으로 미끄러지기만
    var BK = Math.max(1, Math.round(WIN / (narrow ? 30 : 60))), k0 = Math.floor(t0 / BK), nb = Math.floor(tEnd / BK) - k0 + 1, vb = [], vmax = 1;   // 거래 막대
    for (var i = 0; i < nb; i++) vb.push([0, 0]);
    rows.forEach(function (x) { var b = vb[Math.min(nb - 1, Math.max(0, Math.floor(x.t / BK) - k0))]; b[0] += x.b; b[1] += x.s; });
    vb.forEach(function (b) { vmax = Math.max(vmax, b[0], b[1]); });
    var vs = nice(vmax / 2), vtop = vs * Math.ceil(vmax / vs), yV = function (val) { return base - val / vtop * (VH - 2); };
    g.lineWidth = 1; g.strokeStyle = C.line; g.globalAlpha = 0.82;      // 5분 차트 .chart-grid
    for (var gv = vs; gv <= vtop + 1e-9; gv += vs) { var gy = Math.round(yV(gv)) + 0.5; g.beginPath(); g.moveTo(L0, gy); g.lineTo(L0 + PW, gy); g.stroke(); }
    g.globalAlpha = 1;
    var bw2 = PW * BK / WIN, hw = Math.max(1, bw2 * 0.39);
    g.save(); g.beginPath(); g.rect(L0, vy0, PW, VH + 1); g.clip();                  // 양 끝 막대는 화면 경계에서 잘림
    vb.forEach(function (b, i2) {
      var x = X((k0 + i2) * BK) + bw2 * 0.11, hb = base - yV(b[0]), hs = base - yV(b[1]);
      g.globalAlpha = 0.86; g.fillStyle = good; g.fillRect(x, base - hb, hw, hb); g.fillStyle = bad; g.fillRect(x + hw, base - hs, hw, hs);
    });
    g.restore(); g.globalAlpha = 1;

    var ly = Y(last), ps = nice((hi - lo) / Math.max(3, PH / 70)), dec = Math.max(0, -Math.floor(Math.log(ps) / Math.LN10 + 1e-9));
    if (!narrow) {                                                                     // 휴대폰 = 오른쪽 라벨 없음 · 위 «현재가»만(사용자 2026-10-13)
    g.font = "600 12px " + sans; g.textBaseline = "middle"; g.textAlign = "left";    // 가격 축 = 오른쪽(2026-10-13 사용자 «오른쪽에 가격 라벨 되돌려»)
    for (var p = Math.ceil(lo / ps) * ps; p <= hi; p += ps) {
      var y = Y(p); if (y < top + 6 || y > top + PH - 6 || Math.abs(y - ly) < 14) continue;
      halo(p.toFixed(dec), L0 + PW + 6, y, C.muted);
    }
    if (ly >= top && ly <= top + PH) {                                                 // 현재가 = 오른쪽 흰 상자(시안 그대로)
      var lab = fmtPx(last); g.font = "800 " + (narrow ? 11.5 : 13) + "px " + sans;
      var bw = Math.min(R0 - 4, g.measureText(lab).width + 12);
      g.fillStyle = price; g.beginPath();
      if (g.roundRect) g.roundRect(L0 + PW + 2, ly - 11, bw, 22, 6); else g.rect(L0 + PW + 2, ly - 11, bw, 22);
      g.fill(); g.fillStyle = heatBg; g.fillText(lab, L0 + PW + 8, ly + 1);
    }
    g.font = "600 12px " + sans; g.textAlign = "left";                                  // 거래대금 축(오른쪽, 아래 칸)
    for (var gv2 = vs; gv2 <= vtop + 1e-9; gv2 += vs) halo(usd(gv2), L0 + PW + 6, yV(gv2), C.muted);
    halo(BK + "초 합", L0 + PW + 6, base - 8, C.muted);
    }
    g.font = "700 13px " + sans; g.textAlign = "center";                               // 현재가 = 상자 없이 굵은 숫자(청산맵과 같은 규약)
    if (!mini) halo("현재가 " + fmtPx(last) + (back ? " · 지난 구간(더블클릭 = 실시간)" : ""), L0 + PW / 2, 10, C.text);
    else if (back) { g.font = "700 11px " + sans; halo("지난 구간 · 더블클릭 = 실시간", L0 + PW / 2, 10, C.text); }   // 미니 = 현재가 글자 없음(사용자)

    var ev = [30, 60, 120, 300, 600].filter(function (s) { return s >= WIN / Math.max(2, PW / 70); })[0] || 600;   // 시간 축 = 5분 차트 x축 글자
    g.font = "700 " + (narrow ? 12 : 13) + "px " + sans; g.textAlign = "center"; g.textBaseline = "top"; g.strokeStyle = C.line;
    for (var s = Math.ceil(t0 / ev) * ev; s < tEnd; s += ev) {
      var xs = X(s); if (xs < L0 + 24 || xs > L0 + PW - 24) continue;
      g.fillStyle = C.muted; g.fillText(new Date(s * 1000).toTimeString().slice(0, ev < 60 ? 8 : 5), xs, base + 8);
      g.beginPath(); g.moveTo(Math.round(xs) + 0.5, base); g.lineTo(Math.round(xs) + 0.5, base + 5); g.stroke();
      g.globalAlpha = 0.35; g.setLineDash([3, 4]); g.beginPath(); g.moveTo(Math.round(xs) + 0.5, top); g.lineTo(Math.round(xs) + 0.5, base); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    }
    g.strokeStyle = C.line; g.beginPath(); g.moveTo(L0, base + 0.5); g.lineTo(L0 + PW, base + 0.5); g.stroke();
  }

  function redraw() { if (!S.raf) S.raf = requestAnimationFrame(draw); }

  function setView(V, span, back) {          // span = 보이는 초 · back = 오른쪽 끝이 «지금»에서 몇 초 전
    V.span = Math.max(MIN_SPAN, Math.min(MAX, span));
    V.back = Math.max(0, Math.min(MAX - V.span, back));
    if (V.back < 1) V.back = 0;                                          // 끝에 닿으면 다시 실시간으로 따라간다
    if (V.main) { var lv = document.querySelector('#bubbleMapTools [data-view="live"]'); if (lv) lv.setAttribute("aria-pressed", String(V.back === 0 && V.span === DEF_SPAN)); }
    poll(); redraw();
  }
  function timeAt(V, clientX) {
    var v = V.geo; if (!v) return 0;
    var r = v.cv.getBoundingClientRect();
    return v.t0 + (clientX - r.left - v.L) / v.PW * V.span;
  }
  function zoomAt(V, t, f) {                 // t 를 제자리에 두고 시간축만 늘이고 줄인다
    var v = V.geo; if (!v) return;
    var span = Math.max(MIN_SPAN, Math.min(MAX, V.span * f)), frac = (t - v.t0) / V.span;
    setView(V, span, S.end - (t - frac * span + span));
  }
  function panPx(V, dx, back0) { var v = V.geo; if (v) setView(V, V.span, (back0 == null ? V.back : back0) + dx / v.PW * V.span); }
  function resetView(V) { setView(V, V.def, 0); }

  function bindCanvas(cv, V) {               // liq_profile.js bindCanvas 와 같은 조작(축만 시간) -- V = 그 캔버스의 보기 상태
    cv.addEventListener("wheel", function (e) {
      if (!V.geo) return;
      e.preventDefault();
      if (Math.abs(e.deltaX) > Math.abs(e.deltaY) || e.shiftKey) panPx(V, e.deltaX || e.deltaY);
      else zoomAt(V, timeAt(V, e.clientX), Math.exp(e.deltaY * 0.0015));
    }, { passive: false });
    cv.addEventListener("pointerdown", function (e) {
      if (e.pointerType === "touch" || e.button !== 0 || !V.geo) return;
      V.drag = { x: e.clientX, back: V.back, moved: false, id: e.pointerId };
    });
    cv.addEventListener("pointermove", function (e) {
      if (!V.drag || !(e.buttons & 1)) return;
      var dx = e.clientX - V.drag.x;
      if (!V.drag.moved && Math.abs(dx) < 4) return;
      if (!V.drag.moved) { V.drag.moved = true; cv.setPointerCapture(V.drag.id); cv.classList.add("panning"); }
      panPx(V, dx, V.drag.back);
    });
    var stop = function () { V.drag = null; cv.classList.remove("panning"); };
    cv.addEventListener("pointerup", stop); cv.addEventListener("pointercancel", stop);
    cv.addEventListener("dblclick", function () { resetView(V); });
    cv.addEventListener("touchstart", function (e) {         // 한 손가락 = 이동 · 두 손가락 = 확대
      if (!V.geo) return;
      if (e.touches.length === 2) {
        var p = e.touches[0], q = e.touches[1];
        V.pinch = { d: Math.max(20, Math.abs(p.clientX - q.clientX)), span: V.span, at: timeAt(V, (p.clientX + q.clientX) / 2) };
        V.drag = null;
      } else if (e.touches.length === 1) V.drag = { x: e.touches[0].clientX, back: V.back, moved: false };
    }, { passive: true });
    cv.addEventListener("touchmove", function (e) {
      if (V.pinch && e.touches.length === 2) {
        e.preventDefault();
        var f = V.pinch.d / Math.max(20, Math.abs(e.touches[0].clientX - e.touches[1].clientX));
        zoomAt(V, V.pinch.at, V.pinch.span * f / V.span);
      } else if (V.drag && e.touches.length === 1) {
        var dx = e.touches[0].clientX - V.drag.x;
        if (!V.drag.moved && Math.abs(dx) < 6) return;
        e.preventDefault(); V.drag.moved = true;
        panPx(V, dx, V.drag.back);
      }
    }, { passive: false });
    cv.addEventListener("touchend", function () { V.pinch = null; V.drag = null; });
    cv.addEventListener("keydown", function (e) {
      var v = V.geo, k = e.key;
      if (!v) return;
      if (k === "ArrowLeft" || k === "ArrowRight") panPx(V, (k === "ArrowLeft" ? 0.2 : -0.2) * v.PW);
      else if (k === "+" || k === "=") zoomAt(V, v.t0 + V.span / 2, 0.8);
      else if (k === "-" || k === "_") zoomAt(V, v.t0 + V.span / 2, 1.25);
      else if (k === "Home" || k === "End") resetView(V);
      else return;
      e.preventDefault();
    });
  }

  function setOn(on) {
    S.on = on;
    if (typeof window.fpAltSet === "function") window.fpAltSet("bubble", on);
    var box = $("bubbleMap");
    if (box) box.hidden = !on;
    if (on) { S.fineT = S.coarseT = 0; poll(); redraw(); }
  }

  window.addEventListener("fpview", function (e) { setOn(e.detail === "bubble"); });
  window.fpBubbleActive = function () { return S.on; };   // app.js 가 체결 기둥 토글을 숨길 때 읽는다
  function init() {
    var cv = $("bubbleMapCanvas");
    if (!cv) return;
    document.querySelectorAll("#bubbleMapTools [data-view]").forEach(function (b) {
      b.addEventListener("click", function () {
        var v = VM.geo, mid = v ? v.t0 + VM.span / 2 : 0;
        if (b.dataset.view === "live") resetView(VM);
        else if (v) zoomAt(VM, mid, b.dataset.view === "in" ? 0.6 : 1.6);
      });
    });
    bindCanvas(cv, VM);
    new ResizeObserver(redraw).observe(cv);
    setInterval(function () {                     // 테이프 매초 덧붙임 · 래스터 2초(최근 5분 안) / 10초(그 밖) · 코인 탭·5분 차트 배치도 따라간다
      if (!live()) return;
      poll(); redraw();
    }, 1000);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
