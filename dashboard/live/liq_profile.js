// 풋프린트 보기 토글(풋프린트 · 체결 버블) + 오른쪽 기둥 «청산»의 입력(window.fpLiqColTp).
// 2026-10-08 «청산맵» 덮개 보기로 시작 -- 2026-10-27 탭을 없애고 지도를 풋프린트 기둥으로 옮겼다(app.js liqOn). 덮개 그리기·조작 코드는 지웠다.
// 원천 = 1일: app.js 가 60초마다 받는 /api/liquidation-map 의 tier_profile · 7·30일: /api/liquidation-map/tiers?days= (같은 추정 모델, 입력 1시간봉 창만 다름).
// 추정이지 실측 포지션이 아니다(scripts/live_liquidation_map_20260824.py 머리말). 🔴CI esprima: `?.(`·`?.[`·숫자 구분자 금지.
(function () {
  "use strict";
  var S = { ext: {}, extT: {}, sweep: null };
  var $ = function (id) { return document.getElementById(id); };
  var asset = function () { return typeof activeSnapshotAsset === "undefined" ? "eth" : activeSnapshotAsset; };

  function fetchExt(days) {                   // 7·30일: 기둥이 부를 때 60초마다(서버도 60초 캐시) -- 새 진입 반영 1분(10-10 사용자 유지)
    var key = asset() + "|" + days, now = Date.now();
    if (days === 1 || now - (S.extT[key] || 0) < 60000) return;
    S.extT[key] = now;
    fetch("/api/liquidation-map/tiers?asset=" + encodeURIComponent(asset()) + "&days=" + days, { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) {
        if (!j || !j.tier_profile) return;
        S.ext[key] = j;
        if (typeof scheduleSnapshotChartRender === "function") scheduleSnapshotChartRender();
      })
      .catch(function () { S.extT[key] = 0; });
  }
  // 2026-10-10 사용자 «현재가가 청산가를 넘어가면 바로 막대 제거 -- 매초»: 지도가 만들어진 뒤 가격이 지나간 범위(1초 현재가 + 그 뒤 5분봉 고·저)
  //   안의 칸을 0 으로 -- 서버 생존 필터(그 뒤 봉 고·저)와 같은 규칙을 지도 사이 시간에 브라우저에서 이어 붙인다(서버 호출 0).
  //   새 지도가 오면(1·7·30일 60초) 서버가 같은 범위를 이미 걸러 낸 상태라 범위를 그 기준가로 다시 시작한다.
  function sweep(m, tp) {
    var w = S.sweep, cp = tp.current_price, p = livePrice(tp);
    if (!w || w.tp !== tp) w = S.sweep = { tp: tp, lo: cp, hi: cp, out: null, key: "" };
    var lo = Math.min(w.lo, p), hi = Math.max(w.hi, p), a = typeof activeSnapshotAsset === "undefined" ? "eth" : activeSnapshotAsset;
    // 2026-10-11 1일 지도 = 마지막 닫힌 5분봉 끝(known_until)까지의 데이터 -- 그 뒤 봉 = 형성 5분봉부터라 지도 뒤 고·저를 빠짐없이 본다.
    //   7·30일(형성 1시간봉 포함)은 known_until 이 없어 generated_at.
    var gen = Date.parse(tp.known_until || m.generated_at || "") / 1000, cs = typeof candleHistoryByAsset === "undefined" ? null : candleHistoryByAsset[a];
    // 🔴2026-10-11 지도 «뒤»에 시작한 봉만(time >= gen). 지도 시각을 품은 형성 5분봉(time+300 > gen)까지 넣었더니 그 봉의 지도 «앞» 고·저 --
    //   서버가 이미 칸 안 청산가 단위로 걸러 남긴 칸(일부 남은 칸)을 매 갱신마다 다시 통째 0 으로 만들었다(사용자 «업데이트해도 안 채워짐»).
    //   ponytail: 그 봉의 지도 뒤 몫은 1초 현재가 표본(기둥이 그려질 때)만 본다 -- 빠뜨린 건 다음 지도(60초)가 거른다.
    if (gen > 0 && cs) for (var i = cs.length - 1; i >= 0 && cs[i].time >= gen; i--) { lo = Math.min(lo, cs[i].low); hi = Math.max(hi, cs[i].high); }
    w.lo = lo; w.hi = hi;
    return w;
  }
  function liveTp(m) {
    var tp = m && m.tier_profile;
    if (!tp) return tp;
    // 칸 k = 가격 k×bw 를 «가운데»로 한 칸(서버 np.round). 가격이 그 가운데를 지났을 때만 0 -- 5분 차트 밀도 층의 끊기(liqSweepIdx, 칸 가운데)와 같은 규칙.
    //   🔴2026-10-11 전엔 칸 가장자리만 스쳐도 0(ceil(lo/bw−0.5)) → 풋프린트엔 연한 파랑이 남은 칸이 청산맵에선 «추정 청산 없음»(사용자).
    var w = sweep(m, tp), bw = tp.bin_width, k0 = Math.ceil(w.lo / bw), k1 = Math.floor(w.hi / bw), key = k0 + "|" + k1;
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
  function setView(view) {
    var card = $("fpCard");
    if (card) card.classList.toggle("fp-bbm-on", view === "bubble");
    var wt = $("chartWindowTabs");
    if (wt) wt.hidden = view !== "fp";                       // 2026-10-13 체결 버블 = 창 탭 없음(차트 안에서 확대·이동, bubble_map.js)
    window.dispatchEvent(new CustomEvent("fpview", { detail: view }));
    document.querySelectorAll("#fpViewTabs .asset-tab").forEach(function (b) {
      var on = b.dataset.view === view;
      b.classList.toggle("active", on); b.setAttribute("aria-pressed", String(on));
    });
    try { localStorage.setItem("fpView", view); } catch (e) { /* 저장 못 해도 동작 */ }
    if (typeof hideTooltip === "function") hideTooltip();   // app.js 전역
  }

  function init() {
    document.querySelectorAll("#fpViewTabs .asset-tab").forEach(function (b) {
      b.addEventListener("click", function () { setView(b.dataset.view); });
    });
    var saved = null;
    try { saved = localStorage.getItem("fpView"); } catch (e) { /* 없음 */ }
    setView(saved === "bubble" ? saved : "fp");
  }

  // 2026-10-27 풋프린트 오른쪽 기둥 «청산»(사용자 «체결/포지션 칸에 청산을 만들어 청산맵을 풋프린트에 붙여줘») -- 지나간 칸은 지운 지도.
  window.fpLiqColTp = function (days) {
    fetchExt(days);
    return liveTp(days > 1 ? S.ext[asset() + "|" + days] || null : typeof latestLiquidationMap === "undefined" ? null : latestLiquidationMap);
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
