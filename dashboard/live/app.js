const API_EVENTS_URL = "/api/events";
const API_OPS_STATUS_URL = "/api/ops-status";
const API_BINANCE_ACCOUNT_URL = "/api/binance-account";
const API_POSITION_SIZING_URL = "/api/position-sizing";
const API_LIQUIDATION_MAP_URL = "/api/liquidation-map";
const API_REGIME_WIDE24_URL = "/api/regime-wide24";
const API_REGIME_BTC_URL = "/api/regime-btc";
const API_REGIME_XRP_URL = "/api/regime-xrp";
const API_COIN_INDICATORS_URL = "/api/coin-indicators";
const API_MACRO_CALENDAR_URL = "/api/macro-calendar";
const API_LIQ_BURST_STATE_URL = "/api/liq-burst-state";
const API_LIQUIDATION_5M_URL = "/api/liquidation-5m-signal";
// 2026-09-11 사용자 "청산맵 차트에 매 5분봉 청산 데이터를 추가" -- 게이지는 현재 봉 하나만
// 주므로 지나간 봉은 이 이력에서 온다. 캔들과 같은 **5분** 정렬이다(게이지 BAR_MINUTES=30 과 별개).
const API_LIQUIDATION_5M_HIST_URL = "/api/liquidation-5m-history";
const API_SESSION_ALERTS_URL = "/api/session-alerts";
// 2026-09-16 2500 -> 1000, 2026-09-19 1000 -> 500. 각 refresh 는 자기 게이트를 갖고 있어
// 이 값은 «게이트를 얼마나 자주 확인하나»일 뿐이다 -- 하지만 그래서 **가장 잦은 게이트의
// 상한**이기도 하다. 🔴FOOTPRINT_POLL_MS 를 500 으로 내렸는데 여기가 1000 이라 실측 간격이
// 1.0초에 묶여 있었다(브라우저에서 요청 간격을 직접 재서 확인). 둘은 같이 움직여야 한다.
// tick() 은 게이트 확인만 하는 디스패처라(각 refresh 가 `now - last < gate` 로 즉시 반환)
// 이 값을 반으로 줄여도 늘어나는 요청은 풋프린트 하나뿐이다.
const POLL_MS = 500;
// 코인별 실시간 지표 폴링(2026-09-03). 서버 캐시가 20초이므로 그보다 자주 때릴 이유가 없다.
const MODEL_INDICATOR_POLL_MS = 20000;
const CANDLE_HISTORY_POLL_MS = 300000;
const MICRO_HISTORY_MAX = 48; // matches MODEL_INDICATOR_HISTORY_MAX in server.py (4h @ 5min samples)
// Kept post-Live-tab-removal solely as the SSE ticker payload's asset allowlist (see
// applyDashboardEvent()) -- eth/btc still need their live price tracked for the Snapshot tab's own
// coin switcher (activeSnapshotAsset), sol is tracked too for parity even though nothing reads it.
const ASSET_CONFIG = {
  eth: { label: "ETH", symbol: "ETHUSDT" },
  sol: { label: "SOL", symbol: "SOLUSDT" },
  btc: { label: "BTC", symbol: "BTCUSDT" },
};

const el = (id) => document.getElementById(id);
const setT = (id, txt) => {
  const target = el(id);
  if (!target || target.textContent === String(txt)) return;
  target.textContent = txt;
};
const setC = (id, cls) => { const target = el(id); if (target && target.className !== cls) target.className = cls; };
const setB = (id, cls) => {
  const target = el(id);
  if (!target) return;
  target.classList.remove("good-border", "bad-border", "warn-border", "neutral-border");
  if (cls) target.classList.add(`${cls}-border`);
};
// 🔴같은 html 을 다시 넣지 않는다(2026-09-16). 현재가 틱마다 청산맵 목록·배지가 통째로
//   다시 쓰였고, 그 줄은 진입/청산 버튼 바로 아래 **유리 헤더**라 재래스터가 눈에 띈다.
//   innerHTML 재대입은 같은 문자열이어도 자식을 전부 버리고 다시 파싱한다.
const setH = (id, html) => { const target = el(id); if (target && target.innerHTML !== html) target.innerHTML = html; };
const hasOwn = (obj, key) => Object.prototype.hasOwnProperty.call(obj || {}, key);

// Snapshot tab: same 5 model indicators (bot-state ones only -- see the "own fetch cycle" model
// indicators like liq_pressure, which carry their own server-provided
// tone_history instead), but as a tone-per-bar strip (matching the evidence signals' activity-strip
// graph) instead of a continuous sparkline. Stores the ALREADY-COMPUTED tone string
// ("good"/"bad"/"neutral") from each render() pass rather than re-deriving it from raw values
// later -- some tones (tail risk) depend on more than one raw field, so capturing the tone at
// computation time is the only way to stay exactly consistent with what the live cards show,
// instead of an approximation that ignores the cross-field dependency.
// 2026-09-21 whale/liq_cascade/retail_flow 칩 제거 -- 남은 띠는 변동성 수준 하나다.
const toneHistory = { vol_level: [] };
// Parallel to toneHistory, same keys/push/shift cadence -- these 5 indicators have no server-side
// timestamp per reading (client-accumulated tally, see comment above), so the only honest per-bar
// time is "when this browser tab actually pushed the reading", recorded here at push time.
const toneHistoryTimes = { vol_level: [] };
// 🔴이 띠는 «48칸 x 5분 = 4시간»이라고 적혀 있는데(MICRO_HISTORY_MAX 주석, 서버의
//   MODEL_INDICATOR_SAMPLE_SECONDS=300), 라이브 경로는 **상태 푸시마다** 한 칸을 밀어 넣고
//   있었다. SSE 는 dashboard_state.json 이 바뀔 때마다 오므로(실측 수 초) 48칸이 몇 분 만에
//   다 차고, 그 과정에서 seedModelIndicatorHistory() 가 서버에서 받아 온 4시간치가 통째로
//   밀려난다 -- 서버가 그 이력을 디스크에 남기는 이유(재기동해도 띠가 안 사라지게)가 매번
//   무효가 됐다. 라이브 푸시도 같은 5분 주기로 맞춘다.
// `at`(ISO)은 **서버 이력 씨앗** 전용이다: 그 샘플이 실제로 찍힌 시각을 그대로 적고(안 그러면
//   48칸이 전부 «지금»이 되어 축이 거짓말을 한다) 주기 제한도 걸지 않는다.
const TONE_PUSH_MIN_MS = 300000;   // = server.py MODEL_INDICATOR_SAMPLE_SECONDS
const toneHistoryLastAt = {};
function pushToneHistory(key, tone, at) {
  const arr = toneHistory[key];
  if (!arr) return;
  const now = Date.now();
  if (at) {
    toneHistoryLastAt[key] = Date.parse(at) || now;   // 다음 라이브 칸이 씨앗과 같은 격자에 온다
  } else {
    if (now - (toneHistoryLastAt[key] || 0) < TONE_PUSH_MIN_MS) return;
    toneHistoryLastAt[key] = now;
  }
  arr.push(tone || "neutral");
  if (arr.length > MICRO_HISTORY_MAX) arr.shift();
  const times = toneHistoryTimes[key];
  times.push(at || new Date(now).toISOString());
  if (times.length > MICRO_HISTORY_MAX) times.shift();
}

let latestMainState = null;
let latestCompactState = null;
let tickInFlight = false;
let latestLivePriceByAsset = {};
let latestLivePriceTsByAsset = {};
let candleHistoryByAsset = {};
let opsStatusEtag = "";
let opsLastFetchAt = 0;
// 2026-09-09 청산맵 신호 마커(C안 하이브리드): 증거신호는 고정 레인, 이벤트 트리거는 봉 밀착.
let latestChartMarkers = null;
let chartMarkersLastFetchAt = 0;
const CHART_MARKERS_POLL_MS = 60000;
const API_CHART_MARKERS_URL = "/api/chart-markers";


// ── 볼륨 풋프린트 (2026-09-15) ────────────────────────────────────────────────
// 캔들 하나를 가격 행으로 쪼개 «그 가격에서 누가 공격했는지»(시장가 매수/매도 체결량)를 보인다.
// 서버 /api/footprint 가 체결 테이프를 누적한다(WS @trade 실시간 + aggTrades 백필) --
// klines 에는 없는 정보다(봉당 taker_buy 합계 하나뿐).
// ETH 전용: 코인마다 스트림·백필이 붙어서, 일단 하나만 켠다.
let latestFootprint = null;
let footprintLastFetchAt = 0;
// ── 창 토글 (2026-09-19 사용자 요청: 1h/2h/4h) ─────────────────────────────
// 풋프린트와 수급 프로파일이 **같은 창**을 쓴다. 한 카드 안의 위아래 두 그림이 서로 다른
// 구간을 말하면 읽는 사람이 속는다 -- 그래서 토글도 하나다.
// 서버 링은 24시간(288봉)이라 4h 도 이미 쌓여 있는 데이터다. 더 긴 창을 안 주는 건 값이
// 없어서가 아니라, 48봉이면 풋프린트 셀이 25px 라 숫자가 이미 안 들어가서다.
const CHART_WINDOW_BARS = [12, 24, 48];   // 5분봉 기준 1h · 2h · 4h
let chartWindowBars = (() => {
  try {
    const saved = Number(localStorage.getItem("chartWindowBars"));
    return CHART_WINDOW_BARS.includes(saved) ? saved : CHART_WINDOW_BARS[0];
  } catch (e) { return CHART_WINDOW_BARS[0]; }
})();
const API_FOOTPRINT_URL = "/api/footprint";
// 🔴여기 있던 "차트 자체가 5초마다 다시 그려진다"는 **틀린 주석**이었다(실제는 20초였다).
// 서버는 WS 로 계속 누적하므로 **이 폴링 간격이 곧 셀의 지연**이다.
//
// 2026-09-19 2000 -> 500 (사용자 지시 "최대한 빨리 보고 주문"). 근거는 추측이 아니라 다른
// 세션이 서버에서 직접 잰 값이다:
//     footprint 데이터 지연 0.1~0.28초 · /api/footprint 응답 1.0~1.3ms · 12KB
//     30회 연속 요청에 에러 0 · 100ms 초과 0
// 즉 **서버·DB·네트워크 어디에도 대기가 없고** 체감 지연은 전부 이 상수였다.
// 서버 비용은 초당 2회 x 1.2ms 라 사실상 0이고, 회선은 gzip 2.0KB/회 기준 시간당 3.6MB ->
// 14MB 로 는다(사용자가 그 대가를 받아들였다).
// ⭐아래 FOOTPRINT_RENDER_MIN_INTERVAL_MS(400ms)가 자연스러운 하한이다 -- 더 당겨도 그리지
//   않으므로 의미가 없다. 그보다 더 빠르게 하려면 폴링이 아니라 SSE 여야 한다(요청 왕복이
//   사라진다). 지금은 필요 없다.
// 🔴이 값을 틱(POLL_MS)과 **같게** 두면 안 된다. 실측에서 간격이 0.5/1.0 초로 튀었다 --
//   틱이 게이트를 «아슬아슬하게 못 넘는» 주기가 섞이기 때문이다(500ms 틱 vs 500ms 게이트).
//   틱보다 낮춰 두면 틱이 곧 페이서가 되어 간격이 500ms 로 고정된다. 이 게이트는 이제
//   «틱이 더 빨라져도 두 번 쏘지 않게» 막는 안전장치 역할만 한다.
const FOOTPRINT_POLL_MS = 400;
// 최신 봉이 이 봉 수를 넘게 묵으면 증분을 포기하고 전량을 다시 받는다(자가복구).
// 2 봉 = 10 분. 정상 상태에서는 최신 봉 나이가 최대 1 봉(5 분)이라 안 걸린다.
const FOOTPRINT_STALE_BARS = 2;
const FOOTPRINT_MIN_ROW_PX = 11;        // 셀에 숫자가 들어가는 최소 행 높이
const FOOTPRINT_IMBALANCE_RATIO = 3;    // TradingView 기본값 300%
// 셀 배경 4단계(TradingView: 최소~최대의 0~25/25~50/50~75/75%~). 매수·매도는 각자 최대로 나눈다.
// 4단계 농담. 라이트에서는 «흰 유리 위」라 같은 알파가 훨씬 옅게 보여 한 단씩 올린다
// (다크 배열은 현행 그대로다). 숫자를 덮지 않는 선이 상한이라 0.66 에서 멈춘다.
// 색을 «채운» 면 위의 글자색. 다크 팔레트는 --good/--bad/--accent 가 밝은 파스텔이라
// 어두운 글자가 맞고, 라이트에서는 같은 토큰이 진해져 흰 글자가 맞다.
// ⭐그 분기는 CSS 의 --on-fill 한 곳에 있다 -- 여기서 색을 정하지 않는다(2026-09-16).
const inkOnFill = () => "var(--on-fill)";
// 가격축 위아래 여백(캔들 고저 폭 대비). 청산맵은 레벨·라벨이 가장자리에 걸려 더 넓게 준다.
// 풋프린트를 같이 넓히면 안 된다 -- ySpan 이 커져 행 높이가 줄고 셀 숫자가 먼저 깨진다.
// 2026-09-21 청산맵 «모드» 가 사라진 뒤에도 이 값은 남는다 -- 풋프린트는 ETH 전용이라
// 다른 코인·테이프 웜업 중에는 **채워진 캔들 폴백**이 그려지고, 그 차트의 여백이 이것이다.
const CHART_Y_PAD_PLAIN = 0.26;       // 2026-09-21 0.15 -> 0.26 (사용자 요청)
const CHART_Y_PAD_FOOTPRINT = 0.15;   // 종전 값 유지
const FOOTPRINT_SHADE_DARK = [0.10, 0.22, 0.36, 0.54];
const FOOTPRINT_SHADE_LIGHT = [0.16, 0.32, 0.48, 0.66];
const footprintShades = () =>
  (document.documentElement.getAttribute("data-theme") === "light"
    ? FOOTPRINT_SHADE_LIGHT : FOOTPRINT_SHADE_DARK);
// ── 리테일/고래 수급 (2026-09-19) ──────────────────────────────────────────
// 서버가 셀을 [매수, 매도, 고래매수, 고래매도, 리테일매수, 리테일매도] 6칸으로 준다.
// 고래·리테일은 총량의 **부분집합**이고 **중형($10k~$100k)은 칸이 없다** -- 셋을 빼서 얻는다.
//
// ⭐경계가 왜 둘인지(2026-09-19 ETHUSDT 39,712건 실측): ≥$10k 는 건수 4.1%인데 **물량 73.1%**,
//   ≥$100k 는 건수 0.17%에 물량 14.6%다. 하나로 가르면 둘 중 하나가 거짓말이 된다 --
//   $10k 를 고래라 하면 물량의 3/4이 고래고, $100k 위만 고래라 하고 나머지를 리테일이라
//   부르면 그 «리테일»의 대부분이 실은 중형이다. 화면은 경계를 숫자로 적어야 한다.
const API_SUPPLY_PROFILE_URL = "/api/supply-profile";
// ── 수급 · 최근 5분 × 1초 (2026-09-19) ────────────────────────────────────
// 서버가 초 단위 칸을 들고 있고, 여기선 **증분만** 받는다(?since=). 300초를 매초 전부 받으면
// 시간당 60MB 가 넘는다 -- 증분이면 보통 한두 줄이다.
// ⚠️서버는 **진행 중인 초를 안 보낸다**(반쪽이 굳는 걸 막으려고). 그래서 맨 오른쪽 막대는
//   최대 2초 묵은 값이다. 그보다 더 당기려면 폴링이 아니라 SSE 여야 한다 -- 지금 필요 없다.
const API_SUPPLY_1S_URL = "/api/supply-1s";
const SUPPLY_1S_POLL_MS = 1000;
const SUPPLY_1S_SEGMENT = 300;          // 누적을 0으로 되돌리는 **벽시계** 경계(초). 5분봉과 같은 자리.
// 계단 눈금(ETH). 자동정규화를 안 쓰는 이유는 renderSupply1s 주석에 있다.
const SUPPLY_1S_STEPS = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000];
let supply1s = new Map();               // 초 -> [리테일매수, 리테일매도, 고래매수, 고래매도, 총매수, 총매도, 가격]
let supply1sMeta = { retailMaxUsd: 0, whaleMinUsd: 0, now: 0 };
// 🔴`since` 는 «내가 **받은** 마지막 초»여야 한다. 서버가 보낸 `now`(진행 중인 초)를 그대로
//   돌려주면 그 초는 영영 안 온다 -- 1차엔 「진행 중」이라 빠지고, 2차엔 「since 이하」라
//   빠진다. 그러면 첫 응답 뒤 매 폴링이 빈 응답이 되어 **차트가 멈춘다**(2026-09-19 첫
//   렌더에서 오른쪽에 공백 띠가 계속 자라는 걸로 드러났다).
let supply1sSince = 0;
let supply1sLastFetchAt = 0;
// 초 -> 그 초의 미결제약정(ETH). 같은 응답에 얹혀 온다(별도 폴링을 하나 더 두지 않는다).
// ⚠️바이낸스가 OI 를 3~7초에 한 번만 갱신한다(서버 OI_1S_URL 주석의 실측) -- 점이 초마다
//   있지 않은 게 정상이다. 창 시작을 0으로 두고 «그 뒤로 몇 계약이 새로 생겼나»를 그린다.
let oi1s = new Map();
let oi1sSince = 0;
// 2026-09-22 초 단위 청산 [롱수량, 숏수량]. oi1s 와 같은 증분 규약(커서는 따로 -- 청산은
// 이벤트가 없는 초가 많아 체결 초 커서를 공유하면 건너뛰어진다).
let liq1s = new Map();
let liq1sSince = 0;
// 5분 누적 패널(청산맵 아래). 같은 1초 스냅샷을 duckdb 로 남긴 것을 서버가 5분으로 접어 준다 --
// 링은 6분뿐이라 몇 시간을 보려면 저장을 거쳐야 한다. 5분 봉이라 15초 폴링으로 충분하다.
// 캔들 SVG 안의 두 하위 패널(중첩 svg)과 그 상자. 각 fetch 가 **그 패널만** 다시 그릴 수
// 있게 들고 있는다. 없으면 두 그림의 갱신이 캔들 SVG 전체 렌더에 묶이는데, 그 렌더에는
// 게이트가 셋이다 -- ①커서가 SVG 위에 있으면 아예 안 그린다(chartHoverActive, 툴팁이
// 지워지지 않게 하려는 장치) ②스크롤 중 정지 ③400~1000ms 스로틀. 패널이 그 SVG 안으로
// 들어오면서(2026-09-19) 「보려고 커서를 올리면 1초 차트가 멈춘다」가 됐다.
// 노드는 캔들 렌더가 다시 붙여 주므로(subPanelCache) isConnected 로 옛 노드를 거른다.
let supply1sSubBox = null;
let supplyProfileSubBox = null;
// ⭐두 패널을 캔들 렌더와 **분리**한다(2026-09-20). 캔들 SVG 는 풋프린트 모드에서 초당 2.5번
//   통째로 다시 그려지는데, 이 두 그림의 입력은 0.2~1Hz 로만 바뀐다. 실측(헤드리스 크로미움,
//   실제 페이로드): renderSnapshotChart 6.1ms 중 **3.9ms(64%)가 이 두 패널**이었고, 그 위에
//   자기 폴링(1초 히트맵 · 1초 수급 · 5초 프로파일)이 또 겹쳐 프로파일은 초당 3.7번 그려졌다.
//   그래서 «판번호»가 바뀌었을 때만 다시 그리고, 아니면 만들어 둔 <svg> 노드를 그대로 다시
//   붙인다(innerHTML="" 은 DOM 에서 떼어낼 뿐 JS 참조가 쥔 서브트리는 살아 있다).
// 🔴현재가는 판번호에 넣지 않는다 -- 그 한 줄만 updateSupplyProfileNow 가 transform 으로
//   따로 옮긴다(원래 설계). 넣으면 틱마다 캐시가 깨져 이 최적화가 통째로 무효가 된다.
let supplyProfileVer = 0, supply1sVer = 0, flowHeatmapVer = 0;
const subPanelCache = { prof: { node: null, key: "" }, s1: { node: null, key: "" },
                        dens: { node: null, key: "" } };
const subProfileKey = (entry, w, h) => `${supplyProfileVer}|${flowHeatmapVer}|${entry}|${w}|${h}`;
const sub1sKey = (w, h) => `${supply1sVer}|${w}|${h}`;

function repaintSupply1sPanel() {
  const b = supply1sSubBox;
  if (!b || !b.svg.isConnected) return;
  renderSupply1s(b);
  subPanelCache.s1.key = sub1sKey(b.w, b.h);
}
function repaintSupplyProfilePanel() {
  const b = supplyProfileSubBox;
  if (!b || !b.svg.isConnected) return;
  const entry = Number(snapshotAccountPosition()?.entry_price || 0);
  renderSupplyProfileSvg(b.svg, latestSupplyProfile,
    Number(latestLivePriceByAsset[activeSnapshotAsset] || 0) || 0,
    entry, { w: b.w, h: b.h });
  subPanelCache.prof.key = subProfileKey(entry, b.w, b.h);
}

const API_OI_5M_URL = "/api/oi-5m";
const OI_5M_POLL_MS = 15000;
// 2026-09-21 상황 읽기 · 30분 (index.html .situation-panel). 서버가 5초마다 계산해 둔 것을 받는다.
const API_SITUATION_URL = "/api/situation";
const SITUATION_POLL_MS = 1000;   // 09-21 서버 계산도 1초로 -- 응답은 작은 JSON 하나
let latestSituation = null;
let situationLastFetchAt = 0;
let latestOi5m = null;
let oi5mLastFetchAt = 0;
const GEX_POLL_MS = 120000;          // 매시 cron -- 2분 폴링이면 충분히 앞선다
// 2026-09-20 호가 창을 **차트 창 탭에 맞춘다**(사용자 지시).
// 🔴전에는 15분 고정이라 1h 탭에서 왼쪽(호가 15분)과 오른쪽(체결 60분)이 4배, 4h 탭에서는
//   16배 다른 구간을 말하고 있었다. index.html 의 창 토글 주석이 경계한 바로 그 상황이다 --
//   «한 카드 안의 두 그림이 다른 구간을 말하면 읽는 사람이 속는다».
const FLOW_HEATMAP_COLS = 300;       // 열 수는 고정 -- 전송량(≈3KB)과 해상도를 함께 묶는다
const flowHeatmapAgg = () =>         // 창 전체를 300열에 담는 초/열
  Math.max(1, Math.min(60, Math.round(chartWindowBars * 300 / FLOW_HEATMAP_COLS)));
// 2026-09-20 1초(사용자 요청). 「새 열이 agg 초마다 하나니 그보다 자주 받아야 같은 그림」
// 이라 3~15초로 묶고 있었는데, **틀렸다** -- 맨 끝 열은 진행 중이라 매초 바뀌고, 창 전체를
// 접어 내는 행 통계도 같이 움직인다. 실측(1h 탭, 1초 간격 두 번): inst 46/272행 2.52% ·
// d60 217/272행 7.37% · refill 246/272행이 바뀐다(peak 만 0). 빈 그림에 돈을 쓰는 게 아니다.
// 🔴비용은 **서버가 1초 SWR 로 묶는다**(server.py api_flow_heatmap) -- 안 그러면 탭 수만큼
//   곱해진다. 한 번 11~32ms(탭별 창 길이), 1Hz 면 코어 1.1~3.2%.
const flowHeatmapPollMs = () => 1000;
const SUPPLY_PROFILE_POLL_MS = 5000;    // 24시간 창이라 더 자주 받아봐야 같은 그림이다
let latestSupplyProfile = null;
let supplyProfileLastFetchAt = 0;
// 2026-09-19 호가 히트맵. 래스터는 1초/열인데 agg=3 으로 접어 받으므로 3초면 새 열이 하나다.
let latestGex = null;
let gexLastFetchAt = 0;
let latestFlowHeatmap = null;
let flowHeatmapLastFetchAt = 0;
// 현재가 박스가 스스로 움직이는 데 필요한 기하(행 높이·행 키). 렌더가 적고 체결 WS 가 읽는다.
let supplyProfileNow = null;
// 2026-09-11 추세 전환 탐지기. 방향은 예측하지 않는다 -- «전환이 왔다»만 말한다.
// 5분봉 워커라 60초 폴링(극점 탐지기와 같은 주기).
let latestBreakoutDetector = null;
let breakoutDetectorLastFetchAt = 0;
const API_BREAKOUT_DETECTOR_URL = "/api/breakout-detector";
const BREAKOUT_DETECTOR_POLL_MS = 60000;
// Long/short liquidation volume gauge (recreated 2026-08-27, see renderLiquidationVolumeGauge()) --
// backend (scripts/live_liquidation_5m_signal_20260825.py) never stopped running, only this
// frontend consumer had been removed.
let latestLiquidation5m = null;
let latestLiquidation5mHist = [];
let liquidation5mLastFetchAt = 0;
// 베이시스 청산압박 model indicator (replaces 독성/toxicity, 2026-08-27) -- own fetch cycle, same
// dashboard-side-computed category as latestVRebound above (scripts/live_spot_perp_basis_signal_
// 20260827.py). RISK GAUGE, not a price-direction claim -- see MODEL_INDICATOR_DETAIL.liq_pressure.
let latestVolLevel = null;
let volLevelLastFetchAt = 0;
// Sudden-liquidation alert (2026-08-27) -- backed by tail_risk_interceptor.py's event-triggered
// liq_burst_state.json (own file, own writer, updated the instant a new @forceOrder event lands),
// not the once-a-minute tail_risk.duckdb path the gauge above reads. Own short poll interval since
// the source can change sub-second during a real cascade -- see API_LIQ_BURST_STATE_URL below.
let latestLiqBurstState = null;
let liqBurstStateLastFetchAt = 0;
// Liquidation map (Snapshot tab, 2026-08-24) -- estimated support/resistance, own fetch/render
// cycle same as latestVRebound above (computed dashboard-side, not part of trading_bot.py state).
// lastSnapshotHistoryFetchAt tracks the candle history this panel's chart needs (activeSnapshotAsset,
// see below), independently of activeChartAsset (the Live tab's own, separate coin selector).
let latestLiquidationMap = null;
let latestRegimeWide24 = null;
let latestRegimeBtc = null;
let latestRegimeXrp = null;
// 코인별 실시간 지표(수급흐름/리테일수급/청산캐스케이드). ETH는 봇 state를 그대로 쓰고,
// 다른 코인은 그 코인 자신의 duckdb에서 온다 -- 자산별 슬롯(공유 금지, 2026-08-31 교훈).
let latestCoinIndicators = {};
let liquidationMapLastFetchAt = 0;
// Snapshot tab's own coin selector (2026-08-31, BTC then XRP then SOL then HYPE added) -- deliberately separate from
// activeChartAsset (the Live tab's chart asset, which the Snapshot tab has never followed -- see
// the comment on lastSnapshotHistoryFetchAt above). Only backs the 4 signals server.py now accepts
// an ?asset= for (basis liquidation / liquidation direction / liquidation 5m / liquidation map,
// plus that map's own candle chart) -- 증거신호/레짐/특화감지기/수급흐름/리테일수급/청산캐스케이드
// stay ETH-only regardless of this (see docs/eth_dashboard_multicoin_expansion_design_20260831.md
// section 6.4 for why: those are trained-model or trading_bot.py-sourced, not a symbol swap away).
let activeSnapshotAsset = "eth";
const SNAPSHOT_ASSET_KEYS = ["eth", "btc", "sol", "xrp", "hype"];
let regimeWide24LastFetchAt = 0;
let regimeBtcLastFetchAt = 0;
let regimeXrpLastFetchAt = 0;
let coinIndicatorsLastFetchAt = 0;
let macroCalendarLastFetchAt = 0;
// 2026-09-10 거래소 실계좌. ops 탭 패널과 스냅샷 탭 요약이 같은 payload 를 쓰므로 한 곳에 담는다.
// 서버가 이미 30초 캐시(BINANCE_ACCOUNT_CACHE_SECONDS)라 클라 주기도 같게 맞춘다.
let latestBinanceAccount = null;
// 🔴마지막으로 **성공한** 계좌. latestBinanceAccount 는 실패 시 null 이 되는데, 그걸로
// 청산 버튼을 숨기면 «조회 실패»와 «포지션 없음»이 구분되지 않는다 -- 정작 닫아야 할 때
// 버튼이 사라진다(2026-09-13 사용자 신고). 버튼은 이 값으로 판단하고, 낡았으면 표시만 한다.
let lastGoodAccount = null;
let lastGoodAccountAt = 0;
let binanceAccountLastFetchAt = 0;
const BINANCE_ACCOUNT_POLL_MS = 30000;
let sessionAlertsLastFetchAt = 0;
let lastSnapshotHistoryFetchAt = 0;
let lastSnapshotChartRenderAt = 0;
let lastModelIndicatorHtmlByTarget = {};
let activePageTab = "snapshot"; // "ops" | "snapshot" (라이브 탭 제거, 2026-08-31) -- must match index.html's default active tab (data-page-tab="snapshot" carries the initial "active" class)
// 2026-09-21 🔴**불리언 래치를 시각으로 바꿨다.** `isScrolling = true` 를 걸고 150ms
// setTimeout 으로만 풀면, 배경 탭·가려진 창에서 그 타이머가 안 와 **영구히 true** 로 남는다.
// 그러면 maybeRenderSnapshotChartNow()/render() 가 계속 조기 반환해 차트가 그 시점에 굳는데
// updateLivePriceFast() 에는 이 가드가 없어 **현재가 라벨만 움직인다** -- 사용자가 신고한
// 정확히 그 그림이다(풋프린트가 04:40 에 멈췄는데 04:52 까지 가격만 갱신).
// 시각 비교는 타이머 도착에 의존하지 않는다. 타이머는 «따라잡기 tick» 용으로만 남긴다.
const SCROLL_IDLE_MS = 150;
let lastScrollAt = 0;
const isScrolling = () => Date.now() - lastScrollAt < SCROLL_IDLE_MS;
let scrollIdleTimer = 0;
let dashboardEvents = null;
const OPS_POLL_MS = 30000;
const LIQUIDATION_5M_POLL_MS = 60000; // matches server's own 60s cache + the 1-row-per-minute source
const VOL_LEVEL_POLL_MS = 60000;          // 사이징 워커 주기 300초 — 1분 폴링이면 충분하다
// 2026-08-27: liq_burst_state.json is written the instant a new liquidation event arrives (see
// tail_risk_interceptor.py::_write_liq_burst_state()), not on a timer -- polling faster than ~1s
// wouldn't surface anything sooner than the file itself changes, given the remaining hop (this
// fetch) is the last one in the chain.
const LIQ_BURST_STATE_POLL_MS = 1000;
// 2026-09-16 300초 -> 60초. 서버 캐시를 60초로 줄였으므로(입력이 1시간봉이라 그 아래로는
// 의미가 없다) 클라가 5분마다 물으면 **새 시간봉이 최대 5분 늦게** 보인다. 캐시와 같은 주기로.
const LIQUIDATION_MAP_POLL_MS = 60000;
const REGIME_WIDE24_POLL_MS = 300000; // matches server-side cache (REGIME_WIDE24_CACHE_SECONDS)
const MACRO_CALENDAR_POLL_MS = 6 * 3600 * 1000; // matches server-side cache (MACRO_CALENDAR_CACHE_SECONDS)
const SESSION_ALERTS_POLL_MS = 30000; // 2026-08-27: split off evidence-signals' 5min cadence --
                                        // these badges need to feel live to someone watching a
                                        // +-30min window approach in real time, and the endpoint
                                        // is cheap enough (no new external fetch) to poll this often

// --- Chart Global Variables ---
const CHART_CANDLE_MIN = 5;
const CHART_MAX_CANDLES = 100;
// Snapshot tab's own chart only -- narrower than CHART_MAX_CANDLES (Live tab, unaffected) so every
// visible column has a real compute_heatmap_history() snapshot behind it (2026-08-25 user request,
// "차트를 4시간만 보여주는건 어떨까", then same day "4시간은 너무 작다" -> 6h -- see
// live_liquidation_map_20260824.py::compute_heatmap_history and its HEATMAP_HISTORY_DISPLAY_HOURS,
// which this must match).
const SNAPSHOT_CHART_MAX_CANDLES = 72; // 6h at 5-min candles
const MOBILE_CHART_DEFAULT_CANDLES = 34;
const MOBILE_CHART_MIN_CANDLES = 12;
const MOBILE_CHART_MAX_CANDLES = 72;
const mobileChartView = {
  start: null,
  size: MOBILE_CHART_DEFAULT_CANDLES,
  followLatest: true,
};

// 2026-09-03: 코인 전환 중 스켈레톤(로딩 애니메이션).
// 문제: setActiveSnapshotAsset()가 6개 fetch를 Promise.all로 한꺼번에 기다린 뒤에야 render()를
// 불렀기 때문에, 전환 직후~가장 느린 fetch가 끝날 때까지 **이전 코인의 숫자가 새 코인 탭 아래에
// 그대로 보이다가** 갑자기 통째로 바뀌었다(사용자 신고). 이전 코인 값을 새 탭 라벨 밑에 띄우는
// 건 단순히 보기 나쁜 정도가 아니라 오독 위험이다.
// 해결: (1) 전환 즉시(await 이전에) asset-scoped 영역 전부를 스켈레톤으로 덮고, (2) 각 영역은
// **자기 데이터가 도착하는 순간** 개별적으로 스켈레톤을 벗는다 -- 가장 느린 fetch가 나머지를
// 붙잡지 않는다. 영역 지정은 index.html의 data-asset-scope 속성이 단일 소스다.
const ASSET_SCOPES = ["indicators", "liqmap"];
// 전환 세대 번호. 스켈레톤을 벗기 전에 이 값을 대조해서, 느리게 도착한 **이전** 전환의 응답이
// 새 전환의 스켈레톤을 걷어내는 경쟁 상태를 막는다(코인을 빠르게 연타할 때 실제로 발생).
let assetSwitchGeneration = 0;
let pendingAssetScopes = new Set();

function assetScopeNodes(scope) {
  return document.querySelectorAll(`[data-asset-scope="${scope}"]`);
}

function beginAssetScopeLoading() {
  assetSwitchGeneration += 1;
  pendingAssetScopes = new Set(ASSET_SCOPES);
  ASSET_SCOPES.forEach((scope) => {
    assetScopeNodes(scope).forEach((node) => node.classList.add("asset-loading"));
  });
  return assetSwitchGeneration;
}

function endAssetScopeLoading(scope, generation) {
  if (generation !== assetSwitchGeneration) return; // 이미 다음 코인으로 또 전환됨 -- 무시
  if (!pendingAssetScopes.delete(scope)) return;
  assetScopeNodes(scope).forEach((node) => node.classList.remove("asset-loading"));
}

// renderSnapshotChart()는 한 tick 안에서 여러 fetch 콜백(청산맵·레짐·캔들이력·증거신호)이
// 각자 부르기 때문에 같은 SVG를 한 프레임에 여러 번 그리곤 했다. rAF로 합쳐 프레임당 1회만
// 실제로 그린다(그리는 내용은 동일 -- 항상 최신 캐시에서 다시 읽으므로 마지막 1회면 충분).
let snapshotChartRafId = 0;
let snapshotChartRafAt = 0;
// rAF 는 창이 가려지거나 탭이 얼면 «호출되지 않은 채» 남는다. 그 id 를 잠금으로 쓰면
// 위 isScrolling 과 같은 방식으로 영구 조기반환이 된다. 예약이 이보다 오래 묵으면 버린다.
const RAF_STALL_MS = 2000;
function scheduleSnapshotChartRender() {
  const now = Date.now();
  if (snapshotChartRafId) {
    if (now - snapshotChartRafAt < RAF_STALL_MS) return;   // 정상 대기
    try { cancelAnimationFrame(snapshotChartRafId); } catch (e) { /* 이미 소멸 */ }
  }
  snapshotChartRafAt = now;
  snapshotChartRafId = requestAnimationFrame(() => {
    snapshotChartRafId = 0;
    renderSnapshotChart();
  });
}

// 서버가 SSE 로 알려주는 «켜진 코인» 목록. 2026-09-16 사용자 요청으로 기본은 ETH 하나다
// (서버의 DASHBOARD_ASSETS 가 원본이고 여기는 사본이 아니다 -- 받아서 쓴다).
// null 이면 아직 못 받았거나 구버전 서버다 -- 그 경우 전부 보인다(있던 걸 없애지 않는다).
let enabledAssets = null;

function renderSnapshotAssetTabs() {
  document.querySelectorAll("#snapshotAssetTabs .asset-tab").forEach((btn) => {
    const on = !enabledAssets || enabledAssets.includes(btn.dataset.asset);
    // 🔴목록을 받기 전(enabledAssets === null)에는 **되살리지 않는다**.
    //   서버가 index 를 내보낼 때 꺼진 코인에 hidden 을 붙여 주는데(dashboard_index::
    //   _hide_off_assets), 여기서 무조건 `btn.hidden = !on` 을 쓰면 app.js 가 뜨자마자
    //   그걸 전부 되돌려 «5개 번쩍 → ETH 만» 이 된다(실측: 350ms 에 5개, 800ms 에 ETH).
    //   fail-open 은 그대로다 -- 구버전 서버는 hidden 을 안 붙이므로 계속 전부 보인다.
    if (enabledAssets) btn.hidden = !on;
    btn.classList.toggle("active", on && btn.dataset.asset === activeSnapshotAsset);
  });
}

async function setActiveSnapshotAsset(asset) {
  if (!SNAPSHOT_ASSET_KEYS.includes(asset) || asset === activeSnapshotAsset) return;
  activeSnapshotAsset = asset;
  renderSnapshotAssetTabs();
  // 계좌 payload 는 전 코인의 포지션을 담고 있어 재요청 없이 다시 그리기만 하면 된다.
  renderSnapshotAccount();
  // Clear the 4 wired signals' cached readings + their poll-interval gates immediately -- without
  // this, the panels would keep showing the PREVIOUS coin's numbers (mislabeled as the new one)
  // until each signal's own poll interval next elapses (up to 5min for the slowest).
  latestVolLevel = null; volLevelLastFetchAt = 0;
  latestLiquidation5m = null;
  latestLiquidation5mHist = [];
  latestOi5m = null; oi5mLastFetchAt = 0;
  latestLiquidationMap = null;
  liquidation5mLastFetchAt = 0;
  liquidationMapLastFetchAt = 0;
  lastSnapshotHistoryFetchAt = 0;
  // 2026-09-03: 수급흐름/리테일수급/청산캐스케이드(비ETH 코인의 coin-indicators)도 같은 이유로
  // 게이트를 연다. 이 게이트는 자산별이 아니라 **전역 하나**라, 열어주지 않으면 20초(MODEL_
  // INDICATOR_POLL_MS) 안에 코인을 두 번 바꿨을 때 두 번째 코인은 자기 값을 못 받아온 채
  // 스켈레톤만 벗겨진다.
  coinIndicatorsLastFetchAt = 0;
  // 레짐 리본도 코인별 모델이다. tick()이 이제 **활성 코인 것만** 가져오므로(refreshActiveRegime),
  // 전환 시엔 새 코인의 게이트를 열어 즉시 한 번 받아온다.
  regimeWide24LastFetchAt = 0;
  regimeBtcLastFetchAt = 0;
  regimeXrpLastFetchAt = 0;

  // ⭐await보다 먼저 -- 스켈레톤은 첫 fetch가 나가기 전에 이미 화면에 올라가 있어야 한다.
  const generation = beginAssetScopeLoading();

  // 영역별로 따로 해제한다. 예전처럼 Promise.all 하나로 묶으면 가장 느린 하나(보통 증거신호의
  // TabPFN 경로)가 나머지 전부를 인질로 잡는다.
  // ⚠️.catch()가 .then() **앞에** 있는 게 핵심이다 -- fetch가 실패해도 스켈레톤은 반드시 벗겨야
  // 한다. 실패 시 그대로 두면 네트워크가 한 번 끊긴 것만으로 그 패널이 영원히 로딩 애니메이션에
  // 갇힌다(각 refresh 함수는 자체 try/catch로 이미 에러를 삼키지만, 여기서 그걸 가정하지 않는다).
  const settleScope = (scope, work) =>
    Promise.all(work)
      .catch((error) => console.error(`코인 전환 중 ${scope} 갱신 실패:`, error))
      .then(() => {
        if (latestMainState) render(latestMainState, latestCompactState);
        endAssetScopeLoading(scope, generation);
      });

  await Promise.all([
    settleScope("indicators", [
      refreshVolLevel(),
      refreshCoinIndicators(),
    ]),
    settleScope("liqmap", [
      refreshLiquidation5mSignal(),
      refreshLiquidationMap(),
      maybeFetchSnapshotChartHistory(),
      refreshActiveRegime(),
    ]),
  ]);
  if (latestMainState) render(latestMainState, latestCompactState);
}

/* 테마 토글 (2026-09-16). 다크 = 현행 화면, 라이트 = 애플 «Liquid Glass».
   첫 페인트는 index.html 의 인라인 스크립트가 이미 입혔다 -- 여기서는 «바꾸기」만 한다.
   ⭐색을 하드코딩한 사본을 만들지 않는다: 차트는 SVG 속성에 `var(--good)` 를 그대로 넣고
   브라우저가 실시간으로 푼다. 다만 `cssVar()` 게터로 읽는 곳(레짐 색)은 다시 그려야 반영된다. */
const THEME_KEY = "dashTheme";
function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
}
function applyTheme(theme) {
  const light = theme === "light";
  if (light) document.documentElement.setAttribute("data-theme", "light");
  else document.documentElement.removeAttribute("data-theme");
  // 모바일 브라우저 크롬(주소창) 색까지 같이 간다 -- 안 맞추면 유리 화면 위에 검은 띠가 남는다.
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", light ? "#eef0f5" : "#0b0d13");
  const btn = document.getElementById("themeToggle");
  const label = document.getElementById("themeToggleLabel");
  if (btn) {
    btn.setAttribute("aria-pressed", light ? "true" : "false");
    btn.dataset.state = light ? "on" : "off";
  }
  if (label) label.textContent = light ? "글래스" : "다크";
}
function setupThemeToggle() {
  const btn = document.getElementById("themeToggle");
  applyTheme(currentTheme());                 // 버튼 라벨을 저장된 상태에 맞춘다
  if (!btn) return;
  btn.addEventListener("click", () => {
    const next = currentTheme() === "light" ? "dark" : "light";
    applyTheme(next);
    try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* 저장 못 해도 동작은 한다 */ }
    // 게터로 토큰을 읽는 곳(레짐 색 등)은 다시 그려야 새 팔레트가 들어간다.
    renderSnapshotChart();
    if (latestMainState) render(latestMainState, latestCompactState);
  });
}

// 2026-09-21 청산맵 모드 제거(사용자 결정). 풋프린트가 유일한 차트다 --
// 청산 밀도 히트맵·S/R 레벨 목록·최근접 레벨 선은 전부 풋프린트에서도 그대로 나온다.
// ⚠️«채워진 캔들» 폴백은 renderCandleSvg 에 **남아 있다**: 풋프린트는 ETH 전용이고
//   테이프가 웜업 중이면 null 이라, 그때 캔들을 그릴 것이 필요하다.
function setupChartModeTabs() {
  setupChartWindowTabs();
}


function setupChartWindowTabs() {
  document.querySelectorAll("#chartWindowTabs .asset-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      const bars = Number(btn.dataset.bars);
      if (!CHART_WINDOW_BARS.includes(bars) || bars === chartWindowBars) return;
      chartWindowBars = bars;
      try { localStorage.setItem("chartWindowBars", String(bars)); } catch (e) { /* 저장 못 해도 동작은 한다 */ }
      renderChartWindowTabs();
      // 폴링 간격(0.4초/5초)을 기다리지 않고 바로 받는다 -- 누른 티가 나야 한다.
      footprintLastFetchAt = 0;
      supplyProfileLastFetchAt = 0;
      flowHeatmapLastFetchAt = 0;
      refreshFootprint();
      refreshSupplyProfile();
      refreshFlowHeatmap();
      refreshGex();
    });
  });
  renderChartWindowTabs();
}

function renderChartWindowTabs() {
  document.querySelectorAll("#chartWindowTabs .asset-tab").forEach((btn) => {
    btn.classList.toggle("active", Number(btn.dataset.bars) === chartWindowBars);
  });
}

function setupSnapshotAssetTabs() {
  document.querySelectorAll("#snapshotAssetTabs .asset-tab").forEach((btn) => {
    btn.addEventListener("click", () => setActiveSnapshotAsset(btn.dataset.asset));
  });
  renderSnapshotAssetTabs();
}

function fmtNum(v, d = 2) {
  return Number(v || 0).toFixed(d);
}


function fmtUsdCompact(v) {
  const n = Number(v || 0);
  if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `$${(n / 1e3).toFixed(1)}k`;
  return `$${n.toFixed(0)}`;
}

function clamp01(v) {
  return Math.max(0, Math.min(1, Number(v || 0)));
}

function clampNum(v, min, max) {
  return Math.max(min, Math.min(max, Number(v || 0)));
}

function fmtTs(v) {
  if (!v) return "-";
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return String(v);
  const mo = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${mo}-${dd} ${hh}:${mm}:${ss}`;
}

function fmtShortTs(v) {
  if (!v) return "-";
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) {
    const s = String(v);
    return s.length > 16 ? s.slice(0, 16) : s;
  }
  const mo = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${mo}-${dd} ${hh}:${mm}`;
}

// Model-indicator strip-time label only (2026-08-25 user request: "시-분-초만 표시") -- unlike
// fmtShortTs, no date, seconds included since these bars can be sub-minute apart (client-tracked
// indicators record the real push time, not a clock-aligned bar).
function fmtTimeOnly(v) {
  if (!v) return "-";
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return "-";
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

// Evidence-signal strip axis only (2026-08-31 user request: "시와 분만 표시") -- unlike fmtTimeOnly,
// no seconds; unlike fmtShortTs, no date either. Ticks stay short enough to fit 5 across a narrow
// strip without wrapping.
function fmtHourMinute(v) {
  if (!v) return "-";
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return "-";
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${hh}:${mm}`;
}

function fmtNowClock() {
  const d = new Date();
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function buildSessionHtml(sess) {
  const sAsiaOn = Number(sess.session_asia || 0) >= 0.5;
  const sEurOn = Number(sess.session_europe || 0) >= 0.5;
  const sUsOn = Number(sess.session_us || 0) >= 0.5;
  return [
    `<span class="session-item ${sAsiaOn ? "on" : "off"}"><span class="session-led ${sAsiaOn ? "on" : "off"}"></span>아시아</span>`,
    `<span class="session-sep">|</span>`,
    `<span class="session-item ${sEurOn ? "on" : "off"}"><span class="session-led ${sEurOn ? "on" : "off"}"></span>유럽</span>`,
    `<span class="session-sep">|</span>`,
    `<span class="session-item ${sUsOn ? "on" : "off"}"><span class="session-led ${sUsOn ? "on" : "off"}"></span>미국</span>`,
  ].join("");
}



// nif_retail: same _compute_nif_and_taker() split as nif_whale above, retail (small-size) leg
// instead of whale leg. Added 2026-08-25 after a same-day IC screen found real (non-noise) short-
// horizon direction information here -- see MODEL_INDICATOR_DETAIL.retail_flow for the numbers.

// 2026-09-11 청산 규모 칩 제거(중복 지표). 대체 칩 없이 자리를 비운다 -- 헬퍼도 함께 제거됨.


// 2026-08-27: replaces toxRead/toxHint (독성/toxicity chip removed -- shadow_toxicity_score was
// independently confirmed uninformative on both direction and volatility-framing axes, see
// eth_model_indicator_volatility_framing_screen_20260825 memory). sig here is the raw
// latestBasisLiquidation payload (server-computed, not part of classifyIndicators' micro/tail
// inputs -- same "own fetch cycle" category as latestVRebound, see that variable's own comment).

// Sudden-liquidation alert banner (2026-08-27) -- reads liq_burst_state.json (event-triggered, see
// tail_risk_interceptor.py::_write_liq_burst_state()), a faster/more prominent sibling to the
// liq_cascade model-indicator tile below (which reads the same hawkes/z-score concept but via the
// 10s-cadence dashboard_state.json path). Shown only while hawkes_active -- an alert that's always
// visible isn't an alert. (선례로 들던 실행경보 배너는 2026-09-20 은퇴했다 -- 규칙만 남는다.)
// Liquidation long/short volume gauge -- recreated 2026-08-27 at user request. This is the bar
// chart half of the original renderLiquidationCascadeGauge() (2026-08-25): proportional split bar,
// long=red(--bad)/short=green(--good), with real $ labels alongside so a "$5 vs $2" split doesn't
// read as visually skewed as a "$500 vs $2" one would (2026-08-25 design note, preserved). The
// magnet (price/direction/strength) and energy/recommendation sub-parts that used to live in the
// same gauge were NOT recreated -- user confirmed the magnet was redundant with the chart line
// (liquidationMagnetLevel(), itself removed 2026-08-31 per user request) and never asked for
// energy/recommendation back. Always renders a row (never disappears) per 2026-08-27 request, with
// a quiet state for warming-up/no-liquidation.
// Data: /api/liquidation-5m-signal (scripts/live_liquidation_5m_signal_20260825.py, BAR_MINUTES=30
// as of this same request -- server.py imports that module, so a dashboard-server restart is
// needed for the window change, NOT trading_bot.py; this data has nothing to do with that process).
//
// 2026-08-27 follow-up: user asked for this to reflect a detected cascade immediately rather than
// waiting for the current (still-forming) minute to close and land in tail_risk_1m -- but the $
// totals below CANNOT safely fold in liq_burst_state.json's long_usd_1m/short_usd_1m to get there.
// Those share a field name with tail_risk_1m's columns but not a definition: tail_risk_1m stores
// one DISCRETE non-overlapping bucket per completed minute (see this function's data-source comment
// above, and live_liquidation_5m_signal_20260825.py's own docstring on why that matters), while
// liq_burst_state.json's version is a continuously-SLIDING trailing-60s value that overlaps
// whatever's already in the most recent completed bucket -- adding it on top would double-count.
// Instead: a live "감지중" cue sourced from latestLiqBurstState (already polled every ~1s for the
// alert banner above, no new fetch here) that flags "something's happening right now" without
// touching the $ math -- correct-by-construction rather than an approximate merge.
// 2026-09-06 (사용자 요청): 새로고침 직후엔 "집계 중..." 텍스트만 뜨고 게이지 자체가 없다가, 데이터가
// 오면 완성된 막대가 **툭 나타났다**. 대신 **처음부터 롱 0 · 숏 0 짜리 빈 게이지를 그려두고**, 값이
// 도착하면 너비만 바꿔 CSS transition 으로 움직이게 한다. 그래서 innerHTML 을 매번 갈아끼우지 않고
// (그러면 새 엘리먼트가 최종 너비로 태어나 애니메이션이 없다) **한 번 만든 뒤 제자리 갱신**한다.
function liquidationVolumeGaugeSkeletonHtml() {
  return `<div class="liq-vol-gauge">
      <span class="liq-vol-gauge-tag">청산 규모 <span class="liq-vol-gauge-window">(최근 30분 누적)</span><span class="liq-vol-gauge-live-slot"></span></span>
      <div class="liq-vol-gauge-track">
        <div class="liq-vol-gauge-fill long" style="width:0%"></div>
        <div class="liq-vol-gauge-fill short" style="width:0%"></div>
      </div>
      <div class="liq-vol-gauge-labels">
        <span class="liq-vol-gauge-label long">롱 $0</span>
        <span class="liq-vol-gauge-label short">숏 $0</span>
      </div>
    </div>`;
}

function liquidationVolumeLiveCueHtml() {
  const burst = latestLiqBurstState;
  if (!(burst && burst.available && burst.hawkes_active)) return "";
  // 2026-08-27: 독립 배너였던 것을 이 한 줄로 접었다(renderLiqBurstAlert 제거). 측면은 crisis_type
  // 라벨이 아니라 양쪽 실측 최댓값으로 고른다 -- 그 라벨은 낡을 수 있다.
  const bLong = Number(burst.long_usd_1m || 0);
  const bShort = Number(burst.short_usd_1m || 0);
  const bUsd = Math.max(bLong, bShort);
  const bSide = bShort > bLong ? "숏청산" : "롱청산";
  const bPct = Math.round(clamp01(Number(burst.hawkes_decay_level) || 0) * 100);
  const detail = bUsd > 0 ? ` · ${bSide} ${fmtUsdCompact(bUsd)} · 에너지${bPct}%` : ` · 에너지${bPct}%`;
  return `<span class="liq-vol-gauge-live"><span class="liq-vol-gauge-live-dot" aria-hidden="true"></span>지금 감지중${detail}</span>`;
}

function renderLiquidationVolumeGauge() {
  const host = el("liqVolumeGauge");
  if (!host) return;
  const fresh = !host.querySelector(".liq-vol-gauge-track");
  if (fresh) host.innerHTML = liquidationVolumeGaugeSkeletonHtml();

  const liq5m = latestLiquidation5m;
  const warmed = !!(liq5m && liq5m.warmed_up);
  const longUsd = warmed ? Number(liq5m.long_usd_5m || 0) : 0;
  const shortUsd = warmed ? Number(liq5m.short_usd_5m || 0) : 0;
  const total = longUsd + shortUsd;

  const win = host.querySelector(".liq-vol-gauge-window");
  if (win) win.textContent = warmed ? "(최근 30분 누적)" : "(최근 30분 누적) · 웜업";
  const cue = host.querySelector(".liq-vol-gauge-live-slot");
  if (cue) {
    const html = liquidationVolumeLiveCueHtml();
    if (cue.innerHTML !== html) cue.innerHTML = html;
  }
  const longEl = host.querySelector(".liq-vol-gauge-label.long");
  const shortEl = host.querySelector(".liq-vol-gauge-label.short");
  if (longEl) longEl.textContent = `롱 ${fmtUsdCompact(longUsd)}`;
  if (shortEl) shortEl.textContent = `숏 ${fmtUsdCompact(shortUsd)}`;

  const apply = () => {
    const fillLong = host.querySelector(".liq-vol-gauge-fill.long");
    const fillShort = host.querySelector(".liq-vol-gauge-fill.short");
    if (fillLong) fillLong.style.width = total > 0 ? `${(longUsd / total) * 100}%` : "0%";
    if (fillShort) fillShort.style.width = total > 0 ? `${(shortUsd / total) * 100}%` : "0%";
  };
  // 스켈레톤을 방금 만든 프레임에서 바로 최종 너비를 주면 브라우저가 중간 상태를 못 보고 즉시 그린다
  // (transition 없음). 다음 프레임에 적용해 0% -> 실제값 전이가 실제로 보이게 한다.
  if (fresh && typeof requestAnimationFrame === "function") requestAnimationFrame(apply);
  else apply();
}

// 7th model-internal indicator -- 2026-08-25, user asked for a dedicated "청산 캐스케이드" tile
// rather than folding this into 꼬리 리스크's text. Distinct focus from that indicator: 꼬리
// 리스크's aftershock_prob is a forward-looking blended probability of MORE shock still to come;
// this is the raw "is a cascade actually happening right now" state straight from
// tail_risk_interceptor.py's own 3-stage design (detector/discriminator/decay-timer) -- which side,
// and how much of the initial energy is left. z>=2.0 threshold for the 주의 tier reuses the exact
// value tail_risk_interceptor.py's own status_line() already uses for "급증⚠️", not a new number.


// 안정=문제없음(녹색), 주의=경계(호박색), 위험=경계강함(적색) -- liq_cascade(리스크게이지)
// 지표 전용. 방향성 매매신호(롱 진입/숏 진입)는 whale/retail_flow가 directionalCaution()로
// 별도 처리하므로 여기서 다루지 않는다.
function signalTone(signal) {
  const s = String(signal || "");
  if (s === "위험") return "bad";
  if (s.includes("주의")) return "warn";
  if (s === "안정") return "good";
  return "neutral";
}

// Single source of truth for the 3 model-internal indicators' tone/read-text classification --
// called both on the live state (render(), every tick) and on server-provided history samples
// (seedModelIndicatorHistory(), once at page load) so there is exactly one copy of these
// thresholds, not a live copy and a history copy that could quietly drift apart.
// 2026-08-30 (user request): risk(꼬리 리스크)/whale_intent(고래 포지션) removed from this
// dashboard -- risk's aftershock_prob tested NULL at all 5 evaluated horizons (5m/15m/1h direction,
// 1h/4h volatility, see eth_liquidation_shadow_aftershock_prob_signal_check_rejected_20260827),
// and whale_intent was already flagged in this file as a non-independent transform of
// whale+OI-delta (formula comment above EVIDENCE_SIGNAL_KO) plus itself failed direction-IC at
// all 4 tested horizons (worst of the 3 flow signals, one near-pass cell sign-flipped VAL vs
// TRAIN -- eth_whale_position_vs_retail_flow_direction_ic_20260825). liq_cascade's own underlying
// hawkes state stays wired up regardless (still gates the separate liq-burst-state alert banner)
// -- only removing risk/whale_intent's own chip surfaces here.


function niceStep(span, targetTicks = 4) {
  const rough = span / Math.max(targetTicks, 1);
  const mag = Math.pow(10, Math.floor(Math.log10(rough || 1)));
  const norm = rough / mag;
  const stepNorm = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
  return stepNorm * mag;
}

function axisTicks(min, max, targetTicks = 4) {
  const step = niceStep(max - min, targetTicks);
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = start; v <= end + step * 0.5; v += step) ticks.push(v);
  return ticks;
}

async function fetchBinanceHistory(asset) {
  try {
    const res = await fetch(`/api/market-history?asset=${asset}`, { cache: "no-cache" });
    if (!res.ok) return;
    const payload = await res.json();
    candleHistoryByAsset[asset] = Array.isArray(payload?.candles) ? payload.candles : [];
  } catch (e) { console.error("History Error:", e); }
}

async function maybeFetchSnapshotChartHistory() {
  const now = Date.now();
  const cached = candleHistoryByAsset[activeSnapshotAsset] || [];
  if (cached.length && now - lastSnapshotHistoryFetchAt < CANDLE_HISTORY_POLL_MS) return;
  lastSnapshotHistoryFetchAt = now;
  await fetchBinanceHistory(activeSnapshotAsset);
  scheduleSnapshotChartRender();
}

// 스로틀된 차트 갱신. render() 경로와 SSE 시세 경로가 **같은 게이트**를 공유한다.
// 스냅샷 탭이 아닐 때는 그리지 않는다(숨은 패널을 그리는 건 순수 낭비 -- 2026-08-25 규약).
// 풋프린트 모드는 셀이 체결마다 자란다 -- 더 짧은 간격이 실제로 새 그림을 만든다.
// 청산맵 모드는 데이터가 5분·1시간 단위라 짧게 해봐야 같은 그림을 다시 그릴 뿐이다.
// (현재가 선은 전체 렌더와 무관하게 updateLivePriceFast 가 80ms 로 따로 움직인다)
const FOOTPRINT_RENDER_MIN_INTERVAL_MS = 400;
function chartRenderGateMs() {
  return FOOTPRINT_RENDER_MIN_INTERVAL_MS;
}

function maybeRenderSnapshotChartNow() {
  if (activePageTab !== "snapshot" || isScrolling()) return;
  const now = Date.now();
  if (now - lastSnapshotChartRenderAt < chartRenderGateMs()) return;
  lastSnapshotChartRenderAt = now;
  updateSnapshotCandleLive();
  renderSnapshotChart();
  // 지지/저항 목록도 같이 그린다. 2026-09-16 실측: 30초 동안 차트는 67회 다시 그려지는데 이
  // 패널은 **0회**였다 -- render() 경로(상태 변경 푸시)에만 걸려 있어서 몇 분씩 묵었다.
  // 이 패널의 «지금 값»은 거리(%)와 이미 뚫린 레벨 걸러내기(liveRedistanced)이고, 둘 다
  // 현재가로 계산한다. 레벨 **가격** 자체는 시간봉이 바뀔 때만 움직인다(그건 서버 몫).
  renderLiquidationMapPanel();
}

function applyDashboardEvent(payload) {
  if (Array.isArray(payload?.assets) && payload.assets.length) {
    const next = payload.assets.filter((a) => SNAPSHOT_ASSET_KEYS.includes(a));
    if (next.length && String(next) !== String(enabledAssets || [])) {
      enabledAssets = next;
      renderSnapshotAssetTabs();
      // 보고 있던 코인이 꺼졌으면 켜진 첫 코인으로 옮긴다 -- 안 그러면 영원히 빈 패널을 본다.
      if (!next.includes(activeSnapshotAsset)) setActiveSnapshotAsset(next[0]);
    }
  }
  const tickers = payload?.tickers || {};
  Object.entries(tickers).forEach(([asset, ticker]) => {
    const price = Number(ticker?.price || 0);
    if (!(price > 0) || !ASSET_CONFIG[asset]) return;
    latestLivePriceByAsset[asset] = price;
    latestLivePriceTsByAsset[asset] = String(ticker.ts || "");
  });
  // 바이낸스 직결 WS 가 막힌 환경에서는 이 푸시가 유일한 시세다 -- 없으면 현재가 박스가
  // 프로파일 폴링(5초)에 묶여 멈춘 것처럼 보인다.
  updateSupplyProfileNow(Number(latestLivePriceByAsset[activeSnapshotAsset] || 0));
  if (payload?.state?.state) {
    latestMainState = payload.state.state;
    latestCompactState = payload.state.compactState || null;
  }
  // 🔴차트는 **상태(state)가 바뀐 푸시**에서만 다시 그려졌다. 시세만 오는 푸시(대부분)에서는
  //   아무것도 안 그려서, 현재가 선과 진행 중인 봉이 상태 변경 주기(실측 5초)에 묶여 멈춰
  //   보였다(2026-09-16 사용자 "래깅이 있어"). 시세가 곧 그 선의 값이므로 여기서도 그린다.
  //   스로틀은 같은 상수를 쓰므로 아래 render() 경로와 겹쳐도 두 번 그리지 않는다.
  maybeRenderSnapshotChartNow();
  if (!latestMainState || isScrolling() || !payload?.state?.state) return;
  render(latestMainState, latestCompactState, { stateChanged: true });
}

function connectDashboardEvents() {
  if (dashboardEvents) return;
  const events = new EventSource(API_EVENTS_URL);
  dashboardEvents = events;
  events.onmessage = (event) => {
    try {
      applyDashboardEvent(JSON.parse(event.data));
    } catch (error) {
      console.error("Dashboard event parse error:", error);
    }
  };
  events.onerror = () => {
    if (!document.hidden) console.warn("Dashboard event connection interrupted; reconnecting.");
  };
}

function disconnectDashboardEvents() {
  if (!dashboardEvents) return;
  dashboardEvents.close();
  dashboardEvents = null;
}

function opsTone(status) {
  const value = String(status || "").toUpperCase();
  if (value === "OK" || value === "RUNNING") return "good";
  if (value === "WARN") return "warn";
  if (value === "CRITICAL" || value === "BLOCKED" || value === "STOPPED") return "bad";
  return "neutral";
}

function opsLabel(value) {
  return ({ trading_bot: "트레이딩 봇", ops_watchdog: "Ops Watchdog", trading_bot_process: "트레이딩 봇 프로세스", decision_snapshot: "의사결정 스냅샷", trading_bot_heartbeat: "봇 heartbeat", data_pipeline: "데이터 파이프라인", pipeline_contract: "파이프라인 계약", market_data_sources: "시장 데이터 소스", dashboard_state: "대시보드 상태", execution_contract: "실행 안전 계약", runtime_resources: "시스템 자원", watchdog_storage: "watchdog 저장소" })[value] || String(value || "알 수 없음");
}

// 칩 설명문은 처음부터 **강조** 문법으로 쓰여 있었는데 그대로 찍혀 별표가 보였다(사용자 지적).
// 🔴이스케이프를 **먼저** 하고 그 결과에서만 별표를 태그로 바꾼다 -- 순서를 뒤집으면 본문의
//   < & 가 태그가 되어 그대로 주입된다. 별표는 이스케이프 대상이 아니라 순서만 지키면 안전하다.
function emphasizeHtml(value) {
  return escapeHtml(value).replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, "<b>$1</b>");
}
// title= 같은 속성에는 태그를 넣을 수 없다 -- 거기서는 별표만 걷어낸다.
function plainEmphasis(value) {
  return String(value ?? "").replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, "$1");
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[ch]);
}

function renderOpsStatus(payload) {
  const heartbeat = payload?.heartbeat || {};
  const health = payload?.health || {};
  const badge = el("opsWatchdogBadge");
  if (badge) {
    badge.className = `ops-badge ${heartbeat.status === "ok" ? "good" : "bad"}`;
    badge.textContent = `WATCHDOG ${heartbeat.status === "ok" ? "RUNNING" : "UNKNOWN"}`;
  }
  setT("opsUpdatedAt", `서버 갱신 ${fmtTs(health.updated_at_kst || payload?.generated_at)}`);
  setT("opsHeartbeatText", `watchdog heartbeat: ${fmtTs(heartbeat.recorded_at_kst)} · ${heartbeat.check_count || 0}개 점검`);
  const checks = Array.isArray(health.checks) ? health.checks : [];
  const badCount = checks.filter((c) => opsTone(c.status) === "bad").length;
  const warnCount = checks.filter((c) => opsTone(c.status) === "warn").length;
  const summaryEl = el("opsHealthSummary");
  if (summaryEl) {
    if (!checks.length) {
      summaryEl.textContent = "점검 항목 없음";
      summaryEl.className = "ops-health-summary neutral";
    } else if (badCount > 0) {
      summaryEl.textContent = `${badCount}개 이상 감지`;
      summaryEl.className = "ops-health-summary bad";
    } else if (warnCount > 0) {
      summaryEl.textContent = `${warnCount}개 주의`;
      summaryEl.className = "ops-health-summary warn";
    } else {
      summaryEl.textContent = `${checks.length}/${checks.length} 정상`;
      summaryEl.className = "ops-health-summary good";
    }
  }
  setH("opsHealthList", checks.map((check) => {
    const details = check?.details || {};
    const age = Number(details.age_minutes);
    const ageText = Number.isFinite(age) ? `${age.toFixed(age < 10 ? 1 : 0)}분 전` : "-";
    const tone = opsTone(check.status);
    return `<article class="ops-health-row ${tone}">
      <span class="ops-health-dot" aria-hidden="true"></span>
      <div class="ops-health-info">
        <strong>${escapeHtml(opsLabel(check.component))}</strong>
        <span>${escapeHtml(check.summary || "-")}</span>
      </div>
      <div class="ops-health-meta">
        <span class="ops-health-status-badge">${escapeHtml(check.status || "UNKNOWN")}</span>
        <small>${ageText}</small>
      </div>
    </article>`;
  }).join(""));
}

// 잔고가 «어느 자산 지갑»의 것인지, 그리고 화면이 안 쓰는 지갑에 돈이 남아 있는지.
// 🔴단일자산 담보 모드에서 최상위 합계는 USDT 전용이라, 서버가 수동 주문 심볼의 담보 자산
//   지갑을 골라 보낸다(server.py quote / live_binance_account pick_balance). 어느 지갑을
//   보고 있는지 화면이 말하지 않으면 «USDC 1,472 가 있는데 0 으로 보인다»가 반대로
//   «0 인데 1,472 로 보인다»가 될 수 있다 -- 숫자 옆에 자산 이름을 붙인다.
function balanceAssetNote() {
  const acct = latestBinanceAccount || {};
  const used = acct.balance_asset || "";
  if (!used) return "";                       // 멀티에셋 모드 = 합계가 곧 계좌 전체
  const idle = (acct.assets || []).filter((a) => a.asset !== used && Number(a.wallet) > 0);
  const tail = idle.length
    ? ` · ${idle.map((a) => `${a.asset} ${fmtUsd(a.wallet)}`).join(" · ")} 은 다른 지갑`
    : "";
  return ` (${used})${tail}`;
}

// 스냅샷 탭이 보고 있는 코인의 포지션 하나. 없으면 null.
// ASSET_CONFIG 는 eth/sol/btc 만 담고 있어 xrp/hype 는 관례대로 <TICKER>USDT 로 만든다.
// 🔴같은 코인이라도 **심볼이 둘**일 수 있다: 화면·봇은 ETHUSDT 인데 수동 주문은 ETHUSDC 로
//   나간다(2026-09-19). 시장 심볼만 보면 수동으로 연 포지션이 이 카드에 영영 안 뜬다
//   (사용자 보고) -- 카드가 안 뜨면 그 아래 청산 버튼도, 차트의 진입선도 같이 사라진다.
// ⭐우선순위는 서버 청산 경로와 **같다**(server.py 의 resolve_exit_position candidates:
//   수동 심볼 먼저, 그다음 시장 심볼). 카드와 청산 버튼이 다른 포지션을 가리키면 헤지 모드에서
//   «청산하려다 신규 진입»이 된다.
// ⭐심볼은 하드코딩하지 않고 서버가 payload 에 실어 보낸 exec_symbol 을 쓴다 -- 환경변수라
//   바뀔 수 있고, 두 벌로 적어 두면 언젠가 한쪽만 고쳐진다.
function snapshotAccountPosition() {
  const positions = latestBinanceAccount?.positions || [];
  const market = ASSET_CONFIG[activeSnapshotAsset]?.symbol || `${activeSnapshotAsset.toUpperCase()}USDT`;
  const exec = latestBinanceAccount?.exec_symbol || "";
  const base = market.replace(/USDT$/, "");
  // 수동 심볼이 **이 코인의 것일 때만** 본다(ETH 탭에서 ETHUSDC, BTC 탭에서는 무시).
  const order = exec && exec !== market && exec.startsWith(base) ? [exec, market] : [market];
  for (const symbol of order) {
    const hit = positions.find((p) => p.symbol === symbol);
    if (hit) return hit;
  }
  return null;
}

// 스냅샷 탭 청산맵 바로 위 요약. ops 탭 패널의 축약판이라 payload 를 공유한다(추가 요청 없음).
// 「내 계좌」 시각 요약 (2026-09-11, 사용자 요청 "텍스트 말고 그래프나 그림으로 보면 바로
// 알 수 있게끔" -> 목업 3안 x 2회 뒤 "C와 E를 잘 섞어서").
//   왼쪽(C안) 큰 숫자 셋(청산까지·미실현·증거금) + 노출 막대 + 포지션 한 줄
//   오른쪽(E안) 왕복 손익 막대 + 누적선 + 최악 한 건 강조
// ⭐이 배치를 고른 이유: 실측에서 **승률 58%(7/12)인데 실현 -$287** 이고 그 손실이 사실상
//   **한 건(-$536, 나머지 11건 합 +$249)** 이었다. 숫자 나열로는 절대 안 보이는 사실이라
//   오른쪽 막대에 그 한 건을 명시적으로 짚어준다.
// 색 규약 §2 준수 -- good/bad/warn/neutral 넷만 쓴다(5번째 색 없음).
// 목업: scripts/plot_account_panel_mockup_ce_20260911.py · docs/charts/account_panel_mockup_ce_20260911.png
function acctRiskTone(liqPct) {
  return liqPct < 3 ? "bad" : liqPct < 6 ? "warn" : "good";
}

// 닫힌 왕복 손익 막대 + 누적선. 데이터가 없으면 빈 문자열(자리 자체를 안 만든다).
function acctPerfSvg(net, meta) {
  if (!net.length) return "";
  // 2026-09-11 사용자 "높이를 가득 채워줘" -- 칸을 꽉 채우려면 비균일 확대를 피할 수 없다.
  // 🔴그래서 **확대에 걸리면 안 되는 것들은 확대를 끈다**: 선 두께는 vector-effect 로 고정하고,
  //   끝점은 원 대신 **길이 0 짜리 둥근 캡 선**으로 그린다(캡은 stroke 라 늘어나지 않아 정원이다).
  //   막대 모서리 4px 만 세로로 살짝 늘어나는데, 24px 막대에선 눈에 띄지 않는다.
  const W = 320, H = 104, zero = H * 0.54, padX = 3, padY = 8;
  let cum = 0;
  const cums = net.map((v) => (cum += v));
  // 🔴막대와 누적선이 **같은 축**을 쓴다(둘 다 USD). 그러니 상한도 둘을 함께 봐야 한다 --
  //   옛 판은 건당 최대값만 봐서 누적선이 SVG 밖으로 잘려 나갈 수 있었다.
  const peak = Math.max(...net.map(Math.abs), ...cums.map(Math.abs), 1e-9);
  const bw = (W - padX * 2) / net.length;
  const sc = (H / 2 - padY) / peak;
  const xAt = (i) => padX + i * bw;
  const yAt = (v) => zero - v * sc;
  // 막대: 24px 상한 · 인접 막대 사이 2px 표면 간격(테두리를 그리지 않는다)
  const bwFill = Math.min(24, Math.max(1.5, bw - 2));
  const bars = net.map((v, i) => {
    const x = xAt(i) + (bw - bwFill) / 2;
    const h = Math.max(Math.abs(v) * sc, 1);
    const r = Math.min(4, bwFill / 2, h);          // 바깥 끝만 둥글고 **기준선 쪽은 각지게**
    const up = v >= 0;
    const tip = up ? zero - h : zero + h;
    const d = up
      ? `M${x} ${zero}V${tip + r}Q${x} ${tip} ${x + r} ${tip}H${x + bwFill - r}Q${x + bwFill} ${tip} ${x + bwFill} ${tip + r}V${zero}Z`
      : `M${x} ${zero}V${tip - r}Q${x} ${tip} ${x + r} ${tip}H${x + bwFill - r}Q${x + bwFill} ${tip} ${x + bwFill} ${tip - r}V${zero}Z`;
    return `<path d="${d}" fill="var(--${up ? "good" : "bad"})" opacity="0.9"></path>`;
  }).join("");
  // 누적선 아래 워시 -- 계열 색이 아니라 중립 잉크다(누적은 손익 방향이 아니라 '합계'다).
  const line = cums.map((c, i) => `${(xAt(i) + bw / 2).toFixed(1)},${yAt(c).toFixed(1)}`);
  const area = `M${xAt(0) + bw / 2} ${zero}L${line.join("L")}L${xAt(net.length - 1) + bw / 2} ${zero}Z`;
  const lastX = xAt(net.length - 1) + bw / 2, lastY = yAt(cums[net.length - 1]);
  // 값은 점마다 찍지 않는다 -- 끝점 하나만 점으로 표시하고 숫자는 위 칩과 툴팁이 가진다.
  // 히트 영역은 막대가 아니라 **칸 전체 높이**다(얇은 막대를 정조준하게 만들지 않는다).
  // 키보드로도 같은 내용이 나와야 하므로 tabindex 를 준다 -- 툴팁이 값의 유일한 통로가 되면 안 된다.
  const hits = net.map((v, i) => {
    const m = (meta && meta[i]) || {};
    const rows = [
      `${m.when || `${i + 1}번째 왕복`}${m.side ? " · " + m.side : ""}`,
      m.qty ? `수량 ${m.qty} ETH · ${fmtUsd(m.notional)}` : "",
      m.entry && m.exit ? `진입 ${fmtUsd(m.entry)} → 청산 ${fmtUsd(m.exit)}` : "",
      `손익 <b class="${v < 0 ? "bad" : "good"}">${v >= 0 ? "+" : ""}${fmtUsd(v)}</b>`
        + ` · 누적 ${fmtUsd(cums[i])}`,
    ].filter(Boolean);
    // 앞 세 줄은 사용자 데이터라 이스케이프하고, 마지막 줄의 <b> 만 우리가 넣은 마크업이다.
    const html = rows.map((r, k) => (k === rows.length - 1 ? r : escapeHtml(r))).join("<br>");
    return `<rect x="${xAt(i).toFixed(1)}" y="0" width="${bw.toFixed(1)}" height="${H}" `
      + `fill="transparent" tabindex="0" data-tip="${escapeHtml(html)}"></rect>`;
  }).join("");
  // 길이 0 + round cap = 지름이 stroke-width 인 정원. non-scaling 이라 늘어나지 않는다.
  const dot = (w, color, op) => `<line x1="${lastX.toFixed(1)}" y1="${lastY.toFixed(1)}" `
    + `x2="${lastX.toFixed(1)}" y2="${lastY.toFixed(1)}" stroke="${color}" stroke-width="${w}" `
    + `stroke-linecap="round" vector-effect="non-scaling-stroke" opacity="${op}"></line>`;
  return `<svg class="acct-perf-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" `
    + `aria-label="닫힌 왕복 ${net.length}건의 건당 손익 막대와 누적 손익 선">`
    + `<path d="${area}" fill="var(--ink)" opacity="0.08"></path>`
    + bars
    + `<line x1="${padX}" y1="${zero}" x2="${W - padX}" y2="${zero}" stroke="var(--soft-line)" `
    + `stroke-width="1" vector-effect="non-scaling-stroke"></line>`
    + `<polyline points="${line.join(" ")}" fill="none" stroke="var(--ink)" stroke-width="2" `
    + `stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke" opacity="0.72"></polyline>`
    + dot(12, "var(--panel-strong)", "1")      // 2px 표면 링
    + dot(8, "var(--ink)", "0.9")              // 끝점 8px
    + hits
    + `</svg>`;
}

// 계좌 차트 툴팁. 🔴카드는 30초마다 innerHTML 로 통째로 다시 그려진다 -- 막대마다 리스너를
// 달면 매번 새로 달아야 하고 옛 것이 샌다. 컨테이너(#snapAcctPosition)는 안 바뀌므로 위임한다.
function bindAcctChartTip() {
  const host = el("snapAcctPosition");
  if (!host || host.dataset.tipBound) return;
  host.dataset.tipBound = "1";
  const tipOf = (t) => (t && t.closest ? t.closest(".acct-plot") : null)?.querySelector(".acct-tip");
  const show = (target, clientX) => {
    const tip = tipOf(target);
    if (!tip || !target.dataset.tip) return;
    tip.innerHTML = target.dataset.tip;
    tip.hidden = false;
    const box = tip.parentElement.getBoundingClientRect();
    const r = target.getBoundingClientRect();
    const x = (clientX === null || clientX === undefined ? r.left + r.width / 2 : clientX) - box.left;
    tip.style.left = `${Math.max(0, Math.min(box.width - tip.offsetWidth, x - tip.offsetWidth / 2))}px`;
  };
  const hide = (target) => { const tip = tipOf(target); if (tip) tip.hidden = true; };
  host.addEventListener("mousemove", (e) => {
    if (e.target.dataset && e.target.dataset.tip) show(e.target, e.clientX);
  });
  host.addEventListener("mouseout", (e) => { if (e.target.dataset && e.target.dataset.tip) hide(e.target); });
  host.addEventListener("focusin", (e) => { if (e.target.dataset && e.target.dataset.tip) show(e.target, null); });
  host.addEventListener("focusout", (e) => { if (e.target.dataset && e.target.dataset.tip) hide(e.target); });
}

// 값이 바뀐 노드만 짧게 강조한다. 막대는 CSS transition 으로 미끄러지지만 글자는 즉시
// 바뀌므로 눈이 놓친다 -- 그 한 곳만 메운다. 값이 같으면 아무것도 안 한다(폴링 떨림 방지).
function pvText(node, txt) {
  if (!node || node.textContent === txt) return;
  node.textContent = txt;
  node.classList.remove("pv-bump");
  void (node.offsetWidth ?? 0);      // 리플로우 -- 같은 클래스를 다시 붙여도 애니메이션이 재시작된다
  node.classList.add("pv-bump");
}

// 2026-09-20 계좌 카드 미리보기를 **되살렸다**(사용자 요청). 09-16 에 이걸 지우고 진입
// 블록 안에 「지금 넣으면」 4행(#entryProj)을 따로 뒀는데, 09-20 에 진입이 모달에서
// 카드 안으로 나오면서 그 4행이 **바로 위 계좌 카드와 같은 자리에서 같은 말**을 하게 됐다.
// 사용자: 「4가지를 굳이 또 게이지를 만들어서 보여줄 필요가 없어」. 아래 코드는 5e8fe121 에서
// 지운 것을 그대로 되돌린 것이다 -- 새로 쓰지 않았다.
//
// 🔴여기가 미리보기를 입히는 **유일한 곳**이다. 렌더는 항상 실제 계좌만 그린다 -- 두 곳에서
//    값을 만들면 «카드와 타일이 서로 다른 순간을 말하는» 상태가 생긴다.
// 🔴노드를 갈아치우지 않고 제자리에서 고친다. setH 로 다시 그리면 새 노드가 최종값으로
//    태어나 CSS transition 이 안 걸린다 -- 애니메이션이 목적이므로 이 구조가 조건이다.
function applyAcctPreview(pos, mark, equity) {
  const root = el("snapAcctPosition");
  const pv = entryProjPreview;
  if (!root) return;
  const q = (k) => root.querySelector(`[data-pv="${k}"]`);
  root.querySelectorAll(".acct-tiles").forEach((n) => n.classList.toggle("preview", !!pv));
  const cap = root.querySelector(".acct-pv-cap");
  if (cap) cap.hidden = !pv;
  if (!pv) { root.querySelectorAll(".acct-rail u").forEach((u) => u.remove()); return; }

  const a = pv.after, plan = pv.__plan || {};
  const LIQ_FULL = 10, EXPO_CAP = 30;
  const long = pos.side === "LONG";
  // 타일 셋: 값·색·막대·유령눈금(지금 자리)
  const setTile = (k, txt, tone, fill, ghost) => {
    const t = q(k); if (!t) return;
    const v = t.querySelector(".acct-tile-val"), rail = t.querySelector(".acct-rail");
    const bar = rail && rail.querySelector("i");
    if (v) { v.className = `acct-tile-val ${tone}`; pvText(v, txt); }
    if (bar) { bar.className = tone; bar.style.width = `${clamp01(fill) * 100}%`; }
    if (rail) {
      rail.classList.add("entry-rail");
      let u = rail.querySelector("u");
      if (!u) { u = document.createElement("u"); rail.appendChild(u); }
      u.style.left = `${clamp01(ghost) * 100}%`;
      u.title = "지금 자리";
    }
  };
  setTile("liq", `${Number(a.liq_pct).toFixed(2)}%`, acctRiskTone(a.liq_pct),
          a.liq_pct / LIQ_FULL, pv.before.liq_pct / LIQ_FULL);
  setTile("used", `${Number(a.margin_used_pct).toFixed(0)}%`,
          a.margin_used_pct > 80 ? "bad" : a.margin_used_pct > 60 ? "warn" : "good",
          a.margin_used_pct / 100, pv.before.margin_used_pct / 100);
  setTile("expo", `${Number(a.exposure_x).toFixed(1)}배`, a.exposure_x > 15 ? "bad" : "warn",
          a.exposure_x / EXPO_CAP, pv.before.exposure_x / EXPO_CAP);

  // 포지션 카드: 수량·레버리지·평단·청산가·손잡이. 평단은 **체결가 가중평균**이다.
  const addQty = Number(plan.quantity) || 0, addPx = Number(plan.price) || 0;
  const haveQty = Number(pos.qty) || 0, havePx = Number(pos.entry_price) || 0;
  const newQty = haveQty + addQty;
  const newEntry = newQty > 0 ? (haveQty * havePx + addQty * addPx) / newQty : havePx;
  // 청산가는 거래소가 준 **거리(%)** 에서 되돌린다 -- 근사식보다 실측에 앵커된 값이다.
  const newLiq = mark > 0 ? mark * (1 + (long ? -1 : 1) * Number(a.liq_pct) / 100) : 0;
  pvText(q("qty"), `~${newQty.toFixed(3)}`);
  const tg = q("tag");
  if (tg && plan.target_leverage) pvText(tg, `${long ? "롱" : "숏"} ×${plan.target_leverage}`);
  pvText(q("liqpx"), `청산 ~${fmtUsd(newLiq)}`);
  // 🔴평단은 **추측치**다. 체결가를 peg 호가로 가정한 가중평균이라 실제 체결(부분체결·
  //   테이커 폴백·슬리피지)에 따라 달라진다. `~` 와 툴팁으로 그 사실을 남긴다.
  const ep = q("entrypx");
  if (ep) {
    pvText(ep, `진입 ~${fmtUsd(newEntry)}`);
    ep.title = `추측치 — 지금 ${fmtUsd(havePx)} (${haveQty.toFixed(3)} ETH)에`
      + ` ${addQty.toFixed(3)} ETH 를 ${fmtUsd(addPx)}(peg 호가)에 더한 가중평균입니다.`
      + `\n실제 체결가가 다르면(부분체결·테이커 폴백) 평단도 달라집니다.`;
  }
  const kn = q("knob");
  if (kn) {
    const span = Math.abs(newEntry - newLiq) * 2;
    kn.style.left = `${(clamp01(span > 0 ? Math.abs(mark - newLiq) / span : 0) * 100).toFixed(1)}%`;
  }
}

let entryProjPreview = null;
let lastAcctPos = null;   // 패처가 쓰는 마지막 렌더 문맥(포지션·마크가·순자산)
let entryProjKey = "";

function renderSnapshotAccount() {
  const summary = el("snapAcctSummary");
  if (!el("snapAcctPosition")) return;
  if (!latestBinanceAccount) {
    if (summary) { summary.textContent = "데이터 없음"; summary.className = "ops-health-summary neutral"; }
    setT("snapAcctBalance", "-");
    setH("snapAcctPosition", '<p class="muted">계좌를 불러오지 못했습니다.</p>');
    return;
  }
  const b = latestBinanceAccount.balance || {};
  const wallet = Number(b.wallet) || 0;
  const upnl = Number(b.unrealized) || 0;
  const equity = wallet + upnl;
  setT("snapAcctBalance", `지갑 ${fmtUsd(wallet)} · 가용 ${fmtUsd(b.available)}${balanceAssetNote()}`);
  const pos = snapshotAccountPosition();
  const others = (latestBinanceAccount.positions || []).length - (pos ? 1 : 0);
  if (summary) {
    summary.textContent = pos ? (pos.side === "LONG" ? "롱 보유" : "숏 보유") : "포지션 없음";
    summary.className = `ops-health-summary ${pos ? (pos.side === "LONG" ? "good" : "bad") : "neutral"}`;
  }

  // ── 히어로: 숫자 하나가 헤드라인이다 ─────────────────────────────────────────
  // 옛 판은 같은 크기 숫자 셋을 나란히 둬서 무엇부터 볼지 알 수 없었다. 순자산을 키우고
  // 나머지는 타일로 내린다. 미실현은 색 글씨가 아니라 **알약**이라 흑백으로 봐도 읽힌다.
  const upnlPct = wallet > 0 ? upnl / wallet * 100 : 0;
  const dTone = upnl > 0 ? "good" : upnl < 0 ? "bad" : "neutral";
  const hero = `<div class="acct-hero">
      <span class="acct-eyebrow">순자산</span>
      <div class="acct-figure">${fmtUsd(equity)}</div>
      <span class="acct-delta ${dTone}">${upnl > 0 ? "▲" : upnl < 0 ? "▼" : "–"} ${fmtUsd(upnl)}
        <i>${upnlPct >= 0 ? "+" : ""}${upnlPct.toFixed(2)}%</i></span>
    </div>`;

  // 오른쪽 성과 -- 보고 있는 코인의 **닫힌** 왕복만 (패널이 코인 단위이므로 심볼로 거른다)
  const symbol = ASSET_CONFIG[activeSnapshotAsset]?.symbol || `${activeSnapshotAsset.toUpperCase()}USDT`;
  // 🔴/api/binance-account 는 **최신순**으로 준다. 그대로 그리면 누적선이 시간을 거꾸로 달린다.
  const closed = (latestBinanceAccount.trades || []).filter((t) => t.closed && t.symbol === symbol)
    .slice().sort((x, y) => (Number(x.exit_time) || 0) - (Number(y.exit_time) || 0));
  const net = closed.map((t) => Number(t.net_pnl) || 0);
  // 툴팁이 "날짜와 크기 등"을 보여줘야 하므로(사용자 지시) 라벨 문자열이 아니라 원장을 넘긴다.
  const netMeta = closed.map((t) => {
    const d = new Date(Number(t.exit_time) || 0);
    const qty = Number(t.max_qty) || 0, px = Number(t.exit_price) || 0;
    return {
      when: Number.isNaN(d.getTime()) ? "" : `${d.getMonth() + 1}/${d.getDate()} `
        + `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`,
      side: t.side === "LONG" ? "롱" : "숏", qty, notional: qty * px,
      entry: Number(t.entry_price) || 0, exit: px,
    };
  });
  const wins = net.filter((v) => v > 0).length;
  const total = net.reduce((x, y) => x + y, 0);
  let worstIdx = -1;
  net.forEach((v, i) => { if (worstIdx < 0 || v < net[worstIdx]) worstIdx = i; });
  const rest = worstIdx >= 0 ? total - net[worstIdx] : 0;
  const chip = (v, lab) => `<span class="acct-chip"><b>${v}</b><span>${lab}</span></span>`;
  // 기간을 적는다(사용자 지시). ⚠️"이번 달 전부"라고 단정하지 않는다 -- 거래소는 시간 조건을
  // 안 주면 최근 구간만 돌려주므로, 여기 있는 건 **조회된 범위**지 계좌의 전체 이력이 아니다.
  const spanText = (() => {
    if (!closed.length) return "";
    const a = new Date(Number(closed[0].exit_time) || 0);
    const b = new Date(Number(closed[closed.length - 1].exit_time) || 0);
    if (Number.isNaN(a.getTime()) || Number.isNaN(b.getTime())) return "";
    const sameMonth = a.getMonth() === b.getMonth() && a.getFullYear() === b.getFullYear();
    const thisMonth = b.getMonth() === new Date().getMonth() && b.getFullYear() === new Date().getFullYear();
    const range = sameMonth ? `${a.getMonth() + 1}월 ${a.getDate()}일~${b.getDate()}일`
      : `${a.getMonth() + 1}월 ${a.getDate()}일~${b.getMonth() + 1}월 ${b.getDate()}일`;
    return (sameMonth && thisMonth ? "이번 달 · " : "") + range;
  })();
  // 계열이 둘(건당 막대 · 누적선)이므로 범례를 항상 둔다. 적록 2색형에서 이익/손실 색 차이가
  // ΔE 6.3 이라 **색만으로는 부족**하다 -- 영선 위/아래 위치와 이 범례가 보조 부호다.
  // 글씨는 계열 색을 입지 않는다(규약): 색은 옆의 견본이 지고 글자는 muted 잉크다.
  const legend = `<div class="acct-legend">
      <span><i class="sw good"></i>이익</span>
      <span><i class="sw bad"></i>손실</span>
      <span><i class="sw ln"></i>누적</span>
    </div>`;
  const perf = net.length
    ? `<section class="acct-perf">
         <div class="acct-chips">
           ${chip(net.length, "왕복")}
           ${chip(`${Math.round(wins / net.length * 100)}%`, "승률")}
           ${chip(`<span class="${total < 0 ? "bad" : "good"}">${fmtUsd(total)}</span>`, "누적")}
         </div>
         <figure class="acct-plot">
           <div class="acct-plot-head">${legend}
             ${spanText ? `<span class="acct-span" title="거래소가 시간 조건 없이 돌려주는 최근 구간의 기록입니다 — 이보다 과거는 조회 조건을 따로 줘야 나옵니다.">${escapeHtml(spanText)}</span>` : ""}
           </div>
           ${acctPerfSvg(net, netMeta)}
           <div class="acct-tip" hidden></div>
         </figure>
         ${worstIdx >= 0 && net[worstIdx] < 0 && net.length > 1
            ? `<p class="acct-perf-note"><span class="bad">최악 1건 ${fmtUsd(net[worstIdx])}</span>
                 · <span class="${rest < 0 ? "bad" : "good"}">나머지 ${net.length - 1}건 ${fmtUsd(rest)}</span></p>`
            : ""}
       </section>`
    : `<section class="acct-perf"><div class="acct-empty">닫힌 왕복이 아직 없습니다.</div></section>`;

  const otherNote = others > 0
    ? `<p class="acct-foot">다른 코인에 ${others}종목을 더 보유 중입니다 — 운영 관리 탭에서 전부 볼 수 있습니다.</p>`
    : "";
  if (!pos) {
    setH("snapAcctPosition", `<div class="acct-card">
        <div class="acct-main">${hero}
          <div class="acct-empty">${ASSET_CONFIG[activeSnapshotAsset]?.label
            || activeSnapshotAsset.toUpperCase()}에 열린 포지션이 없습니다.</div>
        </div>${perf}
      </div>${otherNote}`);
    bindAcctChartTip();
    return;
  }

  const mark = Number(pos.mark_price) || 0, liq = Number(pos.liquidation_price) || 0;
  const entry = Number(pos.entry_price) || 0;
  const liqPct = mark > 0 ? Math.abs(mark - liq) / mark * 100 : 0;
  // 🔴2026-09-11 정정(사용자 지적). b.margin 은 바이낸스 totalMarginBalance = **순자산**
  //   (지갑+미실현)이지 사용 증거금이 아니다. 그걸 쓰던 탓에 "증거금 사용"이 98.6% 로 떴다
  //   -- 실제는 39.6%. 이 값은 정의상 거의 항상 100% 근처라 **아무것도 재고 있지 않았다**.
  //   이제 수집기가 initial_margin(totalInitialMargin)을 그대로 싣는다. 옛 상태파일을 만나면
  //   순자산−가용으로 되짚는다(바이낸스 정의상 같은 값이고, 실측으로 368.60 일치 확인).
  // equity 는 이 함수 앞쪽(991행)에서 이미 선언돼 있다 -- 같은 식(wallet + upnl)이고
  // wallet/upnl 이 const 라 값이 바뀔 수 없으므로 그대로 쓴다. 여기서 다시 const 로
  // 선언하면 **같은 스코프 중복 선언**이라 app.js 전체가 SyntaxError 로 죽는다(2026-09-11 실장애).
  const usedMargin = Number.isFinite(Number(b.initial_margin)) ? Number(b.initial_margin)
    : Math.max(0, (Number(b.margin) || 0) - (Number(b.available) || 0));
  const usedPct = equity > 0 ? usedMargin / equity * 100 : 0;
  // 노출은 **계좌 전체** 기준이다(명목 ÷ 순자산). 포지션 레버리지(×30)와 다른 값이라
  //   같은 "배"를 써서 혼동이 났다 -- 라벨을 「계좌 노출」로 바꾸고 명목을 툴팁에 적는다.
  const expo = equity > 0 ? (Number(pos.notional) || 0) / equity : 0;
  const EXPO_CAP = 30;   // 막대 상한. 이 계좌 실측이 23배라 30을 만재로 둔다
  const LIQ_FULL = 10;   // 청산까지 10% 를 만재로 본다(그 이상은 사실상 안전)
  // 타일: 라벨·값·레일이 셋 다 같은 모양이라 눈이 세로로 훑힌다(옛 판은 숫자 셋 + 별도 막대).
  // ⭐여기서는 **항상 실제 계좌**를 그린다. 진입 미리보기는 렌더가 아니라 applyAcctPreview 가
  //   **같은 노드를 제자리에서** 고친다 -- 노드를 갈아치우면 transition 이 안 걸린다.
  //   `data-pv` 가 그 손잡이다.
  const tile = (key, lab, val, tone, fill, title) => `<div class="acct-tile" data-pv="${key}"${
      title ? ` title="${escapeHtml(title)}"` : ""}>
      <span class="acct-tile-lab">${lab}</span>
      <b class="acct-tile-val ${tone}">${val}</b>
      <span class="acct-rail"><i class="${tone}" style="width:${clamp01(fill) * 100}%"></i></span>
    </div>`;
  const tiles = `<div class="acct-pv-cap entry-cap" hidden>진입 미리보기 — 지금 넣으면 (실제 계좌 아님)</div>
    <div class="acct-tiles">
      ${tile("liq", "청산까지", `${liqPct.toFixed(2)}%`, acctRiskTone(liqPct), liqPct / LIQ_FULL,
             `마크 ${fmtUsd(mark)} → 청산 ${fmtUsd(liq)}\n교차증거금이라 1/레버리지(${
               (100 / (Number(pos.leverage) || 1)).toFixed(2)}%)가 아니라 지갑 전체가 버팁니다.`)}
      ${tile("used", "증거금 사용", `${usedPct.toFixed(0)}%`,
             usedPct > 80 ? "bad" : usedPct > 60 ? "warn" : "good", usedPct / 100,
             `사용 ${fmtUsd(usedMargin)} ÷ 순자산 ${fmtUsd(equity)}\n= 명목 ${
               fmtUsd(pos.notional)} ÷ 레버리지 ${pos.leverage}배`)}
      ${tile("expo", "계좌 노출", `${expo.toFixed(1)}배`, expo > 15 ? "bad" : "warn", expo / EXPO_CAP,
             `명목 ${fmtUsd(pos.notional)} ÷ 순자산 ${fmtUsd(equity)}\n포지션 레버리지(${
               pos.leverage}배)와 다른 값입니다 — 증거금을 계좌의 일부만 썼기 때문입니다.`)}
    </div>`;

  // ⭐청산 거리 게이지 -- 롱/숏 모두 **왼쪽 끝이 청산**이 되도록 접는다.
  //   0 = 청산가 · 0.5 = 진입가 · 1 = 진입에서 청산 거리만큼 이익 난 가격.
  //   측면마다 부등호를 뒤집지 않아도 되고, 눈은 "왼쪽에 가까울수록 위험"만 기억하면 된다.
  const span = Math.abs(entry - liq) * 2;
  const safe = span > 0 ? clamp01(Math.abs(mark - liq) / span) : 0;
  const sideTone = pos.side === "LONG" ? "good" : "bad";
  const position = `<div class="acct-pos" data-side="${pos.side === "LONG" ? "long" : "short"}">
      <div class="acct-pos-head">
        <b>${escapeHtml(pos.symbol)}</b>
        <span class="acct-tag ${sideTone}" data-pv="tag">${pos.side === "LONG" ? "롱" : "숏"} ×${escapeHtml(pos.leverage)}</span>
        <span class="acct-pos-qty" data-pv="qty">${escapeHtml(pos.qty)}</span>
      </div>
      <div class="acct-gauge" title="왼쪽 끝이 청산가, 가운데 눈금이 진입가입니다. 손잡이가 왼쪽에 붙을수록 위험합니다.">
        <span class="acct-gauge-track"></span>
        <span class="acct-gauge-entry"></span>
        <span class="acct-gauge-knob" data-pv="knob" style="left:${(safe * 100).toFixed(1)}%"></span>
      </div>
      <div class="acct-gauge-legend">
        <span class="bad" data-pv="liqpx">청산 ${fmtUsd(liq)}</span>
        <span data-pv="entrypx">진입 ${fmtUsd(entry)}</span>
        <span class="acct-gauge-now">현재 ${fmtUsd(mark)}</span>
      </div>
    </div>`;

  setH("snapAcctPosition", `<div class="acct-card">
      <div class="acct-main">${hero}${tiles}${position}</div>
      ${perf}
    </div>${otherNote}`);
  bindAcctChartTip();
  // 🔴계좌 폴링이 카드를 다시 그리면 미리보기가 지워진다 -- 렌더 직후 곧바로 다시 입힌다.
  //   (이 경로는 노드가 새로 생겨서 애니메이션은 안 걸린다. 값이 맞는 게 먼저다.)
  lastAcctPos = { pos, mark, equity };
  applyAcctPreview(pos, mark, equity);
}


// 거래소 실계좌(수동 매매 포함) 패널. 봇 원장(trade_journal)과 달리 여기 숫자는 바이낸스가 준 것.
function renderBinanceAccount(payload) {
  // 스냅샷 탭 요약과 청산맵 진입선이 같은 값을 쓴다 -- 두 번 받지 않도록 여기서 보관한다.
  latestBinanceAccount = payload?.ok ? payload : null;
  if (payload?.ok) { lastGoodAccount = payload; lastGoodAccountAt = Date.now(); }
  renderSnapshotAccount();
  renderSnapshotChart();
  // 🔴계좌가 **도착하는 즉시** 청산 버튼을 맞춘다. 예전에는 60초 주기(MANUAL_ENTRY_REFRESH_MS)
  // 에만 맞춰서, 화면을 열고 **63초 뒤에야** 청산 버튼이 나타났다(2026-09-13 실측).
  // 급히 닫으려고 연 사람에게 1분을 기다리게 하는 건 이 버튼의 존재 이유와 정면으로 어긋난다.
  if (typeof manualExitSyncButtons === "function") manualExitSyncButtons();
  const summary = el("acctSummary");
  if (!payload?.ok) {
    const msg = payload?.hint || payload?.error || "계정을 불러오지 못했습니다.";
    if (summary) { summary.textContent = "연결 안 됨"; summary.className = "ops-health-summary bad"; }
    setT("acctBalanceText", msg);
    setH("acctPositions", "");
    setH("acctTrades", "");
    return;
  }
  const b = payload.balance || {};
  setT("acctBalanceText", `지갑 ${fmtUsd(b.wallet)} · 평가 ${fmtUsd(b.margin)} · 가용 ${fmtUsd(b.available)}`
    + ` · 미실현 ${fmtUsd(b.unrealized)}${balanceAssetNote()}`);
  const positions = payload.positions || [];
  const trades = payload.trades || [];
  if (summary) {
    summary.textContent = positions.length ? `보유 ${positions.length}종목` : "포지션 없음";
    summary.className = `ops-health-summary ${positions.length ? "good" : "neutral"}`;
  }
  setH("acctPositions", positions.length ? positions.map((p) => {
    const tone = p.unrealized_pnl > 0 ? "good" : p.unrealized_pnl < 0 ? "bad" : "neutral";
    return `<article class="ops-health-row ${tone}">
      <span class="ops-health-dot" aria-hidden="true"></span>
      <div class="ops-health-info">
        <strong>${escapeHtml(p.symbol)} ${p.side === "LONG" ? "롱" : "숏"} ×${escapeHtml(p.leverage)}</strong>
        <span>진입 ${fmtUsd(p.entry_price)} → 현재 ${fmtUsd(p.mark_price)} · 청산가 ${fmtUsd(p.liquidation_price)} · 수량 ${escapeHtml(p.qty)}</span>
      </div>
      <div class="ops-health-meta">
        <span class="ops-health-status-badge">${fmtUsd(p.unrealized_pnl)}</span>
        <small>${fmtTs(p.entry_at)} 진입</small>
      </div>
    </article>`;
  }).join("") : '<p class="muted">열려 있는 포지션이 없습니다.</p>');
  // 이력이 잘리면 가장 오래된 왕복은 창 밖에서 열렸을 수 있어 진입가/방향이 틀릴 수 있다.
  const truncNote = (payload.trades_truncated || []).length
    ? `<p class="muted">${escapeHtml((payload.trades_truncated || []).join(", "))}는 체결 이력이 잘려 가장 오래된 왕복이 부정확할 수 있습니다.</p>` : "";
  setH("acctTrades", trades.length ? truncNote + trades.slice(0, 20).map((t) => {
    const tone = !t.closed ? "warn" : t.net_pnl > 0 ? "good" : t.net_pnl < 0 ? "bad" : "neutral";
    return `<article class="ops-health-row ${tone}">
      <span class="ops-health-dot" aria-hidden="true"></span>
      <div class="ops-health-info">
        <strong>${escapeHtml(t.symbol)} ${t.side === "LONG" ? "롱" : "숏"}</strong>
        <span>${fmtTs(t.entry_at)} 진입 → ${t.closed ? `${fmtTs(t.exit_at)} 청산` : "보유 중"} · ${escapeHtml(t.fills)}회 체결</span>
      </div>
      <div class="ops-health-meta">
        <span class="ops-health-status-badge">${t.closed ? fmtUsd(t.net_pnl) : "-"}</span>
        <small>수수료 ${fmtUsd(t.commission)}</small>
      </div>
    </article>`;
  }).join("") : '<p class="muted">체결 내역이 없습니다.</p>');
}

function fmtUsd(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return `${n >= 0 ? "" : "-"}$${Math.abs(n).toLocaleString("en-US", { maximumFractionDigits: Math.abs(n) < 10 ? 4 : 2 })}`;
}

async function refreshBinanceAccount() {
  // ops 탭(refreshOpsStatus)과 스냅샷 탭(tick) 양쪽에서 부르므로 자체 게이트를 둔다.
  const now = Date.now();
  if (now - binanceAccountLastFetchAt < BINANCE_ACCOUNT_POLL_MS) return;
  binanceAccountLastFetchAt = now;
  try {
    const res = await fetch(API_BINANCE_ACCOUNT_URL, { cache: "no-cache" });
    renderBinanceAccount(await res.json());
  } catch (error) {
    console.error("Binance account fetch error:", error);
    renderBinanceAccount({ ok: false, error: "대시보드 서버에 연결하지 못했습니다." });
  }
}


async function refreshOpsStatus() {
  const now = Date.now();
  if (now - opsLastFetchAt < OPS_POLL_MS) return;
  opsLastFetchAt = now;
  refreshBinanceAccount();
  try {
    const res = await fetch(API_OPS_STATUS_URL, { cache: "no-cache", headers: opsStatusEtag ? { "If-None-Match": opsStatusEtag } : {} });
    if (res.status === 304) return;
    if (!res.ok) throw new Error(`ops status ${res.status}`);
    opsStatusEtag = res.headers.get("ETag") || opsStatusEtag;
    renderOpsStatus(await res.json());
  } catch (error) {
    console.error("Ops status fetch error:", error);
    const badge = el("opsWatchdogBadge");
    if (badge) { badge.className = "ops-badge bad"; badge.textContent = "WATCHDOG UNREACHABLE"; }
  }
}

// Small per-bar activity strip (oldest bar left, most recent bar right) for a "good"/"bad"/
// "neutral" tone-per-bar history -- builds an SVG string directly (no DOM diffing), discrete
// bars instead of a continuous line. The most recent
// bar gets the "evidence-bar-live" class ONLY when it's non-neutral, which is what styles.css
// hooks the pulsing animation to -- idle bars stay static (just color-transition on change, via
// CSS), so several strips sitting side by side don't all pulse at once when nothing is actually
// happening. Shared by both the evidence-signal strips (bottom/top fired -> tone) and the
// Snapshot tab's model-indicator strips (thresholded value -> tone).
// Builds an oldest-to-newest array of ISO timestamps for a strip whose bars are known to be evenly
// spaced (server-computed histories: evidence signals/v_rebound at 5-min klines) -- the payload only ever sends the LATEST bar's timestamp, so the rest
// are derived by walking back stepMinutes at a time. Returns [] if latestIso is missing (not warmed
// up yet), so hover-time silently does nothing rather than showing a wrong guess.
function evenlySpacedBarTimes(latestIso, n, stepMinutes) {
  if (!latestIso || !(n > 0)) return [];
  const latestMs = Date.parse(latestIso);
  if (!Number.isFinite(latestMs)) return [];
  const stepMs = stepMinutes * 60000;
  return Array.from({ length: n }, (_, i) => new Date(latestMs - (n - 1 - i) * stepMs).toISOString());
}

// 2026-08-25: hover-time -- times[i] (oldest-to-newest, parallel to tones) is optional; a bar with
// no known time just renders without the hover handlers, no error. NOT shown on the graph itself
// (tried that first, user asked to move it off) -- instead read by showStripBarTime/hideStripBarTime
// below, which write into a .strip-time-now label that each row template places on its OWN line
// (model indicators: the "자세히" line; evidence signals: the 바닥/천장 caption line).
// key (2026-08-31, optional): signal identity for hover -- stashed on the <svg> root as data-key so
// showStripBarTime() can look up STRIP_BAR_LABEL_BY_TONE[key][tone] without threading it through
// every <rect>. Omitted, hover falls back to time-only (unchanged old behavior).
// 2026-08-31 user request ("병합 세그먼트로 바꿔줘"): merge consecutive same-tone bars into one
// wider rect instead of drawing every raw 5-min bar -- segment WIDTH now carries duration (design
// candidate "02 병합 세그먼트"), paired with the persistent time axis below it (stripAxisHtml,
// unchanged by this). Height briefly shrunk 20->10 to match .evidence-strip-axis's compact row, but
// the user found that too short to read and asked it back to the original 20 -- no room for
// in-segment text either way, so hover (unchanged mechanism, resolves to the whole segment's
// tone/start~end range) still carries the exact label+time, same tradeoff this dashboard's other
// compact chips already make.
// calls (2026-09-08, optional): 봉별 **판정 단어**. 톤이 방향만 담는 신호(돌파/되돌림)에서
// 같은 ↓ 가 "돌파"일 수도 "되돌림"일 수도 있어, 톤만으로는 세그먼트도 라벨도 구분이 안 된다.
// 넘기지 않는 호출부는 callList[i] 가 undefined 라 "" 로 떨어져 동작이 그대로다.
function toneStripSvg(tones, times, provisionalLast, liveFiring, key, calls) {
  const list = Array.isArray(tones) ? tones : [];
  const timeList = Array.isArray(times) ? times : [];
  const callList = Array.isArray(calls) ? calls : [];
  const n = Math.max(list.length, 1);
  const w = 240, h = 15, gap = 1.5;
  const bw = Math.max((w - gap * (n - 1)) / n, 1);

  // Group consecutive equal tones into segments. The still-forming provisional bar (always the last
  // array entry, see evidenceStripSvg's liveTone param) never merges into the segment before it even
  // when its tone happens to match -- keeps evidence-bar-provisional's softened fill scoped to only
  // the genuinely-unconfirmed portion instead of bleeding across a whole merged block.
  // 🔴2026-09-10 (user report, twice: "연속으로 같은 신호가 나왔는데 게이지가 하나로 안 합쳐진다"
  // -> "아직도 게이지 칸이 끊겨서 나온다"): the 2026-09-01 rawFire boundary is GONE. Two reasons.
  // (1) It read rawFire as an EVENT column, but all 8 raw columns are LEVEL (threshold) conditions
  //     -- `dem <= 0.10`, `kalman_dev_z <= -2.0`, ... (live_evidence_signal_dashboard_20260823.py::
  //     compute_signals) with no edge detection or dedup -- so they stay true on EVERY bar the
  //     condition holds. Measured over 224,353 bars: 30.9% of fire bars had the previous bar firing
  //     too (demarker_extreme 69.5%, runs up to 24 bars = 2h), each split into its own 1-bar cell.
  // (2) Worse, the tone is bottom-wins (see evidenceStripSvg) so a TOP-side fire inside a lit
  //     BOTTOM window broke the strip with **no visible reason at all** -- same green on both
  //     sides of the break. taker_delta_z_climax 2026-09-09 23:25 / 00:35 were exactly this.
  // Segments now merge purely on what the eye can see (tone + call). Every boundary therefore has
  // a visible cause, and 혼재(both sides lit) is its own warn tone rather than hiding under 바닥.
  // Re-fire timing still lives in the caption/hover, which read the same segments.
  const segments = [];
  for (let i = 0; i < n; i++) {
    const tone = list[i] || "neutral";
    const isProvisionalBar = !!(provisionalLast && i === n - 1);
    const call = callList[i] || "";
    const prev = segments[segments.length - 1];
    if (prev && prev.tone === tone && prev.call === call && !isProvisionalBar) {
      prev.end = i;
    } else {
      segments.push({ tone, call, start: i, end: i, isProvisional: isProvisionalBar });
    }
  }

  const bars = segments.map((seg) => {
    const tone = seg.tone;
    // 2026-08-25: this mapping originally had no "warn" branch at all (fell into the generic gray
    // fallback below), then briefly used --amber (yellow) to match .signal-chip.warn's color at the
    // time -- user then asked for the whole Snapshot tab's 주의 color to be yellow-free and unified
    // on --warn (orange) instead, so this now matches that.
    const fill = tone === "good" ? "var(--good)" : tone === "bad" ? "var(--bad)" : tone === "warn" ? "var(--warn)" : "rgba(203,209,227,0.16)";
    const isLastSeg = seg.end === n - 1;
    // 2026-08-27 (user request): the last/rightmost bar used to always get a distinct outline
    // (evidence-bar-now, blinking at first, then static) just for being the "now" position --
    // removed entirely, position alone isn't a meaningful signal on its own. evidence-bar-live
    // (tone-colored pulse) still applies when the last segment is actively firing; the whole-gauge
    // evidence-strip-live blink (see toneStripSvg's return) is the only thing marking "now" at all,
    // and only while genuinely live/provisional.
    let cls = "evidence-bar";
    if (isLastSeg && tone !== "neutral") cls += " evidence-bar-live";
    // 2026-08-26: when the caller appends a still-forming bar (see evidenceStripSvg's liveTone
    // param), that bar lands here as its own segment (see the merge guard above) -- this class marks
    // it as "not yet confirmed" (softened fill, see .evidence-bar-provisional), same honesty-signal
    // requirement as the provisional badge/chip dots elsewhere (see renderEvidenceSignalsProvisional).
    if (seg.isProvisional) cls += " evidence-bar-provisional";
    const count = seg.end - seg.start + 1;
    const x = (seg.start * (bw + gap)).toFixed(1);
    const segWidth = (count * bw + (count - 1) * gap).toFixed(1);
    // data-t/data-t-end carry the segment's START/END bar times (plain ISO strings, safe unescaped
    // in an HTML attribute) -- read back + formatted at hover time (showStripBarTime) so
    // fmtShortTs/fmtTimeOnly/fmtHourMinute runs once per hover instead of once per bar per render.
    // data-t-end (2026-08-31): "그 한 칸의 시작과 끝 시간" -- a 1-bar segment has start===end, shown
    // as a single time rather than a redundant "11:35~11:35" range (see showStripBarTime). data-tone
    // carries the segment's tone so hover can resolve its label -- the fill color alone isn't
    // readable back from the DOM.
    const t = timeList[seg.start];
    const tEnd = timeList[seg.end];
    const callAttr = seg.call ? ` data-call="${escapeHtml(seg.call)}"` : "";
    const hoverAttrs = t ? ` data-t="${t}" data-t-end="${tEnd || t}" data-tone="${tone}"${callAttr} onmouseenter="showStripBarTime(this)" onmouseleave="hideStripBarTime(this)"` : "";
    return `<rect class="${cls}" x="${x}" y="0" width="${segWidth}" height="${h}" rx="2" fill="${fill}"${hoverAttrs}/>`;
  });
  // 2026-08-27 (user request): the whole gauge blinks, but only while it's showing a genuinely
  // in-progress reading -- the still-forming bar (liveFiring, from evidenceStripSvg's liveTone) is
  // both provisional AND currently non-neutral. A provisional-but-neutral forming bar (most common
  // case) or a fully confirmed render (model indicators always, evidence signals between polls)
  // stays static -- blink is reserved for "something is actively firing right now, not yet final".
  const keyAttr = key ? ` data-key="${key}"` : "";
  return `<svg class="evidence-strip${liveFiring ? " evidence-strip-live" : ""}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"${keyAttr}>${bars.join("")}</svg>`;
}


// 2026-08-31 user request: a persistent time axis under each history strip, instead of only
// revealing a bar's time on hover -- 5 evenly spaced ticks (first/quarter/half/three-quarter/last,
// deduped for short arrays) so "roughly when" is visible without any interaction; hover (see
// showStripBarTime) still gives the exact bar's time + label (unaffected by this -- the user asked
// only for the persistent axis to change). timeFmtKind: model indicators use "time" (HH:MM:SS,
// fmtTimeOnly); evidence signals use "hm" (HH:MM only, fmtHourMinute -- 2026-08-31 user request:
// "증거신호에서... 시와 분만 표시", dropping fmtShortTs's date since 5 ticks that close together
// almost never cross midnight); anything else falls back to fmtShortTs (MM-DD HH:MM).
function stripAxisHtml(times, timeFmtKind) {
  const list = (Array.isArray(times) ? times : []).filter(Boolean);
  const n = list.length;
  if (n < 2) return "";
  const fmt = timeFmtKind === "time" ? fmtTimeOnly : timeFmtKind === "hm" ? fmtHourMinute : fmtShortTs;
  const idxs = [...new Set([0, Math.round((n - 1) / 4), Math.round((n - 1) / 2), Math.round((n - 1) * 3 / 4), n - 1])];
  const labels = idxs.map((i) => `<span>${escapeHtml(fmt(list[i]))}</span>`).join("");
  return `<div class="evidence-strip-axis">${labels}</div>`;
}

// tone -> label vocabulary per signal "shape", for hover only (2026-08-31 user request: "커서를
// 막대 배열에 올리면 라벨도 그 커서에 맞는 라벨과 시간을 표시"). Each model-indicator key has its
// own wording (mirrors that signal's own live subText function -- directionalCaution/
// liqDirectionSubText/basisLiquiditySubText/liqCascadeHint/vReboundSubText above); all 8 evidence
// signals share one vocabulary under the "evidence" key (matches evidenceSideLabel). Deliberately
// separate from MODEL_INDICATOR_MEANING (keyed by the exact CURRENT subText, including states a
// single past tone can't reconstruct -- "웜업" 같은 운영 상태
// qualifier, which isn't stored per history bar, only tone is).
// 2026-09-10 사용자 요청: "V자 급등락과 앵커 돌파/되돌림도 증거신호 라벨처럼 익절 가격을 확률
// 아래에, 같은 포맷으로". 증거신호의 `익절 {가격}` (renderEvidenceSignals) 과 같은 문자열을
// 같은 자리(.meter-price, 규약 §3의 "확률이 아닌 수치")에 놓는다. 세 카드가 이제 한 포맷이다.
// ⚠️세 신호의 목표가는 **각자 자기 라벨**에서 온다 -- 증거신호 K×ATR 터치(intrabar), V자
//   1.5×ATR 빠른 다리(종가), 돌파/되돌림 ±0.8×ATR 배리어(intrabar). 같은 포맷이라고 같은
//   규약이 아니다. 컨벤션을 신호 간에 옮기지 않는다(CLAUDE.md 배리어 컨벤션 항목).
// 🔴2026-09-11 사용자 지적 "급락인데 익절이 현재가 위에 있다". 계산은 맞다 -- 익절가는 라벨
//   그대로 **발동봉의 극점**에서 1.5×ATR 이다. 발동봉이 크면 그 목표를 같은 봉이 이미 지나쳐
//   버려서, 앞으로 갈 자리처럼 보이던 숫자가 실제로는 뒤에 있다. 실측: 발동봉 레인지가 ATR 의
//   2~3배면 38.6%, 6배 이상이면 **86.6%** 가 발동봉 종가에서 이미 도달해 있다(전체 7.65%).
//   숫자를 숨기지 않고 **이미 지났다고 말한다** -- 라벨 목표 자체는 그 값이 맞기 때문이다.

const STRIP_BAR_LABEL_BY_TONE = {
  // 2026-09-09 극점 탐지기. 이 칩은 **사건의 측면**을 말하는 자리라 증거신호 어휘를 쓴다
  // (규약 §1: 특화감지기의 롱/숏은 포지션 방향일 때다). 축이 하나뿐이라 §5-4 문제 없음.
  extreme_detector: { good: "바닥 발동", bad: "천장 발동", neutral: "미발동" },
  // 2026-09-11 전환 탐지기. 방향 축이 없고 **단계**가 축이다 -- warn=경보(선행), bad=탐지(즉시).
  breakout_detector: { bad: "전환 발동", neutral: "미발동" },
  // 2026-09-15 E|r| 게이트. 축이 하나(발동 여부)다 — **방향 축이 없다**.
  evr_gate: { bad: "발동", neutral: "미발동" },
  // 2026-09-11 경보기(예고 모델). 축이 하나(warn/neutral)라 §5-4 문제 없음.
  breakout_prewarn: { warn: "전환 예고", neutral: "미발동" },
  // 2026-09-06: 배지 어휘와 같은 말을 쓴다 -- 띠에 커서를 올렸을 때와 배지가 다른 단어를 쓰면
  // 통일한 의미가 없다.
  // 2026-09-08: 라벨을 **모델의 주장**에 맞춘다(사용자 지적). V자는 되돌림(반전) 콜,
  // 앵커 방향은 지속 콜이다 -- 같은 바닥 앵커에서 하나는 롱, 하나는 숏이 나오는데
  // 기존 "롱 발동/숏 보유" 어휘로는 **왜 반대인지**가 화면에 없었다.
  // 2026-09-11 "되돌림" 폐기(사용자 지적). 라벨의 giveback(되돌림)은 **20% 이하로 억제돼야
  // 하는 조건**이라, 발동을 "되돌림"이라 부르면 같은 단어가 한 칩에서 정반대 두 뜻이 된다.
  // 칩 이름(V자 급등락)·툴팁과 같은 어휘로 통일한다.
  v_rebound: { good: "급등", bad: "급락", flat: "미발동", neutral: "데이터 없음" },
  // 2026-09-08: 라벨은 지속/되돌림 **이진**인데 러너가 지속 쪽만 진입해 화면에 지속만 떴다
  // (사용자 지적). 되돌림 우세·지속 약함도 상태로 노출한다 -- 둘 다 진입은 안 한다(회색).
  liq_pressure: { good: "롱압박↑", bad: "숏압박↑", neutral: "안정" },
  liq_cascade: { good: "안정", warn: "주의", bad: "위험" },
  // 2026-09-11 청산 방향압력 -> 청산 규모(사용자 지시). 색은 표시 규약 그대로 측면을 가리킨다.
  // 2026-09-11 청산 규모 -> 청산 위험. 방향 신호가 아니라 위험도 어휘(안정/주의/위험)를 쓴다.
  whale: { good: "롱 진입", bad: "숏 진입", neutral: "중립" },
  retail_flow: { good: "롱 진입", bad: "숏 진입", neutral: "중립" },
  evidence: { good: "바닥 발동", bad: "천장 발동", warn: "혼재 발동", neutral: "미발동" },
};

// data-fmt on .strip-time-now: "time" (model indicators) -> HH:MM:SS, "hm" (evidence signals,
// 2026-08-31 user request) -> HH:MM only, anything else -> MM-DD HH:MM fallback. Shared by
// showStripBarTime and lastSegmentRangeLabel below so both read the exact same format for a given
// row.
function stripTimeFmtByKind(kind) {
  return kind === "time" ? fmtTimeOnly : kind === "hm" ? fmtHourMinute : fmtShortTs;
}

// 2026-08-31 user request: default (non-hover) caption shows the LAST segment's own start~end time
// range + label (replaces the old plain "latest analysis time" default) -- "평소에는 마지막 칸의
// 시작과 끝 시간과 그 라벨의 정보를 보여주고 있어줘". Walks backward from the last bar while the
// tone stays the same, mirroring toneStripSvg's own segment-merge grouping (the still-forming
// provisional bar is always its own 1-wide segment there too, so no special-casing needed here --
// walking backward from index n-1 can only ever include OTHER already-confirmed bars).
// 2026-09-08: 띠 캡션의 라벨은 <주장> <방향> 두 축이다. 톤 사전이 방향(또는 단일 어휘)을 주고,
// 봉별 판정 단어(call)가 있으면 그 앞에 붙인다 -- "되돌림 ↓". 방향이 없는 톤(warn/neutral)은
// 그 자체가 완결된 상태어("혼재 보유"/"미발동")라 단어를 덧붙이지 않는다.
function stripBarLabel(key, tone, call) {
  const base = (STRIP_BAR_LABEL_BY_TONE[key] || {})[tone] || "";
  if (!base || !call || (tone !== "good" && tone !== "bad")) return base;
  return `${call} ${base}`;
}

function lastSegmentRangeLabel(tones, times, key, timeFmtKind, calls) {
  const list = Array.isArray(tones) ? tones : [];
  const timeList = Array.isArray(times) ? times : [];
  const callList = Array.isArray(calls) ? calls : [];
  const n = list.length;
  if (n === 0) return "-";
  const lastTone = list[n - 1] || "neutral";
  const lastCall = callList[n - 1] || "";
  let start = n - 1;
  // 2026-09-10: the rawFire stop is gone with toneStripSvg's (see there) -- this walk mirrors the
  // strip's grouping exactly, so caption and strip can never disagree about where a segment began.
  while (start > 0 && list[start - 1] === lastTone && (callList[start - 1] || "") === lastCall) start--;
  const fmt = stripTimeFmtByKind(timeFmtKind);
  const barLabel = stripBarLabel(key, lastTone, lastCall);
  const rangeText = start === n - 1 ? fmt(timeList[n - 1]) : `${fmt(timeList[start])}~${fmt(timeList[n - 1])}`;
  return barLabel ? `${barLabel} · ${rangeText}` : rangeText;
}

// .strip-time-now (rendered by each row template, NOT inside the strip) defaults to the last
// segment's own range+label (data-default, set once at render time via lastSegmentRangeLabel above)
// and switches to the HOVERED segment's own start~end range + label while the cursor is over the
// strip -- 2026-08-25 user request for the original time-only version ("지금 현재 시간을
// 표시해주고, 마우스를 올리면... 시간을 표시"), extended 2026-08-31 ("막대 한칸을 hover 하면 그 한
// 칸의 시작과 끝 시간과 그 라벨의 정보를 보여줘") to a full range+label on both hover and default.
// data-t/data-t-end on each <rect> (see toneStripSvg) are that segment's own start/end bar times.
function showStripBarTime(rectEl) {
  const startIso = rectEl.getAttribute("data-t");
  if (!startIso) return;
  const endIso = rectEl.getAttribute("data-t-end") || startIso;
  const label = rectEl.closest(".ops-health-info")?.querySelector(".strip-time-now");
  if (!label) return;
  const fmt = stripTimeFmtByKind(label.getAttribute("data-fmt"));
  const key = rectEl.closest("svg")?.getAttribute("data-key");
  const tone = rectEl.getAttribute("data-tone");
  const barLabel = key && tone ? stripBarLabel(key, tone, rectEl.getAttribute("data-call") || "") : null;
  const rangeText = startIso === endIso ? fmt(startIso) : `${fmt(startIso)}~${fmt(endIso)}`;
  label.textContent = barLabel ? `${barLabel} · ${rangeText}` : rangeText;
}

function hideStripBarTime(rectEl) {
  const label = rectEl.closest(".ops-health-info")?.querySelector(".strip-time-now");
  if (label) label.textContent = label.getAttribute("data-default") || "-";
}

// Same row/strip UI as renderEvidenceSignals(), but for the model-internal indicators -- and (as of
// 2026-08-30) reused a second time for the growing "특화 감지기" list of event-triggered detectors
// (see the two separate renderModelIndicatorList(items, targetId) call sites in render() below,
// each with its own target element id and its own memoized-html slot in lastModelIndicatorHtmlByTarget).
// All these panels LOOK identical on purpose (same ops-health-row/-strip markup) -- the caption on every
// row is what tells them apart, because the underlying history is NOT the same kind of window:
// evidence-signal strips are recomputed server-side from real historical klines (always full,
// survives a refresh); this list's strips are an in-memory tally that starts empty on page load
// and grows only while the tab stays open (trading_bot doesn't persist a time series for these
// fields, only the latest reading -- same limitation the Live tab's sparklines already had).
// Shared open/closed state for the per-signal "자세히" detail toggles (model indicators AND
// evidence signals use the same key space, prefixed "model:"/"evidence:" to avoid collisions).
// Kept outside any render function so it survives every re-render -- these lists are rebuilt via
// innerHTML replacement on every tick/poll, so without this a user's open detail panel would snap
// shut a few seconds after they opened it.
const detailOpenKeys = new Set();
function toggleSignalDetail(btn, key) {
  const row = btn.closest(".ops-health-row");
  const detail = row ? row.querySelector(".signal-detail") : null;
  const open = detailOpenKeys.has(key) ? (detailOpenKeys.delete(key), false) : (detailOpenKeys.add(key), true);
  if (detail) detail.classList.toggle("open", open);
  btn.textContent = open ? "접기 ▴" : "자세히 ▾";
  btn.setAttribute("aria-expanded", String(open));
}

// 진입/청산 미리보기 카드의 «자세히». 기존 toggleSignalDetail 은 .ops-health-row 안에서만
// 동작해서(closest) 이 카드에는 못 쓴다. 여기서는 버튼 **바로 다음 형제**를 연다.
// 열림 상태를 detailOpenKeys 에 남겨, 미리보기를 다시 띄워도 사용자의 선택이 유지된다.
function toggleEntryDetail(btn) {
  const detail = btn.nextElementSibling;
  const open = detailOpenKeys.has("entrycard")
    ? (detailOpenKeys.delete("entrycard"), false) : (detailOpenKeys.add("entrycard"), true);
  if (detail) detail.classList.toggle("open", open);
  btn.textContent = open ? "접기 ▴" : "자세히 ▾";
  btn.setAttribute("aria-expanded", String(open));
}

// Full-detail Korean explanations for the 6 model-internal indicators (formula + live threshold +
// what it means for a trader) -- shown only when the user clicks "자세히" next to each tile, so
// the default compact view stays uncluttered. Sourced from microstructure_scanner.py /
// tail_risk_interceptor.py verbatim, not re-derived.
// Always-visible "지금 이게 무슨 뜻인지" line, indexed by the exact subText string each
// indicator currently shows -- no click required (2026-08-24 사용자 요청: 발동되면 의미를 바로
// 볼 수 있게). The deeper formula/기준 stays behind "자세히" in MODEL_INDICATOR_DETAIL below.
const MODEL_INDICATOR_MEANING = {
  // 2026-09-14 변동성 예측. ⚠️키는 subText 문자열이다(규약 §5-1).
  // 🔴2026-09-15 정정: 이 칩은 **표시 전용**이다. 권고 수량을 실제로 정하는 건 MAE 분위 모델
  //   (`live_eth_mae_quantile_model_20260913`, 09-13 교체)이고 이 모델(`pred_vol`)은 그 모델이
  //   없을 때의 폴백(`vol_equivalent_qty`)으로 밀렸다. 칩 문구가 «수량을 정한다»고 읽히면 안 된다.
  vol_level: {
    "안정": "앞으로 4시간 예상 변동폭이 **최근 30일 평소 이하**입니다(그 분포의 하위 80%, 평소의 1.39배 미만). 조용하다는 **눈금**이지 주문 수량이 아닙니다 — 권고 수량은 MAE 분위 모델이 정합니다(이 칩은 그 모델이 없을 때만 대신 씁니다).",
    "주의": "앞으로 4시간 예상 변동폭이 **최근 30일 평소보다 뚜렷이 큽니다**(평소의 1.39~1.88배, 상위 20~5%). 같은 위험을 지려면 수량을 줄여야 하는 국면이라는 **눈금**입니다 — 실제 감축은 **MAE 분위 모델**이 합니다(E|r| 배수는 2026-09-15 켜기 전에 철회됐습니다 — 순차 검정 후반 36건 +0.73bp·Δ>0 47.2%로 동전이고, 상위 10건을 빼면 부호가 뒤집혔습니다).",
    "위험": "앞으로 4시간 예상 변동폭이 **최근 30일 평소의 1.9배 이상**입니다(상위 5%). 방향 경고가 아닙니다 — 이 모델은 방향을 예측하지 않습니다.",
    "웜업": "사이징 워커가 아직 첫 예측을 내지 않았습니다.",
    "데이터 없음": "사이징 워커 상태파일에서 예측값을 읽지 못했습니다.",
  },
  // 2026-09-09 극점 탐지기. ⚠️키는 subText 문자열이다(규약 §5-1).
  breakout_prewarn: {
    "전환 예고": "앞으로 30분 안에 전환이 시작될 확률이 상위 10%에 들었어요 — 새 진입을 미룰 구간입니다.",
    "미발동": "앞으로 30분 안에 전환이 올 확률이 평소 수준이에요.",
    "웜업": "예고 모델이 아직 첫 계산을 끝내지 않았어요.",
    "데이터 없음": "예고 모델이 값을 내지 못하고 있어요.",
    "오류": "시세를 읽지 못해 이번 봉을 채점하지 못했어요.",
  },
  breakout_detector: {
    "전환 발동": "추세 전환이 방금 시작됐습니다 — 반대 방향 포지션이면 청산을 먼저 보세요. 방향은 말하지 않습니다.",
    "미발동": "체결속도·거래대금이 아직 평소 수준이에요.",
    "웜업": "전환 탐지 워커가 아직 첫 계산을 끝내지 않았어요.",
    "데이터 없음": "전환 탐지 워커가 값을 내지 못하고 있어요.",
    "오류": "시세를 읽지 못해 이번 봉을 채점하지 못했어요.",
  },
  // ⚠️키는 subText 문자열이다(규약 §5-1) -- 라벨을 바꾸면 여기도 같이 바꾼다.
};

const MODEL_INDICATOR_DETAIL = {
  vol_level:
    "**앞으로 4시간(48봉) 실현변동성**을 예측합니다. 방향도 수익도 예측하지 않습니다.\n\n"
    + "화면은 두 숫자를 같이 냅니다 — **배수**(예측 ÷ 직전 4시간)와 **확대 확률**(앞 4시간이 "
    + "직전 4시간의 1.3배 이상일 확률). 그 아래 «평소 대비 N배(등급)»는 다른 축입니다"
    + "(최근 30일 중앙값 대비). 수량 배수는 또 다른 기준(학습창 고정)입니다 — 셋 다 분모가 다릅니다.\n\n"
    + "모델은 HGB 8시드 앙상블, 입력은 공개 kline 22열입니다. 학습은 2025-08-31 까지이고 그 뒤는 표본외입니다.\n\n"
    + "⭐**2026-09-21 정정 — 이 카드는 그전까지 「곧 커진다는 못 맞힙니다(AUC .46~.52 = 동전)」라고 "
    + "적고 있었는데 그건 채점 오류였습니다.** 확장은 «비율» 질문인데 모델의 «수준»으로 채점했습니다"
    + "(수준은 스케일을 갖고 비율은 안 갖습니다). 올바른 점수인 예측÷직전으로 재면 표본외 109,267행에서 "
    + "**AUC 0.8237**, rv48 십분위를 통제해도 **0.7865**(10/10 셀 0.69 이상 · 분기 5/5 0.729~0.791 · "
    + "일블록 부트스트랩 CI [0.754, 0.822]로 0.5 배제). 축소 쪽도 대칭입니다(0.8015).\n\n"
    + "크기도 편향이 없습니다: log(실제배수) = −0.0091 + **1.0365**×log(예측배수). 예측 십분위별 "
    + "실제 중앙값이 0.55 → 1.78 로 단조이고, P(1.3배 이상)이 2.3% → 82.5% 로 갈립니다(기저 26.8%).\n\n"
    + "🔴**점으로 읽지 마세요.** 잔차 SD 0.34라 예측 1.5배일 때 실제는 68% 확률로 1.07~2.12배입니다. "
    + "불확실성은 27%만 줄어듭니다.\n\n"
    + "변동성은 4시간 안에 실제로 많이 움직입니다 — 1.3배 이상 변하는 경우가 56.5%(확대 26.8% + "
    + "축소 29.7%), 2배 이상이 13.5%입니다. 평평한 대상이 아닙니다.\n\n"
    + "쓰는 자리는 **수량**(배포 공식 = 기준수량 × 기준예측 ÷ 현재예측)과 **손절폭**입니다. "
    + "«진입할지 말지»의 하드 차단은 이 저장소에서 이미 졌습니다(E|r| 게이트, 실계좌 72왕복).\n\n"
    + "등급 경계는 평소 대비 배수입니다 — 1.39배 미만 안정, 1.88배 이상 위험(기준 = 최근 30일 중앙값).",
  breakout_detector:
    "변동성이 추세로 넘어가는 **시점**만 잡습니다. 2026-09-11 압축 게이트를 제거해 «횡보를 거친» "
    + "전환뿐 아니라 **모든** 전환을 봅니다 -- 실제로 전환의 77%는 압축을 거치지 않고 일어납니다. "
    + "방향은 예측하지 않습니다 -- 이 저장소에서 "
    + "방향 축은 닫혔습니다(MFE-MAE 상관 +0.000). 입력은 공개 kline 의 체결 건수(n)와 거래대금(qv) "
    + "뿐이고, 호가창은 기여가 1/6 수준이라 뺐습니다.\n\n"
    + "[경보 = 신호등 3개] 합치지 않고 각자 켜집니다. 지평이 달라 뜻이 다르기 때문입니다. "
    + "체결속도 3봉지속(z2016 q99) → 앞 2시간 lift 5.75x · 체결속도(z864 q99) → 앞 5~15분 3.65x · "
    + "거래대금(z2016 q99) → 앞 30분 3.06x. 전반 8.56x / 후반 8.58x 로 갈리지 않고, 63셀 전부 2배 "
    + "이상이었습니다(셀 순위 상관 +0.708).\n\n"
    + "[탐지 = 2종 AND] 거래대금·체결속도 z288 q90 이 **둘 다** 넘을 때만. 2026-09-11 압축 게이트를 "
    + "제거했습니다 -- 큰 이동(앞 24봉 최대이탈 상위 5%) 기준 사건 포착이 47.9/50.5/52.8% 에서 "
    + "82.3/82.3/82.9% 로 오르고 발동은 33.9 → 25.1회/일 로 **줄었습니다**. 더 많이 잡으면서 덜 "
    + "울립니다. volexp 자체는 뺐습니다 -- 전환의 정의에 쓰인 양이라 기여가 정의상 보장된 것입니다.\n\n"
    + "[옛 수치 정정] 이 카드가 쓰던 «포착률 98.4% · 지연 5분 · 진행률 6.08%» 는 volexp 1.80 교차의 "
    + "포착률이지 «큰 이동»의 포착률이 아니었습니다.\n\n"
    + "[순서] 경보→탐지→전환이 실제로 그 순서로 431건 중 143건(33.2%). 순환이동 귀무는 20.0%"
    + "(q95 25.8%), p=0.000 입니다.\n\n"
    + "[전환 뒤] 같은 방향 지속 77.2%(1시간)→62.9%(1일), 큰 반전 29.2%(1일). "
    + "⚠️그래서 **반대 포지션은 청산까지**입니다. 뒤집어서 따라가라는 근거는 없습니다.\n\n"
    + "[전환 포착 실측] volexp 1.80 상향교차를 사건으로 두고 1시간 창에서 잰 값입니다. "
    + "**모든 전환**: 옛 판 22.9/26.7/32.2% → 새 판 **89.6/92.0/92.2%**. "
    + "**압축을 거친 전환만**(옛 정의, 전체의 23%): 97.9/97.8/98.9% → 89.4/91.3/93.5%. "
    + "좁은 정의에서 8pp 를 내주고 넓은 정의에서 65pp 를 얻었습니다. 발동은 33.9 → 25.1회/일.\n\n"
    + "⚠️매매 트리거가 아닙니다. **탐지** 임계는 전 봉 후행 2016봉 분위, **경보** 임계는 압축 봉만 "
    + "모아 낸 분위입니다(둘이 다릅니다). 전역 분위를 쓰면 미래참조입니다. "
    + "표본은 ETH 5분봉 493,650개(2021-12~2026-09)이고 전환은 창별 201/176/295건입니다.",
};

// 2026-08-30 (user request): "학습 horizon을 배지로" -- each signal's own validated forward-
// looking prediction/detection window, shown as a small badge next to its name (see
// horizonBadgeHtml() below, used by both renderModelIndicatorList and renderEvidenceSignals).
// Covers the model-indicator keys in one lookup.
// (2026-09-16 증거신호 칩이 내려가면서 EVIDENCE_STRIP_CHIP_IDS 는 사라졌다 -- 이 주석이
//  없는 상수를 계속 가리키고 있었다. 2026-09-20 정정.)
// "상태" (not a number) marks signals whose live formula is a continuous current-state gauge with
// no fixed forward horizon baked in -- forcing a number onto those would overstate what they
// actually claim; each entry's title cites the specific research this is grounded in (verified
// against each script's own docstring/detail text above and this repo's evidence-signal scorecard
// methodology, not guessed from the signal's name -- e.g. "15분 급변"/short_term_return_z names
// its INPUT lookback, not its evaluation horizon, which is 1시간 like its 6 scorecard siblings).
const SIGNAL_HORIZON = {
  // -- model indicators --
  breakout_detector: { text: "탐지 = 즉시", title: "예측이 아니라 즉시 인지다 -- 거래대금·체결속도 z288 이 **둘 다** 후행 q90 을 넘으면 켜진다. 2026-09-11 압축 게이트를 제거해 전환 사건 포착이 26.7% -> 92.0%(OOS, 176건 중 162건)로 오르고 발동은 33.9 -> 25.1회/일 로 줄었다. 발동 2,085회 중 813회가 사건 창 안이다(정밀도 39.0%)." },
  // 2026-09-11 경보기(예고 모델). 옛 «경보 2시간» 은 자명한 대리 타깃 값이라 교체했다.
  breakout_prewarn: { text: "예고 = 30분", title: "앞으로 30분 이내에 **탐지기가 발동할** 확률이다(HGB 5시드 동결 앙상블, 33피쳐). 커버리지 10%(하루 약 28.8회)에서 표본외 정밀도 78.5%, 기저 22.9% -- lift 3.43x. 임계는 확률의 후행 2016봉 분위 q90 이라 인과적이다. ⚠️«탐지기가 켜진다»이지 «큰 이동이 온다»가 아니다 -- 탐지기 자체 정밀도가 39.0% 라 그 위로 못 간다." },
};

// 2026-09-06: `progress`(예 "26/30봉")를 주면 배지 문구에 덧붙인다. 새 배지를 만들지 않는 이유는
// 제목줄이 이미 최대 3개 배지를 달고 있고 이 열은 과거에 오버플로 사고가 있었기 때문이다
// (eth_dashboard_low_atr_warning_overflow_fix_20260901). 인자를 안 주면 기존 동작 그대로다.
function horizonBadgeHtml(key, progress, extraTitle) {
  const h = SIGNAL_HORIZON[key];
  if (!h) return "";
  const text = progress ? `${h.text} · ${progress}` : h.text;
  let title = progress ? `${h.title}\n\n지금 이 발동은 그 창의 ${progress}째입니다.` : h.title;
  if (extraTitle) title += `\n\n${extraTitle}`;
  return ` <span class="horizon-badge" title="${escapeHtml(title)}">${escapeHtml(text)}</span>`;
}

// ⚠️2026-09-03: 스냅샷 탭은 코인을 전환하는데, 아래 지표 중 일부는 **ETH 전용 출처**다:
//   · whale / retail_flow / liq_cascade -- trading_bot.py의 dashboard_state(봇은 ETH만 돌린다)
//   · v_rebound                          -- ETH 전용 TabPFN 모델(/api/v-rebound-signal)
// 자산 게이트가 없어서 XRP/BTC 탭에서도 **ETH 값이 그대로** 보이고 있었다. 사용자가 이미
// 신고했던 "비트코인 페이지에 이더리움 증거신호가 나온다"와 같은 계열의 버그다.
// 값을 지우고 "ETH 전용" 상태로 바꾼다 -- 다른 코인의 값인 척하는 것보다 없는 게 낫다.
function ethOnlyIndicator(item) {
  if (activeSnapshotAsset === "eth") return item;
  return { ...item, tone: "neutral", proba: null, history: [], times: [], callHistory: [],
           subText: "미지원",   // 2026-09-06: 상태 열은 92px nowrap이라 문장이 들어가면 넘친다. 설명은 derivedTitle에 있다.
           derivedTag: "= ETH 전용",
           derivedTitle: "이 지표의 데이터 출처가 ETH 전용입니다(봇 상태 또는 ETH 학습 모델). "
             + "다른 코인 탭에서는 값을 숨깁니다 -- ETH 값을 그 코인 값인 것처럼 보여주지 않기 위해서입니다." };
}

// ⭐2026-09-03: ETH가 아닌 코인은 **그 코인 자신의** 실시간 수집기 값으로 대체한다.
// 데이터가 없으면 ethOnlyIndicator로 폴백해 "미지원"으로 정직하게 표시한다.
//
// ⚠️표시를 ETH와 **똑같이** 맞추는 게 중요하다. 첫 판에서는 subText에 숫자를 직접 넣었는데
// (`유입 0.552 · 방금`), ETH는 `directionalCaution()`이 만드는 **"롱 진입"/"숏 진입"/"중립"**
// 어휘를 쓰고 `MODEL_INDICATOR_MEANING[key][subText]`가 **그 문자열로 조회**된다. 자유 문구를
// 넣으면 (a) 칩 문구가 ETH와 달라 보이고 (b) 의미 설명이 사전에 안 걸려 사라진다.
// 그래서 subText/valueText/톤스트립을 전부 ETH와 같은 함수·같은 모양으로 만든다.
// 숫자와 경과시간은 `liveText`(ETH의 청산캐스케이드가 쓰는 자리)와 툴팁으로 뺀다.

// 2026-09-06 (사용자 신고 "V자 급등락의 미발동만 흰색"): 서버가 주는 V자 톤 **"flat"**(방향 없음)은
// CSS에 대응 규칙이 없다(.meter-state / .signal-chip / .ops-health-row 전부 good·bad·warn·neutral 뿐).
// 클래스는 붙지만 색 규칙이 없어 글자색이 상속돼 흰색으로 떴다 -- 같은 뜻인 다른 감지기의 `미발동`은
// 전부 회색이다. 스트립 SVG는 이미 flat 을 neutral 과 같은 회색으로 칠하고 있었으니(toneStripSvg 의
// fill 폴백) 배지·행·칩만 법칙에서 벗어나 있었다. 여기서 한 번에 정규화한다.
function toneClass(tone) { return tone === "flat" ? "neutral" : (tone || "neutral"); }

// 2026-09-14: 등급(안정/주의/위험)만으로는 «얼마나»를 못 본다. 실시간 배수를 툴팁에 싣는다 --
// 상태 열은 92px nowrap 이라 문장을 못 넣는다(규약 §3).
function volLevelTitle(v) {
  // 🔴2026-09-16 정정: "이미 수량 공식에 쓰이고 있다"는 **틀린 문장**이었다. 요청 경로는 MAE
  //   위험 모델로 수량을 내고(rec_src="risk_model"), 이 변동성 공식은 그게 없을 때의 폴백이다
  //   (rec_src="vol_equivalent"). 화면이 코드보다 강하게 말하고 있었다.
  const base = "봇 내부 상태가 아니라 배포된 사이징 변동성 모델의 예측 -- 권고 수량은 MAE 위험 모델이 내고, 이 변동성 공식(수량 = 기준수량 x 기준예측/현재예측)은 그 모델이 없을 때의 폴백이다.";
  if (!v || !v.available || !Number.isFinite(v.ratio)) return base;
  // 2026-09-15: 등급의 「평소」는 **최근 30일**이고 수량 배수의 기준은 **학습창 고정**이다.
  // 두 숫자가 다른 기준을 쓰므로 말도 다르게 한다(고정으로 떨어졌으면 그대로 말한다).
  const norm = v.ref_source === "recent" ? `최근 ${v.ref_days || 30}일` : "학습창(기준 갱신 실패)";
  return `${base} 지금 예상 변동폭은 ${norm} 평소의 ${v.ratio.toFixed(2)}배, 같은 위험 기준 수량 배수는 ${v.qty_mult.toFixed(2)}배(이쪽 기준은 학습창 고정).`;
}

// 2026-09-21 ⭐**배수 + 확률**. 화면이 오래 «수준»만 띄우면서 이 모델의 가장 좋은 출력을 버리고
// 있었다 -- 「확장은 못 맞힌다(AUC .46~.52)」는 **채점 오류**였다(비율 질문을 수준으로 채점).
// pred/rv48 로 재면 OOS AUC 0.8237, rv48 십분위 통제 후 0.7865(10/10 셀 · 분기 5/5 ·
// 일블록 CI [0.754,0.822]). 보정 기울기 1.0365 라 크기도 편향이 없다.
// 🔴점으로 읽히면 안 되므로 **배수와 확률을 항상 같이** 낸다(예측 1.5배의 실제 68% 구간이
// [1.07,2.12]배다). 그래서 subText 가 "1.36배 · 확대 56%" 형태다.
function volLevelIndicatorItem() {
  const v = latestVolLevel;
  const base = {
    key: "vol_level", label: "변동성 수준 (4시간)",
    history: toneHistory.vol_level, times: toneHistoryTimes.vol_level,
    derivedTag: "= 사이징 모델", derivedTitle: volLevelTitle(v),
  };
  if (!v || !v.available) return { ...base, tone: "neutral", subText: (v && v.grade) || "웜업" };
  const hasExp = Number.isFinite(v.mult) && Number.isFinite(v.p_expand);
  if (!hasExp) return { ...base, tone: v.tone || "neutral", subText: v.grade || "웜업" };
  return {
    ...base,
    tone: v.tone || "neutral",
    // 등급(평소 대비)보다 **확장 읽기**를 앞에 둔다 -- 그쪽이 통제 검정을 통과한 축이다.
    subText: `${v.mult.toFixed(2)}배 · 확대 ${Math.round(v.p_expand * 100)}%`,
    liveText: `평소 대비 ${v.ratio.toFixed(2)}배 (${v.grade}) · 수량 배수 ${v.qty_mult.toFixed(2)}배`,
    probaSlot: true, proba: v.p_expand, meterNote: "확대 확률",
    meterNoteTitle: `«앞으로 4시간 실현변동성이 직전 4시간의 ${v.expand_k || 1.3}배 이상일 확률».`
      + ` 배수(예측÷직전) ${v.mult.toFixed(2)} 를 TRAIN 적합 로지스틱으로 옮긴 값이다`
      + ` -- OOS 109,267행 AUC 0.8237 · Brier 0.1388(상수 0.1961) · 십분위 보정오차 ≤4pp.`
      + ` 🔴점이 아니라 중심값이다: 예측 1.5배일 때 실제는 68% 확률로 1.07~2.12배다.`
      + ` rv48 십분위를 통제해도 AUC 0.7865(10/10 셀 0.69+ · 분기 5/5 · 일블록 CI [0.754,0.822]).`,
  };
}

function renderModelIndicatorList(items, targetId = "snapModelIndicatorList", { forceMeter = false } = {}) {
  // 2026-08-25: perf pass -- render() drives this on every SSE push (~2.5s), but the underlying
  // model_indicator_history only advances once per MODEL_INDICATOR_SAMPLE_SECONDS (300s server-
  // side), so most calls were rebuilding ~500 DOM nodes (9 rows x up to 48 sparkline rects each)
  // for identical output. Chip-element side effects below still run every call (cheap, and their
  // inputs -- it.tone/it.subText -- are also embedded in the html string, so if the string is
  // unchanged those writes are redundant-but-harmless); only the expensive setH() innerHTML
  // rebuild is skipped when nothing actually changed.
  const html = items.map((it) => {
    const tone = toneClass(it.tone);                 // flat -> neutral (위 주석)
    const derivedTag = it.derivedTag
      ? ` <span class="derived-tag" title="${escapeHtml(it.derivedTitle || "")}">${escapeHtml(it.derivedTag)}</span>`
      : "";
    const detailKey = `model:${it.key}`;
    const isOpen = detailOpenKeys.has(detailKey);
    const detailText = MODEL_INDICATOR_DETAIL[it.key] || "";
    const meaningText = (MODEL_INDICATOR_MEANING[it.key] || {})[it.subText] || "";
    // 2026-09-16 사용자 요청 «설명 칸 2개를 1개로». 뜻(고정)과 지금 숫자(liveText)가 같은 `.signal-meaning`
    // 문단으로 **연달아 두 개** 그려지고 있었다 -- 같은 칸이 두 줄로 쪼개져 보인다. 한 문단으로 합친다.
    // liveText 를 쓰는 다른 칩(수급흐름·청산캐스케이드)도 같은 규칙을 따른다 -- 칸 모양은 하나여야 한다.
    const meaningLine = [meaningText, it.liveText].filter(Boolean).join(" · ");
    const times = it.times || [];
    // 2026-08-31 user request: default caption shows the LAST segment's own range+label, not just
    // "지금 시간" -- see lastSegmentRangeLabel().
    const defaultRangeText = lastSegmentRangeLabel(it.history, times, it.key, "time", it.callHistory);
    // 2026-08-31: optional `it.proba` (0-1) opts an item into the same inline probability meter
    // renderEvidenceSignals() uses (see .meter-col in styles.css) -- state text, then the meter bar,
    // stacked vertically ("천장 발동과 익절 사이" layout the user picked). Items with no proba concept
    // (수급 흐름/청산 캐스케이드/베이시스 청산압박/청산 방향압력, all categorical-only) keep the plain
    // .ops-health-status-badge pill, unchanged -- there's no probability to gauge for those.
    // 2026-09-06 (사용자 요청 "스타일까지 통일"): forceMeter 목록(특화감지기)은 증거신호 목록과
    // 같은 규칙 -- **상태와 무관하게 항상 .meter-col**을 그린다. 그 전에는 확률이 있을 때만
    // meter-col이고 없으면 64px 알약으로 떨어져, 같은 행이 상태에 따라 모양이 바뀌고 세 행이
    // 서로 다른 모양이었다(증거신호 목록이 2026-08-31에 같은 이유로 이미 고쳐진 문제다).
    // 게이지 줄은 **확률 개념이 있는 행만**(probaSlot) -- 확률이 아닌 값을 % 게이지로 그리면
    // 다른 뜻이 같은 그림으로 보인다. 확률이 아닌 수치는 meter-price 자리(meterNote)에 적는다.
    const gaugeHtml = (it.probaSlot || it.proba != null)
      ? `<div class="meter-gauge">
          <span class="meter-track"><span class="meter-fill ${tone}" style="width:${clamp01(it.proba || 0) * 100}%"></span></span>
          <span class="meter-pct">${it.proba != null ? `${Math.round(clamp01(it.proba) * 100)}%` : "-"}</span>
        </div>`
      : "";
    const metaHtml = (forceMeter || it.proba != null)
      ? `<div class="meter-col">
          <span class="meter-state ${tone}"${it.stateTitle ? ` title="${escapeHtml(plainEmphasis(it.stateTitle))}"` : ""}>${escapeHtml(it.subText || "-")}</span>
          ${gaugeHtml}
          ${it.meterNote ? `<span class="meter-price" title="${escapeHtml(it.meterNoteTitle || "")}">${escapeHtml(it.meterNote)}</span>` : ""}
        </div>`
      : `<span class="ops-health-status-badge">${escapeHtml(it.subText || "-")}</span>`;
    return `<article class="ops-health-row ${tone}">
      <span class="ops-health-dot" aria-hidden="true"></span>
      <div class="ops-health-info">
        <strong>${escapeHtml(it.label)}${horizonBadgeHtml(it.key)}${derivedTag}</strong>
        ${meaningLine ? `<p class="signal-meaning"${it.liveTitle ? ` title="${escapeHtml(plainEmphasis(it.liveTitle))}"` : ""}>${emphasizeHtml(meaningLine)}</p>` : ""}
        <div class="evidence-strip-wrap">
          ${toneStripSvg(it.history, times, false, false, it.key, it.callHistory)}
          ${stripAxisHtml(times, "time")}
        </div>
        <div class="strip-time-row">
          <button type="button" class="detail-toggle" aria-expanded="${isOpen}" onclick="toggleSignalDetail(this, '${detailKey}')">${isOpen ? "접기 ▴" : "자세히 ▾"}</button>
          <span class="strip-time-now" data-fmt="time" data-default="${escapeHtml(defaultRangeText)}">${escapeHtml(defaultRangeText)}</span>
        </div>
        <div class="signal-detail${isOpen ? " open" : ""}">${emphasizeHtml(detailText)}</div>
      </div>
      <div class="ops-health-meta">
        ${metaHtml}
      </div>
    </article>`;
  }).join("");
  if (html === lastModelIndicatorHtmlByTarget[targetId]) return;
  lastModelIndicatorHtmlByTarget[targetId] = html;
  setH(targetId, html);
}

// One row of the liquidation-map price ladder -- tag+price+density bar+distance, color-coded by
// side. Bar width floors at 4% so even a low-weight surviving level stays visible (a 0%-wide bar
// would look broken/missing rather than "weak").
function liquidationLevelRowHtml(lv, tag, sideClass) {
  const pct = Math.round((lv.weight_pct || 0) * 100);
  const dist = Number(lv.distance_pct);
  const distText = Number.isFinite(dist) ? `${dist > 0 ? "+" : ""}${fmtNum(dist, 2)}%` : "-";
  return `<div class="liq-level-row ${sideClass}">
      <span class="liq-level-tag">${escapeHtml(tag)}</span>
      <span class="liq-level-price">${fmtNum(lv.price, 2)}</span>
      <div class="liq-level-bar-track"><div class="liq-level-bar-fill" style="width:${Math.max(pct, 4)}%;"></div></div>
      <span class="liq-level-dist">${distText}</span>
    </div>`;
}

// Renders the Snapshot tab's liquidation-map list as a price ladder: resistance levels (farthest
// first) above a highlighted current-price row, support levels (nearest first) below -- reads
// top-to-bottom the same way the chart overlay's lines sit above/below current price.
//
// 2026-08-25: backend switched from the event-driven state machine to a fixed rolling recompute
// (compute_liquidation_levels(), currently 24h -- see server.py's load_liquidation_map() comment
// for why, after trying 48h and 168h too); map.support_levels/resistance_levels are the same
// field names either way, so this function's own logic is unchanged, only the badge text below
// (no more per-side reset "staleness" -- a rolling window recomputes fresh every cache cycle, so
// bars_used/lookback_hours are the only freshness numbers left to show). Live-price re-filter
// stays: this list reads the same backend snapshot (map.current_price, up to ~5min stale -- the
// server cache interval) as the chart overlay, so an already-crossed level still needs dropping
// client-side between refreshes.
function renderLiquidationMapPanel() {
  const map = latestLiquidationMap;
  const badge = el("liqMapBadge");
  // 2026-09-06 에는 배지가 비면 **헤더째 접었다**(제목도 없이 배지 하나뿐이라 37px 빈 줄이
  // 남았다). 2026-09-16 그 전제가 사라졌다 -- 같은 헤더에 차트 종류 토글이 상주한다. 접으면
  // 그 토글이 같이 사라져서 **버튼이 아예 안 눌린다**(브라우저 시험에서 "element is not
  // visible" 로 잡혔다). 헤더는 이제 항상 내용이 있으므로 접지 않는다.
  const setMapBadge = (tone, text) => {
    if (!badge) return;
    // 🔴바뀐 것만 쓴다. 이 함수는 현재가 틱마다(≈1.7Hz) 불리는데 정상 구간에서는 늘 같은
    //   값이라, 예전엔 40초에 51번을 **아무 변화 없이** 다시 썼다(브라우저 실측).
    const cls = `ops-badge ${tone}`;
    if (badge.className !== cls) badge.className = cls;
    if (badge.textContent !== text) badge.textContent = text;
    if (badge.hidden !== !text) badge.hidden = !text;   // 빈 배지는 자리만 먹는다
  };
  if (!map || map.error === "fetch_failed") {
    setMapBadge("bad", "연결 실패");
    setH("liquidationMapList", `<p class="muted" style="padding:16px;">청산맵 데이터를 불러오지 못했습니다.</p>`);
    return;
  }
  if (!map.warmed_up) {
    setMapBadge("neutral", "웜업");
    setH("liquidationMapList", `<p class="muted" style="padding:16px;">데이터 수집 중...</p>`);
    return;
  }
  setMapBadge("neutral", "");

  const liveCurrentPrice = Number(latestLivePriceByAsset[activeSnapshotAsset] || map.current_price || 0);
  const liveRedistanced = (levels, side) => {
    if (!(liveCurrentPrice > 0)) return levels || [];
    return (levels || [])
      .filter((lv) => side === "support" ? lv.price < liveCurrentPrice : lv.price > liveCurrentPrice)
      .map((lv) => ({ ...lv, distance_pct: (lv.price - liveCurrentPrice) / liveCurrentPrice * 100 }));
  };
  // 2026-09-11 사용자 요청: 각 측면 **3개**만. 원 배열은 현재가에서 가까운 순이므로
  // 자르고 나서 뒤집는다(저항은 먼 것이 위, 가까운 것이 현재가 줄 바로 위로 온다).
  const SR_ROWS = 3;
  const resistanceRows = liveRedistanced(map.resistance_levels, "resistance").slice(0, SR_ROWS).reverse()
    .map((lv, i, arr) => liquidationLevelRowHtml(lv, `저항${arr.length - i}`, "liq-resistance"));
  const supportRows = liveRedistanced(map.support_levels, "support").slice(0, SR_ROWS)
    .map((lv, i) => liquidationLevelRowHtml(lv, `지지${i + 1}`, "liq-support"));
  const currentRow = `<div class="liq-level-row liq-current">
      <span class="liq-level-tag">현재가</span>
      <span class="liq-level-price">${fmtNum(liveCurrentPrice || map.current_price, 2)}</span>
      <div class="liq-level-bar-track"></div>
      <span class="liq-level-dist">-</span>
    </div>`;
  const rows = [...resistanceRows, currentRow, ...supportRows];
  setH("liquidationMapList", rows.length ? rows.join("") : `<p class="muted" style="padding:16px;">추정 가능한 밀집 구간이 아직 없습니다.</p>`);
}

// 2026-08-25 실측(VAL+OOS 48,853봉): 같은 쪽 신호가 동시에 몇 개 뜨는지(bottom_votes/top_votes)
// 자체가 검증된 신뢰도 축 -- votes>=N lift가 N에 대해 단조증가함을 확인(N>4는 미검증, 4로 캡).
// scripts/research_eth_evidence_signal_indicator_cooking_research_20260825.md 참고.
const VOTE_LIFT_BY_SIDE = {
  bottom: { 1: 1.81, 2: 2.10, 3: 2.32, 4: 2.72 },
  top: { 1: 1.58, 2: 1.85, 3: 1.89, 4: 2.07 },
};
// 🔴2026-09-10 정정: 위 lift 는 **반전 사건이 일어나는가(분류)** 기준이고 단조증가가 맞다.
// 그러나 **손익 기준으로는 반대다.** 순환이동 귀무(발동 군집·개수를 보존한 채 가격 정렬만 파괴)
// 대비 초과수익을 815일에서 재면 겹칠수록 좋아지지 않는다:
//     바닥  1종 +0.20 / 2종 +1.04 / 3종+ +0.60 bp (H=1시간),  H=4시간에서는 3종+ 가 **-5.91**
//     천장  1종 +0.21 / 2종 -0.56 / 3종+ -2.28 bp,            H=4시간 3종+ **-5.99**
// 즉 3종 이상 동시발동은 두 측면 모두에서 가장 나쁘다. 화면이 "겹칠수록 신뢰도가 높아진다"고만
// 쓰면 사용자가 그걸 진입 근거로 읽는다 -- 그래서 두 축을 문장에서 분리한다.
// scripts/research_eth_signal_confluence_null_20260910.py
const VOTE_ECON_BY_SIDE = {   // 동시발동 개수별 귀무 대비 초과 bp (H=1시간 / H=4시간)
  bottom: { 1: [0.20, 1.96], 2: [1.04, 1.29], 3: [0.60, -5.91], 4: [0.60, -5.91] },
  top: { 1: [0.21, -2.78], 2: [-0.56, 1.07], 3: [-2.28, -5.99], 4: [-2.28, -5.99] },
};
function voteLiftNote(side, votes) {
  const capped = Math.min(Math.max(Math.round(votes), 1), 4);
  const lift = VOTE_LIFT_BY_SIDE[side][capped];
  const [e1, e4] = VOTE_ECON_BY_SIDE[side][capped];
  const sideKo = side === "bottom" ? "바닥" : "천장";
  return `실측: ${sideKo} 신호 ${capped}개↑ 동시발동 구간 lift ${lift.toFixed(2)}배(무작위 대비) — `
    + `이건 **반전 사건이 일어나는가(분류)** 기준입니다. `
    + `⚠️손익은 다릅니다: 같은 구간의 귀무 대비 초과수익은 ${e1 >= 0 ? "+" : ""}${e1.toFixed(2)}bp/건`
    + `(H=1시간), ${e4 >= 0 ? "+" : ""}${e4.toFixed(2)}bp(H=4시간)이고 왕복비용은 10bp입니다. `
    + `겹칠수록 좋아지지도 않습니다 — 3종 이상 동시발동이 두 측면 모두에서 가장 나쁩니다.`;
}


// 2026-08-27: split off /api/evidence-signals (see api_session_alerts() docstring in server.py) --
// user reported the badges only updated on a manual page reload, root cause was inheriting
// evidence-signals' 5min client poll. This fetch is independent and fast (30s).
async function refreshSessionAlerts() {
  const now = Date.now();
  if (now - sessionAlertsLastFetchAt < SESSION_ALERTS_POLL_MS) return;
  sessionAlertsLastFetchAt = now;
  try {
    const res = await fetch(API_SESSION_ALERTS_URL, { cache: "no-cache" });
    if (!res.ok) throw new Error(`session alerts ${res.status}`);
    const data = await res.json();
    renderSessionVolatilityAlert(data.session_volatility_alert);
    renderMacroEventAlert(data.macro_event_alert);
  } catch (error) {
    console.error("Session alerts fetch error:", error);
    const alertBadge = el("sessionVolAlertBadge");
    if (alertBadge) alertBadge.style.display = "none";
    const macroAlertBadge = el("macroEventAlertBadge");
    if (macroAlertBadge) macroAlertBadge.style.display = "none";
  }
}

// Session-open volatility risk alert (2026-08-26), centered on the Snapshot tab's top line (same
// row as the EVIDENCE LIVE badge, just below the header clock) -- see scripts/live_session_
// volatility_alert_20260826.py's docstring for the empirical windows (NYSE +-60min real effect;
// LSE/JPX 0..+30min only, marginal effect). Fixed label text by design (user request) -- the
// per-market/minutes detail goes in the title tooltip only, not the visible badge.
function renderSessionVolatilityAlert(alertPayload) {
  const badge = el("sessionVolAlertBadge");
  if (!badge) return;
  const active = alertPayload && Array.isArray(alertPayload.active) ? alertPayload.active : [];
  if (!active.length) { badge.style.display = "none"; return; }
  const a = active[0];
  const when = a.minutes_from_open < 0
    ? `개장 ${Math.round(Math.abs(a.minutes_from_open))}분 전`
    : a.minutes_from_open === 0 ? "개장 순간" : `개장 ${Math.round(a.minutes_from_open)}분 후`;
  badge.style.display = "";
  badge.title = `${a.label} ${when} — 실측(2026-08-26): 미국장 ±60분은 ETH 변동성 평소 대비 1.5~2.3배, 유럽/일본 개장 후 30분은 효과가 약함(참고용, 매매룰 아님)`;
}

// Macro-event (CPI/NFP/GDP/PCE/내구재/FOMC/연준 의장 발언) release-time alert (2026-08-26 follow-up) -- same
// fixed-text/tooltip-detail pattern as renderSessionVolatilityAlert() above, +-30min window (see
// scripts/live_macro_calendar_20260826.py::MACRO_EVENT_ALERT_WINDOW_MIN). Separate badge, separate
// question ("is a scheduled data release imminent" vs "is it near a session open") -- both can be
// active at once, hence the shared flex wrapper in index.html rather than one badge with two texts.
function renderMacroEventAlert(alertPayload) {
  const badge = el("macroEventAlertBadge");
  if (!badge) return;
  const active = alertPayload && Array.isArray(alertPayload.active) ? alertPayload.active : [];
  if (!active.length) { badge.style.display = "none"; return; }
  const names = active.map((a) => a.title_ko).join(", ");
  const m = active[0].minutes_from_event;
  const when = m < 0 ? `발표 ${Math.round(Math.abs(m))}분 전` : m === 0 ? "발표 순간" : `발표 ${Math.round(m)}분 후`;
  badge.style.display = "";
  badge.title = `${names} ${when} — 경제지표/FOMC/연준 의장 발언 전후 ±30분 참고용 안내(검증된 시장개장 알림과 달리 개별 검증은 안 됨)`;
}


async function refreshChartMarkers() {
  const now = Date.now();
  if (now - chartMarkersLastFetchAt < CHART_MARKERS_POLL_MS) return;
  chartMarkersLastFetchAt = now;
  try {
    const res = await fetch(`${API_CHART_MARKERS_URL}?asset=${activeSnapshotAsset}`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`chart markers ${res.status}`);
    latestChartMarkers = await res.json();
  } catch (error) {
    console.error("Chart markers fetch error:", error);
    latestChartMarkers = { available: false, error: "fetch_failed" };
  }
}



// ── 24시간 변동성 전망 (2026-09-10, 2026-09-11 칩 -> 차트 리본) ────────────────────
// 규약: 색 §2 위험/주의=warn(주황) · 안정도 같은 주황을 옅게 -- **5번째 색을 만들지 않는다**.
// ⭐방향 신호가 아니다. 리본은 renderCandleSvg() 안에서 레짐 리본 바로 아래에 그린다.
// 칩이 갖고 있던 숫자는 전부 이 툴팁으로 옮겼다 -- 칩을 지워도 근거가 사라지지 않게.
// ── 변동성 전망 카드 (2026-09-16, 차트 리본에서 옮김) ──────────────────────────────
// 왜 카드인가: 이 값은 **1시간 격자·24시간 지평**이라 5분봉 시간축 위에서는 12봉이 한 색이고
// 풋프린트(1시간 창)에서는 값이 하나다 -- 그림으로 얻는 게 없다. 숫자 한 줄이 정보량이 같다.
// 🔴버리지 않는 이유: 09-14 정면비교에서 셋 중 **유일하게 «확장»을 본다**(현재변동성과 ρ
//   −0.817). 게이트와도 겹치지 않는다(상관 −0.389 · 상위10% 겹침 0.7%).


// ── 극점 탐지기 (2026-09-09) ────────────────────────────────────────────────────────
// 규약: 라벨 §1(측면 어휘) · 색 §2(바닥=good/천장=bad/그 외 neutral) · 제목 밑 데이터 줄 없음 §4
// ⭐5번째 색을 만들지 않는다 -- 억제/미발동은 전부 neutral 이다.

// 2026-09-11 추세 전환 **탐지기** — 발동 여부 한 축. 확률이 없으므로 게이지 없음(규약 §3).
// 2026-09-15 **E|r| 게이트** — 「언제」만 말한다. 방향 축이 아예 없는 카드다.
// 🔴카드에 방향을 넣지 말 것: 같은 아티팩트의 방향 분류기는 실계좌 72왕복에서 적중 47.2%
//   (동전 아래)이고 게이트가 고른 좋은 자리일수록 더 나빴다(−51.18bp · 호메로스 §5.36-R).
const GATE_GAUGE_NOTE = "게이지는 20자산 중 발동 비율입니다 — 확률이 아닙니다."
  + " 게이트 자체가 봉의 10% 만 켜도록 맞춰져 있습니다.";

// /api/evr-gate 처럼 «톤이 박힌 객체»로 오는 이력을 스트립이 먹는 모양으로 바꾼다.
function toneStripFromHistory(rows) {
  const list = Array.isArray(rows) ? rows : [];
  return { history: list.map((h) => (h && h.tone) || "neutral"),
           times: list.map((h) => (h && h.ts) || null) };
}

// 예고는 별도 카드(breakoutPrewarnIndicatorItem)로 뺐다 — 카드당 축 하나.
function breakoutDetectorIndicatorItem() {
  const p = latestBreakoutDetector;
  const base = { key: "breakout_detector", label: "추세 전환 탐지기",
                 derivedTag: "= 대시보드 자체계산",
                 derivedTitle: "봇 내부 상태가 아니라 전용 워커가 5분봉 마감마다 공개 kline 으로 계산합니다. "
                   + "방향은 예측하지 않습니다 -- 전환이 «왔다»만 말합니다. 매매에 연결돼 있지 않습니다." };
  if (!p || p.error || !p.available) {
    const sub = !p ? "웜업"
      : (p.error === "worker_fetch_failed" ? "오류"
        : (p.error === "fetch_failed" ? "오류" : "데이터 없음"));
    return { ...base, tone: "neutral", subText: sub, history: [], times: [] };
  }
  const det = p.detect || {};
  const meterNote = det.on ? `탐지 ${det.count}/${(det.signals || []).length}`
    : (p.volexp != null ? `volexp ${Number(p.volexp).toFixed(2)}` : null);
  const stateTitle = [
    det.on ? "거래대금·체결속도가 둘 다 q90 을 넘었습니다 — 전환이 시작됐습니다" : "발동 없음",
    ...((det.signals || []).map((x) => `${x.on ? "▲ " : "△ "}${x.name}`
      + (x.z != null && x.threshold != null ? ` z ${x.z} / 기준 ${x.threshold}` : ""))),
    p.volexp != null ? `변동성 확장비 ${Number(p.volexp).toFixed(3)} (압축 < 0.70 · 전환 >= 1.80)` : "",
    "전환 사건 176건 중 162건 포착(재현율 92.0%) · 발동 2,085회 중 813회가 사건 창 안(정밀도 39.0%)",
    "⚠️방향은 말하지 않습니다. 반대 포지션이면 청산까지 — 뒤집어 따라가라는 근거는 없습니다.",
  ].filter(Boolean).join("\n");
  return { ...base, probaSlot: true,
    tone: p.tone === "bad" ? "bad" : "neutral",
    subText: p.subText || "미발동",
    // ⚠️이 게이지는 **확률이 아니라 활성 여부(0/1)** 다 -- 규약 §3 예외라 툴팁에 성격을 밝힌다.
    proba: det.active ? 1 : 0,
    meterNote: det.active ? `탐지 지속 ${det.sustain_left_min}분` : meterNote,
    meterNoteTitle: det.active
      ? `발동 후 ${det.sustain_min}분 동안 게이지를 채워 둡니다 -- 신호의 수명입니다(확률 아님)`
      : "탐지는 2종 AND 입니다 — 둘 다 켜져야 발동합니다. 게이지는 활성 여부(0/1)이지 확률이 아닙니다",
    stateTitle,
    history: p.history || [], times: p.times || [] };
}

// 2026-09-11 추세 전환 **경보기** — «앞으로 30분 이내에 탐지기가 발동할 확률»(HGB 5시드 동결).
// 옛 경보 신호등 3종을 교체했다: 그 lift 5.75x 는 «앞 24봉 실현변동성»이라는 자명한 대리
// 타깃 값이었고(atr_pct 단독 7.01), 전환 기준으로 재면 1.98~2.53 으로 무작위 수준이었다.
// ⭐확률 축이 생겼으므로 게이지를 둔다(규약 §3 — 확률인 행만 게이지).
function breakoutPrewarnIndicatorItem() {
  const p = latestBreakoutDetector;
  const w = (p && p.prewarn) || null;
  const base = { key: "breakout_prewarn", label: "추세 전환 경보기", probaSlot: true,
                 derivedTag: "= 대시보드 자체계산",
                 derivedTitle: "전용 워커가 5분봉 마감마다 33개 피쳐를 만들어 동결된 HGB 5시드에 넣습니다. "
                   + "방향은 예측하지 않습니다 -- «곧 전환이 온다»만 말합니다. 매매에 연결돼 있지 않습니다." };
  if (!p || p.error || !p.available || !w || !w.available) {
    const sub = !p ? "웜업"
      : (p.error ? "오류" : (w && w.subText) || "데이터 없음");
    return { ...base, tone: "neutral", subText: sub, proba: null, history: [], times: [] };
  }
  const pct = w.proba != null ? (Number(w.proba) * 100).toFixed(1) : null;
  const stateTitle = [
    w.on ? `예고 발동 — 확률 ${pct}% 가 후행 임계 ${(Number(w.threshold) * 100).toFixed(1)}% 를 넘었습니다`
      : `미발동 — 확률 ${pct}% · 후행 임계 ${(Number(w.threshold) * 100).toFixed(1)}%`,
    `타깃: ${w.horizon} 탐지기가 한 번이라도 발동하는가`,
    w.precision != null
      ? `표본외 실측 정밀도 ${(Number(w.precision) * 100).toFixed(1)}% (기저 ${(Number(w.base_rate) * 100).toFixed(1)}%) · 커버리지 10% = 하루 약 28.8회`
      : "",
    "임계는 확률의 **후행 2016봉 분위** q90 입니다 — 전역 분위를 쓰면 미래참조입니다",
    "⚠️«탐지기가 켜진다»이지 «큰 이동이 온다»가 아닙니다. 탐지기 자체의 정밀도가 39.0% 라 "
      + "그 이상으로 올라갈 수 없습니다.",
  ].filter(Boolean).join("\n");
  return { ...base,
    tone: w.tone === "warn" ? "warn" : "neutral",
    subText: w.subText || "미발동",
    // 2026-09-11 사용자 요청: 울리면 **지속시간 동안 게이지를 채워 둔다**. 비활성일 때만 실제 확률.
    proba: w.active ? 1 : (w.proba != null ? Number(w.proba) : null),
    meterNote: w.active ? `예고 지속 ${w.sustain_left_min}분` : (pct != null ? `예고 ${pct}%` : null),
    meterNoteTitle: w.active
      ? `발동 후 ${w.sustain_min}분 동안 게이지를 채워 둡니다 -- 신호의 수명입니다. 현재 확률 ${pct}%`
      : "앞으로 30분 이내에 탐지기가 발동할 확률입니다(HGB 5시드 평균)",
    stateTitle,
    history: w.history || [], times: w.times || [] };
}


async function refreshBreakoutDetector() {
  const now = Date.now();
  if (now - breakoutDetectorLastFetchAt < BREAKOUT_DETECTOR_POLL_MS) return;
  breakoutDetectorLastFetchAt = now;
  try {
    const res = await fetch(API_BREAKOUT_DETECTOR_URL, { cache: "no-cache" });
    if (!res.ok) throw new Error(`breakout detector ${res.status}`);
    latestBreakoutDetector = await res.json();
  } catch (error) {
    console.error("Breakout detector fetch error:", error);
    latestBreakoutDetector = { error: "fetch_failed" };
  }
}


async function refreshLiquidation5mSignal() {
  const now = Date.now();
  if (now - liquidation5mLastFetchAt < LIQUIDATION_5M_POLL_MS) return;
  liquidation5mLastFetchAt = now;
  // 🔴요청을 **건 시점의** 코인을 들고 간다. 응답이 돌아왔을 때 화면이 다른 코인으로
  //   넘어가 있으면 버린다 -- 안 그러면 늦게 온 ETH 값이 BTC 라벨 아래 앉는다
  //   (2026-08-31 레짐 리본 사고와 같은 부류. 그때는 변수를 갈라 고쳤고 여기는 시점이 문제다).
  const asset = activeSnapshotAsset;
  try {
    const res = await fetch(`${API_LIQUIDATION_5M_URL}?asset=${asset}`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`liquidation 5m signal ${res.status}`);
    const j = await res.json();
    if (asset !== activeSnapshotAsset) return;
    latestLiquidation5m = j;
    try {
      const rh = await fetch(`${API_LIQUIDATION_5M_HIST_URL}?asset=${asset}`, { cache: "no-cache" });
      const jh = await rh.json();
      if (asset !== activeSnapshotAsset) return;
      latestLiquidation5mHist = (jh && jh.warmed_up && Array.isArray(jh.bars)) ? jh.bars : [];
    } catch (e) { if (asset === activeSnapshotAsset) latestLiquidation5mHist = []; }
  } catch (error) {
    console.error("Liquidation 5m signal fetch error:", error);
    if (asset === activeSnapshotAsset) latestLiquidation5m = { warmed_up: false, error: "fetch_failed" };
  }
}

// 2026-09-14 변동성 예측 칩. /api/position-sizing 은 **파일 하나만 읽는 엔드포인트**라
// (서버 position_sizing_payload 주석 참조) 새 계산을 요청 경로에 넣지 않는다.
async function refreshVolLevel() {
  const now = Date.now();
  if (now - volLevelLastFetchAt < VOL_LEVEL_POLL_MS) return;
  volLevelLastFetchAt = now;
  try {
    const res = await fetch(API_POSITION_SIZING_URL, { cache: "no-cache" });
    if (!res.ok) throw new Error(`position sizing ${res.status}`);
    const j = await res.json();
    latestVolLevel = (j && j.vol_level) || { available: false, grade: "데이터 없음", tone: "neutral" };
  } catch (error) {
    console.error("Vol level fetch error:", error);
    latestVolLevel = { available: false, grade: "오류", tone: "neutral" };
  }
  pushToneHistory("vol_level", (latestVolLevel && latestVolLevel.tone) || "neutral");
}


async function refreshLiqBurstState() {
  const now = Date.now();
  if (now - liqBurstStateLastFetchAt < LIQ_BURST_STATE_POLL_MS) return;
  liqBurstStateLastFetchAt = now;
  try {
    const res = await fetch(API_LIQ_BURST_STATE_URL, { cache: "no-cache" });
    if (!res.ok) throw new Error(`liq burst state ${res.status}`);
    latestLiqBurstState = await res.json();
  } catch (error) {
    console.error("Liq burst state fetch error:", error);
    latestLiqBurstState = { available: false };
  }
}

// Unlike latestVRebound (picked up by the next state-driven render() pass), the liquidation map
// has no such host -- it self-triggers both the panel list and the snapshot chart right after a
// fetch resolves, same pattern as refreshEvidenceSignals().
async function refreshLiquidationMap() {
  const now = Date.now();
  if (now - liquidationMapLastFetchAt < LIQUIDATION_MAP_POLL_MS) return;
  liquidationMapLastFetchAt = now;
  const asset = activeSnapshotAsset;          // 늦게 온 응답 버리기 -- 위 5m 신호와 같은 이유
  try {
    const res = await fetch(`${API_LIQUIDATION_MAP_URL}?asset=${asset}`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`liquidation map ${res.status}`);
    const j = await res.json();
    if (asset !== activeSnapshotAsset) return;
    latestLiquidationMap = j;
  } catch (error) {
    console.error("Liquidation map fetch error:", error);
    if (asset !== activeSnapshotAsset) return;
    latestLiquidationMap = { warmed_up: false, error: "fetch_failed" };
  }
  renderLiquidationMapPanel();
  scheduleSnapshotChartRender();
}

// wide24 HMM regime overlay (2026-08-26) for the Snapshot chart -- CONFIRMED research artifact,
// see scripts/live_regime_wide24_signal_20260826.py docstring for why it's independent of whatever
// regime model the live bot itself routes on.
async function refreshRegimeWide24() {
  const now = Date.now();
  if (now - regimeWide24LastFetchAt < REGIME_WIDE24_POLL_MS) return;
  regimeWide24LastFetchAt = now;
  try {
    const res = await fetch(API_REGIME_WIDE24_URL, { cache: "no-cache" });
    if (!res.ok) throw new Error(`regime wide24 ${res.status}`);
    latestRegimeWide24 = await res.json();
  } catch (error) {
    console.error("Regime wide24 fetch error:", error);
    latestRegimeWide24 = { warmed_up: false, error: "fetch_failed", history: [] };
  }
  scheduleSnapshotChartRender();
}

// BTC regime overlay (2026-09-02). Separate endpoint and separate state from latestRegimeWide24
// because they are two different trained models on two different assets -- the 2026-08-31 bug this
// replaces was exactly one variable being reused for both (ETH's ribbon drawn over BTC candles).
// Same poll interval, which matches the server-side cache TTL for both endpoints.
async function refreshRegimeBtc() {
  const now = Date.now();
  if (now - regimeBtcLastFetchAt < REGIME_WIDE24_POLL_MS) return;
  regimeBtcLastFetchAt = now;
  try {
    const res = await fetch(API_REGIME_BTC_URL, { cache: "no-cache" });
    if (!res.ok) throw new Error(`regime btc ${res.status}`);
    latestRegimeBtc = await res.json();
  } catch (error) {
    console.error("Regime BTC fetch error:", error);
    latestRegimeBtc = { warmed_up: false, error: "fetch_failed", history: [] };
  }
  scheduleSnapshotChartRender();
}

// XRP regime overlay (2026-09-03). BTC판과 같은 구조 -- 자산마다 **별도 상태 변수**를 쓴다.
// 하나를 공유하면 2026-08-31의 "ETH 리본이 BTC 캔들 위에 그려지던" 버그가 그대로 재현된다.
async function refreshRegimeXrp() {
  const now = Date.now();
  if (now - regimeXrpLastFetchAt < REGIME_WIDE24_POLL_MS) return;
  regimeXrpLastFetchAt = now;
  try {
    const res = await fetch(API_REGIME_XRP_URL, { cache: "no-cache" });
    if (!res.ok) throw new Error(`regime xrp ${res.status}`);
    latestRegimeXrp = await res.json();
  } catch (error) {
    console.error("Regime XRP fetch error:", error);
    latestRegimeXrp = { warmed_up: false, error: "fetch_failed", history: [] };
  }
  scheduleSnapshotChartRender();
}

// ⭐2026-09-03: ETH가 아닌 코인의 수급흐름/리테일수급/청산캐스케이드를 **그 코인 자신의**
// 실시간 수집기에서 가져온다. XRP/HYPE는 전용 워커가 microstructure까지 모으고
// (supervisor_xrp_worker.sh), tail_risk는 COIN_CONFIG에 5코인 전부 있다.
// ETH는 봇 state(dashboard_state.json)를 그대로 쓰므로 여기서 가져오지 않는다.
// 2026-09-03: 레짐 리본은 renderCandleSvg()의 REGIME_SOURCE_BY_ASSET가 **활성 코인 것 하나만**
// 그린다. 그런데 tick()은 매 사이클 3개(ETH/BTC/XRP)를 전부 받아오고 2개는 그대로 버렸다 --
// 코인이 늘수록 그대로 늘어나는 낭비라 활성 코인 것만 받도록 좁혔다. 자산별 상태 변수를
// 공유하지 않는 구조(refreshRegimeBtc/Xrp 주석의 2026-08-31 버그)는 그대로 유지한다.
function refreshActiveRegime() {
  if (activeSnapshotAsset === "eth") return refreshRegimeWide24();
  if (activeSnapshotAsset === "btc") return refreshRegimeBtc();
  if (activeSnapshotAsset === "xrp") return refreshRegimeXrp();
  return Promise.resolve();   // SOL/HYPE: 학습된 레짐 모델이 아직 없다
}

async function refreshCoinIndicators() {
  if (activeSnapshotAsset === "eth") return;
  const now = Date.now();
  if (now - coinIndicatorsLastFetchAt < MODEL_INDICATOR_POLL_MS) return;
  coinIndicatorsLastFetchAt = now;
  const asset = activeSnapshotAsset;
  try {
    const res = await fetch(`${API_COIN_INDICATORS_URL}?asset=${encodeURIComponent(asset)}`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`coin indicators ${res.status}`);
    latestCoinIndicators[asset] = await res.json();
  } catch (error) {
    console.error("Coin indicators fetch error:", error);
    latestCoinIndicators[asset] = { warmed_up: false, error: "fetch_failed" };
  }
}

// Macro/corporate event calendar (2026-08-26) -- see scripts/live_macro_calendar_20260826.py for
// sources/caveats. Purely informational (same tier as the evidence-signal list below it) -- not a
// trading signal, no economic-viability claim.
// 2026-08-27 (user request): badge date simplified to 오늘/내일 -- this list is already filtered to
// today/tomorrow only (isTodayOrTomorrowLocal below), so the literal MM/DD+weekday it used to show
// was redundant with that filter; 오늘/내일 says the same thing shorter and at a uniform width,
// which also makes the badge's new fixed-width CSS (#macroCalendarList .ops-health-status-badge)
// behave consistently instead of every badge being a different length. Falls back to MM/DD for the
// (currently unreachable, since the list is pre-filtered) case of a caller passing another day.
function fmtMacroCalendarTime(iso) {
  const d = new Date(iso);
  const today = new Date();
  const tomorrow = new Date(today.getTime() + 24 * 3600 * 1000);
  const datePart = d.toDateString() === today.toDateString() ? "오늘"
    : d.toDateString() === tomorrow.toDateString() ? "내일"
    : d.toLocaleDateString(undefined, { month: "2-digit", day: "2-digit" });
  const timePart = d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  return `${datePart} ${timePart}`;
}
async function refreshMacroCalendar() {
  const now = Date.now();
  if (now - macroCalendarLastFetchAt < MACRO_CALENDAR_POLL_MS) return;
  macroCalendarLastFetchAt = now;
  try {
    const res = await fetch(API_MACRO_CALENDAR_URL, { cache: "no-cache" });
    if (!res.ok) throw new Error(`macro calendar ${res.status}`);
    renderMacroCalendar(await res.json());
  } catch (error) {
    console.error("Macro calendar fetch error:", error);
    const sub = el("macroCalendarSub");
    if (sub) sub.textContent = "불러오기 실패";
  }
}
// 2026-08-26 user request: only today+tomorrow, by viewer's own local calendar day (not ET) --
// keeps the filter and the displayed toLocaleString() dates in the same frame of reference, so a
// KST viewer never sees an event dated "tomorrow" that got excluded by an ET-anchored cutoff.
function isTodayOrTomorrowLocal(iso) {
  const d = new Date(iso);
  const startOfToday = new Date();
  startOfToday.setHours(0, 0, 0, 0);
  const startOfDayAfterTomorrow = new Date(startOfToday.getTime() + 2 * 24 * 3600 * 1000);
  return d >= startOfToday && d < startOfDayAfterTomorrow;
}
function renderMacroCalendar(payload) {
  const sub = el("macroCalendarSub");
  const allEvents = payload && Array.isArray(payload.events) ? payload.events : [];
  const events = allEvents.filter((e) => isTodayOrTomorrowLocal(e.time_utc))
    .sort((a, b) => a.time_utc.localeCompare(b.time_utc));
  if (sub) sub.textContent = events.length ? `오늘·내일 ${events.length}건 (경제지표·FOMC·연준 발언·EIA·국채입찰·실적 — 정치일정 미포함)` : "오늘·내일 예정된 일정 없음";
  setH("macroCalendarList", events.length
    ? events.map((e) => {
        const tone = e.importance === "high" ? "warn" : "neutral";
        return `<article class="ops-health-row ${tone}">
          <span class="ops-health-dot" aria-hidden="true"></span>
          <div class="ops-health-info">
            <strong>${e.title_ko}</strong>
            <span>${e.detail || ""}</span>
          </div>
          <span class="ops-health-status-badge">${fmtMacroCalendarTime(e.time_utc)}</span>
        </article>`;
      }).join("")
    : `<div class="macro-calendar-empty">예정된 일정이 없습니다.</div>`
  );
}


function setupPageTabs() {
  document.querySelectorAll(".page-tab").forEach((button) => button.addEventListener("click", () => {
    const target = button.dataset.pageTab; // "ops" | "snapshot" | "notify" (라이브 탭 제거, 2026-08-31)
    activePageTab = target;
    el("opsTabPanel")?.classList.toggle("hidden", target !== "ops");
    el("snapshotTabPanel")?.classList.toggle("hidden", target !== "snapshot");
    el("notifyTabPanel")?.classList.toggle("hidden", target !== "notify");
    document.querySelectorAll(".page-tab").forEach((tab) => tab.classList.toggle("active", tab === button));
    if (target === "notify") {
      // 탭을 열 때마다 다시 읽는다 -- 권한이나 구독은 다른 탭/기기에서 바뀔 수 있고,
      // 낡은 상태를 보여주면 "켰는데 꺼졌다고 나온다"는 혼란만 만든다.
      refreshNotifyPage();
    } else if (target === "ops") {
      opsLastFetchAt = 0; refreshOpsStatus();
    } else if (target === "snapshot") {
      breakoutDetectorLastFetchAt = 0; refreshBreakoutDetector();
      chartMarkersLastFetchAt = 0; latestChartMarkers = null; refreshChartMarkers();
      liquidation5mLastFetchAt = 0; refreshLiquidation5mSignal();
      liqBurstStateLastFetchAt = 0; refreshLiqBurstState();
      liquidationMapLastFetchAt = 0; refreshLiquidationMap();
      regimeWide24LastFetchAt = 0; refreshRegimeWide24();
      regimeBtcLastFetchAt = 0; refreshRegimeBtc();
      regimeXrpLastFetchAt = 0; refreshRegimeXrp();
      coinIndicatorsLastFetchAt = 0; refreshCoinIndicators();
      macroCalendarLastFetchAt = 0; refreshMacroCalendar();
      sessionAlertsLastFetchAt = 0; refreshSessionAlerts();
      lastSnapshotHistoryFetchAt = 0; maybeFetchSnapshotChartHistory();
    }
  }));
}

function setupScrollRendering() {
  document.addEventListener("scroll", () => {
    lastScrollAt = Date.now();
    window.clearTimeout(scrollIdleTimer);
    // 이 타이머는 **따라잡기 전용**이다 -- 안 와도 isScrolling() 은 시간으로 풀린다.
    scrollIdleTimer = window.setTimeout(() => {
      if (!document.hidden) tick();
    }, SCROLL_IDLE_MS);
  }, { passive: true });
}

function isMobileChartMode() {
  return typeof window !== "undefined" && window.matchMedia("(max-width: 720px)").matches;
}

function mobileChartMaxStart(total, size) {
  return Math.max(0, total - size);
}

function normalizedMobileChartSize(total) {
  const maxSize = Math.min(MOBILE_CHART_MAX_CANDLES, Math.max(MOBILE_CHART_MIN_CANDLES, total || MOBILE_CHART_DEFAULT_CANDLES));
  const fallback = Math.min(MOBILE_CHART_DEFAULT_CANDLES, maxSize);
  return Math.round(clampNum(mobileChartView.size || fallback, MOBILE_CHART_MIN_CANDLES, maxSize));
}

function visibleCandleWindow(candles) {
  const source = Array.isArray(candles) ? candles : [];
  const total = source.length;
  if (!isMobileChartMode() || total <= MOBILE_CHART_DEFAULT_CANDLES) {
    return { candles: source, start: 0, end: total, total, includeCurrent: true };
  }

  const size = normalizedMobileChartSize(total);
  let start = mobileChartView.followLatest || mobileChartView.start === null
    ? mobileChartMaxStart(total, size)
    : Math.round(mobileChartView.start);
  start = Math.round(clampNum(start, 0, mobileChartMaxStart(total, size)));
  const end = Math.min(total, start + size);

  mobileChartView.size = size;
  mobileChartView.start = start;
  mobileChartView.followLatest = end >= total;

  return {
    candles: source.slice(start, end),
    start,
    end,
    total,
    includeCurrent: end >= total,
  };
}

// Time series of density snapshots for the chart's background heatmap, 2026-08-25 -- replaces the
// single "now" snapshot (liquidationDensityProfile(), same job through 2026-08-25) whose only way to
// show a swept bin was a one-way "go dark forever after this point" hack in renderCandleSvg()'s
// drawDensitySeg calls. That couldn't show a level re-lighting later as fresh volume re-accumulates
// there, which is exactly what a real Coinglass screenshot shows and what this replaces it with --
// see eth_liquidation_map_coinglass_visual_logic_replication_20260825 memory. map.heatmap_history is
// already the full time series from compute_heatmap_history() (one causal snapshot per hourly kline
// boundary, oldest-to-newest, weight_pct already globally normalized across the whole history --
// see that function's own docstring); this just reshapes each snapshot's bins for renderCandleSvg().
//
// Unlike liquidationDensityProfile() before it, this does NOT re-filter the latest snapshot against
// the live tick price -- every snapshot's own "alive" status is already grounded in real kline
// data as of its own hour boundary (compute_raw_bins()'s crossed-bin filter), so there's no single
// frozen "now" state left to go stale the way a one-shot snapshot could. The newest snapshot can
// still lag the live tick by up to ~1h (it reflects the last COMPLETED hourly candle, same
// staleness class discussed for nearestLiquidationLevel() -- but that function still exists and
// still gets its own live-price refilter for the one number a glance actually leans on; this
// background band is now an explicit history view, not a claimed-current one).
// 🔴결과를 **원본 payload 신원으로 memoize** 한다. 이 함수는 렌더마다 불리는데(청산맵
//   모드 1Hz · 풋프린트 모드에서는 아예 안 쓰임) 스냅샷 9개 x ~115빈 = 1,000여 개 객체를
//   매번 새로 만들고 있었다. 입력은 /api/liquidation-map 이 갱신될 때(60초)만 바뀐다.
//   ⭐배열 신원이 안정되면 아래 계층 캐시의 키로도 쓸 수 있다.
let _densityMemo = { src: null, out: [] };
function liquidationDensityHistory() {
  const map = latestLiquidationMap;
  if (_densityMemo.src === map) return _densityMemo.out;
  const out = (!map || !map.warmed_up || !map.bin_width) ? [] :
    (map.heatmap_history || []).map((snap) => ({
      tsMs: Date.parse(snap.ts_utc),
      binWidth: map.bin_width,
      bins: (snap.bins || []).map((b) => ({ price: b.price, weightPct: b.weight_pct || 0 })),
    }));
  _densityMemo = { src: map, out };
  return out;
}

// Single closest level (either side) to current price, as renderCandleSvg()'s riskLevels shape --
// 2026-08-24: the full 12-line overlay (top-6 support + top-6 resistance) was removed for clutter
// (see liquidationDensityProfile above, which now carries the "show the whole spread" job instead),
// but a bare heatmap gave up the one concrete, labeled number a glance actually wants -- "how far
// to the nearest wall". support_levels[0]/resistance_levels[0] are each already nearest-first
// (see renderLiquidationMapPanel), so this just picks whichever side is closer.
//
// 2026-08-25: re-filters/re-sorts against the LIVE tick price, same fix and same reason as
// liquidationDensityProfile() above -- map.support_levels[0]/resistance_levels[0] are each
// pre-filtered server-side by _redistance() against the backend's own current_price snapshot,
// which can trail the live tick price by up to ~1h (hourly klines + 5-min server cache). Without
// this, a level the live price has already crossed could still be drawn as an un-crossed wall.
function nearestLiquidationLevel() {
  const map = latestLiquidationMap;
  if (!map || !map.warmed_up) return [];
  const liveCurrentPrice = Number(latestLivePriceByAsset[activeSnapshotAsset] || map.current_price || 0);
  if (!(liveCurrentPrice > 0)) return [];
  const candidates = [
    { lv: (map.support_levels || [])[0], color: "var(--liq-support)", tag: "지지1", side: "support" },
    { lv: (map.resistance_levels || [])[0], color: "var(--liq-resistance)", tag: "저항1", side: "resistance" },
  ]
    .filter((c) => c.lv && Number(c.lv.price) > 0)
    .filter((c) => c.side === "support" ? c.lv.price < liveCurrentPrice : c.lv.price > liveCurrentPrice);
  if (!candidates.length) return [];
  candidates.sort((a, b) => Math.abs(a.lv.price - liveCurrentPrice) - Math.abs(b.lv.price - liveCurrentPrice));
  const nearest = candidates[0];
  return [{
    val: nearest.lv.price,
    color: nearest.color,
    label: nearest.tag,
    dashed: true,
    width: Math.max(1, Math.min(4, Math.round(1 + (nearest.lv.weight_pct || 0) * 3))),
  }];
}


// Keeps candleHistoryByAsset[activeSnapshotAsset]'s rightmost candle live between the 5-min
// maybeFetchSnapshotChartHistory() fetches, mirroring updateChart()'s in-place extend/roll logic
// for the Live tab's own candleHistory -- 2026-08-25, user report: the Snapshot chart's last candle
// sat frozen at whatever /api/market-history last returned while the "현재" price line (redrawn
// every 5s by the call below) kept moving, which read as the whole chart being stuck/shifted by one
// bar. Same bucket math as updateChart(): extend the last candle's high/low/close in place while
// still inside its 5-min bucket, or push a fresh one once the live tick crosses into a new bucket.
function updateSnapshotCandleLive() {
  const candles = candleHistoryByAsset[activeSnapshotAsset];
  if (!Array.isArray(candles) || !candles.length) return;
  const price = Number(latestLivePriceByAsset[activeSnapshotAsset] || 0);
  if (!(price > 0)) return;
  const tsMs = Date.parse(latestLivePriceTsByAsset[activeSnapshotAsset] || "");
  const ts = Math.floor((Number.isFinite(tsMs) ? tsMs : Date.now()) / 1000);
  const candleTs = Math.floor(ts / (CHART_CANDLE_MIN * 60)) * (CHART_CANDLE_MIN * 60);
  const last = candles[candles.length - 1];
  if (last.time < candleTs) {
    // 두 봉 이상 벌어졌으면 그 사이 봉은 틱으로 만들어낼 수 없다 -- 그냥 밀어 넣으면 사이가
    // **구멍으로 다음 폴링(5분)까지 굳는다**(2026-09-17 사용자 신고의 정체). 서버 쪽 낡은
    // 프레임 경로는 max_stale=0 으로 막았지만, 탭이 잠들었다 깨거나 fetch 가 한 번 실패하거나
    // 폴링이 봉 경계 직전에 걸리면 여전히 벌어질 수 있다. 폴링 게이트를 열어 다음 렌더가 바로
    // 다시 받게 한다 -- 아래 push 로 last.time == candleTs 가 되므로 이 조건은 한 번만 참이고
    // (요청 폭주 없음), 재요청이 실패해도 예전과 같은 상태로 남을 뿐이다.
    if (candleTs - last.time > CHART_CANDLE_MIN * 60) lastSnapshotHistoryFetchAt = 0;
    candles.push({ time: candleTs, open: price, high: price, low: price, close: price });
    if (candles.length > CHART_MAX_CANDLES) candles.shift();
  } else {
    last.high = Math.max(last.high, price);
    last.low = Math.min(last.low, price);
    last.close = price;
  }
}

// ── 현재가 빠른 갱신 (2026-09-16) ──────────────────────────────────────────────
// 왜 브라우저가 직접 WS 를 여는가: 서버 경유 경로의 상한은 SSE 주기(1초)다. 현재가 선은
// «지금 값»이라 1초도 느리게 보인다. 공개 스트림이라 인증이 없고, 브라우저 -> 바이낸스 직결이
// 서버를 거치지 않으므로 서버 부하도 0 이다. 실패하면 조용히 SSE 값으로 돌아간다(있던 게
// 사라지지 않는다) -- 그래서 «실패하면 아무 일도 없음»이 이 코드의 기본 동작이다.
// ⚠️전체 재렌더를 틱마다 하지 않는다. 한 번이 2~5ms 라 10Hz 면 모바일에서 배터리를 먹는다.
//   대신 표식이 달린 **세 요소만** 옮긴다(선·배지·숫자). 전체 렌더는 그대로 1초.
let liveLineCtx = null;
let priceWs = null, priceWsAsset = null, lastFastPriceAt = 0, priceWsRetryAt = 0;
const FAST_PRICE_MIN_INTERVAL_MS = 80;   // 12.5Hz. 사람 눈에는 연속이고 DOM 은 세 번만 만진다

// 삼각형 좌표는 여기 한 곳에서만 만든다(그리는 쪽·옮기는 쪽이 같은 모양을 써야 한다).
// 꼭짓점이 x 에 닿고 몸통이 오른쪽으로 -- 플롯을 침범하지 않고 행만 가리킨다.
const MARKER_W = 8, MARKER_H = 10;
function markerPoints(x, y) {
  return `${x},${y} ${x + MARKER_W},${y - MARKER_H / 2} ${x + MARKER_W},${y + MARKER_H / 2}`;
}

function updateLivePriceFast(price) {
  const c = liveLineCtx;
  if (!(price > 0)) return;
  const now = Date.now();
  if (now - lastFastPriceAt < FAST_PRICE_MIN_INTERVAL_MS) return;
  lastFastPriceAt = now;
  // 수급 프로파일의 현재가 박스는 캔들 차트와 무관하게 움직인다 -- 청산맵을 보고 있어도
  // 프로파일은 늘 화면에 있으므로 아래 liveLineCtx 가드보다 **앞에** 둔다.
  updateSupplyProfileNow(price);
  if (!c || !c.svg.isConnected) return;
  const span = Math.max(c.yMax - c.yMin, 1e-5);
  const raw = c.mt + ((c.yMax - price) * c.ch) / span;
  // 축 밖으로 나가면 가장자리에 붙인다 -- 다음 전체 렌더(1초 안)가 축을 다시 잡는다.
  const y = Math.max(c.mt, Math.min(c.mt + c.ch, raw));
  const line = c.svg.querySelector('[data-live="line"]');
  const tri = c.svg.querySelector('[data-live="tri"]');
  const box = c.svg.querySelector('[data-live="box"]');
  const txt = c.svg.querySelector('[data-live="text"]');
  const lab = c.svg.querySelector('[data-live="label"]');
  if (!box || !txt || !(line || tri)) return;   // 아직 안 그렸거나 현재가 표시가 없는 판
  if (line) { line.setAttribute("y1", y); line.setAttribute("y2", y); }
  if (tri) tri.setAttribute("points", markerPoints(c.markerX, y));
  const labelY = Math.max(c.mt + 9, Math.min(c.mt + c.ch - 9, y));
  box.setAttribute("y", labelY - 9);
  txt.setAttribute("y", labelY + 4);
  if (lab) lab.setAttribute("y", labelY + 4);
  txt.textContent = fmtNum(price, 1);
}

// ── 진행 중인 봉의 셀을 브라우저가 직접 쌓는다 (2026-09-16) ───────────────────────
// 이미 현재가용으로 **모든 체결**을 WS 로 받고 있다. 같은 규칙으로 버킷에 넣으면 진행 중인
// 봉은 서버 폴링(2초)을 기다릴 필요가 없다 -- 틱 단위로 자란다.
// ⚠️이중계상을 막는 규약: **WS 가 그 봉이 열리기 전부터 붙어 있었을 때만** 서버 값을 내 값으로
//   갈아끼운다. 봉 중간에 붙었으면 앞부분이 없으므로 서버 값을 그대로 쓴다(반쪽을 진짜처럼
//   보여주지 않는다 -- 서버 스냅샷에서 겪은 그 실패다).
let footprintLive = { barStart: 0, since: Infinity, bucket: 0.5, cells: new Map(),
                      orderQty: 0, orderAt: null, orderLastMs: 0, orderLastTid: -1 };

// ⚠️서버(파이썬)의 round() 는 **은행가 반올림**이다: 4880.5 -> 4880, 4881.5 -> 4882.
// JS Math.round 는 올림이라 4881, 4882 가 된다. 버킷이 0.5 이고 ETH 틱이 0.01 이라 가격이
// .25/.75 로 끝나면 정확히 .5 가 되는데, 그게 전체의 약 2% 다 -- 그만큼이 **다른 행**에 들어가
// 인계 순간 셀이 한 칸 튄다. 같은 격자를 쓰려면 같은 규약을 써야 한다.
function roundHalfEven(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

function footprintLiveAdd(price, qty, tsMs, sell, tid) {
  const barSec = Math.floor(tsMs / 1000 / CHART_CANDLE_MIN / 60) * CHART_CANDLE_MIN * 60;
  if (barSec !== footprintLive.barStart) {
    footprintLiveCloseOrder();     // 이전 봉의 마지막 주문을 흘리지 않는다
    footprintLive.barStart = barSec;
    footprintLive.cells = new Map();
  }
  if (footprintLive.since === Infinity) footprintLive.since = tsMs;   // WS 가 붙은 시각
  const key = roundHalfEven(price / footprintLive.bucket);
  const cell = footprintLive.cells.get(key) || [0, 0, 0, 0, 0, 0];
  cell[sell ? 1 : 0] += qty;      // 총량은 체결마다
  footprintLive.cells.set(key, cell);
  // 크기 구간은 **테이커 주문**이 닫힐 때. 파이썬 TakerOrderAggregator 의 거울이다:
  // 같은 가격·방향 + 연속 체결ID + 100ms 이내가 한 주문이다. 규칙이 갈리면 진행 중인 봉만
  // 다르게 갈려, 봉이 끝나고 서버 값으로 바뀌는 순간 셀이 튄다.
  // (「같은 ms」로 묶었다가 고래 물량을 10% 놓쳤다 -- 그 실측은 파이썬 쪽 도크스트링에.)
  const L = footprintLive;
  const same = L.orderAt
    && L.orderAt.price === price && L.orderAt.sell === sell
    && tsMs - L.orderLastMs <= 100
    && Math.floor(tsMs / 1000) === Math.floor(L.orderLastMs / 1000)   // 초에서 자른다
    && (tid == null || L.orderLastTid < 0 || tid === L.orderLastTid + 1);
  if (!same) {
    footprintLiveCloseOrder();
    L.orderQty = 0;
    L.orderAt = { price, sell, tsMs };   // 시각은 **첫** 체결
  }
  L.orderQty += qty;
  L.orderLastMs = tsMs;
  L.orderLastTid = (tid == null ? -1 : tid);
}

// 열려 있던 주문을 닫아 크기 구간에 넣는다. 다음 체결이 와야 닫히지만 ETH 는 초당 100건이
// 넘게 체결되므로 그 지연은 밀리초다.
function footprintLiveCloseOrder() {
  const at = footprintLive.orderAt, q = footprintLive.orderQty;
  footprintLive.orderQty = 0; footprintLive.orderAt = null; footprintLive.orderLastTid = -1;
  if (!at || !(q > 0)) return;
  const fp = latestFootprint || {};
  const whaleMin = Number(fp.whaleMinUsd) || 0, retailMax = Number(fp.retailMaxUsd) || 0;
  const notional = at.price * q;
  let slot = -1;
  if (whaleMin > 0 && notional >= whaleMin) slot = 2;
  else if (retailMax > 0 && notional < retailMax) slot = 4;
  if (slot < 0) return;                                  // 중형 -- 칸이 없다(뺄셈으로 나온다)
  const barSec = Math.floor(at.tsMs / 1000 / CHART_CANDLE_MIN / 60) * CHART_CANDLE_MIN * 60;
  if (barSec !== footprintLive.barStart) return;         // 봉이 넘어갔다 -> 서버 값이 맡는다
  const key = roundHalfEven(at.price / footprintLive.bucket);
  const cell = footprintLive.cells.get(key);
  if (!cell) return;
  cell[slot + (at.sell ? 1 : 0)] += q;
}

// 서버 payload 의 마지막 봉을 내 실시간 버킷으로 바꾼다(조건을 만족할 때만).
function footprintMergeLive(byTime, bucket) {
  footprintLive.bucket = bucket;   // 서버가 정한 격자를 따른다 -- 둘이 다르면 섞이면 안 된다
  const bar = footprintLive.barStart;
  if (!bar || !byTime.has(bar)) return;
  if (footprintLive.since > bar * 1000) return;   // 봉 중간에 붙었다 -> 서버 값 유지
  byTime.set(bar, [...footprintLive.cells.entries()]
    .map(([k, c]) => [k * bucket, c[0], c[1], c[2], c[3], c[4], c[5]])
    .sort((a, b) => a[0] - b[0]));
}

function ensurePriceWs() {
  const asset = activeSnapshotAsset;
  const symbol = ((ASSET_CONFIG[asset] || {}).symbol || "").toLowerCase();
  const want = activePageTab === "snapshot" && symbol && !document.hidden;
  if (!want || priceWsAsset !== asset) {
    if (priceWs) { try { priceWs.close(); } catch (e) { /* 이미 닫혔으면 그만이다 */ } priceWs = null; }
    priceWsAsset = null;
    if (!want) return;
  }
  if (priceWs) return;
  // 막힌 환경(회사망·차단)에서 tick 마다 다시 열면 초당 한 번씩 실패를 반복한다. 5초 간격으로.
  if (Date.now() < priceWsRetryAt) return;
  priceWsRetryAt = Date.now() + 5000;
  priceWsAsset = asset;
  try {
    const ws = new WebSocket(`wss://fstream.binance.com/ws/${symbol}@trade`);
    priceWs = ws;
    ws.onmessage = (ev) => {
      try {
        const d = JSON.parse(ev.data);
        if (d.e !== "trade") return;
        const price = Number(d.p);
        if (!(price > 0)) return;   // 바이낸스가 p:"0" 을 섞어 보낸다(실측 0.3%)
        latestLivePriceByAsset[priceWsAsset] = price;
        updateLivePriceFast(price);
        const qty = Number(d.q);
        if (qty > 0 && priceWsAsset === "eth") {
          footprintLiveAdd(price, qty, Number(d.T), !!d.m,
                           d.t == null ? null : Number(d.t));
          // 체결이 곧 셀의 변화다. 스로틀은 maybeRenderSnapshotChartNow 안에 있다(모드별).
          maybeRenderSnapshotChartNow();
        }
      } catch (e) { /* 한 메시지가 깨져도 스트림은 계속 간다 */ }
    };
    ws.onclose = () => { if (priceWs === ws) { priceWs = null; priceWsAsset = null; } };
    ws.onerror = () => { try { ws.close(); } catch (e) { /* noop */ } };
  } catch (e) {
    priceWs = null; priceWsAsset = null;   // WS 자체가 막힌 환경 -- SSE 값으로 산다
  }
}

async function refreshSupply1s() {
  if (activePageTab !== "snapshot" || document.hidden) return;   // 안 보이는 걸 매초 받지 않는다
  if (activeSnapshotAsset !== "eth") return;                     // 테이프는 ETH 만 수집한다
  const now = Date.now();
  if (now - supply1sLastFetchAt < SUPPLY_1S_POLL_MS) return;
  supply1sLastFetchAt = now;
  try {
    const res = await fetch(`${API_SUPPLY_1S_URL}?since=${supply1sSince}&sinceOi=${oi1sSince}`
                            + `&sinceLiq=${liq1sSince}`,
                            { cache: "no-cache" });
    if (!res.ok) throw new Error(`supply-1s ${res.status}`);
    const payload = await res.json();
    (payload.seconds || []).forEach((r) => {
      supply1s.set(r[0], r.slice(1));
      if (r[0] > supply1sSince) supply1sSince = r[0];
    });
    (payload.liq || []).forEach((r) => {
      liq1s.set(r[0], [r[1], r[2]]);          // [롱청산수량, 숏청산수량]
      if (r[0] > liq1sSince) liq1sSince = r[0];
    });
    (payload.oi || []).forEach((r) => {
      oi1s.set(r[0], r[1]);
      if (r[0] > oi1sSince) oi1sSince = r[0];
    });
    supply1sMeta = {
      retailMaxUsd: Number(payload.retailMaxUsd) || 0,
      whaleMinUsd: Number(payload.whaleMinUsd) || 0,
      now: Number(payload.now) || supply1sMeta.now,
    };
    // 창 밖은 버린다. 안 버리면 탭을 켜둔 채로 며칠이면 Map 이 수십만 칸이 된다.
    // 그리는 구간이 **현재 5분봉 하나**라 그 경계까지만 있으면 된다(최악 now-300).
    // OI 는 갱신이 3~7초라 20초를 더 준다. 넉넉히 두 봉치를 남겨 봉이 바뀌는 순간에도
    // 새 봉의 앞부분이 비지 않게 한다.
    const floor = supply1sMeta.now - 2 * SUPPLY_1S_SEGMENT - 20;
    supply1s.forEach((_v, k) => { if (k < floor) supply1s.delete(k); });
    oi1s.forEach((_v, k) => { if (k < floor) oi1s.delete(k); });
    liq1s.forEach((_v, k) => { if (k < floor) liq1s.delete(k); });
    supply1sVer += 1;
  } catch (error) {
    console.error("Supply 1s fetch error:", error);
  }
  // 받은 즉시 **이 패널만** 다시 그린다. 캔들 SVG 전체를 다시 그리지 않으므로 비싼 패스
  // (캔들·청산밀도·프로파일)는 안 탄다 -- 호버/스크롤 게이트에도 안 걸린다.
  repaintSupply1sPanel();
}

function gexIndicatorItem() {
  /* 옵션 감마 노출(GEX) -- **참고 표시 전용이고 신호가 아니다.**

     🔴라벨을 이론대로 붙이면 안 된다. 실측 rho(GEX, 전방RV) = **+0.44~+0.51** 로 부호가
       반대다(후행RV 통제 후에도 +0.25~+0.34). GEX 는 명목 달러라 «옵션시장 활동 수준 =
       변동성»의 결과 대리변수로 작동한다. 그래서 두 축으로 나눠 적는다:
         수준(total)        -> 변동성 대리. 높다고 «눌린다»가 아니다
         구조(front/total)  -> total 을 통제하면 front 가 이론 부호를 회복한다(t -6.22)
     ⏰판정일 1h 2026-09-28 / 4h 10-17. 그 전까지 CI 는 전부 0 을 포함한다.
     출처: docs/experiments/eth_gamma_zomma_graphic_research_20260916.md */
  const g = latestGex && latestGex.available
    ? (latestGex.currencies || {})[activeSnapshotAsset === "btc" ? "BTC" : "ETH"] : null;
  if (!g) {
    return { key: "gex", label: "옵션 감마 노출 (GEX)", tone: "neutral",
             subText: latestGex && latestGex.error ? "수집 지연" : "대기",
             derivedTag: "= 참고 · 신호 아님", derivedTitle: GEX_TITLE };
  }
  const bn = (v) => (v == null ? "-" : `${v >= 0 ? "+" : "-"}$${Math.abs(v / 1e9).toFixed(2)}B`);
  const ratio = g.front_ratio;
  // 2026-09-20 두 가지를 고쳤다(연구: docs/experiments/eth_realtime_five_stream_1s_joint_analysis_20260920.md §11).
  // 🔴①톤이 상수였다 -- `negative_gamma`(= total<0)가 854 스냅샷 37일 내내 0.0% 로 한 번도 참이 아니었다.
  //    실제로 변하는 축은 front 월물(6.7%)이고, 연구가 이론 부호를 회복한다고 한 축도 그쪽이다.
  // 🔴②달러 절대값($xx.xB)은 보정이 안 된다 -- 이 저장소 규약대로 **분위**를 같이 적는다.
  const pct = g.total_pct == null ? null : Math.round(g.total_pct * 100);
  const negFront = g.front_negative ?? false;      // 새 필드가 오기 전(cron 한 주기)에는 false
  return {
    key: "gex", label: "옵션 감마 노출 (GEX)",
    // 🔴톤은 위험도도 방향도 아니다. front 월물이 음수일 때만 주의(6.7%), 그 외 중립.
    tone: negFront ? "warn" : "neutral",
    subText: negFront ? "front 음감마" : (pct == null ? "양감마" : `수준 상위 ${100 - pct}%`),
    liveText: `수준 ${bn(g.total_gex_usd)}${pct == null ? "" : ` (분위 ${pct}%)`}`
      + ` · 구조 ${ratio == null ? "-" : ratio.toFixed(2)} (front÷total)`
      + (g.history_days ? ` · 기준 ${g.history_days}일` : ""),
    derivedTag: "= 참고 · 신호 아님",
    derivedTitle: GEX_TITLE,
  };
}

const GEX_TITLE = "딜러 감마 노출. Deribit 옵션 체인을 매시 수집해 계산합니다(2026-08-15~, 750+ 스냅샷).\n\n"
  + "🔴이론과 부호가 반대입니다. 실측 상관 rho(GEX, 앞으로의 실현변동성) = +0.44~+0.51 로, "
  + "GEX 가 높을수록 변동성이 «눌린다»가 아니라 «옵션시장 활동이 많다»는 뜻으로 작동합니다 "
  + "(명목 달러라 활동 수준의 결과 대리변수입니다).\n\n"
  + "그래서 두 축으로 나눠 읽습니다 — 수준(total)은 변동성 대리, 구조(front÷total)는 딜러 감마입니다. "
  + "total 을 통제하면 front 가 이론 부호를 회복합니다(t −6.22).\n\n"
  + "⏰아직 판정 전입니다. HAR-RV 대비 증분 R² 는 세 지평 모두 양수·단조지만(+0.011/+0.026/+0.041) "
  + "CI 가 전부 0 을 포함합니다(독립일 31). 판정 예정일은 1시간 지평 2026-09-28, 4시간 10-17 입니다. "
  + "그때까지 이 값은 매매 판단의 근거가 아니라 맥락입니다.\n\n"
  + "🔴방향으로 읽지 마세요. 2026-09-20 재측정(854스냅샷·37일)에서 GEX 와 앞 1~4시간 수익의 상관이 "
  + "−0.25~−0.34 로 크게 나왔지만, GEX 공식에 스팟²이 들어 있어 전부 가격수준의 사본이었습니다 "
  + "— 스팟을 통제하면 −0.05/−0.09(오차 안)로 사라지고, 스팟 단독이 GEX 보다 강합니다.\n\n"
  + "느린 지표입니다. 매시 갱신이라 화면 값은 최대 1시간 묵었고, 24시간 뒤 자기상관이 +0.40 "
  + "(높음/낮음 상태가 중앙 3시간 이어집니다). 초 단위 칩과 시간축이 다릅니다.\n\n"
  + "«미시 참고» 카드에는 넣지 않았습니다. 60초 거래량 분위를 고정하면 GEX 높음/낮음 행이 "
  + "갈리지 않고(앞 5분 고저폭 교차), 호가 방아쇠의 값도 GEX 레짐에 따라 달라지지 않았습니다"
  + "(+1.06 vs +1.17, 겹침 6일).";


async function refreshGex() {
  if (activePageTab !== "snapshot" || document.hidden) return;
  const now = Date.now();
  if (now - gexLastFetchAt < GEX_POLL_MS) return;
  gexLastFetchAt = now;
  try {
    const res = await fetch("/api/gex", { cache: "no-cache" });
    if (!res.ok) throw new Error(`gex ${res.status}`);
    latestGex = await res.json();
  } catch (error) {
    console.error("GEX fetch error:", error);
    latestGex = null;
  }
}

async function refreshFlowHeatmap() {
  if (activePageTab !== "snapshot" || document.hidden) return;
  if (activeSnapshotAsset !== "eth") return;   // 래스터 수집은 ETH 만 한다
  const now = Date.now();
  if (now - flowHeatmapLastFetchAt < flowHeatmapPollMs()) return;
  flowHeatmapLastFetchAt = now;
  try {
    const res = await fetch(
      `/api/flow/heatmap?symbol=ethusdt&cols=${FLOW_HEATMAP_COLS}`
      + `&agg=${flowHeatmapAgg()}&mode=rows`,
      { cache: "no-cache" });
    if (!res.ok) throw new Error(`flow-heatmap ${res.status}`);
    const j = await res.json();
    // 서버가 qty 를 **int8 로 양자화**해 보낸다(±127 = ±p97). 화면은 농도만 쓰므로
    // 원값이 필요 없고 페이로드가 f32 대비 1/4 이다. mid 는 가격이라 f32 그대로다.
    const raw = (b64) => {
      const bin = atob(b64); const u8 = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      return u8;
    };
    // mode=rows 라 이미지 배열(qty_i8/mid_b64 · 102KB)은 안 온다 -- 행 집계와 요약만.
    const f4 = (b64) => new Float32Array(raw(b64).buffer);
    // 2026-09-20 행 통계가 둘 -> 다섯. 247빈 x 4B x 5 = 5KB(걷어낸 이미지가 102KB 였다).
    const ROW_STATS = ["inst", "pers", "peak", "refill", "d60", "blk", "n_up"];
    // 2026-09-20 접근행동은 **4시간 창**이라 bin_lo/n_bins 가 위 다섯과 다르다
    // (그 사이 mid 가 움직여 격자가 넓다). 절대가격으로 따로 찾는다.
    if (j.rows && j.rows.approach_f4) j.rows.approach = f4(j.rows.approach_f4);
    // 🔴**없는 키는 건너뛴다.** 예전엔 ROW_STATS 를 그대로 돌려 f4(undefined) 가 던졌고,
    //   그 예외를 아래 catch 가 잡아 latestFlowHeatmap 을 통째로 null 로 만들었다 --
    //   화면에서 호가 막대가 조용히 사라진다. 서버보다 app.js 가 **먼저** 배포되면
    //   (화면 파일은 재기동 없이 즉시 서빙되므로 실제로 그 순서가 된다) 새 필드가 아직
    //   없어서 매번 그 경로를 탄다. 2026-09-20 blk/n_up 추가 때 실제로 재현했다.
    latestFlowHeatmap = { ...j, rows: j.rows
      ? ROW_STATS.reduce((acc, k) => {
          if (j.rows[k + "_f4"]) acc[k] = f4(j.rows[k + "_f4"]);
          return acc;
        }, { ...j.rows })
      : null };
    flowHeatmapVer += 1;
    repaintSupplyProfilePanel();   // 행별 지속 잔량이 같이 갱신된다
  } catch (error) {
    console.error("Flow heatmap fetch error:", error);
    latestFlowHeatmap = null;
    flowHeatmapVer += 1;
  }
}

async function refreshSupplyProfile() {
  if (activePageTab !== "snapshot" || document.hidden) return;
  if (activeSnapshotAsset !== "eth") return;   // 테이프는 ETH 만 수집한다
  const now = Date.now();
  if (now - supplyProfileLastFetchAt < SUPPLY_PROFILE_POLL_MS) return;
  supplyProfileLastFetchAt = now;
  try {
    const res = await fetch(`${API_SUPPLY_PROFILE_URL}?bars=${chartWindowBars}`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`supply-profile ${res.status}`);
    latestSupplyProfile = await res.json();
  } catch (error) {
    console.error("Supply profile fetch error:", error);
    latestSupplyProfile = null;
  }
  supplyProfileVer += 1;
  repaintSupplyProfilePanel();
}

async function refreshOi5m() {
  if (activePageTab !== "snapshot" || document.hidden) return;
  if (activeSnapshotAsset !== "eth") return;   // OI 수집은 ETH 만 한다
  const now = Date.now();
  if (now - oi5mLastFetchAt < OI_5M_POLL_MS) return;
  oi5mLastFetchAt = now;
  try {
    // 캔들 차트가 최대 72봉(6시간)을 그린다 -- 그보다 넓게 받아 두면 창을 늘려도 레인이 안 끊긴다.
    const res = await fetch(`${API_OI_5M_URL}?bars=96`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`oi-5m ${res.status}`);
    latestOi5m = await res.json();
  } catch (error) {
    console.error("OI 5m fetch error:", error);
    latestOi5m = null;
  }
  // 그리는 건 캔들 차트가 자기 주기에 한다(청산 5분 이력과 같은 방식) -- 여기서 또 부르면
  // 같은 SVG 를 한 번 더 통째로 다시 그린다.
}

// ── 상황 읽기 · 30분 (2026-09-21) ─────────────────────────────────────────
// dashboard/situation.py 가 낸 것을 그대로 그린다. 라벨 = «지금 무슨 상황인가», 시나리오 = 휴리스틱 확률과
// 목표가, 뒤집기 신호 = «이게 켜지면 생각을 바꾼다»의 실시간 판정, 맨 아래 = 장부의 적중률.
async function refreshSituation() {
  if (activePageTab !== "snapshot" || document.hidden) return;
  if (activeSnapshotAsset !== "eth") return;
  const now = Date.now();
  if (now - situationLastFetchAt < SITUATION_POLL_MS) return;
  situationLastFetchAt = now;
  try {
    const res = await fetch(API_SITUATION_URL, { cache: "no-cache" });
    if (!res.ok) throw new Error(`situation ${res.status}`);
    latestSituation = await res.json();
  } catch (error) {
    console.error("Situation fetch error:", error);
    latestSituation = { now: { ok: false, reason: "fetch_failed" } };
  }
  renderSituation();
}

function renderSituation() {
  const body = el("situationBody"); const badge = el("situationBadge");
  if (!body) return;
  const s = latestSituation || {}; const n = s.now || {};
  if (!n.ok) {
    if (badge) { badge.className = "ops-badge neutral"; badge.textContent = "대기"; }
    body.innerHTML = `<div class="sit-cal">${escapeHtml(n.reason || "서버가 첫 값을 계산하는 중")}</div>`;
    return;
  }
  const fmtPx = (v) => (v == null ? "-" : Number(v).toFixed(1));
  const order = ["A", "B", "C"].sort((a, b) => n.prob[b] - n.prob[a]);
  const top = order[0];

  // ── SCENARIOS ── 막대 색은 **시나리오 키에 고정**한다(k-A/k-B/k-C). 순위로 칠하면
  //   순위가 바뀔 때마다 같은 시나리오가 다른 색이 되어 «색이 정체성»이라는 규약이 깨진다.
  const scn = order.map((k, i) => {
    const t = n.targets[k];
    // 목표가 없는 경우는 **두 가지**이고 뜻이 정반대다 -- 한 문구로 묶으면 오해한다(09-21 사용자 «왜 잔여가 뜨지»).
    //  ① 횡보의 «레인지 유지» = 진짜 잔여: 위아래 둘 다 안 닿으면 이게 일어난다.
    //  ② 목표 선점(이미 지나감)·가치영역 없음 = 해당 없음: 이 시나리오는 이번 창에서 **일어날 수 없다**.
    //     «잔여»의 조건은 분명하다 -- **하단·상단 이탈선 사이에 머무는 것**이고 그 두 선은 이미 C·B 목표다.
    //     지켜야 할 값을 숨기지 말고 그대로 띄운다(사용자 요청). 한쪽 이탈선이 선점됐으면 레인지는 이미 깨졌다.
    const residual = n.dir === 0 && k === "A";
    const lo = n.targets.C, hi = n.targets.B;
    const holds = residual && lo != null && hi != null;
    const tgt = t == null
      ? (holds ? `${fmtPx(lo)}~${fmtPx(hi)}` : residual ? "이미이탈" : "해당없음")
      : Array.isArray(t) ? `${fmtPx(t[0])}~${fmtPx(t[1])}` : fmtPx(t);
    const title = t == null
      ? (holds ? "레인지 유지 — 하단·상단 이탈선 사이에 머물면 이것"
               : residual ? "해당 없음 — 이미 한쪽으로 이탈했다" : "해당 없음 — 목표를 이미 지나감")
      : "";
    const dead = t == null && !holds;
    return `<tr class="${i === 0 ? "" : "sub"}${dead ? " dead" : ""}"><td class="p">${n.prob[k]}%</td>`
      + `<td class="nm">${escapeHtml(n.names[k])}</td>`
      + `<td><div class="sit-bar"><i class="k-${k}" style="width:${n.prob[k]}%"></i></div></td>`
      + `<td class="tg"${title ? ` title="${escapeHtml(title)}"` : ""}>${escapeHtml(tgt)}</td></tr>`;
  }).join("");

  // ── STATE ── 서버가 고른 라벨 문장을 그대로 쓴다. 클라이언트가 문장을 쪼개 «키: 값»으로
  //   만들려면 한국어 파싱이 필요하고, 그건 서버 문구가 바뀌는 날 조용히 깨진다.
  const state = (n.labels || []).map((t) => {
    const hot = /스퀴즈|클라이맥스|분배|거부|전환 탐지|활발|쿠션 없음/.test(t);
    return `<div${hot ? ' class="hot"' : ""}>${escapeHtml(t)}</div>`;
  }).join("");

  // ── FLIP TRIGGERS ──
  const fl = n.flips || [];
  const armed = fl.filter((f) => f.on).length;
  const flips = fl.map((f) => `<div class="sit-flip${f.on ? " on" : ""}"><span class="dot"></span>`
    + `<span>${escapeHtml(f.signal)}</span>`
    + `<span class="to">${escapeHtml(n.names[f.toward] || f.toward)}</span></div>`).join("");
  const why = (n.why || []).map((w) => `${w["근거"]}: ${Object.entries(w).filter(([k]) => k !== "근거")
    .map(([k, v]) => `${k}${v >= 0 ? "+" : ""}${v}`).join(" ")}`).join(" · ");

  // ── LEDGER ── «말한 것 vs 실제»를 표로. 문장에 묻혀 있던 게 이 카드의 핵심 숫자다
  //   (실측: 되돌림 39 말하고 2 적중 · 역스퀴즈 33 말하고 83). gap = 실제 − 말한 것.
  const c = s.calibration || {};
  const led = c.n
    ? `<table class="sit-led"><thead><tr><th>시나리오</th><th>말한</th><th>실제</th><th>차</th></tr></thead><tbody>`
      + ["A", "B", "C"].map((k) => {
          const r = c[k] || {}; const g = (r.happened || 0) - (r.said || 0);
          return `<tr><td>${escapeHtml(n.names[k] || k)}</td><td>${r.said}</td><td>${r.happened}</td>`
            + `<td class="gap ${g >= 0 ? "under" : "over"}">${g >= 0 ? "+" : ""}${g}</td></tr>`;
        }).join("")
      + `</tbody></table>`
    : `<div class="sit-cal">${c.samples ? `표본 ${c.samples}건 기록 · 해결대기 ${c.pending}` : "기록 시작 대기"} (첫 해결은 예측 30분 뒤)</div>`;

  const sy = c.sym || {};
  const st = s.streams || {}; const fo = st.fo || {}; const mp = st.mp || {};
  const wsDot = (w, label) => `<span class="sit-ws" title="${escapeHtml(label)} ${w.connected
    ? `연결 · ${w.events || 0}건` : `끊김${w.last_error ? ` (${w.last_error})` : ""}`}">`
    + `<i class="${w.connected ? "" : "off"}"></i>${escapeHtml(label)}</span>`;
  const foot = [
    c.n ? `1순위 적중 <b>${c.top_hit}%</b>` : null,
    c.n ? `아무 목표도 안 닿음 ${c.none}%` : null,
    sy.n ? `방향 적중 <b>${sy.hit}%</b> · 항상 상승이면 ${sy.base_up}% <span title="대칭 ±0.5×창폭 · 추세 구간만">(n ${sy.n})</span>`
         : "방향 적중: 대칭 라벨 해결 대기",
    c.n ? `해결 <b>${c.n}</b> · 대기 ${c.pending} · 에피소드 <b>${c.episodes}</b> · 연속타깃 ${c.with_path} · 판정불가 ${c.amb}% · 최근 ${c.span_h}h` : null,
  ].filter(Boolean).join("</span><span>");

  body.innerHTML = `
    <div class="sit-sec">SCENARIOS · 30m<span>휴리스틱 확률</span></div>
    <table class="sit-tbl"><tbody>${scn}</tbody></table>
    <details class="sit-why"><summary>점수 근거</summary><div>${escapeHtml(why || "기본값만")}</div></details>
    <div class="sit-sec">현재 상황</div>
    <div class="sit-state">${state}</div>
    <div class="sit-sec">FLIP TRIGGERS<span>${armed} / ${fl.length} 켜짐</span></div>
    <div class="sit-flips">${flips || '<div class="sit-cal">없음</div>'}</div>
    <div class="sit-sec">LEDGER${c.n ? `<span>n ${c.n} · ${c.span_h}h</span>` : ""}</div>
    ${led}
    <div class="sit-foot"><span>${foot}</span>${wsDot(fo, "청산 WS")}${wsDot(mp, "마크가격 WS")}</div>`;

  if (badge) {
    const age = s.computed_at ? Math.round(Date.now() / 1000 - s.computed_at) : null;
    badge.className = `ops-badge ${age != null && age <= 15 ? "good" : "neutral"}`;
    badge.textContent = `${escapeHtml(n.names[top])} ${n.prob[top]}%${age != null ? ` · ${age}초 전` : ""}`;
  }
}

// ── 풋프린트 증분 (2026-09-20) ──────────────────────────────────────────
// 400ms 폴링인데 48봉을 통째로 받고 있었다. 서버 실측: 400ms 간격 19번 중 내용이 실제로
// 바뀐 건 12번이고 바뀌는 건 **맨 오른쪽 봉 하나**다(닫힌 봉은 5분에 한 번). 그래서 봉을
// 여기 Map 에 쌓아 두고 **꼬리 두 봉만** 물어본다. 폴링 주기도 렌더 주기도 안 건드린다 --
// 줄어드는 건 «안 바뀐 47봉을 다시 받는» 바이트뿐이라 지연은 1ms 도 안 늘어난다.
// 🔴꼬리가 **두 봉**인 이유: 봉 경계에서 늦게 도착한 체결이 직전 봉에 들어간다.
// 🔴전량으로 되돌리는 판단은 **서버의 `full`** 을 따른다 -- 백필 중(ready=False)에는 과거 봉도
//   바뀌므로 서버가 전량을 주고, 그때 캐시를 통째로 갈아끼운다.
let footprintBars = new Map();          // 봉시각 -> 서버가 준 봉 객체(그대로)
let footprintCacheKey = "";             // 코인|창 -- 달라지면 캐시를 버리고 전량부터

async function refreshFootprint() {
  if (activeSnapshotAsset !== "eth") return;   // 테이프는 ETH 만 수집한다
  const now = Date.now();
  if (now - footprintLastFetchAt < FOOTPRINT_POLL_MS) return;
  footprintLastFetchAt = now;
  const key = `${activeSnapshotAsset}|${chartWindowBars}`;
  if (key !== footprintCacheKey) { footprintBars = new Map(); footprintCacheKey = key; }
  const barSec = Number(latestFootprint && latestFootprint.barSeconds) || 300;
  let newest = footprintBars.size ? Math.max(...footprintBars.keys()) : 0;
  // 🔴2026-09-21 **자가복구**. 502 한 번(배포 재시작은 실측 5~6초)에 화면이 굳어 12분을
  // 옛 봉에 머문 신고가 있었다. 어느 게이트에 걸렸는지 원격으로 특정할 수 없었으므로,
  // «원인»이 아니라 «증상»을 잡는다: 캐시의 최신 봉이 두 봉 넘게 묵었는데 탭이 보이는
  // 중이면 증분을 포기하고 **전량을 다시 받는다**(12.5KB, 드물게 일어난다).
  // 증분(since=)만 믿으면 캐시가 한 번 어긋났을 때 스스로 빠져나올 길이 없다.
  if (newest && !document.hidden && (now / 1000 - newest) > barSec * FOOTPRINT_STALE_BARS) {
    console.warn(`footprint 정체 ${Math.round(now / 1000 - newest)}s -- 전량 재수신`);
    footprintBars = new Map();
    newest = 0;
  }
  const since = newest ? newest - barSec : 0;   // 0 = 전량
  try {
    const res = await fetch(`${API_FOOTPRINT_URL}?bars=${chartWindowBars}&since=${since}`,
                            { cache: "no-cache" });
    if (!res.ok) throw new Error(`footprint ${res.status}`);
    const payload = await res.json();
    if (payload.full) footprintBars = new Map();
    (payload.bars || []).forEach((b) => footprintBars.set(b.time, b));
    // 창 밖으로 밀려난 봉은 버린다 -- 증분이라 서버가 «빠졌다»를 말해 주지 않는다.
    if (footprintBars.size > chartWindowBars) {
      [...footprintBars.keys()].sort((a, b) => a - b)
        .slice(0, footprintBars.size - chartWindowBars)
        .forEach((t) => footprintBars.delete(t));
    }
    // 아래 소비자(footprintForChart)는 예전과 **같은 모양**을 본다 -- 시각순 전체 배열.
    latestFootprint = { ...payload,
                        bars: [...footprintBars.values()].sort((a, b) => a.time - b.time) };
  } catch (error) {
    console.error("Footprint fetch error:", error);
    latestFootprint = null;   // null 이면 차트가 그냥 예전 캔들로 되돌아간다
    // 🔴캐시는 **안 버린다**. 한 번의 네트워크 실패로 12.5KB 를 다시 받을 이유가 없다 --
    //   다음 성공 폴링이 꼬리 두 봉만 얹으면 그대로 이어진다.
  }
  scheduleSnapshotChartRender();
}

// 풋프린트가 없으면(다른 코인 · 서버 웜업 · fetch 실패) null 을 돌려주고, 차트는 캔들로 그린다.
function footprintForChart() {
  if (activeSnapshotAsset !== "eth") return null;
  const payload = latestFootprint;
  const bars = Array.isArray(payload && payload.bars)
    ? payload.bars.filter((b) => Array.isArray(b.levels) && b.levels.length) : [];
  if (!bars.length) return null;
  const bucket = Number(payload.bucket) || 0.5;
  const byTime = new Map(bars.map((b) => [b.time, b.levels]));
  const aggTimes = new Set(bars.filter((b) => b.agg).map((b) => b.time));
  footprintMergeLive(byTime, bucket);   // 진행 중인 봉만 실시간 값으로 대체
  return {
    bucket,
    whaleMinUsd: Number(payload.whaleMinUsd) || 0,
    retailMaxUsd: Number(payload.retailMaxUsd) || 0,
    aggTimes,
    ready: !!payload.ready,
    barsExpected: Number(payload.barsExpected) || bars.length,
    barCount: bars.length,
    firstTime: bars[0].time,
    byTime,
  };
}

// 체결량 표기 -- 셀 폭이 30px 대라 네 글자를 넘기면 안 된다(ETH 수량 기준).
function fmtFootprintQty(v) {
  if (!(v > 0)) return "";
  if (v >= 1000) return (v / 1000).toFixed(v >= 10000 ? 0 : 1) + "k";
  if (v >= 100) return v.toFixed(0);
  return v.toFixed(1);
}


// 한 봉(또는 한 초)의 수급을 셋으로 가른다. 2026-09-19 「수급 리본」(캔들 아래 두 줄)이
// 여기 있었으나 사용자 지시로 걷어냈다 -- 보고 싶은 것은 5분봉이 아니라 **최근 5분을 1초
// 해상도로** 보는 화면이다. 가르는 규칙 자체는 그대로 쓰이므로 이 함수만 남긴다.
function supplyFlowOfBar(levels) {
  let buy = 0, sell = 0, wBuy = 0, wSell = 0, rBuy = 0, rSell = 0;
  (levels || []).forEach((l) => {
    buy += Number(l[1]) || 0; sell += Number(l[2]) || 0;
    wBuy += Number(l[3]) || 0; wSell += Number(l[4]) || 0;
    rBuy += Number(l[5]) || 0; rSell += Number(l[6]) || 0;
  });
  // 고래·리테일은 제 칸에서 읽는다. **중형만** 뺄셈이다 -- 셋을 더하면 전체 델타가 된다.
  return { whale: wBuy - wSell, retail: rBuy - rSell,
           mid: (buy - wBuy - rBuy) - (sell - wSell - rSell) };
}


// ── 수급 · 최근 5분 x 1초 (2026-09-19, 2026-09-20 절대값으로 개편) ─────────
// y 는 **5분 벽시계 경계에서 0으로 다시 쌓는 누적 순수급**(매수-매도, ETH). 끝점이 곧
// 「이번 5분에 순 몇 ETH」이고, 선이 올라가는 중이면 지금 들어오는 중이다.
// 고래는 0선 기준 면적으로 칠한다. 경계는 **아래 캔들과 같은 자리**라 두 그림이 같은 구간을
// 말한다. 기준점이 벽시계라 새로고침·재기동과 무관하다.
//
// 🔴여기 원래 «창 시작을 0으로 둔 누적선»이 있었다. 두 겹으로 상대값이었다: ①기준점이
//   매초 미끄러지고 ②눈금이 창 최대(max|v|)로 자동정규화돼 **조용한 5분과 터진 5분이
//   화면상 같은 크기**였다. 더 근본적으로 누적선은 «지금 들어오나»를 **기울기**에 담는데,
//   사람은 선차트에서 높이를 읽지 기울기를 못 읽는다 -- 절대값으로만 바꿔도 안 풀린다.
//   (2026-09-20 사용자: "상대값이라 눈에 딱 들어오지 않는다")
// 🔴그 다음엔 «30초 롤링 창»이었다. 그것도 틀렸다(2026-09-20, 사용자가 실제로 속았다):
//   고래 매수 +305 가 들어오면 30초 뒤 창에서 빠지면서 **아래로 뚝 떨어지는 획**이 생긴다.
//   그 시각엔 아무 일도 없었는데 «갑작스러운 매도»로 읽힌다 -- 잡음이 아니라 거짓말이다.
// 왜 누적인가: 초별 원값은 꼬리가 무겁다(실측 330초 중앙 1.5 / 최대 401, **260배**).
//   선형 절대 눈금에 얹으면 데이터의 절반이 **0.2픽셀**이라 사실상 안 그려진다 -- 막대로
//   그리든 선으로 그리든 마찬가지였다. 누적은 그 꼬리를 접는다: 같은 실측에서 5분 누적이
//   리테일 80 / 고래 180 / 신규계약 889 로 **11배 안**에 들어와 셋 다 한 눈금에서 보인다.
//   비선형 축(symlog)을 쓰지 않아도 되므로 「높이 2배 = 수량 2배」가 지켜진다.
// 누적이 매초 한 번만 움직이므로 가짜 사건도 안 생긴다.
// 눈금은 **계단 고정**(SUPPLY_1S_STEPS)이다. 완전 고정은 잘리고 자동은 크기를 지운다 --
// 계단이면 «같은 높이 = 같은 수량»이 대체로 성립하고 스케일이 초마다 튀지 않는다.
// 세 선(고래·리테일·신규계약)은 같은 자로 그린다. 단위가 같은 ETH 라서, 들어온 순수급 중
// 얼마가 새 포지션이고 얼마가 손바뀜인지가 세 선의 간격으로 바로 읽힌다.
// 창 누적(옛 화면이 그리던 값)은 선을 지우고 머리글 숫자로만 남겼다.
function renderSupply1s(box = null) {
  // box 가 오면 그 중첩 <svg> 에 그린다(캔들 SVG 안). 없으면 옛 독립 컨테이너를 찾는다.
  const svg = box ? box.svg : el("supply1sSvg");
  if (!svg) return;
  const NS = "http://www.w3.org/2000/svg";
  // 🔴폭을 1200 으로 고정했더니 컨테이너(1318)를 못 채워 **이 차트만 좁게** 그려졌다
  //   (2026-09-19 실측: 캔들·프로파일 1318 vs 여기 1240). viewBox 를 부모에서 받아야
  //   세 차트의 그려지는 폭이 같아진다. 여백도 캔들 차트와 같은 값(45/112)을 쓴다.
  const parentW = box ? 0 : (svg.parentElement ? svg.parentElement.clientWidth : 0);
  // 2026-09-19 가격선 띠를 뺐다(사용자 지시) -- 바로 아래 풋프린트 캔들이 같은 가격을
  // 이미 보여준다. 그만큼 높이를 돌려줘서 패널이 짧아지고 누적선이 커진다(240 -> 150).
  // 🔴모바일에서 폭을 1200 으로 잡으면 뷰박스가 8:1 이 되어 158px 상자 안에서 **42px 로**
  //   줄어든다(meet). 10px 글자가 3px 가 된다 -- 캔들 차트가 쓰는 규약(모바일은 실제 폭)을
  //   여기서도 쓴다. 여백도 좁은 화면에 맞춰 줄인다(45/112 는 336px 폭의 47% 다).
  const w = box ? box.w : (parentW > 0 ? Math.max(parentW, 320) : 1200);
  const h = box ? box.h : 150;
  const mt = 16, mb = 14;
  const narrow = w < 760;   // 모바일이든 좁은 상자든 여백 규칙은 같다
  // mr 은 오른쪽 꼬리표(고래/리테일/신규계약 + 값)가 앉는 자리다. 64 면 «신규계약 +0.4k»
  // 가 약 4px 넘친다(2026-09-19 계산) -- 72 로 두면 셋 다 들어가고 선은 8px 만 짧아진다.
  const ml = narrow ? 34 : 45, mr = narrow ? 72 : 112;
  const cw = w - ml - mr;
  const flowTop = mt, flowH = h - mb - flowTop;
  // 🔴이 줄이 없어서 HTML 의 고정 viewBox(1200) 가 그대로 남아 있었다. 폭을 부모에서 받도록
  //   바꾸는 순간 그림이 viewBox 밖으로 나간다 -- 좌표계와 뷰박스는 같이 움직여야 한다.
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.innerHTML = "";

  const now = supply1sMeta.now || 0;
  // 🔴창이 «최근 5분»(미끄러짐)이 아니라 **지금 만들어지고 있는 5분봉 그 자체**다
  //   (2026-09-20 사용자 선택: 시안 H). x축 왼쪽 끝 = 봉이 열린 시각, 오른쪽 끝 = 봉이
  //   닫힐 시각. 선은 봉이 진행되는 만큼 왼쪽에서 오른쪽으로 자라고, 다음 봉에서 리셋된다.
  //   ⭐이 패널이 말하는 수급 = **바로 아래 풋프린트 봉을 만들고 있는 그 체결들**이다
  //     (server.py footprint_bar_start 와 같은 식으로 자른 같은 경계).
  //   ⚠️캔들 «차트»와 x축이 겹치는 건 아니다 -- 그쪽은 12~48봉(1~4시간)을 같은 폭에 그린다.
  //     겹치는 것은 **데이터 구간**이지 가로 좌표가 아니다.
  const first = Math.floor(now / SUPPLY_1S_SEGMENT) * SUPPLY_1S_SEGMENT;
  // 이제 한 구간만 그리므로 이전 구간을 읽을 이유가 없다(누산기가 이 봉의 경계에서 시작한다).
  const allSecs = [...supply1s.keys()].filter((s) => s >= first && s <= now)
                                      .sort((a, b) => a - b);
  const secs = allSecs;
  if (secs.length < 2) {
    const txt = document.createElementNS(NS, "text");
    txt.setAttribute("x", w / 2); txt.setAttribute("y", h / 2);
    txt.setAttribute("text-anchor", "middle"); txt.setAttribute("fill", "var(--muted)");
    txt.textContent = "체결 테이프 집계 중...";
    svg.appendChild(txt);
    return;
  }

  const xAt = (s) => ml + ((s - first) / SUPPLY_1S_SEGMENT) * cw;
  // 🔴빈 구간을 직선으로 이으면 «그동안 아무 일도 없었다»로 읽힌다 -- 실제로는 «모른다»다
  //   (수집기 재기동·WS 끊김·백필이 아직 안 닿은 구간). 2026-09-19 첫 렌더에서 실제로 긴
  //   사선이 그어졌다. 초가 SUPPLY_1S_GAP_SEC 넘게 비면 선을 **끊는다**.
  // 5초로 둔 이유: 폴링이 1초라 한두 번 늦는 건 일상이고, 그때마다 띠를 그리면 잡음이 된다.
  // 5초가 비면 그건 폴링 지각이 아니라 실제 공백이다.
  const SUPPLY_1S_GAP_SEC = 5;
  // 선을 끊는 두 이유: ①실제 공백 ②**구간 경계**(거기서 누적이 0으로 돌아가므로 이으면
  //   없는 낙차를 그린다). 두 판정을 한 곳에 둬서 선과 면적이 같은 자리에서 끊긴다.
  const segOf = (s) => Math.floor(s / SUPPLY_1S_SEGMENT);
  const brk = (s, prev, gap) => prev === null || s - prev > gap || segOf(s) !== segOf(prev);
  const pathOf = (rows, yOf, gap = SUPPLY_1S_GAP_SEC) => {
    let d = "", prev = null;
    rows.forEach((r) => {
      const cmd = brk(r.s, prev, gap) ? "M" : "L";
      d += (d ? " " : "") + cmd + xAt(r.s).toFixed(1) + " " + yOf(r).toFixed(1);
      prev = r.s;
    });
    return d;
  };
  const line = (d, color, width, opacity, dash) => {
    const path = document.createElementNS(NS, "path");
    path.setAttribute("d", d); path.setAttribute("fill", "none");
    path.setAttribute("stroke", color); path.setAttribute("stroke-width", width);
    path.setAttribute("stroke-opacity", opacity);
    path.setAttribute("stroke-linejoin", "round");
    if (dash) path.setAttribute("stroke-dasharray", dash);
    svg.appendChild(path);
  };
  // 🔴fmtFootprintQty 는 0 에서 **빈 문자열**을 준다(풋프린트 셀에서 «이 칸은 안 그림»이라는
  //   뜻이라 거기서는 그게 맞다). 여기선 꼬리표라 그대로 쓰면 「고래 +」처럼 숫자가 사라진다
  //   -- 30초 창은 정확히 0 이 흔하다(고래 주문이 분당 13건이라 30초에 0건인 구간이 있다).
  //   2026-09-20 배포본 스크린샷에서 실제로 「고래 +」로 떠 있었다.
  const qty = (v) => fmtFootprintQty(Math.abs(v)) || "0";
  const label = (x, y, text, color, anchor) => {
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", x); t.setAttribute("y", y); t.setAttribute("font-size", "10");
    t.setAttribute("fill", color); if (anchor) t.setAttribute("text-anchor", anchor);
    t.textContent = text;
    svg.appendChild(t);
  };

  // 구간 경계에서 0으로 되돌리며 쌓는다. 경계 이전 초도 **계산에는** 들어간다(누산기를
  // 그때 0으로 되돌리는 게 전부이고, 그리는 건 first 이후뿐이다).
  const cumOf = (at) => {
    const rows = [];
    let acc = 0, seg = null;
    allSecs.forEach((s) => {
      if (segOf(s) !== seg) { seg = segOf(s); acc = 0; }
      acc += at(s);
      if (s > first) rows.push({ s, v: acc });
    });
    return rows;
  };
  const whale = cumOf((s) => { const c = supply1s.get(s); return c[2] - c[3]; });
  const retail = cumOf((s) => { const c = supply1s.get(s); return c[0] - c[1]; });
  // 2026-09-22 청산(사용자 요청). 고래·리테일과 **같은 원리**다 -- cumOf 가 봉 경계에서
  //   0으로 되돌리며 쌓는다. 접근자만 주면 되고 좌표·리셋 로직은 한 줄도 안 건드린다.
  // ⭐부호: 롱 청산은 시장에 강제 SELL(아래로), 숏 청산은 강제 BUY(위로)다. 그래서
  //   «숏 − 롱» 이 이 축의 뜻(순유입)과 그대로 맞는다.
  // 🔴양쪽이 동시에 크게 터지면 상쇄돼 0 근처로 보인다 -- 그건 «조용했다»가 아니라
  //   «양방향이었다»다. 규모 자체는 아래 꼬리표에 롱/숏을 따로 적어 그 혼동을 막는다.
  const liqNet = cumOf((s) => { const c = liq1s.get(s); return c ? c[1] - c[0] : 0; });
  // 꼬리표용 누계(규모). 상쇄 없이 각 방향의 총량이다.
  const liqSum = [0, 0];
  allSecs.forEach((s2) => {
    if (s2 <= first) return;
    const c = liq1s.get(s2);
    if (c) { liqSum[0] += c[0]; liqSum[1] += c[1]; }
  });
  // 신규계약(OI)은 **레벨**이라 더하지 않는다: 그 구간 첫 관측 대비 증분이다.
  // 🔴갱신이 3~7초라 구간의 «첫 관측»이 경계보다 조금 뒤다 -- 그만큼 증분이 과소평가된다.
  //   서버가 초 단위 OI 를 안 들고 있어 더 정확히는 못 한다. 체결(고래·리테일)은 초 단위라
  //   이 근사가 없다.
  const oiKeys = [...oi1s.keys()].filter((s) => s > first - SUPPLY_1S_SEGMENT - 20 && s <= now)
                                 .sort((a, b) => a - b);
  const oiRows = [];
  let oiSeg = null, oiBase = 0;
  oiKeys.forEach((s) => {
    if (segOf(s) !== oiSeg) { oiSeg = segOf(s); oiBase = oi1s.get(s); }
    if (s > first) oiRows.push({ s, v: oi1s.get(s) - oiBase });
  });
  const peak = Math.max(0, ...whale.map((r) => Math.abs(r.v)), ...retail.map((r) => Math.abs(r.v)),
                        ...oiRows.map((r) => Math.abs(r.v)), ...liqNet.map((r) => Math.abs(r.v)));
  const span = SUPPLY_1S_STEPS.find((a) => a >= peak) || Math.max(peak, 1e-9);
  const mid = flowTop + flowH / 2;
  const half = flowH / 2 - 10;
  // 마지막 계단을 넘는 폭발은 잘라서 상자 안에 둔다 -- 넘치면 옆 패널을 침범한다.
  const yF = (v) => mid - Math.max(-1, Math.min(1, v / span)) * half;

  let prevSec = null;
  secs.forEach((s) => {
    if (prevSec !== null && s - prevSec > SUPPLY_1S_GAP_SEC) {
      const g = document.createElementNS(NS, "rect");
      g.setAttribute("x", xAt(prevSec)); g.setAttribute("y", flowTop);
      g.setAttribute("width", Math.max(1, xAt(s) - xAt(prevSec)));
      g.setAttribute("height", flowH);
      g.setAttribute("fill", "var(--neutral)"); g.setAttribute("fill-opacity", "0.07");
      const t = document.createElementNS(NS, "title");
      t.textContent = "체결 기록 없음 " + (s - prevSec) + "초 -- 0이 아니라 «모름»이다";
      g.appendChild(t);
      svg.appendChild(g);
    }
    prevSec = s;
  });

  // ±span 눈금선. 절대 눈금이 이 화면의 요점이라 숫자를 축에 적는다.
  [span, -span].forEach((v) => {
    const g = document.createElementNS(NS, "line");
    g.setAttribute("x1", ml); g.setAttribute("x2", ml + cw);
    g.setAttribute("y1", yF(v)); g.setAttribute("y2", yF(v));
    g.setAttribute("stroke", "var(--line)"); g.setAttribute("stroke-opacity", "0.4");
    g.setAttribute("stroke-dasharray", "3 3");
    svg.appendChild(g);
    label(ml - 3, yF(v) + 3, (v > 0 ? "+" : "-") + qty(span), "var(--muted)", "end");
  });

  // 0선 기준 면적. 선 하나보다 «위냐 아래냐»가 훨씬 빨리 읽힌다. 고래만 칠한다 -- 셋 다
  // 칠하면 서로 가려서 되레 안 보인다.
  // ponytail: 0 교차점을 안 구하고 반대쪽을 0선으로 눌러 자른다. 오차는 최대 1초(≈1px)다.
  //           눈에 띄게 어긋나면 그때 교차점 보간으로 올린다.
  const area = (rows, yOf, color) => {
    let d = "", run = null;
    const close = () => {
      if (run !== null) d += " L" + xAt(run).toFixed(1) + " " + mid.toFixed(1) + " Z";
      run = null;
    };
    rows.forEach((r) => {
      if (brk(r.s, run, SUPPLY_1S_GAP_SEC)) {
        close();
        d += (d ? " " : "") + "M" + xAt(r.s).toFixed(1) + " " + mid.toFixed(1);
      }
      d += " L" + xAt(r.s).toFixed(1) + " " + yOf(r.v).toFixed(1);
      run = r.s;
    });
    close();
    const path = document.createElementNS(NS, "path");
    path.setAttribute("d", d); path.setAttribute("fill", color);
    path.setAttribute("fill-opacity", "0.28"); path.setAttribute("stroke", "none");
    svg.appendChild(path);
  };
  area(whale, (v) => Math.min(yF(v), mid), "var(--good)");
  area(whale, (v) => Math.max(yF(v), mid), "var(--bad)");

  // 봉 진행선. x축이 **봉 전체**라 아직 안 온 시간이 오른쪽에 비어 있는데, 그게 «데이터가
  // 없다»가 아니라 «아직 안 왔다»라는 걸 화면이 말해야 한다. 봉이 막 바뀐 직후엔 거의
  // 전부가 빈 상태라 이 선이 없으면 고장난 것처럼 보인다.
  {
    const nx = xAt(now);
    const g = document.createElementNS(NS, "line");
    g.setAttribute("x1", nx); g.setAttribute("x2", nx);
    g.setAttribute("y1", flowTop); g.setAttribute("y2", flowTop + flowH);
    g.setAttribute("stroke", "var(--ink)"); g.setAttribute("stroke-opacity", "0.35");
    g.setAttribute("stroke-dasharray", "2 3");
    svg.appendChild(g);
    const rest = document.createElementNS(NS, "rect");
    rest.setAttribute("x", nx); rest.setAttribute("y", flowTop);
    rest.setAttribute("width", Math.max(0, ml + cw - nx));
    rest.setAttribute("height", flowH);
    rest.setAttribute("fill", "var(--lift-solid, #8b949e)"); rest.setAttribute("fill-opacity", "0.05");
    svg.appendChild(rest);
  }

  const zero = document.createElementNS(NS, "line");
  zero.setAttribute("x1", ml); zero.setAttribute("x2", ml + cw);
  zero.setAttribute("y1", mid); zero.setAttribute("y2", mid);
  zero.setAttribute("stroke", "var(--line)");
  svg.appendChild(zero);

  const draw = (rows, width, opacity, name) => {
    const end = rows[rows.length - 1].v;
    const color = end >= 0 ? "var(--good)" : "var(--bad)";
    line(pathOf(rows, (r) => yF(r.v)), color, width, opacity);
    return { y: yF(end), color,
             text: name + " " + (end >= 0 ? "+" : "-") + qty(end) };
  };
  const tagR = draw(retail, 1.4, 0.5, "리테일");
  const tagW = draw(whale, 2, 0.95, "고래");
  const tags = [tagW, tagR];
  if (oiRows.length >= 2) {
    const end = oiRows[oiRows.length - 1].v;
    // 갱신 간격이 3~7초라 수급선의 5초 절단 기준을 그대로 쓰면 선이 조각난다. 20초를 넘게
    // 비면 그건 폴링 지각이 아니라 실제 공백이다.
    line(pathOf(oiRows, (r) => yF(r.v), 20), "var(--warn)", 1.6, 0.9);
    const tagO = { y: yF(end), color: "var(--warn)",
                   // 좁으면 «신규계약»(73px)이 꼬리표 자리(67px)를 넘는다 -- OI 로 줄인다.
                   text: (narrow ? "OI " : "신규계약 ")
                         + (end >= 0 ? "+" : "-") + qty(end) };
    tags.push(tagO);
  }
  // ── 청산 (2026-09-22 사용자 요청) ─────────────────────────────────────────
  // 🔴색을 새로 만들지 않는다. 3색 계약(초록·빨강·주황)이 이미 꽉 찼고, DESIGN.md 의 규칙이
  //   그대로 답이다: 「새 의미가 필요하면 색이 아니라 **형태·위치·라벨**로 가른다」.
  //   부호색은 다른 선과 같게 쓰고 **점선**으로 가른다.
  if (liqNet.length >= 2 && (liqSum[0] > 0 || liqSum[1] > 0)) {
    const end = liqNet[liqNet.length - 1].v;
    // 청산은 이벤트라 «없는 초»가 정상이다 -- 5초 절단을 쓰면 늘 조각난다. OI 와 같은 20초.
    line(pathOf(liqNet, (r) => yF(r.v), 20),
         end >= 0 ? "var(--good)" : "var(--bad)", 1.6, 0.95, "5 3");
    tags.push({ y: yF(end), color: end >= 0 ? "var(--good)" : "var(--bad)",
                // 순액만 적으면 «양쪽 다 터졌다»가 0 으로 보인다 -- 롱/숏을 같이 적는다.
                text: (narrow ? "청산 " : "청산 ")
                      + "롱" + qty(liqSum[0]) + "/숏" + qty(liqSum[1]) });
  }
  // 값이 가까우면 꼬리표가 그대로 포개진다(가격 라벨과 같은 문제).
  // 🔴짝지어 밀어내는 방식은 **셋에서 깨진다** -- 둘을 벌려도 셋째가 도로 그 자리에 앉는다.
  //   누적선 시절엔 셋이 5분 동안 벌어져서 안 보였는데, 30초 롤링은 셋이 동시에 0 근처인
  //   구간이 흔하다(2026-09-20 배포본 스크린샷에서 실제로 셋이 겹쳐 글자가 읽히지 않았다).
  //   y 로 정렬해 **차례로** 최소 간격을 주고, 아래로 넘치면 묶음째 위로 민다.
  const TAG_GAP = 12;
  tags.sort((a, b) => a.y - b.y);
  for (let i = 1; i < tags.length; i++) {
    if (tags[i].y - tags[i - 1].y < TAG_GAP) tags[i].y = tags[i - 1].y + TAG_GAP;
  }
  const over = tags[tags.length - 1].y - (h - 4);
  if (over > 0) tags.forEach((t) => { t.y -= over; });
  tags.forEach((t) => label(ml + cw + 5, t.y + 3, t.text, t.color));

  // 무엇을 보고 있는지 한 줄. 끝점 꼬리표가 곧 «이번 5분 순수급»이라 여기 숫자를 또 적지 않는다.
  label(ml + 2, mt - 5, "이번 5분봉 누적 순수급 ETH"
        + (narrow ? "" : "  ·  아래 풋프린트 봉과 같은 구간 · 다음 봉에서 0"), "var(--muted)");

  // 왼쪽은 이 봉이 열린 시각, 오른쪽은 닫힐 시각. 가운데에 진행 상황을 적는다 --
  // 「지금」이 오른쪽 끝이 아니라는 걸 분명히 해야 빈 오른쪽이 오해되지 않는다.
  const hhmm = (t) => { const d = new Date(t * 1000);
    return String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0"); };
  label(ml, h - 3, hhmm(first) + " 봉 시작", "var(--muted)");
  label(ml + cw, h - 3, hhmm(first + SUPPLY_1S_SEGMENT), "var(--muted)", "end");
  if (!narrow) label(xAt(now) + 4, h - 3, "지금 (" + (now - first) + "초 경과)", "var(--muted)");
}

// ── 가격축 수급 프로파일 (2026-09-19) ───────────────────────────────────────
// 시간을 버리고 가격만 남긴다. 창 전체(최대 24시간)를 가격빈으로 접어 «어느 값에서 누가
// 공격했는가»를 본다. 리본이 «지금»을 말한다면 이 화면은 «자리»를 말한다.
// 왼쪽이 공격적 매도, 오른쪽이 공격적 매수, 가운데가 가격이다. 한 막대는 가운데부터
// **고래 → 중형 → 리테일** 순으로 이어 붙인 세 토막이다(쌓기이지 겹치기가 아니다 --
// 겹쳐 그리면 가려진 토막의 길이를 눈으로 잴 수 없다). 큰 것부터 안쪽에 두는 이유는
// 가운데 선이 기준이라 거기서 출발하는 토막만 길이를 바로 읽을 수 있기 때문이다.
// 색은 셋 다 방향색(초록/빨강) 하나이고, 구분은 **농담**이다 -- 이 저장소의 색 계약이
// 초록·빨강·주황 셋뿐이라 「고래색」을 새로 만들 수 없다(styles.css 디자인 토큰 주석).
// ⚠️창은 프로세스가 살아 있는 동안만 찬다 -- 스냅샷에는 최근 12봉만 남긴다(server.py의
//   FOOTPRINT_KEEP_BARS 주석). 그래서 실제 창 길이를 머리글에 **항상** 적는다.


function renderSupplyProfileSvg(svg, profile, currentPrice, entryPrice = 0, box = null) {
  const NS = "http://www.w3.org/2000/svg";
  const mobileChart = isMobileChartMode();
  // 2026-09-19 2열 배치(사용자 지시)로 상자가 카드 폭의 68%/32% 가 됐다. 1200/400 을 고정으로
  // 두면 viewBox 가 상자보다 커서 meet 축소가 걸리고 글자가 그만큼 작아진다 -- 상자에서 받는다.
  // 하한은 «아직 레이아웃 전»(parentW/H = 0)일 때의 폴백이다.
  // box 가 오면 재지 않는다 -- 캔들 SVG 안의 중첩 <svg> 로 그릴 때 그 상자가 곧 좌표계다.
  // 🔴«재지 않는다»가 **측정을 건너뛴다**는 뜻이어야 한다. 값만 버리고 호출은 그대로 두면
  //   getBoundingClientRect 가 방금 만든 수천 노드의 레이아웃을 강제로 확정시킨다 -- 이 함수는
  //   캔들 렌더 한가운데서 불린다.
  const measured = box ? null : {
    w: svg.parentElement ? svg.parentElement.clientWidth : 0,
    h: svg.getBoundingClientRect().height
       || (svg.parentElement ? svg.parentElement.clientHeight : 0),
  };
  const w = box ? box.w : (measured.w > 0 ? Math.max(measured.w, 320) : 1200);
  const h = box ? box.h : (measured.h > 0 ? Math.max(measured.h, 260) : 400);
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.innerHTML = "";

  const levels = (profile && Array.isArray(profile.levels) ? profile.levels : [])
    .filter((l) => (Number(l[1]) || 0) + (Number(l[2]) || 0) > 0);
  if (!levels.length) {
    supplyProfileNow = null;   // 그릴 행이 없다 -- 옛 기하를 남기면 박스가 유령 행을 가리킨다
    const txt = document.createElementNS(NS, "text");
    txt.setAttribute("x", w / 2); txt.setAttribute("y", h / 2);
    txt.setAttribute("text-anchor", "middle"); txt.setAttribute("fill", "var(--muted)");
    txt.textContent = "체결 테이프 집계 중...";
    svg.appendChild(txt);
    return;
  }

  // 여백은 캔들 차트와 **같은 값**이다(2026-09-19 사용자 요청: 풋프린트와 같은 너비).
  // 머리글 세 줄(창 길이 · 지지/저항 · 벽)은 같은 날 사용자 지시로 걷어냈다 -- 그 세로
  // 공간을 행에 돌려줬다. 값은 전부 **툴팁**에 남아 있고, 경계($100k/<$10k)는 아래 범례가
  // 계속 적는다.
  // 여백도 상자 폭을 따른다 -- 2열의 오른쪽 열(≈400px)에서 45/112 는 폭의 40% 를 먹는다.
  const narrow = w < 760;
  // mr 은 띠 이름표가 앉는 자리다 -- 좁을 때 64 면 «리테일 매수»가 밖으로 나간다(2026-09-19
  // 모바일 실측). 78 로 늘려도 막대는 한쪽당 7px 만 잃는다.
  const ml = narrow ? 34 : 45, mr = narrow ? 78 : 112, mt = 14, mb = 22;
  const centerW = narrow ? 54 : 68;   // 굵고 커진 가격 라벨 자리(사용자 요청)
  const sideW = (w - ml - mr - centerW) / 2;
  const avail = h - mt - mb;
  const bucket = Number(profile.bucket) || 0.5;
  const prices = levels.map((l) => Number(l[0]));
  const minP = Math.min(...prices), maxP = Math.max(...prices);
  // 행은 «읽히는 높이»가 정한다 -- 가격 폭이 아니라. 창이 24시간까지 자라면 빈이 수천 개라
  // 격자 그대로 그리면 한 행이 0.1px 가 된다.
  const maxRows = Math.max(8, Math.floor(avail / (mobileChart ? 6 : 7)));
  const rowSize = Math.max(bucket, Math.ceil(((maxP - minP + bucket) / maxRows) / bucket) * bucket);

  const rows = new Map();
  levels.forEach((l) => {
    const key = Math.floor(Number(l[0]) / rowSize);
    const row = rows.get(key) || [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < 6; i++) row[i] += Number(l[i + 1]) || 0;
    rows.set(key, row);
  });
  const keys = [...rows.keys()].sort((a, b) => b - a);   // 위가 높은 가격
  const rowPx = avail / keys.length;
  // 🔴합산으로 바뀌었으니 정규화도 «행 총량»이어야 한다. max(buy,sell) 로 두면 막대가
  //   상자를 넘어간다(합계가 그 두 배까지 된다).
  const max = Math.max(...[...rows.values()].map((r) => r[0] + r[1]), 1e-9);
  // 2026-09-19 델타 띠(사용자 지시 B안): 합산 막대 **바깥**에 얇은 띠로 «테이커가 어느
  // 쪽이었나»를 되살린다. 라벨은 안 붙인다 -- 값은 툴팁에 있다.
  // ⭐합산이 «얼마나 거래됐나»(길이), 띠가 «누가 급했나»(색)로 축이 갈린다.
  // 🔴이건 지지·저항이 아니다. 저장소 실측: 3봉 연속 델타 편중 0.44x(**반예측적**),
  //   단일봉 테이커 서지만 2.75x 로 살아남았다 -- «쌓인 급함»은 신호가 아니다.
  const maxDelta = Math.max(...[...rows.values()].map((r) => Math.abs(r[0] - r[1])), 1e-9);
  const centerX = ml + sideW + centerW / 2;
  const leftEdge = ml + sideW, rightEdge = ml + sideW + centerW;
  let pocKey = null, pocVol = -1;
  rows.forEach((r, k) => { if (r[0] + r[1] > pocVol) { pocVol = r[0] + r[1]; pocKey = k; } });


  // 안쪽부터 고래 · 중형 · 리테일. 농담이 곧 크기 계단이다.
  const SEG_OPACITY = [0.95, 0.55, 0.28];
  const SEG_NAME = ["고래", "중형", "리테일"];
  const bar = (x, y, wid, color, opacity, tip) => {
    if (!(wid > 0)) return;
    const rect = document.createElementNS(NS, "rect");
    rect.setAttribute("x", x); rect.setAttribute("y", y);
    rect.setAttribute("width", Math.max(1, wid)); rect.setAttribute("height", Math.max(1, rowPx - 1));
    rect.setAttribute("fill", color); rect.setAttribute("fill-opacity", opacity);
    const title = document.createElementNS(NS, "title");
    title.textContent = tip;
    rect.appendChild(title);
    svg.appendChild(rect);
  };
  // 한 쪽(매수 또는 매도)을 세 토막으로 쌓는다. dir = +1 이면 오른쪽, -1 이면 왼쪽.
  const stack = (edge, dir, y, segs, total, color, side, price) => {
    let cursor = 0;
    segs.forEach((v, s) => {
      const wid = sideW * v / max;
      const x = dir > 0 ? edge + cursor : edge - cursor - wid;
      bar(x, y, wid, color, SEG_OPACITY[s],
        price.toFixed(1) + " · " + side + " " + SEG_NAME[s] + " " + v.toFixed(1) + " ETH ("
          + (total > 0 ? Math.round(v / total * 100) : 0) + "% · 합계 " + total.toFixed(1) + ")");
      cursor += wid;
    });
  };

  // 2026-09-19 «매수/매도 압력 구간»과 오른쪽 이름표·벽 틱을 걷어냈다(사용자 지시).
  // 프로파일이 VPVR 식 **합산**으로 바뀌면서 sign*(buy-sell) 기반 구간은 정의 자체가
  // 사라졌다 -- 방향은 봉별 델타가 말한다. 오른쪽 여백은 이제 비어 있다.

  keys.forEach((key, j) => {
    const [buy, sell, wBuy, wSell, rBuy, rSell] = rows.get(key);
    const y = mt + j * rowPx;
    const price = key * rowSize;
    // 2026-09-19 VPVR 식으로 **매수/매도를 합산**한다(사용자 지시).
    // ⭐근거: 체결에는 언제나 짝이 있다 -- 공격적 매수 N ETH 는 «누군가 그 값에 팔아줬다»와
    //   같은 말이다. 방향을 가르는 건 «누가 샀나»가 아니라 «테이커가 어느 쪽이었나»뿐이고,
    //   같은 수치가 규약에 따라 지지도 저항도 된다. 그래서 가격대별 **총 거래량**만 그린다.
    //   방향 정보는 봉별 델타(풋프린트 셀·하단 델타 숫자)가 이미 말한다.
    // 계층(고래/중형/리테일)은 남긴다 -- 그건 대칭 문제가 없는 진짜 분해다.
    const vol = buy + sell;
    stack(rightEdge, 1, y,
          [wBuy + wSell, Math.max(0, vol - (wBuy + wSell) - (rBuy + rSell)), rBuy + rSell],
          vol, "var(--amber)", "거래량", price);

    // 델타 띠 -- 합산 막대 바깥(오른쪽 여백 안쪽). 폭은 고정, **진하기가 크기**다.
    // 폭으로 크기를 말하면 옆의 거래량 막대와 같은 문법이 되어 둘이 헷갈린다.
    const delta = buy - sell;
    if (delta !== 0) {
      const dRect = document.createElementNS(NS, "rect");
      dRect.setAttribute("x", rightEdge + sideW + 2);
      dRect.setAttribute("y", y + 0.5);
      dRect.setAttribute("width", 5);
      dRect.setAttribute("height", Math.max(1, rowPx - 1));
      dRect.setAttribute("fill", delta > 0 ? "var(--good)" : "var(--bad)");
      dRect.setAttribute("opacity", (0.2 + 0.8 * Math.min(1, Math.abs(delta) / maxDelta)).toFixed(2));
      const dTip = document.createElementNS(NS, "title");
      dTip.textContent = price.toFixed(rowSize >= 1 ? 0 : 1) + " · 순델타 "
        + (delta > 0 ? "+" : "") + delta.toFixed(1) + " ETH"
        + " (공격적 매수 " + buy.toFixed(1) + " / 매도 " + sell.toFixed(1) + ")\n"
        + "⚠️테이커가 어느 쪽이었나일 뿐이다 -- 수동 쪽은 정확히 거울상이다.\n"
        + "⚠️같은 값을 반대로도 읽는다: 매수 델타가 큰데 가격이 안 오르면 흡수(수동 대량매도)다.";
      dRect.appendChild(dTip);
      svg.appendChild(dRect);
    }

    // 🔴행이 $1 보다 촘촘한데 toFixed(0) 로 찍으면 «2608, 2608» 처럼 같은 값이 두 줄 나온다
    //   (2026-09-19 첫 렌더에서 실제로 그랬다). 자릿수는 행 크기가 정한다.
    //   그래도 촘촘하면(9px 미만) 한 줄 걸러 찍는다 -- 글자가 겹치면 둘 다 못 읽는다.
    // 🔴글자를 키우면(사용자 요청) 행 간격보다 글자가 커져 붙는다 -- 한 줄 걸러 찍는
    //   기준을 9px 에서 폰트 크기에 맞춰 올린다. POC 행은 언제나 보인다.
    if (rowPx >= 13 || j % 2 === 0 || key === pocKey) {
      const lbl = document.createElementNS(NS, "text");
      lbl.setAttribute("x", centerX); lbl.setAttribute("y", y + rowPx / 2 + 3);
      lbl.setAttribute("text-anchor", "middle");
      // 2026-09-19 사용자 요청: 가운데 가격을 굵게 + 조금 더 크게.
      lbl.setAttribute("font-size", Math.min(13, Math.max(9, rowPx)));
      lbl.setAttribute("font-weight", key === pocKey ? "700" : "600");
      lbl.setAttribute("fill", key === pocKey ? "var(--text)" : "var(--neutral)");
      lbl.textContent = price.toFixed(rowSize >= 1 ? 0 : 1);
      svg.appendChild(lbl);
    }
  });

  // ── 벽 (2026-09-19 사용자 요청) ─────────────────────────────────────────
  // ⚠️여기서 「벽」은 **체결이 몰린 가격**이지 호가창에 걸린 대기 물량이 아니다. 이 화면의
  //   원천은 체결 테이프뿐이라 «걸려 있는 것»은 볼 수 없다. 그래서 이름표에 «체결»을 적고
  //   툴팁에 뜻을 풀어 둔다 -- 「벽」이라는 말이 오해를 부르기 가장 쉬운 자리다.
  // ⭐여기서도 «지지/저항» 단정은 하지 않는다(위 띠 주석과 같은 이유). 공격적 매도가 몰린
  //   가격은 누군가 받아냈다는 뜻이고 공격적 매수가 몰린 가격은 누군가 넘겼다는 뜻인데,
  //   둘 다 이미 체결된 사실이라 «지금 거기 뭐가 걸려 있나»를 말하지 않는다. 툴팁도 그대로.
  // 세기는 **«균등하게 퍼졌을 때의 몇 배»**로 잰다. 비율(%)로 두면 행 수에 따라 뜻이
  // 달라진다 -- 35행에서 8%는 2.8배지만 15행에서 8%는 1.2배다.
  // 🔴문턱을 데이터 보기 전에 정했다가 두 번 틀렸다(12% -> 3배). 2026-09-19 실측(ETH 55분,
  //   35행): 고래매도 2.7배 · 고래매수 2.0배 · 리테일 양쪽 1.9배, 전체 물량조차 1.8배다.
  //   가격이 머문 자리에 물량이 몰리는 건 기본값이라 «3배»는 실제로 거의 안 나온다.
  //   그래서 문턱은 **1.5배**로 낮게 두고, 대신 **배수를 이름표에 그대로 적어** 세기를
  //   사람이 판단하게 한다. 문턱은 「이름표를 붙일 가치」 선이지 「진짜 벽」 선이 아니다.
  const WALL_MIN_MULT = 1.5;

  // 범례. 농담 세 단계는 설명 없이는 안 읽힌다 -- 견본을 같이 놓는다.
  const kUsd = (v) => "$" + Math.round(v / 1000) + "k";
  // 🔴범례와 아래 바닥 설명은 **같은 줄**(h - 6)에 왼쪽·오른쪽으로 앉는다. 상자가 좁으면
  //   둘이 겹쳐 글자가 서로를 덮는다(2026-09-19 모바일 실측 -- 사용자 보고). 좁을 때는
  //   범례에서 달러 경계를 떼고(막대 툴팁에 그대로 있다) 바닥 설명도 줄인다.
  const legend = narrow
    ? [["고래", ""], ["중형", ""], ["리테일", ""]]
    : [
      ["고래", "≥" + kUsd(profile.whaleMinUsd || 0)],
      ["중형", kUsd(profile.retailMaxUsd || 0) + "~" + kUsd(profile.whaleMinUsd || 0)],
      ["리테일", "<" + kUsd(profile.retailMaxUsd || 0)],
    ];
  // 2026-09-20 사용자 지시: 범례와 요약 숫자의 **자리를 맞바꾼다**(범례 오른쪽 · 숫자 왼쪽).
  // 🔴범례를 오른쪽 끝에 붙이려면 폭을 **먼저** 알아야 한다 -- 아래 루프가 쓰는 증가폭과
  //   같은 식으로 미리 합산한다(식이 둘로 갈리면 한쪽만 고치고 어긋난다).
  const legW = (item) => 13 + (item[0].length + item[1].length) * 6.2 + 16;
  const legendTotal = legend.reduce((acc, it) => acc + legW(it), 0) - 16;  // 마지막 여백 제외
  const legendStart = Math.max(ml, w - mr - legendTotal);
  let lx = legendStart;
  legend.forEach(([name, range], s) => {
    const sw = document.createElementNS(NS, "rect");
    sw.setAttribute("x", lx); sw.setAttribute("y", h - 14);
    sw.setAttribute("width", 9); sw.setAttribute("height", 9);
    // 🔴막대가 --amber(합산)로 바뀌었다. 견본이 초록이면 범례가 다른 그림을 설명한다.
    sw.setAttribute("fill", "var(--amber)"); sw.setAttribute("fill-opacity", SEG_OPACITY[s]);
    svg.appendChild(sw);
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", lx + 13); t.setAttribute("y", h - 6);
    t.setAttribute("font-size", "9"); t.setAttribute("fill", "var(--muted)");
    t.textContent = range ? name + " " + range : name;
    svg.appendChild(t);
    lx += 13 + (name.length + range.length) * 6.2 + 16;
  });
  // 🔴길이로 고른다, `narrow` 로 고르지 않는다 -- 임계값(760)과 «실제로 들어가느냐»는 다른
  //   물음이라, 그걸로 나누면 768px 같은 폭에서 긴 문장이 범례를 덮는다(계산으로 확인).
  //   긴 것 -> 짧은 것 -> 생략 순으로 내려간다. 내용은 전부 툴팁에도 있다.
  // 2026-09-19 방향을 말하던 문구를 버렸다 -- 합산이라 좌우가 방향이 아니라 **원천**이다.
  // 2026-09-20 사용자 요청: 서버가 매 폴링 계산해 보내는 요약 스칼라 8개가 **전부 버려지고
  //   있었다**(app.js 가 window_s 하나만 썼다). 그 자리에 숫자를 넣고, 안내 문구는 툴팁으로
  //   내린다 -- 문구는 한 번 읽으면 끝이고 숫자는 매초 바뀐다.
  // 🔴obi 는 막대가 못 담는 유일한 축이다. 프로파일은 abs(qty) 로 그려 **매수/매도를 버린다**.
  //   「위=매도·아래=매수」로 눈대중할 수는 있지만(실측 지금 이 순간 100%/99.5%), 창이
  //   1~4시간이라 그동안 가격이 지나간 가격대는 창 안에서 측면이 뒤집힌다 -- 실측 1h 8% ·
  //   2h 13% · **4h 33%**. 그 행들의 peak/refill 은 두 측면이 섞여 있어 위치로 복원이 안 된다.
  const sm = latestFlowHeatmap && latestFlowHeatmap.summary;
  const pct1 = (v) => (v == null ? "—" : Math.round(100 * v) + "%");
  // 🔴좁으면 **뒤에서부터 덜어낸다**. 예전에 여기 안내문이 범례를 덮어 글자가 겹쳤다
  //   (2026-09-19 모바일, 사용자 보고). 폭 판정은 `narrow` 임계값이 아니라 «실제로
  //   들어가는가»로 한다 -- 768px 같은 폭에서 임계값만 보면 또 겹친다.
  // 변동 신호는 **맨 앞**이다 -- 뒤에서부터 덜어내므로 좁은 화면에서 마지막까지 남는다.
  // 없으면 «—» 가 아니라 아예 안 적는다(옛 서버 · 워밍업 중이면 자가 없다).
  const footParts = sm
    ? [...(sm.vol_pct == null ? [] : ["변동 " + sm.vol_pct + "%"]),
       "불균형 " + (sm.obi == null ? "—" : (sm.obi > 0 ? "+" : "") + sm.obi.toFixed(2)),
       "지속 " + pct1(sm.persist_share),
       "이탈 " + pct1(sm.offtouch_leave_share),
      ]
    : ["← 호가  ·  거래량 →"];
  // 숫자는 왼쪽(ml)에서 시작해 **범례 시작점 앞까지**만 쓴다. 넘치면 뒤에서부터 덜어낸다.
  const footFits = (t) => ml + t.length * 6.2 < legendStart - 8;
  let footText = footParts.join("  ·  ");
  while (footParts.length > 1 && !footFits(footText)) {
    footParts.pop();
    footText = footParts.join("  ·  ");
  }
  if (!footFits(footText)) footText = "";
  {
    const foot = document.createElementNS(NS, "text");
    foot.setAttribute("x", ml); foot.setAttribute("y", h - 6);
    foot.setAttribute("font-size", "9"); foot.setAttribute("fill", "var(--muted)");
    foot.textContent = footText;
    const ft = document.createElementNS(NS, "title");
    ft.textContent = sm
      ? "왼쪽 = 걸려 있는 호가(길이 = 지금 걸린 양 · 진할수록 이 창에서 자꾸 다시 깔린 것)\n"
        + "오른쪽 = 체결 거래량(매수+매도) · 바깥 띠 = 순델타(초록 매수 · 빨강 매도)\n\n"
        + `불균형(OBI) ${sm.obi} — 현재가 ±${sm.obi_band_pct}% 안에서 (매수−매도)/(매수+매도).\n`
        + "  +면 매수호가가 두껍다. 🔴밴드가 값을 정한다(실측 ±0.1% +0.504 vs ±2% +0.071, 7배).\n"
        + "  🔴막대는 매수·매도를 합쳐 그리므로 이 축은 숫자로만 있습니다. 「위=매도·아래=매수」로\n"
        + "  눈대중할 수 있지만, 이 창 안에서 측면이 뒤집힌 가격대가 있습니다(4h 탭 실측 33%).\n"
        + `지속 ${pct1(sm.persist_share)} — 창 내내 한 번도 안 빠진 양이 지금 걸린 양의 몇 %인가.\n`
        + `이탈 ${pct1(sm.offtouch_leave_share)} — 사라진 호가 중 **체결 없이** 빠진 비율`
        + ` (${sm.fill_source ? "풋프린트 대조" : "대조 불가"} · 판정 가능 ${sm.offtouch_bins}칸).\n`
        + "  🔴«취소율»이 아닙니다 — 터치 구간은 구조적으로 빠져 있고, 취소와 리프라이싱을\n"
        + "  가를 수 없습니다(선물 WS 는 레벨별 총량만 주고 주문 ID 가 없습니다).\n"
        // 2026-09-20 «벽»을 요약에서 뺐다(사용자 요청). 지지·저항이 아니라는 걸 실측으로
        //   확인한 뒤(5.8일 71,293건 · 반등률 0.509 = 동전 · 크기 사분위 0.502/0.510/0.520/0.503)
        //   화면에 남겨두면 「여기서 멈춘다」로 읽히기만 한다. 큰 호가는 막대 길이로 이미 보인다.
        + `기준가 ${sm.spot} · 창 ${Math.round(sm.window_s / 60)}분`
        // 2026-09-20 「벽이 지지·저항이 아니면 이 화면은 뭘 말하나」에 답한다. 화면에 있는
        //   축 전부를 앞으로의 가격과 맞댄 결과다(60초마다 한 표본 · 7,676개 · 독립 일수 7 ·
        //   일자 블록 부트스트랩). 방향은 전부 CI 가 0 을 품고, 움직임 «크기»만 남았다.
        + (sm.vol_pct == null ? ""
           : `\n변동 ${sm.vol_pct}% — 「지금이 최근 4시간 중 몇 분위로 시끄러운가」.\n`
             + `  재료 둘: 직전 300초 실현변동 ${sm.vol_past_pct}% + 재깔림 ${sm.vol_refill_pct}%`
             + ` (자 ${sm.vol_ref_n}표본 · 60초 간격).\n`
             + "  실측 5분위별 실제 |수익률|(300초 뒤) 중앙: 6.0 → 7.2 → 8.2 → 8.7 → 11.5bp.\n"
             + "  상위20%는 하위33%의 **1.77배** 흔들렸습니다(7,864표본 · 독립 일수 7).\n"
             + "  🔴주역은 직전 실현변동입니다(그것만으로 1.63배). 호가(재깔림)가 더한 몫은\n"
             + "  +0.14배인데 CI [−0.01, +0.30] 으로 **0 을 포함**합니다 — 상관에서는 섰지만\n"
             + "  (부분ρ +0.139) 배수에서는 못 섰습니다.\n"
             + "  🔴호가만으로 만든 신호는 전부 최상위 분위에서 꺾입니다(재깔림 단독 Q4 12.0 →\n"
             + "  Q5 8.4bp). 그래서 재깔림 단독도, 블록·지속을 섞은 합성도 안 씁니다.\n"
             + "  🔴방향은 말하지 않습니다. 크기만입니다.\n")
        + "\n\n■ 이 패널의 쓰임 — 「어디서 멈출까」가 아니라 「얼마나 흔들릴까」입니다.\n"
        + "  방향: OBI +0.033 · 60초Δ +0.004 — 둘 다 신뢰구간이 0 을 품습니다(= 못 말합니다).\n"
        + "  크기(|수익률|과의 상관): 재깔림 +0.165 · 블록 +0.124 · 지속률 −0.111,\n"
        + "  셋 다 0 을 배제하고 직전 300초 실현변동을 통제해도 +0.139/+0.093/−0.076 로 남습니다.\n"
        + "  → 방향을 고르는 도구가 아니라 **크기·손절폭·대기 여부**를 정하는 도구입니다.\n"
        + "  🔴독립 일수 7 · 홀드아웃 없음 — 「예측한다」가 아니라 「5.8일 이 데이터에서\n"
        + "  이렇게 보였다」입니다. rho 0.14 는 약한 실재이지 그 자체로 엣지가 아닙니다."
        + (() => {                       // 🔴hm 은 아래에서 선언된다(TDZ) -- 여기선 원본을 직접 본다
             const ap = latestFlowHeatmap && latestFlowHeatmap.rows
                        && latestFlowHeatmap.rows.approach;
             if (!ap) return "";
             const fin = [...ap].filter(Number.isFinite);
             return ` · 접근행동 자격 ${fin.length}행 (◌ 표식 ${fin.filter((v) => v < 0.8).length}개)`;
           })()
      : "← 걸려 있는 호가  ·  체결 거래량 →";
    foot.appendChild(ft);
    svg.appendChild(foot);
  }

  // ── 현재가 (2026-09-19 사용자 요청) ────────────────────────────────────
  // 점선 한 줄이었다. 바꾼 이유: 프로파일에서 제일 자주 보는 건 «내가 지금 어느 행에
  // 서 있나»인데, 점선은 행을 **가리키기만** 하고 그 행의 가격은 다른 글자들과 같은
  // 크기라 눈으로 찾아야 했다. 이제 띠가 그 행을 덮고 값을 크게 적는다.
  //
  // 🔴맨 마지막에 붙인다 -- 칩이 그 행의 작은 가격 라벨을 **가려야** 같은 값이 두 번
  //   겹쳐 보이지 않는다. SVG 는 뒤에 붙은 것이 위에 칠해진다.
  // 🔴프로파일 전체는 5초마다 다시 그려지는데 현재가는 초당 수십 번 바뀐다. 전체를 다시
  //   그리면 비싸고 움직임도 끊긴다. 그래서 **기하만 적어 두고**(supplyProfileNow) 체결
  //   WS 가 아래 updateSupplyProfileNow 로 직접 와서 transform 만 바꾼다. 행에서 행으로
  //   미끄러지는 건 CSS transition(.supply-now)이 잇는다.
  // 띠는 **칩만큼**은 높아야 한다. 행이 11px 인데 칩이 22px 이면 칩이 이웃 행의 가격
  // 라벨을 반만 덮어 «글자가 잘린» 것처럼 보인다 -- 띠가 그만큼 크면 덮인 자리가
  // «강조 구간 안»으로 읽힌다.
  // 🔴2026-09-19 3차. 처음엔 점선, 다음엔 «띠 + 테두리 칩 + 15px 숫자»였는데 사용자가
  //   "너무 정신이 없다". 셋이 동시에 움직이니 시선이 거기 묶인다. 게다가 **값은 중복**이다
  //   -- 가운데 열이 모든 행의 가격을 이미 적고 있다. 필요한 건 «어느 줄인가» 하나다.
  //   그래서 그 줄의 가격 글자 좌우에 화살표만 둔다(사용자 제안). 차트 위에 덮는 것도,
  //   가리는 글자도, 새로 읽을 숫자도 없다.
  // ── 진입가 (2026-09-19 사용자 요청) ────────────────────────────────────
  // 현재가는 «어느 줄인가»만 화살표로 가리킨다(바로 위 주석의 3차 결론). 진입가는 체결
  // 전까지 안 움직이고 읽는 목적도 달라서(«내 값이 이 분포의 어디인가») 얇은 가로선으로
  // 긋는다. 가운데 가격 열은 비우고 좌우 막대 구간만 -- 그 열의 숫자를 덮으면 현재가에서
  // 겪은 문제를 되풀이한다.
  // 🔴창 밖이면 가장자리에 **붙이지 않는다**(현재가와 같은 규약). 이 축은 체결이 있었던
  //   값만 있어서 붙이는 순간 «진입가가 저기 있다»는 거짓말이 된다 -- 대신 위/아래 어느
  //   쪽인지와 값만 여백에 적는다.
  if (entryPrice > 0) {
    const ej = keys.indexOf(Math.floor(entryPrice / rowSize));
    const eLbl = (x, y, text, anchor) => {
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", x); t.setAttribute("y", y); t.setAttribute("font-size", "10");
      t.setAttribute("font-weight", "bold"); t.setAttribute("fill", "var(--amber)");
      if (anchor) t.setAttribute("text-anchor", anchor);
      t.textContent = text;
      svg.appendChild(t);
    };
    if (ej >= 0) {
      const ey = mt + ej * rowPx + rowPx / 2;
      [[ml, leftEdge - 2], [rightEdge + 2, w - mr]].forEach((seg) => {
        const ln = document.createElementNS(NS, "line");
        ln.setAttribute("x1", seg[0]); ln.setAttribute("x2", seg[1]);
        ln.setAttribute("y1", ey); ln.setAttribute("y2", ey);
        ln.setAttribute("stroke", "var(--amber)"); ln.setAttribute("stroke-width", "1.5");
        svg.appendChild(ln);
      });
      eLbl(ml - 5, ey + 3, "진입", "end");
      // 🔴좁은 상자에서는 값을 안 적는다 -- 오른쪽 여백은 띠 이름표 넷이 이미 쓰고 있어서
      //   진입 행이 그중 하나와 같은 높이면 글자가 겹친다. 값은 바로 위 가격 플롯의 진입
      //   배지에 그대로 있다(같은 SVG 안이다).
      if (!narrow) eLbl(w - mr + 6, ey + 3, fmtNum(entryPrice, 1));
    } else if (!narrow) {
      // 창 밖. 좁을 때는 아예 안 적는다 -- 위 가격 플롯이 ↑/↓ 배지로 이미 말하고 있고,
      // 여기 top/bottom 자리는 띠 이름표의 첫·마지막 줄과 겹친다.
      const above = entryPrice > (keys[0] + 1) * rowSize;
      eLbl(w - mr + 6, above ? mt + 8 : h - mb - 2,
           `진입 ${above ? "↑" : "↓"} ${fmtNum(entryPrice, 1)}`);
    }
  }

  supplyProfileNow = { svg, mt, rowPx, rowSize, keys, pocKey, digits: rowSize >= 1 ? 0 : 1 };

  // ── 왼쪽 = 호가(걸려 있는 양) ─────────────────────────────────────────
  // 2026-09-19 사용자 지시. 오른쪽이 «체결»(과거·취소 불가)이고 여기는 «호가»(현재·언제든
  // 취소)다 -- 다른 물건이라 색을 가른다(파랑). 같은 행에서 둘을 나란히 읽는 게 목적이다.
  // ⭐2026-09-20 축을 둘로 갈랐다: **길이 = 지금 걸린 양**, **농도 = 재깔림**(창 안에서
  //   제 크기의 몇 배가 다시 깔렸나). 그전에는 길이가 «창 내내 안 빠진 양»(min) 하나였다.
  // 🔴min 을 그린다는 건 넷 중 가장 작은 걸 그린다는 뜻이었다. 실측 600초(ETH 247빈):
  //   sum(min) 103,531 < sum(now) 170,775 < sum(max) 310,024 << **sum(refill) 1,029,281**.
  //   걸린 양의 6배가 창 안에서 회전하는데(refill/drain = 1.00) 화면에 그 축이 없었다.
  //   더 나쁜 건 **순위가 뒤집힌다**는 것이다: -2.05% 빈은 1,780 ETH 블록을 10분에 18번
  //   재호가하는데(증분 p90 1,750 = 지터가 아니라 덩어리) min 이 675 라 짧은 막대였고,
  //   한 번 깔고 앉은 +1.94% 빈(refill 641)이 2,610 으로 더 길었다.
  // 🔴min 은 터치 근처를 구조적으로 지운다 -- pers/peak 중앙값이 0~0.1% 에서 **0.03**,
  //   0.5~1% 에서 0.42 다. 길이를 inst 로 바꾸면 그 눈멂이 같이 없어진다.
  // 🔴«지지·저항»이 아니다 -- 2026-09-20 에 **실제로 쟀고 없었다**. 5.8일 71,293건에서
  //   벽에 닿은 뒤 반등률 0.509(동전), 크기 사분위 0.502/0.510/0.520/0.503 로 순서조차 없고,
  //   같은 크기 안에서 지속률 상위−하위 +0.001(섞기 귀무 z=0.22). 판정폭 $1.5→$5 ·
  //   지평 5→15분에서도 같다(z=−0.45). 그래서 이 화면은 **서술만** 한다.
  const hm = latestFlowHeatmap && latestFlowHeatmap.rows;
  if (hm && hm.bin_size > 0 && keys.length) {
    const at = (px) => Math.round(px / hm.bin_size) - hm.bin_lo;
    let maxI = 0;
    const per = keys.map((k) => {
      const v = { inst: 0, pers: 0, peak: 0, refill: 0, d60: 0, blk: 0, n_up: 0 };
      for (let q = 0; q < rowSize; q += hm.bin_size) {
        const i = at(k * rowSize + q);
        if (i >= 0 && i < hm.inst.length) {
          v.inst += hm.inst[i]; v.pers += hm.pers[i]; v.peak += hm.peak[i];
          v.refill += hm.refill[i]; v.d60 += hm.d60[i];
          // 🔴blk 은 **더하지 않는다** -- 크기이지 양이 아니다. 한 행이 여러 빈을 덮으면
          //   그중 가장 큰 덩어리를 그 행의 «단위»로 본다. 횟수는 더한다.
          if (hm.blk && hm.blk[i] > v.blk) v.blk = hm.blk[i];
          if (hm.n_up) v.n_up += hm.n_up[i];
        }
      }
      if (v.inst > maxI) maxI = v.inst;
      v.rw = v.refill / Math.max(v.peak, 1e-9);
      return v;
    });
    // 🔴2026-09-20 농도를 **창 안 순위**로 칠한다. 절대 곡선(log2(rw)/2.6)으로 칠했더니
    //   탭마다 죽었다 -- 실측 포화율 15m 11% · 1h 67% · 2h 90% · **4h 97%**(sd 0.037).
    //   refill 은 창에 비례해 쌓이는데 나는 15분 창에서 보정했다. 창 길이로 나누는 것도
    //   답이 아니다 -- peak 도 같이 커져 단순 비례가 아니고, 그렇게 하면 4h 에서 rw 중앙이
    //   0.79 라 전부 최저 농도가 된다(sd 0.015). 양쪽 다 «축이 없는» 상태다.
    //   순위는 탭과 무관하게 잉크 범위를 다 쓴다. 대신 농도는 **이 창 안에서의 상대값**이고
    //   절대 배수는 툴팁이 숫자로 말한다(조용한 시간과 시끄러운 시간이 같아 보이는 것이
    //   이 선택의 대가다).
    const liveRw = per.filter((v) => v.inst > 0).map((v) => v.rw).sort((a, b) => a - b);
    const rwPct = (x) => {            // 0~1 분위. 같은 값이 여럿이면 가운데를 준다.
      if (liveRw.length < 2) return 0.5;
      let lo = 0, hi = liveRw.length;
      while (lo < hi) { const m = (lo + hi) >> 1; if (liveRw[m] < x) lo = m + 1; else hi = m; }
      let hi2 = lo;
      while (hi2 < liveRw.length && liveRw[hi2] === x) hi2++;
      return ((lo + hi2) / 2) / liveRw.length;
    };
    if (maxI > 0) {
      keys.forEach((k, j) => {
        const { inst, pers, peak, refill, d60, blk, n_up } = per[j];
        if (inst <= 0) return;
        const bl = (inst / maxI) * (sideW - 2);
        // 재깔림 = refill/peak. 농도는 그 값의 **창 안 분위**다(위 주석).
        const rw = refill / Math.max(peak, 1e-9);
        const r = document.createElementNS(NS, "rect");
        r.setAttribute("x", leftEdge - 1 - bl); r.setAttribute("y", mt + j * rowPx + 0.5);
        r.setAttribute("width", Math.max(1, bl)); r.setAttribute("height", Math.max(1, rowPx - 1));
        r.setAttribute("fill", "#7dd3fc");
        r.setAttribute("opacity", (0.22 + 0.78 * rwPct(rw)).toFixed(2));
        // ── 접근행동(사용자 요청 2026-09-20) ────────────────────────────
        // 「가격이 다가왔을 때 이 가격대가 얇아졌나」. 자격 빈이 전체의 28%뿐이라
        // 막대 색·길이 같은 **행 채널로는 못 쓴다**(72%가 빈칸이면 «얇다»와 «모른다»가
        // 같은 그림이 된다). 그래서 얇아진 행에만 작은 고리를 얹는다.
        // 🔴0.8 은 임의값이 아니라 **실측 분포에서 잡은 선**이다(정규화 후 p10 0.66 ·
        //   p50 1.22 · shrink<0.8 이 16%). 아무것도 안 걸리는 날은 표식이 0개여도
        //   맞다 -- 분위로 고정하면 «언제나 몇 개»가 나와 거짓 존재감을 만든다.
        // ⭐2026-09-20 검증함(ETH 5.2일·4시간 창 31개): 얇아진 가격대 비율이 창마다
        //   9~20%로 안정적이고, 인접 창 지속성 rho +0.262±0.026(양수 28/30)이 회전
        //   대조군 +0.067±0.032 를 4.7 SE 로 앞선다. 표식 지속률 25% vs 기저 15%.
        //   ⇒ 표식은 잡음이 아니라 그 가격대의 성질이다.
        // 🔴그래도 스푸핑 «판정»이 아니다. 반대 방향(다가오면 두꺼워짐)이 46%로 더 흔하고,
        //   표식의 75%는 4시간 뒤 사라지며, **가격 예측력은 재지 않았다**. 툴팁도 그렇게 적는다.
        const apr = approachAt(hm, k * rowSize, rowSize);
        if (apr !== null && apr < 0.8) {
          const mk = document.createElementNS(NS, "circle");
          mk.setAttribute("cx", Math.max(ml + 4, leftEdge - 1 - bl - 4));
          mk.setAttribute("cy", mt + j * rowPx + rowPx / 2);
          mk.setAttribute("r", Math.min(3, Math.max(1.8, rowPx / 4)));
          mk.setAttribute("fill", "none");
          mk.setAttribute("stroke", "#7dd3fc");
          mk.setAttribute("stroke-width", "1.2");
          svg.appendChild(mk);
        }
        const t = document.createElementNS(NS, "title");
        const winMin = latestFlowHeatmap && latestFlowHeatmap.summary
          ? Math.round(latestFlowHeatmap.summary.window_s / 60) : 0;
        const win = winMin ? winMin + "분 창" : "창";
        t.textContent = "호가 " + (k * rowSize).toFixed(rowSize >= 1 ? 0 : 1)
          + " — 지금 걸린 양 " + Math.round(inst) + " ETH"
          + " · " + win + " 최대 " + Math.round(peak) + " · 내내 남은 것 " + Math.round(pers)
          + " (" + Math.round(100 * pers / Math.max(inst, 1e-9)) + "%)\n"
          + "재깔림 " + rw.toFixed(1) + "배 — " + win + " 안에서 최대치의 "
          + rw.toFixed(1) + "배(" + Math.round(refill) + " ETH)가 다시 깔렸습니다. "
          + "이 창의 상위 " + Math.round(100 * (1 - rwPct(rw))) + "% 입니다"
          + (blk > 0
             // 2026-09-20 «단위». 같은 배수라도 「1,780 ETH 를 18번」과 「15 ETH 를 2,000번」은
             // 전혀 다른 행동인데 농도로는 구별이 안 된다. 실측(1,960행) rho(배수, 블록/peak)
             // = 0.333 이고 같은 배수 구간 안에서 40배까지 갈린다 = 별개 축이다.
             // 🔴blk 은 «물량 가중 중앙값»이다 -- 개수 기준 분위는 잔물결에 묻힌다.
             ? "\n(재깔림이 높던 국면은 이후 더 «크게» 움직였습니다 — 방향은 아닙니다.\n"
               + " 바닥 줄 툴팁에 근거가 있습니다.)"
               + "\n단위: " + fmtNum(blk, blk >= 100 ? 0 : 1) + " ETH 씩 "
               + Math.round(n_up) + "번"
               + (blk / Math.max(peak, 1e-9) >= 0.35
                  ? " — 한 덩어리를 같은 자리에 계속 다시 까는 중입니다(작업자 한 명일 수 있습니다)."
                  : " — 잘게 나눠 계속 깔립니다(알고리즘 잔물결).")
             : "")
          + " — 🔴농도는 **이 창 안의 상대 순위**라, 조용한 시간과 시끄러운 시간이 같은 "
          + "진하기로 보입니다. 절대값은 이 숫자로 보세요.\n"
          + "최근 60초 " + (d60 >= 0 ? "+" : "") + Math.round(d60) + " ETH — "
          + (Math.abs(d60) < 1 ? "변화 없음" : d60 > 0 ? "쌓는 중" : "빼는 중") + "\n"
          + (apr === null ? ""
             : "접근행동 " + apr.toFixed(2) + "배 — 가격이 이 근처(0.35% 안)에 왔을 때 "
               + "같은 거리 평균 두께의 " + apr.toFixed(2) + "배였습니다(최근 4시간). "
               + (apr < 0.8 ? "◌ 표식: 다가오면 얇아진 쪽입니다."
                            : apr > 1.25 ? "다가오면 두꺼워진 쪽입니다." : "거의 그대로입니다.")
               + "\n검증(ETH 5.2일·4시간 창 31개): 이 성질은 창마다 안정적이고"
               + "(얇아진 가격대 9~20%), 지금 표식이 붙은 가격대는 4시간 뒤에도 표식일 확률이 "
               + "25%로 평균 15%의 1.7배입니다. 지속성 rho +0.26 vs 회전 대조군 +0.07.\n"
               + "🔴그래도 «스푸핑» 판정이 아닙니다 — 같은 창에서 반대 방향(다가오면 두꺼워짐)이 "
               + "46%로 더 흔하고, 표식의 75%는 4시간 뒤 사라지며, **가격이 어디로 갈지는 "
               + "재지 않았습니다**.\n")
          + "⚠️체결이 아니라 **지금 걸려 있는** 지정가다 -- 언제든 취소될 수 있다.\n"
          + "⚠️지지·저항이 아닙니다 — 5.8일 71,293건에서 벽에 닿은 뒤 반등률이 0.509 로\n"
          + "   동전이고, 크기·지속 어느 쪽도 예측하지 못했습니다(2026-09-20 실측).";
        r.appendChild(t);
        svg.appendChild(r);
      });
    }
  }

  const nowG = document.createElementNS(NS, "g");
  nowG.setAttribute("class", "supply-now");
  // 🔴화살표만 두면 **라벨이 없는 행**을 가리킬 수 있다. 행이 13px 보다 촘촘하면 위에서
  //   한 줄 걸러 찍기 때문이다(그러지 않으면 글자가 겹친다). 그래서 그룹이 그 줄의 가격을
  //   **직접** 들고 다닌다 -- 위 라벨과 좌표·크기·내용이 같아 있으면 정확히 포개지고,
  //   없으면 이것이 그 줄의 라벨이 된다.
  // 🔴굵기까지 **똑같아야** 한다. 700 으로 올렸더니 글자 폭이 달라져(가운데 정렬이라 각
  //   자리가 어긋난다) 겹친 두 글자가 고스팅으로 번졌다. 구별은 **색**으로만 한다.
  const nowTxt = document.createElementNS(NS, "text");
  nowTxt.setAttribute("x", centerX); nowTxt.setAttribute("y", 3);
  nowTxt.setAttribute("text-anchor", "middle");
  nowTxt.setAttribute("font-size", Math.min(13, Math.max(9, rowPx)));
  nowTxt.setAttribute("fill", "var(--text)");
  nowG.appendChild(nowTxt);
  // 가운데 열 [leftEdge, rightEdge] 안쪽에 둔다 -- 막대 위로 넘어가지 않는다.
  [`${leftEdge + 1},-4 ${leftEdge + 7},0.5 ${leftEdge + 1},5`,
   `${rightEdge - 1},-4 ${rightEdge - 7},0.5 ${rightEdge - 1},5`].forEach((pts) => {
    const a = document.createElementNS(NS, "polygon");
    a.setAttribute("points", pts);
    a.setAttribute("fill", "var(--accent)");
    nowG.appendChild(a);
  });
  svg.appendChild(nowG);
  updateSupplyProfileNow(currentPrice);
}

// 접근행동 값을 **절대 가격**으로 찾는다. 4시간 창이라 격자가 위 다섯 배열과 다르다.
// 한 행이 여러 빈을 덮으면 **가장 얇아진 값**을 쓴다 -- 비를 평균내면 «한 빈만 빠졌다»가
// 이웃에 묻힌다. 값이 없는 빈(NaN)은 건너뛴다(0 으로 읽으면 «완전히 빠졌다»가 된다).
function approachAt(hm, price, rowSize) {
  if (!hm || !hm.approach || !(hm.approach_bin_size > 0)) return null;
  let best = null;
  for (let q = 0; q < rowSize; q += hm.approach_bin_size) {
    const i = Math.round((price + q) / hm.approach_bin_size) - hm.approach_bin_lo;
    if (i < 0 || i >= hm.approach.length) continue;
    const v = hm.approach[i];
    if (Number.isFinite(v) && (best === null || v < best)) best = v;
  }
  return best;
}

// 현재가 박스를 제 행으로 옮긴다. 프로파일을 통째로 다시 그리지 않는 **유일한** 경로다.
// (그려 둔 기하는 renderSupplyProfileSvg 가 supplyProfileNow 에 적어 둔다.)
function updateSupplyProfileNow(price) {
  const g = supplyProfileNow;
  if (!g || !g.svg.isConnected) return;
  const el = g.svg.querySelector(".supply-now");
  if (!el) return;
  // 🔴창 밖이면 **숨긴다**. 가장자리에 붙여 두면 «현재가가 저기 있다»는 거짓말이 된다
  //   (캔들 차트는 축을 다시 잡으니 붙여도 되지만, 여기 축은 체결이 있었던 값뿐이다).
  const j = price > 0 ? g.keys.indexOf(Math.floor(price / g.rowSize)) : -1;
  if (j < 0) { el.setAttribute("opacity", "0"); return; }
  el.setAttribute("opacity", "1");
  el.setAttribute("transform", `translate(0 ${g.mt + j * g.rowPx + g.rowPx / 2})`);
  // 🔴적는 건 «틱 가격»이 아니라 **그 행의 가격**이다. 위아래 이웃과 같은 자에서 읽혀야
  //   한 열로 보인다 -- 2631.36 이 2631.5 · 2630.5 사이에 끼면 그것만 다른 물건이 된다.
  const t = el.querySelector("text");
  if (t) {
    t.textContent = (g.keys[j] * g.rowSize).toFixed(g.digits);
    t.setAttribute("font-weight", g.keys[j] === g.pocKey ? "700" : "600");   // 위 라벨과 동일
  }
}

// Snapshot tab's own candlestick chart -- same renderCandleSvg() the Live tab uses, always ETH, no
// bot position context (entryPrice=0, journal=[]), with the liquidation map drawn as a density
// profile strip plus a single line for the nearest support/resistance level (2026-08-24: the full
// 12-line overlay was removed for clutter, see nearestLiquidationLevel/liquidationDensityHistory
// above; the level list below the chart is still the place to read every level's exact price) and
// the long/short averaged evidence-signal TP lines (see evidenceSignalTpLevels above; the
// liquidation magnet line that used to live here too was removed 2026-08-31 per user request).
// Called both right after its two 5-min data sources (candles, liquidation map) refresh, AND every
// ~5s from render() (see updateSnapshotCandleLive() above and the call site in render()) so the
// candle body and the current-price line stay in sync instead of only one of them moving.
// NOTE: on mobile, pan/zoom (visibleCandleWindow) reads the same module-level mobileChartView the
// Live chart's gestures write to -- shares the same window index, not independently interactive.
// Not wired up to setupMobileCandleGestures() (that's hardcoded to #candleSvg); acceptable since
// this chart is read-only reference, not something a user pinches/pans on its own.
// 차트 위에 커서가 있는 동안은 다시 그리지 않는다. 다시 그리기는 svg 를 통째로 비우므로
// (renderCandleSvg 의 innerHTML = "") **읽고 있던 툴팁·십자선이 사라진다**. 20초 주기일 땐
// 드물어서 안 보였지만 2.5초로 당긴 순간 «호버할 때마다 깜빡임»이 된다.
// 커서가 나가면 밀린 갱신을 즉시 한 번 그린다 -- 멈춘 채로 남겨두지 않는다.
let chartHoverActive = false;
let chartRenderDeferred = false;
// 객체 신원을 문자열 키로 바꾼다. 층 캐시가 «이 배열이 그대로인가»를 물을 때 쓴다 --
// 내용 해시를 뜨지 않아도 되는 이유는 이 화면의 payload 가 폴링마다 **통째로 교체**되기
// 때문이다(같은 내용이면 같은 객체, 새 응답이면 새 객체). WeakMap 이라 누수가 없다.
const objToken = (() => {
  const seen = new WeakMap();
  let n = 0;
  return (o) => {
    if (!o || typeof o !== "object") return "-";
    let t = seen.get(o);
    if (!t) seen.set(o, (t = "#" + (++n)));
    return t;
  };
})();

function renderSnapshotChart() {
  if (chartHoverActive) { chartRenderDeferred = true; return; }
  const svg = el("candleSvgSnapshot");
  if (!svg) return;
  const fullCandles = candleHistoryByAsset[activeSnapshotAsset] || [];
  if (!fullCandles.length) return;
  // Sliced to SNAPSHOT_CHART_MAX_CANDLES (6h) -- narrower than the shared candleHistoryByAsset
  // cache (still 8h, CHART_MAX_CANDLES) so the density-history overlay always has a real snapshot
  // behind every visible column (see that constant's comment).
  // 풋프린트 모드면 테이프가 덮는 구간(최대 12봉=1시간)만 그린다 -- 72봉을 1200px 에 넣으면
  // 봉당 16px 라 셀이 물리적으로 안 들어간다. 대신 청산밀도 히트맵은 끈다(셀과 같은 자리를
  // 다투고, 정확한 레벨은 차트 아래 목록에 그대로 있다). 2026-09-15 사용자 결정 "완전 교체".
  const footprint = footprintForChart();
  const candles = footprint
    ? fullCandles.filter((c) => c.time >= footprint.firstTime)
    : fullCandles.slice(-SNAPSHOT_CHART_MAX_CANDLES);
  const currentPrice = Number(latestLivePriceByAsset[activeSnapshotAsset] || candles[candles.length - 1]?.close || 0);
  const riskLevels = [...nearestLiquidationLevel()];
  // 2026-09-21 사용자 요청: **풋프린트에도 청산 밀도 배경을 깐다**(전에는 청산맵 전용이었다).
  // 비용 걱정은 없다 -- liquidationDensityHistory() 가 payload 신원으로 memoize 돼 있어
  // /api/liquidation-map 이 갱신될 때(60초)만 다시 만든다.
  const densityHistory = liquidationDensityHistory();
  // 2026-09-10: 이 차트는 줄곧 entryPrice=0 을 넘겨 「진입」 선을 안 그렸다. renderCandleSvg 에
  // 그리는 코드는 이미 있으므로(priceLabels 의 amber "진입"), 거래소 실계좌 진입가만 넘긴다.
  const entryPrice = Number(snapshotAccountPosition()?.entry_price || 0);
  // 2026-09-11 8번째 인자 = 봉별 청산 레인. **이 줄이 빠져 있어 레인이 통째로 안 그려졌다**
  //   (liqBars 기본값 [] -> peak 0 -> 블록 전체 skip, 오류도 안 남). 치환 대상 문자열을
  //   확인 없이 바꾸려다 조용히 실패했던 자리다.
  renderCandleSvg(svg, candles, [], entryPrice, currentPrice, riskLevels,
    densityHistory, latestLiquidation5mHist, footprint);
}

// wide24/GBM3 regime overlay -- drawn as a ribbon INSIDE renderCandleSvg() itself (2026-08-26,
// moved in from a standalone strip below the chart per user request: "레짐 그래프를 청산맵 안에
// 넣을 순 없어?"). Dominant-class color (not a 3-way blend) matches the categorical tone convention
// the evidence-signal strips use elsewhere; opacity scales with confidence so an uncertain reading
// fades rather than asserting a false-confident color.
// 토큰을 복사하지 않고 읽는다 -- 하드코딩 사본은 2026-09-12 에 두 번 따로 고쳐야 했다.
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const REGIME_DOMINANT_COLOR = { get bull() { return cssVar("--good"); },
                                get bear() { return cssVar("--bad"); },
                                get chop() { return cssVar("--muted"); } };
function regimeDominant(r) {
  return r.bull_prob >= r.bear_prob && r.bull_prob >= r.chop_prob ? "bull"
    : r.bear_prob >= r.chop_prob ? "bear" : "chop";
}

function fmtDateTick(ts) {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return "";
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${hh}:${mm}`;
}

// 청산 밀도 히트맵 컬러맵 (2026-09-10 교체). 이전엔 matplotlib **viridis**(보라->파랑->초록
// ->노랑)였는데 사용자 지적 "청산밀도 색깔이 너무 어지러운 색깔이야. 봉 차트 색과 잘 조화롭게".
// viridis 의 초록(34,168,132)·연두(122,209,81)·노랑(253,231,37) 구간이 캔들의 상승 초록
// (--good)·경고 앰버(--amber)와 정면으로 부딪혔다 -- 배경 띠가 캔들보다 튀었다.
//
// 교체 원칙 셋:
//   1. **단색(쿨) 램프**: 밝기만 단조 증가시키고 색상은 안 바꾼다 -> 배경으로 읽힌다.
//   2. **초록/빨강/노랑 금지**: 밀도는 방향이 없다(2026-08-25 에 dual-hue 를 뺀 이유).
//      캔들 색과 겹치면 밀도가 방향 정보로 오독된다.
//   3. t=0 은 알파 0 이라 배경에 그대로 녹고, t=1 은 채도를 낮춘 스틸블루라
//      --accent(#22d3ee, 가격선)보다 덜 튄다.
// 2026-09-12: 패널 명도 계단(b5a4790)으로 t=0 칸이 어두운 구멍이 됐다. 고침은 두 번 틀렸다 --
// (1) t=0 색을 패널에 맞추기(a468170): .panel 이 세로 그라디언트라 실제 패널색이 위아래로 달라
//     어떤 단일 색도 전 구간에 안 맞는다(렌더 실측 격차 23.1 잔존).
// (2) fill-opacity 를 t 에 비례시키기: densityClip 이 **양수 밀도의 90분위**라 대부분의 칸이
//     t<0.25 에 몰린다 -- t=0.1 이 0.85 에서 0.34 로 깎여 **히트맵 전체가 사라졌다**(사용자 신고).
// 정답은 단순하다: 밀도 0 인 칸은 **그리지 않는다**(drawDensitySeg 의 `if (!(t > 0)) return`).
// 안 그리면 배경이 그대로 비쳐 어떤 패널색에도 정확히 녹고, 밀도가 있는 칸은 원래 0.85 를 지킨다.
// ⚠️2026-09-13 재수정: 0.0 과 0.25 를 같은 색으로 둔 게 «히트맵이 검정» 의 원인이었다.
// 라이브 /api/liquidation-map 을 실측하니 양수 밀도 칸이 **100%**(=위 그리기 생략은 실제로
// 아무것도 안 거른다)이고 그 중 **53.6% 가 t<=0.25** 라, 램프 아래 25% 를 납작하게 만든
// 순간 화면 절반이 한 가지 색이 됐다. 그 색은 배경 #171b23 위 0.85 합성 기준 배경거리가
// 27.8 뿐이라 파랑이 아니라 검정으로 읽힌다. 그래서 바닥을 올리고 t=0 부터 보간시킨다.
// 합성 후 배경거리 49.3 · 휘도 단조 3.3 -> 6.4 -> 12.9 -> 23.1
// 캔들색 이격(합성 후 RGB 유클리드, 전 구간 60 이상): 초록 #5abc80 최소 75 · 빨강 #d4786c 최소 148
// 범례(캔들 SVG 안, 프로파일 아래 줄)도 이 배열에서 만든다 -- 색 사본을 다른 곳에 두지 않는다.
// 🔴2026-09-21 **테마별로 갈랐다 -- 라이트에서 척도가 뒤집혀 있었다.**
// 색표가 하나뿐이라 «어두운 남색 -> 밝은 파랑» 을 두 배경에 같이 썼는데, 그러면
// 라이트(실효 배경 rgb(236,240,246))에서 **밀도가 낮을수록 진하게** 보인다.
// 실측 배경대비(합성 후, WCAG):
//     다크   t=0 1.22 -> t=1 4.15   단조 증가 ✅ (3.40x)
//     라이트 t=0 6.79 -> t=1 2.19   **역전** 🔴 (0.32x)  ← 사용자 스크린샷의 진한 블록이 이것
// 규칙은 하나다: **밀도가 높을수록 배경 대비가 강하다.** 그걸 배경마다 다른 색으로 구현한다.
//     라이트 신규 t=0 1.13 -> t=1 6.56 단조 증가 ✅ (5.79x)
// 범례도 같은 배열(densityStops)에서 만들므로 자동으로 따라간다 -- 색 사본을
// 다른 곳에 두지 않는다는 기존 규약 그대로다.
const DENSITY_STOPS_DARK = [
  [0.0, [34, 56, 84]],
  [0.35, [44, 82, 120]],
  [0.7, [62, 118, 166]],
  [1.0, [96, 156, 208]],
];
const DENSITY_STOPS_LIGHT = [
  [0.0, [214, 225, 240]],
  [0.35, [158, 186, 221]],
  [0.7, [86, 132, 190]],
  [1.0, [21, 58, 115]],
];
const densityStops = () =>
  (document.documentElement.getAttribute("data-theme") === "light"
    ? DENSITY_STOPS_LIGHT : DENSITY_STOPS_DARK);
function densityColor(t) {
  t = clamp01(t);
  const DENSITY_STOPS = densityStops();
  for (let i = 0; i < DENSITY_STOPS.length - 1; i++) {
    const [t0, c0] = DENSITY_STOPS[i], [t1, c1] = DENSITY_STOPS[i + 1];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0 || 1);
      const rgb = c0.map((v, k) => Math.round(v + (c1[k] - v) * f));
      return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    }
  }
  const last = DENSITY_STOPS[DENSITY_STOPS.length - 1][1];
  return `rgb(${last[0]},${last[1]},${last[2]})`;
}

function renderCandleSvg(svg, candles, journal, entryPrice, currentPrice, riskLevels = [], densityHistory = [], liqBars = [], footprint = null) {
  const parentW = svg.parentElement ? svg.parentElement.clientWidth : 0;
  // 2026-09-18 부모가 아니라 **SVG 자신의** 높이를 본다. 부모는 마진 12px 를 포함하므로
  // (styles.css .chart-container 412px) 그 값으로 viewBox 를 잡으면 모바일에서 meet 축소 +
  // 레터박스가 생긴다. viewBox 높이는 SVG 가 실제로 차지하는 상자와 같아야 한다.
  const parentH = svg.getBoundingClientRect().height
    || (svg.parentElement ? svg.parentElement.clientHeight : 0);
  const mobileChart = isMobileChartMode();
  // 2026-09-19 2열 배치(사용자 지시)로 상자가 카드 폭의 68%/32% 가 됐다. 1200/400 을 고정으로
  // 두면 viewBox 가 상자보다 커서 meet 축소가 걸리고 글자가 그만큼 작아진다 -- 상자에서 받는다.
  // 하한은 «아직 레이아웃 전»(parentW/H = 0)일 때의 폴백이다.
  const w = parentW > 0 ? Math.max(parentW, 320) : 1200;
  const h = parentH > 0 ? Math.max(parentH, 260) : 400;
  // 하단/상단 여백 안의 것들(x축 눈금·라벨·레짐 리본·증거신호 레인)은 전부 `mt` / `h - mb`
  // 상대 오프셋이다 -- 여백을 늘리면 통째로 따라 움직인다.
  // 2026-09-10 mb 40 -> 56 (레짐 리본 20px 확보).
  // 2026-09-11 사용자 요청 "증거신호 레인을 청산맵 밖으로": 레인이 플롯 **위에 겹쳐** 그려져
  //   캔들을 가리던 것을 여백으로 뺐다. mt 20 -> 22(천장 레인 자리), mb 56 -> 74(바닥 레인
  //   자리). 레인이 플롯에서 30px 를 돌려주므로 실제 캔들 영역 손실은 324 -> 304 로 20px 뿐이다.
  // 2026-09-11 mb 74 -> 92: 봉별 청산 레인 18px. 레인은 **플롯 밖**이다 -- 09-11 사용자 요청
  //   "증거신호 레인을 청산맵 밖으로"와 같은 원칙으로 캔들을 가리지 않는다.
  // 2026-09-11 mb 92 -> 112: 청산 레인을 레짐 리본 높이(20px)만큼 키웠다(사용자 요청).
  //   레인 18 -> 38px. 여백도 같이 20px 늘려야 리본과 안 겹친다.
  //   대가: 캔들 영역 ch 가 286 -> 266 으로 20px 줄어든다.
  // 2026-09-11 mb 112 -> 134: 변동성 전망 리본 20px(사용자 요청 "레짐과 같은 스타일로").
  //   칩을 없애고 리본으로 옮긴 것이라 화면의 정보량은 그대로다.
  // 2026-09-11(2차) mb 134 -> 140: 레짐과 변동성 리본 사이 간격 2 -> 6px(사용자 요청).
  //   대가: 캔들 영역 ch 가 266 -> 238(데스크톱), 126 -> 98(모바일 최소높이)로 줄어든다.
  // 하단 여백 순서: 눈금 +0~+5 · x축 라벨 +21 · 바닥 레인 +28~+43 · 레짐 +50~+70 ·
  //   변동성 +76~+96 · 청산 레인 +100~+138.
  // 2026-09-16 청산 레인을 **차트 안 하단 패널**로 옮겼다(사용자 요청). 그전까지는 하단 여백에
  //   떠 있어서(+100~+138) 차트 밖 부속처럼 보였다. 패널은 가격 플롯 바로 아래·x축 바로 위에
  //   앉고, 여백은 그 레인이 쓰던 40px 를 돌려받는다(mb 140 -> 100).
  //   결과: 가격 플롯 238 -> 226(데스크톱), 98 -> 98(모바일 최소높이, 패널을 34 로 줄여 상쇄).
  // ⚠️`ch` 는 이제 **가격 플롯 높이**다. yAt() 이 이 값을 쓰므로, 「플롯 바닥」을 뜻하던
  //   `h - mb` 는 더 이상 가격 영역의 바닥이 아니다 -- 그 자리들은 전부 plotBottom 으로 바꿨다.
  //   (여백 안의 것들 -- x축 눈금·라벨·레짐/변동성 리본·증거신호 레인 -- 은 그대로 h - mb 기준)
  // 2026-09-16 줄 배치: 천장 레인이 위 여백에서 빠져 mt 를 22 -> 12 로 줄이고(플롯이 그만큼
  // 커진다), 아래는 바닥 레인 자리를 회수한 뒤 구간 2줄(추세 전환·변동폭 게이트)을 붙였다.
  // 하단 여백 순서: 눈금 +0~+5 · x축 라벨 +21 · 레짐 +28~+43 · 변동성 +49~+64 ·
  //                 추세 전환 +49~+64
  // 2026-09-20 «게이트» 줄 제거(사용자 지시) -> mb 91 -> 70. 그 21px 는 바로 아래 새로
  //   들어온 수급 리본(15 + 간격 6)이 그대로 받는다 -- 가격 플롯 높이는 변하지 않는다.
  // (2026-09-16 변동성 리본이 카드로 빠지면서 한 줄 21px 를 가격 플롯에 돌려줬다)
  // 2026-09-16 모바일 ml 34 -> 44: 좌측 가격선 라벨이 `ml - 5` 에서 **왼쪽으로** 뻗는데
  //   «저항1↑»(5글자 ~31px)가 x=-2 까지 나가 잘렸다. 차트 폭은 288 -> 278 로 10px 준다.
  // ── 수급 두 패널을 이 SVG 안에 담는다 (2026-09-19 사용자 지시 "svg 안으로") ──────
  // 독립 컨테이너 둘(.supply-profile-container/.supply-1s-container)을 없애고 캔들 SVG
  // 안, **OI 레인 바로 위**에 붙인다(2026-09-19 2차 지시). 가격 플롯 바로 아래 자리다 --
  // 레인 스택의 맨 위가 되고, 아래로 OI · 청산 · x축/리본이 그대로 따라온다.
  // 상자 높이(666)는 그대로이고 이 266 을 여백이 아니라 ch 에서 뺀다 -- 가격 플롯은 205 로
  // 같다(앞 판은 mb 에 더했다. 자리만 위로 옮겼을 뿐 총량은 같은 수다).
  // 🔴ETH 전용 -- OI 레인과 같은 이유다(다른 코인 캔들에 ETH 수급을 얹지 않는다).
  // ⚠️대가: 이 SVG 는 가격 틱마다 통째로 다시 그려진다(초당 ~1.7회). 밖에 있을 때 두 그림은
  //   5초·1초 주기였다 -- 이제 그 주기로 같이 다시 그려진다. 사용자 결정으로 감수한다.
  const subOn = svg.id === "candleSvgSnapshot" && activeSnapshotAsset === "eth";
  // 🔴이 세 값의 합(SUB_TOTAL)은 styles.css 의 #candleSvgSnapshot 높이와 **같이** 움직여야
  //   한다(666 = 400 + 266 -> 706 = 400 + 306 -> 756 = 400 + 356). 상자가 작으면
  //   그만큼 가격 플롯이 눌린다.
  // 2026-09-19 프로파일 150 -> 190 (아티팩트 댓글 "조금만 더 키워줘"). 행 수는 높이가 정하므로
  //   (renderSupplyProfileSvg 의 maxRows) 16행 -> 22행이 된다.
  // 2026-09-20 1초 수급 100 -> 150 (사용자 요청 "좀 더 키워줘"). 상자도 706 -> 756 으로
//   같이 키운다 -- 안 그러면 가격 플롯이 그만큼 눌린다(아래 경고 블록).
  const SUB_GAP = 8, SUB_PROFILE_H = subOn ? 190 : 0, SUB_1S_H = subOn ? 150 : 0;
  // 2026-09-21 청산 밀도 범례를 헤더 행에서 **여기로** 옮겼다(아티팩트 댓글).
  //   «풋프린트 차트 바로 위와 프로파일 바닥글 사이». 밀도는 이제 풋프린트의 배경이라
  //   범례가 헤더에 있으면 설명하는 그림에서 멀다. 글자 크기도 바닥글과 같은 9 로 맞췄다.
  const SUB_LEGEND_H = subOn ? 14 : 0;
  // 2026-09-19 히트맵은 프로파일 **아래 제 줄**이다(사용자 지시). 좌우 반씩 나누던 판을
  // 되돌렸다 -- 프로파일 막대 해상도가 절반이 됐고, 두 패널의 자연 가격범위가 15배 달라
  // (호가 ±2.4% vs 체결 ±0.16%) 나란히 둘 이유였던 «같은 축»도 성립하지 않았다.
  // ⭐데스크톱·모바일이 같은 모양이 되므로 subStack 분기가 통째로 사라진다.
  //   SUB_TOTAL 356 = 8 + 150(1초 수급) + 8 + 190(프로파일)   ← 2026-09-20 위아래 뒤집힘
  const SUB_TOTAL = subOn ? SUB_GAP + SUB_PROFILE_H + SUB_LEGEND_H + SUB_GAP + SUB_1S_H : 0;
  // 🔴상자 높이(styles.css 의 #candleSvgSnapshot/.candle-container)와 위 SUB_* 상수는 두
  //   파일에 갈라져 있다. 한쪽만 고치면 가격 플롯이 **조용히** 눌린다(ch 에서 SUB_TOTAL 을
  //   빼기 때문). 인라인 height 로 JS 가 상자를 정하는 방법은 쓰지 않는다 -- 2열에서는 상자가
  //   열 높이를 따라 늘어나는 게 의도된 동작인데 인라인 height 가 그 auto 를 이기고, h 자체를
  //   getBoundingClientRect 로 읽으므로 자기참조가 된다. 대신 어긋나면 **한 번 알린다**.
  if (SUB_TOTAL > 0 && h < 400 + SUB_TOTAL - 2 && !renderCandleSvg._subBoxWarned) {
    renderCandleSvg._subBoxWarned = true;
    console.warn(`캔들 상자가 ${Math.round(h)}px 인데 수급 패널이 ${SUB_TOTAL}px 를 쓴다 -- `
      + `styles.css 의 #candleSvgSnapshot ${400 + SUB_TOTAL + 55}px / .candle-container `
      + `${412 + SUB_TOTAL + 55}px 로 맞추세요(그만큼 가격 플롯이 눌립니다).`
      + " (55 = 거래대금 15 + 델타·CVD 28 + 간격 12)");
  }

  // 2026-09-20 아티팩트 댓글: 수급(1초)을 1h/2h/4h 버튼 **바로 아래**로 올리고 그 아래에
  // 프로파일을 둔다. 위에서부터 [1초 수급][프로파일][가격 플롯][OI][청산] 이다.
  // ⭐두 패널을 옮기는 대신 **mt(가격 플롯의 윗변)를 그만큼 내린다**. mt 는 이 함수에서
  //   18곳이 쓰는 «플롯 top» 이라, 그 뜻을 유지하면 yAt·클램프·커서 매핑·세로선을 한 줄도
  //   안 건드린다. 상자 총높이는 SUB_TOTAL 과 함께 움직인다(756 = 400 + 356).
  // 2026-09-20 아티팩트 댓글: 「살짝 왼쪽으로 치우쳐 보인다」. 실측(화면 좌표 기준,
  //   데스크톱 1198폭): 왼쪽 여백 13 · 오른쪽 44 -- 오른쪽(112)에 가격 라벨 자리를 넓게
  //   뒀는데 라벨이 그걸 다 안 쓴다. 차 31 의 절반인 16px 을 오른쪽으로 옮긴다.
  // ⭐플롯 «폭»은 그대로다(ml+n, mr−n 이라 cw 불변) -- 봉 너비·x 스케일·열 수가 한 픽셀도
  //   안 움직인다. 순수 이동이다.
  // 🔴모바일은 건드리지 않는다: 실측 왼쪽 12 · 오른쪽 7 로 이미 가운데다(차 −5).
  //   같은 값을 양쪽에 주면 모바일이 반대로 25 밀린다(첫 시도에서 실제로 그랬다).
  // 🔴getBBox() 로 재면 안 된다 -- 조상 transform 을 무시해서 translate 된 마커가 x=-5 로
  //   잡히고, 그 허수 때문에 「왼쪽은 ml 을 안 따라간다」는 틀린 결론이 나왔다.
  //   getBoundingClientRect() 로 잰다.
  // 2026-09-20 2차(사용자 「오른쪽으로 조금 더」): 16 -> 26. 수치상 가운데는 16 이었지만
  //   보기에는 왼쪽 라벨이 짧아(진입↑·저항1·현재) 왼쪽이 더 비어 보인다. 취향값이다.
  // 🔴더 올리면 오른쪽 라벨이 잘린다 -- mr 이 112−n 이라 n=26 이면 86 이 남고 실측 여유가
  //   18px 이다. 그 아래로는 「최대 $210.5k」 같은 긴 꼬리표가 상자를 넘을 수 있다.
  const CENTER_NUDGE = mobileChart ? 0 : 26;
  const ml = (mobileChart ? 44 : 45) + CENTER_NUDGE,
        mr = (mobileChart ? 68 : 112) - CENTER_NUDGE,
        mtTop = 12, mt = mtTop + SUB_TOTAL, mb = 70;
  // 2026-09-21 사용자 요청: 「차트가 너무 많다 -- 레인을 한 덩어리로」 + 「가격 플롯을 키워라」.
  //   레인 5종(거래대금·델타/CVD·OI·수급·청산)이 각자 6px 간격으로 떨어져 있어 **다섯 장의
  //   그림**으로 읽혔다. 간격 6->3, 높이를 서로 가깝게 맞춰 한 블록으로 읽히게 한다:
  //   데스크톱 168 -> 111px. 그 57px 과 상자 증가분을 전부 가격 플롯이 가져간다(205 -> 400).
  // ⭐다섯 레인은 전부 «높이=해상도» 노브다 -- OI·청산은 대칭 이분(half = H/2 - 1), 수급은
  //   SMID 기준, 거래대금은 단극이다. 안에 고정 크기 요소가 없어 높이만 바꾸면 되고 좌표
  //   계산은 한 줄도 안 건드린다. LANE_H(15)는 **다른 것**이다(레짐 리본·구간 줄, 하단 여백).
  // 🔴DCVD 는 24 아래로 내리지 말 것 -- 15 에서 작은 델타 막대가 0.7px 였다(기존 주석).
  const LIQ_PANEL_H = mobileChart ? 20 : 24, LIQ_PANEL_GAP = 3;
  // OI 신규계약 레인 -- 청산 레인 **바로 위**(사용자 지시). 별도 패널이 아니라 이 SVG 안의
  // 서브플롯이라야 캔들과 x축(봉)이 구성상 같아진다(레짐 리본이 같은 이유로 여기 있다).
  // 🔴ETH 전용이다. 다른 코인을 보는 동안 ETH 값을 얹으면 2026-08-31 레짐 리본 사고와 같은
  //   모양이 된다 -- 자산 전환에서 latestOi5m 을 비우는 것과 이 조건, 둘 다 필요하다.
  // 데이터가 없으면 자리를 아예 안 잡는다(웜업·다른 코인에서 가격 플롯만 40px 손해다).
  const oiBars = (svg.id === "candleSvgSnapshot" && activeSnapshotAsset === "eth"
                  && latestOi5m && Array.isArray(latestOi5m.bars)) ? latestOi5m.bars : [];
  // 모바일 26 / 데스크톱 34. h 는 모바일에서도 실제로 400 이다(styles.css 가 #candleSvgSnapshot
  // 높이를 400px 로 고정 -- `Math.max(parentH, 260)` 의 260 은 SVG 가 안 그려질 때의 바닥값이다).
  // 400 기준 가격 플롯은 모바일 233, 데스크톱 205 로 남는다.
  const OI_PANEL_H = oiBars.length ? (mobileChart ? 18 : 20) : 0;
  const OI_PANEL_GAP = oiBars.length ? 3 : 0;
  // ── 고래·리테일 수급 리본 -- OI 레인 바로 아래 (2026-09-20 사용자 지시) ──────────
  // 봉마다 «그 5분에 순 몇 ETH 가 들어왔나»를 두 계층으로 가른다. 풋프린트가 이미 봉별
  // 가격대 셀을 주고 supplyFlowOfBar() 가 그걸 셋으로 가르므로 여기서는 **자리만** 잡는다.
  // 🔴둘은 **한 자**를 나눠 쓴다(각자 정규화하면 「고래 매도 · 리테일 매수」가 같은 크기로
  //   보여 거짓이 된다). 이 리본의 값이 바로 그 엇갈림이다 -- 실측 12봉 중 4봉에서 부호가
  //   반대였다. 자 자체는 log1p 다(아래 hgt 주석).
  // 데이터가 없으면 자리를 아예 안 잡는다(OI 레인과 같은 규약).
  const fpBars = (svg.id === "candleSvgSnapshot" && activeSnapshotAsset === "eth"
                   && latestFootprint && Array.isArray(latestFootprint.bars))
                  ? latestFootprint.bars : [];
  const SUP_PANEL_H = fpBars.length ? 14 : 0;
  const SUP_PANEL_GAP = fpBars.length ? 3 : 0;
  // ── 거래대금 · 델타·CVD -- 풋프린트 바로 아래 (2026-09-21 사용자 지시) ────────────
  // 같은 풋프린트 봉에서 나온다: 거래대금 = Σ가격x(매수+매도) · 델타 = Σ(매수-매도) ·
  // CVD = 그 창 안에서의 델타 누적.
  // 🔴CVD 의 기준점은 **창 시작**이고, 창은 1h/2h/4h 선택기를 그대로 따른다(사용자 결정).
  //   `CHART_WINDOW_BARS` 12/24/48 이 곧 그 셋이고 서버 상한(FOOTPRINT_MAX_WINDOW_BARS)도
  //   48 이라 셋 다 덮인다. 창을 바꾸면 기준점도 같이 옮겨간다 -- 절대 누적이 아니다.
  // 🔴«상황 읽기» 카드의 CVD 는 **30분 고정창**이다(dashboard/situation.py 의 WINDOW=6).
  //   이름이 같아도 값이 다르다. 툴팁에 창을 적는다.
  const TURN_H = fpBars.length ? 14 : 0;
  const DCVD_H = fpBars.length ? 24 : 0;      // 15 면 작은 델타 막대가 0.7px 다(실측)
  const FLOW_GAP = fpBars.length ? 3 : 0;
  const cw = w - ml - mr;
  const ch = h - mt - mb - LIQ_PANEL_H - LIQ_PANEL_GAP - OI_PANEL_H - OI_PANEL_GAP
             - SUP_PANEL_H - SUP_PANEL_GAP - TURN_H - DCVD_H - 2 * FLOW_GAP;
  const plotBottom = mt + ch;                      // 가격 플롯의 바닥
  // 수급 두 패널은 **가격 플롯 위**다(위 mt 주석). 1초 수급이 먼저, 프로파일이 그 아래 --
  // 「체결 계열」 둘은 여전히 이웃한다. OI·청산 레인은 플롯 바로 아래 그대로다.
  const sub1sY = mtTop;
  const subProfileY = sub1sY + SUB_1S_H + SUB_GAP;
  const subLegendY = subProfileY + SUB_PROFILE_H;
  const turnPanelY = plotBottom + FLOW_GAP;
  const dcvdPanelY = turnPanelY + TURN_H + FLOW_GAP;
  const oiPanelY = dcvdPanelY + DCVD_H + OI_PANEL_GAP;
  const supPanelY = oiPanelY + OI_PANEL_H + SUP_PANEL_GAP;
  const liqPanelY = supPanelY + SUP_PANEL_H + LIQ_PANEL_GAP;
  const NS = "http://www.w3.org/2000/svg";
  // 풋프린트는 서버가 주는 12봉이 곧 창이다 -- 모바일 핀치줌(visibleCandleWindow)으로 더
  // 잘라내면 셀만 커지고 볼 구간이 사라진다.
  const viewport = footprint ? { candles, includeCurrent: true } : visibleCandleWindow(candles);
  candles = viewport.candles;
  const includeCurrentPrice = viewport.includeCurrent;

  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.innerHTML = "";
  
  if (!candles.length) {
    const txt = document.createElementNS(NS, "text");
    txt.setAttribute("x", w/2); txt.setAttribute("y", h/2);
    txt.setAttribute("text-anchor", "middle"); txt.setAttribute("fill", "var(--muted)");
    txt.textContent = "시장 데이터 대기 중...";
    svg.appendChild(txt);
    return;
  }

  // Candle visibility takes priority: entry/SL/TP lines never widen the price scale, and neither
  // does densityHistory (2026-08-25: briefly widened the scale to fit the full profile, reverted
  // again immediately -- squeezed candles were rejected a second time, so this stays candle-only
  // for good; a sparse profile that only shows bins near the visible candle range is the accepted
  // tradeoff, not a bug to re-litigate by re-widening). A bin whose price falls outside the
  // resulting range simply doesn't draw (see the clamped top/bottom below) rather than stretching
  // the scale to fit it -- same "off-chart, so omit" treatment priceLabels below gives out-of-range
  // levels, just without an edge arrow since a profile bar has no sensible one.
  const allPrices = candles.flatMap(c => [c.high, c.low]);
  if (includeCurrentPrice && currentPrice > 0) allPrices.push(currentPrice);

  const minP = Math.min(...allPrices), maxP = Math.max(...allPrices);
  // 2026-09-21 사용자 요청 «청산맵 위아래 여유». 여백은 모드마다 값이 다르다:
  //  · 청산맵  -- 넓힐수록 위아래 청산 레벨과 그 라벨이 화면 안으로 들어온다. 캔들이 조금
  //    납작해지는 대가는 청산맵에서 작다(셀 숫자를 읽을 일이 없다).
  //  · 풋프린트 -- 넓히면 행 높이(rowPx)가 줄어 셀 숫자가 먼저 깨진다. 여기는 그대로 둔다.
  //    (rowSize 는 ySpan/ch 로 정해진다 -- 여백을 늘리면 ySpan 이 커져 행이 얇아진다.)
  const padPct = footprint ? CHART_Y_PAD_FOOTPRINT : CHART_Y_PAD_PLAIN;
  const pad = (maxP - minP) * padPct || 1;
  const yMin = minP - pad, yMax = maxP + pad;
  const ySpan = Math.max(yMax - yMin, 1e-5); // Prevent division by zero

  // 2026-09-21 청산 밀도는 **전체폭**이다 -- 체결 봉 뒤로 지나간다(사용자 요청).
  //   한때 풋프린트만 왼쪽 게이트로 뺐는데, 그건 «알파를 낮춰 겹치기»가 척도를 눌러 버린
  //   것을 피하려던 우회였다. 알파는 두 모드 같은 0.85 로 두고 겹치기를 허용한다.
  // 🔴셀 가독성 실측(라이트, 밴드 뒤에 깔 때 셀 숫자 배경대비):
  //     밴드없음 5.30 · t<=0.2 4.59 · t=0.5 3.93 · t=1.0 2.56  (전부 진한 셀 0.66 기준)
  //   densityClip 이 90분위라 t>=1.0 은 빈의 10% 뿐이다 -- 대부분은 옅어서 거의 안 건드리고
  //   강한 군집에서만 흐려진다(그 자리는 어차피 눈에 띄어야 한다).
  //   더 거슬리면 «t 하한을 둬 약한 밴드는 안 그리기»가 다음 손잡이다.
  const xAt = (i) => ml + (i * cw) / candles.length;
  const yAt = (v) => mt + ((yMax - v) * ch) / ySpan;
  const bw = (cw / candles.length) * 0.8;

  // ── 계층 캐시 (2026-09-20) ────────────────────────────────────────────────
  // 봉 셀은 이미 봉별로 캐시한다. 남은 것은 **가격 플롯 바깥의 층들**이다 -- 격자·x눈금·
  // 레짐 리본·구간 줄·OI 레인·청산 레인·청산밀도 히트맵. 이들도 매 렌더 새로 만들고 있었다.
  // 실측(틱 스트림, 셀 캐시가 다 맞은 정상 상태에서 만들어지는 노드):
  //     풋프린트 4h  372개/렌더 (rect 114 · title 160 · text 70)
  //     청산맵  1h  574개/렌더 (rect 301 -- 밀도 히트맵이 압도적)
  // 그런데 이 층들의 **입력은 15~300초마다** 바뀐다(레짐 300s · 청산맵 60s · OI 15s).
  //
  // 🔴키는 «기하 + 봉 시각 + 그 층의 데이터 신원» 셋이다. 기하를 빼면 축이 움직여도 낡은
  //   픽셀이 남고, 봉 시각을 빼면 봉이 한 칸 밀려도 그대로 남는다.
  // 🔴현재가·진행봉 OHLC 는 **일부러 안 넣는다**. 넣으면 틱마다 전부 깨져 캐시가 무의미해진다.
  //   그래서 OHLC 에 의존하는 것(가격 라벨·이벤트 삼각형)은 **캐시하지 않고** 매번 그린다.
  // 🔴테마를 서명에 넣는다 -- 밀도 색표·풋프린트 셰이드·잉크 불투명도가 전부 테마 파생이라,
  //   빠뜨리면 테마를 바꿔도 캐시된 층이 옛 색 그대로 남는다(모드 키와 같은 함정).
  const themeSig = document.documentElement.getAttribute("data-theme") || "dark";
  const baseGeomSig = [themeSig, w, h, mt, ch, ml, mr, cw, bw, yMin, yMax, plotBottom,
                       mobileChart, oiPanelY, liqPanelY, OI_PANEL_H, LIQ_PANEL_H,
                       supPanelY, SUP_PANEL_H, turnPanelY, dcvdPanelY, TURN_H, DCVD_H].join("|");
  // 봉 시각만. 진행 중인 봉의 OHLC 는 여기 없다(위 주석).
  const timesSig = candles.length + ":" + (candles[0] ? candles[0].time : 0)
                   + ":" + (candles[candles.length - 1] ? candles[candles.length - 1].time : 0);
  const layerCache = renderCandleSvg._layers || (renderCandleSvg._layers = new Map());
  /** 층 하나를 <g> 로 묶어 캐시한다. sig 가 같으면 만들어 둔 노드를 그대로 다시 붙인다.
   *  draw(g) 는 그 <g> 안에만 그려야 한다 -- svg 에 직접 붙이면 캐시를 우회한다. */
  const cachedLayer = (name, sig, draw) => {
    const full = baseGeomSig + "|" + timesSig + "|" + sig;
    const prev = layerCache.get(name);
    if (prev && prev.sig === full) { svg.appendChild(prev.g); return prev.g; }
    const g = document.createElementNS(NS, "g");
    draw(g);
    svg.appendChild(g);
    layerCache.set(name, { sig: full, g });
    return g;
  };

  // Regime ribbon (2026-08-26, moved in from the old standalone regimeWide24Strip row below the
  // chart per user request: "레짐 그래프를 청산맵 안에 넣을 순 없어?") -- drawn in this same svg/loop
  // so alignment with the candle columns above it is guaranteed by construction (same xAt/bw, no
  // second element to keep in sync), and the hover crosshair can sweep through it directly. Only on
  // the Snapshot tab's chart (svg id "candleSvgSnapshot"); latestRegimeWide24 is an ETH-only trained
  // model (see docs/eth_dashboard_multicoin_expansion_design_20260831.md -- no BTC regime classifier
  // exists yet), so it must only ever be drawn when the Snapshot tab's own coin switcher is on ETH --
  // 2026-08-31 fix: this used to key off svg.id alone, so picking BTC in the Snapshot tab silently
  // overlaid ETH's bull/bear/chop ribbon on BTC candles (found while building that coin switcher).
  const isSnapshotChart = svg.id === "candleSvgSnapshot";
  // 2026-09-02: BTC now has its own trained regime classifier, so the ribbon is no longer ETH-only.
  // Each asset reads its OWN endpoint's state -- never share one variable across assets, which is
  // precisely the 2026-08-31 bug this structure replaces (ETH's ribbon drawn over BTC candles).
  // Assets with no classifier still fall through to the "unsupported" grey band below.
  // 2026-09-03: XRP도 자체 분류기(S96_K9)를 갖게 돼 추가. 분류기 없는 자산은 아래 회색 밴드로 폴백.
  const REGIME_SOURCE_BY_ASSET = { eth: () => latestRegimeWide24, btc: () => latestRegimeBtc,
    xrp: () => latestRegimeXrp };
  const regimeSource = isSnapshotChart ? (REGIME_SOURCE_BY_ASSET[activeSnapshotAsset] || null) : null;
  const latestRegimeForChart = regimeSource ? regimeSource() : null;
  const regimeByTsForChart = latestRegimeForChart && latestRegimeForChart.warmed_up
    ? new Map((latestRegimeForChart.history || []).map((r) => [Math.floor(r.ts_ms / 1000), r]))
    : null;
  // 2026-08-27 user report: ribbon "turns black" and stops updating for stretches -- tracing the
  // draw loop below, it never paints an invalid color (regimeDominant() only ever returns one of
  // the 3 REGIME_DOMINANT_COLOR keys); what actually happens is this block draws literally nothing
  // whenever latestRegimeWide24.warmed_up is false (backend regime compute degrades to that instead
  // of raising, per load_regime_wide24()'s own docstring), so the ribbon's row just shows the dark
  // chart background underneath -- indistinguishable from "black" at a glance, and easy to mistake
  // for a frozen/broken ribbon rather than "no fresh reading available for this window". Flagging
  // that state explicitly below instead of silently drawing nothing.
  const regimeRibbonWaiting = Boolean(regimeSource) && !regimeByTsForChart;
  // 2026-08-31: distinct from regimeRibbonWaiting above -- BTC (or any future non-ETH Snapshot coin)
  // has no trained regime classifier at all, a permanent gap, not a transient "still warming up"
  // one. Kept as its own flag (rather than folding into regimeRibbonWaiting) so the flat band's own
  // tooltip can say so honestly instead of implying an auto-retry that will never resolve anything --
  // see eth-dashboard-btc-regime-classifier-not-trained-todo-20260831 memory for the follow-up
  // (swap this placeholder out once a real BTC regime classifier is trained).
  const regimeRibbonUnsupported = isSnapshotChart && !regimeSource;
  // 리본 높이 20. 2026-09-11 바닥 레인이 h-mb+28 로 들어오면서 리본은 +28 -> **+50** 으로 내려갔다.
  // 하단 여백 순서: 눈금 +0~+5 · x축 라벨 baseline +21 · 바닥 레인 +28~+43 · 레짐 리본 +50~+70.
  // h=400/mb=74 기준 리본이 396 에서 끝나 SVG 바닥까지 4px 여유.
  // ── 네 줄(천장·바닥·레짐·변동성)은 **같은 모양**을 쓴다 ────────────────────────────
  // 2026-09-16 사용자 "레짐과 변동성 모두 바닥과 천장처럼". 그전까지 레인(15px·트랙 있음)과
  // 리본(20px·트랙 없음)이 따로 자랐다. 치수를 여기 한 곳에서 정하고 네 줄이 받아쓴다 --
  // 각자 들고 있으면 한쪽만 고쳐져 또 어긋난다(이 파일에서 반복된 실패다).
  const LANE_H = 15;
  // 라운딩은 얇은 막대를 지운다: 모바일 34봉이면 bw≈6.8px 인데 rx=1.5 면 평평한 폭이 3.8px 다.
  const laneRx = bw >= 9 ? "1.5" : "0";
  const laneW = Math.max(bw, mobileChart ? 3.5 : 2.5);
  const laneX0 = xAt(0), laneX1 = xAt(Math.max(candles.length - 1, 0)) + bw;
  // 배경 트랙 -- 값이 없는 구간도 «줄이 거기 있다»를 보이게 한다(레인이 원래 하던 일).
  const drawLaneTrack = (y, into) => {
    const track = document.createElementNS(NS, "rect");
    track.setAttribute("x", laneX0); track.setAttribute("y", y);
    track.setAttribute("width", Math.max(laneX1 - laneX0, 1));
    track.setAttribute("height", LANE_H);
    track.setAttribute("rx", "1.5");
    track.setAttribute("fill", "var(--muted)");
    track.setAttribute("fill-opacity", "0.18");
    (into || svg).appendChild(track);   // 층 캐시가 목적지를 넘긴다
    return track;
  };
  const REGIME_RIBBON_Y = h - mb + 28, REGIME_RIBBON_H = LANE_H;
  // 변동성 전망 리본 -- 레짐 바로 아래, 같은 두께. ETH 전용 모델이라 다른 코인에선 안 그린다.
  // 방향 없는 두 신호의 구간 줄. 이름을 왼쪽 여백에 적는 것까지 레짐 리본과 같은 규약이다.
  const TREND_ROW_Y = h - mb + 49;
  // 변동성 값은 이제 카드가 보여준다. 차트에는 **툴팁용 지도**만 남긴다(리본은 내렸다).
  // 2026-09-21 변동성 전망(24h) 제거 -- 워커 중지로 데이터가 끊긴다. 그리기 분기는 그대로
  // 두고 지도만 비운다(diff 를 좁힌다).
  const volMapOn = false;

  // Liquidation-map density heatmap -- drawn first so candles/grid/lines sit on top of it (paint
  // order unchanged). 2026-08-25: replaced the old right-anchored, length-encoded "volume profile"
  // bar (capped at 30% of chart width, "left 70% stays clean for candles") with a full-width
  // background band, color intensity encoding density via a single sequential colormap
  // (densityColor) -- matches Coinglass's liquidation-heatmap convention at the user's explicit
  // request ("전체폭으로 가자"), reversing that earlier candle-clean design (twice rejected before
  // for the opposite reason -- widening the bar ate into candle space; a full-width BACKGROUND
  // band is a different tradeoff the user chose knowingly). Candles are opaque and painted after
  // this block, so they still read clearly on top wherever they overlap a band.
  //
  // Color scale is percentile-clipped (not raw min-max) so one outlier bin doesn't wash every other
  // band down to near-invisible -- mirrors Coinglass's own "유동성 임계값" control (their example
  // reading ~0.91). Computed across every bin in every snapshot of densityHistory (not just the
  // newest), so brightness stays comparable across time -- a genuinely quiet hour reads dim rather
  // than being rescaled to look as loud as the strongest hour (weight_pct is already globally
  // normalized server-side too, see compute_heatmap_history()'s docstring -- this clip is a 2nd,
  // display-only step on top of that, same as before).
  //
  // 2026-08-25 (2nd pass, same day): replaced one-way sweep-darkening -- a bin went from its live
  // color to permanently dark (t=0) the instant price first swept it, then stayed dark for the rest
  // of the chart no matter what happened afterward -- with genuine per-time-column density.
  // densityHistory is now a TIME SERIES (compute_heatmap_history(), one causal snapshot per hourly
  // kline boundary, oldest-to-newest), so a swept bin can go dark and then re-light later if fresh
  // volume genuinely re-accumulates at that price, matching a real Coinglass screenshot the user
  // compared against (see eth_liquidation_map_coinglass_visual_logic_replication_20260825 memory --
  // this exact gap is what they pointed at). Each snapshot draws only across the candle columns its
  // own hour actually covers (from its boundary up to the next snapshot's boundary), so a bin's
  // color changes in discrete steps at each hourly boundary and holds flat across that hour's ~12
  // five-minute candles in between -- there's no finer-grained truth to show between them, since the
  // underlying model itself only updates once per hourly kline. The chart's own visible window was
  // narrowed to SNAPSHOT_CHART_MAX_CANDLES (6h) the same day specifically so every visible column
  // has a real snapshot behind it (compute_heatmap_history()'s HEATMAP_HISTORY_DISPLAY_HOURS).
  const DENSITY_PERCENTILE_CLIP = 0.90;
  const densityValues = (densityHistory || [])
    .flatMap(snap => (snap.bins || []).map(b => b.weightPct || 0))
    .filter(v => v > 0)
    .sort((a, b) => a - b);
  const densityClip = densityValues.length
    ? densityValues[Math.min(densityValues.length - 1, Math.floor(densityValues.length * DENSITY_PERCENTILE_CLIP))]
    : 1;
  const drawDensitySeg = (into, x0, x1, top, bottom, t) => {
    if (x1 <= x0) return;
    // 밀도 0 인 칸은 **그리지 않는다**. 칠하면 패널 위에 띠로 남고(2026-09-12 b5a4790 으로
    // 패널이 밝아진 뒤 «어두운 구멍» 으로 드러났다), 안 그리면 배경이 그대로 비쳐 패널의
    // 세로 그라디언트가 어떻든 정확히 녹는다. 알파를 t 에 비례시키는 건 잘못된 고침이었다 --
    // densityClip 이 양수 밀도의 90분위라 대부분의 칸이 t<0.25 에 몰려 히트맵 전체가 흐려졌다.
    if (!(t > 0)) return;
    const rect = document.createElementNS(NS, "rect");
    rect.setAttribute("x", x0); rect.setAttribute("y", top);
    rect.setAttribute("width", x1 - x0); rect.setAttribute("height", bottom - top);
    rect.setAttribute("fill", densityColor(t));
    // 두 모드가 **같은 알파**를 쓴다 -- 풋프린트는 셀과 겹치지 않는 게이트에 그리므로
    // 낮출 이유가 없고, 낮추면 같은 값이 두 색으로 보인다(위 전체폭 주석의 실측).
    rect.setAttribute("fill-opacity", "0.85");
    into.appendChild(rect);
  };
  const sortedDensityHistory = (densityHistory || []).slice().sort((a, b) => a.tsMs - b.tsMs);
  const densityBoundaryIdx = sortedDensityHistory.map((snap) => {
    const tsSec = Math.floor((snap.tsMs || 0) / 1000);
    const idx = candles.findIndex(c => c.time >= tsSec);
    return idx === -1 ? candles.length : idx;
  });
  // Union of every price bucket seen in ANY snapshot (shared binWidth across the whole history --
  // compute_heatmap_history() holds the price grid fixed, see its docstring), drawn in EVERY
  // snapshot's own time-range even where that snapshot's own weight is 0/absent -- so an
  // already-swept, not-yet-reaccumulated price paints the same darkest color (t=0) a genuinely
  // near-zero bin would, instead of leaving a transparent gap that'd read as a different (background)
  // color -- matches Coinglass's continuous-shading look (every price row painted at every column).
  const densityBinWidth = sortedDensityHistory.length ? (sortedDensityHistory[0].binWidth || 0) : 0;
  // 🔴밀도 층은 이 화면에서 **가장 큰 덩어리**다(청산맵 1h 실측 rect 301/렌더). 입력은
  //   /api/liquidation-map 이 갱신될 때(60초)만 바뀌는데 매 렌더 다시 만들고 있었다.
  //   신원은 스냅샷의 개수와 양 끝 시각으로 충분하다(같은 payload 면 같은 값).
    // 🔴키에 **모드**를 넣는다. 불투명도가 모드마다 다른데(풋프린트 0.25 / 청산맵 0.85)
  //   baseGeomSig 에는 모드가 없어서, 기하가 우연히 같으면 옛 불투명도 층을 그대로
  //   재사용한다. 캐시가 «맞는 그림»을 돌려주는지는 키가 정한다.
  if (sortedDensityHistory.length) cachedLayer("density",
      objToken(densityHistory) + ":" + densityClip, (g) => {
  const densityPriceUnion = Array.from(new Set(sortedDensityHistory.flatMap(snap => (snap.bins || []).map(b => b.price))));
  sortedDensityHistory.forEach((snap, si) => {
    const xStartIdx = densityBoundaryIdx[si];
    const xEndIdx = si + 1 < densityBoundaryIdx.length ? densityBoundaryIdx[si + 1] : candles.length;
    if (xEndIdx <= xStartIdx) return; // this snapshot's hour has no visible candle (off-screen)
    const x0 = xAt(xStartIdx), x1 = xAt(xEndIdx);
    const weightByPrice = new Map((snap.bins || []).map(b => [b.price, b.weightPct || 0]));
    densityPriceUnion.forEach((price) => {
      const half = densityBinWidth / 2;
      const top = Math.max(mt, yAt(price + half));
      const bottom = Math.min(plotBottom, yAt(price - half));
      if (bottom <= top) return;
      const pct = clamp01(weightByPrice.get(price) || 0);
      const t = densityClip > 0 ? Math.min(1, pct / densityClip) : 0;
      drawDensitySeg(g, x0, x1, top, bottom, t);
    });
  });
  });

  // Resistance/support/current/entry price tags -- computed here (before the axis ticks below) so
  // the tick loop can tell when a grid label would land on top of one of these and skip it.
  // Rendering (the actual lines/boxes) still happens later, after candles/markers, so paint order
  // is unchanged.
  const priceLabels = [];
  // 2026-09-16 풋프린트에서는 현재가를 **가로선이 아니라 삼각형**으로 찍는다(사용자 요청).
  // 선은 가격 행을 가로질러 셀 숫자를 덮는데, 하필 현재가 근처가 제일 중요한 행이다.
  // 삼각형은 플롯 **바깥**(오른쪽 가장자리)에 앉아 어느 행인지만 가리키고 아무것도 안 가린다.
  // 청산맵 모드는 그대로 선이다 -- 거기선 덮을 셀이 없고, 선이 가격대를 가로로 읽게 해 준다.
  if (includeCurrentPrice && currentPrice > 0) {
    priceLabels.push({ val: currentPrice, color: "var(--accent)", label: "현재", dashed: true,
                       width: 2, marker: !!footprint });
  }
  // 2026-09-19 풋프린트에서는 진입선도 **삼각형**이다. 선이 가격 행을 가로질러 셀 숫자를
  // 덮는 문제는 현재가에서 이미 겪었고(바로 위 주석), 진입선은 굵기 3이라 더 넓게 덮는다.
  // 청산맵 모드는 그대로 선 -- 거기선 덮을 셀이 없다.
  if (entryPrice > 0) {
    priceLabels.push({ val: entryPrice, color: "var(--amber)", label: "진입", dashed: false,
                       width: 3, marker: !!footprint });
  }
  (riskLevels || []).forEach((level) => {
    if (Number(level.val) > 0) priceLabels.push(level);
  });

  // Precompute clamped position/off-view state before sorting -- the liquidation-map overlay can
  // carry up to a dozen levels several % away from price while this chart's own candle history
  // spans only ~8h (5m x100), so most or all of them land off-screen and clamp to one of two
  // identical edge pixels. The decluttering pass below needs that clamped position, not the raw
  // one, or it can't tell they collide.
  priceLabels.forEach(p => {
    const rawY = yAt(p.val);
    p.offTop = rawY < mt;
    p.offBottom = rawY > plotBottom;
    p.outOfView = p.offTop || p.offBottom;
    p.realY = p.outOfView ? (p.offTop ? mt + 2 : plotBottom - 2) : rawY;
  });

  // Sort by Y position (Price descending = Y ascending). Off-view levels on the same edge tie on
  // realY; stable sort then falls back to insertion order (callers pass nearest-to-price first) --
  // so the cascade below places the most relevant level closest to the edge and farther ones
  // deeper into the chart.
  priceLabels.sort((a, b) => a.realY - b.realY);

  // Adjust Y to avoid overlap. Off-view levels clamped to the same edge pixel need an
  // unconditional cascade, not a "would this collide" check -- a distance-gated nudge only fires
  // once, since every clamped item after the first sits exactly minGap*k away from a still-
  // identical realY and the gate never re-triggers. In-view levels (current/entry price) keep the
  // original "only nudge if actually close" behavior since their true positions are meaningful.
  // minGap must exceed the price-tag box height (18px, drawn below) or cascaded boxes touch
  // edge-to-edge with no visible gap between them -- 22 leaves a small visible seam.
  const minGap = 22;
  let topStack = mt + 2, bottomStack = plotBottom - 2;
  priceLabels.forEach((p, i) => {
    if (p.offTop) {
      p.adjustedY = topStack;
      topStack += minGap;
    } else if (p.offBottom) {
      p.adjustedY = bottomStack;
      bottomStack -= minGap;
    } else if (i > 0) {
      const prev = priceLabels[i - 1];
      const prevY = prev.adjustedY !== undefined ? prev.adjustedY : prev.realY;
      if (Math.abs(p.realY - prevY) < minGap) p.adjustedY = prevY + minGap;
    }
  });

  // 격자·x눈금은 기하와 봉 시각만 보므로 한 층으로 묶어 캐시한다(cachedLayer 주석).
  cachedLayer("grid", "", (g) => {
  // Grid & Y-Axis Ticks
  axisTicks(yMin, yMax, 6).forEach(t => {
    const y = yAt(t);
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", ml); line.setAttribute("x2", w - mr);
    line.setAttribute("y1", y); line.setAttribute("y2", y);
    line.setAttribute("class", "chart-grid");
    g.appendChild(line);

    // 2026-09-09 사용자 요청: **y축 가격 눈금 라벨을 없앤다**(격자선은 유지).
    //   현재/롱익절/지지선 같은 **라인 태그**는 priceLabels 로 계속 그린다 -- 그쪽이 실제로
    //   읽는 값이고, 눈금 숫자는 같은 오른쪽 열에서 그 태그와 자리를 다투기만 했다.
    //   (그래서 있던 collidesWithPriceTag 충돌 회피도 함께 사라진다 -- 눈금이 없으면 충돌도 없다)
  });

  const xTickCount = isMobileChartMode() ? 4 : 6;
  const xTickStep = Math.max(1, Math.floor((candles.length - 1) / Math.max(1, xTickCount - 1)));
  const xTickIndexes = [];
  for (let i = 0; i < candles.length; i += xTickStep) xTickIndexes.push(i);
  const lastIdx = candles.length - 1;
  // Force-adding lastIdx unconditionally could land it only a few px from the previous regular
  // tick (whenever xTickStep doesn't evenly divide candles.length-1), overlapping both bold 13px
  // "HH:MM" labels -- 2026-08-27 user report. Merge into the last regular tick instead of adding a
  // second one when they'd render too close together to read.
  const minTickPx = mobileChart ? 46 : 52;
  if (xTickIndexes.length && xAt(lastIdx) - xAt(xTickIndexes[xTickIndexes.length - 1]) < minTickPx) {
    xTickIndexes[xTickIndexes.length - 1] = lastIdx;
  } else if (!xTickIndexes.includes(lastIdx)) {
    xTickIndexes.push(lastIdx);
  }

  xTickIndexes.forEach((idx) => {
    const c = candles[idx];
    if (!c) return;
    const x = xAt(idx) + bw / 2;
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", x);
    line.setAttribute("x2", x);
    line.setAttribute("y1", h - mb);
    line.setAttribute("y2", h - mb + 5);
    line.setAttribute("stroke", "var(--line)");
    g.appendChild(line);

    const txt = document.createElementNS(NS, "text");
    txt.setAttribute("x", x);
    txt.setAttribute("y", h - mb + 21);
    txt.setAttribute("text-anchor", "middle");
    txt.setAttribute("font-size", isMobileChartMode() ? "12" : "13");
    txt.setAttribute("font-weight", "700");
    txt.setAttribute("fill", "var(--muted)");
    txt.textContent = fmtDateTick(c.time * 1000);
    g.appendChild(txt);
  });

  });
  // 레짐 리본은 워커 주기(300초)에만 바뀐다 -- 봉당 rect+title 이라 48봉이면 96 노드다.
  cachedLayer("regime", objToken(latestRegimeForChart) + ":"
    + regimeRibbonWaiting + ":" + regimeRibbonUnsupported, (g) => {
  if (regimeByTsForChart) {
    drawLaneTrack(REGIME_RIBBON_Y, g);   // 천장·바닥과 같은 트랙 (2026-09-16)
    candles.forEach((c, i) => {
      const r = regimeByTsForChart.get(c.time);
      if (!r) return;
      // 2026-09-16 사용자 "회색칸을 제거해야해. chop 은 그냥 백그라운드로": 횡보는 **안 그린다**.
      // 회색 칸을 채우면 «신호가 있다»처럼 읽히는데 횡보는 오히려 «아무것도 아니다»다.
      // 빈 트랙이 그 뜻을 그대로 말한다 -- 칸이 없는 구간 = 횡보. (값은 툴팁으로 계속 읽힌다)
      if (regimeDominant(r) === "chop") return;
      const rect = document.createElementNS(NS, "rect");
      rect.setAttribute("x", xAt(i)); rect.setAttribute("y", REGIME_RIBBON_Y);
      rect.setAttribute("width", laneW); rect.setAttribute("height", REGIME_RIBBON_H);
      rect.setAttribute("rx", laneRx);
      rect.setAttribute("fill", REGIME_DOMINANT_COLOR[regimeDominant(r)]);
      rect.setAttribute("fill-opacity", (0.55 + 0.45 * clamp01(r.confidence)).toFixed(2));
      const title = document.createElementNS(NS, "title");
      const pct = (v) => Math.round(v * 100);
      title.textContent = `레짐: 강세${pct(r.bull_prob)}% 약세${pct(r.bear_prob)}% 횡보${pct(r.chop_prob)}% (신뢰도${pct(r.confidence)}%)`;
      rect.appendChild(title);
      g.appendChild(rect);
    });
    const ribbonLabel = document.createElementNS(NS, "text");
    ribbonLabel.setAttribute("x", ml - 6);
    // 리본이 두꺼워졌으니 바닥 정렬(H-1) 대신 세로 중앙 (font-size 9 -> baseline +3)
    ribbonLabel.setAttribute("y", REGIME_RIBBON_Y + REGIME_RIBBON_H / 2 + 3);
    ribbonLabel.setAttribute("text-anchor", "end");
    ribbonLabel.setAttribute("font-size", "9");
    ribbonLabel.setAttribute("fill", "var(--muted)");
    ribbonLabel.textContent = "레짐";
    g.appendChild(ribbonLabel);
  } else if ((regimeRibbonWaiting || regimeRibbonUnsupported) && candles.length) {
    // Flat gray placeholder instead of silently drawing nothing, so the row still reads as
    // intentional -- but the two causes get different wording (regimeRibbonWaiting: transient,
    // auto-retries; regimeRibbonUnsupported: this coin has no trained regime model at all yet, see
    // regimeRibbonUnsupported's own definition above) so a permanent gap never reads as "any
    // second now".
    const waitRect = document.createElementNS(NS, "rect");
    waitRect.setAttribute("x", xAt(0)); waitRect.setAttribute("y", REGIME_RIBBON_Y);
    waitRect.setAttribute("width", xAt(candles.length - 1) + bw - xAt(0)); waitRect.setAttribute("height", REGIME_RIBBON_H);
    waitRect.setAttribute("rx", "1.5");
    waitRect.setAttribute("fill", "var(--muted)");
    waitRect.setAttribute("fill-opacity", "0.18");
    const waitTitle = document.createElementNS(NS, "title");
    waitTitle.textContent = regimeRibbonUnsupported
      // ⚠️지원 목록을 하드코딩하지 않는다 -- "(ETH 전용 모델)"이 박혀 있어 BTC·XRP 분류기를
      // 붙인 뒤에도 낡은 문구가 남았다. 실제 소스맵에서 만든다.
      ? `레짐: 이 코인용 레짐분류기가 아직 없음 (${Object.keys(REGIME_SOURCE_BY_ASSET).map((a) => a.toUpperCase()).join("·")} 지원) -- 추후 학습 예정`
      : "레짐: 웜업 중이거나 일시적으로 갱신 실패 -- 다음 5분 주기에 자동 재시도됩니다";
    waitRect.appendChild(waitTitle);
    g.appendChild(waitRect);
    const waitLabel = document.createElementNS(NS, "text");
    waitLabel.setAttribute("x", ml - 6);
    waitLabel.setAttribute("y", REGIME_RIBBON_Y + REGIME_RIBBON_H - 1);
    waitLabel.setAttribute("text-anchor", "end");
    waitLabel.setAttribute("font-size", "9");
    waitLabel.setAttribute("fill", "var(--muted)");
    waitLabel.textContent = regimeRibbonUnsupported ? "레짐 (미지원)" : "레짐";
    g.appendChild(waitLabel);
  }

  });

  // ── 변동성 전망 리본 (2026-09-11, 칩을 대체) ────────────────────────────────────
  // 사용자: "변동성 전망도 청산맵 아래 레짐과 같은 스타일로 주황색으로 칠하고 칩은 제거".
  // 🔴이 신호는 **시간봉**이다(피쳐가 resample("1h")). 캔들은 5분이라 시각으로 접어 칠한다 --
  //   한 시간이 12개 봉에 같은 색으로 깔린다. 청산 레인의 5분 키와 혼동하면 안 된다.
  // ⭐등급을 색 하나로 뭉개지 않는다: 「위험」 홀드아웃 정밀도 0.793 vs 「주의」 0.161 이라
  //   같은 주황으로 칠하면 16% 짜리가 79% 처럼 보인다. 진하기로 셋을 가른다.
  // 변동성 시각->값 매핑. 리본과 **툴팁이 같은 지도를 쓴다** -- 따로 만들면 화면의 두 곳이
  // 다른 값을 말하는 날이 온다(이 파일에서 반복된 실패).
  const volByHour = new Map();
  if (volMapOn) {
    const vf = null;   // 2026-09-21 제거 (volMapOn=false 라 이 분기는 안 돈다)
    const grades = Array.isArray(vf.grades) ? vf.grades : [];
    const probas = Array.isArray(vf.probas) ? vf.probas : [];
    (vf.times || []).forEach((iso, i) => {
      const t = Date.parse(iso);
      if (!Number.isFinite(t)) return;
      volByHour.set(Math.floor(t / 1000), { grade: grades[i] || null, p: probas[i], tone: (vf.history || [])[i] });
    });
  }
  // 2026-09-16 **변동성 리본을 차트에서 내렸다**(카드로 옮김). 중복이어서가 아니라 **해상도**
  // 때문이다 -- 리본은 1시간 격자·24시간 지평인데 차트는 5분봉이라 12봉이 한 색이고, 풋프린트
  // (1시간 창)에서는 값이 **하나**다. 실측: 최근 48시간 중 29시간이 연속 '안정'.
  // ⚠️버리는 게 아니다. 09-14 정면비교에서 셋 중 **유일하게 «확장»을 보는** 지표이고
  //   (현재변동성과 ρ −0.817) 게이트와도 중복이 아니다(상관 −0.389 · 상위10% 겹침 0.7%).
  //   다만 그 우위는 **24시간에서만** 실재한다(1h AUC .612 vs 공짜 대조군 .611 동률,
  //   4h .593 vs .720 더 나쁨, 24h .812 vs .625 진짜). 24시간 판단은 카드 한 줄이 맞는 자리다.


  // Candles -- 풋프린트 모드(ETH)면 몸통 대신 «가격 행별 공격적 매수/매도 체결량» 셀을 그리고,
  // 캔들은 얇은 테두리로만 남긴다(2026-09-15, TradingView 볼륨 풋프린트 '매수 및 매도' 유형).
  if (footprint) {
    // 행 크기는 TradingView '자동'(0.2 x ATR)에서 출발하되, 서버 버킷(0.5달러)의 배수여야 하고
    // 화면에서 FOOTPRINT_MIN_ROW_PX 이상이어야 한다. 조용한 날 ATR 만 따르면 행이 3px 로
    // 뭉개져 숫자가 안 들어간다 -- 읽히게 만드는 건 ATR 이 아니라 이 픽셀 하한이다.
    const bucket = footprint.bucket;
    const trs = candles.map((c, i) => (i === 0 ? c.high - c.low : Math.max(
      c.high - c.low, Math.abs(c.high - candles[i - 1].close), Math.abs(c.low - candles[i - 1].close))));
    const atr = trs.reduce((a, b) => a + b, 0) / Math.max(trs.length, 1);
    const minRow = Math.ceil((ySpan * FOOTPRINT_MIN_ROW_PX / ch) / bucket) * bucket;
    const rowSize = Math.max(bucket, minRow, Math.round(0.2 * atr / bucket) * bucket);
    const rowPx = rowSize * ch / ySpan;

    // 서버 버킷을 화면 행으로 묶는다. 서버는 0.5달러로만 모아 보내고, 행 크기는 여기서 정한다
    // -- 행을 바꾸려고 테이프를 다시 수집할 일이 없게.
    const barRows = candles.map((c) => {
      const rows = new Map();
      (footprint.byTime.get(c.time) || []).forEach((lvl) => {
        const key = Math.floor(lvl[0] / rowSize);
        const cell = rows.get(key) || [0, 0];
        cell[0] += lvl[1]; cell[1] += lvl[2];
        rows.set(key, cell);
      });
      return rows;
    });
    // 배경 4단계는 매수·매도 각각의 최대로 나눈다(TradingView: "매수/매도 측은 별도로 계산").
    let maxBuy = 0, maxSell = 0;
    barRows.forEach((rows) => rows.forEach((cell) => {
      maxBuy = Math.max(maxBuy, cell[0]); maxSell = Math.max(maxSell, cell[1]);
    }));
    const SHADES = footprintShades();
    const shade = (v, max) => SHADES[Math.min(3, Math.floor((max > 0 ? v / max : 0) * 4))];
    const fontPx = Math.min(9, Math.max(6, rowPx - 3));
    const half = Math.max(2, bw / 2 - 0.5);
    // 모바일에선 한 칸이 10px 도 안 된다("1.2k" 가 13px) -- 숫자를 포기하고 **색 농담만** 남긴다.
    // 숫자를 욱여넣으면 옆 칸을 침범해서 둘 다 못 읽는다. 값은 눌러서 툴팁으로 본다.
    const showQty = half >= 18;
    // 라이트는 셀 배경이 밝아 글자를 거의 불투명하게 올려야 읽힌다(다크는 현행 0.82).
    const INK_OPACITY =
      document.documentElement.getAttribute("data-theme") === "light" ? "0.95" : "0.82";

    // ── 봉별 캐시 (2026-09-20) ─────────────────────────────────────────────
    // 풋프린트 모드는 체결이 올 때마다 다시 그리는데(400ms 게이트) **바뀌는 건 맨 오른쪽 봉
    // 하나**다. 그런데 48봉의 셀을 매번 새로 만들고 있었다.
    // 🔴«마지막 봉만 그리면 된다»는 **그냥은 틀리다**. 창의 고저가 바뀌면 yMin/yMax 가 움직여
    //   모든 셀이 자리를 옮겨야 하고, 음영은 maxBuy/maxSell 로 정규화되며, 행 크기는 ATR 로
    //   정해진다. 그래서 캐시 키에 그 기하 전체를 넣는다 -- 하나라도 바뀌면 전부 무효다(맞다).
    // ⭐실측(틱 스트림 30초 · 12봉/48봉): 이 geomSig 는 렌더 124회 중 **0회** 바뀌었다.
    //   가격이 창의 고저 밴드 안에 머무는 동안은 축도 음영 기준도 고정이기 때문이다.
    //   돌파해서 새 고점을 만들면 그때 한 번 전부 다시 그린다.
    // ⭐봉 데이터의 신원은 `levels` **배열 객체 자체**다. ?since= 증분(refreshFootprint) 덕에
    //   안 바뀐 봉은 폴링 사이에 같은 배열을 그대로 들고 있어서 해시를 만들 필요가 없다.
    //   진행 중인 봉은 footprintMergeLive 가 매번 새 배열로 갈아끼우므로 늘 미스다(맞다).
    // 🔴델타 라벨과 POC 점은 **캐시하지 않는다**. 델타 y 는 앞선 봉들의 충돌회피 결과에
    //   의존해서(deltaBoxes) 중간 봉 하나만 바뀌어도 뒤쪽이 전부 틀어진다 -- 그 사슬을
    //   캐시에 들이면 조용히 어긋난다. 둘 다 노드 하나뿐이라 매번 만들어도 싸다.
    const geomSig = [w, h, mt, ch, ml, cw, bw, yMin, yMax, candles.length, rowSize, rowPx,
                     maxBuy, maxSell, half, fontPx, showQty, INK_OPACITY].join("|");
    const barCache = renderCandleSvg._barCache || (renderCandleSvg._barCache = new Map());
    // 델타 라벨은 봉 그룹 **밖**이라 따로 둔다(위 🔴주석: 충돌회피 사슬 때문).
    const deltaCache = renderCandleSvg._deltaCache || (renderCandleSvg._deltaCache = new Map());
    const seenBars = new Set();
    let barG = svg;            // drawCell 이하가 붙을 자리. 봉마다 아래 루프가 갈아끼운다.

    // 한 칸: 배경(거래량 비율 4단계) + 숫자 + 불균형 표시(반대편의 300% 초과면 바깥쪽 세로선).
    // 2026-09-19 «막대 길이» 인코딩(A안)을 되돌렸다 -- 사용자 지시. 길이는 비율을 보여주는
    // 대신 짧은 막대의 숫자를 지웠고, 여기서 읽는 건 그 숫자다. 칸을 다시 고정하고 체결량은
    // 사각형 안에 적는다. 색 농담은 원래대로 «크다/작다»를 말한다.
    const drawCell = (cx, yTop, v, other, max, color, edgeX, price) => {
      const rect = document.createElementNS(NS, "rect");
      rect.setAttribute("x", cx); rect.setAttribute("y", yTop);
      rect.setAttribute("width", half); rect.setAttribute("height", rowPx);
      rect.setAttribute("fill", color); rect.setAttribute("fill-opacity", shade(v, max));
      const title = document.createElementNS(NS, "title");
      const ratio = other > 0 ? v / other : Infinity;
      title.textContent = price.toFixed(1) + " · " + (color === "var(--good)" ? "매수 " : "매도 ")
        + v.toFixed(1) + " (반대편 " + other.toFixed(1) + ")"
        + (v > 0 && v > other * FOOTPRINT_IMBALANCE_RATIO
          ? " · 불균형 " + (Number.isFinite(ratio) ? ratio.toFixed(1) + "배" : "일방") : "");
      rect.appendChild(title);
      barG.appendChild(rect);
      if (v > 0 && rowPx >= 7 && showQty) {
        const txt = document.createElementNS(NS, "text");
        txt.setAttribute("x", cx + half / 2); txt.setAttribute("y", yTop + rowPx / 2 + fontPx * 0.36);
        txt.setAttribute("text-anchor", "middle"); txt.setAttribute("font-size", fontPx);
        txt.setAttribute("fill", "var(--ink)"); txt.setAttribute("fill-opacity", INK_OPACITY);
        txt.textContent = fmtFootprintQty(v);
        barG.appendChild(txt);
      }
      if (v > 0 && v > other * FOOTPRINT_IMBALANCE_RATIO) {
        const mark = document.createElementNS(NS, "rect");
        mark.setAttribute("x", edgeX); mark.setAttribute("y", yTop + 0.5);
        mark.setAttribute("width", 2); mark.setAttribute("height", Math.max(1, rowPx - 1));
        mark.setAttribute("fill", color);
        barG.appendChild(mark);
      }
    };

    // 봉 델타 라벨의 충돌 회피용. 모바일(봉 폭 ~25px)에서는 글자(~38px)가 봉보다 넓어
    // 이웃끼리 겹친다 -- 라벨을 버리지 않고 **겹치면 한 줄씩 내린다**(값은 다 보여야 한다).
    const deltaBoxes = [];
    const deltaFont = bw >= 34 ? 11 : 9;
    const pocPts = [];    // 2026-09-19 사용자 요청: 봉별 POC 를 선으로 잇는다.
    barRows.forEach((rows, i) => {
      const c = candles[i], x = xAt(i);
      let pocKey = null, pocVol = 0, buyTot = 0, sellTot = 0, lowKey = Infinity;
      rows.forEach((cell, key) => {
        buyTot += cell[0]; sellTot += cell[1];
        if (key < lowKey) lowKey = key;
        if (cell[0] + cell[1] > pocVol) { pocVol = cell[0] + cell[1]; pocKey = key; }
      });
      // POC «점»은 캐시 밖에서 매번 구한다(아래 폴리라인이 봉을 가로질러 잇기 때문).
      // 창 밖 행은 아래 그리기 루프가 건너뛰므로 여기서도 **같은 조건**으로 건너뛴다.
      if (pocKey !== null) {
        const pocY = yAt((pocKey + 1) * rowSize);
        if (!(pocY + rowPx < mt || pocY > mt + ch)) pocPts.push([x + bw / 2, pocY + rowPx / 2]);
      }
      seenBars.add(c.time);
      const levelsRef = footprint.byTime.get(c.time);
      const prev = barCache.get(c.time);
      const reuse = !!prev && prev.geom === geomSig && prev.levels === levelsRef && prev.i === i
        && prev.o === c.open && prev.h === c.high && prev.l === c.low && prev.c === c.close;
      barG = reuse ? prev.g : document.createElementNS(NS, "g");
      svg.appendChild(barG);
      if (!reuse) {
        barCache.set(c.time, { geom: geomSig, levels: levelsRef, i,
                               o: c.open, h: c.high, l: c.low, c: c.close, g: barG });
      rows.forEach((cell, key) => {
        const yTop = yAt((key + 1) * rowSize);
        if (yTop + rowPx < mt || yTop > mt + ch) return;   // 창 밖 행은 건너뛴다
        const price = key * rowSize + rowSize / 2;
        drawCell(x, yTop, cell[1], cell[0], maxSell, "var(--bad)", x - 2, price);          // 왼쪽 = 매도
        drawCell(x + bw / 2 + 0.5, yTop, cell[0], cell[1], maxBuy, "var(--good)", x + bw, price); // 오른쪽 = 매수
        if (key === pocKey) {   // POC -- 그 봉에서 가장 많이 거래된 가격 행
          const poc = document.createElementNS(NS, "rect");
          poc.setAttribute("x", x); poc.setAttribute("y", yTop);
          poc.setAttribute("width", bw); poc.setAttribute("height", rowPx);
          poc.setAttribute("fill", "none"); poc.setAttribute("stroke", "var(--amber)");
          poc.setAttribute("stroke-opacity", "0.85");
          const pocTitle = document.createElementNS(NS, "title");
          pocTitle.textContent = "POC " + price.toFixed(1) + " · 총 " + pocVol.toFixed(1);
          poc.appendChild(pocTitle);
          barG.appendChild(poc);
        }
      });

      // 캔들은 테두리로만 남긴다 -- 셀을 덮지 않으면서 시가/종가/꼬리를 잃지 않으려는 것.
    // (POC 선은 이 forEach 가 끝난 뒤 한 번에 긋는다 -- 아래 pocPts 블록)
      const isUp = c.close >= c.open, color = isUp ? "var(--good)" : "var(--bad)";
      const wick = document.createElementNS(NS, "line");
      wick.setAttribute("x1", x + bw / 2); wick.setAttribute("x2", x + bw / 2);
      wick.setAttribute("y1", yAt(c.high)); wick.setAttribute("y2", yAt(c.low));
      wick.setAttribute("stroke", color); wick.setAttribute("stroke-opacity", "0.5");
      barG.appendChild(wick);
      const body = document.createElementNS(NS, "rect");
      const yTop = yAt(Math.max(c.open, c.close)), yBot = yAt(Math.min(c.open, c.close));
      body.setAttribute("x", x); body.setAttribute("y", yTop);
      body.setAttribute("width", bw); body.setAttribute("height", Math.max(yBot - yTop, 1));
      body.setAttribute("fill", "none"); body.setAttribute("stroke", color);
      body.setAttribute("stroke-opacity", "0.6");
      barG.appendChild(body);
      }   // ← if (!reuse)

      // 봉 델타(매수-매도). 2026-09-16 사용자 요청으로 **플롯 맨 위 -> 그 봉 바로 아래**로
      // 옮기고 굵게 했다. 맨 위에 있을 때는 어느 봉의 숫자인지 눈이 세로로 훑어야 했다 --
      // 봉 밑에 붙으면 그 봉의 것임이 위치로 자명하다. y 는 그 봉의 **저가** 기준이라
      // 봉마다 높이가 다르다(고정 행이 아니다).
      const delta = buyTot - sellTot;
      if (buyTot + sellTot > 0) {
        // 기준은 캔들 저가가 아니라 **가장 아래 셀 행의 바닥**이다. 저가로 잡았더니 그 아래로
        // 더 내려오는 마지막 행과 글씨가 겹쳤다(2026-09-16 첫 판에서 실제로 겹쳤다) --
        // 행은 rowSize 격자라 저가보다 최대 한 행만큼 더 내려간다.
        const cellsBottom = Number.isFinite(lowKey) ? yAt(lowKey * rowSize) : yAt(c.low);
        let dY = Math.min(plotBottom - 3, Math.max(cellsBottom, yAt(c.low)) + 13);
        // 글자 폭 근사(굵은 숫자 ~0.62em) 로 상자를 만들고, 겹치면 아래로 한 줄씩 민다.
        const label = (delta >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(delta));
        const halfW = label.length * deltaFont * 0.34 + 2;
        const cxD = x + bw / 2;
        for (let guard = 0; guard < 6; guard++) {
          const hit = deltaBoxes.some((b) =>
            Math.abs(b.y - dY) < deltaFont + 1 && cxD + halfW > b.x1 && cxD - halfW < b.x2);
          if (!hit || dY + deltaFont + 2 > plotBottom - 3) break;
          dY += deltaFont + 2;
        }
        deltaBoxes.push({ x1: cxD - halfW, x2: cxD + halfW, y: dY });
        // 🔴여기까지의 **산수는 늘 돈다** -- dY 는 앞선 봉들의 충돌회피 결과에 달려 있어서
        //   건너뛰면 사슬이 끊긴다. 캐시하는 것은 «만들어진 노드»뿐이다.
        // ⭐그래서 키가 사슬 해시를 들 필요가 없다: dY 가 **이미 그 사슬의 결과**다.
        //   아래 다섯이 이 노드의 입력 전부이므로, 같으면 노드도 같다.
        const dSig = deltaFont + "|" + cxD + "|" + dY + "|" + label + "|" + c.time
                     + "|" + buyTot.toFixed(1) + "|" + sellTot.toFixed(1);
        const dPrev = deltaCache.get(c.time);
        if (dPrev && dPrev.sig === dSig) { svg.appendChild(dPrev.n); }
        else {
        const dTxt = document.createElementNS(NS, "text");
        dTxt.setAttribute("x", cxD); dTxt.setAttribute("y", dY);
        dTxt.setAttribute("text-anchor", "middle"); dTxt.setAttribute("font-size", String(deltaFont));
        dTxt.setAttribute("font-weight", "bold");
        dTxt.setAttribute("fill", delta >= 0 ? "var(--good)" : "var(--bad)");
        dTxt.textContent = label;
        const dTitle = document.createElementNS(NS, "title");
        dTitle.textContent = fmtDateTick(c.time * 1000) + " 델타 " + delta.toFixed(1)
          + " · 매수 " + buyTot.toFixed(1) + " / 매도 " + sellTot.toFixed(1);
        dTxt.appendChild(dTitle);
        svg.appendChild(dTxt);
        deltaCache.set(c.time, { sig: dSig, n: dTxt });
        }
      }
    });

    // 창 밖으로 나간 봉의 캐시는 버린다 -- 안 그러면 탭을 켜둔 채로 며칠이면 노드가 쌓인다.
    // 🔴호출부가 하나뿐이라(renderSnapshotChart) 캐시를 함수에 달아도 안전하다. 두 번째
    //   차트가 풋프린트를 쓰게 되면 svg 별로 갈라야 한다 -- 같은 <g> 를 두 svg 에 붙이면
    //   나중에 붙인 쪽으로 **옮겨간다**(appendChild 는 이동이다).
    if (barCache.size > seenBars.size) {
      [...barCache.keys()].forEach((t) => { if (!seenBars.has(t)) barCache.delete(t); });
    }
    if (deltaCache.size > seenBars.size) {
      [...deltaCache.keys()].forEach((t) => { if (!seenBars.has(t)) deltaCache.delete(t); });
    }

    // ── 봉별 POC 선 (2026-09-19 사용자 요청) ────────────────────────────
    // 봉마다 사각 테두리는 이미 있었지만 **봉끼리 독립**이라 «거래가 몰린 값이 어디로
    // 옮겨가는가»가 안 보였다. 그게 이 선이 유일하게 더 주는 정보다.
    // 🔴창 밖 행은 위 루프가 건너뛰므로 점이 빠진다 -- 선이 그 구간을 건너뛰고 이어지면
    //   없는 이동을 그리는 셈이다. 그래서 x 가 한 봉(bw) 넘게 벌어지면 **끊는다**.
    // 🔴«지지·저항»이 아니다. Dalton 밸런스엣지(비용게이트 0/6)·Yush LAF(무근거)로 닫혔다.
    if (pocPts.length > 1) {
      let seg = [pocPts[0]];
      const flush = () => {
        if (seg.length > 1) {
          const ln = document.createElementNS(NS, "polyline");
          ln.setAttribute("points", seg.map((q) => q[0] + "," + q[1]).join(" "));
          ln.setAttribute("fill", "none"); ln.setAttribute("stroke", "var(--amber)");
          ln.setAttribute("stroke-width", "1.5"); ln.setAttribute("stroke-opacity", "0.75");
          ln.setAttribute("stroke-linejoin", "round");
          const t = document.createElementNS(NS, "title");
          t.textContent = "봉별 POC(최대 체결 가격)의 이동. 체결이 몰린 값이지 지지·저항이 아니다.";
          ln.appendChild(t);
          svg.appendChild(ln);
        }
        seg = [];
      };
      for (let i = 1; i < pocPts.length; i++) {
        if (pocPts[i][0] - pocPts[i - 1][0] > bw * 1.8) { flush(); }
        seg.push(pocPts[i]);
      }
      flush();
    }

    // 백필 중에는 왼쪽 봉들이 아직 비어 있다 -- 그걸 «거래가 없었다»로 읽지 않게 말해 둔다.
    // 판정은 **수집기 상태(ready)** 만 본다. 캔들 개수로 재면, 봉이 바뀌는 순간 캔들 이력이
    // 아직 그 봉을 모를 때 다 찼는데도 「수집 중」이 남는다(2026-09-15 화면에서 실제로 봤다).
    if (!footprint.ready) {
      const warm = document.createElementNS(NS, "text");
      warm.setAttribute("x", ml + 4); warm.setAttribute("y", mt + ch - 4);
      warm.setAttribute("font-size", "9"); warm.setAttribute("fill", "var(--muted)");
      warm.textContent = "체결 테이프 수집 중 " + footprint.barCount + "/" + footprint.barsExpected + "봉";
      svg.appendChild(warm);
    }
  } else {
  // 풋프린트 **폴백**(2026-09-21 청산맵 모드 제거 후에도 남는 경로): 풋프린트는 ETH 전용이고
  // 테이프가 웜업 중이면 footprintForChart() 가 null 이라, 그때 그릴 캔들이 필요하다.
  // 청산 밀도·S/R 레벨·마커는 이 경로에서도 그대로 그려진다(모드 무관).
  candles.forEach((c, i) => {
    const x = xAt(i), isUp = c.close >= c.open, color = isUp ? "var(--good)" : "var(--bad)";
    const wick = document.createElementNS(NS, "line");
    wick.setAttribute("x1", x + bw/2); wick.setAttribute("x2", x + bw/2);
    wick.setAttribute("y1", yAt(c.high)); wick.setAttribute("y2", yAt(c.low));
    wick.setAttribute("stroke", color); svg.appendChild(wick);

    const body = document.createElementNS(NS, "rect");
    const yTop = yAt(Math.max(c.open, c.close)), yBot = yAt(Math.min(c.open, c.close));
    body.setAttribute("x", x); body.setAttribute("y", yTop);
    body.setAttribute("width", bw); body.setAttribute("height", Math.max(yBot - yTop, 1));
    body.setAttribute("fill", isUp ? "transparent" : color);
    body.setAttribute("stroke", color); svg.appendChild(body);
  });
  }

  // Trade Markers
  // Track markers per candle to avoid overlap
  const markerCounts = { top: {}, bottom: {} };

  (journal || []).forEach(t => {
    const ts = new Date(t.ts || t.closed_at).getTime() / 1000;
    const idx = candles.findIndex(c => c.time <= ts && ts < c.time + 300);
    if (idx === -1) return;
    
    const x = xAt(idx) + bw/2;
    const kind = String(t.kind || "").toUpperCase();
    const side = String(t.side || "").toUpperCase();
    const isEntry = kind.includes("OPEN") || kind.includes("ENTRY");
    
    const candle = candles[idx];
    const isLong = side === "LONG";
    const isBuy = (isLong && isEntry) || (!isLong && !isEntry);
    
    const sideKey = isBuy ? "bottom" : "top";
    const count = markerCounts[sideKey][idx] || 0;
    markerCounts[sideKey][idx] = count + 1;
    
    // Position based on Candle High/Low with stacking offset
    const stackOffset = count * 25; // 25px per additional marker
    const basePrice = isBuy ? candle.low : candle.high;
    const baseLineY = yAt(basePrice);
    const mY = isBuy ? baseLineY + 12 + stackOffset : baseLineY - 12 - stackOffset; 
    const lY = isBuy ? mY + 15 : mY - 10;            
    
    const marker = document.createElementNS(NS, "polygon");
    const pts = isBuy ? "0,-6 -6,6 6,6" : "0,6 -6,-6 6,-6";
    marker.setAttribute("points", pts);
    marker.setAttribute("transform", `translate(${x},${mY})`);
    marker.setAttribute("fill", isLong ? "var(--good)" : "var(--bad)");
    marker.setAttribute("stroke", "var(--chart-bg)"); marker.setAttribute("stroke-width", "1");
    svg.appendChild(marker);

    const lbl = document.createElementNS(NS, "text");
    lbl.setAttribute("x", x); lbl.setAttribute("y", lY);
    lbl.setAttribute("text-anchor", "middle"); lbl.setAttribute("font-size", "10");
    lbl.setAttribute("font-weight", "bold"); lbl.setAttribute("fill", isLong ? "var(--good)" : "var(--bad)");
    lbl.textContent = isEntry ? "진입" : "청산";
    svg.appendChild(lbl);
  });

  // ── 청산맵 신호 마커 (2026-09-09, 설계 3안 비교 후 C안 하이브리드 채택) ──────────────
  // 증거신호는 **고정 레인**(종수를 진하기로), 이벤트 트리거만 **봉 밀착 삼각형**.
  // 근거(87일 실측): 이 창은 72봉·6시간이고 컬럼 피치가 14.5px 뿐인데 증거신호는 6시간당
  // 중앙 10개·90분위 22개가 발동한다. 전부 봉에 붙이면 90분위 창에서 2.2컬럼당 하나가 되어
  // 캔들을 덮는다. 이벤트 트리거는 6시간당 0.7~5.5개뿐이라 봉에 붙여도 흩어지지 않는다.
  // 정보 등급도 다르다 -- 증거신호는 배경, 이벤트 트리거는 주장이다.
  // ⚠️ETH 전용. 다른 코인은 레인 자체를 그리지 않는다 -- 빈 레인은 "신호 없음"으로 오독된다.
  const cm = latestChartMarkers;
  if (isSnapshotChart && cm && cm.available && Array.isArray(cm.times)) {
    // 격자 정합은 **UTC epoch** 로 맞춘다(문자열 포맷 비교는 tz 표기 차이로 조용히 어긋난다).
    const idxByEpoch = new Map();
    candles.forEach((c, i) => idxByEpoch.set(c.time, i));
    // 2026-09-09 모바일 신고("천장·바닥 게이지가 잘 안 보이고 청산 밀도에 가려진다"):
    //   원인 둘. (1) 레인은 히트맵 **위에** 그려지지만 히트맵이 fill-opacity 0.85 비리디스
    //   밴드라 반투명 레인이 대비로 지워진다. (2) 모바일 기본 34봉에서 bw≈6.8px 인데
    //   rx=1.5 라운딩이 그 폭을 먹어 점처럼 보인다(최대 축소 72봉이면 bw 3.2px).
    //   → 레인마다 **불투명 트랙**을 깔아 히트맵을 끊고, 모바일에선 높이를 키우고 라운딩을
    //     빼고 최소 폭을 보장하고 불투명도 하한을 올린다.
    // 2026-09-10 데스크톱 6 / 모바일 9 -> 20(레짐 리본과 동일)으로 키웠다가, 2026-09-11
    // 사용자 지시로 **15** 로 낮췄다. 모바일이 데스크톱보다 컸던 건 2026-09-09 가시성
    // 신고("천장·바닥 게이지가 잘 안 보이고 청산 밀도에 가려진다") 때문인데 15 면 그 이유가
    // 해소되므로 한 값으로 통일한다.
    // ⚠️레인은 플롯 **위에 겹쳐** 그린다(레짐 리본과 달리 하단 여백 밖이 아니다). 그래서
    //   높이가 곧 캔들을 가리는 면적이다 -- 상·하 15px 씩이면 데스크톱 플롯 324 중
    //   30px(9%), 모바일 184 중 16%. 20 일 때는 12%/22% 였다.
    // 2026-09-11: 플롯 **안**(top: mt+3 / bottom: h-mb-3-LANE_H)에서 여백 **밖**으로 옮겼다.
    //   천장은 플롯 위, 바닥은 x축 라벨 아래 -- 위/아래 공간 은유는 그대로 유지한다.
    const laneFill = { top: "var(--bad)", bottom: "var(--good)" };
    // 2026-09-16 사용자 지시로 **천장·바닥 레인을 제거**했다. 그 레인은 증거신호 종수를
    // 진하기로 보여주던 것인데(2026-09-09 C안), 증거신호가 화면에서 빠지면서 빈 줄만 남았다.
    // 방향 이벤트는 아래 **봉 밀착 삼각형** 하나로 말한다 -- 같은 것을 두 문법으로 말하지 않는다.

    // ── 방향 없는 신호는 구간으로 (2026-09-16) ──────────────────────────────────
    // 추세 전환·변동폭 게이트는 **천장/바닥을 말하지 않는다**. 그래서 방향 색(초록/빨강)을
    // 쓰지 않고 앰버/주황 계열만 쓰며, 삼각형(순간)이 아니라 **막대 길이**(구간)로 그린다 --
    // 지속 시간이 이 둘의 정보 대부분이라 점으로 찍으면 그게 사라진다.
    // 예고는 **현재 상태만** 아는 값이라(워커가 이력을 안 남긴다) 서버가 spans_partial 로
    // 알려주고, 여기서는 점선 테두리로 «과거는 모른다»를 표시한다.
    const spans = cm.spans || {};
    const partial = new Set(cm.spans_partial || []);
    const spanRows = [
      // 왼쪽 여백은 45px 뿐이다 -- 이름은 기존 규약(레짐·변동성·청산)처럼 두 글자로 줄인다.
      // 긴 이름을 넣었더니 "변동폭 게이트"가 "폭 게이트"로 잘렸다(2026-09-16 렌더 확인).
      { y: TREND_ROW_Y, label: "전환", items: [
        { key: "trend_prewarn", name: "전환 예고", color: "var(--warn)", opacity: 0.30 },
        { key: "trend_detect", name: "전환 탐지", color: "var(--warn)", opacity: 0.95 }] },
      // 2026-09-20 «게이트» 줄을 걷어냈다(사용자 지시). 값 자체는 카드
      //   (evrGateIndicatorItem)에 남아 있다 -- 지운 건 차트 줄 하나다.
    ];
    // 구간 줄은 cm(60초 폴링)에만 달려 있다. 🔴아래 **이벤트 삼각형은 캐시하지 않는다** --
    // y 가 그 봉의 고가/저가에서 나오므로 진행 중인 봉에서 틱마다 움직인다.
    cachedLayer("spans", objToken(cm), (lg) => {
    spanRows.forEach((row) => {
      drawLaneTrack(row.y, lg);
      const lbl = document.createElementNS(NS, "text");
      lbl.setAttribute("x", ml - 6); lbl.setAttribute("y", row.y + LANE_H / 2 + 3);
      lbl.setAttribute("text-anchor", "end"); lbl.setAttribute("font-size", "9");
      lbl.setAttribute("fill", "var(--muted)");
      lbl.textContent = row.label;
      lg.appendChild(lbl);
      row.items.forEach((item) => {
        const raw = Array.isArray(spans[item.key]) ? spans[item.key] : [];
        // 🔴인덱스로 맞추면 안 된다. 서버 격자는 72봉인데 **풋프린트는 12봉**이라 길이가 다르다
        //   -- 첫 판은 길이 검사에 걸려 풋프린트에서 구간이 영영 안 보였다. 시각으로 맞춘다
        //   (삼각형이 idxByEpoch 로 하는 것과 같은 방법이고, 한 칸 밀기도 같이 막힌다).
        const onByTs = new Map();
        (cm.times || []).forEach((t, k) => {
          const sec = Math.floor(Date.parse(t) / 1000);
          if (Number.isFinite(sec)) onByTs.set(sec, raw[k] ? 1 : 0);
        });
        const arr = candles.map((c) => onByTs.get(c.time) || 0);
        let i = 0;
        while (i < arr.length) {
          if (!arr[i]) { i += 1; continue; }
          let j = i;
          while (j + 1 < arr.length && arr[j + 1]) j += 1;   // 이어진 봉을 한 구간으로
          const x0 = xAt(i), x1 = xAt(j) + bw;
          const rect = document.createElementNS(NS, "rect");
          rect.setAttribute("x", x0); rect.setAttribute("y", row.y);
          rect.setAttribute("width", Math.max(x1 - x0, 2));
          rect.setAttribute("height", LANE_H); rect.setAttribute("rx", laneRx);
          rect.setAttribute("fill", item.color);
          rect.setAttribute("fill-opacity", String(item.opacity));
          if (partial.has(item.key)) {
            rect.setAttribute("stroke", item.color);
            rect.setAttribute("stroke-dasharray", "3,2");
            rect.setAttribute("stroke-opacity", "0.9");
          }
          const ti = document.createElementNS(NS, "title");
          const span = (j - i + 1) * CHART_CANDLE_MIN;
          ti.textContent = item.name + " · " + fmtDateTick(candles[i].time * 1000)
            + "부터 " + span + "분"
            + (partial.has(item.key) ? " (현재 상태만 압니다 — 과거 이력이 없습니다)" : "");
          rect.appendChild(ti);
          lg.appendChild(rect);
          // 구간이 충분히 넓을 때만 이름을 넣는다 -- 좁은 칸의 글자는 읽히지 않고 더럽기만 하다.
          if (x1 - x0 >= 64) {
            const t = document.createElementNS(NS, "text");
            t.setAttribute("x", x0 + 6); t.setAttribute("y", row.y + LANE_H / 2 + 3.5);
            t.setAttribute("font-size", "9"); t.setAttribute("font-weight", "bold");
            t.setAttribute("fill", inkOnFill());
            t.textContent = item.name;
            lg.appendChild(t);
          }
          i = j + 1;
        }
      });
    });

    });

    // 이벤트 트리거 -- 매매 저널과 **같은 삼각형 문법**을 쓰고 `markerCounts` 를 공유해
    // 같은 봉에서 저널 마커와 겹치지 않게 한다(스택 25px). 저널보다 한 치수 작게(±5) 그려
    // 실제 체결이 시각적으로 우선하게 둔다. 글자 라벨은 붙이지 않는다 -- 6시간당 최대 11개라
    // "진입/청산" 처럼 글자를 넣으면 글자밭이 된다.
    (cm.events || []).forEach(ev => {
      if (mobileChart && ev.grade === "약") return;   // 피치 3.5px -- 모바일은 강/중만
      const idx = idxByEpoch.get(Date.parse(ev.t) / 1000);
      if (idx === undefined || !candles[idx]) return;
      const isBottom = ev.side === "bottom";
      const sideKey = isBottom ? "bottom" : "top";
      const count = markerCounts[sideKey][idx] || 0;
      markerCounts[sideKey][idx] = count + 1;
      const baseLineY = yAt(isBottom ? candles[idx].low : candles[idx].high);
      const mY = isBottom ? baseLineY + 12 + count * 25 : baseLineY - 12 - count * 25;
      const marker = document.createElementNS(NS, "polygon");
      marker.setAttribute("points", isBottom ? "0,-5 -5,5 5,5" : "0,5 -5,-5 5,-5");
      marker.setAttribute("transform", `translate(${xAt(idx) + bw / 2},${mY})`);
      marker.setAttribute("fill", isBottom ? "var(--good)" : "var(--bad)");
      marker.setAttribute("fill-opacity", ev.grade === "약" ? "0.55" : "0.95");
      marker.setAttribute("stroke", "var(--chart-bg)");
      marker.setAttribute("stroke-width", "1");
      const ti = document.createElementNS(NS, "title");
      ti.textContent = `${ev.label}${ev.grade ? ` ${ev.grade}등급` : ""}`
        + ` · ${isBottom ? "바닥" : "천장"}`
        + `${ev.p != null ? ` · 확률 ${(Number(ev.p) * 100).toFixed(0)}%` : ""}`;
      marker.appendChild(ti);
      svg.appendChild(marker);
    });
  }

  priceLabels.forEach(p => {
    const labelYRaw = p.adjustedY !== undefined ? p.adjustedY : p.realY;
    const labelY = Math.max(mt + 9, Math.min(plotBottom - 9, labelYRaw));
    const lineDashed = p.dashed || p.outOfView;

    // Line stays at real (clamped) price position
    let line = null;
    if (p.marker) {
      // 플롯 오른쪽 가장자리에서 **왼쪽을 가리키는** 삼각형. 꼭짓점이 곧 그 가격의 행이다.
      const tri = document.createElementNS(NS, "polygon");
      tri.setAttribute("points", markerPoints(ml + cw, p.realY));
      tri.setAttribute("fill", p.color);
      if (p.outOfView) tri.setAttribute("opacity", "0.72");
      // ⚠️여기서 append 하지 않는다. 플롯 오른쪽 끝(ml+cw)과 가격 배지(w-mr+4)가 4px 차이라
      //   먼저 그리면 배지에 **가려진다**(2026-09-16 첫 판이 그래서 안 보였다). 배지 뒤에
      //   붙여 배지의 «꼬리»처럼 보이게 한다 -- 꼭짓점은 여전히 진짜 가격 행을 가리킨다
      //   (배지 자체는 겹침 회피로 위아래로 밀릴 수 있어서 행을 정확히 못 가리킨다).
      line = tri;   // 아래 data-live 표식과 빠른 갱신이 같은 변수를 쓴다
    } else {
      line = document.createElementNS(NS, "line");
      line.setAttribute("x1", ml); line.setAttribute("x2", w - mr);
      line.setAttribute("y1", p.realY); line.setAttribute("y2", p.realY);
      line.setAttribute("stroke", p.color);
      line.setAttribute("stroke-width", String(p.width || 2));
      if (lineDashed) line.setAttribute("stroke-dasharray", "4,4");
      if (p.outOfView) line.setAttribute("opacity", "0.72");
      svg.appendChild(line);
    }

    // Left label (follows label position)
    const txt = document.createElementNS(NS, "text");
    txt.setAttribute("x", ml - 5); txt.setAttribute("y", labelY + 4);
    txt.setAttribute("text-anchor", "end"); txt.setAttribute("font-size", "10");
    txt.setAttribute("font-weight", "bold"); txt.setAttribute("fill", p.color);
    txt.textContent = `${p.label}${p.offTop ? "↑" : p.offBottom ? "↓" : ""}`;
    svg.appendChild(txt);

    // Right box (follows label position)
    const boxW = mobileChart ? 56 : 64, boxH = 18;
    const rect = document.createElementNS(NS, "rect");
    rect.setAttribute("x", w - mr + 4); rect.setAttribute("y", labelY - 9);
    rect.setAttribute("width", boxW); rect.setAttribute("height", boxH);
    rect.setAttribute("fill", p.color); rect.setAttribute("rx", "2");
    svg.appendChild(rect);

    const pTxt = document.createElementNS(NS, "text");
    pTxt.setAttribute("x", w - mr + 8); pTxt.setAttribute("y", labelY + 4);
    pTxt.setAttribute("font-size", mobileChart ? "11" : "12"); pTxt.setAttribute("font-weight", "bold");
    pTxt.setAttribute("fill", inkOnFill());
    pTxt.textContent = `${p.offTop ? "↑ " : p.offBottom ? "↓ " : ""}${fmtNum(p.val, 1)}`;
    svg.appendChild(pTxt);
    if (p.marker) svg.appendChild(line);   // 배지 위에 -- 위 주석 참조

    // 현재가 줄만 표식을 단다 -- 틱마다 **이 세 요소만** 옮기려는 것이다(전체 재렌더는 1초).
    // 표식이 없으면 빠른 갱신이 어느 줄을 움직여야 하는지 알 수 없다.
    if (p.label === "현재" && isSnapshotChart) {
      line.dataset.live = p.marker ? "tri" : "line";
      rect.dataset.live = "box";
      pTxt.dataset.live = "text";
      txt.dataset.live = "label";
    }
  });
  // 빠른 갱신이 가격 -> y 를 계산하려면 이번 렌더의 축 규약이 필요하다. 다음 전체 렌더가
  // 덮어쓴다 -- 즉 이 값은 항상 «지금 화면에 그려진 것»과 같다.
  if (isSnapshotChart) {
    liveLineCtx = { svg, yMin, yMax, mt, ch, markerX: ml + cw };
  }

  // 2026-09-09 사용자 요청: 청산 밀도 가이드를 **차트 위(패널 HTML)** 로 옮겼다.
  //   기존에는 SVG 안 오른쪽 위 인셋(backing 이 y=0..34)이라, 같은 자리에 새로 생긴
  //   증거신호 **천장 레인**(당시 y=mt+3, 2026-09-11 에 mt-3-LANE_H 로 이동)을 덮었다.
  //   차트 높이를 잃으므로(모바일 -8%) 아예 SVG 밖으로 뺀다.
  //   렌더는 캔들 SVG 안, 프로파일 바닥글 바로 아래 줄이다(2026-09-21 아티팩트 댓글).

  // Create Hover Layer on Top
  const hoverGroup = document.createElementNS(NS, "g");
  hoverGroup.setAttribute("class", "hover-layer");
  svg.appendChild(hoverGroup);

  // y2 reaches through the regime ribbon on the Snapshot chart so hovering visibly crosses both
  // (2026-08-26, "레짐 그래프도 십자선에 걸쳤으면 좋겠어") -- plain candle chart baseline otherwise.
  const vLine = document.createElementNS(NS, "line");
  vLine.setAttribute("x1", 0); vLine.setAttribute("x2", 0);
  vLine.setAttribute("y1", mt);
  // 십자선은 아래 줄들까지 걸친다 -- 호버한 봉이 어느 구간에 속하는지 눈으로 잇게.
  vLine.setAttribute("y2", TREND_ROW_Y + LANE_H);   // 마지막 줄까지 (게이트 제거 후)
  vLine.setAttribute("stroke", "var(--hover-line)");
  vLine.setAttribute("stroke-dasharray", "4,4");
  vLine.style.display = "none";
  vLine.style.pointerEvents = "none";
  hoverGroup.appendChild(vLine);

  // Horizontal crosshair + price-at-cursor readout (2026-08-27 user request) -- independent of the
  // candle-snapped vLine/tooltip above, which stays unchanged. Follows the raw mouse Y continuously
  // rather than snapping to a candle's OHLC, so it answers "what price is under my cursor right
  // now" instead of "what did this bar do".
  const hLine = document.createElementNS(NS, "line");
  hLine.setAttribute("x1", ml); hLine.setAttribute("x2", w - mr);
  hLine.setAttribute("y1", 0); hLine.setAttribute("y2", 0);
  hLine.setAttribute("stroke", "var(--hover-line)");
  hLine.setAttribute("stroke-dasharray", "4,4");
  hLine.style.display = "none";
  hLine.style.pointerEvents = "none";
  hoverGroup.appendChild(hLine);

  const priceBadgeW = mobileChart ? 56 : 64, priceBadgeH = 18;
  const priceBadgeRect = document.createElementNS(NS, "rect");
  priceBadgeRect.setAttribute("x", w - mr + 4);
  priceBadgeRect.setAttribute("width", priceBadgeW);
  priceBadgeRect.setAttribute("height", priceBadgeH);
  priceBadgeRect.setAttribute("fill", "var(--accent)");
  priceBadgeRect.setAttribute("rx", "2");
  priceBadgeRect.style.display = "none";
  priceBadgeRect.style.pointerEvents = "none";
  hoverGroup.appendChild(priceBadgeRect);

  const priceBadgeText = document.createElementNS(NS, "text");
  priceBadgeText.setAttribute("x", w - mr + 8);
  priceBadgeText.setAttribute("font-size", mobileChart ? "11" : "12");
  priceBadgeText.setAttribute("font-weight", "bold");
  priceBadgeText.setAttribute("fill", inkOnFill());
  priceBadgeText.style.display = "none";
  priceBadgeText.style.pointerEvents = "none";
  hoverGroup.appendChild(priceBadgeText);

  // Candlestick Tooltip Support. regimeByTsForChart/isSnapshotChart computed once near the top of
  // this function (shared with the ribbon drawn above) -- same ETH-only guard applies here.
  if (isSnapshotChart) svg.onmouseenter = () => { chartHoverActive = true; };
  svg.onmousemove = (evt) => {
    if (isSnapshotChart) chartHoverActive = true;   // enter 를 놓친 경우(재렌더 직후)도 잡는다
    const rect = svg.getBoundingClientRect();
    // 2026-08-28 user report: crosshair drifts from the real cursor position increasingly toward
    // the top/bottom edges. Root cause: viewBox="0 0 1200 400" (3:1) with preserveAspectRatio=
    // "xMidYMid meet" scales uniformly and centers -- whenever the container's own aspect ratio
    // isn't exactly 3:1 (the normal case), the rendered chart doesn't fill `rect` on one axis
    // (letterboxed), so the old formula ((evt.clientY - rect.top) * (h / rect.height)), which
    // assumed rect IS the rendered content box, under/over-scaled y more the further the cursor
    // sat from the vertical center -- exactly the reported "worse near top/bottom" symptom.
    // Correct conversion needs the actual uniform "meet" scale plus the centering offset it implies.
    const svgScale = Math.min(rect.width / w, rect.height / h);
    const svgOffsetX = (rect.width - w * svgScale) / 2;
    const svgOffsetY = (rect.height - h * svgScale) / 2;
    const mx = (evt.clientX - rect.left - svgOffsetX) / svgScale;
    const my = (evt.clientY - rect.top - svgOffsetY) / svgScale;

    if (my >= mt && my <= plotBottom) {
      hLine.setAttribute("y1", my); hLine.setAttribute("y2", my);
      hLine.style.display = "block";
      const priceAtCursor = yMax - ((my - mt) * ySpan) / ch;
      const badgeY = Math.max(mt, Math.min(plotBottom - priceBadgeH, my - priceBadgeH / 2));
      priceBadgeRect.setAttribute("y", badgeY);
      priceBadgeText.setAttribute("y", badgeY + 13);
      priceBadgeText.textContent = fmtNum(priceAtCursor, 1);
      priceBadgeRect.style.display = "block";
      priceBadgeText.style.display = "block";
    } else {
      hLine.style.display = "none";
      priceBadgeRect.style.display = "none";
      priceBadgeText.style.display = "none";
    }

    if (mx < ml || mx > w - mr) { hideTooltip(); return; }

    const idx = Math.min(candles.length - 1, Math.max(0, Math.floor(((mx - ml) / cw) * candles.length)));
    const c = candles[idx];
    if (!c) return;

    const tx = xAt(idx) + bw / 2;

    vLine.setAttribute("x1", tx);
    vLine.setAttribute("x2", tx);
    vLine.style.display = "block";

    const r = regimeByTsForChart ? regimeByTsForChart.get(c.time) : null;
    const regimeLine = r
      ? `<br>레짐: ${regimeDominant(r) === "bull" ? "강세" : regimeDominant(r) === "bear" ? "약세" : "횡보"} ${Math.round(Math.max(r.bull_prob, r.bear_prob, r.chop_prob) * 100)}%`
      : "";
    // 2026-09-16: 툴팁이 **그 봉에 표시된 모든 것**을 말한다. 그전에는 시각·OHLC·레짐뿐이라
    // 변동성 리본도, 새로 붙인 트리거 표시도 «왜 떴는지»를 화면에서 물어볼 데가 없었다.
    // 원칙: 값이 없는 항목은 줄 자체를 안 만든다(빈 줄은 «0» 으로 오독된다).
    const vv = volByHour.get(Math.floor(c.time / 3600) * 3600);
    const volLine = vv
      ? `<br>변동성: ${vv.grade || (vv.tone === "warn" ? "주의" : "안정")}`
        + (vv.p == null ? "" : ` · 확장 확률 ${(Number(vv.p) * 100).toFixed(1)}%`)
      : "";
    const cmNow = latestChartMarkers;
    let trigLines = "";
    if (isSnapshotChart && cmNow && cmNow.available) {
      const k = (cmNow.times || []).findIndex((t) => Math.floor(Date.parse(t) / 1000) === c.time);
      (cmNow.events || []).forEach((ev) => {
        if (Math.floor(Date.parse(ev.t) / 1000) !== c.time) return;
        trigLines += `<br>${ev.side === "bottom" ? "▲" : "▼"} ${ev.label}`
          + (ev.grade ? ` ${ev.grade}등급` : "")
          + (ev.p != null ? ` · 확률 ${(Number(ev.p) * 100).toFixed(0)}%` : "")
          + (ev.demoted ? " · 손실가중 헤드가 컷 미달로 강등" : "");
      });
      if (k >= 0) {
        const sp = cmNow.spans || {}, meta = cmNow.span_meta || {};
        const at = (arr) => (Array.isArray(arr) ? arr[k] : undefined);
        if (at(sp.trend_prewarn)) {
          const pp = at(meta.trend_prewarn_p), th = at(meta.trend_prewarn_thr);
          trigLines += "<br>전환 예고 (앞으로 30분 내 발동 확률)"
            + (pp == null ? "" : ` ${(pp * 100).toFixed(1)}%`)
            + (th == null ? "" : ` / 임계 ${(th * 100).toFixed(1)}%`);
        }
        if (at(sp.trend_detect)) trigLines += "<br>전환 탐지 지속 중 (거래대금·체결속도 둘 다 q90 초과)";
      }
    }
    showTooltip(evt.pageX, evt.pageY, `
      <b>${fmtDateTick(c.time * 1000)}</b><br>
      시가: ${fmtNum(c.open, 2)}<br>
      고가: ${fmtNum(c.high, 2)}<br>
      저가: ${fmtNum(c.low, 2)}<br>
      종가: ${fmtNum(c.close, 2)}${regimeLine}${volLine}${trigLines}
    `);
  };
  svg.onmouseleave = () => {
    if (isSnapshotChart) {
      chartHoverActive = false;
      if (chartRenderDeferred) { chartRenderDeferred = false; renderSnapshotChart(); return; }
    }
    hideTooltip();
    if (typeof hoverDot !== 'undefined') hoverDot.style.display = "none";
    if (typeof vLine !== 'undefined') vLine.style.display = "none";
    if (typeof hoverDots !== 'undefined') hoverDots.forEach(d => d.style.display = "none");
    hLine.style.display = "none";
    priceBadgeRect.style.display = "none";
    priceBadgeText.style.display = "none";
  };

  // ── 봉별 OI 신규계약 레인 (2026-09-19 사용자 지시) ──────────────────────────────
  // 그 5분에 **새로 생긴 계약 수**(미결제약정 증분). 바로 아래 청산 레인과 붙여 읽는다 --
  // 한쪽은 «새 포지션이 들어왔다», 다른 쪽은 «강제로 닫혔다»로 같은 사건의 양면이다.
  // ⚠️청산 레인과 달리 **선형 스케일**이다. OI 증분은 봉마다 자릿수가 비슷해서(청산의
  //   23,000배 같은 폭이 없다) 로그로 누르면 오히려 차이가 사라진다.
  // 회색 막대 = 수집 공백 뒤라 직전 봉과 이을 수 없어 봉 안에서만 잰 값(0이 아니라 «모름»).
  // OI 레인은 15초 폴링에만 달려 있다(봉당 rect+title).
  cachedLayer("oiLane", objToken(oiBars), (g) => {
  if (oiBars.length && candles.length) {
    const OI_Y = oiPanelY, OI_H = OI_PANEL_H, OI_MID = OI_Y + OI_H / 2;
    // 봉시각은 서버가 5분으로 바닥내림한 **초**다. 캔들 c.time 과 같은 단위·같은 격자다.
    const oiByTs = new Map(oiBars.map((b) => [Number(b[0]), b]));
    let oiPeak = 0;
    candles.forEach((c) => {
      const b = oiByTs.get(c.time);
      if (b) oiPeak = Math.max(oiPeak, Math.abs(Number(b[1]) || 0));
    });
    const oiTopLine = document.createElementNS(NS, "line");
    oiTopLine.setAttribute("x1", ml); oiTopLine.setAttribute("x2", ml + cw);
    oiTopLine.setAttribute("y1", OI_Y); oiTopLine.setAttribute("y2", OI_Y);
    oiTopLine.setAttribute("stroke", "var(--soft-line)");
    g.appendChild(oiTopLine);
    if (oiPeak > 0) {
      const oiHalf = OI_H / 2 - 1;
      const nowBar = Math.floor(Date.now() / 1000 / 300) * 300;
      candles.forEach((c, i) => {
        const b = oiByTs.get(c.time);
        if (!b) return;
        const d = Number(b[1]) || 0, gap = b[4];
        if (!d) return;
        const hgt = Math.max(1, oiHalf * Math.abs(d) / oiPeak);
        const rect = document.createElementNS(NS, "rect");
        rect.setAttribute("x", xAt(i));
        rect.setAttribute("y", d >= 0 ? OI_MID - hgt : OI_MID);
        rect.setAttribute("width", Math.max(1, bw));
        rect.setAttribute("height", hgt);
        rect.setAttribute("fill", gap ? "var(--neutral)" : (d >= 0 ? "var(--good)" : "var(--bad)"));
        rect.setAttribute("fill-opacity", c.time >= nowBar ? "0.42" : (gap ? "0.35" : "0.85"));
        const title = document.createElementNS(NS, "title");
        title.textContent = fmtDateTick(c.time * 1000) + " 신규계약 "
          + (d >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(d)) + " ETH"
          + " · OI " + Math.round(Number(b[2]) || 0).toLocaleString() + " · 스냅샷 " + b[3] + "개"
          + (gap ? " (앞 봉이 비어 봉 안에서만 쟀다 -- 0이 아니라 «모름»)" : "")
          + (c.time >= nowBar ? " (진행 중)" : "");
        rect.appendChild(title);
        g.appendChild(rect);
      });
      const oiMidLine = document.createElementNS(NS, "line");
      oiMidLine.setAttribute("x1", ml); oiMidLine.setAttribute("x2", ml + cw);
      oiMidLine.setAttribute("y1", OI_MID); oiMidLine.setAttribute("y2", OI_MID);
      oiMidLine.setAttribute("stroke", "var(--line)"); oiMidLine.setAttribute("stroke-width", "1");
      g.appendChild(oiMidLine);
      const oiLbl = document.createElementNS(NS, "text");
      oiLbl.setAttribute("x", ml - 6); oiLbl.setAttribute("y", OI_MID + 3);
      oiLbl.setAttribute("text-anchor", "end"); oiLbl.setAttribute("font-size", "9");
      oiLbl.setAttribute("fill", "var(--muted)");
      oiLbl.textContent = "OI";
      const oiLblTip = document.createElementNS(NS, "title");
      oiLblTip.textContent = "미결제약정(OI)의 5분 증분 -- 그 5분에 새로 생긴 계약 수입니다."
        + " 위(초록)는 새 포지션 유입, 아래(빨강)는 청산·정리, 회색은 수집 공백 뒤라"
        + " 직전 봉과 이을 수 없는 구간입니다. 바이낸스가 OI 를 3~7초에 한 번만 갱신하므로"
        + " 한 봉은 40~90개 스냅샷의 양 끝으로 잽니다.";
      oiLbl.appendChild(oiLblTip);
      g.appendChild(oiLbl);
      const oiPeakLbl = document.createElementNS(NS, "text");
      oiPeakLbl.setAttribute("x", ml + cw + 6); oiPeakLbl.setAttribute("y", OI_MID + 3);
      oiPeakLbl.setAttribute("font-size", "9"); oiPeakLbl.setAttribute("fill", "var(--muted)");
      oiPeakLbl.textContent = "최대 ±" + fmtFootprintQty(oiPeak);
      g.appendChild(oiPeakLbl);
    }
  }

  });

  // ── 거래대금 · 델타·CVD (2026-09-21 사용자 지시, 시안 D) ──────────────────────
  // 풋프린트 봉에서 바로 나온다 -- 새 엔드포인트도 새 폴링도 없다.
  //   거래대금 = Σ 가격 x (매수+매도) [USD]   · 선형 자
  //   델타     = Σ (매수 - 매도)      [ETH]   ┐ 한 자 · 한 0선
  //   CVD      = 창 안에서 델타의 누적 [ETH]  ┘
  // 🔴거래대금은 **선형**이다. 실측 최대/중앙 3.5배라 완만하다(수급은 11배라 log 를 썼다) --
  //   같은 화면의 두 줄이 다른 자를 쓰는 이유가 이것이고, 둘 다 오른쪽에 최대치를 적는다.
  // 🔴델타와 CVD 는 **한 자 · 한 0선**을 나눠 쓴다. 같은 단위(ETH)이고 CVD 가 델타의 누적이라,
  //   자를 따로 주면 «선이 막대들의 달리는 합»이라는 사실이 화면에서 끊긴다. 대가: 창 최대
  //   델타가 자를 지배해 작은 막대는 1~1.5px 다(실측 12봉). 1px 바닥으로 **부호는 항상** 보인다.
  cachedLayer("flowLanes", objToken(fpBars) + "|" + chartWindowBars, (g) => {
  if (fpBars.length && candles.length && TURN_H) {
    const byTs = new Map();
    fpBars.forEach((b) => {
      const t = Number(b && b.time);
      if (!Number.isFinite(t)) return;
      let buy = 0, sell = 0, turn = 0;
      (b.levels || []).forEach((l) => {
        const q = (Number(l[1]) || 0), k = (Number(l[2]) || 0);
        buy += q; sell += k; turn += (Number(l[0]) || 0) * (q + k);
      });
      byTs.set(t, { turn, delta: buy - sell });
    });
    // CVD 는 **화면에 보이는 봉 순서대로** 누적한다 -- 창 시작이 0 이다.
    let run = 0;
    const rows = candles.map((c) => {
      const v = byTs.get(c.time);
      if (!v) return null;
      run += v.delta;
      return { t: c.time, turn: v.turn, delta: v.delta, cvd: run };
    });
    const have = rows.filter(Boolean);
    const nowBar = Math.floor(Date.now() / 1000 / 300) * 300;
    const hours = Math.round(chartWindowBars * 5 / 60);
    const fmtUsd = (v) => (v >= 1e6 ? (v / 1e6).toFixed(v >= 1e7 ? 0 : 1) + "M"
                                    : Math.round(v / 1e3) + "k");
    const line = (y, x1, x2, col) => {
      const el = document.createElementNS(NS, "line");
      el.setAttribute("x1", x1); el.setAttribute("x2", x2);
      el.setAttribute("y1", y); el.setAttribute("y2", y);
      el.setAttribute("stroke", col); g.appendChild(el);
    };
    const sideLabel = (y, txt, tip) => {
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", ml - 6); t.setAttribute("y", y);
      t.setAttribute("text-anchor", "end"); t.setAttribute("font-size", "9");
      t.setAttribute("fill", "var(--muted)"); t.textContent = txt;
      if (tip) {
        const ti = document.createElementNS(NS, "title");
        ti.textContent = tip; t.appendChild(ti);
      }
      g.appendChild(t);
      return t;
    };
    const rightLabel = (y, txt) => {
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", ml + cw + 6); t.setAttribute("y", y);
      t.setAttribute("font-size", "9"); t.setAttribute("fill", "var(--muted)");
      t.textContent = txt; g.appendChild(t);
    };

    // ── 거래대금 (바닥 기준 · 선형) ──────────────────────────────────────────
    if (have.length) {
      const peak = Math.max(...have.map((r) => r.turn));
      line(turnPanelY, ml, ml + cw, "var(--soft-line)");
      if (peak > 0) {
        rows.forEach((r, i) => {
          if (!r || !r.turn) return;
          const hgt = Math.max(1, (TURN_H - 1) * r.turn / peak);
          const rect = document.createElementNS(NS, "rect");
          rect.setAttribute("x", xAt(i));
          rect.setAttribute("y", turnPanelY + TURN_H - hgt);
          rect.setAttribute("width", Math.max(1, bw));
          rect.setAttribute("height", hgt);
          rect.setAttribute("fill", "var(--neutral)");   // 방향 없는 값 -- 초록/빨강 금지
          rect.setAttribute("fill-opacity", r.t >= nowBar ? "0.38" : "0.70");
          const ti = document.createElementNS(NS, "title");
          ti.textContent = fmtDateTick(r.t * 1000) + " 거래대금 $" + fmtUsd(r.turn)
            + (r.t >= nowBar ? " (진행 중)" : "");
          rect.appendChild(ti);
          g.appendChild(rect);
        });
        rightLabel(turnPanelY + TURN_H / 2 + 3, "최대 $" + fmtUsd(peak));
      }
      sideLabel(turnPanelY + TURN_H / 2 + 3, "거래대금",
        "봉마다 그 5분에 오간 금액입니다(Σ 가격 x 체결량, 매수+매도 둘 다).\n"
        + "🔴자는 **선형**입니다 -- 실측 최대/중앙 3.5배라 로그로 누르면 오히려 차이가"
        + " 사라집니다(바로 아래 수급 줄은 11배라 로그를 씁니다).");

      // ── 델타 막대 + CVD 선 (한 자 · 한 0선) ───────────────────────────────
      const lo = Math.min(0, ...have.map((r) => Math.min(r.delta, r.cvd)));
      const hi = Math.max(0, ...have.map((r) => Math.max(r.delta, r.cvd)));
      const rng = Math.max(hi - lo, 1e-9);
      const yAtV = (v) => dcvdPanelY + (hi - v) / rng * DCVD_H;
      const y0 = yAtV(0);
      rows.forEach((r, i) => {
        if (!r || !r.delta) return;
        const hgt = Math.max(1, Math.abs(yAtV(r.delta) - y0));
        const rect = document.createElementNS(NS, "rect");
        rect.setAttribute("x", xAt(i));
        rect.setAttribute("y", r.delta >= 0 ? y0 - hgt : y0);
        rect.setAttribute("width", Math.max(1, bw));
        rect.setAttribute("height", hgt);
        rect.setAttribute("fill", r.delta >= 0 ? "var(--good)" : "var(--bad)");
        rect.setAttribute("fill-opacity", r.t >= nowBar ? "0.30" : "0.55");
        const ti = document.createElementNS(NS, "title");
        ti.textContent = fmtDateTick(r.t * 1000) + " 델타 "
          + (r.delta >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(r.delta)) + " ETH"
          + " · 여기까지 CVD " + (r.cvd >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(r.cvd))
          + (r.t >= nowBar ? " (진행 중)" : "");
        rect.appendChild(ti);
        g.appendChild(rect);
      });
      line(y0, ml, ml + cw, "var(--line)");
      const pts = rows.map((r, i) => (r ? (xAt(i) + bw / 2) + "," + yAtV(r.cvd) : null))
                      .filter(Boolean).join(" ");
      if (pts) {
        const pl = document.createElementNS(NS, "polyline");
        pl.setAttribute("points", pts); pl.setAttribute("fill", "none");
        pl.setAttribute("stroke", "var(--text)"); pl.setAttribute("stroke-width", "2");
        pl.setAttribute("stroke-linejoin", "round");
        g.appendChild(pl);
        const last = have[have.length - 1];
        const dot = document.createElementNS(NS, "circle");
        dot.setAttribute("cx", xAt(rows.lastIndexOf(last)) + bw / 2);
        dot.setAttribute("cy", yAtV(last.cvd)); dot.setAttribute("r", "2.6");
        dot.setAttribute("fill", "var(--text)");
        g.appendChild(dot);
        rightLabel(dcvdPanelY + DCVD_H / 2 + 3,
                   "CVD " + (last.cvd >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(last.cvd)));
      }
      sideLabel(dcvdPanelY + DCVD_H / 2 + 3, "델타·CVD",
        "막대 = 봉마다의 순델타(매수-매도, ETH) · 흰 선 = 그 막대들의 **달리는 합**(CVD).\n"
        + "🔴둘은 한 자 · 한 0선을 나눠 씁니다. 자를 따로 주면 선이 막대의 합이라는 사실이"
        + " 화면에서 끊깁니다. 대가로 창 최대 델타가 자를 지배해 작은 막대는 1~1.5px 입니다"
        + " -- 부호는 항상 보이고 크기는 커서에 나옵니다.\n"
        + "🔴**CVD 의 0 은 이 창의 시작**입니다(지금 " + hours + "시간). 창을 1h/2h/4h 로"
        + " 바꾸면 기준점도 같이 옮겨갑니다 -- 절대 누적이 아닙니다.\n"
        + "🔴«상황 읽기» 카드의 CVD 는 **30분 고정창**이라 이 값과 다릅니다. 이름이 같아도"
        + " 같은 값이 아닙니다.");
    }
  }
  });

  // ── 고래·리테일 수급 리본 (2026-09-20 사용자 지시) ─────────────────────────────
  // 봉마다 순수급(매수-매도)을 **고래(≥$100k)**와 **리테일(<$10k)** 둘로 갈라 0선 기준
  // 좌/우 반폭 막대로 그린다. 값은 supplyFlowOfBar() 가 낸다 -- 5분봉 하나가 곧 5분 누적이다.
  // 🔴같은 자를 쓴다. 실측 |고래|/|리테일| 중앙 4.7배라 리테일이 작게 나오지만, 자를 따로
  //   주면 「고래 −319 · 리테일 +53」이 같은 크기로 보인다. 이 리본이 말하려는 게 바로 그
  //   엇갈림이라(실측 12봉 중 4봉) 크기를 거짓말하면 화면이 뒤집힌다. 대신 최소 1px 을
  //   보장해 **부호는 항상 보이게** 한다.
  // 풋프린트 폴링(2초)에만 달려 있다.
  cachedLayer("supplyLane", objToken(fpBars), (g) => {
  if (fpBars.length && candles.length && SUP_PANEL_H) {
    const SY = supPanelY, SH = SUP_PANEL_H, SMID = SY + SH / 2;
    const flowByTs = new Map();
    fpBars.forEach((b) => {
      const t = Number(b && b.time);
      if (Number.isFinite(t)) flowByTs.set(t, supplyFlowOfBar(b.levels));
    });
    let peak = 0;
    candles.forEach((c) => {
      const f = flowByTs.get(c.time);
      if (f) peak = Math.max(peak, Math.abs(f.whale), Math.abs(f.retail));
    });
    const topLine = document.createElementNS(NS, "line");
    topLine.setAttribute("x1", ml); topLine.setAttribute("x2", ml + cw);
    topLine.setAttribute("y1", SY); topLine.setAttribute("y2", SY);
    topLine.setAttribute("stroke", "var(--soft-line)");
    g.appendChild(topLine);
    if (peak > 0) {
      const half = SH / 2 - 1;
      const nowBar = Math.floor(Date.now() / 1000 / 300) * 300;
      const series = [{ k: "whale", name: "고래", op: 0.9 },
                      { k: "retail", name: "리테일", op: 0.55 }];
      candles.forEach((c, i) => {
        const f = flowByTs.get(c.time);
        if (!f) return;
        // 반폭 둘. 봉이 좁으면(모바일 34봉 bw≈6.8) 각 1px 까지 줄지만 자리는 유지된다.
        const bw2 = Math.max(1, bw / 2 - 0.5);
        series.forEach((sr, si) => {
          const v = f[sr.k];
          if (!v) return;
          // 🔴선형은 못 쓴다 -- 실측 12봉에서 최대 2,964 / 중앙 202 라 **중앙 봉이 0.4px**
          //   가 된다(한 봉이 나머지를 씻어낸다). 청산 레인과 같은 log1p 자를 쓴다.
          const hgt = Math.max(1, half * Math.log1p(Math.abs(v)) / Math.log1p(peak));
          const rect = document.createElementNS(NS, "rect");
          rect.setAttribute("x", xAt(i) + si * (bw / 2));
          rect.setAttribute("y", v >= 0 ? SMID - hgt : SMID);
          rect.setAttribute("width", bw2);
          rect.setAttribute("height", hgt);
          rect.setAttribute("fill", v >= 0 ? "var(--good)" : "var(--bad)");
          rect.setAttribute("fill-opacity",
                            String(c.time >= nowBar ? sr.op * 0.5 : sr.op));
          const ti = document.createElementNS(NS, "title");
          ti.textContent = fmtDateTick(c.time * 1000) + " " + sr.name + " 순수급 "
            + (v >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(v)) + " ETH"
            + " · 고래 " + (f.whale >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(f.whale))
            + " · 중형 " + (f.mid >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(f.mid))
            + " · 리테일 " + (f.retail >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(f.retail))
            + (c.time >= nowBar ? " (진행 중)" : "");
          rect.appendChild(ti);
          g.appendChild(rect);
        });
      });
      const midLine = document.createElementNS(NS, "line");
      midLine.setAttribute("x1", ml); midLine.setAttribute("x2", ml + cw);
      midLine.setAttribute("y1", SMID); midLine.setAttribute("y2", SMID);
      midLine.setAttribute("stroke", "var(--line)"); midLine.setAttribute("stroke-width", "1");
      g.appendChild(midLine);
      const lbl = document.createElementNS(NS, "text");
      lbl.setAttribute("x", ml - 6); lbl.setAttribute("y", SMID + 3);
      lbl.setAttribute("text-anchor", "end"); lbl.setAttribute("font-size", "9");
      lbl.setAttribute("fill", "var(--muted)");
      lbl.textContent = "수급";
      const lblTip = document.createElementNS(NS, "title");
      lblTip.textContent = "봉마다 그 5분의 순수급(매수-매도)입니다. 왼쪽 반이 고래(≥$100k),"
        + " 오른쪽 반이 리테일(<$10k) -- 옅은 쪽이 리테일입니다. 위(초록)가 순매수입니다.\n"
        + "이 줄이 보여주려는 건 **둘의 엇갈림**입니다 -- 실측 12봉 중 4봉에서 고래와 리테일의"
        + " 부호가 반대였습니다(예: 고래 -319 · 리테일 +53).\n"
        + "🔴높이는 log 자입니다(청산 레인과 같은 규약). 선형으로 그리면 한 봉(실측 2,964)이"
        + " 나머지를 씻어내 중앙 봉이 0.4px 가 됩니다. 대가로 **크기 비는 눌립니다** --"
        + " 고래가 리테일의 4.7배(중앙)여도 막대는 1.5배쯤으로 보입니다. 정확한 값은 막대에"
        + " 커서를 올리면 셋(고래·중형·리테일) 다 나옵니다.\n"
        + "🔴«고래»의 단위는 개별 체결이 아니라 테이커 주문(aggTrade)입니다. 쓸어담기 한 건이"
        + " 작은 체결 수십 건으로 쪼개지므로 체결 단위로 세면 고래가 4배 작게 나옵니다.";
      lbl.appendChild(lblTip);
      g.appendChild(lbl);
      const peakLbl = document.createElementNS(NS, "text");
      peakLbl.setAttribute("x", ml + cw + 6); peakLbl.setAttribute("y", SMID + 3);
      peakLbl.setAttribute("font-size", "9"); peakLbl.setAttribute("fill", "var(--muted)");
      peakLbl.textContent = "최대 ±" + fmtFootprintQty(peak);
      g.appendChild(peakLbl);
    }
  }
  });

  // ── 봉별 청산 레인 (2026-09-11 사용자 "청산맵 차트에 매 5분봉 청산 데이터를 추가") ──
  // 데이터: /api/liquidation-5m-history -- tail_risk_1m 의 실제 @forceOrder 체결을 5분으로 접은 것.
  // 🔴게이지(/api/liquidation-5m-signal)는 BAR_MINUTES=30 이다. 여기는 캔들과 같은 **5분**이라야
  //   봉이 안 어긋난다 -- compute_liquidation_5m_history() 가 CHART_BAR_MINUTES=5 로 따로 접는다.
  // 2026-09-16 사용자 요청으로 **차트 안 하단 패널**이 됐다(그전엔 하단 여백에 떠 있었다).
  //   여백의 레인은 차트 밖 부속처럼 읽혔는데, 청산은 같은 봉의 가격 움직임과 **같이** 봐야 하는
  //   값이라 한 프레임 안에 두는 게 맞다. 09-11 "증거신호 레인을 청산맵 밖으로"와 어긋나 보이지만
  //   대상이 다르다 -- 그때 밖으로 뺀 건 캔들을 **덮던** 오버레이였고, 이건 자기 자리를 가진
  //   서브플롯이다(가격 플롯은 그만큼 줄어들 뿐 가려지지 않는다).
  // 색은 표시 규약 그대로 롱=good / 숏=bad. 위로 롱청산, 아래로 숏청산인 발산형.
  // ⚠️로그 스케일이다. 최근 7일 5분봉 중앙 $211 / 최대 $4.9M 로 23,000배라 선형이면 거의 전부가
  //   1픽셀 미만으로 사라진다.
  // 청산 레인은 60초 폴링에만 달려 있다(봉당 최대 rect+title 두 벌).
  cachedLayer("liqLane", objToken(liqBars), (g) => {
  if (Array.isArray(liqBars) && liqBars.length && candles.length) {
    const LIQ_Y = liqPanelY, LIQ_H = LIQ_PANEL_H, LIQ_MID = LIQ_Y + LIQ_H / 2;
    // 🔴캔들의 `time` 은 **초** 단위다(server.py: int(row["timestamp"].timestamp())).
    //   Date.parse 는 밀리초라 그대로 키로 쓰면 절대 안 맞는다 -- 2026-09-11 에 이걸로
    //   레인이 통째로 안 그려졌다. 차트의 다른 코드가 전부 `c.time * 1000` 을 쓰는 이유다.
    const liqByTs = new Map();
    liqBars.forEach((b) => {
      const t = Date.parse(b.ts);
      if (Number.isFinite(t)) liqByTs.set(Math.floor(t / 1000), b);
    });
    let liqPeak = 0;
    candles.forEach((c) => {
      const b = liqByTs.get(c.time);
      if (b) liqPeak = Math.max(liqPeak, Number(b.long_usd) || 0, Number(b.short_usd) || 0);
    });
    const liqTopLine = document.createElementNS(NS, "line");
    liqTopLine.setAttribute("x1", ml); liqTopLine.setAttribute("x2", ml + cw);
    liqTopLine.setAttribute("y1", LIQ_Y); liqTopLine.setAttribute("y2", LIQ_Y);
    liqTopLine.setAttribute("stroke", "var(--soft-line)");
    g.appendChild(liqTopLine);
    if (liqPeak > 0) {
      const liqHalf = LIQ_H / 2 - 1;
      const liqScale = (v) => (v > 0 ? Math.max(1, liqHalf * Math.log1p(v) / Math.log1p(liqPeak)) : 0);
      candles.forEach((c, i) => {
        const b = liqByTs.get(c.time);
        if (!b) return;
        const lu = Number(b.long_usd) || 0, su = Number(b.short_usd) || 0;
        if (lu <= 0 && su <= 0) return;
        [["long", lu, "var(--good)"], ["short", su, "var(--bad)"]].forEach((spec) => {
          const kind = spec[0], v = spec[1], color = spec[2];
          if (v <= 0) return;
          const hgt = liqScale(v);
          const rect = document.createElementNS(NS, "rect");
          rect.setAttribute("x", xAt(i));
          rect.setAttribute("y", kind === "long" ? LIQ_MID - hgt : LIQ_MID);
          rect.setAttribute("width", Math.max(1, bw));
          rect.setAttribute("height", hgt);
          rect.setAttribute("fill", color);
          rect.setAttribute("fill-opacity", b.partial ? "0.42" : "0.85");
          const title = document.createElementNS(NS, "title");
          title.textContent = fmtDateTick(c.time * 1000) + " 롱청산 " + fmtUsdCompact(lu)
            + " · 숏청산 " + fmtUsdCompact(su) + " · " + b.events + "건"
            + (b.partial ? " (진행 중)" : "");
          rect.appendChild(title);
          g.appendChild(rect);
        });
      });
      const liqMidLine = document.createElementNS(NS, "line");
      liqMidLine.setAttribute("x1", ml); liqMidLine.setAttribute("x2", ml + cw);
      liqMidLine.setAttribute("y1", LIQ_MID); liqMidLine.setAttribute("y2", LIQ_MID);
      liqMidLine.setAttribute("stroke", "var(--line)"); liqMidLine.setAttribute("stroke-width", "1");
      g.appendChild(liqMidLine);
      const liqLbl = document.createElementNS(NS, "text");
      liqLbl.setAttribute("x", ml - 6); liqLbl.setAttribute("y", LIQ_MID + 3);
      liqLbl.setAttribute("text-anchor", "end"); liqLbl.setAttribute("font-size", "9");
      liqLbl.setAttribute("fill", "var(--muted)");
      liqLbl.textContent = "청산";
      g.appendChild(liqLbl);
      const liqPeakLbl = document.createElementNS(NS, "text");
      liqPeakLbl.setAttribute("x", ml + cw + 6); liqPeakLbl.setAttribute("y", LIQ_MID + 3);
      liqPeakLbl.setAttribute("font-size", "9"); liqPeakLbl.setAttribute("fill", "var(--muted)");
      liqPeakLbl.textContent = "최대 " + fmtUsdCompact(liqPeak);
      g.appendChild(liqPeakLbl);
    }
  }

  });

  // ── SVG 안의 수급 두 패널 -- OI 레인 바로 위 (2026-09-19 사용자 지시) ──────────
  // 중첩 <svg> 를 쓴다 -- 자식 svg 는 제 viewBox 를 갖는 독립 좌표계라 두 렌더러의 좌표
  // 계산을 한 줄도 안 고쳐도 된다. 상자만 넘기면 그 안에서 평소처럼 그린다.
  if (SUB_TOTAL > 0) {
    // 상자는 매번 다시 앉히되 **내용은 판번호가 바뀌었을 때만** 다시 그린다(subPanelCache
    // 주석의 실측 참고). 노드를 재사용하므로 mousemove 리스너도 한 번만 붙는다.
    const subSvg = (slot, x, y, wid, hgt, key, draw) => {
      const slotState = subPanelCache[slot];
      let g = slotState.node;
      const hit = !!g && slotState.key === key;
      if (!g) {
        g = slotState.node = document.createElementNS(NS, "svg");
        // 캔들 툴팁이 이 위에서도 뜨면 «이 봉»이 아닌 값을 말한다 -- 버블링을 여기서 끊는다.
        g.addEventListener("mousemove", (e) => e.stopPropagation());
      }
      g.setAttribute("x", x); g.setAttribute("y", y);
      g.setAttribute("width", wid); g.setAttribute("height", hgt);
      svg.appendChild(g);
      if (!hit) { draw(g); slotState.key = key; }
      return g;
    };
    // 히트맵이 왼쪽, 프로파일이 오른쪽(사용자 지시). 히트맵의 오른쪽 끝 = 지금 호가라
    // 그 옆에 프로파일 가격행이 바로 이어진다 -- 두 그림이 같은 가격축에서 만난다.
    // 🔴프로파일을 **먼저** 그린다 -- supplyProfileNow 에 행 기하를 남겨야 히트맵이
    //   같은 y 에 포갠다(x 위치는 호출 순서와 무관하다. 각자 제 <svg> 상자를 받는다).
    supplyProfileSubBox = {
      svg: subSvg("prof", 0, subProfileY, w, SUB_PROFILE_H,
                  subProfileKey(entryPrice, w, SUB_PROFILE_H),
                  (g) => renderSupplyProfileSvg(g, latestSupplyProfile, currentPrice,
                                                entryPrice, { w, h: SUB_PROFILE_H })),
      w, h: SUB_PROFILE_H };
    // 재사용했으면 renderSupplyProfileSvg 가 안 돌았으므로 현재가 줄을 여기서 맞춘다.
    updateSupplyProfileNow(currentPrice);
    // 2026-09-19 히트맵도 같은 캐시를 쓴다 -- 래스터는 3초마다 새 열이 오는데 캔들 전체
    // 리렌더(가격 틱)를 기다릴 이유가 없다(2bb2b2f1 이 프로파일/1초수급에 넣은 그 이유).
    supply1sSubBox = {
      svg: subSvg("s1", 0, sub1sY, w, SUB_1S_H, sub1sKey(w, SUB_1S_H),
                  (g) => renderSupply1s({ svg: g, w, h: SUB_1S_H })),
      w, h: SUB_1S_H };
    // ── 청산 밀도 범례 (2026-09-21 아티팩트 댓글) ────────────────────────────
    // 전에는 헤더 행의 HTML(#liqDensityLegend)이었다. 밀도가 풋프린트의 **배경**이 된
    // 뒤로는 설명하는 그림에서 멀어졌으므로 프로파일 바닥글과 풋프린트 사이로 내렸다.
    // 글자 9 = 바닥글과 같은 크기(사용자 지시).
    // 🔴그라디언트는 densityStops() 에서 바로 만든다 -- 사본을 두면 히트맵과 어긋나도
    //   아무도 모른다(2026-09-12 에 실제로 겪었다).
    if (SUB_LEGEND_H && (densityHistory || []).length) {
      const lg = subSvg("dens", 0, subLegendY, w, SUB_LEGEND_H,
                        "dens|" + w + "|" + SUB_LEGEND_H, (g) => {
        const stops = densityStops();
        const defs = document.createElementNS(NS, "defs");
        const grad = document.createElementNS(NS, "linearGradient");
        grad.setAttribute("id", "liqDensLegendGrad");
        stops.forEach(([t, c]) => {
          const st = document.createElementNS(NS, "stop");
          st.setAttribute("offset", (t * 100) + "%");
          st.setAttribute("stop-color", `rgb(${c[0]},${c[1]},${c[2]})`);
          grad.appendChild(st);
        });
        defs.appendChild(grad); g.appendChild(defs);
        const yMid = SUB_LEGEND_H / 2;
        let x = ml;
        const txt = (tx, str) => {
          const t = document.createElementNS(NS, "text");
          t.setAttribute("x", tx); t.setAttribute("y", yMid + 3);
          t.setAttribute("font-size", "9"); t.setAttribute("fill", "var(--muted)");
          t.textContent = str; g.appendChild(t);
          return str.length * 6.2;
        };
        x += txt(x, "청산 밀도") + 10;
        x += txt(x, "낮음") + 4;
        const bar = document.createElementNS(NS, "rect");
        bar.setAttribute("x", x); bar.setAttribute("y", yMid - 3);
        bar.setAttribute("width", 160); bar.setAttribute("height", 6);
        bar.setAttribute("rx", "1");
        bar.setAttribute("fill", "url(#liqDensLegendGrad)");
        g.appendChild(bar);
        x += 164;
        txt(x, "높음");
        const tip = document.createElementNS(NS, "title");
        tip.textContent = "아래 풋프린트 차트의 **배경 띠** 색입니다 -- 그 가격대에 쌓인"
          + " 청산 예상 물량이 많을수록 진합니다. 색은 히트맵과 같은 densityStops() 에서"
          + " 바로 만들어 둘이 어긋날 수 없습니다.";
        g.appendChild(tip);
      });
      lg.style.pointerEvents = "none";   // 아래 캔들 툴팁을 가리지 않는다
    } else {
      subPanelCache.dens.key = "";
      if (subPanelCache.dens.node && subPanelCache.dens.node.parentNode) {
        subPanelCache.dens.node.parentNode.removeChild(subPanelCache.dens.node);
      }
    }
  } else {
    // 다른 코인·웜업이면 자리를 안 잡는다. 캐시 키를 비워 두지 않으면 ETH 로 돌아왔을 때
    // 낡은 그림이 «맞는 판»으로 다시 붙는다.
    supplyProfileSubBox = supply1sSubBox = null;
    subPanelCache.prof.key = subPanelCache.s1.key = subPanelCache.dens.key = "";
  }
}

function render(state, compactState = null, { stateChanged = true } = {}) {
  latestMainState = state;
  if (!stateChanged) return;

  const sess = state.session || {};
  const micro = state.microstructure || {}, tail = state.tail_risk || {};

  const sessionHtml = buildSessionHtml(sess);
  setH("topSession", sessionHtml);
  setT("topClock", fmtNowClock());
  
  // 2026-08-25: perf pass -- this whole block (gauge + chart + model-indicator list) only paints
  // anything the user can see while the Snapshot tab is active (snapshotTabPanel is display:none
  // otherwise), so it's gated the same way as tick()'s Snapshot-only fetches above. Data
  // accumulation (pushToneHistory calls above this block, liqDirTone derivation)
  // stays unconditional -- only the paint work below is skipped, so history strips have no gap when
  // the user switches back to Snapshot.
  if (activePageTab === "snapshot") {
    renderLiquidationVolumeGauge();

    // Bug found 2026-08-25: renderSnapshotChart() (candles + S/R line + the old liquidationMagnetLevel(),
    // removed 2026-08-31) used to be called ONLY from the two data-fetch functions that feed it, each
    // gated to a 5-minute interval (maybeFetchSnapshotChartHistory/refreshLiquidationMap) -- a
    // reasonable cadence for candles/the liquidation map, since neither source changes faster than
    // that. But the current-price line reads latestLivePriceByAsset, which updates on every SSE tick
    // (this render() call itself) -- so it could sit stale for up to 5 minutes after a real change, or
    // simply never have painted yet if the chart's first 5-min-gated render happened before a live
    // price had arrived. Throttled to SNAPSHOT_CHART_RENDER_MIN_INTERVAL_MS, same pattern the Live
    // tab's own chart uses for its own frequent-tick redraws (own constant since 2026-08-25 -- see its
    // definition for why Snapshot can afford a coarser interval) -- cheap since renderSnapshotChart()
    // only redraws from already-cached data, no network fetch of its own.
    const nowForSnapshotChart = Date.now();
    if (nowForSnapshotChart - lastSnapshotChartRenderAt >= chartRenderGateMs()) {
      lastSnapshotChartRenderAt = nowForSnapshotChart;
      updateSnapshotCandleLive();
      renderSnapshotChart();
      /* (아래 목록 갱신은 이 경로에만 있다 -- 시세 푸시용 경로는 maybeRenderSnapshotChartNow) */
      // 2026-08-27: same bug/fix as renderSnapshotChart() above, one component down -- the
      // liq-level-list panel (renderLiquidationMapPanel()) has its own live-price re-filter
      // (liveRedistanced()) that's supposed to drop an already-crossed level immediately, but the
      // function itself was only ever called from refreshLiquidationMap()'s 5-minute-gated fetch,
      // so the filter never got to re-run against a fresher price in between. User report: a
      // broken resistance-1 disappeared from the chart right away but stayed in this list for
      // several minutes. Piggybacking on the same throttle as the chart -- cheap, no fetch, and
      // both now redraw from the same latestLiquidationMap + latestLivePriceByAsset[activeSnapshotAsset] snapshot.
      renderLiquidationMapPanel();
    }

    // 특화 감지기 (2026-08-30 user request): event-triggered, model-driven detectors that don't fit
    // either the always-on model-indicator gauges below or the scorecard-gated evidence-signal tier
    // above -- V자 급등락(2026-08-31, "V자 반등락"에서 개명)이 첫 입주(TabPFN, fires only on a
    // liquidity_sweep, long idle "대기" gaps between events), more will land here over time. Reuses
    // renderModelIndicatorList's row/strip markup verbatim (2nd param = its own target list, own
    // memoized-html slot) rather than a new template -- same reasoning as the model-indicator/
    // evidence-signal panels already sharing one markup. Append new specialized-detector objects to
    // this array as they're built.
    // 2026-08-31: liveText's old "급등 확률(TabPFN) 76%" sentence (shown under the title) dropped in
    // favor of `proba` -- renderModelIndicatorList now shows that as the same inline meter bar the
    // evidence-signal list uses (user: "인라인 미터로 바꿔줘"), in the meta column next to the state,
    // instead of duplicating the same number as a plain sentence.
    renderModelIndicatorList([
      ethOnlyIndicator(breakoutPrewarnIndicatorItem()),   // 2026-09-11 추세 전환 경보기
      ethOnlyIndicator(breakoutDetectorIndicatorItem()),  // 2026-09-11 추세 전환 탐지기
    ], "snapSpecializedSignalList", { forceMeter: true });

    // Snapshot tab: renderModelIndicatorList mirrors renderEvidenceSignals's row/strip UI.
    renderModelIndicatorList([
      volLevelIndicatorItem(),                // 2026-09-21 배수 + 확장 확률
      gexIndicatorItem(),                     // 2026-09-19 옵션 감마 노출 (09-28 판정일까지 한시)
    ]);
  }
}

async function tick() {
  if (document.hidden || tickInFlight) return;
  tickInFlight = true;
  try {
    // 2026-08-25 perf pass (Snapshot's 6 fetches), extended 2026-08-31 to Ops's own status poll
    // now that the Live tab (previously the 3rd, always-unconditional, tab) is gone -- each branch
    // only matters while that tab is actually visible; gating stops background fetch/compute work
    // for a hidden panel (see activePageTab, set by setupPageTabs()'s click handler). Both tabs'
    // click handlers already force an immediate refresh on switching to them, so this doesn't delay
    // first paint after a tab switch -- it only stops the ongoing poll while elsewhere.
    if (activePageTab === "ops") {
      refreshOpsStatus();
    } else if (activePageTab === "snapshot") {
      refreshBreakoutDetector();     // 2026-09-11 횡보→추세 전환
      refreshVolLevel();             // 2026-09-14 사이징 모델 변동성 예측(4시간 수준)
      refreshBinanceAccount();       // 2026-09-10 청산맵 위 계좌 요약 + 진입선 (자체 30초 게이트)
      refreshChartMarkers();         // 2026-09-09 청산맵 신호 마커
      refreshLiquidation5mSignal();
      refreshLiqBurstState();
      refreshLiquidationMap();
      refreshActiveRegime();
      refreshCoinIndicators();
      refreshMacroCalendar();
      refreshSessionAlerts();
      refreshFootprint();            // 2026-09-15 볼륨 풋프린트 체결 테이프
      refreshSupplyProfile();        // 2026-09-19 가격축 수급 프로파일
      refreshFlowHeatmap();          // 2026-09-19 호가 히트맵(프로파일 왼쪽 절반)
      refreshGex();                  // 2026-09-19 옵션 감마 노출(참고 표시 · 신호 아님)
      refreshSupply1s();             // 2026-09-19 최근 5분 x 1초 수급
      refreshOi5m();                 // 2026-09-19 OI 신규계약 5분 누적 (자체 15초 게이트)
      refreshSituation();            // 2026-09-21 상황 읽기 · 30분 (5초, ETH 만)
      ensurePriceWs();               // 2026-09-16 현재가 직결 WS (탭/코인/가시성 변화가 여기로 수렴)
      maybeFetchSnapshotChartHistory();
    }
  } catch (e) {
    console.error("Tick Error:", e);
  } finally {
    tickInFlight = false;
  }
}

// One-time seed of the Snapshot tab's model-indicator strips from the dashboard server's own
// history buffer (populated server-side every 5 min regardless of whether any browser tab is
// open -- see /api/model-indicator-history in dashboard/server.py). Awaited BEFORE the live SSE
// connection starts so no live tick can race ahead and populate toneHistory first: if that raced,
// the "already has data" guard below would (correctly, but uselessly) skip seeding, leaving the
// strip looking exactly as un-warmed-up as before this feature existed.

(async () => {
  connectDashboardEvents();
  tick();
  setInterval(tick, POLL_MS);
})();
setInterval(() => {
  if (!isScrolling()) { setT("topClock", fmtNowClock()); }
}, 1000);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    disconnectDashboardEvents();
    ensurePriceWs();   // 숨으면 닫는다 -- 백그라운드 탭이 초당 수백 메시지를 받을 이유가 없다
    return;
  }
  connectDashboardEvents();
  tick();
});
setupSnapshotAssetTabs();
setupThemeToggle();
setupChartModeTabs();
setupPageTabs();
setupScrollRendering();

function showTooltip(x, y, html) {
  const t = el("chartTooltip");
  if (!t) return;
  t.innerHTML = html;
  t.classList.add("visible");
  
  const w = window.innerWidth;
  const tWidth = t.offsetWidth || 150;
  // Use a smaller offset (8px) and check right boundary
  let left = x + 8;
  if (left + tWidth > w) left = x - tWidth - 8; 
  
  t.style.left = left + "px";
  t.style.top = (y + 15) + "px"; // Position slightly below cursor
}

function hideTooltip() {
  const t = el("chartTooltip");
  if (t) t.classList.remove("visible");
}


// =======================================================================================
// 알림 탭 (2026-09-04). 사용자 요청으로 토프바 토글에서 전용 페이지로 옮겼다.
//
// 이 페이지가 단순한 설정 화면이 아닌 이유: 웹푸시는 **조용히** 실패한다. 푸시 서비스가
// 201을 돌려줘도 브라우저가 안 띄우면 사용자에게는 그냥 "알림이 안 온다"로 보이고, 서버
// 로그만으로는 구분되지 않는다(실제로 이 기능 첫 배포 때 그 상태가 됐다 -- FCM은 201,
// 구독도 등록, 그런데 화면에는 아무것도 안 떴다). 그래서 테스트를 두 구간으로 쪼갠다:
//   (1) 로컬 showNotification -- 푸시 경로를 안 거치고 표시만 검사
//   (2) 서버 발송            -- 실제 경로 전체
// (1)이 되고 (2)가 안 되면 전달 문제, (1)부터 안 되면 이 기기가 알림을 막고 있는 것이다.
// =======================================================================================
const PUSH_SW_URL = "/dashboard/live/sw.js";
const notifyState = { config: null, registration: null, subscription: null, devices: [], installPrompt: null };

function urlBase64ToUint8Array(base64String) {
  // applicationServerKey는 Uint8Array만 받는다. 서버가 주는 건 unpadded base64url이라
  // 패딩을 되살리고 URL-safe 문자를 표준 base64로 되돌린 뒤 디코드한다.
  const padding = "=".repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, "+").replace(/_/g, "/");
  const raw = atob(base64);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) out[i] = raw.charCodeAt(i);
  return out;
}

function pushSupported() {
  // `in` 대신 값 자체를 본다 -- 안드로이드 WebView와 파이어폭스 사생활 보호 모드는 키는
  // 노출하되 값이 undefined다. `"serviceWorker" in navigator`는 그때도 true라 통과해버린다.
  return !!(navigator.serviceWorker && window.PushManager && window.isSecureContext);
}

function notifyCheckRow(ok, name, detail) {
  const tone = ok === true ? "good" : ok === false ? "bad" : "warn";
  const mark = ok === true ? "정상" : ok === false ? "문제" : "확인";
  return `<div class="notify-check ${tone}">
    <span class="notify-check-dot"></span>
    <span class="notify-check-name">${name}</span>
    <span class="notify-check-detail">${detail}</span>
    <span class="notify-check-mark">${mark}</span>
  </div>`;
}

function renderNotifyChecks() {
  const box = el("notifyChecks");
  if (!box) return;
  const cfg = notifyState.config;
  const reg = notifyState.registration;
  const sub = notifyState.subscription;
  const rows = [];

  rows.push(notifyCheckRow(!!(navigator.serviceWorker && window.PushManager), "브라우저 지원",
    navigator.serviceWorker && window.PushManager ? "서비스워커·푸시 사용 가능"
      : "이 브라우저는 웹푸시를 지원하지 않습니다. iOS는 홈 화면에 추가해야 동작합니다."));

  rows.push(notifyCheckRow(!!window.isSecureContext, "보안 연결",
    window.isSecureContext ? location.protocol.replace(":", "").toUpperCase() + " 연결"
      : "HTTPS가 아니면 브라우저가 푸시를 막습니다."));

  const perm = typeof Notification === "undefined" ? "unsupported" : Notification.permission;
  rows.push(notifyCheckRow(perm === "granted" ? true : perm === "denied" ? false : null, "알림 권한",
    perm === "granted" ? "허용됨"
      : perm === "denied" ? "차단됨 — 주소창 자물쇠 아이콘에서 이 사이트의 알림을 허용으로 바꿔주세요."
      : "아직 요청 전입니다."));

  const swState = reg ? (reg.active ? "active" : reg.installing ? "installing" : reg.waiting ? "waiting" : "none") : "none";
  rows.push(notifyCheckRow(swState === "active" ? true : swState === "none" ? false : null, "서비스워커",
    swState === "active" ? "활성 — 알림을 받을 준비가 됐습니다"
      : swState === "none" ? "등록되지 않았습니다."
      : `${swState} 상태입니다. 잠시 뒤 다시 확인해주세요.`));

  rows.push(notifyCheckRow(cfg ? !!cfg.enabled : null, "서버 설정",
    cfg == null ? "서버 상태를 읽지 못했습니다."
      : cfg.enabled ? "발송 키가 설정돼 있습니다"
      : "서버에 VAPID 키가 없습니다. 관리자 설정이 필요합니다."));

  rows.push(notifyCheckRow(!!sub, "이 기기 구독",
    sub ? "구독됨" : "아직 구독하지 않았습니다. 위의 켜기 버튼을 눌러주세요."));

  // 브라우저에는 구독이 있는데 서버 목록에 없으면 발송 대상이 아니다 -- 사용자 눈에는
  // "켜져 있는데 안 온다"로 보이는 상태라 반드시 따로 짚어준다.
  if (sub) {
    const tail = sub.endpoint.slice(-12);
    const known = notifyState.devices.some((d) => d.endpoint_tail === tail);
    rows.push(notifyCheckRow(known, "서버 등록",
      known ? "서버가 이 기기를 알고 있습니다"
        : "브라우저에는 구독이 있는데 서버 목록에 없습니다. 껐다 다시 켜주세요."));
  }

  box.innerHTML = rows.join("");
}

function renderNotifyMain() {
  const cfg = notifyState.config;
  const sub = notifyState.subscription;
  const btn = el("notifyToggleBtn");
  const badge = el("notifyBadge");
  const title = el("notifyStateTitle");
  const detail = el("notifyStateDetail");
  const perm = typeof Notification === "undefined" ? "unsupported" : Notification.permission;

  let state, headline, sub_text, label, disabled = false;
  if (!pushSupported()) {
    state = "bad"; headline = "이 브라우저에서는 알림을 쓸 수 없습니다";
    sub_text = "아래 진단에서 막힌 항목을 확인해주세요."; label = "사용 불가"; disabled = true;
  } else if (cfg && !cfg.enabled) {
    state = "bad"; headline = "서버가 알림을 보낼 수 없는 상태입니다";
    sub_text = "서버에 발송 키가 설정되지 않았습니다."; label = "사용 불가"; disabled = true;
  } else if (perm === "denied") {
    state = "bad"; headline = "브라우저가 이 사이트의 알림을 차단했습니다";
    sub_text = "주소창 자물쇠 아이콘 → 알림 → 허용으로 바꾸면 다시 켤 수 있습니다.";
    label = "차단됨"; disabled = true;
  } else if (sub) {
    state = "good"; headline = "알림이 켜져 있습니다";
    sub_text = "포지션 개시·청산과 운영 이상은 소리와 함께 즉시 옵니다."; label = "알림 끄기";
  } else {
    state = "neutral"; headline = "알림이 꺼져 있습니다";
    sub_text = "켜면 이 기기로 신호가 도착합니다. 창을 닫아도 옵니다."; label = "알림 켜기";
  }

  if (title) title.textContent = headline;
  if (detail) detail.textContent = sub_text;
  if (btn) { btn.textContent = label; btn.disabled = disabled; btn.classList.toggle("primary", !sub); }
  if (badge) {
    badge.textContent = state === "good" ? "알림 켬" : state === "bad" ? "알림 불가" : "알림 끔";
    badge.className = `ops-badge ${state}`;
  }
  const canTest = !!(notifyState.registration && perm === "granted");
  const localBtn = el("notifyLocalTestBtn");
  const serverBtn = el("notifyServerTestBtn");
  if (localBtn) localBtn.disabled = !canTest;
  if (serverBtn) serverBtn.disabled = !canTest || !sub;
}

function renderNotifyDevices() {
  const box = el("notifyDevices");
  if (!box) return;
  const list = notifyState.devices;
  if (!list.length) {
    box.innerHTML = `<span class="muted">등록된 기기가 없습니다.</span>`;
    return;
  }
  const mineTail = notifyState.subscription ? notifyState.subscription.endpoint.slice(-12) : null;
  box.innerHTML = list.map((d) => {
    const mine = d.endpoint_tail === mineTail;
    const when = d.subscribed_utc ? fmtMacroCalendarTime(d.subscribed_utc) : "-";
    // 브라우저 UA 문자열은 길고 읽기 어렵다 -- 사람이 알아볼 만한 부분만 뽑는다.
    const name = (d.label || "").match(/(Chrome|Edg|Firefox|Safari|Android|iPhone|iPad|Windows|Macintosh)/g);
    const pretty = name ? [...new Set(name)].join(" · ") : "알 수 없는 기기";
    return `<div class="notify-device${mine ? " mine" : ""}">
      <div class="notify-device-main">
        <strong>${pretty}${mine ? " <em>이 기기</em>" : ""}</strong>
        <span class="muted">${when} 등록 · ${d.endpoint_tail}</span>
      </div>
      <button type="button" class="notify-btn small" data-revoke="${d.id}">해지</button>
    </div>`;
  }).join("");
  box.querySelectorAll("[data-revoke]").forEach((b) => b.addEventListener("click", async () => {
    b.disabled = true;
    await fetch("/api/push/unsubscribe", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: b.dataset.revoke }),
    }).catch(() => {});
    // 이 기기를 해지했다면 브라우저 쪽 구독도 같이 지운다 -- 서버에서만 지우면 브라우저에는
    // 유령 구독이 남아 "켜져 있는데 안 온다" 상태가 된다.
    const mine = notifyState.subscription
      && notifyState.subscription.endpoint.slice(-12) === (notifyState.devices.find((d) => d.id === b.dataset.revoke) || {}).endpoint_tail;
    if (mine) { try { await notifyState.subscription.unsubscribe(); } catch (e) { /* 이미 해지됨 */ } }
    await refreshNotifyPage();
  }));
}

function setNotifyTestResult(text, tone) {
  const p = el("notifyTestResult");
  if (!p) return;
  p.textContent = text;
  p.className = `notify-test-result ${tone || ""}`;
}

async function refreshNotifyPage() {
  if (!pushSupported()) { renderNotifyChecks(); renderNotifyMain(); return; }
  try {
    notifyState.config = await (await fetch("/api/push/config", { cache: "no-cache" })).json();
  } catch (e) { notifyState.config = null; }
  try {
    notifyState.registration = (await navigator.serviceWorker.getRegistration("/dashboard/live/")) || null;
    notifyState.subscription = notifyState.registration
      ? await notifyState.registration.pushManager.getSubscription() : null;
  } catch (e) { notifyState.registration = null; notifyState.subscription = null; }
  try {
    notifyState.devices = (await (await fetch("/api/push/devices", { cache: "no-cache" })).json()).devices || [];
  } catch (e) { notifyState.devices = []; }
  renderNotifyChecks(); renderNotifyMain(); renderNotifyDevices();
}

async function setupNotifyPage() {
  if (!pushSupported()) { renderNotifyChecks(); renderNotifyMain(); return; }
  try {
    await navigator.serviceWorker.register(PUSH_SW_URL, { scope: "/dashboard/live/" });
    await navigator.serviceWorker.ready;   // active 상태가 될 때까지 기다린다
  } catch (err) {
    console.warn("service worker 등록 실패", err);
  }
  await refreshNotifyPage();

  el("notifyToggleBtn")?.addEventListener("click", async () => {
    const btn = el("notifyToggleBtn");
    btn.disabled = true; btn.textContent = "처리 중…";
    try {
      if (notifyState.subscription) {
        await fetch("/api/push/unsubscribe", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ subscription: notifyState.subscription.toJSON() }),
        });
        await notifyState.subscription.unsubscribe();
      } else {
        const permission = await Notification.requestPermission();
        if (permission === "granted") {
          const sub = await notifyState.registration.pushManager.subscribe({
            userVisibleOnly: true,
            applicationServerKey: urlBase64ToUint8Array(notifyState.config.vapid_public_key),
          });
          await fetch("/api/push/subscribe", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ subscription: sub.toJSON(), label: navigator.userAgent.slice(0, 80) }),
          });
        }
      }
    } catch (err) {
      console.warn("구독 변경 실패", err);
      setNotifyTestResult(`설정을 바꾸지 못했습니다: ${err.message}`, "bad");
    }
    await refreshNotifyPage();
  });

  el("notifyLocalTestBtn")?.addEventListener("click", async () => {
    // 푸시 경로를 통째로 건너뛰고 서비스워커에게 직접 띄우라고 한다. 이게 안 보이면
    // 문제는 전달이 아니라 이 기기(브라우저 설정·OS 알림·집중 모드)에 있다.
    setNotifyTestResult("띄우는 중…", "");
    try {
      await notifyState.registration.showNotification("로컬 테스트", {
        body: "서버를 거치지 않고 이 브라우저가 직접 띄운 알림입니다.",
        icon: "/dashboard/live/icons/icon-192.png",
        badge: "/dashboard/live/icons/badge-96.png",
        tag: "local-test",
      });
      setNotifyTestResult("띄웠습니다. 화면에 안 보이면 브라우저·OS의 알림 설정(집중 지원, 방해 금지)을 확인해주세요.", "warn");
    } catch (err) {
      setNotifyTestResult(`띄우지 못했습니다: ${err.message}`, "bad");
    }
  });

  el("notifyServerTestBtn")?.addEventListener("click", async () => {
    setNotifyTestResult("서버에 발송을 요청했습니다…", "");
    try {
      const r = await (await fetch("/api/push/test", { method: "POST" })).json();
      setNotifyTestResult(
        r.sent > 0
          ? `서버가 ${r.sent}대에 보냈습니다. 몇 초 안에 도착하지 않으면 위의 로컬 테스트로 표시 자체가 되는지 먼저 확인해주세요.`
          : `발송 대상이 없습니다 (보냄 ${r.sent} · 정리 ${r.pruned} · 실패 ${r.failed}). 알림을 껐다 다시 켜주세요.`,
        r.sent > 0 ? "good" : "bad");
      await refreshNotifyPage();
    } catch (err) {
      setNotifyTestResult(`발송 요청이 실패했습니다: ${err.message}`, "bad");
    }
  });
}

// PWA 설치 -- Chrome/Edge의 기본 UI는 주소창 아이콘이라 놓치기 쉬워서 알림 페이지에도 둔다.
window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  notifyState.installPrompt = event;
  const btn = el("notifyInstallBtn");
  if (!btn) return;
  btn.classList.remove("hidden");
  btn.onclick = async () => {
    if (!notifyState.installPrompt) return;
    notifyState.installPrompt.prompt();
    await notifyState.installPrompt.userChoice;
    notifyState.installPrompt = null;
    btn.classList.add("hidden");
  };
});
window.addEventListener("appinstalled", () => el("notifyInstallBtn")?.classList.add("hidden"));

setupNotifyPage();

// 2026-09-11 청산 위험 패널·칩 모두 제거(사용자 지시). /api/position-sizing 은 아직 살아 있다.


// ── 2026-09-12 수동 진입(1단계: 미리보기 전용) ──────────────────────────────────
// 왜: 실계좌 19왕복에서 손실은 방향이 아니라 **크기**에서 왔다(상관 −0.494). 한 건이
// 중앙 명목의 5.2배·28.4배 레버리지로 −548.88 을 냈고, 그것만 잘라도 누적이 뒤집힌다.
// 버튼이 크기를 정하면 진입 순간의 재량이 사라진다.
// 지금은 주문이 나가지 않는다 -- 서버 게이트(DASHBOARD_MANUAL_EXEC_ENABLED)가 닫혀 있고,
// 미리보기는 실주문과 **같은 함수**(build_entry_plan)를 통과한다.
const MANUAL_ENTRY_REFRESH_MS = 60000;

// 숫자가 스스로 뜻을 말하게 쓴다 -- 분모·일상단위·결정연결까지(2026-09-11 사용자 지시).
// 「6,802 USDT」는 명목이지 내 돈이 아니고, 「30배」는 청산 거리가 아니라 잠기는 증거금이다.
// 교차 마진이라 청산 거리는 순자산/총명목으로 정해진다 -- 설정 레버리지는 거기 안 들어간다.
const won = (x) => Number(x).toLocaleString(undefined, { maximumFractionDigits: 0 });

// 2026-09-16 진입 미리보기의 타일 넷을 없애면서 entryTile/entryVal/ENTRY_* 도 함께 나갔다
// -- 그 블록 전용 헬퍼였다. 같은 숫자는 위 계좌 카드가 그린다(applyAcctPreview).

function manualEntryPlanHtml(data) {
  const plan = data.plan || {};
  const cap = data.cap || {};
  const dir = plan.positionSide === "LONG" ? "롱" : "숏";
  const parts = [`<div class="entry-head"><b>${dir} ${escapeHtml(String(plan.quantity))} ETH</b>
      <span>@ ${escapeHtml(Number(plan.price).toLocaleString())}</span>
      <span>내 돈 ${escapeHtml(won(plan.margin_usdt || 0))} USDT${
        plan.leverage ? ` = 명목 ${escapeHtml(won(plan.notional_usdt))} ÷ ${plan.leverage}배` : ""}</span>
    </div>`];

  // 2026-09-16 사용자 요청: 「롱 진입을 누르면 드롭다운으로 나오던 내용 제거 -- 이미 모달 안에
  //   내용이 있어」. 여기 있던 「진입 후 계좌」 타일 넷(청산까지·증거금 사용·계좌 노출·상한
  //   사용) 중 앞 셋은 바로 위 「지금 넣으면」 블록과 **같은 숫자**였다. 같은 사실을 두 번
  //   쓰면 어느 쪽이 결정인지 흐려진다. 남는 것은 위가 말하지 않는 것뿐이다 --
  //   수량·체결가(머리줄) · 막힘 · 손절 · 실주문 경고.

  // 막는 것만 항상 보인다. 나머지 설명은 «자세히» 뒤로 접는다(사용자 요청 2026-09-13) --
  // 진입 화면은 «얼마를 넣나»와 «왜 못 넣나»만 보이면 되고, 근거는 펼쳐서 읽는 것이다.
  if (plan.blocked) parts.push(`<div class="entry-note bad">🔴 ${escapeHtml(plan.blocked)}</div>`);
  // 손절은 **접지 않는다** -- «얼마를 잃을 수 있나»는 행동을 바꾸는 값이다.
  const sl = plan.stop_plan;
  if (sl) {
    parts.push(entryNote(`손절 ${Number(sl.stopPrice).toFixed(2)} `
      + `(평단 ${Number(sl.entry_price).toFixed(2)} 에서 ${(100 * sl.stop_pct).toFixed(1)}%)`
      // 🔴꼬리를 **숫자로** 말한다. 「급락 시 더」는 크기를 안 알려줘 사용자가 배수를 못 잡는다 --
      // 6배에서 보통 19% 인데 100번에 1번은 32% 라 자릿수가 다르다(2026-09-14 봉단위 검사에서
      // 실제 한 건 -22.8%). 값이 없으면 예전 문구로 떨어진다.
      + (sl.account_loss_pct != null
         ? ` — 걸리면 계좌 ${sl.account_loss_pct}% 손실`
           + (sl.account_loss_expected_pct
              ? ` (시장가라 보통 ~${sl.account_loss_expected_pct}%`
                + (sl.account_loss_tail_pct
                   // 🔴꼬리가 100% 를 넘으면 «132%» 가 아니라 «전액»이라고 말해야 한다.
                   // 그 배수에서는 꼬리 손절이 청산선 밖이라 실제로는 청산이 먼저 온다.
                   ? (sl.account_loss_tail_pct >= 100
                      ? `, 100번에 1번은 **전액**)`
                      : `, 100번에 1번은 ~${sl.account_loss_tail_pct}%)`)
                   : `, 급락 시 더)`)
              : "")
         : "")));
  }
  // «자세히» 드롭다운도 같은 요청으로 뺐다. 여기 접혀 있던 위험모델 한 줄·처방 줄은
  // 모달 위쪽(보유 예정·레버리지·「지금 넣으면」)이 이미 같은 말을 한다.
  // 🔴plan.notes 만은 버리지 않는다 -- «수량을 왜 깎았나»라서 위가 말해주지 않는다.
  (plan.notes || []).forEach((n) => parts.push(entryNote(`⚠ ${n}`)));
  parts.push(`<div class="entry-note${plan.dry_run ? "" : " live"}">${
    plan.dry_run ? "미리보기 전용 — 주문은 나가지 않습니다."
                 : "확인 버튼을 누르면 실제 주문이 나갑니다."} · peg 지정가(메이커), 미체결 ${
    plan.fallback_after_sec}초 후 테이커 전환</div>`);
  return parts.join("");
}

// 접히는 설명 블록. 줄이 없으면 버튼도 안 만든다(빈 «자세히»는 노이즈다).
function entryDetailHtml(lines) {
  const rows = (lines || []).filter(Boolean);
  if (!rows.length) return "";
  const open = detailOpenKeys.has("entrycard");
  return `<button type="button" class="detail-toggle" aria-expanded="${open}"`
    + ` onclick="toggleEntryDetail(this)">${open ? "접기 ▴" : "자세히 ▾"}</button>`
    + `<div class="signal-detail${open ? " open" : ""}">`
    + rows.map((r) => `<div class="entry-cap">${r}</div>`).join("") + `</div>`;
}

function entryNote(text, tone) {
  return `<div class="entry-note${tone ? " " + tone : ""}">${escapeHtml(text)}</div>`;
}

// 2026-09-13 «지금 상황» 플랜 한 줄: 보유시간 권고 · 기대 체결 · 예산 사다리. 서버 plan.trade_plan.
function tradePlanLines(tp, targetLev) {
  if (!tp) return [];
  const out = [];
  // ⭐처방: 세 값을 한 줄로. 이게 «이번 진입을 어떻게 하라»의 전부다.
  const rx = tp.prescription;
  if (rx && rx.available) {
    const hb = Object.entries(rx.hold_by_acc || {})
      .map(([a, m]) => `${Math.round(100 * Number(a))}%→${m}분`).join(" ");
    // 🔴손절이 있으면 청산거리는 **도달 못 하는 선**이다. 위험 지표를 손절 기준으로 바꾼다
    // (2026-09-13): 손절당 계좌 손실과 «연속 몇 번 견디나»가 실제로 계좌를 정한다.
    const sr = rx.stop_risk || {};
    out.push(`처방 ${rx.leverage}배 · ${rx.hold_min}분 · ${rx.tranches}회(일괄)`
      + (sr.available
         ? ` — 손절당 계좌 ${sr.per_stop_pct}% · 연속 ${sr.consecutive_to_half}회면 반토막`
           + ` · 연 ${sr.expected_stops_per_year}회 예상`
         : ` — 청산거리 ${rx.liq_distance_pct}%`)
      + ` · 손익분기 실력 ${Math.round(100 * rx.breakeven_acc)}%`);
    // 🔴«파산»이 아니라 이 값이 배수를 묶는다(2026-09-14). 실측 L=16 은 파산 0% 인데
    // 1년 중앙 계좌가 0.013배였다. 소수점은 못 믿으니 10%p 밴드로 말한다.
    if (rx.expected_mdd != null) {
      const lo = Math.max(0, Math.round(100 * rx.expected_mdd / 10) * 10 - 10);
      const hi = Math.min(100, lo + 20);
      out.push(`이 배수로 1년 굴리면 중간에 겪을 최대 낙폭 **약 ${lo}~${hi}%**`
        + ` (파산은 손절이 막지만 낙폭은 안 막습니다)`);
    }
    if (sr.available) {
      // 🔴«도달 불가»는 **명목 3% 기준** 주장이다. 시장가라 실제 이동은 손절폭+슬리피지이고
      // 99분위면 5.3% 라, 20배부터는 꼬리에서 **청산이 먼저 온다**(하드캡 25배 안이다).
      // 중앙값으로 «안전»을 말하고 꼬리를 안 적으면 화면이 낙관을 판다 -- 세 갈래로 말한다.
      out.push(!sr.liq_unreachable
        ? `🔴손절이 청산선 ${sr.liq_distance_pct}% 밖입니다 — 보호가 안 됩니다`
        : sr.liq_unreachable_tail === false
          ? `청산선 ${sr.liq_distance_pct}% 는 보통 손절(3%)이 먼저 막지만, 🔴**급락 시(99분위 5.3%)`
            + `에는 청산이 먼저 옵니다** — 이 배수에서는 손절이 끝까지 지켜주지 않습니다`
          : `청산선 ${sr.liq_distance_pct}% 는 손절이 먼저 와서 **도달 불가** (급락 99분위 5.3% 까지도)`
            + ` — 위험은 손절 반복에서 옵니다`);
    }
    out.push(`크기는 ${rx.size_source} · 분할 ${rx.tranche_reason}`);
    // 거래소 레버리지 설정. 위험이 아니라 «상한을 거래소에 새기는 값»이라 문구도 그렇게 쓴다.
    const lv = rx.exchange_leverage;
    if (lv && lv.available) {
      // 🔴실제로 거래소에 걸리는 값은 **plan.target_leverage** 다. 2026-09-20 부터 화면이
      //   물타기에서는 «기존 포지션과 같은 값», 신규에서는 20배로 고정해 보내므로 모델의
      //   lv.setting 과 다를 수 있다 -- 여기에 모델값을 적으면 «본 숫자»와 «걸리는 숫자»가
      //   갈린다. 다를 때는 뒤따르는 파생 문구도 떼어낸다: margin_pct_of_equity 와
      //   enforces_cap 은 둘 다 lv.setting 으로 계산된 값이라 그대로 붙이면 거짓말이 된다.
      const set = Number(targetLev) || lv.setting;
      const same = set === lv.setting;
      out.push(`거래소 레버리지 ${set}배로 설정`
        + (same ? "" : ` (모델 추천 ${lv.setting}배)`) + ` — ${lv.note}`
        + (!same ? "" : ` · 증거금 ${lv.margin_pct_of_equity}% 잠김`
           + (lv.forced_by_position ? "" :
              lv.enforces_cap ? " · 화면을 우회해도 상한이 걸립니다"
                              : " · ⚠거래소 천장이 상한보다 큽니다(눈금이 성깁니다)")));
    }
  }
  const h = tp.hold || {};
  if (h.available) {
    const cur = (h.table || []).find((r) => r.hold_min === tp.hold_min);
    const grid = Object.entries(h.best_by_acc || {})
      .map(([a, m]) => `${Math.round(100 * Number(a))}%→${m}분`).join(" ");
    out.push(`${h.reason}`
      + (cur ? ` · 선택 ${tp.hold_min}분: 움직임 ${cur.move_bp}bp · 손익분기 ${Math.round(100 * cur.breakeven_acc)}%` : "")
      + (grid ? ` · 정확도별 최적 ${grid}` : ""));
  }
  const ex = (tp.execution || {}).entry || {};
  if (ex.expected_fill_sec != null) {
    out.push(`기대 체결 ${ex.expected_fill_sec}초 (메이커 ${ex.maker_bp}bp · 폴백 테이커 ${ex.taker_bp}bp) — ${ex.note}`);
  }
  const sp = tp.entry_split || {};
  if (sp.rule) out.push(`${sp.tranches === 1 ? "일괄" : sp.tranches + "분할"} — ${sp.rule}`);
  const lad = tp.exit_ladder || {};
  if ((lad.ladder || []).length) {
    const steps = lad.ladder.filter((r) => r.required_fraction > 0)
      .map((r) => `${r.hold_min}분 → ${Math.round(100 * r.required_fraction)}% 닫기`);
    out.push(`${lad.note}${steps.length ? " · " + steps.join(" · ") : ""}`);
  }
  return out;
}

// 청산 비율(%)을 읽는 **유일한** 곳. 미리보기와 실주문이 같은 값을 쓰게 한다.
// 서버도 같은 값을 다시 검증하고 포지션을 다시 읽는다 -- 여기 값은 «요청»이지 «수량»이 아니다.
function sliderPct(id) {
  const el0 = el(id);
  const v = el0 ? Number(el0.value) : 100;
  return Number.isFinite(v) && v > 0 && v <= 100 ? v : 100;
}
const manualExitPct = () => sliderPct("snapExitFrac");
const manualEntryPct = () => sliderPct("snapEntryFrac");

// 🔴보유 예정 지평은 **서버가 정한다**(planning_hold, 2026-09-14). 여기 상수를 두면
// «화면엔 4시간인데 1440분 셀로 크기가 나가는» 일이 생긴다 -- 실제로 그랬다.
// 서버가 `hold_planned_min` 으로 돌려주는 값만 표시하고, 쿼리로는 안 보낸다.

// ── 2026-09-13 거래소 레버리지 게이지 ────────────────────────────────────────
// 위험이 아니라 **총 명목의 천장**을 정하는 값이다(교차 마진이라 청산거리는 순자산/총명목).
// 게이지는 거래소가 받는 눈금 위에서만 움직인다 -- 그 사이 값을 보내면 거래소가 반올림해서
// 화면과 실제가 어긋난다. 서버가 plan.leverage_steps 로 같은 배열을 내려준다.
let LEV_STEPS = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 50, 75, 100, 125, 150];
const manualLevAuto = () => el("snapLevAuto")?.checked !== false;
// 2026-09-20 게이지가 **값 자체**다(5단위, 사용자 지시). 옛 판은 LEV_STEPS 의 인덱스라
// 1·2·3·5·8·10·15… 처럼 간격이 들쭉날쭉했다. 서버는 1~EXCHANGE_MAX 정수면 받는다.
function manualLevValue() {
  const g = el("snapLevGauge");
  if (!g) return null;
  return Math.max(5, Math.round((Number(g.value) || 5) / 5) * 5);
}
// 자동이면 서버에 아무것도 안 보낸다 -- 서버가 모델 추천을 쓴다(단일 진실 원천).
// 2026-09-20 사용자 지시 둘.
// ① **물타기는 레버리지를 고를 수 없다.** 열린 포지션이 있으면 내리는 쪽은 거래소가 막고
//    (-2028 MIN_LEVERAGE_RATIO -- 내리면 기존 포지션 초기증거금이 올라가 순자산을 넘는다,
//    live_eth_trade_plan_20260913.py 의 position_floor), 남는 건 «올리기»뿐이다. 올릴 이유가
//    없으면 기존 값 그대로가 맞다 -- 같은 값이면 설정 자체가 안 바뀌어 거부될 수도 없다.
//    (position_floor = 명목/순자산 과는 다른 값이다. 여기서 고정하는 건 «거래소에 걸린
//     설정»이고, 포지션이 그 설정으로 이미 열려 있으므로 언제나 바닥 위에 있다.)
// ② **신규 진입의 «자동»은 모델 추천이 아니라 20배 고정**이다.
// 🔴세 경로 -- 미리보기 쿼리 · 실주문 pending · 게이지 표시 -- 가 **이 한 함수**를 본다.
//   따로 두면 «미리보기는 20배인데 주문은 모델값»이 된다.
const MANUAL_LEV_AUTO = 20;
const manualLevLocked = () => {
  const v = Number(snapshotAccountPosition()?.leverage) || 0;
  return v > 0 ? v : null;
};
function manualLevEffective() {
  const locked = manualLevLocked();
  if (locked) return locked;
  return manualLevAuto() ? MANUAL_LEV_AUTO : manualLevValue();
}
const manualLevQuery = () => {
  const v = manualLevEffective();
  return v ? `&lev=${v}` : "";
};

function renderLevGauge(plan) {
  const g = el("snapLevGauge");
  const out = el("snapLevVal");
  if (!g || !out) return;
  // 🔴이 함수는 값을 **코드로** 바꾼다(자동 추천). input 이벤트가 안 나므로 칩이 안 따라온다
  //   -- 끝에서 직접 맞춘다. 이벤트를 쏘면 크기 재조회가 돌아 되먹임이 된다.
  setTimeout(syncChipsets, 0);
  // 상한은 여전히 서버가 정한다 -- 정책 목록의 최대값을 5단위로 내림해서 게이지 끝에 둔다.
  if (Array.isArray(plan.leverage_steps) && plan.leverage_steps.length) {
    LEV_STEPS = plan.leverage_steps;
    const top = Math.max(5, Math.floor(Math.max(...LEV_STEPS) / 5) * 5);
    g.max = String(top);
  }
  const locked = manualLevLocked();
  // 물타기면 게이지도 «자동» 체크박스도 **감춘다** -- 고를 수 없는 것을 고를 수 있는 것처럼
  // 보여주면 안 된다. 기존 레버리지가 5의 배수가 아닐 수도 있어(예 ×3) 5단위 게이지로는
  // 그 값을 정확히 나타내지도 못한다. 값은 아래 꼬리표가 숫자로 말한다.
  g.hidden = !!locked;
  el("snapLevAuto")?.closest(".lev-auto")?.toggleAttribute("hidden", !!locked);
  // 자동이면 20배로 스냅한다. 손으로 만지는 중이면 안 건드린다.
  if (!locked && manualLevAuto()) {
    g.value = String(MANUAL_LEV_AUTO);
    syncRangeFill(g);       // 프로그램이 바꾼 값은 input 이벤트가 없다
  }
  g.disabled = !!locked || manualLevAuto();
  const v = manualLevEffective();
  if (v == null) return;
  const min = plan.leverage_min_feasible;
  const floor = plan.leverage_position_floor;
  // 포지션 바닥 아래는 **거래소가 거부한다**(-2028). 상한 경고보다 이게 먼저다.
  // 고정된 값은 정의상 바닥 위지만(그 설정으로 이미 열려 있다) 검사는 남긴다 -- 서버가
  // 주는 값이라 내 가정이 틀리면 조용히 넘어가는 대신 화면이 말하게 한다.
  const rejected = floor != null && v < floor;
  const low = !rejected && min != null && v < min;
  out.textContent = `${v}배`
    + (locked ? " (기존 포지션과 동일)" : manualLevAuto() ? " (자동)" : " (수동)")
    + (plan.leverage_model && v !== plan.leverage_model ? ` · 모델 ${plan.leverage_model}배` : "")
    + (rejected ? ` · 🔴거래소가 거부합니다(포지션 때문에 최소 ${Math.ceil(floor)}배)`
       : low ? ` · ⚠상한만큼 못 엽니다(최소 ${Math.ceil(min)}배)` : "");
  // 🔴같은 줄을 두 번 쓰고 있었다 -- 뒤 줄이 앞 줄을 덮어 **rejected(거래소 거부)가 색을
  //   잃었다**. 둘 중 경고가 더 급한 쪽이 지워지던 셈이라 고친다.
  out.className = (rejected || low) ? "entry-was bad" : "entry-was";
  syncRangeFill(g);
}

async function manualEntryFetch(side, kind = "entry") {
  const q = `&pct=${kind === "exit" ? manualExitPct() : manualEntryPct()}`
    + (kind === "exit" ? "" : manualLevQuery());
  const res = await fetch(`/api/manual-${kind}/preview?side=${side}${q}`, { cache: "no-cache" });
  return res.json();
}

// 위험 모델이 무엇을 말하는지 한 줄로. 모델이 없으면 그 사실을 그대로 쓴다.
function riskLine(r) {
  if (!r) return "";
  if (!r.available) return `위험모델 없음 (${r.reason || "?"}) — 기존 상한만 적용`;
  // 🔴«묶은 것»은 **적용된 상한**을 말해야 한다(2026-09-13). r.binding 은 위험 정책 안에서만
  // 고른 값이라 순자산·원장 상한을 모른다 -- 그대로 쓰면 «최대 25배 · 정책상한»이라고 적히는데
  // 버튼은 8배까지만 낸다. 실효 배수(effective_x)와 applied_binding 이 진짜다.
  const NAMES = { survival: "생존(청산거리)", growth: "성장(켈리 하한)", cap: "정책상한",
                  ledger: "원장 상한", equity: "순자산 상한", model: "위험 모델" };
  const who = NAMES[r.applied_binding || r.binding] || r.applied_binding || r.binding;
  const lev = r.effective_x != null ? r.effective_x : r.leverage;
  // 🔴상한이 꺼져 있으면 «무엇이 묶었나»가 아니라 «아무것도 안 묶었다»가 사실이다.
  if (r.applied_binding === "override") {
    return `${r.hold_min}분 보유 기준 각오할 역행 ${r.safe_mae_pct}%`
      + ` · 🔴사이징 상한 꺼짐 — 최대 ${lev}배 (순자산 × ${lev})`;
  }
  // 위험 정책이 허용한 값과 실제로 적용된 값이 다르면 둘 다 보여 준다.
  const head = r.effective_x != null && r.leverage != null && r.effective_x < r.leverage
    ? `최대 ${lev}배 (위험 모델은 ${r.leverage}배까지 허용)`
    : `최대 ${lev}배`;
  return `${r.hold_min}분 보유 기준 각오할 역행 ${r.safe_mae_pct}% → ${head} · ${who}이 묶음`;
}

// 2026-09-13 청산 미리보기. 진입 카드는 상한·증거금 타일이 주인공이지만 청산은 «얼마를
// 어느 가격에 닫는가»와 «지금 닫으면 몇 %인가»가 전부라 따로 그린다.
function manualExitPlanHtml(plan) {
  const side = plan.position_side === "LONG" ? "롱" : "숏";
  const move = plan.exit_move_pct;
  const pct = Math.round(100 * (plan.fraction ?? 1));
  const of = pct < 100
    ? ` <span class="entry-was">${plan.position_qty} 중 ${pct}% · 남김 ${plan.remaining_qty}</span>`
    : "";
  const parts = [`<div class="entry-head"><b>${side} ${plan.quantity} ETH 청산</b>${of}`
    + `<span>${Number(plan.price ?? plan.reference_price).toFixed(2)}`
    + `${plan.type === "MARKET" ? " 근처" : ""} · ${Number(plan.notional_usdt).toLocaleString()} USDT</span></div>`];
  // 🔴미리보기 카드의 주인공은 «지금 닫으면 순손익 얼마»다(2026-09-14, 사용자 요청).
  // 여기서는 plan.type 을 알므로 고변동 시장가 전환이면 테이커 5.0bp 로 바꿔 계산한다.
  const upnl = Number(plan.unrealized_pnl);
  const feeBp = plan.type === "MARKET" ? EXIT_FEE_BP_TAKER : EXIT_FEE_BP_PEG;
  const fee = (Number(plan.notional_usdt) || 0) * feeBp / 10000;
  if (Number.isFinite(upnl)) {
    const gross = upnl * (plan.fraction ?? 1);
    const net = gross - fee;
    parts.push(`<div class="entry-note"><span class="entry-cap">지금 닫으면</span> `
      + `<b class="exit-net ${net >= 0 ? "good" : "bad"}">${usd2(net)}</b>`
      + `<span class="entry-was"> 순손익 · 미실현 ${usd2(gross)}`
      + ` − ${plan.type === "MARKET" ? "테이커" : "peg"} 수수료 ≈$${fee.toFixed(2)} (${feeBp}bp)`
      + (move != null ? ` · 진입가 대비 ${move > 0 ? "+" : ""}${move}%` : "")
      + ` · 진입 수수료는 이미 나갔습니다</span></div>`);
  } else if (move != null) {
    parts.push(entryNote(`이 가격이면 진입가 대비 ${move > 0 ? "+" : ""}${move}% (수수료 전)`,
      move >= 0 ? null : "bad"));
  }
  const r = plan.risk;
  if (r && r.available && r.required_fraction > 0) {
    parts.push(entryNote(
      `${r.hold_min}분 더 들 생각이면 명목 ${Math.round(r.allowed_notional).toLocaleString()} 까지가 한도입니다`
      + ` (지금 ${Math.round(r.current_notional).toLocaleString()}) — `
      + `최소 ${Math.round(100 * r.required_fraction)}% 는 닫아야 합니다`, "bad"));
  }
  if (plan.blocked) parts.push(entryNote(plan.blocked, "bad"));
  // 시장가로 전환된 경우엔 이유를 **위쪽에** 띄운다 -- 비용이 더 드는 선택이라 묻히면 안 된다.
  if (plan.market_reason) parts.push(entryNote(plan.market_reason, "live"));
  const vol = plan.vol_bpm != null ? ` · 변동성 ${plan.vol_bpm} bp/√분` : "";
  const how = plan.type === "MARKET"
    ? "시장가 즉시 체결"
    : `peg 지정가(메이커), 호가가 달아나면 재호가하고 ${plan.fallback_after_sec}초 뒤에도 `
      + "남으면 테이커 전환";
  // 진입 카드와 **같은 규칙**으로 접는다(2026-09-13 병합). 위험모델 한 줄과 처방 줄들이
  // 대상이고, «최소 N% 는 닫아야 합니다»는 행동을 바꾸는 값이라 위에서 이미 항상 보인다.
  parts.push(entryDetailHtml([
    ...(r && !(r.required_fraction > 0) ? [escapeHtml(riskLine(r))] : []),
    ...tradePlanLines(plan.trade_plan, plan.target_leverage).map(escapeHtml),
  ]));
  parts.push(`<div class="entry-cap">${plan.dry_run
    ? "미리보기 전용 — 주문은 나가지 않습니다."
    : "확인 버튼을 누르면 실제 청산 주문이 나갑니다."} · ${how}${vol}</div>`);
  return parts.join("");
}

const MANUAL_BTN_IDS = ["snapEntryLong", "snapEntryShort", "snapExitLong", "snapExitShort"];
const manualButtonsDisabled = (v) =>
  MANUAL_BTN_IDS.forEach((id) => { const b = el(id); if (b) b.disabled = v; });

async function manualEntryPreview(side, kind = "entry") {
  const box = el("snapEntryResult");
  if (!box || manualOrderBusy) return;   // 진행 중인 주문 표시를 덮지 않는다
  manualButtonsDisabled(true);
  box.hidden = false;
  box.innerHTML = entryNote("확인 중…");
  manualEntryClearConfirm();   // 다른 방향을 눌렀는데 옛 확인 버튼이 남아 있으면 안 된다
  try {
    const data = await manualEntryFetch(side, kind);
    box.innerHTML = data.ok
      ? (kind === "exit" ? manualExitPlanHtml(data.plan || {}) : manualEntryPlanHtml(data))
      : entryNote(`실패: ${data.detail || data.error || "알 수 없음"}`, "bad");
    // 게이트가 꺼져 있으면 확인 버튼을 아예 띄우지 않는다 -- 눌러도 403 이라 헛걸음이다.
    if (data.ok && data.exec_enabled) manualEntryArmConfirm(side, data.plan || {}, kind);
  } catch (err) {
    box.innerHTML = entryNote(`실패: ${err && err.message ? err.message : err}`, "bad");
  } finally {
    if (!manualOrderBusy) manualButtonsDisabled(false);
  }
}

// 2026-09-15 접힌 «추가 진입» 줄에 **대가**를 적는다(사용자 요청). 포지션이 있으면 이 블록은
// 접히는데(09-14, 물타기를 한 번 더 확인시키려고) 그 바람에 청산·노출 변화가 같이 사라졌다.
// ⭐접기를 없애지 않는다 -- 오히려 «대가가 접힘 밖에 보이는 것»이 그 확인의 목적에 더 맞다.
// 새 요청을 만들지 않는다: 크기 갱신이 이미 받아온 `plan.projection` 을 그대로 쓴다.
function renderEntryFoldNote(plan) {
  const note = el("snapEntryFoldNote");
  if (!note) return;
  const box = el("snapEntryBox");
  const pr = plan.projection;
  const a = pr && pr.after, b = pr && pr.before;
  // 펼쳐져 있으면 아래에 카드가 그대로 보인다 -- 같은 숫자를 두 번 쓰지 않는다.
  if (!a || !b || a.liq_pct == null || (box && box.open) || plan.blocked) {
    note.textContent = ""; note.className = "entry-was"; return;
  }
  note.textContent = ` · 지금 추가하면 청산까지 ${Number(b.liq_pct).toFixed(1)}% → `
    + `${Number(a.liq_pct).toFixed(1)}%`
    + (a.exposure_x != null && b.exposure_x != null
       ? ` · 노출 ${Number(b.exposure_x).toFixed(1)} → ${Number(a.exposure_x).toFixed(1)}배` : "");
  note.className = `entry-was ${acctRiskTone(a.liq_pct)}`;
}

// 2026-09-20 계좌 카드 게이지를 진입 미리보기로 움직인다(사용자 요청). 09-16 에 이 자리에
// 있던 «지금 넣으면» 4행(#entryProj 블록과 그 전용 렌더러들)은 통째로 지웠다 -- 진입이
// 모달에서 카드 안으로 나온 뒤로는 **바로 위 계좌 카드**가 같은 네 값을 같은 눈금으로 이미
// 그린다. 같은 사실을 두 번 그리면 어느 쪽이 결정인지 흐려진다.
// 값이 실제로 바뀔 때만 손댄다 -- 30초마다 같은 값으로 재그리면 툴팁이 닫히고 스크롤이 튄다.
// 🔴켜는 조건은 «진입 블록이 펼쳐져 있다» 다. 접혀 있으면 사용자는 진입을 보고 있지 않으므로
//    계좌 카드는 **실제 계좌**여야 한다. 실패·차단·투영 없음도 전부 끄는 쪽이다.
function setEntryProjPreview(plan) {
  const box = el("snapEntryBox");
  const pr = plan && !plan.blocked && Number(plan.quantity) > 0 ? plan.projection : null;
  const on = pr && pr.after && box && box.open ? pr : null;
  const key = on ? `${on.after.liq_pct}|${on.after.margin_used_pct}|${on.after.exposure_x}|${plan.quantity}` : "";
  if (key === entryProjKey) return;
  entryProjKey = key;
  entryProjPreview = on ? { ...on, __plan: plan } : null;
  // ⭐켜거나 값이 바뀌면 **제자리에서** 고친다 -- 그래야 CSS transition 이 걸린다.
  // 🔴끌 때는 **통째로 다시 그린다**. 제자리 수정은 실제 값을 덮어쓴 뒤라 «지우기»만으로는
  //    복구가 안 된다(2026-09-15 테스트에서 실제로 미리보기 값이 굳었다). 렌더는 항상
  //    실제 계좌를 그리므로 재그리기가 곧 복구다.
  const ctx = lastAcctPos;
  if (on && ctx && el("snapAcctPosition")?.querySelector(".acct-tiles")) {
    applyAcctPreview(ctx.pos, ctx.mark, ctx.equity);
  } else {
    renderSnapshotAccount();
  }
}

async function manualEntryRefreshSize() {
  const line = el("snapEntrySize");
  if (!line) return;
  // 🔴2026-09-13: 실패하면 배지를 **반드시 바꾼다**. 예전엔 조기 return 해서 배지가 HTML
  //   초기값("미리보기 전용")이나 마지막 성공값에 굳었다 -- 화면이 "게이트가 꺼져 있다"고
  //   말했지만 실제로는 "사이징 워커가 죽어 미리보기가 503"이었다(재부팅 후 실장애).
  //   **실패 시 이전 값을 남기는 UI 는 조용히 거짓말한다.**
  try {
    const data = await manualEntryFetch("LONG");
    if (!data.ok) {
      const why = data.detail === "worker_stale" ? "크기 워커 정지"
        : data.error === "sizing_unavailable" ? "크기 데이터 없음"
        : (data.error || "알 수 없음");
      line.hidden = false;
      line.textContent = `크기 확인 실패 — ${why}`;
      setEntryProjPreview(null);   // 🔴옛 투영을 남기지 않는다
      return;
    }
    const plan = data.plan || {};
    const cap = data.cap || {};
    // 🔴헤드라인은 **실제로 나가는 수량**이다(2026-09-14). 옛 판은 «권고 6.764 · 상한 2.739»
    //   처럼 둘을 나란히 놨는데, 나가는 건 둘 다 아니라 비율까지 먹인 plan.quantity 였다 --
    //   큰 숫자가 왼쪽에 먼저 오니 그게 주문량으로 읽혔다. 권고·가능치는 뒤로 내린다.
    //   (서버가 이 요청에 슬라이더 pct 를 이미 실어 보내므로 plan.quantity 가 그 비율의 값이다.)
    // 2026-09-16 정상일 때 이 줄은 **비어 있다**(사용자 지시). 수량·명목·권고 사슬은
    //   위 계좌 카드가 미리보기로 이미 보여준다. 주문 불가만 여기 남는다 -- 왜 못 넣는지는
    //   카드가 말해주지 않는다.
    // 🔴사이징 상한이 꺼져 있으면(server.py SIZING_CAP_OVERRIDE_X) **정상일 때도** 이 줄이
    //   말한다. 크기를 막는 게 아무것도 없는 상태를 화면이 조용히 넘기면 안 된다.
    //   riskLine 에도 같은 사실을 적지만 그쪽은 위험모델이 살아 있을 때만 그려진다 --
    //   워커가 죽으면 경고까지 같이 사라지므로 여기 한 곳은 무조건이어야 한다.
    const ovX = cap.override_x;
    line.hidden = !plan.blocked && !ovX;
    line.textContent = plan.blocked ? `주문 불가 — ${plan.blocked}`
      : ovX ? `🔴사이징 상한 꺼짐 — 크기 기준이 «순자산 × ${ovX}» 하나뿐입니다` : "";
    renderEntryFoldNote(plan);
    setEntryProjPreview(plan);
    // 보유시간 옆 배지: 이 시간 기준으로 모델이 각오하라는 역행폭과 허용 배수.
    const hb = el("snapHoldRisk");
    if (hb) {
      const r = (data.cap || {}).risk;
      renderLevGauge(plan);
      // 남은 보유시간을 같이 띄운다 -- 물타기를 해도 시계가 안 늘어난다는 사실이 보여야 한다.
      const left = plan.hold_remaining_min;
      const planned = plan.hold_planned_min;   // 크기를 실제로 정한 그 지평
      const hf = el("snapHoldPlanned");
      if (hf) hf.textContent = !planned ? "—"
        : (left && left < planned ? `${planned / 60}시간 (남은 ~${left}분)`
                                  : `${planned / 60}시간`);
      hb.textContent = r && r.available ? `역행 ${r.safe_mae_pct}% · 최대 ${r.leverage}배`
        : (r ? "모델 없음" : "—");
    }
  } catch (err) {
    line.hidden = false;
    line.textContent = "크기 확인 실패 — 서버 응답 없음";
    setEntryProjPreview(null);
  }
}


// ── 2026-09-12 2단계: 실주문 (2단 확인) ────────────────────────────────────────
// 진입 버튼은 계획을 **보여주기만** 하고, 주문은 확인 버튼에서만 나간다. 오클릭 한 번이
// 주문이 되지 않게 하는 것이 목적이라 확인 버튼은 기본 숨김이고 창이 지나면 사라진다.
// 서버도 confirm=1 을 따로 요구하므로 이 화면 로직이 깨져도 실수로 주문이 나가지 않는다.
// 🔴2026-09-19 15초 -> 30초. 사용자 보고 «청산 버튼을 눌러도 청산이 안 들어간다» 의 정체는
// 서버가 아니라 이 창이었다(진단 당시 서버는 전부 정상: exec_enabled=true · phase=idle ·
// 어느 비율에서도 blocked=null). 청산은 2단계라 미리보기를 읽고 «확인»을 눌러야 나가는데,
// 계획 카드(수량·가격·미실현·위험한도)를 읽는 데 15초가 쉽게 지나간다.
const CONFIRM_WINDOW_MS = 30000;
const STATUS_POLL_MS = 3000;
let manualEntryPending = null;
let manualEntryTimer = null;
let manualEntryTick = null;
// 주문이 나간 뒤 상태를 폴링하는 동안엔 미리보기를 막는다 -- 결과 상자가 하나뿐이라
// 새 미리보기가 «체결 진행 중» 표시를 덮어쓴다(2026-09-14).
let manualOrderBusy = false;

const MANUAL_ENTRY_PHASE_KO = {
  submitting: "주문 전송 중…",
  working: "peg 지정가 대기 중 — 미체결분은 120초 뒤 테이커 전환",
  filled_maker: "✅ 전량 메이커 체결 (peg)",
  filled_taker: "✅ 체결 — 일부/전부 테이커 전환",
  rejected: "거부됨 — post-only 가 테이커가 될 상황이라 거절했습니다(주문 안 나감)",
  taker_failed: "🔴 테이커 전환 실패",
  error: "🔴 오류",
  idle: "대기",
};

function manualEntryClearConfirm() {
  manualEntryPending = null;
  if (manualEntryTimer) { clearTimeout(manualEntryTimer); manualEntryTimer = null; }
  if (manualEntryTick) { clearInterval(manualEntryTick); manualEntryTick = null; }
  const btn = el("snapEntryConfirm");
  if (btn) { btn.hidden = true; btn.textContent = ""; }
}

function manualEntryArmConfirm(side, plan, kind = "entry") {
  if (plan.blocked) return;
  const pct = Math.round(100 * (plan.fraction ?? 1));
  manualEntryPending = { side, quantity: plan.quantity, kind, pct,
                         lev: manualLevEffective() };
  // 🔴진입은 «길게 누르기»가 곧 확인이다 -- 확인 버튼을 띄우면 같은 주문이 두 번 나갈 길이
  //   생긴다(누르고 있는 동안 pending 이 잡히므로). 청산은 그대로 버튼으로 확인한다.
  const btn = el("snapEntryConfirm");
  if (manualHoldFire || !btn) return;
  // 모델이 요구하는 최소 청산 비율보다 적게 닫으려 하면 **확인 버튼에** 적는다.
  // 미리보기에만 띄우면 슬라이더를 다시 내린 뒤에는 안 보인다.
  const need = Math.round(100 * ((plan.risk || {}).required_fraction || 0));
  const short = kind === "exit" && need > pct ? ` ⚠한도 복귀엔 ${need}% 필요` : "";
  const base = `확인: ${side === "LONG" ? "롱" : "숏"} ${plan.quantity} ETH `
    + (kind === "exit" ? (pct < 100 ? `청산 (${pct}%)` : "전량 청산") : "주문") + short;
  btn.hidden = false;
  if (manualEntryTimer) clearTimeout(manualEntryTimer);
  if (manualEntryTick) clearInterval(manualEntryTick);
  // 2026-09-14 남은 초를 버튼에 적는다. 창이 조용히 닫히면 «왜 사라졌지»가 되고, 그 다음
  // 행동은 대개 «버튼을 다시 누른다»라 미리보기를 한 번 더 돌게 된다.
  let left = Math.round(CONFIRM_WINDOW_MS / 1000);
  const paint = () => { btn.textContent = `${base} · ${left}초`; };
  paint();
  manualEntryTick = setInterval(() => { left -= 1; if (left > 0) paint(); }, 1000);
  // 🔴만료를 **말한다**. 예전엔 조용히 사라져서 「눌렀는데 안 나갔다」로 보였다 --
  // 화면에 아무 흔적이 없으니 사람이 고장으로 읽는 게 당연했다.
  // clearConfirm 자체는 방향 전환·비율 변경에서도 불리므로 메시지는 여기(만료)에만 붙인다.
  manualEntryTimer = setTimeout(() => {
    manualEntryClearConfirm();
    const box = el("snapEntryResult");
    if (box && !box.hidden) {
      box.insertAdjacentHTML("beforeend", entryNote(
        `확인 시간 ${Math.round(CONFIRM_WINDOW_MS / 1000)}초가 지나 확인 버튼이 사라졌습니다 `
        + `— 주문은 나가지 않았습니다. 다시 누르세요.`, "bad"));
    }
  }, CONFIRM_WINDOW_MS);
  // 미리보기가 길면 확인 버튼이 접힌 화면 밖에 남는다(모바일). 눈앞으로 데려온다 --
  // 순서를 바꿔 결과 위에 두면 «계획을 읽기 전에» 확인이 먼저 보여서 더 나쁘다.
  btn.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function manualEntryStateText(state) {
  const phase = state?.phase || "idle";
  const rows = [MANUAL_ENTRY_PHASE_KO[phase] || phase];
  if (state?.quantity !== undefined) {
    rows.push(`체결 ${Number(state.filled || 0)} / ${Number(state.quantity)} ETH` +
      (state.taker_qty ? ` (테이커 ${Number(state.taker_qty)})` : ""));
  }
  const sl = state?.stop;
  const filledQty = Number(state?.filled || 0);
  if (sl && sl.placed) {
    rows.push(`손절 ${sl.stop_price} 걸림` + (sl.replaced ? ` (기존 ${sl.replaced}건 교체)` : ""));
  } else if (sl && sl.no_position) {
    // 체결이 0 이면 걸 포지션이 없다 -- 경고가 아니다. 이걸 안 가르면 «무방비» 가 거짓으로 뜬다.
  } else if (sl || filledQty > 0) {
    // 🔴`sl` 이 **없는데 체결은 있는** 경우가 진짜 위험하다(2026-09-13 감사). 예전에는
    // 이 조건이 `sl` 존재에만 걸려 있어, 손절 시도 자체가 없던 경로에서 경고가 조용했다.
    rows.push(`🔴손절을 못 걸었습니다 — 포지션이 무방비입니다 (${sl?.error || sl?.reason || "손절 시도 기록 없음"})`);
  }
  // 2026-09-15 진입도 리페그한다. 쫓아간 횟수는 «의도한 가격보다 높게 들어갔을 수 있다»는
  // 뜻이라 숨기지 않는다 -- 청산과 달리 진입은 안 사도 되는 선택지가 있었기 때문이다.
  if (Number(state?.repegs || 0) > 0) {
    rows.push(`리페그 ${Number(state.repegs)}회 — 호가를 따라갔습니다` +
      (state.limit_price ? ` (현재 지정가 ${state.limit_price})` : ""));
  }
  const lv = state?.leverage;
  if (lv && lv.error) rows.push(`⚠레버리지 ${lv.to}배 설정 실패 — 거래소 천장이 그대로입니다 (${lv.error})`);
  else if (lv && lv.changed) rows.push(`레버리지 ${lv.from}배 → ${lv.to}배 적용`);
  if (state?.error) rows.push(`사유: ${state.error}`);
  return rows.join("\n");
}

async function manualEntryPollStatus() {
  const box = el("snapEntryResult");
  if (!box) return;
  try {
    const res = await fetch("/api/manual-entry/status", { cache: "no-cache" });
    const data = await res.json();
    box.innerHTML = entryNote(manualEntryStateText(data.state),
      /^(error|taker_failed|rejected)$/.test(data.state?.phase || "") ? "bad" : "live");
    const phase = data.state?.phase;
    if (phase === "submitting" || phase === "working") {
      setTimeout(manualEntryPollStatus, STATUS_POLL_MS);
    } else {
      manualOrderBusy = false;
      manualButtonsDisabled(false);
      manualEntryRefreshSize();   // 체결되면 포지션·상한 표시를 갱신한다
    }
  } catch (err) {
    box.innerHTML = entryNote(`상태 조회 실패: ${err && err.message ? err.message : err}`, "bad");
    // 🔴조회가 실패했다고 버튼을 잠근 채 두면 그 포지션을 **못 닫는다**. 푼다.
    manualOrderBusy = false;
    manualButtonsDisabled(false);
  }
}

async function manualEntrySubmit() {
  const pending = manualEntryPending;
  const box = el("snapEntryResult");
  if (!pending || !box) return;
  manualEntryClearConfirm();
  manualOrderBusy = true;
  manualButtonsDisabled(true);
  box.hidden = false;
  box.innerHTML = entryNote("주문 전송 중…", "live");
  try {
    const q = `&pct=${pending.pct ?? 100}`
      + (pending.kind === "exit" ? "" : (pending.lev ? `&lev=${pending.lev}` : ""));
    const res = await fetch(
      `/api/manual-${pending.kind || "entry"}/submit?side=${pending.side}&confirm=1${q}`,
      { method: "POST", cache: "no-cache" });
    const data = await res.json();
    if (!data.ok) {
      box.innerHTML = entryNote(`주문 실패: ${data.detail || data.error || res.status}`, "bad");
      manualOrderBusy = false;
      manualButtonsDisabled(false);
      return;
    }
    box.innerHTML = entryNote(manualEntryStateText(data.state), "live");
    setTimeout(manualEntryPollStatus, STATUS_POLL_MS);
  } catch (err) {
    box.innerHTML = entryNote(`주문 실패: ${err && err.message ? err.message : err}`, "bad");
    manualOrderBusy = false;
    manualButtonsDisabled(false);
  }
}

// 2026-09-16 애플식 슬라이더(사용자 요청). 트랙을 직접 칠하면 브라우저의 accent-color
// 자동 채움이 사라지므로, 채움 지점을 --fill 로 넘겨 트랙 그라디언트가 읽게 한다.
// ⭐위임 한 줄이면 슬라이더가 몇 개든 따라온다 -- 핸들러를 슬라이더마다 늘리지 않는다.
function syncRangeFill(input) {
  if (!input) return;
  const min = Number(input.min) || 0, max = Number(input.max);
  const v = Number(input.value);
  input.style.setProperty("--fill", `${max > min ? ((v - min) / (max - min)) * 100 : 0}%`);
}
document.addEventListener("input", (e) => {
  const t = e.target;
  if (t && t.tagName === "INPUT" && t.type === "range") syncRangeFill(t);
});

el("snapLevAuto")?.addEventListener("change", () => manualEntryRefreshSize());
el("snapLevGauge")?.addEventListener("input", () => {
  const out = el("snapLevVal");
  if (out) out.textContent = `${manualLevValue()}배 (수동)`;
  // 2026-09-20 레버리지도 **크기를 바꾼다** -- 진입 비율과 똑같이 다시 물어서 위 계좌 카드가
  // 따라 움직이게 한다. 예전엔 라벨만 고쳐서, 게이지를 밀어도 미리보기가 옛 값에 굳어 있었다.
  if (manualEntryPending?.kind === "entry") manualEntryClearConfirm();
  clearTimeout(entrySizeDebounce);
  entrySizeDebounce = setTimeout(manualEntryRefreshSize, 250);
});
// input 이벤트 없이 value 가 바뀌는 경로(렌더·모달 열기)를 위한 동기화.
function syncAllRangeFills(root) {
  (root || document).querySelectorAll('input[type="range"]').forEach(syncRangeFill);
}
// ── 길게 누르기로 진입 (2026-09-20 사용자 선택: 시안 B) ─────────────────────────────
// 누른 순간 **미리보기부터 보낸다** -- 0.4초 동안 아래 상자가 «이만큼 나갑니다»로 채워지고,
// 그걸 보고 손을 떼면 주문은 안 나간다. 링이 다 차면 그 미리보기로 바로 발주한다.
// 🔴확인 클릭을 없앤 대신 «의도적 지속»을 받는다. 오클릭 한 번으로는 아무 일도 없다.
// pointer 이벤트라 마우스·터치가 한 경로다. touch-action: none 은 styles.css 에서 준다
// (없으면 모바일에서 누른 채 스크롤하다 발주된다).
const HOLD_FIRE_MS = 400;
let manualHoldFire = false;
let manualHoldTimer = null;
let manualHoldRaf = null;

function manualHoldPaint(btn, ratio) {
  const fill = btn?.querySelector(".hold-fill");
  if (fill) fill.style.width = `${Math.round(100 * ratio)}%`;
}
// 2026-09-20 «0.4초 누르고 있으면…» 안내문을 뺐다(아티팩트 댓글). 동작은 그대로다 --
// 누르는 동안 차오르는 채움 막대(.hold-fill)가 진행을 계속 보여준다. 문구를 지우면서
// 그것만 쓰던 헬퍼 둘(manualHoldHint/manualHoldIdle)도 같이 지웠다.

function manualHoldCancel(btn) {
  if (manualHoldTimer) { clearTimeout(manualHoldTimer); manualHoldTimer = null; }
  if (manualHoldRaf) { cancelAnimationFrame(manualHoldRaf); manualHoldRaf = null; }
  manualHoldPaint(btn, 0);
  if (manualHoldFire) { manualHoldFire = false; manualEntryClearConfirm(); }
}
function manualHoldStart(btn, side, kind) {
  if (manualOrderBusy || btn.disabled) return;
  manualHoldCancel(btn);
  manualHoldFire = true;
  // 청산은 **계좌부터 새로 읽는다**(manualExitPreview) -- 닫으려는 수량이 낡으면 안 된다.
  if (kind === "exit") manualExitPreview(side); else manualEntryPreview(side, "entry");
  const t0 = performance.now();
  const step = () => {
    const r = Math.min(1, (performance.now() - t0) / HOLD_FIRE_MS);
    manualHoldPaint(btn, r);
    if (r < 1) manualHoldRaf = requestAnimationFrame(step);
  };
  manualHoldRaf = requestAnimationFrame(step);
  manualHoldTimer = setTimeout(() => {
    manualHoldTimer = null;
    manualHoldPaint(btn, 0);
    manualHoldFire = false;
    // 미리보기가 막혔거나(blocked) 아직 안 왔으면 pending 이 없다 -- 그때는 안 나간다.
    if (manualEntryPending) manualEntrySubmit();
    else {
      const box = el("snapEntryResult");
      if (box) { box.hidden = false; box.innerHTML = entryNote("미리보기가 아직 안 왔습니다 — 다시 누르세요.", "bad"); }
    }
  }, HOLD_FIRE_MS);
}
[["snapEntryLong", "LONG", "entry"], ["snapEntryShort", "SHORT", "entry"],
 ["snapExitLong", "LONG", "exit"], ["snapExitShort", "SHORT", "exit"]].forEach(([id, side, kind]) => {
  const btn = el(id);
  if (!btn) return;
  btn.addEventListener("pointerdown", (e) => { e.preventDefault(); manualHoldStart(btn, side, kind); });
  ["pointerup", "pointerleave", "pointercancel"].forEach((ev) =>
    btn.addEventListener(ev, () => manualHoldCancel(btn)));
});
// 2026-09-14 사용자 요청: **청산은 강제 조회부터**. 화면 숫자가 30초(조회가 끊겼으면 그
// 이상) 묵어 있을 수 있어서, 미리보기를 그리기 전에 계좌를 다시 받아 카드·아래 줄을 맞춘다.
// 서버의 청산 미리보기도 같은 이유로 fresh 다(server.py api_manual_exit_preview).
// 조회가 실패해도 미리보기는 진행한다 -- 급히 닫으려는 사람을 여기서 세우면 안 된다.
async function manualExitPreview(side) {
  manualButtonsDisabled(true);
  try {
    binanceAccountLastFetchAt = 0;      // 클라이언트 30초 게이트 우회
    await refreshBinanceAccount();
    manualExitSyncButtons();
  } catch (err) {
    /* 무시 -- 아래 미리보기가 서버에서 다시 읽는다 */
  }
  return manualEntryPreview(side, "exit");
}
el("snapEntryConfirm")?.addEventListener("click", manualEntrySubmit);
// 비율을 바꾸면 화면에 떠 있던 확인 버튼은 **다른 계획**의 것이다. 지운다.
// 계좌 강제 조회. refreshBinanceAccount 의 30초 자체 게이트를 넘겨야 하므로 시각을 지운다.
el("snapAcctRefresh")?.addEventListener("click", async () => {
  const btn = el("snapAcctRefresh");
  if (!btn || btn.disabled) return;
  btn.disabled = true; btn.classList.add("spin");
  try {
    binanceAccountLastFetchAt = 0;
    await refreshBinanceAccount();
    manualExitSyncButtons();      // 포지션이 바뀌었으면 버튼도 바로 맞춘다
    manualEntryRefreshSize();
  } finally {
    btn.classList.remove("spin");
    setTimeout(() => { btn.disabled = false; }, 3000);   // 연타로 거래소 한도를 때리지 않게
  }
});

el("snapHold")?.addEventListener("change", () => {
  manualEntryClearConfirm();          // 보유시간이 바뀌면 크기가 바뀐다 -- 다른 계획이다
  manualEntryRefreshSize();
});

// 2026-09-20 시안 B: 슬라이더를 칩으로 갈았다(사용자 선택). 🔴입력 자체는 **지우지 않고
//   숨겨 둔다** -- 칩은 그 값을 써 넣고 input 이벤트를 쏘기만 한다. 그래서 값을 읽는 쪽
//   (sliderPct·manualLevValue)과 아래 input 리스너들을 한 줄도 안 고쳤고, 범위 클램프도
//   브라우저가 계속 해준다(레버 상한이 서버 정책으로 20 밑이면 20x 칩은 그 상한으로 눌린다 --
//   그때는 어느 칩도 안 켜지고 옆 숫자가 진짜 값을 말한다).
function syncChipset(box) {
  const inp = el(box.dataset.for);
  if (!inp) return;
  // 🔴칩은 입력의 **상태까지** 따라가야 한다. renderLevGauge 는 물타기면 게이지를 숨기고
  //   (기존 레버리지가 ×3 처럼 5의 배수가 아닐 수 있어 5단위로는 나타낼 수도 없다) 자동이면
  //   비활성화한다 -- 「고를 수 없는 것을 고를 수 있는 것처럼 보여주면 안 된다」는 그 함수의
  //   주석 그대로다. 이걸 안 따라가면 칩이 «5x 로 나간다»고 적극적으로 거짓말한다
  //   (2026-09-20 미리보기에서 실제로 그랬다: 칩 5x · 꼬리표 「20배 (기존 포지션과 동일)」).
  box.hidden = !!inp.hidden;
  box.classList.toggle("off", !!inp.disabled);
  let hit = false;
  box.querySelectorAll(".chip").forEach((c) => {
    const on = Number(c.dataset.v) === Number(inp.value);
    hit = hit || on;
    c.classList.toggle("on", on);
  });
  // 칩이 맞으면 옆 숫자를 숨긴다(같은 값을 두 번 말하지 않는다). 칩에 없는 값 -- 서버가
  // 추천한 레버리지 15x 같은 -- 일 때만 숫자가 나타나 진짜 값을 말한다.
  box.dataset.matched = hit ? "1" : "0";
}
function syncChipsets() { document.querySelectorAll(".chipset").forEach(syncChipset); }
document.querySelectorAll(".chipset").forEach((box) => {
  const inp = el(box.dataset.for);
  if (!inp) return;
  box.addEventListener("click", (e) => {
    const c = e.target.closest(".chip");
    if (!c) return;
    inp.value = c.dataset.v;
    inp.dispatchEvent(new Event("input", { bubbles: true }));
    syncChipset(box);
  });
  inp.addEventListener("input", () => syncChipset(box));
  syncChipset(box);
});

let entrySizeDebounce = null;
for (const [slider, label, kind] of [["snapExitFrac", "snapExitFracVal", "exit"],
                                     ["snapEntryFrac", "snapEntryFracVal", "entry"]]) {
  el(slider)?.addEventListener("input", () => {
    const lab = el(label);
    if (lab) lab.textContent = `${sliderPct(slider)}%`;
    if (manualEntryPending?.kind === kind) manualEntryClearConfirm();
    // 2026-09-14 비율을 움직이면 «그래서 얼마»가 따라와야 한다. 청산은 계좌 payload 만으로
    // 되므로 화면이 바로 계산하고, 진입 수량은 **서버가 정하므로**(상한·틱·최소수량) 다시
    // 묻는다 -- 슬라이더가 멈춘 뒤에. 손가락 한 번에 요청 열 번을 보내지 않는다.
    if (kind === "exit") renderExitNow();
    else {
      clearTimeout(entrySizeDebounce);
      entrySizeDebounce = setTimeout(manualEntryRefreshSize, 250);
    }
  });
}

// 포지션이 열린 측면의 청산 버튼만 띄운다. latestBinanceAccount 는 계좌 패널이 이미
// 주기적으로 받아 두는 값이라 여기서 따로 요청하지 않는다(없으면 그냥 숨긴 채 둔다).
let entryFoldHadPos = null;
let entryFoldToggleBound = false;
// 슬라이더를 움직일 때마다 계좌를 다시 받지 않으려고 마지막 포지션을 들고 있는다.
let lastExitPositions = new Map();
let lastExitStaleMin = 0;

// peg 청산의 **실측** 왕복비용(2026-09-13 섀도우 23,332legs). 이 계좌 수수료는 메이커 2.0bp ·
// 테이커 5.0bp 이고, peg 는 vol 5분위 전 구간에서 3.03~3.09bp 로 평평했다(체결률 99.2%).
// 시장가로 넘어가는 고변동 구간(EXIT_TAKER_VOL_BPM=30)에서는 테이커 5.0bp 다.
// ponytail: 화면이 vol 을 모르므로 항상 peg 값을 쓴다 -- 미리보기 카드는 plan.type 을 알아서
// 시장가면 5.0 으로 바꿔 계산한다. 수수료 우대는 가정하지 않는다(표준 요율).
const EXIT_FEE_BP_PEG = 3.0;
const EXIT_FEE_BP_TAKER = 5.0;
// 돈은 센트까지만 쓴다. fmtUsd 는 10달러 미만을 소수 4자리로 그리는데(코인 수량용), 수수료가
// «≈$2.0831» 로 찍히면 정밀해 보일 뿐 읽기 나쁘다. 부호는 항상 붙인다 -- 색만으로는 부족하다.
const usd2 = (v) => `${v < 0 ? "-" : "+"}$${Math.abs(Number(v) || 0).toFixed(2)}`;

// 「지금 닫으면 얼마인가」를 미리보기 **전에** 그린다. 손익은 거래소가 준 unrealized_pnl 에
// 비율을 곱한 것이고(우리가 VWAP 을 다시 계산하지 않는다), 체결가는 모르므로 ≈ 다.
function renderExitNow() {
  const box = el("snapExitNow");
  if (!box) return;
  const pct = manualExitPct();
  const rows = [];
  for (const pos of lastExitPositions.values()) {
    const qty = Number(pos.qty) || 0;
    if (!(qty > 0)) continue;
    const entry = Number(pos.entry_price) || 0;
    const mark = Number(pos.mark_price) || 0;
    const dir = pos.side === "LONG" ? 1 : -1;
    const close = qty * pct / 100;
    const move = entry > 0 && mark > 0 ? (mark - entry) / entry * 100 * dir : null;
    // 거래소 값이 없을 때만 가격으로 되짚는다 -- 있으면 그게 진실이다.
    const full = Number.isFinite(Number(pos.unrealized_pnl)) ? Number(pos.unrealized_pnl)
      : (entry > 0 && mark > 0 ? qty * (mark - entry) * dir : null);
    const gross = full === null ? null : full * pct / 100;
    // 청산 수수료만 뺀다. 진입 수수료는 **이미 지갑에서 빠져 나갔으므로**, 여기 숫자가
    // «지금 닫으면 지갑이 얼마 늘어나는가»와 일치하려면 빼면 안 된다(title 에 적어 둔다).
    const fee = mark > 0 ? close * mark * EXIT_FEE_BP_PEG / 10000 : 0;
    const net = gross === null ? null : gross - fee;
    const head = `${pos.side === "LONG" ? "롱" : "숏"} `
      + (pct >= 100 ? `전량 <b>${qty.toFixed(3)}</b>`
                    : `≈<b>${close.toFixed(3)}</b> 닫고 <b>${(qty - close).toFixed(3)}</b> 남김`);
    if (net === null) { rows.push(head); continue; }
    rows.push(`${head} · 지금 닫으면 `
      + `<b class="exit-net ${net >= 0 ? "good" : "bad"}" title="${escapeHtml(
          `미실현 ${usd2(gross)} − 청산 수수료 ≈$${fee.toFixed(2)} (peg 실측 ${EXIT_FEE_BP_PEG}bp)\n`
          + "진입 수수료는 이미 지갑에서 빠졌으므로 여기서 다시 빼지 않습니다.\n"
          + "체결가·고변동 시장가 전환에 따라 달라질 수 있는 추정치입니다.")}">`
      + `${usd2(net)}</b>`
      + `<span class="entry-was"> 순손익 · 미실현 ${usd2(gross)}`
      + ` − 수수료 ≈$${fee.toFixed(2)}`
      + (move === null ? "" : ` · ${move >= 0 ? "+" : ""}${move.toFixed(2)}%`) + `</span>`);
  }
  // 낡음 경고는 버튼 라벨이 아니라 여기 적는다 -- 라벨에 넣으면 「롱 2.754 닫기」가 길어져
  // 모바일에서 줄바꿈되고, 정작 «무엇이 낡았는지»는 안 보인다.
  const warn = lastExitStaleMin > 0
    ? `<span class="bad">⚠조회가 ${lastExitStaleMin}분째 낡음 — ⟳ 로 갱신 (아래는 그때 기준)</span>`
    : "";
  box.innerHTML = [warn, ...rows].filter(Boolean).join("<br>");
}

// 2026-09-20 주문 모달을 없앴다(여는 함수 둘 포함). 진입·청산이 둘 다 카드 안으로 나오면서
// 열 모달도, 여는 버튼(#tradeOpenEntry / #tradeOpenExit)도 사라졌다. 닫을 때 «확인» 상태를
// 지우던 dialog close 리스너의 역할은 manualHoldCancel 이 대신한다(손을 떼면 지운다).
// styles.css 의 .trade-modal* / .trade-actions 규칙도 2026-09-20 에 함께 걷어냈다.
function manualExitSyncButtons() {
  const row = el("snapExitRow");
  if (!row) return;
  // 서버의 청산 경로는 ETH 전용이다(assemble_exit_plan 이 MARKET_SYMBOLS["eth"] 고정).
  // 🔴2026-09-20 «청산 버튼이 사라졌다»(사용자). 이 줄이 ETHUSDT 만 봤는데 수동 주문은
  //   2026-09-19 부터 **ETHUSDC** 로 나간다 -- 그래서 ETHUSDC SHORT 1.659 를 들고 있는데도
  //   hasPos 가 false 였다. 포지션을 **들고 있을 때** 못 닫는 게 이 버튼의 최악이다.
  //   서버(live_manual_peg_entry resolve_exit_position)와 같은 규칙을 쓴다: 수동 심볼에
  //   포지션이 있으면 그것, 없으면 시장 심볼. 심볼은 payload 의 exec_symbol 이 말한다
  //   (환경변수라 하드코딩하면 또 갈라진다). snapshotAccountPosition 은 이미 이 규칙이다.
  const market = ASSET_CONFIG.eth?.symbol || "ETHUSDT";
  const execSym = (latestBinanceAccount || lastGoodAccount)?.exec_symbol || "";
  const base = market.replace(/USDT$/, "");
  const cand = execSym && execSym !== market && execSym.startsWith(base)
    ? [execSym, market] : [market];
  // 🔴낡았다고 **버튼을 치우지 않는다**(2026-09-14). 옛 판은 5분이 지나면 행을 통째로 숨겼다 --
  //   급히 닫으려고 연 사람에게서 버튼이 사라지는 건 이 버튼의 존재 이유와 정면으로 어긋난다.
  //   이미 닫힌 포지션을 눌러도 서버가 no_position 으로 막으므로 대가는 헛클릭 한 번이고,
  //   숨김의 대가는 «못 닫음»이다. 대신 얼마나 낡았는지를 행 아래에 크게 적는다.
  const src = latestBinanceAccount || lastGoodAccount;
  const stale = !latestBinanceAccount && !!src;
  lastExitStaleMin = stale ? Math.max(1, Math.round((Date.now() - lastGoodAccountAt) / 60000)) : 0;
  // 측면만이 아니라 **포지션 전체**를 들고 간다 -- 수량·진입가·마크가로 아래 줄(renderExitNow)이
  // 「≈얼마 닫고 얼마 남김 · 지금 닫으면 얼마」를 그린다.
  // 2026-09-14 버튼 라벨은 「롱 청산 / 숏 청산」 고정이다(사용자 결정). 한때 수량을 박았는데
  // (「롱 2.754 닫기」) 수량은 바로 아래 줄에 이미 있고, 라벨이 안 변하면 여기서 textContent 를
  // 쓸 이유도 없다 -- index.html 의 글자가 그대로 남는다.
  // 한 심볼만 고른다 -- 둘을 섞으면 같은 side 가 겹쳐 아래 줄이 «어느 쪽 수량»인지 흐려진다.
  const sym = cand.find((c) => (src?.positions || [])
    .some((p) => p.symbol === c && Number(p.qty) > 0)) || cand[0];
  lastExitPositions = new Map((src?.positions || [])
    .filter((p) => p.symbol === sym && Number(p.qty) > 0)
    .map((p) => [p.side, p]));
  const bl = el("snapExitLong");
  if (bl) bl.hidden = !lastExitPositions.has("LONG");
  const bs = el("snapExitShort");
  if (bs) bs.hidden = !lastExitPositions.has("SHORT");
  const hasPos = lastExitPositions.size > 0;
  row.hidden = !hasPos;
  renderExitNow();
  // 진입 블록은 포지션이 있으면 접는다. 포지션 유무가 **바뀔 때만** 건드린다 -- 매 갱신마다
  // 쓰면 사람이 물타기를 보려고 펼쳐 둔 걸 30초마다 도로 닫는다.
  const box = el("snapEntryBox");
  // 🔴리스너를 **open 을 건드리기 전에** 건다. `toggle` 은 비동기로 발화하지만 프로그램이
  //   접는 것도 발화시키므로, 뒤에 걸면 첫 접힘 한 번을 놓쳐 대가 줄이 30초 늦게 뜬다.
  if (box && !entryFoldToggleBound) {
    entryFoldToggleBound = true;
    box.addEventListener("toggle", () => {
      const n = el("snapEntryFoldNote");
      // 펼치면 아래에 카드가 그대로 보인다 -- 같은 숫자를 두 번 쓰지 않는다.
      if (box.open) { if (n) { n.textContent = ""; n.className = "entry-was"; } }
      else { setEntryProjPreview(null); }   // 접으면 계좌 카드는 실제 값으로 돌아온다
      manualEntryRefreshSize();
    });
  }
  if (box && entryFoldHadPos !== hasPos) { box.open = !hasPos; entryFoldHadPos = hasPos; }
  const sum = el("snapEntrySummary");
  // 🔴`sum.textContent` 로 쓰면 안 된다 -- 요약 줄 안의 대가 칸(span)까지 지운다.
  const lab = el("snapEntryFoldLabel");
  if (lab) lab.textContent = hasPos ? "추가 진입 (물타기)" : "진입";
}
if (el("snapEntryPlan")) {
  // 트랙 채움은 input 이벤트에서만 갱신된다. 게이지가 이제 **항상 보이므로** 시작할 때
  // 한 번 칠해 준다(옛 판은 모달을 열 때 했고, 그 모달이 없어졌다).
  syncAllRangeFills();
  manualEntryRefreshSize();
  manualExitSyncButtons();
  setInterval(() => { manualEntryRefreshSize(); manualExitSyncButtons(); },
              MANUAL_ENTRY_REFRESH_MS);
}
