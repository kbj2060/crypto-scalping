const API_EVENTS_URL = "/api/events";
const API_OPS_STATUS_URL = "/api/ops-status";
const API_BINANCE_ACCOUNT_URL = "/api/binance-account";
const API_LIQUIDATION_MAP_URL = "/api/liquidation-map";
const API_REGIME_WIDE24_URL = "/api/regime-wide24";
const API_REGIME_BTC_URL = "/api/regime-btc";
const API_REGIME_XRP_URL = "/api/regime-xrp";
const API_MACRO_CALENDAR_URL = "/api/macro-calendar";
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
// 2026-09-26 dp = 화면 가격 소수 자릿수, qtyScale = 수량 눈금 배율(ETH=1 기준, 대략 ETH가격/코인가격).
//   🔴데이터 심볼은 전부 USDT 시장이다(주문만 USDC -- 사용자 규칙).
const ASSET_CONFIG = {
  eth: { label: "ETH", symbol: "ETHUSDT", dp: 1, qtyScale: 1 },
  sol: { label: "SOL", symbol: "SOLUSDT", dp: 2, qtyScale: 20 },
  btc: { label: "BTC", symbol: "BTCUSDT", dp: 0, qtyScale: 0.03 },
  xrp: { label: "XRP", symbol: "XRPUSDT", dp: 4, qtyScale: 2000 },
};
// 풋프린트·1초 수급·OI·호가 히트맵을 서버가 실시간으로 만드는 코인(server.py FLOW_SPECS 와 같은 목록).
const FLOW_ASSETS = new Set(["eth", "sol", "xrp"]);
const flowOn = () => FLOW_ASSETS.has(activeSnapshotAsset);
const coinUnit = () => (ASSET_CONFIG[activeSnapshotAsset] || {}).label || String(activeSnapshotAsset).toUpperCase();
const pxDp = () => { const d = (ASSET_CONFIG[activeSnapshotAsset] || {}).dp; return Number.isFinite(d) ? d : 1; };
const qtyScale = () => (ASSET_CONFIG[activeSnapshotAsset] || {}).qtyScale || 1;

const el = (id) => document.getElementById(id);
// 2026-09-30 한 화면 모드 크기 기억(사용자 «새로고침마다 흩어졌다가 맞춰진다»): 마지막으로 맞춘 차트·사다리·시장 맥락 높이를 창 높이와 함께 저장하고,
//   같은 창 높이로 다시 열면 **첫 그림부터** 그 크기로 그린다(fitLayout · renderMarketCtx 가 갱신).
const FIT0 = (() => { try { const c = JSON.parse(localStorage.getItem("fitCache") || "{}"); return c && c.vh === innerHeight ? c : {}; } catch (e) { return {}; } })();
const setT = (id, txt) => {
  const target = el(id);
  if (!target || target.textContent === String(txt)) return;
  target.textContent = txt;
};
// 🔴같은 html 을 다시 넣지 않는다(2026-09-16). 현재가 틱마다 청산맵 목록·배지가 통째로
//   다시 쓰였고, 그 줄은 진입/청산 버튼 바로 아래 **유리 헤더**라 재래스터가 눈에 띈다.
//   innerHTML 재대입은 같은 문자열이어도 자식을 전부 버리고 다시 파싱한다.
const setH = (id, html) => { const target = el(id); if (target && target.innerHTML !== html) target.innerHTML = html; };

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
// 2026-10-01 창 이동(사용자 «1h·2h·4h·12h 를 창으로 두고 차트에서 이동»): 데이터는 늘 12시간(144봉)을 받고 화면은 창만큼만 그린다.
//   chartPanEnd = 창 오른쪽 끝 봉의 시각(초) · null = 실시간(최신 봉을 따라간다). 과거로 옮기면 그 시각에 고정된다.
const CHART_PAN_BARS = 144;
let chartPanEnd = null;
let footprintLastFetchAt = 0;
// ── 창 토글 (2026-09-19 사용자 요청: 1h/2h/4h) ─────────────────────────────
// 풋프린트와 수급 프로파일이 **같은 창**을 쓴다. 한 카드 안의 위아래 두 그림이 서로 다른
// 구간을 말하면 읽는 사람이 속는다 -- 그래서 토글도 하나다.
// 서버 링은 24시간(288봉)이라 4h 도 이미 쌓여 있는 데이터다. 더 긴 창을 안 주는 건 값이
// 없어서가 아니라, 48봉이면 풋프린트 셀이 25px 라 숫자가 이미 안 들어가서다.
const CHART_WINDOW_BARS = [12, 24, 48, 144];   // 5분봉 기준 1h · 2h · 4h · 12h
// 🔴12h(144봉)에서는 봉이 ~9px 라 셀 숫자가 안 들어간다 -- showQty(half >= 18)가 꺼져
//   색 농담만 남는 «시간축 볼륨 프로파일»이 된다. 4h 에서도 이미 그렇다(봉 26px).
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
// ── 수급 · 최근 5분 × 1초 (2026-09-19) ────────────────────────────────────
// 서버가 초 단위 칸을 들고 있고, 여기선 **증분만** 받는다(?since=). 300초를 매초 전부 받으면
// 시간당 60MB 가 넘는다 -- 증분이면 보통 한두 줄이다.
// ⚠️서버는 진행 중인 초를 목록(커서가 움직이는 쪽)에 안 넣는다 -- 반쪽이 굳는 걸 막으려고.
// ⭐2026-09-24: 대신 `partial` 로 따로 받아 **커서를 안 옮기고** 칸만 채운다. 초가 닫히면 확정본이
//   목록으로 와서 같은 칸을 덮는다. 폴링도 0.25초로 -- 오른쪽 끝 지연이 1~2초에서 ~0.1초대로.
//   (증분이라 회당 보통 수백 B. 틱(0.5초)과 따로 도는 타이머를 쓴다.)
const API_SUPPLY_1S_URL = "/api/supply-1s";
const SUPPLY_1S_POLL_MS = 250;
const SUPPLY_1S_SEGMENT = 300;          // 누적을 0으로 되돌리는 **벽시계** 경계(초). 5분봉과 같은 자리.
let supply1s = new Map();               // 초 -> [리테일매수, 리테일매도, 고래매수, 고래매도, 총매수, 총매도, 가격]
let supply1sMeta = { retailMaxUsd: 0, whaleMinUsd: 0, now: 0 };
// 🔴`since` 는 «내가 **받은** 마지막 초»여야 한다. 서버가 보낸 `now`(진행 중인 초)를 그대로
//   돌려주면 그 초는 영영 안 온다 -- 1차엔 「진행 중」이라 빠지고, 2차엔 「since 이하」라
//   빠진다. 그러면 첫 응답 뒤 매 폴링이 빈 응답이 되어 **차트가 멈춘다**(2026-09-19 첫
//   렌더에서 오른쪽에 공백 띠가 계속 자라는 걸로 드러났다).
let supply1sSince = 0;
let supply1sLastFetchAt = 0;
let supply1sInFlight = false;   // 0.25초 타이머라 응답이 늦으면 겹친다 -- 겹치면 커서가 꼬인다
// 초 -> 그 초의 미결제약정(ETH). 같은 응답에 얹혀 온다(별도 폴링을 하나 더 두지 않는다).
// ⚠️바이낸스가 OI 를 3~7초에 한 번만 갱신한다(서버 OI_1S_URL 주석의 실측) -- 점이 초마다
//   있지 않은 게 정상이다. 창 시작을 0으로 두고 «그 뒤로 몇 계약이 새로 생겼나»를 그린다.
let oi1s = new Map();
let oi1sSince = 0;
// 2026-09-22 초 단위 청산 [롱수량, 숏수량]. oi1s 와 같은 증분 규약(커서는 따로 -- 청산은
// 이벤트가 없는 초가 많아 체결 초 커서를 공유하면 건너뛰어진다).
let liq1s = new Map();
let liq1sSince = 0;
// ── OKX 레인 (2026-09-23) ─────────────────────────────────────────────────
// 🔴**합치지 않는다.** 바이낸스 레인 **바로 아래**에 같은 그림을 따로 그려 눈으로 대조한다.
//   합산 크기는 MM 헤지 이중계상으로 |합산|/|바이낸스| 중앙 **2.1배**로 부푼다(3일 백필 실측)
//   -- 부호는 합쳐도 되지만 크기는 못 합친다. 나란히 두면 ctVal 같은 체계적 버그가
//   «한 레인만 10배»로 즉시 보인다는 게 이 배치의 값이다.
// ⭐칸 구조가 바이낸스와 **정확히 같다**(서버가 같은 모양으로 채운다) -- 같은 렌더러에
//   출처만 갈아끼운다. 사본을 만들면 언젠가 한쪽만 고쳐진다.
let okxSupply1s = new Map();
let okxOi1s = new Map();
let okxLiq1s = new Map();
let okxSupply1sSince = 0, okxOi1sSince = 0, okxLiq1sSince = 0;
let okxMeta = { now: 0, connected: false, tradeAge: null, oiAge: null, inst: "", errors: 0 };
let spotSupply1s = new Map();     // 바이낸스 **현물**. OI·청산 칸은 없다(현물엔 존재하지 않는다).
let spotSupply1sSince = 0;
let spotMeta = { now: 0, connected: false, tradeAge: null, errors: 0 };
// 2026-09-23 패널을 **하나로 합쳤다**(사용자 지시) -- 두 패널의 눈금을 맞추던 supply1sPeaks 는
// 더 볼 짝이 없어 지웠다. 합산 하나가 눈금을 정하고 거래소별 얇은 선이 그 안에 들어간다.
// 5분 누적 패널(청산맵 아래). 같은 1초 스냅샷을 duckdb 로 남긴 것을 서버가 5분으로 접어 준다 --
// 링은 6분뿐이라 몇 시간을 보려면 저장을 거쳐야 한다. 5분 봉이라 15초 폴링으로 충분하다.
// 캔들 SVG 안의 두 하위 패널(중첩 svg)과 그 상자. 각 fetch 가 **그 패널만** 다시 그릴 수
// 있게 들고 있는다. 없으면 두 그림의 갱신이 캔들 SVG 전체 렌더에 묶이는데, 그 렌더에는
// 게이트가 셋이다 -- ①커서가 SVG 위에 있으면 아예 안 그린다(chartHoverActive, 툴팁이
// 지워지지 않게 하려는 장치) ②스크롤 중 정지 ③400~1000ms 스로틀. 패널이 그 SVG 안으로
// 들어오면서(2026-09-19) 「보려고 커서를 올리면 1초 차트가 멈춘다」가 됐다.
// 노드는 캔들 렌더가 다시 붙여 주므로(subPanelCache) isConnected 로 옛 노드를 거른다.
let supply1sSubBox = null;
// ⭐두 패널을 캔들 렌더와 **분리**한다(2026-09-20). 캔들 SVG 는 풋프린트 모드에서 초당 2.5번
//   통째로 다시 그려지는데, 이 두 그림의 입력은 0.2~1Hz 로만 바뀐다. 실측(헤드리스 크로미움,
//   실제 페이로드): renderSnapshotChart 6.1ms 중 **3.9ms(64%)가 이 두 패널**이었고, 그 위에
//   자기 폴링(1초 히트맵 · 1초 수급 · 5초 프로파일)이 또 겹쳐 프로파일은 초당 3.7번 그려졌다.
//   그래서 «판번호»가 바뀌었을 때만 다시 그리고, 아니면 만들어 둔 <svg> 노드를 그대로 다시
//   붙인다(innerHTML="" 은 DOM 에서 떼어낼 뿐 JS 참조가 쥔 서브트리는 살아 있다).
let supply1sVer = 0;
const subPanelCache = { s1: { node: null, key: "" }, dens: { node: null, key: "" } };
const sub1sKey = (w, h) => `${supply1sVer}|${w}|${h}`;

// ── 합산 출처 (2026-09-23) ───────────────────────────────────────────────
// 🔴**무엇을 더하고 무엇을 안 더하는지가 전부 측정에서 나왔다:**
//   · 수급(CVD/스택) = 선물 + OKX. 🔴**2026-09-25 현물을 본선에서 뺐다**(사용자 지시).
//     현물이 틀려서가 아니라 **풋프린트·사분면과 원천을 맞추기 위해서다** -- 현물은 선물보다
//     +4.69bp 높아 가격축에 못 올라가므로($0.1 빈 12.9칸) 풋프린트엔 원리적으로 못 들어간다.
//     셋 중 «한 곳에만» 들어가는 유일한 원천이라 여기가 유일한 제거 지점이었다.
//     값도 가장 싸다: 라이브 647초에서 거래량 몫 7.7% · 1초 델타 부호 뒤집기 3.4%(OKX 12.7%) ·
//     1분 델타 부호 뒤집기 0/12. 3일 측정에서도 기여 +7/+9pp 로 OKX(+23/+41pp)의 1/4 이하다.
//     ⭐**불균형 %(imbSources)에는 그대로 남긴다** -- 거기엔 가격축이 없어 문제가 없고,
//     09-22 에 확인된 기여가 나오는 자리다.
//   · OI = 2026-09-24 사용자 지시로 **더한다**(5분 레인과 같게). ⚠️ΔOI 상관 **+0.000** · 부호
//     일치 48.0%(1시간)라 합은 상쇄될 수 있다(실측: 바이낸스 +228 / OKX -137 → 합 +91) --
//     🔴2026-09-25 정정: 그 «+0.000» 은 **5분봉 2일(593봉)에서 +0.503** 으로 재현되지 않는다.
//     1초에서 0 이 나오는 건 바이낸스 OI 가 REST 3~7초 지연이라서로 추정한다(확정 아님).
//     그래서 거래소별 선을 얇게 남겨 «반대로 갔다»가 안 지워지게 한다(CVD 의 얇은 선과 같은 문법).
//     현물엔 OI 가 **아예 없다**(포지션 개념이 없다).
//   · 청산 = 더한다. 이벤트라 점을 다 찍으면 되고 손실이 없다. 현물엔 강제청산이 없다.
//   · 크기 주의: |합산|/|바이낸스| 중앙 **2.1배**(MM 헤지 이중계상). 라벨에 적는다.
function mergedSupplySrc() {
  const merged = new Map();
  const addSupply = (m) => m.forEach((c, sec) => {
    let t = merged.get(sec);
    if (!t) { t = [0, 0, 0, 0, 0, 0, 0]; merged.set(sec, t); }
    for (let i = 0; i < 6; i++) t[i] += c[i] || 0;
    if (c[6]) t[6] = c[6];
  });
  // 🔴현물은 여기 안 더한다(위 주석) -- 풋프린트·사분면과 같은 원천이어야 셋이 맞는다.
  addSupply(supply1s); addSupply(okxSupply1s);
  const liq = new Map();
  [liq1s, okxLiq1s].forEach((m) => m.forEach((c, sec) => {
    let t = liq.get(sec);
    if (!t) { t = [0, 0, 0, 0]; liq.set(sec, t); }
    for (let i = 0; i < 4; i++) t[i] += c[i] || 0;
  }));
  // 🔴now·dead 는 **합에 들어가는 원천만** 본다(2026-09-25 현물이 합에서 빠지면서 같이 좁혔다).
  //   현물이 죽어도 본선은 멀쩡하므로 «합에서 빠짐»에 적으면 거짓말이다 -- 나이는 아래 age 에 남긴다.
  const now = Math.max(supply1sMeta.now || 0, okxMeta.now || 0);
  // 🔴선물 나이는 «가장 최근 거래소 대비» 로 잰다(예전엔 늘 0 이라 선물이 죽어도 OKX 가
  //   now 를 밀어 합산이 절반짜리로 멀쩡해 보였다). 🔴한 번도 안 붙은 거래소(null)도 죽은 것이다
  //   -- 빠진 거래소는 합에서 0 으로 들어가므로 이름을 화면에 적는다.
  const ages = [["바이낸스", supply1sMeta.now ? now - supply1sMeta.now : null],
                ["OKX", okxMeta.tradeAge]];
  const dead = ages.filter(([, a]) => a == null || a > 10).map(([n]) => n);
  // OI 합: 두 거래소는 갱신 시각이 달라(바이낸스 폴링 · OKX WS) 초마다 **각자의 직전 관측값**을
  //   더한다(계단 채움). 둘 다 한 번은 관측된 뒤부터 -- 한쪽만 있는 앞부분을 합이라 부르면 거짓이다.
  //   OKX 가 아예 없으면 바이낸스만 쓴다(선이 통째로 사라지지 않게).
  const oiSum = new Map();
  let lastBn = null, lastOkx = null;
  [...new Set([...oi1s.keys(), ...okxOi1s.keys()])].sort((a, b) => a - b).forEach((s) => {
    if (oi1s.has(s)) lastBn = oi1s.get(s);
    if (okxOi1s.has(s)) lastOkx = okxOi1s.get(s);
    if (lastBn != null && lastOkx != null) oiSum.set(s, lastBn + lastOkx);
  });
  const oiMain = oiSum.size ? oiSum : oi1s;
  return {
    key: "bn", supply: merged, liq, oi: oiMain,
    now,
    label: "합산 · 바이낸스 선물 · OKX",
    // OI 본선 = 합(굵게). 거래소별은 얇게 눌러 배경 참고선으로 둔다(갈릴 때만 눈에 들어오게).
    oiLanes: [{ oi: oiMain, color: "var(--warn)", width: 2, opacity: 0.95 },
              { oi: oi1s, color: "var(--warn)", width: 1.2, opacity: 0.5 },
              { oi: okxOi1s, color: "var(--warn)", width: 1.2, opacity: 0.5, dash: "3 2" }],
    // 거래소별 CVD 얇은 선 = 본선에 더해진 둘. 현물은 본선에도 얇은 선에도 없고 **불균형 % 로만**
    //   읽는다(2026-09-25). 점유는 4.5% 로 적혀 있었는데 09-25 실측 창에서는 7.7% 였다.
    thin: [{ supply: supply1s, label: "바이낸스" }, { supply: okxSupply1s, label: "OKX" }],
    imbSources: [["바이낸스", supply1s], ["OKX", okxSupply1s], ["현물", spotSupply1s]],
    age: `체결 OKX ${okxMeta.tradeAge == null ? "-" : okxMeta.tradeAge + "s"}`
         + ` · 현물 ${spotMeta.tradeAge == null ? "-" : spotMeta.tradeAge + "s"}`
         + (dead.length ? ` · 합에서 빠짐: ${dead.join("/")}` : ""),
    stale: dead.length > 0,
  };
}

function repaintSupply1sPanel() {
  const b = supply1sSubBox;
  if (!b || !b.svg.isConnected) return;
  renderSupply1s(b);
  subPanelCache.s1.key = sub1sKey(b.w, b.h);
}

const API_OI_5M_URL = "/api/oi-5m";
const OI_5M_POLL_MS = 15000;
// 2026-09-21 상황 읽기 · 30분 -- 2026-09-30 카드는 없애고 차트 머리 칩(#sitChips)·가격판 30분 도달 선으로 옮겼다. 서버가 5초마다 계산해 둔 것을 받는다.
const API_SITUATION_URL = "/api/situation";
const SITUATION_POLL_MS = 1000;   // 09-21 서버 계산도 1초로 -- 응답은 작은 JSON 하나
let latestSituation = null;
let situationLastFetchAt = 0;
// 2026-09-28 30분 카드 «위:아래 방향» 자리 = 일 단위 추세(dashboard/trend_rule.py, 5기간 묶음 7·14·28·56·90일).
//   위:아래는 AUC .525(대부분 51:49 동전)였다. 추세는 ETH 현물 9년 샤프 1.0 대(규칙, 확률은 안 싣는다).
//   일봉이라 하루 한 번 바뀐다 -- 60초 간격이면 충분하고 서버도 5분 캐시다.
const API_TREND_URL = "/api/trend";
const TREND_POLL_MS = 60000;
let latestTrend = null;
let trendLastFetchAt = 0;
let latestOi5m = null;
let oi5mLastFetchAt = 0;
const GEX_POLL_MS = 60000;           // 2026-09-28 체인 10분 · 블록 20초(옵션 카드) -- 1분 폴링
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
// 2026-09-19 호가 히트맵. 래스터는 1초/열인데 agg=3 으로 접어 받으므로 3초면 새 열이 하나다.
let latestGex = null;
let gexLastFetchAt = 0;
// 2026-09-28 칼시 15분 ETH 위/아래 확률(사용자 선택 B · «매 1초»). 서버 /api/kalshi 가 1초 캐시.
const KALSHI_POLL_MS = 1000;
let latestKalshi = null;
let kalshiLastFetchAt = 0;
let latestFlowHeatmap = null;
let flowHeatmapLastFetchAt = 0;
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
// Sudden-liquidation alert (2026-08-27). 2026-09-24 1초 폴링을 걷어냈다 -- 받는 쪽 게이지
// (#liqVolumeGauge)가 HTML 에 없어 분당 55건을 받아 버리고 있었다. 게이지를 되살리면 폴링도 되살린다.
let latestLiqBurstState = null;
// Liquidation map (Snapshot tab, 2026-08-24) -- estimated support/resistance, own fetch/render
// cycle same as latestVRebound above (computed dashboard-side, not part of trading_bot.py state).
// lastSnapshotHistoryFetchAt tracks the candle history this panel's chart needs (activeSnapshotAsset,
// see below), independently of activeChartAsset (the Live tab's own, separate coin selector).
let latestLiquidationMap = null;
let latestRegimeWide24 = null;
let latestRegimeBtc = null;
let latestRegimeXrp = null;
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
let macroCalendarLastFetchAt = 0, macroCalendarOkAt = 0;   // OkAt = 마지막으로 **받은** 시각(⑤ 시간축 아래 «일정 갱신»)
// 2026-09-10 거래소 실계좌. ops 탭 패널과 스냅샷 탭 요약이 같은 payload 를 쓰므로 한 곳에 담는다.
// 서버가 10초 캐시(BINANCE_ACCOUNT_CACHE_SECONDS)라 클라 주기도 같게 맞춘다.
let latestBinanceAccount = null;
// 🔴마지막으로 **성공한** 계좌. latestBinanceAccount 는 실패 시 null 이 되는데, 그걸로
// 청산 버튼을 숨기면 «조회 실패»와 «포지션 없음»이 구분되지 않는다 -- 정작 닫아야 할 때
// 버튼이 사라진다(2026-09-13 사용자 신고). 버튼은 이 값으로 판단하고, 낡았으면 표시만 한다.
let lastGoodAccount = null;
let lastGoodAccountAt = 0;
let binanceAccountLastFetchAt = 0;
const BINANCE_ACCOUNT_POLL_MS = 10000;   // 2026-10-02 30 -> 10초(사용자 지시) -- 서버 캐시(BINANCE_ACCOUNT_CACHE_SECONDS)와 같게
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
// 2026-09-16 300초 -> 60초. 서버 캐시를 60초로 줄였으므로(입력이 1시간봉이라 그 아래로는
// 의미가 없다) 클라가 5분마다 물으면 **새 시간봉이 최대 5분 늦게** 보인다. 캐시와 같은 주기로.
// 🔴2026-09-22 60초 -> 30초. «캐시와 같은 주기»는 최선이 아니라 **최악**이다 -- 둘이 동기화돼
//   있지 않아서, 생성 직후에 물으면 다음 갱신을 60초 더 기다린다(실측 서버 생성 간격 68초,
//   클라 폴링 60초 → 최악 ~120초 묵은 청산선). 절반으로 물으면 최악 지연이 생성주기+30초로
//   묶인다. 비용은 분당 33ms 요청 하나다(실측 p95 33ms · 50KB).
const LIQUIDATION_MAP_POLL_MS = 30000;
const REGIME_WIDE24_POLL_MS = 300000; // matches server-side cache (REGIME_WIDE24_CACHE_SECONDS)
const MACRO_CALENDAR_POLL_MS = 6 * 3600 * 1000; // matches server-side cache (MACRO_CALENDAR_CACHE_SECONDS)
const SESSION_ALERTS_POLL_MS = 30000; // 2026-08-27: split off evidence-signals' 5min cadence --
                                        // these badges need to feel live to someone watching a
                                        // +-30min window approach in real time, and the endpoint
                                        // is cheap enough (no new external fetch) to poll this often

// --- Chart Global Variables ---
const CHART_CANDLE_MIN = 5;
// 🔴2026-09-23 100 -> 200. 이 값이 «화면이 보여줄 수 있는 가장 긴 구간»을 정한다.
//   100 이면 8.3시간이라 12h 창을 골라도 8.3시간에서 끊겼다 -- 서버도 같이 200 으로
//   (tail(200)). 200 = 16.6시간이라 12h 창에 여유가 있다.
const CHART_MAX_CANDLES = 200;
// Snapshot tab's own chart only -- narrower than CHART_MAX_CANDLES (Live tab, unaffected) so every
// visible column has a real compute_heatmap_history() snapshot behind it (2026-08-25 user request,
// "차트를 4시간만 보여주는건 어떨까", then same day "4시간은 너무 작다" -> 6h -- see
// live_liquidation_map_20260824.py::compute_heatmap_history and its HEATMAP_HISTORY_DISPLAY_HOURS,
// which this must match).
// 🔴2026-09-23 72(6h) -> 96(8h), 사용자 지시. 이 값과 청산밀도 이력(LIQUIDATION_MAP_
//   DISPLAY_HOURS · HEATMAP_HISTORY_DISPLAY_HOURS)은 **짝**이다 -- 2026-08-25 에 «보이는
//   칸마다 진짜 스냅샷이 있게» 같이 맞춘 것이다. 셋을 따로 움직이면 배경 없는 칸이 생긴다.
//   🔴«밀도 배경이 빈다»는 내 우려는 **틀렸다**: 그 상수는 버퍼가 아니라 **설정**이고
//     원본은 23시간을 갖고 있다(실측 lookback_hours=23.0 · bars_used=24). 늘리면 스냅샷을
//     그만큼 더 계산할 뿐 빈 칸은 안 생긴다.
const SNAPSHOT_CHART_MAX_CANDLES = 96; // 8h at 5-min candles
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
  // 2026-10-01 코인 탭 = 1h~12h 와 같은 세그먼트(사용자 지시) -- 숨은 코인(BTC·HYPE)은 칸이 아니므로 «보이는 칸» 기준으로 조각 폭·위치를 준다.
  const host = el("snapshotAssetTabs"), vis = [...document.querySelectorAll("#snapshotAssetTabs .asset-tab")].filter((b) => !b.hidden);
  if (host) { host.style.setProperty("--seg-n", String(vis.length || 1)); host.style.setProperty("--i", String(Math.max(0, vis.findIndex((b) => b.classList.contains("active"))))); }
}

async function setActiveSnapshotAsset(asset) {
  if (!SNAPSHOT_ASSET_KEYS.includes(asset) || asset === activeSnapshotAsset) return;
  activeSnapshotAsset = asset;
  chartPanEnd = null;   // 코인을 바꾸면 실시간으로
  renderSnapshotAssetTabs();
  // 🔴2026-09-30 진입 미리보기(plan·cap)는 **코인별**인데 전환 때 안 비워 XRP 탭이 최대 60초(주문 불가 코인이면
  //   계속) ETH 계획(~$2,676)을 «지금 설정으로 넣으면»에 그렸다. 비우고 새 코인 것을 바로 받는다.
  lastEntryPlan = null; lastEntryCap = null; setEntryProjPreview(null);
  manualEntryRefreshSize();
  // 계좌 payload 는 전 코인의 포지션을 담고 있어 재요청 없이 다시 그리기만 하면 된다.
  renderSnapshotAccount();
  // Clear the 4 wired signals' cached readings + their poll-interval gates immediately -- without
  // this, the panels would keep showing the PREVIOUS coin's numbers (mislabeled as the new one)
  // until each signal's own poll interval next elapses (up to 5min for the slowest).
  // 2026-09-22 ETH 전용 카드에 «ETH 전용» 배지를 켠다(styles.css 의 body.not-eth 규칙).
  document.body.classList.toggle("not-eth", asset !== "eth");
  // ETH 전용 그림들의 캐시도 비운다 -- 안 비우면 ETH 로 **돌아올 때** 옛 그림이 한 프레임 번쩍인다.
  footprintBars = new Map(); latestFootprint = null;
  // 2026-09-26 SOL·XRP 도 흐름이 있다 -- 1초 수급 칸·커서·실시간 셀·히트맵을 코인마다 새로 받는다.
  //   🔴커서를 안 비우면 새 코인에 옛 코인의 초 번호로 «그 뒤만» 달라고 해서 창 앞부분이 빈다.
  resetSupply1sState();
  // 🔴2026-09-27 옛 코인 스트림을 **여기서 바로** 닫는다. 다음 틱(ensureLiveStream)까지 열려 있으면 그 사이 온
  //   ETH 한 덩이가 방금 비운 칸에 얹혀 커서를 ETH 최신 초로 옮기고, 새 코인 스트림이 그 뒤만 받아 수급이
  //   몇 초치로 굳었다(실측 ETH->XRP 8초 뒤에도 8초치 · OI 는 커서가 따로라 11분치 = «OI 만 보인다»).
  if (liveStream) { liveStream.close(); liveStream = null; }
  ensureLiveStream();
  footprintLastFetchAt = 0;   // 풋프린트도 폴링 간격을 기다리지 않고 곧바로
  // 🔴칸 폭은 **모른다**(0)로 둔다 -- 옛 코인 폭(ETH 0.5)으로 XRP 체결을 묶으면 진행 봉 셀이 엉뚱한 행에 쌓인다.
  //   새 코인 풋프린트 응답이 폭을 알려 줄 때까지 실시간 셀은 안 쌓고(footprintLiveAdd), 그 봉은 서버 값을 쓴다.
  footprintLive = { barStart: 0, since: Infinity, bucket: 0, cells: new Map(),
                    orderQty: 0, orderAt: null, orderLastMs: 0, orderLastTid: -1 };
  latestFlowHeatmap = null; flowHeatmapLastFetchAt = 0;
  latestLiquidation5m = null;
  latestLiquidation5mHist = [];
  latestOi5m = null; oi5mLastFetchAt = 0;
  latestLiquidationMap = null;
  liquidation5mLastFetchAt = 0;
  liquidationMapLastFetchAt = 0;
  lastSnapshotHistoryFetchAt = 0;
  // 레짐 리본도 코인별 모델이다. tick()이 이제 **활성 코인 것만** 가져오므로(refreshActiveRegime),
  // 전환 시엔 새 코인의 게이트를 열어 즉시 한 번 받아온다.
  regimeWide24LastFetchAt = 0;
  regimeBtcLastFetchAt = 0;
  regimeXrpLastFetchAt = 0;
  // 🔴2026-09-30 차트 표식(전환 예고/탐지)도 코인별이다 -- 안 비우면 ETH 사각형이 SOL 차트에 최대 60초 남았다.
  latestChartMarkers = null; chartMarkersLastFetchAt = 0;

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
    settleScope("indicators", []),   // 비ETH 코인 지표 API 제거(2026-10-01) -- 스켈레톤 해제만
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
  setupChartPan();
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
      flowHeatmapLastFetchAt = 0;
      // 🔴2026-09-25 청산 이력도 이제 창 폭을 들고 간다 -- 여기 안 넣으면 토글을 눌러도
      //   다음 폴링까지 옛 폭으로 남는다(위 셋과 같은 이유).
      liquidation5mLastFetchAt = 0;
      refreshFootprint();
      refreshFlowHeatmap();
      refreshLiquidation5mSignal();
      refreshGex();
    });
  });
  renderChartWindowTabs();
}

function renderChartWindowTabs() {
  const host = document.getElementById("chartWindowTabs");
  const tabs = [...document.querySelectorAll("#chartWindowTabs .asset-tab")];
  let idx = 0;
  tabs.forEach((btn, k) => {
    const on = Number(btn.dataset.bars) === chartWindowBars;
    btn.classList.toggle("active", on);
    if (on) idx = k;
  });
  // 🔴선택 칸(.chart-mode-tabs::before)의 **폭과 위치는 CSS 변수**다. 전에는 둘 다 칸 수를
  //   3 으로 박아 뒀다: `#chartWindowTabs { --seg-n: 3 }` 와 `:has(nth-child(2|3).active)`
  //   두 규칙뿐. 그래서 12h(4번째)를 고르면 칸이 **1/3 폭으로 첫 자리에 돌아갔다** --
  //   데이터는 바뀌는데 하이라이트는 1h 에 남아 «안 눌렸다»로 보였다(사용자 신고).
  //   버튼 수와 선택 위치를 여기서 읽어 넣는다. 칸이 또 늘어도 CSS 를 안 고쳐도 된다.
  if (host) {
    host.style.setProperty("--seg-n", String(tabs.length));
    host.style.setProperty("--i", String(idx));
  }
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

  // 🔴2026-09-25 게이지를 **차트의 청산 원에서 계산한다**. 전에는 봇 DB(tail_risk_1m)의 30분 합을 60초마다
  //   받았는데 ①1~3분 늦었고 ②봇이 청산을 l(마지막 체결 조각)로 세어 ~6배 작았다(24h $6.2M vs $38.0M).
  //   청산 원은 대시보드 자체 @forceOrder(z = 주문 누적 체결량)로 2초마다 갱신되므로, 현재 30분 봉에 든
  //   원들을 더하면 **원과 게이지가 정의상 같은 값**이 되고 새 요청도 없다.
  //   30분 경계는 서버 게이지(_bar_start)와 같다: UTC 분 // 30.
  const liqBars = Array.isArray(latestLiquidation5mHist) ? latestLiquidation5mHist : [];
  const barStart = Math.floor(Date.now() / 1000 / 1800) * 1800;
  const inBar = liqBars.filter((b) => Date.parse(b.ts) / 1000 >= barStart);
  const fromBars = inBar.length > 0;
  const liq5m = latestLiquidation5m;
  const warmed = fromBars || !!(liq5m && liq5m.warmed_up);
  const longUsd = fromBars ? inBar.reduce((s, b) => s + (Number(b.long_usd) || 0), 0)
    : (warmed ? Number(liq5m.long_usd_5m || 0) : 0);        // 청산 원이 아직 없을 때만 서버 게이지
  const shortUsd = fromBars ? inBar.reduce((s, b) => s + (Number(b.short_usd) || 0), 0)
    : (warmed ? Number(liq5m.short_usd_5m || 0) : 0);
  const total = longUsd + shortUsd;
  host.title = fromBars
    ? "현재 30분 봉에 든 차트 청산 원의 합(바이낸스 선물" + (inBar.every((b) => b.okx) ? " + OKX" : "")
      + (inBar.some((b) => b.hl) ? " + HL 고래" : "")
      + ") -- 원과 같은 값이다. 2초마다 갱신."
    : "";

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

// 🔴2026-09-30 SOL·XRP 는 서버 market-history 가 **형성 중 봉**까지 60초 캐시로 준다 -- 통째로 갈아 끼우면
//   라이브 틱이 넓혀 둔 그 봉의 고가·저가가 최대 60초 전 값으로 쪼그라들었다. 같은 시각의 형성 봉이면
//   고가=max·저가=min 으로 합친다(종가는 바로 뒤 updateSnapshotCandleLive 가 최신 라이브가로 덮는다).
//   마감된 봉은 서버 값 그대로. 순수 함수(test/test_merge_forming_candle_20260930.py 가 본문을 떼어 돌린다).
function mergeFormingCandle(prev, next, nowS, barMin = CHART_CANDLE_MIN) {
  const a = Array.isArray(prev) && prev.length ? prev[prev.length - 1] : null;
  const b = next.length ? next[next.length - 1] : null;
  if (a && b && a.time === b.time && nowS < b.time + barMin * 60) {
    b.high = Math.max(b.high, a.high);
    b.low = Math.min(b.low, a.low);
  }
  return next;
}

async function fetchBinanceHistory(asset) {
  try {
    const res = await fetch(`/api/market-history?asset=${asset}`, { cache: "no-cache" });
    if (!res.ok) return;
    const payload = await res.json();
    candleHistoryByAsset[asset] = mergeFormingCandle(candleHistoryByAsset[asset],
      Array.isArray(payload?.candles) ? payload.candles : [], Date.now() / 1000);
    // 🔴2026-09-23 5분봉 깜빡임의 정체. 서버는 **마감봉만** 준다(closed_df) -- 형성 중 봉은
    //   updateSnapshotCandleLive() 가 따로 밀어 넣는다. 그런데 이 교체 직후 호출자가 바로
    //   렌더를 예약하고, scheduleSnapshotChartRender() 는 그 함수를 **안 부른다**
    //   (부르는 건 maybeRenderSnapshotChartNow() 뿐). 그래서 fetch 마다 형성 중 봉이 빠진
    //   프레임이 한 번 그려지고 -- slice(-chartWindowBars) 가 한 칸 왼쪽으로 밀려 맨 과거
    //   봉이 되살아났다가 다음 틱에 되돌아온다. 1h 창(12봉)에서는 화면의 1/12 가 움직인다.
    //   위상이 안 맞는 폴링이라 봉 경계와 무관하게 났고, HISTORY_RETRY_MS(10초)가 빈도를
    //   5분에 한 번에서 10초에 한 번으로 늘렸다.
    //   교체와 형성봉 복원을 **한 틱 안에서** 끝낸다 -- 배열이 형성봉 없이 관측되지 않는다.
    updateSnapshotCandleLive();
  } catch (e) { console.error("History Error:", e); }
}

// 🔴2026-09-22 주기(5분)가 봉 길이와 같은데 **위상이 안 맞는다** -- 경계 직후 새로 마감된
//   봉은 라이브 갱신이 밀어 넣은 OHLC 뿐이라 sma/vn/drop 이 없고, 서버판이 올 때까지
//   SMA144±ATR 선이 멈춘다(실측: 경계 넘자 뒤처짐 1 -> 2봉, 5분 내내 2봉).
//   🔴«경계에서 한 번 연다»로 고치려다 실패했다 -- 그 순간 서버 워커가 아직 그 봉을 안
//     만들었으면 낡은 값을 받고 게이트가 다시 5분 닫힌다(실측으로 확인). 시계가 아니라
//     **증상**을 조건으로 건다: 마감된 봉에 sma 가 없으면 곧 다시 묻는다. 채워지면 저절로
//     원래 주기로 돌아간다(풋프린트 «전량 재수신» 자가복구와 같은 방식).
//   서버는 마감봉 100개를 전부 sma 와 함께 준다(실측 서버 뒤처짐 0) -- 못 받는 건 타이밍뿐이다.
//   🔴2026-09-23 사용자 지시 «5분봉 할 때 한 번만». 10초 재시도는 **원천보다 6배 빨랐다** --
//     서버의 마감봉 프레임은 EVIDENCE_SIGNAL_CACHE_SECONDS(60초) TTL 이라 그 안에 여섯 번
//     물어도 다섯 번은 같은 답이다. 그래서 봉당 **정확히 한 번**으로 줄인다.
//     한 번뿐이라 «언제»가 중요해졌다. 마감 직후에 쏘면 서버가 아직 안 만들어 한 발을
//     헛되이 쓴다(위 «경계에서 한 번»의 실패가 바로 그것) -- TTL 60초를 넘긴 뒤에 쏜다.
//     sma 가 이미 들어와 있으면 재시도 자체가 **0회**다(조건이 증상이라 그대로 남는다).
// 🔴2026-09-26 사용자 «추세 veto 선이 5분마다 업데이트가 안 된다» -- 위 ponytail 이 예고한 그 실패였다.
//   실측(서버, 10초 간격): 마감봉 값은 마감 뒤 ~60~65초에 붙는데 재시도는 +70초 **한 번**뿐이라, 서버 시계
//   톱니·조회 지연으로 그 한 발이 옛 프레임을 받으면 다음 5분 폴링까지 선이 두 봉 모자란 채 멈췄다
//   (라이브 한 칸 연장은 «직전 마감봉에 sma 가 있을 때»만 되므로). ⇒ **값이 붙을 때까지 15초마다** 다시 묻는다.
//   조건이 증상(직전 마감봉에 sma 없음)이라 붙는 순간 멈추고, 마감 뒤 두 봉이 지나면 정기 폴링에 맡긴다.
const HISTORY_RETRY_AFTER_CLOSE_S = 60;   // 서버 프레임 TTL 60초 -- 그 전엔 물어도 같은 답
const HISTORY_RETRY_EVERY_MS = 15000;
let historyRetryAt = 0;
// 순수 함수(test/test_history_retry_due_20260926.py 가 본문을 떼어 돌린다). barMin 은 봉 길이(분).
function historyRetryDue(closedTime, closedSma, nowMs, lastRetryMs, barMin = CHART_CANDLE_MIN) {
  const sinceClose = nowMs / 1000 - (closedTime + barMin * 60);
  // null 도 «없음»이다 -- Number(null) 은 0 이라 «있음»으로 새었다(테스트가 잡았다).
  return closedTime > 0 && (closedSma == null || !Number.isFinite(Number(closedSma)))
    && sinceClose >= HISTORY_RETRY_AFTER_CLOSE_S && sinceClose < 2 * barMin * 60
    && nowMs - lastRetryMs >= HISTORY_RETRY_EVERY_MS;
}
async function maybeFetchSnapshotChartHistory() {
  const now = Date.now();
  const cached = candleHistoryByAsset[activeSnapshotAsset] || [];
  // 마지막은 형성 중 봉이라 sma 가 없는 게 정상이다. 그 **직전**(마감된 봉)에 없으면 밀렸다.
  const closed = cached.length >= 2 ? cached[cached.length - 2] : null;
  const closedTime = Number(closed && closed.time) || 0;
  const retryDue = historyRetryDue(closedTime, closed && closed.sma, now, historyRetryAt);
  if (!retryDue && cached.length && now - lastSnapshotHistoryFetchAt < CANDLE_HISTORY_POLL_MS) return;
  if (retryDue) historyRetryAt = now;
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
// 증거금 사용 % = 사용 증거금(initialMargin) / 순자산(지갑+미실현). 계좌 카드 타일과 떠 있는 주문 버튼이 **같은 값**을
//   말해야 해서 한 곳에 둔다(2026-09-26). 옛 상태파일은 initial_margin 이 없어 순자산−가용으로 되짚는다(바이낸스 정의상 같다).
function acctMarginUsed(b) {
  b = b || {};
  const equity = (Number(b.wallet) || 0) + (Number(b.unrealized) || 0);
  const used = Number.isFinite(Number(b.initial_margin)) && b.initial_margin != null ? Number(b.initial_margin)
    : Math.max(0, (Number(b.margin) || 0) - (Number(b.available) || 0));
  return { used, equity, pct: equity > 0 ? used / equity * 100 : 0 };
}
// 2026-09-26 코인별 주문 심볼(서버 MANUAL_EXEC_SYMBOLS: eth→ETHUSDC · sol→SOLUSDC · xrp→XRPUSDC). 표에 없는 코인은 ""(주문 불가).
function execSymbolFor(asset) {
  const acc = latestBinanceAccount || lastGoodAccount || {};
  return String((acc.exec_symbols || {})[asset] || (asset === "eth" ? acc.exec_symbol || "" : "")).toUpperCase();
}

function snapshotAccountPosition() {
  const positions = latestBinanceAccount?.positions || [];
  const market = ASSET_CONFIG[activeSnapshotAsset]?.symbol || `${activeSnapshotAsset.toUpperCase()}USDT`;
  const exec = execSymbolFor(activeSnapshotAsset);
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
// ── 계좌 차트 기간 (2026-09-22 사용자 «1주일치만 보여주고 내가 늘려서 더 볼 수 있게») ──
//   기본 1주. 원장은 계속 쌓이므로 전체를 그리면 최근 며칠이 몇 픽셀로 눌린다.
//   0 = 전체. 저장은 chartWindowBars 와 같은 방식이다(화이트리스트로 되읽어 쓰레기값 차단).
const ACCT_SPANS = [[7, "1주"], [30, "1개월"], [0, "전체"]];
let acctSpanDays = (() => {
  try {
    // 🔴저장값이 없으면 getItem 은 null 이고 Number(null) 은 **0** 이다. 0 이 «전체»라
    //   화이트리스트를 통과해서, 처음 열면 기본이 1주가 아니라 전체가 됐다(실측).
    //   chartWindowBars 패턴에는 0 이 유효값이 아니라 이 함정이 없었다 -- 그대로 베끼면 안 된다.
    const raw = localStorage.getItem("acctSpanDays");
    if (raw === null || raw === "") return 7;
    const v = Number(raw);
    return ACCT_SPANS.some(([d]) => d === v) ? v : 7;
  } catch (e) { return 7; }
})();

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
  // 2026-09-25 사용자 «막대 높이를 키워줘» -- 누적과 같은 축이면 누적이 커질수록 건당 막대가 납작해진다
  //   (실측 누적 $884 에 건당 대부분 $10 대라 몇 픽셀). 눈금 숫자가 없는 차트라 막대는 **자기 최대값**
  //   으로 칸을 채우고, 크기는 툴팁 숫자가 말한다. 누적선은 위 공동 축 그대로.
  const scBar = Math.min(zero - padY, H - zero - padY) * 0.95 / Math.max(...net.map(Math.abs), 1e-9);
  const xAt = (i) => padX + i * bw;
  const yAt = (v) => zero - v * sc;
  // 막대: 24px 상한 · 인접 막대 사이 2px 표면 간격(테두리를 그리지 않는다)
  const bwFill = Math.min(24, Math.max(1.5, bw - 2));
  const bars = net.map((v, i) => {
    const x = xAt(i) + (bw - bwFill) / 2;
    const h = Math.max(Math.abs(v) * scBar, 1);
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
    // 2026-09-25 사용자 «몇 일에 얼마를 벌었거나 잃었다» -- 첫 줄이 곧 그 문장이다.
    const head = `<b class="${v < 0 ? "bad" : "good"}">${escapeHtml(m.day || `${i + 1}번째 왕복`)}`
      + ` ${escapeHtml(fmtUsd(Math.abs(v)))} ${v < 0 ? "잃음" : "벌었음"}</b>`;
    const rows = [
      `${m.when || ""}${m.side ? " · " + m.side : ""}${m.venue ? " · " + m.venue : ""}`,
      m.qty ? `수량 ${Number(m.qty.toFixed(3))} ETH · ${fmtUsd(m.notional)}` : "",
      m.entry && m.exit ? `진입 ${fmtUsd(m.entry)} → 청산 ${fmtUsd(m.exit)}` : "",
      `누적 ${fmtUsd(cums[i])}`,
    ].filter(Boolean);
    // 머리줄의 <b> 만 우리가 넣은 마크업이고 나머지는 이스케이프한다.
    const html = [head, ...rows.map(escapeHtml)].join("<br>");
    // 2026-09-26 비평: 막대마다 tabindex 0 이라 탭 정지가 28개였고 읽어 줄 이름도 없었다 -- 최근 막대 하나만
    //   탭 순서에 두고(좌우 방향키로 옮긴다, bindAcctChartTip) 머리줄을 이름으로 준다.
    const name = `${m.day || `${i + 1}번째 왕복`} ${fmtUsd(Math.abs(v))} ${v < 0 ? "잃음" : "벌었음"}`;
    return `<rect x="${xAt(i).toFixed(1)}" y="0" width="${bw.toFixed(1)}" height="${H}" `
      + `fill="transparent" role="img" aria-label="${escapeHtml(name)}" `
      + `tabindex="${i === net.length - 1 ? 0 : -1}" data-tip="${escapeHtml(html)}"></rect>`;
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
let acctPerfOpen = (() => { try { return localStorage.getItem("acctPerfOpen") === "1"; } catch (e) { return false; } })();
function bindAcctChartTip() {
  // 🔴2026-09-25 툴팁이 **안 떴다** -- 차트는 #snapAcctPerf 로 옮겨갔는데 위임 호스트가 옛 칸
  //   (#snapAcctPosition)에 남아 있어 이벤트가 닿지 않았다(플레이라이트 호버로 확인).
  const host = el("snapAcctPerf");
  if (!host || host.dataset.tipBound) return;
  host.dataset.tipBound = "1";
  // 펼침 기억 -- toggle 은 버블링하지 않으므로 캡처로 받는다. 카드는 30초마다 다시 그려지므로 상태를 들고 있어야 한다.
  host.addEventListener("toggle", (e) => {
    if (!e.target.classList?.contains("acct-perf-fold")) return;
    acctPerfOpen = e.target.open;
    try { localStorage.setItem("acctPerfOpen", acctPerfOpen ? "1" : "0"); } catch (err) { /* 저장 못 해도 동작은 같다 */ }
  }, true);
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
  // 막대 사이는 좌우 방향키로 옮긴다(탭 정지는 하나 -- 위 acctPerfSvg 주석).
  host.addEventListener("keydown", (e) => {
    if (!e.target.dataset?.tip || (e.key !== "ArrowLeft" && e.key !== "ArrowRight")) return;
    const next = e.key === "ArrowLeft" ? e.target.previousElementSibling : e.target.nextElementSibling;
    if (!next?.dataset?.tip) return;
    e.preventDefault();
    e.target.tabIndex = -1; next.tabIndex = 0; next.focus();
  });
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
  if (!pv) {
    root.querySelectorAll(".acct-rail u, .acct-gauge u").forEach((u) => u.remove());
    root.querySelectorAll("u.pv-after").forEach((u) => u.remove());
    return;
  }

  const a = pv.after, plan = pv.__plan || {};
  const LIQ_FULL = 10, EXPO_CAP = 30;
  const long = pos.side === "LONG";
  // 타일 셋: 값·색·막대·유령눈금(지금 자리)
  // 🔴2026-09-22 시안 B. 전에는 이 패처가 **사실을 덮어썼다** -- manualEntryRefreshSize 가
  //   setInterval 로 계속 돌아 큰 숫자 셋이 상시로 «25% 더 넣었다면»의 값이었고, 실제 계좌는
  //   유령 눈금으로만 남았다. 포지션을 들고 있는 사람이 화면에서 제일 큰 글씨로 «자기가 하지
  //   않은 거래»의 결과를 본다. 이제 **사실은 그대로 두고 «→ 가정»을 뒤에 붙인다** --
  //   막대도 사실 폭을 지키고, 움직이는 건 가정 자리의 유령 눈금뿐이다.
  const pvAfter = (node, txt, tone) => {
    if (!node) return;
    let u = node.querySelector("u.pv-after");
    if (!u) { u = document.createElement("u"); node.appendChild(u); }
    u.className = `pv-after${tone ? ` ${tone}` : ""}`;
    pvText(u, `→ ${txt}`);   // 값이 같으면 안 건드리고(폴링 떨림), 바뀔 때만 반짝인다
  };
  const setTile = (k, txt, tone, fill) => {
    const t = q(k); if (!t) return;
    pvAfter(t.querySelector(".acct-tile-val"), txt, tone);
    const rail = t.querySelector(".acct-rail");
    if (rail) {
      rail.classList.add("entry-rail");
      let u = rail.querySelector("u");
      if (!u) { u = document.createElement("u"); rail.appendChild(u); }
      u.style.left = `${clamp01(fill) * 100}%`;
      u.title = "지금 설정으로 넣으면 여기";
    }
  };
  setTile("liq", `${Number(a.liq_pct).toFixed(2)}%`, acctRiskTone(a.liq_pct), a.liq_pct / LIQ_FULL);
  setTile("used", `${Number(a.margin_used_pct).toFixed(0)}%`,
          a.margin_used_pct > 80 ? "bad" : a.margin_used_pct > 60 ? "warn" : "good",
          a.margin_used_pct / 100);
  setTile("expo", `${Number(a.exposure_x).toFixed(1)}배`, a.exposure_x > 15 ? "bad" : "warn",
          a.exposure_x / EXPO_CAP);

  // 포지션 카드: 수량·레버리지·평단·청산가·손잡이. 평단은 **체결가 가중평균**이다.
  const addQty = Number(plan.quantity) || 0, addPx = Number(plan.price) || 0;
  const haveQty = Number(pos.qty) || 0, havePx = Number(pos.entry_price) || 0;
  const newQty = haveQty + addQty;
  const newEntry = newQty > 0 ? (haveQty * havePx + addQty * addPx) / newQty : havePx;
  // 청산가는 거래소가 준 **거리(%)** 에서 되돌린다 -- 근사식보다 실측에 앵커된 값이다.
  const newLiq = mark > 0 ? mark * (1 + (long ? -1 : 1) * Number(a.liq_pct) / 100) : 0;
  pvAfter(q("qty"), `~${newQty.toFixed(3)}`);
  const tg = q("tag");
  if (tg && plan.target_leverage && String(plan.target_leverage) !== String(pos.leverage)) {
    pvAfter(tg, `×${plan.target_leverage}`);
  }
  pvAfter(q("liqpx"), `~${fmtUsd(newLiq)}`, "bad");
  // 🔴평단은 **추측치**다. 체결가를 peg 호가로 가정한 가중평균이라 실제 체결(부분체결·
  //   테이커 폴백·슬리피지)에 따라 달라진다. `~` 와 툴팁으로 그 사실을 남긴다.
  const ep = q("entrypx");
  if (ep) {
    pvAfter(ep, `~${fmtUsd(newEntry)}`);
    ep.title = `추측치 — 지금 ${fmtUsd(havePx)} (${haveQty.toFixed(3)} ${coinUnit()})에`
      + ` ${addQty.toFixed(3)} ${coinUnit()} 를 ${fmtUsd(addPx)}(peg 호가)에 더한 가중평균입니다.`
      + `\n실제 체결가가 다르면(부분체결·테이커 폴백) 평단도 달라집니다.`;
  }
  // 🔴손잡이는 **지금 내 자리**다 -- 옮기면 사실이 사라진다. 대신 «넣으면 여기»를 같은 레일에
  //   유령 눈금으로 하나 더 세운다(시안 B). 청산선이 어느 쪽으로 얼마나 끌려오는지를 숫자가
  //   아니라 거리로 보여주는 게 이 카드에서 실수가 나는 지점이다.
  const gauge = root.querySelector(".acct-gauge");
  if (gauge) {
    const span = Math.abs(newEntry - newLiq) * 2;
    const at = clamp01(span > 0 ? Math.abs(mark - newLiq) / span : 0) * 100;
    let g = gauge.querySelector("u");
    if (!g) { g = document.createElement("u"); gauge.appendChild(g); }
    g.className = "acct-gauge-pv";
    g.style.left = `${at.toFixed(1)}%`;
    // 🔴유령 손잡이에는 라벨을 안 붙인다. 투영 청산가는 바로 아래 범례가 이미
    //   «청산 $2,197.56 → ~$2,197.01» 로 말한다 -- 같은 말을 두 번 하면서, 접힌 눈금에서
    //   진입은 **항상 50%** 라 진입 라벨과 겹친다(스캘핑에선 현재≈진입이 기본 상태다).
    g.title = `지금 설정으로 넣으면 손잡이가 여기로 옵니다 (청산 ~${fmtUsd(newLiq)})`;
  }
}

let entryProjPreview = null;
let lastAcctPos = null;   // 패처가 쓰는 마지막 렌더 문맥(포지션·마크가·순자산)
let entryProjKey = "";
// 2026-09-22 아티팩트 댓글 «포지션이 없으면 카드가 안 보이는데 없어도 max 수치를».
//   미리보기(/api/manual-entry/preview)가 이미 받아오는 cap 을 그대로 들고 있는다 --
//   새 엔드포인트도 새 폴링도 없다. 포지션이 없을 때 이 값으로 타일을 채운다.
let lastEntryCap = null;
// 2026-09-22 (2)(3): 타일·게이지가 «지금 칩 설정»을 따라야 하므로 cap 만으로는 모자란다.
//   cap 은 천장이라 비율·배수에 안 움직인다(실측). 움직이는 건 plan.projection 이다.
let lastEntryPlan = null;

// 🔴위임으로 한 번만 붙인다. 이 차트는 계좌 폴링마다 통째로 다시 그려지므로 노드에
//   직접 붙이면 렌더할 때마다 다시 붙여야 하고, 한 번 빠뜨리면 조용히 안 먹는다.
document.addEventListener("click", (e) => {
  const b = e.target.closest(".acct-span-pick .chip");
  if (!b) return;
  const d = Number(b.dataset.days);
  if (!ACCT_SPANS.some(([x]) => x === d) || d === acctSpanDays) return;
  acctSpanDays = d;
  try { localStorage.setItem("acctSpanDays", String(d)); } catch (err) { /* 사파리 프라이빗 */ }
  renderSnapshotAccount();
});

function renderSnapshotAccount() {
  syncOrderCoinGate();
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

  // 오른쪽 성과 -- 보고 있는 코인의 닫힌 왕복.
  // 🔴원천은 거래소 payload(trades)가 아니라 **서버 원장(ledger)** 이다. userTrades 는 7일
  //   롤링이라 창 앞에서 열린 포지션은 진입 체결이 사라지고, 폴딩이 포지션 한가운데서 시작해
  //   유령 왕복을 만든다(2026-09-22 실측: ETHUSDT 8건 중 3건이 유령·6건이 항등식 불일치,
  //   승률 50% 로 표시됐지만 실제는 80%). 원장은 항등식·겹침을 통과한 것만 담고 창이 지나도
  //   남는다. server.py load_account_trip_rows() 참조.
  // 🔴심볼이 아니라 **코인**으로 거른다. 같은 ETH 를 두 심볼로 거래한다 -- 대시보드 수동 주문은
  //   ETHUSDC, 급할 때는 바이낸스에서 직접 ETHUSDT. 심볼 하나로 거르면 한쪽이 통째로 사라진다
  //   (실제로 최근 19왕복 +$577 이 화면에 없었다). 바로 위 snapshotAccountPosition() 은 이미
  //   exec_symbol 을 보는데 여기만 ASSET_CONFIG 를 하드코딩하고 있었다.
  const base = (ASSET_CONFIG[activeSnapshotAsset]?.symbol || `${activeSnapshotAsset.toUpperCase()}USDT`)
    .replace(/USD[TC]$/, "");
  // 원장은 오래된 것부터 붙지만 정렬을 믿지 않는다 -- 거꾸로면 누적선이 시간을 거꾸로 달린다.
  const allClosed = (latestBinanceAccount.ledger || [])
    .filter((t) => String(t.symbol || "").replace(/USD[TC]$/, "") === base)
    .slice().sort((x, y) => (Number(x.exit_time) || 0) - (Number(y.exit_time) || 0));
  // 🔴자르기 **전** 건수를 들고 있어야 «밖에 더 있다»를 말할 수 있다. 그 숫자가 없으면
  //   1주 창이 비었을 때 «거래가 없다»인지 «창이 좁다»인지 구분이 안 된다.
  const cutoff = acctSpanDays > 0 ? Date.now() - acctSpanDays * 86400000 : 0;
  const closed = cutoff ? allClosed.filter((t) => (Number(t.exit_time) || 0) >= cutoff) : allClosed;
  const hiddenTrips = allClosed.length - closed.length;
  const net = closed.map((t) => Number(t.net_pnl) || 0);
  // 툴팁이 "날짜와 크기 등"을 보여줘야 하므로(사용자 지시) 라벨 문자열이 아니라 원장을 넘긴다.
  const netMeta = closed.map((t) => {
    const d = new Date(Number(t.exit_time) || 0);
    const qty = Number(t.max_qty) || 0, px = Number(t.exit_price) || 0;
    return {
      when: Number.isNaN(d.getTime()) ? "" : `${d.getMonth() + 1}/${d.getDate()} `
        + `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`,
      day: Number.isNaN(d.getTime()) ? "" : `${d.getMonth() + 1}월 ${d.getDate()}일`,
      side: t.side === "LONG" ? "롱" : "숏", qty, notional: qty * px,
      entry: Number(t.entry_price) || 0, exit: px,
      // 한 코인을 두 심볼로 거래하므로(USDC=대시보드 수동, USDT=바이낸스 직접) 어느 쪽이었는지
      // 툴팁이 말해 준다. 이게 없으면 원장에서 두 경로가 구분 불가능해진다.
      venue: String(t.symbol || "").replace(/^.*?(USD[TC])$/, "$1"),
    };
  });
  const wins = net.filter((v) => v > 0).length;
  const total = net.reduce((x, y) => x + y, 0);
  let worstIdx = -1;
  net.forEach((v, i) => { if (worstIdx < 0 || v < net[worstIdx]) worstIdx = i; });
  const rest = worstIdx >= 0 ? total - net[worstIdx] : 0;
  const chip = (v, lab) => `<span class="acct-chip"><b>${v}</b><span>${lab}</span></span>`;
  // 기간을 적는다(사용자 지시). 원장이 원천이므로 거래소 7일 창보다 길지만, 원장이 처음
  // 돌기 시작한 날 이전은 없다 -- 그래서 여기 있는 건 **기록된 범위**지 계좌의 전체 이력이 아니다.
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
    // 🔴2026-09-26 비평: 회고 차트(≈330px)가 주문 조작부와 시장 사이를 막아 시장이 y≈1150 에서야 시작했다.
    //   기본은 **한 줄 요약**(왕복·승률·누적)만, 차트는 펼쳐서 본다. 펼침 상태는 기억한다(acctPerfOpen).
    ? `<section class="acct-perf"><details class="acct-perf-fold"${acctPerfOpen || document.documentElement.classList.contains("fit1") ? " open" : ""}>
         <summary class="acct-chips">
           ${chip(net.length, "왕복")}
           ${chip(`${Math.round(wins / net.length * 100)}%`, "승률")}
           ${chip(`<span class="${total < 0 ? "bad" : "good"}">${fmtUsd(total)}</span>`, "누적")}
           <span class="acct-perf-more" aria-hidden="true"></span>
         </summary>
         <figure class="acct-plot">
           <div class="acct-plot-head">${legend}
             <span class="acct-span-wrap">
             ${spanText ? `<span class="acct-span" title="대시보드가 쌓아 둔 원장의 범위입니다 — 거래소 조회는 7일치만 주므로 그 앞은 여기에만 남습니다. 바이낸스에서 직접 낸 주문도 함께 들어 있습니다.">${escapeHtml(spanText)}</span>` : ""}
             <div class="chipset acct-span-pick" role="group" aria-label="차트 기간">
               ${ACCT_SPANS.map(([d, lab]) => `<button type="button" class="chip${
                 d === acctSpanDays ? " on" : ""}" data-days="${d}" aria-pressed="${d === acctSpanDays}"${
                 d > 0 && hiddenTrips > 0 ? ` title="이 밖에 ${hiddenTrips}건 더 있습니다"` : ""
               }>${lab}</button>`).join("")}
             </div>
           </span>
           </div>
           ${acctPerfSvg(net, netMeta)}
           <div class="acct-tip" hidden></div>
         </figure>
         ${worstIdx >= 0 && net[worstIdx] < 0 && net.length > 1
            ? `<p class="acct-perf-note"><span class="bad">최악 1건 ${fmtUsd(net[worstIdx])}</span>
                 · <span class="${rest < 0 ? "bad" : "good"}">나머지 ${net.length - 1}건 ${fmtUsd(rest)}</span></p>`
            : ""}
       </details></section>`
    : `<section class="acct-perf"><div class="acct-empty">${
        hiddenTrips > 0
          ? `최근 ${acctSpanDays}일에 닫힌 왕복이 없습니다 — 이 밖에 <b>${hiddenTrips}건</b> 있습니다.`
          : "닫힌 왕복이 아직 없습니다."}</div></section>`;

  const EXPO_CAP = 30;   // 막대 상한. 이 계좌 실측이 23배라 30을 만재로 둔다
  const LIQ_FULL = 10;   // 청산까지 10% 를 만재로 본다(그 이상은 사실상 안전)
  // 타일: 라벨·값·레일이 셋 다 같은 모양이라 눈이 세로로 훑힌다(옛 판은 숫자 셋 + 별도 막대).
  // ⭐여기서는 **항상 실제 계좌**를 그린다. 진입 미리보기는 렌더가 아니라 applyAcctPreview 가
  //   **같은 노드를 제자리에서** 고친다 -- 노드를 갈아치우면 transition 이 안 걸린다.
  //   `data-pv` 가 그 손잡이다.
  // 🔴2026-09-22 «포지션 없음» 분기도 같은 헬퍼를 쓰므로 그 분기보다 **위**에 있어야 한다.
  const tile = (key, lab, val, tone, fill, title) => `<div class="acct-tile" data-pv="${key}"${
      title ? ` title="${escapeHtml(title)}"` : ""}>
      <span class="acct-tile-lab">${lab}</span>
      <b class="acct-tile-val ${tone}">${val}</b>
      <span class="acct-rail"><i class="${tone}" style="width:${clamp01(fill) * 100}%"></i></span>
    </div>`;

  const otherNote = others > 0
    ? `<p class="acct-foot">다른 코인에 ${others}종목을 더 보유 중입니다 — 운영 관리 탭에서 전부 볼 수 있습니다.</p>`
    : "";
  if (!pos) {
    // ── 포지션이 없어도 «지금 설정으로 넣으면» 을 그린다 (2026-09-22 아티팩트 댓글) ──
    // 요청 셋: ①평단가 게이지도 항상 ②레버·비율을 바꾸면 값이 따라 움직이게 ③이 배수에서
    // 증거금으로 얼마까지 되는지.
    // 🔴실측이 설계를 갈랐다(lev 5/10/20/50 × pct 25/100 스윕):
    //   **비율은 전부 움직이고(증거금 151->604 · 노출 1.5->6.0 · 청산 66.7->16.7%),
    //     레버리지는 서버 응답을 하나도 안 움직인다.** want_lev 는 plan.target_leverage
    //     표시에만 쓰이고, 증거금은 «거래소 실제 설정»(leverage_by_symbol, 지금 20배)으로
    //     계산되기 때문이다. 그래서 천장(cap)이 아니라 **projection** 을 읽고, 배수 축은
    //     여기서 직접 나눈다 -- 주문 직전 ensure_leverage(target_leverage) 가 실제로
    //     거래소 배수를 바꾸므로(live_manual_peg_execute_20260912.py:168) 거짓이 아니다.
    //   청산 거리는 교차증거금이라 배수가 아니라 **노출**이 정한다(실측 1/노출).
    const pl = lastEntryPlan, cp = lastEntryCap;
    const pj = pl && pl.projection && pl.projection.after;
    const tgtLev = Number(pl && pl.target_leverage) || 0;
    const exLev = Number(pl && pl.leverage) || 0;          // 거래소 현재 설정
    const notional = Number(pj && pj.notional_usdt) || 0;
    const marginAt = tgtLev > 0 ? notional / tgtLev : 0;   // 그 배수로 바꾼 뒤의 증거금
    const avail = Number(b.available) || 0;
    const liqPct = Number(pj && pj.liq_pct) || 0;
    const capNotional = Number(cp && cp.cap_notional_usdt) || 0;
    const expo = Number(pj && pj.exposure_x) || 0;
    const fracPct = Math.round((Number(pl && pl.fraction) || 0) * 100);
    const BIND = { equity: "순자산", ledger: "원장 중앙", model: "위험 모델", survival: "생존", margin: "증거금 50%" };
    let body;
    if (pj && notional > 0) {
      // ③ 이 배수에서 증거금이 감당하는 명목 vs 정책 천장 -- 작은 쪽이 진짜 상한이다.
      const capN = capNotional;
      const byMargin = avail * (tgtLev || 1);
      const realCap = capN > 0 ? Math.min(capN, byMargin) : byMargin;
      const marginBinds = capN > 0 && byMargin < capN;
      const room = `<p class="acct-foot">레버 ${tgtLev}배 → 가용 ${fmtUsd(avail)} 로 `
        + `<b>${fmtUsd(byMargin)}</b> 까지 · 정책 천장 ${fmtUsd(capN)}`
        + `${cp && cp.binding ? ` (${BIND[cp.binding] || cp.binding})` : ""}`
        + ` → <b class="${marginBinds ? "warn" : ""}">실제 상한 ${fmtUsd(realCap)}</b>`
        + `${marginBinds ? " — 이 배수에서는 <b>증거금이 먼저 막습니다</b>" : ""}</p>`;
      // ① 게이지 = **증거금** (2026-09-22 3차 지시: 「머리줄도 빼주고 증거금은 진입할 때
      //   max 치를 게이지에 보여주고 진입비율에 맞는 증거금을 표시해줘」).
      //   눈금 전체 = 이 배수에서 **쓸 수 있는 최대 증거금**, 채움 = 지금 비율의 증거금.
      //   🔴최대 = min(정책천장/배수, 가용) 이다. 둘 중 작은 쪽이 진짜 한계라 -- 레버가
      //     낮으면 가용이, 높으면 정책 천장이 먼저 막는다(실측 lev5 \$2,015 vs lev20 \$604).
      //   🔴평단·청산은 뺐다(2차 지시). peg 호가라 현재가와 같은 수였다.
      const marginMax = tgtLev > 0 ? Math.min(capNotional / tgtLev, avail) : avail;
      const overTone = marginAt > avail ? "bad" : marginAt / Math.max(marginMax, 1e-9) > 0.8 ? "warn" : "good";
      // ── 척추 레일 (2026-09-22 시안 B, 사용자 «B 안이랑 디자인이 너무 다른데») ──
      //   보유 중과 **같은 문법**을 쓴다. 다만 포지션이 없으면 측면이 안 정해져서 «청산이
      //   어디»에 답이 **둘**이다 -- 한쪽만 그리면 거짓말이므로 양쪽을 다 그린다:
      //     롱 청산(아래) ←── 진입(현재가) ──→ 숏 청산(위)
      //   거리는 양쪽이 같다(liq_pct). 교차증거금이라 배수가 아니라 노출이 정하는 값이고,
      //   비율을 올리면 양쪽이 동시에 안쪽으로 좁혀 온다 -- 그게 이 레일이 보여주는 것이다.
      const px = Number(pl && pl.price) || 0;
      const loLiq = px * (1 - liqPct / 100), hiLiq = px * (1 + liqPct / 100);
      const spine = px > 0 && liqPct > 0 ? `<div class="acct-pos spine proj">
             <div class="acct-gauge" title="포지션이 없으면 측면이 안 정해집니다 -- 지금 설정으로 넣었을 때 롱이면 아래, 숏이면 위 이 가격에서 청산됩니다. 거리는 양쪽이 같습니다(노출의 역수).">
               <span class="acct-gauge-track"></span>
               <span class="acct-gauge-entry"></span>
               <span class="acct-gauge-now" style="left:50%">진입 ~${fmtUsd(px)}</span>
             </div>
             <div class="acct-gauge-legend">
               <span class="bad" style="left:0">롱 청산 ~${fmtUsd(loLiq)}</span>
               <span class="bad" style="left:100%">숏 청산 ~${fmtUsd(hiLiq)}</span>
             </div>
           </div>` : "";
      body = `<div class="acct-pv-cap entry-cap">지금 설정으로 넣으면 — 레버 ${tgtLev}배 · 비율 ${fracPct}% (포지션 아님)</div>
        ${spine}
        <div class="acct-tiles">
          ${tile("liq", "청산까지", `${liqPct.toFixed(1)}%`, acctRiskTone(liqPct), liqPct / LIQ_FULL,
                 `넣은 뒤의 청산 거리입니다.\n교차증거금이라 **레버리지가 아니라 노출**이 정합니다`
                 + ` -- 지금 노출 ${expo.toFixed(1)}배의 역수(${(100 / Math.max(expo, 1e-9)).toFixed(1)}%)입니다.`
                 + `\n비율을 올리면 노출이 커지고 이 값이 줄어듭니다.`)}
          ${tile("used", "명목", `${fmtUsd(notional)}`, "warn",
                 realCap > 0 ? notional / realCap : 0,
                 `지금 비율(${fracPct}%)로 나가는 명목입니다. 눈금은 실제 상한 ${fmtUsd(realCap)} 기준.\n`
                 + `🔴거래소 현재 설정은 ${exLev}배입니다 -- 주문 직전에 ${tgtLev}배로 바꿔서 냅니다`
                 + `(ensure_leverage). 그래서 증거금은 «바꾼 뒤» 기준입니다.`)}
          ${tile("expo", "노출", `${expo.toFixed(1)}배`, expo > 15 ? "bad" : "warn", expo / EXPO_CAP,
                 `명목 ${fmtUsd(notional)} ÷ 순자산 ${fmtUsd(equity)}\n`
                 + `🔴레버리지를 바꿔도 이 값은 안 바뀝니다 -- 명목을 정하는 건 비율과 천장입니다.`)}
          ${/* 증거금 게이지를 **타일로 흡수**했다 -- 라벨+값+막대는 .acct-tile 과 같은 물건이라
                별도 부품을 둘 이유가 없다. 눈금 전체 = 이 배수에서 쓸 수 있는 최대 증거금. */ ""}
          ${tile("margin", "증거금", `${fmtUsd(marginAt)}`, overTone,
                 marginMax > 0 ? marginAt / marginMax : 0,
                 `지금 비율의 증거금입니다. 눈금 전체는 이 배수에서 쓸 수 있는 최대\n`
                 + `= min(정책천장 ÷ ${tgtLev}배, 가용 ${fmtUsd(avail)}) = ${fmtUsd(marginMax)}.`)}
          ${tile("room", "상한 여유", `${fmtUsd(Math.max(0, realCap - notional))}`,
                 notional / Math.max(realCap, 1e-9) > 0.9 ? "warn" : "good",
                 realCap > 0 ? 1 - notional / realCap : 0,
                 `실제 상한 ${fmtUsd(realCap)} 에서 지금 명목 ${fmtUsd(notional)} 을 뺀 값입니다.\n`
                 + `${marginBinds ? "이 배수에서는 증거금이 먼저 막습니다." : "정책 천장이 먼저 막습니다."}`)}
        </div>${room}`;
    } else {
      body = `<div class="acct-empty">${ASSET_CONFIG[activeSnapshotAsset]?.label
        || activeSnapshotAsset.toUpperCase()}에 열린 포지션이 없습니다.${
        pl === null ? " (크기 워커가 값을 내면 여기에 «지금 넣으면»이 뜹니다)" : ""}</div>`;
    }
    setH("snapAcctPosition", `<div class="acct-card">
        <div class="acct-spine-top">${hero}</div>${body}
      </div>${otherNote}`);
    setH("snapAcctPerf", perf);
    bindAcctChartTip();
    // 🔴패처(applyAcctPreview)는 «실제 포지션» 타일을 제자리에서 고치는 물건이다. 포지션이
    //   없을 때 그게 돌면 방금 그린 투영 값을 덮는다 -- 문맥을 비워 전체 재렌더로 보낸다.
    lastAcctPos = null;
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
  const { used: usedMargin, pct: usedPct } = acctMarginUsed(b);
  // 노출은 **계좌 전체** 기준이다(명목 ÷ 순자산). 포지션 레버리지(×30)와 다른 값이라
  //   같은 "배"를 써서 혼동이 났다 -- 라벨을 「계좌 노출」로 바꾸고 명목을 툴팁에 적는다.
  const expo = equity > 0 ? (Number(pos.notional) || 0) / equity : 0;
  const pvCap = `<div class="acct-pv-cap entry-cap" hidden>큰 숫자는 <b>지금 내 계좌</b> · 주황 <u class="pv-after">→</u> 와 점선 눈금은 <b>지금 설정으로 넣었을 때</b></div>`;
  const tiles = `<div class="acct-tiles">
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
      ${tile("upnl", "미실현", usd2(upnl), upnl >= 0 ? "good" : "bad", Math.min(1, Math.abs(upnlPct) / 5),
             `거래소가 준 미실현 손익입니다. 막대는 순자산 대비 ±5% 를 만재로 봅니다.`)}
      ${(() => {
        // 🔴시안 B: «지금 닫으면»이 청산을 누르기 직전 유일하게 중요한 숫자인데 레인 안 작은
        //   글씨였다. 같은 식(exitNetUsd)을 써야 레인과 타일이 어긋나지 않는다.
        const e = exitNetUsd(pos, 100);
        return tile("close", "지금 닫으면", e.net === null ? "—" : usd2(e.net),
                    (e.net || 0) >= 0 ? "good" : "bad", Math.min(1, Math.abs(e.net || 0) / (equity * 0.05 || 1)),
                    `전량 청산 시 지갑이 늘어나는 금액입니다.\n미실현 ${usd2(e.gross || 0)} − 청산 수수료 ≈$${
                      e.fee.toFixed(2)} (peg 실측 ${EXIT_FEE_BP_PEG}bp)\n진입 수수료는 이미 빠졌으므로 다시 빼지 않습니다.`);
      })()}
    </div>`;

  // ⭐청산 거리 게이지 -- 롱/숏 모두 **왼쪽 끝이 청산**이 되도록 접는다.
  //   0 = 청산가 · 0.5 = 진입가 · 1 = 진입에서 청산 거리만큼 이익 난 가격.
  //   측면마다 부등호를 뒤집지 않아도 되고, 눈은 "왼쪽에 가까울수록 위험"만 기억하면 된다.
  const span = Math.abs(entry - liq) * 2;
  const safe = span > 0 ? clamp01(Math.abs(mark - liq) / span) : 0;
  const sideTone = pos.side === "LONG" ? "good" : "bad";
  // 🔴2026-09-22 시안 B: 레일이 이 카드의 척추다. 머리줄(심볼·측면·수량)은 순자산 옆으로
  //   올려 한 줄이 되고, 게이지는 카드 전폭을 가로지른다. data-pv 훅은 **하나도 안 옮겼다**
  //   -- applyAcctPreview 는 한 줄도 안 고친다.
  const posHead = `<div class="acct-pos-head">
        <b>${escapeHtml(pos.symbol)}</b>
        <span class="acct-tag ${sideTone}" data-pv="tag">${pos.side === "LONG" ? "롱" : "숏"} ×${escapeHtml(pos.leverage)}</span>
        <span class="acct-pos-qty" data-pv="qty">${escapeHtml(pos.qty)}</span>
      </div>`;
  const position = `<div class="acct-pos spine" data-side="${pos.side === "LONG" ? "long" : "short"}">
      <div class="acct-gauge" title="왼쪽 끝이 청산가, 가운데 눈금이 진입가입니다. 손잡이가 왼쪽에 붙을수록 위험합니다.">
        <span class="acct-gauge-track"></span>
        <span class="acct-gauge-entry"></span>
        <span class="acct-gauge-knob" data-pv="knob" style="left:${(safe * 100).toFixed(1)}%"></span>
        <span class="acct-gauge-now" style="left:${(safe * 100).toFixed(1)}%">현재 ${fmtUsd(mark)}</span>
      </div>
      <div class="acct-gauge-legend">
        <span class="bad" data-pv="liqpx" style="left:0">청산 ${fmtUsd(liq)}</span>
        <span data-pv="entrypx" style="left:50%">진입 ${fmtUsd(entry)}</span>
      </div>
    </div>`;

  // 시안 B 구성: 척추(순자산+포지션 한 줄 → 전폭 레일) → KPI 5칸 → [index.html 의 두 레인]
  //   → 자산 곡선. 곡선은 «역사»라 조작보다 아래가 맞다(사용자 선택 2026-09-22).
  setH("snapAcctPosition", `<div class="acct-card">
      <div class="acct-spine-top">${hero}${posHead}</div>
      ${pvCap}
      ${position}
      ${tiles}
    </div>${otherNote}`);
  setH("snapAcctPerf", perf);
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
  // 2026-09-09 극점 탐지기. ⚠️키는 subText 문자열이다(규약 §5-1).
  breakout_prewarn: {
    "전환 예고": "앞으로 30분 안에 거래대금·체결속도가 함께 급증할 확률이 상위 10%에 들었어요 — 방향은 말하지 않습니다.",
    "미발동": "앞으로 30분 안에 거래량 급증이 올 확률이 평소 수준이에요.",
    "웜업": "예고 모델이 아직 첫 계산을 끝내지 않았어요.",
    "데이터 없음": "예고 모델이 값을 내지 못하고 있어요.",
    "오류": "시세를 읽지 못해 이번 봉을 채점하지 못했어요.",
  },

  // ⚠️키는 subText 문자열이다(규약 §5-1) -- 라벨을 바꾸면 여기도 같이 바꾼다.
};

const MODEL_INDICATOR_DETAIL = {
  // 2026-09-30 탐지기 카드 제거(사용자 결정) -- 경보기만 남는다. 숫자는 재생 검증(tmp/bd_verify_20260930/replay.py).
  breakout_prewarn:
    "앞으로 30분 안에 **거래대금·체결속도가 함께 급증하는가**(둘 다 z288 이 후행 2016봉 q90 이상 -- 옛 «탐지기» 조건)를 "
    + "HGB 5시드가 확률로 냅니다. 확률이 후행 q90 을 넘으면 켜집니다(하루 약 29회).\n\n"
    + "[재생 검증 2026-09-30] 켜졌을 때 실제로 30분 안에 급증이 온 비율(기저 23%): "
    + "2025-09~12 78.8% · 2026-01~03 78.3% · 04~09-10 76.4% · **배포 후 09-11~28 77.8%** -- 주장(78.5%)이 유지됩니다. "
    + "모델 없이 «거래대금 z + 체결속도 z» 상위 10% 는 68~73% 라 모델이 5~8%p 더 맞힙니다.\n\n"
    + "⚠️방향은 말하지 않습니다. 급증 봉의 방향으로 1시간 이어진 비율은 45~49%(동전 이하)이고, "
    + "«큰 이동이 온다»로 읽으면 거래량 없이 «봉 가격폭 상위»를 보는 쪽이 앞 1시간 이탈을 더 잘 잡았습니다(5개 구간 모두). "
    + "그래서 «곧 거래가 몰린다 -- 새 진입·지정가 대기는 조심»의 참고로만 씁니다."
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
  // 2026-09-11 경보기(예고 모델). 옛 «경보 2시간» 은 자명한 대리 타깃 값이라 교체했다.
  breakout_prewarn: { text: "예고 = 30분", title: "앞으로 30분 이내에 거래대금·체결속도가 **함께** 급증할(둘 다 후행 q90 이상) 확률이다(HGB 5시드 동결, 33피쳐). 임계는 확률의 후행 2016봉 q90 이라 인과적이다. 재생 검증: 배포 후(09-11~28) 정밀도 77.8% · 기저 23%. ⚠️방향도 «큰 이동»도 말하지 않는다 -- «거래가 몰린다»만." },
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
// 2026-09-28 지지·저항 = 청산맵 레벨, 현재가 쪽으로 이미 지나간 것은 버리고 **가까운 순 3개씩**.
//   사분면 값 칸(데스크톱)과 차트 아래 목록(모바일)이 같은 목록을 쓴다.
// 게이지 행(차트 아래 목록)을 쓰는 화면 = 모바일(≤720) 또는 세로 방향. 그 밖(가로 데스크톱)은 사분면 값 칸 아래 글자 줄.
function srGaugeMode() {
  return typeof window !== "undefined" && window.matchMedia("(max-width: 720px), (orientation: portrait)").matches;
}

function srLevelsLive(n = 3) {
  const map = latestLiquidationMap;
  if (!map || !map.warmed_up) return null;
  const cur = Number(latestLivePriceByAsset[activeSnapshotAsset] || map.current_price || 0);
  if (!(cur > 0)) return null;
  const pick = (levels, below) => (levels || []).filter((lv) => (below ? lv.price < cur : lv.price > cur)).slice(0, n);
  return { cur, res: pick(map.resistance_levels, false), sup: pick(map.support_levels, true) };
}

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

  // 2026-09-28 차트(거리 수직선)를 없앴다(사용자 «사분면 라벨 아래에 가격과 강도를 적어줘»). 데스크톱은 renderCandleSvg 가
  //   사분면 값 칸 아래에 적고 이 목록은 비운다. 모바일은 판 아래 값 한 줄뿐이라 여기에 같은 줄을 글자로 적는다.
  //   🔴서술이다 -- 청산 밀집은 추정이고 지지·저항 반등을 예측하지 않는다(벽 반등률 0.509, 09-20).
  // 2026-09-28(2차) 모바일·세로 화면은 **예전 게이지 행**으로 되돌렸다(사용자 지시) -- 이름 · 가격 · 강도 막대 · 거리 %.
  const sr = srLevelsLive();
  if (!srGaugeMode()) { setH("liquidationMapList", ""); return; }
  if (!sr || (!sr.res.length && !sr.sup.length)) {
    setH("liquidationMapList", `<p class="muted" style="padding:16px;">추정 가능한 밀집 구간이 아직 없습니다.</p>`);
    return;
  }
  const row = (lv, tag, cls) => {
    const pct = Math.round((lv.weight_pct || 0) * 100), dist = (lv.price - sr.cur) / sr.cur * 100;
    return `<div class="liq-level-row ${cls}"><span class="liq-level-tag">${tag}</span>`
      + `<span class="liq-level-price">${fmtNum(lv.price, 2)}</span>`
      + `<div class="liq-level-bar-track"><div class="liq-level-bar-fill" style="width:${Math.max(pct, 4)}%;"></div></div>`
      + `<span class="liq-level-dist">${dist > 0 ? "+" : ""}${fmtNum(dist, 2)}%</span></div>`;
  };
  const cur = `<div class="liq-level-row liq-current"><span class="liq-level-tag">현재가</span>`
    + `<span class="liq-level-price">${fmtNum(sr.cur, 2)}</span><div class="liq-level-bar-track"></div><span class="liq-level-dist">-</span></div>`;
  setH("liquidationMapList", [...sr.res.map((lv, i) => row(lv, "저항" + (i + 1), "liq-resistance")).reverse(), cur,
                              ...sr.sup.map((lv, i) => row(lv, "지지" + (i + 1), "liq-support"))].join(""));
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
    renderMacroEventAlert(data.macro_event_alert);
  } catch (error) {
    console.error("Session alerts fetch error:", error);
    const macroAlertBadge = el("macroEventAlertBadge");
    if (macroAlertBadge) macroAlertBadge.style.display = "none";
  }
}

// Macro-event (CPI/NFP/GDP/PCE/내구재/FOMC/연준 의장 발언) release-time alert (2026-08-26 follow-up) -- same
// fixed-text/tooltip-detail pattern as the (2026-09-27 제거) session-open alert, +-30min window (see
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
  const asset = activeSnapshotAsset;
  try {
    const res = await fetch(`${API_CHART_MARKERS_URL}?asset=${asset}`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`chart markers ${res.status}`);
    const data = await res.json();
    // 🔴2026-09-30 늦게 온 옛 코인 응답은 버린다(도착 전에 코인이 바뀌면 옛 코인 표식이 새 차트에 얹혔다).
    if (asset === activeSnapshotAsset) latestChartMarkers = data;
  } catch (error) {
    console.error("Chart markers fetch error:", error);
    if (asset === activeSnapshotAsset) latestChartMarkers = { available: false, error: "fetch_failed" };
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


// 2026-09-25 청산 원의 **꼬리만** 빠르게. 서버가 ETH 청산 원을 수급 1초 차트와 같은 실시간 원천에서
// 만들게 되면서(전에는 봇 DB 1분 행 + 60초 폴링 + 캐시 = 1~3분 지연) 막는 건 이 폴링 주기 하나다.
// 전량(최대 144봉 ≈ 16KB)을 2초마다 받지 않고 최신 2봉만 받아 합친다 -- 2봉인 이유는 봉 경계에서
// 늦게 도착한 청산이 **직전 봉**에 들어가기 때문이다(풋프린트 since = 최신봉-1봉 과 같은 이유).
// 전량은 아래 refreshLiquidation5mSignal 이 60초마다·창 토글마다 그대로 받는다.
const LIQ_5M_TAIL_POLL_MS = 2000;
let liquidation5mTailFetchAt = 0;
async function refreshLiquidation5mTail() {
  // 2026-09-26 서버가 전 종목 @forceOrder 로 코인마다 실시간 누적한다 -- ETH 제한을 풀었다.
  const asset = activeSnapshotAsset;
  const base = latestLiquidation5mHist;
  if (!Array.isArray(base) || !base.length) return;   // 전량이 먼저 와야 합칠 자리가 있다
  const now = Date.now();
  if (now - liquidation5mTailFetchAt < LIQ_5M_TAIL_POLL_MS) return;
  liquidation5mTailFetchAt = now;
  try {
    const res = await fetch(`${API_LIQUIDATION_5M_HIST_URL}?asset=${asset}&bars=2`, { cache: "no-cache" });
    if (!res.ok) return;
    const j = await res.json();
    if (activeSnapshotAsset !== asset || latestLiquidation5mHist !== base) return;   // 그 사이 전량이 왔으면 그쪽이 맞다
    if (!(j && j.warmed_up && Array.isArray(j.bars) && j.bars.length)) return;
    const same = (a, b) => a.long_usd === b.long_usd && a.short_usd === b.short_usd
      && a.events === b.events && Boolean(a.partial) === Boolean(b.partial)
      && Boolean(a.okx) === Boolean(b.okx) && JSON.stringify(a.hl || null) === JSON.stringify(b.hl || null);
    const out = base.slice();
    let changed = false;
    j.bars.forEach((nb) => {
      const i = out.findIndex((b) => b.ts === nb.ts);
      if (i >= 0) {
        if (!same(out[i], nb)) { out[i] = nb; changed = true; }
      } else if (Date.parse(nb.ts) > Date.parse(out[out.length - 1].ts)) {
        out.push(nb); out.shift(); changed = true;       // 새 봉 -- 창 길이를 유지한다
      }
    });
    // 🔴바뀐 게 없으면 **같은 배열**을 둔다. 청산 원 층은 배열 신원(objToken)으로 캐시되므로
    //   2초마다 새 배열을 만들면 내용이 같아도 매번 다시 그린다.
    if (changed) latestLiquidation5mHist = out;
  } catch (e) { /* 다음 2초에 다시 -- 전량 폴링이 따로 돈다 */ }
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
      // 🔴2026-09-25 폭을 보낸다. 안 보내면 서버가 96 봉으로 답해 12시간 창에서 앞 48봉에
      //   청산 원이 사라졌다 -- 화면에서는 «청산이 없었다»로 읽힌다(실제로는 모름).
      //   풋프린트·수급프로파일과 같은 규약(`?bars=${chartWindowBars}`).
      const rh = await fetch(`${API_LIQUIDATION_5M_HIST_URL}?asset=${asset}&bars=${CHART_PAN_BARS}`,
                             { cache: "no-cache" });
      const jh = await rh.json();
      if (asset !== activeSnapshotAsset) return;
      latestLiquidation5mHist = (jh && jh.warmed_up && Array.isArray(jh.bars)) ? jh.bars : [];
    } catch (e) { if (asset === activeSnapshotAsset) latestLiquidation5mHist = []; }
  } catch (error) {
    console.error("Liquidation 5m signal fetch error:", error);
    if (asset === activeSnapshotAsset) latestLiquidation5m = { warmed_up: false, error: "fetch_failed" };
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
    macroCalendarLastFetchAt = now - MACRO_CALENDAR_POLL_MS + 60 * 1000;   // 2026-09-30 실패하면 6시간 기다리지 않고 1분 뒤 다시(배포 재시작 중 실패로 옛 목록에 굳었다)
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
  // 2026-09-30 경제 일정 카드 제거(사용자 지시) -- 시장 맥락 ⑤ 시간축이 이 값을 그린다.
  latestMacroEvents = payload && Array.isArray(payload.events) ? payload.events : [];
  macroCalendarOkAt = Date.now();
  if (typeof renderMarketCtx === "function") renderMarketCtx();   // ⑤ 시간축을 바로 다시 그린다
}


function setupPageTabs() {
  document.querySelectorAll(".page-tab").forEach((button) => button.addEventListener("click", () => {
    const target = button.dataset.pageTab; // "ops" | "snapshot" | "notify" (라이브 탭 제거, 2026-08-31)
    activePageTab = target;
    el("opsTabPanel")?.classList.toggle("hidden", target !== "ops");
    el("snapshotTabPanel")?.classList.toggle("hidden", target !== "snapshot");
    el("notifyTabPanel")?.classList.toggle("hidden", target !== "notify");
    cardRailSync();   // 2026-09-30 카드 이동 레일도 스냅샷 탭에서만
    ofabSync();   // 2026-09-25 떠다니는 주문 버튼은 스냅샷 탭에서만 -- 떠나면 조작부를 카드로 먼저 돌려놓는다
    document.querySelectorAll(".page-tab").forEach((tab) => {
      tab.classList.toggle("active", tab === button);
      if (tab === button) tab.setAttribute("aria-current", "page"); else tab.removeAttribute("aria-current");
    });
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
      liquidationMapLastFetchAt = 0; refreshLiquidationMap();
      regimeWide24LastFetchAt = 0; refreshRegimeWide24();
      regimeBtcLastFetchAt = 0; refreshRegimeBtc();
      regimeXrpLastFetchAt = 0; refreshRegimeXrp();
      macroCalendarLastFetchAt = 0; refreshMacroCalendar();
      sessionAlertsLastFetchAt = 0; refreshSessionAlerts();
      lastSnapshotHistoryFetchAt = 0; maybeFetchSnapshotChartHistory();
    }
  }));
}

// 2026-09-30 카드 이동 레일(사용자 지시 «점만»): 점을 누르면 그 카드로 스크롤 · 지금 보는 카드의 점이 채워진다. 스냅샷 탭에서만.
function cardRailSync() {
  const rail = el("cardRail");
  if (!rail) return;
  rail.hidden = activePageTab !== "snapshot";
  if (rail.hidden) return;
  const y = innerHeight * 0.35;
  let cur = null;
  rail.querySelectorAll("[data-go]").forEach((b) => { const t = el(b.dataset.go); if (t && t.getBoundingClientRect().top < y) cur = b; });
  cur = cur || rail.querySelector("[data-go]");
  rail.querySelectorAll("[data-go]").forEach((b) => { const on = b === cur; b.classList.toggle("on", on); if (on) b.setAttribute("aria-current", "true"); else b.removeAttribute("aria-current"); });
}
// 2026-09-30 «한 화면 모드»(사용자 1920×1080·90%): 넓은 화면에서 카드 하나 = 창 높이 하나 -- 레일 점/스크롤이 카드 단위로 딱 맞게 넘어간다.
//   Footprint 는 차트 상자 높이를, Option 은 행사가 사다리 높이를 «창 높이 − 카드의 나머지»로 계산한다. 계좌는 창 높이까지 늘린다.
//   창 높이 < 900 이면 끈다(가격판이 너무 좁아진다). 켜기/끄기 = 레일 아래 단추(브라우저 기억).
let fitOptLadderH = null, fitOptChartK = FIT0.ck || 1, fitOptCurveK = FIT0.cc || 1, renderOptionsSoon = false, fitAcctPlotH = FIT0.ap || 0;   // 2026-10-01 Option 차트 셋 높이 배율 · 계좌 성과 차트 높이(한 화면 모드, fitLayout 이 잰다)
const optColW = () => {   // .opt-ladder 최대 폭 440 과 같게(넓게 그리면 줄어 글자가 작아진다) · 첫 렌더(칸이 아직 없음)는 카드 폭의 가운데 열 몫으로
  const s = document.querySelector('#optCard .opt-sec:has([data-tip="ladder"])');
  if (s && s.clientWidth > 240) return Math.min(640, Math.round(s.clientWidth));   // 2026-10-01 440 -> 640(사다리 열을 넓혀 막대를 길게)
  const card = el("optCard"), guess = card && innerWidth >= 1100 ? Math.round(((card.clientWidth - 56) * 0.95) / 3.25) : 0;
  return guess > 240 ? Math.min(640, guess) : 336;
};
function fitOn() {
  let on = true; try { on = localStorage.getItem("fit1") !== "0"; } catch (e) { /* 기억은 편의 */ }
  return on && innerWidth >= 1700 && innerWidth > innerHeight && innerHeight >= 900 && activePageTab === "snapshot";
}
function fitLayout() {
  // 2026-09-30(2) 측정해서 고치는 방식을 버렸다(새로고침마다 두세 번 크기가 바뀌었다): 풋프린트 차트 칸은 CSS(html.fit1 — 카드 = 창 높이, 차트 칸 = 남는 공간)가
  //   첫 배치부터 정하고, 행사가 사다리는 창 높이 공식 하나로 정한다. 여기서는 모드 켜기/끄기와 계좌·Option 최소 높이만.
  const on = fitOn(), vh = innerHeight - 20;
  document.documentElement.classList.toggle("fit1", on);
  const acct = el("acctCard"), opt = el("optCard");
  if (acct) acct.style.minHeight = on ? `${vh}px` : "";
  if (opt) opt.style.minHeight = on ? `${vh}px` : "";
  // 2026-10-01(2) Option 칸 높이 = **원래 크기로 매번 계산**(되먹임 없음). 앞 판은 넘칠 때마다 사다리를 깎는 톱니라 한 번 300 에 닿으면
  //   다시 안 커졌고(캐시에도 남음) 차트 상한만 쌓여 «전부 작아지고 바닥은 잘렸다». 이제:
  //   ① 각 칸 = 머리·꼬리(고정) + 그림(원래 높이 = 폭 × viewBox 비율) ② 원래 크기로 들어가면 상한 없음 + 사다리가 1·2줄을 채움
  //   ③ 안 들어가면 만기·흐름·감마곡선에 들어갈 만큼만 상한(이분 탐색). 바닥은 풋프린트처럼 12px 여유.
  let h = on ? (fitOptLadderH || fitLadderFormula()) : null;
  // 화면 밖 카드는 재지 않는다 -- 밖에 있는 동안 잰 값이 틀어져(실측 k 0.6 · 사다리 1200) 돌아올 때 크기가 튀었다
  const inView = (c) => { const r = c.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight; };
  if (on && opt && opt.offsetParent && inView(opt)) {
    const q = (sel) => opt.querySelector(sel), sec = (t) => q(`.opt-sec:has([data-tip="${t}"])`);
    const lane = q("#optLaneBody"), flow = q("#optFlowBody"), blk = q("#optBlkBody"), curve = sec("curve"), lad = sec("ladder"), move = sec("move"), when = q(":scope > #mcWhen");
    const box = (c) => c.getBoundingClientRect(), padB = (c) => parseFloat(getComputedStyle(c).paddingBottom) || 0;
    const fix = (c) => {   // [머리·꼬리 고정 높이, 그림 원래 높이]
      const g = c && c.querySelector("svg");
      if (!g) return c ? [Math.max(0, ...[...c.children].map((k) => box(k).bottom)) - box(c).top + padB(c), 0] : [0, 0];
      const gb = box(g), last = Math.max(gb.bottom, ...[...c.children].map((k) => box(k).bottom)), vb = g.viewBox && g.viewBox.baseVal;
      return [gb.top - box(c).top + (last - gb.bottom) + padB(c), vb && vb.width ? (g.clientWidth * vb.height) / vb.width : gb.height];
    };
    if (lane && flow && blk && curve && lad && move) {
      // 2026-10-01(4) 배치(사용자 지시): 위 = 지금·딜러감마 | 사다리 | 블록 거래·심리 · 아래 한 줄 = 만기 | 순매수 흐름 | 감마 곡선(조금 낮게).
      //   아래 줄 차트 셋은 같은 배율 k(기본 0.85, 남는 칸의 34% 이내) · 사다리는 위 블록을 채운다. 원래(k=1) 높이 = 지금 높이 / 지금 k.
      const k0 = optChartK(), kc0 = optCurveK(), [la, ln] = fix(lane), [fa, fn] = fix(flow), [ca, cn] = fix(curve), [bn] = fix(blk);
      const lnB = ln / k0, fnB = fn / k0, cnB = cn / kc0;
      const bottom = (k) => Math.max(la + k * lnB, fa + k * fnB, ca + k * cnB);
      const whenH = when && when.offsetParent ? box(when).height + (parseFloat(getComputedStyle(when).marginTop) || 0) : 0;
      const avail = box(opt).top + vh - padB(opt) - 12 - whenH - box(move).top;
      const topNeed = bn + 130;   // 블록 거래 목록 + 심리 세 줄(최소) -- 위 블록이 이보다 작으면 넘친다
      let lo = 0.5, hi = 0.95;   // 2026-10-01 0.85 -> 0.95 · 34% -> 38%(사용자 «조금만 더 키워줘»)
      if (bottom(hi) <= Math.min(avail * 0.38, avail - topNeed)) lo = hi;
      else while (hi - lo > 0.01) { const m = (lo + hi) / 2; if (bottom(m) <= Math.min(avail * 0.38, avail - topNeed)) lo = m; else hi = m; }
      const k = Math.round(lo * 50) / 50;   // 0.02 단위 -- 1px 흔들림에 다시 그리지 않게
      if (Math.abs(k - fitOptChartK) > 0.019) { fitOptChartK = k; renderOptionsSoon = true; }
      if (Math.abs(k - fitOptCurveK) > 0.019) { fitOptCurveK = k; renderOptionsSoon = true; }
      const lg = lad.querySelector("svg");
      if (lg) {
        const [lx] = fix(lad), want = Math.max(200, avail - bottom(k) - lx), gh = box(lg).height;
        if (gh > 0 && Math.abs(want - gh) > 4) h = Math.max(200, Math.min(1400, Math.round((h * want) / gh)));
      }
    }
  }
  // 2026-10-01 계좌 카드: 한 화면 모드면 성과 차트를 펼쳐 남는 높이를 그 차트에 준다(사용자 «내 계좌도 화면에 가득»)
  const plot = on && acct && acct.offsetParent && inView(acct) ? acct.querySelector("#snapAcctPerf .acct-plot svg") : null;
  if (plot) {
    const ab = Math.max(0, ...[...acct.children].filter((k) => k.offsetParent).map((k) => k.getBoundingClientRect().bottom));
    const ar = acct.getBoundingClientRect(), aslack = Math.round(ar.top + vh - parseFloat(getComputedStyle(acct).paddingBottom || 0) - 12 - ab);   // 바닥 여유 12 = 풋프린트와 같게
    const cur = plot.getBoundingClientRect().height;
    if (Math.abs(aslack) > 4) { fitAcctPlotH = Math.max(120, Math.min(900, Math.round(cur + aslack))); acct.style.setProperty("--acctplot", `${fitAcctPlotH}px`); }
  }
  if (h !== fitOptLadderH || renderOptionsSoon) { fitOptLadderH = h; renderOptionsSoon = false; if (typeof renderOptions === "function") renderOptions(); }
  if (on) { try { localStorage.setItem("fitCache", JSON.stringify({ vh: innerHeight, mc: mcNeedH, lad: fitOptLadderH, ck: fitOptChartK, cc: fitOptCurveK, ap: fitAcctPlotH })); } catch (e) { /* 기억은 편의 */ } }
}
// Option 격자: 머리·커버 줄(~130) 아래 여섯 줄 중 사다리가 넷(칸 머리·칩 ~70 을 빼고) -- 측정 없이 창 높이로.
// 2026-10-01 ⑤ 다음 24시간(116 + 여백·선 36)이 맨 아래 전폭으로 들어와 그만큼 뺀다.
function fitLadderFormula() { return Math.max(300, Math.min(900, Math.round(((innerHeight - 20 - 130 - 152) * 4) / 6 - 70))); }
// 첫 렌더 전에 모드 클래스와 최소 높이를 바로 입힌다 -- 데이터가 오기 전에 상자 높이가 이미 맞아 있다
function fitApplyCached() {
  if (!fitOn()) return;
  document.documentElement.classList.add("fit1");
  fitOptLadderH = FIT0.lad || fitLadderFormula();   // 지난번에 잰 값(같은 창 높이) -- 새로고침 첫 그림부터 맞는다
  ["acctCard", "optCard"].forEach((id) => { const c = el(id); if (c) c.style.minHeight = `${innerHeight - 20}px`; });
  if (fitAcctPlotH) el("acctCard")?.style.setProperty("--acctplot", `${fitAcctPlotH}px`);
}
function setupCardRail() {
  fitApplyCached();
  const rail = el("cardRail");
  if (!rail) return;
  rail.addEventListener("click", (e) => { const b = e.target.closest("[data-go]"); if (b) el(b.dataset.go)?.scrollIntoView({ behavior: "smooth", block: "start" }); });
  addEventListener("scroll", () => requestAnimationFrame(cardRailSync), { passive: true });
  cardRailSync();
  addEventListener("resize", () => requestAnimationFrame(fitLayout));
  setInterval(fitLayout, 1500);
  // 2026-09-30 ⑤ 시간축은 시장 맥락 응답이 없어도 1분마다 다시 그린다(«지금» 기준이 흐르고 지난 일정이 빠진다) -- 멈춘 채 어제 모습으로 남던 것
  setInterval(() => { if (activePageTab === "snapshot" && !document.hidden && typeof renderMarketCtx === "function") renderMarketCtx(); }, 60 * 1000);   // 데이터가 오며 카드 높이가 바뀐다 -- 1.5초마다 다시 잰다(값이 같으면 아무것도 안 한다)
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
// 2026-09-22 시나리오 목표는 차트의 **가로 위치**다 -- y 축이 이미 가격축이다(사용자).
// nearestLiquidationLevel 과 같은 모양을 돌려주면 priceLabels 가 화면 밖 클램핑·겹침 회피·
// 좌측 라벨·우측 가격 배지를 전부 해 준다(새로 그리는 코드 0).
// 🔴풋프린트에서는 선이 아니라 삼각형이다 -- 가로선이 가격 행을 가로질러 셀 숫자를 덮는다(2026-09-16).
// 🔴색은 --muted. S/R·현재가는 **사실**, 이 선들은 **예측**이라 더 조용해야 한다. 형태도 «비운 배지»(잠정)이고
//   테두리 안을 확률만큼 채운다 -- 배지 자체가 게이지다(시안 Q).
// 2026-09-25 옛 A/B/C 기하 목표를 걷고 **융합 3결과의 두 선**(±0.5×30분 폭 — 위 먼저 · 아래 먼저)으로 바꿨다.
//   배지 숫자 = 카드와 같은 실측 확률. «미도달»은 선이 없다(두 선 사이에 머묾).
// 2026-09-26 사용자 지시(B): 청산맵은 **완성된 1시간봉**까지만 쓸림을 반영한다(서버가 형성 중 봉을 뺀다) --
//   마지막 스냅샷 ts 는 그 봉의 **시작**이라 서버가 아는 건 ts+1h 까지다. 그 뒤 가격이 지나간 레벨은
//   최대 1시간 «안 쓸린 벽»으로 남았다. 차트 5분봉(현재 봉은 틱으로 매초 갱신)으로 그 공백을 메운다.
//   반환: 가격 -> 처음 지나간 봉 인덱스(없으면 -1). 봉의 [저가, 고가] 가 가격을 품으면 지나간 것이다.
function liqSweepIdx(candles, sinceSec, price) {
  for (let i = 0; i < candles.length; i++) {
    const c = candles[i];
    if (c.time >= sinceSec && c.low <= price && price <= c.high) return i;
  }
  return -1;
}
function liqMapKnownUntilSec(map) {
  const h = (map && map.heatmap_history) || [];
  const t = h.length ? Date.parse(h[h.length - 1].ts_utc) : NaN;
  return Number.isFinite(t) ? Math.floor(t / 1000) + 3600 : Infinity;   // 모르면 아무것도 안 지운다
}

function nearestLiquidationLevel() {
  const map = latestLiquidationMap;
  if (!map || !map.warmed_up) return [];
  const liveCurrentPrice = Number(latestLivePriceByAsset[activeSnapshotAsset] || map.current_price || 0);
  if (!(liveCurrentPrice > 0)) return [];
  const since = liqMapKnownUntilSec(map);
  const candles = candleHistoryByAsset[activeSnapshotAsset] || [];
  // 목록은 가까운 순이다 -- 현재가 쪽에 있고 **아직 안 쓸린** 첫 레벨을 고른다.
  const pick = (levels, below) => (levels || []).find((lv) => Number(lv.price) > 0
    && (below ? lv.price < liveCurrentPrice : lv.price > liveCurrentPrice)
    && liqSweepIdx(candles, since, Number(lv.price)) < 0);
  const candidates = [
    { lv: pick(map.support_levels, true), color: "var(--liq-support)", tag: "지지1", side: "support" },
    { lv: pick(map.resistance_levels, false), color: "var(--liq-resistance)", tag: "저항1", side: "resistance" },
  ]
    .filter((c) => c.lv);
  if (!candidates.length) return [];
  candidates.sort((a, b) => Math.abs(a.lv.price - liveCurrentPrice) - Math.abs(b.lv.price - liveCurrentPrice));
  const nearest = candidates[0];
  return [{
    val: nearest.lv.price,
    color: nearest.color,
    label: nearest.tag,
    priceLeft: true,   // 2026-09-26 사용자 지시: 지지/저항 가격은 오른쪽 배지가 아니라 왼쪽 이름 옆에
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
  if (!(line || tri)) return;                  // 아직 안 그렸거나 현재가 표시가 없는 판
  if (line) { line.setAttribute("y1", y); line.setAttribute("y2", y); }
  if (tri) tri.setAttribute("points", markerPoints(c.markerX, y));
  const labelY = Math.max(c.mt + 9, Math.min(c.mt + c.ch - 9, y));
  const labBg = c.svg.querySelector('[data-live="labelbg"]');   // 2026-09-28 풋프린트 현재가 태그 바탕
  if (lab) lab.setAttribute("y", labelY + (labBg ? 4.5 : 4));
  if (lab && lab.dataset.withPrice) lab.textContent = (lab.dataset.withPrice === "num" ? "" : "현재 ") + fmtNum(price, pxDp());   // 2026-09-27 풋프린트: 가격이 왼쪽 글자 안
  if (labBg && lab) {
    labBg.setAttribute("y", labelY - 9);
    try { const bw1 = lab.getComputedTextLength() + 10; labBg.setAttribute("width", bw1); if (labBg.dataset.right) labBg.setAttribute("x", Number(labBg.dataset.right) - bw1); } catch (e) { /* 비렌더 */ }
  }
  // 2026-09-22 모바일에는 배지가 없다(값은 플롯 아래 한 줄에 있다). 옛 판은 box/txt 가
  //   없으면 **여기서 return** 해서 화살표까지 같이 멈췄다 -- 전체 렌더(1초)까지 어긋난다.
  const row = c.svg.querySelector('[data-live="rowtext"]');
  if (row) row.textContent = (row.textContent.split(" ")[0] || "현재") + " " + fmtNum(price, pxDp());
  if (!box || !txt) return;
  box.setAttribute("y", labelY - 9);
  txt.setAttribute("y", labelY + 4);
  txt.textContent = fmtNum(price, pxDp());
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
  if (!(footprintLive.bucket > 0)) return;   // 코인을 막 바꿨다 -- 서버가 칸 폭을 알려 줄 때까지 기다린다
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
  // 🔴빈 셀로 덮으면 봉이 **사라진다**. footprintForChart 가 levels.length 0 인 봉을 걸러내기
  //   때문이다. 봉이 바뀌는 순간 cells 는 비어 있고(footprintLiveAdd 가 롤오버에서 비운다)
  //   barStart 는 체결이 와야 넘어가므로, 그 틈에 이 함수가 서버의 멀쩡한 봉을 빈 배열로
  //   갈아치웠다. 이 함수는 «더 나은 값으로 교체»할 때만 의미가 있다.
  if (!footprintLive.cells.size) return;
  const merged = new Map([...footprintLive.cells.entries()].map(([k, c]) => [k, c.slice(0, 6)]));
  // 2026-09-24 서버가 준 **OKX 몫**을 더한다(`okxLive`, 같은 봉일 때만). 빼면 이 봉만 바이낸스
  //   단독이라 ~2/3 로 그려지다 마감 때 합산본으로 튄다. OKX 쪽만 폴링 주기만큼 늦다.
  const okx = latestFootprint && latestFootprint.okxLive;
  if (okx && okx.time === bar) {
    (okx.levels || []).forEach((l) => {
      const k = Math.round(l[0] / bucket);
      const c = merged.get(k) || [0, 0, 0, 0, 0, 0];
      for (let j = 0; j < 6; j++) c[j] += Number(l[j + 1]) || 0;
      merged.set(k, c);
    });
  }
  byTime.set(bar, [...merged.entries()]
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
  // 🔴2026-09-22 `since` 는 «이 연결이 언제부터 봤나»여야 하는데, 한 번(=== Infinity)만
  //   설정되고 어디서도 안 돌아갔다. 그래서 끊겼다 붙어도 최초 연결 시각 그대로였고,
  //   아래 footprintMergeLive 의 «봉 중간에 붙었다» 가드가 **재연결에는 안 먹었다** --
  //   끊겨 있던 구간이 빠진 셀로 서버 봉을 덮어써서 5분봉이 깜빡였다.
  //   가격 WS 는 document.hidden 이면 닫히므로(바로 위 want), 다른 탭 갔다 오면 매번 그랬다.
  footprintLive.since = Infinity;
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
        if (qty > 0 && FLOW_ASSETS.has(priceWsAsset) && priceWsAsset === activeSnapshotAsset) {
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
  if (!flowOn()) return;                                         // 흐름 엔진이 있는 코인만(FLOW_ASSETS)
  const now = Date.now();
  if (liveStreamOn()) return;                                    // /api/stream 이 밀어주는 중
  if (now - supply1sLastFetchAt < SUPPLY_1S_POLL_MS || supply1sInFlight) return;
  supply1sLastFetchAt = now;
  supply1sInFlight = true;
  try {
    const asset = activeSnapshotAsset;
    const res = await fetch(`${API_SUPPLY_1S_URL}?asset=${asset}&since=${supply1sSince}&sinceOi=${oi1sSince}`
                            + `&sinceLiq=${liq1sSince}&sinceOkx=${okxSupply1sSince}`
                            + `&sinceOkxOi=${okxOi1sSince}&sinceOkxLiq=${okxLiq1sSince}`
                            + `&sinceSpot=${spotSupply1sSince}`,
                            { cache: "no-cache" });
    if (!res.ok) throw new Error(`supply-1s ${res.status}`);
    const j = await res.json();
    if (asset !== activeSnapshotAsset) return;     // 그 사이 코인이 바뀌었다 -- 남의 초를 얹지 않는다
    applySupply1s(j);
  } catch (error) {
    console.error("Supply 1s fetch error:", error);
  } finally {
    supply1sInFlight = false;
  }
  // 받은 즉시 **이 패널만** 다시 그린다. 캔들 SVG 전체를 다시 그리지 않으므로 비싼 패스
  // (캔들·청산밀도·프로파일)는 안 탄다 -- 호버/스크롤 게이트에도 안 걸린다.
  repaintSupply1sPanel();
}

function resetSupply1sState() {
  supply1s = new Map(); oi1s = new Map(); liq1s = new Map();
  okxSupply1s = new Map(); okxOi1s = new Map(); okxLiq1s = new Map(); spotSupply1s = new Map();
  supply1sSince = 0; oi1sSince = 0; liq1sSince = 0;
  okxSupply1sSince = 0; okxOi1sSince = 0; okxLiq1sSince = 0; spotSupply1sSince = 0;
  supply1sMeta = { retailMaxUsd: 0, whaleMinUsd: 0, now: 0 };
  okxMeta = { now: 0, connected: false, tradeAge: null, oiAge: null, inst: "", errors: 0 };
  spotMeta = { now: 0, connected: false, tradeAge: null, errors: 0 };
  supply1sLastFetchAt = 0;
  supply1sVer += 1;
}

// 받은 수급 한 덩이를 칸·커서에 얹는다. 폴링과 /api/stream 이 **같은 함수**를 지난다.
function applySupply1s(payload) {
  (payload.seconds || []).forEach((r) => {
    supply1s.set(r[0], r.slice(1));
    if (r[0] > supply1sSince) supply1sSince = r[0];
  });
  (payload.liq || []).forEach((r) => {
    liq1s.set(r[0], [r[1], r[2], r[3], r[4]]);   // [롱수량, 숏수량, 롱USD, 숏USD]
    if (r[0] > liq1sSince) liq1sSince = r[0];
  });
  (payload.oi || []).forEach((r) => {
    oi1s.set(r[0], r[1]);
    if (r[0] > oi1sSince) oi1sSince = r[0];
  });
  // OKX 레인. 커서가 각자인 이유는 바이낸스 OI/청산이 각자인 것과 같다 -- 거래소마다
  // 체결 초가 앞서가므로 하나를 공유하면 뒤처진 쪽이 통째로 건너뛰어진다.
  (payload.okx || []).forEach((r) => {
    okxSupply1s.set(r[0], r.slice(1));
    if (r[0] > okxSupply1sSince) okxSupply1sSince = r[0];
  });
  (payload.okxLiq || []).forEach((r) => {
    okxLiq1s.set(r[0], [r[1], r[2], r[3], r[4]]);
    if (r[0] > okxLiq1sSince) okxLiq1sSince = r[0];
  });
  (payload.okxOi || []).forEach((r) => {
    okxOi1s.set(r[0], r[1]);
    if (r[0] > okxOi1sSince) okxOi1sSince = r[0];
  });
  (payload.spot || []).forEach((r) => {
    spotSupply1s.set(r[0], r.slice(1));
    if (r[0] > spotSupply1sSince) spotSupply1sSince = r[0];
  });
  // 진행 중인 초 -- 칸만 채우고 **커서(since)는 안 옮긴다**. 그래야 닫힌 뒤 확정본이 온다.
  const part = payload.partial || {};
  [[part.bn, supply1s], [part.okx, okxSupply1s], [part.spot, spotSupply1s]].forEach(([r, m]) => {
    if (r) m.set(r[0], r.slice(1));
  });
  okxMeta = Object.assign({}, payload.okxMeta || {},
                          { now: Number(payload.okxNow) || okxMeta.now });
  spotMeta = Object.assign({}, payload.spotMeta || {},
                           { now: Number(payload.spotNow) || spotMeta.now });
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
  const okxFloor = (okxMeta.now || 0) - 2 * SUPPLY_1S_SEGMENT - 20;
  okxSupply1s.forEach((_v, k) => { if (k < okxFloor) okxSupply1s.delete(k); });
  okxOi1s.forEach((_v, k) => { if (k < okxFloor) okxOi1s.delete(k); });
  okxLiq1s.forEach((_v, k) => { if (k < okxFloor) okxLiq1s.delete(k); });
  const spotFloor = (spotMeta.now || 0) - 2 * SUPPLY_1S_SEGMENT - 20;
  spotSupply1s.forEach((_v, k) => { if (k < spotFloor) spotSupply1s.delete(k); });
  oi1s.forEach((_v, k) => { if (k < floor) oi1s.delete(k); });
  liq1s.forEach((_v, k) => { if (k < floor) liq1s.delete(k); });
  supply1sVer += 1;
}

// ── 밀어주기 스트림 (2026-09-24 속도 2단계) ─────────────────────────────────
// 수급(ETH)·상황 카드를 폴링 대신 /api/stream 하나로 받는다. 서버가 **클라별 커서**를 들고
// 0.25초마다 «그 뒤»를 보낸다 -- 요청 왕복(클라우드플레어 경유 ~100ms)과 폴링 대기가 없어진다.
// 스트림이 조용해지면(마지막 메시지 3초 전) 위 폴링이 그대로 대신한다 -- 폴링을 안 지운 이유.
const API_STREAM_URL = "/api/stream";
let liveStream = null, liveStreamKey = "", liveStreamAt = 0, liveStreamOpenedAt = 0, liveStreamRetryAt = 0;
// 폴링을 쉬는 건 **메시지를 실제로 받고 있을 때만**. 연 시각으로 치면 중간(프록시)이 스트림을
// 붙잡아 두는 환경에서 여는 순간마다 3초씩 비는 구멍이 생긴다.
const liveStreamOn = () => liveStream !== null && Date.now() - liveStreamAt < 3000;

function ensureLiveStream() {
  const want = activePageTab === "snapshot" && !document.hidden;
  const key = want ? (flowOn() ? activeSnapshotAsset : "other") : "";   // 코인이 바뀌면 다시 연다(커서가 코인별)
  if (liveStream && (key !== liveStreamKey
                     || Date.now() - Math.max(liveStreamAt, liveStreamOpenedAt) > 10000)) {
    liveStream.close(); liveStream = null;           // 탭·코인·가시성이 바뀌었거나 10초 침묵
  }
  if (!key || liveStream || Date.now() < liveStreamRetryAt) return;
  liveStreamKey = key;
  // 수급은 ETH 만 수집한다. 커서는 **지금 가진 것**으로 -- 전량은 첫 연결 한 번뿐이다.
  const q = key === "other" ? "" :
    `?supply=1&asset=${key}&since=${supply1sSince}&sinceOi=${oi1sSince}&sinceLiq=${liq1sSince}`
    + `&sinceOkx=${okxSupply1sSince}&sinceOkxOi=${okxOi1sSince}&sinceOkxLiq=${okxLiq1sSince}`
    + `&sinceSpot=${spotSupply1sSince}`;
  const es = new EventSource(API_STREAM_URL + q);
  liveStream = es; liveStreamAt = 0; liveStreamOpenedAt = Date.now();
  es.addEventListener("supply", (ev) => {
    if (es !== liveStream) return;                   // 닫힌(옛 코인) 스트림의 늦은 메시지
    liveStreamAt = Date.now();
    try { applySupply1s(JSON.parse(ev.data)); } catch (e) { console.error("stream supply:", e); return; }
    repaintSupply1sPanel();
  });
  es.addEventListener("situation", (ev) => {
    if (es !== liveStream) return;
    liveStreamAt = Date.now();
    try { latestSituation = JSON.parse(ev.data); } catch (e) { return; }
    renderSituation();
  });
  // 🔴브라우저 자동 재연결은 **처음 URL(옛 커서)** 로 붙는다 -- 닫고 5초 뒤 현재 커서로 연다
  //   (막힌 환경에서 틱마다 실패를 반복하지 않게. 그동안은 폴링이 받는다).
  es.onerror = () => {
    if (liveStream === es) { es.close(); liveStream = null; liveStreamRetryAt = Date.now() + 5000; }
  };
}

// ── 옵션 카드 · 차트 머리 칩 · 가격축 표시 (2026-09-28 사용자 선택 A+C) ───────────────────────────────
// 원천: /api/gex 한 요청 = Deribit 옵션 통합 수집기(scripts/live_deribit_block_trade_collector_20260928.py, duckdb 하나)가
//   10분마다 쓰는 체인 요약(currencies[ETH|BTC].options) + 20초마다 쓰는 블록 거래(block_trades).
// 🔴전부 참고 · 신호 아님. 우리 검정: 방향 예측은 전부 기각(GEX 방향 = 가격수준 사본 · 행사가 자석·핀닝 없음 ·
//   DVOL 방향 0). 남은 쓸모는 «얼마나 움직일 것 같은가»(크기)와 «언제 큰 이벤트가 있는가»(일정)다.
//   옛 «신호» 카드의 GEX 한 줄(gexIndicatorItem)은 이 카드의 «딜러 감마» 칸으로 들어왔다.
const OPT_TIPS = {
  move: "옵션 가격에 들어 있는 기대 움직임. 1σ = 가격 × DVOL × √(기간/1년). 정규분포라면 68% 가 이 안이지만 2026년 실측은 종가 기준 5분 83% · 1시간 84% · 24시간 77% 가 안에 들었다(평균 크기는 맞고, 대부분은 띠보다 작게 움직이다 가끔 크게 벗어난다 — 3σ 넘는 움직임이 정규분포의 약 6배). 봉 중 한 번이라도 띠를 건드린 비율은 5분 33% · 1시간 38%. 시간대 차이도 크다: 04~11 UTC 는 띠가 넉넉하고(안 90% 근처) 13~15 UTC 미국장 초반은 빠듯하다(안 68% 근처). 손절이 5분 1σ 보다 훨씬 좁으면 소음에 걸리기 쉽다.\n차트 띠는 봉 시가에 고정된다: 5분 띠 = 이번 5분봉 시가 ± 5분 1σ, 1시간 띠 = 이번 정시 첫 봉 시가 ± 1시간 1σ(폭도 봉이 열릴 때 값으로 고정). 가격이 띠를 벗어나면 옵션시장 예상보다 큰 움직임.\nIV−실현(VRP) = 30일 내재(DVOL) − 지난 7일 실현(1시간봉): 기간이 달라 30일 실현으로 재면 부호가 34% 다르다. 양수면 옵션시장이 최근보다 큰 움직임을 값에 넣는 중(보통 양수). 어느 기간으로 재도 앞으로의 VRP 예측력은 없었다.\n우리 검정(2026년만): DVOL 24시간 1σ 안에 실제로 든 날 77% · 크기 비 1.00(과거 24h 실현변동성 폭은 1.32 로 좁다). 평균 크기는 맞고 꼬리는 두껍다 — 벗어나는 23% 는 크게 벗어난다.\nVoV = 내재 변동성(DVOL)의 시간당 로그 변화 흔들림, 지난 24시간. ✅검증: 크면 다음 24시간 DVOL 이 더 크게 바뀐다(271일) · VoV 상위 20% 날은 24시간 1σ 띠 적중 66%(평소 80%) — 단 «최근 실현 변동성이 IV 보다 크다»와 같은 정보.\n꼬리 확률 = 옵션 가격(스마일)을 행사가로 미분한 위험중립 확률로, 만기에 선도가에서 2·3·5% 넘게 벗어날 확률(위·아래 합, Breeden·Litzenberger 1978). 🟡실제보다 크게 나오는 방향(±2% 예측 20% vs 실제 18% · ±3% 10% vs 7%)이나 46만기로는 확정 못 함. DVOL 로 환산한 확률(±2% 를 +15pp 과대)보다는 확실히 실제에 가깝다.",
  gamma: "딜러·체결 기준(2026-09-29 사용자 지시): 테이커의 반대편(메이커)을 딜러로 보고, 종목의 **첫 체결부터** 누적한 −(테이커 순매수)를 딜러 순포지션으로 써서 감마·플립·DEX 를 낸다. 2026-10-01 부터 history.deribit.com 으로 09-27 전 상장 종목의 체결까지 채워, 체결 이력이 1번부터 끊김 없는 종목을 «커버»로 센다(카드 맨 위 커버 % = 그 종목들의 미결제 비중). «설명 X%» = 딜러 순포지션 크기(Σ|테이커 순수량|)가 미결제의 몇 %인가 — 3,023 종목 실측 평균 56%: 나머지는 고객끼리 거래했거나 딜러가 테이커로 들어온 몫이라 체결로는 안 보인다. 100% 를 넘으면 «메이커 = 딜러» 가정이 그 범위에서 깨졌다는 뜻. 🔴관행 가정(딜러 = 콜 매수·풋 매도)과는 전 만기 감마 부호가 9~21%만 같다 — 어느 쪽이 맞는지 확인할 방법이 없다(추정). 델타는 Deribit 표준(프리미엄 미조정 BS 선도 델타).\n사다리 칩(가까운 만기 · 7일 안 · 전 만기)과 같은 범위. 금액 = 가격 1% 움직임당 딜러가 사고팔 금액. 양감마 = 오르면 팔고 내리면 사서 움직임을 누른다 · 음감마 = 따라 사고팔아 키운다. 플립 = 부호가 바뀌는 가격. DEX = 딜러 순델타 × 가격, «1시간» = 1시간 전 대비(그 사이 커버가 바뀌면 «커버 변화»). charm = 가격·IV가 그대로여도 시간만 1시간 흐를 때 딜러 델타가 얼마나 변하나 — 딜러는 그만큼 반대로 헤지한다(«헤지 매도» = 딜러가 선물을 판다). 만기가 가까울수록 커진다(Deribit 매일 08:00 UTC).\nDEX 와 charm 은 다른 양이다 — DEX = 딜러 옵션 델타의 «지금 크기»(잔고) · charm = 앞으로 1시간 «변하는 양»(이번 시간 출금). 예: DEX +$18k · charm «헤지 매수 $2k (딜러 델타 −$2k)» → 딜러는 지금 선물 $18k 숏으로 중립을 맞춰 둔 것으로 본다 · 1시간 뒤 옵션 델타가 +$16k 로 줄면(만기가 다가오면 외가격 옵션 델타가 0 쪽으로 준다) 숏이 $2k 과해져 되산다 = 헤지 매수. 부호가 달라도 모순이 아니다.\n규모: 커버 종목만 센 값이라 보통 수천~수만 $ — ETH 선물 거래대금(분당 수백만 $)에 비하면 가격을 움직일 크기가 아니다.\n우리 검정: 감마(관행 가정 기준)의 방향·크기 예측은 불합격 · 체결 기반은 검정 전 — 참고로만. charm 은 검정한 적 없다(만기 1시간 전 max pain 규칙의 메커니즘 후보).",
  exp: "Deribit 만기(매일·매주 금·월말·분기말 08:00 UTC = 17:00 KST). 시각은 모두 KST. 규모 = 콜+풋 미결제 × 지수 = 명목 달러(실제 옵션 값어치인 프리미엄은 그 0.2% 안팎). max pain = 만기에 보유자에게 줄 내재가치 합이 가장 작은 결제가(프리미엄 무시 · 누가 보유했는지 모른 채 미결제만으로 계산). 차트·사다리의 max pain 선은 늘 «가까운 만기» 하나이고, 그 만기 미결제는 보통 전체의 2% 안팎이다(라벨에 비중). P/C = 풋÷콜 미결제 수량 — 심리 지표가 아니다: 풋 매도는 강세·풋 매수는 헤지일 수 있고, 정의마다 값이 크게 다르다(가까운 만기 1.5 · 7일 1.0 · 전 만기 0.6 · 프리미엄 0.2).\n우리 검정: 만기 날 행사가로 끌려가는 핀닝·자석은 없었다(f 0.49~0.52, 표본 밖 1년 동전). 큰 만기 전후는 «이벤트 회피»(레버리지 낮추기) 용도.\n시간축 그림은 풋프린트 카드 맨 아래 왼쪽 «옵션 만기».",
  mood: "리버설·버터플라이는 «7일 고정만기»로 잰다: 만기마다 외가격 옵션의 델타로 보간해 정확히 델타 ±0.25 의 IV 를 구하고, 7일 양옆 두 만기를 시간으로 보간한다(30일 값은 괄호). 2026-10-01 까지는 가까운 만기(늘 24시간 미만) 최근접 행사가로 쟀는데 실제 델타가 0.15~0.31 이고 1시간에 평균 3.6pt 흔들려 잡음이었다(7일 고정만기는 0.58pt). 단위 pt = IV %포인트(가격 % 아님), IV 는 Deribit 평가값(mark_iv).\n25Δ 리스크 리버설 = 델타 +0.25 콜 IV − 델타 −0.25 풋 IV(지금가에서 위아래 비슷한 거리). 음수로 깊으면(−5pt 쯤 아래) 하락 방어 풋 수요 = 공포 · 양수면 상승 콜에 웃돈 · ±1pt 안은 중립.\n버터플라이 = (25Δ 콜 IV + 25Δ 풋 IV)/2 − ATM IV = 스마일이 휜 정도. 클수록 «방향은 몰라도 크게 튈» 꼬리에 값이 붙음. 작은 양수가 평상시.\n기간 구조 = 7·30·60일 고정만기 ATM IV(연율 %, 양옆 만기의 총분산 보간). 가까운 일간 만기는 남은 시간에 미국장이 드느냐에 따라 7일 대비 0.71~0.92배로 출렁여 뺐다(만기별 값은 «옵션 만기» 시간축 선). 뒤로 갈수록 높으면 정상(콘탱고) · 앞이 더 높으면(역전) «지금 당장» 큰 움직임을 값에 넣는 스트레스(급락·이벤트 직전). IV 41 ≈ 하루 1σ ±2.1%(41/√365).\n블록 거래 = 장외에서 합의해 거래소에 올린 큰 거래(원자료 다리 그대로, 전략 이름 추정 안 함).\n우리 검정: 아직 없음 — 스큐는 과거분을 살 수 없어 2026-09-28 부터 쌓는 중. 예측력 모름 → 매매 신호 말고 «분위기가 바뀌었나»(리버설 급락·기간 구조 역전) 확인용.\n기간 구조 기울기 = 고정만기 ATM IV 의 1일−7일 · 7일−30일. 1일이 7일보다 높으면 «역전»(주황). ✅검증: 역전이면 다음 24시간 실현 변동성이 DVOL 예상의 약 1.38배(45일, DVOL·최근 실현 통제 후에도 남음 · 겹치지 않는 하루 1표본 33일로는 경계선). 7일−30일도 같은 방향.\nATM 호가 폭 = 7일 근처 만기 ATM 옵션 매도−매수 호가를 IV 포인트로 — 넓으면 마켓메이커가 위험을 피한다. ⏳검정 불가(과거 호가 없음) — 10-01부터 쌓는 중, 나중에 검증.\n옵션 선도 − 지수 = 같은 스냅샷의 Deribit 가까운 만기 선도가와 Deribit 지수 차(bp). 대개 ±3bp 안이고 10분 변화 SD 1.9bp — 그 정도는 잡음. ⏳검정 불가(과거 Deribit 지수 없음, 09-28부터 쌓는 중) — «콜 수요면 +» 같은 해석은 아직 근거가 없다, 나중에 검증.\n(2026-10-02 의미 검증 뒤 남긴 옵션 정보 — 크기·국면만 말하고 방향은 말하지 않는다. 뺀 줄(검증 결과): 옵션 내재 폭(DVOL 과 동률) · 위험중립 왜도·첨도(왜도는 반대 방향, 늘 음수·늘 3 초과) · 변동성 순매수(이후 24h 예측력 없음) · 블록 요청자(순베가 = 블록 몫과 중복 · 순델타 방향 정보 없음) · O/S(높을수록 오히려 조용) · 정산 창(07~08 UTC 는 평소의 0.8배로 조용 · 미결제 크기와 무관) · 이벤트 예상 폭(공식이 0 으로 잘림).)",
  ladder: "행사가 사다리: 세로 = 행사가(지수 ±6%, 위 = 비쌈 · 풋프린트와 같은 방향). 왼쪽 빨강 = 풋 미결제, 오른쪽 초록 = 콜 미결제(달러), 맨 오른쪽 = 행사가별 순감마(청록 = 콜 쪽 +, 주황 = 풋 쪽 −). 흰 점선 = 지금 가격, 주황 점선 = 가까운 만기 max pain. 칩으로 범위(내일 만기 · 7일 안 만기 합 · 전 만기)를 바꾼다.\n우리 검정: «미결제가 큰 행사가로 가격이 끌린다(자석)»는 1년 표본 밖에서 동전 — 벽·지지저항으로 읽지 말고 «계약이 어디에 쌓였나»로만.",
  curve: "감마 곡선(딜러·체결, 커버 종목만): 가격이 지금에서 ±15% 옮겨 가면 딜러 감마 합이 얼마가 되는가(사다리 칩과 같은 범위). 청록 = 양감마 · 주황 = 음감마. 흰 점선 = 지금 가격, 주황 점선 = 플립. 세로 = 가격 1% 움직임당 딜러가 사고파는 금액($).\n점선 = DEX(딜러·체결 순델타 × 가격) — 단위가 달라 0선만 맞추고 크기는 따로 늘렸다(값은 아래 줄).\n커버가 낮으면 일부 종목의 곡선이다(카드 맨 위 커버 %). 검정 전 — 참고로만.",
  blocks: "Deribit 블록 거래(장외에서 맞춘 큰 옵션 거래, 지난 24시간): 한 줄 = 시각 · 명목 금액 · 다리별 매수/매도 만기 행사가 ×수량. 방향은 테이커 기준(RFQ 는 요청자 · 직접 거래는 수락자). 줄 머리 = 다리 구조로 분류한 전략 이름(Deribit 자체 구조 코드와 313/313 일치) · «+ 선물 헤지» = 같은 블록에 선물 다리가 붙음(보통 델타 중립 패키지) · 델타/베가 롱·숏 = 요청자 쪽 순델타·순베가 부호(체결 IV 로 계산, 총량의 10% 미만이면 중립). 단일 다리 블록은 다른 곳 헤지의 일부일 수 있어 의도를 모른다. 신규/청산·신원은 모른다. 금액 = 다리별 명목 합(스프레드면 두 다리가 다 더해진다).\n우리 검정 없음 — 참고로만.",
  lane: "옵션 만기(지금 ~ +120시간, 블록 거래는 가운데 «블록 거래» 칸): 시각 KST. 막대 = 다가올 만기 규모(콜+풋 미결제 × 지수 = 명목 달러) · pain = max pain(그 만기 미결제만으로 계산) · P/C = 풋÷콜 미결제 수량(심리 지표 아님) · 청록 선 = 만기별 ATM IV(가까운 만기는 남은 시간이 짧아 시간대에 따라 출렁인다).\n우리 검정: 만기 날 행사가로 끌려가는 핀닝·자석은 없었다(f 0.49~0.52). 큰 만기 전후는 «이벤트 회피»(레버리지 낮추기) 용도.",
  flow: "옵션 순매수 흐름(지난 24시간, 정시 버킷): 초록 = 콜 매수−매도, 빨강 = 풋 매수−매도(위 = 순매수 · 아래 = 순매도, 기초자산 수량). 청록 선 = 테이커 체결 순델타 누적(콜 매수·풋 매도 +, 콜 매도·풋 매수 −) — «옵션 시장을 통해 테이커가 롱으로 얼마나 기울었나». «감마 곡선»의 DEX 와는 다른 값이다: DEX 는 딜러(= 테이커 반대편) 순델타라 부호가 대략 반대이고(흐름 + = 테이커 롱 · DEX + = 딜러 롱), 창도 다르다(흐름 = 지난 24시간 · DEX = 종목 상장 이후 전체).\n방향은 Deribit 공개 체결의 테이커 방향(블록 포함). «신규 약 N%» = 그 시간 새로 열린 계약의 비율(체결량 V · 미결제 변화 ΔOI 로 (V+ΔOI)/2V, 매수·매도 양쪽 몫 기준) — 이 비율은 정확하지만 테이커와 메이커 중 누가 열었는지는 모른다(테이커 몫은 약 70% 만 확정). 예: 순매수가 0 근처여도 미결제가 크게 늘면 롱과 숏이 양쪽으로 새로 쌓인 것. «강제청산» = 테이커가 강제청산된 체결(Deribit 표시, 매시 history 로 대조). 만기로 사라지는 계약은 미결제 변화에서 뺀다. 검정 전 — 참고.",
};
// 2026-09-28 네 코인(ETH·BTC 역옵션 · SOL·XRP USDC 선형옵션). DVOL 지수는 ETH·BTC 뿐 -- 나머지는 30일 ATM IV(iv30)로 폭을 잰다.
function optData() {
  const cur = { eth: "ETH", btc: "BTC", sol: "SOL", xrp: "XRP" }[activeSnapshotAsset] || null;
  const g = cur && latestGex && latestGex.available ? (latestGex.currencies || {})[cur] : null;
  const blocks = cur ? ((((latestGex || {}).block_trades || {}).by_coin || {})[cur] || []) : [];
  return { cur, g, o: g && g.options, blocks };
}
const optIv = (o) => o.dvol ?? o.iv30;                      // 연 변동성(%) -- 예상 폭의 기준
// 가격 자릿수: ETH·BTC 는 정수, SOL 은 소수 1~2, XRP 는 소수 3~4 (±0.004$ 가 «±0$» 로 뭉개지지 않게)
const optQ = (v) => (v == null ? "-" : v >= 20 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v >= 0.1 ? v.toFixed(3) : v.toPrecision(2));
function optKst(ms, withDate = true) {
  const d = new Date(ms + 9 * 3600e3), p = (n) => String(n).padStart(2, "0");
  return (withDate ? `${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())} ` : "") + `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}`;
}
// 2026-09-30 검증: M 을 정수로 자르면 $1.38M 이 «$1M»이 되고 사다리 상위 셋이 «$2M $2M $2M»으로 안 갈렸다 -- 유효숫자 3자리 안팎
const optUsd = (v) => (v >= 1e9 ? `$${(v / 1e9).toFixed(2)}B` : v >= 1e6 ? `$${(v / 1e6).toFixed(v >= 1e8 ? 0 : v >= 1e7 ? 1 : 2)}M` : fmtUsdCompact(v));
const optPx = (o) => Number(latestLivePriceByAsset[activeSnapshotAsset] || 0) || o.index;
const optSigma = (o, sec) => (optIv(o) ? optPx(o) * (optIv(o) / 100) * Math.sqrt(sec / 31536000) : null);
const optFront = (o) => (o.expiries || []).find((e) => e.exp_ms > Date.now()) || null;

// 만기 시간축 카드(C, 2026-09-28 사용자 지시 «아래 카드로»): −24h~+120h. 점 = 지난 24h 블록 거래(이름표) · 막대 = 다가올 만기 규모
//   (옆에 max pain · P/C, 아래 날짜) · 선 = 만기별 ATM IV. 폭 W 는 카드에서 받는다 -- 좁으면(휴대폰) 이름표를 줄여 겹침을 피한다
//   (블록 이름표는 «지금» 왼쪽 폭이 모자라 점만, 만기 옆 pain·P/C 는 옵션 카드 표에 있다).
// 2026-10-01 한 화면 모드에서 만기·흐름·감마곡선 차트의 높이 배율(fitLayout 이 남는 칸으로 정한다) -- 글자 크기는 그대로, 그림 높이만.
function optChartK() { return document.documentElement.classList.contains("fit1") ? fitOptChartK : 1; }
// 감마 곡선은 제 배율 -- 블록 거래 칸 높이까지 채운다(사용자 «심리·감마곡선 아래 빈칸 → 블록거래와 높이 맞춰 · 감마곡선이 작다»)
function optCurveK() { return document.documentElement.classList.contains("fit1") ? fitOptCurveK : 1; }
function optLaneSvg(o, W) {
  // 2026-09-29 가운데 블록 칸이 생겨 1920 에서도 545px -- 만기 넷(24h 간격)이면 pain·P/C 세 줄이 들어간다
  const narrow = W < 480,
    H = narrow ? 170 : Math.round(190 * optChartK()), x0 = narrow ? 10 : 20, x1 = W - (narrow ? 10 : 20), t0 = 0, t1 = 120, now = Date.now();   // 2026-09-29 블록 점이 빠져 지난 24h 를 걷었다(사용자 지시)
  const base = narrow ? 104 : Math.round(118 * optChartK()), fs = narrow ? 10 : 11;
  const X = (h) => x0 + ((h - t0) / (t1 - t0)) * (x1 - x0);
  const ex = (o.expiries || []).filter((e) => e.exp_ms > now && (e.exp_ms - now) / 3.6e6 <= t1);
  const mx = Math.max(1, ...ex.map((e) => e.call_oi_usd + e.put_oi_usd));
  let s = "";
  for (let h = t0; h <= t1; h += 24) s += `<line x1="${X(h)}" x2="${X(h)}" y1="8" y2="${H - 18}" stroke="var(--line)" stroke-opacity=".5"/>`;
  s += `<line x1="${x0}" x2="${x1}" y1="${base}" y2="${base}" stroke="var(--line)"/>`
    + `<line x1="${X(0)}" x2="${X(0)}" y1="6" y2="${H - 16}" stroke="var(--ink)" stroke-opacity=".55" stroke-dasharray="3 3"/>`;
  ex.forEach((e) => {
    const v = e.call_oi_usd + e.put_oi_usd, hh = 10 + (narrow ? 50 : 62 * optChartK()) * Math.sqrt(v / mx), cx = X((e.exp_ms - now) / 3.6e6);
    const ly = base - (narrow ? hh : Math.max(hh, 40));   // 규모·pain·P/C 세 줄이 기준선 위에 들어오게 짧은 막대는 글자를 띄운다(반쪽 폭에서 한 줄이면 옆 막대와 겹쳤다)
    s += `<rect x="${(cx - 7).toFixed(1)}" y="${(base - hh).toFixed(1)}" width="14" height="${hh.toFixed(1)}" rx="3" fill="var(--warn)" fill-opacity=".85">`
      + `<title>${optKst(e.exp_ms)} 만기 · ${optUsd(v)} · max pain ${optQ(e.pain)} · P/C ${e.pc == null ? "-" : e.pc.toFixed(2)} · ATM IV ${e.atm_iv.toFixed(1)}%</title></rect>`
      + `<text x="${(cx + 12).toFixed(1)}" y="${(ly + 11).toFixed(1)}" font-size="${fs + 1}" font-weight="800" fill="var(--warn)">${optUsd(v)}</text>`
      + (narrow ? "" : `<text x="${(cx + 12).toFixed(1)}" y="${(ly + 25).toFixed(1)}" font-size="10.5" fill="var(--muted)">pain ${optQ(e.pain)}</text>`
        + `<text x="${(cx + 12).toFixed(1)}" y="${(ly + 39).toFixed(1)}" font-size="10.5" fill="var(--muted)">P/C ${e.pc == null ? "-" : e.pc.toFixed(2)}</text>`)
      + `<text x="${cx.toFixed(1)}" y="${base + 14}" font-size="${narrow ? 9.5 : 10.5}" font-weight="600" fill="var(--muted)" text-anchor="middle">${narrow ? optKst(e.exp_ms).slice(0, 5) : optKst(e.exp_ms)}</text>`;
  });
  if (ex.length >= 2) {
    const ivs = ex.map((e) => e.atm_iv), lo = Math.min(...ivs) - 1, hi = Math.max(...ivs) + 1;
    const yTop = base + 26, yBot = H - 24, yIv = (v) => yBot - ((v - lo) / (hi - lo)) * (yBot - yTop);
    s += `<text x="${x0 + 2}" y="${yBot + 2}" font-size="10.5" font-weight="700" fill="var(--option)">ATM IV</text>`
      + `<path d="M${ex.map((e) => `${X((e.exp_ms - now) / 3.6e6).toFixed(1)} ${yIv(e.atm_iv).toFixed(1)}`).join(" L")}" fill="none" stroke="var(--option)" stroke-width="2"/>`;
    ex.forEach((e) => {
      const cx = X((e.exp_ms - now) / 3.6e6);
      s += `<circle cx="${cx.toFixed(1)}" cy="${yIv(e.atm_iv).toFixed(1)}" r="3.5" fill="var(--option)"/>`
        + `<text x="${(cx + 7).toFixed(1)}" y="${(yIv(e.atm_iv) - 5).toFixed(1)}" font-size="10.5" font-weight="700" fill="var(--option)">${e.atm_iv.toFixed(0)}</text>`;
    });
  }
  for (let h = t0; h <= t1; h += 24) {
    s += `<text x="${X(h)}" y="${H - 2}" font-size="10.5" fill="${h === 0 ? "var(--ink)" : "var(--muted)"}" font-weight="${h === 0 ? 700 : 500}" text-anchor="${h === t0 ? "start" : h === t1 ? "end" : "middle"}">${h === 0 ? "지금" : `${h > 0 ? "+" : ""}${h}h`}</text>`;
  }
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="옵션 만기 시간축: 만기 규모, ATM IV">${s}</svg>`;
}

// 행사가 사다리(1-B, 2026-09-28): 옵션 카드 폭에 맞춘 세로 막대. 범위 칩은 브라우저에 기억한다.
// 딜러 감마 칸·가격축 플립은 사다리에서 고른 칩과 같은 범위(2026-09-29 사용자 지시). 옛 수집기 상태면 가까운 만기(gamma).
const optGamma = (o) => (o.gamma_by || {})[optLadderScope] || o.gamma || {};
// 2026-09-29 사용자 «지금 부족한 데이터로 옵션을 모두 체결 기반으로, 상단에 커버 부족 경고·%»(옵션 세션과 합의 A~F):
//   딜러 감마·플립·DEX·감마 곡선·사다리 감마 막대·가격축 플립 = **딜러·체결**(테이커 반대편, 수집 뒤 상장 종목만).
//   가정 값(now_usd · flip · dex_asm_usd)은 수집만 하고 화면에서 뺐다. 커버 부족은 카드 맨 위 경고(optCovBanner)가 알린다.
//   수집기 재시작 전(키 없음)에도 안 깨지게 전부 null 로 떨어진다. 곡선 행 = [가격, 감마(딜러·체결), DEX(딜러·체결)].
const optDealer = (o) => {
  const g = optGamma(o);
  // 2026-09-30 검증: 수집은 10분마다라 08:00 UTC 만기 직후 최대 10분간 «가까운 만기»가 이미 끝난 만기를 가리킨다 -- 값 대신 «교체 중»
  if (optLadderScope === "front" && g.exp_ms && g.exp_ms <= Date.now()) {
    return { rolled: true, exp_ms: g.exp_ms, cov: null, nr: null, now_usd: null, flip: null, dex_usd: null, charm: null, n: null, exps: null, profile: [] };
  }
  return { exp_ms: g.exp_ms, n: g.dealer_n ?? null, exps: g.exps ?? null, cov: g.dealer_cov ?? null, nr: g.dealer_net_oi ?? null, now_usd: g.dealer_gex_usd ?? null,
           flip: g.dealer_flip ?? null, dex_usd: g.dealer_dex_usd ?? null, charm: g.dealer_charm_1h_usd ?? null,   // 2026-09-29 charm(1시간)
           profile: (g.profile || []).map((r) => [r[0], r[5] ?? null, r[4] ?? null]) };
};
const optCovPct = (c) => (c == null ? "-" : c < 0.1 && c > 0 ? `${(c * 100).toFixed(1)}%` : `${Math.round(c * 100)}%`);
// 2026-10-01 고정만기 값 한 줄: «+1.2pt (30일 −0.4)». 수집기가 새 키(cm)를 아직 안 실었으면 «-».
function optCm(o, k) {
  const cm = o.cm || {}, v7 = (cm["7"] || {})[k], v30 = (cm["30"] || {})[k];
  const f = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}`;
  return v7 == null ? "-" : `${f(v7)}pt${v30 == null ? "" : ` (30일 ${f(v30)})`}`;
}
// 2026-10-01 연구: 한 코인 요약이 실패해도 상태 파일 시각은 새로 찍혀 옛 값이 새 값처럼 보일 수 있다 -- 요약 자신의 시각을 보인다.
function optAsOf(o) {
  const t = Date.parse(o.recorded_at_utc || ""), age = (Date.now() - t) / 60e3;
  if (!Number.isFinite(t)) return "";
  const hm = new Date(t).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit", hour12: false });
  return age > 25 ? ` · <span class="opt-warn">${hm} 기준(${Math.round(age)}분 전 값)</span>` : ` · ${hm} 기준`;
}
function optCovBanner(o) {
  const gb = o.gamma_by || {}, names = { front: "가까운 만기", week: "7일 안", all: "전 만기" };
  const parts = ["front", "week", "all"].map((k) => {
    // 2026-10-01 «가정 깨짐 N배»(>1 일 때만 경보) → 늘 «설명 X%»(Σ|테이커 순|/미결제). 100% 넘으면 경고색.
    const x = gb[k] || {}, broke = x.dealer_net_oi == null ? "" : ` <span class="${x.dealer_net_oi > 1 ? "opt-warn" : ""}">설명 ${Math.round(x.dealer_net_oi * 100)}%</span>`;
    const t = `${names[k]} ${optCovPct(x.dealer_cov)}${broke}`;
    return k === optLadderScope ? `<b>${t}</b>` : t;
  });
  const full = ["front", "week", "all"].every((k) => (gb[k] || {}).dealer_cov >= 0.99 && !((gb[k] || {}).dealer_net_oi > 1));
  // 2026-10-01 제목 옆 한 줄(사용자 «Deribit 참고, 신호 아님 자리를 이 줄로 · 높이 확보») -- 따로 쓰던 줄(.opt-cov)을 없앴다
  return `<span class="opt-cov-h${full ? "" : " warn"}" role="status">${full ? "체결 기반 · 커버" : "⚠ 체결 기반 · 커버 부족 — 딜러 값은 커버 종목만"} · ${parts.join(" · ")}${optAsOf(o)}</span>`;
}
let optLadderScope = (() => { try { return localStorage.getItem("optLadder") || "week"; } catch (e) { return "week"; } })();
// 범위 칩 -- 사다리가 비거나 «만기 교체 중»일 때도 보여야 다른 범위로 옮길 수 있다(2026-09-30 검증)
function optLadderChips() {
  const chip = (k, t) => `<button type="button" class="opt-chip-btn${optLadderScope === k ? " on" : ""}" data-scope="${k}" aria-pressed="${optLadderScope === k}">${t}</button>`;
  return `<div class="opt-chips">${chip("front", "가까운 만기")}${chip("week", "7일 안")}${chip("all", "전 만기")}</div>`;
}
function optLadderSvg(o, W, Hfit = null) {
  const st = o.strikes || {}, rows = st[optLadderScope] || [];
  const rolled = optLadderScope === "front" && st.front_exp_ms && st.front_exp_ms <= Date.now();   // 만기 직후 ≤10분(optDealer 와 같은 규칙)
  if (!rows.length || rolled) return optLadderChips() + `<div class="opt-note">${rolled ? "만기 교체 중 — 다음 수집(10분 안)부터 새 가까운 만기" : "행사가 데이터 없음(수집기 다음 주기에 채워진다)"}</div>`;
  const px = optPx(o), lo = px * 0.94, hi = px * 1.06, H = Hfit || 560,   // 2026-10-01 ±8% -> ±6%(사용자 «막대를 키워서» -- 행사가 간격 10/20 이 섞여 두께는 좁은 간격에 묶이므로 범위를 좁혀 1.33배)
  y0 = 34, y1 = H - 8;   // y0 22 -> 34: 맨 위 막대가 «← 풋 · 콜 →» 머리글과 겹쳤다   // 2026-09-30 380 -> 560(막대를 두껍게 -- 값 글자가 막대 안에 들어가게, 사용자 지시)
  const Y = (p) => y1 - ((p - lo) / (hi - lo)) * (y1 - y0);
  const vis = rows.filter((r) => r[0] >= lo && r[0] <= hi);   // 2026-10-01 범위 밖 행사가는 안 그린다(맨 위 막대가 머리글을 덮었다) · 막대 길이 기준도 범위 안에서
  const mid = Math.round(W * 0.47), half = mid - 46, gW = 34, gx = W - gW;
  const mx = Math.max(1, ...vis.map((r) => Math.max(r[1], r[2]))), gm = Math.max(1, ...vis.map((r) => (Number.isFinite(r[4]) ? Math.abs(r[4]) : 0)));
  const ks = vis.map((r) => r[0]), step = ks.length > 1 ? Math.min(...ks.slice(1).map((k, i) => k - ks[i])) : 25;
  const bh = Math.max(2, Math.min(22, ((y1 - y0) * step) / (hi - lo) - 1)),   // 2026-10-01 상한 15 -> 22(사용자 «막대를 키워서»)
    fsz = Math.max(8, Math.min(10.5, bh - 1));   // 값 글자 = 막대 두께 − 1
  const top = [...vis].sort((a, b) => b[1] + b[2] - a[1] - a[2]).slice(0, 3).map((r) => r[0]);
  let s = `<text x="${mid - 6}" y="12" font-size="10" font-weight="700" fill="var(--bad)" text-anchor="end">← 풋</text>`
    + `<text x="${mid + 6}" y="12" font-size="10" font-weight="700" fill="var(--good)">콜 →</text>`
    + `<text x="${W}" y="12" font-size="10" font-weight="700" fill="var(--option)" text-anchor="end">감마·체결</text>`
    + `<line x1="${mid}" x2="${mid}" y1="${y0 - 4}" y2="${y1}" stroke="var(--line)"/>`;
  vis.forEach(([k, c, p, , g]) => {   // g = 딜러·체결 순감마(커버 종목 없는 행사가는 null = «모름», 막대 없음)
    const has = Number.isFinite(g), y = Y(k), pw = (half * p) / mx, cwid = (half * c) / mx, gwid = has ? 3 + ((gW - 6) * Math.abs(g)) / gm : 0;
    s += `<g><title>${optQ(k)} · 콜 ${optUsd(c)} · 풋 ${optUsd(p)} · 딜러·체결 순감마 ${has ? `${g >= 0 ? "+" : "−"}${optUsd(Math.abs(g))}/1%` : "모름(커버 종목 없음)"}</title>`
      + `<rect x="${(mid - pw).toFixed(1)}" y="${(y - bh / 2).toFixed(1)}" width="${pw.toFixed(1)}" height="${bh.toFixed(1)}" fill="var(--bad)" fill-opacity=".75"/>`
      + `<rect x="${mid}" y="${(y - bh / 2).toFixed(1)}" width="${cwid.toFixed(1)}" height="${bh.toFixed(1)}" fill="var(--good)" fill-opacity=".75"/>`
      + (has ? `<rect x="${(W - gwid).toFixed(1)}" y="${(y - bh / 2).toFixed(1)}" width="${gwid.toFixed(1)}" height="${bh.toFixed(1)}" fill="${g >= 0 ? "var(--option)" : "var(--warn)"}" fill-opacity=".85"/>` : "") + `</g>`;
    if (top.includes(k)) {   // 2026-09-30 큰 막대 값은 막대 **안**(사용자 지시) -- 막대 끝에서 안쪽으로, 글자가 안 들어가면 예전처럼 옆에
      const lab = optUsd(c + p), bw = c >= p ? cwid : pw, fit = bw >= lab.length * fsz * 0.62 + 8;
      const x = mid + (c >= p ? (fit ? cwid - 4 : cwid + 4) : (fit ? -pw + 4 : -pw - 4));
      s += `<text x="${x.toFixed(1)}" y="${(y + fsz * 0.36).toFixed(1)}" font-size="${fsz.toFixed(1)}" font-weight="800" fill="${fit ? "var(--chart-bg)" : "var(--text)"}"`
        + ` text-anchor="${(c >= p) === fit ? "end" : "start"}">${lab}</text>`;
    }
  });
  const lab = (v, y) => `<text x="0" y="${(y + 3.5).toFixed(1)}" font-size="9.5" fill="var(--muted)">${optQ(v)}</text>`;
  const every = step * Math.max(1, Math.round((hi - lo) / step / 8));
  ks.filter((k) => Math.abs(k / every - Math.round(k / every)) < 1e-6).forEach((k) => { s += lab(k, Y(k)); });
  s += `<line x1="30" x2="${gx - 4}" y1="${Y(px).toFixed(1)}" y2="${Y(px).toFixed(1)}" stroke="var(--ink)" stroke-dasharray="4 3" stroke-opacity=".8"/>`
    + `<text x="${gx - 6}" y="${(Y(px) - 4).toFixed(1)}" font-size="10" font-weight="700" fill="var(--ink)" text-anchor="end">지금 ${optQ(px)}</text>`;
  const f = optFront(o);
  if (f && f.pain >= lo && f.pain <= hi) {
    s += `<line x1="30" x2="${gx - 4}" y1="${Y(f.pain).toFixed(1)}" y2="${Y(f.pain).toFixed(1)}" stroke="var(--warn)" stroke-dasharray="2 4"/>`
      + `<text x="32" y="${(Y(f.pain) + (f.pain < px ? 12 : -4)).toFixed(1)}" font-size="9.5" font-weight="700" fill="var(--warn)">max pain ${optQ(f.pain)}${f.oi_share == null ? "" : ` · 미결제 ${Math.round(f.oi_share * 100)}%`}</text>`;
  }
  return optLadderChips()
    + `<svg class="opt-ladder" viewBox="0 0 ${W} ${H}" role="img" aria-label="행사가별 콜·풋 미결제와 순감마">${s}</svg>`;
}

// 2026-09-29 감마 곡선(사용자 지시 «행사가 사다리 바로 아래, 칩에 맞게»): 수집기 profile(지수 ±15% 25점) = [가격, 감마$, DEX$].
//   가로 = 가격 · 세로 = 1% 움직임당 딜러 감마($). 색은 사다리 감마 막대와 같다(+ 청록 · − 주황).
function optGammaCurveSvg(o, W) {
  const g = optDealer(o), pr = g.profile.filter((r) => r[0] > 0 && Number.isFinite(r[1]));
  if (pr.length < 2) return `<div class="opt-note">감마 곡선 없음(체결 기반 — 커버 종목 없음 또는 수집기 재시작 전)</div>`;
  const H = Math.round(150 * optCurveK()), x0 = 42, x1 = W - 4, y0 = 16, y1 = H - 20, px = optPx(o);
  const lo = pr[0][0], hi = pr[pr.length - 1][0], X = (p) => x0 + ((p - lo) / (hi - lo)) * (x1 - x0);
  // DEX(profile 세 번째 칸, 09-29 수집기 추가)는 단위·크기가 달라(수십 배) 0선만 감마와 맞추고 크기는 제 폭으로 늘린다.
  const dc = 2, dNow = g.dex_usd;   // 점선 = DEX(딜러·체결)
  const hasD = pr.every((r) => Number.isFinite(r[dc])), dAbs = hasD ? Math.max(...pr.map((r) => Math.abs(r[dc]))) : 0;
  const dk = dAbs > 0 ? Math.max(...pr.map((r) => Math.abs(r[1])), 1) / dAbs : 0;
  const gmax = Math.max(...pr.map((r) => Math.max(r[1], (r[dc] || 0) * dk)), 0), gmin = Math.min(...pr.map((r) => Math.min(r[1], (r[dc] || 0) * dk)), 0), span = gmax - gmin || 1;
  const Y = (v) => y0 + ((gmax - v) / span) * (y1 - y0), z = Y(0);
  const gTop = Math.max(...pr.map((r) => r[1])), gBot = Math.min(...pr.map((r) => r[1]));
  const pts = pr.map((r) => `${X(r[0]).toFixed(1)} ${Y(r[1]).toFixed(1)}`), area = `M${X(lo).toFixed(1)} ${z.toFixed(1)} L${pts.join(" L")} L${X(hi).toFixed(1)} ${z.toFixed(1)}Z`;
  const m = (v) => `${v >= 0 ? "+" : "−"}${optUsd(Math.abs(v)).replace("$", "")}`;
  const id = `optgc${W}`;
  let s = `<defs><clipPath id="${id}p"><rect x="${x0}" y="0" width="${x1 - x0}" height="${z.toFixed(1)}"/></clipPath>`
    + `<clipPath id="${id}n"><rect x="${x0}" y="${z.toFixed(1)}" width="${x1 - x0}" height="${(H - z).toFixed(1)}"/></clipPath></defs>`
    + `<path d="${area}" fill="var(--option)" fill-opacity=".28" clip-path="url(#${id}p)"/>`
    + `<path d="${area}" fill="var(--warn)" fill-opacity=".28" clip-path="url(#${id}n)"/>`
    + `<line x1="${x0}" x2="${x1}" y1="${z.toFixed(1)}" y2="${z.toFixed(1)}" stroke="var(--line)"/>`
    + `<path d="M${pts.join(" L")}" fill="none" stroke="var(--text)" stroke-opacity=".8" stroke-width="1.5"/>`
    + (dk ? `<path d="M${pr.map((r) => `${X(r[0]).toFixed(1)} ${Y(r[dc] * dk).toFixed(1)}`).join(" L")}" fill="none" stroke="var(--ink)" stroke-opacity=".7" stroke-width="1.5" stroke-dasharray="5 3"/>` : "")
    // 눈금 글자는 감마 자신의 끝값만(늘린 DEX 값이 아니다) -- 폭의 5% 미만이면 0 과 겹쳐 생략
    + (gTop > span * 0.05 ? `<text x="${x0 - 4}" y="${(Y(gTop) + 4).toFixed(1)}" font-size="9.5" fill="var(--option)" text-anchor="end">${m(gTop)}</text>` : "")
    + (gBot < -span * 0.05 ? `<text x="${x0 - 4}" y="${Y(gBot).toFixed(1)}" font-size="9.5" fill="var(--warn)" text-anchor="end">${m(gBot)}</text>` : "")
    + `<text x="${x0 - 4}" y="${(z + 3.5).toFixed(1)}" font-size="9.5" fill="var(--muted)" text-anchor="end">0</text>`;
  [0.9, 1, 1.1].forEach((f) => { const p = px * f; if (p > lo && p < hi) s += `<text x="${X(p).toFixed(1)}" y="${H - 4}" font-size="9.5" fill="var(--muted)" text-anchor="middle">${optQ(p)}</text>`; });
  if (px > lo && px < hi) s += `<line x1="${X(px).toFixed(1)}" x2="${X(px).toFixed(1)}" y1="${y0 - 6}" y2="${y1}" stroke="var(--ink)" stroke-dasharray="4 3" stroke-opacity=".8"/>`
    + `<text x="${X(px).toFixed(1)}" y="${y0 - 8}" font-size="10" font-weight="700" fill="var(--ink)" text-anchor="middle">지금</text>`;
  if (g.flip && g.flip > lo && g.flip < hi) {
    const fx = X(g.flip), right = g.flip >= px;
    s += `<line x1="${fx.toFixed(1)}" x2="${fx.toFixed(1)}" y1="${y0}" y2="${y1}" stroke="var(--warn)" stroke-dasharray="3 3"/>`
      + `<text x="${(fx + (right ? 4 : -4)).toFixed(1)}" y="${(y0 + 14).toFixed(1)}" font-size="10" font-weight="700" fill="var(--warn)" text-anchor="${right ? "start" : "end"}">플립 ${optQ(g.flip)}</text>`;
  }
  pr.forEach((r) => { s += `<rect x="${(X(r[0]) - (x1 - x0) / pr.length / 2).toFixed(1)}" y="${y0}" width="${((x1 - x0) / pr.length).toFixed(1)}" height="${y1 - y0}" fill="transparent"><title>${optQ(r[0])} 에서 감마 ${m(r[1])}/1%${Number.isFinite(r[dc]) ? ` · DEX(딜러) ${m(r[dc])}` : ""}</title></rect>`; });
  // DEX 이름표는 그림 밖 한 줄(끝에서 두 선이 자주 교차해 글자가 곡선에 묻혔다)
  const leg = `<div class="opt-note">실선 = 감마 · 점선 = DEX (둘 다 딜러·체결)${dNow != null ? ` · DEX <b class="opt-dex">${m(dNow)}$</b> (지금 가격)` : ""}</div>`;
  return `<svg class="opt-curve" viewBox="0 0 ${W} ${H}" role="img" aria-label="가격별 딜러 감마 곡선과 DEX">${s}</svg>${dk ? leg : ""}`;
}

// 옵션 순매수 흐름(2-A, 2026-09-28): 만기 시간축 카드 아래 줄. 정시 버킷 25개(마지막은 진행 중).
// 2026-10-01 «새로 연 쪽이냐 닫은 쪽이냐»(연구 research_eth_option_open_close_20260930): 체결량 V 와 미결제 변화 ΔOI 로
//   신규 비율 = (V + ΔOI) / 2V 는 **정확**하다(매수·매도 양쪽 몫 기준). 누가 열었는지(테이커냐 메이커냐)는 모른다.
//   ΔOI 는 수집기가 정시 첫 스냅샷끼리, 두 스냅샷에 다 있는 종목만(만기 소멸 제외)으로 센다 -- 없으면(옛 수집기·첫 시간) 빈칸.
function optNewTxt(bs) {
  const ok = bs.filter((b) => b.doi != null), V = ok.reduce((a, b) => a + b.cb + b.cs + b.pb + b.ps, 0), dOi = ok.reduce((a, b) => a + b.doi, 0);
  if (!(V > 0)) return "";
  const r = Math.max(0, Math.min(1, (V + dOi) / (2 * V)));
  return ` · 미결제 ${dOi >= 0 ? "+" : "−"}${fmtNum(Math.abs(dOi), 0)} · 신규 약 ${Math.round(r * 100)}%`;
}
function optFlowSvg(flow, W) {
  if (!flow || !flow.length) return "";
  const narrow = W < 700, k = narrow ? 1 : optChartK(), H = narrow ? 150 : Math.round(170 * k), x0 = narrow ? 34 : 60, x1 = W - (narrow ? 8 : 20), base = narrow ? 84 : Math.round(94 * k), amp = narrow ? 44 : Math.round(52 * k);
  const n = flow.length, X = (i) => x0 + (i / n) * (x1 - x0), bw = ((x1 - x0) / n) * 0.36;
  const nets = flow.map((b) => [b.cb - b.cs, b.pb - b.ps]);
  const mx = Math.max(1e-9, ...nets.flat().map(Math.abs));
  let acc = 0; const cum = flow.map((b) => (acc += b.dlt));
  const dm = Math.max(1e-9, ...cum.map(Math.abs));
  let s = `<line x1="${x0}" x2="${x1}" y1="${base}" y2="${base}" stroke="var(--line)"/>`
    + `<text x="${x0 - 6}" y="${base - amp + 8}" font-size="9.5" fill="var(--muted)" text-anchor="end">순매수</text>`
    + `<text x="${x0 - 6}" y="${base + amp - 2}" font-size="9.5" fill="var(--muted)" text-anchor="end">순매도</text>`;
  nets.forEach(([c, p], i) => {
    [[c, "var(--good)", 0], [p, "var(--bad)", bw + 1]].forEach(([v, col, dx]) => {
      const hh = (amp * Math.abs(v)) / mx;
      if (hh < 0.5) return;
      s += `<rect x="${(X(i) + 3 + dx).toFixed(1)}" y="${(v >= 0 ? base - hh : base).toFixed(1)}" width="${bw.toFixed(1)}" height="${hh.toFixed(1)}" fill="${col}" fill-opacity="${i === n - 1 ? 0.45 : 0.8}"/>`;
    });
    const b = flow[i];
    s += `<rect x="${X(i).toFixed(1)}" y="${base - amp}" width="${((x1 - x0) / n).toFixed(1)}" height="${2 * amp}" fill="transparent"><title>${optKst(b.h)}~ · 콜 매수 ${fmtNum(b.cb, 0)} / 매도 ${fmtNum(b.cs, 0)} · 풋 매수 ${fmtNum(b.pb, 0)} / 매도 ${fmtNum(b.ps, 0)} · 순델타 ${b.dlt >= 0 ? "+" : ""}${fmtNum(b.dlt, 0)}${optNewTxt([b])}${b.liq > 0 ? ` · 강제청산 ${fmtNum(b.liq, 0)}` : ""}${i === n - 1 ? " (진행 중)" : ""}</title></rect>`;
    if (i % (narrow ? 6 : 4) === 0) s += `<text x="${(X(i) + 3).toFixed(1)}" y="${H - 2}" font-size="9.5" fill="var(--muted)">${optKst(b.h, false).slice(0, 2)}시</text>`;
  });
  s += `<path d="M${cum.map((v, i) => `${X(i + 1).toFixed(1)} ${(base - (amp * v) / dm).toFixed(1)}`).join(" L")}" fill="none" stroke="var(--option)" stroke-width="2"/>`
    + `<text x="${x1}" y="${(base - (amp * cum[n - 1]) / dm - 6).toFixed(1)}" font-size="${narrow ? 10 : 11}" font-weight="700" fill="var(--option)" text-anchor="end">체결 순델타 ${cum[n - 1] >= 0 ? "+" : ""}${fmtNum(cum[n - 1], 0)} ${escapeHtml(optData().cur || "")}</text>`;
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="옵션 순매수 흐름: 시간별 콜·풋 순매수와 순델타 누적">${s}</svg>`;
}

const optTipOpen = new Set();
// 2026-09-29 휴대폰 블록 거래(사용자 선택 C): 시간축 점이 한 칸에 겹쳐 안 보여 «요약 한 줄 + 펼치기 목록». 펼침은 브라우저 기억.
let optBlkOpen = (() => { try { return localStorage.getItem("optBlkOpen") === "1"; } catch (e) { return false; } })();
const OPT_MON = { JAN: "01", FEB: "02", MAR: "03", APR: "04", MAY: "05", JUN: "06", JUL: "07", AUG: "08", SEP: "09", OCT: "10", NOV: "11", DEC: "12" };
// 2026-10-01 블록 한 줄 머리: 구조(Deribit 구조 코드와 313/313 일치하는 규칙) · 선물 헤지 · 순델타 쪽 · 베가 쪽 -- 전부 요청자(테이커) 기준 부호.
const OPT_SHAPE = { single: "단일", straddle: "스트래들", strangle: "스트랭글", risk_reversal: "리스크 리버설", synthetic: "합성 선물",
  vertical: "수직 스프레드", ratio_spread: "비율 스프레드", calendar: "캘린더", diagonal: "대각", butterfly: "버터플라이", ladder: "래더",
  condor: "콘도르", iron_condor: "아이언 콘도르", iron_butterfly: "아이언 버터플라이", box: "박스", other: "복합", future_only: "선물" };
function optBlkShape(b) {
  if (!b.structure) return "";
  const d = { long: "델타 롱", short: "델타 숏", neutral: "델타 중립" }[b.delta_side], v = { long_vol: "베가 롱", short_vol: "베가 숏", neutral: "베가 중립" }[b.vol_side];
  return `<span class="opt-blk-shape">${OPT_SHAPE[b.structure] || b.structure}${b.hedge ? " + 선물 헤지" : ""}${d ? ` · ${d}` : ""}${v ? ` · ${v}` : ""}</span> `;
}
function optBlkLegs(b) {   // ETH-16OCT26-2700-P -> «매수 10-16 2700P ×500» (SOL·XRP 행사가 소수점 d)
  return (b.legs || []).map((l) => {
    const m = String(l.instrument_name).match(/-(\d+)([A-Z]{3})\d+-([\dd]+)-([CP])$/);
    const k = m ? `${OPT_MON[m[2]] || m[2]}-${m[1].padStart(2, "0")} ${m[3].replace("d", ".")}${m[4]}` : escapeHtml(String(l.instrument_name));
    return `<span class="opt-leg"><i class="${l.direction === "buy" ? "opt-good" : "opt-bad"}">${l.direction === "buy" ? "매수" : "매도"}</i> ${k} ×${fmtNum(l.amount, 0)}</span>`;
  }).join(" · ");
}
// 2026-09-29 옵션 카드 «블록 거래» 칸(사용자 «데스크톱도 옵션 카드 안에») -- 데스크톱은 늘 펼침, 휴대폰만 «목록 ▾» 버튼.
function optBlkHtml(blocks, n24, phone, sum24 = null, max24 = null) {
  if (!blocks.length) return `<div class="opt-blk-sum"><span>블록 24h <b>0건</b></span></div>`;
  const open = !phone || optBlkOpen;
  // 합계·최대는 서버가 24h 전체로 준 값(목록은 최신 8건뿐 -- 2026-09-30 검증). 옛 서버면 받은 목록으로.
  const tot = sum24 ?? blocks.reduce((a, b) => a + (b.notional_usd || 0), 0), mx = max24 || blocks.reduce((a, b) => ((b.notional_usd || 0) > (a.notional_usd || 0) ? b : a));
  const rows = [...blocks].sort((a, b) => b.ts_ms - a.ts_ms).slice(0, 5).map((b) =>
    `<div class="opt-blk-row"><span class="t">${optKst(b.ts_ms, false)}</span><b>${optUsd(b.notional_usd || 0)}</b><span class="legs">${optBlkShape(b)}${optBlkLegs(b)}</span></div>`).join("");
  return `<div class="opt-blk-sum"><span>블록 24h <b>${n24 ?? blocks.length}건 · ${optUsd(tot)}</b> · 최대 ${optUsd(mx.notional_usd || 0)} ${optKst(mx.ts_ms, false)}</span>`
    + (phone ? `<button type="button" class="opt-blk-btn" aria-expanded="${optBlkOpen}">${optBlkOpen ? "접기 ▴" : "목록 ▾"}</button>` : "") + `</div>` + (open ? `<div class="opt-blk-list">${rows}</div>` : "");
}
const optClick = (e) => {
  if (e.target.closest(".opt-blk-btn")) {
    optBlkOpen = !optBlkOpen;
    try { localStorage.setItem("optBlkOpen", optBlkOpen ? "1" : "0"); } catch (err) { /* 기억은 편의 */ }
    renderOptions();
    return;
  }
  const sc = e.target.closest(".opt-chip-btn");
  if (sc) {
    optLadderScope = sc.dataset.scope;
    try { localStorage.setItem("optLadder", optLadderScope); } catch (err) { /* 기억은 편의 */ }
    renderOptions();
    return;
  }
  const b = e.target.closest(".opt-q");
  if (!b) return;
  const k = b.dataset.tip, open = !optTipOpen.has(k);
  if (open) optTipOpen.add(k); else optTipOpen.delete(k);
  b.setAttribute("aria-expanded", String(open));
  const tip = (b.closest(".opt-sec") || b.closest(".opt-lane-body"))?.querySelector(".opt-tip");
  if (tip) tip.hidden = !open;
};
["optBody", "optLaneBody", "optBlkBody", "optFlowBody"].forEach((id) => el(id)?.addEventListener("click", optClick));

// 2026-09-29 가운데 «블록 거래» 칸 = 위 풋프린트의 체결+호가 프로파일과 같은 폭·같은 x(사용자 지시). 프로파일이 카드
//   오른쪽 끝에 붙는 폭(수급 칸이 아래로 내려간 1440 이하)에서는 같은 폭으로 가운데. 휴대폰은 CSS 가 한 줄로 쌓는다.
let fpProfileSpan = null;   // [체결 기둥 왼쪽, 호가 띠 오른쪽] -- renderCandleSvg 가 채운다(viewBox 단위). 오른쪽 수급 칸은 그 뒤에 붙는다
function optRowCols() {
  const row = el("optRow"), svg = el("candleSvgSnapshot");
  if (!row) return;
  let cols = "";
  if (fpProfileSpan && svg && !window.matchMedia("(max-width: 720px)").matches) {
    const [l, r] = fpProfileSpan, w = svg.viewBox.baseVal.width || 1, k = svg.getBoundingClientRect().width / w, gap = 18, mid = Math.round((r - l) * k);
    const dx = svg.getBoundingClientRect().left - row.getBoundingClientRect().left;
    cols = r < w - 40 ? `${Math.max(0, Math.round(dx + l * k - gap))}px ${mid}px minmax(0, 1fr)` : `minmax(0, 1fr) ${mid}px minmax(0, 1fr)`;
  }
  if (row.style.gridTemplateColumns !== cols) row.style.gridTemplateColumns = cols;
}

addEventListener("resize", () => requestAnimationFrame(optRowCols));   // 2026-09-30 검증: 창 폭이 바뀌면 가운데 칸을 바로 다시 맞춘다

// 2026-10-01 «옵션 정보»(문헌 조사 → 사용자 «모두 옵션 카드에») → 2026-10-02 의미 검증 뒤 사용자 «다섯만 남기고 제거»:
//   지지(기간 구조 기울기 · VoV) · 방향 맞음·표본 부족(꼬리 확률) · 검정 불가 → 쌓아서 나중에 검증(ATM 호가 폭 · 선도 − 지수).
//   뺀 것(옵션 내재 · 왜도첨도 · 변동성 순매수 · 블록 요청자 · O/S · 정산 창 · 이벤트 폭)은 근거 없음/반대/공식 결함이었다
//   (scripts/research_eth_option_info_{surface,flow}_validate_20261002.py). 빠진 값은 줄째 뺀다.
// 2026-10-02(2) 사용자 «옵션 정보 칸을 풀어서 다른 칸에 넣고 칸은 제거»: 크기 정보(VoV · 꼬리 확률) → 예상 폭 · 변동성 시장의 결(기간 구조 기울기 · ATM 호가 폭 · 선도−지수) → 심리.
//   검증 근거는 각 칸 «?» 설명에 그대로 옮겼다. 빠진 값은 줄째 뺀다.
function optInfoParts(o, g, kv) {
  const sf = o.surface || {}, fr = sf.front, s7 = sf["7"], cm = o.cm || {}, oh = (g && g.opt_hist) || {};
  const pct = (v) => (v == null ? "-" : `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`);
  const sgnN = (v, d = 1) => (v == null ? "-" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`);
  const move = [], mood = [];
  const a1 = (cm["1"] || {}).atm, a7 = (cm["7"] || {}).atm, a30 = (cm["30"] || {}).atm;
  if (oh.vov24 != null) move.push(kv("VoV · 24h", `내재 변동성 시간당 ±${oh.vov24.toFixed(1)}%`));
  if (fr && fr.tail) move.push(kv(`꼬리 확률 · 만기까지(${fr.hours.toFixed(0)}h)`, `±2% 밖 ${pct(fr.tail["2"])} · ±3% ${pct(fr.tail["3"])} · ±5% ${pct(fr.tail["5"])}`));
  if (s7 && s7.tail) move.push(kv(`꼬리 확률 · ${Math.round(s7.hours / 24)}일 만기`, `±5% 밖 ${pct(s7.tail["5"])}`));
  if (a1 != null && a7 != null) mood.push(kv("기간 구조 기울기", `1일−7일 ${sgnN(a1 - a7)}pt${a30 != null ? ` · 7일−30일 ${sgnN(a7 - a30)}pt` : ""}`, a1 > a7 ? "opt-warn" : ""));
  if (o.atm_spread_iv != null) mood.push(kv("ATM 호가 폭 · 7일 근처", `${o.atm_spread_iv.toFixed(1)} vol pt`));
  // 같은 스냅샷끼리만 잰다 -- 10분 묵은 선도가를 지금 바이낸스 가격과 견주면 그 사이 움직임이 섞였다(배포 직후 +30bp 로 보였다)
  if (fr && fr.fwd && o.index) mood.push(kv("옵션 선도 − 지수 · 가까운 만기", `${sgnN((fr.fwd / o.index - 1) * 1e4)}bp`));
  return { move: move.join(""), mood: mood.join("") };
}
function renderOptions() {
  const body = el("optBody"), chip = el("optChip");
  if (!body) return;
  const { cur, g, o, blocks } = optData();
  const row = el("optRow"), laneBody = el("optLaneBody"), blkBody = el("optBlkBody"), flowBody = el("optFlowBody");
  if (row) row.hidden = !o;
  optRowCols();
  if (!o) {
    if (chip) chip.hidden = true;
    body.innerHTML = `<div class="opt-note">${!cur ? "Deribit 옵션은 ETH·BTC·SOL·XRP 만 수집합니다"
      : latestGex && latestGex.error ? `옵션 수집 지연 (${escapeHtml(latestGex.error)})` : "옵션 불러오는 중…"}</div>`;
    return;
  }
  const px = optPx(o), f = optFront(o), hrs = f ? (f.exp_ms - Date.now()) / 3.6e6 : null;
  const s1d = optSigma(o, 86400), s5 = optSigma(o, 300);
  // 2026-09-29 사용자 지시: 예상 폭은 전부 DVOL 하나로(만기별 ATM IV 는 이력이 09-28 부터라 검정 불가였다).
  const sExp = f && hrs > 0 ? optSigma(o, hrs * 3600) : null;
  const vrp = optIv(o) != null && o.rv7 != null ? optIv(o) - o.rv7 : null;   // 2026-09-30 DVOL 없는 SOL·XRP 는 30일 ATM IV(예상 폭과 같은 대체) -- 전엔 늘 «-»
  const pm = (v) => (v == null ? "-" : `±${optQ(v)}$`);
  if (chip) {
    chip.hidden = !f;
    if (f) {
      chip.textContent = `옵션 · 만기 ${optKst(f.exp_ms)} (${hrs.toFixed(0)}h) · ${optUsd(f.call_oi_usd + f.put_oi_usd)} · 24h ${pm(s1d)}`;
      chip.classList.toggle("soon", hrs < 3);
      chip.title = OPT_TIPS.exp;
    }
  }
  const kv = (k, v, cls = "") => `<div class="opt-kv"><span>${k}</span><b class="${cls}">${v}</b></div>`;
  const info = optInfoParts(o, g, kv);   // 2026-10-02 옵션 정보 → 예상 폭·심리에 나눠 넣는다
  // 칸 제목을 누르면(휴대폰 포함 -- 호버가 없다) 정의·읽는 법·우리 검정 결과가 제목 아래에 펼쳐진다. 펼침은 다시 그려도 유지.
  const sec = (key, title, inner) => `<div class="opt-sec"><h4><button type="button" class="opt-q" data-tip="${key}" aria-expanded="${optTipOpen.has(key)}">${title}<span aria-hidden="true">?</span></button></h4>`
    + `<p class="opt-tip"${optTipOpen.has(key) ? "" : " hidden"}>${escapeHtml(OPT_TIPS[key]).replace(/\n/g, "<br>")}</p>${inner}</div>`;
  const gm = optDealer(o), posG = !(gm.now_usd < 0), hasG = gm.now_usd != null;   // 2026-09-29 딜러·체결 · 사다리 칩 범위
  // 2026-09-29 금액을 같이 보인다(사용자 승인) -- 0 근처에서 부호만 뒤집히는 값을 «양/음감마»로 단정하지 않게,
  //   전 만기 합의 5% 미만이면 «거의 중립». ΔDEX = 지금 − 1시간 전(수집기 dex_1h_ago, 가까운 만기는 같은 만기일 때만).
  const sgn = (v) => `${v >= 0 ? "+" : "−"}${optUsd(Math.abs(v))}`;
  // «거의 중립»도 체결 기준 -- 전 만기 커버가 낮으면 기준 자체가 약해 판정을 생략한다(옵션 세션 합의).
  const allX = (o.gamma_by || {}).all || {}, allG = allX.dealer_cov >= 0.99 ? allX.dealer_gex_usd : null;
  const neutralG = hasG && allG != null && optLadderScope !== "all" && Math.abs(gm.now_usd) < 0.05 * Math.abs(allG);
  const ago = ((g.dex_1h_ago || {})[optLadderScope]) || null;
  // 1시간 Δ: 커버 종목 수(dealer_n)와 만기 집합(exps)이 1시간 전과 같을 때만 -- 새 상장·만기 소멸·7일 경계 진입이면 «묶음 변화».
  //   (2026-09-30 검증: 옛 «커버 >0.5%p» 기준은 커버 1% 미만인 전 만기에서 같은 변화를 못 걸렀다. 미결제 증감만으로도 커버 %는 움직인다)
  const dNow = gm.dex_usd, hasSig = ago && ago.dealer_n != null && gm.n != null;
  const covMoved = hasSig && (ago.dealer_n !== gm.n || String(ago.exps) !== String(gm.exps));
  const dexD = dNow != null && hasSig && !covMoved && ago.dealer_dex_usd != null ? dNow - ago.dealer_dex_usd : null;
  const covHead = el("optCovHead");
  if (covHead) covHead.innerHTML = optCovBanner(o);
  // 2026-09-30 검증: 갱신 실패·서버 상태 지연이면 마지막 값을 그대로 두되 «지연»을 먼저 말한다
  body.innerHTML = (latestGex && latestGex.error ? `<div class="opt-note opt-warn">옵션 갱신 지연(${escapeHtml(String(latestGex.error))}) — 마지막으로 받은 값</div>` : "") + [   // 커버 줄은 제목 옆(#optCovHead) -- 아래에서 넣는다
    sec("move", "예상 폭", kv("24시간 1σ", s1d == null ? "-" : `${pm(s1d)} (${(optIv(o) / Math.sqrt(365)).toFixed(1)}%)`)
      + kv("이번 5분 1σ", pm(s5)) + (o.dvol == null && o.iv30 != null ? `<div class="opt-note">DVOL 지수가 없는 코인 — 30일 ATM IV ${o.iv30.toFixed(0)}% 로 계산</div>` : "") + kv(f ? `다음 만기까지(${hrs.toFixed(0)}h)` : "다음 만기까지", pm(sExp))
      + kv("IV(30일) − 실현(7일)", vrp == null ? "-" : `${vrp >= 0 ? "+" : ""}${vrp.toFixed(1)}pt`) + info.move),
    sec("gamma", "딜러 감마", (gm.rolled ? `<div class="opt-note">만기 교체 중 — 다음 수집(10분 안)부터 새 가까운 만기 값</div>` : "") + kv("구간", !hasG ? "-" : `${neutralG ? "거의 중립" : posG ? "양감마 · 눌림 쪽" : "음감마 · 튐 쪽"} ${sgn(gm.now_usd)}/1%`,
           !hasG || neutralG ? "" : posG ? "opt-good" : "opt-warn")
      + kv("플립", !hasG ? "-" : gm.flip ? `${optQ(gm.flip)} (${gm.flip < px ? "아래" : "위"} ${optQ(Math.abs(gm.flip - px))}$)` : "±15% 안 플립 없음(커버 종목 기준)")
      // 과거 분위는 «전체 GEX» 이력뿐이라 가까운 만기 기준과 못 견준다 -- 그 자리에 기준 만기를 보인다.
      + kv("기준", optLadderScope === "week" ? "7일 안 만기 합(사다리 칩)" : optLadderScope === "all" ? "전 만기 합(사다리 칩)"
           : gm.exp_ms ? `${optKst(gm.exp_ms)} 만기 (${Math.max(0, (gm.exp_ms - Date.now()) / 3600e3).toFixed(0)}h)` : "-")
      + kv("DEX", dNow == null ? "-" : `${sgn(dNow)}${dexD != null ? ` · 1시간 ${sgn(dexD)}` : covMoved ? " · 1시간 새 종목·만기 변화" : ""}`)
      // 2026-09-29 사용자 «charm 한 줄»: 시간만 1시간 흐를 때 딜러 델타 변화 → 딜러는 반대로 헤지한다(+ 면 매도). 체결 기반 · 서술
      + kv("charm · 다음 1시간", gm.charm == null ? "-" : Math.abs(gm.charm) < 1 ? "거의 0"
           : `헤지 ${gm.charm > 0 ? "매도" : "매수"} ${optUsd(Math.abs(gm.charm))} (딜러 델타 ${sgn(gm.charm)})`)),
    sec("ladder", "행사가 사다리", optLadderSvg(o, optColW(), fitOptLadderH)),
    sec("curve", "감마 곡선", optGammaCurveSvg(o, Math.round(document.querySelector('#optCard .opt-sec:has([data-tip="curve"])')?.clientWidth || 0) || optColW())),   // 2026-10-01 아래 줄 제 칸 폭
    // 2026-10-01 연구: 가까운 만기 최근접 RR/BF 는 잡음(1시간 SD 3.6pt) → 7일 고정만기 델타 보간(수집기 o.cm). 30일은 괄호.
    sec("mood", "심리", kv("25Δ 리스크 리버설 · 7일", optCm(o, "rr"))
      + kv("버터플라이 · 7일", optCm(o, "bf"))
      // 2026-10-01 기간 구조 = 고정만기 7·30·60일 ATM(가까운 일간 만기는 남은 시간 안에 미국장이 드느냐로 0.71~0.92배 출렁여 «역전»을 흉내 냈다)
      + kv("기간 구조 · ATM IV", ["7", "30", "60"].map((d) => ((o.cm || {})[d] || {}).atm).every((v) => v == null) ? "-"
           : ["7", "30", "60"].map((d) => { const v = ((o.cm || {})[d] || {}).atm; return `${d}일 ${v == null ? "-" : v.toFixed(0)}`; }).join(" → ")) + info.mood),
  ].join("");
  // 2026-10-01 풋프린트 호가 프로파일 아래 «옵션 요약»(사용자 «어떻게 요약하면 좋을지 연구해서»). 고른 것 = **가격 칸과 같은 언어(가격)로 말하는 것**만,
  //   근거 순: ① 24h 1σ 범위(DVOL 띠가 실현 변동성 띠보다 정확 — 09-29 재검정) ② max pain(만기 1h 전 +10.5bp 후보 — 44일, 통과 1회)
  //   ③ 딜러 감마 구간·플립(서술 — 크기 예측은 불합격) ④ DEX·charm 헤지 1시간(서술, 체결 기반) ⑤ 7일 RR(서술).
  //   머리 칩(만기 시각·규모·24h ±)과 겹치는 숫자는 뺐다. 방향 신호 아님.
  { const wall = el("mcWall");
    if (wall) {
      const pct = (v) => `${v >= px ? "+" : "−"}${(Math.abs(v / px - 1) * 100).toFixed(1)}%`;
      const row = (k, v, tag = "", cls = "") => `<div class="os-row"><span class="os-k">${k}</span><b class="os-v ${cls}">${v}</b>${tag ? `<i class="os-tag">${tag}</i>` : ""}</div>`;
      const painWin = f && f.pain && hrs != null ? (hrs <= 1 ? "규칙 창 지금" : hrs <= 6 ? `규칙 창 ${(hrs - 1).toFixed(1)}h 뒤` : "") : "";
      const html = `<div class="os"><h4 class="os-h">옵션 요약 <span>${cur} · ${optLadderScope === "front" ? "가까운 만기" : optLadderScope === "week" ? "7일 안" : "전 만기"}</span></h4>`
        + row("24h 1σ", s1d == null ? "-" : `${optQ(px - s1d)} – ${optQ(px + s1d)}`, "DVOL")
        + row("max pain", f && f.pain ? `${optQ(f.pain)} <small>${pct(f.pain)}</small>` : "-", painWin || "후보", f && f.pain ? (f.pain > px ? "opt-good" : "opt-bad") : "")
        + row("딜러 감마", !hasG ? "-" : `${neutralG ? "중립" : posG ? "양감마" : "음감마"}${gm.flip ? ` · 플립 ${optQ(gm.flip)}` : ""}`, "", !hasG || neutralG ? "" : posG ? "opt-good" : "opt-warn")
        // 2026-10-01 사용자 «DEX·헤지 매도 추가, 콜·풋 최대 제거» -- Option 카드 딜러 감마 칸과 같은 값(체결 기반 · 서술)
        + row("DEX", dNow == null ? "-" : `${sgn(dNow)}${dexD != null ? ` <small>1h ${sgn(dexD)}</small>` : ""}`, "", dNow == null ? "" : dNow >= 0 ? "opt-good" : "opt-bad")
        + row("헤지 1h", gm.charm == null ? "-" : Math.abs(gm.charm) < 1 ? "거의 0" : `${gm.charm > 0 ? "매도" : "매수"} ${optUsd(Math.abs(gm.charm))}`, "charm", gm.charm == null || Math.abs(gm.charm) < 1 ? "" : gm.charm > 0 ? "opt-bad" : "opt-good")
        + row("RR 7일", optCm(o, "rr").split(" (")[0], "", "")
        + `<div class="os-foot">서술 · 방향 신호 아님</div></div>`;
      if (wall.innerHTML !== html) wall.innerHTML = html;
    }
  }
  // 2026-09-29 풋프린트 카드 맨 아래 반반(휴대폰은 위아래) -- 두 칸 폭이 같아 한 번 잰다.
  if (laneBody && flowBody && laneBody.clientWidth > 0) {
    const lw = Math.round(laneBody.clientWidth), fw = Math.round(flowBody.clientWidth) || lw, fl = ((((latestGex || {}).block_trades || {}).flow_by_coin || {})[cur]) || [];
    // 두 그림 제목은 같은 모양(청록 ? 버튼 + 범례, 2026-09-29 사용자 «제목 포맷 통일») -- 누르면 설명이 펼쳐진다.
    const head = (k, title, legend) => `<div class="opt-flow-head"><button type="button" class="opt-q" data-tip="${k}" aria-expanded="${optTipOpen.has(k)}">${title}<span aria-hidden="true">?</span></button>`
      + `<span class="opt-lane-legend">${legend}</span></div><p class="opt-tip"${optTipOpen.has(k) ? "" : " hidden"}>${escapeHtml(OPT_TIPS[k]).replace(/\n/g, "<br>")}</p>`;
    laneBody.innerHTML = head("lane", "옵션 만기 · 지금 ~ +120h", `<b class="opt-warn">막대</b> 만기 규모(명목) · pain · P/C(미결제) · <b class="opt-c">선</b> 만기별 ATM IV · KST`)
      + optLaneSvg(o, lw);   // 2026-09-29 블록 점 제거(사용자 지시) -- 블록은 옵션 카드 «블록 거래» 칸만
    flowBody.innerHTML = (fl.length ? head("flow", "옵션 순매수 흐름 · 지난 24시간", `<b class="opt-good">콜</b> · <b class="opt-bad">풋</b> 매수−매도(${escapeHtml(cur)}) · <b class="opt-c">선</b> 체결 순델타(24h) 누적 · 테이커 기준${optNewTxt(fl.slice(0, -1)).replace(" · 미결제", " · 24h 미결제")}${fl.some((b) => b.liq > 0) ? ` · 강제청산 ${fmtNum(fl.reduce((a, b) => a + (b.liq || 0), 0), 0)}` : ""}`) + optFlowSvg(fl, fw) : "");
    const bt = (latestGex || {}).block_trades || {};
    if (blkBody) blkBody.innerHTML = head("blocks", "블록 거래 · 지난 24시간", "")
      + (bt.available === false ? `<div class="opt-note opt-warn">블록 수집 지연 — 아래 목록은 마지막으로 받은 값</div>` : "")
      // 🔴`?.[` 는 CI 문법 검사(esprima)가 못 읽는다(2026-09-30 배포 막힘) -- `(x || {})[k]` 로 쓴다.
      + optBlkHtml(blocks, (bt.n_by_coin || {})[cur], window.matchMedia("(max-width: 720px)").matches,
                   (bt.sum_by_coin || {})[cur] ?? null, (bt.max_by_coin || {})[cur] ?? null);
  }
}


async function refreshGex() {
  if (activePageTab !== "snapshot" || document.hidden) return;
  const now = Date.now();
  if (now - gexLastFetchAt < GEX_POLL_MS) return;
  gexLastFetchAt = now;
  try {
    const res = await fetch("/api/gex", { cache: "no-cache" });
    if (!res.ok) throw new Error(`gex ${res.status}`);
    latestGex = await res.json();
    renderOptions();
  } catch (error) {
    // 2026-09-30 검증: 한 번 실패로 null 을 넣으면 카드가 다음 폴링(60초)까지 통째로 사라졌다 -- 직전 값을 두고 오류만 싣는다
    console.error("GEX fetch error:", error);
    if (latestGex) { latestGex = { ...latestGex, error: String(error.message || error) }; renderOptions(); }
  }
}

async function refreshKalshi() {
  if (activePageTab !== "snapshot" || document.hidden || activeSnapshotAsset !== "eth") return;
  const now = Date.now();
  if (now - kalshiLastFetchAt < KALSHI_POLL_MS) return;
  kalshiLastFetchAt = now;
  try {
    const res = await fetch("/api/kalshi", { cache: "no-cache" });
    if (!res.ok) throw new Error(`kalshi ${res.status}`);
    latestKalshi = await res.json();
  } catch (error) {
    latestKalshi = null;   // 선을 지운다 -- 묵은 확률을 지금 값처럼 두지 않는다
  }
}

// 칼시 창이 지금 유효하면 {k, pct, color, word}. 서버가 10초 넘게 못 받았거나 창이 끝났으면 null.
function kalshiNow() {
  const k = latestKalshi;
  if (activeSnapshotAsset !== "eth" || !k || !k.ok || !(k.strike > 0) || k.p == null) return null;
  const nowS = Date.now() / 1000;
  if (nowS - k.fetched_ts > 10 || nowS >= k.close_ts) return null;
  const pct = Math.round(k.p * 100);
  // 5pp 안쪽은 «비등» -- 방향색을 주지 않는다(DESIGN.md 비등 규칙과 같은 폭).
  const up = pct >= 55, dn = pct <= 45;
  return { k, pct, color: up ? "var(--good)" : dn ? "var(--bad)" : "var(--muted)",
           word: dn ? `아래 ${100 - pct}%` : up ? `위 ${pct}%` : `비등 위 ${pct}%` };
}

async function refreshFlowHeatmap() {
  if (activePageTab !== "snapshot" || document.hidden) return;
  if (!flowOn()) return;   // 래스터 수집기가 있는 코인만
  const now = Date.now();
  if (now - flowHeatmapLastFetchAt < flowHeatmapPollMs()) return;
  flowHeatmapLastFetchAt = now;
  try {
    const asset = activeSnapshotAsset;
    const res = await fetch(
      `/api/flow/heatmap?symbol=${asset}usdt&cols=${FLOW_HEATMAP_COLS}`
      + `&agg=${flowHeatmapAgg()}&mode=rows`,
      { cache: "no-cache" });
    if (!res.ok) throw new Error(`flow-heatmap ${res.status}`);
    const j = await res.json();
    if (asset !== activeSnapshotAsset) return;
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
    if (j.book && j.book.q_f4) j.book.q = f4(j.book.q_f4);   // 2026-09-27 풋프린트 실시간 호가 띠(마지막 1초, +매수/−매도)
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
  } catch (error) {
    console.error("Flow heatmap fetch error:", error);
    latestFlowHeatmap = null;
  }
}


async function refreshOi5m() {
  if (activePageTab !== "snapshot" || document.hidden) return;
  if (!flowOn()) return;   // 흐름 엔진이 있는 코인만
  const now = Date.now();
  if (now - oi5mLastFetchAt < OI_5M_POLL_MS) return;
  oi5mLastFetchAt = now;
  try {
    // 🔴2026-09-25: 주석이 «최대 72봉» 이라 96 을 받고 있었는데 창 토글은 그 뒤 **144봉(12시간)**
    //   으로 늘었다(CHART_WINDOW_BARS). 12시간을 고르면 앞 48봉이 OI 없이 그려졌고, 아래
    //   사분면이 `|| 0` 으로 받아 **전부 «신규 롱/숏» 으로 가짜 라벨**이 붙었다(OI=0 은 o>=0 이다).
    //   창 최대치에서 파생시킨다 -- 토글을 늘리면 여기가 자동으로 따라온다. +4 는 Δ 의 기준봉 여유.
    const oiBarsWanted = Math.max(...CHART_WINDOW_BARS) + 4;
    const asset = activeSnapshotAsset;
    const res = await fetch(`${API_OI_5M_URL}?asset=${asset}&bars=${oiBarsWanted}`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`oi-5m ${res.status}`);
    const j = await res.json();
    if (asset !== activeSnapshotAsset) return;
    latestOi5m = j;
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
  // 🔴2026-09-22 코인 게이트를 없앴다. 이 카드는 ETH 전용인데(서버 엔드포인트에 asset 이 없다)
  //   게이트가 있으면 비ETH 에서 **마지막 ETH 값이 얼어붙은 채** 남아, 보는 사람은 그게 지금
  //   고른 코인의 상황이라고 읽는다. 멈춘 옛 값보다 «살아 있는 ETH 값 + ETH 전용 배지»가 정직하다.
  //   비용은 1초마다 작은 JSON 하나이고 서버는 어차피 계산하고 있다.
  const now = Date.now();
  if (liveStreamOn()) return;                                    // /api/stream 이 밀어주는 중
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

async function refreshTrend() {
  if (activePageTab !== "snapshot" || document.hidden) return;
  const now = Date.now();
  if (now - trendLastFetchAt < TREND_POLL_MS) return;
  trendLastFetchAt = now;
  try {
    const res = await fetch(API_TREND_URL, { cache: "no-cache" });
    latestTrend = res.ok ? await res.json() : { ok: false, reason: `HTTP ${res.status}` };
  } catch (error) {
    latestTrend = { ok: false, reason: "fetch_failed" };
  }
  renderSituation();
}

// ── 시장 맥락 (2026-09-29, ETH) ───────────────────────────────────────────
// 레포트(ETH 오더플로 105항목) 대조 뒤 사용자 «불합격된 것들도 쓸모 있으면 넣어줘 -- 우리 검증 방식이 틀렸을 수도 있다».
// 서버 /api/market-context(dashboard/market_ctx.py)를 그대로 그린다. 값은 전부 **서술** -- 방향 판정은 30분 카드(융합)가 한다.
// 칸 제목의 «?» = 정의 · 읽는 법 · 우리 검정의 말(DESIGN.md: 매매 화면 본문에 연구 원문 숫자를 올리지 않는다).
const API_MARKET_CTX_URL = "/api/market-context";
const MARKET_CTX_POLL_MS = 5000;
let latestMarketCtx = null, marketCtxLastFetchAt = 0;
let latestMacroEvents = null;        // renderMacroCalendar 가 채운다 -- 카드의 «다음 주요 일정». null = 아직 못 받음(«없음»과 다르다)
const mcTipOpen = new Set();
const mcLine = { vwap: true, bb: false, wvwap: false, vsess: false };   // wvwap = 주간·앵커 VWAP(2026-09-30, 기본 끔)   // 차트 선(카드 «가격 위치»의 스위치). 볼린저는 우리 검정에서 방향 엣지가 없어 기본 끔
try { Object.assign(mcLine, JSON.parse(localStorage.getItem("mcLine") || "{}")); } catch (err) { /* 기억은 편의 */ }
// 2026-09-30 사용자 «세션은 미국·유럽·아시아 시장 VWAP 으로» -- vsess 켜면 지금 시장 세션 개장(현지 시각, 서머타임 반영)부터 센 VWAP.
//   서버가 봉마다 두 벌을 싣는다: vwap/vsd = UTC 00시부터 · svwap/svsd/sstart/sname = 세션 개장부터. seg 가 바뀌는 곳에서 선을 끊는다.
function mcVwapOf(c) {
  if (mcLine.vsess) return c && c.svwap ? { vwap: c.svwap, vsd: c.svsd || 0, seg: c.sstart, name: c.sname || "세션" } : null;
  return c && c.vwap ? { vwap: c.vwap, vsd: c.vsd || 0, seg: Math.floor(c.time / 86400), name: "하루" } : null;
}
const MC_TIPS = {
  lev: "펀딩 = 8시간마다 롱과 숏이 주고받는 이자(양수면 롱이 낸다). 바이낸스는 평온할 때 0.01%에 붙어 거의 안 움직여서 «비싸게 들고 있나»는 베이시스(선물 마크 − 현물 인덱스)의 7일 분위로 본다. OI = 열린 계약 수, z = 1시간 변화가 지난 7일 중 얼마나 튀었나.\n우리 검정: 극단 펀딩 뒤 가격 방향은 없었다(표본 충분). 그래도 넣은 이유 — 청산 연쇄는 쏠린 쪽에서 나고, 상태 줄은 그 쏠림을 말한다. 방향 신호로 쓰지 않는다.",
  ls: "바이낸스 5분 통계: 계정 롱숏비(롱 계정 수 ÷ 숏 계정 수) · 탑트레이더 포지션 비율 · 테이커 매수 ÷ 매도. 괄호 = 하루 전.\n우리 검정: 30분 방향 모델에 넣으면 오히려 조금 나빠졌고 학술 근거도 없다. 군중이 어느 쪽에 몰렸나를 눈으로 보는 용도.",
  quad: "1시간 가격 이동 × OI 변화. 이동이 하루 1시간 이동의 상위 25% 안일 때만 판정한다.\n우리 검정(4.7년): 하락 + OI↑(새 숏 유입) 뒤 1시간은 더 내렸고, 하락 + OI↓(롱 정리) 뒤는 되돌렸다 — 숏을 «언제 거두나»의 근거. 상승 쪽 두 칸은 아직 안 쟀다. 차트 아래 사분면(5분 델타 × OI)은 방향 정보가 0으로 나온 서술용이다.",
  flow: "크기별 순매수 = 테이커 주문 크기로 가른 60분 순매수를, 지난 14일 같은 시간대(UTC 시) 60분 순매수의 보통 크기(중앙값 편차)로 나눈 값(σ) — 한산한 새벽과 미국장이 같은 잣대로 보인다(기준이 아직 없으면 서버 기동 후 12~24시간 분포). 고래 ≥ $10만 · 리테일 < $1만. 30분 체결 = 거래소별 순매수(ETH).\n우리 검정: 혼자서는 셋 다 되돌림 쪽(리테일이 가장 심하다). 고래와 리테일이 갈릴 때 60분 고래 쪽이 약하게 맞았다(메이커 전제). CVD는 «설명»이지 선행지표가 아니다.",
  cross: "BTC 30분 이동과 ETH의 동행 여부 · 거래소 가격차(바이낸스 기준 — OKX 는 마크 대 마크, HL 은 미드 대 미드. 마크는 평활값이라 미드와 섞어 재면 괴리가 부풀어 보인다) · HL 프리미엄(마크 − 오라클).\n우리 검정: BTC→ETH 1분 선행 0, 바이낸스가 HL을 1초 안쪽으로 앞선다 — 30분~2시간 방향에는 못 쓴다. 괴리가 커지는 순간(거래소 장애·한쪽 청산 쏠림)을 알아채는 용도.",
  book: "미드에서 ±25/50/100bp까지 걸린 호가 합 = 그만큼 밀려면 먹어야 할 물량(한 초 스냅샷 · 취소·재보충은 모른다). 얇은 쪽 = ±25bp 호가가 지난 6시간 중 몇 분위인가.\n우리 검정: 미검정. 거래량 계열은 실현변동성에 대부분 먹혔으니 «크기·속도» 참고로만. ETH 스프레드는 거의 늘 1틱이라 벌어지면 그 자체가 이상 신호다.",
  liq: "청산 급증 = 봇의 1분 청산 z. HL 고래 청산가 = 추적 중인 HL 상위 300주소의 실제 청산가(추정 아님)를 $5 단위로 묶은 금액 — 차트에 점선으로도 그린다. 12시간 실측 = 바이낸스 강제청산이 실제로 체결된 가격(차트 체결 기둥 안쪽 눈금).\n우리 검정: 청산 급증 뒤 역매매·추종 둘 다 엣지 0 — «청산 동반 급등에 역매매 금지» 필터만 유효. 추정 청산맵은 «위치»는 맞고 방향은 없다. HL 실측 청산가는 미검정.",
  when: "다음 펀딩 정산 · 가까운 옵션 만기와 max pain · 미국장 · 다음 주요 지표.\n우리 검정: 펀딩 정산 전후 드리프트는 반기마다 부호가 뒤집혀 기각. max pain 쪽 1시간 규칙(만기 1시간 전 → 08:00 UTC)만 2026년 첫 검정 통과 — 후보라 표본외 장부로 계속 잰다.",
  px: "VWAP = 거래량 가중 평균가(σ = 거래량 가중 표준편차). 기본은 하루(UTC 00시부터) · «세션 시작부터»를 켜면 지금 시장 세션 개장부터 센다(아시아 도쿄 09시 · 유럽 런던 08시 · 미국 뉴욕 09:30, 현지 시각 · 서머타임 반영 · 미국장 뒤는 다음 00시까지 미국 세션). 개장 직후 30~60분은 봉이 적어 가격에 붙어 다닌다. 볼린저 %B = (종가 − 하단) ÷ (상단 − 하단), 20봉·2σ. RSI 14 = 5분봉. 스위치로 차트에 선을 켠다.\n우리 검정: 볼린저·VWAP 셋업은 방향 엣지가 없었고 RSI·%B는 짧은 되돌림을 약하게 말한다(비용을 못 넘음). 위치 참고용.",
};
// 2026-09-30 시안 B 판 설명 = 옛 칸 설명을 판 단위로 묶은 것(문장은 그대로).
MC_TIPS.q_lev = MC_TIPS.lev + "\n\n" + MC_TIPS.ls;
MC_TIPS.q_flow = MC_TIPS.flow + "\n\n" + MC_TIPS.quad;
MC_TIPS.q_map = MC_TIPS.liq + "\n\n" + MC_TIPS.px;
MC_TIPS.q_wall = MC_TIPS.book + "\n\n" + MC_TIPS.cross + "\n\n" + MC_TIPS.px;
// 시안 B 그림 부품 -- 전부 SVG 문자열. 폭은 고정(칸 격자가 줄을 맞춘다), 색은 3색 규칙(방향만 초록·빨강, 극단 구간만 주황).
const mcGfx = {
  pct(p, w = mcGfx.gw || 130) {   // 분위 0~1 · 양끝 10% 주황 띠 · 값 없으면 빈 막대
    const x = 4 + (w - 8) * Math.max(0, Math.min(1, p ?? 0));
    return `<svg class="mc-svg" width="${w}" height="16" viewBox="0 0 ${w} 16" aria-hidden="true"><rect x="4" y="6" width="${w - 8}" height="4" rx="2" fill="rgb(var(--lift) / .12)"/>`
      + `<rect x="4" y="6" width="${(w - 8) * 0.1}" height="4" rx="2" fill="var(--warn)" opacity=".4"/><rect x="${4 + (w - 8) * 0.9}" y="6" width="${(w - 8) * 0.1}" height="4" rx="2" fill="var(--warn)" opacity=".4"/>`
      + `<line x1="${w / 2}" x2="${w / 2}" y1="3" y2="13" stroke="var(--line)"/>${p == null ? "" : `<circle cx="${x.toFixed(1)}" cy="8" r="5" fill="var(--text)"/>`}</svg>`;
  },
  sig(zv, w = mcGfx.gw || 130) {   // σ −3~+3 · 0 가운데 · ±2σ 점선
    const c = w / 2, s = (w / 2 - 4) / 3, v = zv == null ? 0 : Math.max(-3, Math.min(3, zv)), x0 = v >= 0 ? c : c + v * s;
    const fill = zv == null || Math.abs(zv) < 0.5 ? "var(--muted)" : zv > 0 ? "var(--good)" : "var(--bad)";
    return `<svg class="mc-svg" width="${w}" height="16" viewBox="0 0 ${w} 16" aria-hidden="true"><rect x="4" y="7" width="${w - 8}" height="2" fill="rgb(var(--lift) / .1)"/>`
      + [-2, 2].map((q) => `<line x1="${c + q * s}" x2="${c + q * s}" y1="3" y2="13" stroke="var(--line)" stroke-dasharray="2 2"/>`).join("")
      + `<line x1="${c}" x2="${c}" y1="2" y2="14" stroke="var(--muted)"/>${zv == null ? "" : `<rect x="${x0.toFixed(1)}" y="4" width="${Math.max(1.5, Math.abs(v) * s).toFixed(1)}" height="8" rx="2" fill="${fill}"/>`}</svg>`;
  },
  ratio(now, prev, w = mcGfx.gw || 130, lo = 0.5, hi = 4) {   // 로그 눈금 · 1 가운데 · 빈 점 = 하루 전
    if (now == null) return mcGfx.pct(null, w);
    const X = (v) => 4 + (w - 8) * Math.max(0, Math.min(1, Math.log(v / lo) / Math.log(hi / lo)));
    return `<svg class="mc-svg" width="${w}" height="16" viewBox="0 0 ${w} 16" aria-hidden="true"><rect x="4" y="7" width="${w - 8}" height="2" fill="rgb(var(--lift) / .1)"/><line x1="${X(1)}" x2="${X(1)}" y1="2" y2="14" stroke="var(--muted)"/>`
      + (prev ? `<line x1="${X(prev)}" x2="${X(now)}" y1="8" y2="8" stroke="var(--muted)"/><circle cx="${X(prev)}" cy="8" r="4" fill="var(--panel)" stroke="var(--muted)"/>` : "")
      + `<circle cx="${X(now)}" cy="8" r="5" fill="var(--text)"/></svg>`;
  },
  seg4(key, w = mcGfx.gw || 130) {   // 2026-09-30 가격×OI 사분면 한 줄: 네 칸 중 지금 칸만 채움(small·없음 = 전부 빈 칸)
    const cells = [["up_up", "신규 롱"], ["up_dn", "숏커버"], ["dn_up", "신규 숏"], ["dn_dn", "롱정리"]], cw = (w - 6) / 4;
    return `<svg class="mc-svg" width="${w}" height="16" viewBox="0 0 ${w} 16" aria-hidden="true">` + cells.map(([k, t], i) => {
      const on = k === key, col = k === "dn_up" ? "var(--bad)" : k === "dn_dn" ? "var(--good)" : "var(--ink)", x = i * (cw + 2);
      return `<rect x="${x.toFixed(1)}" y="1" width="${cw.toFixed(1)}" height="14" rx="3" fill="${on ? col : "rgb(var(--lift) / .08)"}" fill-opacity="${on ? 0.85 : 1}"/>`
        + (cw >= 26 ? `<text x="${(x + cw / 2).toFixed(1)}" y="11.5" font-size="9" font-weight="700" text-anchor="middle" fill="${on ? "var(--chart-bg)" : "var(--muted)"}">${t}</text>` : "");
    }).join("") + `</svg>`;
  },
  quad(move, oiPct, w = mcGfx.qw || 96) {   // 가격(x, 1시간 bp ±60) × OI(y, 1시간 % ±1) -- 가운데 원 = «이동 작음, 판정 보류»
    const c = w / 2, x = c + Math.max(-1, Math.min(1, (move || 0) / 60)) * (c - 10), y = c - Math.max(-1, Math.min(1, (oiPct || 0) / 1)) * (c - 10);
    return `<svg class="mc-svg" width="${w}" height="${w}" viewBox="0 0 ${w} ${w}" style="width:${w}px;height:${w}px" aria-hidden="true"><rect x="1" y="1" width="${w - 2}" height="${w - 2}" rx="6" fill="none" stroke="var(--line)"/>`
      + `<line x1="${c}" x2="${c}" y1="4" y2="${w - 4}" stroke="var(--line)"/><line x1="4" x2="${w - 4}" y1="${c}" y2="${c}" stroke="var(--line)"/>`
      + `<text x="${w - 5}" y="13" font-size="9" fill="var(--muted)" text-anchor="end">신규 롱</text><text x="5" y="13" font-size="9" fill="var(--muted)">신규 숏</text>`
      + `<text x="5" y="${w - 5}" font-size="9" fill="var(--muted)">롱 정리</text><text x="${w - 5}" y="${w - 5}" font-size="9" fill="var(--muted)" text-anchor="end">숏 정리</text>`
      + `<circle cx="${c}" cy="${c}" r="${((c - 10) * 0.25).toFixed(1)}" fill="rgb(var(--lift) / .07)"/>${move == null ? "" : `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6" fill="var(--text)"/>`}</svg>`;
  },
  book(sw, bps, w = mcGfx.pw || 300) {   // ±bp 매수벽 | 매도벽 거울 막대(ETH)
    const b = sw.bid, a = sw.ask, mx = Math.max(1, ...b, ...a), c = w / 2, h = 4 + bps.length * 21;
    const kq = (v) => (v >= 1e3 ? (v / 1e3).toFixed(1) + "k" : String(Math.round(v)));
    return `<svg class="mc-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="호가 벽 매수 대 매도">` + bps.map((bp, i) => {
      const y = 4 + i * 21, bw = (c - 64) * b[i] / mx, aw = (c - 64) * a[i] / mx;   // 64 = 가장 긴 벽의 «80.4k» 글자 자리
      return `<rect x="${(c - 18 - bw).toFixed(1)}" y="${y}" width="${bw.toFixed(1)}" height="13" rx="2" fill="var(--good)" opacity=".75"/><rect x="${c + 18}" y="${y}" width="${aw.toFixed(1)}" height="13" rx="2" fill="var(--bad)" opacity=".75"/>`
        + `<text x="${c}" y="${y + 10}" font-size="10" fill="var(--muted)" text-anchor="middle">±${bp}</text>`
        + `<text x="${(c - 22 - bw).toFixed(1)}" y="${y + 10}" font-size="10" fill="var(--text)" text-anchor="end">${kq(b[i])}</text><text x="${(c + 22 + aw).toFixed(1)}" y="${y + 10}" font-size="10" fill="var(--text)">${kq(a[i])}</text>`;
    }).join("") + `</svg>`;
  },
  timeline(events, w = 900, h = 48) {   // 다음 24시간 -- 24시간 넘는 일정은 오른쪽 끝에 «>»
    const now = Date.now(), X2 = (hr) => 12 + (w - 24) * Math.min(24, hr) / 24;
    if (w < 600) {   // 휴대폰: 이름표가 서로 겹쳐 목록으로(같은 순서 · 남은 시간)
      return events.filter((e) => e.t > now).sort((a, c) => a.t - c.t).map((e) => `<div class="mc-tlrow"><i></i><b>${fmtHourMinute(e.t)}</b> ${escapeHtml(e.nm)} <span>+${((e.t - now) / 3.6e6).toFixed(1)}h</span></div>`).join("");
    }
    let s = `<line x1="12" x2="${w - 12}" y1="22" y2="22" stroke="var(--line)"/>`;
    [0, 6, 12, 18, 24].forEach((q) => { s += `<text x="${X2(q)}" y="${h - 2}" font-size="10" fill="var(--muted)" text-anchor="${q ? (q === 24 ? "end" : "middle") : "start"}">${q ? "+" + q + "h" : "지금"}</text>`; });
    const lastX = [-1e9, -1e9];   // 위·아래 줄마다 마지막 이름표 오른쪽 끝 -- 겹치면 그 이름표는 점의 호버로만
    events.filter((e) => e.t > now).sort((a, c) => a.t - c.t).forEach((e, i) => {
      const hr = (e.t - now) / 3.6e6, x = X2(hr), far = hr > 24, row = i % 2, lab = `${e.nm} ${fmtHourMinute(e.t)}${far ? " >" : ""}`;
      const col = e.hi === false ? "var(--muted)" : "var(--warn)", tw = lab.length * 10.5 * 0.62, anc = x > w - 90 ? "end" : x < 90 ? "start" : "middle";
      const l0 = anc === "end" ? x - tw : anc === "start" ? x : x - tw / 2;
      s += `<circle cx="${x.toFixed(1)}" cy="22" r="5" fill="${col}"${far ? ' opacity=".45"' : ""}><title>${escapeHtml(lab)}</title></circle>`;
      if (l0 > lastX[row] + 8) { s += `<text x="${x.toFixed(1)}" y="${row ? 38 : 12}" font-size="10.5" fill="var(--text)" text-anchor="${anc}">${escapeHtml(lab)}</text>`; lastX[row] = l0 + tw; }
    });
    return `<svg class="mc-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="다음 24시간 일정">${s}</svg>`;
  },
};

async function refreshMarketCtx() {
  if (activePageTab !== "snapshot" || document.hidden) return;
  const now = Date.now();
  if (now - marketCtxLastFetchAt < MARKET_CTX_POLL_MS) return;
  marketCtxLastFetchAt = now;
  try {
    const res = await fetch(API_MARKET_CTX_URL, { cache: "no-cache" });
    latestMarketCtx = res.ok ? await res.json() : { available: false, error: `HTTP ${res.status}` };
  } catch (error) {
    latestMarketCtx = { available: false, error: "fetch_failed" };
  }
  renderMarketCtx();
}

// 🔴2026-09-30 청산 «급증»은 **쪽마다** 판정한다. 옛 판은 `z>=3 || hawkes_active` 를 롱·숏 막대에 똑같이 걸어
//   hawkes_active 하나로 z −0.24 인 숏 막대까지 주황·«급증»이 됐다. 규칙: 그 쪽 z>=3 이거나, hawkes 가 켜져 있고
//   그 쪽이 1분 금액이 0 이 아닌 **큰 쪽**일 때. 파일(tail_risk_interceptor 10초마다 씀)이 60초 넘게 낡았으면 급증 없음.
//   순수 함수(test/test_mc_liq_burst_hot_20260930.py 가 본문을 떼어 돌린다).
const LIQ_BURST_STALE_MS = 60000;
function mcLiqBurstHot(bu, side, nowMs = Date.now()) {
  if (!bu || !(nowMs - Date.parse(bu.updated_at || "") <= LIQ_BURST_STALE_MS)) return false;
  const lu = bu.long_usd_1m || 0, su = bu.short_usd_1m || 0;
  const [z, mine, other] = side === "long" ? [bu.z_long, lu, su] : [bu.z_short, su, lu];
  return (z || 0) >= 3 || (!!bu.hawkes_active && mine > 0 && mine >= other);
}
// 꼬리표·툴팁이 말할 쪽: 급증인 쪽, 둘 다면 1분 금액이 큰 쪽. 없으면 null.
function mcLiqBurstSide(bu, nowMs = Date.now()) {
  const hl = mcLiqBurstHot(bu, "long", nowMs), hs = mcLiqBurstHot(bu, "short", nowMs);
  if (!hl && !hs) return null;
  return hs && (!hl || (bu.short_usd_1m || 0) > (bu.long_usd_1m || 0)) ? "short" : "long";
}

// 볼린저(20, 2σ) -- {봉 시각: [하단, 중앙, 상단]}. 창 슬라이스(1h = 12봉)로는 20봉이 안 되니 전체 이력에서 센다.
function mcBollinger(full, n = 20, k = 2) {
  const out = new Map();
  for (let j = n - 1; j < full.length; j++) {
    let s = 0, s2 = 0;
    for (let q = j - n + 1; q <= j; q++) { const v = +full[q].close; s += v; s2 += v * v; }
    const m = s / n, sd = Math.sqrt(Math.max(0, s2 / n - m * m));
    out.set(full[j].time, [m - k * sd, m, m + k * sd]);
  }
  return out;
}

// RSI(14, 와일더). 값이 모자라면 null.
function mcRsi(closes, n = 14) {
  if (closes.length <= n) return null;
  let up = 0, dn = 0;
  for (let i = 1; i <= n; i++) { const d = closes[i] - closes[i - 1]; if (d > 0) up += d; else dn -= d; }
  up /= n; dn /= n;
  for (let i = n + 1; i < closes.length; i++) {
    const d = closes[i] - closes[i - 1];
    up = (up * (n - 1) + Math.max(d, 0)) / n; dn = (dn * (n - 1) + Math.max(-d, 0)) / n;
  }
  return dn === 0 ? 100 : 100 - 100 / (1 + up / dn);
}

// 다음 미국장 개장·마감(뉴욕 09:30~16:00, 평일). 서머타임은 브라우저의 시간대 표로 가린다 -- 13:30/14:30 UTC 중 뉴욕 09:30 인 쪽.
function mcUsSession(nowMs = Date.now()) {
  const fmt = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", hourCycle: "h23", weekday: "short", hour: "2-digit", minute: "2-digit" });
  const ny = (t) => Object.fromEntries(fmt.formatToParts(t).map((p) => [p.type, p.value]));   // 브라우저마다 쉼표·공백이 달라 문자열로 안 비교한다
  const d0 = new Date(nowMs);
  for (let k = -1; k < 5; k++) {
    for (const h of [13, 14]) {
      const open = Date.UTC(d0.getUTCFullYear(), d0.getUTCMonth(), d0.getUTCDate() + k, h, 30), s = ny(open);
      if (!(["Mon", "Tue", "Wed", "Thu", "Fri"].includes(s.weekday) && s.hour === "09" && s.minute === "30")) continue;
      const close = open + 6.5 * 3600e3;
      if (nowMs < close) return { open, close, live: nowMs >= open };
    }
  }
  return null;
}

// 2026-09-30 시장 맥락 자리(시안 Y): 넓은 화면 2단이면 풋프린트 SVG 의 오른쪽 아래 칸(viewBox 좌표 = 화면 px)에 절대 위치로 겹치고 2열 압축,
//   그 밖(1단·휴대폰)이면 풋프린트 카드 안 차트 아래 일반 흐름. 칸이 바뀔 때만 다시 그린다.
let mcNeedH = FIT0.mc || 0;   // 시장 맥락(좁은 칸) 내용 높이 -- renderMarketCtx 가 재고 renderCandleSvg 가 1초 수급 몫을 정할 때 쓴다
function mcPlace(svg, r, wr) {
  const body = el("mcBody"), card = el("fpCard"), wall = el("mcWall");
  if (!body || !card) return;
  if (wall) {   // 2026-09-30 ④ 자리 = 호가/체결 프로파일 아래
    let wp = { left: "", top: "", width: "", height: "" };
    if (wr) {
      const a = svg.getBoundingClientRect(), c = card.getBoundingClientRect();
      wp = { left: `${Math.round(a.left - c.left + wr.x)}px`, top: `${Math.round(a.top - c.top + wr.y)}px`, width: `${Math.round(wr.w)}px`, height: `${Math.max(0, Math.round(wr.h))}px` };
    }
    Object.entries(wp).forEach(([k, v]) => { if (wall.style[k] !== v) wall.style[k] = v; });
    wall.classList.toggle("on", !!wr);
  }
  let pos = { left: "", top: "", width: "", height: "" };
  if (r) {
    const a = svg.getBoundingClientRect(), c = card.getBoundingClientRect();
    pos = { left: `${Math.round(a.left - c.left + r.x)}px`, top: `${Math.round(a.top - c.top + r.y)}px`, width: `${Math.round(r.w)}px`, height: `${Math.max(0, Math.round(r.h))}px` };
  }
  // 🔴cssText 로 통째로 쓰면 renderMarketCtx 가 못 박은 gridTemplateColumns 가 지워진다 -- 위치 네 값만 쓴다
  Object.entries(pos).forEach(([k, v]) => { if (body.style[k] !== v) body.style[k] = v; });
  if (body.classList.contains("mc-cmp") !== !!r) { body.classList.toggle("mc-cmp", !!r); body._mcHtml = null; renderMarketCtx(); }
}

function renderMarketCtx() {
  const body = el("mcBody"), badge = el("mcBadge");
  if (!body) return;
  const d = latestMarketCtx;
  if (!d || !d.available) {
    body.innerHTML = `<div class="mc-note">${d && d.error ? `시장 맥락 지연 (${escapeHtml(String(d.error))})` : "불러오는 중…"}</div>`;
    body._mcHtml = null;
    if (badge) { badge.textContent = d && d.error ? "지연" : "-"; badge.className = "ops-badge neutral"; }   // 옛 «쏠림» 배지가 남지 않게
    return;
  }
  const n = (v, dp = 0) => (v == null || !Number.isFinite(+v) ? "-" : (+v).toLocaleString("en-US", { maximumFractionDigits: dp, minimumFractionDigits: dp }));
  const sg = (v, dp = 1, u = "") => (v == null || !Number.isFinite(+v) ? "-" : `${+v >= 0 ? "+" : "−"}${n(Math.abs(v), dp)}${u}`);
  const fr = (v) => (v == null ? "-" : `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(4)}%`);
  const left = (ms) => { if (!ms) return "-"; const s = Math.max(0, (ms - Date.now()) / 1000); return s >= 3600 ? `${Math.floor(s / 3600)}시간 ${Math.floor((s % 3600) / 60)}분` : `${Math.ceil(s / 60)}분`; };
  const usd = (v) => (v == null ? "-" : fmtUsdCompact(v));
  const tone = (v, thr = 0.5) => (v == null || Math.abs(v) < thr ? "" : v > 0 ? "mc-good" : "mc-bad");
  const sec = (key, title, inner) => `<div class="mc-sec"><h4><button type="button" class="mc-q" data-tip="${key}" aria-expanded="${mcTipOpen.has(key)}">${title}<span aria-hidden="true">?</span></button></h4>`
    + `<p class="mc-tip"${mcTipOpen.has(key) ? "" : " hidden"}>${escapeHtml(MC_TIPS[key]).replace(/\n/g, "<br>")}</p>${inner}</div>`;
  const f = d.funding || {}, b = d.basis || {}, oi = d.oi || {}, ls = d.ls, fl = d.flow || {}, z = fl.z60h || fl.z60 || {},   // z60h = 같은 UTC 시 14일 기준(2026-10-01), 없으면 옛 12~24h 링
    bk = d.book || {}, bu = d.burst;
  const levWarn = ["long_crowd", "short_crowd", "deleverage"].includes((d.lev || {}).key);
  // 가격 위치 -- 차트와 같은 캔들 이력(서버가 봉마다 vwap/vsd 를 싣는다)
  const full = candleHistoryByAsset.eth || [], lc = full[full.length - 1];
  const lvC = [...full].reverse().find((c) => mcVwapOf(c)), lv = mcVwapOf(lvC);      // 형성 중 봉(클라가 붙인다)엔 vwap 이 없다 -- 마감봉 값
  const bb = lc ? mcBollinger(full).get(lc.time) : null, px = Number(latestLivePriceByAsset.eth || lc?.close || d.mid || 0);
  const pb = bb && bb[2] > bb[0] ? (px - bb[0]) / (bb[2] - bb[0]) : null, rsi = mcRsi(full.map((c) => +c.close));
  const vz = lv && lv.vsd > 0 ? (px - lv.vwap) / lv.vsd : null;
  // 일정 -- 옵션 만기는 옵션 카드와 같은 원천(latestGex). 🔴optData() 는 «지금 보는 코인»이라 SOL 탭에서 SOL max pain 을 ETH 가격과
  //   견줬다(09-29 재검증) -- 이 카드는 ETH 전용이므로 ETH 를 직접 읽는다.
  const gEth = latestGex && latestGex.available ? (latestGex.currencies || {}).ETH : null, o = gEth && gEth.options;
  const ef = o ? optFront(o) : null, us = mcUsSession();
  // 2026-09-30 사용자 «주요 경제 일정이 왜 24시간 축에 없나» -- 전에는 «다음 high 하나»만 실었다. 경제 일정 카드와 같은 원천의 24시간 안 전부.
  const macro24 = (latestMacroEvents || []).map((e) => ({ t: Date.parse(e.time_utc), nm: e.title_ko || e.title || "지표", hi: e.importance === "high" }))
    .filter((e) => e.t > Date.now() && isTodayOrTomorrowLocal(e.t));   // 2026-09-30 경제 일정 카드 제거 -- 카드가 보이던 오늘·내일 전부(24시간 넘는 건 시간축 오른쪽 끝 «>»)
  const prof = d.liq_profile || [], top = prof.reduce((m, r) => (r[1] + r[2] > (m ? m[1] + m[2] : 0) ? r : m), null);
  const sw = bk.sweep, bps = bk.bps || [25, 50, 100];
  const thin = bk.bid25_pct != null && bk.ask25_pct != null && Math.min(bk.bid25_pct, bk.ask25_pct) <= 0.1
    ? (bk.ask25_pct <= bk.bid25_pct ? "위쪽(매도호가)이 얇다" : "아래쪽(매수호가)이 얇다") : null;
  const burstSide = mcLiqBurstSide(bu);
  if (badge) {
    badge.textContent = (d.lev || {}).label || "-";
    badge.className = `ops-badge ${levWarn ? "warn" : "neutral"}`;
  }
  // ── 2026-09-30 시안 B(사용자 선택 «글자가 너무 많다 → 그릴 수 있는 건 그림으로»): 질문 네 판 + 다음 24시간 ──
  //   분위 = 0~100 게이지(양끝 10% 주황) · σ = 0 가운데 막대(±2σ 점선) · 비율 = 1 가운데 로그 눈금(빈 점 = 하루 전).
  //   그림 없는 부값(OKX/HL 펀딩·OI 합계·거래소별 30분 체결·스프레드·12시간 청산 최다·청산 급증)은 판마다 흐린 한 줄로 남긴다.
  // 2026-09-30 사용자 «카드 너비 100% 안을 내용 크기를 키워 채워» -- 판 폭(격자 auto-fit 300px 칸)에서 그림 폭을 정한다.
  { const cmp = body.classList.contains("mc-cmp"), gap = cmp ? 14 : 22;   // 2026-09-30 시안 Y: 풋프린트 1초 수급 아래 칸(~420px)에 2열 압축
    const BW = body.clientWidth || 1200, cols = cmp ? 2 : Math.max(1, Math.min(4, Math.floor((BW + 22) / 322)))   /* 판은 넷 -- auto-fit 이 빈 칸을 접어 넷이 폭을 나눠 가진다 */, colW = Math.floor((BW - (cols - 1) * gap) / cols);
    body.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;   // 🔴auto-fit 은 전폭 줄(⑤)이 있으면 빈 칸을 못 접어 5칸이 됐다 -- 계산과 같은 칸 수로 못 박는다
    mcGfx.pw = Math.min(colW, 560); mcGfx.gw = cmp ? Math.max(48, colW - 64 - 46 - 12) : Math.max(110, Math.min(300, colW - 96 - 90 - 16));
    mcGfx.qw = cmp ? Math.max(70, Math.min(100, Math.round(colW * 0.42))) : Math.max(96, Math.min(150, Math.round(colW * 0.3)));
    mcGfx.th = cmp ? Math.round(Math.max(170, colW * 0.9)) : null; mcGfx.cmpW = cmp ? BW : null; }
  const G = mcGfx, gRow = (label, vis, val, cls = "", tip = "") => `<div class="mc-g"${tip ? ` title="${escapeHtml(tip)}"` : ""}><span>${label}</span>${vis}<b class="${cls}">${val}</b></div>`;
  const note = (s) => `<div class="mc-note">${s}</div>`;
  const qSec = (key, title, inner, extra = "") => sec(key, title, (extra ? `<div class="mc-state">상태${extra}</div>` : "") + inner);
  const levChip = ` <span class="mc-chip${levWarn ? " warn" : ""}">${escapeHtml(((d.lev || {}).label || "-").split(" —")[0])}</span>`;
  const btcTxt = (d.btc || {}).rel === "동행" ? "ETH 같이 간다" : (d.btc || {}).rel === "단독" ? "ETH 혼자 간다" : (d.btc || {}).move_bp == null ? "수집 중" : "ETH 30분 방향 없음";
  const events = [
    f.next_ms ? { t: f.next_ms, nm: "펀딩 정산" } : null,
    ef ? { t: ef.exp_ms, nm: `옵션 만기 · pain ${n(ef.pain)}` } : null,
    us ? { t: us.live ? us.close : us.open, nm: us.live ? "미국장 마감" : "미국장 개장" } : null,
    ...macro24,
  ].filter(Boolean).reduce((acc, e) => {   // 2026-09-30 같은 분(分)의 일정은 이름표 하나로 합친다 -- 겹침 회피가 둘째를 숨겨 ISM PMI(high)가 호버로만 보였다
    const same = acc.find((a) => Math.abs(a.t - e.t) < 60e3);
    if (same) { same.nm = `${same.nm} · ${e.nm}`; same.hi = same.hi === false && e.hi === false ? false : (same.hi || e.hi); } else acc.push({ ...e });
    return acc;
  }, []);
  let htmlB = [
    qSec("q_lev", "① 레버리지 과열?", gRow("펀딩 분위", G.pct(f.bn_at_base ? null : f.bn_pct180), f.bn_at_base ? "기본값" : `${Math.round((f.bn_pct180 ?? 0) * 100)}%`, "", `바이낸스 예상 ${fr(f.bn)}`)
      + gRow("베이시스 분위", G.pct(b.pct7d), `${sg(b.bp, 1, "bp")}`, "", `마크−인덱스 · 7일 ${Math.round((b.pct7d ?? 0) * 100)}분위 · 30분 ${sg(b.d30_bp, 1, "bp")}`)
      + gRow("OI 1시간", G.sig(oi.z1h), `${sg(oi.z1h, 1, "σ")}`, tone(oi.z1h, 1.5), `1시간 ${sg(oi.d1h_pct, 2, "%")} · 24시간 ${sg(oi.d24h_pct, 1, "%")}`)
      + (ls ? gRow("계정 롱숏", G.ratio(ls.global, ls.global_24h), n(ls.global, 2), "", `하루 전 ${n(ls.global_24h, 2)}`)
        + gRow("탑트레이더", G.ratio(ls.top_pos, ls.top_pos_24h), n(ls.top_pos, 2), "", `하루 전 ${n(ls.top_pos_24h, 2)}`) : "")
      + note(`정산까지 ${left(f.next_ms)} · 펀딩 OKX ${fr(f.okx)} / HL ${fr(f.hl_8h)} · OI ${n((oi.oi || 0) / 1e6, 2)}M + OKX ${n((oi.okx || 0) / 1e6, 2)}M + HL ${oi.hl == null ? "-" : n(oi.hl / 1e6, 2) + "M"} ETH`), levChip),
    qSec("q_flow", "② 누가 밀고 있나 · 60분", (z.whale == null ? note("크기별 기준 쌓는 중(6시간)")
        : ["whale", "mid", "retail"].map((k2, i) => gRow(["고래", "중형", "리테일"][i], G.sig(z[k2]), sg(z[k2], 1, "σ"), tone(z[k2]))).join(""))
      + gRow("30분 CVD", G.sig(fl.cvd30_z), sg(fl.cvd30_z, 1, "σ"), tone(fl.cvd30_z))
      + (ls ? gRow("테이커 매수÷매도", G.ratio(ls.taker, null), n(ls.taker, 2), tone(ls.taker == null ? null : ls.taker - 1, 0.1)) : "")
      // 2026-09-30 사분면 그림 → 한 줄 네 칸(사용자 «높이를 쓸데없이 잡아먹는다»): 새 롱 · 숏 커버 · 새 숏 · 롱 정리 중 지금 칸만 채움, 작으면 전부 빈 칸.
      + gRow("가격×OI 1h", G.seg4(d.quad ? d.quad.key : null), d.quad ? ({ up_up: "새 롱", up_dn: "숏 커버", dn_up: "새 숏", dn_dn: "롱 정리", small: "보류" })[d.quad.key] || "-" : "-",
             !d.quad ? "" : d.quad.key === "dn_up" ? "mc-bad" : d.quad.key === "dn_dn" ? "mc-good" : "",
             `${d.quad ? d.quad.label : "-"}\n1시간 ${sg(d.move60, 0, "bp")} · OI ${sg(d.oi60, 0)} ETH${d.quad && d.quad.note ? `\n${d.quad.note}` : ""}`)
      + note(`30분 체결 · 바이낸스 ${sg(fl.bn30, 0)} / OKX ${sg(fl.okx30, 0)} ETH`)),
    // 2026-09-30 사용자 «범례·청산 상태 글은 툴팁 안으로» -- 그림 위 호버에 두 줄. 청산 급증일 때만 판 위에 경고 한 줄을 남긴다(놓치면 안 되는 상태).
  ];
  // 2026-09-30 ④ 는 넓은 화면이면 오른쪽 칸 아랫줄 두 단(①② 아래, ③ 과 자리 바꿈) + 호가 계기 넷(변동·불균형·지속·이탈)을 여기로.
  // 2026-10-01(3) 넓은 화면: 왼쪽 단 = ① 위 + ② 아래 · 오른쪽 단 = ④ 한 덩어리(사용자 지시 «④ 를 반씩 자르지 말고 하나로 · ② 를 ④ 반쪽 자리로») · 프로파일 아래는 비운다.
  const wallW = G.cmpW ? Math.max(200, Math.floor((G.cmpW - 14) / 2)) : null;   // ④ = 한 단 폭
  // 2026-09-30 ③ 가격 지형 = 넓은 화면이면 호가/체결 프로파일 **아래**(#mcWall, 사용자 «높이가 낮아 겹친다 → ④ 와 자리 바꿔») -- 그 칸의 높이를 다 쓴다.
  // 2026-10-01 ③ 가격 지형 제거(사용자 지시 -- VWAP·현재가는 풋프린트와 중복, 나머지는 쓰임이 적었다). ④ 가 그 자리(호가/체결 프로파일 아래)로 돌아간다.
  const hsm = latestFlowHeatmap && latestFlowHeatmap.summary, gwKeep = G.gw;
  if (wallW) G.gw = Math.max(40, wallW - 58 - 44 - 12);   // ④ 단 폭에 맞춘 게이지(아래 wallHtml 을 다 만든 뒤 되돌린다)
  const statRows = G.cmpW && hsm ? STAT_KEYS.map((k) => {
    const f = statFrac(k, hsm), tip = statTipHtml(k, hsm).replace(/<br>/g, "\n").replace(/<[^>]+>/g, "");
    const val = f == null ? "-" : k === "obi" ? `${f > 0 ? "+" : ""}${f.toFixed(2)}` : `${Math.round(f * 100)}%`;
    return gRow(STAT_NAME[k], k === "obi" ? G.sig(f == null ? null : f * 3) : G.pct(f), val, k === "vol" && f >= 0.66 ? "mc-warn" : "", tip);
  }).join("") : "";
  // 2026-10-01 «차트에 그리기» = 넓은 화면이면 풋프린트 청산 밀도 범례 오른쪽 한 줄(#fpLineSwitch, 사용자 지시) -- ④ 는 그만큼 짧아지고 1초 수급이 남는 높이를 가져간다
  const lineSw = `<div class="mc-switch"><span class="mc-switch-t">차트에 그리기</span><label><input type="checkbox" data-line="vwap"${mcLine.vwap ? " checked" : ""}> VWAP ±σ</label>`
      + `<label title="끄면 하루(UTC 00시부터) · 켜면 지금 시장 세션 개장부터(아시아·유럽·미국)"><input type="checkbox" data-line="vsess"${mcLine.vsess ? " checked" : ""}> 세션 시작부터</label>`
      + `<label><input type="checkbox" data-line="bb"${mcLine.bb ? " checked" : ""}> 볼린저</label>`
      + `<label><input type="checkbox" data-line="wvwap"${mcLine.wvwap ? " checked" : ""}> 주간·앵커 VWAP</label></div>`;
  setH("fpLineSwitch", G.cmpW ? lineSw : "");
  const wallHtml0 = qSec("q_wall", "④ 벽 · 교차 · 위치", (sw ? G.book(sw, bps, ...(wallW ? [wallW] : [])) : note("호가 래스터 대기")) + statRows
      + note(`스프레드 ${bk.spread == null ? "-" : "$" + bk.spread.toFixed(2)}${bk.spread > 0.015 ? " — 평소(1틱)보다 넓다" : ""}`
        + (bk.bid25_pct == null ? "" : ` · 얇은 쪽 매수 ${Math.round(bk.bid25_pct * 100)} · 매도 ${Math.round(bk.ask25_pct * 100)}분위${thin ? ` — ${thin}` : ""}`))
      + gRow("BTC 30분", G.sig((d.btc || {}).move_bp == null ? null : d.btc.move_bp / 20), sg((d.btc || {}).move_bp, 0, "bp"), "", btcTxt)
      + gRow("가격차 OKX", G.sig((d.venues || {}).okx_bp == null ? null : d.venues.okx_bp / 5), sg((d.venues || {}).okx_bp, 1, "bp"), "", "마크 대 마크")
      + gRow("가격차 HL", G.sig((d.venues || {}).hl_bp == null ? null : d.venues.hl_bp / 5), sg((d.venues || {}).hl_bp, 1, "bp"), "", `미드 대 미드 · HL 프리미엄 ${sg(b.hl_premium_bp, 1, "bp")}`)
      + gRow("VWAP 거리", G.sig(vz), sg(vz, 1, "σ"), "", lv ? `${lv.name} VWAP ${n(lv.vwap, 1)}` : "")
      + gRow("볼린저 %B", G.pct(pb), pb == null ? "-" : n(pb, 2), "", bb ? `폭 ${n((bb[2] - bb[0]) / bb[1] * 100, 2)}%` : "")
      + gRow("RSI 14", G.pct(rsi == null ? null : rsi / 100), rsi == null ? "-" : n(rsi, 0), rsi != null && (rsi >= 70 || rsi <= 30) ? "mc-warn" : "")
      + (G.cmpW ? "" : lineSw));
  G.gw = gwKeep;
  if (G.cmpW) htmlB = [`<div class="mc-stack">${htmlB.join("")}</div>`];   // ①② 를 한 단에 위아래로
  htmlB.push(wallHtml0);
  htmlB = htmlB.join("");
  // 2026-09-30 ⑤ 다음 24시간은 좁은 칸(mc-cmp)이면 풋프린트 차트 **아래 전폭**(#mcWhen, 사용자 지시) -- 아니면 판 넷 아래 전폭 그대로.
  const whenBox = el("mcWhen");   // 2026-10-01 휴대폰·세로 화면도 Option 카드 맨 아래(넓은 화면과 같게)
  // 2026-09-30 폭: #mcWhen 은 비어 있으면 숨김(display:none)이라 첫 그림 때 clientWidth 0 → 900 으로 그려 2.3배 늘어났다(글자가 커졌다 작아짐) -- 카드 폭에서 잰다
  const whenW = whenBox ? (whenBox.clientWidth || ((el("optCard") || body).clientWidth - 48)) : body.clientWidth;   // 2026-10-01 ⑤ 는 Option 카드 맨 아래
  const whenHtml = qSec("when", "⑤ 다음 24시간", G.timeline(events, Math.max(320, Math.round((whenW || 900) - 8)))
      + (ef && ef.exp_ms - Date.now() > 0 && ef.exp_ms - Date.now() < 6 * 3600e3 ? note(`max pain 규칙 창 = 만기 1시간 전부터(${left(ef.exp_ms - 3600e3)} 뒤)`) : "")
      + note(macroCalendarOkAt ? `경제 일정 갱신 ${fmtHourMinute(macroCalendarOkAt)} · 6시간마다(실패하면 1분 뒤 다시)` : "경제 일정 불러오는 중…"));
  setH("mcWhen", whenBox ? whenHtml : "");
  if (!whenBox) htmlB += `<div class="mc-wide">${whenHtml}</div>`;
  if (body._mcHtml === htmlB) return;     // 같은 내용이면 다시 안 그린다(펼친 설명·포커스 유지)
  body._mcHtml = htmlB;
  keepFocus(body, () => { body.innerHTML = htmlB; });
  if (body.classList.contains("mc-cmp")) {   // 내용 높이(칸 높이가 아니라 자식들의 아래 끝) -- 바뀌면 차트를 다시 그려 1초 수급 몫을 조정
    const top = body.getBoundingClientRect().top, need = Math.ceil(Math.max(0, ...[...body.children].map((k) => k.getBoundingClientRect().bottom - top)) + 6);
    if (need > 40 && Math.abs(need - mcNeedH) > 4) {
      mcNeedH = need; if (typeof scheduleSnapshotChartRender === "function") scheduleSnapshotChartRender();
    }
  }
}

["mcBody", "mcWhen"].forEach((id) => el(id)?.addEventListener("click", (e) => {   // ⑤ 는 #mcWhen 에 있을 수 있다
  const b = e.target.closest(".mc-q");
  if (!b) return;
  const k = b.dataset.tip, open = !mcTipOpen.has(k);
  if (open) mcTipOpen.add(k); else mcTipOpen.delete(k);
  b.setAttribute("aria-expanded", String(open));
  const tip = b.closest(".mc-sec")?.querySelector(".mc-tip");
  if (tip) tip.hidden = !open;
  el("mcBody")._mcHtml = null;         // 다음 그리기에서 펼침 상태를 반영한다
}));
["mcBody", "fpLineSwitch"].forEach((id) => el(id)?.addEventListener("change", (e) => {   // 체크박스는 넓은 화면이면 #fpLineSwitch
  const k = e.target?.dataset?.line;
  if (!k) return;
  mcLine[k] = e.target.checked;
  try { localStorage.setItem("mcLine", JSON.stringify(mcLine)); } catch (err) { /* 기억은 편의 */ }
  el("mcBody")._mcHtml = null;
  scheduleSnapshotChartRender();
}));


// 2026-09-26 비평: 수 초마다 innerHTML 을 통째로 갈아 끼우는 카드에서 Tab 으로 훑던 포커스가 <body> 로 떨어졌다.
//   갈아 끼우기 전 «몇 번째 포커스 가능한 요소»였는지 기억했다가 같은 자리로 돌려놓는다.
const FOCUSABLE = "summary, button, a[href], [tabindex]:not([tabindex='-1'])";
function keepFocus(box, render) {
  const a = document.activeElement;
  const idx = a && a !== document.body && box.contains(a) ? [...box.querySelectorAll(FOCUSABLE)].indexOf(a) : -1;
  render();
  if (idx >= 0) box.querySelectorAll(FOCUSABLE)[idx]?.focus({ preventScroll: true });
}

// 차트에는 **보이는 캔들 범위 안**의 뒤집힘 가격만 그린다 -- 레벨은 세로 축을 넓히지 않고 화면 밖이면 가장자리에
//   쌓이므로, 20% 넘게 떨어진 선 넷이 바닥에 겹치면 읽을 수 없다. 5개 전체는 카드에 거리와 함께 있다.
function trendFlipLevels(footprint, candles) {
  const t = latestTrend;
  if (!t || !t.ok || activeSnapshotAsset !== "eth" || !candles.length) return [];
  const lo = Math.min(...candles.map((c) => c.low)), hi = Math.max(...candles.map((c) => c.high));
  return t.votes.filter((v) => Number(v.L) !== 7 && v.flip_price >= lo && v.flip_price <= hi).map((v) => ({   // 2026-10-01 7일선 제거(사용자 지시) -- 14~90일은 범위 안에 오면 그대로
    val: Number(v.flip_price), color: "var(--muted)", label: `추세${v.L}일`, priceLeft: true,
    dashed: true, width: 1, marker: !!footprint }));
}

// 2026-09-30 사용자 «30분 시나리오를 풋프린트로»(시안 A — 레짐·권장 크기 칩과 검증 꼬리표는 뺌): 30분 칸을 없애고
//   ① 차트 머리 칩 둘(큰 방향 5표 · 융합 신호) ② 가격판 «30분 도달 선»(renderCandleSvg)으로 옮겼다.
//   5표 표(기간·기준가·지금 대비·뒤집힘)는 추세 칩 툴팁, 역추세 보유면 칩이 주황. 서버 계산·SSE(latestSituation)는 그대로.
//   큰 방향 = UTC 00시 일봉 종가를 7·14·28·56·90일 전과 비교한 5표(dashboard/trend_rule.py) · 융합 = 독립 4표 ≥2 + 크기 관문.
function renderSituation() {
  const box = el("sitChips");
  if (!box) return;
  const eth = activeSnapshotAsset === "eth", s = latestSituation || {}, fz = (s.read && s.read.fused) || null;
  const t = latestTrend && latestTrend.ok ? latestTrend : null;
  const pxf = (v) => Number(v).toLocaleString("en-US", { maximumFractionDigits: v >= 100 ? 0 : 2 });
  let h = "";
  if (eth && t) {
    const up = t.signal > 0, live = Number(latestLivePriceByAsset.eth || 0);
    const rows = t.votes.map((v) => {
      const d = live > 0 ? (live / v.flip_price - 1) * 100 : null, flip = live > 0 && (live > v.flip_price) !== v.up;
      return `${v.L}일 · 기준가 ${pxf(v.flip_price)} · 지금 ${d == null ? "-" : `${d >= 0 ? "+" : ""}${d.toFixed(1)}%`} · ${v.up ? "상승" : "하락"}${flip ? " (지금 가격으로 마감하면 뒤집힘)" : ""}`;
    }).join("\n");
    const against = ((latestBinanceAccount || {}).positions || [])
      .filter((p) => String(p.symbol || "").startsWith("ETH") && Math.abs(Number(p.qty) || 0) > 0 && (p.side === "LONG") !== up);
    const tip = `큰 방향(1~13주) — UTC 00시 일봉 종가를 7·14·28·56·90일 전 종가와 비교한 5표 · ${t.age_days}일째\n${rows}`
      + (against.length ? `\n역추세 ${against.map((p) => (p.side === "LONG" ? "롱" : "숏")).join("·")} 보유 — 원장 79왕복: 추세 쪽 +898$ vs 역추세 −468$(최악 −549$)` : "");
    h += `<span class="sit-chip ${up ? "up" : "dn"}${against.length ? " warn" : ""}" title="${escapeHtml(tip)}">${up ? "▲ 추세 롱" : "▼ 추세 숏"}`
      + ` <span class="sit-dots">${t.votes.map((v) => `<b class="${v.up ? "u" : ""}"></b>`).join("")}</span> ${t.ups}/${t.votes.length} · ${t.age_days}일`
      + `${against.length ? " · 역추세 보유" : ""}</span>`;
  }
  if (eth && fz) {
    const d = fz.side > 0 ? "up" : fz.side < 0 ? "dn" : "";
    const tail = String(fz.text || "").replace(/^[^—]*—\s*/, "");   // 서버 문장의 «대기 — » 머리를 뗀다
    h += `<span class="sit-chip ${d || "q"}" title="${escapeHtml(`융합 신호(다음 30분) — ${tail}`)}">${d ? `융합 ${d === "up" ? "롱" : "숏"} 발동` : "융합 대기"}</span>`;
  }
  if (box._h !== h) { box._h = h; box.innerHTML = h; }
  box.hidden = !h;
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
  if (!flowOn()) return;   // 흐름 엔진이 있는 코인만
  const now = Date.now();
  if (now - footprintLastFetchAt < FOOTPRINT_POLL_MS) return;
  footprintLastFetchAt = now;
  const key = `${activeSnapshotAsset}|${CHART_PAN_BARS}`;   // 창과 무관하게 12시간 -- 창 이동은 받아 둔 것에서 고른다
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
    const res = await fetch(`${API_FOOTPRINT_URL}?asset=${activeSnapshotAsset}&bars=${CHART_PAN_BARS}&since=${since}`,
                            { cache: "no-cache" });
    if (!res.ok) throw new Error(`footprint ${res.status}`);
    const payload = await res.json();
    if (`${activeSnapshotAsset}|${CHART_PAN_BARS}` !== key) return;   // 그 사이 코인·창이 바뀌었다
    if (payload.full) footprintBars = new Map();
    (payload.bars || []).forEach((b) => footprintBars.set(b.time, b));
    // 창 밖으로 밀려난 봉은 버린다 -- 증분이라 서버가 «빠졌다»를 말해 주지 않는다.
    if (footprintBars.size > CHART_PAN_BARS) {
      [...footprintBars.keys()].sort((a, b) => a - b)
        .slice(0, footprintBars.size - CHART_PAN_BARS)
        .forEach((t) => footprintBars.delete(t));
    }
    // 아래 소비자(footprintForChart)는 예전과 **같은 모양**을 본다 -- 시각순 전체 배열.
    latestFootprint = { ...payload,
                        bars: [...footprintBars.values()].sort((a, b) => a.time - b.time) };
  } catch (error) {
    console.error("Footprint fetch error:", error);
    // 🔴2026-09-22 전에는 여기서 latestFootprint = null 이었다 -- **한 번의 실패로 화면이
    //   비었다**(400ms 폴링이라 그게 곧 깜빡임이다). 캐시는 이미 안 버리고 있었으므로
    //   직전 값을 그대로 둔다. 진짜로 정체되면 위 «전량 재수신» 자가복구가 잡는다.
    // 🔴캐시는 **안 버린다**. 한 번의 네트워크 실패로 12.5KB 를 다시 받을 이유가 없다 --
    //   다음 성공 폴링이 꼬리 두 봉만 얹으면 그대로 이어진다.
  }
  scheduleSnapshotChartRender();
}

// 풋프린트가 없으면(다른 코인 · 서버 웜업 · fetch 실패) null 을 돌려주고, 차트는 캔들로 그린다.
function footprintForChart() {
  if (!flowOn()) return null;
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
  if (v >= 1e6) return (v / 1e6).toFixed(v >= 1e7 ? 0 : 1) + "M";      // XRP 수량(ETH 의 ~2,000배)
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
// ⭐2026-09-28 사용자 «고래/중형/리테일이 매초 어떻게 거래하는지 5분 내내» -- 누적 한 판을 **다섯 줄 거울 막대**로 바꿨다
//   (CVD · 고래 · 중형 · 리테일 · 신규 OI). 위 «초별 원값은 꼬리가 무겁다(260배)»는 **√ 눈금 + 줄마다 제 최대**로 접는다 --
//   대신 «같은 높이 = 같은 수량»은 줄 안에서만 성립한다(줄끼리 크기는 왼쪽 숫자). 누적은 줄마다 흰 선으로 남는다.
function renderSupply1s(box = null, src = null) {
  // src 가 오면 그 출처로 그린다(OKX 레인). 없으면 바이낸스 전역이다. 칸 구조가 같으므로
  // **수식은 한 줄도 안 바뀐다** -- 바뀌는 건 어느 Map 을 읽느냐뿐이다.
  const S = src || mergedSupplySrc();
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
  // 2026-09-22 모바일(사용자 «프로파일처럼 라벨을 차트 아래에 일자로»): 범례가 플롯 안에서
  //   바닥 한 줄로 내려온다. 시각 꼬리표(h-3)와 같은 줄에 둘 수 없으니 그 아래 한 줄을 더 판다.
  //   400px 패널에서 14px 은 선이 3.5% 짧아지는 대가다.
  // 🔴narrow 를 **먼저** 선언한다. 아래 mb 가 이걸 읽는데 순서를 바꾸면 TDZ
  //   ReferenceError 로 app.js 전체가 죽는다 -- node --check 는 못 잡는다(문법은 옳다).
  const narrow = w < 760;   // 모바일이든 좁은 상자든 여백 규칙은 같다
  // 2026-09-28 모든 폭에서 범례는 플롯 **안 박스 셋**(사용자 «거래소 3 · 크기별 3 · 신규 2»)이고 위 제목 줄도 없앴다 --
  //   mt 는 위 여백만, mb 는 바닥 시각 꼬리표(h-3) 한 줄이다. (09-27 모바일 박스가 모든 폭으로 넓어진 것)
  const mt = 6, mb = 16;
  // mr 은 오른쪽 꼬리표(고래/리테일/신규계약 + 값)가 앉는 자리다. 64 면 «신규계약 +0.4k»
  // 가 약 4px 넘친다(2026-09-19 계산) -- 72 로 두면 셋 다 들어가고 선은 8px 만 짧아진다.
  // 2026-09-22 꼬리표 글자를 키우면서 자리도 넓혔다(사용자 «많이 키워줘»).
  //   112 -> 150 은 선이 38px(플롯 폭의 3%) 짧아지는 대가다. 글자가 잘리는 것보다 낫다.
  // 2026-09-22 모바일(사용자 «라벨들을 차트 안으로»): mr 은 «여백»이 아니라 **그림이 끝나는
  //   자리**다. 88 을 비워 두면 348px 화면에서 선이 260px 로 눌린다(실측). 8 로 줄이면 같은
  //   화면에서 플롯이 344px = **+32%** 가 되고, 범례는 아래에서 플롯 위로 얹는다.
  //   🔴데스크톱은 그대로 둔다 -- 폭이 남는 화면에서 글자를 데이터 위에 올릴 이유가 없다.
  // 2026-09-28 다섯 줄 거울 막대 -- 왼쪽 칸에 줄 이름·누적값(범례 박스를 대체), 오른쪽은 얇은 여백만.
  const ml = 100, mr = 8;   // 2026-09-30 두 줄 선 차트: 왼쪽 칸 = 선 견본 + 값(«선물 +2.4k»), 줄 이름은 줄 위 머리 띠
  const cw = w - ml - mr;
  const flowTop = mt, flowH = h - mb - flowTop;
  // 🔴이 줄이 없어서 HTML 의 고정 viewBox(1200) 가 그대로 남아 있었다. 폭을 부모에서 받도록
  //   바꾸는 순간 그림이 viewBox 밖으로 나간다 -- 좌표계와 뷰박스는 같이 움직여야 한다.
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.innerHTML = "";

  const now = S.now || 0;
  // 🔴창이 «최근 5분»(미끄러짐)이 아니라 **지금 만들어지고 있는 5분봉 그 자체**다
  //   (2026-09-20 사용자 선택: 시안 H). x축 왼쪽 끝 = 봉이 열린 시각, 오른쪽 끝 = 봉이
  //   닫힐 시각. 선은 봉이 진행되는 만큼 왼쪽에서 오른쪽으로 자라고, 다음 봉에서 리셋된다.
  //   ⭐이 패널이 말하는 수급 = **바로 아래 풋프린트 봉을 만들고 있는 그 체결들**이다
  //     (server.py footprint_bar_start 와 같은 식으로 자른 같은 경계).
  //   ⚠️캔들 «차트»와 x축이 겹치는 건 아니다 -- 그쪽은 12~48봉(1~4시간)을 같은 폭에 그린다.
  //     겹치는 것은 **데이터 구간**이지 가로 좌표가 아니다.
  const first = Math.floor(now / SUPPLY_1S_SEGMENT) * SUPPLY_1S_SEGMENT;
  // 이제 한 구간만 그리므로 이전 구간을 읽을 이유가 없다(누산기가 이 봉의 경계에서 시작한다).
  const allSecs = [...S.supply.keys()].filter((s) => s >= first && s <= now)
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
  const label = (x, y, text, color, anchor, size = 10, parent = svg) => {
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", x); t.setAttribute("y", y); t.setAttribute("font-size", size);
    t.setAttribute("fill", color); if (anchor) t.setAttribute("text-anchor", anchor);
    t.textContent = text;
    parent.appendChild(t);
    return t;
  };

  // 구간 경계에서 0으로 되돌리며 쌓는다. 경계 이전 초도 **계산에는** 들어간다(누산기를
  // 그때 0으로 되돌리는 게 전부이고, 그리는 건 first 이후뿐이다).
  // 🔴2026-09-30 `x > first` 는 봉의 **첫 초**(키 first = [first, first+1) 체결)를 빼서 선물 CVD·고래/중형/리테일이
  //   풋프린트 봉 델타와 달랐다(실측 봉 −569 vs 그린 −744, 첫 초 +177). 체결 흐름은 첫 초부터 센다.
  //   OI 는 따로(oiRows) -- 경계 이전 마지막 관측이 기준점이라 `> first` 그대로다.
  const inSeg = allSecs;
  const zero6 = [0, 0, 0, 0, 0, 0];
  const cellOf = (x) => S.supply.get(x) || zero6;
  // ── 두 줄 선 차트 (2026-09-30 사용자 «CVD 와 OI 를 하나로, 고래/중형/리테일을 하나로, 막대는 없애고 선만, 폭 1/3 줄여») ──
  //   줄 1 = CVD·OI: 선물 CVD 누적(흰) · 현물 CVD 누적(보라, ETH 전역일 때) · OI 누적 증분(주황) + 청산 점(OI 선 위). 셋은 단위·크기가
  //     달라 **선마다 제 눈금**(0선만 공유) -- 모양·방향만 비교한다. 값은 왼쪽 칸 숫자.
  //   줄 2 = 고래·중형·리테일 누적 순수급 -- 같은 단위(코인 수량)라 **눈금 하나를 공유**해 누가 봉을 끄는지 크기까지 비교한다.
  //     선 모양으로 구분(고래 굵은 실선 · 중형 파선 · 리테일 점선), 값 글자 색 = 부호(초록 +/빨강 −). 중형 = 전체 − 고래 − 리테일.
  //   🔴초당 거울 막대(09-28 시안 ①)는 뺐다 -- «리듬»은 풋프린트 셀·체결 기둥이 이미 보인다.
  const TIERS = [
    ["고래", (c) => c[2] - c[3], 2.4, 0.95, null],
    ["중형", (c) => Math.max(0, c[4] - c[2] - c[0]) - Math.max(0, c[5] - c[3] - c[1]), 1.7, 0.75, "6 3"],
    ["리테일", (c) => c[0] - c[1], 1.5, 0.6, "2 3"],
  ];
  // OI: 경계 직전 마지막 관측을 0 으로 두고 누적 증분을 잰다(3~7초 갱신).
  const oiKeys = [...S.oi.keys()].filter((x) => x <= now).sort((a, b) => a - b);
  const oiBase = (() => { let v = null; oiKeys.forEach((x) => { if (x <= first || v === null) v = S.oi.get(x); }); return v; })();
  const oiRows = [];
  oiKeys.forEach((x) => { if (x > first) oiRows.push({ s: x, v: S.oi.get(x) - oiBase }); });
  const NL = 2, laneH = flowH / NL;
  const sgn = (v) => (v >= 0 ? "+" : "-") + qty(v);
  // 체결 없는 초(5초 넘게)는 «0» 이 아니라 «모름» -- 판 전체에 흐린 띠.
  let prevSec = null;
  secs.forEach((x) => {
    if (prevSec !== null && x - prevSec > SUPPLY_1S_GAP_SEC) {
      const g = document.createElementNS(NS, "rect");
      g.setAttribute("x", xAt(prevSec)); g.setAttribute("y", flowTop);
      g.setAttribute("width", Math.max(1, xAt(x) - xAt(prevSec))); g.setAttribute("height", flowH);
      g.setAttribute("fill", "var(--neutral)"); g.setAttribute("fill-opacity", "0.07");
      const t = document.createElementNS(NS, "title");
      t.textContent = "체결 기록 없음 " + (x - prevSec) + "초 -- 0이 아니라 «모름»이다";
      g.appendChild(t); svg.appendChild(g);
    }
    prevSec = x;
  });
  // 줄 틀: 0선 · 구분선 · 왼쪽 칸에 이름 + 값 목록([글자, 색, 선 모양 견본]).
  const HEAD = 18;   // 줄 이름 머리 띠 -- 이름(«고래 · 중형 · 리테일»)이 왼쪽 칸보다 길어 선과 겹쳤다(09-30 1920 실측)
  const laneFrame = (k, name, items) => {
    const y0 = flowTop + k * laneH, zc = y0 + HEAD + (laneH - HEAD) / 2;
    const zl = document.createElementNS(NS, "line");
    zl.setAttribute("x1", ml); zl.setAttribute("x2", ml + cw); zl.setAttribute("y1", zc); zl.setAttribute("y2", zc);
    zl.setAttribute("stroke", "var(--ink)"); zl.setAttribute("stroke-opacity", "0.18");
    svg.appendChild(zl);
    if (k) {
      const sep = document.createElementNS(NS, "line");
      sep.setAttribute("x1", 4); sep.setAttribute("x2", w - 4); sep.setAttribute("y1", y0); sep.setAttribute("y2", y0);
      sep.setAttribute("stroke", "var(--line)");
      svg.appendChild(sep);
    }
    const fs = narrow ? 11 : 12, gap = fs + 9;
    label(6, y0 + 14, name, "var(--text)", null, narrow ? 12 : 13).setAttribute("font-weight", "700");
    let y = zc - (items.length * gap) / 2 + fs - gap;
    items.forEach(([txt, col, sw]) => {
      y += gap;
      if (sw) {   // 선 모양 견본(12px) -- 오른쪽 그림의 어느 선인지
        const sm = document.createElementNS(NS, "line");
        sm.setAttribute("x1", 6); sm.setAttribute("x2", 18); sm.setAttribute("y1", y - fs / 3); sm.setAttribute("y2", y - fs / 3);
        sm.setAttribute("stroke", sw.color); sm.setAttribute("stroke-width", Math.min(2.4, sw.width));
        sm.setAttribute("stroke-opacity", sw.op); if (sw.dash) sm.setAttribute("stroke-dasharray", sw.dash);
        svg.appendChild(sm);
      }
      label(sw ? 22 : 6, y, txt, col, null, fs).setAttribute("font-weight", "700");
    });
    return { y0, zc, hh: (laneH - HEAD) / 2 - 4 };
  };
  const scaled = (rows, zc, hh, cs) => (r) => zc - (r.v / cs) * hh * 0.9;
  // ── 줄 1: CVD · OI ──
  let acc = 0;
  const cvdRows = inSeg.map((x) => { const c = cellOf(x); acc += c[4] - c[5]; return { s: x, v: acc }; });
  let spotAcc = 0;
  const spotRows = src ? null : [...spotSupply1s.keys()].filter((x) => x >= first && x <= now).sort((a, b) => a - b)
    .map((x) => { const c = spotSupply1s.get(x); spotAcc += (c[4] || 0) - (c[5] || 0); return { s: x, v: spotAcc }; });
  const spotOn = !!(spotRows && spotRows.length >= 2);
  const oiEnd = oiRows.length ? oiRows[oiRows.length - 1].v : null;
  {
    const items = [["선물 " + sgn(acc), "var(--ink)", { color: "var(--ink)", width: 1.8, op: 0.9 }]];
    if (spotOn) items.push(["현물 " + sgn(spotAcc), "var(--spot)", { color: "var(--spot)", width: 1.6, op: 0.95 }]);
    items.push(["OI " + (oiEnd == null ? "—" : sgn(oiEnd)), "var(--warn)", { color: "var(--warn)", width: 1.8, op: 0.95 }]);
    const { zc, hh } = laneFrame(0, "CVD · OI", items);
    const csOf = (rows) => Math.max(1e-9, ...rows.map((r) => Math.abs(r.v)));
    if (cvdRows.length >= 2) line(pathOf(cvdRows, scaled(cvdRows, zc, hh, csOf(cvdRows))), "var(--ink)", 1.8, 0.9);
    if (spotOn) line(pathOf(spotRows, scaled(spotRows, zc, hh, csOf(spotRows)), 20), "var(--spot)", 1.6, 0.95);
    if (oiRows.length >= 2) {
      const yOi = (v) => zc - (v / csOf(oiRows)) * hh * 0.9;
      line(pathOf(oiRows, (r) => yOi(r.v), 20), "var(--warn)", 1.8, 0.95);
      // 청산: 그 초의 OI 누적선 위 점 -- 청산은 포지션을 강제로 닫아 OI 를 줄이는 사건이다. 크기는 **로그**(건당 0.86~2,556 ETH).
      const oiAt = (sec) => {
        if (sec <= oiRows[0].s) return oiRows[0].v;
        for (let i = 1; i < oiRows.length; i++) {
          if (oiRows[i].s < sec) continue;
          const a = oiRows[i - 1], b = oiRows[i];
          if (b.s - a.s > 20) return b.s - sec <= sec - a.s ? b.v : a.v;
          return a.v + (b.v - a.v) * (sec - a.s) / (b.s - a.s);
        }
        return oiRows[oiRows.length - 1].v;
      };
      inSeg.forEach((x) => {
        const lc = S.liq.get(x);
        if (!lc || !(lc[0] > 0 || lc[1] > 0)) return;
        const v = lc[1] - lc[0];
        const c = document.createElementNS(NS, "circle");
        c.setAttribute("cx", xAt(x).toFixed(1));
        c.setAttribute("cy", yOi(oiAt(x)).toFixed(1));
        c.setAttribute("r", (2 + Math.log10(1 + Math.abs(v)) / Math.log10(3001) * 7).toFixed(1));
        c.setAttribute("fill", v >= 0 ? "var(--good)" : "var(--bad)");
        c.setAttribute("fill-opacity", "0.92");
        const t = document.createElementNS(NS, "title");
        t.textContent = "청산 롱 " + fmtUsdCompact(lc[2]) + " / 숏 " + fmtUsdCompact(lc[3])
          + "  (" + qty(lc[0]) + " / " + qty(lc[1]) + " " + coinUnit() + ")";
        c.appendChild(t); svg.appendChild(c);
      });
    }
  }
  // ── 줄 2: 고래 · 중형 · 리테일 (눈금 공유) ──
  {
    const series = TIERS.map(([name, net, width, op, dash]) => {
      let a = 0;
      return { name, width, op, dash, rows: inSeg.map((x) => { a += net(cellOf(x)); return { s: x, v: a }; }), end: 0 };
    });
    series.forEach((sr) => { sr.end = sr.rows.length ? sr.rows[sr.rows.length - 1].v : 0; });
    const items = series.map((sr) => [sr.name + " " + sgn(sr.end), sr.end >= 0 ? "var(--good)" : "var(--bad)",
                                      { color: "var(--ink)", width: sr.width, op: sr.op, dash: sr.dash }]);
    const { zc, hh } = laneFrame(1, "고래 · 중형 · 리테일", items);
    const cs = Math.max(1e-9, ...series.flatMap((sr) => sr.rows.map((r) => Math.abs(r.v))));
    series.forEach((sr) => { if (sr.rows.length >= 2) line(pathOf(sr.rows, scaled(sr.rows, zc, hh, cs)), "var(--ink)", sr.width, sr.op, sr.dash); });
  }

  // «지금» 세로선과 아직 안 온 시간 음영 -- 빈 오른쪽이 «없음»이 아니라 «아직»으로 읽히게.
  {
    const nx = xAt(now);
    const g = document.createElementNS(NS, "line");
    g.setAttribute("x1", nx); g.setAttribute("x2", nx); g.setAttribute("y1", flowTop); g.setAttribute("y2", flowTop + flowH);
    g.setAttribute("stroke", "var(--ink)"); g.setAttribute("stroke-opacity", "0.35"); g.setAttribute("stroke-dasharray", "2 3");
    svg.appendChild(g);
    const rest = document.createElementNS(NS, "rect");
    rest.setAttribute("x", nx); rest.setAttribute("y", flowTop);
    rest.setAttribute("width", Math.max(0, ml + cw - nx)); rest.setAttribute("height", flowH);
    rest.setAttribute("fill", "var(--lift-solid, #8b949e)"); rest.setAttribute("fill-opacity", "0.05");
    svg.appendChild(rest);
  }

  // 위 제목 줄은 없다(2026-09-28). 🔴스트림이 죽었을 때만 빨간 경고 -- 나이가 없으면 «죽었다»와 «조용하다»가 구별 안 된다.
  if (S.stale) label(ml + 4, flowTop + 12, S.label + (S.age ? "  ·  " + S.age : ""), "var(--bad, #e05260)");

  const hhmm = (t) => { const dd = new Date(t * 1000);
    return String(dd.getHours()).padStart(2, "0") + ":" + String(dd.getMinutes()).padStart(2, "0"); };
  const axisY = h - 3;
  label(ml, axisY, hhmm(first) + " 봉 시작", "var(--muted)");
  label(ml + cw, axisY, hhmm(first + SUPPLY_1S_SEGMENT), "var(--muted)", "end");
  if (!narrow && now - first < SUPPLY_1S_SEGMENT * 0.82) {
    label(xAt(now) + 4, axisY, "지금 (" + (now - first) + "초 경과)", "var(--muted)");
  }
}

// 2026-09-28 호가 요약 네 막대 계기(사용자 선택 시안 A) -- 막대마다 «이 값이 정확히 무엇인가»를 툴팁으로(사용자 지시).
//   원천은 서버 /api/flow/heatmap 의 summary(호가 래스터 · 창 = 차트 창). 🔴넷 다 **서술**이다 -- 방향 예측력은 없다.
const STAT_KEYS = ["vol", "obi", "persist", "leave"];
const STAT_NAME = { vol: "변동", obi: "불균형", persist: "지속", leave: "이탈" };
function statFrac(k, sm) {
  const v = k === "vol" ? (sm.vol_pct == null ? null : sm.vol_pct / 100) : k === "obi" ? sm.obi
    : k === "persist" ? sm.persist_share : sm.offtouch_leave_share;
  return v == null || !Number.isFinite(Number(v)) ? null : Math.max(-1, Math.min(1, Number(v)));
}
function statTipHtml(k, sm) {
  const pct = (v) => (v == null ? "—" : Math.round(100 * v) + "%");
  const win = sm.window_s ? Math.round(sm.window_s / 60) + "분" : "차트";
  const eth = activeSnapshotAsset === "eth" ? "" : "<br><span style=\"opacity:.7\">🔴아래 실측 수치는 ETH 에서 잰 것 — " + coinUnit() + " 에선 검정 안 함.</span>";
  const B = (s) => `<span style="font-weight:700">${s}</span>`;
  const body = {
    vol: `${B("변동 " + (sm.vol_pct == null ? "—" : sm.vol_pct + "%"))} — 지금이 최근 4시간 중 몇 분위로 시끄러운가(100% = 가장 시끄러움).`
      + `<br>재료 둘: 직전 300초 실제 가격 변동 ${sm.vol_past_pct ?? "—"}% 분위 + 호가 재깔림 ${sm.vol_refill_pct ?? "—"}% 분위.`
      + `<br>읽는 법: 높을수록 ${B("앞으로 5분이 크게 흔들렸다")} — 상위 20% 는 하위 33% 의 1.77배(|수익률| 중앙 11.5 vs 6.0bp).`
      + `<br>쓰임: 손절 폭·크기·대기 여부. <span style="opacity:.7">🔴방향은 말하지 않는다 · 7일 실측, 홀드아웃 없음.</span>`,
    obi: `${B("불균형 " + (sm.obi == null ? "—" : (sm.obi > 0 ? "+" : "") + Number(sm.obi).toFixed(2)))} — 현재가 ±${sm.obi_band_pct ?? 0.5}% 안에 걸린`
      + ` (매수 호가 − 매도 호가) ÷ (둘의 합). −1 ~ +1.`
      + `<br>읽는 법: + = 아래(매수) 호가가 두껍다 · − = 위(매도) 호가가 두껍다. 막대는 가운데 0 에서 초록/빨강으로.`
      + `<br><span style="opacity:.7">🔴방향 예측력 없음(실측 상관 +0.03, 신뢰구간 0 포함) · 밴드 폭이 값을 크게 바꾼다(±0.1% 와 ±2% 가 7배).</span>`,
    persist: `${B("지속 " + pct(sm.persist_share))} — 지금 걸린 호가 중 ${win} 창 내내 ${B("한 번도 안 빠진")} 양의 비율.`
      + `<br>읽는 법: 높다 = 호가가 오래 버티는 장(같은 주문이 계속 걸려 있다) · 낮다 = 호가를 자주 걸었다 뺐다 하는 장.`
      + `<br><span style="opacity:.7">🔴크기 쪽으론 약하게 반대(지속 높을수록 덜 흔들림, 상관 −0.11) · 방향 예측력 없음.</span>`,
    leave: `${B("이탈 " + pct(sm.offtouch_leave_share))} — ${win} 창에서 사라진 호가 중 ${B("체결 없이")} 빠진 비율`
      + ` (${sm.fill_source ? "풋프린트 체결과 대조" : "대조 불가"} · 판정 ${sm.offtouch_bins ?? "—"}칸).`
      + `<br>읽는 법: 높다 = 가격이 닿기 전에 호가가 빠지는 장(재호가·허수가 많다) · 낮다 = 사라진 호가가 대부분 실제로 먹혔다.`
      + `<br><span style="opacity:.7">🔴«취소율»이 아니다 — 가격이 닿은 구간은 빠져 있고, 취소와 가격 이동을 구분 못 한다(선물 WS 에 주문 ID 가 없다).</span>`,
  }[k];
  return `<div style="white-space:normal;max-width:min(400px,calc(100vw - 24px))">${body}${eth}</div>`;
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

// 창 오른쪽 끝(배타) 인덱스. 실시간이면 끝 · 과거면 chartPanEnd 시각의 봉 다음 -- 12시간 밖으로는 못 간다.
function chartPanEndIndex(full) {
  const len = full.length, minEnd = Math.min(len, Math.max(chartWindowBars, len - CHART_PAN_BARS + chartWindowBars));
  if (chartPanEnd == null) return len;
  const i = full.findIndex((c) => c.time > chartPanEnd);
  return Math.max(minEnd, i < 0 ? len : i);
}
function chartPanTo(endIdx) {
  const full = candleHistoryByAsset[activeSnapshotAsset] || [];
  if (!full.length) return;
  const len = full.length, minEnd = Math.min(len, Math.max(chartWindowBars, len - CHART_PAN_BARS + chartWindowBars));
  const e = Math.max(minEnd, Math.min(len, Math.round(endIdx)));
  const next = e >= len ? null : full[e - 1].time;
  if (next === chartPanEnd) return;
  chartPanEnd = next;
  chartPanForce = true;
  scheduleSnapshotChartRender();
}
const chartPanBy = (n) => chartPanTo(chartPanEndIndex(candleHistoryByAsset[activeSnapshotAsset] || []) - n);   // n>0 = 과거로
// 헤더 미니맵: 12시간 종가 선 + 지금 창(밝은 띠). 누르거나 끌면 그 자리로 · 방향키 1봉(Shift 6봉).
let chartPanKey = "";
function renderChartPan() {
  const box = el("chartPan");
  if (!box) return;
  const full = candleHistoryByAsset[activeSnapshotAsset] || [];
  if (!full.length || !flowOn()) { box.hidden = true; chartPanKey = ""; return; }
  box.hidden = false;
  const hist = full.slice(-CHART_PAN_BARS), n = hist.length, off = full.length - n;
  const e = chartPanEndIndex(full) - off, s = Math.max(0, e - chartWindowBars), live = chartPanEnd == null;
  const key = `${n}|${s}|${e}|${live}|${hist[n - 1].close}|${hist[0].time}`;
  if (key === chartPanKey) return;
  chartPanKey = key;
  const W = 176, H = 26, lo = Math.min(...hist.map((c) => c.low)), hi = Math.max(...hist.map((c) => c.high));
  const X = (i) => (i / n) * W, Y = (v) => H - 3 - ((v - lo) / (hi - lo || 1)) * (H - 6);
  const line = hist.map((c, i) => `${i ? "L" : "M"}${X(i + 0.5).toFixed(1)} ${Y(c.close).toFixed(1)}`).join("");
  const t = (c) => fmtHourMinute(c.time * 1000), end = hist[Math.max(0, e - 1)];
  const focused = box.contains(document.activeElement) && document.activeElement.classList.contains("chart-pan-map");
  box.innerHTML = `<svg class="chart-pan-map" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="slider" tabindex="0"`
    + ` aria-label="12시간 안에서 차트 창 위치 — 방향키로 이동" aria-valuemin="${chartWindowBars}" aria-valuemax="${n}" aria-valuenow="${e}"`
    + ` aria-valuetext="${t(hist[s])}부터 ${t(end)}까지">`
    + `<rect class="pan-win" x="${X(s).toFixed(1)}" y="0.5" width="${Math.max(3, X(e) - X(s)).toFixed(1)}" height="${H - 1}" rx="4"/>`
    + `<path class="pan-line" d="${line}"/></svg>`
    + `<span class="chart-pan-when">${t(hist[s])}–${live ? "지금" : t(end)}</span>`
    + (live ? `<span class="chart-pan-live" title="최신 봉을 따라간다">실시간</span>`
            : `<button type="button" class="chart-pan-now" title="최신 봉으로 돌아가기 (더블클릭도 같다)">최신으로</button>`);
  if (focused) box.querySelector(".chart-pan-map")?.focus();
}
function setupChartPan() {
  const box = el("chartPan"), svg = el("candleSvgSnapshot");
  if (!box || !svg) return;
  const full = () => candleHistoryByAsset[activeSnapshotAsset] || [];
  const mapIdx = (ev) => { const m = box.querySelector(".chart-pan-map"); if (!m) return null; const r = m.getBoundingClientRect(), f = full(), n = Math.min(CHART_PAN_BARS, f.length);
    return f.length - n + Math.round(((ev.clientX - r.left) / r.width) * n + chartWindowBars / 2); };
  box.addEventListener("pointerdown", (ev) => {
    if (!ev.target.closest(".chart-pan-map")) return;
    box.setPointerCapture(ev.pointerId);
    const go = (e2) => { const i = mapIdx(e2); if (i != null) chartPanTo(i); };
    go(ev);
    const up = () => { box.removeEventListener("pointermove", go); box.removeEventListener("pointerup", up); box.removeEventListener("pointercancel", up); };
    box.addEventListener("pointermove", go); box.addEventListener("pointerup", up); box.addEventListener("pointercancel", up);
  });
  box.addEventListener("click", (ev) => { if (ev.target.closest(".chart-pan-now")) { chartPanEnd = null; scheduleSnapshotChartRender(); } });
  box.addEventListener("keydown", (ev) => {
    const d = { ArrowLeft: 1, ArrowRight: -1 }[ev.key];
    if (ev.key === "End") { ev.preventDefault(); chartPanEnd = null; scheduleSnapshotChartRender(); return; }
    if (!d) return;
    ev.preventDefault(); chartPanBy(d * (ev.shiftKey ? 6 : 1));
  });
  // 차트 위: 휠(세로·가로 모두) 40px = 1봉 · 끌기 = 봉 폭만큼 · 더블클릭 = 최신으로. 12h 창(움직일 곳 없음)이면 휠을 페이지에 돌려준다.
  const canPan = () => flowOn() && Math.min(full().length, CHART_PAN_BARS) > chartWindowBars;
  let acc = 0;
  svg.addEventListener("wheel", (ev) => {
    if (!canPan() || ev.ctrlKey) return;
    ev.preventDefault();
    acc += Math.abs(ev.deltaX) > Math.abs(ev.deltaY) ? ev.deltaX : ev.deltaY;
    const steps = Math.trunc(acc / 40);
    if (steps) { acc -= steps * 40; chartPanBy(-steps); }   // 아래로·오른쪽으로 = 최신 쪽
  }, { passive: false });
  let drag = null;
  svg.addEventListener("pointerdown", (ev) => {
    if (!canPan() || ev.button !== 0 || ev.pointerType === "touch") return;
    const slot = Math.max(4, (svg.getBoundingClientRect().width * 0.67 - 300) / chartWindowBars);
    drag = { x: ev.clientX, end: chartPanEndIndex(full()), slot, moved: false, id: ev.pointerId };
  });
  svg.addEventListener("pointermove", (ev) => {
    if (!drag || !(ev.buttons & 1)) { drag = null; return; }
    const dx = ev.clientX - drag.x;
    if (!drag.moved && Math.abs(dx) < 5) return;
    if (!drag.moved) { drag.moved = true; svg.setPointerCapture(drag.id); svg.classList.add("panning"); }
    chartPanTo(drag.end - dx / drag.slot);   // 오른쪽으로 끌면 과거가 보인다
  });
  const stop = () => { drag = null; svg.classList.remove("panning"); };
  svg.addEventListener("pointerup", stop); svg.addEventListener("pointercancel", stop);
  svg.addEventListener("dblclick", () => { if (chartPanEnd != null) { chartPanEnd = null; scheduleSnapshotChartRender(); } });
}
let chartPanForce = false;   // 창 이동은 호버 중에도 바로 그린다(호버 보류에 막히면 끌어도 안 움직인다)
function renderSnapshotChart() {
  renderChartPan();
  if (chartHoverActive && !chartPanForce) { chartRenderDeferred = true; return; }
  chartPanForce = false;
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
  // 🔴2026-09-23 풋프린트 모드에서 캔들을 **풋프린트 범위로 자르고** 있었다. 그래서 12h 를
  //   골라도 서버 풋프린트 링이 덮는 만큼(실측 54봉=4.5시간)만 보였다 -- 캔들은 200개
  //   (16.7시간) 있는데도. 위 주석의 이유(«72봉을 1200px 에 넣으면 셀이 안 들어간다»)는
  //   drawCells(봉 폭 기준)가 이미 해결했다: 얇아지면 셀을 안 그린다.
  //   이제 창이 자른다. 풋프린트가 없는 봉은 셀·델타가 그냥 비고(buyTot+sellTot>0 가드),
  //   사분면·누적 레인은 **캔들을 돌며 fpBars 를 조회**하는 구조라 빠진 봉을 알아서 건너뛴다.
  // 2026-09-27 흐름 코인은 풋프린트가 **아직 안 와도** 창으로 자른다 -- 전환 직후 0.3~0.7초 동안 96봉 캔들로
  //   넓어졌다 줄어드는 게 «1h 인데 캔들로 12h» 로 보였다.
  const candles = footprint || flowOn()
    ? fullCandles.slice(Math.max(0, chartPanEndIndex(fullCandles) - chartWindowBars), chartPanEndIndex(fullCandles))
    : fullCandles.slice(-SNAPSHOT_CHART_MAX_CANDLES);
  const currentPrice = Number(latestLivePriceByAsset[activeSnapshotAsset] || candles[candles.length - 1]?.close || 0);
  const riskLevels = [...nearestLiquidationLevel(), ...trendFlipLevels(footprint, candles),
];
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

// 가격 꼬리표 세로 배치(2026-09-26). 가격 순서(rawY)를 지키며 겹치지 않게 민다 -- 위→아래로 최소 간격,
// 아래→위로 바닥 안에 되민다. 자리가 모자라면 겹칠 수는 있어도 **순서는 절대 안 뒤집힌다**.
// 입력: [{rawY, realY}] (realY = 화면 밖이면 가장자리로 clamp 한 y). adjustedY 를 채운다.
function declutterTagY(labels, lo, hi, gap) {
  labels.sort((a, b) => a.rawY - b.rawY);
  let prev = -Infinity;
  labels.forEach((p) => { p.adjustedY = Math.max(p.realY, prev + gap, lo); prev = p.adjustedY; });
  prev = Infinity;
  for (let i = labels.length - 1; i >= 0; i--) {
    labels[i].adjustedY = Math.min(labels[i].adjustedY, prev - gap, hi);
    prev = labels[i].adjustedY;
  }
  return labels;
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
  const wAll = parentW > 0 ? Math.max(parentW, 320) : 1200;
  const hAll = parentH > 0 ? Math.max(parentH, 260) : 400;   // 상자 전체(viewBox). 본문의 `h` 는 아래(모바일은 풋프린트 영역만)
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
  const subOn = svg.id === "candleSvgSnapshot" && flowOn();
  // 2026-09-27 넓은 화면 2단(사용자 선택 62:38): CSS 가 --fp-split(오른쪽 칸 비율)을 준다 -- 판정은 CSS 한 곳.
  //   가격 플롯·레인·밀도 범례는 왼쪽 w, 1초 수급은 오른쪽 칸(subX, subW) 전체 높이. viewBox 는 wAll.
  //   아래 본문은 전부 `w` 를 «플롯 폭»으로 쓰므로 한 줄도 안 바뀐다.
  const splitR = subOn ? Number(getComputedStyle(svg).getPropertyValue("--fp-split")) || 0 : 0;
  const SPLIT_GAP = 20;
  const subW = splitR ? Math.round((wAll - SPLIT_GAP) * splitR) : wAll;
  const w = splitR ? wAll - SPLIT_GAP - subW : wAll, subX = splitR ? w + SPLIT_GAP : 0;
  // 🔴SUB_TOTAL(아래)은 styles.css 의 #candleSvgSnapshot 높이와 **같이** 움직여야 한다 -- 상자가 작으면 가격 플롯이 눌린다
  //   (test_sub_panel_height_contract 가 두 파일을 대조한다). 1초 수급 400 = 가격 플롯과 같은 높이(2026-09-22 사용자 «너무 작아서»).
  const SUB_GAP = 8, SUB_1S_H = subOn ? 400 : 0;
  // 1단(데스크톱·모바일)은 호가 요약 네 숫자가 1초 수급 바닥에서 한 줄(14)을 떼어 간다 -- SUB_TOTAL 은 그대로.
  //   (그냥 풋프린트 위에 얹으면 모바일은 밀도 범례, 데스크톱은 1초 수급의 시각 눈금과 겹쳤다.)
  const STATS_ROW_H = subOn && !splitR ? 14 : 0;
  // 청산 밀도 범례(9px 한 줄) -- 풋프린트 배경을 설명하므로 풋프린트 바로 위(2026-09-21).
  const SUB_LEGEND_H = subOn ? 14 : 0;
  // 순서(2026-09-28, 호가·체결 프로파일 제거 후): 1단(데스크톱·모바일) = 1초 수급 → 밀도 범례 → 풋프린트 → 레인,
  //   2단 = 왼쪽 밀도 범례 → 풋프린트 → 레인 · 오른쪽 1초 수급. 본문의 `h` 는 상자 전체(hAll)다.
  // 2026-10-01 1단(휴대폰·세로 화면)은 1초 수급을 풋프린트 **아래**로(사용자 지시): 위 = 밀도 범례 + 호가 요약 줄 → 풋프린트 → 레인·리본 → 1초 수급.
  //   상자 총높이는 그대로 -- 본문의 바닥 `h` 를 1초 수급 몫만큼 올린다(아래 기하는 전부 h 기준이라 한 줄도 안 바뀐다).
  const S1_BELOW = subOn && !splitR;
  const S1_PANEL = S1_BELOW ? SUB_1S_H - STATS_ROW_H - 18 : 0;
  const h = S1_BELOW ? hAll - S1_PANEL - SUB_GAP : hAll;
  const SUB_TOTAL = !subOn ? 0 : splitR ? SUB_LEGEND_H + SUB_GAP : SUB_LEGEND_H + STATS_ROW_H + 10 + SUB_GAP;
  // 🔴상자 높이(styles.css 의 #candleSvgSnapshot/.candle-container)와 위 SUB_* 상수는 두
  //   파일에 갈라져 있다. 한쪽만 고치면 가격 플롯이 **조용히** 눌린다(ch 에서 SUB_TOTAL 을
  //   빼기 때문). 인라인 height 로 JS 가 상자를 정하는 방법은 쓰지 않는다 -- 2열에서는 상자가
  //   열 높이를 따라 늘어나는 게 의도된 동작인데 인라인 height 가 그 auto 를 이기고, h 자체를
  //   getBoundingClientRect 로 읽으므로 자기참조가 된다. 대신 어긋나면 **한 번 알린다**.
  if (SUB_TOTAL > 0 && h < 400 + SUB_TOTAL - 2 && !renderCandleSvg._subBoxWarned) {
    renderCandleSvg._subBoxWarned = true;
    console.warn(`캔들 상자가 ${Math.round(h)}px 인데 수급 패널이 ${SUB_TOTAL}px 를 쓴다 -- `
      // 🔴이 식은 낡아 있었다(55 는 옛 레인 합). 실제 계약은 12 + SUB_TOTAL + 400(가격
      //   플롯) + 368(레인) + 70(mb) = SUB_TOTAL + 593 이다 -- 높이 계약 테스트와 같은 식.
      + `styles.css 의 #candleSvgSnapshot ${SUB_TOTAL + 593}px / .candle-container `
      + `${SUB_TOTAL + 605}px 로 맞추세요(그만큼 가격 플롯이 눌립니다).`
      + " (레인 368 = 사분면 196 + 누적 160 + 간격 12)");
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
  // 2026-09-22 모바일(사용자 «라벨들을 차트 안으로»): 가격 배지를 플롯 **안**으로 들이고
  //   mr 은 얇은 여백만 남긴다. 실측 348px 화면에서 플롯 280 -> 338px = **+21%**.
  //   배지는 아래 boxX 가 오른쪽 끝 기준으로 다시 잡는다(왼쪽정렬 그대로 두면 밖으로 나간다).
  // 2026-09-28 모바일 체결 띠(사용자 «호가처럼 풋프린트 왼쪽 축으로»): 가격 글자와 캔들 사이 50px. ml 이 «플롯 왼쪽»이라 그만큼 민다.
  const TRADE_L = footprint && mobileChart ? 50 : 0;
  // 2026-09-28 모바일 풋프린트는 왼쪽 가격 글자를 없앴다(사용자 «차트가 너무 작다») -- 값은 플롯 아래 한 줄이 갖는다. 여백 44 -> 4.
  // 2026-09-30 풋프린트(데스크톱)는 가격 글자를 오른쪽 꼬리표 칸(TAG_W)으로 옮겨 왼쪽 여백 71 -> 36(레짐·전환 줄 이름 자리만, 사용자 «풋프린트를 왼쪽으로»).
  const ml = (mobileChart ? (footprint ? 4 : 44) : footprint ? 36 - CENTER_NUDGE : 45) + CENTER_NUDGE + TRADE_L,
        // 2026-09-27 풋프린트는 가격 배지를 전부 왼쪽 글자로 옮겼다(사용자 «호가가 잘 보이게») -- 오른쪽은 호가 띠가 쓴다.
        mr = (mobileChart ? 10 : footprint ? 34 : 112) - CENTER_NUDGE,
        mtTop = 12, mt = mtTop + SUB_TOTAL, mb = 70;
  // 2026-09-21 사용자 요청: 「차트가 너무 많다 -- 레인을 한 덩어리로」 + 「가격 플롯을 키워라」.
  //   레인 5종(거래대금·델타/CVD·OI·수급·청산)이 각자 6px 간격으로 떨어져 있어 **다섯 장의
  //   그림**으로 읽혔다. 간격 6->3, 높이를 서로 가깝게 맞춰 한 블록으로 읽히게 한다:
  //   데스크톱 168 -> 111px. 그 57px 과 상자 증가분을 전부 가격 플롯이 가져간다(205 -> 400).
  // ⭐다섯 레인은 전부 «높이=해상도» 노브다 -- OI·청산은 대칭 이분(half = H/2 - 1), 수급은
  //   SMID 기준, 거래대금은 단극이다. 안에 고정 크기 요소가 없어 높이만 바꾸면 되고 좌표
  //   계산은 한 줄도 안 건드린다. LANE_H(15)는 **다른 것**이다(레짐 리본·구간 줄, 하단 여백).
  // 🔴DCVD 는 24 아래로 내리지 말 것 -- 15 에서 작은 델타 막대가 0.7px 였다(기존 주석).
  // OI 신규계약 레인 -- 청산 레인 **바로 위**(사용자 지시). 별도 패널이 아니라 이 SVG 안의
  // 서브플롯이라야 캔들과 x축(봉)이 구성상 같아진다(레짐 리본이 같은 이유로 여기 있다).
  // 🔴ETH 전용이다. 다른 코인을 보는 동안 ETH 값을 얹으면 2026-08-31 레짐 리본 사고와 같은
  //   모양이 된다 -- 자산 전환에서 latestOi5m 을 비우는 것과 이 조건, 둘 다 필요하다.
  // 데이터가 없으면 자리를 아예 안 잡는다(웜업·다른 코인에서 가격 플롯만 40px 손해다).
  const oiBars = (svg.id === "candleSvgSnapshot" && flowOn()
                  && latestOi5m && Array.isArray(latestOi5m.bars)) ? latestOi5m.bars : [];
  // 모바일 26 / 데스크톱 34. h 는 모바일에서도 실제로 400 이다(styles.css 가 #candleSvgSnapshot
  // 높이를 400px 로 고정 -- `Math.max(parentH, 260)` 의 260 은 SVG 가 안 그려질 때의 바닥값이다).
  // 400 기준 가격 플롯은 모바일 233, 데스크톱 205 로 남는다.
  // ── 고래·리테일 수급 리본 -- OI 레인 바로 아래 (2026-09-20 사용자 지시) ──────────
  // 봉마다 «그 5분에 순 몇 ETH 가 들어왔나»를 두 계층으로 가른다. 풋프린트가 이미 봉별
  // 가격대 셀을 주고 supplyFlowOfBar() 가 그걸 셋으로 가르므로 여기서는 **자리만** 잡는다.
  // 🔴둘은 **한 자**를 나눠 쓴다(각자 정규화하면 「고래 매도 · 리테일 매수」가 같은 크기로
  //   보여 거짓이 된다). 이 리본의 값이 바로 그 엇갈림이다 -- 실측 12봉 중 4봉에서 부호가
  //   반대였다. 자 자체는 log1p 다(아래 hgt 주석).
  // 데이터가 없으면 자리를 아예 안 잡는다(OI 레인과 같은 규약).
  const fpBars = (svg.id === "candleSvgSnapshot" && flowOn()
                   && latestFootprint && Array.isArray(latestFootprint.bars))
                  ? latestFootprint.bars : [];
  // ── 5분봉 레인: 여덟 줄 -> 두 행 (2026-09-22 사용자 선택) ─────────────────────
  // 상관 실측이 근거다(라이브 12봉): 거래대금↔청산 r=+0.958 · 거래대금↔|델타| +0.884 ·
  // |델타|↔청산 +0.882 -- 앞의 셋은 사실상 한 축(«얼마나 시끄러웠나»)이라 세 줄을 쓸 이유가
  // 없다. 반면 델타↔OIΔ 는 **+0.194** 로 거의 독립이라 그 **조합**만이 새 정보다.
  //   ① 사분면 행 -- 높이 |델타| · 농도 |OI| · 색 델타 부호 · 막대 안 해석 · 아래 실제값.
  //      거래대금은 이 행에 **선**으로 얹는다(자리를 안 먹고, 막대와 어긋나는 봉이 곧 흡수다).
  //   ② 누적 행 -- 고래/중형/리테일을 0에서 쌓으면 **그 윤곽이 곧 CVD** 다(실측 오차 0 ETH).
  //      누적 OI 는 같은 ETH 축에 주황 실선. 바로 위 1초 차트와 **같은 문법**이다.
  // 청산은 레인에서 빠져 **풋프린트 봉 고가 바로 위 동그라미**가 됐다 -- 청산은 가격에서
  // 일어나는 사건이라 가격 옆에 있어야 한다.
  // 🔴막대 안 글자는 봉이 좁으면 안 들어간다. 창이 2h/4h(24·48봉)면 슬롯이 48/24px 라
  //   «신규 숏»(12px 기준 ~36px)이 넘친다 -- 그때는 글자를 **안 그리고** 그 자리도 안 잡는다.
  const laneSlot = candles.length ? (w - ml - mr) / candles.length : 0;
  const QUAD_TEXT_OK = laneSlot >= 66;
  // 2026-10-01 넓은 ETH 2단: 사분면 줄을 줄여 가격 플롯 바닥 + 12 = ④ 윗변(③ 가격 지형 = ④ 와 같은 높이, 사용자 지시) -- 줄어든 만큼 풋프린트·프로파일이 길어진다.
  //   ④ 윗변 = hAll − 8 − ④ 높이(시장 맥락이 칸 바닥에 붙는다), 플롯 바닥 = h − 76 − QUAD_H − QUAD_TXT ⇒ QUAD_H + QUAD_TXT = ④ 높이 − 56.
  const QUAD_H = fpBars.length ? (mobileChart ? 112 : 162) : 0;
  const QUAD_TXT = (fpBars.length && QUAD_TEXT_OK) ? (mobileChart ? 28 : 34) : 0;
  const CUM_H = fpBars.length ? (mobileChart ? 104 : 160) : 0;
  // 2026-09-27 데스크톱은 누적 CVD·OI 를 사분면 막대 **뒤에** 흐리게 깐다(사용자 선택 A) -- 제 줄(160+6)을 풋프린트에 준다.
  //   가격 플롯 400 -> 566. 모바일은 제 줄 그대로(사용자 «모바일은 지금대로»).
  const LANE_MERGE = true;   // 2026-09-28 시안 E: 모바일도 사분면 칸 위에 누적을 그린다(제 줄 104+6 은 가격 플롯으로)
  const CUM_DRAW_H = LANE_MERGE ? QUAD_H : CUM_H;   // 누적이 그리는 높이 = 합치면 사분면 막대 판
  // 2026-09-23(2차) 사용자 「RVOL 선 2개는 CVD 차트로 옮겨줘」 -- 전용 레인을 없애고
  // cumLane **안에** 자기 축으로 겹쳐 그린다. 높이 예산은 레인 둘로 되돌아간다.
  const LANE_GAP = fpBars.length ? 6 : 0;
  // 2026-09-22 모바일(사용자 «프로파일처럼 라벨을 차트 아래에 일자로»): 가격 배지와 두 레인
  //   범례를 각자 그림 **아래 한 줄**로 내린다. 데스크톱은 0 이라 레인 합이 안 변한다 --
  //   test_sub_panel_height_contract_20260919 가 데스크톱 값만 뽑아 CSS 와 대조하므로
  //   그 계약이 그대로 유지된다. 값 18/16/16 은 가격 플롯에서 나온다(모바일 540 -> 490px).
  const ROW_H = (fpBars.length && mobileChart) ? 16 : 0;   // 레인 아래 한 줄
  const PRICE_ROW_H = mobileChart ? 18 : 0;                // 가격 플롯 아래 한 줄
  // ── 거래대금 · 델타·CVD -- 풋프린트 바로 아래 (2026-09-21 사용자 지시) ────────────
  // 같은 풋프린트 봉에서 나온다: 거래대금 = Σ가격x(매수+매도) · 델타 = Σ(매수-매도) ·
  // CVD = 그 창 안에서의 델타 누적.
  // 🔴CVD 의 기준점은 **창 시작**이고, 창은 1h/2h/4h 선택기를 그대로 따른다(사용자 결정).
  //   `CHART_WINDOW_BARS` 12/24/48/144 가 곧 그 넷이고 서버 상한(FOOTPRINT_MAX_WINDOW_BARS)도
  //   144 라 넷 다 덮인다. 창을 바꾸면 기준점도 같이 옮겨간다 -- 절대 누적이 아니다.
  // 🔴«상황 읽기» 카드의 CVD 는 **30분 고정창**이다(dashboard/situation.py 의 WINDOW=6).
  //   이름이 같아도 값이 다르다. 툴팁에 창을 적는다.
  // 2026-09-27 풋프린트 오른쪽 «실시간 호가 띠»(사용자 지시) -- 캔들 폭에서 띠 폭을 뗀다(데스크톱 70 · 모바일 40).
  // 2026-09-27 넓혔다(사용자 «호가 너비를 키워줘») -- 오른쪽 배지 자리(86px)를 띠에 줬다: 70 -> 148 · 모바일 40 -> 56.
  const BOOK_W = footprint ? (mobileChart ? 56 : 140) : 0;   // 2026-09-30 148 -> 112 -> 140(꼬리표 칸을 없애 돌려줌)(사용자 «차트·프로파일을 조금 줄여도 데이터가 다 보이게»)   // 모바일은 수량 글자 없이
  // 2026-09-28 체결 기둥(사용자 선택 시안 B) -- 풋프린트와 호가 띠 사이 150px. 모바일은 기둥 없이 셀 뒤에 깐다(시안 A).
  const TRADE_W = footprint && !mobileChart ? 140 : 0;   // 2026-09-30 150 -> 112 -> 140
  // 2026-09-30 가격 꼬리표 칸(사용자 «왼쪽 라벨을 오른쪽으로, 겹치지 않게»): 캔들 칸과 체결 기둥 사이 120px.
  //   가격 선·레벨의 이름+가격을 여기 세로로 쌓고(겹침 회피) 실제 가격 행까지 지시선을 긋는다. 오른쪽 끝 30px 는 행사가 감마 막대.
  // 2026-09-30 가격 이름 = 시안 C(사용자 선택): 꼬리표 칸을 없애(폭은 체결·호가 띠로) 플롯 안 오른쪽 끝에 **짧은 약어**(R1·S1·MP·1h·5m·VW·ΓF·HL롱),
  //   이름·가격은 마우스를 올리면. 현재가는 상자 없이 **숫자만** 크게(바탕 외곽선).
  const LBL = "C";
  const TAG_W = 0;   // 2026-09-30 120 -> 100(사용자 «호가/체결 프로파일을 더 왼쪽으로»)
  const cw = w - ml - mr - BOOK_W - TRADE_W - TAG_W;
  const tradeX0 = ml + cw + TAG_W;   // 체결 기둥 왼쪽 끝(꼬리표 칸 뒤)
  fpProfileSpan = TRADE_W ? [tradeX0, w - mr] : null;   // 아래 옵션 줄 가운데 칸이 이 폭·x 를 따른다(optRowCols)
  let tradeInfo = null;       // 체결 행(호버가 읽는다) -- 풋프린트 블록이 채운다
  let fpRowSize = 0;          // 풋프린트 행 크기($) -- 호가 띠가 같은 행으로 묶는다(2026-09-28)
  const ch = h - mt - mb - QUAD_H - QUAD_TXT - (LANE_MERGE ? 0 : CUM_H + LANE_GAP) - LANE_GAP
            - PRICE_ROW_H - 2 * ROW_H;
  const plotBottom = mt + ch;                      // 가격 플롯의 바닥
  // 수급 두 패널은 **가격 플롯 위**다(위 mt 주석). OI·청산 레인은 플롯 바로 아래 그대로다.
  // 2026-09-22 위아래를 뒤집었다(사용자 지시): **프로파일이 먼저, 1초 수급이 그 아래**.
  // 🔴청산밀도 범례는 프로파일을 **따라 올라가지 않는다**. 그건 풋프린트(가격 플롯)의 배경을
  //   설명하는 것이라 그 바로 위에 있어야 한다(SUB_LEGEND_H 주석의 «풋프린트 차트 바로 위»).
  //   프로파일에 붙여 올리면 설명하는 그림에서 400px 멀어진다.
  // 소비 합 = SUB_TOTAL: 190(프로파일) + 8 + 400(1초) + 14(범례) + 8 = 620.
  const sub1sY = S1_BELOW ? h + SUB_GAP : mtTop;
  const subLegendY = splitR ? mtTop : S1_BELOW ? mtTop : sub1sY + SUB_1S_H - STATS_ROW_H;   // 2단: 왼쪽 칸 맨 위 · 그 밖: 1초 수급 바로 아래(= 풋프린트 바로 위)   // 2단: 왼쪽 칸 맨 위(풋프린트 바로 위)
  const quadY = plotBottom + PRICE_ROW_H + LANE_GAP;  // 사분면 막대 바닥 = quadY + QUAD_H
  const cumY = LANE_MERGE ? quadY : quadY + QUAD_H + QUAD_TXT + ROW_H + LANE_GAP; // 누적 행 위쪽
  const cumBottom = cumY + CUM_DRAW_H;                    // 그 아래 한 줄이 ROW_H 를 쓴다
  const NS = "http://www.w3.org/2000/svg";
  // 풋프린트는 서버가 주는 12봉이 곧 창이다 -- 모바일 핀치줌(visibleCandleWindow)으로 더
  // 잘라내면 셀만 커지고 볼 구간이 사라진다.
  const viewport = footprint ? { candles, includeCurrent: true } : visibleCandleWindow(candles);
  candles = viewport.candles;
  const includeCurrentPrice = viewport.includeCurrent;

  svg.setAttribute("viewBox", `0 0 ${wAll} ${hAll}`);
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
  // 2026-10-01 과거 창(chartPanEnd)이면 축은 **보이는 봉만** -- 현재가를 넣고 가운데에 두면 과거 봉이 한쪽에 눌렸다(실측)
  const panned = svg.id === "candleSvgSnapshot" && chartPanEnd != null;
  if (includeCurrentPrice && currentPrice > 0 && !panned) allPrices.push(currentPrice);

  const minP = Math.min(...allPrices), maxP = Math.max(...allPrices);
  // 2026-09-21 사용자 요청 «청산맵 위아래 여유». 여백은 모드마다 값이 다르다:
  //  · 청산맵  -- 넓힐수록 위아래 청산 레벨과 그 라벨이 화면 안으로 들어온다. 캔들이 조금
  //    납작해지는 대가는 청산맵에서 작다(셀 숫자를 읽을 일이 없다).
  //  · 풋프린트 -- 넓히면 행 높이(rowPx)가 줄어 셀 숫자가 먼저 깨진다. 여기는 그대로 둔다.
  //    (rowSize 는 ySpan/ch 로 정해진다 -- 여백을 늘리면 ySpan 이 커져 행이 얇아진다.)
  const padPct = footprint ? CHART_Y_PAD_FOOTPRINT : CHART_Y_PAD_PLAIN;
  const pad = (maxP - minP) * padPct || 1;
  let yMin = minP - pad, yMax = maxP + pad;
  // 2026-09-27 풋프린트는 현재가를 세로 **정중앙**에(사용자 지시) -- 먼 쪽 거리로 위아래 대칭.
  //   가운데는 **5분봉마다 한 번** 잡는다(같은 날 사용자 «매 5분봉마다»): 봉이 바뀔 때의 현재가로 정하고 그 봉 동안은
  //   고정한다 -- 틱마다 따라가면 화면 전체가 계속 출렁인다. 봉 안에서 범위를 벗어나면 가운데는 두고 위아래만 넓힌다.
  //   🔴가운데는 풋프린트 칸 단위, 반폭은 두 칸 단위로 반올림한다(yMin/yMax 가 층 캐시 서명 baseGeomSig 에 들어 있다).
  if (footprint && currentPrice > 0 && !panned) {
    const q = Number(footprint.bucket) || 0;
    const barT = candles.length ? candles[candles.length - 1].time : 0;
    const fc = renderCandleSvg._fpCenter || (renderCandleSvg._fpCenter = { key: "", c: 0 });
    const fcKey = activeSnapshotAsset + "|" + barT;
    if (fc.key !== fcKey) {
      fc.key = fcKey;
      fc.c = q > 0 ? Math.round(currentPrice / q) * q : currentPrice;
    }
    const c = fc.c;
    let half = Math.max(yMax - c, c - yMin);
    if (q > 0) half = Math.ceil(half / (2 * q)) * 2 * q;
    yMin = c - half; yMax = c + half;
  }
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
                       mobileChart, quadY, cumY, QUAD_H, QUAD_TXT, CUM_H, CUM_DRAW_H,
                       QUAD_TEXT_OK, ROW_H, PRICE_ROW_H].join("|");
  // 봉 시각만. 진행 중인 봉의 OHLC 는 여기 없다(위 주석).
  const timesSig = candles.length + ":" + (candles[0] ? candles[0].time : 0)
                   + ":" + (candles[candles.length - 1] ? candles[candles.length - 1].time : 0);
  const layerCache = renderCandleSvg._layers || (renderCandleSvg._layers = new Map());
  /** 층 하나를 <g> 로 묶어 캐시한다. sig 가 같으면 만들어 둔 노드를 그대로 다시 붙인다.
   *  draw(g) 는 그 <g> 안에만 그려야 한다 -- svg 에 직접 붙이면 캐시를 우회한다. */
  const cachedLayer = (name, sig, draw, before = null) => {
    const full = baseGeomSig + "|" + timesSig + "|" + sig;
    const prev = layerCache.get(name);
    // before: 그 층 **뒤에**(먼저) 깐다 -- 2026-09-27 누적 레인을 사분면 막대 뒤로.
    const under = before && layerCache.get(before)?.g;
    const put = (x) => (under && under.parentNode === svg ? svg.insertBefore(x, under) : svg.appendChild(x));
    if (prev && prev.sig === full) { put(prev.g); return prev.g; }
    const g = document.createElementNS(NS, "g");
    draw(g);
    put(g);
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
  // 이력이 시작되는 시각. 리본·툴팁이 «모름»과 «횡보»를 가르는 데 쓴다 -- 호버마다 다시 세지
  // 않도록 여기서 한 번만 구한다(2026-09-25).
  const regimeFromTs = regimeByTsForChart && regimeByTsForChart.size
    ? Math.min(...regimeByTsForChart.keys()) : null;
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
  // B: 마지막 스냅샷이 모르는 구간(ts+1h 이후)에 가격이 지나간 칸 -> 그 봉 가운데서 끊고 뒤는 «쓸림»(t=0)으로.
  //   앞 스냅샷들은 자기 시간봉까지 이미 반영돼 있다. 키에 넣어야 틱으로 새로 쓸릴 때 층이 다시 그려진다.
  const densitySweep = new Map();
  const lastSnap = sortedDensityHistory[sortedDensityHistory.length - 1];
  if (lastSnap) {
    const since = Math.floor((lastSnap.tsMs || 0) / 1000) + 3600;
    (lastSnap.bins || []).forEach((b) => {
      if (!(b.weightPct > 0)) return;
      const i = liqSweepIdx(candles, since, b.price);
      if (i >= 0) densitySweep.set(b.price, i);
    });
  }
  if (sortedDensityHistory.length) cachedLayer("density",
      objToken(densityHistory) + ":" + densityClip + ":" + [...densitySweep].join(","), (g) => {
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
      const cut = si === sortedDensityHistory.length - 1 ? densitySweep.get(price) : undefined;
      if (cut !== undefined && cut >= xStartIdx) {
        const xc = Math.min(x1, Math.max(x0, xAt(cut) + bw / 2));
        drawDensitySeg(g, x0, xc, top, bottom, t);
        drawDensitySeg(g, xc, x1, top, bottom, 0);
      } else {
        drawDensitySeg(g, x0, x1, top, bottom, t);
      }
    });
  });
  });

  // Resistance/support/current/entry price tags -- computed here (before the axis ticks below) so
  // the tick loop can tell when a grid label would land on top of one of these and skip it.
  // Rendering (the actual lines/boxes) still happens later, after candles/markers, so paint order
  // is unchanged.
  const priceLabels = [];
  // 2026-09-30 겹침 선의 이름(VWAP·주간/앵커 VWAP·HL 청산·max pain·감마 플립·옵션 괄호)도 같은 꼬리표 칸으로 -- 선은 제자리에서 긋고 글자만 모은다.
  const fpNotes = [], gutOn = footprint && !mobileChart;
  // 2026-09-16 풋프린트에서는 현재가를 **가로선이 아니라 삼각형**으로 찍는다(사용자 요청).
  // 선은 가격 행을 가로질러 셀 숫자를 덮는데, 하필 현재가 근처가 제일 중요한 행이다.
  // 삼각형은 플롯 **바깥**(오른쪽 가장자리)에 앉아 어느 행인지만 가리키고 아무것도 안 가린다.
  // 청산맵 모드는 그대로 선이다 -- 거기선 덮을 셀이 없고, 선이 가격대를 가로로 읽게 해 준다.
  if (includeCurrentPrice && currentPrice > 0) {
    // 2026-09-27 풋프린트도 다시 **가로선**이다(사용자 «현재가는 가격선»). 09-16 의 걱정(선이 셀 숫자를 덮는다)은
    //   선을 **셀 뒤**(격자 층 바로 위)에 얇게 깔아 푼다 -- 아래 priceLabels 루프의 insertBefore.
    priceLabels.push({ val: currentPrice, color: "var(--accent)", label: "현재", dashed: true,
                       width: footprint ? 1 : 2, marker: false, behindCells: !!footprint });
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
    p.rawY = rawY;
    p.offTop = rawY < mt;
    p.offBottom = rawY > plotBottom;
    p.outOfView = p.offTop || p.offBottom;
    p.realY = p.outOfView ? (p.offTop ? mt + 2 : plotBottom - 2) : rawY;
  });

  // 🔴2026-09-26 비평: 꼬리표가 겹치고 **순서가 뒤집혔다**(«↑ 2765» 가 2708.5 아래). 원인 둘 --
  //   ① 화면 밖 레벨을 같은 가장자리 픽셀(realY)로 정렬해 동점을 삽입순(가까운 것 먼저)으로 풀었다 →
  //     더 먼(더 높은) 가격이 더 아래에 쌓였다. ② 화면 안 꼬리표는 **바로 앞 하나**하고만 비교했다 →
  //     아래 가장자리에서 위로 쌓인 것과 다시 겹쳤다.
  //   고침: 실제 가격 순(rawY)으로 정렬하고, 위→아래 한 번(최소 간격 밀기)·아래→위 한 번(바닥 안으로
  //   되밀기) 훑는다. 순서는 항상 가격 순이고, 자리가 모자라지 않는 한 겹치지 않는다.
  //   minGap 은 꼬리표 상자 높이(18px)보다 커야 틈이 보인다. 범위는 그리는 쪽의 clamp(mt+9 ~ plotBottom-9)와 같다.
  declutterTagY(priceLabels, mt + 9, plotBottom - 9, 22);

  // 격자·x눈금은 기하와 봉 시각만 보므로 한 층으로 묶어 캐시한다(cachedLayer 주석).
  cachedLayer("grid", "", (g) => {
  // Grid & Y-Axis Ticks
  axisTicks(yMin, yMax, 6).forEach(t => {
    const y = yAt(t);
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", ml - TRADE_L); line.setAttribute("x2", w - mr);
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
    // 🔴2026-09-25 «모름»과 «횡보»가 같은 그림이었다. 아래 규약이 «칸 없음 = 횡보»인데,
    //   워커 이력이 창보다 짧으면(HISTORY_BARS_RETURNED) 그 앞 봉도 똑같이 칸이 없다.
    //   12시간 창에서 앞 24봉이 실제로 그랬다 -- 데이터가 없는 구간이 «횡보였다»로 읽혔다.
    //   이력이 시작되기 **전** 구간에만 흐린 띠를 깔아 둘을 가른다(안쪽 결손은 워커가 안 만든다).
    const oldest = regimeFromTs == null ? [] : candles.filter((c) => c.time < regimeFromTs);
    if (oldest.length) {
      const j = candles.indexOf(oldest[oldest.length - 1]);
      const un = document.createElementNS(NS, "rect");
      un.setAttribute("x", xAt(0)); un.setAttribute("y", REGIME_RIBBON_Y);
      un.setAttribute("width", Math.max(1, xAt(j) + laneW - xAt(0)));
      un.setAttribute("height", REGIME_RIBBON_H); un.setAttribute("rx", laneRx);
      un.setAttribute("fill", "var(--muted)"); un.setAttribute("fill-opacity", "0.14");
      const ut = document.createElementNS(NS, "title");
      ut.textContent = `레짐 모름 -- 워커 이력(${regimeByTsForChart.size}봉)이 이 창보다 짧다.`
        + " 빈 칸이 아니라 «안 잰 구간»이다(빈 칸은 횡보를 뜻한다).";
      un.appendChild(ut);
      g.appendChild(un);
    }
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
    fpRowSize = rowSize;

    // ── 체결 프로파일 (2026-09-28 사용자 선택 시안 B · 모바일 A) ─────────────────────────
    //   보이는 봉의 셀을 **이 행 크기로** 가격별로 합친다 -- 옛 오른쪽 열 상자(/api/supply-profile)를 대체한다.
    //   원천·창·행이 풋프린트와 같아 둘이 어긋날 수 없다(옛 상자는 재시작 뒤 OKX 가 덮은 봉부터만 셌다).
    //   데스크톱: 풋프린트와 호가 띠 사이 기둥, 호가 띠 바닥선에서 **왼쪽으로**(체결 ← ┃ → 호가) -- 같은 줄 = 같은 가격이라
    //   «여기서 얼마나 거래됐나 vs 지금 얼마나 걸려 있나»를 한 줄에서 맞댄다. 모바일: 가격 글자와 캔들 사이 띠(TRADE_L), 캔들 왼쪽 끝에서 **왼쪽으로**(호가 띠의 거울).
    //   막대 = 행 총 체결(매수+매도) · 안쪽부터 고래·중형·리테일 농담 · POC(최다 체결) 행은 테두리.
    //   🔴체결 합산에는 방향이 없다(누가 사면 누가 판다) -- 지지·저항으로 읽지 않는다. 방향은 봉별 델타가 말한다.
    {
      const tr = new Map();   // 행 키 -> [총, 고래, 리테일, 매수, 매도]
      candles.forEach((c) => (footprint.byTime.get(c.time) || []).forEach((l) => {
        const k = Math.floor(l[0] / rowSize), a = tr.get(k) || [0, 0, 0, 0, 0];
        a[0] += (+l[1] || 0) + (+l[2] || 0); a[1] += (+l[3] || 0) + (+l[4] || 0);
        a[2] += (+l[5] || 0) + (+l[6] || 0); a[3] += +l[1] || 0; a[4] += +l[2] || 0;
        tr.set(k, a);
      }));
      let tmax = 0, pocK = null;
      tr.forEach((a, k) => { if (a[0] > tmax) { tmax = a[0]; pocK = k; } });
      if (tmax > 0) {
        const anchor = TRADE_W ? tradeX0 + TRADE_W + 4 : ml - 2;   // 데스크톱 = 호가 띠 바닥선 2px 앞 · 모바일 = 캔들 왼쪽 끝(밖으로 자란다)
        const L = TRADE_W ? TRADE_W - 10 : TRADE_L - 6, alpha = 1;
        tradeInfo = { rows: tr, rowSize, max: tmax, pocK, x0: TRADE_W ? tradeX0 : ml - TRADE_L, x1: anchor };
        const g = document.createElementNS(NS, "g");
        tr.forEach((a, k) => {
          const yTop = Math.max(mt, yAt((k + 1) * rowSize)), yBot = Math.min(plotBottom, yAt(k * rowSize));
          if (yBot - yTop < 1) return;
          let cur = 0;
          [a[1], Math.max(0, a[0] - a[1] - a[2]), a[2]].forEach((v, s) => {
            const wd = L * v / tmax;
            if (!(wd > 0)) return;
            const r = document.createElementNS(NS, "rect");
            r.setAttribute("x", anchor - cur - wd); r.setAttribute("y", yTop + 0.5);
            r.setAttribute("width", wd); r.setAttribute("height", Math.max(1, yBot - yTop - 1));
            r.setAttribute("fill", "var(--amber)"); r.setAttribute("fill-opacity", ([0.95, 0.55, 0.28][s] * alpha).toFixed(2));
            g.appendChild(r);
            cur += wd;
          });
          // 2026-09-28 매수·매도 우위(사용자 선택 시안 B): 막대 **바깥 끝**에서 |매수−매도| 만큼을 초록/빨강으로 칠한다 --
          //   총량과 같은 자라 «이 줄 체결 중 한쪽으로 쏠린 몫»이 길이로 읽힌다(|Δ| ≤ 총량이라 막대 밖으로 안 나간다).
          const dq = a[3] - a[4], dw = L * Math.abs(dq) / tmax;
          if (dw > 0.5) {
            const d = document.createElementNS(NS, "rect");
            d.setAttribute("x", anchor - cur); d.setAttribute("y", yTop + 0.5);
            d.setAttribute("width", dw); d.setAttribute("height", Math.max(1, yBot - yTop - 1));
            d.setAttribute("fill", dq >= 0 ? "var(--good)" : "var(--bad)");
            d.setAttribute("fill-opacity", (0.95 * alpha).toFixed(2));
            g.appendChild(d);
          }
          if (k === pocK) {
            const o = document.createElementNS(NS, "rect");
            o.setAttribute("x", anchor - cur); o.setAttribute("y", yTop + 0.5);
            o.setAttribute("width", cur); o.setAttribute("height", Math.max(1, yBot - yTop - 1));
            o.setAttribute("fill", "none"); o.setAttribute("stroke", "var(--ink)");
            o.setAttribute("stroke-width", "1.2"); o.setAttribute("stroke-opacity", "0.9");
            g.appendChild(o);
          }
        });
        if (TRADE_W || TRADE_L) {                 // 기둥 바닥선 -- 호가 띠 바닥선과 한 줄
          const ln = document.createElementNS(NS, "line");
          ln.setAttribute("x1", anchor + 1); ln.setAttribute("x2", anchor + 1);
          ln.setAttribute("y1", mt); ln.setAttribute("y2", plotBottom);
          ln.setAttribute("stroke", "var(--soft-line)");
          g.appendChild(ln);
        }
        svg.appendChild(g);                        // 셀보다 먼저
      }
    }

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
    // 🔴2026-09-23 사용자 «12시간은 풋프린트 말고 일반 캔들에 델타만». 봉이 얇아지면 셀은
    //   정보가 아니라 잡음이다 -- 144봉(12h)이면 bw≈9px 라 매수/매도 칸이 4px 짜리 색 띠가
    //   되어 캔들을 덮는다. 그 아래 **델타·사분면·누적 CVD 는 그대로 나온다**: 셋 다
    //   풋프린트 «데이터»에서 나오고(QUAD_H/CUM_H 가 fpBars.length 로 켜진다), 여기서 막는
    //   것은 «그리기»뿐이다. 캔들(심지·몸통)은 이 블록 뒤에서 어차피 그려진다.
    //   기준을 창(chartWindowBars)이 아니라 **봉 폭**으로 잡는다 -- 모바일·좁은 창에서도
    //   같은 규칙이 되고, 창 상수가 늘어도 따라온다. 48봉 bw≈26 ✓ · 144봉 bw≈9 ✗.
    const drawCells = bw >= 14;
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
                     maxBuy, maxSell, half, fontPx, showQty, drawCells, INK_OPACITY].join("|");
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
      title.textContent = price.toFixed(pxDp()) + " · " + (color === "var(--good)" ? "매수 " : "매도 ")
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
      if (drawCells) rows.forEach((cell, key) => {
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
          pocTitle.textContent = "POC " + price.toFixed(pxDp()) + " · 총 " + pocVol.toFixed(1);
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

    // ── 시장 맥락 겹침 (2026-09-29, 사용자 «불합격이어도 쓸모 있으면 넣어줘 -- 우리 검증이 틀렸을 수도 있다») ──────────
    //   ① 스택 불균형(같은 쪽 3줄 연속) (② 패턴 글자 다이버·소진·흡수?는 09-30 제거 -- 4.7년·2025/2026 검정 방향 0) ③ 세션 VWAP ±1·2σ · 볼린저(카드 스위치)
    //   ④ HL 고래 실측 청산가 점선 (⑤ 12시간 실측 청산 눈금은 09-30 제거). 전부 서술 -- 뜻과 우리 검정은 각 <title>.
    //   선(③④)은 옵션 선과 같이 격자 바로 위(셀 아래)에, 표식(①②⑤)은 셀 위에. 가격 플롯 밖은 네이티브 clipPath 로 자른다.
    if (isSnapshotChart) {
      const clipId = (svg.id || "chart") + "-mcclip";
      let cr = svg.querySelector("clipPath#" + clipId + " rect");
      if (!cr) {
        const defs = document.createElementNS(NS, "defs"), cp = document.createElementNS(NS, "clipPath");
        cp.setAttribute("id", clipId); cr = document.createElementNS(NS, "rect");
        cp.appendChild(cr); defs.appendChild(cp); svg.appendChild(defs);
      }
      cr.setAttribute("x", ml); cr.setAttribute("y", mt); cr.setAttribute("width", cw); cr.setAttribute("height", ch);   // 창·폭이 바뀌면 따라간다
      const mk = (parent, tag, attrs, text) => {
        const e = document.createElementNS(NS, tag);
        Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
        if (text != null) { const t = document.createElementNS(NS, "title"); t.textContent = text; e.appendChild(t); }
        parent.appendChild(e);
        return e;
      };
      const lines = document.createElementNS(NS, "g");
      lines.setAttribute("clip-path", `url(#${clipId})`);
      const cx = (i) => (xAt(i) + bw / 2).toFixed(1);
      const pl = (pts, attrs, tip) => { if (pts.length > 1) mk(lines, "polyline", { points: pts.join(" "), fill: "none", ...attrs }, tip); };
      // ③ VWAP ±σ -- 서버가 봉마다 두 벌(하루 · 시장 세션)을 싣고 mcVwapOf 가 스위치대로 고른다. 기준이 다시 시작하는 곳(seg)에서 선을 끊는다.
      if (mcLine.vwap) {
        const vws = candles.map(mcVwapOf);
        [[0, "var(--ink)", 0.6, "6 3", 1.3], [1, "var(--muted)", 0.45, "1 4", 1], [-1, "var(--muted)", 0.45, "1 4", 1],
         [2, "var(--muted)", 0.35, "1 6", 1], [-2, "var(--muted)", 0.35, "1 6", 1]].forEach(([k, col, op, dash, sw]) => {
          let seg = null, pts = [];
          const tip = k === 0 ? `${mcLine.vsess ? "시장 세션 개장부터" : "하루(UTC 00시부터)"} VWAP · 거래량 가중 평균가. 볼린저·VWAP 셋업은 우리 검정에서 방향 엣지가 없었다 — 위치 참고.`
                              : `VWAP ${k > 0 ? "+" : "−"}${Math.abs(k)}σ(거래량 가중 표준편차)`;
          const flush = () => { pl(pts, { stroke: col, "stroke-opacity": op, "stroke-dasharray": dash, "stroke-width": sw }, tip); pts = []; };
          vws.forEach((v, i) => {
            if (!v) return;
            if (v.seg !== seg) { flush(); seg = v.seg; }
            pts.push(`${cx(i)},${yAt(v.vwap + k * v.vsd).toFixed(1)}`);
          });
          flush();
        });
        const lv = [...vws].reverse().find(Boolean), lab = lv && mcLine.vsess ? `VWAP·${lv.name}` : "VWAP";
        if (lv && yAt(lv.vwap) > mt + 8 && yAt(lv.vwap) < plotBottom - 2) {
          if (gutOn) fpNotes.push({ val: lv.vwap, label: lab, color: "var(--muted)" });
          else mk(svg, "text", { x: ml + 4, y: yAt(lv.vwap) - 3, "font-size": mobileChart ? 9 : 10, "font-weight": 700, fill: "var(--muted)", "pointer-events": "none" }).textContent = lab;
        }
      }
      // ③ 볼린저(20, 2σ) -- 창 슬라이스로는 20봉이 안 돼 전체 이력에서 센다
      if (mcLine.bb) {
        const bbm = mcBollinger(candleHistoryByAsset[activeSnapshotAsset] || []);
        [0, 2].forEach((j) => pl(candles.map((c, i) => (bbm.has(c.time) ? `${cx(i)},${yAt(bbm.get(c.time)[j]).toFixed(1)}` : null)).filter(Boolean),
          { stroke: "var(--muted)", "stroke-opacity": 0.55, "stroke-dasharray": "3 3", "stroke-width": 1 },
          "볼린저 밴드(20봉 · 2σ). 우리 검정: 4시간 볼린저는 평균회귀·이탈 추종 둘 다 비용을 못 넘었다 — 위치 참고."));
      }
      // ③ 주간·앵커 VWAP(2026-09-30 사용자 지시, 기본 끔) -- 체결 테이프(/api/market-context profile). 지금 값 하나라 가로선.
      //   우리 검정: VWAP 되돌림·위/아래 방향 모두 가짜 레벨과 같았다 — 지지·저항 아님, 위치 참고.
      const pf = activeSnapshotAsset === "eth" && mcLine.wvwap ? ((latestMarketCtx || {}).profile || null) : null;
      if (pf && pf.available) {
        let lastY = -99;
        [["vwap_week", "주간 VWAP", "월 00:00 UTC 부터", "var(--ink)", "8 4", 0.55],
         ["avwap_hi", "앵커 VWAP·전일고", "전일 고가를 처음 찍은 시각부터", "var(--muted)", "2 3", 0.7],
         ["avwap_lo", "앵커 VWAP·전일저", "전일 저가를 처음 찍은 시각부터", "var(--muted)", "2 3", 0.7]]
          .filter(([k]) => pf[k] > 0).map((r) => [...r, yAt(pf[r[0]])]).sort((a, c) => a[6] - c[6])
          .forEach(([k, nm, from, col, dash, op, y]) => {
            if (y < mt || y > plotBottom) return;
            mk(lines, "line", { x1: ml, x2: ml + cw, y1: y.toFixed(1), y2: y.toFixed(1), stroke: col, "stroke-opacity": op, "stroke-dasharray": dash, "stroke-width": 1.2 },
               `${nm} ${pf[k].toFixed(1)} (${from} 거래량 가중 평균가). 우리 검정: VWAP 되돌림·방향 모두 가짜 레벨과 같았다 — 지지·저항으로 읽지 말 것.`);
            if (gutOn) { fpNotes.push({ val: pf[k], label: nm.replace("앵커 VWAP·", "앵커·"), color: "var(--muted)" }); return; }
            if (y - lastY < 12) return;
            lastY = y;
            mk(svg, "text", { x: ml + 4, y: (y - 3).toFixed(1), "font-size": mobileChart ? 9 : 10, "font-weight": 700, fill: "var(--muted)", "pointer-events": "none" }).textContent = nm;
          });
      }
      // ④ HL 고래 실측 청산가 -- 추정 청산맵(배경)과 따로, 거래소가 준 실제 청산가(추적 300주소)
      const mc = activeSnapshotAsset === "eth" && latestMarketCtx && latestMarketCtx.available ? latestMarketCtx : null;
      if (mc && mc.hl_liq) {
        [["below", "var(--bad)", "롱"], ["above", "var(--good)", "숏"]].forEach(([side, col, nm]) => {
          [...(mc.hl_liq[side] || [])].sort((a, c) => c.usd - a.usd).slice(0, 2).forEach((l) => {
            const y = yAt(l.px);
            if (y < mt || y > plotBottom) return;
            mk(lines, "line", { x1: ml, x2: ml + cw, y1: y.toFixed(1), y2: y.toFixed(1), stroke: col, "stroke-opacity": 0.55, "stroke-dasharray": "1 3", "stroke-width": 1.2 },
               `HL 고래 ${nm} 청산가 ${l.px} · ${fmtUsdCompact(l.usd)} (${l.n}주소, 추정 아님). 우리 검정: 미검정.`);
            if (gutOn) { fpNotes.push({ val: l.px, label: `HL ${nm} ${fmtUsdCompact(l.usd)}`, color: col, noPrice: true }); return; }
            mk(svg, "text", { x: ml + cw - 4, y: (y - 3 < mt + 11 ? y + 11 : y - 3).toFixed(1), "text-anchor": "end", "font-size": mobileChart ? 9 : 10,   // 선 위(바닥 태그와 안 겹치게) · 위 끝 근처면 선 아래
                              "font-weight": 700, fill: col, "fill-opacity": 0.85, "pointer-events": "none" }).textContent = `HL ${nm}청산 ${fmtUsdCompact(l.usd)}`;
          });
        });
      }
      const gridG = layerCache.get("grid")?.g;
      if (gridG && gridG.parentNode === svg) svg.insertBefore(lines, gridG.nextSibling); else svg.appendChild(lines);
      // ⑤ 12시간 실측 청산 눈금은 2026-09-30 뺐다(사용자 지시) -- 12시간 최다 가격은 시장 맥락 카드 한 줄에 남는다.
      if (drawCells) {
        // ① 스택 불균형 -- 같은 봉 안에서 반대편의 3배 넘는 줄이 같은 쪽으로 3줄 이상 이어지면 바깥에 굵은 막대
        const STACK_MIN = 3;
        barRows.forEach((rows, i) => {
          const keys = [...rows.keys()].sort((a, c) => a - c), x = xAt(i);
          [[0, 1, x + bw + 3, "var(--good)", "매수"], [1, 0, x - 5, "var(--bad)", "매도"]].forEach(([s, o, ex, col, nm]) => {
            let run = [];
            const flush = () => {
              if (run.length >= STACK_MIN) {
                const yTop = Math.max(mt, yAt((run[run.length - 1] + 1) * rowSize)), yBot = Math.min(plotBottom, yAt(run[0] * rowSize));
                if (yBot - yTop > 2) mk(svg, "rect", { x: ex, y: yTop.toFixed(1), width: 2, height: (yBot - yTop).toFixed(1), fill: col, rx: 1 },
                  `스택 불균형 · ${nm} ${run.length}줄 연속(반대편의 ${FOOTPRINT_IMBALANCE_RATIO}배 넘는 줄이 이어짐). 우리 검정: 미검정 — 흡수·소진 가족은 봉·60초·가격행 단위에서 방향 0이었다.`);
              }
              run = [];
            };
            keys.forEach((k) => {
              const c = rows.get(k), imb = c[s] > 0 && c[s] > c[o] * FOOTPRINT_IMBALANCE_RATIO;
              if (imb && run.length && k !== run[run.length - 1] + 1) flush();
              if (imb) run.push(k); else flush();
            });
            flush();
          });
        });
      }
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

  // ── 추세 veto (2026-09-22, 사용자 요청 「풋프린트에 그려줘」) ─────────────────────────
  // SMA144(12h) ± 1.0×ATR144 히스테리시스. 밴드를 벗어날 때만 허용 측면이 바뀌고, 안이면
  // 직전 상태를 유지한다 -- 원시 부호는 중앙 런이 15분(하루 13.4회 뒤집힘)이라 화면에 못 쓴다.
  // 값은 서버가 붙여준다(dashboard/server.py::trend_veto_rows) -- 클라는 100봉만 갖고 있어
  // SMA144 를 스스로 못 만든다. **두 모드 모두에 그린다**(같은 renderCandleSvg 라 분기 없음).
  // 🔴레짐 리본과 다른 물건이다: 리본은 학습된 3분류(강세/약세/횡보), 이건 규칙 하나다.
  {
    const vp = candles.map((c, i) => ({ i, sma: Number(c.sma), atr: Number(c.atr), v: Number(c.veto) }))
                      .filter((p) => Number.isFinite(p.sma) && Number.isFinite(p.atr));
    // 밴드 배수는 **서버가 준다**(dashboard/server.py TREND_VETO_K). 여기 숫자를 또 적으면
    // 두 곳을 따로 고쳐야 하고, 실제로 2026-09-23 에 K 가 1.0 -> 2.0 으로 바뀌었다.
    // 🔴폴백은 **서버 기본값과 같아야** 한다(TREND_VETO_K=1.0). 2 로 두면 vk 를 안 보내는
    //   옛 서버와 붙었을 때 서버는 K=1 로 측면을 정하는데 화면은 ±2 로 그려 **밴드와 색이
    //   어긋난다**(가격이 ±1 을 넘었는데 밴드 안에 있는 것처럼 보인다).
    const bandK = Number((candles.find((c) => Number.isFinite(Number(c.vk))) || {}).vk) || 1;
    // 2026-09-22 사용자 요청 「현재가에 맞게 현재 봉에서 움직이게」. 서버는 **마감된 봉**만 준다
    // (evidence_signal_cache 의 closed_df) -- 형성 중 봉은 updateSnapshotCandleLive() 가 따로
    // 밀어 넣으므로 sma 가 없어 위 filter 에서 빠지고, 선이 오른쪽 끝에서 한 봉 모자랐다.
    // 마감봉에서 한 칸만 전진시킨다: sma += (현재가 - 창에서 빠지는 종가)/n, atr 은 Wilder 한 스텝.
    // 🔴이 한 점은 **미확정**이다 -- 검정은 종가 기준이었다. 그래서 점선으로 그린다.
    let live = null;
    const lastC = candles[candles.length - 1], prevP = vp[vp.length - 1];
    if (lastC && prevP && prevP.i === candles.length - 2 && !Number.isFinite(Number(lastC.sma))) {
      const prevC = candles[candles.length - 2], n = Number(prevC?.vn), drop = Number(prevC?.drop);
      if (n > 0 && Number.isFinite(drop) && Number(lastC.close) > 0) {
        const sma = prevP.sma + (lastC.close - drop) / n;
        const tr = Math.max(lastC.high - lastC.low, Math.abs(lastC.high - prevC.close),
                            Math.abs(lastC.low - prevC.close));
        const atr = prevP.atr + (tr - prevP.atr) / n;
        const dev = (lastC.close - sma) / sma, eps = bandK * atr / lastC.close;
        live = { i: candles.length - 1, sma, atr,
                 v: dev > eps ? 1 : dev < -eps ? -1 : prevP.v };
        vp.push(live);
      }
    }
    if (vp.length >= 2) {
      const g = document.createElementNS(NS, "g");
      // 🔴가격 플롯 **밖으로 새지 않게** 자른다(2026-09-23 회귀 수정). `yAt` 은 경계를 모르므로
      //   SMA144 가 보이는 가격 범위에서 멀어지면 밴드가 플롯 바닥을 넘어 **아래 레인(사분면·
      //   누적 CVD) 위에 그려진다**. 실측: 플롯 바닥 y=1032 인데 밴드 면이 y=1202 까지 내려와
      //   사분면 막대(1038~1170)를 통째로 덮었다. 레인 좌표는 정상이었다 -- 새는 건 밴드였다.
      //   좌표를 손으로 자르지 않고 **SVG 네이티브 clipPath** 를 쓴다(선이 경계에서 끊긴다).
      const clipId = (svg.id || "chart") + "-plotclip";
      if (!svg.querySelector("clipPath#" + clipId)) {
        const defs = document.createElementNS(NS, "defs");
        const cp = document.createElementNS(NS, "clipPath");
        cp.setAttribute("id", clipId);
        const cr = document.createElementNS(NS, "rect");
        cr.setAttribute("x", ml); cr.setAttribute("y", mt);
        cr.setAttribute("width", cw); cr.setAttribute("height", ch);
        cp.appendChild(cr); defs.appendChild(cp); svg.appendChild(defs);
      }
      // 🔴그림만 자르고 **꼬리표는 안 자른다** -- 밴드가 화면 밖이어도 «지금 어느 측면인가»는
      //   계속 읽혀야 한다. 그래서 자르는 하위 그룹을 따로 둔다.
      const gClip = document.createElementNS(NS, "g");
      gClip.setAttribute("clip-path", `url(#${clipId})`);
      g.appendChild(gClip);
      const cx = (i) => xAt(i) + bw / 2;
      const band = document.createElementNS(NS, "polygon");
      band.setAttribute("points",
        vp.map((p) => `${cx(p.i).toFixed(1)},${yAt(p.sma + bandK * p.atr).toFixed(1)}`).join(" ") + " " +
        vp.slice().reverse().map((p) => `${cx(p.i).toFixed(1)},${yAt(p.sma - bandK * p.atr).toFixed(1)}`).join(" "));
      band.setAttribute("fill", "color-mix(in srgb, var(--muted) 14%, transparent)");
      band.setAttribute("stroke", "none");
      // 2026-09-22 사용자 «하단 빨강 · 상단 초록». 뜻과 색이 맞는다: 상단 **위**로 벗어나면
      // veto=+1(롱만 허용 = 상승 추세), 하단 **아래**면 -1(숏만 = 하락). 저장소 규약대로
      // 초록=상승·빨강=하락이다. 선은 밴드 면보다 진해야 «경계»로 읽힌다.
      const edge = (key, tone, pct, dash) => {
        const pl = document.createElementNS(NS, "polyline");
        pl.setAttribute("points", vp.map((p) => `${cx(p.i).toFixed(1)},${yAt(key(p)).toFixed(1)}`).join(" "));
        pl.setAttribute("fill", "none");
        pl.setAttribute("stroke", `color-mix(in srgb, var(--${tone}) ${pct || 62}%, transparent)`);
        pl.setAttribute("stroke-width", "1.25");
        pl.setAttribute("stroke-linejoin", "round");
        if (dash) pl.setAttribute("stroke-dasharray", dash);
        return pl;
      };
      gClip.appendChild(band);
      gClip.appendChild(edge((p) => p.sma + bandK * p.atr, "good"));  // 상단: 위로 벗어나면 상승(롱만)
      gClip.appendChild(edge((p) => p.sma - bandK * p.atr, "bad"));   // 하단: 아래로 벗어나면 하락(숏만)
      // ── 참조선: 밴드와 «편익이 시작되는 2xATR» 중 안 겹치는 쪽을 점선으로 ───────────
      // 🔴둘은 다른 물건이다. 밴드(실선)는 «언제 뒤집히나»(히스테리시스)이고, 2xATR 은
      //   «지금 얼마나 믿을 만한가»다. ETH 5m 4.7년 R1(허용−금지):
      //     |z|<0.5 롱 −0.18/숏 −1.15 · 1~1.5 +0.65/+3.90 · 2~3 **+5.92/+11.76**
      //   2026-09-23 밴드를 2xATR 로 올려봤다가 되돌렸다 -- 지그재그 채점에서 추세 일치율·
      //   전환 포착률이 **단조로 나빠졌다**(서버 TREND_VETO_K 주석). 두 선은 계속 분리한다.
      const refK = bandK < 1.8 ? 2 : 1;
      gClip.appendChild(edge((p) => p.sma + refK * p.atr, "good", 30, "2 4"));
      gClip.appendChild(edge((p) => p.sma - refK * p.atr, "bad", 30, "2 4"));
      const side = vp[vp.length - 1].v;
      const col = side > 0 ? "var(--good)" : side < 0 ? "var(--bad)" : "var(--muted)";
      const closed = live ? vp.slice(0, -1) : vp;
      const mkLine = (pts, dashed) => {
        const el = document.createElementNS(NS, "polyline");
        el.setAttribute("points", pts.map((p) => `${cx(p.i).toFixed(1)},${yAt(p.sma).toFixed(1)}`).join(" "));
        el.setAttribute("fill", "none"); el.setAttribute("stroke", col);
        el.setAttribute("stroke-width", "2"); el.setAttribute("stroke-opacity", dashed ? "0.6" : "0.85");
        if (dashed) el.setAttribute("stroke-dasharray", "3 3");
        const t = document.createElementNS(NS, "title");
        t.textContent = "추세 veto -- SMA144 ±1.0×ATR 히스테리시스. " +
          (side > 0 ? "롱만 허용" : side < 0 ? "숏만 허용" : "워밍업") +
          (dashed ? " · 현재 봉은 미확정(종가로 확정된다)" : "") + " (ETH 5m 에서만 검정됨)";
        el.appendChild(t); gClip.appendChild(el);
      };
      if (closed.length >= 2) mkLine(closed, false);
      if (live) mkLine([closed[closed.length - 1], live], true);
      const last = vp[vp.length - 1];
      const tag = document.createElementNS(NS, "text");
      // 꼬리표는 안 자르는 대신 **플롯 안으로 잡아둔다** -- 안 그러면 밴드가 화면 밖일 때
      // 글자만 아래 레인 위에 떠서 같은 문제가 난다.
      tag.setAttribute("x", cx(last.i) - 4);
      tag.setAttribute("y", Math.min(Math.max(yAt(last.sma) - 5, mt + 10), mt + ch - 4));
      tag.setAttribute("text-anchor", "end"); tag.setAttribute("font-size", "9");
      tag.setAttribute("fill", col);
      // 🔴2026-09-26 비평: 풋프린트 셀 숫자 위에 얹혀 읽을 수 없었다 -- 차트 배경색 외곽선으로 아래 셀과 떼어 낸다.
      tag.setAttribute("stroke", "var(--chart-bg)"); tag.setAttribute("stroke-width", "4");
      tag.setAttribute("stroke-linejoin", "round"); tag.setAttribute("paint-order", "stroke");
      // 세기 = |종가 − SMA| / ATR. 위 R1 표의 세 구간과 같은 경계다(<1 / 1~2 / ≥2).
      const lastClose = Number(candles[last.i] && candles[last.i].close);
      const zAbs = last.atr > 0 && Number.isFinite(lastClose)
        ? Math.abs(lastClose - last.sma) / last.atr : NaN;
      const grade = !Number.isFinite(zAbs) ? "" : zAbs >= 2 ? " · 강" : zAbs >= 1 ? " · 보통" : " · 약";
      tag.textContent = (side > 0 ? "추세 veto: 롱만" : side < 0 ? "추세 veto: 숏만" : "추세 veto: 워밍업")
        + (side === 0 ? "" : grade + (Number.isFinite(zAbs) ? ` (${zAbs.toFixed(1)}×ATR)` : ""));
      const tagT = document.createElementNS(NS, "title");
      tagT.textContent = "실선·음영 = ±" + bandK + "×ATR(여기를 벗어날 때만 측면이 바뀐다) · "
        + "점선 = ±" + refK + "×ATR. ETH 5m 4.7년 실측에서 허용 측면의 우위는 2×ATR 부터 나온다 — "
        + "|거리|<0.5×ATR 롱 −0.2 / 숏 −1.2bp, 1~1.5 롱 +0.7 / 숏 +3.9, 2~3 롱 +5.9 / 숏 +11.8bp.\n"
        + "🔴 우위는 대칭이 아니다: 롱 쪽 값은 «위에서 롱이 좋다»가 아니라 «SMA 한참 아래에서 롱 금지»다"
        + "(십분위 실측 — 가장 아래 롱 −7.5bp, 가장 위 롱 −0.1bp).\n"
        + "\n🔴 밴드 폭을 2×ATR 로 넓히면 **추세는 더 못 맞힌다** — 지그재그 채점(θ=2%)에서 "
        + "일치율 .6378→.6213 · 전환 포착률 .8362→.7932 · 전환 지연 150→180분으로 전부 나빠진다. "
        + "좋아지는 건 유령 뒤집힘(.3238→.2525) 하나이고 그게 히스테리시스의 존재 이유다.\n"
        + "🔴 어느 폭에서도 «맞힌다»고 할 수 없다 — 앞 24시간 방향 정확도가 전 구간 0.47~0.48 로 "
        + "무조건부 상승률 0.5056 보다 낮다. 값은 정확도가 아니라 허용/금지 측면의 «차이»에 있다.\n"
        + "🔴 ETH 5m 에서만 검정됐다.";
      tag.appendChild(tagT);
      g.appendChild(tag);
      svg.appendChild(g);
    }
  }

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

  // 2026-09-28 옵션 가격축 표시(사용자 선택 A+C, 기본 켜짐 · 옵션 카드의 «가격축 표시»로 끔): 예상 폭 띠(DVOL 1σ 5분·1시간) ·
  //   max pain(가까운 만기) · 감마 플립. 격자 바로 위(셀 아래)에 깔아 셀 숫자를 덮지 않는다.
  //   🔴행사가 자석·핀닝은 우리 검정에서 없었다 -- 그래서 흐린 점선 + 글자이고 굵은 벽처럼 그리지 않는다.
  const optBx = ml + cw - 3;   // 옵션 괄호·칼시 눈금이 같이 쓰는 x -- 형성 중 봉 오른쪽 끝(플롯 경계)에 걸친다
  let optLab5Y = null;         // 5분 글자 y -- 칼시 글자가 가까우면 한 줄 비켜 선다
  if (isSnapshotChart && footprint) {   // 2026-09-29 상시 표시(사용자 지시 «체크박스 제거»)
    const { o } = optData();
    if (o && optIv(o)) {
      const og = document.createElementNS(NS, "g");
      og.setAttribute("pointer-events", "none");
      const add = (tag, attrs, text) => {
        const e = document.createElementNS(NS, tag);
        Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
        if (text != null) e.textContent = text;
        og.appendChild(e);
      };
      const px = Number(currentPrice) || o.index, fs = mobileChart ? 10 : 11, cy = (y) => Math.max(mt, Math.min(plotBottom, y));
      // 2026-09-28 사용자 «현재가를 따라 움직이면 정보가 안 된다 -- 고정해야»: 띠는 **봉 시가에 고정**한다.
      //   5분 띠 = 이번 5분봉 시가 ± 5분 1σ (다음 봉까지 고정) · 1시간 띠 = 이번 정시(KST) 첫 봉 시가 ± 1시간 1σ.
      //   폭(σ)도 그 봉이 열릴 때 값으로 얼린다(DVOL 이 10분마다 바뀌어도 봉 중간에 띠가 흔들리지 않게).
      //   그래서 «가격이 띠를 벗어났다 = 옵션시장이 예상한 것보다 큰 움직임»으로 읽힌다. 띠는 그 봉(정시 봉)부터 오른쪽만 칠한다.
      const last = candles[candles.length - 1], hourT = Math.floor(last.time / 3600) * 3600;
      const hi = Math.max(0, candles.findIndex((c) => c.time >= hourT));
      const memo = renderCandleSvg._optBand || (renderCandleSvg._optBand = {});
      if (memo.t5 !== last.time || memo.cur !== activeSnapshotAsset) Object.assign(memo, { t5: last.time, s5: optSigma(o, 300), cur: activeSnapshotAsset });
      if (memo.t1 !== hourT || memo.cur1 !== activeSnapshotAsset) Object.assign(memo, { t1: hourT, s1h: optSigma(o, 3600), cur1: activeSnapshotAsset });
      const s5 = memo.s5, s1h = memo.s1h, o5 = Number(last.open) || px, o1 = Number(candles[hi].open) || px;
      // 2026-09-28 사용자 선택(옵션 시안 B의 괄호): 가격판 전폭 띠 두 장 -> **형성 중 봉 바로 오른쪽 괄호 하나**.
      //   바깥 옅은 세로줄+꺾쇠 = 1시간(정시 봉 시가 ± 1σ) · 안쪽 진한 막대 = 5분(이번 봉 시가 ± 1σ). 고정 규칙은 위와 같다.
      //   괄호는 플롯 오른쪽 끝에 걸치고 글자는 바깥쪽(데스크톱 오른쪽 · 모바일은 호가 띠라 안쪽 왼쪽)에 바탕색 외곽선으로.
      const bx = optBx, a1r = yAt(o1 + s1h), b1r = yAt(o1 - s1h), a1 = cy(a1r), b1 = cy(b1r), a5 = cy(yAt(o5 + s5)), b5 = cy(yAt(o5 - s5));
      const side = mobileChart ? -1 : 1, lx = bx + side * 8, anchor = mobileChart ? "end" : "start";
      const halo = { stroke: "var(--chart-bg)", "stroke-width": 3, "paint-order": "stroke" };
      add("line", { x1: bx, x2: bx, y1: a1, y2: b1, stroke: "var(--option)", "stroke-opacity": 0.55, "stroke-width": 1.5 });
      [[a1r, a1], [b1r, b1]].forEach(([r, y]) => { if (r >= mt && r <= plotBottom) add("line", { x1: bx - 5, x2: bx + 5, y1: y, y2: y, stroke: "var(--option)", "stroke-opacity": 0.55, "stroke-width": 1.5 }); });
      add("rect", { x: bx - 3, y: a5, width: 6, height: Math.max(1, b5 - a5), rx: 2, fill: "var(--option)" });
      optLab5Y = a5 + 4;
      if (gutOn) {   // 2026-09-30 꼬리표 칸으로(괄호 옆 글자가 꼬리표와 겹친다)
        fpNotes.push({ val: o5 + s5, label: `5분 ±${optQ(s5)}`, color: "var(--option)", noPrice: true });
        fpNotes.push({ val: o1 + s1h, label: `1시간 ±${optQ(s1h)}`, color: "var(--option)", noPrice: true });
      } else add("text", { x: lx, y: a5 + 4, "font-size": fs, "font-weight": 700, fill: "var(--option)", "text-anchor": anchor, ...halo }, `5분 ±${optQ(s5)}`);
      if (!gutOn && a1 < a5 - fs - 2) add("text", { x: lx, y: a1 + (a1r < mt ? fs : 4), "font-size": fs, "font-weight": 600, fill: "var(--option)", "text-anchor": anchor, ...halo },
          `1시간 ±${optQ(s1h)}${a1r < mt ? "↑" : ""}`);
      const f = optFront(o);
      if (f) {
        const y = yAt(f.pain), lab = `max pain ${optQ(f.pain)} · ${optKst(f.exp_ms)} KST 만기${f.oi_share == null ? "" : `(전체 미결제의 ${Math.round(f.oi_share * 100)}%)`}`;
        if (y >= mt && y <= plotBottom) {
          add("line", { x1: ml, x2: ml + cw, y1: y, y2: y, stroke: "var(--warn)", "stroke-opacity": 0.6, "stroke-dasharray": "2 5" });
          if (gutOn) fpNotes.push({ val: f.pain, label: "max pain", color: "var(--warn)", title: lab });
          else add("text", { x: ml + cw - 4, y: y - 4, "font-size": fs, "font-weight": 700, fill: "var(--warn)", "text-anchor": "end" }, mobileChart ? `max pain ${optQ(f.pain)}` : lab);   // 2026-10-01 휴대폰은 짧게(긴 글이 화면 밖으로 잘렸다)
        } else if (gutOn) {
          fpNotes.push({ val: f.pain, label: "max pain", color: "var(--warn)", title: lab });
        } else {
          add("text", { x: ml + 4, y: y < mt ? mt + 2 * fs + 8 : plotBottom - fs - 8, "font-size": fs, "font-weight": 700, fill: "var(--warn)" },
              `${mobileChart ? `max pain ${optQ(f.pain)}` : lab} ${y < mt ? "↑" : "↓"} (${f.pain >= px ? "+" : "−"}${optQ(Math.abs(f.pain - px))}$)`);
        }
      }
      // 2026-09-30 행사가별 딜러 감마 = «GEX 레벨»(사용자 참고 사진 -- 꼬리표 칸의 작은 상자를 대체): 최근 봉부터 오른쪽 끝까지 가로 점선 +
      //   선 왼쪽 끝 위에 «GEX + ▼▼» 글자. 청록 = +(양감마) · 주황 = −. 화살표(사용자 선택 a): +GEX 는 지금가 쪽(헤지가 움직임을 받치는 성격),
      //   −GEX 는 지금가 반대쪽(돌파하면 헤지가 움직임을 따라가는 성격) · 개수 = 크기(범위 최대의 절반 이상이면 둘).
      //   값은 r4(딜러·체결 순감마)만(옵션 세션 합의 -- 가정 몫 안 붙임). r5 커버 99% 미만·모름이면 옅게(«일부만 봄»). 범위는 사다리 칩, 많으면 크기 상위 6개.
      const sAll = (o.strikes || {})[optLadderScope] || [], sRolled = optLadderScope === "front" && (o.strikes || {}).front_exp_ms <= Date.now();
      const sVis = sRolled ? [] : sAll.filter((r) => Number.isFinite(r[4]) && r[4] !== 0 && yAt(r[0]) >= mt && yAt(r[0]) <= plotBottom)
        .sort((p, q) => Math.abs(q[4]) - Math.abs(p[4])).slice(0, 6);
      if (sVis.length) {
        const lineG = document.createElementNS(NS, "g"), labG = document.createElementNS(NS, "g");
        const gmax = Math.max(1, ...sAll.map((r) => (Number.isFinite(r[4]) ? Math.abs(r[4]) : 0)));
        const x0 = xAt(Math.max(0, candles.length - 10)), x1 = ml + cw, pxNow = Number(currentPrice) || o.index;
        const mkEl = (parent, tag, attrs, text) => { const e = document.createElementNS(NS, tag); Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v)); if (text != null) e.textContent = text; parent.appendChild(e); return e; };
        sVis.forEach(([k, , , , g, cov]) => {
          const y = yAt(k).toFixed(1), full = cov != null && cov >= 0.99, col = g >= 0 ? "var(--option)" : "var(--warn)", op = full ? 1 : 0.75;
          const toward = k >= pxNow ? "▼" : "▲", away = k >= pxNow ? "▲" : "▼";
          const arrow = (g >= 0 ? toward : away).repeat(Math.abs(g) >= gmax * 0.5 ? 2 : 1);
          mkEl(lineG, "line", { x1: x0, x2: x1, y1: y, y2: y, stroke: col, "stroke-opacity": op, "stroke-width": 1.8, "stroke-dasharray": "7 4" });
          const t = mkEl(labG, "text", { x: x0 + 3, y: (+y - 4).toFixed(1), "font-size": 11, "font-weight": 700, fill: col, "fill-opacity": op,
                                         stroke: "var(--chart-bg)", "stroke-width": 3, "paint-order": "stroke" }, `GEX ${g >= 0 ? "+" : "−"} ${arrow}`);
          mkEl(t, "title", {}, `행사가 ${optQ(k)} · 딜러·체결 감마 ${g >= 0 ? "+" : "−"}${optUsd(Math.abs(g))}/1% · 이 행사가 커버 ${cov == null ? "모름" : optCovPct(cov)}\n`
            + "딜러 = 체결 반대편(메이커)으로 본 순포지션. 커버 밖 미결제(수집 전 상장 종목)는 누가 들었는지 몰라 뺐다(옅은 선 = 일부만 봄).\n"
            + (g >= 0 ? "+GEX: 화살표 = 지금가 쪽 — 딜러 헤지가 움직임을 반대로 받치는 성격(교과서 설명).\n"
                      : "−GEX: 화살표 = 지금가 반대쪽 — 돌파하면 딜러 헤지가 움직임을 따라가 커지는 성격(교과서 설명).\n")
            + "개수 = 크기(범위 최대의 절반 이상이면 둘).\n"
            + "우리 검정: 행사가 자석(가격이 미결제 큰 행사가로 끌림)과 GEX 크기로 변동폭 예측은 불합격 — 지지·저항이 아니라 딜러 헤지가 쌓인 자리의 참고.");
        });
        labG.setAttribute("pointer-events", "visiblePainted");
        const gridS = layerCache.get("grid")?.g;
        if (gridS && gridS.parentNode === svg) svg.insertBefore(lineG, gridS.nextSibling); else svg.appendChild(lineG);   // 선은 셀 뒤
        svg.appendChild(labG);                                                                                           // 글자는 위
      }
      const gm = optDealer(o), fl = gm.flip;   // 2026-09-29 딜러·체결(커버 종목 기준)
      if (gm.now_usd == null) {
        // 체결 기반 값이 아직 없다(커버 종목 없음 · 수집기 재시작 전) -- 플립 글자를 안 쓴다
      } else if (fl && yAt(fl) >= mt && yAt(fl) <= plotBottom) {
        add("line", { x1: ml, x2: ml + cw, y1: yAt(fl), y2: yAt(fl), stroke: "var(--muted)", "stroke-opacity": 0.8, "stroke-dasharray": "8 4" });
        if (gutOn) fpNotes.push({ val: fl, label: "감마 플립", color: "var(--warn)", title: `감마 플립(체결) ${optQ(fl)}` });
        else add("text", { x: ml + 4, y: yAt(fl) - 4, "font-size": fs, "font-weight": 700, fill: "var(--muted)" }, `감마 플립(체결) ${optQ(fl)}`);
      } else {
        add("text", { x: ml + 4, y: plotBottom - 4, "font-size": fs, "font-weight": 600, fill: "var(--muted)" },
            fl ? `감마 플립(체결) ${optQ(fl)} ${fl < px ? "↓" : "↑"}(창 밖)`
              : mobileChart ? `플립 없음 · ${gm.now_usd < 0 ? "음감마" : "양감마"}`
              : `감마 플립 없음(±15%, 체결·커버 종목 기준) — ${gm.now_usd < 0 ? "음감마, 딜러가 키우는 쪽" : "양감마, 딜러가 눌러 주는 쪽"}`);
      }
      const gridG = layerCache.get("grid")?.g;
      if (gridG && gridG.parentNode === svg) svg.insertBefore(og, gridG.nextSibling); else svg.appendChild(og);
    }
  }

  // 2026-09-30 30분 도달 선 제거(사용자 지시 -- «±0.5×30분 폭에 닿는가»는 옵션 예상 폭 띠와 같은 크기 정보라 중복).

  // 2026-09-28 칼시 15분: 가격판 전폭 선·창 음영 -> **옵션 괄호를 가로지르는 짧은 눈금 하나**(사용자 선택, 옵션 시안 A의 칼시 선).
  //   글자 «칼시 아래 72% · 11:00»는 괄호 글자와 같은 쪽에. 참고 · 신호 아님(확률은 대부분 «지금가 vs 기준가 + 남은 시간»의 되비침).
  const kn = isSnapshotChart && footprint ? kalshiNow() : null;
  if (kn) {
    const fs = mobileChart ? 10 : 11, yr = yAt(kn.k.strike), ys = Math.max(mt + 2, Math.min(plotBottom - 2, yr));
    const left = Math.max(0, Math.round(kn.k.close_ts - Date.now() / 1000));
    const tick = document.createElementNS(NS, "line");
    Object.entries({ x1: optBx - 12, x2: optBx + 12, y1: ys, y2: ys, stroke: kn.color, "stroke-width": 2.2, "stroke-linecap": "round",
                     "pointer-events": "none" }).forEach(([k, v]) => tick.setAttribute(k, v));
    const t = document.createElementNS(NS, "text");
    // 글자는 옵션 괄호 글자와 같은 쪽(데스크톱 = 괄호 바깥 오른쪽, 셀을 안 덮는다 · 모바일 = 오른쪽이 호가 띠라 안쪽 왼쪽).
    const ty = optLab5Y != null && Math.abs(ys + 4 - optLab5Y) < fs + 3 ? optLab5Y + fs + 3 : ys + 4;
    Object.entries({ x: optBx + (mobileChart ? -16 : 16), y: ty, "font-size": fs, "font-weight": 700, fill: kn.color,
                     "text-anchor": mobileChart ? "end" : "start", "pointer-events": "none",
                     stroke: "var(--chart-bg)", "stroke-width": 3, "paint-order": "stroke", style: "font-variant-numeric: tabular-nums" })
      .forEach(([k, v]) => t.setAttribute(k, v));
    t.textContent = `칼시 ${kn.word} · ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}${yr < mt ? " ↑" : yr > plotBottom ? " ↓" : ""}`;
    svg.appendChild(tick);
    svg.appendChild(t);
  }

  const lblStay = (q) => !!q.note && /^(5분|1시간) ±|^VWAP/.test(q.label || "");   // 2026-10-01 플롯 오른쪽 끝에 남는 이름(옵션 예상 폭 둘 · VWAP — 사용자 지시)
  // 2026-09-30 꼬리표 칸: 겹침 선 이름(fpNotes)을 가격 꼬리표와 한 줄에 세워 겹침 회피를 **함께** 다시 돈다.
  if (gutOn && fpNotes.length) {
    fpNotes.forEach((n) => {
      const rawY = yAt(n.val);
      Object.assign(n, { note: true, rawY, offTop: rawY < mt, offBottom: rawY > plotBottom });
      n.outOfView = n.offTop || n.offBottom;
      n.realY = n.outOfView ? (n.offTop ? mt + 2 : plotBottom - 2) : rawY;
      priceLabels.push(n);
    });
    priceLabels.forEach((p) => { delete p.adjustedY; });
    // 2026-10-01 두 줄로 나눠 겹침을 따로 돈다: 5분·1시간 폭 = 플롯 안 오른쪽 끝(그대로) · 나머지 = 호가 프로파일 오른쪽 끝(사용자 지시)
    declutterTagY(priceLabels.filter((q) => lblStay(q)), mt + 9, plotBottom - 9, 19);
    declutterTagY(priceLabels.filter((q) => !lblStay(q)), mt + 9, plotBottom - 9, 19);
  }
  const tagX = ml + cw + 4, xR = ml + cw - 4, inR = LBL !== "B";
  const xRR = w - mr - 2;   // 호가 프로파일 오른쪽 끝 -- 현재가·지지/저항·감마 플립·지표·max pain 이름이 여기 오른쪽 정렬
  const xOf = (q) => (lblStay(q) ? xR : xRR);   // B = 체결 기둥 왼쪽 위에 띄움 · A/C = 플롯 안 오른쪽 끝에 오른쪽 정렬
  // 2026-09-30 약어 → 짧은 이름(사용자 «약어가 너무 많아 이해가 힘들다»): 이름은 그대로, 가격만 호버로 뺀다.
  const CODE = (nm) => nm.replace("앵커·전일고", "앵커 전일고").replace("앵커·전일저", "앵커 전일저")
    .replace(/^5분 ±.*/, "5분 폭").replace(/^1시간 ±.*/, "1시간 폭").replace(/^HL (롱|숏).*/, "HL $1청산");
  priceLabels.forEach(p => {
    const labelYRaw = p.adjustedY !== undefined ? p.adjustedY : p.realY;
    if (p.note) {   // 겹침 선 이름 -- 선은 제자리에 이미 있다. 칸 안 글자 + 지시선만.
      const ly = Math.max(mt + 9, Math.min(plotBottom - 9, labelYRaw));
      const ng = document.createElementNS(NS, "g");
      const lead = document.createElementNS(NS, "path");
      const far = inR && !lblStay(p);   // 프로파일 끝으로 간 이름 -- 선(플롯 안)에서 떨어지므로 글자 앞까지 옅은 점선을 잇는다(아래에서 폭을 잰 뒤)
      if (!far && (!inR || Math.abs(ly - p.realY) > 2)) {
        lead.setAttribute("d", inR ? `M${ml + cw} ${p.realY.toFixed(1)} L${xR} ${ly.toFixed(1)}` : `M${ml + cw} ${p.realY.toFixed(1)} L${tagX - 2} ${ly.toFixed(1)}`);
        lead.setAttribute("stroke", p.color); lead.setAttribute("stroke-opacity", "0.45"); lead.setAttribute("fill", "none");
        ng.appendChild(lead);
      }
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", inR ? xOf(p) - 2 : tagX + 2); t.setAttribute("y", ly + 3.5); t.setAttribute("font-size", LBL === "C" ? "10.5" : "10"); t.setAttribute("font-weight", "700");
      if (inR) t.setAttribute("text-anchor", "end");
      t.setAttribute("fill", p.color); t.setAttribute("style", "font-variant-numeric: tabular-nums");
      t.setAttribute("stroke", "var(--chart-bg)"); t.setAttribute("stroke-width", "3"); t.setAttribute("paint-order", "stroke");
      const full = `${p.label}${p.noPrice ? "" : " " + fmtNum(p.val, Math.max(0, pxDp() - 1))}${p.offTop ? " ↑" : p.offBottom ? " ↓" : ""}`;
      t.textContent = LBL === "C" ? `${CODE(p.label)}${p.offTop ? "↑" : p.offBottom ? "↓" : ""}` : full;
      { const ti = document.createElementNS(NS, "title"); ti.textContent = p.title ? `${full}\n${p.title}` : full; t.appendChild(ti); }
      ng.appendChild(t);
      svg.appendChild(ng);
      if (far) t.classList.add("far-lbl");
      if (far) {
        let tw = 0;
        try { tw = t.getComputedTextLength(); } catch (e) { /* 비렌더 */ }
        const x2 = xRR - 2 - (tw > 0 ? tw : 60) - 4;
        lead.setAttribute("d", `M${ml + cw} ${p.realY.toFixed(1)} L${Math.max(ml + cw, x2 - 10).toFixed(1)} ${p.realY.toFixed(1)} L${x2.toFixed(1)} ${ly.toFixed(1)}`);
        lead.setAttribute("stroke", p.color); lead.setAttribute("stroke-opacity", "0.35"); lead.setAttribute("stroke-dasharray", "2 3"); lead.setAttribute("fill", "none");
        ng.insertBefore(lead, t);
      }
      return;
    }
    const labelY = Math.max(mt + 9, Math.min(plotBottom - 9, labelYRaw));
    const lineDashed = p.dashed || p.outOfView;
    // 2026-09-22 사용자 지시: **모바일은 배지를 안 그린다.** 값은 플롯 아래 한 줄이 갖고,
    //   플롯 안에서는 꺾쇠(화살표)가 «어느 행인가»만 가리킨다. 배지를 안에 띄우면 가장 최근
    //   봉을 덮고, 밖에 두면 그 폭만큼 플롯이 짧아진다 -- 아래로 내리면 둘 다 없다.
    // 2026-09-27 풋프린트(데스크톱)는 **모든** 가격을 왼쪽 글자로 -- 오른쪽은 호가 띠 자리다(사용자 지시).
    const priceLeft = (p.priceLeft || !!footprint) && !mobileChart;   // 모바일은 원래 배지가 없다(가격은 아래 줄)
    const subOk = !!p.sub && !mobileChart && !priceLeft;   // 왼쪽 여백엔 값(HL 수량)까지 못 넣는다
    const boxW = subOk ? 76 : 64, boxH = 18;
    const boxX = w - mr + 4;

    // Line stays at real (clamped) price position
    let line = null;
    if (p.marker) {
      // 플롯 오른쪽 가장자리에서 **왼쪽을 가리키는** 삼각형. 꼭짓점이 곧 그 가격의 행이다.
      const tri = document.createElementNS(NS, "polygon");
      tri.setAttribute("points", markerPoints(ml + cw, p.realY));
      if (p.gauge === undefined) {
        tri.setAttribute("fill", p.color);
      } else {                                   // 예측 = 속 빈 꺾쇠
        tri.setAttribute("fill", "none");
        tri.setAttribute("stroke", p.color);
        tri.setAttribute("stroke-width", "1.2");
        tri.setAttribute("stroke-linejoin", "round");
      }
      if (p.faded) tri.setAttribute("opacity", "0.5");         // 지나간 목표
      else if (p.outOfView) tri.setAttribute("opacity", "0.72");
      // ⚠️여기서 append 하지 않는다. 플롯 오른쪽 끝(ml+cw)과 가격 배지(w-mr+4)가 4px 차이라
      //   먼저 그리면 배지에 **가려진다**(2026-09-16 첫 판이 그래서 안 보였다). 배지 뒤에
      //   붙여 배지의 «꼬리»처럼 보이게 한다 -- 꼭짓점은 여전히 진짜 가격 행을 가리킨다
      //   (배지 자체는 겹침 회피로 위아래로 밀릴 수 있어서 행을 정확히 못 가리킨다).
      line = tri;   // 아래 data-live 표식과 빠른 갱신이 같은 변수를 쓴다
    } else {
      line = document.createElementNS(NS, "line");
      line.setAttribute("x1", ml - TRADE_L); line.setAttribute("x2", w - mr);
      line.setAttribute("y1", p.realY); line.setAttribute("y2", p.realY);
      line.setAttribute("stroke", p.color);
      line.setAttribute("stroke-width", String(p.width || 2));
      if (lineDashed) line.setAttribute("stroke-dasharray", "4,4");
      if (p.outOfView) line.setAttribute("opacity", "0.72");
      svg.appendChild(line);
      if (p.behindCells) {                       // 셀 숫자를 덮지 않게 격자 바로 위(셀 아래)로
        line.setAttribute("opacity", "0.85");
        const gridG = layerCache.get("grid")?.g;
        if (gridG && gridG.parentNode === svg) svg.insertBefore(line, gridG.nextSibling);
      }
    }

    // Left label (follows label position)
    // 2026-09-22 label 이 비면 안 그린다 -- 시나리오는 확률을 **오른쪽 배지 안**으로 옮겼다
    // (사용자 «오른쪽 라벨에 가격이랑 확률만»). 왼쪽 여백은 45px 뿐이라 둘을 다 못 넣는다.
    // 이름이 없는 것(30분 시나리오 목표 «↑56%»)은 왼쪽으로 옮길 때 sub 를 이름으로 쓴다.
    const leftName = mobileChart && footprint ? "" : p.label || (priceLeft && p.sub ? p.sub : "");   // 모바일 풋프린트는 아래 한 줄만
    const txt = leftName ? document.createElementNS(NS, "text") : null;
    // 2026-09-28 풋프린트 현재가는 **채운 태그**(사용자 «현재가 라벨을 더 크게») -- 13px(모바일 12) 어두운 글자, 왼쪽 끝에서
    //   시작해 필요하면 플롯 안으로 조금 들어간다(가격 태그 관례). 빠른 갱신(updateLivePriceFast)이 글자·폭·높이를 같이 옮긴다.
    const nowTag = p.label === "현재" && !!footprint && !!txt;
    const gutTag = gutOn && !!txt;   // 2026-09-30 꼬리표 칸: 모든 가격 이름이 상자(현재 = 채움 · 나머지 = 테두리)
    const tagBg = (nowTag || gutTag) && !(nowTag && gutTag) ? document.createElementNS(NS, "rect") : null;   // 풋프린트 현재가는 상자 없이 숫자만
    if (gutTag && (!inR || Math.abs(labelY - p.realY) > 2)) {
      const lead = document.createElementNS(NS, "path");
      // 2026-10-01 가격 이름은 프로파일 오른쪽 끝 -- 밀렸으면 그 끝에서 세로로 짧게 잇는다(선은 이미 전폭이다)
      lead.setAttribute("d", inR ? `M${xRR + 1} ${p.realY.toFixed(1)} L${xRR + 1} ${labelY.toFixed(1)}` : `M${ml + cw} ${p.realY.toFixed(1)} L${tagX - 2} ${labelY.toFixed(1)}`);
      lead.setAttribute("stroke", p.color); lead.setAttribute("stroke-opacity", "0.6"); lead.setAttribute("fill", "none");
      svg.appendChild(lead);
    }
    if (tagBg) {
      tagBg.setAttribute("x", gutTag ? tagX : 2); tagBg.setAttribute("y", labelY - 9); tagBg.setAttribute("height", 18);
      tagBg.setAttribute("rx", 3);
      if (nowTag) tagBg.setAttribute("fill", p.color);
      else { tagBg.setAttribute("fill", "var(--chart-bg)"); tagBg.setAttribute("stroke", p.color); tagBg.setAttribute("stroke-width", "1"); }
      svg.appendChild(tagBg);
    }
    if (txt) {
      txt.setAttribute("x", gutTag ? (inR ? xRR - 5 : tagX + 5) : nowTag ? 7 : ml - TRADE_L - 5); txt.setAttribute("y", labelY + (nowTag ? 4.5 : 4));
      txt.setAttribute("text-anchor", gutTag ? (inR ? "end" : "start") : nowTag ? "start" : "end");
      txt.setAttribute("font-size", nowTag ? (mobileChart ? "12" : "13") : gutTag ? "10.5" : "10");   // 2026-10-01 차트 글씨 상한 10.5(DESIGN 차트 예외)
      txt.setAttribute("font-weight", "bold"); txt.setAttribute("fill", nowTag && !gutTag ? inkOnFill() : p.color);
      if (nowTag && gutTag) {   // 숫자만: 14px · 굵게 · 바탕색 외곽선으로 셀 위에서도 읽힌다
        txt.setAttribute("font-size", "14"); txt.setAttribute("font-weight", "800");
        txt.setAttribute("stroke", "var(--chart-bg)"); txt.setAttribute("stroke-width", "4"); txt.setAttribute("paint-order", "stroke");
        txt.setAttribute("style", "font-variant-numeric: tabular-nums");
      }
      txt.textContent = `${leftName}${p.offTop ? "↑" : p.offBottom ? "↓" : ""}`;
      if (gutTag && inR) { txt.classList.add("far-lbl"); if (!nowTag) txt.setAttribute("style", "font-variant-numeric: tabular-nums"); if (tagBg) tagBg.classList.add("far-lbl"); }
      svg.appendChild(txt);
      if (priceLeft && nowTag && gutTag) {
        txt.textContent = fmtNum(p.val, pxDp());
      } else if (priceLeft && nowTag) {
        txt.textContent = `${txt.textContent} ${fmtNum(p.val, pxDp())}`;
      } else if (priceLeft && gutTag && LBL === "C") {
        const ti = document.createElementNS(NS, "title"); ti.textContent = `${txt.textContent} ${fmtNum(p.val, pxDp())}`;
        txt.textContent = CODE(txt.textContent); txt.appendChild(ti);
      } else if (priceLeft) {
        // 왼쪽 여백(ml-5)을 넘으면 SVG 밖으로 잘린다 -- 재서 소수점을 뗀다.
        const name = txt.textContent;
        txt.textContent = `${name} ${fmtNum(p.val, pxDp())}`;
        try { if (txt.getComputedTextLength() > (gutTag ? 84 : ml - TRADE_L - 7)) txt.textContent = `${name} ${fmtNum(p.val, Math.max(0, pxDp() - 1))}`; } catch (e) { /* 비렌더 */ }
      }
      if (tagBg) {
        let tw = 0;
        try { tw = txt.getComputedTextLength(); } catch (e) { /* 비렌더 */ }
        const bw0 = (tw > 0 ? tw : txt.textContent.length * 7.5) + 10;
        tagBg.setAttribute("width", bw0);
        if (gutTag && inR) { tagBg.setAttribute("x", xRR - bw0); tagBg.dataset.right = String(xRR); }   // 오른쪽 정렬(빠른 갱신도 이 오른쪽 끝을 지킨다)
      }
    }

    // Right box (follows label position). p.sub 가 있으면 배지 안에 «가격 + 값»을 같이 넣는다.
    // 🔴폭에 한계가 있다. 배지는 x = w-mr+4 에서 시작하므로 mr-8 을 넘으면 SVG 밖으로 잘린다
    //   (데스크톱 mr 86 -> 최대 78, 모바일 mr 68 -> 최대 60 = 기존 56 에서 4px 뿐).
    //   ⇒ 모바일에서는 값을 넣지 않는다. 카드에 같은 숫자가 있고, 잘린 배지보다 낫다.
    const rect = document.createElementNS(NS, "rect");
    rect.setAttribute("x", boxX); rect.setAttribute("y", labelY - 9);
    rect.setAttribute("width", boxW); rect.setAttribute("height", boxH);
    rect.setAttribute("rx", "2");
    if (p.gauge === undefined) {
      rect.setAttribute("fill", p.color);
    } else {
      rect.setAttribute("fill", "none");
      rect.setAttribute("stroke", p.color);
      rect.setAttribute("stroke-width", "1.2");
      if (p.gauge > 0 && !mobileChart && !priceLeft) {   // 테두리 «안»을 확률만큼 채운다(배지를 왼쪽 글자로 옮기면 안 그린다)
        const fillW = Math.max(2, (boxW - 2) * (p.gauge / 100));
        const g = document.createElementNS(NS, "rect");
        g.setAttribute("x", boxX + 1); g.setAttribute("y", labelY - 8);
        g.setAttribute("width", fillW); g.setAttribute("height", boxH - 2);
        g.setAttribute("fill", p.color); g.setAttribute("opacity", "0.32");
        g.setAttribute("rx", "1.5");
        svg.appendChild(g);
      }
    }
    if (p.faded) rect.setAttribute("opacity", "0.5");
    if (!mobileChart && !priceLeft) svg.appendChild(rect);

    const pTxt = document.createElementNS(NS, "text");
    pTxt.setAttribute("x", boxX + 4); pTxt.setAttribute("y", labelY + 4);
    // 값이 같이 들어가면 가격을 한 단계 줄인다 -- 76px 안에 «2700.0»(11px, 40) + «56%»(9.5px, 17)
    pTxt.setAttribute("font-size", subOk ? "11" : (mobileChart ? "11" : "12"));
    pTxt.setAttribute("font-weight", "bold");
    pTxt.setAttribute("fill", p.gauge === undefined ? inkOnFill() : "var(--ink)");
    if (p.faded) pTxt.setAttribute("opacity", "0.62");
    // 🔴화면 밖이면 «↑ » 가 앞에 붙어 값(sub)과 겹쳤다(2026-09-24 캡처: «2780.Q.1k»·«2641.63%»).
    //   78px 한계라 폭을 못 늘린다 -- 값이 같이 들어가는 화면 밖 배지만 소수점을 뗀다.
    pTxt.textContent = `${p.offTop ? "↑ " : p.offBottom ? "↓ " : ""}${fmtNum(p.val, subOk && p.outOfView ? Math.max(0, pxDp() - 1) : pxDp())}`;
    if (!mobileChart && !priceLeft) svg.appendChild(pTxt);
    if (subOk) {
      const sTxt = document.createElementNS(NS, "text");
      sTxt.setAttribute("x", boxX + boxW - 5); sTxt.setAttribute("y", labelY + 4);
      sTxt.setAttribute("text-anchor", "end"); sTxt.setAttribute("font-size", "9.5");
      sTxt.setAttribute("font-weight", "700");
      sTxt.setAttribute("fill", p.gauge === undefined ? inkOnFill() : "var(--muted)");
      if (p.gauge === undefined) sTxt.setAttribute("opacity", ".72");
      sTxt.textContent = p.sub;
      svg.appendChild(sTxt);
      // 🔴2026-09-26 비평: 가격과 값이 한 상자 안에서 겹쳤다(«↓ 2535 3.0k»·«2687.7↑25%»). 상자 폭은 늘릴 수 없으니
      //   글자 폭을 재서 안 들어가면 값을 뺀다 -- 이 꼬리표의 주인은 가격이다.
      try {
        if (pTxt.isConnected && pTxt.getComputedTextLength() + sTxt.getComputedTextLength() + 6 > boxW - 9) sTxt.remove();
      } catch (e) { /* 측정 불가(비렌더) -- 그대로 둔다 */ }
    }
    if (p.marker) svg.appendChild(line);   // 배지 위에 -- 위 주석 참조

    // 현재가 줄만 표식을 단다 -- 틱마다 **이 세 요소만** 옮기려는 것이다(전체 재렌더는 1초).
    // 표식이 없으면 빠른 갱신이 어느 줄을 움직여야 하는지 알 수 없다.
    if (p.label === "현재" && isSnapshotChart) {
      line.dataset.live = p.marker ? "tri" : "line";
      rect.dataset.live = "box";
      pTxt.dataset.live = "text";
      if (txt) txt.dataset.live = "label";
      if (tagBg) tagBg.dataset.live = "labelbg";
      if (priceLeft && txt) txt.dataset.withPrice = gutOn ? "num" : "1";   // 빠른 갱신이 글자 속 가격도 바꾼다(num = 숫자만)
    }
  });
  // 2026-09-22 사용자 지시: 모바일은 배지 대신 플롯 **아래 한 줄**이 값을 갖는다.
  //   순서는 priceLabels 그대로다 -- y 오름차순 = 가격 내림차순이라 줄이 위에서 아래로
  //   읽히는 순서와 같다. 폭을 넘으면 거기서 멈춘다(잘린 글자를 남기지 않는다).
  if (mobileChart && PRICE_ROW_H && priceLabels.length) {
    const rowY = plotBottom + 13;
    let x = 3;
    // 🔴**이름 있는 것만** 싣는다(현재·진입·지지n·저항n). 청산맵이 얹는 익명 레벨까지 넣으면
    //   줄이 넘쳐 넘침 가드가 뒤쪽을 자르는데, 하필 그 뒤쪽이 «지지1» 처럼 이름 있는 것이었다
    //   (실측). 익명 레벨은 플롯 안의 선·꺾쇠가 이미 «어느 행인가»를 말한다.
    priceLabels.filter((p) => p.label).forEach((p) => {
      if (x > w - 30) return;
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", x); t.setAttribute("y", rowY);
      t.setAttribute("font-size", "10.5"); t.setAttribute("fill", p.color);
      if (p.faded) t.setAttribute("opacity", "0.55");
      const arrow = p.offTop ? "↑" : p.offBottom ? "↓" : "";
      t.textContent = (p.label || "") + " " + arrow + fmtNum(p.val, pxDp());
      if (p.label === "현재" && isSnapshotChart) t.dataset.live = "rowtext";
      svg.appendChild(t);
      let adv = 0;
      try { adv = t.getComputedTextLength(); } catch (_) { adv = t.textContent.length * 6.4; }
      if (!(adv > 0)) adv = t.textContent.length * 6.4;
      // 자릿수는 데이터가 정한다 -- 상수로 못 막으므로 넘치면 그 항목을 도로 뺀다.
      if (x + adv > w - 3) { t.remove(); x = w; return; }
      x += adv + 9;
    });
  }
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
  hLine.setAttribute("x1", ml - TRADE_L); hLine.setAttribute("x2", w - mr);
  hLine.setAttribute("y1", 0); hLine.setAttribute("y2", 0);
  hLine.setAttribute("stroke", "var(--hover-line)");
  hLine.setAttribute("stroke-dasharray", "4,4");
  hLine.style.display = "none";
  hLine.style.pointerEvents = "none";
  hoverGroup.appendChild(hLine);

  const priceBadgeW = mobileChart ? 56 : 64, priceBadgeH = 18;
  // 위 가격 배지와 **같은 기준**으로 잡는다 -- 둘이 어긋나면 호버 배지만 다른 열에 뜬다.
  // 2026-09-27 풋프린트는 오른쪽이 호가 띠라 호버 배지도 왼쪽 여백으로.
  const priceBadgeX = mobileChart ? w - 4 - priceBadgeW : footprint ? Math.max(2, ml - priceBadgeW - 3) : w - mr + 4;
  const priceBadgeRect = document.createElementNS(NS, "rect");
  priceBadgeRect.setAttribute("x", priceBadgeX);
  priceBadgeRect.setAttribute("width", priceBadgeW);
  priceBadgeRect.setAttribute("height", priceBadgeH);
  priceBadgeRect.setAttribute("fill", "var(--accent)");
  priceBadgeRect.setAttribute("rx", "2");
  priceBadgeRect.style.display = "none";
  priceBadgeRect.style.pointerEvents = "none";
  hoverGroup.appendChild(priceBadgeRect);

  const priceBadgeText = document.createElementNS(NS, "text");
  priceBadgeText.setAttribute("x", priceBadgeX + 4);
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
    const svgScale = Math.min(rect.width / wAll, rect.height / hAll);   // viewBox 는 wAll(2단이면 오른쪽 칸 포함)
    const svgOffsetX = (rect.width - wAll * svgScale) / 2;
    const svgOffsetY = (rect.height - hAll * svgScale) / 2;
    const mx = (evt.clientX - rect.left - svgOffsetX) / svgScale;
    const my = (evt.clientY - rect.top - svgOffsetY) / svgScale;

    if (my >= mt && my <= plotBottom) {
      hLine.setAttribute("y1", my); hLine.setAttribute("y2", my);
      hLine.style.display = "block";
      const priceAtCursor = yMax - ((my - mt) * ySpan) / ch;
      const badgeY = Math.max(mt, Math.min(plotBottom - priceBadgeH, my - priceBadgeH / 2));
      priceBadgeRect.setAttribute("y", badgeY);
      priceBadgeText.setAttribute("y", badgeY + 13);
      priceBadgeText.textContent = fmtNum(priceAtCursor, pxDp());
      priceBadgeRect.style.display = "block";
      priceBadgeText.style.display = "block";
    } else {
      hLine.style.display = "none";
      priceBadgeRect.style.display = "none";
      priceBadgeText.style.display = "none";
    }

    // 2026-09-28 네 숫자(기둥 머리 · 모바일은 풋프린트 위 왼쪽) 위면 그 설명.
    const sm = latestFlowHeatmap && latestFlowHeatmap.summary;
    if (statsGeo && sm && my >= statsGeo.y0 && my <= statsGeo.y1 && mx >= statsGeo.sx && mx < statsGeo.sx + 4 * statsGeo.slotW) {
      vLine.style.display = "none";
      showTooltip(evt.pageX, evt.pageY, statTipHtml(STAT_KEYS[Math.floor((mx - statsGeo.sx) / statsGeo.slotW)], sm));
      return;
    }
    // 2026-09-28 체결 기둥 위면 그 가격 행의 체결 풀이(데스크톱 기둥 · 모바일 왼쪽 띠).
    if (tradeInfo && mx > tradeInfo.x0 && mx < tradeInfo.x1 + 2) {
      vLine.style.display = "none";
      const k = Math.floor((yMax - ((my - mt) * ySpan) / ch) / tradeInfo.rowSize), a = tradeInfo.rows.get(k);
      if (!a) { hideTooltip(); return; }
      const fq = (q) => (q >= 1000 ? (q / 1000).toFixed(1) + "k" : q >= 10 ? q.toFixed(0) : q.toFixed(1));
      const pct = (v) => Math.round(100 * v / Math.max(a[0], 1e-9)) + "%", dp = pxDp(), rs = tradeInfo.rowSize;
      showTooltip(evt.pageX, evt.pageY, `<div style="white-space:normal;max-width:min(360px,calc(100vw - 24px))">`
        + `<span style="color:var(--amber);font-weight:700">체결 ${fq(a[0])} ${coinUnit()}</span> · ${(k * rs).toFixed(dp)}–${((k + 1) * rs).toFixed(dp)}`
        + (k === tradeInfo.pocK ? ` · <span style="font-weight:700">최다 체결(POC)</span>` : "")
        + `<br>매수 ${fq(a[3])} · 매도 ${fq(a[4])} → <span style="color:${a[3] >= a[4] ? "var(--good)" : "var(--bad)"};font-weight:700">${a[3] >= a[4] ? "매수" : "매도"} 우위 ${fq(Math.abs(a[3] - a[4]))}</span> (막대 끝 색칠)`
        + `<br>고래 ${pct(a[1])} · 중형 ${pct(Math.max(0, a[0] - a[1] - a[2]))} · 리테일 ${pct(a[2])}`
        + `<br>보이는 봉 ${candles.length}개 합 · 가장 많이 거래된 줄의 ${Math.round(100 * a[0] / tradeInfo.max)}%`
        + `<br><br>옆 호가 막대와 맞대 읽기: 체결이 긴데 호가가 그대로 남아 있으면 <span style="font-weight:700">흡수</span>,`
        + `<br>체결 없이 호가만 두꺼우면 아직 안 닿은 자리다.`
        + `<br><span style="opacity:.7">우위는 테이커(먼저 친 쪽) 기준 · 지지·저항 신호 아님</span></div>`);
      return;
    }
    // 2026-09-27 호가 띠 위면 봉 툴팁 대신 **호가 해석** -- 커서 가격의 칸(없으면 가장 가까운 칸).
    if (bookInfo && mx > tradeX0 && mx <= w - mr) {
      vLine.style.display = "none";
      const priceAt = yMax - ((my - mt) * ySpan) / ch;
      let best = null;
      bookInfo.cells.forEach((c) => { if (!best || Math.abs(c.p - priceAt) < Math.abs(best.p - priceAt)) best = c; });
      if (best && Math.abs(best.p - priceAt) <= bookInfo.bs) { showTooltip(evt.pageX, evt.pageY, bookTipHtml(best)); return; }
      hideTooltip(); return;
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
    // 🔴«값이 없으면 줄을 안 만든다»가 여기서는 거짓말이 된다 -- 리본은 빈 칸을 «횡보»로 쓰므로
    //   모르는 봉도 횡보로 읽힌다. 이력 시작 전이면 그렇게 적는다(2026-09-25).
    const regimeKnown = regimeFromTs != null && c.time >= regimeFromTs;
    const regimeLine = r
      ? `<br>레짐: ${regimeDominant(r) === "bull" ? "강세" : regimeDominant(r) === "bear" ? "약세" : "횡보"} ${Math.round(Math.max(r.bull_prob, r.bear_prob, r.chop_prob) * 100)}%`
      : (regimeByTsForChart && !regimeKnown ? "<br>레짐: 모름 (워커 이력 밖)" : "");
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
      시가: ${fmtNum(c.open, Math.max(2, pxDp()))}<br>
      고가: ${fmtNum(c.high, Math.max(2, pxDp()))}<br>
      저가: ${fmtNum(c.low, Math.max(2, pxDp()))}<br>
      종가: ${fmtNum(c.close, Math.max(2, pxDp()))}${regimeLine}${volLine}${trigLines}
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

  // ── ① 사분면 행 — 델타 × OI (거래대금은 선) ────────────────────────────────
  // 봉 하나가 «누가 무엇을 했나»를 말한다. 델타 부호 × OI 부호가 네 사분면이다:
  //     델타+ · OI+ = 신규 롱    델타+ · OI− = 숏 정리
  //     델타− · OI+ = 신규 숏    델타− · OI− = 롱 정리
  // 🔴농도 상한 0.58 은 **대비 계산에서 나온 수**다. --ink 글자가 채운 면 위에서 4.5:1 을
  //   지키는 한계가 0.60 이고(실측 bad 4.52 · good 4.51), 그 아래라야 글자 색을 하나로
  //   통일할 수 있다. 더 올리면 밝은 막대에서 글자가 안 읽힌다.
  // 🔴농도는 |OI| 의 **크기**만 말한다 -- 부호가 안 남아 «신규 숏»과 «롱 정리»가 같은
  //   빨강이 된다. 그래서 OI 가 증가한 봉에만 막대 위에 주황 캡을 얹는다(3px, 사분면 복구).
  // 🔴막대 최소 높이 26px 은 해석 글자가 들어갈 자리다 -- 그만큼 «높이=|Δ|» 가 0에서
  //   시작하지 않는다. 정확한 값은 막대 아래 숫자가 말한다.
  // ── RVOL (2026-09-23, 사용자 「거래대금 대신 rvol 은 어때?」) ─────────────────────
  // 원시 USD 는 «많은 건가»를 사용자가 스스로 판단해야 한다. RVOL 은 «평소의 n 배»라 읽힌다.
  // 실측(ETH 5m 4.7년, 앞 30분 레인지 상위25%): 거래대금 z288 .6535 -> RVOL 14일 .7055.
  // 🔴값은 **워커가 준다**(scripts/live_eth_breakout_detector_20260911.py::_rvol) -- 기준선이
  //   7일 이상 필요한데 클라도 서버 evidence 캐시(5.2일)도 그만큼을 못 갖는다.
  // 🔴그 워커는 ETHUSDT **전용**이다. 다른 코인이면 RVOL 이 아니라 원래 거래대금으로 돌아간다.
  // 🔴**캐시 키 밖에 두면 안 된다.** quadLane 은 fpBars|oiBars 로만 키를 만들고 있었는데,
  //   RVOL 은 그 둘과 무관하게 도착하므로 키에 안 넣으면 «거래대금»으로 그린 노드가 그대로
  //   재사용된다. 서명은 **값 기반**이다 -- objToken 은 WeakMap 신원이라 폴링마다 새 객체가
  //   되어 매번 무효화되고(캐시가 무의미해진다), 값은 5분봉 하나당 한 번만 바뀐다.
  // 2026-09-28 사분면 위 «거래량 60분» 띠를 없앴다(사용자 지시) -- RVOL 은 카드 상단 «오늘 거래량» 배지로만 남는다.
  //   봉별 활발함은 사분면 판 안의 **거래대금 선**이 말한다(원시 USD, 코인 공통).
  let rvolBaseDays = null;
  {
    const rv = (activeSnapshotAsset === "eth" && latestBreakoutDetector)
      ? latestBreakoutDetector.rvol : null;
    if (rv) rvolBaseDays = Number(rv.base_days) || null;
    // ── 세션 누적 RVOL -> 카드 상단 배지 (2026-09-23 사용자 지시) ─────────────────
    // 라벨(적음/보통/많음)은 **워커가 붙인다** -- 경계(q25 0.70 / q75 1.37)가 바뀌면
    // 화면 두 곳이 아니라 거기 한 곳만 고친다.
    // 🔴톤은 항상 neutral. warn/bad 면 «맥락»이 «신호»로 읽힌다.
    const vb = el("rvolSessionBadge");
    if (vb) {
      const lab = rv && rv.session_label, sv = Number(rv && rv.session);
      const txt = (lab && Number.isFinite(sv))
        ? "오늘 거래량 " + lab + " " + sv.toFixed(2) + "배" : "";
      // 이 함수는 현재가 틱마다 불린다 -- 바뀐 것만 쓴다(setMapBadge 와 같은 이유).
      if (vb.hidden !== !txt) vb.hidden = !txt;
      if (txt && vb.textContent !== txt) {
        vb.textContent = txt;
        const b = Array.isArray(rv.session_bounds) ? rv.session_bounds : [0.7, 1.37];
        vb.title = "오늘(UTC 00:00~지금) 누적 거래대금 ÷ 평소 같은 시점까지의 누적"
          + " (같은 시각 최근 " + (rvolBaseDays || 14) + "일 중앙값).\n"
          + "경계는 임의 상수가 아니라 분위입니다 -- 적음 < " + b[0] + " ≤ 보통 ≤ " + b[1]
          + " < 많음 (ETH 5m 4.7년 분포의 q25 / q75).\n"
          + "이 경계로 가른 날의 앞 24시간 레인지 중앙값은 적음 378bp / 보통 451 / 많음 516 "
          + "으로 단조입니다.\n"
          + "🔴«오늘의 온도»이지 방향도 진입 근거도 아닙니다. 차트 선(RVOL)과 다른 값입니다 -- "
          + "이쪽은 하루 누적이라 지금 봉 하나의 크기와 무관하고, 그래서 현재 봉 폭을 통제해도 "
          + "변별력이 남습니다(봉폭 십분위 안 AUC .5810, 5분 봉 RVOL 은 .4767).";
      }
    }
  }
  // ── 실시간 호가 띠 (2026-09-27 사용자 지시) ─────────────────────────────────────────
  // 서버가 준 **마지막 1초** 호가 열(부호: +매수 / −매도)을 풋프린트와 같은 가격축에 가로 막대로 그린다.
  //   길이 = √(수량/창 안 최대) -- 비례로 두면 제일 큰 벽 하나에 나머지가 선이 됐다(시안). 창 안 상위 10% 는 진하게 + 수량.
  // 🔴«지지·저항»이 아니다 -- 벽에 닿은 뒤 반등률 0.509(동전, 5.8일 71,293건, 09-20). 서술만 한다.
  //   수량은 수집기 래스터 값(칸 $0.5 합산)이고, 가격선은 이 띠를 가로질러 배지까지 이어진다(호가창 사다리처럼).
  // 2026-09-27 축적 호가를 입혔다(사용자 선택 A + «재깔림도»): 한 칸 = 두 겹.
  //   **심(진함) = 창 내내 한 번도 안 빠진 양(pers)** -- 버텨 온 벽 vs 방금 깔린 호가를 가른다(창 안 호가의 36% 는 창을 못 버틴다).
  //   **겉 = 지금 양, 그 위 빗금 농도 = 재깔림(refill/peak) 4단** -- 촘촘할수록 계속 다시 채워지는 자리, 빗금 없거나 성기면 한 번 깔리고 만 호가.
  //   🔴둘 다 «성격»의 서술이다 -- 지속률은 반등을 못 가렸다(상위−하위 +0.001, 09-20). 창 = 위 1h/2h/4h/12h 토글.
  const book = BOOK_W && latestFlowHeatmap && latestFlowHeatmap.book;
  const acc = latestFlowHeatmap && latestFlowHeatmap.rows;
  // 칸별 값은 층 캐시 **밖에서** 매 렌더 계산한다(≈250칸) -- 캐시가 재사용돼도 호버 해석(bookInfo)이 같은 값을 본다.
  const bookInfo = (() => {
    if (!book || !book.q || !(book.bin_size > 0)) return null;
    // 2026-09-28 호가 칸을 **풋프린트(체결 기둥)와 같은 행**으로 묶는다(사용자 «y축이 벌어지면 막대가 작아진다») --
    //   0.5달러 칸 그대로면 창이 넓을 때 한 칸이 2~3px 로 눌렸다. 행 안에서 수량·버틴 양·재깔림(refill·peak)은 더하고,
    //   매수·매도가 한 행에 섞이면(현재가 행) 큰 쪽으로 칠한다. ○(접근행동)는 행 안 **가장 얇아진** 값(approachAt).
    const bin = book.bin_size, bs = Math.max(bin, fpRowSize || bin), x0 = tradeX0 + TRADE_W + 6, L = BOOK_W - 10;
    const accAt = (key, p) => {               // 축적 통계는 창 전체를 접은 행 배열이라 제 격자(bin_lo) -- 가격으로 찾는다
      if (!acc || !acc[key] || !(acc.bin_size > 0)) return 0;
      const i = Math.round(p / acc.bin_size) - acc.bin_lo;
      return i >= 0 && i < acc[key].length ? acc[key][i] : 0;
    };
    const rowsBy = new Map();                 // 행 키 -> {b: 매수 합, a: 매도 합} 각 {q, pers, peak, refill}
    for (let i = 0; i < book.q.length; i++) {
      const v = book.q[i], p = (book.bin_lo + i) * bin;
      if (!v) continue;
      const k = Math.floor((p + 1e-9) / bs), r = rowsBy.get(k) || { b: null, a: null }, side = v > 0 ? "b" : "a";
      const q = Math.abs(v), s = r[side] || (r[side] = { q: 0, pers: 0, peak: 0, refill: 0 });
      s.q += q; s.pers += Math.min(q, accAt("pers", p)); s.peak += accAt("peak", p); s.refill += accAt("refill", p);
      rowsBy.set(k, r);
    }
    const cells = [];
    rowsBy.forEach((r, k) => {
      const p = (k + 0.5) * bs;                 // 행 가운데(그림은 p ± bs/2 = 풋프린트 행과 같은 줄)
      if (p + bs / 2 < yMin || p - bs / 2 > yMax) return;
      const bid = (r.b ? r.b.q : 0) >= (r.a ? r.a.q : 0), s = bid ? r.b : r.a;
      cells.push({ p, q: s.q, bid, pers: s.pers, rw: s.peak > 0 ? s.refill / s.peak : 0,
                   apr: approachAt(acc, k * bs, bs) });   // 접근행동(4h) -- 프로파일 ◌ 와 같은 값 · null = 모름
    });
    if (!cells.length) return null;
    const mx = Math.max(...cells.map((c) => c.q));
    const qs = cells.map((c) => c.q).sort((a, b) => a - b);
    const p90 = qs[Math.floor(qs.length * 0.9)];
    const rwSorted = cells.map((c) => c.rw).filter((x) => x > 0).sort((a, b) => a - b);
    cells.forEach((c) => {                    // 재깔림 = 보이는 범위 안 순위(0~1) -- 프로파일 호가 막대와 같은 방식
      c.wall = c.q >= p90;
      if (rwSorted.length < 2 || !(c.rw > 0)) { c.rwPct = 0; return; }
      let lo = 0, hi = rwSorted.length;
      while (lo < hi) { const m = (lo + hi) >> 1; if (rwSorted[m] < c.rw) lo = m + 1; else hi = m; }
      c.rwPct = lo / (rwSorted.length - 1);
    });
    // 2026-09-28 재깔림 = 겉의 **빗금 농도 4단**(사용자 선택 시안 ②, ABC 글자 대체): 0 = 상위 1/4(가장 촘촘) … 3 = 하위 1/4. 기록 없으면 빗금 없음.
    cells.forEach((c) => { c.hatch = !(c.rw > 0) || rwSorted.length < 4 ? -1 : c.rwPct >= 0.75 ? 0 : c.rwPct >= 0.5 ? 1 : c.rwPct >= 0.25 ? 2 : 3; });
    return { bs, x0, L, mx, cells };
  })();
  // 네 막대 계기 자리(그림과 호버가 같이 쓴다). 데스크톱 = 기둥 머리(체결 기둥 왼쪽 끝 ~ 오른쪽 여백), 모바일 = 풋프린트 위 한 줄.
  const statsGeo = bookInfo && !(TRADE_W && splitR && activeSnapshotAsset === "eth") ? (() => {   // 2026-09-30 넓은 ETH 화면은 ④(#mcWall)로 옮겼다
    const sx = TRADE_W ? tradeX0 + 6 : ml, room = (TRADE_W ? w - mr : bookInfo.x0 - 8) - sx;
    return TRADE_W ? { sx, slotW: room / 4, labY: mt - 24, barY: mt - 20, y0: mt - 34, y1: mt - 11 }
                   : { sx, slotW: room / 4, labY: mt - 14, barY: mt - 11, y0: mt - 24, y1: mt - 2 };
  })() : null;
  // 2026-09-28 툴팁 = «심 × ◌» 여섯 조합의 풀이(사용자 지시 -- 기존 수치·해석 문구는 뺐다). 이 칸의 조합만 진하게.
  //   심 두꺼움 = 지금 양의 절반 이상이 창 내내 버텼다 · ◌ = 최근 4h 가까울 때(0.35% 안) 두께가 멀 때의 0.8배 미만 · 모름 = 비교 자격 없음.
  //   🔴전부 «성격»의 서술이다 -- 조합이 다음 가격을 말하는지는 안 쟀다(벽 반등률 0.509 = 동전, 09-20).
  const BOOK_COMBOS = [
    [true, "ring", "두꺼운 심 · ○", "멀 때는 버티지만 가격이 오면 빼던 자리 -- 지금 두께를 그대로 믿기 어렵다"],
    [true, "none", "두꺼운 심 · ○ 없음", "창 내내 버텼고 가격이 와도 얇아지지 않았다 -- 가장 «걸려 있는» 모양"],
    [true, "unk", "두꺼운 심 · 모름", "창 내내 버텼지만 4h 동안 가격이 다가온 적이 없어 그때의 행동은 모른다"],
    [false, "ring", "얇은 심 · ○", "이번 창에서도 들락날락했고 다가오면 빼던 자리 -- 벽으로 믿기 가장 어렵다"],
    [false, "none", "얇은 심 · ○ 없음", "자주 바뀌지만 다가와도 얇아지진 않았다 -- 새로 깔리거나 고쳐 거는 호가"],
    [false, "unk", "얇은 심 · 모름", "방금 깔렸고 다가온 이력도 없다 -- 읽을 게 거의 없다"],
  ];
  const bookTipHtml = (c) => {
    const thick = c.q > 0 && c.pers / c.q >= 0.5;
    const ring = c.apr === null ? "unk" : c.apr < 0.8 ? "ring" : "none";
    const col = c.bid ? "var(--good)" : "var(--bad)";
    return `<div style="white-space:normal;max-width:min(380px,calc(100vw - 24px))">`
      + BOOK_COMBOS.map(([t, r, name, read]) => (t === thick && r === ring
          ? `<div style="margin:2px 0"><span style="color:${col};font-weight:700">▸ ${name}</span><br>${read}</div>`
          : `<div style="margin:2px 0;opacity:.5">${name}<br>${read}</div>`)).join("")
      + `<div style="margin-top:6px;opacity:.7">심 두꺼움 = 지금 양의 절반 이상이 창 내내 버팀 · ○ = 최근 4h 가격이 0.35% 안에 왔을 때 `
      + `두께가 멀 때의 0.8배 미만 · 모름 = 가까울 때와 멀 때를 둘 다 겪지 않음 · 빗금 = 재깔림, 촘촘할수록 자주 다시 채움`
      + (c.rw > 0 ? ` (이 칸 ${c.rw.toFixed(1)}배 · 보이는 범위 상위 ${Math.round(100 * (1 - c.rwPct))}%)` : "")
      + `. 성격의 서술 · 예측력 안 잼 · 지지·저항 아님</div></div>`;
  };
  cachedLayer("bookStrip", book ? objToken(book) + "|" + objToken(acc) + "|" + activeSnapshotAsset : "none", (g) => {   // 2026-09-30 신원 = 객체(히트맵 행이 bin_lo 그대로 바뀌어도 다시 그린다)
    if (!bookInfo) return;
    const { bs, x0, L, mx, cells } = bookInfo;
    const fmtQ = (q) => (q >= 1000 ? (q / 1000).toFixed(1) + "k" : q >= 10 ? q.toFixed(0) : q.toFixed(1));
    const head = (x, anchor, fill, str) => {
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", x); t.setAttribute("y", mt - 4); t.setAttribute("font-size", "10");
      t.setAttribute("text-anchor", anchor); t.setAttribute("fill", fill); t.textContent = str;
      g.appendChild(t);
    };
    if (TRADE_W) { head(x0 - 6, "end", "var(--amber)", "← 체결"); head(x0, "start", "var(--muted)", "호가 →"); }
    else { head(x0, "start", "var(--muted)", "호가"); if (TRADE_L) head(ml - 2, "end", "var(--amber)", "체결"); }
    // 2026-09-28 호가 요약 네 숫자 = **막대 계기 넷**(사용자 선택 시안 A) -- 데스크톱은 체결·호가 기둥 머리, 모바일은 풋프린트 위 한 줄.
    //   변동 0→100 채움(높으면 주황) · 불균형 가운데 0 에서 초록(매수 호가 두꺼움)/빨강 · 지속·이탈 0→100 채움.
    //   값·뜻은 막대마다 호버 툴팁(statTipHtml). 좌표는 statsGeo 하나를 그림과 호버가 같이 쓴다.
    const sm = latestFlowHeatmap && latestFlowHeatmap.summary;
    if (sm && statsGeo) {
      const { sx, slotW, labY, barY } = statsGeo, bw = slotW - 10;
      STAT_KEYS.forEach((k, i) => {
        const x = sx + i * slotW, f = statFrac(k, sm);
        const lb = document.createElementNS(NS, "text");
        lb.setAttribute("x", x); lb.setAttribute("y", labY); lb.setAttribute("font-size", "9");
        lb.setAttribute("fill", "var(--muted)"); lb.textContent = STAT_NAME[k];
        g.appendChild(lb);
        const tr = document.createElementNS(NS, "rect");
        tr.setAttribute("x", x); tr.setAttribute("y", barY); tr.setAttribute("width", bw); tr.setAttribute("height", 7);
        tr.setAttribute("rx", 2); tr.setAttribute("fill", "var(--ink)"); tr.setAttribute("fill-opacity", "0.12");
        g.appendChild(tr);
        if (f == null) return;
        const fl = document.createElementNS(NS, "rect");
        let fx = x, fw = bw * f, col = k === "vol" ? (f >= 0.66 ? "var(--warn)" : f >= 0.33 ? "var(--ink)" : "var(--muted)")
                                   : k === "persist" ? "var(--ink)" : "var(--muted)";
        if (k === "obi") {                     // -1..+1, 가운데 0
          const c = x + bw / 2, L = bw / 2 * Math.min(1, Math.abs(f));
          fx = f >= 0 ? c : c - L; fw = L; col = f >= 0 ? "var(--good)" : "var(--bad)";
          const z = document.createElementNS(NS, "line");
          z.setAttribute("x1", c); z.setAttribute("x2", c); z.setAttribute("y1", barY - 2); z.setAttribute("y2", barY + 9);
          z.setAttribute("stroke", "var(--ink)"); z.setAttribute("stroke-opacity", "0.6");
          g.appendChild(z);
        }
        fl.setAttribute("x", fx); fl.setAttribute("y", barY); fl.setAttribute("width", Math.max(1, fw)); fl.setAttribute("height", 7);
        fl.setAttribute("rx", 2); fl.setAttribute("fill", col); fl.setAttribute("fill-opacity", "0.95");
        g.appendChild(fl);
      });
    }
    const base = document.createElementNS(NS, "line");
    base.setAttribute("x1", x0 - 1); base.setAttribute("x2", x0 - 1);
    base.setAttribute("y1", mt); base.setAttribute("y2", plotBottom);
    base.setAttribute("stroke", "var(--soft-line)");
    g.appendChild(base);
    // 빗금 무늬 4단(촘촘 → 성김). 45° 선 -- 심(단색)·겉(연한 단색)과 겹쳐도 질감으로 갈린다.
    const hid = (k) => "bkh" + k + "-" + svg.id;
    const defs = document.createElementNS(NS, "defs");
    [2.6, 4, 6.5, 11].forEach((sp, k) => {
      const pt = document.createElementNS(NS, "pattern");
      pt.setAttribute("id", hid(k)); pt.setAttribute("width", sp); pt.setAttribute("height", sp);
      pt.setAttribute("patternUnits", "userSpaceOnUse"); pt.setAttribute("patternTransform", "rotate(45)");
      const ln = document.createElementNS(NS, "line");
      ln.setAttribute("x1", 0); ln.setAttribute("y1", 0); ln.setAttribute("x2", 0); ln.setAttribute("y2", sp);
      ln.setAttribute("stroke", "var(--ink)"); ln.setAttribute("stroke-opacity", "0.55"); ln.setAttribute("stroke-width", "1.1");
      pt.appendChild(ln); defs.appendChild(pt);
    });
    g.appendChild(defs);
    cells.forEach((c) => {
      const yTop = Math.max(mt, yAt(c.p + bs / 2)), yBot = Math.min(plotBottom, yAt(c.p - bs / 2));
      const len = Math.max(1, Math.sqrt(c.q / mx) * L);
      const color = c.bid ? "var(--good)" : "var(--bad)", rh = Math.max(1, yBot - yTop - 1);
      const shell = document.createElementNS(NS, "rect");
      shell.setAttribute("x", x0); shell.setAttribute("y", yTop + 0.5);
      shell.setAttribute("width", len); shell.setAttribute("height", rh);
      shell.setAttribute("fill", color);
      shell.setAttribute("fill-opacity", "0.3");
      g.appendChild(shell);
      const coreLen = c.pers > 0 ? Math.max(1, Math.sqrt(c.pers / mx) * L) : 0;
      if (c.hatch >= 0 && len - coreLen > 1) {
        const ht = document.createElementNS(NS, "rect");
        ht.setAttribute("x", x0 + coreLen); ht.setAttribute("y", yTop + 0.5);
        ht.setAttribute("width", len - coreLen); ht.setAttribute("height", rh);
        ht.setAttribute("fill", `url(#${hid(c.hatch)})`);
        g.appendChild(ht);
      }
      if (coreLen) {
        const core = document.createElementNS(NS, "rect");
        core.setAttribute("x", x0); core.setAttribute("y", yTop + 0.5);
        core.setAttribute("width", coreLen); core.setAttribute("height", rh);
        core.setAttribute("fill", color); core.setAttribute("fill-opacity", "0.9");
        g.appendChild(core);
      }
      // ◌ = 다가오면 얇아진 가격대(프로파일과 같은 표식) -- 막대 끝에 붙이고, 막대 밖 글자는 그만큼 오른쪽으로 민다.
      let ringW = 0;
      if (c.apr !== null && c.apr < 0.8) {
        const rr = Math.max(1.8, Math.min(3.2, rh / 2));
        const mk = document.createElementNS(NS, "circle");
        mk.setAttribute("cx", x0 + len + 1.5 + rr); mk.setAttribute("cy", (yTop + yBot) / 2);
        mk.setAttribute("r", rr); mk.setAttribute("fill", "none");
        mk.setAttribute("stroke", "var(--ink)"); mk.setAttribute("stroke-width", "1.2");
        g.appendChild(mk);
        ringW = 2 * rr + 2;
      }
      if (c.wall && !mobileChart) {
        const lb = document.createElementNS(NS, "text");
        lb.setAttribute("x", Math.min(x0 + len + 2 + ringW, x0 + L - 12));
        lb.setAttribute("y", Math.max(mt + 8, Math.min(plotBottom - 2, (yTop + yBot) / 2 + 3)));
        lb.setAttribute("font-size", "9"); lb.setAttribute("fill", color); lb.setAttribute("class", "pf-qty");
        lb.textContent = fmtQ(c.q);
        g.appendChild(lb);
      }
    });
  });
  // 2026-09-28 사분면·누적 두 레인이 같은 봉별 값을 쓴다 -- 렌더마다 한 번만 만든다(층 캐시가 둘 다 맞으면 안 만든다).
  //   봉 찾기는 Map 이다(예전 find 는 봉 수의 제곱). 같은 시각이 둘이면 **앞의 것**(find 와 같은 결과).
  //   🔴«OI 모름»과 «ΔOI 0» 을 가른다(2026-09-25) -- 없는 봉은 oi = null(«신규 롱/숏» 이라는 없는 사실을 안 찍는다).
  let laneBars = null;
  const laneBarsOf = () => laneBars || (laneBars = (() => {
    const byT = new Map();
    fpBars.forEach((b) => { const t = Number(b && b.time); if (!byT.has(t)) byT.set(t, b); });
    const oiByTs = new Map(oiBars.map((b) => [Number(b[0]), Number(b[1]) || 0]));
    return candles.map((c) => {
      const b = byT.get(c.time);
      if (!b) return null;
      const f = supplyFlowOfBar(b.levels);
      let turn = 0;
      (b.levels || []).forEach((l) => { turn += (Number(l[0]) || 0) * ((Number(l[1]) || 0) + (Number(l[2]) || 0)); });
      return { t: c.time, turn, delta: f.whale + f.mid + f.retail, whale: f.whale, mid: f.mid, retail: f.retail,
               oi: oiByTs.has(c.time) ? (Number(oiByTs.get(c.time)) || 0) : null };
    });
  })());
  cachedLayer("quadLane", objToken(fpBars) + "|" + objToken(oiBars)
              + "|" + activeSnapshotAsset, (g) => {
  if (fpBars.length && candles.length && QUAD_H) {
    const QB = quadY + QUAD_H;                       // 판 바닥
    const rows = laneBarsOf();
    const have = rows.filter(Boolean);
    if (have.length) {
      // 2026-09-23 사용자 지시로 이 행의 **선을 뺐다**. RVOL 은 누적 CVD 아래 제 레인으로
      // 갔고(누적 CVD 레인 안), 여기 남기면 1시간 RVOL 이 한 카드에 두 번 그려진다.
      // 이 행의 주인공은 막대(델타 x OI)다. 봉별 «평소 대비»는 막대 툴팁이 그대로 답한다.
      const TXT = mobileChart ? 10 : 12;
      // 🔴OI 를 모르면 칸 이름을 **유보한다** -- 사분면은 델타 x OI 라 한 축이 없으면 칸이 없다.
      const QNAME = (d, o) => (o == null ? (d >= 0 ? "매수 우위" : "매도 우위")
                                         : d >= 0 ? (o >= 0 ? "신규 롱" : "숏 정리")
                                                  : (o >= 0 ? "신규 숏" : "롱 정리"));
      const put = (el) => { g.appendChild(el); return el; };
      // 막대 **안**의 해석은 --ink 하나로 통일한다(채운 면 위라 부호색을 쓰면 대비가 깨진다).
      // 막대 **아래**의 숫자는 어두운 배경이라 부호색을 쓸 수 있다(2026-09-22 사용자 지시).
      const mkText = (x, y, txt, anchor, weight, color) => {
        const t = document.createElementNS(NS, "text");
        t.setAttribute("x", x); t.setAttribute("y", y);
        t.setAttribute("font-size", TXT); t.setAttribute("fill", color || "var(--ink)");
        if (anchor) t.setAttribute("text-anchor", anchor);
        if (weight) t.setAttribute("font-weight", weight);
        t.textContent = txt;
        return put(t);
      };
      const base = document.createElementNS(NS, "line");
      base.setAttribute("x1", ml); base.setAttribute("x2", ml + cw);
      base.setAttribute("y1", QB); base.setAttribute("y2", QB);
      base.setAttribute("stroke", "var(--line)");
      put(base);
      // ── 2026-09-28 시안 E(사용자 선택): 막대 대신 **봉 칸 배경을 사분면 색**으로 칠하고 이름은 칸 위에 --
      //   그 위에 누적 CVD·OI(cumLane, 이제 주연)와 봉별 **거래대금 선**이 올라간다. 신규(OI 증가) 칸은 진하게, 정리 칸은 옅게.
      //   OI 모름이면 가장 옅게(칸 이름도 «매수/매도 우위»로 유보 -- QNAME).
      const slotW = cw / candles.length;
      const tMax = Math.max(...have.map((r) => r.turn), 1e-9);
      rows.forEach((r, i) => {
        if (!r) return;
        const rect = document.createElementNS(NS, "rect");
        rect.setAttribute("x", ml + i * slotW + 0.5); rect.setAttribute("y", quadY);
        rect.setAttribute("width", Math.max(1, slotW - 1)); rect.setAttribute("height", QUAD_H);
        rect.setAttribute("fill", r.delta >= 0 ? "var(--good)" : "var(--bad)");
        rect.setAttribute("fill-opacity", r.oi == null ? "0.06" : r.oi >= 0 ? "0.2" : "0.1");
        const tip = document.createElementNS(NS, "title");
        tip.textContent = fmtDateTick(r.t * 1000) + " " + QNAME(r.delta, r.oi)
          + " · 델타 " + (r.delta >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(r.delta)) + " " + coinUnit()
          + " (고래 " + (r.whale >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(r.whale))
          + " · 중형 " + (r.mid >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(r.mid))
          + " · 리테일 " + (r.retail >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(r.retail)) + ")"
          + " · 신규계약 " + (r.oi == null ? "모름"
              : (r.oi >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(r.oi)) + " " + coinUnit())
          + " · 거래대금 " + fmtUsdCompact(r.turn);
        rect.appendChild(tip);
        put(rect);
        if (QUAD_TEXT_OK) mkText(ml + (i + 0.5) * slotW, quadY + TXT + 2, QNAME(r.delta, r.oi), "middle", "700",
                                 r.delta >= 0 ? "var(--good)" : "var(--bad)");
      });
      // 거래대금 선 -- 판 아래쪽 40% 에 제 축(0 = 판 바닥). 누적 CVD·OI 와 **다른 축**이라 세로 위치를 서로 비교하지 않는다.
      {
        const ty = (v) => QB - 4 - (v / tMax) * QUAD_H * 0.4;
        let d = "";
        rows.forEach((r, i) => { if (r) d += (d ? " L" : "M") + (ml + (i + 0.5) * slotW).toFixed(1) + " " + ty(r.turn).toFixed(1); });
        const pl = document.createElementNS(NS, "path");
        pl.setAttribute("d", d); pl.setAttribute("fill", "none"); pl.setAttribute("stroke", "var(--turnover)");
        pl.setAttribute("stroke-width", "1.8"); pl.setAttribute("stroke-dasharray", "5 3"); pl.setAttribute("stroke-linejoin", "round");
        const pt = document.createElementNS(NS, "title");
        pt.textContent = "거래대금 — 봉마다 체결 금액(USD, 풋프린트 셀 합). 판 아래쪽 40% 에 제 축으로 그렸다(0 = 판 바닥) — "
          + "누적 CVD·OI 선과 세로 위치를 비교하지 마세요.";
        pl.appendChild(pt);
        put(pl);
        const lastT = have[have.length - 1];
        if (mobileChart) {
        const tv = document.createElementNS(NS, "text");
        tv.setAttribute("x", w - 2); tv.setAttribute("y", QB - 6); tv.setAttribute("text-anchor", "end");
        tv.setAttribute("font-size", mobileChart ? "10" : "12"); tv.setAttribute("fill", "var(--turnover)");
        tv.textContent = "거래대금 " + fmtUsdCompact(lastT.turn);
        put(tv);
        }
      }
      if (QUAD_TEXT_OK) {
        const sgnCol = (v) => (v >= 0 ? "var(--good)" : "var(--bad)");
        rows.forEach((r, i) => {
          if (!r) return;
          const cx = xAt(i) + bw / 2;
          mkText(cx, QB + TXT + 2, "Δ" + (r.delta >= 0 ? "+" : "-")
                 + fmtFootprintQty(Math.abs(r.delta)), "middle", "700", sgnCol(r.delta));
          mkText(cx, QB + TXT * 2 + 5,
                 r.oi == null ? "OI —" : "OI" + (r.oi >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(r.oi)),
                 "middle", null, r.oi == null ? "var(--muted)" : sgnCol(r.oi));
        });
      }
      // 2026-09-30 왼쪽 판 범례(선 견본·칸 농도) 제거(사용자 지시) -- 이름은 판 안 값 줄(cumLane)의 글자 색이 말한다.
    }
  }
  });

  // ── ② 누적 행 — 고래/중형/리테일 스택, 그 윤곽이 CVD ───────────────────────
  // 🔴**항등식이다**: 누적고래 + 누적중형 + 누적리테일 = CVD (실측 오차 0.0000 ETH).
  //   그래서 스택의 맨 위 윤곽을 그리면 그게 곧 CVD 선이고, 선을 하나도 더 안 쓴다.
  //   바로 위 1초 차트가 똑같은 그림이라 눈이 아래로 그대로 이어진다.
  // 🔴누적 OI 도 **같은 ETH 축**이다. 둘 다 ETH 이고 실측 진폭도 같은 자릿수라(CVD 31k vs
  //   OI 9.8k) 축을 나눌 이유가 없다 -- 나누면 세로 위치에 뜻이 없어진다.
  // 🔴기준점은 **창 시작**이다. 창(1h/2h/4h)을 바꾸면 기준점도 같이 옮겨간다.
  // 🔴RVOL 이 이 레인 안으로 들어왔으므로 **캐시 키에도** 들어가야 한다 -- 안 넣으면
  //   RVOL 만 갱신됐을 때 옛 노드가 그대로 재사용된다(사분면에서 같은 버그를 이미 겪었다).
  cachedLayer("cumLane", objToken(fpBars) + "|" + objToken(oiBars) + "|" + chartWindowBars
              + "|" + activeSnapshotAsset, (g) => {
  if (fpBars.length && candles.length && CUM_DRAW_H) {
    let aw = 0, am = 0, ar = 0, ao = 0;
    const rows = laneBarsOf().map((r) => {
      if (!r) return null;
      aw += r.whale; am += r.mid; ar += r.retail; ao += (r.oi || 0);
      return { t: r.t, w: aw, m: aw + am, c: aw + am + ar, oi: ao, turn: r.turn };
    });
    const have = rows.filter(Boolean);
    if (have.length >= 2) {
      // 🔴2026-10-01 눈금은 **쌓는 층 경계(고래 w · 고래+중형 m)까지** 본다 -- CVD·OI 만 보면 고래 −25k·중형 +18k 가 상쇄돼 CVD −2.3k 일 때
      //   눈금이 작아져 고래 층이 레인 바닥을 뚫고 레짐 줄까지 내려왔다(사용자 신고).
      const amp = Math.max(...have.map((r) => Math.max(Math.abs(r.w), Math.abs(r.m), Math.abs(r.c), Math.abs(r.oi))), 1e-9) * 1.06;
      // 합친 판: 맨 위 RVOL 띠(~20px) 아래로만 그린다 -- 파란 거래량 선과 섞이지 않게.
      // 2026-09-28 시안 E: 합친 판에서 누적이 **주연**이다(막대가 사라짐) -- 위 칸 이름 줄(~20px)만 비우고 판을 다 쓴다.
      const mid = LANE_MERGE ? cumY + 20 + (CUM_DRAW_H - 20) / 2 : cumY + CUM_H / 2;
      const half = LANE_MERGE ? (CUM_DRAW_H - 20) / 2 - 4 : CUM_H / 2 - 6;
      const yv = (v) => mid - (v / amp) * half;
      const cx = (i) => xAt(i) + bw / 2;
      const zero = document.createElementNS(NS, "line");
      zero.setAttribute("x1", ml); zero.setAttribute("x2", ml + cw);
      zero.setAttribute("y1", mid); zero.setAttribute("y2", mid);
      zero.setAttribute("stroke", "var(--line)");
      g.appendChild(zero);
      // 쌓기이지 겹치기가 아니다 -- 겹쳐 그리면 가려진 층의 두께를 눈으로 못 잰다.
      const band = (lo, hi, op) => {
        let d = "";
        rows.forEach((r, i) => { if (r) d += (d ? " L" : "M") + cx(i).toFixed(1) + " " + yv(lo(r)).toFixed(1); });
        for (let i = rows.length - 1; i >= 0; i--) {
          if (rows[i]) d += " L" + cx(i).toFixed(1) + " " + yv(hi(rows[i])).toFixed(1);
        }
        if (!d) return;
        const last = have[have.length - 1];
        const path = document.createElementNS(NS, "path");
        path.setAttribute("d", d + " Z");
        path.setAttribute("fill", hi(last) - lo(last) < 0 ? "var(--bad)" : "var(--good)");
        path.setAttribute("fill-opacity", op); path.setAttribute("stroke", "none");
        g.appendChild(path);
      };
      // 합친 판(데스크톱)에서는 막대가 주연이다 -- 누적은 옅은 배경(사용자 선택 A).
      const fade = 1;   // 2026-09-28 시안 E -- 옅게 깔던 것(1/3)을 되돌렸다
      band(() => 0, (r) => r.w, String(0.42 * fade));
      band((r) => r.w, (r) => r.m, String(0.24 * fade));
      band((r) => r.m, (r) => r.c, String(0.11 * fade));
      const line = (val, color, width, opacity) => {
        let d = "";
        rows.forEach((r, i) => { if (r) d += (d ? " L" : "M") + cx(i).toFixed(1) + " " + yv(val(r)).toFixed(1); });
        if (!d) return;
        const path = document.createElementNS(NS, "path");
        path.setAttribute("d", d); path.setAttribute("fill", "none");
        path.setAttribute("stroke", color); path.setAttribute("stroke-width", width);
        path.setAttribute("stroke-opacity", opacity); path.setAttribute("stroke-linejoin", "round");
        g.appendChild(path);
      };
      line((r) => r.w, "var(--bad)", 1, 0.55);      // 층 경계(농도만으로는 안 갈린다)
      line((r) => r.m, "var(--bad)", 1, 0.55);
      line((r) => r.oi, "var(--warn)", 2.4, 0.95);   // 누적 신규계약
      line((r) => r.c, "var(--accent)", 2.6, 1);     // = CVD (스택의 윤곽)
      const last = have[have.length - 1];
      if (!mobileChart) {   // 2026-09-30 선 끝(마지막 점)에 값(사용자 지시) -- 둘이 가까우면 위아래로 비킨다
        let li = rows.length - 1; while (li >= 0 && !rows[li]) li--;
        const ex = cx(li) - 8, yc = yv(last.c), yo = yv(last.oi), push = Math.abs(yc - yo) < 14 ? (14 - Math.abs(yc - yo)) / 2 : 0;   // 점 왼쪽(오른쪽은 ④ 자리)
        [[yc - (yc <= yo ? push : -push), "CVD", last.c, "var(--accent)"], [yo - (yo < yc ? push : -push), "OI", last.oi, "var(--warn)"]].forEach(([y, nm, v, col]) => {
          const dot = document.createElementNS(NS, "circle");
          dot.setAttribute("cx", cx(li)); dot.setAttribute("cy", nm === "CVD" ? yc : yo); dot.setAttribute("r", 3); dot.setAttribute("fill", col);
          g.appendChild(dot);
          const t = document.createElementNS(NS, "text");
          t.setAttribute("x", ex); t.setAttribute("y", y + 4); t.setAttribute("text-anchor", "end"); t.setAttribute("font-size", "12"); t.setAttribute("font-weight", "700");
          t.setAttribute("fill", col); t.setAttribute("stroke", "var(--chart-bg)"); t.setAttribute("stroke-width", "3"); t.setAttribute("paint-order", "stroke");
          t.textContent = `${nm} ${(v >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(v))}`;
          g.appendChild(t);
        });
      }
      const sgn = (v) => (v >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(v));
      // 2026-09-28 값 칸(사용자 «라벨을 깔끔하게»): 데스크톱 = 판 오른쪽 위에 **지금 값만**(무엇인지는 왼쪽 견본이 말한다) --
      //   CVD 15 굵게 · OI · 거래대금 13, 한 칸 띄우고 고래·중형·리테일 11. 모바일 = 판 아래 한 줄(견본 없이 이름+값).
      const sgnCol = (v) => (v >= 0 ? "var(--good)" : "var(--bad)");
      // 2026-09-28 크기 12 하나로 통일 · **왼쪽 정렬**(사용자 지시) -- 이름 칸 폭을 맞춰 값이 한 세로줄에 선다.
      const vals = [["CVD", sgn(last.c), sgnCol(last.c), 12, "700"],
                    ["OI", sgn(last.oi), "var(--warn)", 12, "700"],
                    ["거래대금", fmtUsdCompact(last.turn), "var(--turnover)", 12, "700"],
                    null,
                    ["고래", sgn(last.w), sgnCol(last.w), 12, null],
                    ["중형", sgn(last.m - last.w), sgnCol(last.m - last.w), 12, null],
                    ["리테일", sgn(last.c - last.m), sgnCol(last.c - last.m), 12, null]];
      if (!mobileChart) {
        // 2026-09-30 값은 판 **안** 한 줄(사용자 «사분면 데이터 표시는 차트 안에 텍스트로») -- 이름 색 = 선 색(범례를 겸한다).
        //   칸 배경(진함 신규 · 옅음 정리)은 줄 툴팁이 말한다. 오른쪽 값 칸·지지/저항 줄은 비웠다(지지/저항 → 시장 맥락 ③ 가격 지형).
        const row = document.createElementNS(NS, "text");
        row.setAttribute("x", ml + 6); row.setAttribute("y", cumY + 34); row.setAttribute("font-size", "12");
        row.setAttribute("stroke", "var(--chart-bg)"); row.setAttribute("stroke-width", "3"); row.setAttribute("paint-order", "stroke");
        row.setAttribute("style", "font-variant-numeric: tabular-nums");
        const nameCol = { CVD: "var(--accent)", OI: "var(--warn)", "거래대금": "var(--turnover)" };
        vals.filter(Boolean).filter((v) => v[0] !== "CVD" && v[0] !== "OI").forEach((v, k) => {   // CVD·OI 는 선 끝 값
          const n = document.createElementNS(NS, "tspan");
          n.setAttribute("fill", nameCol[v[0]] || "var(--muted)"); n.setAttribute("font-weight", "700");
          if (k) n.setAttribute("dx", k === 1 ? "18" : "12");
          n.textContent = v[0] + " ";
          const t = document.createElementNS(NS, "tspan");
          t.setAttribute("fill", v[2]); if (v[4]) t.setAttribute("font-weight", v[4]);
          t.textContent = v[1];
          row.append(n, t);
        });
        const ti = document.createElementNS(NS, "title");
        ti.textContent = "봉 칸 배경 = 사분면: 색은 델타 부호(초록 매수·빨강 매도), 진하면 OI 증가(신규 롱/숏) · 옅으면 OI 감소(정리). "
          + "흰 선 = 창 시작부터 누적 CVD(고래·중형·리테일 면의 윤곽), 주황 = 누적 OI(같은 축), 파란 점선 = 봉별 거래대금(판 아래 40% 제 축).";
        row.appendChild(ti);
        g.appendChild(row);
      } else {
        let rowX = 3;
        vals.filter(Boolean).filter((v) => v[0] !== "거래대금").forEach((v, k) => {
          const t = document.createElementNS(NS, "text");
          t.setAttribute("x", rowX); t.setAttribute("y", cumBottom + 11);
          t.setAttribute("font-size", k === 0 ? 11 : 10); t.setAttribute("fill", v[2]);
          if (k === 0) t.setAttribute("font-weight", "700");
          t.textContent = v[0] + " " + v[1];
          g.appendChild(t);
          let adv = 0;
          try { adv = t.getComputedTextLength(); } catch (_) { adv = t.textContent.length * 6.2; }
          rowX += (adv > 0 ? adv : t.textContent.length * 6.2) + 8;
        });
      }
    }
  }
  }, LANE_MERGE ? "quadLane" : null);   // 합친 판: 사분면 막대 **뒤에**

  // 2026-09-30 지지·저항 글자 줄은 시장 맥락 ③ 가격 지형으로 옮겼다(사용자 지시) -- mcGfx.terrain 이 srLevelsLive() 를 그린다.


  // ── ③ 청산 — 풋프린트 봉 고가 «바로 위» 동그라미 (2026-09-22 사용자 지시) ────
  // 레인에서 뺐다. 청산은 **가격에서 일어나는 사건**이라 가격 옆에 있어야 하고, 레인에
  // 두면 어느 봉의 청산인지 눈이 세로로 훑어야 한다.
  // 🔴크기는 √다. 5분봉 청산은 중앙 $211 / 최대 $4.9M 로 23,000배라 선형이면 큰 것 하나만
  //   남는다(옛 레인이 로그를 쓴 이유와 같다). 반지름이라 √면 **면적이 금액에 비례**한다.
  cachedLayer("liqDots", objToken(liqBars) + "|" + objToken(fpBars) + "|" + timesSig, (g) => {
  if (Array.isArray(liqBars) && liqBars.length && candles.length) {
    // 🔴«봉 위»의 기준은 OHLC 고가가 **아니다**. 풋프린트 셀은 별도 원천이라 고가보다 위
    //   가격대까지 그려지는 봉이 있고, 고가만 보고 앉히면 큰 원이 셀 숫자를 덮는다
    //   (실렌더에서 r=13·12.2 짜리 둘이 9px 셀 숫자 «18.6»·«66.1» 을 가렸다).
    //   그래서 그 봉이 실제로 그린 **가장 높은 가격**을 쓴다.
    const topByTs = new Map();
    fpBars.forEach((b) => {
      const t = Number(b && b.time);
      if (!Number.isFinite(t)) return;
      let top = -Infinity;
      (b.levels || []).forEach((l) => { const pz = Number(l[0]); if (pz > top) top = pz; });
      if (Number.isFinite(top)) topByTs.set(t, top);
    });
    const byTs = new Map();
    liqBars.forEach((b) => {
      const t = Date.parse(b.ts);          // ⚠️ms -> 초. 캔들 time 은 초다(옛 레인의 그 함정).
      if (Number.isFinite(t)) byTs.set(Math.floor(t / 1000), b);
    });
    let peak = 0;
    candles.forEach((c) => {
      const b = byTs.get(c.time);
      if (b) peak = Math.max(peak, (Number(b.long_usd) || 0) + (Number(b.short_usd) || 0));
    });
    if (peak > 0) {
      const rMax = mobileChart ? 9 : 13;
      candles.forEach((c, i) => {
        const b = byTs.get(c.time);
        if (!b) return;
        const lo = Number(b.long_usd) || 0, sh = Number(b.short_usd) || 0;
        const v = lo + sh;
        if (v <= 0) return;
        const r = 3 + Math.sqrt(v / peak) * (rMax - 3);
        // 그 봉이 그린 것 중 가장 높은 것 바로 위(셀 숫자 한 줄 9px 만큼 더 띄운다).
        const top = Math.max(c.high, topByTs.get(c.time) || -Infinity);
        const cy = Math.max(mt + r + 1, yAt(top) - r - 14);
        const dot = document.createElementNS(NS, "circle");
        dot.setAttribute("cx", (xAt(i) + bw / 2).toFixed(1));
        dot.setAttribute("cy", cy.toFixed(1));
        dot.setAttribute("r", r.toFixed(1));
        dot.setAttribute("fill", sh >= lo ? "var(--good)" : "var(--bad)");
        dot.setAttribute("fill-opacity", "0.9");
        const tip = document.createElementNS(NS, "title");
        tip.textContent = fmtDateTick(c.time * 1000) + " 청산 " + fmtUsdCompact(v)
          + " (롱 " + fmtUsdCompact(lo) + " / 숏 " + fmtUsdCompact(sh) + ")"
          + (b.partial ? " · 진행 중" : "")
          + (b.okx ? " · 바이낸스+OKX" : "")    // 2026-09-24 재기동 전 봉은 바이낸스만(서버 주석)
          // 2026-09-24 HL 고래 청산도 합산에 들어 있다 -- 무엇이 얼마인지 따로 적는다(추적 300지갑 한정).
          + (b.hl ? " · HL 고래 청산 " + fmtUsdCompact((b.hl.long_usd || 0) + (b.hl.short_usd || 0))
             + " (" + b.hl.n + "건 · 롱 " + fmtUsdCompact(b.hl.long_usd || 0) + " / 숏 "
             + fmtUsdCompact(b.hl.short_usd || 0) + " · 추적 300지갑 한정)" : "");
        dot.appendChild(tip);
        g.appendChild(dot);
      });
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
    // 2026-09-19 히트맵도 같은 캐시를 쓴다 -- 래스터는 3초마다 새 열이 오는데 캔들 전체
    // 리렌더(가격 틱)를 기다릴 이유가 없다(2bb2b2f1 이 프로파일/1초수급에 넣은 그 이유).
    // 2026-09-30 시안 Y(사용자 선택): ETH 는 오른쪽 칸 위 절반만 1초 수급, 아래 절반은 시장 맥락(#mcBody 를 그 자리에 겹쳐 놓는다).
    const mcSplit = !!splitR && activeSnapshotAsset === "eth";
    // 2026-09-30 1초 수급 높이 = 칸 − 시장 맥락의 **실제 내용 높이**(mcNeedH, renderMarketCtx 가 잰다) -- 한 화면 모드로 칸이 줄어도 시장 맥락이 스크롤 없이 다 들어간다.
    //   아직 못 쟀으면 47% · 1초 수급은 칸의 30% 아래로는 안 줄인다.
    // 2026-10-01 바닥 = 전환 리본 바닥(사용자 «전환 리본까지 나머지 바닥을 맞춰») -- 시장 맥락·옵션 요약 칸이 같은 선에서 끝난다
    const floorY = Math.min(hAll - 2, TREND_ROW_Y + LANE_H);
    const s1Avail = floorY + 12 - mtTop - 4;   // +10 = 시장 맥락 내용 높이의 여유(+6)·칸 아래 여백 -- 마지막 줄 글자가 리본 바닥에 닿게
    const s1H = splitR ? (mcSplit ? Math.max(Math.round(s1Avail * 0.3), mcNeedH ? s1Avail - mcNeedH - 16 : Math.round(s1Avail * 0.47)) : s1Avail) : S1_BELOW ? S1_PANEL : SUB_1S_H - STATS_ROW_H;   // 2단: 오른쪽 칸(다섯 줄이 고르게 나눈다)
    { const fs = el("fpLineSwitch"), card = el("fpCard");   // 청산 밀도 범례(왼쪽 위, ~256px) 오른쪽
      if (fs && card) {
        const on = mcSplit && SUB_LEGEND_H > 0, a = svg.getBoundingClientRect(), c = card.getBoundingClientRect();
        fs.classList.toggle("on", on);
        if (on) { fs.style.left = `${Math.round(a.left - c.left + ml + 290)}px`; fs.style.top = `${Math.round(a.top - c.top + subLegendY + SUB_LEGEND_H / 2)}px`; }
      } }
    mcPlace(svg, mcSplit ? { x: subX, y: sub1sY + s1H + 12, w: subW - 16, h: floorY - (sub1sY + s1H + 12) } : null,
            mcSplit && TRADE_W ? (() => { const y = plotBottom + 12;   // 2026-10-01(4) 프로파일 아래 = 옵션 요약(renderOptions 가 채운다)
              return { x: ml + cw + 4, y, w: TAG_W + TRADE_W + BOOK_W - 8, h: floorY - y }; })() : null);
    supply1sSubBox = {
      svg: subSvg("s1", subX, sub1sY, subW, s1H, sub1sKey(subW, s1H),
                  (g) => renderSupply1s({ svg: g, w: subW, h: s1H })),
      w: subW, h: s1H };

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
    supply1sSubBox = null;
    subPanelCache.s1.key = subPanelCache.dens.key = "";
  }
  // 2026-10-01 프로파일 오른쪽 끝 가격 이름과 겹치는 호가 벽 수량 글자(«11.8k»)는 숨긴다(사용자 지시) -- 층 캐시라 매번 되살린 뒤 다시 판정
  if (svg.id === "candleSvgSnapshot") {
    const far = [...svg.querySelectorAll(".far-lbl")].map((n) => n.getBoundingClientRect()).filter((r) => r.width > 0);
    svg.querySelectorAll(".pf-qty").forEach((q) => {
      q.removeAttribute("visibility");
      if (!far.length) return;
      const b = q.getBoundingClientRect();
      if (far.some((f) => b.left < f.right + 2 && b.right > f.left - 2 && b.top < f.bottom && b.bottom > f.top)) q.setAttribute("visibility", "hidden");
    });
  }
}

function render(state, compactState = null, { stateChanged = true } = {}) {
  latestMainState = state;
  if (!stateChanged) return;

  const sess = state.session || {};
  const micro = state.microstructure || {}, tail = state.tail_risk || {};

  const sessionHtml = buildSessionHtml(sess);
  setH("topSession", sessionHtml);
  
  // 2026-08-25: perf pass -- this whole block (gauge + chart + model-indicator list) only paints
  // anything the user can see while the Snapshot tab is active (snapshotTabPanel is display:none
  // otherwise), so it's gated the same way as tick()'s Snapshot-only fetches above. Data
  // accumulation (pushToneHistory calls above this block, liqDirTone derivation)
  // stays unconditional -- only the paint work below is skipped, so history strips have no gap when
  // the user switches back to Snapshot.
  if (activePageTab === "snapshot") {
    renderLiquidationVolumeGauge();
    renderOfab();                  // 2026-09-25 떠다니는 주문 버튼 글자(포지션·손익)

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

    // 2026-09-30 특화 신호(추세 전환 경보기) 카드 제거(사용자 지시) -- 같은 것이 풋프린트 차트 아래에 있다.

    renderOptions();                          // 2026-09-28 옵션 카드(옛 «신호» 카드의 GEX 한 줄 대체)
  }
}

async function tick() {
  if (document.hidden || tickInFlight) return;
  tickInFlight = true;
  try {
    ensureLiveStream();              // 2026-09-24 수급·상황 밀어주기 (탭/코인/가시성 변화가 여기로 수렴)
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
      refreshBinanceAccount();       // 2026-09-10 청산맵 위 계좌 요약 + 진입선 (자체 30초 게이트)
      refreshChartMarkers();         // 2026-09-09 청산맵 신호 마커
      refreshLiquidation5mSignal();
      refreshLiquidation5mTail();    // 2026-09-25 청산 원 최신 2봉 (자체 2초 게이트)
      refreshLiquidationMap();
      refreshActiveRegime();
      refreshMacroCalendar();
      refreshSessionAlerts();
      refreshFootprint();            // 2026-09-15 볼륨 풋프린트 체결 테이프
      refreshFlowHeatmap();          // 2026-09-19 호가 히트맵(프로파일 왼쪽 절반)
      refreshGex();                  // 2026-09-19 옵션 감마 노출(참고 표시 · 신호 아님)
      refreshKalshi();               // 2026-09-28 칼시 15분 확률 (1초, ETH 만 · 참고 · 신호 아님)
      refreshSupply1s();             // 2026-09-19 최근 5분 x 1초 수급
      refreshOi5m();                 // 2026-09-19 OI 신규계약 5분 누적 (자체 15초 게이트)
      refreshSituation();            // 2026-09-21 상황 읽기 · 30분 (5초, ETH 만)
      refreshTrend();                // 2026-09-28 30분 카드 추세 칸 (60초, 일봉)
      refreshMarketCtx();            // 2026-09-29 시장 맥락 카드 (5초, ETH 만 · 서술)
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
  setInterval(refreshSupply1s, SUPPLY_1S_POLL_MS);   // 수급만 틱(0.5초)보다 빠르게 -- 자체 게이트가 있다
})();
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    disconnectDashboardEvents();
    ensurePriceWs();   // 숨으면 닫는다 -- 백그라운드 탭이 초당 수백 메시지를 받을 이유가 없다
    ensureLiveStream();
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
setupCardRail();

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
  left = Math.max(4, left);   // 2026-09-27 좁은 화면에서 왼쪽 밖으로 나가지 않게
  
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
  const parts = [`<div class="entry-head"><b>${dir} ${escapeHtml(String(plan.quantity))} ${coinUnit()}</b>
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
  // 2026-09-26 막힘(상한 여유 없음·최소 수량)은 평상 상태라 흐린 문구다 -- 빨강은 실패에만.
  if (plan.blocked) parts.push(`<div class="entry-note">진입 불가 — ${escapeHtml(plan.blocked)}</div>`);
  // 2026-09-25 청산맵 TP/SL(사용자 지시) -- 고정 3% 손절을 대신한다. **접지 않는다**: «어디서 닫히나»는
  //   행동을 바꾸는 값이다. 못 거는 경우도 반드시 말한다(옛 손절은 조용히 안 걸려 있었다).
  const br = plan.bracket;
  if (br && br.disabled) {
    parts.push(entryNote("SL/TP 끔 — 이번 진입에 TP·비상 스탑을 걸지 않고 SL 감시도 하지 않습니다. 이미 걸린 SL/TP 는 그대로 둡니다."));
  } else if (br && br.available) {
    const f = (v) => Number(v).toFixed(2);
    const pc = (v) => (v > 0 ? "+" : "") + Number(v).toFixed(1) + "%";
    parts.push(entryNote(
      (br.tp_price ? `TP ${br.tp_name} ${f(br.tp_price)} (${pc(br.tp_pct)}) 지정가` : `TP 없음(${br.tp_name} 없음)`)
      + " · "
      + (br.sl_price ? `SL ${br.sl_name} ${f(br.sl_price)}${br.sl_pct == null ? "" : ` (${pc(br.sl_pct)})`} 5분봉 종가 이탈 시 청산`
                       + ` · 비상 스탑 ${f(br.backstop_price)}`
                     : `🔴SL 없음(${br.sl_name} 없음)`),
      br.sl_price ? "" : "bad"));
  } else if (br) {
    parts.push(entryNote(`🔴TP/SL 을 못 겁니다 — ${br.reason || "청산맵 레벨 없음"}`, "bad"));
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
// 2026-09-28 SL/TP 체크(사용자 지시) -- 해제면 진입에 TP·비상 스탑을 안 걸고 SL 감시도 안 무장, 물타기면 기존 SL/TP 를 그대로 둔다.
//   서버가 sltp=0 을 받으면 계획의 bracket 을 «끔»으로 바꾼다(주문·감시 파일 둘 다 안 건드림). 선택은 브라우저에 기억한다.
const manualSltpOn = () => el("snapSltp")?.checked !== false;
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
    // «자동»은 바로 옆 체크박스가 말한다 -- 값 글자에 또 쓰면 같은 말이 두 번이다(2026-09-25 통일).
    + (locked ? " (기존 포지션과 동일)" : "")
    + (rejected ? ` · 🔴거래소가 거부합니다(포지션 때문에 최소 ${Math.ceil(floor)}배)`
       : low ? ` · ⚠상한만큼 못 엽니다(최소 ${Math.ceil(min)}배)` : "");
  // 🔴같은 줄을 두 번 쓰고 있었다 -- 뒤 줄이 앞 줄을 덮어 **rejected(거래소 거부)가 색을
  //   잃었다**. 둘 중 경고가 더 급한 쪽이 지워지던 셈이라 고친다.
  out.className = (rejected || low) ? "entry-was bad" : "entry-was";
  syncRangeFill(g);
}

// 2026-09-26 SOL·XRP 탭이 생기면서 주문 버튼이 다른 코인 탭에도 보인다 -- 그런데 서버 주문 경로는 **exec_symbol 하나**
//   (ETHUSDC)로만 나간다. SOL 탭에서 누르면 ETH 가 체결된다(데스크톱은 길게 누르면 확인 없이). 탭의 코인이 주문 심볼의
//   코인과 다르면 미리보기·전송 둘 다 거절하고, 화면에서도 조작부를 막는다(CSS body.order-coin-off).
function orderCoinOk() {
  // 2026-09-26 SOL·XRP 주문이 생겼다 -- 서버가 이 코인의 주문 심볼을 주면 주문할 수 있다(계좌 전이면 ETH 만).
  const exec = execSymbolFor(activeSnapshotAsset) || (activeSnapshotAsset === "eth" ? "ETHUSDC" : "");
  return !!exec && exec.startsWith(coinUnit().toUpperCase());
}
function syncOrderCoinGate() {
  const off = !orderCoinOk();
  document.body.classList.toggle("order-coin-off", off);
  const lanes = document.querySelector(".acct-lanes");
  if (lanes) {
    lanes.dataset.orderNote = off ? `이 탭(${coinUnit()})에서는 주문을 낼 수 없습니다 — 주문은 ETH·SOL·XRP 탭에서만 됩니다` : "";
  }
}

async function manualEntryFetch(side, kind = "entry") {
  if (!orderCoinOk()) {
    return { ok: false, error: "order_coin_mismatch",
             detail: `${coinUnit()} 탭에서는 주문하지 않습니다` };
  }
  const q = `&pct=${kind === "exit" ? manualExitPct() : manualEntryPct()}`
    + (kind === "exit" ? "" : manualLevQuery() + (manualSltpOn() ? "" : "&sltp=0"));
  const res = await fetch(`/api/manual-${kind}/preview?side=${side}&asset=${activeSnapshotAsset}${q}`, { cache: "no-cache" });
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
                  ledger: "원장 상한", equity: "순자산 상한", model: "위험 모델",
                  margin: "증거금 상한" };
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
  const parts = [`<div class="entry-head"><b>${side} ${plan.quantity} ${coinUnit()} 청산</b>${of}`
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
  manualPreviewInFlight = true;
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
    manualPreviewInFlight = false;
    manualFireOnPreview = false;   // 실패·막힘으로 발주가 안 됐으면 다음 미리보기로 새지 않게
    if (!manualOrderBusy) manualButtonsDisabled(false);
  }
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
  // 🔴키에는 **카드가 그리는 것 전부**가 들어가야 한다. 2026-09-22 에 `target_leverage` 가
  //   빠져 있어, 레버리지만 바꾸면(서버 투영은 배수에 안 움직이므로) 키가 같아 조기 반환했고
  //   카드가 «레버 20배»에 굳었다. 배수는 화면이 직접 나눠 쓰는 값이라 키에 있어야 한다.
  const key = on ? `${on.after.liq_pct}|${on.after.margin_used_pct}|${on.after.exposure_x}`
                   + `|${plan.quantity}|${plan.target_leverage}|${plan.price}` : "";
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
  const asset = activeSnapshotAsset;
  try {
    // 2026-09-26 양쪽을 본다 -- 위험모델 상한은 방향마다 다르다(롱이 막혀도 숏은 열릴 수 있다).
    const [data, dataShort] = await Promise.all([manualEntryFetch("LONG"), manualEntryFetch("SHORT")]);
    // 🔴2026-09-30 응답 도착 전에 코인이 바뀌었으면 버린다 -- 옛 코인의 계획을 새 코인 카드에 얹지 않는다.
    if (asset !== activeSnapshotAsset) return;
    if (!data.ok) {
      line.className = "entry-note bad";
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
    // 🔴2026-09-26 비평 «상시 빨강은 경보가 아니라 벽지»: 상한에 걸린 건 **평상 상태**라 흐린 문구로 내리고,
    //   막힌 쪽 버튼만 흐리게 한다. 빨강은 조회 실패·상한 꺼짐 같은 진짜 이상에만 남긴다.
    const planShort = (dataShort && dataShort.ok && dataShort.plan) || {};
    const blockedL = plan.blocked, blockedS = planShort.blocked;
    el("snapEntryLong")?.setAttribute("aria-disabled", String(!!blockedL));
    el("snapEntryShort")?.setAttribute("aria-disabled", String(!!blockedS));
    const capNote = (plan.capped || planShort.capped)
      ? ((plan.capped ? plan : planShort).notes || []).find((n) => n.startsWith("요청 ")) || "" : "";
    const blockText = blockedL && blockedS
      ? (blockedL === blockedS ? `진입 불가 — ${blockedL}` : `롱 진입 불가 — ${blockedL} · 숏 진입 불가 — ${blockedS}`)
      : blockedL ? `롱 진입 불가 — ${blockedL}` : blockedS ? `숏 진입 불가 — ${blockedS}` : "";
    line.hidden = !blockText && !ovX && !capNote;
    line.className = "entry-note" + (ovX && !blockText ? " bad" : "");
    line.textContent = blockText
      || (ovX ? `🔴사이징 상한 꺼짐 — 크기 기준이 «순자산 × ${ovX}» 하나뿐입니다` : "") || capNote;
    lastEntryCap = cap && cap.available ? cap : null;
    lastEntryPlan = plan && !plan.blocked ? plan : null;
    setEntryProjPreview(plan);
    // 2026-09-25 «보유 N시간 · 역행 · 최대 N배» 줄은 뺐다(사용자 지시). 레버 게이지는 그대로 맞춘다.
    renderLevGauge(plan);
  } catch (err) {
    line.hidden = false;
    line.className = "entry-note bad";
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
  if (plan.blocked) { manualFireOnPreview = false; return; }
  const pct = Math.round(100 * (plan.fraction ?? 1));
  manualEntryPending = { side, quantity: plan.quantity, kind, pct, asset: activeSnapshotAsset,
                         lev: manualLevEffective(), sltp: manualSltpOn() };
  // 🔴2026-09-26 비평 P0 + 사용자 결정 «길게 누르면 바로 발주»: 0.4초를 채운 뒤 미리보기가 **늦게** 오면
  //   예전엔 확인 버튼이 떴다 -- 네트워크 속도에 따라 한 단계/두 단계가 갈렸다. 채움을 끝낸 사람은 이미
  //   확인했다. 미리보기가 오는 즉시 발주한다(막혔으면 위에서 이미 멈췄다).
  if (manualFireOnPreview) { manualFireOnPreview = false; manualEntrySubmit(); return; }
  // 🔴진입은 «길게 누르기»가 곧 확인이다 -- 확인 버튼을 띄우면 같은 주문이 두 번 나갈 길이
  //   생긴다(누르고 있는 동안 pending 이 잡히므로). 청산은 그대로 버튼으로 확인한다.
  const btn = el("snapEntryConfirm");
  if (manualHoldFire || !btn) return;
  // 모델이 요구하는 최소 청산 비율보다 적게 닫으려 하면 **확인 버튼에** 적는다.
  // 미리보기에만 띄우면 슬라이더를 다시 내린 뒤에는 안 보인다.
  const need = Math.round(100 * ((plan.risk || {}).required_fraction || 0));
  const short = kind === "exit" && need > pct ? ` ⚠한도 복귀엔 ${need}% 필요` : "";
  const base = `확인: ${side === "LONG" ? "롱" : "숏"} ${plan.quantity} ${coinUnit()} `
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
    rows.push(`체결 ${Number(state.filled || 0)} / ${Number(state.quantity)} ${(ASSET_CONFIG[state.asset] || {}).label || coinUnit()}` +
      (state.taker_qty ? ` (테이커 ${Number(state.taker_qty)})` : ""));
  }
  // 2026-09-25 청산맵 TP/SL 결과. 체결이 있는데 결과가 없거나 실패면 크게 말한다 -- 무방비 포지션은 조용하면 안 된다.
  const br = state?.bracket;
  const filledQty = Number(state?.filled || 0);
  if (br && (br.tp || br.backstop)) {
    const leg = (name, x) => !x ? "" : x.placed ? `${name} ${x.price} 걸림` : `🔴${name} 실패 (${x.error || "?"})`;
    rows.push([leg("TP", br.tp), leg("비상 스탑", br.backstop)].filter(Boolean).join(" · "));
  } else if (br && br.disabled) {
    rows.push("SL/TP 끔 — 걸지 않았습니다(기존 SL/TP 는 그대로)");
  } else if (state?.trigger === "bracket_sl") {
    rows.push("SL 이탈 — 메이커 추격 청산");
  } else if (filledQty > 0 && state?.kind !== "exit" && !br && /^filled|failed|error|rejected/.test(state?.phase || "")) {
    rows.push("TP/SL 거는 중…");
  } else if (br) {
    rows.push(`🔴TP/SL 을 못 걸었습니다 (${br?.reason || "시도 기록 없음"})`);
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
    // 진입이 끝나도 TP/SL 은 그 **뒤에** 걸린다 -- 체결이 있는데 결과가 아직 없으면 한 번 더 본다.
    const bracketPending = data.state?.kind !== "exit" && Number(data.state?.filled || 0) > 0
      && !data.state?.bracket;
    if (phase === "submitting" || phase === "working" || bracketPending) {
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
  if (!orderCoinOk() || (pending.asset && pending.asset !== activeSnapshotAsset)) {   // 미리보기 뒤 탭을 바꿨으면 막는다
    manualEntryClearConfirm();
    box.hidden = false;
    box.innerHTML = entryNote(`주문 취소 — 지금 탭(${coinUnit()})은 주문 심볼의 코인이 아닙니다`, "bad");
    return;
  }
  manualEntryClearConfirm();
  manualOrderBusy = true;
  manualButtonsDisabled(true);
  box.hidden = false;
  box.innerHTML = entryNote("주문 전송 중…", "live");
  try {
    const q = `&pct=${pending.pct ?? 100}`
      + (pending.kind === "exit" ? "" : (pending.lev ? `&lev=${pending.lev}` : "") + (pending.sltp === false ? "&sltp=0" : ""));
    const res = await fetch(
      `/api/manual-${pending.kind || "entry"}/submit?side=${pending.side}&asset=${pending.asset || "eth"}&confirm=1${q}`,
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
{
  const box = el("snapSltp");
  try { if (box && localStorage.getItem("manualSltp") === "0") box.checked = false; } catch (e) { /* 저장소 없음 -- 기본 켜짐 */ }
  box?.addEventListener("change", () => {
    try { localStorage.setItem("manualSltp", box.checked ? "1" : "0"); } catch (e) { /* 기억 못 해도 동작은 한다 */ }
    manualEntryClearConfirm();          // 끄고 켠 뒤의 확인 버튼은 옛 선택을 들고 있다 -- 다시 미리본다
    manualEntryRefreshSize();
  });
}
el("snapLevGauge")?.addEventListener("input", () => {
  const out = el("snapLevVal");
  if (out) out.textContent = `${manualLevValue()}배`;
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
let manualPreviewInFlight = false;
let manualFireOnPreview = false;
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
    // 미리보기가 아직 오는 중이면 **오는 즉시** 발주한다(manualEntryArmConfirm) -- 채움을 끝냈으면 확인은 끝났다.
    // 🔴미리보기가 **왔는데** pending 이 없으면 막혔거나 게이트가 꺼진 것이다 -- 상자에 이미 사유가 있다.
    else if (manualPreviewInFlight) {
      manualFireOnPreview = true;
      const box = el("snapEntryResult");
      if (box) { box.hidden = false; box.innerHTML = entryNote("미리보기 받는 중 — 오면 바로 나갑니다.", "live"); }
    }
  }, HOLD_FIRE_MS);
}
[["snapEntryLong", "LONG", "entry"], ["snapEntryShort", "SHORT", "entry"],
 ["snapExitLong", "LONG", "exit"], ["snapExitShort", "SHORT", "exit"]].forEach(([id, side, kind]) => {
  const btn = el(id);
  if (!btn) return;
  // 🔴2026-09-25 사용자 지시: **터치(모바일)는 길게 누르기 발주를 끈다** -- 스크롤하다 버튼 위에서
  //   0.4초가 지나 주문이 나갈 위험. 터치는 탭 → 미리보기 → 확인 버튼(30초) → 확인, 두 번 눌러야 나간다.
  //   탭은 click 으로 받는다 -- 브라우저는 스크롤로 끝난 터치에 click 을 안 보낸다. 마우스는 그대로 길게 누르기.
  //   판정은 **이벤트마다** pointerType 으로 한다(터치 노트북은 마우스·손가락을 둘 다 쓴다).
  let touched = false;
  btn.addEventListener("pointerdown", (e) => {
    touched = e.pointerType === "touch";
    if (touched) return;
    e.preventDefault(); manualHoldStart(btn, side, kind);
  });
  // 2026-09-26 비평 P1: **키보드**(Enter/Space → detail 0 인 click)도 터치와 같은 «미리보기 → 확인» 경로다.
  //   예전엔 키보드로 누르면 아무 반응도 없었다(길게 누르기는 포인터 전용). 마우스 click 은 여전히 무시한다.
  btn.addEventListener("click", (e) => {
    if (!(touched || e.detail === 0) || manualOrderBusy || btn.disabled) return;
    touched = false;
    if (kind === "exit") manualExitPreview(side); else manualEntryPreview(side, "entry");
  });
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
    // 2026-09-26 비평: 선택이 색으로만 보였다(스크린리더 무음) · 잠긴(«자동») 칩이 키보드로는 눌렸다.
    c.setAttribute("aria-pressed", String(on));
    c.disabled = !!inp.disabled;
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

// ── 떠다니는 주문 버튼 (2026-09-25 사용자 지시) ─────────────────────────────────────
// 차트를 보다가 계좌 카드까지 내려가지 않고 바로 주문하려는 것. 사용자 선택 셋:
//   ①진입 + 청산 ②안전장치는 카드와 같게(길게 누르기) ③레버·비율은 **게이지**로 버튼 안에서 바꾼다.
// ⭐새 주문 경로를 만들지 않는다 -- 펼치면 카드의 .acct-lanes 를 **통째로 옮겨 오고** 접으면 되돌린다.
//   핸들러가 전부 el("id") 기반이라(모달 ↔ 카드 이전 때와 같은 방식) 길게 누르기·미리보기·30초 확인·
//   진행 중인 확인 버튼까지 그대로 따라온다. 같은 요소라 카드와 값이 어긋날 수도 없다.
// ⭐게이지는 새로 안 만든다 -- 칩은 숨긴 <input type="range"> 위에 씌운 껍데기라, 패널 안에서만
//   칩을 감추고 그 range 를 드러낸다(styles.css .ofab-panel). 칩은 range 의 input 을 듣고 있으니
//   게이지를 밀면 되돌아간 카드의 칩도 이미 맞춰져 있다.
// 🔴끄는 건 손잡이(⠿)로만 -- 주문 버튼이 «길게 누르기»라 끌기와 한 표면을 쓰면 옮기다 멈춘 순간 발주된다.
const OFAB_POS_KEY = "ofabPos";
const ofab = { open: false, home: null, next: null, x: null, y: null };

function ofabPlace(x, y, save) {
  const box = el("ofab"), bar = box?.querySelector(".ofab-bar");
  if (!box || !bar) return;
  const w = bar.offsetWidth || 120, h = bar.offsetHeight || 44;
  ofab.x = Math.max(8, Math.min(x, innerWidth - w - 8));
  ofab.y = Math.max(8, Math.min(y, innerHeight - h - 8));
  box.style.left = `${ofab.x}px`;
  box.style.top = `${ofab.y}px`;
  // 패널은 **남은 공간이 넓은 쪽**으로 연다 -- 버튼을 어디에 두든 화면 밖으로 안 나간다.
  const up = ofab.y + h / 2 > innerHeight / 2;
  box.classList.toggle("up", up);
  box.classList.toggle("rightside", ofab.x + w / 2 > innerWidth / 2);
  const panel = el("ofabPanel");
  if (panel) panel.style.maxHeight = `${Math.max(160, (up ? ofab.y : innerHeight - ofab.y - h) - 16)}px`;
  if (save) {
    try { localStorage.setItem(OFAB_POS_KEY, JSON.stringify([ofab.x, ofab.y])); } catch (e) { /* 위치 기억은 편의일 뿐 */ }
  }
}

function ofabSetOpen(open) {
  const panel = el("ofabPanel"), lanes = document.querySelector(".acct-lanes");
  if (!panel || !lanes || open === ofab.open) return;
  if (open) {
    ofab.home = lanes.parentNode;
    ofab.next = lanes.nextSibling;
    panel.appendChild(lanes);
  } else if (ofab.home) {
    ofab.home.insertBefore(lanes, ofab.next);
  }
  ofab.open = open;
  panel.hidden = !open;
  el("ofabAway").hidden = !open;
  el("ofabToggle")?.setAttribute("aria-expanded", String(open));
  // 카드에서는 range 가 칩 뒤에 숨어 있어 탭 순서에서 빠져 있다(tabindex -1). 게이지로 드러나면 넣는다.
  lanes.querySelectorAll(".chip-input").forEach((i) => { i.tabIndex = open ? 0 : -1; });
  if (open) ofabApplyDefaults();
  syncAllRangeFills(lanes);
  syncChipsets();
  ofabPlace(ofab.x, ofab.y, false);   // 열리면 위/아래·최대 높이를 다시 정한다
}

// 2026-09-28 사용자 지시: 펼치면 **진입 비율 5% · SL/TP 해제**가 기본. 체크는 저장하지 않는다(카드의 기억된 선택은 그대로 --
//   다음에 카드에서 새로고침하면 기억값으로 돌아간다). 비율은 input 을 쏴서 칩·글자·크기 재조회가 평소 경로로 따라오게 한다.
function ofabApplyDefaults() {
  const fr = el("snapEntryFrac"), sl = el("snapSltp");
  if (sl) sl.checked = false;
  if (fr) { fr.value = "5"; fr.dispatchEvent(new Event("input", { bubbles: true })); }
}
// 2026-09-28 사용자 지시: 버튼을 2초 꾹 누르면 **지금 포지션 방향으로 바로 추가 진입**(5% · SL/TP 해제). 마우스만 -- 터치(모바일)는 없다.
//   패널을 펼쳐 결과를 보여 주고, 미리보기가 오는 즉시 발주한다 -- 진입 버튼의 «길게 누르기 = 확인»과 같은 경로
//   (manualFireOnPreview → manualEntryArmConfirm). 포지션이 없으면 방향을 모르니 안 나간다. 서버 게이트·코인 검사는 그대로.
// 2026-10-01 꾹 눌러 추가 진입은 **패널을 열지 않는다**(사용자 지시) -- 결과는 버튼 위 말풍선(#ofabSay)이 카드의 결과 칸(snapEntryResult)을 따라 말한다.
let ofabSayObs = null, ofabSayTimer = 0;
function ofabSay(html, tone = "") {
  const b = el("ofabSay");
  if (!b) return;
  b.className = `ofab-say${tone ? " " + tone : ""}`;
  b.innerHTML = html; b.hidden = false;
  clearTimeout(ofabSayTimer);
  ofabSayTimer = setTimeout(() => { b.hidden = true; ofabSayObs?.disconnect(); ofabSayObs = null; }, 8000);
}
function ofabQuickAdd() {
  const p = snapshotAccountPosition();
  const side = Number(p?.qty) ? String(p.side || "").toUpperCase() : "";
  ofabApplyDefaults();
  const box = el("snapEntryResult");
  const say = (msg) => ofabSay(escapeHtml(msg), "bad");
  if (box) {   // 카드 결과 칸이 바뀌면(미리보기 → 발주 → 체결/거부) 그 글을 말풍선으로
    ofabSayObs?.disconnect();
    ofabSayObs = new MutationObserver(() => { const t = (box.textContent || "").trim(); if (t && !box.hidden) ofabSay(escapeHtml(t), box.querySelector(".bad") ? "bad" : ""); });
    ofabSayObs.observe(box, { childList: true, subtree: true, characterData: true, attributes: true });
  }
  if (side === "LONG" || side === "SHORT") ofabSay(`${side === "LONG" ? "롱" : "숏"} 5% 추가 진입 — 미리보기 받는 중…`);
  if (side !== "LONG" && side !== "SHORT") return say("추가 진입 안 함 — 열린 포지션이 없어 방향을 모릅니다.");
  if (manualOrderBusy || manualPreviewInFlight) return say("추가 진입 안 함 — 진행 중인 주문·미리보기가 있습니다.");
  clearTimeout(entrySizeDebounce);   // 기본값이 건 크기 재조회는 필요 없다(미리보기가 같은 값을 받는다)
  manualFireOnPreview = true;
  manualEntryPreview(side, "entry");
}

// 탭이 스냅샷일 때만 뜬다(주문 조작부가 사는 탭). 다른 탭으로 가면 조작부를 카드로 먼저 돌려놓는다.
function ofabSync() {
  const box = el("ofab");
  if (!box) return;
  const on = activePageTab === "snapshot";
  if (!on) ofabSetOpen(false);
  box.hidden = !on;
  if (on) ofabPlace(ofab.x ?? ofabDefaultX(), ofab.y ?? innerHeight, false);
}
// 2026-09-26 비평: 모바일 기본 자리는 **하단 가운데**(엄지 영역) -- 오른쪽 아래는 «지금 닫으면»·미실현 타일을 덮었다.
function ofabDefaultX() {
  const w = el("ofab")?.querySelector(".ofab-bar")?.offsetWidth || 220;
  return matchMedia("(pointer: coarse)").matches ? (innerWidth - w) / 2 : innerWidth;
}

// 버튼 글자 = 지금 포지션. 펼치지 않아도 «들고 있나·얼마 벌었나»가 보여야 누를지 정한다.
function renderOfab() {
  const box = el("ofab");
  if (!box || box.hidden) return;
  const p = snapshotAccountPosition();
  const qty = Number(p?.qty) || 0;
  const side = qty ? String(p.side || "").toUpperCase() : "";
  const pnl = ofabLivePnl(p, qty, side);
  // 센트까지 -- 달러 반올림이면 3초 갱신이 작은 움직임에서 안 보인다(계좌 카드 미실현과 같은 자리수).
  // 2026-09-26 사용자 지시: 코인 수량 대신 **증거금 사용 %**(계좌 카드 «증거금 사용» 타일과 같은 값).
  // 🔴2026-10-01 비율도 수익금과 **같은 3초 시세**로(사용자 «업데이트 주기가 비율과 수익금이 안 맞는다») -- 전엔 비율만 30초 계좌 조회값이라
  //   수익금이 움직이는 동안 비율이 멈춰 있다 조회 때 한꺼번에 튀었다. 교차증거금 사용 = 명목 ÷ 레버리지 → 시세 비율만큼, 순자산 = 지갑 + 미실현(같은 3초 값).
  //   ponytail: 사용 증거금 전체를 이 코인 시세 비율로 민다 -- 다른 코인 포지션이 같이 열려 있으면 근사(지금은 ETH 하나), 다음 계좌 조회가 오면 거래소 값으로 돌아간다.
  const bal = latestBinanceAccount?.balance || {}, m = acctMarginUsed(bal), live = Number(latestLivePriceByAsset[activeSnapshotAsset] || 0);
  const k = qty && live > 0 && ofabPnlRef.price > 0 ? live / ofabPnlRef.price : 1;
  const eqLive = (Number(bal.wallet) || 0) + (Number(bal.unrealized) || 0) + (pnl - (Number(p?.unrealized_pnl) || 0));
  const used = eqLive > 0 ? (m.used * k) / eqLive * 100 : m.pct;
  setT("ofabPos", qty ? `${side} ${used.toFixed(0)}% · ${pnl >= 0 ? "+" : "−"}$${Math.abs(pnl).toFixed(2)}` : "주문");
  box.classList.toggle("long", side === "LONG");
  box.classList.toggle("short", side === "SHORT");
}
// 🔴2026-09-26 사용자 지시 «미실현손익 3초 갱신». 계좌 조회는 30초라 그 사이 숫자가 멈춰 있었다. 거래소를 3초마다
//   부르는 대신 **거래소 값 + 그 뒤 시세 변화 × 수량**으로 민다. 기준 시세는 계좌 스냅샷이 바뀐 순간의 실시간 시세라
//   심볼 차이(주문 ETHUSDC · 시세 ETHUSDT, ~1.7bp)는 변화분에서 지워진다. 다음 계좌 조회가 오면 거래소 값으로 되돌아간다.
// 🔴2026-09-30 사용자 «플로팅 버튼 미실현이 실제와 안 맞는다»: 기준 시세를 «계좌 스냅샷이 브라우저에 도착한 순간»으로 잡았는데,
//   거래소 손익은 서버가 조회한 시각(generated_at) 값이고 서버 캐시 때문에 도착까지 30초 넘게 늦다(실측 33.8초).
//   그 사이 움직임이 통째로 빠져 숏 1.6 ETH 면 $2 움직임에 $3 틀렸다. 기준 = generated_at 초의 시세(1초 수급 칸의 마지막가).
const ofabPnlRef = { acct: null, price: 0 };
function ofabPriceAt(sec) {                     // 그 초(없으면 5초 안 직전 초)의 마지막 체결가 -- 1초 수급 칸 [.., 가격]
  for (let s = sec; s > sec - 5; s -= 1) { const r = supply1s.get(s); if (r && r[6] > 0) return r[6]; }
  return 0;
}
function ofabLivePnl(p, qty, side) {
  const base = Number(p?.unrealized_pnl) || 0;
  const live = Number(latestLivePriceByAsset[activeSnapshotAsset] || 0);
  if (!qty || !(live > 0)) return base;
  if (ofabPnlRef.acct !== latestBinanceAccount) {
    ofabPnlRef.acct = latestBinanceAccount;
    const g = Math.floor(Date.parse(latestBinanceAccount?.generated_at || "") / 1000);
    ofabPnlRef.price = (Number.isFinite(g) && ofabPriceAt(g)) || live;   // 기록이 없는 코인·재연결 직후는 옛 방식
  }
  return base + (live - ofabPnlRef.price) * qty * (side === "SHORT" ? -1 : 1);
}
setInterval(renderOfab, 3000);

(() => {
  const grip = el("ofabGrip"), tgl = el("ofabToggle");
  if (!grip || !tgl) return;
  // 🔴2026-09-26 비평: 버튼 폭은 배치 **뒤에** 바뀐다 -- 글자(«주문» 94px → 포지션 214px)와 웹폰트 로딩(195 → 206px).
  //   첫 배치 폭으로 clamp 한 자리에서 오른쪽이 최대 112px 화면 밖으로 잘렸다. 크기 변화 자체를 따라가 다시 잡는다.
  if (window.ResizeObserver) {
    new ResizeObserver(() => {
      if (!el("ofab").hidden) ofabPlace(ofab.x ?? ofabDefaultX(), ofab.y ?? innerHeight, false);
    }).observe(el("ofab").querySelector(".ofab-bar"));
  }
  // 2026-09-27 사용자 «주문 움직이는 버튼은 상시 띄워줘» -- 계좌 카드 앞에서 접던 관찰자(09-26)를 걷었다.
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(OFAB_POS_KEY) || "null"); } catch (e) { saved = null; }
  // 기본 자리 = 오른쪽 아래(엄지가 닿는 곳). ofabPlace 가 화면 안으로 끌어넣는다.
  [ofab.x, ofab.y] = Array.isArray(saved) ? saved : [innerWidth, innerHeight - 24];
  // 짧게 = 펼치기/접기 · 1.5초 꾹 = 추가 진입(ofabQuickAdd). 8px 넘게 움직이면(스크롤·끌기) 취소 -- 터치 스크롤은 pointercancel 로도 끊긴다.
  const OFAB_HOLD_MS = 1500;   // 2026-09-28 사용자 지시 0.5 -> 2초 · 2026-09-30 «너무 느리다» -> 1.5초 (styles.css .ofab-toggle.holding 채움 시간과 같이)
  let hold = null, held = false;
  const holdEnd = () => { if (hold) { clearTimeout(hold.t); hold = null; } tgl.classList.remove("holding"); };
  tgl.addEventListener("pointerdown", (e) => {
    holdEnd(); held = false;
    // 🔴2026-09-28 사용자 지시 «모바일에서는 넣으면 안돼» -- 터치는 꾹 누르기 발주가 없다(짧게 = 펼치기만).
    //   진입 버튼과 같은 판정(이벤트마다 pointerType) -- 터치 노트북의 마우스는 그대로 된다.
    if (e.pointerType === "touch") return;
    hold = { x: e.clientX, y: e.clientY, t: setTimeout(() => { hold = null; held = true; tgl.classList.remove("holding"); ofabQuickAdd(); }, OFAB_HOLD_MS) };
    tgl.classList.add("holding");
  });
  tgl.addEventListener("pointermove", (e) => { if (hold && Math.hypot(e.clientX - hold.x, e.clientY - hold.y) > 8) holdEnd(); });
  ["pointerup", "pointerleave", "pointercancel"].forEach((ev) => tgl.addEventListener(ev, holdEnd));
  tgl.addEventListener("contextmenu", (e) => e.preventDefault());   // 모바일 길게 누르기 메뉴
  tgl.addEventListener("click", () => { if (held) { held = false; return; } ofabSetOpen(!ofab.open); });
  el("ofabBack")?.addEventListener("click", () => ofabSetOpen(false));
  grip.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    grip.setPointerCapture(e.pointerId);
    const dx = e.clientX - ofab.x, dy = e.clientY - ofab.y;
    const move = (ev) => ofabPlace(ev.clientX - dx, ev.clientY - dy, false);
    const end = () => {
      grip.removeEventListener("pointermove", move);
      grip.removeEventListener("pointerup", end);
      grip.removeEventListener("pointercancel", end);
      ofabPlace(ofab.x, ofab.y, true);
    };
    grip.addEventListener("pointermove", move);
    grip.addEventListener("pointerup", end);
    grip.addEventListener("pointercancel", end);
  });
  // 끌 수 없는 사람도 옮길 수 있게 -- 방향키 10px, Shift 는 40px.
  grip.addEventListener("keydown", (e) => {
    const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
    if (!d) return;
    e.preventDefault();
    const s = e.shiftKey ? 40 : 10;
    ofabPlace(ofab.x + d[0] * s, ofab.y + d[1] * s, true);
  });
  // ESC 는 패널 안에 초점이 있을 때만 -- 다른 곳의 ESC(모달 닫기 등)를 가로채지 않는다.
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && ofab.open && el("ofab").contains(document.activeElement)) {
      ofabSetOpen(false);
      tgl.focus();
    }
  });
  addEventListener("resize", () => { if (!el("ofab").hidden) ofabPlace(ofab.x, ofab.y, false); });
  ofabSync();
})();

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

// 2026-09-22 시안 B: KPI 타일 「지금 닫으면」과 청산 레인이 **같은 식**을 써야 한다.
//   두 군데서 따로 계산하면 반올림 한 자리만 달라져도 «어느 쪽이 맞나»가 된다.
function exitNetUsd(pos, pct) {
  const qty = Number(pos.qty) || 0, mark = Number(pos.mark_price) || 0;
  const entry = Number(pos.entry_price) || 0, dir = pos.side === "LONG" ? 1 : -1;
  const close = qty * pct / 100;
  const full = Number.isFinite(Number(pos.unrealized_pnl)) ? Number(pos.unrealized_pnl)
    : (entry > 0 && mark > 0 ? qty * (mark - entry) * dir : null);
  const gross = full === null ? null : full * pct / 100;
  const fee = mark > 0 ? close * mark * EXIT_FEE_BP_PEG / 10000 : 0;
  return { close, gross, fee, net: gross === null ? null : gross - fee,
           move: entry > 0 && mark > 0 ? (mark - entry) / entry * 100 * dir : null };
}

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
    // 청산 수수료만 뺀다. 진입 수수료는 **이미 지갑에서 빠져 나갔으므로**, 여기 숫자가
    // «지금 닫으면 지갑이 얼마 늘어나는가»와 일치하려면 빼면 안 된다(title 에 적어 둔다).
    const { close, gross, fee, net, move } = exitNetUsd(pos, pct);
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
  const market = ASSET_CONFIG[activeSnapshotAsset]?.symbol || `${activeSnapshotAsset.toUpperCase()}USDT`;
  const execSym = execSymbolFor(activeSnapshotAsset);     // 2026-09-26 탭 코인의 포지션만(SOL·XRP 주문)
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
  // 2026-09-25 사용자 지시 «추가 진입을 드롭다운으로 만들지 말고 상시 표시» -- 09-14 물타기 접힘
  //   폐지. <details open> 은 그대로 두고(안쪽 CSS·검사가 [open] 에 걸려 있다) 제목 줄 클릭만 막는다.
  if (!entryFoldToggleBound) {
    entryFoldToggleBound = true;
    el("snapEntrySummary")?.addEventListener("click", (e) => e.preventDefault());
  }
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
