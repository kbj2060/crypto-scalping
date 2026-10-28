const API_EVENTS_URL = "/api/events";
const API_OPS_STATUS_URL = "/api/ops-status";
const API_BINANCE_ACCOUNT_URL = "/api/binance-account";
const API_LIQUIDATION_MAP_URL = "/api/liquidation-map";
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
const CANDLE_HISTORY_POLL_MS = 300000;
// Kept post-Live-tab-removal solely as the SSE ticker payload's asset allowlist (see
// applyDashboardEvent()) -- eth/btc still need their live price tracked for the Snapshot tab's own
// coin switcher (activeSnapshotAsset), sol is tracked too for parity even though nothing reads it.
// 2026-09-26 dp = 화면 가격 소수 자릿수.
//   🔴데이터 심볼은 전부 USDT 시장이다(주문만 USDC -- 사용자 규칙).
const ASSET_CONFIG = {
  eth: { label: "ETH", symbol: "ETHUSDT", dp: 1 },
  sol: { label: "SOL", symbol: "SOLUSDT", dp: 2 },
  btc: { label: "BTC", symbol: "BTCUSDT", dp: 0 },
  xrp: { label: "XRP", symbol: "XRPUSDT", dp: 4 },
};
// 풋프린트·1초 수급·OI·호가 히트맵을 서버가 실시간으로 만드는 코인(server.py FLOW_SPECS 와 같은 목록).
const FLOW_ASSETS = new Set(["eth", "sol", "xrp"]);
const flowOn = () => FLOW_ASSETS.has(activeSnapshotAsset);
const coinUnit = () => (ASSET_CONFIG[activeSnapshotAsset] || {}).label || String(activeSnapshotAsset).toUpperCase();
const pxDp = () => { const d = (ASSET_CONFIG[activeSnapshotAsset] || {}).dp; return Number.isFinite(d) ? d : 1; };

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
let chartPanFrac = 0;   // 2026-10-20 끌기 1:1(사용자 «봉 단위로 뚝뚝»): 오른쪽 끝 = chartPanEnd 봉 다음 + 이 소수 봉(0~1) -- chartPanEdge
// 2026-10-21 사용자 «맨 마지막 봉을 자세히 보려면 오른쪽 끝에서 막히면 안 돼»: 실시간일 때 최신 봉 오른쪽에 빈 칸(봉 수, 소수).
//   최신 봉은 계속 따라가고(chartPanEnd = null 그대로) 빈 칸만 남는다 -- 트레이딩뷰처럼. 최대 = 창의 80%(봉이 2~3개는 늘 보이게).
let chartPanRight = 0;
const chartPanRightMax = () => chartWindowBars * 0.8;
// 2026-10-09 세로 이동(사용자 «모든 차트에서 마우스로 위아래로도»): 가격축을 이만큼(가격 단위) 올린다 · 0 = 자동 축. «최신으로»가 함께 0 으로.
let chartPanY = 0, chartYPerPx = 0;
let footprintLastFetchAt = 0;
// ── 창 토글 (2026-09-19 사용자 요청: 1h/2h/4h) ─────────────────────────────
// 풋프린트와 수급 프로파일이 **같은 창**을 쓴다. 한 카드 안의 위아래 두 그림이 서로 다른
// 구간을 말하면 읽는 사람이 속는다 -- 그래서 토글도 하나다.
// 서버 링은 24시간(288봉)이라 4h 도 이미 쌓여 있는 데이터다. 더 긴 창을 안 주는 건 값이
// 없어서가 아니라, 48봉이면 풋프린트 셀이 25px 라 숫자가 이미 안 들어가서다.
// 2026-10-20 사용자 «1h 만 남기고 휠로 축소하면 체결 박스 없이 일반 캔들, 확대하거나 1h 를 누르면 풋프린트»:
//   새로 열면 늘 12봉(줌은 저장 안 함 -- 열자마자 캔들이면 «풋프린트가 사라졌다»로 읽힌다).
// 2026-10-20 사용자 «5m, 30m, 1h 모두 데이터를 일주일 치»: 창은 tf 마다(그 tf 봉 수) 1주일까지 -- 12봉 = 풋프린트(5m 1h · 30m 6h · 1h 12h).
//   라이브 12시간 밖은 /api/footprint-week 한 번(아래 maybeFetchFpWeek)으로 채운다.
// 2026-10-20 사용자 «바이낸스·트레이딩뷰처럼 부드럽게»: 휠 단계를 없앴다 -- 창 = 소수 봉 수(12봉 ~ 1주일), 휠마다 연속으로 커서 자리 기준 줌(실시간이면 오른쪽 끝 기준 -- chartZoomAt).
//   체결 박스는 창이 아니라 **봉 폭**이 정한다(renderCandleSvg 의 fpThin: 칸 숫자가 들어가는 폭이면 박스).
const CHART_WINDOW_MAX = { 1: 2016, 6: 336, 12: 168 };   // tf 마다 1주일(5m · 30m · 1h 봉 수)
let chartWindowBars = 12;
const fpZoomed = () => chartWindowBars > 12;   // 넓힌 창(1주일 받기 · 창 칸 불 끄기) -- 박스 유무는 fpBoxesOff
let fpBoxesOff = false;   // 마지막 렌더가 박스 없이 캔들로 그렸나(머리 «· 캔들»)
// 2026-10-20 사용자 «5m과 같은 구성으로 30m과 1h 도»: 5분봉 k 개를 UTC 경계로 묶은 봉(12봉 = 6시간·12시간 -- 서버 풋프린트 144봉 안).
//   렌더러는 그대로 두고 **입력만** 묶는다(캔들·풋프린트 칸·OI·청산 · tfCandles 등). 휠 줌·가로 이동·미니맵도 5m 과 같다(10-20 1주일).
let chartTfK = 1;   // 1 = 5m · 6 = 30m · 12 = 1h
const tfT = (t) => (chartTfK > 1 ? t - (t % (300 * chartTfK)) : t);   // 5분봉 시각(초) -> 그 봉이 든 tf 봉 시각
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
// ⚠️바이낸스가 OI 를 3~7초에 한 번만 갱신한다(실측) -- 점이 초마다
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
let bybitLiq1s = new Map(), bybitLiq1sSince = 0;   // 2026-10-03 Bybit 청산 레인(서버가 따로 보내고 여기서 더한다)
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
// 들어오면서(2026-09-19) 「보려고 커서를 올리면 1초 차트가 멈춘다」가 됐다. (①은 2026-10-20 없앴다 -- 호버 중에도 그리고 커서를 재생한다, chartHoverPt)
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
const sub1sKey = (w, h) => `${supply1sVer}|${w}|${h}|${liqBadgeKey()}`;
// 2026-10-04 CVD·OI 제목 옆 청산 급증 배지가 바뀌면 다시 그린다(쪽 + $10만 단위 금액). 시장 맥락이 아직 없으면 빈 키.
function liqBadgeKey() {
  try {
    const bu = latestMarketCtx && latestMarketCtx.burst, side = mcLiqBurstSide(bu);
    return side ? side + Math.round(((side === "long" ? bu.long_usd_60s : bu.short_usd_60s) || 0) / 1e5) : "";
  } catch (err) { return ""; }   // 스크립트 초기화 중(let 초기화 전) 호출 대비
}
let measureCtx2d = null;
function measureTextW(txt, font) {   // SVG 글자 폭(배지 자리 계산) -- 붙이기 전 텍스트는 getComputedTextLength 가 0 이라 캔버스로 잰다
  if (!measureCtx2d) measureCtx2d = document.createElement("canvas").getContext("2d");
  measureCtx2d.font = font;
  return measureCtx2d.measureText(txt).width;
}

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
// 2026-10-03 청산 원 툴팁의 거래소별 내역(사용자 «거래소 0건인데 왼쪽에 금액») -- 금액이 있는 거래소만 «이름 $금액 N건(롱/숏)».
//   바이낸스 = 합 − OKX − Bybit − HL(서버가 바이낸스를 바탕으로 나머지를 더한다). 그 봉을 본 거래소 목록은 끝에 «합산:».
//   순수 함수 -- test/test_market_ctx_patterns_20260929.mjs 가 본문을 떼어 돌린다.
function liqVenueText(b) {
  const u = (x) => (Number(x && x.long_usd) || 0) + (Number(x && x.short_usd) || 0);
  const parts = [["OKX", b.okx_detail], ["Bybit", b.bybit], ["HL 고래", b.hl]];
  const bnN = Math.max(0, (Number(b.events) || 0) - parts.reduce((a, [, x]) => a + ((x && x.n) || 0), 0));
  const rows = [["바이낸스", { long_usd: (Number(b.long_usd) || 0) - parts.reduce((a, [, x]) => a + (Number(x && x.long_usd) || 0), 0),
                                short_usd: (Number(b.short_usd) || 0) - parts.reduce((a, [, x]) => a + (Number(x && x.short_usd) || 0), 0), n: bnN }], ...parts]
    .filter(([, x]) => x && u(x) >= 0.5)
    .map(([nm, x]) => `${nm} ${fmtUsdCompact(u(x))} ${x.n}건(롱 ${fmtUsdCompact(Math.max(0, x.long_usd || 0))}/숏 ${fmtUsdCompact(Math.max(0, x.short_usd || 0))})`);
  const seen = ["바이낸스", b.okx ? "OKX" : null, b.bybit ? "Bybit" : null, b.hl ? "HL 고래(추적 300지갑)" : null].filter(Boolean);
  return (rows.length ? "\n" + rows.join("\n") : "") + "\n합산: " + seen.join(" · ");
}

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
  [liq1s, okxLiq1s, bybitLiq1s].forEach((m) => m.forEach((c, sec) => {
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
// 2026-10-10 풋프린트 체결 기둥 «포지션» 보기(사용자 «체결 | 포지션 토글 · A + C 미니맵 · 1초») -- /api/position-profile.
//   최근 1·7·30일에 열려 아직 남은 계약(추정 · 회전+나이 가중 닫기 · 10-22 롱·숏 분리 근사, dashboard/position_profile.py). 지도로만, 신호 아님.
const API_POSITION_PROFILE_URL = "/api/position-profile";
let latestPositionProfile = null, posProfileFetchAt = 0, posProfileInflight = false;
// 2026-10-28 펜 메모 «체결/청산(왼쪽)과 포지션/호가(오른쪽)를 버튼으로»: 기둥 둘을 따로 고른다 -- 옛 fpCol(trade|pos|liq) 저장값을 이어받는다
const fpColLoad = (key, old, alt, dflt) => { try { const v = localStorage.getItem(key) || (localStorage.getItem("fpCol") === old ? alt : dflt); return v === alt ? alt : dflt; } catch (e) { return dflt; } };
let fpColL = fpColLoad("fpColL", "liq", "liq", "trade"), fpColR = fpColLoad("fpColR", "pos", "pos", "book");
let fpPosWin = (() => { try { const v = Number(localStorage.getItem("fpPosWin")); return v === 1 || v === 30 ? v : 7; } catch (e) { return 7; } })();
// 2026-10-27 «청산» 기둥 기간 -- 옛 청산맵 보기의 1d·7d·30d 저장값(liqDays)을 그대로 잇는다
let fpLiqWin = (() => { try { const v = Number(localStorage.getItem("liqDays")); return v === 7 || v === 30 ? v : 1; } catch (e) { return 1; } })();
// 토글은 SVG 가 아니라 **HTML 버튼**을 차트 위 같은 자리에 띄운다(10-10 검토): 차트는 매초 통째로 다시 그려져서
//   SVG 버튼이면 포커스가 사라지고 누르는 순간 교체되면 키 입력이 빠졌다. 이 상자는 한 번 만들고 자리·상태만 바꾼다.
function fpColTogEl(svg) {
  let el = document.getElementById("fpColTog");
  if (el || !svg.parentElement) return el;
  el = document.createElement("div");
  el.id = "fpColTog"; el.className = "fp-col-tog"; el.setAttribute("role", "toolbar"); el.setAttribute("aria-label", "체결 기둥 보기");
  const wins = (side) => [1, 7, 30].map((d) => `<button type="button" data-win="${d}" data-side="${side}">${d}d</button>`).join("");
  el.innerHTML = `<span class="fp-col-seg fp-col-win l" role="group" aria-label="청산 기간">${wins("l")}</span>`
    + '<span class="fp-col-seg" role="group" aria-label="왼쪽 기둥"><button type="button" data-col="trade">체결</button><button type="button" data-col="liq">청산</button></span>'
    + '<span class="fp-col-seg" role="group" aria-label="오른쪽 기둥"><button type="button" data-col="pos">포지션</button><button type="button" data-col="book">호가</button></span>'
    + `<span class="fp-col-seg fp-col-win r" role="group" aria-label="포지션 기간">${wins("r")}</span>`
    + '<span class="fp-col-seg" role="group" aria-label="SLTP 선 주문"><button type="button" data-lines="1" title="차트 위 익절·손절 선을 끌어 놓고 peg 메이커로 진입(자동 2×) — 청산도 둘 다 메이커">SLTP</button></span>';
  el.addEventListener("pointerdown", (e) => { e.stopPropagation(); if (fpIndOpen) fpIndSet(false); });   // 차트 끌기로 번지지 않게 · 열린 지표 팝업은 닫는다
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    e.stopPropagation();
    if (b.dataset.lines) { lineToolToggle(); b.setAttribute("aria-pressed", String(!!lineTool)); return; }
    if (b.dataset.col) setFpCol(b.dataset.col); else setFpCol(null, Number(b.dataset.win), b.dataset.side);
  });
  svg.parentElement.appendChild(el);
  return el;
}
function placeFpColTog(svg, xr, yTop, show) {   // xr·yTop = viewBox 좌표(체결 기둥 오른쪽 끝 · 머리 줄 위)
  const el = fpColTogEl(svg);
  if (!el) return;
  if (el.hidden !== !show) el.hidden = !show;
  if (!show) return;
  el.querySelectorAll("[data-col]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.col === fpColL || b.dataset.col === fpColR)));
  el.querySelectorAll("[data-win]").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.win) === (b.dataset.side === "l" ? fpLiqWin : fpPosWin))));
  el.querySelector("[data-lines]")?.setAttribute("aria-pressed", String(!!lineTool));
  el.querySelector(".fp-col-win.l").hidden = fpColL !== "liq"; el.querySelector(".fp-col-win.r").hidden = fpColR !== "pos";
  const m = svg.getScreenCTM(), pr = el.parentElement.getBoundingClientRect();
  if (!m) return;
  const L = Math.round(m.a * xr + m.e - pr.left) + "px", T = Math.round(m.d * yTop + m.f - pr.top) + "px";
  if (el.style.left !== L) el.style.left = L;
  if (el.style.top !== T) el.style.top = T;
}
function setFpCol(mode, win, side) {
  if (mode === "trade" || mode === "liq") fpColL = mode; else if (mode === "pos" || mode === "book") fpColR = mode;
  if (win) { if (side === "l") fpLiqWin = win; else fpPosWin = win; }
  try { localStorage.setItem("fpColL", fpColL); localStorage.setItem("fpColR", fpColR); localStorage.setItem("fpPosWin", String(fpPosWin)); localStorage.setItem("liqDays", String(fpLiqWin)); } catch (e) { /* 저장 못 해도 동작 */ }
  if (fpColR === "pos" && (mode === "pos" || (win && side !== "l"))) { posProfileFetchAt = 0; refreshPositionProfile(); }
  hideTooltip(); scheduleSnapshotChartRender();
}
// ── 2026-10-28 차트 «선 주문»(사용자 «차트 위 라인 2개를 끌어 위치를 잡고 진입 → peg 메이커 · 익절/손절도 둘 다 메이커 · 닿으면 닫기 · 자동 2×»)
//   시안 A 꼬리표 + B 포지션 상자(꼬리표 = 선 가운데). 선 = USDT 차트 가격 → 서버 미리보기·주문에 tp·sl 로(같은 assemble_entry_plan).
//   익절 = 체결 직후 post-only 지정가 · 손절 = 서버가 1초 체결가 닿음 판정 → peg 청산(마감 뒤 남은 수량 시장가) · 거래소 스탑 없음.
//   상자·꼬리표는 차트가 매초 다시 그려도 남는 HTML 층(fpColTog 와 같은 이유) -- 자리만 renderLineTool 이 맞춘다.
let lineTool = null;   // {tp, sl, plan, err, arm(터치 확인 마감 ms)} · null = 꺼짐
let lineDrag = null, lineHold = 0, linePrevT = 0, linePt = "";
const lineSide = (t, p) => (t.tp > p && t.sl < p ? "LONG" : t.tp < p && t.sl > p ? "SHORT" : "");
const lineDig = (v) => (v >= 100 ? 2 : 4);   // XRP(~2.5) 는 소수 넷째 자리 -- 둘째 자리로 자르면 선이 0.2% 옮겨진 채 나간다(10-28 검증)
const lineFmt = (v) => Number(v).toLocaleString("en-US", { minimumFractionDigits: lineDig(v), maximumFractionDigits: lineDig(v) });
const linePx = (v) => v.toFixed(lineDig(v));
function lineToolToggle() {
  if (lineTool) { lineTool = null; renderLineTool(); return; }
  const G = chartGeo, p = Number(latestLivePriceByAsset[activeSnapshotAsset] || 0);
  if (!G || !(G.ymax > G.ymin) || !(p > 0)) return;
  lineTool = { tp: p + 0.6 * (G.ymax - p), sl: p - 0.6 * (p - G.ymin), plan: null, err: "", asset: activeSnapshotAsset };
  renderLineTool(); linePreview();
}
function linePreview() {   // 크기는 서버 미리보기(자동 2×)가 정한다 -- 끌기를 멈추고 0.5초 뒤 한 번
  clearTimeout(linePrevT);
  linePrevT = setTimeout(async () => {
    const t = lineTool, p = Number(latestLivePriceByAsset[activeSnapshotAsset] || 0), side = t && lineSide(t, p);
    if (!t) return;
    if (!side) { t.plan = null; t.err = "익절과 손절이 현재가 양쪽에 있어야 합니다"; renderLineTool(); return; }
    try {
      const d = await manualEntryFetch(side, "entry", { rule: "c", tp: linePx(t.tp), sl: linePx(t.sl) });
      if (lineTool !== t) return;
      t.plan = d.ok ? d.plan : null; t.err = d.ok ? (d.plan.blocked || "") : (d.detail || d.error || "미리보기 실패");
    } catch (e) { if (lineTool === t) { t.plan = null; t.err = "미리보기 실패"; } }
    renderLineTool();
  }, 500);
}
function lineFire() {   // 미리보기 → 오는 즉시 발주(manualFireOnPreview) -- 진입 버튼과 같은 길(상한·게이트·확인 서버 검사 그대로)
  const t = lineTool, p = Number(latestLivePriceByAsset[activeSnapshotAsset] || 0), side = t && lineSide(t, p);
  if (!side || manualOrderBusy || manualPreviewInFlight) return;
  ofabWatchResult(); ofabSay(escapeHtml(`${side === "LONG" ? "롱" : "숏"} 선 주문 — 미리보기 받는 중…`));
  manualFireOnPreview = true;
  manualEntryPreview(side, "entry", { rule: "c", tp: linePx(t.tp), sl: linePx(t.sl), fresh: true });
  lineTool = null; renderLineTool();
}
function lineToolEl(box) {
  let h = document.getElementById("lineTool");
  if (h || !box) return h;
  h = document.createElement("div"); h.id = "lineTool"; h.className = "lt";
  h.innerHTML = '<div class="lt-zone up"></div><div class="lt-zone dn"></div><i class="lt-ln tp"></i><i class="lt-ln sl"></i><i class="lt-ln en"></i>'
    + '<button type="button" class="lt-tag tp" data-k="tp"></button><button type="button" class="lt-tag sl" data-k="sl"></button><span class="lt-tag en"></span>'
    + '<div class="lt-tk"><div class="lt-row"><b class="lt-side"></b><span class="lt-qty"></span></div><div class="lt-row lt-rr"></div>'
    + '<button type="button" class="lt-go"></button><div class="lt-note"></div></div>'
    + '<button type="button" class="lt-x">선 지우기</button>';
  h.addEventListener("pointerdown", (e) => {
    e.stopPropagation();   // 차트 끌기로 번지지 않게
    linePt = e.pointerType;
    const tag = e.target.closest(".lt-tag[data-k]"), go = e.target.closest(".lt-go");
    if (tag && lineTool) { lineDrag = { k: tag.dataset.k, id: e.pointerId }; tag.setPointerCapture(e.pointerId); e.preventDefault(); }
    if (go && e.pointerType === "mouse" && !go.disabled) {   // 마우스 = 1.5초 꾹(떠 있는 버튼과 같은 규약)
      go.classList.add("hold"); lineHold = setTimeout(() => { go.classList.remove("hold"); lineFire(); }, 1500);
    }
  });
  h.addEventListener("pointermove", (e) => {
    if (!lineDrag || e.pointerId !== lineDrag.id || !lineTool) return;
    const svg = el("candleSvgSnapshot"), G = chartGeo, m = svg && svg.getScreenCTM();
    if (!m || !G) return;
    const yv = (e.clientY - m.f) / m.d;   // 화면 → viewBox y(화면 밖으로 끌어도 같은 비율로 이어진다)
    lineTool[lineDrag.k] = Math.max(0.01, G.ymax - (yv - G.mt) * (G.ymax - G.ymin) / (G.pb - G.mt));
    lineTool.plan = null; lineTool.arm = 0; renderLineTool();
  });
  const end = (e) => {
    if (lineDrag && e.pointerId === lineDrag.id) { lineDrag = null; linePreview(); }
    clearTimeout(lineHold); h.querySelector(".lt-go")?.classList.remove("hold");
  };
  h.addEventListener("pointerup", end); h.addEventListener("pointercancel", end); h.addEventListener("pointerleave", end);
  h.querySelector(".lt-go").addEventListener("pointerleave", end);   // 꾹 누르다 버튼 밖으로 나가면 취소(표 안에서도)
  h.addEventListener("click", (e) => {
    e.stopPropagation();
    if (e.target.closest(".lt-x")) { lineTool = null; renderLineTool(); return; }
    const go = e.target.closest(".lt-go");
    if (!go || go.disabled || !lineTool || (linePt === "mouse" && e.detail)) return;   // 마우스는 꾹(위) -- 터치·키보드 = 두 번(확인 30초, 09-25 부터 터치 꾹 발주 없음)
    if (Date.now() < (lineTool.arm || 0)) { lineFire(); return; }
    lineTool.arm = Date.now() + CONFIRM_WINDOW_MS; renderLineTool();
    setTimeout(renderLineTool, CONFIRM_WINDOW_MS + 50);
  });
  box.appendChild(h);
  return h;
}
function fpMobRowSet(mob, card, fs) {   // 휴대폰 머리 줄에 SLTP·지표 버튼을 옮기거나(mob) 제자리(카드)로 되돌린다 -- 노드째 옮겨 리스너·팝업 상태가 따라간다
  const row = el("fpMobRow");
  if (!row || !card || !fs) return;
  if (row.hidden !== !mob) row.hidden = !mob;
  card.classList.toggle("fp-mobrow", mob);
  fs.classList.toggle("mob", mob);
  if (!fs._home) fs._home = fs.parentElement;
  if (mob) {
    const lb = fpLineBtnEl(card);
    if (lb.parentElement !== row) row.append(lb);
    if (fs.parentElement !== row) row.append(fs);
  } else {
    const lb = el("fpLineBtn");
    if (lb && lb.parentElement === row) card.append(lb);
    if (fs.parentElement === row) fs._home.prepend(fs);
  }
}
function fpTopRowOff() {   // 넓은 화면 차트 위 줄(미니맵·SLTP·지표)을 걷는다 -- renderCandleSvg 의 on=false 분기와 같은 일
  const card = el("fpCard"), fs = el("fpLineSwitch"), pan = el("chartPan"), home = el("fpPanHome");
  fpMobRowSet(false, card, fs);
  card?.classList.remove("fp-toprow"); fs?.classList.remove("on");
  if (pan && pan.classList.contains("inrow")) { if (home && pan.previousElementSibling !== home) home.after(pan); pan.classList.remove("inrow"); pan.style.top = pan.style.left = ""; }
}
function fpLineBtnEl(card) {
  let b = document.getElementById("fpLineBtn");
  if (b) return b;
  b = document.createElement("button");
  b.type = "button"; b.id = "fpLineBtn"; b.className = "fp-line-btn"; b.setAttribute("aria-pressed", String(!!lineTool));
  b.title = "차트 위 익절·손절 선을 끌어 놓고 peg 메이커로 진입(자동 2×) — 청산도 둘 다 메이커";
  b.innerHTML = '<svg viewBox="0 0 16 16" aria-hidden="true"><path class="u" d="M2 4.5h12"/><path class="m" d="M2 8h12"/><path class="d" d="M2 11.5h12"/></svg><span>SLTP</span>';   // 2026-10-28 펜 메모 «텍스트를 SLTP 로»
  b.addEventListener("pointerdown", (e) => e.stopPropagation());
  b.addEventListener("click", (e) => { e.stopPropagation(); if (fpIndOpen) fpIndSet(false); lineToolToggle(); });
  card.appendChild(b);
  return b;
}
function renderLineTool() {
  const svg = el("candleSvgSnapshot"), G = chartGeo, box = svg && svg.parentElement;
  const p = Number(latestLivePriceByAsset[activeSnapshotAsset] || 0), m = svg && svg.getScreenCTM();
  if (lineTool && lineTool.asset !== activeSnapshotAsset) lineTool = null;   // 다른 코인 가격의 선을 그대로 두지 않는다
  el("fpLineBtn")?.setAttribute("aria-pressed", String(!!lineTool));
  const show = !!(lineTool && G && m && G.ymax > G.ymin && p > 0 && !fpAltOn() && orderCoinOk());
  const h = show ? lineToolEl(box) : el("lineTool");
  if (!h) return;
  h.hidden = !show;
  if (!show) return;
  const t = lineTool, pr = box.getBoundingClientRect(), mob = isMobileChartMode();
  const X = (x) => m.a * x + m.e - pr.left, Yv = (v) => G.mt + (G.ymax - v) * (G.pb - G.mt) / (G.ymax - G.ymin);
  const top = m.d * G.mt + m.f - pr.top, bot = m.d * G.pb + m.f - pr.top, Y = (v) => m.d * Yv(v) + m.f - pr.top;
  const clampY = (y) => Math.min(bot, Math.max(top, y));
  const L = X(G.ml), R = X(G.ml + G.cw), x0 = L + (R - L) * (mob ? 0.36 : 0.58), w = R - x0, cx = x0 + w / 2;
  const side = lineSide(t, p), sgn = side === "SHORT" ? -1 : 1;
  const yP = clampY(Y(p)), yT = clampY(Y(t.tp)), yS = clampY(Y(t.sl));
  const qty = Number(t.plan?.quantity) || 0, unit = coinUnit();
  const set = (sel, css) => { const n = h.querySelector(sel); if (n) Object.assign(n.style, css); return n; };
  set(".lt-zone.up", { left: x0 + "px", width: w + "px", top: Math.min(yT, yP) + "px", height: Math.abs(yP - yT) + "px" });
  set(".lt-zone.dn", { left: x0 + "px", width: w + "px", top: Math.min(yS, yP) + "px", height: Math.abs(yP - yS) + "px" });
  [["tp", t.tp], ["sl", t.sl]].forEach(([k, v]) => {
    const y = Y(v), off = y < top ? "↑ " : y > bot ? "↓ " : "", d = v / p - 1;
    set(`.lt-ln.${k}`, { left: L + "px", width: R - L + "px", top: clampY(y) + "px", opacity: off ? "0" : "" });
    const tag = set(`.lt-tag.${k}`, { left: cx + "px", top: clampY(y) + "px" });
    const pnl = qty ? ` <span class="lt-s">${d * sgn >= 0 ? "+" : "−"}$${Math.abs(d * p * qty).toFixed(1)}</span>` : "";
    tag.innerHTML = `<span class="lt-grip" aria-hidden="true"></span>${off}${k === "tp" ? "익절" : "손절"} ${lineFmt(v)} <span>${d >= 0 ? "+" : "−"}${Math.abs(100 * d).toFixed(2)}%</span>${mob ? "" : pnl}`;
    tag.setAttribute("aria-label", `${k === "tp" ? "익절" : "손절"}선 ${lineFmt(v)} — 위아래로 끌어 옮기기`);
  });
  set(".lt-ln.en", { left: x0 + "px", width: w + "px", top: yP + "px" });
  set(".lt-tag.en", { left: x0 + 6 + "px", top: yP + "px" }).textContent = `진입 peg ${lineFmt(p)}`;
  h.classList.toggle("short", side === "SHORT");
  // 표는 플롯 왼쪽 -- 두 꼬리표의 높이 띠(±22px)를 피한 빈 구간 중 표가 들어가는 첫 곳(위부터, 대개 두 선 사이). 없으면 가장 넓은 곳.
  const tk = h.querySelector(".lt-tk"), th = tk.offsetHeight + 8, ys = [yT, yS].sort((a, b) => a - b);
  const gaps = [[top + 8, ys[0] - 22], [ys[0] + 22, ys[1] - 22], [ys[1] + 22, bot - 8]];
  const g = gaps.find(([a, b]) => b - a >= th) || gaps.reduce((x, y) => (y[1] - y[0] > x[1] - x[0] ? y : x));
  Object.assign(tk.style, { left: L + 8 + "px", top: Math.max(top + 8, Math.min(g[0], bot - th)) + "px" });
  const sd = h.querySelector(".lt-side");
  sd.textContent = side ? `${side} · 자동 2×` : "방향 없음"; sd.className = `lt-side ${side === "SHORT" ? "bad" : side ? "good" : ""}`;
  h.querySelector(".lt-qty").textContent = side && !t.err ? (qty ? `${qty} ${unit}` : "크기 계산 중…") : "";
  const rr = side ? Math.abs((t.tp - p) / (t.sl - p)) : 0, rrEl = h.querySelector(".lt-rr");
  rrEl.textContent = t.err || (side ? `손익비 ${rr.toFixed(2)} : 1` + (qty ? ` · 손절 시 −$${Math.abs((t.sl - p) * qty).toFixed(1)}` : "") : "");
  rrEl.classList.toggle("bad", !!t.err);
  const go = h.querySelector(".lt-go"), armed = Date.now() < (t.arm || 0);
  go.disabled = !side || !!t.err || !qty;
  go.classList.toggle("short", side === "SHORT"); go.classList.toggle("armed", armed);   // className 을 덮으면 꾹 채움(hold)이 매초 지워진다
  go.innerHTML = armed ? `확인: ${side === "LONG" ? "롱" : "숏"} ${qty} ${unit} 주문`
    : `${side || "진입"} 진입 · peg <small>${matchMedia("(pointer: coarse)").matches ? (mob ? "두 번" : "두 번 눌러 확인") : "1.5초 꾹"}</small>`;
  h.querySelector(".lt-note").textContent = mob ? "익절·손절 모두 메이커로 닫음"
    : "청산 둘 다 메이커 — 익절 = 체결 즉시 지정가(post-only) · 손절 = 1초 가격이 닿으면 peg(마감 뒤 남은 수량 시장가)";
  set(".lt-x", { left: R - 6 + "px", top: top + 8 + "px" });
}
async function refreshPositionProfile() {   // 포지션 보기일 때만 1초(서버도 1초 캐시)
  if (fpColR !== "pos" || activePageTab !== "snapshot" || document.hidden || !flowOn()) return;   // 2026-10-20 ETH → 흐름 코인 전부(XRP)
  const now = Date.now();
  if (posProfileInflight || now - posProfileFetchAt < 900) return;   // 900: 1초 루프 흔들림에 한 번씩 건너뛰지 않게
  posProfileFetchAt = now; posProfileInflight = true;
  try {
    const win = fpPosWin;   // 고른 창 하나만 받는다(서버가 매초 세 창을 다 계산·전송하던 것, 10-10 검토)
    const asset = activeSnapshotAsset, r = await fetch(`${API_POSITION_PROFILE_URL}?asset=${asset}&days=${win}`, { cache: "no-store" });
    if (r.ok) {
      const j = await r.json();
      if (j && j.win && j.win[String(fpPosWin)] && asset === activeSnapshotAsset) {   // 그사이 창·코인을 바꿨으면 버린다
        latestPositionProfile = j; scheduleSnapshotChartRender();
      }
    }
  } catch (e) { /* 다음 초에 다시 */ } finally { posProfileInflight = false; }
}
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
// 2026-10-15 지평 사다리의 일 단위 신호 둘(/api/dir-signals, dashboard/dir_signals.py) -- 코인 무관(BTC 기준), 서버 1시간 캐시
const DIR_SIG_POLL_MS = 600000;
let latestDirSig = null;
let dirSigLastFetchAt = 0;
let latestOi5m = null;
let oi5mLastFetchAt = 0;
const GEX_POLL_MS = 60000;           // 2026-09-28 체인 10분 · 블록 20초(옵션 카드) -- 1분 폴링
// 2026-09-20 호가 창을 **차트 창 탭에 맞춘다**(사용자 지시).
// 🔴전에는 15분 고정이라 1h 탭에서 왼쪽(호가 15분)과 오른쪽(체결 60분)이 4배, 4h 탭에서는
//   16배 다른 구간을 말하고 있었다. index.html 의 창 토글 주석이 경계한 바로 그 상황이다 --
//   «한 카드 안의 두 그림이 다른 구간을 말하면 읽는 사람이 속는다».
const FLOW_HEATMAP_COLS = 300;       // 열 수는 고정 -- 전송량(≈3KB)과 해상도를 함께 묶는다
const flowHeatmapAgg = () =>         // 창 전체를 300열에 담는 초/열
  Math.max(1, Math.min(60, Math.round(chartWindowBars * chartTfK * 300 / FLOW_HEATMAP_COLS)));   // 30m·1h 는 상한 60초/열 = 5시간까지
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
let latestFlowHeatmap = null;
let flowHeatmapLastFetchAt = 0;
let latestLiquidation5mHist = [];
let liquidation5mLastFetchAt = 0;
// Liquidation map (Snapshot tab, 2026-08-24) -- estimated support/resistance, own fetch/render
// cycle (computed dashboard-side, not part of trading_bot.py state).
// lastSnapshotHistoryFetchAt tracks the candle history this panel's chart needs (activeSnapshotAsset,
// see below), independently of activeChartAsset (the Live tab's own, separate coin selector).
let latestLiquidationMap = null;
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
let macroCalendarLastFetchAt = 0;
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
const MACRO_CALENDAR_POLL_MS = 6 * 3600 * 1000; // matches server-side cache (MACRO_CALENDAR_CACHE_SECONDS)
// 2026-10-05 알림 센터 · 머리 칩 실전 성적 -- tick() 이 파일 아래 정의보다 먼저 돌아서 상태는 여기(위)에 둔다(TDZ)
const NOTIFY_CENTER_POLL_MS = 5 * 60 * 1000;
const NC_SEEN_KEY = "ncSeenAt";
let notifyCenterData = null;
let notifyCenterLastFetchAt = 0;
const CHIP_SCORE_POLL_MS = 5 * 60 * 1000;
const chipScore = {};
const chipScoreFetchAt = {};
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
  try { renderNews(); } catch (e) { console.error("news render:", e); }   // 2026-10-05 뉴스 카드의 «지금 코인» 강조·필터(예외가 코인 전환을 끊지 않게)
  chartPanReset();   // 코인을 바꾸면 실시간으로(관성도 멈춘다 -- 리뷰: 미끄러지던 게 새 코인을 과거로 끌었다)
  renderSnapshotAssetTabs();
  // 🔴2026-09-30 진입 미리보기(plan·cap)는 **코인별**인데 전환 때 안 비워 XRP 탭이 최대 60초(주문 불가 코인이면
  //   계속) ETH 계획(~$2,676)을 «지금 설정으로 넣으면»에 그렸다. 비우고 새 코인 것을 바로 받는다.
  lastEntryPlan = null; setEntryProjPreview(null);
  manualEntryRefreshSize();
  // 계좌 payload 는 전 코인의 포지션을 담고 있어 재요청 없이 다시 그리기만 하면 된다.
  renderSnapshotAccount();
  // Clear the 4 wired signals' cached readings + their poll-interval gates immediately -- without
  // this, the panels would keep showing the PREVIOUS coin's numbers (mislabeled as the new one)
  // until each signal's own poll interval next elapses (up to 5min for the slowest).
  // 2026-09-22 ETH 전용 카드에 «ETH 전용» 배지를 켠다(styles.css 의 body.not-eth 규칙).
  document.body.classList.toggle("not-eth", asset !== "eth");
  // ETH 전용 그림들의 캐시도 비운다 -- 안 비우면 ETH 로 **돌아올 때** 옛 그림이 한 프레임 번쩍인다.
  footprintBars = new Map(); latestFootprint = null; fpWeek = null; fpWeekTriedAt = 0;   // 1주일도 코인별(넓히면 새 코인 것을 바로 받는다 -- 옛 코인 30초 문턱 무시)
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
  latestLiquidation5mHist = [];
  latestOi5m = null; oi5mLastFetchAt = 0;
  latestLiquidationMap = null;
  liquidation5mLastFetchAt = 0;
  liquidationMapLastFetchAt = 0;
  lastSnapshotHistoryFetchAt = 0;
  // 🔴2026-09-30 차트 표식도 코인별이다 -- 안 비우면 ETH 사각형이 SOL 차트에 최대 60초 남았다.
  latestChartMarkers = null; chartMarkersLastFetchAt = 0;
  latestPositionProfile = null; posProfileFetchAt = 0;   // 2026-10-20 포지션 보기도 코인별
  // 2026-10-04 시장 맥락·추세 칩도 코인별(서버 ?asset=) -- 옛 코인 값을 새 코인 이름 아래 두지 않는다.
  latestMarketCtx = null; marketCtxLastFetchAt = 0; latestTrend = null; trendLastFetchAt = 0;
  latestPaper = null; paperLastFetchAt = 0;   // 2026-10-04 모의 엔진도 코인별(paper_<coin>/)
  renderMarketCtx();   // 옛 코인 카드를 다음 폴링(최대 1초)까지 두지 않는다
  renderPaper();

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
        if (latestMainState) render(latestMainState);
        endAssetScopeLoading(scope, generation);
      });

  await Promise.all([
    settleScope("indicators", []),   // 비ETH 코인 지표 API 제거(2026-10-01) -- 스켈레톤 해제만
    settleScope("liqmap", [
      refreshLiquidation5mSignal(),
      refreshLiquidationMap(),
      maybeFetchSnapshotChartHistory(),
    ]),
  ]);
  if (latestMainState) render(latestMainState);
}

/* 테마 토글 (2026-09-16). 다크 = 현행 화면, 라이트 = 애플 «Liquid Glass».
   첫 페인트는 index.html 의 인라인 스크립트가 이미 입혔다 -- 여기서는 «바꾸기」만 한다.
   ⭐색을 하드코딩한 사본을 만들지 않는다: 차트는 SVG 속성에 `var(--good)` 를 그대로 넣고
   브라우저가 실시간으로 푼다. 다만 getComputedStyle 로 읽는 곳은 다시 그려야 반영된다. */
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
    if (latestMainState) render(latestMainState);
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


// 창 바꾸기(5m·30m·1h 버튼). 오른쪽 끝(chartPanEnd)은 그대로. 휠 줌은 아래 chartZoomAt(폴링을 당기지 않는다 -- 휠 한 번마다 네 요청이 나간다).
function setChartWindow(bars, k = chartTfK) {
  chartFlingStop();
  if (bars === chartWindowBars && k === chartTfK) return;
  chartWindowBars = bars;
  chartTfK = k;
  renderChartWindowTabs();
  scheduleSnapshotChartRender();   // 데이터가 오기 전에도 이미 있는 봉으로 바로 넓힌다(누른·굴린 티)
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
  maybeFetchFpWeek();
  refreshOb5m();
}

function setupChartWindowTabs() {
  document.querySelectorAll("#chartWindowTabs .asset-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (!btn.dataset.bars) return;   // 4h·1d 는 fp_daily.js 가 켠다(4시간봉·일봉 풋프린트)
      // 2026-10-20 사용자 «5m 을 누르면 최신 1시간을 5분봉으로»: 넓혀 둔 창·과거로 옮긴 창도 처음으로
      setChartWindow(Number(btn.dataset.bars), Number(btn.dataset.tfk) || 1);   // 30m·1h = data-tfk 6·12
      if (chartPanEnd != null || chartPanY || chartPanRight) chartPanReset();
    });
  });
  renderChartWindowTabs();
}
function renderChartWindowTabs() {
  const host = document.getElementById("chartWindowTabs");
  const tabs = [...document.querySelectorAll("#chartWindowTabs .asset-tab")];
  let idx = 0;
  // 2026-10-09 다섯째 칸 1d(#fpDailyBtn, data-bars 없음) = 일봉 풋프린트. 켜져 있으면 선택 칸이 거기 가고 시간 칸은 꺼진다(fp_daily.js 가 이 함수를 다시 부른다)
  const daily = typeof window.fpDailyActive === "function" && window.fpDailyActive();
  const per = typeof window.fpDailyPeriod === "function" ? window.fpDailyPeriod() : "1d";   // 2026-10-20 같은 그림이 4h(4시간봉)도 그린다
  tabs.forEach((btn, k) => {
    const on = btn.id === "fpDailyBtn" ? daily && per === "1d" : btn.dataset.tf ? daily && per === btn.dataset.tf : !daily && Number(btn.dataset.bars) === chartWindowBars && (Number(btn.dataset.tfk) || 1) === chartTfK;
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
    host.classList.toggle("zoomed", !daily && fpZoomed());   // 넓힌 창 = 어느 칸도 켜지 않는다(1h 를 누르면 풋프린트로)
  }
  const note = el("fpZoomNote");
  if (note) {
    const on = !daily && fpZoomed();
    note.hidden = !on;
    const min = chartWindowBars * 5 * chartTfK;
    // 2026-10-20 연속 줌: 창이 아무 폭이라 반올림(«2.4시간») · «캔들»은 박스를 실제로 뺐을 때만(fpBoxesOff -- 렌더가 바뀔 때 이 함수를 다시 부른다)
    if (on) note.textContent = `${min >= 1440 ? `${+(min / 1440).toFixed(1)}일` : `${+(min / 60).toFixed(1)}시간`}${fpBoxesOff ? " · 캔들" : ""}`;
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

// Hour:minute only (2026-08-31 user request: "시와 분만 표시").
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
  const b = next.length ? next[next.length - 1] : null;
  // 2026-10-11 ETH·BTC 도 형성 봉을 partial 로 준다(«닫히기 전에 찍은 값», 서버 캐시로 최대 60초 묵음 -- 그새 닫힌 봉일 수도 있다).
  //   같은 시각의 라이브 봉과 **시각으로** 짝짓는다(마지막끼리가 아니라): 시가 = 서버(REST 시가가 참) · 고저 = 합집합 · 종가 = 라이브(더 새다,
  //   지금 봉이면 바로 뒤 updateSnapshotCandleLive 가 최신가로 덮는다). 새로 연 화면(prev 없음)은 partial 이 곧 씨앗이다.
  const a = b && Array.isArray(prev) ? prev.find((c) => c.time === b.time) : null;
  if (a && (b.partial || nowS < b.time + barMin * 60)) {
    b.high = Math.max(b.high, a.high);
    b.low = Math.min(b.low, a.low);
    if (b.partial) b.close = a.close;
  }
  // 🔴2026-10-11 사용자 «PC에서 진행 중인 봉이 실제 봉과 달라». ETH 서버는 마감봉만, 그것도 닫힌 뒤 ~50초에 준다(서버 실측).
  //   통째로 갈아 끼우면 라이브가 만든 형성 봉(·아직 안 온 직전 봉)이 버려지고 updateSnapshotCandleLive 가 **지금 가격으로**
  //   새 봉을 다시 열었다 -- 마감 +60초 재시도·5분 폴링마다 시가·고저가 그 순간 값으로 리셋(시뮬 시가 오차 최대 $4.55).
  //   서버 마지막 봉보다 새 라이브 봉은 이어 붙인다(test/test_forming_candle_vs_exchange_20261011.py).
  if (b && Array.isArray(prev)) return next.concat(prev.filter((c) => c.time > b.time));
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
  }
  // 🔴차트는 **상태(state)가 바뀐 푸시**에서만 다시 그려졌다. 시세만 오는 푸시(대부분)에서는
  //   아무것도 안 그려서, 현재가 선과 진행 중인 봉이 상태 변경 주기(실측 5초)에 묶여 멈춰
  //   보였다(2026-09-16 사용자 "래깅이 있어"). 시세가 곧 그 선의 값이므로 여기서도 그린다.
  //   스로틀은 같은 상수를 쓰므로 아래 render() 경로와 겹쳐도 두 번 그리지 않는다.
  maybeRenderSnapshotChartNow();
  if (!latestMainState || isScrolling() || !payload?.state?.state) return;
  render(latestMainState, null, { stateChanged: true });
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
    // 🔴HTTP 오류 응답(터널 502·배포 재시작)에는 브라우저가 재연결하지 않고 CLOSED 로 굳는다 -- 그러면 render() 가 영영 멈춘다
    if (events.readyState === EventSource.CLOSED && dashboardEvents === events) {
      dashboardEvents = null;
      setTimeout(() => { if (!document.hidden) connectDashboardEvents(); }, 5000);
    }
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
  }).join("") + opsOkToggleHtml(checks.filter((c) => opsTone(c.status) === "good").length));
}
// 2026-10-17 휴대폰(≤720)만 보이는 «정상 N개 보기» -- 정상 줄은 CSS 가 접고, 펼침은 목록의 show-ok 클래스(30초 재그리기에도 유지)
function opsOkToggleHtml(n) {
  const open = !!el("opsHealthList")?.classList.contains("show-ok");
  return n ? `<button type="button" class="ops-ok-toggle" aria-expanded="${open}">정상 ${n}개 ${open ? "접기" : "보기"}</button>` : "";
}
document.addEventListener("click", (e) => {
  const b = e.target instanceof Element ? e.target.closest(".ops-ok-toggle") : null;
  if (!b) return;
  const open = b.parentElement.classList.toggle("show-ok");
  b.setAttribute("aria-expanded", String(open));
  b.textContent = b.textContent.replace(/보기|접기/, open ? "접기" : "보기");
});

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

// ── Account 카드 = 오늘 성과 + 과거 기록 (2026-10-14 사용자 시안 C · «주문은 플로팅 버튼이라 카드에서 주문은 최소화») ──
//   위 = 포지션 한 줄(+ 진입 미리보기 한 마디) · 오늘 띠(손익 · 0~24시 흐름 · 오늘 닫힌 거래)
//   아래 = 손익 달력(한 달씩) | 자산 곡선 + 낙폭 + 기간 성적. 주문 조작부(.acct-lanes)는 DOM 에 남아 #ofab 가 옮겨 쓴다(카드 안에서는 CSS 로 숨김).
//   원천: 원장(latestBinanceAccount.ledger, 모든 코인 · 서버 꼬리 400왕복) + 잔고·포지션 + 서버 «0시 순자산»(day_start).
const ACCT_SPANS = [["오늘", 0], ["1주", 7], ["1개월", 30], ["전체", -1]];   // 일수 · 0 = 오늘 0시부터 · -1 = 원장 전체
const ACCT_DAY = 86400000, ACCT_KST = 9 * 3600000;
let acctSpanIdx = (() => { try { const raw = localStorage.getItem("acctSpanIdx"), v = Number(raw); return raw !== null && raw !== "" && v >= 0 && v < ACCT_SPANS.length ? v : 2; } catch (e) { return 2; } })();   // 🔴Number(null) = 0 = «오늘» 함정(옛 acctSpanDays 와 같은 것) -- 저장값이 없으면 1개월
let acctCalBack = 0;   // 달력: 0 = 이번 달, 1 = 지난달 …
let acctSelDay = null;   // 2026-10-20 사용자 «달력에서 날을 누르면 위 0~24시 차트로 그날 손익»: KST 날짜 키 · null = 오늘
const acctKey = (ms) => new Date(ms + ACCT_KST).toISOString().slice(0, 10);   // KST 날짜(서버 day_start 와 같은 경계)
const acctDay0 = (key) => Date.parse(key + "T00:00:00+09:00");
const acctHm = (ms) => new Date(ms + ACCT_KST).toISOString().slice(11, 16);
const acctMd = (ms) => { const d = new Date(ms + ACCT_KST); return `${d.getUTCMonth() + 1}/${d.getUTCDate()}`; };
const acctSu = (v) => `${v >= 0 ? "+" : "−"}$${Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: Math.abs(v) < 100 ? 2 : 0 })}`;
const acctSuc = (v) => { const a = Math.abs(v); return `${v >= 0 ? "+" : "−"}$${a >= 1000 ? (a / 1000).toFixed(1) + "k" : a.toFixed(0)}`; };
const acctTone = (v) => (v > 0 ? "good" : v < 0 ? "bad" : "neutral");
const acctPct = (p) => (p == null || !Number.isFinite(p) ? "—" : `${p >= 0 ? "+" : "−"}${Math.abs(p * 100) < 10 ? Math.abs(p * 100).toFixed(1) : Math.round(Math.abs(p * 100))}%`);
const acctDur = (ms) => { const m = Math.round(ms / 60000); return m < 60 ? `${m}분` : m < 1440 ? `${Math.floor(m / 60)}시간 ${String(m % 60).padStart(2, "0")}분` : `${(m / 1440).toFixed(1)}일`; };
const acctCoin = (s) => String(s || "").replace(/USD[TC]$/, "");
// 2026-10-20 XRP 거래: $10 밑(XRP 1.30)은 4자리 -- 2자리 고정이면 1.3007 이 1.30 으로 보였다(계좌·미리보기·손절 패널·호가 칩 공통).
const acctPx = (v) => { const d = Math.abs(Number(v)) < 10 ? 4 : 2; return Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }); };
function acctTripTip(t) {
  const dir = t.side === "LONG" ? 1 : -1;
  return `${acctMd(t.exit_time)} ${acctHm(t.entry_time)}→${acctHm(t.exit_time)} (${acctDur(t.exit_time - t.entry_time)})\n`
    + `${acctCoin(t.symbol)} ${t.side === "LONG" ? "롱" : "숏"} ${t.max_qty} · ${acctPx(t.entry_price)} → ${acctPx(t.exit_price)}\n`
    + `추가 ${t.adds ?? "?"}회 · 최대 ${t.peak_x ?? "?"}배 · ${t.trend == null ? "추세 ?" : t.trend === dir ? "추세 쪽" : "역추세"}\n손익 ${acctSu(Number(t.net_pnl) || 0)}`;
}

// 기간 칩·달 넘김 -- 카드는 폴링마다 통째로 다시 그려지므로 위임으로 한 번만 붙인다.
document.addEventListener("click", (e) => {
  const s = e.target.closest("#acctCard [data-acct-span]"), m = e.target.closest("#acctCard [data-acct-month]");
  const dy = e.target.closest("#acctCard [data-acct-day]"), back = e.target.closest("#acctCard [data-acct-today]");
  if (dy || back) {   // 같은 날을 다시 누르거나 오늘을 누르면 오늘로
    const k = dy ? dy.dataset.acctDay : null;
    acctSelDay = !k || k === acctSelDay || k === acctKey(Date.now()) ? null : k;
    renderSnapshotAccount();
  }
  if (s) { acctSpanIdx = Number(s.dataset.acctSpan); try { localStorage.setItem("acctSpanIdx", String(acctSpanIdx)); } catch (err) { /* 기억은 편의 */ } renderSnapshotAccount(); }
  if (m) { acctCalBack = Math.max(0, acctCalBack + Number(m.dataset.acctMonth)); renderSnapshotAccount(); }
});

// 오늘 0~24시: 닫힌 왕복 = 그 시각 막대(손익만큼) · 누적 = 계단선 · 지금 포지션 = 빗금 막대
function acctTodaySvg(tT, unreal, hasPos, t0, now, w, h) {
  const x0 = 6, x1 = w - 6, y0 = 8, y1 = h - 16, X = (t) => x0 + Math.max(0, Math.min(1, (t - t0) / ACCT_DAY)) * (x1 - x0);
  let c = 0; const cum = tT.map((t) => { const a = c; c += Number(t.net_pnl) || 0; return [a, c]; });
  const vals = [0, ...cum.flat(), c + unreal], lo = Math.min(...vals), hi = Math.max(...vals), Y = (v) => y1 - (v - lo) / Math.max(1e-9, hi - lo) * (y1 - y0);
  let s = `<svg class="acct-today-svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="오늘 0~24시 손익 흐름">`
    + `<rect x="${X(now)}" y="${y0}" width="${Math.max(0, x1 - X(now))}" height="${y1 - y0}" fill="var(--ink)" fill-opacity=".03"/>`
    + `<line x1="${x0}" x2="${x1}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--ink)" stroke-opacity=".22"/>`;
  [0, 6, 12, 18, 24].forEach((hr) => { s += `<text x="${x0 + hr / 24 * (x1 - x0)}" y="${h - 3}" font-size="10" fill="var(--muted)" text-anchor="${hr ? hr === 24 ? "end" : "middle" : "start"}">${hr}시</text>`; });
  let d = `M${x0} ${Y(0)}`;
  tT.forEach((t, i) => { const x = X(t.exit_time), [a, b] = cum[i]; d += ` H${x.toFixed(1)} V${Y(b).toFixed(1)}`;
    s += `<rect x="${(x - 4).toFixed(1)}" y="${Math.min(Y(a), Y(b)).toFixed(1)}" width="8" height="${Math.max(1.5, Math.abs(Y(a) - Y(b))).toFixed(1)}" rx="1.5" fill="var(--${b >= a ? "good" : "bad"})"><title>${escapeHtml(acctTripTip(t))}</title></rect>`; });
  s += `<path d="${d} H${X(now).toFixed(1)}" fill="none" stroke="var(--ink)" stroke-opacity=".55" stroke-width="1.4"/>`;
  if (hasPos) s += `<rect x="${(X(now) - 4).toFixed(1)}" y="${Math.min(Y(c), Y(c + unreal)).toFixed(1)}" width="8" height="${Math.max(1.5, Math.abs(Y(c + unreal) - Y(c))).toFixed(1)}" rx="1.5" fill="var(--${unreal >= 0 ? "good" : "bad"})" fill-opacity=".3" stroke="var(--${unreal >= 0 ? "good" : "bad"})" stroke-dasharray="2 2"><title>지금 미실현 ${acctSu(unreal)} (닫히기 전)</title></rect>`;
  return s + `<line x1="${X(now)}" x2="${X(now)}" y1="${y0}" y2="${y1}" stroke="var(--ink)" stroke-opacity=".4" stroke-dasharray="2 3"/></svg>`;
}

// 손익 달력 한 달(월요일 시작 · 맨 오른쪽 = 주 합). 색 진하기 = √(|그날 손익| ÷ 원장 최대)
// 2026-10-20 사용자 «손익 옆에 지갑 대비 % · 달력에도»: walletAt(t) = t 시점 지갑(income 이 덮는 날은 정확 · 그 앞은 닫힌 거래로 추정)
// 같은 날 저녁: 칸 값 = 그날 지갑에 들어온 실현+펀딩+수수료(서버 income · o.inc). 옛 «청산한 날에 왕복 전체»는 전날 부분 익절까지 몰아 넣었다.
const acctIncTip = (o) => [["체결 실현", o.realized], ["펀딩", o.funding], ["수수료", o.commission], ["기타", o.other]]
  .filter(([, v], i) => !i || Math.abs(Number(v) || 0) >= 0.005).map(([k, v]) => `${k} ${acctSu(Number(v) || 0)}`).join(" · ")
  + `${Math.abs(Number(o.transfer) || 0) >= 0.005 ? `\n입출금 ${acctSu(Number(o.transfer))}(손익에서 뺌)` : ""}`
  + Object.entries(o.other_assets || {}).map(([k, v]) => `\n${k} ${v >= 0 ? "+" : "−"}${Math.abs(v).toPrecision(3)}(달러 합에서 뺌)`).join("");
function acctCalendar(days, ym, todayKey, selKey, walletAt) {
  const pctOf = (pnl, t, t1) => { const w = walletAt(t, t1); return w > 0 ? pnl / w : null; };
  const maxAbs = Math.max(1e-9, ...[...days.values()].map((o) => Math.abs(o.pnl)));
  const first = acctDay0(ym + "-01"), dim = new Date(Date.UTC(+ym.slice(0, 4), +ym.slice(5, 7), 0)).getUTCDate(), lead = (new Date(first + ACCT_KST).getUTCDay() + 6) % 7;
  let h = ["월", "화", "수", "목", "금", "토", "일", "주"].map((d) => `<span class="dow">${d}</span>`).join(""), wk = 0, col = 0, mo = 0, wd = 0, ld = 0, wk0 = first;
  const pad = () => { h += `<span class="d out"></span>`; col++; };
  for (let i = 0; i < lead; i++) pad();
  for (let d = 1; d <= dim; d++) {
    const key = `${ym}-${String(d).padStart(2, "0")}`, o = days.get(key);
    if (o) { wk += o.pnl; mo += o.pnl; if (o.n || (o.inc && o.inc.realized)) { if (o.pnl > 0) wd++; else ld++; } }   // 이익/손실일 = 거래한 날만(펀딩만 있는 날 제외)
    const a = o ? Math.round((0.1 + 0.6 * Math.sqrt(Math.abs(o.pnl) / maxAbs)) * 100) : 0;
    const dp = o ? pctOf(o.pnl, acctDay0(key), acctDay0(key) + ACCT_DAY) : null;
    if (col === 0) wk0 = acctDay0(key);   // 이 줄(주)의 첫 날 -- 주 합 % 의 분모 시점
    const tl = o ? o.trips.map((t) => `${acctHm(t.exit_time)} ${acctCoin(t.symbol)} ${t.side === "LONG" ? "롱" : "숏"} ${acctSu(Number(t.net_pnl) || 0)}`).join("\n") : "";
    const tip = !o ? "" : o.inc ? `${key} · ${acctSu(o.pnl)} (그날 시작 지갑${(Number(o.inc.transfer) || 0) > 0 ? "+그날 입금" : ""} 대비 ${acctPct(dp)})\n그날 지갑에 들어온 실현+펀딩+수수료 — ${acctIncTip(o.inc)}`
        + (o.n ? `\n그날 닫힌 거래 ${o.n}건(왕복 전체 손익 · 진입 뒤 다른 날 실현분 포함)\n${tl}` : "")
      : `${key} · ${o.n}건 ${acctSu(o.pnl)} (그날 시작 지갑 대비 ${acctPct(dp)})\n닫힌 거래 왕복 합(이 날은 지갑 입금 기록 밖 -- 펀딩 빠짐)\n${tl}`;
    // 오늘까지의 날 = 누르면 위 띠가 그날 0~24시로(미래 날은 그냥 칸)
    const tag = key <= todayKey ? "button" : "span", sel = key === selKey;
    h += `<${tag} class="d${key === todayKey ? " today" : ""}${sel ? " sel" : ""}"${tag === "button" ? ` type="button" data-acct-day="${key}" aria-pressed="${sel}" aria-label="${key} ${o ? `${o.n ? `${o.n}건 ` : ""}${acctSu(o.pnl)} ${acctPct(dp)}` : "거래 없음"} — 위 차트에 그날 보기"` : ""}`
      + `${o ? ` style="background:color-mix(in srgb, var(--${o.pnl >= 0 ? "good" : "bad"}) ${a}%, transparent)" title="${escapeHtml(tip)}"` : ""}><i>${d}</i>${o ? `<span class="v"><b>${Math.abs(o.pnl) < 9.95 ? `${o.pnl >= 0 ? "+" : "−"}$${Math.abs(o.pnl).toFixed(1)}` : acctSuc(o.pnl)}</b><em>${acctPct(dp)}</em></span>` : ""}</${tag}>`;
    col++;
    if (col === 7 || d === dim) { while (col < 7) pad(); h += `<span class="wk ${acctTone(wk)}">${wk ? `<b>${acctSuc(wk)}</b><em>${acctPct(pctOf(wk, wk0, acctDay0(key) + ACCT_DAY))}</em>` : ""}</span>`; wk = 0; col = 0; }
  }
  return { html: `<div class="acct-cal">${h}</div>`, mo, wd, ld, moPct: pctOf(mo, first, first + dim * ACCT_DAY) };
}

// 자산 곡선(누적 실현 계단) + 아래 «직전 최고점 대비 낙폭» + 최악 3건 표시
function acctCurveSvg(a, unreal, hasPos, now, w, h) {
  const ml = 54, mr = 10, top = 10, mid = Math.round(h * 0.66), bot = h - 18;
  const ts = a[0].entry_time, X = (t) => ml + (t - ts) / Math.max(1, now - ts) * (w - ml - mr);
  let c = 0, peak = 0; const pts = a.map((t) => { c += Number(t.net_pnl) || 0; peak = Math.max(peak, c); return { t: t.exit_time, c, dd: c - peak, trip: t }; });
  const endC = c + unreal, lo = Math.min(0, ...pts.map((p) => p.c), endC), hi = Math.max(0, ...pts.map((p) => p.c), endC);
  const Y = (v) => mid - (v - lo) / Math.max(1e-9, hi - lo) * (mid - top);
  const ddMin = Math.min(-1e-9, ...pts.map((p) => p.dd)), YD = (v) => mid + 16 + (v / ddMin) * (bot - mid - 16);
  let s = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="누적 실현 손익과 최고점 대비 낙폭">`;
  [lo, 0, hi].forEach((v) => { s += `<line x1="${ml}" x2="${w - mr}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--ink)" stroke-opacity="${v === 0 ? 0.22 : 0.07}"/><text x="${ml - 6}" y="${Y(v) + 4}" font-size="10" fill="var(--muted)" text-anchor="end">${acctSuc(v)}</text>`; });
  let d = `M${ml} ${Y(0)}`, dd = `M${ml} ${YD(0)}`;
  pts.forEach((p) => { d += ` H${X(p.t).toFixed(1)} V${Y(p.c).toFixed(1)}`; dd += ` H${X(p.t).toFixed(1)} V${YD(p.dd).toFixed(1)}`; });
  d += ` H${X(now).toFixed(1)}`; dd += ` H${X(now).toFixed(1)}`;
  s += `<path d="${d} V${Y(0)} H${ml} Z" fill="var(--ink)" fill-opacity=".05"/><path d="${d}" fill="none" stroke="var(--ink)" stroke-opacity=".85" stroke-width="1.6"/>`;
  if (hasPos) s += `<line x1="${X(now)}" x2="${X(now)}" y1="${Y(c)}" y2="${Y(endC)}" stroke="var(--${unreal >= 0 ? "good" : "bad"})" stroke-width="3" stroke-dasharray="3 2"><title>지금 미실현 ${acctSu(unreal)}</title></line>`;
  s += `<text x="${ml}" y="${mid + 12}" font-size="10" fill="var(--muted)">낙폭(직전 최고점 대비)</text><path d="${dd} V${YD(0)} H${ml} Z" fill="var(--bad)" fill-opacity=".2"/><path d="${dd}" fill="none" stroke="var(--bad)" stroke-opacity=".7"/>`
    + `<text x="${ml - 6}" y="${YD(ddMin) + 4}" font-size="10" fill="var(--bad)" text-anchor="end">${acctSuc(ddMin)}</text>`;
  const labX = [];   // 이름표끼리 90px 안이면 둘째부터는 동그라미만(호버로 값)
  a.slice().sort((x, y) => x.net_pnl - y.net_pnl).slice(0, 3).filter((t) => t.net_pnl < 0).forEach((t) => {
    const p = pts.find((q) => q.trip === t), x = X(t.exit_time), rx = x > w - 110, room = labX.every((q) => Math.abs(q - x) > 90);
    if (room) labX.push(x);
    s += `<circle cx="${x}" cy="${Y(p.c)}" r="5" fill="none" stroke="var(--bad)" stroke-width="1.6"><title>${escapeHtml(acctTripTip(t))}</title></circle>`
      + (room ? `<text x="${x + (rx ? -8 : 8)}" y="${Y(p.c) + (rx ? -8 : 4)}" font-size="10.5" fill="var(--bad)" text-anchor="${rx ? "end" : "start"}">${acctMd(t.exit_time)} ${acctSuc(t.net_pnl)}</text>` : "");
  });
  pts.forEach((p) => { s += `<rect x="${(X(p.t) - 3).toFixed(1)}" y="${top}" width="6" height="${mid - top}" fill="transparent"><title>${escapeHtml(acctTripTip(p.trip) + "\n누적 " + acctSu(p.c))}</title></rect>`; });
  const tk = now - ts > 20 * ACCT_DAY ? 7 * ACCT_DAY : ACCT_DAY;
  for (let t = acctDay0(acctKey(ts)) + tk; t < now; t += tk) s += `<text x="${X(t)}" y="${h - 3}" font-size="10" fill="var(--muted)" text-anchor="middle">${acctMd(t)}</text>`;
  return s + "</svg>";
}

function acctStatsHtml(a) {
  const sum = (r) => r.reduce((x, t) => x + (Number(t.net_pnl) || 0), 0);
  const W = a.filter((t) => t.net_pnl > 0), L = a.filter((t) => t.net_pnl <= 0), tot = sum(a);
  const aw = W.length ? sum(W) / W.length : 0, al = L.length ? sum(L) / L.length : 0;
  let c = 0, pk = 0, mdd = 0; a.forEach((t) => { c += Number(t.net_pnl) || 0; pk = Math.max(pk, c); mdd = Math.min(mdd, c - pk); });
  const worst = a.reduce((x, t) => (t.net_pnl < x.net_pnl ? t : x)), lossSum = sum(L), last = a[a.length - 1];
  let st = 0; for (let i = a.length - 1; i >= 0 && (a[i].net_pnl > 0) === (last.net_pnl > 0); i--) st++;
  const row = (k, v, tone, tip) => `<div class="acct-stat"${tip ? ` title="${escapeHtml(tip)}"` : ""}><span>${k}</span><b class="${tone || ""}">${v}</b></div>`;
  return row("누적 실현", acctSu(tot), acctTone(tot)) + row("왕복 · 승률", `${a.length} · ${Math.round(W.length / a.length * 100)}%`)
    + row("평균 이익/손실", `${acctSuc(aw)} / ${acctSuc(al)}`) + row("손익비", al ? (aw / Math.abs(al)).toFixed(2) : "—", "", "평균 이익 ÷ |평균 손실|")
    + row("건당 기대값", acctSu(tot / a.length), acctTone(tot)) + row("최대 낙폭", acctSu(mdd), "bad", "누적 실현의 직전 최고점 대비 가장 깊은 골")
    + row("최악 1건", `${acctSu(worst.net_pnl)} (${acctMd(worst.exit_time)})`, worst.net_pnl < 0 ? "bad" : "")
    + row("최악 1건 비중", lossSum < 0 && worst.net_pnl < 0 ? `${Math.round(worst.net_pnl / lossSum * 100)}%` : "—", "warn", "모든 손실 합에서 최악 1건이 차지하는 비율 — 꼬리 한 번이 결과를 정한다")
    + row("지금 연속", `${last.net_pnl > 0 ? "승" : "패"} ${st}`, last.net_pnl > 0 ? "good" : "bad");
}

// 2026-09-22 진입 미리보기(/api/manual-entry/preview)가 받아 둔 계획 -- 카드 포지션 줄의 «넣으면 →» 한 마디가 읽는다.
let entryProjPreview = null;
let entryProjKey = "";
let lastEntryPlan = null;

function renderSnapshotAccount() {
  syncOrderCoinGate();
  const summary = el("snapAcctSummary");
  if (!el("snapAcctPosition")) return;
  if (!latestBinanceAccount) {
    if (summary) { summary.textContent = "데이터 없음"; summary.className = "ops-health-summary neutral"; }
    setT("snapAcctBalance", "-");
    setH("snapAcctPosition", '<p class="muted">계좌를 불러오지 못했습니다.</p>');
    setH("snapAcctPerf", "");
    setH("snapAcctCycle", "");
    return;
  }
  const A = latestBinanceAccount, b = A.balance || {}, wallet = Number(b.wallet) || 0;
  const positions = (A.positions || []).filter((p) => Math.abs(Number(p.qty) || 0) > 0);
  const unreal = positions.reduce((x, p) => x + (Number(p.unrealized_pnl) || 0), 0), equity = wallet + unreal;
  const mine = snapshotAccountPosition();
  setT("snapAcctBalance", `지갑 ${fmtUsd(wallet)} · 순자산 ${fmtUsd(equity)} · 가용 ${fmtUsd(b.available)}${balanceAssetNote()}`);
  if (summary) {
    summary.textContent = mine ? (mine.side === "LONG" ? "롱 보유" : "숏 보유") : "포지션 없음";
    summary.className = `ops-health-summary ${mine ? (mine.side === "LONG" ? "good" : "bad") : "neutral"}`;
  }
  const now = Date.now(), todayKey = acctKey(now), t0 = acctDay0(todayKey);
  const T = (A.ledger || []).filter((t) => Number(t.exit_time) > 0).slice().sort((x, y) => x.exit_time - y.exit_time);
  const tT = T.filter((t) => t.exit_time >= t0), real = tT.reduce((x, t) => x + (Number(t.net_pnl) || 0), 0);
  // 2026-10-20 날별 손익 = 서버 income(KST 날마다 지갑에 들어온 실현·펀딩·수수료·입출금). income_from 부터는 다 들어와 있다 -- 그 앞은 원장 추정.
  const inc = A.income_days && A.income_from ? A.income_days : null;
  const incOn = (k) => !!inc && k >= A.income_from;
  const incSum = (k) => Object.entries(inc || {}).reduce((x, [d, o]) => x + (d >= k ? (Number(o.pnl) || 0) + (Number(o.transfer) || 0) : 0), 0);
  const incCut = inc ? acctDay0(A.income_from) : Infinity;
  // walletAt(t, t1) = t 시점 지갑 + [t, t1) 사이 입금(있으면) -- 달 중간에 넣은 돈을 분모에 넣지 않으면 «한 달 −157%»(10-07·09 입금 $2.5k)가 나온다
  const walletAt = (t, t1) => (incOn(acctKey(t)) ? wallet - incSum(acctKey(t))   // 그날 시작 지갑 = 지금 − 그 뒤 들어온 돈(입출금 포함)
    : wallet - (inc ? incSum(A.income_from) : 0) - T.reduce((x, tr) => x + (tr.exit_time >= t && tr.exit_time < incCut ? Number(tr.net_pnl) || 0 : 0), 0))   // 추정(펀딩·입출금 섞임)
    + (t1 ? Object.entries(inc || {}).reduce((x, [d, o]) => x + (d >= acctKey(t) && d < acctKey(t1) ? Math.max(0, Number(o.transfer) || 0) : 0), 0) : 0);
  // 오늘 시작 순자산: 서버가 0시 뒤 15분 안에 잰 값이 있으면 그것, 없으면 «0시 지갑»(미실현 0 가정)
  const dsOk = A.day_start && A.day_start.date === todayKey && Number(A.day_start.late_s) <= 900;
  const ds = dsOk ? Number(A.day_start.equity) : null, unreal0 = dsOk ? ds - Number(A.day_start.wallet) : null;
  const incT = incOn(todayKey) ? inc[todayKey] || {} : null, realT = incT ? Number(incT.pnl) || 0 : real, trfT = incT ? Number(incT.transfer) || 0 : 0;
  // 큰 숫자 = 실현(지갑 입금) + 미실현 변화 -- 오늘 입출금은 빠진다(옛 «지금 순자산 − 0시»는 입금도 손익으로 셌다)
  const start = ds != null ? ds : incT ? walletAt(t0) : wallet - real;
  const net = incT ? realT + unreal - (unreal0 ?? 0) : ds != null ? equity - ds : real + unreal, pct = start > 0 ? net / start : null;
  const startTip = (ds != null ? `0시 ${Math.round(A.day_start.late_s / 60)}분에 잰 순자산 ${fmtUsd(ds)} 기준` : incT ? `0시 지갑 ${fmtUsd(start)} 기준(0시 미실현 기록 없음 -- 0으로 봄)` : "0시 순자산 기록이 없어 «지갑 − 오늘 실현»으로 추정(펀딩·입출금 섞임)")
    + (incT ? `\n= 실현 ${acctSu(realT)} + 미실현 변화 ${acctSu(unreal - (unreal0 ?? 0))}${Math.abs(trfT) >= 0.005 ? `\n오늘 입출금 ${acctSu(trfT)} 은 손익에서 뺐다` : ""}` : ds != null ? "(입출금이 있으면 섞임)" : "");

  // ── 포지션 한 줄 + 진입 미리보기 한 마디(주문 조작부는 플로팅 버튼) ──
  const pj = (entryProjPreview || (lastEntryPlan && lastEntryPlan.projection) || {}).after;
  const pvNote = pj && Number(pj.notional_usdt) > 0
    ? `<span class="acct-pv" title="플로팅 버튼의 지금 설정(비율·배수)으로 넣으면 — 실제 포지션 아님">넣으면 → 청산까지 <b>${Number(pj.liq_pct).toFixed(1)}%</b> · 노출 <b>${Number(pj.exposure_x).toFixed(1)}배</b></span>` : "";
  const posLine = positions.length ? positions.map((p) => {
    const long = p.side === "LONG", mk = Number(p.mark_price) || 0, liq = Number(p.liquidation_price) || 0;
    const liqD = mk > 0 && liq > 0 ? (long ? mk - liq : liq - mk) / mk * 100 : null;
    const oo = (A.open_orders || []).filter((o) => o.symbol === p.symbol);
    return `<div class="acct-posline"><span class="acct-side ${long ? "good" : "bad"}">${long ? "롱" : "숏"} ×${Number(p.leverage) || "?"}</span>
      <span>${escapeHtml(p.symbol)} <b>${Number(p.qty)}</b></span><span>진입 <b>${acctPx(p.entry_price)}</b> → <b>${acctPx(mk)}</b></span>
      ${(A.unprotected || []).includes(`${p.symbol}:${String(p.side).toUpperCase()}`) ? `<span class="bad" title="거래소 스탑도 서버 5분봉 감시도 없습니다">🔴손절 없음</span>` : ""}
      ${liqD != null ? `<span>청산 <b>${acctPx(liq)}</b> <em class="${liqD < 3 ? "bad" : liqD < 6 ? "warn" : ""}">(${liqD.toFixed(1)}% 남음)</em></span>` : ""}
      <span>노출 <b>${equity > 0 ? (Math.abs(Number(p.notional) || 0) / equity).toFixed(1) : "—"}배</b></span>
      <span>미실현 <b class="${acctTone(Number(p.unrealized_pnl))}">${acctSu(Number(p.unrealized_pnl) || 0)}</b></span>
      ${p.entry_at ? `<span>보유 <b>${acctDur(now - Date.parse(p.entry_at))}</b></span>` : ""}
      ${oo.length ? `<span>걸린 주문 ${oo.map((o) => `${o.side === "BUY" ? "매수" : "매도"} ${acctPx(o.price)}×${o.qty}`).join(", ")}</span>` : ""}
      ${p === mine ? pvNote : ""}</div>`;
  }).join("") : `<div class="acct-posline"><span class="acct-side neutral">포지션 없음</span>${pvNote}</div>`;

  // SVG 는 칸의 실제 폭(·한 화면 모드면 fitLayout 이 준 높이)으로 1:1 로 그린다 -- 고정 viewBox 를 늘리면 글자가 커지거나 가운데로 찌그러진다.
  const narrowCard = matchMedia("(max-width: 900px)").matches;
  const posW = (el("snapAcctPosition").clientWidth || 1200) - 32, perfW = (el("snapAcctPerf").clientWidth || 1200) - 36;
  const todayW = Math.max(280, Math.round(narrowCard ? posW : (posW - 40) * 1.2 / 3.2));
  const curveW = Math.max(300, Math.round(narrowCard ? perfW : (perfW - 20) / 2));
  const curveH = document.documentElement.classList.contains("fit1") && fitAcctPlotH > 0 ? fitAcctPlotH : 280;
  // ── 오늘 띠(달력에서 고른 날이면 그날 0~24시 -- 실현만, 그날 미실현·펀딩은 원장에 없다) ──
  const selKey = acctSelDay && acctSelDay < todayKey ? acctSelDay : null, s0 = selKey ? acctDay0(selKey) : t0;
  const vT = selKey ? T.filter((t) => t.exit_time >= s0 && t.exit_time < s0 + ACCT_DAY) : tT, vReal = vT.reduce((x, t) => x + (Number(t.net_pnl) || 0), 0);
  const incV = selKey && incOn(selKey) ? inc[selKey] || {} : null, vFig = incV ? Number(incV.pnl) || 0 : vReal;
  // 막대 = 그날 닫힌 왕복의 전체 손익 -- 숫자(그날 지갑 입금)와 다르면 한 줄로 밝힌다
  const barNote = (fig, sum) => (vT.length && Math.abs(fig - sum) >= 0.01 ? `<span class="acct-bar-note muted">막대 = 그날 닫힌 왕복 전체 손익(합 ${acctSu(sum)} · 다른 날 실현분 포함)</span>` : "");
  const wins = vT.filter((t) => t.net_pnl > 0).length;
  const dow = "일월화수목금토"[new Date(s0 + ACCT_KST).getUTCDay()];
  const trips = vT.length ? vT.map((t) => `<li title="${escapeHtml(acctTripTip(t))}"><time>${acctHm(t.exit_time)}</time><span>${acctCoin(t.symbol)} ${t.side === "LONG" ? "롱" : "숏"} · ${acctDur(t.exit_time - t.entry_time)}${t.adds ? ` · 추가 ${t.adds}회` : ""}${t.peak_x > 6 ? ` · <em class="warn">${t.peak_x}배</em>` : ""}</span><b class="${acctTone(t.net_pnl)}">${acctSu(Number(t.net_pnl) || 0)}</b></li>`).join("")
    : `<li class="muted">닫힌 거래 없음</li>`;
  const dayLab = `${(selKey || todayKey).slice(5).replace("-", "/")}(${dow})`;
  const todayHtml = selKey ? `<div class="acct-today acct-today-past">
      <div class="acct-today-hero">
        <span class="acct-day-head"><span class="acct-eyebrow">손익 · ${dayLab}</span><span class="chipset"><button type="button" class="chip" data-acct-today>오늘로</button></span></span>
        <div class="acct-figline"><div class="acct-figure ${acctTone(vFig)}"${incV ? ` title="${escapeHtml(acctIncTip(incV))}"` : ""}>${acctSu(vFig)}</div><span class="acct-figpct ${acctTone(vFig)}" title="그날 시작 지갑(${incV ? "" : "추정 "}${fmtUsd(walletAt(s0, s0 + ACCT_DAY))}${walletAt(s0, s0 + ACCT_DAY) !== walletAt(s0) ? " · 그날 입금 포함" : ""}) 대비">${acctPct(walletAt(s0, s0 + ACCT_DAY) > 0 ? vFig / walletAt(s0, s0 + ACCT_DAY) : null)}</span></div>
        <span class="acct-today-pct"><span class="muted">${incV ? `그날 지갑에 들어온 실현+펀딩+수수료 · 지갑 ${fmtUsd(walletAt(s0))} 대비(그날 시작)` : `그날 닫힌 거래의 실현 손익 · 지갑 ${fmtUsd(walletAt(s0))} 대비(그날 시작, 추정) · 미실현·펀딩 빠짐`}</span></span>
        <dl class="acct-today-kv">${incV ? `<dt>실현 <span class="acct-kv-sub">· 펀딩·수수료 포함</span></dt><dd class="${acctTone(vFig)}" title="${escapeHtml(acctIncTip(incV))}">${acctSu(vFig)}</dd>`
          : `<dt>실현 · ${vT.length}건${vT.length ? ` (승 ${wins} 패 ${vT.length - wins})` : ""}</dt><dd class="${acctTone(vReal)}">${acctSu(vReal)}</dd>`}</dl>
      </div>
      <div class="acct-today-flow">${acctTodaySvg(vT, 0, false, s0, s0 + ACCT_DAY, todayW, 120)}${incV ? barNote(vFig, vReal) : ""}</div>
      <div class="acct-today-trips"><span class="acct-eyebrow">그날 닫힌 거래${vT.length ? ` · ${vT.length}건 (승 ${wins} 패 ${vT.length - wins})` : ""}</span><ol>${trips}</ol></div>
    </div>` : `<div class="acct-today">
      <div class="acct-today-hero">
        <span class="acct-eyebrow">오늘 손익 · ${dayLab}</span>
        <div class="acct-figline"><div class="acct-figure ${acctTone(net)}" title="${escapeHtml(startTip)}">${acctSu(net)}</div><span class="acct-figpct ${acctTone(net)}" title="${escapeHtml(startTip)}">${acctPct(pct)}</span></div>
        <span class="acct-today-pct"><span class="muted">시작 ${fmtUsd(start)} → 지금 ${fmtUsd(equity)} 대비</span></span>
        <dl class="acct-today-kv">${incT ? `<dt>실현 <span class="acct-kv-sub">· 펀딩·수수료 포함</span></dt><dd class="${acctTone(realT)}" title="${escapeHtml(acctIncTip(incT))}">${acctSu(realT)}</dd>`
            + `<dt>미실현 변화 <span class="acct-kv-sub">· ${unreal0 != null ? `0시 ${acctSu(unreal0)} → ` : ""}지금 ${acctSu(unreal)}</span></dt><dd class="${acctTone(unreal - (unreal0 ?? 0))}">${acctSu(unreal - (unreal0 ?? 0))}</dd>`
          : `<dt>실현 · ${tT.length}건${tT.length ? ` (승 ${wins} 패 ${tT.length - wins})` : ""}</dt><dd class="${acctTone(real)}">${acctSu(real)}</dd>
          <dt>미실현${positions.length ? ` · ${positions.map((p) => `${p.side === "LONG" ? "롱" : "숏"} ${Number(p.qty)}`).join(", ")}` : ""}</dt><dd class="${acctTone(unreal)}">${acctSu(unreal)}</dd>`}</dl>
      </div>
      <div class="acct-today-flow">${acctTodaySvg(tT, unreal, positions.length > 0, t0, now, todayW, 120)}${incT ? barNote(realT, real) : ""}</div>
      <div class="acct-today-trips"><span class="acct-eyebrow">오늘 닫힌 거래${incT && tT.length ? ` · ${tT.length}건 (승 ${wins} 패 ${tT.length - wins})` : ""}</span><ol>${trips}</ol></div>
    </div>`;
  // 리뷰: 카드를 통째로 다시 쓰면(포지션이 있으면 폴링마다 -- 미실현이 바뀐다) 키보드로 고른 날·칩의 포커스가 <body> 로 떨어졌다 -- 같은 칸에 되돌린다
  const fa = document.activeElement, fAt = fa && fa.closest && fa.closest("#acctCard")
    ? ["day", "month", "span"].map((k) => (fa.hasAttribute(`data-acct-${k}`) ? `#acctCard [data-acct-${k}="${fa.getAttribute(`data-acct-${k}`)}"]` : "")).find(Boolean) : "";
  // 2026-10-22 사용자 «여러 코인 한꺼번에 청산 -- 지금은 다른 청산을 기다려야 한다»: 포지션이 둘 이상이면 «전체 청산».
  //   서버가 코인별 청산 칸으로 동시에 보낸다(/api/manual-exit/submit-all). 두 번 눌러야 나간다 · 체결 방식은 플로팅 패널의 청산 방식(지정가/즉시)을 따른다.
  const closeAllRow = positions.length >= 2 || closeAllMsg ? `<div class="acct-posline acct-closeall">${positions.length >= 2
    ? `<button type="button" class="acct-closeall-btn${closeAllArmed ? " arm" : ""}" data-close-all="${positions.length}"${closeAllBusy ? " disabled" : ""}>${closeAllLabel(positions.length)}</button>` : ""}${closeAllMsg ? `<span>${escapeHtml(closeAllMsg)}</span>` : ""}</div>` : "";
  setH("snapAcctPosition", posLine + closeAllRow + todayHtml);

  // ── 아래: 달력 | 자산 곡선 ──
  const days = new Map();
  T.forEach((t) => { const k = acctKey(t.exit_time), o = days.get(k) || { pnl: 0, n: 0, trips: [] }; o.pnl += Number(t.net_pnl) || 0; o.n++; o.trips.push(t); days.set(k, o); });
  // income 이 덮는 날 = 그날 지갑 입금으로 바꾼다(펀딩만 있는 날·부분 청산만 있는 날도 칸이 생긴다)
  if (inc) { days.forEach((o, k) => { if (incOn(k)) { o.inc = inc[k] || {}; o.pnl = Number(o.inc.pnl) || 0; } });
    Object.entries(inc).forEach(([k, o]) => { if (incOn(k) && !days.has(k) && Math.abs(Number(o.pnl) || 0) >= 0.005) days.set(k, { pnl: Number(o.pnl) || 0, n: 0, trips: [], inc: o }); }); }
  const ym = (() => { const d = new Date(t0 + ACCT_KST); d.setUTCDate(1); d.setUTCMonth(d.getUTCMonth() - acctCalBack); return d.toISOString().slice(0, 7); })();
  const oldest = [T.length ? acctKey(T[0].exit_time).slice(0, 7) : ym, ...(inc ? [A.income_from.slice(0, 7)] : [])].sort()[0];
  const cal = acctCalendar(days, ym, todayKey, selKey, walletAt);
  const calHtml = `<div class="acct-cal-box">
      <div class="acct-cal-head"><button type="button" class="chip" data-acct-month="1" ${ym <= oldest ? "disabled" : ""} aria-label="지난달">‹</button>
        <b>${ym.slice(0, 4)}년 ${+ym.slice(5)}월</b><button type="button" class="chip" data-acct-month="-1" ${acctCalBack ? "" : "disabled"} aria-label="다음 달">›</button>
        <span class="muted">이익일 ${cal.wd} · 손실일 ${cal.ld} · <b class="${acctTone(cal.mo)}">${acctSu(cal.mo)}</b> <b class="${acctTone(cal.mo)}" title="${inc ? "그달 1일 지갑 + 그달 입금 대비" : "그달 1일 지갑(추정) 대비"}">${acctPct(cal.moPct)}</b></span></div>
      ${cal.html}
      ${inc && ym <= A.income_from.slice(0, 7) ? `<p class="acct-foot muted">칸 = 그날 지갑에 들어온 실현+펀딩+수수료 · ${acctMd(incCut)}부터(바이낸스 기록은 최근 3개월)${T.length && T[0].exit_time < incCut ? ` · 그 앞은 닫힌 거래 왕복 합` : ""}</p>`
        : T.length && ym <= oldest ? `<p class="acct-foot muted">원장은 ${acctMd(T[0].exit_time)}부터입니다(서버 꼬리 최근 400왕복).</p>` : ""}
    </div>`;
  const spanDays = ACCT_SPANS[acctSpanIdx][1];
  const from = spanDays === 0 ? t0 : spanDays > 0 ? now - spanDays * ACCT_DAY : 0;
  const a = T.filter((t) => t.exit_time >= from);
  const plot = spanDays === 0 ? acctTodaySvg(tT, unreal, positions.length > 0, t0, now, curveW, curveH)
    : a.length ? acctCurveSvg(a, unreal, positions.length > 0, now, curveW, curveH) : `<p class="acct-empty">이 기간에 닫힌 왕복이 없습니다.</p>`;
  const curveHtml = `<div class="acct-curve-box">
      <div class="chipset" role="group" aria-label="자산 곡선 기간">${ACCT_SPANS.map(([lab], i) => `<button type="button" class="chip${i === acctSpanIdx ? " on" : ""}" data-acct-span="${i}" aria-pressed="${i === acctSpanIdx}">${lab}</button>`).join("")}</div>
      <div class="acct-plot">${plot}</div>
      ${a.length ? `<div class="acct-stats">${acctStatsHtml(a)}</div>` : ""}
    </div>`;
  setH("snapAcctPerf", `<div class="acct-hist">${calHtml}${curveHtml}</div>`);
  renderCycleCard(A, positions, equity);
  if (fAt && !fa.isConnected) document.querySelector(fAt)?.focus();
}


// 2026-10-22 «진행 중 순환»(시안 A, 사용자 «청산할 때까지 달력 아래에 · 얼마에 사고 팔았는지 정확한 가격»): 열린 포지션마다
//   첫 체결부터 지금·미실현·실현·합계·그냥 보유·순환 효과 + 가격선 위 체결가(×수량) + 체결 칩. 체결 = 서버 왕복 legs
//   [시각ms, 가격, 수량, 늘림, 메이커, 실현, 수수료] · 가격선 = /api/cycle-path(체결 심볼 1m/5m/15m 종가). 청산하면 왕복이 닫혀 사라진다.
const cyclePaths = {};   // "심볼|쪽|첫 체결ms" -> {at, rows}
let cycleArgs = null;
const cycleQ = (v) => (+Number(v).toFixed(4)).toLocaleString("en-US", { maximumFractionDigits: 4 });
function cycleSvg(t, pos, rows, now, W) {
  const H = 250, L = 8, R = 104, T = 10, B = 22, legs = t.legs, long = t.side === "LONG";
  let q = 0, avg = 0;
  const steps = legs.map(([ms, p, qty, inc]) => { if (inc) { avg = (avg * q + p * qty) / (q + qty); q += qty; } else q -= qty; return [ms, avg]; });
  const t0 = legs[0][0] - 20 * 60000;
  const path = rows.filter((r) => r[0] >= t0);
  if (!path.length) path.push(...legs.map((g) => [g[0], g[1]]));
  path.push([now, pos.mark_price]);
  const ps = [...path.map((r) => r[1]), ...legs.map((g) => g[1])];
  let lo = Math.min(...ps), hi = Math.max(...ps); const pad = Math.max(hi - lo, hi * 1e-4) * 0.22; lo -= pad; hi += pad;
  const x0 = Math.min(path[0][0], legs[0][0]);
  const X = (ms) => L + (ms - x0) / Math.max(1, now - x0) * (W - L - R), Y = (p) => T + (hi - p) / (hi - lo) * (H - T - B);
  let d = "";
  steps.forEach(([ms, a]) => { d += d ? ` H${X(ms).toFixed(1)} V${Y(a).toFixed(1)}` : `M${X(ms).toFixed(1)},${Y(a).toFixed(1)}`; });
  const first = legs[0][1];
  let s = `<svg class="cy-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="진행 중 ${long ? "롱" : "숏"} 가격과 체결">`
    + `<path d="${d} H${X(now).toFixed(1)}" fill="none" stroke="var(--accent)" stroke-width="1.3" stroke-dasharray="3 3"/>`
    + `<line x1="${L}" x2="${W - R}" y1="${Y(first).toFixed(1)}" y2="${Y(first).toFixed(1)}" stroke="var(--amber)" stroke-dasharray="5 4"/>`
    + `<polyline fill="none" stroke="var(--neutral)" stroke-width="1.5" points="${path.map((r) => `${X(r[0]).toFixed(1)},${Y(r[1]).toFixed(1)}`).join(" ")}"/>`;
  const buy = (inc) => (inc ? long : !long);
  // ponytail: 체결가 글자는 30체결까지만(그 위는 칩이 가격을 말한다) -- 겹침 회피가 차트 밖으로 밀어낸다
  if (legs.length <= 30) {
    const placed = [], narrow = W < 600, gap = narrow ? 62 : 104;   // 매수 = 표시 아래 · 매도 = 위, 가까운 글자와 겹치면 바깥으로 14px 씩 · 좁은 화면은 ×수량을 칩에만
    legs.forEach(([ms, p, qty, inc]) => {
      const x = Math.min(Math.max(X(ms), L + 46), W - R - 46), up = !buy(inc); let ty = Y(p) + (up ? -10 : 18), moved = false;
      while (placed.some(([px, py]) => Math.abs(ty - py) < 13 && Math.abs(x - px) < gap)) { ty += up ? -14 : 14; moved = true; }
      // 2026-10-26 사용자 «모바일 최적화»: 좁은 화면은 밀려난 글자(표시에서 떨어져 기둥으로 쌓이고 서로·삼각형에 겹쳤다)·다른 체결 삼각형을 덮는 글자·차트 밖 글자를 뺀다 -- 가격은 아래 체결 칩에 다 있다
      if (narrow && (moved || ty < T + 10 || ty > H - B || legs.some(([m2, p2]) => Math.abs(x - X(m2)) < gap / 2 + 6 && Math.abs(ty - 4 - Y(p2)) < 13))) return;
      placed.push([x, ty]);
      s += `<text x="${x.toFixed(1)}" y="${ty.toFixed(1)}" class="cy-fx" fill="var(--${up ? "bad" : "good"})" text-anchor="middle">${acctPx(p)}${narrow ? "" : `<tspan class="cy-fq"> ×${cycleQ(qty)}</tspan>`}</text>`;
    });
  }
  legs.forEach(([ms, p, qty, inc, mk]) => {
    const x = X(ms), y = Y(p), b = buy(inc), c = b ? "var(--good)" : "var(--bad)";
    const tri = b ? `M${x - 5},${y + 4} L${x + 5},${y + 4} L${x},${y - 5} Z` : `M${x - 5},${y - 4} L${x + 5},${y - 4} L${x},${y + 5} Z`;
    s += `<path d="${tri}" fill="${mk ? "var(--chart-bg)" : c}" stroke="${c}" stroke-width="1.6"><title>${acctHm(ms)} ${b ? "매수" : "매도"} ${cycleQ(qty)} @ ${acctPx(p)}${mk ? "" : " · 테이커"}</title></path>`;
  });
  s += `<circle cx="${X(now).toFixed(1)}" cy="${Y(pos.mark_price).toFixed(1)}" r="3.5" fill="var(--ink)"/>`;
  const labs = [[Y(pos.mark_price), `지금 ${acctPx(pos.mark_price)}`, "ink"], [Y(pos.entry_price), `평단 ${acctPx(pos.entry_price)}`, "accent"], [Y(first), `첫 ${acctPx(first)}`, "amber"]].sort((a, b) => a[0] - b[0]);
  let prev = -1e9;
  labs.forEach(([y, txt, col]) => { y = Math.max(y, prev + 12); prev = y; s += `<text x="${W - R + 6}" y="${(y + 4).toFixed(1)}" class="cy-ax" fill="var(--${col})">${txt}</text>`; });
  return s + `<text x="${L}" y="${H - 6}" class="cy-ax" fill="var(--muted)">${acctHm(x0)}</text><text x="${W - R}" y="${H - 6}" class="cy-ax" fill="var(--muted)" text-anchor="end">${acctHm(now)} KST</text></svg>`;
}
function renderCycleCard(A, positions, equity) {
  cycleArgs = [A, positions, equity];
  const now = Date.now(), W = Math.max(320, el("snapAcctCycle")?.clientWidth || 760);   // 화면 폭 그대로 -- 고정 viewBox 는 넓은 화면에서 가운데 작게 떴다
  const open = (A.trades || []).map((t) => [t, positions.find((p) => p.symbol === t.symbol && p.side === t.side)])
    .filter(([t, p]) => !t.closed && p && (t.legs || []).length);
  setH("snapAcctCycle", open.map(([t, pos]) => {
    const key = `${t.symbol}|${t.side}|${t.entry_time}`, c = cyclePaths[key] || (cyclePaths[key] = { at: 0, rows: [] });
    if (now - c.at > 60000) {
      c.at = now;
      fetch(`/api/cycle-path?symbol=${encodeURIComponent(t.symbol)}&start=${t.legs[0][0] - 20 * 60000}`, { cache: "no-store" })
        .then((r) => r.json()).then((j) => { if (j.ok) { c.rows = j.rows; if (cycleArgs) renderCycleCard(...cycleArgs); } }).catch(() => {});
    }
    const legs = t.legs, long = t.side === "LONG", coin = acctCoin(t.symbol), dir = long ? 1 : -1;
    const real = legs.reduce((a, g) => a + g[5], 0), fee = legs.reduce((a, g) => a + g[6], 0), unreal = Number(pos.unrealized_pnl) || 0;
    const total = real + unreal - fee, [, fpx] = legs[0];
    const nIn = legs.filter((g) => g[3]).length, mins = (now - legs[0][0]) / 60000, cut = t.truncated_open;
    const hm = (ms) => (acctKey(ms) === acctKey(now) ? "" : acctMd(ms) + " ") + acctHm(ms);
    const tile = (lab, val, tone, sub) => `<div><i>${lab}</i><b class="${tone}">${val}</b><s>${sub}</s></div>`;
    // 2026-10-22 사용자 «합계·그냥 보유에 %도 -- 달러는 더 나쁜데 %로는 나아 보인다»: 합계 = 순환 중 최대 원가(수량×평단) 대비.
    // 2026-10-24 사용자 «같은 진입 비율로»: 그냥 보유도 **최대 원금과 같은 돈**을 첫 진입가에 사서 들고 있었다면 -- 옛 판은 첫 체결 수량(첫 1/3)만이라
    //   순환 효과가 크기가 다른 두 손익을 뺐다. % 는 그대로 첫 진입가 대비 가격 변화다.
    let qq = 0, aa = 0, peak = 0;
    legs.forEach(([, p, qty, inc]) => { if (inc) { aa = (aa * qq + p * qty) / (qq + qty); qq += qty; } else qq -= qty; peak = Math.max(peak, qq * aa); });
    const pct2 = (f) => `${f >= 0 ? "+" : "−"}${Math.abs(f * 100).toFixed(2)}%`;
    const hold0 = peak, hold = peak / fpx * (pos.mark_price - fpx) * dir, cyc = total - hold, totR = total / peak, holdR = hold / hold0;
    const withPct = (usd, r) => `${usd} <small>${pct2(r)}</small>`;
    const chips = legs.map(([ms, p, qty, inc, mk, rp]) => { const b = inc ? long : !long;
      return `<span class="${b ? "b" : "s"}"><i>${hm(ms)}</i>${b ? "매수" : "매도"} <b>${acctPx(p)}</b> ×${cycleQ(qty)}${mk ? "" : " · 테이커"}${rp ? ` <em class="${acctTone(rp)}">${acctSu(rp)}</em>` : ""}</span>`; }).join("");
    return `<div class="cy-box">
      <div class="cy-head"><span class="cy-live">진행 중</span><b class="cy-title">${coin} ${long ? "롱" : "숏"} 순환</b>
        <span class="muted">${hm(legs[0][0])} 시작 · ${acctDur(now - legs[0][0])} · ${long ? "매수" : "매도"} ${nIn} · ${long ? "매도" : "매수"} ${legs.length - nIn}</span></div>
      <div class="cy-tiles">
        ${tile("지금", `${cycleQ(pos.qty)} ${coin}`, "", `평단 ${acctPx(pos.entry_price)}`)}
        ${tile("미실현", acctSu(unreal), acctTone(unreal), `현재가 ${acctPx(pos.mark_price)}`)}
        ${tile("실현(부분 정리)", acctSu(real), acctTone(real), `수수료 $${fee.toFixed(2)}`)}
        ${tile("합계", cut ? acctSu(total) : withPct(acctSu(total), totR), acctTone(total), (cut ? "" : `최대 원금 ${acctSuc(peak).slice(1)} 대비 · `) + (equity > 0 ? `순자산 ${acctPct(total / equity)}` : ""))}
        ${tile("그냥 보유였다면", cut ? "—" : withPct(acctSu(hold), holdR), cut ? "" : acctTone(hold), cut ? "체결 이력이 잘려 첫 체결 모름" : `최대 원금 ${acctSuc(hold0).slice(1)} · 첫 진입가 @${acctPx(fpx)}`)}
        ${tile("순환 효과", cut ? "—" : `${acctSu(cyc)} <small>${(totR - holdR) * 100 >= 0 ? "+" : "−"}${Math.abs((totR - holdR) * 100).toFixed(2)}%p</small>`, cut ? "" : acctTone(cyc), "합계 − 그냥 보유 · 수익률 차")}
      </div>
      ${cycleSvg(t, pos, c.rows, now, W)}
      <div class="cy-fills">${chips}</div>
      <p class="cy-note">▲ 매수(초록 = 체결가 ×수량) ▼ 매도(빨강) · 속 빈 표시 = 메이커 · 점선 = 평단 · 주황 = 첫 진입 · 청산하면 이 칸은 사라지고 달력으로 넘어가요</p>
    </div>`;
  }).join(""));
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

async function refreshBinanceAccount(force = false) {
  // ops 탭(refreshOpsStatus)과 스냅샷 탭(tick) 양쪽에서 부르므로 자체 게이트를 둔다.
  // force = 주문 체결 직후(2026-10-06 사용자 «완료되면 플로팅 버튼에 바로») -- 게이트·서버 캐시(10초)를 건너뛰고 떠 있는 버튼을 바로 다시 그린다.
  const now = Date.now();
  if (!force && now - binanceAccountLastFetchAt < BINANCE_ACCOUNT_POLL_MS) return;
  binanceAccountLastFetchAt = now;
  try {
    const res = await fetch(API_BINANCE_ACCOUNT_URL + (force ? "?fresh=1" : ""), { cache: "no-cache" });
    renderBinanceAccount(await res.json());
    if (force) renderOfab();
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

// Shared open/closed state for the per-signal "자세히" detail toggles (model indicators AND
// evidence signals use the same key space, prefixed "model:"/"evidence:" to avoid collisions).
// Kept outside any render function so it survives every re-render -- these lists are rebuilt via
// innerHTML replacement on every tick/poll, so without this a user's open detail panel would snap
// shut a few seconds after they opened it.
const detailOpenKeys = new Set();
// 진입/청산 미리보기 카드의 «자세히». 버튼 **바로 다음 형제**를 연다.
// 열림 상태를 detailOpenKeys 에 남겨, 미리보기를 다시 띄워도 사용자의 선택이 유지된다.
function toggleEntryDetail(btn) {
  const detail = btn.nextElementSibling;
  const open = detailOpenKeys.has("entrycard")
    ? (detailOpenKeys.delete("entrycard"), false) : (detailOpenKeys.add("entrycard"), true);
  if (detail) detail.classList.toggle("open", open);
  btn.textContent = open ? "접기 ▴" : "자세히 ▾";
  btn.setAttribute("aria-expanded", String(open));
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
  // 2026-10-07 «게이지 지지1 ≠ 풋프린트 지지1»(사용자): 여기는 쓸림 필터가 없어 청산맵 마지막 봉 뒤 가격이 이미 지나간 레벨을
  //   남겼다(급락 저점 2,587 아래로 뚫린 2,600·2,592·2,590 이 게이지 지지1~3, 풋프린트는 2,579). 풋프린트와 같은 규칙으로 하나로.
  const since = liqMapKnownUntilSec(map);
  const candles = candleHistoryByAsset[activeSnapshotAsset] || [];
  const pick = (levels, below) => (levels || []).filter((lv) => Number(lv.price) > 0
    && (below ? lv.price < cur : lv.price > cur)
    && liqSweepIdx(candles, since, Number(lv.price)) < 0).slice(0, n);
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
  renderPaper();   // 넓은 화면 프로파일 아래 지지/저항 칸(#mcWall)
  if (!srGaugeMode()) { setH("liquidationMapList", ""); return; }
  if (!sr || (!sr.res.length && !sr.sup.length)) {
    setH("liquidationMapList", `<p class="muted" style="padding:16px;">추정 가능한 밀집 구간이 아직 없습니다.</p>`);
    return;
  }
  setH("liquidationMapList", srRowsHtml(sr));
}

// 지지·저항 게이지 행(이름 · 가격 · 강도 막대 · 거리 %, 가운데 현재가 줄) -- 모바일 목록과 2026-10-06 넓은 화면 모의 판 칸 서랍이 같이 쓴다.
function srRowsHtml(sr) {
  const dp = Math.max(2, pxDp());   // 2026-10-22 사용자 «XRP 지지/저항 소수 4자리»: 고정 2자리는 XRP 1.3914 를 1.39 로 뭉갰다
  const row = (lv, tag, cls) => {
    const pct = Math.round((lv.weight_pct || 0) * 100), dist = (lv.price - sr.cur) / sr.cur * 100;
    return `<div class="liq-level-row ${cls}"><span class="liq-level-tag">${tag}</span>`
      + `<span class="liq-level-price">${fmtNum(lv.price, dp)}</span>`
      + `<div class="liq-level-bar-track"><div class="liq-level-bar-fill" style="width:${Math.max(pct, 4)}%;"></div></div>`
      + `<span class="liq-level-dist">${dist > 0 ? "+" : ""}${fmtNum(dist, 2)}%</span></div>`;
  };
  const cur = `<div class="liq-level-row liq-current"><span class="liq-level-tag">현재가</span>`
    + `<span class="liq-level-price">${fmtNum(sr.cur, dp)}</span><div class="liq-level-bar-track"></div><span class="liq-level-dist">-</span></div>`;
  return [...sr.res.map((lv, i) => row(lv, "저항" + (i + 1), "liq-resistance")).reverse(), cur,
          ...sr.sup.map((lv, i) => row(lv, "지지" + (i + 1), "liq-support"))].join("");
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
      && Boolean(a.okx) === Boolean(b.okx) && JSON.stringify(a.hl || null) === JSON.stringify(b.hl || null)
      && JSON.stringify(a.bybit || null) === JSON.stringify(b.bybit || null);
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
  }
}

// The liquidation map self-triggers the panel list and the snapshot chart right after a fetch resolves.
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

// 2026-10-06 레짐 리본·계산 제거(사용자 «레짐 리본 제거해주고 리소스도 줄이게 레짐 계산도 제거해줘») -- 서버 regime 워커도 껐다.

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
// 2026-10-05 뉴스 카드(사용자 «뉴스 카드를 하나 만들어서» · «중복은 하나로») -- 9개 피드 + jevk5:4b 판정.
//   판정은 제목·요약만 읽고 검증 전이다 → 머리에 «신호 아님» 고정, 방향색은 값에만. 12시간 창, 30초 게이트.
//   같은 글이 여러 출처로 오면 서버(news_dedupe)가 하나로 합치고 sources 에 출처를 모은다.
const NEWS_POLL_MS = 30 * 1000;
const NEWS_SRC = { theblock: "Block", cointelegraph: "CT", coindesk: "CD", tree: "Tree", bwenews: "BWE",
  financialjuice: "FJ", trumpstruth: "Truth", fed: "Fed", sec: "SEC" };
const NEWS_SRC_FULL = { theblock: "The Block", cointelegraph: "Cointelegraph", coindesk: "CoinDesk", tree: "Tree News", bwenews: "BWEnews",
  financialjuice: "FinancialJuice", trumpstruth: "Truth Social", fed: "Fed", sec: "SEC" };
const NEWS_ASSETS = ["ETH", "BTC", "SOL", "XRP", "HYPE", "macro"];
const newsAsset = (a) => (a === "macro" ? "매크로" : a === "other" ? "기타" : a || "");
let newsLastFetchAt = 0, latestNews = null;
const newsView = { relevant: true, coin: false };
async function refreshNews(force) {
  const now = Date.now();
  if (!force && now - newsLastFetchAt < NEWS_POLL_MS) return;
  newsLastFetchAt = now;
  try {
    const res = await fetch("/api/news", { cache: "no-cache" });
    if (!res.ok) throw new Error(`news ${res.status}`);
    latestNews = await res.json();
    renderNews();
    renderMarketCtx();   // ④ 아래 ⑤ 뉴스
  } catch (error) {
    console.error("News fetch error:", error);
  }
}
// 2026-10-06 뉴스 카드 «기울기»·코인 줄과 풋프린트 머리 띠가 같은 계산을 쓴다(사본을 두면 한쪽만 고쳐진다).
const newsRel = (it) => (it.relevant ?? 0) >= 0.5;
function newsTiltHtml(p, h) {   // 최근 h시간 관련 기사의 호재·중립·악재 몫 -- 기사마다 영향 × 관련성 가중
  const now = p.now_ms || Date.now(), sum = { bullish: 0, neutral: 0, bearish: 0 };
  const xs = p.items.filter((it) => it.sentiment && it.ts_ms >= now - h * 3600e3 && newsRel(it));
  xs.forEach((it) => { if (it.sentiment in sum) sum[it.sentiment] += (Number(it.impact) || 0) * (it.relevant ?? 0) || 0.01; });
  const tot = sum.bullish + sum.neutral + sum.bearish, pc = (k) => (tot ? 100 * sum[k] / tot : 0), cnt = (k) => xs.filter((it) => it.sentiment === k).length;
  const bar = tot ? `<i class="b" style="width:${pc("bullish")}%"></i><i class="n" style="width:${pc("neutral")}%"></i><i class="s" style="width:${pc("bearish")}%"></i>` : "";
  return `<div class="nw-tilt"><div class="nw-tilt-lab"><span class="w">${h}시간 · ${xs.length}건</span><span><b class="opt-good">호재 ${cnt("bullish")}</b> ${pc("bullish").toFixed(0)}% · 중립 ${cnt("neutral")} · ${pc("bearish").toFixed(0)}% <b class="opt-bad">악재 ${cnt("bearish")}</b></span></div>
      <div class="nw-tilt-bar" role="img" aria-label="최근 ${h}시간 호재 ${pc("bullish").toFixed(0)}% · 중립 ${pc("neutral").toFixed(0)}% · 악재 ${pc("bearish").toFixed(0)}% (영향×관련성 가중)">${bar}</div></div>`;
}
function newsAssetCounts(p, a) {   // 12시간 창(서버 NEWS_WINDOW_H) 안 그 코인의 판정된 관련 기사 -- 호재·악재 수 · 전체 · 최대 영향
  const xs = p.items.filter((it) => it.sentiment && it.asset === a && newsRel(it));
  return { b: xs.filter((it) => it.sentiment === "bullish").length, s: xs.filter((it) => it.sentiment === "bearish").length,
           n: xs.length, mx: Math.max(0, ...xs.map((it) => Number(it.impact) || 0)) };
}

// ── Macro Trend 카드 (2026-10-28 사용자 «위 지표들을 대시보드 카드 하나 만들어서 모두 추가») ──────────────────────────────
//   /api/macro-trend = dashboard/macro_trend.py. ETF 이후(2024-01-12~) 기준 연구 판정표(L_p3_trend_verdict.csv)의 «사용 가능»·«관찰» 지표를
//   지금 데이터(FRED·DefiLlama·바이낸스, 일·주 단위)로 다시 계산한다. 🔴표본내 연구 · 표본외 검증 전 → 머리에 «신호 아님» 고정, 방향색은 값에만.
//   동행 = «지금 큰 흐름이 코인 강세 쪽인가 약세 쪽인가» · 선행 = «다음 13주 편향». 감시 = 최근 52주 상관이 연구 부호와 같은가(뒤집히면 주황 «이탈»).
const MT_POLL_MS = 10 * 60 * 1000;
let latestMacroTrend = null, mtLastFetchAt = 0;
const MT_TIPS = {
  co: "지금 국면 · 동행 13주: 코인과 같은 13주 동안 같이 움직인 지표다(ETF 이후 2024-01~2026-10 연구). 화살표 = 지난 완결 주의 13주 변화가 «코인 강세 쪽(↑)»인지 «약세 쪽(↓)»인지(변화 부호 × 연구 상관 부호). 13주 변화가 ETF 이후 분포 평균에서 0.5σ 안이면 중립(–).\n앞으로를 말하지 않는다 -- 같은 기간에 같이 간다는 뜻이라 «지금 흐름 확인»용이다. 13주 뒤 수익과의 상관은 유의하지 않았다.",
  lead: "다음 13주 편향 · 선행: 지표의 직전 13주 변화(공표 1주 시차)와 코인의 다음 13주 수익 사이 상관(ETF 이후 연구). 화살표 = 그 상관 부호대로 읽은 다음 13주 편향.\n🔴표본내 상관이다(ETF 이후 ≈2.8년 · 13주 단위 독립 관측 ≈11 · 큰 엔고 충격 2번). 확장 윈도 표본외 검증 전이고, 달러 13주 → 4주 표본외 R² 는 2024–26 에 −1.2% 였다. 크기·진입이 아니라 방향 편향 배경으로만.",
  sum: "세는 규칙: 판정 «사용 가능» 이고 감시 «유효»이며 변동이 살아 있고 데이터가 21일 안인 칸만 센다(채운 칸). 나머지는 테두리만 -- 칸에 올리거나 줄을 누르면 왜 안 세는지 나온다. 머리 숫자 = 그 열의 ↑·↓ 개수일 뿐 가중합이 아니다.",
  watch: "감시: 최근 52주 상관이 연구 부호와 같고 |r|≥0.15 면 «유효», 같지만 약하면 «약화», 반대 부호로 |r|≥0.15 면 «이탈»(주황). 선행은 미래 13주가 필요해 13주 전에 끝나는 52주 창이다. 사용자 결정(2026-10-28): 2020–23 대비 뒤집히거나 새로 생긴 관계도 ETF 이후 꾸준하면 쓴다 -- 대신 다시 뒤집히는지 이 칸으로 지켜본다.",
};
async function refreshMacroTrend(force) {
  const now = Date.now();
  if (!force && now - mtLastFetchAt < MT_POLL_MS) return;
  mtLastFetchAt = now;
  try {
    const res = await fetch("/api/macro-trend", { cache: "no-cache" });
    if (!res.ok) throw new Error(`macro-trend ${res.status}`);
    latestMacroTrend = await res.json();
    renderMacroTrend();
  } catch (error) {
    console.error("Macro trend fetch error:", error);
    mtLastFetchAt = now - MT_POLL_MS + 60 * 1000;   // 실패하면 10분 말고 1분 뒤 다시
  }
}
const mtFmt = (v, unit) => {
  if (v == null || !Number.isFinite(v)) return "–";
  const d = unit === "bp" ? 0 : unit === "$bn" ? (Math.abs(v) < 10 ? 1 : 0) : (unit === "%" ? 1 : 2), sign = v > 0 ? "+" : v < 0 ? "−" : "";
  return `${sign}${Math.abs(v).toFixed(d)}${unit === "$bn" ? "bn" : unit}`;
};
const mtR = (v) => (v == null || !Number.isFinite(v) ? "–" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(2)}`);
// 2026-10-28 시안 C «지표 × 코인 판»(사용자 선택): 고유 지표 한 줄 × [BTC 지금 · ETH 지금 · BTC 13주 뒤 · ETH 13주 뒤] 칸, 줄을 누르면 오른쪽(좁으면 아래)에 근거.
//   칸 상태·이유는 서버(macro_trend.cell_state)가 낸다 -- 요약 개수도 같은 상태로 세므로 칸과 머리 합계가 어긋나지 않는다.
//   채운 칸 = 셈(↑↓, 중립 –) · 테두리만 = 안 셈(약화·변동 없음·지연 회색 / 이탈 주황 / 관찰 흐림). 이유 문장은 칸 title 과 상세 «왜» 줄.
const MT_FAM = [["주식·위험", ["나스닥", "S&P500", "VIX", "STLFSI"]], ["신용", ["HY스프레드", "Baa스프레드"]], ["금리", ["10Y금리", "기대인플레", "기간프리미엄", "연방기금"]],
  ["달러·환율", ["달러지수", "USD/JPY", "EUR/USD", "원/달러", "USD/CNY"]], ["유동성", ["순유동성", "연준BS", "지준", "TGA", "RRP", "스테이블코인"]], ["원자재", ["브렌트"]]];
// 2026-10-28 사용자 «칸의 화살표를 다른 아이콘으로»(시안 A 셰브런): 방향 = 꺾쇠(^ v) · 중립 = 가로줄. 색은 칸 글자색 그대로(초록·빨강·회색)
const MTI_P = { 1: '<polyline points="3,9.5 7,5 11,9.5"/>', "-1": '<polyline points="3,4.5 7,9 11,4.5"/>', 0: '<line x1="3.5" y1="7" x2="10.5" y2="7"/>' };
const MTI = (d) => `<svg class="mt-ico" viewBox="0 0 14 14" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${MTI_P[d]}</svg>`;
const MT_CELLS = [["co", "BTC"], ["co", "ETH"], ["lead", "BTC"], ["lead", "ETH"]];
let mtSel = null;
// 2026-10-28 «오늘» 칸(사용자 «나스닥 줄 + 오늘 칸»): 장중 시세가 있는 줄만 -- 나스닥 ← XYZ100 실시간 · 10Y금리 ← 10년 국채선물(ZN) 가격(CME 약 10분 지연).
//   13주 판정(FRED 금요일 주간값)과 원천·시점이 달라 «오늘 맥락»만. 색 = 가격 방향(국채 가격↓ = 금리↑) -- 풋프린트 머리에 있던 그래프와 같은 규칙.
//   장중은 코인과 «같은 시간에 같이» 움직일 뿐 먼저 움직이지 않는다(10-13 5분봉: 같은 봉 상관 .4 · 앞뒤 1~3봉 ≈0).
// 2026-10-28 S&P500 ← HL xyz:SP500 실시간 · VIX ← ^VIX 5분봉(CBOE 무료 시세 약 15분 지연 + 진행 봉 버림 → 마지막 봉 20~30분 전) -- /api/macro-today.
//   색은 «코인에 우호적인 쪽 = 초록»: 나스닥·S&P·국채 가격은 오르면 초록, VIX 는 내리면 초록(ETF 이후 ETH 동행 r −.57).
const MT_TODAY = { "나스닥": "NQ", "S&P500": "SP", "VIX": "VIX", "10Y금리": "ZN" };
const MT_LIVE = new Set(["NQ", "SP"]);   // 지연 없는 원천(상세 차트에 «지연 구간» 회색 띠를 안 그린다)
const MT_TNAME = { NQ: "나스닥100(XYZ100)", SP: "S&P500 무기한(xyz:SP500)", VIX: "VIX 지수(^VIX)", ZN: "10년 국채선물(ZN) 가격" };
let mtTreas = null, mtIntra = null, mtTodayAt = 0;
async function refreshMtToday() {   // 1분마다 «오늘» 칸을 다시 그린다 · 국채 24시간은 5분마다 받는다(서버 swr 5분)
  const now = Date.now();
  if (now - mtTodayAt < 60000) return;
  mtTodayAt = now;
  if (!mtTreas || now - mtTreas.at > 5 * 60000) {
    try { const res = await fetch("/api/treasury-futures?h=24", { cache: "no-cache" }); if (res.ok) mtTreas = { at: now, p: await res.json() }; }
    catch (error) { console.error("Treasury 24h fetch error:", error); }
  }
  try { const res = await fetch("/api/macro-today", { cache: "no-cache" }); if (res.ok) mtIntra = { at: Date.now() / 1000, p: await res.json() }; }   // S&P 실시간이라 1분마다(서버 swr 30초)
  catch (error) { console.error("Macro today fetch error:", error); }
  if (latestMacroTrend) renderMacroTrend();
}
function mtDay(k) {   // 최근 24시간 [[초, 가격], …] -- 나스닥은 끝에 지금 가격, 국채는 마지막 봉까지(약 10분 지연)
  if (k === "NQ") {
    const t = xyzClock(), live = xyzNow();
    return xyzBars.filter((b) => b[0] + 60 >= t - 86400 && b[0] + 60 <= t).map((b) => [b[0] + 60, b[1]]).concat(live ? [[t, live]] : []);
  }
  if (k === "SP" || k === "VIX") {   // 봉 끝 시각 · S&P 는 끝에 마지막 초 가격(90초 안이면)
    const v = mtIntra && mtIntra.p && mtIntra.p.ok ? (mtIntra.p.symbols || {})[k] : null, t = Date.now() / 1000;
    if (!v) return [];
    const lv = v.live, age = mtIntraAge(k);
    return v.bars.filter((b) => b[0] >= t - 86400).concat(lv && age != null && age < 90 ? [[t, lv.px]] : []);
  }
  const v = mtTreas && mtTreas.p && mtTreas.p.ok ? (mtTreas.p.symbols || {})[k] : null;
  return v && v.bars ? v.bars.filter((b) => b[0] >= Date.now() / 1000 - 86400) : [];
}
function mtIntraAge(k) {   // 초 -- S&P = 마지막 초 가격 나이 · VIX = 마지막 봉 끝 나이. 서버 시계 기준(브라우저 시계가 틀려도) + 받은 뒤 지난 시간
  const v = mtIntra && mtIntra.p && mtIntra.p.ok ? (mtIntra.p.symbols || {})[k] : null;
  if (!v) return null;
  const t = k === "SP" ? v.live && v.live.recv_sec : v.bars.length && v.bars[v.bars.length - 1][0];
  return t ? mtIntra.p.now_sec - t + (Date.now() / 1000 - mtIntra.at) : null;
}
function vixOpen(d) {   // ^VIX 산출 시간: 미 동부 평일 03:15~16:15(09:25~09:30 쉼은 무시). ponytail: 미국 공휴일은 모른다 -- 공휴일엔 빨간불일 수 있다
  const p = Object.fromEntries(new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", weekday: "short", hour: "numeric", minute: "numeric", hourCycle: "h23" }).formatToParts(d).map((x) => [x.type, x.value]));
  const m = Number(p.hour) * 60 + Number(p.minute);
  return p.weekday !== "Sat" && p.weekday !== "Sun" && m >= 195 && m < 975;
}
const mtDayChg = (b) => (b.length > 1 ? (b[b.length - 1][1] / b[0][1] - 1) * 100 : null);
const mtDayCol = (v) => (v == null || Math.abs(v) < 0.02 ? "var(--muted)" : v > 0 ? "var(--good)" : "var(--bad)");
const mtDayColK = (k, v) => mtDayCol(k === "VIX" && v != null ? -v : v);   // VIX 는 내리는 쪽이 초록(위 MT_LIVE 주석)
function mtDayChart(k, W, H, fut) {   // 24시간 선 · fut 시간 > 0 이면 앞으로의 주요 발표(점선 + 이름표, 2시간 안에 몰리면 하나로). 글자는 HTML -- svg 늘이기가 글자를 찌그러뜨린다
  const b = mtDay(k);
  if (b.length < 2) return "";
  const now = k === "NQ" ? xyzClock() : Date.now() / 1000, t0 = now - 86400, t1 = now + fut * 3600;
  const lo = Math.min(...b.map((x) => x[1])), hi = Math.max(...b.map((x) => x[1])), rg = hi - lo || 1e-9;
  const X = (t) => ((t - t0) / (t1 - t0)) * W, Y = (v) => 3 + (hi - v) / rg * (H - 6), P = (t) => `${(((t - t0) / (t1 - t0)) * 100).toFixed(2)}%`;
  const col = mtDayColK(k, mtDayChg(b)), sw = `vector-effect="non-scaling-stroke"`, nowX = X(now).toFixed(1), last = b[b.length - 1];
  const ev = fut ? (latestMacroEvents || []).filter((e) => e.importance === "high" || e.category === "fomc").map((e) => ({ t: Date.parse(e.time_utc) / 1000, nm: e.title_ko || e.title || "지표" }))
    .filter((e) => e.t > now && e.t <= t1).sort((p, q) => p.t - q.t) : [];
  const labs = [];
  ev.forEach((e) => { const g = labs[labs.length - 1]; if (g && e.t - g.t < 7200) { if (!g.nm.includes(e.nm)) g.nm += ` · ${e.nm}`; } else labs.push({ ...e }); });
  const hm = (t) => { const d = new Date(t * 1000); return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`; };
  return `<div class="mt-dc">${labs.map((g, i) => { const pp = (g.t - t0) / (t1 - t0); return `<span class="mt-dl" style="top:${2 + (i % 2) * 13}px;${pp > .85 ? `right:${((1 - pp) * 100).toFixed(2)}%;text-align:right` : `left:${P(g.t)}`};max-width:${(pp > .85 ? pp : 1 - pp) * 100}%" title="${escapeHtml(g.nm)}">${escapeHtml(g.nm)} ${hm(g.t)}</span>`; }).join("")}
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">${fut ? `<rect x="${nowX}" y="0" width="${(W - nowX).toFixed(1)}" height="${H}" fill="rgb(var(--lift) / .04)"/>` : ""}
    <line x1="0" x2="${fut ? nowX : W}" y1="${Y(b[0][1]).toFixed(1)}" y2="${Y(b[0][1]).toFixed(1)}" stroke="var(--line)" stroke-dasharray="2 3" ${sw}/>
    ${ev.map(({ t }) => `<line x1="${X(t).toFixed(1)}" x2="${X(t).toFixed(1)}" y1="0" y2="${H}" stroke="var(--warn)" stroke-opacity=".75" stroke-dasharray="2 3" ${sw}/>`).join("")}
    ${fut && !MT_LIVE.has(k) ? `<rect x="${X(last[0]).toFixed(1)}" y="0" width="${Math.max(0, X(now) - X(last[0])).toFixed(1)}" height="${H}" fill="rgb(var(--lift) / .08)"/>` : ""}
    <path d="${b.map((x, i) => `${i ? "L" : "M"}${X(x[0]).toFixed(1)},${Y(x[1]).toFixed(1)}`).join("")}" fill="none" stroke="${col}" stroke-width="1.6" stroke-linejoin="round" ${sw}/></svg>
    ${fut ? `<div class="mt-dt"><span style="left:0">${hm(t0)}</span><span style="right:${(100 - parseFloat(P(now))).toFixed(2)}%">지금 ${hm(now)}</span><span style="right:0">+${fut}시간</span></div>` : ""}</div>`;
}
// 2026-10-28 사용자 «수신 칸에 정해진 주기마다 데이터가 들어오는지 초록불/빨간불 + 공표/실시간/n분 지연»:
//   불 = 서버 feed_status(원천 계열의 마지막 관측일이 공표 주기 한도 안인가 · 받기 실패 없나) + «오늘» 칸이 있는 줄은 장중 시세도.
//   글 = 공표 주기(매일 공표 · 주 1회 공표) · 나스닥 «실시간» · 10Y금리 «n분 지연»(CME 무료 시세 마지막 봉 나이) · 휴장.
function cmeOpen(d) {   // CME Globex(미 국채선물): 일 17:00 ~ 금 16:00 CT, 매일 16~17시 정비. ponytail: 미국 공휴일은 모른다 -- 공휴일엔 30분 넘게 묵어 빨간불일 수 있다
  const p = Object.fromEntries(new Intl.DateTimeFormat("en-US", { timeZone: "America/Chicago", weekday: "short", hour: "numeric", hourCycle: "h23" }).formatToParts(d).map((x) => [x.type, x.value]));
  const h = Number(p.hour), w = p.weekday;
  return w === "Sat" ? false : w === "Sun" ? h >= 17 : w === "Fri" ? h < 16 : h !== 16;
}
function mtFeed(r, n) {   // → [불 ok|bad|wait, 글, 툴팁]
  const f = r.feed || {}, k = MT_TODAY[n];
  const tips = [`${f.cadence || "공표 주기 모름"} · 마지막 값 ${f.last || "–"}${f.age_d == null ? "" : ` (${f.age_d}일 전 · 한도 ${f.max_d}일)`}`];
  let st = f.ok === false ? "bad" : f.ok ? "ok" : "wait", lab = f.short || "공표";
  if (f.why) tips.push(`늦음: ${f.why}`);
  if (k === "NQ") {
    const fresh = xyzLive && xyzClock() - xyzLive.recv_sec < 90;
    if (!xyzBars.length) { lab = "받는 중"; if (st === "ok") st = "wait"; } else { lab = fresh ? "실시간" : "끊김"; if (!fresh) st = "bad"; }
    tips.push(`장중: 나스닥100 실시간(XYZ100)${xyzBars.length && !fresh ? " -- 90초 넘게 새 값 없음" : ""}`);
  } else if (k === "ZN") {
    const b = mtDay("ZN"), age = b.length ? Date.now() / 1000 - b[b.length - 1][0] : null, open = cmeOpen(new Date());
    if (!open) lab = "휴장";
    else if (age == null) { lab = "받는 중"; if (st === "ok") st = "wait"; }
    else { lab = age < 1800 ? `${Math.max(1, Math.round(age / 60))}분 지연` : "늦음"; if (age >= 1800) st = "bad"; }
    tips.push(`장중: 10년 국채선물(ZN) CME 무료 시세 -- ${open ? (age == null ? "받는 중" : `마지막 봉 ${Math.round(age / 60)}분 전(30분 넘으면 빨간불)`) : "CME 휴장(평일 16~17시 CT 정비 · 주말)"}`);
  } else if (k === "SP") {
    const age = mtIntraAge("SP"), fresh = age != null && age < 90;
    if (age == null) { lab = "받는 중"; if (st === "ok") st = "wait"; } else { lab = fresh ? "실시간" : "끊김"; if (!fresh) st = "bad"; }
    tips.push(`장중: S&P500 무기한 실시간(xyz:SP500)${age != null && !fresh ? " -- 90초 넘게 새 값 없음" : ""} · 주말·미국 휴장 중 값은 코인 쪽 투기`);
  } else if (k === "VIX") {
    const age = mtIntraAge("VIX"), open = vixOpen(new Date());
    if (!open) lab = "휴장";
    else if (age == null) { lab = "받는 중"; if (st === "ok") st = "wait"; }
    else { lab = age < 3600 ? `${Math.max(1, Math.round(age / 60))}분 지연` : "늦음"; if (age >= 3600) st = "bad"; }
    tips.push(`장중: VIX 지수(^VIX) 무료 시세 5분봉 -- ${open ? (age == null ? "받는 중" : `마지막 봉 ${Math.round(age / 60)}분 전(원천 약 15분 지연 + 5분 수집 · 60분 넘으면 빨간불)`) : "산출 안 하는 시간(미 동부 평일 03:15~16:15 · 주말)"}`);
  }
  return [st, lab, `${tips.join("\n")}\n${r.realtime_note || ""}`];
}
function mtTodayCell(n) {   // 표 «오늘» 칸
  const k = MT_TODAY[n];
  if (!k) return `<span class="mt-td na" title="장중 시세 없음 -- 하루·주 단위 공표">—</span>`;
  const b = mtDay(k), v = mtDayChg(b);
  if (v == null) return `<span class="mt-td na" title="받는 중">…</span>`;
  const pc = `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}%`;
  const tip = k === "NQ" ? `나스닥100 실시간(XYZ100) 24시간 ${pc} · 주말엔 거의 안 움직인다`
    : k === "SP" ? `S&P500 무기한 실시간(xyz:SP500) 24시간 ${pc} · 주말엔 거의 안 움직인다`
    : k === "VIX" ? `VIX 지수(^VIX) 24시간 ${pc} · 오르면 위험 회피(빨강) · 약 15분 지연`
    : `10년 국채선물(ZN) 가격 24시간 ${pc} → 금리${v < 0 ? "↑" : "↓"} · CME 약 10분 지연`;
  return `<span class="mt-td" title="${tip} · 13주 판정과 원천·시점이 달라 오늘 맥락만">${mtDayChart(k, 64, 20, 0)}<b style="color:${mtDayColK(k, v)}">${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}%</b></span>`;
}
// 2026-10-28 사용자 «풋프린트 상단 나스닥·2년물·10년물 차트 자리에 Macro Trend 요약»(시안 H1) -- renderMarketCtx 가 #fpRates 에 넣는다.
//   개수 = /api/macro-trend summary 그대로(카드 머리 합계와 같다) · 감시 이탈 · 다음 주요 발표 · 누르면 2번째 카드(Macro Trend)로.
function mtHeadHtml() {
  const p = latestMacroTrend;
  if (!p || !p.ok) return `<button type="button" class="fr-mt" data-mt-go><span class="fr-mtt">Macro<small>${p ? "불러오지 못함" : "불러오는 중…"}</small></span><span class="fr-go">Macro Trend ›</span></button>`;
  const s = p.summary || {}, week = p.rows.map((r) => r.live && r.live.week).filter(Boolean).sort().pop() || "–";
  const row = (a) => { const x = s[a] || {}; return `<span class="fr-c">${a}</span><span>지금 <b class="opt-good">${MTI(1)}${x.co_up || 0}</b> <b class="opt-bad">${MTI(-1)}${x.co_dn || 0}</b></span>`
    + `<span>13주 뒤 <b class="opt-good">${MTI(1)}${x.lead_up || 0}</b> <b class="opt-bad">${MTI(-1)}${x.lead_dn || 0}</b> <span class="muted">· 중립 ${x.neutral || 0} · 안 셈 ${x.excluded || 0}</span></span>`; };
  const off = [...new Set(p.rows.flatMap((r) => Object.entries((r.live || {}).assets || {}).filter(([, c]) => c.state === "off").map(([a]) => `${r.name}→${a}`)))];
  const nx = (latestMacroEvents || []).filter((e) => e.importance === "high" || e.category === "fomc").map((e) => ({ t: Date.parse(e.time_utc), nm: e.title_ko || e.title || "지표" }))
    .filter((e) => e.t > Date.now()).sort((a, b) => a.t - b.t)[0];
  const left = nx ? nx.t - Date.now() : 0;
  return `<button type="button" class="fr-mt" data-mt-go title="Macro Trend 카드로 -- ETF 이후 13주 매크로 추세(표본내 연구, 신호 아님)">`
    + `<span class="fr-mtt">Macro<small>13주 · 기준 주 ${week.slice(5)}</small></span><span class="fr-mtg">${row("BTC")}${row("ETH")}</span>`
    + `<span class="fr-sep"></span><span class="fr-mts"><small>감시</small><span class="${off.length ? "mt-warn" : ""}">이탈 ${off.length}${off.length ? ` · ${escapeHtml(off.join(", "))}` : ""}</span></span>`
    + (nx ? `<span class="fr-sep"></span><span class="fr-mts"><small>다음 발표</small><b class="fr-ev">${escapeHtml(nx.nm)} ${Math.floor(left / 3600e3)}:${String(Math.floor((left % 3600e3) / 60e3)).padStart(2, "0")} 뒤</b></span>` : "")
    + `<span class="fr-go">Macro Trend ›</span></button>`;
}
document.addEventListener("click", (e) => { if (e.target.closest("[data-mt-go]")) el("mtCard")?.scrollIntoView({ behavior: "smooth", block: "start" }); });
function mtOverlay(r, a) {   // 최근 52주: 지표 13주 변화(연구 부호로 맞춤, 위 = 코인 강세 쪽) · 같은 주에 맞춘 코인 13주 수익(서버) · 선행은 지표를 14주(공표 1주 + 13주) 앞으로
  const L = r.live || {}, spark = L.spark || [], c = (L.coin13 || {})[a] || [];
  if (spark.filter(Number.isFinite).length < 20 || c.filter(Number.isFinite).length < 20) return "";
  const z = (xs) => { const v = xs.filter(Number.isFinite), m = v.reduce((p, q) => p + q, 0) / v.length, s = Math.sqrt(v.reduce((p, q) => p + (q - m) ** 2, 0) / v.length) || 1;
    return xs.map((x) => (Number.isFinite(x) ? (x - m) / s : null)); };
  const W = 520, H = 120, lead = r.kind === "lead", off = lead ? 14 : 0, x1 = lead ? 14 : 0;
  const ind = z(spark).map((x) => (x == null || r.assets[a].r_p3 >= 0 ? x : -x)), cz = z(c), n = ind.length, m = cz.length;
  const X = (i) => ((i + 51) / (51 + x1)) * W, Y = (v) => H / 2 - Math.max(-2.6, Math.min(2.6, v)) / 2.6 * (H / 2 - 5);
  const d = (xs, from, o) => xs.map((v, i) => (v == null ? "" : `${i && xs[i - 1] != null ? "L" : "M"}${X(i + from + o).toFixed(1)},${Y(v).toFixed(1)}`)).join("");   // 빈 주에서 선을 끊는다
  const nowX = X(0).toFixed(1), sw = `vector-effect="non-scaling-stroke"`;
  return `<svg class="mt-ov" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="${escapeHtml(r.name)} 13주 변화와 ${a} 13주 수익, 최근 52주">
    ${lead ? `<rect x="${nowX}" y="0" width="${(W - nowX).toFixed(1)}" height="${H}" fill="rgb(var(--lift) / .06)"/>` : ""}
    <line x1="0" x2="${W}" y1="${H / 2}" y2="${H / 2}" stroke="var(--soft-line)" ${sw}/><line x1="${nowX}" x2="${nowX}" y1="0" y2="${H}" stroke="var(--line)" stroke-dasharray="2 3" ${sw}/>
    <path d="${d(cz, 1 - m, 0)}" fill="none" stroke="var(--muted)" stroke-opacity=".6" stroke-width="1.3" ${sw}/>
    <path d="${d(lead ? ind.slice(0, n - 14) : ind, 1 - n, off)}" fill="none" stroke="var(--text)" stroke-width="2" ${sw}/>
    ${lead ? `<path d="${d(ind.slice(n - 15), n - 15 + 1 - n, off)}" fill="none" stroke="var(--text)" stroke-width="2" stroke-dasharray="4 3" ${sw}/>` : ""}</svg>`;
}
function renderMacroTrend() {
  const body = el("mtBody"), p = latestMacroTrend;
  if (!body) return;
  if (!p || !p.ok) {
    body.innerHTML = `<p class="mt-empty muted">${p ? `매크로 데이터를 불러오지 못했다${p.error ? ` — ${escapeHtml(p.error)}` : ""}. 1분 뒤 다시 시도한다.` : "불러오는 중…"}</p>`;
    return;
  }
  const by = (n, k) => p.rows.find((r) => r.name === n && r.kind === k);
  const names = [...new Set(p.rows.map((r) => r.name))], known = new Set(MT_FAM.flatMap(([, ns]) => ns));
  const fams = [...MT_FAM.map(([f, ns]) => [f, ns.filter((n) => names.includes(n))]), ["기타", names.filter((n) => !known.has(n))]].filter(([, ns]) => ns.length);
  const prim = (n) => by(n, "lead") || by(n, "co");
  const live = (r) => (r && r.live) || {};
  if (!names.includes(mtSel)) {   // 처음엔 «셈» 칸 중 ETF 이후 위치가 가장 먼 지표
    const cnt = names.filter((n) => MT_CELLS.some(([k, a]) => ((live(by(n, k)).assets || {})[a] || {}).state === "count"));
    mtSel = (cnt.length ? cnt : names).sort((x, y) => Math.abs(live(prim(y)).z || 0) - Math.abs(live(prim(x)).z || 0))[0];
  }
  const OBS = { obs: " obs", off: " off", weak: " weak", stale: " weak", old: " weak", unknown: " weak" };
  const tile = (r, a) => {
    const c = (live(r).assets || {})[a];
    if (!r || !r.assets[a] || !c) return `<span class="mt-tile na" aria-hidden="true">·</span>`;
    const ch = MTI(c.bias > 0 ? 1 : c.bias < 0 ? -1 : 0), base = c.bias > 0 ? "u" : c.bias < 0 ? "d" : "z";
    return `<span class="mt-tile ${base}${OBS[c.state] || ""}" title="${a} ${r.kind === "co" ? "지금" : "13주 뒤"} · ${escapeHtml(c.why || "")}">${ch}</span>`;
  };
  const gauge = (zv) => zv == null || !Number.isFinite(zv) ? `<span class="mt-g"></span>`
    : `<span class="mt-g" role="img" aria-label="ETF 이후 분포에서 ${zv.toFixed(1)}σ"><i style="left:${(50 + Math.max(-3, Math.min(3, zv)) / 3 * 50).toFixed(1)}%"></i></span>`;
  const s = p.summary || {}, tot = ([k, a]) => { const x = s[a] || {}; return `<span class="mt-tot" title="${a} ${k === "co" ? "지금" : "13주 뒤"} 셈 칸의 ↑·↓ 개수 · ${a} 전체(지금+13주 뒤) 중립 ${x.neutral || 0} · 안 셈 ${x.excluded || 0}"><b class="opt-good">${MTI(1)}${x[k + "_up"] || 0}</b> <b class="opt-bad">${MTI(-1)}${x[k + "_dn"] || 0}</b></span>`; };
  const q = (k, t) => `<button type="button" class="mt-q" ${helpAttrs(k)} title="${t} — 설명">${t}<span aria-hidden="true">?</span></button>`;
  // 격자 줄 = 머리 3 + 계열마다 [이름 줄 + 지표 줄들] -- 지표 줄만 1fr 라 한 화면 모드(카드 = 창 높이)에서 남는 높이를 나눠 갖는다
  const rowsT = ["auto auto auto", ...fams.map(([, ns]) => `auto${" minmax(22px, 1fr)".repeat(ns.length)}`)].join(" ");
  const mx = `<div class="mt-mx" style="grid-template-rows:${rowsT}" aria-label="매크로 지표 × 코인">
      <span class="mt-h l">지표</span><span class="mt-h r">13주 변화</span><span class="mt-h">오늘 24h</span><span class="mt-h hx">위치(σ)</span>
      <span class="mt-h grp">${q("co", "지금 국면")}</span><span class="mt-h grp">${q("lead", "13주 뒤 편향")}</span><span class="mt-h l hx">수신</span>
      <span class="mt-h"></span><span class="mt-h"></span><span class="mt-h"></span><span class="mt-h hx"></span><span class="mt-h c">BTC</span><span class="mt-h c">ETH</span><span class="mt-h c">BTC</span><span class="mt-h c">ETH</span><span class="mt-h hx"></span>
      <span class="mt-tot"></span><span class="mt-tot"></span><span class="mt-tot"></span><span class="mt-tot hx"></span>${MT_CELLS.map(tot).join("")}<span class="mt-tot hx"></span>
      ${fams.map(([f, ns]) => `<span class="mt-fam">${f}</span>` + ns.map((n) => { const r = prim(n), L = live(r), fd = mtFeed(r, n);
        return `<div class="mt-r${n === mtSel ? " sel" : ""}"><button type="button" class="mt-nm" data-n="${escapeHtml(n)}" aria-pressed="${n === mtSel}">${escapeHtml(n)}${r.note ? "<sup>*</sup>" : ""}</button>
          <span class="mt-v">${mtFmt(L.chg13, r.unit)}</span>${mtTodayCell(n)}<span class="mt-gw hx">${gauge(L.z)}</span>${MT_CELLS.map(([k, a]) => `<span class="mt-t">${tile(by(n, k), a)}</span>`).join("")}
          <span class="mt-rt hx" title="${escapeHtml(fd[2])}"><i class="mt-dot ${fd[0]}" role="img" aria-label="${fd[0] === "ok" ? "정해진 주기대로 수신" : fd[0] === "bad" ? "늦음" : "받는 중"}"></i>${fd[1]}</span></div>`; }).join("")).join("")}
    </div>
    <div class="mt-key"><span><i class="mt-tile u">${MTI(1)}</i><i class="mt-tile d">${MTI(-1)}</i>셈</span><span><i class="mt-tile z">${MTI(0)}</i>중립 ±0.5σ</span>
      <span><i class="mt-tile u weak">${MTI(1)}</i>약화·변동 없음·지연</span><span><i class="mt-tile d off">${MTI(-1)}</i>이탈</span><span><i class="mt-tile u obs">${MTI(1)}</i>관찰</span>
      <span>테두리만 = 안 셈 · 칸에 올리거나 줄을 누르면 이유 · ${q("sum", "세는 규칙")} ${q("watch", "감시")}</span></div>`;
  const rs = [by(mtSel, "co"), by(mtSel, "lead")].filter(Boolean), r0 = rs[rs.length - 1], L0 = live(r0);
  const a0 = (x) => (x.assets.ETH ? "ETH" : "BTC");
  const cells = rs.flatMap((x) => ["BTC", "ETH"].filter((a) => x.assets[a]).map((a) => [x, a, x.assets[a], (live(x).assets || {})[a] || {}]));
  const det = `<h4>${escapeHtml(mtSel)}<span class="mt-v">${mtFmt(L0.chg13, r0.unit)} · ${L0.z == null ? "–" : `${L0.z >= 0 ? "+" : "−"}${Math.abs(L0.z).toFixed(1)}σ`}</span></h4>
    <p class="mt-meta">${L0.week || "–"} 주까지 13주 변화 · 수신 ${escapeHtml(mtFeed(r0, mtSel)[2]).replace(/\n/g, " · ")}${r0.note ? `<br><b class="mt-warn">주의</b> ${escapeHtml(r0.note)}` : ""}</p>
    <ul class="mt-why">${cells.map(([x, a, , c]) => `<li><span class="mt-why-k">${a} ${x.kind === "co" ? "지금" : "13주 뒤"}</span>${tile(x, a)}<span>${escapeHtml(c.why || (x.live ? "표본 부족(ETF 이후 13주 변화 30주 미만)" : "지금 값 없음(원천 실패)"))}</span></li>`).join("")}</ul>
    ${MT_TODAY[mtSel] ? (() => { const tk = MT_TODAY[mtSel], b = mtDay(tk), v = mtDayChg(b);
      const sub = tk === "ZN" ? ` · 금리${v < 0 ? "↑" : "↓"} · 약 10분 지연` : tk === "VIX" ? ` · ${v > 0 ? "위험 회피↑" : "위험 선호↑"} · 약 15분 지연` : "";
      return v == null ? "" : `<figure class="mt-ch"><figcaption><span>${MT_TNAME[tk]} 24시간 · 앞 12시간 주요 발표</span><span class="mt-v">오늘 ${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}%${sub}</span></figcaption>${mtDayChart(MT_TODAY[mtSel], 520, 110, 12)}</figure>`; })() : ""}
    <div class="mt-tw"><table class="mt-rtab"><thead><tr><th>관계</th><th>판정</th><th>ETF 이후 r</th><th>순열 p</th><th>전반 / 후반</th><th>2020–23</th><th>최근 52주</th></tr></thead>
      <tbody>${cells.map(([x, a, v, c]) => `<tr><td>${a} ${x.kind === "co" ? "지금" : "13주 뒤"}</td><td>${v.verdict}</td><td>${mtR(v.r_p3)}</td><td>${v.perm_p == null ? "–" : v.perm_p.toFixed(3)}</td><td>${mtR(v.r_first_half)} / ${mtR(v.r_second_half)}</td><td>${mtR(v.r_2020_23)}</td><td>${mtR(c.roll52)}</td></tr>`).join("")}</tbody></table></div>
    ${rs.map((x) => { const a = a0(x), svg = mtOverlay(x, a); return svg ? `<figure class="mt-ch"><figcaption><span>${a} · ${x.kind === "lead" ? "13주 뒤 — 지표를 14주 앞으로" : "지금 — 같은 13주"}</span><span class="mt-v">최근 52주 r ${mtR(((live(x).assets || {})[a] || {}).roll52)}</span></figcaption>${svg}</figure>` : ""; }).join("")}
    <p class="mt-meta">굵은 선 = 지표 13주 변화(위 = 코인 강세 쪽) · 회색 = 코인 13주 수익 · 52주 안 표준화(모양만) · 표본내 연구(13주 독립 관측 ≈11), 신호 아님</p>`;
  const week = p.rows.map((r) => r.live && r.live.week).filter(Boolean).sort().pop(), errs = Object.keys(p.errors || {});
  const late = names.filter((n) => mtFeed(prim(n), n)[0] === "bad");
  const focus = document.activeElement?.closest("#mtBody .mt-nm")?.dataset.n;
  body.innerHTML = `<div class="mt-wrap"><div class="mt-left">${mx}</div><div class="mt-det">${det}</div></div>
    <p class="mt-foot muted">기준 주 ${week || "–"} · FRED·DefiLlama·바이낸스 일·주 단위(1시간마다 갱신)${errs.length ? ` · <span class="mt-warn">원천 실패 ${errs.length}(${escapeHtml(errs.join(", "))})</span>` : ""}${late.length ? ` · <span class="mt-warn">수신 늦음 ${late.length}(${escapeHtml(late.join(", "))})</span>` : ""}</p>`;
  if (focus) body.querySelector(`.mt-nm[data-n="${CSS.escape(focus)}"]`)?.focus({ preventScroll: true });
}
el("mtBody")?.addEventListener("click", (e) => {   // 줄 어디를 눌러도(칸 포함) 그 지표 -- 터치엔 title 이 없어서 이유는 상세에서 본다
  const b = e.target.closest(".mt-r")?.querySelector(".mt-nm");
  if (!b) return;
  mtSel = b.dataset.n; renderMacroTrend();
  if (matchMedia("(max-width: 1500px)").matches) el("mtBody").querySelector(".mt-det")?.scrollIntoView({ block: "nearest", behavior: "smooth" });   // 한 단이면 상세가 표 아래
});

// ── 풋프린트 카드 머리 띠 (2026-10-06 사용자 선택 C) -- 미 국채선물 4종 · 뉴스 기울기 1시간 · ETH/BTC/매크로 게이지(가로) ──
//   넓은 화면(2단)에서만 보인다(styles.css .fp-strip). 한 화면 모드는 카드가 세로 flex 라 띠 높이만큼 차트 칸이 줄고 카드 높이는 그대로다.
//   국채선물 값은 가격(금리 아님, 가격↓ = 금리↑) · CME 무료 시세 ~10분 지연 + 매시 수집 -- 실시간 신호가 아니라 «오늘 금리 흐름»의 지도.
let latestTreasury = null, treasuryLastFetchAt = 0;
async function refreshTreasury() {
  const now = Date.now();
  if (now - treasuryLastFetchAt < 60000) return;
  treasuryLastFetchAt = now;
  try {
    const res = await fetch("/api/treasury-futures", { cache: "no-cache" });
    if (!res.ok) throw new Error(`treasury ${res.status}`);
    latestTreasury = await res.json();
  } catch (error) {
    console.error("Treasury fetch error:", error);
  }
  renderFpStrip();
}
const TREASURY_NM = { ZT: "2년", ZF: "5년", ZN: "10년", ZB: "30년" };
// 최소 호가 단위(CME: ZT 1/256 · ZF 1/128 · ZN 1/64 · ZB 1/32). 1시간 차트의 세로 폭이 4틱보다 좁아지지 않게 -- 꽉 맞추면 1틱 흔들림도 급락처럼 보인다.
const TREASURY_TICK = { ZT: 1 / 256, ZF: 1 / 128, ZN: 1 / 64, ZB: 1 / 32 };
// 2026-10-06(3) 띠 = 금리 4종만, 전폭(사용자 «뉴스는 ⑤ 로 옮기고 그 자리를 국채선물로 가득») · 차트 아래 마지막 봉 시각.
//   🔴금리 화살표 = 국채 가격 차트의 **최근 1시간** 움직임의 반대(사용자 선택 A + «차트와 반대로»): 12시간 첫 봉 대비로 붙였더니
//   차트 끝은 반등(가격↑)하는데 «금리↑»가 떠 거꾸로 읽혔다. 화살표·변화율 둘 다 1시간 기준이고 12시간 변화는 호버에.
let mcPlaced = false;   // 넓은/좁은 배치(mc-cmp)가 한 번이라도 정해졌나 -- 그 전엔 머리 띠(renderFpStrip)를 띄우지 않는다
function renderFpStrip() {
  const box = el("fpStrip");
  if (!box) return;
  const T = latestTreasury && latestTreasury.ok ? latestTreasury.symbols : null;
  // 2026-10-13 시안(아티팩트 댓글 «금리도 5년 10년 것만 아래 4개 판에»): 넓은 화면 금리는 시장 맥락 한 줄의 ⑥ 금리로 옮겼다 -- 머리 띠는 끈다
  // 2026-10-17 사용자 «새로고침하면 옛 국채 차트가 잠깐 보였다 사라진다»: mc-cmp 는 첫 차트 배치(mcPlace) 때 정해지는데 국채 응답이 더 빨리 오면
  //   그 사이(실측 0.15초+) 띠가 64px 로 떴다 꺼졌다 -- 배치가 한 번 정해지기 전엔 띄우지 않는다(mcPlace 가 처음 정할 때 다시 부른다).
  if (!T || !mcPlaced || el("mcBody")?.classList.contains("mc-cmp")) { box.hidden = true; return; }
  box.hidden = false;
  const hm = (ts) => { const d = new Date(ts * 1000), today = new Date().toDateString() === d.toDateString();
    return (today ? "" : `${d.getMonth() + 1}-${d.getDate()} `) + `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`; };
  const cellW = Math.floor(((box.clientWidth || 1800) - 28 - 3 * 24) / 4), SW = Math.max(60, cellW - 96 - 92 - 16), SH = 30;
  const dir = (chg) => `<em class="${chg < 0 ? "u" : ""}">금리${chg < 0 ? "↑" : "↓"}</em>`;   // 가격↓ = 금리↑
  const rate = (k) => {
    const v = T[k], b = v ? v.bars : [];
    const nm = `<b>${k}</b> <span>${TREASURY_NM[k]}</span>`;
    if (b.length < 2) return `<div class="fps-rate"><div class="fps-rn"><div>${nm}</div></div><span></span><div class="fps-rv"><span class="muted">—</span></div></div>`;
    // 2026-10-06 차트 = 최근 1시간 1분봉(사용자 «급변 포착», 서버 treasury_payload). 세로 폭은 최소 4틱 -- 가운데 정렬로 넓힌다.
    const t0 = b[0][0], t1 = b[b.length - 1][0], lo0 = Math.min(...b.map((x) => x[1])), hi0 = Math.max(...b.map((x) => x[1]));
    const pad = Math.max(0, 4 * (TREASURY_TICK[k] || 0) - (hi0 - lo0)) / 2, lo = lo0 - pad, hi = hi0 + pad, rg = hi - lo || 1e-9;
    const X = (t) => 2 + (t - t0) / Math.max(1, t1 - t0) * (SW - 6), Y = (y) => 3 + (hi - y) / rg * (SH - 6);
    const d = b.map((x, i) => `${i ? "L" : "M"}${X(x[0]).toFixed(1)} ${Y(x[1]).toFixed(1)}`).join(" ");
    const last = b[b.length - 1], chg = Number.isFinite(v.chg12_pct) ? v.chg12_pct : (last[1] / b[0][1] - 1) * 100;
    const h1 = b.find((x) => x[0] >= t1 - 3600) || b[0], chg1 = (last[1] / h1[1] - 1) * 100;
    const lab = `<span class="w">1시간</span> ${dir(chg1)}`;
    return `<div class="fps-rate" title="${escapeHtml(v.contract || k)} · ${hm(t0)}~${hm(t1)} 가격(가격↓ = 금리↑) · 12시간 첫 봉 대비 ${chg >= 0 ? "+" : "−"}${Math.abs(chg).toFixed(2)}% · 1시간 ${chg1 >= 0 ? "+" : "−"}${Math.abs(chg1).toFixed(2)}% · CME 무료 시세 약 10분 지연">`
      + `<div class="fps-rn"><div>${nm}</div><div class="fps-lab">${lab}</div></div>`
      + `<div class="fps-ch"><svg width="${SW}" height="${SH}" viewBox="0 0 ${SW} ${SH}" aria-hidden="true"><path d="${d}" fill="none" stroke="var(--text)" stroke-opacity=".85" stroke-width="1.3" stroke-linejoin="round"/>`
      + `<circle cx="${X(last[0]).toFixed(1)}" cy="${Y(last[1]).toFixed(1)}" r="2.2" fill="var(--text)"/></svg><div class="fps-t"><span>${hm(t0)}</span><span>마지막 ${hm(t1)}</span></div></div>`
      + `<div class="fps-rv"><span>${last[1].toFixed(3)}</span><span class="c">1시간 ${chg1 >= 0 ? "+" : "−"}${Math.abs(chg1).toFixed(2)}%</span></div></div>`;
  };
  const html = `<div class="fps-rates">${["ZT", "ZF", "ZN", "ZB"].map(rate).join("")}</div>`;
  if (box._h !== html) { box._h = html; box.innerHTML = html; }
}

// ── 나스닥100 실시간 (2026-10-14 사용자 «XYZ100 실시간 차트») -- Hyperliquid xyz:XYZ100 무기한(NQ 선물 5분 상관 .98, 지연 없음) ──
//   서버 /api/xyz100 = 1분봉 종가 24시간 + 마지막 초 가격(수집기 live_xyz100_collector_20261014). 1초마다 꼬리(since)만 받는다.
//   화면 = 넓은 화면 시장 맥락 «⑥ 나스닥 · 금리 · 1시간» 첫 칸(사용자 시안 A 선택, 10-14).
//   🔴참고 · 신호 아님 -- 나스닥은 ETH 를 앞서지 않는다(10-13 5분봉: 같은 봉 상관 .40, 앞뒤 1~3봉 ≈0). 쓸모 = «이 움직임이 나스닥과 함께인가».
let xyzBars = [], xyzLive = null, xyzLastFetchAt = 0, xyzSkew = 0, xyzBusy = false;   // xyzSkew = 서버 시계 − 브라우저 시계(초)
const xyzClock = () => Date.now() / 1000 + xyzSkew;   // 모든 나이·창 계산은 서버 시계로(브라우저 시계가 틀려도 «끊김»이 맞게)
async function refreshXyz() {
  const now = Date.now();
  // 2026-10-28 나스닥 그래프는 Macro Trend 카드 «오늘» 칸(1분마다 다시 그림)으로 옮겼다 -- 넓은 화면은 지금처럼 1초(급변 칩·시장 맥락), 그 밖은 1분
  if (xyzBusy || now - xyzLastFetchAt < (el("mcBody")?.classList.contains("mc-cmp") ? 1000 : 60000)) return;
  xyzLastFetchAt = now; xyzBusy = true;   // 응답이 1초 넘게 걸려도 겹치지 않게(늦은 옛 응답이 새 봉을 잘라내지 않게)
  let first = false;
  try {
    const since = xyzBars.length ? xyzBars[xyzBars.length - 1][0] - 60 : 0;   // 마지막 두 봉(진행 중 봉 + 막 닫힌 봉)부터
    const res = await fetch(`/api/xyz100${since ? `?since=${since}` : ""}`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`xyz100 ${res.status}`);
    const p = await res.json();
    if (Number.isFinite(p.now_sec)) xyzSkew = p.now_sec - Date.now() / 1000;
    if (!p.ok) { xyzLive = null; return; }
    if (p.full || !xyzBars.length) { first = !xyzBars.length; xyzBars = p.bars; }
    else if (p.bars.length) {
      const t0 = p.bars[0][0], cut = p.bars[p.bars.length - 1][0] - 86400;
      xyzBars = xyzBars.filter((b) => b[0] < t0 && b[0] >= cut).concat(p.bars);
    }
    xyzLive = p.live;
    if (first && latestMacroTrend) renderMacroTrend();   // 첫 24시간이 오면 Macro Trend «오늘»·수신 칸을 바로(1분 기다리지 않게) -- 🔴xyzLive 를 넣은 뒤에(전엔 그 앞이라 첫 1분 «끊김» 빨간불)
  } catch (error) {
    console.error("XYZ100 fetch error:", error);
  } finally {
    xyzBusy = false;
  }
  // 그리기는 시장 맥락 1초 렌더(refreshMarketCtx)가 한다 -- 여기서 또 부르면 1초에 두 번 통째로 다시 그린다
}
function xyzNow() {   // 지금 가격 -- 초 가격이 30초 넘게 묵었으면(수집기 끊김) null
  return xyzLive && xyzLive.px && xyzClock() - xyzLive.recv_sec < 30 ? xyzLive.px : null;
}
// 2026-10-14 사용자 «5년·10년·나스닥 급변하면 칩 알림» -- 서버 /api/macro-spikes(10-13 연구 정의: 5분 수익이 직전 활동 봉 표준편차의 4배 이상).
//   «⑥» 제목 줄 오른쪽 칩. 봉이 끝나고 SPIKE_SHOW_MIN 분 동안 보인다(국채선물은 ~10분 늦게 도착하므로 처음 보일 때 이미 10분 전).
let macroSpk = null, macroSpkAt = 0, macroSpkSkew = 0;
const SPIKE_SHOW_MIN = 40, SPIKE_NM = { ZF: "5년", ZN: "10년", NQ: "나스닥" };
async function refreshMacroSpikes() {
  const now = Date.now();
  if (now - macroSpkAt < 30000 || !el("mcBody")?.classList.contains("mc-cmp")) return;
  macroSpkAt = now;
  try {
    const res = await fetch("/api/macro-spikes", { cache: "no-cache" });
    if (!res.ok) throw new Error(`macro-spikes ${res.status}`);
    macroSpk = await res.json();
    if (Number.isFinite(macroSpk.now_sec)) macroSpkSkew = macroSpk.now_sec - Date.now() / 1000;
  } catch (error) {
    console.error("Macro spikes fetch error:", error);
  }
}
function macroSpikeChips(chip) {   // 종목마다 가장 최근 급변 1개 · 5년·10년이 같은 쪽이면 한 칩. 색 = 가격 방향(국채 가격↓ = 금리↑ · 나스닥↓ = ETH 에 불리) -- 차트 선 색과 같다
  const S = macroSpk && macroSpk.ok ? macroSpk.symbols || {} : null;
  if (!S) return "";
  const now = Date.now() / 1000 + macroSpkSkew;
  const last = (k) => ((S[k] || {}).events || []).filter((x) => x[0] + 300 >= now - SPIKE_SHOW_MIN * 60).pop();
  const ago = (e) => Math.max(0, Math.round((now - e[0] - 300) / 60));
  const tipOf = (k, e) => `${SPIKE_NM[k]} ${new Date(e[0] * 1000).toTimeString().slice(0, 5)} 5분봉 ${e[2] >= 0 ? "+" : "−"}${Math.abs(e[2]).toFixed(2)}% = 평소 변동의 ${Math.abs(e[1]).toFixed(1)}배(기준 4배)`
    + (k === "NQ" ? " · 실시간" : " · CME 무료 시세라 약 10분 늦게 보인다");
  const NOTE = "\n요즘(08-11~10-11) 이런 봉에서 ETH 는 같은 5분 안에 같은 쪽으로 움직였다 -- 국채 가격↓(금리↑)·나스닥↓ 이면 ETH↓(5년물 기준 84%). 미리 알려 주지는 않는다(동시).";
  const out = [], nq = last("NQ"), rs = ["ZF", "ZN"].map((k) => [k, last(k)]).filter(([, e]) => e);
  if (nq) out.push([ago(nq), chip(nq[2] < 0 ? "bad" : "good", `나스닥 ${nq[2] >= 0 ? "+" : "−"}${Math.abs(nq[2]).toFixed(2)}% · ${ago(nq)}분`, tipOf("NQ", nq) + NOTE)]);
  const same = rs.length === 2 && Math.sign(rs[0][1][2]) === Math.sign(rs[1][1][2]);
  (same ? [rs] : rs.map((r) => [r])).forEach((g) => {
    const e = g[0][1], a = Math.min(...g.map(([, x]) => ago(x)));
    out.push([a, chip(e[2] < 0 ? "bad" : "good", `${g.map(([k]) => SPIKE_NM[k].replace("년", "")).join("·")}년 금리${e[2] < 0 ? "↑" : "↓"} · ${a}분`, g.map(([k, x]) => tipOf(k, x)).join("\n") + NOTE)]);
  });
  return out.length ? `<i>급변</i>${out.sort((x, y) => x[0] - y[0]).map((x) => x[1]).join("")}` : "";   // 최근 것부터 -- 좁으면 오래된 쪽이 잘린다
}

function newsAgo(ms, now) {
  const m = Math.max(0, Math.round((now - ms) / 60000));
  return m < 60 ? `${m}분` : `${Math.floor(m / 60)}시간${m % 60 ? ` ${m % 60}분` : ""}`;
}
function renderNews() {
  const card = el("newsCard"), p = latestNews;
  if (!card) return;
  const show = !!(p && p.ok);
  card.hidden = !show;
  const railBtn = document.querySelector('#cardRail [data-go="newsCard"]'); if (railBtn) railBtn.hidden = !show;
  if (!show) return;
  const now = p.now_ms || Date.now(), coin = String(activeSnapshotAsset || "eth").toUpperCase();
  const judged = p.items.filter((it) => it.sentiment);
  const rel = (it) => (it.relevant ?? 0) >= 0.5;
  const w = (it) => (Number(it.impact) || 0) * (it.relevant ?? 0);   // 기울기 가중 = 영향 × 관련성
  const srcs = (it) => (it.sources && it.sources.length ? it.sources : [it.source]);
  // 기울기: 1·6·12시간, 관련 기사의 영향 가중 몫
  const tiltOf = (h) => newsTiltHtml(p, h);
  const tilts = `<div><div class="nw-sec-h">기울기 · 영향×관련성 가중</div><div class="nw-tilts">${[1, 6, 12].map(tiltOf).join("")}</div></div>`;
  // 코인별: 12시간 관련 기사 수 · 호재/악재 막대 · 최대 영향
  const rows = NEWS_ASSETS.map((a) => {
    const { b, s, n, mx } = newsAssetCounts(p, a);
    return `<span class="a${a === coin ? " on" : ""}">${newsAsset(a)}</span><span class="mini" title="호재 ${b} · 악재 ${s} · 중립 ${n - b - s}">${n ? `<i class="b" style="width:${100 * b / n}%"></i><i style="width:${100 * (n - b - s) / n}%"></i><i class="s" style="width:${100 * s / n}%"></i>` : ""}</span><span class="c">${n}건</span><span class="m">${n ? mx.toFixed(1) : "–"}</span>`;
  }).join("");
  const assets = `<div class="nw-assets"><span class="h">코인</span><span class="h">호재 · 악재</span><span class="h c">12시간</span><span class="h m">최대 영향</span>${rows}</div>`;
  const incs = judged.filter((it) => (it.incident ?? 0) >= 0.8);   // 2026-10-05 왼쪽 칸이 아니라 목록 맨 위에 고정(사용자 지시) -- 12시간 창 안에 있는 동안만
  const byS = {}; p.items.forEach((it) => srcs(it).forEach((s) => { byS[s] = (byS[s] || 0) + 1; }));
  const merged = p.items.filter((it) => srcs(it).length > 1).length;
  const srcLine = `<div class="nw-srcs">${Object.entries(byS).sort((x, y) => y[1] - x[1]).map(([s, n]) => `<span title="${NEWS_SRC_FULL[s] || escapeHtml(s)}">${NEWS_SRC[s] || escapeHtml(s)} <b>${n}</b></span>`).join("")}${merged ? `<span>· 여러 출처 합침 <b>${merged}</b></span>` : ""}</div>`;
  // 고영향 상위: 12시간 관련 기사 중 영향×관련성 큰 순 6건
  const top = judged.filter(rel).slice().sort((x, y) => w(y) - w(x)).slice(0, 6);
  const sw = { bullish: ["opt-good", "호재"], bearish: ["opt-bad", "악재"], neutral: ["", "중립"] };
  const tops = top.length ? `<div><div class="nw-sec-h">고영향 상위 · 12시간</div><div class="nw-top">${top.map((it) => `<div class="nw-top-row"><span class="${(sw[it.sentiment] || sw.neutral)[0]}">${(sw[it.sentiment] || sw.neutral)[1]}</span><span class="i">${Number(it.impact).toFixed(1)}</span><span class="tt" title="${escapeHtml(it.title)}">${escapeHtml(it.title.replace(/\s+/g, " ").slice(0, 120))}</span><span class="ag">${newsAgo(it.ts_ms, now)} 전 · ${escapeHtml(newsAsset(it.asset))}</span></div>`).join("")}</div></div>` : "";
  // 12시간 축: x = 시각, y = 영향, 색 = 감성, 크기 = 관련성, 주황 테두리 = 해킹·장애. 지금 코인 기사는 진하게.
  const W = Math.max(320, Math.round(el("newsBody")?.querySelector(".nw-main")?.clientWidth || (card.clientWidth * 0.68) || 700)), H = 150, L = 30, R = 8, T = 10, B = 18;
  const t0 = now - (p.window_h || 12) * 3600e3, x = (t) => L + (W - L - R) * (t - t0) / (now - t0), y = (v) => T + (H - T - B) * (1 - (Number(v) || 0) / 3);
  const grid = [0, 1, 2, 3].map((v) => `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--soft-line)"/><text x="${L - 6}" y="${y(v) + 3}" text-anchor="end">${v}</text>`).join("");
  const ticks = [12, 9, 6, 3, 0].map((h) => `<text x="${x(now - h * 3600e3)}" y="${H - 4}" text-anchor="middle">${h ? `-${h}h` : "지금"}</text>`).join("");
  const col = { bullish: "var(--good)", bearish: "var(--bad)", neutral: "var(--muted)" };
  const dots = judged.filter((it) => it.ts_ms >= t0).slice().reverse().map((it) => {
    const r = 2.2 + 3.6 * (it.relevant ?? 0), op = rel(it) ? (it.asset === coin ? 1 : 0.7) : 0.25;
    const ring = (it.incident ?? 0) >= 0.8 ? ` stroke="var(--warn)" stroke-width="2"` : "";
    return `<circle cx="${x(it.ts_ms).toFixed(1)}" cy="${y(it.impact).toFixed(1)}" r="${r.toFixed(1)}" fill="${col[it.sentiment] || col.neutral}" fill-opacity="${op}"${ring}><title>${escapeHtml(`${newsAgo(it.ts_ms, now)} 전 · ${srcs(it).map((s) => NEWS_SRC[s] || s).join("·")} · ${newsAsset(it.asset)} · 영향 ${Number(it.impact).toFixed(1)}\n${it.title}`)}</title></circle>`;
  }).join("");
  const lane = `<svg class="nw-lane" viewBox="0 0 ${W} ${H}" role="img" aria-label="최근 12시간 뉴스 영향(세로)과 감성(색)">${grid}${ticks}${dots}</svg>`;
  // 목록: 최신부터 60건(관련만 · 이 코인+매크로 토글)
  const pins = incs.slice(0, 5), pinned = new Set(pins);   // 상한 5 -- 검증 전 모델이 incident 를 남발해도 목록이 고정 줄로 덮이지 않게(나머지는 시간순 자리)
  const list = [...pins, ...p.items.filter((it) => !pinned.has(it) && (!newsView.relevant || (it.sentiment ? rel(it) : !it.model)) && (!newsView.coin || it.asset === coin || it.asset === "macro")).slice(0, 60)].map((it) => {
    const t = new Date(it.ts_ms).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", hour12: false });
    const sen = it.sentiment ? ({ bullish: ["b", "호재"], bearish: ["s", "악재"], neutral: ["n", "중립"] }[it.sentiment] || ["n", escapeHtml(it.sentiment)]) : ["w", it.model ? "본문 없음" : "대기"];
    const prob = it.sentiment ? Math.round(100 * Number(it[{ bullish: "p_bull", bearish: "p_bear", neutral: "p_neu" }[it.sentiment]] || 0)) : null;
    const imp = Number(it.impact) || 0, bars = [1, 2, 3].map((k) => `<i class="${imp >= k - 0.5 ? "f" : ""}"></i>`).join("");
    const tag = (it.incident ?? 0) >= 0.8 ? `<span class="tag">해킹·장애</span>` : "";
    const href = /^https?:/.test(it.link) ? it.link : "";
    const title = escapeHtml(it.title.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim());
    const ss = srcs(it);
    return `<div class="nw-row${pinned.has(it) ? " pin" : it.sentiment && !rel(it) ? " dim" : ""}"><span class="t">${t}</span><span class="src${ss.length > 1 ? " multi" : ""}" title="${ss.map((s) => NEWS_SRC_FULL[s] || s).join(" · ")}">${ss.map((s) => NEWS_SRC[s] || escapeHtml(s)).join("·")}</span>
      <span class="sen ${sen[0]}">${sen[1]}${prob != null ? ` ${prob}%` : ""}</span><span class="as${it.asset === coin ? " on" : ""}">${escapeHtml(newsAsset(it.asset))}</span>
      <span class="imp" title="영향 ${imp.toFixed(1)} / 3">${bars}</span>${href ? `<a href="${escapeHtml(href)}" target="_blank" rel="noopener" title="${title}">${tag}${title}</a>` : `<a title="${title}">${tag}${title}</a>`}</div>`;
  }).join("") || `<div class="nw-note">조건에 맞는 기사가 없다.</div>`;
  const chip = (k, on, lab) => `<button type="button" class="opt-chip-btn${on ? " on" : ""}" data-nw="${k}" aria-pressed="${on}">${lab}</button>`;
  setH("newsMeta", `12시간 ${p.items.length}건 · 판정 ${judged.length} · 관련 ${judged.filter(rel).length} · 참고, 신호 아님`);
  const head = `<div class="nw-head"><span class="nw-chips">${chip("relevant", newsView.relevant, "관련만")}${chip("coin", newsView.coin, `${coin}+매크로`)}</span></div>`;
  const prevList = el("newsBody")?.querySelector(".nw-list"), keepTop = prevList ? prevList.scrollTop : 0;   // setH 는 SVG 직렬화 차이로 늘 다시 그린다 -- 스크롤 보존
  // 2026-10-07 ⑤ 다음 24시간은 목록 아래(같은 폭, 사용자 지시) -- innerHTML 교체에 같이 지워지지 않게 카드로 잠깐 빼 두었다가 새 .nw-main 끝에 다시 붙인다(내용·클릭 핸들러는 노드에 그대로).
  const when = el("mcWhen");
  if (when && el("newsBody").contains(when)) card.appendChild(when);
  setH("newsBody", `${head}<div class="nw-sum">${tilts}${assets}${tops}${srcLine}</div><div class="nw-main">${lane}<div class="nw-list">${list}</div></div>`);
  if (when) { el("newsBody").querySelector(".nw-main")?.appendChild(when); nwWhenFit(); }
  const nextList = el("newsBody")?.querySelector(".nw-list"); if (nextList && keepTop) nextList.scrollTop = keepTop;
}
// 2026-10-07 뉴스 카드 크기 유지: 목록 최대 높이에서 ⑤ 의 **실제** 높이를 뺀다(휴대폰은 일정 목록이라 행 수에 따라 높이가 바뀐다). 0 이면(안 그려짐) CSS 기본값 128px.
function nwWhenFit() {
  const w = el("mcWhen"), m = w && w.parentElement;
  if (m && m.classList.contains("nw-main") && w.offsetHeight) m.style.setProperty("--nw-when-h", `${w.offsetHeight + (parseFloat(getComputedStyle(w).marginTop) || 0)}px`);
}
el("newsBody")?.addEventListener("click", (e) => {
  const b = e.target.closest("[data-nw]");
  if (!b) return;
  newsView[b.dataset.nw] = !newsView[b.dataset.nw];
  renderNews();
});

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
  if (typeof renderMarketCtx === "function") renderMarketCtx();   // ⑤ 시간축을 바로 다시 그린다
}


// 2026-10-16 펜 메모(3차 «화살표대로 순서»): 넓은 화면(≥1200) 스냅샷 = 맨 위 [세션(왼쪽) | 코인 탭(가운데) | 경보·다크·알림(오른쪽)] ·
//   스냅샷/운영 관리 = 맨 위 제일 왼쪽(10-17 펜 메모 «카드 머리에서 대시보드 최상단 제일 왼쪽으로», 세션은 그 오른쪽). 운영 관리 화면 = 탭은 맨 위 왼쪽, 세션 가운데.
//   좁은 화면은 옛 자리 그대로(탭 = 맨 위 첫 줄 · 코인 = 카드 머리 · 세션 = CSS 배치).
const PAGE_TABS_WIDE = window.matchMedia ? window.matchMedia("(min-width: 1200px)") : null;
function placeTopLeft() {
  const nav = document.querySelector(".page-tabs"), top = document.querySelector(".topbar"), slot = el("topLeftSlot"), mid = el("topCenterSlot");
  const coins = el("snapshotAssetTabs"), home = el("coinHome"), sess = document.querySelector(".top-right");
  if (!nav || !top || !slot || !mid) return;
  const wide = !!(PAGE_TABS_WIDE && PAGE_TABS_WIDE.matches), snap = activePageTab === "snapshot";
  const coinsHome = () => { if (coins && home && coins.previousElementSibling !== home) home.after(coins); };
  const sessHome = () => { if (sess && sess.parentElement !== top) top.appendChild(sess); };
  if (wide && snap) {
    if (slot.firstElementChild !== nav) slot.prepend(nav);
    if (sess && sess.parentElement !== slot) slot.appendChild(sess);
    if (coins && coins.parentElement !== mid) mid.appendChild(coins);
  } else {
    coinsHome(); sessHome();
    if (wide) { if (nav.parentElement !== slot) slot.appendChild(nav); }
    else if (nav.parentElement !== top) slot.after(nav);
  }
}
if (PAGE_TABS_WIDE && PAGE_TABS_WIDE.addEventListener) PAGE_TABS_WIDE.addEventListener("change", placeTopLeft);   // 🔴CI 검사기(esprima)는 ?.( 를 못 읽는다
function setupPageTabs() {
  document.querySelectorAll(".page-tab").forEach((button) => button.addEventListener("click", () => {
    const target = button.dataset.pageTab; // "ops" | "snapshot" (라이브 탭 제거 08-31 · 알림 탭 → 알림 센터 모달 10-05)
    activePageTab = target;
    el("opsTabPanel")?.classList.toggle("hidden", target !== "ops");
    el("snapshotTabPanel")?.classList.toggle("hidden", target !== "snapshot");
    cardRailSync();   // 2026-09-30 카드 이동 레일도 스냅샷 탭에서만
    ofabSync();   // 2026-09-25 떠다니는 주문 버튼은 스냅샷 탭에서만 -- 떠나면 조작부를 카드로 먼저 돌려놓는다
    document.querySelectorAll(".page-tab").forEach((tab) => {
      tab.classList.toggle("active", tab === button);
      if (tab === button) tab.setAttribute("aria-current", "page"); else tab.removeAttribute("aria-current");
    });
    placeTopLeft();
    if (target === "ops") {
      opsLastFetchAt = 0; refreshOpsStatus();
    } else if (target === "snapshot") {
      chartMarkersLastFetchAt = 0; latestChartMarkers = null; refreshChartMarkers();
      liquidation5mLastFetchAt = 0; refreshLiquidation5mSignal();
      liquidationMapLastFetchAt = 0; refreshLiquidationMap();
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
  rail.querySelectorAll("[data-go]").forEach((b) => { const t = el(b.dataset.go); if (t && !t.hidden && t.getBoundingClientRect().top < y) cur = b; });   // 숨긴 카드(top 0)는 빼야 맨 위에서 엉뚱한 점이 켜지지 않는다
  cur = cur || rail.querySelector("[data-go]");
  rail.querySelectorAll("[data-go]").forEach((b) => { const on = b === cur; b.classList.toggle("on", on); if (on) b.setAttribute("aria-current", "true"); else b.removeAttribute("aria-current"); });
}
// 2026-09-30 «한 화면 모드»(사용자 1920×1080·90%): 넓은 화면에서 카드 하나 = 창 높이 하나 -- 레일 점/스크롤이 카드 단위로 딱 맞게 넘어간다.
//   Footprint 는 차트 상자 높이를, Option 은 행사가 사다리 높이를 «창 높이 − 카드의 나머지»로 계산한다. 계좌는 창 높이까지 늘린다.
//   창 높이 < 900 이면 끈다(가격판이 너무 좁아진다). 켜기/끄기 = 레일 아래 단추(브라우저 기억).
let fitOptLadderH = null, fitOptChartK = FIT0.ck || 1, renderOptionsSoon = false, fitAcctPlotH = FIT0.ap || 0, fitAcctZ = FIT0.az || 1;   // 2026-10-01 Option 차트 셋 높이 배율 · 계좌 성과 차트 높이(한 화면 모드, fitLayout 이 잰다)
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
  if (acct) { acct.style.minHeight = on ? `${vh}px` : ""; if (!on) acct.style.removeProperty("--acctz"); }
  if (opt) opt.style.minHeight = on ? `${vh}px` : "";
  // 2026-10-01(2) Option 칸 높이 = **원래 크기로 매번 계산**(되먹임 없음). 앞 판은 넘칠 때마다 사다리를 깎는 톱니라 한 번 300 에 닿으면
  //   다시 안 커졌고(캐시에도 남음) 차트 상한만 쌓여 «전부 작아지고 바닥은 잘렸다». 이제:
  //   ① 각 칸 = 머리·꼬리(고정) + 그림(원래 높이 = 폭 × viewBox 비율) ② 원래 크기로 들어가면 상한 없음 + 사다리가 1·2줄을 채움
  //   ③ 안 들어가면 만기·흐름에 들어갈 만큼만 상한(이분 탐색). 바닥은 풋프린트처럼 12px 여유. (2026-10-02 감마 곡선 칸 제거)
  let h = on ? (fitOptLadderH || fitLadderFormula()) : null;
  // 화면 밖 카드는 재지 않는다 -- 밖에 있는 동안 잰 값이 틀어져(실측 k 0.6 · 사다리 1200) 돌아올 때 크기가 튀었다
  const inView = (c) => { const r = c.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight; };
  if (on && opt && opt.offsetParent && inView(opt)) {
    const q = (sel) => opt.querySelector(sel), sec = (t) => q(`.opt-sec:has([data-tip="${t}"])`);
    const lane = q("#optLaneBody"), flow = q("#optFlowBody"), blk = q("#optBlkBody"), lad = sec("ladder"), move = sec("move"), when = q(":scope > #mcWhen");
    const box = (c) => c.getBoundingClientRect(), padB = (c) => parseFloat(getComputedStyle(c).paddingBottom) || 0;
    const fix = (c) => {   // [머리·꼬리 고정 높이, 그림 원래 높이]
      const g = c && c.querySelector("svg");
      if (!g) return c ? [Math.max(0, ...[...c.children].map((k) => box(k).bottom)) - box(c).top + padB(c), 0] : [0, 0];
      const gb = box(g), last = Math.max(gb.bottom, ...[...c.children].map((k) => box(k).bottom)), vb = g.viewBox && g.viewBox.baseVal;
      return [gb.top - box(c).top + (last - gb.bottom) + padB(c), vb && vb.width ? (g.clientWidth * vb.height) / vb.width : gb.height];
    };
    if (lane && flow && blk && lad && move) {
      // 2026-10-01(4) 배치(사용자 지시): 위 = 지금·실현감마 | 사다리 | 블록 거래·심리 · 아래 한 줄 = 만기 | 순매수 흐름(2026-10-02 감마 곡선 자리까지).
      //   아래 줄 차트 셋은 같은 배율 k(기본 0.85, 남는 칸의 34% 이내) · 사다리는 위 블록을 채운다. 원래(k=1) 높이 = 지금 높이 / 지금 k.
      const k0 = optChartK(), [la, ln] = fix(lane), [fa, fn] = fix(flow), [bn] = fix(blk);
      const lnB = ln / k0, fnB = fn / k0;
      const bottom = (k) => Math.max(la + k * lnB, fa + k * fnB);
      const whenH = when && when.offsetParent ? box(when).height + (parseFloat(getComputedStyle(when).marginTop) || 0) : 0;
      const avail = box(opt).top + vh - padB(opt) - 12 - whenH - box(move).top;
      const topNeed = bn + 130;   // 블록 거래 목록 + 심리 세 줄(최소) -- 위 블록이 이보다 작으면 넘친다
      let lo = 0.5, hi = 0.95;   // 2026-10-01 0.85 -> 0.95 · 34% -> 38%(사용자 «조금만 더 키워줘»)
      if (bottom(hi) <= Math.min(avail * 0.38, avail - topNeed)) lo = hi;
      else while (hi - lo > 0.01) { const m = (lo + hi) / 2; if (bottom(m) <= Math.min(avail * 0.38, avail - topNeed)) lo = m; else hi = m; }
      const k = Math.round(lo * 50) / 50;   // 0.02 단위 -- 1px 흔들림에 다시 그리지 않게
      if (Math.abs(k - fitOptChartK) > 0.019) { fitOptChartK = k; renderOptionsSoon = true; }
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
    // 2026-10-05 성과 차트를 최소(120)로 줄여도 넘치면 카드 전체를 비율로 줄인다(사용자 «비율을 줄여서 전체 화면에 딱») --
    //   1080p 모니터의 실제 창(~950px)에서 30px 넘쳤다. 원래 크기(zoom 1)에서 재고, 같은 작업 안에서 다시 입혀 깜빡임이 없다.
    acct.style.removeProperty("--acctz"); acct.style.minHeight = `${vh}px`;
    const ab = Math.max(0, ...[...acct.children].filter((k) => k.offsetParent).map((k) => k.getBoundingClientRect().bottom));
    const ar = acct.getBoundingClientRect(), aslack = Math.round(ar.top + vh - parseFloat(getComputedStyle(acct).paddingBottom || 0) - 12 - ab);   // 바닥 여유 12 = 풋프린트와 같게
    const cur = plot.getBoundingClientRect().height, want = Math.round(cur + aslack);
    if (want >= 120) {
      fitAcctZ = 1;
      if (Math.abs(aslack) > 4) { fitAcctPlotH = Math.min(900, want); acct.style.setProperty("--acctplot", `${fitAcctPlotH}px`); setTimeout(renderSnapshotAccount, 0); }   // 2026-10-14 곡선은 이 높이로 1:1 로 다시 그린다(안 그러면 다음 계좌 조회까지 칸 가운데에 작게)
    } else {
      if (fitAcctPlotH !== 120) setTimeout(renderSnapshotAccount, 0);
      fitAcctPlotH = 120; acct.style.setProperty("--acctplot", "120px");
      // 줄인 뒤 **다시 잰다** -- 차트 상자는 120 아래로 안 줄어(축·여백) 계산으로 빼면 모자랐다(1920×960 에서 12px 넘침)
      const ab2 = Math.max(0, ...[...acct.children].filter((k) => k.offsetParent).map((k) => k.getBoundingClientRect().bottom));
      const need = ab2 - ar.top + parseFloat(getComputedStyle(acct).paddingBottom || 0) + 12;   // 원래 크기로 필요한 카드 높이
      // 2026-10-05(2) 머리(제목 줄)는 안 줄인다 -- 카드째 줄이면 «Account» 글자만 다른 카드 제목보다 작아졌다(사용자 «account 글자가 안맞아»)
      const hd = acct.querySelector(":scope > .ops-health-head"), fixed = need - (ab2 - (hd ? hd.getBoundingClientRect().bottom : ar.top));
      fitAcctZ = Math.max(0.7, Math.floor(((vh - fixed) / (need - fixed)) * 100) / 100);
    }
  }
  if (on && acct && fitAcctZ < 1) acct.style.setProperty("--acctz", String(fitAcctZ));   // 본문만 줄인다(CSS html.fit1 #acctCard > :not(.ops-health-head))
  if (h !== fitOptLadderH || renderOptionsSoon) { fitOptLadderH = h; renderOptionsSoon = false; if (typeof renderOptions === "function") renderOptions(); }
  if (on) { try { localStorage.setItem("fitCache", JSON.stringify({ vh: innerHeight, mc: mcNeedH, lad: fitOptLadderH, ck: fitOptChartK, ap: fitAcctPlotH, az: fitAcctZ })); } catch (e) { /* 기억은 편의 */ } }
}
// Option 격자: 머리·커버 줄(~130) 아래 여섯 줄 중 사다리가 넷(칸 머리·칩 ~70 을 빼고) -- 측정 없이 창 높이로.
// 2026-10-07 ⑤ 다음 24시간이 뉴스 카드로 가서(사용자 지시) 예전에 빼던 152(116 + 여백·선 36)를 사다리에 돌려준다.
function fitLadderFormula() { return Math.max(300, Math.min(900, Math.round(((innerHeight - 20 - 130) * 4) / 6 - 70))); }
// 첫 렌더 전에 모드 클래스와 최소 높이를 바로 입힌다 -- 데이터가 오기 전에 상자 높이가 이미 맞아 있다
function fitApplyCached() {
  if (!fitOn()) return;
  document.documentElement.classList.add("fit1");
  fitOptLadderH = FIT0.lad || fitLadderFormula();   // 지난번에 잰 값(같은 창 높이) -- 새로고침 첫 그림부터 맞는다
  ["acctCard", "optCard"].forEach((id) => { const c = el(id); if (c) c.style.minHeight = `${innerHeight - 20}px`; });
  if (fitAcctPlotH) el("acctCard")?.style.setProperty("--acctplot", `${fitAcctPlotH}px`);
  const ac = el("acctCard");
  if (ac && fitAcctZ < 1) ac.style.setProperty("--acctz", String(fitAcctZ));
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
// 2026-10-12 사용자 시안 A «현재가에서 풋프린트 청산 데이터 위/아래로 얼마나 가까운지»: 밀도 층 마지막 열에서
//   **보이는** 칸(t = 무게/clip ≥ 0.25) 중 안 쓸린 것의 가장 가까운 가장자리(칸 중심 ∓ 반 칸 = 가격이 처음 닿는 곳).
//   지지1/저항1(가장 진한 봉우리)이 아니라 «빈 구간이 어디서 끝나나»다. swept = 밀도 층이 끊은 칸(densitySweep).
function liqBandEdges(bins, clip, binWidth, price, swept) {
  let up = null, dn = null;
  for (const b of bins || []) {
    if (!(clip > 0) || !(b.weightPct / clip >= 0.25) || swept.has(b.price)) continue;
    const lo = b.price - binWidth / 2, hi = b.price + binWidth / 2;
    if (lo > price && (up === null || lo < up)) up = lo;
    if (hi < price && (dn === null || hi > dn)) dn = hi;
  }
  return { up, dn };
}
// 2026-10-11 이력 스냅샷 ts = 그 지도를 아는 5분 경계(닫힌 5분봉까지). 레벨(support/resistance_levels)은 여전히 닫힌 1시간봉만 --
//   그래서 레벨이 아는 끝 = 마지막 스냅샷이 든 시간의 시작. 밀도 층은 스냅샷 ts 자체를 쓴다(아래 densitySweep).
function liqMapKnownUntilSec(map) {
  const h = (map && map.heatmap_history) || [];
  const t = h.length ? Date.parse(h[h.length - 1].ts_utc) : NaN;
  return Number.isFinite(t) ? Math.floor(t / 3600e3) * 3600 : Infinity;   // 모르면 아무것도 안 지운다
}

function nearestLiquidationLevel() {
  // 게이지·서랍·가격 지형과 **같은 목록**(srLevelsLive: 현재가 쪽 · 아직 안 쓸린 · 가까운 순)의 첫 칸.
  // 2026-10-07 사용자 «가까운 지지1과 저항1은 보이게»: 더 가까운 한쪽만 그리던 것(08-24)을 **양쪽 다** 그린다 --
  //   현재가가 둘 사이 중간을 오가면 지지선이 나타났다 사라졌다 했다. 이름표 겹침은 priceLabels 가 피한다.
  const sr = srLevelsLive(1);
  if (!sr) return [];
  return [
    { lv: sr.sup[0], color: "var(--liq-support)", tag: "지지1" },
    { lv: sr.res[0], color: "var(--liq-resistance)", tag: "저항1" },
  ].filter((c) => c.lv).map((c) => ({
    val: c.lv.price,
    color: c.color,
    label: c.tag,
    priceLeft: true,   // 2026-09-26 사용자 지시: 지지/저항 가격은 오른쪽 배지가 아니라 왼쪽 이름 옆에
    dashed: true,
    width: Math.max(1, Math.min(4, Math.round(1 + (c.lv.weight_pct || 0) * 3))),
  }));
}


// Keeps candleHistoryByAsset[activeSnapshotAsset]'s rightmost candle live between the 5-min
// maybeFetchSnapshotChartHistory() fetches, mirroring updateChart()'s in-place extend/roll logic
// for the Live tab's own candleHistory -- 2026-08-25, user report: the Snapshot chart's last candle
// sat frozen at whatever /api/market-history last returned while the "현재" price line (redrawn
// every 5s by the call below) kept moving, which read as the whole chart being stuck/shifted by one
// bar. Same bucket math as updateChart(): extend the last candle's high/low/close in place while
// still inside its 5-min bucket, or push a fresh one once the live tick crosses into a new bucket.
// 체결 하나 = 형성 봉 한 번 갱신(2026-10-11). 렌더 틱(400ms)에서만 반영하면 그 사이 고저를 놓치고, 봉 시각을 SSE 시각(1초 묵음)으로
//   갈라 경계의 첫 체결이 직전 봉에 들어갔다. 바이낸스 @trade 가 거래소 봉의 원천이라 체결가·체결 시각을 그대로 쓴다.
function noteLiveTrade(asset, price, tsMs) {
  latestLivePriceByAsset[asset] = price;
  latestLivePriceTsByAsset[asset] = new Date(tsMs).toISOString();
  if (asset === activeSnapshotAsset) updateSnapshotCandleLive();
}

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
  const cl = c.cvLine && fpCv.bg && fpCv.bg.items.includes(c.cvLine) ? c.cvLine : null;   // 2026-10-20 풋프린트 = bg 캔버스의 칸 뒤 선
  if (!(line || tri || cl)) return;            // 아직 안 그렸거나 현재가 표시가 없는 판
  if (line) { line.setAttribute("y1", y); line.setAttribute("y2", y); }
  if (cl && cl.y !== y) {   // 옛 띠와 새 띠만 다시 칠한다(밀도·격자·다른 선은 같은 명령으로 되칠해져 이음매가 없다)
    const a = cl.y;
    cl.y = y; cl.y0 = y - 2; cl.y1 = y + 2;
    fpCvPaint(fpCv.bg, { x0: cl.x0, x1: cl.x1, y0: Math.min(a, y) - 2, y1: Math.min(a, y) + 2 });
    fpCvPaint(fpCv.bg, { x0: cl.x0, x1: cl.x1, y0: Math.max(a, y) - 2, y1: Math.max(a, y) + 2 });
  }
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
        noteLiveTrade(priceWsAsset, price, Number(d.T));
        updateLivePriceFast(price);
        const qty = Number(d.q);
        if (qty > 0 && FLOW_ASSETS.has(priceWsAsset) && priceWsAsset === activeSnapshotAsset) {
          footprintLiveAdd(price, qty, Number(d.T), !!d.m,
                           d.t == null ? null : Number(d.t));
          // 체결이 곧 셀의 변화다. 스로틀은 maybeRenderSnapshotChartNow 안에 있다(모드별).
          // 2026-10-20 그 사이 프레임은 형성 봉만 캔버스에 다시 칠한다(scheduleFpLiveFrame -- rAF 로 묶어 프레임당 한 번 · 먼저 세야 지금 전체 렌더가 이 체결까지 칠한 걸로 남는다).
          scheduleFpLiveFrame();
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
                            + `&sinceSpot=${spotSupply1sSince}&sinceBybitLiq=${bybitLiq1sSince}`,
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
  bybitLiq1s = new Map(); bybitLiq1sSince = 0;
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
  (payload.bybitLiq || []).forEach((r) => {
    bybitLiq1s.set(r[0], [r[1], r[2], r[3], r[4]]);
    if (r[0] > bybitLiq1sSince) bybitLiq1sSince = r[0];
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
  bybitLiq1s.forEach((_v, k) => { if (k < floor) bybitLiq1s.delete(k); });
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
    + `&sinceSpot=${spotSupply1sSince}&sinceBybitLiq=${bybitLiq1sSince}`;
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
let helpKey = null;   // 열린 «?» 설명 팝업(helpOpen)의 data-tip -- 다시 그려도 aria-expanded 가 맞게
const helpAttrs = (k) => `data-tip="${k}" aria-haspopup="dialog" aria-expanded="${helpKey === k}"`;
let mcTipOf = (k) => MC_TIPS[k];   // renderMarketCtx 가 코인별 고래·리테일 경계로 바꿔 끼운다
const OPT_TIPS = {
  move: "옵션 가격에 들어 있는 기대 움직임. 1σ = 가격 × DVOL × √(기간/1년). 정규분포라면 68% 가 이 안이지만 2026년 실측은 종가 기준 5분 83% · 1시간 84% · 24시간 77% 가 안에 들었다(평균 크기는 맞고, 대부분은 띠보다 작게 움직이다 가끔 크게 벗어난다 — 3σ 넘는 움직임이 정규분포의 약 6배). 봉 중 한 번이라도 띠를 건드린 비율은 5분 33% · 1시간 38%. 시간대 차이도 크다: 04~11 UTC 는 띠가 넉넉하고(안 90% 근처) 13~15 UTC 미국장 초반은 빠듯하다(안 68% 근처). 손절이 5분 1σ 보다 훨씬 좁으면 소음에 걸리기 쉽다.\n차트 띠는 봉 시가에 고정된다: 5분 띠 = 이번 5분봉 시가 ± 5분 1σ, 1시간 띠 = 이번 정시 첫 봉 시가 ± 1시간 1σ(폭도 봉이 열릴 때 값으로 고정). 가격이 띠를 벗어나면 옵션시장 예상보다 큰 움직임.\nIV−실현(VRP) = 30일 내재(DVOL) − 지난 7일 실현(1시간봉): 기간이 달라 30일 실현으로 재면 부호가 34% 다르다. 양수면 옵션시장이 최근보다 큰 움직임을 값에 넣는 중(보통 양수). 어느 기간으로 재도 앞으로의 VRP 예측력은 없었다.\n우리 검정(2026년만): DVOL 24시간 1σ 안에 실제로 든 날 77% · 크기 비 1.00(과거 24h 실현변동성 폭은 1.32 로 좁다). 평균 크기는 맞고 꼬리는 두껍다 — 벗어나는 23% 는 크게 벗어난다.\nVoV = 내재 변동성(DVOL)의 시간당 로그 변화 흔들림, 지난 24시간. ✅검증: 크면 다음 24시간 DVOL 이 더 크게 바뀐다(271일) · VoV 상위 20% 날은 24시간 1σ 띠 적중 66%(평소 80%) — 단 «최근 실현 변동성이 IV 보다 크다»와 같은 정보.\n꼬리 확률 = 옵션 가격(스마일)을 행사가로 미분한 위험중립 확률로, 만기에 선도가에서 2·3·5% 넘게 벗어날 확률(위·아래 합, Breeden·Litzenberger 1978). 🟡실제보다 크게 나오는 방향(±2% 예측 20% vs 실제 18% · ±3% 10% vs 7%)이나 46만기로는 확정 못 함. DVOL 로 환산한 확률(±2% 를 +15pp 과대)보다는 확실히 실제에 가깝다.",
  rg: "실현 감마 국면(2026-10-02, 딜러 감마를 대신한다): 직전 12시간 ETH 5분 종가 수익률의 1차 자기상관. 음수 = 움직임이 곧 되돌아가는 장(롱감마형 — 누가 헤지하든 결과가 «누름»), 양수 = 이어지는 장(숏감마형 — «키움»). 딜러가 누구인지 가정하지 않고 효과를 가격에서 직접 잰다.\n왜 바꿨나: Deribit 은 누가 딜러인지 공개하지 않는다. 체결 기반·블록만·RFQ·관행·뒤집은 관행 등 여섯 가정 모두 검증에서 일관된 지지가 없었고(체결 기반은 종목-시점의 7.9% 에서 «딜러 포지션 > 미결제»), 업계(Glassnode·Amberdata)도 같은 가정을 검증 없이 쓴다.\n분위 = 2021~24 ETHUSDT 5분(정시 표본 35,051개) 중 위치. 국면 = 하위 1/3(−0.077 이하) 되돌림 쪽 · 상위 1/3(+0.012 이상) 추세 쪽.\n우리 검정(손대지 않은 2025-01~2026-09): 다음 4시간 실현 변동성 ÷ 평소 예측(HAR) = 되돌림 쪽 0.94배 · 보통 0.99배 · 추세 쪽 1.02배 — 크기 정보로 약하게 맞는다. 🔴국면은 오래 안 간다(지금 값이 다음 4시간 값을 거의 예측 못 함) · 큰 움직임의 1시간 되돌림 차이(56% vs 52%)는 확정 못 함 — «되돌림 장이니 역매매»로 쓰지 말 것. 체결 기반 딜러 감마와 같은 조건으로 맞붙이면 정확도는 구분 불가였지만, 가정이 없고 우연 점검에서 더 강했다(p .002 vs .07). ETH 만 검증.",
  exp: "Deribit 만기(매일·매주 금·월말·분기말 08:00 UTC = 17:00 KST). 시각은 모두 KST. 규모 = 콜+풋 미결제 × 지수 = 명목 달러(실제 옵션 값어치인 프리미엄은 그 0.2% 안팎). max pain = 만기에 보유자에게 줄 내재가치 합이 가장 작은 결제가(프리미엄 무시 · 누가 보유했는지 모른 채 미결제만으로 계산). 차트·사다리의 max pain 선은 늘 «가까운 만기» 하나이고, 그 만기 미결제는 보통 전체의 2% 안팎이다(라벨에 비중). P/C = 풋÷콜 미결제 수량 — 심리 지표가 아니다: 풋 매도는 강세·풋 매수는 헤지일 수 있고, 정의마다 값이 크게 다르다(가까운 만기 1.5 · 7일 1.0 · 전 만기 0.6 · 프리미엄 0.2).\n우리 검정: 만기 날 행사가로 끌려가는 핀닝·자석은 없었다(f 0.49~0.52, 표본 밖 1년 동전). 큰 만기 전후는 «이벤트 회피»(레버리지 낮추기) 용도.\n시간축 그림은 풋프린트 카드 맨 아래 왼쪽 «옵션 만기».",
  mood: "리버설·버터플라이는 «7일 고정만기»로 잰다: 만기마다 외가격 옵션의 델타로 보간해 정확히 델타 ±0.25 의 IV 를 구하고, 7일 양옆 두 만기를 시간으로 보간한다(30일 값은 괄호). 2026-10-01 까지는 가까운 만기(늘 24시간 미만) 최근접 행사가로 쟀는데 실제 델타가 0.15~0.31 이고 1시간에 평균 3.6pt 흔들려 잡음이었다(7일 고정만기는 0.58pt). 단위 pt = IV %포인트(가격 % 아님), IV 는 Deribit 평가값(mark_iv).\n25Δ 리스크 리버설 = 델타 +0.25 콜 IV − 델타 −0.25 풋 IV(지금가에서 위아래 비슷한 거리). 음수로 깊으면(−5pt 쯤 아래) 하락 방어 풋 수요 = 공포 · 양수면 상승 콜에 웃돈 · ±1pt 안은 중립.\n버터플라이 = (25Δ 콜 IV + 25Δ 풋 IV)/2 − ATM IV = 스마일이 휜 정도. 클수록 «방향은 몰라도 크게 튈» 꼬리에 값이 붙음. 작은 양수가 평상시.\n기간 구조 = 7·30·60일 고정만기 ATM IV(연율 %, 양옆 만기의 총분산 보간). 가까운 일간 만기는 남은 시간에 미국장이 드느냐에 따라 7일 대비 0.71~0.92배로 출렁여 뺐다(만기별 값은 «옵션 만기» 시간축 선). 뒤로 갈수록 높으면 정상(콘탱고) · 앞이 더 높으면(역전) «지금 당장» 큰 움직임을 값에 넣는 스트레스(급락·이벤트 직전). IV 41 ≈ 하루 1σ ±2.1%(41/√365).\n블록 거래 = 장외에서 합의해 거래소에 올린 큰 거래(원자료 다리 그대로, 전략 이름 추정 안 함).\n우리 검정: 아직 없음 — 스큐는 과거분을 살 수 없어 2026-09-28 부터 쌓는 중. 예측력 모름 → 매매 신호 말고 «분위기가 바뀌었나»(리버설 급락·기간 구조 역전) 확인용.\n기간 구조 기울기 = 고정만기 ATM IV 의 1일−7일 · 7일−30일. 1일이 7일보다 높으면 «역전»(주황). ✅검증: 역전이면 다음 24시간 실현 변동성이 DVOL 예상의 약 1.38배(45일, DVOL·최근 실현 통제 후에도 남음 · 겹치지 않는 하루 1표본 33일로는 경계선). 7일−30일도 같은 방향.\nATM 호가 폭 = 7일 근처 만기 ATM 옵션 매도−매수 호가를 IV 포인트로 — 넓으면 마켓메이커가 위험을 피한다. ⏳검정 불가(과거 호가 없음) — 10-01부터 쌓는 중, 나중에 검증.\n옵션 선도 − 지수 = 같은 스냅샷의 Deribit 가까운 만기 선도가와 Deribit 지수 차(bp). 대개 ±3bp 안이고 10분 변화 SD 1.9bp — 그 정도는 잡음. ⏳검정 불가(과거 Deribit 지수 없음, 09-28부터 쌓는 중) — «콜 수요면 +» 같은 해석은 아직 근거가 없다, 나중에 검증.\n(2026-10-02 의미 검증 뒤 남긴 옵션 정보 — 크기·국면만 말하고 방향은 말하지 않는다. 뺀 줄(검증 결과): 옵션 내재 폭(DVOL 과 동률) · 위험중립 왜도·첨도(왜도는 반대 방향, 늘 음수·늘 3 초과) · 변동성 순매수(이후 24h 예측력 없음) · 블록 요청자(순베가 = 블록 몫과 중복 · 순델타 방향 정보 없음) · O/S(높을수록 오히려 조용) · 정산 창(07~08 UTC 는 평소의 0.8배로 조용 · 미결제 크기와 무관) · 이벤트 예상 폭(공식이 0 으로 잘림).)",
  ladder: "행사가 사다리: 세로 = 행사가(지수 ±6%, 위 = 비쌈 · 풋프린트와 같은 방향). 왼쪽 빨강 = 풋 미결제, 오른쪽 초록 = 콜 미결제(달러). 흰 점선 = 지금 가격, 주황 점선 = 가까운 만기 max pain. 칩으로 범위(내일 만기 · 7일 안 만기 합 · 전 만기)를 바꾼다.\n우리 검정: «미결제가 큰 행사가로 가격이 끌린다(자석)»는 1년 표본 밖에서 동전 — 벽·지지저항으로 읽지 말고 «계약이 어디에 쌓였나»로만.\n청록 선 = 가까운 만기 전에 한 번이라도 닿을 확률 50%(실선)·25%·10%(점선), 옅은 띠 = 50% 안쪽. Deribit 행사가별 IV 로 역산한 위험중립 확률이고 «닿음 ≈ 2×그 가격을 넘어 끝날 확률» 근사다(ETH·BTC). 주황 실선 = 내 청산가, 초록·빨강 점선 = 내 지정가 매수·매도. 참고 표시 — 방향 신호가 아니다.",
  blocks: "Deribit 블록 거래(장외에서 맞춘 큰 옵션 거래, 지난 24시간): 한 줄 = 시각 · 명목 금액 · 다리별 매수/매도 만기 행사가 ×수량. 방향은 테이커 기준(RFQ 는 요청자 · 직접 거래는 수락자). 줄 머리 = 다리 구조로 분류한 전략 이름(Deribit 자체 구조 코드와 313/313 일치) · «+ 선물 헤지» = 같은 블록에 선물 다리가 붙음(보통 델타 중립 패키지) · 델타/베가 롱·숏 = 요청자 쪽 순델타·순베가 부호(체결 IV 로 계산, 총량의 10% 미만이면 중립). 단일 다리 블록은 다른 곳 헤지의 일부일 수 있어 의도를 모른다. 신규/청산·신원은 모른다. 금액 = 다리별 명목 합(스프레드면 두 다리가 다 더해진다).\n우리 검정 없음 — 참고로만.",
  lane: "옵션 만기(지금 ~ +120시간, 블록 거래는 가운데 «블록 거래» 칸): 시각 KST. 막대 = 다가올 만기 규모(콜+풋 미결제 × 지수 = 명목 달러) · pain = max pain(그 만기 미결제만으로 계산) · P/C = 풋÷콜 미결제 수량(심리 지표 아님) · 청록 선 = 만기별 ATM IV(가까운 만기는 남은 시간이 짧아 시간대에 따라 출렁인다).\n우리 검정: 만기 날 행사가로 끌려가는 핀닝·자석은 없었다(f 0.49~0.52). 큰 만기 전후는 «이벤트 회피»(레버리지 낮추기) 용도.",
  flow: "옵션 순매수 흐름(지난 24시간, 정시 버킷): 초록 = 콜 매수−매도, 빨강 = 풋 매수−매도(위 = 순매수 · 아래 = 순매도, 기초자산 수량). 청록 선 = 테이커 체결 순델타 누적(콜 매수·풋 매도 +, 콜 매도·풋 매수 −) — «옵션 시장을 통해 테이커가 롱으로 얼마나 기울었나».\n방향은 Deribit 공개 체결의 테이커 방향(블록 포함). «신규 약 N%» = 그 시간 새로 열린 계약의 비율(체결량 V · 미결제 변화 ΔOI 로 (V+ΔOI)/2V, 매수·매도 양쪽 몫 기준) — 이 비율은 정확하지만 테이커와 메이커 중 누가 열었는지는 모른다(테이커 몫은 약 70% 만 확정). 예: 순매수가 0 근처여도 미결제가 크게 늘면 롱과 숏이 양쪽으로 새로 쌓인 것. «강제청산» = 테이커가 강제청산된 체결(Deribit 표시, 매시 history 로 대조). 만기로 사라지는 계약은 미결제 변화에서 뺀다. 검정 전 — 참고.",
};
// 2026-10-02 실현 감마 국면(사용자 «딜러 감마를 실현 감마로 교체»): 직전 12시간(5분 종가 145개 → 수익률 144개)의 1차 자기상관.
//   정의·기준 분포 = scripts/research_eth_realized_gamma_regime_20261002.py 의 rg1_12h(평균을 뺀 acf1) · 2021~24 ETHUSDT 정시 표본 분위.
//   마감봉만 쓴다(형성 중 봉은 값이 계속 변한다). 순수 함수 -- test/test_market_ctx_patterns_20260929.mjs 가 본문을 떼어 돌린다.
const RG_REF = [[1, -0.2952], [5, -0.2068], [10, -0.1671], [20, -0.1185], [30, -0.0866], [40, -0.0587], [50, -0.0321],
                [60, -0.0061], [70, 0.022], [80, 0.0558], [90, 0.1017], [95, 0.1419], [99, 0.2252]];
function optRealizedGamma(candles, nowS = Date.now() / 1000) {
  const cl = (candles || []).filter((c) => c.time + 300 <= nowS && c.close > 0).map((c) => Math.log(c.close)).slice(-145);
  if (cl.length < 145) return null;
  const r = cl.slice(1).map((v, i) => v - cl[i]), m = r.reduce((a, b) => a + b, 0) / r.length, x = r.map((v) => v - m);
  let num = 0, den = 0;
  x.forEach((v, i) => { den += v * v; if (i) num += v * x[i - 1]; });
  if (!(den > 0)) return null;
  const v = num / den, hi = RG_REF.findIndex(([, q]) => q >= v);
  const pct = hi < 0 ? 99.5 : hi === 0 ? 0.5 : RG_REF[hi - 1][0] + (RG_REF[hi][0] - RG_REF[hi - 1][0]) * (v - RG_REF[hi - 1][1]) / (RG_REF[hi][1] - RG_REF[hi - 1][1]);
  return { v, pct, side: v <= -0.0769 ? "rev" : v >= 0.0124 ? "trend" : "mid" };   // 1/3·2/3 분위 경계
}
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
// 2026-10-01 한 화면 모드에서 만기·흐름 차트의 높이 배율(fitLayout 이 남는 칸으로 정한다) -- 글자 크기는 그대로, 그림 높이만.
function optChartK() { return document.documentElement.classList.contains("fit1") ? fitOptChartK : 1; }
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
      // 2026-10-26 끝(+120h 가까이) 만기는 규모 글자가 오른쪽 가장자리에서 «$76…»로 잘렸다 -- 넘칠 때만 막대 왼쪽으로(넘치지 않으면 그대로)
      + (cx + 12 + (fs + 1) * 0.62 * optUsd(v).length > W
        ? `<text x="${(cx - 10).toFixed(1)}" y="${(ly + 11).toFixed(1)}" font-size="${fs + 1}" font-weight="800" fill="var(--warn)" text-anchor="end">${optUsd(v)}</text>`
        : `<text x="${(cx + 12).toFixed(1)}" y="${(ly + 11).toFixed(1)}" font-size="${fs + 1}" font-weight="800" fill="var(--warn)">${optUsd(v)}</text>`)
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

// 행사가 사다리(1-B, 2026-09-28): 옵션 카드 폭에 맞춘 세로 막대. 범위 칩은 브라우저에 기억한다. 2026-10-02 감마 열 제거(딜러 감마 교체).
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
// 제목 옆 한 줄 = 수집 시각(2026-10-02 딜러 감마 «커버» 줄 제거와 함께 시각만 남김)
const optHead = (o) => `<span class="opt-cov-h" role="status">Deribit · 참고, 신호 아님${optAsOf(o)}</span>`;
let optLadderScope = (() => { try { return localStorage.getItem("optLadder") || "week"; } catch (e) { return "week"; } })();
// 범위 칩 -- 사다리가 비거나 «만기 교체 중»일 때도 보여야 다른 범위로 옮길 수 있다(2026-09-30 검증)
// 2026-10-08 닿음 등고선(사용자 시안 3 선택): 가까운 만기 행사가별 IV 스마일(서버 /api/opt-smile) → «만기 전 한 번이라도 닿을 확률».
//   위에서 끝날 확률 = N(d2)(σ = 그 가격의 스마일 IV · 고정 행사가) · 닿음 ≈ 2×min(위, 아래)(반사 원리 근사). 위험중립 · 참고, 신호 아님.
let latestSmile = {};
function optNcdf(z) {
  const t = 1 / (1 + 0.2316419 * Math.abs(z)), d = 0.3989423 * Math.exp(-z * z / 2);
  const q = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z > 0 ? 1 - q : q;
}
function optTouchModel(cur, px) {
  const s = latestSmile[cur], T = s && s.ok ? (s.exp_ms - Date.now()) / 31536e6 : 0;
  if (!(T > 0) || !(px > 0)) return null;
  const sig = (p) => { const x = Math.max(-s.xlim, Math.min(s.xlim, Math.log(p / s.S))); return Math.max(0.05, s.coef[0] * x * x + s.coef[1] * x + s.coef[2]); };
  const up = (p) => { const v = sig(p); return optNcdf((Math.log(px / p) - 0.5 * v * v * T) / (v * Math.sqrt(T))); };
  const touch = (p) => Math.min(1, 2 * Math.min(up(p), 1 - up(p)));
  const at = (t, a, b) => { for (let i = 0; i < 50; i++) { const c = (a + b) / 2; if ((touch(a) - t) * (touch(c) - t) <= 0) b = c; else a = c; } return (a + b) / 2; };
  return { exp_ms: s.exp_ms, touch, level: (t, above) => (above ? at(t, px * 1.0001, px * 1.25) : at(t, px * 0.75, px * 0.9999)) };
}
// 내 청산가(포지션) · 내 지정가(미체결 LIMIT). 계좌를 못 읽었으면(open_orders === null) 지정가는 안 그린다 -- «모름»이지 «없음»이 아니다.
function optMyLevels(cur, lo, hi) {
  const a = latestBinanceAccount;
  if (!a || !a.ok) return [];
  const mine = (sym) => sym === `${cur}USDT` || sym === `${cur}USDC`;
  const out = (a.positions || []).filter((p) => mine(p.symbol) && p.liquidation_price > 0).map((p) => ({ p: p.liquidation_price, kind: "liq", t: "청산" }));
  (a.open_orders || []).filter((o) => mine(o.symbol) && o.price > 0)
    .forEach((o) => out.push({ p: o.price, kind: o.side === "BUY" ? "buy" : "sell", t: o.side === "BUY" ? "매수" : "매도" }));
  return out.filter((r) => r.p >= lo && r.p <= hi);
}
const optPct = (v) => (v < 0.01 ? "<1%" : v > 0.99 ? ">99%" : `${Math.round(v * 100)}%`);
const optPx1 = (v) => (v >= 100 ? v.toLocaleString("en-US", { maximumFractionDigits: 1 }) : optQ(v));
// 막대 뒤: 닿음 50%(실선)·25%·10%(점선) 위아래 + 50% 안쪽 옅은 띠 + 오른쪽 끝 값 표식
function optTouchBack(m, Y, W, lo, hi) {
  if (!m) return "";
  const ln = [[0.5, 0.95, ""], [0.25, 0.6, "6 4"], [0.1, 0.38, "6 4"]].map(([t, op, dash]) => ({ t, op, dash, a: m.level(t, true), b: m.level(t, false) }));
  const inR = (p) => p >= lo && p <= hi;
  let s = `<rect x="30" y="${Y(Math.min(ln[0].a, hi)).toFixed(1)}" width="${W - 34}" height="${(Y(Math.max(ln[0].b, lo)) - Y(Math.min(ln[0].a, hi))).toFixed(1)}" fill="var(--option)" fill-opacity=".07"/>`;
  ln.forEach(({ t, op, dash, a, b }) => [a, b].filter(inR).forEach((p) => {
    const txt = `${Math.round(t * 100)}% ${optQ(p)}`, w = txt.length * 6.2 + 12, y = Y(p);
    s += `<line x1="30" x2="${W - 4}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" stroke="var(--option)" stroke-opacity="${op}" stroke-width="1.1"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`
      + `<rect x="${(W - 4 - w).toFixed(1)}" y="${(y - 8).toFixed(1)}" width="${w.toFixed(1)}" height="15" rx="7.5" fill="var(--chart-bg)" fill-opacity=".9"/>`
      + `<text x="${W - 9}" y="${(y + 3.5).toFixed(1)}" font-size="10" font-weight="800" font-family="var(--font-mono)" fill="var(--option)" fill-opacity="${Math.max(op, 0.7)}" text-anchor="end">${txt}</text>`;
  }));
  return s;
}
// 막대 앞: 내 선(청산 주황 실선 · 매수 초록/매도 빨강 점선) + 한 칩에 묶은 «닿음 %»(가까운 가격끼리 겹치지 않게)
function optTouchFront(m, lv, Y, W, y0, y1, px) {
  if (!m || !lv.length) return "";
  const col = { liq: "var(--warn)", buy: "var(--good)", sell: "var(--bad)" };
  let s = lv.map((r) => `<line x1="30" x2="${W - 4}" y1="${Y(r.p).toFixed(1)}" y2="${Y(r.p).toFixed(1)}" stroke="${col[r.kind]}" stroke-width="${r.kind === "liq" ? 1.6 : 1.4}"${r.kind === "liq" ? "" : ' stroke-dasharray="5 3"'}/>`).join("");
  const rows = [...lv].sort((a, b) => Math.abs(a.p - px) - Math.abs(b.p - px)).slice(0, 4).sort((a, b) => b.p - a.p);
  const ch = 6 + rows.length * 14, cw = 180, cx = 34, hiY = Y(rows[0].p), loY = Y(rows[rows.length - 1].p);
  const below = rows.every((r) => r.p < px);
  const cy = Math.max(y0, Math.min(y1 - ch, below ? hiY - ch - 6 : loY + 6));
  s += `<rect x="${cx}" y="${cy.toFixed(1)}" width="${cw}" height="${ch}" rx="7" fill="var(--chart-bg)" fill-opacity=".94" stroke="var(--line)"/>`
    + `<line x1="${cx + 10}" x2="${cx + 10}" y1="${(below ? cy + ch : cy).toFixed(1)}" y2="${(below ? hiY - 1 : loY + 1).toFixed(1)}" stroke="var(--line)"/>`;
  rows.forEach((r, i) => {
    const y = cy + 15 + i * 14;
    s += `<text x="${cx + 9}" y="${y.toFixed(1)}" font-size="10.5" font-weight="700" font-family="var(--font-mono)" fill="${col[r.kind]}">${r.t} ${optPx1(r.p)}</text>`
      + `<text x="${cx + cw - 9}" y="${y.toFixed(1)}" font-size="10.5" font-weight="800" font-family="var(--font-mono)" fill="${r.kind === "liq" ? "var(--warn)" : "var(--text)"}" text-anchor="end">닿음 ${optPct(m.touch(r.p))}</text>`;
  });
  return s;
}
function optLadderChips(legend = "") {
  const chip = (k, t) => `<button type="button" class="opt-chip-btn${optLadderScope === k ? " on" : ""}" data-scope="${k}" aria-pressed="${optLadderScope === k}">${t}</button>`;
  return `<div class="opt-chips">${chip("front", "가까운 만기")}${chip("week", "7일 안")}${chip("all", "전 만기")}${legend ? `<span class="opt-lane-legend opt-touch-legend">${legend}</span>` : ""}</div>`;
}
function optLadderSvg(o, W, Hfit = null) {
  const st = o.strikes || {}, rows = st[optLadderScope] || [];
  const rolled = optLadderScope === "front" && st.front_exp_ms && st.front_exp_ms <= Date.now();   // 만기 직후 ≤10분(수집 10분 주기)
  if (!rows.length || rolled) return optLadderChips() + `<div class="opt-note">${rolled ? "만기 교체 중 — 다음 수집(10분 안)부터 새 가까운 만기" : "행사가 데이터 없음(수집기 다음 주기에 채워진다)"}</div>`;
  const px = optPx(o), lo = px * 0.94, hi = px * 1.06, H = Hfit || 560,   // 2026-10-01 ±8% -> ±6%(사용자 «막대를 키워서» -- 행사가 간격 10/20 이 섞여 두께는 좁은 간격에 묶이므로 범위를 좁혀 1.33배)
  y0 = 34, y1 = H - 8;   // y0 22 -> 34: 맨 위 막대가 «← 풋 · 콜 →» 머리글과 겹쳤다   // 2026-09-30 380 -> 560(막대를 두껍게 -- 값 글자가 막대 안에 들어가게, 사용자 지시)
  const Y = (p) => y1 - ((p - lo) / (hi - lo)) * (y1 - y0);
  const vis = rows.filter((r) => r[0] >= lo && r[0] <= hi);   // 2026-10-01 범위 밖 행사가는 안 그린다(맨 위 막대가 머리글을 덮었다) · 막대 길이 기준도 범위 안에서
  const mid = Math.round((W + 46) / 2), half = mid - 46, gx = W;
  const mx = Math.max(1, ...vis.map((r) => Math.max(r[1], r[2])));
  const ks = vis.map((r) => r[0]), step = ks.length > 1 ? Math.min(...ks.slice(1).map((k, i) => k - ks[i])) : 25;
  const bh = Math.max(2, Math.min(22, ((y1 - y0) * step) / (hi - lo) - 1)),   // 2026-10-01 상한 15 -> 22(사용자 «막대를 키워서»)
    fsz = Math.max(8, Math.min(10.5, bh - 1));   // 값 글자 = 막대 두께 − 1
  const top = [...vis].sort((a, b) => b[1] + b[2] - a[1] - a[2]).slice(0, 3).map((r) => r[0]);
  let s = `<text x="${mid - 6}" y="12" font-size="10" font-weight="700" fill="var(--bad)" text-anchor="end">← 풋</text>`
    + `<text x="${mid + 6}" y="12" font-size="10" font-weight="700" fill="var(--good)">콜 →</text>`
    + `<line x1="${mid}" x2="${mid}" y1="${y0 - 4}" y2="${y1}" stroke="var(--line)"/>`;
  const cur = optData().cur, tm = optTouchModel(cur, px);   // 2026-10-08 닿음 등고선 -- 막대 뒤에 먼저 깐다
  s += optTouchBack(tm, Y, W, lo, hi);
  vis.forEach(([k, c, p]) => {
    const y = Y(k), pw = (half * p) / mx, cwid = (half * c) / mx;
    s += `<g><title>${optQ(k)} · 콜 ${optUsd(c)} · 풋 ${optUsd(p)}</title>`
      + `<rect x="${(mid - pw).toFixed(1)}" y="${(y - bh / 2).toFixed(1)}" width="${pw.toFixed(1)}" height="${bh.toFixed(1)}" fill="var(--bad)" fill-opacity=".75"/>`
      + `<rect x="${mid}" y="${(y - bh / 2).toFixed(1)}" width="${cwid.toFixed(1)}" height="${bh.toFixed(1)}" fill="var(--good)" fill-opacity=".75"/>` + `</g>`;
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
  s += optTouchFront(tm, tm ? optMyLevels(cur, lo, hi) : [], Y, W, y0, y1, px);
  return optLadderChips(tm ? `<b class="opt-c">선</b> ${optKst(tm.exp_ms)} 만기 전 닿을 확률 50·25·10% · 옵션가 역산 · 참고` : "")
    + `<svg class="opt-ladder" viewBox="0 0 ${W} ${H}" role="img" aria-label="행사가별 콜·풋 미결제">${s}</svg>`;
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
    if (i % (narrow ? 6 : 4) === 0 && X(i) + 25 <= W) s += `<text x="${(X(i) + 3).toFixed(1)}" y="${H - 2}" font-size="9.5" fill="var(--muted)">${optKst(b.h, false).slice(0, 2)}시</text>`;   // 2026-10-26 오른쪽 끝 눈금(«02시»)이 잘리면 뺀다
  });
  s += `<path d="M${cum.map((v, i) => `${X(i + 1).toFixed(1)} ${(base - (amp * v) / dm).toFixed(1)}`).join(" L")}" fill="none" stroke="var(--option)" stroke-width="2"/>`
    + `<text x="${x1}" y="${(base - (amp * cum[n - 1]) / dm - 6).toFixed(1)}" font-size="${narrow ? 10 : 11}" font-weight="700" fill="var(--option)" text-anchor="end">체결 순델타 ${cum[n - 1] >= 0 ? "+" : ""}${fmtNum(cum[n - 1], 0)} ${escapeHtml(optData().cur || "")}</text>`;
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="옵션 순매수 흐름: 시간별 콜·풋 순매수와 순델타 누적">${s}</svg>`;
}

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
};   // «?»(.opt-q) 는 문서 전체 처리기가 설명 팝업을 연다(helpOpen)
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
  const body = el("optBody");
  if (!body) return;
  const { cur, g, o, blocks } = optData();
  const row = el("optRow"), laneBody = el("optLaneBody"), blkBody = el("optBlkBody"), flowBody = el("optFlowBody");
  if (row) row.hidden = !o;
  optRowCols();
  if (!o) {
    body.innerHTML = `<div class="opt-note">${!cur ? "Deribit 옵션은 ETH·BTC·SOL·XRP 만 수집합니다"
      : latestGex && latestGex.error ? `옵션 수집 지연 (${escapeHtml(latestGex.error)})` : "옵션 불러오는 중…"}</div>`;
    return;
  }
  const f = optFront(o), hrs = f ? (f.exp_ms - Date.now()) / 3.6e6 : null;
  const s1d = optSigma(o, 86400), s5 = optSigma(o, 300);
  // 2026-09-29 사용자 지시: 예상 폭은 전부 DVOL 하나로(만기별 ATM IV 는 이력이 09-28 부터라 검정 불가였다).
  const sExp = f && hrs > 0 ? optSigma(o, hrs * 3600) : null;
  const vrp = optIv(o) != null && o.rv7 != null ? optIv(o) - o.rv7 : null;   // 2026-09-30 DVOL 없는 SOL·XRP 는 30일 ATM IV(예상 폭과 같은 대체) -- 전엔 늘 «-»
  const pm = (v) => (v == null ? "-" : `±${optQ(v)}$`);
  const kv = (k, v, cls = "") => `<div class="opt-kv"><span>${k}</span><b class="${cls}">${v}</b></div>`;
  const info = optInfoParts(o, g, kv);   // 2026-10-02 옵션 정보 → 예상 폭·심리에 나눠 넣는다
  // 칸 제목을 누르면(휴대폰 포함 -- 호버가 없다) 정의·읽는 법·우리 검정 결과가 설명 팝업으로 뜬다(2026-10-26, 옛 제목 아래 펼침)
  const sec = (key, title, inner) => `<div class="opt-sec"><h4><button type="button" class="opt-q" ${helpAttrs(key)}>${title}<span aria-hidden="true">?</span></button></h4>${inner}</div>`;
  const rg = cur === "ETH" ? optRealizedGamma(candleHistoryByAsset.eth) : null;   // 2026-10-02 딜러 감마 → 실현 감마 국면
  const covHead = el("optCovHead");
  if (covHead) covHead.innerHTML = optHead(o);
  // 2026-09-30 검증: 갱신 실패·서버 상태 지연이면 마지막 값을 그대로 두되 «지연»을 먼저 말한다
  body.innerHTML = (latestGex && latestGex.error ? `<div class="opt-note opt-warn">옵션 갱신 지연(${escapeHtml(String(latestGex.error))}) — 마지막으로 받은 값</div>` : "") + [   // 커버 줄은 제목 옆(#optCovHead) -- 아래에서 넣는다
    sec("move", "예상 폭", kv("24시간 1σ", s1d == null ? "-" : `${pm(s1d)} (${(optIv(o) / Math.sqrt(365)).toFixed(1)}%)`)
      + kv("이번 5분 1σ", pm(s5)) + (o.dvol == null && o.iv30 != null ? `<div class="opt-note">DVOL 지수가 없는 코인 — 30일 ATM IV ${o.iv30.toFixed(0)}% 로 계산</div>` : "") + kv(f ? `다음 만기까지(${hrs.toFixed(0)}h)` : "다음 만기까지", pm(sExp))
      + kv("IV(30일) − 실현(7일)", vrp == null ? "-" : `${vrp >= 0 ? "+" : ""}${vrp.toFixed(1)}pt`) + info.move),
    sec("rg", "실현 감마", !rg ? `<div class="opt-note">${cur === "ETH" ? "5분봉 12시간치 대기" : "ETH 만 검증 — 이 코인은 표시 안 함"}</div>`
      : kv("12시간 자기상관", `${rg.v >= 0 ? "+" : "−"}${Math.abs(rg.v).toFixed(3)} · 2021~24 중 ${rg.pct < 50 ? `하위 ${Math.max(1, Math.round(rg.pct))}` : `상위 ${Math.max(1, Math.round(100 - rg.pct))}`}%`)
      + kv("국면", rg.side === "rev" ? "되돌림 쪽 · 롱감마형" : rg.side === "trend" ? "추세 쪽 · 숏감마형" : "보통", rg.side === "rev" ? "opt-good" : rg.side === "trend" ? "opt-warn" : "")
      + kv("다음 4시간 변동성", rg.side === "rev" ? "예상보다 약간 작은 쪽 · 0.94배" : rg.side === "trend" ? "예상보다 약간 큰 쪽 · 1.02배" : "예상과 비슷 · 0.99배")),
    sec("ladder", "행사가 사다리", optLadderSvg(o, optColW(), fitOptLadderH)),
    // 2026-10-01 연구: 가까운 만기 최근접 RR/BF 는 잡음(1시간 SD 3.6pt) → 7일 고정만기 델타 보간(수집기 o.cm). 30일은 괄호.
    sec("mood", "심리", kv("25Δ 리스크 리버설 · 7일", optCm(o, "rr"))
      + kv("버터플라이 · 7일", optCm(o, "bf"))
      // 2026-10-01 기간 구조 = 고정만기 7·30·60일 ATM(가까운 일간 만기는 남은 시간 안에 미국장이 드느냐로 0.71~0.92배 출렁여 «역전»을 흉내 냈다)
      + kv("기간 구조 · ATM IV", ["7", "30", "60"].map((d) => ((o.cm || {})[d] || {}).atm).every((v) => v == null) ? "-"
           : ["7", "30", "60"].map((d) => { const v = ((o.cm || {})[d] || {}).atm; return `${d}일 ${v == null ? "-" : v.toFixed(0)}`; }).join(" → ")) + info.mood),
  ].join("");
  renderPaper();   // 2026-10-04 옛 «옵션 요약» 자리(#mcWall) = 모의 판(사용자 지시). 옵션 범위·max pain·RR 은 머리 칩·Option 카드에 그대로
  // 2026-09-29 풋프린트 카드 맨 아래 반반(휴대폰은 위아래) -- 두 칸 폭이 같아 한 번 잰다.
  if (laneBody && flowBody && laneBody.clientWidth > 0) {
    const lw = Math.round(laneBody.clientWidth), fw = Math.round(flowBody.clientWidth) || lw, fl = ((((latestGex || {}).block_trades || {}).flow_by_coin || {})[cur]) || [];
    // 두 그림 제목은 같은 모양(청록 ? 버튼 + 범례, 2026-09-29 사용자 «제목 포맷 통일») -- 누르면 설명 팝업.
    const head = (k, title, legend) => `<div class="opt-flow-head"><button type="button" class="opt-q" ${helpAttrs(k)}>${title}<span aria-hidden="true">?</span></button>`
      + `<span class="opt-lane-legend">${legend}</span></div>`;
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
    const cur = optData().cur;   // 2026-10-08 닿음 등고선용 스마일(ETH·BTC) -- 실패하면 등고선만 빠진다
    if (cur === "ETH" || cur === "BTC") {
      try {
        const sm = await fetch(`/api/opt-smile?asset=${cur.toLowerCase()}`, { cache: "no-cache" });
        if (sm.ok) latestSmile[cur] = await sm.json();
      } catch (e) { latestSmile[cur] = null; }
    }
    renderOptions();
  } catch (error) {
    // 2026-09-30 검증: 한 번 실패로 null 을 넣으면 카드가 다음 폴링(60초)까지 통째로 사라졌다 -- 직전 값을 두고 오류만 싣는다
    console.error("GEX fetch error:", error);
    if (latestGex) { latestGex = { ...latestGex, error: String(error.message || error) }; renderOptions(); }
  }
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
    const oiBarsWanted = CHART_PAN_BARS + 4;   // 2026-10-20 12시간 밖(1주일)은 /api/footprint-week 가 한 번 준다(fpWeek.oi)
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
}

async function refreshTrend() {
  if (activePageTab !== "snapshot" || document.hidden) return;
  const now = Date.now();
  if (now - trendLastFetchAt < TREND_POLL_MS) return;
  trendLastFetchAt = now;
  const asset = activeSnapshotAsset;
  let got;
  try {
    const res = await fetch(`${API_TREND_URL}?asset=${asset}`, { cache: "no-cache" });
    got = res.ok ? await res.json() : { ok: false, reason: `HTTP ${res.status}` };
  } catch (error) {
    got = { ok: false, reason: "fetch_failed" };
  }
  if (asset !== activeSnapshotAsset) return;   // 그 사이 코인이 바뀌었다
  latestTrend = got;
}

// 2026-10-16 사용자 «같이 해서 시안» → «배포»(시안 ①): 칸별 아이스버그(/api/iceberg = scripts/live_eth_iceberg_20261016, 30초 · ETH 만).
//   표시 전용 -- 신호·알림 아님. 연구 28일 23건 15분 +7.6bp[2.6,15.0] → 전진 150건이면 판정(research_eth_dom_levels_absorption_20261016 --judge).
let latestIceberg = null, icebergLastFetchAt = 0;
async function refreshIceberg() {
  renderIceChip();
  const asset = activeSnapshotAsset;   // 2026-10-20 XRP 판(ICEBERG_SYMBOL=XRPUSDT) -- 코인이 바뀌면 30초를 기다리지 않고 다시 받는다
  if (activePageTab !== "snapshot" || document.hidden || (asset !== "eth" && asset !== "xrp")) return;
  const now = Date.now();
  if (now - icebergLastFetchAt < 30000 && latestIceberg && latestIceberg._asset === asset) return;
  icebergLastFetchAt = now;
  try {
    const res = await fetch(`/api/iceberg?asset=${asset}`, { cache: "no-cache" });
    latestIceberg = res.ok ? await res.json() : { ok: false, reason: `HTTP ${res.status}` };
    latestIceberg._at = Date.now(); latestIceberg._asset = asset;   // 10-17 리뷰: 받아오기가 멈추면(SOL·운영 탭·숨은 창) 마지막 값이 «검증 중»으로 남던 것
  } catch (error) {
    latestIceberg = { ok: false, reason: "fetch_failed" };
  }
  renderIceChip();
}
// 2026-10-16 아티팩트 댓글 «아이스버그 검증은 알림창 안으로 · 풋프린트 머리에서는 제거»: 머리 칩(#iceChip) → 알림 센터 «전진 검증 중»(#ncIce)
function renderIceChip() {
  const c = el("ncIce"), g = latestIceberg;
  if (!c) return;
  if (!g) { c.innerHTML = '<p class="nc-empty">불러오는 중… (스냅샷 · ETH 에서 30초마다)</p>'; return; }
  const age = g.ok ? g.age_s + (Date.now() - (g._at || Date.now())) / 1000 : 0;
  const stale = !g.ok || age > 120, n = g.ok ? g.n : 0, target = g.ok ? g.target : 150, pct = Math.min(100, Math.round(n / target * 100));
  c.classList.toggle("stale", stale);
  const html = g.ok && target == null   // 2026-10-20 XRP = 사전등록 없음(표시 전용) -- 검증 막대 대신 24시간 건수
    ? `<div class="nc-ice-row"><strong>칸별 아이스버그 · XRP</strong><span class="nc-ice-st">${stale ? "멈춤" : "표시 전용"}</span><b>24h ${g.events.length}건</b></div>`
      + `<p class="nc-hint">ETH 와 같은 정의를 XRP 0.0001 칸(1틱)에 적용 · 사전등록·판정 없음 · 신호·알림 아님.</p>`
    : `<div class="nc-ice-row"><strong>칸별 아이스버그</strong><span class="nc-ice-st">${stale ? "멈춤" : "검증 중"}</span><b>${n}/${target}</b></div>`
    + `<span class="nc-ice-bar" aria-hidden="true"><i style="width:${pct}%"></i></span>`
    + `<p class="nc-hint">같은 $0.1 칸에서 30초 동안 보이던 잔량의 2배 넘게 체결됐는데 칸이 버티고 다시 채워진 사건. 연구 28일 23건 · 15분 +7.6bp[2.6, 15.0] — `
    + `2026-10-16부터 ${target}건 모이면 판정합니다. 신호·알림 아님 · 풋프린트 육각형 = 최근 24시간 사건(초록 = 숨은 매수 · 빨강 = 숨은 매도).</p>`
    + (stale ? `<p class="nc-hint nc-ice-bad">${escapeHtml(g.ok ? `탐지기 ${Math.round(age)}초째 응답 없음` : `불러오기 실패: ${g.reason}`)}</p>` : "");
  if (c.innerHTML !== html) c.innerHTML = html;
}
// 풋프린트 위 육각형(renderCandleSvg 끝에서) -- 창 가운데 시각 · 그 칸 가격. 이름표는 가장 최근 하나만.
function iceMarks(svg, candles, xAt, yAt, bw, yMin, yMax, NS) {
  const ev = latestIceberg?.ok && latestIceberg._asset === activeSnapshotAsset ? latestIceberg.events : [];
  const bar = candles[1].time - candles[0].time, t1 = candles[candles.length - 1].time + bar;
  const hits = ev.filter((e) => e.t_end - 15 >= candles[0].time && e.t_end - 15 < t1 && e.px >= yMin && e.px <= yMax);
  if (!hits.length) return;
  const g = document.createElementNS(NS, "g");
  g.setAttribute("class", "ice-marks");
  hits.forEach((e, j) => {
    const t = e.t_end - 15, i = candles.findLastIndex((c) => c.time <= t);
    const x = xAt(i) + Math.min(1, (t - candles[i].time) / bar) * bw, y = yAt(e.px), buy = e.side === 1;
    const m = document.createElementNS(NS, "g");
    m.setAttribute("class", `ice-mk ${buy ? "buy" : "sell"}`);
    m.setAttribute("transform", `translate(${x.toFixed(1)},${y.toFixed(1)})`);
    const hex = document.createElementNS(NS, "path");
    hex.setAttribute("d", "M0 -6.5 5.6 -3.25v6.5L0 6.5-5.6 3.25v-6.5z");
    m.appendChild(hex);
    const word = `숨은 ${buy ? "매수" : "매도"} ${(e.vol / Math.max(e.max, 1e-9)).toFixed(1)}×`;
    if (j === hits.length - 1) {
      const lb = document.createElementNS(NS, "text");
      lb.setAttribute("x", "10"); lb.setAttribute("y", "3.5");
      lb.textContent = `${word} · 검증 중`;
      m.appendChild(lb);
    }
    const tip = document.createElementNS(NS, "title");
    tip.textContent = `${fmtHourMinute(t * 1000)} ${word} — ${e.px.toFixed(1)} 칸 · 30초 체결 ${e.vol.toFixed(1)} ETH = 보이던 최대 잔량 ${e.max.toFixed(1)} 의 ${(e.vol / Math.max(e.max, 1e-9)).toFixed(1)}배 · 칸 버팀`
      + " · 전진 검증 중(신호 아님)";
    m.appendChild(tip);
    g.appendChild(m);
  });
  svg.appendChild(g);
}

// 2026-10-15 ⑤ 예측 판 «4h 방향»(ETH, /api/dir4h = scripts/live_eth_dir4h_20261015 · 서버 5분 캐시) -- 참고용
let latestDir4h = null, dir4hLastFetchAt = 0;
async function refreshDir4h() {
  if (activePageTab !== "snapshot" || document.hidden || activeSnapshotAsset !== "eth") return;
  const now = Date.now();
  if (now - dir4hLastFetchAt < 120000) return;
  dir4hLastFetchAt = now;
  try {
    const res = await fetch("/api/dir4h", { cache: "no-cache" });
    if (res.ok) latestDir4h = await res.json();
  } catch (error) { /* 다음 주기 */ }
}
// 2026-10-28 ⑤ 예측 줄 «15분·1시간 예측»(ETH, /api/dir-short = scripts/live_eth_dir_short_20261028 · 서버 5분 캐시) -- 참고용
let latestDirShort = null, dirShortLastFetchAt = 0;
async function refreshDirShort() {
  if (activePageTab !== "snapshot" || document.hidden || activeSnapshotAsset !== "eth") return;
  const now = Date.now();
  if (now - dirShortLastFetchAt < 60000) return;
  dirShortLastFetchAt = now;
  try {
    const res = await fetch("/api/dir-short", { cache: "no-cache" });
    if (res.ok) latestDirShort = await res.json();
  } catch (error) { /* 다음 주기 */ }
}
// 2026-10-16 시안 ②③ 09시 개장 되돌림(ETH, /api/open-rev = scripts/live_eth_direction_forward_recorder_20261010.decide · 서버 5분 캐시)
let latestOpenRev = null, openRevLastFetchAt = 0;
async function refreshOpenRev() {
  if (activePageTab !== "snapshot" || document.hidden || activeSnapshotAsset !== "eth") return;
  const now = Date.now();
  if (now - openRevLastFetchAt < 120000) return;
  openRevLastFetchAt = now;
  try {
    const res = await fetch("/api/open-rev", { cache: "no-cache" });
    if (res.ok) latestOpenRev = await res.json();
  } catch (error) { /* 다음 주기 */ }
}
// 2026-10-16 사용자 «①부터 재검증하고 시안»: 박스 유지 확률(ETH, /api/box-hold = dashboard/box_hold.py · 서버 1분 캐시) -- 방향 아님.
//   박스 = 지금가 ± 지난 W분 고저폭의 절반 · 값 = 다음 W분 동안 한 번도 안 벗어날 확률(2022~24 학습, 2025·2026 보정 오차 ≤6pp).
let latestBoxHold = null, boxHoldLastFetchAt = 0;
// 2026-10-20 XRP 전용 모델(30분·1시간 통과 · 4시간 불통과 = 서버 ok false) -- 박스 줄·차트 1h 박스선이 그 코인 값만 쓴다
const BOX_ASSETS = new Set(["eth", "xrp"]);
const boxNow = () => (latestBoxHold && latestBoxHold.ok && (latestBoxHold.asset || "eth") === activeSnapshotAsset ? latestBoxHold : null);
const BOX_ON_CHART = true;    // 2026-10-16 사용자 «B로 배포» → 10-26 미래 칸 상자(renderCandleSvg «미래 상자»)
const BOX_STALE_MS = 180e3;   // 10-16 리뷰: 기준가가 3분 넘게 묵으면 «지연»·차트 선 숨김(30분 박스는 3분 뒤 9%가 이미 깨져 있다)
async function refreshBoxHold() {
  if (activePageTab !== "snapshot" || document.hidden || !BOX_ASSETS.has(activeSnapshotAsset)) return;
  const now = Date.now();
  if (now - boxHoldLastFetchAt < 20000 && (latestBoxHold?.asset || "eth") === activeSnapshotAsset) return;   // 서버 15초 캐시와 맞춤 · 코인을 바꾸면 바로
  boxHoldLastFetchAt = now;
  try {
    const res = await fetch(`/api/box-hold?asset=${activeSnapshotAsset}`, { cache: "no-cache" });
    if (res.ok) latestBoxHold = await res.json();
  } catch (error) { /* 다음 주기 -- 묵으면 BOX_STALE_MS 가 «지연»으로 드러낸다 */ }
}
const BOX_NM = { "30m": "30분", "1h": "1시간", "4h": "4시간" };
// 상태: 묵음(stale) · 이미 벗어남(out, 실시간 가격이 띠 밖) · 깨질 쪽(brk, 학습 예측 하위 20%) · 머물 쪽(hold, 상위 20%) · 보통
function boxState(b, s) {
  const live = Number(latestLivePriceByAsset[activeSnapshotAsset] || 0);
  if (Date.now() - b.asof_ms > BOX_STALE_MS) return ["stale", "지연"];
  if (live > 0 && (live > s.up || live < s.dn)) return ["out", "벗어남"];
  const lo = s.q20 != null ? s.q20 : s.base * 0.75, hi = s.q80 != null ? s.q80 : s.base * 1.33;
  return s.p < lo ? ["brk", "깨질 쪽"] : s.p > hi ? ["hold", "머물 쪽"] : ["", "보통"];
}
// 2026-10-17 사용자 «? 설명 단추 되살려»(제목은 없앤 채): ①②④ 처럼 «?» 만 -- 키보드로도 연다(설명 팝업, helpOpen)
const mcTipBtn = (key, name) => `<button type="button" class="mc-q mc-q1" ${helpAttrs(key)} title="${name} — 설명" aria-label="${name} 설명"><span aria-hidden="true">?</span></button>`;
// 2026-10-15 ⑤ 예측 판 «4h 변동성»(ETH, /api/vol-forecast · 서버 5분 캐시) -- 표시 전용
let latestVol4h = null, vol4hLastFetchAt = 0;
const VOL4H_ASSETS = new Set(["eth", "xrp"]);   // 2026-10-20 XRP 4h 변동성(ETH+XRP 합동 학습 · 2025~ R² .680 > HAR .568)
async function refreshVol4h() {
  if (activePageTab !== "snapshot" || document.hidden || !VOL4H_ASSETS.has(activeSnapshotAsset)) return;
  const now = Date.now();
  // 코인을 바꾸면 바로 다시 · 2026-10-26 리뷰: 서버 swr 이 묵은 값(봉 10분↑)을 먼저 주면 60초 뒤 다시(부채꼴 15분 신선도에 5분 공백이 났다)
  const gap = latestVol4h && now - latestVol4h.asof_ms > 600e3 ? 60000 : 300000;
  if (now - vol4hLastFetchAt < gap && latestVol4h?.asset === activeSnapshotAsset) return;
  vol4hLastFetchAt = now;
  try {
    const res = await fetch(`/api/vol-forecast?asset=${activeSnapshotAsset}`, { cache: "no-cache" });
    if (res.ok) latestVol4h = await res.json();
  } catch (error) { /* 다음 주기 */ }
}

// 2026-10-16 사용자 «지금 추세만 이것만 봐서는 이해하기 어려워» → 시안 A: 점 다섯 개에 기간 이름(7·14·28·56·90)을 붙이고 위/아래 개수를 말로.
//   표 = 매일 UTC 00시(KST 09시) 일봉 종가가 N일 전 종가(ref_now)보다 높으면 ↑(dashboard/trend_rule.py).
function fcTrendCell(t, cell) {
  const up = t.signal > 0, n = t.votes.length, ups = t.ups;
  const px = (v) => Number(v).toLocaleString("en-US", { maximumFractionDigits: 0 });
  const tags = `<span class="fc-tl">${t.votes.map((x) => `<i class="${x.up ? "u" : ""}" title="${x.L}일 전 종가 ${px(x.ref_now)}보다 ${x.up ? "높게" : "낮게"} 마감">${x.L}</i>`).join("")}</span>`;
  return cell("지금 추세", `${up ? "up" : "dn"} now`, `${up ? "▲ 상승" : "▼ 하락"} <small class="fc-cnt">${ups}:${n - ups}</small>`, "",
    `지금 추세 — 예측 아님. 매일 09시(KST) 일봉 종가를 7·14·28·56·90일 전 종가와 비교한 5표 중 다수 · ${t.age_days}일째 같은 방향`,
    tags + `<span class="sub">일 전 종가보다 위 ${ups} · 아래 ${n - ups}</span><span class="sub">${t.age_days}일째 · 매일 09시 마감</span>`).replace('<span class="sub"></span>', "");
}

// ⑤ 예측 줄(시장 맥락 판 아래, 사용자 시안 B): 지금 추세 → 15분·1시간 예측(10-28) → 4h 예측(+변동성) → 7일(4~13주는 10-28 Macro Trend 카드로). sec = renderMarketCtx 의 판 틀(? 설명).
function fcRowHtml(sec) {
  const t = latestTrend && latestTrend.ok ? latestTrend : null, ds = latestDirSig && latestDirSig.ok ? latestDirSig : {};
  const eth = activeSnapshotAsset === "eth", v = latestVol4h && latestVol4h.ok && (latestVol4h.asset || "eth") === activeSnapshotAsset
    && ["sig4h_pct", "sig4h_usd", "lvl", "x_now", "p_expand"].every((k) => Number.isFinite(latestVol4h[k])) ? latestVol4h : null;   // 10-21: 키가 빠지면 ⑤ 줄 전체가 죽었다
  const cls = (s) => (s > 0 ? "up" : s < 0 ? "dn" : "flat"), word = (s) => (s > 0 ? "▲ 상승" : s < 0 ? "▼ 하락" : "보류");
  const zf = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(2)}`;
  const cell = (hz, c, main, sub, tip, extra = "") => `<div class="fc-c ${c}" title="${escapeHtml(tip)}"><span class="hz">${hz}</span><b>${main}</b><span class="sub">${sub}</span>${extra}</div>`;
  const sig = (g, hz, nm) => (!g || !g.ok ? cell(hz, "flat", "-", "데이터 대기", nm)
    : cell(hz, cls(g.state), word(g.state), `${nm.split(" ")[0]} ${zf(g.v)}${g.state ? ` · ${g.run}일째` : ""}`, `${nm} · ${g.date} UTC 종가 · |값| ≥ 1 일 때만 방향`));
  const vTip = v ? `4시간 1σ ±${v.sig4h_pct.toFixed(2)}% (±$${v.sig4h_usd < 10 ? v.sig4h_usd.toFixed(4) : v.sig4h_usd.toFixed(0)}) · 최근 ${v.ref_days}일 평소의 ${v.lvl.toFixed(2)}배 · 직전 4시간의 ${v.x_now.toFixed(2)}배 · 더 커질 확률 ${Math.round(v.p_expand * 100)}%` : "";
  const volLine = v ? `<span class="sub fc-vol" title="${escapeHtml(vTip)}">σ ±${v.sig4h_pct.toFixed(2)}% · ×${v.lvl.toFixed(2)}</span>` : `<span class="sub">변동 ${VOL4H_ASSETS.has(activeSnapshotAsset) ? "불러오는 중" : "ETH·XRP 전용"}</span>`;
  const g4 = eth && latestDir4h && latestDir4h.ok ? latestDir4h : null, old4 = g4 && Date.now() - g4.decided_at_ms > 15 * 60e3;
  const kst = (ms) => new Date(ms + 9 * 3600e3).toISOString().slice(11, 16);
  const d4Tip = g4 ? `4h 방향 예측 — 매달 재학습 HGB(ETH 65피쳐, 최근 365일) · ${g4.model_month} 모델(학습 ${g4.train_window[0].slice(0, 10)}~${g4.train_window[1].slice(0, 10)})\n`
      + `예측 ${g4.pred_bp >= 0 ? "+" : ""}${g4.pred_bp.toFixed(1)}bp · 상승 문턱 ≥ +${g4.hi.toFixed(1)} · 하락 문턱 ≤ ${g4.lo.toFixed(1)} (과거 표본 밖 예측 상·하위 20%) · ${kst(g4.decided_at_ms)} KST 마감 5분봉\n`
      + `2025~26 IC .06 · 2023~24 검증 불통과 · 참고용(전진 판정 전)${g4.nan_feats.length ? `\n🔴빈 피쳐 ${g4.nan_feats.length}: ${g4.nan_feats.join(", ")}` : ""}\n${vTip}`
    : `4h 방향 예측(ETH 전용) — ${eth ? (latestDir4h && latestDir4h.reason ? `불러오기 실패: ${latestDir4h.reason}` : "불러오는 중") : "ETH 에서만 계산"}\n${vTip}`;
  // 2026-10-16 시안 ④: 다음 문턱까지 거리 + 문턱 눈금(왼쪽 색 = 숏 문턱 밖 · 오른쪽 색 = 롱 문턱 밖 · ● 지금 · ○ 1시간 전)
  const sgn = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(0)}`;
  // 칸 이름이 «4h 예측»이라 «예측» 글자는 뺀다(1700 폭에서 넘쳤다) · 신호 중이면 거리 대신 «신호»
  const d4Next = !g4 ? "" : g4.pred_bp >= g4.hi || g4.pred_bp <= g4.lo ? " · 신호"
    : g4.pred_bp >= 0 ? ` · 롱까지 ${(g4.hi - g4.pred_bp).toFixed(0)}` : ` · 숏까지 ${(g4.pred_bp - g4.lo).toFixed(0)}`;
  const d4Sub = g4 ? (old4 ? `${sgn(g4.pred_bp)}bp · 지연` : `${sgn(g4.pred_bp)}bp${d4Next}`) : (eth ? "불러오는 중" : "ETH 전용");
  const d4Gauge = (() => {
    if (!g4) return "";
    const R = Math.max(Math.abs(g4.lo), Math.abs(g4.hi)) * 1.6, X = (bp) => Math.max(0, Math.min(100, 50 + bp / R * 50)).toFixed(1);
    const ago = (g4.hist || []).filter((h) => h[0] <= g4.decided_at_ms - 3600e3 + 150e3).pop();
    return `<span class="d4g" aria-hidden="true"><i class="z dn" style="width:${X(g4.lo)}%"></i><i class="z up" style="left:${X(g4.hi)}%"></i>`
      + (ago ? `<u class="ago" style="left:${X(ago[1])}%"></u>` : "") + `<u style="left:${X(g4.pred_bp)}%"></u></span>`;
  })();
  // 2026-10-16 시안 ②③: 종합 = 사전등록 portfolio3 세 부품의 «지금 자리» -- 4h HGB 신호 · 일봉 추세(|5표 평균| ≥ .5, 즉 4:1 이상) ·
  //   09시(UTC 00시) 개장 되돌림(첫 30분 |수익| > 0.5σ 면 반대, 00:30~04:30 UTC 만). 🔴공식 가중(직전 90일 일 변동 역수)은 실시간에 없어 같은 가중.
  const orv = eth && latestOpenRev && latestOpenRev.ok ? latestOpenRev : null, nowMs = Date.now();
  const orvOn = orv && nowMs >= orv.entry_ms && nowMs < orv.exit_due_ms;
  const parts = [
    ["4h", g4 && !old4 ? g4.signal : null, g4 ? `4h 모델 ${word(g4.signal)} (예측 ${sgn(g4.pred_bp)}bp, 문턱 ${g4.lo.toFixed(0)} / +${g4.hi.toFixed(0)})` : "4h 모델 -"],
    ["추세", t ? (Math.abs(t.signal) >= 0.5 ? Math.sign(t.signal) : 0) : null,
      t ? `일봉 추세 5표 ${t.ups}:${t.votes.length - t.ups} — 4:1 이상일 때만 방향(사전등록 규칙) · 하루 보유` : "일봉 추세 -"],
    ["09시", orv ? (orvOn ? orv.open_pos : 0) : null,
      !orv ? "09시 개장 되돌림 -" : nowMs < orv.entry_ms ? `09시 개장 되돌림 — 오늘 ${kst(orv.entry_ms)} KST 에 판정`
        : `09시 개장 되돌림 — ${orv.date} 첫 30분 ${orv.f30_bp >= 0 ? "+" : ""}${orv.f30_bp.toFixed(1)}bp vs 문턱 ±${(0.5 * orv.sigma30_bp).toFixed(1)}bp → ${orv.open_pos ? (orv.open_pos > 0 ? "롱" : "숏") : "해당 없음"}${orv.open_pos ? (orvOn ? ` · ${kst(orv.exit_due_ms)} KST 까지` : " · 끝남") : ""}`],
  ];
  const net = parts.reduce((a, p) => a + (p[1] || 0), 0), agree = parts.filter((p) => p[1] && Math.sign(p[1]) === Math.sign(net)).length;
  const pw = (s) => (s == null ? "-" : s > 0 ? "▲" : s < 0 ? "▼" : "·");
  const sumTip = `종합 — 사전등록(10-10) 세 부품의 지금 자리, 같은 가중으로 더함(공식 판정은 직전 90일 변동 역수 가중)\n${parts.map((p) => p[2]).join("\n")}\n`
    + "2023~26 결합 샤프 1.56(3자산 1.75) · 과거에서 고른 조합 — 전진 판정 2027-03-14 이후 · 참고용";
  const fcSum = `<span class="fc-sum ${cls(net)}" title="${escapeHtml(sumTip)}">종합 <b>${net > 0 ? "▲ 롱 쪽" : net < 0 ? "▼ 숏 쪽" : "중립"}</b>`
    + `${net ? `<small>${agree}/3</small>` : ""}${parts.map((p) => `<i class="${p[1] ? cls(p[1]) : ""}">${p[0]} ${pw(p[1])}</i>`).join("")}</span>`;
  // 2026-10-28 시안 A(사용자 선택): «15분 예측» + «1시간 예측».
  //   분위 = 표본 밖 예측(2026-06~10, 41,760봉) 중 자리 · 하위/상위 20% 밖만 방향(4h 칸과 같은 규칙) · ○ = 한 지평 전
  const gs = eth && latestDirShort && latestDirShort.ok ? latestDirShort : null, oldS = gs && Date.now() - gs.decided_at_ms > 15 * 60e3;
  const sCell = (nm, ko, back, col) => {
    const p = gs && gs.pred && !oldS ? gs.pred[nm] : null, hist = gs ? gs.hist || [] : [];
    const ago = hist[hist.length - 1 - back], acc = nm === "15m" ? ["56", "51"] : ["55", "52"];
    const fail = latestDirShort && !latestDirShort.ok ? latestDirShort.reason || "?" : "";
    const why = !gs ? (!eth ? "ETH 전용" : fail ? "준비 안 됨" : "불러오는 중") : oldS ? "지연" : gs.book === "warm" ? "호가 깊이 예열 중" : "호가 깊이 없음";
    const sub = !p ? why : (p.pct < 50 ? `하위 ${Math.round(p.pct)}%` : `상위 ${Math.round(100 - p.pct)}%`)
      + (p.signal ? "" : p.pct >= 50 ? ` · 롱까지 ${Math.round(80 - p.pct)}` : ` · 숏까지 ${Math.round(p.pct - 20)}`);
    const sg = hist.map((h) => (h[col] >= 80 ? 1 : h[col] <= 20 ? -1 : 0));
    const tip = p ? `${ko} 방향 예측 — 매달 재학습 LightGBM(65피쳐 + 호가 깊이 10열 · 표적 4종 × 시드 5) · ${gs.model_month} 모델\n`
        + `점수 ${p.score >= 0 ? "+" : ""}${p.score.toFixed(2)} · 표본 밖 분위 ${p.pct.toFixed(1)} (하위 20% ≤ ${gs.thr[nm][0].toFixed(2)} · 상위 20% ≥ +${gs.thr[nm][1].toFixed(2)}) · ${kst(gs.decided_at_ms)} KST 마감 5분봉\n`
        + `문턱 밖 적중 ${acc[0]}% · 안쪽 ${acc[1]}% · 전체 53% (2026-06~10 표본 밖, 문턱도 같은 표본의 분위) — 약한 신호 · 참고용(전진 판정 전)\n`
        + `● 지금 · ○ ${ko} 전 · 최근 1시간 ▲${sg.filter((x) => x > 0).length} ▼${sg.filter((x) => x < 0).length} 보류 ${sg.filter((x) => !x).length}`
        + (gs.nan_feats.length ? `\n🔴빈 피쳐 ${gs.nan_feats.length}: ${gs.nan_feats.join(", ")}` : "")
      : `${ko} 방향 예측(ETH 전용) — ${fail ? `불러오기 실패: ${fail}` : why}`;
    const X = (v) => Math.max(0, Math.min(100, v)).toFixed(1);
    const gauge = p ? `<span class="d4g" aria-hidden="true"><i class="z dn" style="width:20%"></i><i class="z up" style="left:80%"></i>`
      + (ago ? `<u class="ago" style="left:${X(ago[col])}%"></u>` : "") + `<u style="left:${X(p.pct)}%"></u></span>` : "";
    return cell(`${ko} 예측`, p ? cls(p.signal) : "flat", p ? word(p.signal) : "-", sub, tip, gauge);
  };
  // 2026-10-17 펜 메모 «타이틀 제거 · 예측은 지금 유지»: 제목 단추(⑤ 예측 · 지금에서 멀리 ?)를 빼고 종합 칩만 머리에 -- 설명은 머리 툴팁
  return `<div class="mc-sec mc-fc"><h4>${eth ? fcSum : ""}${mcTipBtn("q_fc", "⑤ 예측 · 지금에서 멀리")}</h4><div class="fc-row">`
    + (t ? fcTrendCell(t, cell)
         : cell("지금 추세", "flat now", "-", "불러오는 중", "지금 추세"))
    + sCell("15m", "15분", 3, 1) + sCell("1h", "1시간", 12, 2)
    + cell("4h 예측", g4 && !old4 ? cls(g4.signal) : "flat", g4 ? word(g4.signal) : "-", d4Sub, d4Tip, d4Gauge + volLine)
    + sig(ds.cb, "7일 예측", "z 코인베이스 프리미엄 7일 변화(BTC 기준)")
    + `</div></div>`;   // 2026-10-28 사용자 «4~13주 예측은 Macro Trend 카드로 대체» -- z91 칸 뺌(서버 계산·전진 판정 2027-06 은 그대로)
}

async function refreshDirSig() {
  if (activePageTab !== "snapshot" || document.hidden) return;
  const now = Date.now();
  if (now - dirSigLastFetchAt < DIR_SIG_POLL_MS) return;
  dirSigLastFetchAt = now;
  try {
    const res = await fetch("/api/dir-signals", { cache: "no-cache" });
    if (res.ok) latestDirSig = await res.json();
  } catch (error) { /* 다음 주기에 다시 -- 받아 둔 값은 그대로 */ }
}

// ── 시장 맥락 (2026-09-29, ETH) ───────────────────────────────────────────
// 레포트(ETH 오더플로 105항목) 대조 뒤 사용자 «불합격된 것들도 쓸모 있으면 넣어줘 -- 우리 검증 방식이 틀렸을 수도 있다».
// 서버 /api/market-context(dashboard/market_ctx.py)를 그대로 그린다. 값은 전부 **서술** -- 방향 판정은 30분 카드(융합)가 한다.
// 칸 제목의 «?» = 정의 · 읽는 법 · 우리 검정의 말(DESIGN.md: 매매 화면 본문에 연구 원문 숫자를 올리지 않는다).
const API_MARKET_CTX_URL = "/api/market-context";
const MARKET_CTX_POLL_MS = 1000;   // 2026-10-03 5초 -> 1초(사용자 «1초 원천은 화면도 1초») -- 호가 벽·스프레드·계기·BTC 동행은 서버 1초 루프 · 응답 중앙 14ms·3KB(서버 실측)
let latestMarketCtx = null, marketCtxLastFetchAt = 0;
const API_PAPER_URL = "/api/paper-arms";
// 2026-10-04 코인별 검정 통과 목록 -- 벽 = 깊은 벽 다코인 사전등록 통과(deep_wall_ho3_multicoin_prereg_20261004: SOL t5.6 · XRP t3.7) ·
//   맞대결 = 고래↔리테일 60분(ETH 4.7년 · SOL·XRP 는 docs/experiments/whale_duel_solxrp_20261004.md 판정)
const PAPER_SIG_OK = { wall: new Set(["eth", "sol", "xrp"]), duel: new Set(["eth"]) };
const PAPER_POLL_MS = 10000;   // 2026-10-04 모의 매매 엔진은 1분마다 쓴다 -- 10초면 충분
let latestPaper = null, paperLastFetchAt = 0;
let latestMacroEvents = null;        // renderMacroCalendar 가 채운다 -- 카드의 «다음 주요 일정». null = 아직 못 받음(«없음»과 다르다)
const mcLine = { poc: true, vwap: true, veto: true, bb: false, wvwap: false, vsess: false, ema5: false, ema20: false, vwma: false };   // 2026-10-26 펜 메모: 이동평균 셋(기본 끔)   // wvwap = 주간·앵커 VWAP(2026-09-30, 기본 끔)   // 차트 선(카드 «가격 위치»의 스위치). 볼린저는 우리 검정에서 방향 엣지가 없어 기본 끔
try { Object.assign(mcLine, JSON.parse(localStorage.getItem("mcLine") || "{}")); } catch (err) { /* 기억은 편의 */ }
// 2026-09-30 사용자 «세션은 미국·유럽·아시아 시장 VWAP 으로» -- vsess 켜면 지금 시장 세션 개장(현지 시각, 서머타임 반영)부터 센 VWAP.
//   서버가 봉마다 두 벌을 싣는다: vwap/vsd = UTC 00시부터 · svwap/svsd/sstart/sname = 세션 개장부터. seg 가 바뀌는 곳에서 선을 끊는다.
function mcVwapOf(c) {
  if (mcLine.vsess) return c && c.svwap ? { vwap: c.svwap, vsd: c.svsd || 0, seg: c.sstart, name: c.sname || "세션" } : null;
  return c && c.vwap ? { vwap: c.vwap, vsd: c.vsd || 0, seg: Math.floor(c.time / 86400), name: "하루" } : null;
}
const MC_TIPS = {
  lev: "펀딩 = 8시간마다 롱과 숏이 주고받는 이자(양수면 롱이 낸다). 바이낸스는 평온할 때 0.01%에 붙어 거의 안 움직여서 «비싸게 들고 있나»는 베이시스(선물 마크 − 현물 인덱스)의 7일 분위로 본다. OI = 열린 계약 수(바이낸스+OKX+HL 합 · Bybit 는 7일 쌓인 뒤 합류 — 줄에 마우스를 올리면 거래소별), z = 30분 변화가(줄 값 · 위 «레버리지» 판정 칩은 검정된 1시간 변화 기준) 지난 7일 중 같은 요일유형(평일끼리·주말끼리) 시간들에 견줘 얼마나 튀었나 — 주말이 평일 잣대에 눌리지 않는다.\n우리 검정: 극단 펀딩 뒤 가격 방향은 없었다(표본 충분). 그래도 넣은 이유 — 청산 연쇄는 쏠린 쪽에서 나고, 상태 줄은 그 쏠림을 말한다. 방향 신호로 쓰지 않는다.",
  ls: "바이낸스 5분 통계: 계정 롱숏비(롱 계정 수 ÷ 숏 계정 수) · 탑트레이더 포지션 비율 · 테이커 매수 ÷ 매도. 괄호 = 하루 전.\n우리 검정: 30분 방향 모델에 넣으면 오히려 조금 나빠졌고 학술 근거도 없다. 군중이 어느 쪽에 몰렸나를 눈으로 보는 용도.",
  quad: "1시간 가격 이동 × OI 변화. 이동이 하루 1시간 이동의 상위 25% 안일 때만 판정한다.\n우리 검정(4.7년): 하락 + OI↑(새 숏 유입) 뒤 1시간은 더 내렸고, 하락 + OI↓(롱 정리) 뒤는 되돌렸다 — 숏을 «언제 거두나»의 근거. 상승 쪽 두 칸은 아직 안 쟀다. 차트 아래 사분면(5분 델타 × OI)은 방향 정보가 0으로 나온 서술용이다.",
  flow: "크기별 순매수 = 테이커 주문 크기로 가른 30분 순매수(바이낸스+OKX 합 · Bybit 는 기준선이 쌓이면 합류 — 줄 호버에 거래소)를, 지난 14일 중 같은 요일유형(평일끼리·주말끼리)·같은 시간대(UTC 시) 30분 순매수의 보통 크기(중앙값 편차)로 나눈 값(σ) — 한산한 새벽·미국장·주말이 같은 잣대로 보인다(기준이 아직 없으면 서버 기동 후 12~24시간 분포). 고래 ≥ $10만 · 리테일 < $1만. 델타(30분 CVD)·OI(30분 변화, ① 판)·테이커 매수÷매도·가격×OI(1시간) 도 같은 합산. 30분 체결 = 거래소별 순매수(ETH). 합산이 안 될 때(거래소 봉이 비면)는 바이낸스만 — 줄 호버에 «바이낸스만».\n우리 검정: 혼자서는 셋 다 되돌림 쪽(리테일이 가장 심하다). 고래와 리테일이 갈릴 때 60분 고래 쪽이 약하게 맞았다(메이커 전제). CVD는 «설명»이지 선행지표가 아니다.",
  cross: "BTC 30분 이동과 ETH의 동행 여부 · 거래소 가격차(바이낸스 기준 — OKX 는 마크 대 마크, HL 은 미드 대 미드. 마크는 평활값이라 미드와 섞어 재면 괴리가 부풀어 보인다) · HL 프리미엄(마크 − 오라클).\n우리 검정: BTC→ETH 1분 선행 0, 바이낸스가 HL을 1초 안쪽으로 앞선다 — 30분~2시간 방향에는 못 쓴다. 괴리가 커지는 순간(거래소 장애·한쪽 청산 쏠림)을 알아채는 용도.",
  book: "미드에서 ±25/50/100bp까지 걸린 호가 합 = 그만큼 밀려면 먹어야 할 물량(한 초 스냅샷 · 취소·재보충은 모른다). 얇은 쪽 = ±25bp 호가가 지난 6시간 중 몇 분위인가.\n우리 검정: 미검정. 거래량 계열은 실현변동성에 대부분 먹혔으니 «크기·속도» 참고로만. ETH 스프레드는 거의 늘 1틱이라 벌어지면 그 자체가 이상 신호다.",
  liq: "청산 급증 = 직전 60초 그 쪽 강제청산 합이 학습 7일 상위 0.5%(롱 $34만 · 숏 $55만) 초과(ETH 만 — SOL·XRP 는 검정 불통과). HL 고래 청산가 = 추적 중인 HL 상위 300주소의 실제 청산가(추정 아님)를 가격의 ~0.2% 칸(ETH $5)으로 묶은 금액 — 차트에 점선으로도 그린다. 12시간 실측 = 바이낸스 강제청산이 실제로 체결된 가격(차트 체결 기둥 안쪽 눈금).\n우리 검정: 청산 급증 뒤 역매매·추종 둘 다 엣지 0 — «청산 동반 급등에 역매매 금지» 필터만 유효. 추정 청산맵은 «위치»는 맞고 방향은 없다. HL 실측 청산가는 미검정.",
  when: "다음 펀딩 정산 · 가까운 옵션 만기와 max pain · 미국장 · 다음 주요 지표.\n우리 검정: 펀딩 정산 전후 드리프트는 반기마다 부호가 뒤집혀 기각. max pain 쪽 1시간 규칙(만기 1시간 전 → 08:00 UTC)만 2026년 첫 검정 통과 — 후보라 표본외 장부로 계속 잰다.",
  px: "VWAP = 거래량 가중 평균가(σ = 거래량 가중 표준편차). 기본은 하루(UTC 00시부터) · «세션 시작부터»를 켜면 지금 시장 세션 개장부터 센다(아시아 도쿄 09시 · 유럽 런던 08시 · 미국 뉴욕 09:30, 현지 시각 · 서머타임 반영 · 미국장 뒤는 다음 00시까지 미국 세션). 개장 직후 30~60분은 봉이 적어 가격에 붙어 다닌다. 볼린저 %B = (종가 − 하단) ÷ (상단 − 하단), 20봉·2σ. RSI 14 = 5분봉. 스위치로 차트에 선을 켠다.\n우리 검정: 볼린저·VWAP 셋업은 방향 엣지가 없었고 RSI·%B는 짧은 되돌림을 약하게 말한다(비용을 못 넘음). 위치 참고용.",
};
// 2026-09-30 시안 B 판 설명 = 옛 칸 설명을 판 단위로 묶은 것(문장은 그대로).
MC_TIPS.q_lev = MC_TIPS.lev + "\n\n" + MC_TIPS.ls;
MC_TIPS.q_flow = MC_TIPS.flow + "\n\n" + MC_TIPS.quad;
MC_TIPS.q_map = MC_TIPS.liq + "\n\n" + MC_TIPS.px;
MC_TIPS.q_wall = MC_TIPS.book;   // 2026-10-06 교차(BTC 30분·가격차)·위치(VWAP·%B·RSI) 줄 제거 -- 설명도 벽만
MC_TIPS.q_fc = "지평 = 지금에서 멀리. «지금 추세»만 예측이 아니다(지난 7·14·28·56·90일 종가 대비 5표, 미래 적중 48~58%).\n"
  + "15분·1시간 예측 = 매달 재학습 LightGBM(ETH 65피쳐 + 호가 깊이 10열 · 표적 4종 × 시드 5) · 표본 밖 예측(2026-06~10) 하위·상위 20% 밖만 방향 · "
  + "그 밖 적중 15분 56% · 1시간 55%(안쪽 51~52%) · 호가 깊이 예열 중(워커 재시작 뒤 12시간)이면 비움 · 참고용(전진 판정 전).\n"
  + "4h 예측 = 매달 재학습 HGB(ETH 65피쳐 · 최근 365일, 과거 표본 밖 예측 상·하위 20% 만) · 2025~26 IC .06 · 2023~24 검증 불통과.\n"
  + "7일 예측 = 코인베이스 프리미엄 7일 변화 z(BTC 기준, |z|≥1 만) · 2024~26 보류 BTC 59% · ETH 65%.\n"
  + "4~13주 = Macro Trend 카드(달러지수·순유동성 등 지표별 13주 변화 · ETF 이후 기준).\n"
  + "변동성 = 4시간 실현변동성 QX4(ETH·XRP·BTC·SOL 4코인 HGB·MLP 결합 · 2026-10-26 배포 · 굴림 3창 옛 GBM 이김) · 방향 정보 없음 · 표시 전용(주문 크기에 안 씀). "
  + "ETH 차트 미래 칸 옵션 색 부채꼴 = QX4I(QX4 + 직전 정시 옵션 ATM IV 보정 · 조건부 채택, 전진 판정 2027-01-22) ±1σ — 이 줄 σ 와 조금 다를 수 있다.\n"
  + "셋 다 참고용 — 전진 판정 전."
  + "주황 «깨질 쪽» = 학습 기간(2022~24) 예측의 하위 20% 밑 · «머물 쪽» = 상위 20% 위 · 그 사이 «보통» · «벗어남» = 지금가가 이미 띠 밖 · «지연» = 기준가가 3분 넘게 묵음(차트 선도 숨김). 방향 정보는 없다 — 어느 쪽으로 깨질지는 모른다.\n"
  + "우리 검정(ETH 4.7년, 2022~24 학습 · 2025·2026 따로 채점): AUC .75~.78 · 보정 오차 10분위 최대 3~6.4pp · 하위 10% 유지 3~5% ↔ 상위 10% 61~71%. "
  + "대부분은 «변동성 대비 띠 거리»(기하)이고, 최근 1시간/24시간 변동성 비 · 시각 · 주말이 +.02~.04 를 더한다. 24시간 박스는 보정이 나빠 뺐다.";
// 2026-10-04 맞대결·벽 신호·모의 판(사용자 승인 시안) -- 원천 = 모의 매매 엔진 상태(/api/paper-arms). 연구와 같은 계산을 엔진이 한 번만 한다.
MC_TIPS.q_flow += "\n\n맞대결 = 정시마다 고래(체결 한 줄 $10만+)·리테일($1만 미만)의 60분 순매수를 자기 지난 30일과 견준 z 가 반대이고 둘 다 0.5 이상이면 고래 쪽, 다음 정시까지. 위 줄들(14일 같은 요일유형·시간대 잣대·거래소 합산)과 잣대가 달라 가끔 엇갈린다 — 맞대결은 바이낸스만 — 합산판(+OKX·+Bybit)은 10-06 검정에서 바이낸스판보다 약해 불통과(2025~ 건당 +7.8 vs +7.1 · +3.8bp, docs/experiments/whale_duel_multivenue_20261006.md). 4.7년 검정에서 약하게 맞았다(메이커 체결 전제).";
MC_TIPS.q_wall += "\n\n벽 신호 = 5분봉 마감 때 ±50bp 매수·매도 잔량 불균형이 지난 24시간 상위 20% 문턱(주황 점선)을 넘는 쪽, 다음 5분. 직전 5분이 이미 그 방향으로 움직였으면 «추격 주의». 본질은 «방금 움직인 반대쪽»이라 한 방향으로 계속 가는 날엔 틀린다.";
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
    // 2026-10-21 사용자 «신규 롱/숏커버는 초록불, 신규 숏/롱정리는 빨간불 -- 표시등으로»: 칸마다 등(●)+이름 · 지금 칸만 불이 켜짐(등·옅은 바탕·글자 밝게) · 오른쪽 «보류» 값 없음(다 꺼짐 = 보류)
    const cells = [["up_up", "신규 롱"], ["up_dn", "숏커버"], ["dn_up", "신규 숏"], ["dn_dn", "롱정리"]], cw = (w - 6) / 4, txt = cw >= 44;
    return `<svg class="mc-svg" width="${w}" height="16" viewBox="0 0 ${w} 16" aria-hidden="true">` + cells.map(([k, t], i) => {
      const on = k === key, col = k.startsWith("up") ? "var(--good)" : "var(--bad)", x = i * (cw + 2), dx = txt ? x + 9 : x + cw / 2;
      return `<rect x="${x.toFixed(1)}" y="1" width="${cw.toFixed(1)}" height="14" rx="3" fill="${on ? col : "rgb(var(--lift) / .06)"}" fill-opacity="${on ? 0.16 : 1}"${on ? ` stroke="${col}" stroke-opacity=".55"` : ""}/>`
        + `<circle cx="${dx.toFixed(1)}" cy="8" r="3.5" fill="${col}" fill-opacity="${on ? 1 : 0.22}"/>`
        + (on ? `<circle cx="${dx.toFixed(1)}" cy="8" r="6" fill="${col}" fill-opacity=".25"/>` : "")
        + (txt ? `<text x="${(x + 17).toFixed(1)}" y="11.5" font-size="9.5" font-weight="${on ? 700 : 500}" fill="${on ? "var(--text)" : "var(--muted)"}" fill-opacity="${on ? 1 : 0.7}">${t}</text>` : "");
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
  wall(imb, thr, w = mcGfx.gw || 130, span = 0.4) {   // 2026-10-04 깊은 벽 ±50bp 불균형 · 가운데 0 · 주황 점선 = 적응 문턱(±) · 문턱 밖이면 방향색
    const c = w / 2, s = (w / 2 - 4) / span, v = imb == null ? 0 : Math.max(-span, Math.min(span, imb)), x0 = v >= 0 ? c : c + v * s;
    const fill = imb == null || Math.abs(imb) <= thr ? "var(--muted)" : imb > 0 ? "var(--good)" : "var(--bad)";
    return `<svg class="mc-svg" width="${w}" height="16" viewBox="0 0 ${w} 16" aria-hidden="true"><rect x="4" y="7" width="${w - 8}" height="2" fill="rgb(var(--lift) / .1)"/>`
      + [-1, 1].map((q) => `<line x1="${(c + q * thr * s).toFixed(1)}" x2="${(c + q * thr * s).toFixed(1)}" y1="2" y2="14" stroke="var(--warn)" stroke-opacity=".75" stroke-dasharray="2 2"/>`).join("")
      + `<line x1="${c}" x2="${c}" y1="2" y2="14" stroke="var(--muted)"/>${imb == null ? "" : `<rect x="${x0.toFixed(1)}" y="4" width="${Math.max(1.5, Math.abs(v) * s).toFixed(1)}" height="8" rx="2" fill="${fill}"/>`}</svg>`;
  },
  book(sw, bps, w = mcGfx.pw || 300, mid = 0, dp = 2, rh = 21, fs = 10, bh = 13) {   // rh·fs·bh = 줄 간격·글자·막대 두께(2026-10-13 요약 줄은 크게)   // 매수벽 | 매도벽 거울 막대 -- 양 끝 = 그 벽이 닿는 가격(현재가 ±25·50·100bp, 2026-10-06 사용자 «±25 말고 가격으로»)
    const b = sw.bid, a = sw.ask, mx = Math.max(1, ...b, ...a), c = w / 2, h = 4 + bps.length * rh, gap = mid > 0 ? 6 : 18, P = mid > 0 ? 56 : 0, Q = 46;
    const kq = (v) => (v >= 1e3 ? (v / 1e3).toFixed(1) + "k" : String(Math.round(v)));
    const pf = (v) => v.toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp });
    return `<svg class="mc-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="호가 벽 매수 대 매도">` + bps.map((bp, i) => {
      const y = 4 + i * rh, ty = y + Math.round(bh / 2 + fs * 0.36), room = Math.max(10, c - gap - Q - P), bw = room * b[i] / mx, aw = room * a[i] / mx, lo = mid * (1 - bp / 1e4), hi = mid * (1 + bp / 1e4);
      return `<g><title>${mid > 0 ? `현재가에서 ${pf(lo)}까지 매수호가 ${kq(b[i])} · ${pf(hi)}까지 매도호가 ${kq(a[i])} (±${bp}bp)` : `±${bp}bp`}</title>`
        + `<rect x="${(c - gap - bw).toFixed(1)}" y="${y}" width="${bw.toFixed(1)}" height="${bh}" rx="2" fill="var(--good)" opacity=".75"/><rect x="${c + gap}" y="${y}" width="${aw.toFixed(1)}" height="${bh}" rx="2" fill="var(--bad)" opacity=".75"/>`
        + `<text x="${(c - gap - 4 - bw).toFixed(1)}" y="${ty}" font-size="${fs}" fill="var(--text)" text-anchor="end">${kq(b[i])}</text><text x="${(c + gap + 4 + aw).toFixed(1)}" y="${ty}" font-size="${fs}" fill="var(--text)">${kq(a[i])}</text>`
        + (mid > 0 ? `<text x="0" y="${ty}" font-size="${fs}" fill="var(--good)">${pf(lo)}</text><text x="${w - 1}" y="${ty}" font-size="${fs}" fill="var(--bad)" text-anchor="end">${pf(hi)}</text>`
          : `<text x="${c}" y="${ty}" font-size="${fs}" fill="var(--muted)" text-anchor="middle">±${bp}</text>`)
        + `</g>`;
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
  if (activePageTab !== "snapshot" || document.hidden || !flowOn()) return;   // 흐름 엔진 없는 코인은 서버가 404
  const now = Date.now();
  if (now - marketCtxLastFetchAt < MARKET_CTX_POLL_MS) return;
  marketCtxLastFetchAt = now;
  const asset = activeSnapshotAsset;
  let got;
  try {
    const res = await fetch(`${API_MARKET_CTX_URL}?asset=${asset}`, { cache: "no-cache" });
    got = res.ok ? await res.json() : { available: false, error: `HTTP ${res.status}` };
  } catch (error) {
    got = { available: false, error: "fetch_failed" };
  }
  if (asset !== activeSnapshotAsset) return;   // 그 사이 코인이 바뀌었다
  latestMarketCtx = got;
  renderMarketCtx();
}

async function refreshPaper() {
  if (activePageTab !== "snapshot" || document.hidden) return;
  const now = Date.now();
  if (now - paperLastFetchAt < PAPER_POLL_MS) return;
  paperLastFetchAt = now;
  const asset = activeSnapshotAsset;
  let got;
  try {
    const res = await fetch(`${API_PAPER_URL}?asset=${asset}`, { cache: "no-cache" });
    got = res.ok ? await res.json() : { available: false, error: `HTTP ${res.status}` };
  } catch (error) {
    got = { available: false, error: "fetch_failed" };
  }
  if (asset !== activeSnapshotAsset) return;   // 그 사이 코인이 바뀌었다
  latestPaper = got;
  renderPaper();
}

// 2026-10-04 모의 엔진 신호(마지막 5분봉 마감 값): 맞대결(정시에 잡은 고래 방향, 다음 정시까지) · 벽 신호(적응 문턱) · 추격 주의.
//   엔진이 없거나 멈췄으면 null -- 칩은 «엔진 대기»(조용히 «없음»이라 말하지 않는다).
function paperSignals(d = latestPaper, nowS = Date.now() / 1000) {
  const sg = d && d.available && !d.stale ? d.signals : null;
  if (!sg || sg.bar_s == null || sg.thr == null) return null;
  const remain = Math.ceil(((Math.floor((sg.bar_s + 1) / 3600) + 1) * 3600 - nowS) / 60);
  const dir = sg.imb50 > sg.thr ? 1 : sg.imb50 < -sg.thr ? -1 : 0;
  return { duel: sg.whale_hold && remain > 0 ? sg.whale_hold : 0, remain, zw: sg.zw, zr: sg.zr,
           imb: sg.imb50, thr: sg.thr, dir, chase: dir !== 0 && sg.r5 * dir > 0, r5: sg.r5 };
}

// ── 2026-10-06 넓은 화면 모의 판 칸(#mcWall) = 서랍 둘: 모의 판 · 지지/저항 (사용자 «드로우로 올리고 내리게 · 둘 중 하나만 열려 카드 높이가 안 움직이게») ──
//   칸 높이는 풋프린트가 정한다(mcPlace) -- 열린 서랍이 남은 높이를 다 쓰고, 닫힌 서랍은 머리 한 줄(+ 요약, 사용자 선택 B). 머리를 누르면 그 서랍이 열리고(열린 걸 누르면 다른 쪽).
function wallDrawerHtml() {   // 2026-10-13 아티팩트 댓글 «라벨 지우고 숫자·게이지만»: 머리·줄 이름 없이 가격 · 강도 게이지 · 거리 %(CSS #mcWall .wdr-sr)
  const sr = srLevelsLive();
  return `<div class="wdr-wrap"><section class="wdr open" aria-label="지지 · 저항(청산맵 밀집 · 가까운 3개씩)"><div class="wdr-b">`
    + (sr && (sr.res.length || sr.sup.length) ? `<div class="liq-level-list wdr-sr">${srRowsHtml(sr)}</div>`
      : `<div class="ppm-empty">${latestLiquidationMap && latestLiquidationMap.warmed_up ? "추정 가능한 밀집 구간이 아직 없습니다" : "청산맵 데이터 수집 중"}</div>`)
    + `</div></section></div>`;
}

function renderPaper() {
  const wall = el("mcWall");   // 2026-10-13 모의 판이 #mcBody 에서 빠져 강제 다시 그리기(_mcHtml = null)는 뺐다 -- renderMarketCtx 가 내용 비교로 바뀜을 잡는다
  if (wall) { const html = wallDrawerHtml(); if (wall._ppm !== html) { wall._ppm = html; keepFocus(wall, () => { wall.innerHTML = html; }); } }
}

// 🔴2026-09-30 청산 «급증»은 **쪽마다** 판정한다. 옛 판은 `z>=3 || hawkes_active` 를 롱·숏 막대에 똑같이 걸어
//   hawkes_active 하나로 z −0.24 인 숏 막대까지 주황·«급증»이 됐다. 규칙: 그 쪽 z>=3 이거나, hawkes 가 켜져 있고
//   그 쪽이 1분 금액이 0 이 아닌 **큰 쪽**일 때. 파일(tail_risk_interceptor 10초마다 씀)이 60초 넘게 낡았으면 급증 없음.
//   순수 함수(test/test_mc_liq_burst_hot_20260930.py 가 본문을 떼어 돌린다).
const LIQ_BURST_STALE_MS = 60000;
// 🔴2026-10-04 판정을 연구(H2)와 같은 사건으로: 직전 60초 그 쪽 청산 합 > 문턱(서버 LIQ_BURST_60S_USD, 학습 7일 상위 0.5%).
//   옛 판(봇 1분 z≥3 · hawkes)은 σ 바닥 $1 이라 조용한 30분 뒤 몇천 달러에도 켜졌다(하루 11.8번 · 발동 중앙 $1,711 · 15분 되돌림 −0.71bp).
function mcLiqBurstHot(bu, side, nowMs = Date.now()) {
  if (!bu || !Array.isArray(bu.thr) || !(nowMs / 1000 - Number(bu.ts || 0) <= LIQ_BURST_STALE_MS / 1000)) return false;
  return Number((side === "long" ? bu.long_usd_60s : bu.short_usd_60s) || 0) > bu.thr[side === "long" ? 0 : 1];
}
// 꼬리표·툴팁이 말할 쪽: 급증인 쪽, 둘 다면 1분 금액이 큰 쪽. 없으면 null.
function mcLiqBurstSide(bu, nowMs = Date.now()) {
  const hl = mcLiqBurstHot(bu, "long", nowMs), hs = mcLiqBurstHot(bu, "short", nowMs);
  if (!hl && !hs) return null;
  return hs && (!hl || bu.short_usd_60s / bu.thr[1] > bu.long_usd_60s / bu.thr[0]) ? "short" : "long";   // 둘 다면 문턱 대비 더 큰 쪽
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

// 2026-10-26 펜 메모 «여기에 vwma 100, ema5, ema 20 도 추가해줘» -- 볼린저처럼 전체 이력에서 센다. {봉 시각: 값}
//   EMA = 종가 지수이동평균(α = 2/(n+1)) · 첫 값 = 처음 n봉 단순평균.
function mcEma(full, n) {
  const out = new Map(), a = 2 / (n + 1);
  let e = 0;
  full.forEach((c, j) => {
    e = j < n ? e + +c.close / n : e + a * (+c.close - e);
    if (j >= n - 1) out.set(c.time, e);
  });
  return out;
}
//   VWMA = Σ(종가 × 거래량) ÷ Σ거래량, n봉. 거래량 = 바이낸스 5분봉 거래량(캔들 volume -- 종가와 같은 시장 · 트레이딩뷰 VWMA 와 같은 정의).
//   2026-10-26 사용자 선택: 풋프린트 체결(바이낸스+OKX)에서 바꿨다 -- 1주일 실측 선 차이 중앙 0.8bp·최대 3.7bp 인데 수집기 빈 봉에 선이 안 끊긴다.
//   1주일 첫 봉 앞은 서버 volPre(100시간 [시각, 종가, 거래량])를 tf 봉으로 묶어 이어 붙인다 -- 1h 에서도 1주일 첫 봉부터 선이 있다.
//   n봉 중 하나라도 거래량이 없으면(형성 5분봉 등) 그 점은 없다(부분 창을 n봉인 척하지 않는다).
function mcVwma(full, n) {
  const wk = fpWeekFor(), t0 = full.length ? full[0].time : 0, pre = [];
  ((wk && wk.pre) || []).forEach(([t, c, q]) => {
    const T = tfT(t), a = pre[pre.length - 1];
    if (T >= t0) return;
    if (a && a.time === T) { a.close = c; a.volume += q; } else pre.push({ time: T, close: c, volume: q });
  });
  const rows = pre.length && pre[pre.length - 1].time + 300 * chartTfK === t0 ? pre.concat(full) : full;   // 이어질 때만(라이브 16시간만 볼 땐 사이가 빈다)
  const out = new Map();
  let pv = 0, v = 0, have = 0;
  rows.forEach((c, j) => {
    const q = +c.volume;
    if (q > 0) { pv += q * c.close; v += q; have++; }
    const o = j >= n ? rows[j - n] : null, qo = o ? +o.volume : 0;
    if (qo > 0) { pv -= qo * o.close; v -= qo; have--; }
    if (have === n) out.set(c.time, pv / v);
  });
  return out;
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
  if (body.classList.contains("mc-cmp") !== !!r) { body.classList.toggle("mc-cmp", !!r); body._mcHtml = null; renderMarketCtx(); mcPlaced = true; renderFpStrip(); }   // 10-13 넓은 화면이면 머리 띠(금리)를 끈다
  else if (!mcPlaced) { mcPlaced = true; renderFpStrip(); }
}

function renderMarketCtx() {
  const body = el("mcBody");
  if (!body) return;
  const A = activeSnapshotAsset, eth = A === "eth", U = coinUnit();   // 2026-10-04 코인별(SOL·XRP)
  const ratesOff = () => { const f = el("fpRates"); if (f && !f.hidden) { f.hidden = true; f.innerHTML = ""; } };   // 10-17 리뷰: 아래 두 return 에서 머리 칸이 마지막 값으로 멈춰 있었다
  if (!flowOn()) { body.innerHTML = ""; body._mcHtml = null; ratesOff(); return; }
  const d = latestMarketCtx && latestMarketCtx.asset === A ? latestMarketCtx : null;   // 다른 코인 응답은 «불러오는 중»
  if (!d || !d.available) {
    body.innerHTML = `<div class="mc-note">${d && d.error ? `시장 맥락 지연 (${escapeHtml(String(d.error))})` : "불러오는 중…"}</div>`;
    body._mcHtml = null;
    ratesOff();
    return;
  }
  const n = (v, dp = 0) => (v == null || !Number.isFinite(+v) ? "-" : (+v).toLocaleString("en-US", { maximumFractionDigits: dp, minimumFractionDigits: dp }));
  const sg = (v, dp = 1, u = "") => (v == null || !Number.isFinite(+v) ? "-" : `${+v >= 0 ? "+" : "−"}${n(Math.abs(v), dp)}${u}`);
  const fr = (v) => (v == null ? "-" : `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(4)}%`);
  const left = (ms) => { if (!ms) return "-"; const s = Math.max(0, (ms - Date.now()) / 1000); return s >= 3600 ? `${Math.floor(s / 3600)}시간 ${Math.floor((s % 3600) / 60)}분` : `${Math.ceil(s / 60)}분`; };
  const tone = (v, thr = 0.5) => (v == null || Math.abs(v) < thr ? "" : v > 0 ? "mc-good" : "mc-bad");
  // 2026-10-04 고래·리테일 경계는 코인별(서버 bands_usd) -- 설명 글의 ETH 값($10만/$1만)을 바꿔 끼운다
  //   🔴키는 «q_flow»(= flow + quad 이어 붙임)다 -- 키 이름으로 고르지 말고 글에서 바꾼다(첫 판은 "flow" 로 걸러 한 번도 안 맞았다)
  mcTipOf = (key) => (!eth && Array.isArray(d.bands_usd)   // 팝업이 열 때 읽는다(helpOpen)
    ? MC_TIPS[key].replace("고래 ≥ $10만 · 리테일 < $1만", `고래 ≥ ${fmtUsdCompact(d.bands_usd[1])} · 리테일 < ${fmtUsdCompact(d.bands_usd[0])}`) : MC_TIPS[key]);
  const sec = (key, title, inner) => `<div class="mc-sec"><h4><button type="button" class="mc-q" ${helpAttrs(key)}>${title}<span aria-hidden="true">?</span></button></h4>${inner}</div>`;
  const f = d.funding || {}, b = d.basis || {}, oi = d.oi || {}, ls = d.ls, fl = d.flow || {}, z = fl.z30h_all || fl.z30h || {},   /* 2026-10-26 사용자 «② 다섯 줄 모두 30분»: 30분 합 σ(같은 시간대 14일 기준) -- 60분 값으로 섞어 채우지 않는다(없으면 «기준 쌓는 중») */   // z60h = 같은 UTC 시 14일 기준(2026-10-01) · 10-05 요일유형(평일/주말) 분리, 없으면 옛 12~24h 링
    bk = d.book || {};
  const levWarn = ["long_crowd", "short_crowd", "deleverage"].includes((d.lev || {}).key);
  // 일정 -- 옵션 만기는 옵션 카드와 같은 원천(latestGex). 🔴optData() 는 «지금 보는 코인»이라 SOL 탭에서 SOL max pain 을 ETH 가격과
  //   견줬다(09-29 재검증) -- 이 카드는 ETH 전용이므로 ETH 를 직접 읽는다.
  const gEth = latestGex && latestGex.available ? (latestGex.currencies || {}).ETH : null, o = gEth && gEth.options;
  const ef = o ? optFront(o) : null, us = mcUsSession();
  // 2026-09-30 사용자 «주요 경제 일정이 왜 24시간 축에 없나» -- 전에는 «다음 high 하나»만 실었다. 경제 일정 카드와 같은 원천의 24시간 안 전부.
  const macro24 = (latestMacroEvents || []).map((e) => ({ t: Date.parse(e.time_utc), nm: e.title_ko || e.title || "지표", hi: e.importance === "high" }))
    .filter((e) => e.t > Date.now() && isTodayOrTomorrowLocal(e.t));   // 2026-09-30 경제 일정 카드 제거 -- 카드가 보이던 오늘·내일 전부(24시간 넘는 건 시간축 오른쪽 끝 «>»)
  const sw = bk.sweep, bps = bk.bps || [25, 50, 100], sdp = Math.max(2, (ASSET_CONFIG[A] || {}).dp || 2);   // 호가 1틱 = ETH·SOL 0.01 · XRP 0.0001
  const thin = bk.bid25_pct != null && bk.ask25_pct != null && Math.min(bk.bid25_pct, bk.ask25_pct) <= 0.1
    ? (bk.ask25_pct <= bk.bid25_pct ? "위쪽(매도호가)이 얇다" : "아래쪽(매수호가)이 얇다") : null;
  // 2026-10-04 ② 맞대결 · ④ 벽 신호 상태 줄 + 깊은 벽 줄(원천 = 모의 엔진, 5분봉 마감 값)
  // 맞대결·벽 신호 = 그 코인 모의 엔진의 마지막 봉 값. 줄은 그 코인에서 검정을 통과한 것만(PAPER_SIG_OK, 2026-10-04 사용자 «각각 테스트 및 적용»)
  const ps = latestPaper && latestPaper.asset === A ? paperSignals() : null, chip = (cls, txt, tip = "") => `<span class="mc-chip ${cls}"${tip ? ` title="${escapeHtml(tip)}"` : ""}>${txt}</span>`;
  const zTip = ps ? `30일 기준 고래 z ${sg(ps.zw, 1)} · 리테일 z ${sg(ps.zr, 1)} (정시 판정)` : "";
  const duelLine = !PAPER_SIG_OK.duel.has(A) ? "" : `<div class="mc-state"><span class="mc-k">맞대결</span>${!ps ? chip("", "엔진 대기")
    : ps.duel ? chip(ps.duel > 0 ? "good" : "bad", `고래 ${ps.duel > 0 ? "롱" : "숏"} · 남은 ${ps.remain}분`, zTip) : chip("", "없음", zTip)}</div>`;
  const wallLine = !PAPER_SIG_OK.wall.has(A) ? "" : `<div class="mc-state"><span class="mc-k">벽 신호</span>${!ps ? chip("", "엔진 대기") : !ps.dir ? chip("", "없음")
    : ps.chase ? chip("warn", `${ps.dir > 0 ? "매수" : "매도"} · 추격 주의`, `직전 5분 ${sg(ps.r5, 0, "bp")} — 이미 그 방향으로 움직였다`)
    : chip(ps.dir > 0 ? "good" : "bad", `${ps.dir > 0 ? "매수" : "매도"} 우세 · 5분`, `직전 5분 ${sg(ps.r5, 0, "bp")}`)}</div>`;
  // ── 2026-09-30 시안 B(사용자 선택 «글자가 너무 많다 → 그릴 수 있는 건 그림으로»): 질문 네 판 + 다음 24시간 ──
  //   분위 = 0~100 게이지(양끝 10% 주황) · σ = 0 가운데 막대(±2σ 점선) · 비율 = 1 가운데 로그 눈금(빈 점 = 하루 전).
  //   그림 없는 부값(OKX/HL 펀딩·OI 합계·거래소별 30분 체결·스프레드·12시간 청산 최다·청산 급증)은 판마다 흐린 한 줄로 남긴다.
  // 2026-09-30 사용자 «카드 너비 100% 안을 내용 크기를 키워 채워» -- 판 폭(격자 auto-fit 300px 칸)에서 그림 폭을 정한다.
  { const cmp = body.classList.contains("mc-cmp"), gap = cmp ? 14 : 22;   // 2026-09-30 시안 Y: 풋프린트 1초 수급 아래 칸(~420px)에 2열 압축
    const BW = body.clientWidth || 1200, cols = cmp ? 1 : Math.max(1, Math.min(4, Math.floor((BW + 22) / 322)))   /* 판은 넷 -- auto-fit 이 빈 칸을 접어 넷이 폭을 나눠 가진다 */, colW = Math.floor((BW - (cols - 1) * gap) / cols);
    body.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;   // 🔴auto-fit 은 전폭 줄(⑤)이 있으면 빈 칸을 못 접어 5칸이 됐다 -- 계산과 같은 칸 수로 못 박는다
    mcGfx.pw = Math.min(colW, 560); mcGfx.gw = cmp ? Math.max(48, colW - 64 - 46 - 12) : Math.max(110, Math.min(300, colW - 96 - 90 - 16));
    mcGfx.qw = cmp ? Math.max(70, Math.min(100, Math.round(colW * 0.42))) : Math.max(96, Math.min(150, Math.round(colW * 0.3)));
    mcGfx.th = cmp ? Math.round(Math.max(170, colW * 0.9)) : null; mcGfx.cmpW = cmp ? BW : null; }
  // 2026-10-06 거래소 합산(사용자 «바로 합산») -- 줄 호버에 무엇을 더했나. 합산이 안 될 때는 «바이낸스만»(조용히 정의가 바뀌지 않게)
  const VN = { binance: "바이낸스", okx: "OKX", hl: "HL", bybit: "Bybit" }, venTxt = (v) => (v && v.length ? v.map((k) => VN[k] || k).join("+") : "바이낸스만");
  const oiA = oi.all && oi.all.z30m != null ? oi.all : null, OI = oiA || oi, flowVen = fl.z30h_all ? venTxt(fl.venues) : "바이낸스만";   // 2026-10-26 OI·크기별 줄 = 30분판   // 합산 OI z 가 없으면(한 거래소 1시간 전 칸 빔) 바이낸스로
  const G = mcGfx, gRow = (label, vis, val, cls = "", tip = "") => `<div class="mc-g"${tip ? ` title="${escapeHtml(tip)}"` : ""}><span>${label}</span>${vis}<b class="${cls}">${val}</b></div>`;
  const note = (s) => `<div class="mc-note">${s}</div>`;
  const qSec = (key, title, inner, extra = "") => sec(key, title, (extra ? `<div class="mc-state">상태${extra}</div>` : "") + inner);
  const levChip = ` <span class="mc-chip${levWarn ? " warn" : ""}">${escapeHtml(((d.lev || {}).label || "-").split(" —")[0])}</span>`;
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
      // 2026-10-26 사용자 «델타와 OI 는 펀딩·베이시스 쪽으로»: 휴대폰 ① 도 델타 30분·OI 30분(데스크톱과 같은 구성 · 다섯 줄 모두 30분)
      + gRow("델타 30분", G.sig(fl.cvd30_all_z ?? fl.cvd30_z), sg(fl.cvd30_all_z ?? fl.cvd30_z, 1, "σ"), tone(fl.cvd30_all_z ?? fl.cvd30_z),
             fl.cvd30_all_z != null ? flowVen : "바이낸스만")
      + gRow("OI 30분", G.sig(OI.z30m), `${sg(OI.z30m, 1, "σ")}`, tone(OI.z30m, 1.5), `${oiA ? venTxt(oiA.venues) : "바이낸스만"} · 30분 ${sg(OI.d30m_pct, 2, "%")} · 24시간 ${sg(OI.d24h_pct, 1, "%")}`
          + (oiA ? `\n${Object.entries(oiA.by_venue || {}).map(([k, v]) => `${VN[k] || k} ${sg(v, 2, "%")}`).join(" · ")}` : ""))
      + (ls ? gRow("계정 롱숏", G.ratio(ls.global, ls.global_24h), n(ls.global, 2), "", `하루 전 ${n(ls.global_24h, 2)}`)
        + gRow("탑트레이더", G.ratio(ls.top_pos, ls.top_pos_24h), n(ls.top_pos, 2), "", `하루 전 ${n(ls.top_pos_24h, 2)}`) : "")
      + note(`정산까지 ${left(f.next_ms)} · 펀딩 OKX ${fr(f.okx)} / HL ${fr(f.hl_8h)} · OI ${n((oi.oi || 0) / 1e6, 2)}M + OKX ${n((oi.okx || 0) / 1e6, 2)}M + HL ${oi.hl == null ? "-" : n(oi.hl / 1e6, 2) + "M"} ${U}`), levChip),
    qSec("q_flow", "② 누가 밀고 있나 · 30분", duelLine + (z.whale == null ? note("크기별 기준 쌓는 중(6시간)")
        : ["whale", "mid", "retail"].map((k2, i) => gRow(["고래", "중형", "리테일"][i], G.sig(z[k2]), sg(z[k2], 1, "σ"), tone(z[k2]),
            `${flowVen} · 30분 ${sg((fl.net30_all || {})[k2], 0)} ${U}`)).join(""))
      + ((fl.taker_all ?? (ls && ls.taker)) != null ? (() => { const tk = fl.taker_all ?? ls.taker;
          return gRow("테이커 매수÷매도", G.ratio(tk, null), n(tk, 2), tone(tk - 1, 0.1), fl.taker_all != null ? `${flowVen} · 마지막 5분봉 체결` : "바이낸스 API"); })() : "")
      // 2026-09-30 사분면 그림 → 한 줄 네 칸(사용자 «높이를 쓸데없이 잡아먹는다»): 새 롱 · 숏 커버 · 새 숏 · 롱 정리 중 지금 칸만 채움, 작으면 전부 빈 칸.
      + gRow("가격×OI 1h", G.seg4(d.quad ? d.quad.key : null), "", "",
             `네 칸 = 신규 롱 · 숏커버 · 신규 숏 · 롱정리 (초록 = 가격↑ · 빨강 = 가격↓ · 다 비면 보류)\n${d.quad ? `지금 ${({ up_up: "새 롱", up_dn: "숏 커버", dn_up: "새 숏", dn_dn: "롱 정리", small: "보류" })[d.quad.key] || "-"} · ${d.quad.label}` : "-"}\n1시간 ${sg(d.move60, 0, "bp")} · OI ${sg(oiA && oiA.d1h != null ? oiA.d1h : d.oi60, 0)} ${U} (${oiA && oiA.d1h != null ? venTxt(oiA.venues) : "바이낸스만"})${d.quad && d.quad.note ? `\n${d.quad.note} — 근거 검정은 바이낸스 OI 기준` : ""}`)
      + note(`30분 체결 · 바이낸스 ${sg(fl.bn30, 0)} / OKX ${sg(fl.okx30, 0)} ${U}`)),
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
  // 2026-10-26 사용자 «차트에 그리기 체크리스트도 깔끔하게»: 체크박스 7개 → 칩 두 묶음(이동평균·볼린저 | VWAP 셋). 칩 앞 견본 = 그 선의 색·모양(범례 겸)
  //   «세션 시작부터»는 VWAP 계산 기준이라 VWAP 칩에 붙인 보조 칩 · VWAP 를 끄면 못 누른다. 입력(data-line)·저장·이벤트는 그대로.
  const swChip = (k, nm, sw, t = "", sub = false) => `<label class="mc-chip${sub ? " sub" : ""}"${t ? ` title="${t}"` : ""}${sw ? ` style="--sw: ${sw}"` : ""}>`
    + `<input type="checkbox" class="chip-input" data-line="${k}"${mcLine[k] ? " checked" : ""}${sub && !mcLine.vwap ? " disabled" : ""}>`
    + `${sw ? `<i class="mc-sw-key${/ma-|amber/.test(sw) ? "" : " dash"}" aria-hidden="true"></i>` : ""}${nm}</label>`;
  const lineSw = `<div class="mc-switch"><span class="mc-switch-t">차트에 그리기</span>`
      + `<span class="mc-sw-g mc-sw-ma">${[["ema5", "EMA 5", "종가 지수이동평균 5봉"], ["ema20", "EMA 20", "종가 지수이동평균 20봉"], ["vwma", "VWMA 100", "거래량 가중 이동평균 100봉 · 거래량 = 바이낸스 5분봉"]]
        .map(([k, nm, t]) => swChip(k, nm, `var(--ma-${k})`, t)).join("")}${swChip("bb", "볼린저", "var(--muted)", "20봉 · 2σ")}${swChip("veto", "추세 veto", "var(--muted)", "SMA144 ±1×ATR 히스테리시스 밴드 · 롱만/숏만 꼬리표")}</span>`
      + `<span class="mc-sw-g"><span class="mc-chip-pair">${swChip("vwap", "VWAP ±σ", "var(--muted)")}${swChip("vsess", "세션 시작부터", "", "끄면 하루(UTC 00시부터) · 켜면 지금 시장 세션 개장부터(아시아·유럽·미국)", true)}</span>`
      + `${swChip("wvwap", "주간·앵커 VWAP", "var(--ink)")}</span>`
      // 2026-10-29 사용자 «5분봉 거래량 제일 많은 부분을 이은 선도 on/off» -- 봉별 POC 선(기본 켬 · 봉마다 POC 테두리는 그대로)
      + `<span class="mc-sw-g">${swChip("poc", "POC 선", "var(--amber)", "봉마다 가장 많이 거래된 가격(POC)을 이은 선 · 봉 안 POC 테두리는 그대로")}</span></div>`;
  // 2026-10-28 펜 메모 «지표 아이콘 버튼을 누르면 이 지표 버튼들이 팝업으로 나와 추가/제거» -- 넓은 화면 줄 = 버튼 하나(켠 개수) · 칩·입력·저장은 그대로 팝업 안
  const indOn = ["ema5", "ema20", "vwma", "bb", "veto", "vwap", "wvwap", "poc"].filter((k) => mcLine[k]).length;
  const fsRow = G.cmpW || (isMobileChartMode() && flowOn() && !!el("fpMobRow"));   // 옛 index.html(줄 없음)이면 칩을 시장 맥락에 그대로   // 2026-10-28 사용자 «모바일도 데스크탑처럼» -- 휴대폰 흐름 코인도 버튼 + 팝업(#fpMobRow)
  if (!fsRow) fpIndOpen = false;   // 버튼 줄이 없는 화면으로 가면 닫힌 채로(돌아왔을 때 저절로 열려 있지 않게)
  const fsEl = el("fpLineSwitch"), fsHtml = fsRow ? `<button type="button" class="fp-ind-btn" aria-haspopup="true" aria-expanded="${fpIndOpen}" aria-controls="fpIndPop">`
    + `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M1.5 12.5l4-4.5 3 2.5 6-6.5"/><path d="M1.5 14.5h13"/></svg>지표<b>${indOn}</b></button>`
    + `<div class="fp-ind-pop" id="fpIndPop" aria-label="차트에 그릴 지표"${fpIndOpen ? "" : ' hidden=""'}>`
    + lineSw.replace('<span class="mc-switch-t">차트에 그리기</span>', '<span class="mc-switch-t">이동평균·밴드 · VWAP · POC — 눌러서 켜고 끄기</span>') + `</div>` : "";
  // 🔴setH(직렬화한 innerHTML 과 비교)는 여기서 매번 다시 썼다 -- `checked`→`checked=""`·`<path/>`→`</path>` 라 늘 달라 클릭·포커스가 씹혔다(10-28 검증). 마지막으로 쓴 글과 비교한다.
  if (fsEl && fsEl._h !== fsHtml) {
    const fk = fsEl.contains(document.activeElement) ? document.activeElement.dataset.line || (document.activeElement.classList.contains("fp-ind-btn") ? "btn" : "") : "";
    fsEl._h = fsHtml; fsEl.innerHTML = fsHtml;
    if (fk) fsEl.querySelector(fk === "btn" ? ".fp-ind-btn" : `[data-line="${fk}"]`)?.focus();   // 키보드로 칩을 바꾸면 다시 써도 그 칩에 포커스
  }
  const wallRow = ps && PAPER_SIG_OK.wall.has(A) ? gRow("깊은 벽", G.wall(ps.imb, ps.thr), sg(ps.imb, 2), ps.dir > 0 ? "mc-good" : ps.dir < 0 ? "mc-bad" : "",
    `±50bp 매수·매도 잔량 불균형(5분봉 마감) · 점선 = 지난 24시간 상위 20% 문턱 ±${ps.thr.toFixed(2)}`) : "";
  const wallHtml0 = qSec("q_wall", "④ 벽 · 교차 · 위치", wallLine + (sw ? G.book(sw, bps, wallW || undefined, Number(d.mid) || 0, (ASSET_CONFIG[A] || {}).dp ?? 2) : note("호가 래스터 대기")) + wallRow + statRows
      + note(`스프레드 ${bk.spread == null ? "-" : "$" + bk.spread.toFixed(sdp)}${bk.spread > 1.5 * 10 ** -sdp ? " — 평소(1틱)보다 넓다" : ""}`
        + (bk.bid25_pct == null ? "" : ` · 얇은 쪽 매수 ${Math.round(bk.bid25_pct * 100)} · 매도 ${Math.round(bk.ask25_pct * 100)}분위${thin ? ` — ${thin}` : ""}`))
      + (fsRow ? "" : lineSw));
  // 2026-10-15 ⑤ 뉴스 판 → ⑤ 예측 줄(사용자 시안 B): 지금 추세 → 4h → 7일 → 4~13주, 4h 칸 아래 변동성. 뉴스는 뉴스 카드에 그대로.
  const fcHtml = fcRowHtml(sec);   // 2026-10-26 펜 메모 «풋프린트 미래 차트로 옮겼으니 제거»: ⑤ 아래 박스 유지 30분·1시간·4시간 줄 삭제(풋프린트 미래 칸 상자 d38cb502 가 같은 값)
  G.gw = gwKeep;
  // 2026-10-13 시안(아티팩트 댓글 «4개 판을 함축해 가로 1줄 · 나머지 1줄에 체결 버블»): 넓은 화면은 판마다 상태 칩 + 값 셋만 한 줄로,
  //   판 설명(?)은 그대로. 자세한 그림은 좁은 화면·이전 판에. («지금 들어가면» 가격 자는 2026-10-28 펜 메모로 지웠다 -- 그 높이는 1초 수급으로)
  if (G.cmpW) {
    // 2026-10-13 사용자 «글자 말고 모두 게이지나 그래프로» -- 줄 = 짧은 이름 + 그림, 숫자는 호버(title). 상태 칩(①②④)만 글자로 남긴다.
    // 2026-10-13 아티팩트 댓글(정리 3차): 오른쪽 칸 = 1초 수급 → 체결 버블(수급 차트 한 개 높이, renderCandleSvg 가 맞춘다) →
    //   요약 한 줄 ①②④⑤(폭 1 : 1 : 1.6 : 1 -- ④ 는 호가 벽 막대) → 금리 줄(5년·10년 긴 스파크라인, «수급차트와 같은 너비»).
    //   칸마다 그림 줄은 칸 높이에 고르게 펼친다(.mc-rows) -- 줄 수가 달라도 빈 데 없이 찬다. 불균형·이탈은 뺐다(사용자).
    // 2026-10-13 사용자 «라벨은 게이지 위 or 아래로»: 이름표는 그림 아래 작은 글씨(CSS .mc-strip .mc-gk) -- 그림이 칸 폭을 다 쓴다
    const unit = (G.cmpW - 2 * 12) / 4.0, gw = Math.max(40, Math.floor(unit * 1.2) - 4), gwWall = Math.max(60, Math.floor(unit * 1.6) - 4);   // 폭 1.2 : 1.2 : 1.6 (값이 붙는 ①② 를 넓게) · 10-15 ⑤ 뉴스 칸 → 아래 ⑤ 예측 줄
    const gk = (k, vis, tip = "", val = null, cls = "") => `<div class="mc-gk${val == null ? "" : " v"}"${tip ? ` title="${escapeHtml(tip)}"` : ""}><span>${k}</span>${vis}${val == null ? "" : `<b class="${cls}">${val}</b>`}</div>`;   // val = 오른쪽 값(2026-10-13 사용자 «라벨 되돌려»)
    const gv = gw;   // 2026-10-13 «값도 게이지 아래로» -- 값이 아래 줄로 가서 그림은 칸 폭 전체
    // 2026-10-15 사용자 «①②④ 제목 제거»: 판 이름은 «?» 의 title·aria 로만 -- 설명은 팝업 제목으로도 쓴다(helpOpen)
    const tile = (key, title, chipHtml, rows, cls = "") => `<div class="mc-sec${cls}" aria-label="${escapeHtml(title)}"><div class="mc-state">${chipHtml || ""}`
      + `<button type="button" class="mc-q mc-q1" ${helpAttrs(key)} title="${escapeHtml(title)} — 설명" aria-label="${escapeHtml(title)} 설명"><span aria-hidden="true">?</span></button></div>`
      + `<div class="mc-rows">${rows}</div></div>`;
    const duelChip = !PAPER_SIG_OK.duel.has(A) ? chip("", "검정 밖") : !ps ? chip("", "엔진 대기")
      : ps.duel ? chip(ps.duel > 0 ? "good" : "bad", `고래 ${ps.duel > 0 ? "롱" : "숏"} · ${ps.remain}분`, zTip) : chip("", "맞대결 없음", zTip);
    const wallChip = !PAPER_SIG_OK.wall.has(A) ? chip("", "검정 밖") : !ps ? chip("", "엔진 대기") : !ps.dir ? chip("", "벽 신호 없음")
      : ps.chase ? chip("warn", `${ps.dir > 0 ? "매수" : "매도"} · 추격 주의`) : chip(ps.dir > 0 ? "good" : "bad", `${ps.dir > 0 ? "매수" : "매도"} 우세`);
    const cvd = fl.cvd30_all_z ?? fl.cvd30_z;
    htmlB = [`<div class="mc-strip">`
      + tile("q_lev", "① 레버리지 과열?", levChip.trim(),
          gk("펀딩", G.pct(f.bn_at_base ? null : f.bn_pct180, gv), f.bn_at_base ? "펀딩 기본값(0.01%)" : `펀딩 분위 ${Math.round((f.bn_pct180 ?? 0) * 100)}% · 바이낸스 예상 ${fr(f.bn)}`,
             f.bn_at_base ? "기본" : `${Math.round((f.bn_pct180 ?? 0) * 100)}%`)
          + gk("베이시스", G.pct(b.pct7d, gv), `베이시스 ${sg(b.bp, 1, "bp")} · 7일 ${Math.round((b.pct7d ?? 0) * 100)}분위`, sg(b.bp, 1, "bp"))
          // 2026-10-26 사용자 «델타와 OI 는 펀딩·베이시스 쪽으로 · 밸런스»: ① = 펀딩·베이시스·델타(30분 CVD)·OI(1시간) 4줄 · ② = 고래·중형·리테일+가격×OI
          + gk("델타", G.sig(cvd, gv), `델타 = 30분 CVD(테이커 매수−매도 누적) ${sg(cvd, 1, "σ")} · 풋프린트 아래 «델타» 줄과 같은 데이터`, sg(cvd, 1, "σ"), tone(cvd))
          + gk("OI", G.sig(OI.z30m, gv), `OI 30분 ${sg(OI.z30m, 1, "σ")} · ${oiA ? venTxt(oiA.venues) : "바이낸스만"} · ${sg(OI.d30m_pct, 2, "%")} · 풋프린트 아래 «신규계약» 줄과 같은 데이터(30분 변화)`, sg(OI.z30m, 1, "σ"), tone(OI.z30m, 1.5)))
      + tile("q_flow", "② 누가 밀고 있나", duelChip,
          gk("고래", G.sig(z.whale, gv), `고래 30분 ${sg(z.whale, 1, "σ")} · ${flowVen}`, sg(z.whale, 1, "σ"), tone(z.whale))
          + gk("중형", G.sig(z.mid, gv), `중형 30분 ${sg(z.mid, 1, "σ")}`, sg(z.mid, 1, "σ"), tone(z.mid))
          + gk("리테일", G.sig(z.retail, gv), `리테일 30분 ${sg(z.retail, 1, "σ")}`, sg(z.retail, 1, "σ"), tone(z.retail))
          // 2026-10-22 아티팩트 펜 메모 «이 텍스트를 제거해줘»: 줄 이름 «가격×OI» 글자 빼고(표시등 네 칸 이름이 이미 말한다) 이름은 툴팁 첫머리로
          + gk("", G.seg4(d.quad ? d.quad.key : null, gv), `가격×OI 1시간 · 네 칸 = 신규 롱 · 숏커버 · 신규 숏 · 롱정리 (초록 = 가격↑ · 빨강 = 가격↓ · 다 비면 보류)\n` + (d.quad ? `지금 ${({ up_up: "새 롱", up_dn: "숏 커버", dn_up: "새 숏", dn_dn: "롱 정리", small: "보류" })[d.quad.key] || "-"} · ${d.quad.label} · 1시간 ${sg(d.move60, 0, "bp")}` : "가격×OI 1시간 -"), ""))
      + tile("q_wall", "④ 벽 · 교차 · 위치", wallChip,
          (sw ? G.book(sw, bps, Math.floor(unit * 1.6), Number(d.mid) || 0, (ASSET_CONFIG[A] || {}).dp ?? 2, 30, 11.5, 16) : "")   // 10-13 «벽 내용 상하 간격 넓히고 크게»
          + gk("깊은 벽", ps && PAPER_SIG_OK.wall.has(A) ? G.wall(ps.imb, ps.thr, gwWall) : G.pct(null, gwWall), ps ? `깊은 벽 ${sg(ps.imb, 2)} · 문턱 ±${ps.thr.toFixed(2)}` : ""), " mc-wide-t")
      + `</div>` + fcHtml];
    // 2026-10-28 사용자 «풋프린트 상단 나스닥/2년물/10년물 차트 자리에 Macro Trend 요약»(시안 H1) -- 나스닥·10년은 Macro Trend 카드 «오늘» 칸으로 옮겼다.
    //   급변 칩(5분 |z|≥4, 10-14)은 그대로 끝에 둔다.
    setH("fpRates", mtHeadHtml() + `<span class="mc-spk">${macroSpikeChips(chip)}</span>`);
    el("fpRates") && (el("fpRates").hidden = false);
  } else { htmlB.push(wallHtml0 + fcHtml); ratesOff(); }
  // 2026-10-13 좁은 화면 시장 맥락 끝 모의 판도 뺐다(사용자 «모의판은 제거»)
  htmlB = htmlB.join("");
  // 2026-09-30 ⑤ 다음 24시간은 좁은 칸(mc-cmp)이면 풋프린트 차트 **아래 전폭**(#mcWhen, 사용자 지시) -- 아니면 판 넷 아래 전폭 그대로.
  const whenBox = el("mcWhen");   // 2026-10-01 휴대폰·세로 화면도 Option 카드 맨 아래(넓은 화면과 같게)
  // 2026-09-30 폭: #mcWhen 은 비어 있으면 숨김(display:none)이라 첫 그림 때 clientWidth 0 → 900 으로 그려 2.3배 늘어났다(글자가 커졌다 작아짐) -- 카드 폭에서 잰다
  const whenW = whenBox ? (whenBox.clientWidth || el("newsBody")?.querySelector(".nw-main")?.clientWidth || 700) : body.clientWidth;   // 2026-10-07 ⑤ 는 뉴스 목록 아래(같은 폭)
  // 2026-10-05 «max pain 규칙 창»·«경제 일정 갱신» 두 줄 제거(사용자 지시) -- 시간축만
  const whenHtml = qSec("when", "⑤ 다음 24시간", G.timeline(events, Math.max(320, Math.round((whenW || 900) - 8))));
  setH("mcWhen", whenBox ? whenHtml : "");
  nwWhenFit();
  if (!whenBox) htmlB += `<div class="mc-wide">${whenHtml}</div>`;
  if (body._mcHtml === htmlB) return;     // 같은 내용이면 다시 안 그린다(포커스 유지)
  body._mcHtml = htmlB;
  keepFocus(body, () => { body.innerHTML = htmlB; });   // 2026-10-28 펜 메모 «가격 자 제거 · 그 자리에 수급 차트 2개를 키워» -- 맨 위 «지금 들어가면» 가격 자(#mcEntry) 삭제
  if (body.classList.contains("mc-cmp")) {   // 내용 높이(칸 높이가 아니라 자식들의 아래 끝) -- 바뀌면 차트를 다시 그려 1초 수급 몫을 조정
    const top = body.getBoundingClientRect().top;
    const need = Math.ceil(Math.max(0, ...[...body.children].map((k) => k.getBoundingClientRect().bottom - top)) + 6);
    if (need > 40 && Math.abs(need - mcNeedH) > 4) {
      mcNeedH = need; if (typeof scheduleSnapshotChartRender === "function") scheduleSnapshotChartRender();
    }
  }
}

// 2026-10-26 사용자 «? 를 누르면 칸이 늘어나는 게 아니라 팝업창 · 대시보드 전체»: «?»(.mc-q 시장 맥락 · .opt-q 옵션) = 공유 설명 팝업 하나.
//   <dialog> 는 body 끝(매초 다시 그리는 칸 밖)이라 다시 그려도 안 닫히고 포커스도 안 뺏긴다 · 제목 = 단추 title(«— 설명» 뺀) 또는 글자 · 본문 = 같은 설명(줄바꿈 유지)
//   ✕·Esc·바깥 클릭(= ::backdrop 클릭은 dialog 자신이 받는다 -- 그래서 dialog 안쪽 여백 0)으로 닫고, 포커스는 «?» 로(그새 다시 그려졌으면 같은 data-tip 단추로).
function helpOpen(b) {
  let d = el("helpDlg");
  if (!d) {
    d = document.createElement("dialog"); d.id = "helpDlg"; d.className = "help-dlg"; d.setAttribute("aria-labelledby", "helpDlgT");
    d.innerHTML = `<div class="help-head"><h3 id="helpDlgT"></h3><button type="button" class="help-x" aria-label="닫기">`
      + `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"/></svg></button></div><p class="help-body"></p>`;
    d.addEventListener("click", (e) => { if (e.target === d || e.target.closest(".help-x")) d.close(); });
    d.addEventListener("close", () => {
      const btns = [...document.querySelectorAll(`.mc-q[data-tip="${helpKey}"], .opt-q[data-tip="${helpKey}"], .mt-q[data-tip="${helpKey}"]`)];
      helpKey = null;
      btns.forEach((x) => x.setAttribute("aria-expanded", "false"));
      btns.find((x) => x.getClientRects().length)?.focus({ preventScroll: true });
    });
    document.body.appendChild(d);
  }
  const k = b.dataset.tip, opt = b.classList.contains("opt-q");
  d.classList.toggle("opt", opt);
  d.querySelector("h3").textContent = (b.title ? b.title.replace(/ — 설명$/, "") : b.textContent.replace(/\?$/, "")).trim();   // 글자 끝 «?» = 단추 원(«① 레버리지 과열?» 의 물음표는 남긴다)
  d.querySelector(".help-body").textContent = (opt ? OPT_TIPS[k] : b.classList.contains("mt-q") ? MT_TIPS[k] : mcTipOf(k)) || "";
  helpKey = k; b.setAttribute("aria-expanded", "true");
  d.style.left = d.style.top = "";
  d.showModal();
  d.scrollTop = 0;
  if (window.matchMedia("(max-width: 720px)").matches) return;   // 휴대폰 = 아래 시트(CSS)
  const r = b.getBoundingClientRect(), w = d.offsetWidth, h = d.offsetHeight;   // 데스크톱 = «?» 바로 아래(모자라면 위), 화면 안으로
  d.style.left = `${Math.round(Math.max(8, Math.min(r.left, innerWidth - w - 8)))}px`;
  d.style.top = `${Math.round(r.bottom + 6 + h <= innerHeight - 8 ? r.bottom + 6 : Math.max(8, r.top - 6 - h))}px`;
}
document.addEventListener("click", (e) => { const b = e.target.closest(".mc-q, .opt-q, .mt-q"); if (b) helpOpen(b); });
// 2026-10-28 지표 팝업 열고 닫기(버튼 · 밖을 누르면 · Esc) -- 열림 상태는 fpIndOpen 이 들고 다시 그려도 유지
let fpIndOpen = false;
function fpIndSet(open) {
  fpIndOpen = open;
  const fs = el("fpLineSwitch"), pop = fs?.querySelector(".fp-ind-pop"), btn = fs?.querySelector(".fp-ind-btn");
  if (pop) pop.hidden = !open;
  if (btn) btn.setAttribute("aria-expanded", String(open));
}
el("fpLineSwitch")?.addEventListener("click", (e) => { if (e.target.closest(".fp-ind-btn")) fpIndSet(!fpIndOpen); });
document.addEventListener("pointerdown", (e) => { if (fpIndOpen && !e.target.closest("#fpLineSwitch")) fpIndSet(false); });
document.addEventListener("keydown", (e) => {
  if (!fpIndOpen || e.key !== "Escape") return;
  const inside = !!document.activeElement?.closest("#fpLineSwitch");
  fpIndSet(false);
  if (inside) el("fpLineSwitch")?.querySelector(".fp-ind-btn")?.focus();   // 팝업 안에 있던 포커스만 버튼으로 돌린다
});
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
  if (!t || !t.ok || !candles.length) return [];
  const lo = Math.min(...candles.map((c) => c.low)), hi = Math.max(...candles.map((c) => c.high));
  return t.votes.filter((v) => Number(v.L) !== 7 && v.flip_price >= lo && v.flip_price <= hi).map((v) => ({   // 2026-10-01 7일선 제거(사용자 지시) -- 14~90일은 범위 안에 오면 그대로
    val: Number(v.flip_price), color: "var(--muted)", label: `추세${v.L}일`, priceLeft: true,
    dashed: true, width: 1, marker: !!footprint }));
}

// 2026-10-15 사용자 «풋프린트 카드의 추세와 예측 칩 제거»: 머리 칩(#sitChips·renderSituation)을 지웠다 -- 같은 값은 시장 맥락 ⑤ 예측 줄에 있다(latestTrend·latestDirSig).
// 2026-10-14 아티팩트 댓글 «풋프린트 상단이 청산맵·체결 버블 상단과 크기가 안 맞아 -- 통일»: 넓은 화면 풋프린트의 그래프 상단을 그 둘(캔버스 위 여백 22)과
//   같은 높이로 내리고(FP_TOP_ALIGN), 청산 밀도 범례를 .lqp-legend 와 같은 11px·카드 왼쪽 17px 로 둔다.
const FP_TOP_ALIGN = 26;
// 2026-10-15 아티팩트 댓글 «풋프린트를 크기 그대로 왼쪽 ‹청산 밀도› 글자 위치까지 옮기고 그만큼 오른쪽 정보를 크게»: 넓은 화면 왼쪽 여백 36 → 1, 오른쪽 칸 +35.
const FP_LEFT_SHIFT = 35;

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

let footprintInFlight = false;   // 터널이 멎으면 400ms 폴링이 쌓였다가 순서 없이 풀려 옛 응답(전량 포함)이 형성 봉을 되돌렸다
async function refreshFootprint() {
  if (!flowOn()) return;   // 흐름 엔진이 있는 코인만
  const now = Date.now();
  if (footprintInFlight || now - footprintLastFetchAt < FOOTPRINT_POLL_MS) return;
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
  footprintInFlight = true;
  try {
    const res = await fetch(`${API_FOOTPRINT_URL}?asset=${activeSnapshotAsset}&bars=${CHART_PAN_BARS}&since=${since}`,
                            { cache: "no-cache" });
    if (!res.ok) throw new Error(`footprint ${res.status}`);
    const payload = await res.json();
    if (`${activeSnapshotAsset}|${CHART_PAN_BARS}` !== key) return;   // 그 사이 코인·창이 바뀌었다
    if (payload.full) footprintBars = new Map();
    (payload.bars || []).forEach((b) => footprintBars.set(b.time, b));
    // 창 밖으로 밀려난 봉은 버린다 -- 증분이라 서버가 «빠졌다»를 말해 주지 않는다.
    //   1주일을 받아 뒀으면 거기로 옮긴다(2026-10-20) -- 화면을 오래 열어 둬도 1주일과 라이브 12시간 사이가 안 빈다.
    const wk = fpWeekFor();
    if (footprintBars.size > CHART_PAN_BARS) {
      [...footprintBars.keys()].sort((a, b) => a - b)
        .slice(0, footprintBars.size - CHART_PAN_BARS)
        .forEach((t) => { if (wk) wk.bars.set(t, footprintBars.get(t)); footprintBars.delete(t); });
    }
    // bars = 라이브 144봉만(시각순). 1주일 몫은 쓰는 쪽(fpBarsNow)이 필요할 때만 붙인다 -- 리뷰: 폴링(400ms)마다 2천 봉을 복사·정렬했다.
    latestFootprint = { ...payload, bars: [...footprintBars.values()].sort((a, b) => a.time - b.time) };
  } catch (error) {
    console.error("Footprint fetch error:", error);
    // 🔴2026-09-22 전에는 여기서 latestFootprint = null 이었다 -- **한 번의 실패로 화면이
    //   비었다**(400ms 폴링이라 그게 곧 깜빡임이다). 캐시는 이미 안 버리고 있었으므로
    //   직전 값을 그대로 둔다. 진짜로 정체되면 위 «전량 재수신» 자가복구가 잡는다.
    // 🔴캐시는 **안 버린다**. 한 번의 네트워크 실패로 12.5KB 를 다시 받을 이유가 없다 --
    //   다음 성공 폴링이 꼬리 두 봉만 얹으면 그대로 이어진다.
  } finally {
    footprintInFlight = false;
  }
  scheduleSnapshotChartRender();
}

// ── 1주일 (2026-10-20 사용자 «5m, 30m, 1h 모두 데이터를 일주일 치를 가져와서 보여줘») ───────────────────────
// 지난 7일 5분봉 캔들·칸·OI·청산을 /api/footprint-week 로 **한 번** 받는다(~0.6MB gzip -- 400ms 폴링에 안 싣는다).
//   받는 때 = 라이브 12시간 밖이 필요해질 때(휠로 넓히기 · 옮기기 · 30m/1h). 처음 열면 안 받는다 -- 대부분 최신 1시간만 본다.
//   같은 봉이면 라이브 값이 이긴다(weekJoin · fpBarsWithWeek). 6시간이 지나면 다시 받는다(라이브 12시간과 이어지게).
//   서버 원천 = 우리 수집기 테이프·OI·청산(캔들만 바이낸스 5분봉 -- 차트 캔들 원천 그대로).
let fpWeek = null, fpWeekAt = 0, fpWeekTriedAt = 0, fpWeekInFlight = null;   // fpWeekInFlight = 받는 중인 코인(리뷰: 옛 코인 요청이 새 코인을 막았다)
const FP_WEEK_REFRESH_MS = 6 * 3600e3;
const fpWeekFor = () => (fpWeek && fpWeek.asset === activeSnapshotAsset ? fpWeek : null);
const fpWantWeek = () => chartTfK > 1 || fpZoomed() || chartPanEnd != null;   // 라이브 12시간 밖이 필요한 보기
async function maybeFetchFpWeek() {
  if (!flowOn() || fpWeekInFlight === activeSnapshotAsset || !fpWantWeek()) return;
  const now = Date.now(), asset = activeSnapshotAsset;
  if ((fpWeekFor() && now - fpWeekAt < FP_WEEK_REFRESH_MS) || now - fpWeekTriedAt < 30000) return;   // 실패면 30초 뒤 다시
  fpWeekInFlight = asset; fpWeekTriedAt = now;
  try {
    const res = await fetch(`/api/footprint-week?asset=${asset}`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`footprint-week ${res.status}`);
    const j = await res.json();
    if (asset !== activeSnapshotAsset) return;
    // 리뷰: 서버가 바이낸스 캔들을 못 받으면 candles: [] 를 준다 -- 성공으로 적으면 6시간 안 다시 받는다. 실패로 치고 30초 뒤 다시.
    //   서버가 이 빈 몸을 300초 캐시한다(fp_week_{asset}) -- 30초마다 0.5MB 를 받아 버리지 않게 300초 뒤 다시(리뷰).
    if (!(j.candles && j.candles.length)) { fpWeekTriedAt = Date.now() + 270e3; throw new Error("footprint-week: 캔들 없음"); }
    fpWeek = { asset, candles: j.candles, pre: j.volPre || [], bars: new Map((j.bars || []).map((x) => [x.time, x])), oi: j.oi || [], liq: j.liq || [] };
    fpWeekAt = Date.now();
    scheduleSnapshotChartRender();
  } catch (e) {
    console.error("footprint week:", e);
  } finally {
    if (fpWeekInFlight === asset) fpWeekInFlight = null;
  }
}
// ── 5분 호가 요약 (2026-10-26 사용자 «호가 5분 지표 연구 세션의 차트처럼 풋프린트 안에 호가 정보 · 거래대금 자리에 기울기 막대») ──
// /api/ob5m 한 번에 지난 1주일: bars[{t ms, tilt_pct, book_ok, events_known, live?}] · walls[{side, px, t_start, t_end}] · marks[{bar_t, side, kind, live?}].
//   live = 진행 중 봉(지금까지 초들의 평균) · 임시 판정 표식(테이프가 온 데까지) -- 흐리게 그리고 봉 끝(+70초)에 확정값으로 바뀐다.
//   정의 = scripts/research_orderbook_5m_indicator_20261026.py(호가 5분 지표 연구 세션) 그대로 -- 🔴정의 확정 전(4코인 검증 중), 정보 표시용(예측 아님).
//   기울기 = ±50bp 매수·매도 잔량 (b−a)/(b+a)×100 의 봉 안 1초 평균 · 벽 = 쪽별 ±50bp 깊이 5%+ 한 칸 중 그 봉에서 가장 오래 선 것 ·
//   ▲▼ = 30초+ 선 5%+ 벽이 체결 80%+ 로 사라짐(돌파) · ◆ = 체결 20% 밑·현재가 5bp 안에서 사라짐(가짜벽 의심).
let ob5m = null, ob5mAt = 0, ob5mKey = null;
async function refreshOb5m() {
  const asset = activeSnapshotAsset, now = Date.now();
  if (!flowOn() || (ob5mKey === asset && now - ob5mAt < 5000)) return;   // 코인마다 5초(실패도) -- 10-26 진행 중 봉(워커 5초마다) · 전 15초 · 코인을 바꾸면 곧바로 -- 10-26 «기울기 막대가 늦다»: 60초 → 15초(워커가 봉 끝에 북 값을 바로 쓴다)
  ob5mKey = asset; ob5mAt = now;
  try {
    const res = await fetch(`/api/ob5m?asset=${asset}`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`ob5m ${res.status}`);
    const j = await res.json();
    if (asset !== activeSnapshotAsset) return;
    const marks = new Map();
    (j.marks || []).forEach((m) => { const k = m.bar_t / 1000; if (!marks.has(k)) marks.set(k, []); marks.get(k).push(m); });
    ob5m = { asset, bars: new Map((j.bars || []).map((b) => [b.t / 1000, b])), walls: j.walls || [], marks };
    scheduleSnapshotChartRender();
  } catch (e) {
    if (ob5m && ob5m.asset !== asset) ob5m = null;
  }
}
const ob5mFor = () => (ob5m && ob5m.asset === activeSnapshotAsset ? ob5m : null);
// 지금 tf 봉 시각 -> {tilt(평균, 북 없는 5분봉 뺌), up/dn(돌파), fu/fd(가짜벽 위/아래)} -- 렌더마다 1주일(2천 개)을 한 번 돈다
function ob5mByTf(OB) {
  const m = new Map(), at = (T) => m.get(T) || m.set(T, { s: 0, n: 0, up: 0, dn: 0, fu: 0, fd: 0, pup: 0, pdn: 0, pfu: 0, pfd: 0, live: false }).get(T);
  OB.bars.forEach((b, t) => { if (b.book_ok && b.tilt_pct != null) { const a = at(tfT(t)); a.s += b.tilt_pct; a.n++; if (b.live) a.live = true; } });
  // 임시 판정(진행 중·확정 전 봉)은 p 붙은 칸에 따로 센다 -- 15분·1시간 봉에 확정과 임시가 같이 있어도 확정 표식은 진하게(리뷰 10-26)
  OB.marks.forEach((ms, t) => ms.forEach((x) => { const a = at(tfT(t)); a[(x.live ? "p" : "") + (x.kind === "eat" ? (x.side === "ask" ? "up" : "dn") : (x.side === "ask" ? "fu" : "fd"))]++; }));
  m.forEach((a) => { a.tilt = a.n ? a.s / a.n : null; });
  return m;
}

// 풋프린트 봉(시각순) -- 1주일이 필요한 보기면 앞에 1주일 몫(라이브에 없는 봉)을 붙인다. 폴링·1주일 도착마다 한 번만 만든다.
let fpBarsMemo = null;
function fpBarsNow() {
  const lf = latestFootprint, wk = fpWantWeek() ? fpWeekFor() : null;
  if (!lf || !Array.isArray(lf.bars) || !wk) return lf && lf.bars;
  if (fpBarsMemo && fpBarsMemo.lf === lf && fpBarsMemo.wk === wk) return fpBarsMemo.out;
  const out = lf.bars.slice();
  wk.bars.forEach((b, t) => { if (!footprintBars.has(t)) out.push(b); });
  fpBarsMemo = { lf, wk, out: out.sort((a, b) => a.time - b.time) };
  return fpBarsMemo.out;
}
// 라이브 배열 앞에 1주일 몫(라이브 첫 시각보다 앞)만 붙인다. 입력이 그대로면 **같은 배열**을 준다(끌기 한 프레임에 여러 번 불린다 -- 2천 개 복사 안 하게).
//   라이브 배열은 제자리에서 자란다(push → 200 개 넘으면 shift -- 신원·길이가 그대로) -- 그래서 마지막 원소도 본다(리뷰: 새 5분봉이 안 붙었다).
const weekJoinMemo = {};
function weekJoin(name, live, tOf) {
  const wk = fpWeekFor(), old = wk && wk[name];
  if (!old || !old.length) return live;
  const m = weekJoinMemo[name], z = live[live.length - 1];
  if (m && m.old === old && m.live === live && m.n === live.length && m.z === z) return m.out;
  const t0 = live.length ? tOf(live[0]) : Infinity;
  const out = old.filter((x) => tOf(x) < t0).concat(live);
  weekJoinMemo[name] = { old, live, n: live.length, z, out };
  return out;
}
const chartFull = () => weekJoin("candles", candleHistoryByAsset[activeSnapshotAsset] || [], (c) => c.time);   // 차트 캔들(5분봉) -- 창 이동·미니맵·렌더가 같이 쓴다
const chartTfFull = () => tfCandles(chartFull());

// 풋프린트가 없으면(다른 코인 · 서버 웜업 · fetch 실패) null 을 돌려주고, 차트는 캔들로 그린다.
function footprintForChart() {
  if (!flowOn()) return null;
  const payload = latestFootprint, all = fpBarsNow();
  const bars = Array.isArray(all)
    ? all.filter((b) => Array.isArray(b.levels) && b.levels.length) : [];
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

// ── 30m·1h 묶기 (2026-10-20) -- chartTfK === 1 이면 전부 입력을 그대로 돌려준다(5m 은 한 줄도 안 바뀐다) ──
// 캔들: 시가 = 첫 봉 · 고가 = 최대 · 저가 = 최소 · 거래량 = 합 · 나머지(종가·sma·vwap 등) = 마지막 봉(그 시각의 값). 형성 tf 봉 = 라이브 5분봉 포함.
//   맨 앞 tf 봉이 덜 찼으면(1주일·라이브 첫 봉이 시간 중간부터) 뺀다 -- 시가가 거짓이 된다(리뷰).
//   기억(리뷰: 끌기 한 프레임에 3~4번 2천 봉을 다시 묶었다): 입력 배열·길이·첫/끝 원소가 그대로면 같은 배열을 주고
//   형성 tf 봉만 **제자리에서** 다시 묶는다(라이브 5분봉이 체결마다 제자리 갱신 -- 렌더러·fpCv.live 가 같은 객체를 쥐고 있다).
const tfCMemo = {};
function tfCandles(full) {
  if (chartTfK === 1) return full;
  const m = tfCMemo, n = full.length;
  if (m.src === full && m.n === n && m.k === chartTfK && m.a === full[0] && m.z === full[n - 1]) {
    const a = m.out[m.out.length - 1];
    if (a) {
      let j = n, hi = -Infinity, lo = Infinity, vol = 0;
      while (j > 0 && tfT(full[j - 1].time) === a.time) { j--; hi = Math.max(hi, full[j].high); lo = Math.min(lo, full[j].low); vol += +full[j].volume || 0; }
      if (j < n) Object.assign(a, full[n - 1], { time: a.time, open: full[j].open, high: hi, low: lo, volume: vol });
    }
    return m.out;
  }
  const out = [];
  let n0 = 0;   // 맨 앞 tf 봉에 든 5분봉 수
  full.forEach((c) => {
    const t = tfT(c.time), a = out[out.length - 1];
    if (a && a.time === t) Object.assign(a, c, { time: t, open: a.open, high: Math.max(a.high, c.high), low: Math.min(a.low, c.low), volume: (+a.volume || 0) + (+c.volume || 0) });
    else out.push(Object.assign({}, c, { time: t, volume: +c.volume || 0 }));
    if (out.length === 1) n0++;
  });
  if (out.length > 1 && n0 < chartTfK) out.shift();
  Object.assign(m, { src: full, n, k: chartTfK, a: full[0], z: full[n - 1], out });
  return out;
}
// 풋프린트 칸: 같은 가격 행의 7열(가격·매수·매도·고래 매수/매도·리테일 매수/매도)을 5분봉끼리 더한다.
//   5분봉이 모자란 tf 봉(빠진 봉 · 1주일 첫 묶음)은 «모름»으로 뺀다 -- 덜 더한 값을 온전한 봉처럼 보이지 않게(리뷰). 형성 중인 마지막 tf 봉만 예외.
//   ponytail: «마지막» = byTime 의 마지막 tf 봉. 끝 봉들이 통째로 빠지면 그 앞 덜 찬 봉이 형성 봉 대접을 받는다 -- 드물어서 둔다.
//   닫힌 tf 봉은 기억해 둔 배열을 돌려준다(폴링마다 같은 5분봉을 다시 더하지 않게 -- 진행 봉만 매번 새로).
const tfLvMemo = new Map();
function tfLevels(byTime, bucket) {
  const groups = new Map(), out = new Map();
  let lastT = -Infinity;
  [...byTime.keys()].sort((a, b) => a - b).forEach((t) => {
    const T = tfT(t);
    if (!groups.has(T)) groups.set(T, []);
    groups.get(T).push(byTime.get(t));
    lastT = T;
  });
  groups.forEach((refs, T) => {
    if (refs.length < chartTfK && T !== lastT) return;
    const m = tfLvMemo.get(T);
    if (m && m.k === chartTfK && m.refs.length === refs.length && m.refs.every((r, i) => r === refs[i])) { out.set(T, m.lv); return; }
    const sum = new Map();
    refs.forEach((lv) => (lv || []).forEach((l) => {
      const key = Math.round(l[0] / bucket);
      if (!sum.has(key)) sum.set(key, [Number(l[0]), 0, 0, 0, 0, 0, 0]);
      const a = sum.get(key);
      for (let j = 1; j < 7; j++) a[j] += Number(l[j]) || 0;
    }));
    const lv = [...sum.values()].sort((a, b) => a[0] - b[0]);
    tfLvMemo.set(T, { k: chartTfK, refs, lv });
    out.set(T, lv);
  });
  if (tfLvMemo.size > 400) tfLvMemo.clear();   // ponytail: 통째로 비움(코인·tf 를 오가면 쌓인다) -- 다음 렌더가 다시 채운다
  return out;
}
function tfFootprint(fp) {
  return fp && chartTfK > 1 ? Object.assign({}, fp, { byTime: tfLevels(fp.byTime, fp.bucket) }) : fp;
}
// 레인 입력은 원본 배열 신원으로 기억한다(입력은 폴링 때만 새 배열 -- 렌더마다 2천 봉을 다시 묶지 않게).
const tfMemo = (fn) => { let src, k, out; return (x) => (chartTfK === 1 ? x : x === src && k === chartTfK ? out : (src = x, k = chartTfK, out = fn(x))); };
const tfFpBars = tfMemo((bars) => [...tfLevels(new Map(bars.filter((b) => Array.isArray(b.levels)).map((b) => [b.time, b.levels])),
  Number(latestFootprint && latestFootprint.bucket) || 0.5)].map(([time, levels]) => ({ time, levels })));
const tfOi = tfMemo((bars) => {   // [시각, ΔOI] -- tf 봉 ΔOI = 그 안 5분 Δ 합(= 봉 끝 OI − 직전 봉 끝 OI) · 5분 Δ 가 모자란 tf 봉은 뺀다(= OI 모름, tfLevels 와 같은 규칙)
  const m = new Map();
  bars.forEach((b) => { const T = tfT(Number(b[0])), a = m.get(T) || [0, 0]; a[0] += Number(b[1]) || 0; a[1]++; m.set(T, a); });
  const all = [...m];
  return all.filter(([, a], i) => a[1] >= chartTfK || i === all.length - 1).map(([T, a]) => [T, a[0]]);
});
const tfLiq = tfMemo((bars) => {   // 청산: 금액·건수 합(거래소별 몫도) · 진행 중 = 하나라도 진행 중
  const m = new Map();
  bars.forEach((b) => {
    const T = tfT(Math.floor(Date.parse(b.ts) / 1000));
    if (!Number.isFinite(T)) return;
    if (!m.has(T)) m.set(T, { ts: new Date(T * 1000).toISOString(), long_usd: 0, short_usd: 0, events: 0 });
    const a = m.get(T);
    a.long_usd += Number(b.long_usd) || 0; a.short_usd += Number(b.short_usd) || 0; a.events += Number(b.events) || 0;
    a.partial = a.partial || !!b.partial; a.okx = a.okx || !!b.okx;
    ["okx_detail", "bybit", "hl"].forEach((v) => {
      if (!b[v]) return;
      const x = a[v] || (a[v] = { long_usd: 0, short_usd: 0, n: 0 });
      x.long_usd += Number(b[v].long_usd) || 0; x.short_usd += Number(b[v].short_usd) || 0; x.n += Number(b[v].n) || 0;
    });
  });
  return [...m.values()];
});

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
function renderSupply1s(box) {
  const S = mergedSupplySrc();
  const svg = box.svg;
  if (!svg) return;
  const NS = "http://www.w3.org/2000/svg";
  const w = box.w;
  const h = box.h;
  // 2026-09-22 모바일(사용자 «프로파일처럼 라벨을 차트 아래에 일자로»): 범례가 플롯 안에서
  //   바닥 한 줄로 내려온다. 시각 꼬리표(h-3)와 같은 줄에 둘 수 없으니 그 아래 한 줄을 더 판다.
  //   400px 패널에서 14px 은 선이 3.5% 짧아지는 대가다.
  // 🔴narrow 를 **먼저** 선언한다. 아래 mb 가 이걸 읽는데 순서를 바꾸면 TDZ
  //   ReferenceError 로 app.js 전체가 죽는다 -- node --check 는 못 잡는다(문법은 옳다).
  const narrow = w < 760;   // 모바일이든 좁은 상자든 여백 규칙은 같다
  // 2026-09-28 모든 폭에서 범례는 플롯 **안 박스 셋**(사용자 «거래소 3 · 크기별 3 · 신규 2»)이고 위 제목 줄도 없앴다 --
  //   mt 는 위 여백만, mb 는 바닥 시각 꼬리표(h-3) 한 줄이다. (09-27 모바일 박스가 모든 폭으로 넓어진 것)
  // 2026-10-15 아티팩트 댓글 «봉 시작·끝 시간은 수급 차트 위로, 남는 공간은 미니 체결 버블»: 시각 꼬리표가 첫 줄(CVD · OI) 머리 띠로 올라가 바닥 줄(16)이 빈다 → 4
  const mt = 6, mb = 4;
  // mr 은 오른쪽 꼬리표(고래/리테일/신규계약 + 값)가 앉는 자리다. 64 면 «신규계약 +0.4k»
  // 가 약 4px 넘친다(2026-09-19 계산) -- 72 로 두면 셋 다 들어가고 선은 8px 만 짧아진다.
  // 2026-09-22 꼬리표 글자를 키우면서 자리도 넓혔다(사용자 «많이 키워줘»).
  //   112 -> 150 은 선이 38px(플롯 폭의 3%) 짧아지는 대가다. 글자가 잘리는 것보다 낫다.
  // 2026-09-22 모바일(사용자 «라벨들을 차트 안으로»): mr 은 «여백»이 아니라 **그림이 끝나는
  //   자리**다. 88 을 비워 두면 348px 화면에서 선이 260px 로 눌린다(실측). 8 로 줄이면 같은
  //   화면에서 플롯이 344px = **+32%** 가 되고, 범례는 아래에서 플롯 위로 얹는다.
  //   🔴데스크톱은 그대로 둔다 -- 폭이 남는 화면에서 글자를 데이터 위에 올릴 이유가 없다.
  // 2026-09-28 다섯 줄 거울 막대 -- 왼쪽 칸에 줄 이름·누적값(범례 박스를 대체), 오른쪽은 얇은 여백만.
  // 2026-10-10 휴대폰(사용자 «모바일 최적화»): 왼쪽 칸(100px)이 348px 폭의 1/3 이라 선이 오른쪽 2/3 에 눌렸다 --
  //   좁은 폭에서는 견본·값을 줄 머리 띠(이름 아래 한 줄)로 올리고 선이 전체 폭을 쓴다.
  const compact = narrow && isMobileChartMode();   // 넓은 화면 2단의 오른쪽 칸(좁은 상자)은 그대로
  const ml = compact ? 8 : 100, mr = 8;   // 2026-09-30 두 줄 선 차트: 왼쪽 칸 = 선 견본 + 값(«선물 +2.4k»), 줄 이름은 줄 위 머리 띠
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
  const HEAD = compact ? 38 : 18;   // 줄 이름 머리 띠 -- 이름(«고래 · 중형 · 리테일»)이 왼쪽 칸보다 길어 선과 겹쳤다(09-30 1920 실측) · 휴대폰은 견본 줄까지
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
    if (compact) {   // 이름 아래 한 줄: 견본 + 값을 가로로(넘치면 거기서 멈춘다 -- 잘린 글자를 남기지 않는다)
      let x = 6;
      items.forEach(([txt, col, sw]) => {
        if (x > w - 24) return;
        const yy = y0 + 31;
        let sm = null;
        if (sw) {
          sm = document.createElementNS(NS, "line");
          sm.setAttribute("x1", x); sm.setAttribute("x2", x + 12); sm.setAttribute("y1", yy - fs / 3); sm.setAttribute("y2", yy - fs / 3);
          sm.setAttribute("stroke", sw.color); sm.setAttribute("stroke-width", Math.min(2.4, sw.width));
          sm.setAttribute("stroke-opacity", sw.op); if (sw.dash) sm.setAttribute("stroke-dasharray", sw.dash);
          svg.appendChild(sm);
        }
        const t = label(x + (sw ? 16 : 0), yy, txt, col, null, fs);
        t.setAttribute("font-weight", "700");
        let tw = 0;
        try { tw = t.getComputedTextLength(); } catch (e) { /* 비렌더 */ }
        if (!(tw > 0)) tw = txt.length * fs * 0.62;
        const end = x + (sw ? 16 : 0) + tw;
        if (end > w - 4) { t.remove(); if (sm) sm.remove(); x = w; return; }
        x = end + 12;
      });
      return { y0, zc, hh: (laneH - HEAD) / 2 - 4 };
    }
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
  const spotRows = [...spotSupply1s.keys()].filter((x) => x >= first && x <= now).sort((a, b) => a - b)
    .map((x) => { const c = spotSupply1s.get(x); spotAcc += (c[4] || 0) - (c[5] || 0); return { s: x, v: spotAcc }; });
  const spotOn = !!(spotRows && spotRows.length >= 2);
  const oiEnd = oiRows.length ? oiRows[oiRows.length - 1].v : null;
  {
    const items = [["선물 " + sgn(acc), "var(--ink)", { color: "var(--ink)", width: 1.8, op: 0.9 }]];
    if (spotOn) items.push(["현물 " + sgn(spotAcc), "var(--spot)", { color: "var(--spot)", width: 1.6, op: 0.95 }]);
    items.push(["OI " + (oiEnd == null ? "—" : sgn(oiEnd)), "var(--warn)", { color: "var(--warn)", width: 1.8, op: 0.95 }]);
    const { y0: cvdY0, zc, hh } = laneFrame(0, "CVD · OI", items);
    let headEnd = 6 + measureTextW("CVD · OI", `700 ${narrow ? 12 : 13}px Pretendard Variable, Pretendard, 'Noto Sans KR', sans-serif`) + 10;   // 머리 띠의 시각 꼬리표는 이름(·청산 급증 배지) 뒤부터
    { // 2026-10-04 청산 급증 배지(사용자 지시 -- 계좌 카드 말고 이 차트). 판정 = 시장 맥락과 같은 mcLiqBurstSide.
      const bu = latestMarketCtx && latestMarketCtx.burst, side = activeSnapshotAsset === "eth" ? mcLiqBurstSide(bu) : null;
      if (side) {
        const fsz = narrow ? 10.5 : 11, ff = "Pretendard Variable, Pretendard, 'Noto Sans KR', sans-serif";
        const t1 = `${side === "long" ? "롱" : "숏"} 청산 급증 ${fmtUsdCompact((side === "long" ? bu.long_usd_60s : bu.short_usd_60s) || 0)}/60초`, t2 = " · 시장가 청산 주의";
        const bx = 6 + measureTextW("CVD · OI", `700 ${narrow ? 12 : 13}px ${ff}`) + 10, bw = measureTextW(t1, `700 ${fsz}px ${ff}`) + measureTextW(t2, `600 ${fsz}px ${ff}`) + 18, bh = fsz + 9;
        const g = document.createElementNS(NS, "g"), r = document.createElementNS(NS, "rect"), tt = document.createElementNS(NS, "title");
        // 2026-10-04 문구를 근거만큼만(사용자 «청산 급증 배지 문제도 고쳐줘»): «대개 되돌아온다»는 일반 시점에서 약하다(15분 +6.6bp, CI 하한 +0.03)
        tt.textContent = `직전 60초 ${side === "long" ? "롱" : "숏"} 청산이 큰 버스트(롱 ${fmtUsdCompact(bu.thr[0])} · 숏 ${fmtUsdCompact(bu.thr[1])} 초과, 학습 7일 상위 0.5%)입니다.\n`
          + "이런 버스트를 만난 보유 포지션을 그 순간 시장가로 닫으면 평균 −17bp 손해였습니다(35건, 95% CI −24~−11, 수수료 포함 모의 원장).\n"
          + "그 뒤 15분 되돌림 자체는 약합니다(+6.6bp, CI 하한 +0.03) — «되돌아온다»가 아니라 «급히 시장가로 닫지 말 것»만 근거가 있습니다."
          + (chipScoreLine("burst") ? `\n\n${chipScoreLine("burst")}` : "");
        r.setAttribute("x", bx); r.setAttribute("y", cvdY0 + 14 - bh + 4); r.setAttribute("width", bw); r.setAttribute("height", bh); r.setAttribute("rx", bh / 2);
        r.setAttribute("fill", "var(--warn)"); r.setAttribute("fill-opacity", ".12"); r.setAttribute("stroke", "var(--warn)"); r.setAttribute("stroke-opacity", ".55");
        g.appendChild(tt); g.appendChild(r); headEnd = bx + bw + 10;
        const tx = label(bx + 9, cvdY0 + 14, t1, "var(--warn)", null, fsz, g); tx.setAttribute("font-weight", "700");
        const ts = document.createElementNS(NS, "tspan"); ts.setAttribute("fill", "var(--muted)"); ts.setAttribute("font-weight", "600"); ts.textContent = t2; tx.appendChild(ts);
        svg.appendChild(g);
      }
    }
    { // 시각 꼬리표 = 머리 띠(봉 시작 · 끝 · 지금). 이름·배지와 겹치면 시작 꼬리표를 그 뒤로 민다
      const hhmm = (t) => { const dd = new Date(t * 1000);
        return String(dd.getHours()).padStart(2, "0") + ":" + String(dd.getMinutes()).padStart(2, "0"); };
      const ty = cvdY0 + 14, t0 = hhmm(first) + " 봉 시작", x0 = Math.max(ml, headEnd), x0End = x0 + measureTextW(t0, `500 10px ${"Pretendard Variable, Pretendard, 'Noto Sans KR', sans-serif"}`) + 8;
      label(x0, ty, t0, "var(--muted)");
      label(ml + cw, ty, hhmm(first + SUPPLY_1S_SEGMENT), "var(--muted)", "end");
      if (!narrow && now - first < SUPPLY_1S_SEGMENT * 0.82 && xAt(now) + 4 > x0End) label(xAt(now) + 4, ty, "지금 (" + (now - first) + "초 경과)", "var(--muted)");
    }
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

// Snapshot tab's candlestick chart (renderCandleSvg) with the liquidation-map density profile.
// Called right after its 5-min data sources refresh AND from render(), so the candle body and the
// current-price line move together.
// 2026-10-20 커서가 차트 위에 있어도 그린다(사용자 «호버하면 차트가 멈춘다»). 전에는 커서가 있으면 안 그리고 나갈 때 몰아 그렸다 --
//   다시 그리기가 svg 를 통째로 비워(renderCandleSvg 의 innerHTML = "") 읽던 십자선·툴팁이 «깜빡»이던 것을 막던 장치(65fc0396).
//   이제는 그린 **같은 작업 안에서** 마지막 커서 자리로 호버 처리를 다시 돌린다(renderSnapshotChart 끝) -- 칠하기 전이라 깜빡임이 없고,
//   창이 움직여도 툴팁은 늘 지금 커서 밑 봉을 말한다(옛 봉에 머물지 않는다).
let chartHoverPt = null;   // 마지막 커서(client 좌표) · 밖이거나 안쪽 패널(1초 수급 -- 중첩 svg, 제 툴팁) 위면 null
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

// 창 오른쪽 끝(배타, **소수** 봉 인덱스). 실시간이면 끝 · 과거면 chartPanEnd 시각의 봉 다음 + chartPanFrac. full = 지금 tf 의 캔들(chartTfFull) --
//   2026-10-20 받아 둔 구간 전체(1주일을 받았으면 1주일)로 갈 수 있다(전에는 12시간 안). 끝을 시각으로 들고 있어 1주일이 앞에 붙어도 안 밀린다.
function chartPanEdge(full) {
  const len = full.length;
  if (chartPanEnd == null) return len + chartPanRight;
  const i = full.findIndex((c) => c.time > chartPanEnd);
  return Math.max(Math.min(len, chartWindowBars), Math.min(len, (i < 0 ? len : i) + chartPanFrac));
}
function chartPanTo(edge) {   // edge = 소수 봉 인덱스(끌기·관성·휠이 픽셀 그대로 넣는다 -- 봉 단위로 반올림하지 않는다)
  const full = chartTfFull();
  if (!full.length) return false;
  const len = full.length;
  const e = Math.max(Math.min(len, chartWindowBars), Math.min(len + chartPanRightMax(), edge)), i = Math.floor(e);
  const next = e >= len - 1e-3 ? null : full[i - 1].time, frac = next == null ? 0 : e - i, right = next == null ? Math.max(0, e - len) : 0;
  if (next === chartPanEnd && Math.abs(frac - chartPanFrac) < 1e-4 && Math.abs(right - chartPanRight) < 1e-4) return false;
  chartPanEnd = next; chartPanFrac = frac; chartPanRight = right;
  scheduleSnapshotChartRender();
  maybeFetchFpWeek();   // 처음 옮기는 순간 1주일을 받는다
  return true;
}
function chartPanReset() { chartFlingStop(); chartPanEnd = null; chartPanFrac = 0; chartPanRight = 0; chartPanY = 0; scheduleSnapshotChartRender(); }   // 최신 봉 · 자동 축
const chartPanStep = () => Math.max(1, Math.round(chartWindowBars / 12));   // 한 칸 이동 = 창의 1/12(12봉 창 = 1봉 · 1주일 창 = 168봉)
const chartPanBy = (n) => chartPanTo(chartPanEdge(chartTfFull()) - n * chartPanStep());   // n>0 = 과거로
// 마지막 렌더의 가격 플롯 자리(viewBox 단위) -- 끌기·휠이 화면 픽셀을 봉으로 바꾼다(renderCandleSvg 가 채운다)
let chartGeo = null;
const chartSlotPx = (svg) => {   // 봉 한 칸 = 화면 몇 px · 커서 x → 플롯 안 비율(0~1)
  const g = chartGeo, r = svg.getBoundingClientRect();
  if (!g || !r.width) return null;
  const k = Math.min(r.width / g.wAll, r.height / g.hAll), x0 = r.left + (r.width - g.wAll * k) / 2 + g.ml * k;
  return { px: (g.cw * k) / g.n, at: (cx) => Math.max(0, Math.min(1, (cx - x0) / (g.cw * k))) };
};
// 관성(놓은 뒤 미끄러짐) -- 새로 누르기·휠·버튼·더블클릭이 멈춘다
let chartFling = 0;
function chartFlingStop() { if (chartFling) cancelAnimationFrame(chartFling); chartFling = 0; }
function chartFlingStart(v) {   // v = 봉/ms(+ = 최신 쪽)
  chartFlingStop();
  let t0 = performance.now();
  const step = (t) => {
    const dt = Math.min(50, t - t0); t0 = t;
    v *= Math.exp(-dt / 140);   // 140ms 시상수 -- 0.5초면 3% 로 잦아든다
    if (Math.abs(v) * dt < 0.002 || !chartPanTo(chartPanEdge(chartTfFull()) + v * dt)) { chartFling = 0; return; }
    chartFling = requestAnimationFrame(step);
  };
  chartFling = requestAnimationFrame(step);
}
// 휠 줌(2026-10-20 사용자 «단계로 뚝뚝 · 오른쪽 끝 기준»): 창 = bars(연속) · 커서 밑 시각이 커서 밑에 남는다(at = 플롯 안 커서 비율 -- 실시간이면 오른쪽 끝).
//   1주일을 아직 안 받았으면 받아 둔 만큼까지만 넓힌다(헛돈 휠이 쌓였다가 1주일이 오는 순간 튀지 않게).
function chartZoomAt(bars, at) {
  const full = chartTfFull(), len = full.length;
  if (!len) return;
  const hiB = Math.min(CHART_WINDOW_MAX[chartTfK] || 2016, Math.max(12, len));
  bars = Math.max(12, Math.min(hiB, bars));
  if (bars < 12 + 1e-3) bars = 12; else if (bars > hiB - 1e-3) bars = hiB;   // 리뷰: 끝에 붙이고 정확히 비교 -- «1e-3 안이면 안 바뀜»이면 12.0004 에 갇혀 12봉(풋프린트)로 못 돌아왔다
  if (bars === chartWindowBars) return;
  // 실시간(오른쪽 끝)이면 오른쪽 끝 기준 -- 리뷰: 커서 기준이면 넓혔다(끝에 막혀 실시간) 좁힐 때 끝이 len 아래로 내려가 조용히 실시간을 벗어났다
  const a = chartPanEnd == null ? 1 : at;
  const n0 = Math.min(chartWindowBars, len), e0 = chartPanEdge(full), u = e0 - n0 + a * n0;   // 커서 밑 봉 자리(소수)
  chartWindowBars = bars;
  const n1 = Math.min(bars, len);
  if (!chartPanTo(u - a * n1 + n1)) { scheduleSnapshotChartRender(); maybeFetchFpWeek(); }   // 옮겼으면 chartPanTo 가 둘 다 했다
  renderChartWindowTabs();
}
// 헤더 미니맵: 받아 둔 구간 종가 선 + 지금 창(밝은 띠). 누르거나 끌면 그 자리로 · 방향키 한 칸(Shift 6칸).
let chartPanKey = "", chartPanGeo = null;
function renderChartPan() {
  const box = el("chartPan");
  if (!box) return;
  const full = chartTfFull();
  if (!full.length || !flowOn()) { box.hidden = true; chartPanKey = ""; return; }
  box.hidden = false;
  const hist = full, n = hist.length;
  const eF = chartPanEdge(full), sF = Math.max(0, eF - chartWindowBars), live = chartPanEnd == null && !chartPanY && !chartPanRight;
  const s = Math.min(n - 1, Math.floor(sF + 1e-6)), e = Math.min(n, Math.ceil(eF - 1e-6));   // 오른쪽 빈 칸(chartPanRight)이면 eF > n   // 글자·aria 는 걸친 봉까지 · 띠는 소수 그대로(끌기 따라 미끄러진다)
  // 리뷰: 끌기·관성 한 프레임마다 미니맵을 통째로(전 구간 고저 · 350점 선 · innerHTML) 다시 만들었다 -- 선은 데이터가 바뀔 때만, 창이 움직이면 띠·글자만
  const W = 176, H = 26, X = (i) => (i / n) * W;
  const t = (c) => (hist[n - 1].time - hist[0].time > 86400 ? `${new Date(c.time * 1000).getMonth() + 1}/${new Date(c.time * 1000).getDate()} ` : "") + fmtHourMinute(c.time * 1000), end = hist[Math.max(0, e - 1)];
  const when = `${t(hist[s])}–${chartPanEnd == null ? "지금" : new Date(end.time * 1000).toDateString() === new Date(hist[s].time * 1000).toDateString() ? fmtHourMinute(end.time * 1000) : t(end)}`;
  const vt = `${t(hist[s])}부터 ${t(end)}까지`, vmin = Math.min(n, Math.round(chartWindowBars)), wx = X(sF).toFixed(1), ww = Math.max(3, X(Math.min(n, eF)) - X(sF)).toFixed(1);
  //   형성 봉은 체결마다 바뀐다 -- 고저가 그린 범위 안이고 종가가 같은 반 픽셀이면 선은 그대로 둔다(chartPanGeo = 마지막으로 그린 범위)
  const z = hist[n - 1], g = chartPanGeo, key = `${n}|${live}|${hist[0].time}|${z.time}|${chartTfK}`, map = box.querySelector(".chart-pan-map");
  if (key === chartPanKey && map && g && z.low >= g.lo && z.high <= g.hi && Math.round(g.Y(z.close) * 2) === g.yz) {
    const r = map.querySelector(".pan-win"), w2 = box.querySelector(".chart-pan-when");
    r.setAttribute("x", wx); r.setAttribute("width", ww);
    map.setAttribute("aria-valuemin", vmin); map.setAttribute("aria-valuenow", e); map.setAttribute("aria-valuetext", vt);
    if (w2 && w2.textContent !== when) w2.textContent = when;
    return;
  }
  chartPanKey = key;
  const lo = Math.min(...hist.map((c) => c.low)), hi = Math.max(...hist.map((c) => c.high));
  const Y = (v) => H - 3 - ((v - lo) / (hi - lo || 1)) * (H - 6);
  chartPanGeo = { lo, hi, Y, yz: Math.round(Y(z.close) * 2) };
  const every = Math.max(1, Math.ceil(n / (2 * W)));   // 1주일 5분봉 2,000여 점 → 선은 픽셀당 두 점
  const pts = [];
  for (let i = 0; i < n; i += every) pts.push(i);
  if (pts[pts.length - 1] !== n - 1) pts.push(n - 1);
  const line = pts.map((i, j) => `${j ? "L" : "M"}${X(i + 0.5).toFixed(1)} ${Y(hist[i].close).toFixed(1)}`).join("");
  const focused = box.contains(document.activeElement) && document.activeElement.classList.contains("chart-pan-map");
  box.innerHTML = `<svg class="chart-pan-map" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" preserveAspectRatio="none" role="slider" tabindex="0"`
    + ` aria-label="받아 둔 구간 안에서 차트 창 위치 — 방향키로 이동" aria-valuemin="${vmin}" aria-valuemax="${n}" aria-valuenow="${e}"`
    + ` aria-valuetext="${vt}">`
    + `<rect class="pan-win" x="${wx}" y="0.5" width="${ww}" height="${H - 1}" rx="4"/>`
    + `<path class="pan-line" d="${line}"/></svg>`
    + `<span class="chart-pan-when">${when}</span>`;   // 2026-10-26 펜 메모 «차트를 오가다 보면 두 글자가 겹친다 · 실시간/최신으로 제거»: 최신으로 = 차트 더블클릭 · End 키
  if (focused) box.querySelector(".chart-pan-map")?.focus();
}
function setupChartPan() {
  const box = el("chartPan"), svg = el("candleSvgSnapshot");
  if (!box || !svg) return;
  const full = chartTfFull;
  const mapIdx = (ev) => { const m = box.querySelector(".chart-pan-map"); if (!m) return null; const r = m.getBoundingClientRect(), n = full().length;
    return ((ev.clientX - r.left) / r.width) * n + chartWindowBars / 2; };
  box.addEventListener("pointerdown", (ev) => {
    if (!ev.target.closest(".chart-pan-map")) return;
    chartFlingStop();
    box.setPointerCapture(ev.pointerId);
    const go = (e2) => { const i = mapIdx(e2); if (i != null) chartPanTo(i); };
    go(ev);
    const up = () => { box.removeEventListener("pointermove", go); box.removeEventListener("pointerup", up); box.removeEventListener("pointercancel", up); };
    box.addEventListener("pointermove", go); box.addEventListener("pointerup", up); box.addEventListener("pointercancel", up);
  });
  box.addEventListener("keydown", (ev) => {
    const d = { ArrowLeft: 1, ArrowRight: -1 }[ev.key];
    if (ev.key === "End") { ev.preventDefault(); chartPanReset(); return; }
    if (!d) return;
    ev.preventDefault(); chartFlingStop(); chartPanBy(d * (ev.shiftKey ? 6 : 1));
  });
  // 차트 위 휠(2026-10-20 사용자): **세로 = 줌** -- 아래로 = 넓게(봉이 얇아지면 체결 박스 없이 캔들) · 위로 = 좁게(12봉 = 풋프린트).
  //   10-20 «트레이딩뷰처럼»: 단계 없이 휠 양만큼 연속(px × 0.0025 의 지수 -- 마우스 한 칸 100px ≈ 1.28배), 커서 밑 시각이 커서 밑에 남는다.
  //   **가로 휠(트랙패드 옆으로) = 이동**, 픽셀 그대로(손가락과 1:1). 끌기 = 1:1 + 놓으면 관성 · 더블클릭 = 최신으로 · 방향키(미니맵) = 이동은 그대로.
  const canPan = () => flowOn() && full().length > 0;   // 2026-10-20 30m·1h 도 · 10-21 창이 받은 구간 전체여도 오른쪽 빈 칸으로는 간다
  // 2026-10-20 사용자 «차트 위에 마우스가 있으면 휠을 아무리 움직여도 페이지는 안 움직이게»: 휠은 늘 차트가 삼킨다
  //   (끝(12봉·1주일), 30m·1h 창, 이동 못 하는 가로 휠도). 페이지 스크롤은 차트 밖에서. ctrl+휠(브라우저 확대)만 통과.
  svg.addEventListener("wheel", (ev) => {
    if (!flowOn() || ev.ctrlKey) return;
    ev.preventDefault();
    if (fpAltOn()) return;   // 리뷰: 4h·1d·청산맵·버블이 덮은 동안 안 보이는 5분 차트를 줌/이동하고 1주일까지 받았다(페이지 스크롤은 그대로 막는다)
    chartFlingStop();
    const g = chartSlotPx(svg);
    if (!g) return;
    const k = ev.deltaMode === 1 ? 40 : ev.deltaMode === 2 ? 800 : 1;   // 줄·쪽 단위(파이어폭스) 휠도 픽셀로
    const dx = ev.deltaX * k, dy = ev.deltaY * k;
    if (Math.abs(dx) > Math.abs(dy)) { if (canPan()) chartPanTo(chartPanEdge(full()) + dx / g.px); return; }   // 오른쪽으로 = 최신 쪽
    chartZoomAt(chartWindowBars * Math.exp(Math.max(-400, Math.min(400, dy)) * 0.0025), g.at(ev.clientX));   // 한 번에 400px 까지(빠른 휠에 12봉→1주일로 튀지 않게)
  }, { passive: false });
  // 안쪽 패널(1초 수급 등 중첩 svg)은 mousemove 버블을 끊는다 -- 잡는 단계(capture)에서 커서를 받아 둔다(다시 그린 뒤 호버 재생용)
  svg.addEventListener("mousemove", (ev) => { chartHoverPt = ev.target.closest("svg") === svg ? { x: ev.clientX, y: ev.clientY } : null; }, true);
  svg.addEventListener("mouseleave", () => { chartHoverPt = null; });
  // 리뷰: 화면 좌표라 스크롤하면 다른 봉을 가리킨다(손가락 탭은 mouseleave 가 없어 스크롤 뒤에도 렌더마다 되살아났다). 탭한 툴팁은 그대로 둔다(휴대폰 읽기)
  //   차트를 담은 것(페이지 · 감싼 칸)이 스크롤될 때만 -- 뉴스 목록 scrollTop 복원(30초 갱신) 같은 남의 스크롤은 커서를 안 지운다(리뷰)
  window.addEventListener("scroll", (ev) => {
    const t = ev.target;
    if (t === document || (t.contains && t.contains(svg))) chartHoverPt = null;
  }, { passive: true, capture: true });
  let drag = null;
  svg.addEventListener("pointerdown", (ev) => {
    chartFlingStop();
    if (ev.button !== 0 || ev.pointerType === "touch") return;   // 세로 이동은 창 크기와 무관하게 늘 된다(가로만 canPan)
    const g = chartSlotPx(svg);
    drag = { x: ev.clientX, y: ev.clientY, lx: ev.clientX, py: chartPanY, k: chartYPerPx, slot: g ? g.px : 1, moved: false, id: ev.pointerId, v: [] };
  });
  // 끌기 한 걸음(마우스·손가락 같이 쓴다)
  const dragTo = (d, cx, cy, ts) => {
    if (d.k) { chartPanY = d.py + (cy - d.y) * d.k; scheduleSnapshotChartRender(); }   // 아래로 끌면 위쪽 가격이 보인다
    // 오른쪽으로 끌면 과거가 보인다(1:1 -- 소수 봉 그대로). 지난 이동만큼 **지금 끝(시각 기준)**에서 뺀다 -- 리뷰: 누를 때 봉 번호를 들고 있으면
    //   끄는 중 1주일이 앞에 붙거나(첫 이동이 받기를 부른다) 새 봉이 오면 번호가 다른 봉을 가리켜 며칠 과거로 튀었다 · 끄는 중 휠 줌이면 봉 폭도 새로
    const g = chartSlotPx(svg);
    if (g) d.slot = g.px;
    if (canPan() && !d.vOnly) chartPanTo(chartPanEdge(full()) - (cx - d.lx) / d.slot);
    d.lx = cx;
    d.v.push([ts, cx]); if (d.v.length > 6) d.v.shift();
  };
  const flingFrom = (d, ts) => {   // 놓기 직전 ~100ms 속도로 관성(손을 멈췄다 놓으면 0)
    if (!d || !d.moved || d.vOnly || !canPan()) return;
    const v = d.v.filter((q) => ts - q[0] < 100);
    if (v.length < 2 || ts - v[v.length - 1][0] > 40) return;
    const pxPerMs = (v[v.length - 1][1] - v[0][1]) / Math.max(8, v[v.length - 1][0] - v[0][0]);
    if (Math.abs(pxPerMs) > 0.25) chartFlingStart(-pxPerMs / d.slot);
  };
  svg.addEventListener("pointermove", (ev) => {
    if (!drag || !(ev.buttons & 1)) { drag = null; return; }
    const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) < 5) return;
    if (!drag.moved) { drag.moved = true; svg.setPointerCapture(drag.id); svg.classList.add("panning"); }
    dragTo(drag, ev.clientX, ev.clientY, ev.timeStamp);
  });
  const stop = (ev) => {
    const d = drag; drag = null; svg.classList.remove("panning");
    if (ev.type === "pointerup") flingFrom(d, ev.timeStamp);
  };
  svg.addEventListener("pointerup", stop); svg.addEventListener("pointercancel", stop);
  svg.addEventListener("dblclick", () => { if (chartPanEnd != null || chartPanY || chartPanRight) chartPanReset(); });
  // 2026-10-27 휴대폰(사용자 «모바일에서도 풋프린트 차트를 자유롭게»): **가격 칸 안에서 시작한** 손가락만 차트가 받는다 --
  //   한 손가락 = 마우스 끌기와 같다(가로 시각 · 세로 가격 1:1 + 관성) · 두 손가락 = 벌리면 좁게(봉이 커짐)·오므리면 넓게 · 두 번 톡 = 최신으로.
  //   10-27(2) 사용자 «프로파일도 위아래로 드래그»: 아래 프로파일 칸(PROF)에서 끌면 **세로(가격)만** -- 같은 가격 범위라 차트도 같이 간다(칸 높이 기준 1:1).
  //   그 밖(레인·1초 수급)에서 시작하면 페이지 스크롤 그대로. 탭은 막지 않는다(툴팁 읽기 그대로).
  let tch = null, lastTap = 0;
  const zoneY = (cy) => {   // "plot" | "prof" | null
    const g = chartGeo, r = svg.getBoundingClientRect();
    if (!g || !r.height) return null;
    const k = Math.min(r.width / g.wAll, r.height / g.hAll), y = (cy - r.top - (r.height - g.hAll * k) / 2) / k;
    return y >= g.mt && y <= g.pb ? "plot" : g.prof && y >= g.prof[0] && y <= g.prof[1] ? "prof" : null;
  };
  const span = (t) => Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY) || 1;
  // 🔴움직임·떼기는 **손가락이 닿은 노드**에서 듣는다 -- 끌기 첫 걸음이 차트를 다시 그려 그 노드가 svg 밖으로 빠지면
  //   뒤따르는 touchmove 는 빠진 노드로만 가고 svg 까지 안 올라온다(실측: 한 걸음 뒤 페이지가 스크롤됐다).
  const bound = new Set();
  const bind = (tg) => { if (!tg || bound.has(tg)) return; bound.add(tg);
    tg.addEventListener("touchmove", onMove, { passive: false }); tg.addEventListener("touchend", onEnd); tg.addEventListener("touchcancel", onCancel); };
  const unbind = () => { bound.forEach((tg) => { tg.removeEventListener("touchmove", onMove); tg.removeEventListener("touchend", onEnd); tg.removeEventListener("touchcancel", onCancel); }); bound.clear(); };
  svg.addEventListener("touchstart", (ev) => {
    const t = ev.touches;
    if (t.length === 1) unbind();
    if (t.length === 1) {
      tch = null;
      const z = flowOn() && !fpAltOn() ? zoneY(t[0].clientY) : null;
      if (!z) return;
      chartFlingStop();
      const g = chartSlotPx(svg), G = chartGeo, vOnly = z === "prof";
      const k = vOnly ? chartYPerPx * (G.pb - G.mt) / (G.prof[1] - G.prof[0] - 16) : chartYPerPx;   // 칸 높이(머리 줄 16 뺀 그림 높이) 1px = 가격
      tch = { x: t[0].clientX, y: t[0].clientY, lx: t[0].clientX, py: chartPanY, k, slot: g ? g.px : 1, moved: false, v: [], vOnly };
      bind(ev.target);
    } else if (t.length === 2 && tch && !tch.vOnly && !tch.pinch) {   // 첫 손가락이 가격 칸 안이었을 때만(프로파일 칸은 세로 끌기만)
      chartFlingStop(); bind(ev.target);
      const g = chartSlotPx(svg);
      tch = { pinch: true, d0: span(t), b0: chartWindowBars, at: g ? g.at((t[0].clientX + t[1].clientX) / 2) : 1 };
    }
  }, { passive: true });
  function onMove(ev) {
    if (!tch) return;
    if (ev.cancelable) ev.preventDefault();   // 페이지 스크롤·확대 대신 차트
    const t = ev.touches;
    if (tch.pinch) { if (t.length >= 2) chartZoomAt(tch.b0 * tch.d0 / span(t), tch.at); return; }
    if (!tch.moved && Math.hypot(t[0].clientX - tch.x, t[0].clientY - tch.y) < 6) return;
    tch.moved = true;
    dragTo(tch, t[0].clientX, t[0].clientY, ev.timeStamp);
  }
  function onEnd(ev) {
    const d = tch;
    if (!d || ev.touches.length) { if (d && d.pinch) tch = null; return; }   // 두 손가락 중 하나만 떼면 끝(남은 손가락으로 튀지 않게)
    tch = null; unbind();
    if (d.pinch) return;
    if (d.moved) { flingFrom(d, ev.timeStamp); return; }
    if (ev.timeStamp - lastTap < 320) { lastTap = 0; if (chartPanEnd != null || chartPanY || chartPanRight) chartPanReset(); } else lastTap = ev.timeStamp;
  }
  function onCancel() { tch = null; unbind(); }
  // 2026-10-20 캔버스(fpCv)는 svg(viewBox meet)처럼 상자를 따라 저절로 늘지 않는다 -- 상자가 바뀌면 바로 다시 그린다(다음 폴링까지 어긋나지 않게)
  if (window.ResizeObserver) new ResizeObserver(() => scheduleSnapshotChartRender()).observe(svg);
  // 리뷰: 1배 → 2배 모니터로 옮기면 CSS 크기가 같아 위 관찰이 안 울린다 -- 장치 픽셀 비가 바뀌면 다시 그린다(새 비율로 다시 건다)
  const onDpr = () => matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`).addEventListener("change", () => { scheduleSnapshotChartRender(); onDpr(); }, { once: true });
  onDpr();
}
// ── 풋프린트 캔버스 두 장 (2026-10-20 사용자 «4번·5번도 모두 고쳐서» -- 바이낸스·트레이딩뷰처럼 부드럽게) ──────────
//   무거운 봉별 층을 svg **밑** 캔버스로 옮겼다(그림은 같다 -- 같은 좌표·같은 CSS 변수·같은 칠하기 순서):
//     bg = 청산 밀도 지도 · 가로 격자 · 칸 뒤 선(현재가 · max pain · VWAP·볼린저·주간/앵커 VWAP·HL 청산가)
//     fg = 체결 칸·숫자·POC 테두리 · 옆 5분봉/캔들 · 봉 델타 · POC 선 · 스택 불균형 · 청산 원 · 레인 막대·리본
//   svg 에는 글자·꼬리표·레벨 선·축·툴팁·십자선·표식만 남는다(끌기 한 프레임에 봉마다 노드를 다시 만들지 않는다).
//   좌표 = svg viewBox 그대로(변환 S = 장치 픽셀 비 × meet 배율). 색 = svg 의 계산된 CSS 변수(테마를 따른다).
//   그림 = 칠하기 목록 items: {x0,x1,y0,y1 = 차지하는 자리, c = 자르기 [x,y,w,h], p(ctx, q)}. 다시 칠할 때는 띠 q 와 겹치는 것만:
//     ① 전체 렌더 = 두 장 다 처음부터 ② 형성 봉 갱신(scheduleFpLiveFrame) = fg 의 그 봉 자리만 ③ 현재가 빠른 갱신 = bg 의 옛·새 선 띠만.
//   🔴svg 가 위라 svg 배경은 늘 투명이어야 한다(지금 그렇다). 일봉·청산맵·버블이 덮을 때는 CSS(#fpCard.fp-alt-on .fp-cv)가 두 장을 숨긴다
//     -- svg 는 fp_daily.js 의 구멍(clip-path)으로 숨는데 그 구멍은 캔버스에 안 걸린다.
//   잃은 것: 칸·델타·POC·청산 원의 브라우저 기본 제목(<title>) -- 같은 값을 봉 호버 툴팁이 말한다(칸 줄·청산 줄 추가).
const fpCv = { bg: null, fg: null, S: 1, ox: 0, oy: 0, cs: null, col: new Map(), fam: "", live: null, seq: 0 };
const CV_ALL = { x0: -1e9, x1: 1e9, y0: -1e9, y1: 1e9 };
const fpAltOn = () => { const c = el("fpCard"); return !!(c && c.classList.contains("fp-alt-on")); };   // 일봉·청산맵·버블이 5분 차트를 덮었나(fp_daily.js syncHole)
function fpCvLayer(svg, key) {
  let L = fpCv[key];
  if (!L || L.cv.parentNode !== svg.parentNode) {
    const cv = document.createElement("canvas");
    cv.className = "fp-cv"; cv.setAttribute("aria-hidden", "true");
    cv.style.cssText = "position:absolute;left:0;top:0;pointer-events:none";
    svg.parentNode.insertBefore(cv, svg);   // bg 를 먼저 만든다 → [bg, fg, svg]
    L = fpCv[key] = { cv, ctx: cv.getContext("2d"), items: [] };
  }
  return L;
}
// 렌더 시작: 두 장을 svg 상자 왼쪽 위에 맞추고 목록을 비운다. box = 방금 잰 svg 상자(렌더가 이미 잰 것 -- 강제 레이아웃을 늘리지 않는다).
//   크기 = 캔버스가 칠하는 자리(viewBox 0~w × 0~h = 왼쪽 열 · 넓은 화면 오른쪽 칸·좁은 화면 아래 1초 수급은 svg 만) -- 칠하고 합성할 픽셀이 넓은 화면에서 ~40% 준다.
function fpCvBegin(svg, box, wAll, hAll, w, h) {
  const pr = svg.parentNode.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const k = box.width > 0 && box.height > 0 ? Math.min(box.width / wAll, box.height / hAll) : 0;
  const ox = ((box.width - wAll * k) / 2) * dpr, oy = ((box.height - hAll * k) / 2) * dpr;
  const W = Math.min(Math.round(box.width * dpr), Math.ceil(ox + w * k * dpr)), H = Math.min(Math.round(box.height * dpr), Math.ceil(oy + h * k * dpr));
  const l = Math.round(box.left - pr.left - svg.parentNode.clientLeft) + "px", t = Math.round(box.top - pr.top - svg.parentNode.clientTop) + "px";
  ["bg", "fg"].forEach((key) => {
    const L = fpCvLayer(svg, key), c = L.cv, st = c.style;
    if (c.width !== W) c.width = W;
    if (c.height !== H) c.height = H;
    const cw = W / dpr + "px", chh = H / dpr + "px";   // 장치 픽셀에 딱 맞는 css 크기(반 픽셀이라도 늘이면 흐려진다)
    if (st.left !== l) st.left = l;
    if (st.top !== t) st.top = t;
    if (st.width !== cw) st.width = cw;
    if (st.height !== chh) st.height = chh;
    L.items = [];
  });
  fpCv.S = dpr * k; fpCv.ox = ox; fpCv.oy = oy;
  fpCv.cs = getComputedStyle(svg); fpCv.col.clear(); fpCv.fam = fpCv.cs.fontFamily; fpCv.live = null; fpCv.seq = fpTradeSeq;
  // 쓰는 색은 지금(svg 를 비운 직후 -- 스타일 계산이 싸다) 다 읽어 둔다. 렌더 끝에서 처음 읽으면 그때까지 쌓은 svg 의 스타일을 강제로 다시 계산한다(1주일 창 ~2ms).
  ["good", "bad", "amber", "ink", "line", "accent", "turnover", "muted", "warn", "text"].forEach((n) => cvc(`var(--${n})`));
}
const cvc = (v) => {   // "var(--good)" → 계산된 색(렌더마다 한 번) · 그 밖(rgb(…)·#…)은 그대로
  let c = fpCv.col.get(v);
  if (c === undefined) { c = typeof v === "string" && v.startsWith("var(") ? fpCv.cs.getPropertyValue(v.slice(4, -1).trim()).trim() || "transparent" : v; fpCv.col.set(v, c); }
  return c;
};
// 칠하기: q = 다시 칠할 띠(viewBox 단위, 없으면 전부). 띠는 장치 픽셀에 맞춰 넓힌다 -- 안쪽 픽셀은 전체 렌더와 같은 명령으로 다시 칠해져 이음매가 없다.
function fpCvPaint(L, r = null) {
  const S = fpCv.S;
  if (!L || !L.cv.width) return;
  const { cv, ctx } = L;
  let X0 = 0, Y0 = 0, X1 = cv.width, Y1 = cv.height;
  if (r) {
    X0 = Math.max(0, Math.floor(fpCv.ox + r.x0 * S)); X1 = Math.min(cv.width, Math.ceil(fpCv.ox + r.x1 * S));
    Y0 = Math.max(0, Math.floor(fpCv.oy + r.y0 * S)); Y1 = Math.min(cv.height, Math.ceil(fpCv.oy + r.y1 * S));
    if (X1 <= X0 || Y1 <= Y0) return;
  }
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(X0, Y0, X1 - X0, Y1 - Y0);
  if (!(S > 0)) { ctx.restore(); return; }
  if (r) { ctx.beginPath(); ctx.rect(X0, Y0, X1 - X0, Y1 - Y0); ctx.clip(); }
  ctx.setTransform(S, 0, 0, S, fpCv.ox, fpCv.oy);
  const q = r && { x0: (X0 - fpCv.ox) / S, x1: (X1 - fpCv.ox) / S, y0: (Y0 - fpCv.oy) / S, y1: (Y1 - fpCv.oy) / S };
  let clip = null;
  ctx._f = "";
  for (const it of L.items) {
    if (q && (it.x1 < q.x0 || it.x0 > q.x1 || it.y1 < q.y0 || it.y0 > q.y1)) continue;
    if (it.c !== clip) {
      if (clip) ctx.restore();
      clip = it.c || null; ctx._f = "";
      if (clip) { ctx.save(); ctx.beginPath(); ctx.rect(clip[0], clip[1], clip[2], clip[3]); ctx.clip(); }
    }
    it.p(ctx, q);
  }
  if (clip) ctx.restore();
  ctx.restore();
}
// svg 속성과 같은 뜻의 붓 넷(상태를 매번 다 정한다 -- 칠하기끼리 save/restore 없이)
function cvRect(ctx, x, y, w, h, fill, op = 1, rx = 0) {
  if (!(w > 0) || !(h > 0)) return;
  ctx.globalAlpha = op; ctx.fillStyle = cvc(fill);
  if (rx > 0 && ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, w, h, Math.min(rx, w / 2, h / 2)); ctx.fill(); } else ctx.fillRect(x, y, w, h);   // roundRect 없는 옛 사파리(<16)는 모서리만 각지게
}
function cvStroke(ctx, pts, stroke, lw = 1, op = 1, dash = null, join = "miter", close = false) {
  ctx.globalAlpha = op; ctx.strokeStyle = cvc(stroke); ctx.lineWidth = lw; ctx.lineJoin = join;
  ctx.setLineDash(dash ? String(dash).split(/[\s,]+/).map(Number) : []);
  ctx.beginPath();
  for (let i = 0; i < pts.length; i++) i ? ctx.lineTo(pts[i][0], pts[i][1]) : ctx.moveTo(pts[i][0], pts[i][1]);
  if (close) ctx.closePath();
  ctx.stroke();
}
function cvStrokeRect(ctx, x, y, w, h, stroke, lw = 1, op = 1, rx = 0) {
  ctx.globalAlpha = op; ctx.strokeStyle = cvc(stroke); ctx.lineWidth = lw; ctx.setLineDash([]);
  ctx.beginPath(); if (rx > 0 && ctx.roundRect) ctx.roundRect(x, y, w, h, Math.min(rx, w / 2, h / 2)); else ctx.rect(x, y, w, h); ctx.stroke();
}
// 1px 가로·세로 선(심지·격자·0선) = 폭 1 사각형 -- 1px 선(맞대기 끝)과 덮는 픽셀이 같고 소프트웨어 래스터에서 선 긋기보다 싸다
const cvVLine = (ctx, x, y0, y1, col, op = 1) => cvRect(ctx, x - 0.5, Math.min(y0, y1), 1, Math.abs(y1 - y0), col, op);
const cvHLine = (ctx, x0, x1, y, col, op = 1) => cvRect(ctx, x0, y - 0.5, x1 - x0, 1, col, op);
function cvText(ctx, x, y, s, size, weight, fill, op = 1, align = "start") {
  const f = `${weight} ${size}px ${fpCv.fam}`;
  if (ctx._f !== f) { ctx.font = f; ctx._f = f; }
  ctx.globalAlpha = op; ctx.fillStyle = cvc(fill); ctx.textAlign = align === "middle" ? "center" : align;
  ctx.fillText(s, x, y);
}
// 옛 «같은 색끼리 path 하나»(svg denseAdd)와 같은 칠 -- 겹친 곳·맞닿은 가장자리가 두 번 칠해지지 않는다(반투명 이음매·겹친 원이 진해지지 않게).
function cvUnion() {
  const m = new Map();
  return {
    add(fill, op, draw) { const k = fill + "|" + op; let e = m.get(k); if (!e) m.set(k, (e = { fill, op, p: new Path2D() })); draw(e.p); },
    flush(push) {
      const list = [...m.values()]; m.clear();
      if (list.length) push((ctx) => list.forEach((e) => { ctx.globalAlpha = e.op; ctx.fillStyle = cvc(e.fill); ctx.fill(e.p); }));
    },
  };
}
// svg 의 <line>/<polyline> 을 같은 속성 그대로 bg 칠하기로 옮기고 svg 쪽은 투명하게 남긴다(<title> 호버는 그대로 산다).
function cvFromSvgLines(g, items, at, clip) {
  const add = [];
  [...g.children].forEach((e) => {
    if (e.tagName !== "line" && e.tagName !== "polyline") return;
    const a = (k, d) => { const v = e.getAttribute(k); return v == null ? d : v; };
    let pts = [[+a("x1", 0), +a("y1", 0)], [+a("x2", 0), +a("y2", 0)]];
    if (e.tagName === "polyline") { const n = a("points", "").trim().split(/[\s,]+/); pts = []; for (let i = 0; i + 1 < n.length; i += 2) pts.push([+n[i], +n[i + 1]]); }
    const stroke = a("stroke", "none"), lw = +a("stroke-width", 1), op = +a("stroke-opacity", 1) * +a("opacity", 1), dash = a("stroke-dasharray", null);
    if (stroke !== "none" && pts.length > 1) add.push({ ...CV_ALL, c: clip, p: (ctx) => cvStroke(ctx, pts, stroke, lw, op, dash) });
    e.style.opacity = "0";
  });
  items.splice(at, 0, ...add);
}
// 형성 봉 갱신(2026-10-20 사용자 4번): 체결이 온 프레임마다(rAF 로 묶음) 맨 오른쪽 봉의 캔들·칸·숫자·POC·델타·청산 원만 fg 에 다시 칠한다.
//   나머지는 전체 렌더(400ms 게이트) 그대로. 축(가격 범위)·행 크기·음영 기준·밀도 쓸림·델타 겹침 판정이 바뀌면 전체 렌더로 넘긴다(fpCv.live 가 false).
let fpLiveRaf = 0, fpLiveRafAt = 0, fpTradeSeq = 0;   // fpTradeSeq = 체결 수 -- 전체 렌더·형성 봉 갱신이 «어디까지 칠했나»(fpCv.seq)를 남겨 새 체결 없는 프레임은 건너뛴다
function scheduleFpLiveFrame() {
  fpTradeSeq++;
  if (!fpCv.live) return;
  if (fpLiveRaf) { if (Date.now() - fpLiveRafAt < RAF_STALL_MS) return; cancelAnimationFrame(fpLiveRaf); }   // 묵은 예약(가려진 창) -- 위 RAF_STALL_MS 와 같은 이유
  fpLiveRafAt = Date.now();
  fpLiveRaf = requestAnimationFrame(() => {
    fpLiveRaf = 0;
    if (activePageTab !== "snapshot" || isScrolling() || !fpCv.live || snapshotChartRafId || fpCv.seq === fpTradeSeq || fpAltOn()) return;   // 전체 렌더가 이 프레임에 오거나 방금 다 그렸으면 그쪽 · 일봉 등이 덮었으면(캔버스 숨김) 안 칠한다(리뷰)
    const seq = fpTradeSeq;
    if (fpCv.live() === false) maybeRenderSnapshotChartNow(); else fpCv.seq = seq;
  });
}
function renderSnapshotChart() {
  renderChartPan();
  const svg = el("candleSvgSnapshot");
  if (!svg) return;
  const fullCandles = chartTfFull();   // 2026-10-20 1주일을 받았으면 그 앞에 붙은 것
  if (!fullCandles.length) return;
  // Sliced to SNAPSHOT_CHART_MAX_CANDLES (6h) -- narrower than the shared candleHistoryByAsset
  // cache (still 8h, CHART_MAX_CANDLES) so the density-history overlay always has a real snapshot
  // behind every visible column (see that constant's comment).
  // 풋프린트 모드면 테이프가 덮는 구간(최대 12봉=1시간)만 그린다 -- 72봉을 1200px 에 넣으면
  // 봉당 16px 라 셀이 물리적으로 안 들어간다. 대신 청산밀도 히트맵은 끈다(셀과 같은 자리를
  // 다투고, 정확한 레벨은 차트 아래 목록에 그대로 있다). 2026-09-15 사용자 결정 "완전 교체".
  const footprint = tfFootprint(footprintForChart());
  // 🔴2026-09-23 풋프린트 모드에서 캔들을 **풋프린트 범위로 자르고** 있었다. 그래서 12h 를
  //   골라도 서버 풋프린트 링이 덮는 만큼(실측 54봉=4.5시간)만 보였다 -- 캔들은 200개
  //   (16.7시간) 있는데도. 위 주석의 이유(«72봉을 1200px 에 넣으면 셀이 안 들어간다»)는
  //   drawCells(봉 폭 기준)가 이미 해결했다: 얇아지면 셀을 안 그린다.
  //   이제 창이 자른다. 풋프린트가 없는 봉은 셀·델타가 그냥 비고(buyTot+sellTot>0 가드),
  //   사분면·누적 레인은 **캔들을 돌며 fpBars 를 조회**하는 구조라 빠진 봉을 알아서 건너뛴다.
  // 2026-09-27 흐름 코인은 풋프린트가 **아직 안 와도** 창으로 자른다 -- 전환 직후 0.3~0.7초 동안 96봉 캔들로
  //   넓어졌다 줄어드는 게 «1h 인데 캔들로 12h» 로 보였다.
  // 2026-10-20 끌기·휠 1:1(사용자 «트레이딩뷰처럼»): 창 = 소수 봉 [edge − n, edge). 양 끝에 걸친 봉까지 잘라 넘기고(최대 +2봉)
  //   어긋남(off, 봉)을 renderCandleSvg 에 준다 -- 걸친 봉은 플롯 클립에 반쯤 잘려 보인다. 풋프린트가 없으면(웜업) 옛 정수 창.
  const len = fullCandles.length, n = Math.min(chartWindowBars, len), edge = chartPanEdge(fullCandles), s0 = Math.max(0, Math.floor(edge - n + 1e-6));
  const flowCut = footprint || flowOn();
  const candles = flowCut ? fullCandles.slice(s0, Math.min(len, Math.ceil(edge - 1e-6))) : fullCandles.slice(-SNAPSHOT_CHART_MAX_CANDLES);
  const view = flowCut && footprint ? { n, off: s0 - (edge - n) } : null;
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
  const boxesOff = fpBoxesOff;
  renderCandleSvg(svg, candles, entryPrice, currentPrice, riskLevels,
    densityHistory, tfLiq(weekJoin("liq", latestLiquidation5mHist, (b) => Date.parse(b.ts))), footprint, view);
  if (boxesOff !== fpBoxesOff) renderChartWindowTabs();   // 머리 «· 캔들»
  // 호버 재생(위 chartHoverPt 주석) -- 이 렌더가 새로 단 핸들러에 마지막 커서를 넣는다(툴팁·십자선이 새 그림·새 봉 기준으로)
  const hp = chartHoverPt;
  if (hp && svg.onmousemove) svg.onmousemove({ clientX: hp.x, clientY: hp.y, pageX: hp.x + window.scrollX, pageY: hp.y + window.scrollY });
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


function renderCandleSvg(svg, candles, entryPrice, currentPrice, riskLevels = [], densityHistory = [], liqBars = [], footprint = null, view = null) {
  // 2026-10-20 view = { n: 창 봉 수(소수), off: 첫 봉의 어긋남(봉, ≤0) } -- 끌기·휠 연속(renderSnapshotChart). 없으면 봉 수 = 창(옛 그림).
  const N = view ? view.n : candles.length, X0 = view ? view.off : 0;
  // 2026-10-20 끌기·휠 연속 그리기의 비용 절반이 «강제 레이아웃»이었다(렌더 중간에 글자 폭·상자를 재면 그때까지 쌓은 svg 전체를 다시 배치한다).
  //   ① 비우고 나서 잰다(빈 svg 배치는 공짜) ② 글자 폭은 캔버스(textW) ③ 화면 상자를 재는 HTML 배치는 끝(post)으로 몰아 한 번만.
  svg.innerHTML = "";
  const post = [];
  const svgFam = getComputedStyle(svg).fontFamily;
  const textW = (t) => measureTextW(t.firstChild ? t.firstChild.nodeValue || "" : "", `${t.getAttribute("font-weight") || 400} ${t.getAttribute("font-size") || 16}px ${svgFam}`);
  const parentW = svg.parentElement ? svg.parentElement.clientWidth : 0;
  // 2026-09-18 부모가 아니라 **SVG 자신의** 높이를 본다. 부모는 마진 12px 를 포함하므로
  // (styles.css .chart-container 412px) 그 값으로 viewBox 를 잡으면 모바일에서 meet 축소 +
  // 레터박스가 생긴다. viewBox 높이는 SVG 가 실제로 차지하는 상자와 같아야 한다.
  const box0 = svg.getBoundingClientRect();   // 비운 뒤라 싸다(위 ①)
  const parentH = box0.height
    || (svg.parentElement ? svg.parentElement.clientHeight : 0);
  const mobileChart = isMobileChartMode();
  // 2026-09-19 2열 배치(사용자 지시)로 상자가 카드 폭의 68%/32% 가 됐다. 1200/400 을 고정으로
  // 두면 viewBox 가 상자보다 커서 meet 축소가 걸리고 글자가 그만큼 작아진다 -- 상자에서 받는다.
  // 하한은 «아직 레이아웃 전»(parentW/H = 0)일 때의 폴백이다.
  const wAll = parentW > 0 ? Math.max(parentW, 320) : 1200;
  const hAll = parentH > 0 ? Math.max(parentH, 260) : 400;   // 상자 전체(viewBox). 본문의 `h` 는 아래(모바일은 풋프린트 영역만)
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
  if (svg.id === "candleSvgSnapshot" && !subOn) fpTopRowOff();   // 2026-10-28 검증: 아래 SUB 블록을 안 지나는 코인(BTC)은 위 줄이 넓은 화면 상태로 남았다
  // 2026-09-27 넓은 화면 2단(사용자 선택 62:38): CSS 가 --fp-split(오른쪽 칸 비율)을 준다 -- 판정은 CSS 한 곳.
  //   가격 플롯·레인·밀도 범례는 왼쪽 w, 1초 수급은 오른쪽 칸(subX, subW) 전체 높이. viewBox 는 wAll.
  //   아래 본문은 전부 `w` 를 «플롯 폭»으로 쓰므로 한 줄도 안 바뀐다.
  const splitR = subOn ? Number(getComputedStyle(svg).getPropertyValue("--fp-split")) || 0 : 0;
  const SPLIT_GAP = 20;
  const subW = splitR ? Math.round((wAll - SPLIT_GAP) * splitR) + FP_LEFT_SHIFT : wAll;   // 10-15 풋프린트를 왼쪽으로 옮긴 만큼 오른쪽 칸을 넓힌다(플롯 폭은 그대로 -- 아래 ml 도 같은 만큼 줄인다)
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
  // 2026-10-27 휴대폰 풋프린트(사용자 «좌우 프로파일을 차트 아래로 별도로 · 차트는 크게»): 캔들 옆 체결·호가 띠를 빼고
  //   1초 수급 위 제 칸(PROF)에 «← 체결 | 호가 →» 를 차트와 같은 가격 범위로 맞대 그린다. 덮개(일봉·버블)가 덮으면 안 그린다.
  const PROF_H = footprint && mobileChart && S1_BELOW && !fpAltOn() ? 260 : 0, PROF_GAP = 28;   // PROF_GAP = 가운데 가격 등뼈 반폭
  const PROF_PANEL = PROF_H ? PROF_H + SUB_GAP : 0;   // 아래 칸 + 간격 -- 본문 바닥 h 와 1초 수급 사이
  const h = S1_BELOW ? hAll - S1_PANEL - SUB_GAP - PROF_PANEL : hAll;
  const SUB_TOTAL = !subOn ? 0 : splitR ? SUB_LEGEND_H + SUB_GAP : SUB_LEGEND_H + STATS_ROW_H + 10 + SUB_GAP;
  fpCvBegin(svg, box0, wAll, hAll, w, h);   // 2026-10-20 무거운 층은 캔버스 두 장(fpCv 주석) -- bg·fg 칠하기 목록을 아래 블록들이 채우고 끝에서 한 번 칠한다
  const bgI = fpCv.bg.items, fgI = fpCv.fg.items;
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
  const TRADE_L = footprint && mobileChart && !PROF_H ? 50 : 0;
  // 2026-09-28 모바일 풋프린트는 왼쪽 가격 글자를 없앴다(사용자 «차트가 너무 작다») -- 값은 플롯 아래 한 줄이 갖는다. 여백 44 -> 4.
  // 2026-09-30 풋프린트(데스크톱)는 왼쪽 여백 71 -> 36(사용자 «풋프린트를 왼쪽으로»).
  const ml = (mobileChart ? (footprint ? 4 : 44) : footprint ? 36 - CENTER_NUDGE : 45) + CENTER_NUDGE + TRADE_L - (splitR ? FP_LEFT_SHIFT : 0),
        // 2026-09-27 풋프린트는 가격 배지를 전부 왼쪽 글자로 옮겼다(사용자 «호가가 잘 보이게») -- 오른쪽은 호가 띠가 쓴다.
        mr = (mobileChart ? 10 : footprint ? 34 : 112) - CENTER_NUDGE,
        mtTop = 12, mt = mtTop + SUB_TOTAL + (splitR && svg.id === "candleSvgSnapshot" ? FP_TOP_ALIGN : 0), mb = svg.id === "candleSvgSnapshot" ? 28 : 70;   // 2026-10-06 풋프린트는 레짐·전환 리본 두 줄(+28~+64)이 빠져 x축 글자(+21)까지만 -- 빈 42px 는 가격 플롯에
  // 2026-09-21 사용자 요청: 「차트가 너무 많다 -- 레인을 한 덩어리로」 + 「가격 플롯을 키워라」.
  //   레인 5종(거래대금·델타/CVD·OI·수급·청산)이 각자 6px 간격으로 떨어져 있어 **다섯 장의
  //   그림**으로 읽혔다. 간격 6->3, 높이를 서로 가깝게 맞춰 한 블록으로 읽히게 한다:
  //   데스크톱 168 -> 111px. 그 57px 과 상자 증가분을 전부 가격 플롯이 가져간다(205 -> 400).
  // ⭐다섯 레인은 전부 «높이=해상도» 노브다 -- OI·청산은 대칭 이분(half = H/2 - 1), 수급은
  //   SMID 기준, 거래대금은 단극이다. 안에 고정 크기 요소가 없어 높이만 바꾸면 되고 좌표
  //   계산은 한 줄도 안 건드린다.
  // 🔴DCVD 는 24 아래로 내리지 말 것 -- 15 에서 작은 델타 막대가 0.7px 였다(기존 주석).
  // OI 신규계약 레인 -- 청산 레인 **바로 위**(사용자 지시). 별도 패널이 아니라 이 SVG 안의
  // 서브플롯이라야 캔들과 x축(봉)이 구성상 같아진다(레짐 리본이 같은 이유로 여기 있다).
  // 🔴ETH 전용이다. 다른 코인을 보는 동안 ETH 값을 얹으면 2026-08-31 레짐 리본 사고와 같은
  //   모양이 된다 -- 자산 전환에서 latestOi5m 을 비우는 것과 이 조건, 둘 다 필요하다.
  // 데이터가 없으면 자리를 아예 안 잡는다(웜업·다른 코인에서 가격 플롯만 40px 손해다).
  const oiBars = (svg.id === "candleSvgSnapshot" && flowOn()
                  && latestOi5m && Array.isArray(latestOi5m.bars)) ? tfOi(weekJoin("oi", latestOi5m.bars, (b) => Number(b[0]))) : [];
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
                  ? tfFpBars(fpBarsNow()) : [];
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
  // 2026-10-01 넓은 ETH 2단: 사분면 줄을 줄여 가격 플롯 바닥 + 12 = ④ 윗변(③ 가격 지형 = ④ 와 같은 높이, 사용자 지시) -- 줄어든 만큼 풋프린트·프로파일이 길어진다.
  //   ④ 윗변 = hAll − 8 − ④ 높이(시장 맥락이 칸 바닥에 붙는다), 플롯 바닥 = h − 76 − QUAD_H − QUAD_TXT ⇒ QUAD_H + QUAD_TXT = ④ 높이 − 56.
  // 2026-10-26 펜 메모 «이 박스는 이제 제거해줘»: 리본 아래 델타·신규계약·고래·중형·리테일 다섯 줄을 뺐다(시장 맥락 ①·② σ 게이지가 같은 값을 말한다).
  //   남는 것 = 리본(22) + 간격(8) + 거래대금 한 줄 + 바닥 4. 데스크톱 196 → 88(+108px), 모바일 172 → 42(+130px, 거래대금 20px)는 가격 플롯으로(카드 높이 그대로).
  //   🔴데스크톱 거래대금 줄은 24 → 54px: 넓은 화면은 프로파일 아래 지지/저항 칸(#mcWall, 7줄 · 바닥 = 플롯 바닥 + 12)이 이 높이를 같이 쓴다 --
  //     58 로 줄이면 칸이 216 → 78px, 줄이 10px 라 글자가 겹쳤다(실측). 88 이면 108px · 줄 ~15px.
  //   막대 밑 글자 칸(QUAD_TXT)·모바일 «레인 아래 한 줄»(ROW_H)도 같이 뺐다 -- 다섯 줄 높이 몫이었다.
  const QUAD_H = fpBars.length ? (mobileChart ? 42 : 88) : 0;
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
  const PRICE_ROW_H = mobileChart ? 18 : 0;                // 가격 플롯 아래 한 줄
  // ── 거래대금 · 델타·CVD -- 풋프린트 바로 아래 (2026-09-21 사용자 지시) ────────────
  // 같은 풋프린트 봉에서 나온다: 거래대금 = Σ가격x(매수+매도) · 델타 = Σ(매수-매도) ·
  // CVD = 그 창 안에서의 델타 누적.
  // 🔴CVD 의 기준점은 **창 시작**이고, 창은 휠 줌·이동(12봉 ~ 1주일, chartWindowBars)을 그대로 따른다(사용자 결정).
  //   창을 바꾸면 기준점도 같이 옮겨간다 -- 절대 누적이 아니다.
  // 🔴«상황 읽기» 카드의 CVD 는 **30분 고정창**이다(dashboard/situation.py 의 WINDOW=6).
  //   이름이 같아도 값이 다르다. 툴팁에 창을 적는다.
  // 2026-09-27 풋프린트 오른쪽 «실시간 호가 띠»(사용자 지시) -- 캔들 폭에서 띠 폭을 뗀다(데스크톱 70 · 모바일 40).
  // 2026-09-27 넓혔다(사용자 «호가 너비를 키워줘») -- 오른쪽 배지 자리(86px)를 띠에 줬다: 70 -> 148 · 모바일 40 -> 56.
  const BOOK_W = footprint ? (mobileChart ? (PROF_H ? 0 : 56) : 140) : 0;   // 2026-09-30 148 -> 112 -> 140(꼬리표 칸을 없애 돌려줌)(사용자 «차트·프로파일을 조금 줄여도 데이터가 다 보이게»)   // 모바일은 수량 글자 없이
  // 2026-09-28 체결 기둥(사용자 선택 시안 B) -- 풋프린트와 호가 띠 사이 150px. 모바일은 기둥 없이 셀 뒤에 깐다(시안 A).
  const TRADE_W = footprint && !mobileChart ? 140 : 0;   // 2026-09-30 150 -> 112 -> 140
  // 2026-09-30 가격 꼬리표 칸(사용자 «왼쪽 라벨을 오른쪽으로, 겹치지 않게»): 캔들 칸과 체결 기둥 사이 120px.
  //   가격 선·레벨의 이름+가격을 여기 세로로 쌓고(겹침 회피) 실제 가격 행까지 지시선을 긋는다. 오른쪽 끝 30px 는 행사가 감마 막대.
  // 2026-09-30 가격 이름 = 시안 C(사용자 선택): 꼬리표 칸을 없애(폭은 체결·호가 띠로) 플롯 안 오른쪽 끝에 **짧은 약어**(R1·S1·MP·1h·5m·VW·ΓF·HL롱),
  //   이름·가격은 마우스를 올리면. 현재가는 상자 없이 **숫자만** 크게(바탕 외곽선).
  const LBL = "C";
  const cw = w - ml - mr - BOOK_W - TRADE_W;
  const tradeX0 = ml + cw;   // 체결 기둥 왼쪽 끝
  fpProfileSpan = TRADE_W ? [tradeX0, w - mr] : null;   // 아래 옵션 줄 가운데 칸이 이 폭·x 를 따른다(optRowCols)
  let tradeInfo = null;       // 체결 행(호버가 읽는다) -- 풋프린트 블록이 채운다
  let stackItem = null, liqLive = null, cellTip = () => "";   // 2026-10-20 형성 봉 갱신(fpCv.live)이 다시 부르는 봉별 칠하기 · 칸 호버 줄
  let posInfo = null;         // 2026-10-10 포지션 행(호버가 읽는다) -- 포지션 보기일 때 체결 대신
  let liqInfo = null;         // 2026-10-27 청산 행(호버가 읽는다) -- 청산 보기일 때 체결 대신
  let fpRowSize = 0;          // 풋프린트 행 크기($) -- 호가 띠가 같은 행으로 묶는다(2026-09-28)
  let fpRowOff = 0;           // 그 행 격자의 어긋남($) = 버킷/2 (2026-10-11, 아래 off)
  const ch = h - mt - mb - QUAD_H - (LANE_MERGE ? 0 : CUM_H + LANE_GAP) - LANE_GAP
            - PRICE_ROW_H;
  const plotBottom = mt + ch;                      // 가격 플롯의 바닥
  // 2026-10-08 사용자 «풋프린트부 차트부터 모의 판+지지·저항까지 일봉·청산맵으로 대체»: 교체 영역 = 왼쪽 열 전체(0,0)~(w,h).
  //   넓은 화면(split)은 오른쪽 1초 수급 칸이 남고, 좁은 화면은 아래 1초 수급 패널(h 밑)이 남는다. split 이면 왼쪽 아래 #mcWall(모의 판·지지/저항)도 덮인다.
  if (svg.id === "candleSvgSnapshot") window.fpRegion = { w: w, h: h, W: wAll, H: hAll, split: !!splitR };
  // 수급 두 패널은 **가격 플롯 위**다(위 mt 주석). OI·청산 레인은 플롯 바로 아래 그대로다.
  // 2026-09-22 위아래를 뒤집었다(사용자 지시): **프로파일이 먼저, 1초 수급이 그 아래**.
  // 🔴청산밀도 범례는 프로파일을 **따라 올라가지 않는다**. 그건 풋프린트(가격 플롯)의 배경을
  //   설명하는 것이라 그 바로 위에 있어야 한다(SUB_LEGEND_H 주석의 «풋프린트 차트 바로 위»).
  //   프로파일에 붙여 올리면 설명하는 그림에서 400px 멀어진다.
  // 소비 합 = SUB_TOTAL: 190(프로파일) + 8 + 400(1초) + 14(범례) + 8 = 620.
  const sub1sY = S1_BELOW ? h + SUB_GAP + PROF_PANEL : mtTop;
  const PROF = PROF_H ? { top: h + SUB_GAP + 36, bot: h + SUB_GAP + PROF_H, cx: Math.round((ml + w - mr) / 2) } : null;   // 머리 = 체결·포지션·청산 토글 줄 + 글자 줄(36) 아래부터
  const subLegendY = splitR ? mtTop : S1_BELOW ? mtTop : sub1sY + SUB_1S_H - STATS_ROW_H;   // 2단: 왼쪽 칸 맨 위 · 그 밖: 1초 수급 바로 아래(= 풋프린트 바로 위)   // 2단: 왼쪽 칸 맨 위(풋프린트 바로 위)
  const quadY = plotBottom + PRICE_ROW_H + LANE_GAP;  // 사분면 막대 바닥 = quadY + QUAD_H
  const cumY = LANE_MERGE ? quadY : quadY + QUAD_H + LANE_GAP; // 누적 행 위쪽
  const cumBottom = cumY + CUM_DRAW_H;
  const NS = "http://www.w3.org/2000/svg";
  // 풋프린트는 서버가 주는 12봉이 곧 창이다 -- 모바일 핀치줌(visibleCandleWindow)으로 더
  // 잘라내면 셀만 커지고 볼 구간이 사라진다.
  const viewport = footprint ? { candles, includeCurrent: true } : visibleCandleWindow(candles);
  candles = viewport.candles;
  const includeCurrentPrice = viewport.includeCurrent;

  svg.setAttribute("viewBox", `0 0 ${wAll} ${hAll}`);
  
  if (!candles.length) {
    const txt = document.createElementNS(NS, "text");
    txt.setAttribute("x", w/2); txt.setAttribute("y", h/2);
    txt.setAttribute("text-anchor", "middle"); txt.setAttribute("fill", "var(--muted)");
    txt.textContent = "시장 데이터 대기 중...";
    svg.appendChild(txt);
    fpCvPaint(fpCv.bg); fpCvPaint(fpCv.fg);   // 옛 그림을 비운다
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
  // 2026-09-27 풋프린트는 현재가를 세로 **정중앙**에(사용자 지시) -- 먼 쪽 거리로 위아래 대칭.
  //   가운데는 **5분봉마다 한 번** 잡는다(같은 날 사용자 «매 5분봉마다»): 봉이 바뀔 때의 현재가로 정하고 그 봉 동안은
  //   고정한다 -- 틱마다 따라가면 화면 전체가 계속 출렁인다. 봉 안에서 범위를 벗어나면 가운데는 두고 위아래만 넓힌다.
  //   🔴가운데는 풋프린트 칸 단위, 반폭은 두 칸 단위로 반올림한다(yMin/yMax 가 층 캐시 서명 baseGeomSig 에 들어 있다).
  const fpQ = footprint ? Number(footprint.bucket) || 0 : 0, fc = renderCandleSvg._fpCenter || (renderCandleSvg._fpCenter = { key: "", c: 0 });
  if (footprint && currentPrice > 0 && !panned) {
    const barT = candles.length ? candles[candles.length - 1].time : 0;
    const fcKey = activeSnapshotAsset + "|" + barT;
    if (fc.key !== fcKey) {
      fc.key = fcKey;
      fc.c = fpQ > 0 ? Math.round(currentPrice / fpQ) * fpQ : currentPrice;
    }
  }
  // 2026-10-20 함수로 묶었다 -- 형성 봉 갱신(fpCv.live)이 «새 고저·현재가로 축이 그대로인가»를 같은 식으로 다시 잰다
  const yRangeOf = (minP, maxP) => {
    const pad = (maxP - minP) * padPct || 1;
    let yMin = minP - pad, yMax = maxP + pad;
    if (footprint && currentPrice > 0 && !panned) {
      const c = fc.c;
      let half = Math.max(yMax - c, c - yMin);
      if (fpQ > 0) half = Math.ceil(half / (2 * fpQ)) * 2 * fpQ;
      yMin = c - half; yMax = c + half;
    }
    if (svg.id === "candleSvgSnapshot" && chartPanY) { yMin += chartPanY; yMax += chartPanY; }   // 2026-10-09 세로 끌기
    return [yMin, yMax];
  };
  const [yMin, yMax] = yRangeOf(minP, maxP);
  const ySpan = Math.max(yMax - yMin, 1e-5); // Prevent division by zero
  if (svg.id === "candleSvgSnapshot") chartYPerPx = ySpan / Math.max(1, ch);

  // 2026-09-21 청산 밀도는 **전체폭**이다 -- 체결 봉 뒤로 지나간다(사용자 요청).
  //   한때 풋프린트만 왼쪽 게이트로 뺐는데, 그건 «알파를 낮춰 겹치기»가 척도를 눌러 버린
  //   것을 피하려던 우회였다. 알파는 두 모드 같은 0.85 로 두고 겹치기를 허용한다.
  // 🔴셀 가독성 실측(라이트, 밴드 뒤에 깔 때 셀 숫자 배경대비):
  //     밴드없음 5.30 · t<=0.2 4.59 · t=0.5 3.93 · t=1.0 2.56  (전부 진한 셀 0.66 기준)
  //   densityClip 이 90분위라 t>=1.0 은 빈의 10% 뿐이다 -- 대부분은 옅어서 거의 안 건드리고
  //   강한 군집에서만 흐려진다(그 자리는 어차피 눈에 띄어야 한다).
  //   더 거슬리면 «t 하한을 둬 약한 밴드는 안 그리기»가 다음 손잡이다.
  const slot = cw / N;   // 봉 한 칸(px)
  const xAt = (i) => ml + (i + X0) * slot;
  const yAt = (v) => mt + ((yMax - v) * ch) / ySpan;
  const pyAt = PROF ? (v) => PROF.top + ((yMax - v) * (PROF.bot - PROF.top)) / ySpan : yAt, pTop = PROF ? PROF.top : mt, pBot = PROF ? PROF.bot : plotBottom;   // 체결·호가 띠 자리(휴대폰 = 아래 칸)
  const bw = slot * 0.8;
  if (svg.id === "candleSvgSnapshot") chartGeo = { ml, cw, n: N, wAll, hAll, mt, pb: plotBottom, prof: PROF && [PROF.top - 16, PROF.bot], ymin: yMin, ymax: yMax };   // 끌기·휠이 화면 px → 봉(chartSlotPx)
  // 2026-10-20 1주일 창(5분봉 2,016개 · 봉 칸 ~0.5px): 봉마다 노드(캔들 <g>·선·사각형, 레인 막대 7개, 청산 원, 제목)를 만들면
  //   3.7만 노드 · 렌더 180ms 였다. 봉 칸이 4px 밑이면 같은 색끼리 <path> 하나로 모으고 봉별 <title> 은 뺀다(호버 툴팁이 그 봉 값을 말한다).
  // 2026-10-20 연속 줌(36~300봉이 «무거운 구간»): 문턱 4 → 8px. 4~8px 봉은 박스 없는 캔들이라 path 로 모아도 같은 그림(아래 DENSE 캔들 폭).
  const DENSE = slot < 8;
  // 2026-10-20 캔버스로 옮기며 «같은 색끼리 path 하나로 모으기»(denseAdd)는 뺐다 -- 캔버스는 칠하기가 곧 픽셀이라 모을 노드가 없다.
  //   DENSE 는 «무엇을 그리나»(박스 없는 캔들 · 1px 밑 열 합치기 · 작은 청산 원)만 정한다.

  // ── 계층 캐시 (2026-09-20) ────────────────────────────────────────────────
  // 봉 셀은 이미 봉별로 캐시한다. 남은 것은 **가격 플롯 바깥의 층들**이다 -- 격자·x눈금·
  // OI 레인·청산 레인·청산밀도 히트맵. 이들도 매 렌더 새로 만들고 있었다.
  // 실측(틱 스트림, 셀 캐시가 다 맞은 정상 상태에서 만들어지는 노드):
  //     풋프린트 4h  372개/렌더 (rect 114 · title 160 · text 70)
  //     청산맵  1h  574개/렌더 (rect 301 -- 밀도 히트맵이 압도적)
  // 그런데 이 층들의 **입력은 15~60초마다** 바뀐다(청산맵 60s · OI 15s).
  //
  // 🔴키는 «기하 + 봉 시각 + 그 층의 데이터 신원» 셋이다. 기하를 빼면 축이 움직여도 낡은
  //   픽셀이 남고, 봉 시각을 빼면 봉이 한 칸 밀려도 그대로 남는다.
  // 🔴현재가·진행봉 OHLC 는 **일부러 안 넣는다**. 넣으면 틱마다 전부 깨져 캐시가 무의미해진다.
  //   그래서 OHLC 에 의존하는 것(가격 라벨·이벤트 삼각형)은 **캐시하지 않고** 매번 그린다.
  // 🔴테마를 서명에 넣는다 -- 밀도 색표·풋프린트 셰이드·잉크 불투명도가 전부 테마 파생이라,
  //   빠뜨리면 테마를 바꿔도 캐시된 층이 옛 색 그대로 남는다(모드 키와 같은 함정).
  // 2026-10-20 봉 셀·밀도·레인·청산 원은 캔버스(fpCv)로 갔다 -- svg 층 캐시는 x눈금(grid)·호가 띠(bookStrip)만 남고, 같은 키가 밀도 칸 목록 기억(densSig)에 쓰인다.
  const themeSig = document.documentElement.getAttribute("data-theme") || "dark";
  const baseGeomSig = [themeSig, w, h, mt, ch, ml, mr, cw, bw, X0, yMin, yMax, plotBottom,
                       mobileChart, quadY, cumY, QUAD_H, CUM_H, CUM_DRAW_H,
                       PRICE_ROW_H].join("|");
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

  const isSnapshotChart = svg.id === "candleSvgSnapshot";
  // 가격 플롯 클립(네이티브 clipPath) -- 시장 맥락 선·풋프린트 칸이 같이 쓴다. 2026-10-10 세로 끌기(사용자 «풋프린트도 위아래»)로 칸이
  //   사분면·레인 줄까지 넘쳐 겹쳐서 칸·델타·POC 선도 여기에 자른다.
  const mkClip = (name, y, ht) => {
    const clipId = (svg.id || "chart") + name;
    let cr = svg.querySelector("clipPath#" + clipId + " rect");
    if (!cr) {
      const defs = document.createElementNS(NS, "defs"), cp = document.createElementNS(NS, "clipPath");
      cp.setAttribute("id", clipId); cr = document.createElementNS(NS, "rect");
      cp.appendChild(cr); defs.appendChild(cp); svg.appendChild(defs);
    }
    cr.setAttribute("x", ml); cr.setAttribute("y", y); cr.setAttribute("width", cw); cr.setAttribute("height", ht);   // 창·폭이 바뀌면 따라간다
    return `url(#${clipId})`;
  };
  const plotClip = isSnapshotChart ? mkClip("-mcclip", mt, ch) : null;
  // 2026-10-20 끌기 연속: 양 끝에 반쯤 걸친 봉이 레인(사분면·막대)에서 왼쪽 여백·오른쪽 값 칸으로 삐져나오지 않게 가로만 자른다
  const laneClip = isSnapshotChart ? mkClip("-laneclip", plotBottom, hAll - plotBottom) : null;
  const plotC = plotClip && [ml, mt, cw, ch], laneC = laneClip && [ml, plotBottom, cw, hAll - plotBottom];   // 같은 자르기의 캔버스판

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
  // 2026-10-20 데이터에서만 나오는 것(90분위 · 정렬 · 가격 합집합 · 스냅샷별 가중)은 payload 신원으로 한 번만 -- 끌기·휠은 매 프레임 다시 그린다(스냅샷 144 × 가격 152 정렬이 렌더마다 돌았다)
  const dm = renderCandleSvg._dens || (renderCandleSvg._dens = { h: undefined });
  if (dm.h !== densityHistory) {
    const densityValues = (densityHistory || [])
      .flatMap(snap => (snap.bins || []).map(b => b.weightPct || 0))
      .filter(v => v > 0)
      .sort((a, b) => a - b);
    dm.clip = densityValues.length
      ? densityValues[Math.min(densityValues.length - 1, Math.floor(densityValues.length * DENSITY_PERCENTILE_CLIP))]
      : 1;
    dm.sorted = (densityHistory || []).slice().sort((a, b) => a.tsMs - b.tsMs);
    dm.union = Array.from(new Set(dm.sorted.flatMap(snap => (snap.bins || []).map(b => b.price))));
    dm.w = dm.sorted.map((snap) => new Map((snap.bins || []).map(b => [b.price, b.weightPct || 0])));
    dm.h = densityHistory;
  }
  const densityClip = dm.clip;
  const drawDensitySeg = (x0, x1, top, bottom, t) => {
    if (x1 <= x0) return;
    // 밀도 0 인 칸은 **그리지 않는다**. 칠하면 패널 위에 띠로 남고(2026-09-12 b5a4790 으로
    // 패널이 밝아진 뒤 «어두운 구멍» 으로 드러났다), 안 그리면 배경이 그대로 비쳐 패널의
    // 세로 그라디언트가 어떻든 정확히 녹는다. 알파를 t 에 비례시키는 건 잘못된 고침이었다 --
    // densityClip 이 양수 밀도의 90분위라 대부분의 칸이 t<0.25 에 몰려 히트맵 전체가 흐려졌다.
    if (!(t > 0)) return;
    // 2026-10-20 농도는 64단(눈에 같은 색) -- 아래 이어 붙이기가 같은 단끼리만 잇는다.
    // 두 모드가 **같은 알파**를 쓴다 -- 풋프린트는 셀과 겹치지 않는 게이트에 그리므로
    // 낮출 이유가 없고, 낮추면 같은 값이 두 색으로 보인다(위 전체폭 주석의 실측).
    //   같은 줄(가격)에서 바로 이어지는 같은 농도 칸은 한 사각형으로 늘린다(densRun -- 지도는 천천히 변해 144봉 2.2만 칸이 몇 분의 1 로 준다).
    const q = Math.round(t * 64), k = top + "|" + bottom, r = densRun.get(k);
    if (r && r[2] === q && Math.abs(r[1] - x0) < 0.01) { r[1] = x1; return; }
    if (r) densRunPut(r);
    densRun.set(k, [x0, x1, q, top, bottom]);
  };
  const densRun = new Map();
  const densRects = [];   // 2026-10-20 bg 캔버스가 칠한다(아래 «density» 칠하기)
  const densRunPut = ([x0, x1, q, top, bottom]) => densRects.push({ fill: densityColor(q / 64), "fill-opacity": 0.85, x0, x1, top, bottom });
  const sortedDensityHistory = dm.sorted;
  let dbi = 0;   // 스냅샷·봉 둘 다 시각순 -- 한 번 훑기(2026-10-20 1주일 창: 144 × findIndex 2천 봉이 렌더마다 돌았다)
  const densityBoundaryIdx = sortedDensityHistory.map((snap) => {
    const tsSec = Math.floor((snap.tsMs || 0) / 1000);
    while (dbi < candles.length && candles[dbi].time < tsSec) dbi++;
    return dbi;
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
  // B: 마지막 스냅샷이 모르는 구간(ts 이후 -- 10-11 전엔 ts+1h)에 가격이 지나간 칸 -> 그 봉 가운데서 끊고 뒤는 «쓸림»(t=0)으로.
  //   앞 스냅샷들은 자기 시간봉까지 이미 반영돼 있다. 키에 넣어야 틱으로 새로 쓸릴 때 층이 다시 그려진다.
  const densitySweep = new Map();
  const lastSnap = sortedDensityHistory[sortedDensityHistory.length - 1];
  if (lastSnap) {
    const since = tfT(Math.floor((lastSnap.tsMs || 0) / 1000));   // 30m·1h = 그 시각이 든 tf 봉부터 · 2026-10-11 스냅샷 ts = 아는 시각(5분 경계) -- 그 뒤 봉 = 형성 5분봉뿐
    (lastSnap.bins || []).forEach((b) => {
      if (!(b.weightPct > 0)) return;
      const i = liqSweepIdx(candles, since, b.price);
      if (i >= 0) densitySweep.set(b.price, i);
    });
  }
  // 2026-10-20 bg 캔버스(fpCv): 칸 목록은 옛 층 캐시와 같은 키(기하·봉 시각·밀도 신원·쓸림)로 기억하고 칠하기는 매 렌더(싸다).
  //   자르기 = 가격 플롯(끌기 연속: 걸친 봉 열이 왼쪽 여백으로 나가지 않게) · 가장자리는 장치 픽셀로 반올림 = 옛 shape-rendering crispEdges
  //   (2026-10-11 열이 5분봉마다 -- 붙은 칸 사이 안티앨리어싱 틈이 봉마다 세로 줄무늬로 보였다).
  const densSig = baseGeomSig + "|" + timesSig + "|" + objToken(densityHistory) + ":" + densityClip + ":" + [...densitySweep].join(",");
  const densMemo = renderCandleSvg._densRects;
  if (sortedDensityHistory.length && densMemo && densMemo.sig === densSig) densRects.push(...densMemo.rects);
  else if (sortedDensityHistory.length) {
  const densityPriceUnion = dm.union;
  let gx0 = null;   // 2026-10-20 열이 1.5px 밑(넓은 창)이면 다음 스냅샷과 한 열로 합친다 -- 색은 묶음의 마지막(가장 최근) 스냅샷. 1주일 창 2.2만 칸 → 천여 칸, 눈에 같은 그림
  sortedDensityHistory.forEach((snap, si) => {
    const xStartIdx = densityBoundaryIdx[si];
    const xEndIdx = si + 1 < densityBoundaryIdx.length ? densityBoundaryIdx[si + 1] : candles.length;
    if (xEndIdx <= xStartIdx) return; // this snapshot's hour has no visible candle (off-screen)
    if (gx0 == null) gx0 = xAt(xStartIdx);
    const x0 = gx0, x1 = xAt(xEndIdx);
    if (x1 - x0 < 1.5 && si < sortedDensityHistory.length - 1) return;
    gx0 = null;
    const weightByPrice = dm.w[si];
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
        drawDensitySeg(x0, xc, top, bottom, t);
        drawDensitySeg(xc, x1, top, bottom, 0);
      } else {
        drawDensitySeg(x0, x1, top, bottom, t);
      }
    });
  });
  densRun.forEach(densRunPut); densRun.clear();
  renderCandleSvg._densRects = { sig: densSig, rects: densRects.slice() };
  }
  if (densRects.length) bgI.push({ ...CV_ALL, c: plotC, p: (ctx, q) => {
    const S = fpCv.S, ox = fpCv.ox, oy = fpCv.oy;
    ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = 0.85;
    for (const d of densRects) {
      if (q && (d.x1 < q.x0 || d.x0 > q.x1 || d.bottom < q.y0 || d.top > q.y1)) continue;
      const X0 = Math.round(ox + d.x0 * S), X1 = Math.round(ox + d.x1 * S), Y0 = Math.round(oy + d.top * S), Y1 = Math.round(oy + d.bottom * S);
      if (X1 > X0 && Y1 > Y0) { ctx.fillStyle = d.fill; ctx.fillRect(X0, Y0, X1 - X0, Y1 - Y0); }
    }
    ctx.restore();
  } });

  // Resistance/support/current/entry price tags -- computed here (before the axis ticks below) so
  // the tick loop can tell when a grid label would land on top of one of these and skip it.
  // Rendering (the actual lines/boxes) still happens later, after candles/markers, so paint order
  // is unchanged.
  const priceLabels = [];
  let cvLiveLine = null;   // 2026-10-20 칸 뒤 현재가 선(bg 캔버스 칠하기) -- 빠른 갱신이 y 만 바꿔 띠를 다시 칠한다
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

  // Grid & Y-Axis Ticks -- 2026-10-20 가로 격자선은 bg 캔버스(밀도 위 · 칸 뒤 선 아래 = 옛 순서). .chart-grid = stroke var(--line) · 1 · opacity .82
  const gridYs = axisTicks(yMin, yMax, 6).map(yAt).filter((y) => y >= mt && y <= plotBottom);   // 2026-10-06 눈금이 범위 밖으로 나오면 레인·모의 판 칸을 가로지르는 선이 됐다(사용자 신고)
  bgI.push({ ...CV_ALL, p: (ctx) => gridYs.forEach((y) => cvHLine(ctx, ml - TRADE_L, w - mr, y, "var(--line)", 0.82)) });
  const bgBehind = bgI.length;   // 칸 뒤 선(현재가·max pain·VWAP 등)은 여기(격자 바로 위)에 끼운다 -- 옛 svg.insertBefore(…, gridG.nextSibling)
  // x눈금은 기하와 봉 시각만 보므로 한 층으로 묶어 캐시한다(cachedLayer 주석).
  // 2026-09-09 사용자 요청: **y축 가격 눈금 라벨을 없앤다**(격자선은 유지).
  //   현재/롱익절/지지선 같은 **라인 태그**는 priceLabels 로 계속 그린다 -- 그쪽이 실제로
  //   읽는 값이고, 눈금 숫자는 같은 오른쪽 열에서 그 태그와 자리를 다투기만 했다.
  //   (그래서 있던 collidesWithPriceTag 충돌 회피도 함께 사라진다 -- 눈금이 없으면 충돌도 없다)
  cachedLayer("grid", "", (g) => {

  const xTickCount = isMobileChartMode() ? 4 : 6;
  const lastIdx = candles.length - 1;
  const dated = lastIdx > 0 && candles[lastIdx].time - candles[0].time >= 86400;   // 하루 넘는 창 = 날짜도(아래)
  const minTickPx = (mobileChart ? 46 : 52) * (dated ? 1.7 : 1);   // 날짜 붙은 글자(«10/20 17:40»)는 1.7배 넓다
  // 2026-10-20 끌기·휠 연속: 눈금을 창 첫 봉부터 세면 소수 봉만 움직여도 글자가 다른 봉으로 튄다 -- **시각 격자**(현지 시:분이 딱 떨어지는 봉)에 둔다.
  //   간격 = 봉 시간의 배수 중 칸 ≥ 플롯/(눈금 수+1) 인 가장 작은 «둥근» 시간(10분·15분·30분·1시간…1일).
  const barSec = candles.length > 1 ? candles[1].time - candles[0].time : 300, tzOff = new Date().getTimezoneOffset() * 60;
  const tickSec = [300, 600, 900, 1800, 3600, 7200, 10800, 14400, 21600, 43200, 86400, 172800, 345600]
    .find((q) => q % barSec === 0 && (q / barSec) * slot >= cw / (xTickCount + 1)) || 604800;
  const xTickIndexes = [];
  candles.forEach((c, i) => { const x = xAt(i) + bw / 2; if (x >= ml && x <= ml + cw && (c.time - tzOff) % tickSec === 0) xTickIndexes.push(i); });
  // 실시간(오른쪽 끝 = 최신 봉)이면 최신 봉 시각도 -- 앞 눈금과 너무 붙으면(2026-08-27 겹침 신고) 앞 것을 그 자리로 옮긴다.
  if (!(svg.id === "candleSvgSnapshot" && chartPanEnd != null) && lastIdx >= 0 && !xTickIndexes.includes(lastIdx)) {
    if (xTickIndexes.length && xAt(lastIdx) - xAt(xTickIndexes[xTickIndexes.length - 1]) < minTickPx) xTickIndexes[xTickIndexes.length - 1] = lastIdx;
    else xTickIndexes.push(lastIdx);
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
    txt.setAttribute("y", h - mb + 21);
    txt.setAttribute("text-anchor", "middle");
    txt.setAttribute("font-size", isMobileChartMode() ? "12" : "13");
    txt.setAttribute("font-weight", "700");
    txt.setAttribute("fill", "var(--muted)");
    // 2026-10-20 하루 넘는 창(1주일 휠)은 날짜도 -- 시:분만이면 어느 날인지 모른다
    const dd = new Date(c.time * 1000);
    txt.textContent = (dated ? `${dd.getMonth() + 1}/${dd.getDate()} ` : "") + fmtDateTick(c.time * 1000);
    // 2026-10-20 왼쪽 끝 눈금은 글자 반폭만큼 안으로(전에는 첫 봉이면 start 정렬 -- 끌기로 눈금이 미끄러지면 정렬이 바뀌며 튄다)
    txt.setAttribute("x", Math.max(x, 2 + textW(txt) / 2));
    g.appendChild(txt);
  });

  });

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
    // 🔴2026-10-11 행 경계 = **버킷 가장자리**. 서버 칸 키는 round(가격/버킷) = 버킷 중심이라 [중심−½버킷, 중심+½버킷) 이 그 칸의 가격이다.
    //   floor(중심/행)으로 행 [k, k+1] 에 넣던 때는 칸이 반 버킷($0.25) 위로 그려져 고가 위로 삐져나오고 저가 칸이 비었다
    //   (1920·1h 실측: 고저가 맨 위·아래 칸 밖 11/12봉). 행 k = [k·행 − off, (k+1)·행 − off) -- 행이 버킷의 배수여도 경계는 늘 버킷 가장자리.
    //   칸·POC·델타·스택·체결 기둥·포지션 기둥·호가 띠·툴팁이 이 격자 하나를 쓴다(test/render_fp_row_align_20261011.py).
    const off = bucket / 2, rowOf = (price) => Math.floor((price + off) / rowSize), rowLo = (k) => k * rowSize - off;
    fpRowOff = off;

    // ── 체결 프로파일 (2026-09-28 사용자 선택 시안 B · 모바일 A) ─────────────────────────
    //   보이는 봉의 셀을 **이 행 크기로** 가격별로 합친다 -- 옛 오른쪽 열 상자(/api/supply-profile)를 대체한다.
    //   원천·창·행이 풋프린트와 같아 둘이 어긋날 수 없다(옛 상자는 재시작 뒤 OKX 가 덮은 봉부터만 셌다).
    //   데스크톱: 풋프린트와 호가 띠 사이 기둥, 호가 띠 바닥선에서 **왼쪽으로**(체결 ← ┃ → 호가) -- 같은 줄 = 같은 가격이라
    //   «여기서 얼마나 거래됐나 vs 지금 얼마나 걸려 있나»를 한 줄에서 맞댄다. 모바일: 가격 글자와 캔들 사이 띠(TRADE_L), 캔들 왼쪽 끝에서 **왼쪽으로**(호가 띠의 거울).
    //   막대 = 행 총 체결(매수+매도) · 안쪽부터 고래·중형·리테일 농담 · POC(최다 체결) 행은 테두리.
    //   🔴체결 합산에는 방향이 없다(누가 사면 누가 판다) -- 지지·저항으로 읽지 않는다. 방향은 봉별 델타가 말한다.
    // ── 체결 | 포지션 토글 (2026-10-10 사용자 «체결/호가 프로파일 바로 위에 체결과 포지션 토글» · 시안 A + C 미니맵) ──────────
    //   포지션 = 최근 1·7·30일에 열려 아직 남은 계약(추정 · 2026-10-22 롱·숏 분리 근사, 시안 A). 체결 기둥과 같은 자리·같은 행·같은 바닥선에서 왼쪽으로.
    //   풋프린트 가격 폭이 좁아 대부분이 화면 밖이다 -- 위·아래 끝에 «화면 밖 몫», 기둥 왼쪽 10px 에 창 전체 미니맵 + 보이는 범위 괄호.
    //   🔴지도다. 덩어리 저항·재방문·고통 지수는 매매 검정 불통과(= VWAP 평균회귀 재표현, 2026-10-09).
    //   2026-10-27 휴대폰(사용자 «모바일도 체결/포지션/청산 프로파일»): 같은 기둥을 아래 칸(PROF) 왼쪽 절반에 -- colX0·colW·pTop·pBot·pyAt 이 그 자리다.
    const colX0 = PROF ? ml + 2 : tradeX0, colW = PROF ? PROF.cx - PROF_GAP - 4 - colX0 : TRADE_W;
    //   2026-10-28 펜 메모: 왼쪽 기둥(colX0·colW) = 체결|청산 · 오른쪽(호가 띠 자리 bkX0·bkW) = 포지션|호가 -- 포지션 그림은 자리만 바꿔 같은 코드
    const bkX0 = PROF ? PROF.cx + PROF_GAP : tradeX0 + TRADE_W + 6, bkW = PROF ? w - mr - bkX0 - 8 : BOOK_W - 10;   // = 아래 bookInfo 의 x0·L
    const posOn = !!((TRADE_W || PROF) && (BOOK_W || PROF) && fpColR === "pos" && flowOn()), liqOn = !!((TRADE_W || PROF) && fpColL === "liq" && flowOn());
    post.push(() => placeFpColTog(svg, w - mr, PROF ? h + SUB_GAP + 2 : mt - 16, !!((TRADE_W || PROF) && flowOn()
      && !(typeof window.fpBubbleActive === "function" && window.fpBubbleActive())
      && !(typeof window.fpDailyActive === "function" && window.fpDailyActive()))));   // 버블·일봉이 덮을 땐 숨김 · 화면 상자를 재므로 끝에서(post)
    if (posOn) {
      const colX0 = bkX0, colW = bkW;   // 오른쪽 기둥(호가 자리)에 -- 바닥선 = 기둥 오른쪽 끝, 왼쪽으로 자란다
      const P = latestPositionProfile, W = P && P.win && P.win[String(fpPosWin)];
      const anchor = colX0 + colW + 4, Lw = colW - 10, g = document.createElementNS(NS, "g");
      const rect = (x, y, wd, ht, fill, op) => {
        const r = document.createElementNS(NS, "rect");
        r.setAttribute("x", x); r.setAttribute("y", y); r.setAttribute("width", Math.max(0, wd)); r.setAttribute("height", Math.max(1, ht));
        r.setAttribute("fill", fill); r.setAttribute("fill-opacity", op); g.appendChild(r); return r;
      };
      const label = (x, y, str, fill) => {
        const t = document.createElementNS(NS, "text");
        t.setAttribute("x", x); t.setAttribute("y", y); t.setAttribute("text-anchor", "end"); t.setAttribute("font-size", "10"); t.setAttribute("font-weight", "700");
        t.setAttribute("fill", fill); t.setAttribute("paint-order", "stroke"); t.setAttribute("stroke", "var(--chart-bg)"); t.setAttribute("stroke-width", "3");
        t.textContent = str; g.appendChild(t);
      };
      if (!W || !(W.tot > 0)) {
        label(anchor - 4, pTop + (PROF ? 14 : 62), P && P.win && P.win[String(fpPosWin)] ? "포지션 데이터 없음" : "포지션 불러오는 중…", "var(--muted)");
      } else {
        const toRows = (V) => {
          const m = new Map();
          V.vals.forEach((q, j) => {   // $bw 칸을 걸치는 행마다 겹친 길이만큼 나눈다(행이 칸보다 잘면 한 줄씩 비던 것, 10-10 배포 후)
            const p0 = (V.lo + j) * P.bw, p1 = p0 + P.bw;
            for (let k = rowOf(p0); rowLo(k) < p1; k++) {
              const ov = Math.min(p1, rowLo(k + 1)) - Math.max(p0, rowLo(k));
              if (ov > 0) m.set(k, (m.get(k) || 0) + q * ov / P.bw);
            }
          });
          return m;
        };
        const rows = toRows(W);
        const vis = (k) => rowLo(k + 1) > yMin && rowLo(k) < yMax;
        // 롱·숏 분리(추정, 2026-10-22 사용자 «포지션을 롱/숏 비율대로» · 시안 A 이어 붙인 막대) -- 손바뀜을 직전 1h·24h 수익으로 가른 근사
        //   (HL 채점: 롱−숏 평균 진입 차 부호 일치 1일 98% · 7일 51%). 기둥 바닥선부터 롱(초록) → 숏(빨강), 길이 = 롱+숏.
        //   물린 쪽(롱 = 현재가 위 · 숏 = 현재가 아래)은 진하게, 수익 쪽은 옅게. 서버가 long/short 를 안 주면 옛 롱=숏 막대.
        const side = W.long && W.short && W.long.tot > 0 && W.short.tot > 0;
        const rL = side ? toRows(W.long) : null, rS = side ? toRows(W.short) : null;
        const UNDER = 0.92, OVER = 0.38;
        let mxq = 0; rows.forEach((q, k) => { if (vis(k) && q > mxq) mxq = q; });
        let mxSide = 0;
        if (side) rows.forEach((q, k) => { if (vis(k)) mxSide = Math.max(mxSide, (rL.get(k) || 0) + (rS.get(k) || 0)); });
        rows.forEach((q, k) => {
          if (!vis(k) || !(mxq > 0)) return;
          const yTop = Math.max(pTop, pyAt(rowLo(k + 1))), yBot = Math.min(pBot, pyAt(rowLo(k)));
          if (yBot - yTop < 1) return;
          const y0 = yTop + 0.5, h = yBot - yTop - 1, mid = (rowLo(k) + rowLo(k + 1)) / 2, up = mid > currentPrice;
          if (!side) { const wd = Lw * q / mxq; rect(anchor - wd, y0, wd, h, "var(--position)", 0.9); return; }
          if (!(mxSide > 0)) return;
          const wa = Lw * (rL.get(k) || 0) / mxSide, wb = Lw * (rS.get(k) || 0) / mxSide;
          rect(anchor - wa, y0, wa, h, "var(--good)", up ? UNDER : OVER);
          rect(anchor - wa - wb, y0, wb, h, "var(--bad)", up ? OVER : UNDER);
        });
        let above = 0, below = 0;
        W.vals.forEach((q, j) => { const p = (W.lo + j + 0.5) * P.bw; if (p > yMax) above += q; else if (p < yMin) below += q; });
        const pct = (x) => Math.round(100 * x / W.tot) + "%";
        let topY = pTop + (PROF ? 12 : 60);   // 2026-10-28 데스크톱 오른쪽 기둥 위 = 지지·저항 꼬리표 자리(두 개 쌓여 40 으로는 겹쳤다)
        if (side) {
          const upShare = (V) => { let u = 0; V.vals.forEach((q, j) => { if ((V.lo + j + 0.5) * P.bw > currentPrice) u += q; }); return u / V.tot; };
          const uL = upShare(W.long), uS = upShare(W.short);
          label(anchor - 4, topY, `롱 위 ${Math.round(100 * uL)}% · 아래 ${Math.round(100 * (1 - uL))}%`, "var(--good)"); topY += 12;
          label(anchor - 4, topY, `숏 위 ${Math.round(100 * uS)}% · 아래 ${Math.round(100 * (1 - uS))}%`, "var(--bad)"); topY += 12;
        }
        if (above / W.tot >= 0.01) label(anchor - 4, topY, "↑ 화면 위 " + pct(above), "var(--muted)");
        if (below / W.tot >= 0.01) label(anchor - 4, pBot - 4, "↓ 화면 아래 " + pct(below), "var(--muted)");
        if (side && currentPrice > 0) {                                   // 롱·숏 평균 진입 두 줄 -- 가까우면 높은 쪽 글자는 위, 낮은 쪽은 아래
          const hiAvg = Math.max(W.long.avg, W.short.avg);
          [[W.long, "롱", "var(--good)"], [W.short, "숏", "var(--bad)"]].forEach(([V, nm, col]) => {
            if (!V.avg) return;
            const dv = (V.avg / currentPrice - 1) * 100, txt = `${nm} 평균 ${V.avg.toFixed(pxDp())} (${dv >= 0 ? "+" : ""}${dv.toFixed(1)}%)`;
            if (V.avg < yMax && V.avg > yMin) {
              const ya = pyAt(V.avg), l = document.createElementNS(NS, "line");
              l.setAttribute("x1", colX0 + 28); l.setAttribute("x2", anchor); l.setAttribute("y1", ya); l.setAttribute("y2", ya);
              l.setAttribute("stroke", col); l.setAttribute("stroke-dasharray", "4 3"); l.setAttribute("stroke-width", "1.4"); g.appendChild(l);
              label(anchor - 4, Math.min(Math.max(V.avg === hiAvg ? ya - 4 : ya + 12, topY + 12), pBot - 30), txt, col);   // 머리 줄·바닥 줄과 안 겹치게
            } else label(anchor - 4, V.avg > yMax ? topY + (nm === "롱" ? 0 : 12) : pBot - 17 - (nm === "롱" ? 0 : 12), (V.avg > yMax ? "↑ " : "↓ ") + txt, col);
          });
        } else if (W.avg && currentPrice > 0) {
          const dv = (W.avg / currentPrice - 1) * 100, avTxt = "평균 진입 " + W.avg.toFixed(pxDp()) + " (" + (dv >= 0 ? "+" : "") + dv.toFixed(1) + "%)";
          if (W.avg < yMax && W.avg > yMin) {
            const ya = pyAt(W.avg), l = document.createElementNS(NS, "line");
            l.setAttribute("x1", colX0 + 28); l.setAttribute("x2", anchor); l.setAttribute("y1", ya); l.setAttribute("y2", ya);
            l.setAttribute("stroke", "var(--position)"); l.setAttribute("stroke-dasharray", "4 3"); l.setAttribute("stroke-width", "1.4"); g.appendChild(l);
            label(anchor - 4, ya - 4, avTxt, "var(--position)");
          } else label(anchor - 4, W.avg > yMax ? pTop + 25 : pBot - 17, (W.avg > yMax ? "↑ " : "↓ ") + avTxt, "var(--position)");
        }
        // 미니맵(시안 C): 기둥 왼쪽 16px 뒤 10px 띠 = 이 창 전체 분포 · 테두리 = 지금 보이는 가격 범위 · 가로선 = 현재가
        const lo = W.lo * P.bw, hi = (W.lo + W.vals.length) * P.bw, mmX = colX0 + 16, mmW = 10, H = pBot - pTop;
        if (hi > lo) {
          const my = (p) => pBot - (p - lo) / (hi - lo) * H, vmax = Math.max(...W.vals), bh = Math.max(1, H / W.vals.length);
          W.vals.forEach((q, j) => { if (q > 0) rect(mmX + mmW - mmW * q / vmax, my(lo + (j + 1) * P.bw), mmW * q / vmax, bh, "var(--position)", 0.75); });
          const b0 = my(Math.min(hi, yMax)), b1 = my(Math.max(lo, yMin));
          if (b1 > b0) { const fr = rect(mmX - 1.5, b0, mmW + 3, b1 - b0, "none", 1); fr.setAttribute("stroke", "var(--ink)"); fr.setAttribute("stroke-width", "1.2"); }
          if (currentPrice > lo && currentPrice < hi) {
            const yc = my(currentPrice), l2 = document.createElementNS(NS, "line");
            l2.setAttribute("x1", mmX - 3); l2.setAttribute("x2", mmX + mmW + 3); l2.setAttribute("y1", yc); l2.setAttribute("y2", yc);
            l2.setAttribute("stroke", "var(--ink)"); l2.setAttribute("stroke-width", "1.5"); g.appendChild(l2);
          }
          posInfo = { rows, rL, rS, rowSize, off, tot: W.tot, win: fpPosWin, x0: colX0 + 28, x1: anchor, mm: { x0: mmX - 3, x1: mmX + mmW + 3, my, lo, hi, W, bw: P.bw }, hours: W.hours };
        }
      }
      const ln = document.createElementNS(NS, "line");
      ln.setAttribute("x1", anchor + 1); ln.setAttribute("x2", anchor + 1); ln.setAttribute("y1", pTop); ln.setAttribute("y2", pBot);
      ln.setAttribute("stroke", "var(--soft-line)"); g.appendChild(ln);
      svg.appendChild(g);
    }
    // ── 청산 기둥 (2026-10-27 사용자 «체결/포지션 칸에 청산을 만들어서 청산맵을 풋프린트에 붙여줘» · «청산맵» 보기 탭은 뺐다) ──────────
    //   옛 청산맵 보기의 지도(liq_profile.js · 1d = /api/liquidation-map · 7·30d = /tiers)를 풋프린트 행 격자에 얹는다. 지나간 칸은 지운 값(fpLiqColTp).
    //   막대 = 그 행의 추정 청산(바닥선부터 10·25·50·100배 쌓기) · 선 = 현재가에서 그 가격까지 누적(위 = 숏 청산 초록 · 아래 = 롱 청산 빨강).
    //   🔴추정이지 실측 청산이 아니다. 벽 «뚫림»은 거리비가 거의 다 설명(2026-10-13) -- 지도로만.
    if (liqOn) {
      const tp = typeof window.fpLiqColTp === "function" ? window.fpLiqColTp(fpLiqWin) : null;
      const anchor = colX0 + colW + 4, Lw = colW - 10, g = document.createElementNS(NS, "g");
      const rect = (x, y, wd, ht, fill, op) => {
        const r = document.createElementNS(NS, "rect");
        r.setAttribute("x", x); r.setAttribute("y", y); r.setAttribute("width", Math.max(0, wd)); r.setAttribute("height", Math.max(1, ht));
        r.setAttribute("fill", fill); r.setAttribute("fill-opacity", op); g.appendChild(r); return r;
      };
      const label = (x, y, str, fill) => {
        const t = document.createElementNS(NS, "text");
        t.setAttribute("x", x); t.setAttribute("y", y); t.setAttribute("text-anchor", "end"); t.setAttribute("font-size", "10"); t.setAttribute("font-weight", "700");
        t.setAttribute("fill", fill); t.setAttribute("paint-order", "stroke"); t.setAttribute("stroke", "var(--chart-bg)"); t.setAttribute("stroke-width", "3");
        t.textContent = str; g.appendChild(t);
      };
      if (!tp || !tp.tiers || !tp.tiers.length || !(currentPrice > 0)) label(anchor - 4, pTop + 14, "청산맵 불러오는 중…", "var(--muted)");
      else {
        const bw = tp.bin_width, nT = tp.tiers.length, TV = ["--lev10", "--lev25", "--lev50", "--lev100"];
        const fu = (v) => tp.unit === "share" ? (v * 100).toFixed(v >= 0.1 ? 0 : 1) + "%" : v >= 1e9 ? "$" + (v / 1e9).toFixed(2) + "B" : fmtUsdCompact(v);
        const tot = tp.tiers[0].values.map((_, j) => tp.tiers.reduce((a, t) => a + (t.values[j] > 0 ? t.values[j] : 0), 0));
        // 누적 = 현재가와 그 가격 사이에 가운데가 든 칸의 합(칸 k 의 가운데 = k×bw, liq_profile.js liveTp)
        // up = 숏 쪽(현재가 위로)인지 명시 -- 위아래로 끌어 현재가가 화면 밖이면 «가격 > 현재가»로 쪽을 짐작하면 뒤집힌다(리뷰)
        const cum = (price, up) => tot.reduce((a, v, j) => { const c = (tp.lo + j) * bw; return a + (up ? c > currentPrice && c <= price : c < currentPrice && c >= price) * v; }, 0);
        const rows = new Map();   // 행 -> 배수 구간별 값
        tp.tiers.forEach((t, ti) => t.values.forEach((v, j) => {
          const p0 = (tp.lo + j - 0.5) * bw, p1 = p0 + bw;
          if (!(v > 0) || p1 <= yMin - rowSize || p0 >= yMax + rowSize) return;   // 줄 하나 여유 -- 화면 끝에 걸친 줄도 화면 밖 칸 몫까지 다 센다(검증에서 잘린 줄이 작게 나옴)
          for (let k = rowOf(p0); rowLo(k) < p1; k++) {   // 칸을 걸치는 행마다 겹친 길이만큼(포지션 기둥과 같은 나눔)
            const ov = Math.min(p1, rowLo(k + 1)) - Math.max(p0, rowLo(k));
            if (ov > 0) { const a = rows.get(k) || new Array(nT).fill(0); a[ti] += v * ov / bw; rows.set(k, a); }
          }
        }));
        rows.forEach((a, k) => { if (rowLo(k + 1) <= yMin || rowLo(k) >= yMax) rows.delete(k); });   // 여유로 생긴 화면 밖 줄은 최대·테두리에서 뺀다
        const sum = (a) => a.reduce((x, y) => x + y, 0);
        let mx = 0, topUp = null, topDn = null;
        rows.forEach((a, k) => {
          const q = sum(a), up = rowLo(k) + rowSize / 2 > currentPrice;
          if (q > mx) mx = q;
          if (up && (!topUp || q > topUp[1])) topUp = [k, q];
          if (!up && (!topDn || q > topDn[1])) topDn = [k, q];
        });
        // 누적 = 옛 청산맵과 같은 모양(면 10% + 2px 선 · 롱 빨강 / 숏 초록 · 두 쪽 같은 자) -- 면은 막대 아래, 선은 막대 위(좁은 기둥이라 가려지지 않게)
        // 현재가가 화면 밖이면 그 쪽 선은 없고, 반대쪽 선은 화면 끝(c0)에서 그때까지의 누적부터 시작한다
        const cU = currentPrice < yMax ? cum(yMax, true) : 0, cD = currentPrice > yMin ? cum(yMin, false) : 0, cmax = Math.max(cU, cD);
        const c0 = Math.min(Math.max(currentPrice, yMin), yMax), xAt = (c, up) => (anchor - Lw * cum(c, up) / cmax).toFixed(1);
        const curves = !(cmax > 0) ? [] : [[true, cU, "var(--good)"], [false, cD, "var(--bad)"]].filter(([, c]) => c > 0).map(([up, , col]) => {
          const y0 = pyAt(c0).toFixed(1), pts = [`${xAt(c0, up)},${y0}`];
          for (let k = rowOf(c0); up ? rowLo(k) < yMax : rowLo(k + 1) > yMin; k += up ? 1 : -1) {
            const b = up ? Math.min(rowLo(k + 1), yMax) : Math.max(rowLo(k), yMin);
            pts.push(`${xAt(b, up)},${pyAt(b).toFixed(1)}`);
          }
          const area = document.createElementNS(NS, "polygon");
          area.setAttribute("points", pts.concat(`${anchor},${pts[pts.length - 1].split(",")[1]}`, `${anchor},${y0}`).join(" "));
          area.setAttribute("fill", col); area.setAttribute("fill-opacity", "0.1"); g.appendChild(area);
          return [pts, col];
        });
        rows.forEach((a, k) => {
          const yTop = Math.max(pTop, pyAt(rowLo(k + 1))), yBot = Math.min(pBot, pyAt(rowLo(k)));
          if (yBot - yTop < 1 || !(mx > 0)) return;
          let cur = 0;
          a.forEach((v, ti) => { const wd = Lw * v / mx; if (wd > 0) { rect(anchor - cur - wd, yTop + 0.5, wd, yBot - yTop - 1, `var(${TV[Math.min(ti, 3)]})`, 0.9); cur += wd; } });
          if ((topUp && k === topUp[0]) || (topDn && k === topDn[0])) {   // 위·아래 가장 큰 행 = 테두리(체결 POC 와 같은 표시)
            const o = rect(anchor - cur, yTop + 0.5, cur, yBot - yTop - 1, "none", 1); o.setAttribute("stroke", "var(--ink)"); o.setAttribute("stroke-width", "1.2");
          }
        });
        curves.forEach(([pts, col]) => {
          const pl = document.createElementNS(NS, "polyline");
          pl.setAttribute("points", pts.join(" ")); pl.setAttribute("fill", "none"); pl.setAttribute("stroke", col); pl.setAttribute("stroke-width", "2");
          pl.setAttribute("stroke-linejoin", "round"); g.appendChild(pl);
        });
        // 누적 자 눈금(옛 청산맵 오른쪽 축) -- 기둥 왼쪽 끝 세로 점선 = 누적 자의 끝 값 · 머리·바닥 줄 = 화면 끝까지 누적 / 지도 전체(옛 범례)
        if (cmax > 0) {
          const x = anchor - Lw, t = document.createElementNS(NS, "line");
          t.setAttribute("x1", x); t.setAttribute("x2", x); t.setAttribute("y1", pTop); t.setAttribute("y2", pBot);
          t.setAttribute("stroke", "var(--line)"); t.setAttribute("stroke-dasharray", "2 4"); g.insertBefore(t, g.firstChild);
          label(x, pTop + 38, `누적 자 끝 ${fu(cmax)}`, "var(--muted)"); g.lastChild.setAttribute("text-anchor", "start");
        }
        const all = (up) => tot.reduce((a, v, j) => a + (up ? (tp.lo + j) * bw > currentPrice : (tp.lo + j) * bw < currentPrice) * v, 0);
        label(anchor - 4, pTop + 12, `누적 숏 청산 ↑ ${fu(cU)}`, "var(--good)");
        label(anchor - 4, pTop + 24, `지도 전체 ${fu(all(true))}`, "var(--muted)");
        label(anchor - 4, pBot - 16, `지도 전체 ${fu(all(false))}`, "var(--muted)");
        label(anchor - 4, pBot - 4, `누적 롱 청산 ↓ ${fu(cD)}`, "var(--bad)");
        liqInfo = { rows, rowSize, off, x0: colX0, x1: anchor, cum, fu, names: tp.tiers.map((t) => t.name), win: fpLiqWin };
      }
      const ln = document.createElementNS(NS, "line");
      ln.setAttribute("x1", anchor + 1); ln.setAttribute("x2", anchor + 1); ln.setAttribute("y1", pTop); ln.setAttribute("y2", pBot);
      ln.setAttribute("stroke", "var(--soft-line)"); g.appendChild(ln);
      svg.appendChild(g);
    }
    if (!liqOn) {
      const tr = new Map();   // 행 키 -> [총, 고래, 리테일, 매수, 매도]
      candles.forEach((c) => (footprint.byTime.get(c.time) || []).forEach((l) => {
        const k = rowOf(l[0]), a = tr.get(k) || [0, 0, 0, 0, 0];
        a[0] += (+l[1] || 0) + (+l[2] || 0); a[1] += (+l[3] || 0) + (+l[4] || 0);
        a[2] += (+l[5] || 0) + (+l[6] || 0); a[3] += +l[1] || 0; a[4] += +l[2] || 0;
        tr.set(k, a);
      }));
      let tmax = 0, pocK = null;
      tr.forEach((a, k) => { if (a[0] > tmax) { tmax = a[0]; pocK = k; } });
      if (tmax > 0) {
        const anchor = PROF ? PROF.cx - PROF_GAP : TRADE_W ? tradeX0 + TRADE_W + 4 : ml - 2;   // 데스크톱 = 호가 띠 바닥선 2px 앞 · 모바일 = 아래 칸 등뼈 왼쪽(왼쪽으로 자란다)
        const L = PROF ? anchor - ml - 2 : TRADE_W ? TRADE_W - 10 : TRADE_L - 6, alpha = 1;
        tradeInfo = { rows: tr, rowSize, off, max: tmax, pocK, x0: PROF ? ml : TRADE_W ? tradeX0 : ml - TRADE_L, x1: anchor };
        const g = document.createElementNS(NS, "g");
        tr.forEach((a, k) => {
          const yTop = Math.max(pTop, pyAt(rowLo(k + 1))), yBot = Math.min(pBot, pyAt(rowLo(k)));
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
        if (TRADE_W || TRADE_L || PROF) {         // 기둥 바닥선 -- 호가 띠 바닥선과 한 줄
          const ln = document.createElementNS(NS, "line");
          ln.setAttribute("x1", anchor + 1); ln.setAttribute("x2", anchor + 1);
          ln.setAttribute("y1", pTop); ln.setAttribute("y2", pBot);
          ln.setAttribute("stroke", "var(--soft-line)");
          g.appendChild(ln);
        }
        svg.appendChild(g);                        // 셀보다 먼저
      }
    }
    if (PROF) {   // 아래 프로파일 칸 가운데 등뼈 = 위·아래 끝 가격 + 현재가 점선(차트와 같은 범위라 높이로 맞대 읽는다)
      const g = document.createElementNS(NS, "g");
      const t = (y, str, fill, wt) => {
        const e = document.createElementNS(NS, "text");
        e.setAttribute("x", PROF.cx); e.setAttribute("y", y); e.setAttribute("text-anchor", "middle"); e.setAttribute("font-size", "10");
        e.setAttribute("font-family", "var(--font-mono)"); e.setAttribute("font-weight", wt); e.setAttribute("fill", fill);
        e.setAttribute("paint-order", "stroke"); e.setAttribute("stroke", "var(--chart-bg)"); e.setAttribute("stroke-width", "3");
        e.textContent = str; g.appendChild(e);
      };
      const yc = pyAt(currentPrice);
      if (currentPrice > 0 && yc > pTop && yc < pBot) {
        const ln = document.createElementNS(NS, "line");
        ln.setAttribute("x1", ml); ln.setAttribute("x2", w - mr); ln.setAttribute("y1", yc); ln.setAttribute("y2", yc);
        ln.setAttribute("stroke", "var(--ink)"); ln.setAttribute("stroke-opacity", "0.55"); ln.setAttribute("stroke-dasharray", "3 3");
        g.appendChild(ln);
        t(Math.max(pTop + 22, Math.min(pBot - 14, yc + 3.5)), fmtNum(currentPrice, pxDp()), "var(--ink)", 700);
      }
      t(pTop + 9, fmtNum(yMax, pxDp()), "var(--muted)", 500);
      t(pBot - 2, fmtNum(yMin, pxDp()), "var(--muted)", 500);
      svg.appendChild(g);
    }

    // 서버 버킷을 화면 행으로 묶는다. 서버는 0.5달러로만 모아 보내고, 행 크기는 여기서 정한다
    // -- 행을 바꾸려고 테이프를 다시 수집할 일이 없게.
    const rowsOfLv = (lv) => {
      const rows = new Map();
      (lv || []).forEach((lvl) => {
        const key = rowOf(lvl[0]);
        const cell = rows.get(key) || [0, 0];
        cell[0] += lvl[1]; cell[1] += lvl[2];
        rows.set(key, cell);
      });
      return rows;
    };
    const barRows = candles.map((c) => rowsOfLv(footprint.byTime.get(c.time)));
    // 배경 4단계는 매수·매도 각각의 최대로 나눈다(TradingView: "매수/매도 측은 별도로 계산").
    //   형성 봉을 뺀 최대(maxBuy0·maxSell0)도 둔다 -- 형성 봉 갱신(fpCv.live)이 새 최대를 이걸로 다시 잰다.
    const nB = candles.length, iL = nB - 1;
    let maxBuy = 0, maxSell = 0, maxBuy0 = 0, maxSell0 = 0;
    barRows.forEach((rows, i) => rows.forEach((cell) => {
      maxBuy = Math.max(maxBuy, cell[0]); maxSell = Math.max(maxSell, cell[1]);
      if (i < iL) { maxBuy0 = Math.max(maxBuy0, cell[0]); maxSell0 = Math.max(maxSell0, cell[1]); }
    }));
    const SHADES = footprintShades();
    const shade = (v, max) => SHADES[Math.min(3, Math.floor((max > 0 ? v / max : 0) * 4))];
    const fontPx = Math.min(9, Math.max(6, rowPx - 3));
    // 2026-10-10 사용자 «체결 박스 옆에 5분봉을 같이» → 시안 C(매도 | 5분봉 | 매수): 채운 몸통·심지를 두 칸 사이에.
    //   옆 캔들은 칸 폭에서 떼어 낸다(봉 사이 간격은 그대로) -- 폭 = 봉 칸의 14%(3~8px) + 틈 2px. 셀을 안 그리는 얇은 봉(bw<14)은 옛 테두리.
    //   (시안 때 쓴 "L"·"R" 분기는 남겨 둔다 -- 위치만 다른 같은 그림이다.)
    // 2026-10-20 연속 줌: 박스는 창(fpZoomed)이 아니라 **봉 폭**이 정한다 -- 12봉을 넘겨 넓혔는데 칸 숫자(showQty: half ≥ 18)가 안 들어가는 폭이면
    //   박스 없이 가운데 채운 캔들(fpThin). 12봉(버튼 자리)은 폭과 무관하게 늘 풋프린트. 휠로 좁히면 숫자가 들어가는 폭에서 박스가 돌아온다.
    const fpcWC = Math.max(3, Math.min(8, slot * 0.14)), fpThin = svg.id === "candleSvgSnapshot" && chartWindowBars > 12 + 1e-6 && Math.max(2, (bw - fpcWC - 2) / 2 - 0.5) < 18;
    if (svg.id === "candleSvgSnapshot") fpBoxesOff = fpThin;
    const FPC = bw >= 14 || fpThin ? "C" : "";
    // 2026-10-12 사용자 «모바일에서는 5분봉에 있는 체결박스들만 모두 제거 · 데스크탑은 수정하지 마»(실데이터 시안 B):
    //   휴대폰은 칸(매도·매수·불균형·POC 테두리)을 안 그리고, 비는 자리만큼 옆 5분봉을 칸 폭의 60% 로 키워 가운데에 둔다.
    //   델타·POC 선·밀도 층은 그대로(칸이 아니다). 데스크탑(isMobileChartMode 거짓)은 한 줄도 안 바뀐다.
    // 2026-10-20 넓힌 창(fpThin -- 봉이 얇아 칸 숫자가 안 들어갈 때)도 같은 그림: 체결 박스 없이 가운데 채운 캔들 · 델타·POC 선은 그대로.
    const MOB = isMobileChartMode() || fpThin;
    // 2026-10-21 사용자 «축소해서 체결 박스가 안 보이면 일반 캔들 차트처럼 캔들이 붙어 있게»: 박스 끈 넓힌 창(fpThin)은 몸통 = 봉 폭(칸의 80%) · 휴대폰 12봉(시안 B)은 60% 그대로
    const fpcW = FPC ? (fpThin ? bw : MOB ? Math.max(3, bw * 0.6) : fpcWC) : 0, fpcGap = FPC ? 2 : 0;
    const half = Math.max(2, (bw - fpcW - fpcGap) / 2 - 0.5);
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
    const drawCells = bw >= 14 && !MOB;
    // 라이트는 셀 배경이 밝아 글자를 거의 불투명하게 올려야 읽힌다(다크는 현행 0.82).
    const INK_OPACITY =
      document.documentElement.getAttribute("data-theme") === "light" ? "0.95" : "0.82";

    // ── 봉 그림 = fg 캔버스 칠하기 봉마다 하나 (2026-10-20 -- 옛 봉별 <g> 캐시·델타 노드 캐시 대체) ─────────────
    //   노드를 안 만드니 캐시도 없다: 전체 렌더는 다 다시 칠하고(48봉 칸 ~1ms), 형성 봉 갱신은 그 봉의 칠하기만 갈아 끼운다.
    //   옛 캐시 키(기하·행 크기·음영 기준·축)가 지키던 것은 이제 형성 봉 갱신의 «전체 렌더로 넘길 조건»(아래 fpCv.live)이 지킨다.
    // 한 칸: 배경(거래량 비율 4단계) + 숫자 + 불균형 표시(반대편의 300% 초과면 바깥쪽 세로선).
    // 2026-09-19 «막대 길이» 인코딩(A안)을 되돌렸다 -- 사용자 지시. 길이는 비율을 보여주는
    // 대신 짧은 막대의 숫자를 지웠고, 여기서 읽는 건 그 숫자다. 칸을 다시 고정하고 체결량은
    // 사각형 안에 적는다. 색 농담은 원래대로 «크다/작다»를 말한다. (칸 값·불균형 배수는 봉 호버 툴팁의 «칸» 줄)
    const fw0 = fpCv.cs.fontWeight || 400;   // svg <text> 는 굵기를 안 줘 svg 의 굵기를 물려받았다
    const drawCell = (ctx, cx, yTop, v, other, max, color, edgeX) => {
      cvRect(ctx, cx, yTop, half, rowPx, color, +shade(v, max));
      if (v > 0 && rowPx >= 7 && showQty)
        cvText(ctx, cx + half / 2, yTop + rowPx / 2 + fontPx * 0.36, fmtFootprintQty(v), fontPx, fw0, "var(--ink)", +INK_OPACITY, "middle");
      if (v > 0 && v > other * FOOTPRINT_IMBALANCE_RATIO) cvRect(ctx, edgeX, yTop + 0.5, 2, Math.max(1, rowPx - 1), color);
    };
    const barInfo = (rows) => {
      let pocKey = null, pocVol = 0, buyTot = 0, sellTot = 0, lowKey = Infinity;
      rows.forEach((cell, key) => {
        buyTot += cell[0]; sellTot += cell[1];
        if (key < lowKey) lowKey = key;
        if (cell[0] + cell[1] > pocVol) { pocVol = cell[0] + cell[1]; pocKey = key; }
      });
      return { pocKey, pocVol, buyTot, sellTot, lowKey };
    };
    // 1px 밑 봉 합치기(아래 DENSE) · 4px 위는 노드 캔들(MOB fpcW)과 같은 폭
    const denseCandle = (ctx, x, c, fw = fpThin ? Math.max(0.6, bw) : slot >= 4 ? Math.max(3, bw * 0.6) : Math.max(0.6, bw * 0.8)) => {
      const col = c.close >= c.open ? "var(--good)" : "var(--bad)", cxm = x + bw / 2;
      const yT = yAt(Math.max(c.open, c.close)), yB = yAt(Math.min(c.open, c.close));
      cvVLine(ctx, cxm, yAt(c.high), yAt(c.low), col);
      cvRect(ctx, cxm - fw / 2, yT, fw, Math.max(yB - yT, 1), col);
    };
    const barItem = (i, rows, inf) => {
      const c = candles[i], x = xAt(i);
      return { x0: x - 6, x1: x + bw + 6, y0: mt, y1: plotBottom, c: plotC, p: DENSE ? (ctx) => denseCandle(ctx, x, c) : (ctx) => {
        if (drawCells) rows.forEach((cell, key) => {
          const yTop = yAt(rowLo(key + 1));
          if (yTop + rowPx < mt || yTop > mt + ch) return;   // 창 밖 행은 건너뛴다
          const sx = x + (FPC === "L" ? fpcW + fpcGap : 0);                                 // 매도 칸 왼쪽 끝
          const bx = sx + half + 1 + (FPC === "C" ? fpcW + fpcGap : 0);                     // 매수 칸 왼쪽 끝
          drawCell(ctx, sx, yTop, cell[1], cell[0], maxSell, "var(--bad)", sx - 2);          // 왼쪽 = 매도
          drawCell(ctx, bx, yTop, cell[0], cell[1], maxBuy, "var(--good)", bx + half + 2); // 오른쪽 = 매수
          if (key === inf.pocKey) cvStrokeRect(ctx, sx, yTop, bx + half - sx, rowPx, "var(--amber)", 1, 0.85);   // POC -- 그 봉에서 가장 많이 거래된 가격 행
        });
        // 캔들은 테두리로만 남긴다 -- 셀을 덮지 않으면서 시가/종가/꼬리를 잃지 않으려는 것.
        const color = c.close >= c.open ? "var(--good)" : "var(--bad)";
        const yT = yAt(Math.max(c.open, c.close)), yB = yAt(Math.min(c.open, c.close));
        if (FPC) {   // 옆 5분봉: 채운 몸통 + 심지(칸을 덮지 않는 자리)
          const cx0 = MOB ? x + (bw - fpcW) / 2 : FPC === "L" ? x : FPC === "R" ? x + bw - fpcW : x + half + 1 + fpcGap / 2, cxm = cx0 + fpcW / 2;
          cvVLine(ctx, cxm, yAt(c.high), yAt(c.low), color);
          cvRect(ctx, cx0, yT, fpcW, Math.max(yB - yT, 1), color, 1, fpcW >= 5 ? 1 : 0);
        } else {
          cvVLine(ctx, x + bw / 2, yAt(c.high), yAt(c.low), color, 0.5);
          cvStrokeRect(ctx, x, yT, bw, Math.max(yB - yT, 1), color, 1, 0.6);
        }
      } };
    };

    // 봉 델타 라벨의 충돌 회피용. 모바일(봉 폭 ~25px)에서는 글자(~38px)가 봉보다 넓어
    // 이웃끼리 겹친다 -- 라벨을 버리지 않고 **겹치면 한 줄씩 내린다**(값은 다 보여야 한다).
    const deltaBoxes = [];
    const deltaFont = bw >= 34 ? 11 : 9;
    // 2026-10-20 사용자 «어느 순간까지는 델타가 보이다가 아예 겹치는 구간부터는 델타를 제거»: 자리는 늘 다 계산하고(아래 사슬),
    //   그리기는 봉 루프 뒤에서 정한다 -- 제자리(봉 바로 아래)에 못 앉고 밀려난 라벨이 많으면(DELTA_CROWD 초과) 그 창은 델타를 통째로 안 그린다.
    //   일부만 지우면 «어느 봉은 델타가 없나»로 읽히니 전부 아니면 전무. 값은 봉 호버 툴팁에 그대로 있다.
    const DELTA_CROWD = 0.3;   // 실측(10-20 1920·1700): 2h .12~.17 · 3h .17 · 4h .23~.27 · 6h .39~.42 · 12h .55~.58 → 4h 까지 보이고 6h 부터 뺀다
    const dState = { on: false };   // 그릴지(전부 아니면 전무) -- 형성 봉 갱신이 다시 정할 수 있게 칠하기가 이 값을 읽는다
    const deltaOf = (i, inf) => {
      // 봉 델타(매수-매도). 2026-09-16 사용자 요청으로 **플롯 맨 위 -> 그 봉 바로 아래**로
      // 옮기고 굵게 했다. 맨 위에 있을 때는 어느 봉의 숫자인지 눈이 세로로 훑어야 했다 --
      // 봉 밑에 붙으면 그 봉의 것임이 위치로 자명하다. y 는 그 봉의 **저가** 기준이라
      // 봉마다 높이가 다르다(고정 행이 아니다).
      // 2026-10-20 DENSE(봉 8px 밑)면 라벨이 늘 DELTA_CROWD 를 넘어 통째로 빠진다 -- 사슬 계산을 건너뛴다
      if (!(inf.buyTot + inf.sellTot > 0) || DENSE) return null;
      const c = candles[i], delta = inf.buyTot - inf.sellTot;
      // 기준은 캔들 저가가 아니라 **가장 아래 셀 행의 바닥**이다. 저가로 잡았더니 그 아래로
      // 더 내려오는 마지막 행과 글씨가 겹쳤다(2026-09-16 첫 판에서 실제로 겹쳤다) --
      // 행은 rowSize 격자라 저가보다 최대 한 행만큼 더 내려간다.
      const cellsBottom = Number.isFinite(inf.lowKey) ? yAt(rowLo(inf.lowKey)) : yAt(c.low);
      let dY = Math.min(plotBottom - 3, Math.max(cellsBottom, yAt(c.low)) + 13);
      // 글자 폭 근사(굵은 숫자 ~0.62em) 로 상자를 만들고, 겹치면 아래로 한 줄씩 민다.
      const label = (delta >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(delta));
      const halfW = label.length * deltaFont * 0.34 + 2;
      const cxD = xAt(i) + bw / 2, dY0 = dY;
      for (let guard = 0; guard < 6; guard++) {
        // 상자는 x 순서로 쌓인다 -- 반폭 최대(6글자 ≈ 24px)보다 멀어진 상자 앞은 볼 필요 없다(10-20 1주일 창: some() 전수 = O(n²) 가 렌더의 20%)
        let hit = false;
        for (let q = deltaBoxes.length - 1; q >= 0 && !hit; q--) {
          const b = deltaBoxes[q];
          if (b.x2 + 26 < cxD - halfW) break;
          hit = Math.abs(b.y - dY) < deltaFont + 1 && cxD + halfW > b.x1 && cxD - halfW < b.x2;
        }
        if (!hit || dY + deltaFont + 2 > plotBottom - 3) break;
        dY += deltaFont + 2;
      }
      deltaBoxes.push({ x1: cxD - halfW, x2: cxD + halfW, y: dY });
      // 🔴여기까지의 **산수는 늘 돈다** -- dY 는 앞선 봉들의 충돌회피 결과에 달려 있어서 건너뛰면 사슬이 끊긴다.
      return { bumped: dY !== dY0, it: { x0: cxD - halfW - 8, x1: cxD + halfW + 8, y0: dY - deltaFont - 2, y1: dY + 4, c: plotC,
        p: (ctx) => { if (dState.on) cvText(ctx, cxD, dY, label, deltaFont, "bold", delta >= 0 ? "var(--good)" : "var(--bad)", 1, "middle"); } } };
    };
    const pocPts = [];    // 2026-09-19 사용자 요청: 봉별 POC 를 선으로 잇는다.
    // POC «점»은 창 밖 행을 건너뛰는 칸 그리기와 **같은 조건**으로 건너뛴다.
    const pocPt = (i, inf) => {
      if (inf.pocKey === null) return null;
      const pocY = yAt(rowLo(inf.pocKey + 1));
      return pocY + rowPx < mt || pocY > mt + ch ? null : [xAt(i) + bw / 2, pocY + rowPx / 2];
    };
    const denseCol = new Map();   // 1px 밑 봉 합치기(아래 DENSE)
    const deltaIts = [];
    let deltaBumped = 0;
    const live = { bar: -1, delta: -1, stack: -1, pocN: 0, dbN: 0, nD: 0, bump: 0 };   // 형성 봉의 칠하기 자리·사슬 상태(fpCv.live)
    barRows.forEach((rows, i) => {
      const c = candles[i], x = xAt(i), inf = barInfo(rows);
      if (i === iL) Object.assign(live, { pocN: pocPts.length, dbN: deltaBoxes.length, nD: deltaIts.length, bump: deltaBumped });
      const pp = pocPt(i, inf);
      if (pp) pocPts.push(pp);
      if (DENSE && slot < 1) {   // 2026-10-20 봉이 1px 밑(1주일 창)이면 같은 픽셀 열의 봉을 한 캔들로 합친다(첫 시가·끝 종가·고저) -- 2.5봉이 한 픽셀에 겹쳐 칠하던 것
        const k = Math.round(x + bw / 2), a = denseCol.get(k);
        if (a) { a.h = Math.max(a.h, c.high); a.l = Math.min(a.l, c.low); a.c = c.close; }
        else denseCol.set(k, { o: c.open, h: c.high, l: c.low, c: c.close });
      } else {
        if (i === iL) live.bar = fgI.length;
        fgI.push(barItem(i, rows, inf));
      }
      const d = deltaOf(i, inf);
      if (d) { deltaIts.push(d.it); if (d.bumped) deltaBumped++; }
    });
    denseCol.forEach((a, k) => fgI.push({ x0: k - 2, x1: k + 2, y0: mt, y1: plotBottom, c: plotC,
      p: (ctx) => denseCandle(ctx, k - bw / 2, { open: a.o, high: a.h, low: a.l, close: a.c }, 1) }));   // 합친 캔들 = 그 픽셀 열 한 칸
    renderCandleSvg._deltaStats = { n: deltaIts.length, bumped: deltaBumped, bars: candles.length };   // 시험·진단용
    // 커서 밑 칸(옛 칸·POC <title>) -- 봉 호버 툴팁 한 줄. 칸을 그리는 폭에서만.
    cellTip = (i, price) => {
      const k = price == null ? null : rowOf(price), cell = k != null && drawCells && barRows[i] ? barRows[i].get(k) : null;
      if (!cell) return "";
      const [b, sl] = cell, imb = (v, o) => v > 0 && v > o * FOOTPRINT_IMBALANCE_RATIO ? ` · 불균형 ${o > 0 ? (v / o).toFixed(1) + "배" : "일방"}` : "";
      return `<br>칸 ${(rowLo(k) + rowSize / 2).toFixed(pxDp())} · 매수 ${b.toFixed(1)}${imb(b, sl)} · 매도 ${sl.toFixed(1)}${imb(sl, b)}`
        + (k === barInfo(barRows[i]).pocKey ? " · POC(최다 체결)" : "");
    };
    dState.on = deltaIts.length > 0 && deltaBumped / deltaIts.length <= DELTA_CROWD;
    const NOOP = { x0: 1, x1: 0, y0: 1, y1: 0, p: () => {} };   // 아무 띠와도 안 겹친다(형성 봉 델타 자리 비움)
    deltaIts.forEach((it, j) => { if (j < live.nD) fgI.push(it); });
    live.delta = fgI.length; fgI.push(deltaIts.length > live.nD ? deltaIts[live.nD] : NOOP);

    // ── 봉별 POC 선 (2026-09-19 사용자 요청) ────────────────────────────
    // 봉마다 사각 테두리는 이미 있었지만 **봉끼리 독립**이라 «거래가 몰린 값이 어디로
    // 옮겨가는가»가 안 보였다. 그게 이 선이 유일하게 더 주는 정보다.
    // 🔴창 밖 행은 위 루프가 건너뛰므로 점이 빠진다 -- 선이 그 구간을 건너뛰고 이어지면
    //   없는 이동을 그리는 셈이다. 그래서 x 가 한 봉(bw) 넘게 벌어지면 **끊는다**.
    // 🔴«지지·저항»이 아니다. Dalton 밸런스엣지(비용게이트 0/6)·Yush LAF(무근거)로 닫혔다.
    if (mcLine.poc) fgI.push({ ...CV_ALL, c: plotC, p: (ctx) => {   // «지표» 팝업 «POC 선»(2026-10-29)
      if (pocPts.length < 2) return;
      let seg = [pocPts[0]];
      const flush = () => { if (seg.length > 1) cvStroke(ctx, seg, "var(--amber)", 1.5, 0.75, null, "round"); seg = []; };
      for (let i = 1; i < pocPts.length; i++) {
        if (pocPts[i][0] - pocPts[i - 1][0] > bw * 1.8) { flush(); }
        seg.push(pocPts[i]);
      }
      flush();
    } });

    // ── 시장 맥락 겹침 (2026-09-29, 사용자 «불합격이어도 쓸모 있으면 넣어줘 -- 우리 검증이 틀렸을 수도 있다») ──────────
    //   ① 스택 불균형(같은 쪽 3줄 연속) (② 패턴 글자 다이버·소진·흡수?는 09-30 제거 -- 4.7년·2025/2026 검정 방향 0) ③ 세션 VWAP ±1·2σ · 볼린저(카드 스위치)
    //   ④ HL 고래 실측 청산가 점선 (⑤ 12시간 실측 청산 눈금은 09-30 제거). 전부 서술 -- 뜻과 우리 검정은 각 <title>.
    //   선(③④)은 옵션 선과 같이 격자 바로 위(셀 아래)에, 표식(①②⑤)은 셀 위에. 가격 플롯 밖은 네이티브 clipPath 로 자른다.
    if (isSnapshotChart) {
      const mk = (parent, tag, attrs, text) => {
        const e = document.createElementNS(NS, tag);
        Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
        if (text != null) { const t = document.createElementNS(NS, "title"); t.textContent = text; e.appendChild(t); }
        parent.appendChild(e);
        return e;
      };
      const lines = document.createElementNS(NS, "g");
      lines.setAttribute("clip-path", plotClip);
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
        const bbm = mcBollinger(chartTfFull());   // 리뷰: 라이브 200봉만 세면 1h(≈17봉)는 20봉이 안 돼 밴드가 없었다 -- 1주일까지
        [0, 2].forEach((j) => pl(candles.map((c, i) => (bbm.has(c.time) ? `${cx(i)},${yAt(bbm.get(c.time)[j]).toFixed(1)}` : null)).filter(Boolean),
          { stroke: "var(--muted)", "stroke-opacity": 0.55, "stroke-dasharray": "3 3", "stroke-width": 1 },
          "볼린저 밴드(20봉 · 2σ). 우리 검정: 4시간 볼린저는 평균회귀·이탈 추종 둘 다 비용을 못 넘었다 — 위치 참고."));
      }
      // ③ 이동평균 셋(2026-10-26 펜 메모, 기본 끔) -- 지금 tf 봉(5m·30m·1h)으로 전체 이력(1주일까지)에서 센다. 이름 = 가격 칸 오른쪽 끝(점선 지시선) · 색 = 스위치 옆 막대.
      [["ema5", "EMA 5", 1.2, () => mcEma(chartTfFull(), 5), "종가 지수이동평균 5봉"],
       ["ema20", "EMA 20", 1.4, () => mcEma(chartTfFull(), 20), "종가 지수이동평균 20봉"],
       ["vwma", "VWMA 100", 1.6, () => mcVwma(chartTfFull(), 100), "거래량 가중 이동평균 100봉 = Σ(종가×거래량) ÷ Σ거래량 · 거래량 = 바이낸스 5분봉 거래량(트레이딩뷰와 같은 정의 · 아래 거래대금 줄은 바이낸스+OKX 체결이라 원천이 다르다)"]]
        .filter(([k]) => mcLine[k]).forEach(([k, nm, sw, calc, what]) => {
          const m = calc(), col = `var(--ma-${k})`;
          pl(candles.map((c, i) => (m.has(c.time) ? `${cx(i)},${yAt(m.get(c.time)).toFixed(1)}` : null)).filter(Boolean),
             { stroke: col, "stroke-opacity": 0.85, "stroke-width": sw, "stroke-linejoin": "round" }, `${nm} -- ${what}. 위치 참고.`);
          const lc = [...candles].reverse().find((c) => m.has(c.time)), v = lc ? m.get(lc.time) : null;
          // 화면 밖이면 이름도 없다(VWAP 과 같은 규약 · 축은 캔들만 본다) -- 위·아래 끝 «↓» 꼬리표는 바닥의 추세 veto 꼬리표와 겹쳤다(실측: VWMA 100 이 2341, 화면 바닥 2350)
          if (v == null || yAt(v) <= mt + 8 || yAt(v) >= plotBottom - 2) return;
          if (gutOn) fpNotes.push({ val: v, label: nm, color: col });   // 휴대폰(칸 없음)은 이름 없이 색만 -- 스위치 옆 색 막대가 범례다(최신 봉을 안 가린다)
        });
      // ③ 주간·앵커 VWAP(2026-09-30 사용자 지시, 기본 끔) -- 체결 테이프(/api/market-context profile). 지금 값 하나라 가로선.
      //   우리 검정: VWAP 되돌림·위/아래 방향 모두 가짜 레벨과 같았다 — 지지·저항 아님, 위치 참고.
      const pf = mcLine.wvwap && (latestMarketCtx || {}).asset === activeSnapshotAsset ? latestMarketCtx.profile || null : null;
      if (pf && pf.available) {
        let lastY = -99;
        [["vwap_week", "주간 VWAP", "월 00:00 UTC 부터", "var(--ink)", "8 4", 0.55],
         ["avwap_hi", "앵커 VWAP·전일고", "전일 고가를 처음 찍은 시각부터", "var(--muted)", "2 3", 0.7],
         ["avwap_lo", "앵커 VWAP·전일저", "전일 저가를 처음 찍은 시각부터", "var(--muted)", "2 3", 0.7]]
          .filter(([k]) => pf[k] > 0).map((r) => [...r, yAt(pf[r[0]])]).sort((a, c) => a[6] - c[6])
          .forEach(([k, nm, from, col, dash, op, y]) => {
            if (y < mt || y > plotBottom) return;
            mk(lines, "line", { x1: ml, x2: ml + cw, y1: y.toFixed(1), y2: y.toFixed(1), stroke: col, "stroke-opacity": op, "stroke-dasharray": dash, "stroke-width": 1.2 },
               `${nm} ${pf[k].toFixed((ASSET_CONFIG[activeSnapshotAsset] || {}).dp ?? 1)} (${from} 거래량 가중 평균가). 우리 검정: VWAP 되돌림·방향 모두 가짜 레벨과 같았다 — 지지·저항으로 읽지 말 것.`);
            if (gutOn) { fpNotes.push({ val: pf[k], label: nm.replace("앵커 VWAP·", "앵커·"), color: "var(--muted)" }); return; }
            if (y - lastY < 12) return;
            lastY = y;
            mk(svg, "text", { x: ml + 4, y: (y - 3).toFixed(1), "font-size": mobileChart ? 9 : 10, "font-weight": 700, fill: "var(--muted)", "pointer-events": "none" }).textContent = nm;
          });
      }
      // ④ HL 고래 실측 청산가 -- 추정 청산맵(배경)과 따로, 거래소가 준 실제 청산가(추적 300주소)
      const mc = latestMarketCtx && latestMarketCtx.available && latestMarketCtx.asset === activeSnapshotAsset ? latestMarketCtx : null;
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
      cvFromSvgLines(lines, bgI, bgBehind, plotC);   // 2026-10-20 선은 bg 캔버스(칸 뒤) · svg 쪽은 투명하게 남아 <title> 호버만 한다
      // ⑤ 12시간 실측 청산 눈금은 2026-09-30 뺐다(사용자 지시) -- 12시간 최다 가격은 시장 맥락 카드 한 줄에 남는다.
      if (drawCells) {
        // ① 스택 불균형 -- 같은 봉 안에서 반대편의 3배 넘는 줄이 같은 쪽으로 3줄 이상 이어지면 바깥에 굵은 막대
        //   (우리 검정: 미검정 — 흡수·소진 가족은 봉·60초·가격행 단위에서 방향 0이었다.) 2026-10-20 fg 캔버스 -- 봉마다 칠하기 하나(형성 봉 갱신이 갈아 끼운다)
        const STACK_MIN = 3;
        stackItem = (i, rows) => {
          const keys = [...rows.keys()].sort((a, c) => a - c), x = xAt(i), marks = [];
          [[0, 1, x + bw + 3, "var(--good)"], [1, 0, x - 5, "var(--bad)"]].forEach(([s, o, ex, col]) => {
            let run = [];
            const flush = () => {
              if (run.length >= STACK_MIN) {
                const yTop = Math.max(mt, yAt(rowLo(run[run.length - 1] + 1))), yBot = Math.min(plotBottom, yAt(rowLo(run[0])));
                if (yBot - yTop > 2) marks.push([ex, yTop, yBot - yTop, col]);
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
          return { x0: x - 6, x1: x + bw + 6, y0: mt, y1: plotBottom, c: plotC, p: (ctx) => marks.forEach(([ex, y, ht, col]) => cvRect(ctx, ex, y, 2, ht, col, 1, 1)) };
        };
        barRows.forEach((rows, i) => { if (i === iL) live.stack = fgI.length; fgI.push(stackItem(i, rows)); });
      }
    }

    // ── 형성 봉 갱신 (2026-10-20 사용자 4번 · scheduleFpLiveFrame) ─────────────────────────
    //   체결이 온 프레임마다 맨 오른쪽 봉만 다시 계산해 fg 의 그 자리만 칠한다(svg 는 안 건드린다 -- 호버 툴팁이 안 깜빡인다).
    //   false = 전체 렌더가 필요하다: 봉이 바뀜 · 창/코인/tf 가 바뀜 · 축(고저·현재가)이 움직임 · 행 크기(ATR)가 바뀜 · 밀도 쓸림이 생김.
    //   음영 기준(창 최대 칸)이나 델타 «전부 아니면 전무» 판정이 바뀌면 fg 만 통째로 다시 칠한다(svg 무관 -- 둘 다 칠하기 안 값).
    //   1px 밑 봉(1주일 5분봉)은 열을 합쳐 그리므로 전체 렌더 박자 그대로(형성 봉이 반 픽셀이라 차이가 안 보인다).
    const trSum0 = trs.reduce((a, b) => a + b, 0) - trs[iL], live0 = { tf: chartTfK, asset: activeSnapshotAsset, win: chartWindowBars };
    let oMin = Infinity, oMax = -Infinity;
    for (let i = 0; i < iL; i++) { oMin = Math.min(oMin, candles[i].low); oMax = Math.max(oMax, candles[i].high); }
    if (isSnapshotChart && !panned && !chartPanY && iL >= 0 && live.bar >= 0) fpCv.live = () => {
      if (chartTfK !== live0.tf || activeSnapshotAsset !== live0.asset || chartWindowBars !== live0.win || chartPanEnd != null || chartPanY || !svg.isConnected) return false;
      // 형성 봉 OHLC: 5m 은 캔들 이력의 그 객체(체결마다 제자리 갱신) · 30m·1h 는 chartTfFull 이 그 tf 봉을 제자리에서 다시 묶는다(tfCandles -- 호버도 새 값)
      //   마지막 봉이 다른 객체 = 새 5분봉이 왔다(또는 이력 교체) → 전체 렌더
      const tfl = chartTfFull(), c = candles[iL];
      if (tfl[tfl.length - 1] !== c) return false;
      const cur = Number(latestLivePriceByAsset[activeSnapshotAsset] || 0), withCur = includeCurrentPrice && cur > 0;
      const [y0, y1] = yRangeOf(Math.min(oMin, c.low, withCur ? cur : Infinity), Math.max(oMax, c.high, withCur ? cur : -Infinity));
      if (y0 !== yMin || y1 !== yMax) return false;
      const pc = candles[iL - 1], tr = pc ? Math.max(c.high - c.low, Math.abs(c.high - pc.close), Math.abs(c.low - pc.close)) : c.high - c.low;
      if (Math.max(bucket, minRow, Math.round(0.2 * ((trSum0 + tr) / Math.max(nB, 1)) / bucket) * bucket) !== rowSize) return false;
      if (lastSnap) for (const b of lastSnap.bins || []) {   // 새로 쓸린 밀도 칸 = bg 가 바뀐다
        if (b.weightPct > 0 && !densitySweep.has(b.price) && c.time >= tfT(Math.floor((lastSnap.tsMs || 0) / 1000)) && c.low <= b.price && b.price <= c.high) return false;
      }
      // 형성 봉 칸: 그 tf 봉에 든 5분봉만 떼어 실시간 값을 얹고 묶는다(리뷰: 체결 프레임마다 2천 봉 footprintForChart 를 다 돌았다)
      const fb = fpBarsNow() || [], one = new Map();
      for (let j = fb.length - 1; j >= 0 && tfT(fb[j].time) >= c.time; j--) {
        if (tfT(fb[j].time) === c.time && Array.isArray(fb[j].levels) && fb[j].levels.length) one.set(fb[j].time, fb[j].levels);
      }
      footprintMergeLive(one, bucket);
      const rows = rowsOfLv((chartTfK > 1 ? tfLevels(one, bucket) : one).get(c.time)), inf = barInfo(rows);
      let mb = maxBuy0, ms = maxSell0;
      rows.forEach((cell) => { mb = Math.max(mb, cell[0]); ms = Math.max(ms, cell[1]); });
      const olds = [fgI[live.bar], fgI[live.delta], live.stack >= 0 ? fgI[live.stack] : NOOP, liqLive ? fgI[liqLive.at] : NOOP];
      barRows[iL] = rows;
      fgI[live.bar] = barItem(iL, rows, inf);
      pocPts.length = live.pocN;
      const pp = pocPt(iL, inf);
      if (pp) pocPts.push(pp);
      deltaBoxes.length = live.dbN;
      const d = deltaOf(iL, inf), nD = live.nD + (d ? 1 : 0), bumped = live.bump + (d && d.bumped ? 1 : 0);
      const on = nD > 0 && bumped / nD <= DELTA_CROWD, wide = on !== dState.on || mb !== maxBuy || ms !== maxSell;
      dState.on = on; maxBuy = mb; maxSell = ms;
      renderCandleSvg._deltaStats = { n: nD, bumped, bars: nB };
      fgI[live.delta] = d ? d.it : NOOP;
      if (live.stack >= 0) fgI[live.stack] = stackItem(iL, rows);
      if (liqLive) fgI[liqLive.at] = liqLive.of(iL);
      const xs = [...olds, fgI[live.bar], fgI[live.delta], live.stack >= 0 ? fgI[live.stack] : NOOP, liqLive ? fgI[liqLive.at] : NOOP]
        .filter((it) => it.x1 >= it.x0).map((it) => it.x0);
      fpCvPaint(fpCv.fg, { x0: wide ? ml : Math.min(xAt(iL) + bw / 2 - 1.8 * bw - 2, ...xs), x1: ml + cw, y0: mt, y1: plotBottom });
      return true;
    };

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
  // 2026-10-20 fg 캔버스(심지 · 하락은 채운 몸통 · 테두리)
  candles.forEach((c, i) => {
    const x = xAt(i), isUp = c.close >= c.open, color = isUp ? "var(--good)" : "var(--bad)";
    const yTop = yAt(Math.max(c.open, c.close)), yBot = yAt(Math.min(c.open, c.close));
    fgI.push({ x0: x - 2, x1: x + bw + 2, y0: -1e9, y1: 1e9, p: (ctx) => {
      cvVLine(ctx, x + bw / 2, yAt(c.high), yAt(c.low), color);
      if (!isUp) cvRect(ctx, x, yTop, bw, Math.max(yBot - yTop, 1), color);
      cvStrokeRect(ctx, x, yTop, bw, Math.max(yBot - yTop, 1), color);
    } });
  });
  }

  // Track markers per candle to avoid overlap
  const markerCounts = { top: {}, bottom: {} };

  // ── 추세 veto (2026-09-22, 사용자 요청 「풋프린트에 그려줘」) ─────────────────────────
  // SMA144(12h) ± 1.0×ATR144 히스테리시스. 밴드를 벗어날 때만 허용 측면이 바뀌고, 안이면
  // 직전 상태를 유지한다 -- 원시 부호는 중앙 런이 15분(하루 13.4회 뒤집힘)이라 화면에 못 쓴다.
  // 값은 서버가 붙여준다(dashboard/server.py::trend_veto_rows) -- 클라는 100봉만 갖고 있어
  // SMA144 를 스스로 못 만든다. **두 모드 모두에 그린다**(같은 renderCandleSvg 라 분기 없음).
  // 🔴레짐 리본과 다른 물건이다: 리본은 학습된 3분류(강세/약세/횡보), 이건 규칙 하나다.
  {
    const vp = (mcLine.veto ? candles : []).map((c, i) => ({ i, sma: Number(c.sma), atr: Number(c.atr), v: Number(c.veto) }))   // 2026-10-26 «차트에 그리기» 칩으로 끈다(기본 켬)
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

    // 이벤트 트리거 -- 매매 저널과 **같은 삼각형 문법**을 쓰고 `markerCounts` 를 공유해
    // 같은 봉에서 저널 마커와 겹치지 않게 한다(스택 25px). 저널보다 한 치수 작게(±5) 그려
    // 실제 체결이 시각적으로 우선하게 둔다. 글자 라벨은 붙이지 않는다 -- 6시간당 최대 11개라
    // "진입/청산" 처럼 글자를 넣으면 글자밭이 된다.
    (cm.events || []).forEach(ev => {
      if (mobileChart && ev.grade === "약") return;   // 피치 3.5px -- 모바일은 강/중만
      const idx = idxByEpoch.get(tfT(Date.parse(ev.t) / 1000));
      if (idx === undefined || !candles[idx] || xAt(idx) + bw / 2 < ml || xAt(idx) + bw / 2 > ml + cw) return;   // 2026-10-20 끌기로 반쯤 걸친 봉 = 플롯 밖
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

  // 2026-09-28 옵션 가격축 표시(사용자 선택 A+C, 기본 켜짐 · 옵션 카드의 «가격축 표시»로 끔): (10-15 예상 폭 띠 제거) ·
  //   max pain(가까운 만기)(2026-10-02 감마 플립 선 제거). 격자 바로 위(셀 아래)에 깔아 셀 숫자를 덮지 않는다.
  //   🔴행사가 자석·핀닝은 우리 검정에서 없었다 -- 그래서 흐린 점선 + 글자이고 굵은 벽처럼 그리지 않는다.
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
      const px = Number(currentPrice) || o.index, fs = mobileChart ? 10 : 11;
      // 2026-10-15 사용자 «5분·1시간 변동성 폭은 계산과 프론트에서 제거» -- 괄호·꼬리표·글자 삭제, max pain 만 남긴다.
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
      const gridG = layerCache.get("grid")?.g;
      if (gridG && gridG.parentNode === svg) svg.insertBefore(og, gridG.nextSibling); else svg.appendChild(og);
      cvFromSvgLines(og, bgI, bgBehind, null);   // 2026-10-20 선은 bg 캔버스(칸 뒤) · 글자는 svg
    }
  }

  // 2026-10-26 사용자 «박스 유지 확률은 미래 정보 -- 현재 5분봉 이후에 박스권 유지/이탈로 · 프로파일의 박스 선은 제거» (시안 A «미래 상자»):
  //   옛 1시간 박스 선(10-16, 지나간 봉 위까지 전폭)·호가 띠 위 박스 경계선(10-17 boxColG, 벽 꼬리표 대신)을 없애고 -- 벽 꼬리표는 되돌아온다 --
  //   현재 봉 오른쪽 끝 ~ 예측 끝(until_ms)에만 30분·1시간·4시간 상자를 겹쳐 그린다. 오른쪽으로 끌어 빈 칸(chartPanRight)을 내면 보인다.
  //   주황 = «깨질 쪽» 또는 이미 벗어남 · 화면 끝에서 잘리면 꺾쇠 · 묵은 값(3분↑)은 안 그림 · 글자 «30분 유지 9%»(휴대폰 «30분 9%»).
  const bxF = BOX_ON_CHART && isSnapshotChart && footprint ? boxNow() : null;
  if (bxF && Date.now() - bxF.asof_ms <= BOX_STALE_MS && candles.length) {
    const L = candles.length - 1, tfS = 300 * chartTfK, xs = xAt(L + 1), xr = ml + cw;
    const xT = (ms) => xAt(L) + ((ms / 1e3 - candles[L].time) / tfS) * slot;
    const hz = Object.keys(BOX_NM).map((k, i) => ({ k, i, s: bxF.scales[k] })).filter((h) => h.s && h.s.ok && h.s.until_ms && xT(h.s.until_ms) > xs + 2);
    if (xs < xr - 4 && hz.length) {
      const g = document.createElementNS(NS, "g"), fs = mobileChart ? 9.5 : 10.5;
      const add = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v)); if (text != null) e.textContent = text; g.appendChild(e); return e; };
      const halo = { stroke: "var(--chart-bg)", "stroke-width": 3, "paint-order": "stroke", "font-family": "var(--font-mono)", "font-weight": 800, "font-size": fs };
      if (plotClip) g.setAttribute("clip-path", plotClip);
      g.setAttribute("pointer-events", "none");
      hz.forEach((h) => {
        const [tone] = boxState(bxF, h.s);
        h.col = tone === "brk" || tone === "out" ? "var(--warn)" : "var(--text)";
        h.p = Math.round(h.s.p * 100);
        h.xe = Math.min(xT(h.s.until_ms), xr); h.cut = xT(h.s.until_ms) > xr + 1;
        h.yu = yAt(h.s.up); h.yd = yAt(h.s.dn);
      });
      const chev = (x, y, col) => add("path", { d: `M${x - 5},${y - 4} L${x - 1},${y} L${x - 5},${y + 4}`, fill: "none", stroke: col, "stroke-width": 1.6, "stroke-linecap": "round", "stroke-linejoin": "round" });
      add("line", { x1: xs, x2: xs, y1: mt, y2: plotBottom, stroke: "var(--text)", "stroke-opacity": 0.22, "stroke-dasharray": "2 3" });   // 지금(현재 봉 끝)
      [...hz].reverse().forEach((h) => {   // 4시간 → 30분 순서로 겹침
        add("rect", { x: xs, y: h.yu, width: Math.max(0, h.xe - xs), height: Math.max(0, h.yd - h.yu), fill: h.col, "fill-opacity": 0.045 });
        const ln = { stroke: h.col, "stroke-opacity": 0.75, "stroke-width": 1 + 0.4 * h.i, "stroke-dasharray": "5 3" };
        add("line", { x1: xs, x2: h.xe, y1: h.yu, y2: h.yu, ...ln }); add("line", { x1: xs, x2: h.xe, y1: h.yd, y2: h.yd, ...ln });
        if (h.cut) { chev(xr - 2, h.yu, h.col); chev(xr - 2, h.yd, h.col); }
        else add("line", { x1: h.xe, x2: h.xe, y1: h.yu, y2: h.yd, ...ln, "stroke-dasharray": "none", "stroke-opacity": 0.5 });
      });
      const labs = [], yTop = mt + fs + 6;   // 화면 끝에서 잘린 상자끼리 위 경계가 12px 안이면 한 줄로
      hz.forEach((h) => {   // 위 경계가 화면 위로 나가면(급변 뒤 넓은 박스, 휴대폰 실측) 글자를 플롯 맨 위에 붙이고 «↑»
        const x = h.cut ? xr - 10 : h.xe - 4, y = Math.max(h.yu, yTop), up = h.yu < yTop ? " ↑" : "", near = labs.find((l) => Math.abs(l.x - x) < 40 && Math.abs(l.y - y) < 12);
        if (near) near.txt += ` · ${BOX_NM[h.k]} ${h.p}%${up}`;
        else labs.push({ x, y, txt: `${BOX_NM[h.k]} ${mobileChart ? "" : "유지 "}${h.p}%${up}`, col: h.col });
      });
      const wOf = (l) => [...l.txt].reduce((a, c) => a + (c > "ᄀ" ? 1.05 : 0.62), 0) * fs;   // 글자 폭 어림(한글 ≈ 1em · 측정은 강제 레이아웃) -- 같은 높이에서 가로로 겹치면 한 줄 아래로
      labs.forEach((l, j) => {
        while (labs.slice(0, j).some((o) => Math.abs(o.y - l.y) < fs + 2 && l.x - wOf(l) < o.x && o.x - wOf(o) < l.x)) l.y += fs + 3;
        add("text", { x: l.x, y: l.y - 4, "text-anchor": "end", fill: l.col, ...halo }, l.txt);
      });
      svg.appendChild(g);
    }
  }

  // 2026-10-26 사용자 «B1 QX4I 도 풋프린트 차트에 박스 유지처럼»: 4h 변동성 QX4I(QX4 + 옵션 ATM IV 보정 · ETH 전용 · 조건부 채택, 전진 판정 2027-01-22)
  //   ±1σ 를 같은 미래 칸에 «부채꼴»로 -- 예측 봉 마감가(px) 중심, 폭 = σ4h·√((t − asof)/4h)(현재 봉 끝에서 그 시각의 폭으로 시작) · 예측 끝(asof + 4h)에 ±σ4h(중앙값형).
  //   2026-10-26 사용자 «현재가에 붙어 계속 움직이지 말고 5분마다»: 중심·폭 모두 서버 5분 예측값에만 묶임(실시간 시세로 안 흔들림).
  //   박스(직사각형 = 유지 확률)와 구별되게 옵션 색 실선 · 묵은 값(15분↑)은 안 그림 · 화면 끝에서 잘리면 꺾쇠.
  const vI = isSnapshotChart && footprint && activeSnapshotAsset === "eth" && latestVol4h && latestVol4h.ok && latestVol4h.asset === "eth"
    && Number.isFinite(latestVol4h.sig4h_iv_pct) && Date.now() - latestVol4h.asof_ms <= 900e3 ? latestVol4h : null;
  if (vI && vI.px > 0 && candles.length) {
    const L = candles.length - 1, tfS = 300 * chartTfK, xs = xAt(L + 1), xr = ml + cw;
    const xT = (ms) => xAt(L) + ((ms / 1e3 - candles[L].time) / tfS) * slot;
    const t0 = (candles[L].time + tfS) * 1e3, t1 = vI.asof_ms + 4 * 3600e3, xEnd = xT(t1);
    if (xs < xr - 4 && xEnd > xs + 2 && t1 > t0) {
      const g = document.createElementNS(NS, "g"), fs = mobileChart ? 9.5 : 10.5, col = "var(--option)", s = vI.sig4h_iv_pct / 100;
      const add = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v)); if (text != null) e.textContent = text; g.appendChild(e); return e; };
      if (plotClip) g.setAttribute("clip-path", plotClip);
      g.setAttribute("pointer-events", "none");
      const up = [], dn = [], xMax = Math.min(xEnd, xr);
      const px = vI.px, xToT = (x) => t0 + (x - xs) / (xEnd - xs) * (t1 - t0);   // 30분·1시간 봉은 현재 봉 끝이 asof 보다 최대 1시간 뒤 -- 폭은 asof 부터 잰 시간으로
      for (let i = 0; i <= 32; i++) {
        const x = xs + (xMax - xs) * i / 32, k = s * Math.sqrt(Math.max(0, (xToT(x) - vI.asof_ms) / (t1 - vI.asof_ms)));
        up.push([x, yAt(px * (1 + k))]); dn.push([x, yAt(px * (1 - k))]);
      }
      const pts = (a) => a.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
      add("polygon", { points: `${pts(up)} ${pts([...dn].reverse())}`, fill: col, "fill-opacity": 0.045 });
      [up, dn].forEach((a) => add("polyline", { points: pts(a), fill: "none", stroke: col, "stroke-width": 1.5, "stroke-opacity": 0.9, "stroke-linejoin": "round" }));
      const [xe, yu] = up[up.length - 1], yd = dn[dn.length - 1][1];
      if (xEnd > xr + 1) [yu, yd].forEach((y) => add("path", { d: `M${xr - 7},${y - 4} L${xr - 3},${y} L${xr - 7},${y + 4}`, fill: "none", stroke: col, "stroke-width": 1.6, "stroke-linecap": "round", "stroke-linejoin": "round" }));
      else add("line", { x1: xe, x2: xe, y1: yu, y2: yd, stroke: col, "stroke-opacity": 0.55, "stroke-width": 1 });
      // 글자는 아래 가장자리 밑(박스 글자는 위 경계 위) -- 화면 아래로 나가면 플롯 바닥에 붙인다
      const ty = Math.min(yd + fs + 3, plotBottom - 4), pct = `±${vI.sig4h_iv_pct.toFixed(2)}%`;
      add("text", { x: Math.min(xe, xr) - 4, y: ty, "text-anchor": "end", fill: col, stroke: "var(--chart-bg)", "stroke-width": 3, "paint-order": "stroke",
        "font-family": "var(--font-mono)", "font-weight": 800, "font-size": fs },
        mobileChart ? `4h 1σ ${pct}` : `4h 1σ ${pct} · 옵션 IV ${Math.round(vI.iv_atm)}`);
      svg.appendChild(g);
    }
  }

  // 2026-10-26 펜 메모 «호가벽 데이터 선 표시들도 제거해줘»: 호가 띠 위 ±25·±50·±100bp 경계선 + «가격 · 수량» 꼬리표(2026-10-06 시안 G)를 뺐다 -- 같은 값은 ④ 벽 판(book.sweep)에 그대로.

  // 2026-09-30 30분 도달 선 제거(사용자 지시 -- «±0.5×30분 폭에 닿는가»는 옵션 예상 폭 띠와 같은 크기 정보라 중복).

  // 2026-10-12 시안 A «가격축 자»: 플롯 오른쪽 끝에 현재가 → 위·아래 첫 청산 띠 가장자리 괄호 + 거리($·%). 가까운 쪽 진하게.
  //   화면 밖이면 플롯 끝까지만 긋고 끝 눈금을 뺀다(거리 숫자는 그대로).
  if (lastSnap && currentPrice > 0) {
    const edges = liqBandEdges(lastSnap.bins, densityClip, densityBinWidth, currentPrice, densitySweep);
    const dU = edges.up === null ? Infinity : edges.up - currentPrice, dD = edges.dn === null ? Infinity : currentPrice - edges.dn;
    // 2026-10-22 아티팩트 펜 메모 «이 선 때문에 풋프린트가 잘 안 보여 · 살짝 오른쪽으로»: 데스크톱은 플롯 안(형성 봉 매수 칸 위)이 아니라 오른쪽 기둥 왼쪽 끝 바로 안 · 휴대폰은 그대로
    const bx = mobileChart ? ml + cw - 6 : tradeX0 + 3, cy = yAt(currentPrice);
    [[true, edges.up, dU], [false, edges.dn, dD]].forEach(([isUp, edge, d]) => {
      if (edge === null) return;
      const near = isUp ? dU <= dD : dD < dU, col = isUp ? "var(--liq-resistance)" : "var(--liq-support)";
      const yRaw = yAt(edge), ye = Math.max(mt, Math.min(plotBottom, yRaw)), pct = (d / currentPrice) * 100;
      const g = document.createElementNS(NS, "g");
      const seg = (x1, y1, x2, y2, sw) => { const l = document.createElementNS(NS, "line");
        Object.entries({ x1, y1, x2, y2, stroke: col, "stroke-width": sw }).forEach(([k, v]) => l.setAttribute(k, v)); g.appendChild(l); };
      seg(bx, cy, bx, ye, near ? 2.5 : 1.5);
      if (ye === yRaw) seg(bx - 7, ye, bx + 4, ye, 2);
      const t = document.createElementNS(NS, "text");
      const ty = isUp ? Math.min(cy - 8, (cy + ye) / 2 + 4) : Math.max(cy + 16, (cy + ye) / 2 + 4);
      // 2026-10-12 사용자 «거리 텍스트를 풋프린트 차트 안이 아니라 체결/포지션 프로파일 쪽으로»: 글자만 프로파일 칸 왼쪽 끝(데스크톱 = 플롯 오른쪽 기둥 · 휴대폰 = 플롯 왼쪽 기둥).
      //   2026-10-27 휴대폰 아래 칸(PROF)이면 왼쪽 기둥이 없다 -- 자 바로 왼쪽에 오른쪽 맞춤.
      Object.entries({ x: PROF ? bx - 8 : mobileChart ? ml - TRADE_L + 2 : tradeX0 + 4, y: ty, "text-anchor": PROF ? "end" : "start",
        "font-size": mobileChart ? 10 : 11.5, "font-weight": near ? 700 : 600,
        fill: near ? col : `color-mix(in srgb, ${col} 62%, transparent)`, stroke: "var(--chart-bg)", "stroke-width": 3,
        "paint-order": "stroke", "stroke-linejoin": "round", style: "font-variant-numeric: tabular-nums" }).forEach(([k, v]) => t.setAttribute(k, v));
      const sign = isUp ? "+" : "−";
      t.textContent = mobileChart ? `${sign}${pct.toFixed(2)}%` : `${sign}${fmtNum(d, pxDp())} · ${pct.toFixed(2)}%`;
      const ti = document.createElementNS(NS, "title");
      ti.textContent = `${isUp ? "위" : "아래"} 첫 청산 띠 ${fmtNum(edge, pxDp())} 까지 ${sign}${fmtNum(d, pxDp())} (${pct.toFixed(2)}%)`;
      t.appendChild(ti); g.appendChild(t);
      svg.appendChild(g);
    });
  }
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
    // 2026-10-26 사용자 «라벨이 최신 봉 쪽에 있어 가린다»: VWAP·EMA·VWMA 이름도 플롯 안 오른쪽 끝을 떠나 다른 선 이름처럼 호가 프로파일 오른쪽 끝으로(점선 지시선) -- 한 줄로 겹침을 돈다
    declutterTagY(priceLabels, mt + 9, plotBottom - 9, 19);
  }
  const tagX = ml + cw + 4, xR = ml + cw - 4, inR = LBL !== "B";
  const xRR = w - mr - 2;   // 호가 프로파일 오른쪽 끝 -- 현재가·지지/저항·감마 플립·지표·max pain 이름이 여기 오른쪽 정렬
  // 2026-09-30 약어 → 짧은 이름(사용자 «약어가 너무 많아 이해가 힘들다»): 이름은 그대로, 가격만 호버로 뺀다.
  const CODE = (nm) => nm.replace("앵커·전일고", "앵커 전일고").replace("앵커·전일저", "앵커 전일저")
    .replace(/^HL (롱|숏).*/, "HL $1청산");
  priceLabels.forEach(p => {
    const labelYRaw = p.adjustedY !== undefined ? p.adjustedY : p.realY;
    if (p.note) {   // 겹침 선 이름 -- 선은 제자리에 이미 있다. 칸 안 글자 + 지시선만.
      const ly = Math.max(mt + 9, Math.min(plotBottom - 9, labelYRaw));
      const ng = document.createElementNS(NS, "g");
      const lead = document.createElementNS(NS, "path");
      const far = inR;   // 프로파일 끝으로 간 이름 -- 선(플롯 안)에서 떨어지므로 글자 앞까지 옅은 점선을 잇는다(아래에서 폭을 잰 뒤)
      if (!far && (!inR || Math.abs(ly - p.realY) > 2)) {
        lead.setAttribute("d", inR ? `M${ml + cw} ${p.realY.toFixed(1)} L${xR} ${ly.toFixed(1)}` : `M${ml + cw} ${p.realY.toFixed(1)} L${tagX - 2} ${ly.toFixed(1)}`);
        lead.setAttribute("stroke", p.color); lead.setAttribute("stroke-opacity", "0.45"); lead.setAttribute("fill", "none");
        ng.appendChild(lead);
      }
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", inR ? xRR - 2 : tagX + 2); t.setAttribute("y", ly + 3.5); t.setAttribute("font-size", LBL === "C" ? "10.5" : "10"); t.setAttribute("font-weight", "700");
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
        try { tw = textW(t); } catch (e) { /* 비렌더 */ }
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
      if (p.behindCells) {                       // 셀 숫자를 덮지 않게 격자 바로 위(셀 아래) = 2026-10-20 bg 캔버스(칸이 fg 라). 빠른 갱신은 bg 띠만 다시 칠한다(liveLineCtx.cvLine)
        const it = { x0: ml - TRADE_L - 1, x1: w - mr + 1, y: p.realY, y0: p.realY - 2, y1: p.realY + 2,
                     p: (ctx) => cvStroke(ctx, [[ml - TRADE_L, it.y], [w - mr, it.y]], p.color, p.width || 2, 0.85, lineDashed ? "4,4" : null) };
        bgI.splice(bgBehind, 0, it);
        cvLiveLine = it;
      } else svg.appendChild(line);
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
        try { if (textW(txt) > (gutTag ? 84 : ml - TRADE_L - 7)) txt.textContent = `${name} ${fmtNum(p.val, Math.max(0, pxDp() - 1))}`; } catch (e) { /* 비렌더 */ }
      }
      if (tagBg) {
        let tw = 0;
        try { tw = textW(txt); } catch (e) { /* 비렌더 */ }
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
        if (pTxt.isConnected && textW(pTxt) + textW(sTxt) + 6 > boxW - 9) sTxt.remove();
      } catch (e) { /* 측정 불가(비렌더) -- 그대로 둔다 */ }
    }
    if (p.marker) svg.appendChild(line);   // 배지 위에 -- 위 주석 참조
    // 2026-10-10 휴대폰 풋프린트 현재가(사용자 «모바일 차트에 현재가가 안 보인다»): 가격축이 없어 아래 한 줄에만 있었다 --
    //   데스크톱처럼 굵은 숫자를 호가 띠 오른쪽 끝에(바탕색 외곽선으로 막대 위에서도 읽힘). 빠른 갱신(updateLivePriceFast)이 따라 옮긴다.
    let mobNow = null;
    if (mobileChart && footprint && p.label === "현재" && isSnapshotChart) {
      mobNow = document.createElementNS(NS, "text");
      Object.entries({ x: xRR - 2, y: labelY + 4, "text-anchor": "end", "font-size": 13, "font-weight": 800, fill: p.color,
                       stroke: "var(--chart-bg)", "stroke-width": 4, "paint-order": "stroke", "pointer-events": "none",
                       style: "font-variant-numeric: tabular-nums" }).forEach(([k, v]) => mobNow.setAttribute(k, v));
      mobNow.textContent = fmtNum(p.val, pxDp());
      svg.appendChild(mobNow);
    }

    // 현재가 줄만 표식을 단다 -- 틱마다 **이 세 요소만** 옮기려는 것이다(전체 재렌더는 1초).
    // 표식이 없으면 빠른 갱신이 어느 줄을 움직여야 하는지 알 수 없다.
    if (p.label === "현재" && isSnapshotChart) {
      line.dataset.live = p.marker ? "tri" : "line";
      rect.dataset.live = "box";
      pTxt.dataset.live = "text";
      if (txt) txt.dataset.live = "label";
      if (mobNow) { mobNow.dataset.live = "label"; mobNow.dataset.withPrice = "num"; }
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
    // 2026-10-10 풋프린트는 현재가가 차트 안(호가 띠 끝)에 있어 줄에서 뺀다 -- 그 자리로 지지·저항이 덜 잘린다.
    priceLabels.filter((p) => p.label && !(footprint && p.label === "현재")).forEach((p) => {
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
      try { adv = textW(t); } catch (_) { adv = t.textContent.length * 6.4; }
      if (!(adv > 0)) adv = t.textContent.length * 6.4;
      // 자릿수는 데이터가 정한다 -- 상수로 못 막으므로 넘치면 그 항목을 도로 뺀다.
      if (x + adv > w - 3) { t.remove(); x = w; return; }
      x += adv + 9;
    });
  }
  // 빠른 갱신이 가격 -> y 를 계산하려면 이번 렌더의 축 규약이 필요하다. 다음 전체 렌더가
  // 덮어쓴다 -- 즉 이 값은 항상 «지금 화면에 그려진 것»과 같다.
  if (isSnapshotChart) {
    liveLineCtx = { svg, yMin, yMax, mt, ch, markerX: ml + cw, cvLine: cvLiveLine };
  }

  // 2026-09-09 사용자 요청: 청산 밀도 가이드를 **차트 위(패널 HTML)** 로 옮겼다.
  //   기존에는 SVG 안 오른쪽 위 인셋(backing 이 y=0..34)이라 그때의 증거신호 **천장 레인**을 덮었다.
  //   차트 높이를 잃으므로(모바일 -8%) 아예 SVG 밖으로 뺀다.
  //   렌더는 캔들 SVG 안, 프로파일 바닥글 바로 아래 줄이다(2026-09-21 아티팩트 댓글).

  // Create Hover Layer on Top
  const hoverGroup = document.createElementNS(NS, "g");
  hoverGroup.setAttribute("class", "hover-layer");
  svg.appendChild(hoverGroup);

  // 십자선은 풋프린트 위에서 레인 바닥까지 걸친다(2026-10-06 레짐·전환 리본 줄 제거).
  // (2026-08-26, "레짐 그래프도 십자선에 걸쳤으면 좋겠어") -- plain candle chart baseline otherwise.
  const vLine = document.createElementNS(NS, "line");
  vLine.setAttribute("x1", 0); vLine.setAttribute("x2", 0);
  vLine.setAttribute("y1", mt);
  // 십자선은 아래 줄들까지 걸친다 -- 호버한 봉이 어느 구간에 속하는지 눈으로 잇게.
  vLine.setAttribute("y2", h - mb);   // 레인 바닥까지(2026-10-06 아래 리본 줄 제거)
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

  // Candlestick Tooltip Support.
  let laneTip = () => "";   // 풋프린트 아래 리본·차트 두 개의 그 봉 값(아래 레인 블록이 채운다 -- 핸들러가 먼저 붙으므로 let)
  svg.onmousemove = (evt) => {
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
    // 2026-10-27 휴대폰 아래 프로파일 칸: 등뼈 왼쪽 = 체결·포지션·청산 · 오른쪽 = 호가(같은 풀이 · 가격은 칸 높이로 잰다) -- 가격 칸에선 이 기둥들이 안 잡힌다
    const inProf = !!(PROF && my >= PROF.top - 16 && my <= PROF.bot + 4), colY = !PROF || inProf;
    const hT = inProf ? PROF.top : mt, hB = inProf ? PROF.bot : plotBottom;
    const priceAtY = yMax - ((my - hT) * ySpan) / (hB - hT);
    // 2026-10-10 포지션 보기: 기둥 위면 그 가격 행 · 미니맵 위면 그 높이의 가격 칸.
    if (posInfo && colY && mx > posInfo.mm.x0 && mx < posInfo.x1 + 2) {
      vLine.style.display = "none";
      const fq = (q) => (q >= 1000 ? (q / 1000).toFixed(1) + "k" : q >= 10 ? q.toFixed(0) : q.toFixed(1)), dp = pxDp();
      let lo, hi, q;
      if (mx <= posInfo.mm.x1) {                             // 미니맵: 화면 밖까지 창 전체
        const m = posInfo.mm, p = m.lo + (hB - my) / (hB - hT) * (m.hi - m.lo), j = Math.floor(p / m.bw) - m.W.lo;
        const span = Math.max(1, Math.round(rowSize / m.bw)), j0 = Math.max(0, j - Math.floor(span / 2));
        q = m.W.vals.slice(j0, j0 + span).reduce((a, b) => a + b, 0); lo = (m.W.lo + j0) * m.bw; hi = lo + span * m.bw;
      } else {
        const k = Math.floor((priceAtY + posInfo.off) / posInfo.rowSize); q = posInfo.rows.get(k) || 0; lo = k * posInfo.rowSize - posInfo.off; hi = lo + posInfo.rowSize;
      }
      if (!(q > 0)) { hideTooltip(); return; }
      const mid = (lo + hi) / 2, up = mid > currentPrice;
      showTooltip(evt.pageX, evt.pageY, `<div style="white-space:normal;max-width:min(360px,calc(100vw - 24px))">`
        + `<span style="color:var(--position);font-weight:700">남은 계약 ${fq(q)} ${coinUnit()}</span> · ${lo.toFixed(dp)}–${hi.toFixed(dp)}`
        + ` · 최근 ${posInfo.win}일 중 ${(100 * q / posInfo.tot).toFixed(1)}%`
        + (posInfo.rL && mx > posInfo.mm.x1
          ? (() => { const k = Math.floor((priceAtY + posInfo.off) / posInfo.rowSize), l = posInfo.rL.get(k) || 0, sh = posInfo.rS.get(k) || 0;
              return `<br><span style="color:var(--good);font-weight:700">롱 ${fq(l)}</span> · <span style="color:var(--bad);font-weight:700">숏 ${fq(sh)}</span>`
                + ` · 롱 몫 ${(100 * l / Math.max(l + sh, 1e-9)).toFixed(0)}% (≈ $${(l * mid / 1e6).toFixed(1)}M / $${(sh * mid / 1e6).toFixed(1)}M)`
                + `<br><span style="opacity:.7">롱·숏 나눔은 직전 1h·24h 흐름으로 손바뀜을 가른 근사 -- 1일 창은 롱·숏 평균 진입 차의 방향이 실측과 98% 맞지만 7·30일 창은 반반이라 롱·숏 차를 믿기 어렵다</span>`; })()
          : `<br>한쪽 금액 ≈ $${(q * mid / 1e6).toFixed(1)}M (롱 ${fq(q)} + 숏 ${fq(q)} · 계약 하나 = 롱 1 + 숏 1)`)
        + `<br>지금 상태: <span style="font-weight:700">${up ? "롱 물림 · 숏 수익" : "롱 수익 · 숏 물림"}</span> (현재가 ${up ? "위" : "아래"} ${(Math.abs(mid / currentPrice - 1) * 100).toFixed(1)}%)`
        + (posInfo.hours < posInfo.win * 24 - 1 ? `<br>창 실제 범위 ${posInfo.hours}시간(서버 기록이 아직 덜 참)` : "")
        + `<br><span style="opacity:.7">추정(5분 OI·거래량 · 손바뀜 + 새 포지션 먼저 닫힘) · 실측 아님 · 지도로만, 저항·지지 신호 아님</span></div>`);
      return;
    }
    // 2026-10-27 청산 기둥 위면 그 가격 행의 추정 청산(배수 구간별)과 현재가에서의 누적.
    if (liqInfo && colY && mx > liqInfo.x0 && mx < liqInfo.x1 + 2) {
      vLine.style.display = "none";
      const k = Math.floor((priceAtY + liqInfo.off) / liqInfo.rowSize), a = liqInfo.rows.get(k);
      const q = a ? a.reduce((x, y) => x + y, 0) : 0;
      if (!(q > 0)) { hideTooltip(); return; }
      const lo = k * liqInfo.rowSize - liqInfo.off, hi = lo + liqInfo.rowSize, up = (lo + hi) / 2 > currentPrice;
      const dp = Math.max(pxDp(), (String(liqInfo.off).split(".")[1] || "").length);   // 체결 툴팁과 같은 이유(경계 .25/.75)
      showTooltip(evt.pageX, evt.pageY, `<div style="white-space:normal;max-width:min(360px,calc(100vw - 24px))">`
        + `<span style="color:${up ? "var(--good)" : "var(--bad)"};font-weight:700">${up ? "숏" : "롱"} 청산 ${liqInfo.fu(q)}</span> · ${lo.toFixed(dp)}–${hi.toFixed(dp)}`
        + ` (현재가 ${up ? "위" : "아래"} ${(Math.abs((lo + hi) / 2 / currentPrice - 1) * 100).toFixed(1)}%)`
        + `<br>${a.map((v, i) => `${liqInfo.names[i]}배 ${liqInfo.fu(v)}`).join(" · ")}`
        + `<br>현재가에서 여기까지 누적 <span style="font-weight:700">${liqInfo.fu(liqInfo.cum(up ? hi : lo, up))}</span>`
        + `<br><span style="opacity:.7">추정 청산맵(최근 ${liqInfo.win}일 · 가격이 지나간 칸은 지움) · 실측 아님 · 지도로만</span></div>`);
      return;
    }
    // 2026-09-28 체결 기둥 위면 그 가격 행의 체결 풀이(데스크톱 기둥 · 모바일 왼쪽 띠).
    if (tradeInfo && (inProf ? mx < PROF.cx : !PROF && mx > tradeInfo.x0 && mx < tradeInfo.x1 + 2)) {   // 휴대폰 tradeInfo.x 는 아래 칸 자리
      vLine.style.display = "none";
      const k = Math.floor((priceAtY + tradeInfo.off) / tradeInfo.rowSize), a = tradeInfo.rows.get(k);
      if (!a) { hideTooltip(); return; }
      const fq = (q) => (q >= 1000 ? (q / 1000).toFixed(1) + "k" : q >= 10 ? q.toFixed(0) : q.toFixed(1));
      const pct = (v) => Math.round(100 * v / Math.max(a[0], 1e-9)) + "%", rs = tradeInfo.rowSize, o = tradeInfo.off;
      const dp = Math.max(pxDp(), (String(o).split(".")[1] || "").length);   // 경계가 .25/.75 라 한 자리 더(반올림하면 다른 줄을 가리킨다)
      showTooltip(evt.pageX, evt.pageY, `<div style="white-space:normal;max-width:min(360px,calc(100vw - 24px))">`
        + `<span style="color:var(--amber);font-weight:700">체결 ${fq(a[0])} ${coinUnit()}</span> · ${(k * rs - o).toFixed(dp)}–${((k + 1) * rs - o).toFixed(dp)}`
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
    if (bookInfo && !bookHidden && (inProf ? mx >= PROF.cx : mx > tradeX0 && mx <= w - mr)) {
      vLine.style.display = "none";
      const priceAt = priceAtY;
      let best = null;
      bookInfo.cells.forEach((c) => { if (!best || Math.abs(c.p - priceAt) < Math.abs(best.p - priceAt)) best = c; });
      if (best && Math.abs(best.p - priceAt) <= bookInfo.bs) { showTooltip(evt.pageX, evt.pageY, bookTipHtml(best)); return; }
      hideTooltip(); return;
    }
    if (inProf || mx < ml || mx > w - mr || (!PROF && TRADE_W && mx > tradeX0)) { vLine.style.display = "none"; hideTooltip(); return; }   // 10-28 기둥 칸 빈 곳 = 봉 툴팁 아님

    const idx = Math.min(candles.length - 1, Math.max(0, Math.floor((mx - ml) / slot - X0)));
    const c = candles[idx];
    if (!c) return;

    const tx = xAt(idx) + bw / 2;

    vLine.setAttribute("x1", tx);
    vLine.setAttribute("x2", tx);
    vLine.style.display = "block";

    // 2026-09-16: 툴팁이 **그 봉에 표시된 모든 것**을 말한다. 그전에는 시각·OHLC·레짐뿐이라
    // 변동성 리본도, 새로 붙인 트리거 표시도 «왜 떴는지»를 화면에서 물어볼 데가 없었다.
    // 원칙: 값이 없는 항목은 줄 자체를 안 만든다(빈 줄은 «0» 으로 오독된다).
    const cmNow = latestChartMarkers;
    let trigLines = "";
    if (isSnapshotChart && cmNow && cmNow.available) {
      (cmNow.events || []).forEach((ev) => {
        if (tfT(Math.floor(Date.parse(ev.t) / 1000)) !== c.time) return;
        trigLines += `<br>${ev.side === "bottom" ? "▲" : "▼"} ${ev.label}`
          + (ev.grade ? ` ${ev.grade}등급` : "")
          + (ev.p != null ? ` · 확률 ${(Number(ev.p) * 100).toFixed(0)}%` : "")
          + (ev.demoted ? " · 손실가중 헤드가 컷 미달로 강등" : "");
      });
    }
    showTooltip(evt.pageX, evt.pageY, `
      <b>${fmtDateTick(c.time * 1000)}</b><br>
      시가: ${fmtNum(c.open, Math.max(2, pxDp()))}<br>
      고가: ${fmtNum(c.high, Math.max(2, pxDp()))}<br>
      저가: ${fmtNum(c.low, Math.max(2, pxDp()))}<br>
      종가: ${fmtNum(c.close, Math.max(2, pxDp()))}${trigLines}${my >= mt && my <= plotBottom ? cellTip(idx, yMax - ((my - mt) * ySpan) / ch) : ""}${liqTip(idx)}${isSnapshotChart ? laneTip(idx) : ""}
    `);
  };
  svg.onmouseleave = () => {
    hideTooltip();
    if (typeof hoverDot !== 'undefined') hoverDot.style.display = "none";
    if (typeof vLine !== 'undefined') vLine.style.display = "none";
    if (typeof hoverDots !== 'undefined') hoverDots.forEach(d => d.style.display = "none");
    hLine.style.display = "none";
    priceBadgeRect.style.display = "none";
    priceBadgeText.style.display = "none";
  };

  // 사분면 리본·차트 두 개(풋프린트 아래)는 아래 «풋프린트 아래 = 사분면 리본 + 차트 둘» 블록이 그린다.
  // 오늘 거래량(세션 누적 RVOL)은 2026-10-06 카드 머리 칩에서 고래·중형·리테일 차트의 거래대금 괄호로 옮겼다(rvolTxt) --
  //   값·라벨(적음/보통/많음, q25 0.70 / q75 1.37)·주말 기준은 워커(live_eth_breakout_detector_20260911.py::_rvol)가 준다.
  // ── 실시간 호가 띠 (2026-09-27 사용자 지시) ─────────────────────────────────────────
  // 서버가 준 **마지막 1초** 호가 열(부호: +매수 / −매도)을 풋프린트와 같은 가격축에 가로 막대로 그린다.
  //   길이 = √(수량/창 안 최대) -- 비례로 두면 제일 큰 벽 하나에 나머지가 선이 됐다(시안). 창 안 상위 10% 는 진하게 + 수량.
  // 🔴«지지·저항»이 아니다 -- 벽에 닿은 뒤 반등률 0.509(동전, 5.8일 71,293건, 09-20). 서술만 한다.
  //   수량은 수집기 래스터 값(칸 $0.5 합산)이고, 가격선은 이 띠를 가로질러 배지까지 이어진다(호가창 사다리처럼).
  // 2026-09-27 축적 호가를 입혔다(사용자 선택 A + «재깔림도»): 한 칸 = 두 겹.
  //   **심(진함) = 창 내내 한 번도 안 빠진 양(pers)** -- 버텨 온 벽 vs 방금 깔린 호가를 가른다(창 안 호가의 36% 는 창을 못 버틴다).
  //   **겉 = 지금 양, 그 위 빗금 농도 = 재깔림(refill/peak) 4단** -- 촘촘할수록 계속 다시 채워지는 자리, 빗금 없거나 성기면 한 번 깔리고 만 호가.
  //   🔴둘 다 «성격»의 서술이다 -- 지속률은 반등을 못 가렸다(상위−하위 +0.001, 09-20). 창 = 위 1h/2h/4h/12h 토글.
  const book = (BOOK_W || PROF) && latestFlowHeatmap && latestFlowHeatmap.book;
  const bookHidden = fpColR === "pos" && flowOn();   // 2026-10-28 오른쪽 기둥 = 포지션이면 호가 띠·호버만 숨김(위 막대 계기 4개는 그대로 -- 2차 검증)
  const acc = latestFlowHeatmap && latestFlowHeatmap.rows;
  // 칸별 값은 층 캐시 **밖에서** 매 렌더 계산한다(≈250칸) -- 캐시가 재사용돼도 호버 해석(bookInfo)이 같은 값을 본다.
  const bookInfo = (() => {
    if (!book || !book.q || !(book.bin_size > 0)) return null;
    // 2026-09-28 호가 칸을 **풋프린트(체결 기둥)와 같은 행**으로 묶는다(사용자 «y축이 벌어지면 막대가 작아진다») --
    //   0.5달러 칸 그대로면 창이 넓을 때 한 칸이 2~3px 로 눌렸다. 행 안에서 수량·버틴 양·재깔림(refill·peak)은 더하고,
    //   매수·매도가 한 행에 섞이면(현재가 행) 큰 쪽으로 칠한다. ○(접근행동)는 행 안 **가장 얇아진** 값(approachAt).
    const bin = book.bin_size, bs = Math.max(bin, fpRowSize || bin), x0 = PROF ? PROF.cx + PROF_GAP : tradeX0 + TRADE_W + 6, L = PROF ? w - mr - x0 - 8 : BOOK_W - 10;   // 휴대폰 = 아래 칸 등뼈 오른쪽(◌ 자리 8)
    const accAt = (key, p) => {               // 축적 통계는 창 전체를 접은 행 배열이라 제 격자(bin_lo) -- 가격으로 찾는다
      if (!acc || !acc[key] || !(acc.bin_size > 0)) return 0;
      const i = Math.round(p / acc.bin_size) - acc.bin_lo;
      return i >= 0 && i < acc[key].length ? acc[key][i] : 0;
    };
    const rowsBy = new Map();                 // 행 키 -> {b: 매수 합, a: 매도 합} 각 {q, pers, peak, refill}
    for (let i = 0; i < book.q.length; i++) {
      const v = book.q[i], p = (book.bin_lo + i) * bin;
      if (!v) continue;
      // 2026-10-11 호가 칸은 floor 격자([p, p+칸)) · 풋프린트 행은 버킷 가장자리(−fpRowOff) 격자라 칸 **가운데**로 행을 고른다 -- 같은 줄을 지키고
      //   행 = 칸이면 가운데가 경계에 걸려 위 행으로 간다(≤ 칸/2 = 옛 체결 기둥이 지던 그 어긋남). ponytail: 정확히 맞추려면 수집기가 round 격자로 저장해야 한다.
      const k = Math.floor((p + bin / 2 + fpRowOff + 1e-9) / bs), r = rowsBy.get(k) || { b: null, a: null }, side = v > 0 ? "b" : "a";
      const q = Math.abs(v), s = r[side] || (r[side] = { q: 0, pers: 0, peak: 0, refill: 0 });
      s.q += q; s.pers += Math.min(q, accAt("pers", p)); s.peak += accAt("peak", p); s.refill += accAt("refill", p);
      rowsBy.set(k, r);
    }
    const cells = [];
    rowsBy.forEach((r, k) => {
      const p = (k + 0.5) * bs - fpRowOff;      // 행 가운데(그림은 p ± bs/2 = 풋프린트 행과 같은 줄)
      if (p + bs / 2 < yMin || p - bs / 2 > yMax) return;
      const bid = (r.b ? r.b.q : 0) >= (r.a ? r.a.q : 0), s = bid ? r.b : r.a;
      cells.push({ p, q: s.q, bid, pers: s.pers, rw: s.peak > 0 ? s.refill / s.peak : 0,
                   apr: approachAt(acc, k * bs - fpRowOff, bs) });   // 접근행동(4h) -- 프로파일 ◌ 와 같은 값 · null = 모름
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
  const statsGeo = bookInfo && !(TRADE_W && splitR && flowOn()) ? (() => {   // 2026-09-30 넓은 화면은 ④로 옮겼다(2026-10-04 SOL·XRP 도)
    const sx = TRADE_W ? tradeX0 + 6 : ml, room = (TRADE_W ? w - mr : PROF ? w - mr - 60 : bookInfo.x0 - 8) - sx;   // 휴대폰 아래 칸이면 오른쪽 끝 60 = 맨 위 레벨 꼬리표(max pain 등) 자리
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
  cachedLayer("bookStrip", book ? objToken(book) + "|" + objToken(acc) + "|" + activeSnapshotAsset + "|" + fpRowSize + "|" + fpRowOff + "|" + bookHidden : "none", (g) => {   // 2026-09-30 신원 = 객체(히트맵 행이 bin_lo 그대로 바뀌어도 다시 그린다)
    if (!bookInfo) return;
    const { bs, x0, L, mx, cells } = bookInfo;
    const fmtQ = (q) => (q >= 1000 ? (q / 1000).toFixed(1) + "k" : q >= 10 ? q.toFixed(0) : q.toFixed(1));
    const head = (x, anchor, fill, str) => {
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", x); t.setAttribute("y", pTop - 4); t.setAttribute("font-size", "10");
      t.setAttribute("text-anchor", anchor); t.setAttribute("fill", fill); t.textContent = str;
      g.appendChild(t);
    };
    if (TRADE_W) {   // 2026-10-10 ETH 는 «← 체결» 자리가 [체결 | 포지션] 토글(풋프린트 블록이 매번 그린다) · SOL·XRP 는 글자 그대로
      if (!flowOn()) { head(x0 - 6, "end", "var(--amber)", "← 체결"); head(x0, "start", "var(--muted)", "호가 →"); }   // 2026-10-28 흐름 코인은 머리 토글이 두 기둥 이름(이 글자와 겹친다)
    }
    else if (PROF) { if (!flowOn()) { head(PROF.cx - PROF_GAP, "end", "var(--amber)", "← 체결"); head(x0, "start", "var(--muted)", "호가 →"); } }   // 흐름 코인은 머리 줄 토글이 왼쪽 이름
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
    base.setAttribute("y1", pTop); base.setAttribute("y2", pBot);
    base.setAttribute("stroke", "var(--soft-line)");
    g.appendChild(base);
    // 빗금 무늬 4단(촘촘 → 성김). 45° 선 -- 심(단색)·겉(연한 단색)과 겹쳐도 질감으로 갈린다.
    if (bookHidden) return;   // 여기부터 호가 띠(오른쪽 기둥이 포지션이면 그 자리)
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
      const yTop = Math.max(pTop, pyAt(c.p + bs / 2)), yBot = Math.min(pBot, pyAt(c.p - bs / 2));
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
  // ── 풋프린트 아래 = 사분면 리본 + 차트 둘 (2026-10-06 사용자 선택 A) ─────────────────────────
  // «사분면은 글자만 살려서 레짐처럼 리본, 차트는 OI·CVD 하나 / 고래·중형·리테일·거래대금 하나».
  //   리본: 색 = 델타 부호 · 신규(OI↑) = 꽉 찬 칸 · 정리(OI↓) = 테두리 칸 · OI 모름 = 흐린 칸(이름도 «매수/매도 우위»로 유보).
  //   ① CVD·OI ② 고래·중형·리테일 -- 60분 합을 줄마다 각자 축으로(10-08) + ② 에 거래대금 꺾은선 + 옅은 면(제 축, 바닥 = 0) -- 문법은 아래 «차트 둘» 주석.
  //   60분 합 = 그 봉까지 12봉, 다 있어야 점(10-05 B안 규약 그대로 -- 일부 합을 60분으로 말하지 않는다).
  //   봉별 Δ/OI 숫자 줄·창 시작 누적 흐린 선은 뺐다 -- 봉 호버 툴팁(laneTip)이 그 봉 값을 글자로 말한다.
  // 🔴OI 모름(null)과 ΔOI 0 을 가른다(2026-09-25) -- 없는 봉에 «신규 롱/숏» 이라는 없는 사실을 안 찍는다.
  const QNAME = (d, o) => (o == null ? (d >= 0 ? "매수 우위" : "매도 우위")
                                     : d >= 0 ? (o >= 0 ? "신규 롱" : "숏 정리") : (o >= 0 ? "신규 숏" : "롱 정리"));
  const sgnQ = (v) => (v == null ? "—" : (v >= 0 ? "+" : "-") + fmtFootprintQty(Math.abs(v)));
  const obAgg = isSnapshotChart && ob5mFor() ? ob5mByTf(ob5mFor()) : null;   // 2026-10-26 5분 호가 요약(tf 봉별) -- 레인 막대·툴팁
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
      return { t: c.time, turn, delta: f.whale + f.mid + f.retail,
               oi: oiByTs.has(c.time) ? (Number(oiByTs.get(c.time)) || 0) : null };
    });
  })());
  const obTip = (i) => {
    const a = obAgg && candles[i] && obAgg.get(candles[i].time);
    if (!a) return "";
    const t = a.tilt == null ? "모름" : `${a.tilt >= 0 ? "+" : ""}${a.tilt.toFixed(1)}% (${a.tilt >= 0 ? "아래 매수 잔량이 두껍다" : "위 매도 잔량이 두껍다"})`;
    const n = (k) => (a[k] || a["p" + k] ? `${a[k] + a["p" + k]}${a["p" + k] ? `(임시 ${a["p" + k]})` : ""}` : "");
    const ev = [n("up") && `▲ 매도벽 돌파 ${n("up")}`, n("dn") && `▼ 매수벽 돌파 ${n("dn")}`, n("fu") && `◆ 위 가짜벽 의심 ${n("fu")}`, n("fd") && `◆ 아래 가짜벽 의심 ${n("fd")}`].filter(Boolean).join(" · ");
    const prov = a.pup + a.pdn + a.pfu + a.pfd > 0;
    return `<br>호가 기울기 ${t}${a.live ? " · 진행 중(봉 끝에 확정)" : ""}${ev ? `<br>${ev}${prov ? " · 임시 = 봉 끝 +70초에 확정" : ""}` : ""}`;
  };
  laneTip = (i) => {
    if (!fpBars.length) return "";
    const r = laneBarsOf()[i], col = (v) => (v >= 0 ? "var(--good)" : "var(--bad)");
    if (!r) return "<br><br>풋프린트 없음";
    return `<br><br><b style="color:${col(r.delta)}">${QNAME(r.delta, r.oi)}</b> · 델타 ${sgnQ(r.delta)} · 신규계약 ${r.oi == null ? "모름" : sgnQ(r.oi)} ${coinUnit()}`
      + `<br><span style="color:var(--turnover)">거래대금 ${fmtUsdCompact(r.turn)}</span>` + obTip(i);   // 2026-10-26 펜 메모 «이 박스는 이제 제거»: 고래·중형·리테일·60분 합 줄은 다섯 줄과 같이 뺐다
  };
  // 2026-10-20 리본·막대·0선 = fg 캔버스(칠하기 하나 -- 형성 봉 갱신은 플롯 안만 다시 칠해 여기는 전체 렌더 박자) · 글자(리본 이름·줄 이름·값) = svg.
  //   옛 층 캐시(quadLane)는 뺐다 -- 남은 svg 노드는 글자 수십 개라 매 렌더 만들어도 싸다.
  if (fpBars.length && candles.length && QUAD_H) (() => {
    const rows = laneBarsOf(), have = rows.filter(Boolean);
    if (!have.length) return;
    const g = document.createElementNS(NS, "g");
    svg.appendChild(g);
    const ops = [];   // 캔버스 칠하기(순서 = 옛 svg 순서)
    fgI.push({ x0: -1e9, x1: 1e9, y0: quadY, y1: hAll, c: laneC, p: (ctx) => ops.forEach((f) => f(ctx)) });
    const mk = (tag, at, parent = g) => { const e = document.createElementNS(NS, tag); for (const k in at) e.setAttribute(k, at[k]); parent.appendChild(e); return e; };
    const txt = (x, y, s, at = {}) => { const t = mk("text", Object.assign({ x, y, "font-size": mobileChart ? 10 : 12 }, at)); t.textContent = s; return t; };
    const top = quadY, bot = h - mb - 4, RIB = mobileChart ? 14 : 22, GAP = mobileChart ? 4 : 8;
    const cx = (i) => xAt(i) + bw / 2, slotW = slot;
    const chartTop = top + RIB + GAP;
    const ribCol = slot < 1 ? new Map() : null;   // 2026-10-20 1px 밑이면 픽셀 열마다 마지막 봉(겹쳐 칠할 때 맨 위에 오던 것) 하나
    const un = cvUnion();   // DENSE 리본·막대 = 색·농도별 한 칠(cvUnion)
    const ribbon = (x, wd, r) => un.add(r.delta >= 0 ? "var(--good)" : "var(--bad)", r.oi == null ? 0.12 : r.oi >= 0 ? 0.85 : 0.4, (pp) => pp.rect(x, top, wd, RIB));
    rows.forEach((r, i) => {
      if (!r) return;
      const col = r.delta >= 0 ? "var(--good)" : "var(--bad)", fresh = r.oi != null && r.oi >= 0;
      const x = xAt(i) + 1, wd = Math.max(1, slotW - 2);
      // DENSE(1주일 창): 1px 칸엔 테두리가 안 보여 «정리»(테두리 칸)는 중간 농도(.4)로
      if (ribCol) { ribCol.set(Math.round(xAt(i)), i); return; }
      if (DENSE) { ribbon(xAt(i), Math.max(0.6, slotW), r); return; }
      ops.push((ctx) => {
        cvRect(ctx, x, top, wd, RIB, col, r.oi == null ? 0.12 : fresh ? 0.85 : 0.14, 3);
        if (r.oi != null && !fresh) cvStrokeRect(ctx, x, top, wd, RIB, col, 1.2, 0.8, 3);
      });
      if (slotW >= (mobileChart ? 40 : 52)) txt(x + wd / 2, top + RIB / 2 + (mobileChart ? 3.5 : 4.5), QNAME(r.delta, r.oi),
        { "text-anchor": "middle", "font-weight": 700, "font-size": mobileChart ? 9 : 12, fill: fresh ? "var(--on-fill)" : col, "pointer-events": "none" }).setAttribute("clip-path", laneClip || "none");
    });
    if (ribCol) ribCol.forEach((i, k) => ribbon(k, 1, rows[i]));   // 열 = 1px 칸
    un.flush((f) => ops.push(f));
    // 2026-10-26 사용자 «거래대금 막대 자리에 기울기 막대, 거래대금은 선»: 0선 위아래 = 5분 호가 기울기(±50bp 매수 − 매도 잔량, 봉 평균) · 파랑 선 = 거래대금(제 축, 바닥 = 0).
    //   옛 봉별 거래대금 막대·프로파일 아래 값 칸(대금 · 이 봉 · 창 합)은 뺐다 -- 봉 값은 호버 툴팁(laneTip). 이름은 왼쪽 위 한 줄(오래된 봉 쪽 -- 최신 봉을 안 가린다).
    const LH = bot - chartTop, y0 = chartTop, zero = y0 + LH / 2;
    const bwid = Math.max(1, Math.min(slotW * 0.62, slotW - 2));
    const tilts = candles.map((c) => { const a = obAgg && obAgg.get(c.time); return a ? a.tilt : null; }), tOk = tilts.filter((v) => v != null);
    const tLive = candles.map((c) => { const a = obAgg && obAgg.get(c.time); return !!(a && a.live); });   // 진행 중 봉 = 흐리게(봉 끝에 확정값)
    const tMax = Math.max(5, ...tOk.map(Math.abs)), tSc = (LH / 2 - 1) / tMax;
    const colMax = slot < 1 ? new Map() : null;   // 1px 밑(1주일 창): 픽셀 열마다 가장 긴 막대 하나
    const bar = (x, v, rx = 0, al = 0.7) => {
      const hh = Math.max(1, Math.abs(v) * tSc), y = v >= 0 ? zero - hh : zero, col = v >= 0 ? "var(--ob-bid)" : "var(--ob-ask)";
      if (DENSE) un.add(col, al, (pp) => pp.rect(x, y, bwid, hh)); else ops.push((ctx) => cvRect(ctx, x, y, bwid, hh, col, al, rx));
    };
    ops.push((ctx) => cvHLine(ctx, ml, ml + cw, zero, "var(--muted)", 0.35));
    tilts.forEach((v, i) => {
      if (v == null) return;                        // 북 모름은 칸을 비운다 -- 0 으로 그리지 않는다
      const x = cx(i) - bwid / 2;
      if (colMax) { const k = Math.round(x), o = colMax.get(k); if (!o || Math.abs(o[1]) < Math.abs(v)) colMax.set(k, [x, v, tLive[i]]); return; }
      bar(x, v, DENSE ? 0 : 1.5, tLive[i] ? 0.32 : 0.7);
    });
    if (colMax) colMax.forEach(([x, v, lv]) => bar(x, v, 0, lv ? 0.32 : 0.7));
    un.flush((f) => ops.push(f));
    // 거래대금 선 -- 값 없는 봉에서 끊는다(모름을 이어 그리지 않는다)
    const turns = rows.map((r) => (r ? r.turn : null)), uMax = Math.max(1e-9, ...turns.filter((v) => v != null)), uSc = (LH - 2) / uMax;
    const runs = [];
    let cur = null;
    turns.forEach((v, i) => { if (v == null) { cur = null; return; } if (!cur) runs.push((cur = [])); cur.push([cx(i), y0 + LH - 1 - v * uSc]); });
    ops.push((ctx) => runs.forEach((pts) => (pts.length > 1 ? cvStroke(ctx, pts, "var(--turnover)", 1.4, 0.9, null, "round") : cvRect(ctx, pts[0][0] - 1, pts[0][1] - 1, 2, 2, "var(--turnover)", 0.9))));
    const lg = txt(ml + 4, y0 + (mobileChart ? 8 : 9), "", { "font-size": mobileChart ? 9 : 10, fill: "var(--muted)", "pointer-events": "auto" });
    [[tOk.length ? "■ 호가 기울기 ±50bp" : "■ 호가 기울기 (준비 중)", "var(--muted)"], ["  ─ 거래대금", "var(--turnover)"]].forEach(([t, f]) => { const sp = mk("tspan", { fill: f }, lg); sp.textContent = t; });
    { const tt = document.createElementNS(NS, "title");
      tt.textContent = "호가 기울기 = 현재가 ±50bp 안 매수 잔량 − 매도 잔량 비율(봉 안 1초 평균). 위(초록) = 아래쪽 매수 잔량이 두껍다 · 아래(빨강) = 위쪽 매도 잔량이 두껍다.\n"
        + "가격이 내린 뒤엔 아래 잔량이 두꺼워 보인다(직전 1시간 수익과 순위상관 −0.44) — «오를 것»이 아니다. 다음 봉과 상관 0.03. 정의 검증 중 · 정보 표시용.\n흐린 막대·표식 = 진행 중 봉(5초마다 갱신 · 봉 끝에 확정, 표식은 +70초).\n파랑 선 = 봉별 거래대금(바이낸스+OKX 체결, 제 축).";
      lg.append(tt); }
  })();

  // 2026-09-30 지지·저항 글자 줄은 시장 맥락 ③ 가격 지형으로 옮겼다(사용자 지시) -- mcGfx.terrain 이 srLevelsLive() 를 그린다.


  // ── ③ 청산 — 풋프린트 봉 고가 «바로 위» 동그라미 (2026-09-22 사용자 지시) ────
  // 레인에서 뺐다. 청산은 **가격에서 일어나는 사건**이라 가격 옆에 있어야 하고, 레인에
  // 두면 어느 봉의 청산인지 눈이 세로로 훑어야 한다.
  // 🔴크기는 √다. 5분봉 청산은 중앙 $211 / 최대 $4.9M 로 23,000배라 선형이면 큰 것 하나만
  //   남는다(옛 레인이 로그를 쓴 이유와 같다). 반지름이라 √면 **면적이 금액에 비례**한다.
  // 2026-10-20 fg 캔버스 -- 봉마다 칠하기 하나(형성 봉은 고가가 움직이면 원도 따라 오른다: liqLive). 자르기 = 가격 플롯(양 끝에 걸친 봉의 원이 여백으로 나가지 않게).
  //   봉별 <title>(청산 금액·롱/숏·거래소)은 봉 호버 툴팁의 «청산» 줄로 옮겼다(liqTip).
  let liqTip = () => "";
  const liqTopAt = new Map();   // 봉 시각 -> 청산 원 윗끝 y (5분 호가 ▲ 가 원 위로 비킨다)
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
      const liqItemOf = (i) => {
        const c = candles[i], b = byTs.get(c.time);
        if (!b) return { x0: 1, x1: 0, y0: 1, y1: 0, p: () => {} };
        const lo = Number(b.long_usd) || 0, sh = Number(b.short_usd) || 0;
        const v = lo + sh;
        if (v <= 0) return { x0: 1, x1: 0, y0: 1, y1: 0, p: () => {} };
        const r0 = slot < 4 ? 1.5 : 3, r1 = slot < 4 ? rMax * 0.6 : rMax;   // 2026-10-20 1주일 창은 원 수백 개가 캔들을 덮어 작게(크기는 옛 4px 문턱 그대로)
        const r = r0 + Math.sqrt(v / peak) * (r1 - r0);
        // 그 봉이 그린 것 중 가장 높은 것 바로 위(셀 숫자 한 줄 9px 만큼 더 띄운다).
        const top = Math.max(c.high, topByTs.get(c.time) || -Infinity);
        const cy = Math.max(mt + r + 1, yAt(top) - r - 14), cxD = xAt(i) + bw / 2, col = sh >= lo ? "var(--good)" : "var(--bad)";
        liqTopAt.set(c.time, cy - r);
        return { x0: cxD - r - 1, x1: cxD + r + 1, y0: cy - r - 1, y1: cy + r + 1, c: plotC, cx: cxD, cy, r, col, p: (ctx) => {
          ctx.globalAlpha = 0.9; ctx.fillStyle = cvc(col); ctx.beginPath(); ctx.arc(cxD, cy, r, 0, 2 * Math.PI); ctx.fill();
        } };
      };
      if (DENSE) {   // 1주일 창: 색별 한 칠(옛 path 하나 -- 겹친 원이 진해지지 않는다) · 형성 봉 원은 전체 렌더 박자
        const un = cvUnion();
        candles.forEach((c, i) => { const it = liqItemOf(i); if (it.r) un.add(it.col, 0.9, (pp) => { pp.moveTo(it.cx + it.r, it.cy); pp.arc(it.cx, it.cy, it.r, 0, 2 * Math.PI); }); });
        un.flush((f) => fgI.push({ ...CV_ALL, c: plotC, p: f }));
      } else candles.forEach((c, i) => { if (i === candles.length - 1) liqLive = { at: fgI.length, of: liqItemOf }; fgI.push(liqItemOf(i)); });
      liqTip = (i) => {
        const c = candles[i], b = c && byTs.get(c.time);
        if (!b) return "";
        const lo = Number(b.long_usd) || 0, sh = Number(b.short_usd) || 0;
        return lo + sh > 0 ? `<br>청산 ${fmtUsdCompact(lo + sh)} (롱 ${fmtUsdCompact(lo)} / 숏 ${fmtUsdCompact(sh)})${b.partial ? " · 진행 중" : ""}${liqVenueText(b).replace(/\n/g, "<br>")}` : "";
      };
    }
  }

  // ── 5분 호가 요약 덧그림(2026-10-26) -- 벽 = 쪽별 점선(초록 매수·빨강 매도) · ▲▼ 돌파 · ◆ 가짜벽 의심. 위치 참고, 신호 아님.
  const OB5 = isSnapshotChart && ob5mFor();
  if (OB5 && candles.length) {
    const agg = ob5mByTf(OB5), idx = new Map(candles.map((c, i) => [c.time, i])), t0 = candles[0].time, tN = candles[candles.length - 1].time;
    const ext = new Map();   // 봉이 실제로 그린 가장 높은/낮은 가격(풋프린트 칸이 고저 밖까지 간다 -- 청산 원과 같은 이유)
    fpBars.forEach((b) => { let hi = -Infinity, lo = Infinity; (b.levels || []).forEach((l) => { const pz = Number(l[0]); if (pz > hi) hi = pz; if (pz < lo) lo = pz; }); ext.set(Number(b.time), [hi, lo]); });
    const segs = [];
    OB5.walls.forEach((w) => {
      const tS = w.t_start / 1000, tE = w.t_end / 1000 - 300;
      if (tE < t0 || tS > tN) return;
      const i0 = tS <= t0 ? 0 : idx.get(tfT(tS)), i1 = tE >= tN ? candles.length - 1 : idx.get(tfT(tE));
      if (i0 == null || i1 == null) return;
      segs.push([xAt(i0) + 1, xAt(i1) + bw - 1, yAt(w.px), w.side === "bid" ? "var(--ob-bid)" : "var(--ob-ask)"]);
    });
    const s = DENSE ? 4 : 5, boxes = bw >= 14 && !fpBoxesOff && fpBars.length > 0;   // 체결 칸이 있으면 위 = 칸 위 숫자 한 줄·청산 원 위 · 아래 = 봉 델타 라벨 밑
    const tri = (ctx, x, y, dir) => { ctx.beginPath(); ctx.moveTo(x - s, y); ctx.lineTo(x + s, y); ctx.lineTo(x, y + dir * s * 1.6); ctx.closePath(); ctx.fill(); };
    const dia = (ctx, x, y) => { ctx.beginPath(); ctx.moveTo(x, y - s); ctx.lineTo(x + s, y); ctx.lineTo(x, y + s); ctx.lineTo(x - s, y); ctx.closePath(); ctx.fill(); };
    fgI.push({ ...CV_ALL, c: plotC, p: (ctx) => {
      segs.forEach(([x0, x1, y, col]) => cvStroke(ctx, [[x0, y], [x1, y]], col, 2.5, 0.9, "2 3"));
      candles.forEach((c, i) => {
        const a = agg.get(c.time);
        if (!a) return;
        const up = a.up || a.pup, dn = a.dn || a.pdn, fu = a.fu || a.pfu, fd = a.fd || a.pfd;   // 그릴지 = 확정|임시 · 농도 = 확정이 하나라도 있으면 진하게
        if (!(up || dn || fu || fd)) return;
        const al = (k) => (a[k] ? 1 : 0.45);
        const [eh, el] = ext.get(c.time) || [-Infinity, Infinity], x = xAt(i) + bw / 2;   // 임시 판정 = 흐리게
        const yt = Math.min(yAt(Math.max(c.high, eh)) - (boxes ? 12 : 4), (liqTopAt.get(c.time) ?? Infinity) - 3), yb = yAt(Math.min(c.low, el)) + (boxes ? 30 : 4);
        if (up) { ctx.globalAlpha = al("up"); ctx.fillStyle = cvc("var(--ob-bid)"); tri(ctx, x, yt, -1); }
        if (dn) { ctx.globalAlpha = al("dn"); ctx.fillStyle = cvc("var(--ob-ask)"); tri(ctx, x, yb, 1); }
        ctx.fillStyle = cvc("var(--ob-fake)");
        if (fu) { ctx.globalAlpha = al("fu"); dia(ctx, x, yt - (up ? s * 1.6 + 2 : 0) - s); }
        if (fd) { ctx.globalAlpha = al("fd"); dia(ctx, x, yb + (dn ? s * 1.6 + 2 : 0) + s); }
      });
      ctx.globalAlpha = 1;
    } });
  }

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
      if (!hit) { g.replaceChildren(); draw(g); slotState.key = key; }   // 10-17: 키가 바뀌어 다시 그릴 때 옛 그림을 비운다(범례는 스스로 안 비워 폭·코인 바꿀 때마다 쌓였다 -- 중복 id)
      return g;
    };
    // 2026-09-19 히트맵도 같은 캐시를 쓴다 -- 래스터는 3초마다 새 열이 오는데 캔들 전체
    // 리렌더(가격 틱)를 기다릴 이유가 없다(2bb2b2f1 이 프로파일/1초수급에 넣은 그 이유).
    // 2026-09-30 시안 Y(사용자 선택): ETH 는 오른쪽 칸 위 절반만 1초 수급, 아래 절반은 시장 맥락(#mcBody 를 그 자리에 겹쳐 놓는다).
    const mcSplit = !!splitR && flowOn();   // 2026-10-04 SOL·XRP 도 시장 맥락(코인별)
    // 2026-09-30 1초 수급 높이 = 칸 − 시장 맥락의 **실제 내용 높이**(mcNeedH, renderMarketCtx 가 잰다) -- 한 화면 모드로 칸이 줄어도 시장 맥락이 스크롤 없이 다 들어간다.
    //   아직 못 쟀으면 47% · 1초 수급은 칸의 22% 아래로는 안 줄인다(2026-10-15 30 → 22%: ⑤ 예측 줄·⑥ 그래프 키운 몫, 사용자 «모자라면 수급 차트 높이를 줄여»).
    // 2026-10-01 바닥 = 전환 리본 바닥(사용자 «전환 리본까지 나머지 바닥을 맞춰») -- 시장 맥락·모의 판(옛 옵션 요약) 칸이 같은 선에서 끝난다
    const floorY = hAll - 2;   // 2026-10-06 리본 줄이 빠져 칸 바닥 = 상자 바닥
    const s1Avail = floorY + 12 - mtTop - 4;   // +10 = 시장 맥락 내용 높이의 여유(+6)·칸 아래 여백 -- 마지막 줄 글자가 리본 바닥에 닿게
    const s1H = splitR ? (mcSplit ? Math.max(Math.round(s1Avail * 0.22), mcNeedH ? Math.round(s1Avail - mcNeedH - 16 - 8) : Math.round(s1Avail * 0.47)) : s1Avail) : S1_BELOW ? S1_PANEL : SUB_1S_H - STATS_ROW_H;   // 2단: 오른쪽 칸(다섯 줄이 고르게 나눈다)
    { const fs = el("fpLineSwitch"), card = el("fpCard");   // 청산 밀도 범례(왼쪽 위, ~256px) 오른쪽
      if (fs && card) {
        const on = mcSplit && SUB_LEGEND_H > 0;
        fs.classList.toggle("on", on);
        // 2026-10-16 펜 메모: 차트 바로 위 한 줄 = 청산 밀도(왼쪽) | 창 미니맵·실시간(가운데) | VWAP·세션·볼린저·주간 앵커(오른쪽 = 가격판 오른쪽 끝)
        const pan = el("chartPan"), home = el("fpPanHome");
        if (pan) {
          if (on && pan.parentElement !== card) card.appendChild(pan);
          else if (!on && home && pan.previousElementSibling !== home) home.after(pan);
          pan.classList.toggle("inrow", on);
        }
        if (on) post.push(() => {   // 화면 상자를 재므로 끝에서(post)
          // 기준 = 왼쪽 칸 전체(풋프린트 + 호가 막대): 왼쪽 끝 = 범례 시작(카드 왼쪽 17) · 오른쪽 끝 = 오른쪽 칸(subX) 앞 20px
          const a = svg.getBoundingClientRect(), c = card.getBoundingClientRect(), y = `${Math.round(a.top - c.top + subLegendY + SUB_LEGEND_H / 2)}px`;
          const colL = 17, colR = a.left - c.left + subX - 20;
          fs.style.top = y; fs.style.right = `${Math.round(c.width - colR)}px`;
          if (pan) { pan.style.top = y; pan.style.left = `${Math.round(colL + pan.offsetWidth / 2)}px`; }   // 2026-10-28 펜 메모: 미니맵 = 옛 청산 밀도 자리(왼쪽 끝)
          const lb = fpLineBtnEl(card);   // 2026-10-28 펜 메모 «주문은 풋프린트 바로 위 가운데에 버튼으로 크게»
          lb.style.top = y; lb.style.left = `${Math.round((colL + colR) / 2)}px`;
        }); else if (pan) { pan.style.top = pan.style.left = ""; }
        card.classList.toggle("fp-toprow", on);   // 이 줄이 있으면 체결 기둥 토글의 작은 «선 주문»은 숨긴다(큰 버튼 하나)
        fpMobRowSet(!on && mobileChart && flowOn() && !fpAltOn(), card, fs);   // 2026-10-28 휴대폰 = 머리 마지막 줄 [· | SLTP | 지표]
      } }
    // 2026-10-15 아티팩트 댓글 «봉 시간은 수급 차트 위로, 남는 공간은 미니 체결 버블»: 수급 바닥 시각 줄(12px, renderSupply1s mb 16→4)을 버블에 넘긴다
    const s1Give = mcSplit ? 12 : 0, s1Hd = s1H - s1Give;
    post.push(() => mcPlace(svg, mcSplit ? { x: subX, y: sub1sY + s1Hd + 12, w: subW - 16, h: floorY - (sub1sY + s1Hd + 12) } : null,
            mcSplit && TRADE_W ? (() => { const y = plotBottom + 12;   // 2026-10-01(4) 프로파일 아래 = 2026-10-04 모의 판(renderPaper 가 채운다)
              return { x: ml + cw + 4, y, w: TRADE_W + BOOK_W - 8, h: floorY - y }; })() : null));   // 10-26 레인 값 칸(옛 LANE_VAL_W 144)을 빼 지지/저항 칸이 제자리로
    supply1sSubBox = {
      svg: subSvg("s1", subX, sub1sY, subW, s1Hd, sub1sKey(subW, s1Hd),
                  (g) => renderSupply1s({ svg: g, w: subW, h: s1Hd })),
      w: subW, h: s1Hd };

    // ── 청산 밀도 범례 (2026-09-21 아티팩트 댓글) ────────────────────────────
    // 전에는 헤더 행의 HTML(#liqDensityLegend)이었다. 밀도가 풋프린트의 **배경**이 된
    // 뒤로는 설명하는 그림에서 멀어졌으므로 프로파일 바닥글과 풋프린트 사이로 내렸다.
    // 글자 9 = 바닥글과 같은 크기(사용자 지시).
    // 🔴그라디언트는 densityStops() 에서 바로 만든다 -- 사본을 두면 히트맵과 어긋나도
    //   아무도 모른다(2026-09-12 에 실제로 겪었다).
    // 2026-10-28 펜 메모 «왼쪽 청산밀도를 제거하고 미니맵을 그 자리로»: 넓은 화면(splitR)은 범례를 안 그린다 -- 휴대폰·좁은 화면은 그대로
    if (SUB_LEGEND_H && !splitR && !(mobileChart && flowOn()) && (densityHistory || []).length) {   // 10-28 휴대폰 흐름 코인도(사용자 «모바일도 데스크탑처럼»)
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
  if (footprint && svg.id === "candleSvgSnapshot" && (activeSnapshotAsset === "eth" || activeSnapshotAsset === "xrp") && candles.length > 1) iceMarks(svg, candles, xAt, yAt, bw, yMin, yMax, NS);   // 2026-10-16 시안 ①
  { const mn = mobileChart && svg.id === "candleSvgSnapshot" ? svg.querySelector('[data-live="label"][data-with-price="num"]') : null;
    if (mn && !mn.classList.contains("far-lbl")) svg.appendChild(mn); }   // 2026-10-10 휴대폰 현재가 숫자 = 호가 띠(◌ 표식) 위 맨 앞
  fpCvPaint(fpCv.bg); fpCvPaint(fpCv.fg);   // 2026-10-20 캔버스 두 장(fpCv) -- 칠하기는 레이아웃을 안 건드린다
  if (svg.id === "candleSvgSnapshot") post.push(renderLineTool);   // 2026-10-28 선 주문 상자 자리
  post.forEach((f) => f());   // 화면 상자를 재는 HTML 배치(위 ③) -- svg 를 다 쌓은 뒤 한 번만 배치된다
  // 2026-10-01 프로파일 오른쪽 끝 가격 이름과 겹치는 호가 벽 수량 글자(«11.8k»)는 숨긴다(사용자 지시) -- 층 캐시라 매번 되살린 뒤 다시 판정
  //   (2026-10-20 읽기를 먼저 다 하고 쓰기를 나중에 -- 글자마다 되살리고 재면 그때마다 레이아웃이 다시 돈다)
  if (svg.id === "candleSvgSnapshot") {
    const qs = [...svg.querySelectorAll(".pf-qty")];
    qs.forEach((q) => q.removeAttribute("visibility"));
    const far = [...svg.querySelectorAll(".far-lbl")].map((n) => n.getBoundingClientRect()).filter((r) => r.width > 0);
    if (far.length) qs.map((q) => [q, q.getBoundingClientRect()]).forEach(([q, b]) => {
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

  // 그리기는 스냅샷 탭이 보일 때만(아니면 snapshotTabPanel 이 display:none).
  if (activePageTab === "snapshot") {
    renderOfab();                  // 2026-09-25 떠다니는 주문 버튼 글자(포지션·손익)

    // 현재가 선은 SSE 틱마다 바뀌는 latestLivePriceByAsset 을 읽는다 -- 5분 주기 데이터 갱신만
    // 기다리지 않고 여기서도 다시 그린다(chartRenderGateMs() 스로틀, 캐시된 데이터만 다시 그림).
    const nowForSnapshotChart = Date.now();
    if (nowForSnapshotChart - lastSnapshotChartRenderAt >= chartRenderGateMs()) {
      lastSnapshotChartRenderAt = nowForSnapshotChart;
      updateSnapshotCandleLive();
      renderSnapshotChart();
      /* (아래 목록 갱신은 이 경로에만 있다 -- 시세 푸시용 경로는 maybeRenderSnapshotChartNow) */
      // 지지/저항 목록도 이미 지나간 레벨을 현재가로 다시 거른다 -- 차트와 같은 스로틀·같은 스냅샷.
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
    refreshNotifyCenter();           // 2026-10-05 종 옆 점(판정일·새 알림) -- 탭과 무관 (자체 5분 게이트)
    // 2026-08-25 perf pass (Snapshot's 6 fetches), extended 2026-08-31 to Ops's own status poll
    // now that the Live tab (previously the 3rd, always-unconditional, tab) is gone -- each branch
    // only matters while that tab is actually visible; gating stops background fetch/compute work
    // for a hidden panel (see activePageTab, set by setupPageTabs()'s click handler). Both tabs'
    // click handlers already force an immediate refresh on switching to them, so this doesn't delay
    // first paint after a tab switch -- it only stops the ongoing poll while elsewhere.
    if (activePageTab === "ops") {
      refreshOpsStatus();
    } else if (activePageTab === "snapshot") {
      refreshBinanceAccount();       // 2026-09-10 청산맵 위 계좌 요약 + 진입선 (자체 30초 게이트)
      refreshChartMarkers();         // 2026-09-09 청산맵 신호 마커
      refreshLiquidation5mSignal();
      refreshLiquidation5mTail();    // 2026-09-25 청산 원 최신 2봉 (자체 2초 게이트)
      refreshLiquidationMap();
      refreshMacroCalendar();
      refreshNews();                 // 2026-10-05 뉴스 카드 (자체 30초 게이트)
      refreshMacroTrend();           // 2026-10-28 Macro Trend 카드 (자체 10분 게이트)
      refreshMtToday();              // 2026-10-28 Macro Trend «오늘» 칸 (자체 1분 게이트 · 국채 24시간 5분)
      refreshTreasury();             // 2026-10-06 풋프린트 머리 띠 국채선물 (자체 60초 게이트)
      refreshXyz();                  // 2026-10-14 나스닥100 실시간(XYZ100, 자체 1초 게이트)
      refreshMacroSpikes();          // 2026-10-14 «⑥» 급변 칩(자체 30초 게이트)
      refreshSessionAlerts();
      refreshFootprint();            // 2026-09-15 볼륨 풋프린트 체결 테이프
      maybeFetchFpWeek();            // 2026-10-20 1주일(12시간 밖이 필요할 때 한 번)
      refreshOb5m();                 // 2026-10-26 5분 호가 요약(자체 15초 게이트)
      refreshFlowHeatmap();          // 2026-09-19 호가 히트맵(프로파일 왼쪽 절반)
      refreshGex();                  // 2026-09-19 옵션 감마 노출(참고 표시 · 신호 아님)
      refreshSupply1s();             // 2026-09-19 최근 5분 x 1초 수급
      refreshOi5m();                 // 2026-09-19 OI 신규계약 5분 누적 (자체 15초 게이트)
      refreshPositionProfile();      // 2026-10-10 체결 기둥 «포지션» 보기 (켤 때만 1초)
      refreshSituation();            // 2026-09-21 상황 읽기 · 30분 (5초, ETH 만)
      refreshTrend();                // 2026-09-28 30분 카드 추세 칸 (60초, 일봉)
      refreshIceberg();              // 2026-10-16 풋프린트 칸별 아이스버그 (30초, ETH · 검증 중 · 신호 아님)
      refreshDirSig();               // 2026-10-15 지평 사다리 7일·4~13주 (10분, 서버 1시간 캐시)
      refreshVol4h();                // 2026-10-15 ⑤ 예측 판 4h 변동성 (5분, ETH)
      refreshDir4h();                // 2026-10-15 ⑤ 예측 판 4h 방향 (2분, ETH · 서버 5분 캐시)
      refreshDirShort();             // 2026-10-28 ⑤ 예측 판 15분·1시간 방향 (1분, ETH · 서버 5분 캐시)
      refreshBoxHold();              // 2026-10-16 박스 유지 확률 (1분, ETH · 방향 아님)
      refreshOpenRev();              // 2026-10-16 ⑤ 종합의 09시 개장 되돌림 (2분, ETH)
      refreshMarketCtx();            // 2026-09-29 시장 맥락 카드 (1초 -- 10-03, ETH 만 · 서술)
      refreshPaper();                // 2026-10-04 모의 판·맞대결·벽 신호 (10초, 모의 매매 엔진 상태)
      refreshChipScore();            // 2026-10-05 머리 칩 실전 성적 (코인별 5분)
      ensurePriceWs();               // 2026-09-16 현재가 직결 WS (탭/코인/가시성 변화가 여기로 수렴)
      maybeFetchSnapshotChartHistory();
    }
  } catch (e) {
    console.error("Tick Error:", e);
  } finally {
    tickInFlight = false;
  }
}

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
placeTopLeft();
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
    sub_text = "포지션 위험·보유 중 청산 급증은 소리와 함께, 판정일은 무음으로 옵니다."; label = "알림 끄기";
  } else {
    state = "neutral"; headline = "알림이 꺼져 있습니다";
    sub_text = "켜면 이 기기로 신호가 도착합니다. 창을 닫아도 옵니다."; label = "알림 켜기";
  }

  if (title) { title.textContent = headline; title.dataset.tone = state; }
  if (detail) detail.textContent = sub_text;
  if (btn) { btn.textContent = label; btn.disabled = disabled; btn.classList.toggle("primary", !sub); }
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
    // 🔴브라우저는 구독 중인데 서버 목록에 없으면(410 정리·파일 유실) 다시 등록한다 -- 10-07 뒤 모든 푸시가 0대로 갔다
    const sub = notifyState.subscription;
    if (sub && !notifyState.devices.some((d) => d.endpoint_tail === sub.endpoint.slice(-12))) {
      await fetch("/api/push/subscribe", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subscription: sub.toJSON(), label: navigator.userAgent.slice(0, 80) }),
      });
      notifyState.devices = (await (await fetch("/api/push/devices", { cache: "no-cache" })).json()).devices || [];
    }
  } catch (e) { notifyState.devices = notifyState.devices || []; }
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

// ── 알림 센터 (2026-10-05, 사용자 «글래스/다크 모드 옆 알림 아이콘 → 팝업 모달 · 판정 예정일 + 푸시 알림») ──
// 원천 /api/notify-center = 판정 달력(dashboard/verdict_calendar.json) + 데몬이 보내는 종류(PUSH_KINDS) + 최근 보낸 알림.
// 종 옆 점 = 판정일이 내일·오늘이거나 지났는데 아직 판정 전 · 또는 마지막으로 연 뒤 새로 보낸 알림이 있을 때.

function ncDaysUntil(dateStr) {   // 판정일(UTC 날짜 00시 = 09:00 KST)까지 보는 사람 달력으로 며칠
  const d = new Date(`${dateStr}T00:00:00Z`), t = new Date();
  return Math.round((new Date(d.getFullYear(), d.getMonth(), d.getDate()) - new Date(t.getFullYear(), t.getMonth(), t.getDate())) / 86400000);
}

function ncSeenAt() { try { return Number(localStorage.getItem(NC_SEEN_KEY)) || 0; } catch (e) { return 0; } }

function renderNotifyCenter() {
  const d = notifyCenterData || {};
  const vs = d.verdicts || [], recent = d.recent || [];
  const now = new Date(), DOW = "일월화수목금토";
  const vBox = el("ncVerdicts");
  if (vBox) {
    vBox.innerHTML = vs.length ? vs.map((v) => {
      const n = ncDaysUntil(v.date), dt = new Date(`${v.date}T00:00:00Z`);
      const yr = dt.getFullYear() !== now.getFullYear() ? `<span>${dt.getFullYear()}년</span>` : "";   // 다른 해면 한 줄 더(좁은 칸에서 어색하게 안 꺾이게)
      const when = v.approx ? `${dt.getMonth() + 1}월 무렵` : `${dt.getMonth() + 1}월 ${dt.getDate()}일 ${DOW[dt.getDay()]}`;
      const tag = v.done ? "판정 끝" : v.approx ? "건수 기준" : n > 0 ? `D-${n}` : n === 0 ? "오늘" : "판정 대기";
      const tone = v.done ? "done" : v.approx ? "" : n <= 0 ? "due" : n <= 7 ? "soon" : "";
      return `<li class="nc-v ${tone}">
        <div class="nc-v-when"><b>${escapeHtml(tag)}</b>${yr}<span>${escapeHtml(when)}</span></div>
        <div class="nc-v-main">
          <p class="nc-v-title"><strong>${escapeHtml(v.title || "")}</strong><span class="nc-kind">${escapeHtml(v.kind || "")}</span></p>
          <p class="nc-v-what">${escapeHtml(v.what || "")}</p>
          ${v.done ? `<p class="nc-v-done">${escapeHtml(v.done)}</p>` : ""}
          <code class="nc-v-how" title="${escapeHtml(v.doc || "")}">${escapeHtml(v.how || "")}</code>
        </div></li>`;
    }).join("") : `<li class="nc-empty">예정된 판정이 없습니다.</li>`;
  }
  const kBox = el("ncKinds");
  if (kBox) kBox.innerHTML = (d.kinds || []).map((k) => `<li><b>${escapeHtml(k.name)}</b><span>${escapeHtml(k.when)}</span></li>`).join("");
  const rBox = el("ncRecent");
  if (rBox) {
    rBox.innerHTML = recent.length ? recent.slice(0, 12).map((r) => {
      const t = new Date(r.ts);
      const at = Number.isNaN(t.getTime()) ? "" : `${t.getMonth() + 1}/${t.getDate()} ${String(t.getHours()).padStart(2, "0")}:${String(t.getMinutes()).padStart(2, "0")}`;
      return `<li class="${r.tier === "t1" ? "t1" : ""}"><time>${at}</time><div><b>${escapeHtml(r.title || "")}</b>`
        + `<span>${escapeHtml(String(r.body || "").split("\n")[0])}</span></div></li>`;
    }).join("") : `<li class="nc-empty">아직 보낸 알림이 없습니다 — 10월 5일 재배선 뒤부터 여기 남습니다.</li>`;
  }
  // 종 버튼: 점과 읽히는 이름
  const due = vs.filter((v) => !v.done && !v.approx && ncDaysUntil(v.date) <= 1);
  const fresh = recent.filter((r) => Date.parse(r.ts) > ncSeenAt()).length;
  const btn = el("notifyCenterBtn"), dot = el("notifyCenterDot");
  if (dot) dot.hidden = !(due.length || fresh);
  if (btn) {
    const why = [due.length ? `판정 ${due.map((v) => (ncDaysUntil(v.date) > 0 ? "내일" : ncDaysUntil(v.date) === 0 ? "오늘" : "대기")).join("·")}` : "",
      fresh ? `새 알림 ${fresh}건` : ""].filter(Boolean).join(" · ");
    btn.setAttribute("aria-label", `알림 · 판정 예정일${why ? ` — ${why}` : ""}`);
    btn.title = btn.getAttribute("aria-label");
  }
}

async function refreshNotifyCenter(force) {
  const now = Date.now();
  if (!force && now - notifyCenterLastFetchAt < NOTIFY_CENTER_POLL_MS) return;
  notifyCenterLastFetchAt = now;
  try {
    const res = await fetch("/api/notify-center", { cache: "no-cache" });
    if (!res.ok) throw new Error(`notify center ${res.status}`);
    notifyCenterData = await res.json();
  } catch (error) {
    console.error("Notify center fetch error:", error);
    notifyCenterLastFetchAt = now - NOTIFY_CENTER_POLL_MS + 60 * 1000;   // 실패하면 1분 뒤 다시
  }
  renderNotifyCenter();
}

function openNotifyCenter() {
  const dlg = el("notifyCenter");
  if (!dlg || dlg.open) return;
  dlg.showModal();
  // 열 때마다 다시 읽는다 -- 권한·구독은 다른 탭/기기에서 바뀔 수 있다(옛 알림 탭과 같은 규칙)
  refreshNotifyPage();
  refreshNotifyCenter(true).then(() => {
    try { localStorage.setItem(NC_SEEN_KEY, String(Date.now())); } catch (e) { /* 사생활 모드 */ }
    renderNotifyCenter();
  });
}

el("notifyCenterBtn")?.addEventListener("click", openNotifyCenter);
// 바깥(backdrop) 클릭으로 닫기 -- 다이얼로그 자신이 클릭 대상이면 안쪽 내용이 아닌 바깥이다
el("notifyCenter")?.addEventListener("click", (e) => { if (e.target === e.currentTarget) e.currentTarget.close(); });
if (location.hash === "#notify-center") openNotifyCenter();   // 푸시 알림을 누르고 들어온 경우

// ── 머리 칩 «실전 성적» (2026-10-05, /api/chip-score) ─────────────────────────────
// 칩이 말한 것을 서버가 기록하고 결과를 채운다(server.py chip_score_loop). 툴팁 끝에 한두 줄로 -- 백테스트 숫자 옆에 «실제로는».

async function refreshChipScore() {
  const a = activeSnapshotAsset, now = Date.now();
  if (now - (chipScoreFetchAt[a] || 0) < CHIP_SCORE_POLL_MS) return;
  chipScoreFetchAt[a] = now;
  try {
    const res = await fetch(`/api/chip-score?asset=${encodeURIComponent(a)}`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`chip score ${res.status}`);
    chipScore[a] = await res.json();
  } catch (error) {
    chipScoreFetchAt[a] = now - CHIP_SCORE_POLL_MS + 60 * 1000;
  }
}

function chipScoreLine(kind) {
  const s = chipScore[activeSnapshotAsset], pc = (v) => `${Math.round(v * 100)}%`;
  if (!s) return "";
  const head = `실전 성적(최근 ${s.days}일`, few = (n) => (n < 30 ? " · 표본 적음" : "");
  if (kind === "burst") {
    const b = s.burst;
    if (!b) return "실전 성적: 아직 0건 — 켜진 뒤 15분 되돌림을 기록합니다";
    return `${head}, ${b.n}건): 켜진 뒤 15분 되돌림 평균 ${b.rev_mean_bp >= 0 ? "+" : ""}${b.rev_mean_bp.toFixed(1)}bp · 되돌아온 비율 ${pc(b.rev_share)}${few(b.n)}`;
  }
  return "";
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
// -- 그 블록 전용 헬퍼였다. 같은 숫자는 계좌 카드 포지션 줄의 «넣으면 →» 한 마디가 그린다(2026-10-14 카드 개편).

// 2026-10-10 주문은 ETHUSDC · 화면 시세는 ETHUSDT(같은 시각 USDT 가 중앙 1.93bp 비싸다) -- 숏 진입·롱 청산 지정가가 화면 가격보다
//   싸 보여 «손해 쪽에 걸렸다»로 읽히던 착시를 막는다. 값은 서버 미리보기가 이미 준 지정가(plan.price/reference_price) -- REST 를 늘리지 않는다.
function execQuoteBp(px, ref) {
  return px > 0 && ref > 0 ? (px - ref) / ref * 1e4 : null;
}
function execQuoteHtml(plan) {
  const px = Number(plan?.price ?? plan?.reference_price), sym = String(plan?.symbol || "");
  const bp = execQuoteBp(px, Number(latestLivePriceByAsset[activeSnapshotAsset] || 0));
  if (bp == null || !/USDC$/.test(sym)) return "";   // USDT 심볼(옛 포지션 청산)이면 차이가 없다
  const book = plan.side === "BUY" ? "매수호가" : "매도호가";
  return `<div class="entry-quote">${escapeHtml(sym)} 최우선 ${book} <b>${escapeHtml(acctPx(px))}</b>`
    + ` · 화면(USDT) 대비 <b>${bp > 0 ? "+" : "−"}${Math.abs(bp).toFixed(1)}bp</b></div>`;
}
// 2026-10-10 주문 심볼의 계정 실수수료율(서버 commissionRate, 1시간 캐시) -- plan.trade_plan.execution.cost.fee.
//   🔴«이번 주문이 낼 예상 비용» 표시 전용. 크기·손절·판정은 서버가 표준 수수료(2/5bp) 상수로 계산한다(사용자 정책).
function planFee(plan) {
  return ((((plan || {}).trade_plan || {}).execution || {}).cost || {}).fee || null;
}
function feeLine(plan) {
  const f = planFee(plan), rt = (((plan.trade_plan || {}).execution || {}).cost || {}).round_trip_bp;
  if (!f) return "";
  const bp = (v) => `${+Number(v).toFixed(2)}bp`;
  return `수수료 ${f.source === "fallback" ? "추정(요율 조회 실패 · 표준)" : f.symbol || ""} 메이커 ${bp(f.maker_bp)} · 테이커 ${bp(f.taker_bp)}`
    + (rt != null ? ` · 정상 왕복 약 ${bp(rt)}` : "")
    + (f.source === "stale" ? " · 지난 조회값" : "")
    + (f.promo ? ` · ${String(f.symbol || "").slice(-4)} 프로모션 요율(기간 한정)` : "");
}
function manualEntryPlanHtml(data) {
  const plan = data.plan || {};
  const dir = plan.positionSide === "LONG" ? "롱" : "숏";
  const rl = plan.rule;   // 2026-10-09 권고 규칙 -- 무엇이 크기·손절을 정했는지 머리 바로 아래에 한 줄
  const ruleNote = rl ? entryNote(`자동 규칙 — 상한 ${won(rl.cap_notional)} USDT(순자산 × ${rl.l}, ${rl.binding}) · 이번 `
    + `${rl.first ? "첫 진입 = 상한의 1/3" : `물타기 = 남은 여유 ${won(rl.room)} USDT`} · 손절 ${rl.sl_name || ""} ${rl.sl_price ? acctPx(rl.sl_price) : "-"}`
    + (rl.first && rl.add ? ` · 체결되면 물타기 지정가 ${acctPx(rl.add.price)} × ${rl.add.qty} 자동(첫 진입가 ${plan.positionSide === "LONG" ? "−" : "+"}0.75%)` : "")
    + ` 거래소 스탑(물타기해도 안 물러남) · 익절 없음 · 교차 마진에서만`) : "";
  const parts = [`<div class="entry-head"><b>${dir} ${escapeHtml(String(plan.quantity))} ${coinUnit()}</b>
      <span>@ ${escapeHtml(acctPx(plan.price))}</span>
      <span>내 돈 ${escapeHtml(won(plan.margin_usdt || 0))} USDT${
        plan.leverage ? ` = 명목 ${escapeHtml(won(plan.notional_usdt))} ÷ ${plan.leverage}배` : ""}</span>
    </div>` + execQuoteHtml(plan)];
  if (ruleNote) parts.push(ruleNote);

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
    parts.push(br.unprotected   // 2026-10-20 이 측면에 무장된 손절이 없다 -- 넣으면 거래소 스탑도 서버 감시도 없는 포지션(10-20 XRP)
      ? entryNote("🔴SL/TP 끔 — 이 포지션엔 손절이 하나도 없습니다(거래소 스탑·서버 5분봉 감시 둘 다 없음). 교차 마진이라 크게 빠지면 다른 코인 증거금까지 깎입니다. 손절을 걸려면 SL/TP 를 켜세요.", "bad")
      : entryNote("SL/TP 끔 — 이번 진입에 TP·비상 스탑을 걸지 않고 SL 감시도 하지 않습니다. 이미 걸린 SL/TP 는 그대로 둡니다."));
  } else if (br && br.available) {
    const f = acctPx;
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
  if (feeLine(plan)) parts.push(entryNote(feeLine(plan)));
  parts.push(`<div class="entry-note${plan.dry_run ? "" : " live"}">${
    plan.dry_run ? "미리보기 전용 — 주문은 나가지 않습니다."
                 : "확인 버튼을 누르면 실제 주문이 나갑니다."} · ${(() => {
    const w = /wait_min=(\d+)/.exec(manualExecQuery());   // 2026-10-10 체결 방식이 «대기»면 그 약속을 말한다
    return w ? `최우선 호가에 한 번 걸고 따라가지 않음, ${w[1]}분 뒤 남은 수량 시장가(대기 중 취소·지금 시장가)`
      : `peg 지정가(메이커), 미체결 ${plan.fallback_after_sec}초 후 테이커 전환`;
  })()}</div>`);
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
    out.push(`기대 체결 ${ex.expected_fill_sec}초 (메이커 우선 · 미체결분 테이커) — ${ex.note}`);   // 요율은 feeLine 한 곳에서
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
// ② **신규 진입의 «자동»은 모델 추천이 아니라 고정값**이다(2026-10-13 사용자 «청산 잦을 것 같아 10배로» 20→10).
// 🔴세 경로 -- 미리보기 쿼리 · 실주문 pending · 게이지 표시 -- 가 **이 한 함수**를 본다.
//   따로 두면 «미리보기는 20배인데 주문은 모델값»이 된다.
const MANUAL_LEV_AUTO = 10;
const manualLevLocked = () => {
  const v = Number(snapshotAccountPosition()?.leverage) || 0;
  return v > 0 ? v : null;
};
function manualLevEffective() {
  const locked = manualLevLocked();
  if (locked) return locked;
  return manualLevAuto() ? MANUAL_LEV_AUTO : manualLevValue();
}
// 2026-10-09 사용자 «주문·물타기·청산 규칙 모두 제거해 수동으로, 규칙 확정되면 다시 적용»: 자동 규칙 칩을 숨기고 보내지 않는다.
//   되돌리기 = true + server.py MANUAL_RULES_ENABLED = True.
// 2026-10-24 사용자 «주문 규칙 정리»: SL/TP 끄기 체크를 뺐다 -- 손절은 모든 진입에 첫 진입가 ∓5%(서버가 건다).
const MANUAL_RULES = true;   // 2026-10-09 사용자 «1번(재생 크기식)으로 배포» -- 자동 칩 복귀
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
  // 자동이면 MANUAL_LEV_AUTO(10배)로 스냅한다. 손으로 만지는 중이면 안 건드린다.
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

// ov(2026-10-05 «스위칭»): 화면 게이지 대신 쓸 값 {pct, lev, sltp, fresh} -- 전량 청산(pct 100)과 반대 진입(그 포지션의 증거금 %·배수).
// 2026-10-09 자동 규칙(index.html #snapRule "1"): 서버가 재생 크기식으로 크기·손절을 정한다(rule=c). ETH 만 -- 다른 코인은 "직접".
// 2026-10-16 "h" = «보유 1.5×»(rule=h) -- 같은 식에 상한만 1.5(서버 HOLD_CAP_X). 옛 «권고 1.5/2» 저장값과 겹치지 않게 숫자가 아니다.
// 2026-10-20 사용자 «XRP 도 ETH 처럼 거래» -- 자동 규칙·권장 배수 코인(서버 orders.RULE_ASSETS 와 같게). 권장 배수는 그 코인 것(vol_mult_by).
const RULE_ASSETS = new Set(["eth", "xrp"]);
const volMultNow = () => { const by = latestSituation?.vol_mult_by; return by ? by[activeSnapshotAsset] || null : activeSnapshotAsset === "eth" ? latestSituation?.vol_mult : null; };
function manualRule() {
  const v = el("snapRule")?.value || "0";
  return MANUAL_RULES && RULE_ASSETS.has(activeSnapshotAsset) ? ({ 1: "c", h: "h" }[v] || "") : "";
}
const ruleCap = () => (el("snapRule")?.value === "h" ? 1.5 : 2);   // = live_manual_peg_entry HOLD_CAP_X / COMMIT_CAP_X
// 서버 live_manual_peg_entry.commit_rule_l 과 같은 식(화면 표시용 -- 실제 크기·손절은 누를 때 서버 미리보기가 정한다)
// 2026-10-10 동일위험: L = min(손절 시 순자산 손실 b ÷ 손절폭, 청산 안전, 10) -- 옛 «12.4 × 권장배수»는 손절 1회에 순자산 69~82%(원장 재생)
// 2026-10-11 사용자 최종 «×15.2 + −3% 스톱»: L = min(12.4 × 권장배수, 청산 안전(손절 거리 max(3σ24, 3%)), 10) · 손절 첫 진입가 ∓3% 고정
// 2026-10-14 사용자 «권장 B»: 상한 6 → 3배 · 손절 ∓3% → ∓5%(첫 1/3 · ∓0.75% 물타기 그대로) · 10-25 «크기는 2배로»: 상한 3 → 2배
const RULE_STOP_PCT = 0.05;       // = live_manual_peg_entry.RULE_STOP_PCT (test_rule_size_parity 가 대조)
const REC_FIRST_SHARE = 1 / 3;    // = live_manual_peg_entry.REC_FIRST_SHARE -- 첫 진입 = 상한의 1/3(10-14 사용자 «첫 1/3 → −0.75%에 2/3»)
function commitRuleL(side, sigmaBp4h, vm, cap = 2) {
  const sd = Math.max(3 * sigmaBp4h * Math.sqrt(6) / 1e4, RULE_STOP_PCT);   // 청산 안전용 거리
  const den = side === "LONG" ? 1 - 0.98 * (1 - sd) * 0.995 / (1 - 0.0066) : 1.02 * (1 + sd) * 1.005 / 1.0066 - 1;
  return { l: Math.min(4 * 0.303 * 10.26 * vm, den > 0 ? 1 / den : Infinity, cap), sd: RULE_STOP_PCT };   // 상한 2 = live_manual_peg_entry.COMMIT_CAP_X(10-25, 옛 3 · 6 · 10)
}
// «손절 시 순자산 −X%» = 지금 L × 3%(두 방향 중 큰 쪽) -- 권장 배수가 없으면 손절 거리만
function ruleRiskText() {
  const v = volMultNow();
  if (!(v?.mult && v?.sigma_bp)) return `손절 첫 진입가 −${100 * RULE_STOP_PCT}%`;
  const lev = Number(manualLevEffective()) || 0;   // 2026-10-13 서버 rec_rule_size 와 같은 상한: 실효 L ≤ 0.9 × 거래소 레버리지
  const l = Math.max(...["LONG", "SHORT"].map((s) => commitRuleL(s, Number(v.sigma_bp), Number(v.mult), ruleCap()).l).map((x) => (lev > 0 ? Math.min(x, 0.9 * lev) : x)));
  return `손절 시 순자산 −${Math.round(100 * l * RULE_STOP_PCT)}%`;
}
function renderRuleBox() {
  const box = el("snapRuleBox"), r = manualRule();
  if (box) {
    box.hidden = !MANUAL_RULES || !RULE_ASSETS.has(activeSnapshotAsset);
    box.title = activeSnapshotAsset === "eth" ? "" : `🔴자동 규칙 상수(상한 2배 · 첫 1/3 · 물타기 ∓0.75% · 손절 ∓5%)는 ETH 원장으로 검정했습니다 — ${coinUnit()} 는 권장 배수만 ${coinUnit()} 기준으로 맞췄고 규칙 자체는 검정 안 함`;
  }
  const t = ruleRiskText();
  document.querySelectorAll(".rule-risk").forEach((n) => { if (n.textContent !== t) n.textContent = t; });
  el("snapRuleBox")?.closest(".entry-line")?.classList.toggle("rule-on", !!r);
}
el("snapRule")?.addEventListener("input", () => {
  try { localStorage.setItem("entryRule", el("snapRule").value); } catch (e) { /* 저장 못 해도 동작 */ }
  renderRuleBox();
  manualEntryRefreshSize();
  if (typeof renderOfab === "function") renderOfab();
});
(() => {
  let v = "0";
  try { v = localStorage.getItem("entryRule") || "0"; } catch (e) { /* 기본 직접 */ }
  const inp = el("snapRule");
  if (inp) inp.value = v === "1" || v === "h" ? v : "0";   // 옛 «권고 1.5/2» 저장값은 «직접»으로(모르는 사이 10배 규칙으로 넘어가지 않게)
  renderRuleBox();
})();

// 2026-10-10 진입 체결 방식(#snapExec, 칩은 숫자 값만 맞춘다): 0 = 현행 따라가기 · 60/15 = 최우선 호가에 한 번 걸고 그 분 뒤 잔량 시장가(서버 run_entry_wait).
//   제출에만 붙는다(미리보기 계획은 같다). 선택은 브라우저에 기억한다. 청산·스위칭의 반대 진입은 따라가기 그대로.
function manualExecQuery() {
  const m = /^(15|60)$/.exec(el("snapExec")?.value || "");
  return m ? `&mode=wait&wait_min=${m[1]}` : "";
}
el("snapExec")?.addEventListener("input", () => {
  try { localStorage.setItem("entryExec", el("snapExec").value); } catch (e) { /* 저장 못 해도 동작 */ }
});
(() => {
  let v = "0";
  try { v = localStorage.getItem("entryExec") || "0"; } catch (e) { /* 기본 따라가기 */ }
  const inp = el("snapExec");
  if (inp) inp.value = /^(15|60)$/.test(v) ? v : "0";
})();
// 2026-10-10 청산 체결 방식(#snapExitExec): 0 = 지정가(현행 peg, 기본) · 1 = 즉시(시장가). 스위칭은 칩과 무관하게 늘 즉시(서버가 강제).
//   미리보기·제출 둘 다 &exec=market 을 실어 미리보기 비용(테이커)과 실제 주문이 같다.
const manualExitMarket = () => el("snapExitExec")?.value === "1";
const exitExecQuery = (ov) => (ov?.switch || manualExitMarket() ? "&exec=market" : "");
el("snapExitExec")?.addEventListener("input", () => {
  try { localStorage.setItem("exitExec", el("snapExitExec").value); } catch (e) { /* 저장 못 해도 동작 */ }
});
(() => {
  let v = "0";
  try { v = localStorage.getItem("exitExec") || "0"; } catch (e) { /* 기본 지정가 */ }
  const inp = el("snapExitExec");
  if (inp) inp.value = v === "1" ? "1" : "0";
})();

async function manualEntryFetch(side, kind = "entry", ov = null) {
  if (!orderCoinOk()) {
    return { ok: false, error: "order_coin_mismatch",
             detail: `${coinUnit()} 탭에서는 주문하지 않습니다` };
  }
  const q = `&pct=${ov?.pct ?? (kind === "exit" ? manualExitPct() : manualEntryPct())}`
    + (kind === "exit" ? "" : (ov?.lev ? `&lev=${ov.lev}` : manualLevQuery()))
    + (kind === "exit" ? exitExecQuery(ov) : ((ov?.rule ?? manualRule()) ? `&rule=${ov?.rule ?? manualRule()}` : ""))
    + (ov?.fresh ? "&fresh=1" : "") + (kind !== "exit" && ov?.tp ? `&tp=${ov.tp}&sl=${ov.sl}` : "");
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
    + `<span>${acctPx(plan.price ?? plan.reference_price)}`
    + `${plan.type === "MARKET" ? " 근처" : ""} · ${Number(plan.notional_usdt).toLocaleString()} USDT</span></div>` + execQuoteHtml(plan)];
  // 🔴미리보기 카드의 주인공은 «지금 닫으면 순손익 얼마»다(2026-09-14, 사용자 요청).
  // 여기서는 plan.type 을 알므로 고변동 시장가 전환이면 테이커 요율로 바꿔 계산한다.
  // 2026-10-10 요율은 청산할 포지션 심볼의 계정 실요율(없으면 표준 2/5bp) -- peg 는 실측 3.0 에서 표준 메이커 2.0 을 실요율로 바꾼 값.
  const upnl = Number(plan.unrealized_pnl), pf = planFee(plan);
  const feeBp = +(plan.type === "MARKET" ? (pf ? pf.taker_bp : EXIT_FEE_BP_TAKER)
    : (pf ? EXIT_FEE_BP_PEG - 2.0 + pf.maker_bp : EXIT_FEE_BP_PEG)).toFixed(2);
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
    escapeHtml(feeLine(plan)),
  ]));
  parts.push(`<div class="entry-cap">${plan.dry_run
    ? "미리보기 전용 — 주문은 나가지 않습니다."
    : "확인 버튼을 누르면 실제 청산 주문이 나갑니다."} · ${how}${vol}</div>`);
  return parts.join("");
}

const MANUAL_BTN_IDS = ["snapEntryLong", "snapEntryShort", "snapExitLong", "snapExitShort", "snapSwitch", "ofabLong", "ofabShort"];
const manualButtonsDisabled = (v) =>
  MANUAL_BTN_IDS.forEach((id) => { const b = el(id); if (b) b.disabled = v; });

async function manualEntryPreview(side, kind = "entry", ov = null) {
  const box = el("snapEntryResult");
  if (!box || manualOrderBusy) return;   // 진행 중인 주문 표시를 덮지 않는다
  manualPreviewInFlight = true;
  manualButtonsDisabled(true);
  box.hidden = false;
  box.innerHTML = entryNote("확인 중…");
  manualEntryClearConfirm();   // 다른 방향을 눌렀는데 옛 확인 버튼이 남아 있으면 안 된다
  try {
    const asset = activeSnapshotAsset;
    const data = await manualEntryFetch(side, kind, ov);
    // 🔴응답 전에 코인 탭을 바꾸면 옛 코인 계획이 새 코인 이름으로 무장됐다(길게 누르기면 바로 발주)
    if (asset !== activeSnapshotAsset) throw new Error("미리보기 중 코인을 바꿨습니다 — 다시 누르세요(주문 안 나감)");
    box.innerHTML = data.ok
      ? (kind === "exit" ? manualExitPlanHtml(data.plan || {}) : manualEntryPlanHtml(data))
      : entryNote(`실패: ${data.detail || data.error || "알 수 없음"}`, "bad");
    // 게이트가 꺼져 있으면 확인 버튼을 아예 띄우지 않는다 -- 눌러도 403 이라 헛걸음이다.
    if (data.ok && data.exec_enabled) manualEntryArmConfirm(side, data.plan || {}, kind, ov);
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
  renderSnapshotAccount();   // 2026-10-14 카드 포지션 줄의 «넣으면 →» 한 마디만 바뀐다(타일·제자리 패처는 카드 개편으로 없어짐)
}

let execQuote = null;   // 2026-10-10 떠 있는 버튼용 주문 심볼 호가(크기 미리보기 LONG = 매수호가 · SHORT = 매도호가)
async function manualEntryRefreshSize() {
  const line = el("snapEntrySize");
  if (!line) return;
  renderRuleBox();   // 코인을 바꾸면 규칙 칩도(ETH 만)
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
        : (data.detail || data.error || "알 수 없음");   // 2026-10-09 자동 규칙의 막힘(격리 마진·변동성 없음)은 서버 문장 그대로
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
    const refNow = Number(latestLivePriceByAsset[asset] || 0);   // 미리보기가 온 그 순간의 USDT -- 차이(bp)는 그때 잰다(값이 60초 묵어도 차이는 거의 안 변한다)
    execQuote = /USDC$/.test(String(plan.symbol || "")) && plan.price > 0 && planShort.price > 0 && refNow > 0
      ? { asset, sym: plan.symbol, bid: Number(plan.price), ask: Number(planShort.price), ref: refNow, at: Date.now() } : null;
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
  cancelled: "대기 취소 — 남은 수량은 사지 않았습니다(시장가 없음)",
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

function manualEntryArmConfirm(side, plan, kind = "entry", ov = null) {
  if (plan.blocked) { manualFireOnPreview = false; return; }
  const pct = ov?.pct ?? Math.max(1, Math.round(100 * (plan.fraction ?? 1)));   // 규칙 물타기 몫이 0.5% 미만이면 0 → 서버 400(규칙 모드는 pct 를 안 쓴다)
  manualEntryPending = { side, quantity: plan.quantity, kind, pct, asset: activeSnapshotAsset,
                         lev: ov?.lev ?? manualLevEffective(),
                         rule: kind === "exit" ? "" : (ov?.rule ?? manualRule()),
                         fresh: !!ov?.fresh, sw: ov?.switch || null, tp: ov?.tp || "", sl: ov?.sl || "",
                         exec: kind === "exit" ? exitExecQuery(ov) : (ov?.tp ? "" : manualExecQuery()) };   // 2026-10-28 선 주문 = 따라가기(run_entry)만
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
    + (kind === "exit" ? (pct < 100 ? `청산 (${pct}%)` : "전량 청산") : "주문") + short
    + (ov?.switch ? ` → ${ov.switch.to === "LONG" ? "롱" : "숏"} 진입(스위칭 · 손절 걸림)` : "")
    // 2026-10-10 즉시 체결(스위칭은 늘) -- 테이커 요율을 확인 버튼에 적는다(실요율, 못 읽었으면 표준 5bp)
    + (kind === "exit" && plan.type === "MARKET"
      ? ` · 즉시 체결(시장가, 테이커 ${+Number(planFee(plan)?.taker_bp ?? EXIT_FEE_BP_TAKER).toFixed(2)}bp)` : "");
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
  const sw = state?.switch, ko = (s) => (s === "LONG" ? "롱" : "숏");   // 2026-10-10 서버 스위칭(진입으로 넘어가도 남는다)
  if (sw) {
    rows.unshift(sw.phase === "aborted" ? `🔴스위칭 중단 — ${sw.error || "?"}`
      : `스위칭 ${ko(sw.from)} → ${ko(sw.to)} · ${{ exit: "전량 청산 중 — 다 닫히면 서버가 반대 진입", entry: "청산 완료 → 반대 진입 중", done: "완료" }[sw.phase] || sw.phase}`);
  }
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
    rows.push(br.unprotected ? "🔴SL/TP 끔 — 이 포지션엔 손절이 없습니다(거래소 스탑·서버 감시 없음)" : "SL/TP 끔 — 걸지 않았습니다(기존 SL/TP 는 그대로)");
  } else if (state?.trigger === "bracket_sl") {
    rows.push("SL 이탈 — 메이커 추격 청산");
  } else if (filledQty > 0 && state?.kind !== "exit" && !br && /^filled|failed|error|rejected|cancelled/.test(state?.phase || "")) {
    rows.push("TP/SL 거는 중…");
  } else if (br) {
    rows.push(`🔴TP/SL 을 못 걸었습니다 (${br?.reason || "시도 기록 없음"})`);
  }
  // 2026-10-14 자동 물타기 지정가(첫 진입가 ∓0.75%) -- 포지션이 닫히면 서버 감시 루프가 지운다
  const ad = state?.add_order;
  if (ad && state?.kind !== "exit") {
    rows.push(ad.pending ? "물타기 지정가 거는 중…" : ad.placed ? `물타기 지정가 ${acctPx(ad.price)} × ${ad.qty} 걸림(포지션 닫히면 자동 취소)`
      : `🔴물타기 지정가 못 걸었습니다 (${ad.error || ad.reason || "?"}) — 필요하면 손으로`);
  }
  // 2026-09-15 진입도 리페그한다. 쫓아간 횟수는 «의도한 가격보다 높게 들어갔을 수 있다»는
  // 뜻이라 숨기지 않는다 -- 청산과 달리 진입은 안 사도 되는 선택지가 있었기 때문이다.
  if (Number(state?.repegs || 0) > 0 && state?.mode !== "wait") {   // 기다리기의 repegs = 처음 -5022 로 다시 건 횟수(따라간 게 아니다)
    rows.push(`리페그 ${Number(state.repegs)}회 — 호가를 따라갔습니다` +
      (state.limit_price ? ` (현재 지정가 ${state.limit_price})` : ""));
  }
  const lv = state?.leverage;
  if (lv && lv.error) rows.push(`⚠레버리지 ${lv.to}배 설정 실패 — 거래소 천장이 그대로입니다 (${lv.error})`);
  else if (lv && lv.changed) rows.push(`레버리지 ${lv.from}배 → ${lv.to}배 적용`);
  if (state?.error) rows.push(`사유: ${state.error}`);
  return rows.join("\n");
}

// 2026-10-15 사용자 «손절 해제 버튼»(10-14 사고: 바이낸스에서 스탑만 지웠는데 서버 5분봉 종가 감시가 청산했다).
//   서버가 무장한 손절(/api/binance-account 의 bracket_armed)을 **플로팅 버튼**에 보이고(사용자 «계좌카드 말고 플로팅 버튼 안에»),
//   두 번 눌러(5초 무장) 거래소 스탑 + 서버 감시를 함께 끄고, 꺼진 뒤 다시 두 번 누르면 다시 켠다(/rearm). 데스크탑 = 알약 안 칩(#ofabSl, 시안 B) · 휴대폰 = 패널 맨 위 칸(#ofabSlPanel, 시안 C).
//   둘 다 같은 data-sl-act 핸들러 -- 어느 쪽을 보일지는 CSS(720px)가 고른다. 자동 물타기 지정가는 남고, 다시 진입·규칙 물타기하면 손절이 다시 걸린다.
let slOffArmedKey = null, slOffArmT = 0, slOffBusyKey = null;
const SL_OFF_TIP = "거래소의 손절 스탑과 서버의 5분봉 종가 손절을 함께 끕니다. 자동 물타기 지정가는 남고, 다시 진입·물타기하면 손절이 다시 걸립니다";
const SL_ICO = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/></svg>';
const SL_ICO_OFF = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m2 2 20 20"/><path d="M5 5a1 1 0 0 0-1 1v7c0 5 3.5 7.5 7.67 8.94a1 1 0 0 0 .67.01c2.35-.82 4.48-1.97 5.9-3.71"/><path d="M9.309 3.652A12.252 12.252 0 0 0 11.24 2.28a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1v7a9.784 9.784 0 0 1-.08 1.264"/></svg>';
function renderOfabSl(p) {
  const chip = el("ofabSl"), box = el("ofabSlPanel");
  if (!chip || !box) return;
  const k = p && Number(p.qty) ? `${p.symbol}:${String(p.side).toUpperCase()}` : null;
  const ba = k ? (latestBinanceAccount?.bracket_armed || {})[k] : null;
  if (k && !(ba && ba.sl_price)) {   // 2026-10-20 무장 없는 열린 포지션 = «손절 없음»(옛 판은 칩을 숨겨 조용히 무방비였다)
    chip.hidden = box.hidden = false;
    chip.className = "ofab-sl off none"; chip.disabled = true; delete chip.dataset.slAct; chip.dataset.slKey = k;
    chip.title = "손절 없음 — 이 포지션엔 거래소 스탑도 서버 5분봉 감시도 없습니다(SL/TP 를 켜고 진입하거나 바이낸스에서 직접 거세요)";
    chip.setAttribute("aria-label", chip.title);
    chip.innerHTML = `${SL_ICO_OFF}<span>손절 없음</span>`;
    box.className = "ofab-slp off";
    box.innerHTML = `<div class="ofab-slp-head">${SL_ICO_OFF}<span>손절</span><b>없음</b><em>무장 안 됨</em></div>`
      + `<div class="ofab-slp-layers"><span><i></i>거래소 스탑 없음</span><span><i></i>서버 5분봉 감시 없음</span></div>`;
    return;
  }
  chip.disabled = false;
  chip.hidden = box.hidden = !ba || !ba.sl_price;
  if (chip.hidden) return;
  // 2026-10-15 사용자 «다시 누르면 다시 손절을 켜도록» -- 꺼진 상태도 같은 두 번 누름으로 켠다(act = 누르면 할 일).
  const off = !!ba.sl_off, act = off ? "on" : "off", armed = slOffArmedKey === `${act}:${k}`, busy = slOffBusyKey === k, px = acctPx(ba.sl_price);
  const mk = Number(p.mark_price) || 0, dist = mk > 0 && !off ? Math.abs(mk - ba.sl_price) / mk * 100 : null;
  const hot = busy || armed;
  const word = off ? "켜기" : "해제";
  chip.className = `ofab-sl${hot ? " arm" : off ? " off" : ""}`;
  chip.disabled = busy;
  chip.dataset.slAct = act; chip.dataset.slKey = k;
  chip.title = off ? `손절 없음 — 두 번 누르면 손절 ${px} 를 다시 켭니다(거래소 스탑 + 서버 5분봉 감시)`
    : `손절 ${px}${ba.sl_name ? ` · ${ba.sl_name}` : ""} — 두 번 누르면 해제. ${SL_OFF_TIP}`;
  chip.setAttribute("aria-label", armed ? `한 번 더 누르면 손절 ${word}` : off ? `손절 없음 — 다시 켜려면 두 번 누르기` : `손절 ${px} — 해제하려면 두 번 누르기`);
  chip.innerHTML = busy ? `${off ? SL_ICO_OFF : SL_ICO}<span>${off ? "켜는 중…" : "해제 중…"}</span>`
    : armed ? `${off ? SL_ICO_OFF : SL_ICO}<span>한 번 더 = ${word}</span>`
    : off ? `${SL_ICO_OFF}<span>손절 없음</span>` : `${SL_ICO}<span>${px}</span>`;
  box.className = `ofab-slp${off ? " off" : ""}`;
  const btn = `<button type="button" class="ofab-slp-btn${hot ? " arm" : ""}" data-sl-act="${act}" data-sl-key="${escapeHtml(k)}"`
    + ` title="${escapeHtml(off ? "거래소 스탑과 서버 5분봉 감시를 다시 켭니다" : SL_OFF_TIP)}"${busy ? " disabled" : ""}>`
    + `${busy ? (off ? "켜는 중…" : "해제 중…") : armed ? `한 번 더 누르면 둘 다 ${off ? "켜기" : "해제"}` : off ? `손절 ${px} 다시 켜기` : "손절 해제"}</button>`;
  box.innerHTML = off
    ? `<div class="ofab-slp-head">${SL_ICO_OFF}<span>손절</span><b>없음</b><em>해제함</em></div>`
      + `<div class="ofab-slp-layers"><span><i></i>거래소 스탑 꺼짐</span><span><i></i>서버 5분봉 감시 꺼짐</span></div>` + btn
    : `<div class="ofab-slp-head">${SL_ICO}<span>손절</span><b>${px}</b>${ba.sl_name ? `<span>${escapeHtml(ba.sl_name)}</span>` : ""}${dist != null ? `<em>지금가 ${ba.sl_price > mk ? "+" : "−"}${dist.toFixed(1)}%</em>` : ""}</div>`
      + `<div class="ofab-slp-layers"><span><i></i>거래소 스탑</span><span><i></i>서버 5분봉 감시</span></div>` + btn;
}
document.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-sl-act]");
  if (!b || b.disabled || !b.dataset.slKey) return;
  const k = b.dataset.slKey, act = b.dataset.slAct, [symbol, side] = k.split(":"), redraw = () => renderOfabSl(snapshotAccountPosition());
  if (slOffArmedKey !== `${act}:${k}`) {
    slOffArmedKey = `${act}:${k}`; redraw();
    clearTimeout(slOffArmT);
    slOffArmT = setTimeout(() => { slOffArmedKey = null; redraw(); }, 5000);
    return;
  }
  clearTimeout(slOffArmT); slOffArmedKey = null; slOffBusyKey = k; redraw();
  const what = act === "on" ? "손절 다시 켜기" : "손절 해제";
  try {
    const r = await (await fetch(`/api/manual-bracket/${act === "on" ? "rearm" : "disarm"}?side=${encodeURIComponent(side)}&symbol=${encodeURIComponent(symbol)}&confirm=1`,
      { method: "POST", cache: "no-cache" })).json();
    ofabSay(r.ok ? (act === "on" ? "손절 다시 켬 — 거래소 스탑·서버 5분봉 감시 둘 다 켜짐" : "손절 해제 — 거래소 스탑·서버 5분봉 감시 둘 다 꺼짐")
      : escapeHtml(`${what} 못 함: ${r.detail || r.error}`), r.ok ? "" : "bad");
  } catch (err) {
    ofabSay(escapeHtml(`${what} 못 함: ${err && err.message ? err.message : err}`), "bad");
  } finally {
    slOffBusyKey = null; redraw();   // 계좌를 다시 못 읽어도 «…중»이 남지 않게
    refreshBinanceAccount(true);
  }
});

// 2026-10-11 «따라가기» 진입이 거래소에 걸려 있는 동안(peg · 120초 뒤 잔량 시장가) «대기 취소» -- 기다리기 상자(#snapWait)와 같은 줄 문법.
//   취소는 남은 수량을 안 사는 쪽이라 위험을 줄이지만, 노리던 진입을 버리는 것이라 «지금 시장가»처럼 두 번(5초 무장).
//   청산·스위칭은 안 붙인다(서버도 409) -- 청산 취소는 포지션을 그대로 두는 것이다.
let entryCancelArmT = 0, entryCancelArmed = false, entryCancelLive = false;
// 2026-10-22 «전체 청산»(계좌 카드) -- 카드가 폴링마다 다시 그려지므로 상태는 여기 둔다
let closeAllArmed = false, closeAllBusy = false, closeAllMsg = "", closeAllT = 0, closeAllMsgT = 0;
const closeAllLabel = (n) => (closeAllBusy ? "전체 청산 보내는 중…"
  : closeAllArmed ? `한 번 더 = ${n}개 전부 ${manualExitMarket() ? "즉시(시장가)" : "지정가"} 청산` : `전체 청산 · ${n}개`);
document.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-close-all]");
  if (!b || b.disabled) return;
  const n = Number(b.dataset.closeAll) || 2;
  if (!closeAllArmed) {
    closeAllArmed = true; b.classList.add("arm"); b.textContent = closeAllLabel(n);
    clearTimeout(closeAllT);
    closeAllT = setTimeout(() => { closeAllArmed = false; const x = document.querySelector("[data-close-all]");
      if (x && !x.disabled) { x.classList.remove("arm"); x.textContent = closeAllLabel(n); } }, 5000);
    return;
  }
  clearTimeout(closeAllT); closeAllArmed = false; closeAllBusy = true;
  b.disabled = true; b.classList.remove("arm"); b.textContent = closeAllLabel(n);
  try {
    const r = await (await fetch(`/api/manual-exit/submit-all?confirm=1${exitExecQuery()}`, { method: "POST", cache: "no-cache" })).json();
    closeAllMsg = (r.results || []).map((x) => `${x.symbol ? acctCoin(x.symbol) : "?"} ${x.side === "LONG" ? "롱" : "숏"} ${x.ok ? "청산 접수" : (x.detail || x.error)}`).join(" · ")
      || `전체 청산 못 함: ${r.detail || r.error}`;
  } catch (err) {
    closeAllMsg = `전체 청산 요청 실패: ${err && err.message ? err.message : err} — 거래소에서 확인하세요`;
  }
  closeAllBusy = false;
  clearTimeout(closeAllMsgT); closeAllMsgT = setTimeout(() => { closeAllMsg = ""; refreshBinanceAccount(true); }, 60000);
  manualEntryPollStatus();
  refreshBinanceAccount(true);
});
const entryCancelableP = (st) => /^(working|submitting)$/.test(st?.phase || "") && st?.kind !== "exit" && !st?.switch && st?.plan?.type !== "MARKET";
function entryResultHtml(st, tone) {
  entryCancelLive = entryCancelableP(st);
  el("ofabWait")?.toggleAttribute("hidden", !(entryCancelLive || manualWaitLive));
  if (!entryCancelLive) { entryCancelArmed = false; return entryNote(manualEntryStateText(st), tone); }
  const label = st.action ? "취소 요청됨 — 3초 안에 처리" : entryCancelArmed ? "한 번 더 = 대기 취소(남은 수량 안 삼)" : "대기 취소";
  return entryNote(manualEntryStateText(st), tone)
    + `<div class="manual-entry-row wait-acts"><button type="button" class="notify-btn${entryCancelArmed && !st.action ? " arm" : ""}"`
    + ` data-entry-cancel${st.action ? " disabled" : ""}>${label}</button></div>`;
}
el("snapEntryResult")?.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-entry-cancel]");
  if (!b || b.disabled) return;
  if (!entryCancelArmed) {
    entryCancelArmed = true; b.classList.add("arm"); b.textContent = "한 번 더 = 대기 취소(남은 수량 안 삼)";
    clearTimeout(entryCancelArmT);
    entryCancelArmT = setTimeout(() => { entryCancelArmed = false; const x = el("snapEntryResult")?.querySelector("[data-entry-cancel]");
      if (x && !x.disabled) { x.classList.remove("arm"); x.textContent = "대기 취소"; } }, 5000);
    return;
  }
  clearTimeout(entryCancelArmT); entryCancelArmed = false;
  b.disabled = true; b.classList.remove("arm"); b.textContent = "취소 요청 중…";
  try {
    const r = await (await fetch("/api/manual-entry/cancel?confirm=1", { method: "POST", cache: "no-cache" })).json();
    b.textContent = r.ok ? "취소 요청됨 — 3초 안에 처리" : `취소 못 함: ${r.detail || r.error}`;   // 끝났으면(409) 다음 조회가 결과를 그린다
  } catch (err) { b.disabled = false; b.textContent = "대기 취소"; }
});

async function manualEntryPollStatus() {
  const box = el("snapEntryResult");
  if (!box) return;
  try {
    const res = await fetch("/api/manual-entry/status", { cache: "no-cache" });
    const data = await res.json();
    // 2026-10-20 청산만 코인별: 주 칸을 다른 코인이 쓰는 동안 이 코인 청산은 side_exits[코인] 칸으로 나간다 -- 그 칸을 본다
    const sx = (data.side_exits || {})[activeSnapshotAsset];
    if (sx && sx.phase && sx.phase !== "idle" && (data.state?.asset || "eth") !== activeSnapshotAsset) data.state = sx;
    box.innerHTML = entryResultHtml(data.state,
      /^(error|taker_failed|rejected)$/.test(data.state?.phase || "") ? "bad" : "live");
    const phase = data.state?.phase;
    // 진입이 끝나도 TP/SL 은 그 **뒤에** 걸린다 -- 체결이 있는데 결과가 아직 없으면 한 번 더 본다.
    const bracketPending = data.state?.kind !== "exit" && Number(data.state?.filled || 0) > 0
      && (!data.state?.bracket || data.state?.add_order?.pending);
    const switching = /^(exit|entry)$/.test(data.state?.switch?.phase || "");   // 2026-10-10 스위칭은 서버가 잇는다 -- 끝날 때까지 본다
    if (phase === "submitting" || phase === "working" || bracketPending || switching) {
      setTimeout(manualEntryPollStatus, STATUS_POLL_MS);
    } else {
      manualOrderBusy = false;
      manualButtonsDisabled(false);
      manualEntryRefreshSize();   // 체결되면 포지션·상한 표시를 갱신한다
      refreshBinanceAccount(true);   // 떠 있는 버튼·계좌 카드도 바로(서버 캐시 10초를 건너뛴다)
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
      + (pending.kind === "exit" ? "" : (pending.lev ? `&lev=${pending.lev}` : ""))
      + (pending.kind !== "exit" && pending.rule ? `&rule=${pending.rule}` : "")
      + (pending.fresh ? "&fresh=1" : "") + (pending.exec || "")
      + (pending.kind !== "exit" && pending.tp ? `&tp=${pending.tp}&sl=${pending.sl}` : "")   // 2026-10-28 차트 선 주문
      // 2026-10-10 스위칭 = 서버 한 작업(청산 다 닫히면 서버가 반대 진입) -- 화면이 꺼져도 이어진다
      + (pending.kind === "exit" && pending.sw ? `&switch_to=${pending.sw.to}&sw_pct=${pending.sw.pct}`
        + (pending.sw.lev ? `&sw_lev=${pending.sw.lev}` : "")
        + (pending.sw.rule ? `&sw_rule=${pending.sw.rule}` : "") : "");
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
    if (data.wait) {   // 2026-10-10 기다리기 -- 주문 칸을 잡지 않는다(대기 중에도 청산·다른 버튼). 상태는 아래 #snapWait 가 말한다.
      manualOrderBusy = false;
      manualButtonsDisabled(false);
      box.innerHTML = entryNote("기다리기 — 최우선 호가에 걸어 두었습니다. 아래에서 취소·지금 시장가", "live");
      manualWaitPoll();
      return;
    }
    box.innerHTML = entryResultHtml(data.state, "live");
    if (pending.kind === "exit" && pending.sw) ofabWatchResult();   // 말풍선이 서버 스위칭 상태(결과 칸)를 따라 말한다
    setTimeout(manualEntryPollStatus, STATUS_POLL_MS);
  } catch (err) {
    box.innerHTML = entryNote(`주문 실패: ${err && err.message ? err.message : err}`, "bad");
    manualOrderBusy = false;
    manualButtonsDisabled(false);
  }
}

// ── 2026-10-10 기다리는 진입 ─────────────────────────────────────────────────────────
// 서버 별도 칸(/api/manual-entry/status 의 wait). 진행 중이면 3초마다 다시 본다 -- 페이지를 새로 열어도 첫 조회가 다시 잇는다.
let manualWaitTimer = 0, manualWaitLive = false, manualWaitArmT = 0;
const waitLiveP = (w) => /^(working|submitting)$/.test(w?.phase || "");
function renderManualWait(w) {
  const box = el("snapWait"), txt = el("snapWaitText"), acts = el("snapWaitActs");
  if (!box || !txt) return;
  const live = waitLiveP(w), ph = w?.phase || "idle";
  const recent = Date.now() - Date.parse(w?.done_at || "") < 10 * 60e3;   // 끝난 결과는 10분만 남긴다
  box.hidden = ph === "idle" || (!live && !recent);
  el("ofabWait")?.toggleAttribute("hidden", !live && !entryCancelLive);   // 따라가기 진입 대기 중에도(대기 취소 버튼으로 데려간다)
  if (acts) acts.hidden = !live;
  if (box.hidden) return;
  const ko = w.side === "LONG" ? "롱" : "숏", unit = (ASSET_CONFIG[w.asset] || {}).label || coinUnit();
  let t, tone = "live";
  if (ph === "restart_cancelled") {
    const n = (w.cancelled || []).length, errs = (w.errors || []).join(" · ");
    t = `서버 재시작 — 기다리던 진입 주문 ${n}건을 거래소에서 취소했습니다(이미 든 몫·손절은 그대로)` + (errs ? ` · 🔴정리 실패: ${errs}` : "");
    tone = errs ? "bad" : "live";
  } else if (live) {
    const left = Math.max(0, Math.ceil((Date.parse(w.wait_until || "") - Date.now()) / 60e3));
    t = `기다리기 ${ko} ${Number(w.filled || 0)} / ${Number(w.quantity ?? w.plan?.quantity ?? 0)} ${unit}`
      + (w.limit_price ? ` · 지정가 ${w.limit_price} (따라가지 않음)` : "")
      + (Number.isFinite(left) && w.wait_until ? ` · ${left}분 뒤 남은 수량 시장가` : "")
      + (w.action ? ` · ${w.action === "cancel" ? "취소" : "지금 시장가"} 요청됨 — 3초 안에 처리` : "");
  } else {
    t = `기다리기 끝 — ${manualEntryStateText(w).replace(/\n/g, " · ")}`
      + (w.taker_reason ? ` (${w.taker_reason})` : "") + (w.cancel_reason ? ` (${w.cancel_reason})` : "");
    tone = /^(error|taker_failed|rejected)$/.test(ph) ? "bad" : "";   // 끝난 결과는 흐리게 -- 주황(주의)은 거래소에 걸려 있는 동안만
  }
  txt.className = `entry-note${tone ? " " + tone : ""}`;
  txt.textContent = t;
  ["snapWaitCancel", "snapWaitMarket"].forEach((id) => { const b = el(id); if (b) b.disabled = !!w.action; });
}
async function manualWaitPoll() {
  clearTimeout(manualWaitTimer);
  let w = null;
  try {
    w = (await (await fetch("/api/manual-entry/status", { cache: "no-cache" })).json()).wait || null;
  } catch (e) {
    manualWaitTimer = setTimeout(manualWaitPoll, 10000);   // 조회 실패 -- 표시는 그대로 두고 늦게 다시
    return;
  }
  renderManualWait(w);
  const live = waitLiveP(w);
  if (manualWaitLive && !live) { refreshBinanceAccount(true); manualEntryRefreshSize(); }   // 방금 끝났다 -- 포지션 다시
  manualWaitLive = live;
  if (live) manualWaitTimer = setTimeout(manualWaitPoll, STATUS_POLL_MS);
}
async function manualWaitAction(action) {
  ["snapWaitCancel", "snapWaitMarket"].forEach((id) => { const b = el(id); if (b) b.disabled = true; });
  try {
    const r = await (await fetch(`/api/manual-entry/wait-action?action=${action}&confirm=1`, { method: "POST", cache: "no-cache" })).json();
    if (!r.ok) { const t = el("snapWaitText"); if (t) { t.className = "entry-note bad"; t.textContent = `요청 실패: ${r.detail || r.error}`; } }
  } catch (e) { /* 아래 조회가 실제 상태를 다시 그린다 */ }
  manualWaitPoll();
}
el("snapWaitCancel")?.addEventListener("click", () => manualWaitAction("cancel"));   // 위험을 줄이는 쪽 -- 한 번에
// «지금 시장가»는 테이커 주문이라 두 번: 첫 클릭 = 5초 무장(글자가 바뀐다) · 그 안에 한 번 더 = 전송
el("snapWaitMarket")?.addEventListener("click", () => {
  const b = el("snapWaitMarket");
  if (!b.classList.contains("arm")) {
    b.classList.add("arm"); b.textContent = "한 번 더 = 남은 수량 시장가";
    clearTimeout(manualWaitArmT);
    manualWaitArmT = setTimeout(() => { b.classList.remove("arm"); b.textContent = "지금 시장가"; }, 5000);
    return;
  }
  clearTimeout(manualWaitArmT); b.classList.remove("arm"); b.textContent = "지금 시장가";
  manualWaitAction("market");
});
el("ofabWait")?.addEventListener("click", () => { if (!ofab.open) ofabSetOpen(true); el(entryCancelLive ? "snapEntryResult" : "snapWait")?.scrollIntoView({ block: "nearest" }); });
setTimeout(manualWaitPoll, 1500);   // 새로고침·재접속 뒤에도 걸린 대기 주문을 다시 보인다

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
function manualHoldStart(btn, side, kind, ov = null) {
  if (manualOrderBusy || btn.disabled) return;
  manualHoldCancel(btn);
  manualHoldFire = true;
  // 청산은 **계좌부터 새로 읽는다**(manualExitPreview) -- 닫으려는 수량이 낡으면 안 된다.
  if (kind === "exit") manualExitPreview(side, ov); else manualEntryPreview(side, "entry");
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
// 🔴2026-10-10 사용자 결정 «스위칭은 즉시 체결 기본»: 아래 peg 설명은 옛 판 -- 이제 청산·반대 진입 둘 다 시장가(서버 exec=market 강제).
// 2026-10-05 «스위칭»(사용자 지시, 떠 있는 패널에서만): 게이지와 상관없이 지금 쪽 **100% peg 메이커 청산** → 전량 체결되면
//   **반대쪽 peg 메이커 진입**. 반대 진입 크기 = 닫은 포지션의 증거금 %(명목 ÷ 배수 ÷ 순자산)·같은 배수 → 거의 같은 수량
//   (체결가·손익만큼 차이). 증거금 50% 상한은 서버가 그대로 걸고, 반대 진입은 계좌를 새로 읽는다(fresh=1 -- 닫은 포지션이
//   30초 캐시에 남아 노출로 잡히면 상한이 반대 진입을 깎는다). 청산이 전량 안 채워지면(거부·오류) 반대 진입을 안 낸다.
//   확인 규칙은 청산과 같다(마우스 0.4초 꾹 = 바로 · 터치/키보드 = 미리보기 → 확인). 120초 뒤 테이커 전환도 청산 그대로.
// 🔴2026-10-10 청산 → 반대 진입 연결은 **서버**(server.py exit_then_switch)가 한다 -- 브라우저 폴링이 잇던 판은 화면이 꺼지면 반대 진입이 안 나갔다.
function switchPlan(side) {
  const p = lastExitPositions.get(side);
  const qty = Number(p?.qty) || 0, mark = Number(p?.mark_price) || 0, lev = Number(p?.leverage) || manualLevEffective() || 0;
  const eq = acctMarginUsed(latestBinanceAccount?.balance).equity;
  if (!(qty > 0 && mark > 0 && lev > 0 && eq > 0)) return null;
  const pct = Math.min(100, Math.max(0.01, Math.round(qty * mark / lev / eq * 10000) / 100));
  // 2026-10-10 사용자 결정: 반대 진입엔 SL/TP 체크와 무관하게 손절이 걸린다(서버가 강제) -- sltp 를 보내지 않는다
  return { pct: 100, switch: { to: side === "LONG" ? "SHORT" : "LONG", pct, lev, rule: manualRule() } };
}
{
  const btn = el("snapSwitch");
  const held = () => (lastExitPositions.has("LONG") ? "LONG" : lastExitPositions.has("SHORT") ? "SHORT" : "");
  const plan = (s) => {
    const ov = s && switchPlan(s);
    if (!ov) ofabSay(escapeHtml("스위칭 안 함 — 포지션·증거금 정보를 아직 못 읽었습니다. 잠시 뒤 다시 누르세요."), "bad");
    return ov;
  };
  let touched = false;
  btn?.addEventListener("pointerdown", (e) => {
    touched = e.pointerType === "touch";
    if (touched) return;
    e.preventDefault();
    const s = held(), ov = plan(s);
    if (ov) manualHoldStart(btn, s, "exit", ov);
  });
  btn?.addEventListener("click", (e) => {
    if (!(touched || e.detail === 0) || manualOrderBusy || btn.disabled) return;
    touched = false;
    const s = held(), ov = plan(s);
    if (ov) manualExitPreview(s, ov);
  });
  ["pointerup", "pointerleave", "pointercancel"].forEach((ev) => btn?.addEventListener(ev, () => manualHoldCancel(btn)));
}
// 2026-09-14 사용자 요청: **청산은 강제 조회부터**. 화면 숫자가 30초(조회가 끊겼으면 그
// 이상) 묵어 있을 수 있어서, 미리보기를 그리기 전에 계좌를 다시 받아 카드·아래 줄을 맞춘다.
// 서버의 청산 미리보기도 같은 이유로 fresh 다(server.py api_manual_exit_preview).
// 조회가 실패해도 미리보기는 진행한다 -- 급히 닫으려는 사람을 여기서 세우면 안 된다.
async function manualExitPreview(side, ov = null) {
  manualEntryClearConfirm();   // 🔴옛 확인(다른 방향·진입)이 계좌 조회 동안 남아 있으면 0.4초 길게 누르기 타이머가 그걸 낸다(10-16 감사)
  manualButtonsDisabled(true);
  try {
    binanceAccountLastFetchAt = 0;      // 클라이언트 30초 게이트 우회
    await refreshBinanceAccount();
    manualExitSyncButtons();
  } catch (err) {
    /* 무시 -- 아래 미리보기가 서버에서 다시 읽는다 */
  }
  return manualEntryPreview(side, "exit", ov);
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
    const on = c.dataset.v === inp.value || Number(c.dataset.v) === Number(inp.value);   // 2026-10-16 "h"(보유 1.5×) = 숫자 아님
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
  // 2026-10-05 말풍선 = 아래가 기본(사용자 지시). 아래 자리가 140px 안 되거나 아래로 펼친 패널과 겹치면 위로.
  box.classList.toggle("say-up", innerHeight - ofab.y - h < 140 || (ofab.open && !up));
  ofabClampX(el("ofabPanel"));
  ofabClampX(el("ofabSay"));
  const panel = el("ofabPanel");
  if (panel) panel.style.maxHeight = `${Math.max(160, (up ? ofab.y : innerHeight - ofab.y - h) - 16)}px`;
  if (save) {
    try { localStorage.setItem(OFAB_POS_KEY, JSON.stringify([ofab.x, ofab.y])); } catch (e) { /* 위치 기억은 편의일 뿐 */ }
  }
}

// 패널·말풍선이 화면 옆으로 넘치면 안쪽으로 민다. 🔴좁은 화면에서 버튼이 가운데보다 약간 왼쪽이면 패널(왼쪽 정렬, 380px)이
//   오른쪽으로 잘렸다(2026-10-05 시안 캡처에서 발견, 이전 판에도 있던 버그). 위치 규칙(.rightside)은 그대로 두고 넘친 만큼만 옮긴다.
function ofabClampX(node) {
  if (!node || node.hidden) return;
  node.style.transform = "";
  const r = node.getBoundingClientRect();
  const dx = r.left < 8 ? 8 - r.left : r.right > innerWidth - 8 ? innerWidth - 8 - r.right : 0;
  if (dx) node.style.transform = `translateX(${Math.round(dx)}px)`;
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
  ["ofabToggle", "ofabMore"].forEach((id) => el(id)?.setAttribute("aria-expanded", String(open)));   // 2026-10-06 LONG·SHORT 는 더 이상 창을 열지 않는다(마우스)
  // 카드에서는 range 가 칩 뒤에 숨어 있어 탭 순서에서 빠져 있다(tabindex -1). 게이지로 드러나면 넣는다.
  lanes.querySelectorAll(".chip-input").forEach((i) => { i.tabIndex = open ? 0 : -1; });
  if (open) ofabApplyDefaults();
  syncAllRangeFills(lanes);
  syncChipsets();
  ofabPlace(ofab.x, ofab.y, false);   // 열리면 위/아래·최대 높이를 다시 정한다
}

// 2026-09-28 사용자 지시: 펼치면 **진입 비율 5%**가 기본(카드의 기억된 선택은 그대로). 비율은 input 을 쏴서 칩·글자·크기 재조회가 평소 경로로 따라오게 한다.
//   (10-24 SL/TP 끄기 체크 제거 -- 손절은 서버가 모든 진입에 건다.)
function ofabApplyDefaults() {
  const fr = el("snapEntryFrac");
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
  ofabClampX(b);
  clearTimeout(ofabSayTimer);
  ofabSayTimer = setTimeout(() => { b.hidden = true; ofabSayObs?.disconnect(); ofabSayObs = null; }, 8000);
}
// 카드 결과 칸이 바뀌면(미리보기 → 발주 → 체결/거부) 그 글을 말풍선으로
function ofabWatchResult() {
  const box = el("snapEntryResult");
  if (!box) return;
  ofabSayObs?.disconnect();
  ofabSayObs = new MutationObserver(() => { const t = ([...box.querySelectorAll(".entry-note")].map((n) => n.textContent).join(" ") || box.textContent || "").trim();   /* 버튼 글자(대기 취소)는 빼고 */ if (t && !box.hidden) ofabSay(escapeHtml(t), box.querySelector(".bad") ? "bad" : ""); });
  ofabSayObs.observe(box, { childList: true, subtree: true, characterData: true, attributes: true });
}
// 꾹 눌러 바로 진입(5% · SL/TP 해제, 미리보기 오는 즉시 발주). add = 지금 포지션 방향 추가 진입(물타기),
//   아니면 2026-10-05 포지션 없음의 LONG·SHORT 버튼(사용자 지시).
function ofabQuickEntry(side, add) {
  const r = manualRule();   // 2026-10-09 자동이면 크기·손절은 서버 규칙(5% 비율·SL/TP 끔은 무시된다)
  const what = `${side === "LONG" ? "롱" : "숏"} ${r ? "자동" : "5%"} ${add ? "추가 " : ""}진입`;
  ofabApplyDefaults();
  if (manualOrderBusy || manualPreviewInFlight) return ofabSay(escapeHtml(`${what} 안 함 — 진행 중인 주문·미리보기가 있습니다.`), "bad");
  ofabWatchResult();
  ofabSay(escapeHtml(`${what} — 미리보기 받는 중…`));
  clearTimeout(entrySizeDebounce);   // 기본값이 건 크기 재조회는 필요 없다(미리보기가 같은 값을 받는다)
  manualFireOnPreview = true;
  manualEntryPreview(side, "entry");
}
function ofabQuickAdd() {
  const p = snapshotAccountPosition();
  const side = Number(p?.qty) ? String(p.side || "").toUpperCase() : "";
  if (side !== "LONG" && side !== "SHORT") return ofabSay(escapeHtml("추가 진입 안 함 — 열린 포지션이 없어 방향을 모릅니다."), "bad");
  ofabQuickEntry(side, true);
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
  // 🔴2026-10-06 사용자 «실제 계좌와 다르다 -- 진입 비율이 23% 가 아니야»: 계좌 전체 사용 증거금(거래소 totalInitialMargin)은 **미체결 주문이 묶은
  //   증거금까지** 센다(실측 $616.78 = 포지션 $262.73 + 앱에서 건 주문 ~$354 → 23% vs 포지션 10%). 버튼 = **이 포지션의 증거금**(명목 ÷ 레버리지,
  //   같은 3초 시세로 민다) ÷ 순자산. 계좌 전체(주문 포함)는 호버로.
  const posIm = Math.abs(Number(p?.notional) || 0) * k / (Number(p?.leverage) || 1);
  const used = eqLive > 0 ? posIm / eqLive * 100 : m.pct, acctUsed = eqLive > 0 ? (m.used * k) / eqLive * 100 : m.pct;
  setT("ofabPos", qty ? `${side} ${used.toFixed(0)}% · ${pnl >= 0 ? "+" : "−"}$${Math.abs(pnl).toFixed(2)}` : "주문");
  const tg0 = el("ofabToggle");
  if (tg0) tg0.title = qty ? `포지션 증거금 ${used.toFixed(1)}% (명목 ÷ ${Number(p?.leverage) || 1}배 ÷ 순자산) · 계좌 전체 사용 증거금 ${acctUsed.toFixed(1)}%`
    + (acctUsed - used > 1 ? " — 차이는 미체결 주문이 묶은 증거금" : "") : "";
  // 2026-10-05 포지션이 없으면 «주문» 대신 LONG·SHORT(꾹 = 그 방향 진입 · 짧게 = 펼치기)
  const flat = el("ofabFlat"), tg = el("ofabToggle");
  if (flat && tg) { flat.hidden = !!qty; tg.hidden = !qty; }
  box.classList.toggle("long", side === "LONG");
  box.classList.toggle("short", side === "SHORT");
  renderOfabSl(p);
  renderOfabRule(bal, live);
}
// 2026-10-09 B안(사용자 선택): 위 띠 = #snapRule 과 같은 값 · 자동이면 버튼 안에 «첫 $ · 손절 가격».
//   숫자는 서버 규칙(live_manual_peg_entry rec_rule_size·rec_rule_bracket)과 같은 식으로 여기서 3초마다 -- 첫 = REC_FIRST_SHARE(1/3) × 실효 L × 순자산,
//   손절 = 지금 가격 ∓RULE_STOP_PCT(10-14 권장 B: L = min(12.4 × 권장배수, 청산 안전, 3) · ∓5%). 실제 크기는 누를 때 서버 미리보기가 다시 정한다(다른 코인 포지션이 있으면 그만큼 줄어든다).
function renderOfabRule(bal, live) {
  const r = manualRule(), eth = RULE_ASSETS.has(activeSnapshotAsset), vmx = volMultNow();
  const strip = el("ofabRule");
  if (strip) {
    strip.hidden = !MANUAL_RULES || !eth;
    const v = el("snapRule")?.value || "0";
    strip.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === v)));
    const rr = strip.querySelector(".rule-risk");
    if (rr) rr.hidden = v === "0";
    if (v !== "0") renderRuleBox();            // «−X%»는 권장 배수가 바뀌면 같이 바뀐다(3초)
  }
  const qn = el("ofabQuote"), q = execQuote && execQuote.asset === activeSnapshotAsset && Date.now() - execQuote.at < 3 * 60e3 ? execQuote : null;
  if (qn) {
    qn.hidden = !q;
    if (q) {
      const bp = execQuoteBp((q.bid + q.ask) / 2, q.ref), txt = `${q.sym.replace(/^[A-Z]+?(?=USDC$)/, "")} ${bp > 0 ? "+" : "−"}${Math.abs(bp).toFixed(1)}bp`;
      if (qn.textContent !== txt) qn.textContent = txt;
      qn.title = `주문은 ${q.sym} — 최우선 매수 ${acctPx(q.bid)} · 매도 ${acctPx(q.ask)}, 화면(USDT) ${acctPx(q.ref)} 대비 ${bp > 0 ? "+" : "−"}${Math.abs(bp).toFixed(1)}bp`
        + ` (${Math.round((Date.now() - q.at) / 1000)}초 전 미리보기). 숏 진입·롱 청산 지정가가 화면 가격보다 싸 보이는 건 이 차이 때문입니다`;
      qn.setAttribute("aria-label", qn.title);
    }
  }
  const eq = Number(bal?.margin) || 0;
  for (const [id, side] of [["ofabLong", "LONG"], ["ofabShort", "SHORT"]]) {
    const b = el(id), sm = b?.querySelector("small");
    if (!b || !sm) continue;
    b.classList.toggle("two", !!r);
    const dir = side === "LONG" ? "롱" : "숏";
    const cr = r && vmx?.mult && vmx?.sigma_bp ? commitRuleL(side, Number(vmx.sigma_bp), Number(vmx.mult), ruleCap()) : null;
    if (cr && eq > 0 && live > 0) {
      const sl = live * (side === "LONG" ? 1 - cr.sd : 1 + cr.sd);
      // 2026-10-11 두 조각(첫 $ · 손절)을 따로 감싼다 -- 좁은 화면(styles.css ≤480px)은 «·»를 숨기고 두 줄로 쌓아 버튼 줄이 화면 안에 든다
      const lev = Number(manualLevEffective()) || 0, lEff = lev > 0 ? Math.min(cr.l, 0.9 * lev) : cr.l;   // 서버 rec_rule_size 와 같은 실효 L
      const first = `첫 $${Math.round(REC_FIRST_SHARE * lEff * eq).toLocaleString()}`, stop = `손절 ${Math.abs(sl) < 10 ? sl.toFixed(4) : Math.round(sl).toLocaleString()}`;
      if (sm.textContent !== `${first} · ${stop}`) {
        const part = (t, c) => Object.assign(document.createElement("span"), { textContent: t, className: c || "" });
        sm.replaceChildren(part(first), part(" · ", "sep"), part(stop));
      }
      b.title = `1.5초 꾹 = ${dir} 자동 진입 — 손절 시 순자산 −${Math.round(100 * lEff * cr.sd)}%: 왕복 상한 순자산 × ${lEff.toFixed(2)}(첫 진입 그 1/3 · 물타기 권장 첫 진입가 ${side === "LONG" ? "−" : "+"}0.75%에 나머지), 손절 첫 진입가 ${side === "LONG" ? "−" : "+"}${100 * cr.sd}%(거래소 스탑), 익절 없음`;
    } else if (r) {
      setT(sm.id || id + "Frac", "자동 — 변동성 대기");
      b.title = `${dir} 자동 진입 — 권장 배수(직전 24시간 변동성)나 잔고가 아직 없습니다`;
    } else {
      setT(sm.id || id + "Frac", "5%");
      b.title = `1.5초 꾹 = ${dir} 5% 진입 — 손절 첫 진입가 ${side === "LONG" ? "−" : "+"}5%(거래소 스탑, 물타기해도 안 물러남) · 익절 없음`;
    }
  }
  ofabTrendTags();
}
// 2026-10-16 사용자 «같이 해서 시안» → «배포»(시안 ③, ETH 만): 일봉 추세(latestTrend 5표)와 반대 버튼 = 점선 «역추세» ·
//   추세 쪽 버튼 = 직전 30초 추세 반대 테이커 순흐름이 최근 1시간 30초 창들의 90분위 이상이면 «지정가 자리»(창 10개 미만이면 대기).
//   근거: 실원장 79왕복 추세 쪽 +898$ · 역추세 −468$(최악 −549$ 포함) · 반대 흐름 뒤 1~5분 되돌림(IC −.03) -- 수익 신호 아님, 지정가 걸 순간 표시.
function ofabTrendTags() {
  const t = latestTrend?.ok && RULE_ASSETS.has(activeSnapshotAsset) ? latestTrend : null, up = !!t && t.signal > 0;
  let now30 = 0, q90 = Infinity, last = 0;
  if (t) {
    for (const k of supply1s.keys()) if (k > last) last = k;
    const net = (e) => { let x = 0; for (let k = e - 29; k <= e; k++) { const r = supply1s.get(k); if (r) x += r[4] - r[5]; } return up ? -x : x; };
    now30 = net(last);
    const hist = [];
    for (let e = last - 30; e > last - 3600; e -= 30) if (supply1s.has(e) && supply1s.has(e - 29)) hist.push(net(e));
    hist.sort((a, b) => a - b);
    if (hist.length >= 10) q90 = hist[Math.floor(hist.length * 0.9)];
  }
  const on = now30 > 0 && now30 >= q90;
  for (const [id, side] of [["ofabLong", "LONG"], ["ofabShort", "SHORT"]]) {
    const b = el(id), tag = b?.querySelector(".ofab-tag");
    if (!b || !tag) continue;
    const ct = !!t && up !== (side === "LONG"), mk = !!t && !ct;
    b.classList.toggle("ct", ct);
    tag.hidden = !t;
    tag.className = `ofab-tag ${ct ? "ct" : "mk"}${mk && on ? " on" : ""}`;
    const txt = ct ? "역추세" : on ? "지정가 자리" : "지정가 대기";
    if (tag.textContent !== txt) tag.textContent = txt;
    if (ct) b.title += `\n\n⚠ 역추세 — 일봉 추세 ${up ? "▲ 상승" : "▼ 하락"} ${t.ups}/${t.votes.length}표의 반대. 실원장 79왕복: 추세 쪽 +898$ · 역추세 −468$(최악 −549$ 가 여기서)`;
    else if (mk) b.title += `\n\n지정가 ${on ? "자리" : "대기"} — 직전 30초 ${up ? "매도" : "매수"} 쏠림 ${now30.toFixed(0)} (최근 1시간 90분위 ${isFinite(q90) ? q90.toFixed(0) : "-"}). 추세 쪽 진입을 지정가로 걸기 좋은 순간 · 1~5분 되돌림(IC −.03) 근거, 수익 신호 아님`;
  }
}
// 🔴2026-09-26 사용자 지시 «미실현손익 3초 갱신». 계좌 조회는 30초라 그 사이 숫자가 멈춰 있었다. 거래소를 3초마다
//   부르는 대신 **거래소 값 + 그 뒤 시세 변화 × 수량**으로 민다. 기준 시세는 계좌 스냅샷이 바뀐 순간의 실시간 시세라
//   심볼 차이(주문 ETHUSDC · 시세 ETHUSDT, ~1.7bp)는 변화분에서 지워진다. 다음 계좌 조회가 오면 거래소 값으로 되돌아간다.
// 🔴2026-09-30 사용자 «플로팅 버튼 미실현이 실제와 안 맞는다»: 기준 시세를 «계좌 스냅샷이 브라우저에 도착한 순간»으로 잡았는데,
//   거래소 손익은 서버가 조회한 시각(generated_at) 값이고 서버 캐시 때문에 도착까지 30초 넘게 늦다(실측 33.8초).
//   그 사이 움직임이 통째로 빠져 숏 1.6 ETH 면 $2 움직임에 $3 틀렸다. 기준 = generated_at 초의 시세(1초 수급 칸의 마지막가).
const ofabPnlRef = { acct: null, price: 0, asset: null };
function ofabPriceAt(sec) {                     // 그 초(없으면 5초 안 직전 초)의 마지막 체결가 -- 1초 수급 칸 [.., 가격]
  for (let s = sec; s > sec - 5; s -= 1) { const r = supply1s.get(s); if (r && r[6] > 0) return r[6]; }
  return 0;
}
function ofabLivePnl(p, qty, side) {
  const base = Number(p?.unrealized_pnl) || 0;
  const live = Number(latestLivePriceByAsset[activeSnapshotAsset] || 0);
  if (!qty || !(live > 0)) return base;
  if (ofabPnlRef.acct !== latestBinanceAccount || ofabPnlRef.asset !== activeSnapshotAsset) {   // 탭을 바꾸면 다른 코인 기준가로 셈했다
    ofabPnlRef.acct = latestBinanceAccount;
    ofabPnlRef.asset = activeSnapshotAsset;
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
  const OFAB_HOLD_MS = 1500;   // 2026-09-28 사용자 지시 0.5 -> 2초 · 2026-09-30 «너무 느리다» -> 1.5초 (styles.css .holding 채움 시간과 같이)
  // 같은 손짓을 세 버튼이 쓴다: «주문»(포지션 있음 = 추가 진입) · 2026-10-05 LONG·SHORT(포지션 없음 = 그 방향 진입).
  // onTap = 짧게 눌렀을 때(기본 = 펼치기/접기). 인자 = 그 누름의 pointerType(터치는 꾹 발주가 없으니 호출 측이 갈라 쓴다).
  const ofabHold = (btn, fire, onTap = () => ofabSetOpen(!ofab.open)) => {
    if (!btn) return;
    let hold = null, held = false, ptype = "mouse";
    const holdEnd = () => { if (hold) { clearTimeout(hold.t); hold = null; } btn.classList.remove("holding"); };
    btn.addEventListener("pointerdown", (e) => {
      holdEnd(); held = false; ptype = e.pointerType || "mouse";
      // 🔴2026-09-28 사용자 지시 «모바일에서는 넣으면 안돼» -- 터치는 꾹 누르기 발주가 없다(짧게 = 펼치기만).
      //   진입 버튼과 같은 판정(이벤트마다 pointerType) -- 터치 노트북의 마우스는 그대로 된다.
      if (e.pointerType === "touch" || btn.disabled) return;
      hold = { x: e.clientX, y: e.clientY, t: setTimeout(() => { hold = null; held = true; btn.classList.remove("holding"); fire(); }, OFAB_HOLD_MS) };
      btn.classList.add("holding");
    });
    btn.addEventListener("pointermove", (e) => { if (hold && Math.hypot(e.clientX - hold.x, e.clientY - hold.y) > 8) holdEnd(); });
    ["pointerup", "pointerleave", "pointercancel"].forEach((ev) => btn.addEventListener(ev, holdEnd));
    btn.addEventListener("contextmenu", (e) => e.preventDefault());   // 모바일 길게 누르기 메뉴
    btn.addEventListener("click", () => { if (held) { held = false; return; } onTap(ptype); });
  };
  // 2026-10-06 사용자 «LONG 누른 뒤 다른 창이 떠서 또 눌러야 한다»: 실측(주문 차단 하네스) 1.5초 꾹은 이미 창 없이 발주(미리보기 1.51초·제출 1.68초) --
  //   두 번 누르게 된 건 1.5초 전에 떼서 «짧게 = 펼치기»로 갔을 때다. 마우스는 짧게 눌러도 창을 열지 않고 꾹 누르라고 말한다(상세는 옆 펼치기 버튼).
  //   🔴터치는 꾹 발주가 없으므로(09-28 «모바일에서는 넣으면 안돼») 짧게 = 펼치기 그대로 -- 안 그러면 휴대폰에서 주문할 길이 없어진다.
  const dirTap = (side) => (ptype) => (ptype === "touch" ? ofabSetOpen(!ofab.open)
    : ofabSay(escapeHtml(`${side === "LONG" ? "롱" : "숏"} ${manualRule() ? "자동" : "5%"} 진입은 1.5초 꾹 누르세요 — 상세 주문은 옆 펼치기 버튼`)));
  ofabHold(tgl, ofabQuickAdd);
  ofabHold(el("ofabLong"), () => ofabQuickEntry("LONG", false), dirTap("LONG"));
  ofabHold(el("ofabShort"), () => ofabQuickEntry("SHORT", false), dirTap("SHORT"));
  el("ofabMore")?.addEventListener("click", () => ofabSetOpen(!ofab.open));   // 2026-10-06 상세 펼치기(누르기만 -- 주문 없음)
  el("ofabRule")?.addEventListener("click", (e) => {   // 2026-10-09 B안 띠 -- 카드 «크기» 칩과 같은 입력을 바꾼다(주문 없음)
    const b = e.target.closest("button[data-v]"), inp = el("snapRule");
    if (!b || !inp || inp.value === b.dataset.v) return;
    inp.value = b.dataset.v;
    inp.dispatchEvent(new Event("input", { bubbles: true }));
    renderOfab();
  });
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
  // 2026-10-05 스위칭 = 지금 쪽 100% peg 청산 → 반대쪽 peg 진입. 색·라벨 = 갈 쪽.
  const sw = el("snapSwitch"), held = lastExitPositions.has("LONG") ? "LONG" : lastExitPositions.has("SHORT") ? "SHORT" : "";
  if (sw) {
    sw.hidden = !held;
    sw.classList.toggle("to-short", held === "LONG");
    sw.classList.toggle("to-long", held === "SHORT");
    setT("snapSwitchTo", held === "LONG" ? "→ 숏" : "→ 롱");
  }
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
