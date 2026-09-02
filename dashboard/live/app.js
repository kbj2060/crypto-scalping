const API_EVENTS_URL = "/api/events";
const API_OPS_STATUS_URL = "/api/ops-status";
const API_VREB_ECON_SHADOW_URL = "/api/v-rebound-econ-shadow";
const API_EVIDENCE_SIGNALS_URL = "/api/evidence-signals";
const API_EVIDENCE_SIGNALS_PROVISIONAL_URL = "/api/evidence-signals-provisional";
// BTC 코인 페이지 전용 증거신호 패널(2026-09-02) -- 이전엔 코인탭과 무관하게 항상 ETH
// 엔드포인트를 썼다(사용자 신고: "비트코인 페이지에 이더리움 증거신호가 나온다"). SOL/XRP/HYPE는
// 증거신호 파이프라인 자체가 없어(ETH+BTC만 존재) EVIDENCE_SIGNAL_SUPPORTED_ASSETS로 게이팅한다.
const API_BTC_EVIDENCE_SIGNALS_URL = "/api/btc-evidence-signals";
const EVIDENCE_SIGNAL_SUPPORTED_ASSETS = ["eth", "btc"];
const API_V_REBOUND_URL = "/api/v-rebound-signal";
const API_BASIS_LIQUIDATION_URL = "/api/basis-liquidation-signal";
const API_LIQUIDATION_DIRECTION_URL = "/api/liquidation-direction-signal";
const API_LIQUIDATION_MAP_URL = "/api/liquidation-map";
const API_REGIME_WIDE24_URL = "/api/regime-wide24";
const API_REGIME_BTC_URL = "/api/regime-btc";
const API_MACRO_CALENDAR_URL = "/api/macro-calendar";
const API_LIQ_BURST_STATE_URL = "/api/liq-burst-state";
const API_LIQUIDATION_5M_URL = "/api/liquidation-5m-signal";
const API_SESSION_ALERTS_URL = "/api/session-alerts";
const POLL_MS = 2500;
// 2026-08-25 perf pass: split from the Live tab's own chart-render throttle (removed with the Live
// tab) -- the Snapshot chart's own data
// (latestLiquidationMap) only changes once per LIQUIDATION_MAP_POLL_MS (5min), so redrawing its
// ~250-400 SVG nodes every 5s (60x more often than the data changes) was pure waste. Coarser than
// the Live chart's interval on purpose: this chart is read-only reference, not something a user
// pans/zooms live (see renderSnapshotChart()'s own comment). Kept as ONE throttle interval (not
// split further into "redraw candles at 5s, density at 5min") because the density band's sweep-
// darkening depends on live candle highs/lows, not just latestLiquidationMap -- decoupling those
// two redraw cadences would let a real price sweep sit un-darkened for up to 5min, undermining why
// sweep-darkening exists. Slower-but-synchronized beats faster-but-inconsistent here.
const SNAPSHOT_CHART_RENDER_MIN_INTERVAL_MS = 20000;
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
const setH = (id, html) => { const target = el(id); if (target) target.innerHTML = html; };
const hasOwn = (obj, key) => Object.prototype.hasOwnProperty.call(obj || {}, key);

// Snapshot tab: same 5 model indicators (bot-state ones only -- see the "own fetch cycle" model
// indicators like liq_pressure/liq_direction, which carry their own server-provided
// tone_history instead), but as a tone-per-bar strip (matching the evidence signals' activity-strip
// graph) instead of a continuous sparkline. Stores the ALREADY-COMPUTED tone string
// ("good"/"bad"/"neutral") from each render() pass rather than re-deriving it from raw values
// later -- some tones (tail risk) depend on more than one raw field, so capturing the tone at
// computation time is the only way to stay exactly consistent with what the live cards show,
// instead of an approximation that ignores the cross-field dependency.
const toneHistory = { whale: [], liq_cascade: [], retail_flow: [] };
// Parallel to toneHistory, same keys/push/shift cadence -- these 5 indicators have no server-side
// timestamp per reading (client-accumulated tally, see comment above), so the only honest per-bar
// time is "when this browser tab actually pushed the reading", recorded here at push time.
const toneHistoryTimes = { whale: [], liq_cascade: [], retail_flow: [] };
function pushToneHistory(key, tone) {
  const arr = toneHistory[key];
  if (!arr) return;
  arr.push(tone || "neutral");
  if (arr.length > MICRO_HISTORY_MAX) arr.shift();
  const times = toneHistoryTimes[key];
  times.push(new Date().toISOString());
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
let evidenceLastFetchAt = 0;
let evidenceProvisionalLastFetchAt = 0;
// Confirmed signals poll every 5min (EVIDENCE_POLL_MS) but the provisional preview polls every 10s
// and already knows bar_open_utc -- so it's the first to find out when the bar it was tracking has
// closed. Tracked here so refreshEvidenceSignalsProvisional() can detect that transition and force
// an immediate confirmed refresh instead of leaving the user looking at a stale confirmed count for
// up to 5 more minutes (2026-08-27, user report: had to manually reload to see a just-closed bar's
// signals show up in the confirmed view).
let lastSeenProvisionalBarOpenUtc = null;
// Per-signal confirmed history (bottom_history/top_history/latest_bar_utc), captured each time
// renderEvidenceSignals() runs -- lets refreshEvidenceSignalsProvisional() redraw each signal's
// strip with one extra live bar every ~10s without waiting on (or duplicating) the 5-min confirmed
// fetch. Keyed by signal name (same keys as EVIDENCE_STRIP_CHIP_IDS).
let evidenceHistoryBySignal = {};
// 2026-09-01 (user request): V자반등도 증거신호처럼 진행중인(아직 안 닫힌) 봉에서 신호가 뜨게 해달라는
// 요청 -- 새 fetch/새 TabPFN 호출을 추가하는 대신, 이미 10초마다 도는 evidence-signals-provisional
// 응답(9트리거 중 8개와 완전히 동일한 이름/정의)을 여기 저장해뒀다가 render()에서 재사용. TabPFN은
// 여전히 CONFIRMED(닫힌 봉) 경로에서만 돔 -- 진행중 미리보기는 "트리거가 지금 형성중"만 알려줄 뿐
// rebound/continuation 판정은 하지 않음(그 판정엔 닫힌 봉 기준 23피쳐가 필요, 증거신호 진행중
// 미리보기가 확률 없이 발동여부만 보여주는 것과 같은 이유). local_extreme(9번째 트리거)은 구조상
// 미래 봉이 있어야 확정 가능해 진행중 판정 자체가 불가능 -- 8개만 씀, 의도적 누락.
let latestEvidenceSignalsProvisional = null;
let latestVRebound = null;
let vReboundLastFetchAt = 0;
// Long/short liquidation volume gauge (recreated 2026-08-27, see liquidationVolumeGaugeHtml()) --
// backend (scripts/live_liquidation_5m_signal_20260825.py) never stopped running, only this
// frontend consumer had been removed.
let latestLiquidation5m = null;
let liquidation5mLastFetchAt = 0;
// 베이시스 청산압박 model indicator (replaces 독성/toxicity, 2026-08-27) -- own fetch cycle, same
// dashboard-side-computed category as latestVRebound above (scripts/live_spot_perp_basis_signal_
// 20260827.py). RISK GAUGE, not a price-direction claim -- see MODEL_INDICATOR_DETAIL.liq_pressure.
let latestBasisLiquidation = null;
let basisLiquidationLastFetchAt = 0;
// Sudden-liquidation alert (2026-08-27) -- backed by tail_risk_interceptor.py's event-triggered
// liq_burst_state.json (own file, own writer, updated the instant a new @forceOrder event lands),
// not the once-a-minute tail_risk.duckdb path the gauge above reads. Own short poll interval since
// the source can change sub-second during a real cascade -- see API_LIQ_BURST_STATE_URL below.
let latestLiqBurstState = null;
let liqBurstStateLastFetchAt = 0;
// Directional-only liquidation tilt (liq_net_z_12, 2026-08-25) -- own fetch cycle, same
// dashboard-side-computed category as latestVRebound above. Model-indicator tier (like 수급
// 흐름), explicitly NOT an evidence-signal-tier chip -- see
// scripts/live_liquidation_direction_signal_20260825.py docstring for why (no PnL/economic claim).
let latestLiquidationDirection = null;
let liquidationDirectionLastFetchAt = 0;
// Liquidation map (Snapshot tab, 2026-08-24) -- estimated support/resistance, own fetch/render
// cycle same as latestVRebound above (computed dashboard-side, not part of trading_bot.py state).
// lastSnapshotHistoryFetchAt tracks the candle history this panel's chart needs (activeSnapshotAsset,
// see below), independently of activeChartAsset (the Live tab's own, separate coin selector).
let latestLiquidationMap = null;
let latestRegimeWide24 = null;
let latestRegimeBtc = null;
// Confirmed evidence-signals payload (2026-08-31, feeds evidenceSignalTpLevels() below) -- stashed
// globally like latestLiquidationMap/latestRegimeWide24 so the Snapshot chart's own render cycle
// can read it without renderEvidenceSignals() needing to know about the chart.
let latestEvidenceSignals = null;
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
let macroCalendarLastFetchAt = 0;
let vrebEconShadowLastFetchAt = 0;
const VREB_ECON_SHADOW_POLL_MS = 60000;
let sessionAlertsLastFetchAt = 0;
let lastSnapshotHistoryFetchAt = 0;
let lastSnapshotChartRenderAt = 0;
let lastModelIndicatorHtmlByTarget = {};
let activePageTab = "snapshot"; // "ops" | "snapshot" (라이브 탭 제거, 2026-08-31) -- must match index.html's default active tab (data-page-tab="snapshot" carries the initial "active" class)
let isScrolling = false;
let scrollIdleTimer = 0;
let dashboardEvents = null;
const OPS_POLL_MS = 30000;
// Matches CANDLE_HISTORY_POLL_MS's reasoning: the underlying data is 5m bars, so a new reading
// only ever exists once every 5 minutes -- polling more often than that just re-fetches the same
// latest-closed-bar result (server-side cache would absorb it, but there's no reason to ask).
const EVIDENCE_POLL_MS = 300000;
// Live PREVIEW of the currently-forming bar (2026-08-26, user request) -- deliberately much
// faster than EVIDENCE_POLL_MS above, which stays 5min because the CONFIRMED signal genuinely
// only changes on bar close. This one changes every ~10s because it's reading the in-progress
// bar's still-changing high/low/volume -- see renderEvidenceSignalsProvisional()'s "미확정"
// labeling, never merged into the confirmed reading.
const EVIDENCE_PROVISIONAL_POLL_MS = 10000;
const V_REBOUND_POLL_MS = 60000; // matches server's own 60s cache (EVIDENCE_SIGNAL_CACHE_SECONDS) --
                                  // event-triggered, so a fresher poll matters more than for OI 급변's old 5m-poller data
const LIQUIDATION_5M_POLL_MS = 60000; // matches server's own 60s cache + the 1-row-per-minute source
// 2026-08-29 (user report + fix): was 300000 ("basis_z48 is a 5m-bar z-score, no faster") -- that
// reasoning conflated the DATA's own update cadence (every 5m, on bar close, unchanged) with the
// POLL interval (how often the browser checks for a new value, which is a separate concern).
// Polling exactly as often as the bar closes is the risky choice, not the safe one: poll phase vs.
// bar-close phase is unsynchronized, so in the worst case a poll lands just before a bar closes and
// the next one doesn't fire until nearly a full 5m later -- confirmed live (direct SSH timing showed
// the backend cache itself refreshes within 20-41s of each bar close; the user-visible multi-minute
// lag was entirely this poll/bar-close phase misalignment). Matches V_REBOUND_POLL_MS now
// (60s, comfortably shorter than the 5m bar period they're also built on,
// and already proven not to have this problem) -- polling faster than the data changes never shows
// a value sooner than it's true, it only shrinks the worst-case detection lag.
const BASIS_LIQUIDATION_POLL_MS = 60000;
// 2026-08-27: liq_burst_state.json is written the instant a new liquidation event arrives (see
// tail_risk_interceptor.py::_write_liq_burst_state()), not on a timer -- polling faster than ~1s
// wouldn't surface anything sooner than the file itself changes, given the remaining hop (this
// fetch) is the last one in the chain.
const LIQ_BURST_STATE_POLL_MS = 1000;
const LIQUIDATION_DIRECTION_POLL_MS = 60000; // same source cadence as liquidation-5m signal above
const LIQUIDATION_MAP_POLL_MS = 300000; // matches server-side cache -- structure moves slowly
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

function renderSnapshotAssetTabs() {
  document.querySelectorAll("#snapshotAssetTabs .asset-tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.asset === activeSnapshotAsset);
  });
}

async function setActiveSnapshotAsset(asset) {
  if (!SNAPSHOT_ASSET_KEYS.includes(asset) || asset === activeSnapshotAsset) return;
  activeSnapshotAsset = asset;
  renderSnapshotAssetTabs();
  // Clear the 4 wired signals' cached readings + their poll-interval gates immediately -- without
  // this, the panels would keep showing the PREVIOUS coin's numbers (mislabeled as the new one)
  // until each signal's own poll interval next elapses (up to 5min for the slowest).
  latestBasisLiquidation = null;
  latestLiquidationDirection = null;
  latestLiquidation5m = null;
  latestLiquidationMap = null;
  basisLiquidationLastFetchAt = 0;
  liquidationDirectionLastFetchAt = 0;
  liquidation5mLastFetchAt = 0;
  liquidationMapLastFetchAt = 0;
  lastSnapshotHistoryFetchAt = 0;
  // 2026-09-02: 증거신호도 코인탭에 따라 ETH/BTC(그 외엔 미지원)로 갈리므로 위 4개와 같은 이유로
  // 즉시 초기화+재조회한다. resetEvidenceStripChips()는 재조회 전에 먼저 불러 이전 자산의 칩이
  // 잠깐이라도 남아있지 않게 한다(특히 smt_divergence처럼 상대 자산엔 없는 신호의 칩).
  latestEvidenceSignals = null;
  evidenceLastFetchAt = 0;
  resetEvidenceStripChips();
  // ETH 전용 진행중(미확정) 미리보기 배지 -- renderEvidenceSignalsProvisional()가 ETH가 아닌
  // 탭에선 더 이상 이 배지를 건드리지 않으므로(그 함수의 early-return 참고), 스위치 직후엔
  // 여기서 직접 중립 문구로 비워 마지막으로 봤던 ETH 값이 새 탭에 남아있지 않게 한다.
  if (asset !== "eth") {
    const provisionalBadge = el("evidenceProvisionalBadge");
    if (provisionalBadge) { provisionalBadge.className = "ops-badge neutral"; provisionalBadge.textContent = "진행중 미리보기 (ETH 전용)"; }
  }
  await Promise.all([
    refreshBasisLiquiditySignal(),
    refreshLiquidationDirectionSignal(),
    refreshLiquidation5mSignal(),
    refreshLiquidationMap(),
    refreshEvidenceSignals(),
    maybeFetchSnapshotChartHistory(),
  ]);
  if (latestMainState) render(latestMainState, latestCompactState);
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

function fmtPct(v, d = 2) {
  const n = Number(v || 0);
  const s = n >= 0 ? "+" : "";
  return `${s}${n.toFixed(d)}%`;
}

function fmtPctNoPlus(v, d = 2) {
  return `${Number(v || 0).toFixed(d)}%`;
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

function directionalCaution(score, th = 0.1) {
  const x = Number(score || 0);
  if (x >= th) return "롱 진입";
  if (x <= -th) return "숏 진입";
  return "중립";
}

function flowRead(micro) {
  const x = Number(micro.nif_whale || 0);
  if (x >= 0.2) return "큰손 매수가 강하게 들어옴";
  if (x >= 0.05) return "큰손 매수 유입";
  if (x <= -0.2) return "큰손 매도가 강하게 나옴";
  if (x <= -0.05) return "큰손 매도 유입";
  return "큰손 수급은 뚜렷하지 않음";
}

// nif_retail: same _compute_nif_and_taker() split as nif_whale above, retail (small-size) leg
// instead of whale leg. Added 2026-08-25 after a same-day IC screen found real (non-noise) short-
// horizon direction information here -- see MODEL_INDICATOR_DETAIL.retail_flow for the numbers.
function retailFlowRead(micro) {
  const x = Number(micro.nif_retail || 0);
  if (x >= 0.2) return "리테일 매수가 강하게 들어옴";
  if (x >= 0.05) return "리테일 매수 유입";
  if (x <= -0.2) return "리테일 매도가 강하게 나옴";
  if (x <= -0.05) return "리테일 매도 유입";
  return "리테일 수급은 뚜렷하지 않음";
}

function liqDirectionSubText(sig) {
  if (!sig || !sig.warmed_up) return "웜업 중";
  if (sig.direction === "bullish") return "상승압력";
  if (sig.direction === "bearish") return "하락압력";
  return "중립";
}

function fmtBarsAgo(bars) {
  if (bars === null || bars === undefined) return "발화 이력 없음";
  if (bars === 0) return "지금";
  const min = bars * 5;
  return min < 60 ? `${min}분 전` : `${(min / 60).toFixed(1)}시간 전`;
}

// 2026-08-27: replaces toxRead/toxHint (독성/toxicity chip removed -- shadow_toxicity_score was
// independently confirmed uninformative on both direction and volatility-framing axes, see
// eth_model_indicator_volatility_framing_screen_20260825 memory). sig here is the raw
// latestBasisLiquidation payload (server-computed, not part of classifyIndicators' micro/tail
// inputs -- same "own fetch cycle" category as latestVRebound, see that variable's own comment).
function basisLiquiditySubText(sig) {
  if (!sig || !sig.warmed_up) return "웜업 중";
  if (sig.direction === "short_pressure") return "숏압박↑";
  if (sig.direction === "long_pressure") return "롱압박↑";
  return "안정";
}

// Sudden-liquidation alert banner (2026-08-27) -- reads liq_burst_state.json (event-triggered, see
// tail_risk_interceptor.py::_write_liq_burst_state()), a faster/more prominent sibling to the
// liq_cascade model-indicator tile below (which reads the same hawkes/z-score concept but via the
// 10s-cadence dashboard_state.json path). Shown only while hawkes_active -- an alert that's always
// visible isn't an alert, see execution-alert-banner's own hidden-by-default precedent above.
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
function liquidationVolumeGaugeHtml() {
  const liq5m = latestLiquidation5m;
  const burst = latestLiqBurstState;
  const cascadeNow = !!(burst && burst.available && burst.hawkes_active);
  let liveCue = "";
  if (cascadeNow) {
    // 2026-08-27: the standalone liqBurstAlertBanner was folded into this one-line cue at user
    // request (renderLiqBurstAlert() removed) -- same side-selection fix (max of both sides, not
    // the possibly-stale crisis_type label) carried over, see that removed function's history in
    // feedback_dashboard_liq_burst_alert memory if this needs revisiting.
    const bLong = Number(burst.long_usd_1m || 0);
    const bShort = Number(burst.short_usd_1m || 0);
    const bUsd = Math.max(bLong, bShort);
    const bSide = bShort > bLong ? "숏청산" : "롱청산";
    const bPct = Math.round(clamp01(Number(burst.hawkes_decay_level) || 0) * 100);
    const detail = bUsd > 0 ? ` · ${bSide} ${fmtUsdCompact(bUsd)} · 에너지${bPct}%` : ` · 에너지${bPct}%`;
    liveCue = `<span class="liq-vol-gauge-live"><span class="liq-vol-gauge-live-dot" aria-hidden="true"></span>지금 감지중${detail}</span>`;
  }
  const tag = `<span class="liq-vol-gauge-tag">청산 규모 <span class="liq-vol-gauge-window">(최근 30분 누적)</span>${liveCue}</span>`;
  if (!liq5m || !liq5m.warmed_up) {
    return `<div class="liq-vol-gauge quiet">${tag}<span class="liq-vol-gauge-quiet-text">집계 중...</span></div>`;
  }
  const longUsd = Number(liq5m.long_usd_5m || 0);
  const shortUsd = Number(liq5m.short_usd_5m || 0);
  const total = longUsd + shortUsd;
  if (!(total > 0)) {
    return `<div class="liq-vol-gauge quiet">${tag}<span class="liq-vol-gauge-quiet-text">청산 없음 — 안정</span></div>`;
  }
  const longPct = (longUsd / total) * 100;
  const shortPct = (shortUsd / total) * 100;
  return `<div class="liq-vol-gauge">
      ${tag}
      <div class="liq-vol-gauge-track">
        <div class="liq-vol-gauge-fill long" style="width:${longPct}%;"></div>
        <div class="liq-vol-gauge-fill short" style="width:${shortPct}%;"></div>
      </div>
      <div class="liq-vol-gauge-labels">
        <span class="liq-vol-gauge-label long">롱 ${fmtUsdCompact(longUsd)}</span>
        <span class="liq-vol-gauge-label short">숏 ${fmtUsdCompact(shortUsd)}</span>
      </div>
    </div>`;
}

// 7th model-internal indicator -- 2026-08-25, user asked for a dedicated "청산 캐스케이드" tile
// rather than folding this into 꼬리 리스크's text. Distinct focus from that indicator: 꼬리
// 리스크's aftershock_prob is a forward-looking blended probability of MORE shock still to come;
// this is the raw "is a cascade actually happening right now" state straight from
// tail_risk_interceptor.py's own 3-stage design (detector/discriminator/decay-timer) -- which side,
// and how much of the initial energy is left. z>=2.0 threshold for the 주의 tier reuses the exact
// value tail_risk_interceptor.py's own status_line() already uses for "급증⚠️", not a new number.
function liqCascadeHint(tail) {
  if (tail.hawkes_active) return "위험";
  const zPeak = Math.max(Number(tail.z_long || 0), Number(tail.z_short || 0));
  if (zPeak >= 2.0) return "주의";
  return "안정";
}

function liqCascadeLiveDetail(tail) {
  // 2026-08-25: both the active (진행중 · 에너지 잔량 ...%) and calm (평온 (Z:...)) detail lines
  // were removed at the user's request -- the row's own badge (위험/안정) already says enough for
  // those two states. Only the watch tier keeps a detail line, since "주의" alone doesn't explain
  // why (a live Z number does).
  if (tail.hawkes_active) return "";
  const zPeak = Math.max(Number(tail.z_long || 0), Number(tail.z_short || 0));
  if (zPeak >= 2.0) return `청산 급증 감지(Z:${zPeak.toFixed(1)}) · 캐스케이드 전환 전`;
  return "";
}

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
function classifyIndicators(micro, tail) {
  micro = micro || {};
  tail = tail || {};
  const cascadeSignal = liqCascadeHint(tail);
  const whaleTone = Number(micro.nif_whale || 0) > 0.05 ? "good" : (Number(micro.nif_whale || 0) < -0.05 ? "bad" : "neutral");
  const retailFlowTone = Number(micro.nif_retail || 0) > 0.05 ? "good" : (Number(micro.nif_retail || 0) < -0.05 ? "bad" : "neutral");
  const cascadeTone = signalTone(cascadeSignal);
  return {
    liq_cascade: { tone: cascadeTone, valueText: liqCascadeLiveDetail(tail), subText: cascadeSignal },
    whale: { tone: whaleTone, valueText: flowRead(micro), subText: directionalCaution(micro.nif_whale, 0.05) },
    retail_flow: { tone: retailFlowTone, valueText: retailFlowRead(micro), subText: directionalCaution(micro.nif_retail, 0.05) },
    cascadeSignal, whaleTone, cascadeTone,
  };
}

function setMeter(fillId, value01, tone = "good") {
  const fill = el(fillId);
  if (!fill) return;
  fill.style.width = `${Math.round(clamp01(value01) * 100)}%`;
  fill.className = tone;
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

async function fetchBinanceHistory(asset) {
  try {
    const res = await fetch(`/api/market-history?asset=${asset}`, { cache: "no-store" });
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
  renderSnapshotChart();
}

function applyDashboardEvent(payload) {
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
  if (!latestMainState || isScrolling || !payload?.state?.state) return;
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
  return ({ trading_bot: "트레이딩 봇", ops_watchdog: "Ops Watchdog", trading_bot_process: "트레이딩 봇 프로세스", decision_snapshot: "의사결정 스냅샷", trading_bot_heartbeat: "봇 heartbeat", data_pipeline: "데이터 파이프라인", pipeline_contract: "파이프라인 계약", market_data_sources: "시장 데이터 소스", dashboard_state: "대시보드 상태", execution_contract: "실행 안전 계약", runtime_resources: "시스템 자원", watchdog_storage: "watchdog 저장소", btc_multislot_shadow_process: "BTC 멀티슬롯 shadow" })[value] || String(value || "알 수 없음");
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

async function refreshOpsStatus() {
  const now = Date.now();
  if (now - opsLastFetchAt < OPS_POLL_MS) return;
  opsLastFetchAt = now;
  try {
    const res = await fetch(API_OPS_STATUS_URL, { cache: "no-store", headers: opsStatusEtag ? { "If-None-Match": opsStatusEtag } : {} });
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
// spaced (server-computed histories: evidence signals/v_rebound at 5-min klines, liq_direction at
// 1-min tail_risk_1m rows) -- the payload only ever sends the LATEST bar's timestamp, so the rest
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
function toneStripSvg(tones, times, provisionalLast, liveFiring, key, rawFire) {
  const list = Array.isArray(tones) ? tones : [];
  const timeList = Array.isArray(times) ? times : [];
  const fireList = Array.isArray(rawFire) ? rawFire : [];
  const n = Math.max(list.length, 1);
  const w = 240, h = 15, gap = 1.5;
  const bw = Math.max((w - gap * (n - 1)) / n, 1);

  // Group consecutive equal tones into segments. The still-forming provisional bar (always the last
  // array entry, see evidenceStripSvg's liveTone param) never merges into the segment before it even
  // when its tone happens to match -- keeps evidence-bar-provisional's softened fill scoped to only
  // the genuinely-unconfirmed portion instead of bleeding across a whole merged block. 2026-09-01
  // (user request): a bar where the signal genuinely re-fired (rawFire[i] -- independent of tone,
  // see evidenceStripSvg's fill-window history) also never merges backward, so a second real
  // trigger inside an already-active fill window still shows as a visible new segment boundary
  // instead of silently vanishing into one long block. Callers that don't pass rawFire (model
  // indicators etc.) get fireList=[] -- fireList[i] is always undefined/falsy, so behavior is
  // unchanged for them.
  const segments = [];
  for (let i = 0; i < n; i++) {
    const tone = list[i] || "neutral";
    const isProvisionalBar = !!(provisionalLast && i === n - 1);
    const isFreshFire = !!fireList[i];
    const prev = segments[segments.length - 1];
    if (prev && prev.tone === tone && !isProvisionalBar && !isFreshFire) {
      prev.end = i;
    } else {
      segments.push({ tone, start: i, end: i, isProvisional: isProvisionalBar });
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
    const hoverAttrs = t ? ` data-t="${t}" data-t-end="${tEnd || t}" data-tone="${tone}" onmouseenter="showStripBarTime(this)" onmouseleave="hideStripBarTime(this)"` : "";
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

// liveTone/liveIso (2026-08-26, optional) append one extra bar for the still-forming (unconfirmed)
// bar after latestIso, sourced from the provisional preview -- see refreshEvidenceSignalsProvisional,
// which re-calls this every ~10s reusing the SAME confirmed bottomHist/topHist/latestIso (cached in
// evidenceHistoryBySignal) so the 47 confirmed bars don't flicker, only the new live one changes.
function evidenceStripSvg(bottomHist, topHist, latestIso, stepMinutes, liveTone, liveIso, key, bottomRawFire, topRawFire) {
  const n = Math.max(bottomHist.length, topHist.length, 1);
  const tones = Array.from({ length: n }, (_, i) => (bottomHist[i] ? "good" : topHist[i] ? "bad" : "neutral"));
  const times = evenlySpacedBarTimes(latestIso, n, stepMinutes);
  // 2026-09-01: bottomHist/topHist are now the "fill" window (see dashboard/server.py), which can
  // stay lit across several genuine re-fires -- rawFire[i] marks the bars where the signal's RAW
  // (un-filled) column actually fired, so toneStripSvg can force a visible break there. Optional:
  // undefined for callers that don't pass these (bottomRawFire?.[i] is undefined -> falsy).
  const rawFire = Array.from({ length: n }, (_, i) => !!(bottomRawFire?.[i] || topRawFire?.[i]));
  if (liveTone) { tones.push(liveTone); times.push(liveIso || ""); rawFire.push(false); }
  return toneStripSvg(tones, times, !!liveTone, !!liveTone && liveTone !== "neutral", key, rawFire);
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
// single past tone can't reconstruct -- "웜업 중", or liq_direction's 강한/약한 percentile-strength
// qualifier, which isn't stored per history bar, only tone is).
const STRIP_BAR_LABEL_BY_TONE = {
  v_rebound: { good: "급등", bad: "급락", flat: "미반등", neutral: "신호 없음" },
  liq_pressure: { good: "롱압박↑", bad: "숏압박↑", neutral: "안정" },
  liq_cascade: { good: "안정", warn: "주의", bad: "위험" },
  liq_direction: { good: "상승압력", bad: "하락압력", neutral: "중립" },
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
function lastSegmentRangeLabel(tones, times, key, timeFmtKind, rawFire) {
  const list = Array.isArray(tones) ? tones : [];
  const timeList = Array.isArray(times) ? times : [];
  const fireList = Array.isArray(rawFire) ? rawFire : [];
  const n = list.length;
  if (n === 0) return "-";
  const lastTone = list[n - 1] || "neutral";
  let start = n - 1;
  // 2026-09-01: also stop at a rawFire boundary (mirrors toneStripSvg's own segment-merge guard)
  // -- once `start` itself is a genuine re-fire bar, that's where its segment begins, so the walk
  // must not continue past it even if the tone on both sides matches.
  while (start > 0 && list[start - 1] === lastTone && !fireList[start]) start--;
  const fmt = stripTimeFmtByKind(timeFmtKind);
  const barLabel = (STRIP_BAR_LABEL_BY_TONE[key] || {})[lastTone] || "";
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
  const barLabel = key && tone ? (STRIP_BAR_LABEL_BY_TONE[key] || {})[tone] : null;
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

// Full-detail Korean explanations for the 6 model-internal indicators (formula + live threshold +
// what it means for a trader) -- shown only when the user clicks "자세히" next to each tile, so
// the default compact view stays uncluttered. Sourced from microstructure_scanner.py /
// tail_risk_interceptor.py verbatim, not re-derived.
// Always-visible "지금 이게 무슨 뜻인지" line, indexed by the exact subText string each
// indicator currently shows -- no click required (2026-08-24 사용자 요청: 발동되면 의미를 바로
// 볼 수 있게). The deeper formula/기준 stays behind "자세히" in MODEL_INDICATOR_DETAIL below.
const MODEL_INDICATOR_MEANING = {
  v_rebound: {
    "웜업 중": "가격 데이터를 충분히 모으는 중이에요 — 잠시 후 값이 나와요.",
    "신호 없음": "방금 마감된 봉의 지표 일부가 아직 계산되지 않아 채점을 건너뛰었어요 — 드문 경우이고, 다음 봉에서 정상으로 돌아와요.",
    "급등": "방금 마감된 5분봉을 TabPFN 모델이 채점한 결과 '여기가 바닥이고 진짜 반등(V자반등)이 온다'는 쪽이에요 — 앞으로 30분 안에 종가가 1.5×ATR 이상 오르고, 60분 전체로 봐도 정점 대비 20% 이하만 반납할 거라는 판정이에요. 이 표시는 **목표(1.5×ATR)에 닿거나 60분이 지날 때까지 유지**되고, 그 뒤 '미반등'으로 내려갑니다(칩의 '○분 전' 숫자가 경과 시간이에요). 자세한 계산 방식은 '자세히'를 확인하세요.",
    "급락": "방금 마감된 5분봉을 TabPFN 모델이 채점한 결과 '여기가 천장이고 진짜 반전(V자반등)이 온다'는 쪽이에요 — 앞으로 30분 안에 종가가 1.5×ATR 이상 내리고, 60분 전체로 봐도 정점 대비 20% 이하만 반납할 거라는 판정이에요. 이 표시는 **목표(1.5×ATR)에 닿거나 60분이 지날 때까지 유지**되고, 그 뒤 '미반등'으로 내려갑니다(칩의 '○분 전' 숫자가 경과 시간이에요). 자세한 계산 방식은 '자세히'를 확인하세요.",
    "미반등": "지금 V자반등의 근거가 없다는 뜻이에요 — 직전에 뜬 급등/급락 신호도 이미 목표(1.5×ATR)에 닿았거나 60분이 지나 내려간 상태입니다. 대부분의 봉이 여기 해당하는 **평상시 상태**예요(급등/급락은 전체 봉의 약 5%뿐). '반대방향으로 움직인다'는 뜻이 아니라 '지금은 반등/반전을 말할 근거가 없다'는 뜻이에요. 2026-09-01 매 봉 채점으로 바뀌기 전에는 트리거가 발동한 봉에서만 나오던 판정이라 지금보다 훨씬 드물었어요.",
  },
  liq_pressure: {
    "안정": "현물-선물 가격차(베이시스)가 평소 범위 안이라, 어느 한쪽이 특별히 강제청산 압박을 더 받을 조짐은 안 보여요.",
    "숏압박↑": "베이시스가 콘탱고(선물이 현물보다 비쌈) 쪽 극단이에요 — 실측상 이런 국면 이후 1~4시간 숏 강제청산액이 늘고 롱 청산액은 줄어드는 경향이 있었어요(약 1개월치 탐색적 관측). 가격이 오른다는 뜻은 아니고, '숏 쪽이 청산 압박을 더 받을 수 있다'는 리스크 정보예요.",
    "롱압박↑": "베이시스가 백워데이션(선물이 현물보다 쌈) 쪽 극단이에요 — 실측상 이런 국면 이후 1~4시간 롱 강제청산액이 늘고 숏 청산액은 줄어드는 경향이 있었어요(약 1개월치 탐색적 관측). 가격이 내린다는 뜻은 아니고, '롱 쪽이 청산 압박을 더 받을 수 있다'는 리스크 정보예요.",
  },
  liq_cascade: {
    "안정": "지금 진행 중인 청산 캐스케이드가 없어요 — 청산 흐름이 평소 수준이에요.",
    "주의": "한쪽 청산량이 평소보다 급증했지만 아직 본격적인 캐스케이드로 번지진 않았어요.",
    "위험": "한쪽 포지션들이 연쇄적으로 강제청산되며 캐스케이드가 실제로 진행 중이에요 — 그 방향으로 가격이 더 튈 수 있어요.",
  },
  liq_direction: {
    "상승압력": "최근 강제청산이 롱(매수) 쪽에 몰려 있어요 — 투매가 소진되며 반등할 수 있다는 컨트래리언 해석이에요. 실제로 오를지는 검증되지 않았어요.",
    "하락압력": "최근 강제청산이 숏(매도) 쪽에 몰려 있어요 — 숏스퀴즈가 소진되며 눌릴 수 있다는 컨트래리언 해석이에요. 실제로 내릴지는 검증되지 않았어요.",
    "중립": "롱/숏 청산이 어느 한쪽으로 뚜렷하게 쏠려 있지 않아요.",
  },
  whale: {
    "롱 진입": "큰 금액 단위 거래가 최근 5분간 매수 쪽으로 쏠렸어요 — 개인 소액 매매와는 구분된 흐름이에요.",
    "숏 진입": "큰 금액 단위 거래가 최근 5분간 매도 쪽으로 쏠렸어요 — 개인 소액 매매와는 구분된 흐름이에요.",
    "중립": "큰손 거래 방향이 뚜렷하지 않아요.",
  },
  retail_flow: {
    "롱 진입": "소액 단위(리테일) 체결이 최근 5분간 매수 쪽으로 쏠렸어요 — 큰손 흐름과는 별도 정보예요.",
    "숏 진입": "소액 단위(리테일) 체결이 최근 5분간 매도 쪽으로 쏠렸어요 — 큰손 흐름과는 별도 정보예요.",
    "중립": "리테일 체결 방향이 뚜렷하지 않아요.",
  },
};

const MODEL_INDICATOR_DETAIL = {
  v_rebound: "[계산] **매 5분봉마다** 22개 캔들/오더플로우/모멘텀 피쳐(Tier0)+RSI를 계산해 바닥쪽·천장쪽 양방향으로 TabPFN(사전학습된 트랜스포머가 in-context로 추론하는 표형 파운데이션 모델 — 데이터셋별 재학습이 없음)에 입력하고, 둘 중 확률이 높은 쪽을 그 봉의 판정으로 씁니다. 학습 컨텍스트는 전체 봉 TRAIN 182,969건 중 무작위 18,000건에 고정(자연 라벨비율 14.6% 그대로 보존·재균형 안 함, 라이브에서도 매번 이 컨텍스트를 그대로 재사용, 최신 데이터로 자동 갱신되지 않음).\n" +
    "[배지 유지 규칙] 급등/급락 배지는 **목표(1.5×ATR 도달) 또는 60분 경과 중 먼저 오는 쪽까지 유지**됩니다 — 다른 증거신호 칩들과 같은 방식입니다. 매 봉 채점으로 바꾼 직후에는 배지가 현재 봉만 반영해 대부분 5분만 떴다 사라졌는데(사건당 평균 1.2봉), 놓치기 쉬워서 2026-09-01 지속성을 넣었습니다. **아래 막대 게이지(히스토리)도 같은 규칙으로 칠해집니다** — 신호가 뜬 봉부터 목표 도달 또는 60분 경과까지가 한 덩어리로 이어집니다(다른 증거신호 칩들과 동일). 구간이 겹치면 나중 신호가 덮어씁니다.\n" +
    "[2026-09-01 재설계: 트리거 게이트 제거] 그전에는 9개 트리거(liquidity_sweep/taker_delta_z_climax/short_term_return_z/orthogonal_combo/smt_divergence/fib_extension_exhaustion/demarker_extreme/kalman_deviation_meanrev/local_extreme) 중 하나라도 발동한 봉만 채점했습니다. 그런데 그중 호출량의 73~76%를 공급하던 local_extreme은 정의상 '앞뒤 30분 안에서 이 봉이 최저/최고'라, 라벨이 요구하는 선행조건(반등 전까지 더 내려가지 않았을 것)을 **100% 만족하는 봉만** 골라 올리고 있었습니다 — 트리거·자산과 무관하게 라벨 발생률을 4.2~4.8배 부풀리는 기계적 얽힘이고, 모델은 그 공짜 크레딧을 성능으로 계상해왔습니다. 라이브에서 미래를 훔쳐본 건 아니지만(인과성은 정상) 성능 수치는 과대평가였습니다. 게다가 local_extreme은 30분이 지나야 확정되므로 '신호가 갑자기 과거 기록과 함께 나타나는' 표시 문제와 경제성 백테스트의 비현실적 진입시점(+9.28bp→실제로는 +4.75bp)의 원인이기도 했습니다. 그래서 게이트를 없애고 매 봉을 채점하도록 **전면 재학습**했습니다(게이트만 없애고 기존 모델을 쓰면 AUC 0.53으로 붕괴 — 실측 확인).\n" +
    "[기준] **확률≥60%**면 '반등 콜'(이후 30분 내 종가로 ATR(직전 기준) 1.5배 이상 반등 AND 60분 전체에서 정점 대비 20% 이하만 반납), 미만이면 '미반등 콜'(반등 근거 없음 — 반대방향으로 뚜렷하게 움직인다는 뜻은 아닙니다). 이 둘 사이(애매한 경우)는 라벨 자체가 없어 **학습에서 통째로 제외** — 라벨 정의(giveback 방식) 자체는 재설계 이후에도 안 바뀌었습니다. 매 봉이 채점되므로 대부분의 봉은 '미반등'입니다(60% 기준에서 급등/급락은 봉의 약 4.6~5.0%).\n" +
    "[배지 표시: 반등 콜만 급등/급락, 미반등 콜은 별도 '미반등'] 2026-08-31 두 차례 정정했습니다. 처음엔 '반등'/'반락'을 콜 이름 그대로 노출해서 상승 트리거 후 반등 콜처럼 실제로는 하락이 예상되는데도 '반등'이라고 표시되며 빨간색이 뜨는 경우가 있었습니다(사용자 지적) — 그래서 반등/미반등 콜을 트리거 방향과 조합해 항상 급등(초록)/급락(빨강) 중 하나로 바꿨습니다. 그런데 미반등 콜은 '반등 시도 자체가 없었다'는 뜻일 뿐 반대방향으로 결정적으로 움직였다는 근거가 아닌데도 급등/급락이라는 강한 단어를 그대로 썼던 게 다시 지적받아(사용자 지적), 지금은 **진짜 반등(V자반등) 콜만** 트리거 방향과 조합해 급등(하락 트리거 후 반등, 초록)/급락(상승 트리거 후 반등, 빨강)으로 표시하고, **미반등 콜은 트리거 방향과 무관하게 항상 '미반등'**(회색, 중립 취급)으로 따로 표시합니다 — 활동-스트립을 마우스오버했을 때 나오는 막대별 라벨도 동일한 기준(백엔드 tone: good/bad/flat/neutral)으로 구분됩니다. '반락 콜'이라는 예전 이름 자체도 마치 반대방향으로 결정적으로 움직였다는 뜻처럼 들려 실제 정의와 어긋나 '미반등 콜'로 정정했습니다.\n" +
    "[의미] 매 봉 채점판의 VAL AUC는 **0.6942**(TabPFN 3시드, std 0.0023)입니다. 게이트가 있던 시절의 헤드라인(VAL 0.829 / OOS 0.813 / HOLDOUT 0.847)보다 낮아 보이지만, **그 수치들이 위에서 설명한 공짜 크레딧을 포함한 값**이었습니다 — 같은 모델을 얽힘 조건으로 층화해 다시 재면 내부 AUC가 0.66~0.69로 내려앉고, 매 봉 재학습판이 독립적으로 그 대역에 수렴합니다. 라벨 정의가 달라졌으므로 두 숫자의 직접 비교는 성립하지 않습니다(다른 난이도의 문제를 푼 두 모델의 AUC를 나란히 놓고 우열을 판정하면 안 된다는 이 저장소의 규칙). **0.6942는 같은 문제를 정직하게 푼 점수**로 읽으세요. 전체 방법론: docs/homer/README.md + docs/homer/v_rebound_open_issues_20260901.md.\n" +
    "[기준선을 50%→60%로 올린 이유, 2026-09-01] 결과 라벨이 붙은 봉을 대상으로 트레일링 스톱 경제성을 재면 50% 기준은 통과하지 못했고(방향 뒤집기 대조군이 정방향을 이김), 55%부터 역전돼 60%가 가장 깨끗했습니다. precision도 60%가 가장 높습니다(0.713/0.683). 빈도 손실도 없습니다 — 화면 기준 하루 11~12건, 신호 간격 중앙값 약 1시간, 신호 없는 날 0%(50%일 때는 하루 18건으로 오히려 과했습니다). **그래서 60%는 '분류 운영점'으로는 근거가 있습니다.**\n" +
    "[⚠️매매 신호로 쓰지 마세요 — 2026-09-01 확인] 이 칩의 경제성 수치는 전부 **결과 라벨이 붙은 봉(전체의 약 53%) 기준**입니다. 그런데 화면은 라벨 유무와 무관하게 **모든 봉**을 채점하므로, 뜨는 신호의 상당수는 결과가 '급반등'도 '횡보'도 아니었던 애매한 봉에 앉습니다. 그 전체 모집단에서 방향뒤집기 대조군을 돌려보니(ARM≥1.0 80셀 전수): VAL은 정방향 28 / 뒤집기 0으로 방향이 맞았지만, **OOS에서는 정방향 21 / 뒤집기 31로 역전**됩니다(아티팩트 무영향 구간만 보면 4 대 16). 즉 **화면에 뜨는 콜을 그대로 매매로 옮기면 최근 구간에서는 오히려 반대가 나았습니다.**\n" +
    "[유의] 위 결과가 분류 성능을 부정하는 건 아닙니다 — OOS AUC 0.7051로 '여기가 바닥/천장이다'를 가려내는 능력 자체는 유지됩니다. 무너진 건 그 판정을 트레일링 스톱 매매로 옮겼을 때의 방향성입니다. 봇 내부 상태가 아니라 대시보드 서버가 별도로 계산하며, 실제 매매 결정(trading_bot.py)에는 연결되어 있지 않습니다. **재량 판단의 참고 재료로만 쓰세요.**",
  liq_pressure: "[계산] basis_raw = (선물 종가 − 현물 종가) / 현물 종가 (ETHUSDT, fapi.binance.com 선물 vs api.binance.com 현물). basis_z48 = basis_raw를 직전 48봉(4시간) 평균·표준편차로 정규화한 z값.\n" +
    "[기준] |z| 2.0 이상 위험 · 1.0~2.0 주의 · 1.0 미만 안정. 양수 극단=콘탱고(숏압박↑ 힌트), 음수 극단=백워데이션(롱압박↑ 힌트).\n" +
    "[의미] 2026-08-20에 '베이시스가 방향(다음 봉이 오를지 내릴지)을 예측하는가'로 먼저 테스트했으나 REJECTED(구간마다 부호가 뒤집힘) — 문헌(Schmeling/Schrimpf/Todorov 'Crypto Carry' BIS WP1087; He/Manela/Ross/von Wachter arXiv:2212.06888)이 원래 예측한 축은 방향이 아니라 '미래 변동성'과 '청산 크라우딩'이었습니다. 2026-08-27 그 방향으로 재검정: 변동성 예측은 이것도 부호가 안정적이지 않아 REJECTED급이었지만, 청산 크라우딩(어느 쪽이 강제청산 더 받는가)은 실제 청산 데이터(tail_risk.duckdb)로 확인한 결과 문헌과 부호까지 정확히 일치했습니다 — 베이시스 극단(양수) 이후 1h/4h 숏청산액이 유의하게 늘고(z=+3.9~+4.4) 롱청산액은 유의하게 줄었습니다(z=-4.3~-5.7), 음수 극단은 반대.\n" +
    "[유의] 이 청산크라우딩 검증은 **약 1개월치 탐색적 표본**입니다(청산 데이터의 신뢰 가능 구간이 2026-07-18부터 시작) — 이 저장소가 다른 모든 신호에 쓰는 VAL/OOS 3-split 재현성 검증은 아직 못 거쳤습니다. 표본이 더 쌓이면 정식 재검증 예정. 가격이 오르내린다는 뜻이 아니라 '어느 쪽 포지션이 청산 압박을 더 받을 가능성'만 알려주는 리스크 정보입니다. 봇 내부 상태가 아니라 대시보드 서버가 매 사이클 spot/perp klines를 직접 fetch해 계산합니다 — 아직 실제 매매 결정에는 연결되지 않았습니다.",
  liq_cascade: "[계산] 최근 1분 롱/숏 청산 금액을 각각 30분 과거 평균·표준편차로 정규화한 Z값 중 큰 쪽(z_peak). z_peak이 임계값(3.5)을 넘으면 그 순간부터 '캐스케이드 진행중'으로 전환되고, 이후 시간이 지나며 지수적으로 감쇠합니다(현재 파라미터 기준 반감기 약 2~3분). 감쇠된 에너지가 임계값의 35% 아래로 내려가면 캐스케이드가 종료된 것으로 판정합니다. 2026-08-27부터(ETH만) Z값 조건에 더해 그 쪽 청산 금액이 최소 $10,000 이상이어야 전환됩니다 — 청산이 성긴(대부분 분(分)이 $0) 데이터라 조용한 구간이 이어지면 평균·표준편차 자체가 거의 0으로 붕괴해, 그 직후엔 평범한 청산 하나에도 Z값만으로는 오검출되던 문제를 막기 위함입니다.\n" +
    "[기준] 캐스케이드 진행중이면 위험 · 아직 진행중은 아니지만 Z≥2.0(청산 급증)이면 주의 · 그 외 안정.\n" +
    "[의미] 예측이 아니라 '지금 이 순간 실제로 캐스케이드가 벌어지고 있는가'를 가공 없이 그대로 보여주는 원시 상태값입니다. 방향(롱/숏)이나 에너지 잔량(감쇠율) 같은 세부 숫자는 화면에 안 나옵니다 — 안정/주의/위험 배지만으로 충분하다는 판단(사용자 요청으로 정리)이라, 이 지표는 '지금 캐스케이드가 있는가/없는가'만 한눈에 보는 용도입니다.",
  liq_direction: "[계산] liq_net_z_12 = (최근 12분 롱청산 합 − 숏청산 합) / (최근 2일 총청산 롤링평균 + 1% 여유값). 양수면 롱청산 우세, 음수면 숏청산 우세.\n" +
    "[갱신 주기] 원천 데이터(tail_risk_1m)가 1분마다 1행씩 쌓이고 서버도 60초 캐시를 걸어서, 이 값은 최소 1분에 한 번만 바뀝니다 — 수급 흐름처럼 몇 초 단위로 바뀌는 지표가 아닙니다. 그래프도 1분×48칸, 즉 최근 48분을 보여줍니다.\n" +
    "[기준] 부호로 방향(상승압력/하락압력), 최근 이력 대비 백분위(상하위 10% 안이면 '강한', 25~75% 사이면 '약한')로 세기를 표시.\n" +
    "[의미] 컨트래리언 해석입니다 — 롱청산이 몰리면(강제매도 소진) 상승압력, 숏청산이 몰리면(숏스퀴즈 소진) 하락압력으로 읽습니다. **2026-08-25 정식 IC 검증(37일, n>10,200)**: 5분·15분 지평은 forward-return과의 상관이 통계적으로 유의(순열검정 z=+2.96/+2.50, 전반·후반 구간 모두 같은 부호), 1시간 지평은 근소 미달(z=+1.91)이지만 상관 크기 자체는 문턱을 넘고 전반/후반 부호·크기도 일관돼(+0.041/+0.035) 표본부족(정식 문턱 56일의 66%) 때문으로 보입니다 — 방향 정보 자체는 탄탄합니다. 다만 **같은 원천 데이터로 스윕과 결합해 실제 손익을 검정한 결과(§14, 08-25)는 8개 지평 전부 비용 차감 후 순손실**이었습니다(15분 -9.19bp~2시간 -5.00bp, taker 10bp 기준) — 통계적으로 진짜인 정보와 수수료를 넘기고 이익이 나는지는 별개 질문입니다. 방향 부호는 참고할 만하지만 이 신호 하나만으로 매매를 결정할 만큼 이익을 낸다는 근거는 없습니다 — 수급 흐름과 마찬가지로 재량 판단의 한 재료로만 쓰세요.\n" +
    "[유의] 09-15 정식 게이트(56일치 데이터, §13/§14 포함)가 이 조기 IC 결과를 대체합니다 — 지금 수치는 37일치 조기 계산입니다.",
  whale: "[계산] 최근 5분간 체결을 건당 금액 기준으로 큰손/소액으로 나눠, 큰손 거래만 (매수금액−매도금액)/(매수금액+매도금액)으로 계산 (-1~+1).\n" +
    "[기준] +0.2 이상 강하게 매수유입 · +0.05~0.2 매수유입 · -0.05~-0.2 매도유입 · -0.2 이하 강하게 매도.\n" +
    "[의미] 큰 금액 단위 거래(개인 소액 매매와 구분)가 최근 5분간 실제로 어느 방향으로 체결됐는지를 보여줍니다. '포지션'이 아니라 '최근 흐름'이라는 점에 유의하세요.",
  retail_flow: "[계산] 위 '수급 흐름'과 같은 함수·같은 5분창에서 나온 리테일(소액) 쪽 짝 — (매수금액−매도금액)/(매수금액+매도금액), 소액 체결만 (-1~+1).\n" +
    "[기준] +0.2 이상 강하게 매수유입 · +0.05~0.2 매수유입 · -0.05~-0.2 매도유입 · -0.2 이하 강하게 매도.\n" +
    "[의미] **2026-08-25 검증**: 1~15분 초단기 지평에서 통계적으로 유의한 방향 정보가 있습니다(5개 지평 전부 유의, 노이즈로 설명되는 수준을 훨씬 넘음). 다만 시장가로 그대로 매매하면 비용(10bp)이 총이익보다 커서 45개 지평×조합 전부 순손실이었습니다(4차례 재검증 포함, min_periods 계산결함까지 잡아낸 뒤 재확인) — 방향 정보 자체는 진짜지만 수수료를 넘길 만큼 크지는 않다는 뜻으로, 재량 판단 참고용입니다. 수급 흐름(고래)과는 상관 0.36 정도로 상당히 다른 정보라 같이 보면 유용합니다 — 둘이 같은 방향을 가리키면 좀 더 무게를 둘 근거, 엇갈리면 큰손/리테일이 다르게 움직이고 있다는 뜻입니다.",
};

// 2026-08-30 (user request): "학습 horizon을 배지로" -- each signal's own validated forward-
// looking prediction/detection window, shown as a small badge next to its name (see
// horizonBadgeHtml() below, used by both renderModelIndicatorList and renderEvidenceSignals).
// Covers both model-indicator keys (MODEL_CHIP_IDS below) and evidence-signal keys
// (EVIDENCE_STRIP_CHIP_IDS further down) in one lookup since neither namespace collides.
// "상태" (not a number) marks signals whose live formula is a continuous current-state gauge with
// no fixed forward horizon baked in -- forcing a number onto those would overstate what they
// actually claim; each entry's title cites the specific research this is grounded in (verified
// against each script's own docstring/detail text above and this repo's evidence-signal scorecard
// methodology, not guessed from the signal's name -- e.g. "15분 급변"/short_term_return_z names
// its INPUT lookback, not its evaluation horizon, which is 1시간 like its 6 scorecard siblings).
const SIGNAL_HORIZON = {
  // -- model indicators --
  v_rebound: { text: "60분", title: "매 5분봉을 채점해 이후 60분(12봉) 안 실제 가격방향(급등/급락)을 예측 -- 확률>=60%인 '반등 콜'은 30분 내 종가로 1.5xATR 반등 후 60분 전체에서 정점 대비 20% 이하만 반납을 요구, 바닥쪽/천장쪽 중 확률 높은 방향과 조합해 급등/급락으로 표시(2026-09-01 트리거 게이트 제거 + 기준선 50%->60% 상향)" },
  liq_pressure: { text: "1시간·4시간", title: "베이시스 극단 이후 1시간·4시간 시점의 강제청산 물량(방향)을 예측 -- 약 1개월 탐색적 표본, 이 저장소 표준 VAL/OOS 3-split 재현 전" },
  liq_direction: { text: "상태", title: "고정 예측 시간창 없이 매분 갱신되는 현재 청산 방향압력 -- 5·15분 지평 IC는 유의했으나(탐색적), 손익 결합 검정(8개 지평)은 전부 순손실" },
  liq_cascade: { text: "상태", title: "예측이 아니라 '지금 캐스케이드가 진행 중인가'를 보여주는 현재 상태값(반감기 약 2~3분으로 감쇠)" },
  whale: { text: "상태", title: "최근 5분간 큰손 체결 순유입 방향 -- 고정 예측 시간창 없는 현재 흐름 지표(방향-IC 검정 4개 지평 전부 무정보)" },
  retail_flow: { text: "상태", title: "최근 5분간 리테일 체결 순유입 방향 -- 1~15분 지평 방향-IC는 유의했으나 수수료 반영 손익은 전부 순손실, 고정 예측 시간창은 없음" },
  // -- evidence signals (모두 이 저장소 표준 스코어카드: 1시간/4시간/8시간 중 1시간이 대표 지평) --
  orthogonal_combo: { text: "2시간", title: "발동 조건 자체는 오실레이터(p_fast/p_slow) 이중극단+delta_z/funding_z 확인이지만, 신뢰도는 발동 시점 피쳐를 TabPFN에 넣어 '2시간 안 3.57xATR 이상 강하게 도달할 확률'로 평가(2026-08-31 교체, 이 저장소 분류·경제성 둘 다 역대 최고 성적)" },
  fib_extension_exhaustion: { text: "100분", title: "발동 조건 자체는 48봉 스윙 반대극값 기준 27.2~61.8% 확장(소진)이지만, 신뢰도는 발동 시점 피쳐를 TabPFN에 넣어 '100분 안 2.35xATR 도달(같은 구간 대형역행 없이) 확률'로 평가(2026-08-31 교체, 이 저장소 최초로 라벨에 최대역행폭(MAE) 상한 적용)" },
  smt_divergence: { text: "6시간", title: "발동 조건 자체는 ETH-BTC 48봉 스윙 교차자산 비확인이지만, 신뢰도는 발동 시점 피쳐를 TabPFN에 넣어 '6시간 안 4.2xATR 도달 확률'로 평가(2026-08-31 교체, 이 저장소 분류 AUC 역대 최고)" },
  short_term_return_z: { text: "1시간", title: "발동 조건 자체는 15분(3봉) 수익률 급변이지만, 신뢰도는 발동 시점 피쳐를 TabPFN에 넣어 '1시간 안 1.75xATR 도달 확률'로 평가" },
  taker_delta_z_climax: { text: "2시간", title: "발동 조건 자체는 이번 봉 체결 쏠림이지만, 신뢰도는 발동 시점 피쳐를 TabPFN에 넣어 '2시간 안 2.0xATR 도달 확률'로 평가(2026-08-30 교체)" },
  liquidity_sweep: { text: "2.5시간", title: "발동 조건 자체는 48봉 스윙 저/고점 스윕이지만, 신뢰도는 발동 시점 피쳐를 TabPFN에 넣어 '2.5시간 안 4.0xATR 도달 확률'로 평가(2026-08-30 표준방식 재학습)" },
  demarker_extreme: { text: "40분", title: "발동 조건 자체는 DeMarker(14) 오실레이터 극단(≥0.90/≤0.10)이지만, 신뢰도는 발동 시점 피쳐를 TabPFN에 넣어 '40분 안 0.70xATR 도달 확률'로 평가(2026-08-31 신규, 호메로스 후보풀, 이 저장소 분류 AUC 역대 최고)" },
  kalman_deviation_meanrev: { text: "1시간", title: "발동 조건 자체는 칼만필터 추세선 대비 이탈도(rolling 288봉 z-score) 극단(≥2.0/≤-2.0)이지만, 신뢰도는 발동 시점 피쳐를 TabPFN에 넣어 '1시간 안 2.5xATR 도달 확률'로 평가(2026-08-31 신규, 호메로스 후보풀)" },
};

function horizonBadgeHtml(key) {
  const h = SIGNAL_HORIZON[key];
  if (!h) return "";
  return ` <span class="horizon-badge" title="${escapeHtml(h.title)}">${escapeHtml(h.text)}</span>`;
}

// Snapshot tab "12신호 한눈에" overview: id lookup so the compact chip row (.signal-chip-row in
// index.html) can be updated from the same per-tick data as the full snapModelIndicatorList below.
const MODEL_CHIP_IDS = {
  v_rebound: "modelChipVRebound",
  liq_pressure: "modelChipBasisLiq",
  liq_cascade: "modelChipLiqCascade",
  liq_direction: "modelChipLiqDirection",
  whale: "modelChipWhale",
  retail_flow: "modelChipRetailFlow",
};
// whale/liq_direction/retail_flow/liq_pressure/v_rebound are directional (tone:
// good=롱 쪽/bad=숏 쪽/neutral=무신호); liq_cascade is risk-level (tone: good=안정/warn=주의/
// bad=위험). Both families reuse the same colors, so a bare red chip is ambiguous ("숏" vs
// "위험") -- 2026-08-24 사용자 리포트. Prefixing an explicit ▲/▼/– arrow on the directional chips
// only disambiguates by text, not just color.
//
// 2026-08-29 user request: liq_pressure/liq_sweep_trend/v_rebound moved INTO the directional
// family (were previously good=안정/warn=주의/bad=위험 risk-level, or event-triggered warn-only --
// see live_spot_perp_basis_signal_20260827.py / live_liquidation_cascade_sweep_trend_signal_
// 20260828.py / live_eth_sweep_v_rebound_signal_20260829.py for the backend tone-mapping change).
// The server now resolves each one's own direction field + call/pressure read into a single
// good/bad/neutral tone before this ever reaches app.js, so no client-side remapping needed here --
// only their DIRECTIONAL_MODEL_CHIP_KEYS membership (for the arrow) changes.
//
// 2026-08-30 user request: risk(꼬리 리스크)/whale_intent(고래 포지션) removed from MODEL_CHIP_IDS
// entirely (tested null / flagged non-independent, see classifyIndicators()'s own comment) -- no
// longer members of either family here.
const DIRECTIONAL_MODEL_CHIP_KEYS = new Set([
  "whale", "liq_direction", "retail_flow", "liq_pressure", "v_rebound",
]);

function renderModelIndicatorList(items, targetId = "snapModelIndicatorList") {
  // 2026-08-25: perf pass -- render() drives this on every SSE push (~2.5s), but the underlying
  // model_indicator_history only advances once per MODEL_INDICATOR_SAMPLE_SECONDS (300s server-
  // side), so most calls were rebuilding ~500 DOM nodes (9 rows x up to 48 sparkline rects each)
  // for identical output. Chip-element side effects below still run every call (cheap, and their
  // inputs -- it.tone/it.subText -- are also embedded in the html string, so if the string is
  // unchanged those writes are redundant-but-harmless); only the expensive setH() innerHTML
  // rebuild is skipped when nothing actually changed.
  const html = items.map((it) => {
    const chipId = MODEL_CHIP_IDS[it.key];
    const chip = chipId ? el(chipId) : null;
    if (chip) {
      chip.className = `signal-chip ${it.tone}`;
      const stateEl = chip.querySelector(".signal-chip-state");
      if (stateEl) {
        const arrow = DIRECTIONAL_MODEL_CHIP_KEYS.has(it.key)
          ? (it.tone === "good" ? "▲ " : it.tone === "bad" ? "▼ " : "– ")
          : "";
        stateEl.textContent = `${arrow}${it.subText || "-"}`;
      }
    }
    const derivedTag = it.derivedTag
      ? ` <span class="derived-tag" title="${escapeHtml(it.derivedTitle || "")}">${escapeHtml(it.derivedTag)}</span>`
      : "";
    const detailKey = `model:${it.key}`;
    const isOpen = detailOpenKeys.has(detailKey);
    const detailText = MODEL_INDICATOR_DETAIL[it.key] || "";
    const meaningText = (MODEL_INDICATOR_MEANING[it.key] || {})[it.subText] || "";
    const times = it.times || [];
    // 2026-08-31 user request: default caption shows the LAST segment's own range+label, not just
    // "지금 시간" -- see lastSegmentRangeLabel().
    const defaultRangeText = lastSegmentRangeLabel(it.history, times, it.key, "time");
    // 2026-08-31: optional `it.proba` (0-1) opts an item into the same inline probability meter
    // renderEvidenceSignals() uses (see .meter-col in styles.css) -- state text, then the meter bar,
    // stacked vertically ("천장 발동과 익절 사이" layout the user picked). Items with no proba concept
    // (수급 흐름/청산 캐스케이드/베이시스 청산압박/청산 방향압력, all categorical-only) keep the plain
    // .ops-health-status-badge pill, unchanged -- there's no probability to gauge for those.
    const metaHtml = it.proba != null
      ? `<div class="meter-col">
          <span class="meter-state ${it.tone}">${escapeHtml(it.subText || "-")}</span>
          <div class="meter-gauge">
            <span class="meter-track"><span class="meter-fill ${it.tone}" style="width:${clamp01(it.proba) * 100}%"></span></span>
            <span class="meter-pct">${Math.round(clamp01(it.proba) * 100)}%</span>
          </div>
        </div>`
      : `<span class="ops-health-status-badge">${escapeHtml(it.subText || "-")}</span>`;
    return `<article class="ops-health-row ${it.tone}">
      <span class="ops-health-dot" aria-hidden="true"></span>
      <div class="ops-health-info">
        <strong>${escapeHtml(it.label)}${horizonBadgeHtml(it.key)}${derivedTag}</strong>
        ${meaningText ? `<p class="signal-meaning">${escapeHtml(meaningText)}</p>` : ""}
        ${it.liveText ? `<p class="signal-meaning">${escapeHtml(it.liveText)}</p>` : ""}
        <div class="evidence-strip-wrap">
          ${toneStripSvg(it.history, times, false, false, it.key)}
          ${stripAxisHtml(times, "time")}
        </div>
        <div class="strip-time-row">
          <button type="button" class="detail-toggle" aria-expanded="${isOpen}" onclick="toggleSignalDetail(this, '${detailKey}')">${isOpen ? "접기 ▴" : "자세히 ▾"}</button>
          <span class="strip-time-now" data-fmt="time" data-default="${escapeHtml(defaultRangeText)}">${escapeHtml(defaultRangeText)}</span>
        </div>
        <div class="signal-detail${isOpen ? " open" : ""}">${escapeHtml(detailText)}</div>
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
  if (!map || map.error === "fetch_failed") {
    if (badge) { badge.className = "ops-badge bad"; badge.textContent = "연결 실패"; }
    setH("liquidationMapList", `<p class="muted" style="padding:16px;">청산맵 데이터를 불러오지 못했습니다.</p>`);
    return;
  }
  if (!map.warmed_up) {
    if (badge) { badge.className = "ops-badge neutral"; badge.textContent = "웜업 중"; }
    setH("liquidationMapList", `<p class="muted" style="padding:16px;">데이터 수집 중...</p>`);
    return;
  }
  if (badge) { badge.className = "ops-badge neutral hidden"; badge.textContent = ""; }

  const liveCurrentPrice = Number(latestLivePriceByAsset[activeSnapshotAsset] || map.current_price || 0);
  const liveRedistanced = (levels, side) => {
    if (!(liveCurrentPrice > 0)) return levels || [];
    return (levels || [])
      .filter((lv) => side === "support" ? lv.price < liveCurrentPrice : lv.price > liveCurrentPrice)
      .map((lv) => ({ ...lv, distance_pct: (lv.price - liveCurrentPrice) / liveCurrentPrice * 100 }));
  };
  const resistanceRows = liveRedistanced(map.resistance_levels, "resistance").slice().reverse()
    .map((lv, i, arr) => liquidationLevelRowHtml(lv, `저항${arr.length - i}`, "liq-resistance"));
  const supportRows = liveRedistanced(map.support_levels, "support")
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

// Display-only Korean name/description for the 6 evidence signals -- the underlying signal key
// (s.name from the server, matching scripts/live_evidence_signal_dashboard_20260823.py's
// SIGNAL_ORDER) is left untouched for traceability to the research docs; only the rendered text
// is translated here.
// 4가지 독립 시도(강제청산 베토/exit_head 피쳐/사이징 피쳐/투표식 진입 공식) 전부 always_long/
// always_short 벤치마크에 패배 -- docs/experiments/eth_evidence_signal_top6_confluence_standalone_backtest_20260814.md
const EVIDENCE_SIGNAL_DISCLAIMER = "이 신호를 자동매매 트리거로 직접 연결하는 시도가 4가지(강제청산 베토 · exit_head 피쳐 · 사이징 피쳐 · 투표식 진입 공식) 전부 실패했습니다(always_long/always_short 벤치마크에 짐). 실제 반전이 맞아도 역사적으로 0.5~0.85% 더 불리한 방향으로 움직인 뒤에야 진짜 전환점이 왔습니다 — 정확한 바닥/천장을 찍어주는 신호가 아니라, 사람이 재량 판단할 때 참고하는 확률적 맥락으로만 쓰세요.";
// 2026-08-25 실측(VAL+OOS 48,853봉): 같은 쪽 신호가 동시에 몇 개 뜨는지(bottom_votes/top_votes)
// 자체가 검증된 신뢰도 축 -- votes>=N lift가 N에 대해 단조증가함을 확인(N>4는 미검증, 4로 캡).
// scripts/research_eth_evidence_signal_indicator_cooking_research_20260825.md 참고.
const VOTE_LIFT_BY_SIDE = {
  bottom: { 1: 1.81, 2: 2.10, 3: 2.32, 4: 2.72 },
  top: { 1: 1.58, 2: 1.85, 3: 1.89, 4: 2.07 },
};
function voteLiftNote(side, votes) {
  const capped = Math.min(Math.max(Math.round(votes), 1), 4);
  const lift = VOTE_LIFT_BY_SIDE[side][capped];
  return `실측: ${side === "bottom" ? "바닥" : "천장"} 신호 ${capped}개↑ 동시발동 구간 lift ${lift.toFixed(2)}배(무작위 대비) — 신호가 겹칠수록 신뢰도가 실제로 높아짐이 확인됨`;
}
// 2026-08-31 user request: "증거신호 제목 바로 아래에 있는 신호 설명은 모두 제거해줘. 증거신호에
// 있는 나머지 텍스트들 모두 정리 요약해서 줄여줘" -- desc 필드 삭제(제목 바로 아래 렌더링 자체를
// 없앰, renderEvidenceSignals 쪽도 같이 수정) + detail을 조건/신뢰도/검증/경제성 4줄 안팎으로
// 압축(원래 여러 문단이던 배경 설명·역사적 경위는 뺐고, 숫자·결론은 원문과 동일하게 보존 -- 전체
// 이력은 docs/homer/README.md와 각 신호의 memory 파일에 남아있음).
const EVIDENCE_SIGNAL_KO = {
  orthogonal_combo: {
    name: "복합 오실레이터 신호",
    detail: "[조건] p_fast·p_slow(스토캐스틱 백분위) 극단 + delta_z(체결쏠림) 또는 funding_z(펀딩비율, 바닥만) 극단 동시충족 — 서로 독립적인 두 정보축의 교차확인.\n" +
      "[신뢰도] 발동 시점 20피쳐(Tier0 23개 중 과적합 기여 3개 제외)를 TabPFN에 넣어 '2시간 안 3.57×ATR 도달확률' 산출.\n" +
      "[검증] VAL 0.684 / OOS 0.727 / HOLDOUT 0.725 — 이 저장소 분류성능 역대 최고.\n" +
      "[경제성] 트레일링스톱 VAL +9.36bp(승률91.5%) / OOS +15.13bp(승률96.0%) / 홀드아웃 +3.78bp(승률91.5%) — 96조합 중 73개 통과, 이 저장소 경제성 역대 최고.",
  },
  liquidity_sweep: {
    name: "유동성 스윕(저점·고점 사냥)",
    detail: "[조건] 직전 4시간 저점/고점을 살짝 뚫었다가 종가는 그 안으로 복귀(stop hunt 패턴).\n" +
      "[신뢰도] 발동 시점 23피쳐(Tier0+rsi)를 TabPFN에 넣어 '150분 안 4.0×ATR 도달확률' 산출('특화 감지기'의 'V자 급등락'과는 별개 모델).\n" +
      "[검증] VAL 0.659 / OOS 0.637 / HOLDOUT 0.661.\n" +
      "[경제성] 트레일링스톱 VAL +10.70bp / OOS +14.49bp / 홀드아웃 +1.97bp(승률67.7%) — 이 신호 최초로 3구간 전부 통과(단, 실거래 미배포 · 재량 참고용).",
  },
  short_term_return_z: {
    name: "단기(15분) 수익률 급변",
    detail: "[조건] 최근 15분 수익률이 하루평균 대비 ±2.5표준편차 이상.\n" +
      "[신뢰도] 발동 시점 23피쳐를 TabPFN에 넣어 '1시간 안 1.75×ATR 도달확률' 산출.\n" +
      "[검증] VAL 0.674 / OOS 0.649 / HOLDOUT 0.643(다수결 대비 +4.5~11%p) — 경제성 게이트 미검증, 자동매매 근거 아님.",
  },
  taker_delta_z_climax: {
    name: "체결 매수매도 쏠림 극단",
    detail: "[조건] 이번 봉 순매수 체결량이 하루평균 대비 ±2표준편차 이상.\n" +
      "[신뢰도] 발동 시점 23피쳐를 TabPFN에 넣어 '2시간 안 2.0×ATR 도달확률' 산출(v5: 연속발동 병합기준 15분→60분 확대).\n" +
      "[검증] VAL 0.633 / OOS 0.645 / HOLDOUT 0.667.\n" +
      "[경제성] 트레일링스톱 VAL+OOS +8.68bp / 홀드아웃 +2.17bp(승률64.7%) — 이 신호 최초 완주(단, 실거래 미배포 · 재량 참고용).",
  },
  smt_divergence: {
    name: "SMT 다이버전스(ETH·BTC 엇갈림)",
    detail: "[조건] ETH는 직전 48봉(4시간) 저점(고점) 갱신, BTC는 자기 48봉 스윙을 미갱신 — 상관자산 비확인(ICT SMT 다이버전스, 유동성 스윕과 형제신호).\n" +
      "[신뢰도] 발동 시점 23피쳐(Tier0 그대로 — ablation 결과 변동성레짐·세션타이밍 전부 진짜 기여로 확인돼 축소 안 함)를 TabPFN에 넣어 '6시간 안 4.2×ATR 도달확률' 산출.\n" +
      "[검증] VAL 0.661 / OOS 0.625 / HOLDOUT 0.682 — 이 저장소 분류성능 역대 최고.\n" +
      "[경제성] 트레일링스톱 VAL +7.00bp(승률72.4%) / OOS +6.18bp(승률69.6%) / 홀드아웃 +3.24bp(승률70.3%) — 96조합 중 71개 통과, 승률이 구간마다 거의 안 줄어드는 이 저장소 최고 안정성.",
  },
  // 2026-08-31: 호메로스 8번(마지막) 신호, TabPFN 메타라벨 교체. 기존 "실험적/표본n~190" 등급은
  // 5.5개월 좁은 창 기준 오기로 정정됨(전체이력 재측정시 bottom1078/top1072).
  fib_extension_exhaustion: {
    name: "피보나치 확장 소진",
    detail: "[조건] 직전 48봉 레그의 반대 극값 기준 27.2~61.8% 확장(소진) — 레그반대방향 되돌림에 베팅(liquidity_sweep·smt_divergence와 같은 스윙계열, 겹침은 가장 낮은 6.0~9.5%).\n" +
      "[신뢰도] 발동 시점 23피쳐(Tier0 그대로)를 TabPFN에 넣어 '100분 안 2.35×ATR 도달(같은 구간 최대역행 4.70×ATR 미만)' 확률 산출 — 이 신호부터 라벨에 최대역행폭(MAE) 상한을 처음 적용.\n" +
      "[검증] VAL 0.605 / OOS 0.620 / HOLDOUT 0.621. ablation상 변동성레짐 의존도가 이 저장소 최고(제거시 VAL 0.514로 붕괴) — 경제적으로 타당한 메커니즘.\n" +
      "[경제성] 트레일링스톱 VAL +15.15bp(승률93.2%) / OOS +3.00bp(승률87.6%) / 홀드아웃 +2.54bp(승률90.6%) — 단 승률은 exit구조(ARM0.5) 자체효과(무작위진입도 82~84%), bp 우위(무작위 대비 +5~18bp)가 진짜 근거.",
  },
  // 2026-08-31: 호메로스 '후보 풀' 트랙 신규 2종(기존 8개와 별개, docs/homer/README.md "후보 풀"
  // 절 참조) — 사전점검→HORIZON/GAP/K그리드→TabPFN확인→순열중요도→경제성게이트→룩어헤드감사→
  // 홀드아웃까지 전체 파이프라인 완료 후 배포.
  demarker_extreme: {
    name: "DeMarker 오실레이터 극단",
    detail: "[조건] DeMarker(14, 고/저가 기반 오실레이터) ≥0.90(과매수)/≤0.10(과매도).\n" +
      "[신뢰도] 발동 시점 24피쳐(Tier0 23개+dem)를 TabPFN에 넣어 '40분 안 0.70×ATR 도달확률' 산출.\n" +
      "[검증] VAL 0.753 / OOS 0.716 / HOLDOUT 0.746 — 이 저장소 분류성능 역대 최고(단, 순열중요도상 실제 드라이버는 bb_pctb이고 dem 자신은 단독으론 무의미 — 룩어헤드 감사로 계산버그 아님 확인).\n" +
      "[경제성] 트레일링스톱 VAL +12.14bp(승률70.7%) / OOS +20.20bp(승률80.0%, 이 저장소 OOS bp 역대 최고) / 홀드아웃 +11.53bp(승률77.9%) — 96개 조합 전부 통과(역대 최고), 홀드아웃 축소폭도 이 저장소 최소.",
  },
  kalman_deviation_meanrev: {
    name: "칼만필터 추세이탈 평균회귀",
    detail: "[조건] (종가-칼만필터 추세선)/추세선을 288봉 롤링 z-score, ≥2.0(과열)/≤-2.0(과냉).\n" +
      "[신뢰도] 발동 시점 24피쳐(Tier0 23개+kalman_dev_z)를 TabPFN에 넣어 '1시간 안 2.5×ATR 도달확률' 산출.\n" +
      "[검증] VAL 0.657 / OOS 0.631 / HOLDOUT 0.628.\n" +
      "[경제성] 트레일링스톱 VAL +10.26bp(승률71.2%) / OOS +11.00bp(승률71.0%) / 홀드아웃 +5.80bp(승률71.8%) — 96개 조합 전부 통과.",
  },
};

// BTC 코인 페이지 전용 라벨(2026-09-02) -- ETH와 신호 정의(원시 트리거)는 같아도 그리드스크린이
// K/HORIZON/GAP을 BTC 전용으로 재선정했고(live_btc_evidence_signal_metalabel_20260902.py 참고)
// 검증·경제성 수치도 전부 다르므로, EVIDENCE_SIGNAL_KO의 ETH 수치를 그대로 보여주면 안 된다.
// 근거: docs/experiments/btc_evidence_signal_economics_gate_20260902.md(§3/§6/§7),
// scripts/live_btc_evidence_signal_shadow_runner_20260902.py(HOLDOUT_AUC).
// ⚠️ETH와 달리 대부분 경제성 게이트를 통과하지 못했다 — BTC ATR이 ETH보다 작아 비용/ATR
// 비율이 더 나쁘기 때문(자세한 원인은 섀도우 러너 docstring). 그래도 노출하는 건 이 대시보드의
// 증거신호 티어가 손익 주장이 아니라 정보성(IC) 표시이기 때문 — 각 항목에 실제 판정을 명시한다.
const BTC_EVIDENCE_SIGNAL_KO = {
  orthogonal_combo: {
    name: "복합 오실레이터 신호",
    detail: "[조건] 스토캐스틱 백분위 극단 + 체결쏠림(delta_z) 극단 동시충족 (BTC 그리드스크린 H=8/K=2.0/GAP=6).\n" +
      "[신뢰도] TRAIN hit률 42.71%(8봉 안 2.0×ATR 도달) · HOLDOUT AUC 0.5933.\n" +
      "[경제성] VAL +1.15 / OOS +10.95 / HOLDOUT +1.47bp(승률75.4%, n=338) — 2군(VAL 숏 구간 일부 약함) 생존, 무작위진입 귀무 백분위 100%.",
  },
  liquidity_sweep: {
    name: "유동성 스윕(저점·고점 사냥)",
    detail: "[조건] 직전 100분 저점/고점을 살짝 뚫었다가 되돌림 (BTC 그리드스크린 H=20/K=2.0/GAP=6).\n" +
      "[신뢰도] TRAIN hit률 10.22% · HOLDOUT AUC 0.5214(사실상 무작위).\n" +
      "[경제성] 트레일링스톱 게이트 0/96 전패, HOLDOUT 미도달 — 7종 중 유일하게 방향 정보 자체가 거의 없다.",
  },
  short_term_return_z: {
    name: "3봉 수익률 급변(z-score)",
    detail: "[조건] 3봉 수익률의 z-score 극단 (BTC 그리드스크린 H=6/K=2.0/GAP=12).\n" +
      "[신뢰도] TRAIN hit률 31.63% · HOLDOUT AUC 0.6443.\n" +
      "[경제성] VAL +4.94 / OOS +8.98 / HOLDOUT +0.06bp(승률66.3%, n=525) — 1군 생존이나 사실상 0(비용 상쇄 수준).",
  },
  taker_delta_climax: {
    name: "체결 쏠림 극단(taker delta)",
    detail: "[조건] 순공격적 매수/매도 체결량 z-score ≥2.0/≤-2.0 (BTC 그리드스크린 H=6/K=2.0/GAP=3, ETH는 K=2.5로 별개).\n" +
      "[신뢰도] TRAIN hit률 13.88% · HOLDOUT AUC 0.6276.\n" +
      "[경제성] VAL +0.64 / OOS +2.90 / HOLDOUT −0.94bp(n=1,172) — 방향뒤집기는 이기지만(갭+8.81) HOLDOUT 절대수익 미통과.",
  },
  fib_extension_exhaustion: {
    name: "피보나치 확장 소진",
    detail: "[조건] 48봉 추세 방향 대비 127.2~161.8% 확장구간 터치 (BTC 그리드스크린 H=10/K=2.75/GAP=6).\n" +
      "[신뢰도] TRAIN hit률 19.28% · HOLDOUT AUC 0.5657.\n" +
      "[경제성] VAL +0.56 / OOS +0.35로 게이트는 겨우 통과했으나 무작위진입 대조에서 탈락(롱 OOS 갭 −3.19) — HOLDOUT 미도달, 사실상 미검증.",
  },
  demarker_extreme: {
    name: "DeMarker 오실레이터 극단",
    detail: "[조건] DeMarker(14) 극단 (BTC 그리드스크린 H=8/K=0.70/GAP=6 — K가 낮아 발동은 잦으나 변별력은 낮음, TRAIN hit률 90.03%).\n" +
      "[신뢰도] HOLDOUT AUC 0.7286 — 7종 중 최고.\n" +
      "[경제성] VAL +6.46 / OOS +8.91 / HOLDOUT +3.25bp(승률81.3%, n=428) — 1군 생존, 7종 중 유일하게 여유 있는 양수.",
  },
  kalman_deviation_meanrev: {
    name: "칼만필터 추세이탈 평균회귀",
    detail: "[조건] (종가-칼만필터 추세선)/추세선 z-score ≥3.5/≤-3.5 (BTC 그리드스크린 H=10/K=3.5/GAP=6, ETH는 K=2.5로 별개).\n" +
      "[신뢰도] TRAIN hit률 14.25% · HOLDOUT AUC 0.6709.\n" +
      "[경제성] VAL +2.41 / OOS +6.17 / HOLDOUT −1.27bp(n=1,021) — 방향뒤집기는 이기지만 HOLDOUT 절대수익 미통과.",
  },
};

// Snapshot tab "13신호 한눈에" overview: id lookup so the compact chip row (.signal-chip-row in
// index.html) can be updated from the exact same fetch/loop as the full evidenceSignalList below
// -- one source of truth per tick, no separate poll or duplicated tone logic.
const EVIDENCE_STRIP_CHIP_IDS = {
  orthogonal_combo: "eviChipOrthogonal",
  liquidity_sweep: "eviChipSweep",
  short_term_return_z: "eviChipReturnZ",
  taker_delta_z_climax: "eviChipTakerDelta",
  // BTC 페이지(2026-09-02)는 같은 칩을 재사용하되 이름이 다르다(BTC 그리드스크린/frozen-context
  // 리포트가 "z" 없는 축약명을 씀 -- live_btc_evidence_signal_metalabel_20260902.py::
  // RAW_COLUMN_ALIAS 참고). smt_divergence는 BTC에 대응 신호가 없어 별칭 없음(그 칩은 BTC
  // 탭에서 계속 "-" 중립으로 남는다 -- resetEvidenceStripChips()가 탭 전환시 정리).
  taker_delta_climax: "eviChipTakerDelta",
  smt_divergence: "eviChipSmt",
  fib_extension_exhaustion: "eviChipFibExt",
  demarker_extreme: "eviChipDemarker",
  kalman_deviation_meanrev: "eviChipKalman",
};

function resetEvidenceStripChips() {
  Object.values(EVIDENCE_STRIP_CHIP_IDS).forEach((id) => {
    const chip = el(id);
    if (!chip) return;
    chip.className = "signal-chip neutral";
    const stateEl = chip.querySelector(".signal-chip-state");
    if (stateEl) stateEl.textContent = "-";
    // 2026-09-02: also clear the live-preview dot's own class (renderEvidenceSignalsProvisional
    // sets it separately from the chip's own className) -- without this, switching FROM the eth
    // tab (where a signal was live-firing) TO another coin leaves that dot stuck on "live-bottom"/
    // "live-top" since this asset's tab no longer runs the ETH-only provisional preview at all
    // (see that function's early-return guard) to ever clear it back.
    const dot = chip.querySelector(".signal-chip-live-dot");
    if (dot) dot.className = "signal-chip-live-dot";
  });
  const meaningEl = el("evidenceStripMeaning");
  if (meaningEl) { meaningEl.className = "evidence-strip-meaning hidden"; meaningEl.textContent = "-"; }
  // sessionVolAlertBadge/macroEventAlertBadge intentionally NOT touched here (2026-08-27) -- they
  // moved to their own independent poll (refreshSessionAlerts()), unrelated to evidence-signal
  // health, so an evidence-signal fetch failure must not blank out an otherwise-valid calendar alert.
}

// 2026-08-27: split off /api/evidence-signals (see api_session_alerts() docstring in server.py) --
// user reported the badges only updated on a manual page reload, root cause was inheriting
// evidence-signals' 5min client poll. This fetch is independent and fast (30s).
async function refreshSessionAlerts() {
  const now = Date.now();
  if (now - sessionAlertsLastFetchAt < SESSION_ALERTS_POLL_MS) return;
  sessionAlertsLastFetchAt = now;
  try {
    const res = await fetch(API_SESSION_ALERTS_URL, { cache: "no-store" });
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

// Informational only -- NOT a trading signal. See the disclaimer text baked into the Snapshot
// tab's markup and docs/experiments/eth_evidence_signal_top6_confluence_standalone_backtest_20260814.md:
// 4/4 independent attempts to wire these into automated decisions lost to simple benchmarks.
// A signal's bottom_fired/top_fired are independent sustain-window flags (see
// eth_dashboard_sustain_window_decay_correction_20260824), not mutually exclusive -- a signal can
// have both true at once if its bottom trigger and top trigger each fired within the last few bars
// (2026-08-27 real example: taker_delta_z_climax bottom_last=15:00, top_last=15:05, both still
// within their sustain windows at latest_bar=15:15). Every render site below used to do
// `bottom_fired ? ... : top_fired ? ... : ...`, which always picked bottom and silently dropped a
// simultaneous top -- the top vote a signal contributed to payload.top_votes then had no visible
// row anywhere ("천장↓ 1 배지에는 잡히는데 목록에는 안 보임"). This centralizes the 3-way (bottom /
// top / both) resolution so every call site agrees, with a distinct "혼재" state instead of one
// side winning silently.
function evidenceSideTone(s) {
  return s.bottom_fired && s.top_fired ? "warn" : s.bottom_fired ? "good" : s.top_fired ? "bad" : "neutral";
}
function evidenceSideLabel(s, { bottom, top, both, none }) {
  return s.bottom_fired && s.top_fired ? both : s.bottom_fired ? bottom : s.top_fired ? top : none;
}

function renderEvidenceSignals(payload) {
  const badge = el("snapshotEvidenceBadge");
  const stripBadge = el("evidenceStripBadge");
  // 2026-09-02: SOL/XRP/HYPE엔 증거신호 파이프라인 자체가 없다(ETH+BTC만 존재) -- 예전엔 코인탭과
  // 무관하게 항상 ETH 데이터를 보여줬다(사용자 신고: "비트코인 페이지에 이더리움 증거신호가
  // 나온다"). 다른 자산 탭에선 이전 자산의 값이 남아있지 않도록 명시적으로 "지원 안 함" 상태로 비운다.
  if (payload && payload.unsupported) {
    latestEvidenceSignals = null;
    if (badge) { badge.className = "ops-badge neutral"; badge.textContent = "미지원 자산"; }
    if (stripBadge) { stripBadge.className = "ops-badge neutral"; stripBadge.textContent = "미지원"; }
    setH("evidenceSignalList", `<div class="macro-calendar-empty">이 자산은 증거신호를 아직 지원하지 않습니다(ETH·BTC만 지원).</div>`);
    resetEvidenceStripChips();
    return;
  }
  if (!payload || payload.error) {
    latestEvidenceSignals = null;
    if (badge) { badge.className = "ops-badge bad"; badge.textContent = "EVIDENCE UNREACHABLE"; }
    if (stripBadge) { stripBadge.className = "ops-badge bad"; stripBadge.textContent = "UNREACHABLE"; }
    resetEvidenceStripChips();
    return;
  }
  if (!payload.warmed_up) {
    latestEvidenceSignals = null;
    if (badge) { badge.className = "ops-badge warn"; badge.textContent = "WARMING UP"; }
    if (stripBadge) { stripBadge.className = "ops-badge warn"; stripBadge.textContent = "WARMING UP"; }
    setH("evidenceSignalList", "");
    resetEvidenceStripChips();
    return;
  }
  if (badge) { badge.className = "ops-badge good"; badge.textContent = "EVIDENCE LIVE"; }
  const net = Number(payload.net_score || 0);
  const scoreBadge = el("evidenceNetScoreBadge");
  if (scoreBadge) {
    scoreBadge.className = `ops-badge ${net > 0 ? "good" : net < 0 ? "bad" : "neutral"}`;
    scoreBadge.textContent = `NET_SCORE ${net > 0 ? "+" : ""}${net} (바닥=↑힌트 ${payload.bottom_votes}표 · 천장=↓힌트 ${payload.top_votes}표)`;
  }
  if (stripBadge) {
    stripBadge.className = `ops-badge ${net > 0 ? "good" : net < 0 ? "bad" : "neutral"}`;
    stripBadge.textContent = `바닥↑ ${payload.bottom_votes || 0} · 천장↓ ${payload.top_votes || 0}`;
  }
  latestEvidenceSignals = payload;
  const signals = Array.isArray(payload.signals) ? payload.signals : [];
  const firedMeanings = [];
  setH("evidenceSignalList", signals.map((s) => {
    const tone = evidenceSideTone(s);
    // "바닥"/"천장" (not "BOTTOM"/"TOP") to match both the compact chip row's own state text
    // (EVIDENCE_STRIP_CHIP_IDS block below: s.bottom_fired ? "바닥" : ...) and this badge's new
    // fixed width (2026-08-27 user request) -- keeps text length closer to the other badge values
    // (안정/주의/미발동/상승압력 etc.) instead of the one outlier using an English word.
    const state = evidenceSideLabel(s, { bottom: "바닥 발동", top: "천장 발동", both: "혼재 발동", none: "미발동" });
    // taker_delta_z_climax/short_term_return_z만 갖는 필드(2026-08-30, 규칙기반 칩을 TabPFN
    // 메타라벨 모델로 교체) -- 발동 시 그 순간의 실시간 확률을 상태뱃지에 덧붙임. 발동 조건
    // 자체(bottom_fired/top_fired)는 기존 규칙 그대로(모델이 학습된 조건과 동일해야 하므로) -- 이
    // 확률은 그 발동을 "얼마나 신뢰할지"만 대체하는 것이지 "발동 여부"를 바꾸는 게 아님.
    const modelPctText = s.model_proba != null ? `${Math.round(s.model_proba * 100)}%` : null;
    // 2026-08-31 user request: the price level implied by this signal's own trained K*ATR% target
    // (server-computed from the fire bar's own entry/ATR, see _tp_price in
    // live_evidence_signal_metalabel_20260829.py) -- NOT this repo's separate trailing-stop
    // economics config, just the label's own touch target, shown next to the state/probability.
    const modelTpText = s.model_tp_price != null ? `익절 ${fmtNum(s.model_tp_price, 2)}` : null;
    // 2026-09-01 저ATR 경고. 발동봉 ATR이 이 신호 자신의 발동시 ATR 중앙값보다 낮을 때만 표시.
    // 왜: 저변동 구간에선 SL/ARM/Trail이 전부 ATR 배수로 줄어드는데 왕복비용 10bp는 고정이라,
    // 방향이 맞아도 수수료를 못 넘기는 비율이 커진다(실측 '방향정확도-수익승률' 격차 fib 23.0pp
    // /kalman 5.3pp/demarker 5.1pp). 표시 전용 -- 모델 확률도 발동 여부도 바꾸지 않는다.
    // 근거: docs/homer/evidence_signal_economics_tuning_protocol.md
    const lowAtrText = (s.model_low_atr === true && s.model_atr_bp != null)
      ? `저변동 ATR ${fmtNum(s.model_atr_bp, 1)}bp < 평소 ${fmtNum(s.model_atr_median_bp, 1)}bp`
      : null;
    // 2026-09-01 (user request): moved next to the horizon-badge on the title line (was stacked in
    // .meter-col on the right, see eth_dashboard_low_atr_warning_overflow_fix_20260901 memory for
    // that version's overflow saga) -- same pill shape as horizon-badge (.horizon-badge base class,
    // low-atr-badge modifier just recolors it warn/amber). ⚠️first cut showed only the current bp
    // value with the "평소"(median) baseline hidden in the tooltip -- user pointed out that drops
    // the actual context (is 22bp low for THIS signal or not depends entirely on its own median, and
    // that varies per signal). Visible text now carries the full comparison, same as the original
    // .meter-col text; only the plain-language explanation sentence stays in the tooltip.
    const lowAtrBadgeHtml = lowAtrText
      ? ` <span class="horizon-badge low-atr-badge" title="이 신호가 평소 발동하던 변동성보다 낮은 구간입니다. 방향이 맞아도 왕복 수수료(10bp)를 넘기지 못할 확률이 평소보다 큽니다.">⚠ ${escapeHtml(lowAtrText)}</span>`
      : "";
    // Evolution of this meta column, all 2026-08-31 (see eth_dashboard_evidence_signal_tp_price_
    // display_20260831 memory for the full history): joined into the status badge itself (broke its
    // fixed 64px pill sizing) -> split into badge + <small> below it (user: "not clean") -> a
    // probability ring ("옵션 B" of a 4-way mockup), unconditional (no model_proba = empty "0%"
    // reading, not a different fallback element, so fired/not-fired rows' columns still line up) ->
    // user decided the ring itself "생각보다 별로" -> an inline meter bar instead, first as a
    // horizontal row (gauge left, state+price stacked right) -> user asked for the gauge to sit
    // BETWEEN the state text and price instead, matching the original mockup's own "옵션 A" layout
    // exactly (see the vertical .meter-col template below: state, then gauge, then price). Class
    // names generalized from evidence-meter-* to meter-* in this same pass -- renderModelIndicatorList
    // now reuses this same component for "V자 급등락" (see its own it.proba handling), so it's no
    // longer evidence-signal-specific. metaTone priority (below) carries over unchanged.
    // metaTone: follows model_side (the side the probability was computed for) EXCEPT when the
    // row's raw tone is "warn" (혼재 -- bottom_fired AND top_fired at once), which always shows
    // amber/주의 color regardless of which single side model_side picked (model_side is otherwise
    // "bottom"/"top" whenever model_proba is present, even for a raw-혼재 row, since the backend's
    // side field is singular -- see compute_evidence_signal_metalabels()'s bottom-priority
    // tie-break -- so without this warn check first, 혼재 rows would silently render as plain
    // good/bad and never show as the mixed-signal warning they actually are).
    const metaTone = tone === "warn" ? "warn" : s.model_side === "bottom" ? "good" : s.model_side === "top" ? "bad" : tone;
    // 2026-09-02: BTC는 신호 정의는 ETH와 같아도 그리드스크린 K/HORIZON과 검증·경제성 수치가
    // 전부 다르므로 별도 라벨 사전을 쓴다 (BTC_EVIDENCE_SIGNAL_KO 선언부 주석 참고).
    const koDict = activeSnapshotAsset === "btc" ? BTC_EVIDENCE_SIGNAL_KO : EVIDENCE_SIGNAL_KO;
    const ko = koDict[s.name] || { name: s.name };
    const detailKey = `evidence:${s.name}`;
    const isOpen = detailOpenKeys.has(detailKey);
    const detailText = ko.detail ? `${ko.detail}\n\n[주의] ${EVIDENCE_SIGNAL_DISCLAIMER}` : "";
    // 발동 중일 때 바로 보이는 의미(클릭 불필요) -- 2026-08-24 사용자 요청, 2026-08-31 축약(제목
    // 아래 desc 줄 제거에 맞춰 이 문구도 desc 인용 없이 짧게 -- 상세 설명은 "자세히"에 있음).
    const meaningText = evidenceSideLabel(s, {
      bottom: "바닥 신호 발동 — 반등 확률↑(단, 반전 전 0.5~0.85% 불리한 움직임 흔함).",
      top: "천장 신호 발동 — 하락반전 확률↑(단, 반전 전 0.5~0.85% 불리한 움직임 흔함).",
      both: "바닥·천장 동시 발동(혼재) — 방향이 엇갈려 해석에 더 주의가 필요해요.",
      none: "",
    });
    if (meaningText) firedMeanings.push({ tone, text: meaningText });
    const stripChip = EVIDENCE_STRIP_CHIP_IDS[s.name] ? el(EVIDENCE_STRIP_CHIP_IDS[s.name]) : null;
    if (stripChip) {
      stripChip.className = `signal-chip ${tone}`;
      const stripStateEl = stripChip.querySelector(".signal-chip-state");
      if (stripStateEl) {
        const stripBase = evidenceSideLabel(s, { bottom: "바닥", top: "천장", both: "혼재", none: "-" });
        stripStateEl.textContent = modelPctText && stripBase !== "-" ? `${stripBase} ${modelPctText}` : stripBase;
      }
    }
    evidenceHistoryBySignal[s.name] = { bottom_history: s.bottom_history || [], top_history: s.top_history || [], bottom_raw_fire: s.bottom_raw_fire || [], top_raw_fire: s.top_raw_fire || [], latest_bar_utc: payload.latest_bar_utc };
    // eviTones/eviTimes mirror evidenceStripSvg's own internal tone derivation (bottom_history[i] ->
    // good, top_history[i] -> bad, else neutral) -- recomputed here (not returned by that function,
    // which keeps its plain-string contract for the provisional-refresh outerHTML-replace call site)
    // so the axis/default-caption below can share the exact same tone/time arrays it draws from.
    const eviN = Math.max((s.bottom_history || []).length, (s.top_history || []).length, 1);
    const eviTones = Array.from({ length: eviN }, (_, i) => (s.bottom_history?.[i] ? "good" : s.top_history?.[i] ? "bad" : "neutral"));
    const eviTimes = evenlySpacedBarTimes(payload.latest_bar_utc, eviN, 5);
    // 2026-09-01: same rawFire derivation as evidenceStripSvg, needed here too so the default
    // caption's segment boundary matches what the strip visually shows (see toneStripSvg).
    const eviRawFire = Array.from({ length: eviN }, (_, i) => !!(s.bottom_raw_fire?.[i] || s.top_raw_fire?.[i]));
    // 2026-08-31 user request: drop the old "바닥 {ts} · 천장 {ts}" last-fired caption -- this
    // range+label already tells you when the CURRENT segment started, which is what that text was
    // approximating anyway.
    const defaultRangeText = lastSegmentRangeLabel(eviTones, eviTimes, "evidence", "hm", eviRawFire);
    return `<article class="ops-health-row evidence-row ${tone}" data-signal="${s.name}">
      <span class="ops-health-dot" aria-hidden="true"></span>
      <div class="ops-health-info">
        <strong>${escapeHtml(ko.name)}${horizonBadgeHtml(s.name)}${lowAtrBadgeHtml}</strong>
        ${meaningText ? `<p class="signal-meaning">${escapeHtml(meaningText)}</p>` : ""}
        <div class="evidence-strip-wrap">
          ${evidenceStripSvg(s.bottom_history || [], s.top_history || [], payload.latest_bar_utc, 5, undefined, undefined, "evidence", s.bottom_raw_fire || [], s.top_raw_fire || [])}
          ${stripAxisHtml(eviTimes, "hm")}
          <small class="evidence-strip-caption">
            <span class="strip-time-now" data-fmt="hm" data-default="${escapeHtml(defaultRangeText)}">${escapeHtml(defaultRangeText)}</span>
          </small>
        </div>
        <button type="button" class="detail-toggle" aria-expanded="${isOpen}" onclick="toggleSignalDetail(this, '${detailKey}')">${isOpen ? "접기 ▴" : "자세히 ▾"}</button>
        <div class="signal-detail${isOpen ? " open" : ""}">${escapeHtml(detailText)}</div>
      </div>
      <div class="ops-health-meta">
        <div class="meter-col">
          <span class="meter-state ${metaTone}">${escapeHtml(state)}</span>
          <div class="meter-gauge">
            <span class="meter-track"><span class="meter-fill ${metaTone}" style="width:${clamp01((s.model_proba != null ? s.model_proba * 100 : 0) / 100) * 100}%"></span></span>
            <span class="meter-pct">${modelPctText || "0%"}</span>
          </div>
          ${modelTpText ? `<span class="meter-price">${escapeHtml(modelTpText)}</span>` : ""}
        </div>
      </div>
    </article>`;
  }).join(""));
  const meaningEl = el("evidenceStripMeaning");
  if (meaningEl) {
    if (!firedMeanings.length) {
      meaningEl.className = "evidence-strip-meaning hidden";
      meaningEl.textContent = "-";
    } else {
      const mixedTone = firedMeanings.some((f) => f.tone === "bad") && firedMeanings.some((f) => f.tone === "good");
      meaningEl.className = `evidence-strip-meaning ${mixedTone ? "warn" : firedMeanings[0].tone}`;
      // 혼재(바닥+천장 동시발동)가 아닐 때만 votes 기반 lift 근거를 붙임 -- bottom_votes/top_votes는
      // 이 side 전용 카운트라 firedMeanings.length(양쪽 합산)와는 다른 수치이므로 그걸 그대로 씀.
      const dominantSide = !mixedTone && firedMeanings[0].tone === "good" ? "bottom" : !mixedTone ? "top" : null;
      const dominantVotes = dominantSide === "bottom" ? payload.bottom_votes : dominantSide === "top" ? payload.top_votes : 0;
      const liftNote = dominantSide && dominantVotes >= 2 ? ` (${voteLiftNote(dominantSide, dominantVotes)})` : "";
      meaningEl.textContent = firedMeanings.length > 1
        ? `${firedMeanings.length}개 신호 동시 발동 — ${firedMeanings[0].text}${liftNote}`
        : firedMeanings[0].text;
    }
  }
}

async function refreshEvidenceSignals() {
  const now = Date.now();
  if (now - evidenceLastFetchAt < EVIDENCE_POLL_MS) return;
  evidenceLastFetchAt = now;
  if (!EVIDENCE_SIGNAL_SUPPORTED_ASSETS.includes(activeSnapshotAsset)) {
    renderEvidenceSignals({ unsupported: true, asset: activeSnapshotAsset });
    return;
  }
  const url = activeSnapshotAsset === "btc" ? API_BTC_EVIDENCE_SIGNALS_URL : API_EVIDENCE_SIGNALS_URL;
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`evidence signals ${res.status}`);
    renderEvidenceSignals(await res.json());
  } catch (error) {
    console.error("Evidence signal fetch error:", error);
    renderEvidenceSignals({ error: true });
  }
  // 2026-08-31: redraw the Snapshot chart's evidenceSignalTpLevels() lines right away (same idiom
  // as refreshLiquidationMap()/refreshRegimeWide24() below) instead of waiting up to ~5s for the
  // next render() tick to pick up the updated latestEvidenceSignals.
  renderSnapshotChart();
}

// Live PREVIEW of the currently-forming bar (2026-08-26) -- small live-dot overlay on the strip
// chips + a standalone badge, deliberately kept separate from renderEvidenceSignals()/the confirmed
// dot/state text above so a provisional reading can never be mistaken for (or silently overwrite)
// the validated confirmed one. "미확정" in the badge text is load-bearing, not decoration -- see
// load_evidence_signals_provisional()'s docstring in dashboard/server.py for why this reading has
// no lift track record of its own and can flicker before the bar closes.
function renderEvidenceSignalsProvisional(payload) {
  // 2026-09-02: 이 진행중(미확정) 미리보기는 ETH 전용 엔드포인트 데이터다(BTC 라이브 미리보기는
  // 이번에 만들지 않음, 확정봉 패널만 코인탭을 따라가도록 함). BTC 탭에서 이 함수가 그대로
  // 돌면 ETH의 실시간 발동으로 BTC 신호행/칩을 덮어써 버린다(evidence-row는 data-signal 이름으로만
  // 찾으므로 지금 BTC 데이터가 들어있는 바로 그 행을 잘못 골라 씀) -- 그래서 탭이 ETH가 아니면
  // DOM은 건드리지 않고 조용히 빠진다. latestEvidenceSignalsProvisional 자체(V자반등 폴백이
  // 재사용, 위 refreshEvidenceSignalsProvisional 참고)는 이 함수 밖에서 이미 갱신되므로 영향 없음.
  if (activeSnapshotAsset !== "eth") return;
  const badge = el("evidenceProvisionalBadge");
  const clearDots = () => {
    Object.values(EVIDENCE_STRIP_CHIP_IDS).forEach((id) => {
      const chip = el(id);
      const dot = chip?.querySelector(".signal-chip-live-dot");
      if (dot) dot.className = "signal-chip-live-dot";
      chip?.classList.remove("signal-chip-live-firing");
    });
  };
  if (!payload || payload.error || !payload.available) {
    clearDots();
    if (badge) { badge.className = "ops-badge neutral"; badge.textContent = "진행중 미리보기 -"; }
    return;
  }
  if (!payload.warmed_up) {
    clearDots();
    if (badge) { badge.className = "ops-badge warn"; badge.textContent = "진행중 미리보기 워밍업중"; }
    return;
  }
  const signals = Array.isArray(payload.signals) ? payload.signals : [];
  signals.forEach((s) => {
    const chip = EVIDENCE_STRIP_CHIP_IDS[s.name] ? el(EVIDENCE_STRIP_CHIP_IDS[s.name]) : null;
    if (!chip) return;
    const dot = chip.querySelector(".signal-chip-live-dot");
    if (dot) dot.className = `signal-chip-live-dot ${evidenceSideLabel(s, { bottom: "live-bottom", top: "live-top", both: "live-mixed", none: "" })}`;
    // 2026-08-27 (user request): border + dot + text all blink together (not just the small live
    // dot) specifically while this signal is firing in the still-forming bar -- same condition as
    // the dot's own color above, just also driving one animation on the chip root so its border/
    // background/label/state fade with it (see .signal-chip-live-firing, an opacity pulse).
    chip.classList.toggle("signal-chip-live-firing", evidenceSideTone(s) !== "neutral");
  });
  // 2026-08-26 user request: the compact chip dots above aren't enough -- also extend each
  // signal's own strip (in the "자세히" detail list) with a live bar for the still-forming bar,
  // every ~10s. Reuses the confirmed history captured by renderEvidenceSignals() (evidenceHistoryBySignal)
  // so this never re-fetches or re-derives the 47 confirmed bars -- only the appended live bar changes.
  signals.forEach((s) => {
    const hist = evidenceHistoryBySignal[s.name];
    if (!hist) return; // confirmed history not loaded yet -- nothing to append a live bar onto
    const row = document.querySelector(`.evidence-row[data-signal="${s.name}"]`);
    const svgEl = row?.querySelector(".evidence-strip-wrap > svg.evidence-strip");
    if (svgEl) {
      const liveTone = evidenceSideTone(s);
      svgEl.outerHTML = evidenceStripSvg(hist.bottom_history, hist.top_history, hist.latest_bar_utc, 5, liveTone, payload.bar_open_utc, "evidence", hist.bottom_raw_fire, hist.top_raw_fire);
    }
    const timeLabel = row?.querySelector(".strip-time-now");
    if (timeLabel) {
      const liveTimeText = fmtShortTs(payload.bar_open_utc);
      timeLabel.setAttribute("data-default", liveTimeText);
      timeLabel.textContent = liveTimeText;
    }
  });
  if (badge) {
    const net = Number(payload.net_score || 0);
    badge.className = `ops-badge ${net > 0 ? "good" : net < 0 ? "bad" : "neutral"}`;
    badge.textContent = `진행중(미확정) ${payload.bar_elapsed_seconds}s경과 · 바닥${payload.bottom_votes || 0}·천장${payload.top_votes || 0}`;
  }
}

async function refreshEvidenceSignalsProvisional() {
  const now = Date.now();
  if (now - evidenceProvisionalLastFetchAt < EVIDENCE_PROVISIONAL_POLL_MS) return;
  evidenceProvisionalLastFetchAt = now;
  try {
    const res = await fetch(API_EVIDENCE_SIGNALS_PROVISIONAL_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`evidence signals provisional ${res.status}`);
    const payload = await res.json();
    latestEvidenceSignalsProvisional = payload; // see its declaration above -- V자반등 진행중 미리보기가 재사용
    if (payload && payload.available && payload.bar_open_utc) {
      if (lastSeenProvisionalBarOpenUtc && payload.bar_open_utc !== lastSeenProvisionalBarOpenUtc) {
        // The bar we were just previewing closed and a new one started forming -- the confirmed
        // signal for that just-closed bar should be available server-side within its own 60s cache,
        // so pull it now instead of waiting out the rest of the 5-min confirmed poll interval.
        evidenceLastFetchAt = 0;
        refreshEvidenceSignals();
      }
      lastSeenProvisionalBarOpenUtc = payload.bar_open_utc;
    }
    renderEvidenceSignalsProvisional(payload);
  } catch (error) {
    console.error("Evidence signal (provisional) fetch error:", error);
    renderEvidenceSignalsProvisional({ error: true });
  }
}

async function refreshVReboundSignal() {
  const now = Date.now();
  if (now - vReboundLastFetchAt < V_REBOUND_POLL_MS) return;
  vReboundLastFetchAt = now;
  try {
    const res = await fetch(API_V_REBOUND_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`v-rebound signal ${res.status}`);
    latestVRebound = await res.json();
  } catch (error) {
    console.error("V-rebound signal fetch error:", error);
    latestVRebound = { warmed_up: false, error: "fetch_failed" };
  }
}

async function refreshLiquidation5mSignal() {
  const now = Date.now();
  if (now - liquidation5mLastFetchAt < LIQUIDATION_5M_POLL_MS) return;
  liquidation5mLastFetchAt = now;
  try {
    const res = await fetch(`${API_LIQUIDATION_5M_URL}?asset=${activeSnapshotAsset}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`liquidation 5m signal ${res.status}`);
    latestLiquidation5m = await res.json();
  } catch (error) {
    console.error("Liquidation 5m signal fetch error:", error);
    latestLiquidation5m = { warmed_up: false, error: "fetch_failed" };
  }
}

async function refreshBasisLiquiditySignal() {
  const now = Date.now();
  if (now - basisLiquidationLastFetchAt < BASIS_LIQUIDATION_POLL_MS) return;
  basisLiquidationLastFetchAt = now;
  try {
    const res = await fetch(`${API_BASIS_LIQUIDATION_URL}?asset=${activeSnapshotAsset}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`basis liquidation signal ${res.status}`);
    latestBasisLiquidation = await res.json();
  } catch (error) {
    console.error("Basis liquidation signal fetch error:", error);
    latestBasisLiquidation = { warmed_up: false, error: "fetch_failed" };
  }
}

async function refreshLiqBurstState() {
  const now = Date.now();
  if (now - liqBurstStateLastFetchAt < LIQ_BURST_STATE_POLL_MS) return;
  liqBurstStateLastFetchAt = now;
  try {
    const res = await fetch(API_LIQ_BURST_STATE_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`liq burst state ${res.status}`);
    latestLiqBurstState = await res.json();
  } catch (error) {
    console.error("Liq burst state fetch error:", error);
    latestLiqBurstState = { available: false };
  }
}

async function refreshLiquidationDirectionSignal() {
  const now = Date.now();
  if (now - liquidationDirectionLastFetchAt < LIQUIDATION_DIRECTION_POLL_MS) return;
  liquidationDirectionLastFetchAt = now;
  try {
    const res = await fetch(`${API_LIQUIDATION_DIRECTION_URL}?asset=${activeSnapshotAsset}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`liquidation direction signal ${res.status}`);
    latestLiquidationDirection = await res.json();
  } catch (error) {
    console.error("Liquidation direction signal fetch error:", error);
    latestLiquidationDirection = { warmed_up: false, error: "fetch_failed" };
  }
}

// Unlike latestVRebound (picked up by the next state-driven render() pass), the liquidation map
// has no such host -- it self-triggers both the panel list and the snapshot chart right after a
// fetch resolves, same pattern as refreshEvidenceSignals().
async function refreshLiquidationMap() {
  const now = Date.now();
  if (now - liquidationMapLastFetchAt < LIQUIDATION_MAP_POLL_MS) return;
  liquidationMapLastFetchAt = now;
  try {
    const res = await fetch(`${API_LIQUIDATION_MAP_URL}?asset=${activeSnapshotAsset}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`liquidation map ${res.status}`);
    latestLiquidationMap = await res.json();
  } catch (error) {
    console.error("Liquidation map fetch error:", error);
    latestLiquidationMap = { warmed_up: false, error: "fetch_failed" };
  }
  renderLiquidationMapPanel();
  renderSnapshotChart();
}

// wide24 HMM regime overlay (2026-08-26) for the Snapshot chart -- CONFIRMED research artifact,
// see scripts/live_regime_wide24_signal_20260826.py docstring for why it's independent of whatever
// regime model the live bot itself routes on.
async function refreshRegimeWide24() {
  const now = Date.now();
  if (now - regimeWide24LastFetchAt < REGIME_WIDE24_POLL_MS) return;
  regimeWide24LastFetchAt = now;
  try {
    const res = await fetch(API_REGIME_WIDE24_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`regime wide24 ${res.status}`);
    latestRegimeWide24 = await res.json();
  } catch (error) {
    console.error("Regime wide24 fetch error:", error);
    latestRegimeWide24 = { warmed_up: false, error: "fetch_failed", history: [] };
  }
  renderSnapshotChart();
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
    const res = await fetch(API_REGIME_BTC_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`regime btc ${res.status}`);
    latestRegimeBtc = await res.json();
  } catch (error) {
    console.error("Regime BTC fetch error:", error);
    latestRegimeBtc = { warmed_up: false, error: "fetch_failed", history: [] };
  }
  renderSnapshotChart();
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
// 2026-09-02 (사용자 요청): V자반등 **경제라벨** 후보의 섀도우 원장 표시.
// ⚠️위쪽 V자반등 칩(매 봉 giveback 모델)과는 **다른 모델**이다 -- 라벨 정의부터 다르다.
// 이건 주문을 내지 않는 가상 원장이고, HOLDOUT이 1회 노출로 소진돼 남은 유일한 검증 경로다.
// 근거: docs/model_contracts/eth_v_rebound_econ_label_autotrade_spec_20260902.md
async function refreshVrebEconShadow() {
  const now = Date.now();
  if (now - vrebEconShadowLastFetchAt < VREB_ECON_SHADOW_POLL_MS) return;
  vrebEconShadowLastFetchAt = now;
  try {
    const res = await fetch(API_VREB_ECON_SHADOW_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`vreb econ shadow ${res.status}`);
    renderVrebEconShadow(await res.json());
  } catch (error) {
    console.error("V-rebound econ shadow fetch error:", error);
    const sub = el("vrebEconShadowSub");
    if (sub) sub.textContent = "불러오기 실패";
  }
}
function renderVrebEconShadow(p) {
  const sub = el("vrebEconShadowSub");
  if (!p || typeof p !== "object") {
    if (sub) sub.textContent = "데이터 없음";
    setH("vrebEconShadowList", `<div class="macro-calendar-empty">섀도우 러너가 아직 기록을 남기지 않았습니다.</div>`);
    return;
  }
  const n = Number(p.closed_trades || 0);
  const ref = p.backtest_reference || {};
  if (sub) {
    sub.textContent = n
      ? `가상 ${n}건 청산 · 보유 ${p.n_open || 0} · 백테스트 기대 HOLDOUT ${ref.holdout_exp_bp}bp`
      : `보유 ${p.n_open || 0}건 · 아직 청산 기록 없음 (백테스트 기대 ${ref.holdout_exp_bp}bp)`;
  }
  const rows = [];
  const tone = (v) => (v == null ? "neutral" : v > 0 ? "good" : "warn");
  const bp = (v) => (v == null ? "-" : `${v > 0 ? "+" : ""}${Number(v).toFixed(2)}bp`);
  if (n) {
    rows.push(`<article class="ops-health-row ${tone(p.exp_bp)}">
      <span class="ops-health-dot" aria-hidden="true"></span>
      <div class="ops-health-info"><strong>건당 기대값</strong>
        <span>누적 ${bp(p.total_bp)} · 승률 ${p.win_rate == null ? "-" : (p.win_rate * 100).toFixed(1) + "%"} · 손익비 ${p.payoff ?? "-"}</span></div>
      <span class="ops-health-status-badge">${bp(p.exp_bp)}</span>
    </article>`);
    rows.push(`<article class="ops-health-row neutral">
      <span class="ops-health-dot" aria-hidden="true"></span>
      <div class="ops-health-info"><strong>최대 낙폭 / 연속손실</strong>
        <span>백테스트 HOLDOUT: ${ref.holdout_exp_bp}bp · 승률 ${(ref.holdout_win_rate * 100).toFixed(1)}% · 손익비 ${ref.holdout_payoff}</span></div>
      <span class="ops-health-status-badge">${bp(p.max_dd_bp)} / ${p.consec_loss ?? 0}</span>
    </article>`);
  }
  for (const q of (p.open_positions || [])) {
    rows.push(`<article class="ops-health-row neutral">
      <span class="ops-health-dot" aria-hidden="true"></span>
      <div class="ops-health-info"><strong>보유 ${q.side === "long" ? "롱" : "숏"}${q.armed ? " · 무장" : ""}</strong>
        <span>진입 ${Number(q.entry).toFixed(2)} · 손절 ${Number(q.stop).toFixed(2)} · p=${Number(q.proba).toFixed(3)}</span></div>
      <span class="ops-health-status-badge">보유중</span>
    </article>`);
  }
  for (const t of (p.recent_trades || []).slice().reverse()) {
    rows.push(`<article class="ops-health-row ${tone(t.pnl_bp)}">
      <span class="ops-health-dot" aria-hidden="true"></span>
      <div class="ops-health-info"><strong>${t.side === "long" ? "롱" : "숏"} 청산 · ${t.reason === "stop" ? "손절선" : "만기"}</strong>
        <span>${fmtMacroCalendarTime(t.exit_utc)} · p=${t.proba == null ? "-" : Number(t.proba).toFixed(3)}</span></div>
      <span class="ops-health-status-badge">${bp(t.pnl_bp)}</span>
    </article>`);
  }
  setH("vrebEconShadowList", rows.length ? rows.join("")
    : `<div class="macro-calendar-empty">아직 가상 진입이 없습니다 (임계값 0.8158).</div>`);
}
async function refreshMacroCalendar() {
  const now = Date.now();
  if (now - macroCalendarLastFetchAt < MACRO_CALENDAR_POLL_MS) return;
  macroCalendarLastFetchAt = now;
  try {
    const res = await fetch(API_MACRO_CALENDAR_URL, { cache: "no-store" });
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
    const target = button.dataset.pageTab; // "ops" | "snapshot" (라이브 탭 제거, 2026-08-31)
    activePageTab = target;
    el("opsTabPanel")?.classList.toggle("hidden", target !== "ops");
    el("snapshotTabPanel")?.classList.toggle("hidden", target !== "snapshot");
    document.querySelectorAll(".page-tab").forEach((tab) => tab.classList.toggle("active", tab === button));
    if (target === "ops") {
      opsLastFetchAt = 0; refreshOpsStatus();
    } else if (target === "snapshot") {
      evidenceLastFetchAt = 0; refreshEvidenceSignals();
      evidenceProvisionalLastFetchAt = 0; refreshEvidenceSignalsProvisional();
      vReboundLastFetchAt = 0; refreshVReboundSignal();
      liquidation5mLastFetchAt = 0; refreshLiquidation5mSignal();
      basisLiquidationLastFetchAt = 0; refreshBasisLiquiditySignal();
      liqBurstStateLastFetchAt = 0; refreshLiqBurstState();
      liquidationDirectionLastFetchAt = 0; refreshLiquidationDirectionSignal();
      liquidationMapLastFetchAt = 0; refreshLiquidationMap();
      regimeWide24LastFetchAt = 0; refreshRegimeWide24();
      regimeBtcLastFetchAt = 0; refreshRegimeBtc();
      macroCalendarLastFetchAt = 0; refreshMacroCalendar();
      vrebEconShadowLastFetchAt = 0; refreshVrebEconShadow();
      sessionAlertsLastFetchAt = 0; refreshSessionAlerts();
      lastSnapshotHistoryFetchAt = 0; maybeFetchSnapshotChartHistory();
    }
  }));
}

function setupScrollRendering() {
  document.addEventListener("scroll", () => {
    isScrolling = true;
    window.clearTimeout(scrollIdleTimer);
    scrollIdleTimer = window.setTimeout(() => {
      isScrolling = false;
      if (!document.hidden) tick();
    }, 150);
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
function liquidationDensityHistory() {
  const map = latestLiquidationMap;
  if (!map || !map.warmed_up || !map.bin_width) return [];
  return (map.heatmap_history || []).map((snap) => ({
    tsMs: Date.parse(snap.ts_utc),
    binWidth: map.bin_width,
    bins: (snap.bins || []).map((b) => ({ price: b.price, weightPct: b.weight_pct || 0 })),
  }));
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

// Evidence-signal take-profit lines (2026-08-31 user request, follow-up to the label-chip TP-price
// text) -- one AVERAGED line per side (long/short), not one per fired signal. First shipped as one
// line per signal, but with several signals often firing at once (see
// eth_dashboard_evidence_signal_tp_price_display_20260831 memory) that cluttered the chart with a
// pile of near-identical dashed lines; user asked to collapse each side down to a single average
// instead ("롱과 숏 각각 익절가들 모아서 평균을 낸 익절가를 보여줘"). Same generic
// {val,color,label,dashed,width} shape nearestLiquidationLevel() above already feeds into
// riskLevels. Reads latestEvidenceSignals (set by renderEvidenceSignals()), not a parameter, so
// this chart's other callers don't need to know how to fetch evidence signals themselves.
//
// ETH-only: /api/evidence-signals is a single, non-per-asset endpoint (every K/ATR% value it was
// trained on is ETH's own), so it must not draw on a BTC/SOL/XRP/HYPE chart just because that's the
// active Snapshot coin.
//
// Label is "롱익절"/"숏익절" (bottom-side signals bet long, top-side bet short), not each signal's
// own EVIDENCE_SIGNAL_KO name -- those are full phrases ("칼만필터 추세이탈 평균회귀") far too long
// for the chart's left-margin tag column, which every existing tag ("지지1"/"저항1"/"진입"/"현재")
// keeps to <=3 characters.
function evidenceSignalTpLevels() {
  if (activeSnapshotAsset !== "eth") return [];
  const signals = latestEvidenceSignals?.signals;
  if (!Array.isArray(signals)) return [];
  const withTp = signals.filter((s) => s.model_tp_price != null);
  const avg = (arr) => arr.reduce((sum, v) => sum + v, 0) / arr.length;
  const longTp = withTp.filter((s) => s.model_side === "bottom").map((s) => s.model_tp_price);
  const shortTp = withTp.filter((s) => s.model_side === "top").map((s) => s.model_tp_price);
  const levels = [];
  if (longTp.length) levels.push({ val: avg(longTp), color: "var(--good)", label: "롱익절", dashed: true, width: 2 });
  if (shortTp.length) levels.push({ val: avg(shortTp), color: "var(--bad)", label: "숏익절", dashed: true, width: 2 });
  return levels;
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
    candles.push({ time: candleTs, open: price, high: price, low: price, close: price });
    if (candles.length > CHART_MAX_CANDLES) candles.shift();
  } else {
    last.high = Math.max(last.high, price);
    last.low = Math.min(last.low, price);
    last.close = price;
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
function renderSnapshotChart() {
  const svg = el("candleSvgSnapshot");
  const fullCandles = candleHistoryByAsset[activeSnapshotAsset] || [];
  if (!svg || !fullCandles.length) return;
  // Sliced to SNAPSHOT_CHART_MAX_CANDLES (6h) -- narrower than the shared candleHistoryByAsset
  // cache (still 8h, CHART_MAX_CANDLES) so the density-history overlay always has a real snapshot
  // behind every visible column (see that constant's comment).
  const candles = fullCandles.slice(-SNAPSHOT_CHART_MAX_CANDLES);
  const currentPrice = Number(latestLivePriceByAsset[activeSnapshotAsset] || candles[candles.length - 1]?.close || 0);
  const riskLevels = [...nearestLiquidationLevel(), ...evidenceSignalTpLevels()];
  renderCandleSvg(svg, candles, [], 0, currentPrice, riskLevels, liquidationDensityHistory());
}

// wide24/GBM3 regime overlay -- drawn as a ribbon INSIDE renderCandleSvg() itself (2026-08-26,
// moved in from a standalone strip below the chart per user request: "레짐 그래프를 청산맵 안에
// 넣을 순 없어?"). Dominant-class color (not a 3-way blend) matches the categorical tone convention
// the evidence-signal strips use elsewhere; opacity scales with confidence so an uncertain reading
// fades rather than asserting a false-confident color.
const REGIME_DOMINANT_COLOR = { bull: "#6bab84", bear: "#cf6a5c", chop: "#8b91a6" };
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

// Sequential colormap (matplotlib viridis stops) for the liquidation density heatmap band --
// 2026-08-25, replaces the old dual-hue support/orange scheme so density alone (not which side)
// drives color, matching Coinglass's liquidation-heatmap convention the user asked to replicate.
// Dark purple at t=0 reads as near-background (low density fades out); bright yellow at t=1 pops.
const VIRIDIS_STOPS = [
  [0.0, [68, 1, 84]],
  [0.2, [65, 68, 135]],
  [0.4, [42, 120, 142]],
  [0.6, [34, 168, 132]],
  [0.8, [122, 209, 81]],
  [1.0, [253, 231, 37]],
];
function viridisColor(t) {
  t = clamp01(t);
  for (let i = 0; i < VIRIDIS_STOPS.length - 1; i++) {
    const [t0, c0] = VIRIDIS_STOPS[i], [t1, c1] = VIRIDIS_STOPS[i + 1];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0 || 1);
      const rgb = c0.map((v, k) => Math.round(v + (c1[k] - v) * f));
      return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    }
  }
  const last = VIRIDIS_STOPS[VIRIDIS_STOPS.length - 1][1];
  return `rgb(${last[0]},${last[1]},${last[2]})`;
}

function renderCandleSvg(svg, candles, journal, entryPrice, currentPrice, riskLevels = [], densityHistory = []) {
  const parentW = svg.parentElement ? svg.parentElement.clientWidth : 0;
  const parentH = svg.parentElement ? svg.parentElement.clientHeight : 0;
  const mobileChart = isMobileChartMode();
  const w = mobileChart ? Math.max(parentW, 320) : Math.max(parentW, 1200);
  const h = mobileChart ? Math.max(parentH, 260) : 400;
  const ml = mobileChart ? 34 : 45, mr = mobileChart ? 68 : 112, mt = 20, mb = 40;
  const cw = w - ml - mr, ch = h - mt - mb;
  const NS = "http://www.w3.org/2000/svg";
  const viewport = visibleCandleWindow(candles);
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
  const pad = (maxP - minP) * 0.15 || 1;
  const yMin = minP - pad, yMax = maxP + pad;
  const ySpan = Math.max(yMax - yMin, 1e-5); // Prevent division by zero

  const xAt = (i) => ml + (i * cw) / candles.length;
  const yAt = (v) => mt + ((yMax - v) * ch) / ySpan;
  const bw = (cw / candles.length) * 0.8;

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
  const REGIME_SOURCE_BY_ASSET = { eth: () => latestRegimeWide24, btc: () => latestRegimeBtc };
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
  const REGIME_RIBBON_Y = h - mb + 28, REGIME_RIBBON_H = 8;

  // Liquidation-map density heatmap -- drawn first so candles/grid/lines sit on top of it (paint
  // order unchanged). 2026-08-25: replaced the old right-anchored, length-encoded "volume profile"
  // bar (capped at 30% of chart width, "left 70% stays clean for candles") with a full-width
  // background band, color intensity encoding density via a single sequential colormap
  // (viridisColor) -- matches Coinglass's liquidation-heatmap convention at the user's explicit
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
  const drawDensitySeg = (x0, x1, top, bottom, t) => {
    if (x1 <= x0) return;
    const rect = document.createElementNS(NS, "rect");
    rect.setAttribute("x", x0); rect.setAttribute("y", top);
    rect.setAttribute("width", x1 - x0); rect.setAttribute("height", bottom - top);
    rect.setAttribute("fill", viridisColor(t));
    rect.setAttribute("fill-opacity", "0.85");
    svg.appendChild(rect);
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
      const bottom = Math.min(h - mb, yAt(price - half));
      if (bottom <= top) return;
      const pct = clamp01(weightByPrice.get(price) || 0);
      const t = densityClip > 0 ? Math.min(1, pct / densityClip) : 0;
      drawDensitySeg(x0, x1, top, bottom, t);
    });
  });

  // Resistance/support/current/entry price tags -- computed here (before the axis ticks below) so
  // the tick loop can tell when a grid label would land on top of one of these and skip it.
  // Rendering (the actual lines/boxes) still happens later, after candles/markers, so paint order
  // is unchanged.
  const priceLabels = [];
  if (includeCurrentPrice && currentPrice > 0) priceLabels.push({ val: currentPrice, color: "var(--accent)", label: "현재", dashed: true, width: 2 });
  if (entryPrice > 0) priceLabels.push({ val: entryPrice, color: "var(--amber)", label: "진입", dashed: false, width: 3 });
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
    p.offBottom = rawY > h - mb;
    p.outOfView = p.offTop || p.offBottom;
    p.realY = p.outOfView ? (p.offTop ? mt + 2 : h - mb - 2) : rawY;
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
  let topStack = mt + 2, bottomStack = h - mb - 2;
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

  // Grid & Y-Axis Ticks
  axisTicks(yMin, yMax, 6).forEach(t => {
    const y = yAt(t);
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", ml); line.setAttribute("x2", w - mr);
    line.setAttribute("y1", y); line.setAttribute("y2", y);
    line.setAttribute("class", "chart-grid");
    svg.appendChild(line);

    // Skip the tick's price label (not the gridline) when a resistance/support/current-price tag
    // already sits here -- both are text anchored in the same right-edge column (tag box spans
    // w-mr+4..w-mr+4+boxW, tick text ends at w-6), so without this a grid number like "2520.0"
    // renders directly on top of a tag box like "2526.1", especially once several tags cascade
    // near an edge (see priceLabels above).
    const collidesWithPriceTag = priceLabels.some(p => {
      const py = p.adjustedY !== undefined ? p.adjustedY : p.realY;
      return Math.abs(y - py) < minGap;
    });

    if (!mobileChart && !collidesWithPriceTag) {
      const txt = document.createElementNS(NS, "text");
      txt.setAttribute("x", w - 6); txt.setAttribute("y", y + 4);
      txt.setAttribute("text-anchor", "end");
      txt.setAttribute("font-size", "13");
      txt.setAttribute("font-weight", "700");
      txt.setAttribute("fill", "var(--muted)");
      txt.textContent = fmtNum(t, 1);
      svg.appendChild(txt);
    }
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
    svg.appendChild(line);

    const txt = document.createElementNS(NS, "text");
    txt.setAttribute("x", x);
    txt.setAttribute("y", h - mb + 21);
    txt.setAttribute("text-anchor", "middle");
    txt.setAttribute("font-size", isMobileChartMode() ? "12" : "13");
    txt.setAttribute("font-weight", "700");
    txt.setAttribute("fill", "var(--muted)");
    txt.textContent = fmtDateTick(c.time * 1000);
    svg.appendChild(txt);
  });

  if (regimeByTsForChart) {
    candles.forEach((c, i) => {
      const r = regimeByTsForChart.get(c.time);
      if (!r) return;
      const rect = document.createElementNS(NS, "rect");
      rect.setAttribute("x", xAt(i)); rect.setAttribute("y", REGIME_RIBBON_Y);
      rect.setAttribute("width", bw); rect.setAttribute("height", REGIME_RIBBON_H);
      rect.setAttribute("rx", "1.5");
      rect.setAttribute("fill", REGIME_DOMINANT_COLOR[regimeDominant(r)]);
      rect.setAttribute("fill-opacity", (0.55 + 0.45 * clamp01(r.confidence)).toFixed(2));
      const title = document.createElementNS(NS, "title");
      const pct = (v) => Math.round(v * 100);
      title.textContent = `레짐: 강세${pct(r.bull_prob)}% 약세${pct(r.bear_prob)}% 횡보${pct(r.chop_prob)}% (신뢰도${pct(r.confidence)}%)`;
      rect.appendChild(title);
      svg.appendChild(rect);
    });
    const ribbonLabel = document.createElementNS(NS, "text");
    ribbonLabel.setAttribute("x", ml - 6);
    ribbonLabel.setAttribute("y", REGIME_RIBBON_Y + REGIME_RIBBON_H - 1);
    ribbonLabel.setAttribute("text-anchor", "end");
    ribbonLabel.setAttribute("font-size", "9");
    ribbonLabel.setAttribute("fill", "var(--muted)");
    ribbonLabel.textContent = "레짐";
    svg.appendChild(ribbonLabel);
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
      ? "레짐: 이 코인용 레짐분류기가 아직 없음 (ETH 전용 모델) -- 추후 학습 예정"
      : "레짐: 웜업 중이거나 일시적으로 갱신 실패 -- 다음 5분 주기에 자동 재시도됩니다";
    waitRect.appendChild(waitTitle);
    svg.appendChild(waitRect);
    const waitLabel = document.createElementNS(NS, "text");
    waitLabel.setAttribute("x", ml - 6);
    waitLabel.setAttribute("y", REGIME_RIBBON_Y + REGIME_RIBBON_H - 1);
    waitLabel.setAttribute("text-anchor", "end");
    waitLabel.setAttribute("font-size", "9");
    waitLabel.setAttribute("fill", "var(--muted)");
    waitLabel.textContent = regimeRibbonUnsupported ? "레짐 (미지원)" : "레짐";
    svg.appendChild(waitLabel);
  }

  // Candles
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

  priceLabels.forEach(p => {
    const labelYRaw = p.adjustedY !== undefined ? p.adjustedY : p.realY;
    const labelY = Math.max(mt + 9, Math.min(h - mb - 9, labelYRaw));
    const lineDashed = p.dashed || p.outOfView;

    // Line stays at real (clamped) price position
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", ml); line.setAttribute("x2", w - mr);
    line.setAttribute("y1", p.realY); line.setAttribute("y2", p.realY);
    line.setAttribute("stroke", p.color);
    line.setAttribute("stroke-width", String(p.width || 2));
    if (lineDashed) line.setAttribute("stroke-dasharray", "4,4");
    if (p.outOfView) line.setAttribute("opacity", "0.72");
    svg.appendChild(line);

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
    pTxt.setAttribute("fill", "#1a1208");
    pTxt.textContent = `${p.offTop ? "↑ " : p.offBottom ? "↓ " : ""}${fmtNum(p.val, 1)}`;
    svg.appendChild(pTxt);
  });

  // Liquidation-density color legend -- top-right inset over the plot, 2026-08-25 user request
  // ("오른쪽 위에 색깔별로 크기를 표시해줘"). Labeled 낮음/높음 (low/high), not a $ scale like
  // Coinglass's own colorbar: weightPct here is a synthetic, percentile-clipped RELATIVE density
  // (compute_raw_bins() has no real notional/OI data to draw from -- see its docstring), so a dollar
  // figure would misrepresent it as real magnitude. pointer-events:none so it never blocks the
  // hover/tooltip layer appended right after this.
  if ((densityHistory || []).length) {
    const legendW = mobileChart ? 56 : 72, legendH = 8;
    // legendY nudged up 2026-08-27 (user report: covering candle wicks near the top of the price
    // range) -- backing box now sits flush with the SVG's top edge (y=0) instead of dipping well
    // into the plot area below mt.
    const legendX = w - mr - legendW - 6, legendY = mt - 4;
    const legendGroup = document.createElementNS(NS, "g");
    legendGroup.setAttribute("pointer-events", "none");

    // #viridisGradient is a static, document-level <defs> in index.html (hoisted 2026-08-25) --
    // not recreated here every call.
    const backing = document.createElementNS(NS, "rect");
    backing.setAttribute("x", legendX - 6); backing.setAttribute("y", legendY - 16);
    backing.setAttribute("width", legendW + 12); backing.setAttribute("height", 34);
    backing.setAttribute("rx", "4"); backing.setAttribute("fill", "var(--chart-bg)");
    backing.setAttribute("opacity", "0.78");
    legendGroup.appendChild(backing);

    const title = document.createElementNS(NS, "text");
    title.setAttribute("x", legendX + legendW / 2); title.setAttribute("y", legendY - 6);
    title.setAttribute("text-anchor", "middle"); title.setAttribute("font-size", "9.5");
    title.setAttribute("fill", "var(--muted)");
    title.textContent = "청산 밀도";
    legendGroup.appendChild(title);

    const bar = document.createElementNS(NS, "rect");
    bar.setAttribute("x", legendX); bar.setAttribute("y", legendY);
    bar.setAttribute("width", legendW); bar.setAttribute("height", legendH);
    bar.setAttribute("rx", "2"); bar.setAttribute("fill", "url(#viridisGradient)");
    legendGroup.appendChild(bar);

    const lowLabel = document.createElementNS(NS, "text");
    lowLabel.setAttribute("x", legendX); lowLabel.setAttribute("y", legendY + legendH + 10);
    lowLabel.setAttribute("font-size", "9"); lowLabel.setAttribute("fill", "var(--muted)");
    lowLabel.textContent = "낮음";
    legendGroup.appendChild(lowLabel);

    const highLabel = document.createElementNS(NS, "text");
    highLabel.setAttribute("x", legendX + legendW); highLabel.setAttribute("y", legendY + legendH + 10);
    highLabel.setAttribute("text-anchor", "end"); highLabel.setAttribute("font-size", "9");
    highLabel.setAttribute("fill", "var(--muted)");
    highLabel.textContent = "높음";
    legendGroup.appendChild(highLabel);

    svg.appendChild(legendGroup);
  }

  // Create Hover Layer on Top
  const hoverGroup = document.createElementNS(NS, "g");
  hoverGroup.setAttribute("class", "hover-layer");
  svg.appendChild(hoverGroup);

  // y2 reaches through the regime ribbon on the Snapshot chart so hovering visibly crosses both
  // (2026-08-26, "레짐 그래프도 십자선에 걸쳤으면 좋겠어") -- plain candle chart baseline otherwise.
  const vLine = document.createElementNS(NS, "line");
  vLine.setAttribute("x1", 0); vLine.setAttribute("x2", 0);
  vLine.setAttribute("y1", mt);
  vLine.setAttribute("y2", regimeByTsForChart ? REGIME_RIBBON_Y + REGIME_RIBBON_H : h - mb);
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
  priceBadgeText.setAttribute("fill", "#0b1220");
  priceBadgeText.style.display = "none";
  priceBadgeText.style.pointerEvents = "none";
  hoverGroup.appendChild(priceBadgeText);

  // Candlestick Tooltip Support. regimeByTsForChart/isSnapshotChart computed once near the top of
  // this function (shared with the ribbon drawn above) -- same ETH-only guard applies here.
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
    const svgScale = Math.min(rect.width / w, rect.height / h);
    const svgOffsetX = (rect.width - w * svgScale) / 2;
    const svgOffsetY = (rect.height - h * svgScale) / 2;
    const mx = (evt.clientX - rect.left - svgOffsetX) / svgScale;
    const my = (evt.clientY - rect.top - svgOffsetY) / svgScale;

    if (my >= mt && my <= h - mb) {
      hLine.setAttribute("y1", my); hLine.setAttribute("y2", my);
      hLine.style.display = "block";
      const priceAtCursor = yMax - ((my - mt) * ySpan) / ch;
      const badgeY = Math.max(mt, Math.min(h - mb - priceBadgeH, my - priceBadgeH / 2));
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

    const tx = ml + (idx * cw) / candles.length + bw/2;

    vLine.setAttribute("x1", tx);
    vLine.setAttribute("x2", tx);
    vLine.style.display = "block";

    const r = regimeByTsForChart ? regimeByTsForChart.get(c.time) : null;
    const regimeLine = r
      ? `<br>레짐: ${regimeDominant(r) === "bull" ? "강세" : regimeDominant(r) === "bear" ? "약세" : "횡보"} ${Math.round(Math.max(r.bull_prob, r.bear_prob, r.chop_prob) * 100)}%`
      : "";
    showTooltip(evt.pageX, evt.pageY, `
      <b>${fmtDateTick(c.time * 1000)}</b><br>
      시가: ${fmtNum(c.open, 2)}<br>
      고가: ${fmtNum(c.high, 2)}<br>
      저가: ${fmtNum(c.low, 2)}<br>
      종가: ${fmtNum(c.close, 2)}${regimeLine}
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
}

function render(state, compactState = null, { stateChanged = true } = {}) {
  latestMainState = state;
  if (!stateChanged) return;

  const sess = state.session || {};
  const micro = state.microstructure || {}, tail = state.tail_risk || {};

  const sessionHtml = buildSessionHtml(sess);
  setH("topSession", sessionHtml);
  setT("topClock", fmtNowClock());
  
  // classifyIndicators() is the single source of truth for these thresholds (also reused by
  // seedModelIndicatorHistory() against server-provided history, so there is only one copy of
  // this logic to keep correct, not a live copy and a history copy). The 3 model indicators only
  // render on the Snapshot tab now (renderModelIndicatorList below) -- no Live-tab cards consume
  // ci/toneHistory anymore.
  const ci = classifyIndicators(micro, tail);
  const { whaleTone, cascadeTone } = ci;
  // toneHistory feeds the Snapshot tab's activity-strip rows (renderModelIndicatorList below).
  pushToneHistory("whale", whaleTone);
  pushToneHistory("retail_flow", ci.retail_flow.tone);
  pushToneHistory("liq_cascade", cascadeTone);

  // V자 급등락 (2026-08-29, TabPFN Tier0+rsi 모델, 2026-08-30 "유동성스윕 반등예측"에서 개명,
  // 2026-08-31 "V자 반등락"에서 다시 "V자 급등락"으로 개명 -- 사용자 요청) -- fetched
  // separately by refreshVReboundSignal(), "own fetch cycle, dashboard-side compute" category (own
  // klines fetch + frozen TabPFN context, not part of ci/toneHistory). Fires on EVERY liquidity_sweep
  // (14,259건, 2024-01~) -- frequent, large validated sample. tone is neutral (no recent sweep) or,
  // once one fires, good/bad -- 2026-08-29 user request: the backend now resolves the swept side +
  // rebound-vs-continuation call into a real predicted price direction (_predicted_tone()
  // server-side), replacing the old always-"warn" reading.
  // 2026-08-31 user request: "반등"/"반락"이 스윕 방향 대비 반전 여부(call)를 가리키는 말이라 실제
  // 예상 가격방향(tone/색깔)과 어긋나는 경우가 있었음(예: 상승스윕 후 반등=call은 "반등"이지만
  // 실제 방향은 하락이라 빨간색 -- 사용자가 "반등인데 왜 빨간색?" 지적). call 대신 tone에서 직접
  // 파생시켜 "급등"/"급락"으로 교체 -- 이 단어는 스윕 방향과 무관하게 항상 실제 예상 가격방향과
  // 일치하도록 설계(더는 call 값을 그대로 노출하지 않음). proba_rebound도 원래 call="rebound"의
  // 확률이라 direction에 따라 반전해서 "지금 실제 표시되는 방향(급등/급락)"의 확률로 재계산해야
  // 같은 어긋남이 확률 문구에서 재발하지 않음.
  const vReboundWarmedUp = !!(latestVRebound && latestVRebound.warmed_up);
  const vReboundActive = vReboundWarmedUp && !!latestVRebound.event_active;
  // 2026-09-01 (user request): CONFIRMED 콜이 없을 때("신호 없음")만, 증거신호 진행중 미리보기가
  // 이미 계산해둔 8개 트리거(latestEvidenceSignalsProvisional, 위 선언부 참고)를 재사용해 지금
  // 2026-09-01 매 봉 스코어링: 트리거 게이트가 없어져 웜업만 끝나면 항상 현재 봉 점수가 있다.
  // 그래서 "진행중(미확정) 바닥/천장/양쪽"(형성중 봉에서 트리거가 떴는지 미리 보여주던 예비 힌트)은
  // 도달 불가능해져 제거했다 -- 이제 형성중 봉을 기다릴 것 없이 확정 봉마다 실제 점수가 나온다.
  // "신호 없음"은 마지막 봉 피쳐가 NaN이라 채점에서 빠지는 드문 경우에만 남는 폴백.
  const vReboundTone = vReboundActive ? (latestVRebound.tone || "warn") : "neutral";
  // 2026-08-31 user request: 미반등 콜(반등 시도 자체가 없었다는 판정)을 더는 급등/급락으로 억지로
  // 묶지 않고 방향 무관 "미반등"으로 따로 표시 -- tone="flat"(백엔드 _predicted_tone, 2026-08-31
  // 개정)일 때 전용 단어. good/bad(진짜 반등 콜)만 급등/급락을 씁니다.
  const vReboundSubText = !vReboundWarmedUp ? "웜업 중"
    : vReboundActive ? (vReboundTone === "good" ? "급등" : vReboundTone === "bad" ? "급락" : "미반등")
    : "신호 없음";
  // P(급등) -- proba_rebound는 call="rebound"의 확률이라, direction="up"(상승스윕)일 때는 call=
  // "continuation"이 급등에 해당하므로 1-proba_rebound로 뒤집어야 함(direction="down"일 때는
  // call="rebound" 그대로가 급등이므로 안 뒤집음). 그다음 실제 표시되는 쪽의 확률만 골라 보여줌 --
  // good/bad는 항상 50% 이상(taker/short_term_return_z가 "발동방향이 맞을 확률"을 보여주는 것과
  // 같은 틀), flat(미반등)은 방향이 없으므로 P(continuation)=1-proba_rebound를 그대로 보여줌.
  const vReboundProbaGood = vReboundActive && Number.isFinite(latestVRebound.proba_rebound)
    ? (latestVRebound.direction === "down" ? latestVRebound.proba_rebound : 1 - latestVRebound.proba_rebound)
    : null;
  const vReboundProbaShown = vReboundProbaGood == null ? null
    : vReboundTone === "good" ? vReboundProbaGood
    : vReboundTone === "bad" ? 1 - vReboundProbaGood
    : 1 - latestVRebound.proba_rebound;

  // 베이시스 청산압박 (replaces 독성/toxicity, 2026-08-27) -- fetched separately by
  // refreshBasisLiquiditySignal(), same external-fetch category as latestVRebound above.
  // Directional (good=롱압박↑/bad=숏압박↑/neutral) as of 2026-08-29 user request -- was previously
  // a good/warn/bad calm/caution/danger risk gauge; see scripts/live_spot_perp_basis_signal_
  // 20260827.py's _direction()/_tone() for the short_pressure/long_pressure -> good/bad mapping.
  const basisLiqWarmedUp = !!(latestBasisLiquidation && latestBasisLiquidation.warmed_up);
  const basisLiqTone = basisLiqWarmedUp ? latestBasisLiquidation.tone : "neutral";

  // 청산 방향압력 (2026-08-25) -- fetched separately by refreshLiquidationDirectionSignal(), same
  // external-fetch category as latestVRebound above. Directional model-indicator, NOT evidence-
  // signal tier -- see scripts/live_liquidation_direction_signal_20260825.py docstring.
  const liqDirWarmedUp = !!(latestLiquidationDirection && latestLiquidationDirection.warmed_up);
  const liqDirTone = liqDirWarmedUp
    ? (latestLiquidationDirection.direction === "bullish" ? "good"
      : latestLiquidationDirection.direction === "bearish" ? "bad" : "neutral")
    : "neutral";

  // 2026-08-25: perf pass -- this whole block (gauge + chart + model-indicator list) only paints
  // anything the user can see while the Snapshot tab is active (snapshotTabPanel is display:none
  // otherwise), so it's gated the same way as tick()'s Snapshot-only fetches above. Data
  // accumulation (pushToneHistory calls above this block, liqDirTone derivation)
  // stays unconditional -- only the paint work below is skipped, so history strips have no gap when
  // the user switches back to Snapshot.
  if (activePageTab === "snapshot") {
    setH("liqVolumeGauge", liquidationVolumeGaugeHtml());

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
    if (nowForSnapshotChart - lastSnapshotChartRenderAt >= SNAPSHOT_CHART_RENDER_MIN_INTERVAL_MS) {
      lastSnapshotChartRenderAt = nowForSnapshotChart;
      updateSnapshotCandleLive();
      renderSnapshotChart();
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
      {
        key: "v_rebound", label: "V자 급등락", tone: vReboundTone, subText: vReboundSubText,
        history: (latestVRebound && latestVRebound.history) || [],
        times: (latestVRebound && latestVRebound.times) || [],
        proba: vReboundProbaShown,
        derivedTag: "= 대시보드 자체계산",
        derivedTitle: "봇 내부 상태가 아니라 대시보드 서버가 별도로(TabPFN 모델, 고정된 과거 학습 컨텍스트) 계산 -- 아직 실제 매매 결정에는 연결되지 않음. 자세히 보기 참고.",
      },
    ], "snapSpecializedSignalList");

    // Snapshot tab: renderModelIndicatorList mirrors renderEvidenceSignals's row/strip UI.
    renderModelIndicatorList([
      {
        key: "liq_pressure", label: "베이시스 청산압박", tone: basisLiqTone,
        subText: basisLiquiditySubText(latestBasisLiquidation),
        history: (latestBasisLiquidation && latestBasisLiquidation.tone_history) || [],
        times: evenlySpacedBarTimes(latestBasisLiquidation && latestBasisLiquidation.latest_ts_utc, (latestBasisLiquidation && latestBasisLiquidation.tone_history || []).length, 5),
        derivedTag: "= 대시보드 자체계산·탐색적",
        derivedTitle: "봇 내부 상태가 아니라 대시보드 서버가 spot/perp klines를 직접 fetch해 계산 -- 아직 실제 매매 결정에는 연결되지 않음. 청산크라우딩 상관은 ~1개월 탐색적 표본(3-split 재현 전). 자세히 보기 참고.",
      },
      {
        key: "liq_cascade", label: "청산 캐스케이드", tone: ci.liq_cascade.tone,
        subText: ci.liq_cascade.subText, history: toneHistory.liq_cascade, times: toneHistoryTimes.liq_cascade,
        liveText: liqCascadeLiveDetail(tail),
      },
      {
        key: "liq_direction", label: "청산 방향압력", tone: liqDirTone,
        subText: liqDirWarmedUp ? liqDirectionSubText(latestLiquidationDirection) : "웜업 중",
        history: (latestLiquidationDirection && latestLiquidationDirection.tone_history) || [],
        times: evenlySpacedBarTimes(latestLiquidationDirection && latestLiquidationDirection.latest_ts_utc, (latestLiquidationDirection && latestLiquidationDirection.tone_history || []).length, 1),
      },
      { key: "whale", label: "수급 흐름", tone: ci.whale.tone, subText: ci.whale.subText, history: toneHistory.whale, times: toneHistoryTimes.whale },
      { key: "retail_flow", label: "리테일 수급", tone: ci.retail_flow.tone, subText: ci.retail_flow.subText, history: toneHistory.retail_flow, times: toneHistoryTimes.retail_flow },
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
      refreshEvidenceSignals();
      refreshEvidenceSignalsProvisional();
      refreshVReboundSignal();
      refreshLiquidation5mSignal();
      refreshBasisLiquiditySignal();
      refreshLiqBurstState();
      refreshLiquidationDirectionSignal();
      refreshLiquidationMap();
      refreshRegimeWide24();
      refreshRegimeBtc();
      refreshMacroCalendar();
      refreshVrebEconShadow();
      refreshSessionAlerts();
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
async function seedModelIndicatorHistory() {
  try {
    const res = await fetch("/api/model-indicator-history", { cache: "no-store", signal: AbortSignal.timeout(5000) });
    if (!res.ok) return;
    const payload = await res.json();
    const samples = Array.isArray(payload.samples) ? payload.samples : [];
    if (!samples.length) return;
    if (Object.values(toneHistory).some((arr) => arr.length)) return; // live tick already won the race
    for (const sample of samples) {
      const c = classifyIndicators(sample.microstructure, sample.tail_risk);
      pushToneHistory("liq_cascade", c.liq_cascade.tone);
      pushToneHistory("whale", c.whale.tone);
      pushToneHistory("retail_flow", c.retail_flow.tone);
    }
  } catch (error) {
    console.error("Model indicator history seed error (non-fatal, strip just starts empty):", error);
  }
}

(async () => {
  await seedModelIndicatorHistory();
  connectDashboardEvents();
  tick();
  setInterval(tick, POLL_MS);
})();
setInterval(() => {
  if (!isScrolling) { setT("topClock", fmtNowClock()); }
}, 1000);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    disconnectDashboardEvents();
    return;
  }
  connectDashboardEvents();
  tick();
});
setupSnapshotAssetTabs();
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
