const TRADE_JOURNAL_URL = "../../data/live/trade_journal.jsonl";
const API_EVENTS_URL = "/api/events";
const API_TRADES_URL = "/api/trades";
const API_OPS_STATUS_URL = "/api/ops-status";
const API_BTC_MULTISLOT_SHADOW_URL = "/api/btc-multislot-shadow";
const API_ETH_ODYSSEY4_SHADOW_URL = "/api/eth-odyssey4-shadow";
const API_EVIDENCE_SIGNALS_URL = "/api/evidence-signals";
const API_EVIDENCE_SIGNALS_PROVISIONAL_URL = "/api/evidence-signals-provisional";
const API_V_REBOUND_URL = "/api/v-rebound-signal";
const API_BASIS_LIQUIDATION_URL = "/api/basis-liquidation-signal";
const API_LIQUIDATION_DIRECTION_URL = "/api/liquidation-direction-signal";
const API_LIQUIDATION_MAP_URL = "/api/liquidation-map";
const API_REGIME_WIDE24_URL = "/api/regime-wide24";
const API_MACRO_CALENDAR_URL = "/api/macro-calendar";
const API_LIQ_BURST_STATE_URL = "/api/liq-burst-state";
const API_LIQUIDATION_5M_URL = "/api/liquidation-5m-signal";
const API_SESSION_ALERTS_URL = "/api/session-alerts";
const POLL_MS = 2500;
const CHART_RENDER_MIN_INTERVAL_MS = 5000;
// 2026-08-25 perf pass: split from CHART_RENDER_MIN_INTERVAL_MS -- the Snapshot chart's own data
// (latestLiquidationMap) only changes once per LIQUIDATION_MAP_POLL_MS (5min), so redrawing its
// ~250-400 SVG nodes every 5s (60x more often than the data changes) was pure waste. Coarser than
// the Live chart's interval on purpose: this chart is read-only reference, not something a user
// pans/zooms live (see renderSnapshotChart()'s own comment). Kept as ONE throttle interval (not
// split further into "redraw candles at 5s, density at 5min") because the density band's sweep-
// darkening depends on live candle highs/lows, not just latestLiquidationMap -- decoupling those
// two redraw cadences would let a real price sweep sit un-darkened for up to 5min, undermining why
// sweep-darkening exists. Slower-but-synchronized beats faster-but-inconsistent here.
const SNAPSHOT_CHART_RENDER_MIN_INTERVAL_MS = 20000;
const JOURNAL_POLL_MS = 10000;
const CANDLE_HISTORY_POLL_MS = 300000;
const MICRO_HISTORY_MAX = 48; // matches MODEL_INDICATOR_HISTORY_MAX in server.py (4h @ 5min samples)
const ASSET_CONFIG = {
  eth: { label: "ETH", symbol: "ETHUSDT", accountSymbol: "ETH/USDT:USDT", priceDigits: 2 },
  sol: { label: "SOL", symbol: "SOLUSDT", accountSymbol: "SOL/USDT:USDT", priceDigits: 3 },
  btc: { label: "BTC", symbol: "BTCUSDT", accountSymbol: "BTC/USDT:USDT", priceDigits: 1 },
};
const ASSET_KEYS = Object.keys(ASSET_CONFIG);
const DEFAULT_ASSET = "eth";

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

let latestState = null;
let latestMainState = null;
let latestCompactState = null;
let latestTradeJournal = [];
let latestTradeEquitySeries = [];
let latestLivePrice = 0;
let latestLivePriceTs = "";
let lastJournalFetchAt = 0;
let tradeJournalLoaded = false;
let tradePanelsRendered = false;
let tickInFlight = false;
let tradesEtag = "";
let latestChartRiskLevels = [];
let activeChartAsset = DEFAULT_ASSET;
let latestLivePriceByAsset = {};
let latestLivePriceTsByAsset = {};
let candleHistoryByAsset = {};
let lastCandleHistoryFetchAtByAsset = {};
let opsStatusEtag = "";
let opsLastFetchAt = 0;
let btcMultislotEtag = "";
let btcMultislotLastFetchAt = 0;
let latestBtcMultislotPayload = null;
let btcMultislotActiveSlot = 0;
let ethOdyssey4Etag = "";
let ethOdyssey4LastFetchAt = 0;
let latestEthOdyssey4Payload = null;
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
let liquidationMapLastFetchAt = 0;
// Snapshot tab's own coin selector (2026-08-31, BTC added) -- deliberately separate from
// activeChartAsset (the Live tab's chart asset, which the Snapshot tab has never followed -- see
// the comment on lastSnapshotHistoryFetchAt above). Only backs the 4 signals server.py now accepts
// an ?asset= for (basis liquidation / liquidation direction / liquidation 5m / liquidation map,
// plus that map's own candle chart) -- 증거신호/레짐/특화감지기/수급흐름/리테일수급/청산캐스케이드
// stay ETH-only regardless of this (see docs/eth_dashboard_multicoin_expansion_design_20260831.md
// section 6.4 for why: those are trained-model or trading_bot.py-sourced, not a symbol swap away).
let activeSnapshotAsset = "eth";
const SNAPSHOT_ASSET_KEYS = ["eth", "btc"];
let regimeWide24LastFetchAt = 0;
let macroCalendarLastFetchAt = 0;
let sessionAlertsLastFetchAt = 0;
let lastSnapshotHistoryFetchAt = 0;
let lastChartRenderAt = 0;
let lastSnapshotChartRenderAt = 0;
let lastModelIndicatorHtmlByTarget = {};
let activePageTab = "snapshot"; // "live" | "ops" | "snapshot" -- must match index.html's default active tab (data-page-tab="snapshot" carries the initial "active" class)
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
let candleHistory = []; // Array of {time, open, high, low, close}
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

function activeAssetConfig() {
  return ASSET_CONFIG[activeChartAsset] || ASSET_CONFIG[DEFAULT_ASSET];
}

function assetLabel(asset = activeChartAsset) {
  return (ASSET_CONFIG[asset] || ASSET_CONFIG[DEFAULT_ASSET]).label;
}

function syncActiveMarketState() {
  candleHistory = candleHistoryByAsset[activeChartAsset] || [];
  latestLivePrice = Number(latestLivePriceByAsset[activeChartAsset] || 0);
  latestLivePriceTs = String(latestLivePriceTsByAsset[activeChartAsset] || "");
}

function normalizeAssetKey(value) {
  const s = String(value || "").toLowerCase();
  if (s.includes("sol")) return "sol";
  if (s.includes("btc")) return "btc";
  return "eth";
}

function tradeAssetKey(row) {
  const basis = [
    row?.symbol,
    row?.account_symbol,
    row?.execution_symbol,
    row?.asset,
    row?.market,
    row?.raw_source,
    row?.source,
  ].filter(Boolean).join(" ");
  return normalizeAssetKey(basis);
}

function chartJournalRows() {
  return (latestTradeJournal || []).filter((row) => tradeAssetKey(row) === activeChartAsset);
}

function renderAssetTabs() {
  // 2026-08-31: scoped to #assetTabs (was a bare ".asset-tab" query) -- the Snapshot tab's own
  // coin switcher (#snapshotAssetTabs, see renderSnapshotAssetTabs()) reuses the same "asset-tab"
  // CSS class for identical styling but must stay driven by activeSnapshotAsset, not this
  // Live-tab-chart-only activeChartAsset. No behavior change for #assetTabs itself.
  document.querySelectorAll("#assetTabs .asset-tab").forEach((btn) => {
    const asset = normalizeAssetKey(btn.dataset.asset);
    btn.classList.toggle("active", asset === activeChartAsset);
  });
}

async function setActiveChartAsset(asset) {
  const next = normalizeAssetKey(asset);
  if (!ASSET_CONFIG[next] || next === activeChartAsset) return;
  activeChartAsset = next;
  mobileChartView.start = null;
  mobileChartView.followLatest = true;
  latestChartRiskLevels = [];
  setT("riskLevelNote", "-");
  syncActiveMarketState();
  renderAssetTabs();
  await fetchBinanceHistory(activeChartAsset);
  lastChartRenderAt = 0;
  render(latestMainState || latestState || {}, latestCompactState);
}

function setupAssetTabs() {
  document.querySelectorAll("#assetTabs .asset-tab").forEach((btn) => {
    btn.addEventListener("click", () => setActiveChartAsset(btn.dataset.asset));
  });
  renderAssetTabs();
}

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
  await Promise.all([
    refreshBasisLiquiditySignal(),
    refreshLiquidationDirectionSignal(),
    refreshLiquidation5mSignal(),
    refreshLiquidationMap(),
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
const mobileChartGesture = {
  panStartX: 0,
  panStartIndex: 0,
  pinchStartDistance: 0,
  pinchStartSize: MOBILE_CHART_DEFAULT_CANDLES,
  pinchStartCenter: 0,
};

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

function qualityText(score, threshold) {
  const s = Number(score);
  if (!Number.isFinite(s)) return "-";
  const t = Number(threshold);
  return Number.isFinite(t) ? `${s.toFixed(3)} / ${t.toFixed(2)}` : s.toFixed(3);
}

function rowTs(row) {
  const raw = row?.closed_at || row?.ts || row?.opened_at || "";
  const ms = Date.parse(raw);
  return Number.isFinite(ms) ? ms : 0;
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

function tsAgeSec(v) {
  if (!v) return null;
  const ms = Date.parse(v);
  if (!Number.isFinite(ms)) return null;
  return Math.max(0, Math.round((Date.now() - ms) / 1000));
}

function isToday(ms) {
  if (!Number.isFinite(ms) || ms <= 0) return false;
  const d = new Date(ms);
  const now = new Date();
  return d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth() && d.getDate() === now.getDate();
}

function isFinalGovernorState(state) {
  if (!state) return false;
  const src = String(state?.signal?.source || state?.agents?.governor?.source || "").toLowerCase();
  return Boolean(
    state.governor_mode ||
    state?.agents?.governor ||
    src.includes("final_governor") ||
    src.includes("sniper") ||
    src.includes("trend") ||
    src.includes("micro") ||
    src.includes("cash")
  );
}

function usableGovernorShadowState(state) {
  if (!isFinalGovernorState(state)) return null;
  const age = tsAgeSec(state?.microstructure?.updated_at || state?.updated_at || state?.cycle_timestamp_kst);
  if (Number.isFinite(age) && age > 180) return null;
  return state;
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

function actionLabel(a) {
  if (a === 1) return { text: "롱", icon: "▲", cls: "long" };
  if (a === 2) return { text: "숏", icon: "▼", cls: "short" };
  return { text: "대기", icon: "⏸", cls: "hold" };
}

function sideLabel(value) {
  const s = String(value || "NONE").toUpperCase();
  if (s === "LONG") return "롱";
  if (s === "SHORT") return "숏";
  if (s === "NONE") return "없음";
  return s;
}

function openPosition(state) {
  const pos = (state || {}).position || {};
  const side = String(pos.current || "NONE").toUpperCase();
  const entryPrice = Number(pos.entry_price || (state || {}).entry_price || 0);
  if ((side === "LONG" || side === "SHORT") && entryPrice > 0) {
    return { ...pos, current: side, entry_price: entryPrice };
  }
  return null;
}

function chartPositionState(mainState, compactState) {
  const shadowState = usableGovernorShadowState(compactState);
  if (openPosition(shadowState)) return shadowState;
  if (openPosition(mainState)) return mainState;
  return shadowState || mainState;
}

function assetDecisionState(mainState, compactState, asset = activeChartAsset) {
  const key = normalizeAssetKey(asset);
  if (key === "eth") return chartPositionState(mainState, compactState);
  const sources = [compactState, mainState].filter(Boolean);
  for (const src of sources) {
    const direct =
      src?.assets?.[key] ||
      src?.asset_decisions?.[key] ||
      src?.asset_states?.[key] ||
      src?.market_assets?.[key];
    if (direct) return direct;
    const upper = key.toUpperCase();
    const upperDirect =
      src?.assets?.[upper] ||
      src?.asset_decisions?.[upper] ||
      src?.asset_states?.[upper] ||
      src?.market_assets?.[upper];
    if (upperDirect) return upperDirect;
  }
  return null;
}

function strategyTagFromRow(row) {
  const src = String(row?.source || "").toUpperCase();
  const raw = String(row?.raw_source || "").toUpperCase();
  const basis = `${src} ${raw}`;
  if (basis.includes("SNIPER")) return "SNIPER";
  if (basis.includes("TREND")) return "TREND";
  if (basis.includes("MICRO") || basis.includes("WNC")) return "MICRO";
  if (basis.includes("GOVERNOR") || basis.includes("CASH")) return "GOVERNOR";
  if (basis.includes("COMPACT")) return "COMPACT";
  if (basis.includes("CONTROLLER")) return "CONTROLLER";
  return "GOVERNOR";
}

function strategyDisplayLabel(value) {
  const s = String(value || "").toUpperCase();
  if (s === "SNIPER") return "스나이퍼";
  if (s === "TREND") return "추세";
  if (s === "MICRO") return "마이크로";
  if (s === "GOVERNOR") return "거버너";
  if (s === "COMPACT") return "컴팩트";
  if (s === "CONTROLLER") return "컨트롤러";
  if (s.includes("SNIPER")) return "스나이퍼";
  if (s.includes("TREND")) return "추세";
  if (s.includes("MICRO") || s.includes("WNC")) return "마이크로";
  if (s.includes("GOVERNOR")) return "거버너";
  if (s.includes("COMPACT")) return "컴팩트";
  if (s.includes("CONTROLLER")) return "컨트롤러";
  return value || "-";
}

function normalizeModelVersion(value) {
  const raw = String(value || "").trim();
  if (!raw) return "";
  const compact = raw.replace(/_/g, ".").replace(/\s+/g, "");
  const vMatch = compact.match(/v?(\d+(?:\.\d+)*)/i);
  return vMatch ? `v${vMatch[1]}` : compact.toLowerCase();
}

function tradeGovernorLabel(row) {
  const basis = [
    row?.model_version,
    row?.model_id,
    row?.decision_logic,
    row?.raw_source,
    row?.source,
  ].filter(Boolean).join(" ");
  const alphaBasis = basis.toLowerCase();
  if (
    alphaBasis.includes("alpha4_3_sticky") ||
    alphaBasis.includes("alpha4.3 sticky") ||
    alphaBasis.includes("sticky_current")
  ) {
    return "Alpha4.3 Sticky";
  }
  if (
    alphaBasis.includes("alpha3") ||
    alphaBasis.includes("alpha2_1") ||
    alphaBasis.includes("alpha2.1") ||
    alphaBasis.includes("alpha2 1")
  ) {
    return "Alpha3";
  }
  let version = normalizeModelVersion(row?.model_version);
  if (!version && /v22[_\s.]?1/i.test(basis)) version = "v22.1";
  if (!version && /v21/i.test(basis)) version = "v21";
  const sleeve = strategyDisplayLabel(strategyTagFromRow(row));
  const base = version ? `거버너 ${version}` : "거버너";
  return sleeve && sleeve !== "거버너" ? `${base} · ${sleeve}` : base;
}

function cleanTradeReason(value) {
  const raw = cleanDisplaySource(value || "");
  if (!raw || raw === "-") return "";
  return raw
    .replace(/^alpha2[._-]?1\s*[|:]/i, "")
    .replace(/^lifecycle_v1\s*[|:]/i, "")
    .replace(/^v31_deep_alpha_/i, "V31 ")
    .replace(/^learned_/i, "")
    .replace(/_/g, " ")
    .trim();
}

function closeReasonText(row) {
  return cleanTradeReason(
    row?.exit_reason ||
    row?.close_reason ||
    row?.reason ||
    row?.source ||
    ""
  );
}

function liquidityLabel(value) {
  const s = String(value || "").toLowerCase();
  if (s === "maker_miss" || s.includes("maker_miss")) return "maker miss";
  if (s.includes("maker") && s.includes("dry_run") && s.includes("taker")) return "maker→taker shadow";
  if (s.includes("maker") && s.includes("dry_run")) return "maker shadow";
  if (s.includes("taker") && s.includes("dry_run")) return "taker shadow";
  if (s === "maker_taker" || s === "maker+taker" || s === "maker-taker" || s === "mixed") return "maker→taker";
  if (s.includes("maker") && s.includes("taker")) return "maker→taker";
  if (s.includes("maker")) return "maker";
  if (s.includes("taker") || s.includes("market")) return "taker";
  if (s.includes("synthetic") || s.includes("shadow")) return "shadow model";
  return "";
}

function executionLegLiquidity(row, leg) {
  const prefix = leg === "exit" ? "exit" : "entry";
  const direct = liquidityLabel(
    row?.[`${prefix}_execution_liquidity`] ||
    row?.[`${prefix}_liquidity`] ||
    row?.[`${prefix}_exec_liquidity`] ||
    row?.[`${prefix}_order_liquidity`]
  );
  if (direct) return direct;

  const route = liquidityLabel(row?.[`${prefix}_execution_route`] || row?.[`${prefix}_execution_order_type`]);
  if (route) return route;

  const kind = String(row?.[`${prefix}_exec_price_kind`] || "").toLowerCase();
  if (kind.includes("synthetic_fee_slippage_model")) return "shadow model";
  if (kind.includes("maker")) return "maker";
  if (kind.includes("taker") || kind.includes("market")) return "taker";
  return "";
}

function priceWithLiquidity(label, price, liquidity) {
  const liq = liquidity ? ` <span class="trade-journal-liquidity">${liquidity}</span>` : "";
  return `${label} ${fmtNum(price, 2)}${liq}`;
}

function feeModelText(row) {
  const model = String(row?.fee_model || "").replaceAll("_", " ");
  const entry = Number(row?.entry_fee_rate);
  const exit = Number(row?.exit_fee_rate);
  if (!model && !Number.isFinite(entry) && !Number.isFinite(exit)) return "";
  const entryBps = Number.isFinite(entry) ? `${(entry * 10000).toFixed(2)}bp` : "-";
  const exitBps = Number.isFinite(exit) ? `${(exit * 10000).toFixed(2)}bp` : "-";
  return `수수료: ${model || "unknown"} (${entryBps}/${exitBps})`;
}

function closeTradeRows(filter) {
  let rows = (latestTradeJournal || []).filter((row) => String(row?.kind || "").toUpperCase() === "CLOSE");
  if (filter && filter !== "ALL") rows = rows.filter((row) => strategyTagFromRow(row) === filter);
  return rows.slice().sort((a, b) => rowTs(a) - rowTs(b));
}

function pnlPctFromRow(row) {
  if (Number.isFinite(Number(row?.pnl_pct))) return Number(row.pnl_pct);
  if (Number.isFinite(Number(row?.pnl_frac))) return Number(row.pnl_frac) * 100;
  return 0;
}

function executionDelaySec(row) {
  const n = Number(row?.execution_delay_sec);
  return Number.isFinite(n) ? n : null;
}

function executionTimingText(row) {
  const decisionTs = row?.decision_bar_ts || row?.decision_at || "";
  const execTs = row?.execution_bar_ts || row?.ts || row?.closed_at || row?.opened_at || "";
  const delay = executionDelaySec(row);
  const source = row?.execution_price_source || row?.entry_price_source || row?.exit_price_source || "";
  const ledger = row?.ledger_ts_kind || "";
  const parts = [];
  if (decisionTs) parts.push(`신호 ${fmtShortTs(decisionTs)}`);
  if (execTs) parts.push(`체결 ${fmtShortTs(execTs)}`);
  if (delay !== null) parts.push(`지연 ${fmtNum(delay, 1)}s`);
  if (source) parts.push(String(source));
  if (ledger) parts.push(String(ledger));
  return parts.join(" · ");
}

function aiTimingText(row) {
  const timing = row?.ai_timing || {};
  if (!timing || typeof timing !== "object") return "";
  const total = Number(timing.total_sec ?? timing.total?.sec ?? timing.predict_all?.sec ?? timing.predict_all_sec);
  return Number.isFinite(total) && total > 0 ? `AI ${fmtNum(total, 2)}s` : "";
}

function buildTradeEquitySeries(filter) {
  let equity = 1;
  return closeTradeRows(filter).map((row, idx) => {
    const pnlPct = pnlPctFromRow(row);
    equity *= 1 + pnlPct / 100;
    return {
      ...row,
      chart_index: idx + 1,
      pnl_pct: pnlPct,
      equity,
      cumulative_return_pct: (equity - 1) * 100,
      ts: row.closed_at || row.ts,
    };
  });
}

function normalizeStateTradeTail(rows) {
  return (rows || [])
    .filter((row) => row && Number.isFinite(Number(row.equity)))
    .map((row, idx) => ({
      ...row,
      chart_index: idx + 1,
      pnl_pct: pnlPctFromRow(row),
      equity: Number(row.equity),
      cumulative_return_pct: (Number(row.equity) - 1) * 100,
    }));
}

function selectTradeRowsForCharts(filter) {
  if (latestTradeEquitySeries.length) return latestTradeEquitySeries;
  const journalSeries = buildTradeEquitySeries(filter);
  if (journalSeries.length) return journalSeries;
  const stateRows = latestMainState?.trades_tail || latestCompactState?.trades_tail;
  return normalizeStateTradeTail(stateRows || []);
}

function firstPositive(...vals) {
  for (const v of vals) {
    const n = Number(v);
    if (Number.isFinite(n) && n > 0) return n;
  }
  return 0;
}

function signedRiskPairLabel(tp, sl) {
  const parts = [];
  const tpN = Number(tp);
  const slN = Number(sl);
  if (Number.isFinite(tpN) && tpN > 0) parts.push(fmtPct(tpN * 100, 1));
  if (Number.isFinite(slN) && slN > 0) parts.push(`-${fmtPctNoPlus(slN * 100, 1)}`);
  return parts.join("/");
}

function signedPriceMovePairLabel(tpMove, slMove) {
  const parts = [];
  const tpN = Number(tpMove);
  const slN = Number(slMove);
  if (Number.isFinite(tpN) && tpN > 0) parts.push(fmtPctNoPlus(tpN * 100, 1));
  if (Number.isFinite(slN) && slN > 0) parts.push(fmtPctNoPlus(slN * 100, 1));
  return parts.join("/");
}

function riskSummaryText(row) {
  const tp = firstPositive(row?.effective_take_profit, row?.take_profit);
  const sl = firstPositive(row?.effective_stop_loss, row?.stop_loss);
  const exposure = firstPositive(row?.total_exposure, row?.notional_exposure);
  const parts = [];
  const riskPair = signedRiskPairLabel(tp, sl);
  if (riskPair) parts.push(`계정 ${riskPair}`);
  const priceMovePair = exposure > 0 ? signedPriceMovePairLabel(tp / exposure, sl / exposure) : "";
  if (priceMovePair) parts.push(`가격 ${priceMovePair}`);
  return parts.join(" · ");
}

function thresholdPrice(side, entry, rawMove, takeProfit) {
  const sideU = String(side || "").toUpperCase();
  const entryN = Number(entry);
  const moveN = Number(rawMove);
  if (!(entryN > 0) || !(moveN > 0) || !["LONG", "SHORT"].includes(sideU)) return 0;
  if (sideU === "LONG") return takeProfit ? entryN * (1 + moveN) : entryN * Math.max(0, 1 - moveN);
  return takeProfit ? entryN * Math.max(0, 1 - moveN) : entryN * (1 + moveN);
}

function activeRiskModel(state, compactState, selectedState = null) {
  const s = selectedState || chartPositionState(state, compactState) || {};
  const alt = selectedState
    ? (activeChartAsset === "eth" ? (s === state ? (compactState || {}) : (state || {})) : {})
    : (s === state ? (compactState || {}) : (state || {}));
  const pos = s.position || {};
  const sig = s.signal || {};
  const altPos = alt.position || {};
  const altSig = alt.signal || {};
  const agents = s.agents || {};
  const lifecycle = agents.lifecycle_v1 || {};
  const omega = agents.omega1_2_1 || {};
  const omega461 = agents.omega4_6_1 || {};
  const fullyLearned = agents.fully_learned || {};
  const macro = agents.macro || {};
  const trace = sig.sleeve_trace || {};
  const v31 = trace.v31 || {};
  const cfg = v31.selected_config || {};
  const side = String(pos.current || altPos.current || "").toUpperCase();
  const entry = Number(pos.entry_price || altPos.entry_price || s.entry_price || alt.entry_price || 0);
  const exposure = Number(pos.total_exposure ?? pos.notional_exposure ?? sig.notional_exposure ?? sig.unified_kelly ?? altPos.total_exposure ?? altPos.notional_exposure ?? altSig.notional_exposure ?? altSig.unified_kelly ?? 0);
  if (!["LONG", "SHORT"].includes(side) || !(entry > 0) || !(exposure > 0)) {
    return { tp: 0, sl: 0, maxHold: 0, remaining: NaN, tpPrice: 0, slPrice: 0 };
  }
  // pos.effective_take_profit / pos.take_profit are raw price-move fractions (e.g. 0.075 = 7.5%
  // price move), not account-level PnL fractions -- see the Futures Risk Sizing Contract
  // (PnL = price_move * notional). Account-level threshold = rawMove * exposure, not rawMove / exposure.
  const rawTp = firstPositive(pos.effective_take_profit, pos.take_profit, sig.effective_take_profit, sig.take_profit, altPos.effective_take_profit, altPos.take_profit, altSig.effective_take_profit, altSig.take_profit, v31.effective_tp, omega461.active_take_profit, omega.active_take_profit, lifecycle.active_take_profit, fullyLearned.active_take_profit, macro.active_take_profit, macro.take_profit);
  const rawSl = firstPositive(pos.effective_stop_loss, pos.stop_loss, sig.effective_stop_loss, sig.stop_loss, altPos.effective_stop_loss, altPos.stop_loss, altSig.effective_stop_loss, altSig.stop_loss, v31.effective_sl, omega461.active_stop_loss, omega.active_stop_loss, lifecycle.active_stop_loss, fullyLearned.active_stop_loss, macro.active_stop_loss, macro.stop_loss);
  const maxHold = 0;
  const rem = Number(pos.max_hold_remaining_bars ?? sig.max_hold_remaining_bars ?? altPos.max_hold_remaining_bars ?? altSig.max_hold_remaining_bars);
  const tpPrice = firstPositive(pos.take_profit_price, pos.tp_price, sig.take_profit_price, sig.tp_price, altPos.take_profit_price, altPos.tp_price, altSig.take_profit_price, altSig.tp_price, thresholdPrice(side, entry, rawTp, true));
  const slPrice = firstPositive(pos.stop_price, pos.sl_price, sig.stop_price, sig.sl_price, altPos.stop_price, altPos.sl_price, altSig.stop_price, altSig.sl_price, thresholdPrice(side, entry, rawSl, false));
  return {
    tp: rawTp > 0 ? rawTp * exposure : 0,
    sl: rawSl > 0 ? rawSl * exposure : 0,
    maxHold,
    remaining: rem,
    tpPrice,
    slPrice,
    exposure,
    tpMove: rawTp,
    slMove: rawSl,
  };
}

function chartRiskLevels(state, compactState, selectedState = null) {
  const s = selectedState || chartPositionState(state, compactState) || {};
  if (!openPosition(s)) return [];
  const pos = s.position || {};
  const sig = s.signal || {};
  const pb = s.playbook || {};
  const activeRisk = activeRiskModel(state, compactState, selectedState);
  const out = [];
  const stop = firstPositive(pos.stop_price, sig.stop_price, activeRisk.slPrice, pb.stop_price, pb.trailing_stop_price, sig.trailing_stop_price);
  const tp = firstPositive(pos.take_profit_price, pos.tp_price, sig.take_profit_price, sig.tp_price, activeRisk.tpPrice, pb.take_profit_price);
  const trail = firstPositive(pos.trailing_stop_price, sig.trailing_stop_price, pb.trailing_stop_price);
  if (stop) out.push({ val: stop, color: "#cf6a5c", label: "손절", dashed: false, width: 2 });
  if (tp) out.push({ val: tp, color: "#6bab84", label: "익절", dashed: false, width: 2 });
  if (trail && trail !== stop) out.push({ val: trail, color: "#c48ca8", label: "추적", dashed: true, width: 2 });
  return out;
}

function chartEntryLiquidity(state, compactState) {
  const s = chartPositionState(state, compactState) || {};
  const alt = s === state ? (compactState || {}) : (state || {});
  const pos = s.position || {};
  const sig = s.signal || {};
  const altPos = alt.position || {};
  const altSig = alt.signal || {};
  const route = liquidityLabel(
    pos.entry_execution_route ||
    pos.entry_execution_liquidity ||
    sig.entry_execution_route ||
    sig.entry_execution_liquidity ||
    s.entry_execution_route ||
    altPos.entry_execution_route ||
    altSig.entry_execution_route ||
    ""
  );
  if (route === "maker") return "지정가";
  if (route === "taker") return "시장가";
  if (route === "maker→taker") return "지정가→시장가";
  const priceSource = String(pos.entry_price_source || sig.entry_price_source || s.entry_price_source || altPos.entry_price_source || altSig.entry_price_source || "").toLowerCase();
  if (priceSource.includes("next_bar_open")) return "다음봉 체결";
  if (priceSource.includes("synthetic")) return "모의 체결";
  if (Number(pos.entry_price || altPos.entry_price || 0) > 0) return "원장 체결";
  return route || "-";
}

function assetUnrealizedPnl(state, compactState, asset) {
  const selectedAssetState = assetDecisionState(state, compactState, asset);
  const active = selectedAssetState || usableGovernorShadowState(compactState) || state || {};
  const pos = active.position || {};
  const sig = active.signal || state?.signal || {};
  const posSide = String(pos.current || "NONE").toUpperCase();
  if (posSide !== "LONG" && posSide !== "SHORT") return { pnlPct: 0, posSide: "NONE" };
  const pnlPct = Number(pos.unrealized_pnl_pct ?? sig.position_unrealized_pnl_pct ?? active.position_unrealized_pnl_pct ?? 0);
  return { pnlPct, posSide };
}

// Collapses the live governor's 5-way RegimeEngine label (bull/bear/chop/whipsaw/normal, see
// features/elite.py::RegimeEngine) into the 3 buckets that matter for a discretionary fade/follow
// call: 추세(bull+bear), 안정횡보(chop only), 불안정·전환구간(whipsaw+normal -- neither a clean
// trend nor a clean quiet range, the condition most likely to break out against a fade).
function liveRegimeLabel(raw) {
  const r = String(raw || "").toLowerCase();
  if (r === "bull" || r === "bear") return { bucket: "trend", label: "추세", tone: "" };
  if (r === "chop") return { bucket: "chop", label: "안정횡보", tone: "" };
  if (r === "whipsaw" || r === "normal") return { bucket: "unstable", label: "불안정·전환구간", tone: "warn-text" };
  return { bucket: "other", label: String(raw || "-").toUpperCase(), tone: "" };
}

// A single instant read of "trend" can be boundary flicker (RegimeEngine re-evaluates every
// render tick, not just once per bar). Only call a 횡보->추세 change trustworthy enough to act on
// once it has held continuously (per asset, wall-clock) for REGIME_TREND_CONFIRM_MS -- resets the
// moment the bucket leaves "trend". Client-side only: a page reload forgets the streak and starts
// re-confirming from 0, so a stale "확인중" right after opening the tab is expected, not a bug.
const REGIME_TREND_CONFIRM_MS = 15 * 60 * 1000;
const regimeTrendStreakSince = {};
function regimeConfirmState(bucket, assetKey) {
  if (bucket !== "trend") {
    regimeTrendStreakSince[assetKey] = null;
    return { suffix: "", tone: "" };
  }
  if (!regimeTrendStreakSince[assetKey]) regimeTrendStreakSince[assetKey] = Date.now();
  const elapsedMs = Date.now() - regimeTrendStreakSince[assetKey];
  const confirmMin = Math.round(REGIME_TREND_CONFIRM_MS / 60000);
  if (elapsedMs >= REGIME_TREND_CONFIRM_MS) return { suffix: " 확정", tone: "good-text" };
  return { suffix: ` (확인중 ${Math.floor(elapsedMs / 60000)}/${confirmMin}분)`, tone: "" };
}

// Sums each asset's account-level unrealized PnL (already notional-scaled, all sleeves of the
// same account equity per docs/model_contracts -- see 3-asset portfolio design), independent of
// whichever asset tab is currently active in the chart above.
function renderCombinedUnrealizedPnl(state, compactState) {
  let total = 0;
  let openCount = 0;
  const parts = [];
  ASSET_KEYS.forEach((asset) => {
    const { pnlPct, posSide } = assetUnrealizedPnl(state, compactState, asset);
    if (posSide === "LONG" || posSide === "SHORT") {
      total += pnlPct;
      openCount += 1;
      parts.push(`${assetLabel(asset)} ${posSide === "LONG" ? "롱" : "숏"} ${fmtPct(pnlPct, 2)}`);
    }
  });
  setT("heroUnrealizedPnl", openCount > 0 ? fmtPct(total, 2) : "-");
  const heroUnrealEl = el("heroUnrealizedPnl");
  if (heroUnrealEl) {
    heroUnrealEl.classList.remove("good-text", "bad-text", "muted-text");
    heroUnrealEl.classList.add(`${openCount > 0 ? riskClass(total) : "muted"}-text`);
  }
  setT("heroUnrealizedSub", openCount > 0 ? parts.join(" · ") : "포지션 없음");
}

function renderOpsCards(state, compactState) {
  const selectedAssetState = assetDecisionState(state, compactState, activeChartAsset);
  if (!selectedAssetState) {
    const cfg = activeAssetConfig();
    const price = Number(latestLivePriceByAsset[activeChartAsset] || 0);
    setT("chartDecisionText", "추적");
    setT("chartEntryLiquidityText", "상태 없음");
    setT("chartPositionPctText", "-");
    setT("chartEntryText", "-");
    setT("chartEntryTimeText", "-");
    setT("chartExposureText", "-");
    setT("chartUnrealizedPnlText", "-");
    setT("chartRiskText", `${cfg.label} 모델 상태 대기`);
    setT("chartRegimeText", price > 0 ? fmtNum(price, cfg.priceDigits) : "-");
    const regimeEl = el("chartRegimeText");
    if (regimeEl) regimeEl.classList.remove("warn-text", "good-text");
    regimeTrendStreakSince[activeChartAsset] = null;
    const riskEl = el("chartRiskText");
    if (riskEl) riskEl.title = `${cfg.accountSymbol} decision state not present in /api/state`;
    const unrealizedEl = el("chartUnrealizedPnlText");
    if (unrealizedEl) {
      unrealizedEl.classList.remove("good-text", "bad-text", "muted-text");
      unrealizedEl.classList.add("muted-text");
    }
    const ribbon = el("chartAiRibbon");
    if (ribbon) ribbon.className = "chart-ai-ribbon hold";
    setT("chartAiSummaryText", `${cfg.label} 모델 상태 대기 중`);
    const gaugeEl0 = el("chartRiskGauge");
    if (gaugeEl0) gaugeEl0.innerHTML = "";
    return;
  }
  const active = selectedAssetState || usableGovernorShadowState(compactState) || state || {};
  const pos = active.position || {};
  const sig = active.signal || state?.signal || {};
  const agents = active.agents || state?.agents || {};
  const gov = agents.governor || {};
  const decision = actionLabel(Number(sig.final_action ?? sig.rl_action ?? 0));
  const exposure = Number(pos.total_exposure ?? pos.notional_exposure ?? gov.notional_exposure ?? sig.notional_exposure ?? pos.position_fraction ?? 0);
  const leverage = Number(pos.execution_leverage ?? gov.execution_leverage ?? sig.execution_leverage ?? 1);
  const positionPct = Number(pos.position_fraction ?? sig.position_fraction ?? 0) * 100;
  const regimeRaw = String(sig.governor_regime || gov.regime || active.regime || "-");
  const regimeInfo = liveRegimeLabel(regimeRaw);
  const regimeConfirm = regimeConfirmState(regimeInfo.bucket, activeChartAsset);
  const entryPrice = Number(pos.entry_price || active.entry_price || 0);
  const entryTime = pos.opened_at || pos.decision_at || active.opened_at || "";
  const unrealizedPnl = Number(pos.unrealized_pnl_pct ?? sig.position_unrealized_pnl_pct ?? active.position_unrealized_pnl_pct ?? 0);
  const risk = activeRiskModel(state, compactState, active);
  const tp = risk.tp;
  const sl = risk.sl;
  const tpPrice = risk.tpPrice;
  const slPrice = risk.slPrice;
  const priceMovePair = signedPriceMovePairLabel(risk.tpMove, risk.slMove);

  setT("chartDecisionText", decision.text);
  setT("chartEntryLiquidityText", chartEntryLiquidity(state, compactState));
  setT("chartPositionPctText", fmtPctNoPlus(positionPct, 1));
  setT("chartEntryText", entryPrice > 0 ? fmtNum(entryPrice, 2) : "-");
  setT("chartEntryTimeText", entryPrice > 0 && entryTime ? fmtShortTs(entryTime) : "-");
  setT("chartExposureText", `${fmtNum(leverage, 2)}x / ${fmtNum(exposure, 2)}x`);
  setT("chartUnrealizedPnlText", String(pos.current || "NONE").toUpperCase() === "NONE" ? "-" : fmtPct(unrealizedPnl, 2));
  setT("chartRiskText", signedRiskPairLabel(tp, sl) ? `계정 ${signedRiskPairLabel(tp, sl)}${priceMovePair ? ` · 가격 ${priceMovePair}` : ""}` : "-");
  setT("chartRegimeText", regimeInfo.label + regimeConfirm.suffix);
  const regimeEl = el("chartRegimeText");
  if (regimeEl) {
    regimeEl.classList.remove("warn-text", "good-text");
    const tone = regimeConfirm.tone || regimeInfo.tone;
    if (tone) regimeEl.classList.add(tone);
    regimeEl.title = regimeRaw !== "-" ? `세부 레짐: ${regimeRaw.toUpperCase()}` : "";
  }
  const riskEl = el("chartRiskText");
  if (riskEl) {
    const detail = [
      risk.exposure > 0 ? `exposure ${fmtNum(risk.exposure, 2)}x` : "",
      tpPrice > 0 ? `TP ${fmtNum(tpPrice, 2)}` : "",
      slPrice > 0 ? `SL ${fmtNum(slPrice, 2)}` : "",
    ].filter(Boolean).join(" / ");
    riskEl.title = detail || "";
  }
  const unrealizedEl = el("chartUnrealizedPnlText");
  const posSide = String(pos.current || "NONE").toUpperCase();
  if (unrealizedEl) {
    unrealizedEl.classList.remove("good-text", "bad-text", "muted-text");
    const unrealizedClass = posSide === "NONE" ? "muted" : riskClass(unrealizedPnl);
    unrealizedEl.classList.add(`${unrealizedClass}-text`);
  }
  const ribbon = el("chartAiRibbon");
  if (ribbon) ribbon.className = `chart-ai-ribbon ${decision.cls === "long" ? "good" : decision.cls === "short" ? "bad" : "hold"}`;

  // AI decision one-line summary
  const currentPrice = Number(latestLivePriceByAsset[activeChartAsset] || active.last_price || active.price || 0);
  let summaryText;
  if (posSide === "LONG" || posSide === "SHORT") {
    const sideKr = posSide === "LONG" ? "롱" : "숏";
    const parts = [`${sideKr} ${fmtPctNoPlus(positionPct, 0)} 비중 보유`, `미실현 ${fmtPct(unrealizedPnl, 2)}`];
    if (currentPrice > 0 && tpPrice > 0) {
      const distTp = posSide === "LONG" ? ((tpPrice - currentPrice) / currentPrice) * 100 : ((currentPrice - tpPrice) / currentPrice) * 100;
      if (distTp > 0) parts.push(`TP까지 ${fmtNum(distTp, 2)}%`);
    }
    if (currentPrice > 0 && slPrice > 0) {
      const distSl = posSide === "LONG" ? ((currentPrice - slPrice) / currentPrice) * 100 : ((slPrice - currentPrice) / currentPrice) * 100;
      if (distSl > 0) parts.push(`SL까지 ${fmtNum(distSl, 2)}%`);
    }
    summaryText = parts.join(" · ");
  } else {
    summaryText = `포지션 없음 · AI 판단 ${decision.text}`;
  }
  setT("chartAiSummaryText", summaryText);
  renderChartRiskGauge(tp, sl, unrealizedPnl, posSide, tpPrice, slPrice, currentPrice);
}

function executionAlertState(state, compactState) {
  const explicit = state?.execution_alert || compactState?.execution_alert;
  if (explicit && typeof explicit === "object") return explicit;
  const execution = state?.account?.execution || state?.signal?.live_execution || compactState?.signal?.live_execution || {};
  const enabled = Boolean(execution.enabled);
  const blocking = Boolean(execution.blocking);
  const requested = execution.requested_enabled === undefined ? enabled : Boolean(execution.requested_enabled);
  const error = String(execution.error || execution.last_error || "");
  const decisionReason = String(state?.signal?.block_reason || state?.signal?.governor_reason || state?.signal?.hold_reason || "");
  const decisionIssue = /(error|failed|mismatch|bad_|blocked|unavailable|not_ready|pending_reconcile)/i.test(decisionReason);
  const reason = error || (decisionIssue ? decisionReason : "") || String(execution.disabled_reason || execution.status || "");
  if (enabled && !blocking && !error && !decisionIssue) return { active: false };
  const severity = error || /(error|failed|mismatch|bad_)/i.test(decisionReason) ? "error" : (requested ? "blocked" : "disabled");
  return {
    active: true,
    severity,
    title: severity === "error" ? "트레이딩봇 실행 오류" : (severity === "blocked" ? "실제 주문 실행 차단" : "실제 주문 실행 비활성"),
    reason: reason || "unknown_execution_state",
    occurred_at: execution.last_error_at || execution.disabled_at || state?.updated_at || "",
  };
}

function renderExecutionAlert(state, compactState) {
  const banner = el("executionAlertBanner");
  if (!banner) return;
  const alert = executionAlertState(state, compactState);
  if (!alert?.active) {
    banner.className = "execution-alert-banner hidden";
    return;
  }
  const severity = ["error", "blocked", "disabled"].includes(String(alert.severity)) ? String(alert.severity) : "blocked";
  banner.className = "execution-alert-banner " + severity;
  setT("executionAlertTitle", String(alert.title || "트레이딩봇 실행 알림"));
  setT("executionAlertReason", String(alert.reason || "unknown_execution_state"));
  setT("executionAlertTime", alert.occurred_at ? "발생 " + fmtTs(alert.occurred_at) : "발생 시각 미상");
}

function exposureFromRow(row) {
  return firstPositive(row.total_exposure, row.notional_exposure, row.new_total_exposure, row.new_notional_exposure, row.position_fraction, row.new_position_fraction, row.margin_fraction, row.new_margin_fraction);
}

function exposureSeries(filter) {
  let rows = latestTradeJournal || [];
  if (filter && filter !== "ALL") rows = rows.filter((row) => strategyTagFromRow(row) === filter);
  // Only CLOSE events: one point per position that has actually been closed.
  // OPEN/RESIZE rows are excluded so still-open positions (of any asset) never show up here.
  return rows
    .filter((row) => String(row.kind || "").toUpperCase() === "CLOSE")
    .map((row) => ({ ts: row.closed_at || row.ts, exposure: exposureFromRow(row), side: row.side, kind: row.kind }))
    .filter((row) => Number.isFinite(Number(row.exposure)) && Number(row.exposure) >= 0)
    .slice(-80);
}

function riskClass(v) {
  const x = Number(v || 0);
  if (x > 0) return "good";
  if (x < 0) return "bad";
  return "muted";
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
// same gauge were NOT recreated -- user confirmed the magnet is redundant with the chart line
// (liquidationMagnetLevel()) and never asked for energy/recommendation back. Always renders a row
// (never disappears) per 2026-08-27 request, with a quiet state for warming-up/no-liquidation.
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

function cleanDisplaySource(value) {
  const head = String(value || "-").split("|")[0].trim();
  const lower = head.toLowerCase();
  if (lower.includes("alpha4_3_sticky") || lower.includes("alpha4.3 sticky") || lower.includes("sticky_current")) return "Alpha4.3 Sticky";
  if (lower === "fully_learned" || lower.includes("fully_learned")) return "완전학습 거버너";
  if (lower === "sniper" || lower.includes("high_conviction")) return "스나이퍼 5x";
  if (lower === "trend" || lower.includes("bull_bear")) return "추세 5x";
  if (lower === "micro" || lower.includes("wnc")) return "W/N/C 마이크로 5x";
  if (lower === "cash" || lower.includes("no_sleeve_entry")) return "현금 대기";
  if (lower.includes("final_governor")) return "최종 거버너";
  if (lower.includes("sniper_priority_entry")) return "스나이퍼 우선 진입";
  if (lower.includes("trend_sleeve_entry")) return "추세 슬리브 진입";
  if (lower.includes("micro_sleeve_entry")) return "마이크로 슬리브 진입";
  return head
    .replaceAll("DSAC_CONTROLLER", "DSAC 컨트롤러")
    .replaceAll("DSAC_COMPACT", "DSAC 컴팩트")
    .replaceAll("DSAC_PRIMARY", "DSAC 기본")
    .replaceAll("UNIFIED_BUCKET", "통합 버킷")
    .replaceAll("UNIFIED_NATIVE", "통합 기본")
    .replaceAll("DISPLAY_ONLY_DISABLED", "표시 전용 비활성")
    .replaceAll("_", " ");
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

function renderLineSvg(svg, points) {
  const parentW = svg.parentElement ? svg.parentElement.clientWidth : 0;
  const w = Math.max(parentW, 400), h = 280;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.innerHTML = "";
  
  const vals = (points || []).map(p => Number(p.equity || 1));
  if (!vals.length) return;

  const ml = 60, mr = 20, mt = 20, mb = 50;
  const cw = w - ml - mr, ch = h - mt - mb;
  const min = Math.min(...vals), max = Math.max(...vals);
  const pad = Math.max((max - min) * 0.1, 0.001);
  const yMin = min - pad, yMax = max + pad, ySpan = yMax - yMin;
  
  const xAt = (i) => ml + (i * cw) / Math.max(vals.length - 1, 1);
  const yAt = (v) => mt + ((yMax - v) * ch) / ySpan;
  const NS = "http://www.w3.org/2000/svg";

  // Gradient
  const defs = document.createElementNS(NS, "defs");
  const grad = document.createElementNS(NS, "linearGradient");
  grad.setAttribute("id", "equityGrad"); grad.setAttribute("x1", "0"); grad.setAttribute("y1", "0"); grad.setAttribute("x2", "0"); grad.setAttribute("y2", "1");
  const s1 = document.createElementNS(NS, "stop"); s1.setAttribute("offset", "0%"); s1.setAttribute("stop-color", "var(--accent)"); s1.setAttribute("stop-opacity", "0.2");
  const s2 = document.createElementNS(NS, "stop"); s2.setAttribute("offset", "100%"); s2.setAttribute("stop-color", "var(--accent)"); s2.setAttribute("stop-opacity", "0");
  grad.appendChild(s1); grad.appendChild(s2);
  defs.appendChild(grad);
  svg.appendChild(defs);

  // Grid
  axisTicks(yMin, yMax, 4).forEach(t => {
    const y = yAt(t);
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", ml); line.setAttribute("x2", w - mr);
    line.setAttribute("y1", y); line.setAttribute("y2", y);
    line.setAttribute("stroke", "var(--line)"); line.setAttribute("stroke-width", "1");
    svg.appendChild(line);
    const txt = document.createElementNS(NS, "text");
    txt.setAttribute("x", ml - 10); txt.setAttribute("y", y + 4);
    txt.setAttribute("text-anchor", "end"); txt.setAttribute("font-size", "10"); txt.setAttribute("fill", "var(--muted)");
    txt.textContent = `${fmtNum((t - 1) * 100, 0)}%`;
    svg.appendChild(txt);
  });

  if (yMin <= 1 && yMax >= 1) {
    const zero = document.createElementNS(NS, "line");
    zero.setAttribute("x1", ml); zero.setAttribute("x2", w - mr);
    zero.setAttribute("y1", yAt(1)); zero.setAttribute("y2", yAt(1));
    zero.setAttribute("stroke", "var(--hover-line)");
    zero.setAttribute("stroke-width", "1.5");
    svg.appendChild(zero);
  }

  // Time Axis
  const timeIndices = [0, Math.floor(vals.length / 2), vals.length - 1];
  [...new Set(timeIndices)].forEach(idx => {
    const x = xAt(idx);
    const txt = document.createElementNS(NS, "text");
    txt.setAttribute("x", x); txt.setAttribute("y", h - 20);
    txt.setAttribute("text-anchor", "middle"); txt.setAttribute("font-size", "10"); txt.setAttribute("fill", "var(--muted)");
    txt.textContent = fmtDateTick(points[idx]?.ts || points[idx]?.closed_at);
    svg.appendChild(txt);
  });

  // Line & Area
  const pts = vals.map((v, i) => `${xAt(i)},${yAt(v)}`).join(" ");
  const areaPath = document.createElementNS(NS, "polygon");
  areaPath.setAttribute("points", `${ml},${h - mb} ${pts} ${w - mr},${h - mb}`);
  areaPath.setAttribute("fill", "url(#equityGrad)");
  svg.appendChild(areaPath);

  const linePath = document.createElementNS(NS, "polyline");
  linePath.setAttribute("points", pts); linePath.setAttribute("fill", "none");
  linePath.setAttribute("stroke", "var(--accent)"); linePath.setAttribute("stroke-width", "2.8");
  linePath.setAttribute("stroke-linejoin", "round");
  svg.appendChild(linePath);

  const vLine = document.createElementNS(NS, "line");
  vLine.setAttribute("y1", mt); vLine.setAttribute("y2", h - mb);
  vLine.setAttribute("stroke", "var(--hover-line)"); vLine.setAttribute("stroke-dasharray", "4,4");
  vLine.style.display = "none"; vLine.style.pointerEvents = "none";
  svg.appendChild(vLine);

  const hoverDot = document.createElementNS(NS, "circle");
  hoverDot.setAttribute("r", "5");
  hoverDot.setAttribute("fill", "var(--accent)");
  hoverDot.setAttribute("stroke", "var(--chart-bg)");
  hoverDot.setAttribute("stroke-width", "2");
  hoverDot.style.display = "none";
  hoverDot.style.pointerEvents = "none";
  svg.appendChild(hoverDot);

  // Line Chart Tooltip Support
  svg.onmousemove = (evt) => {
    const rect = svg.getBoundingClientRect();
    const mx = (evt.clientX - rect.left) * (w / rect.width);
    if (mx < ml || mx > w - mr) { hideTooltip(); return; }
    
    const idx = Math.min(vals.length - 1, Math.max(0, Math.round(((mx - ml) / cw) * (vals.length - 1))));
    const p = points[idx];
    if (!p) return;
    const timingText = executionTimingText(p);
    const aiText = aiTimingText(p);
    
    const tx = xAt(idx);
    hoverDot.setAttribute("cx", tx);
    hoverDot.setAttribute("cy", yAt(vals[idx]));
    hoverDot.style.display = "block";

    vLine.setAttribute("x1", tx); vLine.setAttribute("x2", tx);
    vLine.style.display = "block";

    showTooltip(evt.pageX, evt.pageY, `
      <div style="font-weight:bold;margin-bottom:4px;border-bottom:1px solid rgba(255,255,255,0.2);">자산 분석 #${idx + 1}</div>
      누적자산: <b>${fmtNum(p.equity, 4)}</b><br>
      누적수익률: <b>${fmtPct(Number(p.cumulative_return_pct ?? ((p.equity - 1) * 100)), 2)}</b><br>
      거래수익률: <span style="color:${p.pnl_pct >= 0 ? "var(--good)" : "var(--bad)"}">${fmtPct(p.pnl_pct, 2)}</span><br>
      ${timingText ? `<span style="font-size:10px;color:var(--muted);">${timingText}${aiText ? ` · ${aiText}` : ""}</span><br>` : ""}
      <span style="font-size:10px;color:var(--muted);">${p.closed_at || p.ts || ""}</span>
    `);
  };
  svg.onmouseleave = () => {
    hideTooltip();
    if (typeof hoverDot !== 'undefined') hoverDot.style.display = "none";
    if (typeof vLine !== 'undefined') vLine.style.display = "none";
    if (typeof hoverDots !== 'undefined') hoverDots.forEach(d => d.style.display = "none");
  };
}

function renderBarSvg(svg, points) {
  const parentW = svg.parentElement ? svg.parentElement.clientWidth : 0;
  const w = Math.max(parentW, 400), h = 280;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.innerHTML = "";

  const vals = (points || []).map(p => Number(p.pnl_pct || 0));
  if (!vals.length) return;

  const ml = 45, mr = 10, mt = 10, mb = 35;
  const cw = w - ml - mr, ch = h - mt - mb;
  const min = Math.min(...vals), max = Math.max(...vals);
  // Flexible but balanced Y scale
  const yMax = Math.max(max, Math.abs(min), 0.5) * 1.05;
  const yMin = -yMax, ySpan = yMax - yMin;
  const yAt = (v) => mt + ((yMax - v) * ch) / ySpan;
  const NS = "http://www.w3.org/2000/svg";

  const vLine = document.createElementNS(NS, "line");
  vLine.setAttribute("y1", mt); vLine.setAttribute("y2", h - mb);
  vLine.setAttribute("stroke", "var(--hover-line)"); vLine.setAttribute("stroke-dasharray", "4,4");
  vLine.style.display = "none"; vLine.style.pointerEvents = "none";
  svg.appendChild(vLine);

  const hoverDot = document.createElementNS(NS, "circle");
  hoverDot.setAttribute("r", "5");
  hoverDot.setAttribute("fill", "var(--good)");
  hoverDot.setAttribute("stroke", "var(--chart-bg)");
  hoverDot.setAttribute("stroke-width", "2");
  hoverDot.style.display = "none";
  hoverDot.style.pointerEvents = "none";
  svg.appendChild(hoverDot);

  // Grid
  axisTicks(yMin, yMax, 4).forEach(t => {
    const y = yAt(t);
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", ml); line.setAttribute("x2", w - mr);
    line.setAttribute("y1", y); line.setAttribute("y2", y);
    line.setAttribute("stroke", Math.abs(t) < 0.001 ? "var(--hover-line)" : "var(--line)");
    line.setAttribute("stroke-width", Math.abs(t) < 0.001 ? "1.5" : "1");
    svg.appendChild(line);
    const txt = document.createElementNS(NS, "text");
    txt.setAttribute("x", ml - 12); txt.setAttribute("y", y + 4);
    txt.setAttribute("text-anchor", "end"); txt.setAttribute("font-size", "10"); txt.setAttribute("fill", "var(--muted)");
    txt.textContent = `${t.toFixed(0)}%`;
    svg.appendChild(txt);
  });

  const bw = Math.min(cw / vals.length, 40);
  const zeroY = yAt(0);
  vals.forEach((v, i) => {
    const bar = document.createElementNS(NS, "rect");
    const hh = Math.abs(zeroY - yAt(v));
    const x = ml + (cw / vals.length) * i + (cw / vals.length - bw) / 2;
    bar.setAttribute("x", x); bar.setAttribute("y", v >= 0 ? zeroY - hh : zeroY);
    bar.setAttribute("width", Math.max(bw - 4, 2)); bar.setAttribute("height", Math.max(hh, 1));
    bar.setAttribute("fill", v >= 0 ? "var(--good)" : "var(--bad)");
    bar.setAttribute("rx", "3");
    svg.appendChild(bar);
  });

  // Bar Chart Tooltip Support
  svg.onmousemove = (evt) => {
    const rect = svg.getBoundingClientRect();
    const mx = (evt.clientX - rect.left) * (w / rect.width);
    if (mx < ml || mx > w - mr) { hideTooltip(); return; }
    
    const idx = Math.min(vals.length - 1, Math.max(0, Math.floor(((mx - ml) / cw) * vals.length)));
    const p = points[idx];
    if (!p) return;
    const timingText = executionTimingText(p);
    const aiText = aiTimingText(p);
    
    const pnl = Number(p.pnl_pct || 0);
    const bwVal = cw / vals.length;
    const tx = ml + idx * bwVal + bwVal / 2;
    hoverDot.setAttribute("cx", tx);
    hoverDot.setAttribute("cy", yAt(pnl));
    hoverDot.style.display = "block";
    hoverDot.setAttribute("fill", pnl >= 0 ? "var(--good)" : "var(--bad)");

    vLine.setAttribute("x1", tx); vLine.setAttribute("x2", tx);
    vLine.style.display = "block";

    showTooltip(evt.pageX, evt.pageY, `
      <div style="font-weight:bold;margin-bottom:4px;border-bottom:1px solid rgba(255,255,255,0.2);">청산 수익 분석</div>
      거래수익률: <span style="color:${pnl >= 0 ? "var(--good)" : "var(--bad)"}">${fmtPct(pnl, 2)}</span><br>
      누적수익률: <b>${fmtPct(Number(p.cumulative_return_pct ?? ((p.equity - 1) * 100)), 2)}</b><br>
      누적자산: <b>${fmtNum(p.equity, 4)}</b><br>
      ${timingText ? `<span style="font-size:10px;color:var(--muted);">${timingText}${aiText ? ` · ${aiText}` : ""}</span><br>` : ""}
      <span style="font-size:10px;color:var(--muted);">${p.closed_at || p.ts || ""}</span>
    `);
  };
  svg.onmouseleave = () => {
    hideTooltip();
    if (typeof hoverDot !== 'undefined') hoverDot.style.display = "none";
    if (typeof vLine !== 'undefined') vLine.style.display = "none";
    if (typeof hoverDots !== 'undefined') hoverDots.forEach(d => d.style.display = "none");
  };
}

function renderExposureSvg(svg, points) {
  const parentW = svg.parentElement ? svg.parentElement.clientWidth : 0;
  const w = Math.max(parentW, 400), h = 240;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.innerHTML = "";
  const vals = (points || []).map((p) => Number(p.exposure || 0));
  const NS = "http://www.w3.org/2000/svg";
  if (!vals.length) {
    const txt = document.createElementNS(NS, "text");
    txt.setAttribute("x", w / 2);
    txt.setAttribute("y", h / 2);
    txt.setAttribute("text-anchor", "middle");
    txt.setAttribute("fill", "var(--muted)");
    txt.textContent = "노출 이력 대기 중...";
    svg.appendChild(txt);
    return;
  }

  const ml = 48, mr = 18, mt = 16, mb = 34;
  const cw = w - ml - mr, ch = h - mt - mb;
  const max = Math.max(...vals, 1);
  const yAt = (v) => mt + ((max - v) * ch) / max;
  const xAt = (i) => ml + (i * cw) / Math.max(vals.length - 1, 1);

  axisTicks(0, max, 4).forEach((t) => {
    const y = yAt(t);
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", ml);
    line.setAttribute("x2", w - mr);
    line.setAttribute("y1", y);
    line.setAttribute("y2", y);
    line.setAttribute("stroke", "var(--line)");
    svg.appendChild(line);
    const txt = document.createElementNS(NS, "text");
    txt.setAttribute("x", ml - 8);
    txt.setAttribute("y", y + 4);
    txt.setAttribute("text-anchor", "end");
    txt.setAttribute("font-size", "10");
    txt.setAttribute("fill", "var(--muted)");
    txt.textContent = `${fmtNum(t, 2)}x`;
    svg.appendChild(txt);
  });

  const pts = vals.map((v, i) => `${xAt(i)},${yAt(v)}`).join(" ");
  const line = document.createElementNS(NS, "polyline");
  line.setAttribute("points", pts);
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", "var(--accent)");
  line.setAttribute("stroke-width", "2.5");
  line.setAttribute("stroke-linejoin", "round");
  svg.appendChild(line);

  vals.forEach((v, i) => {
    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("cx", xAt(i));
    dot.setAttribute("cy", yAt(v));
    dot.setAttribute("r", "3");
    dot.setAttribute("fill", String(points[i].side || "").toUpperCase() === "SHORT" ? "var(--bad)" : "var(--good)");
    svg.appendChild(dot);
  });

  const start = document.createElementNS(NS, "text");
  start.setAttribute("x", ml);
  start.setAttribute("y", h - 12);
  start.setAttribute("font-size", "10");
  start.setAttribute("fill", "var(--muted)");
  start.textContent = fmtDateTick(points[0]?.ts);
  svg.appendChild(start);
  const end = document.createElementNS(NS, "text");
  end.setAttribute("x", w - mr);
  end.setAttribute("y", h - 12);
  end.setAttribute("text-anchor", "end");
  end.setAttribute("font-size", "10");
  end.setAttribute("fill", "var(--muted)");
  end.textContent = fmtDateTick(points[points.length - 1]?.ts);
  svg.appendChild(end);
}

function parseTradeJournal(text) {
  return String(text || "").split(/\r?\n/).map(line => {
    try { return JSON.parse(line.trim()); } catch (e) { return null; }
  }).filter(Boolean);
}

function tradeSideClass(side) {
  const s = String(side || "").toUpperCase();
  return s === "LONG" ? "long" : s === "SHORT" ? "short" : "";
}

let latestJournalFilter = "ALL";

function renderTradeJournal() {
  const listEl = el("tradeJournalList");
  const pnlEl = el("journalTotalPnl");
  if (!listEl) return;

  // 1. Filter
  const filtered = closeTradeRows(latestJournalFilter);

  // 2. Calculate compounded return for selected filter
  const equitySeries = buildTradeEquitySeries(latestJournalFilter);
  const totalPnl = equitySeries.length ? equitySeries[equitySeries.length - 1].cumulative_return_pct : 0;
  if (pnlEl) {
    pnlEl.textContent = `누적 ${fmtPct(totalPnl)}`;
    pnlEl.className = `pnl-badge ${riskClass(totalPnl)}`;
  }

  // 2b. Hero metrics: cumulative return + today's realized P&L
  setT("heroCumulativePnl", filtered.length ? fmtPct(totalPnl) : "-");
  const heroCumEl = el("heroCumulativePnl");
  if (heroCumEl) {
    heroCumEl.classList.remove("good-text", "bad-text", "muted-text");
    heroCumEl.classList.add(`${filtered.length ? riskClass(totalPnl) : "muted"}-text`);
  }
  setT("heroCumulativeSub", `총 ${filtered.length}건 체결`);

  const todayRows = filtered.filter((row) => isToday(rowTs(row)));
  const todayPnl = todayRows.reduce((sum, row) => sum + pnlPctFromRow(row), 0);
  setT("heroTodayPnl", todayRows.length ? fmtPct(todayPnl) : "-");
  const heroTodayEl = el("heroTodayPnl");
  if (heroTodayEl) {
    heroTodayEl.classList.remove("good-text", "bad-text", "muted-text");
    heroTodayEl.classList.add(`${todayRows.length ? riskClass(todayPnl) : "muted"}-text`);
  }
  setT("heroTodaySub", todayRows.length ? `오늘 ${todayRows.length}건 체결` : "오늘 체결 없음");

  // 3. Render list (latest 10)
  const recent = filtered.slice(-10).reverse();
  listEl.innerHTML = recent.map(row => {
    const side = String(row.side || "-").toUpperCase();
    const sideCls = tradeSideClass(side);
    const pnlPct = pnlPctFromRow(row);
    const coin = assetLabel(tradeAssetKey(row));
    const source = tradeGovernorLabel(row);
    const bucket = fmtNum(exposureFromRow(row) || row.execution_leverage || row.leverage || 1, 1);
    const reasonText = closeReasonText(row);
    const riskText = riskSummaryText(row);
    const feeText = feeModelText(row);
    const subText = [reasonText ? `청산 이유: ${reasonText}` : "", riskText, feeText].filter(Boolean).join(" · ");
    const entryLiquidity = executionLegLiquidity(row, "entry");
    const exitLiquidity = executionLegLiquidity(row, "exit");
    return `
      <div class="trade-journal-row">
        <div class="trade-journal-left">
          <div class="trade-journal-ts">${fmtTs(row.closed_at || row.ts)}</div>
          <div class="trade-journal-side ${sideCls}">
            <span class="trade-journal-asset">${coin}</span><span class="journal-label">${source}</span>${sideLabel(side)} <span class="trade-journal-bucket">${bucket}x</span>
          </div>
        </div>
        <div class="trade-journal-meta">
          <div class="trade-journal-main">${priceWithLiquidity("진입", row.entry_price, entryLiquidity)} → ${priceWithLiquidity("청산", row.exit_price, exitLiquidity)}</div>
          <div class="trade-journal-sub muted">${subText}</div>
        </div>
        <div class="trade-journal-pnl ${riskClass(pnlPct)}">${fmtPct(pnlPct)}</div>
      </div>`;
  }).join("");
}

function renderTradePanels() {
  renderTradeJournal();

  try {
    const eqSvg = el("equitySvg");
    if (eqSvg) renderLineSvg(eqSvg, selectTradeRowsForCharts(latestJournalFilter));
  } catch (e) { console.error("Equity Render Error:", e); }

  try {
    const pnSvg = el("pnlSvg");
    if (pnSvg) renderBarSvg(pnSvg, selectTradeRowsForCharts(latestJournalFilter));
  } catch (e) { console.error("PnL Render Error:", e); }

  try {
    const exSvg = el("exposureSvg");
    if (exSvg) renderExposureSvg(exSvg, exposureSeries(latestJournalFilter));
  } catch (e) { console.error("Exposure Render Error:", e); }
}

async function fetchBinanceHistory(asset = activeChartAsset) {
  try {
    const res = await fetch(`/api/market-history?asset=${asset}`, { cache: "no-store" });
    if (!res.ok) return;
    const payload = await res.json();
    candleHistoryByAsset[asset] = Array.isArray(payload?.candles) ? payload.candles : [];
    if (asset === activeChartAsset) syncActiveMarketState();
  } catch (e) { console.error("History Error:", e); }
}

async function maybeFetchBinanceHistory() {
  const now = Date.now();
  const cached = candleHistoryByAsset[activeChartAsset] || [];
  const lastAt = Number(lastCandleHistoryFetchAtByAsset[activeChartAsset] || 0);
  if (cached.length && now - lastAt < CANDLE_HISTORY_POLL_MS) return;
  lastCandleHistoryFetchAtByAsset[activeChartAsset] = now;
  await fetchBinanceHistory(activeChartAsset);
}

// Snapshot tab's chart is always ETH (matches the liquidation map's ETH-only scope), independent
// of whichever asset the Live tab's chart is currently showing -- so it needs its own fetch
// rather than piggybacking on maybeFetchBinanceHistory()'s activeChartAsset gating.
async function maybeFetchSnapshotChartHistory() {
  const now = Date.now();
  const cached = candleHistoryByAsset[activeSnapshotAsset] || [];
  if (cached.length && now - lastSnapshotHistoryFetchAt < CANDLE_HISTORY_POLL_MS) return;
  lastSnapshotHistoryFetchAt = now;
  await fetchBinanceHistory(activeSnapshotAsset);
  renderSnapshotChart();
}

async function refreshTradeJournals(nonce) {
  const now = Date.now();
  if (lastJournalFetchAt && now - lastJournalFetchAt < JOURNAL_POLL_MS) return false;

  const apiHeaders = tradesEtag ? { "If-None-Match": tradesEtag } : {};
  const apiRes = await fetch(`${API_TRADES_URL}?source=${latestJournalFilter}`, {
    cache: "no-store",
    headers: apiHeaders,
  }).catch(() => null);
  if (apiRes?.status === 304) {
    lastJournalFetchAt = now;
    return false;
  }
  if (apiRes && apiRes.ok) {
    const payload = await apiRes.json();
    tradesEtag = apiRes.headers.get("ETag") || tradesEtag;
    latestTradeJournal = Array.isArray(payload?.rows) ? payload.rows : [];
    latestTradeEquitySeries = Array.isArray(payload?.equity) ? payload.equity : [];
    lastJournalFetchAt = now;
    tradeJournalLoaded = true;
    return true;
  }

  const journalRes = await fetch(`${TRADE_JOURNAL_URL}?t=${nonce}`, { cache: "no-store" }).catch(() => null);
  const merged = [];
  if (journalRes && journalRes.ok) {
    merged.push(...parseTradeJournal(await journalRes.text()).map(r => ({ ...r, raw_source: r.source || "", source: strategyTagFromRow(r) })));
  }
  latestTradeJournal = merged.sort((a, b) => rowTs(a) - rowTs(b));
  latestTradeEquitySeries = [];
  lastJournalFetchAt = now;
  tradeJournalLoaded = true;
  return true;
}

function renderLiveMarket() {
  if (!latestMainState) return;
  const state = latestMainState;
  const compactState = latestCompactState;
  const activeState = usableGovernorShadowState(compactState) || state;
  const chartState = assetDecisionState(state, compactState, activeChartAsset);
  const currentPrice = Number(latestLivePrice || chartState?.last_price || chartState?.price || activeState.last_price || activeState.price || 0);
  const entryPrice = Number(openPosition(chartState)?.entry_price || 0);
  updateChart(
    currentPrice,
    latestLivePriceTs || chartState?.updated_at || chartState?.cycle_timestamp_kst || activeState.updated_at || activeState.cycle_timestamp_kst,
    entryPrice,
  );
  setT("chartStamp", latestLivePriceTs ? fmtTs(latestLivePriceTs) : fmtTs(state.updated_at || state.cycle_timestamp_kst));
}

function applyDashboardEvent(payload) {
  const tickers = payload?.tickers || {};
  let btcPriceUpdated = false;
  let ethPriceUpdated = false;
  Object.entries(tickers).forEach(([asset, ticker]) => {
    const price = Number(ticker?.price || 0);
    if (!(price > 0) || !ASSET_CONFIG[asset]) return;
    latestLivePriceByAsset[asset] = price;
    latestLivePriceTsByAsset[asset] = String(ticker.ts || "");
    if (asset === "btc") btcPriceUpdated = true;
    if (asset === "eth") ethPriceUpdated = true;
  });
  if (btcPriceUpdated && latestBtcMultislotPayload) renderBtcMultislotSlots(latestBtcMultislotPayload);
  if (ethPriceUpdated && latestEthOdyssey4Payload) renderEthOdyssey4Position(latestEthOdyssey4Payload);
  if (payload?.state?.state) {
    latestMainState = payload.state.state;
    latestCompactState = payload.state.compactState || null;
  }
  syncActiveMarketState();
  if (!latestMainState || isScrolling) return;
  if (payload?.state?.state) {
    render(latestMainState, latestCompactState, {
      stateChanged: true,
      journalChanged: tradeJournalLoaded && !tradePanelsRendered,
    });
    return;
  }
  renderLiveMarket();
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
function toneStripSvg(tones, times, provisionalLast, liveFiring, key) {
  const list = Array.isArray(tones) ? tones : [];
  const timeList = Array.isArray(times) ? times : [];
  const n = Math.max(list.length, 1);
  const w = 240, h = 15, gap = 1.5;
  const bw = Math.max((w - gap * (n - 1)) / n, 1);

  // Group consecutive equal tones into segments. The still-forming provisional bar (always the last
  // array entry, see evidenceStripSvg's liveTone param) never merges into the segment before it even
  // when its tone happens to match -- keeps evidence-bar-provisional's softened fill scoped to only
  // the genuinely-unconfirmed portion instead of bleeding across a whole merged block.
  const segments = [];
  for (let i = 0; i < n; i++) {
    const tone = list[i] || "neutral";
    const isProvisionalBar = !!(provisionalLast && i === n - 1);
    const prev = segments[segments.length - 1];
    if (prev && prev.tone === tone && !isProvisionalBar) {
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
function evidenceStripSvg(bottomHist, topHist, latestIso, stepMinutes, liveTone, liveIso, key) {
  const n = Math.max(bottomHist.length, topHist.length, 1);
  const tones = Array.from({ length: n }, (_, i) => (bottomHist[i] ? "good" : topHist[i] ? "bad" : "neutral"));
  const times = evenlySpacedBarTimes(latestIso, n, stepMinutes);
  if (liveTone) { tones.push(liveTone); times.push(liveIso || ""); }
  return toneStripSvg(tones, times, !!liveTone, !!liveTone && liveTone !== "neutral", key);
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
  v_rebound: { good: "급등", bad: "급락", flat: "미반등", neutral: "대기" },
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
function lastSegmentRangeLabel(tones, times, key, timeFmtKind) {
  const list = Array.isArray(tones) ? tones : [];
  const timeList = Array.isArray(times) ? times : [];
  const n = list.length;
  if (n === 0) return "-";
  const lastTone = list[n - 1] || "neutral";
  let start = n - 1;
  while (start > 0 && list[start - 1] === lastTone) start--;
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
    "대기": "최근 30분 안에 유동성 스윕(지지/저항선을 살짝 뚫었다 되돌아온 것)이 없었어요 — 지금은 참고할 신호가 없어요.",
    "급등": "방금 하락스윕(지지선을 살짝 뚫었다 되돌아온 것) 후 TabPFN 모델이 '진짜 반등'(V자반등)이라고 판단했어요 — 30분 내 종가로 1.5×ATR 이상 반등했고, 60분 전체에서도 정점 대비 20% 이하만 반납할 거라는 판정이에요(정밀도 검증구간 62~74%, 이 라벨 평균발생률의 1.7배로 비교적 신뢰 가능한 쪽). 자세한 계산 방식은 '자세히'를 확인하세요.",
    "급락": "방금 상승스윕(저항선을 살짝 뚫었다 되돌아온 것) 후 TabPFN 모델이 '진짜 반전'(V자반등)이라고 판단했어요 — 30분 내 종가로 1.5×ATR 이상 반전했고, 60분 전체에서도 정점 대비 20% 이하만 반납할 거라는 판정이에요(정밀도 검증구간 62~74%, 1.7배로 비교적 신뢰 가능한 쪽). 자세한 계산 방식은 '자세히'를 확인하세요.",
    "미반등": "방금 유동성 스윕이 나왔지만, TabPFN 모델이 '반등 시도 자체가 없었다'고 판단했어요 — 30분 안에 종가가 1.0×ATR도 못 갔다는 뜻일 뿐, 반대방향으로 뚜렷하게 움직였다는 뜻은 아니에요(정밀도 70~75%지만 이 라벨의 다수쪽이라 실제 추가정보는 1.2배 정도로 약함). 급등/급락(진짜 반등 콜)보다 근거가 약한 판정이라 방향 없이 회색으로 표시돼요 — 자세한 계산 방식은 '자세히'를 확인하세요.",
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
  v_rebound: "[계산] 유동성스윕(48봉 causal 스윙 고/저 이탈 후 종가 재진입, 이 대시보드의 liquidity_sweep과 동일 정의)이 발생하면 그 순간 22개 캔들/오더플로우/모멘텀 피쳐(Tier0)+RSI를 계산해 TabPFN(사전학습된 트랜스포머가 in-context로 추론하는 표형 파운데이션 모델 — 데이터셋별 재학습이 없음)에 입력합니다. 학습 컨텍스트는 2024-01~2025-08 스윕 중 '확실한' 3,783건에 고정(라이브에서도 매번 이 컨텍스트를 그대로 재사용, 최신 데이터로 자동 갱신되지 않음).\n" +
    "[기준] 확률≥50%면 '반등 콜'(스윕 후 30분 내 종가로 ATR(스윕 전 기준) 1.5배 이상 반등 AND 60분 전체에서 정점 대비 20% 이하만 반납), 미만이면 '미반등 콜'(30분 안에 종가가 ATR 1배에도 못 미침 — 반등 시도 자체가 없었던 경우만이고, 반대방향으로 뚜렷하게 움직였다는 뜻은 아닙니다). 이 둘 사이(느리게 반등했거나, 반등은 했는데 많이 반납한 애매한 경우, 전체의 58%)는 라벨 자체가 없어 **학습에서 통째로 제외** — 애매한 경우를 억지로 어느 한쪽으로 분류하지 않기 위한 설계입니다. 최근 60분 안에 스윕이 없으면 '대기'.\n" +
    "[배지 표시: 반등 콜만 급등/급락, 미반등 콜은 별도 '미반등'] 2026-08-31 두 차례 정정했습니다. 처음엔 '반등'/'반락'을 콜 이름 그대로 노출해서 상승스윕 후 반등 콜처럼 실제로는 하락이 예상되는데도 '반등'이라고 표시되며 빨간색이 뜨는 경우가 있었습니다(사용자 지적) — 그래서 반등/미반등 콜을 스윕 방향과 조합해 항상 급등(초록)/급락(빨강) 중 하나로 바꿨습니다. 그런데 미반등 콜은 '반등 시도 자체가 없었다'는 뜻일 뿐 반대방향으로 결정적으로 움직였다는 근거가 아닌데도 급등/급락이라는 강한 단어를 그대로 썼던 게 다시 지적받아(사용자 지적), 지금은 **진짜 반등(V자반등) 콜만** 스윕 방향과 조합해 급등(하락스윕 후 반등, 초록)/급락(상승스윕 후 반등, 빨강)으로 표시하고, **미반등 콜은 스윕 방향과 무관하게 항상 '미반등'**(회색, 중립 취급)으로 따로 표시합니다 — 활동-스트립을 마우스오버했을 때 나오는 막대별 라벨도 동일한 기준(백엔드 tone: good/bad/flat/neutral)으로 구분됩니다. '반락 콜'이라는 예전 이름 자체도 마치 반대방향으로 결정적으로 움직였다는 뜻처럼 들려 실제 정의와 어긋나 '미반등 콜'로 정정했습니다.\n" +
    "[의미] 2026-08-30 사용자가 육안으로 기존 라벨('반등' 판정에 애매한 사례가 섞여 보인다는 지적)을 재검토해 라벨을 근본적으로 재설계 — 애매한 중간지대를 제외하고 확실한 양극단만 대비하도록 바꾸자 시간순으로 분리된 3개 독립 구간 전부에서 큰 폭으로 개선됐습니다: VAL AUC 0.663→**0.734**, OOS 0.667→**0.762**, 예비 홀드아웃 0.682→**0.779**(전 구간·전 시드 일관, 다만 학습에 쓰는 이벤트 자체가 전체의 42%로 줄어든 '더 쉬운 문제'라는 점도 감안해야 합니다). **콜별 정밀도**(2026-08-30, 스윕 방향과 무관하게 콜 기준으로만 측정 — 급등/급락별로 세분화된 정밀도는 아직 측정 안 함): '반등 콜' 판정은 VAL·OOS 각각 62.3%/74.1%(평균발생률 35~43% 대비 1.7배) — 상대적으로 믿을 만한 쪽입니다. '미반등 콜' 판정은 75.0%/69.8%로 숫자만 보면 높아 보이지만 원래 이 라벨의 다수쪽이라 실제 추가정보는 1.2배뿐입니다.\n" +
    "[유의] 이 신호의 가치는 '스윕이 반전을 예측한다'는 게 아니라 '스윕이 이미 발생했다는 조건 안에서 어느 쪽으로 갈지 더 가려낸다'는 2차 정제입니다. **⚠️자동매매 경제성 게이트: 라벨을 새로 바꾼 뒤에도(ATR 트레일링스톱 방식으로 재검증) 여전히 실패했습니다**(2026-08-30, VAL·OOS를 각각 독립으로 봤을 때 둘 다 동시에 이익인 설정을 200개 이상 조합에서 하나도 찾지 못함 — 가장 좋아 보이는 조합도 한쪽 구간은 이익, 다른쪽은 손실로 갈렸습니다). 분류 정확도는 크게 올랐지만 그게 자동매매 수익으로 바로 이어지진 않았습니다. 봇 내부 상태가 아니라 대시보드 서버가 별도로 계산합니다 — 실제 매매 결정에는 연결되지 않았고, 재량 판단의 한 재료로만 쓰세요(반등 콜에서 나온 급등/급락이 미반등보다 상대적으로 더 신뢰 가능하며, 이제 화면에서도 미반등은 급등/급락과 별도 배지로 구분됩니다). 전체 방법론: docs/experiments/eth_liquidity_sweep_v_rebound_feature_plan_20260829.md.",
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
  v_rebound: { text: "60분", title: "스윕 후 60분(12봉) 안 실제 가격방향(급등/급락)을 예측 -- 확률≥50%인 '반등 콜'은 30분 내 종가로 1.5xATR 반등 후 60분 전체에서 정점 대비 20% 이하만 반납을 요구, 스윕 방향과 조합해 급등/급락으로 표시" },
  liq_pressure: { text: "1시간·4시간", title: "베이시스 극단 이후 1시간·4시간 시점의 강제청산 물량(방향)을 예측 -- 약 1개월 탐색적 표본, 이 저장소 표준 VAL/OOS 3-split 재현 전" },
  liq_direction: { text: "상태", title: "고정 예측 시간창 없이 매분 갱신되는 현재 청산 방향압력 -- 5·15분 지평 IC는 유의했으나(탐색적), 손익 결합 검정(8개 지평)은 전부 순손실" },
  liq_cascade: { text: "상태", title: "예측이 아니라 '지금 캐스케이드가 진행 중인가'를 보여주는 현재 상태값(반감기 약 2~3분으로 감쇠)" },
  whale: { text: "상태", title: "최근 5분간 큰손 체결 순유입 방향 -- 고정 예측 시간창 없는 현재 흐름 지표(방향-IC 검정 4개 지평 전부 무정보)" },
  retail_flow: { text: "상태", title: "최근 5분간 리테일 체결 순유입 방향 -- 1~15분 지평 방향-IC는 유의했으나 수수료 반영 손익은 전부 순손실, 고정 예측 시간창은 없음" },
  // -- evidence signals (모두 이 저장소 표준 스코어카드: 1시간/4시간/8시간 중 1시간이 대표 지평) --
  orthogonal_combo: { text: "2시간", title: "발동 조건 자체는 오실레이터(p_fast/p_slow) 이중극단+delta_z/funding_z 확인이지만, 신뢰도는 발동 시점 피쳐를 TabPFN에 넣어 '2시간 안 3.57xATR 이상 강하게 도달할 확률'로 평가(2026-08-31 교체, 이 저장소 분류·경제성 둘 다 역대 최고 성적)" },
  fib_extension_exhaustion: { text: "1시간", title: "1시간 기준 평가(실험적 등급, 표본 n≈190로 다른 6종보다 얇고 VAL→OOS lift 감쇠 확인)" },
  smt_divergence: { text: "1시간", title: "1시간·4시간·8시간 중 1시간 기준 정밀도/lift로 평가" },
  volume_wick_climax: { text: "1시간", title: "1시간·4시간·8시간 중 1시간 기준 정밀도/lift로 평가" },
  short_term_return_z: { text: "1시간", title: "발동 조건 자체는 15분(3봉) 수익률 급변이지만, 신뢰도는 발동 시점 피쳐를 TabPFN에 넣어 '1시간 안 1.75xATR 도달 확률'로 평가" },
  taker_delta_z_climax: { text: "2시간", title: "발동 조건 자체는 이번 봉 체결 쏠림이지만, 신뢰도는 발동 시점 피쳐를 TabPFN에 넣어 '2시간 안 2.0xATR 도달 확률'로 평가(2026-08-30 교체)" },
  dalton_rule2_balance_edge: { text: "2.5시간", title: "발동 조건 자체는 기존과 동일, 신뢰도는 2026-08-30부터 TabPFN 메타라벨 모델의 실시간 확률(2.5시간 안 도달 확률)로 교체" },
  liquidity_sweep: { text: "2.5시간", title: "발동 조건 자체는 48봉 스윙 저/고점 스윕이지만, 신뢰도는 발동 시점 피쳐를 TabPFN에 넣어 '2.5시간 안 4.0xATR 도달 확률'로 평가(2026-08-30 표준방식 재학습)" },
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
        <span class="ops-health-status-badge">${escapeHtml(it.subText || "-")}</span>
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
// is translated here, same pattern as cleanDisplaySource().
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
      "[신뢰도] 발동 시점 23피쳐(Tier0+rsi)를 TabPFN에 넣어 '150분 안 4.0×ATR 도달확률' 산출('특화 감지기'의 'V자 반등락'과는 별개 모델).\n" +
      "[검증] VAL 0.659 / OOS 0.637 / HOLDOUT 0.661.\n" +
      "[경제성] 트레일링스톱 VAL +10.70bp / OOS +14.49bp / 홀드아웃 +1.97bp(승률67.7%) — 이 신호 최초로 3구간 전부 통과(단, 실거래 미배포 · 재량 참고용).",
  },
  volume_wick_climax: {
    name: "거래량 꼬리 클라이맥스",
    detail: "[조건] 거래량 2표준편차↑ 폭증 + 꼬리비율(캔들범위 대비) 50%↑ (꼬리비율 = (시가·종가 중 작은값−저가)/(고가−저가), 천장은 반대쪽 꼬리로 정반대).\n" +
      "[의미] 패닉성 매도/매수가 몰렸다가 즉시 흡수됐다는 신호.",
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
  // 2026-08-24 추가(같은 날 후속) — ICT 2022 잔여요소 연구(오더블록/SMT/Po3)에서 유일하게 살아남은
  // SMT 다이버전스(3.12x/2.84x).
  smt_divergence: {
    name: "SMT 다이버전스(ETH·BTC 엇갈림)",
    detail: "[조건] ETH는 직전 4시간 저점(고점) 갱신, BTC는 미갱신 — 상관자산 비확인.\n" +
      "[의미] ICT SMT 다이버전스 — 두 자산 중 하나만 신저점이면 '진짜 매도세'가 아닐 가능성. 유동성 스윕과 정밀도 동급(42~43%).",
  },
  // 2026-08-24 추가(같은 날 후속) — 피보나치/하모닉 기하학 계열 연구에서 유일하게 sweep급 lift를
  // 보인 확장소진(3.27x/2.32x). 다른 6종보다 표본이 훨씬 얇고(n~190 vs 수백~수천) 경제성 게이트는
  // 0/16으로 실패해 "실험적" 등급으로만 편입 — 6종과 동일 신뢰도로 읽지 말 것.
  fib_extension_exhaustion: {
    name: "피보나치 확장 소진 (실험적)",
    detail: "[조건] 가격이 직전 스윙 구간의 127.2~161.8% 지점까지 확장.\n" +
      "[의미] 피보나치 확장 소진 — 리프트는 준수(3.27x/2.32x)하나 표본 n~190로 얇고(다른 6종은 수백~수천) 경제성 게이트 0/16 실패, 실험적 등급.",
  },
  // 2026-08-25 추가 — AMT(마켓프로파일 이론) Dalton 룰2. "경제성 아니라 통계적 정보성이 대시보드
  // 노출 기준"이라는 사용자 원칙 재확인 이후 첫 추가 사례(feedback_dashboard_indicators_ic_bar_not_
  // pnl_bar 메모리). 다른 7종과 실패 사유가 다름 — 경제성 게이트(시장가 비용)가 아니라 고정 TP:SL
  // 번역 자체가 실패(비용 0이어도 짐). 탐지 자체는 실재하는 정보라 사용자가 재량 참고용으로 채택.
  dalton_rule2_balance_edge: {
    name: "Dalton 룰2 — 레인지 가장자리 반응",
    detail: "[조건] 저변동성 국면(ATR%백분위 30%이하)에서 가격이 직전 4시간 레인지 가장자리(±15%이내)에 위치 — Dalton 룰2.\n" +
      "[신뢰도] 상태진입 시점 23피쳐를 TabPFN에 넣어 '2.5시간 안 1.90×ATR 도달확률' 산출.\n" +
      "[검증] VAL 0.598 / OOS 0.605 / HOLDOUT 0.576(이 계열 중 가장 안정) — 단, 리프트 0.86배 · 정의변수 미포함 · 경제성 미검증 등 유보사항 있어 재량 참고용.",
  },
};

// Snapshot tab "13신호 한눈에" overview: id lookup so the compact chip row (.signal-chip-row in
// index.html) can be updated from the exact same fetch/loop as the full evidenceSignalList below
// -- one source of truth per tick, no separate poll or duplicated tone logic.
const EVIDENCE_STRIP_CHIP_IDS = {
  orthogonal_combo: "eviChipOrthogonal",
  liquidity_sweep: "eviChipSweep",
  volume_wick_climax: "eviChipVolWick",
  short_term_return_z: "eviChipReturnZ",
  taker_delta_z_climax: "eviChipTakerDelta",
  smt_divergence: "eviChipSmt",
  fib_extension_exhaustion: "eviChipFibExt",
  dalton_rule2_balance_edge: "eviChipDalton",
};

function resetEvidenceStripChips() {
  Object.values(EVIDENCE_STRIP_CHIP_IDS).forEach((id) => {
    const chip = el(id);
    if (!chip) return;
    chip.className = "signal-chip neutral";
    const stateEl = chip.querySelector(".signal-chip-state");
    if (stateEl) stateEl.textContent = "-";
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
  if (!payload || payload.error) {
    if (badge) { badge.className = "ops-badge bad"; badge.textContent = "EVIDENCE UNREACHABLE"; }
    if (stripBadge) { stripBadge.className = "ops-badge bad"; stripBadge.textContent = "UNREACHABLE"; }
    resetEvidenceStripChips();
    return;
  }
  if (!payload.warmed_up) {
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
    const stateWithModel = modelPctText ? `${state} · ${modelPctText}` : state;
    const ko = EVIDENCE_SIGNAL_KO[s.name] || { name: s.name };
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
    evidenceHistoryBySignal[s.name] = { bottom_history: s.bottom_history || [], top_history: s.top_history || [], latest_bar_utc: payload.latest_bar_utc };
    // eviTones/eviTimes mirror evidenceStripSvg's own internal tone derivation (bottom_history[i] ->
    // good, top_history[i] -> bad, else neutral) -- recomputed here (not returned by that function,
    // which keeps its plain-string contract for the provisional-refresh outerHTML-replace call site)
    // so the axis/default-caption below can share the exact same tone/time arrays it draws from.
    const eviN = Math.max((s.bottom_history || []).length, (s.top_history || []).length, 1);
    const eviTones = Array.from({ length: eviN }, (_, i) => (s.bottom_history?.[i] ? "good" : s.top_history?.[i] ? "bad" : "neutral"));
    const eviTimes = evenlySpacedBarTimes(payload.latest_bar_utc, eviN, 5);
    // 2026-08-31 user request: drop the old "바닥 {ts} · 천장 {ts}" last-fired caption -- this
    // range+label already tells you when the CURRENT segment started, which is what that text was
    // approximating anyway.
    const defaultRangeText = lastSegmentRangeLabel(eviTones, eviTimes, "evidence", "hm");
    return `<article class="ops-health-row evidence-row ${tone}" data-signal="${s.name}">
      <span class="ops-health-dot" aria-hidden="true"></span>
      <div class="ops-health-info">
        <strong>${escapeHtml(ko.name)}${horizonBadgeHtml(s.name)}</strong>
        ${meaningText ? `<p class="signal-meaning">${escapeHtml(meaningText)}</p>` : ""}
        <div class="evidence-strip-wrap">
          ${evidenceStripSvg(s.bottom_history || [], s.top_history || [], payload.latest_bar_utc, 5, undefined, undefined, "evidence")}
          ${stripAxisHtml(eviTimes, "hm")}
          <small class="evidence-strip-caption">
            <span class="strip-time-now" data-fmt="hm" data-default="${escapeHtml(defaultRangeText)}">${escapeHtml(defaultRangeText)}</span>
          </small>
        </div>
        <button type="button" class="detail-toggle" aria-expanded="${isOpen}" onclick="toggleSignalDetail(this, '${detailKey}')">${isOpen ? "접기 ▴" : "자세히 ▾"}</button>
        <div class="signal-detail${isOpen ? " open" : ""}">${escapeHtml(detailText)}</div>
      </div>
      <div class="ops-health-meta">
        <span class="ops-health-status-badge">${escapeHtml(stateWithModel)}</span>
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
  try {
    const res = await fetch(API_EVIDENCE_SIGNALS_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`evidence signals ${res.status}`);
    renderEvidenceSignals(await res.json());
  } catch (error) {
    console.error("Evidence signal fetch error:", error);
    renderEvidenceSignals({ error: true });
  }
}

// Live PREVIEW of the currently-forming bar (2026-08-26) -- small live-dot overlay on the strip
// chips + a standalone badge, deliberately kept separate from renderEvidenceSignals()/the confirmed
// dot/state text above so a provisional reading can never be mistaken for (or silently overwrite)
// the validated confirmed one. "미확정" in the badge text is load-bearing, not decoration -- see
// load_evidence_signals_provisional()'s docstring in dashboard/server.py for why this reading has
// no lift track record of its own and can flicker before the bar closes.
function renderEvidenceSignalsProvisional(payload) {
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
      svgEl.outerHTML = evidenceStripSvg(hist.bottom_history, hist.top_history, hist.latest_bar_utc, 5, liveTone, payload.bar_open_utc, "evidence");
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

function slotHoldText(holdBars, barSeconds) {
  const totalMinutes = Math.round((Number(holdBars || 0) * Number(barSeconds || 300)) / 60);
  if (!(totalMinutes > 0)) return "-";
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return hours > 0 ? `${hours}시간 ${minutes}분 (${holdBars}봉)` : `${minutes}분 (${holdBars}봉)`;
}

// SL/TP/MFE/MAE come from state as raw price-move fractions (e.g. 0.075 = 7.5% price move),
// not account-level PnL -- see the Futures Risk Sizing Contract (PnL = price_move * notional).
// All five inputs here must already be converted to account-level % (rawMove * notional * 100)
// so they share the same scale as the live unrealized-PnL marker.
function shadowSlotRangeBar(slPct, tpPct, maePct, mfePct, currentPct, currentTitle) {
  const known = [slPct, tpPct, maePct, mfePct, 0, currentPct].filter((v) => Number.isFinite(v));
  const lo = Math.min(...known);
  const hi = Math.max(...known);
  const span = hi - lo || 1;
  const pos = (v) => clamp01((v - lo) / span) * 100;
  const zero = pos(0);
  const maeFill = Number.isFinite(maePct)
    ? (() => { const e = pos(maePct); return `<div class="shadow-slot-range-fill bad" style="left:${Math.min(zero, e)}%; width:${Math.abs(zero - e)}%;" title="MAE ${fmtPct(maePct, 2)}"></div>`; })()
    : "";
  const mfeFill = Number.isFinite(mfePct)
    ? (() => { const e = pos(mfePct); return `<div class="shadow-slot-range-fill good" style="left:${Math.min(zero, e)}%; width:${Math.abs(zero - e)}%;" title="MFE ${fmtPct(mfePct, 2)}"></div>`; })()
    : "";
  const currentKnown = Number.isFinite(currentPct);
  const currentCls = currentKnown ? riskClass(currentPct) : "muted";
  return `
    <div class="shadow-slot-range">
      <div class="shadow-slot-range-track">
        <div class="shadow-slot-range-zero" style="left:${zero}%;"></div>
        ${maeFill}
        ${mfeFill}
        <div class="shadow-slot-range-mark sl" style="left:${pos(slPct)}%;" title="SL ${fmtPct(slPct, 2)}"></div>
        <div class="shadow-slot-range-mark tp" style="left:${pos(tpPct)}%;" title="TP ${fmtPct(tpPct, 2)}"></div>
        ${currentKnown ? `<div class="shadow-slot-range-current ${currentCls}" style="left:${pos(currentPct)}%;" title="${currentTitle || `현재 ${fmtPct(currentPct, 2)}`}"></div>` : ""}
      </div>
      <div class="shadow-slot-range-labels">
        <span class="tag sl">SL ${fmtPct(slPct, 1)}</span>
        <span class="tag zero">0</span>
        <span class="tag tp">TP ${fmtPct(tpPct, 1)}</span>
      </div>
    </div>`;
}

// tp/sl are account-level fractions from activeRiskModel() (e.g. 0.12 = 12%, rawMove * exposure)
// -- same convention as signedRiskPairLabel(), which also multiplies by 100 for display.
// No MFE/MAE tracking is exposed for the real live position, so the gauge shows SL/TP bounds
// and the current unrealized marker only -- no progress-fill shading.
function renderChartRiskGauge(tp, sl, unrealizedPnl, posSide, tpPrice, slPrice, currentPrice) {
  const gaugeEl = el("chartRiskGauge");
  if (!gaugeEl) return;
  if (posSide !== "LONG" && posSide !== "SHORT") {
    gaugeEl.innerHTML = "";
    return;
  }
  if (!(tp > 0) && !(sl > 0)) {
    gaugeEl.innerHTML = "";
    return;
  }
  const tpPct = tp * 100;
  const slPct = sl * 100;
  const distTp = currentPrice > 0 && tpPrice > 0
    ? (posSide === "LONG" ? (tpPrice - currentPrice) / currentPrice : (currentPrice - tpPrice) / currentPrice) * 100 : null;
  const distSl = currentPrice > 0 && slPrice > 0
    ? (posSide === "LONG" ? (currentPrice - slPrice) / currentPrice : (slPrice - currentPrice) / currentPrice) * 100 : null;
  const distParts = [];
  if (distTp !== null) distParts.push(`TP까지 ${fmtNum(Math.max(distTp, 0), 2)}%`);
  if (distSl !== null) distParts.push(`SL까지 ${fmtNum(Math.max(distSl, 0), 2)}%`);
  const currentTitle = `현재 ${fmtPct(unrealizedPnl, 2)}${distParts.length ? " · " + distParts.join(" · ") : ""}`;
  gaugeEl.innerHTML = shadowSlotRangeBar(-slPct, tpPct, NaN, NaN, unrealizedPnl, currentTitle);
}

function shadowPositionCardHtml(pos, idx, livePrice, barSeconds) {
  if (!pos) {
    return `<div class="shadow-slot-card empty"><span class="muted">${idx !== null ? "비어있음" : "열린 포지션 없음"}</span></div>`;
  }
  const isLong = Number(pos.side) > 0;
  const sideCls = isLong ? "long" : "short";
  const entryPrice = Number(pos.entry_price || 0);
  const notional = Number(pos.notional_exposure ?? pos.notional ?? 0);
  const tpRaw = Number(pos.take_profit || 0);
  const slRaw = Number(pos.stop_loss || 0);
  const mfeRaw = Number(pos.mfe || 0);
  const maeRaw = Number(pos.mae || 0);
  const priceMoveFrac = livePrice > 0 && entryPrice > 0
    ? (isLong ? (livePrice - entryPrice) / entryPrice : (entryPrice - livePrice) / entryPrice)
    : null;
  const unrealizedPct = priceMoveFrac !== null ? priceMoveFrac * notional * 100 : null;
  const tpAcct = tpRaw * notional * 100;
  const slAcct = -slRaw * notional * 100;
  const mfeAcct = mfeRaw * notional * 100;
  const maeAcct = maeRaw * notional * 100;
  const tpPrice = entryPrice > 0 ? (isLong ? entryPrice * (1 + tpRaw) : entryPrice * (1 - tpRaw)) : 0;
  const slPrice = entryPrice > 0 ? (isLong ? entryPrice * (1 - slRaw) : entryPrice * (1 + slRaw)) : 0;
  const distTp = livePrice > 0 && tpPrice > 0
    ? (isLong ? (tpPrice - livePrice) / livePrice : (livePrice - tpPrice) / livePrice) * 100 : null;
  const distSl = livePrice > 0 && slPrice > 0
    ? (isLong ? (livePrice - slPrice) / livePrice : (slPrice - livePrice) / livePrice) * 100 : null;
  const distParts = [];
  if (distTp !== null) distParts.push(`TP까지 ${fmtNum(Math.max(distTp, 0), 2)}%`);
  if (distSl !== null) distParts.push(`SL까지 ${fmtNum(Math.max(distSl, 0), 2)}%`);
  const currentTitle = `현재 ${unrealizedPct === null ? "-" : fmtPct(unrealizedPct, 2)}${distParts.length ? " · " + distParts.join(" · ") : ""}`;
  const srcLabel = pos.source_component ? `<span class="shadow-slot-src muted">${pos.source_component}</span>` : "";
  return `
    <div class="shadow-slot-card">
      <div class="shadow-slot-head">
        <span class="shadow-slot-side ${sideCls}">${sideLabel(isLong ? "LONG" : "SHORT")} · ${fmtNum(Number(pos.leverage || 1), 1)}x</span>
        ${srcLabel}
      </div>
      <div class="shadow-ribbon">
        <div class="ribbon-item ribbon-primary">
          <span>미실현손익</span>
          <strong class="${unrealizedPct === null ? "muted" : riskClass(unrealizedPct)}-text">${unrealizedPct === null ? "-" : fmtPct(unrealizedPct, 2)}</strong>
        </div>
        <div class="ribbon-item">
          <span>진입가 → 현재가</span>
          <strong>${fmtNum(entryPrice, 1)} → ${livePrice > 0 ? fmtNum(livePrice, 1) : "-"}</strong>
        </div>
        <div class="ribbon-item">
          <span>진입 시각</span>
          <strong>${fmtTs(pos.entry_timestamp || pos.entry_ts)}</strong>
        </div>
        <div class="ribbon-item">
          <span>보유 시간</span>
          <strong>${slotHoldText(pos.hold_bars, barSeconds)}</strong>
        </div>
        <div class="ribbon-item">
          <span>마진 · 명목</span>
          <strong>${fmtPctNoPlus(Number(pos.margin_fraction || 0) * 100, 2)} · ${fmtPctNoPlus(notional * 100, 2)}</strong>
        </div>
        <div class="ribbon-item ribbon-risk">
          <span>TP / SL</span>
          <strong>계정 ${fmtPct(tpAcct, 2)}/${fmtPct(slAcct, 2)} · 가격 ${fmtPctNoPlus(tpRaw * 100, 1)}/-${fmtPctNoPlus(slRaw * 100, 1)}</strong>
        </div>
      </div>
      ${shadowSlotRangeBar(slAcct, tpAcct, maeAcct, mfeAcct, unrealizedPct, currentTitle)}
    </div>`;
}

function renderBtcMultislotSlots(payload) {
  const listEl = el("btcMultislotSlotList");
  const tabsEl = el("btcMultislotSlotTabs");
  if (!listEl) return;
  const slots = Array.isArray(payload?.slots) ? payload.slots : [];
  const barSeconds = Number(payload?.bar_seconds || 300);
  const livePrice = Number(latestLivePriceByAsset["btc"] || 0);
  if (!slots.length) {
    if (tabsEl) tabsEl.innerHTML = "";
    listEl.innerHTML = `<div class="shadow-slot-card empty"><span class="muted">슬롯 정보 없음</span></div>`;
    return;
  }
  if (btcMultislotActiveSlot >= slots.length) btcMultislotActiveSlot = 0;
  if (tabsEl) {
    tabsEl.innerHTML = slots.map((slot, idx) => {
      const dotCls = slot ? (Number(slot.side) > 0 ? "long" : "short") : "";
      return `<button type="button" class="shadow-slot-tab ${idx === btcMultislotActiveSlot ? "active" : ""}" data-slot-idx="${idx}">
        <span class="shadow-slot-tab-dot ${dotCls}"></span>슬롯 ${idx + 1}
      </button>`;
    }).join("");
    tabsEl.querySelectorAll(".shadow-slot-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        btcMultislotActiveSlot = Number(btn.dataset.slotIdx || 0);
        renderBtcMultislotSlots(latestBtcMultislotPayload || payload);
      });
    });
  }
  listEl.innerHTML = shadowPositionCardHtml(slots[btcMultislotActiveSlot], btcMultislotActiveSlot, livePrice, barSeconds);
}

function svgEmptyState(svg, text) {
  const parentW = svg.parentElement ? svg.parentElement.clientWidth : 0;
  const h = svg.viewBox?.baseVal?.height || 200;
  const w = Math.max(parentW, 400);
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.innerHTML = "";
  const NS = "http://www.w3.org/2000/svg";
  const txt = document.createElementNS(NS, "text");
  txt.setAttribute("x", w / 2);
  txt.setAttribute("y", h / 2);
  txt.setAttribute("text-anchor", "middle");
  txt.setAttribute("fill", "var(--muted)");
  txt.textContent = text;
  svg.appendChild(txt);
}

function renderShadowCharts(payload, pnlSvgId, eqSvgId) {
  const points = (payload?.equity_curve || []).map((row) => ({
    ts: row.ts,
    equity: 1 + Number(row.cumulative_return_pct || 0) / 100,
    pnl_pct: Number(row.trade_return_pct || 0),
    cumulative_return_pct: Number(row.cumulative_return_pct || 0),
  }));
  const pnlSvg = el(pnlSvgId);
  const eqSvg = el(eqSvgId);
  if (pnlSvg) points.length ? renderBarSvg(pnlSvg, points) : svgEmptyState(pnlSvg, "청산된 거래 없음");
  if (eqSvg) points.length ? renderLineSvg(eqSvg, points) : svgEmptyState(eqSvg, "청산된 거래 없음");
}

function renderBtcMultislotShadow(payload) {
  latestBtcMultislotPayload = payload;
  const badge = el("btcMultislotBadge");
  const stale = Boolean(payload?.stale);
  if (badge) {
    badge.className = `ops-badge ${stale ? "bad" : "good"}`;
    badge.textContent = stale ? "STALE" : "LIVE";
  }
  const age = Number(payload?.age_minutes);
  const ageText = Number.isFinite(age) ? `${age.toFixed(age < 10 ? 1 : 0)}분 전` : "-";
  setT("btcMultislotSub", `마지막 bar ${fmtTs(payload?.last_bar)} · ${ageText} 갱신`);
  setT("btcMultislotSlots", `${payload?.open_slots ?? "-"} / ${payload?.slot_count ?? "-"}`);
  setT("btcMultislotTrades", `${payload?.total_trades ?? 0}건`);
  const pnl = Number(payload?.cumulative_return_pct);
  const pnlEl = el("btcMultislotPnl");
  setT("btcMultislotPnl", Number.isFinite(pnl) ? fmtPct(pnl, 2) : "-");
  if (pnlEl) {
    pnlEl.classList.remove("good-text", "bad-text", "muted-text");
    pnlEl.classList.add(`${Number.isFinite(pnl) ? riskClass(pnl) : "muted"}-text`);
  }
  renderBtcMultislotSlots(payload);
  renderShadowCharts(payload, "btcMultislotPnlSvg", "btcMultislotEquitySvg");
}

async function refreshBtcMultislotShadow() {
  const now = Date.now();
  if (now - btcMultislotLastFetchAt < OPS_POLL_MS) return;
  btcMultislotLastFetchAt = now;
  try {
    const res = await fetch(API_BTC_MULTISLOT_SHADOW_URL, { cache: "no-store", headers: btcMultislotEtag ? { "If-None-Match": btcMultislotEtag } : {} });
    if (res.status === 304) return;
    if (!res.ok) throw new Error(`btc multislot shadow ${res.status}`);
    btcMultislotEtag = res.headers.get("ETag") || btcMultislotEtag;
    renderBtcMultislotShadow(await res.json());
  } catch (error) {
    console.error("BTC multislot shadow fetch error:", error);
    const badge = el("btcMultislotBadge");
    if (badge) { badge.className = "ops-badge bad"; badge.textContent = "UNREACHABLE"; }
  }
}

function renderEthOdyssey4Position(payload) {
  const cardEl = el("ethOdyssey4PositionCard");
  if (!cardEl) return;
  const livePrice = Number(latestLivePriceByAsset["eth"] || 0);
  const barSeconds = Number(payload?.bar_seconds || 300);
  cardEl.innerHTML = shadowPositionCardHtml(payload?.position || null, null, livePrice, barSeconds);
}

function renderEthOdyssey4Shadow(payload) {
  latestEthOdyssey4Payload = payload;
  const badge = el("ethOdyssey4Badge");
  const stale = Boolean(payload?.stale);
  if (badge) {
    badge.className = `ops-badge ${stale ? "bad" : "good"}`;
    badge.textContent = stale ? "STALE" : "LIVE";
  }
  const age = Number(payload?.age_minutes);
  const ageText = Number.isFinite(age) ? `${age.toFixed(age < 10 ? 1 : 0)}분 전` : "-";
  setT("ethOdyssey4Sub", `마지막 bar ${fmtTs(payload?.last_bar)} · ${ageText} 갱신`);

  const side = Number(payload?.position_side || 0);
  const posText = side > 0 ? "LONG" : side < 0 ? "SHORT" : "FLAT";
  const src = payload?.position_source_component;
  setT("ethOdyssey4Position", src ? `${posText} (${src})` : posText);

  setT("ethOdyssey4Trades", `${payload?.total_trades ?? 0}건`);

  const pnl = Number(payload?.cumulative_return_pct);
  const pnlEl = el("ethOdyssey4Pnl");
  setT("ethOdyssey4Pnl", Number.isFinite(pnl) ? fmtPct(pnl, 2) : "-");
  if (pnlEl) {
    pnlEl.classList.remove("good-text", "bad-text", "muted-text");
    pnlEl.classList.add(`${Number.isFinite(pnl) ? riskClass(pnl) : "muted"}-text`);
  }

  const mdd = Number(payload?.mdd_pct);
  setT("ethOdyssey4Mdd", Number.isFinite(mdd) ? fmtPctNoPlus(mdd, 2) : "-");

  setT("ethOdyssey4GuardBars", `${payload?.h48qual_guard_active_bars ?? 0}bar`);
  setT("ethOdyssey4VetoBars", `${payload?.zig075_short_veto_bars ?? 0}bar`);

  setT("ethOdyssey4H48qualQuality", qualityText(payload?.h48qual_quality_score, payload?.h48qual_quality_threshold));
  setT("ethOdyssey4Zig075Quality", qualityText(payload?.zig075_quality_score, payload?.zig075_quality_threshold));

  renderEthOdyssey4Position(payload);
  renderShadowCharts(payload, "ethOdyssey4PnlSvg", "ethOdyssey4EquitySvg");
}

async function refreshEthOdyssey4Shadow() {
  const now = Date.now();
  if (now - ethOdyssey4LastFetchAt < OPS_POLL_MS) return;
  ethOdyssey4LastFetchAt = now;
  try {
    const res = await fetch(API_ETH_ODYSSEY4_SHADOW_URL, { cache: "no-store", headers: ethOdyssey4Etag ? { "If-None-Match": ethOdyssey4Etag } : {} });
    if (res.status === 304) return;
    if (!res.ok) throw new Error(`eth odyssey4 shadow ${res.status}`);
    ethOdyssey4Etag = res.headers.get("ETag") || ethOdyssey4Etag;
    renderEthOdyssey4Shadow(await res.json());
  } catch (error) {
    console.error("ETH Odyssey4 shadow fetch error:", error);
    const badge = el("ethOdyssey4Badge");
    if (badge) { badge.className = "ops-badge bad"; badge.textContent = "UNREACHABLE"; }
  }
}

function setupPageTabs() {
  document.querySelectorAll(".page-tab").forEach((button) => button.addEventListener("click", () => {
    const target = button.dataset.pageTab; // "live" | "ops" | "snapshot"
    activePageTab = target;
    el("liveTabPanel")?.classList.toggle("hidden", target !== "live");
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
      macroCalendarLastFetchAt = 0; refreshMacroCalendar();
      sessionAlertsLastFetchAt = 0; refreshSessionAlerts();
      lastSnapshotHistoryFetchAt = 0; maybeFetchSnapshotChartHistory();
    } else {
      btcMultislotLastFetchAt = 0; refreshBtcMultislotShadow(); ethOdyssey4LastFetchAt = 0; refreshEthOdyssey4Shadow();
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

function renderLatestCandleChart() {
  const svg = el("candleSvg");
  syncActiveMarketState();
  if (!svg || !candleHistory.length) return;
  const currentPrice = Number(latestLivePrice || candleHistory[candleHistory.length - 1]?.close || 0);
  const selected = assetDecisionState(latestMainState || latestState, latestCompactState, activeChartAsset);
  const entryPrice = selected ? Number(selected?.position?.entry_price || selected?.entry_price || 0) : 0;
  const riskLevels = selected ? latestChartRiskLevels : [];
  renderCandleSvg(svg, candleHistory, chartJournalRows(), entryPrice, currentPrice, riskLevels);
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

// Liquidation magnet as a chart line -- 2026-08-25, user asked for the same tail_risk-derived
// magnet the gauge above this chart already shows (real @forceOrder events clustered over the last
// 15min) to also appear ON the chart, not just as a number in the gauge. Distinct color
// (--liq-magnet) from --liq-support/--liq-resistance on purpose: this is a different data source
// (real recent events, not the candle-estimated liquidation map nearestLiquidationLevel() draws
// from) and the two can sit close together on the chart, so reusing the S/R colors here would make
// them impossible to tell apart at a glance. Reads latestMainState (updated every tick by render(),
// not scoped to this chart's own ~5min refresh cadence) rather than taking tail as a parameter, so
// this chart's other two callers (maybeFetchSnapshotChartHistory/refreshLiquidationMap) don't each
// need to know how to fetch tail_risk themselves.
function liquidationMagnetLevel() {
  // 2026-08-31: latestMainState.tail_risk is dashboard_state.json's top-level block, written by
  // trading_bot.py for ETH only (see docs/eth_dashboard_multicoin_expansion_design_20260831.md
  // section 2.2) -- no BTC equivalent exists yet, so this line is simply omitted rather than drawn
  // from the wrong coin's data.
  if (activeSnapshotAsset !== "eth") return [];
  const tail = latestMainState?.tail_risk || {};
  const clusterDir = Number(tail.liq_cluster_direction || 0);
  const price = Number(tail.liq_cluster_price || 0);
  if (clusterDir === 0 || !(price > 0)) return [];
  const strength = clamp01(Number(tail.liq_cluster_strength) || 0);
  return [{
    val: price,
    color: "var(--liq-magnet)",
    label: "자석",
    dashed: true,
    width: Math.max(1, Math.min(4, Math.round(1 + strength * 3))),
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
// above; the level list below the chart is still the place to read every level's exact price) and,
// 2026-08-25, the liquidation magnet line (see liquidationMagnetLevel above). Called both right
// after its two 5-min data sources (candles, liquidation map) refresh, AND every ~5s from render()
// (see updateSnapshotCandleLive() above and the call site in render()) so the candle body, the
// magnet line, and the current-price line all stay in sync instead of only some of them moving.
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
  const riskLevels = [...nearestLiquidationLevel(), ...liquidationMagnetLevel()];
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

function chartTouchDistance(touches) {
  if (!touches || touches.length < 2) return 0;
  const dx = touches[0].clientX - touches[1].clientX;
  const dy = touches[0].clientY - touches[1].clientY;
  return Math.hypot(dx, dy);
}

function setupMobileCandleGestures() {
  const svg = el("candleSvg");
  if (!svg) return;

  svg.addEventListener("touchstart", (evt) => {
    if (!isMobileChartMode() || !candleHistory.length) return;
    if (evt.touches.length === 1) {
      mobileChartGesture.panStartX = evt.touches[0].clientX;
      mobileChartGesture.panStartIndex = mobileChartView.start ?? mobileChartMaxStart(candleHistory.length, normalizedMobileChartSize(candleHistory.length));
    } else if (evt.touches.length >= 2) {
      const size = normalizedMobileChartSize(candleHistory.length);
      const start = mobileChartView.start ?? mobileChartMaxStart(candleHistory.length, size);
      mobileChartGesture.pinchStartDistance = chartTouchDistance(evt.touches);
      mobileChartGesture.pinchStartSize = size;
      mobileChartGesture.pinchStartCenter = start + size / 2;
    }
  }, { passive: true });

  svg.addEventListener("touchmove", (evt) => {
    if (!isMobileChartMode() || !candleHistory.length) return;
    evt.preventDefault();
    const total = candleHistory.length;

    if (evt.touches.length >= 2) {
      const distance = chartTouchDistance(evt.touches);
      if (!(distance > 0) || !(mobileChartGesture.pinchStartDistance > 0)) return;
      const rawSize = mobileChartGesture.pinchStartSize * (mobileChartGesture.pinchStartDistance / distance);
      const size = Math.round(clampNum(rawSize, MOBILE_CHART_MIN_CANDLES, Math.min(MOBILE_CHART_MAX_CANDLES, total)));
      const start = Math.round(clampNum(mobileChartGesture.pinchStartCenter - size / 2, 0, mobileChartMaxStart(total, size)));
      mobileChartView.size = size;
      mobileChartView.start = start;
      mobileChartView.followLatest = start + size >= total;
      renderLatestCandleChart();
      return;
    }

    if (evt.touches.length === 1) {
      const rect = svg.getBoundingClientRect();
      const size = normalizedMobileChartSize(total);
      const dx = evt.touches[0].clientX - mobileChartGesture.panStartX;
      const candleDelta = Math.round((-dx / Math.max(rect.width, 1)) * size);
      const start = Math.round(clampNum(mobileChartGesture.panStartIndex + candleDelta, 0, mobileChartMaxStart(total, size)));
      mobileChartView.start = start;
      mobileChartView.size = size;
      mobileChartView.followLatest = start + size >= total;
      renderLatestCandleChart();
    }
  }, { passive: false });

  svg.addEventListener("wheel", (evt) => {
    if (!isMobileChartMode() || !candleHistory.length) return;
    evt.preventDefault();
    const total = candleHistory.length;
    const size = normalizedMobileChartSize(total);
    const start = mobileChartView.start ?? mobileChartMaxStart(total, size);
    const center = start + size / 2;
    const factor = evt.deltaY > 0 ? 1.14 : 0.88;
    const nextSize = Math.round(clampNum(size * factor, MOBILE_CHART_MIN_CANDLES, Math.min(MOBILE_CHART_MAX_CANDLES, total)));
    const nextStart = Math.round(clampNum(center - nextSize / 2, 0, mobileChartMaxStart(total, nextSize)));
    mobileChartView.size = nextSize;
    mobileChartView.start = nextStart;
    mobileChartView.followLatest = nextStart + nextSize >= total;
    renderLatestCandleChart();
  }, { passive: false });

  window.addEventListener("resize", () => {
    mobileChartView.start = null;
    mobileChartView.followLatest = true;
    renderLatestCandleChart();
  });
}

function updateChart(price, timestamp, entryPriceArg = 0, force = false) {
  if (!(Number(price) > 0)) return;
  const tsMs = Date.parse(timestamp || "");
  const ts = Math.floor((Number.isFinite(tsMs) ? tsMs : Date.now()) / 1000);
  const candleTs = Math.floor(ts / (CHART_CANDLE_MIN * 60)) * (CHART_CANDLE_MIN * 60);
  let last = candleHistory[candleHistory.length - 1];
  if (!last || last.time < candleTs) {
    last = { time: candleTs, open: price, high: price, low: price, close: price };
    candleHistory.push(last);
    if (candleHistory.length > CHART_MAX_CANDLES) candleHistory.shift();
  } else {
    last.high = Math.max(last.high, price); last.low = Math.min(last.low, price); last.close = price;
  }
  const now = Date.now();
  if (!force && now - lastChartRenderAt < CHART_RENDER_MIN_INTERVAL_MS) return;
  lastChartRenderAt = now;
  const svg = el("candleSvg");
  if (svg) {
    const selected = assetDecisionState(latestMainState || latestState, latestCompactState, activeChartAsset);
    const entryPrice = entryPriceArg || (selected ? Number(selected?.position?.entry_price || selected?.entry_price || 0) : 0);
    const riskLevels = selected ? latestChartRiskLevels : [];
    renderCandleSvg(svg, candleHistory, chartJournalRows(), entryPrice, price, riskLevels);
  }
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
  // the Snapshot tab's chart (svg id "candleSvgSnapshot", always ETH -- see that call site's own
  // comment); latestRegimeWide24 is always ETH too, so other renderCandleSvg() callers (BTC/SOL on
  // the Live tab) must not draw or tooltip-show it.
  const isSnapshotChart = svg.id === "candleSvgSnapshot";
  const regimeByTsForChart = isSnapshotChart && latestRegimeWide24 && latestRegimeWide24.warmed_up
    ? new Map((latestRegimeWide24.history || []).map((r) => [Math.floor(r.ts_ms / 1000), r]))
    : null;
  // 2026-08-27 user report: ribbon "turns black" and stops updating for stretches -- tracing the
  // draw loop below, it never paints an invalid color (regimeDominant() only ever returns one of
  // the 3 REGIME_DOMINANT_COLOR keys); what actually happens is this block draws literally nothing
  // whenever latestRegimeWide24.warmed_up is false (backend regime compute degrades to that instead
  // of raising, per load_regime_wide24()'s own docstring), so the ribbon's row just shows the dark
  // chart background underneath -- indistinguishable from "black" at a glance, and easy to mistake
  // for a frozen/broken ribbon rather than "no fresh reading available for this window". Flagging
  // that state explicitly below instead of silently drawing nothing.
  const regimeRibbonWaiting = isSnapshotChart && !regimeByTsForChart;
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
  } else if (regimeRibbonWaiting && candles.length) {
    // See regimeRibbonWaiting's definition above -- an explicit flat placeholder instead of
    // silently drawing nothing, so a temporary backend hiccup reads as "waiting", not "broken".
    const waitRect = document.createElementNS(NS, "rect");
    waitRect.setAttribute("x", xAt(0)); waitRect.setAttribute("y", REGIME_RIBBON_Y);
    waitRect.setAttribute("width", xAt(candles.length - 1) + bw - xAt(0)); waitRect.setAttribute("height", REGIME_RIBBON_H);
    waitRect.setAttribute("rx", "1.5");
    waitRect.setAttribute("fill", "var(--muted)");
    waitRect.setAttribute("fill-opacity", "0.18");
    const waitTitle = document.createElementNS(NS, "title");
    waitTitle.textContent = "레짐: 웜업 중이거나 일시적으로 갱신 실패 -- 다음 5분 주기에 자동 재시도됩니다";
    waitRect.appendChild(waitTitle);
    svg.appendChild(waitRect);
    const waitLabel = document.createElementNS(NS, "text");
    waitLabel.setAttribute("x", ml - 6);
    waitLabel.setAttribute("y", REGIME_RIBBON_Y + REGIME_RIBBON_H - 1);
    waitLabel.setAttribute("text-anchor", "end");
    waitLabel.setAttribute("font-size", "9");
    waitLabel.setAttribute("fill", "var(--muted)");
    waitLabel.textContent = "레짐";
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

function render(state, compactState = null, { stateChanged = true, journalChanged = true } = {}) {
  const shadowState = usableGovernorShadowState(compactState);
  const activeState = shadowState || state;
  syncActiveMarketState();
  const chartState = assetDecisionState(state, compactState, activeChartAsset);
  latestState = activeState;
  latestMainState = state;
  latestCompactState = shadowState;
  try {
    latestChartRiskLevels = chartState ? chartRiskLevels(state, compactState, chartState) : [];
	    const currentP = Number(latestLivePrice || chartState?.last_price || chartState?.price || activeState.last_price || activeState.price || 0);
	    const pos = chartState ? openPosition(chartState) : null;
	    const entryP = Number(pos?.entry_price || 0);
	    updateChart(currentP, latestLivePriceTs || chartState?.updated_at || chartState?.cycle_timestamp_kst || activeState.updated_at || activeState.cycle_timestamp_kst, entryP);
	    const riskLineText = latestChartRiskLevels.length ? `리스크 라인: ${latestChartRiskLevels.map((x) => `${x.label} ${fmtNum(x.val, 2)}`).join(" / ")}` : "";
	    setT("riskLevelNote", riskLineText || "-");
	  } catch (e) { console.error("Chart Update Error:", e); }
  
  const globalStamp = fmtTs(state.updated_at || state.cycle_timestamp_kst);

  renderExecutionAlert(state, compactState);
  renderOpsCards(state, compactState);
  renderCombinedUnrealizedPnl(state, compactState);
  setT("chartStamp", latestLivePriceTs ? fmtTs(latestLivePriceTs) : globalStamp);
  if (journalChanged) {
    renderTradePanels();
    tradePanelsRendered = true;
  }
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

  // V자 반등락 (2026-08-29, TabPFN Tier0+rsi 모델, 2026-08-30 "유동성스윕 반등예측"에서 개명) -- fetched
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
  const vReboundTone = vReboundActive ? (latestVRebound.tone || "warn") : "neutral";
  // 2026-08-31 user request: 미반등 콜(반등 시도 자체가 없었다는 판정)을 더는 급등/급락으로 억지로
  // 묶지 않고 방향 무관 "미반등"으로 따로 표시 -- tone="flat"(백엔드 _predicted_tone, 2026-08-31
  // 개정)일 때 전용 단어. good/bad(진짜 반등 콜)만 급등/급락을 씁니다.
  const vReboundSubText = !vReboundWarmedUp ? "웜업 중"
    : !vReboundActive ? "대기"
    : vReboundTone === "good" ? "급등"
    : vReboundTone === "bad" ? "급락"
    : "미반등";
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

    // Bug found 2026-08-25: renderSnapshotChart() (candles + S/R line + liquidationMagnetLevel())
    // used to be called ONLY from the two data-fetch functions that feed it, each gated to a 5-minute
    // interval (maybeFetchSnapshotChartHistory/refreshLiquidationMap) -- a reasonable cadence for
    // candles/the liquidation map, since neither source changes faster than that. But
    // liquidationMagnetLevel() reads latestMainState.tail_risk, which updates on every SSE tick (this
    // render() call itself) -- so the magnet line could sit stale for up to 5 minutes after a real
    // change, or simply never have painted yet if the chart's first 5-min-gated render happened
    // before tail_risk had arrived. Throttled to SNAPSHOT_CHART_RENDER_MIN_INTERVAL_MS, same
    // pattern the Live tab's own chart uses for its own frequent-tick redraws (own constant since
    // 2026-08-25 -- see its definition for why Snapshot can afford a coarser interval) -- cheap
    // since renderSnapshotChart() only redraws from already-cached data, no network fetch of its own.
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
    // above -- V자 반등락 is the first resident (TabPFN, fires only on a liquidity_sweep, long idle
    // "대기" gaps between events), more will land here over time. Reuses renderModelIndicatorList's
    // row/strip markup verbatim (2nd param = its own target list, own memoized-html slot) rather than
    // a new template -- same reasoning as the model-indicator/evidence-signal panels already sharing
    // one markup. Append new specialized-detector objects to this array as they're built.
    renderModelIndicatorList([
      {
        key: "v_rebound", label: "V자 반등락", tone: vReboundTone, subText: vReboundSubText,
        history: (latestVRebound && latestVRebound.history) || [],
        times: (latestVRebound && latestVRebound.times) || [],
        liveText: vReboundProbaShown != null
          ? `${vReboundSubText} 확률(TabPFN) ${Math.round(vReboundProbaShown * 100)}%`
          : "",
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
    const refreshCandles = maybeFetchBinanceHistory();
    const refreshJournals = refreshTradeJournals(Date.now());
    const [, journalChanged] = await Promise.all([refreshCandles, refreshJournals]);
    if (journalChanged && latestMainState && !isScrolling) {
      renderTradePanels();
      tradePanelsRendered = true;
    }
    refreshOpsStatus();
    refreshBtcMultislotShadow();
    refreshEthOdyssey4Shadow();
    // 2026-08-25: perf pass -- these 6 only matter while the Snapshot tab is actually visible;
    // gating them stops background fetch/compute work for a hidden panel (see activePageTab,
    // set by setupPageTabs()'s click handler). The tab-click force-refresh block in
    // setupPageTabs() still fires immediately on switching to Snapshot, so this doesn't delay
    // first paint after a tab switch -- it only stops the ongoing poll while elsewhere.
    if (activePageTab === "snapshot") {
      refreshEvidenceSignals();
      refreshEvidenceSignalsProvisional();
      refreshVReboundSignal();
      refreshLiquidation5mSignal();
      refreshBasisLiquiditySignal();
      refreshLiqBurstState();
      refreshLiquidationDirectionSignal();
      refreshLiquidationMap();
      refreshRegimeWide24();
      refreshMacroCalendar();
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
setupAssetTabs();
setupSnapshotAssetTabs();
setupPageTabs();
setupScrollRendering();
setupMobileCandleGestures();

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
