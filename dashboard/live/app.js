const TRADE_JOURNAL_URL = "../../data/live/trade_journal.jsonl";
const API_EVENTS_URL = "/api/events";
const API_TRADES_URL = "/api/trades";
const API_OPS_STATUS_URL = "/api/ops-status";
const API_BTC_MULTISLOT_SHADOW_URL = "/api/btc-multislot-shadow";
const API_ETH_ODYSSEY4_SHADOW_URL = "/api/eth-odyssey4-shadow";
const API_EVIDENCE_SIGNALS_URL = "/api/evidence-signals";
const API_EVIDENCE_SIGNALS_PROVISIONAL_URL = "/api/evidence-signals-provisional";
const API_OI_SIGNAL_URL = "/api/oi-signal";
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
// indicators like oi_delta/liq_pressure/liq_direction, which carry their own server-provided
// tone_history instead), but as a tone-per-bar strip (matching the evidence signals' activity-strip
// graph) instead of a continuous sparkline. Stores the ALREADY-COMPUTED tone string
// ("good"/"bad"/"neutral") from each render() pass rather than re-deriving it from raw values
// later -- some tones (tail risk) depend on more than one raw field, so capturing the tone at
// computation time is the only way to stay exactly consistent with what the live cards show,
// instead of an approximation that ignores the cross-field dependency.
const toneHistory = { whale: [], whale_intent: [], risk: [], liq_cascade: [], retail_flow: [] };
// Parallel to toneHistory, same keys/push/shift cadence -- these 5 indicators have no server-side
// timestamp per reading (client-accumulated tally, see comment above), so the only honest per-bar
// time is "when this browser tab actually pushed the reading", recorded here at push time.
const toneHistoryTimes = { whale: [], whale_intent: [], risk: [], liq_cascade: [], retail_flow: [] };
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
// OI 급변 model indicator (replaces OBI, 2026-08-24) -- fetched separately from classifyIndicators'
// other 5 fields because it's computed dashboard-side (scripts/live_oi_delta_signal_20260824.py),
// not read from trading_bot.py's dashboard_state.json. render() reads this module-level var each
// tick, same pattern as evidence signals feeding their own chip row independently of classifyIndicators.
let latestOiSignal = null;
let oiSignalLastFetchAt = 0;
// Long/short liquidation volume gauge (recreated 2026-08-27, see liquidationVolumeGaugeHtml()) --
// backend (scripts/live_liquidation_5m_signal_20260825.py) never stopped running, only this
// frontend consumer had been removed.
let latestLiquidation5m = null;
let liquidation5mLastFetchAt = 0;
// 베이시스 청산압박 model indicator (replaces 독성/toxicity, 2026-08-27) -- own fetch cycle, same
// dashboard-side-computed category as latestOiSignal above (scripts/live_spot_perp_basis_signal_
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
// dashboard-side-computed category as latestOiSignal above. Model-indicator tier (like 수급
// 흐름/고래 포지션), explicitly NOT an evidence-signal-tier chip -- see
// scripts/live_liquidation_direction_signal_20260825.py docstring for why (no PnL/economic claim).
let latestLiquidationDirection = null;
let liquidationDirectionLastFetchAt = 0;
// Liquidation map (Snapshot tab, 2026-08-24) -- estimated support/resistance, own fetch/render
// cycle same as latestOiSignal above (computed dashboard-side, not part of trading_bot.py state).
// lastSnapshotHistoryFetchAt tracks the ETH candle history this panel's chart needs independently
// of activeChartAsset (the Live tab's chart may be showing SOL/BTC while this stays ETH-only).
let latestLiquidationMap = null;
let latestRegimeWide24 = null;
let liquidationMapLastFetchAt = 0;
let regimeWide24LastFetchAt = 0;
let macroCalendarLastFetchAt = 0;
let sessionAlertsLastFetchAt = 0;
let lastSnapshotHistoryFetchAt = 0;
let lastChartRenderAt = 0;
let lastSnapshotChartRenderAt = 0;
let lastModelIndicatorHtml = "";
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
const OI_SIGNAL_POLL_MS = 300000; // same reasoning -- underlying data is a 5m poller, no faster
const LIQUIDATION_5M_POLL_MS = 60000; // matches server's own 60s cache + the 1-row-per-minute source
const BASIS_LIQUIDATION_POLL_MS = 300000; // same reasoning -- basis_z48 is a 5m-bar z-score, no faster
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
  document.querySelectorAll(".asset-tab").forEach((btn) => {
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
  document.querySelectorAll(".asset-tab").forEach((btn) => {
    btn.addEventListener("click", () => setActiveChartAsset(btn.dataset.asset));
  });
  renderAssetTabs();
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

function whalePositionRead(micro) {
  const score = Number(micro.whale_position_score || 0);
  const confidence = Number(micro.whale_position_confidence || 0);
  if (score >= 0.25) return `고래 포지션이 롱 쪽으로 기움`;
  if (score <= -0.25) return `고래 포지션이 숏 쪽으로 기움`;
  if (confidence >= 70) return "고래 포지션은 관망에 가까움";
  return "고래 포지션 판단 약함";
}

// 꼬리리스크는 "위험도" 지표이지 방향(롱/숏) 매매신호가 아니다 -- 예전엔
// obi/z_bias 부호를 빌려와 "롱 진입"/"숏 진입"으로 표시했는데, 이러면 위험이 높을 때도
// signalTone()이 우연히 "good"(녹색)을 줘서 "진입해도 된다"는 잘못된 인상을 줄 수 있었다
// (2026-08-24 사용자 리포트로 발견). 방향 힌트가 필요하면 tailRiskRead() 함수가 이미
// "상방/하방 ~위험"처럼 서술형으로 제공하므로 여기서는 순수 위험도만 반환.
// 2026-08-27: replaces toxRead/toxHint (독성/toxicity chip removed -- shadow_toxicity_score was
// independently confirmed uninformative on both direction and volatility-framing axes, see
// eth_model_indicator_volatility_framing_screen_20260825 memory). sig here is the raw
// latestBasisLiquidation payload (server-computed, not part of classifyIndicators' micro/tail
// inputs -- same "own fetch cycle" category as latestOiSignal, see that variable's own comment).
function basisLiquiditySubText(sig) {
  if (!sig || !sig.warmed_up) return "웜업 중";
  if (sig.direction === "short_pressure") return "숏압박↑";
  if (sig.direction === "long_pressure") return "롱압박↑";
  return "안정";
}

function tailRiskRead(tail) {
  const x = clamp01(tail.aftershock_prob);
  const dir = Number(tail.z_bias || 0);
  if (x >= 0.7) return dir < 0 ? "하방 급변 위험 높음" : dir > 0 ? "상방 급변 위험 높음" : "급변 위험 높음";
  if (x >= 0.4) return "급변 가능성 주의";
  return "꼬리 리스크 안정";
}

function tailRiskHint(tail) {
  const x = clamp01(tail.aftershock_prob);
  if (x < 0.4) return "안정";
  if (x >= 0.7) return "위험";
  return "주의";
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

// 안정=문제없음(녹색), 주의=경계(호박색), 위험=경계강함(적색) -- risk류(리스크게이지)
// 지표 전용. 방향성 매매신호(롱 진입/숏 진입)는 whale/whale_intent가 directionalCaution()로
// 별도 처리하므로 여기서 다루지 않는다.
function signalTone(signal) {
  const s = String(signal || "");
  if (s === "위험") return "bad";
  if (s.includes("주의")) return "warn";
  if (s === "안정") return "good";
  return "neutral";
}

// Single source of truth for the 5 model-internal indicators' tone/read-text classification --
// called both on the live state (render(), every tick) and on server-provided history samples
// (seedModelIndicatorHistory(), once at page load) so there is exactly one copy of these
// thresholds, not a live copy and a history copy that could quietly drift apart.
function classifyIndicators(micro, tail) {
  micro = micro || {};
  tail = tail || {};
  const riskV = clamp01(tail.aftershock_prob);
  const riskSignal = tailRiskHint(tail);
  const cascadeSignal = liqCascadeHint(tail);
  const whaleTone = Number(micro.nif_whale || 0) > 0.05 ? "good" : (Number(micro.nif_whale || 0) < -0.05 ? "bad" : "neutral");
  const whalePosTone = Number(micro.whale_position_score || 0) > 0.2 ? "good" : (Number(micro.whale_position_score || 0) < -0.2 ? "bad" : "neutral");
  const retailFlowTone = Number(micro.nif_retail || 0) > 0.05 ? "good" : (Number(micro.nif_retail || 0) < -0.05 ? "bad" : "neutral");
  const riskTone = signalTone(riskSignal);
  const cascadeTone = signalTone(cascadeSignal);
  return {
    risk: { tone: riskTone, valueText: tailRiskRead(tail), subText: riskSignal },
    liq_cascade: { tone: cascadeTone, valueText: liqCascadeLiveDetail(tail), subText: cascadeSignal },
    whale: { tone: whaleTone, valueText: flowRead(micro), subText: directionalCaution(micro.nif_whale, 0.05) },
    retail_flow: { tone: retailFlowTone, valueText: retailFlowRead(micro), subText: directionalCaution(micro.nif_retail, 0.05) },
    whale_intent: { tone: whalePosTone, valueText: whalePositionRead(micro), subText: directionalCaution(micro.whale_position_score, 0.2) },
    riskV, riskSignal, cascadeSignal, whaleTone, whalePosTone, riskTone, cascadeTone,
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
  const cached = candleHistoryByAsset.eth || [];
  if (cached.length && now - lastSnapshotHistoryFetchAt < CANDLE_HISTORY_POLL_MS) return;
  lastSnapshotHistoryFetchAt = now;
  await fetchBinanceHistory("eth");
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
// spaced (server-computed histories: evidence signals/oi_delta at 5-min klines, liq_direction at
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
function toneStripSvg(tones, times, provisionalLast, liveFiring) {
  const list = Array.isArray(tones) ? tones : [];
  const timeList = Array.isArray(times) ? times : [];
  const n = Math.max(list.length, 1);
  const w = 240, h = 20, gap = 1.5;
  const bw = Math.max((w - gap * (n - 1)) / n, 1);
  const bars = [];
  for (let i = 0; i < n; i++) {
    const tone = list[i] || "neutral";
    // 2026-08-25: this mapping originally had no "warn" branch at all (fell into the generic gray
    // fallback below), then briefly used --amber (yellow) to match .signal-chip.warn's color at the
    // time -- user then asked for the whole Snapshot tab's 주의 color to be yellow-free and unified
    // on --warn (orange) instead, so this now matches that.
    const fill = tone === "good" ? "var(--good)" : tone === "bad" ? "var(--bad)" : tone === "warn" ? "var(--warn)" : "rgba(203,209,227,0.16)";
    const isLast = i === n - 1;
    // 2026-08-27 (user request): the last/rightmost bar used to always get a distinct outline
    // (evidence-bar-now, blinking at first, then static) just for being the "now" position --
    // removed entirely, position alone isn't a meaningful signal on its own. evidence-bar-live
    // (tone-colored pulse) still applies when the last bar is actively firing; the whole-gauge
    // evidence-strip-live blink (see toneStripSvg's return) is the only thing marking "now" at all,
    // and only while genuinely live/provisional.
    let cls = "evidence-bar";
    if (isLast && tone !== "neutral") cls += " evidence-bar-live";
    // 2026-08-26: when the caller appends a still-forming bar (see evidenceStripSvg's liveTone
    // param), that bar lands here as the new last index -- this class marks it as "not yet
    // confirmed" (softened fill, see .evidence-bar-provisional), same honesty-signal requirement
    // as the provisional badge/chip dots elsewhere (see renderEvidenceSignalsProvisional).
    if (isLast && provisionalLast) cls += " evidence-bar-provisional";
    const x = (i * (bw + gap)).toFixed(1);
    // data-t carries a plain ISO string (digits/-/:/./T/Z only) -- safe unescaped in an HTML
    // attribute, and read back + formatted at hover time so fmtShortTs runs once per hover instead
    // of once per bar per render.
    const t = timeList[i];
    const hoverAttrs = t ? ` data-t="${t}" onmouseenter="showStripBarTime(this)" onmouseleave="hideStripBarTime(this)"` : "";
    bars.push(`<rect class="${cls}" x="${x}" y="0" width="${bw.toFixed(1)}" height="${h}" rx="2" fill="${fill}"${hoverAttrs}/>`);
  }
  // 2026-08-27 (user request): the whole gauge blinks, but only while it's showing a genuinely
  // in-progress reading -- the still-forming bar (liveFiring, from evidenceStripSvg's liveTone) is
  // both provisional AND currently non-neutral. A provisional-but-neutral forming bar (most common
  // case) or a fully confirmed render (model indicators always, evidence signals between polls)
  // stays static -- blink is reserved for "something is actively firing right now, not yet final".
  return `<svg class="evidence-strip${liveFiring ? " evidence-strip-live" : ""}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">${bars.join("")}</svg>`;
}

// liveTone/liveIso (2026-08-26, optional) append one extra bar for the still-forming (unconfirmed)
// bar after latestIso, sourced from the provisional preview -- see refreshEvidenceSignalsProvisional,
// which re-calls this every ~10s reusing the SAME confirmed bottomHist/topHist/latestIso (cached in
// evidenceHistoryBySignal) so the 47 confirmed bars don't flicker, only the new live one changes.
function evidenceStripSvg(bottomHist, topHist, latestIso, stepMinutes, liveTone, liveIso) {
  const n = Math.max(bottomHist.length, topHist.length, 1);
  const tones = Array.from({ length: n }, (_, i) => (bottomHist[i] ? "good" : topHist[i] ? "bad" : "neutral"));
  const times = evenlySpacedBarTimes(latestIso, n, stepMinutes);
  if (liveTone) { tones.push(liveTone); times.push(liveIso || ""); }
  return toneStripSvg(tones, times, !!liveTone, !!liveTone && liveTone !== "neutral");
}

// .strip-time-now (rendered by each row template, NOT inside the strip) defaults to the latest/
// current analysis time (data-default, set once at render time) and switches to the hovered bar's
// own time while the cursor is over the strip -- 2026-08-25 user request: "지금 현재 시간을
// 표시해주고, 마우스를 올리면 마지막 분석 시간이 아닌 커서를 올린 분석 시간을 표시". data-fmt="time"
// (model indicators only) picks the HH:MM:SS-only formatter; its absence (evidence signals) falls
// back to fmtShortTs (MM-DD HH:MM), matching that row's existing 바닥/천장 caption format.
function showStripBarTime(rectEl) {
  const iso = rectEl.getAttribute("data-t");
  if (!iso) return;
  const label = rectEl.closest(".ops-health-info")?.querySelector(".strip-time-now");
  if (!label) return;
  const fmt = label.getAttribute("data-fmt") === "time" ? fmtTimeOnly : fmtShortTs;
  label.textContent = fmt(iso);
}

function hideStripBarTime(rectEl) {
  const label = rectEl.closest(".ops-health-info")?.querySelector(".strip-time-now");
  if (label) label.textContent = label.getAttribute("data-default") || "-";
}

// Same row/strip UI as renderEvidenceSignals(), but for the 6 model-internal indicators. The two
// panels LOOK identical on purpose (same ops-health-row/-strip markup) -- the caption on every
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
  oi_delta: {
    "안정": "OI(미결제약정)가 평소와 비슷한 속도로 움직이고 있어요 — 변동성 확대 신호는 없어요.",
    "주의": "OI가 평소보다 빠르게 움직이고 있어요 — 새 포지션이 몰리거나 정리되는 중일 수 있고, 이후 변동성이 조금 커질 가능성이 있어요.",
    "위험": "OI가 평소보다 훨씬 빠르게 움직이고 있어요 — 실측상 이후 1시간 변동폭이 평소의 1.35배 이상으로 커지는 경향이 있었어요. 방향은 알려주지 않으니 사이즈·스탑 여유를 고려하세요.",
  },
  liq_pressure: {
    "안정": "현물-선물 가격차(베이시스)가 평소 범위 안이라, 어느 한쪽이 특별히 강제청산 압박을 더 받을 조짐은 안 보여요.",
    "숏압박↑": "베이시스가 콘탱고(선물이 현물보다 비쌈) 쪽 극단이에요 — 실측상 이런 국면 이후 1~4시간 숏 강제청산액이 늘고 롱 청산액은 줄어드는 경향이 있었어요(약 1개월치 탐색적 관측). 가격이 오른다는 뜻은 아니고, '숏 쪽이 청산 압박을 더 받을 수 있다'는 리스크 정보예요.",
    "롱압박↑": "베이시스가 백워데이션(선물이 현물보다 쌈) 쪽 극단이에요 — 실측상 이런 국면 이후 1~4시간 롱 강제청산액이 늘고 숏 청산액은 줄어드는 경향이 있었어요(약 1개월치 탐색적 관측). 가격이 내린다는 뜻은 아니고, '롱 쪽이 청산 압박을 더 받을 수 있다'는 리스크 정보예요.",
  },
  risk: {
    "안정": "최근 청산 활동이 평온해서 연쇄 청산으로 인한 급변 가능성은 낮아요.",
    "주의": "최근 청산이 늘어나는 중이라 그 여파로 후속 급변동이 올 수 있어요. 포지션을 보수적으로 가져가는 걸 고려하세요.",
    "위험": "청산 캐스케이드가 진행 중이거나 막 끝났을 가능성이 높아요. 신규 진입은 특히 주의하세요.",
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
  whale_intent: {
    "롱 진입": "고래가 롱 포지션을 새로 쌓고 있는 것으로 추정돼요(미결제약정 증가 동반) — 단순 매수 체결보다 신뢰도 있는 신호로 설계됐어요.",
    "숏 진입": "고래가 숏 포지션을 새로 쌓고 있는 것으로 추정돼요(미결제약정 증가 동반) — 단순 매도 체결보다 신뢰도 있는 신호로 설계됐어요.",
    "중립": "고래 포지션 방향이 뚜렷하지 않거나 관망 상태로 추정돼요.",
  },
};

const MODEL_INDICATOR_DETAIL = {
  oi_delta: "[계산] 5분마다 미결제약정(OI) 변화량을 직전 1일(288개 샘플) 평균·표준편차로 정규화한 z값.\n" +
    "[기준] |z| 2.0 이상 위험 · 1.0~2.0 주의 · 1.0 미만 안정.\n" +
    "[의미] '포지션이 얼마나 빨리 쌓이거나 풀리는가'만 봅니다 — 방향(롱/숏)은 전혀 반영하지 않습니다(호가 불균형이 방향을 주장하다 45/45 경제성 실패로 걷힌 자리라, 일부러 방향 주장을 안 하도록 설계). 실측(2024-01~2026-02, 약 22만 봉, 사전등록 없는 탐색적 분석)상 |z|≥2일 때 이후 1시간 변동폭이 평소의 1.35배, 4시간은 1.19배 컸고, OI 급증·급감 양쪽 다 비슷한 크기로 변동성이 커졌습니다(신규 포지션 구축이든 급청산이든 둘 다 변동성 전조). 기존 '에너지 지수'(OI변화/가격변동폭 비율)와는 상관 0.15로 중복 아님을 확인했습니다.\n" +
    "[유의] 이 값은 봇 내부 상태가 아니라 대시보드 서버가 별도 duckdb(OI 전용 poller)에서 직접 계산합니다 — 아직 실제 매매 결정에는 연결되지 않았습니다. 꼬리 리스크/수급 흐름/고래 포지션은 봇이 실제로 참고하는 값인 것과 다릅니다.",
  liq_pressure: "[계산] basis_raw = (선물 종가 − 현물 종가) / 현물 종가 (ETHUSDT, fapi.binance.com 선물 vs api.binance.com 현물). basis_z48 = basis_raw를 직전 48봉(4시간) 평균·표준편차로 정규화한 z값.\n" +
    "[기준] |z| 2.0 이상 위험 · 1.0~2.0 주의 · 1.0 미만 안정. 양수 극단=콘탱고(숏압박↑ 힌트), 음수 극단=백워데이션(롱압박↑ 힌트).\n" +
    "[의미] 2026-08-20에 '베이시스가 방향(다음 봉이 오를지 내릴지)을 예측하는가'로 먼저 테스트했으나 REJECTED(구간마다 부호가 뒤집힘) — 문헌(Schmeling/Schrimpf/Todorov 'Crypto Carry' BIS WP1087; He/Manela/Ross/von Wachter arXiv:2212.06888)이 원래 예측한 축은 방향이 아니라 '미래 변동성'과 '청산 크라우딩'이었습니다. 2026-08-27 그 방향으로 재검정: 변동성 예측은 이것도 부호가 안정적이지 않아 REJECTED급이었지만, 청산 크라우딩(어느 쪽이 강제청산 더 받는가)은 실제 청산 데이터(tail_risk.duckdb)로 확인한 결과 문헌과 부호까지 정확히 일치했습니다 — 베이시스 극단(양수) 이후 1h/4h 숏청산액이 유의하게 늘고(z=+3.9~+4.4) 롱청산액은 유의하게 줄었습니다(z=-4.3~-5.7), 음수 극단은 반대.\n" +
    "[유의] 이 청산크라우딩 검증은 **약 1개월치 탐색적 표본**입니다(청산 데이터의 신뢰 가능 구간이 2026-07-18부터 시작) — 이 저장소가 다른 모든 신호에 쓰는 VAL/OOS 3-split 재현성 검증은 아직 못 거쳤습니다. 표본이 더 쌓이면 정식 재검증 예정. 가격이 오르내린다는 뜻이 아니라 '어느 쪽 포지션이 청산 압박을 더 받을 가능성'만 알려주는 리스크 정보입니다. 봇 내부 상태가 아니라 대시보드 서버가 매 사이클 spot/perp klines를 직접 fetch해 계산합니다 — 아직 실제 매매 결정에는 연결되지 않았습니다.",
  risk: "[계산] 최근 1분간 롱/숏 강제청산 금액을 각각 자기 과거 평균·표준편차로 정규화(z값) 후 더 큰 쪽(z_peak). 롱/숏 청산 쏠림 비율(imbalance). 지진 여진 모델(Hawkes process)식으로, 큰 청산 발생 후 시간이 지나며 지수적으로 잦아드는 '활성 상태' 값. prob = 0.45×(z_peak/기준치) + 0.35×쏠림비율 + 0.20×활성상태.\n" +
    "[기준] 0.7 이상 급변위험 높음 · 0.4~0.7 급변 가능성 주의 · 미만 꼬리 리스크 안정.\n" +
    "[의미] '여진(aftershock)'이라는 이름 그대로, 청산 캐스케이드(한쪽 포지션들이 연쇄적으로 강제청산되며 가격이 그 방향으로 더 튀는 현상)가 막 일어났거나 후속 충격이 올 확률을 추정합니다. 큰 충격 직후엔 이 값이 즉시 0으로 리셋되지 않고 서서히 감쇠합니다.",
  liq_cascade: "[계산] 최근 1분 롱/숏 청산 금액을 각각 30분 과거 평균·표준편차로 정규화한 Z값 중 큰 쪽(z_peak). z_peak이 임계값(3.5)을 넘으면 그 순간부터 '캐스케이드 진행중'으로 전환되고, 이후 시간이 지나며 지수적으로 감쇠합니다(현재 파라미터 기준 반감기 약 2~3분). 감쇠된 에너지가 임계값의 35% 아래로 내려가면 캐스케이드가 종료된 것으로 판정합니다. 2026-08-27부터(ETH만) Z값 조건에 더해 그 쪽 청산 금액이 최소 $10,000 이상이어야 전환됩니다 — 청산이 성긴(대부분 분(分)이 $0) 데이터라 조용한 구간이 이어지면 평균·표준편차 자체가 거의 0으로 붕괴해, 그 직후엔 평범한 청산 하나에도 Z값만으로는 오검출되던 문제를 막기 위함입니다.\n" +
    "[기준] 캐스케이드 진행중이면 위험 · 아직 진행중은 아니지만 Z≥2.0(청산 급증)이면 주의 · 그 외 안정.\n" +
    "[의미] 꼬리 리스크(aftershock_prob)가 'z_peak+쏠림비율+활성상태'를 섞어 향후 추가 충격 확률을 종합 추정하는 지표라면, 이건 그 재료 중 하나인 '지금 이 순간 실제로 캐스케이드가 벌어지고 있는가'를 가공 없이 그대로 보여주는 원시 상태값입니다. 방향(롱/숏)이나 에너지 잔량(감쇠율) 같은 세부 숫자는 화면에 안 나옵니다 — 안정/주의/위험 배지만으로 충분하다는 판단(사용자 요청으로 정리)이라, 이 지표는 '지금 캐스케이드가 있는가/없는가'만 한눈에 보는 용도입니다.",
  liq_direction: "[계산] liq_net_z_12 = (최근 12분 롱청산 합 − 숏청산 합) / (최근 2일 총청산 롤링평균 + 1% 여유값). 양수면 롱청산 우세, 음수면 숏청산 우세.\n" +
    "[갱신 주기] 원천 데이터(tail_risk_1m)가 1분마다 1행씩 쌓이고 서버도 60초 캐시를 걸어서, 이 값은 최소 1분에 한 번만 바뀝니다 — 수급 흐름/고래 포지션처럼 몇 초 단위로 바뀌는 지표가 아닙니다. 그래프도 1분×48칸, 즉 최근 48분을 보여줍니다.\n" +
    "[기준] 부호로 방향(상승압력/하락압력), 최근 이력 대비 백분위(상하위 10% 안이면 '강한', 25~75% 사이면 '약한')로 세기를 표시.\n" +
    "[의미] 컨트래리언 해석입니다 — 롱청산이 몰리면(강제매도 소진) 상승압력, 숏청산이 몰리면(숏스퀴즈 소진) 하락압력으로 읽습니다. **2026-08-25 정식 IC 검증(37일, n>10,200)**: 5분·15분 지평은 forward-return과의 상관이 통계적으로 유의(순열검정 z=+2.96/+2.50, 전반·후반 구간 모두 같은 부호), 1시간 지평은 근소 미달(z=+1.91)이지만 상관 크기 자체는 문턱을 넘고 전반/후반 부호·크기도 일관돼(+0.041/+0.035) 표본부족(정식 문턱 56일의 66%) 때문으로 보입니다 — 방향 정보 자체는 탄탄합니다. 다만 **같은 원천 데이터로 스윕과 결합해 실제 손익을 검정한 결과(§14, 08-25)는 8개 지평 전부 비용 차감 후 순손실**이었습니다(15분 -9.19bp~2시간 -5.00bp, taker 10bp 기준) — 통계적으로 진짜인 정보와 수수료를 넘기고 이익이 나는지는 별개 질문입니다. 방향 부호는 참고할 만하지만 이 신호 하나만으로 매매를 결정할 만큼 이익을 낸다는 근거는 없습니다 — 수급 흐름/고래 포지션과 마찬가지로 재량 판단의 한 재료로만 쓰세요.\n" +
    "[유의] 09-15 정식 게이트(56일치 데이터, §13/§14 포함)가 이 조기 IC 결과를 대체합니다 — 지금 수치는 37일치 조기 계산입니다.",
  whale: "[계산] 최근 5분간 체결을 건당 금액 기준으로 큰손/소액으로 나눠, 큰손 거래만 (매수금액−매도금액)/(매수금액+매도금액)으로 계산 (-1~+1).\n" +
    "[기준] +0.2 이상 강하게 매수유입 · +0.05~0.2 매수유입 · -0.05~-0.2 매도유입 · -0.2 이하 강하게 매도.\n" +
    "[의미] 큰 금액 단위 거래(개인 소액 매매와 구분)가 최근 5분간 실제로 어느 방향으로 체결됐는지를 보여줍니다. '포지션'이 아니라 '최근 흐름'이라는 점에 유의하세요 — 아래 '고래 포지션'이 여기에 미결제약정 변화까지 더해 포지션 방향을 추정합니다.",
  retail_flow: "[계산] 위 '수급 흐름'과 같은 함수·같은 5분창에서 나온 리테일(소액) 쪽 짝 — (매수금액−매도금액)/(매수금액+매도금액), 소액 체결만 (-1~+1).\n" +
    "[기준] +0.2 이상 강하게 매수유입 · +0.05~0.2 매수유입 · -0.05~-0.2 매도유입 · -0.2 이하 강하게 매도.\n" +
    "[의미] **2026-08-25 검증**: 1~15분 초단기 지평에서 통계적으로 유의한 방향 정보가 있습니다(5개 지평 전부 유의, 노이즈로 설명되는 수준을 훨씬 넘음). 다만 시장가로 그대로 매매하면 비용(10bp)이 총이익보다 커서 45개 지평×조합 전부 순손실이었습니다(4차례 재검증 포함, min_periods 계산결함까지 잡아낸 뒤 재확인) — 방향 정보 자체는 진짜지만 수수료를 넘길 만큼 크지는 않다는 뜻으로, 재량 판단 참고용입니다. 수급 흐름(고래)과는 상관 0.36 정도로 상당히 다른 정보라 같이 보면 유용합니다 — 둘이 같은 방향을 가리키면 좀 더 무게를 둘 근거, 엇갈리면 큰손/리테일이 다르게 움직이고 있다는 뜻입니다.",
  whale_intent: "[계산] 위 '수급 흐름'의 방향/세기 + 미결제약정(OI) 변화 방향을 결합하되, 'OI 증가 + 같은 방향 흐름'(신규 포지션 구축)은 가중치를 크게(1.0), 'OI 감소 + 같은 방향'(기존 포지션 청산)은 가중치를 작고 오히려 음수로(-0.35) 반영합니다 — 신규 진입이 청산보다 정보가치가 크다고 보는 설계입니다. pos_score = 0.7×흐름강도(부호포함) + 0.3×OI방향가중×OI강도, -1~+1. 신뢰도 = |pos_score|×100.\n" +
    "[기준] 0.25 이상 롱 쪽으로 기움 · -0.25 이하 숏 쪽으로 기움 · 신뢰도 70 이상인데 애매하면 관망 · 그 외 판단 약함.\n" +
    "[의미] 단순히 '누가 사고팔았나'가 아니라 '고래가 새 포지션을 쌓고 있는가, 기존 걸 정리하고 있는가'까지 추정합니다. 온체인 지갑을 직접 보는 게 아니라 체결+미결제약정 조합의 파생 추정치라서 위 태그에 '수급흐름 파생'이라고 표기했습니다.",
};

// Snapshot tab "12신호 한눈에" overview: id lookup so the compact chip row (.signal-chip-row in
// index.html) can be updated from the same per-tick data as the full snapModelIndicatorList below.
const MODEL_CHIP_IDS = {
  oi_delta: "modelChipOiDelta",
  liq_pressure: "modelChipBasisLiq",
  risk: "modelChipRisk",
  liq_cascade: "modelChipLiqCascade",
  liq_direction: "modelChipLiqDirection",
  whale: "modelChipWhale",
  retail_flow: "modelChipRetailFlow",
  whale_intent: "modelChipWhaleIntent",
};
// whale/whale_intent/liq_direction/retail_flow are directional (tone: good=상승 쪽/bad=하락
// 쪽/neutral=중립); oi_delta/liq_pressure/risk/liq_cascade are risk-level (tone: good=안정/warn=주의/
// bad=위험). Both families reuse the same good/warn/bad colors, so a bare red chip is ambiguous
// ("숏" vs "위험") -- 2026-08-24 사용자 리포트. Prefixing an explicit ▲/▼/– arrow on the
// directional chips only disambiguates by text, not just color.
const DIRECTIONAL_MODEL_CHIP_KEYS = new Set(["whale", "whale_intent", "liq_direction", "retail_flow"]);

function renderModelIndicatorList(items) {
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
    const nowTimeText = fmtTimeOnly(times[times.length - 1]);
    return `<article class="ops-health-row ${it.tone}">
      <span class="ops-health-dot" aria-hidden="true"></span>
      <div class="ops-health-info">
        <strong>${escapeHtml(it.label)}${derivedTag}</strong>
        ${meaningText ? `<p class="signal-meaning">${escapeHtml(meaningText)}</p>` : ""}
        ${it.liveText ? `<p class="signal-meaning">${escapeHtml(it.liveText)}</p>` : ""}
        <div class="evidence-strip-wrap">
          ${toneStripSvg(it.history, times)}
        </div>
        <div class="strip-time-row">
          <button type="button" class="detail-toggle" aria-expanded="${isOpen}" onclick="toggleSignalDetail(this, '${detailKey}')">${isOpen ? "접기 ▴" : "자세히 ▾"}</button>
          <span class="strip-time-now" data-fmt="time" data-default="${nowTimeText}">${nowTimeText}</span>
        </div>
        <div class="signal-detail${isOpen ? " open" : ""}">${escapeHtml(detailText)}</div>
      </div>
      <div class="ops-health-meta">
        <span class="ops-health-status-badge">${escapeHtml(it.subText || "-")}</span>
      </div>
    </article>`;
  }).join("");
  if (html === lastModelIndicatorHtml) return;
  lastModelIndicatorHtml = html;
  setH("snapModelIndicatorList", html);
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

  const liveCurrentPrice = Number(latestLivePriceByAsset.eth || map.current_price || 0);
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
const EVIDENCE_SIGNAL_KO = {
  orthogonal_combo: {
    name: "복합 오실레이터 신호",
    desc: "스토캐스틱 계열 오실레이터가 최근 3일 중 상/하위 10% 극단 + (바닥: 순매수 체결량 또는 펀딩비율 둘 중 하나라도 ±2표준편차 극단 / 천장: 순매수 체결량만) — 오실레이터와 확인 지표가 동시에 확인될 때만 발동",
    detail: "[바닥 조건] p_fast≤0.10 AND p_slow≤0.10 AND (delta_z≤-2.0 OR funding_z≤-2.0).\n" +
      "[천장 조건] p_fast≥0.90 AND p_slow≥0.90 AND delta_z≥2.0 — funding_z는 천장에는 안 씀(아래 2026-08-27 참고).\n" +
      "[p_fast/p_slow] 스토캐스틱 %K(14)와 그 3봉 평활선을, 최근 864봉(3일) 안에서 백분위 순위로 표현한 값 — '지금 이 값이 최근 3일 중 몇 번째로 낮은/높은가'.\n" +
      "[delta_z] 이번 봉의 순매수 체결량(시장가매수량×2−거래량)을 하루(288봉) 평균/표준편차로 정규화한 값.\n" +
      "[funding_z] ETHUSDT 무기한선물 펀딩비율을 최근 90개 관측(8시간 간격, 약 30일)의 평균/표준편차로 정규화한 값 — 8시간마다만 갱신, 봉 시각 기준 그 시점에 이미 공표된 값만 인과적으로 결합.\n" +
      "['직교(orthogonal)'의 의미] 가격형태 기반 정보(오실레이터)와 체결량/펀딩비율 기반 정보라는 서로 독립적인 두 축이 동시에 극단을 가리켜야 발동합니다 — 하나의 정보원만 보는 게 아니라 교차확인.\n" +
      "[2026-08-27] 원래 펀딩비율은 '펀딩+오실레이터 결합'이라는 별도 신호였으나, 거의 발동하지 않아(최대 55일 무발동) 이 신호의 바닥 조건에 OR로 합쳤습니다(검증: scripts/research_eth_funding_oscillator_union_combo_20260827.py) — 두 독립 구간 모두에서 lift는 유지되고 발동 빈도는 약 3배 늘었습니다. 천장에는 안 합쳤습니다 — funding_z 천장 조건은 원래도 드물게 발동했고, 검증 구간에서 그 드문 발동이 오히려 무작위보다 나빴습니다(lift 0.78배).",
  },
  liquidity_sweep: {
    name: "유동성 스윕(저점·고점 사냥)",
    desc: "직전 4시간 저점/고점을 살짝 뚫었다가 종가는 그 안으로 바로 되돌아와 마감 — 손절/청산 주문을 훑고 반전하는 캔들 패턴",
    detail: "[바닥 조건] 이번 봉 저가가 직전 48봉(4시간) 최저가보다 낮게 뚫고 내려갔다가, 종가는 그 직전 최저가 위로 다시 올라와 마감 (천장은 정반대).\n" +
      "[의미] 차트에서 흔히 'stop hunt'라 부르는 패턴입니다 — 직전 저점 아래 쌓여있는 손절/청산 주문들을 순간적으로 건드려 유동성을 흡수한 뒤 바로 되돌리는 움직임. 아래꼬리가 길게 뚫고 몸통은 위에서 마감하는 캔들 모양으로 나타납니다.",
  },
  volume_wick_climax: {
    name: "거래량 꼬리 클라이맥스",
    desc: "거래량이 하루평균 대비 2표준편차 이상 폭증 + 캔들 범위의 절반 이상이 반대방향 꼬리 — 패닉성 매도/매수가 즉시 흡수됨",
    detail: "[바닥 조건] vol_z≥2.0(이번 봉 거래량이 하루 평균 대비 2표준편차 이상 폭증) AND 아래꼬리비율≥0.5(캔들 전체 범위의 절반 이상이 아래꼬리) (천장은 정반대).\n" +
      "[아래꼬리비율] (시가·종가 중 작은값 − 저가) / (고가 − 저가).\n" +
      "[의미] 거래량이 갑자기 크게 튀면서, 그게 전부 한쪽으로의 밀어내기(예: 급락 후 강한 매수로 되받아침)로 끝난 캔들입니다 — 패닉성 투매/추격매수가 몰렸다가 즉시 흡수됐다는 신호.",
  },
  short_term_return_z: {
    name: "단기(15분) 수익률 급변",
    desc: "최근 3봉(15분) 수익률이 하루평균 대비 ±2.5표준편차 이상 — 통계적으로 이례적인 단기 쏠림",
    detail: "[바닥 조건] 최근 3봉(15분) 수익률을 하루 평균/표준편차로 정규화한 값이 -2.5 이하 (천장은 +2.5 이상).\n" +
      "[의미] 가장 단순하고 직접적인 형태의 신호입니다 — '15분 사이 얼마나 많이/빠르게 움직였나'를 통계적으로 극단적인 수준까지 좁혀서 판단하는, 단기 과매도/과매수의 원형에 가까운 지표입니다.",
  },
  taker_delta_z_climax: {
    name: "체결 매수매도 쏠림 극단",
    desc: "이번 봉 시장가 순매수(매수-매도) 체결량이 하루평균 대비 ±2표준편차 이상 — 오실레이터 조건 없이 체결량만으로 판단하는 독립 신호",
    detail: "[바닥 조건] delta_z≤-2.0 — 복합 오실레이터 신호에도 쓰이는 계산이지만, 오실레이터 조건 없이 이것 단독으로도 하나의 신호로 씁니다 (천장은 +2.0 이상).\n" +
      "[의미] '이번 봉에 시장가 매도가 평소보다 압도적으로 많이 쏟아졌다'는 사실 하나만으로 판단하는 독립 신호입니다 — 가격 형태(오실레이터)는 전혀 고려하지 않습니다.",
  },
  // 2026-08-24 추가(같은 날 후속) — ICT 2022 잔여요소 연구(오더블록/SMT/Po3)에서 유일하게 살아남은
  // SMT 다이버전스(3.12x/2.84x).
  smt_divergence: {
    name: "SMT 다이버전스(ETH·BTC 엇갈림)",
    desc: "ETH는 직전 4시간 저점(고점)을 갱신했는데 BTC는 갱신하지 않음 — 두 자산의 스윙 흐름이 어긋나는 상관자산 비확인 신호",
    detail: "[바닥 조건] ETH 저가가 직전 48봉(4시간) 최저가보다 낮게 갱신 AND BTC 저가는 같은 기간 자기 자신의 직전 최저가를 갱신하지 못함 (천장은 정반대).\n" +
      "[의미] ICT(스마트머니 콘셉트)의 SMT 다이버전스를 그대로 이식한 신호입니다 — 두 상관자산 중 하나만 신저점을 찍고 다른 하나는 버텨준다면, 그 신저점은 '진짜 매도세'가 아니라 개별 종목성 소진일 가능성을 시사합니다. 유동성 스윕과 형제 신호지만 되돌림(종가 복귀) 대신 상관자산 비확인을 가짜 돌파 판별 기준으로 씁니다 — 같은 날 검증에서 두 방식이 정밀도상 동급(42~43%)이고 서로 다른 봉의 약 40%에서 발동한다는 것까지 확인됐습니다.",
  },
  // 2026-08-24 추가(같은 날 후속) — 피보나치/하모닉 기하학 계열 연구에서 유일하게 sweep급 lift를
  // 보인 확장소진(3.27x/2.32x). 다른 6종보다 표본이 훨씬 얇고(n~190 vs 수백~수천) 경제성 게이트는
  // 0/16으로 실패해 "실험적" 등급으로만 편입 — 6종과 동일 신뢰도로 읽지 말 것.
  fib_extension_exhaustion: {
    name: "피보나치 확장 소진 (실험적)",
    desc: "가격이 직전 스윙 구간의 127.2~161.8% 지점까지 확장 — 다른 6종보다 표본이 얇아 실험적 등급으로 표시",
    detail: "[바닥 조건] 직전 48봉 안에서 고점보다 저점이 먼저 나온 하락 레그일 때, 가격이 그 레그 저점보다 27.2~61.8%만큼 더 아래(레그 폭 기준)까지 내려감 (천장은 정반대: 저점이 먼저 나온 상승 레그의 고점보다 더 위로 확장).\n" +
      "[의미] 피보나치 확장 비율(127.2%/161.8%)은 되돌림이 끝나고 추세가 소진되는 지점으로 흔히 쓰입니다. 실측 lift는 바닥 3.27배/천장 2.32배로 유동성 스윕과 비슷한 수준이었지만, 표본이 190건 안팎으로 다른 6종(수백~수천 건)보다 훨씬 적고, VAL에서 OOS로 갈 때 lift가 뚜렷이 줄어드는 경향도 확인됐습니다. 시장가 진입 경제성 게이트도 0/16으로 실패했습니다(다른 6종도 자동화는 전부 실패했으나, 이 신호는 표본 자체가 얇다는 점이 추가 약점).",
  },
  // 2026-08-25 추가 — AMT(마켓프로파일 이론) Dalton 룰2. "경제성 아니라 통계적 정보성이 대시보드
  // 노출 기준"이라는 사용자 원칙 재확인 이후 첫 추가 사례(feedback_dashboard_indicators_ic_bar_not_
  // pnl_bar 메모리). 다른 7종과 실패 사유가 다름 — 경제성 게이트(시장가 비용)가 아니라 고정 TP:SL
  // 번역 자체가 실패(비용 0이어도 짐). 탐지 자체는 실재하는 정보라 사용자가 재량 참고용으로 채택.
  dalton_rule2_balance_edge: {
    name: "Dalton 룰2 — 레인지 가장자리 반응",
    desc: "저변동성 국면(ATR% 백분위 30% 이하)에서 가격이 직전 4시간 레인지 가장자리(±15% 이내)에 닿음 — '박스권 안에서는 가장자리가 거부된다'는 마켓프로파일 이론",
    detail: "[바닥 조건] 최근 288봉(1일, 최소 144봉) ATR% 백분위가 30% 이하인 저변동성 국면에서, 이번 봉 저가가 직전 48봉(4시간) 레인지 저점으로부터 그 레인지 폭의 15% 이내에 위치 (천장은 정반대: 레인지 고점 근처).\n" +
      "[의미] 마켓프로파일 창시자 짐 달튼(Jim Dalton)의 3원칙 중 두 번째 — '박스권(밸런스) 안에서는 가격이 가장자리에 닿으면 거부되고 안쪽으로 되돌아온다'는 규칙입니다. 저변동성 조건 없이 테스트했을 때는 무의미했지만(0.96배/0.81배), 저변동성 게이트를 추가하자 실측 lift가 바닥 1.69→1.89배(VAL→OOS), 천장 1.66→1.42배(VAL→OOS)로 뒤집혔습니다(표본 VAL 1,360~2,500건/OOS 509~689건). 볼린저밴드 극값 신호와 겹침도 낮아 독립적인 정보입니다.\n" +
      "[유의] 다른 7종과 다른 이유로 자동매매에 못 들어갔습니다 — 시장가 진입 비용 문제가 아니라, 고정 TP:SL(1.6배ATR:1배ATR) 백테스트에서 TP 적중률(38.9%)이 손익분기 승률(38.5%)에 정확히 걸려 수수료가 0이어도 6개 구간 전부 졌습니다(0/6). 원인은 진짜 반전이 오기 전 가격이 평균 4~5봉·최대 0.44%(바닥)/0.32%(천장) 더 불리하게 움직인다는 것 — 지금의 고정 TP/SL 설계로는 이 신호를 못 담아냈을 뿐, 탐지 자체가 틀렸다는 뜻은 아닙니다.",
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

// Macro-event (CPI/NFP/GDP/PCE/내구재/FOMC) release-time alert (2026-08-26 follow-up) -- same
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
  badge.title = `${names} ${when} — 경제지표/FOMC 발표 전후 ±30분 참고용 안내(검증된 시장개장 알림과 달리 개별 검증은 안 됨)`;
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
  // Same latest_bar_utc anchor evidenceStripSvg's times[] is built from below -- shared across all
  // signals (one shared 5-min kline index), so computed once rather than per row.
  const evidenceNowTimeText = fmtShortTs(payload.latest_bar_utc);
  setH("evidenceSignalList", signals.map((s) => {
    const tone = evidenceSideTone(s);
    // "바닥"/"천장" (not "BOTTOM"/"TOP") to match both the compact chip row's own state text
    // (EVIDENCE_STRIP_CHIP_IDS block below: s.bottom_fired ? "바닥" : ...) and this badge's new
    // fixed width (2026-08-27 user request) -- keeps text length closer to the other badge values
    // (안정/주의/미발동/상승압력 etc.) instead of the one outlier using an English word.
    const state = evidenceSideLabel(s, { bottom: "바닥 발동", top: "천장 발동", both: "혼재 발동", none: "미발동" });
    const ko = EVIDENCE_SIGNAL_KO[s.name] || { name: s.name, desc: s.description };
    const detailKey = `evidence:${s.name}`;
    const isOpen = detailOpenKeys.has(detailKey);
    const detailText = ko.detail ? `${ko.detail}\n\n[주의] ${EVIDENCE_SIGNAL_DISCLAIMER}` : "";
    // 발동 중일 때 바로 보이는 의미(클릭 불필요) -- 2026-08-24 사용자 요청.
    const meaningText = evidenceSideLabel(s, {
      bottom: `지금 바닥(BOTTOM) 신호가 발동 중이에요 — ${ko.desc || s.description || ""}. 통계적으로 이 부근에서 반등 확률이 평소보다 높다는 신호지만, 실제 반전 전에도 0.5~0.85% 더 불리하게 움직인 사례가 많았어요.`,
      top: `지금 천장(TOP) 신호가 발동 중이에요 — ${ko.desc || s.description || ""}. 통계적으로 이 부근에서 하락 반전 확률이 평소보다 높다는 신호지만, 실제 반전 전에도 0.5~0.85% 더 불리하게 움직인 사례가 많았어요.`,
      both: `바닥(BOTTOM)과 천장(TOP) 신호가 최근 몇 봉 안에 둘 다 발동됐어요(혼재) — ${ko.desc || s.description || ""}. 방향이 엇갈린 상태라 단독 신호일 때보다 해석에 더 주의가 필요해요.`,
      none: "",
    });
    if (meaningText) firedMeanings.push({ tone, text: meaningText });
    const stripChip = EVIDENCE_STRIP_CHIP_IDS[s.name] ? el(EVIDENCE_STRIP_CHIP_IDS[s.name]) : null;
    if (stripChip) {
      stripChip.className = `signal-chip ${tone}`;
      const stripStateEl = stripChip.querySelector(".signal-chip-state");
      if (stripStateEl) stripStateEl.textContent = evidenceSideLabel(s, { bottom: "바닥", top: "천장", both: "혼재", none: "-" });
    }
    evidenceHistoryBySignal[s.name] = { bottom_history: s.bottom_history || [], top_history: s.top_history || [], latest_bar_utc: payload.latest_bar_utc };
    return `<article class="ops-health-row evidence-row ${tone}" data-signal="${s.name}">
      <span class="ops-health-dot" aria-hidden="true"></span>
      <div class="ops-health-info">
        <strong>${escapeHtml(ko.name)}</strong>
        <span>${escapeHtml(ko.desc || "-")}</span>
        ${meaningText ? `<p class="signal-meaning">${escapeHtml(meaningText)}</p>` : ""}
        <div class="evidence-strip-wrap">
          ${evidenceStripSvg(s.bottom_history || [], s.top_history || [], payload.latest_bar_utc, 5)}
          <small class="evidence-strip-caption">
            <span>바닥 ${fmtShortTs(s.bottom_last_fired_ts)} · 천장 ${fmtShortTs(s.top_last_fired_ts)}</span>
            <span class="strip-time-now" data-default="${evidenceNowTimeText}">${evidenceNowTimeText}</span>
          </small>
        </div>
        <button type="button" class="detail-toggle" aria-expanded="${isOpen}" onclick="toggleSignalDetail(this, '${detailKey}')">${isOpen ? "접기 ▴" : "자세히 ▾"}</button>
        <div class="signal-detail${isOpen ? " open" : ""}">${escapeHtml(detailText)}</div>
      </div>
      <div class="ops-health-meta">
        <span class="ops-health-status-badge">${escapeHtml(state)}</span>
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
      svgEl.outerHTML = evidenceStripSvg(hist.bottom_history, hist.top_history, hist.latest_bar_utc, 5, liveTone, payload.bar_open_utc);
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

async function refreshOiSignal() {
  const now = Date.now();
  if (now - oiSignalLastFetchAt < OI_SIGNAL_POLL_MS) return;
  oiSignalLastFetchAt = now;
  try {
    const res = await fetch(API_OI_SIGNAL_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`oi signal ${res.status}`);
    latestOiSignal = await res.json();
  } catch (error) {
    console.error("OI signal fetch error:", error);
    latestOiSignal = { warmed_up: false, error: "fetch_failed" };
  }
}

async function refreshLiquidation5mSignal() {
  const now = Date.now();
  if (now - liquidation5mLastFetchAt < LIQUIDATION_5M_POLL_MS) return;
  liquidation5mLastFetchAt = now;
  try {
    const res = await fetch(API_LIQUIDATION_5M_URL, { cache: "no-store" });
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
    const res = await fetch(API_BASIS_LIQUIDATION_URL, { cache: "no-store" });
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
    const res = await fetch(API_LIQUIDATION_DIRECTION_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`liquidation direction signal ${res.status}`);
    latestLiquidationDirection = await res.json();
  } catch (error) {
    console.error("Liquidation direction signal fetch error:", error);
    latestLiquidationDirection = { warmed_up: false, error: "fetch_failed" };
  }
}

// Unlike latestOiSignal (picked up by the next state-driven render() pass), the liquidation map
// has no such host -- it self-triggers both the panel list and the snapshot chart right after a
// fetch resolves, same pattern as refreshEvidenceSignals().
async function refreshLiquidationMap() {
  const now = Date.now();
  if (now - liquidationMapLastFetchAt < LIQUIDATION_MAP_POLL_MS) return;
  liquidationMapLastFetchAt = now;
  try {
    const res = await fetch(API_LIQUIDATION_MAP_URL, { cache: "no-store" });
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
  if (sub) sub.textContent = events.length ? `오늘·내일 ${events.length}건 (경제지표·FOMC·EIA·국채입찰·실적 — 정치일정 미포함)` : "오늘·내일 예정된 일정 없음";
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
      oiSignalLastFetchAt = 0; refreshOiSignal();
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
  const liveCurrentPrice = Number(latestLivePriceByAsset.eth || map.current_price || 0);
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


// Keeps candleHistoryByAsset.eth's rightmost candle live between the 5-min
// maybeFetchSnapshotChartHistory() fetches, mirroring updateChart()'s in-place extend/roll logic
// for the Live tab's own candleHistory -- 2026-08-25, user report: the Snapshot chart's last candle
// sat frozen at whatever /api/market-history last returned while the "현재" price line (redrawn
// every 5s by the call below) kept moving, which read as the whole chart being stuck/shifted by one
// bar. Same bucket math as updateChart(): extend the last candle's high/low/close in place while
// still inside its 5-min bucket, or push a fresh one once the live tick crosses into a new bucket.
function updateSnapshotCandleLive() {
  const candles = candleHistoryByAsset.eth;
  if (!Array.isArray(candles) || !candles.length) return;
  const price = Number(latestLivePriceByAsset.eth || 0);
  if (!(price > 0)) return;
  const tsMs = Date.parse(latestLivePriceTsByAsset.eth || "");
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
  const fullCandles = candleHistoryByAsset.eth || [];
  if (!svg || !fullCandles.length) return;
  // Sliced to SNAPSHOT_CHART_MAX_CANDLES (6h) -- narrower than the shared candleHistoryByAsset.eth
  // cache (still 8h, CHART_MAX_CANDLES) so the density-history overlay always has a real snapshot
  // behind every visible column (see that constant's comment).
  const candles = fullCandles.slice(-SNAPSHOT_CHART_MAX_CANDLES);
  const currentPrice = Number(latestLivePriceByAsset.eth || candles[candles.length - 1]?.close || 0);
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
    const mx = (evt.clientX - rect.left) * (w / rect.width);
    const my = (evt.clientY - rect.top) * (h / rect.height);

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
  // this logic to keep correct, not a live copy and a history copy). The 5 model indicators only
  // render on the Snapshot tab now (renderModelIndicatorList below) -- no Live-tab cards consume
  // ci/toneHistory anymore.
  const ci = classifyIndicators(micro, tail);
  const { whaleTone, whalePosTone, riskTone, cascadeTone } = ci;
  // toneHistory feeds the Snapshot tab's activity-strip rows (renderModelIndicatorList below).
  pushToneHistory("whale", whaleTone);
  pushToneHistory("retail_flow", ci.retail_flow.tone);
  pushToneHistory("whale_intent", whalePosTone);
  pushToneHistory("risk", riskTone);
  pushToneHistory("liq_cascade", cascadeTone);

  // OI 급변 (replaces 호가 불균형, 2026-08-24) -- fetched separately by refreshOiSignal(), not
  // part of ci/toneHistory (see scripts/live_oi_delta_signal_20260824.py for why). History comes
  // from the server payload directly (survives refresh), not client-accumulated toneHistory.
  const oiWarmedUp = !!(latestOiSignal && latestOiSignal.warmed_up);
  const oiTone = oiWarmedUp ? latestOiSignal.tone : "neutral";
  const oiSubText = oiWarmedUp
    ? (oiTone === "bad" ? "위험" : oiTone === "warn" ? "주의" : "안정")
    : "웜업 중";

  // 베이시스 청산압박 (replaces 독성/toxicity, 2026-08-27) -- fetched separately by
  // refreshBasisLiquiditySignal(), same external-fetch category as latestOiSignal above. RISK
  // GAUGE (good/warn/bad), NOT directional -- see scripts/live_spot_perp_basis_signal_20260827.py.
  const basisLiqWarmedUp = !!(latestBasisLiquidation && latestBasisLiquidation.warmed_up);
  const basisLiqTone = basisLiqWarmedUp ? latestBasisLiquidation.tone : "neutral";

  // 청산 방향압력 (2026-08-25) -- fetched separately by refreshLiquidationDirectionSignal(), same
  // external-fetch category as latestOiSignal above. Directional model-indicator, NOT evidence-
  // signal tier -- see scripts/live_liquidation_direction_signal_20260825.py docstring.
  const liqDirWarmedUp = !!(latestLiquidationDirection && latestLiquidationDirection.warmed_up);
  const liqDirTone = liqDirWarmedUp
    ? (latestLiquidationDirection.direction === "bullish" ? "good"
      : latestLiquidationDirection.direction === "bearish" ? "bad" : "neutral")
    : "neutral";

  // 2026-08-25: perf pass -- this whole block (gauge + chart + model-indicator list) only paints
  // anything the user can see while the Snapshot tab is active (snapshotTabPanel is display:none
  // otherwise), so it's gated the same way as tick()'s Snapshot-only fetches above. Data
  // accumulation (pushToneHistory calls above this block, oiTone/liqDirTone derivation) stays
  // unconditional -- only the paint work below is skipped, so history strips have no gap when the
  // user switches back to Snapshot.
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
      // both now redraw from the same latestLiquidationMap + latestLivePriceByAsset.eth snapshot.
      renderLiquidationMapPanel();
    }

    // Snapshot tab: renderModelIndicatorList mirrors renderEvidenceSignals's row/strip UI.
    renderModelIndicatorList([
      {
        key: "oi_delta", label: "OI 급변", tone: oiTone, subText: oiSubText,
        history: (latestOiSignal && latestOiSignal.tone_history) || [],
        times: evenlySpacedBarTimes(latestOiSignal && latestOiSignal.latest_ts_utc, (latestOiSignal && latestOiSignal.tone_history || []).length, 5),
        derivedTag: "= 대시보드 자체계산",
        derivedTitle: "봇 내부 상태가 아니라 대시보드 서버가 별도 duckdb(OI 전용 poller)에서 직접 계산 -- 아직 실제 매매 결정에는 연결되지 않음. 자세히 보기 참고.",
      },
      {
        key: "liq_pressure", label: "베이시스 청산압박", tone: basisLiqTone,
        subText: basisLiquiditySubText(latestBasisLiquidation),
        history: (latestBasisLiquidation && latestBasisLiquidation.tone_history) || [],
        times: evenlySpacedBarTimes(latestBasisLiquidation && latestBasisLiquidation.latest_ts_utc, (latestBasisLiquidation && latestBasisLiquidation.tone_history || []).length, 5),
        derivedTag: "= 대시보드 자체계산·탐색적",
        derivedTitle: "봇 내부 상태가 아니라 대시보드 서버가 spot/perp klines를 직접 fetch해 계산 -- 아직 실제 매매 결정에는 연결되지 않음. 청산크라우딩 상관은 ~1개월 탐색적 표본(3-split 재현 전). 자세히 보기 참고.",
      },
      { key: "risk", label: "꼬리 리스크", tone: ci.risk.tone, subText: ci.risk.subText, history: toneHistory.risk, times: toneHistoryTimes.risk },
      {
        key: "liq_cascade", label: "청산 캐스케이드", tone: ci.liq_cascade.tone,
        subText: ci.liq_cascade.subText, history: toneHistory.liq_cascade, times: toneHistoryTimes.liq_cascade,
        liveText: liqCascadeLiveDetail(tail),
        derivedTag: "= 꼬리 리스크와 다른 축",
        derivedTitle: "꼬리 리스크의 aftershock_prob는 향후 추가 충격 확률을 여러 요소로 blend한 종합값 -- 이건 그중 하나인 '지금 이 순간 캐스케이드가 실제로 진행 중인가'만 그대로 보여주는 원시 상태값(tail_risk_interceptor.py의 호크스 감쇠 타이머).",
      },
      {
        key: "liq_direction", label: "청산 방향압력", tone: liqDirTone,
        subText: liqDirWarmedUp ? liqDirectionSubText(latestLiquidationDirection) : "웜업 중",
        history: (latestLiquidationDirection && latestLiquidationDirection.tone_history) || [],
        times: evenlySpacedBarTimes(latestLiquidationDirection && latestLiquidationDirection.latest_ts_utc, (latestLiquidationDirection && latestLiquidationDirection.tone_history || []).length, 1),
      },
      { key: "whale", label: "수급 흐름", tone: ci.whale.tone, subText: ci.whale.subText, history: toneHistory.whale, times: toneHistoryTimes.whale },
      { key: "retail_flow", label: "리테일 수급", tone: ci.retail_flow.tone, subText: ci.retail_flow.subText, history: toneHistory.retail_flow, times: toneHistoryTimes.retail_flow },
      {
        key: "whale_intent",
        label: "고래 포지션", tone: ci.whale_intent.tone,
        subText: ci.whale_intent.subText, history: toneHistory.whale_intent, times: toneHistoryTimes.whale_intent,
        derivedTag: "= 수급흐름 파생",
        derivedTitle: "whale_position_score = 0.7 x sign(수급흐름) x |수급흐름| 정규화 + 0.3 x OI방향 보정항 (microstructure_scanner.py) -- 부호와 크기 대부분이 수급 흐름(nif_whale)에서 그대로 옴, 별도 모듈이 두 값만으로 이 값을 재구성 가능",
      },
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
      refreshOiSignal();
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
      pushToneHistory("risk", c.risk.tone);
      pushToneHistory("liq_cascade", c.liq_cascade.tone);
      pushToneHistory("whale", c.whale.tone);
      pushToneHistory("retail_flow", c.retail_flow.tone);
      pushToneHistory("whale_intent", c.whale_intent.tone);
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
