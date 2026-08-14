const TRADE_JOURNAL_URL = "../../data/live/trade_journal.jsonl";
const API_EVENTS_URL = "/api/events";
const API_TRADES_URL = "/api/trades";
const API_OPS_STATUS_URL = "/api/ops-status";
const API_BTC_MULTISLOT_SHADOW_URL = "/api/btc-multislot-shadow";
const API_ETH_JMLAM4_SHADOW_URL = "/api/eth-jmlam4-shadow";
const API_ETH_EXITHEAD_SHADOW_URL = "/api/eth-exithead-shadow";
const POLL_MS = 2500;
const CHART_RENDER_MIN_INTERVAL_MS = 5000;
const JOURNAL_POLL_MS = 10000;
const CANDLE_HISTORY_POLL_MS = 300000;
const MICRO_HISTORY_MAX = 40;
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

const microHistory = {
  obi: [],
  whale: [],
  whale_intent: [],
  eai: [],
  tox: [],
  risk: [],
};

function pushMicroHistory(key, value) {
  const v = Number(value);
  if (!Number.isFinite(v)) return;
  const arr = microHistory[key];
  if (!arr) return;
  arr.push(v);
  if (arr.length > MICRO_HISTORY_MAX) arr.shift();
}

function renderSparkline(cardId, values, tone) {
  const svg = document.querySelector(`#${cardId} .ind-spark`);
  if (!svg) return;
  if (!values || values.length < 2) { svg.innerHTML = ""; return; }
  const w = 100, h = 24;
  const min = Math.min(...values), max = Math.max(...values);
  const span = Math.max(max - min, 1e-9);
  const step = w / (values.length - 1);
  const linePts = values.map((v, i) => `${(i * step).toFixed(1)},${(h - ((v - min) / span) * h).toFixed(1)}`);
  const areaPts = [`0,${h}`, ...linePts, `${w},${h}`].join(" ");
  const color = tone === "good" ? "var(--good)" : tone === "bad" ? "var(--bad)" : tone === "warn" ? "var(--amber)" : "var(--muted)";
  svg.innerHTML = `
    <polygon points="${areaPts}" fill="${color}" opacity="0.14"/>
    <polyline points="${linePts.join(" ")}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
  `;
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
let ethJmlam4Etag = "";
let ethJmlam4LastFetchAt = 0;
let latestEthJmlam4Payload = null;
let ethExitheadEtag = "";
let ethExitheadLastFetchAt = 0;
let latestEthExitheadPayload = null;
let lastChartRenderAt = 0;
let isScrolling = false;
let scrollIdleTimer = 0;
let dashboardEvents = null;
const OPS_POLL_MS = 30000;

// --- Chart Global Variables ---
let candleHistory = []; // Array of {time, open, high, low, close}
const CHART_CANDLE_MIN = 5; 
const CHART_MAX_CANDLES = 100;
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
  const regime = String(sig.governor_regime || gov.regime || active.regime || "-").toUpperCase();
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
  setT("chartRegimeText", regime);
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

function whalePositionRead(micro) {
  const score = Number(micro.whale_position_score || 0);
  const confidence = Number(micro.whale_position_confidence || 0);
  if (score >= 0.25) return `고래 포지션이 롱 쪽으로 기움`;
  if (score <= -0.25) return `고래 포지션이 숏 쪽으로 기움`;
  if (confidence >= 70) return "고래 포지션은 관망에 가까움";
  return "고래 포지션 판단 약함";
}

function obiRead(v) {
  const x = Number(v || 0);
  if (x >= 0.3) return "매수 호가가 강하게 받침";
  if (x >= 0.1) return "매수 호가가 우세함";
  if (x <= -0.3) return "매도 호가가 강하게 누름";
  if (x <= -0.1) return "매도 호가가 우세함";
  return "호가는 균형 상태";
}

function eaiRead(micro) {
  const eai = Number(micro.eai || 0);
  const bias = Number(micro.eai_bias || micro.signal_bias || micro.obi || 0);
  if (eai >= 2.5) return bias < 0 ? "하방 변동성 폭발 주의" : bias > 0 ? "상방 변동성 폭발 주의" : "변동성 폭발 주의";
  if (eai >= 1.5) return "변동성 확대 중";
  if (eai <= 0.5) return "에너지가 낮아 추격 약함";
  return "변동성은 보통 수준";
}

function eaiHint(micro) {
  const eai = Number(micro.eai || 0);
  const bias = Number(micro.eai_bias || micro.signal_bias || micro.obi || 0);
  if (eai < 0.5) return "안정";
  if (bias >= 0.1) return "롱 진입";
  if (bias <= -0.1) return "숏 진입";
  return eai >= 1.5 ? "추격 조심" : "중립";
}

function toxRead(v) {
  const x = clamp01(v);
  if (x >= 0.7) return "체결 독성이 높아 진입 위험";
  if (x >= 0.4) return "체결 독성 주의 구간";
  return "체결 환경은 안정적";
}

function toxHint(v, dir = 0) {
  const x = clamp01(v);
  if (x < 0.4) return "안정";
  if (dir > 0) return "롱 진입";
  if (dir < 0) return "숏 진입";
  return "진입 조심";
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
  const dir = Number(tail.z_bias || 0);
  if (x < 0.4) return "안정";
  if (dir > 0) return "롱 진입";
  if (dir < 0) return "숏 진입";
  return "진입 조심";
}

function signalTone(signal) {
  if (String(signal || "").includes("조심")) return "warn";
  if (signal === "롱 진입") return "good";
  if (signal === "숏 진입") return "bad";
  return "neutral";
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
  if (ethPriceUpdated && latestEthJmlam4Payload) renderEthJmlam4Position(latestEthJmlam4Payload);
  if (ethPriceUpdated && latestEthExitheadPayload) renderEthExitheadPosition(latestEthExitheadPayload);
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

function renderEthJmlam4Position(payload) {
  const cardEl = el("ethJmlam4PositionCard");
  if (!cardEl) return;
  const livePrice = Number(latestLivePriceByAsset["eth"] || 0);
  const barSeconds = Number(payload?.bar_seconds || 300);
  cardEl.innerHTML = shadowPositionCardHtml(payload?.position || null, null, livePrice, barSeconds);
}

function renderEthJmlam4Shadow(payload) {
  latestEthJmlam4Payload = payload;
  const badge = el("ethJmlam4Badge");
  const stale = Boolean(payload?.stale);
  if (badge) {
    badge.className = `ops-badge ${stale ? "bad" : "good"}`;
    badge.textContent = stale ? "STALE" : "LIVE";
  }
  const age = Number(payload?.age_minutes);
  const ageText = Number.isFinite(age) ? `${age.toFixed(age < 10 ? 1 : 0)}분 전` : "-";
  setT("ethJmlam4Sub", `마지막 bar ${fmtTs(payload?.last_bar)} · ${ageText} 갱신`);

  const side = Number(payload?.position_side || 0);
  const posText = side > 0 ? "LONG" : side < 0 ? "SHORT" : "FLAT";
  const src = payload?.position_source_component;
  setT("ethJmlam4Position", src ? `${posText} (${src})` : posText);

  setT("ethJmlam4Trades", `${payload?.total_trades ?? 0}건`);

  const pnl = Number(payload?.cumulative_return_pct);
  const pnlEl = el("ethJmlam4Pnl");
  setT("ethJmlam4Pnl", Number.isFinite(pnl) ? fmtPct(pnl, 2) : "-");
  if (pnlEl) {
    pnlEl.classList.remove("good-text", "bad-text", "muted-text");
    pnlEl.classList.add(`${Number.isFinite(pnl) ? riskClass(pnl) : "muted"}-text`);
  }

  const mdd = Number(payload?.mdd_pct);
  setT("ethJmlam4Mdd", Number.isFinite(mdd) ? fmtPctNoPlus(mdd, 2) : "-");

  renderEthJmlam4Position(payload);
  renderShadowCharts(payload, "ethJmlam4PnlSvg", "ethJmlam4EquitySvg");
}

async function refreshEthJmlam4Shadow() {
  const now = Date.now();
  if (now - ethJmlam4LastFetchAt < OPS_POLL_MS) return;
  ethJmlam4LastFetchAt = now;
  try {
    const res = await fetch(API_ETH_JMLAM4_SHADOW_URL, { cache: "no-store", headers: ethJmlam4Etag ? { "If-None-Match": ethJmlam4Etag } : {} });
    if (res.status === 304) return;
    if (!res.ok) throw new Error(`eth jmlam4 shadow ${res.status}`);
    ethJmlam4Etag = res.headers.get("ETag") || ethJmlam4Etag;
    renderEthJmlam4Shadow(await res.json());
  } catch (error) {
    console.error("ETH JM lambda4 shadow fetch error:", error);
    const badge = el("ethJmlam4Badge");
    if (badge) { badge.className = "ops-badge bad"; badge.textContent = "UNREACHABLE"; }
  }
}

function renderEthExitheadPosition(payload) {
  const cardEl = el("ethExitheadPositionCard");
  if (!cardEl) return;
  const livePrice = Number(latestLivePriceByAsset["eth"] || 0);
  const barSeconds = Number(payload?.bar_seconds || 300);
  cardEl.innerHTML = shadowPositionCardHtml(payload?.position || null, null, livePrice, barSeconds);
}

function renderEthExitheadShadow(payload) {
  latestEthExitheadPayload = payload;
  const badge = el("ethExitheadBadge");
  const stale = Boolean(payload?.stale);
  if (badge) {
    badge.className = `ops-badge ${stale ? "bad" : "good"}`;
    badge.textContent = stale ? "STALE" : "LIVE";
  }
  const age = Number(payload?.age_minutes);
  const ageText = Number.isFinite(age) ? `${age.toFixed(age < 10 ? 1 : 0)}분 전` : "-";
  setT("ethExitheadSub", `마지막 bar ${fmtTs(payload?.last_bar)} · ${ageText} 갱신`);

  const side = Number(payload?.position_side || 0);
  const posText = side > 0 ? "LONG" : side < 0 ? "SHORT" : "FLAT";
  const src = payload?.position_source_component;
  setT("ethExitheadPosition", src ? `${posText} (${src})` : posText);

  setT("ethExitheadTrades", `${payload?.total_trades ?? 0}건`);

  const pnl = Number(payload?.cumulative_return_pct);
  const pnlEl = el("ethExitheadPnl");
  setT("ethExitheadPnl", Number.isFinite(pnl) ? fmtPct(pnl, 2) : "-");
  if (pnlEl) {
    pnlEl.classList.remove("good-text", "bad-text", "muted-text");
    pnlEl.classList.add(`${Number.isFinite(pnl) ? riskClass(pnl) : "muted"}-text`);
  }

  const mdd = Number(payload?.mdd_pct);
  setT("ethExitheadMdd", Number.isFinite(mdd) ? fmtPctNoPlus(mdd, 2) : "-");

  renderEthExitheadPosition(payload);
  renderShadowCharts(payload, "ethExitheadPnlSvg", "ethExitheadEquitySvg");
}

async function refreshEthExitheadShadow() {
  const now = Date.now();
  if (now - ethExitheadLastFetchAt < OPS_POLL_MS) return;
  ethExitheadLastFetchAt = now;
  try {
    const res = await fetch(API_ETH_EXITHEAD_SHADOW_URL, { cache: "no-store", headers: ethExitheadEtag ? { "If-None-Match": ethExitheadEtag } : {} });
    if (res.status === 304) return;
    if (!res.ok) throw new Error(`eth exithead shadow ${res.status}`);
    ethExitheadEtag = res.headers.get("ETag") || ethExitheadEtag;
    renderEthExitheadShadow(await res.json());
  } catch (error) {
    console.error("ETH exit-head shadow fetch error:", error);
    const badge = el("ethExitheadBadge");
    if (badge) { badge.className = "ops-badge bad"; badge.textContent = "UNREACHABLE"; }
  }
}

function setupPageTabs() {
  document.querySelectorAll(".page-tab").forEach((button) => button.addEventListener("click", () => {
    const ops = button.dataset.pageTab === "ops";
    el("liveTabPanel")?.classList.toggle("hidden", ops);
    el("opsTabPanel")?.classList.toggle("hidden", !ops);
    document.querySelectorAll(".page-tab").forEach((tab) => tab.classList.toggle("active", tab === button));
    if (ops) { opsLastFetchAt = 0; refreshOpsStatus(); } else { btcMultislotLastFetchAt = 0; refreshBtcMultislotShadow(); ethJmlam4LastFetchAt = 0; refreshEthJmlam4Shadow(); ethExitheadLastFetchAt = 0; refreshEthExitheadShadow(); }
  }));
}

function setupIndicatorToggle() {
  const btn = el("indicatorToggleBtn");
  const row = el("indicatorExtraRow");
  const label = el("indicatorToggleLabel");
  if (!btn || !row) return;
  const apply = (expanded) => {
    row.classList.toggle("collapsed", !expanded);
    btn.classList.toggle("expanded", expanded);
    btn.setAttribute("aria-expanded", String(expanded));
    if (label) label.textContent = expanded ? "보조 지표 접기" : "보조 지표 더 보기 (+3)";
  };
  let expanded = false;
  try { expanded = localStorage.getItem("dashIndicatorExpanded") === "1"; } catch (e) {}
  apply(expanded);
  btn.addEventListener("click", () => {
    expanded = !expanded;
    apply(expanded);
    try { localStorage.setItem("dashIndicatorExpanded", expanded ? "1" : "0"); } catch (e) {}
  });
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

function renderCandleSvg(svg, candles, journal, entryPrice, currentPrice, riskLevels = []) {
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

  // Candle visibility takes priority: entry/SL/TP lines never widen the price scale.
  // If a level falls outside the resulting range, it's drawn clamped to the edge with an off-chart arrow (see priceLabels below).
  const allPrices = candles.flatMap(c => [c.high, c.low]);
  if (includeCurrentPrice && currentPrice > 0) allPrices.push(currentPrice);

  const minP = Math.min(...allPrices), maxP = Math.max(...allPrices);
  const pad = (maxP - minP) * 0.15 || 1;
  const yMin = minP - pad, yMax = maxP + pad;
  const ySpan = Math.max(yMax - yMin, 1e-5); // Prevent division by zero

  const xAt = (i) => ml + (i * cw) / candles.length;
  const yAt = (v) => mt + ((yMax - v) * ch) / ySpan;
  const bw = (cw / candles.length) * 0.8;

  // Grid & Y-Axis Ticks
  axisTicks(yMin, yMax, 6).forEach(t => {
    const y = yAt(t);
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", ml); line.setAttribute("x2", w - mr);
    line.setAttribute("y1", y); line.setAttribute("y2", y);
    line.setAttribute("class", "chart-grid");
    svg.appendChild(line);

    if (!mobileChart) {
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
  if (!xTickIndexes.includes(lastIdx)) xTickIndexes.push(lastIdx);

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

  const priceLabels = [];
  if (includeCurrentPrice && currentPrice > 0) priceLabels.push({ val: currentPrice, color: "var(--accent)", label: "현재", dashed: true, width: 2 });
  if (entryPrice > 0) priceLabels.push({ val: entryPrice, color: "var(--amber)", label: "진입", dashed: false, width: 3 });
  (riskLevels || []).forEach((level) => {
    if (Number(level.val) > 0) priceLabels.push(level);
  });

  // Sort by Y position (Price descending = Y ascending)
  priceLabels.sort((a, b) => yAt(a.val) - yAt(b.val));

  // Adjust Y to avoid overlap
  const minGap = 18;
  for (let i = 1; i < priceLabels.length; i++) {
    const prevY = yAt(priceLabels[i - 1].val);
    const currY = yAt(priceLabels[i].val);
    if (Math.abs(currY - prevY) < minGap) {
      // Move current label down if it overlaps with previous
      priceLabels[i].adjustedY = prevY + minGap;
    }
  }

  priceLabels.forEach(p => {
    const rawY = yAt(p.val);
    const offTop = rawY < mt;
    const offBottom = rawY > h - mb;
    const outOfView = offTop || offBottom;
    const realY = outOfView ? (offTop ? mt + 2 : h - mb - 2) : rawY;
    const labelYRaw = p.adjustedY !== undefined ? p.adjustedY : realY;
    const labelY = Math.max(mt + 9, Math.min(h - mb - 9, labelYRaw));
    const lineDashed = p.dashed || outOfView;

    // Line stays at real price
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", ml); line.setAttribute("x2", w - mr);
    line.setAttribute("y1", realY); line.setAttribute("y2", realY);
    line.setAttribute("stroke", p.color);
    line.setAttribute("stroke-width", String(p.width || 2));
    if (lineDashed) line.setAttribute("stroke-dasharray", "4,4");
    if (outOfView) line.setAttribute("opacity", "0.72");
    svg.appendChild(line);

    // Left label (follows label position)
    const txt = document.createElementNS(NS, "text");
    txt.setAttribute("x", ml - 5); txt.setAttribute("y", labelY + 4);
    txt.setAttribute("text-anchor", "end"); txt.setAttribute("font-size", "10");
    txt.setAttribute("font-weight", "bold"); txt.setAttribute("fill", p.color);
    txt.textContent = `${p.label}${offTop ? "↑" : offBottom ? "↓" : ""}`;
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
    pTxt.textContent = `${offTop ? "↑ " : offBottom ? "↓ " : ""}${fmtNum(p.val, 1)}`;
    svg.appendChild(pTxt);
  });

  // Create Hover Layer on Top
  const hoverGroup = document.createElementNS(NS, "g");
  hoverGroup.setAttribute("class", "hover-layer");
  svg.appendChild(hoverGroup);

  const vLine = document.createElementNS(NS, "line");
  vLine.setAttribute("x1", 0); vLine.setAttribute("x2", 0);
  vLine.setAttribute("y1", mt); vLine.setAttribute("y2", h - mb);
  vLine.setAttribute("stroke", "var(--hover-line)");
  vLine.setAttribute("stroke-dasharray", "4,4");
  vLine.style.display = "none";
  vLine.style.pointerEvents = "none";
  hoverGroup.appendChild(vLine);

  // Candlestick Tooltip Support
  svg.onmousemove = (evt) => {
    const rect = svg.getBoundingClientRect();
    const mx = (evt.clientX - rect.left) * (w / rect.width);
    if (mx < ml || mx > w - mr) { hideTooltip(); return; }
    
    const idx = Math.min(candles.length - 1, Math.max(0, Math.floor(((mx - ml) / cw) * candles.length)));
    const c = candles[idx];
    if (!c) return;
    
    const tx = ml + (idx * cw) / candles.length + bw/2;
    
    vLine.setAttribute("x1", tx);
    vLine.setAttribute("x2", tx);
    vLine.style.display = "block";

    showTooltip(evt.pageX, evt.pageY, `
      <b>${fmtDateTick(c.time * 1000)}</b><br>
      시가: ${fmtNum(c.open, 2)}<br>
      고가: ${fmtNum(c.high, 2)}<br>
      저가: ${fmtNum(c.low, 2)}<br>
      종가: ${fmtNum(c.close, 2)}
    `);
  };
  svg.onmouseleave = () => {
    hideTooltip();
    if (typeof hoverDot !== 'undefined') hoverDot.style.display = "none";
    if (typeof vLine !== 'undefined') vLine.style.display = "none";
    if (typeof hoverDots !== 'undefined') hoverDots.forEach(d => d.style.display = "none");
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
  
  // 6 Indicators Grid Borders
  const toxV = clamp01(micro.toxicity_score);
  const riskV = clamp01(tail.aftershock_prob);
  const toxDir = Number(micro.obi || 0) > 0 ? 1 : Number(micro.obi || 0) < 0 ? -1 : 0;
  const toxSignal = toxHint(toxV, toxDir);
  const eaiSignal = eaiHint(micro);
  const riskSignal = tailRiskHint(tail);
  const whaleTone = Number(micro.nif_whale || 0) > 0.05 ? "good" : (Number(micro.nif_whale || 0) < -0.05 ? "bad" : "neutral");
  const whalePosTone = Number(micro.whale_position_score || 0) > 0.2 ? "good" : (Number(micro.whale_position_score || 0) < -0.2 ? "bad" : "neutral");
  const obiTone = Number(micro.obi || 0) > 0.1 ? "good" : (Number(micro.obi || 0) < -0.1 ? "bad" : "neutral");
  const eaiTone = signalTone(eaiSignal);
  const toxTone = signalTone(toxSignal);
  const riskTone = signalTone(riskSignal);
  setB("cardWhale", whaleTone);
  setB("cardWhalePos", whalePosTone);
  setB("cardObi", obiTone);
  setB("cardEai", eaiTone);
  setB("cardTox", toxTone);
  setB("cardRisk", riskTone);

  pushMicroHistory("whale", micro.nif_whale);
  pushMicroHistory("whale_intent", micro.whale_position_score);
  pushMicroHistory("obi", micro.obi);
  pushMicroHistory("eai", micro.eai);
  pushMicroHistory("tox", toxV);
  pushMicroHistory("risk", riskV);
  renderSparkline("cardWhale", microHistory.whale, whaleTone);
  renderSparkline("cardWhalePos", microHistory.whale_intent, whalePosTone);
  renderSparkline("cardObi", microHistory.obi, obiTone);
  renderSparkline("cardEai", microHistory.eai, eaiTone);
  renderSparkline("cardTox", microHistory.tox, toxTone);
  renderSparkline("cardRisk", microHistory.risk, riskTone);

  setT("whaleText", flowRead(micro));
  setT("whaleStatusText", directionalCaution(micro.nif_whale, 0.05));
  setT("whaleIntentText", whalePositionRead(micro));
  setT("whaleIntentPct", directionalCaution(micro.whale_position_score, 0.2));
  setT("obiText", obiRead(micro.obi));
  setT("eaiText", eaiRead(micro));
  setT("toxText", toxRead(toxV));
  setT("riskText", tailRiskRead(tail));

  // Gauges
  const obiVal = Number(micro.obi || 0);
  setT("obiGaugeRightTxt", directionalCaution(obiVal, 0.1));
  setT("eaiGaugeRightTxt", eaiSignal);

  setMeter("toxFill", toxV, toxV > 0.65 ? "bad" : toxV > 0.35 ? "warn" : "good");
  setT("toxText", toxRead(toxV));
  setT("toxStatusLabel", toxSignal);

  const aftVal = clamp01(tail.aftershock_prob);
  setMeter("riskFill", aftVal, aftVal > 0.65 ? "bad" : aftVal > 0.35 ? "warn" : "good");
  setT("riskText", tailRiskRead(tail));
  setT("riskStatusLabel", riskSignal);

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
    refreshEthJmlam4Shadow();
    refreshEthExitheadShadow();
  } catch (e) {
    console.error("Tick Error:", e);
  } finally {
    tickInFlight = false;
  }
}

connectDashboardEvents();
tick();
setInterval(tick, POLL_MS);
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
setupIndicatorToggle();
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
