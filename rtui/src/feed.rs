//! 자료가 들어오는 자리 하나: 캔들 이력 · 풋프린트(묻기/밀기) · 1주일 · 바이낸스 @trade(형성 봉 칸·현재가). 순수 상태 기계 -- 입출력은 부르는 쪽(웹: index.html 의 얇은 접착 · 시험: examples/feed_replay.rs).
//! 줄 번호 = dashboard/live/app.js(이 작업 시점). 시각은 전부 부르는 쪽이 넣는다(now_ms = Date.now()).
//! 옮기지 않은 것: SSE(/api/events) 시세 -- 지금 화면은 1초마다 서버 시세로 latestLivePrice 를 덮는다(2263). WS 가 살아 있으면 다음 체결이 곧 다시 덮어 쓰는 값이라 뺐다.
//!   formingVwap·lvol(형성 봉 VWAP -- 이 카드는 VWAP 선을 안 그린다) · 코인 바꾸기(이 페이지는 ?asset= 하나).
use crate::data::{Candle, Footprint, FpBar, LiqBar, LiqHist, MarketHistory, Week, iso_sec};
use crate::view::CHART_PAN_BARS;
use std::collections::BTreeMap;

/// 풋프린트가 들어오는 길. Poll = 지금 서버 /api/footprint?since= (FOOTPRINT_POLL_MS) + 바이낸스 @trade 직결
/// · Push = rtcore 허브(urls.hub · 구독 → 스냅숏 + 증분) 하나: 풋프린트 밀기(apply_push) + 체결(Trade 사건)이 같은 소켓으로 온다. 캔들 이력·1주일은 둘 다 묻기.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Source { Poll, Push }

/// 주소 한 곳. api = 같은 출처 앞붙이("" = 지금 페이지 출처) · trade_ws = {sym} 자리에 소문자 심볼 · hub = rtcore 허브("/" 로 시작하면 접착이 지금 페이지 출처의 ws(s):// 로 바꾼다)
pub struct Urls { pub api: String, pub trade_ws: String, pub hub: String }
impl Default for Urls {
    fn default() -> Self { Self { api: String::new(), trade_ws: "wss://fstream.binance.com/ws/{sym}@trade".into(), hub: "/rt".into() } }
}

/// 부르는 쪽이 할 일(웹 접착이 fetch·WebSocket 으로 바꾼다 · 응답은 on_* 로 되돌려 준다)
#[derive(Debug, PartialEq, serde::Serialize)]
pub enum Req { Get { kind: Kind, url: String }, WsOpen(String), WsClose, /** 열린(또는 열리는 중인) 소켓에 보낼 글 -- 허브 구독 */ WsSend(String) }
#[derive(Debug, PartialEq, Clone, Copy, serde::Serialize)]
pub enum Kind { Footprint, History, Week, LiqSig, Liq, LiqTail, Mc, Acct, Srl, LiqTab, LiqMap }

/// 145 FOOTPRINT_POLL_MS · 148 FOOTPRINT_STALE_BARS · 38 CANDLE_HISTORY_POLL_MS · 1680 CHART_MAX_CANDLES · 4400 FAST_PRICE_MIN_INTERVAL_MS · 2224 FOOTPRINT_RENDER_MIN_INTERVAL_MS · 6638 FP_WEEK_REFRESH_MS
const FP_POLL_MS: f64 = 400.0;
const FP_STALE_BARS: f64 = 2.0;
const HIST_POLL_MS: f64 = 300_000.0;
const CHART_MAX_CANDLES: usize = 200;
const FAST_PRICE_MS: f64 = 80.0;
const RENDER_GATE_MS: f64 = 400.0;
const WEEK_REFRESH_MS: f64 = 6.0 * 3600e3;
const LIQ_POLL_MS: f64 = 60_000.0; // 1653 LIQUIDATION_5M_POLL_MS
const MARKET_CTX_POLL_MS: f64 = 1000.0; // 6116
const LIQ_TAIL_MS: f64 = 2000.0;   // 3204 LIQ_5M_TAIL_POLL_MS
const ACCT_POLL_MS: f64 = 10_000.0; // 1649 BINANCE_ACCOUNT_POLL_MS
const SR_LEAN_POLL_MS: f64 = 2000.0; // 5810
const LIQ_TAB_MS: f64 = 600_000.0; // liq_tab.js 9 DATA_POLL_MS
const LIQ_MAP1_MS: f64 = 30_000.0; // 1672 LIQUIDATION_MAP_POLL_MS
const LIQ_TIERS_MS: f64 = 60_000.0; // liq_profile.js 13 fetchExt
const BAR: i64 = 300; // CHART_CANDLE_MIN 5 × 60

/// 4471 footprintLive -- 브라우저가 쌓는 형성 봉 칸. cells: 키 = roundHalfEven(가격/칸 폭) → [매수, 매도, 고래매수, 고래매도, 리테일매수, 리테일매도]
pub struct Live {
    pub bar_start: i64, pub since: f64, pub bucket: f64, pub cells: BTreeMap<i64, [f64; 6]>,
    order_qty: f64, order_at: Option<(f64, bool, f64)>, order_last_ms: f64, order_last_tid: i64,
}
impl Default for Live {
    fn default() -> Self { Self { bar_start: 0, since: f64::INFINITY, bucket: 0.5, cells: BTreeMap::new(), order_qty: 0.0, order_at: None, order_last_ms: 0.0, order_last_tid: -1 } }
}

pub struct Feed {
    pub source: Source,
    pub urls: Urls,
    pub asset: String,
    pub symbol: String,
    /// obWant()·dirWant() -- 지표 스위치(오더블록·ORB·CCI·RSI2 · 기본 켜짐)가 1주일을 늘 부른다. ponytail: 그 지표들을 옮길 때 스위치에서 읽는다
    pub week_always: bool,
    pub candles: Vec<Candle>,           // candleHistoryByAsset[asset]
    pub fp_bars: BTreeMap<i64, FpBar>,  // footprintBars
    pub fp: Option<Footprint>,          // latestFootprint 의 머리(bars 는 fp_bars) · None = 아직 없음
    pub week: Option<(Vec<Candle>, BTreeMap<i64, FpBar>)>,
    /// 1607 latestLiquidation5mHist(창 144봉 · 꼬리 2봉이 2초마다 합쳐진다) · liq_gen = 전량이 바뀐 횟수(꼬리 응답이 그 사이 전량을 덮지 않게 -- 3222)
    pub liq: Vec<LiqBar>,
    liq_gen: u64, liq_tail_gen: u64, liq_at: f64, liq_tail_at: f64,
    /// 6651 fpWeek.liq
    pub week_liq: Vec<LiqBar>,
    pub live: Live,
    pub price: f64,                     // latestLivePriceByAsset
    pub price_ts_ms: Option<f64>,       // latestLivePriceTsByAsset
    /// 현재가 선·알약이 그리는 값: 체결마다 80ms 최소 간격(updateLivePriceFast) + 전체 그리기 때 최신(renderCandleSvg currentPrice)
    pub shown: f64,
    pub last_fast: f64, pub last_full: f64,
    fp_last_fetch: f64, fp_in_flight: bool,
    hist_last_fetch: f64, hist_retry_at: f64,
    week_at: f64, week_tried_at: f64, week_in_flight: bool,
    ws: bool, ws_retry_at: f64,
    /// 허브: 지금 연결에서 마지막으로 받은 seq(None = 스냅숏 전) -- 빈 번호면 끊고 다시 붙어 새 스냅숏을 받는다(proto.rs Stamped.seq)
    hub_seq: Option<u64>,
    /// 허브 Footprint.holes(복원이 못 메운 봉 시작 시각 · 메시지마다 전체 목록). 들고만 있다 -- 그리지 않는다(지금 화면에 없는 표시)
    pub holes: Vec<i64>,
    /// 그릴 것이 바뀌었다(부르는 쪽이 다시 그리기를 요청하고 끈다)
    pub dirty: bool,
    pub out: Vec<Req>,
    /// 1초 수급(허브 밀기만)
    pub s1: crate::s1::State,
    /// 1초 수급 카드만 띄운다 -- 허브만 붙고 풋프린트·캔들·청산 응답은 묻지 않는다
    pub s1_only: bool,
    /// 시장 맥락 응답(mc.rs · /api/market-context) · 마지막으로 물은 시각 · 이 카드만 띄운다(허브·풋프린트 없이 이 응답만 1초마다)
    pub mc: Option<serde_json::Value>,
    mc_at: f64,
    pub mc_only: bool,
    /// 계좌 칩(acct.rs): /api/binance-account 응답(ok 인 것만 -- 2972) · 마지막으로 물은 시각 · 허브 «모든 종목» 가격(코인 → 가격 · 4916) · 그 번호 · 이 카드만
    pub acct: Option<serde_json::Value>,
    acct_at: f64,
    pub prices: std::collections::HashMap<String, f64>,
    prices_seq: Option<u64>,
    pub acct_only: bool,
    /// «레벨 5분»(srl.rs): /api/sr-lean 응답(5820 latestSrLean) · 마지막으로 물은 시각 · 서버 시계 − 이 시계(초 · 5822 srLeanSkew) · 이 카드만
    pub srl: Option<serde_json::Value>,
    srl_at: f64,
    srl_skew: f64,
    pub srl_only: bool,
    /// «청산» 탭 청산맵(liq.rs): 지도 기간(1·7·30·60·90·180·365) · /api/liq-tab(ok 만) · 지도 응답 · 이 지도에서 가격이 지나간 [저, 고] · 이 카드만
    pub liq_days: u32,
    pub liq_tab: Option<serde_json::Value>,
    pub liq_map: Option<serde_json::Value>,
    pub liq_sw: Option<[f64; 2]>,
    liq_tab_at: f64,
    liq_map_at: f64,
    pub liq_only: bool,
    /// «남은 매물» 카드만(rem.rs · /api/liq-tab 만 묻는다)
    pub rem_only: bool,
}

/// 4477 roundHalfEven(파이썬 round 와 같은 격자)
pub fn round_half_even(x: f64) -> i64 {
    let f = x.floor();
    let d = x - f;
    if d > 0.5 { f as i64 + 1 } else if d < 0.5 { f as i64 } else if (f as i64) % 2 == 0 { f as i64 } else { f as i64 + 1 }
}

/// 2135 mergeFormingCandle(prev, next, nowS)
pub fn merge_forming_candle(prev: &[Candle], mut next: Vec<Candle>, now_s: f64) -> Vec<Candle> {
    let Some(b) = next.last().copied() else { return next };
    if let Some(a) = prev.iter().find(|c| c.time == b.time) {
        if b.partial() || now_s < (b.time + BAR) as f64 {
            let l = next.last_mut().unwrap();
            l.high = b.high.max(a.high); l.low = b.low.min(a.low);
            if b.partial() { l.close = a.close; }
        }
    }
    next.extend(prev.iter().filter(|c| c.time > b.time).copied());
    next
}

/// 2190 historyRetryDue -- 마감 봉에 서버 sma 가 아직 없으면 마감 뒤 60초~2봉 사이 15초마다 다시 묻는다
pub fn history_retry_due(closed_time: i64, closed_sma: Option<f64>, now_ms: f64, last_retry_ms: f64) -> bool {
    let since_close = now_ms / 1000.0 - (closed_time + BAR) as f64;
    closed_time > 0 && !closed_sma.is_some_and(f64::is_finite) && since_close >= 60.0 && since_close < 2.0 * BAR as f64 && now_ms - last_retry_ms >= 15000.0
}

impl Feed {
    pub fn new(asset: &str, symbol: &str, source: Source, urls: Urls, week_always: bool) -> Self {
        Self { source, urls, asset: asset.into(), symbol: symbol.into(), week_always, candles: vec![], fp_bars: BTreeMap::new(), fp: None, week: None,
               live: Live::default(), price: 0.0, price_ts_ms: None, shown: 0.0, last_fast: 0.0, last_full: 0.0, fp_last_fetch: 0.0, fp_in_flight: false,
               hist_last_fetch: 0.0, hist_retry_at: 0.0, week_at: 0.0, week_tried_at: 0.0, week_in_flight: false, ws: false, ws_retry_at: 0.0, hub_seq: None, holes: vec![], liq: vec![], liq_gen: 0, liq_tail_gen: 0, liq_at: 0.0, liq_tail_at: 0.0, week_liq: vec![], dirty: false, out: vec![], s1: Default::default(), s1_only: false, mc: None, mc_at: 0.0, mc_only: false, acct: None, acct_at: 0.0, prices: Default::default(), prices_seq: None, acct_only: false, srl: None, srl_at: 0.0, srl_skew: 0.0, srl_only: false,
               liq_days: 7, liq_tab: None, liq_map: None, liq_sw: None, liq_tab_at: f64::NEG_INFINITY, liq_map_at: f64::NEG_INFINITY, liq_only: false, rem_only: false }
    }
    fn get(&mut self, kind: Kind, path: String) { let url = format!("{}{}", self.urls.api, path); self.out.push(Req::Get { kind, url }); }

    /// 11553 tick()(500ms · 숨으면 아무것도 안 한다)의 이 카드 몫 + 4556 ensurePriceWs
    pub fn tick(&mut self, now: f64, hidden: bool, want_week: bool) {
        if self.mc_only { // 6259 refreshMarketCtx(1초 · 숨으면 안 묻는다)
            if !hidden && now - self.mc_at >= MARKET_CTX_POLL_MS { self.mc_at = now; let path = format!("/api/market-context?asset={}", self.asset); self.get(Kind::Mc, path); }
            return;
        }
        if self.srl_only { // 5812 refreshSrLean(2초 · 숨으면 · ETH 아니면 안 묻는다)
            if !hidden && self.asset == "eth" && now - self.srl_at >= SR_LEAN_POLL_MS { self.srl_at = now; self.get(Kind::Srl, "/api/sr-lean".into()); }
            return;
        }
        self.ensure_ws(now, hidden);
        if self.acct_only { // 2991 refreshBinanceAccount(tick 에서 · 10초 게이트) -- 가격은 허브가 민다
            if !hidden && now - self.acct_at >= ACCT_POLL_MS { self.acct_at = now; self.get(Kind::Acct, "/api/binance-account".into()); }
            return;
        }
        if self.liq_only || self.rem_only { // liq_tab.js tick(5초 · 숨으면 안 한다) fetchData 10분 · 지도 = 1일 app.js refreshLiquidationMap 30초 · 7일+ fetchExt 60초 · 지나간 칸용 5분봉
            if hidden { return; }
            if now - self.liq_tab_at >= LIQ_TAB_MS { self.liq_tab_at = now; let path = format!("/api/liq-tab?asset={}", self.asset); self.get(Kind::LiqTab, path); }
            if self.rem_only { return; }
            let gap = if self.liq_days == 1 { LIQ_MAP1_MS } else { LIQ_TIERS_MS };
            if now - self.liq_map_at >= gap {
                self.liq_map_at = now;
                let path = if self.liq_days == 1 { format!("/api/liquidation-map?asset={}", self.asset) } else { format!("/api/liquidation-map/tiers?asset={}&days={}", self.asset, self.liq_days) };
                self.get(Kind::LiqMap, path);
            }
            self.maybe_history(now);
            return;
        }
        if hidden || self.s1_only { return; }
        if self.source == Source::Poll { self.refresh_footprint(now); }
        self.maybe_week(now, want_week);
        self.maybe_history(now);
        self.maybe_liq(now);
    }
    /// 3240 refreshLiquidation5mSignal(60초 · 신호 응답이 성공해야 이력을 묻는다) + 3206 refreshLiquidation5mTail(2초 · 전량이 먼저 있어야)
    pub fn maybe_liq(&mut self, now: f64) {
        if now - self.liq_at >= LIQ_POLL_MS {
            self.liq_at = now;
            let path = format!("/api/liquidation-5m-signal?asset={}", self.asset);
            self.get(Kind::LiqSig, path);
        }
        if !self.liq.is_empty() && now - self.liq_tail_at >= LIQ_TAIL_MS {
            self.liq_tail_at = now; self.liq_tail_gen = self.liq_gen;
            let path = format!("/api/liquidation-5m-history?asset={}&bars=2", self.asset);
            self.get(Kind::LiqTail, path);
        }
    }
    /// 6267 실패(접착은 HTTP 오류도 null 로 준다) = {available: false, error: "fetch_failed"}(asset 없음 → 화면은 «불러오는 중…» -- 지금 화면도 같다)
    pub fn on_mc(&mut self, body: Option<&str>) {
        self.mc = Some(body.and_then(|b| serde_json::from_str(b).ok()).unwrap_or_else(|| serde_json::json!({"available": false, "error": "fetch_failed"})));
        self.dirty = true;
    }
    /// 2972 실패·ok 아님 = null(숨김)
    pub fn on_acct(&mut self, body: Option<&str>) {
        self.acct = body.and_then(|b| serde_json::from_str::<serde_json::Value>(b).ok()).filter(|v| v["ok"] == true);
        self.dirty = true;
    }
    /// 5819 실패·못 읽음 = 받아 둔 값 그대로 · ok 면 서버 시계 차를 다시 잰다
    pub fn on_srl(&mut self, body: Option<&str>, now: f64) {
        let Some(v) = body.and_then(|b| serde_json::from_str::<serde_json::Value>(b).ok()) else { return };
        if v["ok"] == true { self.srl_skew = v["now"].as_f64().unwrap_or(f64::NAN) - now / 1000.0; }
        self.srl = Some(v);
        self.dirty = true;
    }
    /// liq_tab.js fetchData: ok 면 받고 · ok 아님 = 없음 · 실패·못 읽음 = 옛 값 그대로(접착이 HTTP 오류도 None 으로 줘서 «오류 = null» 갈래는 옛 값 쪽으로 감)
    pub fn on_liq_tab(&mut self, body: Option<&str>) {
        let Some(v) = body.and_then(|b| serde_json::from_str::<serde_json::Value>(b).ok()) else { return };
        self.liq_tab = Some(v).filter(|v| v["ok"] == true);
        self.dirty = true;
    }
    /// 1일(app.js 3275-3283): 실패 = 지도 없음 · 7일+(liq_profile.js 17-19): tier_profile 없는 응답·실패 = 옛 지도. 새 지도 = 지나간 범위를 그 기준가부터 다시(sweep WeakMap)
    pub fn on_liq_map(&mut self, body: Option<&str>) {
        let v = body.and_then(|b| serde_json::from_str::<serde_json::Value>(b).ok());
        if self.liq_days == 1 { self.liq_map = v; } else {
            let Some(v) = v.filter(|v| v["tier_profile"].is_object()) else { return };
            self.liq_map = Some(v);
        }
        self.liq_sw = None;
        self.dirty = true;
    }
    /// 5827 srLeanNowS(초)
    pub fn srl_now(&self, now: f64) -> f64 { now / 1000.0 + self.srl_skew }
    /// 신호 응답(내용은 이 카드가 안 쓴다 -- 성공했을 때만 이력 전량을 묻는다)
    pub fn on_liq_sig(&mut self, body: Option<&str>) {
        if body.is_none() { return; }
        let path = format!("/api/liquidation-5m-history?asset={}&bars={}", self.asset, CHART_PAN_BARS);
        self.get(Kind::Liq, path);
    }
    /// 3261 전량: warmed_up 이 아니거나 실패면 빈 목록
    pub fn on_liq(&mut self, body: Option<&str>) {
        let h = body.and_then(|b| serde_json::from_str::<LiqHist>(b).ok()).unwrap_or_default();
        let next = if h.warmed_up { h.bars } else { vec![] };
        if next != self.liq { self.dirty = true; }
        self.liq = next; self.liq_gen += 1;
    }
    /// 3215-3236 꼬리 2봉: 같은 ts 는 바뀌었으면 갈아 끼우고, 더 새 봉은 붙이고 맨 앞을 버린다(창 길이 그대로)
    pub fn on_liq_tail(&mut self, body: Option<&str>) {
        if self.liq_tail_gen != self.liq_gen || self.liq.is_empty() { return; }
        let Some(h) = body.and_then(|b| serde_json::from_str::<LiqHist>(b).ok()) else { return };
        if !h.warmed_up { return; }
        let ms = |b: &LiqBar| iso_sec(&b.ts).map_or(f64::NAN, |s| s as f64);
        for nb in h.bars {
            if let Some(i) = self.liq.iter().position(|b| b.ts == nb.ts) {
                let o = &self.liq[i];
                let same = o.long_usd == nb.long_usd && o.short_usd == nb.short_usd && o.events == nb.events && o.partial == nb.partial && o.okx == nb.okx && o.hl == nb.hl && o.bybit == nb.bybit;
                if !same { self.liq[i] = nb; self.dirty = true; }
            } else if ms(&nb) > ms(self.liq.last().unwrap()) {
                self.liq.push(nb); self.liq.remove(0); self.dirty = true;
            }
        }
    }
    /// 6709 weekJoin("liq") -- 1주일 몫(라이브 첫 시각보다 앞)을 앞에
    pub fn liq_full(&self) -> Vec<LiqBar> {
        let t0 = self.liq.first().and_then(|b| iso_sec(&b.ts)).map_or(f64::INFINITY, |s| s as f64);
        if self.week_liq.is_empty() { return self.liq.clone(); }
        self.week_liq.iter().filter(|b| iso_sec(&b.ts).is_some_and(|s| (s as f64) < t0)).chain(self.liq.iter()).cloned().collect()
    }
    /// 1964 setChartWindow 가 폴링 간격을 기다리지 않고 바로 묻는다(footprintLastFetchAt = 0)
    pub fn kick_footprint(&mut self, now: f64, want_week: bool) {
        self.fp_last_fetch = 0.0;
        if self.source == Source::Poll { self.refresh_footprint(now); }
        self.maybe_week(now, want_week);
    }

    /// 6577 refreshFootprint 의 묻기 몫(since 규칙: 최신 봉 − 한 봉 · 두 봉 넘게 묵으면 전량)
    pub fn refresh_footprint(&mut self, now: f64) {
        if self.fp_in_flight || now - self.fp_last_fetch < FP_POLL_MS { return; }
        self.fp_last_fetch = now;
        let bar_sec = self.fp.as_ref().map(|f| f.bar_seconds).filter(|&b| b > 0.0).unwrap_or(300.0);
        let mut newest = self.fp_bars.keys().next_back().copied().unwrap_or(0);
        if newest != 0 && (now / 1000.0 - newest as f64) > bar_sec * FP_STALE_BARS { self.fp_bars.clear(); newest = 0; } // 6592 자가복구(숨은 탭은 tick 이 안 온다)
        let since = if newest != 0 { newest as f64 - bar_sec } else { 0.0 };
        self.fp_in_flight = true;
        let path = format!("/api/footprint?asset={}&bars={}&since={}", self.asset, CHART_PAN_BARS, since);
        self.get(Kind::Footprint, path);
    }
    /// 6600-6626 응답(실패면 캐시를 그대로 둔다)
    pub fn on_footprint(&mut self, body: Option<&str>) {
        self.fp_in_flight = false;
        let Some(p) = body.and_then(|b| serde_json::from_str::<Footprint>(b).ok()) else { return };
        let Footprint { bars, .. } = p.clone();
        // 다시 그리기는 값이 바뀌었을 때만(지금 화면은 응답마다 전체를 다시 그린다 6626 -- 같은 그림)
        let old = if p.full { Some(std::mem::take(&mut self.fp_bars)) } else { None };
        let mut changed = false;
        for b in bars { if self.fp_bars.get(&b.time) != Some(&b) { changed = true; self.fp_bars.insert(b.time, b); } }
        if self.fp_bars.len() > CHART_PAN_BARS { changed = true; }
        self.trim_bars(CHART_PAN_BARS);
        if let Some(old) = old { changed = old != self.fp_bars; }
        let meta = Footprint { bars: vec![], ..p };
        self.dirty |= changed || self.fp.as_ref() != Some(&meta);
        self.fp = Some(meta);
        self.shown = self.price; // 응답마다 전체 그리기(scheduleSnapshotChartRender) -- 선·알약이 최신가로
    }
    /// 6609 창 밖으로 밀려난 봉 = 1주일로 옮긴다(받아 뒀으면)
    fn trim_bars(&mut self, keep: usize) {
        while self.fp_bars.len() > keep {
            let (t, b) = self.fp_bars.pop_first().unwrap();
            if let Some(w) = self.week.as_mut() { w.1.insert(t, b); }
        }
    }

    /// rtcore 밀기(rtcore/src/proto.rs Footprint 의 받는 규칙): full = 통째로 · 아니면 봉마다 칸을 가격으로 덮어쓰고 dhl 은 갈아 끼움 → drop 봉 지움 → barsExpected 넘으면 오래된 쪽부터.
    pub fn apply_push(&mut self, p: Push) {
        self.holes = p.holes;
        if p.full { self.fp_bars.clear(); }
        for b in p.bars {
            let e = self.fp_bars.entry(b.time).or_insert_with(|| FpBar { time: b.time, levels: vec![], dhl: None });
            for l in b.levels {
                match e.levels.iter_mut().find(|x| x[0] == l[0]) { Some(x) => *x = l, None => e.levels.push(l) }
            }
            e.levels.sort_by(|a, b| a[0].total_cmp(&b[0]));
            e.dhl = b.dhl;
        }
        for t in p.drop { self.fp_bars.remove(&t); }
        self.trim_bars(p.bars_expected.max(1));
        self.fp = Some(Footprint { bucket: p.bucket, bars: vec![], bar_seconds: p.bar_seconds, whale_min_usd: p.whale_min_usd, retail_max_usd: p.retail_max_usd, full: p.full });
        self.shown = self.price;
        self.dirty = true;
    }

    /// 6640 maybeFetchFpWeek
    pub fn maybe_week(&mut self, now: f64, want: bool) {
        if self.week_in_flight || !(want || self.week_always) { return; }
        if (self.week.is_some() && now - self.week_at < WEEK_REFRESH_MS) || now - self.week_tried_at < 30000.0 { return; }
        self.week_in_flight = true; self.week_tried_at = now;
        let path = format!("/api/footprint-week?asset={}", self.asset);
        self.get(Kind::Week, path);
    }
    pub fn on_week(&mut self, body: Option<&str>, now: f64) {
        self.week_in_flight = false;
        let Some(w) = body.and_then(|b| serde_json::from_str::<Week>(b).ok()) else { return };
        if w.candles.is_empty() { self.week_tried_at = now + 270e3; return; } // 6652 빈 몸 = 300초 뒤 다시
        self.week = Some((w.candles, w.bars.into_iter().map(|b| (b.time, b)).collect()));
        self.week_liq = w.liq;
        self.week_at = now;
        self.dirty = true;
    }

    /// 2200 maybeFetchSnapshotChartHistory(5분 · 마감 봉 sma 증상 재시도)
    fn maybe_history(&mut self, now: f64) {
        let n = self.candles.len();
        let closed = if n >= 2 { Some(self.candles[n - 2]) } else { None };
        let due = history_retry_due(closed.map_or(0, |c| c.time), closed.and_then(|c| c.sma), now, self.hist_retry_at);
        if !due && n > 0 && now - self.hist_last_fetch < HIST_POLL_MS { return; }
        if due { self.hist_retry_at = now; }
        self.hist_last_fetch = now;
        let path = format!("/api/market-history?asset={}", self.asset);
        self.get(Kind::History, path);
    }
    /// 2155 fetchBinanceHistory 응답
    pub fn on_history(&mut self, body: Option<&str>, now: f64) {
        let Some(h) = body.and_then(|b| serde_json::from_str::<MarketHistory>(b).ok()) else { return };
        let next = merge_forming_candle(&self.candles, h.candles, now / 1000.0);
        let before = std::mem::replace(&mut self.candles, next);
        self.update_candle_live(now);
        self.dirty |= before != self.candles;
        self.shown = self.price;
    }
    /// 4362 updateSnapshotCandleLive -- 형성 봉 고저·종가를 최신 체결가로(경계 넘으면 새 봉 · 두 봉 넘게 벌어지면 이력을 바로 다시 묻는다)
    fn update_candle_live(&mut self, now: f64) {
        if self.candles.is_empty() || !(self.price > 0.0) { return; }
        let ts = (self.price_ts_ms.unwrap_or(now) / 1000.0).floor() as i64;
        let candle_ts = ts.div_euclid(BAR) * BAR;
        let last = self.candles.last_mut().unwrap();
        let p = self.price;
        if last.time < candle_ts {
            if candle_ts - last.time > BAR { self.hist_last_fetch = 0.0; }
            self.candles.push(Candle { time: candle_ts, open: p, high: p, low: p, close: p, partial: None, volume: 0.0, sma: None });
            if self.candles.len() > CHART_MAX_CANDLES { self.candles.remove(0); }
        } else {
            last.high = last.high.max(p); last.low = last.low.min(p); last.close = p;
        }
    }

    /// 4556 ensurePriceWs(숨으면 닫는다 · 끊겼으면 5초 간격으로 다시 · 붙을 때마다 since 를 비운다)
    pub fn ensure_ws(&mut self, now: f64, hidden: bool) {
        if hidden {
            if self.ws { self.ws = false; self.out.push(Req::WsClose); }
            return;
        }
        if self.ws || now < self.ws_retry_at { return; }
        self.ws_retry_at = now + 5000.0;
        self.live.since = f64::INFINITY;
        self.ws = true;
        let sym = self.symbol.to_lowercase();
        if self.source == Source::Push {
            (self.hub_seq, self.prices_seq) = (None, None);
            self.out.push(Req::WsOpen(self.urls.hub.clone()));
            let subs: Vec<&str> = if self.acct_only || self.liq_only || self.rem_only { vec![] } else { vec![&sym] }; // 계좌 칩은 «모든 종목» 가격만(구독과 상관없이 온다)
            self.out.push(Req::WsSend(serde_json::json!({ "subscribe": subs }).to_string()));
        } else {
            self.out.push(Req::WsOpen(self.urls.trade_ws.replace("{sym}", &sym)));
        }
    }
    /// 4636 ws.onclose
    pub fn on_ws_close(&mut self) { self.ws = false; }
    /// 4591 ws.onmessage -- 한 체결(Poll) · 허브 메시지 하나(Push)
    pub fn on_ws(&mut self, text: &str, now: f64) {
        if self.source == Source::Push { return self.on_hub(text, now); }
        let Ok(d) = serde_json::from_str::<serde_json::Value>(text) else { return };
        if d["e"] != "trade" { return; }
        let num = |k: &str| match &d[k] { serde_json::Value::String(s) => s.trim().parse::<f64>().unwrap_or(f64::NAN), v => v.as_f64().unwrap_or(f64::NAN) };
        let (price, t_ms, q) = (num("p"), num("T"), num("q"));
        let tid = d.get("t").filter(|v| !v.is_null()).map(|v| v.as_f64().unwrap_or(f64::NAN) as i64);
        self.trade(price, t_ms, q, d["m"].as_bool().unwrap_or(false), tid, now);
    }
    /// 허브(rtcore/src/proto.rs Stamped): snapshot = 풋프린트 전체 · footprint = 바뀐 칸 · trade = 체결. 그 밖 사건(book·mark·supply_1s·liq·flow·oi·모르는 것)은 seq 만 센다.
    /// 스냅숏 안 trades(코어 링 · 접속 전 체결)는 칸에 넣지 않는다 -- 직결 소켓이 접속 뒤 체결만 보는 것과 같게(그래야 «봉 중간에 붙었다» 규칙 since 가 같다).
    fn on_hub(&mut self, text: &str, now: f64) {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(text) else { return };
        let (seq, ty) = (v["seq"].as_u64(), v["type"].as_str().unwrap_or(""));
        // «모든 종목»(s = "*" · Prices)은 번호가 따로다(hub.rs: 붙을 때 지금 번호로 한 벌 · 그 뒤 +1씩) -- 심볼 번호와 섞으면 빈 번호로 보고 끊는다
        if v["s"] == "*" {
            match (self.prices_seq, seq) {
                (None, Some(s)) => self.prices_seq = Some(s),
                (Some(p), Some(s)) if s == p + 1 => self.prices_seq = Some(s),
                (Some(p), Some(s)) if s <= p => return,
                _ => { self.out.push(Req::WsClose); self.ws = false; self.ws_retry_at = 0.0; self.prices_seq = None; return; }
            }
            if ty == "prices" { // 4916 RT_SYM_ASSET(ASSET_CONFIG 심볼만)
                for r in v["rows"].as_array().into_iter().flatten() {
                    let (Some(s), Some(px)) = (r[0].as_str(), r[1].as_f64()) else { continue };
                    if let Some(a) = s.strip_suffix("usdt").filter(|a| crate::asset_cfg(a).is_some()).filter(|_| px > 0.0) { self.prices.insert(a.into(), px); self.dirty = true; }
                }
            }
            return;
        }
        if ty == "snapshot" {
            self.hub_seq = seq;
        } else {
            match (self.hub_seq, seq) {
                (Some(p), Some(s)) if s == p + 1 => self.hub_seq = Some(s),
                (Some(p), Some(s)) if s <= p => return, // 겹침(허브는 안 보낸다 -- 지켜만 둔다)
                _ => { // 스냅숏 전 메시지 · 빈 번호: 다시 붙는다(다음 tick 이 연다)
                    self.out.push(Req::WsClose); self.ws = false; self.ws_retry_at = 0.0; self.hub_seq = None;
                    return;
                }
            }
        }
        // 1초 수급(s1.rs) -- 스냅숏 supply 와 supply1s 본문이 같은 모양(app.js 4911 이 같은 applySupply1s 로 넣는다)
        match ty {
            "snapshot" => { self.s1.apply(&v["supply"]); self.dirty = true; }
            "supply1s" => { self.s1.apply(&v); self.dirty = true; }
            _ => {}
        }
        match ty {
            "snapshot" | "footprint" => {
                let fp = if ty == "snapshot" { v.get("footprint") } else { Some(&v) };
                if let Some(p) = fp.filter(|f| !f.is_null()).and_then(|f| serde_json::from_value::<Push>(f.clone()).ok()) { self.apply_push(p); }
            }
            "trade" => {
                let f = |k: &str| v[k].as_f64().unwrap_or(f64::NAN);
                self.trade(f("px"), f("ex_ms"), f("qty"), v["sell"].as_bool().unwrap_or(false), v["id"].as_u64().map(|x| x as i64), now);
            }
            _ => {}
        }
    }
    /// 체결 하나(직결 @trade · 허브 Trade 같은 길)
    fn trade(&mut self, price: f64, t_ms: f64, q: f64, sell: bool, tid: Option<i64>, now: f64) {
        if !(price > 0.0) { return; } // 바이낸스가 p:"0" 을 섞어 보낸다
        // 4342 noteLiveTrade
        self.price = price; self.price_ts_ms = Some(t_ms);
        self.update_candle_live(now);
        // 4419 updateLivePriceFast(80ms)
        if now - self.last_fast >= FAST_PRICE_MS { self.last_fast = now; self.shown = price; }
        if q > 0.0 {
            self.live_add(price, q, t_ms, sell, tid);
            // 2229 maybeRenderSnapshotChartNow(400ms): 전체 그리기 -- 선·알약도 최신가로
            if now - self.last_full >= RENDER_GATE_MS { self.last_full = now; self.update_candle_live(now); self.shown = self.price; }
        }
        self.dirty = true;
    }

    /// 4484 footprintLiveAdd
    pub fn live_add(&mut self, price: f64, qty: f64, ts_ms: f64, sell: bool, tid: Option<i64>) {
        if !(self.live.bucket > 0.0) { return; }
        let bar_sec = (ts_ms / 1000.0 / 300.0).floor() as i64 * BAR;
        if bar_sec != self.live.bar_start { self.close_order(); self.live.bar_start = bar_sec; self.live.cells.clear(); }
        let l = &mut self.live;
        if l.since == f64::INFINITY { l.since = ts_ms; }
        let key = round_half_even(price / l.bucket);
        l.cells.entry(key).or_insert([0.0; 6])[if sell { 1 } else { 0 }] += qty;
        let same = l.order_at.is_some_and(|(p, s, _)| p == price && s == sell)
            && ts_ms - l.order_last_ms <= 100.0
            && (ts_ms / 1000.0).floor() == (l.order_last_ms / 1000.0).floor()
            && (tid.is_none() || l.order_last_tid < 0 || tid == Some(l.order_last_tid + 1));
        if !same { self.close_order(); self.live.order_qty = 0.0; self.live.order_at = Some((price, sell, ts_ms)); }
        let l = &mut self.live;
        l.order_qty += qty; l.order_last_ms = ts_ms; l.order_last_tid = tid.unwrap_or(-1);
    }
    /// 4525 footprintLiveCloseOrder -- 테이커 주문 크기(고래 ≥ whaleMinUsd · 리테일 < retailMaxUsd · 중형은 칸 없음)
    fn close_order(&mut self) {
        let (at, q) = (self.live.order_at.take(), self.live.order_qty);
        self.live.order_qty = 0.0; self.live.order_last_tid = -1;
        let Some((price, sell, ts)) = at else { return };
        if !(q > 0.0) { return; }
        let (whale, retail) = self.fp.as_ref().map_or((0.0, 0.0), |f| (f.whale_min_usd, f.retail_max_usd));
        let notional = price * q;
        let slot = if whale > 0.0 && notional >= whale { 2 } else if retail > 0.0 && notional < retail { 4 } else { return };
        if (ts / 1000.0 / 300.0).floor() as i64 * BAR != self.live.bar_start { return; }
        let key = round_half_even(price / self.live.bucket);
        if let Some(c) = self.live.cells.get_mut(&key) { c[slot + sell as usize] += q; }
    }

    /// 6698 fpBarsNow: 라이브 144봉(+ 1주일이 필요한 보기면 앞에 1주일 몫) -- 시각순
    pub fn bars_now(&self, want_week: bool) -> Vec<&FpBar> {
        let mut v: Vec<&FpBar> = self.fp_bars.values().collect();
        if let (true, Some(w)) = (want_week, self.week.as_ref()) {
            v.extend(w.1.values().filter(|b| !self.fp_bars.contains_key(&b.time)));
            v.sort_by_key(|b| b.time);
        }
        v
    }
    /// 6723 footprintForChart + 4539 footprintMergeLive: 칸 있는 봉만 시각 → 칸, 형성 봉은 조건이 맞으면 브라우저 값. None = 풋프린트 없음(아직 안 옴)
    /// merged = 형성 봉을 바꿔 끼울 칸(부르는 쪽이 들고 있다가 by_time 에 꽂는다 -- 빌림 때문에 둘로 나눴다)
    pub fn for_chart(&mut self, want_week: bool) -> Option<(f64, Option<(i64, Vec<Vec<f64>>)>)> {
        let fp = self.fp.as_ref()?;
        if !self.bars_now(want_week).iter().any(|b| !b.levels.is_empty()) { return None; }
        let bucket = if fp.bucket != 0.0 && !fp.bucket.is_nan() { fp.bucket } else { 0.5 };
        self.live.bucket = bucket; // 서버가 정한 격자를 따른다
        let bar = self.live.bar_start;
        let has = |t: i64| self.bars_now(want_week).iter().any(|b| b.time == t && !b.levels.is_empty());
        let merged = (bar != 0 && has(bar) && !(self.live.since > bar as f64 * 1000.0) && !self.live.cells.is_empty())
            .then(|| (bar, self.live.cells.iter().map(|(&k, c)| vec![k as f64 * bucket, c[0], c[1], c[2], c[3], c[4], c[5]]).collect()));
        Some((bucket, merged))
    }
    /// 6713 chartFull: 라이브 캔들 앞에 1주일 몫(라이브 첫 시각보다 앞)
    pub fn chart_full(&self) -> Vec<Candle> {
        match self.week.as_ref().filter(|w| !w.0.is_empty()) {
            None => self.candles.clone(),
            Some(w) => {
                let t0 = self.candles.first().map_or(i64::MAX, |c| c.time);
                w.0.iter().filter(|c| c.time < t0).chain(self.candles.iter()).copied().collect()
            }
        }
    }
    /// 그릴 것이 바뀌었나(부르는 쪽이 다시 그리기를 요청)
    pub fn take_dirty(&mut self) -> bool { std::mem::take(&mut self.dirty) }
}

/// rtcore/src/proto.rs `Footprint`(밀기 메시지)와 같은 모양 -- 그 크레이트는 tokio 를 끌어와 웹어셈블리에 못 넣어 형만 거울로 둔다
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Push {
    pub full: bool,
    pub bucket: f64,
    pub bar_seconds: f64,
    pub bars_expected: usize,
    #[serde(default)]
    pub ready: bool,
    pub retail_max_usd: f64,
    pub whale_min_usd: f64,
    pub bars: Vec<FpBar>,
    #[serde(default)]
    pub drop: Vec<i64>,
    #[serde(default)]
    pub holes: Vec<i64>,
}

#[cfg(test)]
mod t {
    use super::*;
    #[test]
    fn rounding_and_merge_rules() {
        assert_eq!((round_half_even(3490.5), round_half_even(3491.5), round_half_even(3490.25)), (3490, 3492, 3490));
        let mut f = Feed::new("eth", "ETHUSDT", Source::Poll, Urls::default(), false);
        f.fp = Some(Footprint { bucket: 0.5, ..Default::default() });
        f.fp_bars.insert(600, FpBar { time: 600, levels: vec![vec![1.0, 9.0, 9.0, 0.0, 0.0, 0.0, 0.0]], dhl: None });
        // 봉 중간에 붙음(since > 봉 시작) → 서버 값 유지
        f.live_add(1.0, 1.0, 650_000.0, false, Some(1));
        assert_eq!(f.for_chart(false).unwrap().1, None);
        // 봉이 열리기 전부터 붙어 있었다 → 브라우저 값으로 바꿔 낀다 · 가격 0 은 on_ws 가 버린다
        f.live = Live { since: 599_000.0, ..Live::default() };
        f.on_ws(r#"{"e":"trade","p":"0","q":"5","T":600100,"m":false,"t":2}"#, 0.0);
        f.on_ws(r#"{"e":"trade","p":"1.0","q":"2","T":600100,"m":true,"t":3}"#, 0.0);
        assert_eq!(f.for_chart(false).unwrap().1, Some((600, vec![vec![1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]])));
        // 밀기: 칸 단위로 덮어쓰고 안 실린 칸은 그대로
        f.apply_push(Push { full: false, bucket: 0.5, bar_seconds: 300.0, bars_expected: 144, ready: true, retail_max_usd: 0.0, whale_min_usd: 0.0,
                            bars: vec![FpBar { time: 600, levels: vec![vec![1.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dhl: None }], drop: vec![], holes: vec![] });
        assert_eq!(f.fp_bars[&600].levels.len(), 2);
    }
    #[test]
    fn hub_messages() {
        let mut f = Feed::new("eth", "ETHUSDT", Source::Push, Urls::default(), false);
        f.ensure_ws(0.0, false);
        assert_eq!(f.out.drain(..).collect::<Vec<_>>(), vec![Req::WsOpen("/rt".into()), Req::WsSend(r#"{"subscribe":["ethusdt"]}"#.into())]);
        let fp = r#"{"full":true,"bucket":0.5,"barSeconds":300,"barsExpected":144,"ready":false,"retailMaxUsd":1,"whaleMinUsd":2,"bars":[{"time":600,"levels":[[1.0,1,2,0,0,0,0]]}],"holes":[600],"새칸":1}"#;
        f.on_ws(&format!(r#"{{"s":"ethusdt","seq":5,"ex_ms":0,"rx_us":0,"tx_us":1,"type":"snapshot","book":null,"mark":null,"trades":[],"footprint":{fp}}}"#), 0.0);
        assert_eq!((f.fp_bars.len(), f.holes.clone(), f.hub_seq), (1, vec![600], Some(5)));
        // 모르는 사건 종류 = seq 만 센다 · 체결 = 직결과 같은 길
        f.on_ws(r#"{"s":"ethusdt","seq":6,"ex_ms":1,"rx_us":0,"tx_us":1,"type":"뭔가_새것","x":1}"#, 0.0);
        f.on_ws(r#"{"s":"ethusdt","seq":7,"ex_ms":600100,"rx_us":0,"tx_us":1,"type":"trade","id":9,"px":1.0,"qty":2.0,"sell":true}"#, 0.0);
        assert_eq!((f.price, f.live.cells.len(), f.hub_seq), (1.0, 1, Some(7)));
        // 증분에 holes 가 없으면 = 구멍 없음
        f.on_ws(r#"{"s":"ethusdt","seq":8,"ex_ms":0,"rx_us":0,"tx_us":1,"type":"footprint","full":false,"bucket":0.5,"barSeconds":300,"barsExpected":144,"ready":true,"retailMaxUsd":1,"whaleMinUsd":2,"bars":[]}"#, 0.0);
        assert!(f.holes.is_empty());
        // 1초 수급 증분 = s1 칸(초마다 덮어쓰기)
        f.on_ws(r#"{"s":"ethusdt","seq":9,"ex_ms":0,"rx_us":0,"tx_us":1,"type":"supply1s","now":601,"seconds":[[600,1,0,0,0,2,0,9.5]],"partial":{"bn":[601,0,0,0,0,0,1,9.6],"okx":null,"spot":null}}"#, 0.0);
        assert_eq!((f.s1.now, f.s1.bn.len(), f.s1.bn[&600][4], f.hub_seq), (601, 2, 2.0, Some(9)));
        // 빈 번호 → 끊고 다시(다음 tick 이 연다) · 그 메시지는 안 먹는다
        f.on_ws(r#"{"s":"ethusdt","seq":11,"ex_ms":600200,"rx_us":0,"tx_us":1,"type":"trade","id":10,"px":3.0,"qty":2.0,"sell":true}"#, 0.0);
        assert_eq!((f.price, f.out.drain(..).collect::<Vec<_>>()), (1.0, vec![Req::WsClose]));
        f.tick(1.0, false, false);
        assert!(f.out.contains(&Req::WsOpen("/rt".into())));
    }
}
