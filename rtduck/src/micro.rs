//! 미시구조 스캐너(`microstructure_scanner.py`, 줄 번호는 그 파일) -- market_data_worker 의 마지막 몫. 종목마다(ETH·BTC·SOL):
//!   웹소켓 `@depth20@100ms`(호가 20단) · `@aggTrade`(체결) + REST 10초 폴링(`openInterest` · `premiumIndex` · 체결 웹소켓이 30초 멎으면 `aggTrades` 대체)
//!   → 10초마다 compute_signal · 분이 바뀐 첫 10초 틱에 `microstructure.duckdb::microstructure_1m{,_btc,_sol}` 34칸 한 행.
//! 읽는 곳: 감시 check_data_sources(연결 칸) · 신선도 3줄 · seal_to_lake micro_1m. 수치는 numpy 와 같은 식(np_sum 짝 더하기 · percentile linear).
use duckdb::{params, Connection};
use serde_json::{json, Map, Value};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

pub const DEPTH_WS: &str = "wss://fstream.binance.com/ws/{symbol}@depth20@100ms";
pub const TRADE_WS: &str = "wss://fstream.binance.com/market/ws/{symbol}@aggTrade";
pub const OI_URL: &str = "https://fapi.binance.com/fapi/v1/openInterest?symbol={SYMBOL}";
pub const FUND_URL: &str = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol={SYMBOL}";
pub const AGG_URL: &str = "https://fapi.binance.com/fapi/v1/aggTrades?symbol={SYMBOL}&limit={LIMIT}";

use rtcollect::ob5m::np_sum;

fn np_mean(a: &[f64]) -> f64 {
    np_sum(a) / a.len() as f64
}

fn np_std(a: &[f64]) -> f64 {
    let m = np_mean(a);
    let sq: Vec<f64> = a.iter().map(|x| (x - m) * (x - m)).collect();
    (np_sum(&sq) / a.len() as f64).sqrt()
}

/// np.percentile(a, q) 기본(linear): 정렬 · 가상 자리 (n-1)·q/100 · numpy _lerp(t ≥ 0.5 면 위쪽에서 뺀다). 2만 배열 비트 대조(2026-11-21).
pub fn np_percentile(a: &[f64], q: f64) -> f64 {
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = s.len();
    let vi = (n - 1) as f64 * (q / 100.0);
    if vi >= (n - 1) as f64 {
        return s[n - 1];
    }
    let i = vi.floor();
    let g = vi - i;
    let (x, y) = (s[i as usize], s[i as usize + 1]);
    let d = y - x;
    if g >= 0.5 { y - d * (1.0 - g) } else { x + d * g }
}

fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

fn sign(x: f64) -> i64 {
    if x > 0.0 { 1 } else if x < 0.0 { -1 } else { 0 }
}

fn push<T>(d: &mut VecDeque<T>, cap: usize, v: T) {
    if d.len() == cap {
        d.pop_front();
    }
    d.push_back(v);
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn pf(v: Option<&Value>, d: f64) -> Result<f64, ()> {
    match v {
        None => Ok(d),
        Some(Value::Number(n)) => n.as_f64().ok_or(()),
        Some(Value::String(s)) => s.trim().parse::<f64>().map_err(|_| ()),
        Some(Value::Bool(b)) => Ok(*b as u8 as f64),
        _ => Err(()),
    }
}

fn pi(v: Option<&Value>) -> Result<i64, ()> {
    match v {
        None => Ok(0),
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().filter(|f| f.is_finite()).map(|f| f.trunc() as i64)).ok_or(()),
        Some(Value::String(s)) => s.trim().parse::<i64>().map_err(|_| ()),
        Some(Value::Bool(b)) => Ok(*b as i64),
        _ => Err(()),
    }
}

fn env_f(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// `MicrostructureScanner` 상태(:34-114).
pub struct Scanner {
    pub symbol: String,
    pub table: String,
    pub db: PathBuf,
    pub depth_connected: bool,
    pub trade_connected: bool,
    pub poll_connected: bool,
    depth_last: f64,
    pub trade_last: f64,
    poll_last: f64,
    bids: Vec<(f64, f64)>,
    asks: Vec<(f64, f64)>,
    trades: VecDeque<(i64, bool, f64)>,
    trade_nq: VecDeque<(i64, f64, f64)>,
    price_hist: VecDeque<f64>,
    oi_hist: VecDeque<f64>,
    price_obs: VecDeque<(i64, f64)>,
    oi_obs: VecDeque<(i64, f64)>,
    nif_hist: VecDeque<f64>,
    oi_delta_hist: VecDeque<f64>,
    abs_hist: VecDeque<f64>,
    tox_hist: VecDeque<f64>,
    bias_hist: VecDeque<i64>,
    eai_hist: VecDeque<f64>,
    fund_rate: f64,
    last_agg_id: Option<i64>,
    prev_bid10: f64,
    prev_ask10: f64,
    /// `_cached` 중 compute_signal 이 되읽는 세 칸(oi_delta_pct · oi_delta_cum_5m · oi_delta_cum_5m_bucket_start_ts). 없으면 None.
    c_oi: Option<f64>,
    c_cum: Option<f64>,
    c_bucket: Option<i64>,
    warmup: bool,
    boot_rows: i64,
    last_hist_min: Option<i64>,
    last_oi_tick: Option<f64>,
    oi_cum_5m: f64,
    oi_5m_bucket: i64,
    whale_5m_bucket: i64,
    whale_buy_5m: f64,
    whale_sell_5m: f64,
    whale_state: i64,
    obi_wall_th: f64,
    taker_buy_th: f64,
    whale_usd_th: f64,
    eai_threshold: f64,
    pub poll_interval: f64,
    pub fallback_enabled: bool,
    pub fallback_limit: i64,
    nif_th: f64,
    entry_th: f64,
    exit_th: f64,
    oi_th: f64,
    window_min: i64,
    pos_th: f64,
    last_saved_minute: i64,
}

impl Scanner {
    pub fn new(symbol: &str, db: &Path) -> Self {
        let s = symbol.to_lowercase();
        Self {
            table: if s == "ethusdt" { "microstructure_1m".into() } else { format!("microstructure_1m_{}", s.replace("usdt", "")) },
            symbol: s, db: db.into(), depth_connected: false, trade_connected: false, poll_connected: false, depth_last: 0.0, trade_last: 0.0, poll_last: 0.0,
            bids: vec![], asks: vec![], trades: VecDeque::new(), trade_nq: VecDeque::new(), price_hist: VecDeque::new(), oi_hist: VecDeque::new(),
            price_obs: VecDeque::new(), oi_obs: VecDeque::new(), nif_hist: VecDeque::new(), oi_delta_hist: VecDeque::new(), abs_hist: VecDeque::new(),
            tox_hist: VecDeque::new(), bias_hist: VecDeque::new(), eai_hist: VecDeque::new(), fund_rate: 0.0, last_agg_id: None, prev_bid10: 0.0, prev_ask10: 0.0,
            c_oi: None, c_cum: None, c_bucket: None, warmup: false, boot_rows: 0, last_hist_min: None, last_oi_tick: None, oi_cum_5m: 0.0, oi_5m_bucket: -1,
            whale_5m_bucket: -1, whale_buy_5m: 0.0, whale_sell_5m: 0.0, whale_state: 0,
            obi_wall_th: env_f("MS_OBI_WALL_TH", -0.40), taker_buy_th: env_f("MS_TAKER_BUY_TH", 0.60), whale_usd_th: env_f("MS_WHALE_USD_TH", 100000.0),
            eai_threshold: env_f("MS_EAI_THRESHOLD", 2.0), poll_interval: env_f("MS_POLL_INTERVAL_SEC", 10.0),
            fallback_enabled: ["1", "true", "yes", "on"].contains(&std::env::var("MS_TRADE_REST_FALLBACK_ENABLE").unwrap_or_else(|_| "true".into()).trim().to_lowercase().as_str()),
            fallback_limit: env_f("MS_TRADE_REST_FALLBACK_LIMIT", 500.0) as i64, nif_th: env_f("MS_WHALE_INTENT_NIF_TH", 0.10),
            entry_th: env_f("MS_WHALE_FLOW_ENTRY_TH", 0.12), exit_th: env_f("MS_WHALE_FLOW_EXIT_TH", 0.08), oi_th: env_f("MS_WHALE_INTENT_OI_TH", 0.0001),
            window_min: env_f("MS_WHALE_INTENT_WINDOW_MIN", 5.0) as i64, pos_th: env_f("MS_WHALE_POS_LONG_SHORT_TH", 0.25), last_saved_minute: -1,
        }
    }

    fn ready30(&self) -> bool {
        [self.price_hist.len(), self.nif_hist.len(), self.abs_hist.len(), self.tox_hist.len(), self.bias_hist.len()].iter().all(|n| *n >= 30)
    }

    /// `_db_init`(:117-223). now = 부트스트랩 기준 시각(원본은 DuckDB now() · time.time() -- 운영은 같은 벽시계를 넣는다).
    pub fn init(&mut self, now: f64) -> duckdb::Result<usize> {
        if let Some(d) = self.db.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let con = Connection::open(&self.db)?;
        con.execute_batch(&format!("
            CREATE TABLE IF NOT EXISTS {} (
                ts TIMESTAMPTZ, obi DOUBLE, taker_buy_ratio DOUBLE, spoofing_score DOUBLE,
                nif_whale DOUBLE, nif_retail DOUBLE, eai DOUBLE, oi_delta_pct DOUBLE,
                funding_rate DOUBLE, kelly_mult DOUBLE, signal_bias INTEGER,
                shadow_toxicity_score DOUBLE, shadow_toxicity_regime VARCHAR,
                shadow_queue_collapse DOUBLE, shadow_absorption_score DOUBLE,
                shadow_queue_bias INTEGER, shadow_regime_tag VARCHAR, shadow_regime_conf DOUBLE
            )
        ", self.table))?;
        let have: Vec<String> = con.prepare(&format!("PRAGMA table_info('{}')", self.table))?.query_map([], |r| r.get::<_, String>(1))?.collect::<Result<_, _>>()?;
        for (c, ty) in [("data_stale", "BOOLEAN"), ("depth_connected", "BOOLEAN"), ("trade_connected", "BOOLEAN"), ("poll_connected", "BOOLEAN"),
                        ("depth_age_sec", "DOUBLE"), ("trade_age_sec", "DOUBLE"), ("poll_age_sec", "DOUBLE"), ("recent_trade_count_5m", "INTEGER"),
                        ("recent_trade_notional_5m", "DOUBLE"), ("recent_whale_count_5m", "INTEGER"), ("valid_taker_flow", "BOOLEAN"), ("valid_nif", "BOOLEAN"),
                        ("warmup_30m_ready", "BOOLEAN"), ("schema_version", "INTEGER"), ("mark_price", "DOUBLE"), ("whale_position_score", "DOUBLE")] {
            if !have.iter().any(|h| h == c) {
                con.execute_batch(&format!("ALTER TABLE {} ADD COLUMN {c} {ty}", self.table))?;
            }
        }
        // 🔴묶은 DuckDB 에는 ICU 가 없어 TIMESTAMPTZ − INTERVAL 이 안 된다(Binder Error) -- 같은 순간 비교를 epoch µs 로(원본 ts >= now() - INTERVAL)
        let cut = |s: f64| (now * 1e6).round() as i64 - (s * 1e6) as i64;
        type R = (Option<f64>, Option<f64>, Option<f64>, Option<f64>, Option<f64>, Option<i64>, Option<f64>, Option<f64>);
        let rows: Vec<R> = con.prepare(&format!("
                SELECT
                    epoch_ms(ts),
                    nif_whale, oi_delta_pct, shadow_absorption_score, shadow_toxicity_score, shadow_queue_bias, eai, mark_price
                FROM {}
                WHERE epoch_us(ts) >= ?
                ORDER BY ts ASC", self.table)).and_then(|mut st| st.query_map([cut(3600.0)], |r| Ok((r.get::<_, Option<i64>>(0)?.map(|x| x as f64), r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?)))?.collect()).unwrap_or_else(|e| { eprintln!("MS bootstrap skip: {e}"); vec![] });
        let rows5: Vec<Option<f64>> = con.prepare(&format!("SELECT oi_delta_pct FROM {} WHERE epoch_us(ts) >= ? ORDER BY ts ASC", self.table))
            .and_then(|mut st| st.query_map([cut(300.0)], |r| r.get(0))?.collect()).unwrap_or_default();
        for (ts, nif, oid, ab, tox, bias, eai, mark) in &rows {
            let ts_ms = ts.unwrap_or(0.0) as i64;
            push(&mut self.nif_hist, 60, nif.unwrap_or(0.0));
            push(&mut self.oi_delta_hist, 60, oid.unwrap_or(0.0));
            push(&mut self.abs_hist, 60, ab.unwrap_or(0.0));
            push(&mut self.tox_hist, 60, tox.unwrap_or(0.0));
            push(&mut self.bias_hist, 60, bias.unwrap_or(0));
            push(&mut self.eai_hist, 60, eai.unwrap_or(0.0));
            let p = mark.unwrap_or(0.0);
            if p > 0.0 {
                push(&mut self.price_hist, 60, p);
                push(&mut self.price_obs, 120, (ts_ms, p));
                self.last_hist_min = Some(ts_ms.div_euclid(60_000));
            }
        }
        self.boot_rows = rows.len() as i64;
        if !rows5.is_empty() {
            let mut oi5 = 0.0;
            for d in &rows5 {
                oi5 = (1.0 + oi5) * (1.0 + d.unwrap_or(0.0)) - 1.0;
            }
            (self.c_oi, self.c_cum, self.c_bucket) = (Some(oi5), Some(oi5), Some((now as i64).div_euclid(300) * 300));
        }
        self.warmup = self.ready30();
        Ok(rows.len())
    }

    /// 깊이 메시지(:292-301) -- 시각은 JSON 이 읽히면 찍는다(호가가 깨져도).
    pub fn on_depth(&mut self, now: f64, raw: &str) {
        let Ok(Value::Object(m)) = serde_json::from_str::<Value>(raw) else { return };
        self.depth_last = now;
        let side = |k: &str| -> Result<Vec<(f64, f64)>, ()> {
            match m.get(k) {
                None => Ok(vec![]),
                Some(Value::Array(a)) => a.iter().map(|r| match r {
                    Value::Array(x) if x.len() == 2 => Ok((pf(Some(&x[0]), 0.0)?, pf(Some(&x[1]), 0.0)?)),
                    _ => Err(()),
                }).collect(),
                _ => Err(()),
            }
        };
        if let (Ok(b), Ok(a)) = (side("b"), side("a")) {
            (self.bids, self.asks) = (b, a);
        }
    }

    /// 체결 메시지(:330-341).
    pub fn on_trade(&mut self, now: f64, raw: &str) {
        let Ok(Value::Object(m)) = serde_json::from_str::<Value>(raw) else { return };
        self.trade_last = now;
        let bm = m.get("m").is_some_and(truthy);
        let (Ok(q), Ok(p), Ok(ts)) = (pf(m.get("q"), 0.0), pf(m.get("p"), 0.0), pi(m.get("T"))) else { return };
        let usd = q * p;
        if usd > 0.0 {
            push(&mut self.trades, 50_000, (ts, bm, usd));
            push(&mut self.trade_nq, 50_000, (ts, usd, q));
        }
    }

    /// 폴링 한 바퀴(:400-447). get(url) = 응답 JSON(상태 코드는 보지 않는다 -- 원본 json(content_type=None)) · Err = 예외.
    pub fn poll(&mut self, get: &mut dyn FnMut(&str) -> Result<Value, String>, now: &mut dyn FnMut() -> f64) {
        self.poll_connected = true;
        let sym = self.symbol.to_uppercase();
        let r = (|| -> Result<(), String> {
            let d = get(&OI_URL.replace("{SYMBOL}", &sym))?;
            let Value::Object(d) = d else { return Err("AttributeError".into()) };
            let oi = pf(d.get("openInterest"), 0.0).map_err(|_| "ValueError")?;
            let t = now();
            let now_ms = (t * 1000.0) as i64;
            let prev = self.last_oi_tick.unwrap_or(oi);
            self.last_oi_tick = Some(oi);
            push(&mut self.oi_obs, 120, (now_ms, oi));
            let tick = (oi - prev) / (prev.abs() + 1e-8);
            self.c_oi = Some(tick);
            let b5 = (t / 300.0).floor() as i64;
            if b5 != self.oi_5m_bucket {
                self.oi_5m_bucket = b5;
                self.oi_cum_5m = 0.0;
            }
            self.oi_cum_5m = (1.0 + self.oi_cum_5m) * (1.0 + tick) - 1.0;
            (self.c_cum, self.c_bucket) = (Some(self.oi_cum_5m), Some(b5 * 300));
            let d = get(&FUND_URL.replace("{SYMBOL}", &sym))?;
            let Value::Object(d) = d else { return Err("AttributeError".into()) };
            self.fund_rate = pf(d.get("lastFundingRate"), 0.0).map_err(|_| "ValueError")?;
            let mark = pf(d.get("markPrice"), 0.0).map_err(|_| "ValueError")?;
            if mark > 0.0 {
                push(&mut self.price_obs, 120, (now_ms, mark));
            }
            let bm = (now() / 60.0).floor() as i64;
            if self.last_hist_min != Some(bm) {
                self.last_hist_min = Some(bm);
                if oi > 0.0 {
                    push(&mut self.oi_hist, 60, oi);
                }
                if mark > 0.0 {
                    push(&mut self.price_hist, 60, mark);
                }
            }
            if now() - self.trade_last > 30.0 {
                self.agg_fallback(get, now);
            }
            self.poll_last = now();
            Ok(())
        })();
        if let Err(e) = r {
            self.poll_connected = false;
            eprintln!("MS poll REST failed; retrying... ({e})");
        }
    }

    /// `_poll_recent_agg_trades`(:354-392).
    fn agg_fallback(&mut self, get: &mut dyn FnMut(&str) -> Result<Value, String>, now: &mut dyn FnMut() -> f64) {
        if !self.fallback_enabled {
            return;
        }
        let url = AGG_URL.replace("{SYMBOL}", &self.symbol.to_uppercase()).replace("{LIMIT}", &self.fallback_limit.clamp(1, 1000).to_string());
        let Ok(Value::Array(rows)) = get(&url) else { return };
        if rows.is_empty() {
            return;
        }
        let (mut appended, mut max_id) = (0, self.last_agg_id);
        for row in &rows {
            let Value::Object(r) = row else { continue };
            let Ok(id) = pi(r.get("a")) else { continue };
            if self.last_agg_id.is_some_and(|l| id <= l) {
                continue;
            }
            let (Ok(p), Ok(q), Ok(ts)) = (pf(r.get("p"), 0.0), pf(r.get("q"), 0.0), pi(r.get("T"))) else { continue };
            let bm = r.get("m").is_some_and(truthy);
            let usd = p * q;
            if usd <= 0.0 || ts <= 0 {
                continue;
            }
            push(&mut self.trades, 50_000, (ts, bm, usd));
            push(&mut self.trade_nq, 50_000, (ts, usd, q));
            appended += 1;
            max_id = Some(max_id.map_or(id, |m| m.max(id)));
        }
        if max_id.is_some() {
            self.last_agg_id = max_id;
        }
        if appended > 0 {
            self.trade_last = now();
            self.trade_connected = true;
        }
    }

    fn bid10(&self) -> f64 {
        self.bids.iter().take(10).fold(0.0, |s, x| s + x.1)
    }
    fn ask10(&self) -> f64 {
        self.asks.iter().take(10).fold(0.0, |s, x| s + x.1)
    }

    /// `compute_signal`(:550-818) -- 원본 dict 와 같은 열쇠·순서의 JSON(DB 칸은 여기서 뽑는다).
    pub fn compute_signal(&mut self, now: f64) -> Map<String, Value> {
        let age = |l: f64| (l > 0.0).then(|| now - l);
        let (da, ta, pa) = (age(self.depth_last), age(self.trade_last), age(self.poll_last));
        let depth_stale = da.is_none_or(|a| a > 10.0) || !self.depth_connected;
        let trade_stale = ta.is_none_or(|a| a > 30.0) || !self.trade_connected;
        let poll_stale = pa.is_none_or(|a| a > 130.0) || !self.poll_connected;
        // _compute_obi(:449-454)
        let obi = if self.bids.is_empty() || self.asks.is_empty() { 0.0 } else { let (b, a) = (self.bid10(), self.ask10()); (b - a) / (b + a + 1e-8) };
        // _compute_nif_and_taker(:456-479)
        let cut = (now * 1000.0) as i64 - 300_000;
        let (mut wb, mut ws, mut rb, mut rs, mut tb, mut tsum, mut n, mut wn, mut tot) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0i64, 0i64, 0.0);
        for (t, bm, usd) in &self.trades {
            if *t < cut {
                continue;
            }
            n += 1;
            tot += usd;
            let buy = !bm;
            if buy { tb += usd } else { tsum += usd }
            if *usd >= self.whale_usd_th {
                wn += 1;
                if buy { wb += usd } else { ws += usd }
            } else if buy {
                rb += usd
            } else {
                rs += usd
            }
        }
        let nif_whale = (wb - ws) / (wb + ws + 1e-8);
        let nif_retail = (rb - rs) / (rb + rs + 1e-8);
        let tbr_raw = tb / (tb + tsum + 1e-8);
        let valid_taker = !trade_stale && n > 0;
        let valid_nif = !trade_stale && wn > 0;
        let tbr = if valid_taker { tbr_raw } else { 0.5 };
        // _compute_eai(:497-507)
        let c5 = (now * 1000.0) as i64 - 300_000;
        let prices: Vec<f64> = self.price_obs.iter().filter(|x| x.0 >= c5).map(|x| x.1).collect();
        let ois: Vec<f64> = self.oi_obs.iter().filter(|x| x.0 >= c5).map(|x| x.1).collect();
        let eai = if prices.len() < 2 || ois.len() < 2 { 0.0 } else {
            let (mx, mn) = (prices.iter().copied().fold(f64::NEG_INFINITY, f64::max), prices.iter().copied().fold(f64::INFINITY, f64::min));
            let pr = (mx - mn) / (np_mean(&prices) + 1e-8);
            let od = (ois[ois.len() - 1] - ois[0]).abs() / (ois[0].abs() + 1e-8);
            clip(od / (pr + 1e-6), 0.0, 20.0)
        };
        // _detect_spoofing(:542-548)
        let (wall, buy_th, sell_th) = (self.obi_wall_th.abs(), self.taker_buy_th, 1.0 - self.taker_buy_th);
        let (spoof, spoof_bias) = if obi <= -wall && tbr >= buy_th { (clip((obi.abs() - wall) * (tbr - buy_th) * 10.0, 0.0, 1.0), 1) }
            else if obi >= wall && tbr <= sell_th { (clip((obi - wall) * (sell_th - tbr) * 10.0, 0.0, 1.0), -1) } else { (0.0, 0) };
        let valid_ois: Vec<f64> = self.oi_obs.iter().filter(|x| x.0 >= c5).map(|x| x.1).collect();
        let oi_delta = if valid_ois.len() >= 2 { (valid_ois[valid_ois.len() - 1] - valid_ois[0]) / (valid_ois[0].abs() + 1e-8) } else { self.c_oi.unwrap_or(0.0) };
        // _compute_whale_flow_window(10)
        let c10 = (now * 1000.0) as i64 - 10_000;
        let (mut wb10, mut ws10) = (0.0, 0.0);
        for (t, bm, usd) in &self.trades {
            if *t < c10 || *usd < self.whale_usd_th {
                continue;
            }
            if !bm { wb10 += usd } else { ws10 += usd }
        }
        let wf10 = (wb10 - ws10) / (wb10 + ws10 + 1e-8);
        let b5 = (now / 300.0).floor() as i64;
        if b5 != self.whale_5m_bucket {
            (self.whale_5m_bucket, self.whale_buy_5m, self.whale_sell_5m) = (b5, 0.0, 0.0);
        }
        self.whale_buy_5m += wb10;
        self.whale_sell_5m += ws10;
        let wf5 = (self.whale_buy_5m - self.whale_sell_5m) / (self.whale_buy_5m + self.whale_sell_5m + 1e-8);
        // MFT(:598-645)
        let ph: Vec<f64> = self.price_hist.iter().copied().collect();
        let (mut pc30, mut pv30, mut bo, mut bd) = (0.0, 0.0, false, false);
        if ph.len() >= 30 {
            let (pn, p30) = (ph[ph.len() - 1], ph[ph.len() - 30]);
            pc30 = (pn - p30) / (p30 + 1e-8);
            let w = &ph[ph.len() - 30..];
            pv30 = (w.iter().copied().fold(f64::NEG_INFINITY, f64::max) - w.iter().copied().fold(f64::INFINITY, f64::min)) / (pn + 1e-8);
        }
        if ph.len() >= 15 {
            let pn = ph[ph.len() - 1];
            let past = &ph[..ph.len() - 5];
            if !past.is_empty() {
                bo = pn > past.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                bd = pn < past.iter().copied().fold(f64::INFINITY, f64::min);
            }
        }
        let mut vwap_gap = 0.0;
        if !self.trade_nq.is_empty() && !ph.is_empty() {
            let c15 = (now * 1000.0) as i64 - 15 * 60 * 1000;
            let (mut nt, mut qs) = (0.0, 0.0);
            for (t, ntl, q) in self.trade_nq.iter().rev() {
                if *t < c15 {
                    break;
                }
                nt += ntl;
                qs += q;
            }
            if qs > 0.0 {
                let v = nt / qs;
                vwap_gap = (ph[ph.len() - 1] - v) / (v + 1e-8);
            }
        }
        let last30 = |d: &VecDeque<f64>| d.iter().skip(d.len().saturating_sub(30)).copied().collect::<Vec<f64>>();
        let nif30 = last30(&self.nif_hist);
        let nif_sum = nif30.iter().fold(0.0, |s, x| s + x);
        let (nif_avg, nif_std) = if nif30.is_empty() { (0.0, 0.0) } else { (np_mean(&nif30), np_std(&nif30)) };
        let m30 = |v: Vec<f64>| if v.is_empty() { 0.0 } else { np_mean(&v) };
        let abs_avg = m30(last30(&self.abs_hist));
        let bias_avg = m30(self.bias_hist.iter().skip(self.bias_hist.len().saturating_sub(30)).map(|x| *x as f64).collect());
        let tox_avg = m30(last30(&self.tox_hist));
        let eai_d15 = if self.eai_hist.len() >= 15 { self.eai_hist[self.eai_hist.len() - 1] - self.eai_hist[self.eai_hist.len() - 15] } else { 0.0 };
        // 고래 의도(:649-718)
        let flow = nif_whale;
        let flow_th = 0.05f64.max(self.nif_th);
        let entry = self.entry_th.max(self.exit_th + 1e-6);
        let exit = self.exit_th.min(entry - 1e-6);
        match self.whale_state {
            0 => {
                if flow >= entry {
                    self.whale_state = 1
                } else if flow <= -entry {
                    self.whale_state = -1
                }
            }
            1 => {
                if flow <= exit {
                    self.whale_state = 0
                }
            }
            _ => {
                if flow >= -exit {
                    self.whale_state = 0
                }
            }
        }
        let fs = self.whale_state;
        let oi5 = oi_delta;
        let mut r = [0.0f64; 8]; // short_build, long_close, sell_presence, sell_effective, long_build, short_cover, buy_presence, buy_effective
        let bias_txt = if fs > 0 {
            r[6] = 1.0;
            if oi5 > self.oi_th { r[4] = 1.0; r[7] = 1.0; "🟢 신규 롱 구축 우세" } else if oi5 < -self.oi_th { r[5] = 1.0; r[7] = 1.0; "🔵 숏 커버링(항복) 우세" } else { "🟣 매수 우세(의도 불명)" }
        } else if fs < 0 {
            r[2] = 1.0;
            if oi5 > self.oi_th { r[0] = 1.0; r[3] = 1.0; "🔴 신규 숏 구축 우세" } else if oi5 < -self.oi_th { r[1] = 1.0; r[3] = 1.0; "🟡 기존 롱 청산 우세" } else { "🟣 매도 우세(의도 불명)" }
        } else {
            "관망/중립"
        };
        let flow_str = clip(flow.abs() / (flow_th + 1e-8), 0.0, 1.5);
        let oi_str = clip(oi5.abs() / (self.oi_th + 1e-8), 0.0, 1.5);
        let sf = if flow >= 0.0 { 1.0 } else { -1.0 };
        let oiw = if oi5 > self.oi_th { 1.0 } else if oi5 < -self.oi_th { -0.35 } else { 0.0 };
        let pos = clip((0.7 * sf * flow_str.min(1.0)) + (0.3 * sf * oiw * oi_str.min(1.0)), -1.0, 1.0);
        let est = if pos >= self.pos_th { "LONG" } else if pos <= -self.pos_th { "SHORT" } else { "NEUTRAL" };
        let conf = clip(pos.abs() * 100.0, 0.0, 99.0) as i64;
        let nif_bias = if nif_whale < -0.30 && nif_retail > 0.10 { -1 } else if nif_whale > 0.30 && nif_retail < -0.10 { 1 } else { 0 };
        let eai_bias = if eai > self.eai_threshold && self.fund_rate > 0.0010 { -1 } else if eai > self.eai_threshold && self.fund_rate < -0.0010 { 1 } else { 0 };
        let mut kelly = 1.0;
        if nif_bias == -1 {
            kelly *= 0.40;
        }
        if spoof > 0.3 {
            let pb = spoof_bias + nif_bias + eai_bias;
            kelly *= if pb != 0 && sign(pb as f64) == sign(spoof_bias as f64) { 1.20 } else { 0.70 };
        }
        if eai > self.eai_threshold {
            kelly *= 1.30;
        }
        let kelly = clip(kelly, 0.30, 2.0);
        let raw = spoof_bias + nif_bias + eai_bias;
        let signal_bias = if raw != 0 { sign(raw as f64) } else { 0 };
        // 그림자(:509-540)
        let mismatch = (obi - (2.0 * tbr - 1.0)).abs();
        let mut burst = 0.0;
        if !self.trades.is_empty() {
            let rec: Vec<f64> = self.trades.iter().skip(self.trades.len().saturating_sub(120)).map(|x| x.2).collect();
            if !rec.is_empty() {
                burst = clip(np_percentile(&rec, 95.0) / (np_mean(&rec) + 1e-8), 0.0, 5.0);
            }
        }
        let tox = clip(0.7 * mismatch + 0.3 * (burst / 5.0), 0.0, 1.0);
        let tox_reg = if tox >= 0.75 { "toxic" } else if tox >= 0.50 { "watch" } else { "normal" };
        let (bv, av) = (if self.bids.is_empty() { 0.0 } else { self.bid10() }, if self.asks.is_empty() { 0.0 } else { self.ask10() });
        let pb = if self.prev_bid10 > 0.0 { self.prev_bid10 } else { bv };
        let pa_ = if self.prev_ask10 > 0.0 { self.prev_ask10 } else { av };
        let qc = clip(((pb - bv) / (pb + 1e-8)).max((pa_ - av) / (pa_ + 1e-8)).max(0.0), 0.0, 1.0);
        let fsg = 2.0 * tbr - 1.0;
        let bsg = (bv - av) / (bv + av + 1e-8);
        let absorb = clip((fsg - bsg).abs() / 2.0, 0.0, 1.0);
        let qbias = if fsg.abs() > 0.10 { sign(fsg) } else { 0 };
        (self.prev_bid10, self.prev_ask10) = (bv, av);
        let f = self.fund_rate;
        let (rtag, rconf) = if eai > self.eai_threshold && f.abs() > 0.0008 {
            ("squeeze", clip((eai / self.eai_threshold.max(1e-6)) * (f.abs() / 0.002).min(1.0), 0.0, 1.0))
        } else if oi_delta.abs() > 0.01 && eai < 1.2 {
            ("trend", clip(oi_delta.abs() / 0.03, 0.0, 1.0))
        } else {
            ("normal", clip(1.0 - (eai / self.eai_threshold.max(1e-6)).min(1.0), 0.0, 1.0))
        };
        let mark = self.price_obs.back().map_or(0.0, |x| x.1);
        self.warmup = self.ready30();
        let mut out = json!({
            "obi": obi, "taker_buy_ratio": tbr, "spoofing_score": spoof, "spoofing_bias": spoof_bias, "nif_whale": nif_whale, "nif_retail": nif_retail,
            "nif_bias": nif_bias, "eai": eai, "eai_bias": eai_bias, "oi_delta_pct": oi_delta, "funding_rate": self.fund_rate, "mark_price": mark,
            "oi_delta_cum_5m": self.c_cum.unwrap_or(0.0), "oi_delta_cum_5m_bucket_start_ts": self.c_bucket.unwrap_or(0),
            "whale_flow_10s_ratio": wf10, "whale_buy_10s_usd": wb10, "whale_sell_10s_usd": ws10, "whale_flow_cum_5m_ratio": wf5,
            "whale_buy_cum_5m_usd": self.whale_buy_5m, "whale_sell_cum_5m_usd": self.whale_sell_5m, "whale_flow_cum_5m_bucket_start_ts": b5 * 300,
            "kelly_mult": kelly, "signal_bias": signal_bias, "price_change_30m": pc30, "price_volatility_30m": pv30, "vwap_gap_15m": vwap_gap,
            "price_breakout_60m": bo, "price_breakdown_60m": bd, "nif_whale_sum_30m": nif_sum, "nif_whale_avg_30m": nif_avg, "nif_whale_std_30m": nif_std,
            "absorption_avg_30m": abs_avg, "bias_avg_30m": bias_avg, "toxicity_avg_30m": tox_avg, "eai_delta_15m": eai_d15,
            "whale_short_build_ratio_30m": r[0], "whale_long_close_ratio_30m": r[1], "whale_sell_presence_ratio_30m": r[2], "whale_sell_effective_ratio_30m": r[3],
            "whale_long_build_ratio_30m": r[4], "whale_short_cover_ratio_30m": r[5], "whale_buy_presence_ratio_30m": r[6], "whale_buy_effective_ratio_30m": r[7],
            "whale_position_bias_30m": bias_txt, "whale_position_window_min": self.window_min.max(1), "whale_position_estimate": est,
            "whale_position_confidence": conf, "whale_position_score": pos,
            "data_stale": depth_stale || trade_stale || poll_stale, "depth_connected": self.depth_connected, "trade_connected": self.trade_connected,
            "poll_connected": self.poll_connected, "depth_age_sec": da, "trade_age_sec": ta, "poll_age_sec": pa,
            "recent_trade_count_5m": n, "recent_trade_notional_5m": tot, "recent_whale_count_5m": wn, "valid_taker_flow": valid_taker, "valid_nif": valid_nif,
            "warmup_30m_ready": self.warmup, "warmup_price_samples": ph.len().min(30), "warmup_nif_samples": self.nif_hist.len().min(30),
            "warmup_abs_samples": self.abs_hist.len().min(30), "warmup_tox_samples": self.tox_hist.len().min(30), "warmup_bias_samples": self.bias_hist.len().min(30),
            "warmup_bootstrap_rows": self.boot_rows,
            "shadow_toxicity_score": tox, "shadow_toxicity_regime": tox_reg, "shadow_queue_collapse": qc, "shadow_absorption_score": absorb,
            "shadow_queue_bias": qbias, "shadow_regime_tag": rtag, "shadow_regime_conf": rconf,
        });
        std::mem::take(out.as_object_mut().unwrap())
    }

    /// `_scan_loop` 한 바퀴(:836-854) -- 10초 경계 + 0.05초. 분(서버 KST = UTC 와 분·초가 같다)이 바뀐 첫 15초 안 틱에 한 행.
    pub fn scan(&mut self, now: f64) -> Map<String, Value> {
        let sig = self.compute_signal(now);
        let g = |k: &str| sig[k].as_f64().unwrap_or(0.0);
        // self._cached = sig -- 되읽는 세 칸
        (self.c_oi, self.c_cum, self.c_bucket) = (Some(g("oi_delta_pct")), Some(g("oi_delta_cum_5m")), sig["oi_delta_cum_5m_bucket_start_ts"].as_i64());
        let sec = now.floor() as i64;
        let (minute, second) = (sec.div_euclid(60).rem_euclid(60), sec.rem_euclid(60));
        if minute != self.last_saved_minute && second < 15 {
            self.last_saved_minute = minute;
            push(&mut self.nif_hist, 60, g("nif_whale"));
            push(&mut self.oi_delta_hist, 60, g("oi_delta_pct"));
            push(&mut self.abs_hist, 60, g("shadow_absorption_score"));
            push(&mut self.tox_hist, 60, g("shadow_toxicity_score"));
            push(&mut self.bias_hist, 60, sig["shadow_queue_bias"].as_i64().unwrap_or(0));
            push(&mut self.eai_hist, 60, g("eai"));
            if let Err(e) = self.insert(sec.div_euclid(60) * 60 - 60, &sig) {
                eprintln!("MS duckdb insert failed: {e}");
            }
        }
        sig
    }

    fn insert(&self, bucket: i64, s: &Map<String, Value>) -> duckdb::Result<()> {
        let f = |k: &str| s[k].as_f64().unwrap_or(0.0);
        let o = |k: &str| s[k].as_f64();
        let b = |k: &str| s[k].as_bool().unwrap_or(false);
        let i = |k: &str| s[k].as_i64().unwrap_or(0);
        let st = |k: &str| s[k].as_str().unwrap_or("").to_string();
        let (vt, vn) = (b("valid_taker_flow"), b("valid_nif"));
        let mut con = None;
        for _ in 0..25 {
            match Connection::open(&self.db) {
                Ok(c) => {
                    con = Some(c);
                    break;
                }
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(200)),
            }
        }
        let con = match con { Some(c) => c, None => Connection::open(&self.db)? };
        con.execute(&format!("
                INSERT INTO {} (
                    ts, obi, taker_buy_ratio, spoofing_score,
                    nif_whale, nif_retail, eai, oi_delta_pct,
                    funding_rate, kelly_mult, signal_bias,
                    shadow_toxicity_score, shadow_toxicity_regime,
                    shadow_queue_collapse, shadow_absorption_score,
                    shadow_queue_bias, shadow_regime_tag, shadow_regime_conf,
                    data_stale, depth_connected, trade_connected, poll_connected,
                    depth_age_sec, trade_age_sec, poll_age_sec,
                    recent_trade_count_5m, recent_trade_notional_5m, recent_whale_count_5m,
                    valid_taker_flow, valid_nif, warmup_30m_ready, schema_version,
                    mark_price, whale_position_score
                ) VALUES (make_timestamptz(?),?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ", self.table), params![bucket * 1_000_000, f("obi"), vt.then(|| f("taker_buy_ratio")), f("spoofing_score"),
                    vn.then(|| f("nif_whale")), vt.then(|| f("nif_retail")), f("eai"), f("oi_delta_pct"), f("funding_rate"), f("kelly_mult"), i("signal_bias"),
                    f("shadow_toxicity_score"), st("shadow_toxicity_regime"), f("shadow_queue_collapse"), f("shadow_absorption_score"), i("shadow_queue_bias"),
                    st("shadow_regime_tag"), f("shadow_regime_conf"), b("data_stale"), b("depth_connected"), b("trade_connected"), b("poll_connected"),
                    o("depth_age_sec"), o("trade_age_sec"), o("poll_age_sec"), i("recent_trade_count_5m"), f("recent_trade_notional_5m"), i("recent_whale_count_5m"),
                    vt, vn, b("warmup_30m_ready"), 3i64, f("mark_price"), f("whale_position_score")])?;
        Ok(())
    }
}

/// 정답지 입력 먹이기(tools/gen_micro_fixture.py 와 같은 순서): ETH 빈 표 → 씨앗 행 → 종목마다 부트스트랩(now = t0) → 사건 열 + 10초 틱.
/// 사건 = [초, 종류, 종목, 본문]: D 호가 · T 체결 · P 폴링(본문 = {"oi","fund","agg"} 응답 -- 글자면 JSON 아님) · do/de 호가 접속/끊김 · to 체결 접속.
/// 반환 = (부트스트랩 행 수, 요청 URL 열, 틱마다 (종목, 신호 글자)).
pub fn replay(inp: &Value, db: &Path) -> (Vec<(String, usize)>, Vec<String>, Vec<(String, String)>) {
    let t0 = inp["t0"].as_f64().unwrap();
    Scanner::new("ethusdt", db).init(t0).unwrap();
    {
        let con = Connection::open(db).unwrap();
        let mut st = con.prepare("INSERT INTO microstructure_1m (ts, nif_whale, oi_delta_pct, shadow_absorption_score, shadow_toxicity_score, shadow_queue_bias, eai, mark_price)
               VALUES (make_timestamptz(?),?,?,?,?,?,?,?)").unwrap();
        for r in inp["seeds"].as_array().unwrap() {
            st.execute(params![r[0].as_i64(), r[1].as_f64(), r[2].as_f64(), r[3].as_f64(), r[4].as_f64(), r[5].as_i64(), r[6].as_f64(), r[7].as_f64()]).unwrap();
        }
    }
    let syms = ["ethusdt", "btcusdt"];
    let mut sc: Vec<Scanner> = syms.iter().map(|s| Scanner::new(s, db)).collect();
    let boot: Vec<(String, usize)> = sc.iter_mut().map(|c| (c.symbol.clone(), c.init(t0).unwrap())).collect();
    let (mut sent, mut sigs) = (vec![], vec![]);
    let mut nxt = (t0 / 10.0).floor() * 10.0 + 10.0 + 0.05;
    for e in inp["events"].as_array().unwrap() {
        let (t, kind, sym, body) = (e[0].as_f64().unwrap(), e[1].as_str().unwrap(), e[2].as_str().unwrap(), e[3].as_str().unwrap_or(""));
        while nxt <= t {
            for c in sc.iter_mut() {
                let s = c.scan(nxt);
                sigs.push((c.symbol.clone(), rtcollect::py_dumps_default(&Value::Object(s))));
            }
            nxt = ((nxt / 10.0).floor() + 1.0) * 10.0 + 0.05;
        }
        let c = sc.iter_mut().find(|c| c.symbol == sym).unwrap();
        match kind {
            "D" => c.on_depth(t, body),
            "T" => c.on_trade(t, body),
            "P" => {
                let resp: Value = serde_json::from_str(body).unwrap();
                let mut get = |u: &str| {
                    sent.push(u.to_string());
                    let k = if u.contains("openInterest") { "oi" } else if u.contains("premiumIndex") { "fund" } else { "agg" };
                    match &resp[k] {
                        Value::String(_) => Err("JSON 아님".to_string()),
                        v => Ok(v.clone()),
                    }
                };
                c.poll(&mut get, &mut || t);
            }
            "do" => c.depth_connected = true,
            "de" => c.depth_connected = false,
            "to" => c.trade_connected = true,
            _ => {}
        }
    }
    (boot, sent, sigs)
}

type Sc = std::rc::Rc<std::cell::RefCell<Scanner>>;

/// 호가·체결 웹소켓 두 줄기(:279-352). 호가 = 정상 종료면 곧바로 다시 · 체결 = 35초 수신 시한, 어떤 끊김이든 쉬고 다시.
pub async fn run_streams(c: Sc, depth_url: String, trade_url: String, wall: fn() -> f64) {
    use crate::market::{ws_loop, Stream};
    let s = |c: &Sc| c.clone();
    let (a, b, d) = (s(&c), s(&c), s(&c));
    let depth = ws_loop(Stream { on_connect: Box::new(move |_| a.borrow_mut().depth_connected = true), on_text: Box::new(move |t, x| b.borrow_mut().on_depth(t, x)),
                                 on_error: Box::new(move || d.borrow_mut().depth_connected = false), recv: None, close_ok_again: true }, depth_url, wall);
    let (a, b, d) = (s(&c), s(&c), s(&c));
    let trade = ws_loop(Stream { on_connect: Box::new(move |_| a.borrow_mut().trade_connected = true), on_text: Box::new(move |t, x| b.borrow_mut().on_trade(t, x)),
                                 on_error: Box::new(move || d.borrow_mut().trade_connected = false), recv: Some(std::time::Duration::from_secs(35)), close_ok_again: false },
                        trade_url, wall);
    tokio::join!(depth, trade);
}

/// 폴링(:394-447): 한 바퀴 뒤 max(3, MS_POLL_INTERVAL_SEC) 초. get = 막는 HTTP(시한 5초 -- 바퀴마다 몇십 ms 라 웹소켓 수신을 잠깐 미룰 뿐 잃지 않는다).
/// ponytail: 원본은 비동기 HTTP -- 응답이 느려지면 스레드로 옮길 것.
pub async fn run_poll(c: Sc, mut get: Box<dyn FnMut(&str) -> Result<Value, String>>, wall: fn() -> f64) {
    loop {
        let iv = c.borrow().poll_interval.max(3.0);
        c.borrow_mut().poll(&mut *get, &mut || wall());
        tokio::time::sleep(std::time::Duration::from_secs_f64(iv)).await;
    }
}

/// 스캔(:829-856): 10초 경계 + 0.05초.
pub async fn run_scan(c: Sc, wall: fn() -> f64) {
    loop {
        let now = wall();
        tokio::time::sleep(std::time::Duration::from_secs_f64(10.0 - now.rem_euclid(10.0) + 0.05)).await;
        c.borrow_mut().scan(wall());
    }
}

#[cfg(test)]
mod tests {
    /// numpy 2.4 np.percentile(a, 95) 값(파이썬으로 확인) -- t ≥ 0.5 면 위쪽에서 뺀다(아래쪽에서 더하면 4096.2832499999995).
    #[test]
    fn percentile_matches_numpy_lerp() {
        assert_eq!(super::np_percentile(&[1466.604, 389.6, 68.222, 4972.843, 85.035, 61.984], 95.0), 4096.28325);
        assert_eq!(super::np_percentile(&[7.0], 95.0), 7.0);
    }
}
