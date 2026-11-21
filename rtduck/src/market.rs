//! market_data_worker(`scripts/duckdb_persist_worker.py` 서버 형태) 중 **화면·감시가 읽는 몫**: 
//!   ① 청산 요격기(`tail_risk_interceptor.py` -- 바이낸스 @forceOrder 웹소켓만 · REST 없음) → `tail_risk.duckdb::tail_risk_1m`(매 분) + `liq_burst_state.json`(사건마다·10초마다)
//!   ② `dashboard_state.json`(10초마다: 장 세션 세 칸 + updated_at -- 화면이 «바뀔 때» SSE 로 render() · ops_watchdog check_dashboard 신선도)
//! 장 달력 = 원본이 쓰는 pandas_market_calendars(JPX·LSE·NYSE) schedule 을 떠 둔 표 `sessions.json`(tools/gen_sessions.py) -- 범위 밖이면 원본 예외 길(UTC 시각 규칙).
//! 미시구조(microstructure_scanner.py · 웹소켓 둘 + REST 폴링)는 아직 안 옮겼다(rtcollect/REMAINING.md).
use duckdb::{params, Connection};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub const FORCE_ORDER_WS: &str = "wss://fstream.binance.com/market/ws/{symbol}@forceOrder";
pub const WINDOW: usize = 30;
pub const EVENTS_MAX: usize = 10_000;

// ── 시각 글자 ────────────────────────────────────────────────────────────────

/// 파이썬 datetime.fromtimestamp 의 (초, 마이크로초): 소수부를 마이크로초로 짝수 반올림.
fn sec_us(t: f64) -> (i64, i64) {
    let (mut s, frac) = (t.floor() as i64, t - t.floor());
    let mut us = (frac * 1e6).round_ties_even() as i64;
    if us >= 1_000_000 {
        s += 1;
        us -= 1_000_000;
    }
    (s, us)
}

/// isoformat: `YYYY-MM-DDTHH:MM:SS[.ffffff]` + 꼬리(마이크로초 0 이면 소수부 없음 -- 파이썬 그대로).
fn iso(s: i64, us: i64, tail: &str) -> String {
    let k = rtcollect::depth::hour_key(s * 1000).unwrap_or_default();
    let base = format!("{k}:{:02}:{:02}", s.rem_euclid(3600) / 60, s.rem_euclid(60));
    if us == 0 { format!("{base}{tail}") } else { format!("{base}.{us:06}{tail}") }
}

/// `datetime.utcnow().isoformat() + "Z"`.
pub fn utc_iso_z(t: f64) -> String {
    let (s, us) = sec_us(t);
    iso(s, us, "Z")
}

/// `pd.Timestamp.now(tz="UTC").tz_convert(KST).isoformat()`.
pub fn kst_iso(t: f64) -> String {
    let (s, us) = sec_us(t);
    iso(s + 9 * 3600, us, "+09:00")
}

// ── 장 세션 ──────────────────────────────────────────────────────────────────

type Sessions = HashMap<String, HashMap<String, Vec<i64>>>;

fn sessions() -> &'static Sessions {
    static S: OnceLock<Sessions> = OnceLock::new();
    S.get_or_init(|| {
        let v: Value = serde_json::from_str(include_str!("sessions.json")).unwrap();
        v.as_object().unwrap().iter().filter(|(k, _)| !k.starts_with('_'))
            .map(|(k, d)| (k.clone(), d.as_object().unwrap().iter().map(|(day, x)| (day.clone(), x.as_array().unwrap().iter().map(|m| m.as_i64().unwrap()).collect())).collect()))
            .collect()
    })
}

/// `session_flags`(duckdb_persist_worker.py:56-77): (asia=JPX · europe=LSE · us=NYSE). 그 UTC 날짜의 장 -- 분으로 내린 시각이 개장 ≤ t ≤ 마감이고 점심(시작 ≤ t < 끝)이 아니면 1.0.
/// 표 범위(2024~2035) 밖이면 원본 except 길과 같은 UTC 시각 규칙.
pub fn session_flags(t: f64) -> [f64; 3] {
    let s = t.floor() as i64;
    let day = rtcollect::depth::hour_key(s * 1000).unwrap_or_default()[..10].to_string();
    let (d0, t_min) = (s.div_euclid(86_400) * 86_400, s.div_euclid(60) * 60);
    let tab = sessions();
    if !("2024-01-01"..="2035-12-31").contains(&day.as_str()) {
        let hour = (s.rem_euclid(86_400) / 3600) as f64 + (s.rem_euclid(3600) / 60) as f64 / 60.0; // ts.hour + ts.minute / 60.0
        let f = |a: f64, b: f64| if a <= hour && hour < b { 1.0 } else { 0.0 };
        return [f(0.0, 8.0), f(8.0, 16.0), f(14.5, 21.0)];
    }
    ["JPX", "LSE", "NYSE"].map(|c| match tab[c].get(&day) {
        Some(v) => {
            let at = |k: usize| d0 + v[k] * 60;
            let mut on = at(0) <= t_min && t_min <= at(1);
            if v.len() == 4 && at(2) <= t_min && t_min < at(3) {
                on = false;
            }
            if on { 1.0 } else { 0.0 }
        }
        None => 0.0,
    })
}

/// `write_dashboard_state` 의 글자(json.dumps 기본 모양).
pub fn dashboard_state_text(t: f64) -> String {
    let f = session_flags(t);
    rtcollect::py_dumps_default(&json!({"session": {"session_asia": f[0], "session_europe": f[1], "session_us": f[2]},
                                        "updated_at": kst_iso(t), "source": "duckdb_persist_worker"}))
}

/// tmp 에 쓰고 바꿔치기(원본 `with_suffix(".json.tmp")` → replace).
pub fn write_atomic(path: &Path, text: &str) -> std::io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)
}

// ── numpy ────────────────────────────────────────────────────────────────────

fn np_std(a: &[f64]) -> f64 {
    let m = rtcollect::ob5m::np_sum(a) / a.len() as f64;
    let sq: Vec<f64> = a.iter().map(|x| (x - m) * (x - m)).collect();
    (rtcollect::ob5m::np_sum(&sq) / a.len() as f64).sqrt()
}

fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi) // np.clip(NaN) = NaN -- f64::max/min 은 NaN 을 버린다: 원본도 NaN 이 나올 자리가 없다(분모 하한)
}

// ── 청산 요격기 ──────────────────────────────────────────────────────────────

fn py_float(v: Option<&Value>, d: f64) -> Result<f64, ()> {
    match v {
        None => Ok(d),
        Some(Value::Number(n)) => n.as_f64().ok_or(()),
        Some(Value::String(s)) => s.trim().parse::<f64>().map_err(|_| ()),
        Some(Value::Bool(b)) => Ok(*b as u8 as f64),
        _ => Err(()),
    }
}

fn py_int(v: Option<&Value>) -> Result<i64, ()> {
    match v {
        None => Ok(0),
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().filter(|f| f.is_finite()).map(|f| f.trunc() as i64)).ok_or(()),
        Some(Value::String(s)) => s.trim().parse::<i64>().map_err(|_| ()),
        Some(Value::Bool(b)) => Ok(*b as i64),
        _ => Err(()),
    }
}

fn py_str(v: Option<&Value>) -> String {
    match v {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) => "None".into(),
        Some(Value::Bool(b)) => if *b { "True" } else { "False" }.into(),
        Some(Value::Number(n)) => n.as_i64().map_or_else(|| rtcollect::py_float_repr(n.as_f64().unwrap_or(f64::NAN)), |i| i.to_string()),
        Some(v) => v.to_string(),
    }
}

/// `TailRiskInterceptor` 상태(:56-124). 시각은 부르는 쪽이 준다(시험 = 가짜 시계).
pub struct TailRisk {
    pub symbol: String,
    pub table: String,
    pub burst_path: PathBuf,
    pub db: PathBuf,
    events: VecDeque<(i64, bool, f64)>, // (ts_ms, SELL?, usd)
    hist_long: VecDeque<f64>,
    hist_short: VecDeque<f64>,
    pub mu_long: f64,
    pub sigma_long: f64,
    pub mu_short: f64,
    pub sigma_short: f64,
    pub warmed: bool,
    hawkes_active: bool,
    crisis: Option<&'static str>,
    last_crisis_ts: f64,
    peak: f64,
    decay: f64,
    shadow: (f64, f64, &'static str),
    z_threshold: f64,
    beta: f64,
    release: f64,
    min_trigger: f64,
    pub ws_connected: bool,
    connected_since: f64,
    last_msg: f64,
    stale_s: f64,
    last_saved_minute: i64,
    /// 시험: 쓴 liq_burst_state 글자를 차례로 모은다.
    pub burst_log: Option<Vec<String>>,
    /// 부트스트랩으로 읽은 행 수.
    pub boot: usize,
}

fn env_f(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

impl TailRisk {
    /// live = data/live 에 해당하는 폴더(JSON · DuckDB 가 그 아래).
    pub fn new(symbol: &str, db: &Path, live: &Path) -> Self {
        let s = symbol.to_lowercase();
        let coin = s.replace("usdt", "");
        let eth = s == "ethusdt";
        Self {
            table: if eth { "tail_risk_1m".into() } else { format!("tail_risk_1m_{coin}") },
            burst_path: live.join(if eth { "liq_burst_state.json".into() } else { format!("liq_burst_state_{coin}.json") }),
            db: db.into(), symbol: s, events: VecDeque::new(), hist_long: VecDeque::new(), hist_short: VecDeque::new(),
            mu_long: 0.0, sigma_long: 1.0, mu_short: 0.0, sigma_short: 1.0, warmed: false, hawkes_active: false, crisis: None,
            last_crisis_ts: 0.0, peak: 0.0, decay: 0.0, shadow: (0.0, 0.0, "normal"),
            z_threshold: env_f("TR_Z_THRESHOLD", 3.5), beta: env_f("TR_HAWKES_BETA", 0.005), release: env_f("TR_HAWKES_RELEASE_RATIO", 0.35),
            min_trigger: env_f("TR_HAWKES_MIN_TRIGGER_USD", if eth { 10_000.0 } else { 0.0 }),
            ws_connected: false, connected_since: 0.0, last_msg: 0.0, stale_s: env_f("TR_LIQ_STALE_THRESHOLD_SEC", 14_400.0), last_saved_minute: -1, burst_log: None, boot: 0,
        }
    }

    fn open(&self) -> duckdb::Result<Connection> {
        // duckdb_connect_retry: 읽는 쪽 잠금이면 0.2초씩 25번
        let mut last = None;
        for _ in 0..25 {
            match Connection::open(&self.db) {
                Ok(c) => return Ok(c),
                Err(e) => {
                    last = Some(e);
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
            }
        }
        Err(last.unwrap())
    }

    /// `_db_init`(:128-177): 표 · 없는 칸 더하기 · 최근 30분 부트스트랩(원본 now() = 그 순간 -- 여기는 부르는 쪽이 준다).
    pub fn init(&mut self, now: f64) -> duckdb::Result<usize> {
        if let Some(d) = self.db.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let con = Connection::open(&self.db)?;
        con.execute_batch(&format!("
            CREATE TABLE IF NOT EXISTS {} (
                ts TIMESTAMPTZ,
                long_usd_1m DOUBLE,
                short_usd_1m DOUBLE,
                mu_long DOUBLE,
                sigma_long DOUBLE,
                mu_short DOUBLE,
                sigma_short DOUBLE,
                shadow_aftershock_prob DOUBLE,
                shadow_decay_half_life DOUBLE,
                shadow_risk_bucket VARCHAR
            )
        ", self.table))?;
        let have: Vec<String> = con.prepare(&format!("PRAGMA table_info('{}')", self.table))?.query_map([], |r| r.get::<_, String>(1))?.collect::<Result<_, _>>()?;
        for (c, ty) in [("ws_connected", "BOOLEAN"), ("ws_stale", "BOOLEAN"), ("ws_age_sec", "DOUBLE"), ("liq_event_count_1m", "INTEGER"),
                        ("valid_liq_stream", "BOOLEAN"), ("schema_version", "INTEGER")] {
            if !have.iter().any(|h| h == c) {
                con.execute_batch(&format!("ALTER TABLE {} ADD COLUMN {c} {ty}", self.table))?;
            }
        }
        // 🔴묶은 DuckDB 에는 ICU 가 없어 TIMESTAMPTZ − INTERVAL 이 Binder Error -- epoch µs 로 같은 비교
        let rows: Vec<(Option<f64>, Option<f64>)> = con.prepare(&format!("
                SELECT long_usd_1m, short_usd_1m FROM {}
                WHERE epoch_us(ts) >= ?
                ORDER BY ts ASC
            ", self.table)).and_then(|mut st| st.query_map([(now * 1e6).round() as i64 - WINDOW as i64 * 60_000_000], |r| Ok((r.get(0)?, r.get(1)?)))?.collect()).unwrap_or_default();
        for (l, s) in &rows {
            self.push_hist(l.unwrap_or(0.0), s.unwrap_or(0.0));
        }
        if self.hist_long.len() >= 15 {
            self.warmed = true;
        }
        self.recalc();
        Ok(rows.len())
    }

    fn push_hist(&mut self, l: f64, s: f64) {
        for (h, v) in [(&mut self.hist_long, l), (&mut self.hist_short, s)] {
            if h.len() == WINDOW {
                h.pop_front();
            }
            h.push_back(v);
        }
    }

    /// `_recalculate_stats`(:241-248): np.mean · np.std + 1e-6.
    fn recalc(&mut self) {
        if !self.hist_long.is_empty() {
            let (l, s): (Vec<f64>, Vec<f64>) = (self.hist_long.iter().copied().collect(), self.hist_short.iter().copied().collect());
            self.mu_long = rtcollect::ob5m::np_mean(&l);
            self.sigma_long = np_std(&l) + 1e-6;
            self.mu_short = rtcollect::ob5m::np_mean(&s);
            self.sigma_short = np_std(&s) + 1e-6;
        }
    }

    /// `_aggregate_1m`(:250-266): 최근 이벤트부터 1분 안(롱 청산 = SELL).
    fn agg(&self, now: f64) -> (f64, f64) {
        let cut = (now * 1000.0) as i64 - 60_000;
        let (mut l, mut s) = (0.0, 0.0);
        for (ts, sell, usd) in self.events.iter().rev() {
            if *ts < cut {
                break;
            }
            if *sell { l += usd } else { s += usd }
        }
        (l, s)
    }

    fn count(&self, now: f64) -> i64 {
        let cut = (now * 1000.0) as i64 - 60_000;
        self.events.iter().rev().take_while(|e| e.0 >= cut).filter(|e| e.2 > 0.0).count() as i64
    }

    fn z(&self, l: f64, s: f64) -> (f64, f64) {
        ((l - self.mu_long) / self.sigma_long.max(1.0), (s - self.mu_short) / self.sigma_short.max(1.0))
    }

    /// `_compute_shadow_aftershock`(:279-302).
    fn shadow_of(&self, l: f64, s: f64) -> (f64, f64, &'static str) {
        let (zl, zs) = self.z(l, s);
        let zp = zl.max(zs).max(0.0);
        let imb = (l - s).abs() / (l + s + 1e-8);
        let active = clip(self.decay, 0.0, 1.0);
        let prob = clip(0.45 * (zp / self.z_threshold.max(1e-8)) + 0.35 * imb + 0.20 * active, 0.0, 1.0);
        let hl = 2f64.ln() / (self.beta * 60.0).max(1e-6);
        (prob, hl, if prob >= 0.75 { "high" } else if prob >= 0.45 { "watch" } else { "normal" })
    }

    /// `_update_hawkes_state`(:304-334).
    fn hawkes(&mut self, l: f64, s: f64, now: f64) {
        let (zl, zs) = self.z(l, s);
        let zp = zl.max(zs).max(0.0);
        let (crisis, peak_usd) = if zl >= zs { ("LONG_CRISIS", l) } else { ("SHORT_CRISIS", s) };
        if zp >= self.z_threshold && peak_usd >= self.min_trigger {
            self.peak = if !self.hawkes_active || self.crisis != Some(crisis) { zp } else { self.peak.max(zp) };
            (self.hawkes_active, self.crisis, self.last_crisis_ts, self.decay) = (true, Some(crisis), now, 1.0);
            return;
        }
        if !self.hawkes_active {
            self.decay = 0.0;
            return;
        }
        let decayed = self.peak * (-self.beta * (now - self.last_crisis_ts).max(0.0)).exp();
        self.decay = clip(decayed / self.z_threshold.max(1e-8), 0.0, 1.0);
        if decayed <= self.z_threshold * self.release {
            (self.hawkes_active, self.crisis, self.peak) = (false, None, 0.0);
        }
    }

    /// `_stream_health`(:179-200): (stale, age, valid).
    fn health(&self, now: f64) -> (bool, Option<f64>, bool) {
        let age = if self.last_msg > 0.0 { Some(now - self.last_msg) } else if self.connected_since > 0.0 { Some(now - self.connected_since) } else { None };
        let stale = !self.ws_connected || age.is_none_or(|a| a > self.stale_s);
        (stale, age, self.ws_connected && !stale)
    }

    /// `_write_liq_burst_state`(:336-374) 글자.
    pub fn burst_text(&self, l: f64, s: f64, now: f64) -> String {
        let (zl, zs) = self.z(l, s);
        let (_, age, valid) = self.health(now);
        rtcollect::py_dumps_default(&json!({
            "updated_at": utc_iso_z(now), "hawkes_active": self.hawkes_active, "hawkes_decay_level": self.decay,
            "crisis_type": self.crisis.unwrap_or(""), "z_long": zl, "z_short": zs, "long_usd_1m": l, "short_usd_1m": s,
            "liq_event_count_1m": self.count(now), "ws_connected": self.ws_connected, "ws_age_sec": age, "valid_liq_stream": valid,
        }))
    }

    fn write_burst(&mut self, l: f64, s: f64, now: f64) {
        let t = self.burst_text(l, s, now);
        if let Err(e) = write_atomic(&self.burst_path, &t) {
            eprintln!("liq_burst_state write failed: {e}"); // 원본은 debug 로그만
        }
        if let Some(log) = self.burst_log.as_mut() {
            log.push(t);
        }
    }

    /// 접속됨(:385-389).
    pub fn on_connect(&mut self, now: f64) {
        (self.ws_connected, self.connected_since) = (true, now);
    }

    /// 예외로 끊김(:418-423). 서버가 정상 종료(1000·1001)하면 원본은 ws_connected 를 그대로 두고 곧바로 다시 붙는다 -- 부르지 않는다.
    pub fn on_error(&mut self) {
        self.ws_connected = false;
    }

    /// 메시지 하나(:393-417) -- 못 읽으면 조용히 버린다(원본 except: pass).
    pub fn on_text(&mut self, now: f64, raw: &str) {
        let Ok(Value::Object(m)) = serde_json::from_str::<Value>(raw) else { return };
        let empty = serde_json::Map::new();
        let o = match m.get("o") {
            None => &empty,
            Some(Value::Object(o)) => o,
            Some(_) => return,
        };
        let side = py_str(o.get("S"));
        let (Ok(q), Ok(p), Ok(ts)) = (py_float(o.get("z"), 0.0), py_float(o.get("ap"), 0.0), py_int(m.get("E"))) else { return };
        let usd = q * p;
        self.last_msg = now;
        if usd > 0.0 && (side == "BUY" || side == "SELL") {
            if self.events.len() == EVENTS_MAX {
                self.events.pop_front();
            }
            self.events.push_back((ts, side == "SELL", usd));
            let (l, s) = self.agg(now);
            self.hawkes(l, s, now);
            self.write_burst(l, s, now);
        }
    }

    /// `_agg_loop` 한 바퀴(:437-462) -- 10초 경계 + 0.05초에 부른다. 반환 = 이번에 DuckDB 에 쓴 분(UTC epoch 초).
    pub fn tick(&mut self, now: f64) -> Option<i64> {
        let (l, s) = self.agg(now);
        let n = self.count(now);
        self.hawkes(l, s, now);
        self.write_burst(l, s, now);
        self.shadow = self.shadow_of(l, s);
        // 원본은 지역 시각 datetime.now() 의 분·초 -- 시간대가 분 단위로 어긋나지 않으면 UTC 와 같다(ponytail: 30분 시간대면 분 판정이 같고 초도 같다)
        let sec = now.floor() as i64;
        let (minute, second) = (sec.div_euclid(60).rem_euclid(60), sec.rem_euclid(60));
        if minute != self.last_saved_minute && second < 15 {
            self.last_saved_minute = minute;
            self.push_hist(l, s);
            if self.hist_long.len() >= 15 {
                self.warmed = true;
            }
            self.recalc();
            let bucket = sec.div_euclid(60) * 60 - 60; // (지금 - 1분) 의 분 시작
            if let Err(e) = self.insert(bucket, l, s, n, now) {
                eprintln!("🚨 사후 요격기 DB 저장 에러: {e}");
            }
            return Some(bucket);
        }
        None
    }

    fn insert(&self, bucket: i64, l: f64, s: f64, n: i64, now: f64) -> duckdb::Result<()> {
        let (stale, age, valid) = self.health(now);
        let con = self.open()?;
        con.execute(&format!("
                INSERT INTO {} (
                    ts, long_usd_1m, short_usd_1m, mu_long, sigma_long, mu_short, sigma_short,
                    shadow_aftershock_prob, shadow_decay_half_life, shadow_risk_bucket,
                    ws_connected, ws_stale, ws_age_sec, liq_event_count_1m, valid_liq_stream, schema_version
                ) VALUES (make_timestamptz(?),?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ", self.table), params![bucket * 1_000_000, l, s, self.mu_long, self.sigma_long, self.mu_short, self.sigma_short,
                                       self.shadow.0, self.shadow.1, self.shadow.2, self.ws_connected, stale, age, n, valid, 3i64])?;
        Ok(())
    }
}

/// 정답지 입력 먹이기. 사건 = [벽시계 초, 종류, 본문]: 'o' 접속 · 't' 메시지 · 'e' 예외로 끊김 · 'n' 정상 종료(아무것도 안 바뀜) · 'd' dashboard_state 쓰기 · 'z' 틱 루프가 본문 초만큼 멎음.
/// 10초 경계 + 0.05초 틱은 사건 사이에 원본 _agg_loop 처럼 끼운다(그 시각 ≤ 사건 시각이면 먼저). 반환 = (liq_burst 글자들, dashboard 글자들).
pub fn replay(inp: &Value, db: &Path, live: &Path) -> (Vec<String>, Vec<String>) {
    let t0 = inp["t0"].as_f64().unwrap();
    TailRisk::new(inp["symbol"].as_str().unwrap(), db, live).init(t0).unwrap(); // 빈 표 → 씨앗 → 새 요격기의 부트스트랩
    {
        let con = Connection::open(db).unwrap();
        for r in inp["seeds"].as_array().into_iter().flatten() {
            con.execute("INSERT INTO tail_risk_1m (ts, long_usd_1m, short_usd_1m) VALUES (make_timestamptz(?),?,?)", params![r[0].as_i64(), r[1].as_f64(), r[2].as_f64()]).unwrap();
        }
    }
    let mut tr = TailRisk::new(inp["symbol"].as_str().unwrap(), db, live);
    tr.burst_log = Some(vec![]);
    tr.boot = tr.init(t0).unwrap();
    let mut dash = vec![];
    let mut next_tick = (inp["t0"].as_f64().unwrap() / 10.0).floor() * 10.0 + 10.0 + 0.05;
    for e in inp["events"].as_array().unwrap() {
        let (t, kind, body) = (e[0].as_f64().unwrap(), e[1].as_str().unwrap(), e[2].as_str().unwrap_or(""));
        while next_tick <= t {
            tr.tick(next_tick);
            next_tick = ((next_tick / 10.0).floor() + 1.0) * 10.0 + 0.05; // sleep(10 - now%10 + 0.05)
        }
        match kind {
            "z" => {
                let wake = t + body.parse::<f64>().unwrap(); // 루프 멎음 -- 그동안 틱 없음 · 깨면 다음 10초 경계 + 0.05초
                next_tick = ((wake / 10.0).floor() + 1.0) * 10.0 + 0.05;
            }
            "o" => tr.on_connect(t),
            "t" => tr.on_text(t, body),
            "e" => tr.on_error(),
            "d" => {
                let s = dashboard_state_text(t);
                write_atomic(&live.join("dashboard_state.json"), &s).unwrap();
                dash.push(s);
            }
            _ => {}
        }
    }
    for e in inp["probes"].as_array().unwrap() {
        let s = dashboard_state_text(e[0].as_f64().unwrap());
        write_atomic(&live.join("dashboard_state.json"), &s).unwrap();
        dash.push(s);
    }
    (tr.burst_log.take().unwrap(), dash)
}

/// 웹소켓 한 줄기(요격기 `_ws_loop` :378-423 · 미시구조 `_depth_loop`·`_trade_loop` :279-352 의 같은 뼈대): 접속(시한 10초 -- websockets open_timeout 기본)
/// → 메시지마다 on_text · 20초 핑·10초 퐁 시한 · recv = 수신 시한(체결 35초 -- 넘으면 끊고 다시) ·
/// 서버 정상 종료(1000·1001)면 close_ok_again 일 때 곧바로 다시(상태 그대로 -- `async for` 가 조용히 끝난다) · 아니면 on_error 후 3초부터 1.5배씩 60초까지 쉬고 다시.
pub struct Stream {
    pub on_connect: Box<dyn FnMut(f64)>,
    pub on_text: Box<dyn FnMut(f64, &str)>,
    pub on_error: Box<dyn FnMut()>,
    pub recv: Option<std::time::Duration>,
    pub close_ok_again: bool,
}

pub async fn ws_loop(mut s: Stream, url: String, wall: fn() -> f64) {
    use futures_util::{SinkExt, StreamExt};
    use tokio::time::{sleep, timeout, Duration, Instant};
    use tokio_tungstenite::tungstenite::{protocol::frame::coding::CloseCode, Message};
    let mut delay = 3.0f64;
    loop {
        let err: Option<String> = match timeout(Duration::from_secs(10), tokio_tungstenite::connect_async(url.as_str())).await {
            Ok(Ok((ws, _))) => {
                delay = 3.0;
                (s.on_connect)(wall());
                let (mut tx, mut rx) = ws.split();
                let mut ping_at = Instant::now() + Duration::from_secs(20);
                let mut pong_by: Option<Instant> = None;
                let mut recv_by = s.recv.map(|d| Instant::now() + d);
                loop {
                    let deadline = [Some(pong_by.unwrap_or(ping_at)), recv_by].into_iter().flatten().min().unwrap();
                    match timeout(deadline.saturating_duration_since(Instant::now()), rx.next()).await {
                        Err(_) if recv_by.is_some_and(|r| Instant::now() >= r) => break Some("TimeoutError: recv 시한".into()),
                        Err(_) if pong_by.is_some() => break Some("ConnectionClosedError: keepalive ping timeout".into()),
                        Err(_) => {
                            if let Err(e) = tx.send(Message::Ping(Default::default())).await {
                                break Some(format!("ConnectionClosedError: {e}"));
                            }
                            pong_by = Some(Instant::now() + Duration::from_secs(10));
                        }
                        Ok(Some(Ok(Message::Text(t)))) => {
                            recv_by = s.recv.map(|d| Instant::now() + d);
                            (s.on_text)(wall(), t.as_str())
                        }
                        Ok(Some(Ok(Message::Pong(_)))) => {
                            pong_by = None;
                            ping_at = Instant::now() + Duration::from_secs(20);
                        }
                        Ok(Some(Ok(Message::Close(f)))) => {
                            let ok = f.as_ref().is_some_and(|f| matches!(f.code, CloseCode::Normal | CloseCode::Away));
                            break if ok && s.close_ok_again { None } else { Some(format!("ConnectionClosed: {f:?}")) };
                        }
                        Ok(Some(Ok(_))) => {}
                        Ok(Some(Err(e))) => break Some(format!("ConnectionClosedError: {e}")),
                        Ok(None) => break Some("ConnectionClosedError: 끊김(닫기 프레임 없음)".into()),
                    }
                }
            }
            Ok(Err(e)) => Some(format!("{e}")),
            Err(_) => Some("TimeoutError".into()),
        };
        if let Some(e) = err {
            (s.on_error)();
            eprintln!("WS 재연결 ({delay:.0}s) {url}: {e}");
            sleep(Duration::from_secs_f64(delay)).await;
            delay = (delay * 1.5).min(60.0);
        }
    }
}

/// 요격기 줄기(@forceOrder -- 정상 종료면 곧바로 다시 · 수신 시한 없음).
pub async fn run_ws(tr: std::rc::Rc<std::cell::RefCell<TailRisk>>, url: String, wall: fn() -> f64) {
    let (a, b, c) = (tr.clone(), tr.clone(), tr);
    ws_loop(Stream { on_connect: Box::new(move |t| a.borrow_mut().on_connect(t)), on_text: Box::new(move |t, x| b.borrow_mut().on_text(t, x)),
                     on_error: Box::new(move || c.borrow_mut().on_error()), recv: None, close_ok_again: true }, url, wall).await
}

/// `_agg_loop`(:425-465): 10초 경계 + 0.05초마다 tick.
pub async fn run_ticks(tr: std::rc::Rc<std::cell::RefCell<TailRisk>>, wall: fn() -> f64) {
    loop {
        let now = wall();
        tokio::time::sleep(std::time::Duration::from_secs_f64(10.0 - now.rem_euclid(10.0) + 0.05)).await;
        tr.borrow_mut().tick(wall());
    }
}
