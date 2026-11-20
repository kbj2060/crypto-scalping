//! 바이낸스 맥락 수집기(`scripts/live_binance_ctx_collector_20261001.py`, 줄 번호는 그 파일) → binance_ctx.sqlite.
//! · 마크 1초: WS `/market/ws/<sym>@markPrice@1s`(🔴`/ws/` 는 연결돼도 이벤트 0) → mark_price_1s(PK ts_ms·symbol, INSERT OR IGNORE)
//! · 강제청산: WS `/market/ws/!forceOrder@arr` 전 종목에서 COIN_CONFIG 종목(+ 느린 OI 종목)만 → liquidations. 수량 = z(누적 체결량, 없으면 q),
//!   가격 = ap(없으면 p) · 롱 청산 = 시장에 SELL.
//! · OI: REST `/fapi/v1/openInterest` 폴링(0.5초 · 느린 종목 5초) → oi_1s(stamp `time` 의 집합으로 중복을 거른다 -- 도착 순서가 뒤바뀐다
//!   2026-09-19 실측 3.5%). 🔴기본 꺼짐(main.rs `RTCOLLECT_REST=1`): 그림자 동안 켜면 같은 IP 에서 OI 호출이 두 배다(분당 약 540 -- SHADOW.md).
//! 보류 행은 표마다 하나(원본 `Pending`), 10초마다 한 트랜잭션. 못 쓰면 앞에 되돌려 붙이고 20만 행을 넘으면 말하고 버린다.
//! 🔧의도한 차이(SHADOW.md §11): ① 깨진 JSON 한 프레임은 건너뛴다(원본은 그 연결을 끊는다) ② OI 는 값을 읽은 뒤에 seen 에 넣는다(원본은 먼저 넣어
//!   못 읽은 시각이 영구히 버려진다) ③ 재연결 공백을 `gaps` 표에 적는다(원본에 없다 -- 표를 «더하기»만 해서 읽는 쪽은 그대로) ④ SIGTERM 때 보류분을 쓴다.
use crate::rest::{ban_left, ban_note, Clock, Fail};
use crate::store::{in_tx, open_hot};
use crate::{ms, py_float, py_int, py_str, py_truthy, Reconnect};
use rusqlite::{params, Connection};
use serde_json::{Map, Value};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub const WS_BASE: &str = "wss://fstream.binance.com/market/ws/";
pub const OI_URL: &str = "https://fapi.binance.com/fapi/v1/openInterest";
/// 2026-11-01 0.25 -> 0.5: 네 코인 OI 폴링이 분당 960 가중치로 IP 한도(2,400)의 4할을 먹었다(:34).
pub const OI_POLL_S: f64 = 0.5;
/// 2026-11-01 ZEC·SNDK -- 0.25초면 분당 240 가중치씩 더해 IP 한도를 먹는다. 거래소 OI 는 5~10초마다 갱신(:35).
pub const OI_SLOW_POLL_S: f64 = 5.0;
pub const OI_SYMBOLS: &str = "ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT";
pub const OI_SLOW_SYMBOLS: &str = "ZECUSDT,SNDKUSDT,SOXLUSDT,SKHYNIXUSDT,MUUSDT";
/// 2026-10-31 BTC 도(사용자 «BTC 도 ETH 와 같은 대시보드») -- OI 1초 레인·마크
pub const MARK_SYMBOLS: &str = "ethusdt,btcusdt,solusdt,xrpusdt,zecusdt,sndkusdt,soxlusdt,skhynixusdt,muusdt";
/// scripts/coin_config.py COIN_CONFIG 의 binance_symbol(2026-11-21 import 해서 확인). 원본은 여기에 느린 OI 종목을 더한다(TradFi 청산 확인용).
pub const COIN_CONFIG_SYMBOLS: [&str; 9] = ["ETHUSDT", "BTCUSDT", "SOLUSDT", "XRPUSDT", "HYPEUSDT", "SNDKUSDT", "SOXLUSDT", "SKHYNIXUSDT", "MUUSDT"];
pub const FLUSH_S: f64 = 10.0;
pub const PENDING_CAP: usize = 200_000;
/// rest ClientSession(timeout=ClientTimeout(total=10)) (:176)
pub const REST_TIMEOUT: Duration = Duration::from_secs(10);

const DDL: [&str; 7] = [
    "CREATE TABLE IF NOT EXISTS oi_1s(ts_ms INTEGER, symbol TEXT, open_interest REAL, PRIMARY KEY (ts_ms, symbol)) WITHOUT ROWID",
    "CREATE TABLE IF NOT EXISTS mark_price_1s(ts_ms INTEGER, symbol TEXT, mark REAL, index_px REAL, funding_rate REAL, next_funding_ms INTEGER, PRIMARY KEY (ts_ms, symbol)) WITHOUT ROWID",
    "CREATE TABLE IF NOT EXISTS liquidations(ts_ms INTEGER, side TEXT, qty REAL, price REAL, usd REAL, symbol TEXT)",
    // 🔧더한 표: 스트림(마크 = 소문자 종목 · 강제청산 = NULL = 전 종목)마다 끊긴 구간. 1.5초 미만은 안 적는다(다른 수집기와 같다).
    "CREATE TABLE IF NOT EXISTS gaps(symbol TEXT, from_ms INTEGER, to_ms INTEGER, reason TEXT)",
    "CREATE INDEX IF NOT EXISTS oi_1s_sym ON oi_1s(symbol, ts_ms)",
    "CREATE INDEX IF NOT EXISTS mark_sym ON mark_price_1s(symbol, ts_ms)",
    "CREATE INDEX IF NOT EXISTS liq_sym ON liquidations(symbol, ts_ms)",
];

pub fn open_db(path: &Path) -> rusqlite::Result<Connection> {
    open_hot(path, &DDL)
}

type Oi = (i64, String, f64);
type Mark = (i64, String, f64, f64, f64, i64);
type Liq = (i64, &'static str, f64, f64, f64, String);
type Gap = (Option<String>, i64, i64, &'static str);

#[derive(Default)]
pub struct Pending {
    pub oi: Vec<Oi>,
    pub mark: Vec<Mark>,
    pub liq: Vec<Liq>,
    pub gaps: Vec<Gap>,
}

impl Pending {
    pub fn len(&self) -> usize {
        self.oi.len() + self.mark.len() + self.liq.len() + self.gaps.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub type Shared = Arc<Mutex<Pending>>;

/// `Pending.flush`(:104-118). 반환 = 쓴 행 수.
pub fn flush(con: &Connection, p: &Shared) -> usize {
    let batch = std::mem::take(&mut *p.lock().unwrap());
    if batch.is_empty() {
        return 0;
    }
    let r = in_tx(con, || {
        for r in &batch.oi {
            con.prepare_cached("INSERT OR IGNORE INTO oi_1s VALUES (?,?,?)")?.execute(params![r.0, r.1, r.2])?;
        }
        for r in &batch.mark {
            con.prepare_cached("INSERT OR IGNORE INTO mark_price_1s VALUES (?,?,?,?,?,?)")?.execute(params![r.0, r.1, r.2, r.3, r.4, r.5])?;
        }
        for r in &batch.liq {
            con.prepare_cached("INSERT INTO liquidations VALUES (?,?,?,?,?,?)")?.execute(params![r.0, r.1, r.2, r.3, r.4, r.5])?;
        }
        for r in &batch.gaps {
            con.prepare_cached("INSERT INTO gaps VALUES (?,?,?,?)")?.execute(params![r.0, r.1, r.2, r.3])?;
        }
        Ok(())
    });
    let n = batch.len();
    match r {
        Ok(()) => n,
        Err(e) => {
            let mut g = p.lock().unwrap();
            let (mut oi, mut mark, mut liq, mut gaps) = (batch.oi, batch.mark, batch.liq, batch.gaps);
            oi.append(&mut g.oi);
            mark.append(&mut g.mark);
            liq.append(&mut g.liq);
            gaps.append(&mut g.gaps);
            *g = Pending { oi, mark, liq, gaps };
            let n = g.len();
            if n > PENDING_CAP {
                *g = Pending::default();
                eprintln!("⚠️쓰기가 계속 막혀 {n}행 버림: {e}");
            } else {
                eprintln!("쓰기 보류 {n}행, 다음에 재시도: {e}");
            }
            0
        }
    }
}

/// `json.loads(msg.data) or {}` → 사전. 거짓 값 = {} · 참인 비사전 = `.get` AttributeError(재연결). 깨진 JSON 은 부르는 쪽이 먼저 거른다(`Feed`).
fn obj(text: &str) -> Result<Map<String, Value>, Reconnect> {
    let v: Value = serde_json::from_str(text).map_err(|e| Reconnect(format!("JSONDecodeError: {e}")))?;
    match v {
        Value::Object(o) => Ok(o),
        v if !py_truthy(&v) => Ok(Map::new()),
        _ => Err(Reconnect("AttributeError: .get".into())),
    }
}

fn key<'a>(o: &'a Map<String, Value>, k: &str) -> Result<&'a Value, Reconnect> {
    crate::need(o, k)
}

/// `parse_mark`(:80-83). 마크 이벤트가 아니면 None.
pub fn parse_mark(text: &str) -> Result<Option<Mark>, Reconnect> {
    let o = obj(text)?;
    if o.get("e").and_then(Value::as_str) != Some("markPriceUpdate") {
        return Ok(None);
    }
    let t = match o.get("T").filter(|v| py_truthy(v)) {
        Some(v) => Some(v),
        None => None,
    };
    Ok(Some((
        py_int(key(&o, "E")?)?,
        py_str(Some(key(&o, "s")?), "").to_lowercase(),
        py_float(key(&o, "p")?)?,
        py_float(key(&o, "i")?)?,
        py_float(key(&o, "r")?)?,
        match t {
            Some(v) => py_int(v)?,
            None => 0,
        },
    )))
}

/// `parse_force`(:86-92). 고른 종목 밖이면 None. `wall` = T 가 없을 때 쓰는 벽시계(원본 time.time()).
pub fn parse_force(text: &str, liq_symbols: &HashSet<String>, wall: f64) -> Result<Option<Liq>, Reconnect> {
    let m = obj(text)?;
    let o = match m.get("o").filter(|v| py_truthy(v)) {
        None => return Ok(None), // {} .get("s") = None -- 집합에 없다
        Some(Value::Object(o)) => o,
        Some(_) => return Err(Reconnect("AttributeError: .get".into())),
    };
    let s = match o.get("s") {
        Some(Value::String(s)) if liq_symbols.contains(s) => s.clone(),
        Some(Value::Array(_) | Value::Object(_)) => return Err(Reconnect("TypeError: unhashable".into())),
        _ => return Ok(None),
    };
    // float(a or b or 0.0) -- 참인 첫 값
    let first = |a: &str, b: &str| -> Result<f64, Reconnect> {
        match o.get(a).filter(|v| py_truthy(v)).or_else(|| o.get(b).filter(|v| py_truthy(v))) {
            Some(v) => py_float(v),
            None => Ok(0.0),
        }
    };
    let (qty, price) = (first("z", "q")?, first("ap", "p")?);
    let ts = match o.get("T").filter(|v| py_truthy(v)) {
        Some(v) => py_int(v)?,
        None => ms(wall),
    };
    let side = if o.get("S").and_then(Value::as_str) == Some("SELL") { "long" } else { "short" };
    Ok(Some((ts, side, qty, price, qty * price, s)))
}

/// 고른 청산 종목 = COIN_CONFIG ∪ 느린 OI 종목(:41).
pub fn liq_symbols(slow: &[String]) -> HashSet<String> {
    COIN_CONFIG_SYMBOLS.iter().map(|s| s.to_string()).chain(slow.iter().cloned()).collect()
}

/// 마크·청산 WS 한 연결의 상태 없는 처리: 글자 → 보류(Err = 그 연결을 접는다).
pub enum Feed {
    Mark,
    Force(HashSet<String>),
}

impl Feed {
    pub fn on_text(&self, p: &Shared, wall: f64, text: &str) -> Result<(), Reconnect> {
        // 🔧깨진 JSON 한 프레임 = 건너뜀(원본: ValueError 로 그 연결을 끊는다)
        if serde_json::from_str::<Value>(text).is_err() {
            eprintln!("⚠️깨진 JSON 프레임 건너뜀: {}", text.chars().take(120).collect::<String>());
            return Ok(());
        }
        match self {
            Feed::Mark => {
                if let Some(r) = parse_mark(text)? {
                    p.lock().unwrap().mark.push(r);
                }
            }
            Feed::Force(s) => {
                if let Some(r) = parse_force(text, s, wall)? {
                    p.lock().unwrap().liq.push(r);
                }
            }
        }
        Ok(())
    }
}

/// `poll_oi`(:121-149) 한 종목. 잠은 `clock`(가짜면 두 시계를 같이 민다 -- 원본 정답지의 가짜 asyncio.sleep 과 같은 덧셈 순서).
pub struct OiPoll {
    symbol: String,
    period: f64,
    url: String,
    agent: ureq::Agent,
    clock: Arc<Clock>,
    ban_file: PathBuf,
    seen: HashSet<i64>,
    out: Shared,
    /// 가짜 시계일 때만: (단조, 벽, 경로?쿼리)
    pub sent: Vec<(f64, f64, String)>,
}

impl OiPoll {
    #[allow(clippy::too_many_arguments)]
    pub fn new(symbol: &str, period: f64, url: &str, clock: Arc<Clock>, ban_file: PathBuf, out: Shared, timeout: Duration) -> Self {
        let agent = crate::rest::agent(timeout);
        Self { symbol: symbol.to_uppercase(), period, url: format!("{url}?symbol={}", symbol.to_uppercase()), agent, clock, ban_file,
               seen: HashSet::new(), out, sent: vec![] }
    }

    pub fn run(mut self) {
        loop {
            self.step();
        }
    }

    /// 한 바퀴(잠 포함).
    pub fn step(&mut self) {
        let left = ban_left(&self.ban_file, self.clock.wall());
        if left > 0.0 {
            // 다른 프로세스가 받은 차단도 따른다
            self.clock.sleep((left + 1.0).min(300.0));
            return;
        }
        match self.once() {
            Ok(Some(wait)) => return self.clock.sleep(wait), // 거래소 거절 -- period 잠 없이
            Ok(None) => {}
            Err(e) => {
                eprintln!("oi {} poll failed (will retry): {e:?}", self.symbol);
                self.clock.sleep(1.0);
            }
        }
        self.clock.sleep(self.period);
    }

    /// Ok(Some(초)) = 거래소 거절 뒤 쉴 초 · Ok(None) = 정상 · Err = 원본 `except Exception`.
    fn once(&mut self) -> Result<Option<f64>, Fail> {
        if self.clock.is_fake() {
            let path = self.url.splitn(4, '/').nth(3).map_or(String::new(), |p| format!("/{p}"));
            self.sent.push((self.clock.mono(), self.clock.wall(), path));
        }
        let mut r = match self.agent.get(&self.url).call() {
            Ok(r) => r,
            Err(ureq::Error::Timeout(_)) => return Err(Fail::Other("TimeoutError".into())),
            Err(e) => return Err(Fail::Other(format!("ClientError: {e}"))),
        };
        let h = |n: &str| r.headers().get(n).and_then(|v| v.to_str().ok()).map(str::to_string);
        let (status, ctype, retry_after) = (r.status().as_u16(), h("content-type").unwrap_or_default(), h("retry-after"));
        let body = r.body_mut().read_to_string().map_err(|e| Fail::Other(format!("ClientPayloadError: {e}")))?;
        let data = crate::rest::aio_json(&ctype, &body)?;
        // "time" not in data -- 사전은 열쇠 · 목록은 원소 · 글자는 부분 글자 · 그 밖은 TypeError
        let has_time = match &data {
            Value::Object(o) => o.contains_key("time"),
            Value::Array(a) => a.iter().any(|x| x.as_str() == Some("time")),
            Value::String(s) => s.contains("time"),
            _ => return Err(Fail::Other("TypeError: argument of type is not iterable".into())),
        };
        if !has_time {
            // ponytail: 원본은 json.dumps(data) 를 넘긴다 -- «banned until <13자리>» 만 보므로 원문과 같다(이스케이프된 글자가 끼면 다를 수 있다).
            ban_note(&self.ban_file, status, &body, retry_after.as_deref(), self.clock.wall());
            let Value::Object(o) = &data else { return Err(Fail::Other("AttributeError: .get".into())) };
            let msg = py_str(o.get("msg"), "");
            let wait = match msg.find("banned until ").map(|i| &msg[i + 13..]).map(|d| d.bytes().take_while(u8::is_ascii_digit).count()) {
                Some(n) if n > 0 => {
                    let d = &msg[msg.find("banned until ").unwrap() + 13..][..n];
                    (d.parse::<f64>().unwrap_or(f64::INFINITY) / 1000.0 - self.clock.wall() + 1.0).max(5.0)
                }
                _ => 30.0,
            };
            eprintln!("oi {}: 거래소 거절 {status} {:?} -- {wait:.0}초 쉰다", self.symbol, o.get("code"));
            return Ok(Some(wait));
        }
        let obj = data.as_object().ok_or_else(|| Fail::Other("TypeError: indices".into()))?;
        let ts = py_int(&obj["time"]).map_err(|e| Fail::Other(e.0))?;
        if !self.seen.contains(&ts) {
            // 🔧값을 먼저 읽는다(원본은 seen 에 먼저 넣어, 이 응답의 OI 를 못 읽으면 그 시각이 영구히 버려진다)
            let oi = py_float(obj.get("openInterest").ok_or_else(|| Fail::Other("KeyError: 'openInterest'".into()))?).map_err(|e| Fail::Other(e.0))?;
            self.seen.insert(ts);
            if self.seen.len() > 8192 {
                self.seen.retain(|s| *s >= ts - 600_000);
            }
            self.out.lock().unwrap().oi.push((ts, self.symbol.to_lowercase(), oi));
        }
        Ok(None)
    }
}

/// 웹소켓 하나(마크 종목 또는 강제청산)의 연결 상태 -- 🔧끊긴 구간을 gaps 에(원본에 없다).
pub struct Stream {
    pub feed: Feed,
    /// gaps.symbol: 마크 = 소문자 종목 · 강제청산 = None(전 종목)
    pub sym: Option<String>,
    pub p: Shared,
    down: Option<i64>,
}

impl Stream {
    pub fn new(feed: Feed, sym: Option<String>, p: Shared) -> Self {
        Self { feed, sym, p, down: None }
    }
    pub fn connected(&mut self, wall: f64) {
        let now = ms(wall);
        if let Some(d) = self.down.take() {
            if now - d >= 1500 {
                self.p.lock().unwrap().gaps.push((self.sym.clone(), d, now, "reconnect"));
            }
        }
    }
    pub fn text(&mut self, wall: f64, t: &str) -> Result<(), Reconnect> {
        self.feed.on_text(&self.p, wall, t)
    }
    /// 끊김·접속 실패: 처음 끊긴 시각만 잡는다.
    pub fn ended(&mut self, wall: f64) {
        self.down.get_or_insert(ms(wall));
    }
}

/// 정답지 입력의 웹소켓 몫을 먹인다: [{kind: mark|force, url, events: [[초, 종류, 본문]…]}…] -- 사건 'o' 접속 · 't' 글자 · 'c' 끝 · 'f' 접속 실패.
/// 행은 `p` 에 쌓인다(쓰기는 부르는 쪽 `flush`).
pub fn replay_ws(feeds: &[Value], slow: &[String], p: &Shared) {
    let liq = liq_symbols(slow);
    for f in feeds {
        let (feed, sym) = if f["kind"] == "mark" {
            (Feed::Mark, Some(f["url"].as_str().unwrap_or("").split('@').next().unwrap().to_string()))
        } else {
            (Feed::Force(liq.clone()), None)
        };
        let mut s = Stream::new(feed, sym, p.clone());
        let (mut open, mut prev) = (false, 0.0);
        for e in f["events"].as_array().unwrap() {
            let (t, kind, body) = (e[0].as_f64().unwrap(), e[1].as_str().unwrap(), e[2].as_str().unwrap_or(""));
            match kind {
                "o" => {
                    // 🔧고친 쪽은 원본이 끊었던 자리(깨진 JSON)에서 안 끊는다 -- 접속 사건이 연결 중에 오면 서버가 닫은 것으로(정답지도 같게)
                    if open {
                        s.ended(prev);
                    }
                    open = true;
                    s.connected(t);
                }
                "t" if open => {
                    if s.text(t, body).is_err() {
                        open = false;
                        s.ended(t);
                    }
                }
                "t" => {} // 원본이 끊은 뒤의 프레임(이 연결은 이미 접었다 -- 다음 'o' 까지 버린다)
                _ => {
                    open = false;
                    s.ended(t);
                }
            }
            prev = t;
        }
    }
}
