//! Bybit 체결 테이프 + 맥락(OI·마크·펀딩) 수집기(`scripts/live_bybit_trade_tape_collector_20261006.py`, 줄 번호는 그 파일)
//! 에서 소켓·REST(1분봉 대조)를 뺀 것. 공개 WS 하나(`publicTrade` + `tickers`)로
//! · 체결 → bybit_tape.sqlite `trade_tape_1s`(바이낸스 테이프와 같은 표·같은 칸 뜻 -- `store`·`tape` 를 그대로 쓴다)
//! · tickers → bybit_ctx.sqlite `bybit_oi`(바뀔 때) · `bybit_mark`(종목당 1초 1점) · `bybit_funding`(바뀔 때) · `gaps`(끊긴 구간)
//! 🔴`S` 는 **테이커** 방향이다(Buy = 테이커 매수) -- 뒤집지 않는다. 청산은 따로(`bybit_liq`).
//! 시계 둘: `now` = 단조 초(time.monotonic -- 플러시·ping·묶음 닫기) · `wall` = 벽시계 초(time.time -- 맥락 gaps 의 시각).
use crate::store::{open_db as open_tape, TapeStore};
use crate::tape::{bucket, TapeBuffer};
use crate::{py_float, py_int, py_iter, py_truthy, Reconnect};
use rusqlite::{params, Connection};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::Path;

pub const FLUSH_S: f64 = 5.0; // :44
pub const PING_S: f64 = 20.0;
pub const IDLE_S: f64 = 60.0;
pub const RECV_TIMEOUT_S: f64 = 5.0; // ws.receive(timeout=5) (:225)
/// 2026-10-31 멈춤 자가 종료: 10-28~31 프로세스가 살아 있는 채 72시간 멈췄다 -- 체결이 이만큼 하나도 안 오면 끝낸다(종료코드 3).
pub const STALL_S: f64 = 600.0;
pub const MARK_EVERY_MS: i64 = 1000; // ponytail(원본): 마크는 종목당 1초 1점(tickers 는 100ms)
pub const SYMBOLS: &str = "ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT,HYPEUSDT"; // BYBIT_TAPE_SYMBOLS 기본(:41)
const ORDER_IDLE_S: f64 = 1.0; // 1초 넘게 조용한 묶음은 플러시 때 닫는다(:200)

// ctx_conn(:88-103) -- 글자 그대로
const CTX_DDL: [&str; 7] = [
    "CREATE TABLE IF NOT EXISTS bybit_oi(inst TEXT, ts_ms INTEGER, oi_base REAL, oi_usd REAL)",
    "CREATE TABLE IF NOT EXISTS bybit_mark(inst TEXT, ts_ms INTEGER, mark_px REAL, index_px REAL)",
    "CREATE TABLE IF NOT EXISTS bybit_funding(inst TEXT, ts_ms INTEGER, funding_rate REAL, next_funding_ms INTEGER)",
    "CREATE TABLE IF NOT EXISTS gaps(from_ms INTEGER, to_ms INTEGER, reason TEXT)",
    "CREATE INDEX IF NOT EXISTS bybit_oi_inst_ts ON bybit_oi(inst, ts_ms)",
    "CREATE INDEX IF NOT EXISTS bybit_mark_inst_ts ON bybit_mark(inst, ts_ms)",
    "CREATE INDEX IF NOT EXISTS bybit_funding_inst_ts ON bybit_funding(inst, ts_ms)",
];

pub fn open_ctx(path: &Path) -> rusqlite::Result<Connection> {
    if let Some(d) = path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let con = Connection::open(path)?;
    con.busy_timeout(std::time::Duration::from_secs(30))?;
    con.execute_batch("PRAGMA auto_vacuum = INCREMENTAL")?; // 새 파일에서만, WAL 전환보다 먼저(봉인기 prune_hot 이 공간을 돌려받게)
    con.query_row("PRAGMA journal_mode=WAL", [], |_| Ok(()))?;
    con.execute_batch("PRAGMA synchronous = NORMAL")?;
    for ddl in CTX_DDL {
        con.execute(ddl, [])?;
    }
    Ok(con)
}

/// 체결 → (시각 ms, 방향, 가격) 묶음(:51-72). ⭐한 테이커 주문의 체결은 시각이 같다(실측 5코인) -- 가격까지 묶으면 바이낸스
/// aggTrade · OKX `trades` 와 같은 단위. `seq` 대신 시각인 이유: 과거 일별 덤프에 seq 가 없다(검정과 라이브가 같은 규칙).
/// 메시지 경계를 넘어도 이어 붙는다 -- 닫는 건 «다른 키» 또는 플러시(1초 넘게 조용할 때).
#[derive(Default)]
pub struct OrderGrouper {
    key: Option<(i64, bool, f64)>,
    qty: f64,
    n: i64,
    last: f64,
}

/// (ts_ms, px, qty, sell, 체결수)
pub type Order = (i64, f64, f64, bool, i64);

impl OrderGrouper {
    pub fn add(&mut self, sell: bool, px: f64, qty: f64, ts_ms: i64, now: f64) -> Option<Order> {
        let k = Some((ts_ms, sell, px));
        let mut done = None;
        if k != self.key {
            done = self.flush();
            (self.key, self.qty, self.n) = (k, 0.0, 0);
        }
        self.qty += qty;
        self.n += 1;
        self.last = now;
        done
    }

    pub fn flush(&mut self) -> Option<Order> {
        let (ts, sell, px) = self.key.take()?;
        Some((ts, px, self.qty, sell, self.n))
    }
}

/// publicTrade → [(심볼 값, sell, px, qty, ts_ms)](:75-85). 못 믿을 행은 버린다(그 행만). 심볼은 `d.get("s")` 그대로(없으면 Null).
pub fn parse_trades(msg: &Map<String, Value>) -> Result<Vec<(Value, bool, f64, f64, i64)>, Reconnect> {
    let mut out = vec![];
    for d in py_iter(msg, "data")? {
        let Value::Object(d) = d else { continue }; // 사전 아닌 원소 = d["p"] TypeError -> 그 행만
        let get = |k: &str| d.get(k).ok_or(());
        let one = || -> Result<(f64, f64, i64, &Value), ()> {
            Ok((py_float(get("p")?).map_err(drop)?, py_float(get("v")?).map_err(drop)?, py_int(get("T")?).map_err(drop)?, get("S")?))
        };
        let Ok((px, qty, ts, side)) = one() else { continue };
        let side = side.as_str().unwrap_or("");
        if px > 0.0 && qty > 0.0 && ts > 0 && (side == "Buy" || side == "Sell") {
            out.push((d.get("s").cloned().unwrap_or(Value::Null), side == "Sell", px, qty, ts));
        }
    }
    Ok(out)
}

#[derive(Default)]
struct Rows {
    oi: Vec<(String, i64, f64, Option<f64>)>,
    funding: Vec<(String, i64, f64, i64)>,
    mark: Vec<(String, i64, Option<f64>, Option<f64>)>,
}

impl Rows {
    fn is_empty(&self) -> bool {
        self.oi.is_empty() && self.funding.is_empty() && self.mark.is_empty()
    }
}

/// 행을 만드는 숫자 칸(float 로 읽는다 · nextFundingTime 은 int 로도).
const NUM_KEYS: [&str; 6] = ["openInterest", "openInterestValue", "fundingRate", "nextFundingTime", "markPrice", "indexPrice"];

fn num_ok(k: &str, v: &Value) -> bool {
    py_float(v).is_ok_and(|x| k != "nextFundingTime" || x.is_finite())
}

/// tickers 는 첫 메시지만 스냅샷이고 뒤는 «바뀐 칸만» 온다 -- 종목별로 합쳐 들고 있다가 행으로 낸다(:106-135).
/// 평가 순서·중간 상태 변경까지 원본과 같다(OI 를 갱신한 뒤 펀딩에서 예외가 나면 OI 상태만 바뀐 채 행은 안 나간다).
#[derive(Default)]
pub struct TickerState {
    cur: HashMap<String, Map<String, Value>>,
    last_oi: HashMap<String, f64>,
    last_fr: HashMap<String, (f64, i64)>,
    last_mark_ms: HashMap<String, i64>,
}

impl TickerState {
    fn rows(&mut self, msg: &Map<String, Value>, out: &mut Rows) -> Result<(), Reconnect> {
        let d = msg.get("data").filter(|v| py_truthy(v));
        let ts = match msg.get("ts").filter(|v| py_truthy(v)) {
            Some(v) => py_int(v)?,
            None => 0,
        };
        let d = match d {
            None => return Ok(()), // {} .get("symbol") = None
            Some(Value::Object(d)) => d,
            Some(_) => return Err(Reconnect("AttributeError".into())),
        };
        // `if not s or ts <= 0: return {}`
        let Some(sym) = d.get("symbol").filter(|v| py_truthy(v)) else { return Ok(()) };
        if ts <= 0 {
            return Ok(());
        }
        let s = match sym {
            Value::String(s) => s.as_str(),
            Value::Array(_) | Value::Object(_) => return Err(Reconnect("TypeError".into())), // cur.setdefault(목록·사전) = unhashable
            // ponytail: 참인 숫자·True 를 원본은 그 값을 열쇠로 쌓는다(inst 칸이 INTEGER) -- Bybit 는 문자열만 보내 건너뛴다.
            _ => return Ok(()),
        };
        let c = self.cur.entry(s.to_string()).or_default();
        for (k, v) in d {
            if !(v.is_null() || v.as_str() == Some("")) {
                // 🔧의도한 차이(SHADOW.md §11): 원본은 못 읽는 숫자 칸(예: fundingRate "x")도 합쳐 두어 그 종목의 **모든 다음 tickers 가 예외**
                //   (= 재연결 반복, 상태 오염)이고, 그 메시지에서 OI 상태만 먼저 바뀌어 **OI 행이 영영 빠진다**. 여기는 그 칸만 버린다.
                if NUM_KEYS.contains(&k.as_str()) && !num_ok(k, v) {
                    eprintln!("⚠️{s} tickers 칸 {k}={v} 못 읽음 -- 그 칸만 버림");
                    continue;
                }
                c.insert(k.clone(), v.clone());
            }
        }
        let f = |k: &str| c.get(k).map(py_float).transpose();
        let mut rows = Rows::default();
        if let Some(oi) = f("openInterest")? {
            if self.last_oi.get(s) != Some(&oi) {
                self.last_oi.insert(s.to_string(), oi);
                rows.oi.push((s.to_string(), ts, oi, f("openInterestValue")?));
            }
        }
        let fr0 = f("fundingRate")?;
        let nft = match f("nextFundingTime")? {
            Some(x) if x != 0.0 => {
                if x.is_nan() {
                    return Err(Reconnect("ValueError".into())); // int(nan)
                }
                if x.is_infinite() {
                    return Err(Reconnect("OverflowError".into())); // int(inf)
                }
                x.trunc() as i64
            }
            _ => 0,
        };
        if let Some(fr0) = fr0 {
            if self.last_fr.get(s) != Some(&(fr0, nft)) {
                self.last_fr.insert(s.to_string(), (fr0, nft));
                rows.funding.push((s.to_string(), ts, fr0, nft));
            }
        }
        if d.contains_key("markPrice") && ts - self.last_mark_ms.get(s).copied().unwrap_or(0) >= MARK_EVERY_MS {
            self.last_mark_ms.insert(s.to_string(), ts);
            rows.mark.push((s.to_string(), ts, f("markPrice")?, f("indexPrice")?));
        }
        out.oi.extend(rows.oi);
        out.funding.extend(rows.funding);
        out.mark.extend(rows.mark);
        Ok(())
    }
}

struct Sym {
    name: String,
    store: TapeStore,
    buf: TapeBuffer,
    group: OrderGrouper,
    last_ms: i64,
    first: bool,
}

/// `run()`(:175-270)의 상태. 부르는 쪽: 접속(구독 전송 뒤) → `on_open` · 글자 → `on_text` · 5초 수신 시한 → `on_timeout` ·
/// 끊김·예외·접속 실패 → `on_error`(그다음 3초 쉬고 다시). Ok(true) = 지금 {"op":"ping"} 을 보낸다.
pub struct BybitCollector {
    syms: Vec<Sym>,
    ctx: Connection,
    tick: TickerState,
    ctx_buf: Rows,
    down_ms: Option<(i64, String)>,
    t_ping: f64,
    t_flush: f64,
    t_msg: f64,
    /// 마지막으로 받은(구독 종목) 체결의 단조 시각 -- 정지 감시(STALL_S)가 본다.
    pub beat: f64,
}

impl BybitCollector {
    /// 원본 순서 그대로: 종목마다 TapeStore → 종목마다 meta 두 줄 → 맥락 파일 → 종목마다 last_ts_ms.
    pub fn new(tape_db: &Path, ctx_db: &Path, symbols: &[String], now: f64) -> rusqlite::Result<Self> {
        let con = open_tape(tape_db)?;
        let mut syms = vec![];
        for s in symbols {
            let b = bucket(&s.to_lowercase());
            let store = TapeStore::new(con.clone(), s, b)?;
            syms.push(Sym { name: s.clone(), store, buf: TapeBuffer::new(b, s), group: Default::default(), last_ms: 0, first: true });
        }
        for s in &syms {
            s.store.put_meta(&format!("max_unit:{}", s.name), "taker_order_price_level")?;
            s.store.put_meta(&format!("source:{}", s.name), "bybit v5 publicTrade grouped by consecutive (T ms, side, price)")?;
        }
        let ctx = open_ctx(ctx_db)?;
        for s in &mut syms {
            s.last_ms = s.store.last_ts_ms()?;
        }
        Ok(Self { syms, ctx, tick: Default::default(), ctx_buf: Default::default(), down_ms: None, t_ping: 0.0, t_flush: 0.0, t_msg: 0.0, beat: now })
    }

    /// 구독 메시지(:219) -- 체결 다음 tickers, 종목 순서대로. 글자는 aiohttp send_json(= json.dumps 기본) 그대로.
    pub fn subscribe(&self) -> String {
        let args: Vec<String> =
            ["publicTrade", "tickers"].iter().flat_map(|t| self.syms.iter().map(move |s| Value::from(format!("{t}.{}", s.name)).to_string())).collect();
        crate::py_subscribe(&args)
    }

    pub fn on_open(&mut self, now: f64) {
        for s in &mut self.syms {
            s.first = true; // 연결 뒤 첫 체결 -- 그 앞은 «모름»
        }
        (self.t_ping, self.t_flush, self.t_msg) = (now, now, now);
    }

    pub fn on_text(&mut self, now: f64, wall: f64, text: &str) -> Result<bool, Reconnect> {
        self.t_msg = now;
        let v: Value = serde_json::from_str(text).map_err(|_| Reconnect("JSONDecodeError".into()))?;
        let Value::Object(msg) = &v else { return Err(Reconnect("AttributeError".into())) };
        let topic = msg.get("topic").and_then(Value::as_str).unwrap_or(""); // str(...) 이 아닌 값은 두 접두어 어느 쪽도 아니다
        if topic.starts_with("publicTrade.") {
            for (s, sell, px, qty, ts) in parse_trades(msg)? {
                // `if s not in groups` -- 목록·사전은 해시가 안 돼 TypeError(= 재연결), 문자열 아닌 값은 «없음»
                if matches!(s, Value::Array(_) | Value::Object(_)) {
                    return Err(Reconnect("TypeError".into()));
                }
                let Some(x) = self.syms.iter_mut().find(|x| Some(x.name.as_str()) == s.as_str()) else { continue };
                if x.first {
                    x.first = false;
                    let (from, why) = if x.last_ms != 0 { (x.last_ms, "ws_reconnect") } else { (ts.div_euclid(60_000) * 60_000, "startup") };
                    x.store.record_gap(from, ts, why);
                }
                if let Some(o) = x.group.add(sell, px, qty, ts, now) {
                    x.buf.add_agg(o.0, o.1, o.2, o.3, o.4);
                }
                x.last_ms = ts;
                self.beat = now;
            }
        } else if topic.starts_with("tickers.") {
            self.tick.rows(msg, &mut self.ctx_buf)?;
        }
        self.after(now, wall)
    }

    pub fn on_timeout(&mut self, now: f64, wall: f64) -> Result<bool, Reconnect> {
        if now - self.t_msg > IDLE_S {
            return Err(Reconnect("ConnectionError".into()));
        }
        self.after(now, wall)
    }

    fn after(&mut self, now: f64, wall: f64) -> Result<bool, Reconnect> {
        let ping = now - self.t_ping >= PING_S;
        if ping {
            self.t_ping = now;
        }
        if now - self.t_flush >= FLUSH_S {
            self.flush(false, now).map_err(|e| Reconnect(format!("OperationalError: {e}")))?;
            self.t_flush = now;
        }
        if let Some((from, why)) = &self.down_ms {
            self.ctx
                .execute("INSERT INTO gaps VALUES (?, ?, ?)", params![from, (wall * 1000.0) as i64, why])
                .map_err(|e| Reconnect(format!("OperationalError: {e}")))?;
            self.down_ms = None;
        }
        Ok(ping)
    }

    /// :197-213. 테이프 쓰기 실패는 TapeStore 가 보류하고(조용한 유실 방지), 맥락 쓰기 실패는 예외(행은 남아 다음에 다시).
    fn flush(&mut self, force: bool, now: f64) -> rusqlite::Result<()> {
        for s in &mut self.syms {
            if s.group.key.is_some() && (force || now - s.group.last > ORDER_IDLE_S) {
                if let Some(o) = s.group.flush() {
                    s.buf.add_agg(o.0, o.1, o.2, o.3, o.4);
                }
            }
        }
        for s in &mut self.syms {
            let rows = s.buf.take_closed(force);
            s.store.write(rows);
        }
        if self.ctx_buf.is_empty() {
            return Ok(());
        }
        self.ctx.execute_batch("BEGIN")?;
        let r = (|| {
            let b = &self.ctx_buf;
            for r in &b.oi {
                self.ctx.execute("INSERT INTO bybit_oi VALUES (?,?,?,?)", params![r.0, r.1, r.2, r.3])?;
            }
            for r in &b.funding {
                self.ctx.execute("INSERT INTO bybit_funding VALUES (?,?,?,?)", params![r.0, r.1, r.2, r.3])?;
            }
            for r in &b.mark {
                self.ctx.execute("INSERT INTO bybit_mark VALUES (?,?,?,?)", params![r.0, r.1, r.2, r.3])?;
            }
            self.ctx.execute_batch("COMMIT")
        })();
        match r {
            Ok(()) => {
                self.ctx_buf = Rows::default();
                Ok(())
            }
            Err(e) => {
                let _ = self.ctx.execute_batch("ROLLBACK"); // 열린 트랜잭션을 남기면 다음 BEGIN 이 전부 실패한다(10-06 리뷰)
                Err(e)
            }
        }
    }

    /// :261-269. 처음 끊긴 시각을 남기고, 열린 묶음·진행 중인 초까지 다 쓴 뒤 그 초를 닫는다(재연결 뒤 같은 초는 버린다).
    /// gaps.reason = 원본 `type(exc).__name__` -- 예외 글의 «이름:» 앞부분만 남긴다.
    pub fn on_error(&mut self, now: f64, wall: f64, why: &str) {
        if self.down_ms.is_none() {
            self.down_ms = Some(((wall * 1000.0) as i64, why.split(':').next().unwrap_or(why).to_string()));
        }
        if let Err(e) = self.flush(true, now) {
            eprintln!("재연결 전 쓰기 실패(다음 주기 재시도): {e}");
        }
        for s in &mut self.syms {
            s.buf.seal();
        }
    }
}

/// 정답지 입력을 원본과 같은 규칙으로 먹인다. 실행(판)마다 새 수집기(같은 파일 -- 재시작). 사건 = [초, 종류, 본문] --
/// 초는 단조·벽시계 둘 다(가짜 시계가 같은 값을 준다). 'o' 접속 · 'f' 접속 실패(본문 = 예외 이름) · 't' 글자 · 'w' 5초 수신 시한 ·
/// 'c' 끊김(본문 = 예외 이름). 입력이 끝나면 멈춘다(파이썬 쪽도 CancelledError 로 끝난다 -- 마지막 쓰기 없음).
pub fn replay(runs: &[Value], dir: &Path) -> rusqlite::Result<()> {
    for run in runs {
        let syms: Vec<String> = run["symbols"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
        let mut c = BybitCollector::new(&dir.join("bybit_tape.sqlite"), &dir.join("bybit_ctx.sqlite"), &syms, run["t0"].as_f64().unwrap())?;
        let (mut open, mut prev) = (false, run["t0"].as_f64().unwrap());
        for e in run["events"].as_array().unwrap() {
            let (t, kind, body) = (e[0].as_f64().unwrap(), e[1].as_str().unwrap(), e[2].as_str().unwrap_or(""));
            // 🔧고친 동작은 원본이 끊었던 자리(못 읽는 tickers 칸)에서 안 끊는다 -- 그 뒤 접속 사건이 오면 서버가 닫은 것으로(정답지도 같게)
            if kind == "o" && open {
                c.on_error(prev, prev, "ConnectionError: ws CLOSE");
            }
            prev = t;
            open = kind == "o" || (open && (kind == "t" || kind == "w"));
            let r = match kind {
                "o" => {
                    c.on_open(t);
                    Ok(false)
                }
                "t" => c.on_text(t, t, body),
                "w" => c.on_timeout(t, t),
                _ => Err(Reconnect(body.to_string())),
            };
            if let Err(Reconnect(why)) = r {
                open = false;
                c.on_error(t, t, &why);
            }
        }
    }
    Ok(())
}
