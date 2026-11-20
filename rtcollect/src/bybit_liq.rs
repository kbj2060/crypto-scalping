//! Bybit 청산 수집기(`scripts/live_bybit_liquidation_collector_20261003.py`, 줄 번호는 그 파일)에서 소켓을 뺀 것.
//! REST 없음. 시계는 원본처럼 **벽시계**(time.time() 초) 하나 -- `recv_ms` 칸과 공백의 끝이 이 시계다.
//! 🔴`side` 는 Bybit 규약 그대로 = **청산된 포지션의 방향**(Buy = 롱 청산). 바이낸스 S(강제 주문, 롱 청산 = SELL)와 반대.
use crate::{ms, need, py_float, py_int, py_sql, py_truthy, Reconnect};
use rusqlite::types::Value as Sql;
use rusqlite::{params, Connection};
use serde_json::Value;
use std::path::Path;

pub const FLUSH_S: f64 = 2.0; // :29
pub const PING_S: f64 = 20.0;
pub const IDLE_S: f64 = 60.0; // pong 도 안 오면 죽은 연결(:82)
pub const RECV_TIMEOUT_S: f64 = 5.0; // wait_for(ws.recv(), 5) (:78)
pub const SYMBOLS: &str = "ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT,HYPEUSDT"; // BYBIT_LIQ_SYMBOLS 기본(:27)

// 표(:44-47) -- 줄바꿈·들여쓰기까지 원본 글자(sqlite_master 에 그대로 남는다). auto_vacuum 은 원본도 안 건다.
const DDL: [&str; 3] = [
    "CREATE TABLE IF NOT EXISTS bybit_liquidations(
        ts_ms INTEGER, symbol TEXT, side TEXT, qty REAL, price REAL, recv_ms INTEGER)",
    "CREATE INDEX IF NOT EXISTS bybit_liq_sym_ts ON bybit_liquidations(symbol, ts_ms)",
    "CREATE TABLE IF NOT EXISTS gaps(from_ms INTEGER, to_ms INTEGER, reason TEXT)",
];

/// (거래소 시각 ms, 종목, 쪽, 수량, 가격, 받은 시각 ms). 종목·쪽은 메시지 값 그대로 묶는다(파이썬 바인딩 = 문자열·정수·실수·None).
pub type Row = (i64, Sql, Sql, f64, f64, i64);

/// allLiquidation 메시지 → 행(:32-36). 다른 메시지(구독 응답·pong)는 []. 한 건이라도 못 읽으면 메시지 전체가 예외(= 재연결).
pub fn parse(msg: &Value, recv_ms: i64) -> Result<Vec<Row>, Reconnect> {
    let Value::Object(m) = msg else { return Err(Reconnect("AttributeError: .get".into())) };
    // str(msg.get("topic", "")) -- 문자열이 아닌 주제는 str() 해도 "allLiquidation." 로 시작할 수 없다
    if !m.get("topic").and_then(Value::as_str).is_some_and(|t| t.starts_with("allLiquidation.")) {
        return Ok(vec![]);
    }
    let data = match m.get("data") {
        Some(Value::Array(a)) => a.as_slice(),
        Some(v) if py_truthy(v) => return Err(Reconnect("TypeError: 원소가 사전이 아님".into())), // 사전 열쇠·글자·숫자
        _ => &[],
    };
    data.iter()
        .map(|d| {
            let Value::Object(d) = d else { return Err(Reconnect("TypeError: 원소가 사전이 아님".into())) };
            // 평가 순서 = 튜플 순서(T, s, S, v, p)
            let ts = py_int(need(d, "T")?)?;
            let s = py_sql(need(d, "s")?)?;
            let side = py_sql(need(d, "S")?)?;
            Ok((ts, s, side, py_float(need(d, "v")?)?, py_float(need(d, "p")?)?, recv_ms))
        })
        .collect()
}

pub fn open_db(path: &Path) -> rusqlite::Result<Connection> {
    if let Some(d) = path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let con = Connection::open(path)?;
    con.busy_timeout(std::time::Duration::from_secs(30))?;
    con.query_row("PRAGMA journal_mode=WAL", [], |_| Ok(()))?;
    con.execute_batch("PRAGMA synchronous = NORMAL")?; // data_store.rw_connect 와 같은 이유
    for ddl in DDL {
        con.execute(ddl, [])?;
    }
    Ok(con)
}

/// 한 트랜잭션(:51-59). 열린 트랜잭션을 남기면 다음 BEGIN 이 전부 실패한다 -- 실패하면 ROLLBACK.
fn write(con: &Connection, rows: &[Row], gaps: &[(i64, i64, String)]) -> rusqlite::Result<()> {
    con.execute_batch("BEGIN")?;
    let r = (|| {
        let mut st = con.prepare_cached("INSERT INTO bybit_liquidations VALUES (?, ?, ?, ?, ?, ?)")?;
        for (a, b, c, d, e, f) in rows {
            st.execute(params![a, b, c, d, e, f])?;
        }
        for (a, b, c) in gaps {
            con.execute("INSERT INTO gaps VALUES (?, ?, ?)", params![a, b, c])?;
        }
        con.execute_batch("COMMIT")
    })();
    if r.is_err() {
        let _ = con.execute_batch("ROLLBACK");
    } else {
        crate::wrote();
    }
    r
}

/// `run()`(:62-99)의 상태. 부르는 쪽: 접속 → `on_open` · 글자 → `on_text` · 5초 수신 시한 → `on_timeout` ·
/// 끊김·예외·접속 실패 → `on_error`(돌려준 초만큼 쉬고 다시 붙는다). Ok(true) = 지금 {"op":"ping"} 을 보낸다.
pub struct LiqCollector {
    con: Connection,
    pub buf: Vec<Row>,
    down: Option<(i64, String)>, // (끊긴 시각 ms, 이유)
    backoff: f64,
    last_ping: f64,
    last_flush: f64,
    last_msg: f64,
}

impl LiqCollector {
    pub fn new(con: Connection) -> Self {
        Self { con, buf: vec![], down: None, backoff: 1.0, last_ping: 0.0, last_flush: 0.0, last_msg: 0.0 }
    }

    /// 구독을 보낸 직후(:72-75). 끊겼던 구간을 gaps 에 쓴다(못 쓰면 예외 = 재연결, 공백은 들고 간다).
    pub fn on_open(&mut self, t: f64) -> Result<(), Reconnect> {
        if let Some((from, why)) = &self.down {
            write(&self.con, &[], &[(*from, ms(t), why.clone())]).map_err(|e| Reconnect(format!("OperationalError: {e}")))?;
            self.down = None;
        }
        (self.backoff, self.last_ping, self.last_flush, self.last_msg) = (1.0, t, t, t);
        Ok(())
    }

    pub fn on_text(&mut self, t: f64, text: &str) -> Result<bool, Reconnect> {
        self.last_msg = t;
        let v: Value = serde_json::from_str(text).map_err(|e| Reconnect(format!("JSONDecodeError: {e}")))?;
        let rows = parse(&v, ms(t))?;
        self.buf.extend(rows);
        self.after(t)
    }

    pub fn on_timeout(&mut self, t: f64) -> Result<bool, Reconnect> {
        if t - self.last_msg > IDLE_S {
            return Err(Reconnect("ConnectionError: 응답 없음 60초".into()));
        }
        self.after(t)
    }

    fn after(&mut self, now: f64) -> Result<bool, Reconnect> {
        let ping = now - self.last_ping >= PING_S;
        if ping {
            self.last_ping = now;
        }
        if !self.buf.is_empty() && now - self.last_flush >= FLUSH_S {
            write(&self.con, &self.buf, &[]).map_err(|e| Reconnect(format!("OperationalError: {e}")))?;
            self.buf.clear();
            self.last_flush = now;
        }
        Ok(ping)
    }

    /// :91-99. 남은 행을 쓰고(못 쓰면 들고 다음 연결에서 다시) 처음 끊긴 시각을 남긴다. 반환 = 쉴 초(1·2·4…60).
    pub fn on_error(&mut self, t: f64, reason: &str) -> f64 {
        if !self.buf.is_empty() {
            match write(&self.con, &self.buf, &[]) {
                Ok(()) => self.buf.clear(),
                Err(e) => eprintln!("잔여 {}건 기록 보류: {e}", self.buf.len()),
            }
        }
        if self.down.is_none() {
            self.down = Some((ms(t), reason.chars().take(160).collect())); // 파이썬 [:160] = 글자 단위
        }
        let b = self.backoff;
        self.backoff = (b * 2.0).min(60.0);
        b
    }
}

/// 정답지 입력을 원본과 같은 규칙으로 먹인다. 사건 = [벽시계 초, 종류, 본문]:
/// 'o' 접속 성공 · 'f' 접속 실패(본문 = 예외 글자) · 't' 글자 메시지 · 'w' 5초 수신 시한 · 'c' 끊김(본문 = 예외 글자).
/// 연결 안에서 예외가 나면 그 연결은 끝이다(다음 사건은 'o'·'f' 여야 한다 -- 생성기가 그렇게 만든다).
pub fn replay(events: &[Value], path: &Path) -> rusqlite::Result<()> {
    let mut c = LiqCollector::new(open_db(path)?);
    for e in events {
        let (t, kind, body) = (e[0].as_f64().unwrap(), e[1].as_str().unwrap(), e[2].as_str().unwrap_or(""));
        let r = match kind {
            "o" => c.on_open(t).map(|_| false),
            "t" => c.on_text(t, body),
            "w" => c.on_timeout(t),
            _ => Err(Reconnect(body.to_string())), // 'c' · 'f'
        };
        if let Err(Reconnect(why)) = r {
            c.on_error(t, &why);
        }
    }
    Ok(())
}
