//! 나스닥100 등 무기한(Hyperliquid xyz:*) 수집기(`scripts/live_xyz100_collector_20261014.py`, 줄 번호는 그 파일) → hl_xyz.sqlite.
//! · hl_xyz_1m(coin, ts_sec = 1분봉 시작, OHLCV, n) -- 웹소켓 candle 갱신을 그 분에 덮어쓴다(진행 중 봉도 계속 바뀐다).
//! · hl_xyz_ctx_1s(coin, recv_sec, mid, mark, oracle) -- activeAssetCtx(~1/초). 🔴거래소 시각이 없다 -- **받은 시각**(벽시계)이라 열 이름이 recv_sec.
//! · 접속할 때마다 REST candleSnapshot 으로 마지막 저장 봉(없으면 72시간 전)부터 메운다 -- 🔴기본 꺼짐(`backfill`, main.rs `RTCOLLECT_REST=1`).
//!   메움이 실패해도 웹소켓은 그대로 붙는다(실시간이 REST 에 묶이지 않게, 10-14 검토).
//! 🔴주말엔 CME 가 쉬어 XYZ100 도 거의 안 움직인다 · JPY·EUR 은 얇다(체결 없는 분은 1분봉이 빈다 -- 가격은 ctx 의 oracle 이 낫다).
//! 🔧의도한 차이(SHADOW.md §11): 원본은 끊기면(예외·시한 셋·글자 아닌 메시지) 마지막 플러시 뒤 받은 행(최대 2초)을 버린다 -- 여기는 끊길 때 쓴다(종료 때도).
use crate::store::{in_tx, open_hot};
use crate::{py_float, py_int, py_sql, py_truthy, Reconnect};
use rusqlite::{params, Connection};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::Path;

pub const WS_URL: &str = "wss://api.hyperliquid.xyz/ws";
pub const INFO_URL: &str = "https://api.hyperliquid.xyz/info";
pub const COINS: &str = "xyz:XYZ100,xyz:SP500,xyz:JPY,xyz:EUR";
pub const FLUSH_S: f64 = 2.0;
/// candleSnapshot 은 5000봉까지 -- 72시간(4,320봉)이면 긴 끊김도 메운다(:38).
pub const BACKFILL_H: i64 = 72;
/// ctx 는 ~1초마다 온다 -- 30초 조용하면 ping, 세 번이면 끊김(:129).
pub const RECV_TIMEOUT_S: f64 = 30.0;
pub const PING: &str = r#"{"method": "ping"}"#;

const DDL: [&str; 2] = [
    "CREATE TABLE IF NOT EXISTS hl_xyz_1m(coin TEXT NOT NULL, ts_sec INTEGER NOT NULL, open REAL, high REAL, low REAL, close REAL, volume REAL, n INTEGER, PRIMARY KEY(coin, ts_sec))",
    "CREATE TABLE IF NOT EXISTS hl_xyz_ctx_1s(coin TEXT NOT NULL, recv_sec INTEGER NOT NULL, mid REAL, mark REAL, oracle REAL, PRIMARY KEY(coin, recv_sec))",
];

pub fn open_db(path: &Path) -> rusqlite::Result<Connection> {
    open_hot(path, &DDL)
}

pub type Bar = (Value, i64, f64, f64, f64, f64, f64, i64);
pub type Ctx = (Value, i64, Option<f64>, Option<f64>, Option<f64>);

/// `candle_row`(:47-55): 웹소켓 candle·REST candleSnapshot 공통 원소 → 행. 종목을 모르거나 못 읽으면 None.
pub fn candle_row(c: &Map<String, Value>, coin: Option<&str>) -> Option<Bar> {
    let coin = match (c.get("s").filter(|v| py_truthy(v)), coin) {
        (Some(s), _) => s.clone(),
        (None, Some(c)) if !c.is_empty() => Value::from(c),
        _ => return None,
    };
    let f = |k: &str| c.get(k).ok_or(()).and_then(|v| py_float(v).map_err(drop));
    let or0 = |k: &str| c.get(k).filter(|v| py_truthy(v));
    let t = py_int(c.get("t")?).ok()?.div_euclid(1000);
    let (o, h, l, cl) = (f("o").ok()?, f("h").ok()?, f("l").ok()?, f("c").ok()?);
    let v = match or0("v") {
        Some(v) => py_float(v).ok()?,
        None => 0.0,
    };
    let n = match or0("n") {
        Some(v) => py_int(v).ok()?,
        None => 0,
    };
    Some((coin, t, o, h, l, cl, v, n))
}

/// `ctx_row`(:58-68): activeAssetCtx → (coin, 받은 초, mid, mark, oracle). 셋 다 없거나 종목을 모르면 None.
/// 🔴ctx 가 참인 비사전이면 원본은 AttributeError(재연결) -- 잡는 예외가 TypeError·ValueError 뿐이다.
pub fn ctx_row(data: &Map<String, Value>, recv_ms: i64) -> Result<Option<Ctx>, Reconnect> {
    let ctx = data.get("ctx").filter(|v| py_truthy(v));
    let Some(coin) = data.get("coin").filter(|v| py_truthy(v)) else { return Ok(None) };
    let empty = Map::new();
    let ctx = match ctx {
        None => &empty,
        Some(Value::Object(o)) => o,
        Some(_) => return Err(Reconnect("AttributeError: .get".into())),
    };
    // float(ctx[k]) if ctx.get(k) not in (None, "") else None
    let f = |k: &str| match ctx.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.is_empty() => Ok(None),
        Some(v) => py_float(v).map(Some),
    };
    let (Ok(mid), Ok(mark), Ok(ora)) = (f("midPx"), f("markPx"), f("oraclePx")) else { return Ok(None) };
    if mid.is_none() && mark.is_none() && ora.is_none() {
        return Ok(None);
    }
    Ok(Some((coin.clone(), recv_ms.div_euclid(1000), mid, mark, ora)))
}

/// 파이썬 사전 열쇠(종목, 시각)의 같음: 글자는 글자끼리 · 숫자·참거짓은 값으로(True == 1 == 1.0). 목록·사전은 hash 불가(TypeError = 재연결).
fn key(coin: &Value, t: i64) -> Result<(String, i64), Reconnect> {
    Ok((
        match coin {
            Value::String(s) => format!("s{s}"),
            Value::Bool(b) => format!("n{}", *b as i64 as f64),
            Value::Number(n) => format!("n{}", n.as_f64().unwrap_or(f64::NAN) + 0.0),
            _ => return Err(Reconnect("TypeError: unhashable type".into())),
        },
        t,
    ))
}

/// 삽입 순서를 지키는 사전(원본 `buffer`: 같은 열쇠면 자리는 그대로, 값만 새것).
pub struct Ordered<R> {
    rows: Vec<R>,
    at: HashMap<(String, i64), usize>,
}

impl<R> Default for Ordered<R> {
    fn default() -> Self {
        Self { rows: vec![], at: HashMap::new() }
    }
}

impl<R> Ordered<R> {
    fn put(&mut self, k: (String, i64), r: R) {
        match self.at.get(&k) {
            Some(&i) => self.rows[i] = r,
            None => {
                self.at.insert(k, self.rows.len());
                self.rows.push(r);
            }
        }
    }
    fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// 한 트랜잭션으로 덮어쓴다(:85-91).
pub fn write(con: &Connection, bars: &[Bar], ctxs: &[Ctx]) -> rusqlite::Result<()> {
    in_tx(con, || {
        for b in bars {
            con.prepare_cached("INSERT OR REPLACE INTO hl_xyz_1m VALUES (?,?,?,?,?,?,?,?)")?
                .execute(params![py_sql(&b.0).expect("열쇠에서 걸렀다"), b.1, b.2, b.3, b.4, b.5, b.6, b.7])?;
        }
        for c in ctxs {
            con.prepare_cached("INSERT OR REPLACE INTO hl_xyz_ctx_1s VALUES (?,?,?,?,?)")?
                .execute(params![py_sql(&c.0).expect("열쇠에서 걸렀다"), c.1, c.2, c.3, c.4])?;
        }
        Ok(())
    })
}

/// `run()`(:107-152)의 상태.
pub struct XyzCollector {
    pub con: Connection,
    pub coins: Vec<String>,
    bars: Ordered<Bar>,
    ctxs: Ordered<Ctx>,
    last_flush: f64,
    quiet: u32,
    backoff: f64,
}

impl XyzCollector {
    pub fn new(con: Connection, coins: &[String]) -> Self {
        Self { con, coins: coins.to_vec(), bars: Ordered::default(), ctxs: Ordered::default(), last_flush: 0.0, quiet: 0, backoff: 1.0 }
    }

    /// 접속 직후: 보낼 구독 글자(종목마다 candle · activeAssetCtx, json.dumps 기본 모양). 버퍼는 이 연결의 것.
    pub fn on_open(&mut self, now: f64) -> Vec<String> {
        (self.bars, self.ctxs, self.last_flush, self.quiet) = (Ordered::default(), Ordered::default(), now, 0);
        self.coins
            .iter()
            .flat_map(|c| {
                let c = Value::from(c.as_str());
                [format!(r#"{{"method": "subscribe", "subscription": {{"type": "candle", "coin": {c}, "interval": "1m"}}}}"#),
                 format!(r#"{{"method": "subscribe", "subscription": {{"type": "activeAssetCtx", "coin": {c}}}}}"#)]
            })
            .collect()
    }

    /// 글자 메시지(:135-148). Err = 연결을 접는다. `wall` = 받은 시각(recv_sec).
    pub fn on_text(&mut self, now: f64, wall: f64, text: &str) -> Result<(), Reconnect> {
        let d: Value = serde_json::from_str(text).map_err(|e| Reconnect(format!("JSONDecodeError: {e}")))?;
        let Value::Object(d) = d else { return Err(Reconnect("AttributeError: .get".into())) };
        let (ch, data) = (d.get("channel").and_then(Value::as_str), d.get("data"));
        match (ch, data) {
            // 구독 거부 등 -- 조용히 버리면 칸이 영구 «끊김»인데 원인이 안 남는다
            (Some("error"), _) => eprintln!("[xyz100] 거래소 오류 {}", data.map(|v| v.to_string()).unwrap_or_default().chars().take(200).collect::<String>()),
            (Some("candle"), Some(Value::Object(c))) => {
                if let Some(r) = candle_row(c, None) {
                    let k = key(&r.0, r.1)?;
                    self.bars.put(k, r);
                    self.quiet = 0;
                }
            }
            (Some("activeAssetCtx"), Some(Value::Object(c))) => {
                if let Some(r) = ctx_row(c, crate::ms(wall))? {
                    let k = key(&r.0, r.1)?;
                    self.ctxs.put(k, r);
                    self.quiet = 0;
                }
            }
            _ => {}
        }
        if now - self.last_flush >= FLUSH_S && !(self.bars.is_empty() && self.ctxs.is_empty()) {
            write(&self.con, &self.bars.rows, &self.ctxs.rows).map_err(|e| Reconnect(format!("OperationalError: {e}")))?;
            (self.bars, self.ctxs, self.last_flush, self.backoff) = (Ordered::default(), Ordered::default(), now, 1.0);
        }
        Ok(())
    }

    /// 30초 수신 시한: Ok = {"method": "ping"} 을 보낸다 · 세 번째면 끊김.
    pub fn on_timeout(&mut self) -> Result<(), Reconnect> {
        self.quiet += 1;
        if self.quiet >= 3 {
            return Err(Reconnect("ConnectionError: 90초 무소식".into()));
        }
        Ok(())
    }

    /// 끊김·접속 실패(:149-152). 남은 버퍼를 쓴다(못 쓰면 로그 -- 목록·사전 종목은 열쇠에서 이미 걸렀다). 반환 = 쉴 초(1·2·4…60).
    pub fn on_error(&mut self) -> f64 {
        self.write_rest();
        let b = self.backoff;
        self.backoff = (b * 2.0).min(60.0);
        b
    }

    /// 마지막 플러시 뒤 받은 행을 쓴다(끊길 때·종료 때).
    pub fn write_rest(&mut self) -> usize {
        let n = self.bars.rows.len() + self.ctxs.rows.len();
        if n > 0 {
            if let Err(e) = write(&self.con, &self.bars.rows, &self.ctxs.rows) {
                eprintln!("[xyz100] 끊길 때 남은 {n}행 쓰기 실패: {e}");
            }
        }
        (self.bars, self.ctxs) = (Ordered::default(), Ordered::default());
        n
    }

    /// REST 메움(:71-104) -- 종목마다: 마지막 저장 봉(포함)부터 지금까지. 한 종목이 실패해도 로그만 남기고 다음 종목으로.
    /// 반환 = 보낸 요청 본문(시험용).
    pub fn backfill(&self, agent: &ureq::Agent, url: &str, wall: f64) -> Vec<String> {
        let mut sent = vec![];
        for coin in &self.coins {
            let r = (|| -> Result<usize, String> {
                let last: Option<i64> = self.con.query_row("SELECT max(ts_sec) FROM hl_xyz_1m WHERE coin = ?", [coin], |r| r.get(0)).map_err(|e| e.to_string())?;
                let now_ms = crate::ms(wall);
                let start = (now_ms - BACKFILL_H * 3_600_000).max(last.unwrap_or(0) * 1000);
                let body = format!(r#"{{"type": "candleSnapshot", "req": {{"coin": {}, "interval": "1m", "startTime": {start}, "endTime": {now_ms}}}}}"#,
                                   Value::from(coin.as_str()));
                sent.push(body.clone());
                let mut resp = agent.post(url).header("Content-Type", "application/json").send(body.as_str()).map_err(|e| format!("URLError: {e}"))?;
                let st = resp.status().as_u16();
                if !(200..300).contains(&st) {
                    return Err(format!("HTTPError: HTTP Error {st}"));
                }
                let text = resp.body_mut().read_to_string().map_err(|e| format!("IncompleteRead: {e}"))?;
                let v: Value = serde_json::from_str(&text).map_err(|e| format!("JSONDecodeError: {e}"))?;
                // for c in json.loads(...): candle_row(c, coin) -- 원소는 사전이어야(.get) · 빈 목록·사전·글자는 0봉 · 그 밖은 TypeError
                let items: Vec<&Value> = match &v {
                    Value::Array(a) => a.iter().collect(),
                    Value::Object(o) if o.is_empty() => vec![],
                    Value::String(s) if s.is_empty() => vec![],
                    Value::Object(_) | Value::String(_) => return Err("AttributeError: 'str' object has no attribute 'get'".into()),
                    _ => return Err("TypeError: object is not iterable".into()),
                };
                let mut rows = vec![];
                for c in items {
                    let Value::Object(c) = c else { return Err("AttributeError: .get".into()) };
                    rows.extend(candle_row(c, Some(coin)));
                }
                for r in &rows {
                    key(&r.0, r.1).map_err(|e| e.0)?; // 목록·사전 종목은 바인딩 오류(원본 InterfaceError)
                }
                write(&self.con, &rows, &[]).map_err(|e| e.to_string())?;
                Ok(rows.len())
            })();
            match r {
                Ok(n) => eprintln!("[xyz100] 접속 · {coin} REST 메움 {n}봉"),
                Err(e) => eprintln!("[xyz100] {coin} REST 메움 실패 {e} -- 웹소켓은 그대로 붙는다"),
            }
        }
        sent
    }
}

/// 정답지 입력 먹이기. 사건 = [초(단조=벽), 종류, 본문]: 'o' 접속 · 'f' 접속 실패 · 't' 글자 · 'w' 30초 수신 시한 · 'c' 글자 아닌 메시지.
/// 반환 = 보낸 글자(구독·ping) 차례.
pub fn replay(events: &[Value], coins: &[String], path: &Path) -> rusqlite::Result<Vec<String>> {
    let mut c = XyzCollector::new(open_db(path)?, coins);
    let mut sent = vec![];
    for e in events {
        let (t, kind, body) = (e[0].as_f64().unwrap(), e[1].as_str().unwrap(), e[2].as_str().unwrap_or(""));
        let r = match kind {
            "o" => {
                sent.extend(c.on_open(t));
                Ok(())
            }
            "t" => c.on_text(t, t, body),
            "w" => c.on_timeout().map(|_| sent.push(PING.into())),
            _ => Err(Reconnect(body.into())), // 'c' · 'f'
        };
        if r.is_err() {
            c.on_error();
        }
    }
    Ok(sent)
}
