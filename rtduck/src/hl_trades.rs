//! 하이퍼리퀴드 주소단위 체결(`scripts/live_hyperliquid_trade_collector_20260916.py`, 줄 번호는 그 파일) → UTC 날짜별 DuckDB
//! `<HL_ROOT>/<코인_코인>/<YYYY-MM-DD>.duckdb`(표 hl_trades · gaps). 집계하지 않는다(주소가 이 축의 전부).
//! 날짜가 넘어가면 **3일 전** 파일을 gzip(포지션 수집기가 최근 3개 날짜를 .duckdb 로 읽는다) -- 성공한 뒤에만 지운다.
//! 쓸 때마다 열고 닫는다(붙들면 감시기·포지션 수집기 universe 가 막힌다). 연결이 끊기면 버퍼를 쓰고 공백을 들고 있다가 다시 붙을 때 gaps 에.
use duckdb::{params, Connection};
use rtcollect::Reconnect;
use serde_json::Value;
use std::path::{Path, PathBuf};

pub const WS: &str = "wss://api.hyperliquid.xyz/ws";
pub const FLUSH_N: usize = 500;
pub const FLUSH_SEC: f64 = 20.0;
pub const KEEP_ON_FAIL: usize = 200_000;

/// 수집기 벽시계. 시험 전용 `RTDUCK_FAKE_START`(epoch 초)가 있으면 «그 시각에 시작한 것처럼» 민다(자정 넘기 시험) -- 운영에서는 비워 둔다.
pub fn wall() -> f64 {
    // 시험(RTDUCK_FAKE_START)이면 «시작 + 단조 경과» -- 벽시계 + 차이로 하면 WSL 벽시계가 1.5~2.4초씩 뒤로 뛸 때 자정 넘기가 밀려
    //   hl_trades_loop 가 부하 아래 20번 중 2번 공백 줄을 그 전날 파일에 썼다(2026-11-21 아홉째 조각). 실제 실행은 원본처럼 벽시계 그대로.
    static FAKE: std::sync::OnceLock<Option<(std::time::Instant, f64)>> = std::sync::OnceLock::new();
    match FAKE.get_or_init(|| std::env::var("RTDUCK_FAKE_START").ok().and_then(|v| v.parse::<f64>().ok()).map(|t| (std::time::Instant::now(), t))) {
        Some((i0, t)) => t + i0.elapsed().as_secs_f64(),
        None => rtcollect::rest::wall_s(),
    }
}

/// (coin, ts_ms, tid, hash, side, px, sz, buyer, seller)
pub type Trade = (String, i64, i64, String, String, f64, f64, String, String);

/// 파이썬 str(x). ponytail: 목록·사전은 JSON 글자(파이썬 repr 과 다르다) -- 거래소는 그 칸에 글자·숫자만 보낸다.
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => n.as_i64().map_or_else(|| rtcollect::py_float_repr(n.as_f64().unwrap_or(f64::NAN)), |i| i.to_string()),
        v => v.to_string(),
    }
}

fn py_float(v: &Value) -> Result<f64, Reconnect> {
    match v {
        Value::Number(n) => n.as_f64().ok_or_else(|| Reconnect("OverflowError".into())),
        Value::String(s) => s.trim().parse::<f64>().map_err(|_| Reconnect("ValueError".into())),
        Value::Bool(b) => Ok(*b as u8 as f64),
        _ => Err(Reconnect("TypeError".into())),
    }
}

fn py_int(v: &Value) -> Result<i64, Reconnect> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().filter(|f| f.is_finite()).map(|f| f.trunc() as i64)).ok_or_else(|| Reconnect("ValueError".into())),
        Value::String(s) => s.trim().parse::<i64>().map_err(|_| Reconnect("ValueError".into())),
        Value::Bool(b) => Ok(*b as i64),
        _ => Err(Reconnect("TypeError".into())),
    }
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

/// `parse`(:80-92). Err = 원본 예외(연결을 접는다 -- 이 메시지의 행은 버퍼에 못 들어간다).
pub fn parse(msg: &Value) -> Result<Vec<Trade>, Reconnect> {
    let Value::Object(m) = msg else { return Err(Reconnect("AttributeError".into())) };
    if m.get("channel").and_then(Value::as_str) != Some("trades") {
        return Ok(vec![]);
    }
    let data = match m.get("data").filter(|v| truthy(v)) {
        None => return Ok(vec![]),
        Some(Value::Array(a)) => a,
        Some(Value::Object(_) | Value::String(_)) => return Err(Reconnect("AttributeError".into())), // 열쇠·글자를 돌며 .get
        Some(_) => return Err(Reconnect("TypeError".into())),
    };
    let mut out = vec![];
    for t in data {
        let Value::Object(t) = t else { return Err(Reconnect("AttributeError".into())) };
        let users: Vec<Value> = match t.get("users").filter(|v| truthy(v)) {
            None => vec![],
            Some(Value::Array(a)) => a.clone(),
            Some(Value::String(s)) => s.chars().map(|c| Value::from(c.to_string())).collect(),
            Some(Value::Object(o)) if o.len() != 2 => vec![],
            Some(Value::Object(_)) => return Err(Reconnect("KeyError".into())),
            Some(_) => return Err(Reconnect("TypeError".into())),
        };
        if users.len() != 2 {
            continue;
        }
        let need = |k: &str| t.get(k).ok_or_else(|| Reconnect("KeyError".into()));
        out.push((py_str(need("coin")?), py_int(need("time")?)?, py_int(need("tid")?)?, t.get("hash").map_or_else(String::new, py_str),
                  py_str(need("side")?), py_float(need("px")?)?, py_float(need("sz")?)?, py_str(&users[0]).to_lowercase(), py_str(&users[1]).to_lowercase()));
    }
    Ok(out)
}

/// UTC 날짜 글자(YYYY-MM-DD).
fn day(wall: f64) -> String {
    rtcollect::depth::hour_key((wall.floor() as i64) * 1000).unwrap_or_default()[..10].to_string()
}

pub struct HlTrades {
    pub root: PathBuf,
    pub coins: Vec<String>,
    pub buf: Vec<Trade>,
    last: f64,
    down: Option<(i64, String)>,
    resume_ms: i64,
    cut: i64,
    last_db: Option<PathBuf>,
    /// 원본 «WS 끊김» 의 type(e).__name__ -- 글자 처리에서 난 예외면 그 이름, 아니면(서버가 닫음·접속 실패) 부르는 쪽이 준다.
    err: Option<String>,
}

impl HlTrades {
    pub fn new(root: &Path, coins: &[String], wall: f64) -> Self {
        Self { root: root.into(), coins: coins.to_vec(), buf: vec![], last: wall, down: None, resume_ms: 0, cut: 0, last_db: None, err: None }
    }

    pub fn db_path(&self, wall: f64) -> PathBuf {
        self.root.join(self.coins.join("_")).join(format!("{}.duckdb", day(wall)))
    }

    /// `write`(:105-128): 그날 파일에 한 트랜잭션. 날짜가 바뀌었으면 3일 전 파일을 먼저 압축.
    pub fn write(&mut self, rows: &[Trade], gap: Option<(String, i64, i64, String)>, wall: f64) -> duckdb::Result<()> {
        let db = self.db_path(wall);
        if self.last_db.as_ref().is_some_and(|l| *l != db) {
            let old = self.db_path(wall - 3.0 * 86400.0);
            let gz = PathBuf::from(format!("{}.gz", old.display()));
            if !gz.exists() && old.exists() {
                rtcollect::depth::gzip_done(&old);
            }
        }
        self.last_db = Some(db.clone());
        if let Some(d) = db.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let mut con = Connection::open(&db)?;
        con.execute_batch("CREATE TABLE IF NOT EXISTS hl_trades (
            coin VARCHAR, ts_ms BIGINT, tid BIGINT, hash VARCHAR, side VARCHAR,
            px DOUBLE, sz DOUBLE, buyer VARCHAR, seller VARCHAR)")?;
        con.execute_batch("CREATE TABLE IF NOT EXISTS gaps (coin VARCHAR, from_ms BIGINT, to_ms BIGINT, reason VARCHAR)")?;
        let tx = con.transaction()?;
        {
            let mut st = tx.prepare("INSERT INTO hl_trades VALUES (?,?,?,?,?,?,?,?,?)")?;
            for r in rows {
                st.execute(params![r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7, r.8])?;
            }
            if let Some(g) = gap {
                tx.execute("INSERT INTO gaps VALUES (?,?,?,?)", params![g.0, g.1, g.2, g.3])?;
            }
        }
        tx.commit()
    }

    /// 접속·구독 직후(:140-153): 구독 글자들 · 들고 있던 공백을 쓴다(실패해도 연결은 산다) · cut = 지난 연결의 최신 체결 시각.
    pub fn on_open(&mut self, wall: f64) -> Vec<String> {
        let subs = self.coins.iter().map(|c| format!(r#"{{"method": "subscribe", "subscription": {{"type": "trades", "coin": {}}}}}"#, Value::from(c.as_str()))).collect();
        if let Some((from, why)) = self.down.take() {
            let g = (self.coins.join(","), from, (wall * 1000.0) as i64, why);
            if let Err(e) = self.write(&[], Some(g), wall) {
                eprintln!("  gap 기록 실패: {e}");
            }
        }
        self.cut = self.resume_ms;
        subs
    }

    /// 글자 메시지(:160-168). Err = 연결을 접는다(깨진 JSON 포함).
    pub fn on_text(&mut self, wall: f64, raw: &str) -> Result<(), Reconnect> {
        let r = (|| {
            let v: Value = serde_json::from_str(raw).map_err(|_| Reconnect("JSONDecodeError".into()))?;
            parse(&v)
        })();
        let rows = match r {
            Ok(x) => x,
            Err(e) => {
                self.err = Some(e.0.clone());
                return Err(e);
            }
        };
        // new_rows(:95-99): cut 이하는 재구독 스냅숏 중복
        let rows: Vec<Trade> = rows.into_iter().filter(|r| r.1 > self.cut).collect();
        self.resume_ms = rows.iter().map(|r| r.1).fold(self.resume_ms, i64::max);
        self.buf.extend(rows);
        if self.buf.len() >= FLUSH_N || (!self.buf.is_empty() && wall - self.last >= FLUSH_SEC) {
            let b = std::mem::take(&mut self.buf);
            match self.write(&b, None, wall) {
                Ok(()) => println!("  +{}행", b.len()),
                Err(e) => {
                    println!("  기록 보류 {}행: {e}", b.len()); // 읽는 쪽 락 -- 버퍼는 다음 주기에
                    self.buf = b;
                }
            }
            self.last = wall;
        }
        Ok(())
    }

    /// 연결 끝(:169-179): 처음 끊긴 시각·이유만 잡고 남은 버퍼를 쓴다(못 쓰면 뒤 20만 행만 남긴다). 반환 = 쉴 초.
    pub fn on_fail(&mut self, wall: f64, why: &str) -> f64 {
        let name = self.err.take().unwrap_or_else(|| why.to_string());
        if self.down.is_none() {
            self.down = Some(((wall * 1000.0) as i64, name));
        }
        let b = std::mem::take(&mut self.buf);
        if let Err(e) = self.write(&b, None, wall) {
            println!("  잔여 {}행 기록 보류: {e}", b.len());
            self.buf = b;
            let n = self.buf.len();
            if n > KEEP_ON_FAIL {
                self.buf.drain(..n - KEEP_ON_FAIL);
            }
        }
        5.0
    }
}

/// 정답지 입력 먹이기. 사건 = [벽시계 초, 종류, 본문]: 'o' 접속 · 'f' 접속 실패(본문 = «이름: 글») · 't' 글자 · 'w' 30초 시한(ping) · 'c' 끊김(본문 = «이름: 글»).
/// 반환 = 보낸 글자.
pub fn replay(inp: &Value, root: &Path) -> Vec<String> {
    let coins: Vec<String> = inp["coins"].as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect();
    let mut h = HlTrades::new(root, &coins, inp["t0"].as_f64().unwrap());
    let mut sent = vec![];
    for e in inp["events"].as_array().unwrap() {
        let (t, kind, body) = (e[0].as_f64().unwrap(), e[1].as_str().unwrap(), e[2].as_str().unwrap_or(""));
        let name = body.split(": ").next().unwrap_or("").to_string();
        match kind {
            "o" => sent.extend(h.on_open(t)),
            "t" => {
                if h.on_text(t, body).is_err() {
                    h.on_fail(t, "");
                }
            }
            "w" => sent.push(rtcollect::xyz::PING.into()),
            _ => {
                h.on_fail(t, &name);
            }
        }
    }
    sent
}

/// rtcollect 의 하이퍼리퀴드 접속 뼈대(net::run_hl)에 붙인다: 접속 시한 10초(websockets open_timeout 기본) · 30초 무음 = ping · 끊기면 5초.
/// ponytail: 살아 있음 확인은 그 뼈대의 25초 핑(원본 websockets 는 20초 핑·20초 시한) -- 끊김 판정이 몇 초 늦을 수 있다.
impl rtcollect::net::Hl for HlTrades {
    const CONNECT: std::time::Duration = std::time::Duration::from_secs(10);
    fn open(&mut self, _now: f64) -> Vec<String> {
        self.on_open(wall())
    }
    fn text(&mut self, _now: f64, _wall: f64, t: &str) -> Result<(), Reconnect> {
        self.on_text(wall(), t)
    }
    fn idle(&mut self) -> Result<(), Reconnect> {
        Ok(())
    }
    fn fail(&mut self) -> f64 {
        // 서버가 닫음·접속 실패의 이름은 뼈대가 넘겨주지 않는다 -- ConnectionClosedError 로 적는다(글자 처리 예외면 그 이름)
        self.on_fail(wall(), "ConnectionClosedError")
    }
}
