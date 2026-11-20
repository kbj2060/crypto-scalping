//! 바이낸스 체결 테이프 수집기(`scripts/live_trade_tape_collector_20260916.py`)의 «웹소켓 메시지 → 행 → SQLite» 몫.
//! 조건: 같은 입력 → 디스크에 같은 행(표·인덱스·PRAGMA 글자 그대로 · INSERT OR REPLACE · 5초 플러시 · gaps).
//! 정답지 대조 = tests/tape_parity.rs (tools/gen_tape_fixture.py 가 원본 collect() 로 만든다).
//!
//! 옮기지 않은 것(REST): `verify_recent`(5분마다 1분봉 대조 → verify_1m) · `backfill_loop`(60초마다 aggTrades 로 분 교체
//! → backfill_1m) -- 표는 만들되 쓰지 않는다. 코어가 REST 를 부를지 따로 정한다. 정지 감시(_watchdog)·재연결 대기 3초·
//! 웹소켓 접속 자체도 여기 없다: 호출하는 쪽이 `on_connect`/`on_text`/`on_disconnect` 를 부른다.
pub mod tape;

use rusqlite::{params, Connection};
use serde_json::Value;
use std::path::Path;
use std::rc::Rc;
use tape::{bucket, size_bands, TakerOrderAggregator, TapeBuffer};

pub const FLUSH_SECONDS: f64 = 5.0; // 완결된 초만 쓴다 -- 한 초는 정확히 한 번 기록된다(:104)
pub const PENDING_CAP: usize = 50_000; // ~3분치. 이보다 밀리면 버리며 말한다(:484)
pub const SCHEMA_VERSION: i64 = 2;

// ── 표(:405-436) ─ 글자 그대로. sqlite_master 의 SQL 문자열을 정답지와 대조한다 ─────────────────────
const DDL_TAPE: &str = "CREATE TABLE IF NOT EXISTS trade_tape_1s(symbol TEXT, ts_sec INTEGER, price_bin INTEGER, buy_qty REAL, sell_qty REAL, buy_n INTEGER, sell_n INTEGER, buy_max REAL, sell_max REAL, retail_buy_qty REAL, retail_sell_qty REAL, whale_buy_qty REAL, whale_sell_qty REAL, whale_buy_n INTEGER, whale_sell_n INTEGER, retail_buy_n INTEGER, retail_sell_n INTEGER, order_buy_n INTEGER, order_sell_n INTEGER, PRIMARY KEY (symbol, ts_sec, price_bin)) WITHOUT ROWID";
const DDL_REST: [&str; 7] = [
    "CREATE TABLE IF NOT EXISTS gaps(symbol TEXT, from_ms INTEGER, to_ms INTEGER, reason TEXT)",
    "CREATE TABLE IF NOT EXISTS verify_1m(symbol TEXT, ts_min INTEGER, tape_qty REAL, kline_qty REAL, rel_err REAL, checked_at TEXT)",
    "CREATE TABLE IF NOT EXISTS backfill_1m(symbol TEXT, ts_min INTEGER, source TEXT, n_rows INTEGER, tape_qty REAL, kline_qty REAL, rel_err REAL, ok INTEGER, attempts INTEGER, note TEXT, done_at TEXT)",
    "CREATE TABLE IF NOT EXISTS meta(key TEXT, value TEXT)",
    "CREATE INDEX IF NOT EXISTS verify_1m_sym_min ON verify_1m(symbol, ts_min)",
    "CREATE INDEX IF NOT EXISTS gaps_sym_to ON gaps(symbol, to_ms)",
    "CREATE INDEX IF NOT EXISTS backfill_1m_sym_min ON backfill_1m(symbol, ts_min)",
];
// 풋프린트(5분봉 × 0.5달러빈)는 이 표를 접으면 나온다 -- 따로 쌓지 않는다(2026-09-19 사용자 «풋프린트도 duckdb 에»).
// 원본은 TapeStore 의 DuckDB 용 DDL(:499-558)을 SQLite 에도 그대로 돌리고, 그중 SQLite 에 실제로 남는 것은 이 뷰뿐이다
// (표는 _init_sqlite 가 먼저 만들어 «이미 있음», ALTER 는 «이미 있는 열»로 실패해 지나간다). 줄바꿈·들여쓰기까지 원본 글자.
// ponytail: 그 무효 DDL 은 안 돌린다 -- 표가 없는 파일에서 원본과 갈라질 일은 없다(표를 먼저 만든다).
const DDL_VIEW: &str = "
                CREATE VIEW IF NOT EXISTS footprint_5m AS
                SELECT symbol,
                       ts_sec - (ts_sec % 300)                     AS bar_start,
                       CAST(round(price_bin / 5.0) AS INTEGER)     AS price_bin_5,
                       round(price_bin / 5.0) * 0.5                AS price,
                       sum(buy_qty)          AS buy_qty,
                       sum(sell_qty)         AS sell_qty,
                       sum(whale_buy_qty)    AS whale_buy_qty,
                       sum(whale_sell_qty)   AS whale_sell_qty,
                       sum(retail_buy_qty)   AS retail_buy_qty,
                       sum(retail_sell_qty)  AS retail_sell_qty,
                       sum(buy_n)            AS fill_buy_n,
                       sum(sell_n)           AS fill_sell_n,
                       sum(order_buy_n)      AS order_buy_n,
                       sum(order_sell_n)     AS order_sell_n
                FROM trade_tape_1s
                GROUP BY 1, 2, 3, 4";
const INSERT: &str = "INSERT OR REPLACE INTO trade_tape_1s VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";

/// hot 파일 열기 = 원본 `_init_sqlite`(:415-436, 따로 연 연결) + `sqlite_shared`(:445-467, 붙들고 쓰는 연결).
/// auto_vacuum 은 새 파일에서만, WAL 전환보다 먼저(뒤면 조용히 0). 같은 파일에 쓰는 심볼은 이 연결 하나를 나눠 쓴다.
pub fn open_db(path: &Path) -> rusqlite::Result<Rc<Connection>> {
    if let Some(d) = path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    {
        let con = Connection::open(path)?;
        con.busy_timeout(std::time::Duration::from_secs(30))?;
        con.execute_batch("PRAGMA auto_vacuum = INCREMENTAL")?;
        con.query_row("PRAGMA journal_mode = WAL", [], |_| Ok(()))?;
        con.execute(DDL_TAPE, [])?;
        for ddl in DDL_REST {
            con.execute(ddl, [])?;
        }
    }
    // 쓰기마다 열고 닫으면 close() 가 체크포인트를 돌려 1GB 파일에서 10분 넘게 걸렸다(2026-11-02 09:05 스택) -- 붙든다.
    // ponytail: 체크포인트는 SQLite 자동(1000쪽) 몫 -- WAL 이 커지면 따로 TRUNCATE(원본과 같다).
    let con = Connection::open(path)?;
    con.busy_timeout(std::time::Duration::from_secs(30))?;
    con.execute_batch("PRAGMA synchronous = NORMAL")?; // data_store.rw_connect 와 같은 이유(WAL 이라 깨지지 않는다)
    Ok(Rc::new(con))
}

/// 원본 repr(float). ponytail: 1e-5 미만·1e16 이상에서 파이썬은 `1e-05` 꼴, Rust 는 `1e-5` -- 칸·경계 값엔 그런 수가 없다.
fn py_repr(x: f64) -> String {
    format!("{x:?}")
}

/// `TapeStore` 의 hot(.sqlite) 몫(:470-688). DuckDB 몫(OKX·옛 현물)은 옮기지 않는다.
pub struct TapeStore {
    con: Rc<Connection>,
    pub symbol: String,
    pub pending: Vec<(i64, i64, tape::Cell)>,
}

impl TapeStore {
    pub fn new(con: Rc<Connection>, symbol: &str, bucket: f64) -> rusqlite::Result<Self> {
        con.execute(DDL_VIEW, [])?;
        // 경계는 데이터와 함께 남는다(:559). hot 은 여러 코인 한 파일이라 열쇠에 심볼(2026-09-27).
        let (lo, hi) = size_bands(symbol);
        let k = |key: &str| format!("{key}:{symbol}");
        let old: Option<String> = con
            .query_row("SELECT value FROM meta WHERE key = ?", [k("whale_min_usd")], |r| r.get(0))
            .ok();
        if let Some(old) = old {
            if old.trim().parse::<f64>().ok() != Some(hi) {
                // 경계가 바뀐 순간 -- 이 앞 행은 옛 경계다
                let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
                con.execute(
                    "INSERT INTO meta VALUES (?, ?)",
                    params![format!("size_bands_changed:{symbol}"), format!("{now:.0} whale {old} -> {}", py_repr(hi))],
                )?;
            }
        }
        for (key, value) in [
            ("schema_version".to_string(), SCHEMA_VERSION.to_string()),
            (format!("bucket:{symbol}"), py_repr(bucket)),
            (k("retail_max_usd"), py_repr(lo)),
            (k("whale_min_usd"), py_repr(hi)),
        ] {
            con.execute("DELETE FROM meta WHERE key = ?", [&key])?;
            con.execute("INSERT INTO meta VALUES (?, ?)", params![key, value])?;
        }
        Ok(Self { con, symbol: symbol.to_string(), pending: Vec::new() })
    }

    /// 지난 판이 남긴 끝(:651-655) -- 재시작 공백을 gaps 에 적으려고. 파이썬 `if row and row[0]` = 0 이면 없음.
    pub fn last_ts_ms(&self) -> rusqlite::Result<i64> {
        let m: Option<i64> =
            self.con.query_row("SELECT max(ts_sec) FROM trade_tape_1s WHERE symbol = ?", [&self.symbol], |r| r.get(0))?;
        Ok(match m {
            Some(s) if s != 0 => (s + 1) * 1000,
            _ => 0,
        })
    }

    /// 트랜잭션 하나로(:657-676). 🔴자동커밋이면 행마다 fsync -- 2026-09-23 서버 실측 50행 8~10초 vs 0.1~0.24초.
    /// 못 쓴 행은 보류했다 다음 주기에 다시 쓴다(조용한 유실 방지). 반환 = 이번에 쓴 행 수.
    pub fn write(&mut self, rows: Vec<(i64, i64, tape::Cell)>) -> usize {
        self.pending.extend(rows);
        if self.pending.is_empty() {
            return 0;
        }
        match self.commit_pending() {
            Ok(()) => {
                let n = self.pending.len();
                self.pending.clear();
                n
            }
            Err(e) => {
                let _ = self.con.execute_batch("ROLLBACK"); // 원본은 연결을 버린다(sqlite_shared) -- 같은 효과
                if self.pending.len() > PENDING_CAP {
                    let dropped = self.pending.len() - PENDING_CAP / 2;
                    self.pending.drain(..dropped);
                    eprintln!("[{}] ⚠️쓰기가 계속 막혀 {dropped}행 버림 -- 그 구간은 벌크 zip 으로 메워야 한다", self.symbol);
                } else {
                    eprintln!("[{}] 쓰기 보류 {}행, 다음 주기 재시도: {e}", self.symbol, self.pending.len());
                }
                0
            }
        }
    }

    fn commit_pending(&self) -> rusqlite::Result<()> {
        self.con.execute_batch("BEGIN")?;
        {
            let mut st = self.con.prepare_cached(INSERT)?;
            for (sec, b, c) in &self.pending {
                // 건수 칸은 파이썬 int(c[k]) -- INTEGER 로, 나머지는 REAL 로 묶는다.
                let n = |j: usize| c[j] as i64;
                st.execute(params![
                    self.symbol, sec, b, c[0], c[1], n(2), n(3), c[4], c[5], c[6], c[7], c[8], c[9],
                    n(10), n(11), n(12), n(13), n(14), n(15)
                ])?;
            }
        }
        self.con.execute_batch("COMMIT")
    }

    /// 재연결 한 번에 1.5초 미만이면 기록할 값이 없다(:678-688). 실패해도 수집은 계속.
    pub fn record_gap(&self, from_ms: i64, to_ms: i64, reason: &str) {
        if to_ms - from_ms < 1500 {
            return;
        }
        if let Err(e) = self.con.execute("INSERT INTO gaps VALUES (?,?,?,?)", params![self.symbol, from_ms, to_ms, reason]) {
            eprintln!("[{}] gap 기록 실패(수집은 계속): {e}", self.symbol);
        }
    }
}

/// 수집 루프 한 심볼(`collect()` :923-1000)에서 소켓·REST 를 뺀 것. 시계(`now`)는 부르는 쪽이 준다(단조 초).
pub struct Collector {
    pub store: TapeStore,
    pub buffer: TapeBuffer,
    orders: TakerOrderAggregator,
    spot: bool,
    last_ms: i64,
    first: bool,
    flushed_at: f64,
}

/// 원본에서 예외가 났다 = 그 연결을 접고 다시 붙는다(`except Exception` :996). 깨진 JSON·빠진 필드·숫자 아님.
#[derive(Debug)]
pub struct Reconnect(pub String);

fn need<'a>(o: &'a serde_json::Map<String, Value>, k: &str) -> Result<&'a Value, Reconnect> {
    o.get(k).ok_or_else(|| Reconnect(format!("KeyError {k}")))
}

/// 파이썬 float(x). ponytail: 문자열의 밑줄(`1_0`)·JSON 의 NaN/Infinity 낱말은 파이썬만 받는다 -- 바이낸스는 안 보낸다.
fn py_float(v: &Value) -> Result<f64, Reconnect> {
    match v {
        Value::Number(n) => n.as_f64().ok_or_else(|| Reconnect("float".into())),
        Value::String(s) => s.trim().parse::<f64>().map_err(|_| Reconnect(format!("ValueError float {s:?}"))),
        Value::Bool(b) => Ok(*b as i64 as f64),
        _ => Err(Reconnect("TypeError float".into())),
    }
}

/// 파이썬 int(x): 정수 · 실수는 0 쪽으로 자름 · 정수 문자열.
fn py_int(v: &Value) -> Result<i64, Reconnect> {
    match v {
        Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().filter(|f| f.is_finite()).map(|f| f.trunc() as i64))
            .ok_or_else(|| Reconnect("int".into())),
        Value::String(s) => s.trim().parse::<i64>().map_err(|_| Reconnect(format!("ValueError int {s:?}"))),
        Value::Bool(b) => Ok(*b as i64),
        _ => Err(Reconnect("TypeError int".into())),
    }
}

/// 파이썬 bool(x)
fn py_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// 파이썬 int(trade.get(k) or 0)
fn py_int_or0(o: &serde_json::Map<String, Value>, k: &str) -> Result<i64, Reconnect> {
    match o.get(k) {
        Some(v) if py_truthy(v) => py_int(v),
        _ => Ok(0),
    }
}

impl Collector {
    /// market = "futures"(@trade, 되묶기) | "spot"(@aggTrade, 이미 주문 단위 -- 2026-10-01 저장 재설계 4d).
    pub fn new(con: Rc<Connection>, symbol: &str, market: &str, now: f64) -> rusqlite::Result<Self> {
        let b = bucket(symbol);
        let store = TapeStore::new(con, symbol, b)?;
        let last_ms = store.last_ts_ms()?;
        Ok(Self {
            store,
            buffer: TapeBuffer::new(b, symbol),
            orders: TakerOrderAggregator::default(),
            spot: market == "spot",
            last_ms,
            first: true,
            flushed_at: now,
        })
    }

    pub fn on_connect(&mut self) {
        self.first = true;
    }

    /// 연결이 끝났다(정상 끝·글자 아닌 메시지·예외·접속 실패 전부, :999). 진행 중인 초는 버퍼에 남아 다음 연결로 간다.
    pub fn on_disconnect(&mut self) -> usize {
        let rows = self.buffer.take_closed(false);
        self.store.write(rows)
    }

    /// 글자 메시지 하나(:955-993). 평가 순서까지 원본과 같다 -- 어디서 예외가 나느냐에 따라 버퍼·공백에 남는 것이 다르다.
    /// Ok(Some(n)) = 이번에 플러시해 n행 썼다.
    pub fn on_text(&mut self, now: f64, text: &str) -> Result<Option<usize>, Reconnect> {
        let v: Value = serde_json::from_str(text).map_err(|e| Reconnect(format!("JSONDecodeError {e}")))?;
        let Value::Object(o) = &v else { return Err(Reconnect("AttributeError .get".into())) };
        let want = if self.spot { "aggTrade" } else { "trade" };
        if o.get("e").and_then(Value::as_str) != Some(want) {
            return Ok(None);
        }
        let price = py_float(need(o, "p")?)?;
        let qty = py_float(need(o, "q")?)?;
        if !(price > 0.0 && qty > 0.0) {
            // 바이낸스가 {"p":"0","q":"0","X":"NA","st":1} 를 섞어 보낸다(2026-09-16 실측 12,312건 중 37건=0.3%).
            // 합계는 안 틀리지만 가격 0 자리에 빈이 생긴다 -- 받는 쪽에서 막는다.
            return Ok(None);
        }
        let ts_ms = py_int(need(o, "T")?)?;
        if self.first {
            self.first = false;
            // 처음 켠 판이면 그 «분의 시작부터 여기까지»가 안 받은 구간이다 -- 안 적으면 완전성 검사가 유실로 오해한다.
            self.orders.reset(); // 끊김 전에 열려 있던 묶음은 버린다
            let (from, why) =
                if self.last_ms != 0 { (self.last_ms, "ws_reconnect") } else { (ts_ms.div_euclid(60_000) * 60_000, "startup") };
            self.store.record_gap(from, ts_ms, why);
        }
        let sell = py_truthy(need(o, "m")?);
        if self.spot {
            // aggTrade = 이미 테이커 주문 하나(대시보드 현물 레인과 같은 규칙). 체결 수 = l-f+1
            let fills = (py_int_or0(o, "l")? - py_int_or0(o, "f")? + 1).max(1);
            self.buffer.add_agg(ts_ms, price, qty, sell, fills);
        } else {
            self.buffer.add(ts_ms, price, qty, sell);
            let tid = match o.get("t") {
                None | Some(Value::Null) => None,
                Some(t) => Some(py_int(t)?), // 칸엔 이미 들어갔다 -- 원본도 여기서 예외면 묶음만 빠진다
            };
            if let Some((p, q, ms, s)) = self.orders.add(price, qty, ts_ms, sell, tid) {
                self.buffer.add_order(ms, p, q, s);
            }
        }
        self.last_ms = ts_ms;
        if now - self.flushed_at >= FLUSH_SECONDS {
            self.flushed_at = now;
            let rows = self.buffer.take_closed(false);
            return Ok(Some(self.store.write(rows)));
        }
        // 원본은 여기서 VERIFY_SECONDS(300초)마다 verify_recent(REST 1분봉 대조)를 부른다 -- 이번 범위 밖.
        Ok(None)
    }
}

/// 정답지 입력을 원본과 같은 연결 규칙으로 먹인다(tools/gen_tape_fixture.py 의 가짜 웹소켓과 같은 뜻).
/// 사건 = [recv, "t"|"c"|"f", 본문]: 't' 글자 메시지 · 'c' 글자 아닌 메시지(연결 끝) · 'f' 접속 실패(연결 사이에만).
/// 입력이 끝나면 다음 접속에서 멈춘다(마지막 플러시 없음 -- 파이썬 쪽도 CancelledError 로 끝난다).
/// `on_flush(recv, n)` = 플러시마다(속도 측정용).
pub fn replay(runs: &[Value], dir: &Path, mut on_flush: impl FnMut(f64, usize)) -> rusqlite::Result<()> {
    let mut dbs: std::collections::HashMap<&str, Rc<Connection>> = Default::default();
    for run in runs {
        let market = run["market"].as_str().unwrap();
        let file = if market == "spot" { "binance_spot_tape.sqlite" } else { "binance_tape.sqlite" };
        let con = match dbs.get(file) {
            Some(c) => c.clone(),
            None => dbs.entry(file).or_insert(open_db(&dir.join(file))?).clone(),
        };
        let mut c = Collector::new(con, run["symbol"].as_str().unwrap(), market, run["t0"].as_f64().unwrap())?;
        let ev = run["events"].as_array().unwrap();
        let mut i = 0;
        while i < ev.len() {
            if ev[i][1] == "f" {
                i += 1;
                c.on_disconnect();
                continue;
            }
            c.on_connect();
            while i < ev.len() {
                match ev[i][1].as_str().unwrap() {
                    "f" => break,
                    "c" => {
                        i += 1;
                        break;
                    }
                    _ => {
                        let now = ev[i][0].as_f64().unwrap();
                        i += 1;
                        match c.on_text(now, ev[i - 1][2].as_str().unwrap()) {
                            Ok(Some(n)) => on_flush(now, n),
                            Ok(None) => {}
                            Err(_) => break,
                        }
                    }
                }
            }
            c.on_disconnect();
        }
    }
    Ok(())
}
