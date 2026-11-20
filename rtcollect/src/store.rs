//! 체결 테이프 hot SQLite(바이낸스·Bybit·OKX 공용) = `TapeStore`·`_init_sqlite`·`sqlite_shared`
//! (scripts/live_trade_tape_collector_20260916.py :405-688, 줄 번호는 그 파일). 표·인덱스·뷰·PRAGMA 글자 그대로 · INSERT OR REPLACE.
use crate::py_repr;
use crate::tape::{size_bands, Cell};
use rusqlite::{params, Connection};
use std::path::Path;
use std::rc::Rc;

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

/// `TapeStore` 의 hot(.sqlite) 몫(:470-688). DuckDB 몫(옛 OKX·현물 파일)은 옮기지 않는다.
pub struct TapeStore {
    con: Rc<Connection>,
    pub symbol: String,
    pub pending: Vec<(i64, i64, Cell)>,
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
        let s = Self { con, symbol: symbol.to_string(), pending: Vec::new() };
        s.put_meta("schema_version", &SCHEMA_VERSION.to_string())?;
        s.put_meta(&format!("bucket:{symbol}"), &py_repr(bucket))?;
        s.put_meta(&k("retail_max_usd"), &py_repr(lo))?;
        s.put_meta(&k("whale_min_usd"), &py_repr(hi))?;
        Ok(s)
    }

    /// meta 열쇠 하나 갈아 끼우기(DELETE → INSERT, 자동커밋) -- 원본 :571-572 · Bybit·OKX `_write_meta` 와 같은 두 문장.
    pub fn put_meta(&self, key: &str, value: &str) -> rusqlite::Result<()> {
        self.con.execute("DELETE FROM meta WHERE key = ?", [key])?;
        self.con.execute("INSERT INTO meta VALUES (?, ?)", params![key, value])?;
        Ok(())
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
    pub fn write(&mut self, rows: Vec<(i64, i64, Cell)>) -> usize {
        self.pending.extend(rows);
        if self.pending.is_empty() {
            return 0;
        }
        match self.commit_pending() {
            Ok(()) => {
                let n = self.pending.len();
                self.pending.clear();
                crate::wrote();
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
