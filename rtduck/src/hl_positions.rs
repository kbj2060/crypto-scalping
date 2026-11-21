//! 하이퍼리퀴드 고래 포지션(`scripts/live_hyperliquid_positions_collector_20260924.py`, 줄 번호는 그 파일) → hot SQLite.
//! 대상 = 최근 48시간 HL 체결(rtduck hl_trades·원본 체결 수집기의 날짜별 DuckDB를 읽기 전용으로) 거래액 상위 주소(코인별 300 · 합집합) · 6시간마다 다시 뽑는다.
//! 바퀴마다 주소별 `clearinghouseState`(0.4초 간격) → hl_positions · 줄었거나 사라진 주소 중 그 사이 1분봉 범위가 직전 청산가 ±0.5% 를
//! 지난 것만 `userFillsByTime` → 본인이 청산당한 체결만 hl_liquidations · 바퀴 요약 hl_cycles. 읽는 곳: 서버 HL 고래 청산 원(hl_whale_liq_events)
//! · 시장 맥락 hl_pos(마지막으로 다 돈 바퀴) · seal_to_lake.
use rusqlite::{params, Connection};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

pub const INFO_URL: &str = "https://api.hyperliquid.xyz/info";
pub const UNIVERSE_N: usize = 300;
pub const UNIVERSE_HOURS: f64 = 48.0;
pub const UNIVERSE_REFRESH_S: f64 = 6.0 * 3600.0;
pub const REQ_GAP_S: f64 = 0.4;
pub const LIQ_TOL: f64 = 0.005;

// :72-87 -- 글자 그대로(SQLite 는 CREATE 글자를 sqlite_master 에 남긴다)
const DDL: [&str; 4] = [
    "CREATE TABLE IF NOT EXISTS hl_positions(
         ts_ms BIGINT, ex_ms BIGINT, user VARCHAR, coin VARCHAR, szi DOUBLE, entry_px DOUBLE,
         liq_px DOUBLE, leverage DOUBLE, lev_type VARCHAR, margin_used DOUBLE, position_value DOUBLE,
         unrealized_pnl DOUBLE, cum_funding DOUBLE, account_value DOUBLE)",
    "CREATE TABLE IF NOT EXISTS hl_universe(
         ts_ms BIGINT, user VARCHAR, rank INTEGER, notional_48h DOUBLE)",
    "CREATE TABLE IF NOT EXISTS hl_liquidations(
         detected_ms BIGINT, tid BIGINT, user VARCHAR, coin VARCHAR, fill_ms BIGINT, px DOUBLE,
         sz DOUBLE, side VARCHAR, dir VARCHAR, start_position DOUBLE, closed_pnl DOUBLE,
         mark_px DOUBLE, method VARCHAR)",
    "CREATE TABLE IF NOT EXISTS hl_cycles(
         ts_ms BIGINT, n_users INTEGER, n_ok INTEGER, n_pos INTEGER, seconds DOUBLE)",
];
const INDEXES: [&str; 4] = ["CREATE INDEX IF NOT EXISTS hl_positions_i ON hl_positions(coin, ts_ms)",
                            "CREATE INDEX IF NOT EXISTS hl_liquidations_i ON hl_liquidations(coin, fill_ms)",
                            "CREATE INDEX IF NOT EXISTS hl_cycles_i ON hl_cycles(ts_ms)",
                            "CREATE INDEX IF NOT EXISTS hl_universe_i ON hl_universe(ts_ms)"];

/// 파일 이름(:61-63): ETH 혼자면 hl_positions.sqlite · 아니면 hl_positions_<코인_코인>.sqlite.
pub fn default_db(coins: &[String]) -> String {
    if coins == ["ETH"] { "hl_positions.sqlite".into() } else { format!("hl_positions_{}.sqlite", coins.iter().map(|c| c.to_lowercase()).collect::<Vec<_>>().join("_")) }
}

/// `init_db`(:246-265): sqlite_init(auto_vacuum·WAL) → DDL → ALTER(이미 있으면 오류를 삼킨다) → 인덱스.
pub fn init_db(db: &Path) -> rusqlite::Result<Connection> {
    let con = rtcollect::store::open_hot(db, &DDL[..2])?;
    let _ = con.execute("ALTER TABLE hl_universe ADD COLUMN coin VARCHAR", []);
    for d in DDL[2..].iter().chain(&INDEXES) {
        con.execute(d, [])?;
    }
    Ok(con)
}

/// 파이썬 `_f`: float(v), TypeError·ValueError 면 None.
fn f(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        Value::Bool(b) => Some(*b as u8 as f64),
        _ => None,
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

/// `x.get(k) or {}` 다음에 `.get` 을 부르는 모양 -- 참인 비사전이면 AttributeError.
fn sub<'a>(m: &'a Map<String, Value>, k: &str, empty: &'a Map<String, Value>) -> Result<&'a Map<String, Value>, String> {
    match m.get(k).filter(|v| truthy(v)) {
        None => Ok(empty),
        Some(Value::Object(o)) => Ok(o),
        Some(_) => Err("AttributeError".into()),
    }
}

/// 파이썬 str(x)(이 수집기에 오는 모양만: 글자·숫자·참거짓·None). ponytail: 목록·사전은 JSON 글자.
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => n.as_i64().map_or_else(|| rtcollect::py_float_repr(n.as_f64().unwrap_or(f64::NAN)), |i| i.to_string()),
        v => v.to_string(),
    }
}

/// hl_positions 한 행(14칸).
pub type Pos = (i64, Option<i64>, String, String, f64, Option<f64>, Option<f64>, Option<f64>, String, Option<f64>, Option<f64>, Option<f64>, Option<f64>, Option<f64>);
/// hl_liquidations 한 행(13칸).
pub type Liq = (i64, i64, String, String, i64, Option<f64>, Option<f64>, String, String, Option<f64>, Option<f64>, Option<f64>, String);

/// `parse_state`(:98-118). Err = 원본 예외(그 주소는 응답 수에만 들고 행은 없다 -- 원본은 ok += 1 뒤에 터진다).
pub fn parse_state(state: &Value, user: &str, coins: &[String], ts_ms: i64) -> Result<Vec<Pos>, String> {
    let Value::Object(st) = state else { return Err("AttributeError".into()) };
    let empty = Map::new();
    let acct = f(sub(st, "marginSummary", &empty)?.get("accountValue"));
    let t = st.get("time").map(py_str).unwrap_or_default();
    let ex_ms = (!t.is_empty() && t.bytes().all(|b| b.is_ascii_digit())).then(|| t.parse::<i64>().ok()).flatten();
    let mut out = vec![];
    let aps: &[Value] = match st.get("assetPositions").filter(|v| truthy(v)) {
        None => &[],
        Some(Value::Array(a)) => a,
        Some(_) => return Err("AttributeError".into()), // 사전·글자를 돌며 .get
    };
    for ap in aps {
        let Value::Object(ap) = ap else { return Err("AttributeError".into()) };
        let p = sub(ap, "position", &empty)?;
        let Some(Value::String(coin)) = p.get("coin") else { continue };
        if !coins.contains(coin) {
            continue;
        }
        let Some(szi) = f(p.get("szi")).filter(|x| *x != 0.0) else { continue };
        let lev = sub(p, "leverage", &empty)?;
        let lt = lev.get("type").filter(|v| truthy(v)).map(py_str).unwrap_or_default();
        out.push((ts_ms, ex_ms, user.to_string(), coin.clone(), szi, f(p.get("entryPx")), f(p.get("liquidationPx")), f(lev.get("value")), lt,
                  f(p.get("marginUsed")), f(p.get("positionValue")), f(p.get("unrealizedPnl")), f(sub(p, "cumFunding", &empty)?.get("sinceOpen")), acct));
    }
    Ok(out)
}

/// `liquidation_candidates`(:121-134). prev = 직전 바퀴 (주소, szi, 청산가) 순서대로.
pub fn liquidation_candidates(prev: &[(String, f64, Option<f64>)], cur: &HashMap<String, f64>, low: f64, high: f64, tol: f64) -> Vec<String> {
    let mut out = vec![];
    for (u, szi, liq) in prev {
        let Some(liq) = liq.filter(|l| *l != 0.0) else { continue };
        if *szi == 0.0 {
            continue;
        }
        let now = cur.get(u).copied().unwrap_or(0.0);
        let shrank = now * szi <= 0.0 || now.abs() < szi.abs() * 0.999;
        if shrank && low * (1.0 - tol) <= liq && liq <= high * (1.0 + tol) {
            out.push(u.clone());
        }
    }
    out
}

fn pyint0(v: Option<&Value>) -> Result<i64, String> {
    match v.filter(|v| truthy(v)) {
        None => Ok(0),
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().map(|x| x.trunc() as i64)).ok_or_else(|| "ValueError".into()),
        Some(Value::String(s)) => s.trim().parse::<i64>().map_err(|_| "ValueError".into()),
        Some(Value::Bool(b)) => Ok(*b as i64),
        Some(_) => Err("TypeError".into()),
    }
}

/// `parse_liq_fills`(:137-148): 그 주소가 청산당한 체결만(상대방 쪽 기록은 뺀다).
pub fn parse_liq_fills(fills: &Value, user: &str, coin: &str, detected_ms: i64) -> Result<Vec<Liq>, String> {
    let empty = Map::new();
    let list: &[Value] = match fills {
        Value::Array(a) => a,
        v if !truthy(v) => &[],
        _ => return Err("AttributeError".into()),
    };
    let mut rows = vec![];
    for x in list {
        let Value::Object(x) = x else { return Err("AttributeError".into()) };
        let lq = sub(x, "liquidation", &empty)?;
        if x.get("coin").and_then(Value::as_str) != Some(coin) || py_str(lq.get("liquidatedUser").unwrap_or(&Value::from(""))).to_lowercase() != user.to_lowercase() {
            continue;
        }
        let s = |k: &str| x.get(k).filter(|v| truthy(v)).map(py_str).unwrap_or_default();
        rows.push((detected_ms, pyint0(x.get("tid"))?, user.to_string(), coin.to_string(), pyint0(x.get("time"))?, f(x.get("px")), f(x.get("sz")),
                   s("side"), s("dir"), f(x.get("startPosition")), f(x.get("closedPnl")), f(lq.get("markPx")),
                   lq.get("method").filter(|v| truthy(v)).map(py_str).unwrap_or_default()));
    }
    Ok(rows)
}

/// `universe`(:151-177): 그 코인 폴더의 최근 3개 UTC 날짜 DuckDB(읽기 전용 · 잠겨 있으면 0.2초씩 25번)에서 48시간 거래액(매수+매도 쪽) 상위 n.
pub fn universe(root: &Path, now: f64, coin: &str, n: usize) -> Vec<(String, f64)> {
    let since = ((now - UNIVERSE_HOURS * 3600.0) * 1000.0) as i64;
    let days: HashSet<String> = (0..3).map(|i| rtcollect::depth::hour_key(((now - i as f64 * 86400.0).floor() as i64) * 1000).unwrap_or_default()[..10].to_string()).collect();
    let mut agg: Vec<(String, f64)> = vec![];
    let mut at: HashMap<String, usize> = HashMap::new();
    // glob 순서 = 디렉터리 순서(read_dir 와 같은 getdents 순서) -- 파일마다 합을 차례로 더한다
    let files: Vec<PathBuf> = std::fs::read_dir(root.join(coin)).map(|d| d.flatten().map(|e| e.path()).collect()).unwrap_or_default();
    for p in files {
        if p.extension().and_then(|x| x.to_str()) != Some("duckdb") || !p.file_stem().and_then(|s| s.to_str()).is_some_and(|s| days.contains(s)) {
            continue;
        }
        let mut con = None;
        for _ in 0..25 {
            match duckdb::Config::default().access_mode(duckdb::AccessMode::ReadOnly).and_then(|c| duckdb::Connection::open_with_flags(&p, c)) {
                Ok(c) => {
                    con = Some(c);
                    break;
                }
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(200)),
            }
        }
        let Some(con) = con else { continue };
        let rows: Vec<(String, f64)> = match con.prepare("
                SELECT u, sum(n) FROM (
                  SELECT buyer AS u, px * sz AS n FROM hl_trades WHERE coin = ? AND ts_ms >= ?
                  UNION ALL SELECT seller, px * sz FROM hl_trades WHERE coin = ? AND ts_ms >= ?)
                GROUP BY 1").and_then(|mut st| st.query_map(duckdb::params![coin, since, coin, since], |r| Ok((r.get(0)?, r.get(1)?)))?.collect()) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("universe {}: {e}", p.display()); // 원본은 여기서 예외가 바퀴 밖으로 나간다(드묾)
                continue;
            }
        };
        for (u, x) in rows {
            match at.get(&u) {
                Some(&i) => agg[i].1 += x,
                None => {
                    at.insert(u.clone(), agg.len());
                    agg.push((u, x));
                }
            }
        }
    }
    agg.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)); // sorted(key=-n) -- 안정 정렬
    agg.truncate(n);
    agg
}

/// `merge_universe`(:180-190): 코인을 번갈아 1위부터 · 중복 주소는 한 번.
pub fn merge_universe(ranked: &[(String, Vec<(String, f64)>)]) -> Vec<String> {
    let (mut users, mut seen) = (vec![], HashSet::new());
    for i in 0..ranked.iter().map(|r| r.1.len()).max().unwrap_or(0) {
        for (_, v) in ranked {
            if i < v.len() && seen.insert(v[i].0.clone()) {
                users.push(v[i].0.clone());
            }
        }
    }
    users
}

/// 바깥(HTTP·시계·잠). post(본문 글자) → Ok((상태, JSON)) · Err(이름) = 연결·시한 오류 · Stop 은 정답지 끝(시험).
pub trait Env {
    fn post(&mut self, body: &str) -> Result<(u16, Value), String>;
    fn now(&mut self) -> f64;
    fn sleep(&mut self, s: f64);
    /// 시험용: 각본이 끝났다(원본 하네스의 CancelledError -- 그 바퀴는 쓰지 않는다).
    fn stopped(&self) -> bool {
        false
    }
}

fn js(s: &str) -> String {
    Value::from(s).to_string()
}

pub struct Collector {
    pub coins: Vec<String>,
    pub trades_root: PathBuf,
    pub con: Connection,
    pub universe_n: usize,
    pub refresh_s: f64,
    users: Vec<String>,
    picked_at: f64,
    prev: Vec<(String, Vec<(String, f64, Option<f64>)>)>,
    prev_t0: f64,
    seen_tids: HashSet<i64>,
}

fn write(con: &Connection, pos: &[Pos], liq: &[Liq], cyc: Option<(i64, i64, i64, i64, f64)>, uni: &[(i64, String, i64, f64, String)]) -> rusqlite::Result<()> {
    rtcollect::store::in_tx(con, || {
        for r in uni {
            con.execute("INSERT INTO hl_universe VALUES (?,?,?,?,?)", params![r.0, r.1, r.2, r.3, r.4])?;
        }
        for r in pos {
            con.execute("INSERT INTO hl_positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", params![r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7, r.8, r.9, r.10, r.11, r.12, r.13])?;
        }
        for r in liq {
            con.execute("INSERT INTO hl_liquidations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", params![r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7, r.8, r.9, r.10, r.11, r.12])?;
        }
        if let Some(c) = cyc {
            con.execute("INSERT INTO hl_cycles VALUES (?,?,?,?,?)", params![c.0, c.1, c.2, c.3, c.4])?;
        }
        Ok(())
    })
}

impl Collector {
    pub fn new(db: &Path, coins: &[String], trades_root: &Path) -> rusqlite::Result<Self> {
        Ok(Self { coins: coins.to_vec(), trades_root: trades_root.into(), con: init_db(db)?, universe_n: UNIVERSE_N, refresh_s: UNIVERSE_REFRESH_S,
                  users: vec![], picked_at: 0.0, prev: vec![], prev_t0: 0.0, seen_tids: HashSet::new() })
    }

    /// `_info` + raise_for_status + json.
    fn info(env: &mut dyn Env, body: &str) -> Result<Value, String> {
        let (st, v) = env.post(body)?;
        if st >= 400 {
            return Err(format!("ClientResponseError {st}"));
        }
        Ok(v)
    }

    /// `detect_liquidations`(:209-243). Err = 원본 예외(그 바퀴 청산 행 없음 · 그때까지 본 tid 는 남는다).
    fn detect(&mut self, env: &mut dyn Env, rows: &[Pos]) -> Result<Vec<Liq>, String> {
        let now_ms = (env.now() * 1000.0) as i64;
        let start = (self.prev_t0 * 1000.0) as i64 - 60_000;
        let mut out = vec![];
        let mut cache: HashMap<String, Value> = HashMap::new();
        for coin in self.coins.clone() {
            let Some(prev) = self.prev.iter().find(|p| p.0 == coin).map(|p| p.1.clone()).filter(|p| !p.is_empty()) else { continue };
            let candles = Self::info(env, &format!(r#"{{"type": "candleSnapshot", "req": {{"coin": {}, "interval": "1m", "startTime": {start}, "endTime": {now_ms}}}}}"#, js(&coin)))?;
            if env.stopped() {
                return Ok(out);
            }
            let cs: &[Value] = match &candles {
                Value::Array(a) if !a.is_empty() => a,
                v if !truthy(v) => continue,
                _ => return Err("TypeError".into()),
            };
            let hl = |k: &str| cs.iter().map(|c| f(c.get(k)).ok_or_else(|| "KeyError/ValueError".to_string())).collect::<Result<Vec<f64>, String>>();
            let low = hl("l")?.into_iter().fold(f64::INFINITY, f64::min);
            let high = hl("h")?.into_iter().fold(f64::NEG_INFINITY, f64::max);
            let mut cur = HashMap::new();
            for r in rows.iter().filter(|r| r.3 == coin) {
                cur.insert(r.2.clone(), r.4);
            }
            let cands = liquidation_candidates(&prev, &cur, low, high, LIQ_TOL);
            let mut got_coin = vec![];
            for u in &cands {
                if !cache.contains_key(u) {
                    let v = Self::info(env, &format!(r#"{{"type": "userFillsByTime", "user": {}, "startTime": {start}, "endTime": {now_ms}}}"#, js(u)))?;
                    if env.stopped() {
                        return Ok(out);
                    }
                    cache.insert(u.clone(), v);
                    env.sleep(REQ_GAP_S);
                }
                let got: Vec<Liq> = parse_liq_fills(&cache[u], u, &coin, now_ms)?.into_iter().filter(|r| !self.seen_tids.contains(&r.1)).collect();
                self.seen_tids.extend(got.iter().map(|r| r.1));
                if !got.is_empty() {
                    let sz: f64 = got.iter().map(|r| r.6.unwrap_or(f64::NAN)).sum();
                    println!("🔥고래 청산 {}… {coin} {sz:.1} ({}체결 · {})", &u[..u.len().min(10)], got.len(), got[0].12);
                }
                got_coin.extend(got);
            }
            if !cands.is_empty() && got_coin.is_empty() {
                println!("{coin} 청산 후보 {}주소 조회 -- 청산 체결 없음(스스로 줄임)", cands.len());
            }
            out.extend(got_coin);
        }
        Ok(out)
    }

    /// `run()` 한 바퀴(:280-333). 반환 = false 면 시험 각본이 끝났다(그 바퀴는 안 쓴다).
    pub fn cycle(&mut self, env: &mut dyn Env) -> bool {
        let t0 = env.now();
        if self.users.is_empty() || t0 - self.picked_at >= self.refresh_s {
            let ranked: Vec<(String, Vec<(String, f64)>)> = self.coins.iter().map(|c| (c.clone(), universe(&self.trades_root, t0, c, self.universe_n))).filter(|r| !r.1.is_empty()).collect();
            if !ranked.is_empty() {
                (self.users, self.picked_at) = (merge_universe(&ranked), t0);
                let uni: Vec<_> = ranked.iter().flat_map(|(c, v)| v.iter().enumerate().map(move |(i, (u, n))| ((t0 * 1000.0) as i64, u.clone(), i as i64 + 1, *n, c.clone()))).collect();
                if let Err(e) = write(&self.con, &[], &[], None, &uni) {
                    eprintln!("hl_universe 쓰기 실패: {e}"); // 원본은 여기서 예외가 run 밖으로 -- 감독이 다시 띄운다
                }
                println!("대상 {}주소 갱신 ({})", self.users.len(), ranked.iter().map(|(c, v)| format!("{c} {}", v.len())).collect::<Vec<_>>().join(" · "));
            } else if self.users.is_empty() {
                println!("대상 주소를 못 뽑았다(체결 DB 없음?) -- 60초 뒤 다시");
                env.sleep(60.0);
                return !env.stopped();
            }
        }
        let (mut rows, mut ok) = (vec![], 0i64);
        for u in self.users.clone() {
            let r = env.post(&format!(r#"{{"type": "clearinghouseState", "user": {}}}"#, js(&u)));
            if env.stopped() {
                return false;
            }
            match r {
                Ok((429, _)) => {
                    println!("429 -- 30초 쉰다");
                    env.sleep(30.0);
                    continue; // 원본: 간격 잠 없이 다음 주소
                }
                Ok((st, state)) if st < 400 => {
                    ok += 1;
                    let ts = (env.now() * 1000.0) as i64;
                    match parse_state(&state, &u, &self.coins, ts) {
                        Ok(p) => rows.extend(p),
                        Err(e) => {
                            if ok == 0 && u == self.users[0] {
                                println!("조회 실패: {e}");
                            }
                        }
                    }
                }
                Ok((st, _)) => {
                    if ok == 0 && u == self.users[0] {
                        println!("조회 실패: ClientResponseError {st}");
                    }
                }
                Err(e) => {
                    if ok == 0 && u == self.users[0] {
                        println!("조회 실패: {e}");
                    }
                }
            }
            env.sleep(REQ_GAP_S);
        }
        let took = env.now() - t0;
        let mut liq = vec![];
        if !self.prev.is_empty() && ok > 0 {
            match self.detect(env, &rows) {
                Ok(l) => liq = l,
                Err(e) => println!("청산 감지 실패: {e}"),
            }
            if env.stopped() {
                return false;
            }
        }
        if ok > 0 {
            self.prev = self.coins.iter().map(|c| {
                let mut m: Vec<(String, f64, Option<f64>)> = vec![];
                for r in rows.iter().filter(|r| &r.3 == c) {
                    match m.iter_mut().find(|x| x.0 == r.2) {
                        Some(x) => (x.1, x.2) = (r.4, r.6), // 사전 같은 열쇠 = 자리는 처음, 값은 마지막
                        None => m.push((r.2.clone(), r.4, r.6)),
                    }
                }
                (c.clone(), m)
            }).collect();
            self.prev_t0 = t0;
        }
        if let Err(e) = write(&self.con, &rows, &liq, Some(((t0 * 1000.0) as i64, self.users.len() as i64, ok, rows.len() as i64, took)), &[]) {
            println!("쓰기 실패(이번 바퀴 유실): {e}");
        }
        println!("바퀴 {took:.0}s · 응답 {ok}/{} · {}", self.users.len(), self.coins.iter().map(|c| format!("{c} 보유 {}주소", rows.iter().filter(|r| &r.3 == c).count())).collect::<Vec<_>>().join(" | "));
        true
    }
}

/// 정답지 각본: 요청마다 [본문 글자, 응답] · 응답 = {"status": n, "json": …} | {"err": "이름"}. 시계는 잠만큼 간다(요청은 0초).
pub struct Script {
    pub steps: Vec<Value>,
    pub i: usize,
    pub t: f64,
    pub sent: Vec<String>,
    stop: bool,
}

impl Script {
    pub fn new(steps: Vec<Value>, t0: f64) -> Self {
        Self { steps, i: 0, t: t0, sent: vec![], stop: false }
    }
}

impl Env for Script {
    fn post(&mut self, body: &str) -> Result<(u16, Value), String> {
        let Some(s) = self.steps.get(self.i) else {
            self.stop = true;
            return Err("CancelledError".into());
        };
        self.i += 1;
        self.sent.push(body.to_string());
        if let Some(e) = s["err"].as_str() {
            return Err(e.into());
        }
        Ok((s["status"].as_u64().unwrap_or(200) as u16, s["json"].clone()))
    }
    fn now(&mut self) -> f64 {
        self.t
    }
    fn sleep(&mut self, s: f64) {
        self.t += s;
    }
    fn stopped(&self) -> bool {
        self.stop
    }
}

/// 정답지 입력 먹이기 -- 각본이 끝날 때까지 바퀴를 돈다. 반환 = 보낸 본문 글자.
pub fn replay(inp: &Value, db: &Path, trades_root: &Path) -> Vec<String> {
    let coins: Vec<String> = inp["coins"].as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect();
    let mut c = Collector::new(db, &coins, trades_root).unwrap();
    c.universe_n = inp["universe_n"].as_u64().unwrap() as usize;
    c.refresh_s = inp["refresh_s"].as_f64().unwrap();
    let mut env = Script::new(inp["steps"].as_array().unwrap().clone(), inp["t0"].as_f64().unwrap());
    while c.cycle(&mut env) {}
    env.sent
}

/// 시험용: 정답지의 체결 행({"<코인>/<날짜>": [[coin, ts_ms, tid, hash, side, px, sz, buyer, seller]…]})으로 날짜별 DuckDB 를 만든다(체결 수집기와 같은 표).
pub fn make_trades(rows: &Value, root: &Path) -> duckdb::Result<()> {
    for (key, rs) in rows.as_object().unwrap() {
        let p = root.join(format!("{key}.duckdb"));
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p.parent().unwrap().join("notes.txt"), "x").unwrap();
        let con = duckdb::Connection::open(&p)?;
        con.execute_batch("CREATE TABLE hl_trades (
            coin VARCHAR, ts_ms BIGINT, tid BIGINT, hash VARCHAR, side VARCHAR,
            px DOUBLE, sz DOUBLE, buyer VARCHAR, seller VARCHAR)")?;
        let mut st = con.prepare("INSERT INTO hl_trades VALUES (?,?,?,?,?,?,?,?,?)")?;
        for r in rs.as_array().unwrap() {
            let s = |i: usize| r[i].as_str().unwrap().to_string();
            st.execute(duckdb::params![s(0), r[1].as_i64(), r[2].as_i64(), s(3), s(4), r[5].as_f64(), r[6].as_f64(), s(7), s(8)])?;
        }
    }
    Ok(())
}
