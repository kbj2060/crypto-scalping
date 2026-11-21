//! 롱숏비·OI 워커(`oi_lsratio_collector.py` -- `duckdb_persist_worker.py` 의 COLLECT_OI_LSRATIO 모드, 줄 번호는 oi_lsratio_collector.py).
//! 바이낸스 /futures/data 다섯 원천(5분 · 최근 OI_FETCH_LIMIT 점)을 ts 로 합쳐 DuckDB 표에 «읽기-합치기-지우기-넣기» upsert.
//! 표 = ETH `oi_lsratio_5m` · 나머지 `oi_lsratio_5m_<coin>`. 🔴쓸 때마다 열고 닫는다(원본 그대로) -- 닫을 때 체크포인트되어
//! 서버(읽기 전용 열기)·seal_to_lake(파일 사본)가 본체만 읽고, 열어 둔 채면 다른 프로세스의 열기가 «Conflicting lock» 이다.
use duckdb::{params_from_iter, Connection};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

pub const BASE: &str = "https://fapi.binance.com";
pub const UA: &str = "crypto-scalping-oi-lsratio-collector/1.0";
pub const PERIOD_S: f64 = 300.0;
pub const PATHS: [&str; 5] = ["/futures/data/openInterestHist", "/futures/data/globalLongShortAccountRatio", "/futures/data/topLongShortPositionRatio",
                              "/futures/data/topLongShortAccountRatio", "/futures/data/takerlongshortRatio"];
pub const MERGE_COLS: [&str; 14] = ["sum_open_interest", "sum_open_interest_value", "global_ls_ratio", "global_ls_long_account", "global_ls_short_account",
                                   "top_pos_ls_ratio", "top_pos_ls_long_account", "top_pos_ls_short_account",
                                   "top_acct_ls_ratio", "top_acct_ls_long_account", "top_acct_ls_short_account",
                                   "taker_ls_ratio", "taker_buy_vol", "taker_sell_vol"];
/// 원천마다 (응답 열쇠 → MERGE_COLS 자리)(:220-247).
const FIELDS: [&[(&str, usize)]; 5] = [&[("sumOpenInterest", 0), ("sumOpenInterestValue", 1)],
                                       &[("longShortRatio", 2), ("longAccount", 3), ("shortAccount", 4)],
                                       &[("longShortRatio", 5), ("longAccount", 6), ("shortAccount", 7)],
                                       &[("longShortRatio", 8), ("longAccount", 9), ("shortAccount", 10)],
                                       &[("buySellRatio", 11), ("buyVol", 12), ("sellVol", 13)]];

pub fn table(symbol: &str) -> String {
    let s = symbol.to_lowercase();
    if s == "ethusdt" { "oi_lsratio_5m".into() } else { format!("oi_lsratio_5m_{}", s.replace("usdt", "")) }
}

/// `_db_init`(:71-110) -- 글자 그대로의 DDL. 반환 = 표의 마지막 ts(epoch µs).
pub fn init(path: &std::path::Path, table: &str) -> duckdb::Result<Option<i64>> {
    if let Some(d) = path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let con = Connection::open(path)?;
    con.execute_batch(&format!(
        "CREATE TABLE IF NOT EXISTS {table} (
                ts TIMESTAMPTZ,
                symbol VARCHAR,
                sum_open_interest DOUBLE,
                sum_open_interest_value DOUBLE,
                global_ls_ratio DOUBLE,
                global_ls_long_account DOUBLE,
                global_ls_short_account DOUBLE,
                top_pos_ls_ratio DOUBLE,
                top_pos_ls_long_account DOUBLE,
                top_pos_ls_short_account DOUBLE,
                top_acct_ls_ratio DOUBLE,
                top_acct_ls_long_account DOUBLE,
                top_acct_ls_short_account DOUBLE,
                taker_ls_ratio DOUBLE,
                taker_buy_vol DOUBLE,
                taker_sell_vol DOUBLE,
                sources_ok INTEGER,
                collected_at TIMESTAMPTZ,
                schema_version INTEGER
            )"))?;
    for c in &MERGE_COLS[8..] {
        con.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {c} DOUBLE"))?;
    }
    con.query_row(&format!("SELECT epoch_us(MAX(ts)) FROM {table}"), [], |r| r.get(0))
}

/// `_fetch_one` 의 URL(urlencode 순서 symbol·period·limit).
pub fn url(base: &str, path: &str, symbol: &str, limit: i64) -> String {
    format!("{base}{path}?symbol={}&period=5m&limit={limit}", symbol.to_uppercase())
}

/// 파이썬 float(x): 숫자 그대로 · 글자는 앞뒤 공백을 떼고 읽는다(그 밖은 예외 = 그 폴 전체를 버린다).
/// ponytail: 파이썬이 받는 «1_000» 같은 밑줄 숫자는 안 받는다(거래소 응답에 없다).
fn pyfloat(v: &Value) -> Result<f64, String> {
    match v {
        Value::Number(n) => n.as_f64().ok_or_else(|| "OverflowError".into()),
        Value::String(s) => s.trim().parse::<f64>().map_err(|_| format!("ValueError: could not convert string to float: {s:?}")),
        Value::Bool(b) => Ok(*b as u8 as f64),
        _ => Err("TypeError: float()".into()),
    }
}

/// 파이썬 int(x): 정수 · 실수는 0 쪽으로 · 글자는 정수 글자만.
fn pyint(v: &Value) -> Result<i64, String> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)).ok_or_else(|| "OverflowError".into()),
        Value::String(s) => s.trim().parse::<i64>().map_err(|_| format!("ValueError: invalid literal for int(): {s:?}")),
        Value::Bool(b) => Ok(*b as i64),
        _ => Err("TypeError: int()".into()),
    }
}

/// 합친 한 봉: ts(ms) · 14칸(없으면 None) · sources_ok.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub ts_ms: i64,
    pub vals: [Option<f64>; 14],
    pub sources_ok: i64,
}

/// `_poll_once_blocking`(:199-255): 다섯 원천을 차례로 받아(get = None 이면 그 원천 실패) ts 로 합친다.
/// Err = 응답 행 하나가 깨졌다(KeyError·ValueError -- 원본은 폴 루프 except 로 그 폴을 통째로 버린다). Ok((행, 닿은 원천 수)).
pub fn poll(symbol: &str, limit: i64, base: &str, get: &mut dyn FnMut(&str) -> Option<Value>) -> Result<(Vec<Row>, usize), String> {
    let got: Vec<Option<Vec<Value>>> = PATHS.iter().map(|p| match get(&url(base, p, symbol, limit)) {
        Some(Value::Array(a)) => Some(a),
        Some(other) => {
            eprintln!("oi_lsratio({}) unexpected response shape from {p}: {other}", symbol.to_uppercase());
            None
        }
        None => None,
    }).collect();
    let mut by_ts: BTreeMap<i64, Row> = BTreeMap::new();
    for (rows, fields) in got.iter().zip(FIELDS) {
        for r in rows.iter().flatten() {
            let ts = pyint(r.get("timestamp").ok_or("KeyError: 'timestamp'")?)?;
            let e = by_ts.entry(ts).or_insert(Row { ts_ms: ts, vals: [None; 14], sources_ok: 0 });
            let mut v = [None; 3];
            for (k, (key, _)) in fields.iter().enumerate() {
                v[k] = Some(pyfloat(r.get(*key).ok_or_else(|| format!("KeyError: '{key}'"))?)?); // dict 를 다 만든 뒤 update(:217)
            }
            for (k, (_, col)) in fields.iter().enumerate() {
                e.vals[*col] = v[k];
            }
            e.sources_ok += 1;
        }
    }
    Ok((by_ts.into_values().collect(), got.iter().filter(|g| g.is_some()).count()))
}

/// `_db_upsert_rows`(:120-173): 같은 ts 의 기존 행과 COALESCE(새 값, 기존) · sources_ok = 큰 쪽 · collected_at = now_us · schema_version 3.
pub fn upsert(path: &std::path::Path, table: &str, symbol: &str, rows: &[Row], now_us: i64) -> duckdb::Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let con = Connection::open(path)?;
    let ts_us: Vec<i64> = rows.iter().map(|r| r.ts_ms * 1000).collect();
    let ph = vec!["make_timestamptz(?)"; rows.len()].join(",");
    let mut st = con.prepare(&format!("SELECT epoch_us(ts), sources_ok, {} FROM {table} WHERE ts IN ({ph})", MERGE_COLS.join(", ")))?;
    let mut ex: HashMap<i64, (Option<i64>, Vec<Option<f64>>)> = HashMap::new();
    let mut q = st.query(params_from_iter(&ts_us))?;
    while let Some(r) = q.next()? {
        ex.insert(r.get(0)?, (r.get(1)?, (0..14).map(|j| r.get(2 + j)).collect::<duckdb::Result<_>>()?));
    }
    con.execute(&format!("DELETE FROM {table} WHERE ts IN ({ph})"), params_from_iter(&ts_us))?;
    let mut ins = con.prepare(&format!(
        "INSERT INTO {table} (
                    ts, symbol, sum_open_interest, sum_open_interest_value,
                    global_ls_ratio, global_ls_long_account, global_ls_short_account,
                    top_pos_ls_ratio, top_pos_ls_long_account, top_pos_ls_short_account,
                    top_acct_ls_ratio, top_acct_ls_long_account, top_acct_ls_short_account,
                    taker_ls_ratio, taker_buy_vol, taker_sell_vol,
                    sources_ok, collected_at, schema_version
                ) VALUES (make_timestamptz(?),?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,make_timestamptz(?),?)"))?;
    for (r, us) in rows.iter().zip(&ts_us) {
        let e = ex.get(us);
        let mut p: Vec<duckdb::types::Value> = vec![(*us).into(), symbol.to_uppercase().into()];
        for j in 0..14 {
            p.push(r.vals[j].or_else(|| e.and_then(|e| e.1[j])).into());
        }
        p.push(r.sources_ok.max(e.and_then(|e| e.0).unwrap_or(0)).into());
        p.push(now_us.into());
        p.push(3i64.into());
        ins.execute(params_from_iter(p))?;
    }
    Ok(())
}

/// `_poll_loop` 의 잠(:270-272): 다음 주기 경계 + 90초(거래소가 그 봉을 낸 뒤).
pub fn sleep_s(now: f64, interval: f64) -> f64 {
    (interval - now.rem_euclid(interval)) + 90.0
}

/// 정답지 입력 먹이기: 폴마다 종목마다 poll → upsert(원본 폴 루프의 except = 그 종목 그 폴은 버림). 반환 = (요청 URL 열, 기록).
pub fn replay(inp: &Value, db: &std::path::Path) -> (Vec<String>, Vec<Value>) {
    let syms: Vec<String> = inp["symbols"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
    let limit = inp["fetch_limit"].as_i64().unwrap();
    for s in &syms {
        init(db, &table(s)).unwrap();
    }
    let (mut reqs, mut log) = (vec![], vec![]);
    for p in inp["polls"].as_array().unwrap() {
        let now_us = (p["t"].as_f64().unwrap() * 1e6).round_ties_even() as i64;
        for s in &syms {
            let resp = &p["resp"][s.as_str()];
            let mut get = |u: &str| {
                reqs.push(u.to_string());
                let r = &resp[u.split('?').next().unwrap().trim_start_matches(BASE)];
                if r.get("err").is_some() || r["status"].as_u64().unwrap_or(200) >= 400 {
                    return None;
                }
                serde_json::from_str::<Value>(r["body"].as_str().unwrap()).ok() // JSONDecodeError = 그 원천만 None
            };
            match poll(s, limit, BASE, &mut get) {
                Ok((rows, ok)) => {
                    upsert(db, &table(s), s, &rows, now_us).unwrap();
                    log.push(serde_json::json!([s, "ok", rows.len(), ok]));
                }
                Err(e) => log.push(serde_json::json!([s, e.split(':').next().unwrap(), 0, 0])),
            }
        }
    }
    (reqs, log)
}

#[cfg(test)]
mod tests {
    #[test]
    fn sleep_lands_90s_after_boundary() {
        assert_eq!(super::sleep_s(1_790_000_000.0, 300.0), 300.0 - 1_790_000_000f64.rem_euclid(300.0) + 90.0);
        assert_eq!(super::sleep_s(600.0, 300.0), 390.0); // 경계 위면 한 주기 + 90(원본 식 그대로)
        assert_eq!(super::sleep_s(899.5, 300.0), 90.5);
    }
}
