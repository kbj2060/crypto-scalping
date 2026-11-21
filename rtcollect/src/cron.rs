//! 예약(cron) 수집기 셋 -- 한 번 돌고 끝난다(종료코드 = 원본: 한 종목이라도 성공 0 · 전부 실패 1 · cboe 는 하나라도 실패 1).
//!   treasury_futures(`live_treasury_futures_collector_20261006.py`): ZT·ZF·ZN·ZB Yahoo 1분봉 → treasury_fut_1m · polls
//!   macro_5m(`live_macro_5m_collector_20261028.py`): Yahoo 5분봉 16종 → yahoo_5m · polls (마지막 봉 버림 · 4일 넘게 비면 60d 백필)
//!   cboe_chain(`live_cboe_option_chain_collector_20261101.py`): CBOE 지연 체인 원본 → <SYM>/<UTC날짜>/<HHMMSS>.json.gz · last_seqno(같으면 안 씀)
//!     🔴원본은 저장 직후 요약기(cboe_option_summary_20261101.write_state → cboe_option_state.json)를 부른다 -- 요약기(pandas·Deribit 계산 도구)는 아직 안 옮겼다.
//! HTTP 는 부르는 쪽이 준다(get(url, UA) → (상태, 이유, 본문) · Err = 연결 오류) -- 시험은 127.0.0.1 가짜 서버.
use rusqlite::params;
use serde_json::Value;
use std::path::{Path, PathBuf};

pub type Get<'a> = &'a mut dyn FnMut(&str, &str) -> Result<(u16, String, Vec<u8>), String>;

pub const YAHOO: &str = "https://query1.finance.yahoo.com";
pub const CBOE: &str = "https://cdn-api.cboe.com";
pub const YAHOO_UA: &str = "Mozilla/5.0";
pub const CBOE_UA: &str = "Mozilla/5.0 (compatible; crypto-scalping-collector/1.0)";
pub const TREASURY: [&str; 4] = ["ZT", "ZF", "ZN", "ZB"];
pub const MACRO: [(&str, &str); 16] = [("USDJPY", "JPY=X"), ("USDKRW", "KRW=X"), ("EURUSD", "EURUSD=X"), ("USDCNH", "CNH=X"), ("USDCNY", "CNY=X"),
    ("DXY", "DX-Y.NYB"), ("ES", "ES=F"), ("SPX", "^GSPC"), ("6J", "6J=F"), ("6E", "6E=F"), ("ZQ", "ZQ=F"), ("VIX", "^VIX"),
    ("HYG", "HYG"), ("LQD", "LQD"), ("IEF", "IEF"), ("TIP", "TIP")];
/// coin_config 의 equity 코인(2026-11-21: sndk · soxl · mu). ponytail: 원본은 COIN_CONFIG 를 읽는다 -- 주식 코인을 더하면 여기도.
pub const CBOE_DEFAULT: [&str; 3] = ["SNDK", "SOXL", "MU"];

/// 시험(RTCOLLECT_TEST=1)이면 주소가 127.0.0.1 이 아닐 때 멈춘다 -- 바깥 호스트로 절대 안 나가게.
pub fn base(var: &str, default: &str) -> String {
    let b = std::env::var(var).unwrap_or_else(|_| default.to_string());
    if std::env::var("RTCOLLECT_TEST").as_deref() == Ok("1") && !b.starts_with("http://127.0.0.1:") {
        panic!("시험인데 바깥 주소 {b} -- 127.0.0.1 만");
    }
    b
}

/// 파이썬 `f"error:{e!r}"[:200]` -- HTTP 오류는 urllib HTTPError repr 그대로 · 그 밖은 이름과 글(근사 -- 파이썬 예외 글은 못 맞춘다).
fn err_status(e: &str) -> String {
    format!("error:{e}").chars().take(200).collect()
}

fn http_json(get: Get, url: &str, ua: &str) -> Result<Value, String> {
    let (st, reason, body) = get(url, ua)?;
    if st >= 400 {
        return Err(format!("<HTTPError {st}: {}>", Value::from(reason).to_string().replace('"', "'"))); // repr(str) 꼴
    }
    serde_json::from_slice(&body).map_err(|e| format!("JSONDecodeError('{e}')"))
}

type Bar = (i64, String, Option<f64>, Option<f64>, Option<f64>, f64, i64, Option<String>);

fn num(v: &Value) -> Option<f64> {
    v.as_f64()
}

/// Yahoo chart → (봉들, shortName). step = 분(60)·5분(300) 경계 · 종가 없는 봉 버림 · zip 은 가장 짧은 열까지.
fn yahoo_rows(key: &str, chart: &Value, step: i64, vol_may_lack: bool) -> Result<(Vec<Bar>, String), String> {
    let r = chart.get("chart").and_then(|c| c.get("result")).and_then(|r| r.get(0)).ok_or("KeyError('chart')")?;
    let q = r.get("indicators").and_then(|i| i.get("quote")).and_then(|q| q.get(0)).ok_or("KeyError('indicators')")?;
    let contract = r.get("meta").and_then(|m| m.get("shortName")).and_then(Value::as_str).unwrap_or("").to_string();
    let empty = vec![];
    let ts = r.get("timestamp").and_then(Value::as_array).unwrap_or(&empty);
    let col = |k: &str| q.get(k).and_then(Value::as_array).ok_or(format!("KeyError('{k}')"));
    let (o, h, l, c) = (col("open")?, col("high")?, col("low")?, col("close")?);
    let zeros: Vec<Value> = vec![Value::from(0); ts.len()];
    // macro: q.get("volume") or [0]*len(ts) -- 없거나 빈 열이면 0 들 · treasury: q["volume"] 그대로
    let v = match q.get("volume").and_then(Value::as_array) {
        Some(a) if !a.is_empty() || !vol_may_lack => a,
        _ if vol_may_lack => &zeros,
        _ => return Err("KeyError('volume')".into()),
    };
    let n = [ts.len(), o.len(), h.len(), l.len(), c.len(), v.len()].into_iter().min().unwrap();
    let mut out = vec![];
    for i in 0..n {
        let Some(t) = ts[i].as_i64() else { continue };
        let Some(cl) = num(&c[i]) else { continue };
        if t.rem_euclid(step) != 0 {
            continue;
        }
        out.push((t, key.to_string(), num(&o[i]), num(&h[i]), num(&l[i]), cl, v[i].as_i64().unwrap_or(0), None));
    }
    Ok((out, contract))
}

fn open_db(db: &Path, ddl: &[&str]) -> rusqlite::Result<rusqlite::Connection> {
    crate::store::open_hot(db, ddl)
}

/// treasury_futures main(rng)(:52-73). 반환 = 종료코드.
pub fn treasury(db: &Path, rng: &str, base: &str, get: Get, now: &mut dyn FnMut() -> f64) -> rusqlite::Result<i32> {
    let con = open_db(db, &["CREATE TABLE IF NOT EXISTS treasury_fut_1m(ts_sec INTEGER NOT NULL, symbol TEXT NOT NULL, open REAL, high REAL, low REAL, close REAL, volume INTEGER, contract TEXT, PRIMARY KEY(symbol, ts_sec))",
                            "CREATE TABLE IF NOT EXISTS polls(ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL, status TEXT NOT NULL, bars INTEGER)"])?;
    let mut ok = 0;
    for sym in TREASURY {
        let url = format!("{base}/v8/finance/chart/{sym}%3DF?range={rng}&interval=1m");
        let (rows, status) = match http_json(get, &url, YAHOO_UA).and_then(|j| yahoo_rows(sym, &j, 60, false)) {
            Ok((mut r, c)) => {
                for x in r.iter_mut() {
                    x.7 = Some(c.clone());
                }
                (r, "ok".to_string())
            }
            Err(e) => {
                eprintln!("{sym} 실패: {e}");
                (vec![], err_status(&e))
            }
        };
        crate::store::in_tx(&con, || {
            for r in &rows {
                con.execute("INSERT OR REPLACE INTO treasury_fut_1m VALUES(?,?,?,?,?,?,?,?)", params![r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7])?;
            }
            con.execute("INSERT INTO polls VALUES(?,?,?,?)", params![(now() * 1000.0) as i64, sym, status, rows.len() as i64])?;
            Ok(())
        })?;
        ok += (status == "ok") as i32;
        if !rows.is_empty() && rng == "5d" {
            println!("{sym} {}봉 {}", rows.len(), rows.last().unwrap().7.as_deref().unwrap_or(""));
        }
    }
    Ok(if ok > 0 { 0 } else { 1 })
}

/// 파이썬 urllib.parse.quote(티커) -- 영숫자·`_.-~/` 그대로, 나머지 %XX.
fn quote(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"_.-~/".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

/// macro_5m main()(:72-97).
pub fn macro_5m(db: &Path, base: &str, get: Get, now: &mut dyn FnMut() -> f64, sleep: &mut dyn FnMut(f64)) -> rusqlite::Result<i32> {
    let con = open_db(db, &["CREATE TABLE IF NOT EXISTS yahoo_5m(ts_sec INTEGER NOT NULL, symbol TEXT NOT NULL, open REAL, high REAL, low REAL, close REAL, volume INTEGER, PRIMARY KEY(symbol, ts_sec))",
                            "CREATE TABLE IF NOT EXISTS polls(ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL, status TEXT NOT NULL, bars INTEGER)"])?;
    let last: std::collections::HashMap<String, i64> = con.prepare("SELECT symbol, max(ts_sec) FROM yahoo_5m GROUP BY symbol")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?;
    let mut ok = 0;
    for (i, (key, ticker)) in MACRO.iter().enumerate() {
        if i > 0 {
            sleep(1.5);
        }
        let rng = if last.get(*key).is_some_and(|l| now() - (*l as f64) < 4.0 * 86400.0) { "5d" } else { "60d" };
        let url = format!("{base}/v8/finance/chart/{}?range={rng}&interval=5m", quote(ticker));
        let (rows, status) = match http_json(get, &url, YAHOO_UA).and_then(|j| yahoo_rows(key, &j, 300, true)) {
            Ok((mut r, _)) => {
                r.pop(); // 마지막 봉은 진행 중일 수 있다 -- 버린다
                (r, "ok".to_string())
            }
            Err(e) => {
                eprintln!("{key} 실패: {e}");
                (vec![], err_status(&e))
            }
        };
        crate::store::in_tx(&con, || {
            for r in &rows {
                con.execute("INSERT OR REPLACE INTO yahoo_5m VALUES(?,?,?,?,?,?,?)", params![r.0, r.1, r.2, r.3, r.4, r.5, r.6])?;
            }
            con.execute("INSERT INTO polls VALUES(?,?,?,?)", params![(now() * 1000.0) as i64, key, status, rows.len() as i64])?;
            Ok(())
        })?;
        ok += (status == "ok") as i32;
        if !rows.is_empty() && rng == "60d" {
            println!("{key} 백필 {}봉", rows.len());
        }
    }
    Ok(if ok > 0 { 0 } else { 1 })
}

/// cboe collect(sym)(:35-55): 저장한 경로 · 같은 seqno 면 None. Err = 원본 예외(그 종목 실패).
pub fn cboe_collect(sym: &str, out: &Path, base: &str, get: Get, now: f64) -> Result<Option<PathBuf>, String> {
    if sym.is_empty() || sym.len() > 12 || !sym.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'.') {
        return Err(format!("ValueError: 종목 표기가 이상하다: {sym:?}")); // 신뢰경계: URL·파일 경로
    }
    let (st, reason, raw) = get(&format!("{base}/api/global/delayed_quotes/options/{sym}.json"), CBOE_UA)?;
    if st >= 400 {
        return Err(format!("HTTPError: HTTP Error {st}: {reason}"));
    }
    let j: Value = serde_json::from_slice(&raw).map_err(|e| format!("JSONDecodeError: {e}"))?;
    let seq = j.get("data").and_then(|d| d.get("seqno")).ok_or("KeyError: 'seqno'")?;
    let seq_s = match seq {
        Value::String(s) => s.clone(),
        v => crate::py_str(Some(v), ""),
    };
    let d = out.join(sym);
    let state = d.join("last_seqno");
    if std::fs::read_to_string(&state).is_ok_and(|s| s.trim() == seq_s) {
        return Ok(None);
    }
    let s = now.floor() as i64;
    let k = crate::depth::hour_key(s * 1000).unwrap_or_default(); // YYYY-MM-DDTHH
    let day = k[..10].replace('-', "");
    let path = d.join(day).join(format!("{}{:02}{:02}.json.gz", &k[11..13], s.rem_euclid(3600) / 60, s.rem_euclid(60)));
    let e = |x: std::io::Error| format!("OSError: {x}");
    std::fs::create_dir_all(path.parent().unwrap()).map_err(e)?;
    let tmp = path.with_file_name(format!("{}.{}.part", path.file_name().unwrap().to_string_lossy(), std::process::id()));
    {
        use std::io::Write;
        let mut z = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
        z.write_all(&raw).map_err(e)?;
        std::fs::write(&tmp, z.finish().map_err(e)?).map_err(e)?;
    }
    std::fs::rename(&tmp, &path).map_err(e)?;
    std::fs::write(&state, &seq_s).map_err(e)?; // 파일 먼저 · 표식 나중
    Ok(Some(path))
}

/// cboe main(argv)(:67-79): 인자 + 기본 주식 코인(순서 유지·중복 제거). 반환 = (종료코드, 저장한 경로들).
pub fn cboe(args: &[String], out: &Path, base: &str, get: Get, now: f64) -> (i32, Vec<PathBuf>) {
    let mut syms: Vec<String> = vec![];
    for s in args.iter().map(|a| a.to_uppercase()).chain(CBOE_DEFAULT.iter().map(|s| s.to_string())) {
        if !syms.contains(&s) {
            syms.push(s);
        }
    }
    let (mut rc, mut saved) = (0, vec![]);
    for s in &syms {
        match cboe_collect(s, out, base, get, now) {
            Ok(Some(p)) => {
                println!("{s} 저장 {}", p.display());
                saved.push(p);
            }
            Ok(None) => println!("{s} 변화 없음(휴장·장외)"),
            Err(e) => {
                println!("{s} 실패 {e}");
                rc = 1;
            }
        }
    }
    (rc, saved)
}
