//! 예약 수집기 셋 파이썬 일치 시험. 정답지 = tools/gen_cron_fixture.py 가 원본 main() 들을 가짜 urlopen·가짜 시계로 돌린 것:
//! 요청 (경로·쿼리, User-Agent) 열 · 종료코드 · treasury_futures.sqlite / macro_5m.sqlite 덤프(스키마 글자 · 행) · cboe 원본 파일(푼 글자) · last_seqno.
//! 둘째 시험: 실행 파일을 127.0.0.1 가짜 HTTP 로 한 번씩(RTCOLLECT_TEST=1 -- 바깥 주소면 멈춘다).
use rusqlite::{types::ValueRef, Connection};
use serde_json::Value;
use std::cell::{Cell, RefCell};
use std::io::{Read, Write};

fn dump(db: &std::path::Path) -> Value {
    let c = Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let rows = |sql: &str| -> Vec<Value> {
        let mut st = c.prepare(sql).unwrap();
        let n = st.column_count();
        st.query_map([], |r| Ok(Value::Array((0..n).map(|j| match r.get_ref(j).unwrap() {
            ValueRef::Null => Value::Null,
            ValueRef::Integer(i) => Value::from(i),
            ValueRef::Real(f) => Value::from(rtcollect::py_float_repr(f)),
            ValueRef::Text(t) => Value::from(String::from_utf8_lossy(t).to_string()),
            ValueRef::Blob(_) => Value::Null,
        }).collect()))).unwrap().map(Result::unwrap).collect()
    };
    let mut out = serde_json::Map::new();
    out.insert("schema".into(), Value::Array(rows("SELECT sql FROM sqlite_master ORDER BY rowid").into_iter().map(|r| r[0].clone()).collect()));
    let names: Vec<String> = rows("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").into_iter().map(|r| r[0].as_str().unwrap().to_string()).collect();
    for t in names {
        out.insert(t.clone(), Value::Array(rows(&format!("SELECT * FROM {t} ORDER BY rowid"))));
    }
    Value::Object(out)
}

fn tmp(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rtcollect_cron_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn cron_collectors_match_python() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/cron_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/cron_golden.json")).unwrap();
    let d = tmp("parity");
    let clock = Cell::new(inp["t0"].as_f64().unwrap());
    let reqs = RefCell::new(vec![]);
    let resp = inp["resp"].as_object().unwrap();
    let mut get = |u: &str, ua: &str| -> Result<(u16, String, Vec<u8>), String> {
        let path = u.strip_prefix("http://h").unwrap().to_string();
        reqs.borrow_mut().push(Value::from(vec![path.clone(), ua.to_string()]));
        clock.set(clock.get() + 0.25);
        let r = &resp[&path];
        Ok((r["status"].as_u64().unwrap() as u16, r["reason"].as_str().unwrap_or("OK").to_string(), r["body"].as_str().unwrap().as_bytes().to_vec()))
    };
    let mut now = || clock.get();
    let rc_t5 = rtcollect::cron::treasury(&d.join("treasury_futures.sqlite"), "5d", "http://h", &mut get, &mut now).unwrap();
    let rc_t1 = rtcollect::cron::treasury(&d.join("treasury_futures.sqlite"), "1d", "http://h", &mut get, &mut now).unwrap();
    {
        let con = Connection::open(d.join("macro_5m.sqlite")).unwrap();
        con.execute("CREATE TABLE IF NOT EXISTS yahoo_5m(ts_sec INTEGER NOT NULL, symbol TEXT NOT NULL, open REAL, high REAL, low REAL, close REAL, volume INTEGER, PRIMARY KEY(symbol, ts_sec))", []).unwrap();
        for r in inp["seed_macro"].as_array().unwrap() {
            con.execute("INSERT INTO yahoo_5m VALUES(?,?,?,?,?,?,?)", rusqlite::params![r[0].as_i64(), r[1].as_str(), r[2].as_f64(), r[3].as_f64(), r[4].as_f64(), r[5].as_f64(), r[6].as_i64()]).unwrap();
        }
    }
    let mut sleep = |s: f64| clock.set(clock.get() + s);
    let mut now2 = || clock.get();
    let rc_m = rtcollect::cron::macro_5m(&d.join("macro_5m.sqlite"), "http://h", &mut get, &mut now2, &mut sleep).unwrap();
    for (s, v) in inp["cboe_seed"].as_object().unwrap() {
        std::fs::create_dir_all(d.join("cboe").join(s)).unwrap();
        std::fs::write(d.join("cboe").join(s).join("last_seqno"), v.as_str().unwrap()).unwrap();
    }
    let args: Vec<String> = inp["cboe_args"].as_array().unwrap().iter().map(|a| a.as_str().unwrap().to_string()).collect();
    let (rc_c, _) = rtcollect::cron::cboe(&args, &d.join("cboe"), "http://h", &mut get, clock.get());
    assert_eq!(serde_json::json!({"treasury_5d": rc_t5, "treasury_1d": rc_t1, "macro": rc_m, "cboe": rc_c}), gold["rc"], "종료코드");
    assert_eq!(Value::from(reqs.borrow().clone()), gold["requests"], "요청 열");
    for (k, f) in [("treasury", "treasury_futures.sqlite"), ("macro", "macro_5m.sqlite")] {
        let (g, w) = (dump(&d.join(f)), &gold[k]);
        for (t, rows) in w.as_object().unwrap() {
            assert_eq!(&g[t], rows, "{k} {t}");
        }
    }
    let mut files = serde_json::Map::new();
    for e in walk(&d.join("cboe")) {
        let rel = e.strip_prefix(d.join("cboe")).unwrap().to_string_lossy().to_string();
        let raw = std::fs::read(&e).unwrap();
        let txt = if rel.ends_with(".gz") { let mut s = String::new(); flate2::read::GzDecoder::new(&raw[..]).read_to_string(&mut s).unwrap(); s } else { String::from_utf8(raw).unwrap() };
        files.insert(rel, Value::from(txt));
    }
    assert_eq!(Value::Object(files), gold["cboe_files"], "cboe 파일");
    eprintln!("예약 수집기 일치: 요청 {} · treasury {}행 · macro {}행 · cboe 파일 {}", reqs.borrow().len(), gold["treasury"]["treasury_fut_1m"].as_array().unwrap().len(),
              gold["macro"]["yahoo_5m"].as_array().unwrap().len(), gold["cboe_files"].as_object().unwrap().len());
    let _ = std::fs::remove_dir_all(&d);
}

fn walk(p: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut v = vec![];
    for e in std::fs::read_dir(p).unwrap() {
        let e = e.unwrap().path();
        if e.is_dir() { v.extend(walk(&e)) } else { v.push(e) }
    }
    v.sort();
    v
}

/// 실행 파일 한 번씩: 127.0.0.1 가짜 서버(정답지 응답) · 요청 머리글 User-Agent · 종료코드 · 쓰인 행 수. 바깥 주소면 멈춘다.
#[test]
fn cron_binaries_against_fake_http() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/cron_input.json")).unwrap();
    let resp = inp["resp"].clone();
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    let seen = std::sync::Arc::new(std::sync::Mutex::new(vec![]));
    let s2 = seen.clone();
    std::thread::spawn(move || {
        for c in l.incoming() {
            let mut c = c.unwrap();
            let mut buf = [0u8; 4096];
            let n = c.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let path = req.split(' ').nth(1).unwrap_or("").to_string();
            let ua = req.lines().find(|x| x.to_lowercase().starts_with("user-agent:")).unwrap_or("").to_string();
            s2.lock().unwrap().push((path.clone(), ua));
            let r = &resp[&path];
            let (st, body) = (r["status"].as_u64().unwrap_or(404), r["body"].as_str().unwrap_or("").to_string());
            let _ = write!(c, "HTTP/1.1 {st} {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", r["reason"].as_str().unwrap_or("OK"), body.len());
        }
    });
    let d = tmp("bin");
    let run = |kind: &str, extra: &[(&str, String)], args: &[&str]| {
        let mut c = std::process::Command::new(env!("CARGO_BIN_EXE_rtcollect"));
        c.env("RTCOLLECT_KIND", kind).env("RTCOLLECT_TEST", "1").env("RTCOLLECT_YAHOO_BASE", &base).env("RTCOLLECT_CBOE_BASE", &base)
            .env("TREASURY_FUTURES_DB", d.join("t.sqlite")).env("MACRO_5M_DB", d.join("m.sqlite")).env("CBOE_OUT", d.join("cboe")).args(args);
        for (k, v) in extra {
            c.env(k, v);
        }
        c.output().unwrap()
    };
    assert_eq!(run("treasury_futures", &[], &["--range", "5d"]).status.code(), Some(0));
    assert_eq!(run("macro_5m", &[("RTCOLLECT_MACRO_GAP_S", "0".into())], &[]).status.code(), Some(0));
    assert_eq!(run("cboe_chain", &[], &["SNDK"]).status.code(), Some(1), "MU 500 = 1");
    let bad = run("treasury_futures", &[("RTCOLLECT_YAHOO_BASE", "https://query1.finance.yahoo.com".into())], &[]);
    assert!(!bad.status.success() && String::from_utf8_lossy(&bad.stderr).contains("127.0.0.1 만"), "바깥 주소면 멈춘다");
    let s = seen.lock().unwrap().clone();
    assert!(s.iter().all(|(_, ua)| ua.ends_with("Mozilla/5.0") || ua.ends_with("Mozilla/5.0 (compatible; crypto-scalping-collector/1.0)")), "{s:?}");
    assert_eq!(s.len(), 4 + 16 + 3, "treasury 4 · macro 16 · cboe 3");
    let n: i64 = Connection::open(d.join("t.sqlite")).unwrap().query_row("SELECT count(*) FROM polls", [], |r| r.get(0)).unwrap();
    assert_eq!(n, 4);
    assert!(d.join("cboe/SNDK/last_seqno").exists());
    let _ = std::fs::remove_dir_all(&d);
}
