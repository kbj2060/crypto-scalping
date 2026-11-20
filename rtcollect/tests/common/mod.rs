//! 정답지 대조(tape_parity · venue_parity 공용). 정답지 = 원본 파이썬 수집기가 만든 SQLite 의 덤프
//! {파일: {pragma, schema, tables: {표: 행들}}}. 같은 질의로 Rust 가 만든 파일을 덤프해 **행 단위**로 비교한다:
//! 행 수 · 열마다 저장 형식(INTEGER/REAL/TEXT/NULL) · REAL 은 비트까지 · 정렬(PK 또는 rowid 순) · sqlite_master SQL · PRAGMA.
#![allow(dead_code)]
use rusqlite::{types::ValueRef, Connection, OpenFlags};
use serde_json::Value;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rtcollect_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn cell_eq(got: ValueRef, want: &Value) -> bool {
    match (got, want) {
        (ValueRef::Null, Value::Null) => true,
        (ValueRef::Integer(g), Value::Number(w)) => w.as_i64() == Some(g) && !w.is_f64(),
        (ValueRef::Real(g), Value::Number(w)) => w.is_f64() && w.as_f64().unwrap().to_bits() == g.to_bits(),
        (ValueRef::Text(g), Value::String(w)) => g == w.as_bytes(),
        // 정답지의 무한대 REAL(JSON 에 없다) = {"f": "inf"|"-inf"}
        (ValueRef::Real(g), Value::Object(o)) => g.is_infinite() && o["f"] == if g > 0.0 { "inf" } else { "-inf" },
        _ => false,
    }
}

fn show(v: ValueRef) -> String {
    match v {
        ValueRef::Real(f) => format!("{f:?}(REAL)"),
        ValueRef::Integer(i) => format!("{i}(INT)"),
        ValueRef::Text(t) => format!("{:?}", String::from_utf8_lossy(t)),
        ValueRef::Null => "NULL".into(),
        ValueRef::Blob(_) => "BLOB".into(),
    }
}

/// 칸 하나를 비교 전에 고친다: (파일, 표, 열 번호, 값) → 같으면 비교할 글자. 대부분은 그대로(None).
pub type Norm = fn(&str, &str, usize, &str) -> Option<String>;

/// 반환 = (표 수, 행 수, 칸 수). `tables` = (표, 정렬) -- 정답지의 "order" 가 있으면 그것.
pub fn compare(dir: &Path, golden: &Value, tables: &[(&str, &str)], norm: Option<Norm>) -> (usize, usize, usize) {
    let (mut n_tables, mut n_rows, mut n_cells) = (0, 0, 0);
    for (db, want) in golden.as_object().unwrap().iter().filter(|(k, _)| k.as_str() != "intended") {
        let con = Connection::open_with_flags(dir.join(db), OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        for (p, w) in want["pragma"].as_object().unwrap() {
            let g: Value = con
                .query_row(&format!("PRAGMA {p}"), [], |r| {
                    Ok(match r.get_ref(0)? {
                        ValueRef::Integer(i) => Value::from(i),
                        ValueRef::Text(t) => Value::from(String::from_utf8_lossy(t).to_string()),
                        v => panic!("{v:?}"),
                    })
                })
                .unwrap();
            assert_eq!(&g, w, "{db} PRAGMA {p}");
        }
        let mut st = con.prepare("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").unwrap();
        let schema: Vec<Value> = st
            .query_map([], |r| Ok((0..4).map(|j| Value::from(r.get::<_, Option<String>>(j).unwrap())).collect())) // PK 자동 인덱스의 sql = NULL
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(Value::from(schema), want["schema"], "{db} sqlite_master SQL");

        let order: Vec<(String, String)> = match want["order"].as_object() {
            Some(o) => o.iter().map(|(t, v)| (t.clone(), v.as_str().unwrap().to_string())).collect(),
            None => tables.iter().map(|(t, o)| (t.to_string(), o.to_string())).collect(),
        };
        for (t, ord) in &order {
            let rows = want["tables"][t].as_array().unwrap();
            let mut st = con.prepare(&format!("SELECT * FROM {t} ORDER BY {ord}")).unwrap();
            let ncol = st.column_count();
            let mut got = st.query([]).unwrap();
            let mut i = 0;
            while let Some(r) = got.next().unwrap() {
                assert!(i < rows.len(), "{db}.{t}: Rust 행이 더 많다 (정답 {}행)\n  남는 행 {:?}", rows.len(),
                    (0..ncol).map(|j| show(r.get_ref(j).unwrap())).collect::<Vec<_>>());
                let w = rows[i].as_array().unwrap();
                assert_eq!(w.len(), ncol, "{db}.{t} 열 수");
                for j in 0..ncol {
                    let g = r.get_ref(j).unwrap();
                    let ok = match (norm, g, &w[j]) {
                        (Some(f), ValueRef::Text(gt), Value::String(wt)) => match (f(db, t, j, &String::from_utf8_lossy(gt)), f(db, t, j, wt)) {
                            (Some(a), Some(b)) => a == b,
                            _ => cell_eq(g, &w[j]),
                        },
                        _ => cell_eq(g, &w[j]),
                    };
                    assert!(ok, "{db}.{t} 행 {i} 열 {j}: rust {} != python {}\n  정답 행 {:?}", show(g), w[j], w);
                    n_cells += 1;
                }
                i += 1;
            }
            assert_eq!(i, rows.len(), "{db}.{t} 행 수");
            n_rows += i;
            n_tables += 1;
        }
    }
    (n_tables, n_rows, n_cells)
}

/// 가짜 REST(rest_parity · ctx_parity 공용): 요청 대상(경로?쿼리)마다 정해 둔 응답을 차례로. `timeout` = 1.5초 동안 답하지 않는다(클라이언트 시한 0.3초).
pub fn serve(responses: &serde_json::Map<String, Value>) -> (String, Arc<Mutex<Vec<String>>>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    let q: Arc<Mutex<HashMap<String, Vec<Value>>>> =
        Arc::new(Mutex::new(responses.iter().map(|(k, v)| (k.clone(), v.as_array().unwrap().clone())).collect()));
    let log = Arc::new(Mutex::new(vec![]));
    let log2 = log.clone();
    std::thread::spawn(move || {
        for s in l.incoming() {
            let (mut s, q, log) = (s.unwrap(), q.clone(), log2.clone());
            std::thread::spawn(move || {
                let mut buf = vec![];
                let mut b = [0u8; 4096];
                while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    match s.read(&mut b) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => buf.extend_from_slice(&b[..n]),
                    }
                }
                let target = String::from_utf8_lossy(&buf).split_whitespace().nth(1).unwrap().to_string();
                log.lock().unwrap().push(target.clone());
                let r = q.lock().unwrap().get_mut(&target).filter(|v| !v.is_empty()).map(|v| v.remove(0));
                let Some(r) = r else {
                    let _ = s.write_all(b"HTTP/1.1 599 NO FIXTURE\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                    return;
                };
                if r["timeout"] == true {
                    std::thread::sleep(Duration::from_millis(1500));
                    return;
                }
                // "raw" = 본문 글자 그대로(깨진 JSON·빈 본문) · "ctype" = 본문 형식(기본 application/json)
                let body = r["raw"].as_str().map_or_else(|| r["body"].to_string(), str::to_string);
                let ctype = r["ctype"].as_str().unwrap_or("application/json");
                let mut h = format!("HTTP/1.1 {} X\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n", r["status"], body.len());
                for (k, v) in r["headers"].as_object().into_iter().flatten() {
                    h += &format!("{k}: {}\r\n", v.as_str().unwrap());
                }
                let _ = s.write_all(format!("{h}\r\n{body}").as_bytes());
            });
        }
    });
    (base, log)
}

