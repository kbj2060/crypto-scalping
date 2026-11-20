//! 정답지 대조(tape_parity · venue_parity 공용). 정답지 = 원본 파이썬 수집기가 만든 SQLite 의 덤프
//! {파일: {pragma, schema, tables: {표: 행들}}}. 같은 질의로 Rust 가 만든 파일을 덤프해 **행 단위**로 비교한다:
//! 행 수 · 열마다 저장 형식(INTEGER/REAL/TEXT/NULL) · REAL 은 비트까지 · 정렬(PK 또는 rowid 순) · sqlite_master SQL · PRAGMA.
#![allow(dead_code)]
use rusqlite::{types::ValueRef, Connection, OpenFlags};
use serde_json::Value;
use std::path::{Path, PathBuf};

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
    for (db, want) in golden.as_object().unwrap() {
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
            .query_map([], |r| Ok((0..4).map(|j| Value::from(r.get::<_, String>(j).unwrap())).collect()))
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
