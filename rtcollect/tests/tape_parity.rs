//! 체결 테이프 파이썬 일치 시험. 정답지 = tools/gen_tape_fixture.py 가 원본 collect() 로 만든 SQLite 의 덤프.
//! 같은 메시지 열을 rtcollect::replay 에 넣어 만든 SQLite 를 같은 질의로 덤프해 **행 단위**로 비교한다:
//! 행 수 · 열마다 저장 형식(INTEGER/REAL/TEXT/NULL) · REAL 은 비트까지 · 정렬(PK 또는 rowid 순) · sqlite_master SQL · PRAGMA.
use rusqlite::{types::ValueRef, Connection, OpenFlags};
use serde_json::Value;
use std::path::PathBuf;

const INPUT: &str = include_str!("fixtures/tape_input.json");
const GOLDEN: &str = include_str!("fixtures/tape_golden.json");
const TABLES: [(&str, &str); 5] = [
    ("trade_tape_1s", "symbol, ts_sec, price_bin"),
    ("gaps", "rowid"),
    ("verify_1m", "rowid"),
    ("backfill_1m", "rowid"),
    ("meta", "rowid"),
];

fn tmpdir(tag: &str) -> PathBuf {
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

#[test]
fn tape_rows_match_python_collector() {
    let input: Value = serde_json::from_str(INPUT).unwrap();
    let golden: Value = serde_json::from_str(GOLDEN).unwrap();
    let dir = tmpdir("parity");
    let mut flushes = 0;
    rtcollect::replay(input["runs"].as_array().unwrap(), &dir, |_, _| flushes += 1).unwrap();

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

        for (t, order) in TABLES {
            let rows = want["tables"][t].as_array().unwrap();
            let mut st = con.prepare(&format!("SELECT * FROM {t} ORDER BY {order}")).unwrap();
            let ncol = st.column_count();
            let mut got = st.query([]).unwrap();
            let mut i = 0;
            while let Some(r) = got.next().unwrap() {
                assert!(i < rows.len(), "{db}.{t}: Rust 행이 더 많다 (정답 {}행)", rows.len());
                let w = rows[i].as_array().unwrap();
                assert_eq!(w.len(), ncol, "{db}.{t} 열 수");
                for j in 0..ncol {
                    let g = r.get_ref(j).unwrap();
                    assert!(cell_eq(g, &w[j]), "{db}.{t} 행 {i} 열 {j}: rust {} != python {}\n  정답 행 {:?}", show(g), w[j], w);
                    n_cells += 1;
                }
                i += 1;
            }
            assert_eq!(i, rows.len(), "{db}.{t} 행 수");
            n_rows += i;
            n_tables += 1;
        }
    }
    eprintln!("일치: 파일 {} · 표 {n_tables} · 행 {n_rows} · 칸 {n_cells} · Rust 플러시 {flushes}회", golden.as_object().unwrap().len());
    let _ = std::fs::remove_dir_all(&dir);
}
