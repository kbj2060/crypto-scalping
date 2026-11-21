//! HL 고래 포지션(live_hyperliquid_positions_collector_20260924.py) 파이썬 일치 시험. 정답지 = tools/gen_hl_positions_fixture.py 가 원본 run() 을
//! 가짜 aiohttp·가짜 시계로 돌려 기록한 응답 열 + 그 SQLite 덤프(스키마 글자 · 표 넷 행 · WAL/auto_vacuum). 입력 체결 DuckDB 는 시험마다 만든다.
use rusqlite::{types::ValueRef, Connection};
use serde_json::Value;

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
    for t in ["hl_positions", "hl_universe", "hl_liquidations", "hl_cycles"] {
        out.insert(t.into(), Value::Array(rows(&format!("SELECT * FROM {t} ORDER BY rowid"))));
    }
    let jm: String = c.query_row("PRAGMA journal_mode", [], |r| r.get(0)).unwrap();
    let av: i64 = c.query_row("PRAGMA auto_vacuum", [], |r| r.get(0)).unwrap();
    out.insert("pragma".into(), serde_json::json!([jm, av]));
    Value::Object(out)
}

#[test]
fn hl_positions_db_matches_python_collector() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/hl_positions_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/hl_positions_golden.json")).unwrap();
    let dir = std::env::temp_dir().join(format!("rtduck_hl_pos_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    rtduck::hl_positions::make_trades(&inp["trades"], &dir.join("tr")).unwrap();
    let db = dir.join("hl_positions_btc_sol.sqlite");
    let sent = rtduck::hl_positions::replay(&inp, &db, &dir.join("tr"));
    let want: Vec<&str> = gold["sent"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect();
    assert_eq!(sent, want, "요청 본문 글자 열");
    let got = dump(&db);
    let g = &gold["db"];
    assert_eq!(got["schema"], g["schema"], "스키마 글자");
    assert_eq!(got["pragma"], g["pragma"], "WAL · auto_vacuum");
    let mut n = 0;
    for t in ["hl_positions", "hl_universe", "hl_liquidations", "hl_cycles"] {
        let (a, b) = (got[t].as_array().unwrap(), g[t].as_array().unwrap());
        assert_eq!(a.len(), b.len(), "{t} 행 수");
        for (k, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x, y, "{t} 행 {k}");
        }
        n += a.iter().map(|r| r.as_array().unwrap().len()).sum::<usize>();
    }
    eprintln!("hl_positions 일치: 요청 {} · 칸 {n} · 표 4", sent.len());
    let _ = std::fs::remove_dir_all(&dir);
}
