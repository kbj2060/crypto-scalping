//! DuckDB 파일을 tools/duckdump.py `dump()` 와 같은 꼴로: 표마다 열·형 · (ts 열이 있으면) 물리 순서 ts · 행(ts 열이 있으면 ts 순, 없으면 물리 순서)
//! -- 실수 = 파이썬 repr · 시각 = epoch µs.
use duckdb::{types::ValueRef, Connection};
use serde_json::{json, Map, Value};

pub fn dump(path: &std::path::Path) -> duckdb::Result<Value> {
    let con = Connection::open_with_flags(path, duckdb::Config::default().access_mode(duckdb::AccessMode::ReadOnly)?)?;
    let tables: Vec<String> = con.prepare("SELECT table_name FROM duckdb_tables() ORDER BY table_name")?.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()?;
    let mut out = Map::new();
    for t in tables {
        let cols: Vec<(String, String)> = con.prepare("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position")?
            .query_map([&t], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?;
        let sel: Vec<String> = cols.iter().map(|(c, ty)| if ty.starts_with("TIMESTAMP") { format!("epoch_us({c})") } else { c.clone() }).collect();
        let has_ts = cols.iter().any(|c| c.0 == "ts");
        let mut st = con.prepare(&format!("SELECT {} FROM {t}{}", sel.join(", "), if has_ts { " ORDER BY ts" } else { "" }))?;
        let mut q = st.query([])?;
        let mut rows = vec![];
        while let Some(r) = q.next()? {
            rows.push(Value::Array((0..cols.len()).map(|j| match r.get_ref(j).unwrap() {
                ValueRef::Null => Value::Null,
                ValueRef::Double(f) => Value::from(rtcollect::py_float_repr(f)),
                ValueRef::BigInt(i) => Value::from(i),
                ValueRef::Int(i) => Value::from(i),
                ValueRef::Text(s) => Value::from(String::from_utf8_lossy(s).to_string()),
                v => Value::from(format!("{v:?}")),
            }).collect()));
        }
        let phys: Vec<i64> = if has_ts { con.prepare(&format!("SELECT epoch_us(ts) FROM {t}"))?.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()? } else { vec![] };
        out.insert(t, json!({"cols": cols.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>(), "phys_ts": phys, "rows": rows}));
    }
    Ok(Value::Object(out))
}
