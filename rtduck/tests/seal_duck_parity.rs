//! seal_to_lake DuckDB 원천 나머지(micro_1m · tail_risk_1m · deribit 6) 일치: tools/gen_seal_duck_fixture.py 정답지(서버 판 duckdb 1.5.1)와 stats·lake(열·형·값 -- BOOLEAN·INTEGER 포함).
//! 반대 방향: RTDUCK_KEEP=<폴더> → `$PY tools/gen_seal_duck_fixture.py --check <폴더>`(스트림마다 원본 dry-run = 전부 건너뜀).
use duckdb::Connection;
use serde_json::{json, Value};
use std::path::Path;

fn describe(lake: &Path) -> Value {
    let mut files = vec![];
    let mut st = vec![lake.to_path_buf()];
    while let Some(d) = st.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() { st.push(p) } else if p.extension().is_some_and(|x| x == "parquet") { files.push(p) }
        }
    }
    files.sort();
    let c = Connection::open_in_memory().unwrap();
    let mut out = serde_json::Map::new();
    for p in files {
        let q = format!("read_parquet('{}', hive_partitioning = false)", p.display());
        let cols: Vec<(String, String)> = c.prepare(&format!("describe select * from {q}")).unwrap().query_map([], |r| Ok((r.get(0)?, r.get(1)?))).unwrap().map(|x| x.unwrap()).collect();
        let sel: Vec<String> = cols.iter().map(|(n, t)| if t.contains("TIMESTAMP") { format!("epoch_us(\"{n}\")") } else { format!("\"{n}\"") }).collect();
        let n = cols.len();
        let rows: Vec<Value> = c.prepare(&format!("select {} from {q} order by all", sel.join(", "))).unwrap().query_map([], |r| Ok(Value::Array((0..n).map(|i| match r.get_ref(i).unwrap() {
            duckdb::types::ValueRef::Null => Value::Null,
            duckdb::types::ValueRef::BigInt(x) => json!(x),
            duckdb::types::ValueRef::Int(x) => json!(x),
            duckdb::types::ValueRef::Double(x) => json!(x),
            duckdb::types::ValueRef::Boolean(b) => json!(b),
            duckdb::types::ValueRef::Text(t) => json!(String::from_utf8_lossy(t)),
            v => panic!("{v:?}"),
        }).collect()))).unwrap().map(|x| x.unwrap()).collect();
        out.insert(p.strip_prefix(lake).unwrap().to_string_lossy().to_string(), json!({"cols": cols.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>(), "rows": rows}));
    }
    Value::Object(out)
}

#[test]
fn seal_duckdb_sources_match_python() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/seal_duck_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/seal_duck_golden.json")).unwrap();
    let d = std::env::var("RTDUCK_KEEP").map(std::path::PathBuf::from).unwrap_or_else(|_| std::env::temp_dir().join(format!("rtduck_sealduck_{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&d);
    let run_sql = |k: &str| {
        for s in inp[k].as_array().unwrap() {
            let p = d.join(s[0].as_str().unwrap());
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            Connection::open(&p).unwrap().execute_batch(s[1].as_str().unwrap()).unwrap();
        }
    };
    run_sql("build");
    let want: Vec<&str> = inp["streams"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
    let all = rtduck::seal::specs();
    let sp: Vec<rtduck::seal::Spec> = all.iter().filter(|(v, s, _, _)| want.contains(&format!("{v}/{s}").as_str())).map(|(v, s, src, ts)| (*v, s.as_str(), src.clone(), *ts)).collect();
    assert_eq!(sp.len(), want.len());
    let (lake, now) = (d.join("lake"), inp["now_s"].as_f64().unwrap());
    let (st1, l1) = rtduck::seal::seal(&sp, &d, &lake, now, false, None).unwrap();
    assert_eq!(Value::Object(st1), gold["run1"], "1회 {l1:?}");
    run_sql("mods");
    let (st2, l2) = rtduck::seal::seal(&sp, &d, &lake, now, false, None).unwrap();
    assert_eq!(Value::Object(st2.clone()), gold["run2"], "2회 {l2:?}");
    let got = describe(&lake);
    for (f, v) in gold["lake"].as_object().unwrap() {
        assert_eq!(got[f], *v, "{f}");
    }
    assert_eq!(got, gold["lake"], "파일 목록");
    if std::env::var("RTDUCK_KEEP").is_err() {
        let _ = std::fs::remove_dir_all(&d);
    }
    eprintln!("seal DuckDB 원천 일치: 스트림 {} · 파일 {} · 행 {} · 2회 {:?}", want.len(), got.as_object().unwrap().len(),
              got.as_object().unwrap().values().map(|v| v["rows"].as_array().unwrap().len()).sum::<usize>(), st2);
}
