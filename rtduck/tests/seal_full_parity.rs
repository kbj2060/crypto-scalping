//! seal_to_lake 전 스트림 일치: tools/gen_seal_full_fixture.py 가 원본 seal()·prune_hot·seal_status 를 돌린 정답지와
//!   SPECS·HOT_PRUNE 글자 · 1회 stats·lake(파일·열·형·값) · 2회 stats(hot_pruned 포함)·lake·status 글자·hot 남은 행 수를 대조한다.
//!   반대 방향: RTDUCK_KEEP=<폴더> 로 돌리면 원천·lake 를 남긴다 → `python3 tools/gen_seal_full_fixture.py --check <폴더>`(스트림마다 원본 dry-run = 전부 건너뜀).
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
        let sel: Vec<String> = cols.iter().map(|(n, t)| if t.contains("TIMESTAMP") { format!("epoch_ms(\"{n}\")") } else { format!("\"{n}\"") }).collect();
        let n = cols.len();
        let rows: Vec<Value> = c.prepare(&format!("select {} from {q} order by all", sel.join(", "))).unwrap().query_map([], |r| Ok(Value::Array((0..n).map(|i| match r.get_ref(i).unwrap() {
            duckdb::types::ValueRef::Null => Value::Null,
            duckdb::types::ValueRef::BigInt(x) => json!(x),
            duckdb::types::ValueRef::Int(x) => json!(x),
            duckdb::types::ValueRef::Double(x) => json!(x),
            duckdb::types::ValueRef::Text(t) => json!(String::from_utf8_lossy(t)),
            v => panic!("{v:?}"),
        }).collect()))).unwrap().map(|x| x.unwrap()).collect();
        out.insert(p.strip_prefix(lake).unwrap().to_string_lossy().to_string(), json!({"cols": cols.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>(), "rows": rows}));
    }
    Value::Object(out)
}

fn hot_counts(root: &Path) -> Value {
    let mut out = serde_json::Map::new();
    let mut fs: Vec<_> = std::fs::read_dir(root.join("data/hot")).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "sqlite")).collect();
    fs.sort();
    for f in fs {
        let c = rusqlite::Connection::open(&f).unwrap();
        let ts: Vec<String> = c.prepare("select name from sqlite_master where type='table' order by name").unwrap().query_map([], |r| r.get(0)).unwrap().map(|x| x.unwrap()).collect();
        for t in ts {
            let n: i64 = c.query_row(&format!("select count(*) from {t}"), [], |r| r.get(0)).unwrap();
            out.insert(format!("{}:{t}", f.file_name().unwrap().to_string_lossy()), json!(n));
        }
    }
    Value::Object(out)
}

#[test]
fn seal_all_streams_match_python() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/seal_full_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/seal_full_golden.json")).unwrap();
    // SPECS · HOT_PRUNE 글자
    let specs: Value = rtduck::seal::specs().iter().map(|(v, s, src, ts)| json!([v, s, src.iter().map(|x| json!([x.0, x.1, x.2])).collect::<Vec<_>>(), ts])).collect();
    assert_eq!(specs, inp["specs"], "SPECS");
    for (r, p) in rtduck::seal::hot_prune().iter().zip(inp["hot_prune"].as_array().unwrap()) {
        assert_eq!(json!([r.0, r.1, r.2, r.3, r.4, r.5, r.7]), json!([p[0], p[1], p[2], p[3], p[4], p[5], p[7]]), "HOT_PRUNE");
        let f = p[6].as_str().unwrap();
        let ok = matches!((r.6, f), (rtduck::seal::CoinOf::Bn, "<lambda>") | (rtduck::seal::CoinOf::Okx, "<lambda>") | (rtduck::seal::CoinOf::Same, "str"))
            || (matches!(r.6, rtduck::seal::CoinOf::All) && r.5 == "liquidations") || (matches!(r.6, rtduck::seal::CoinOf::Colon) && r.4 == "hl");
        assert!(ok, "코인 바꾸기 {:?} vs {f}", r.6);
    }
    assert_eq!(rtduck::seal::hot_prune().len(), inp["hot_prune"].as_array().unwrap().len());
    let d = std::env::var("RTDUCK_KEEP").map(std::path::PathBuf::from).unwrap_or_else(|_| std::env::temp_dir().join(format!("rtduck_sealfull_{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&d);
    for (rel, tables) in inp["files"].as_object().unwrap() {
        let p = d.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        let mut c = rusqlite::Connection::open(&p).unwrap();
        let tx = c.transaction().unwrap();
        for t in tables.as_array().unwrap() {
            tx.execute_batch(t[1].as_str().unwrap()).unwrap();
            let rows = t[2].as_array().unwrap();
            let n = rows[0].as_array().unwrap().len();
            let mut st = tx.prepare(&format!("INSERT INTO {} VALUES ({})", t[0].as_str().unwrap(), vec!["?"; n].join(","))).unwrap();
            for r in rows {
                let v: Vec<rusqlite::types::Value> = r.as_array().unwrap().iter().map(|x| match x {
                    Value::Null => rusqlite::types::Value::Null,
                    Value::String(s) => rusqlite::types::Value::Text(s.clone()),
                    Value::Number(n) if n.is_i64() => rusqlite::types::Value::Integer(n.as_i64().unwrap()),
                    Value::Number(n) => rusqlite::types::Value::Real(n.as_f64().unwrap()),
                    _ => unreachable!(),
                }).collect();
                st.execute(rusqlite::params_from_iter(v)).unwrap();
            }
        }
        tx.commit().unwrap();
    }
    for s in inp["duck"].as_array().unwrap() {
        let p = d.join(s[0].as_str().unwrap());
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        Connection::open(&p).unwrap().execute_batch(s[1].as_str().unwrap()).unwrap();
    }
    let (lake, now) = (d.join("lake"), inp["now_s"].as_f64().unwrap());
    let sp = rtduck::seal::specs();
    let sp: Vec<rtduck::seal::Spec> = sp.iter().map(|(v, s, src, ts)| (*v, s.as_str(), src.clone(), *ts)).collect();
    let (st1, log1) = rtduck::seal::seal(&sp, &d, &lake, now, false, None).unwrap();
    assert_eq!(Value::Object(st1), gold["run1"]["stats"], "1회 stats {log1:?}");
    let l1 = describe(&lake);
    for (f, v) in gold["run1"]["lake"].as_object().unwrap() {
        assert_eq!(l1[f], *v, "1회 {f}");
    }
    assert_eq!(l1, gold["run1"]["lake"], "1회 파일 목록");
    for m in inp["mods"].as_array().unwrap() {
        rusqlite::Connection::open(d.join(m[0].as_str().unwrap())).unwrap().execute_batch(m[1].as_str().unwrap()).unwrap();
    }
    let status = d.join("status.json");
    let (st2, log2, code) = rtduck::seal::main_once(&d, &lake, Some(&status), now, false, None, true).unwrap();
    assert_eq!(code, 0);
    assert_eq!(Value::Object(st2.clone()), gold["run2"]["stats"], "2회 stats {log2:?}");
    assert_eq!(std::fs::read_to_string(&status).unwrap(), gold["run2"]["status"].as_str().unwrap(), "seal_status 글자");
    let l2 = describe(&lake);
    assert_eq!(l2, gold["run2"]["lake"], "2회 lake");
    assert_eq!(hot_counts(&d), gold["run2"]["hot"], "hot 남은 행");
    let streams: std::collections::BTreeSet<String> = l2.as_object().unwrap().keys().map(|k| k.split('/').take(2).collect::<Vec<_>>().join("/")).collect();
    if std::env::var("RTDUCK_KEEP").is_err() {
        let _ = std::fs::remove_dir_all(&d);
    }
    eprintln!("seal 전 스트림 일치: 스트림 {} · lake 파일 {} · 행 {} · 2회 {:?}", streams.len(), l2.as_object().unwrap().len(),
              l2.as_object().unwrap().values().map(|v| v["rows"].as_array().unwrap().len()).sum::<usize>(), st2);
}
