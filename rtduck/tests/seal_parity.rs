//! seal_to_lake 첫 표(binance/oi_lsratio_5m) 일치: tools/gen_seal_fixture.py 가 원본 seal() 을 두 번 돌린 정답지(서버 판 duckdb 1.5.1)와
//!   stats 사전 · lake 파일 이름 · 열 이름·형 · 행 값(정렬해서 -- 원본 COPY 는 threads 2 라 행 순서가 정해지지 않는다)을 대조한다.
//!   반대 방향: RTDUCK_KEEP=<폴더> 로 돌리면 lake 를 남긴다 → `$PY tools/gen_seal_fixture.py --check <폴더>/lake`.
use duckdb::Connection;
use serde_json::{json, Value};
use std::path::Path;

fn run_sql(root: &Path, stmts: &Value) {
    for s in stmts.as_array().unwrap() {
        let p = root.join(s[0].as_str().unwrap());
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        Connection::open(&p).unwrap().execute_batch(s[1].as_str().unwrap()).unwrap();
    }
}

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
        let cols: Vec<Value> = c.prepare(&format!("describe select * from {q}")).unwrap().query_map([], |r| Ok(json!([r.get::<_, String>(0)?, r.get::<_, String>(1)?])))
            .unwrap().map(|x| x.unwrap()).collect();
        let rows: Vec<Value> = c.prepare(&format!("select epoch_ms(ts), symbol, sum_open_interest, sum_open_interest_value, global_ls_ratio, global_ls_long_account, epoch_ms(collected_at) from {q} order by all"))
            .unwrap().query_map([], |r| Ok(json!([r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, Option<f64>>(2)?, r.get::<_, Option<f64>>(3)?,
                                                 r.get::<_, Option<f64>>(4)?, r.get::<_, Option<f64>>(5)?, r.get::<_, i64>(6)?]))).unwrap().map(|x| x.unwrap()).collect();
        out.insert(p.strip_prefix(lake).unwrap().to_string_lossy().to_string(), json!({"cols": cols, "rows": rows}));
    }
    Value::Object(out)
}

#[test]
fn seal_oi_lsratio_matches_python() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/seal_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/seal_golden.json")).unwrap();
    let d = std::env::var("RTDUCK_KEEP").map(std::path::PathBuf::from).unwrap_or_else(|_| std::env::temp_dir().join(format!("rtduck_seal_{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&d);
    run_sql(&d, &inp["build"]);
    // 원본 SPECS 의 binance/oi_lsratio_5m 그대로
    let (l, a) = ("data/live", "data/archive/live_retired_20261001");
    let sfx = |c: &str| if c == "eth" { String::new() } else { format!("_{c}") };
    let mut src: Vec<(String, String, String)> = ["eth", "btc", "sol", "xrp"].iter().map(|c| (format!("{l}/oi_lsratio.duckdb"), format!("oi_lsratio_5m{}", sfx(c)), format!("'{}'", c.to_uppercase()))).collect();
    src.extend(["xrp", "hype"].iter().map(|c| (format!("{a}/oi_lsratio_{c}.duckdb"), format!("oi_lsratio_5m_{c}"), format!("'{}'", c.to_uppercase()))));
    let specs = vec![("binance", "oi_lsratio_5m", src, "epoch_ms(ts)")];
    let lake = d.join("lake");
    let mut summary = vec![];
    for (k, g) in gold["runs"].as_array().unwrap().iter().enumerate() {
        if k == 1 {
            run_sql(&d, &inp["mods"]);
        }
        let (st, log) = rtduck::seal::seal(&specs, &d, &lake, inp["now_s"].as_f64().unwrap(), false, None).unwrap();
        assert_eq!(Value::Object(st.clone()), g["stats"], "{k}회 stats {log:?}");
        let got = describe(&lake);
        for (f, v) in g["lake"].as_object().unwrap() {
            assert_eq!(got[f], *v, "{k}회 {f}");
        }
        assert_eq!(got, g["lake"], "{k}회 lake 파일 목록");
        summary.push(format!("{}회 {} 파일·{} 행 · {:?}", k + 1, got.as_object().unwrap().len(), got.as_object().unwrap().values().map(|v| v["rows"].as_array().unwrap().len()).sum::<usize>(), st));
    }
    if std::env::var("RTDUCK_KEEP").is_err() {
        let _ = std::fs::remove_dir_all(&d);
    }
    eprintln!("seal 일치(binance/oi_lsratio_5m · 원본 duckdb {}): {summary:?}", inp["duckdb"]);
}
