//! Deribit ② 일치: tools/gen_deribit_gex_fixture.py 가 원본 poll_once 를 두 번 돌린 정답지(서버 판 venv · TZ=Asia/Seoul)와
//!   요청(URL·쿼리 순서·timeout · 캐시로 빠지는 것까지) · deribit_gex_state.json 글자 · option_chain_snapshot·option_summary·option_oi_other 덤프를 대조한다.
use serde_json::Value;

fn first_diff(a: &str, b: &str) -> String {
    let i = a.bytes().zip(b.bytes()).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
    format!("@{i}: rust …{}… / python …{}…", &a[i.saturating_sub(150)..(i + 150).min(a.len())], &b[i.saturating_sub(150)..(i + 150).min(b.len())])
}

#[test]
fn deribit_gex_poll_matches_python() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/deribit_gex_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/deribit_gex_golden.json")).unwrap();
    let d = std::env::temp_dir().join(format!("rtduck_gex_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let con = duckdb::Connection::open(d.join("d.duckdb")).unwrap();
    rtduck::deribit_gex::ensure_tables(&con).unwrap();
    for h in inp["hist"].as_array().unwrap() {
        con.execute("INSERT INTO option_summary VALUES (make_timestamp(?)::TIMESTAMPTZ, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            duckdb::params![h[0].as_i64(), h[1].as_str(), h[2].as_f64(), h[3].as_f64(), h[4].as_f64(), h[5].as_f64(), h[6].as_f64(), h[7].as_f64(), h[8].as_f64(), h[9].as_str()]).unwrap();
    }
    let state = d.join("deribit_gex_state.json");
    let mut cache = rtduck::deribit_gex::Cache::default();
    let now0 = (inp["now_s"].as_f64().unwrap() * 1e6).round() as i64;
    let mut ncalls = 0;
    for (k, g) in gold["polls"].as_array().unwrap().iter().enumerate() {
        let calls = g["calls"].as_array().unwrap();
        let mut i = 0;
        let log = rtduck::deribit_gex::poll_once(&con, &state, now0 + k as i64 * 600_000_000, g["db_now_us"].as_i64().unwrap(), inp["tz_offset_s"].as_i64().unwrap(), &mut cache,
            &mut |u: &str, timeout: u64, ua: bool| {
                let c = &calls[i];
                i += 1;
                let q: Vec<String> = c[1].as_object().unwrap().iter().map(|(k, v)| format!("{k}={}", v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()))).collect();
                assert_eq!(u, format!("{}?{}", c[0].as_str().unwrap(), q.join("&")), "{k}회 요청 {i}");
                assert_eq!(serde_json::json!(timeout), c[2], "timeout {u}");
                assert_eq!(ua, c[3].get("User-Agent").is_some(), "UA {u}");
                if c[4].as_i64().unwrap() >= 400 { Err(format!("HTTPError {}", c[4])) } else { Ok(serde_json::to_vec(&c[5]).unwrap()) }
            }).unwrap();
        assert_eq!(i, calls.len(), "{k}회 요청 수 {log:?}");
        ncalls += i;
        let got = std::fs::read_to_string(&state).unwrap();
        let want = g["state"].as_str().unwrap();
        assert!(got == want, "{k}회 상태 {}", first_diff(&got, want));
    }
    let mut rows_n = 0;
    for t in ["option_chain_snapshot", "option_summary", "option_oi_other"] {
        let desc: Vec<(String, String)> = con.prepare(&format!("DESCRIBE {t}")).unwrap().query_map([], |r| Ok((r.get(0)?, r.get(1)?))).unwrap().map(|x| x.unwrap()).collect();
        let sel: Vec<String> = desc.iter().map(|(c, ty)| if ty.contains("TIMESTAMP") { format!("epoch_us({c})") } else if ty == "DOUBLE" { format!("CASE WHEN isnan({c}) THEN 'NaN' ELSE {c}::VARCHAR END") } else { c.clone() }).collect();
        let n = desc.len();
        let rows: Vec<Value> = con.prepare(&format!("SELECT {} FROM {t} ORDER BY ALL", sel.join(", "))).unwrap().query_map([], |r| Ok(Value::Array((0..n).map(|i| match r.get_ref(i).unwrap() {
            duckdb::types::ValueRef::Null => Value::Null,
            duckdb::types::ValueRef::BigInt(x) => Value::from(x),
            duckdb::types::ValueRef::Text(s) => Value::from(String::from_utf8_lossy(s).to_string()),
            v => panic!("{v:?}"),
        }).collect()))).unwrap().map(|x| x.unwrap()).collect();
        for (a, b) in rows.iter().zip(gold["dump"][t].as_array().unwrap()) {
            assert_eq!(a, b, "{t}");
        }
        assert_eq!(rows.len(), gold["dump"][t].as_array().unwrap().len(), "{t} 행 수");
        rows_n += rows.len();
    }
    drop(con);
    let _ = std::fs::remove_dir_all(&d);
    let st: Value = serde_json::from_str(gold["polls"][1]["state"].as_str().unwrap()).unwrap();
    eprintln!("deribit ② 일치: 폴링 2 · 요청 {ncalls} · 상태 글자 2 · 코인 {} · 표 행 {rows_n}", st["currencies"].as_object().unwrap().len());
}
