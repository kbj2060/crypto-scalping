//! Deribit ③ 일치: tools/gen_deribit_recon_fixture.py 가 원본 reconcile() → write() 를 돌린 정답지(서버 판 venv)와
//!   요청(URL·쿼리 순서 · RFQ continuation · 실패 건너뜀) · 결과 사전 · option_trades·block_future_legs·block_rfq_trades 덤프 · 상태 JSON 글자(헤지 다리 · ΔOI)를 대조한다.
use serde_json::{json, Value};

#[test]
fn deribit_reconcile_matches_python() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/deribit_recon_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/deribit_recon_golden.json")).unwrap();
    let d = std::env::temp_dir().join(format!("rtduck_recon_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let (db, state) = (d.join("deribit_options.duckdb"), d.join("deribit_block_trades_state.json"));
    {
        let c = rtduck::deribit::connect(&db).unwrap();
        rtduck::deribit_gex::ensure_tables(&c).unwrap();
        for t in inp["seed"].as_array().unwrap() {
            let r = rtduck::deribit::row(t).unwrap();
            c.execute(&format!("INSERT INTO option_trades VALUES ({})", vec!["?"; 18].join(",")), duckdb::params_from_iter(r.iter())).unwrap();
        }
        for f in inp["fut0"].as_array().unwrap() {
            c.execute("INSERT INTO block_future_legs VALUES (?,?,?,?,?,?,?,?)", duckdb::params![f[0].as_str(), f[1].as_i64(), f[2].as_str(), f[3].as_str(), f[4].as_f64(), f[5].as_f64(), f[6].as_f64(), f[7].as_str()]).unwrap();
        }
        for f in inp["rfq0"].as_array().unwrap() {
            c.execute("INSERT INTO block_rfq_trades VALUES (?,?,?,?,?,?,?,?,?,?,?)", duckdb::params![f[0].as_str(), f[1].as_i64(), f[2].as_str(), f[3].as_str(), f[4].as_f64(), f[5].as_f64(),
                      f[6].as_f64(), f[7].as_str(), f[8].as_str(), f[9].as_str(), f[10].as_str()]).unwrap();
        }
        for q in inp["chain"].as_array().unwrap() {
            c.execute_batch(q.as_str().unwrap()).unwrap();
        }
    }
    // 🔴헤지 다리 조회 순서 = 원본 GROUP BY(ORDER BY 없음)의 해시 순서 -- 두 쪽 다 정해지지 않으므로 URL 로 짝짓는다(같은 URL 은 한 번씩)
    let calls = gold["calls"].as_array().unwrap();
    let mut used = vec![false; calls.len()];
    let mut i = 0;
    let mut log = vec![];
    let now_s = inp["now_s"].as_f64().unwrap();
    let res = rtduck::deribit::reconcile(&db, (now_s * 1000.0) as i64, &mut |u: &str| {
        let url_of = |c: &Value| {
            let q: Vec<String> = c[1].as_object().unwrap().iter().map(|(k, v)| format!("{k}={}", v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()))).collect();
            format!("{}?{}", c[0].as_str().unwrap(), q.join("&"))
        };
        let j = (0..calls.len()).find(|j| !used[*j] && url_of(&calls[*j]) == u).unwrap_or_else(|| panic!("정답지에 없는 요청 {u}"));
        used[j] = true;
        i += 1;
        let c = &calls[j];
        assert_eq!(c[2], 30, "timeout 30");
        if c[3].as_i64().unwrap() >= 400 { Err(format!("HTTPError {}", c[3])) } else { Ok(serde_json::to_vec(&c[4]).unwrap()) }
    }, &mut |l| log.push(l)).unwrap();
    assert_eq!(i, calls.len(), "요청 수 {log:?}");
    assert_eq!(Value::Object(res), gold["result"], "결과");
    rtduck::deribit::write_with(&db, &state, &[], &[], now_s, inp["iso_now"].as_str().unwrap(), gold["db_now_us"].as_i64().unwrap()).unwrap();
    let got = std::fs::read_to_string(&state).unwrap();
    let want = gold["state"].as_str().unwrap();
    if got != want {
        let k = got.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(0);
        panic!("상태 @{k}: rust …{}… / python …{}…", &got[k.saturating_sub(150)..(k + 150).min(got.len())], &want[k.saturating_sub(150)..(k + 150).min(want.len())]);
    }
    let c = duckdb::Connection::open(&db).unwrap();
    for t in ["option_trades", "block_future_legs", "block_rfq_trades"] {
        let mut st = c.prepare(&format!("SELECT * FROM {t} ORDER BY ALL")).unwrap();
        let mut rows = st.query([]).unwrap();
        let mut out = vec![];
        while let Some(r) = rows.next().unwrap() {
            let n = r.as_ref().column_count();
            out.push(Value::Array((0..n).map(|i| match r.get_ref(i).unwrap() {
                duckdb::types::ValueRef::Null => Value::Null,
                duckdb::types::ValueRef::BigInt(x) => json!(x),
                duckdb::types::ValueRef::Int(x) => json!(x),
                duckdb::types::ValueRef::Double(x) => json!(x),
                duckdb::types::ValueRef::Boolean(b) => json!(b),
                duckdb::types::ValueRef::Text(s) => json!(String::from_utf8_lossy(s)),
                v => panic!("{v:?}"),
            }).collect()));
        }
        assert_eq!(Value::Array(out), gold["dump"][t], "{t}");
    }
    drop(c);
    let st: Value = serde_json::from_str(&got).unwrap();
    let doi = st["flow_by_coin"].as_object().unwrap().values().flat_map(|v| v.as_array().unwrap().iter()).filter(|b| !b["doi"].is_null()).count();
    let _ = std::fs::remove_dir_all(&d);
    eprintln!("deribit ③ 일치: 요청 {i} · {:?} · ΔOI 칸 {doi} · 상태 글자 {}B", gold["result"], got.len());
}
