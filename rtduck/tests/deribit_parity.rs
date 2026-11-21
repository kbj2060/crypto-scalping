//! Deribit 첫 단계 일치: tools/gen_deribit_fixture.py 가 원본 write()(세 번)·backfill() 을 돌린 정답지(서버 판 venv · duckdb 1.5.1)와
//!   상태 JSON 글자(블록 구조 13종+ · 헤지 · 흐름 25칸 × 코인 넷) · 새 행 수 · option_trades·gaps 덤프 · backfill 요청 URL·행을 대조한다.
use duckdb::types::Value as D;
use serde_json::{json, Value};

fn dv(v: &D) -> Value {
    match v {
        D::Null => Value::Null,
        D::BigInt(x) => json!(x),
        D::Double(x) => json!(x),
        D::Text(s) => json!(s),
        D::Boolean(b) => json!(b),
        o => panic!("{o:?}"),
    }
}

fn first_diff(a: &str, b: &str) -> String {
    let i = a.bytes().zip(b.bytes()).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
    format!("@{i}: rust …{}… / python …{}…", &a[i.saturating_sub(120)..(i + 120).min(a.len())], &b[i.saturating_sub(120)..(i + 120).min(b.len())])
}

#[test]
fn deribit_write_state_backfill_match_python() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/deribit_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/deribit_golden.json")).unwrap();
    let d = std::env::temp_dir().join(format!("rtduck_deribit_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let (db, state) = (d.join("deribit_options.duckdb"), d.join("deribit_block_trades_state.json"));
    {
        let c = rtduck::deribit::connect(&db).unwrap();
        for f in inp["fut"].as_array().unwrap() {
            c.execute("INSERT INTO block_future_legs VALUES (?,?,?,?,?,?,?,?)", duckdb::params![f[0].as_str(), f[1].as_i64(), f[2].as_str(), f[3].as_str(), f[4].as_f64(), f[5].as_f64(), f[6].as_f64(), f[7].as_str()]).unwrap();
        }
    }
    let (now, iso) = (inp["now_s"].as_f64().unwrap(), inp["iso_now"].as_str().unwrap());
    for (k, ((batch, gaps), g)) in inp["batches"].as_array().unwrap().iter().zip(inp["gaps"].as_array().unwrap()).zip(gold["writes"].as_array().unwrap()).enumerate() {
        let rows: Vec<Vec<D>> = batch.as_array().unwrap().iter().map(|t| rtduck::deribit::row(t).unwrap()).collect();
        let gaps: Vec<(i64, i64, String, Option<i64>)> = gaps.as_array().unwrap().iter().map(|x| (x[0].as_i64().unwrap(), x[1].as_i64().unwrap(), x[2].as_str().unwrap().to_string(), x[3].as_i64())).collect();
        let added = rtduck::deribit::write(&db, &state, &rows, &gaps, now, iso).unwrap();
        assert_eq!(json!(added), g["added"], "{k}회 새 행");
        let got = std::fs::read_to_string(&state).unwrap();
        let want = g["state"].as_str().unwrap();
        assert!(got == want, "{k}회 상태 JSON {}", first_diff(&got, want));
    }
    let c = duckdb::Connection::open(&db).unwrap();
    for t in ["option_trades", "gaps"] {
        let mut st = c.prepare(&format!("SELECT * FROM {t} ORDER BY ALL")).unwrap();
        let mut rows = st.query([]).unwrap();
        let mut got = vec![];
        while let Some(r) = rows.next().unwrap() {
            let n = r.as_ref().column_count();
            got.push(Value::Array((0..n).map(|i| match r.get_ref(i).unwrap() {
                duckdb::types::ValueRef::Null => Value::Null,
                duckdb::types::ValueRef::BigInt(x) => json!(x),
                duckdb::types::ValueRef::Int(x) => json!(x),
                duckdb::types::ValueRef::Double(x) => json!(x),
                duckdb::types::ValueRef::Boolean(b) => json!(b),
                duckdb::types::ValueRef::Text(s) => json!(String::from_utf8_lossy(s)),
                v => panic!("{v:?}"),
            }).collect()));
        }
        assert_eq!(Value::Array(got), gold["dump"][t], "{t} 덤프");
    }
    drop(c);
    // backfill
    let pages = &inp["pages"];
    let mut served = std::collections::HashMap::<String, usize>::new();
    let mut calls = vec![];
    let since: Vec<i64> = inp["since"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
    let bf = rtduck::deribit::backfill(&since, (now * 1000.0) as i64, &mut |u: &str| {
        calls.push(u.to_string());
        let api = u.split("currency=").nth(1).unwrap().split('&').next().unwrap().to_string();
        let k = served.entry(api.clone()).or_default();
        let ps = pages[&api].as_array().unwrap();
        let p = &ps[(*k).min(ps.len() - 1)];
        *k += 1;
        Ok(serde_json::to_vec(&json!({"result": p})).unwrap())
    }).unwrap();
    let want: Vec<String> = gold["backfill"]["calls"].as_array().unwrap().iter().map(|c| {
        let q: Vec<String> = c[1].as_object().unwrap().iter().map(|(k, v)| format!("{k}={}", v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()))).collect();
        format!("{}?{}", c[0].as_str().unwrap(), q.join("&"))
    }).collect();
    assert_eq!(calls, want, "backfill 요청");
    assert!(gold["backfill"]["calls"].as_array().unwrap().iter().all(|c| c[2] == 20), "timeout 20");
    let got: Vec<Value> = bf.iter().map(|r| Value::Array(r.iter().map(dv).collect())).collect();
    assert_eq!(Value::Array(got), gold["backfill"]["rows"], "backfill 행");
    let st: Value = serde_json::from_str(&std::fs::read_to_string(&state).unwrap()).unwrap();
    let _ = std::fs::remove_dir_all(&d);
    eprintln!("deribit 일치: 쓰기 3번(상태 JSON 글자 · 블록 {} · 흐름 칸 {}) · option_trades {}행 · gaps {}행 · backfill 요청 {}·행 {}",
              st["n_blocks_by_coin"], 4 * 25, gold["dump"]["option_trades"].as_array().unwrap().len(), gold["dump"]["gaps"].as_array().unwrap().len(), calls.len(), bf.len());
}
