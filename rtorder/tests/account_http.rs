//! 계좌 조회 동시화(2026-11-21 여섯째 조각 E): 실제 `Binance`(HTTP) 로 127.0.0.1 가짜 서버에 fetch_account 를 보내 --
//! 원본 정답지(tests/fixtures/desk_fixture.json 의 account 사례 · 파이썬 fetch_account + gather)와 «같은 요청 집합(순서 무관)» · «같은 스냅숏».
include!("common/server.rs");

use rtorder::account::{fetch_account, DEFAULT_TRADE_LIMIT};
use rtorder::http::Binance;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

fn qmap(q: &str) -> Map<String, Value> {
    q.split('&').filter(|x| !x.is_empty()).filter_map(|kv| kv.split_once('=')).filter(|(k, _)| !["recvWindow", "timestamp", "signature"].contains(k))
        .map(|(k, v)| (k.to_string(), json!(v))).collect()
}

#[tokio::test]
async fn concurrent_fetch_account_same_request_set_and_snapshot_as_python() {
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/desk_fixture.json")).unwrap()).unwrap();
    let mut n = 0;
    // trades_fail_one(429)은 뺀다: 정답지 생성기가 밴 판정을 상수로 막아 두어 원본은 뒤 요청을 보내지만, 실제로는(Rust·원본 모두) 429 기록 뒤 요청이 밴으로 막힌다
    for name in ["normal_usdc", "multi_assets", "orders_fail", "fold_truncated_resync", "fold_flip_both"] {
        let c = fx["account"].as_array().unwrap().iter().find(|c| c["name"] == name).unwrap();
        let routes: HashMap<String, Value> = c["in"]["routes"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let srv = serve(Arc::new(move |g: &Got| {
            let (path, q) = g.target.split_once('?').unwrap_or((&g.target, ""));
            let m = qmap(q);
            let key = match (m.get("symbol").and_then(Value::as_str), path) {
                (Some(s), "/fapi/v1/userTrades" | "/fapi/v1/openOrders") => format!("{path}?{s}"),
                _ => path.to_string(),
            };
            let r = routes.get(&key).or(routes.get(path)).cloned().unwrap_or(json!({"__status__": 599, "__body__": {"msg": "각본 없음"}}));
            let (status, body) = match r.get("__status__") {
                Some(s) => (s.as_u64().unwrap() as u16, r["__body__"].clone()),
                None => (200, r),
            };
            Act::Send { status, ctype: "application/json".into(), body: body.to_string(), headers: vec![] }
        }));
        local_only(&srv.base);
        let ban = std::env::temp_dir().join(format!("rtorder_account_http_{}_{n}", std::process::id()));
        let _ = std::fs::remove_file(&ban);
        let mut bx = Binance::new(&srv.base, "test_key", "test_secret", std::time::Duration::from_secs(5), ban.clone());
        bx.clock = Arc::new(|| 1_800_000_000.0);
        let syms: Vec<String> = c["in"]["symbols"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().into()).collect();
        let out = fetch_account(&mut bx, &syms, "USDC", DEFAULT_TRADE_LIMIT).await.unwrap();
        // 스냅숏 = 원본과 같다(수는 값으로)
        assert_eq!(norm(&out), norm(&c["out"]), "{name}: 스냅숏");
        // 요청 집합 = 원본과 같다(동시에 보내므로 순서는 안 본다)
        let mut got: Vec<String> = srv.log.lock().unwrap().iter().map(|g| { let (p, q) = g.target.split_once('?').unwrap_or((&g.target, "")); json!([p, qmap(q)]).to_string() }).collect();
        let mut want: Vec<String> = c["calls"].as_array().unwrap().iter().map(|x| x.to_string()).collect();
        got.sort();
        want.sort();
        assert_eq!(got, want, "{name}: 요청 집합");
        let _ = std::fs::remove_file(&ban);
        n += 1;
    }
    assert_eq!(n, 5);
}

/// 수를 f64 로 맞춰 비교(파이썬 int/float 표기 차이 무시).
fn norm(v: &Value) -> Value {
    match v {
        Value::Number(n) => json!(n.as_f64().unwrap()),
        Value::Array(a) => Value::Array(a.iter().map(norm).collect()),
        Value::Object(m) => Value::Object(m.iter().map(|(k, x)| (k.clone(), norm(x))).collect()),
        x => x.clone(),
    }
}
