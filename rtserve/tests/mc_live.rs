//! /api/market-context 붙이기(src/mc_live.rs) -- 계산 글자는 ctx_mc_parity 가 본다. 여기는 재료 사슬: 허브 Mark → 마크 링 → 펀딩·베이시스 · 코인 밖 404 ·
//! REST 꺼짐 = 펀딩 분위 없음(원본 실패 갈래 []) · 파일 없음 = 수집기 몫 null.
mod common;

use serde_json::Value;

fn mark(s: &str, seq: u64, ex_ms: i64, mark: f64, index: f64, funding: f64) -> String {
    format!(r#"{{"s":"{s}","seq":{seq},"ex_ms":{ex_ms},"rx_us":{},"tx_us":0,"type":"mark","mark":{mark},"index":{index},"funding":{funding},"next_funding_ms":0}}"#, ex_ms * 1000)
}

#[test]
fn market_context_wiring() {
    let now = 1.795e9;
    let (addr, root) = common::start_with(&serde_json::Map::new(), 0, now, "eth,sol", "mc_live", |c| {
        let t = (now * 1000.0) as i64;
        c.live.apply(&mark("ethusdt", 1, t - 2000, 2500.0, 2500.0, 0.0002));
        c.live.apply(&mark("ethusdt", 2, t - 1000, 2500.5, 2500.0, 0.0001));
    });
    let get = |p: &str| {
        let g = common::request(addr, "GET", p, &[]);
        (g.status, serde_json::from_slice::<Value>(&g.body).unwrap_or(Value::Null))
    };
    let (st, e) = get("/api/market-context");
    assert_eq!(st, 200);
    assert_eq!(e["available"], true);
    assert_eq!(e["asset"], "eth");
    assert_eq!(e["funding"]["bn"].as_f64(), Some(0.0001), "마지막 마크의 펀딩");
    assert_eq!(e["funding"]["bn_at_base"], true);
    assert!((e["basis"]["bp"].as_f64().unwrap() - 2.0).abs() < 1e-9, "베이시스 = (2500.5−2500)/2500·1e4");
    assert_eq!(e["mid"].as_f64(), Some(2500.5), "미시 없음 → 마크로 물러섬(원본 mp.get(mid) or bn_mark)");
    assert_eq!(e["funding"]["bn_pct180"], Value::Null, "REST 꺼짐 = 펀딩 이력 []");
    assert_eq!(e["oi"]["okx"], Value::Null);
    assert_eq!(e["profile"]["available"], false, "테이프 없음 = 레벨만 빠짐");
    assert!(e["burst"].is_null());
    // 양성 대조: 마크를 안 받은 코인은 펀딩·베이시스가 빈다
    let (st, s) = get("/api/market-context?asset=SOL");
    assert_eq!((st, &s["asset"], &s["funding"]["bn"], &s["basis"]["bp"]), (200, &Value::from("sol"), &Value::Null, &Value::Null));
    assert_eq!(get("/api/market-context?asset=xrp").0, 404, "꺼진 코인 = flow_off");
    let _ = std::fs::remove_dir_all(root);
}
