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

/// 2026-11-28 단계 3: ETH 밖 코인 미시 틱(hub::micro_once · RTSERVE_MICRO_ALL=1 일 때 situation_run 이 부르는 것) → /api/market-context?asset=sol 의
/// mid·book.spread·book.sweep 이 찬다(합성 래스터 + 코어 호가 · REST 0). 양성 대조 = 틱 전엔 셋 다 빈다.
#[test]
fn other_coin_micro_tick_fills_market_context() {
    let now = 1.795e9;
    let held: std::sync::Arc<std::sync::Mutex<Option<rtserve::Config>>> = Default::default();
    let h2 = held.clone();
    let (addr, root) = common::start_with(&serde_json::Map::new(), 0, now, "eth,sol", "mc_live_sol", move |c| *h2.lock().unwrap() = Some(c.clone()));
    let cfg = held.lock().unwrap().clone().unwrap();
    // 합성 래스터(rtcollect raster.rs 형식: 머리 32B `<4sHHfIqII` · 행 `<qif` + f32[n]) -- 미드 150.01 아래 = 매수(+) · 위 = 매도(−)
    let (n, bs, lo) = (40u16, 0.02f32, 7480i32);
    let now_ms = (now * 1000.0) as i64;
    let hour = now_ms - now_ms.rem_euclid(3_600_000);
    let p = rtserve::raster_read::hour_path(&rtserve::raster_read::default_root(&root), "solusdt", hour);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    let mut b = Vec::new();
    b.extend_from_slice(b"FLWR");
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&n.to_le_bytes());
    b.extend_from_slice(&bs.to_le_bytes());
    b.extend_from_slice(&[0u8; 20]);
    for j in 0..=(now_ms - hour) / 1000 {
        let live = j >= (now_ms - hour) / 1000 - 15;
        b.extend_from_slice(&(if live { hour + j * 1000 } else { 0 }).to_le_bytes());
        b.extend_from_slice(&lo.to_le_bytes());
        b.extend_from_slice(&(if live { 150.01f32 } else { f32::NAN }).to_le_bytes());
        for k in 0..n as i32 {
            let px = (lo + k) as f64 * bs as f64;
            let q: f32 = if !live { 0.0 } else if px < 150.0 { 10.0 + (j % 3) as f32 } else if px > 150.02 { -8.0 } else { 0.0 };
            b.extend_from_slice(&q.to_le_bytes());
        }
    }
    std::fs::write(&p, &b).unwrap();
    let get = |q: &str| serde_json::from_slice::<Value>(&common::request(addr, "GET", q, &[]).body).unwrap_or(Value::Null);
    let before = get("/api/market-context?asset=sol");
    assert!(before["mid"].is_null() && before["book"]["spread"].is_null() && before["book"]["sweep"].is_null(), "틱 전엔 빈다: {}", before["book"]);
    cfg.live.apply(r#"{"s":"solusdt","seq":1,"ex_ms":0,"rx_us":0,"tx_us":0,"type":"book","u":1,"bid":150.0,"bid_qty":50.0,"ask":150.02,"ask_qty":40.0}"#);
    let r = cfg.live.runs.lock().unwrap().entry("sol".into()).or_default().clone();
    rtserve::hub::micro_once(&cfg, &mut r.lock().unwrap(), "sol", 1);
    let e = get("/api/market-context?asset=sol");
    assert_eq!(e["mid"].as_f64(), Some(150.01), "미시 mid = 코어 호가 중간");
    assert!((e["book"]["spread"].as_f64().unwrap() - 0.02).abs() < 1e-9, "{}", e["book"]);
    assert!(e["book"]["sweep"].is_object() || e["book"]["sweep"].is_array(), "sweep 이 차야: {}", e["book"]);
    assert!(r.lock().unwrap().micro.baseline_at > now, "첫 기준선은 10분 뒤로(코인 1번)");
    let _ = std::fs::remove_dir_all(root);
}
