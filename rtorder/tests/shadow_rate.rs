//! 그림자 가중치 상한(2026-11-21 · 서버 실측 분당 최고 1,431 + 그림자 → 밴 위험 때문에): 127.0.0.1 가짜 거래소 + 가짜 사용자 스트림(2초마다 체결 알림)으로
//! 무장 하나가 있는 그림자 rtorder 를 **5분** 돌린다(이 파일은 약 5분 걸린다).
//! ① 분 창마다 거래소가 받은 요청의 가중치 합 ≤ 상한(감시 틱 60초 캐시 1회 22 + 체결 뒤 다시 읽기 30초 간격 최대 3회 66 + listenKey = 90 미만) · userTrades 0
//! ② 응답 머리 X-MBX-USED-WEIGHT-1M 이 1,800 을 넘으면 그 분 끝까지 그림자는 REST 를 보내지 않는다 · 다음 분에 다시
//! ③ 계량 한 줄(«rtorder 요청 계량 …»)이 분마다 남는다. 키는 합성 값 · 쓰기는 0.
include!("common/server.rs");

use futures_util::SinkExt;
use rtorder::app::{self, Stop};
use rtorder::http::{meter, Binance};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

fn wall() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64()
}

#[tokio::test(flavor = "current_thread")]
async fn shadow_five_minutes_stays_under_weight_cap_and_pauses_above_1800() {
    std::env::set_var("DASHBOARD_MANUAL_EXEC_ENABLED", "1");
    let fx: Value = serde_json::from_str(include_str!("fixtures/desk_fixture.json")).unwrap();
    let routes = fx["account"].as_array().unwrap().iter().find(|c| c["name"] == "normal_usdc").unwrap()["in"]["routes"].clone();
    // 받은 요청 (시각, 메서드, 대상) · 응답에 실을 사용 가중치(바꿀 수 있다)
    let got: Arc<Mutex<Vec<(f64, String, String)>>> = Arc::new(Mutex::new(vec![]));
    let used = Arc::new(Mutex::new((0u64, 0.0f64)));   // (알릴 사용 가중치 · 알린 시각)
    let (g2, u2) = (got.clone(), used.clone());
    let srv = serve(Arc::new(move |g: &Got| {
        g2.lock().unwrap().push((wall(), g.method.clone(), g.target.clone()));
        let path = g.target.split('?').next().unwrap();
        let body = match path {
            "/fapi/v1/time" => json!({"serverTime": (wall() * 1000.0) as i64}),
            "/fapi/v2/account" => routes["/fapi/v2/account"].clone(),
            "/fapi/v2/positionRisk" => json!([{"symbol": "ETHUSDC", "positionSide": "LONG", "positionAmt": "0.500", "entryPrice": "2400.5", "markPrice": "2450.25",
                                                "liquidationPrice": "1900.1", "leverage": "10", "notional": "1225.125", "unRealizedProfit": "24.875", "updateTime": 1799990000123i64}]),
            "/fapi/v1/listenKey" => json!({"listenKey": "lk-rate"}),
            _ => json!([]),
        };
        // 1,900 은 한 번만(그 응답의 시각을 적는다) -- 요청이 드문드문이라 시각 창으로 주면 못 받을 수 있다
        let mut uu = u2.lock().unwrap();
        let u = if uu.0 > 0 && uu.1 == 0.0 { uu.1 = wall(); uu.0 } else { 500 };
        drop(uu);
        Act::Send { status: 200, ctype: "application/json".into(), body: body.to_string(), headers: vec![("X-MBX-USED-WEIGHT-1M".into(), u.to_string())] }
    }));
    local_only(&srv.base);
    // 가짜 사용자 스트림: 붙으면 2초마다 체결 알림(계좌 다시 읽기를 몰아 부른다)
    let lis = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let ws = format!("ws://{}/private/ws/", lis.local_addr().unwrap());
    tokio::spawn(async move {
        loop {
            let Ok((s, _)) = lis.accept().await else { return };
            tokio::spawn(async move {
                let Ok(mut w) = tokio_tungstenite::accept_async(s).await else { return };
                let ev = json!({"e": "ACCOUNT_UPDATE", "a": {"P": [{"s": "ETHUSDC", "ps": "LONG", "pa": "0.5", "ep": "2400.5", "up": "24.875"}]}});
                loop {
                    if w.send(tokio_tungstenite::tungstenite::Message::Text(ev.to_string().into())).await.is_err() {
                        return;
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                }
            });
        }
    });
    let dir = std::env::temp_dir().join(format!("rtorder_shadow_rate_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // 무장 하나(포지션 있음 · 손절선은 멀다) -- 감시 틱이 계좌를 읽는다
    let arm = json!({"ETHUSDC:LONG": {"symbol": "ETHUSDC", "side": "LONG", "sl_price": 1000.0, "sl_level": 1000.0, "tp_price": 9000.0, "tp_level": 9000.0,
                                       "backstop_price": null, "rule": false, "lines": true, "tp_resting": true, "armed_at": wall() - 600.0, "placed": {"placed": true}}});
    std::fs::write(dir.join("manual_bracket_state.json"), arm.to_string()).unwrap();
    let (d, base, wsb) = (dir.to_string_lossy().to_string(), srv.base.clone(), ws.clone());
    let get = move |k: &str| match k {
        "RTORDER_DIR" => Some(d.clone()),
        "RTORDER_BASE" => Some(base.clone()),
        "RTORDER_WS" => Some(wsb.clone()),
        "RTORDER_STREAM" => Some("1".into()),
        "RTORDER_PERIOD" => Some("0.2".into()),
        _ => None,
    };
    let cfg = app::config_from(&get).unwrap();
    assert_eq!((cfg.shadow_account_ttl, cfg.meter_sec), (60.0, 60.0), "기본값");
    let ban = dir.join("ban");
    let base = srv.base.clone();
    let mk = move || {
        let mut b = Binance::new(&base, "test_key", "test_secret", std::time::Duration::from_secs(3), ban.clone());
        b.shadow_on();
        b
    };
    let stop = Stop::default();
    let hub: Arc<rtorder::userstream::Hub> = Default::default();
    let t0 = wall();
    // 3분째 어느 순간 거래소가 «이번 분 1,900» 을 알린다 → 그 분 끝까지 쉼 → 다음 분부터 다시
    let (s2, u3) = (stop.clone(), used.clone());
    let set_at = Arc::new(Mutex::new(0.0f64)); // 1,900 을 켠 벽시계(개발기 WSL 벽시계가 tokio 잠보다 3초 넘게 늦게 간 적이 있다 -- t0+150 으로 보면 안 된다)
    let sa = set_at.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(150)).await;
        *sa.lock().unwrap() = wall();
        u3.lock().unwrap().0 = 1900;
        tokio::time::sleep(std::time::Duration::from_secs(150)).await;
        s2.stop();
    });
    let s = app::run(&cfg, &mk, stop, hub, None).await;
    let got = got.lock().unwrap().clone();
    // ① 분 창(시작부터 60초씩)마다 가중치 합
    let mut per = vec![0u64; 6];
    for (t, _m, target) in &got {
        per[(((t - t0) / 60.0) as usize).min(5)] += meter::weight_of(&format!("http://x{target}"));
    }
    println!("분 창별 가중치 {per:?} · 요청 {}", got.len());
    assert!(per.iter().all(|w| *w < 90), "분당 가중치 상한(감시 22 + 체결 뒤 최대 3회 66 + listenKey) {per:?}");
    assert!(per[1] > 0 && per[3] > 0, "그림자도 계좌는 읽는다 {per:?}");
    assert!(got.iter().all(|x| !x.2.starts_with("/fapi/v1/userTrades")), "그림자는 userTrades 를 건너뛴다");
    assert!(got.iter().all(|x| x.1 == "GET" || x.2.starts_with("/fapi/v1/listenKey")), "쓰기 0");
    // ② 1,900 을 받은 뒤 그 분 경계까지 요청 0 · 다음 분에 다시
    let hit = used.lock().unwrap().1;
    let on = *set_at.lock().unwrap();
    assert!(on > 0.0 && hit >= on, "1,900 을 알린 응답이 있어야 한다(켠 뒤 {:.1}초 · 켠 뒤 요청 {}건)", hit - on, got.iter().filter(|x| x.0 >= on).count());
    let boundary = ((hit / 60.0).floor() + 1.0) * 60.0;
    let between: Vec<_> = got.iter().filter(|x| x.0 > hit + 0.5 && x.0 < boundary - 0.5).collect();
    assert!(between.is_empty(), "1,800 을 넘은 분에는 쉰다 {between:?}");
    assert!(got.iter().any(|x| x.0 > boundary + 0.5), "다음 분에는 다시 보낸다");
    // ③ 계량 줄
    let lines: Vec<&String> = s.logs.iter().filter(|l| l.starts_with("rtorder 요청 계량")).collect();
    assert!(lines.len() >= 4, "{:?}", s.logs);
    assert!(lines.iter().all(|l| l.contains("· 그림자")));
    assert!(lines.iter().any(|l| !l.contains("쉰 요청 0")), "쉰 요청이 계량 줄에 보인다 {lines:?}");
    let _ = std::fs::remove_dir_all(&dir);
}
