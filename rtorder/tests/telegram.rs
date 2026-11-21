//! 텔레그램 알림: 원본 bracket_tick_alert + send_telegram 을 소스에서 꺼내 돌린 정답지(tools/gen_telegram_fixture.py)와
//! 로그 줄 · 보낸 요청(경로·메서드·Content-Type·본문 바이트) · 중복 억제(칸·예외 종류) 대조. 127.0.0.1 가짜 서버만 · 그림자에서는 안 보내고 «안 보낸 알림».
include!("common/server.rs");

use rtorder::app::Ops;
use rtorder::http::Binance;
use rtorder::telegram::Telegram;
use rtorder::watch::Watcher;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

const TOKEN: &str = "123456:test_token";

fn tg(base: &str) -> Telegram {
    let base = base.to_string();
    Telegram::from_env(&move |k: &str| match k {
        "TELEGRAM_BOT_TOKEN" => Some(TOKEN.into()),
        "TELEGRAM_CHAT_ID" => Some("-100987".into()),
        "RTORDER_TELEGRAM_BASE" => Some(base.clone()),
        _ => None,
    })
    .unwrap()
}

fn ops(base: &str, shadow: bool, telegram: Telegram) -> Ops {
    let mut bx = Binance::new(base, "test_key", "test_secret", std::time::Duration::from_secs(3), std::env::temp_dir().join("rtorder_tg_ban"));
    if shadow {
        bx.shadow_on();
    }
    Ops { bx, feed: Default::default(), hub: Default::default(), stop: Default::default(), stopped: Default::default(), would: Arc::new(Mutex::new(vec![])), alerts: vec![], telegram: Some(telegram) }
}

#[tokio::test]
async fn alerts_match_python_bytes_dedup_and_shadow_sends_nothing() {
    let fx: Value = serde_json::from_str(include_str!("fixtures/telegram.json")).unwrap();
    let srv = serve(Arc::new(|_g: &Got| Act::Send { status: 200, ctype: "application/json".into(), body: r#"{"ok":true}"#.into(), headers: vec![] }));
    let cases = fx["cases"].as_array().unwrap();
    let dir = std::env::temp_dir().join(format!("rtorder_tg_{}", std::process::id()));
    // 실전송(가짜 서버)
    let mut o = ops(&srv.base, false, tg(&srv.base));
    let mut w = Watcher::new(&dir, "ETHUSDC");
    let mut want: Vec<Value> = vec![];
    for c in cases {
        let n = w.logs.len();
        w.tick_alert(&mut o, c["key"].as_str().unwrap(), c["why"].as_str().unwrap());
        assert_eq!(json!(w.logs[n..]), c["printed"], "{}", c["why"]);
        want.extend(c["sent"].as_array().unwrap().iter().cloned());
    }
    assert_eq!(want.len(), 5, "같은 칸·같은 예외 종류는 한 번");
    for _ in 0..100 {
        if srv.log.lock().unwrap().len() >= want.len() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    }
    let got = srv.log.lock().unwrap().clone();
    assert_eq!(got.len(), want.len());
    let path = |u: &str| u.strip_prefix(telegram_api()).unwrap().to_string();
    let mut a: Vec<(String, String, Option<String>, String)> = got.iter().map(|g| (g.method.clone(), g.target.clone(), g.ctype.clone(), String::from_utf8(g.body.clone()).unwrap())).collect();
    let mut b: Vec<(String, String, Option<String>, String)> = want.iter().map(|s| (s["method"].as_str().unwrap().into(), path(s["url"].as_str().unwrap()), s["ctype"].as_str().map(String::from), s["body"].as_str().unwrap().into())).collect();
    a.sort(); // 뒤에서 따로 보내니 도착 순서는 다를 수 있다
    b.sort();
    assert_eq!(a, b);
    assert!(want.iter().all(|s| s["timeout"] == json!(8)));
    assert_eq!(rtorder::telegram::TIMEOUT_S, 8.0);
    // 그림자: 같은 호출 -- 요청 0 · 안 보낸 알림 5
    let mut o = ops(&srv.base, true, tg(&srv.base));
    let mut w = Watcher::new(&dir, "ETHUSDC");
    for c in cases {
        w.tick_alert(&mut o, c["key"].as_str().unwrap(), c["why"].as_str().unwrap());
    }
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert_eq!(srv.log.lock().unwrap().len(), want.len(), "그림자에서 보냈다");
    let would = o.would.lock().unwrap().clone();
    assert_eq!(would.len(), 5);
    assert!(would.iter().all(|x| x.starts_with("TELEGRAM 🔴 손절 감시: 무장 칸 ")), "{would:?}");
}

fn telegram_api() -> &'static str {
    rtorder::telegram::API
}

#[test]
fn not_configured_failures_and_token_hidden() {
    let fx: Value = serde_json::from_str(include_str!("fixtures/telegram.json")).unwrap();
    for (t, c) in [(None, Some("1")), (Some("x"), None), (Some(""), Some("1")), (Some("x"), Some(""))] {
        assert!(Telegram::from_env(&move |k: &str| match k { "TELEGRAM_BOT_TOKEN" => t.map(String::from), "TELEGRAM_CHAT_ID" => c.map(String::from), _ => None }).is_none());
    }
    let t = Telegram::from_env(&|k: &str| match k { "TELEGRAM_BOT_TOKEN" => Some(TOKEN.into()), "TELEGRAM_CHAT_ID" => Some("-100987".into()), _ => None }).unwrap();
    assert_eq!(t.base, rtorder::telegram::API, "기본 = 원본 주소(시험에서는 보내지 않는다)");
    assert_eq!(t.body("실패"), fx["fail_request"]["body"].as_str().unwrap());
    assert!(!format!("{t:?}").contains("test_token"), "Debug 에 토큰이 보이면 안 된다");
    // HTTP 500 · 닫힌 포트 = false(원본 OSError → False)
    let srv = serve(Arc::new(|_g: &Got| Act::Send { status: 500, ctype: "application/json".into(), body: "{}".into(), headers: vec![] }));
    assert!(!tg(&srv.base).send("x"));
    let dead = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", dead.local_addr().unwrap());
    drop(dead);
    assert!(!tg(&base).send("x"));
    let ok = serve(Arc::new(|_g: &Got| Act::Send { status: 200, ctype: "application/json".into(), body: "{}".into(), headers: vec![] }));
    assert!(tg(&ok.base).send("x"));
}
