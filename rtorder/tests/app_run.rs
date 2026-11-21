//! 실행 파일 뼈대(2026-11-21 D): 127.0.0.1 가짜 거래소(상태 있음 -- 걸린 주문·포지션)로
//! ① 기동 → 시작 복구(남은 대기 주문 정리 · 지정가 대기 복구 파일로 무장) → 감시 틱(포지션 없는 무장 정리) → 멈춤
//! ② 같은 폴더로 재시작 → 이미 정리됐으니 쓰기 0 · 무장 그대로
//! ③ 그림자 모드 → 같은 상황에서 쓰기 요청 0 · 안 보낸 쓰기 목록 · panic 없음(그림자 막이까지 안 간다)
//! ④ 진짜 실행 파일: 기동 → SIGTERM → 정상 종료 → 재시작 · 설정 오류(폴더 없음·live·파이썬 폴더)는 종료코드 2.
include!("common/server.rs");

use rtorder::app::{self, Config, Stop};
use rtorder::http::Binance;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64()
}

/// 상태 있는 가짜 거래소: 포지션 ETHUSDC LONG 0.5 · 걸린 주문(처음 = 남은 대기 dbwtL 70 · 닫힌 숏의 익절 dbtpS 71).
struct Ex {
    open: Vec<Value>,
    next: i64,
    log: Vec<String>,
}
fn exchange(routes: Value) -> (Srv, Arc<Mutex<Ex>>) {
    let ex = Arc::new(Mutex::new(Ex {
        open: vec![json!({"orderId": 70, "clientOrderId": "dbwtL1799990000000", "symbol": "ETHUSDC", "side": "BUY", "positionSide": "LONG", "type": "LIMIT",
                          "price": "2400", "origQty": "0.1", "executedQty": "0", "status": "NEW", "reduceOnly": false}),
                   json!({"orderId": 71, "clientOrderId": "dbtpS1799990000000", "symbol": "ETHUSDC", "side": "BUY", "positionSide": "SHORT", "type": "LIMIT",
                          "price": "2300", "origQty": "0.2", "executedQty": "0", "status": "NEW", "reduceOnly": false})],
        next: 100,
        log: vec![],
    }));
    let e2 = ex.clone();
    let srv = serve(Arc::new(move |g: &Got| {
        let mut e = e2.lock().unwrap();
        let (path, q) = g.target.split_once('?').unwrap_or((&g.target, ""));
        let qm: std::collections::HashMap<&str, &str> = q.split('&').filter_map(|kv| kv.split_once('=')).collect();
        e.log.push(format!("{} {path}", g.method));
        let body: Value = match (g.method.as_str(), path) {
            ("GET", "/fapi/v1/time") => json!({"serverTime": (now() * 1000.0) as i64}),
            ("GET", "/fapi/v2/account") => routes["/fapi/v2/account"].clone(),
            ("GET", "/fapi/v2/positionRisk") => json!([{"symbol": "ETHUSDC", "positionSide": "LONG", "positionAmt": "0.500", "entryPrice": "2400.5", "markPrice": "2450.25",
                                                       "liquidationPrice": "1900.1", "leverage": "10", "notional": "1225.125", "unRealizedProfit": "24.875", "updateTime": 1799990000123i64}]),
            ("GET", "/fapi/v1/userTrades") => json!([]),
            ("GET", "/fapi/v1/openOrders") => json!(e.open.iter().filter(|o| qm.get("symbol").is_none_or(|s| o["symbol"] == *s)).cloned().collect::<Vec<_>>()),
            ("GET", "/fapi/v1/openAlgoOrders") => json!([]),
            ("POST", "/fapi/v1/order") => {
                e.next += 1;
                let o = json!({"orderId": e.next, "clientOrderId": qm.get("newClientOrderId").copied().unwrap_or(""), "symbol": qm["symbol"], "side": qm["side"],
                               "positionSide": qm["positionSide"], "type": qm["type"], "price": qm.get("price").copied().unwrap_or("0"), "origQty": qm["quantity"],
                               "executedQty": "0", "status": "NEW", "reduceOnly": false});
                e.open.push(o.clone());
                o
            }
            ("DELETE", "/fapi/v1/order") => {
                let id: i64 = qm.get("orderId").and_then(|s| s.parse().ok()).unwrap_or(-1);
                match e.open.iter().position(|o| o["orderId"] == id) {
                    Some(i) => {
                        let mut o = e.open.remove(i);
                        o["status"] = json!("CANCELED");
                        o
                    }
                    None => return Act::Send { status: 400, ctype: "application/json".into(), body: r#"{"code":-2011,"msg":"Unknown order sent."}"#.into(), headers: vec![] },
                }
            }
            _ => return Act::Send { status: 599, ctype: "application/json".into(), body: format!(r#"{{"msg":"규칙 없음 {} {path}"}}"#, g.method), headers: vec![] },
        };
        Act::Send { status: 200, ctype: "application/json".into(), body: body.to_string(), headers: vec![] }
    }));
    local_only(&srv.base);
    (srv, ex)
}

fn seed(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let t = now();
    // 닫힌 숏의 선 무장(포지션 없음 → 감시가 정리) · 지정가 대기 중 재시작(그 사이 롱이 체결 → 복구 파일의 선으로 무장)
    let short = json!({"symbol": "ETHUSDC", "side": "SHORT", "sl_price": 2600.0, "sl_level": 2600.0, "tp_price": 2300.0, "tp_level": 2300.0, "backstop_price": null, "rule": false,
                       "sl_name": "선 손절 2600.00", "lines": true, "sl_up": null, "tp_resting": true, "armed_at": t - 600.0, "placed": {"placed": true}});
    std::fs::write(dir.join("manual_bracket_state.json"), json!({"ETHUSDC:SHORT": short}).to_string()).unwrap();
    let plan = json!({"symbol": "ETHUSDC", "side": "BUY", "positionSide": "LONG", "type": "LIMIT", "price": 2400.0, "quantity": 0.5, "wid": 1, "limit": {"usdt": 2400.0},
                      "bracket": {"available": true, "lines": true, "rule": false, "tp_price": 2600.0, "tp_level": 2600.0, "sl_price": 2380.0, "sl_level": 2380.0,
                                  "sl_name": "선 손절 2380.00", "sl_up": null, "backstop_price": null}});
    std::fs::write(dir.join("manual_limit_wait_ETHUSDC.json"), json!({"saved_at": t - 60.0, "plan": plan}).to_string()).unwrap();
}

fn cfg(dir: &Path, base: &str) -> Config {
    let d = dir.to_string_lossy().to_string();
    let get = move |k: &str| match k {
        "RTORDER_DIR" => Some(d.clone()),
        "RTORDER_BASE" => Some(base.to_string()),
        "RTORDER_PERIOD" => Some("0.05".into()),
        "RTORDER_MAX_TICKS" => Some("1".into()),
        _ => None,
    };
    app::config_from(&get).unwrap()
}

fn mk(base: String, ban: PathBuf, shadow: bool) -> impl Fn() -> Binance {
    move || {
        let mut b = Binance::new(&base, "test_key", "test_secret", std::time::Duration::from_secs(3), ban.clone());
        if shadow {
            b.shadow_on();
        }
        b
    }
}

fn file(dir: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("manual_bracket_state.json")).unwrap()).unwrap()
}

fn routes() -> Value {
    let fx: Value = serde_json::from_str(include_str!("fixtures/desk_fixture.json")).unwrap();
    fx["account"].as_array().unwrap().iter().find(|c| c["name"] == "normal_usdc").unwrap()["in"]["routes"].clone()
}

#[tokio::test]
async fn start_recovers_ticks_stops_and_restart_is_idempotent_then_shadow_sends_nothing() {
    std::env::set_var("DASHBOARD_MANUAL_EXEC_ENABLED", "1"); // 원본 실행 스위치(감시는 꺼져 있으면 아무것도 안 한다)
    let (srv, ex) = exchange(routes());
    let dir = std::env::temp_dir().join(format!("rtorder_app_{}", std::process::id()));
    seed(&dir);
    let ban = dir.join("ban");
    // ① 기동 → 복구 → 틱 → 멈춤(실전송 = 루프백 가짜 거래소)
    let s = app::run(&cfg(&dir, &srv.base), &mk(srv.base.clone(), ban.clone(), false), Stop::default()).await;
    let log: Vec<String> = std::mem::take(&mut ex.lock().unwrap().log);
    let writes: Vec<&String> = log.iter().filter(|l| !l.starts_with("GET")).collect();
    println!("① 쓰기 {writes:?}\n로그 {:?}", s.logs);
    assert!(!s.shadow && s.ticks == 1 && s.would.is_empty());
    let open: Vec<i64> = ex.lock().unwrap().open.iter().map(|o| o["orderId"].as_i64().unwrap()).collect();
    assert!(!open.contains(&70), "남은 대기 주문(dbwt)을 기동 때 지운다");
    assert!(!open.contains(&71), "포지션 없는 숏의 익절을 감시가 지운다");
    let tp: Vec<Value> = ex.lock().unwrap().open.iter().filter(|o| o["clientOrderId"].as_str().unwrap().starts_with("dbtpL")).cloned().collect();
    assert_eq!(tp.len(), 1, "복구 파일의 선으로 롱 익절을 건다");
    let f = file(&dir);
    assert!(f.get("ETHUSDC:SHORT").is_none() && f["ETHUSDC:LONG"]["lines"] == json!(true) && f["ETHUSDC:LONG"]["tp_resting"] == json!(true), "{f}");
    assert!(!dir.join("manual_limit_wait_ETHUSDC.json").exists(), "복구 끝 = 복구 파일 지움");
    assert!(s.logs.iter().any(|l| l.starts_with("wait cleanup:")), "{:?}", s.logs);
    // ② 재시작(같은 폴더·같은 거래소) -- 더 할 일 없음: 쓰기 0 · 무장 그대로
    let armed_at = f["ETHUSDC:LONG"]["armed_at"].clone();
    let s = app::run(&cfg(&dir, &srv.base), &mk(srv.base.clone(), ban.clone(), false), Stop::default()).await;
    let log: Vec<String> = std::mem::take(&mut ex.lock().unwrap().log);
    assert!(log.iter().all(|l| l.starts_with("GET")), "재시작에서 쓰기가 나갔다 {log:?}");
    assert_eq!(file(&dir)["ETHUSDC:LONG"]["armed_at"], armed_at, "무장을 다시 쓰지 않는다");
    assert_eq!(s.ticks, 1);
    // ③ 그림자: 처음 상황 그대로 · 쓰기 요청 0 · 안 보낸 쓰기가 목록에
    let (srv2, ex2) = exchange(routes());
    seed(&dir);
    let s = app::run(&cfg(&dir, &srv2.base), &mk(srv2.base.clone(), ban.clone(), true), Stop::default()).await;
    let log = ex2.lock().unwrap().log.clone();
    assert!(log.iter().all(|l| l.starts_with("GET")), "그림자에서 쓰기가 나갔다 {log:?}");
    println!("③ 안 보낸 쓰기 {:?}", s.would);
    assert!(s.shadow);
    for w in ["CANCEL_WAIT_ORPHANS", "BRACKET ETHUSDC LONG", "CLEAR ETHUSDC SHORT"] {
        assert!(s.would.iter().any(|x| x.starts_with(w)), "그림자 목록에 {w} 없음: {:?}", s.would);
    }
    assert_eq!(ex2.lock().unwrap().open.len(), 2, "거래소 주문 그대로");
    assert!(dir.join("manual_limit_wait_ETHUSDC.json").exists(), "못 걸었으니 복구 파일을 남긴다(다음 기동에 다시)");
    let _ = std::fs::remove_dir_all(&dir);
}

/// 정해진 시간 안에 끝나면 (종료코드, 표준출력) · 안 끝나면 죽이고 None(멈추지 않는 실행 파일이 시험을 묶지 않게).
fn finish(mut child: std::process::Child, secs: f64) -> Option<(Option<i32>, String)> {
    let t0 = std::time::Instant::now();
    loop {
        if child.try_wait().unwrap().is_some() {
            let out = child.wait_with_output().unwrap();
            return Some((out.status.code(), String::from_utf8_lossy(&out.stdout).into_owned()));
        }
        if t0.elapsed().as_secs_f64() > secs {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

#[test]
fn binary_starts_stops_on_sigterm_restarts_and_refuses_bad_config() {
    let bin = env!("CARGO_BIN_EXE_rtorder");
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let (srv, ex) = rt.block_on(async { exchange(routes()) });
    let dir = std::env::temp_dir().join(format!("rtorder_bin_{}", std::process::id()));
    seed(&dir);
    let run = |extra: &[(&str, &str)]| {
        let mut c = std::process::Command::new(bin);
        c.env_clear().env("PATH", std::env::var("PATH").unwrap_or_default()).env("RTORDER_DIR", &dir).env("RTORDER_BASE", &srv.base).env("RTORDER_PERIOD", "0.05");
        for (k, v) in extra {
            c.env(k, v);
        }
        c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped());
        c
    };
    for round in ["기동", "재시작"] {
        let child = run(&[]).spawn().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(800));
        let k = std::process::Command::new("kill").args(["-TERM", &child.id().to_string()]).status().unwrap();
        assert!(k.success());
        let (code, so) = finish(child, 10.0).unwrap_or_else(|| panic!("{round}: SIGTERM 뒤 10초 안에 안 멈췄다"));
        assert_eq!(code, Some(0), "{round}: {so}");
        assert!(so.contains("rtorder 시작 · 그림자") && so.contains("rtorder 종료"), "{round}: {so}");
        println!("{round}: {}", so.lines().last().unwrap_or(""));
    }
    let log = ex.lock().unwrap().log.clone();
    assert!(log.iter().all(|l| l.starts_with("GET")), "실행 파일(기본 빌드 = 그림자·키 없음)이 쓰기를 보냈다 {log:?}");
    // 설정 오류 = 시작 안 함(종료코드 2)
    let code = |c: &mut std::process::Command| finish(c.spawn().unwrap(), 10.0).and_then(|x| x.0);
    assert_eq!(code(run(&[("RTORDER_MODE", "live"), ("RTORDER_LIVE_CONFIRM", rtorder::shadow::LIVE_CONFIRM)]).env_remove("RTORDER_X")), Some(2), "기본 빌드에서 live");
    assert_eq!(code(run(&[("RTORDER_MODE", "turbo")]).env_remove("RTORDER_X")), Some(2));
    assert_eq!(code(std::process::Command::new(bin).env_clear()), Some(2), "RTORDER_DIR 없음");
    std::fs::write(dir.join("order_requests.jsonl"), "").unwrap();
    assert_eq!(code(&mut run(&[])), Some(2), "파이썬 서버 폴더");
    let _ = std::fs::remove_dir_all(&dir);
}
