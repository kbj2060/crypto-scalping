//! `rtorder replay <order_shadow.jsonl>`(2026-11-21 · 서버 실기록 재생): 진짜 실행 파일로
//! ① 합성 기록(tools/gen_py_shadow_fixture.py -- 원본 파이썬을 가짜 거래소로 돌린 진입·청산·감시 발동·무장 파일만 바뀐 틱·잘린 chain·접수만 chain) = 어긋남 0
//! ② 출력에 값이 없다(수량·가격을 바꾼 기록 → «요청 #i … 칸 quantity» 만 · 값은 --show-values 일 때만)
//! ③ 돌린 파일(.1)·쓰는 중이던 반쪽 줄을 견딘다 ④ 감시 틱 안 조립 구간 왕복(/time·commissionRate)은 따로 센다
//! ⑤ 바깥으로 아무것도 안 보낸다: 망 없는 이름공간(unshare -rn)에서도 같은 결과 · 127.0.0.1 밖 주소는 보내기 전에 panic ⑥ 입력 파일은 그대로.
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/py_shadow_cases.jsonl")).unwrap()
}
fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rtorder_replay_cmd_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}
fn run(file: &Path, extra: &[&str]) -> (i32, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_rtorder")).arg("replay").arg(file).args(["--symbol", "ETHUSDT"]).args(extra).env_clear().env("PATH", "/usr/bin:/bin").output().unwrap();
    (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stdout).into_owned())
}
fn edit(text: &str, f: impl Fn(&mut Vec<Value>)) -> String {
    let mut rows: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    f(&mut rows);
    rows.iter().map(|r| r.to_string() + "\n").collect()
}

#[test]
fn replay_command_zero_diff_value_free_output_rotation_assembly_and_no_network() {
    let text = fixture();
    let d = tmp("main");
    let f = d.join("order_shadow.jsonl");
    std::fs::write(&f, &text).unwrap();
    let before = std::fs::read(&f).unwrap();
    // ① 합성 기록 = 어긋남 0
    let (code, out) = run(&f, &[]);
    println!("{out}");
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("견준 작업 4 · 요청 23 · 어긋난 작업 0 · 잘림 1"), "{out}");
    assert!(out.contains("접수만 chain 2") && out.contains("| tick | 요청 10 | 같음") && out.contains("| tick | 요청 0 | 같음"), "{out}");
    assert_eq!(std::fs::read(&f).unwrap(), before, "입력 파일은 읽기만");
    // ② 값 없는 출력: 진입 주문 수량·끝 무장 손절선을 바꾼 기록
    let bad = d.join("bad.jsonl");
    std::fs::write(&bad, edit(&text, |rows| {
        let w = rows.iter_mut().find(|r| r["rec"] == "wire" && r["method"] == "POST" && r["target"].as_str().unwrap().contains("side=BUY")).unwrap();
        w["target"] = json!(w["target"].as_str().unwrap().replace("quantity=0.8", "quantity=0.9"));
        let e = rows.iter_mut().find(|r| r["rec"] == "end" && r["chain"].as_str().unwrap().starts_with("entry") && !r["chain"].as_str().unwrap().ends_with("cut")).unwrap();
        e["file"]["ETHUSDT:LONG"]["sl_level"] = json!(2390.0);
    })).unwrap();
    let (code, out) = run(&bad, &[]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("칸 quantity") && out.contains("무장 파일 ETHUSDT:LONG.sl_level"), "어긋난 위치(칸 이름) {out}");
    assert!(out.contains("첫 어긋남: entry-") && out.contains("요청 #"), "{out}");
    let quiet = out.replace(d.to_str().unwrap(), ""); // 임시 폴더 이름의 pid 숫자(예: …2239004 안의 2390)는 값이 아니다
    for v in ["0.9", "2390", "0.8", "2400", "test_key"] {
        assert!(!quiet.contains(v), "값이 출력에 나왔다 {v}: {out}");
    }
    let (_, shown) = run(&bad, &["--show-values"]);
    assert!(shown.contains("quantity=0.9") && shown.contains("2390"), "--show-values 일 때만 값 {shown}");
    // ③ 돌린 파일(.1) + 쓰는 중이던 반쪽 줄
    let r = tmp("rot");
    let rf = r.join("order_shadow.jsonl");
    let lines: Vec<&str> = text.lines().collect();
    let half = lines.len() / 2;
    std::fs::write(r.join("order_shadow.jsonl.1"), lines[..half].join("\n") + "\n").unwrap();
    std::fs::write(&rf, lines[half..].join("\n") + "\n" + &lines[0][..lines[0].len() / 2]).unwrap();
    let (code, out) = run(&rf, &[]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("(+ .1)") && out.contains("깨진 줄 1") && out.contains("어긋난 작업 0 · 잘림 1"), "{out}");
    // ④ 감시 틱 안 조립 구간 왕복: 발동 틱의 청산 계획 입력 바로 앞에 파이썬이 /time·commissionRate 를 보냈다(실서버에서 생기는 꼴) -- 따로 세고 같음
    let asm = d.join("asm.jsonl");
    std::fs::write(&asm, edit(&text, |rows| {
        let i = rows.iter().position(|r| r["what"] == "exit_plan" && r["value"]["plan"].is_object()).unwrap();
        let (chain, t) = (rows[i]["chain"].clone(), rows[i]["t"].as_f64().unwrap());
        for (k, p) in [(2.0, "/fapi/v1/time"), (1.0, "/fapi/v1/commissionRate?symbol=ETHUSDT&recvWindow=5000&timestamp=1")] {
            rows.insert(i, json!({"rec": "wire", "chain": chain, "t": t - k * 0.0001, "method": "GET", "target": p, "status": 200, "body": "{}", "error": null}));
        }
    })).unwrap();
    let (code, out) = run(&asm, &[]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("조립 구간 왕복 2(따로 셈)"), "{out}");
    // ⑤ 망 없는 이름공간에서도 같은 결과(바깥 접속이 필요 없다)
    if Command::new("unshare").args(["-rn", "true"]).status().is_ok_and(|s| s.success()) {
        let o = Command::new("unshare").args(["-rn", "sh", "-c", "ip link set lo up 2>/dev/null; exec \"$0\" replay \"$1\" --symbol ETHUSDT"])
            .arg(env!("CARGO_BIN_EXE_rtorder")).arg(&f).output().unwrap();
        let s = String::from_utf8_lossy(&o.stdout);
        assert_eq!(o.status.code(), Some(0), "{s}");
        assert!(s.contains("어긋난 작업 0"), "{s}");
    } else {
        println!("unshare 없음 -- 망 없는 실행 확인은 건너뜀");
    }
    let _ = std::fs::remove_dir_all(&d);
    let _ = std::fs::remove_dir_all(&r);
}

#[test]
fn loopback_only_panics_before_any_outside_request() {
    // 재생이 켜는 막이: 127.0.0.1 밖 주소로 가는 요청은 보내기 전에 panic(이 시험 파일은 이 프로세스에서 바깥 요청을 만들지 않는다)
    rtorder::http::LOOPBACK_ONLY.store(true, std::sync::atomic::Ordering::SeqCst);
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let bx = rtorder::http::Binance::new("https://fapi.binance.com", "", "", std::time::Duration::from_secs(1), std::env::temp_dir().join("rtorder_lo_ban"));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.block_on(bx.clock_offset())));
    assert!(r.is_err(), "바깥 주소 요청이 막히지 않았다");
}

/// 2026-11-22 서버 실기록 어긋남 1(SOXL 진입): 체결 알림(chain "*" · wake_fills)이 원본 nap(3초)을 2.2초에 깨워 «주문 조회만 한 번 더»(3초 따라가기 판단 전 = 호가 조회 없음).
/// 합성 진입 chain 을 그 꼴로 바꾼다: 첫 조회(1.2초 · 깨움) = 아직 NEW · 둘째 조회(1.9초 · 깨움) = 체결. 원본 fill_maker 2546~2566 을 Read 로 확인한 순서(깨움 → 조회 → `monotonic-last_chk < POLL_SEC` 면 continue).
/// 깨움을 재생에 넣으면 어긋남 0 · 깨움 줄을 지우면(= 고치기 전 재생기) 둘째 조회 자리에 bookTicker = 서버에서 본 것과 같은 어긋남 · 요청 수는 원본·Rust 따로.
#[test]
fn replay_wakes_nap_early_like_python_fill_alert() {
    let d = tmp("wake");
    let chain = fixture().lines().map(|l| serde_json::from_str::<Value>(l).unwrap()).find(|r| r["rec"] == "start" && r["op"] == "entry").unwrap()["chain"].as_str().unwrap().to_string();
    let woke = edit(&fixture(), |rows| {
        rows.retain(|r| r["chain"] == chain.as_str());
        let post = rows.iter().position(|r| r["rec"] == "wire" && r["method"] == "POST" && r["target"].as_str().unwrap().contains("type=LIMIT")).unwrap();
        let t0 = rows[post]["t"].as_f64().unwrap();
        let mut early = rows[post + 1].clone(); // 체결 조회(FILLED)를 본떠 아직 NEW
        assert_eq!(early["method"], "GET");
        let body = early["body"].as_str().unwrap().replace("\"FILLED\"", "\"NEW\"");
        let mut b: Value = serde_json::from_str(&body).unwrap();
        b["executedQty"] = json!("0");
        early["body"] = json!(b.to_string());
        early["t"] = json!(t0 + 1.21);
        rows[post + 1]["t"] = json!(t0 + 1.91);
        rows.insert(post + 1, early);
        rows.insert(post + 1, json!({"rec": "input", "chain": "*", "t": t0 + 1.2, "mono": 0, "what": "wake"}));
        rows.insert(post + 3, json!({"rec": "input", "chain": "*", "t": t0 + 1.9, "mono": 0, "what": "wake"}));
    });
    let f = d.join("order_shadow.jsonl");
    std::fs::write(&f, &woke).unwrap();
    let (code, out) = run(&f, &[]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("깨움 줄 2") && out.contains("견준 작업 1 · 요청 11 · 어긋난 작업 0"), "{out}");
    let (code, out) = run(&f, &[]);
    assert_eq!(code, 0, "두 번 돌려도 같다: {out}");
    // 깨움 줄 없이 = 고치기 전 재생기가 본 것
    std::fs::write(&f, woke.lines().filter(|l| !l.contains("\"chain\":\"*\"")).map(|l| l.to_string() + "\n").collect::<String>()).unwrap();
    let (code, out) = run(&f, &[]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("경로 다름(원본 GET /fapi/v1/order · Rust GET /fapi/v1/ticker/bookTicker)"), "{out}");
    assert!(out.contains("| 요청 11(Rust "), "원본·Rust 요청 수를 따로: {out}");
    let _ = std::fs::remove_dir_all(&d);
}

/// 2026-11-22 서버 실기록 tick 5건 «요청 1(Rust 0) · Rust 가 덜 보냄 (GET /fapi/v1/time)»: 원본 감시 틱의 swr_cached 가 낡은 계좌를 내주며 띄운 뒤 계좌 갱신
/// (_schedule_refresh → create_task 가 틱의 그림자 버퍼를 복사)의 `_clock_offset` 왕복만 틱 기록에 남은 것 -- 판정과 무관. 틱 끝에 홀로 남은 /time 은 따로 센다.
/// 양성 대조: /time 이 아닌 요청이 하나 빠지면 그대로 어긋남 · 틱이 아닌 작업의 /time 은 안 뗀다.
#[test]
fn trailing_refresh_clock_in_tick_is_counted_apart_but_other_requests_still_differ() {
    let d = tmp("rclk");
    let rows: Vec<Value> = fixture().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let ticks: Vec<String> = rows.iter().filter(|r| r["rec"] == "start" && r["op"] == "tick").map(|r| r["chain"].as_str().unwrap().to_string()).collect();
    let (fire, quiet) = (ticks[0].clone(), ticks[1].clone());
    let time_wire = |chain: &str, t: f64| json!({"rec": "wire", "chain": chain, "t": t, "mono": 0, "method": "GET", "target": "/fapi/v1/time", "status": 200, "error": null, "body": format!("{{\"serverTime\": {}}}", (t * 1000.0) as i64)});
    // 무장 파일만 바뀌는 틱(요청 0) 끝에 /time 하나 = 서버 tick 5건 꼴
    let with_clock = edit(&fixture(), |rows| {
        rows.retain(|r| r["chain"] == quiet.as_str());
        let end = rows.iter().position(|r| r["rec"] == "end").unwrap();
        let t = rows[end]["t"].as_f64().unwrap() - 0.01;
        rows.insert(end, time_wire(&quiet, t));
    });
    let f = d.join("order_shadow.jsonl");
    std::fs::write(&f, &with_clock).unwrap();
    let (code, out) = run(&f, &[]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("| 요청 0 · 뒤 계좌 갱신 시계 왕복 1 | 같음") && out.contains("뒤 계좌 갱신 시계 왕복 1(따로 셈)"), "{out}");
    // 대조 ①: 발동 틱의 마지막 요청(openAlgoOrders)을 빼면 -- 끝에 /time 이 남아도 -- 어긋남
    let dropped = edit(&fixture(), |rows| {
        rows.retain(|r| r["chain"] == fire.as_str());
        let last = rows.iter().rposition(|r| r["rec"] == "wire").unwrap();
        assert!(rows[last]["target"].as_str().unwrap().starts_with("/fapi/v1/openAlgoOrders"));
        let t = rows[last]["t"].as_f64().unwrap();
        rows[last] = time_wire(&fire, t);
    });
    std::fs::write(&f, &dropped).unwrap();
    let (code, out) = run(&f, &[]);
    assert_eq!(code, 1, "/time 아닌 요청이 빠지면 어긋남: {out}");
    assert!(out.contains("Rust 가 더 보냄") || out.contains("경로 다름"), "{out}");
    // 대조 ②: 진입 작업 끝의 /time 은 안 뗀다(틱만)
    let entry = rows.iter().find(|r| r["rec"] == "start" && r["op"] == "entry").unwrap()["chain"].as_str().unwrap().to_string();
    let entry_clock = edit(&fixture(), |rows| {
        rows.retain(|r| r["chain"] == entry.as_str());
        let end = rows.iter().position(|r| r["rec"] == "end").unwrap();
        let t = rows[end]["t"].as_f64().unwrap() - 0.0001;
        rows.insert(end, time_wire(&entry, t));
    });
    std::fs::write(&f, &entry_clock).unwrap();
    let (code, out) = run(&f, &[]);
    assert_eq!(code, 1, "진입 끝 /time 은 어긋남 그대로: {out}");
    let _ = std::fs::remove_dir_all(&d);
}
