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
