//! 롱숏비 워커(oi_lsratio) 파이썬 일치 시험. 정답지 = tools/gen_oi_fixture.py 가 원본 `_db_init`·`_poll_once_blocking`·`_db_upsert_rows` 를 가짜 응답·가짜 시계로
//! 돌려 파이썬 duckdb 로 덤프한 것(표·열 형·물리 순서·행 -- 실수는 repr 글자 = 비트). 같은 입력을 Rust 에 넣고 Rust DuckDB 로 덤프해 비교한다.
use serde_json::Value;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::Command;

fn tmp(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rtduck_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn oi_lsratio_db_matches_python_worker() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/oi_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/oi_golden.json")).unwrap();
    let dir = tmp("oi");
    let db = dir.join("oi.duckdb");
    let (reqs, log) = rtduck::oi::replay(&inp, &db);
    let want: Vec<&str> = gold["requests"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect();
    assert_eq!(reqs, want, "요청 URL 열");
    assert_eq!(Value::from(log), gold["log"], "폴마다 결과(행 수 · 닿은 원천 · 예외 이름)");
    let got = rtduck::duckdump::dump(&db).unwrap();
    for (t, w) in gold["tables"].as_object().unwrap() {
        assert_eq!(got[t]["cols"], w["cols"], "{t} 열·형");
        assert_eq!(got[t]["phys_ts"], w["phys_ts"], "{t} 물리 순서");
        let (g, w) = (got[t]["rows"].as_array().unwrap(), w["rows"].as_array().unwrap());
        assert_eq!(g.len(), w.len(), "{t} 행 수");
        for (k, (a, b)) in g.iter().zip(w).enumerate() {
            assert_eq!(a, b, "{t} 행 {k}");
        }
    }
    assert_eq!(got.as_object().unwrap().len(), gold["tables"].as_object().unwrap().len(), "표 수");
    let cells: usize = gold["tables"].as_object().unwrap().values().map(|v| v["rows"].as_array().unwrap().iter().map(|r| r.as_array().unwrap().len()).sum::<usize>()).sum();
    eprintln!("oi_lsratio 일치: 표 {} · 칸 {cells} · 요청 {} (정답지 파이썬 duckdb {})", got.as_object().unwrap().len(), reqs.len(), gold["duckdb_version"]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// 실행 파일 한 바퀴(RTDUCK_ONCE=1): 127.0.0.1 가짜 바이낸스가 정답지 첫 폴 응답을 주면 DB 가 정답지 첫 폴 뒤와 같다(collected_at = 벽시계라 뺀다)
/// · 요청 머리글(User-Agent)·경로·쿼리 순서 · 밴 파일이 있으면 아무것도 안 보낸다.
#[test]
fn oi_binary_one_round_against_fake_binance() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/oi_input.json")).unwrap();
    let first = inp["polls"][0]["resp"].clone();
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    assert!(base.starts_with("http://127.0.0.1:"), "시험은 127.0.0.1 만");
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let s2 = seen.clone();
    std::thread::spawn(move || {
        for c in l.incoming() {
            let mut c = c.unwrap();
            let mut buf = [0u8; 4096];
            let n = c.read(&mut buf).unwrap();
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let line = req.lines().next().unwrap().to_string();
            let ua = req.lines().find(|x| x.to_lowercase().starts_with("user-agent:")).unwrap_or("").to_string();
            s2.lock().unwrap().push(format!("{line} | {ua}"));
            let target = line.split(' ').nth(1).unwrap();
            let path = target.split('?').next().unwrap();
            let sym = target.split("symbol=").nth(1).unwrap().split('&').next().unwrap().to_lowercase();
            let r = &first[sym.as_str()][path];
            let (code, body) = if r.get("err").is_some() { (500, "x".to_string()) } else if let Some(s) = r["status"].as_u64() { (s, "x".into()) } else { (200, r["body"].as_str().unwrap().to_string()) };
            let _ = write!(c, "HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
        }
    });
    let dir = tmp("oi_bin");
    let db = dir.join("oi.duckdb");
    let run = |ban: &str| {
        let st = Command::new(env!("CARGO_BIN_EXE_rtduck"))
            .env("RTDUCK_KIND", "oi_lsratio").env("BOT_SYMBOLS", "ETHUSDT,BTCUSDT").env("QUANT_OI_LSRATIO_DB_PATH", &db).env("RTDUCK_REST", "1")
            .env("RTDUCK_ONCE", "1").env("RTDUCK_OI_BASE", &base).env("RTCOLLECT_BAN_FILE", dir.join(ban)).env("OI_FETCH_LIMIT", "12")
            .output().unwrap();
        assert!(st.status.success(), "{}", String::from_utf8_lossy(&st.stderr));
    };
    std::fs::write(dir.join("ban_on"), format!("{:.3}\n", rtcollect::rest::wall_s() + 600.0)).unwrap();
    run("ban_on");
    assert!(seen.lock().unwrap().is_empty(), "밴 중이면 보내지 않는다");
    run("ban_none");
    let reqs = seen.lock().unwrap().clone();
    assert_eq!(reqs.len(), 10, "종목 둘 × 원천 다섯");
    assert!(reqs.iter().all(|r| r.ends_with("| User-Agent: crypto-scalping-oi-lsratio-collector/1.0") || r.ends_with("| user-agent: crypto-scalping-oi-lsratio-collector/1.0")), "{reqs:#?}");
    assert_eq!(reqs[0].split(' ').nth(1).unwrap(), "/futures/data/openInterestHist?symbol=ETHUSDT&period=5m&limit=12");
    // 정답지 입력의 첫 폴만 먹인 Rust replay 와 같다(collected_at 칸만 다르다)
    let one = serde_json::json!({"symbols": inp["symbols"], "fetch_limit": inp["fetch_limit"], "polls": [inp["polls"][0]]});
    let rd = dir.join("replay.duckdb");
    rtduck::oi::replay(&one, &rd);
    let (a, b) = (rtduck::duckdump::dump(&db).unwrap(), rtduck::duckdump::dump(&rd).unwrap());
    for (t, w) in b.as_object().unwrap() {
        let ci = w["cols"].as_array().unwrap().iter().position(|c| c[0] == "collected_at").unwrap();
        let strip = |v: &Value| v["rows"].as_array().unwrap().iter().map(|r| { let mut r = r.as_array().unwrap().clone(); r.remove(ci); r }).collect::<Vec<_>>();
        assert_eq!(strip(&a[t]), strip(w), "{t}");
        assert!(!strip(w).is_empty());
    }
    let _ = std::fs::remove_dir_all(&dir);
}
