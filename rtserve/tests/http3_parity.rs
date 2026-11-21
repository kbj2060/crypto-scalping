//! 셋째 조각 대조: tools/gen_http3_fixture.py 가 진짜 aiohttp 객체로 뜬 바이트 == rtserve(127.0.0.1 · 실제 HTTP).
//! 원본 계산 함수(compute_macro_event_alert · compute_liquidation_5m_signal)를 합성 트리(파이썬 duckdb 로 만든 tail_risk 파일 포함)·고정 시계로 돌렸다.

mod common;

use common::{b64, norm, request, start};
use serde_json::Value;

#[test]
fn http3_parity() {
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/http3.json")).unwrap()).unwrap();
    let (now, mtime, assets) = (fx["now"].as_f64().unwrap(), fx["mtime_ns"].as_u64().unwrap(), fx["dashboard_assets"].as_str().unwrap());
    let (mut n, mut bad) = (0, Vec::new());
    for sc in fx["scenarios"].as_array().unwrap() {
        let name = sc["name"].as_str().unwrap();
        let (addr, root) = start(sc["files"].as_object().unwrap(), mtime, now, assets, &format!("h3_{name}"));
        for c in sc["cases"].as_array().unwrap() {
            n += 1;
            let p = c["path"].as_str().unwrap();
            let got = request(addr, "GET", p, &[]);
            let want_h: Vec<(String, String)> = c["headers"].as_array().unwrap().iter().map(|h| (h[0].as_str().unwrap().into(), h[1].as_str().unwrap().into())).collect();
            let want = b64(c["body_b64"].as_str().unwrap());
            if got.status != c["status"].as_u64().unwrap() as u16 || norm(&got.headers) != norm(&want_h) || got.body != want {
                bad.push(format!("[{name}] {p}: {} {:?}\n  rust {}\n  py   {}\n  rh {:?}\n  ph {:?}", got.status, c["status"], String::from_utf8_lossy(&got.body),
                                 String::from_utf8_lossy(&want), norm(&got.headers), norm(&want_h)));
            }
        }
        let _ = std::fs::remove_dir_all(root);
    }
    eprintln!("http3_parity: 요청 {n} · 불일치 {} · 원본 duckdb {}", bad.len(), fx["duckdb"]);
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// 표가 없는 파일(만드는 중) -- 원본처럼 0.4·0.8·1.6초 다시 읽고 warmed_up=false · error 는 db_read_error: 로 시작(문구는 duckdb-rs)
#[test]
fn liq5m_db_read_error() {
    let dir = std::env::temp_dir().join(format!("rtserve_liqerr_{}", std::process::id()));
    std::fs::create_dir_all(dir.join("data/live")).unwrap();
    duckdb::Connection::open(dir.join("data/live/tail_risk.duckdb")).unwrap().execute_batch("CREATE TABLE other (x INT)").unwrap();
    let cfg = rtserve::Config::new(dir.clone(), vec!["eth".into()], std::sync::Arc::new(|| 1.7e9));
    let t = std::time::Instant::now();
    let v = rtserve::api3::liq5m_payload(&cfg, "eth").unwrap();
    assert!(t.elapsed().as_secs_f64() >= 2.7, "다시 읽기 0.4+0.8+1.6초");
    assert_eq!(v["warmed_up"], false);
    assert!(v["error"].as_str().unwrap().starts_with("db_read_error: "), "{}", v["error"]);
    let _ = std::fs::remove_dir_all(dir);
}
