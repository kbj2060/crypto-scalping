//! /api/macro-trend · /api/opt-smile = 워커 파일 바이트 그대로 · 없음·낡음 = 원본 실패 갈래 모양(filefeed.rs)
//!     cargo test --release -j 2 --offline --test filefeed
mod common;

use common::{request, set_mtime, start};

#[test]
fn file_bytes_and_failure_shapes() {
    let now = 1_800_000_000.0;
    let (addr, root) = start(&serde_json::Map::new(), 0, now, "eth,btc,sol,xrp", "filefeed");
    let live = root.join("data/live");
    std::fs::create_dir_all(&live).unwrap();
    let get = |p: &str| {
        let g = request(addr, "GET", p, &[]);
        assert_eq!(g.status, 200, "{p}");
        String::from_utf8(g.body).unwrap()
    };
    let put = |name: &str, body: &str, age: f64| {
        let p = live.join(name);
        std::fs::write(&p, body).unwrap();
        set_mtime(&p, ((now - age) * 1e9) as u64);
    };
    let macro_fail = r#"{"ok": false, "error": "FileNotFoundError: data/live/macro_trend.json", "rows": [], "summary": {}, "errors": {}}"#;
    assert_eq!(get("/api/macro-trend"), macro_fail);
    let m = "{\"ok\": true, \"asof\": 1, \"rows\": [{\"name\": \"\\uad6d\\ucc44\", \"v\": NaN}], \"summary\": {}}";
    put("macro_trend.json", m, 25_000.0);
    assert_eq!(get("/api/macro-trend"), m);
    let g = request(addr, "GET", "/api/macro-trend", &[]);
    assert!(g.headers.iter().any(|(k, v)| k == "content-type" && v.starts_with("application/json")));
    assert!(g.headers.iter().any(|(k, v)| k == "cache-control" && v == "no-cache"));
    put("macro_trend.json", m, 25_300.0);
    assert!(get("/api/macro-trend").starts_with(r#"{"ok": false, "error": "stale: data/live/macro_trend.json age 25300s""#));

    let s = r#"{"ok": true, "exp_ms": 1796025600000, "S": 3000.0, "coef": [1.0, 0.1, 0.5], "xlim": 0.08, "n": 7, "fetched_ts": 1799999990.5}"#;
    put("opt_smile_eth.json", s, 600.0);
    assert_eq!(get("/api/opt-smile"), s); // asset 기본 eth
    assert_eq!(get("/api/opt-smile?asset=ETH"), s);
    put("opt_smile_xrp.json", s, 700.0);
    let unavail = r#"{"ok": false, "reason": "deribit_unavailable"}"#;
    assert_eq!(get("/api/opt-smile?asset=xrp"), unavail); // 660초 넘음
    assert_eq!(get("/api/opt-smile?asset=btc"), unavail); // 파일 없음
    assert_eq!(get("/api/opt-smile?asset=sol"), r#"{"ok": false, "reason": "eth_btc_xrp_only"}"#);
    assert_eq!(get("/api/opt-smile?asset="), r#"{"ok": false, "reason": "eth_btc_xrp_only"}"#);
}
