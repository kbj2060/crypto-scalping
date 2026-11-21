//! 엔드포인트별 한 요청 처리 시간(rtserve 쪽만 · 파이썬은 개발기에서 서버를 안 띄우므로 재지 않는다).
//! 입력 = 시험 정답지(http·http2·http3.json)의 합성 파일 트리 그대로. 잰 구간 = `rtserve::handle`(라우터 → 핸들러 → 미들웨어 → 압축·길이) 한 번 --
//! 소켓·hyper 는 빼고. 두 가지: «계산»(캐시 없이 매번 새로 = 새 Config) · «캐시 맞음»(같은 Config 로 두 번째부터 · swr 신선).
//!
//!     cargo run --release --example bench -- [반복 수(기본 300)]

use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

use serde_json::Value;

fn b64(s: &str) -> Vec<u8> {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let (mut out, mut acc, mut bits) = (Vec::new(), 0u32, 0);
    for c in s.bytes().filter(|c| *c != b'=') {
        acc = acc << 6 | T.iter().position(|t| *t == c).unwrap() as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    out
}

fn mkreq(path: &str, gzip: bool) -> rtserve::Req {
    let (p, q) = path.split_once('?').unwrap_or((path, ""));
    rtserve::Req {
        method: "GET".into(),
        path: p.into(),
        query: q.split('&').filter(|x| !x.is_empty()).map(|kv| kv.split_once('=').map_or((kv.into(), String::new()), |(a, b)| (a.into(), b.into()))).collect(),
        raw_query: q.to_string(),
        headers: if gzip { vec![("accept-encoding".into(), "gzip, deflate, br, zstd".into())] } else { vec![] },
        body: Vec::new(),
        peer: "127.0.0.1".into(),
    }
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() as f64 - 1.0) * q).round() as usize]
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|x| x.parse().ok()).unwrap_or(300);
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    // (정답지, 시나리오, 주소들) -- 엔드포인트마다 대표 요청 하나
    let plan: [(&str, &str, &[&str]); 5] = [
        ("http.json", "static", &["/dashboard/live/", "/dashboard/live/app.js"]),
        ("http.json", "api", &["/api/acct-notes", "/api/risk-radar", "/api/sr-lean", "/api/regime-band", "/api/liq-tab", "/api/iceberg", "/api/ob5m",
            "/api/treasury-futures", "/api/xyz100", "/api/macro-today", "/api/macro-calendar"]),
        ("http.json", "api_big", &["/api/sr-lean", "/api/regime-band"]),
        ("http2.json", "api2", &["/api/state", "/api/breakout-detector", "/api/gex", "/api/chip-score", "/api/notify-center", "/api/news", "/api/macro-spikes",
            "/api/bubble-tape", "/api/pred-record"]),
        ("http3.json", "calc", &["/api/session-alerts", "/api/liquidation-5m-signal"]),
    ];
    println!("| 주소 | 시나리오 | 계산 중앙 | 계산 상위1% | 캐시 맞음 중앙 | 캐시 맞음 상위1% | 건수 | 응답 바이트(압축 전→후) |");
    println!("|---|---|---|---|---|---|---|---|");
    for (file, scen, paths) in plan {
        let fx: Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/{file}")).unwrap()).unwrap();
        let sc = fx["scenarios"].as_array().unwrap().iter().find(|s| s["name"] == scen).unwrap();
        let now = fx["now"].as_f64().unwrap();
        let root = std::env::temp_dir().join(format!("rtserve_bench_{scen}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("dashboard/live")).unwrap();
        let mt = UNIX_EPOCH + Duration::from_nanos(fx["mtime_ns"].as_u64().unwrap());
        for (rel, data) in sc["files"].as_object().unwrap() {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, b64(data.as_str().unwrap())).unwrap();
            std::fs::File::options().write(true).open(&p).unwrap().set_times(std::fs::FileTimes::new().set_modified(mt)).unwrap();
        }
        let assets = rtserve::parse_assets(fx["dashboard_assets"].as_str());
        let mk = || {
            let mut c = rtserve::Config::new(root.clone(), assets.clone(), Arc::new(move || now));
            c.log = Arc::new(|_| {});
            c
        };
        for path in paths {
            let req = mkreq(path, true);
            let (mut cold, mut warm) = (Vec::with_capacity(n), Vec::with_capacity(n));
            let mut sizes = (0, 0);
            let cfg = mk();
            for k in 0..n {
                let c = mk();
                let t = Instant::now();
                let r = rtserve::handle(&c, &req);
                cold.push(t.elapsed().as_secs_f64() * 1e6);
                if k == 0 {
                    let raw = rtserve::handle(&mk(), &mkreq(path, false));
                    sizes = (raw.body.as_ref().map_or(0, Vec::len), r.body.as_ref().map_or(0, Vec::len));
                    let _ = rtserve::handle(&cfg, &req); // 캐시 채우기
                }
                let t = Instant::now();
                let _ = rtserve::handle(&cfg, &req);
                warm.push(t.elapsed().as_secs_f64() * 1e6);
            }
            let us = |x: f64| if x >= 1000.0 { format!("{:.2}ms", x / 1000.0) } else { format!("{x:.0}µs") };
            let (c50, c99, w50, w99) = (pct(&mut cold, 0.5), pct(&mut cold, 0.99), pct(&mut warm, 0.5), pct(&mut warm, 0.99));
            println!("| `{path}` | {scen} | {} | {} | {} | {} | {n} | {}→{} |", us(c50), us(c99), us(w50), us(w99), sizes.0, sizes.1);
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
