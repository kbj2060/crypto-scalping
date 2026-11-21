//! 다섯째 조각: 일봉·4시간 풋프린트 4 주소(api3.rs 라우팅 → fpd.rs)를 rtserve HTTP 로 -- 원본 엔드포인트(server.py:3861~3912)의 손잡이 대조.
//! 정답 = tools/gen_fpd_fixture.py 가 원본 DailyFootprint 로 뜬 첫 걸음 본문(tests/fixtures/gen/fpd · fpd_parity.rs 와 같은 정답지) +
//! 400 갈래(asset · bad_date · bad_key · from_to · range) 글자. 원천 = 같은 합성 소급본·lake·hot(처음 사본)을 저장소 꼴(data/lake · data/footprint_daily · data/hot)로 편다.

mod common;

use common::{request, start_with};
use serde_json::Value;
use std::path::PathBuf;

fn gen() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen/fpd")
}

#[test]
fn fpd_routes_over_http() {
    let mut n = 0;
    let mut bad = Vec::new();
    for asset in ["eth", "xrp"] {
        let fx: Value = serde_json::from_str(&std::fs::read_to_string(gen().join(asset).join(format!("fpd_{asset}.json"))).expect("gen_fpd_fixture.py 먼저")).unwrap();
        let st = &fx["steps"][0];
        let now = st["now"].as_f64().unwrap();
        let (addr, root) = start_with(&serde_json::Map::new(), 0, now, "eth,sol,xrp", &format!("fpd_{asset}"), |_| {});
        std::fs::create_dir_all(root.join("data/hot")).unwrap();
        std::os::unix::fs::symlink(gen().join("lake"), root.join("data/lake")).unwrap();
        std::os::unix::fs::symlink(gen().join("hist"), root.join("data/footprint_daily")).unwrap();
        std::fs::copy(gen().join(asset).join("hot_tape_start.sqlite"), root.join("data/hot/binance_tape.sqlite")).unwrap();
        std::fs::copy(gen().join(asset).join("hot_ctx_start.sqlite"), root.join("data/hot/binance_ctx.sqlite")).unwrap();
        let cell = rtserve::fpd::Spec::of(asset).unwrap().cell;
        let mut want_full = None;
        for c in st["calls"].as_array().unwrap() {
            let a = c["args"].as_array().unwrap();
            let row = |i: usize| format!("{}", a[i].as_i64().unwrap() as f64 * cell); // 칸 수 → 원본 화면이 보내는 행 달러
            let p = match c["kind"].as_str().unwrap() {
                "days_payload" => format!("/api/footprint-daily?asset={asset}"),
                "cells_payload" => format!("/api/footprint-daily/cells?asset={asset}&from={}&to={}&row={}", a[0].as_str().unwrap(), a[1].as_str().unwrap(), row(2)),
                "h4_payload" => format!("/api/footprint-4h?asset={asset}"),
                "h4_cells_payload" => format!("/api/footprint-4h/cells?asset={asset}&from={}&to={}&row={}", a[0].as_str().unwrap(), a[1].as_str().unwrap(), row(2)),
                k => panic!("{k}"),
            };
            let g = request(addr, "GET", &p, &[]);
            let (ws, wb) = match c.get("error") {
                Some(e) => (400, format!("400: {}", e.as_str().unwrap())),
                None => (200, c["body"].as_str().unwrap().to_string()),
            };
            if c["kind"] == "days_payload" && want_full.is_none() {
                want_full = Some(c["body"].as_str().unwrap().to_string());
            }
            n += 1;
            if g.status != ws || g.body != wb.as_bytes() {
                bad.push(format!("{asset} {p}: {} {:.200}", g.status, String::from_utf8_lossy(&g.body)));
            }
        }
        // ?since= (원본 _cols_since) · 400 갈래
        let full: Value = serde_json::from_str(want_full.as_ref().unwrap()).unwrap();
        let d = full["cols"]["d"].as_array().unwrap();
        let since = d[d.len() - 3].as_str().unwrap();
        let want_since = rtserve::py::dumps(&rtserve::fpd::cols_since(&full, Some(since)), (", ", ": "), true, None);
        let mut cases: Vec<(String, u16, String)> = vec![(format!("/api/footprint-daily?asset={asset}&since={since}"), 200, want_since)];
        for (q, why) in [("/api/footprint-daily?asset=doge", "asset"), ("/api/footprint-daily?asset=ETH", "asset"), ("/api/footprint-daily?asset={a}&since=2026-02-30", "bad_date"),
                         ("/api/footprint-daily?asset={a}&since=", "bad_date"), ("/api/footprint-daily/cells?asset={a}&from=2026-11-01", "from_to"),
                         ("/api/footprint-daily/cells?asset={a}&from=x&to=2026-11-02", "bad_date"), ("/api/footprint-4h?asset={a}&since=2026-11-21T05", "bad_key"),
                         ("/api/footprint-4h/cells?asset={a}&to=2026-11-21T08", "from_to"), ("/api/footprint-4h/cells?asset={a}&from=2026-11-21T05", "bad_key"), ("/api/footprint-4h/cells?asset={a}&from=2026-11-21&to=2026-11-21T08", "bad_key")] {
            cases.push((q.replace("{a}", asset), 400, format!("400: {why}")));
        }
        for (p, ws, wb) in cases {
            let g = request(addr, "GET", &p, &[]);
            n += 1;
            if g.status != ws || g.body != wb.as_bytes() {
                bad.push(format!("{asset} {p}: {} {:.200}", g.status, String::from_utf8_lossy(&g.body)));
            }
        }
        let _ = std::fs::remove_dir_all(root);
    }
    println!("fpd_http: 요청 {n} · 불일치 {}", bad.len());
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
