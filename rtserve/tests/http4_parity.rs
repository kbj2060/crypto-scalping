//! 넷째 조각 대조: tools/gen_http4_fixture.py 가 원본 api_oi_5m·oi_5m_buckets 로 뜬 바이트 == rtserve.
//! OKX 몫(원본 okx_d)은 코어 Micro 메시지(okx_oi5m·okx_from)로 · 청산 원(원본 liq_5m·okx_liq_events·bybit)은 코어 Liq5m 메시지로 Live 에 넣는다
//! -- 허브가 보내는 꼴 그대로(Live::apply). Liq5m 은 첫 몫(통째로)과 나머지(바뀐 봉 밀기)로 갈라 두 번 넣어 밀기 합치기도 같이 본다.

mod common;

use common::{b64, norm, request, start_with};
use serde_json::{json, Value};

#[test]
fn http4_parity() {
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/http4.json")).unwrap()).unwrap();
    let (now, mtime, assets) = (fx["now"].as_f64().unwrap(), fx["mtime_ns"].as_u64().unwrap(), fx["dashboard_assets"].as_str().unwrap());
    let (mut n, mut bad) = (0, Vec::new());
    for sc in fx["scenarios"].as_array().unwrap() {
        let name = sc["name"].as_str().unwrap();
        let micro = fx["micro"].as_object().unwrap().clone();
        let liq = fx["liq5m"].as_object().unwrap().clone();
        let (addr, root) = start_with(sc["files"].as_object().unwrap(), mtime, now, assets, &format!("h4_{name}"), move |cfg| {
            for (s, m) in micro {
                let msg = json!({"s": s, "seq": 1, "ex_ms": 0, "rx_us": 0, "tx_us": 0, "type": "micro", "ts": now as i64, "mid": null, "spread": null,
                                 "qi": null, "vol60": 0.0, "liq60": {"long": 0.0, "short": 0.0, "n": 0}, "okx_oi5m": m["okx_oi5m"], "okx_from": m["okx_from"]});
                cfg.live.apply(&msg.to_string());
                assert!(!cfg.live.syms.lock().unwrap()[&s].okx_oi5m.is_empty(), "Micro 가 안 들어감");
            }
            for (s, l) in liq {
                let bars = l["bars"].as_array().unwrap();
                let (a, b) = bars.split_at(bars.len() / 2);
                let head = |m: Value| json!({"s": s, "seq": 2, "ex_ms": 0, "rx_us": 0, "tx_us": 0, "type": "liq5m"}).as_object().unwrap().clone().into_iter().chain(m.as_object().unwrap().clone()).collect::<serde_json::Map<_, _>>();
                cfg.live.apply(&Value::Object(head(json!({"from_s": l["from_s"], "bars": a, "okx": {"from": null, "bars": []}}))).to_string());
                cfg.live.apply(&Value::Object(head(json!({"from_s": l["from_s"], "bars": b, "okx": l["okx"], "bybit": l["bybit"]}))).to_string());
                assert_eq!(cfg.live.syms.lock().unwrap()[&s].liq.as_ref().map(|x| x.bars.len()), Some(bars.len()), "Liq5m 이 안 들어감");
            }
        });
        for c in sc["cases"].as_array().unwrap() {
            n += 1;
            let p = c["path"].as_str().unwrap();
            let rh: Vec<(String, String)> = c["req_headers"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_str().unwrap().into())).collect();
            let got = request(addr, "GET", p, &rh);
            let want_h: Vec<(String, String)> = c["headers"].as_array().unwrap().iter().map(|h| (h[0].as_str().unwrap().into(), h[1].as_str().unwrap().into())).collect();
            let want = b64(c["body_b64"].as_str().unwrap());
            if got.status != c["status"].as_u64().unwrap() as u16 || norm(&got.headers) != norm(&want_h) || got.body != want {
                bad.push(format!("[{name}] {p}: {} {:?}\n  rust {}\n  py   {}\n  rh {:?}\n  ph {:?}", got.status, c["status"], String::from_utf8_lossy(&got.body[..got.body.len().min(600)]),
                                 String::from_utf8_lossy(&want[..want.len().min(600)]), norm(&got.headers), norm(&want_h)));
            }
        }
        let _ = std::fs::remove_dir_all(root);
    }
    eprintln!("http4_parity: 요청 {n} · 불일치 {}", bad.len());
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
