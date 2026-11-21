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

/// supply-1s 의 `oi` = hot oi_1s 꼬리(원본 collect_oi_1s 641~665 · supply_1s_payload 4299~4301): 초 → 마지막 값 · 지금−660 앞 버림 · 가장 늦은 초 −660 · sinceOi 뒤만
#[test]
fn supply_oi_from_hot() {
    let now = 1_795_000_000.4f64;
    let n = now as i64;
    let dir = std::env::temp_dir().join(format!("rtserve_oi1s_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("data/hot")).unwrap();
    let c = rusqlite::Connection::open(dir.join("data/hot/binance_ctx.sqlite")).unwrap();
    c.execute_batch("CREATE TABLE oi_1s (symbol TEXT, ts_ms INTEGER, open_interest REAL)").unwrap();
    for (s, ms, v) in [("ethusdt", n - 700, 0, 1.0), ("ethusdt", n - 655, 100, 2.0), ("ethusdt", n - 655, 900, 2.5), ("ethusdt", n - 300, 5, 3.0), ("ethusdt", n - 2, 0, 4.0), ("solusdt", n - 675, 0, 8.0), ("solusdt", n - 300, 0, 9.0)]
        .map(|(s, t, ms, v)| (s, t * 1000 + ms, v)) {
        c.execute("INSERT INTO oi_1s VALUES (?,?,?)", rusqlite::params![s, ms, v]).unwrap();
    }
    let cfg = rtserve::Config::new(dir.clone(), vec!["eth".into()], std::sync::Arc::new(move || now));
    let q = |since: Option<i64>| since.map(|x| [("sinceOi".to_string(), x.to_string())].into_iter().collect()).unwrap_or_default();
    let v = rtserve::hub::supply_payload(&cfg, &cfg.live, "ethusdt", &q(None));
    assert_eq!(v["oi"], json!([[n - 655, 2.5], [n - 300, 3.0], [n - 2, 4.0]]), "{}", v["oi"]);
    let v = rtserve::hub::supply_payload(&cfg, &cfg.live, "ethusdt", &q(Some(n - 300)));
    assert_eq!(v["oi"], json!([[n - 2, 4.0]]));
    // 가장 늦은 초가 오래되면(n−300) 창 −660 은 n−960 까지라 n−675 도 들어올 뻔했다 -- 링의 «지금−660 앞 버림»이 막는다
    let v = rtserve::hub::supply_payload(&cfg, &cfg.live, "solusdt", &q(None));
    assert_eq!(v["oi"], json!([[n - 300, 9.0]]));
    let _ = std::fs::remove_dir_all(dir);
}

/// /api/situation: 코어 Situation 이 오기 전 = 원본 «계산 전»(_coin_ctx 1777) 모양 · 한 틱 뒤에도 코어가 없으면 그대로
#[test]
fn situation_before_core() {
    let (addr, root) = common::start(&serde_json::Map::new(), 0, 1.795e9, "eth", "sit0");
    let g = request(addr, "GET", "/api/situation", &[]);
    let body = String::from_utf8_lossy(&g.body).into_owned();
    assert_eq!(g.status, 200);
    assert!(body.contains("\"reason\": \"\\uacc4\\uc0b0 \\uc804\"") && body.contains("\"computed_at\": 0.0"), "{body}");
    let _ = std::fs::remove_dir_all(root);
}
