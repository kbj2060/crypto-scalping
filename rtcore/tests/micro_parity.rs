//! 미시 값(situation.rs micro · okx_oi_rows) ↔ 원본 식. 정답지: rtcore/tools/gen_micro_fixture.py → tests/fixtures/micro_eth.json.
//! vol60 만 1e-9 상대 허용: 원본은 1초 dict 를 넣은 차례로 더한다(늦은 체결이 새 초를 만들면 그 초가 끝에 온다) -- 코어는 초 순서(situation.rs micro 주석).
//! 사건을 받은 시각 순으로 State 에 넣고 1초 틱(situation_tick)마다 밀린 Micro 를 받아 쌓은 값 · 스냅숏을 원본 값과 비트 대조한다.
use rtcore::proto::{Event, Micro, Stamped, Venue};
use rtcore::state::State;
use serde_json::Value;
use std::collections::BTreeMap;

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

#[test]
fn micro_matches_python() {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/micro_eth.json")).unwrap();
    let fx: Value = serde_json::from_str(&raw).unwrap();
    let st = State::new(&["ethusdt".into()], 16, 1 << 16);
    let mut rx = st.tx.subscribe();
    let evs = fx["events"].as_array().unwrap();
    let (mut ei, mut n, mut order, mut bad) = (0, 0, 0, vec![]);
    let mut last: Option<Micro> = None;
    let mut oi_rows: BTreeMap<i64, (f64, f64)> = BTreeMap::new();
    for t in fx["ticks"].as_array().unwrap() {
        let now = f(&t["now"]);
        while ei < evs.len() && f(&evs[ei][2]) / 1000.0 <= now {
            let e = &evs[ei];
            let (ex_ms, rx_us) = (e[1].as_i64().unwrap(), e[2].as_i64().unwrap() * 1000);
            let ev = match e[0].as_str().unwrap() {
                "trade" => Event::Trade { id: e[3].as_u64().unwrap(), px: f(&e[4]), qty: f(&e[5]), sell: e[6].as_bool().unwrap() },
                "book" => Event::Book { u: 0, bid: f(&e[3]), bid_qty: f(&e[4]), ask: f(&e[5]), ask_qty: f(&e[6]), bid_s: String::new(), ask_s: String::new() },
                "liq" => Event::Liq { venue: Venue::Bn, long: e[3].as_bool().unwrap(), qty: f(&e[4]), px: f(&e[5]) },
                _ => Event::Oi { venue: Venue::Okx, oi: f(&e[3]) },
            };
            st.apply(Stamped { s: "ethusdt".into(), seq: 0, ex_ms, rx_us, tx_us: 0, ev });
            ei += 1;
        }
        st.situation_tick((now * 1e6).round() as i64);
        while let Ok(m) = rx.try_recv() {
            if let Event::Micro(mc) = m.ev {
                for r in &mc.okx_oi5m {
                    oi_rows.insert(r.0, (r.1, r.2));
                }
                last = Some(mc);
            }
        }
        let m = last.as_ref().expect("첫 틱에 밀린다");
        n += 1;
        let opt = |k: &str| t.get(k).map(f);
        let want_rows: Vec<(i64, f64, f64)> = t["okx_d"].as_array().unwrap().iter().map(|r| (r[0].as_i64().unwrap(), f(&r[1]), f(&r[2]))).collect();
        let got_rows: Vec<(i64, f64, f64)> = want_rows.iter().filter_map(|r| oi_rows.get(&r.0).map(|v| (r.0, v.0, v.1))).collect();
        // ts 는 안 본다: 값이 그대로면 다시 밀지 않는다(받은 쪽의 ts = 마지막으로 바뀐 초)
        let same = m.mid == opt("mid")
            && m.spread == opt("spread")
            && m.qi == opt("qi")
            && (m.vol60 - f(&t["vol60"])).abs() <= 1e-9 * f(&t["vol60"]).abs().max(1.0)
            && m.liq60.long == f(&t["liq60"]["long"])
            && m.liq60.short == f(&t["liq60"]["short"])
            && m.liq60.n == t["liq60"]["n"].as_i64().unwrap()
            && got_rows == want_rows;
        order += (m.vol60 != f(&t["vol60"])) as usize;
        if !same {
            bad.push(format!("now {now}: 코어 {m:?} 봉 {:?}\n  원본 {t} ", got_rows.last()));
        }
    }
    let Event::Snapshot { micro, .. } = st.snapshot("ethusdt", 0).unwrap().ev else { panic!() };
    let want: Vec<(i64, f64, f64)> =
        fx["ticks"].as_array().unwrap().last().unwrap()["okx_d"].as_array().unwrap().iter().map(|r| (r[0].as_i64().unwrap(), f(&r[1]), f(&r[2]))).collect();
    println!("미시 값 대조: 틱 {n} · OKX 봉 {} · 어긋남 {} · vol60 합 차례만 다름(1e-9 안) {order}", want.len(), bad.len());
    assert!(order * 20 < n, "합 차례 차이가 늦은 체결 몫보다 많다");
    assert!(bad.is_empty(), "{}", bad[..bad.len().min(3)].join("\n"));
    assert_eq!(micro.unwrap().okx_oi5m, want, "스냅숏 = 가진 봉 전부");
    assert!(n > 3000);
}
