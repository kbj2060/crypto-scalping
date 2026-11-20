//! 1초 수급 «나머지 원천»(바이낸스 강제청산 · 현물 · OKX · Bybit 청산) 파이썬 일치 시험.
//! 정답지 = rtcore/tools/gen_supply1s_venues_fixture.py(파이썬 수신 코드 원본 + server.py 조립 옮겨 적음).
//! 같은 메시지 원문을 실제 수신 경로(feed::parse* → State::apply)에 넣고 ① 중간·끝 칸(반올림 전) ② 스냅숏 전체
//! ③ 밀기만 받아 다시 지은 화면 쪽 칸(파이썬 창 안)을 대조한다. 실수는 상대 오차 1e-9, 나머지는 같음.
use rtcore::feed::{parse, parse_bybit, parse_okx, parse_spot};
use rtcore::proto::{Event, Stamped, Venue};
use rtcore::state::State;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};

const FX: &str = include_str!("fixtures/supply1s_venues_eth.json");

#[derive(Default)]
struct Tally {
    nums: usize,
    other: usize,
}

impl Tally {
    /// 파이썬 쪽(want)에 있는 키만 본다(Rust 는 retailMaxUsd 같은 키를 더 싣는다).
    fn eq(&mut self, path: &str, got: &Value, want: &Value) {
        match (got, want) {
            (Value::Number(g), Value::Number(w)) => {
                let (g, w) = (g.as_f64().unwrap(), w.as_f64().unwrap());
                assert!(g == w || (g - w).abs() <= 1e-9 * g.abs().max(w.abs()), "{path}: rust {g} != python {w}");
                self.nums += 1;
            }
            (Value::Array(g), Value::Array(w)) => {
                assert_eq!(g.len(), w.len(), "{path}: 길이 rust {got} python {want}");
                for (i, (a, b)) in g.iter().zip(w).enumerate() {
                    self.eq(&format!("{path}[{i}]"), a, b);
                }
            }
            (Value::Object(g), Value::Object(w)) => {
                for (k, b) in w {
                    self.eq(&format!("{path}.{k}"), g.get(k).unwrap_or_else(|| panic!("{path}.{k}: rust 에 없다")), b);
                }
            }
            _ => {
                assert_eq!(got, want, "{path}");
                self.other += 1;
            }
        }
    }
}

fn cells_json<const N: usize>(v: Vec<(i64, [f64; N])>) -> Value {
    Value::Object(v.into_iter().map(|(k, c)| (k.to_string(), json!(c.to_vec()))).collect())
}

/// 화면 쪽: 초 → 한 줄(app.js applySupply1s 처럼 덮어쓴다).
#[derive(Default)]
struct View {
    lanes: HashMap<&'static str, BTreeMap<i64, Value>>,
}

impl View {
    fn put(&mut self, lane: &'static str, rows: &Value) {
        for r in rows.as_array().unwrap() {
            self.lanes.entry(lane).or_default().insert(r[0].as_i64().unwrap(), r.clone());
        }
    }

    /// (lo, hi) 열린 구간 -- 닫힌 끝은 +1/-1 로 맞춰 부른다.
    fn range(&self, lane: &str, lo: i64, hi: i64) -> Value {
        let m = self.lanes.get(lane).cloned().unwrap_or_default();
        Value::Array(m.range(lo + 1..hi).map(|(_, v)| v.clone()).collect())
    }
}

#[test]
fn supply1s_venues_match_python() {
    let fx: Value = serde_json::from_str(FX).unwrap();
    // 파이썬 표의 사본이 갈라지지 않았는가.
    for (inst, ct) in fx["okx_ct_vals"].as_object().unwrap() {
        assert_eq!(rtcore::supply::okx_ct_val(inst), ct.as_f64(), "{inst} ctVal");
    }
    assert_eq!(rtcore::supply::OKX_CT_VALS.len(), fx["okx_ct_vals"].as_object().unwrap().len(), "ctVal 표 종목 수");
    let no_spot: Vec<&str> = fx["no_spot"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut ours = rtcore::supply::NO_SPOT.to_vec();
    ours.sort_unstable();
    assert_eq!(ours, no_spot, "현물 없는 종목");
    let by: Vec<&str> = fx["bybit_symbols"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(rtcore::supply::BYBIT_SYMBOLS, by.as_slice(), "Bybit 종목");
    assert_eq!(fx["liq_ring"], json!([rtcore::supply::LIQ_RING, rtcore::supply::LIQ_RING]), "청산 링 길이");

    let sym = fx["symbol"].as_str().unwrap().to_string();
    let st = State::new(std::slice::from_ref(&sym), 16, 1 << 16);
    let mut rx = st.tx.subscribe();
    let rows = fx["rows"].as_array().unwrap();
    let cps = fx["checkpoints"].as_array().unwrap();
    let mut t = Tally::default();
    let mut conn: HashMap<String, i64> = HashMap::new();
    let (mut ci, mut pushes) = (0, 0);
    let mut per_src: BTreeMap<&str, usize> = BTreeMap::new();
    let mut view = View::default();
    for (i, r) in rows.iter().enumerate() {
        let (src, c, wall, text) = (r[0].as_str().unwrap(), r[1].as_i64().unwrap(), r[2].as_i64().unwrap(), r[3].as_str().unwrap());
        let venue = match src {
            "spot" => Some(Venue::Spot),
            "okx" => Some(Venue::Okx),
            _ => None,
        };
        // feed::run_src 가 접속·끊김마다 부르는 것
        match conn.get(src).copied() {
            None => venue.into_iter().for_each(|v| st.feed_status(&sym, v, true)),
            Some(old) if old != c => {
                if let Some(v) = venue {
                    st.feed_status(&sym, v, false);
                    st.feed_status(&sym, v, true);
                }
                if src == "bn" {
                    st.reset_orders(&sym);
                }
            }
            _ => {}
        }
        conn.insert(src.to_string(), c);
        let rx_us = wall * 1000;
        let ms: Vec<Stamped> = match src {
            "bn" | "force" => parse(text, rx_us).into_iter().collect(),
            "spot" => parse_spot(text, &sym, rx_us).into_iter().collect(),
            "okx" => parse_okx(text, rx_us),
            "bybit" => parse_bybit(text, rx_us),
            _ => panic!("모르는 원천 {src}"),
        };
        *per_src.entry(src).or_default() += ms.len();
        for m in ms {
            st.apply(m);
        }
        while let Ok(m) = rx.try_recv() {
            if let Event::Supply1s(up) = m.ev {
                pushes += 1;
                let up = serde_json::to_value(&up).unwrap();
                for (lane, key) in [("bn", "seconds"), ("liq", "liq"), ("spot", "spot"), ("okx", "okx"), ("okxLiq", "okxLiq"), ("okxOi", "okxOi"), ("bybitLiq", "bybitLiq")] {
                    view.put(lane, &up[key]);
                }
                for lane in ["bn", "okx", "spot"] {
                    if !up["partial"][lane].is_null() {
                        view.put(lane, &json!([up["partial"][lane]]));
                    }
                }
            }
        }
        if ci < cps.len() && cps[ci]["after"].as_u64().unwrap() as usize == i + 1 {
            let cp = &cps[ci];
            let v = if cp["src"] == "spot" { Venue::Spot } else { Venue::Okx };
            t.eq(&format!("중간 {} {}", cp["src"], i + 1), &cells_json(st.venue_cells(&sym, v).unwrap()), &cp["sec"]);
            if !cp["oi"].is_null() {
                let oi = st.okx_oi_cells(&sym).unwrap();
                t.eq(&format!("중간 okx oi {}", i + 1), &Value::Object(oi.into_iter().map(|(k, x)| (k.to_string(), json!(x))).collect()), &cp["oi"]);
            }
            ci += 1;
        }
    }
    assert_eq!(ci, cps.len(), "중간 대조를 다 못 했다");
    let fc = &fx["final_cells"];
    t.eq("끝 바이낸스 선물 칸", &cells_json(st.supply_cells(&sym).unwrap()), &fc["bn"]);
    for (k, top) in [("bn", "now"), ("spot", "spotNow"), ("okx", "okxNow")] {
        let edge = (fx["final"][top].as_i64().unwrap() - 660).to_string();
        assert!(fc[k].get(&edge).is_some(), "정답지 {k} 에 보관 경계 칸({edge})이 없으면 정리 규칙을 못 본다");
    }
    t.eq("끝 현물 칸", &cells_json(st.venue_cells(&sym, Venue::Spot).unwrap()), &fc["spot"]);
    t.eq("끝 OKX 칸", &cells_json(st.venue_cells(&sym, Venue::Okx).unwrap()), &fc["okx"]);
    let oi = st.okx_oi_cells(&sym).unwrap();
    t.eq("끝 OKX OI", &Value::Object(oi.into_iter().map(|(k, x)| (k.to_string(), json!(x))).collect()), &fc["okx_oi"]);

    // 스냅숏 = 파이썬 supply_1s_payload(since=0) 전체(oi·symbol 빼고)
    let fin = &fx["final"];
    let final_us = fx["final_ms"].as_i64().unwrap() * 1000;
    let Event::Snapshot { supply, .. } = st.snapshot(&sym, final_us).unwrap().ev else { panic!() };
    let snap = serde_json::to_value(&supply).unwrap();
    t.eq("스냅숏", &snap, fin);
    for k in ["liq", "spot", "okx", "okxOi", "okxLiq", "bybitLiq"] {
        assert!(!fin[k].as_array().unwrap().is_empty(), "정답지 {k} 가 비면 시험이 아무것도 안 본다");
    }

    // 화면 쪽: 밀기만으로 지은 칸 중 파이썬 창 안이 since=0 응답과 같아야 한다.
    let w = 660;
    let (now, sp, ok) = (fin["now"].as_i64().unwrap(), fin["spotNow"].as_i64().unwrap(), fin["okxNow"].as_i64().unwrap());
    let with_partial = |rows: &Value, part: &Value| {
        let mut v = rows.as_array().unwrap().clone();
        v.push(part.clone());
        Value::Array(v)
    };
    t.eq("밀기 bn", &view.range("bn", now - w, now + 1), &with_partial(&fin["seconds"], &fin["partial"]["bn"]));
    t.eq("밀기 spot", &view.range("spot", sp - w, sp + 1), &with_partial(&fin["spot"], &fin["partial"]["spot"]));
    t.eq("밀기 okx", &view.range("okx", ok - w, ok + 1), &with_partial(&fin["okx"], &fin["partial"]["okx"]));
    t.eq("밀기 liq", &view.range("liq", now - w, now), &fin["liq"]);
    t.eq("밀기 okxLiq", &view.range("okxLiq", ok - w, ok), &fin["okxLiq"]);
    let oi_top = view.lanes["okxOi"].keys().next_back().copied().unwrap();
    t.eq("밀기 okxOi", &view.range("okxOi", oi_top - w, i64::MAX), &fin["okxOi"]);
    let b = final_us / 1_000_000 - 3;
    t.eq("밀기 bybitLiq", &view.range("bybitLiq", b - w, b + 1), &fin["bybitLiq"]);
    println!(
        "1초 수급 다른 원천 일치: 메시지 {} · 사건 {per_src:?} · 밀기 {pushes} · 중간 대조 {} · 비교한 수 {} · 그 밖의 값 {}",
        rows.len(),
        cps.len(),
        t.nums,
        t.other
    );
}
