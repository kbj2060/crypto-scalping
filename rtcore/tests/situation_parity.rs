//! 30분 상황 근거(situation.rs) ↔ 원본(_situation_inputs 사본 + situation.classify + micro_ref.deriv_from_ring) 일치.
//! 정답지: rtcore/tools/gen_situation_fixture.py → tests/fixtures/situation_eth.json.
//! 걸음마다 같은 입력(풋프린트 봉·1초 링·캔들·BTC·청산·호가 중간값)을 넣고, 마크 사건은 받은 시각 순으로 MarkRing 에 넣어
//! 링(처음·끝 초·크기)과 classify 결과(JSON 값 -- 실수는 비트까지)를 맞춘다. prev_dir 는 앞 걸음 결과에서 잇는다.
use rtcore::proto::{Event, Stamped};
use rtcore::situation::{classify, inputs, MarkRing};
use rtcore::state::State;
use serde_json::Value;
use std::collections::BTreeMap;

fn map3(v: &Value) -> BTreeMap<i64, [f64; 3]> {
    v.as_object().map_or_else(BTreeMap::new, |o| {
        o.iter().map(|(k, x)| (k.parse().unwrap(), [x[0].as_f64().unwrap(), x[1].as_f64().unwrap(), x[2].as_f64().unwrap()])).collect()
    })
}

#[test]
fn situation_matches_python() {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/situation_eth.json")).unwrap();
    let fx: Value = serde_json::from_str(&raw).unwrap();
    let bucket = fx["bucket"].as_f64().unwrap();
    let (mut n, mut ok_n, mut bad) = (0, 0, vec![]);
    for (ci, c) in fx["cases"].as_array().unwrap().iter().enumerate() {
        let bars: BTreeMap<i64, BTreeMap<i64, [f64; 6]>> = c["bars"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(b, cells)| {
                let cells = cells.as_object().unwrap().iter().map(|(k, v)| {
                    let a: Vec<f64> = v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
                    (k.parse().unwrap(), [a[0], a[1], a[2], a[3], a[4], a[5]])
                });
                (b.parse().unwrap(), cells.collect())
            })
            .collect();
        let sec_all: BTreeMap<i64, [f64; 7]> = c["sec"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(s, v)| {
                let a: Vec<f64> = v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
                (s.parse().unwrap(), [a[0], a[1], a[2], a[3], a[4], a[5], a[6]])
            })
            .collect();
        let candles = map3(&c["candles"]);
        let btc = (!c["btc"].is_null()).then(|| map3(&c["btc"]));
        let liq: BTreeMap<i64, [f64; 2]> =
            c["liq"].as_object().unwrap().iter().map(|(k, x)| (k.parse().unwrap(), [x[0].as_f64().unwrap(), x[1].as_f64().unwrap()])).collect();
        let marks = c["marks"].as_array().unwrap();
        let mut ring = MarkRing::new(c["mark_t0"].as_f64().unwrap());
        let (mut mi, mut prev) = (0, 0i64);
        for (st, mid) in c["steps"].as_array().unwrap().iter().zip(c["mids"].as_array().unwrap()) {
            let now = st["now"].as_f64().unwrap();
            while mi < marks.len() && (marks[mi][0].as_i64().unwrap() + marks[mi][4].as_i64().unwrap()) as f64 / 1000.0 <= now {
                let m = &marks[mi];
                let (ts, lag) = (m[0].as_i64().unwrap(), m[4].as_i64().unwrap());
                ring.push(ts, (m[1].as_f64().unwrap(), m[2].as_f64().unwrap(), m[3].as_f64().unwrap()), (ts + lag) as f64 / 1000.0);
                mi += 1;
            }
            let rk = ring.ring.keys();
            let got_keys = (!ring.ring.is_empty()).then(|| serde_json::json!([rk.clone().next().unwrap(), rk.clone().next_back().unwrap(), ring.ring.len()]));
            let want_keys = (!st["ring_keys"].is_null()).then(|| st["ring_keys"].clone());
            let bs = (now as i64).div_euclid(300) * 300;
            let sec: BTreeMap<i64, [f64; 7]> = sec_all.iter().filter(|(&s, _)| now - 660.0 < s as f64 && s as f64 <= now).map(|(&s, v)| (s, *v)).collect();
            let closed = |m: &BTreeMap<i64, [f64; 3]>| m.iter().filter(|(&b, _)| b < bs).map(|(&b, v)| (b, *v)).collect::<BTreeMap<_, _>>();
            let fb = bars.range(..=bs).map(|(&b, cells)| (b, cells));
            let liq_c: BTreeMap<i64, [f64; 2]> = liq.iter().filter(|(&b, _)| b < bs).map(|(&b, v)| (b, *v)).collect();
            let inp = inputs(now, bucket, fb, &sec, &closed(&candles), btc.as_ref().map(closed).as_ref(), &liq_c, &ring.ring, mid.as_f64());
            let res = classify(&inp, prev);
            prev = res.dir.unwrap_or(0);
            let mut got = serde_json::to_value(&res).unwrap();
            let o = got.as_object_mut().unwrap();
            o.remove("computed_at");
            o.remove("missing");
            n += 1;
            ok_n += res.ok as usize;
            if got != st["res"] || got_keys != want_keys {
                bad.push(format!("경우 {ci} now {now}: 코어 {got} 링 {got_keys:?}\n  원본 {} 링 {want_keys:?}", st["res"]));
            }
        }
    }
    println!("상황 근거 대조: 걸음 {n} · ok {ok_n} · 어긋남 {}", bad.len());
    assert!(bad.is_empty(), "{}", bad[..bad.len().min(5)].join("\n"));
    assert!(n > 500 && ok_n > 400);
}

/// 배선: 체결 → 5분봉 캔들 · 1초 박자 → situation 밀기(바뀌었을 때만) · 스냅숏에 마지막 결과.
#[test]
fn situation_tick_pushes_on_change() {
    let st = State::new(&["ethusdt".into()], 16, 4096);
    let mut rx = st.tx.subscribe();
    let t0 = 1_763_700_000_000i64 / 300_000 * 300_000;
    let mut id = 0;
    for b in 0..9i64 {
        for (k, px) in [2600.0, 2610.0, 2590.0, 2600.0 + b as f64 * 5.0].into_iter().enumerate() {
            id += 1;
            let ex_ms = t0 + b * 300_000 + k as i64 * 1000;
            st.apply(Stamped { s: "ethusdt".into(), seq: 0, ex_ms, rx_us: ex_ms * 1000, tx_us: 0, ev: Event::Trade { id, px, qty: 1.0, sell: k == 1 } });
        }
    }
    assert_eq!(st.ohlc("ethusdt").unwrap()[2].1, [2610.0, 2590.0, 2610.0]);
    while rx.try_recv().is_ok() {}
    let now = (t0 + 8 * 300_000 + 5_000) * 1000;
    st.situation_tick(now);
    st.situation_tick(now + 1_000_000);
    let mut sits = vec![];
    while let Ok(m) = rx.try_recv() {
        if let Event::Situation(s) = m.ev {
            sits.push(s);
        }
    }
    assert_eq!(sits.len(), 1, "같은 결과는 한 번만 민다");
    let s = &sits[0];
    assert!(s.ok && s.dir == Some(1), "{s:?}");
    assert_eq!(s.missing, ["veto", "book"]);
    let Event::Snapshot { situation, .. } = st.snapshot("ethusdt", 0).unwrap().ev else { panic!() };
    assert_eq!(situation.unwrap().evidence, s.evidence);
}
