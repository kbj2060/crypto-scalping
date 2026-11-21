//! 30분 카드 방향·닿음(dashboard/flow_read.card30_features + server.py _card30_pred)을 원본과 비트 대조. 정답지 = python rtmodel/tools/gen_card30_fixture.py
//!     cargo test -j 2 --release --test card30 -- --nocapture
use rtmodel::card30::*;
use serde_json::Value;
use std::path::PathBuf;

#[test]
fn card30_parity() {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let m = Card30::from_json(&rtmodel::read_json::<Value>(&d.join("card30.json")).expect("card30.json")).unwrap();
    let fx: Value = rtmodel::read_json(&d.join("card30_fixture.json")).expect("정답지 없음 -- 머리말");
    let bars: Vec<Bar> = fx["bars"].as_array().unwrap().iter().map(|b| Bar { time: b["time"].as_i64().unwrap(), high: b["high"].as_f64().unwrap(), low: b["low"].as_f64().unwrap(),
        close: b["close"].as_f64().unwrap(), volume: b["volume"].as_f64().unwrap(), taker: b["taker"].as_f64().unwrap() }).collect();
    let (mut bad, mut nf, mut np) = (vec![], 0, 0);
    for cs in fx["cases"].as_array().unwrap() {
        let w: Vec<Bar> = if cs.get("gap").is_some() { bars[..1200].iter().chain(&bars[1201..1300]).copied().collect() } else {
            let e = cs["end"].as_u64().unwrap() as usize;
            bars[e.saturating_sub(300)..e].to_vec()
        };
        let f = features(&w);
        match (&f, cs["f"].as_object()) {
            (None, None) => assert!(predict(&m, &w).is_none()),
            (Some(g), Some(wf)) => {
                assert_eq!(g.len(), wf.len());
                for (k, v) in g {
                    nf += 1;
                    if v.to_bits() != wf[k].as_u64().unwrap() {
                        bad.push(format!("{} {k}: Rust {v:?} 대 파이썬 {:?}", cs["end"], f64::from_bits(wf[k].as_u64().unwrap())));
                    }
                }
                let (pd, pr) = predict(&m, &w).unwrap();
                for (k, g) in [("dir", pd), ("reach", pr)] {
                    np += 1;
                    if g.to_bits() != cs["p"][k].as_u64().unwrap() {
                        bad.push(format!("{} {k}: Rust {g:?} 대 파이썬 {:?}", cs["end"], f64::from_bits(cs["p"][k].as_u64().unwrap())));
                    }
                }
            }
            _ => bad.push(format!("{}: 있음/없음 다름", cs["end"])),
        }
    }
    println!("card30: 벌 {} · 피쳐 {nf}칸 · 확률 {np}개 · 다름 {}", fx["cases"].as_array().unwrap().len(), bad.len());
    assert!(bad.is_empty(), "{}", bad[..bad.len().min(20)].join("\n"));
}
