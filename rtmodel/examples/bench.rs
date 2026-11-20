//! 한 줄 예측 시간(중앙·상위 1%). 정답지 행을 차례로 쓴다.  cargo run -j 2 --release --example bench
use rtmodel::*;
use std::path::PathBuf;
use std::time::Instant;

fn stats(name: &str, mut ns: Vec<u128>) {
    ns.sort();
    let q = |p: f64| ns[((ns.len() - 1) as f64 * p) as usize] as f64 / 1000.0;
    println!("{name:<28} n={} 중앙 {:.1}µs · 상위1% {:.1}µs · 최대 {:.1}µs", ns.len(), q(0.5), q(0.99), q(1.0));
}

fn main() {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let fx: serde_json::Value = read_json(&d.join("pred_fixture.json")).unwrap();
    let rows = |v: &serde_json::Value| -> Vec<Vec<f64>> {
        v.as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|b| f64::from_bits(b.as_u64().unwrap())).collect()).collect()
    };
    let b: BoxBundle = read_json(&d.join("box_eth.json")).unwrap();
    let case = fx["hgb_cases"].as_array().unwrap().iter().find(|c| c["file"] == "box_eth.json" && c["scale"] == "1h").unwrap();
    let xs = rows(&case["rows"]);
    let m = &b.scales["1h"].model;
    let mut ns = vec![];
    for i in 0..20_000 {
        let t = Instant::now();
        std::hint::black_box(predict_proba(m, std::hint::black_box(&xs[i % xs.len()])));
        ns.push(t.elapsed().as_nanos());
    }
    stats("박스 ETH 1h HGB(400트리)", ns);
    let sr: SrLean = read_json(&d.join("sr_lean.json")).unwrap();
    let xs = rows(&fx["sr_lean"]["rows"]);
    let mut ns = vec![];
    for i in 0..20_000 {
        let t = Instant::now();
        std::hint::black_box(sr.predict(std::hint::black_box(&xs[i % xs.len()])));
        ns.push(t.elapsed().as_nanos());
    }
    stats("레벨 5분 앙상블(HGB+ET300+LR)", ns);

    // 박스 유지: 봉 열 1,559개 → payload 한 번(피쳐 세 척도 + HGB 세 번 + 4h 등조). tests/fixtures/box_fixture.json 의 ETH·SOL 보통 봉 열
    let bx: serde_json::Value = read_json(&d.join("box_fixture.json")).unwrap();
    let iso: Qx4iIso = read_json(&d.join("qx4i_iso.json")).unwrap();
    let vol = box_hold::Vol { ok: true, asof_ms: 0.0, log_rv: -10.0, log_rv_iv: Some(-10.0) };
    for asset in ["eth", "sol"] {
        let b: BoxBundle = read_json(&d.join(format!("box_{asset}.json"))).unwrap();
        let s = bx["sets"].as_array().unwrap().iter().find(|s| s["asset"] == asset && s["name"] == "normal").unwrap();
        let bars: Vec<box_hold::Bar> = s["bars"].as_array().unwrap().iter().map(|r| {
            let f = |i: usize| f64::from_bits(r[i].as_u64().unwrap());
            box_hold::Bar { t_ms: r[0].as_i64().unwrap(), h: f(1), l: f(2), c: f(3), v: f(4), tb: f(5) }
        }).collect();
        let vol = box_hold::Vol { asof_ms: (bars[bars.len() - 1].t_ms + 60_000) as f64, ..vol };
        let qx = (asset == "eth").then_some((&iso, &vol));
        let mut ns = vec![];
        for _ in 0..3_000 {
            let t = Instant::now();
            std::hint::black_box(box_hold::payload(std::hint::black_box(&bars), &[], &b, asset, None, qx));
            ns.push(t.elapsed().as_nanos());
        }
        stats(&format!("박스 {asset} 봉 열 → payload"), ns);
    }

    // 레벨 5분: 정답지 한 벌(normal0)의 연산을 Live 로 다시 흘려 second() 한 번 시간(사건 채점이 든 초 · 안 든 초 따로) -- tools/bench_py.py sr 와 같은 벌
    let sx: serde_json::Value = read_json(&d.join("sr_fixture.json")).unwrap();
    let set = sx["sets"].as_array().unwrap().iter().find(|s| s["name"] == "normal0").unwrap();
    let kl = |v: &serde_json::Value| -> Vec<sr_lean::Kline> { v.as_array().unwrap().iter().map(|r| sr_lean::Kline::from_rest(r.as_array().unwrap()).unwrap()).collect() };
    let mut lv = sr_lean::Live::new(&sr);
    let (mut plain, mut scored) = (vec![], vec![]);
    for op in set["ops"].as_array().unwrap() {
        let o = op.as_array().unwrap();
        let i = |k: usize| o[k].as_i64().unwrap();
        match o[0].as_str().unwrap() {
            "trade" => lv.trade(i(1), o[2].as_f64().unwrap(), o[3].as_f64().unwrap(), i(4), i(5), o[6].as_bool().unwrap()),
            "init" => (lv.t0, lv.done) = (Some(i(1)), Some(i(1))),
            "klines" => lv.klines(&kl(&o[1]), i(2)),
            "recheck" => lv.recheck(&kl(&o[1])),
            "hole" => lv.hole(i(1)),
            _ => {
                for s in lv.done.unwrap() + 1..=i(1) {
                    let k0 = lv.feat.len();
                    let t = Instant::now();
                    lv.second(s);
                    let dt = t.elapsed().as_nanos();
                    if lv.feat.len() > k0 { scored.push(dt) } else { plain.push(dt) }
                }
            }
        }
    }
    stats("레벨 5분 second() 채점 없는 초", plain);
    stats("레벨 5분 second() 채점이 든 초", scored);

    // 4시간 방향: SOL «normal» 벌(9,000봉 · 20코인) → 피쳐 + payload 한 번 -- tools/bench_py_dir.py 와 같은 입력
    let dx: serde_json::Value = read_json(&d.join("dir4h_fixture.json")).unwrap();
    let s = dx["sets"].as_array().unwrap().iter().find(|s| s["coin"] == "SOL" && s["name"] == "normal").unwrap();
    let col = |v: &serde_json::Value| v.as_array().unwrap().iter().map(|b| f64::from_bits(b.as_u64().unwrap())).collect::<Vec<f64>>();
    let mc = ["sum_open_interest", "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"];
    let fr: std::collections::HashMap<String, dir4h::Frame> = s["frames"].as_object().unwrap().iter().map(|(k, f)| (k.clone(), dir4h::Frame {
        ts: f["ts"].as_array().unwrap().iter().map(|t| t.as_i64().unwrap()).collect(), high: col(&f["high"]), low: col(&f["low"]), close: col(&f["close"]),
        volume: col(&f["volume"]), trades: col(&f["trades"]), tbb: col(&f["taker_buy_base"]), m: mc.map(|c| col(&f[c])) })).collect();
    let dc: Vec<(i64, f64)> = s["dc"][0].as_array().unwrap().iter().zip(col(&s["dc"][1])).map(|(t, v)| (t.as_i64().unwrap(), v)).collect();
    let art = dir4h::Art::load(&d.join("dir4h_sol.json")).unwrap();
    let mut ns = vec![];
    for _ in 0..20 {
        let t = Instant::now();
        let f = dir4h::build_features(&dir4h::frames_for(&fr, "SOL"), &dc).unwrap();
        std::hint::black_box(dir4h::payload(&art, &f));
        ns.push(t.elapsed().as_nanos());
    }
    stats("4h 피쳐 + payload(9,000봉·20코인)", ns);
    // 15분·1시간 LightGBM: ETH 65열판 모델 40개(지평 2 × 시드 5 × 표적 4) 앙상블 한 줄
    let lx: serde_json::Value = read_json(&d.join("lgb_eth.json")).unwrap();
    let hs: Vec<Vec<Vec<(lgb::Booster, f64)>>> = lx["models"].as_object().unwrap().values().map(|v| lgb::load_seeds(v).unwrap()).collect();
    let rows = col(&lx["rows"]);
    let mut ns = vec![];
    for i in 0..2000 {
        let r = &rows[(i % 10_000) * 65..(i % 10_000 + 1) * 65];
        let t = Instant::now();
        for h in &hs {
            std::hint::black_box(lgb::ensemble(h, r));
        }
        ns.push(t.elapsed().as_nanos());
    }
    stats("15분·1시간 LightGBM 앙상블 한 줄(40)", ns);

    // 위험 레이더: ETH 한 코인(31일 1분봉 + DVOL) -- 정답지 첫 벌
    let rx: serde_json::Value = read_json(&d.join("radar_fixture.json")).unwrap();
    let pp: risk_radar::Params = serde_json::from_str(&std::fs::read_to_string(d.join("../../../dashboard").join(rx["params"].as_str().unwrap())).unwrap()).unwrap();
    let x = &rx["sets"][0]["inputs"]["ETH"];
    let fl = |v: &serde_json::Value| v.as_array().unwrap().iter().map(|a| a.as_f64().unwrap()).collect::<Vec<f64>>();
    let il = |v: &serde_json::Value| v.as_array().unwrap().iter().map(|a| a.as_i64().unwrap()).collect::<Vec<i64>>();
    let b = risk_radar::Bars { t: il(&x["t"]), h: fl(&x["h"]), l: fl(&x["l"]), c: fl(&x["c"]), v: fl(&x["v"]) };
    let dv = risk_radar::Dvol { ts: il(&x["dvol"]["ts"]), close: fl(&x["dvol"]["close"]), got: il(&x["dvol"]["got"]) };
    let now = rx["sets"][0]["now_ms"].as_i64().unwrap();
    let mut ns = vec![];
    for _ in 0..30 {
        let t = Instant::now();
        std::hint::black_box(risk_radar::coin_out(&b, &pp.coins["ETH"], &pp, Some(&dv), now));
        ns.push(t.elapsed().as_nanos());
    }
    stats("위험 레이더 ETH 한 코인(31일)", ns);
}
