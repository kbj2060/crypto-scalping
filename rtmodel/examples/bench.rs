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
}
