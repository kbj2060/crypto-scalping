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
}
