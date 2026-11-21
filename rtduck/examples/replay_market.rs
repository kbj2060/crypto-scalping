//! cargo run --release --example replay_market -- <market_input.json> <out 폴더>
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let inp: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let d = std::path::Path::new(&a[2]);
    rtduck::market::replay(&inp, &d.join("tail_risk.duckdb"), d);
}
