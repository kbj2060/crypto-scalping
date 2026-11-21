//! cargo run --release --example replay_oi -- <oi_input.json> <out.duckdb>
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let inp: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    rtduck::oi::replay(&inp, std::path::Path::new(&a[2]));
}
