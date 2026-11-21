//! cargo run --release --example replay_hl_trades -- <hl_trades_input.json> <root 폴더>
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let inp: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    rtduck::hl_trades::replay(&inp, std::path::Path::new(&a[2]));
}
