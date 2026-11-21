//! cargo run --release --example replay_hl_positions -- <hl_positions_input.json> <out.sqlite> <체결 폴더(만든다)>
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let inp: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let tr = std::path::Path::new(&a[3]);
    rtduck::hl_positions::make_trades(&inp["trades"], tr).unwrap();
    rtduck::hl_positions::replay(&inp, std::path::Path::new(&a[2]), tr);
}
