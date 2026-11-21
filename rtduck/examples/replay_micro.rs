//! cargo run --release --example replay_micro -- <micro_input.json> <out.duckdb>  → 신호 글자 JSON 목록(표준 출력)
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let inp: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let (_, _, sigs) = rtduck::micro::replay(&inp, std::path::Path::new(&a[2]));
    println!("{}", serde_json::Value::from(sigs.into_iter().map(|x| x.1).collect::<Vec<_>>()));
}
