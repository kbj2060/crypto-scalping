//! 정답지 입력을 Rust 수집기에 먹여 SQLite 를 만든다(tools/gen_venue_fixture.py --check-readers 가 부른다).
//!   cargo run --release --example replay_venue -- <bybit_liq|bybit_tape|okx> <input.json> <out_dir>
use serde_json::Value;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let inp: Value = serde_json::from_str(&std::fs::read_to_string(&a[2]).unwrap()).unwrap();
    let (input, dir) = (&inp["input"], Path::new(&a[3]));
    std::fs::create_dir_all(dir).unwrap();
    let v = input.as_array().unwrap();
    match a[1].as_str() {
        "bybit_liq" => rtcollect::bybit_liq::replay(v, &dir.join("bybit_liq.sqlite")),
        "bybit_tape" => rtcollect::bybit_tape::replay(v, dir),
        "okx" => rtcollect::okx::replay(v, &dir.join("okx_tape.sqlite")),
        k => panic!("모르는 종류 {k}"),
    }
    .unwrap();
}
