//! depth 정답지 입력을 Rust 수집기에 먹여 시각 파일을 만든다(tools/gen_depth_fixture.py --check-readers 가 부른다).
//!   cargo run --release --example replay_depth -- <input.json> <out_dir>
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let inp: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let i = &inp["input"];
    rtcollect::depth::replay(i["events"].as_array().unwrap(), i["symbol"].as_str().unwrap(), std::path::Path::new(&a[2]), i["snaps"].as_array().unwrap()).unwrap();
}
