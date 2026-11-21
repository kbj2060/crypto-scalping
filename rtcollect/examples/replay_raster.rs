//! raster 정답지 입력을 Rust 수집기에 먹여 .f32 를 만든다(tools/gen_raster_fixture.py --check-readers 가 부른다).
//!   cargo run --release --example replay_raster -- <input.json> <out_dir>
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let inp: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let i = &inp["input"];
    for run in i["runs"].as_array().unwrap() {
        rtcollect::raster::replay(run["t0"].as_f64().unwrap(), run["events"].as_array().unwrap(), i["symbol"].as_str().unwrap(),
                                  std::path::Path::new(&a[2]), run["snaps"].as_array().unwrap()).unwrap();
    }
}
