//! 맥락 정답지 입력을 Rust 수집기에 먹여 SQLite 를 만든다(tools/gen_ctx_fixture.py --check-readers 가 부른다). REST 몫은 빼고 웹소켓 몫만.
//!   cargo run --release --example replay_ctx -- <okx_ctx|binance_ctx|xyz> <input.json> <out_dir>
use serde_json::Value;
use std::path::Path;
use std::sync::{Arc, Mutex};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let inp: Value = serde_json::from_str(&std::fs::read_to_string(&a[2]).unwrap()).unwrap();
    let (i, dir) = (&inp["input"], Path::new(&a[3]));
    std::fs::create_dir_all(dir).unwrap();
    let strs = |v: &Value| v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect::<Vec<_>>();
    match a[1].as_str() {
        "okx_ctx" => drop(rtcollect::okx_ctx::replay(i.as_array().unwrap(), &dir.join("okx_ctx.sqlite")).unwrap()),
        "binance_ctx" => {
            use rtcollect::binance_ctx as bc;
            let con = bc::open_db(&dir.join("binance_ctx.sqlite")).unwrap();
            let p = Arc::new(Mutex::new(bc::Pending::default()));
            bc::replay_ws(i["ws"].as_array().unwrap(), &strs(&i["slow"]), &p);
            bc::flush(&con, &p);
        }
        "xyz" => drop(rtcollect::xyz::replay(i["events"].as_array().unwrap(), &strs(&i["coins"]), &dir.join("hl_xyz.sqlite")).unwrap()),
        k => panic!("모르는 종류 {k}"),
    }
}
