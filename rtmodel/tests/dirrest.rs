//! 4h·15분/1시간 «REST 요청 열 → 캐시 → payload → 적중 기록» 을 원본(서버 pred_record_loop 의 길: rest_frames·daily_closes·live_payload·pred_record)과 대조.
//! 가짜 거래소의 요청 열이 같아야 하고(경로·쿼리 순서까지) · payload 숫자 비트 · data/live/pred_record/{코인}.jsonl 글자(전진 검정 fwd_dirshort 가 읽는 기록).
//!     python rtmodel/tools/gen_dirshort_fixture.py && python rtmodel/tools/gen_dir4h_fixture.py(dir4h_*.json) && python rtmodel/tools/gen_dirrest_fixture.py
//!     cargo test -j 2 --release --test dirrest -- --nocapture
use rtmodel::dirrest::*;
use rtmodel::{dir4h, dirshort};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

fn fb(v: &Value) -> f64 {
    f64::from_bits(v.as_u64().unwrap())
}

fn same(g: &Value, w: &Value, path: &str, bad: &mut Vec<String>) {
    match w {
        Value::Object(m) if m.len() == 1 && m.contains_key("b") => {
            if g.as_f64().map(f64::to_bits) != Some(fb(&m["b"]).to_bits()) {
                bad.push(format!("{path}: Rust {g} 대 파이썬 {:?}", fb(&m["b"])));
            }
        }
        Value::Object(m) => {
            for (k, v) in m {
                same(&g[k], v, &format!("{path}.{k}"), bad);
            }
        }
        Value::Array(a) => {
            for (i, v) in a.iter().enumerate() {
                same(&g[i], v, &format!("{path}[{i}]"), bad);
            }
        }
        _ if g != w => bad.push(format!("{path}: Rust {g} 대 파이썬 {w}")),
        _ => {}
    }
}

#[test]
fn rest_payload_record_parity() {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let fx: Value = rtmodel::read_json(&d.join("dirrest_fixture.json")).expect("정답지 없음 -- 머리말");
    let oos: Vec<(&str, Vec<f64>)> = dirshort::NAMES.iter().map(|(_, nm)| (*nm, fx["oos"][*nm].as_array().unwrap().iter().map(fb).collect())).collect();
    let mut a4: HashMap<String, dir4h::Art> = HashMap::new();
    let mut as_: HashMap<String, dirshort::Art> = HashMap::new();
    let (mut r4, mut rs) = (Rest::default(), Rest::default()); // 서버의 두 모듈 캐시
    let out = std::env::temp_dir().join(format!("rt_dirrest_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let mut rec = Record::default();
    let (mut bad, mut nreq) = (vec![], 0);
    for st in fx["steps"].as_array().unwrap() {
        let (kind, coin, now) = (st["kind"].as_str().unwrap(), st["coin"].as_str().unwrap(), st["now_ms"].as_i64().unwrap());
        let mut q: VecDeque<&Value> = st["requests"].as_array().unwrap().iter().collect();
        let tag = format!("{kind}/{coin}/{now}");
        let mut get = |path: &str, params: &[(&'static str, String)]| -> Result<Value, String> {
            let w = q.pop_front().unwrap_or_else(|| panic!("{tag}: 원본보다 요청이 많다 {path} {params:?}"));
            let wp: Vec<(String, String)> = w[1].as_array().unwrap().iter().map(|p| (p[0].as_str().unwrap().into(), p[1].as_str().unwrap().into())).collect();
            let gp: Vec<(String, String)> = params.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
            assert_eq!((path, &gp), (w[0].as_str().unwrap(), &wp), "{tag}: 요청이 다르다");
            nreq += 1;
            Ok(w[2].clone())
        };
        let r = if kind == "dir4h" { &mut r4 } else { &mut rs };
        let fr = r.frames(&mut get, now, now as f64, coin).unwrap();
        let dc = daily_closes(&mut get, now, coin).unwrap();
        assert!(q.is_empty(), "{tag}: 원본 요청이 {} 개 남았다", q.len());
        let f = dir4h::build_features(&dir4h::frames_for(&fr, coin), &dc).unwrap();
        let pay = if kind == "dir4h" {
            let art = a4.entry(coin.into()).or_insert_with(|| dir4h::Art::load(&d.join(format!("dir4h_{}.json", coin.to_lowercase()))).unwrap());
            dir4h::payload(art, &f)
        } else {
            let art = as_.entry(coin.into()).or_insert_with(|| dirshort::Art::from_json(&rtmodel::read_json::<Value>(&d.join(format!("ds_{}.json", coin.to_lowercase()))).unwrap()).unwrap());
            dirshort::payload(art, &f, 13, &oos)
        };
        same(&pay, &st["payload"], &tag, &mut bad);
        let cl = closes(r, coin).unwrap();
        if kind == "dir4h" { rec.h4(&out, coin, &pay, &cl).unwrap() } else { rec.short(&out, coin, &pay, &cl).unwrap() };
    }
    let mut nrec = 0;
    for (name, want) in fx["records"].as_object().unwrap() {
        let got = std::fs::read_to_string(out.join(name)).unwrap_or_default();
        assert_eq!(got, want.as_str().unwrap(), "{name}: 적중 기록 글자");
        nrec += got.lines().count();
    }
    println!("단계 {} · 요청 {nreq} 전부 같은 순서·쿼리 · payload 다른 값 {} · 적중 기록 {nrec}줄 글자 같음", fx["steps"].as_array().unwrap().len(), bad.len());
    for b in bad.iter().take(8) {
        println!("  {b}");
    }
    assert!(bad.is_empty());
}
