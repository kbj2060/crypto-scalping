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

/// 4h 상대 강도(xsec): 서버 tick 의 9코인 `_klines(c, 60, now)` 요청 열 → xsec_record 기록 글자·스냅숏. 정답지 = python rtmodel/tools/gen_xsec_fixture.py
#[test]
fn xsec_parity() {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let fx: Value = rtmodel::read_json(&d.join("xsec_fixture.json")).expect("python rtmodel/tools/gen_xsec_fixture.py 먼저");
    let tmp = std::env::temp_dir().join(format!("rt_xsec_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let mut rec = Record::default();
    let (mut n_req, mut n_line) = (0, 0);
    for run in fx["runs"].as_array().unwrap() {
        let now = run["now"].as_i64().unwrap();
        let mut want: VecDeque<&Value> = run["requests"].as_array().unwrap().iter().collect();
        let mut g = |path: &str, q: &[(&'static str, String)]| -> Result<Value, String> {
            let w = want.pop_front().expect("요청이 더 많다");
            let wq: Vec<(String, String)> = w[1].as_array().unwrap().iter().map(|p| (p[0].as_str().unwrap().into(), p[1].as_str().unwrap().into())).collect();
            assert_eq!((path, q.iter().map(|(k, v)| (k.to_string(), v.clone())).collect::<Vec<_>>()), (w[0].as_str().unwrap(), wq), "{now} 요청");
            n_req += 1;
            Ok(w[2].clone())
        };
        let px = xsec_px(&mut g, now).unwrap();
        assert!(want.is_empty(), "{now}: 요청이 모자란다");
        let (n, latest) = rec.xsec(&tmp, &px).unwrap();
        assert_eq!(n as u64, run["n"].as_u64().unwrap(), "{now} 새 줄");
        assert_eq!(latest.as_deref(), run["latest"].as_str(), "{now} 스냅숏");
        let file = std::fs::read_to_string(tmp.join("xsec.jsonl")).ok();
        assert_eq!(file.as_deref(), run["file"].as_str(), "{now} xsec.jsonl");
        n_line = file.map_or(0, |f| f.lines().count());
    }
    let _ = std::fs::remove_dir_all(&tmp);
    println!("xsec: 회차 {} · 요청 {n_req} · 기록 줄 {n_line}", fx["runs"].as_array().unwrap().len());
}

/// NUL 바이트가 낀 기록(서버 실측: BTC·SOL·XRP pred_record 에 NUL 59~115개 -- 쓰다 끊긴 자리를 파일 시스템이 0 으로 메움):
/// 원본 pred_record._tail 은 그 줄을 json.loads 실패로 건너뛴다(그 결정은 «안 쓴 것»이 되어 다시 쓴다) -- Rust 의 이어 쓰기도 같아야 한다.
/// 원본 실측(서버 판 venv): 줄 t=1 · NUL+t=2 · t=3 인 파일에 t=1~4 를 _append → _tail = [1, 3] · 새로 쓴 줄 2(t=2·t=4).
#[test]
fn nul_bytes_in_record_tail() {
    let d = std::env::temp_dir().join(format!("rt_nul_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let f = d.join("BTC.jsonl");
    std::fs::write(&f, b"{\"k\":\"4\",\"t\":1,\"px\":1.0,\"d\":2.0,\"s\":1}\n\0\0\0{\"k\":\"4\",\"t\":2,\"px\":1.0,\"d\":2.0,\"s\":1}\n{\"k\":\"4\",\"t\":3,\"px\":1.0,\"d\":2.0,\"s\":1}\n").unwrap();
    let pay: Value = serde_json::json!({"ok": true, "lo": -1.0, "hi": 1.0, "nan_feats": [], "decided_at_ms": 4, "hist": [[1, 2.0], [2, 2.0], [3, 2.0], [4, 2.0]]});
    let closes: HashMap<i64, f64> = (1..=4).map(|t| (t - 300_000, 1.0)).collect();
    let n = Record::default().h4(&d, "BTC", &pay, &closes).unwrap();
    assert_eq!(n, 2, "NUL 낀 줄(t=2)은 깨진 줄 = 안 쓴 것 → 다시 씀 + 새 t=4");
    let b = std::fs::read(&f).unwrap();
    assert!(b.ends_with(b"{\"k\":\"4\",\"t\":2,\"px\":1.0,\"d\":2.0,\"s\":1}\n{\"k\":\"4\",\"t\":4,\"px\":1.0,\"d\":2.0,\"s\":1}\n"), "원본과 같은 두 줄을 같은 순서로");
    let _ = std::fs::remove_dir_all(&d);
}

/// ETH 15분·1시간(75열판) 서빙 길 = 원본 ds.live_payload(서버 핸들러가 부르는 것)를 가짜 거래소·임시 sqlite 로 부른 것과 대조:
/// 요청 열(경로·쿼리 순서) · payload 숫자 비트 · 회차 둘(처음 31일치 → 5분 뒤 꼬리). 정답지 = python rtmodel/tools/gen_dirshort_eth_fixture.py
#[test]
fn eth_dir_short_handler() {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let fx: Value = rtmodel::read_json(&d.join("dirshort_eth_fixture.json")).expect("python rtmodel/tools/gen_dirshort_eth_fixture.py 먼저");
    let art = dirshort::Art::from_json(&rtmodel::read_json::<Value>(&d.join("ds_eth.json")).unwrap()).unwrap();
    let oos: Vec<(&str, Vec<f64>)> = dirshort::NAMES.iter().map(|(_, nm)| (*nm, fx["oos"][*nm].as_array().unwrap().iter().map(fb).collect())).collect();
    let db = std::env::temp_dir().join(format!("rt_eth_book_{}.sqlite", std::process::id()));
    let _ = std::fs::remove_file(&db);
    let con = rusqlite::Connection::open(&db).unwrap();
    con.execute(&format!("CREATE TABLE bars (t INTEGER PRIMARY KEY, warm INTEGER, {})", dirshort::BOOKF.iter().map(|c| format!("\"{c}\" REAL")).collect::<Vec<_>>().join(", ")), []).unwrap();
    for (t, r) in fx["book"].as_object().unwrap() {
        let mut vals: Vec<rusqlite::types::Value> = vec![rusqlite::types::Value::Integer(t.parse().unwrap()), rusqlite::types::Value::Integer(r[0].as_i64().unwrap())];
        vals.extend(r.as_array().unwrap()[1..].iter().map(|x| x.as_f64().map_or(rusqlite::types::Value::Null, rusqlite::types::Value::Real)));
        con.execute(&format!("INSERT INTO bars VALUES ({})", vec!["?"; vals.len()].join(", ")), rusqlite::params_from_iter(vals)).unwrap();
    }
    drop(con);
    let book = dirshort::book_sqlite(&db).unwrap();
    let mut rs = Rest::default();
    let (mut bad, mut nreq) = (vec![], 0);
    for st in fx["steps"].as_array().unwrap() {
        let now = st["now_ms"].as_i64().unwrap();
        let mut q: VecDeque<&Value> = st["requests"].as_array().unwrap().iter().collect();
        let mut get = |path: &str, params: &[(&'static str, String)]| -> Result<Value, String> {
            let w = q.pop_front().expect("원본보다 요청이 많다");
            let wp: Vec<(String, String)> = w[1].as_array().unwrap().iter().map(|p| (p[0].as_str().unwrap().into(), p[1].as_str().unwrap().into())).collect();
            assert_eq!((path, params.iter().map(|(k, v)| (k.to_string(), v.clone())).collect::<Vec<_>>()), (w[0].as_str().unwrap(), wp));
            nreq += 1;
            Ok(w[2].clone())
        };
        let pay = rtmodel::handlers::dir_short_body(&mut rs, &mut get, now, now as f64, "ETH", &art, &oos, Some(&book)).unwrap();
        assert!(q.is_empty(), "{now}: 원본 요청이 남았다");
        same(&pay, &st["payload"], &format!("ETH/{now}"), &mut bad);
    }
    let _ = std::fs::remove_file(&db);
    println!("ETH 15분·1시간 핸들러: 회차 {} · 요청 {nreq} · payload 다른 값 {}", fx["steps"].as_array().unwrap().len(), bad.len());
    for b in bad.iter().take(8) {
        println!("  {b}");
    }
    assert!(bad.is_empty());
}
