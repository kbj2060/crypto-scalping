//! 4h 변동성 QX4(+QX4I)를 원본(scripts/live_vol4h_qx4_20261025.py)과 대조. 정답지는 커밋하지 않는다(약 70MB):
//!     (tools/server_env.sh 의 환경) python rtmodel/tools/export_qx4.py && python rtmodel/tools/gen_qx4_fixture.py
//!     cargo test -j 2 --release --test qx4 -- --nocapture
use rtmodel::qx4::*;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

fn fb(v: &Value) -> f64 {
    f64::from_bits(v.as_u64().unwrap())
}

fn same(g: &Value, w: &Value, path: &str, bad: &mut Vec<String>, worst: &mut f64) {
    match w {
        Value::Object(m) if m.len() == 1 && m.contains_key("b") => {
            let (a, b) = (g.as_f64().unwrap_or(f64::NAN), fb(&m["b"]));
            if a.to_bits() != b.to_bits() {
                *worst = worst.max((a - b).abs() / b.abs().max(1e-300));
                if !path.ends_with("log_rv") && !path.ends_with("log_rv_iv") {
                    bad.push(format!("{path}: Rust {a:?} 대 파이썬 {b:?}"));
                }
            }
        }
        Value::Object(m) => {
            for (k, v) in m {
                same(&g[k], v, &format!("{path}.{k}"), bad, worst);
            }
            assert_eq!(g.as_object().map(|o| o.len()), Some(m.len()), "{path}: 필드 {g}");
        }
        _ if g != w => bad.push(format!("{path}: Rust {g} 대 파이썬 {w}")),
        _ => {}
    }
}

#[test]
fn qx4_parity() {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let art = Art::from_json(&rtmodel::read_json::<Value>(&d.join("qx4.json")).expect("python rtmodel/tools/export_qx4.py 먼저"));
    let fx: Value = rtmodel::read_json(&d.join("qx4_fixture.json")).expect("정답지 없음 -- 머리말");
    let (mut bad, mut worst) = (vec![], 0.0f64);
    let mut fe: BTreeMap<String, (usize, usize, f64)> = BTreeMap::new();
    let mut me: BTreeMap<&str, (usize, usize, f64)> = BTreeMap::new();
    for s in fx["sets"].as_array().unwrap() {
        let name = s["name"].as_str().unwrap();
        let now = s["now_ms"].as_f64().unwrap();
        let rows: HashMap<String, Vec<Value>> = s["rows"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_array().unwrap().clone())).collect();
        for coin in COINS {
            let mut g = payload(&rows, &art, now, coin);
            let iv = &s["out"][format!("{coin}_iv_input")];
            if let (Some(iv), true) = (iv.as_object().map(|_| fb(&iv["b"])), g["ok"] == Value::Bool(true) && has_iv(coin)) {
                let h = ((now as i64 - 60_000).div_euclid(3_600_000)) * 3_600_000;
                add_iv(&mut g, coin, iv, h);
            }
            same(&g, &s["out"][coin], &format!("{name}/{coin}"), &mut bad, &mut worst);
        }
        // 끝 40봉 피쳐·구성원(원인 보기)
        if let Some(dbg) = s["dbg"].as_object().filter(|o| !o.contains_key("err")) {
            let (grid, fs, _, _) = frames(&rows, now).unwrap();
            for coin in COINS {
                let dd = &dbg[coin];
                for (k, t) in dd["ts"].as_array().unwrap().iter().enumerate() {
                    let i = grid.iter().position(|x| *x == t.as_i64().unwrap()).unwrap();
                    for (c, v) in dd["feats"].as_object().unwrap() {
                        let (a, b) = (fs[coin][c.as_str()][i], fb(&v[k]));
                        let e = fe.entry(c.clone()).or_default();
                        e.1 += 1;
                        if a.to_bits() == b.to_bits() { e.0 += 1 } else { e.2 = e.2.max((a - b).abs() / b.abs().max(1e-300)) }
                    }
                    let m = debug_members(&art, coin, &fs[coin], i);
                    for (j, nm) in ["M3", "M6", "PX3", "PX6"].iter().enumerate() {
                        let b = fb(&dd[*nm][k]);
                        let e = me.entry(nm).or_default();
                        e.1 += 1;
                        if m[j].to_bits() == b.to_bits() { e.0 += 1 } else { e.2 = e.2.max((m[j] - b).abs() / b.abs().max(1e-300)) }
                    }
                }
            }
        }
    }
    let mut n_atm = 0;
    for a in fx["atm_iv"].as_array().unwrap() {
        let got = atm_iv(a["trades"].as_array().unwrap(), a["d"].as_i64().unwrap(), a["coin"].as_str().unwrap());
        assert_eq!(got.map(f64::to_bits), a["iv"].as_f64().map(f64::to_bits), "atm_iv d {}", a["d"]);
        n_atm += 1;
    }
    println!("피쳐(끝 40봉 · 41열 × 4코인):");
    for (c, (s, n, r)) in &fe {
        if s != n {
            println!("  {c:<16} {s}/{n} 비트 같음 · 최대 상대 {r:.2e}");
        }
    }
    println!("  나머지 {} 열 전부 비트 같음", fe.values().filter(|e| e.0 == e.1).count());
    for (nm, (s, n, r)) in &me {
        println!("구성원 {nm:<4} {s}/{n} 비트 같음 · 최대 상대 {r:.2e}");
    }
    println!("atm_iv {n_atm}칸 같음 · payload 다른 값 {} · log_rv 최대 상대 {worst:.2e}", bad.len());
    for b in bad.iter().take(10) {
        println!("  {b}");
    }
    assert!(bad.is_empty(), "반올림한 화면 값·문자열이 다르다");
    // 🔴반올림 전 로그 RV 는 MLP 의 f32 sgemm 가장자리 타일 합 순서 몫으로 1e-12 를 못 맞춘다(src/qx4.rs 머리말) -- 1e-6 상대로 본다
    assert!(worst <= 1e-6, "log_rv 상대 {worst:e}");
}

#[test]
fn qx4_refresh() {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let fx: Value = rtmodel::read_json(&d.join("qx4_rest_fixture.json")).expect("python rtmodel/tools/gen_qx4_rest_fixture.py 먼저");
    let mut rows: HashMap<String, Vec<Value>> = HashMap::new();
    let mut n = 0;
    for run in fx["runs"].as_array().unwrap() {
        let mut q: std::collections::VecDeque<&Value> = run["requests"].as_array().unwrap().iter().collect();
        let mut fetch = |p: &[(&'static str, String)]| -> Vec<Value> {
            let w = q.pop_front().expect("원본보다 요청이 많다");
            let wp: Vec<(String, String)> = w[0].as_array().unwrap().iter().map(|x| (x[0].as_str().unwrap().into(), x[1].as_str().unwrap().into())).collect();
            assert_eq!(p.iter().map(|(k, v)| (k.to_string(), v.clone())).collect::<Vec<_>>(), wp, "요청이 다르다");
            n += 1;
            w[1].as_array().unwrap().clone()
        };
        let ok = refresh(&mut fetch, &mut rows);
        assert!(q.is_empty(), "원본 요청이 남았다");
        assert_eq!(ok, run["ok"].as_bool().unwrap());
        for (c, ts) in run["rows"].as_object().unwrap() {
            let want: Vec<i64> = ts.as_array().unwrap().iter().map(|t| t.as_i64().unwrap()).collect();
            assert_eq!(rows[c].iter().map(|r| r[0].as_i64().unwrap()).collect::<Vec<_>>(), want, "{c} 캐시 행");
        }
    }
    println!("QX4 이어 받기 4회차 · 요청 {n} 개 원본 순서·쿼리 · 캐시 행 같음");
}

/// QX4I 옵션 ATM IV 받기(iv_now · fetch_option_trades) = 원본을 가짜 deribit_get 으로 부른 것과: 요청 URL 순서 · (iv, 정시) · 기억한 시간 칸. 정답지 = python rtmodel/tools/gen_qx4i_fixture.py
#[test]
fn qx4i_iv_now() {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let fx: Value = rtmodel::read_json(&d.join("qx4i_fixture.json")).expect("python rtmodel/tools/gen_qx4i_fixture.py 먼저");
    let mut state: HashMap<String, BTreeMap<i64, Option<f64>>> = HashMap::new();
    let (mut n, mut nreq) = (0, 0);
    for r in fx["runs"].as_array().unwrap() {
        let (now, coin) = (r["now"].as_f64().unwrap(), r["coin"].as_str().unwrap());
        let mut q: std::collections::VecDeque<&Value> = r["calls"].as_array().unwrap().iter().collect();
        let mut get = |url: &str| -> Option<Value> {
            let w = q.pop_front().expect("원본보다 요청이 많다");
            assert_eq!(url, w[0].as_str().unwrap());
            nreq += 1;
            (!w[1].is_null()).then(|| w[1].clone())
        };
        let st = state.entry(coin.into()).or_default();
        let (iv, hour) = iv_now(st, now, &mut get, coin);
        assert!(q.is_empty(), "{now} {coin}: 원본 요청이 남았다");
        assert_eq!(iv.map(f64::to_bits), r["iv"].as_f64().map(f64::to_bits), "{now} {coin} iv");
        assert_eq!(hour, r["hour"].as_i64(), "{now} {coin} 정시");
        assert_eq!(st.keys().copied().collect::<Vec<_>>(), r["keys"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect::<Vec<_>>(), "{now} {coin} 기억 칸");
        n += 1;
    }
    println!("QX4I iv_now: 회차 {n} · 요청 {nreq} · iv·정시·기억 칸 같음");
}
