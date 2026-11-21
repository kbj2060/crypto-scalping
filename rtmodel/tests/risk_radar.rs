//! 위험 레이더(risk_radar.rs)를 원본 워커(scripts/live_risk_radar_20261115.py compute + main 조립)와 대조 -- latest.json·forward.jsonl 글자.
//! 정답지는 커밋하지 않는다(약 120MB):  NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL" python rtmodel/tools/gen_radar_fixture.py
//!     cargo test -j 2 --release --test risk_radar -- --nocapture
use rtmodel::risk_radar::*;
use serde_json::Value;
use std::path::PathBuf;

fn fl(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}
fn il(v: &Value) -> Vec<i64> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}

/// 두 JSON 의 숫자 칸별 최대 상대 오차(글자가 다를 때 원인 보기).
fn diff(a: &Value, b: &Value, path: &str, out: &mut Vec<(String, f64)>) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for (k, v) in y {
                diff(&x[k], v, &format!("{path}.{k}"), out)
            }
        }
        (Value::Number(x), Value::Number(y)) if x != y => {
            let (p, q) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            out.push((path.into(), (p - q).abs() / q.abs().max(1e-300)));
        }
        _ if a != b => out.push((format!("{path} {a} 대 {b}"), f64::INFINITY)),
        _ => {}
    }
}

#[test]
fn radar_parity() {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fx: Value = rtmodel::read_json(&d.join("tests/fixtures/radar_fixture.json")).expect("정답지 없음 -- 머리말의 python 한 줄을 먼저");
    assert!(fx["npy_disable"].as_str().is_some_and(|s| s.contains("AVX512")), "AVX-512 끈 numpy 로 만든 정답지(tools/server_env.sh)");
    let pname = fx["params"].as_str().unwrap();
    let pp: Params = serde_json::from_str(&std::fs::read_to_string(d.join("../dashboard").join(pname)).unwrap()).unwrap();
    let order = ["ETH", "BTC", "SOL", "XRP"];
    // 세션 계수 파일(SOXL)엔 cols·문턱이 없다 -- 코인 칸만 읽는다
    let sv: Value = serde_json::from_str(&std::fs::read_to_string(d.join("../dashboard/risk_radar_params_soxl_20261117.json")).unwrap()).unwrap();
    let ps: std::collections::HashMap<String, CoinP> = serde_json::from_value(sv["coins"].clone()).unwrap();
    let (mut same, mut tot, mut worst, mut nlog, mut nlog_all) = (0, 0, Vec::<(String, f64)>::new(), 0, 0);
    for s in fx["sets"].as_array().unwrap() {
        let now = s["now_ms"].as_i64().unwrap();
        let mut coins: Vec<(&'static str, J)> = order.iter().map(|&c| {
            let x = &s["inputs"][c];
            let b = Bars { t: il(&x["t"]), h: fl(&x["h"]), l: fl(&x["l"]), c: fl(&x["c"]), v: fl(&x["v"]) };
            let dv = x["dvol"].as_object().map(|v| Dvol { ts: il(&v["ts"]), close: fl(&v["close"]), got: il(&v["got"]) });
            (c, coin_out(&b, &pp.coins[c], &pp, dv.as_ref(), now))
        }).collect();
        let x = &s["inputs"]["SOXL"];
        let b = Bars { t: il(&x["t"]), h: fl(&x["h"]), l: fl(&x["l"]), c: fl(&x["c"]), v: fl(&x["v"]) };
        coins.push(("SOXL", session_out(&b, &ps["SOXL"], now)));
        let (latest, log) = assemble(now, pname, coins);
        tot += 1;
        if latest == s["latest"].as_str().unwrap() {
            same += 1;
        } else {
            diff(&serde_json::from_str(&latest).unwrap(), &serde_json::from_str(s["latest"].as_str().unwrap()).unwrap(), s["name"].as_str().unwrap(), &mut worst);
        }
        let want: Vec<&str> = s["log"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(log.len(), want.len());
        for (g, w) in log.iter().zip(&want) {
            nlog_all += 1;
            nlog += (g == w) as usize;
        }
    }
    worst.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("벌 {tot} · latest.json 글자 같음 {same} · forward 줄 같음 {nlog}/{nlog_all} · 숫자가 다른 칸 {} (문자열·참거짓·None 은 전부 같음 -- 다르면 ∞ 로 찍힌다)", worst.len());
    for w in worst.iter().take(12) {
        println!("  {} {:.2e}", w.0, w.1);
    }
    // 🔴글자까지는 못 맞춘다: σ̂ = c·exp(2·X@β) 의 X@β 가 numpy BLAS(개발기 MKL) 합 순서다(차례 합과 1~수 ulp). 서버 BLAS 는 미확인 -- 1e-12 상대로 본다.
    assert!(worst.iter().all(|w| w.1 <= 1e-12), "1e-12 넘는 차이");
}

#[test]
fn session_clock() {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fx: Value = rtmodel::read_json(&d.join("tests/fixtures/radar_fixture.json")).unwrap();
    let mut n = 0;
    for r in fx["clock"].as_array().unwrap() {
        let (tau, h) = (r[0].as_i64().unwrap(), r[1].as_f64().unwrap());
        let (s, oi, f60, l30) = clock(tau, h);
        assert_eq!((s as i64, oi, f60, l30), (r[2].as_i64().unwrap(), r[3].as_f64().unwrap(), r[4].as_f64().unwrap(), r[5].as_f64().unwrap()), "τ {} H {h}", iso(tau * 1000));
        n += 1;
    }
    println!("세션 시계 {n}칸 같음(서머타임 전환·휴장·개장/마감 경계 포함)");
}

#[test]
fn iso_fmt() {
    assert_eq!(iso(1_792_067_400_000), "2026-10-15T12:30:00Z");
    assert_eq!(iso(0), "1970-01-01T00:00:00Z");
    assert_eq!(iso(951_782_400_000), "2000-02-29T00:00:00Z");
}
