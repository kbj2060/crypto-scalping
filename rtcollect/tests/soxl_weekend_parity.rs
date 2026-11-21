//! soxl_weekend 일치: tools/gen_soxl_weekend_fixture.py 가 원본 record() 를 세 번 돌린 정답지와
//!   덧붙인 jsonl 글자 · 요청 URL(순서) · 중간 실패(HTTP 500) 때 남는 줄을 대조한다. 가짜 응답 = 원본에 준 응답 본문(정답지 bodies)을 같은 URL 에 그대로.
use serde_json::Value;

#[test]
fn soxl_weekend_matches_python() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/soxl_weekend_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/soxl_weekend_golden.json")).unwrap();
    let d = std::env::temp_dir().join(format!("rtc_soxl_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let out = d.join("out.jsonl");
    std::fs::write(&out, inp["pre"].as_str().unwrap()).unwrap();
    let (mut lines, mut reqs_n) = (0, 0);
    for (run, g) in inp["runs"].as_array().unwrap().iter().zip(gold["runs"].as_array().unwrap()) {
        let want: Vec<&str> = g["reqs"].as_array().unwrap().iter().map(|r| r[0].as_str().unwrap()).collect();
        assert!(g["reqs"].as_array().unwrap().iter().all(|r| r[1] == 15), "timeout 15 = main.rs binance_rest(_, 15)");
        let bodies = gold_bodies(&want);
        let fail = run["fail_after"].as_u64().map(|x| x as usize);
        let mut got = vec![];
        let before = std::fs::read_to_string(&out).unwrap();
        let r = rtcollect::soxl_weekend::record(&out, run["now_s"].as_f64().unwrap(), &mut |u: &str| {
            got.push(u.to_string());
            if fail.is_some_and(|f| got.len() > f) {
                return Err("HTTP 500".into());
            }
            Ok(bodies[got.len() - 1].clone())
        });
        assert_eq!(got, want, "요청 순서");
        assert_eq!(r.is_err(), !g["error"].is_null(), "실패 여부 {r:?}");
        let after = std::fs::read_to_string(&out).unwrap();
        assert_eq!(&after[before.len()..], g["added"].as_str().unwrap(), "덧붙인 줄");
        lines += g["added"].as_str().unwrap().lines().count();
        reqs_n += got.len();
    }
    assert_eq!(rtcollect::soxl_weekend::status(&out).unwrap(), "전진 4주말 기록 · 마지막 2026-11-14 · 1차 점검 26 · 최종 52");
    let _ = std::fs::remove_dir_all(&d);
    eprintln!("SOXL 주말 일치: 새 줄 {lines} · 요청 {reqs_n} · 실행 3(마지막 = 중간 실패)");
}

/// URL → 원본에 준 본문(실패한 요청은 본문 없음).
fn gold_bodies(urls: &[&str]) -> Vec<Vec<u8>> {
    let gold: Value = serde_json::from_str(include_str!("fixtures/soxl_weekend_golden.json")).unwrap();
    let all: Vec<(String, String)> = gold["bodies"].as_array().unwrap().iter().map(|b| (b[0].as_str().unwrap().into(), b[1].as_str().unwrap().into())).collect();
    urls.iter().map(|u| all.iter().find(|(x, _)| x == u).map(|(_, b)| b.clone().into_bytes()).unwrap_or_default()).collect()
}
