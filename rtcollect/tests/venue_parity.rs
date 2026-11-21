//! Bybit 청산 · Bybit 체결 테이프 · OKX 체결 테이프 파이썬 일치 시험. 정답지 = tools/gen_venue_fixture.py 가 **원본 수집 루프**
//! (`run()`·`collect()`)를 가짜 웹소켓으로 돌려 만든 SQLite 의 덤프. 같은 사건 열을 Rust 에 넣어 행 단위로 비교한다(tests/common).
mod common;
use serde_json::Value;

fn run(kind: &str, input: &str, golden: &str, norm: Option<common::Norm>, replay: impl Fn(&[Value], &std::path::Path)) {
    let input: Value = serde_json::from_str(input).unwrap();
    let golden: Value = serde_json::from_str(golden).unwrap();
    let dir = common::tmpdir(kind);
    replay(input["input"].as_array().unwrap(), &dir);
    let (t, r, c) = common::compare(&dir, &golden, &[], norm);
    eprintln!("{kind} 일치: 파일 {} · 표 {t} · 행 {r} · 칸 {c}", golden.as_object().unwrap().len());
    let _ = std::fs::remove_dir_all(&dir);
}

/// 청산 gaps.reason = 파이썬 `f"{type(exc).__name__}: {exc}"[:160]`. 연결 안 예외(깨진 JSON 등)의 글은 구현마다 다르다 --
/// 그 넷(Rust 가 메시지를 읽다 내는 예외)만 이름(«:» 앞)으로 비교한다. 연결 끝·접속 실패·KeyError 는 글 전체(160 글자 자르기 포함).
fn liq_reason(db: &str, t: &str, col: usize, v: &str) -> Option<String> {
    let name = v.split(':').next().unwrap();
    (db == "bybit_liq.sqlite" && t == "gaps" && col == 2 && ["JSONDecodeError", "TypeError", "ValueError", "AttributeError"].contains(&name))
        .then(|| name.to_string())
}

#[test]
fn bybit_liq_rows_match_python_collector() {
    run("bybit_liq", include_str!("fixtures/bybit_liq_input.json"), include_str!("fixtures/bybit_liq_golden.json"), Some(liq_reason),
        |v, d| rtcollect::bybit_liq::replay(v, &d.join("bybit_liq.sqlite")).unwrap());
}

#[test]
fn bybit_tape_rows_match_python_collector() {
    run("bybit_tape", include_str!("fixtures/bybit_tape_input.json"), include_str!("fixtures/bybit_tape_golden.json"), None,
        |v, d| rtcollect::bybit_tape::replay(v, d).unwrap());
}

#[test]
fn okx_tape_rows_match_python_collector() {
    run("okx", include_str!("fixtures/okx_input.json"), include_str!("fixtures/okx_golden.json"), None,
        |v, d| rtcollect::okx::replay(v, &d.join("okx_tape.sqlite")).unwrap());
}
