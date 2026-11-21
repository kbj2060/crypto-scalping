//! 체결 테이프 파이썬 일치 시험. 정답지 = tools/gen_tape_fixture.py 가 원본 collect() 로 만든 SQLite 의 덤프.
//! 같은 메시지 열을 rtcollect::binance::replay 에 넣어 만든 SQLite 를 같은 질의로 덤프해 **행 단위**로 비교한다(tests/common).
mod common;
use serde_json::Value;

const INPUT: &str = include_str!("fixtures/tape_input.json");
const GOLDEN: &str = include_str!("fixtures/tape_golden.json");
const TABLES: [(&str, &str); 5] = [
    ("trade_tape_1s", "symbol, ts_sec, price_bin"),
    ("gaps", "rowid"),
    ("verify_1m", "rowid"),
    ("backfill_1m", "rowid"),
    ("meta", "rowid"),
];

#[test]
fn tape_rows_match_python_collector() {
    let input: Value = serde_json::from_str(INPUT).unwrap();
    let golden: Value = serde_json::from_str(GOLDEN).unwrap();
    let dir = common::tmpdir("parity");
    let mut flushes = 0;
    rtcollect::binance::replay(input["runs"].as_array().unwrap(), &dir, |_, _| flushes += 1).unwrap();
    let (n_tables, n_rows, n_cells) = common::compare(&dir, &golden, &TABLES, None);
    eprintln!("일치: 파일 {} · 표 {n_tables} · 행 {n_rows} · 칸 {n_cells} · Rust 플러시 {flushes}회", golden.as_object().unwrap().len());
    let _ = std::fs::remove_dir_all(&dir);
}
