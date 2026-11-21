//! 일봉·4시간 풋프린트(src/fpd.rs) ↔ 원본 footprint_daily.DailyFootprint 걸음별 본문 글자 일치.
//! 정답지: rtserve/tools/gen_fpd_fixture.py(서버 판 venv 로 · 커밋 안 함 · tests/fixtures/gen/fpd/).
//! 같은 합성 원천(소급본 · lake · hot)에서 hot 은 처음 사본으로 시작해 걸음마다 원본과 같은 행을 더하고, 같은 시각으로 부른다.
//! 비교 = 원본 json.dumps(값)(aiohttp json_response 와 같은 기본 구분자) 글자 == rtserve::py::dumps(코어 값).
//! fpd.rs 는 아직 lib.rs 에 안 붙였다(라우팅은 나중) -- 그 파일만 끌어다 돌린다.
#[path = "../src/fpd.rs"]
#[allow(dead_code)]
mod fpd;

use serde_json::Value;
use std::path::PathBuf;

fn gen() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen/fpd")
}

#[test]
fn fpd_matches_python_step_by_step() {
    for a in ["eth", "xrp"] {
        one(a);
    }
}

fn one(asset: &str) {
    let p = gen().join(asset).join(format!("fpd_{asset}.json"));
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!("{}: {e} -- 서버 판 venv 로 `rtserve/tools/gen_fpd_fixture.py` 를 먼저 돌린다(커밋 안 함)", p.display())
    }))
    .unwrap();
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("fpd_{asset}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (tape, ctx) = (dir.join("hot_tape.sqlite"), dir.join("hot_ctx.sqlite"));
    std::fs::copy(gen().join(asset).join("hot_tape_start.sqlite"), &tape).unwrap();
    std::fs::copy(gen().join(asset).join("hot_ctx_start.sqlite"), &ctx).unwrap();
    let f = fpd::Fpd::new(fx["asset"].as_str().unwrap(), fpd::Src { lake: gen().join("lake"), hist: gen().join("hist"), hot_tape: tape.clone(), hot_ctx: ctx.clone() }).unwrap();
    let (mut calls, mut chars, mut errs, mut cols_n) = (0, 0usize, 0, 0usize);
    // 반올림 전 값: 원본 day_from_tape 와 비트까지(합 차례 · 안정 정렬 · Kahan · np.average)
    let mut raw_n = 0;
    for r in fx["raw"].as_array().unwrap() {
        let inp = &r["in"];
        let col = |i: usize| inp[i].as_array().unwrap().clone();
        let t = fpd::Tape {
            sec: col(0).iter().map(|v| v.as_i64().unwrap()).collect(),
            pbin: col(1).iter().map(|v| v.as_i64().unwrap()).collect(),
            buy: col(2).iter().map(|v| v.as_f64().unwrap()).collect(),
            sell: col(3).iter().map(|v| v.as_f64().unwrap()).collect(),
        };
        let (s, c) = fpd::day_from_tape(&f.sp, &t);
        let w = &r["summ"];
        for (k, g) in [("open", s.open), ("close", s.close), ("high", s.high), ("low", s.low), ("buy_qty", s.buy_qty), ("sell_qty", s.sell_qty), ("buy_quote", s.buy_quote), ("sell_quote", s.sell_quote)] {
            assert_eq!(g.to_bits(), w[k].as_f64().unwrap().to_bits(), "{asset} {} {k}: {g} != {}", r["day"], w[k]);
            raw_n += 1;
        }
        let wc = &r["cells"];
        assert_eq!(c.0, wc[0].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect::<Vec<_>>(), "{asset} {} 칸 번호", r["day"]);
        for (j, side) in [&c.1, &c.2].into_iter().enumerate() {
            for (x, y) in side.iter().zip(wc[j + 1].as_array().unwrap()) {
                assert_eq!(x.to_bits(), y.as_f64().unwrap().to_bits(), "{asset} {} 칸 합", r["day"]);
                raw_n += 1;
            }
        }
    }
    for (si, (st, add)) in fx["steps"].as_array().unwrap().iter().zip(fx["adds"].as_array().unwrap()).enumerate() {
        if !add.is_null() {
            let t = rusqlite::Connection::open(&tape).unwrap();
            for r in add["tape"].as_array().unwrap() {
                t.execute("INSERT INTO trade_tape_1s VALUES (?,?,?,?,?)", rusqlite::params![r[0].as_str(), r[1].as_i64(), r[2].as_i64(), r[3].as_f64(), r[4].as_f64()]).unwrap();
            }
            let c = rusqlite::Connection::open(&ctx).unwrap();
            for r in add["oi"].as_array().unwrap() {
                c.execute("INSERT INTO oi_1s VALUES (?,?,?)", rusqlite::params![r[0].as_str(), r[1].as_i64(), r[2].as_f64()]).unwrap();
            }
            for r in add["liq"].as_array().unwrap() {
                c.execute("INSERT INTO liquidations VALUES (?,?,?,?,?,?)", rusqlite::params![r[0].as_str(), r[1].as_i64(), r[2].as_str(), r[3].as_f64(), r[4].as_f64(), r[5].as_f64()]).unwrap();
            }
        }
        let now = st["now"].as_f64().unwrap();
        for c in st["calls"].as_array().unwrap() {
            let a = c["args"].as_array().unwrap();
            let got = match c["kind"].as_str().unwrap() {
                "days_payload" => f.days_payload(now),
                "cells_payload" => f.cells_payload(now, a[0].as_str().unwrap(), a[1].as_str().unwrap(), a[2].as_i64().unwrap()),
                "h4_payload" => f.h4_payload(now),
                "h4_cells_payload" => f.h4_cells_payload(now, a[0].as_str().unwrap(), a[1].as_str().unwrap(), a[2].as_i64().unwrap()),
                k => panic!("{k}"),
            };
            calls += 1;
            match (got, c.get("error")) {
                (Err(e), Some(w)) => {
                    assert_eq!(e, w.as_str().unwrap(), "걸음 {si} {} 오류", c["kind"]);
                    errs += 1;
                }
                (Ok(v), None) => {
                    let mine = rtserve::py::dumps(&v, (", ", ": "), true, None);
                    let want = c["body"].as_str().unwrap();
                    if mine != want {
                        let i = mine.bytes().zip(want.bytes()).position(|(x, y)| x != y).unwrap_or(mine.len().min(want.len()));
                        panic!("걸음 {si} {} {:?}: 글자 {i} 부터 다름\n코어 …{}\n원본 …{}", c["kind"], a, &mine[i.saturating_sub(80)..(i + 80).min(mine.len())], &want[i.saturating_sub(80)..(i + 80).min(want.len())]);
                    }
                    chars += want.len();
                    if let Some(d) = v["cols"]["d"].as_array() {
                        cols_n += d.len() * fpd::COLS.len();
                    }
                }
                (g, w) => panic!("걸음 {si} {}: 코어 {:?} · 원본 오류 {:?}", c["kind"], g.map(|_| "값"), w),
            }
        }
    }
    println!("일봉·4시간 풋프린트 일치({asset}): 걸음 {} · 호출 {calls}(오류 {errs}) · 본문 {chars}글자 · 열 칸 {cols_n} · 반올림 전 값 {raw_n}(비트)", fx["steps"].as_array().unwrap().len());
    let _ = std::fs::remove_dir_all(&dir);
}

/// server.py 손잡이(옮겨 적음 -- 원본 줄은 fpd.rs 머리 주석): parse_row · _iso_day · _iso_h4 · _cols_since.
#[test]
fn handles_match_server() {
    let c = 1.0;
    // footprint_daily.selftest 586~587 과 같은 값
    assert_eq!(fpd::parse_row(Some("abc"), c), 5);
    assert_eq!(fpd::parse_row(Some("0"), c), 1);
    assert_eq!(fpd::parse_row(Some("9999"), c), 500);
    assert_eq!(fpd::parse_row(Some("25"), c), 25);
    assert_eq!(fpd::parse_row(Some("nan"), c), 5);
    assert_eq!(fpd::parse_row(Some("inf"), c), 5);
    assert_eq!(fpd::parse_row(None, c), 5);
    assert_eq!(fpd::parse_row(Some(""), c), 5);
    assert_eq!(fpd::parse_row(Some("2.5"), c), 2, "파이썬 round 짝수");
    assert_eq!(fpd::parse_row(Some(" 1_0 "), c), 10, "float() 은 공백·밑줄을 받는다");
    assert_eq!(fpd::parse_row(Some("0.0015"), 0.0005), 3, "XRP 칸");
    assert_eq!(fpd::iso_day("2026-11-21").as_deref(), Some("2026-11-21"));
    assert_eq!(fpd::iso_day("20261121").as_deref(), Some("2026-11-21"));
    assert_eq!(fpd::iso_day("2026-02-30"), None);
    assert_eq!(fpd::iso_day("2026-1-5"), None);
    assert!(fpd::iso_h4("2026-11-21T04") && fpd::iso_h4("2026-11-21T00"));
    assert!(!fpd::iso_h4("2026-11-21T05") && !fpd::iso_h4("2026-11-21T4") && !fpd::iso_h4("2026-11-21") && !fpd::iso_h4("2026-11-21T24"));
    let full = serde_json::json!({"symbol": "ETHUSDT", "cols": {"d": ["2026-11-19", "2026-11-20", "2026-11-21"], "o": [1.0, 2.0, 3.0]}, "at": 1.0});
    assert_eq!(fpd::cols_since(&full, Some("2026-11-20")), serde_json::json!({"symbol": "ETHUSDT", "cols": {"d": ["2026-11-20", "2026-11-21"], "o": [2.0, 3.0]}, "at": 1.0}));
    assert_eq!(fpd::cols_since(&full, None), full);
    assert_eq!(fpd::h4_key(1795255200), "2026-11-21T08", "10:00 = 08시 칸");
    // rebin 하한 = 파이썬 round(lo × row × CELL, 6): XRP 2627 × 0.0005 = 1.3135000000000001(실수) → 1.3135 (원본 python3 -c 로 확인)
    let x = fpd::Spec::of("xrp").unwrap();
    let v = fpd::rebin(&x, &(vec![2627], vec![1.0], vec![0.0]), 1).unwrap();
    assert_eq!(rtserve::py::dumps(&v, (", ", ": "), true, None), "[1.3135, [1.0], [0.0]]");
    let e = fpd::rebin(&x, &(vec![], vec![], vec![]), 5).unwrap();
    // 🔴엉뚱한 가격 칸(±20억 칸 폭) = 할당 전에 오류(원본 np.zeros MemoryError → 그 요청만 500) -- 죽지 않는다
    assert!(fpd::rebin(&x, &(vec![-2_000_000_000, 2_000_000_000], vec![1.0, 1.0], vec![0.0, 0.0]), 1).unwrap_err().starts_with("MemoryError"));
    assert_eq!(rtserve::py::dumps(&e, (", ", ": "), true, None), "[0, [], []]", "빈 칸 = 정수 0");
    assert_eq!(fpd::h4_key(1795255200 + 2 * 3600), "2026-11-21T12");
}
