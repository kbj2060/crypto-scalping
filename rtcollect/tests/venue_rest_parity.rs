//! OKX·Bybit 체결 테이프 REST 파이썬 일치 시험. 정답지 = tools/gen_venue_rest_fixture.py 가 원본 `assert_ct_val`·`verify_recent`·
//! `okx_minute_fetcher`(+`_pace`)·공용 `backfill_loop` 과 Bybit `verify_recent` 를 가짜 aiohttp 세션·가짜 시계로 돌린 결과(DB 표 + 나간 요청 열 +
//! 대조·ctVal 결과). 여기서는 같은 응답을 **127.0.0.1 가짜 HTTP 서버**가 주고(실제 전송 = ureq) 같은 가짜 시계로 Rust `rest::Rest` 를 돌린다.
mod common;
use rtcollect::rest::{agent, Clock, Rest, BACKFILL_SECONDS};
use rtcollect::store::{open_db, TapeStore};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn row(v: &Value) -> (i64, i64, [f64; 16]) {
    let a = v.as_array().unwrap();
    let mut c = [0.0; 16];
    for (j, x) in c.iter_mut().enumerate() {
        *x = a[j + 2].as_f64().unwrap_or(f64::NAN);
    }
    (a[0].as_i64().unwrap(), a[1].as_i64().unwrap(), c)
}

/// checked_at·done_at = CURRENT_TIMESTAMP -- 모양만
fn ts_shape(_db: &str, t: &str, col: usize, v: &str) -> Option<String> {
    let shaped = v.len() == 19 && v.as_bytes()[4] == b'-' && v.as_bytes()[10] == b' ';
    ((t == "verify_1m" && col == 5 || t == "backfill_1m" && col == 10) && shaped).then(|| "TS".into())
}

#[test]
fn venue_rest_matches_python_collectors() {
    let input: Value = serde_json::from_str(include_str!("fixtures/venue_rest_input.json")).unwrap();
    let golden: Value = serde_json::from_str(include_str!("fixtures/venue_rest_golden.json")).unwrap();
    let (mut n_req, mut n_rows, mut n_cells, mut n_out) = (0, 0, 0, 0);
    for (k, (sc, want)) in input["scenarios"].as_array().unwrap().iter().zip(golden["scenarios"].as_array().unwrap()).enumerate() {
        let (name, venue, sym) = (sc["name"].as_str().unwrap(), sc["venue"].as_str().unwrap(), sc["symbol"].as_str().unwrap());
        let (base, served) = common::serve(sc["responses"].as_object().unwrap());
        assert!(base.starts_with("http://127.0.0.1:"), "가짜 서버만 -- {base}");
        let dir = common::tmpdir(&format!("vrest_{k}"));
        let db = dir.join(if venue == "bybit" { "bybit_tape.sqlite" } else { "okx_tape.sqlite" });
        let con = open_db(&db).unwrap();
        let mut store = TapeStore::new(con.clone(), sym, rtcollect::okx::bucket(sym)).unwrap();
        for op in sc["setup"].as_array().unwrap() {
            match op[0].as_str().unwrap() {
                "write" => assert!(store.write(op[1].as_array().unwrap().iter().map(row).collect()) > 0),
                "gap" => store.record_gap(op[1].as_i64().unwrap(), op[2].as_i64().unwrap(), op[3].as_str().unwrap()),
                k => panic!("{k}"),
            }
        }
        let clock = Arc::new(Clock::fake(sc["mono0"].as_f64().unwrap(), sc["wall0"].as_f64().unwrap()));
        let t = Duration::from_millis(300);
        let mut outcome: Vec<String> = vec![];
        let mut sent = vec![];
        if venue == "ctval" {
            let url = format!("{base}/api/v5/public/instruments?instType=SWAP&instId={sym}");
            for _ in sc["steps"].as_array().unwrap() {
                outcome.push(match rtcollect::rest_venue::ct_val_of(&agent(t), &url) {
                    Err(_) => "fail-open".into(),
                    Ok((got, _)) if (got - rtcollect::okx::ct_val(sym).unwrap()).abs() > 1e-12 => "exit".into(),
                    Ok(_) => "pass".into(),
                });
            }
        } else {
            let mut r = if venue == "okx" {
                Rest::okx(&db, sym, &base, clock.clone(), Arc::new(Mutex::new(0.0)), t).unwrap()
            } else {
                Rest::bybit(&db, sym, &format!("{base}/v5/market/kline"), clock.clone(), t).unwrap()
            };
            for step in sc["steps"].as_array().unwrap() {
                if step == "cycle" {
                    clock.sleep(BACKFILL_SECONDS);
                    r.backfill_cycle();
                } else {
                    outcome.push(if r.verify_recent().is_ok() { "ok".into() } else { "raised".into() });
                }
            }
            sent = r.sent.clone();
        }
        // 결과(대조 예외 · ctVal 판정): 파이썬 «raised <이름>» 은 이름까지는 안 맞춘다
        let want_out: Vec<String> = want["outcome"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().split(' ').next().unwrap().to_string()).collect();
        assert_eq!(outcome, want_out, "{name} 결과");
        n_out += outcome.len();
        // 요청 열: 시각 비트까지(가짜 시계가 있는 것만) · 서버가 받은 것
        let want_req = want["requests"].as_array().unwrap();
        if venue != "ctval" {
            for (i, (g, w)) in sent.iter().zip(want_req).enumerate() {
                assert_eq!((g.0.to_bits(), g.1.to_bits(), g.2.as_str()), (w[0].as_f64().unwrap().to_bits(), w[1].as_f64().unwrap().to_bits(), w[2].as_str().unwrap()),
                           "{name} 요청 {i}: rust {g:?} != python {w}");
            }
            assert_eq!(sent.len(), want_req.len(), "{name} 요청 수");
        }
        let wire: Vec<Value> = served.lock().unwrap().iter().map(|t| Value::from(t.as_str())).collect();
        assert_eq!(wire, want_req.iter().map(|w| w[2].clone()).collect::<Vec<_>>(), "{name} 서버가 받은 요청");
        n_req += want_req.len();
        drop((store, con));
        let (_, rows, cells) = common::compare(&dir, &want["db"], &[], Some(ts_shape));
        eprintln!("{name}: 요청 {} · 결과 {outcome:?} · 행 {rows} · 칸 {cells}", want_req.len());
        (n_rows, n_cells) = (n_rows + rows, n_cells + cells);
        let _ = std::fs::remove_dir_all(&dir);
    }
    eprintln!("OKX·Bybit REST 일치: 시나리오 {} · 요청 {n_req} · 결과 {n_out} · 행 {n_rows} · 칸 {n_cells}", input["scenarios"].as_array().unwrap().len());
}
