//! 바이낸스 테이프 REST 검증·복구 파이썬 일치 시험. 정답지 = tools/gen_rest_fixture.py 가 원본 `backfill_loop`·`binance_minute_fetcher`·
//! `verify_recent` 를 가짜 aiohttp 세션·가짜 시계로 돌린 결과(DB 표 전체 + 나간 요청 열). 여기서는 같은 응답을 **127.0.0.1 가짜 HTTP 서버**가
//! 주고(실제 전송 = ureq), 같은 가짜 시계로 Rust `rest::Rest` 를 돌려 ① 요청 열(단조·벽 시각 비트까지, 경로?쿼리) ② 서버가 받은 요청
//! ③ 결과 DB(행·저장 형식·REAL 비트·스키마 글자 -- checked_at·done_at 은 CURRENT_TIMESTAMP 라 모양만)를 비교한다.
mod common;
use rtcollect::rest::{Clock, Rest, BACKFILL_SECONDS};
use rtcollect::store::{open_db, TapeStore};
use rtcollect::tape::bucket;
use serde_json::Value;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const INPUT: &str = include_str!("fixtures/rest_input.json");
const GOLDEN: &str = include_str!("fixtures/rest_golden.json");

/// 가짜 바이낸스 REST: 요청 대상(경로?쿼리)마다 정해 둔 응답을 차례로. `timeout` = 1.5초 동안 답하지 않는다(클라이언트 시한 0.3초).
fn serve(responses: &serde_json::Map<String, Value>) -> (String, Arc<Mutex<Vec<String>>>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    let q: Arc<Mutex<HashMap<String, Vec<Value>>>> =
        Arc::new(Mutex::new(responses.iter().map(|(k, v)| (k.clone(), v.as_array().unwrap().clone())).collect()));
    let log = Arc::new(Mutex::new(vec![]));
    let log2 = log.clone();
    std::thread::spawn(move || {
        for s in l.incoming() {
            let (mut s, q, log) = (s.unwrap(), q.clone(), log2.clone());
            std::thread::spawn(move || {
                let mut buf = vec![];
                let mut b = [0u8; 4096];
                while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    match s.read(&mut b) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => buf.extend_from_slice(&b[..n]),
                    }
                }
                let target = String::from_utf8_lossy(&buf).split_whitespace().nth(1).unwrap().to_string();
                log.lock().unwrap().push(target.clone());
                let r = q.lock().unwrap().get_mut(&target).filter(|v| !v.is_empty()).map(|v| v.remove(0));
                let Some(r) = r else {
                    let _ = s.write_all(b"HTTP/1.1 599 NO FIXTURE\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                    return;
                };
                if r["timeout"] == true {
                    std::thread::sleep(Duration::from_millis(1500));
                    return;
                }
                let body = r["body"].to_string();
                let mut h = format!("HTTP/1.1 {} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n", r["status"], body.len());
                for (k, v) in r["headers"].as_object().into_iter().flatten() {
                    h += &format!("{k}: {}\r\n", v.as_str().unwrap());
                }
                let _ = s.write_all(format!("{h}\r\n{body}").as_bytes());
            });
        }
    });
    (base, log)
}

fn row(v: &Value) -> (i64, i64, [f64; 16]) {
    let a = v.as_array().unwrap();
    let mut c = [0.0; 16];
    for (j, x) in c.iter_mut().enumerate() {
        *x = a[j + 2].as_f64().unwrap_or(f64::NAN);
    }
    (a[0].as_i64().unwrap(), a[1].as_i64().unwrap(), c)
}

/// checked_at·done_at = CURRENT_TIMESTAMP(실행 시각) -- «YYYY-MM-DD HH:MM:SS» 모양이면 같은 것으로.
fn ts_shape(_db: &str, t: &str, col: usize, v: &str) -> Option<String> {
    let shaped = v.len() == 19 && v.bytes().enumerate().all(|(i, b)| match i {
        4 | 7 => b == b'-',
        10 => b == b' ',
        13 | 16 => b == b':',
        _ => b.is_ascii_digit(),
    });
    ((t == "verify_1m" && col == 5 || t == "backfill_1m" && col == 10) && shaped).then(|| "TS".into())
}

#[test]
fn rest_rows_match_python_collector() {
    let input: Value = serde_json::from_str(INPUT).unwrap();
    let golden: Value = serde_json::from_str(GOLDEN).unwrap();
    let (mut n_req, mut n_rows, mut n_cells) = (0, 0, 0);
    for (k, (sc, want)) in input["scenarios"].as_array().unwrap().iter().zip(golden["scenarios"].as_array().unwrap()).enumerate() {
        let name = sc["name"].as_str().unwrap();
        let (base, served) = serve(sc["responses"].as_object().unwrap());
        assert!(base.starts_with("http://127.0.0.1:"), "가짜 서버만 -- {base}");
        let (sym, spot) = (sc["symbol"].as_str().unwrap(), sc["market"] == "spot");
        let dir = common::tmpdir(&format!("rest_{k}"));
        let file = if spot { "binance_spot_tape.sqlite" } else { "binance_tape.sqlite" };
        let db = dir.join(file);
        let con = open_db(&db).unwrap();
        let mut store = TapeStore::new(con.clone(), sym, bucket(sym)).unwrap();
        for op in sc["setup"].as_array().unwrap() {
            match op[0].as_str().unwrap() {
                "write" => assert!(store.write(op[1].as_array().unwrap().iter().map(row).collect()) > 0),
                "gap" => store.record_gap(op[1].as_i64().unwrap(), op[2].as_i64().unwrap(), op[3].as_str().unwrap()),
                "verify" => {
                    con.execute("INSERT INTO verify_1m VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)",
                        rusqlite::params![sym, op[1].as_i64(), op[2].as_f64(), op[3].as_f64(), op[4].as_f64()]).unwrap();
                }
                "backfill" => {
                    con.execute("INSERT INTO backfill_1m VALUES (?,?,'setup',0,0.0,0.0,0.0,?,?,'setup',CURRENT_TIMESTAMP)",
                        rusqlite::params![sym, op[1].as_i64(), op[2].as_i64(), op[3].as_i64()]).unwrap();
                }
                k => panic!("{k}"),
            }
        }
        let clock = Arc::new(Clock::fake(sc["mono0"].as_f64().unwrap(), sc["wall0"].as_f64().unwrap()));
        let root = if spot { "/api/v3" } else { "/fapi/v1" };
        let ban = dir.join("binance_ban_until");
        let mut r = Rest::new(&db, sym, &format!("{base}{root}/klines"), &format!("{base}{root}/aggTrades"), clock.clone(),
                              Arc::new(Mutex::new(0.0)), ban.clone(), Duration::from_millis(300)).unwrap();
        for step in sc["steps"].as_array().unwrap() {
            if step == "cycle" {
                clock.sleep(BACKFILL_SECONDS);
                r.backfill_cycle();
            } else {
                let _ = r.verify_recent();
            }
        }
        // ① 요청 열: 시각 비트까지
        let want_req = want["requests"].as_array().unwrap();
        let got_req: Vec<Value> = r.sent.iter().map(|(m, w, t)| serde_json::json!([m, w, t])).collect();
        for (i, (g, w)) in got_req.iter().zip(want_req).enumerate() {
            assert_eq!((g[0].as_f64().unwrap().to_bits(), g[1].as_f64().unwrap().to_bits(), &g[2]),
                       (w[0].as_f64().unwrap().to_bits(), w[1].as_f64().unwrap().to_bits(), &w[2]), "{name} 요청 {i}: rust {g} != python {w}");
        }
        assert_eq!(got_req.len(), want_req.len(), "{name} 요청 수");
        // ② 서버가 실제로 받은 것(재시도·덧붙은 요청 없음)
        let wire: Vec<Value> = served.lock().unwrap().iter().map(|t| Value::from(t.as_str())).collect();
        assert_eq!(wire, want_req.iter().map(|w| w[2].clone()).collect::<Vec<_>>(), "{name} 서버가 받은 요청");
        n_req += want_req.len();
        // Rust 만: 418·429 는 공용 밴 파일에 적는다(scripts/binance_ban_guard.py 규칙) -- «오류» 의 418 은 banned until 그대로
        if name == "오류" {
            let until: f64 = std::fs::read_to_string(&ban).unwrap().trim().parse().unwrap();
            assert_eq!(until, sc["wall0"].as_f64().unwrap() + 270.0, "418 본문의 banned until");
            // 밴 중이면 보내지 않는다: 후보를 하나 만들고 밴을 걸면 요청 0
            store.record_gap(1_763_694_000_000 - 900_000, 1_763_694_000_000 - 880_000, "ws_reconnect");
            std::fs::write(&ban, format!("{:.3}\n", clock.wall() + 100.0)).unwrap();
            let before = (r.sent.len(), served.lock().unwrap().len());
            clock.sleep(BACKFILL_SECONDS);
            r.backfill_cycle();
            assert_eq!((r.sent.len(), served.lock().unwrap().len()), before, "밴 중인데 요청이 나갔다");
            con.execute("DELETE FROM gaps WHERE from_ms = ?", [1_763_694_000_000i64 - 900_000]).unwrap();
        }
        drop((r, store, con));
        let (_, rows, cells) = common::compare(&dir, &want["db"], &[], Some(ts_shape));
        eprintln!("{name}: 요청 {} · 행 {rows} · 칸 {cells}", want_req.len());
        (n_rows, n_cells) = (n_rows + rows, n_cells + cells);
        let _ = std::fs::remove_dir_all(&dir);
    }
    eprintln!("REST 일치: 시나리오 {} · 요청 {n_req} · 행 {n_rows} · 칸 {n_cells}", input["scenarios"].as_array().unwrap().len());
}
