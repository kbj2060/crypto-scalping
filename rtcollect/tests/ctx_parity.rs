//! 맥락 수집기(OKX 맥락 · 바이낸스 맥락 · xyz100) 파이썬 일치 시험. 정답지 = tools/gen_ctx_fixture.py 가 **원본 수집 루프**를 가짜 aiohttp·가짜 시계로
//! 돌려 만든 SQLite 의 덤프 + 보낸 글자(구독·ping). 같은 사건 열을 Rust 에 넣어 행 단위·스키마 글자까지 비교한다(tests/common).
mod common;
use serde_json::Value;

fn load(input: &str, golden: &str) -> (Value, Value) {
    (serde_json::from_str(input).unwrap(), serde_json::from_str(golden).unwrap())
}

#[test]
fn okx_ctx_rows_match_python_collector() {
    let (inp, gold) = load(include_str!("fixtures/okx_ctx_input.json"), include_str!("fixtures/okx_ctx_golden.json"));
    let dir = common::tmpdir("okx_ctx");
    let subs = rtcollect::okx_ctx::replay(inp["input"].as_array().unwrap(), &dir.join("okx_ctx.sqlite")).unwrap();
    // 보낸 글자: 접속마다 구독 하나(원본 send_json = json.dumps 기본 모양)
    let mut n_sent = 0;
    for (sub, sent) in subs.iter().zip(gold["extra"].as_array().unwrap()) {
        for s in sent.as_array().unwrap().iter().map(|s| s.as_str().unwrap()).filter(|s| !s.starts_with("<connect")) {
            assert_eq!(s, sub, "구독 글자");
            n_sent += 1;
        }
    }
    let (t, r, c) = common::compare(&dir, &gold["db"], &[], None);
    eprintln!("okx_ctx 일치: 표 {t} · 행 {r} · 칸 {c} · 보낸 구독 {n_sent}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// 바이낸스 맥락: 웹소켓 몫(마크·강제청산 사건 열) + OI REST 폴링(127.0.0.1 가짜 서버 · 가짜 시계) → 한 번 쓰기. 원본 정답지도 같은 순서
/// (웹소켓 피드 차례로 → OI 시나리오 차례로 → `Pending.flush`) -- 표는 PK·INSERT OR IGNORE(먼저 온 값) 또는 한 원천(청산)이라 순서가 값을 안 바꾼다.
#[test]
fn binance_ctx_rows_match_python_collector() {
    use rtcollect::binance_ctx as bc;
    use rtcollect::rest::Clock;
    use std::sync::{Arc, Mutex};
    let (inp, gold) = load(include_str!("fixtures/binance_ctx_input.json"), include_str!("fixtures/binance_ctx_golden.json"));
    let inp = &inp["input"];
    let dir = common::tmpdir("binance_ctx");
    let db = dir.join("binance_ctx.sqlite");
    let con = bc::open_db(&db).unwrap();
    let p: bc::Shared = Arc::new(Mutex::new(bc::Pending::default()));
    let slow: Vec<String> = inp["slow"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
    bc::replay_ws(inp["ws"].as_array().unwrap(), &slow, &p);
    let (mut n_req, mut n_sc) = (0, 0);
    for (k, (sc, want)) in inp["oi"].as_array().unwrap().iter().zip(gold["extra"]["requests"].as_array().unwrap()).enumerate() {
        let sym = sc["symbol"].as_str().unwrap();
        let mut m = serde_json::Map::new();
        m.insert(format!("/fapi/v1/openInterest?symbol={sym}"), sc["responses"].clone());
        let (base, served) = common::serve(&m);
        let ban = dir.join(format!("ban_{sym}"));
        let clock = Arc::new(Clock::fake(50_000.0, 1_763_683_200.0 - 40.0));
        let mut r = bc::OiPoll::new(sym, sc["period"].as_f64().unwrap(), &format!("{base}/fapi/v1/openInterest"), clock, ban.clone(), p.clone(),
                                    std::time::Duration::from_millis(300));
        let want = want.as_array().unwrap();
        while r.sent.len() < want.len() {
            r.step();
        }
        for (i, (g, w)) in r.sent.iter().zip(want).enumerate() {
            assert_eq!((g.0.to_bits(), g.1.to_bits(), g.2.as_str()), (w[0].as_f64().unwrap().to_bits(), w[1].as_f64().unwrap().to_bits(), w[2].as_str().unwrap()),
                       "{} 요청 {i}: rust {g:?} != python {w}", sc["name"]);
        }
        assert_eq!(served.lock().unwrap().len(), want.len(), "{} 서버가 받은 요청", sc["name"]);
        // 밴 파일(binance_ban_guard.note 규칙) -- 원본과 같은 글자
        assert_eq!(std::fs::read_to_string(&ban).ok().as_deref(), gold["extra"]["ban"][k].as_str(), "{} 밴 파일", sc["name"]);
        n_req += want.len();
        n_sc += 1;
    }
    let wrote = bc::flush(&con, &p);
    drop(con);
    let (t, rows, c) = common::compare(&dir, &gold["db"], &[], None);
    eprintln!("binance_ctx 일치: 표 {t} · 행 {rows}(한 번에 {wrote}) · 칸 {c} · OI 시나리오 {n_sc} · 요청 {n_req}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// xyz100: 웹소켓 사건 열(REST 메움 끔 = Rust 기본) + REST 메움 시나리오(127.0.0.1 가짜 서버 -- 보낸 본문 글자·결과 행).
#[test]
fn xyz_rows_match_python_collector() {
    use rtcollect::xyz;
    let (inp, gold) = load(include_str!("fixtures/xyz_input.json"), include_str!("fixtures/xyz_golden.json"));
    let inp = &inp["input"];
    let dir = common::tmpdir("xyz");
    let coins: Vec<String> = inp["coins"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
    let sent = xyz::replay(inp["events"].as_array().unwrap(), &coins, &dir.join("hl_xyz.sqlite")).unwrap();
    let want: Vec<&str> = gold["extra"]["sent"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).filter(|s| !s.starts_with("<connect")).collect();
    assert_eq!(sent, want, "보낸 글자(구독·ping)");
    let mut n_req = 0;
    for (k, (sc, want)) in inp["rest"].as_array().unwrap().iter().zip(gold["extra"]["rest"].as_array().unwrap()).enumerate() {
        let mut m = serde_json::Map::new();
        m.insert("/info".into(), sc["responses"].clone());
        let (base, served) = common::serve(&m);
        let c = xyz::XyzCollector::new(xyz::open_db(&dir.join(format!("rest_{k}.sqlite"))).unwrap(), &coins);
        let setup: Vec<xyz::Bar> = sc["setup"].as_array().unwrap().iter()
            .map(|r| { let p = r[2].as_f64().unwrap(); (r[0].clone(), r[1].as_i64().unwrap(), p, p, p, p, 1.0, 1) }).collect();
        xyz::write(&c.con, &setup, &[]).unwrap();
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(std::time::Duration::from_millis(300))).http_status_as_error(false).build().into();
        let bodies = c.backfill(&agent, &format!("{base}/info"), sc["wall"].as_f64().unwrap());
        assert_eq!(bodies, want.as_array().unwrap().iter().map(|b| b.as_str().unwrap().to_string()).collect::<Vec<_>>(), "{} 보낸 본문", sc["name"]);
        assert_eq!(served.lock().unwrap().len(), bodies.len(), "{} 서버가 받은 요청", sc["name"]);
        n_req += bodies.len();
    }
    let (t, r, c) = common::compare(&dir, &gold["db"], &[], None);
    eprintln!("xyz 일치: 파일 {} · 표 {t} · 행 {r} · 칸 {c} · 보낸 글자 {} · REST 요청 {n_req}", gold["db"].as_object().unwrap().len(), sent.len());
    let _ = std::fs::remove_dir_all(&dir);
}

/// 하이퍼리퀴드 자산 맥락(CTX_ONLY): 구독·ping 글자 + 행(recv_ms 는 정답지와 같은 가짜 시계).
#[test]
fn hl_ctx_rows_match_python_collector() {
    let (inp, gold) = load(include_str!("fixtures/hl_ctx_input.json"), include_str!("fixtures/hl_ctx_golden.json"));
    let dir = common::tmpdir("hl_ctx");
    let sent = rtcollect::hl_ctx::replay(inp["input"].as_array().unwrap(), &dir.join("hl_ctx.sqlite"), None).unwrap();
    let want: Vec<&str> = gold["extra"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).filter(|s| !s.starts_with("<connect")).collect();
    assert_eq!(sent, want, "보낸 글자");
    let (t, r, c) = common::compare(&dir, &gold["db"], &[], None);
    eprintln!("hl_ctx 일치: 표 {t} · 행 {r} · 칸 {c} · 보낸 글자 {}", sent.len());
    let _ = std::fs::remove_dir_all(&dir);
}

/// 하이퍼리퀴드 호가 + 맥락 전체(hl_bt): 구독·ping 글자 · 맥락 행 · `.bt` 시각 파일 바이트(지난 시각은 .gz -- 푼 내용)가 원본 run() 과 같다.
#[test]
fn hl_bt_files_match_python_collector() {
    let (inp, gold) = load(include_str!("fixtures/hl_bt_input.json"), include_str!("fixtures/hl_bt_golden.json"));
    let dir = common::tmpdir("hl_bt");
    let sent = rtcollect::hl_ctx::replay(inp["input"].as_array().unwrap(), &dir.join("hl_ctx.sqlite"), Some(&dir.join("bt"))).unwrap();
    let want: Vec<&str> = gold["extra"]["sent"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).filter(|s| !s.starts_with("<connect")).collect();
    assert_eq!(sent, want, "보낸 글자");
    let (t, r, c) = common::compare(&dir, &gold["db"], &[], None);
    let mut got = std::collections::BTreeMap::new();
    for e in std::fs::read_dir(dir.join("bt/ETH")).unwrap() {
        let p = e.unwrap().path();
        let raw = std::fs::read(&p).unwrap();
        let b = if p.extension().is_some_and(|x| x == "gz") {
            let mut v = vec![];
            std::io::Read::read_to_end(&mut flate2::read::GzDecoder::new(&raw[..]), &mut v).unwrap();
            v
        } else {
            raw
        };
        got.insert(format!("ETH/{}", p.file_name().unwrap().to_string_lossy()), b.iter().map(|x| format!("{x:02x}")).collect::<String>());
    }
    let want_bt: std::collections::BTreeMap<String, String> = gold["extra"]["bt"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string())).collect();
    assert_eq!(got.keys().collect::<Vec<_>>(), want_bt.keys().collect::<Vec<_>>(), ".bt 파일 이름");
    for (k, v) in &want_bt {
        assert_eq!(got[k].len(), v.len(), "{k} 길이");
        if let Some(i) = (0..v.len() / 64).find(|i| got[k][i * 64..(i + 1) * 64] != v[i * 64..(i + 1) * 64]) {
            panic!("{k} 행 {} 다름(0 = 헤더):\n rust {}\n py   {}", i, &got[k][i * 64..(i + 1) * 64], &v[i * 64..(i + 1) * 64]);
        }
    }
    let n_rows: usize = want_bt.values().map(|v| v.len() / 64 - 1).sum();
    eprintln!("hl_bt 일치: .bt 파일 {} · 행 {n_rows} · 맥락 표 {t} · 행 {r} · 칸 {c} · 보낸 글자 {}", want_bt.len(), sent.len());
    let _ = std::fs::remove_dir_all(&dir);
}
