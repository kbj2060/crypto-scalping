//! hl_bt 수신 고리 지연(11-28): 맥락 SQLite 쓰기·시 파일 gzip 은 고리 밖이다 -- 다른 연결이 DB 를 1.5초 잠가도 메시지 하나 처리가 0.3초를 넘지 않는다.
//! 시 넘김을 가로질러 .bt(지난 시 = .gz) 바이트와 맥락 행이 그대로인지도 본다. 양성 대조: 고리 안 2초 잠 변이 → 실패.
mod common;
use rtcollect::{bt::HourFile, hl_ctx};
use std::time::{Duration, Instant};

const H: i64 = 1_795_248_000_000; // UTC 시 경계

fn bbo(ts: i64, i: i64) -> String {
    format!(r#"{{"channel":"bbo","data":{{"coin":"ETH","time":{ts},"bbo":[{{"px":"100.{i:03}","sz":"1.5","n":1}},{{"px":"101","sz":"2","n":1}}]}}}}"#)
}
fn ctx(i: i64) -> String {
    format!(r#"{{"channel":"activeAssetCtx","data":{{"coin":"ETH","ctx":{{"openInterest":"{i}","markPx":"100"}}}}}}"#)
}
fn row(ts: i64, i: i64) -> Vec<u8> {
    let mut r = ts.to_le_bytes().to_vec();
    r.extend((100.0 + i as f64 / 1000.0).to_le_bytes());
    r.extend(1.5f32.to_le_bytes());
    r.extend(101.0f64.to_le_bytes());
    r.extend(2.0f32.to_le_bytes());
    r
}
fn header(hour: i64, first: i64) -> Vec<u8> {
    let mut h = b"BTKR".to_vec();
    h.extend(1u16.to_le_bytes());
    h.extend(32u16.to_le_bytes());
    h.extend(hour.to_le_bytes());
    h.extend((first as u64).to_le_bytes());
    h.extend([0u8; 8]);
    h
}

#[test]
fn hl_bt_loop_not_blocked_by_db_lock_or_gzip_across_hour() {
    let dir = common::tmpdir("hl_latency");
    let db = dir.join("hl_ctx.sqlite");
    drop(hl_ctx::open_db(&db).unwrap());
    let mut c = hl_ctx::HlCtx::new(&db, "ETH", 0.0);
    c.bt = Some(HourFile::new(&dir.join("bt"), "ETH", 0.0).unwrap());
    // 맥락 10행을 모아 둔다(30초 전 -- 아직 안 씀)
    for i in 0..10 {
        c.on_text(1.0, 1.0, &ctx(i)).unwrap();
    }
    // 다른 연결이 쓰기 잠금을 1.5초 쥔다
    let (tx, rx) = std::sync::mpsc::channel();
    let p = db.clone();
    let lock = std::thread::spawn(move || {
        let con = rusqlite::Connection::open(&p).unwrap();
        con.execute_batch("BEGIN IMMEDIATE").unwrap();
        tx.send(()).unwrap();
        std::thread::sleep(Duration::from_millis(1500));
        con.execute_batch("COMMIT").unwrap();
    });
    rx.recv().unwrap();
    // 30초 지남 → 첫 메시지가 맥락 쓰기를 부른다 · bbo 는 시 경계를 넘는다(넘는 순간 지난 시 압축)
    let mut worst = Duration::ZERO;
    let mut want_old = header(H - 3_600_000, H - 2000);
    let mut want_new = header(H, H);
    for i in 0..40 {
        let ts = H - 2000 + 100 * i;
        let t0 = Instant::now();
        c.on_text(31.0 + i as f64 * 0.01, 31.0, &bbo(ts, i)).unwrap();
        worst = worst.max(t0.elapsed());
        if ts < H { want_old.extend(row(ts, i)) } else { want_new.extend(row(ts, i)) }
    }
    eprintln!("메시지 하나 처리 최대 {worst:?}(DB 잠금 1.5초 · 시 넘김 압축 포함)");
    assert!(worst < Duration::from_millis(300), "수신 고리가 {worst:?} 막혔다");
    lock.join().unwrap();
    c.on_disconnect();
    c.finish();
    let raw = std::fs::read(dir.join("bt/ETH/2026-11-21T07.bt.gz")).expect("지난 시 .gz");
    let mut old = vec![];
    std::io::Read::read_to_end(&mut flate2::read::GzDecoder::new(&raw[..]), &mut old).unwrap();
    assert_eq!(old, want_old, "지난 시 .bt 바이트");
    assert!(!dir.join("bt/ETH/2026-11-21T07.bt").exists(), "압축 뒤 원본은 지운다");
    assert_eq!(std::fs::read(dir.join("bt/ETH/2026-11-21T08.bt")).unwrap(), want_new, "이번 시 .bt 바이트");
    let con = rusqlite::Connection::open(&db).unwrap();
    let ois: Vec<f64> = con.prepare("SELECT open_interest FROM hl_asset_ctx ORDER BY rowid").unwrap().query_map([], |r| r.get(0)).unwrap().map(|x| x.unwrap()).collect();
    assert_eq!(ois, (0..10).map(|i| i as f64).collect::<Vec<_>>(), "잠금이 풀린 뒤 맥락 10행이 차례대로");
    let _ = std::fs::remove_dir_all(&dir);
}
