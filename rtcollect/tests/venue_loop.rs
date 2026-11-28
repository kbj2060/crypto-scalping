//! 실행 파일(rtcollect) 접속 루프 시험 -- Bybit 청산 · Bybit 체결 테이프 · OKX 체결 테이프. 거래소 대신 127.0.0.1 가짜 웹소켓 서버.
//! 종류마다 «접속 → 구독 확인 → 메시지 → 끊김 → 재접속 → 침묵 → 재접속 → 메시지 → SIGTERM» 을 **원본 상수 그대로**(ping 20초 · 침묵 60/25초 ·
//! 재접속 대기 1초 지수/3초) 돌려
//! ① 구독 글자가 원본(json.dumps 기본 모양)과 같다 ② ping 이 원본 주기·모양으로 나간다(Bybit 글자 {"op": "ping"} 20초 · OKX PING 프레임 20초)
//! ③ 끊김·침묵 뒤 다시 붙는 시각이 원본 규칙대로다 ④ 만든 DB 가 같은 사건 열을 라이브러리 `replay` 에 넣은 DB 와 같다
//!    (벽시계에서 오는 칸 -- 청산 recv_ms · 끊긴 구간의 시각 -- 은 두 쪽이 원래 다르니 빼고, 끊긴 구간은 이유 글자만).
//! 침묵 단계가 원본 상수라 시험 셋이 나란히 약 70초 걸린다.
use futures_util::{SinkExt, StreamExt};
use rusqlite::{types::ValueRef, Connection, OpenFlags};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_async, tungstenite::Message};

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rtcollect_vloop_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// 표들을 글자로(REAL 은 비트). `skip` = 뺄 열 번호.
fn dump(db: &Path, tables: &[(&str, &str, &[usize])]) -> Vec<String> {
    let con = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let mut out = vec![];
    for (t, o, skip) in tables {
        let mut st = con.prepare(&format!("SELECT * FROM {t} ORDER BY {o}")).unwrap();
        let n = st.column_count();
        let mut rows = st.query([]).unwrap();
        while let Some(r) = rows.next().unwrap() {
            let cells: Vec<String> = (0..n)
                .filter(|j| !skip.contains(j))
                .map(|j| match r.get_ref(j).unwrap() {
                    ValueRef::Real(f) => format!("{:016x}R", f.to_bits()),
                    ValueRef::Integer(i) => format!("{i}I"),
                    ValueRef::Text(s) => format!("{:?}", String::from_utf8_lossy(s)),
                    ValueRef::Null => "NULL".into(),
                    ValueRef::Blob(_) => "BLOB".into(),
                })
                .collect();
            out.push(format!("{t}|{}", cells.join("|")));
        }
    }
    out
}

enum End {
    Close,
    /// 받으며(ping 을 적으며) 이만큼 기다린 뒤 닫는다 -- 클라이언트가 먼저 끊으면 거기서 끝.
    Read(Duration),
    /// 읽지도 쓰지도 않는다(pong 도 안 간다) -- 죽은 연결 흉내.
    Mute(Duration),
}

struct Conn {
    msgs: Vec<String>,
    end: End,
}

#[derive(Default)]
struct Seen {
    /// (연결 번호, 그 연결에서 마지막 메시지를 보낸 뒤 (단조 초, 벽시계 초), 받은 것: 글자 또는 "PING")
    frames: Vec<(usize, [f64; 2], String)>,
    /// 연결마다 (받아들인 시각, 메시지를 다 보낸 시각) -- 시험 시작 뒤 (단조 초, 벽시계 초)
    conns: Vec<([f64; 2], [f64; 2])>,
}

/// 연결마다 정해 둔 대로: 첫 프레임(구독)을 적고 → 메시지를 3ms 간격으로 → End. 마지막 연결 뒤 추가 연결은 받아서 쥐고만 있는다.
async fn serve(script: Vec<Conn>) -> (u16, Arc<Mutex<Seen>>) {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = l.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(Seen::default()));
    let s2 = seen.clone();
    let t0 = Instant::now();
    let w0 = rtcollect::rest::wall_s();
    // 🔴두 시계를 같이 잰다: 이 개발기(WSL)는 벽시계가 약 32초마다 2.4초씩 뒤로 뛴다(실측) -- 청산 수집기는 원본처럼 벽시계로 잰다.
    let now2 = move || [t0.elapsed().as_secs_f64(), rtcollect::rest::wall_s() - w0];
    tokio::spawn(async move {
        for (k, c) in script.into_iter().enumerate() {
            let (s, _) = l.accept().await.unwrap();
            let seen = s2.clone();
            let acc = now2();
            tokio::spawn(async move {
                let mut ws = accept_async(s).await.unwrap();
                if let Some(Ok(Message::Text(t))) = ws.next().await {
                    seen.lock().unwrap().frames.push((k, [0.0; 2], format!("sub:{}", t.as_str())));
                }
                for m in &c.msgs {
                    if let Some(ms) = m.strip_prefix("SLEEP:") {
                        tokio::time::sleep(Duration::from_millis(ms.parse().unwrap())).await; // 각본의 쉼(xyz 플러시 2초)
                        continue;
                    }
                    ws.send(Message::text(m.clone())).await.unwrap();
                    tokio::time::sleep(Duration::from_millis(3)).await;
                }
                let (sent, s2) = (Instant::now(), now2());
                seen.lock().unwrap().conns.push((acc, s2));
                match c.end {
                    End::Close => {
                        let _ = ws.close(None).await;
                    }
                    End::Read(d) => {
                        while let Ok(Some(Ok(f))) = tokio::time::timeout_at((sent + d).into(), ws.next()).await {
                            let what = match f {
                                Message::Text(t) => t.as_str().to_string(),
                                Message::Ping(_) => "PING".into(),
                                m => format!("{m:?}"),
                            };
                            let n = now2();
                            seen.lock().unwrap().frames.push((k, [n[0] - s2[0], n[1] - s2[1]], what));
                        }
                        let _ = ws.close(None).await;
                    }
                    End::Mute(d) => {
                        tokio::time::sleep(d).await;
                        drop(ws);
                    }
                }
            });
        }
        loop {
            let (s, _) = l.accept().await.unwrap();
            tokio::spawn(async move {
                let _ws = accept_async(s).await;
                tokio::time::sleep(Duration::from_secs(600)).await;
            });
        }
    });
    (port, seen)
}

/// 실행 파일을 띄워 마지막 연결의 메시지가 다 간 뒤 1.5초에 SIGTERM, 종료코드 0 확인.
async fn run_bin(envs: &[(&str, String)], seen: &Arc<Mutex<Seen>>, n_conns: usize, limit_s: u64) -> String {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rtcollect"));
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let mut child = cmd.stdout(Stdio::null()).stderr(Stdio::piped()).spawn().unwrap();
    let t0 = Instant::now();
    while seen.lock().unwrap().conns.len() < n_conns {
        assert!(t0.elapsed() < Duration::from_secs(limit_s), "연결 {n_conns}개가 {limit_s}초 안에 안 왔다");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert!(Command::new("kill").args(["-TERM", &child.id().to_string()]).status().unwrap().success());
    let (st, err) = tokio::task::spawn_blocking(move || {
        let st = child.wait().unwrap();
        let mut e = String::new();
        std::io::Read::read_to_string(child.stderr.as_mut().unwrap(), &mut e).unwrap();
        (st, e)
    })
    .await
    .unwrap();
    assert_eq!(st.code(), Some(0), "SIGTERM 은 0 으로 끝난다:\n{err}");
    err
}

/// clock 0 = 단조 · 1 = 벽시계
fn pings(seen: &Seen, k: usize, what: &str, clock: usize) -> Vec<f64> {
    seen.frames.iter().filter(|f| f.0 == k && f.2 == what).map(|f| f.1[clock]).collect()
}

fn sub(seen: &Seen, k: usize) -> String {
    seen.frames.iter().find(|f| f.0 == k && f.2.starts_with("sub:")).map(|f| f.2[4..].to_string()).unwrap_or_default()
}

/// 다시 붙기까지: 연결 k 의 메시지를 다 보낸 뒤 연결 k+1 을 받아들일 때까지 초.
fn redial(seen: &Seen, k: usize, clock: usize) -> f64 {
    seen.conns[k + 1].0[clock] - seen.conns[k].1[clock]
}

fn within(x: f64, lo: f64, hi: f64, what: &str) {
    assert!(lo <= x && x <= hi, "{what}: {x:.2}초 (기대 {lo}~{hi})");
}

const T0: i64 = 1_763_683_200_000 + 3_437; // 분 시작 3.4초 뒤 -- startup 공백(1.5초 문턱 위)

// ── Bybit 청산 ─────────────────────────────────────────────────────────────────────────────────────
fn liq_msgs(k: i64) -> Vec<String> {
    let mut out = vec![json!({"success": true, "ret_msg": "", "conn_id": "c", "op": "subscribe"}).to_string()];
    for i in 0..6i64 {
        let ts = T0 + k * 7_000 + i * 300;
        let data: Vec<Value> = (0..(1 + i % 2))
            .map(|j| json!({"T": ts + j, "s": if j == 0 { "ETHUSDT" } else { "BTCUSDT" }, "S": if (i + j) % 2 == 0 { "Buy" } else { "Sell" },
                             "v": format!("{:.3}", 0.1 + (i * 3 + j) as f64 * 0.37), "p": format!("{:.2}", 2500.0 + i as f64)}))
            .collect();
        out.push(json!({"topic": "allLiquidation.ETHUSDT", "type": "snapshot", "ts": ts + 5, "data": data}).to_string());
    }
    out
}

#[tokio::test]
async fn bybit_liq_loop_matches_replay() {
    let dir = tmpdir("liq");
    let (port, seen) = serve(vec![
        Conn { msgs: liq_msgs(0), end: End::Close },
        Conn { msgs: liq_msgs(1), end: End::Read(Duration::from_secs(75)) }, // 침묵 -- 클라이언트가 60초에 끊는다
        Conn { msgs: liq_msgs(2), end: End::Read(Duration::from_secs(60)) },
    ])
    .await;
    let db = dir.join("rs").join("bybit_liq.sqlite");
    let env = [("RTCOLLECT_KIND", "bybit_liq".into()), ("BYBIT_LIQ_SYMBOLS", "ethusdt, btcusdt".into()),
               ("BYBIT_LIQ_DB", db.display().to_string()), ("RTCOLLECT_WS_URL", format!("ws://127.0.0.1:{port}/v5/public/linear"))];
    run_bin(&env, &seen, 3, 100).await;
    let s = seen.lock().unwrap();
    let want_sub = r#"{"op": "subscribe", "args": ["allLiquidation.ETHUSDT", "allLiquidation.BTCUSDT"]}"#;
    for k in 0..3 {
        assert_eq!(sub(&s, k), want_sub, "연결 {k} 구독 글자");
    }
    // 🔧의도한 차이: 원본은 ping·침묵을 벽시계(time.time)로 잰다(개발기 WSL 벽시계는 약 32초마다 2.4초씩 뒤로 뛴다) -- Rust 는 단조 시계로 잰다.
    //   그래서 단조 시각으로 판정한다. 판정은 5초 수신 시한마다라 «20초 이상 · 25초 미만».
    let p = pings(&s, 1, r#"{"op": "ping"}"#, 0);
    assert_eq!(p.len(), 2, "침묵 60초 동안 ping 20초마다 둘: {:?}", s.frames);
    within(p[0], 19.95, 25.0, "첫 ping(단조)");
    within(p[1] - p[0], 19.95, 25.0, "ping 간격(단조)");
    within(redial(&s, 0, 0), 0.9, 2.0, "끊김 뒤 다시 붙기(지수 대기 첫 1초)");
    within(redial(&s, 1, 0) - 1.0, 59.95, 65.5, "침묵 60초(단조) + 대기 1초(붙으면 대기가 1초로 돌아간다)");

    let mut ev = vec![];
    let mut t = 1.0;
    for (k, why) in [(0, "ConnectionClosedOK: received close"), (1, "ConnectionError: 응답 없음 60초"), (2, "SIGTERM")] {
        ev.push(json!([t, "o", ""]));
        for m in liq_msgs(k) {
            t += 0.003;
            ev.push(json!([t, "t", m]));
        }
        ev.push(json!([t, "c", why]));
        t += 5.0;
    }
    let rd = dir.join("replay.sqlite");
    rtcollect::bybit_liq::replay(&ev, &rd).unwrap();
    // recv_ms(5번 열)·공백 시각(0·1번 열)은 벽시계 -- 이유만
    let tables: [(&str, &str, &[usize]); 2] = [("bybit_liquidations", "rowid", &[5]), ("gaps", "rowid", &[0, 1])];
    let (a, b) = (dump(&db, &tables), dump(&rd, &tables));
    assert_eq!(a, b, "실행 파일 DB != replay DB");
    assert_eq!(a.iter().filter(|r| r.starts_with("gaps|")).count(), 2, "{a:#?}");
    eprintln!("Bybit 청산 접속 루프 = replay: 행 {} · ping(단조) {:?} · 다시 붙기 {:.2}·{:.2}초(단조) · 벽시계 {:?}", a.len(), p, redial(&s, 0, 0), redial(&s, 1, 0),
              pings(&s, 1, r#"{"op": "ping"}"#, 1));
    let _ = std::fs::remove_dir_all(&dir);
}

// ── Bybit 체결 테이프 ───────────────────────────────────────────────────────────────────────────────
fn tape_msgs(k: i64) -> Vec<String> {
    let mut out = vec![json!({"success": true, "ret_msg": "", "conn_id": "c", "op": "subscribe"}).to_string()];
    let t0 = T0 + k * 7_000;
    for s in 0..3i64 {
        for (sym, px) in [("ETHUSDT", 2500.0), ("XRPUSDT", 2.1)] {
            // 한 메시지에 묶음 셋: 같은 (시각, 방향, 가격) 둘 + 다른 가격 하나 · 다음 메시지가 같은 키로 이어 붙기도 한다
            let ts = t0 + s * 1000 + 120;
            let d: Vec<Value> = [(0, "Buy", 0.0), (0, "Buy", 0.0), (3, "Sell", 1.0)]
                .iter()
                .map(|&(dt, side, dp)| json!({"T": ts + dt, "s": sym, "S": side, "v": format!("{:.2}", 1.5 + dp + k as f64), "p": format!("{:.4}", px + dp * 0.1), "BT": false}))
                .collect();
            out.push(json!({"topic": format!("publicTrade.{sym}"), "type": "snapshot", "ts": ts + 9, "data": d}).to_string());
            out.push(json!({"topic": format!("publicTrade.{sym}"), "type": "snapshot", "ts": ts + 10,
                            "data": [{"T": ts + 3, "s": sym, "S": "Sell", "v": "0.25", "p": format!("{:.4}", px + 0.1)}]}).to_string());
        }
        let ts = t0 + s * 1000 + 500;
        let data = if s == 0 {
            json!({"symbol": "ETHUSDT", "openInterest": format!("{}", 1000 + k), "openInterestValue": "2500000", "fundingRate": "0.0001",
                   "nextFundingTime": "1763712000000", "markPrice": "2500.5", "indexPrice": "2500.4"})
        } else {
            json!({"symbol": "ETHUSDT", "markPrice": format!("{:.1}", 2500.5 + s as f64), "openInterest": format!("{}", 1000 + k + s)})
        };
        out.push(json!({"topic": "tickers.ETHUSDT", "type": if s == 0 { "snapshot" } else { "delta" }, "ts": ts, "data": data}).to_string());
    }
    out
}

#[tokio::test]
async fn bybit_tape_loop_matches_replay() {
    let dir = tmpdir("tape");
    let (port, seen) = serve(vec![
        Conn { msgs: tape_msgs(0), end: End::Close },
        Conn { msgs: tape_msgs(1), end: End::Read(Duration::from_secs(75)) },
        Conn { msgs: tape_msgs(2), end: End::Read(Duration::from_secs(60)) },
    ])
    .await;
    let (tdb, cdb) = (dir.join("rs").join("bybit_tape.sqlite"), dir.join("rs").join("bybit_ctx.sqlite"));
    let env = [("RTCOLLECT_KIND", "bybit_tape".into()), ("BYBIT_TAPE_SYMBOLS", "ETHUSDT,XRPUSDT".into()), ("BYBIT_TAPE_DB", tdb.display().to_string()),
               ("BYBIT_CTX_DB", cdb.display().to_string()), ("RTCOLLECT_WS_URL", format!("ws://127.0.0.1:{port}/v5/public/linear"))];
    run_bin(&env, &seen, 3, 100).await;
    let s = seen.lock().unwrap();
    let want_sub = r#"{"op": "subscribe", "args": ["publicTrade.ETHUSDT", "publicTrade.XRPUSDT", "tickers.ETHUSDT", "tickers.XRPUSDT"]}"#;
    for k in 0..3 {
        assert_eq!(sub(&s, k), want_sub, "연결 {k} 구독 글자");
    }
    // 원본은 단조 시계(time.monotonic)로 잰다
    let p = pings(&s, 1, r#"{"op": "ping"}"#, 0);
    assert_eq!(p.len(), 2, "침묵 60초 동안 ping 20초마다 둘: {:?}", s.frames);
    within(p[0], 19.5, 21.5, "첫 ping");
    within(p[1] - p[0], 19.5, 20.5, "ping 간격");
    within(redial(&s, 0, 0), 2.9, 4.0, "끊김 뒤 3초");
    within(redial(&s, 1, 0), 62.0, 64.5, "침묵 60초 + 3초");

    let mut ev = vec![];
    let mut t = 1.0;
    for (k, why) in [(0, "ConnectionError"), (1, "ConnectionError"), (2, "SIGTERM")] {
        ev.push(json!([t, "o", ""]));
        for m in tape_msgs(k) {
            t += 0.003;
            ev.push(json!([t, "t", m]));
        }
        ev.push(json!([t, "c", why]));
        t += 5.0;
    }
    let rd = dir.join("replay");
    rtcollect::bybit_tape::replay(&[json!({"symbols": ["ETHUSDT", "XRPUSDT"], "t0": 0.5, "events": ev})], &rd).unwrap();
    let tape: [(&str, &str, &[usize]); 3] = [("trade_tape_1s", "symbol, ts_sec, price_bin", &[]), ("gaps", "rowid", &[]), ("meta", "rowid", &[])];
    let ctx: [(&str, &str, &[usize]); 4] = [("bybit_oi", "rowid", &[]), ("bybit_mark", "rowid", &[]), ("bybit_funding", "rowid", &[]), ("gaps", "rowid", &[0, 1])];
    let (a, b) = (dump(&tdb, &tape), dump(&rd.join("bybit_tape.sqlite"), &tape));
    assert_eq!(a, b, "테이프: 실행 파일 DB != replay DB");
    let (c, d) = (dump(&cdb, &ctx), dump(&rd.join("bybit_ctx.sqlite"), &ctx));
    assert_eq!(c, d, "맥락: 실행 파일 DB != replay DB");
    assert_eq!(a.iter().filter(|r| r.starts_with("gaps|")).count(), 6, "종목 둘 × (startup + 재연결 둘): {a:#?}");
    assert_eq!(c.iter().filter(|r| r.starts_with("gaps|")).count(), 2, "{c:#?}");
    eprintln!("Bybit 테이프 접속 루프 = replay: 테이프 {} · 맥락 {} · ping {:?} · 다시 붙기 {:.2}·{:.2}초", a.len(), c.len(), p, redial(&s, 0, 0), redial(&s, 1, 0));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn bybit_tape_exits_3_when_trades_stall() {
    let dir = tmpdir("tape_wd");
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = l.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            let (s, _) = l.accept().await.unwrap();
            tokio::spawn(async move {
                let mut ws = accept_async(s).await.unwrap();
                let mut ts = T0;
                loop {
                    // tickers 만 -- 수신은 살아 있는데 체결이 없다
                    ts += 300;
                    let m = json!({"topic": "tickers.ETHUSDT", "ts": ts, "data": {"symbol": "ETHUSDT", "markPrice": "2500.5"}}).to_string();
                    if ws.send(Message::text(m)).await.is_err() {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(300)).await;
                }
            });
        }
    });
    let mut child = Command::new(env!("CARGO_BIN_EXE_rtcollect"))
        .env("RTCOLLECT_KIND", "bybit_tape")
        .env("BYBIT_TAPE_SYMBOLS", "ETHUSDT")
        .env("BYBIT_TAPE_DB", dir.join("t.sqlite"))
        .env("BYBIT_CTX_DB", dir.join("c.sqlite"))
        .env("RTCOLLECT_WS_URL", format!("ws://127.0.0.1:{port}/"))
        .env("RTCOLLECT_WD_TRADE_S", "2")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let t0 = Instant::now();
    let out = tokio::time::timeout(Duration::from_secs(20), tokio::task::spawn_blocking(move || {
        let st = child.wait().unwrap();
        let mut e = String::new();
        std::io::Read::read_to_string(child.stderr.as_mut().unwrap(), &mut e).unwrap();
        (st, e)
    }))
    .await
    .expect("20초 안에 안 끝났다")
    .unwrap();
    assert_eq!(out.0.code(), Some(3), "{}", out.1);
    assert!(out.1.contains("멈춤으로 보고 종료"), "{}", out.1);
    eprintln!("Bybit 테이프 정지 감시: {:.1}초 만에 종료코드 3", t0.elapsed().as_secs_f64());
    let _ = std::fs::remove_dir_all(&dir);
}

// ── OKX 체결 테이프 ─────────────────────────────────────────────────────────────────────────────────
fn okx_msgs(k: i64) -> Vec<String> {
    let arg = json!({"channel": "trades", "instId": "ETH-USDT-SWAP"});
    let mut out = vec![json!({"event": "subscribe", "arg": arg, "connId": "c"}).to_string()];
    let t0 = T0 + k * 7_000;
    let mut id = 1_000 + k * 100;
    for s in 0..3i64 {
        for o in 0..4i64 {
            id += 1;
            let d = json!({"instId": "ETH-USDT-SWAP", "tradeId": id.to_string(), "px": format!("{:.2}", 2500.0 + o as f64 * 0.37),
                           "sz": format!("{}", 1 + o * 7 + k), "side": if o % 2 == 0 { "buy" } else { "sell" },
                           "ts": (t0 + s * 1000 + o * 150).to_string(), "count": format!("{}", 1 + o % 3)});
            out.push(json!({"arg": arg, "data": [d]}).to_string());
        }
    }
    out
}

#[tokio::test]
async fn okx_loop_matches_replay() {
    let dir = tmpdir("okx");
    let (port, seen) = serve(vec![
        Conn { msgs: okx_msgs(0), end: End::Close },
        Conn { msgs: okx_msgs(1), end: End::Read(Duration::from_secs(23)) }, // PING 프레임 하나(20초) -- 서버가 pong 하니 끊기지 않는다
        Conn { msgs: okx_msgs(2), end: End::Mute(Duration::from_secs(40)) }, // 죽은 연결: pong 도 없다 -- 25초에 다시 붙는다
        Conn { msgs: okx_msgs(3), end: End::Read(Duration::from_secs(60)) },
    ])
    .await;
    let db = dir.join("rs").join("okx_tape.sqlite");
    let env = [("RTCOLLECT_KIND", "okx_tape".into()), ("OKX_TAPE_INST", "eth-usdt-swap".into()), ("OKX_TAPE_DB_PATH", db.display().to_string()),
               ("RTCOLLECT_WS_URL", format!("ws://127.0.0.1:{port}/ws/v5/public"))];
    run_bin(&env, &seen, 4, 100).await;
    let s = seen.lock().unwrap();
    for k in 0..4 {
        assert_eq!(sub(&s, k), r#"{"op": "subscribe", "args": [{"channel": "trades", "instId": "ETH-USDT-SWAP"}]}"#, "연결 {k} 구독 글자");
    }
    let p = pings(&s, 1, "PING", 0);
    assert_eq!(p.len(), 1, "23초 동안 PING 프레임 하나: {:?}", s.frames);
    within(p[0], 19.5, 21.0, "PING(마지막 수신 뒤 20초)");
    within(redial(&s, 0, 0), 2.9, 4.0, "끊김 뒤 3초");
    within(redial(&s, 1, 0), 25.9, 27.0, "서버가 23초에 닫음 + 3초");
    within(redial(&s, 2, 0), 27.9, 29.0, "25초 무음 + 3초");

    let mut ev = vec![];
    let mut t = 1.0;
    for (k, end) in [(0, "c"), (1, "c"), (2, "w"), (3, "c")] {
        ev.push(json!([t, "o", ""]));
        for m in okx_msgs(k) {
            t += 0.003;
            ev.push(json!([t, "t", m]));
        }
        ev.push(json!([t, end, ""]));
        t += 5.0;
    }
    let rd = dir.join("replay.sqlite");
    rtcollect::okx::replay(&[json!({"inst": "ETH-USDT-SWAP", "t0": 0.5, "events": ev})], &rd).unwrap();
    let tables: [(&str, &str, &[usize]); 3] = [("trade_tape_1s", "symbol, ts_sec, price_bin", &[]), ("gaps", "rowid", &[]), ("meta", "rowid", &[])];
    let (a, b) = (dump(&db, &tables), dump(&rd, &tables));
    assert_eq!(a, b, "실행 파일 DB != replay DB");
    assert_eq!(a.iter().filter(|r| r.starts_with("gaps|")).count(), 4, "startup + 재연결 셋: {a:#?}");
    eprintln!("OKX 접속 루프 = replay: 행 {} · PING {:?} · 다시 붙기 {:.2}·{:.2}·{:.2}초", a.len(), p, redial(&s, 0, 0), redial(&s, 1, 0), redial(&s, 2, 0));
    let _ = std::fs::remove_dir_all(&dir);
}

// ── OKX 맥락 ────────────────────────────────────────────────────────────────────────────────────────
fn okx_ctx_msgs(k: i64) -> Vec<String> {
    let i = "ETH-USDT-SWAP";
    let mut out = vec![json!({"event": "subscribe", "arg": {"channel": "mark-price", "instId": i}, "connId": "c"}).to_string()];
    let t0 = T0 + k * 7_000;
    for s in 0..6i64 {
        let ts = (t0 + s * 200).to_string();
        out.push(json!({"arg": {"channel": "mark-price", "instId": i}, "data": [{"instId": i, "markPx": format!("{:.2}", 2745.0 + (s + k) as f64 * 0.13), "ts": ts}]}).to_string());
        if s % 2 == 0 {
            out.push(json!({"arg": {"channel": "open-interest", "instId": i}, "data": [{"instId": i, "oi": format!("{}", 5_000_000 + s * 10 + k), "oiCcy": format!("{}", 500_000 + s + k), "oiUsd": "1.5e9", "ts": ts}]}).to_string());
        }
    }
    out.push(json!({"arg": {"channel": "funding-rate", "instId": i}, "data": [{"instId": i, "fundingRate": "0.0000588", "fundingTime": "1790092800000", "minFundingRate": "-0.0075", "maxFundingRate": "0.0075", "method": "current_period", "ts": (t0 + 900).to_string()}]}).to_string());
    out.push(json!({"arg": {"channel": "liquidation-orders", "instType": "SWAP"}, "data": [{"instId": i, "instFamily": "ETH-USDT", "details": [{"bkLoss": "0", "bkPx": "2735.52", "posSide": "short", "side": "buy", "sz": format!("{}.97", 7 + k), "ts": (t0 + 950).to_string()}]}]}).to_string());
    out
}

#[tokio::test]
async fn okx_ctx_loop_matches_replay() {
    let dir = tmpdir("okx_ctx");
    let (port, seen) = serve(vec![
        Conn { msgs: okx_ctx_msgs(0), end: End::Close },
        Conn { msgs: okx_ctx_msgs(1), end: End::Read(Duration::from_secs(23)) }, // PING 프레임 하나(20초)
        Conn { msgs: okx_ctx_msgs(2), end: End::Mute(Duration::from_secs(50)) }, // 죽은 연결 -- 40초에 다시 붙는다
        Conn { msgs: okx_ctx_msgs(3), end: End::Read(Duration::from_secs(60)) },
    ])
    .await;
    let db = dir.join("rs").join("okx_ctx.sqlite");
    let env = [("RTCOLLECT_KIND", "okx_ctx".into()), ("OKX_CTX_INST", "eth-usdt-swap".into()), ("OKX_CTX_DB_PATH", db.display().to_string()),
               ("RTCOLLECT_WS_URL", format!("ws://127.0.0.1:{port}/ws/v5/public"))];
    run_bin(&env, &seen, 4, 100).await;
    let s = seen.lock().unwrap();
    let want = r#"{"op": "subscribe", "args": [{"channel": "open-interest", "instId": "ETH-USDT-SWAP"}, {"channel": "mark-price", "instId": "ETH-USDT-SWAP"}, {"channel": "funding-rate", "instId": "ETH-USDT-SWAP"}, {"channel": "liquidation-orders", "instType": "SWAP"}]}"#;
    for k in 0..4 {
        assert_eq!(sub(&s, k), want, "연결 {k} 구독 글자");
    }
    let p = pings(&s, 1, "PING", 0);
    assert_eq!(p.len(), 1, "23초 동안 PING 프레임 하나: {:?}", s.frames);
    within(p[0], 19.5, 21.0, "PING(마지막 수신 뒤 20초)");
    within(redial(&s, 0, 0), 2.9, 4.0, "끊김 뒤 3초");
    within(redial(&s, 1, 0), 25.9, 27.0, "서버가 23초에 닫음 + 3초");
    within(redial(&s, 2, 0), 42.9, 44.0, "40초 무음 + 3초");

    let mut ev = vec![];
    let mut t = 1.0;
    for (k, end) in [(0, "c"), (1, "c"), (2, "w"), (3, "c")] {
        ev.push(json!([t, "o", ""]));
        for m in okx_ctx_msgs(k) {
            t += 0.003;
            ev.push(json!([t, "t", m]));
        }
        ev.push(json!([t, end, ""]));
        t += 5.0;
    }
    let rd = dir.join("replay.sqlite");
    rtcollect::okx_ctx::replay(&[json!({"inst": "ETH-USDT-SWAP", "t0": 0.5, "events": ev})], &rd).unwrap();
    // 공백의 시각은 벽시계라 두 쪽이 다르다 -- 채널·이유만
    let tables: [(&str, &str, &[usize]); 6] = [("okx_oi", "rowid", &[]), ("okx_mark", "rowid", &[]), ("okx_funding", "rowid", &[]),
        ("okx_liquidations", "rowid", &[]), ("gaps", "rowid", &[1, 2]), ("meta", "rowid", &[])];
    let (a, b) = (dump(&db, &tables), dump(&rd, &tables));
    assert_eq!(a, b, "실행 파일 DB != replay DB");
    assert_eq!(a.iter().filter(|r| r.starts_with("gaps|")).count(), 3, "재연결 셋: {a:#?}");
    eprintln!("OKX 맥락 접속 루프 = replay: 행 {} · PING {:?} · 다시 붙기 {:.2}·{:.2}·{:.2}초", a.len(), p, redial(&s, 0, 0), redial(&s, 1, 0), redial(&s, 2, 0));
    let _ = std::fs::remove_dir_all(&dir);
}

// ── 바이낸스 맥락(구독 글자 없음 · 경로가 스트림) ─────────────────────────────────────────────────────────
/// 경로마다 연결 각본 차례로. 반환 = 경로별 (받아들인 시각, 메시지를 다 보낸 시각) · 받은 PING 시각(보낸 뒤 초).
async fn serve_paths(script: Vec<(&'static str, Vec<Conn>)>) -> (u16, Arc<Mutex<Vec<(String, f64, f64)>>>, Arc<Mutex<Vec<(String, f64)>>>) {
    use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = l.local_addr().unwrap().port();
    let conns = Arc::new(Mutex::new(vec![]));
    let pings = Arc::new(Mutex::new(vec![]));
    let queues = Arc::new(Mutex::new(script.into_iter().map(|(p, c)| (p.to_string(), c.into_iter().collect::<std::collections::VecDeque<_>>())).collect::<Vec<_>>()));
    let (c2, p2) = (conns.clone(), pings.clone());
    let t0 = Instant::now();
    tokio::spawn(async move {
        loop {
            let (s, _) = l.accept().await.unwrap();
            let (conns, pings, queues) = (c2.clone(), p2.clone(), queues.clone());
            tokio::spawn(async move {
                let path = Arc::new(Mutex::new(String::new()));
                let p3 = path.clone();
                let cb = move |req: &Request, resp: Response| {
                    *p3.lock().unwrap() = req.uri().path().to_string();
                    Ok(resp)
                };
                let mut ws = tokio_tungstenite::accept_hdr_async(s, cb).await.unwrap();
                let path = path.lock().unwrap().clone();
                let acc = t0.elapsed().as_secs_f64();
                let c = queues.lock().unwrap().iter_mut().find(|q| path.ends_with(&q.0)).and_then(|q| q.1.pop_front());
                let Some(c) = c else {
                    tokio::time::sleep(Duration::from_secs(600)).await; // 각본 끝 -- 쥐고만 있는다
                    return;
                };
                for m in &c.msgs {
                    ws.send(Message::text(m.clone())).await.unwrap();
                    tokio::time::sleep(Duration::from_millis(3)).await;
                }
                let sent = Instant::now();
                conns.lock().unwrap().push((path.clone(), acc, t0.elapsed().as_secs_f64()));
                match c.end {
                    End::Close => {
                        let _ = ws.close(None).await;
                    }
                    End::Read(d) => {
                        while let Ok(Some(Ok(f))) = tokio::time::timeout_at((sent + d).into(), ws.next()).await {
                            if let Message::Ping(_) = f {
                                pings.lock().unwrap().push((path.clone(), sent.elapsed().as_secs_f64()));
                            }
                        }
                        let _ = ws.close(None).await;
                    }
                    End::Mute(d) => {
                        tokio::time::sleep(d).await;
                        drop(ws);
                    }
                }
            });
        }
    });
    (port, conns, pings)
}

fn bn_mark_msgs(k: i64) -> Vec<String> {
    (0..5i64).map(|s| json!({"e": "markPriceUpdate", "E": T0 + (k * 10 + s) * 1000, "s": "ETHUSDT", "p": format!("{:.8}", 2500.0 + s as f64 * 0.1),
        "P": "0", "i": "2500.00000000", "r": "0.00010000", "T": 1_763_712_000_000i64}).to_string())
        .chain([json!({"e": "forceOrder", "E": T0, "o": {"s": "ETHUSDT", "S": "SELL", "q": "1", "p": "1", "ap": "1", "z": "1", "T": T0}}).to_string()])
        .collect()
}

/// 마지막 마크 연결: 셋째 프레임 자리에 깨진 JSON -- 고친 Rust 는 건너뛰고 같은 연결로 나머지를 받는다(되돌리면 끊겨 뒤 행이 사라지고 gaps 가 하나 는다).
fn bn_mark_broken() -> Vec<String> {
    let mut v = bn_mark_msgs(3);
    v.insert(2, r#"{"e":"markPriceUpdate","E":17"#.into());
    v
}

fn bn_force_msgs() -> Vec<String> {
    let syms = ["ETHUSDT", "DOGEUSDT", "ZECUSDT", "BTCUSDT"];
    (0..4i64).map(|s| json!({"e": "forceOrder", "E": T0 + s, "o": {"s": syms[s as usize],
        "S": if s % 2 == 0 { "SELL" } else { "BUY" }, "q": "0.5", "p": "2400", "ap": "2410.5", "z": format!("0.{}", s + 1), "T": T0 + s * 100}}).to_string())
        .chain([json!({"e": "markPriceUpdate", "E": T0, "s": "ETHUSDT", "p": "1", "i": "1", "r": "1", "T": 0}).to_string()])
        .collect()
}

#[tokio::test]
async fn binance_ctx_loop_matches_replay() {
    let dir = tmpdir("binance_ctx");
    let (port, conns, pings) = serve_paths(vec![
        ("ethusdt@markPrice@1s", vec![
            Conn { msgs: bn_mark_msgs(0), end: End::Close },
            Conn { msgs: bn_mark_msgs(1), end: End::Read(Duration::from_secs(33)) }, // PING 하나(30초) -- 서버가 pong 하니 안 끊긴다
            Conn { msgs: bn_mark_msgs(2), end: End::Mute(Duration::from_secs(60)) }, // pong 없음 -- 30+15초에 닫고 3초 뒤
            Conn { msgs: bn_mark_broken(), end: End::Read(Duration::from_secs(120)) }, // 🔧깨진 프레임 뒤 정상 프레임이 같은 연결로(원본이면 여기서 끊긴다)
        ]),
        ("!forceOrder@arr", vec![Conn { msgs: bn_force_msgs(), end: End::Read(Duration::from_secs(200)) }]),
    ])
    .await;
    let db = dir.join("rs").join("binance_ctx.sqlite");
    let env = [("RTCOLLECT_KIND", "binance_ctx".into()), ("BN_CTX_MARK_SYMBOLS", "ETHUSDT".into()), ("BN_CTX_DB", db.display().to_string()),
               ("RTCOLLECT_WS_URL", format!("ws://127.0.0.1:{port}/market/ws/"))];
    // run_bin 은 Seen 을 보므로 따로 띄운다: 연결 5개(마크 4 + 청산 1)가 다 오면 1.5초 뒤 SIGTERM
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rtcollect"));
    for (k, v) in &env {
        cmd.env(k, v);
    }
    let mut child = cmd.stdout(Stdio::null()).stderr(Stdio::piped()).spawn().unwrap();
    let t0 = Instant::now();
    while conns.lock().unwrap().len() < 5 {
        assert!(t0.elapsed() < Duration::from_secs(150), "연결 5개가 150초 안에 안 왔다");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert!(Command::new("kill").args(["-TERM", &child.id().to_string()]).status().unwrap().success());
    let st = tokio::task::spawn_blocking(move || child.wait().unwrap()).await.unwrap();
    assert_eq!(st.code(), Some(0), "SIGTERM 은 0");
    let c = conns.lock().unwrap().clone();
    let m: Vec<_> = c.iter().filter(|x| x.0.ends_with("markPrice@1s")).collect();
    let redial = |k: usize| m[k + 1].1 - m[k].2;
    within(redial(0), 2.9, 4.0, "끊김 뒤 3초");
    within(redial(1), 35.9, 37.0, "서버가 33초에 닫음 + 3초");
    within(redial(2), 47.9, 49.0, "PING 30초 + pong 15초 없음 + 3초");
    let p: Vec<f64> = pings.lock().unwrap().iter().filter(|x| x.0.ends_with("markPrice@1s")).map(|x| x.1).collect();
    assert!(p.len() == 1 && (29.5..31.0).contains(&p[0]), "33초 동안 PING 하나(30초): {p:?}");

    let mut feeds = vec![];
    let mut ev = vec![];
    for k in 0..4 {
        ev.push(json!([1.0, "o", ""]));
        ev.extend((if k == 3 { bn_mark_broken() } else { bn_mark_msgs(k) }).into_iter().map(|m| json!([1.0, "t", m])));
        ev.push(json!([1.0, "c", ""]));
    }
    feeds.push(json!({"kind": "mark", "events": ev}));
    let mut ev = vec![json!([1.0, "o", ""])];
    ev.extend(bn_force_msgs().into_iter().map(|m| json!([1.0, "t", m])));
    feeds.push(json!({"kind": "force", "events": ev}));
    let rd = dir.join("replay.sqlite");
    let con = rtcollect::binance_ctx::open_db(&rd).unwrap();
    let pend = Arc::new(Mutex::new(rtcollect::binance_ctx::Pending::default()));
    let slow: Vec<String> = rtcollect::binance_ctx::OI_SLOW_SYMBOLS.split(',').map(str::to_string).collect();
    rtcollect::binance_ctx::replay_ws(&feeds, &slow, &pend);
    rtcollect::binance_ctx::flush(&con, &pend);
    drop(con);
    let tables: [(&str, &str, &[usize]); 3] = [("oi_1s", "ts_ms, symbol", &[]), ("mark_price_1s", "ts_ms, symbol", &[]), ("liquidations", "rowid", &[])];
    let (a, b) = (dump(&db, &tables), dump(&rd, &tables));
    assert_eq!(a, b, "실행 파일 DB != replay DB");
    assert_eq!((a.iter().filter(|r| r.starts_with("mark_price_1s|")).count(), a.iter().filter(|r| r.starts_with("liquidations|")).count()), (20, 3));
    // 🔧더한 gaps: 마크 연결이 세 번 끊겼다(3초·36초·48초 -- 다 1.5초 넘음) · 강제청산은 안 끊겼다
    let g = dump(&db, &[("gaps", "rowid", &[1, 2])]);
    assert_eq!(g, vec![r#"gaps|"ethusdt"|"reconnect""#; 3], "{g:?}");
    eprintln!("바이낸스 맥락 접속 루프 = replay: 행 {} · PING {p:?} · 다시 붙기 {:.2}·{:.2}·{:.2}초", a.len(), redial(0), redial(1), redial(2));
    let _ = std::fs::remove_dir_all(&dir);
}

// ── xyz100 ──────────────────────────────────────────────────────────────────────────────────────────
/// 연결 k: ctx(종목마다 다른 이름 -- 받은 «초»가 두 쪽에서 달라도 열쇠가 안 겹치게) · candle · 2.1초 쉼 · ctx 하나(= 플러시) · 뒤 행(끊기면 버려진다).
fn xyz_msgs(k: i64) -> Vec<String> {
    let ctx = |j: i64| json!({"channel": "activeAssetCtx", "data": {"coin": format!("xyz:C{k}_{j}"), "ctx": {"midPx": format!("{}.5", 30_000 + j), "markPx": "30001.0", "oraclePx": null}}}).to_string();
    let candle = |j: i64| json!({"channel": "candle", "data": {"t": T0 / 60_000 * 60_000 + (k * 10 + j) * 60_000, "s": "xyz:XYZ100", "o": "1", "h": "2", "l": "0.5", "c": format!("1.{j}"), "v": "0.1", "n": j}}).to_string();
    let mut out = vec![json!({"channel": "subscriptionResponse", "data": {"method": "subscribe"}}).to_string()];
    out.extend((0..3).map(ctx));
    out.extend((0..2).map(candle));
    out.push("SLEEP:2100".into());
    out.push(ctx(3));
    out.extend([ctx(4), candle(5)]); // 플러시 뒤 -- 🔧원본은 연결이 끝나면 버린다 · Rust 는 끊길 때(종료 때도) 쓴다
    out
}

#[tokio::test]
async fn xyz_loop_matches_replay() {
    let dir = tmpdir("xyz");
    let (port, seen) = serve(vec![
        Conn { msgs: xyz_msgs(0), end: End::Close },
        Conn { msgs: xyz_msgs(1), end: End::Read(Duration::from_secs(28)) }, // PING 프레임(25초) → pong → 30초 시한이 다시 재진다 = 글자 ping 없음
        Conn { msgs: xyz_msgs(2)[..4].to_vec(), end: End::Mute(Duration::from_secs(60)) }, // pong 없음: 25초 PING + 12.5초 = 닫힘
        Conn { msgs: xyz_msgs(3), end: End::Read(Duration::from_secs(60)) },
    ])
    .await;
    let db = dir.join("rs").join("hl_xyz.sqlite");
    let env = [("RTCOLLECT_KIND", "xyz".into()), ("HL_XYZ_COINS", "xyz:XYZ100,xyz:SP500".into()), ("HL_XYZ_DB", db.display().to_string()),
               ("RTCOLLECT_WS_URL", format!("ws://127.0.0.1:{port}/ws"))];
    run_bin(&env, &seen, 4, 130).await;
    let s = seen.lock().unwrap();
    // 첫 프레임(구독 넷 중 첫째)만 serve 가 sub: 로 적는다 -- 나머지 셋·ping 은 Read 동안 받은 것
    for k in 0..4 {
        assert_eq!(sub(&s, k), r#"{"method": "subscribe", "subscription": {"type": "candle", "coin": "xyz:XYZ100", "interval": "1m"}}"#, "연결 {k} 첫 구독");
    }
    let p = pings(&s, 1, "PING", 0);
    assert!(p.len() == 1 && (24.5..26.0).contains(&p[0]), "28초 동안 PING 프레임 하나(마지막 수신 뒤 25초): {p:?}");
    assert!(!s.frames.iter().any(|f| f.2 == xyz_ping()), "pong 이 오면 글자 ping 은 안 나간다: {:?}", s.frames);
    within(redial(&s, 0, 0), 0.9, 1.6, "끊김 뒤 1초(플러시가 대기를 1로)");
    within(redial(&s, 1, 0), 28.9, 29.6, "28초 닫힘 + 1초");
    within(redial(&s, 2, 0), 39.3, 40.2, "PING 25초 + pong 12.5초 + 2초(플러시 없던 연결 -- 대기 두 배)");

    let mut ev = vec![];
    let mut t = 10.0;
    for (k, end) in [(0, "c"), (1, "c"), (2, "c"), (3, "c")] {
        ev.push(json!([t, "o", ""]));
        let msgs = if k == 2 { xyz_msgs(2)[..4].to_vec() } else { xyz_msgs(k) };
        for m in msgs {
            if let Some(ms) = m.strip_prefix("SLEEP:") {
                t += ms.parse::<f64>().unwrap() / 1000.0;
                continue;
            }
            t += 0.003;
            ev.push(json!([t, "t", m]));
        }
        ev.push(json!([t, end, ""]));
        t += 50.0;
    }
    let rd = dir.join("replay.sqlite");
    rtcollect::xyz::replay(&ev, &["xyz:XYZ100".into(), "xyz:SP500".into()], &rd).unwrap();
    let tables: [(&str, &str, &[usize]); 2] = [("hl_xyz_1m", "rowid", &[]), ("hl_xyz_ctx_1s", "rowid", &[1])]; // recv_sec = 벽시계
    let (a, b) = (dump(&db, &tables), dump(&rd, &tables));
    assert_eq!(a, b, "실행 파일 DB != replay DB");
    // 🔧의도한 차이: 끊길 때·종료 때 남은 행을 쓴다(원본이면 (6, 12) -- 플러시 뒤 행·플러시 없던 연결 2 의 행이 버려진다)
    assert_eq!((a.iter().filter(|r| r.starts_with("hl_xyz_1m|")).count(), a.iter().filter(|r| r.starts_with("hl_xyz_ctx_1s|")).count()), (9, 18), "{a:#?}");
    eprintln!("xyz 접속 루프 = replay: 행 {} · PING {p:?} · 다시 붙기 {:.2}·{:.2}·{:.2}초", a.len(), redial(&s, 0, 0), redial(&s, 1, 0), redial(&s, 2, 0));
    let _ = std::fs::remove_dir_all(&dir);
}

fn xyz_ping() -> String {
    rtcollect::xyz::PING.to_string()
}

// ── 하이퍼리퀴드 자산 맥락 ─────────────────────────────────────────────────────────────────────────────
fn hl_ctx_msgs(k: i64) -> Vec<String> {
    (0..5i64).map(|j| json!({"channel": "activeAssetCtx", "data": {"coin": "ETH", "ctx": {"funding": "0.0000125", "openInterest": format!("{}.5", 1_144_000 + k * 10 + j),
        "premium": "0.0004", "oraclePx": "2746.1", "markPx": "2747.4", "midPx": "2747.45", "dayNtlVlm": "1.3e9", "dayBaseVlm": "487476.9"}}}).to_string()).collect()
}

#[tokio::test]
async fn hl_ctx_loop_matches_replay() {
    let dir = tmpdir("hl_ctx");
    let (port, seen) = serve(vec![
        Conn { msgs: hl_ctx_msgs(0), end: End::Close },
        Conn { msgs: hl_ctx_msgs(1), end: End::Read(Duration::from_secs(28)) }, // PING 프레임(25초) → pong 이 시한을 다시 잰다 = 글자 ping 없음
        Conn { msgs: hl_ctx_msgs(2), end: End::Mute(Duration::from_secs(60)) }, // pong 없음: 25초 PING + 12.5초 = 닫힘 (그 사이 30초에 글자 ping)
        Conn { msgs: hl_ctx_msgs(3), end: End::Read(Duration::from_secs(60)) },
    ])
    .await;
    let db = dir.join("rs").join("hl_ctx.sqlite");
    let env = [("RTCOLLECT_KIND", "hl_ctx".into()), ("HL_BT_COIN", "eth".into()), ("HL_CTX_DB_PATH", db.display().to_string()),
               ("RTCOLLECT_WS_URL", format!("ws://127.0.0.1:{port}/ws"))];
    run_bin(&env, &seen, 4, 120).await;
    let s = seen.lock().unwrap();
    for k in 0..4 {
        assert_eq!(sub(&s, k), r#"{"method": "subscribe", "subscription": {"type": "activeAssetCtx", "coin": "ETH"}}"#, "연결 {k} 구독");
    }
    let p = pings(&s, 1, "PING", 0);
    assert!(p.len() == 1 && (24.5..26.0).contains(&p[0]), "PING 하나(25초): {p:?}");
    within(redial(&s, 0, 0), 2.9, 4.0, "끊김 뒤 3초");
    within(redial(&s, 1, 0), 30.9, 32.0, "28초 닫힘 + 3초");
    within(redial(&s, 2, 0), 40.3, 41.5, "PING 25초 + pong 12.5초 + 3초");
    let mut ev = vec![];
    let mut t = 10.0;
    for k in 0..4 {
        ev.push(json!([t, "o", ""]));
        for m in hl_ctx_msgs(k) {
            t += 0.003;
            ev.push(json!([t, "t", m]));
        }
        ev.push(json!([t, "c", ""]));
        t += 50.0;
    }
    let rd = dir.join("replay.sqlite");
    rtcollect::hl_ctx::replay(&[json!({"coin": "ETH", "t0": 9.0, "events": ev})], &rd, None).unwrap();
    let tables: [(&str, &str, &[usize]); 2] = [("hl_asset_ctx", "rowid", &[1]), ("meta", "rowid", &[])]; // recv_ms = 벽시계
    let (a, b) = (dump(&db, &tables), dump(&rd, &tables));
    assert_eq!(a, b, "실행 파일 DB != replay DB");
    assert_eq!(a.iter().filter(|r| r.starts_with("hl_asset_ctx|")).count(), 20);
    eprintln!("HL 맥락 접속 루프 = replay: 행 {} · PING {p:?} · 다시 붙기 {:.2}·{:.2}·{:.2}초", a.len(), redial(&s, 0, 0), redial(&s, 1, 0), redial(&s, 2, 0));
    let _ = std::fs::remove_dir_all(&dir);
}

// ── hl_bt 수신 공백·끊김 로그(11-28) ── 1초 넘는 공백은 «수신 공백 N초»(글자 «끊김» 없음 -- takeover.sh 가 «끊김» 줄 = 재접속으로 센다) · 끊김 줄에 UTC 시각
#[tokio::test]
async fn hl_bt_logs_gap_and_disconnect_with_utc() {
    let dir = tmpdir("hl_bt_gap");
    let b = |ts: i64| json!({"channel": "bbo", "data": {"coin": "ETH", "time": ts, "bbo": [{"px": "100", "sz": "1", "n": 1}, {"px": format!("101.{ts}"), "sz": "1", "n": 1}]}}).to_string();
    let (port, seen) = serve(vec![
        Conn { msgs: vec![b(T0), "SLEEP:1500".into(), b(T0 + 1500)], end: End::Close },
        Conn { msgs: vec![b(T0 + 5000)], end: End::Read(Duration::from_secs(10)) },
    ])
    .await;
    let env = [("RTCOLLECT_KIND", "hl_bt".into()), ("HL_BT_COIN", "eth".into()), ("HL_BT_ROOT", dir.join("bt").display().to_string()),
               ("HL_CTX_DB_PATH", dir.join("hl_ctx.sqlite").display().to_string()), ("RTCOLLECT_WS_URL", format!("ws://127.0.0.1:{port}/ws"))];
    let err = run_bin(&env, &seen, 2, 30).await;
    let gap: Vec<&str> = err.lines().filter(|l| l.contains("수신 공백")).collect();
    let cut: Vec<&str> = err.lines().filter(|l| l.contains("끊김")).collect();
    assert!(gap.len() == 1 && regex::Regex::new(r"^\[hl\] \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z 수신 공백 1\.[5-9]초").unwrap().is_match(gap[0]), "공백 줄: {err}");
    assert!(cut.len() == 1 && regex::Regex::new(r"^\[hl\] \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z 끊김 .* 마지막 수신 .*Z \(\d+\.\d초 전\) -- 3초 뒤 재접속$").unwrap().is_match(cut[0]), "끊김 줄: {err}");
    eprintln!("{}\n{}", gap[0], cut[0]);
    let _ = std::fs::remove_dir_all(&dir);
}

// ── depth_diff ─────────────────────────────────────────────────────────────────────────────────────────
fn depth_msgs(k: i64) -> Vec<String> {
    (0..6i64).map(|j| {
        let u = 1_000 + k * 100 + j * 10;
        json!({"e": "depthUpdate", "E": T0 + k * 7_000 + j * 100 + 2, "T": T0 + k * 7_000 + j * 100, "s": "ETHUSDT", "U": u - 9, "u": u, "pu": if j == 0 { u - 10 - 5 } else { u - 10 },
               "b": [["2500.00", format!("{}.5", j)]], "a": []}).to_string()
    }).collect()
}

/// 파일 내용(시 파일 이름 · 줄) -- 첫 줄 스냅숏의 at_ms(벽시계)는 지운다.
fn depth_dump(root: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(root.join("ETHUSDT")).unwrap().flat_map(|e| {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        std::fs::read_to_string(&p).unwrap().lines().map(|l| {
            let l = if l.starts_with(r#"{"_snapshot""#) { l.split(r#","at_ms""#).next().unwrap().to_string() } else { l.to_string() };
            format!("{name}|{l}")
        }).collect::<Vec<_>>()
    }).collect();
    v.sort();
    v
}

#[tokio::test]
async fn depth_loop_matches_replay() {
    let dir = tmpdir("depth");
    let (port, conns, pings) = serve_paths(vec![("ethusdt@depth@100ms", vec![
        Conn { msgs: depth_msgs(0), end: End::Close },                            // 깨끗한 닫힘 = 곧바로 다시(파일 유지)
        Conn { msgs: depth_msgs(1), end: End::Mute(Duration::from_secs(60)) },    // pong 없음: 20초 PING + 10초 = 끊고 3초 뒤
        Conn { msgs: depth_msgs(2), end: End::Read(Duration::from_secs(45)) },    // PING 20·40초 -- 서버가 pong
    ])])
    .await;
    let root = dir.join("rs");
    let env = [("RTCOLLECT_KIND", "depth".into()), ("DD_SYMBOL", "ETHUSDT".into()), ("DD_ROOT", root.display().to_string()),
               ("RTCOLLECT_WS_URL", format!("ws://127.0.0.1:{port}/ws/{{symbol}}@depth@100ms"))];
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rtcollect"));
    for (k, v) in &env {
        cmd.env(k, v);
    }
    let mut child = cmd.stdout(Stdio::null()).stderr(Stdio::piped()).spawn().unwrap();
    let t0 = Instant::now();
    while conns.lock().unwrap().len() < 3 {
        assert!(t0.elapsed() < Duration::from_secs(90), "연결 3개가 90초 안에 안 왔다");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    tokio::time::sleep(Duration::from_secs(42)).await; // 셋째 연결의 PING 둘(20·40초)
    assert!(Command::new("kill").args(["-TERM", &child.id().to_string()]).status().unwrap().success());
    let st = tokio::task::spawn_blocking(move || child.wait().unwrap()).await.unwrap();
    assert_eq!(st.code(), Some(0), "SIGTERM 은 0");
    let c = conns.lock().unwrap().clone();
    let redial = |k: usize| c[k + 1].1 - c[k].2;
    within(redial(0), 0.0, 0.5, "깨끗한 닫힘 = 곧바로 다시");
    within(redial(1), 32.9, 34.0, "PING 20초 + pong 10초 + 3초");
    let p: Vec<f64> = pings.lock().unwrap().iter().map(|x| x.1).collect();
    assert!(p.len() == 2 && (19.0..21.0).contains(&p[0]) && (39.0..41.0).contains(&p[1]), "PING 20초마다(받는 것과 무관): {p:?}");
    let mut ev = vec![];
    for (k, end) in [(0, "e"), (1, "c"), (2, "c")] {
        ev.push(json!([1.0, "o", ""]));
        ev.extend(depth_msgs(k).into_iter().map(|m| json!([1.0, "t", m])));
        ev.push(json!([1.0, end, ""]));
    }
    let rd = dir.join("replay");
    rtcollect::depth::replay(&ev, "ethusdt", &rd, &[]).unwrap();
    let (a, b) = (depth_dump(&root), depth_dump(&rd));
    assert_eq!(a, b, "실행 파일 파일 != replay 파일");
    // 원본 그대로: 접속마다 prev_u 를 비워 첫 프레임은 대조하지 않는다 -- 재접속 사이 구멍은 _gap 으로 안 남는다(첫 프레임 pu 가 어긋나도)
    assert_eq!(a.iter().filter(|l| l.contains("_gap")).count(), 0, "{a:#?}");
    eprintln!("depth 접속 루프 = replay: 줄 {} · PING {p:?} · 다시 붙기 {:.2}·{:.2}초", a.len(), redial(0), redial(1));
    let _ = std::fs::remove_dir_all(&dir);
}

// ── 호가 래스터 ─────────────────────────────────────────────────────────────────────────────────────────
/// 가짜 스냅숏 REST(언제나 같은 북 · lastUpdateId 1000) -- 받은 요청 수를 센다.
fn snapshot_server() -> (String, Arc<Mutex<usize>>) {
    use std::io::{Read, Write};
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    let n = Arc::new(Mutex::new(0usize));
    let n2 = n.clone();
    std::thread::spawn(move || {
        for s in l.incoming() {
            let mut s = s.unwrap();
            let mut b = [0u8; 4096];
            let _ = s.read(&mut b);
            *n2.lock().unwrap() += 1;
            let bids: Vec<Value> = (1..=50).map(|k| json!([format!("{:.2}", 2500.0 - 0.01 * k as f64), "1.000"])).collect();
            let asks: Vec<Value> = (0..50).map(|k| json!([format!("{:.2}", 2500.0 + 0.01 * k as f64), "2.000"])).collect();
            let body = json!({"lastUpdateId": 1000, "E": 1, "T": 1, "bids": bids, "asks": asks}).to_string();
            let _ = s.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes());
        }
    });
    (base, n)
}

/// 접속마다: 스냅숏을 부르는 첫 이벤트(버려진다) → U <= 1000 <= u 인 동기 이벤트 → 이어지는 이벤트.
fn raster_msgs() -> Vec<String> {
    let ev = |uu: i64, u: i64, pu: i64, b: &str| json!({"e": "depthUpdate", "E": T0, "T": T0, "s": "ETHUSDT", "U": uu, "u": u, "pu": pu, "b": [[b, "3.000"]], "a": []}).to_string();
    vec![ev(900, 950, 899, "2499.90"), ev(990, 1005, 989, "2499.80"), ev(1006, 1010, 1005, "2499.70"), ev(1011, 1020, 1010, "2499.60")]
}

#[tokio::test]
async fn raster_loop_writes_rows_and_redials() {
    let dir = tmpdir("raster");
    let (rest, n_snap) = snapshot_server();
    let (port, conns, _pings) = serve_paths(vec![("ethusdt@depth@500ms", vec![
        Conn { msgs: raster_msgs(), end: End::Read(Duration::from_secs(4)) },       // 서버가 닫음 = recv 예외 → 3초(동기됐으니 대기 3초)
        Conn { msgs: raster_msgs(), end: End::Mute(Duration::from_secs(60)) },      // pong 없음: 20초 PING + 10초 = 끊고 3초
        Conn { msgs: raster_msgs(), end: End::Read(Duration::from_secs(60)) },
    ])])
    .await;
    let root = dir.join("rs");
    let env = [("RTCOLLECT_KIND", "raster".into()), ("OF_SYMBOL", "ETHUSDT".into()), ("OF_ROOT", root.display().to_string()), ("RTCOLLECT_REST", "1".into()),
               ("RTCOLLECT_DEPTH_URL", format!("{rest}/fapi/v1/depth?symbol={{SYMBOL}}&limit=1000")), ("RTCOLLECT_BAN_FILE", dir.join("ban").display().to_string()),
               ("RTCOLLECT_WS_URL", format!("ws://127.0.0.1:{port}/ws/{{symbol}}@depth@500ms"))];
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rtcollect"));
    for (k, v) in &env {
        cmd.env(k, v);
    }
    let mut child = cmd.stdout(Stdio::null()).stderr(Stdio::piped()).spawn().unwrap();
    let t0 = Instant::now();
    while conns.lock().unwrap().len() < 3 {
        assert!(t0.elapsed() < Duration::from_secs(90), "연결 3개가 90초 안에 안 왔다");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert!(Command::new("kill").args(["-TERM", &child.id().to_string()]).status().unwrap().success());
    let st = tokio::task::spawn_blocking(move || child.wait().unwrap()).await.unwrap();
    assert_eq!(st.code(), Some(0), "SIGTERM 은 0");
    let c = conns.lock().unwrap().clone();
    let redial = |k: usize| c[k + 1].1 - c[k].2;
    within(redial(0), 6.9, 8.2, "서버가 4초에 닫음 + 대기 3초");
    within(redial(1), 32.9, 34.2, "PING 20초 + pong 10초 + 대기 3초");
    assert_eq!(*n_snap.lock().unwrap(), 3, "접속마다 스냅숏 한 번");
    // 파일: 헤더 · 초마다 한 행(이어진 초) · 동기된 동안은 유효(mid·bin_lo) · 5초 넘게 조용하면 무효
    let files: Vec<_> = std::fs::read_dir(root.join("ETHUSDT")).unwrap().map(|e| std::fs::read(e.unwrap().path()).unwrap()).collect();
    let (mut valid, mut total) = (0, 0);
    for b in &files {
        assert_eq!(&b[..4], b"FLWR");
        assert_eq!(u16::from_le_bytes([b[6], b[7]]), 240);
        for k in 0..(b.len() - 32) / 976 {
            let r = &b[32 + k * 976..32 + (k + 1) * 976];
            if r[..8] == [0u8; 8] {
                continue;
            }
            total += 1;
            let (lo, mid) = (i32::from_le_bytes(r[8..12].try_into().unwrap()), f32::from_le_bytes(r[12..16].try_into().unwrap()));
            if mid.is_finite() {
                assert_eq!((lo, mid), (4879, 2499.995), "유효 행의 bin_lo·mid");
                let q = |j: usize| f32::from_le_bytes(r[16 + 4 * j..20 + 4 * j].try_into().unwrap());
                assert!(q(120) > 0.0 && q(121) < 0.0, "매수 + · 매도 − (2499.99 → 칸 120 · 2500.00 → 121)");
                valid += 1;
            }
        }
    }
    // 행 자리 = 벽시계 초(원본 그대로). 이 개발기(WSL)는 벽시계가 32초마다 1.5~2.4초 뒤로 뛰어 43초 동안 같은 초를 두 번 쓴다 -- 실측 39행(2026-11-21)
    assert!(total >= 35 && valid >= 6, "행 {total} · 유효 {valid}");
    eprintln!("래스터 접속 루프: 행 {total}(유효 {valid}) · 스냅숏 {} · 다시 붙기 {:.2}·{:.2}초", n_snap.lock().unwrap(), redial(0), redial(1));
    let _ = std::fs::remove_dir_all(&dir);
}

/// 래스터: 스냅숏(lastUpdateId 1000)이 스트림보다 낡으면(동기 전 U > L) **그 연결을 끊고** 물러났다 다시 붙는다(원본: IP 공유 -- 곧바로 다시 받지 않는다).
#[tokio::test]
async fn raster_stale_snapshot_backs_off() {
    let dir = tmpdir("raster_stale");
    let (rest, n_snap) = snapshot_server();
    let ev = |uu: i64, u: i64| json!({"e": "depthUpdate", "E": T0, "T": T0, "s": "ETHUSDT", "U": uu, "u": u, "pu": uu - 1, "b": [], "a": []}).to_string();
    let (port, conns, _p) = serve_paths(vec![("ethusdt@depth@500ms", vec![
        Conn { msgs: vec![ev(900, 950), ev(1100, 1110), ev(1111, 1120)], end: End::Read(Duration::from_secs(20)) },
        Conn { msgs: raster_msgs(), end: End::Read(Duration::from_secs(20)) },
    ])])
    .await;
    let env = [("RTCOLLECT_KIND", "raster".into()), ("OF_SYMBOL", "ETHUSDT".into()), ("OF_ROOT", dir.join("rs").display().to_string()), ("RTCOLLECT_REST", "1".into()),
               ("RTCOLLECT_DEPTH_URL", format!("{rest}/x?symbol={{SYMBOL}}")), ("RTCOLLECT_BAN_FILE", dir.join("ban").display().to_string()),
               ("RTCOLLECT_WS_URL", format!("ws://127.0.0.1:{port}/ws/{{symbol}}@depth@500ms"))];
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rtcollect"));
    for (k, v) in &env {
        cmd.env(k, v);
    }
    let mut child = cmd.stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
    let t0 = Instant::now();
    while conns.lock().unwrap().len() < 2 && t0.elapsed() < Duration::from_secs(8) {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let _ = Command::new("kill").args(["-TERM", &child.id().to_string()]).status();
    let _ = tokio::task::spawn_blocking(move || child.wait()).await;
    let c = conns.lock().unwrap().clone();
    assert_eq!(c.len(), 2, "낡은 스냅숏 → 끊고 다시 붙는다(8초 안 둘째 연결)");
    let redial = c[1].1 - c[0].2;
    within(redial, 2.9, 4.0, "물러남 3초");
    assert_eq!(*n_snap.lock().unwrap(), 2, "연결마다 한 번 -- 같은 연결에서 곧바로 다시 받지 않는다");
    let _ = std::fs::remove_dir_all(&dir);
}
