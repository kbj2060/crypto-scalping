//! 밀기(Push) == 묻기(Poll) 화면: 같은 체결 열을 프로세스 안 가짜 거래소 → **진짜 rtcore**(rtcore::start · 프로세스 안) 에 넣고
//!   Push = 허브 연결 하나(스냅숏 + 증분 + 체결 사건) 로 지은 feed · Poll = 같은 체결의 바이낸스 원문(직결 @trade) + 단계마다 허브 스냅숏의 풋프린트를
//!   /api/footprint 응답처럼(코어 스냅숏 = 파이썬 since=0 응답 -- rtcore/tests/footprint_parity.rs) 넣은 feed.
//!   단계마다 두 화면(Chart 장면 · 5m/30m/1h) 과 feed 상태(봉·칸·형성 봉·캔들·현재가·since)가 글자까지 같아야 한다.
//! 바깥 연결 없음: 거래소 주소는 전부 127.0.0.1 가짜 · Cfg 는 글자 그대로 적는다(from_env 기본값 = 실제 바이낸스 -- 칸이 늘면 여기서 컴파일이 깨지는 게 안전).
//! cargo test --release --test push_parity -- --nocapture
use futures_util::{SinkExt, StreamExt};
use rtui::feed::{Feed, Source, Urls};
use rtui::{Chart, default_cfg, new_ctx, scene_json};
use serde_json::{Value, json};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::{broadcast, mpsc};
use tokio::time::{sleep, timeout};
use tokio_tungstenite::tungstenite::Message;

/// 가짜 거래소: …@trade 연결에는 tx 로 넣은 원문을 그대로 보내고, 나머지(bookTicker·markPrice)는 말없이 열어 둔다.
async fn fake_exchange(tx: broadcast::Sender<String>) -> String {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap().to_string();
    tokio::spawn(async move {
        loop {
            let (tcp, _) = l.accept().await.unwrap();
            let mut rx = tx.subscribe();
            tokio::spawn(async move {
                let path = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
                let p2 = path.clone();
                let cb = move |r: &tokio_tungstenite::tungstenite::handshake::server::Request, resp| { *p2.lock().unwrap() = r.uri().path().to_string(); Ok(resp) };
                let Ok(ws) = tokio_tungstenite::accept_hdr_async(tcp, cb).await else { return };
                let (mut out, mut inp) = ws.split();
                if !path.lock().unwrap().ends_with("@trade") { while let Some(Ok(_)) = inp.next().await {} return; }
                while let Ok(t) = rx.recv().await { if out.send(Message::text(t)).await.is_err() { return; } }
            });
        }
    });
    addr
}

async fn core(ex: &str) -> String {
    let cfg = rtcore::Cfg {
        symbols: vec!["ethusdt".into()], quote_symbols: vec![], // 호가만 받는 주문 심볼(USDC) 연결은 안 연다
        ws_base: format!("ws://{ex}/ws/"),
        market_ws_base: format!("ws://{ex}/market/ws/"),
        force_url: String::new(), spot_ws_base: String::new(), okx_url: String::new(), bybit_url: String::new(),
        silence: Duration::from_secs(60),
        hub: rtcore::hub::HubCfg { path: "/rt".into(), origins: vec![], max_conns: 16, max_subs: 8, ping: Duration::from_secs(60) },
        snap_dir: String::new(), tape_db: String::new(), // 복원 끔(아래 테이프 경로도 비움 -- 디스크를 안 읽는다)
        okx_tape_db: String::new(), okx_ctx_db: String::new(), bn_ctx_db: String::new(), bybit_liq_db: String::new(),
    };
    for u in [&cfg.ws_base, &cfg.market_ws_base] { assert!(u.starts_with("ws://127.0.0.1:"), "가짜 거래소만: {u}"); }
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let hub = l.local_addr().unwrap().to_string();
    rtcore::start(&cfg, l);
    hub
}

/// 허브 연결: 구독 → 받은 글을 rx 로
async fn hub_conn(hub: &str) -> (mpsc::UnboundedReceiver<String>, tokio::task::JoinHandle<()>) {
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{hub}/rt")).await.unwrap();
    ws.send(Message::text(json!({"subscribe": ["ethusdt"]}).to_string())).await.unwrap();
    let (tx, rx) = mpsc::unbounded_channel();
    let h = tokio::spawn(async move { while let Some(Ok(m)) = ws.next().await { if let Message::Text(t) = m { if tx.send(t.to_string()).is_err() { return; } } } });
    (rx, h)
}

/// 묻기 몫: 새 연결의 스냅숏 풋프린트 = /api/footprint(since=0) 응답
async fn poll_body(hub: &str) -> String {
    let (mut rx, h) = hub_conn(hub).await;
    let t = timeout(Duration::from_secs(5), rx.recv()).await.unwrap().unwrap();
    h.abort();
    let v: Value = serde_json::from_str(&t).unwrap();
    assert_eq!(v["type"], "snapshot");
    v["footprint"].to_string()
}

fn trade_raw(id: u64, t_ms: i64, px: f64, q: f64, m: bool) -> String {
    json!({"e":"trade","E":t_ms,"T":t_ms,"s":"ETHUSDT","t":id,"p":format!("{px:.2}"),"q":format!("{q:.3}"),"X":"MARKET","m":m}).to_string()
}

/// 체결 열: t0 부터 간격 dt ms 로 n 건(가격 걸음 · 매수/매도 반반 · 정해진 시드)
struct Gen { id: u64, px: f64, seed: u64 }
impl Gen {
    fn rnd(&mut self) -> f64 { self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (self.seed >> 33) as f64 / (1u64 << 31) as f64 }
    fn run(&mut self, n: usize, t0: i64, dt: i64) -> Vec<(u64, i64, f64, f64, bool)> {
        (0..n).map(|k| {
            self.id += 1;
            if self.rnd() < 0.5 { self.px = ((self.px + (self.rnd() - 0.5) * 1.2) * 100.0).round() / 100.0; }
            let q = (self.rnd() * 2.0 * 1000.0).round() / 1000.0 + 0.001;
            (self.id, t0 + k as i64 * dt, self.px, q, self.rnd() < 0.5)
        }).collect()
    }
}

/// 두 feed 를 감싼 화면
struct Side { ch: Chart }
fn side(src: Source, hist: &str, now: f64) -> Side {
    let mut f = Feed::new("eth", "ETHUSDT", src, Urls::default(), false);
    f.on_history(Some(hist), now);
    f.ensure_ws(now, false);
    f.out.clear();
    let mut ch = Chart::new(f, default_cfg(36.0, 1751.0, 844.0, -540), eframe::egui::pos2(113.0, 126.0), eframe::egui::Color32::from_rgb(0x1e, 0x22, 0x29));
    ch.view.bars = 36.0;
    Side { ch }
}

/// 비교할 것: 장면(5m·30m·1h) + feed 상태
fn dump(s: &mut Side, ctx: &eframe::egui::Context) -> Vec<String> {
    let mut out = vec![];
    for k in [1, 6, 12] {
        s.ch.view.tf_k = k;
        let raw = eframe::egui::RawInput { screen_rect: Some(eframe::egui::Rect::from_min_size(eframe::egui::Pos2::ZERO, eframe::egui::vec2(1920.0, 1080.0))), ..Default::default() };
        let mut sc = None;
        let _ = ctx.run_ui(raw, |ui| { sc = s.ch.ui(ui, 0.0).map(|(a, h)| scene_json(&a, h.as_ref())); });
        out.push(sc.unwrap_or_else(|| "장면 없음".into()));
    }
    s.ch.view.tf_k = 1;
    let f = &mut s.ch.feed;
    let merged = f.for_chart(false).map(|(b, m)| json!([b, m]));
    let bars: Vec<Value> = f.bars_now(false).iter().map(|b| json!([b.time, b.levels, b.dhl])).collect();
    out.push(json!({"bars": bars, "merged": merged, "cells": f.live.cells.iter().collect::<Vec<_>>(), "since": f.live.since, "bar": f.live.bar_start,
                    "candles": f.candles.iter().map(|c| [c.time as f64, c.open, c.high, c.low, c.close]).collect::<Vec<_>>(), "price": f.price, "shown": f.shown, "holes": f.holes}).to_string());
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn push_equals_poll() {
    let (etx, _keep) = broadcast::channel::<String>(1 << 16);
    let ex = fake_exchange(etx.clone()).await;
    let hub = core(&ex).await;
    // 코어가 가짜 거래소 @trade 에 붙을 때까지
    timeout(Duration::from_secs(10), async { while etx.receiver_count() < 4 { sleep(Duration::from_millis(20)).await; } }).await.expect("코어가 안 붙음"); // 남겨 둔 1 + 코어 연결 3(@trade·bookTicker·markPrice)
    sleep(Duration::from_millis(300)).await;

    let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64;
    let b0 = (now_ms / 300_000 - 4) * 300_000; // 봉 시작(ms) -- 코어 봉 창(287봉) 안
    let hist = json!({"candles": (0..40).map(|i| { let t = b0 / 1000 - (37 - i) * 300; let o = 1745.0 + (i % 7) as f64 * 0.8;
        json!({"time": t, "open": o, "high": o + 2.5, "low": o - 2.0, "close": o + 0.5, "volume": 100.0 + i as f64})}).collect::<Vec<_>>()}).to_string();
    let ctx = new_ctx();
    let mut p = side(Source::Poll, &hist, b0 as f64);
    let mut q = side(Source::Push, &hist, b0 as f64);
    let (mut qrx, mut qh) = hub_conn(&hub).await;
    // 페이지가 뜨자마자 한 번 묻는다(app.js 첫 tick) -- 체결이 오기 전 고래·리테일 기준값(whaleMinUsd·retailMaxUsd)을 둘 다 안다
    let b = poll_body(&hub).await;
    p.ch.feed.on_footprint(Some(&b));
    let mut g = Gen { id: 1000, px: 1745.0, seed: 7 };
    let mut report = vec![];

    // 한 단계: 체결을 가짜 거래소와 Poll(원문) 에 같이 넣고 → Push 가 마지막 체결과 그 뒤 증분까지 받을 때까지 → Poll 은 스냅숏을 묻는다 → 비교
    macro_rules! step { ($name:expr, $trades:expr) => {{
        let tr: Vec<(u64, i64, f64, f64, bool)> = $trades;
        let last = tr.iter().filter(|t| t.2 > 0.0 && t.3 > 0.0).map(|t| t.0).max();
        for &(i, t, x, qq, m) in &tr { let raw = trade_raw(i, t, x, qq, m); etx.send(raw.clone()).unwrap(); p.ch.feed.on_ws(&raw, t as f64); }
        let mut seen = last.is_none();
        let mut n_msg = 0;
        let mut now_q = tr.last().map_or(b0, |t| t.1) as f64;
        loop {
            match timeout(Duration::from_millis(if seen { 150 } else { 5000 }), qrx.recv()).await {
                Ok(Some(t)) => {
                    let v: Value = serde_json::from_str(&t).unwrap();
                    if v["type"] == "trade" { now_q = v["ex_ms"].as_f64().unwrap(); if Some(v["id"].as_u64().unwrap()) == last { seen = true; } }
                    q.ch.feed.on_ws(&t, now_q); n_msg += 1;
                }
                Ok(None) => panic!("허브 끊김"),
                Err(_) => { assert!(seen, "{}: 5초 안에 마지막 체결이 안 옴", $name); break; }
            }
        }
        let body = poll_body(&hub).await;
        p.ch.feed.on_footprint(Some(&body));
        let (a, b) = (dump(&mut p, &ctx), dump(&mut q, &ctx));
        let names = ["장면 5m", "장면 30m", "장면 1h", "feed 상태"];
        let bad: Vec<&str> = (0..4).filter(|&k| a[k] != b[k]).map(|k| names[k]).collect();
        let nprim = serde_json::from_str::<Value>(&a[0]).ok().and_then(|v| v["prims"].as_array().map(|x| x.len())).unwrap_or(0);
        let st: Value = serde_json::from_str(&a[3]).unwrap();
        let line = format!("| {} | {} | {} | {} | {} | {} | {} |", $name, tr.len(), n_msg, st["bars"].as_array().unwrap().len(), nprim,
                           if st["merged"][1].is_null() { "서버 값" } else { "브라우저 칸" }, if bad.is_empty() { "같음".to_string() } else { format!("다름: {bad:?}") });
        println!("{line}");
        if !bad.is_empty() { for k in 0..4 { if a[k] != b[k] { let i = a[k].bytes().zip(b[k].bytes()).position(|(x, y)| x != y).unwrap_or(0);
            println!("  {}: poll …{}… / push …{}…", names[k], &a[k][i.saturating_sub(80)..(i + 120).min(a[k].len())], &b[k][i.saturating_sub(80)..(i + 120).min(b[k].len())]); } } }
        report.push((line, bad.is_empty()));
    }}}
    println!("| 단계 | 체결 | 허브 메시지 | 봉 | 장면 요소(5m) | 형성 봉 | 판정 |\n|---|---|---|---|---|---|---|");
    step!("1 봉 중간 접속(B0) 체결 120", g.run(120, b0 + 100_000, 700));
    step!("2 봉 경계 걸침 → B1 체결 150", g.run(150, b0 + 290_000, 120));
    // 3 고래·리테일: 같은 가격·같은 쪽·이어지는 id·100ms 안 = 한 주문(고래 = $100k 넘음)
    let mut w = vec![];
    for k in 0..8 { g.id += 1; w.push((g.id, b0 + 330_000 + k * 10, g.px, 9.5, false)); }
    for k in 0..6 { g.id += 1; w.push((g.id, b0 + 331_000 + k * 10, g.px - 0.5, 11.0, true)); }
    w.extend(g.run(40, b0 + 332_000, 50));
    step!("3 고래 주문 2개 + 체결 40", w);
    // 4 가격 0 섞임(바이낸스 {"p":"0","q":"0"}) -- 둘 다 버린다
    let mut z = g.run(30, b0 + 340_000, 90);
    for k in [3usize, 11, 20] { z[k].2 = 0.0; z[k].3 = 0.0; }
    step!("4 가격 0 섞인 체결 30", z);
    // 5 재연결(봉 중간): Push = 허브 끊고 새 연결(새 스냅숏) · Poll = 직결 소켓 끊고 다시(since 비움)
    qh.abort();
    let t5 = b0 + 400_000;
    for s in [&mut p, &mut q] { s.ch.feed.on_ws_close(); s.ch.feed.ensure_ws(t5 as f64, false); s.ch.feed.out.clear(); }
    (qrx, qh) = hub_conn(&hub).await;
    step!("5 재연결(B1 중간) 뒤 체결 100", g.run(100, t5, 300));
    step!("6 새 봉 B2 체결 150", g.run(150, b0 + 600_000 + 50, 200));
    step!("7 B2 이어서 체결 400(빠르게)", g.run(400, b0 + 640_000, 5));
    qh.abort();
    let ok = report.iter().filter(|r| r.1).count();
    println!("같음 {ok}/{}", report.len());
    assert_eq!(ok, report.len(), "밀기 화면 ≠ 묻기 화면");
}
