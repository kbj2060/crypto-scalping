//! ⚠️기본 웹소켓·REST 주소는 **실제 거래소**다. 개발기에서 실행하지 않는다(바이낸스는 서버와 IP 한도를 같이 쓴다 -- 밴 위험). 서버 그림자 전용.
//! ⚠️그림자: 파이썬 수집기와 다른 파일(기본 data/hot/rtcollect/…)에 쓴다 -- 파이썬 파일을 가리키면 켜지지 않는다(전환 때만 RTCOLLECT_TAKEOVER=1).
//! 절차·대조·전환: rtcollect/SHADOW.md.
//!
//! 수집기 하나를 고른다: RTCOLLECT_KIND = binance_tape(기본) | bybit_liq | bybit_tape | okx_tape | okx_ctx | binance_ctx | xyz. 종류마다 원본과 같은 이름의 환경변수:
//!   binance_tape  TAPE_SYMBOL(쉼표, 기본 ethusdt) · TAPE_MARKET(futures @trade | spot @aggTrade) · TAPE_DB_PATH
//!                 (기본 data/hot/rtcollect/binance_tape.sqlite · 현물 …/binance_spot_tape.sqlite)
//!                 RTCOLLECT_REST=1 이면 REST 1분봉 대조·aggTrades 복구도(기본 꺼짐 -- src/rest.rs) · RTCOLLECT_KLINES_URL · RTCOLLECT_AGG_URL ·
//!                 RTCOLLECT_BAN_FILE(기본 data/live/binance_ban_until = scripts/binance_ban_guard.py 와 같은 파일)
//!   bybit_liq     BYBIT_LIQ_SYMBOLS(기본 ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT,HYPEUSDT) · BYBIT_LIQ_DB(기본 data/hot/rtcollect/bybit_liq.sqlite)
//!   bybit_tape    BYBIT_TAPE_SYMBOLS(같은 기본) · BYBIT_TAPE_DB · BYBIT_CTX_DB(기본 data/hot/rtcollect/bybit_{tape,ctx}.sqlite)
//!   okx_tape      OKX_TAPE_INST(쉼표, 기본 ETH-USDT-SWAP) · OKX_TAPE_DB_PATH(기본 data/hot/rtcollect/okx_tape.sqlite)
//!                 RTCOLLECT_REST=1 이면 ctVal 대조(갈라지면 종료코드 1)·history-candles 대조·history-trades 복구(src/rest_venue.rs) · RTCOLLECT_OKX_REST
//!   bybit_tape    RTCOLLECT_REST=1 이면 v5 kline 1분봉 대조(복구 없음 -- 원본 그대로) · RTCOLLECT_BYBIT_KLINE_URL
//!   okx_ctx       OKX_CTX_INST(쉼표, 기본 ETH-USDT-SWAP -- 청산 전 종목 구독은 ETH 만) · OKX_CTX_DB_PATH(기본 data/hot/rtcollect/okx_ctx.sqlite)
//!                 RTCOLLECT_REST=1 이면 시작 때 ctVal 대조
//!   binance_ctx   BN_CTX_MARK_SYMBOLS · BN_CTX_OI_SYMBOLS · BN_CTX_OI_SLOW_SYMBOLS(원본 기본값) · BN_CTX_DB(기본 data/hot/rtcollect/binance_ctx.sqlite)
//!                 RTCOLLECT_WS_URL = 스트림 앞부분(기본 wss://fstream.binance.com/market/ws/) · RTCOLLECT_REST=1 이면 OI 폴링(RTCOLLECT_OI_URL · 밴 파일 공용)
//!   xyz           HL_XYZ_COINS(글자 그대로) · HL_XYZ_DB(기본 data/hot/rtcollect/hl_xyz.sqlite) · RTCOLLECT_REST=1 이면 접속 때 candleSnapshot 메움(RTCOLLECT_INFO_URL)
//! 공통: RTCOLLECT_WS_URL(주소 틀, 바이낸스는 {symbol} 자리 -- 시험은 가짜 서버로) · 경로는 현재 폴더 기준.
//! 시험만: RTCOLLECT_SILENCE_S(바이낸스 침묵 45) · RTCOLLECT_WD_LOOP_S·RTCOLLECT_WD_WRITE_S(정지 감시 120·600) · RTCOLLECT_WD_TRADE_S(Bybit 테이프 600).
//! 종료코드(원본 그대로): 바이낸스·OKX 정지 감시 70 · 수집 태스크가 끝나면 70 · Bybit 테이프 체결 멈춤 3 · OKX 모르는 종목(ctVal) 1 ·
//!   설정 오류 2 · SIGTERM/SIGINT 0(원본이 연결 끝에 하던 쓰기를 하고 끝낸다 -- 원본은 안 쓰고 죽는다).
//!   맥락: OKX 맥락은 바이낸스처럼 정지 감시 70 · 바이낸스 맥락·xyz·HL 맥락은 정지 감시 없음(원본 그대로) · 종료 때 남은 행을 쓴다.
use rtcollect::net::{self, EXIT_STALL};
use rtcollect::rest::{wall_s, Clock, Rest, TIMEOUT};
use rtcollect::store::open_db;
use rtcollect::{binance, binance_ctx, bybit_liq, bybit_tape, depth, hl_ctx, okx, okx_ctx, raster, rest_venue, xyz};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::signal::unix::{signal, SignalKind};
use tokio::task::JoinSet;

const KINDS: &str = "binance_tape|bybit_liq|bybit_tape|okx_tape|okx_ctx|binance_ctx|xyz|hl_ctx|hl_bt|depth|raster|ob5m|bookdepth|iceberg|treasury_futures|macro_5m|cboe_chain|liqmap_scorer|soxl_weekend|news";

fn env_f(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn env_s(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.into())
}

fn die(code: i32, msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(code)
}

/// 쉼표 목록(원본 `[s.strip().upper() for s in …split(",") if s.strip()]` · 바이낸스는 lower).
fn list(k: &str, d: &str, upper: bool) -> Vec<String> {
    env_s(k, d).split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).map(|s| if upper { s.to_uppercase() } else { s.to_lowercase() }).collect()
}

/// 기본 data/hot/rtcollect/<file>. 🔴파이썬 수집기의 파일(data/hot/<같은 이름>)이면 그림자 기록이 섞인다(gaps·meta 두 벌 · 전진 검정 자료 오염)
/// -- 전환 때만 RTCOLLECT_TAKEOVER=1 로 연다.
fn db_path(var: &str, file: &str) -> PathBuf {
    let p = std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("data/hot/rtcollect").join(file));
    let python = p.file_name() == Some(file.as_ref()) && p.parent().and_then(|d| d.file_name()) == Some("hot".as_ref());
    if python && env_s("RTCOLLECT_TAKEOVER", "") != "1" {
        die(2, &format!("🔴{} 는 파이썬 수집기의 파일이다 -- 그림자는 다른 경로에(기본 data/hot/rtcollect/). 전환이면 RTCOLLECT_TAKEOVER=1", p.display()));
    }
    p
}

/// SIGTERM·SIGINT 까지. 수집 태스크 하나라도 끝나면(= 죽으면) 프로세스를 끝낸다(원본 run_all -- 감독이 전부 다시 띄운다). 60초마다 `tick`.
async fn wait_term(tasks: &mut JoinSet<()>, tick: impl Fn()) {
    let (mut term, mut int) = (signal(SignalKind::terminate()).unwrap(), signal(SignalKind::interrupt()).unwrap());
    let mut every = tokio::time::interval(Duration::from_secs(60));
    every.tick().await;
    loop {
        tokio::select! {
            _ = term.recv() => return,
            _ = int.recv() => return,
            r = tasks.join_next() => die(EXIT_STALL, &format!("수집 태스크 종료: {r:?} -- 감시자가 다시 띄운다")),
            _ = every.tick() => tick(),
        }
    }
}

fn main() {
    let kind = env_s("RTCOLLECT_KIND", "binance_tape");
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, async move {
        match kind.as_str() {
            "binance_tape" => run_binance().await,
            "bybit_liq" => run_bybit_liq().await,
            "bybit_tape" => run_bybit_tape().await,
            "okx_tape" => run_okx().await,
            "okx_ctx" => run_okx_ctx().await,
            "binance_ctx" => run_binance_ctx().await,
            "xyz" => run_xyz().await,
            "hl_ctx" => run_hl_ctx().await,
            "hl_bt" => run_hl_bt().await,
            "depth" => run_depth().await,
            "raster" => run_raster().await,
            "ob5m" | "bookdepth" => run_book_worker(&kind),
            "iceberg" => run_iceberg(),
            "treasury_futures" | "macro_5m" | "cboe_chain" => run_cron(&kind),
            "liqmap_scorer" => run_liqmap(),
            "soxl_weekend" => run_soxl_weekend(),
            "news" => run_news(),
            k => die(2, &format!("RTCOLLECT_KIND={k} -- {KINDS}")),
        }
    });
}

/// 바이낸스 체결 테이프(live_trade_tape_collector_20260916.py `main()`·`run_all()`).
async fn run_binance() {
    let syms = list("TAPE_SYMBOL", "ethusdt", false);
    let market = env_s("TAPE_MARKET", "futures");
    let (url, file, rest) = match market.as_str() {
        "futures" => ("wss://fstream.binance.com/ws/{symbol}@trade", "binance_tape.sqlite", "https://fapi.binance.com/fapi/v1"),
        "spot" => ("wss://stream.binance.com:9443/ws/{symbol}@aggTrade", "binance_spot_tape.sqlite", "https://api.binance.com/api/v3"),
        m => die(2, &format!("TAPE_MARKET={m} -- futures|spot")),
    };
    let url = env_s("RTCOLLECT_WS_URL", url);
    let db = db_path("TAPE_DB_PATH", file);
    let silence = Duration::from_secs_f64(env_f("RTCOLLECT_SILENCE_S", 45.0));
    net::start_watchdog(env_f("RTCOLLECT_WD_LOOP_S", net::WD_LOOP_STALL_S), env_f("RTCOLLECT_WD_WRITE_S", net::WD_WRITE_STALL_S));
    let con = open_db(&db).expect("DB 열기");
    let mut cs = vec![];
    let mut tasks = JoinSet::new();
    for s in &syms {
        let c = Rc::new(RefCell::new(binance::Collector::new(con.clone(), s, &market, net::mono_s()).expect("TapeStore")));
        println!("{s} {market} 수집 시작 (빈 {}, db {})", rtcollect::tape::bucket(s), db.display());
        tasks.spawn_local(net::run_symbol(c.clone(), url.replace("{symbol}", s), silence));
        cs.push(c);
    }
    tasks.spawn_local(net::beat());
    if env_s("RTCOLLECT_REST", "") == "1" {
        // 꺼 두는 게 기본: 그림자 기간엔 파이썬이 같은 REST 를 이미 부른다. 심볼마다 스레드 · 속도 슬롯은 프로세스 하나.
        let (ku, au) = (env_s("RTCOLLECT_KLINES_URL", &format!("{rest}/klines")), env_s("RTCOLLECT_AGG_URL", &format!("{rest}/aggTrades")));
        let (clock, pace) = (Arc::new(Clock::real()), Arc::new(Mutex::new(0.0)));
        let ban = PathBuf::from(env_s("RTCOLLECT_BAN_FILE", "data/live/binance_ban_until"));
        for s in &syms {
            let r = Rest::new(&db, s, &ku, &au, clock.clone(), pace.clone(), ban.clone(), TIMEOUT).expect("REST 용 DB 연결");
            std::thread::spawn(move || r.run());
        }
        println!("REST 검증·복구 켬 ({ku} · {au})");
    }
    wait_term(&mut tasks, || {
        let st: Vec<String> = cs.iter().map(|c| { let c = c.borrow(); format!("{} 보류 {} 늦은버림 {}", c.store.symbol, c.store.pending.len(), c.buffer.dropped_late) }).collect();
        println!("rtcollect 60s · 재연결 {} (누적) · {}", net::RECONNECTS.load(Relaxed), st.join(" · "));
    })
    .await;
    // 닫힌 초만 쓰고 끝낸다(진행 중인 초는 버린다 = 다음 판 gaps 가 덮는다). 값은 원본이 그 순간 연결이 끊겼을 때 쓰는 것과 같다.
    let n: usize = cs.iter().map(|c| c.borrow_mut().on_disconnect()).sum();
    println!("종료 신호 -- 닫힌 초 {n}행 쓰고 끝낸다");
}

/// OKX 체결 테이프(live_okx_trade_tape_collector_20260923.py `main()` -- `_bn.run_all` 이라 정지 감시도 바이낸스와 같다).
async fn run_okx() {
    let insts = list("OKX_TAPE_INST", "ETH-USDT-SWAP", true);
    let url = env_s("RTCOLLECT_WS_URL", "wss://ws.okx.com:8443/ws/v5/public");
    let db = db_path("OKX_TAPE_DB_PATH", "okx_tape.sqlite");
    net::start_watchdog(env_f("RTCOLLECT_WD_LOOP_S", net::WD_LOOP_STALL_S), env_f("RTCOLLECT_WD_WRITE_S", net::WD_WRITE_STALL_S));
    let con = open_db(&db).expect("DB 열기");
    let mut cs = vec![];
    let mut tasks = JoinSet::new();
    for i in &insts {
        let Some(c) = okx::Collector::new(con.clone(), i, net::mono_s()).expect("TapeStore") else {
            die(1, &format!("🔴{i} 의 ctVal 을 모른다 -- CT_VALS 에 실측값을 적고 다시 돌릴 것")) // 원본 SystemExit
        };
        println!("{i} 수집 시작 (빈 {}, db {})", okx::bucket(i), db.display());
        let c = Rc::new(RefCell::new(c));
        tasks.spawn_local(net::run_okx(c.clone(), url.clone()));
        cs.push(c);
    }
    if env_s("RTCOLLECT_REST", "") == "1" {
        // 꺼 두는 게 기본(그림자 동안 파이썬이 같은 REST 를 부른다). ctVal 대조 → 종목마다 복구·대조 스레드 · 속도 슬롯은 프로세스 하나(원본 _PACE).
        let (base, clock, pace) = (env_s("RTCOLLECT_OKX_REST", rest_venue::BASE), Arc::new(Clock::real()), Arc::new(Mutex::new(0.0)));
        for i in &insts {
            rest_venue::assert_ct_val(&base, i);
            let r = Rest::okx(&db, i, &base, clock.clone(), pace.clone(), rest_venue::TIMEOUT).expect("REST 용 DB 연결");
            std::thread::spawn(move || r.run());
        }
        println!("REST 대조·복구 켬 ({base})");
    }
    tasks.spawn_local(net::beat());
    wait_term(&mut tasks, || println!("rtcollect okx 60s · 재연결 {} (누적)", net::RECONNECTS.load(Relaxed))).await;
    let n: usize = cs.iter().map(|c| c.borrow_mut().on_disconnect()).sum();
    println!("종료 신호 -- 닫힌 초 {n}행 쓰고 끝낸다");
}

/// OKX 맥락(live_okx_context_collector_20260923.py `main()` -- `_bn.run_all` 이라 정지 감시도 바이낸스와 같다). 종목들은 파일·연결 하나.
async fn run_okx_ctx() {
    let insts = list("OKX_CTX_INST", "ETH-USDT-SWAP", true);
    let url = env_s("RTCOLLECT_WS_URL", "wss://ws.okx.com:8443/ws/v5/public");
    let db = db_path("OKX_CTX_DB_PATH", "okx_ctx.sqlite");
    net::start_watchdog(env_f("RTCOLLECT_WD_LOOP_S", net::WD_LOOP_STALL_S), env_f("RTCOLLECT_WD_WRITE_S", net::WD_WRITE_STALL_S));
    let con = okx_ctx::open_db(&db).expect("DB 열기");
    let rest = env_s("RTCOLLECT_REST", "") == "1";
    let mut cs = vec![];
    let mut tasks = JoinSet::new();
    for i in &insts {
        let Some(c) = okx_ctx::CtxCollector::new(con.clone(), i, net::mono_s(), wall_s()).expect("meta") else {
            die(1, &format!("🔴{i} 의 ctVal 을 모른다 -- CT_VALS 에 실측값을 적고 다시 돌릴 것")) // 원본 SystemExit
        };
        if rest {
            rest_venue::assert_ct_val(&env_s("RTCOLLECT_OKX_REST", rest_venue::BASE), i); // 갈라졌으면 종료코드 1
        }
        println!("{i} 컨텍스트 수집 시작 (db {})", db.display());
        let c = Rc::new(RefCell::new(c));
        tasks.spawn_local(net::run_okx(c.clone(), url.clone()));
        cs.push(c);
    }
    tasks.spawn_local(net::beat());
    wait_term(&mut tasks, || println!("rtcollect okx_ctx 60s · 재연결 {} (누적)", net::RECONNECTS.load(Relaxed))).await;
    let n: usize = cs.iter().map(|c| c.borrow_mut().flush()).sum();
    println!("종료 신호 -- 보류 {n}행 쓰고 끝낸다");
}

/// 바이낸스 맥락(live_binance_ctx_collector_20261001.py `run()`): 마크(종목마다 WS) · 강제청산(WS 하나) · OI(REST, 기본 꺼짐) → 10초마다 쓰기.
/// 정지 감시 없음(원본 그대로).
async fn run_binance_ctx() {
    let oi = list("BN_CTX_OI_SYMBOLS", binance_ctx::OI_SYMBOLS, true);
    let slow = list("BN_CTX_OI_SLOW_SYMBOLS", binance_ctx::OI_SLOW_SYMBOLS, true);
    let marks = list("BN_CTX_MARK_SYMBOLS", binance_ctx::MARK_SYMBOLS, false);
    let base = env_s("RTCOLLECT_WS_URL", binance_ctx::WS_BASE);
    let db = db_path("BN_CTX_DB", "binance_ctx.sqlite");
    let con = binance_ctx::open_db(&db).expect("DB 열기");
    let p: binance_ctx::Shared = Arc::new(Mutex::new(binance_ctx::Pending::default()));
    let mut tasks = JoinSet::new();
    for s in &marks {
        let st = binance_ctx::Stream::new(binance_ctx::Feed::Mark, Some(s.clone()), p.clone());
        tasks.spawn_local(net::run_aio(format!("{base}{s}@markPrice@1s"), format!("mark_price_1s {s}"), st));
    }
    let st = binance_ctx::Stream::new(binance_ctx::Feed::Force(binance_ctx::liq_symbols(&slow)), None, p.clone());
    tasks.spawn_local(net::run_aio(format!("{base}!forceOrder@arr"), "liquidations".into(), st));
    if env_s("RTCOLLECT_REST", "") == "1" {
        // 꺼 두는 게 기본: 그림자 기간엔 파이썬이 같은 OI 를 이미 폴링한다(켜면 같은 IP 에서 호출 두 배 -- SHADOW.md).
        let url = env_s("RTCOLLECT_OI_URL", binance_ctx::OI_URL);
        let (clock, ban) = (Arc::new(Clock::real()), PathBuf::from(env_s("RTCOLLECT_BAN_FILE", "data/live/binance_ban_until")));
        for (s, period) in oi.iter().map(|s| (s, binance_ctx::OI_POLL_S)).chain(slow.iter().map(|s| (s, binance_ctx::OI_SLOW_POLL_S))) {
            let r = binance_ctx::OiPoll::new(s, period, &url, clock.clone(), ban.clone(), p.clone(), binance_ctx::REST_TIMEOUT);
            std::thread::spawn(move || r.run());
        }
        println!("OI REST 폴링 켬 ({url})");
    }
    println!("시작 -- OI {} · 마크 {marks:?} · 청산 {:?} -> {}", if env_s("RTCOLLECT_REST", "") == "1" { format!("{oi:?} {slow:?}") } else { "꺼짐".into() },
             { let mut v: Vec<_> = binance_ctx::liq_symbols(&slow).into_iter().collect(); v.sort(); v }, db.display());
    let (pw, cw) = (p.clone(), Rc::new(con));
    let c2 = cw.clone();
    tasks.spawn_local(async move {
        loop {
            tokio::time::sleep(Duration::from_secs_f64(binance_ctx::FLUSH_S)).await;
            binance_ctx::flush(&c2, &pw);
        }
    });
    wait_term(&mut tasks, || println!("rtcollect binance_ctx 60s · 재연결 {} (누적) · 보류 {}", net::RECONNECTS.load(Relaxed), p.lock().unwrap().len())).await;
    let n = binance_ctx::flush(&cw, &p); // 원본 finally: 종료(재기동) 때 보류분을 버리지 않는다
    println!("종료 신호 -- 보류 {n}행 쓰고 끝낸다");
}

/// xyz100(live_xyz100_collector_20261014.py `run()`). 정지 감시 없음(원본 그대로). REST 메움은 RTCOLLECT_REST=1 일 때만.
async fn run_xyz() {
    // 원본 os.environ.get(...).split(",") -- 공백을 벗기지 않는다(글자 그대로)
    let coins: Vec<String> = env_s("HL_XYZ_COINS", xyz::COINS).split(',').map(str::to_string).collect();
    let db = db_path("HL_XYZ_DB", "hl_xyz.sqlite");
    let url = env_s("RTCOLLECT_WS_URL", xyz::WS_URL);
    let c = Rc::new(RefCell::new(xyz::XyzCollector::new(xyz::open_db(&db).expect("DB 열기"), &coins)));
    let rest = env_s("RTCOLLECT_REST", "") == "1";
    let info = env_s("RTCOLLECT_INFO_URL", xyz::INFO_URL);
    let agent = rtcollect::rest::agent(Duration::from_secs(20)); // urlopen(timeout=20)
    let cb = c.clone();
    // ponytail: 메움(최대 종목 수 × 20초)이 이 스레드를 막는다 -- 원본도 메움을 기다린 뒤에 접속한다(그동안 수신 없음).
    let before = move || if rest { cb.borrow().backfill(&agent, &info, wall_s()); };
    println!("xyz100 {coins:?} → {} (REST 메움 {})", db.display(), if rest { "켬" } else { "꺼짐" });
    let mut tasks = JoinSet::new();
    let cx = c.clone();
    tasks.spawn_local(net::run_hl(c, url, before));
    wait_term(&mut tasks, || println!("rtcollect xyz 60s · 재연결 {} (누적)", net::RECONNECTS.load(Relaxed))).await;
    // 🔧원본은 끊길 때·종료 때 마지막 플러시 뒤 행을 버린다 -- 여기는 쓴다(SHADOW.md §11).
    let n = cx.borrow_mut().write_rest();
    println!("종료 신호 -- 남은 {n}행 쓰고 끝낸다");
}

/// 하이퍼리퀴드 자산 맥락(live_hyperliquid_book_ticker_collector_20260923.py · HL_BT_CTX_ONLY=1). 코인마다 WS 하나(원본은 코인마다 프로세스).
/// 정지 감시 없음(원본 그대로). 종료 때 남은 행을 쓴다(원본이 연결 끝에 하는 쓰기).
async fn run_hl_ctx() {
    let coins = list("HL_BT_COIN", "ETH", true);
    let db = db_path("HL_CTX_DB_PATH", "hl_ctx.sqlite");
    let url = env_s("RTCOLLECT_WS_URL", hl_ctx::WS_URL);
    let _meta = hl_ctx::open_db(&db).expect("DB 열기"); // 표·meta -- 행은 코인마다 쓰기 스레드
    let mut cs = vec![];
    let mut tasks = JoinSet::new();
    for c in &coins {
        let c = Rc::new(RefCell::new(hl_ctx::HlCtx::new(&db, c, net::mono_s())));
        tasks.spawn_local(net::run_hl(c.clone(), url.clone(), || {}));
        cs.push(c);
    }
    println!("hl_ctx {coins:?} → {}", db.display());
    wait_term(&mut tasks, || println!("rtcollect hl_ctx 60s · 재연결 {} (누적)", net::RECONNECTS.load(Relaxed))).await;
    let n: usize = cs.iter().map(|c| { let mut c = c.borrow_mut(); let n = c.write(); c.finish(); n }).sum();
    println!("종료 신호 -- 남은 {n}행 쓰고 끝냈다");
}

/// 하이퍼리퀴드 호가 + 맥락 전체(live_hyperliquid_book_ticker_collector_20260923.py 기본 형태 -- Pi 에서 도는 것). 코인 하나(원본과 같다).
/// bbo → HL_BT_ROOT/<COIN>/<UTC 시>.bt(기본 data/live/rtcollect/orderflow/hyperliquid_bookticker -- 파이썬 폴더면 RTCOLLECT_TAKEOVER=1 없이는 안 켠다) · 맥락 → HL_CTX_DB_PATH.
async fn run_hl_bt() {
    let coin = env_s("HL_BT_COIN", "ETH").to_uppercase();
    let root = PathBuf::from(env_s("HL_BT_ROOT", "data/live/rtcollect/orderflow/hyperliquid_bookticker"));
    if root.ends_with("data/live/orderflow/hyperliquid_bookticker") && env_s("RTCOLLECT_TAKEOVER", "") != "1" {
        die(2, &format!("🔴{} 는 파이썬 수집기의 폴더다 -- 그림자는 다른 경로에. 전환이면 RTCOLLECT_TAKEOVER=1", root.display()));
    }
    let db = db_path("HL_CTX_DB_PATH", "hl_ctx.sqlite");
    let url = env_s("RTCOLLECT_WS_URL", hl_ctx::WS_URL);
    let _meta = hl_ctx::open_db(&db).expect("DB 열기");
    let mut h = hl_ctx::HlCtx::new(&db, &coin, net::mono_s());
    h.bt = Some(rtcollect::bt::HourFile::new(&root, &coin, net::mono_s()).expect(".bt 폴더"));
    let c = Rc::new(RefCell::new(h));
    let mut tasks = JoinSet::new();
    tasks.spawn_local(net::run_hl(c.clone(), url, || {}));
    println!("hl_bt {coin} → {} · ctx → {}", root.display(), db.display());
    wait_term(&mut tasks, || { let c = c.borrow(); println!("rtcollect hl_bt 60s · bbo {} · 버림 {} · 재연결 {}", c.n_bbo, c.dropped, net::RECONNECTS.load(Relaxed)) }).await;
    let mut c = c.borrow_mut();
    c.on_disconnect();
    c.finish();
    println!("종료 신호 -- .bt 닫고(압축 기다림) 남은 맥락 행 쓰고 끝냈다");
}

/// depth_diff(live_depth_diff_collector_20260915.py `run()`). 종목 하나(원본과 같다 -- DD_SYMBOL). 정지 감시 없음(원본 그대로).
/// 🔴기본 폴더 = data/live/rtcollect/orderflow/depthdiff -- 파이썬 폴더(data/live/orderflow/depthdiff)면 켜지지 않는다(전환 때만 RTCOLLECT_TAKEOVER=1).
async fn run_depth() {
    let sym = env_s("DD_SYMBOL", "ethusdt").to_lowercase();
    let root = PathBuf::from(env_s("DD_ROOT", "data/live/rtcollect/orderflow/depthdiff"));
    if root.ends_with("data/live/orderflow/depthdiff") && env_s("RTCOLLECT_TAKEOVER", "") != "1" {
        die(2, &format!("🔴{} 는 파이썬 수집기의 폴더다 -- 그림자는 다른 경로에. 전환이면 RTCOLLECT_TAKEOVER=1", root.display()));
    }
    let url = env_s("RTCOLLECT_WS_URL", depth::WS).replace("{symbol}", &sym);
    let c = Rc::new(RefCell::new(depth::DepthCollector::new(&root, &sym).expect("폴더")));
    let rest = env_s("RTCOLLECT_REST", "") == "1";
    let rest_url = env_s("RTCOLLECT_DEPTH_URL", depth::REST).replace("{SYM}", &sym.to_uppercase());
    let ban = PathBuf::from(env_s("RTCOLLECT_BAN_FILE", "data/live/binance_ban_until"));
    let agent = rtcollect::rest::agent(Duration::from_secs(10)); // urlopen(timeout=10)
    // 새 시 파일마다 REST 스냅숏 1회(가중치 20). 끄면 null(원본이 실패 때 쓰는 줄). 🔧밴 중이면 안 보내고 418·429 는 공용 밴 파일에(원본은 모른다).
    let snap = move || -> Option<serde_json::Value> {
        if !rest || rtcollect::rest::ban_left(&ban, wall_s()) > 0.0 {
            return None;
        }
        let mut r = agent.get(&rest_url).call().map_err(|e| eprintln!("스냅샷 실패: {e} -- diff 만 적는다")).ok()?;
        let st = r.status().as_u16();
        let ra = r.headers().get("retry-after").and_then(|v| v.to_str().ok()).map(str::to_string);
        let body = r.body_mut().read_to_string().ok()?;
        rtcollect::rest::ban_note(&ban, st, &body, ra.as_deref(), wall_s());
        if !(200..300).contains(&st) {
            eprintln!("스냅샷 실패: HTTP {st} -- diff 만 적는다");
            return None;
        }
        serde_json::from_str(&body).map_err(|e| eprintln!("스냅샷 실패: {e} -- diff 만 적는다")).ok()
    };
    println!("depth {sym} → {} (스냅숏 REST {})", root.display(), if rest { "켬" } else { "꺼짐 = 첫 줄 null" });
    let mut tasks = JoinSet::new();
    tasks.spawn_local(net::run_depth(c.clone(), url, snap));
    wait_term(&mut tasks, || { let c = c.borrow(); println!("rtcollect depth 60s · 재연결 {} · 프레임 {} · 구멍 {}", net::RECONNECTS.load(Relaxed), c.frames, c.gaps) }).await;
    c.borrow_mut().out.close(false);
    c.borrow_mut().out.join();
    println!("종료 신호 -- 파일 닫고 끝낸다");
}

/// 호가 래스터(live_orderflow_raster_collector_20260914.py). 종목 하나(OF_SYMBOL). 🔴스냅숏 REST 없이는 북을 못 세운다 -- RTCOLLECT_REST=1 필수
/// (시간당 2회 안팎 · 가중치 20). 기본 폴더 data/live/rtcollect/orderflow/raster -- 파이썬 폴더면 켜지지 않는다. 보관·지우기는 하지 않는다.
async fn run_raster() {
    let sym = env_s("OF_SYMBOL", "ethusdt").to_lowercase();
    let root = PathBuf::from(env_s("OF_ROOT", "data/live/rtcollect/orderflow/raster"));
    if root.ends_with("data/live/orderflow/raster") && env_s("RTCOLLECT_TAKEOVER", "") != "1" {
        die(2, &format!("🔴{} 는 파이썬 수집기의 폴더다 -- 그림자는 다른 경로에. 전환이면 RTCOLLECT_TAKEOVER=1", root.display()));
    }
    let url = env_s("RTCOLLECT_WS_URL", raster::WS).replace("{symbol}", &sym);
    let rest = env_s("RTCOLLECT_REST", "") == "1";
    let snap_url = env_s("RTCOLLECT_DEPTH_URL", raster::SNAPSHOT).replace("{SYMBOL}", &sym.to_uppercase());
    let ban = PathBuf::from(env_s("RTCOLLECT_BAN_FILE", "data/live/binance_ban_until"));
    let agent = rtcollect::rest::agent(Duration::from_secs(10)); // ClientTimeout(total=10)
    // _fetch_snapshot: raise_for_status · json(content_type=None). 🔧밴 중이면 안 보내고 418·429 는 공용 밴 파일에(원본은 모른다).
    let fetch: Arc<dyn Fn() -> Result<serde_json::Value, String> + Send + Sync> = Arc::new(move || {
        if !rest {
            return Err("REST 꺼짐(RTCOLLECT_REST=1)".into());
        }
        if rtcollect::rest::ban_left(&ban, wall_s()) > 0.0 {
            return Err("binance IP ban -- 안 보냄".into());
        }
        let mut r = agent.get(&snap_url).call().map_err(|e| e.to_string())?;
        let st = r.status().as_u16();
        let ra = r.headers().get("retry-after").and_then(|v| v.to_str().ok()).map(str::to_string);
        let body = r.body_mut().read_to_string().map_err(|e| e.to_string())?;
        rtcollect::rest::ban_note(&ban, st, &body, ra.as_deref(), wall_s());
        if st >= 400 {
            return Err(format!("ClientResponseError: {st}"));
        }
        serde_json::from_str(&body).map_err(|e| e.to_string())
    });
    // 원본과 같은 환경변수(:65-66) -- 종목마다 칸 크기가 다르다(crontab: xrp 0.0002 · sol 0.02 · mu 0.2 …)
    let bin = env_f("OF_BIN_SIZE", raster::BIN_SIZE);
    let n: usize = env_s("OF_N_BINS", &raster::N_BINS.to_string()).parse().unwrap_or_else(|_| die(2, "OF_N_BINS 는 정수"));
    if !(bin > 0.0) || n == 0 || n > u16::MAX as usize {
        die(2, &format!("OF_BIN_SIZE {bin} · OF_N_BINS {n} -- 양수(칸 수 ≤ 65535)여야 한다"));
    }
    let r = Rc::new(RefCell::new(raster::Raster::new(&root, &sym, bin, n)));
    println!("raster {sym} 칸 {bin} × {n} → {} (스냅숏 REST {})", root.display(), if rest { "켬" } else { "🔴꺼짐 = 모든 행 무효" });
    let mut tasks = JoinSet::new();
    tasks.spawn_local(net::run_raster(r.clone(), url, fetch));
    tasks.spawn_local(net::raster_ticks(r.clone()));
    wait_term(&mut tasks, || { let r = r.borrow(); println!("rtcollect raster 60s · 행 {}(무효 {}) · 호가 {}/{}", r.rows_written, r.rows_invalid, r.bids.len(), r.asks.len()) }).await;
    println!("종료 신호 -- 끝낸다");
}

/// ob5m(live_ob5m_20261026.py) · 호가 깊이 10열(live_bookdepth_feat_20261028.py): depthdiff 시 파일을 이어 읽는 워커(웹소켓·REST 없음 · 막는 루프).
/// 입력은 읽기만(DD_ROOT/<SYM> · hot 테이프) · 출력 기본 data/live/rtcollect/{ob5m,bookdepth}/<sym>.sqlite -- 파이썬 파일이면 켜지지 않는다.
fn run_book_worker(kind: &str) {
    let sym = env_s("OB_SYMBOL", "ETHUSDT").to_uppercase();
    if rtcollect::ob5m::coin(&sym).is_none() || (kind == "bookdepth" && sym != "ETHUSDT") {
        die(2, &format!("OB_SYMBOL={sym} -- 원본이 아는 코인만(bookdepth 는 ETHUSDT 만)"));
    }
    let dd = PathBuf::from(env_s("DD_ROOT", "data/live/orderflow/depthdiff")).join(&sym);
    let file = format!("{}.sqlite", sym.to_lowercase());
    let db = PathBuf::from(env_s("OB_DB", &format!("data/live/rtcollect/{kind}/{file}")));
    if db.ends_with(format!("data/live/{kind}/{file}")) && env_s("RTCOLLECT_TAKEOVER", "") != "1" {
        die(2, &format!("🔴{} 는 파이썬 워커의 파일이다 -- 그림자는 다른 경로에. 전환이면 RTCOLLECT_TAKEOVER=1", db.display()));
    }
    // RTCOLLECT_FAKE_NOW=<초>: 「같은 시작점 재생 대조」(tools/replay_book.py) 전용 -- 시계를 그 순간에 멈추고, 기다려야 하면(파일 끝) 끝낸다
    let fake: Option<f64> = std::env::var("RTCOLLECT_FAKE_NOW").ok().map(|v| v.parse().unwrap_or_else(|_| die(2, "RTCOLLECT_FAKE_NOW 는 초(실수)")));
    if fake.is_some() && db.ends_with(format!("data/live/{kind}/{file}")) {
        die(2, "🔴RTCOLLECT_FAKE_NOW 는 대조용 -- 진짜 DB 에 쓰지 않는다");
    }
    let now = move || fake.unwrap_or_else(wall_s);
    let sleep = move |s: f64| {
        if fake.is_some() {
            return false;
        }
        std::thread::sleep(Duration::from_secs_f64(s));
        true
    };
    // ponytail: SIGTERM 은 기본 처리(바로 끝남) -- 원본도 그렇다 · 봉은 쓸 때마다 커밋되니 잃는 것은 진행 중 봉뿐(다음 판이 재생으로 다시 만든다)
    let r = if kind == "ob5m" {
        let tape = PathBuf::from(env_s("TAPE_DB_PATH", "data/hot/binance_tape.sqlite"));
        rtcollect::ob5m::run(&sym, &dd, &tape, &db, &now, &sleep, &|_| {})
    } else {
        rtcollect::bookdepth::run(&sym, &dd, &db, &now, &sleep, &|_| {})
    };
    if let Err(e) = r {
        die(70, &format!("{kind} 끝남: {e}"));
    }
}

/// Bybit 청산(live_bybit_liquidation_collector_20261003.py `run()`). 정지 감시 없음(원본 그대로).
async fn run_bybit_liq() {
    let syms = list("BYBIT_LIQ_SYMBOLS", bybit_liq::SYMBOLS, true);
    let db = db_path("BYBIT_LIQ_DB", "bybit_liq.sqlite");
    let url = env_s("RTCOLLECT_WS_URL", "wss://stream.bybit.com/v5/public/linear");
    let c = Rc::new(RefCell::new(bybit_liq::LiqCollector::new(bybit_liq::open_db(&db).expect("DB 열기"))));
    let sub = rtcollect::py_subscribe(&syms.iter().map(|s| serde_json::Value::from(format!("allLiquidation.{s}")).to_string()).collect::<Vec<_>>());
    println!("구독 {} → {}", syms.join(","), db.display());
    let mut tasks = JoinSet::new();
    tasks.spawn_local(net::run_bybit(c.clone(), url, sub));
    wait_term(&mut tasks, || println!("rtcollect bybit_liq 60s · 재연결 {} (누적)", net::RECONNECTS.load(Relaxed))).await;
    // 원본이 연결이 끊길 때 하는 쓰기(남은 행) -- 끊긴 시각은 프로세스와 함께 사라진다(다음 판은 공백을 모른다 -- 원본도 같다).
    c.borrow_mut().on_error(wall_s(), "SIGTERM");
    println!("종료 신호 -- 남은 행 쓰고 끝낸다");
}

/// Bybit 체결 테이프 + 맥락(live_bybit_trade_tape_collector_20261006.py `run()`). 체결이 600초 없으면 종료코드 3.
async fn run_bybit_tape() {
    let syms = list("BYBIT_TAPE_SYMBOLS", bybit_tape::SYMBOLS, true);
    let (tape, ctx) = (db_path("BYBIT_TAPE_DB", "bybit_tape.sqlite"), db_path("BYBIT_CTX_DB", "bybit_ctx.sqlite"));
    let url = env_s("RTCOLLECT_WS_URL", "wss://stream.bybit.com/v5/public/linear");
    net::start_trade_guard(env_f("RTCOLLECT_WD_TRADE_S", bybit_tape::STALL_S));
    let c = bybit_tape::BybitCollector::new(&tape, &ctx, &syms, net::mono_s()).expect("DB 열기");
    if env_s("RTCOLLECT_REST", "") == "1" {
        // 1분봉 대조만(원본에 복구 없음). 기본 꺼짐.
        let (ku, clock) = (env_s("RTCOLLECT_BYBIT_KLINE_URL", rest_venue::BYBIT_KLINE_URL), Arc::new(Clock::real()));
        for s in &syms {
            let r = Rest::bybit(&tape, s, &ku, clock.clone(), rest_venue::TIMEOUT).expect("REST 용 DB 연결");
            std::thread::spawn(move || r.run());
        }
        println!("REST 1분봉 대조 켬 ({ku})");
    }
    let sub = c.subscribe();
    println!("구독 {} → {} · {}", syms.join(","), tape.display(), ctx.display());
    let c = Rc::new(RefCell::new(c));
    let mut tasks = JoinSet::new();
    tasks.spawn_local(net::run_bybit(c.clone(), url, sub));
    wait_term(&mut tasks, || println!("rtcollect bybit_tape 60s · 재연결 {} (누적)", net::RECONNECTS.load(Relaxed))).await;
    // 원본이 연결이 끊길 때 하는 쓰기(flush(True): 열린 묶음·진행 중인 초까지) -- 다음 판은 last_ts_ms 부터 공백을 적는다.
    c.borrow_mut().on_error(net::mono_s(), wall_s(), "SIGTERM");
    println!("종료 신호 -- 남은 행 쓰고 끝낸다");
}

/// 칸별 아이스버그(live_eth_iceberg_20261016.py main()): 2초마다 한 바퀴. ICEBERG_SYMBOL(ETHUSDT·XRPUSDT) · DD_ROOT(depthdiff 읽기만) · TAPE_DB_PATH(hot 테이프 읽기만)
/// · ICEBERG_OUT(기본 data/live/rtcollect/iceberg -- 파이썬 폴더 data/live/iceberg 는 RTCOLLECT_TAKEOVER=1 일 때만).
fn run_iceberg() {
    let sym = env_s("ICEBERG_SYMBOL", "ETHUSDT").to_uppercase();
    let dd = PathBuf::from(env_s("DD_ROOT", "data/live/orderflow/depthdiff")).join(&sym);
    let tape = PathBuf::from(env_s("TAPE_DB_PATH", "data/hot/binance_tape.sqlite"));
    let out = PathBuf::from(env_s("ICEBERG_OUT", "data/live/rtcollect/iceberg"));
    if out.ends_with("data/live/iceberg") && env_s("RTCOLLECT_TAKEOVER", "") != "1" {
        die(2, &format!("🔴{} 는 파이썬 탐지기의 폴더다 -- 그림자는 다른 경로에. 전환이면 RTCOLLECT_TAKEOVER=1", out.display()));
    }
    let mut ic = rtcollect::iceberg::Iceberg::new(&sym, &dd, &tape, &out).unwrap_or_else(|e| die(2, &format!("{e}")));
    println!("시작 · 기존 아이스버그 {}건(전진) · {sym} {} → {}", ic.n, dd.display(), out.display());
    let mut term = false;
    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let f2 = flag.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async { let _ = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap().recv().await; });
        f2.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    while !term {
        ic.step(rtcollect::rest::wall_s());
        std::thread::sleep(Duration::from_secs(2));
        term = flag.load(std::sync::atomic::Ordering::Relaxed);
    }
    println!("종료 신호 -- 끝낸다");
}

/// 예약 수집기(cron 이 부른다 -- 한 번 돌고 끝 · 종료코드는 원본과 같다). 쓰는 곳 기본 = 그림자 자리(파이썬 파일이면 RTCOLLECT_TAKEOVER=1 일 때만):
///   treasury_futures [--range 1d|5d] → TREASURY_FUTURES_DB(기본 data/hot/rtcollect/treasury_futures.sqlite)
///   macro_5m → MACRO_5M_DB(기본 data/hot/rtcollect/macro_5m.sqlite) · cboe_chain [종목…] → CBOE_OUT(기본 data/live/rtcollect/options_equity/cboe)
fn run_cron(kind: &str) -> ! {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let agent = rtcollect::rest::agent(Duration::from_secs(30)); // urlopen(timeout=30)
    let mut get = |u: &str, ua: &str| -> Result<(u16, String, Vec<u8>), String> {
        let mut r = agent.get(u).header("User-Agent", ua).call().map_err(|e| format!("URLError({e})"))?;
        let st = r.status();
        let mut body = vec![];
        std::io::Read::read_to_end(&mut r.body_mut().as_reader(), &mut body).map_err(|e| format!("OSError({e})"))?;
        Ok((st.as_u16(), st.canonical_reason().unwrap_or("").to_string(), body))
    };
    let mut now = rtcollect::rest::wall_s;
    let code = match kind {
        "treasury_futures" => {
            let rng = args.iter().position(|a| a == "--range").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "5d".into());
            if rng != "1d" && rng != "5d" {
                die(2, &format!("--range 는 1d 또는 5d: {rng}"));
            }
            let db = db_path("TREASURY_FUTURES_DB", "treasury_futures.sqlite");
            let b = rtcollect::cron::base("RTCOLLECT_YAHOO_BASE", rtcollect::cron::YAHOO);
            rtcollect::cron::treasury(&db, &rng, &b, &mut get, &mut now).unwrap_or_else(|e| die(1, &format!("DB: {e}")))
        }
        "macro_5m" => {
            let db = db_path("MACRO_5M_DB", "macro_5m.sqlite");
            let b = rtcollect::cron::base("RTCOLLECT_YAHOO_BASE", rtcollect::cron::YAHOO);
            let gap: f64 = env_s("RTCOLLECT_MACRO_GAP_S", "1.5").parse().unwrap_or(1.5); // 원본 GAP_S(시험만 0)
            let mut sleep = |s: f64| std::thread::sleep(Duration::from_secs_f64(s * gap / 1.5));
            rtcollect::cron::macro_5m(&db, &b, &mut get, &mut now, &mut sleep).unwrap_or_else(|e| die(1, &format!("DB: {e}")))
        }
        _ => {
            let out = PathBuf::from(env_s("CBOE_OUT", "data/live/rtcollect/options_equity/cboe"));
            if out.ends_with("data/live/options_equity/cboe") && env_s("RTCOLLECT_TAKEOVER", "") != "1" {
                die(2, &format!("🔴{} 는 파이썬 수집기의 폴더다 -- 전환 때만 RTCOLLECT_TAKEOVER=1", out.display()));
            }
            let b = rtcollect::cron::base("RTCOLLECT_CBOE_BASE", rtcollect::cron::CBOE);
            // 요약기 → CBOE_STATE(기본 data/live/rtcollect/cboe_option_state.json) · 실현변동성 캐시 CBOE_RV_CACHE · 바이낸스 klines 는 RTCOLLECT_REST=1 일 때만(밴 파일 지킴)
            let state = PathBuf::from(env_s("CBOE_STATE", "data/live/rtcollect/cboe_option_state.json"));
            if state.ends_with("data/live/cboe_option_state.json") && env_s("RTCOLLECT_TAKEOVER", "") != "1" {
                die(2, &format!("🔴{} 는 파이썬 요약기의 파일이다 -- 전환 때만 RTCOLLECT_TAKEOVER=1", state.display()));
            }
            let cache = PathBuf::from(env_s("CBOE_RV_CACHE", "data/live/rtcollect/cboe_rv7_cache.json"));
            let mut klines = binance_rest("crypto-scalping-collector/1.0", 20);
            let mut after = |p: &std::path::Path| {
                let now = rtcollect::rest::wall_s();
                rtcollect::cboe_summary::write_state(p, &state, now, &mut |sym: &str| rtcollect::cboe_summary::realized_vol_7d(sym, &cache, now, &mut klines))
            };
            rtcollect::cron::cboe(&args, &out, &b, &mut get, rtcollect::rest::wall_s(), &mut after).0
        }
    };
    std::process::exit(code)
}

/// 바이낸스 선물 REST GET(timeout = 원본 urlopen 의 timeout). 🔴RTCOLLECT_REST=1 일 때만 · 밴 파일(RTCOLLECT_BAN_FILE) 지킴 ·
/// RTCOLLECT_BINANCE_BASE = 주소 앞부분(시험 = 127.0.0.1 가짜 서버). 4xx·5xx = Err(원본 HTTPError).
fn binance_rest(ua: &'static str, timeout_s: u64) -> impl FnMut(&str) -> Result<Vec<u8>, String> {
    let (rest, ban) = (env_s("RTCOLLECT_REST", "") == "1", PathBuf::from(env_s("RTCOLLECT_BAN_FILE", "data/live/binance_ban_until")));
    let kbase = rtcollect::cron::base("RTCOLLECT_BINANCE_BASE", rtcollect::cboe_summary::BINANCE);
    let kagent = rtcollect::rest::agent(Duration::from_secs(timeout_s));
    move |u: &str| -> Result<Vec<u8>, String> {
        if !rest {
            return Err("REST 꺼짐".into());
        }
        let now = rtcollect::rest::wall_s();
        if rtcollect::rest::ban_left(&ban, now) > 0.0 {
            return Err("binance IP ban".into());
        }
        let u = u.replacen(rtcollect::cboe_summary::BINANCE, &kbase, 1);
        let mut r = kagent.get(&u).header("User-Agent", ua).call().map_err(|e| e.to_string())?;
        let st = r.status().as_u16();
        let body = r.body_mut().read_to_string().map_err(|e| e.to_string())?;
        rtcollect::rest::ban_note(&ban, st, &body, None, now);
        if st >= 400 { Err(format!("HTTP {st}")) } else { Ok(body.into_bytes()) }
    }
}

/// 청산지도 전진 점수(live_liqmap_scorer_20261008 -- cron 매시 07분 · 한 번 돌고 끝). LIQMAP_OUT(기본 data/live/rtcollect/liqmap_score.jsonl) ·
/// LIQMAP_STATUS(기본 …/liqmap_score_status.json) · LIQMAP_HOT(기본 data/hot -- 읽기만). 파이썬 파일(data/live/liqmap_score*.json*)은 RTCOLLECT_TAKEOVER=1 일 때만.
/// 🔴klines REST(시간당 1번 · limit 198) 없이는 할 일이 없다 -- RTCOLLECT_REST=1 이 아니면 그 말만 찍고 0 으로 끝난다.
fn run_liqmap() -> ! {
    let out = PathBuf::from(env_s("LIQMAP_OUT", "data/live/rtcollect/liqmap_score.jsonl"));
    let status = PathBuf::from(env_s("LIQMAP_STATUS", "data/live/rtcollect/liqmap_score_status.json"));
    if (out.ends_with("data/live/liqmap_score.jsonl") || status.ends_with("data/live/liqmap_score_status.json")) && env_s("RTCOLLECT_TAKEOVER", "") != "1" {
        die(2, "🔴data/live/liqmap_score* 는 파이썬 기록기의 파일이다(전진 판정 자료) -- 전환 때만 RTCOLLECT_TAKEOVER=1");
    }
    if env_s("RTCOLLECT_REST", "") != "1" {
        println!("RTCOLLECT_REST 꺼짐 -- klines 없이는 할 일이 없다");
        std::process::exit(0)
    }
    let hot = PathBuf::from(env_s("LIQMAP_HOT", "data/hot"));
    match rtcollect::liqmap_scorer::run(&out, &status, &hot, wall_s(), &mut binance_rest("liqmap-scorer", 20)) {
        Ok(m) => {
            println!("{m}");
            std::process::exit(0)
        }
        Err(e) => die(1, &e),
    }
}

/// SOXL 주말 전진(live_soxl_weekend_forward_recorder_20261022 -- cron 매일 21:30 UTC `--record` · `--status`). SOXL_WEEKEND_OUT(기본 data/live/rtcollect/soxl_weekend_forward.jsonl) --
/// 파이썬 장부(data/live/soxl_weekend_forward.jsonl = 전진 판정 자료)는 RTCOLLECT_TAKEOVER=1 일 때만. 🔴klines·fundingRate REST(새 주말마다 9요청) = RTCOLLECT_REST=1 일 때만.
/// UA: 원본은 urllib 기본(Python-urllib/3.11) -- Rust 는 "soxl-weekend-forward".
fn run_soxl_weekend() -> ! {
    let out = PathBuf::from(env_s("SOXL_WEEKEND_OUT", "data/live/rtcollect/soxl_weekend_forward.jsonl"));
    if out.ends_with("data/live/soxl_weekend_forward.jsonl") && env_s("RTCOLLECT_TAKEOVER", "") != "1" {
        die(2, "🔴data/live/soxl_weekend_forward.jsonl 는 파이썬 전진 장부다 -- 전환 때만 RTCOLLECT_TAKEOVER=1");
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--status") {
        match rtcollect::soxl_weekend::status(&out) {
            Ok(m) => println!("{m}"),
            Err(e) => die(1, &e),
        }
        std::process::exit(0)
    }
    if !args.iter().any(|a| a == "--record") {
        die(2, "--record | --status (판정 --look 은 파이썬 원본으로)");
    }
    if env_s("RTCOLLECT_REST", "") != "1" {
        println!("RTCOLLECT_REST 꺼짐 -- klines 없이는 할 일이 없다");
        std::process::exit(0)
    }
    match rtcollect::soxl_weekend::record(&out, wall_s(), &mut binance_rest("soxl-weekend-forward", 15)) {
        Ok(w) => {
            for x in w {
                println!("기록 {x}");
            }
            std::process::exit(0)
        }
        Err(e) => die(1, &e),
    }
}

/// 뉴스 RSS(live_news_rss_collector_20261005 -- 상주). NEWS_RSS_DB(기본 data/hot/rtcollect/news_rss.sqlite -- data/hot/news_rss.sqlite 는 RTCOLLECT_TAKEOVER=1 일 때만) ·
/// NEWS_RSS_TICK_S(5) · RTCOLLECT_NEWS_BASE(시험만 -- 피드 주소의 "https://<호스트>" 를 이 앞부분으로 바꾼다 · 127.0.0.1 만).
/// 🔴바깥 사이트 8곳(CoinDesk·Cointelegraph·The Block·Tree·BWE·FinancialJuice·Fed·SEC)을 원본과 같은 주기로 부른다 -- 그림자로 켜면 그 사이트들 호출이 두 배(429 를 본 피드: tree 15초·financialjuice 30초).
fn run_news() -> ! {
    let db = db_path("NEWS_RSS_DB", "news_rss.sqlite");
    let tick = env_f("NEWS_RSS_TICK_S", 5.0);
    let base = std::env::var("RTCOLLECT_NEWS_BASE").ok().map(|_| rtcollect::cron::base("RTCOLLECT_NEWS_BASE", ""));
    let agent = rtcollect::rest::agent(Duration::from_secs(20)); // urlopen(timeout=20)
    let mut get = |u: &str, h: &[(&str, String)]| -> rtcollect::news::Resp {
        let u = match &base {
            Some(b) => format!("{b}/{}", u.splitn(4, '/').nth(3).unwrap_or("")),
            None => u.to_string(),
        };
        let mut rq = agent.get(&u);
        for (k, v) in h {
            rq = rq.header(*k, v);
        }
        let mut r = rq.call().map_err(|e| format!("URLError({e})"))?;
        let st = r.status().as_u16();
        let hs = r.headers().iter().map(|(k, v)| (k.as_str().to_lowercase(), v.to_str().unwrap_or("").to_string())).collect();
        let mut body = vec![];
        std::io::Read::read_to_end(&mut r.body_mut().as_reader(), &mut body).map_err(|e| format!("OSError({e})"))?;
        Ok((st, body, hs))
    };
    let mut clock = wall_s;
    let mut sleep = |s: f64| {
        std::thread::sleep(Duration::from_secs_f64(s));
        true
    };
    match rtcollect::news::run(&db, &rtcollect::news::FEEDS, &mut clock, &mut sleep, &mut get, tick, &mut |l| println!("{l}")) {
        Ok(()) => std::process::exit(0),
        Err(e) => die(1, &format!("DB: {e}")),
    }
}
