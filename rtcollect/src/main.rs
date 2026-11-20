//! ⚠️기본 웹소켓 주소는 **실제 바이낸스**다. 개발기에서 실행하지 않는다(서버와 IP 한도를 같이 쓴다 -- 밴 위험). 서버 그림자 전용.
//! ⚠️파이썬 수집기를 대체하지 않는다: REST 백필·1분봉 대조(backfill_1m·verify_1m 쓰기)가 없다. 그림자로 다른 파일에 쓴다.
//!
//! 바이낸스 체결 테이프 수집기(scripts/live_trade_tape_collector_20260916.py 의 `main()`·`run_all()`). 환경변수:
//!   TAPE_SYMBOL         쉼표 목록, 기본 ethusdt (원본과 같은 이름)
//!   TAPE_MARKET         futures(@trade, 기본) | spot(@aggTrade)
//!   TAPE_DB_PATH        기본 data/hot/rtcollect/binance_tape.sqlite · 현물 data/hot/rtcollect/binance_spot_tape.sqlite (현재 폴더 기준)
//!                       🔴파이썬과 같은 파일이면 gaps·meta 가 두 벌로 쌓인다 -- 기본값이 일부러 다르다
//!   RTCOLLECT_WS_URL    주소 틀({symbol} 자리), 기본 = 원본 MARKETS 의 WS (시험은 가짜 서버로 덮는다)
//!   RTCOLLECT_SILENCE_S 아무 프레임 없이 이만큼이면 재연결, 기본 45 (aiohttp heartbeat=30 + pong 15초)
//!   RTCOLLECT_WD_LOOP_S · RTCOLLECT_WD_WRITE_S  정지 감시 문턱, 기본 120 · 600 (원본 상수 -- 시험만 짧게)
//! 종료코드: 정지 감시 70 · 심볼 루프가 끝나면 70 · SIGTERM/SIGINT 0(닫힌 초를 쓰고 끝낸다 -- 원본은 안 쓰고 죽는다, 보고서 참고).
use rtcollect::binance::Collector;
use rtcollect::net::{self, EXIT_STALL};
use rtcollect::store::open_db;
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Duration;
use tokio::signal::unix::{signal, SignalKind};

fn env_f(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let syms: Vec<String> =
        std::env::var("TAPE_SYMBOL").unwrap_or_else(|_| "ethusdt".into()).to_lowercase().split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    let market = std::env::var("TAPE_MARKET").unwrap_or_else(|_| "futures".into());
    let (url, file) = match market.as_str() {
        "futures" => ("wss://fstream.binance.com/ws/{symbol}@trade", "binance_tape.sqlite"),
        "spot" => ("wss://stream.binance.com:9443/ws/{symbol}@aggTrade", "binance_spot_tape.sqlite"),
        m => panic!("TAPE_MARKET={m} -- futures|spot"),
    };
    let url = std::env::var("RTCOLLECT_WS_URL").unwrap_or_else(|_| url.into());
    let db = std::env::var("TAPE_DB_PATH").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("data/hot/rtcollect").join(file));
    let silence = Duration::from_secs_f64(env_f("RTCOLLECT_SILENCE_S", 45.0));
    net::start_watchdog(env_f("RTCOLLECT_WD_LOOP_S", net::WD_LOOP_STALL_S), env_f("RTCOLLECT_WD_WRITE_S", net::WD_WRITE_STALL_S));

    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, async move {
        let con = open_db(&db).expect("DB 열기");
        let mut cs = vec![];
        let mut tasks = tokio::task::JoinSet::new();
        for s in &syms {
            let c = Rc::new(RefCell::new(Collector::new(con.clone(), s, &market, net::mono_ms() as f64 / 1000.0).expect("TapeStore")));
            println!("{s} {market} 수집 시작 (빈 {}, db {})", rtcollect::tape::bucket(s), db.display());
            tasks.spawn_local(net::run_symbol(c.clone(), url.replace("{symbol}", s), silence));
            cs.push(c);
        }
        tasks.spawn_local(net::beat());
        let (mut term, mut int) = (signal(SignalKind::terminate()).unwrap(), signal(SignalKind::interrupt()).unwrap());
        let mut tick = tokio::time::interval(Duration::from_secs(60));
        tick.tick().await;
        loop {
            tokio::select! {
                _ = term.recv() => break,
                _ = int.recv() => break,
                r = tasks.join_next() => {
                    // 하나라도 끝나면(= 죽으면) 전부 걷는다(run_all) -- supervisor 가 다시 띄운다
                    eprintln!("수집 태스크 종료: {r:?} -- 감시자가 다시 띄운다");
                    std::process::exit(EXIT_STALL);
                }
                _ = tick.tick() => {
                    let st: Vec<String> = cs.iter().map(|c| { let c = c.borrow(); format!("{} 보류 {} 늦은버림 {}", c.store.symbol, c.store.pending.len(), c.buffer.dropped_late) }).collect();
                    println!("rtcollect 60s · 재연결 {} (누적) · {}", net::RECONNECTS.load(Relaxed), st.join(" · "));
                }
            }
        }
        // 닫힌 초만 쓰고 끝낸다(진행 중인 초는 버린다 = 다음 판 gaps 가 덮는다). 값은 원본이 그 순간 플러시했을 때와 같다.
        let n: usize = cs.iter().map(|c| c.borrow_mut().on_disconnect()).sum();
        println!("종료 신호 -- 닫힌 초 {n}행 쓰고 끝낸다");
    });
}
