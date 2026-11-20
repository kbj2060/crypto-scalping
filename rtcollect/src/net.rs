//! 접속 루프(실행 파일 main.rs 만 쓴다). 원본마다 소켓 몫만 옮겼다 -- 메시지 → 행은 각 모듈:
//! · 바이낸스 테이프 `run_symbol` = scripts/live_trade_tape_collector_20260916.py `collect()` :923-1000(줄 번호는 그 파일)
//!   + 정지 감시(`_watchdog`·`_beat` :883-906, 종료코드 70) + 심볼 여럿 한 프로세스(`run_all` :909-920)
//! · OKX 테이프 `run_okx` = live_okx_trade_tape_collector_20260923.py `collect()` :333-392 (정지 감시는 바이낸스 것을 그대로 -- 원본도 `_bn.run_all`)
//! · Bybit 청산·테이프 `run_bybit` = live_bybit_liquidation_collector_20261003.py `run()` :62-99 · live_bybit_trade_tape_collector_20261006.py
//!   `run()` :175-270 (+ 체결 정지 감시 `start_stall_guard`, 종료코드 3). 청산 수집기는 정지 감시가 없다(원본 그대로).
//! 한 스레드(tokio current_thread + LocalSet): 심볼들이 SQLite 연결 하나(Rc)를 나눠 쓴다 -- 원본 sqlite_shared 와 같은 모양.
// ponytail: 쓰기(5초마다 트랜잭션 하나, NVMe ~ms)가 같은 스레드에서 돈다(원본은 to_thread) -- 쓰기가 수신을 눈에 띄게 막으면 쓰기 스레드로 뺀다.
use crate::binance::Collector;
use crate::bybit_liq::LiqCollector;
use crate::bybit_tape::BybitCollector;
use crate::{okx, Reconnect, LAST_WRITE_MS};
use futures_util::{SinkExt, StreamExt};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokio::time::{sleep, sleep_until, timeout, timeout_at};
use tokio_tungstenite::{connect_async, tungstenite::Message};

pub const RECONNECT_WAIT: Duration = Duration::from_secs(3); // asyncio.sleep(3.0) (:1000)
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(15); // WS_CONNECT_S -- 반쯤 열린 접속에서 영원히 안 끝났다(11-02 재현, :90-97)
/// aiohttp heartbeat=30(:952): 30초마다 ping, pong 이 15초 안에 안 오면 닫는다. 여기는 «ping 30초 · 아무 프레임도 없이 45초 = 침묵».
pub const PING_EVERY: Duration = Duration::from_secs(30);
/// 2026-11-01 정지 감시: 프로세스는 살아 있는데 7심볼이 15~18시간 기록이 끊겼다(오류 로그 없음) -- 루프 박동·쓰기 성공이 끊기면 종료코드 70.
pub const WD_LOOP_STALL_S: f64 = 120.0;
pub const WD_WRITE_STALL_S: f64 = 600.0;
pub const EXIT_STALL: i32 = 70;

static START: OnceLock<Instant> = OnceLock::new();
/// 루프 박동(5초마다, `_beat`) -- 단조 ms.
pub static LOOP_BEAT_MS: AtomicU64 = AtomicU64::new(0);
pub static RECONNECTS: AtomicU64 = AtomicU64::new(0);

/// 프로세스 시작 뒤 단조 ms(원본 time.monotonic 대신). 0 은 «아직 없음» 이라 1 부터.
pub fn mono_ms() -> u64 {
    START.get_or_init(Instant::now).elapsed().as_millis() as u64 + 1
}

pub fn mono_s() -> f64 {
    mono_ms() as f64 / 1000.0
}

/// 정지 판정(:890-900). None = 괜찮음.
pub fn stall_reason(now_ms: u64, loop_ms: u64, write_ms: u64, loop_s: f64, write_s: f64) -> Option<String> {
    let (l, w) = ((now_ms.saturating_sub(loop_ms)) as f64 / 1000.0, (now_ms.saturating_sub(write_ms)) as f64 / 1000.0);
    if l > loop_s {
        Some(format!("이벤트 루프 {l:.0}초 정지"))
    } else if w > write_s {
        Some(format!("쓰기 성공이 {w:.0}초 없음"))
    } else {
        None
    }
}

/// 감시 스레드(이벤트 루프가 쓰기에 묶여 멈춰도 돈다). 원본은 10초마다 본다 -- 짧은 문턱(시험)이면 그만큼 자주.
pub fn start_watchdog(loop_s: f64, write_s: f64) {
    let now = mono_ms();
    LOOP_BEAT_MS.store(now, Relaxed);
    LAST_WRITE_MS.store(now, Relaxed);
    let every = Duration::from_secs_f64((loop_s.min(write_s) / 3.0).min(10.0));
    std::thread::spawn(move || loop {
        std::thread::sleep(every);
        if let Some(why) = stall_reason(mono_ms(), LOOP_BEAT_MS.load(Relaxed), LAST_WRITE_MS.load(Relaxed), loop_s, write_s) {
            // 원본은 여기서 모든 스레드 스택을 남긴다(faulthandler) -- Rust 는 스택을 못 떠서 이유만 남긴다.
            eprintln!("⚠️정지 감지({why}) -- 종료, supervisor 가 재기동한다");
            std::process::exit(EXIT_STALL);
        }
    });
}

/// 루프 박동(`_beat`): 5초마다.
pub async fn beat() {
    loop {
        LOOP_BEAT_MS.store(mono_ms(), Relaxed);
        sleep(Duration::from_secs(5)).await;
    }
}

/// 한 심볼 영원히(:950-1000). 끊김·글자 아닌 메시지·침묵·접속 실패·메시지 예외 → 진행 중인 초를 빼고 쓰고 → 3초 → 다시.
pub async fn run_symbol(c: Rc<RefCell<Collector>>, url: String, silence: Duration) {
    let sym = c.borrow().store.symbol.clone();
    loop {
        match timeout(CONNECT_TIMEOUT, connect_async(url.as_str())).await {
            Ok(Ok((ws, _))) => {
                c.borrow_mut().on_connect();
                let (mut tx, mut rx) = ws.split();
                let mut last = tokio::time::Instant::now();
                let mut ping_at = last + PING_EVERY;
                loop {
                    let r = tokio::select! {
                        _ = sleep_until(ping_at) => {
                            ping_at += PING_EVERY;
                            if let Err(e) = tx.send(Message::Ping(Default::default())).await {
                                eprintln!("[{sym}] 연결 실패, 3초 뒤 재시도: ping {e}");
                                break;
                            }
                            continue;
                        }
                        r = timeout_at(last + silence, rx.next()) => r,
                    };
                    match r {
                        Err(_) => {
                            eprintln!("[{sym}] 연결 실패, 3초 뒤 재시도: {:.0}초 침묵", silence.as_secs_f64());
                            break;
                        }
                        Ok(Some(Ok(Message::Text(t)))) => {
                            last = tokio::time::Instant::now();
                            if let Err(e) = c.borrow_mut().on_text(mono_s(), t.as_str()) {
                                eprintln!("[{sym}] 연결 실패, 3초 뒤 재시도: {}", e.0);
                                break;
                            }
                        }
                        // ping 은 tungstenite 가 알아서 pong 한다 · pong 도 «살아 있다»
                        Ok(Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_)))) => last = tokio::time::Instant::now(),
                        // `if msg.type is not WSMsgType.TEXT: break` -- 바이너리·닫힘
                        Ok(Some(Ok(_))) | Ok(None) => break,
                        Ok(Some(Err(e))) => {
                            eprintln!("[{sym}] 연결 실패, 3초 뒤 재시도: {e}");
                            break;
                        }
                    }
                }
            }
            Ok(Err(e)) => eprintln!("[{sym}] 연결 실패, 3초 뒤 재시도: {e}"),
            Err(_) => eprintln!("[{sym}] 연결 실패, 3초 뒤 재시도: 접속 {}초 초과", CONNECT_TIMEOUT.as_secs()),
        }
        RECONNECTS.fetch_add(1, Relaxed);
        c.borrow_mut().on_disconnect(); // await asyncio.to_thread(store.write, buffer.take_closed()) (:999)
        sleep(RECONNECT_WAIT).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stall_boundaries() {
        assert_eq!(stall_reason(200_000, 80_000, 199_000, 120.0, 600.0), None, "루프 120초는 아직");
        assert!(stall_reason(200_001, 80_000, 199_000, 120.0, 600.0).unwrap().contains("루프"), "루프 120.001초 = 정지");
        assert!(stall_reason(700_001, 699_000, 100_000, 120.0, 600.0).unwrap().contains("쓰기"), "쓰기 600.001초 = 정지");
        assert!(stall_reason(900_000, 100_000, 100_000, 120.0, 600.0).unwrap().contains("루프"), "둘 다면 루프가 먼저(원본 순서)");
    }
}

// ── OKX ──────────────────────────────────────────────────────────────────────────────────────────
/// aiohttp heartbeat=20(:347): 마지막 수신(제어 프레임 포함) 뒤 20초면 PING 프레임 하나. pong 시한(10초)은 안 옮겼다 --
/// 수신 시한 25초(`okx::RECV_TIMEOUT_S`)가 늘 먼저 온다(20+10 > 25). 🔴원본 주석은 «25초 조용하면 ping» 이지만 코드는 25초면 재연결이다.
pub const OKX_HEARTBEAT: Duration = Duration::from_secs(20);

/// 한 종목 영원히: 접속(15초 시한) → 구독 → 글자 메시지 → 25초 무음·글자 아닌 메시지·구독 거부·예외면 진행 중인 초를 빼고 쓰고 → 3초 → 다시.
/// aiohttp receive 는 제어 프레임(pong)을 삼키고 시한을 다시 잰다 -- 여기도 어떤 프레임이든 «마지막 수신»을 민다.
pub async fn run_okx(c: Rc<RefCell<okx::Collector>>, url: String) {
    let inst = c.borrow().store.symbol.clone();
    let silence = Duration::from_secs_f64(okx::RECV_TIMEOUT_S);
    loop {
        match timeout(CONNECT_TIMEOUT, connect_async(url.as_str())).await {
            Ok(Ok((ws, _))) => {
                let (mut tx, mut rx) = ws.split();
                let sub = c.borrow().subscribe();
                if let Err(e) = tx.send(Message::text(sub)).await {
                    eprintln!("[{inst}] 연결 실패, 3초 뒤 재시도: 구독 {e}");
                } else {
                    c.borrow_mut().on_connect();
                    let mut last = tokio::time::Instant::now();
                    let mut pinged = false;
                    loop {
                        let r = tokio::select! {
                            _ = sleep_until(last + OKX_HEARTBEAT), if !pinged => {
                                pinged = true;
                                if let Err(e) = tx.send(Message::Ping(Default::default())).await {
                                    eprintln!("[{inst}] 연결 실패, 3초 뒤 재시도: ping {e}");
                                    break;
                                }
                                continue;
                            }
                            r = timeout_at(last + silence, rx.next()) => r,
                        };
                        match r {
                            Err(_) => {
                                eprintln!("[{inst}] 25초 무음 -- 재연결");
                                break;
                            }
                            Ok(Some(Ok(Message::Text(t)))) => {
                                (last, pinged) = (tokio::time::Instant::now(), false);
                                if c.borrow_mut().on_text(mono_s(), t.as_str()).is_err() {
                                    break; // 구독 거부(on_text 가 로그) · 파싱 예외 = 재연결
                                }
                            }
                            Ok(Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_)))) => (last, pinged) = (tokio::time::Instant::now(), false),
                            Ok(Some(Ok(_))) | Ok(None) => break, // CLOSE/ERROR/BINARY -- `if msg.type is not WSMsgType.TEXT: break`
                            Ok(Some(Err(e))) => {
                                eprintln!("[{inst}] 연결 실패, 3초 뒤 재시도: {e}");
                                break;
                            }
                        }
                    }
                }
            }
            Ok(Err(e)) => eprintln!("[{inst}] 연결 실패, 3초 뒤 재시도: {e}"),
            Err(_) => eprintln!("[{inst}] 연결 실패, 3초 뒤 재시도: 접속 {}초 초과", CONNECT_TIMEOUT.as_secs()),
        }
        RECONNECTS.fetch_add(1, Relaxed);
        c.borrow_mut().on_disconnect(); // await asyncio.to_thread(store.write, buffer.take_closed()) (:391)
        sleep(RECONNECT_WAIT).await;
    }
}

// ── Bybit(청산·테이프) ───────────────────────────────────────────────────────────────────────────
/// 두 원본 `run()` 의 같은 뼈대: 접속 → 구독 → 5초 수신 시한으로 돌며(시한·메시지마다 ping·플러시 판정) → 예외면 정한 만큼 쉬고 다시.
/// 상태(무엇을 쓰고 언제 ping 하나)는 각 모듈의 수집기, 여기는 소켓과 예외 이름만.
pub trait Bybit {
    /// 원본 웹소켓 라이브러리가 aiohttp(테이프)인가 websockets(청산)인가 -- 예외 이름과 수신 시한 규칙이 다르다:
    /// websockets `wait_for(ws.recv(), 5)` 는 제어 프레임이 와도 시한이 그대로, aiohttp `receive(timeout=5)` 는 제어 프레임을 삼키고 다시 잰다.
    const AIOHTTP: bool;
    /// 접속 시한: websockets 기본 open_timeout 10초 · aiohttp 는 `_bn.ws_open` 15초.
    const CONNECT: Duration;
    fn open(&mut self, now: f64, wall: f64) -> Result<(), Reconnect>;
    /// Ok(true) = 지금 {"op": "ping"} 을 보낸다.
    fn text(&mut self, now: f64, wall: f64, t: &str) -> Result<bool, Reconnect>;
    fn idle(&mut self, now: f64, wall: f64) -> Result<bool, Reconnect>;
    /// 반환 = 다시 붙기 전에 쉴 초.
    fn fail(&mut self, now: f64, wall: f64, why: &str) -> f64;
}

impl Bybit for LiqCollector {
    const AIOHTTP: bool = false;
    const CONNECT: Duration = Duration::from_secs(10);
    fn open(&mut self, _now: f64, wall: f64) -> Result<(), Reconnect> {
        self.on_open(wall)
    }
    fn text(&mut self, _now: f64, wall: f64, t: &str) -> Result<bool, Reconnect> {
        self.on_text(wall, t)
    }
    fn idle(&mut self, _now: f64, wall: f64) -> Result<bool, Reconnect> {
        self.on_timeout(wall)
    }
    fn fail(&mut self, _now: f64, wall: f64, why: &str) -> f64 {
        self.on_error(wall, why) // 1 → 2 → 4 … 60초
    }
}

impl Bybit for BybitCollector {
    const AIOHTTP: bool = true;
    const CONNECT: Duration = CONNECT_TIMEOUT;
    fn open(&mut self, now: f64, _wall: f64) -> Result<(), Reconnect> {
        self.on_open(now);
        Ok(())
    }
    fn text(&mut self, now: f64, wall: f64, t: &str) -> Result<bool, Reconnect> {
        let r = self.on_text(now, wall, t);
        TRADE_BEAT_MS.store((self.beat * 1000.0) as u64, Relaxed);
        r
    }
    fn idle(&mut self, now: f64, wall: f64) -> Result<bool, Reconnect> {
        self.on_timeout(now, wall)
    }
    fn fail(&mut self, now: f64, wall: f64, why: &str) -> f64 {
        self.on_error(now, wall, why);
        3.0
    }
}

pub const BYBIT_PING: &str = r#"{"op": "ping"}"#; // json.dumps({"op": "ping"})
const BYBIT_RECV: Duration = Duration::from_secs(5);

/// 영원히(원본 `while True`). `sub` = 구독 글자.
pub async fn run_bybit<C: Bybit>(c: Rc<RefCell<C>>, url: String, sub: String) {
    loop {
        let why = match timeout(C::CONNECT, connect_async(url.as_str())).await {
            Ok(Ok((ws, _))) => bybit_session(&c, ws, &sub).await,
            Ok(Err(e)) if C::AIOHTTP => format!("ClientConnectorError: {e}"),
            Ok(Err(e)) => format!("OSError: {e}"),
            Err(_) => "TimeoutError: ".to_string(),
        };
        RECONNECTS.fetch_add(1, Relaxed);
        let wait = c.borrow_mut().fail(mono_s(), crate::rest::wall_s(), &why);
        eprintln!("끊김 {why} -- {wait:.0}초 뒤 재접속");
        sleep(Duration::from_secs_f64(wait)).await;
    }
}

/// 한 연결. 반환 = 끊긴 이유(원본 `f"{type(exc).__name__}: {exc}"` 모양 -- 테이프는 이름만 남긴다).
async fn bybit_session<C: Bybit, S>(c: &Rc<RefCell<C>>, ws: tokio_tungstenite::WebSocketStream<S>, sub: &str) -> String
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let send_err = |e: tokio_tungstenite::tungstenite::Error| {
        if C::AIOHTTP { format!("ClientConnectionResetError: {e}") } else { format!("ConnectionClosedError: {e}") }
    };
    let closed = |e: Option<String>| match (C::AIOHTTP, e) {
        (true, _) => "ConnectionError: ws CLOSE".to_string(), // `raise ConnectionError(f"ws {m.type}")`
        (false, None) => "ConnectionClosedOK: received close".to_string(),
        (false, Some(e)) => format!("ConnectionClosedError: {e}"),
    };
    let (mut tx, mut rx) = ws.split();
    if let Err(e) = tx.send(Message::text(sub)).await {
        return send_err(e);
    }
    if let Err(e) = c.borrow_mut().open(mono_s(), crate::rest::wall_s()) {
        return e.0;
    }
    loop {
        let mut dl = tokio::time::Instant::now() + BYBIT_RECV;
        let r = loop {
            match timeout_at(dl, rx.next()).await {
                Ok(Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_)))) => {
                    if C::AIOHTTP {
                        dl = tokio::time::Instant::now() + BYBIT_RECV;
                    }
                }
                r => break r,
            }
        };
        let (now, wall) = (mono_s(), crate::rest::wall_s());
        let step = match r {
            Err(_) => c.borrow_mut().idle(now, wall),
            Ok(Some(Ok(Message::Text(t)))) => c.borrow_mut().text(now, wall, t.as_str()),
            // websockets recv() 는 바이너리도 돌려주고 json.loads 는 bytes 도 읽는다 · aiohttp 는 글자 아닌 메시지 = ConnectionError
            Ok(Some(Ok(Message::Binary(b)))) if !C::AIOHTTP => c.borrow_mut().text(now, wall, &String::from_utf8_lossy(&b)),
            Ok(Some(Ok(Message::Binary(_)))) => return "ConnectionError: ws BINARY".into(),
            Ok(Some(Ok(_))) | Ok(None) => return closed(None),
            Ok(Some(Err(e))) => return closed(Some(e.to_string())),
        };
        match step {
            Ok(true) => {
                if let Err(e) = tx.send(Message::text(BYBIT_PING)).await {
                    return send_err(e);
                }
            }
            Ok(false) => {}
            Err(e) => return e.0,
        }
    }
}

/// Bybit 테이프의 체결 박동(단조 ms, `BybitCollector::beat`) -- 정지 감시가 본다.
pub static TRADE_BEAT_MS: AtomicU64 = AtomicU64::new(0);
/// 2026-10-31 멈춤 자가 종료: 10-28~31 프로세스가 살아 있는 채 72시간 멈췄다 -- 구독 종목 체결이 600초 없으면 종료코드 3.
pub const EXIT_TRADE_STALL: i32 = 3;

/// 원본 `start_stall_guard`(30초마다 본다 -- 짧은 문턱(시험)이면 그만큼 자주). 이벤트 루프가 쓰기에 묶여 멈춰도 도는 별도 스레드.
pub fn start_trade_guard(stall_s: f64) {
    TRADE_BEAT_MS.store(mono_ms(), Relaxed);
    let every = Duration::from_secs_f64((stall_s / 3.0).min(30.0));
    std::thread::spawn(move || loop {
        std::thread::sleep(every);
        if mono_ms().saturating_sub(TRADE_BEAT_MS.load(Relaxed)) as f64 / 1000.0 > stall_s {
            eprintln!("체결 {stall_s:.0}초 없음 -- 멈춤으로 보고 종료(감독 프로세스가 다시 띄운다)");
            std::process::exit(EXIT_TRADE_STALL);
        }
    });
}
