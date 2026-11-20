//! 바이낸스 체결 테이프 접속 루프 = 원본 `collect()` 의 소켓 몫(scripts/live_trade_tape_collector_20260916.py :923-1000, 줄 번호는 그 파일)
//! + 정지 감시(`_watchdog`·`_beat` :883-906) + 심볼 여럿 한 프로세스(`run_all` :909-920). 실행 파일(main.rs)만 쓴다.
//! 한 스레드(tokio current_thread + LocalSet): 심볼들이 SQLite 연결 하나(Rc)를 나눠 쓴다 -- 원본 sqlite_shared 와 같은 모양.
// ponytail: 쓰기(5초마다 트랜잭션 하나, NVMe ~ms)가 같은 스레드에서 돈다(원본은 to_thread) -- 쓰기가 수신을 눈에 띄게 막으면 쓰기 스레드로 뺀다.
use crate::binance::Collector;
use crate::LAST_WRITE_MS;
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

fn mono_s() -> f64 {
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
