//! 레벨 5분 실행 파일 -- scripts/live_sr_lean_20261118.py `run()` 의 소켓·HTTP 몫(메시지 처리는 rtmodel::sr_live::Runner).
//! 그림자 운영: 파이썬 워커 옆에서 **다른 디렉터리**에 같은 파일 셋을 쓴다(SHADOW.md). 주문·계정 호출 없음.
//!
//! 환경변수(기본값):
//!   RT_SR_OUT       data/live/rt_sr_lean         ethusdt.jsonl · ethusdt_feat.jsonl · ethusdt_state.json 을 쓰는 곳
//!                   🔴파이썬 워커의 data/live/sr_lean 을 가리키면 종료코드 2(전환 때만 RT_SR_TAKEOVER=1)
//!   RT_SR_MODEL     <RT_SR_OUT>/sr_lean.json      `python rtmodel/tools/export_hgb.py <디렉터리>` 가 쓴 얼린 모델
//!   RT_SR_WS        wss://fstream.binance.com/market/ws/ethusdt@aggTrade   (/ws/ 경로는 aggTrade 를 안 보낸다 -- 원본 주석 2026-09-21)
//!   RT_SR_REST      https://fapi.binance.com/fapi/v1/
//!   RT_SR_BAN_FILE  data/live/binance_ban_until   scripts/binance_ban_guard.py 와 같은 공용 파일(밴 중엔 REST 를 안 보낸다)
//!   RT_SR_KL_GAP_S  60                           1분봉 요청 간격(시험만 줄인다)
//! 종료코드: 2 = 설정 거부 · 1 = 원본이 잡지 않는 예외(깨진 메시지 등 -- 감시 스크립트가 다시 띄운다).
use futures_util::{SinkExt, StreamExt};
use rtmodel::ban::{ban_left, ban_note};
use rtmodel::sr_live::Runner;
use rtmodel::{read_json, SrLean};
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::time::{sleep, sleep_until, timeout, Instant};
use tokio_tungstenite::{connect_async, tungstenite::Message};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(15); // ponytail: 원본 aiohttp 는 세션 기본(총 300초) -- 반쯤 열린 접속을 빨리 접으려고 rtcollect 와 같은 15초
const RECV_TICK: Duration = Duration::from_secs(2); // asyncio.wait_for(ws.receive(), 2)
const HEARTBEAT: Duration = Duration::from_secs(20); // ws_connect(heartbeat=20): 마지막 수신 20초 뒤 PING · 10초 안에 아무 프레임도 없으면 닫힘
const RECONNECT: Duration = Duration::from_secs(5);
const REST_TIMEOUT: Duration = Duration::from_secs(10); // ClientTimeout(total=10)

fn env(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.into())
}

fn die(code: i32, msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(code)
}

fn wall() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

/// REST 한 번(동기 -- 원본도 그 호출을 기다리는 동안 수신을 안 한다). 밴 중이면 보내지 않고 실패.
fn rest_get(agent: &ureq::Agent, base: &str, ban: &Path, pq: &str) -> Result<Value, String> {
    let left = ban_left(ban, wall());
    if left > 0.0 {
        return Err(format!("binance IP ban: {left:.0}s left -- request not sent"));
    }
    let mut r = agent.get(&format!("{base}{pq}")).call().map_err(|e| format!("ClientError: {e}"))?;
    let h = |n: &str| r.headers().get(n).and_then(|v| v.to_str().ok()).map(str::to_string);
    let (status, retry) = (r.status().as_u16(), h("retry-after"));
    let body = r.body_mut().read_to_string().map_err(|e| format!("ClientPayloadError: {e}"))?;
    ban_note(ban, status, &body, retry.as_deref(), wall());
    if status >= 400 {
        return Err(format!("ClientResponseError: {status}")); // raise_for_status
    }
    serde_json::from_str(&body).map_err(|e| format!("ContentTypeError: {e}"))
}

fn append(p: &Path, lines: &mut Vec<String>) {
    if lines.is_empty() {
        return;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(p).unwrap_or_else(|e| die(1, &format!("{}: {e}", p.display())));
    for l in lines.drain(..) {
        writeln!(f, "{l}").unwrap_or_else(|e| die(1, &format!("{}: {e}", p.display())));
    }
}

fn main() {
    let out = PathBuf::from(env("RT_SR_OUT", "data/live/rt_sr_lean"));
    let python = out.file_name() == Some("sr_lean".as_ref()) && out.parent().and_then(|d| d.file_name()) == Some("live".as_ref());
    if python && env("RT_SR_TAKEOVER", "") != "1" {
        die(2, &format!("🔴{} 는 파이썬 워커의 기록 디렉터리다 -- 그림자는 다른 경로에(기본 data/live/rt_sr_lean). 전환이면 RT_SR_TAKEOVER=1", out.display()));
    }
    let model = PathBuf::from(env("RT_SR_MODEL", &out.join("sr_lean.json").to_string_lossy()));
    let art: SrLean = read_json(&model).unwrap_or_else(|e| die(2, &format!("모델: {e}")));
    let (ws_url, base, ban) = (env("RT_SR_WS", "wss://fstream.binance.com/market/ws/ethusdt@aggTrade"), env("RT_SR_REST", "https://fapi.binance.com/fapi/v1/"),
                               PathBuf::from(env("RT_SR_BAN_FILE", "data/live/binance_ban_until")));
    std::fs::create_dir_all(&out).unwrap_or_else(|e| die(2, &format!("{}: {e}", out.display())));
    let (f_main, f_feat, f_state) = (out.join("ethusdt.jsonl"), out.join("ethusdt_feat.jsonl"), out.join("ethusdt_state.json"));
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(REST_TIMEOUT)).http_status_as_error(false).proxy(None).build().into();
    let mut run = Runner::new(&art);
    run.kl_gap = env("RT_SR_KL_GAP_S", "60").parse().unwrap_or_else(|_| die(2, "RT_SR_KL_GAP_S"));
    println!("시작 · 모델 {} · 문턱 {:?} · 재료 {}개 · 기록 {}", model.display(), art.thr, art.cols.len(), out.display());
    let mut rest = |pq: &str| rest_get(&agent, &base, &ban, pq);
    let mut clock = wall;
    let mut nap = |s: f64| std::thread::sleep(Duration::from_secs_f64(s));
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        loop {
            match timeout(CONNECT_TIMEOUT, connect_async(ws_url.as_str())).await {
                Ok(Ok((ws, _))) => {
                    println!("웹소켓 연결");
                    let (mut tx, mut rx) = ws.split();
                    let (mut last, mut pinged) = (Instant::now(), None::<Instant>);
                    loop {
                        // 수신 2초 · 마지막 수신 20초 뒤 PING · 그 뒤 10초 안에 아무 프레임도 없으면 닫힘(aiohttp heartbeat)
                        let tick = Instant::now() + RECV_TICK;
                        let r = tokio::select! {
                            _ = sleep_until(last + HEARTBEAT), if pinged.is_none() => {
                                pinged = Some(Instant::now());
                                if tx.send(Message::Ping(Default::default())).await.is_err() { break; }
                                continue;
                            }
                            _ = sleep_until(pinged.unwrap_or(last) + HEARTBEAT / 2), if pinged.is_some() => { println!("pong 없음 -- 닫는다"); break; }
                            _ = sleep_until(tick) => None,
                            m = rx.next() => Some(m),
                        };
                        let text = match r {
                            None => None,
                            Some(Some(Ok(Message::Text(t)))) => {
                                (last, pinged) = (Instant::now(), None);
                                Some(t)
                            }
                            Some(Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_)))) => {
                                (last, pinged) = (Instant::now(), None);
                                continue;
                            }
                            // 글자 아닌 메시지·닫힘·오류 = ConnectionError → 5초 뒤 다시(빠진 묶음은 번호로 메운다)
                            other => {
                                println!("웹소켓 닫힘 {other:?}");
                                break;
                            }
                        };
                        match run.on_msg(text.as_deref(), &mut rest, &mut clock, &mut nap) {
                            Ok(state) => {
                                append(&f_main, &mut run.live.log);
                                append(&f_feat, &mut run.live.feat);
                                if let Some(s) = state {
                                    let tmp = out.join("ethusdt_state.tmp");
                                    if let Err(e) = std::fs::write(&tmp, s).and_then(|_| std::fs::rename(&tmp, &f_state)) {
                                        die(1, &format!("생존 표시: {e}"));
                                    }
                                }
                            }
                            Err(e) => {
                                append(&f_main, &mut run.live.log);
                                append(&f_feat, &mut run.live.feat);
                                die(1, &format!("처리 못 한 예외 -- 감시 스크립트가 다시 띄운다: {}", e.0));
                            }
                        }
                    }
                }
                Ok(Err(e)) => println!("연결 실패 {e}"),
                Err(_) => println!("연결 실패: 접속 {}초 초과", CONNECT_TIMEOUT.as_secs()),
            }
            println!("연결 끊김 -- 5초 뒤 다시(빠진 묶음은 번호로 메운다)");
            sleep(RECONNECT).await;
        }
    });
}
