//! 사용자 스트림 전송(2026-11-21 여섯째 조각 D) -- `account::user_stream_run` 이 닿는 `StreamIo` 의 실제 구현:
//! listenKey = `Binance::listen_key`(HTTP) · 웹소켓 = tokio-tungstenite 0.26(rtcore·rtcollect 와 같은 판 · rustls ring).
//! 사건은 `Hub` 에 모인다(깨움·계좌 다시 읽기 신호·스트림 계좌·익절 조각 체결 대기열·로그) -- 감시·실행기·계좌 갱신기가 거기서 읽는다.
//! **어디에도 연결하지 않는다**(시험 = tests/userstream_ws.rs -- 127.0.0.1 가짜 HTTP·웹소켓 서버).

use crate::account::{Recv, StreamAccount, StreamIo, USER_STREAM_WS};
use crate::http::Binance;
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

/// 스트림이 바깥에 남기는 것. 원본의 wake_fills · acct_dirty · tp_leg_filled · print 자리.
#[derive(Default)]
pub struct Hub {
    pub wake: tokio::sync::Notify,
    pub dirty: tokio::sync::Notify,
    pub stream_account: Mutex<StreamAccount>,
    /// (심볼, 포지션 측면, 표식) -- 감시 쪽이 무장에서 그 조각을 뺀다(원본 tp_leg_filled). ponytail: 대기열로 넘김 -- 감시와 같은 줄에서 돌게 되면 바로 부른다.
    pub tp_fills: Mutex<Vec<(String, String, String)>>,
    pub logs: Mutex<Vec<String>>,
}

pub struct LiveStream {
    pub bx: Binance,
    /// 웹소켓 주소 앞부분(운영 = USER_STREAM_WS · 시험 = ws://127.0.0.1:…/private/ws/).
    pub ws_base: String,
    pub hub: Arc<Hub>,
    ws: Option<WebSocketStream<MaybeTlsStream<TcpStream>>>,
    t0: Instant,
    heartbeat: f64,
    last_ping: f64,
    /// 시험용 손잡이: 재연결 대기 배율(운영 1.0) · 잠 횟수 상한(None = 끝없이). 시험이 5초·10초를 실제로 안 기다리고 루프를 끝내게.
    pub sleep_scale: f64,
    pub max_sleeps: Option<u32>,
    pub sleeps: Vec<f64>,
}

impl LiveStream {
    pub fn new(bx: Binance, ws_base: &str, hub: Arc<Hub>) -> Self {
        let _ = rustls::crypto::ring::default_provider().install_default(); // 2026-11-20 rtcore 실서버: 제공자를 못 골라 wss 마다 패닉 -- 하나를 박아 둔다(이미 있으면 그대로)
        LiveStream { bx, ws_base: ws_base.into(), hub, ws: None, t0: Instant::now(), heartbeat: 0.0, last_ping: 0.0,
                     sleep_scale: 1.0, max_sleeps: None, sleeps: vec![] }
    }
}

impl StreamIo for LiveStream {
    fn ban_remaining(&self) -> f64 {
        self.bx.ban_left()
    }
    fn mono(&self) -> f64 {
        self.t0.elapsed().as_secs_f64()
    }
    fn wall(&self) -> f64 {
        self.bx.time()
    }
    async fn sleep(&mut self, sec: f64) -> bool {
        self.sleeps.push(sec);
        if self.max_sleeps.is_some_and(|m| self.sleeps.len() as u32 > m) {
            return false;
        }
        tokio::time::sleep(Duration::from_secs_f64(sec * self.sleep_scale)).await;
        true
    }
    async fn listen_key(&mut self, method: &str) -> Result<String, String> {
        let m: &'static str = match method { "POST" => "POST", "PUT" => "PUT", _ => "DELETE" };
        self.bx.listen_key(m).await
    }
    async fn connect(&mut self, url: &str, heartbeat: f64) -> Result<(), String> {
        self.ws = None; // 앞 연결은 버린다(원본 async with 가 닫는다)
        let url = url.replacen(USER_STREAM_WS, &self.ws_base, 1);
        let (ws, _) = tokio_tungstenite::connect_async(url).await.map_err(|e| format!("WSServerHandshakeError: {e}"))?;
        self.ws = Some(ws);
        (self.heartbeat, self.last_ping) = (heartbeat, self.mono());
        Ok(())
    }
    /// 원본 ws.receive(timeout=60). heartbeat(aiohttp 120초 ping) = 그 주기로 Ping 을 보낸다. ponytail: pong 을 안 기다린다(aiohttp 는 못 받으면 끊는다) -- 실서버에서 끊김이 늦게 잡히면 넣는다.
    async fn recv(&mut self, timeout: f64) -> Result<Recv, String> {
        let deadline = self.mono() + timeout;
        loop {
            if self.heartbeat > 0.0 && self.mono() - self.last_ping >= self.heartbeat {
                self.last_ping = self.mono();
                if let Some(ws) = self.ws.as_mut() {
                    ws.send(Message::Ping(Vec::new().into())).await.map_err(|e| format!("ClientConnectionResetError: {e}"))?;
                }
            }
            let left = (deadline - self.mono()).min(if self.heartbeat > 0.0 { self.heartbeat } else { f64::INFINITY });
            if left <= 0.0 {
                return Ok(Recv::Timeout);
            }
            let Some(ws) = self.ws.as_mut() else { return Ok(Recv::Other) };
            match tokio::time::timeout(Duration::from_secs_f64(left), ws.next()).await {
                Err(_) if self.mono() >= deadline => return Ok(Recv::Timeout),
                Err(_) => continue, // ping 차례
                Ok(Some(Ok(Message::Text(t)))) => return Ok(Recv::Text(t.to_string())),
                Ok(Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_)))) => continue,
                Ok(Some(Ok(_))) | Ok(None) => {
                    self.ws = None;
                    return Ok(Recv::Other); // CLOSE·바이너리·끝 -- 다시 연결
                }
                Ok(Some(Err(e))) => {
                    self.ws = None;
                    return Err(format!("ClientError: {e}"));
                }
            }
        }
    }
    fn wake(&mut self) {
        self.hub.wake.notify_waiters();
    }
    fn dirty(&mut self) {
        self.hub.dirty.notify_one(); // acct_dirty.set() -- 갱신기가 한 번 받는다(여러 번 와도 한 번 -- 원본 Event 와 같다)
    }
    fn event(&mut self, msg: &Value) {
        self.hub.stream_account.lock().unwrap().apply(msg);
    }
    fn tp_leg_filled(&mut self, s: &str, ps: &str, c: &str) -> Result<(), String> {
        self.hub.tp_fills.lock().unwrap().push((s.into(), ps.into(), c.into()));
        Ok(())
    }
    fn log(&mut self, s: String) {
        self.hub.logs.lock().unwrap().push(s);
    }
}
