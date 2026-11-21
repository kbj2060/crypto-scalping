//! 화면 쪽 웹소켓 서버. 접속 → 첫 메시지 `{"subscribe":[..]}` → 심볼마다 스냅숏 1회 → 이후 사건마다 밀기.
//! 느린 구독자: broadcast 채널이 구독자마다 상한(QUEUE_CAP)까지만 쌓고, 넘치면 그 구독자만 Lagged 가 되어 끊는다.
//! 끊는 쪽을 고른 이유: 체결은 누적(풋프린트·수급)이라 몇 건을 조용히 합치거나 건너뛰면 화면 합계가 틀린다 --
//! 끊으면 화면이 다시 붙어 새 스냅숏을 받는다. 보내는 쪽(feed)과 다른 구독자는 기다리지 않는다.
//!
//! 문 단속(docs/rtcore_browser_route_20261120.md §5 «코드가 할 일»). 읽기 전용 시세만 내보낸다 -- 계좌·주문은 이 문으로 안 낸다(다른 문).
//! - 경로: `RTCORE_PATH`(기본 `/rt`)로 온 업그레이드만. 그 밖은 404. Cloudflare 규칙이 `^/rt$` 를 이 포트로 보낸다(경로는 안 잘린다).
//! - Origin: `RTCORE_ALLOW_ORIGINS`(쉼표, 예 `https://<호스트>`)에 든 것만. **Origin 머리가 없는 요청은 늘 받는다**:
//!   브라우저는 웹소켓에 Origin 을 반드시 싣고 페이지가 바꿀 수 없다 -- 다른 사이트 페이지가 로그인된 사용자의 쿠키로 여는 공격(CSWSH)은
//!   늘 «목록 밖 Origin» 으로 온다. Origin 이 없는 것은 브라우저가 아닌 도구(그림자 대조기·휴대폰 앱)다. 그래서 목록이 비면
//!   «브라우저는 하나도 못 붙음» 이 되어 기본값이 «아무나» 가 되지 않는다. 사람 인증은 앞단 Cloudflare Access 몫이다.
//!   2026-11-21 더함 -- **같은 호스트**: Origin 의 호스트·포트가 그 요청의 `Host` 머리와 같으면 목록에 없어도 받는다(`same_host`).
//!   공개 호스트 이름은 비공개 값이라 저장소·문서에 못 적는다 -- 사용자가 서버 환경에 따로 적지 않아도 화면(같은 호스트의 페이지)이 붙게.
//!   CSWSH 는 계속 막힌다: 남의 사이트 페이지가 열면 Origin 은 그 사이트, Host 는 우리 호스트라 다르다.
//!   근거: cloudflared 는 `httpHostHeader` 를 안 쓰면 원래 요청의 Host 를 그대로 넘긴다 -- cloudflared 소스 ingress/origin_proxy.go
//!   `if o.hostHeader != "" { req.Header.Set("X-Forwarded-Host", req.Host); req.Host = o.hostHeader }`(빈 값이면 그 블록을 건너뛰어 req.Host 가
//!   들어온 그대로 · 2026-11-21 확인. 공식 문서 origin-parameters 는 기본값 "" 만 적고 빈 값일 때 무엇이 가는지는 안 적는다).
//!   🔴터널 설정에 httpHostHeader 를 넣으면 Host 가 그 값(예: 127.0.0.1:8788)이 되어 같은 호스트 판정이 늘 거짓이 된다 -- 그때는 목록에 적는다.
//! - 동시 연결 `RTCORE_MAX_CONNS`(기본 16 -- 사용자 한 명의 데스크톱·휴대폰·탭 몇 개) · 연결당 구독 심볼 `RTCORE_MAX_SUBS`(기본 8).
//!   연결 수 × QUEUE_CAP 이 큐 메모리 상한이다.
//! - ping: `RTCORE_PING_SECS`(기본 20)마다 보내고, 다음 ping 때까지 아무것도 못 받았으면(pong 포함) 끊는다 -- Cloudflare 가 조용한
//!   웹소켓을 닫고(시간 미공개), 반쯤 죽은 연결이 칸을 쥐고 있지 않게.
//! - 받는 메시지 상한 MAX_IN · 핸드셰이크·첫 구독 각 10초.
use crate::proto::{Stamped, Subscribe};
use crate::state::State;
use crate::{now_us, record_latency};
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::broadcast::error::{RecvError, TryRecvError};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::{interval_at, timeout, Instant};
use tokio_tungstenite::tungstenite::handshake::server::{ErrorResponse, Request, Response};
use tokio_tungstenite::tungstenite::http::StatusCode;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::{CloseFrame, WebSocketConfig};
use tokio_tungstenite::{accept_hdr_async_with_config, tungstenite::Message};

type Err = Box<dyn std::error::Error + Send + Sync>;
const BATCH: usize = 64;
/// 받는 메시지 상한. 구독 메시지는 심볼 8개여도 200바이트 안팎이다.
pub const MAX_IN: usize = 4096;

#[derive(Clone, Debug)]
pub struct HubCfg {
    pub path: String,
    pub origins: Vec<String>,
    pub max_conns: usize,
    pub max_subs: usize,
    pub ping: Duration,
}

impl HubCfg {
    pub fn from_env() -> HubCfg {
        let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
        HubCfg {
            path: env("RTCORE_PATH", "/rt"),
            origins: env("RTCORE_ALLOW_ORIGINS", "").split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
            max_conns: env("RTCORE_MAX_CONNS", "16").parse().unwrap_or(16),
            max_subs: env("RTCORE_MAX_SUBS", "8").parse().unwrap_or(8),
            ping: Duration::from_secs_f64(env("RTCORE_PING_SECS", "20").parse().unwrap_or(20.0)),
        }
    }
}

pub async fn serve(listener: TcpListener, state: Arc<State>, cfg: HubCfg) {
    let cfg = Arc::new(cfg);
    let slots = Arc::new(Semaphore::new(cfg.max_conns));
    loop {
        let Ok((tcp, peer)) = listener.accept().await else { continue };
        let _ = tcp.set_nodelay(true);
        let (state, cfg) = (state.clone(), cfg.clone());
        let slot = slots.clone().try_acquire_owned().ok(); // 칸은 이 연결 태스크가 끝날 때 돌려준다
        tokio::spawn(async move {
            if let Err(e) = client(tcp, state, cfg, slot).await {
                println!("rtcore hub {peer}: {e}");
            }
        });
    }
}

/// 업그레이드 요청 검사(머리 주석의 순서: 경로 → Origin → 칸).
fn gate(cfg: &HubCfg, req: &Request, has_slot: bool) -> Result<(), (StatusCode, &'static str)> {
    if req.uri().path() != cfg.path {
        return Err((StatusCode::NOT_FOUND, "경로"));
    }
    if let Some(o) = req.headers().get("origin") {
        let host = req.headers().get("host").and_then(|h| h.to_str().ok());
        let same = o.to_str().ok().zip(host).is_some_and(|(o, h)| same_host(o, h));
        if !same && !cfg.origins.iter().any(|a| a.as_bytes() == o.as_bytes()) {
            return Err((StatusCode::FORBIDDEN, "Origin"));
        }
    }
    if !has_slot {
        return Err((StatusCode::SERVICE_UNAVAILABLE, "연결 상한"));
    }
    Ok(())
}

/// Origin(`scheme://호스트[:포트]`)과 Host 머리(`호스트[:포트]`)가 같은 곳인가. 호스트는 대소문자 무시(RFC 3986 §3.2.2) ·
/// 포트가 없으면 Origin 의 기본 포트(https 443 · http 80)로 본다 -- 브라우저는 기본 포트를 Origin·Host 어디에도 안 적는다.
/// Origin 에 경로·사용자 정보가 있거나 http(s) 가 아니거나(예: `null`) Host 가 비면 거짓.
pub fn same_host(origin: &str, host: &str) -> bool {
    let o = origin.trim().to_ascii_lowercase();
    let h = host.trim().to_ascii_lowercase();
    let Some((scheme, auth)) = o.split_once("://") else { return false };
    let def = match scheme {
        "https" => "443",
        "http" => "80",
        _ => return false,
    };
    let hp = |a: &str| -> Option<(String, String)> {
        if a.is_empty() || a.contains(['/', '@', '?', '#', ' ']) {
            return None;
        }
        let (host, port) = match a.strip_prefix('[') {
            Some(rest) => {
                let (h6, tail) = rest.split_once(']')?; // [IPv6]:포트
                let port = match tail.strip_prefix(':') {
                    Some(p) => p,
                    None if tail.is_empty() => def,
                    None => return None,
                };
                (format!("[{h6}]"), port)
            }
            None => match a.rsplit_once(':') {
                Some((h, p)) => (h.to_string(), p),
                None => (a.to_string(), def),
            },
        };
        (!host.is_empty() && !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit())).then(|| (host, port.trim_start_matches('0').to_string()))
    };
    matches!((hp(auth), hp(&h)), (Some(a), Some(b)) if a == b)
}

fn text(m: &Stamped) -> Message {
    Message::text(serde_json::to_string(m).expect("Stamped 는 늘 직렬화된다"))
}

// ponytail: 구독은 첫 메시지 한 번뿐 -- 연결 중 심볼 바꾸기가 필요하면 그때 받는다(지금은 다시 접속).
async fn client(tcp: TcpStream, state: Arc<State>, cfg: Arc<HubCfg>, slot: Option<OwnedSemaphorePermit>) -> Result<(), Err> {
    let cb = |req: &Request, resp: Response| match gate(&cfg, req, slot.is_some()) {
        Ok(()) => Ok(resp),
        Err((code, why)) => {
            let mut r = ErrorResponse::new(Some(why.into()));
            *r.status_mut() = code;
            Err(r)
        }
    };
    let conf = WebSocketConfig::default().max_message_size(Some(MAX_IN)).max_frame_size(Some(MAX_IN));
    let (mut out, mut inp) = timeout(Duration::from_secs(10), accept_hdr_async_with_config(tcp, cb, Some(conf))).await??.split();
    // 스냅숏보다 먼저 구독해야 그 사이 사건을 놓치지 않는다(겹치는 것은 seq 로 거른다).
    let mut rx = state.tx.subscribe();
    let req: Subscribe = loop {
        match timeout(Duration::from_secs(10), inp.next()).await? {
            Some(Ok(Message::Text(t))) => break serde_json::from_str(t.as_str())?,
            Some(Ok(_)) => continue,
            Some(Err(e)) => return Err(e.into()), // 큰 메시지(MAX_IN 초과)도 여기
            None => return Ok(()),
        }
    };
    if req.subscribe.len() > cfg.max_subs {
        let why = format!("구독 심볼 {}개 > 상한 {}", req.subscribe.len(), cfg.max_subs);
        out.send(Message::Close(Some(CloseFrame { code: CloseCode::Policy, reason: why.clone().into() }))).await?;
        return Err(why.into());
    }
    let mut seen: HashMap<String, u64> = HashMap::new();
    for s in req.subscribe {
        let s = s.to_lowercase();
        if let Some(mut snap) = state.snapshot(&s, now_us()) {
            seen.insert(s, snap.seq);
            snap.tx_us = now_us();
            out.send(text(&snap)).await?;
        }
    }
    // 2026-11-21 «모든 종목» 메시지(`s = "*"` · 마지막 체결가 Prices)는 구독과 상관없이 모두에게 -- 붙을 때 전부 한 번, 그 뒤 번호가 큰 것만.
    let mut prices = state.prices_snapshot();
    seen.insert("*".into(), prices.seq);
    prices.tx_us = now_us();
    out.send(text(&prices)).await?;
    let mut ping = interval_at(Instant::now() + cfg.ping, cfg.ping);
    let mut heard = true; // 지난 ping 뒤로 무엇이든 받았나(브라우저는 ping 에 pong 을 스스로 낸다)
    loop {
        tokio::select! {
            r = rx.recv() => {
                // 이미 와 있는 것은 최대 BATCH 건까지 버퍼에 넣고 flush 한 번(syscall 1회). 조용할 땐 1건씩 곧바로 나간다.
                // tx_us = 쓰기 버퍼에 넣은 시각 -- 실제 소켓 쓰기는 같은 묶음의 flush 때(묶음 상한이 그 차이를 묶는다).
                let mut r = r;
                for k in 1..=BATCH {
                    match r {
                        Ok(mut m) => {
                            if seen.get(&m.s).is_some_and(|&seq0| m.seq > seq0) {
                                m.tx_us = now_us();
                                record_latency(m.tx_us - m.rx_us);
                                out.feed(text(&m)).await?;
                            }
                        }
                        Err(RecvError::Lagged(n)) => return Err(format!("느린 구독자: {n}건 밀려 끊는다").into()),
                        Err(RecvError::Closed) => return Ok(()),
                    }
                    if k == BATCH { break } // 여기서 더 꺼내면 처리 못 하고 버려진다
                    r = match rx.try_recv() {
                        Ok(m) => Ok(m),
                        Err(TryRecvError::Lagged(n)) => Err(RecvError::Lagged(n)),
                        Err(_) => break,
                    };
                }
                out.flush().await?;
            },
            _ = ping.tick() => {
                if !heard {
                    return Err(format!("{:?} 동안 pong 없음: 끊는다", cfg.ping).into());
                }
                heard = false;
                out.send(Message::Ping(Vec::new().into())).await?;
            },
            r = inp.next() => match r {
                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => return Ok(()), // 탭 닫힘·끊김은 로그 안 남김
                Some(Ok(_)) => heard = true,
            },
        }
    }
}
