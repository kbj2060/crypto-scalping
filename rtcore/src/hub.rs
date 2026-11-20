//! 화면 쪽 웹소켓 서버. 접속 → 첫 메시지 `{"subscribe":[..]}` → 심볼마다 스냅숏 1회 → 이후 사건마다 밀기.
//! 느린 구독자: broadcast 채널이 구독자마다 상한(QUEUE_CAP)까지만 쌓고, 넘치면 그 구독자만 Lagged 가 되어 끊는다.
//! 끊는 쪽을 고른 이유: 체결은 누적(풋프린트·수급)이라 몇 건을 조용히 합치거나 건너뛰면 화면 합계가 틀린다 --
//! 끊으면 화면이 다시 붙어 새 스냅숏을 받는다. 보내는 쪽(feed)과 다른 구독자는 기다리지 않는다.
use crate::proto::{Stamped, Subscribe};
use crate::state::State;
use crate::{now_us, record_latency};
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::broadcast::error::{RecvError, TryRecvError};
use tokio::time::timeout;
use tokio_tungstenite::{accept_async, tungstenite::Message};

type Err = Box<dyn std::error::Error + Send + Sync>;
const BATCH: usize = 64;

pub async fn serve(listener: TcpListener, state: Arc<State>) {
    loop {
        let Ok((tcp, peer)) = listener.accept().await else { continue };
        let _ = tcp.set_nodelay(true);
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = client(tcp, state).await {
                println!("rtcore hub {peer}: {e}");
            }
        });
    }
}

fn text(m: &Stamped) -> Message {
    Message::text(serde_json::to_string(m).expect("Stamped 는 늘 직렬화된다"))
}

// ponytail: 구독은 첫 메시지 한 번뿐 -- 연결 중 심볼 바꾸기가 필요하면 그때 받는다(지금은 다시 접속).
async fn client(tcp: TcpStream, state: Arc<State>) -> Result<(), Err> {
    let (mut out, mut inp) = accept_async(tcp).await?.split();
    // 스냅숏보다 먼저 구독해야 그 사이 사건을 놓치지 않는다(겹치는 것은 seq 로 거른다).
    let mut rx = state.tx.subscribe();
    let req: Subscribe = loop {
        match timeout(Duration::from_secs(10), inp.next()).await? {
            Some(Ok(Message::Text(t))) => break serde_json::from_str(t.as_str())?,
            Some(Ok(_)) => continue,
            _ => return Ok(()),
        }
    };
    let mut seen: HashMap<String, u64> = HashMap::new();
    for s in req.subscribe {
        let s = s.to_lowercase();
        if let Some(mut snap) = state.snapshot(&s, now_us()) {
            seen.insert(s, snap.seq);
            snap.tx_us = now_us();
            out.send(text(&snap)).await?;
        }
    }
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
            r = inp.next() => match r {
                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => return Ok(()),
                Some(Ok(_)) => {}
            },
        }
    }
}
