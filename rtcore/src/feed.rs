//! 바이낸스 USDS-M 선물 웹소켓 수신. (심볼, 스트림)마다 연결 하나 -- 저장소의 살아 있는 수집기와 같은 주소 규칙:
//! `@trade`·`@bookTicker` = `/ws/<sym>@..`, `@markPrice@1s` = `/market/ws/<sym>@..`
//! (scripts/live_binance_ctx_collector_20261001.py: 마크는 `/ws/` 로 붙으면 이벤트 0).
//! `@aggTrade` 는 쓰지 않는다: 2026-09-02 부터 배달이 멈췄다(dashboard/coin_flow.py 주석).
//!
//! ⚠️실서버 대조 전: 아래 필드 모양은 바이낸스 공식 문서 필드명을 기억으로 옮기고 저장소 수신 코드
//! (coin_flow.py `@trade` p/q/T/m/t · ctx 수집기 parse_mark E/s/p/i/r/T · app.js `@trade`)와 맞춘 것이다.
//! bookTicker 의 u/b/B/a/A/T 는 저장소에 파싱 코드가 없어 문서 기억뿐이다. M1 통과 전 서버에서 녹화본으로 대조한다.
// ponytail: 연결을 (심볼×3)개 연다 -- 심볼이 늘어 연결 한도가 문제되면 /stream?streams= 묶음으로 바꾼다(감싸기 모양 대조 필요).
use crate::proto::{Event, Stamped};
use crate::state::State;
use crate::{now_us, BAD_JSON, RECONNECTS, SKIPPED};
use futures_util::StreamExt;
use serde::Deserialize;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{sleep, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message};

const BACKOFF_MIN: Duration = Duration::from_millis(500);
const BACKOFF_MAX: Duration = Duration::from_secs(30);

#[derive(Deserialize)]
#[serde(tag = "e")]
enum Raw {
    #[serde(rename = "trade")]
    Trade {
        #[serde(rename = "T")]
        t_ms: i64,
        s: String,
        t: u64,
        p: String,
        q: String,
        m: bool,
    },
    #[serde(rename = "bookTicker")]
    Book {
        #[serde(rename = "T", default)]
        t_ms: i64,
        #[serde(rename = "E", default)]
        e_ms: i64,
        s: String,
        u: u64,
        b: String,
        #[serde(rename = "B")]
        bq: String,
        a: String,
        #[serde(rename = "A")]
        aq: String,
    },
    #[serde(rename = "markPriceUpdate")]
    Mark {
        #[serde(rename = "E")]
        e_ms: i64,
        s: String,
        p: String,
        i: String,
        r: String,
        #[serde(rename = "T", default)]
        next: i64,
    },
}

fn num(x: &str) -> Option<f64> {
    x.parse().ok().filter(|v: &f64| v.is_finite())
}

/// 쓸 수 없는 메시지는 None(건수는 BAD_JSON·SKIPPED). 연결을 끊지 않는다.
pub fn parse(text: &str, rx_us: i64) -> Option<Stamped> {
    let raw: Raw = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(_) => {
            let c = if serde_json::from_str::<serde_json::Value>(text).is_ok() { &SKIPPED } else { &BAD_JSON };
            c.fetch_add(1, Relaxed);
            return None;
        }
    };
    let (s, ex_ms, ev) = match raw {
        Raw::Trade { t_ms, s, t, p, q, m } => {
            let (px, qty) = (num(&p)?, num(&q)?);
            if !(px > 0.0 && qty > 0.0) {
                // 바이낸스가 {"p":"0","q":"0","X":"NA"} 를 섞어 보낸다(2026-09-16 실측 0.3%, coin_flow.py)
                SKIPPED.fetch_add(1, Relaxed);
                return None;
            }
            (s, t_ms, Event::Trade { id: t, px, qty, sell: m })
        }
        Raw::Book { t_ms, e_ms, s, u, b, bq, a, aq } => {
            let ev = Event::Book { u, bid: num(&b)?, bid_qty: num(&bq)?, ask: num(&a)?, ask_qty: num(&aq)? };
            (s, if t_ms > 0 { t_ms } else { e_ms }, ev)
        }
        Raw::Mark { e_ms, s, p, i, r, next } => {
            (s, e_ms, Event::Mark { mark: num(&p)?, index: num(&i)?, funding: num(&r)?, next_funding_ms: next })
        }
    };
    Some(Stamped { s: s.to_lowercase(), seq: 0, ex_ms, rx_us, tx_us: 0, ev })
}

/// 영원히 돈다. 끊김·침묵(`silence` 동안 아무 프레임 없음)·접속 실패 → 재연결.
/// 바이낸스 24시간 강제 끊김도 같은 길이다: 데이터를 받던 연결이 끊기면 첫 재시도는 기다리지 않는다.
// ponytail: 24시간 끊김 때 재연결 사이(수십~수백 ms)가 빈다 -- 빈 구간이 문제되면 23시간에 새 연결을 먼저 열고 겹쳐 넘긴다.
pub async fn run(url: String, state: Arc<State>, silence: Duration) {
    let mut backoff = Duration::ZERO;
    loop {
        let mut got = false;
        match timeout(Duration::from_secs(10), connect_async(url.as_str())).await {
            Ok(Ok((mut ws, _))) => loop {
                match timeout(silence, ws.next()).await {
                    Err(_) => {
                        println!("rtcore feed {url}: {}s 침묵 -> 재연결", silence.as_secs_f64());
                        break;
                    }
                    Ok(Some(Ok(Message::Text(t)))) => {
                        let rx = now_us();
                        got = true;
                        if let Some(m) = parse(t.as_str(), rx) {
                            state.apply(m);
                        }
                    }
                    Ok(Some(Ok(Message::Close(_)))) | Ok(None) => break,
                    Ok(Some(Ok(_))) => {} // ping 은 tungstenite 가 알아서 pong 한다
                    Ok(Some(Err(e))) => {
                        println!("rtcore feed {url}: {e}");
                        break;
                    }
                }
            },
            Ok(Err(e)) => println!("rtcore feed {url}: 접속 실패 {e}"),
            Err(_) => println!("rtcore feed {url}: 접속 10초 초과"),
        }
        RECONNECTS.fetch_add(1, Relaxed);
        if got {
            backoff = Duration::ZERO;
        }
        sleep(backoff).await;
        backoff = (backoff * 2).clamp(BACKOFF_MIN, BACKOFF_MAX);
    }
}
