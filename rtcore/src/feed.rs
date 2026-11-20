//! 바이낸스 USDS-M 선물 웹소켓 수신. (심볼, 스트림)마다 연결 하나 -- 저장소의 살아 있는 수집기와 같은 주소 규칙:
//! `@trade`·`@bookTicker` = `/ws/<sym>@..`, `@markPrice@1s` = `/market/ws/<sym>@..`
//! (scripts/live_binance_ctx_collector_20261001.py: 마크는 `/ws/` 로 붙으면 이벤트 0).
//! `@aggTrade` 는 쓰지 않는다: 2026-09-02 부터 배달이 멈췄다(dashboard/coin_flow.py 주석). 현물 `@aggTrade` 는 살아 있다(아래).
//!
//! 나머지 원천(1초 수급 · 2026-11-20). 주소는 전부 lib.rs Cfg(환경변수)로 덮는다. 파싱은 파이썬 수신 코드를 옮겼다:
//! - 바이낸스 강제청산 `/market/ws/!forceOrder@arr`(전 종목 한 연결 · server.py collect_force_orders): `o.z or o.q` 수량,
//!   `o.ap or o.p` 가격, `o.T or 수신 시각`, S=SELL → 롱 청산. 파이썬처럼 수량 0 도 거르지 않는다.
//! - 바이낸스 현물 `<sym>@aggTrade`(종목마다 한 연결 · coin_flow.py collect_spot_flow): e=aggTrade 만, 가격·수량 > 0, m = 매도 공격.
//! - OKX `/ws/v5/public` 한 연결(coin_flow.py collect_okx_flow): 접속 뒤 trades·open-interest(종목마다) + liquidation-orders(SWAP 전체)
//!   구독. sz × ctVal, side 가 곧 테이커 방향(바이낸스 m 처럼 뒤집지 않는다), 청산은 posSide=long → 롱 청산 · bkPx(파산가) 로 USD.
//! - Bybit `/v5/public/linear` 한 연결(scripts/live_bybit_liquidation_collector_20261003.py parse): allLiquidation.<SYM> 구독,
//!   🔴S 는 **청산된 포지션 방향**(Buy = 롱 청산) -- 바이낸스 S(강제 주문 방향)와 반대.
//! 파이썬 `x or y` · float() · int() 의 뜻은 py_* 가 옮긴다. 파이썬과 다른 곳(일부러): 깨진 메시지에 파이썬은 연결을 끊고
//!   3초 쉬고 다시 붙어 그 사이를 잃는다 -- 여기는 그 메시지만 건너뛴다(BAD_JSON·SKIPPED).
//!
//! ⚠️실서버 대조 전: 아래 필드 모양은 바이낸스 공식 문서 필드명을 기억으로 옮기고 저장소 수신 코드
//! (coin_flow.py `@trade` p/q/T/m/t · ctx 수집기 parse_mark E/s/p/i/r/T · app.js `@trade`)와 맞춘 것이다.
//! bookTicker 의 u/b/B/a/A/T 는 저장소에 파싱 코드가 없어 문서 기억뿐이다. M1 통과 전 서버에서 녹화본으로 대조한다.
// ponytail: 연결을 (심볼×3)개 연다 -- 심볼이 늘어 연결 한도가 문제되면 /stream?streams= 묶음으로 바꾼다(감싸기 모양 대조 필요).
use crate::proto::{Event, Stamped, Venue};
use crate::supply::okx_ct_val;
use crate::state::State;
use crate::{now_us, BAD_JSON, LIVE_FEEDS, RECONNECTS, SKIPPED};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::Value;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{sleep, sleep_until, timeout, timeout_at, Instant};
use tokio_tungstenite::{connect_async, tungstenite::Message};

const BACKOFF_MIN: Duration = Duration::from_millis(500);
const BACKOFF_MAX: Duration = Duration::from_secs(30);
/// 조용한 스트림(청산)·OKX(30초 무응답이면 끊음)·Bybit(20초 ping 권장)에 보내는 주기. 침묵 판정(기본 30초)보다 짧아야 한다.
const PING_EVERY: Duration = Duration::from_secs(20);

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    /// 바이낸스 선물(@trade·@bookTicker·@markPrice·!forceOrder) -- `parse`.
    Futures,
    Spot,
    Okx,
    Bybit,
}

/// 수신 연결 하나. `syms` = 이 연결이 맡은 종목(현물·OKX 의 connected·errors 를 적을 곳, 현물은 메시지 대신 이 종목으로 싣는다).
pub struct Src {
    pub url: String,
    pub kind: Kind,
    /// 접속 직후 보낼 구독 JSON(OKX·Bybit).
    pub sub: Option<String>,
    /// PING_EVERY 마다 보낼 것(바이낸스는 웹소켓 ping 프레임 -- 거래소가 pong 프레임으로 답해 침묵 판정이 풀린다).
    pub ping: Option<Message>,
    pub syms: Vec<String>,
}

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
    /// 파이썬은 `o` 만 본다(json.loads(..).get("o")) -- 필드 해석은 parse 의 py_* 로.
    #[serde(rename = "forceOrder")]
    Force { o: Value },
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

/// 파이썬 참거짓(None·""·0·False·빈 목록 = 거짓). `"0"` 문자열은 **참**이다 -- `o.get("z") or o.get("q")` 가 "0" 에서 멈춘다.
fn py_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// 파이썬 `a or b or ...`: 첫 참인 값, 없으면 None(= 뒤의 기본값).
fn py_or<'a>(vs: &[Option<&'a Value>]) -> Option<&'a Value> {
    vs.iter().flatten().copied().find(|v| py_truthy(v))
}

/// 파이썬 float(x). 못 읽으면 None(파이썬은 예외 → 재연결 · 여기는 메시지 건너뜀). 무한·NaN 도 None(파이썬은 inf 를 그대로 쌓는다).
fn py_float(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => num(s.trim()),
        _ => None,
    }
}

/// 파이썬 int(x): 정수 · 실수는 0 쪽으로 자름 · 문자열은 정수 꼴만.
fn py_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().filter(|f| f.is_finite()).map(|f| f.trunc() as i64)),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// `float(d.get(k) or 0.0)`.
fn py_float_or0(d: &Value, k: &str) -> Option<f64> {
    py_or(&[d.get(k)]).map_or(Some(0.0), py_float)
}

/// `int(d.get(k) or <수신 시각 ms>)` -- 파이썬 int(time.time() * 1000).
fn py_ts(d: &Value, k: &str, rx_us: i64) -> Option<i64> {
    py_or(&[d.get(k)]).map_or(Some(rx_us.div_euclid(1000)), py_int)
}

fn stamped(s: &str, ex_ms: i64, rx_us: i64, ev: Event) -> Stamped {
    Stamped { s: s.to_lowercase(), seq: 0, ex_ms, rx_us, tx_us: 0, ev }
}

fn json_or_count(text: &str) -> Option<Value> {
    let v = serde_json::from_str(text);
    if v.is_err() {
        BAD_JSON.fetch_add(1, Relaxed);
    }
    v.ok()
}

/// 바이낸스 현물 @aggTrade(coin_flow.py collect_spot_flow 879~898). `sym` = 연결이 맡은 종목(파이썬도 메시지의 s 를 안 본다).
pub fn parse_spot(text: &str, sym: &str, rx_us: i64) -> Option<Stamped> {
    let t = json_or_count(text)?;
    if t.get("e").and_then(Value::as_str) != Some("aggTrade") {
        SKIPPED.fetch_add(1, Relaxed);
        return None;
    }
    let (px, qty) = (py_float_or0(&t, "p"), py_float_or0(&t, "q"));
    let (Some(px), Some(qty)) = (px, qty) else {
        SKIPPED.fetch_add(1, Relaxed);
        return None;
    };
    if !(px > 0.0 && qty > 0.0) {
        SKIPPED.fetch_add(1, Relaxed);
        return None;
    }
    let ts = py_ts(&t, "T", rx_us)?;
    let sell = t.get("m").is_some_and(py_truthy); // 매수자가 메이커 -> 공격자는 매도
    Some(stamped(sym, ts, rx_us, Event::Flow { venue: Venue::Spot, px, qty, sell }))
}

/// `ETH-USDT-SWAP` → `ethusdt`.
fn okx_sym(inst: &str) -> Option<String> {
    inst.strip_suffix("-USDT-SWAP").map(|b| format!("{}usdt", b.to_lowercase()))
}

/// OKX 공개 채널(coin_flow.py collect_okx_flow 750~832). 한 메시지에 여러 건. 못 읽는 건은 그 건만 건너뛴다.
pub fn parse_okx(text: &str, rx_us: i64) -> Vec<Stamped> {
    if text == "pong" {
        return vec![];
    }
    let Some(d) = json_or_count(text) else { return vec![] };
    if let Some(ev) = d.get("event").filter(|e| py_truthy(e)) {
        if ev.as_str() == Some("error") {
            // 파이썬은 끊고 다시 붙는다(곧바로 -- 쉬지 않음). 여기는 남기기만: 같은 구독은 다시 붙어도 또 거부된다.
            println!("rtcore okx: 구독 거부 {}", d.get("msg").map_or(String::new(), Value::to_string));
        }
        return vec![];
    }
    let arg = d.get("arg").cloned().unwrap_or(Value::Null);
    let data = d.get("data").and_then(Value::as_array).map_or(&[][..], |v| v.as_slice());
    let mut out = vec![];
    let mut n_bad = 0u64;
    match arg.get("channel").and_then(Value::as_str) {
        Some("trades") => {
            let inst = arg.get("instId").and_then(Value::as_str).unwrap_or("");
            let (Some(sym), Some(ct)) = (okx_sym(inst), okx_ct_val(inst)) else { return vec![] };
            for t in data {
                let (Some(px), Some(sz)) = (py_float_or0(t, "px"), py_float_or0(t, "sz")) else {
                    n_bad += 1;
                    continue;
                };
                let qty = sz * ct;
                let side = t.get("side").and_then(Value::as_str).unwrap_or("");
                if !(px > 0.0 && qty > 0.0 && (side == "buy" || side == "sell")) {
                    n_bad += 1;
                    continue;
                }
                let Some(ts) = py_ts(t, "ts", rx_us) else {
                    n_bad += 1;
                    continue;
                };
                out.push(stamped(&sym, ts, rx_us, Event::Flow { venue: Venue::Okx, px, qty, sell: side == "sell" }));
            }
        }
        Some("open-interest") => {
            let inst = arg.get("instId").and_then(Value::as_str).unwrap_or("");
            let Some(sym) = okx_sym(inst) else { return vec![] };
            for r in data {
                let oi = py_float_or0(r, "oiCcy"); // oiCcy = 기초자산(ETH)
                let ts = py_or(&[r.get("ts")]).map_or(Some(0), py_int);
                match (oi, ts) {
                    (Some(oi), Some(ts)) if oi > 0.0 && ts > 0 => out.push(stamped(&sym, ts, rx_us, Event::Oi { venue: Venue::Okx, oi })),
                    _ => n_bad += 1,
                }
            }
        }
        Some("liquidation-orders") => {
            for r in data {
                let inst = r.get("instId").and_then(Value::as_str).unwrap_or("");
                // instType:SWAP 은 전 종목이 온다 -- 계약 크기를 아는 종목만(파이썬도 ctVal 모르는 종목은 레인을 안 켠다)
                let (Some(sym), Some(ct)) = (okx_sym(inst), okx_ct_val(inst)) else { continue };
                for det in r.get("details").and_then(Value::as_array).map_or(&[][..], |v| v.as_slice()) {
                    let (Some(sz), Some(px)) = (py_float_or0(det, "sz"), py_float_or0(det, "bkPx")) else {
                        n_bad += 1;
                        continue;
                    };
                    let qty = sz * ct;
                    if qty <= 0.0 {
                        continue;
                    }
                    let Some(ts) = py_ts(det, "ts", rx_us) else {
                        n_bad += 1;
                        continue;
                    };
                    let long = det.get("posSide").and_then(Value::as_str) == Some("long");
                    out.push(stamped(&sym, ts, rx_us, Event::Liq { venue: Venue::Okx, long, qty, px }));
                }
            }
        }
        _ => {}
    }
    SKIPPED.fetch_add(n_bad, Relaxed);
    out
}

/// Bybit allLiquidation(수집기 parse). 파이썬은 한 건이라도 못 읽으면 메시지 전체가 예외라 **통째로** 빠진다 -- 같게 둔다.
pub fn parse_bybit(text: &str, rx_us: i64) -> Vec<Stamped> {
    let Some(m) = json_or_count(text) else { return vec![] };
    if !m.get("topic").and_then(Value::as_str).is_some_and(|t| t.starts_with("allLiquidation.")) {
        return vec![]; // 구독 응답·pong
    }
    let data = m.get("data").and_then(Value::as_array).map_or(&[][..], |v| v.as_slice());
    let one = |d: &Value| -> Option<Stamped> {
        let ts = py_int(d.get("T")?)?;
        let s = d.get("s")?.as_str()?;
        let side = d.get("S")?.as_str()?;
        let (qty, px) = (py_float(d.get("v")?)?, py_float(d.get("p")?)?);
        Some(stamped(s, ts, rx_us, Event::Liq { venue: Venue::Bybit, long: side == "Buy", qty, px }))
    };
    let out: Option<Vec<Stamped>> = data.iter().map(one).collect();
    out.unwrap_or_else(|| {
        SKIPPED.fetch_add(1, Relaxed);
        vec![]
    })
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
        Raw::Force { o } => {
            // server.py collect_force_orders 1831~1840 -- 모르는 종목은 State 가 버린다(파이썬 LIQ_ASSET_BY_SYMBOL).
            let s = o.get("s").and_then(Value::as_str)?.to_string();
            let qty = py_or(&[o.get("z"), o.get("q")]).map_or(Some(0.0), py_float);
            let px = py_or(&[o.get("ap"), o.get("p")]).map_or(Some(0.0), py_float);
            let (Some(qty), Some(px), Some(ts)) = (qty, px, py_ts(&o, "T", rx_us)) else {
                SKIPPED.fetch_add(1, Relaxed);
                return None;
            };
            let long = o.get("S").and_then(Value::as_str) == Some("SELL"); // 롱 청산 = 시장에 SELL
            (s, ts, Event::Liq { venue: Venue::Bn, long, qty, px })
        }
        Raw::Mark { e_ms, s, p, i, r, next } => {
            (s, e_ms, Event::Mark { mark: num(&p)?, index: num(&i)?, funding: num(&r)?, next_funding_ms: next })
        }
    };
    Some(Stamped { s: s.to_lowercase(), seq: 0, ex_ms, rx_us, tx_us: 0, ev })
}

struct Live;
impl Live {
    fn up() -> Live {
        LIVE_FEEDS.fetch_add(1, Relaxed);
        Live
    }
}
impl Drop for Live {
    fn drop(&mut self) {
        LIVE_FEEDS.fetch_sub(1, Relaxed);
    }
}

/// 바이낸스 선물 한 스트림(옛 호출 모양).
pub async fn run(url: String, state: Arc<State>, silence: Duration) {
    run_src(Src { url, kind: Kind::Futures, sub: None, ping: None, syms: vec![] }, state, silence).await
}

fn parse_any(src: &Src, text: &str, rx: i64) -> Vec<Stamped> {
    match src.kind {
        Kind::Futures => parse(text, rx).into_iter().collect(),
        Kind::Spot => src.syms.first().and_then(|s| parse_spot(text, s, rx)).into_iter().collect(),
        Kind::Okx => parse_okx(text, rx),
        Kind::Bybit => parse_bybit(text, rx),
    }
}

fn status(src: &Src, state: &State, up: bool) {
    let venue = match src.kind {
        Kind::Spot => Venue::Spot,
        Kind::Okx => Venue::Okx,
        _ => return,
    };
    for s in &src.syms {
        state.feed_status(s, venue, up);
    }
}

/// 영원히 돈다. 끊김·침묵(`silence` 동안 아무 프레임 없음)·접속 실패 → 재연결.
/// 바이낸스 24시간 강제 끊김도 같은 길이다: 데이터를 받던 연결이 끊기면 첫 재시도는 기다리지 않는다.
// ponytail: 24시간 끊김 때 재연결 사이(수십~수백 ms)가 빈다 -- 빈 구간이 문제되면 23시간에 새 연결을 먼저 열고 겹쳐 넘긴다.
pub async fn run_src(src: Src, state: Arc<State>, silence: Duration) {
    let url = src.url.as_str();
    let mut backoff = Duration::ZERO;
    loop {
        let mut got = false;
        match timeout(Duration::from_secs(10), connect_async(url)).await {
            Ok(Ok((mut ws, _))) => {
                let _live = Live::up(); // 패닉으로 풀려도 Drop 이 센다
                if let Some(sym) = url.rsplit('/').next().and_then(|t| t.strip_suffix("@trade")) {
                    state.reset_orders(sym); // 끊김 전 열린 주문 묶음은 버린다(coin_flow.py 613)
                }
                let subscribed = match &src.sub {
                    Some(sub) => ws.send(Message::text(sub.clone())).await.map_err(|e| println!("rtcore feed {url}: 구독 전송 실패 {e}")).is_ok(),
                    None => true,
                };
                if subscribed {
                    status(&src, &state, true);
                    let mut last = Instant::now();
                    let mut ping_at = last + PING_EVERY;
                    loop {
                        let r = tokio::select! {
                            _ = sleep_until(ping_at), if src.ping.is_some() => {
                                ping_at += PING_EVERY;
                                if let Err(e) = ws.send(src.ping.clone().expect("if 로 걸렀다")).await {
                                    println!("rtcore feed {url}: ping 실패 {e}");
                                    break;
                                }
                                continue;
                            }
                            r = timeout_at(last + silence, ws.next()) => r,
                        };
                        match r {
                            Err(_) => {
                                println!("rtcore feed {url}: {}s 침묵 -> 재연결", silence.as_secs_f64());
                                break;
                            }
                            Ok(Some(Ok(Message::Text(t)))) => {
                                let rx = now_us();
                                last = Instant::now();
                                got = true;
                                for m in parse_any(&src, t.as_str(), rx) {
                                    state.apply(m);
                                }
                            }
                            Ok(Some(Ok(Message::Close(_)))) | Ok(None) => break,
                            Ok(Some(Ok(_))) => last = Instant::now(), // ping 은 tungstenite 가 알아서 pong 한다 · pong 도 살아 있다는 뜻
                            Ok(Some(Err(e))) => {
                                println!("rtcore feed {url}: {e}");
                                break;
                            }
                        }
                    }
                    status(&src, &state, false);
                }
            }
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
