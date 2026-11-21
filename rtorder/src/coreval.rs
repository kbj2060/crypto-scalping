//! 실시간 코어(rtcore) 값 → rtorder 자리(2026-11-21 · 좁은 판: rtcore 를 의존성으로 넣지 않고 **값의 모양**만 받는다).
//! 모양 = rtcore/src/quote.rs·proto.rs: `book_ticker` = (REST 꼴 dict {symbol, bidPrice, askPrice} 글자 그대로, 받은 시각 µs) ·
//! `vol_bpm`(마감 1분봉 31개 전엔 None) + `kline_rx_us` · `bar5m` = (끝 초, 종가) · 1초 가격 = (초, 가격).
//! 2026-11-21 E: 실제 구독(`subscribe`) -- rtcore 허브(ws://127.0.0.1:8788/rt · 첫 메시지 `{"subscribe":[..]}` → 심볼마다 스냅숏 → 사건 밀기)를 읽어 `Feed` 에 모은다.
//!   rtcore 를 의존성으로 넣지 않고 JSON 모양만 읽는다(proto 는 «더하기만» -- 모르는 칸·모르는 type 은 무시).
//!   주문 심볼(…USDC): book(bid_s·ask_s + rx_us) · kline(vol_bpm + rx_us) · 화면 심볼(…USDT): 1초 수급 seconds(가격 = 행[7]) · kline bar5m · 스냅숏 quote(bar5m·vol_bpm·kline_rx_us).
//!   끊기면 값은 그대로 두고 나이만 먹는다 -- `Cached` 가 한도를 넘기면 REST 로 떨어진다 · 다시 붙으면 스냅숏으로 다시 채운다.

use crate::assemble::{Cached, Quotes};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// 허브에서 받아 둔 값(심볼은 대문자). secs = 화면 심볼의 1초 마지막 체결가 [(초, 가격)] -- 초 단위로 덮어쓴다 · 최근 SEC_KEEP 초만.
#[derive(Debug, Default)]
pub struct Feed {
    pub books: HashMap<String, (Value, i64)>,
    pub vols: HashMap<String, (Option<f64>, i64)>,
    pub bar5m: HashMap<String, (i64, f64)>,
    pub secs: HashMap<String, Vec<(i64, f64)>>,
    pub connected: bool,
    pub connects: u64,
    /// 마지막으로 허브 메시지를 받은 벽시계 초(0 = 한 번도 못 받음)
    pub last_rx: f64,
    /// 구독 루프가 남긴 줄(닫힘 사유 · 같은 사유는 1분에 한 번) -- 계량 줄 자리가 비운다
    pub logs: Vec<String>,
}
/// 이만큼 허브 메시지가 없으면 «못 받음»(계량 줄 자리에 한 줄 · /internal/acct feed.stale)
pub const FEED_STALE_S: f64 = 10.0;
pub const SEC_KEEP: i64 = 1800; // 감시 1초 닿음은 무장 뒤만 본다 -- 30분이면 넉넉(ponytail: 더 길게 쉬면 그 사이는 «모름»)

impl Feed {
    /// 허브 상태(값 없이 나이만): 연결 · 붙은 횟수 · 마지막 수신 뒤 초 · 심볼별 마지막 가격 뒤 초(호가 rx · 1초가 끝 초 중 늦은 것).
    pub fn health(&self, now: f64) -> Value {
        let mut syms: std::collections::BTreeMap<String, f64> = Default::default();
        for (s, (_, rx)) in &self.books {
            syms.insert(s.clone(), sec(*rx));
        }
        for (s, v) in &self.secs {
            if let Some(&(t, _)) = v.last() {
                let e = syms.entry(s.clone()).or_insert(0.0);
                *e = e.max(t as f64);
            }
        }
        let age = |t: f64| if t > 0.0 { json!(((now - t) * 10.0).round() / 10.0) } else { Value::Null };
        json!({"connected": self.connected, "connects": self.connects, "last_rx_age_s": age(self.last_rx),
               "stale": self.last_rx <= 0.0 || now - self.last_rx > FEED_STALE_S,
               "symbols": syms.into_iter().map(|(s, t)| (s, age(t))).collect::<serde_json::Map<_, _>>()})
    }
    /// 허브 메시지 하나(Stamped JSON)를 반영한다.
    pub fn apply(&mut self, m: &Value) {
        let sym = m.get("s").and_then(Value::as_str).unwrap_or("").to_uppercase();
        let rx = m.get("rx_us").and_then(Value::as_i64).unwrap_or(0);
        match m.get("type").and_then(Value::as_str) {
            Some("book") => self.book(&sym, m, rx),
            Some("kline") => {
                self.vols.insert(sym.clone(), (m.get("vol_bpm").and_then(Value::as_f64), rx));
                if let Some(b) = pair(m.get("bar5m")) {
                    self.bar5m.insert(sym, b);
                }
            }
            Some("supply1s") => self.seconds(&sym, m.get("seconds")),
            Some("snapshot") => {
                if let Some(b) = m.get("book").filter(|b| b.is_object()) {
                    self.book(&sym, b, b.get("rx_us").and_then(Value::as_i64).unwrap_or(0));
                }
                if let Some(q) = m.get("quote").filter(|q| q.is_object()) {
                    let krx = q.get("kline_rx_us").and_then(Value::as_i64).unwrap_or(0);
                    self.vols.insert(sym.clone(), (q.get("vol_bpm").and_then(Value::as_f64), krx));
                    if let Some(b) = pair(q.get("bar5m")) {
                        self.bar5m.insert(sym.clone(), b);
                    }
                }
                self.seconds(&sym, m.get("supply").and_then(|s| s.get("seconds")));
            }
            _ => {}
        }
    }
    fn book(&mut self, sym: &str, b: &Value, rx: i64) {
        let g = |k: &str| b.get(k).and_then(Value::as_str).filter(|s| !s.is_empty()).map(String::from);
        // 글자 칸(bid_s·ask_s)이 없는 옛 코어 = 받아 두지 않는다(REST 와 같은 글자 꼴을 못 맞춘다 -- 조용히 숫자 표기로 바꾸지 않는다)
        if let (Some(bid), Some(ask)) = (g("bid_s"), g("ask_s")) {
            self.books.insert(sym.into(), (json!({"symbol": sym, "bidPrice": bid, "askPrice": ask}), rx));
        }
    }
    fn seconds(&mut self, sym: &str, rows: Option<&Value>) {
        let Some(rows) = rows.and_then(Value::as_array) else { return };
        let v = self.secs.entry(sym.into()).or_default();
        for r in rows {
            if let (Some(t), Some(px)) = (r.get(0).and_then(Value::as_i64), r.get(7).and_then(Value::as_f64)) {
                match v.binary_search_by_key(&t, |x| x.0) {
                    Ok(i) => v[i].1 = px,
                    Err(i) => v.insert(i, (t, px)),
                }
            }
        }
        if let Some(&(last, _)) = v.last() {
            v.retain(|x| x.0 > last - SEC_KEEP);
        }
    }
}

fn pair(v: Option<&Value>) -> Option<(i64, f64)> {
    let a = v?.as_array()?;
    Some((a.first()?.as_i64()?, a.get(1)?.as_f64()?))
}

/// 받아 둔 값 → `Cached`(나이는 rx_us). 감시 틱·조립 앞에서 부른다.
pub fn sync_cached<Q: Quotes>(f: &Feed, c: &mut Cached<Q>) {
    for (s, (b, rx)) in &f.books {
        put_book(c, s, b.clone(), *rx);
    }
    for (s, (v, rx)) in &f.vols {
        put_vol(c, s, *v, *rx);
    }
}

/// `subscribe` 를 tokio::spawn 으로 띄울 때(주소를 소유).
pub async fn subscribe_owned(url: String, feed: Arc<Mutex<Feed>>, stop: Arc<AtomicBool>) {
    subscribe(&url, feed, stop).await
}

/// 허브 구독 루프: 붙고 → 구독 보내고 → 받는 대로 `Feed` 에. 끊기면 1·2·4…30초 뒤 다시(stop 이 서면 끝).
/// 물러나기는 **첫 메시지를 받은 뒤**에만 1초로 되돌린다(2026-11-22 · 허브가 구독을 Close(Policy)로 거절하면 전에는 1초마다 다시 붙었다).
pub async fn subscribe(url: &str, feed: Arc<Mutex<Feed>>, stop: Arc<AtomicBool>) {
    use tokio_tungstenite::tungstenite::Message;
    let symbols = ["ethusdc", "ethusdt", "solusdc", "solusdt", "xrpusdc", "xrpusdt", "btcusdc", "btcusdt", "soxlusdt", "sndkusdt"];
    let mut wait = 1.0f64;
    let mut said: HashMap<String, f64> = HashMap::new(); // 사유 → 마지막으로 적은 시각
    while !stop.load(Ordering::SeqCst) {
        let why = match tokio_tungstenite::connect_async(url).await {
            Err(e) => format!("접속 실패: {e}"),
            Ok((mut ws, _)) => {
                let mut why = "끊김(닫힘 사유 없음)".to_string();
                let sub = json!({"subscribe": symbols}).to_string();
                if ws.send(Message::Text(sub.into())).await.is_ok() {
                    {
                        let mut f = feed.lock().unwrap();
                        f.connected = true;
                        f.connects += 1;
                    }
                    while let Some(Ok(m)) = ws.next().await {
                        match m {
                            Message::Text(t) => {
                                wait = 1.0;
                                if let Ok(v) = serde_json::from_str::<Value>(t.as_str()) {
                                    let mut f = feed.lock().unwrap();
                                    f.last_rx = wall();
                                    f.apply(&v);
                                }
                            }
                            Message::Close(fr) => {
                                why = fr.map_or(why, |f| format!("닫힘 {} {}", u16::from(f.code), f.reason));
                            }
                            _ => {}
                        }
                        if stop.load(Ordering::SeqCst) {
                            break;
                        }
                    }
                }
                feed.lock().unwrap().connected = false;
                why
            }
        };
        let now = wall();
        if said.get(&why).is_none_or(|t| now - t >= 60.0) && !stop.load(Ordering::SeqCst) {
            said.insert(why.clone(), now);
            feed.lock().unwrap().logs.push(format!("rtorder 허브 연결 끝 · {why} · {wait:.0}초 뒤 다시"));
        }
        tokio::time::sleep(std::time::Duration::from_secs_f64(wait)).await;
        wait = (wait * 2.0).min(30.0);
    }
}

pub fn wall() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64()
}

fn sec(us: i64) -> f64 {
    us as f64 / 1e6
}

/// bookTicker → `Cached.book`(나이 = 받은 시각 · 2초 넘으면 Cached 가 REST 로 떨어진다).
pub fn put_book<Q: Quotes>(c: &mut Cached<Q>, symbol: &str, book: Value, rx_us: i64) {
    c.book.insert(symbol.into(), (book, sec(rx_us)));
}

/// 1분 변동성 → `Cached.vol`. None(31개 전·끊긴 뒤) 이면 넣지 않는다 -- 받아 둔 «None» 을 답으로 주지 않고 REST 로 떨어지게(낡은 값도 지운다).
pub fn put_vol<Q: Quotes>(c: &mut Cached<Q>, symbol: &str, vol_bpm: Option<f64>, kline_rx_us: i64) {
    match vol_bpm {
        Some(v) => {
            c.vol.insert(symbol.into(), (Some(v), sec(kline_rx_us)));
        }
        None => {
            c.vol.remove(symbol);
        }
    }
}

/// `WatchIo::bars` 자리: 마감 5분봉 하나 = [{time: 끝 초 − 300, close}](감시는 «time+300 ≤ now 인 마지막 봉»만 본다).
/// 끝 초가 낡았으면(그 뒤 봉도 이미 닫혔어야 함 = end + 300 ≤ now) 빈 목록 -- 감시가 옛 봉 종가로 손절하지 않고 hold(1초 닿음 감시는 그대로).
pub fn bars(bar5m: Option<(i64, f64)>, now: f64) -> Vec<Value> {
    match bar5m {
        Some((end, close)) if (end as f64) <= now && now < end as f64 + 300.0 => vec![json!({"time": end - 300, "close": close})],
        _ => vec![],
    }
}

/// `WatchIo::sec_prices` 자리: (초, 가격) 그대로 · 원천 없는 코인 None.
pub fn sec_prices(rows: Option<&[(i64, f64)]>) -> Option<Vec<(f64, f64)>> {
    rows.map(|r| r.iter().map(|(s, p)| (*s as f64, *p)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::Filters;

    #[derive(Default)]
    struct Rest {
        books: u32,
        vols: u32,
    }
    impl Quotes for Rest {
        async fn book(&mut self, _: &str) -> Result<Value, String> {
            self.books += 1;
            Ok(json!({"rest": true}))
        }
        async fn filters(&mut self, _: &str) -> Result<Filters, String> {
            unreachable!()
        }
        async fn vol(&mut self, _: &str) -> Option<f64> {
            self.vols += 1;
            Some(-1.0)
        }
        async fn funding(&mut self, _: &str) -> Option<f64> {
            unreachable!()
        }
        async fn fees(&mut self, _: &str) -> Value {
            unreachable!()
        }
    }

    #[tokio::test]
    async fn ages_none_and_stale_bar() {
        let mut c = Cached::new(Rest::default());
        let b = json!({"symbol": "ETHUSDC", "bidPrice": "2500.00", "askPrice": "2500.10"});
        put_book(&mut c, "ETHUSDC", b.clone(), 1_000_000_000);
        c.now = 1000.0 + 2.0;
        assert_eq!(c.book("ETHUSDC").await.unwrap(), b, "2초 = 아직 신선 · 글자 그대로");
        c.now = 1000.0 + 2.001;
        assert_eq!(c.book("ETHUSDC").await.unwrap(), json!({"rest": true}), "2초 넘음 = REST");
        put_vol(&mut c, "ETHUSDC", Some(7.5), 1_000_000_000);
        c.now = 1060.0;
        assert_eq!(c.vol("ETHUSDC").await, Some(7.5), "60초 = 신선");
        c.now = 1060.5;
        assert_eq!(c.vol("ETHUSDC").await, Some(-1.0), "60초 넘음 = REST");
        put_vol(&mut c, "ETHUSDC", None, 1_060_000_000);
        c.now = 1060.0;
        assert_eq!(c.vol("ETHUSDC").await, Some(-1.0), "31개 전 None = REST(받아 둔 None 을 답으로 주지 않음)");
        assert_eq!((c.inner.books, c.inner.vols), (1, 2));
        assert_eq!(bars(Some((1200, 2500.5)), 1250.0), [json!({"time": 900, "close": 2500.5})]);
        assert!(bars(Some((1200, 2500.5)), 1500.0).is_empty(), "다음 봉도 닫혔어야 할 시각 = 낡은 봉 안 씀");
        assert!(bars(Some((1200, 2500.5)), 1199.0).is_empty(), "아직 안 닫힌 봉");
        assert!(bars(None, 1250.0).is_empty());
        assert_eq!(sec_prices(Some(&[(10, 1.5)])), Some(vec![(10.0, 1.5)]));
        assert_eq!(sec_prices(None), None);
    }
}
