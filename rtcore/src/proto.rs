//! 화면으로 보내는 메시지 형. serde 만 쓴다(tokio 없음) -- 나중에 웹어셈블리 화면 크레이트가 그대로 가져다 쓴다.
//!
//! 선 위 모양(JSON): `{"s":"ethusdt","seq":7,"ex_ms":..,"rx_us":..,"tx_us":..,"type":"trade","id":..,"px":..,"qty":..,"sell":false}`
//! 시각 셋: `ex_ms` 거래소 사건 시각(ms) · `rx_us` 코어 수신(µs, 벽시계) · `tx_us` 코어 송신(µs, 벽시계).
//! `seq` 는 심볼별 1부터 늘어나는 번호다 -- 화면은 빈 번호로 끊김을 안다.
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Stamped {
    pub s: String,
    pub seq: u64,
    pub ex_ms: i64,
    pub rx_us: i64,
    pub tx_us: i64,
    #[serde(flatten)]
    pub ev: Event,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// `sell` = 공격자가 매도(바이낸스 `m` = 매수자가 메이커).
    Trade { id: u64, px: f64, qty: f64, sell: bool },
    Book { u: u64, bid: f64, bid_qty: f64, ask: f64, ask_qty: f64 },
    Mark { mark: f64, index: f64, funding: f64, next_funding_ms: i64 },
    /// 구독 직후 1회. 안의 메시지는 따로 송신된 적이 없어 `tx_us` = 0 이다.
    /// 이후 밀기는 `seq` > 이 스냅숏의 `seq` 인 것만 온다(빠짐·겹침 없음).
    Snapshot { book: Option<Box<Stamped>>, mark: Option<Box<Stamped>>, trades: Vec<Stamped> },
}

/// 화면 → 코어 첫 메시지: `{"subscribe":["ethusdt"]}`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Subscribe {
    pub subscribe: Vec<String>,
}
