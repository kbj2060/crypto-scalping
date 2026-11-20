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
    /// `supply` = 1초 수급 전체(파이썬 `/api/supply-1s?since=0` 의 바이낸스 선물 몫과 같은 행).
    Snapshot { book: Option<Box<Stamped>>, mark: Option<Box<Stamped>>, trades: Vec<Stamped>, supply: Supply1s },
    /// 1초 수급 바뀐 몫: 체결마다 1건(supply.rs). `seconds` = 이번에 닫혔거나 고쳐진 초(받은 쪽은 덮어쓴다), `partial.bn` = 진행 중인 초.
    Supply1s(Supply1s),
}

/// `[초, 리테일매수, 리테일매도, 고래매수, 고래매도, 총매수, 총매도, 가격]` -- 파이썬 seconds 한 줄과 같은 배열(값은 round(x, 3)).
pub type SecRow = (i64, f64, f64, f64, f64, f64, f64, f64);

/// 이름은 파이썬 supply_1s_payload 와 같다 -- app.js applySupply1s 가 그대로 먹는다(그 함수는 없는 키를 건너뛴다).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Supply1s {
    pub now: i64,
    pub retail_max_usd: f64,
    pub whale_min_usd: f64,
    pub seconds: Vec<SecRow>,
    pub partial: Partial,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Partial {
    pub bn: Option<SecRow>,
}

/// 화면 → 코어 첫 메시지: `{"subscribe":["ethusdt"]}`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Subscribe {
    pub subscribe: Vec<String>,
}
