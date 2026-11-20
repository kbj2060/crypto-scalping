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
    /// `footprint` = 풋프린트 전체(파이썬 `/api/footprint?bars=144&since=0` 의 봉·칸). 흐름 엔진이 없는 종목은 null.
    Snapshot { book: Option<Box<Stamped>>, mark: Option<Box<Stamped>>, trades: Vec<Stamped>, supply: Supply1s, footprint: Option<Footprint> },
    /// 1초 수급 바뀐 몫: 묶음(state.rs FLUSH_MS)마다 1건. 목록 칸 = 그 사이 바뀐 초만(받은 쪽은 초 단위로 덮어쓴다),
    /// `partial` = 원천별 진행 중인 초. 빈 목록은 «바뀐 것 없음»이지 «지워라»가 아니다.
    Supply1s(Supply1s),
    /// 풋프린트 바뀐 몫: 묶음마다 1건(footprint.rs take_delta). 받는 규칙은 `Footprint` 주석.
    Footprint(Footprint),
    /// 강제청산 한 건. `long` = **롱 포지션이 청산됨**(바이낸스 S=SELL · OKX posSide=long · Bybit S=Buy -- 원천마다 뜻이 달라 feed 가 맞춘다).
    Liq { venue: Venue, long: bool, qty: f64, px: f64 },
    /// 테이커 주문 하나(바이낸스 현물 @aggTrade · OKX trades -- 둘 다 이미 주문 단위라 되묶지 않는다). qty = 기초자산(OKX 는 ctVal 곱함).
    Flow { venue: Venue, px: f64, qty: f64, sell: bool },
    /// 미결제약정(기초자산 수량). OKX open-interest 만.
    Oi { venue: Venue, oi: f64 },
}

/// `bn` = 바이낸스 선물. 현물은 `spot`.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Venue {
    Bn,
    Spot,
    Okx,
    Bybit,
}

/// `[초, 리테일매수, 리테일매도, 고래매수, 고래매도, 총매수, 총매도, 가격]` -- 파이썬 seconds 한 줄과 같은 배열(값은 round(x, 3)).
pub type SecRow = (i64, f64, f64, f64, f64, f64, f64, f64);

/// `[초, 롱수량, 숏수량, 롱USD, 숏USD]` -- 파이썬 liq·okxLiq·bybitLiq 한 줄(수량 round(x, 3) · USD round(x) 정수).
pub type LiqRow = (i64, f64, f64, i64, i64);
/// `[초, 미결제약정]` -- 파이썬 okxOi 한 줄(반올림 없음).
pub type OiRow = (i64, f64);

/// 이름은 파이썬 supply_1s_payload 와 같다 -- app.js applySupply1s 가 그대로 먹는다(그 함수는 없는 키를 건너뛴다).
/// 파이썬에 있고 여기 없는 것: `symbol`(Stamped.s 가 같은 값) · `oi`(바이낸스 OI 는 수집기 REST 폴링 -- 코어 범위 밖).
/// `spotMeta`·`okxMeta` 는 밀기마다 싣는다: applySupply1s 가 meta 를 통째로 갈아 끼워서, 빼면 화면의 connected 가 사라진다.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Supply1s {
    pub now: i64,
    pub retail_max_usd: f64,
    pub whale_min_usd: f64,
    pub seconds: Vec<SecRow>,
    pub partial: Partial,
    pub liq: Vec<LiqRow>,
    pub spot: Vec<SecRow>,
    pub spot_now: i64,
    pub spot_meta: SpotMeta,
    pub okx: Vec<SecRow>,
    pub okx_now: i64,
    pub okx_meta: OkxMeta,
    pub okx_liq: Vec<LiqRow>,
    pub okx_oi: Vec<OiRow>,
    pub bybit_liq: Vec<LiqRow>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Partial {
    pub bn: Option<SecRow>,
    pub okx: Option<SecRow>,
    pub spot: Option<SecRow>,
}

/// `tradeAge`·`oiAge` = 보낸 시각 − 마지막 사건 시각(초, round(x, 1), 0 에서 자름). 사건이 없으면 null.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SpotMeta {
    pub connected: bool,
    pub trade_age: Option<f64>,
    pub errors: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OkxMeta {
    pub connected: bool,
    pub trade_age: Option<f64>,
    pub oi_age: Option<f64>,
    pub inst: String,
    pub errors: u64,
}

impl Supply1s {
    /// 묶기: `n`(뒤 메시지)을 덮어 합친다 -- 받은 쪽이 둘을 차례로 덮어쓴 것과 같은 결과(app.js applySupply1s: 초마다 덮어쓰기,
    /// partial 도 같은 칸에 넣는다). 진행 중인 초가 바뀌면 옛 partial 을 목록으로 내린다 -- 안 내리면 그 초가 화면에서 빠진다.
    pub fn merge(&mut self, n: Supply1s) {
        for (old, new, rows) in [
            (&mut self.partial.bn, &n.partial.bn, &mut self.seconds),
            (&mut self.partial.okx, &n.partial.okx, &mut self.okx),
            (&mut self.partial.spot, &n.partial.spot, &mut self.spot),
        ] {
            if let Some(o) = old.take() {
                if new.is_none_or(|x| x.0 != o.0) {
                    upsert(rows, vec![o], |r| r.0);
                }
            }
        }
        upsert(&mut self.seconds, n.seconds, |r| r.0);
        upsert(&mut self.liq, n.liq, |r| r.0);
        upsert(&mut self.spot, n.spot, |r| r.0);
        upsert(&mut self.okx, n.okx, |r| r.0);
        upsert(&mut self.okx_liq, n.okx_liq, |r| r.0);
        upsert(&mut self.okx_oi, n.okx_oi, |r| r.0);
        upsert(&mut self.bybit_liq, n.bybit_liq, |r| r.0);
        (self.now, self.retail_max_usd, self.whale_min_usd, self.partial) = (n.now, n.retail_max_usd, n.whale_min_usd, n.partial);
        (self.spot_now, self.spot_meta, self.okx_now, self.okx_meta) = (n.spot_now, n.spot_meta, n.okx_now, n.okx_meta);
    }
}

/// 같은 초면 갈아 끼우고 없으면 붙인다.
// ponytail: 선형 탐색 -- 한 묶음에 바뀐 초는 보통 1~3개다. 묶음 간격을 초 단위로 늘리면 BTreeMap 으로.
fn upsert<T>(rows: &mut Vec<T>, new: Vec<T>, key: fn(&T) -> i64) {
    for x in new {
        match rows.iter().position(|r| key(r) == key(&x)) {
            Some(i) => rows[i] = x,
            None => rows.push(x),
        }
    }
}

/// `[가격, 매수, 매도, 고래매수, 고래매도, 리테일매수, 리테일매도]` -- 파이썬 levels 한 줄(가격 round(키 × 칸 폭, price_dp) · 나머지 round(x, 3)).
pub type Level = (f64, f64, f64, f64, f64, f64, f64);

/// 봉 하나. `dhl` = 봉 안 누적 델타 [최저, 최고, 마감](없으면 키가 없다 -- 파이썬과 같다).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct FpBar {
    pub time: i64,
    pub levels: Vec<Level>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dhl: Option<[f64; 3]>,
}

/// 이름은 파이썬 /api/footprint 응답과 같다(app.js · rtui/src/data.rs 가 읽는 것만 -- footprint.rs 머리 주석의 «뺀 것»).
/// 받는 규칙: `full` 이면 봉 목록을 통째로 갈아 끼운다. 아니면 실린 봉마다 ① 없던 봉은 만들고 ② 칸은 **가격으로 덮어쓰고**
/// (실리지 않은 칸은 그대로) ③ `dhl` 은 통째로 갈아 끼운다(키가 없으면 지운다) -- 그다음 `drop` 의 봉을 지우고,
/// 봉이 `barsExpected` 보다 많으면 오래된 쪽부터 버린다(app.js refreshFootprint 의 창 자르기와 같다).
/// 🔴app.js 의 지금 증분 처리(봉 객체를 통째로 `set`)로 이 메시지를 먹으면 안 실린 칸이 사라진다 -- 칸 단위로 합쳐야 한다.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Footprint {
    pub full: bool,
    pub bucket: f64,
    pub bar_seconds: i64,
    pub bars_expected: usize,
    pub ready: bool,
    pub retail_max_usd: f64,
    pub whale_min_usd: f64,
    pub bars: Vec<FpBar>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drop: Vec<i64>,
}

/// 화면 → 코어 첫 메시지: `{"subscribe":["ethusdt"]}`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Subscribe {
    pub subscribe: Vec<String>,
}
