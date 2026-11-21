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
    /// 2026-11-21 더함: `bid_s`·`ask_s` = 거래소가 보낸 가격 글자 그대로(`b`·`a`) -- 주문 계획이 REST `/fapi/v1/ticker/bookTicker` 의
    /// `bidPrice`·`askPrice` 와 같은 글자 꼴로 쓰게(quote.rs book_ticker). 옛 받는 쪽은 모르는 칸이라 건너뛴다.
    Book {
        u: u64,
        bid: f64,
        bid_qty: f64,
        ask: f64,
        ask_qty: f64,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        bid_s: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        ask_s: String,
    },
    Mark { mark: f64, index: f64, funding: f64, next_funding_ms: i64 },
    /// 구독 직후 1회. 안의 메시지는 따로 송신된 적이 없어 `tx_us` = 0 이다.
    /// 이후 밀기는 `seq` > 이 스냅숏의 `seq` 인 것만 온다(빠짐·겹침 없음).
    /// `supply` = 1초 수급 전체(파이썬 `/api/supply-1s?since=0` 의 바이낸스 선물 몫과 같은 행).
    /// `footprint` = 풋프린트 전체(파이썬 `/api/footprint?bars=144&since=0` 의 봉·칸). 흐름 엔진이 없는 종목은 null.
    /// 2026-11-21 더함: `quote` = 주문 경로가 쓰는 받아 둔 시세(quote.rs) · `liq5m` = 5분봉 강제청산 원(바이낸스 몫 · 흐름 엔진 종목만).
    Snapshot {
        book: Option<Box<Stamped>>,
        mark: Option<Box<Stamped>>,
        trades: Vec<Stamped>,
        supply: Supply1s,
        footprint: Option<Footprint>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        quote: Option<Quote>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        liq5m: Option<Liq5m>,
        /// 2026-11-21 더함 -- 마지막 30분 상황 근거(situation.rs · 흐름 엔진 종목만 · 아직 안 셌으면 없음).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        situation: Option<Situation>,
        /// 2026-11-21 더함 -- 마지막 미시 값(`okx_oi5m` 은 가진 봉 전부).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        micro: Option<Micro>,
    },
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
    /// 2026-11-21 더함 -- 마감된 1분봉(`@kline_1m` 의 x=true). `ex_ms` = 봉 시작(t). `close_s` = 종가 글자 그대로.
    /// `vol_bpm` = 이 봉까지 마감 1분봉 31개 종가의 실현변동성(bp/√분 · live_manual_peg_entry realized_vol_bpm 과 같은 식) -- 31개가 안 모였으면 null.
    /// `bar5m` = 이 봉이 5분봉의 마지막 분이면 [5분봉 끝 초, 종가](손절 감시가 쓰는 «마감된 5분봉 종가») -- 아니면 null.
    Kline {
        open_ms: i64,
        close: f64,
        close_s: String,
        vol_bpm: Option<f64>,
        bar5m: Option<(i64, f64)>,
    },
    /// 2026-11-21 더함 -- 종목마다 마지막 체결가(구독 안 한 종목도 · 화면 전광판·다른 포지션 칩 · 원본 `/api/events` 의 `tickers`).
    /// `s` = "*"(모든 구독자에게 간다 · hub.rs). 행 = [종목, 가격, 그 체결의 거래소 시각 ms]. 바뀐 종목만 0.1초마다 · 붙을 때 전부 한 번.
    Prices { rows: Vec<(String, f64, i64)> },
    /// 2026-11-21 더함 -- 5분봉 강제청산 원의 바뀐 봉(바이낸스 · server.py liq_5m_add 와 같은 칸). 받는 쪽은 봉마다 덮어쓴다.
    Liq5m(Liq5m),
    /// 2026-11-21 더함 -- 30분 상황 근거(파이썬 `situation_state["now"]` = situation.classify 결과 + computed_at). 1초마다 세고 바뀌었을 때만 민다
    /// (computed_at 만 바뀐 것은 안 민다). `missing` = 코어가 못 채우는 입력(원본은 채운다 · situation.rs 머리 주석).
    Situation(Situation),
    /// 2026-11-21 더함 -- 미시 값(원본 `micro_state["payload"]` 의 코어 입력만으로 되는 칸 + /api/oi-5m 의 OKX 몫). situation 과 같은 1초 박자,
    /// 바뀌었을 때만. `okx_oi5m` 은 바뀐 봉부터 끝까지(받는 쪽은 봉마다 덮어쓴다 -- 봉 Δ 가 앞 봉 끝값에 걸려 있어 뒤 봉도 같이 싣는다).
    Micro(Micro),
}

/// 이름은 원본 payload 키 그대로. 호가를 받기 전이면 mid·spread·qi 가 null.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Micro {
    /// 센 초(벽시계 정수 초 · 원본 `ts`).
    pub ts: i64,
    pub mid: Option<f64>,
    pub spread: Option<f64>,
    /// micro_ref.qi(bidQty, askQty).
    pub qi: Option<f64>,
    /// 지난 60초(진행 초 뺌) 바이낸스 선물 체결량(기초자산).
    pub vol60: f64,
    pub liq60: Liq60,
    /// [봉, ΔOI, 봉 끝 OI] -- server.py api_oi_5m 의 okx_d(반올림 전 · 바이낸스 봉과 더하는 것은 받는 쪽).
    #[serde(default)]
    pub okx_oi5m: Vec<(i64, f64, f64)>,
    /// 2026-11-22 더함 -- OKX 봉 칸 합 [매수, 매도, 고래매수, 고래매도, 리테일매수, 리테일매도](first_bar «뒤» 봉만). 시장 맥락의 재료:
    /// server.py _flow_read_ctx onet(2332~2339 · 거래소 합산 60분·30분·테이커 비율)·okx30(2374~2376). 풋프린트 그림에는 안 들어간다(원본 11-02 «바이낸스만»).
    /// 바뀐 봉만 · `okx_full` 이면 통째로 갈아 끼운다(first_bar 가 바뀜).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub okx_from: Option<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub okx_bars: Vec<(i64, [f64; 6])>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub okx_full: bool,
}

/// 지난 60초 바이낸스 강제청산 USD 합(롱 = 롱 포지션 청산) · 건수.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Liq60 {
    pub long: f64,
    pub short: f64,
    pub n: i64,
}

/// situation.classify 반환과 같은 칸(ok=false 면 `reason` 만). 이름은 파이썬 dict 키 그대로.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Situation {
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dir: Option<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<Evidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regime: Option<Regime>,
    /// 센 시각(초, 벽시계).
    pub computed_at: f64,
    #[serde(default)]
    pub missing: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Evidence {
    pub mid: f64,
    pub move_bp: f64,
    pub range_bp: f64,
    pub dir: i64,
    pub move_ratio: f64,
    pub cvd: i64,
    pub reject: bool,
    pub liq_long: i64,
    pub liq_short: i64,
    pub veto: i64,
    pub obi: Option<f64>,
    pub persist: Option<f64>,
    pub funding: Option<f64>,
    pub basis_bp: Option<f64>,
    pub basis_d_bp: Option<f64>,
    pub basis_thr_bp: Option<f64>,
    pub crowd: i64,
    pub lead: i64,
    pub btc_move_bp: Option<f64>,
    pub btc_rel: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Regime {
    pub ratio: f64,
    pub thr: f64,
    pub margin: f64,
    pub enter: f64,
    pub exit: f64,
}

/// 주문 경로가 쓰는 받아 둔 시세(스냅숏 칸 · quote.rs). 값마다 코어가 받은 시각(rx_us)을 같이 싣는다 -- rtorder 가 나이를 잰다.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Quote {
    /// 마감 1분봉 [봉 시작 ms, 종가] 최근 것부터 최대 32개(오래된 것 먼저).
    pub closes_1m: Vec<(i64, f64)>,
    pub vol_bpm: Option<f64>,
    /// 마지막 마감 1분봉을 받은 시각(µs) -- vol_bpm·bar5m 의 나이.
    pub kline_rx_us: i64,
    /// [5분봉 끝 초, 종가] -- 마지막으로 마감된 5분봉.
    pub bar5m: Option<(i64, f64)>,
    /// [가격, 거래소 시각 ms, 받은 시각 µs] -- 마지막 체결.
    pub last: Option<(f64, i64, i64)>,
}

/// `[봉 시작 초, 롱USD, 숏USD, 건수, 롱 최저가, 숏 최고가]` -- server.py liq_5m_payload 한 봉(USD 는 round(x, 2) · 가격은 그대로, 없으면 null).
pub type Liq5mRow = (i64, f64, f64, i64, Option<f64>, Option<f64>);

/// 2026-11-21 -- `from_s` = 기록 시작(그 초가 든 봉과 그 앞은 «모름» -- liq_5m_payload 의 b <= from_bar 는 안 싣는 규칙).
/// 스냅숏은 가진 봉 전부(이벤트 없는 봉은 행이 없다 = 0 -- 파이썬은 0 을 채워 싣는다), 밀기는 바뀐 봉만.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Liq5m {
    pub from_s: f64,
    pub bars: Vec<Liq5mRow>,
    /// 2026-11-21 더함 -- OKX·Bybit 몫(server.py api_liquidation_5m_history 4413~4455 · merge_okx_liq 의 메모리 갈래 · merge_bybit_liq). 🔴OKX first_bar «앞» 봉(원본 = hot okx_liq_hot_bars)은 안 싣는다 -- 받는 쪽이 hot 에서. 실리면 **통째로 갈아 끼운다**
    /// (없으면 그대로). `okx_from` = OKX first_bar(이 봉 «뒤» 봉만 OKX 를 안다 · 0 = 모름) · `bybit_from` = Bybit 수집 시작이 걸친 봉(그 «뒤»만).
    /// 바이낸스 봉과 더하기·round 는 받는 쪽(원본 응답의 long_usd 합·okx_detail·bybit 칸). HL 은 코어 밖.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub okx: Option<VenueLiq>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bybit: Option<VenueLiq>,
}

/// `[봉 시작 초, 롱USD, 숏USD, 건수]`(반올림 전).
pub type VenueLiqRow = (i64, f64, f64, i64);

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct VenueLiq {
    /// 이 봉 «뒤» 봉만 그 거래소를 안다(None = 모름 → 원본은 그 거래소를 안 더한다).
    pub from: Option<i64>,
    pub bars: Vec<VenueLiqRow>,
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
    /// 구멍 난 봉(2026-11-21 재시작 복원 · restore.rs): 창 안 봉 중 «복원이 못 메운 초 구간»(테이프를 못 읽음 · 테이프가 덜 따라옴 ·
    /// 테이프 수집기 끊김 `gaps` · 지금 메우는 중)에 걸친 봉의 시작 시각. 화면은 이 봉을 온전한 봉으로 보지 않는다(반쪽 칸·델타).
    /// `ready`(창 144봉이 다 참)와 따로다 -- 봉이 다 있어도 구멍이 있을 수 있다. **메시지마다 지금 목록 전체**를 싣는다
    /// (밀기도 같다 · 빈 목록이면 키를 뺀다 = 구멍 없음). 파이썬 /api/footprint 에는 없는 칸 -- 모르는 화면은 그냥 건너뛴다.
    /// 경계 두 초(끊기기 직전 초·붙은 첫 초의 일부 체결)는 파이썬도 빼는 몫이라 구멍으로 치지 않는다(RESTORE.md).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub holes: Vec<i64>,
}

/// 화면 → 코어 첫 메시지: `{"subscribe":["ethusdt"]}`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Subscribe {
    pub subscribe: Vec<String>,
}
