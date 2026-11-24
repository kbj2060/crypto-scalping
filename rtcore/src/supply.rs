//! 1초 수급(2026-11-20 파이썬에서 옮김). tokio 비의존. 바이낸스 선물 `@trade` 몫이 먼저, 그 뒤 나머지 원천(아래 «다른 원천»).
//!
//! 원본: dashboard/coin_flow.py `footprint_add`(338~394) · `supply_1s_cell`(323~333) · `collect_footprint`(576~636)
//!       · scripts/live_trade_tape_collector_20260916.py `TakerOrderAggregator`(162~248) · `size_bands`(145~159)
//!       · dashboard/server.py `supply_1s_payload` 의 바이낸스 선물 몫(seconds · partial.bn · now).
//! 칸 = [리테일매수, 리테일매도, 고래매수, 고래매도, 총매수, 총매도, 마지막가격](coin_flow.py 319~321).
//! 일치 정답지: rtcore/tools/gen_supply1s_fixture.py → rtcore/tests/supply1s_parity.rs.
//!
//! 옮긴 규칙(괄호 = 파이썬 위치 · 사고/근거 날짜):
//! - 총량(4·5칸)과 가격(6칸)은 **개별 체결**로, 크기 구간(0~3칸)은 **테이커 주문**으로 센다(coin_flow.py 340~353 · 2026-09-19:
//!   @trade 로 고래를 세면 큰 주문이 작은 체결 수십 건으로 쪼개져 고래가 사라졌다 -- 같은 11.3초 aggTrade 37.4% vs @trade 9.6%).
//! - 주문 되묶기 = 같은 가격 · 같은 방향 · 연속 체결ID · 직전 체결과 100ms 이내 · 같은 초(collector 173~199 · 2026-09-19:
//!   «같은 ms» 규칙은 고래 물량 −10%, 20ms 는 분 단위 −7.96% 꼬리, «연속ID+100ms» 채택. «같은 초»는 구간 합 > 총량 이 되는 초를 없앤다).
//! - 닫힌 주문의 시각 = **첫 체결** 시각(collector 239~241 -- 마지막으로 잡으면 이웃 초에 실린다).
//! - 마지막 묶음은 다음 체결이 와야 닫힌다(collector 201~203). 재연결 때 열린 묶음은 **버린다**(collector 245~248 · coin_flow.py 613).
//! - 고래 = 명목 ≥ 고래 하한, 리테일 = 명목 < 리테일 상한, 사이(중형)는 칸 없음(coin_flow.py 78~85 · 373~380 · collector 108~136).
//!   경계는 코인별 달러 고정(collector 137~154 · 2026-09-27). 표는 아래 SIZE_BANDS_USD -- 파이썬 표와의 일치는 정답지가 시험한다.
//! - 가격 0 체결 건너뛰기는 feed::parse 가 한다(coin_flow.py 599~603 · 2026-09-16 실측 0.3%). 건너뛴 체결도 체결ID 를 먹으므로
//!   그 자리에서 묶음이 끊긴다 -- 파이썬과 같다(되묶기보다 먼저 거른다).
//! - 초 = 거래소 체결 시각 `T` 의 내림(coin_flow.py 325). 수신 시각이 아니다.
//! - 보관: 새 초가 생길 때만 «그 초 − 660» 보다 오래된 초를 지운다(coin_flow.py 87~94 · 323~333 · 2026-09-20 660 결정).
//!   그래서 뒤늦게 온 오래된 체결도 칸을 만든다(다음 새 초 때 지워진다) -- 파이썬과 같게 둔다.
//! - 24시간 봉 창: 체결의 5분봉이 아직 없고 «수신 시각의 봉 − 287봉» 보다 오래되면 **통째로 버린다**(1초 칸도 안 건드린다)
//!   (coin_flow.py 355~365 · KEEP_BARS 288). 봉과 가격 칸은 footprint.rs 가 들고, 같은 add 에서 같이 쌓는다(체결을 한 번만 처리).
//! - 내보내기: 진행 중인 초(가장 새 초)는 seconds 에 넣지 않고 partial 로 따로(server.py supply_1s_payload 도크스트링 · 2026-09-24),
//!   seconds 는 (가장 새 초 − 660, 가장 새 초) 열린 구간, 값은 파이썬 round(x, 3) 과 같은 수(아래 r3).
//!
//! 다른 원천(정답지: tools/gen_supply1s_venues_fixture.py → tests/supply1s_venues_parity.rs):
//! - `liq` 바이낸스 강제청산(server.py collect_force_orders · supply_1s_payload 의 liq_rows): 사건을 받은 순서대로 링(5000건,
//!   deque maxlen)에 넣고, 내보낼 때 초별로 더한다. 창 = (바이낸스 선물 가장 새 초 − 660, 그 초) -- **@trade 의 초로 자른다**
//!   (청산 스트림에는 «진행 중인 초»가 없다). 칸 = [롱수량, 숏수량, 롱USD, 숏USD], usd = 사건마다 qty × 그 사건 가격
//!   (2026-09-22: 수량 × «그 초 마지막 체결가»로 곱하면 캐스케이드에서 벌어진다). 수량 round(x,3) · USD round(x) 정수.
//! - `spot` 바이낸스 현물 @aggTrade · `okx` OKX trades(coin_flow.py collect_spot_flow · collect_okx_flow): 둘 다 **이미 테이커
//!   주문 단위**라 되묶지 않고 한 체결로 총량·가격·크기 구간을 함께 센다(2026-09-23 실측: OKX trades-all 4,810건 vs trades 1,780건,
//!   sz 합 1.0000x). 24시간 봉 창 없음(봉은 풋프린트 몫). 보관·내보내기 규칙은 바이낸스 선물 칸과 같고 창은 각자의 가장 새 초.
//! - `okxOi`: 같은 초는 마지막 값으로 덮고, 넣을 때마다 «그 초 − 660» 앞을 지운다. 창 = 가장 새 OI 초 − 660 **초과**(끝 포함).
//! - `okxLiq`: OKX 체결의 가장 새 초로 자른다(바이낸스 liq 가 @trade 로 자르는 것과 같은 모양).
//! - `bybitLiq`(server.py bybit_liq_events · 2026-10-03): 파이썬은 수집기 SQLite 를 `ORDER BY ts_ms` 로 읽어 더한다 -- 그래서
//!   여기도 사건을 **거래소 시각 순**(같은 ms 는 받은 순)으로 끼워 넣어 더하는 순서까지 같게 한다. 창 = (벽시계 초 − 3 − 660,
//!   벽시계 초 − 3] -- 파이썬의 «수집기 2초 flush 를 기다리는 3초»를 스냅숏은 그대로 따른다(밀기는 기다리지 않는다).
//! - 밀기: 사건마다 «그 사건으로 바뀐 초»만 싣는다. 청산은 파이썬 창의 오른쪽 끝(아직 안 닫힌 초)이어도 바로 민다 --
//!   파이썬은 그 초가 닫혀야 보이지만 값은 같다(화면은 초 단위로 덮어쓴다). 창 왼쪽 밖은 앞으로도 안 보이므로 밀지 않는다.
use crate::footprint::{bar_of, ring_floor, Bars};
use crate::proto::{Footprint, LiqRow, OkxMeta, Partial, SecRow, SpotMeta, Supply1s, Venue};
use std::collections::{BTreeMap, VecDeque};

pub const SUPPLY_1S_SECONDS: i64 = 660;
const ORDER_GAP_MS: i64 = 100;
/// (심볼, 리테일 상한, 고래 하한) USD. 표에 없으면 ETH 경계(collector 135~136).
// ponytail: 파이썬 표를 옮겨 적은 사본 -- 경계를 바꾸면 tools/gen_supply1s_fixture.py 를 다시 돌려 정답지 bands 대조가 잡는다.
const SIZE_BANDS_USD: &[(&str, f64, f64)] = &[
    ("solusdt", 7_500.0, 55_000.0),
    ("xrpusdt", 3_400.0, 23_000.0),
    ("btcusdt", 13_000.0, 160_000.0),
    ("zecusdt", 1_000.0, 6_000.0),
    ("sndkusdt", 1_000.0, 6_700.0),
    ("soxlusdt", 3_900.0, 36_000.0),
    ("skhynixusdt", 700.0, 4_300.0),
    ("muusdt", 700.0, 5_400.0),
];

/// server.py liq_events_by · coin_flow.py okx_liq_events 의 deque(maxlen=5000).
pub const LIQ_RING: usize = 5000;
/// server.py supply_1s_payload `_bnow = int(time.time()) - 3`(수집기가 2초마다 쓴다).
pub(crate) const BYBIT_LAG_S: i64 = 3;
/// OKX 계약 크기(1계약 = 기초자산 몇 개). scripts/live_okx_trade_tape_collector_20260923.py CT_VALS 사본.
/// 🔴안 곱하면 레인이 10배(coin_flow.py 112~114). 표에 없는 종목은 OKX 레인을 켜지 않는다(collect_okx_flow 723~726).
// ponytail: 파이썬 표 사본 + 정답지 대조 -- 파이썬은 기동 때 거래소 REST(ctVal)와도 대조하지만 코어는 REST 를 안 부른다(범위 밖).
pub const OKX_CT_VALS: &[(&str, f64)] = &[
    ("ETH-USDT-SWAP", 0.1),
    ("BTC-USDT-SWAP", 0.01),
    ("SOL-USDT-SWAP", 1.0),
    ("XRP-USDT-SWAP", 100.0),
    ("HYPE-USDT-SWAP", 0.1),
    ("ZEC-USDT-SWAP", 0.01),
    ("SNDK-USDT-SWAP", 1.0),
    ("SOXL-USDT-SWAP", 1.0),
    ("SKHYNIX-USDT-SWAP", 1.0),
    ("MU-USDT-SWAP", 1.0),
];
/// 현물이 없는 종목(TradFi 선물) -- coin_flow.py FLOW_SPECS 의 spot=False. 정답지가 대조한다.
pub const NO_SPOT: &[&str] = &["sndkusdt", "soxlusdt", "skhynixusdt", "muusdt"];
/// Bybit 청산을 받는 종목 -- 수집기(live_bybit_liquidation_collector_20261003.py) SYMBOLS 기본값 사본. 파이썬 bybitLiq 는
/// 이 수집기가 쓴 표만 읽으므로 여기 없는 종목은 파이썬도 늘 빈 목록이다. 정답지가 대조한다.
// ponytail: 서버가 BYBIT_LIQ_SYMBOLS 환경변수로 수집기 목록을 바꿨다면 어긋난다 -- 그때 이 표를 맞춘다.
pub const BYBIT_SYMBOLS: &[&str] = &["ethusdt", "btcusdt", "solusdt", "xrpusdt", "hypeusdt"];

/// `ethusdt` → `ETH-USDT-SWAP`(coin_flow.py _flow_spec okx_inst).
pub fn okx_inst(symbol: &str) -> String {
    format!("{}-USDT-SWAP", symbol.strip_suffix("usdt").unwrap_or(symbol).to_uppercase())
}

pub fn okx_ct_val(inst: &str) -> Option<f64> {
    OKX_CT_VALS.iter().find(|c| c.0 == inst).map(|c| c.1)
}

pub fn size_bands(symbol: &str) -> (f64, f64) {
    let s = symbol.to_lowercase();
    SIZE_BANDS_USD.iter().find(|b| b.0 == s).map_or((10_000.0, 100_000.0), |b| (b.1, b.2))
}

/// 파이썬 round(x, 3) 과 같은 수: 정확한 이진값을 십진 3자리로 반올림(동률은 짝수) → 다시 f64.
/// Rust `{:.3}` 도 정확값 기준·동률 짝수라 같다(0.0625→0.062 · 2.0005→2.001 · 1.0005→1.0 대조, 2026-11-20).
pub(crate) fn r3(x: f64) -> f64 {
    format!("{x:.3}").parse().unwrap_or(x)
}

/// 파이썬 round(x, 1)(같은 방식 -- r3 주석).
fn r1(x: f64) -> f64 {
    format!("{x:.1}").parse().unwrap_or(x)
}

/// 파이썬 round(x) -- 정수, 동률은 짝수.
fn r0(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// server.py supply_1s_payload `_age`: 보낸 시각 − 마지막 사건(초, 0 에서 자름), 사건 없으면 None.
fn age(now_us: i64, ms: i64) -> Option<f64> {
    (ms != 0).then(|| r1(f64::max(0.0, (now_us as f64 / 1000.0 - ms as f64) / 1000.0)))
}

pub(crate) fn newest(m: &BTreeMap<i64, [f64; 7]>) -> i64 {
    m.keys().next_back().copied().unwrap_or(0)
}

/// 그 초의 칸. 오래된 초는 새 초가 생길 때만 버린다(coin_flow.py supply_1s_cell · okx_sec_cell · spot_sec_cell 셋이 같은 몸).
fn sec_cell(m: &mut BTreeMap<i64, [f64; 7]>, s: i64) -> &mut [f64; 7] {
    if !m.contains_key(&s) {
        m.insert(s, [0.0; 7]);
        *m = m.split_off(&(s - SUPPLY_1S_SECONDS));
    }
    m.get_mut(&s).expect("방금 넣었다")
}

pub(crate) fn sec_row(m: &BTreeMap<i64, [f64; 7]>, s: i64) -> Option<SecRow> {
    let c = m.get(&s)?;
    Some((s, r3(c[0]), r3(c[1]), r3(c[2]), r3(c[3]), r3(c[4]), r3(c[5]), r3(c[6])))
}

/// (가장 새 초 − 660, 가장 새 초) -- 진행 중인 초는 partial 로 따로.
fn window_rows(m: &BTreeMap<i64, [f64; 7]>, top: i64) -> Vec<SecRow> {
    m.range(top - SUPPLY_1S_SECONDS + 1..top).filter_map(|(&s, _)| sec_row(m, s)).collect()
}

pub(crate) struct LiqEv {
    pub(crate) ts_ms: i64,
    pub(crate) long: bool,
    pub(crate) qty: f64,
    pub(crate) usd: f64,
}

/// 링 순서대로 초별로 더한다(파이썬 setdefault 루프와 같은 순서 -- 더하는 순서가 달라지면 끝자리가 갈린다).
pub(crate) fn liq_rows(ev: &VecDeque<LiqEv>, keep: impl Fn(i64) -> bool) -> Vec<LiqRow> {
    let mut by: BTreeMap<i64, [f64; 4]> = BTreeMap::new();
    for e in ev {
        let s = e.ts_ms.div_euclid(1000);
        if keep(s) {
            let c = by.entry(s).or_insert([0.0; 4]);
            let i = (!e.long) as usize;
            c[i] += e.qty;
            c[2 + i] += e.usd;
        }
    }
    by.into_iter().map(|(s, c)| (s, r3(c[0]), r3(c[1]), r0(c[2]), r0(c[3]))).collect()
}

pub(crate) fn push_capped(ring: &mut VecDeque<LiqEv>, e: LiqEv) {
    if ring.len() == LIQ_RING {
        ring.pop_front();
    }
    ring.push_back(e);
}

/// 현물·OKX 한 레인: 초 칸 + 화면 meta(server.py okx_meta · spot_meta).
#[derive(Default)]
pub(crate) struct Lane {
    pub(crate) sec: BTreeMap<i64, [f64; 7]>,
    last_ms: i64,
    connected: bool,
    errors: u64,
}

struct Order {
    px: f64,
    sell: bool,
    qty: f64,
    first_ms: i64,
    last_ms: i64,
    last_tid: i64,
}

pub struct Supply {
    pub retail_max: f64,
    pub whale_min: f64,
    pub(crate) sec: BTreeMap<i64, [f64; 7]>,
    pub(crate) fp: Bars,
    /// 지금까지 받은 `@trade` 의 가장 늦은 거래소 시각(ms) -- 코어 스냅숏의 `last_ms`, 다음 공백 메우기의 시작(restore.rs).
    /// 파이썬은 «마지막으로 처리한 체결의 T»(coin_flow.py 625)라 늦게 온 체결 하나가 값을 뒤로 당긴다 -- 그러면 다음 메우기가
    /// 이미 센 초를 테이프로 또 더한다. 코어는 가장 늦은 값을 든다(RESTORE.md «파이썬과 달라지는 지점»).
    pub(crate) last_ms: i64,
    /// 재시작·재연결 복원 상태(restore.rs).
    pub(crate) rs: crate::restore::Pending,
    open: Option<Order>,
    pub(crate) liq: VecDeque<LiqEv>,
    spot: Lane,
    pub(crate) okx: Lane,
    pub(crate) okx_inst: String,
    pub(crate) okx_oi: BTreeMap<i64, f64>,
    /// OKX OI 5분봉 [처음, 끝](coin_flow.py okx_oi_5m · 288봉) · 마지막 밀기 뒤 바뀐 봉.
    pub(crate) okx_oi5m: BTreeMap<i64, [f64; 2]>,
    pub(crate) okx_oi5m_dirty: std::collections::BTreeSet<i64>,
    /// OKX 풋프린트 봉(coin_flow.py okx_bars · 2026-11-21) -- 칸은 바이낸스 봉과 같은 [매수, 매도, 고래매수, 고래매도, 리테일매수, 리테일매도].
    pub(crate) okx_fp: crate::footprint::Bars,
    /// okx_fp["first_bar"](이 봉 «뒤»만 OKX 를 더한다 · 0 = 아직 OKX 체결 없음) · okx_fp["t0_ms"](시작 뒤 첫 OKX 체결).
    pub(crate) okx_first_bar: i64,
    pub(crate) okx_t0_ms: i64,
    /// Micro.okx_bars 로 다시 실을 OKX 봉 · first_bar 가 바뀌었다(통째로).
    pub(crate) okx_sum_dirty: std::collections::BTreeSet<i64>,
    pub(crate) okx_sum_full: bool,
    pub(crate) okx_fr: crate::restore::OkxFpRestore,
    /// OKX·Bybit 5분봉 청산(liquidation-5m-history 몫)이 바뀌었다 -- 다음 밀기에 통째로 싣는다.
    pub(crate) okx_liq5m_dirty: bool,
    /// Bybit 5분봉 청산 [롱USD, 숏USD, 건수](server.py bybit_liq_events 보관 288봉) · 수집 시작 ms(표 전체 첫 청산 · 없으면 첫 실시간 사건).
    pub(crate) bybit5m: BTreeMap<i64, [f64; 3]>,
    pub(crate) bybit_start_ms: Option<i64>,
    okx_last_oi_ms: i64,
    pub(crate) okx_liq: VecDeque<LiqEv>,
    pub(crate) bybit: VecDeque<LiqEv>,
    /// 5분봉 강제청산 원(바이낸스 · server.py liq_5m_add 517~535): 봉 시작 초 → ([롱USD, 숏USD], 건수, 롱 최저가, 숏 최고가).
    pub(crate) liq5m: BTreeMap<i64, ([f64; 2], i64, Option<f64>, Option<f64>)>,
    /// 기록 시작(초) -- 그 봉과 앞은 «모름»(liq_5m_payload). 원본은 시작 시각, DB 에서 되살리면 되살린 첫 사건 시각(liq_events_load 1796~1811).
    pub(crate) liq5m_from: f64,
    /// 마지막 밀기 뒤 바뀐 봉.
    pub(crate) liq5m_dirty: std::collections::BTreeSet<i64>,
    /// 틱 봉(tick.rs · 2026-11-24) -- 흐름 엔진 종목만 켠다.
    pub(crate) tick: crate::tick::Ticks,
}

const LIQ5M_BAR_S: i64 = 300;
const LIQ5M_KEEP_BARS: i64 = 288;

/// server.py liq_5m_add 그대로(봉 = ts//1000//300*300 · 극값 = 롱 청산의 가장 낮은 체결가·숏 청산의 가장 높은 체결가 · 가격 0 은 극값에 안 넣음 ·
/// 이 사건 봉 − 288봉보다 오래된 봉은 지운다).
pub(crate) fn liq5m_add(store: &mut BTreeMap<i64, ([f64; 2], i64, Option<f64>, Option<f64>)>, ts_ms: i64, long: bool, usd: f64, px: f64) -> i64 {
    let bar = ts_ms.div_euclid(1000).div_euclid(LIQ5M_BAR_S) * LIQ5M_BAR_S;
    let a = store.entry(bar).or_insert(([0.0; 2], 0, None, None));
    a.0[(!long) as usize] += usd;
    a.1 += 1;
    if px > 0.0 {
        if long {
            a.2 = Some(a.2.map_or(px, |x: f64| if px < x { px } else { x })); // 파이썬 min(a, px): 같으면 앞의 것
        } else {
            a.3 = Some(a.3.map_or(px, |x: f64| if px > x { px } else { x }));
        }
    }
    let cut = bar - LIQ5M_KEEP_BARS * LIQ5M_BAR_S;
    *store = store.split_off(&cut);
    bar
}

fn r2(x: f64) -> f64 {
    format!("{x:.2}").parse().unwrap_or(x)
}

impl Supply {
    pub fn new(symbol: &str) -> Supply {
        let (retail_max, whale_min) = size_bands(symbol);
        Supply {
            retail_max,
            whale_min,
            sec: BTreeMap::new(),
            fp: Bars::new(symbol),
            last_ms: 0,
            rs: Default::default(),
            open: None,
            liq: VecDeque::new(),
            spot: Lane::default(),
            okx: Lane::default(),
            okx_inst: okx_inst(symbol),
            okx_oi: BTreeMap::new(),
            okx_oi5m: BTreeMap::new(),
            okx_oi5m_dirty: Default::default(),
            okx_fp: crate::footprint::Bars::new(symbol),
            okx_first_bar: 0,
            okx_t0_ms: 0,
            okx_sum_dirty: Default::default(),
            okx_sum_full: false,
            okx_fr: Default::default(),
            okx_liq5m_dirty: false,
            bybit5m: BTreeMap::new(),
            bybit_start_ms: None,
            okx_last_oi_ms: 0,
            okx_liq: VecDeque::new(),
            bybit: VecDeque::new(),
            liq5m: BTreeMap::new(),
            liq5m_from: crate::now_us() as f64 / 1e6, // 복원이 없으면 «지금부터 기록»(원본 liq_events_load 의 now_ms / 1000)
            liq5m_dirty: Default::default(),
            tick: crate::tick::Ticks::new(crate::footprint::spec(symbol).is_some()),
        }
    }

    /// 재연결: 열린 주문 묶음을 버린다(끊긴 사이가 빠진 주문을 «한 주문»이라 부르지 않게).
    pub fn reset_orders(&mut self) {
        self.open = None;
        self.rs.await_first = self.rs.on; // 복원이 켜져 있으면 이 연결의 첫 체결이 공백을 메운다(restore.rs)
    }

    /// 시험용: 원천별 1초 칸(반올림 전).
    pub fn cells_of(&self, venue: Venue) -> &BTreeMap<i64, [f64; 7]> {
        match venue {
            Venue::Spot => &self.spot.sec,
            Venue::Okx => &self.okx.sec,
            _ => &self.sec,
        }
    }

    /// 시험용: OKX OI 칸.
    pub fn okx_oi_cells(&self) -> &BTreeMap<i64, f64> {
        &self.okx_oi
    }

    /// 시험용: 청산 링(받은·되살린 순서) -- (거래소 시각 ms, 롱 청산, 수량, USD).
    pub fn liq_events(&self, venue: Venue) -> Vec<(i64, bool, f64, f64)> {
        let ring = match venue {
            Venue::Okx => &self.okx_liq,
            Venue::Bybit => &self.bybit,
            _ => &self.liq,
        };
        ring.iter().map(|e| (e.ts_ms, e.long, e.qty, e.usd)).collect()
    }

    /// 수신 연결이 붙거나 끊김(feed). 파이썬 connected 와 errors(끊김마다 +1).
    pub fn feed_status(&mut self, venue: Venue, up: bool) {
        let lane = match venue {
            Venue::Spot => &mut self.spot,
            Venue::Okx => &mut self.okx,
            _ => return,
        };
        lane.connected = up;
        lane.errors += (!up) as u64;
    }

    /// 체결 하나(가격 0 은 이미 걸렀다). `now_ms` = 수신 벽시계(24시간 봉 창 판정).
    /// 밀 것이 있으면 «바뀐 닫힌 초 + 진행 중인 초»를 돌려준다.
    pub fn trade(&mut self, px: f64, qty: f64, ts_ms: i64, sell: bool, tid: i64, now_us: i64) -> Option<Supply1s> {
        let now_ms = now_us.div_euclid(1000);
        self.tick.add(tid as u64, ts_ms, px, qty, sell);
        self.last_ms = self.last_ms.max(ts_ms);
        let mut touched: Vec<i64> = Vec::with_capacity(2);
        // 순서도 파이썬 그대로: 총량 먼저, 그다음 이 체결로 닫힌 주문의 구간(coin_flow.py 619~624).
        self.add(px, qty, ts_ms, sell, false, now_ms, &mut touched);
        if let Some((p, q, t, s)) = self.aggregate(px, qty, ts_ms, sell, tid) {
            self.add(p, q, t, s, true, now_ms, &mut touched);
        }
        let newest = *self.sec.keys().next_back()?;
        if touched.is_empty() {
            return None;
        }
        // 진행 중이던 초가 닫혀도 따로 다시 보내지 않는다 -- 마지막 partial 이 이미 그 값이다(닫힐 때 값이 안 바뀐다).
        touched.sort_unstable();
        touched.dedup();
        // 뒤늦은 체결로 바뀐 닫힌 초도 다시 보낸다(덮어쓰기). 파이썬 커서 방식은 이미 받은 초를 다시 안 보내 화면이 옛값에 머문다.
        let mut m = self.base(now_us);
        m.seconds = touched.into_iter().filter(|&s| newest - SUPPLY_1S_SECONDS < s && s < newest).filter_map(|s| sec_row(&self.sec, s)).collect();
        Some(m)
    }

    /// 강제청산 한 건(바이낸스·OKX·Bybit). 받은 시각 `now_us` 는 Bybit 창(벽시계)과 meta 나이에만 쓴다.
    pub fn liq(&mut self, venue: Venue, long: bool, qty: f64, px: f64, ts_ms: i64, now_us: i64) -> Option<Supply1s> {
        let s = ts_ms.div_euclid(1000);
        let e = LiqEv { ts_ms, long, qty, usd: qty * px };
        // 창 왼쪽 끝. 링(5000건)은 창 밖 사건도 넣어야 파이썬 deque 와 같은 때 같은 것이 밀려난다.
        let lo = match venue {
            Venue::Bn => {
                let b = liq5m_add(&mut self.liq5m, ts_ms, long, e.usd, px); // 청산 원도 같은 사건으로(collect_force_orders 1842)
                self.liq5m_dirty.insert(b);
                push_capped(&mut self.liq, e);
                newest(&self.sec) - SUPPLY_1S_SECONDS
            }
            Venue::Okx => {
                push_capped(&mut self.okx_liq, e);
                self.okx_liq5m_dirty = true;
                newest(&self.okx.sec) - SUPPLY_1S_SECONDS
            }
            Venue::Bybit => {
                // 파이썬은 24시간을 들지만 내보내는 창은 벽시계를 따라 앞으로만 간다 -- 창 밖은 버린다.
                let lo = now_us.div_euclid(1_000_000) - BYBIT_LAG_S - SUPPLY_1S_SECONDS;
                while self.bybit.front().is_some_and(|f| f.ts_ms.div_euclid(1000) <= lo) {
                    self.bybit.pop_front();
                }
                // 복원(restore.rs)이 DB 에서 읽은 사건이 재시작 직후 실시간으로 한 번 더 오면 건너뛴다 -- 파이썬은 DB 만 읽어 한 번만 센다.
                let dup = ts_ms <= self.rs.bybit_max_ms && self.bybit.iter().any(|f| f.ts_ms == ts_ms && f.long == e.long && f.qty == e.qty && f.usd == e.usd);
                if dup {
                    return None;
                }
                bybit5m_add(&mut self.bybit5m, ts_ms, long, e.usd, now_us);
                self.bybit_start_ms.get_or_insert(ts_ms);
                self.okx_liq5m_dirty = true;
                if s > lo {
                    let i = self.bybit.partition_point(|f| f.ts_ms <= ts_ms);
                    self.bybit.insert(i, e);
                }
                lo
            }
            Venue::Spot => return None,
        };
        if s <= lo {
            return None;
        }
        let ring = match venue {
            Venue::Bn => &self.liq,
            Venue::Okx => &self.okx_liq,
            _ => &self.bybit,
        };
        let rows = liq_rows(ring, |x| x == s);
        let mut m = self.base(now_us);
        match venue {
            Venue::Bn => m.liq = rows,
            Venue::Okx => m.okx_liq = rows,
            _ => m.bybit_liq = rows,
        }
        Some(m)
    }

    /// 현물·OKX 테이커 주문 하나(가격·수량 0 이하는 feed 가 걸렀다). coin_flow.py collect_spot_flow 887~898 · collect_okx_flow 767~777.
    pub fn flow(&mut self, venue: Venue, px: f64, qty: f64, sell: bool, ts_ms: i64, now_us: i64) -> Option<Supply1s> {
        let (whale_min, retail_max) = (self.whale_min, self.retail_max);
        if venue == Venue::Okx {
            // coin_flow.py 778~796: 봉 칸(라이브) · 첫 체결이 first_bar·t0
            let bar = crate::footprint::bar_of(ts_ms);
            if self.okx_first_bar == 0 {
                (self.okx_first_bar, self.okx_t0_ms) = (bar, ts_ms);
                self.okx_liq5m_dirty = true;
                self.okx_sum_full = true;
            }
            if self.okx_fp.keep(bar, now_us.div_euclid(1000)) {
                self.okx_fp.add(bar, px, qty, sell, false, whale_min, retail_max);
                self.okx_fp.add(bar, px, qty, sell, true, whale_min, retail_max);
                self.okx_sum_dirty.insert(bar);
            }
        }
        let lane = match venue {
            Venue::Spot => &mut self.spot,
            Venue::Okx => &mut self.okx,
            _ => return None,
        };
        let s = ts_ms.div_euclid(1000);
        let c = sec_cell(&mut lane.sec, s);
        let i = sell as usize;
        c[4 + i] += qty;
        c[6] = px;
        let notional = px * qty;
        if notional >= whale_min {
            c[2 + i] += qty;
        } else if notional < retail_max {
            c[i] += qty;
        }
        lane.last_ms = ts_ms;
        let top = newest(&lane.sec);
        let row = if top - SUPPLY_1S_SECONDS < s && s < top { sec_row(&lane.sec, s) } else { None };
        let mut m = self.base(now_us);
        match venue {
            Venue::Spot => m.spot.extend(row),
            _ => m.okx.extend(row),
        }
        Some(m)
    }

    /// OKX 미결제약정(oi > 0 · ts > 0 은 feed 가 걸렀다). coin_flow.py collect_okx_flow 797~814.
    pub fn oi(&mut self, oi: f64, ts_ms: i64, now_us: i64) -> Option<Supply1s> {
        let s = ts_ms.div_euclid(1000);
        self.okx_oi.insert(s, oi);
        self.okx_last_oi_ms = ts_ms;
        if oi > 0.0 && ts_ms > 0 {
            // coin_flow.py 805~812: 봉 [처음, 끝] · 새 봉이 생길 때 288봉 밖을 지운다
            let ob = s.div_euclid(300) * 300;
            match self.okx_oi5m.get_mut(&ob) {
                Some(v) => v[1] = oi,
                None => {
                    self.okx_oi5m.insert(ob, [oi, oi]);
                    self.okx_oi5m.retain(|&x, _| x >= ob - crate::footprint::KEEP_BARS * 300);
                }
            }
            self.okx_oi5m_dirty.insert(ob);
        }
        self.okx_oi = self.okx_oi.split_off(&(s - SUPPLY_1S_SECONDS));
        let top = *self.okx_oi.keys().next_back().expect("방금 넣었다");
        let mut m = self.base(now_us);
        if s > top - SUPPLY_1S_SECONDS {
            m.okx_oi.push((s, oi));
        }
        Some(m)
    }

    /// 구독 직후 1회: 파이썬 `supply_1s_payload(since=0)` 와 같은 행(`symbol`·`oi` 빼고). `now_us` = 보내는 벽시계.
    pub fn snapshot(&self, now_us: i64) -> Supply1s {
        let mut m = self.base(now_us);
        let (bn, sp, ok) = (m.now, m.spot_now, m.okx_now);
        m.seconds = window_rows(&self.sec, bn);
        m.liq = liq_rows(&self.liq, |s| bn - SUPPLY_1S_SECONDS < s && s < bn);
        m.spot = window_rows(&self.spot.sec, sp);
        m.okx = window_rows(&self.okx.sec, ok);
        let oi_top = self.okx_oi.keys().next_back().copied().unwrap_or(0);
        m.okx_oi = self.okx_oi.range(oi_top - SUPPLY_1S_SECONDS + 1..).map(|(&s, &v)| (s, v)).collect();
        m.okx_liq = liq_rows(&self.okx_liq, |s| ok - SUPPLY_1S_SECONDS < s && s < ok);
        let b = now_us.div_euclid(1_000_000) - BYBIT_LAG_S;
        m.bybit_liq = liq_rows(&self.bybit, |s| b - SUPPLY_1S_SECONDS < s && s <= b);
        m
    }

    /// 5분봉 강제청산 원 전부(스냅숏). 기록 시작 봉과 그 앞은 싣지 않는다(liq_5m_payload 의 b <= from_bar).
    pub fn liq5m_all(&self) -> Option<crate::proto::Liq5m> {
        self.fp.spec?;
        let mut m = self.liq5m_rows(self.liq5m.keys().copied());
        self.put_venues(&mut m);
        Some(m)
    }

    /// 마지막 밀기 뒤 바뀐 봉만(없으면 None).
    pub fn liq5m_delta(&mut self) -> Option<crate::proto::Liq5m> {
        self.fp.spec?;
        let keys = std::mem::take(&mut self.liq5m_dirty);
        let mut m = self.liq5m_rows(keys.into_iter());
        if std::mem::take(&mut self.okx_liq5m_dirty) {
            self.put_venues(&mut m);
        }
        (!m.bars.is_empty() || m.okx.is_some()).then_some(m)
    }

    fn liq5m_rows(&self, keys: impl Iterator<Item = i64>) -> crate::proto::Liq5m {
        let from_bar = (self.liq5m_from as i64).div_euclid(LIQ5M_BAR_S) * LIQ5M_BAR_S;
        let bars = keys.filter(|b| *b > from_bar).filter_map(|b| self.liq5m.get(&b).map(|a| (b, r2(a.0[0]), r2(a.0[1]), a.1, a.2, a.3))).collect();
        crate::proto::Liq5m { from_s: self.liq5m_from, bars, okx: None, bybit: None }
    }

    /// 풋프린트 스냅숏(파이썬 /api/footprint since=0 의 봉·칸). 표에 없는 종목은 None.
    pub fn footprint(&self) -> Option<Footprint> {
        self.fp.snapshot(&self.sec, self.retail_max, self.whale_min)
    }

    /// 풋프린트 바뀐 몫(밀기 묶음마다 한 번 -- state.rs flush).
    pub fn footprint_delta(&mut self) -> Option<Footprint> {
        self.fp.take_delta(&self.sec, self.retail_max, self.whale_min)
    }

    /// 시험용: 봉 → 칸(반올림 전).
    pub fn footprint_raw(&self) -> Vec<(i64, Vec<(i64, [f64; 6])>)> {
        self.fp.raw()
    }

    /// 목록 칸이 빈 메시지(밀기는 바뀐 초만 채운다).
    pub(crate) fn base(&self, now_us: i64) -> Supply1s {
        let (bn, sp, ok) = (newest(&self.sec), newest(&self.spot.sec), newest(&self.okx.sec));
        Supply1s {
            now: bn,
            retail_max_usd: self.retail_max,
            whale_min_usd: self.whale_min,
            seconds: vec![],
            partial: Partial { bn: sec_row(&self.sec, bn), okx: sec_row(&self.okx.sec, ok), spot: sec_row(&self.spot.sec, sp) },
            liq: vec![],
            spot: vec![],
            spot_now: sp,
            spot_meta: SpotMeta { connected: self.spot.connected, trade_age: age(now_us, self.spot.last_ms), errors: self.spot.errors },
            okx: vec![],
            okx_now: ok,
            okx_meta: OkxMeta {
                connected: self.okx.connected,
                trade_age: age(now_us, self.okx.last_ms),
                oi_age: age(now_us, self.okx_last_oi_ms),
                inst: self.okx_inst.clone(),
                errors: self.okx.errors,
            },
            okx_liq: vec![],
            okx_oi: vec![],
            bybit_liq: vec![],
        }
    }

    /// TakerOrderAggregator.add -- 묶음이 닫히면 (가격, 수량, 첫 체결 ms, 매도).
    fn aggregate(&mut self, px: f64, qty: f64, ts_ms: i64, sell: bool, tid: i64) -> Option<(f64, f64, i64, bool)> {
        if let Some(o) = &mut self.open {
            // ts_ms − last_ms 는 음수(뒤늦은 체결)여도 «100 이내»다 -- 파이썬 식 그대로.
            if o.px == px && o.sell == sell && ts_ms - o.last_ms <= ORDER_GAP_MS && ts_ms.div_euclid(1000) == o.last_ms.div_euclid(1000) && tid == o.last_tid + 1 {
                o.qty += qty;
                o.last_ms = ts_ms;
                o.last_tid = tid;
                return None;
            }
        }
        let done = self.open.take().filter(|o| o.qty > 0.0).map(|o| (o.px, o.qty, o.first_ms, o.sell));
        self.open = Some(Order { px, sell, qty, first_ms: ts_ms, last_ms: ts_ms, last_tid: tid });
        done
    }

    /// footprint_add: 봉 칸(footprint.rs)과 1초 칸을 같은 호출에서. order=false 면 총량·가격, true 면 크기 구간.
    #[allow(clippy::too_many_arguments)]
    fn add(&mut self, px: f64, qty: f64, ts_ms: i64, sell: bool, order: bool, now_ms: i64, touched: &mut Vec<i64>) {
        let bar = bar_of(ts_ms);
        if !self.fp.keep(bar, now_ms) {
            return; // 창을 벗어난 봉(coin_flow.py 360~361) -- 1초 칸도 안 건드린다
        }
        self.fp.add(bar, px, qty, sell, order, self.whale_min, self.retail_max);
        self.fp.add_1m(ts_ms, px, qty, sell, order, self.whale_min, self.retail_max);
        let s = ts_ms.div_euclid(1000);
        // 가장 새 초가 앞으로 가면 링이 s − 660 밑을 지운다(sec_cell) -- 그 사이에서 시작하는 봉은 지우기 전 링으로 dhl 을 얼린다.
        if let Some(f) = ring_floor(&self.sec) {
            if s - SUPPLY_1S_SECONDS > f {
                self.fp.freeze(&self.sec, f, s - SUPPLY_1S_SECONDS);
            }
        }
        let side = sell as usize;
        if !order && bar < s.max(newest(&self.sec)) - SUPPLY_1S_SECONDS {
            // 링이 첫 초를 이미 지운 봉에 체결이 왔다 -- 얼려 둔 dhl 은 이제 틀리다(footprint.rs 머리 주석).
            // 주문 단위 칸(order)은 dhl 이 안 쓰는 칸(크기 구간)만 바꾸므로 안 지운다 -- 드문 체결에서 닫히는 주문이 늦게 와도 그대로.
            self.fp.forget_dhl(bar);
        }
        let c = sec_cell(&mut self.sec, s);
        if !order {
            c[4 + side] += qty;
            c[6] = px; // 그 초의 «마지막으로 받은» 체결가 = 1초 가격선
        } else {
            let notional = px * qty;
            if notional >= self.whale_min {
                c[2 + side] += qty;
            } else if notional < self.retail_max {
                c[side] += qty;
            }
        }
        touched.push(s);
    }
}

/// Bybit 5분봉에 사건 하나(merge_bybit_liq 의 add) · 288봉 보관(bybit_liq_events since = 지금 − 288봉).
pub(crate) fn bybit5m_add(store: &mut BTreeMap<i64, [f64; 3]>, ts_ms: i64, long: bool, usd: f64, now_us: i64) {
    let b = ts_ms.div_euclid(1000).div_euclid(300) * 300;
    let a = store.entry(b).or_insert([0.0; 3]);
    a[if long { 0 } else { 1 }] += usd;
    a[2] += 1.0;
    let since = now_us.div_euclid(1_000_000) - 288 * 300;
    store.retain(|&k, _| k + 300 > since);
}

impl Supply {
    /// liquidation-5m-history 의 OKX·Bybit 몫(server.py api_liquidation_5m_history 4413~4455 · merge_okx_liq 878~904 의 «메모리» 갈래 · merge_bybit_liq 833~853):
    /// 🔴OKX 의 first_bar «앞» 봉은 원본이 hot 봉 합(okx_liq_hot_bars)으로 채운다 -- 코어는 안 싣는다(응답 서버가 hot SQLite 에서 · REMAINING.md).
    /// OKX = 링(okx_liq_events) 사건을 봉마다 더한 것 중 first_bar «뒤» 봉 · Bybit = 수집 시작이 걸친 봉 «뒤» 봉. 행 = [봉, 롱USD, 숏USD, 건수].
    /// OKX 봉 칸 합(Micro.okx_bars): first_bar «뒤» 봉 중 `keys`(None = 전부). 칸 합은 가격 키 순서(원본은 칸이 생긴 순서 -- 마지막 비트만 갈릴 수 있다).
    pub(crate) fn okx_sums(&self, keys: Option<&std::collections::BTreeSet<i64>>) -> Vec<(i64, [f64; 6])> {
        if self.okx_first_bar == 0 {
            return vec![];
        }
        let sum = |bar: &crate::footprint::Bar| {
            let mut t = [0.0; 6];
            for c in bar.cells.values() {
                for j in 0..6 {
                    t[j] += c[j];
                }
            }
            t
        };
        let from = self.okx_first_bar + 1;
        match keys {
            None => self.okx_fp.bars.range(from..).map(|(&b, bar)| (b, sum(bar))).collect(),
            Some(ks) => ks.range(from..).filter_map(|b| self.okx_fp.bars.get(b).map(|bar| (*b, sum(bar)))).collect(),
        }
    }

    fn put_venues(&self, m: &mut crate::proto::Liq5m) {
        let (of, okx, bf, bybit) = self.venue_liq5m();
        m.okx = Some(crate::proto::VenueLiq { from: (of != 0).then_some(of), bars: okx });
        m.bybit = Some(crate::proto::VenueLiq { from: bf, bars: bybit });
    }

    pub fn venue_liq5m(&self) -> (i64, Vec<crate::proto::VenueLiqRow>, Option<i64>, Vec<crate::proto::VenueLiqRow>) {
        let mut okx: BTreeMap<i64, [f64; 3]> = BTreeMap::new();
        if self.okx_first_bar != 0 {
            for e in &self.okx_liq {
                let b = e.ts_ms.div_euclid(1000).div_euclid(300) * 300;
                if b > self.okx_first_bar {
                    let a = okx.entry(b).or_insert([0.0; 3]);
                    a[if e.long { 0 } else { 1 }] += e.usd;
                    a[2] += 1.0;
                }
            }
        }
        let row = |(&b, a): (&i64, &[f64; 3])| (b, a[0], a[1], a[2] as i64);
        let bfirst = self.bybit_start_ms.map(|s| s.div_euclid(1000).div_euclid(300) * 300);
        let bybit = bfirst.map_or_else(Vec::new, |f| self.bybit5m.range(f + 1..).map(row).collect());
        (self.okx_first_bar, okx.iter().map(row).collect(), bfirst, bybit)
    }
}
