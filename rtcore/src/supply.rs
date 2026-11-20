//! 1초 수급 -- 바이낸스 선물 `@trade` 만으로 쌓는 몫(2026-11-20 파이썬에서 옮김). tokio 비의존.
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
//!   (coin_flow.py 355~365 · KEEP_BARS 288). 봉 목록 자체는 풋프린트 몫이라 여기선 봉 시작 시각 집합만 든다.
//! - 내보내기: 진행 중인 초(가장 새 초)는 seconds 에 넣지 않고 partial 로 따로(server.py supply_1s_payload 도크스트링 · 2026-09-24),
//!   seconds 는 (가장 새 초 − 660, 가장 새 초) 열린 구간, 값은 파이썬 round(x, 3) 과 같은 수(아래 r3).
use crate::proto::{Partial, SecRow, Supply1s};
use std::collections::{BTreeMap, BTreeSet};

pub const SUPPLY_1S_SECONDS: i64 = 660;
const BAR_SECONDS: i64 = 300;
const KEEP_BARS: i64 = 288;
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

pub fn size_bands(symbol: &str) -> (f64, f64) {
    let s = symbol.to_lowercase();
    SIZE_BANDS_USD.iter().find(|b| b.0 == s).map_or((10_000.0, 100_000.0), |b| (b.1, b.2))
}

/// 파이썬 round(x, 3) 과 같은 수: 정확한 이진값을 십진 3자리로 반올림(동률은 짝수) → 다시 f64.
/// Rust `{:.3}` 도 정확값 기준·동률 짝수라 같다(0.0625→0.062 · 2.0005→2.001 · 1.0005→1.0 대조, 2026-11-20).
fn r3(x: f64) -> f64 {
    format!("{x:.3}").parse().unwrap_or(x)
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
    sec: BTreeMap<i64, [f64; 7]>,
    bars: BTreeSet<i64>,
    open: Option<Order>,
}

fn bar_of(ms: i64) -> i64 {
    ms.div_euclid(1000).div_euclid(BAR_SECONDS) * BAR_SECONDS
}

impl Supply {
    pub fn new(symbol: &str) -> Supply {
        let (retail_max, whale_min) = size_bands(symbol);
        Supply { retail_max, whale_min, sec: BTreeMap::new(), bars: BTreeSet::new(), open: None }
    }

    /// 재연결: 열린 주문 묶음을 버린다(끊긴 사이가 빠진 주문을 «한 주문»이라 부르지 않게).
    pub fn reset_orders(&mut self) {
        self.open = None;
    }

    pub fn cells(&self) -> &BTreeMap<i64, [f64; 7]> {
        &self.sec
    }

    /// 체결 하나(가격 0 은 이미 걸렀다). `now_ms` = 수신 벽시계(24시간 봉 창 판정).
    /// 밀 것이 있으면 «바뀐 닫힌 초 + 진행 중인 초»를 돌려준다.
    pub fn trade(&mut self, px: f64, qty: f64, ts_ms: i64, sell: bool, tid: i64, now_ms: i64) -> Option<Supply1s> {
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
        let seconds = touched.into_iter().filter(|&s| newest - SUPPLY_1S_SECONDS < s && s < newest).filter_map(|s| self.row(s)).collect();
        Some(self.msg(newest, seconds))
    }

    /// 구독 직후 1회: 파이썬 `supply_1s_payload(since=0)` 의 바이낸스 선물 몫과 같은 행.
    pub fn snapshot(&self) -> Supply1s {
        let Some(&newest) = self.sec.keys().next_back() else { return self.msg(0, vec![]) };
        let seconds = self.sec.range(newest - SUPPLY_1S_SECONDS + 1..newest).filter_map(|(&s, _)| self.row(s)).collect();
        self.msg(newest, seconds)
    }

    fn msg(&self, now: i64, seconds: Vec<SecRow>) -> Supply1s {
        Supply1s {
            now,
            retail_max_usd: self.retail_max,
            whale_min_usd: self.whale_min,
            seconds,
            partial: Partial { bn: if now > 0 { self.row(now) } else { None } },
        }
    }

    fn row(&self, s: i64) -> Option<SecRow> {
        let c = self.sec.get(&s)?;
        Some((s, r3(c[0]), r3(c[1]), r3(c[2]), r3(c[3]), r3(c[4]), r3(c[5]), r3(c[6])))
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

    /// footprint_add 의 1초 칸 몫. order=false 면 총량·가격, true 면 크기 구간.
    #[allow(clippy::too_many_arguments)]
    fn add(&mut self, px: f64, qty: f64, ts_ms: i64, sell: bool, order: bool, now_ms: i64, touched: &mut Vec<i64>) {
        let bar = bar_of(ts_ms);
        if !self.bars.contains(&bar) {
            let cutoff = bar_of(now_ms) - (KEEP_BARS - 1) * BAR_SECONDS;
            if bar < cutoff {
                return; // 창을 벗어난 봉 -- 넣어봐야 바로 지워진다(coin_flow.py 360~361)
            }
            self.bars.insert(bar);
            self.bars = self.bars.split_off(&cutoff); // 새 봉이 생길 때만 정리(체결마다 돌 일이 아니다)
        }
        let s = ts_ms.div_euclid(1000);
        if !self.sec.contains_key(&s) {
            self.sec.insert(s, [0.0; 7]);
            self.sec = self.sec.split_off(&(s - SUPPLY_1S_SECONDS)); // 오래된 초는 새 초가 생길 때만 버린다
        }
        let c = self.sec.get_mut(&s).expect("방금 넣었다");
        let side = sell as usize;
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
