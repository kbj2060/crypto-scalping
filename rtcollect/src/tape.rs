//! 집계 규칙 -- `scripts/live_trade_tape_collector_20260916.py` 의 `BUCKETS`·`size_bands`·`TakerOrderAggregator`·`TapeBuffer`
//! 를 그대로 옮겼다(줄 번호는 그 파일 기준). DB 도 네트워크도 모른다.
use std::collections::HashMap;

/// 가격빈(:65-67). 자산마다 틱과 가격대가 달라 한 값을 못 쓴다 -- 대략 «가격의 0.4bp» 로 맞췄다.
/// 2026-11-01 ZEC·SNDK · 2026-11-02 SOXL(틱 0.01 이 이미 1.2bp 라 틱이 곧 칸) · 2026-11-13 SKHYNIX·MU(~1bp).
pub fn bucket(symbol: &str) -> f64 {
    match symbol {
        "ethusdt" => 0.1,
        "btcusdt" => 1.0,
        "solusdt" => 0.01,
        "xrpusdt" => 0.0001,
        "hypeusdt" => 0.001,
        "zecusdt" | "sndkusdt" => 0.05,
        "soxlusdt" => 0.01,
        "skhynixusdt" => 0.1,
        "muusdt" => 0.05,
        _ => 0.01, // collect(): BUCKETS.get(symbol, 0.01)
    }
}

/// (리테일 상한, 고래 하한) 달러(:135-159). 경계를 하드코딩한 이유: 쌓인 행의 뜻이 영원히 고정돼야 한다(:132).
/// ETH $10k/$100k(2026-09-19 실측 · `MS_WHALE_USD_TH` 와 같은 값). 나머지는 2026-09-27 «거래량 중 리테일·고래 비중이
/// ETH 와 같아지는 금액» -- BTC 2026-10-31 · ZEC·SNDK 2026-11-01 잠정 · SOXL 2026-11-02 잠정 · SKHYNIX·MU 2026-11-13.
pub fn size_bands(symbol: &str) -> (f64, f64) {
    match symbol.to_lowercase().replace("-usdt-swap", "usdt").as_str() {
        "solusdt" => (7_500.0, 55_000.0),
        "xrpusdt" => (3_400.0, 23_000.0),
        "btcusdt" => (13_000.0, 160_000.0),
        "zecusdt" => (1_000.0, 6_000.0),
        "sndkusdt" => (1_000.0, 6_700.0),
        "soxlusdt" => (3_900.0, 36_000.0),
        "skhynixusdt" => (700.0, 4_300.0),
        "muusdt" => (700.0, 5_400.0),
        _ => (10_000.0, 100_000.0),
    }
}

/// `@trade` 개별 체결을 테이커 주문 단위로 되묶는다(:162-248).
/// 규칙: 같은 가격 · 같은 방향 · 연속된 체결ID · 직전 체결과 100ms 이내 · 같은 초.
/// 🔴2026-09-19: «같은 ms» 로 묶었다가 고래 물량을 10% 놓쳤다 → 연속ID + ms100 채택(바이낸스 aggTrades 대조 주문수 −0.7~+0.5%).
/// 🔴«같은 초»: 주문이 초 경계를 걸치면 그 초에서 구간 합 > 총량이 된다(251초 수집에서 12행) -- 초에서 자른다.
pub struct TakerOrderAggregator {
    key: Option<(f64, bool)>,
    qty: f64,
    last_ms: i64,
    last_tid: i64,
    first_ms: i64,
}

/// 닫힌 주문: (가격, 수량, 첫 체결 ms, 매도 공격)
pub type Order = (f64, f64, i64, bool);

pub const ORDER_GAP_MS: i64 = 100;

impl Default for TakerOrderAggregator {
    fn default() -> Self {
        Self { key: None, qty: 0.0, last_ms: 0, last_tid: -1, first_ms: 0 }
    }
}

impl TakerOrderAggregator {
    /// 묶음이 닫히면 돌려준다. tid 가 없으면 연속성 조건만 빠진다(:213-232).
    pub fn add(&mut self, price: f64, qty: f64, ts_ms: i64, sell: bool, tid: Option<i64>) -> Option<Order> {
        let same = self.key == Some((price, sell))
            && ts_ms - self.last_ms <= ORDER_GAP_MS
            && ts_ms.div_euclid(1000) == self.last_ms.div_euclid(1000)
            && (tid.is_none() || self.last_tid < 0 || tid == Some(self.last_tid + 1));
        let mut done = None;
        if !same {
            done = self.take();
            self.key = Some((price, sell));
            self.qty = 0.0;
            self.first_ms = ts_ms;
        }
        self.qty += qty;
        self.last_ms = ts_ms;
        self.last_tid = tid.unwrap_or(-1);
        done
    }

    pub fn take(&mut self) -> Option<Order> {
        let Some((price, sell)) = self.key else { return None };
        if self.qty <= 0.0 {
            self.key = None;
            return None;
        }
        // 주문의 시각은 **첫 체결** -- 마지막으로 잡으면 100ms 뒤의 초로 넘어가 드물게 이웃 초에 실린다(:239).
        let out = (price, self.qty, self.first_ms, sell);
        self.key = None;
        self.qty = 0.0;
        self.last_tid = -1;
        Some(out)
    }

    /// 스트림이 끊겼다 -- 열려 있던 묶음을 **버린다**(:245). 끊김 뒤 첫 체결에 이어 붙이면 그 사이가 빠진 주문을 «한 주문»이라 부른다.
    pub fn reset(&mut self) {
        self.key = None;
        self.qty = 0.0;
        self.last_tid = -1;
    }
}

/// 꺼낸 행: (초, 가격빈, 16칸). 칸 2·3·10~15 는 건수(파이썬 int(c[k])).
pub type Cell = [f64; 16];

/// (초, 가격빈) → 16칸(:297-402). 0~1 매수·매도량 · 2~3 체결수 · 4~5 최대 체결 · 6~7 리테일량 · 8~9 고래량 ·
/// 10~11 고래 주문수 · 12~13 리테일 주문수 · 14~15 전체 주문수. ⚠️2~3 은 개별 체결 수, 10~15 는 테이커 주문 수.
pub struct TapeBuffer {
    bucket: f64,
    retail_max: f64,
    whale_min: f64,
    rows: HashMap<(i64, i64), Cell>,
    max_sec: i64,
    closed_before: i64,
    pub dropped_late: u64,
}

impl TapeBuffer {
    pub fn new(bucket: f64, symbol: &str) -> Self {
        let (retail_max, whale_min) = size_bands(symbol);
        Self { bucket, retail_max, whale_min, rows: HashMap::new(), max_sec: 0, closed_before: 0, dropped_late: 0 }
    }

    /// 🔴이미 DB 로 나간 초는 되살리지 않는다(:329-343) -- 늦게 닫힌 주문이 같은 (ts_sec, price_bin) 을 두 줄로 만든다.
    fn cell(&mut self, ts_ms: i64, price: f64) -> Option<&mut Cell> {
        let sec = ts_ms.div_euclid(1000);
        if sec < self.closed_before {
            self.dropped_late += 1;
            return None;
        }
        self.max_sec = self.max_sec.max(sec);
        // 파이썬 round() = 짝수 쪽 반올림(BTC 칸 1.0 에서 90000.5 → 90000).
        let key = (sec, (price / self.bucket).round_ties_even() as i64);
        Some(self.rows.entry(key).or_insert([0.0; 16]))
    }

    /// 개별 체결 하나 -- 총량·건수·최대만(:345-353).
    pub fn add(&mut self, ts_ms: i64, price: f64, qty: f64, sell: bool) {
        let Some(c) = self.cell(ts_ms, price) else { return };
        let i = sell as usize;
        c[i] += qty;
        c[2 + i] += 1.0;
        c[4 + i] = py_max(c[4 + i], qty);
    }

    /// 되묶은 테이커 주문 하나 -- 크기 구간은 여기서만(:355-369).
    pub fn add_order(&mut self, ts_ms: i64, price: f64, qty: f64, sell: bool) {
        let (rm, wm) = (self.retail_max, self.whale_min);
        let Some(c) = self.cell(ts_ms, price) else { return };
        let i = sell as usize;
        c[14 + i] += 1.0;
        bands(c, i, price * qty, qty, rm, wm);
    }

    /// 이미 주문 단위인 것(현물 @aggTrade) -- 모든 칸을 한 번에(:371-389). 🔴4~5 는 «최대 주문».
    pub fn add_agg(&mut self, ts_ms: i64, price: f64, qty: f64, sell: bool, fills: i64) {
        let (rm, wm) = (self.retail_max, self.whale_min);
        let Some(c) = self.cell(ts_ms, price) else { return };
        let i = sell as usize;
        c[i] += qty;
        c[2 + i] += fills as f64;
        c[4 + i] = py_max(c[4 + i], qty);
        c[14 + i] += 1.0;
        bands(c, i, price * qty, qty, rm, wm);
    }

    /// 재연결 전에 진행 중인 초까지 닫는다(Bybit 수집기 :268-269) -- 재연결 뒤 그 초에 온 체결은 버린다(dropped_late).
    pub fn seal(&mut self) {
        self.closed_before = self.closed_before.max(self.max_sec + 1);
    }

    /// 진행 중인 초(max_sec)를 빼고 꺼낸다(:391-402). `everything` = 끝난 구간을 통째로(Bybit 재연결 전 · 원본 REST 복구).
    pub fn take_closed(&mut self, everything: bool) -> Vec<(i64, i64, Cell)> {
        let max_sec = self.max_sec;
        let mut done: Vec<(i64, i64, Cell)> = Vec::new();
        self.rows.retain(|&(sec, b), c| {
            let out = everything || sec < max_sec;
            if out {
                done.push((sec, b, *c));
            }
            !out
        });
        self.closed_before = self.closed_before.max(max_sec);
        done.sort_by_key(|r| (r.0, r.1));
        done
    }
}

fn bands(c: &mut Cell, i: usize, notional: f64, qty: f64, retail_max: f64, whale_min: f64) {
    if notional < retail_max {
        c[6 + i] += qty;
        c[12 + i] += 1.0;
    } else if notional >= whale_min {
        c[8 + i] += qty;
        c[10 + i] += 1.0;
    }
}

/// 파이썬 max(a, b): 같으면 a. (qty > 0 이 먼저 걸러져 NaN 은 없다)
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 원본 selftest(:1003-1076)의 집계 부분을 그대로.
    #[test]
    fn python_selftest_rules() {
        let mut buf = TapeBuffer::new(0.1, "ethusdt");
        buf.add(1_000_000, 2440.04, 1.5, false);
        buf.add(1_000_400, 2440.02, 0.5, true);
        buf.add(1_000_900, 2440.44, 3.0, false);
        buf.add(1_001_000, 2440.04, 9.0, false);
        let rows = buf.take_closed(false);
        assert_eq!(rows.iter().map(|r| (r.0, r.1)).collect::<Vec<_>>(), vec![(1000, 24400), (1000, 24404)]);
        assert_eq!(&rows[0].2[..6], &[1.5, 0.5, 1.0, 1.0, 1.5, 0.5]);

        let mut agg = TakerOrderAggregator::default();
        assert_eq!(agg.add(2500.0, 10.0, 1_000, false, Some(1)), None);
        assert_eq!(agg.add(2500.0, 12.0, 1_003, false, Some(2)), None);
        assert_eq!(agg.add(2500.5, 1.0, 1_004, false, Some(3)), Some((2500.0, 22.0, 1_000, false)));
        assert_eq!(agg.add(2500.5, 1.0, 1_005, true, Some(4)), Some((2500.5, 1.0, 1_004, false)));
        assert_eq!(agg.add(2500.5, 1.0, 1_006, true, Some(99)), Some((2500.5, 1.0, 1_005, true)));
        assert_eq!(agg.add(2500.5, 1.0, 1_200, true, Some(100)), Some((2500.5, 1.0, 1_006, true)));
        assert_eq!(agg.add(2500.5, 1.0, 1_999, true, Some(101)), Some((2500.5, 1.0, 1_200, true)));
        assert_eq!(agg.add(2500.5, 1.0, 2_001, true, Some(102)), Some((2500.5, 1.0, 1_999, true)));
        assert_eq!(agg.take(), Some((2500.5, 1.0, 2_001, true)));
        assert_eq!(agg.take(), None);

        let mut late = TapeBuffer::new(0.1, "ethusdt");
        late.add(5_000_000, 2500.0, 1.0, false);
        late.add(5_002_000, 2500.0, 1.0, false);
        assert_eq!(late.take_closed(false).len(), 1);
        late.add_order(5_000_000, 2500.0, 40.0, false);
        assert!(late.take_closed(false).is_empty());
        assert_eq!(late.dropped_late, 1);
    }
}
