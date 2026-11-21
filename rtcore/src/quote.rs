//! 주문 경로가 쓰는 받아 둔 시세(2026-11-21) -- rtorder `assemble.rs` 의 «받아 둔 값»(`Cached`)이 가져간다. 값마다 코어가 받은 시각을 단다.
//!
//! - 최우선 호가: `@bookTicker`(선물 · 주문 심볼 USDC 와 화면 심볼 USDT 둘 다 -- basis_for 가 둘의 중간가 비를 쓴다). `book_ticker()` 는
//!   REST `/fapi/v1/ticker/bookTicker` 의 dict 꼴({symbol, bidPrice, askPrice}, 가격은 **거래소 글자 그대로**)을 낸다.
//! - 1분 변동성: `@kline_1m` 의 마감 봉(x=true) 종가. 원본(live_manual_peg_entry realized_vol_now)은 REST klines limit 32 의 마지막(미완결)을 버린
//!   마감 31개로 `realized_vol_bpm` 을 잰다 -- 코어는 마감 봉을 31개 모은 뒤부터 같은 식으로 낸다(모이기 전엔 None → rtorder 는 REST 로 떨어진다).
//!   🔴`@trade` 로 1분봉을 만들지 않는다: 거래소 1분봉이 어떤 체결을 넣는지(`X` = MARKET 아닌 보험기금·ADL 체결)를 코어가 확인할 길이 없고,
//!   재시작 직후 앞 30분이 없다 -- 거래소 1분봉을 그대로 받는 것이 «정확히 같다»를 지키는 길이다.
//!   빈 분: 체결 없는 분은 거래소가 마감 봉을 안 보낼 수 있다 -- 같은 연결 안의 빠진 분은 직전 종가로 채운다(REST 의 빈 분 봉 = 직전 종가 · 거래량 0 이라는
//!   기억에 기댄 가정 -- 실서버 녹화본 대조 전). 연결이 끊기면 빠진 분을 모르니 모은 것을 버리고 다시 31개를 모은다(`reset`).
//! - 마감된 5분봉 종가(손절 감시 · orders.py bracket_tick 의 last_bar): 5분봉 끝에서 마감되는 1분봉의 종가 = 5분봉 종가(같은 마지막 체결).
//!   원본은 REST 5분봉 60초 캐시(load_market_history)라 새 봉이 닫힌 뒤 최대 60초 옛 봉을 본다 -- 코어는 닫힌 즉시(RESTORE 아님 · 파이썬과 다른 곳).
//! - 마지막 체결가: `@trade` 의 마지막(가격 0 거름 -- feed::parse). 원본 `/api/events` tickers 는 REST `/fapi/v1/ticker/price`(= 마지막 체결가)를 1초마다.
use std::collections::VecDeque;

/// 원본 EXIT_VOL_WINDOW 30(수익률 30개) + 1 = 종가 31개.
pub const VOL_CLOSES: usize = 31;
const KEEP: usize = 32;
const MIN_MS: i64 = 60_000;
const BAR5_MS: i64 = 300_000;

pub struct Book {
    pub bid_s: String,
    pub ask_s: String,
    pub bid: f64,
    pub ask: f64,
    pub ex_ms: i64,
    pub rx_us: i64,
}

#[derive(Default)]
pub struct Quote {
    pub book: Option<Book>,
    /// 마감 1분봉 (봉 시작 ms, 종가) -- 같은 연결 안에서 빈틈 없이.
    closes: VecDeque<(i64, f64)>,
    pub vol_bpm: Option<f64>,
    pub kline_rx_us: i64,
    pub bar5m: Option<(i64, f64)>,
    /// (가격, 거래소 시각 ms, 받은 시각 µs)
    pub last: Option<(f64, i64, i64)>,
    /// 마지막 Prices 밀기 뒤 last 가 바뀌었다.
    pub price_dirty: bool,
}

/// live_manual_peg_entry_20260912.py realized_vol_bpm(548~558) 그대로: 양수 종가만 · 10개 미만이면 None · 로그 수익률의 표본 표준편차(n − 1) × 1e4.
pub fn realized_vol_bpm(closes: &[f64]) -> Option<f64> {
    let px: Vec<f64> = closes.iter().copied().filter(|c| *c != 0.0 && *c > 0.0).collect();
    if px.len() < 10 {
        return None;
    }
    let rets: Vec<f64> = (1..px.len()).map(|i| (px[i] / px[i - 1]).ln()).collect();
    let n = rets.len() as f64;
    let mean = rets.iter().sum::<f64>() / n;
    let var = rets.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / (n - 1.0);
    Some(var.sqrt() * 1e4)
}

impl Quote {
    /// 마감 1분봉 하나. 돌려주는 것 = (이 봉까지의 변동성, 이 봉이 5분봉 끝이면 [끝 초, 종가]).
    pub fn kline(&mut self, open_ms: i64, close: f64, rx_us: i64) -> (Option<f64>, Option<(i64, f64)>) {
        if let Some(&(last_open, last_close)) = self.closes.back() {
            if open_ms <= last_open {
                return (self.vol_bpm, None); // 같은 봉 다시(재전송) · 뒤로 간 봉 -- 이미 센 것
            }
            if open_ms - last_open > KEEP as i64 * MIN_MS {
                self.closes.clear(); // 32분 넘게 빈 것은 «체결 없음»으로 보지 않는다 -- 다시 모은다
            } else {
                let mut m = last_open + MIN_MS;
                while m < open_ms {
                    self.closes.push_back((m, last_close)); // 같은 연결 안의 빈 분 = 직전 종가(머리 주석의 가정)
                    if (m + MIN_MS).rem_euclid(BAR5_MS) == 0 {
                        self.bar5m = Some(((m + MIN_MS).div_euclid(1000), last_close)); // 5분봉의 마지막 분이 비었다 -- 5분봉 종가도 직전 체결가
                    }
                    m += MIN_MS;
                }
            }
        }
        self.closes.push_back((open_ms, close));
        while self.closes.len() > KEEP {
            self.closes.pop_front();
        }
        self.kline_rx_us = rx_us;
        self.vol_bpm = if self.closes.len() >= VOL_CLOSES {
            let c: Vec<f64> = self.closes.iter().skip(self.closes.len() - VOL_CLOSES).map(|x| x.1).collect();
            realized_vol_bpm(&c)
        } else {
            None
        };
        let bar5 = ((open_ms + MIN_MS).rem_euclid(BAR5_MS) == 0).then_some(((open_ms + MIN_MS).div_euclid(1000), close));
        if bar5.is_some() {
            self.bar5m = bar5;
        }
        (self.vol_bpm, bar5)
    }

    /// 1분봉 연결이 끊겼다 붙음 -- 빠진 분을 모르니 모은 종가를 버린다(5분봉 종가는 이미 마감된 값이라 둔다).
    pub fn reset(&mut self) {
        self.closes.clear();
        self.vol_bpm = None;
    }

    pub fn closes(&self) -> Vec<(i64, f64)> {
        self.closes.iter().copied().collect()
    }

    /// REST `/fapi/v1/ticker/bookTicker` 꼴 dict(주문 계획 조립이 bidPrice·askPrice 를 float() 로 읽는다) + 받은 시각(µs).
    pub fn book_ticker(&self, symbol: &str) -> Option<(serde_json::Value, i64)> {
        let b = self.book.as_ref()?;
        Some((serde_json::json!({"symbol": symbol.to_uppercase(), "bidPrice": b.bid_s, "askPrice": b.ask_s}), b.rx_us))
    }

    pub fn msg(&self) -> crate::proto::Quote {
        crate::proto::Quote { closes_1m: self.closes(), vol_bpm: self.vol_bpm, kline_rx_us: self.kline_rx_us, bar5m: self.bar5m, last: self.last }
    }
}

#[cfg(test)]
mod t {
    use super::*;

    #[test]
    fn vol_and_bar5m() {
        let mut q = Quote::default();
        let t0 = 1_763_700_000_000; // 5분봉 경계
        let mut last = (None, None);
        for i in 0..40 {
            if i == 7 {
                continue; // 빈 분 -- 직전 종가로 채운다
            }
            last = q.kline(t0 + i * MIN_MS, 100.0 + (i % 5) as f64 * 0.1, 1);
        }
        assert_eq!(q.closes().len(), 32);
        let c: Vec<f64> = (8..40).map(|i| 100.0 + (i % 5) as f64 * 0.1).collect();
        assert_eq!(last.0, realized_vol_bpm(&c[1..]));
        assert_eq!(q.bar5m, Some(((t0 + 40 * MIN_MS) / 1000, 100.4)), "39번째 분(끝 40분)이 5분봉 끝");
        q.reset();
        assert_eq!(q.kline(t0 + 41 * MIN_MS, 1.0, 2).0, None, "끊긴 뒤엔 다시 31개");
        assert_eq!(realized_vol_bpm(&[1.0; 9]), None);
    }
}
