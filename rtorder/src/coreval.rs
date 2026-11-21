//! 실시간 코어(rtcore) 값 → rtorder 자리(2026-11-21 · 좁은 판: rtcore 를 의존성으로 넣지 않고 **값의 모양**만 받는다).
//! 모양 = rtcore/src/quote.rs·proto.rs: `book_ticker` = (REST 꼴 dict {symbol, bidPrice, askPrice} 글자 그대로, 받은 시각 µs) ·
//! `vol_bpm`(마감 1분봉 31개 전엔 None) + `kline_rx_us` · `bar5m` = (끝 초, 종가) · 1초 가격 = (초, 가격).
//! 구독 연결·path 의존성은 다음 조각.

use crate::assemble::{Cached, Quotes};
use serde_json::{json, Value};

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
        async fn margin_type(&mut self, _: &str) -> Option<String> {
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
