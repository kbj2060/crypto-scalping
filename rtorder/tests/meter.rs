//! 요청 계량(2026-11-21): 가중치 표 · 쉼은 그림자만(실전송은 1,800 을 넘어도 보낸다 -- 손절 청산을 막지 않는다).
//! (계량은 프로세스 전체 값이라 5분 시험과 다른 파일에 둔다)
use rtorder::http::meter;

#[test]
fn weights_and_shadow_only_pause() {
    let w = |p: &str| meter::weight_of(&format!("https://fapi.binance.com{p}"));
    // 그림자 계좌 한 번 읽기 = time + account + positionRisk + 심볼 11 × openOrders = 22 · 원본(userTrades 11 × 5 더) = 77
    let mut shadow = w("/fapi/v1/time") + w("/fapi/v2/account?timestamp=1&signature=x") + w("/fapi/v2/positionRisk?timestamp=1");
    shadow += (0..11).map(|_| w("/fapi/v1/openOrders?symbol=ETHUSDC&timestamp=1")).sum::<u64>();
    assert_eq!(shadow, 22);
    assert_eq!(shadow + (0..11).map(|_| w("/fapi/v1/userTrades?symbol=ETHUSDC&limit=1000")).sum::<u64>(), 77);
    assert_eq!((w("/fapi/v1/commissionRate?symbol=X"), w("/fapi/v1/openOrders?timestamp=1"), w("/fapi/v1/klines?symbol=X&limit=31")), (20, 40, 1));
    let now = 1_800_000_030_000u64;
    meter::seen(Some(1900), false, now);
    assert!(meter::paused(now).is_none(), "실전송은 쉬지 않는다");
    meter::seen(Some(1700), true, now);
    assert!(meter::paused(now).is_none(), "1,800 이하는 쉬지 않는다");
    meter::seen(Some(1801), true, now);
    assert!(meter::paused(now + 1000).is_some_and(|e| e.starts_with("RateLimitPause")));
    assert!(meter::paused(1_800_000_060_000).is_none(), "다음 분 경계부터 다시");
}
