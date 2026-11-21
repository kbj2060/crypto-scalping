//! 강제청산·마크 수신 상태(원본 server.py fo_state 1801 · mp_state 1774)가 Micro.streams 로 실리는지.
use rtcore::proto::{Event, Stamped, StreamStat, Venue};
use rtcore::state::State;

fn ev(ex_ms: i64, ev: Event) -> Stamped {
    Stamped { s: "ethusdt".into(), seq: 0, ex_ms, rx_us: ex_ms * 1000, tx_us: 0, ev }
}

#[test]
fn streams_follow_python_state() {
    let st = State::new(&["ethusdt".into()], 16, 1 << 10);
    let mut rx = st.tx.subscribe();
    st.stream_status(None, None, Some("접속 실패"), 99.0);
    st.stream_status(None, Some(true), None, 100.5);
    st.stream_status(Some("ethusdt"), Some(true), None, 101.25);
    st.apply(ev(1_000, Event::Book { u: 1, bid: 10.0, bid_qty: 1.0, ask: 11.0, ask_qty: 1.0, bid_s: String::new(), ask_s: String::new() }));
    st.apply(ev(2_000, Event::Mark { mark: 10.5, index: 10.4, funding: 1e-4, next_funding_ms: 0 }));
    st.apply(ev(3_000, Event::Mark { mark: 10.5, index: 10.4, funding: 1e-4, next_funding_ms: 28_800_000 }));
    st.apply(ev(4_000, Event::Liq { venue: Venue::Bn, long: true, qty: 1.0, px: 10.0 }));
    st.apply(ev(4_500, Event::Liq { venue: Venue::Okx, long: true, qty: 1.0, px: 10.0 })); // OKX 는 fo 가 아니다
    st.stream_status(Some("ethusdt"), Some(false), None, 102.0);
    st.situation_tick(5_000_000);
    let mut got = None;
    while let Ok(m) = rx.try_recv() {
        if let Event::Micro(mc) = m.ev {
            got = mc.streams;
        }
    }
    let s = got.expect("Micro 에 streams");
    let fo = StreamStat { connected: true, since: Some(100.5), last_ms: Some(4_000), events: 1, errors: 1, last_error: Some("접속 실패".into()), next_funding_ms: None };
    let mp = StreamStat { connected: false, since: Some(101.25), last_ms: Some(3_000), events: 2, errors: 0, last_error: None, next_funding_ms: Some(28_800_000) };
    assert_eq!((s.fo, s.mp), (fo, mp));
    // 옛 받는 쪽: streams 없는 Micro 도 읽힌다
    let old: rtcore::proto::Micro = serde_json::from_str(r#"{"ts":1,"mid":null,"spread":null,"qi":null,"vol60":0.0,"liq60":{"long":0.0,"short":0.0,"n":0}}"#).unwrap();
    assert!(old.streams.is_none());
}

/// 관계 읽기 재료(Micro.flow): 봉 순합은 칸이 생긴 차례로 더한다(원본 dict 차례) · 첫 밀기 = 통째로 · 그 뒤 = 바뀐 봉만.
#[test]
fn flow_sums_follow_insertion_order() {
    let st = State::new(&["ethusdt".into()], 16, 1 << 12);
    let mut rx = st.tx.subscribe();
    let t0 = 1_763_700_000_000i64 / 300_000 * 300_000; // 봉 시작(ms)
    let trade = |id: u64, ms: i64, px: f64, qty: f64, sell: bool| ev(ms, Event::Trade { id, px, qty, sell });
    st.apply(ev(t0, Event::Book { u: 1, bid: 10.0, bid_qty: 1.0, ask: 11.0, ask_qty: 1.0, bid_s: String::new(), ask_s: String::new() }));
    // 칸 차례: 3000.0 → 2000.0 → 2500.0(가격 키 차례와 다르다) · 1 + 1e16 − 1e16 은 차례에 따라 0 또는 1
    st.apply(trade(1, t0 + 1_000, 3000.0, 1.0, false));
    st.apply(trade(2, t0 + 2_000, 2000.0, 1e16, false));
    st.apply(trade(3, t0 + 3_000, 2500.0, 1e16, true));
    st.situation_tick((t0 + 300_500) * 1000); // 봉이 닫힌 뒤
    let flows: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok()).filter_map(|m| if let Event::Micro(mc) = m.ev { mc.flow } else { None }).collect();
    let f = flows.last().expect("Micro.flow");
    assert!(f.full);
    let b = t0 / 1000;
    let want = ((0.0 + 1.0) + 1e16) - 1e16; // 생긴 차례
    assert_eq!((f.bn.len(), f.bn[0].0, f.bn[0].1[0]), (1, b, want)); // 고래·리테일 칸은 주문 묶음 몫이라 여기선 안 본다
    assert_eq!(f.bn_last, Some((b, [((0.0 + 1.0) + 1e16) + 0.0, ((0.0 + 0.0) + 0.0) + 1e16])));
    // 바뀐 것 없으면 행 없음 · 새 체결로 봉이 바뀌면 그 봉만(full 아님)
    st.situation_tick((t0 + 301_500) * 1000);
    st.apply(trade(4, t0 + 4_000, 3000.0, 2.0, false)); // 닫힌 봉에 늦게 온 체결(최근 두 봉이라 다시 센다)
    st.situation_tick((t0 + 302_500) * 1000);
    let flows: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok()).filter_map(|m| if let Event::Micro(mc) = m.ev { mc.flow } else { None }).collect();
    let f = flows.last().unwrap();
    assert!(!f.full);
    assert_eq!((f.bn.len(), f.bn[0].1[0]), (1, ((0.0 + 3.0) + 1e16) - 1e16));
    let Event::Snapshot { micro, .. } = st.snapshot("ethusdt", 0).unwrap().ev else { panic!() };
    assert!(micro.unwrap().flow.unwrap().full, "스냅숏 = 통째로");
}
