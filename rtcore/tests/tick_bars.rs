//! 틱 봉(tick.rs · 2026-11-24)이 허브 메시지 경로로 받는 쪽에 그대로 닿는지: State::apply → flush 의 TickBars 밀기만으로 지은 상태 ==
//! Snapshot.tick(처음부터 받은 쪽 · 가운데서 스냅숏으로 붙은 쪽 둘). 흐름 엔진 없는 종목은 틱 봉이 없다.
use rtcore::proto::{Event, Stamped, TickBars};
use rtcore::state::State;

fn trade(s: &str, id: u64, ms: i64, px: f64, qty: f64, sell: bool) -> Stamped {
    Stamped { s: s.into(), seq: 0, ex_ms: ms, rx_us: ms * 1000, tx_us: 0, ev: Event::Trade { id, px, qty, sell } }
}

fn snap_tick(st: &State, s: &str) -> Option<TickBars> {
    let Event::Snapshot { tick, .. } = st.snapshot(s, 0).unwrap().ev else { panic!() };
    tick
}

#[test]
fn pushes_rebuild_snapshot_through_state() {
    let sym = "ethusdt";
    let st = State::new(&[sym.into(), "dogeusdt".into()], 16, 1 << 16);
    let mut rx = st.tx.subscribe();
    let mut from_start = snap_tick(&st, sym).expect("흐름 엔진 종목은 빈 틱 봉이라도 싣는다");
    assert!(snap_tick(&st, "dogeusdt").is_none(), "흐름 엔진 없는 종목은 없음");
    let mut mid: Option<TickBars> = None;
    let mut seen_seq = 0;
    let t0 = 1_763_700_000_000i64;
    let mut x = 7u64;
    for i in 1..=26_000u64 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let px = 3000.0 + ((x >> 40) % 400) as f64 * 0.01;
        let id = if i % 997 == 0 { i - 5 } else { i }; // 가끔 순서 밖 id -- 넣지 않아야 한다
        st.apply(trade(sym, id + 1_000_000, t0 + i as i64 * 3, px, ((x >> 20) % 1000) as f64 / 1000.0 + 0.001, x & 1 == 1));
        st.apply(trade("dogeusdt", i, t0 + i as i64, 0.1, 1.0, false));
        if (x >> 33) % 50 == 0 {
            st.flush();
        }
        if i == 13_000 {
            st.flush();
            let m = st.snapshot(sym, 0).unwrap();
            let Event::Snapshot { tick, .. } = m.ev else { panic!() };
            mid = tick;
            seen_seq = m.seq; // 이 번호 뒤 밀기만 받는다(허브 규칙)
        }
    }
    st.flush();
    let mut n = 0;
    while let Ok(m) = rx.try_recv() {
        if let Event::TickBars(d) = m.ev {
            assert_eq!(m.s, sym);
            n += 1;
            from_start.merge(d.clone());
            if m.seq > seen_seq && mid.is_some() {
                mid.as_mut().unwrap().merge(d);
            }
        }
    }
    let fin = snap_tick(&st, sym).unwrap();
    assert!(n > 100, "밀기 {n}건");
    assert_eq!(from_start, fin, "처음부터 받은 쪽");
    assert_eq!(mid.unwrap(), fin, "가운데서 붙은 쪽");
    // 순서 밖 26건을 뺀 수만 셌다: 100틱 k = (26000 − 26) / 100
    let l = &fin.lanes[0];
    assert_eq!((l.n, l.k, l.bars.len()), (100, 259, 240));
    // JSON 왕복(옛 받는 쪽: tick 없는 스냅숏도 읽힌다)
    let line = serde_json::to_string(&st.snapshot(sym, 0).unwrap()).unwrap();
    let back: Stamped = serde_json::from_str(&line).unwrap();
    let Event::Snapshot { tick, .. } = back.ev else { panic!() };
    assert_eq!(tick.unwrap(), fin);
}
