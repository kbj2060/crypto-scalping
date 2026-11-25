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
    // 가격 칸(2026-11-25): 칸 합 == 봉 수량 · 칸이 실제로 찼다
    for b in fin.lanes.iter().flat_map(|l| l.bars.iter().chain(&l.cur)) {
        assert!(!b.10.is_empty() && (b.10.iter().map(|c| c.1).sum::<f64>() - b.7).abs() < 1e-9);
    }
    // JSON 왕복(옛 받는 쪽: tick 없는 스냅숏도 읽힌다)
    let line = serde_json::to_string(&st.snapshot(sym, 0).unwrap()).unwrap();
    let back: Stamped = serde_json::from_str(&line).unwrap();
    let Event::Snapshot { tick, .. } = back.ev else { panic!() };
    assert_eq!(tick.unwrap(), fin);
}

/// 메시지 크기 실측(2026-11-25 · 보고용): 초당 300체결 · ETH 칸 폭 0.5 · 묶음 = FLUSH_MS(16ms) · 무작위 걸음.
/// `cargo test --release --test tick_bars -- --ignored --nocapture`
#[test]
#[ignore]
fn message_size_with_price_cells() {
    for (name, step) in [("보통(16초 σ≈$2)", 0.05), ("급변(16초 σ≈$10)", 0.25)] {
        let sym = "ethusdt";
        let st = State::new(&[sym.into()], 16, 1 << 20);
        let mut rx = st.tx.subscribe();
        let (mut px, mut x, t0) = (3000.0f64, 9u64, 1_763_700_000_000i64);
        let (mut n, mut bytes, mut bytes_old, mut max) = (0usize, 0usize, 0usize, 0usize);
        let mut ms = 0i64;
        for i in 1..=300 * 120u64 {
            // 120초 · 체결 간격 ≈ 3.33ms
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            px += ((x >> 33) % 2001) as f64 / 1000.0 * step - step;
            let t = t0 + (i as i64 * 1000) / 300;
            st.apply(trade(sym, i, t, (px * 100.0).round() / 100.0, ((x >> 20) % 1000) as f64 / 100.0 + 0.001, x & 1 == 1));
            if t - t0 >= ms + 16 {
                ms = t - t0;
                st.flush();
                while let Ok(m) = rx.try_recv() {
                    if let Event::TickBars(_) = &m.ev {
                        let len = serde_json::to_string(&m).unwrap().len();
                        let mut o = m.clone();
                        if let Event::TickBars(d) = &mut o.ev {
                            d.lanes.iter_mut().flat_map(|l| l.bars.iter_mut().chain(l.cur.as_mut())).for_each(|r| r.10.clear());
                        }
                        let old = serde_json::to_string(&o).unwrap().len();
                        (n, bytes, bytes_old, max) = (n + 1, bytes + len, bytes_old + old, max.max(len));
                    }
                }
            }
        }
        let snap = serde_json::to_string(&st.snapshot(sym, 0).unwrap()).unwrap().len();
        let cells: Vec<usize> = snap_tick(&st, sym).unwrap().lanes.iter().map(|l| l.cur.as_ref().map_or(0, |c| c.10.len())).collect();
        eprintln!(
            "{name}: 묶음 {n}건 · 평균 {} B(칸 없으면 {} B) · 최대 {max} B · 초당 {:.1} KB · 형성 봉 칸 수 {cells:?} · 스냅숏 {} KB",
            bytes / n,
            bytes_old / n,
            bytes as f64 / 120.0 / 1024.0,
            snap / 1024
        );
    }
}

/// 고래 주문(2026-11-25): 같은 가격·방향·연속 id 체결을 되묶어 명목 ≥ 하한만 · 시각 = 첫 체결 · 밀기 == 스냅숏. ETH 하한 $100,000.
#[test]
fn big_orders_follow_order_grouping() {
    let sym = "ethusdt";
    let st = State::new(&[sym.into()], 16, 1 << 16);
    let mut rx = st.tx.subscribe();
    let mut got = snap_tick(&st, sym).unwrap();
    let t0 = chrono_now_ms() / 1000 * 1000;
    let rows: [(u64, i64, f64, f64, bool); 9] = [
        (1, 0, 2000.0, 30.0, false), (2, 10, 2000.0, 25.0, false), // 55 개 = $110,000 → 하나
        (3, 20, 2000.0, 60.0, true),                               // 방향 바뀜 $120,000 → 하나
        (4, 30, 2001.0, 49.0, true),                               // $98,049 → 하한 미만
        (6, 40, 2001.0, 49.0, true),                               // id 끊김 → 앞 것과 안 묶임
        (7, 200, 2001.0, 49.0, true),                              // 100ms 넘음 → 안 묶임
        (8, 1000, 2000.0, 50.0, false),                            // 딱 $100,000 → 하나
        (9, 1010, 1999.0, 1.0, false), (10, 1020, 1999.5, 1.0, false), // 앞 주문을 닫는다
    ];
    for (id, dt, px, q, s) in rows {
        st.apply(trade(sym, id, t0 + dt, px, q, s));
        st.flush();
    }
    while let Ok(m) = rx.try_recv() {
        if let Event::TickBars(d) = m.ev {
            got.merge(d);
        }
    }
    let fin = snap_tick(&st, sym).unwrap();
    assert_eq!(got, fin);
    let b: Vec<_> = fin.big.iter().map(|b| (b.0, b.1 - t0, b.2, b.3, b.4)).collect();
    assert_eq!(b, vec![(0, 0, 2000.0, 55.0, false), (1, 20, 2000.0, 60.0, true), (2, 1000, 2000.0, 50.0, false)]);
}

fn chrono_now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64
}
