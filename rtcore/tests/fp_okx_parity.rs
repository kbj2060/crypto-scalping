//! 풋프린트 바이낸스+OKX 합산(footprint.rs levels add · from) ↔ server.py api_footprint(옮겨 적음). 정답지: tools/gen_fp_okx_fixture.py.
//! 바이낸스 @trade · OKX trades 를 State 에 넣고 대조 시점마다:
//! ① 스냅숏(= since=0 전량) 봉·칸 ② 밀기(Footprint 바뀐 몫 · full 이면 통째로)를 받아 쌓은 화면 쪽 봉·칸 ③ 그 화면에서 b ≥ since 인 봉(= 이어받기 응답).
use rtcore::proto::{Event, Footprint, Level, Stamped};
use rtcore::state::State;
use serde_json::Value;
use std::collections::BTreeMap;

type Bars = Vec<(i64, Vec<Level>)>;

fn want(v: &Value) -> Bars {
    v.as_array()
        .unwrap()
        .iter()
        .map(|b| {
            let lv = b[1].as_array().unwrap().iter().map(|l| {
                let x: Vec<f64> = l.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
                (x[0], x[1], x[2], x[3], x[4], x[5], x[6])
            });
            (b[0].as_i64().unwrap(), lv.collect())
        })
        .collect()
}

/// 받는 쪽 규칙(proto.rs Footprint 주석): full 이면 갈아 끼움 · 아니면 봉마다 칸을 가격으로 덮어씀 → drop → 144봉 넘으면 오래된 것부터.
fn apply(screen: &mut BTreeMap<i64, BTreeMap<u64, Level>>, fp: Footprint) {
    if fp.full {
        screen.clear();
    }
    for b in fp.bars {
        let e = screen.entry(b.time).or_default();
        for l in b.levels {
            e.insert(l.0.to_bits(), l);
        }
    }
    for d in fp.drop {
        screen.remove(&d);
    }
    while screen.len() > fp.bars_expected {
        let k = *screen.keys().next().unwrap();
        screen.remove(&k);
    }
}

fn flat(screen: &BTreeMap<i64, BTreeMap<u64, Level>>) -> Bars {
    screen
        .iter()
        .map(|(&b, ls)| {
            let mut v: Vec<Level> = ls.values().copied().collect();
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            (b, v)
        })
        .collect()
}

#[test]
fn footprint_okx_merge_matches_python() {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/fp_okx_sol.json")).unwrap();
    let fx: Value = serde_json::from_str(&raw).unwrap();
    let sym = fx["symbol"].as_str().unwrap().to_string();
    let st = State::new(std::slice::from_ref(&sym), 16, 1 << 20);
    let mut rx = st.tx.subscribe();
    let evs = fx["events"].as_array().unwrap();
    let mut screen = BTreeMap::new();
    // 미시 값(Micro.okx_bars)은 호가를 받은 뒤부터 민다 -- 호가 하나
    st.apply(Stamped { s: sym.clone(), seq: 0, ex_ms: 1, rx_us: 1, tx_us: 0, ev: Event::Book { u: 1, bid: 139.0, bid_qty: 1.0, ask: 139.01, ask_qty: 1.0, bid_s: String::new(), ask_s: String::new() } });
    let mut okx_rows: BTreeMap<i64, [f64; 6]> = BTreeMap::new();
    let (mut n_sum, mut n_sum_bits) = (0, 0);
    if let Event::Snapshot { footprint: Some(fp), .. } = st.snapshot(&sym, 0).unwrap().ev {
        apply(&mut screen, fp);
    }
    let (mut ei, mut n_bars, mut n_lv, mut fulls) = (0, 0, 0, 0);
    for cp in fx["checkpoints"].as_array().unwrap() {
        let upto = cp["after"].as_u64().unwrap() as usize;
        while ei < upto {
            let e = &evs[ei];
            let ms = e[1].as_i64().unwrap();
            if e[0] == "bn" {
                let ev = Event::Trade { id: e[2].as_u64().unwrap(), px: e[3].as_f64().unwrap(), qty: e[4].as_f64().unwrap(), sell: e[5].as_bool().unwrap() };
                st.apply(Stamped { s: sym.clone(), seq: 0, ex_ms: ms, rx_us: ms * 1000, tx_us: 0, ev });
            } else {
                for x in rtcore::feed::parse_okx(e[2].as_str().unwrap(), ms * 1000) {
                    st.apply(x);
                }
            }
            ei += 1;
            if ei % 5 == 0 {
                st.flush(); // 밀기 묶음을 몇 사건마다(실제 16ms 묶음처럼 여러 사건이 한 묶음)
            }
        }
        st.flush();
        let last_ms = evs[upto - 1][1].as_i64().unwrap();
        st.situation_tick(last_ms * 1000 + 500_000);
        while let Ok(m) = rx.try_recv() {
            match m.ev {
                Event::Footprint(fp) => {
                    fulls += fp.full as usize;
                    apply(&mut screen, fp);
                }
                Event::Micro(mc) => {
                    if mc.okx_full {
                        okx_rows.clear();
                    }
                    okx_rows.extend(mc.okx_bars);
                }
                _ => {}
            }
        }
        let ws: Vec<(i64, Vec<f64>)> = cp["okx_sums"].as_array().unwrap().iter().map(|r| (r[0].as_i64().unwrap(), r[1].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect())).collect();
        // 받는 쪽은 거르지 않는다 -- 코어가 first_bar «뒤» 봉만 실어야 한다
        let got: Vec<_> = okx_rows.iter().collect();
        assert_eq!(got.len(), ws.len(), "after {upto}: OKX 봉 합 봉 수");
        for ((gb, gv), (wb, wv)) in got.iter().zip(&ws) {
            assert_eq!(**gb, *wb);
            for j in 0..6 {
                assert!((gv[j] - wv[j]).abs() <= 1e-9 * wv[j].abs().max(1.0), "after {upto}: OKX 봉 {wb} 합[{j}] {} != {}", gv[j], wv[j]);
                n_sum += 1;
                n_sum_bits += (gv[j] == wv[j]) as usize;
            }
        }
        let full = want(&cp["full"]);
        let Event::Snapshot { footprint: Some(snap), .. } = st.snapshot(&sym, 0).unwrap().ev else { panic!() };
        let ofr = cp["okx_from"].as_i64().unwrap();
        assert_eq!(snap.okx_from, (ofr != 0).then_some(ofr), "after {upto}: okxFrom");
        let snap_bars: Bars = snap.bars.into_iter().map(|b| (b.time, b.levels)).collect();
        assert_eq!(snap_bars, full, "after {upto}: 스냅숏 = since=0 전량");
        assert_eq!(flat(&screen), full, "after {upto}: 밀기를 쌓은 화면 = since=0 전량");
        let since = cp["since"].as_i64().unwrap();
        let inc: Bars = flat(&screen).into_iter().filter(|b| b.0 >= since).collect();
        assert_eq!(inc, want(&cp["inc"]), "after {upto}: 이어받기(since={since})");
        n_bars += full.len();
        n_lv += full.iter().map(|b| b.1.len()).sum::<usize>();
    }
    println!("풋프린트 OKX 합산 일치: 대조 {} · 봉 {n_bars} · 칸 {n_lv} · 전량 밀기 {fulls} · OKX 봉 합 {n_sum}칸(비트까지 같음 {n_sum_bits})", fx["checkpoints"].as_array().unwrap().len());
    assert_eq!(fulls, 1, "first_bar 가 생길 때 한 번 전량 밀기(라이브 첫 OKX 체결 · 복원 회차도 같은 길 -- okx_fp_restore)");
}
