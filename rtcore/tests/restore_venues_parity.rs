//! 나머지 원천 복원 일치 시험(2026-11-21 · restore.rs «나머지 원천»). 정답지 = tools/gen_restore_venues_fixture.py(시험 전에 만든다 -- 커밋 안 함):
//! 원본 수집기가 만든 DB 넷(OKX 테이프 · OKX 맥락 · 바이낸스 맥락 · Bybit 청산) + 재시작 시각 R 에서 원본 coin_flow 복원 함수(supply_1s_restore ·
//! okx_footprint_restore)와 server.py 를 옮겨 적은 것(liq_events_load · Bybit 창)이 낸 값.
//! 코어: R 에 with_restore(강제청산·Bybit 는 그 자리에서) → R 뒤 실시간 OKX 메시지(feed::parse_okx) → 첫 체결·첫 OI 가 일감 → 합친 뒤 대조.
use rtcore::proto::{Event, Venue};
use rtcore::restore::Cfg;
use rtcore::state::State;
use serde_json::Value;
use std::path::PathBuf;
use std::time::Duration;

fn gen_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen/restore_venues")
}

fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= 1e-9 * a.abs().max(b.abs())
}

fn ev4(v: &Value) -> (i64, bool, f64, f64) {
    (v[0].as_i64().unwrap(), v[1].as_bool().unwrap(), v[2].as_f64().unwrap(), v[3].as_f64().unwrap())
}

#[test]
fn venues_restore_matches_python() {
    let p = gen_dir().join("restore_venues_sol.json");
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!("{}: {e} -- 저장소 루트에서 `python3 rtcore/tools/gen_restore_venues_fixture.py` 로 먼저 만든다(커밋 안 함)", p.display())
    }))
    .unwrap();
    for (inst, b) in fx["okx_tape_buckets"].as_object().unwrap() {
        assert_eq!(rtcore::restore::okx_tape_bucket(inst), b.as_f64(), "{inst} OKX 테이프 칸 폭");
    }
    let sym = fx["symbol"].as_str().unwrap().to_string();
    let g = gen_dir();
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("restore_venues_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = Cfg {
        snap_dir: dir.clone(),
        tape: dir.join("없는_테이프.sqlite"),
        tries: 2,
        wait: Duration::ZERO,
        okx_tape: g.join("okx_tape.sqlite"),
        okx_ctx: g.join("okx_ctx.sqlite"),
        bn_ctx: g.join("binance_ctx.sqlite"),
        bybit_liq: g.join("bybit_liq.sqlite"),
    };
    let r_us = (fx["restart_s"].as_f64().unwrap() * 1e6) as i64;
    let (st, _) = State::with_restore(std::slice::from_ref(&sym), 16, 1 << 16, cfg, r_us);
    let mut fields = 0;

    // ① 바이낸스 강제청산: 시작 때 링(시각·쪽·수량·USD 비트)
    let want: Vec<_> = fx["bn_liq"].as_array().unwrap().iter().map(ev4).collect();
    let got = st.liq_events(&sym, Venue::Bn).unwrap();
    assert_eq!(got.len(), want.len(), "강제청산 건수");
    for (g, w) in got.iter().zip(&want) {
        assert_eq!((g.0, g.1, g.2.to_bits(), g.3.to_bits()), (w.0, w.1, w.2.to_bits(), w.3.to_bits()), "강제청산");
        fields += 4;
    }
    // ② Bybit: 링(창 안 사건) · 스냅숏 창 행(파이썬 supply_1s_payload bybitLiq 와 같은 행)
    let want: Vec<_> = fx["bybit_events"].as_array().unwrap().iter().map(ev4).collect();
    let bnow = r_us.div_euclid(1_000_000) - 3;
    let all = st.liq_events(&sym, Venue::Bybit).unwrap();
    let got: Vec<_> = all.iter().filter(|e| e.0.div_euclid(1000) <= bnow).collect(); // 지금 − 3초 뒤 몫은 파이썬 창에 아직 없다(다음 초에 들어온다)
    assert!(all.iter().all(|e| e.0 <= r_us / 1000), "지금보다 늦은 사건을 읽었다");
    assert_eq!(got.len(), want.len(), "Bybit 사건 수");
    for (g, w) in got.iter().zip(&want) {
        assert!(g.0 == w.0 && g.1 == w.1 && close(g.2, w.2) && close(g.3, w.3), "Bybit {g:?} != {w:?}");
        fields += 4;
    }
    let Event::Snapshot { supply, .. } = st.snapshot(&sym, r_us).unwrap().ev else { panic!() };
    let want_rows: Vec<(i64, f64, f64, i64, i64)> = fx["bybit_rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| (r[0].as_i64().unwrap(), r[1].as_f64().unwrap(), r[2].as_f64().unwrap(), r[3].as_i64().unwrap(), r[4].as_i64().unwrap()))
        .collect();
    assert_eq!(supply.bybit_liq, want_rows, "Bybit 창 행");
    fields += want_rows.len() * 5;

    // ③ OKX: 재시작 뒤 실시간 메시지 → 첫 체결·첫 OI 가 일감을 만든다 → 합친다
    let mut rx = st.tx.subscribe();
    let mut live_liq = vec![]; // 재시작 뒤 실시간으로 받은 청산(시각이 첫 체결보다 앞설 수 있다 -- 되살린 몫과 따로 센다)
    for m in fx["live"].as_array().unwrap() {
        let rx_us = (m[0].as_f64().unwrap() * 1e6) as i64;
        for x in rtcore::feed::parse_okx(m[1].as_str().unwrap(), rx_us) {
            if let Event::Liq { qty, .. } = x.ev {
                live_liq.push((x.ex_ms, qty));
            }
            st.apply(x);
        }
    }
    assert!(st.run_restore_okx(&sym).is_some() && st.run_restore_okx_oi(&sym).is_some(), "첫 OKX 체결·OI 가 일감을 만든다");
    assert!(st.run_restore_okx(&sym).is_none() && st.run_restore_okx_oi(&sym).is_none(), "한 번만");
    // OKX 청산 = 풋프린트 봉 복원 회차(원본 okx_footprint_restore -- 정답지는 시계를 R + 20 에 둔 채 회차를 몰아 돌렸다)
    let now_fp = ((fx["restart_s"].as_f64().unwrap() + 20.0) * 1e6) as i64;
    let mut n_fp = 0;
    while let Some((_, done)) = st.run_restore_okx_fp(&sym, now_fp) {
        n_fp += 1;
        if done {
            break;
        }
    }
    assert!(n_fp >= 1, "풋프린트 봉 복원 회차가 돌았다");
    let (t0_ms, t0_oi) = (fx["t0_ms"].as_i64().unwrap(), fx["t0_oi"].as_i64().unwrap());
    let t0 = t0_ms.div_euclid(1000);
    // 1초 칸: t0 앞은 전부 파이썬이 테이프로 되살린 값과 같다(키까지)
    let cells = st.venue_cells(&sym, Venue::Okx).unwrap();
    let want = fx["okx_sec"].as_object().unwrap();
    let before: Vec<_> = cells.iter().filter(|c| c.0 < t0).collect();
    assert_eq!(before.len(), want.len(), "OKX 1초 칸 수(t0 앞)");
    for (s, c) in before {
        let w = want[&s.to_string()].as_array().unwrap();
        for j in 0..7 {
            assert!(close(c[j], w[j].as_f64().unwrap()), "OKX 1초 {s}[{j}]: {} != {}", c[j], w[j]);
            fields += 1;
        }
    }
    // OI: t0 앞 초는 비트까지
    let oi = st.okx_oi_cells(&sym).unwrap();
    let want = fx["okx_oi"].as_object().unwrap();
    let before: Vec<_> = oi.iter().filter(|o| o.0 < t0_oi).collect();
    assert_eq!(before.len(), want.len(), "OKX OI 초 수(t0 앞)");
    for (s, v) in before {
        assert_eq!(v.to_bits(), want[&s.to_string()].as_f64().unwrap().to_bits(), "OKX OI {s}");
        fields += 1;
    }
    // OKX 청산: 링 앞 = 파이썬이 되살린 것 전부(2026-11-21 부터 원본과 같은 «덮은 구간»)
    let want: Vec<_> = fx["okx_liq"].as_array().unwrap().iter().map(ev4).collect();
    // 링 = [되살린 것(앞에 넣음) …, 실시간 …] -- 앞 want.len() 건이 되살린 몫, 그 뒤 t0 앞 시각은 전부 실시간으로 받은 것이어야 한다
    let ring = st.liq_events(&sym, Venue::Okx).unwrap();
    let (got, rest) = ring.split_at(want.len().min(ring.len()));
    assert!(rest.iter().filter(|e| e.0 < t0_ms).all(|e| live_liq.contains(&(e.0, e.2))), "되살린 몫 뒤에 t0 앞 사건이 더 있다(건수 다름)");
    assert_eq!(got.len(), want.len(), "OKX 청산 건수(되살린 것 전부)");
    for (g, w) in got.iter().zip(&want) {
        assert!(g.0 == w.0 && g.1 == w.1 && close(g.2, w.2) && close(g.3, w.3), "OKX 청산 {g:?} != {w:?}");
        fields += 4;
    }
    // 되살린 칸이 밀기로도 나간다(붙어 있던 화면)
    st.flush();
    let (mut okx_rows, mut oi_rows, mut liq_rows) = (0, 0, 0);
    while let Ok(m) = rx.try_recv() {
        if let Event::Supply1s(s) = m.ev {
            okx_rows += s.okx.iter().filter(|r| r.0 < t0).count();
            oi_rows += s.okx_oi.iter().filter(|r| r.0 < t0_oi).count();
            liq_rows += s.okx_liq.iter().filter(|r| r.0 < t0).count();
        }
    }
    assert!(okx_rows > 600 && oi_rows > 100 && liq_rows > 0, "밀기: OKX 1초 {okx_rows} · OI {oi_rows} · 청산 {liq_rows}");
    println!("나머지 원천 복원 일치: 강제청산 {} · Bybit 사건 {} 행 {} · OKX 1초 칸 {} · OI {} · OKX 청산 {} · 필드 {fields} · 밀기 OKX {okx_rows}·OI {oi_rows}·청산 {liq_rows}",
        fx["bn_liq"].as_array().unwrap().len(), fx["bybit_events"].as_array().unwrap().len(), want_rows.len(),
        fx["okx_sec"].as_object().unwrap().len(), fx["okx_oi"].as_object().unwrap().len(), want.len());
    let _ = std::fs::remove_dir_all(&dir);
}
