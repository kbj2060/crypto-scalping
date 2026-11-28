//! 바이낸스 OI 1초(bnoi.rs): 합성 hot oi_1s → 허브 메시지 행이 원본 규칙(collect_oi_1s + supply_1s_payload oi_floor)과 같은가.
//! 정답 = 아래 `reference`(원본 파이썬 두 함수를 그대로 옮긴 것) · 옛 받는 쪽 호환.
use rtcore::bnoi;
use rtcore::proto::{Event, Supply1s};
use rtcore::state::State;
use rtcore::supply::Supply;
use std::collections::BTreeMap;

const N: i64 = 1_800_000_000; // 지금(초)

/// 원본: 읽은 행 → oi_1s[ts//1000] = v · 행이 있으면 x < now − 660 지움 · payload(since) = 초 > max(since, 꼭대기 − 660).
fn reference(ring: &mut BTreeMap<i64, f64>, rows: &[(i64, f64)], now: i64, since: i64) -> Vec<(i64, f64)> {
    for &(t, v) in rows {
        ring.insert(t.div_euclid(1000), v);
    }
    if !rows.is_empty() {
        ring.retain(|k, _| *k >= now - 660);
    }
    let top = ring.keys().next_back().copied().unwrap_or(0);
    ring.range(since.max(top - 660) + 1..).map(|(k, v)| (*k, *v)).collect()
}

fn db(rows: &[(&str, i64, f64)]) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("rtcore_bnoi_{}_{}.sqlite", std::process::id(), rows.len()));
    let _ = std::fs::remove_file(&p);
    let c = rusqlite::Connection::open(&p).unwrap();
    c.execute_batch("CREATE TABLE oi_1s (symbol TEXT, ts_ms INTEGER, open_interest REAL)").unwrap();
    for (s, t, v) in rows {
        c.execute("INSERT INTO oi_1s VALUES (?, ?, ?)", rusqlite::params![s, t, v]).unwrap();
    }
    p
}

#[test]
fn hot_tail_to_hub_rows() {
    let ms = |s: i64| s * 1000;
    // 창 밖(지금 − 700) · 같은 초 두 번(뒤 값) · 다른 종목 · 창 안 여럿
    let mut rows = vec![("ethusdt", ms(N - 700), 1.0), ("ethusdt", ms(N - 650) + 100, 2.0), ("ethusdt", ms(N - 650) + 900, 3.0), ("btcusdt", ms(N - 10), 9.0)];
    rows.extend((0..40).map(|i| ("ethusdt", ms(N - 400 + i * 7) + 250, 100.0 + i as f64)));
    let p = db(&rows);
    let now_us = N * 1_000_000;
    assert_eq!(bnoi::read(&p, "ETHUSDT", ms(N - 660)).unwrap().len(), rows.len() - 2, "30초 겹침 밖(지금 − 700)·다른 종목 빼고");
    let got_rows = bnoi::read(&p, "ETHUSDT", ms(N - 760)).unwrap(); // 지금 − 700 행도 읽힌다 → 링 지움이 빼야 한다
    assert_eq!(got_rows.len(), rows.len() - 1);

    let mut sup = Supply::new("ethusdt");
    let mut ring = BTreeMap::new();
    let push = sup.bn_oi(&got_rows, now_us).expect("새 초");
    let want = reference(&mut ring, &got_rows, N, 0);
    assert_eq!(push.oi, want, "첫 밀기");
    assert_eq!(push.oi[0], (N - 650, 3.0), "같은 초는 마지막 값 · 지금 − 660 앞은 지움");
    assert_eq!(sup.snapshot(now_us).oi, want, "스냅숏 = since 0");

    // 다음 읽기: 같은 행 다시(겹침) + 새 초 둘 -- 밀기는 새 초만(원본 sinceOi 커서)
    let top = want.last().unwrap().0;
    let next = [(ms(top) + 10, 555.0), (ms(N + 1), 600.0), (ms(N + 2), 601.0)];
    let mut again = got_rows.clone();
    again.extend(next);
    let push2 = sup.bn_oi(&again, now_us + 2_000_000).expect("새 초");
    assert_eq!(push2.oi, reference(&mut ring, &again, N + 2, top), "둘째 밀기");
    assert_eq!(push2.oi, vec![(N + 1, 600.0), (N + 2, 601.0)]);
    assert!(sup.bn_oi(&[], now_us).is_none() && sup.bn_oi(&next, now_us + 2_000_000).is_none(), "새 초 없으면 안 민다");
    assert_eq!(sup.snapshot(now_us).oi, reference(&mut ring.clone(), &[], N + 2, 0), "스냅숏 둘째");

    // State 경로: 스냅숏 supply.oi
    let st = State::new(&["ethusdt".to_string()], 16, 16);
    st.bn_oi("ethusdt", &got_rows, now_us);
    let Event::Snapshot { supply, .. } = st.snapshot("ethusdt", now_us).unwrap().ev else { panic!() };
    assert_eq!(supply.oi, want);
    println!("oi 행 {} 개 일치 · 둘째 밀기 {} 개", want.len(), push2.oi.len());
    let _ = std::fs::remove_file(&p);
}

#[test]
fn wire_compat() {
    let mut s = Supply::new("ethusdt").snapshot(N * 1_000_000);
    let j = serde_json::to_value(&s).unwrap();
    assert!(j.get("oi").is_none(), "비면 직렬화에서 빠진다");
    s.oi = vec![(N, 5.0)];
    let j = serde_json::to_value(&s).unwrap();
    assert_eq!(j["oi"], serde_json::json!([[N, 5.0]]), "파이썬 /api/supply-1s 의 oi 와 같은 모양");
    // 옛 받는 쪽(새 칸 모름) = oi 없는 JSON 도 새 형으로 읽힌다 · 새 JSON 도 옛 형(oi 빼고 같은 칸)으로 읽힌다
    let mut old = j.clone();
    old.as_object_mut().unwrap().remove("oi");
    let back: Supply1s = serde_json::from_value(old).unwrap();
    assert!(back.oi.is_empty());
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    #[allow(dead_code)]
    struct Old { now: i64, okx_oi: Vec<(i64, f64)> }
    let o: Old = serde_json::from_value(j).unwrap();
    assert_eq!(o.now, s.now);
}
