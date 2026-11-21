//! 주문 경로 시세·마지막 체결가·5분봉 청산 원 일치 시험(2026-11-21 · quote.rs · supply.rs liq5m). 정답지 = tools/gen_quote_fixture.py(커밋함):
//! 원본 realized_vol_bpm(import) · orders.py bracket_tick 의 last_bar · REST bookTicker 꼴 · ticker/price = 마지막 체결 · server.py liq_5m_add/payload(옮겨 적음).
use rtcore::proto::{Event, Liq5mRow, Stamped};
use rtcore::restore::Cfg;
use rtcore::state::State;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

const FX: &str = include_str!("fixtures/quote_sol.json");

fn fx() -> Value {
    serde_json::from_str(FX).unwrap()
}

fn apply(st: &State, text: &str, rx_ms: i64) {
    if let Some(m) = rtcore::feed::parse(text, rx_ms * 1000) {
        st.apply(m);
    }
}

#[test]
fn vol_and_bar5m_match_python() {
    let fx = fx();
    let k = &fx["kline"];
    let sym = k["symbol"].as_str().unwrap().to_string();
    assert_eq!(fx["vol_window"].as_u64().unwrap() as usize + 1, rtcore::quote::VOL_CLOSES, "원본 EXIT_VOL_WINDOW + 1");
    let st = State::new(std::slice::from_ref(&sym), 16, 1 << 16);
    let mut rx = st.tx.subscribe();
    let msgs = k["msgs"].as_array().unwrap();
    let checks = k["checks"].as_array().unwrap();
    let (mut ci, mut since, mut same, mut fallback, mut bars) = (0, 0usize, 0, 0, 0);
    for (i, m) in msgs.iter().enumerate() {
        if m["kind"] == "reconnect" {
            st.kline_reset(&sym); // feed 가 @kline_1m 접속마다 부르는 것
            since = 0;
        } else {
            let closed = serde_json::from_str::<Value>(m["text"].as_str().unwrap()).unwrap()["k"]["x"] == true;
            apply(&st, m["text"].as_str().unwrap(), m["rx_ms"].as_i64().unwrap());
            since += closed as usize;
        }
        while ci < checks.len() && checks[ci]["after"].as_u64().unwrap() as usize == i + 1 {
            let c = &checks[ci];
            let want = c["vol_py"].as_f64();
            let (got, _) = st.vol_bpm(&sym).unwrap();
            // 받은 마감 봉 수(빈 분 채움 포함)가 31 이 안 되면 None -- rtorder 는 REST 로 떨어진다(원본 값을 지어내지 않는다)
            let minute = c["minute"].as_i64().unwrap();
            let have = minute - (if minute >= k["reconnect_before"].as_i64().unwrap() { k["reconnect_before"].as_i64().unwrap() } else { 0 }) + 1;
            match got {
                Some(g) => {
                    let w = want.expect("코어만 값이 있다");
                    assert!((g - w).abs() <= 1e-12 * w.abs(), "분 {minute}: vol rust {g} != python {w}");
                    same += 1;
                }
                None => {
                    assert!(have < 31, "분 {minute}: 마감 봉 {have}개({since}건 받음)인데 vol 이 없다");
                    fallback += 1;
                }
            }
            let lb = c["last_bar"].as_array().map(|a| (a[0].as_i64().unwrap(), a[1].as_f64().unwrap()));
            assert_eq!(st.quote(&sym).unwrap().bar5m, lb, "분 {minute}: 마감된 5분봉 종가");
            bars += lb.is_some() as usize;
            ci += 1;
        }
    }
    assert_eq!(ci, checks.len());
    // 밀기(Kline)에도 같은 값이 실린다
    let mut pushed = 0;
    while let Ok(m) = rx.try_recv() {
        if let Event::Kline { vol_bpm, .. } = m.ev {
            pushed += 1;
            let _ = vol_bpm;
        }
    }
    let n_closed = msgs.iter().filter(|m| m["text"].as_str().is_some_and(|t| serde_json::from_str::<Value>(t).unwrap()["k"]["x"] == true)).count();
    assert_eq!(pushed, n_closed, "마감 봉마다 Kline 하나(형성 중 갱신은 안 민다)");
    assert!(same > 20 && fallback > 0 && bars > 10, "대조가 얇다: 같음 {same} · 없음 {fallback} · 5분봉 {bars}");
    println!("1분 변동성: 같음 {same} · 31개 전이라 없음 {fallback} · 5분봉 종가 대조 {bars} · 빈 분 {:?}", k["empty"]);
}

#[test]
fn book_ticker_matches_rest_form() {
    let fx = fx();
    let st = State::new(&["solusdc".to_string()], 16, 1 << 16);
    for b in fx["books"].as_array().unwrap() {
        let rx = b["rx_ms"].as_i64().unwrap();
        apply(&st, b["text"].as_str().unwrap(), rx);
        let (got, at) = st.book_ticker("solusdc").unwrap();
        assert_eq!(got, b["want"], "REST bookTicker 꼴(글자 그대로)");
        assert_eq!(at, rx * 1000, "받은 시각");
    }
    println!("호가 {}건 REST 꼴 같음", fx["books"].as_array().unwrap().len());
}

#[test]
fn last_prices_match_ticker() {
    let fx = fx();
    let st = State::new(&["solusdt".to_string(), "ethusdt".to_string()], 16, 1 << 16);
    let mut rx = st.tx.subscribe();
    let mut view: BTreeMap<String, f64> = BTreeMap::new();
    let mut n = 0;
    for t in fx["trades"].as_array().unwrap() {
        apply(&st, t["text"].as_str().unwrap(), t["rx_ms"].as_i64().unwrap());
        if let Some(w) = t.get("want") {
            st.flush_prices();
            while let Ok(m) = rx.try_recv() {
                let m: Stamped = m;
                if let Event::Prices { rows } = m.ev {
                    assert_eq!(m.s, "*", "모든 종목 메시지");
                    for (s, px, _) in rows {
                        view.insert(s, px);
                    }
                }
            }
            for (s, px) in w.as_object().unwrap() {
                assert_eq!(view.get(s).copied(), px.as_f64(), "{s} 마지막 체결가(밀기)");
                n += 1;
            }
            let Event::Prices { rows } = st.prices_snapshot().ev else { panic!() };
            let snap: BTreeMap<String, f64> = rows.into_iter().map(|r| (r.0, r.1)).collect();
            assert_eq!(snap, view, "붙을 때 받는 전부 == 밀기로 지은 것");
        }
    }
    println!("마지막 체결가 대조 {n}");
}

/// 1초 마지막 체결가(손절 감시 · server.py sec_prices) -- 1초 수급 정답지(원본 collect_footprint 의 sec)로.
#[test]
fn sec_prices_match_python() {
    let fx: Value = serde_json::from_str(include_str!("fixtures/supply1s_eth.json")).unwrap();
    let sym = fx["symbol"].as_str().unwrap().to_string();
    let st = State::new(std::slice::from_ref(&sym), 16, 1 << 16);
    let mut conn = 0;
    for r in fx["rows"].as_array().unwrap() {
        let r = r.as_array().unwrap();
        if r[0].as_i64().unwrap() != conn {
            st.reset_orders(&sym);
            conn = r[0].as_i64().unwrap();
        }
        if r[1].as_i64() == Some(1) {
            continue;
        }
        let x = if r[4].as_str() == Some("0") { "NA" } else { "MARKET" };
        let text = serde_json::json!({"e": "trade", "E": r[7], "T": r[2], "s": sym.to_uppercase(), "t": r[3], "p": r[4], "q": r[5], "X": x, "m": r[6]}).to_string();
        apply(&st, &text, r[7].as_i64().unwrap());
    }
    // 파이썬 [(t, float(c[6])) for t, c in sec.items() if c[6]] -- 순서는 dict(처음 만든 순)라 정렬해 비교(쓰는 쪽은 min·max·any)
    let want: Vec<(i64, f64)> = fx["final_sec"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(t, c)| (t.parse::<i64>().unwrap(), c[6].as_f64().unwrap()))
        .filter(|x| x.1 != 0.0)
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .collect();
    let got = st.sec_prices(&sym).unwrap();
    assert_eq!(got.iter().map(|x| (x.0, x.1.to_bits())).collect::<Vec<_>>(), want.iter().map(|x| (x.0, x.1.to_bits())).collect::<Vec<_>>());
    println!("1초 마지막 체결가 {}초 비트 일치", got.len());
}

#[test]
fn liq5m_matches_python() {
    let fx = fx();
    let l = &fx["liq5m"];
    let sym = l["symbol"].as_str().unwrap().to_string();
    let nowhere = PathBuf::from("/rtcore_없는_경로/x.sqlite");
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("quote_liq5m_{}", std::process::id()));
    let cfg = Cfg {
        snap_dir: dir.clone(),
        tape: nowhere.clone(),
        tries: 1,
        wait: Duration::ZERO,
        okx_tape: nowhere.clone(),
        okx_ctx: nowhere.clone(),
        bn_ctx: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/quote_bn_ctx.sqlite"),
        bybit_liq: nowhere,
    };
    let (st, _) = State::with_restore(std::slice::from_ref(&sym), 16, 1 << 16, cfg, (l["restart_s"].as_f64().unwrap() * 1e6) as i64);
    let mut rx = st.tx.subscribe();
    let snap0 = |st: &State| {
        let Event::Snapshot { liq5m, .. } = st.snapshot(&sym, 0).unwrap().ev else { panic!() };
        liq5m.unwrap()
    };
    let mut view: BTreeMap<i64, Liq5mRow> = snap0(&st).bars.into_iter().map(|r| (r.0, r)).collect();
    for m in l["live"].as_array().unwrap() {
        apply(&st, m["text"].as_str().unwrap(), m["rx_ms"].as_i64().unwrap());
        st.flush();
        while let Ok(m) = rx.try_recv() {
            if let Event::Liq5m(x) = m.ev {
                for r in x.bars {
                    view.insert(r.0, r);
                }
            }
        }
    }
    let got = snap0(&st);
    assert_eq!(got.from_s, l["from_s"].as_f64().unwrap(), "기록 시작");
    let end_bar = (l["end_s"].as_f64().unwrap() as i64).div_euclid(300) * 300;
    let lo = end_bar - 287 * 300; // payload 창(288봉)
    let rows: Vec<Liq5mRow> = got.bars.iter().filter(|r| r.0 >= lo).cloned().collect();
    let want: Vec<Liq5mRow> = l["bars"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| (r[0].as_i64().unwrap(), r[1].as_f64().unwrap(), r[2].as_f64().unwrap(), r[3].as_i64().unwrap(), r[4].as_f64(), r[5].as_f64()))
        .collect();
    assert_eq!(rows, want, "5분봉 청산 원(이벤트 있는 봉)");
    let pushed: Vec<Liq5mRow> = view.into_values().filter(|r| r.0 >= lo).collect();
    assert_eq!(pushed, want, "스냅숏 + 밀기로 지은 것");
    println!("5분봉 청산 원: 봉 {} 일치(파이썬 창 {}봉 중 이벤트 있는 봉)", want.len(), l["payload_bars"]);
    let _ = std::fs::remove_dir_all(&dir);
}
