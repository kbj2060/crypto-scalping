//! 칸별 아이스버그(live_eth_iceberg_20261016.py) 파이썬 일치 시험. 정답지 = tools/gen_iceberg_fixture.py 가 원본 main() 을 가짜 시계로 두 시간 돌린 것:
//! 사건 jsonl 글자(이미 있던 줄 포함) · 2초 바퀴마다 state 글자(crc32·길이 · 앞뒤 5개는 글자 그대로).
use serde_json::Value;

fn crc32(b: &[u8]) -> u32 {
    let mut c = !0u32;
    for &x in b {
        c ^= x as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 };
        }
    }
    !c
}

const TAPE_DDL: &str = "CREATE TABLE trade_tape_1s(symbol TEXT, ts_sec INTEGER, price_bin INTEGER, buy_qty REAL, sell_qty REAL, buy_n INTEGER, sell_n INTEGER, buy_max REAL, sell_max REAL, retail_buy_qty REAL, retail_sell_qty REAL, whale_buy_qty REAL, whale_sell_qty REAL, whale_buy_n INTEGER, whale_sell_n INTEGER, retail_buy_n INTEGER, retail_sell_n INTEGER, order_buy_n INTEGER, order_sell_n INTEGER, PRIMARY KEY (symbol, ts_sec, price_bin)) WITHOUT ROWID";

#[test]
fn iceberg_matches_python_detector() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/iceberg_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/iceberg_golden.json")).unwrap();
    let d = std::env::temp_dir().join(format!("rtcollect_iceberg_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let dd = d.join("dd/ETHUSDT");
    std::fs::create_dir_all(&dd).unwrap();
    for (name, text) in inp["files"].as_object().unwrap() {
        std::fs::write(dd.join(name), text.as_str().unwrap()).unwrap();
    }
    let con = rusqlite::Connection::open(d.join("tape.sqlite")).unwrap();
    con.execute(TAPE_DDL, []).unwrap();
    for r in inp["tape"].as_array().unwrap() {
        con.execute("INSERT INTO trade_tape_1s(symbol, ts_sec, price_bin, buy_qty, sell_qty) VALUES ('ethusdt', ?, ?, ?, ?)",
                    rusqlite::params![r[0].as_i64(), r[1].as_i64(), r[2].as_f64(), r[3].as_f64()]).unwrap();
    }
    drop(con);
    std::fs::create_dir_all(d.join("out")).unwrap();
    std::fs::write(d.join("out/eth_events.jsonl"), inp["prior_events"].as_str().unwrap()).unwrap();
    let st = rtcollect::iceberg::replay("ETHUSDT", &dd, &d.join("tape.sqlite"), &d.join("out"), inp["t0"].as_f64().unwrap(), inp["t_end"].as_f64().unwrap());
    assert_eq!(std::fs::read_to_string(d.join("out/eth_events.jsonl")).unwrap(), gold["events"].as_str().unwrap(), "사건 jsonl 글자");
    assert_eq!(st.len() as u64, gold["n_states"].as_u64().unwrap(), "바퀴 수");
    for (k, w) in gold["states_text"].as_object().unwrap() {
        assert_eq!(st[k.parse::<usize>().unwrap()], w.as_str().unwrap(), "state {k}");
    }
    for (k, (s, w)) in st.iter().zip(gold["states_crc"].as_array().unwrap()).enumerate() {
        assert_eq!((crc32(s.as_bytes()) as u64, s.len() as u64), (w[0].as_u64().unwrap(), w[1].as_u64().unwrap()), "state {k}: {s}");
    }
    let n_ev = gold["events"].as_str().unwrap().lines().count();
    eprintln!("아이스버그 일치: 사건 {n_ev}줄 · state 글자 {} · 마지막 {}", st.len(), st.last().unwrap());
    let _ = std::fs::remove_dir_all(&d);
}
