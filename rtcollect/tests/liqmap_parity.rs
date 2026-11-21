//! liqmap_scorer 일치: tools/gen_liqmap_fixture.py 가 원본 main() 을 세 번 돌린 정답지와
//! 덧붙인 jsonl 글자 · status 글자 · klines 요청 URL 을 대조한다(같은 합성 hot SQLite 셋 · 같은 klines 본문 · 같은 시계).
use rusqlite::Connection;
use serde_json::Value;

fn db(p: &std::path::Path, ddl: &[&str], ins: &str, rows: &Value) {
    let mut c = Connection::open(p).unwrap();
    for d in ddl {
        c.execute(d, []).unwrap();
    }
    let tx = c.transaction().unwrap();
    {
        let mut s = tx.prepare(ins).unwrap();
        for r in rows.as_array().unwrap() {
            let v: Vec<rusqlite::types::Value> = r.as_array().unwrap().iter().map(|x| match x {
                Value::Null => rusqlite::types::Value::Null,
                Value::String(s) => rusqlite::types::Value::Text(s.clone()),
                Value::Number(n) if n.is_i64() => rusqlite::types::Value::Integer(n.as_i64().unwrap()),
                Value::Number(n) => rusqlite::types::Value::Real(n.as_f64().unwrap()),
                _ => unreachable!(),
            }).collect();
            s.execute(rusqlite::params_from_iter(v)).unwrap();
        }
    }
    tx.commit().unwrap();
}

#[test]
fn liqmap_scorer_matches_python() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/liqmap_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/liqmap_golden.json")).unwrap();
    let d = std::env::temp_dir().join(format!("rtc_liqmap_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.join("out")).unwrap();
    let h = &inp["hot"];
    db(&d.join("binance_ctx.sqlite"), &["CREATE TABLE mark_price_1s(ts_ms INTEGER, symbol TEXT, mark REAL, index_px REAL, funding_rate REAL, next_funding_ms INTEGER, PRIMARY KEY (ts_ms, symbol)) WITHOUT ROWID",
        "CREATE TABLE liquidations(ts_ms INTEGER, side TEXT, qty REAL, price REAL, usd REAL, symbol TEXT)", "CREATE INDEX mark_sym ON mark_price_1s(symbol, ts_ms)"],
        "INSERT INTO mark_price_1s(ts_ms, symbol, mark) VALUES (?,?,?)", &h["marks"]);
    db(&d.join("binance_ctx.sqlite"), &[], "INSERT INTO liquidations(ts_ms, side, qty, symbol) VALUES (?,?,?,?)", &h["binance"]);
    db(&d.join("okx_ctx.sqlite"), &["CREATE TABLE okx_liquidations(inst_id VARCHAR, inst_family VARCHAR, ts_ms BIGINT, side VARCHAR, pos_side VARCHAR, bk_px DOUBLE, sz_contracts DOUBLE, sz_base DOUBLE, bk_loss DOUBLE)"],
        "INSERT INTO okx_liquidations(inst_id, ts_ms, pos_side, sz_base) VALUES (?,?,?,?)", &h["okx"]);
    db(&d.join("bybit_liq.sqlite"), &["CREATE TABLE bybit_liquidations(ts_ms INTEGER, symbol TEXT, side TEXT, qty REAL, price REAL, recv_ms INTEGER)",
        "CREATE INDEX bybit_liq_sym_ts ON bybit_liquidations(symbol, ts_ms)"], "INSERT INTO bybit_liquidations(ts_ms, symbol, side, qty) VALUES (?,?,?,?)", &h["bybit"]);
    let (out, status) = (d.join("out/liqmap_score.jsonl"), d.join("out/status.json"));
    let pre = inp["pre"].as_str().unwrap();
    std::fs::write(&out, pre).unwrap();
    let mut seen = pre.len();
    let (mut lines, mut cells) = (0, 0);
    for (run, g) in inp["runs"].as_array().unwrap().iter().zip(gold["runs"].as_array().unwrap()) {
        let mut reqs = vec![];
        let body = run["body"].as_str().unwrap().to_string();
        let log = rtcollect::liqmap_scorer::run(&out, &status, &d, run["now"].as_f64().unwrap(), &mut |u: &str| {
            reqs.push(u.to_string());
            Ok(body.clone().into_bytes())
        }).unwrap();
        let txt = std::fs::read_to_string(&out).unwrap();
        let (got, want) = (&txt[seen..], g["out_added"].as_str().unwrap());
        for (i, (a, b)) in got.lines().zip(want.lines()).enumerate() {
            assert_eq!(a, b, "줄 {i} 갈림 ({log})");
        }
        assert_eq!(got, want, "덧붙인 글자");
        lines += want.lines().count();
        for l in want.lines() {
            let v: Value = serde_json::from_str(l).unwrap();
            let n = |o: &Value| o.as_object().unwrap().values().map(|s| s.as_object().unwrap().len()).sum::<usize>();
            cells += n(&v["real"]) + v["maps"].as_object().unwrap().values().map(n).sum::<usize>();
        }
        seen = txt.len();
        assert_eq!(std::fs::read_to_string(&status).unwrap(), g["status"].as_str().unwrap(), "status");
        let want_reqs: Vec<&str> = g["reqs"].as_array().unwrap().iter().map(|r| r[0].as_str().unwrap()).collect();
        assert_eq!(reqs, want_reqs, "klines 요청");
        assert!(g["reqs"].as_array().unwrap().iter().all(|r| r[1] == "liqmap-scorer" && r[2] == 20), "UA·timeout = main.rs binance_rest(\"liqmap-scorer\")");
    }
    let _ = std::fs::remove_dir_all(&d);
    eprintln!("청산지도 점수 일치: 새 줄 {lines} · 지도·실측 칸 {cells} · 실행 {} 번", inp["runs"].as_array().unwrap().len());
}
