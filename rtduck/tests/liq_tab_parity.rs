//! liq_tab 일치: tools/gen_liq_tab_fixture.py 가 원본 main() 을 두 번 돌린 정답지(서버와 같은 numpy·pandas venv)와
//!   liq_tab_eth.json · liq_tab_all.json 글자 · 1h 봉 캐시 parquet(열·형·행) · klines 요청 URL 을 대조한다.
use duckdb::Connection;
use serde_json::{json, Value};

fn sql(c: &Connection, s: &str) {
    c.execute_batch(s).unwrap();
}

fn parquet_rows(p: &std::path::Path) -> Value {
    let c = Connection::open_in_memory().unwrap();
    let mut st = c.prepare(&format!("SELECT t, high, low, close, qv FROM read_parquet('{}')", p.display())).unwrap();
    let rows: Vec<Value> = st.query_map([], |r| Ok(json!([r.get::<_, i64>(0)?, r.get::<_, f64>(1)?, r.get::<_, f64>(2)?, r.get::<_, f64>(3)?, r.get::<_, f64>(4)?]))).unwrap().map(|x| x.unwrap()).collect();
    let mut st = c.prepare(&format!("DESCRIBE SELECT * FROM read_parquet('{}')", p.display())).unwrap();
    let cols: Vec<Value> = st.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))).unwrap().map(|x| x.unwrap())
        .map(|(n, t)| json!([n, match t.as_str() { "BIGINT" => "int64", "DOUBLE" => "double", o => panic!("{o}") }])).collect();
    json!({"cols": cols, "rows": rows})
}

#[test]
fn liq_tab_matches_python() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/liq_tab_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/liq_tab_golden.json")).unwrap();
    let d = std::env::var("RTDUCK_KEEP").map(std::path::PathBuf::from).unwrap_or_else(|_| std::env::temp_dir().join(format!("rtduck_liqtab_{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&d);
    let (lake, live, hot) = (d.join("lake"), d.join("live"), d.join("hot"));
    for p in [&lake, &live, &hot] {
        std::fs::create_dir_all(p).unwrap();
    }
    // lake: 날짜(일 번호)마다 parquet 하나 -- 원본 생성기와 같은 배치
    let c = Connection::open_in_memory().unwrap();
    sql(&c, "CREATE TABLE bn(ts_ms BIGINT, side VARCHAR, usd DOUBLE, price DOUBLE, symbol VARCHAR); CREATE TABLE okx(inst_id VARCHAR, ts_ms BIGINT, pos_side VARCHAR, bk_px DOUBLE, sz_base DOUBLE);");
    {
        let mut a = c.appender("bn").unwrap();
        for r in inp["lake"]["bn"].as_array().unwrap() {
            a.append_row(duckdb::params![r[0].as_i64().unwrap(), r[1].as_str().unwrap(), r[2].as_f64(), r[3].as_f64().unwrap(), "ETHUSDT"]).unwrap();
        }
        let mut a = c.appender("okx").unwrap();
        for r in inp["lake"]["okx"].as_array().unwrap() {
            a.append_row(duckdb::params![r[0].as_str().unwrap(), r[1].as_i64().unwrap(), r[2].as_str().unwrap(), r[3].as_f64().unwrap(), r[4].as_f64().unwrap()]).unwrap();
        }
    }
    for (tbl, sub) in [("bn", "binance/liquidations/coin=ETH"), ("okx", "okx/liquidations/coin=ALL")] {
        let days: Vec<i64> = c.prepare(&format!("SELECT DISTINCT ts_ms // 86400000 FROM {tbl} ORDER BY 1")).unwrap().query_map([], |r| r.get(0)).unwrap().map(|x| x.unwrap()).collect();
        for day in days {
            let p = lake.join(sub).join(format!("date={day}"));
            std::fs::create_dir_all(&p).unwrap();
            sql(&c, &format!("COPY (SELECT * FROM {tbl} WHERE ts_ms // 86400000 = {day}) TO '{}' (FORMAT PARQUET)", p.join("part.parquet").display()));
        }
    }
    std::fs::create_dir_all(lake.join("binance/liquidations/coin=BTC/date=1")).unwrap();
    let h = &inp["hot"];
    let q = rusqlite::Connection::open(hot.join("binance_ctx.sqlite")).unwrap();
    q.execute_batch("CREATE TABLE liquidations(ts_ms INTEGER, side TEXT, qty REAL, price REAL, usd REAL, symbol TEXT); CREATE TABLE oi_1s(ts_ms INTEGER, symbol TEXT, open_interest REAL, PRIMARY KEY (ts_ms, symbol));").unwrap();
    for r in h["hot_bn"].as_array().unwrap() {
        q.execute("INSERT INTO liquidations VALUES (?,?,?,?,?,?)", rusqlite::params![r[0].as_i64(), r[1].as_str(), r[2].as_f64(), r[3].as_f64(), r[4].as_f64(), r[5].as_str()]).unwrap();
    }
    for r in h["oi"].as_array().unwrap() {
        q.execute("INSERT INTO oi_1s VALUES (?,?,?)", rusqlite::params![r[0].as_i64(), r[1].as_str(), r[2].as_f64()]).unwrap();
    }
    for (db, tbl, key) in [("okx_ctx.sqlite", "okx_oi", "okx_oi"), ("bybit_ctx.sqlite", "bybit_oi", "bybit_oi")] {
        let q = rusqlite::Connection::open(hot.join(db)).unwrap();
        q.execute_batch(&format!("CREATE TABLE {tbl}(ts_ms INTEGER, inst TEXT, oi_base REAL)")).unwrap();
        for r in h[key].as_array().unwrap() {
            q.execute(&format!("INSERT INTO {tbl} VALUES (?,?,?)"), rusqlite::params![r[0].as_i64(), r[1].as_str(), r[2].as_f64()]).unwrap();
        }
    }
    std::fs::write(live.join("liq_tab_all.json"), inp["all_pre"].as_str().unwrap()).unwrap();
    let cx = rtduck::liq_tab::Ctx { lake: &lake, live: &live, hot: &hot, no_rest: false };
    let mut summary = vec![];
    for (run, g) in inp["runs"].as_array().unwrap().iter().zip(gold["runs"].as_array().unwrap()) {
        let now = run["now_ms"].as_i64().unwrap();
        let mut reqs = vec![];
        let log = rtduck::liq_tab::run(&cx, None, now, &mut |u: &str| {
            reqs.push(u.to_string());
            let qs: std::collections::HashMap<&str, &str> = u.split('?').nth(1).unwrap().split('&').filter_map(|x| x.split_once('=')).collect();
            let st: i64 = qs["startTime"].parse().unwrap();
            let lim: usize = qs["limit"].parse().unwrap();
            let rows: Vec<&Value> = inp["klines"][qs["symbol"]].as_array().unwrap().iter().filter(|r| r[0].as_i64().unwrap() >= st && r[0].as_i64().unwrap() <= now).take(lim).collect();
            Ok(serde_json::to_vec(&rows).unwrap())
        }, &mut |_| {}).unwrap();
        let want: Vec<&str> = g["reqs"].as_array().unwrap().iter().map(|r| r[0].as_str().unwrap()).collect();
        assert_eq!(reqs, want, "klines 요청 {log:?}");
        assert!(g["reqs"].as_array().unwrap().iter().all(|r| r[1] == "Liq-tab" || r[1] == "liq-tab"), "UA");
        for (name, v) in g["files"].as_object().unwrap() {
            let p = live.join(name);
            if name.ends_with(".parquet") {
                assert_eq!(parquet_rows(&p), *v, "{name}");
            } else {
                let got = std::fs::read_to_string(&p).unwrap();
                let w = v.as_str().unwrap();
                if got != w {
                    let (a, b): (Value, Value) = (serde_json::from_str(&got).unwrap(), serde_json::from_str(w).unwrap());
                    for k in b.as_object().unwrap().keys() {
                        if a[k] != b[k] {
                            let (x, y) = (a[k].to_string(), b[k].to_string());
                            let i = x.bytes().zip(y.bytes()).position(|(p, q)| p != q).unwrap_or(0);
                            panic!("{name} 열쇠 {k} 갈림 @{i}: rust …{}… / python …{}…", &x[i.saturating_sub(80)..(i + 80).min(x.len())], &y[i.saturating_sub(80)..(i + 80).min(y.len())]);
                        }
                    }
                    panic!("{name} 글자 다름(값은 같음 -- 숫자 모양)");
                }
            }
        }
        summary.push(format!("요청 {} · 파일 {}", reqs.len(), g["files"].as_object().unwrap().len()));
    }
    if std::env::var("RTDUCK_KEEP").is_err() {
        let _ = std::fs::remove_dir_all(&d);
    }
    let e: Value = serde_json::from_str(gold["runs"][1]["files"]["liq_tab_eth.json"].as_str().unwrap()).unwrap();
    eprintln!("청산 탭 일치: {summary:?} · eth.json {}B · 일 {} · 표시 창 {} · 띠 {} · 실측 칸 {} · 최대 고통 주기 {}",
              gold["runs"][1]["files"]["liq_tab_eth.json"].as_str().unwrap().len(), e["days"].as_array().unwrap().len(), e["remain"].as_object().unwrap().len(),
              e["band"]["n"], e["real"].as_array().unwrap().len(), e["pain"]["per"].as_object().unwrap().len());
}
