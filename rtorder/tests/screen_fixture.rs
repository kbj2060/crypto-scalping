// 계좌 카드 화면 몫 정답지 대조(tools/gen_screen_fixture.py -- 원본 dashboard/orders.py 함수를 그대로 돌린 것):
// 원장 기록(가드·두 번째 호출 0건·로그·seen) · 원장 꼬리·습관·추세 · 0시 순자산(파일 포함) · income 접기·밀기(요청 열·상태).
use rtorder::account::AccountRest;
use rtorder::screen;
use rtorder::sign::P;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;

fn fixture() -> Value {
    serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/screen_fixture.json")).unwrap()).unwrap()
}
fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rtorder_screen_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}
fn block_on<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(f)
}

#[test]
fn record_trips_matches_python() {
    let fx = &fixture()["record"];
    let d = tmp("record");
    std::fs::write(d.join(screen::TRIP_LEDGER), fx["pre"].as_str().unwrap()).unwrap();
    let mut seen = screen::load_trip_keys(&d);
    let (n, mut logs) = screen::record_trips(&d, &fx["payload"], &mut seen, 1_763_830_000.0);
    let (again, logs2) = screen::record_trips(&d, &fx["payload"], &mut seen, 1_763_830_000.0);
    logs.extend(logs2);
    assert_eq!((json!(n), json!(again)), (fx["n"].clone(), fx["again"].clone()));
    let text = std::fs::read_to_string(d.join(screen::TRIP_LEDGER)).unwrap();
    let mut rows: Vec<Value> = text[fx["pre"].as_str().unwrap().len()..].lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let mut want = fx["appended"].as_array().unwrap().clone();
    for r in rows.iter_mut().chain(want.iter_mut()) {
        assert!(r.as_object_mut().unwrap().remove("recorded_at").is_some(), "recorded_at");
    }
    assert_eq!(Value::Array(rows), Value::Array(want), "덧붙인 줄");
    assert_eq!(json!(logs), fx["logs"], "로그");
    let mut keys: Vec<String> = seen.keys().cloned().collect();
    keys.sort();
    assert_eq!(json!(keys), fx["seen"]);
}

#[test]
fn ledger_rows_habits_trend_match_python() {
    let fx = &fixture()["ledger"];
    let d = tmp("ledger");
    std::fs::write(d.join(screen::TRIP_LEDGER), fx["file"].as_str().unwrap()).unwrap();
    std::fs::write(d.join(screen::TRIP_HABITS), fx["habits"].as_str().unwrap()).unwrap();
    let mut rows = screen::load_trip_rows(&d);
    let daily: HashMap<String, (Vec<f64>, Vec<i64>)> = fx["daily"].as_object().unwrap().iter()
        .map(|(k, v)| (k.clone(), (serde_json::from_value(v[0].clone()).unwrap(), serde_json::from_value(v[1].clone()).unwrap()))).collect();
    screen::annotate_trend(&mut rows, &daily);
    assert_eq!(Value::Array(rows), fx["rows"]);
}

#[test]
fn day_start_matches_python() {
    let d = tmp("day");
    for c in fixture()["day_start"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let p = d.join(screen::DAY_START);
        let _ = std::fs::remove_file(&p);
        if !c["pre"].is_null() {
            std::fs::write(&p, c["pre"].to_string()).unwrap();
        }
        let out = screen::day_start(&d, &c["payload"], c["now"].as_f64().unwrap());
        assert_eq!(out.unwrap_or(Value::Null), c["out"], "{name}: 반환");
        let file: Value = std::fs::read_to_string(&p).ok().map_or(Value::Null, |s| serde_json::from_str(&s).unwrap());
        assert_eq!(file, c["file"], "{name}: 파일");
    }
}

struct IncomeRest {
    master: Vec<Value>,
    calls: Vec<Value>,
}
impl AccountRest for IncomeRest {
    fn has_keys(&self) -> bool {
        true
    }
    fn now(&self) -> f64 {
        0.0
    }
    async fn clock_offset(&mut self) -> i64 {
        0
    }
    async fn get(&mut self, path: &str, params: &[(&str, P)], _offset: i64) -> Value {
        assert_eq!(path, "/fapi/v1/income");
        let g = |k: &str| params.iter().find(|(n, _)| *n == k).map(|(_, v)| v.py_str().parse::<i64>().unwrap()).unwrap();
        let (s, e) = (g("startTime"), g("endTime"));
        assert_eq!(g("limit"), screen::INCOME_LIMIT as i64);
        self.calls.push(json!([s, e]));
        Value::Array(self.master.iter().filter(|r| (s..=e).contains(&r["time"].as_i64().unwrap())).take(screen::INCOME_LIMIT).cloned().collect())
    }
}

#[test]
fn income_refresh_matches_python() {
    let fx = &fixture()["income"];
    let master = fx["master"].as_array().unwrap().clone();
    for (max_calls, st_key, calls_key) in [(40, "full", "full_calls"), (3, "part", "part_calls")] {
        let mut rest = IncomeRest { master: master.clone(), calls: vec![] };
        let mut st = json!({});
        block_on(screen::income_refresh(&mut st, &mut rest, 0, fx["now_ms"].as_i64().unwrap(), max_calls)).unwrap();
        assert_eq!(Value::Array(rest.calls), fx[calls_key], "{st_key}: 요청 열");
        // 합계는 부동소수 덧셈 순서가 같아 비트까지 같다
        assert_eq!(st, fx[st_key], "{st_key}: 상태");
    }
}
