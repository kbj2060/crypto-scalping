//! 상황 카드 REST 5분봉 재료(src/ctx_rest.rs) ↔ 원본(옮겨 적은 trend_veto_rows·kline_candles·증거 프레임·load_5m_day) 값 일치.
//! 정답지: rtserve/tools/gen_ctx_rest_fixture.py(서버 판 venv · 커밋 안 함 · tests/fixtures/gen/ctx_rest/).
#[path = "../src/ctx.rs"]
#[allow(dead_code)]
mod ctx;
#[path = "../src/ctx_rest.rs"]
#[allow(dead_code)]
mod ctx_rest;
#[path = "../src/ctx_x.rs"]
#[allow(dead_code)]
mod ctx_x;

use rtserve::py; // ctx.rs 가 crate::py 로 부른다
use serde_json::{json, Value};

fn load() -> Value {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen/ctx_rest/rest.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|_| panic!("{} 없음 -- tools/gen_ctx_rest_fixture.py 먼저", p.display()))).unwrap()
}

#[test]
fn rolling_and_ewm_match_pandas() {
    let d = load();
    for (k, s) in d["series"].as_array().unwrap().iter().enumerate() {
        let v: Vec<f64> = s["v"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let r: Vec<Value> = ctx_rest::rolling_mean(&v, 144).iter().map(|x| if x.is_nan() { Value::Null } else { json!(x) }).collect();
        assert_eq!(Value::Array(r), s["roll"], "#{k} rolling");
        assert_eq!(json!(ctx_rest::ewm_mean(&v, 1.0 / 144.0)), s["ewm"], "#{k} ewm");
    }
}

#[test]
fn candles_and_day5_match_python() {
    let d = load();
    for (k, c) in d["cases"].as_array().unwrap().iter().enumerate() {
        let now_ms = c["now_ms"].as_i64().unwrap();
        let raw = &c["raw"];
        let got = match c["kind"].as_str().unwrap() {
            "sol" => ctx_rest::kline_candles(raw, now_ms).unwrap(),
            _ => ctx_rest::candles_from_frame(&ctx_rest::evidence_frame(raw, now_ms).unwrap()),
        };
        let got: Vec<Value> = got.iter().map(|x| json!({"time": x.time, "high": x.high, "low": x.low, "close": x.close, "veto": x.veto, "partial": x.partial})).collect();
        assert_eq!(Value::Array(got), c["candles"], "#{k} {}", c["kind"]);
        let tail = Value::Array(raw.as_array().unwrap().iter().rev().take(300).rev().cloned().collect());
        let d5: Vec<Value> = ctx_rest::day5(&tail, c["now_s"].as_f64().unwrap()).unwrap().iter()
            .map(|x| json!({"time": x.time, "high": x.high, "low": x.low, "close": x.close, "volume": x.volume, "taker": x.taker})).collect();
        assert_eq!(Value::Array(d5), c["day5"], "#{k} day5");
    }
    let _ = py::dumps;
}

#[test]
fn card30_cache_and_stale_rules() {
    let mut cc = ctx_rest::Card30Cache::default();
    assert_eq!(cc.get(None, &[]), (None, None), "모델 없음 = 표로");
    let c = |t: i64| ctx_x::Candle { time: t, high: 1.0, low: 1.0, close: 1.0, volume: 1.0, taker: Some(0.5) };
    let day = vec![c(0), c(300), ctx_x::Candle { taker: None, ..c(600) }, c(900)];
    assert_eq!(ctx_rest::card30_bars(&day, 1000.0).iter().map(|b| b.time).collect::<Vec<_>>(), vec![0, 300], "지금 봉(900) 앞 · taker 있는 것만");
    assert!(ctx_rest::day5_stale(&day, 1500, 2.5) && !ctx_rest::day5_stale(&day, 1200, 2.5) && !ctx_rest::day5_stale(&day, 1500, 1.5));
}
