//! 시장 맥락(src/ctx_mc.rs) ↔ 원본(import 한 market_ctx·micro_ref + 옮겨 적은 market_context_payload·_mc_collectors·_mc_history·_mc_liq_profile·tape_levels) 글자 일치.
//! 정답지: rtserve/tools/gen_ctx_mc_fixture.py(서버 판 venv · 커밋 안 함 · tests/fixtures/gen/ctx_mc/).
#[path = "../src/ctx_mc.rs"]
#[allow(dead_code)]
mod ctx_mc;

use rtserve::py; // ctx_mc.rs 가 crate::py 로 부른다
use rtserve::py::dumps;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;

const D: (&str, &str) = (", ", ": ");
fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen/ctx_mc")
}
fn load() -> Value {
    let p = dir().join("mc.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|_| panic!("{} 없음 -- tools/gen_ctx_mc_fixture.py 먼저", p.display()))).unwrap()
}
fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}
fn pairs(v: &Value) -> Vec<(i64, f64)> {
    v.as_array().unwrap().iter().map(|r| (r[0].as_i64().unwrap(), f(&r[1]))).collect()
}
/// errors 칸의 글자는 대조에서 뺀다(원본 repr(exc)[:100] · Rust 오류 글자 -- 열쇠만 본다).
fn mask(v: &mut Value) {
    if let Some(e) = v.get_mut("errors").and_then(Value::as_object_mut) {
        for x in e.values_mut() {
            *x = json!("<오류>");
        }
    }
}
fn same(got: &Value, want: &str, what: &str) {
    let (mut g, mut w): (Value, Value) = (got.clone(), serde_json::from_str(want).unwrap());
    mask(&mut g);
    mask(&mut w);
    let want = &dumps(&w, D, true, None);
    let g = dumps(&g, D, true, None);
    if &g != want {
        let n = g.chars().zip(want.chars()).take_while(|(a, b)| a == b).count();
        panic!("{what}\n  원본: {}\n  Rust: {}", want.chars().skip(n.saturating_sub(80)).take(220).collect::<String>(), g.chars().skip(n.saturating_sub(80)).take(220).collect::<String>());
    }
}

#[test]
fn market_context_matches_python() {
    let d = load();
    let dd = dir();
    let (okx, by, hl, hp, ls, bn, tape) = (dd.join("okx_ctx.sqlite"), dd.join("bybit_ctx.sqlite"), dd.join("hl_ctx.sqlite"), dd.join("hl_pos.sqlite"), dd.join("oi_lsratio.duckdb"), dd.join("binance_ctx.sqlite"), dd.join("tape.sqlite"));
    let p = ctx_mc::Paths { okx_ctx: &okx, bybit_ctx: &by, hl_ctx: &hl, hl_pos: &hp, lsratio: &ls, bn_ctx: &bn, tape: &tape };
    let now = f(&d["now"]);
    let mut lg = Map::new();
    let col = ctx_mc::mc_collectors(&p, "eth", "ETH-USDT-SWAP", "ETHUSDT", &mut lg);
    same(&Value::Object(col.clone()), &dumps(&d["col"], D, true, None), "collectors eth");
    let col_x = ctx_mc::mc_collectors(&p, "xrp", "XRP-USDT-SWAP", "XRPUSDT", &mut Map::new());
    same(&Value::Object(col_x.clone()), &dumps(&d["col_x"], D, true, None), "collectors xrp");
    let hist = ctx_mc::mc_history(&p, (now * 1000.0) as i64, "ethusdt", "ETH-USDT-SWAP", "ETH", "ETHUSDT");
    let hv = json!({"oi": hist.oi, "oi_per": Value::Object(hist.oi_per.iter().map(|(k, v)| (k.clone(), json!(v))).collect()), "basis": hist.basis});
    same(&hv, &dumps(&d["hist"], D, true, None), "history");
    let prof = ctx_mc::mc_liq_profile(&p, "ETHUSDT", now, 1.0);
    same(&prof, &dumps(&d["prof"], D, true, None), "liq profile");
    let vprof = ctx_mc::tape_levels(&tape, "ethusdt", 0.1, now as i64).unwrap();
    same(&vprof, &dumps(&d["vprof"], D, true, None), "tape levels");
    for (k, c) in d["cases"].as_array().unwrap().iter().enumerate() {
        let mark: BTreeMap<i64, (f64, f64, f64)> = c["mark_ring"].as_array().unwrap().iter().map(|r| (r[0].as_i64().unwrap(), (f(&r[1][0]), f(&r[1][1]), f(&r[1][2])))).collect();
        let mid: BTreeMap<i64, f64> = pairs(&c["mid_ring"]).into_iter().collect();
        let oi1: BTreeMap<i64, f64> = pairs(&c["oi_1s"]).into_iter().collect();
        let d25: Vec<(f64, f64)> = c["depth25"].as_array().unwrap().iter().map(|r| (f(&r[0]), f(&r[1]))).collect();
        let fh: Vec<f64> = c["fh"].as_array().unwrap().iter().map(f).collect();
        let dvy: Vec<f64> = c["dvy"].as_array().unwrap().iter().map(f).collect();
        let asset = c["asset"].as_str().unwrap();
        let x = c["x"].as_object().unwrap().clone();
        let i = ctx_mc::McIn { asset, now, bands_usd: (10_000.0, 100_000.0), micro: &c["micro"], x: &x, ev: &c["ev"], mark_ring: &mark, mid_ring: &mid, depth25: &d25,
                               oi_1s: &oi1, next_funding_ms: c["next_ms"].as_i64(), px_scale: if asset == "eth" { 1.0 } else { 0.0003 / 0.5 } };
        let got = ctx_mc::market_context_payload(&i, if asset == "eth" { &col } else { &col_x }, &hist, &prof, &fh, &dvy, &vprof);
        same(&got, c["payload"].as_str().unwrap(), &format!("#{k} {asset}"));
    }
}

#[test]
fn market_ctx_functions_match() {
    let d = load();
    let fnn = &d["fn"];
    for c in fnn["oi_stats"].as_array().unwrap() {
        same(&Value::Object(ctx_mc::oi_stats(&pairs(&c["s"]))), c["out"].as_str().unwrap(), "oi_stats");
    }
    for c in fnn["sum"].as_array().unwrap() {
        let per: Vec<(String, Vec<(i64, f64)>)> = c["per"].as_array().unwrap().iter().map(|e| (e[0].as_str().unwrap().to_string(), pairs(&e[1]))).collect();
        let (s, u) = ctx_mc::oi_sum_series(&per, c["since"].as_i64().unwrap());
        same(&json!([s, u]), c["out"].as_str().unwrap(), "oi_sum_series");
    }
    for c in fnn["hl"].as_array().unwrap() {
        let rows: Vec<(Option<f64>, Option<f64>)> = c["rows"].as_array().unwrap().iter().map(|r| (r[0].as_f64(), r[1].as_f64())).collect();
        same(&ctx_mc::hl_liq_levels(&rows, f(&c["mid"]), f(&c["bin"])), c["out"].as_str().unwrap(), "hl_liq_levels");
    }
    for c in fnn["lp"].as_array().unwrap() {
        let ev: Vec<(f64, f64, bool)> = c["ev"].as_array().unwrap().iter().map(|r| (f(&r[0]), f(&r[1]), r[2].as_bool().unwrap())).collect();
        same(&ctx_mc::liq_profile(&ev, f(&c["bin"])), c["out"].as_str().unwrap(), "liq_profile");
    }
    for (key, which) in [("va", 0), ("pn", 1)] {
        for c in fnn[key].as_array().unwrap() {
            let k: Vec<i64> = c["k"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
            let v: Vec<f64> = c["v"].as_array().unwrap().iter().map(f).collect();
            let got = if which == 0 { ctx_mc::value_area(&k, &v, f(&c["bw"])) } else { ctx_mc::profile_nodes(&k, &v, f(&c["bw"])) };
            same(&got, c["out"].as_str().unwrap(), key);
        }
    }
    assert_eq!(json!(ctx_mc::liq_burst()), d["burst"]);
    let mut last = None;
    let a = ctx_mc::mc_response(Ok(json!({"available": true, "a": 1})), &mut last, 100.0);
    let b = ctx_mc::mc_response(Err("TimeoutError: ".into()), &mut last, 150.0);
    let b2 = ctx_mc::mc_response(Err("TimeoutError: ".into()), &mut last, 190.0);
    let c = ctx_mc::mc_response(Err(format!("RuntimeError: {}", "x".repeat(300))), &mut last, 230.0);
    for (g, w) in [a, b, b2, c].iter().zip(d["stale"].as_array().unwrap()) {
        same(g, w.as_str().unwrap(), "stale");
    }
    let _ = py::round_nd;
}
