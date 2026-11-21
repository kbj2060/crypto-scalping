//! 관계 읽기 입력 x(src/ctx_x.rs) · 장부 한 줄·꼬리(src/ctx.rs) · bybit_bars · 기준선 ↔ 원본(옮겨 적은 server.py 함수 · import 한 micro_ref) 글자 일치.
//! 정답지: rtserve/tools/gen_ctx_x_fixture.py(서버 판 venv · 커밋 안 함 · tests/fixtures/gen/ctx_x/).
//! x 는 걸음마다 같은 FlowState 로 이어 부른다(원본 fr_bar_net·okx_bar_net_by·flow_all_hold 가 이어지듯).
#[path = "../src/ctx.rs"]
#[allow(dead_code)]
mod ctx;
#[path = "../src/ctx_x.rs"]
#[allow(dead_code)]
mod ctx_x;

use ctx_x::{Candle, FlowState, Scales, XIn};
use rtcore::situation::{flow_delta, flow_nets_tick, FlowNets};

type Cells = Vec<[f64; 6]>;
use rtserve::py; // ctx.rs 가 crate::py 로 부른다(lib 에 붙으면 그대로)
use rtserve::py::dumps;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen/ctx_x")
}
fn load(name: &str) -> Value {
    let p = dir().join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|_| panic!("{} 없음 -- tools/gen_ctx_x_fixture.py 먼저", p.display()))).unwrap()
}
const D: (&str, &str) = (", ", ": ");
fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}
fn opt(v: &Value) -> Option<f64> {
    v.as_f64()
}
fn bars_of(v: &Value) -> BTreeMap<i64, Cells> {
    v.as_array().unwrap().iter().map(|b| (b[0].as_i64().unwrap(), b[1].as_array().unwrap().iter().map(|c| std::array::from_fn(|j| f(&c[j]))).collect())).collect()
}
fn scales_of(v: &Value) -> Scales {
    v.as_array().unwrap().iter().map(|e| ((e[0][0].as_i64().unwrap(), e[0][1].as_i64().unwrap()), e[1].as_object().unwrap().clone())).collect()
}
fn enc(s: &Scales) -> Value {
    Value::Array(s.iter().map(|(k, v)| json!([[k.0, k.1], Value::Object(v.clone())])).collect())
}

#[test]
fn flow_read_ctx_matches_python_step_by_step() {
    let mut total = 0;
    for run in ["okx_late", "okx_mid", "no_okx", "short", "twelve", "grow"] {
        let steps = load(&format!("x_{run}.json"));
        let mut st = FlowState::default();
        // 코어 쪽(풋프린트 칸 → 봉 순합)과 받는 쪽(Micro.flow 를 쌓은 값)을 따로 든다 -- 받는 쪽으로 x 를 센다
        let (mut core, mut sent, mut recv): (FlowNets, Option<FlowNets>, FlowNets) = Default::default();
        for (k, s) in steps.as_array().unwrap().iter().enumerate() {
            let fbars = bars_of(&s["fbars"]);
            let okx_bars = bars_of(&s["okx_bars"]);
            let bybit: BTreeMap<i64, [f64; 4]> = s["bybit"].as_array().unwrap().iter().map(|b| (b[0].as_i64().unwrap(), std::array::from_fn(|j| f(&b[1][j])))).collect();
            let fs = scales_of(&s["flow_scales"]);
            let fsa: HashMap<String, Scales> = s["flow_scales_all"].as_array().unwrap().iter().map(|e| (e[0].as_str().unwrap().to_string(), scales_of(&e[1]))).collect();
            let bars: Vec<(Option<f64>, Option<f64>)> = s["bars"].as_array().unwrap().iter().map(|b| (opt(&b[0]), opt(&b[1]))).collect();
            let day5: Vec<Candle> = s["day5"].as_array().unwrap().iter().map(|c| Candle { time: c[0].as_i64().unwrap(), high: f(&c[1]), low: f(&c[2]), close: f(&c[3]) }).collect();
            let ring: Vec<f64> = s["ring"].as_array().unwrap().iter().map(f).collect();
            let now = f(&s["now"]);
            flow_nets_tick(&mut core, now, &fbars, s["okx_first_bar"].as_i64().unwrap(), &okx_bars, |c: &Cells| c.clone());
            let d = flow_delta(&core, sent.as_ref());
            let d: rtcore::proto::FlowSums = serde_json::from_str(&serde_json::to_string(&d).unwrap()).unwrap(); // 선 위를 지난 값
            ctx_x::apply_flow(&mut recv, &d);
            sent = Some(core.clone());
            assert_eq!(recv, core, "{run} #{k} 받은 값 = 코어 값");
            let i = XIn {
                now,
                nets: &recv,
                bybit: &bybit,
                flow_scales: &fs,
                flow_scales_all: &fsa,
                bars: &bars,
                day5: &day5,
                card30: (opt(&s["card30"][0]), opt(&s["card30"][1])),
                micro: &s["micro"],
                imb40_ring: &ring,
            };
            let got = dumps(&Value::Object(ctx_x::flow_read_ctx(&i, &mut st)), D, true, None);
            let want = s["x"].as_str().unwrap();
            if got != want {
                let n = got.chars().zip(want.chars()).take_while(|(a, b)| a == b).count();
                panic!("{run} #{k}\n  원본: {}\n  Rust: {}", want.chars().skip(n.saturating_sub(80)).take(200).collect::<String>(), got.chars().skip(n.saturating_sub(80)).take(200).collect::<String>());
            }
            total += 1;
        }
    }
    assert_eq!(total, 202);
}

#[test]
fn fused_line_and_tail_match() {
    for (k, c) in load("fused.json").as_array().unwrap().iter().enumerate() {
        let got = ctx::fused_log_line(f(&c["now"]), &c["read"], &c["mid"], &c["card_p"], &c["feat_bar"]);
        assert_eq!(got.as_deref(), c["line"].as_str(), "#{k}");
    }
    for t in load("tails.json").as_array().unwrap() {
        assert_eq!(ctx::last_logged_bar(&dir().join(t["file"].as_str().unwrap())), t["bar"].as_i64(), "{}", t["file"]);
    }
}

#[test]
fn fused_log_writes_once_per_bar_only_when_not_readonly() {
    let p = std::env::temp_dir().join(format!("rtserve_fused_{}.jsonl", std::process::id()));
    std::fs::write(&p, "{\"bar\": 1763700000}\n").unwrap();
    let rd = load("fused.json").as_array().unwrap().iter().find(|c| !c["line"].is_null()).unwrap()["read"].clone();
    let mut fl = ctx::FusedLog::new(p.clone());
    // 병행 중(기본) = 안 쓴다
    std::env::remove_var("RTSERVE_READONLY");
    assert!(fl.tick(1763700100.0, &rd, &json!(1.0), &json!({}), &Value::Null).is_none());
    std::env::set_var("RTSERVE_READONLY", "0");
    assert!(fl.tick(1763700100.0, &rd, &json!(1.0), &json!({}), &Value::Null).is_none(), "되살린 마지막 봉과 같은 봉");
    assert!(fl.tick(1763700300.5, &rd, &json!(1.0), &json!({}), &Value::Null).is_some());
    assert!(fl.tick(1763700599.0, &rd, &json!(1.0), &json!({}), &Value::Null).is_none(), "같은 봉 두 번");
    assert!(fl.tick(1763700600.0, &rd, &json!(1.0), &json!({}), &Value::Null).is_some());
    std::env::remove_var("RTSERVE_READONLY");
    assert_eq!(std::fs::read_to_string(&p).unwrap().lines().count(), 3);
    assert_eq!(ctx::last_logged_bar(&p), Some(1763700600));
    let _ = std::fs::remove_file(&p);
}

#[test]
fn bybit_bars_and_scales_match() {
    let t = load("tape.json");
    let db = dir().join("tape.sqlite");
    let now = t["now"].as_i64().unwrap();
    let b = &t["bybit"];
    let got = ctx_x::bybit_bars("ETHUSDT", b["t0"].as_i64().unwrap(), b["t1"].as_i64().unwrap(), &db);
    let enc_b = Value::Array(got.iter().map(|(k, v)| json!([k, v])).collect());
    assert_eq!(dumps(&enc_b, D, true, None), dumps(&b["out"], D, true, None));
    assert!(ctx_x::bybit_bars("ETHUSDT", 0, 1, &dir().join("none.sqlite")).is_empty());
    let fs = ctx_x::flow_hour_scales_from_bars(&ctx_x::tape_flow_rows(&db, "ethusdt", now, 14, 0).unwrap());
    assert_eq!(dumps(&enc(&fs), D, true, None), dumps(&t["fs"], D, true, None));
    let fs2 = ctx_x::flow_hour_scales_from_bars(&ctx_x::tape_flow_rows(&db, "ethusdt", now, 14, t["since"].as_i64().unwrap()).unwrap());
    assert_eq!(dumps(&enc(&fs2), D, true, None), dumps(&t["fs_since"], D, true, None));
    for (key, bybit_db) in [("combo", db.clone()), ("combo_nb", dir().join("none.sqlite"))] {
        let srcs = [("binance", db.as_path(), "ethusdt"), ("okx", db.as_path(), "ETH-USDT-SWAP"), ("bybit", bybit_db.as_path(), "ETHUSDT")];
        let got = ctx_x::flow_hour_scales_combos(&srcs, now, 14, 0);
        for e in t[key].as_array().unwrap() {
            assert_eq!(dumps(&enc(&got[e[0].as_str().unwrap()]), D, true, None), dumps(&e[1], D, true, None), "{key} {}", e[0]);
        }
    }
}

#[test]
fn streams_payload_has_python_shape() {
    use rtcore::proto::{StreamStat, Streams};
    // 원본 fo_state 1801 · mp_state 1774 (첫 마크 전에는 next_funding_ms 키가 없다 · 1894)
    let s = Streams::default();
    let (fo, mp) = ctx::streams_payload(&s);
    assert_eq!(dumps(&fo, D, true, None), r#"{"connected": false, "since": null, "last_event_ms": null, "events": 0, "errors": 0, "last_error": null}"#);
    assert_eq!(dumps(&mp, D, true, None), r#"{"connected": false, "since": null, "last_ms": null, "events": 0, "errors": 0, "last_error": null}"#);
    let s = Streams { fo: StreamStat::default(), mp: StreamStat { connected: true, since: Some(1763699000.5), last_ms: Some(1763700000000), events: 3, errors: 1, last_error: Some("x".into()), next_funding_ms: None } };
    let (_, mp) = ctx::streams_payload(&s);
    assert_eq!(dumps(&mp, D, true, None), r#"{"connected": true, "since": 1763699000.5, "last_ms": 1763700000000, "events": 3, "errors": 1, "last_error": "x", "next_funding_ms": null}"#);
}

/// 붙이는 쪽 함수 하나(situation_tick): 계산 전 · 코어 값 있음 두 경우의 본문 모양.
#[test]
fn situation_tick_assembles_payload() {
    let nets = FlowNets::default();
    let (bybit, fs, fsa) = (BTreeMap::new(), Scales::new(), HashMap::new());
    let micro = json!({"available": true, "mid": 3000.5});
    let i = XIn { now: 1763700123.25, nets: &nets, bybit: &bybit, flow_scales: &fs, flow_scales_all: &fsa, bars: &[], day5: &[], card30: (None, None), micro: &micro, imb40_ring: &[] };
    let streams = rtcore::proto::Streams::default();
    let by = vec![("eth".to_string(), json!({"mult": 1.0})), ("xrp".to_string(), Value::Null)];
    let (p, sse, _) = ctx_x::situation_tick(None, 0, None, None, &streams, &i, &mut FlowState::default(), &by, None);
    assert_eq!(dumps(&p["now"], D, true, None), r#"{"ok": false, "reason": "계산 전"}"#.replace("계산 전", "\\uacc4\\uc0b0 \\uc804"));
    assert_eq!((p["computed_at"].clone(), p["read"].clone(), p["vol_mult"].clone()), (json!(0.0), Value::Null, json!({"mult": 1.0})));
    assert!(sse.starts_with("event: situation\ndata: {\"now\":{\"ok\":false,\"reason\":\"계산 전\"}") && sse.ends_with("\n\n"));
    let core = json!({"ok": true, "dir": 1, "labels": ["상승 +12bp"], "evidence": {"mid": 3000.0, "move_bp": 12.0, "range_bp": 20.0, "dir": 1, "move_ratio": 0.6, "cvd": 10,
        "reject": false, "liq_long": 0, "liq_short": 0, "veto": 0, "obi": null, "persist": null, "funding": null, "basis_bp": null, "basis_d_bp": null,
        "basis_thr_bp": null, "crowd": 0, "lead": 0, "btc_move_bp": null, "btc_rel": null}, "regime": {"ratio": 0.6, "thr": 0.25, "margin": 0.35, "enter": 0.45, "exit": 0.25},
        "computed_at": 1763700120.0, "missing": ["veto", "book"]});
    let (p, _, x) = ctx_x::situation_tick(Some(&core), 1, Some(0.25), None, &streams, &i, &mut FlowState::default(), &by, None);
    assert_eq!(p["now"]["computed_at"], json!(1763700123.25));
    assert_eq!(p["computed_at"], json!(1763700123.25));
    assert!(p["now"].get("missing").is_none() && p["now"]["evidence"]["veto"] == json!(1) && p["now"]["evidence"]["obi"] == json!(0.25));
    assert_eq!(p["read"], ctx::read(&p["now"]["evidence"], &Value::Object(x)));
    assert_eq!(p["streams"]["fo"]["events"], json!(0));
}
