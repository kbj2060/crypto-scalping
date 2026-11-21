//! 미시 참고 payload(src/ctx_micro.rs) ↔ 원본(import 한 read_window·micro_ref·market_ctx + 옮겨 적은 collect_micro_ref 한 틱) 글자 일치.
//! 정답지: rtserve/tools/gen_ctx_micro_fixture.py(서버 판 venv · 커밋 안 함 · tests/fixtures/gen/ctx_micro/).
#[path = "../src/ctx_micro.rs"]
#[allow(dead_code)]
mod ctx_micro;

use ctx_micro::MicroRef;
use rtserve::raster_read; // ctx_micro.rs 가 crate::raster_read 로 부른다
use rtserve::py::dumps;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen/ctx_micro")
}
fn load(n: &str) -> Value {
    let p = dir().join(n);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|_| panic!("{} 없음 -- tools/gen_ctx_micro_fixture.py 먼저", p.display()))).unwrap()
}
const D: (&str, &str) = (", ", ": ");
fn enc(b: &Option<BTreeMap<i64, Vec<f64>>>) -> Value {
    b.as_ref().map_or(Value::Null, |b| Value::Array(b.iter().map(|(h, v)| json!([h, v])).collect()))
}

#[test]
fn micro_payload_matches_python_tick_by_tick() {
    let d = load("micro.json");
    let now0 = d["now0"].as_i64().unwrap();
    let base = ctx_micro::baseline_from_tape(&dir().join("tape.sqlite"), "ethusdt", now0, 7).unwrap();
    // 알려진 차이(ctx_micro.rs baseline_from_tape 주석): 원본 값은 SQLite 글자·pandas 읽기를 거쳐 1ulp 안팎 비켜 있다 -- 상대 1e-14 안 · 칸 수는 같아야 한다
    let (bb, want) = (base.as_ref().unwrap(), d["base"].as_array().unwrap());
    assert_eq!(bb.len(), want.len());
    let mut worst = 0f64;
    for e in want {
        let (h, w) = (&e[0], &e[1]);
        let w: Vec<f64> = w.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let g = &bb[&h.as_i64().unwrap()];
        assert_eq!(g.len(), w.len(), "시 {h}");
        for (a, b) in g.iter().zip(&w) {
            worst = worst.max((a - b).abs() / b.abs().max(1e-300));
        }
    }
    assert!(worst <= 1e-14, "기준선 상대차 {worst}");
    eprintln!("기준선 분 합 최대 상대차 {worst:e}");
    assert_eq!(enc(&ctx_micro::baseline_from_tape(&dir().join("small.sqlite"), "ethusdt", now0, 7).unwrap()), d["base_small"]);
    assert_eq!(enc(&ctx_micro::baseline_from_tape(&dir().join("tape.sqlite"), "solusdt", now0, 7).unwrap()), d["base_other"]);
    assert_eq!(enc(&ctx_micro::baseline_from_tape(&dir().join("edge.sqlite"), "ethusdt", now0, 7).unwrap()), d["base_edge"]);
    assert!(ctx_micro::baseline_from_tape(&dir().join("none.sqlite"), "ethusdt", now0, 7).is_err());
    let mut m = MicroRef::default();
    for (k, t) in d["ticks"].as_array().unwrap().iter().enumerate() {
        let now = t["now"].as_f64().unwrap();
        let did = m.baseline_due(now);
        assert_eq!(did, t["baseline_did"].as_bool().unwrap(), "#{k} 기준선 때");
        if did {
            m.baseline_set(now, if t["baseline_got"].is_null() { None } else { base.clone() });
        }
        assert_eq!(m.baseline_at, t["baseline_at"].as_f64().unwrap(), "#{k}");
        let b = &t["book"];
        let book = (b[0].as_f64().unwrap(), b[1].as_f64().unwrap(), b[2].as_f64().unwrap(), b[3].as_f64().unwrap());
        let w = raster_read::read_window(&dir().join("raster"), "ethusdt", (now * 1000.0) as i64, 12, 1);
        let p = m.tick(now, book, &w, t["vol60"].as_f64().unwrap(), &t["levels"], &t["liq60"]);
        let got = dumps(p, D, true, None);
        let want = t["payload"].as_str().unwrap();
        if got != want {
            let n = got.chars().zip(want.chars()).take_while(|(a, b)| a == b).count();
            panic!("#{k} now {now}\n  원본: {}\n  Rust: {}", want.chars().skip(n.saturating_sub(60)).take(160).collect::<String>(), got.chars().skip(n.saturating_sub(60)).take(160).collect::<String>());
        }
        assert_eq!(m.depth25_ring.back().map(|x| json!([x.0, x.1])).unwrap_or(Value::Null), t["depth25_last"], "#{k}");
        assert_eq!((m.imb40_ring.len(), m.mid_ring.len()), (t["imb40_n"].as_u64().unwrap() as usize, t["mid_ring_n"].as_u64().unwrap() as usize), "#{k}");
    }
}

#[test]
fn small_functions_match() {
    let f = load("funcs.json");
    for c in f["sr"].as_array().unwrap() {
        let got = ctx_micro::sr_context(Some(&c["levels"]), c["mid"].as_f64().unwrap(), c["imb40"].as_f64());
        assert_eq!(dumps(&Value::Object(got), D, true, None), c["sr"].as_str().unwrap(), "{c}");
    }
    for c in f["agree"].as_array().unwrap() {
        let q = if c["nan"].as_bool().unwrap() { f64::NAN } else { c["q"].as_f64().unwrap() };
        let (a, p) = ctx_micro::agree_state(q, c["o"].as_f64().unwrap(), c["thr"].as_f64().unwrap(), c["prev"].as_str().unwrap());
        assert_eq!(json!([a, p]), c["out"], "{c}");
    }
    for c in f["windows"].as_array().unwrap() {
        let w = raster_read::read_window(&dir().join("raster"), c["sym"].as_str().unwrap(), c["to"].as_i64().unwrap(), 12, 1);
        let mid: Vec<Value> = w.mid.iter().map(|m| if m.is_nan() { Value::Null } else { json!(*m as f64) }).collect();
        let qty: Vec<Vec<f64>> = (0..w.cols).map(|i| w.row(i).iter().map(|v| *v as f64).collect()).collect();
        assert_eq!((json!(w.bin_size), json!(w.bin_lo), json!(mid), json!(qty)), (c["bin_size"].clone(), c["bin_lo"].clone(), c["mid"].clone(), c["qty"].clone()), "{} {}", c["sym"], c["to"]);
    }
    for c in f["pct"].as_array().unwrap() {
        let a: Vec<f64> = c["a"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let x = if c["inf"].as_bool().unwrap() { f64::INFINITY } else { c["x"].as_f64().unwrap() };
        assert_eq!(ctx_micro::pct_rank(x, &a), c["out"].as_f64(), "{c}");
    }
}
