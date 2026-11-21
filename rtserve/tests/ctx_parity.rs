//! 상황 카드(src/ctx.rs) ↔ 원본 flow_read.read · vol_size_mult · server.py situation_payload · SSE 상황 한 건 본문 글자 일치.
//! 정답지: rtserve/tools/gen_ctx_fixture.py(서버 판 venv 로 · 커밋 안 함 · tests/fixtures/gen/ctx/).
//! 비교 = 원본 json.dumps 기본(/api/situation = aiohttp json_response) 과 compact·ensure_ascii=False(SSE) 둘 다.
//! ctx.rs 는 아직 lib.rs 에 안 붙였다(라우팅·hub 구독은 다음 조각) -- 그 파일만 끌어다 돌린다.
#[path = "../src/ctx.rs"]
#[allow(dead_code)]
mod ctx;

use rtserve::py::dumps;
use serde_json::{json, Value};
use std::path::PathBuf;

fn load(name: &str) -> Vec<Value> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen/ctx").join(name);
    let t = std::fs::read_to_string(&p).unwrap_or_else(|_| panic!("{} 없음 -- tools/gen_ctx_fixture.py 먼저", p.display()));
    serde_json::from_str::<Vec<Value>>(&t).unwrap()
}
const D: (&str, &str) = (", ", ": ");
const C: (&str, &str) = (",", ":");

fn first_diff(a: &str, b: &str) -> String {
    let i = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
    let s = |t: &str| t.chars().skip(i.saturating_sub(60)).take(140).collect::<String>();
    format!("\n  원본: {}\n  Rust: {}", s(a), s(b))
}

#[test]
fn read_matches_flow_read() {
    let cases = load("read.json");
    let mut bad = 0;
    for (i, c) in cases.iter().enumerate() {
        let out = ctx::read(&c["ev"], &c["x"]);
        let (d, s) = (dumps(&out, D, true, None), dumps(&out, C, false, None));
        if d != c["dflt"].as_str().unwrap() || s != c["sse"].as_str().unwrap() {
            bad += 1;
            if bad <= 3 {
                eprintln!("#{i} ev={} x={}{}", c["ev"], c["x"], first_diff(c["dflt"].as_str().unwrap(), &d));
            }
        }
    }
    assert_eq!(bad, 0, "{bad}/{} 다름", cases.len());
}

#[test]
fn vol_size_mult_matches() {
    let cases = load("vol.json");
    let mut bad = 0;
    for (i, c) in cases.iter().enumerate() {
        let bars: Vec<(i64, f64)> = c["bars"].as_array().unwrap().iter().map(|b| (b[0].as_i64().unwrap(), b[1].as_f64().unwrap())).collect();
        let got = dumps(&ctx::vol_size_mult(&bars, c["asset"].as_str().unwrap()), D, true, None);
        if got != c["out"].as_str().unwrap() {
            bad += 1;
            if bad <= 3 {
                eprintln!("#{i}{}", first_diff(c["out"].as_str().unwrap(), &got));
            }
        }
    }
    assert_eq!(bad, 0, "{bad}/{} 다름", cases.len());
}

#[test]
fn situation_payload_and_sse_match() {
    let cases = load("payload.json");
    for (i, c) in cases.iter().enumerate() {
        let st = &c["state"];
        let now = if i == 0 { ctx::initial_now() } else { st["now"].clone() };
        let by: Vec<(String, Value)> = c["by"].as_array().unwrap().iter().map(|p| (p[0].as_str().unwrap().to_string(), p[1].clone())).collect();
        let rd = st.get("read").cloned().unwrap_or(Value::Null);
        let vm = st.get("vol_mult").cloned().unwrap_or(Value::Null);
        let p = ctx::situation_payload(&now, &st["computed_at"], &c["fo"], &c["mp"], &rd, &vm, &by);
        let d = dumps(&p, D, true, None);
        assert_eq!(d, c["dflt"].as_str().unwrap(), "#{i}{}", first_diff(c["dflt"].as_str().unwrap(), &d));
        let s = ctx::sse_situation(&p);
        assert_eq!(s, c["sse"].as_str().unwrap(), "#{i} sse");
    }
}

/// 코어 Situation(veto 0 · obi/persist None · missing 붙음) → patch_now → 원본 모양(classify 결과 + 받는 쪽 veto·obi·persist).
#[test]
fn patch_now_restores_passthrough_fields() {
    for c in load("payload.json").iter().skip(1) {
        let want = &c["state"]["now"];
        let mut core = want.clone();
        let (veto, obi, pers) = (want["evidence"]["veto"].as_i64().unwrap(), want["evidence"]["obi"].as_f64(), want["evidence"]["persist"].as_f64());
        core["evidence"]["veto"] = json!(0);
        core["evidence"]["obi"] = Value::Null;
        core["evidence"]["persist"] = Value::Null;
        core.as_object_mut().unwrap().insert("missing".into(), json!(["veto", "book"]));
        assert_eq!(dumps(&ctx::patch_now(&core, veto, obi, pers), D, true, None), dumps(want, D, true, None));
    }
    let off = json!({"ok": false, "reason": "완결 봉 3 < 7", "missing": ["veto", "book"]});
    assert_eq!(ctx::patch_now(&off, 1, None, None), json!({"ok": false, "reason": "완결 봉 3 < 7"}));
}
