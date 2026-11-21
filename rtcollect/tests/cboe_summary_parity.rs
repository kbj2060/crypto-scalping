//! CBOE 옵션 요약기(cboe_option_summary_20261101.write_state) 파이썬 일치 시험. 정답지 = tools/gen_cboe_fixture.py:
//! 스냅숏 6개(두 날 · 두 종목)를 차례로 두며 4번 쓴 cboe_option_state.json 글자 전체 · 실현변동성 klines 요청 열 · rv 캐시 글자.
use serde_json::Value;
use std::io::Write;

#[test]
fn cboe_state_json_matches_python() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/cboe_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/cboe_golden.json")).unwrap();
    let d = std::env::temp_dir().join(format!("rtcollect_cboe_sum_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let mut reqs: Vec<String> = vec![];
    let mut states = vec![];
    for st in inp["steps"].as_array().unwrap() {
        let p = d.join("cboe").join(st["rel"].as_str().unwrap());
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        let mut z = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
        z.write_all(st["raw"].as_str().unwrap().as_bytes()).unwrap();
        std::fs::write(&p, z.finish().unwrap()).unwrap();
        if st["write"].as_bool().unwrap() {
            let now = st["now"].as_f64().unwrap();
            let cache = d.join("rv.json");
            let mut rv = |sym: &str| {
                rtcollect::cboe_summary::realized_vol_7d(sym, &cache, now, &mut |u: &str| {
                    reqs.push(u.to_string());
                    let s = u.split("symbol=").nth(1).unwrap().split("USDT").next().unwrap();
                    inp["klines"][s].as_str().map(|b| b.as_bytes().to_vec()).ok_or_else(|| "HTTP 500".to_string())
                })
            };
            rtcollect::cboe_summary::write_state(&p, &d.join("state.json"), now, &mut rv).unwrap();
            states.push(std::fs::read_to_string(d.join("state.json")).unwrap());
        }
    }
    let want: Vec<&str> = gold["states"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect();
    assert_eq!(states.len(), want.len(), "쓴 횟수");
    for (k, (a, b)) in states.iter().zip(&want).enumerate() {
        if a != b {
            let (x, y): (Value, Value) = (serde_json::from_str(a).unwrap(), serde_json::from_str(b).unwrap());
            let mut diffs = vec![];
            diff("", &x, &y, &mut diffs);
            panic!("state {k} 글자 다름 -- 값 같음? {} · 다른 칸 {}:\n{}", x == y, diffs.len(), diffs.iter().take(12).cloned().collect::<Vec<_>>().join("\n"));
        }
    }
    assert_eq!(Value::from(reqs.clone()), gold["requests"], "klines 요청 열");
    assert_eq!(std::fs::read_to_string(d.join("rv.json")).ok().map(Value::from).unwrap_or(Value::Null), gold["rv_cache"], "rv 캐시 글자");
    eprintln!("CBOE 요약 일치: state 글자 {}개(마지막 {}B) · klines 요청 {}", states.len(), states.last().unwrap().len(), reqs.len());
    let _ = std::fs::remove_dir_all(&d);
}

fn diff(path: &str, a: &Value, b: &Value, out: &mut Vec<String>) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let xk: Vec<&String> = x.keys().collect();
            let yk: Vec<&String> = y.keys().collect();
            if xk != yk {
                out.push(format!("{path} 열쇠: rust {xk:?} · py {yk:?}"));
            }
            for (k, v) in x {
                if let Some(w) = y.get(k) {
                    diff(&format!("{path}.{k}"), v, w, out);
                }
            }
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            for (i, (v, w)) in x.iter().zip(y).enumerate() {
                diff(&format!("{path}[{i}]"), v, w, out);
            }
        }
        _ => {
            if a.to_string() != b.to_string() {
                out.push(format!("{path}: rust {a} · py {b}"));
            }
        }
    }
}
