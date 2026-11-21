//! 정답지 대조: `tools/gen_plan_fixture.py` 가 **원본 파이썬 함수**로 만든 입력·출력 쌍을 Rust 함수에 넣어 필드 단위로 비교한다.
//! 실수 = 상대 1e-12 · 거래소로 나가는 값(가격·수량 칸) = 파이썬 repr 문자열까지 · 거절 = 예외 종류·사유 문자열까지.

use rtorder::plan::*;
use rtorder::pyfmt;
use serde_json::{json, Map, Value};

fn f(a: &Value, k: &str) -> f64 {
    a[k].as_f64().unwrap_or_else(|| panic!("{k}: {a}"))
}
fn of(a: &Value, k: &str) -> Option<f64> {
    a.get(k).and_then(Value::as_f64)
}
fn s<'a>(a: &'a Value, k: &str) -> &'a str {
    a[k].as_str().unwrap()
}
fn filters(a: &Value) -> Filters {
    serde_json::from_value(a["filters"].clone()).unwrap()
}
fn pair(v: Option<&Value>) -> Option<(f64, f64)> {
    v.and_then(Value::as_array).map(|x| (x[0].as_f64().unwrap(), x[1].as_f64().unwrap()))
}
fn val<T: serde::Serialize>(r: Result<T, PlanError>) -> Result<Value, PlanError> {
    r.map(|x| serde_json::to_value(x).unwrap())
}

/// 한 사례를 Rust 로 돌린다. Err = 파이썬 예외와 같아야 하는 거절.
fn run(fname: &str, a: &Value) -> Result<Value, PlanError> {
    match fname {
        "build_entry_plan" => val(build_entry_plan(&EntryInput {
            side: s(a, "side"), best_bid: f(a, "best_bid"), best_ask: f(a, "best_ask"), recommended_qty: f(a, "recommended_qty"),
            cap_notional: of(a, "cap_notional"), filters: filters(a), symbol: a.get("symbol").and_then(Value::as_str).unwrap_or("ETHUSDT"),
            existing_notional: of(a, "existing_notional").unwrap_or(0.0), equity: of(a, "equity").unwrap_or(0.0),
            leverage: of(a, "leverage").unwrap_or(0.0), fraction: of(a, "fraction").unwrap_or(1.0),
            fraction_of_equity: a.get("fraction_of_equity").and_then(Value::as_bool).unwrap_or(false),
            order_leverage: of(a, "order_leverage").unwrap_or(0.0), cap_label: a.get("cap_label").and_then(Value::as_str).unwrap_or("상한"),
            available_margin: of(a, "available_margin"),
        })),
        "build_exit_plan" => val(build_exit_plan(&ExitInput {
            position_side: s(a, "position_side"), position_qty: f(a, "position_qty"), best_bid: f(a, "best_bid"), best_ask: f(a, "best_ask"),
            filters: filters(a), symbol: a.get("symbol").and_then(Value::as_str).unwrap_or("ETHUSDT"),
            entry_price: of(a, "entry_price").unwrap_or(0.0), mark_price: of(a, "mark_price").unwrap_or(0.0),
            vol_bpm: of(a, "vol_bpm"), fraction: of(a, "fraction").unwrap_or(1.0),
        })),
        "exit_deadline_sec" => val(exit_deadline_sec(of(a, "vol_bpm"))),
        "realized_vol_bpm" => {
            let c: Vec<f64> = a["closes"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
            Ok(json!(realized_vol_bpm(&c)))
        }
        "line_bracket" => val(line_bracket(s(a, "position_side"), f(a, "ref_level"), f(a, "tp_level"), f(a, "sl_level"), f(a, "basis"), &filters(a),
            a.get("sl_watch").and_then(Value::as_bool).unwrap_or(false))),
        "split_tp_qty" => {
            let legs: Vec<TpLegIn> = serde_json::from_value(a["legs"].clone()).unwrap();
            val(split_tp_qty(f(a, "qty"), &legs, &filters(a), s(a, "position_side")))
        }
        "armed_suite" => {
            let arm = Armed::from_value(&a["armed"]);
            let side = s(a, "position_side");
            let pos: Vec<Position> = serde_json::from_value(a["positions"].clone()).unwrap();
            let (lb, t) = (pair(a.get("last_bar")), pair(a.get("touch")));
            Ok(json!({
                "action": bracket_action(&arm, side, &pos, a["account_ok"].as_bool().unwrap(), f(a, "account_ts"), lb, t),
                "line_hit": line_hit(side, &arm, t), "sl_legs_hit": sl_legs_hit(side, &arm, t, lb),
                "sl_hit": bracket_sl_hit(side, lb, &arm), "sl_up": sl_up_of(side, &arm),
            }))
        }
        "bracket_merge" => {
            let m = |k: &str| a[k].as_object().unwrap().clone();
            Ok(Value::Object(bracket_merge(&m("current"), &m("mine"), &m("seen_at"))))
        }
        "bracket_rekey" => Ok(Value::Object(bracket_rekey(a["state"].as_object().unwrap()))),
        "resolve_exit_position" => {
            let pos: Vec<Position> = serde_json::from_value(a["positions"].clone()).unwrap();
            let cands: Vec<&str> = a["candidates"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
            Ok(match resolve_exit_position(&pos, s(a, "position_side"), &cands) {
                None => json!([null, null, []]),
                Some((p, sym, left)) => json!([p, sym, left]),
            })
        }
        other => panic!("모르는 함수 {other}"),
    }
}

fn diff(py: &Value, rs: &Value, path: &str, out: &mut Vec<String>, exact: &mut (usize, usize)) {
    match (py, rs) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            exact.1 += 1;
            if a.to_bits() == b.to_bits() {
                exact.0 += 1;
            } else if a == 0.0 && b == 0.0 {
                out.push(format!("{path}: 0 의 부호 py {a:?} rs {b:?}")); // 화면·repr 에 "-0.0" 으로 드러난다
            } else if (a - b).abs() > 1e-12 * a.abs().max(b.abs()) {
                out.push(format!("{path}: py {a:?} rs {b:?}"));
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            // 키 «집합» 비교(serde_json preserve_order 를 켠 뒤에도 전과 같은 뜻 -- 전엔 BTreeMap 이라 양쪽이 정렬돼 있었다)
            let (mut ka, mut kb): (Vec<_>, Vec<_>) = (a.keys().collect(), b.keys().collect());
            (ka.sort(), kb.sort());
            if ka != kb {
                out.push(format!("{path}: 키 py {ka:?} rs {kb:?}"));
                return;
            }
            for (k, v) in a {
                diff(v, &b[k], &format!("{path}.{k}"), out, exact);
            }
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                diff(x, y, &format!("{path}[{i}]"), out, exact);
            }
        }
        _ if py == rs => {}
        _ => out.push(format!("{path}: py {py} rs {rs}")),
    }
}

#[test]
fn plan_matches_python_fixture() {
    assert!(!exec_enabled(), "정답지는 DASHBOARD_MANUAL_EXEC_ENABLED 없이 만들었다(dry_run 칸)");
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/plan_fixture.json")).unwrap();
    let doc: Value = serde_json::from_str(&text).unwrap();
    let cases = doc["cases"].as_array().unwrap();
    let (mut bad, mut per, mut exact, mut reprs, mut errs) = (vec![], std::collections::BTreeMap::<String, usize>::new(), (0, 0), 0, 0);
    for (i, c) in cases.iter().enumerate() {
        let fname = c["fn"].as_str().unwrap();
        *per.entry(fname.to_string()).or_default() += 1;
        let got = run(fname, &c["args"]);
        let mut d = vec![];
        match (c.get("err"), &got) {
            (Some(e), Err(g)) => {
                errs += 1;
                if (e[0].as_str().unwrap(), e[1].as_str().unwrap()) != (g.py_type(), g.py_msg().as_str()) {
                    d.push(format!("거절 사유 py {e} rs {} {}", g.py_type(), g.py_msg()));
                }
            }
            (Some(e), Ok(v)) => d.push(format!("파이썬은 거절 {e}, Rust 는 {v}")),
            (None, Err(g)) => d.push(format!("Rust 만 거절 {g:?}")),
            (None, Ok(v)) => {
                diff(&c["out"], v, "", &mut d, &mut exact);
                // 거래소로 나가는 값: 파이썬 repr 문자열까지 같아야 한다
                match c.get("repr") {
                    Some(Value::Object(r)) => for (k, want) in r {
                        let have = pyfmt::repr(v[k].as_f64().unwrap());
                        reprs += 1;
                        if have != want.as_str().unwrap() {
                            d.push(format!("{k} 문자열 py {want} rs {have}"));
                        }
                    },
                    Some(Value::Array(r)) => for (j, want) in r.iter().enumerate() {
                        let have = pyfmt::repr(v[j].as_f64().unwrap());
                        reprs += 1;
                        if have != want.as_str().unwrap() {
                            d.push(format!("[{j}] 문자열 py {want} rs {have}"));
                        }
                    },
                    _ => {}
                }
            }
        }
        if !d.is_empty() {
            bad.push(format!("#{i} {fname} {}: {}", c["args"], d.join(" | ")));
        }
    }
    println!("정답지 python {} · 사례 {} · 함수별 {per:?}", doc["python"], cases.len());
    println!("실수 칸 {} 중 비트까지 같음 {} · 주문 문자열 {} · 거절 경로 {}", exact.1, exact.0, reprs, errs);
    for b in bad.iter().take(25) {
        println!("불일치 {b}");
    }
    assert!(bad.is_empty(), "불일치 {}건 / {}건", bad.len(), cases.len());
}

/// 파이썬 문자열 repr 을 쓰는 거절 사유 하나와 Map 순서 없는 비교를 따로 고정(정답지 밖 자리).
#[test]
fn reject_message_shapes() {
    let e = build_entry_plan(&EntryInput { side: "BAD", best_bid: 1.0, best_ask: 1.0, ..Default::default() }).unwrap_err();
    assert_eq!(e.py_msg(), "side must be LONG/SHORT, got 'BAD'");
    let m: Map<String, Value> = Map::new();
    assert!(bracket_rekey(&m).is_empty());
}
