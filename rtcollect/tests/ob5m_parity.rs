//! ob5m(5분 호가 요약) · bookdepth(호가 깊이 10열) 파이썬 일치 시험. 정답지 = tools/gen_ob5m_fixture.py 가 원본 `main()` 둘을 합성 depthdiff 시 파일 14개
//! (스냅숏·10~15초 diff·벽 사건·끊김·.gz 와 .jsonl)와 테이프로 돌린 DB(NPY_DISABLE_CPU_FEATURES=X86_V4 = 서버와 같은 np.log).
//! ob5m = bars·marks·live 글자까지 **비트 일치**. bookdepth = dm·dp(원본 BLAS 내적)와 거기서 나온 열만 상대·절대 1e-12, 나머지 비트 일치.
use rusqlite::{types::ValueRef, Connection};
use serde_json::Value;

fn rows(db: &std::path::Path, sql: &str) -> Vec<Vec<Value>> {
    let c = Connection::open(db).unwrap();
    let mut st = c.prepare(sql).unwrap();
    let n = st.column_count();
    st.query_map([], |r| Ok((0..n).map(|j| match r.get_ref(j).unwrap() {
        ValueRef::Null => Value::Null,
        ValueRef::Integer(i) => Value::from(i),
        ValueRef::Real(f) => Value::from(f),
        ValueRef::Text(t) => Value::from(String::from_utf8_lossy(t).to_string()),
        ValueRef::Blob(_) => Value::Null,
    }).collect())).unwrap().map(Result::unwrap).collect()
}

fn same(a: &Value, b: &Value) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) if a.is_f64() || b.is_f64() => x.to_bits() == y.to_bits(),
        _ => a == b,
    }
}

#[test]
fn ob5m_and_bookdepth_match_python_workers() {
    let fix = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let gold: Value = serde_json::from_str(&std::fs::read_to_string(fix.join("ob5m_golden.json")).unwrap()).unwrap();
    let out = std::env::temp_dir().join(format!("rtcollect_ob5m_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    rtcollect::ob5m::replay_fixture(&fix, &out);
    let (ob, bd) = (out.join("ob5m.sqlite"), out.join("bookdepth.sqlite"));
    let mut n_cells = 0;
    for (k, sql) in [("ob5m_bars", "SELECT * FROM bars ORDER BY t"), ("ob5m_marks", "SELECT * FROM marks ORDER BY t, side, px"),
                     ("ob5m_live", "SELECT k, bar, marks FROM live"), ("ob5m_schema", "SELECT sql FROM sqlite_master ORDER BY name")] {
        let got = rows(&ob, sql);
        let want = gold[k].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{k} 행 수");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let w = w.as_array().map_or_else(|| vec![w.clone()], |a| a.clone());
            for (j, (x, y)) in g.iter().zip(&w).enumerate() {
                assert!(same(x, y), "{k} 행 {i} 열 {j}: rust {x} != python {y}");
                n_cells += 1;
            }
        }
    }
    let cols: Vec<String> = gold["bd_cols"].as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect();
    let loose = ["dm1", "dm2", "dm5", "dp1", "dp2", "dp5", "dimb1", "dimb2", "dimb5", "dimb1_d1h", "dimb2_d1h", "dimb5_d1h", "depth1_z", "bid_near", "ask_near"];
    let got = rows(&bd, "SELECT * FROM bars ORDER BY t");
    let want = gold["bd_bars"].as_array().unwrap();
    assert_eq!(got.len(), want.len(), "bookdepth 행 수");
    let (mut exact, mut tol, mut worst) = (0, 0, 0f64);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        for (j, (x, y)) in g.iter().zip(w.as_array().unwrap()).enumerate() {
            if same(x, y) {
                exact += 1;
                continue;
            }
            let (a, b) = (x.as_f64(), y.as_f64());
            assert!(loose.contains(&cols[j].as_str()) && a.is_some() && b.is_some(), "bookdepth 행 {i} {}: rust {x} != python {y}", cols[j]);
            let (a, b) = (a.unwrap(), b.unwrap());
            let d = (a - b).abs() / b.abs().max(1.0);
            assert!(d <= 1e-12, "bookdepth 행 {i} {}: 차 {d:e} (rust {a} · python {b})", cols[j]);
            (tol, worst) = (tol + 1, worst.max(d));
        }
    }
    assert_eq!(gold["bd_schema"].as_array().unwrap().len(), 1);
    eprintln!("ob5m 비트 일치 칸 {n_cells} · bookdepth 칸 비트 {exact} · 1e-12 안 {tol}(BLAS 내적 몫, 최대 {worst:.1e})");
    let _ = std::fs::remove_dir_all(&out);
}

/// 빈 북(2026-11-21 그림자 SNDKUSDT 패닉): 한쪽·양쪽이 diff 로 비는 사슬 · 스냅숏 한쪽이 빈 사슬 · 그 뒤 온전한 스냅숏에서 다시 -- 원본 `replay` 와
/// 상태 글자·매 초 sec_row·사건이 비트까지 같다. 원본이 IndexError 로 죽는 사슬(스냅숏 한쪽 빔)만 Rust 는 «empty_book»(의도한 차이 #16).
#[test]
fn ob5m_replay_empty_book_matches_python() {
    let fix = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ob5m_empty.json");
    let cases: Value = serde_json::from_str(&std::fs::read_to_string(fix).unwrap()).unwrap();
    let (tick, _, ks, idx_max) = rtcollect::ob5m::coin("ETHUSDT").unwrap();
    let v = |x: &Value| x.as_str().map_or_else(|| x.as_i64().unwrap().to_string(), str::to_string);
    let (mut n_sec, mut n_ev, mut n_ch) = (0, 0, 0);
    for c in cases.as_array().unwrap() {
        for (k, ch) in c["chains"].as_array().unwrap().iter().enumerate() {
            let name = format!("{} 사슬 {k}", c["name"].as_str().unwrap());
            let mut lines = ch["lines"].as_array().unwrap().iter().map(|l| l.as_str().unwrap().as_bytes().to_vec());
            let (mut secs, mut evs) = (vec![], vec![]);
            let st = rtcollect::ob5m::replay(&mut lines, tick, ks, idx_max,
                &mut |t, b, a, bb, ba| {
                    let r = rtcollect::ob5m::sec_row(t, b, a, bb, ba);
                    let f = rtcollect::py_float_repr;
                    secs.push(vec![r.0.to_string(), r.1.to_string(), r.2.to_string(), f(r.3), f(r.4), f(r.5), f(r.6), r.7.to_string(), f(r.8), r.9.to_string(), f(r.10)]);
                },
                &mut |e| evs.push(vec![e.t.to_string(), e.side.to_string(), rtcollect::py_float_repr(e.px), rtcollect::py_float_repr(e.qref),
                                       rtcollect::py_float_repr(e.q), rtcollect::py_float_repr(e.share), rtcollect::py_float_repr(e.dist_bp), e.age20.to_string()]));
            let want = ch["status"].as_str().unwrap();
            assert_eq!(st, if want == "IndexError" { "empty_book" } else { want }, "{name} 상태");
            let ws: Vec<Vec<String>> = ch["secs"].as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(v).collect()).collect();
            let we: Vec<Vec<String>> = ch["events"].as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(v).collect()).collect();
            assert_eq!(secs, ws, "{name} sec_row");
            assert_eq!(evs, we, "{name} 사건");
            (n_sec, n_ev, n_ch) = (n_sec + secs.len(), n_ev + evs.len(), n_ch + 1);
        }
    }
    assert!(n_ev >= 1 && n_sec >= 60, "사례가 비었다: 초 {n_sec} · 사건 {n_ev}");
    eprintln!("빈 북 사례 사슬 {n_ch} · sec_row {n_sec} · 사건 {n_ev} 글자 일치");
}
