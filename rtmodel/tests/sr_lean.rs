//! 레벨 5분 «체결·1분봉 → 사건 → 피쳐 → 확률 → 기록 줄» 을 원본 워커(scripts/live_sr_lean_20261118.py Live)와 대조.
//! 정답지는 커밋하지 않는다(약 40MB · 시험 전에 도구로 다시 만든다):
//!     python rtmodel/tools/export_hgb.py && NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR" python rtmodel/tools/gen_sr_fixture.py
//!     cargo test -j 2 --release --test sr_lean -- --nocapture
use rtmodel::sr_lean::*;
use rtmodel::*;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// 피쳐 줄 → (id, 열 → 값). 파이썬 repr 은 왕복 십진이라 비트 그대로 돌아온다(NaN·Infinity 는 serde 가 못 읽어 바꿔 읽는다).
fn feat_vals(ln: &str) -> (String, Vec<(String, f64)>) {
    let s = ln.replace("-Infinity", "\"-inf\"").replace("Infinity", "\"inf\"").replace("NaN", "\"nan\"");
    let v: serde_json::Map<String, Value> = serde_json::from_str(&s).unwrap();
    let id = v["id"].as_str().unwrap().to_string();
    let vals = v.iter().filter(|(k, _)| *k != "k" && *k != "id").map(|(k, x)| {
        (k.clone(), x.as_f64().unwrap_or_else(|| x.as_str().unwrap().parse().unwrap()))
    }).collect();
    (id, vals)
}

fn kl(rows: &Value) -> Vec<Kline> {
    rows.as_array().unwrap().iter().map(|r| Kline::from_rest(r.as_array().unwrap()).unwrap()).collect()
}

#[derive(Default)]
struct Err {
    abs: f64,
    rel: f64,
    n: usize,
    same: usize,
}

#[test]
fn sr_lean_parity() {
    let art: SrLean = read_json(&dir().join("sr_lean.json")).expect("모델 없음 -- 파일 머리말의 python 두 줄을 먼저 돌려라");
    let fx: Value = read_json(&dir().join("sr_fixture.json")).expect("정답지 없음 -- 파일 머리말의 python 두 줄을 먼저 돌려라");
    assert!(fx["npy_disable"].as_str().is_some_and(|s| s.contains("AVX512")), "정답지는 서버와 같은 계산(AVX-512 끈 numpy · tools/server_env.sh)으로 만든다");
    let (mut fe, mut pe) = (BTreeMap::<String, Err>::new(), Err::default());
    let (mut bad_lines, mut tot) = (vec![], BTreeMap::<&str, usize>::new());
    let mut grp_diff = 0;
    for set in fx["sets"].as_array().unwrap() {
        let name = set["name"].as_str().unwrap();
        let mut lv = Live::new(&art);
        let states = set["state"].as_array().unwrap();
        let mut si = 0;
        for op in set["ops"].as_array().unwrap() {
            let o = op.as_array().unwrap();
            let i = |k: usize| o[k].as_i64().unwrap();
            match o[0].as_str().unwrap() {
                "trade" => lv.trade(i(1), o[2].as_f64().unwrap(), o[3].as_f64().unwrap(), i(4), i(5), o[6].as_bool().unwrap()),
                "init" => (lv.t0, lv.done) = (Some(i(1)), Some(i(1))),
                "klines" => lv.klines(&kl(&o[1]), i(2)),
                "recheck" => lv.recheck(&kl(&o[1])),
                "hole" => lv.hole(i(1)),
                "close" => {
                    for s in lv.done.unwrap() + 1..=i(1) {
                        lv.second(s);
                    }
                    if let Some(st) = states.get(si).filter(|st| st[0].as_i64() == Some(i(1) + 2)) {
                        let got = lv.state_json(st[0].as_i64().unwrap());
                        if got != st[1].as_str().unwrap() {
                            bad_lines.push(format!("{name} 상태\n  Rust   {got}\n  파이썬 {}", st[1]));
                        }
                        si += 1;
                    }
                }
                x => panic!("모르는 연산 {x}"),
            }
        }
        assert_eq!(si, states.len(), "{name}: 상태 줄 수");
        // (a) 사건 열 · (d) 기록 줄(글자 그대로)
        let want: Vec<&str> = set["main"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let key = |ln: &str| -> String {
            let v: Value = serde_json::from_str(&ln.replace("NaN", "null").replace("Infinity", "null")).unwrap();
            format!("{} {} {} {} {}", v["k"], v["a"], v["look"], v["kind"], v["side"])
        };
        let (gk, wk): (Vec<String>, Vec<String>) = (lv.log.iter().map(|l| key(l)).collect(), want.iter().map(|l| key(l)).collect());
        assert_eq!(gk.iter().filter(|k| !k.starts_with("\"res\"")).collect::<Vec<_>>(), wk.iter().filter(|k| !k.starts_with("\"res\"")).collect::<Vec<_>>(),
                   "{name}: 사건 열(종류·시각·레벨·쪽)이 다르다");
        assert_eq!(lv.log.len(), want.len(), "{name}: 기록 줄 수");
        for (g, w) in lv.log.iter().zip(&want) {
            *tot.entry(if w.contains("\"k\": \"ev\"") { "ev" } else if w.contains("\"k\": \"res\"") { "res" } else { "skip" }).or_default() += 1;
            if g != w {
                bad_lines.push(format!("{name}\n  Rust   {g}\n  파이썬 {w}"));
                if g.contains("\"grp\"") && g.split("\"grp\"").nth(1) != w.split("\"grp\"").nth(1) {
                    grp_diff += 1;
                }
            }
        }
        // (b) 피쳐 87칸 · (c) 반올림 전 확률
        let wf: Vec<&str> = set["feat"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(lv.feat.len(), wf.len(), "{name}: 피쳐 줄 수");
        for ((g, w), pw) in lv.feat.iter().zip(&wf).zip(set["p"].as_array().unwrap()) {
            *tot.entry("feat").or_default() += 1;
            if g != w {
                bad_lines.push(format!("{name} 피쳐\n  Rust   {g}\n  파이썬 {w}"));
            }
            let ((gi, gv), (wi, wv)) = (feat_vals(g), feat_vals(w));
            assert_eq!(gi, wi);
            for ((c, a), (_, b)) in gv.iter().zip(&wv) {
                let e = fe.entry(c.clone()).or_default();
                e.n += 1;
                if a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) {
                    e.same += 1;
                } else {
                    e.abs = e.abs.max((a - b).abs());
                    e.rel = e.rel.max((a - b).abs() / b.abs().max(f64::MIN_POSITIVE));
                }
            }
            let gm: BTreeMap<&str, f64> = gv.iter().map(|(c, v)| (c.as_str(), *v)).collect(); // serde 맵은 열 이름 순 -- 모델 열 순서로 다시
            let x: Vec<f64> = art.cols.iter().map(|c| gm[c.as_str()]).collect();
            let p = art.predict(&x).unwrap().0;
            let w = f64::from_bits(pw[0].as_u64().unwrap());
            pe.n += 1;
            if p.to_bits() == w.to_bits() {
                pe.same += 1;
            }
            pe.abs = pe.abs.max((p - w).abs());
        }
    }
    println!("벌 {} · 기록 줄 {:?}", fx["sets"].as_array().unwrap().len(), tot);
    println!("(b) 피쳐 열별 최대 |오차| · 최대 상대 · 비트 같음");
    let mut worst = 0.0f64;
    for (c, e) in &fe {
        if e.same != e.n {
            println!("  {c:<12} {:.2e} {:.2e} {}/{}", e.abs, e.rel, e.same, e.n);
        }
        worst = worst.max(e.abs.min(e.rel));
    }
    println!("  나머지 {} 열은 {}개 사건 전부 비트 같음", fe.values().filter(|e| e.same == e.n).count(), fe.values().next().map_or(0, |e| e.n));
    println!("(c) 확률 최대 |오차| {:.2e} · 비트 같음 {}/{} · 구분이 다른 사건 {grp_diff}", pe.abs, pe.same, pe.n);
    println!("(d) 글자가 다른 줄 {}", bad_lines.len());
    for b in bad_lines.iter().take(8) {
        println!("{b}");
    }
    assert!(worst <= 1e-12, "피쳐 오차 {worst:e} > 1e-12");
    assert!(pe.abs <= 1e-9, "확률 오차 > 1e-9");
    assert_eq!(grp_diff, 0, "구분(top/mid/bot)이 다른 사건");
    assert!(bad_lines.is_empty(), "기록 줄 글자 불일치 {}", bad_lines.len());
}

/// pandas rolling std 를 직접(평균 ≫ 산포 · 창 전체 같은 값 · NaN) -- 재생 대조에서 끝자리가 안 갈리는 규칙(Kahan·같은 값 연속)까지 본다.
///     python rtmodel/tools/check_rollvar.py --write
#[test]
fn roll_std_pandas() {
    let sets: Value = read_json(&dir().join("sr_roll.json")).expect("python rtmodel/tools/check_rollvar.py --write 를 먼저");
    let (mut n, mut same) = (0, 0);
    for s in sets.as_array().unwrap() {
        let b = |k: &str| s[k].as_array().unwrap().iter().map(|v| f64::from_bits(v.as_u64().unwrap())).collect::<Vec<f64>>();
        let got = roll_std(&b("x"), s["w"].as_u64().unwrap() as usize, s["minp"].as_u64().unwrap() as usize);
        for (g, w) in got.iter().zip(b("std")) {
            n += 1;
            same += (g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan())) as usize;
        }
    }
    println!("rolling std 칸 {n} 중 비트 같음 {same}");
    assert_eq!(same, n);
}

#[test]
fn repr_like_python() {
    for (x, s) in [(0.0, "0.0"), (-0.0, "-0.0"), (1.0, "1.0"), (0.1, "0.1"), (1e-4, "0.0001"), (1e-5, "1e-05"), (1.5e-7, "1.5e-07"), (123.456, "123.456"),
                   (1e16, "1e+16"), (1234567890123456.0, "1234567890123456.0"), (2.5e22, "2.5e+22"), (-3.0e-300, "-3e-300"), (f64::NAN, "NaN"),
                   (f64::INFINITY, "Infinity"), (f64::NEG_INFINITY, "-Infinity"), (0.47081371012345, "0.47081371012345"), (5e-324, "5e-324"), (1e100, "1e+100")] {
        assert_eq!(jf(x), s, "{x:e}");
    }
}
