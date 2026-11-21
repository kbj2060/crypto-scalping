//! 4시간 방향 «REST 꼴 5분봉·metrics → 피쳐 65 → HGB 회귀 → payload» 를 원본(scripts/live_eth_dir4h_20261015.py)과 대조. 정답지는 커밋하지 않는다(약 60MB):
//!     python rtmodel/tools/export_dir4h.py && NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR" python rtmodel/tools/gen_dir4h_fixture.py
//!     cargo test -j 2 --release --test dir4h -- --nocapture
use rtmodel::dir4h::*;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}
fn fb(v: &Value) -> f64 {
    f64::from_bits(v.as_u64().unwrap())
}
fn col(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(fb).collect()
}

const MCOLS: [&str; 5] = ["sum_open_interest", "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"];

fn frames(v: &Value) -> HashMap<String, Frame> {
    v.as_object().unwrap().iter().map(|(s, f)| {
        let fr = Frame { ts: f["ts"].as_array().unwrap().iter().map(|t| t.as_i64().unwrap()).collect(), high: col(&f["high"]), low: col(&f["low"]), close: col(&f["close"]),
                         volume: col(&f["volume"]), trades: col(&f["trades"]), tbb: col(&f["taker_buy_base"]), m: MCOLS.map(|c| col(&f[c])) };
        (s.clone(), fr)
    }).collect()
}

/// 파이썬 payload(실수 = {"b": 비트}) 대 Rust -- 숫자는 비트까지.
fn same(g: &Value, w: &Value, path: &str, bad: &mut Vec<String>) {
    match w {
        Value::Object(m) if m.len() == 1 && m.contains_key("b") => {
            let (a, b) = (g.as_f64().unwrap_or(f64::NAN), fb(&m["b"]));
            if a.to_bits() != b.to_bits() {
                bad.push(format!("{path}: Rust {a:?} 대 파이썬 {b:?}"));
            }
        }
        Value::Object(m) => {
            assert_eq!(g.as_object().unwrap().len(), m.len(), "{path}: 필드 수");
            for (k, v) in m {
                same(&g[k], v, &format!("{path}.{k}"), bad);
            }
        }
        Value::Array(a) => {
            assert_eq!(g.as_array().unwrap().len(), a.len(), "{path}: 길이");
            for (i, v) in a.iter().enumerate() {
                same(&g[i], v, &format!("{path}[{i}]"), bad);
            }
        }
        _ => {
            if g != w {
                bad.push(format!("{path}: Rust {g} 대 파이썬 {w}"));
            }
        }
    }
}

#[test]
fn dir4h_parity() {
    let fx: Value = rtmodel::read_json(&dir().join("dir4h_fixture.json")).expect("정답지 없음 -- 파일 머리말의 python 두 줄을 먼저");
    let arts: HashMap<String, Art> = ["sol", "xrp", "eth"].iter().map(|c| (c.to_uppercase(), Art::load(&dir().join(format!("dir4h_{c}.json"))).unwrap())).collect();
    let mut err: BTreeMap<&str, (f64, usize, usize)> = BTreeMap::new(); // 열 → (최대 상대 오차, 비트 같음, 칸)
    let (mut pmax, mut bad, mut nsets) = (0.0f64, vec![], 0);
    for s in fx["sets"].as_array().unwrap() {
        let (coin, name) = (s["coin"].as_str().unwrap(), s["name"].as_str().unwrap());
        let fr = frames(&s["frames"]);
        let dc: Vec<(i64, f64)> = s["dc"][0].as_array().unwrap().iter().zip(s["dc"][1].as_array().unwrap()).map(|(t, v)| (t.as_i64().unwrap(), fb(v))).collect();
        let got = build_features(&frames_for(&fr, coin), &dc);
        if name == "dup_ts" {
            // 같은 시각 두 줄: 원본 grid 의 sort_values(quicksort · 불안정)가 어느 줄을 남길지 정해져 있지 않다(실측: 둘 중 하나는 뒤 줄 · 하나는 앞 줄).
            //   REST 원천은 _merge·쪽 넘김이 겹친 시각을 안 만든다(소스 확인) -- 비교에서 뺀다(Rust 는 안정 정렬 = 뒤 줄).
            continue;
        }
        if s.get("err").is_some() {
            assert!(got.is_err(), "{coin}/{name}: 파이썬은 예외 -- Rust 도 Err 여야");
            continue;
        }
        let f = got.unwrap_or_else(|e| panic!("{coin}/{name}: {e}"));
        nsets += 1;
        let n = f.ts.len();
        let ts: Vec<i64> = s["ts"].as_array().unwrap().iter().map(|t| t.as_i64().unwrap()).collect();
        assert_eq!(&f.ts[n - ts.len()..], &ts[..], "{coin}/{name}: 행 시각");
        for c in FEATS {
            let w = col(&s["feats"][c]);
            let e = err.entry(c).or_default();
            for (k, b) in w.iter().enumerate() {
                let a = f.cols[c][n - w.len() + k];
                e.2 += 1;
                if a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) {
                    e.1 += 1;
                } else {
                    assert!(a.is_finite() && b.is_finite(), "{coin}/{name} {c}: Rust {a} 대 파이썬 {b}");
                    e.0 = e.0.max((a - b).abs() / b.abs().max(f64::MIN_POSITIVE));
                }
            }
        }
        let art = &arts[coin];
        for (k, w) in col(&s["pred"]).iter().enumerate() {
            pmax = pmax.max((art.predict(&f, n - 60 + k) - w).abs());
        }
        same(&payload(art, &f), &s["payload"], &format!("{coin}/{name}"), &mut bad);
    }
    println!("벌 {nsets}(+ 예외 벌) · 피쳐 열별 비트가 다른 칸이 있는 열 (최대 상대 오차 · 같음/칸):");
    let mut worst = 0.0f64;
    for (c, (r, sm, t)) in &err {
        if sm != t {
            println!("  {c:<16} {r:.2e} {sm}/{t}");
        }
        worst = worst.max(*r);
    }
    println!("  나머지 {} 열은 전부 비트 같음 · 예측 최대 |오차| {pmax:.2e} · payload 다른 값 {}", err.values().filter(|e| e.1 == e.2).count(), bad.len());
    for b in bad.iter().take(10) {
        println!("  {b}");
    }
    assert!(worst <= 1e-12, "피쳐 오차 {worst:e}");
    assert!(pmax <= 1e-9, "예측 오차 {pmax:e}");
    assert!(bad.is_empty(), "payload 불일치 {}", bad.len());
}
