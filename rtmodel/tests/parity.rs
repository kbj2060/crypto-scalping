//! 저장 모델 6개(박스 유지 5 · 레벨 5분 1) + 감싼 식(Platt90 · card30 · QX4I 등조)의 파이썬 일치 시험.
//! 자료는 커밋하지 않는다(약 37MB). 먼저 만든다:
//!     python rtmodel/tools/export_hgb.py && python rtmodel/tools/gen_pred_fixture.py
//!     cargo test -j 2 --release -- --nocapture
use rtmodel::*;
use serde::Deserialize;
use std::path::PathBuf;

const TOL: f64 = 1e-9;

#[derive(Deserialize)]
struct Platt {
    a: u64,
    b: u64,
    p: Vec<u64>,
}
#[derive(Deserialize)]
struct HgbCase {
    file: String,
    scale: String,
    rows: Vec<Vec<u64>>,
    p: Vec<u64>,
    platt: Option<Platt>,
}
#[derive(Deserialize)]
struct SrCase {
    rows: Vec<Vec<u64>>,
    p_each: Vec<Vec<Option<u64>>>,
    p: Vec<Option<u64>>,
    grp: Vec<Option<String>>,
}
#[derive(Deserialize)]
struct Card30Case {
    files: Vec<String>,
    scale: String,
    k: u64,
    base: u64,
    rows: Vec<Vec<u64>>,
    p: Vec<u64>,
}
#[derive(Deserialize)]
struct Fixture {
    hgb_cases: Vec<HgbCase>,
    sr_lean: SrCase,
    card30: Card30Case,
    qx4i: Vec<[Option<u64>; 4]>,
    platt_edge: Vec<[u64; 4]>,
}

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}
fn f(v: &[u64]) -> Vec<f64> {
    v.iter().map(|b| f64::from_bits(*b)).collect()
}
fn fx() -> Fixture {
    read_json(&dir().join("pred_fixture.json")).expect("정답지 없음 -- 파일 머리말의 python 두 줄을 먼저 돌려라")
}

/// (최대 절대 오차, 비트 같은 수). 기대 None(파이썬 예외)과 Rust None 이 어긋나면 실패.
fn compare(name: &str, got: &[Option<f64>], want: &[Option<f64>]) -> f64 {
    let (mut worst, mut same) = (0.0f64, 0);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        match (g, w) {
            (Some(g), Some(w)) => {
                let e = (g - w).abs();
                assert!(e.is_finite(), "{name} 행 {i}: {g} 대 {w}");
                worst = worst.max(e);
                same += (g.to_bits() == w.to_bits()) as usize;
            }
            (None, None) => same += 1,
            _ => panic!("{name} 행 {i}: Rust {g:?} 대 파이썬 {w:?}(None = 예외)"),
        }
    }
    println!("{name:<22} 행 {:>5} · 최대 |오차| {worst:.3e} · 비트 같음 {same}/{}", got.len(), got.len());
    assert!(worst <= TOL, "{name}: {worst:e} > {TOL:e}");
    worst
}

#[test]
fn parity() {
    let fx = fx();
    let mut boxes = std::collections::HashMap::new();
    for c in &fx.hgb_cases {
        let b: &BoxBundle = boxes.entry(c.file.clone()).or_insert_with(|| read_json(&dir().join(&c.file)).unwrap());
        let m = &b.scales[&c.scale].model;
        let got: Vec<Option<f64>> = c.rows.iter().map(|r| predict_proba(m, &f(r))).collect();
        let want: Vec<Option<f64>> = f(&c.p).into_iter().map(Some).collect();
        compare(&format!("{} {}", b.source.trim_end_matches(".joblib"), c.scale), &got, &want);
        if let Some(pl) = &c.platt {
            let (a, bb) = (f64::from_bits(pl.a), f64::from_bits(pl.b));
            let got: Vec<Option<f64>> = got.iter().map(|p| Some(platt90(p.unwrap(), a, bb))).collect();
            compare(&format!("  └ Platt90 {}", c.scale), &got, &f(&pl.p).into_iter().map(Some).collect::<Vec<_>>());
        }
    }

    let got: Vec<Option<f64>> = fx.platt_edge.iter().map(|r| Some(platt90(f64::from_bits(r[2]), f64::from_bits(r[0]), f64::from_bits(r[1])))).collect();
    compare("Platt90 자르기 경계", &got, &fx.platt_edge.iter().map(|r| Some(f64::from_bits(r[3]))).collect::<Vec<_>>());

    let sr: SrLean = read_json(&dir().join("sr_lean.json")).unwrap();
    let rows: Vec<Vec<f64>> = fx.sr_lean.rows.iter().map(|r| f(r)).collect();
    for (k, name) in ["sr_lean HGB", "sr_lean ExtraTrees", "sr_lean 로지스틱"].iter().enumerate() {
        let got: Vec<Option<f64>> = rows.iter().map(|r| predict_proba(&sr.models[k], r)).collect();
        let want: Vec<Option<f64>> = fx.sr_lean.p_each[k].iter().map(|v| v.map(f64::from_bits)).collect();
        compare(name, &got, &want);
    }
    let out: Vec<Option<(f64, &str)>> = rows.iter().map(|r| sr.predict(r)).collect();
    let want: Vec<Option<f64>> = fx.sr_lean.p.iter().map(|v| v.map(f64::from_bits)).collect();
    compare("sr_lean 평균", &out.iter().map(|o| o.map(|x| x.0)).collect::<Vec<_>>(), &want);
    for (i, (o, g)) in out.iter().zip(&fx.sr_lean.grp).enumerate() {
        assert_eq!(o.map(|x| x.1), g.as_deref(), "sr_lean grp 행 {i}");
    }

    let c = &fx.card30;
    let bundles: Vec<BoxBundle> = c.files.iter().map(|x| read_json(&dir().join(x)).unwrap()).collect();
    let ms: Vec<&Model> = bundles.iter().map(|b| &b.scales[&c.scale].model).collect();
    let (k, base) = (f64::from_bits(c.k), f64::from_bits(c.base));
    let got: Vec<Option<f64>> = c.rows.iter().map(|r| Some(card30(&ms, &f(r), k, base))).collect();
    compare("card30 식(대역 모델 3)", &got, &f(&c.p).into_iter().map(Some).collect::<Vec<_>>());

    let iso: Qx4iIso = read_json(&dir().join("qx4i_iso.json")).unwrap();
    let g = |v: Option<u64>| v.map(f64::from_bits);
    let got: Vec<Option<f64>> = fx.qx4i.iter().map(|q| Some(iso.p(g(q[0]).unwrap(), g(q[1]), g(q[2]).unwrap()))).collect();
    compare("QX4I 4h 등조", &got, &fx.qx4i.iter().map(|q| g(q[3])).collect::<Vec<_>>());
}

/// model_stale: 원본 joblib 이 JSON 보다 새것이면(재학습 뒤 내보내기 안 함) 사유 · 없으면 사유 · 아니면 None.
#[test]
fn model_stale_rule() {
    use std::time::{Duration, SystemTime};
    let d = std::env::temp_dir().join(format!("rt_stale_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let (j, src) = (d.join("m.json"), d.join("m.joblib"));
    let v: serde_json::Value = serde_json::json!({"source": "m.joblib", "dir": {"source": "m.joblib"}});
    std::fs::write(&j, "{}").unwrap();
    assert!(rtmodel::model_stale(&j, &v, &d).unwrap().contains("원본 없음"));
    std::fs::write(&src, "x").unwrap();
    let t0 = SystemTime::now() - Duration::from_secs(100);
    std::fs::File::options().write(true).open(&src).unwrap().set_modified(t0).unwrap();
    assert_eq!(rtmodel::model_stale(&j, &v, &d), None, "JSON 이 더 새것 = 신선");
    std::fs::File::options().write(true).open(&src).unwrap().set_modified(SystemTime::now() + Duration::from_secs(5)).unwrap();
    assert!(rtmodel::model_stale(&j, &v, &d).unwrap().contains("모델 낡음"));
    let _ = std::fs::remove_dir_all(&d);
}
