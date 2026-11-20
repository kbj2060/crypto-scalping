//! 박스 유지 «봉 열 → 피쳐 → 확률 → payload» 를 원본 파이썬(dashboard/box_hold.py)과 대조. 정답지는 커밋하지 않는다(약 33MB):
//!     python rtmodel/tools/export_hgb.py && python rtmodel/tools/gen_box_fixture.py
//!     cargo test -j 2 --test box_hold -- --nocapture
use rtmodel::box_hold::*;
use rtmodel::*;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// 파이썬 payload 에는 있지만 화면·LLM 요약이 안 읽어 옮기지 않은 필드.
const DROPPED: [&str; 7] = ["px", "model", "events_n", "reason", "half_frac", "calib", "p_hgb"];

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}
fn fb(v: &Value) -> f64 {
    f64::from_bits(v.as_u64().unwrap())
}

#[derive(Default)]
struct Err {
    abs: f64,
    rel: f64,
    n: usize,
    same: usize,
}
impl Err {
    fn add(&mut self, g: f64, w: f64, what: &str) {
        self.n += 1;
        if g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()) {
            self.same += 1;
            return;
        }
        assert!(g.is_finite() && w.is_finite(), "{what}: Rust {g} 대 파이썬 {w}");
        let a = (g - w).abs();
        self.abs = self.abs.max(a);
        self.rel = self.rel.max(a / w.abs().max(f64::MIN_POSITIVE));
    }
}

/// rust 와 파이썬(실수 = {"b": 비트}) 비교. raw = 반올림 전(확률 1e-9 · 나머지 상대 1e-12) · 아니면 비트까지.
fn cmp(g: &Value, w: &Value, path: &str, raw: bool, errs: &mut BTreeMap<String, Err>) {
    match w {
        Value::Object(m) if m.len() == 1 && m.contains_key("b") => {
            let (gv, wv) = (g.as_f64().unwrap_or_else(|| panic!("{path}: Rust {g} 가 실수가 아님")), fb(&m["b"]));
            let key = path.rsplit('.').next().unwrap().to_string();
            errs.entry(key.clone()).or_default().add(gv, wv, path);
            if !raw {
                assert_eq!(gv.to_bits(), wv.to_bits(), "{path}: 반올림 값 Rust {gv:?} 대 파이썬 {wv:?}");
            }
        }
        Value::Object(m) => {
            let gm = g.as_object().unwrap_or_else(|| panic!("{path}: Rust {g}"));
            for k in gm.keys() {
                assert!(m.contains_key(k), "{path}.{k}: 파이썬에 없는 필드");
            }
            for (k, wv) in m {
                match gm.get(k) {
                    Some(gv) => cmp(gv, wv, &format!("{path}.{k}"), raw, errs),
                    None => assert!(DROPPED.contains(&k.as_str()), "{path}.{k}: 옮기지 않은 필드(빼도 되는 목록에 없음)"),
                }
            }
        }
        _ => assert_eq!(g, w, "{path}"),
    }
}

#[test]
fn box_hold_parity() {
    let fx: Value = read_json(&dir().join("box_fixture.json")).expect("정답지 없음 -- 파일 머리말의 python 두 줄을 먼저 돌려라");
    let now_ms = fx["now_ms"].as_i64().unwrap();
    let iso: Qx4iIso = read_json(&dir().join("qx4i_iso.json")).unwrap();
    let bundles: BTreeMap<String, BoxBundle> =
        ["eth", "btc", "sol", "xrp", "soxl"].iter().map(|a| (a.to_string(), read_json(&dir().join(format!("box_{a}.json"))).unwrap())).collect();
    let (mut fe, mut pe) = (BTreeMap::<String, Err>::new(), BTreeMap::<String, Err>::new());
    let mut count = BTreeMap::<(String, String), (usize, usize)>::new(); // (코인, 척도) → (봉 열, ok)
    let mut cal_on = 0;
    for s in fx["sets"].as_array().unwrap() {
        let asset = s["asset"].as_str().unwrap();
        let name = format!("{asset}/{}", s["name"].as_str().unwrap());
        let b = &bundles[asset];
        let bars: Vec<Bar> = s["bars"].as_array().unwrap().iter().map(|r| {
            let r = r.as_array().unwrap();
            Bar { t_ms: r[0].as_i64().unwrap(), h: fb(&r[1]), l: fb(&r[2]), c: fb(&r[3]), v: fb(&r[4]), tb: fb(&r[5]) }
        }).collect();
        let events = s["events"].as_array().unwrap();
        // (a) 피쳐
        if bars.len() >= NEED {
            let k = &bars[bars.len().saturating_sub(NEED + 60)..];
            let got = features(k, events, &b.prof, &WIN.map(|w| w.1));
            match (&got, s["features"].as_object()) {
                (Ok(g), Some(w)) => {
                    for (i, (sc, _)) in WIN.iter().enumerate() {
                        for (c, v) in w[*sc].as_object().unwrap() {
                            fe.entry(c.clone()).or_default().add(g[i][c.as_str()], fb(v), &format!("{name} {sc} {c}"));
                        }
                    }
                }
                (Err(_), None) => assert!(s["features_err"].is_string(), "{name}"),
                _ => panic!("{name}: 피쳐 예외 불일치 Rust {:?} 대 파이썬 {}", got.as_ref().err(), s["features_err"]),
            }
        }
        // (b)(c) payload
        let pdir = match s["platt"].as_str().unwrap() {
            "none" => dir().join("platt/__없음__"),
            v => dir().join("platt").join(v),
        };
        let cal = platt_load(asset, &pdir, &b.source, now_ms);
        cal_on += cal.is_some() as usize;
        let vol = s["vol"].as_object().map(|v| Vol {
            ok: v["ok"].as_bool().unwrap(),
            asof_ms: v["asof_ms"].as_f64().unwrap(),
            log_rv: fb(&v["log_rv"]),
            log_rv_iv: v["log_rv_iv"].as_u64().map(f64::from_bits),
        });
        let qx = vol.as_ref().map(|v| (&iso, v));
        cmp(&payload_unrounded(&bars, events, b, asset, cal.as_ref(), qx), &s["payload_raw"], &name, true, &mut pe);
        let got = payload(&bars, events, b, asset, cal.as_ref(), qx);
        cmp(&got, &s["payload"], &name, false, &mut BTreeMap::new());
        for (sc, _) in WIN {
            let e = count.entry((asset.into(), sc.into())).or_default();
            e.0 += 1;
            e.1 += got["scales"][sc]["ok"].as_bool().unwrap_or(false) as usize;
        }
    }
    println!("코인·척도별 봉 열 수(ok 척도):");
    for ((a, sc), (n, ok)) in &count {
        println!("  {a:<5}{sc:<4} {n:>3} ({ok})");
    }
    println!("Platt90 이 켜진 봉 열 {cal_on}");
    println!("(a) 피쳐 열별 · 최대 |오차| · 최대 상대 · 비트 같음");
    let mut worst_rel = 0.0f64;
    for (c, e) in &fe {
        println!("  {c:<15} {:.2e} {:.2e} {}/{}", e.abs, e.rel, e.same, e.n);
        worst_rel = worst_rel.max(e.abs.min(e.rel));
    }
    println!("(b) 반올림 전 payload 숫자");
    for (c, e) in &pe {
        println!("  {c:<15} {:.2e} {:.2e} {}/{}", e.abs, e.rel, e.same, e.n);
    }
    println!("(c) 반올림한 payload: 봉 열 {} 개 숫자·문자열·불리언 전부 같음", fx["sets"].as_array().unwrap().len());
    assert!(worst_rel <= 1e-12, "피쳐 오차 {worst_rel:e} > 1e-12");
    assert!(pe.get("p").map_or(0.0, |e| e.abs) <= 1e-9, "확률 오차 > 1e-9");
    for (k, e) in &pe {
        assert!(e.abs.min(e.rel) <= 1e-12, "{k}: {:e}", e.rel);
    }
    // 파이썬 round(x, nd) 규칙
    for r in fx["round"].as_array().unwrap() {
        let (x, nd, w) = (fb(&r[0]), r[1].as_u64().unwrap() as usize, fb(&r[2]));
        assert_eq!(py_round(x, nd).to_bits(), w.to_bits(), "round({x:?}, {nd})");
    }
}

#[test]
fn iso_parse() {
    let ms = |s: &str| parse_iso_ns(s).map(|x| (x.0 / 1_000_000, x.1));
    assert_eq!(ms("2026-10-15T12:30:00+00:00"), Some((1_792_067_400_000, true)));
    assert_eq!(ms("2026-10-15T08:30:00-04:00"), Some((1_792_067_400_000, true)));
    assert_eq!(ms("2026-10-15 12:30:00.5Z"), Some((1_792_067_400_500, true)));
    assert_eq!(ms("2026-10-15"), Some((1_792_022_400_000, false)));
    assert_eq!(ms("bad"), None);
}
