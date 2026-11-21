//! 15분·1시간 방향 65열판(SOL·BTC·XRP) «REST 꼴 5분봉 → 피쳐 → LightGBM 시드 앙상블 → payload» 를 원본 payload_from_features 와 대조(숫자 비트).
//! 🔴표본 밖 분위 씨앗은 개발기에 ETH 것만 있어 세 코인 모두 그것을 대역으로 쓴다(식 대조) -- tools/gen_dirshort_fixture.py 머리말.
//!     python rtmodel/tools/gen_dirshort_fixture.py && cargo test -j 2 --release --test dirshort -- --nocapture
use rtmodel::dir4h::{build_features, frames_for, Frame};
use rtmodel::dirshort::*;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

fn fb(v: &Value) -> f64 {
    f64::from_bits(v.as_u64().unwrap())
}
fn col(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(fb).collect()
}
const MCOLS: [&str; 5] = ["sum_open_interest", "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"];

fn same(g: &Value, w: &Value, path: &str, bad: &mut Vec<String>) {
    match w {
        Value::Object(m) if m.len() == 1 && m.contains_key("b") => {
            let (a, b) = (g.as_f64().unwrap_or(f64::NAN), fb(&m["b"]));
            if a.to_bits() != b.to_bits() {
                bad.push(format!("{path}: Rust {a:?} 대 파이썬 {b:?}"));
            }
        }
        Value::Object(m) => {
            assert_eq!(g.as_object().map(|o| o.len()), Some(m.len()), "{path}: 필드 {g}");
            for (k, v) in m {
                same(&g[k], v, &format!("{path}.{k}"), bad);
            }
        }
        Value::Array(a) => {
            assert_eq!(g.as_array().map(|x| x.len()), Some(a.len()), "{path}");
            for (i, v) in a.iter().enumerate() {
                same(&g[i], v, &format!("{path}[{i}]"), bad);
            }
        }
        _ if g != w => bad.push(format!("{path}: Rust {g} 대 파이썬 {w}")),
        _ => {}
    }
}

#[test]
fn dirshort_payload_parity() {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let fx: Value = rtmodel::read_json(&d.join("dirshort_fixture.json")).expect("정답지 없음 -- 머리말의 python 한 줄");
    let oos: Vec<(&str, Vec<f64>)> = NAMES.iter().map(|(_, nm)| (*nm, col(&fx["oos"][*nm]))).collect();
    let mut arts: HashMap<String, Art> = HashMap::new();
    let mut bad = vec![];
    for s in fx["sets"].as_array().unwrap() {
        let coin = s["coin"].as_str().unwrap();
        let art = arts.entry(coin.into()).or_insert_with(|| Art::from_json(&rtmodel::read_json::<Value>(&d.join(format!("ds_{}.json", coin.to_lowercase()))).unwrap()).unwrap());
        let fr: HashMap<String, Frame> = s["frames"].as_object().unwrap().iter().map(|(k, f)| (k.clone(), Frame {
            ts: f["ts"].as_array().unwrap().iter().map(|t| t.as_i64().unwrap()).collect(), high: col(&f["high"]), low: col(&f["low"]), close: col(&f["close"]),
            volume: col(&f["volume"]), trades: col(&f["trades"]), tbb: col(&f["taker_buy_base"]), m: MCOLS.map(|c| col(&f[c])) })).collect();
        let dc: Vec<(i64, f64)> = s["dc"][0].as_array().unwrap().iter().zip(col(&s["dc"][1])).map(|(t, v)| (t.as_i64().unwrap(), v)).collect();
        let f = build_features(&frames_for(&fr, coin), &dc).unwrap();
        same(&payload(art, &f, 13, &oos), &s["payload"], &format!("{coin}/{}", s["name"].as_str().unwrap()), &mut bad);
        let n = f.ts.len();
        for (h, seeds) in &art.models {
            let nm = NAMES.iter().find(|x| x.0 == *h).unwrap().1;
            for (k, w) in col(&s["raw"][nm]).iter().enumerate() {
                let x: Vec<f64> = rtmodel::dir4h::FEATS.iter().map(|c| f.cols[c][n - 13 + k] as f32 as f64).collect();
                let g = rtmodel::lgb::ensemble(seeds, &x);
                if g.to_bits() != w.to_bits() {
                    bad.push(format!("{coin} {nm} 행 {k}: 반올림 전 예측 Rust {g:?} 대 파이썬 {w:?}"));
                }
            }
        }
    }
    for q in fx["quant"].as_array().unwrap() {
        let (x, p, w) = (col(&q[0]), col(&q[1])[0], col(&q[2])[0]);
        if np_quantile(&x, p).to_bits() != w.to_bits() {
            bad.push(format!("np.quantile 길이 {} p {p}: Rust {:?} 대 numpy {w:?}", x.len(), np_quantile(&x, p)));
        }
    }
    println!("벌 {} · payload 다른 값 {}", fx["sets"].as_array().unwrap().len(), bad.len());
    for b in bad.iter().take(10) {
        println!("  {b}");
    }
    assert!(bad.is_empty());
}

#[test]
fn quantile_and_book() {
    // np.quantile([1,2,3,4], .2) = 1.6 · .8 = 3.4000000000000004(numpy 가 b − (b−a)(1−γ) 로 낸 값)
    assert_eq!(np_quantile(&[1.0, 2.0, 3.0, 4.0], 0.2), 1.6);
    assert_eq!(np_quantile(&[1.0, 2.0, 3.0, 4.0], 0.8).to_bits(), 3.4000000000000004f64.to_bits());
    let v = [Some(0.5), Some(1.0)];
    assert_eq!(book_status(None).1, "missing");
    assert_eq!(book_status(Some((1.0, &v))).1, "warm");
    assert_eq!(book_status(Some((0.0, &[Some(1.0), None]))).1, "nan");
    assert_eq!(book_status(Some((0.0, &v))), (Some(vec![0.5, 1.0]), "ok"));
}
