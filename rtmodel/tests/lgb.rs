//! LightGBM 추론(lgb.rs)을 설치된 lightgbm 4.7.0 predict 와 비트 대조 -- 모델 = 개발기 65열판 eth·btc·xrp(지평 3·12봉 × 시드 5 × 표적 4).
//! 정답지는 커밋하지 않는다(약 140MB):  python rtmodel/tools/gen_lgb_fixture.py && cargo test -j 2 --release --test lgb -- --nocapture
use rtmodel::lgb::*;
use serde_json::Value;
use std::path::PathBuf;

fn fb(v: &Value) -> f64 {
    f64::from_bits(v.as_u64().unwrap())
}

#[test]
fn lgb_parity() {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let (mut rows_tot, mut cells, mut same, mut worst, mut ens_same, mut ens_n, mut ens_worst) = (0, 0usize, 0usize, 0.0f64, 0usize, 0usize, 0.0f64);
    for coin in ["eth", "btc", "xrp"] {
        let fx: Value = rtmodel::read_json(&d.join(format!("lgb_{coin}.json"))).expect("정답지 없음 -- 머리말의 python 한 줄을 먼저");
        println!("{coin}: 정답지 lightgbm {}", fx["lightgbm"]);
        let nf = fx["nf"].as_u64().unwrap() as usize;
        let flat: Vec<f64> = fx["rows"].as_array().unwrap().iter().map(fb).collect();
        let rows: Vec<&[f64]> = flat.chunks(nf).collect();
        rows_tot += rows.len();
        for (h, seeds) in fx["models"].as_object().unwrap() {
            let seeds = load_seeds(seeds).unwrap();
            for (si, ms) in seeds.iter().enumerate() {
                for (ti, (b, _)) in ms.iter().enumerate() {
                    for (r, w) in rows.iter().zip(fx["each"][h][si][ti].as_array().unwrap()) {
                        let (g, w) = (b.predict(r), fb(w));
                        cells += 1;
                        if g.to_bits() == w.to_bits() { same += 1 } else { worst = worst.max((g - w).abs()) }
                    }
                }
            }
            for (r, w) in rows.iter().zip(fx["pred"][h].as_array().unwrap()) {
                let (g, w) = (ensemble(&seeds, r), fb(w));
                ens_n += 1;
                if g.to_bits() == w.to_bits() { ens_same += 1 } else { ens_worst = ens_worst.max((g - w).abs()) }
            }
        }
    }
    println!("행 {rows_tot} · 모델 하나 예측 {cells}칸 중 비트 같음 {same} (최대 |오차| {worst:.2e}) · 앙상블 {ens_n}칸 중 비트 같음 {ens_same} (최대 {ens_worst:.2e})");
    assert_eq!(same, cells, "모델 하나 예측이 다르다");
    assert_eq!(ens_same, ens_n, "앙상블 예측이 다르다(최대 {ens_worst:e})");
}
