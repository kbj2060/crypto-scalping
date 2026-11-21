//! np_sum 일치: tools/gen_np_sum_fixture.py 가 서버 판 numpy 로 뜬 경계 길이(1 ~ 200,001 · 8192 앞뒤) 배열 합과 비트 대조.
#[test]
fn np_sum_matches_numpy() {
    let g: serde_json::Value = serde_json::from_str(include_str!("fixtures/np_sum_golden.json")).unwrap();
    let mut n_ok = 0;
    for (n, h) in g["sums"].as_object().unwrap() {
        let n: i64 = n.parse().unwrap();
        let x: Vec<f64> = (0..n).map(|i| ((i * 2654435761) % 1000003) as f64 / 997.0 - 500.3).collect();
        let got = rtcollect::ob5m::np_sum(&x);
        let want = parse_hex(h.as_str().unwrap());
        assert_eq!(got.to_bits(), want.to_bits(), "n={n}: {got:e} vs numpy {want:e}");
        n_ok += 1;
    }
    eprintln!("np.sum 비트 일치: 길이 {n_ok}개(8192 앞뒤 포함 · numpy {})", g["numpy"]);
}

/// 파이썬 float.hex() 글자("0x1.8p+3" · "-0x0.0p+0") → f64.
fn parse_hex(s: &str) -> f64 {
    let (neg, s) = s.strip_prefix('-').map_or((false, s), |r| (true, r));
    let s = s.strip_prefix("0x").unwrap();
    let (m, e) = s.split_once('p').unwrap();
    let (ip, fp) = m.split_once('.').unwrap_or((m, ""));
    let mut v = u64::from_str_radix(ip, 16).unwrap() as f64;
    let mut scale = 1.0 / 16.0;
    for c in fp.chars() {
        v += c.to_digit(16).unwrap() as f64 * scale;
        scale /= 16.0;
    }
    let r = v * 2f64.powi(e.parse().unwrap());
    if neg { -r } else { r }
}
