//! 호가 래스터(.f32) 파이썬 일치 시험. 정답지 = tools/gen_raster_fixture.py 가 원본 `_depth_loop`·북 병합·`_row`·`_RasterWriter` 를 가짜 websockets·
//! 가짜 스냅숏·가짜 시계로 돌려 만든 시각 파일의 크기·crc32·헤더·쓴 행마다 crc32. Rust 도 같은 사건 열로 파일을 만들어 비교한다(바이트 전체 = 파일 crc32,
//! 어긋나면 어느 행인지 행 crc 로 · 그 행의 mid·bin_lo 까지 찍는다).
use std::collections::BTreeMap;

fn crc32(b: &[u8]) -> u32 {
    let mut c = !0u32;
    for &x in b {
        c ^= x as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 };
        }
    }
    !c
}

#[test]
fn raster_files_match_python_collector() {
    let inp: serde_json::Value = serde_json::from_str(include_str!("fixtures/raster_input.json")).unwrap();
    let gold: serde_json::Value = serde_json::from_str(include_str!("fixtures/raster_golden.json")).unwrap();
    let i = &inp["input"];
    let dir = std::env::temp_dir().join(format!("rtcollect_raster_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    for run in i["runs"].as_array().unwrap() {
        rtcollect::raster::replay(run["t0"].as_f64().unwrap(), run["events"].as_array().unwrap(), i["symbol"].as_str().unwrap(), &dir,
                                  run["snaps"].as_array().unwrap()).unwrap();
    }
    let got: BTreeMap<String, Vec<u8>> = std::fs::read_dir(dir.join("ETHUSDT")).unwrap()
        .map(|e| { let p = e.unwrap().path(); (p.file_name().unwrap().to_string_lossy().to_string(), std::fs::read(&p).unwrap()) }).collect();
    let want = gold.as_object().unwrap();
    assert_eq!(got.keys().collect::<Vec<_>>(), want.keys().collect::<Vec<_>>(), "파일 이름");
    let (mut n_rows, mut n_valid) = (0, 0);
    for (name, b) in &got {
        let w = &want[name];
        assert_eq!(b.len() as u64, w["size"].as_u64().unwrap(), "{name} 크기");
        let hex: String = b[..32].iter().map(|x| format!("{x:02x}")).collect();
        assert_eq!(hex, w["header"].as_str().unwrap(), "{name} 헤더");
        let rows = (b.len() - 32) / 976;
        let real: Vec<(usize, u32)> = (0..rows).filter(|k| b[32 + k * 976..32 + k * 976 + 8] != [0u8; 8]).map(|k| (k, crc32(&b[32 + k * 976..32 + (k + 1) * 976]))).collect();
        let wr: Vec<(usize, u32)> = w["rows"].as_array().unwrap().iter().map(|r| (r[0].as_u64().unwrap() as usize, r[1].as_u64().unwrap() as u32)).collect();
        for (g, x) in real.iter().zip(&wr) {
            let r = &b[32 + g.0 * 976..];
            let (lo, mid) = (i32::from_le_bytes(r[8..12].try_into().unwrap()), f32::from_le_bytes(r[12..16].try_into().unwrap()));
            assert_eq!(g, x, "{name} 행 {} (bin_lo {lo} · mid {mid})", g.0);
            n_valid += mid.is_finite() as usize;
        }
        assert_eq!(real.len(), wr.len(), "{name} 쓴 행 수");
        assert_eq!(crc32(b) as u64, w["crc32"].as_u64().unwrap(), "{name} 파일 바이트 전체");
        n_rows += real.len();
    }
    eprintln!("raster 일치: 파일 {} · 쓴 행 {n_rows}(유효 {n_valid}) · 바이트 {}", got.len(), got.values().map(|b| b.len()).sum::<usize>());
    let _ = std::fs::remove_dir_all(&dir);
}
