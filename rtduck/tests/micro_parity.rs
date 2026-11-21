//! 미시구조(microstructure_scanner.py) 파이썬 일치 시험. 정답지 = tools/gen_micro_fixture.py(TZ=Asia/Seoul · X86_V4): 부트스트랩 행 수 · 폴링 요청 URL 열 ·
//! 10초마다 compute_signal 글자 전체(crc32·길이 -- 앞 12개·50번째마다는 글자 그대로) · microstructure_1m{,_btc} 덤프(씨앗 + 분마다 34칸).
use serde_json::Value;

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
fn microstructure_matches_python_scanner() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/micro_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/micro_golden.json")).unwrap();
    let dir = std::env::temp_dir().join(format!("rtduck_micro_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let db = dir.join("microstructure.duckdb");
    let (boot, sent, sigs) = rtduck::micro::replay(&inp, &db);
    for (s, n) in &boot {
        assert_eq!(gold["boot"][s].as_u64(), Some(*n as u64), "{s} 부트스트랩 행 수");
    }
    let want: Vec<&str> = gold["requests"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
    assert_eq!(sent, want, "요청 URL 열");
    let gs = gold["sigs"].as_array().unwrap();
    assert_eq!(sigs.len(), gs.len(), "신호 수");
    for (k, ((s, txt), g)) in sigs.iter().zip(gs).enumerate() {
        assert_eq!(s, g[0].as_str().unwrap(), "{k} 종목");
        if let Some(w) = g[3].as_str().filter(|w| !w.is_empty()) {
            assert_eq!(txt, w, "{k}번째 신호 글자({s})");
        }
        assert_eq!((crc32(txt.as_bytes()) as u64, txt.len() as u64), (g[1].as_u64().unwrap(), g[2].as_u64().unwrap()), "{k}번째 신호({s}) crc·길이:\n{txt}");
    }
    let got = rtduck::duckdump::dump(&db).unwrap();
    let mut n = 0;
    for (t, w) in gold["tables"].as_object().unwrap() {
        assert_eq!(got[t]["cols"], w["cols"], "{t} 열");
        assert_eq!(got[t]["phys_ts"], w["phys_ts"], "{t} 물리 순서");
        let (a, b) = (got[t]["rows"].as_array().unwrap(), w["rows"].as_array().unwrap());
        assert_eq!(a.len(), b.len(), "{t} 행 수");
        for (k, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x, y, "{t} 행 {k}");
        }
        n += a.len();
    }
    eprintln!("미시구조 일치: 부트스트랩 {boot:?} · 요청 {} · 신호 글자 {} · 표 {} 행 {n} × 34칸", sent.len(), sigs.len(), got.as_object().unwrap().len());
    let _ = std::fs::remove_dir_all(&dir);
}
