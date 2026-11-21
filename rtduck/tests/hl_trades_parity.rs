//! HL 주소단위 체결(live_hyperliquid_trade_collector_20260916.py) 파이썬 일치 시험. 정답지 = tools/gen_hl_trades_fixture.py 가 원본 run() 을
//! 가짜 websockets·가짜 시계로 돌려 만든 날짜 파일들(파이썬 duckdb 덤프 · 압축된 3일 전 파일은 푼 바이트) + 보낸 글자.
use serde_json::Value;

#[test]
fn hl_trades_files_match_python_collector() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/hl_trades_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/hl_trades_golden.json")).unwrap();
    let dir = std::env::temp_dir().join(format!("rtduck_hl_trades_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let sub = dir.join("ETH_BTC");
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::write(sub.join("2025-11-18.duckdb"), b"3-days-old placeholder\n".repeat(100)).unwrap(); // 생성기 seed_old 와 같다
    let sent = rtduck::hl_trades::replay(&inp, &dir);
    let want: Vec<&str> = gold["sent"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect();
    assert_eq!(sent, want, "보낸 글자");
    let files = gold["files"].as_object().unwrap();
    let mut names: Vec<String> = std::fs::read_dir(&sub).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().to_string()).collect();
    names.sort();
    assert_eq!(names, files.keys().cloned().collect::<Vec<_>>(), "파일 이름(3일 전 파일은 .gz 로)");
    let mut n_rows = 0;
    for (name, w) in files {
        let p = sub.join(name);
        if name.ends_with(".gz") {
            let v = std::process::Command::new("gzip").arg("-dc").arg(&p).output().unwrap().stdout; // 시험 의존성 없이(Cargo.toml 주석)
            assert_eq!(v.iter().map(|x| format!("{x:02x}")).collect::<String>(), w.as_str().unwrap(), "{name} 푼 바이트");
            continue;
        }
        let got = rtduck::duckdump::dump(&p).unwrap();
        for (t, wt) in w.as_object().unwrap() {
            assert_eq!(got[t]["cols"], wt["cols"], "{name} {t} 열");
            let (g, ww) = (got[t]["rows"].as_array().unwrap(), wt["rows"].as_array().unwrap());
            assert_eq!(g.len(), ww.len(), "{name} {t} 행 수");
            for (k, (a, b)) in g.iter().zip(ww).enumerate() {
                assert_eq!(a, b, "{name} {t} 행 {k}");
            }
            n_rows += g.len();
        }
    }
    eprintln!("hl_trades 일치: 파일 {} · 행 {n_rows} · 보낸 글자 {}", files.len(), sent.len());
    let _ = std::fs::remove_dir_all(&dir);
}
