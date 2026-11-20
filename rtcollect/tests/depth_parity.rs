//! depth_diff 파이썬 일치 시험. 정답지 = tools/gen_depth_fixture.py 가 원본 `run()`·`HourFile`·`_gzip_done` 을 가짜 websockets·가짜 REST·가짜 시계로
//! 돌려 만든 시각 파일들의 **압축을 푼 내용**(+ 파일 이름·.gz 여부). Rust 도 같은 사건 열로 파일을 만들어 바이트까지 비교한다.
use std::io::Read;

#[test]
fn depth_files_match_python_collector() {
    let inp: serde_json::Value = serde_json::from_str(include_str!("fixtures/depth_input.json")).unwrap();
    let gold: serde_json::Value = serde_json::from_str(include_str!("fixtures/depth_golden.json")).unwrap();
    let i = &inp["input"];
    let dir = std::env::temp_dir().join(format!("rtcollect_depth_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    rtcollect::depth::replay(i["events"].as_array().unwrap(), i["symbol"].as_str().unwrap(), &dir, i["snaps"].as_array().unwrap()).unwrap();
    let mut got: Vec<(String, String)> = std::fs::read_dir(dir.join("ETHUSDT")).unwrap().map(|e| {
        let p = e.unwrap().path();
        let raw = std::fs::read(&p).unwrap();
        let text = if p.extension().is_some_and(|x| x == "gz") {
            let mut s = String::new();
            flate2::read::GzDecoder::new(&raw[..]).read_to_string(&mut s).unwrap(); // 파이썬 gzip 으로도 풀린다(--check-readers)
            s
        } else {
            String::from_utf8(raw).unwrap()
        };
        (p.file_name().unwrap().to_string_lossy().to_string(), text)
    }).collect();
    got.sort();
    let want: Vec<(String, String)> = gold["files"].as_array().unwrap().iter().map(|f| (f[0].as_str().unwrap().to_string(), f[1].as_str().unwrap().to_string())).collect();
    assert_eq!(got.iter().map(|g| &g.0).collect::<Vec<_>>(), want.iter().map(|w| &w.0).collect::<Vec<_>>(), "파일 이름");
    let mut n_lines = 0;
    for ((name, g), (_, w)) in got.iter().zip(&want) {
        for (k, (a, b)) in g.split('\n').zip(w.split('\n')).enumerate() {
            assert_eq!(a, b, "{name} 줄 {k}");
        }
        assert_eq!(g, w, "{name} 바이트");
        n_lines += w.matches('\n').count();
    }
    eprintln!("depth 일치: 파일 {} · 줄 {n_lines} · 바이트 {} · 스냅숏 REST {}", want.len(), want.iter().map(|w| w.1.len()).sum::<usize>(), gold["urls"].as_array().unwrap().len());
    let _ = std::fs::remove_dir_all(&dir);
}
