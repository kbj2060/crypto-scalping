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

/// 🔧의도한 차이(SHADOW.md §11 #13): 시각이 바뀌어 지난 시 파일을 압축하는 중(또는 끝난 뒤)에 그 시의 늦은 프레임이 오면 -- 원본은 그 파일을 다시 열어
/// 이어 쓰고 압축 스레드가 원본을 지워 **그 줄이 사라지거나**(경쟁), 새 .jsonl 을 만들어 읽는 쪽(.jsonl 먼저)이 그 시의 .gz 를 못 본다.
/// Rust 는 그 시 압축을 기다리고 .gz 뒤에 gzip 멤버로 붙인다 -- 줄이 하나도 안 사라지고, .gz 하나로 이어 읽힌다(다음 시 파일은 그대로 이어 쓴다).
#[test]
fn depth_late_frame_during_compress_is_kept() {
    use serde_json::json;
    let dir = std::env::temp_dir().join(format!("rtcollect_depth_late_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let h23 = 1_763_683_200_000i64 - 3_600_000; // 2025-11-20T23 시작
    let fr = |t: i64, u: i64| json!({"e": "depthUpdate", "E": t, "T": t, "s": "ETHUSDT", "U": u, "u": u, "pu": u - 1, "b": [], "a": []}).to_string();
    let mut ev = vec![json!([1.0, "o", ""])];
    let mut u = 100;
    // 23시 큰 파일(압축이 잠깐 걸리게) → 00시 첫 프레임(23시 압축 시작) → 곧바로 23시 늦은 프레임 셋 → 00시 이어서
    for k in 0..20_000 {
        u += 1;
        ev.push(json!([1.0, "t", fr(h23 + 3_000_000 + k, u)]));
    }
    for (t, n) in [(h23 + 3_600_000 + 10, 2), (h23 + 3_599_000, 3), (h23 + 3_600_000 + 500, 4)] {
        for _ in 0..n {
            u += 1;
            ev.push(json!([1.0, "t", fr(t, u)]));
        }
    }
    rtcollect::depth::replay(&ev, "ethusdt", &dir, &[]).unwrap();
    let d = dir.join("ETHUSDT");
    let mut names: Vec<String> = std::fs::read_dir(&d).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().to_string()).collect();
    names.sort();
    assert_eq!(names, ["2025-11-20T23.jsonl.gz", "2025-11-21T00.jsonl"], "지난 시는 .gz 하나뿐(새 .jsonl 을 만들지 않는다)");
    let mut s23 = String::new();
    std::io::Read::read_to_string(&mut flate2::read::MultiGzDecoder::new(std::fs::File::open(d.join("2025-11-20T23.jsonl.gz")).unwrap()), &mut s23).unwrap();
    let l23: Vec<&str> = s23.lines().collect();
    assert_eq!(l23.len(), 1 + 20_000 + 3, "스냅숏 줄 + 23시 프레임 + 늦은 셋 -- 하나도 안 사라진다");
    assert!(l23[l23.len() - 3..].iter().all(|l| l.contains(&format!("\"T\":{}", h23 + 3_599_000))), "늦은 셋이 .gz 끝에 붙는다");
    let s00 = std::fs::read_to_string(d.join("2025-11-21T00.jsonl")).unwrap();
    assert_eq!(s00.lines().count(), 1 + 2 + 4, "00시: 스냅숏 하나 + 앞 둘 + 뒤 넷(다시 열 때 스냅숏을 또 쓰지 않는다)");
    let _ = std::fs::remove_dir_all(&dir);
}
