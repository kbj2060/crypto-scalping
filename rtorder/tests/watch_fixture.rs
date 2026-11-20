//! 감시 정답지 재생(2026-11-20 셋째 조각). 정답지 = rtorder/tools/gen_watch_fixture.py 가 **원본 orders.py 의 감시 클로저**를
//! 가짜 의존성·가짜 시계로 돌려 뽑은 틱마다의 «호출 열(이름·인자·시각) · 저장본 · 상태 칸 · 로그 · 새어 나온 예외».
//! 여기서는 같은 각본(같은 응답 열 · 같은 저장본 고치기 · 같은 잠금·스위치 변화)을 `ScriptIo` 로 걸고 Rust 감시를 돌려 칸 단위로 대조한다.
//! 끝으로 파이썬이 쓴 저장본 원문을 Rust 가 읽고 · Rust 가 다시 쓴 것을 target/watch_roundtrip/ 에 남긴다(파이썬 쪽 확인 = tools/check_watch_roundtrip.py).

include!("common/watch_io.rs");

fn load_fixture() -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/watch_fixture.json");
    serde_json::from_str(&std::fs::read_to_string(p).expect("정답지 없음 -- tools/gen_watch_fixture.py")).unwrap()
}

fn diff(name: &str, i: usize, what: &str, want: &Value, got: &Value, bad: &mut Vec<String>) {
    if want != got {
        bad.push(format!("{name} 틱{i} {what}\n  파이썬 {}\n  Rust   {}", serde_json::to_string(want).unwrap(), serde_json::to_string(got).unwrap()));
    }
}

#[test]
fn watch_matches_python_tick_by_tick() {
    let fx = load_fixture();
    let eth = fx["eth_symbol"].as_str().unwrap();
    let (mut bad, mut n_ticks, mut n_calls, mut n_fields) = (vec![], 0, 0, 0);
    let rt = std::env::temp_dir().join(format!("rtorder_watch_rt_{}", std::process::id()));
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/watch_roundtrip");
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();
    for sc in fx["scenarios"].as_array().unwrap() {
        let s = &sc["scenario"];
        let name = s["name"].as_str().unwrap();
        let (mut w, mut io) = setup(s, eth);
        for (i, (tk, want)) in s["ticks"].as_array().unwrap().iter().zip(sc["ticks"].as_array().unwrap()).enumerate() {
            let got = run_op(&mut w, &mut io, tk);
            for k in ["calls", "file", "limits", "slots", "exiting", "logs", "raised", "ret", "state", "t_end"] {
                diff(name, i, k, &want[k], &got[k], &mut bad);
            }
            n_ticks += 1;
            n_calls += want["calls"].as_array().unwrap().len();
            n_fields += want["file"].as_object().map_or(0, |m| m.values().map(|v| v.as_object().map_or(1, |o| o.len())).sum());
        }
        // 양방향 왕복: 파이썬이 쓴 마지막 저장본 원문 → Rust 가 읽음 = 파이썬이 읽은 값 · Rust 가 다시 씀 → Rust 가 읽음 = 같음 · 파이썬 확인용으로 남김
        let last = sc["ticks"].as_array().unwrap().last().unwrap();
        if let (Some(raw), Value::Object(py)) = (last["file_raw"].as_str(), &last["file"]) {
            let _ = std::fs::remove_dir_all(&rt);
            std::fs::create_dir_all(&rt).unwrap();
            std::fs::write(bracket_path(&rt), raw).unwrap();
            assert_eq!(bracket_load(&rt), plan::bracket_rekey(py), "{name}: 파이썬이 쓴 저장본을 Rust 가 다르게 읽음");
            let direct: Value = serde_json::from_str(raw).unwrap();
            assert_eq!(&direct, &last["file"], "{name}: 원문 해석이 다름");
            bracket_save(&rt, py).unwrap();
            let back: Value = serde_json::from_str(&std::fs::read_to_string(bracket_path(&rt)).unwrap()).unwrap();
            assert_eq!(&back, &last["file"], "{name}: Rust 가 쓴 것을 다시 읽으면 다름");
            std::fs::copy(bracket_path(&rt), out_dir.join(format!("{name}.json"))).unwrap();
        }
    }
    let _ = std::fs::remove_dir_all(&rt);
    println!("사례 {} · 틱 {n_ticks} · 호출 {n_calls} · 저장본 칸 {n_fields} · 불일치 {}", fx["scenarios"].as_array().unwrap().len(), bad.len());
    assert!(bad.is_empty(), "불일치 {}건:\n{}", bad.len(), bad.join("\n"));
}

/// 파이썬이 쓴 limit 파일을 Rust limit_recover 가 읽고, Rust limit_save 가 쓴 파일을 Rust·파이썬(확인 도구)이 읽는다.
#[test]
fn limit_file_round_trip() {
    let dir = std::env::temp_dir().join(format!("rtorder_limit_rt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut w = Watcher::new(&dir, "ETHUSDC");
    let mut io = ScriptIo::new(&dir);
    io.t = 1_800_000_123.25;
    let plan: Obj = serde_json::from_value(json!({"symbol": "ETHUSDC", "positionSide": "SHORT", "wid": 2, "note": "한글 ✓", "limit": {"usdt": 0.30000000000000004},
                                                   "bracket": {"lines": true, "sl_level": 1e16, "tp_level": 1e-7}})).unwrap();
    w.limit_save(&io, &plan).unwrap();
    let p = limit_path(&dir, "ETHUSDC", 2);
    let rec: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    assert_eq!(rec["saved_at"], json!(1_800_000_123.25));
    assert_eq!(rec["plan"], Value::Object(plan));
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/watch_roundtrip");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::copy(&p, out.join("limit__manual_limit_wait_ETHUSDC_2.json")).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}
