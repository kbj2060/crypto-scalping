//! POST /api/push/test -- 원본 웹 푸시 모듈을 부르는 갈래 셋(가짜 파이썬 · 네트워크 없음).
use std::path::PathBuf;
use std::sync::Arc;

fn req() -> rtserve::Req {
    rtserve::Req { method: "POST".into(), path: "/api/push/test".into(), query: vec![], headers: vec![], raw_query: String::new(), body: vec![], peer: "127.0.0.1".into() }
}

fn fake_py(dir: &PathBuf, body: &str) -> String {
    let p = dir.join(format!("py_{}", body.len()));
    std::fs::write(&p, format!("#!/bin/sh\n[ \"$2\" = test ] || exit 9\n{body}\n")).unwrap();
    std::process::Command::new("chmod").arg("+x").arg(&p).status().unwrap();
    p.to_string_lossy().into_owned()
}

#[test]
fn push_test_branches() {
    let dir = std::env::temp_dir().join(format!("rtserve_push_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut cfg = rtserve::Config::new(dir.clone(), vec!["eth".into()], Arc::new(|| 1.0e9));

    // 열쇠 없음 = 원본처럼 503 글자(파이썬을 부르지 않는다)
    std::env::set_var("RTSERVE_PY", fake_py(&dir, "exit 1"));
    let r = rtserve::handle(&cfg, &req());
    assert_eq!((r.status, r.body.as_deref()), (503, Some(b"VAPID_PRIVATE_KEY not configured".as_slice())));

    cfg.vapid_private_set = true;
    // 정상 = 모듈이 찍은 결과 JSON
    std::env::set_var("RTSERVE_PY", fake_py(&dir, r#"echo '{"sent": 2, "pruned": 1, "failed": 0}'"#));
    let r = rtserve::handle(&cfg, &req());
    assert_eq!(r.status, 200);
    let v: serde_json::Value = serde_json::from_slice(r.body.as_deref().unwrap()).unwrap();
    assert_eq!(v, serde_json::json!({"sent": 2, "pruned": 1, "failed": 0}));

    // 모듈이 환경 없음(exit 3) = 503 · 그 밖의 실패 = 500
    std::env::set_var("RTSERVE_PY", fake_py(&dir, "exit 3"));
    assert_eq!(rtserve::handle(&cfg, &req()).status, 503);
    std::env::set_var("RTSERVE_PY", fake_py(&dir, "echo not-json"));
    assert_eq!(rtserve::handle(&cfg, &req()).status, 500);

    // 읽기 전용(병행) = 보내지 않음
    cfg.readonly = true;
    std::env::set_var("RTSERVE_PY", fake_py(&dir, r#"echo '{"sent": 9}'"#));
    assert_eq!(rtserve::handle(&cfg, &req()).status, 503);
    let _ = std::fs::remove_dir_all(&dir);
}
