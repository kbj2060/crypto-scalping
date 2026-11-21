//! 원본 «전체 흐름» 정답지 재생(2026-11-21 일곱째 조각 · 그림자 재생 도구로 옮김). 정답지 = tools/gen_flow_fixture.py 가 진짜 `orders.entry_then_bracket`·`bracket_tick` +
//! 진짜 집행·브래킷 함수 + aiohttp 를 127.0.0.1 모의 거래소에 붙여 뽑은 «걸음마다 나간 요청 열 · 받은 응답 열 · 상태 칸 · 무장 파일 · 복구 파일 · 가짜 의존성 호출 · 로그 · 끝 시각».
//! 여기서는 그것을 그림자 기록(`rtorder::shadow` JSONL)으로 바꿔 `shadow::replay`(진짜 `http::Binance`·`runner`·`Watcher`·`bracket`)로 **글자 그대로(exact)** 대조한다.
//! 2026-11-21 PR 61 판부터 요청 열이 통째로 같다(배포 전 dbsl 정리 목록 openAlgoOrders 도 양쪽).

#[path = "common/flowlog.rs"]
mod flowlog;

use rtorder::shadow;
use serde_json::Value;

#[tokio::test]
async fn flow_fixture_matches_python_full_flow() {
    let doc: Value = serde_json::from_str(include_str!("fixtures/flow_fixture.json")).unwrap();
    let chs = shadow::chains(&flowlog::to_jsonl(&flowlog::flow_to_log(&doc))).unwrap();
    let (mut bad, mut nreq, mut nsigned) = (vec![], 0, 0);
    // 시나리오마다 새 폴더(무장 파일·복구 파일이 앞 시나리오와 섞이지 않게)
    let mut by_sc: Vec<Vec<shadow::Chain>> = vec![];
    for c in chs {
        let sc = c.id.split('.').next().unwrap().to_string();
        if by_sc.last().is_none_or(|v| v[0].id.split('.').next().unwrap() != sc) {
            by_sc.push(vec![]);
        }
        by_sc.last_mut().unwrap().push(c);
    }
    for (i, grp) in by_sc.iter().enumerate() {
        let dir = std::env::temp_dir().join(format!("rtorder_flow_{}_{i}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for r in shadow::replay(grp, "ETHUSDC", &dir, true, Some("test_secret")).await {
            nreq += r.requests;
            if !r.diffs.is_empty() {
                let name = grp[0].start["name"].as_str().unwrap_or("");
                bad.push(format!("{name} {}: {}", r.chain, r.diffs.join(" | ")));
            }
        }
        nsigned += grp.iter().flat_map(|c| &c.wire).filter(|w| w["target"].as_str().unwrap().contains("signature=")).count();
        let _ = std::fs::remove_dir_all(&dir);
    }
    println!("시나리오 {} · 걸음 {} · 요청 {nreq}(서명 URL 글자까지 {nsigned})", by_sc.len(), by_sc.iter().map(Vec::len).sum::<usize>());
    for b in &bad {
        println!("불일치 {b}");
    }
    assert!(bad.is_empty(), "불일치 {}건", bad.len());
    assert_eq!((by_sc.len(), nreq), (9, 282), "시나리오·요청 수(청산 접수 3걸음 포함)");
}
