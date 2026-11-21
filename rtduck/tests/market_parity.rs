//! market_data_worker 의 화면·감시 몫 파이썬 일치 시험. 정답지 = tools/gen_market_fixture.py 가 원본 TailRiskInterceptor·write_dashboard_state 를
//! 가짜 시계로 돌려 모은 liq_burst_state.json 글자(쓸 때마다) · dashboard_state.json 글자(세션 경계 시각 포함) · tail_risk_1m 덤프.
use serde_json::Value;

#[test]
fn market_state_json_and_tail_risk_match_python() {
    let inp: Value = serde_json::from_str(include_str!("fixtures/market_input.json")).unwrap();
    let gold: Value = serde_json::from_str(include_str!("fixtures/market_golden.json")).unwrap();
    let dir = std::env::temp_dir().join(format!("rtduck_market_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (burst, dash) = rtduck::market::replay(&inp, &dir.join("tail_risk.duckdb"), &dir);
    let strs = |k: &str| gold[k].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect::<Vec<_>>();
    let (wb, wd) = (strs("burst"), strs("dash"));
    assert_eq!(burst.len(), wb.len(), "liq_burst_state 쓴 횟수");
    for (k, (a, b)) in burst.iter().zip(&wb).enumerate() {
        assert_eq!(a, b, "liq_burst_state {k}번째 글자");
    }
    assert_eq!(dash.len(), wd.len(), "dashboard_state 쓴 횟수");
    for (k, (a, b)) in dash.iter().zip(&wd).enumerate() {
        assert_eq!(a, b, "dashboard_state {k}번째 글자");
    }
    let got = rtduck::duckdump::dump(&dir.join("tail_risk.duckdb")).unwrap();
    let w = &gold["tables"]["tail_risk_1m"];
    assert_eq!(got["tail_risk_1m"]["cols"], w["cols"], "열·형");
    let (g, ww) = (got["tail_risk_1m"]["rows"].as_array().unwrap(), w["rows"].as_array().unwrap());
    assert_eq!(g.len(), ww.len(), "tail_risk_1m 행 수");
    for (k, (a, b)) in g.iter().zip(ww).enumerate() {
        assert_eq!(a, b, "tail_risk_1m 행 {k}");
    }
    eprintln!("market 일치: liq_burst 글자 {} · dashboard 글자 {} · tail_risk_1m {}행 × {}칸", burst.len(), dash.len(), g.len(), w["cols"].as_array().unwrap().len());
    let _ = std::fs::remove_dir_all(&dir);
}
