//! 지정가 집행 불변식(파이썬 test_manual_peg_execute_invariants_20260913.py · test_manual_entry_cancel_20261011.py ·
//! test_manual_entry_wait_20261010.py · test_manual_limit_entry_20261031.py 의 사례 + 규칙 문서 «시험 없는 규칙» 중 집행 몫).
//! 고정하는 셋: ① 총 체결 ≤ 계획(거래소 쪽에서 다시 센다) ② 살아남은 지정가 ≤ 1 이고 그때는 phase=error ③ 어떤 경로로도 종료 상태.
//! 🔴메모: 2026-09-13 취소 확인 없이 재호가해 지정가 41개가 동시에 살았다.
//! 네트워크 0 · 시계는 가짜(3초·120초를 기다리지 않는다) · 가짜 호출은 상한 20,000번(넘으면 panic -- 끝없이 도는 재호가 방지).

include!("common/fake.rs");

// ── 파이썬 inv #1~#11 ───────────────────────────────────────────────────────

#[test]
fn inv01_normal_fill() {
    let mut fx = Fake { fill_after: 1, ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("정상", &st, &fx, 2.0);
    assert!(st.phase == Phase::FilledMaker && st.filled == 2.0 && fx.placed.len() == 1);
}

#[test]
fn inv02_drift_repegs_then_needs_market_decision() {
    let mut fx = Fake { drift: true, ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("리페그", &st, &fx, 2.0);
    // 재호가가 **실제로 돈다**(파이썬 시험이 가짜 시그니처 불일치로 재호가 0회인 채 통과하던 사고, 2026-11-20 f6aa1c14)
    // 마지막 앞지름은 리페그 수를 올린 뒤 마감에 걸려 새로 걸지 않는다(원본과 같다) -- 그래서 걸린 수 = 리페그 수 또는 +1
    assert!(st.repegs >= 30 && (fx.placed.len() as u32 == st.repegs || fx.placed.len() as u32 == st.repegs + 1), "repegs {} placed {}", st.repegs, fx.placed.len());
    assert_eq!(st.phase, Phase::FilledTaker); // 파이썬 = 잔량 시장가(filled_taker)
    assert!(fx.placed.windows(2).all(|w| w[1].price < w[0].price), "매도 재호가는 새 최우선 매도호가로 내려간다");
}

fn fail_cases(name: &str, kind: &'static str, phase: Phase, entry: bool) {
    for n in [0, 1, 3] {
        let label = format!("{name}(재호가 {n}회 뒤)");
        let mut fx = Fake { drift: true, fail: vec![kind], ok_first: n, ..Default::default() };
        let st = go(&mut fx, &if entry { entry_plan() } else { exit_plan() }, if entry { Kind::Entry } else { Kind::Exit });
        check(&label, &st, &fx, 2.0);
        assert_eq!(st.phase, phase, "{label}");
        assert_eq!(st.repegs, n, "{label}: 재호가 수 -- 0 이면 주입이 재호가 경로를 안 지났다");
        assert_eq!(fx.placed.len() as u32, n + (kind != "place") as u32, "{label}: 실패 뒤에 또 걸었다");
        assert_eq!(fx.live().len(), (kind == "cancel") as usize, "{label}");
        assert!(fx.markets.is_empty(), "{label}: 멈춰야 하는데 시장가로 갔다");
    }
}

#[test]
fn inv03_cancel_fail_during_repeg_stops_with_error() {
    fail_cases("취소실패+리페그", "cancel", Phase::Error, false);
}

#[test]
fn inv04_partial_fill_then_drift_rests_only_remainder() {
    let mut fx = Fake { fill_after: 1, partial: Some(0.5), drift: true, ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("부분체결+리페그", &st, &fx, 2.0);
    assert!(st.repegs >= 1 && fx.placed[1].quantity == 1.0 && fx.placed.iter().map(|o| o.quantity).sum::<f64>() <= 2.0 * (REPEG_MAX as f64 + 2.0));
}

#[test]
fn inv05_entry_cancel_fail() {
    fail_cases("진입/취소실패", "cancel", Phase::Error, true);
}

#[test]
fn inv06_place_fail_stops() {
    fail_cases("접수실패", "place", Phase::Rejected, false);
}

#[test]
fn inv07_reread_fail_after_cancel() {
    fail_cases("재조회실패", "get_after_cancel", Phase::Error, false);
}

#[test]
fn inv08_partial_exit_plan_never_exceeds() {
    let mut fx = Fake { fill_after: 1, ..Default::default() };
    let st = go(&mut fx, &ExecPlan { quantity: 1.0, ..exit_plan() }, Kind::Exit);
    check("부분청산", &st, &fx, 1.0);
    assert!(fx.placed.iter().map(|o| o.quantity).sum::<f64>() <= 1.0 + 1e-9);
}

#[test]
fn inv09_market_exit_plan_places_nothing() {
    let mut fx = Fake { fill_after: 1, ..Default::default() };
    let st = go(&mut fx, &ExecPlan { market: true, market_reason: Some("변동성".into()), ..exit_plan() }, Kind::Exit);
    check("시장가계획", &st, &fx, 2.0);
    assert!(fx.placed.is_empty() && st.phase == Phase::FilledTaker && fx.markets.len() == 1 && fx.markets[0].2 == 2.0 && st.filled == 2.0);
}

#[test]
fn inv10_entry_repegs_and_books_match_exchange() {
    let mut fx = Fake { drift: true, ..Default::default() };
    let st = go(&mut fx, &entry_plan(), Kind::Entry);
    check("진입/리페그", &st, &fx, 2.0);
    assert!(st.repegs >= 30 && st.phase == Phase::FilledTaker && fx.markets.len() == 1, "{st:?}");
    assert!(fx.placed.windows(2).all(|w| w[1].price > w[0].price), "매수 재호가는 위로 따라간다");
}

#[test]
fn inv11_post_only_reject_repegs_then_caps() {
    for (k, phase) in [(3, Phase::FilledMaker), (REPEG_MAX + 5, Phase::Rejected)] {
        let mut fx = Fake { fill_after: 1, post_only_rejects: k, ..Default::default() };
        let st = go(&mut fx, &exit_plan(), Kind::Exit);
        check("post-only", &st, &fx, 2.0);
        assert_eq!((st.phase, st.repegs, fx.placed.len()), (phase, k.min(REPEG_MAX), (phase == Phase::FilledMaker) as usize));
        assert!(fx.book_times.len() as u32 == k.min(REPEG_MAX), "-5022 마다 새 호가를 읽는다");
    }
}

// ── 규칙 문서 «시험 없는 규칙» 중 집행 몫 + 추가 실패 주입 ──────────────────────

#[test]
fn e06_unknown_outcome_lookup_adopts_ghost_order() {
    // 시간초과(숫자로 시작 안 함)인데 실제로는 접수됨 → 표식 조회로 그 주문을 이어 간다(두 번 걸지 않는다)
    let mut fx = Fake { fill_after: 2, place_err: Some(("TimeoutError: read".into(), true)), ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("유령 입양", &st, &fx, 2.0);
    assert!(st.phase == Phase::FilledMaker && fx.placed.len() == 1 && fx.lookups == 1);
    // 접수 안 됐으면 세 번 본 뒤 표식 DELETE → error(2026-11-20 D2 해결 전: 한 번 보고 DELETE 없이 rejected)
    let mut fx = Fake { fill_after: 1, place_err: Some(("TimeoutError: read".into(), false)), ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("유령 없음", &st, &fx, 2.0);
    assert!(st.phase == Phase::Error && fx.lookups == 3 && fx.deletes.len() == 1);
}

#[test]
fn d2_unknown_outcome_same_on_all_paths() {
    // D2 해결(2026-11-20): «나갔는지 모르는 주문»(5xx·-1007·-1006·네트워크) 처리가 세 경로(fill_maker = 청산·진입 · 기다리기 · 지정가 대기)에서 같다 --
    // 표식으로 3번 조회 → 찾으면 이어 감 → 끝내 모르면 표식 DELETE 뒤 error. 해결 전엔 fill_maker 는 네트워크 오류만 한 번 · 기다리기는 조회 없음(유령 생존 + rejected).
    let to = || "TimeoutError: read".to_string();
    // (이름, 설정, 기대 phase, 표식 조회 수, 표식 DELETE 수, 끝에 살아 있는 지정가 수)
    let cases: Vec<(&str, Fake, Phase, u32, usize, usize)> = vec![
        ("a 503·접수됨", Fake { place_err: Some(("503 Service Unavailable".into(), true)), ..Default::default() }, Phase::FilledMaker, 1, 0, 0),
        ("b -1007·접수됨", Fake { place_err: Some(("408 [-1007] Timeout waiting for response from backend server. Send status unknown; execution status unknown.".into(), true)), ..Default::default() }, Phase::FilledMaker, 1, 0, 0),
        ("b' -1006·접수됨", Fake { place_err: Some(("400 [-1006] An unexpected response was received from the message bus. Execution status unknown.".into(), true)), ..Default::default() }, Phase::FilledMaker, 1, 0, 0),
        ("c 시간초과·둘째 조회에 보임", Fake { place_err: Some((to(), true)), lookup_miss: 1, ..Default::default() }, Phase::FilledMaker, 2, 0, 0),
        ("d 접수 안 됨·3번 다 없음", Fake { place_err: Some((to(), false)), ..Default::default() }, Phase::Error, 3, 1, 0),
        ("e 조회·DELETE 전부 실패", Fake { place_err: Some((to(), true)), fail: vec!["lookup", "delete_cid"], ..Default::default() }, Phase::Error, 3, 1, 1),
        ("음성: 4xx 확정 거부", Fake { place_err: Some(("400 [-2019] Margin is insufficient.".into(), false)), ..Default::default() }, Phase::Rejected, 0, 0, 0),
    ];
    for (path, tag) in [("exit", PEG_TAG), ("entry", PEG_TAG), ("limit", WAIT_TAG)] {
        for (name, base, phase, lookups, dels, live_n) in &cases {
            let label = format!("불명 POST {path}/{name}");
            let mut fx = Fake { fill_after: 1, place_err: base.place_err.clone(), lookup_miss: base.lookup_miss, fail: base.fail.clone(), ..Default::default() };
            let (p, k) = match path {
                "exit" => (exit_plan(), Kind::Exit),
                "entry" => (entry_plan(), Kind::Entry),
                _ => (entry_plan(), Kind::Limit(60.0, false, false)),
            };
            let st = go(&mut fx, &p, k);
            check(&label, &st, &fx, 2.0);
            assert!(!(fx.live().len() > 0 && st.phase == Phase::Rejected), "{label}: 살아 있는 지정가인데 rejected");
            assert_eq!(st.phase, *phase, "{label}: {:?}", st.error);
            assert_eq!((fx.lookups, fx.deletes.len(), fx.live().len()), (*lookups, *dels, *live_n), "{label}: (표식 조회, DELETE, 살아 있는 지정가)");
            assert!(fx.deletes.iter().all(|d| matches!(d, OrderRef::Client(c) if c.starts_with(tag))), "{label}: DELETE 는 이 경로 표식으로 {:?}", fx.deletes);
            assert!(fx.placed.len() <= 1 && fx.markets.is_empty(), "{label}: 두 번 걸었거나 시장가로 갔다");
            if *phase == Phase::Error {
                assert!(st.error.as_deref().unwrap().contains("접수 여부를 확인하지 못했습니다"), "{label}");
            }
        }
    }
}

#[test]
fn fill_just_before_cancel_is_counted() {
    let mut fx = Fake { drift: true, fill_on_cancel: true, ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("취소 직전 체결", &st, &fx, 2.0);
    assert!(st.phase == Phase::FilledMaker && fx.placed.len() == 1 && st.filled == 2.0, "재조회 상태(FILLED)를 믿고 다시 걸지 않는다");
}

#[test]
fn partial_then_book_runs_away() {
    let mut fx = Fake { fill_after: 1, partial: Some(0.4), script: vec![(3.5, Cmd::Drift(true))], ..Default::default() };
    let st = go(&mut fx, &entry_plan(), Kind::Entry);
    check("부분 체결 뒤 달아남", &st, &fx, 2.0);
    assert!(st.repegs >= 1 && fx.placed[1].quantity == 1.2 && fx.placed[1].price > fx.placed[0].price);
}

#[test]
fn e16_early_wake_checks_book_only_every_poll_sec() {
    let mut fx = Fake { drift: false, wake_every: Some(0.5), ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("알림 일찍 깸", &st, &fx, 2.0);
    let gaps: Vec<f64> = fx.book_times.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(!gaps.is_empty() && gaps.iter().all(|g| *g >= POLL_SEC - 1e-6), "리페그 판단 간격 {gaps:?}");
    assert!(fx.gets as usize > 4 * fx.book_times.len(), "상태 조회는 알림마다");
}

#[test]
fn e12_status_read_failure_is_retried() {
    // 처음 5번 조회 실패 → 주문은 살아 있다고 보고 재시도 → 체결
    let mut fx = Fake { fill_after: 1, get_fail_first: 5, ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("조회 실패 뒤 회복", &st, &fx, 2.0);
    assert!(st.phase == Phase::FilledMaker && fx.placed.len() == 1);
    // 끝까지 실패 → 버리고 재호가하지 않는다 → 취소 확인도 못 함 → error(사람에게)
    let mut fx = Fake { fill_after: 1, fail: vec!["get"], ..Default::default() };
    let st = go(&mut fx, &ExecPlan { fallback_after_sec: 30.0, ..exit_plan() }, Kind::Exit);
    check("조회 계속 실패", &st, &fx, 2.0);
    assert!(st.phase == Phase::Error && fx.placed.len() == 1);
}

#[test]
fn e05_book_failure_after_post_only_reject_is_rejected() {
    let mut fx = Fake { post_only_rejects: 1, fail: vec!["book"], ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("호가 실패", &st, &fx, 2.0);
    assert!(st.phase == Phase::Rejected && st.error.as_deref().unwrap().starts_with("호가 조회 실패"));
}

#[test]
fn e09_unconfirmed_cancel_reports_last_seen_fill() {
    let mut fx = Fake { fill_after: 1, partial: Some(0.5), drift: true, fail: vec!["cancel"], ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("취소 확인 실패 + 부분", &st, &fx, 2.0);
    assert!(st.phase == Phase::Error && st.filled == 1.0, "마지막에 본 체결량(하한)을 돌려준다: {}", st.filled);
}

// ── 진입 대기 취소(cancel 시험) · 감시 진입 · 시장가 계획 ───────────────────────

#[test]
fn e18_entry_cancel_paths() {
    let mut fx = Fake { script: vec![(5.0, Cmd::Action(Action::Cancel))], ..Default::default() };
    let st = go(&mut fx, &plan("BUY", "LONG", 1.0), Kind::Entry);
    check("취소", &st, &fx, 1.0);
    assert!(st.phase == Phase::Cancelled && fx.placed.len() == 1 && fx.deletes.len() == 1 && st.cancel_reason.as_deref() == Some("사용자 취소"));
    assert!(fx.t < 5.0 + POLL_SEC + 1.0, "취소는 다음 조회(≤3초)에 먹어야 한다 -- 마감까지 기다렸다: t={}", fx.t);
    let mut fx = Fake { fill_after: 1, partial: Some(0.3), script: vec![(5.0, Cmd::Action(Action::Cancel))], ..Default::default() };
    let st = go(&mut fx, &plan("BUY", "LONG", 1.0), Kind::Entry);
    check("부분 뒤 취소", &st, &fx, 1.0);
    assert!(st.phase == Phase::Cancelled && (st.filled - 0.3).abs() < 1e-12);
    let mut fx = Fake { fail: vec!["cancel"], script: vec![(5.0, Cmd::Action(Action::Cancel))], ..Default::default() };
    let st = go(&mut fx, &plan("BUY", "LONG", 1.0), Kind::Entry);
    check("취소 확인 실패", &st, &fx, 1.0);
    assert!(st.phase == Phase::Error && st.error.unwrap().contains("거래소에서 직접 확인") && fx.placed.len() == 1);
    // 음성 대조: 취소 없이 같은 상황이면 마감에 잔량 → (파이썬 시장가) 시장가 결정 대기
    let mut fx = Fake { fill_after: 1, partial: Some(0.3), ..Default::default() };
    let st = go(&mut fx, &plan("BUY", "LONG", 1.0), Kind::Entry);
    check("취소 없음", &st, &fx, 1.0);
    assert!(st.phase == Phase::FilledTaker && (fx.markets[0].2 - 0.7).abs() < 1e-12);
}

#[test]
fn maker_only_and_market_entry_plans() {
    let mut fx = Fake::default();
    let st = go(&mut fx, &ExecPlan { maker_only: true, ..entry_plan() }, Kind::Entry);
    check("감시 진입 따라가기 마감", &st, &fx, 2.0);
    assert!(st.phase == Phase::Cancelled && st.cancel_reason.unwrap().contains("시장가 없음"));
    let mut fx = Fake::default();
    let st = go(&mut fx, &ExecPlan { market: true, ..entry_plan() }, Kind::Entry);
    check("시장가 진입 계획", &st, &fx, 2.0);
    assert!(fx.placed.is_empty() && st.phase == Phase::FilledTaker && fx.markets.len() == 1);
}

// 기다리기(run_entry_wait · M3)는 2026-11-21 원본 PR 61 처럼 지웠다.

// ── 지정가 대기(limit 시험) ──────────────────────────────────────────────────

#[test]
fn limit_paths() {
    let lp = || entry_plan();
    let mut fx = Fake { fill_after: 3, drift: true, ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, false, false));
    check("고정가", &st, &fx, 2.0);
    assert!(st.phase == Phase::FilledMaker && fx.placed.len() == 1 && fx.placed[0].price == 2500.0 && st.repegs == 0);
    let mut fx = Fake { script: vec![(5.0, Cmd::Action(Action::Cancel))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, false, false));
    check("취소", &st, &fx, 2.0);
    assert_eq!((st.phase, st.cancel_reason.as_deref(), fx.deletes.len()), (Phase::Cancelled, Some("사용자 취소"), 1));
    let mut fx = Fake { script: vec![(5.0, Cmd::Amend(2490.0, Some(2.008))), (12.0, Cmd::FillAfter(1))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, false, false));
    check("수정", &st, &fx, 2.008);
    assert_eq!(fx.placed.iter().map(|o| (o.price, o.quantity)).collect::<Vec<_>>(), vec![(2500.0, 2.0), (2490.0, 2.008)]);
    assert_eq!((st.amends, st.phase), (1, Phase::FilledMaker));
    // 일부 체결 뒤 수정 = 버림(검증 D4): 남은 수량만 옛 가격에
    let mut fx = Fake { fill_after: 1, partial: Some(0.4), script: vec![(5.0, Cmd::Amend(2490.0, Some(2.4))), (12.0, Cmd::Action(Action::Cancel))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, false, false));
    check("부분 뒤 수정", &st, &fx, 2.0);
    assert_eq!(fx.placed.iter().map(|o| (o.price, o.quantity)).collect::<Vec<_>>(), vec![(2500.0, 2.0), (2500.0, 1.2)]);
    let mut fx = Fake { post_only_rejects: 1, ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, false, false));
    check("-5022", &st, &fx, 2.0);
    assert!(st.phase == Phase::Rejected && st.error.unwrap().contains("메이커") && fx.placed.is_empty());
    let mut fx = Fake::default();
    let st = go(&mut fx, &lp(), Kind::Limit(10.0, false, false));
    check("만료", &st, &fx, 2.0);
    assert_eq!((st.phase, st.cancel_reason.as_deref()), (Phase::Cancelled, Some("대기 시간 만료")));
    let mut fx = Fake { kill_at: Some(2), ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, false, false));
    check("거래소 취소", &st, &fx, 2.0);
    assert_eq!(st.cancel_reason.as_deref(), Some("거래소에서 취소됨"));
    let mut fx = Fake { fill_after: 1, partial: Some(0.2), script: vec![(8.0, Cmd::Action(Action::Cancel))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, true, false));
    check("on_fill 한 번", &st, &fx, 2.0);
    assert_eq!((fx.on_fills, st.phase), (1, Phase::Cancelled));
    let mut fx = Fake { fill_after: 1, partial: Some(0.2), script: vec![(8.0, Cmd::Action(Action::Market))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, false, false));
    check("지금 시장가", &st, &fx, 2.0);
    assert!(st.phase == Phase::FilledTaker && (fx.markets[0].2 - 1.6).abs() < 1e-12 && fx.deletes.len() == 1, "걸린 지정가를 취소 확인한 뒤");
    let mut fx = Fake { kill_at: Some(2), script: vec![(4.0, Cmd::Action(Action::Market))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, false, false));
    check("사람이 취소한 뒤 시장가", &st, &fx, 2.0);
    assert_eq!(st.phase, Phase::Cancelled);
    // 자동 재조정: 한 번만(두 번째 확인은 1bp 미만) · 체결 시작 뒤엔 확인조차 안 한다
    let mut fx = Fake { reprice_vals: vec![2499.4], script: vec![(50.0, Cmd::FillAfter(1))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(120.0, false, true));
    check("재조정", &st, &fx, 2.0);
    assert_eq!(fx.placed.iter().map(|o| o.price).collect::<Vec<_>>(), vec![2500.0, 2499.4]);
    assert_eq!((st.reprices, st.amends, st.phase), (1, 0, Phase::FilledMaker));
    let mut fx = Fake { fill_after: 1, partial: Some(0.4), reprice_vals: vec![2400.0], script: vec![(60.0, Cmd::Action(Action::Cancel))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(120.0, false, true));
    check("부분 뒤 재조정 없음", &st, &fx, 2.0);
    assert_eq!((fx.reprice_calls, fx.placed.len()), (0, 1));
    // 수정 직후 -5022 → 죽지 않고 새 목표가로 다시(검증 D6)
    let mut fx = Fake { reprice_vals: vec![2491.0], reject_at_placed: Some(1), script: vec![(5.0, Cmd::Amend(2490.0, None)), (30.0, Cmd::FillAfter(1))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(120.0, false, true));
    check("수정 직후 -5022", &st, &fx, 2.0);
    assert_eq!((st.phase, fx.placed.iter().map(|o| o.price).collect::<Vec<_>>()), (Phase::FilledMaker, vec![2500.0, 2491.0]));
    // PEG 동작: 취소 확인 → 따라가기 → 잔량 버림(시장가 없음)
    let mut fx = Fake { script: vec![(5.0, Cmd::Action(Action::Peg)), (5.0, Cmd::Drift(true))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(600.0, false, false));
    check("PEG", &st, &fx, 2.0);
    assert!(st.phase == Phase::Cancelled && st.pegged.is_some() && st.repegs >= 30 && fx.markets.is_empty() && st.cancel_reason.unwrap().contains("시장가 없음"));
    let mut fx = Fake { script: vec![(5.0, Cmd::Action(Action::Peg)), (6.0, Cmd::FillAfter(1))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(600.0, false, false));
    check("PEG 체결", &st, &fx, 2.0);
    assert_eq!(st.phase, Phase::FilledMaker);
}

// ── 무작위 실패 주입(고정 시드) ──────────────────────────────────────────────
// D2 해결(2026-11-20) 뒤 «불명» 접수를 네 경로 모두에 넣는다(해결 전엔 fill_maker·기다리기가 유령을 남겨 지정가 대기에만 넣었다).

#[test]
fn fuzz_random_faults_keep_invariants() {
    let mut seed: u64 = 20261120;
    let mut rnd = |n: u64| { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) % n };
    let (mut runs, mut repegs) = (0, 0u64);
    for _ in 0..400 {
        let fail: Vec<&'static str> = ["cancel", "get_after_cancel", "get", "place", "book"].into_iter().filter(|_| rnd(5) == 0).collect();
        let mut fx = Fake {
            fill_after: [0, 1, 2, 5, 30][rnd(5) as usize] as u32, partial: [None, Some(0.3), Some(0.5)][rnd(3) as usize],
            fail, ok_first: rnd(4) as u32, post_only_rejects: rnd(4) as u32, drift: rnd(2) == 0, fill_on_cancel: rnd(6) == 0,
            kill_at: (rnd(8) == 0).then_some(3), wake_every: (rnd(4) == 0).then_some(0.7), ..Default::default()
        };
        let k = rnd(4);
        fx.place_err = (rnd(3) == 0).then(|| (["TimeoutError: x", "503 [-1007] Timeout"][rnd(2) as usize].into(), rnd(2) == 0));
        if rnd(3) == 0 {
            fx.script.push((rnd(60) as f64, Cmd::Action([Action::Cancel, Action::Market, Action::Peg][rnd(3) as usize])));
        }
        let p = if k == 1 { exit_plan() } else { entry_plan() };
        let st = go(&mut fx, &p, match k { 0 => Kind::Entry, 1 => Kind::Exit, 2 => Kind::Limit(300.0, true, false), _ => Kind::Limit(300.0, true, true) });
        check(&format!("무작위 #{runs}"), &st, &fx, 2.0);
        repegs += st.repegs as u64;
        runs += 1;
    }
    assert!(repegs > 100, "무작위 사례에서 재호가가 실제로 돌았다: {repegs}");
    println!("무작위 {runs}건 통과 · 재호가 합 {repegs}");
}

// ── 감시 진입(run_entry_stop · 파이썬 test_manual_stop_entry_20261109.py 의 사례 · 둘째 조각) ─────────────────

/// touched 는 «n 번째 확인에 닿음»(0 = 안 닿음). 닿음 확인에 넘어온 since 를 모은다.
fn go_stop(fx: &mut Fake, p: &ExecPlan, touch_at: u32, max_sec: f64) -> (ExecState, Vec<f64>) {
    let s: Shared = Arc::new(Mutex::new(ExecState::default()));
    fx.shared = Some(s.clone());
    let (mut n, mut sinces) = (0u32, Vec::new());
    block_on(run_entry_stop(fx, p, &s, |since| {
        n += 1;
        sinces.push(since);
        assert!(n < 20_000, "닿음 확인 상한 -- 루프가 안 끝난다");
        touch_at > 0 && n >= touch_at
    }, max_sec));
    let out = s.lock().unwrap().clone();
    (out, sinces)
}

#[test]
fn e25_stop_entry_paths() {
    // 닿기 전엔 거래소 주문 0 · 닿으면 최우선 호가에 따라가기 → 체결
    let mut fx = Fake { fill_after: 1, ..Default::default() };
    let (st, sinces) = go_stop(&mut fx, &entry_plan(), 5, 3600.0);
    check("감시 닿음", &st, &fx, 2.0);
    assert!(st.phase == Phase::FilledMaker && st.stop && st.triggered.is_some() && sinces.len() == 5 && fx.placed.len() == 1 && fx.placed[0].price == 2500.0);
    assert!(fx.placed[0].client_order_id.starts_with(PEG_TAG));
    // 따라가기 마감 뒤 잔량은 버린다 -- 시장가 없음(사용자 «지정가 체결은 절대 시장가로 바꾸지 않는다»)
    let mut fx = Fake { fill_after: 1, partial: Some(0.5), ..Default::default() };
    let (st, _) = go_stop(&mut fx, &entry_plan(), 1, 3600.0);
    check("감시 마감 잔량", &st, &fx, 2.0);
    assert!(st.phase == Phase::Cancelled && fx.markets.is_empty() && st.cancel_reason.unwrap().contains("시장가 없음"));
    // 선 옮김 = 닿음 판정을 그 시각부터 다시 · 수량·가격 칸 갱신
    let mut fx = Fake { fill_after: 1, script: vec![(2.5, Cmd::Amend(2490.0, Some(1.5)))], ..Default::default() };
    let (st, sinces) = go_stop(&mut fx, &entry_plan(), 6, 3600.0);
    check("감시 선 옮김", &st, &fx, 1.5);
    assert!(st.amends == 1 && st.quantity == 1.5 && sinces[0] < sinces[5] && sinces[1] == sinces[0] && fx.placed[0].quantity == 1.5, "{sinces:?}");
    // 수량 없는 수정 = 원본 KeyError → settle 의 error(주문 없음)
    let mut fx = Fake { script: vec![(1.5, Cmd::Amend(2490.0, None))], ..Default::default() };
    let (st, _) = go_stop(&mut fx, &entry_plan(), 0, 3600.0);
    assert!(st.phase == Phase::Error && fx.placed.is_empty());
    // 취소 = 주문 없이 끝 · 만료 = «대기 시간 만료»
    let mut fx = Fake { script: vec![(3.5, Cmd::Action(Action::Cancel))], ..Default::default() };
    let (st, _) = go_stop(&mut fx, &entry_plan(), 0, 3600.0);
    check("감시 취소", &st, &fx, 2.0);
    assert!(st.phase == Phase::Cancelled && st.cancel_reason.as_deref() == Some("사용자 취소") && fx.placed.is_empty() && fx.calls == 0);
    let mut fx = Fake::default();
    let (st, sinces) = go_stop(&mut fx, &entry_plan(), 0, 10.0);
    check("감시 만료", &st, &fx, 2.0);
    assert!(st.phase == Phase::Cancelled && st.cancel_reason.as_deref() == Some("대기 시간 만료") && fx.calls == 0 && sinces.len() == 11);
    // TAKER = 미정 시장가 분기 M5(파이썬은 run_entry(MARKET) 전량 시장가) -- 주문 없이 멈춘다
    let mut fx = Fake { script: vec![(2.5, Cmd::Action(Action::Market))], ..Default::default() };
    let (st, _) = go_stop(&mut fx, &entry_plan(), 0, 3600.0);
    check("감시 TAKER", &st, &fx, 2.0);
    assert!(st.phase == Phase::FilledTaker && fx.markets.len() == 1 && fx.markets[0].2 == 2.0 && fx.placed.is_empty());
    // PEG = 지금 닿은 것으로 치고 따라가기
    let mut fx = Fake { fill_after: 1, script: vec![(2.5, Cmd::Action(Action::Peg))], ..Default::default() };
    let (st, _) = go_stop(&mut fx, &entry_plan(), 0, 3600.0);
    check("감시 PEG", &st, &fx, 2.0);
    assert!(st.phase == Phase::FilledMaker && st.triggered.is_some() && st.action.is_none());
    // 닿은 뒤 호가 조회 실패(D11: 원본은 try 밖 → settle «집행 오류») -- 걸린 주문 없음
    let mut fx = Fake { fail: vec!["book"], ..Default::default() };
    let (st, _) = go_stop(&mut fx, &entry_plan(), 1, 3600.0);
    check("감시 호가 실패", &st, &fx, 2.0);
    assert!(st.phase == Phase::Error && st.error.unwrap().starts_with("집행 오류") && fx.placed.is_empty());
}
