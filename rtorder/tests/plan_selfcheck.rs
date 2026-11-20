//! 원본 자체점검(`_self_check` 85개 · `_self_check_rec_rule`)이 고정한 사례를 그대로 옮긴 것. 정답지 대조와 별개로 «이 숫자여야 한다»를 박아 둔다.

use rtorder::plan::*;
use serde_json::json;

const F: Filters = Filters { step: 0.001, tick: 0.01, min_qty: 0.001, min_notional: 20.0 };

fn entry(side: &'static str, bid: f64, ask: f64, rq: f64, cap: Option<f64>) -> EntryInput<'static> {
    EntryInput { side, best_bid: bid, best_ask: ask, recommended_qty: rq, cap_notional: cap, filters: F, ..Default::default() }
}
fn exit(side: &'static str, qty: f64) -> ExitInput<'static> {
    ExitInput { position_side: side, position_qty: qty, best_bid: 2470.0, best_ask: 2470.01, filters: F, symbol: "ETHUSDT", entry_price: 0.0, mark_price: 0.0, vol_bpm: None, fraction: 1.0 }
}
fn arm(v: serde_json::Value) -> Armed {
    Armed::from_value(&v)
}

#[test]
fn self_check_entry_exit() {
    let p = build_entry_plan(&entry("LONG", 2470.0, 2470.01, 3.4, None)).unwrap();
    assert!(p.side == "BUY" && p.position_side == "LONG" && p.price == 2470.0 && p.time_in_force == "GTX" && p.blocked.is_none()); // 롱 peg = 최우선 매수호가
    let p = build_entry_plan(&entry("SHORT", 2470.0, 2470.01, 3.4, None)).unwrap();
    assert!(p.side == "SELL" && p.price == 2470.01); // 숏 peg = 최우선 매도호가
    // 상한: 명목 15,201 → 6.154 ETH. 내림이라 상한을 **넘지 않는다**.
    let p = build_entry_plan(&entry("LONG", 2470.0, 2470.01, 14.273, Some(15201.0))).unwrap();
    assert!(p.quantity == 6.154 && p.notional_usdt <= 15201.0 && !p.notes.is_empty());
    let p = build_entry_plan(&entry("LONG", 2470.0, 2470.01, 2.0, Some(15201.0))).unwrap();
    assert!(p.quantity == 2.0 && p.notes.is_empty());
    // 🔴상한은 **합산**: 기존 7,642 가 열린 채 6,803 을 더하면 14,444 로 13,579 를 넘었다(예전 코드는 통과시켰다).
    let mut e = entry("SHORT", 2526.08, 2526.09, 2.693, Some(13579.31));
    (e.existing_notional, e.equity, e.leverage) = (7641.6, 1064.69, 30.0);
    let p = build_entry_plan(&e).unwrap();
    assert!(p.total_notional_usdt <= 13579.31 + 1e-6 && p.quantity < 2.693 && !p.notes.is_empty());
    e.existing_notional = 14000.0;
    e.recommended_qty = 2.0;
    let p = build_entry_plan(&e).unwrap();
    assert!(p.quantity == 0.0 && p.blocked.is_some());
    let mut e = entry("SHORT", 2526.08, 2526.09, 2.693, None);
    (e.equity, e.leverage) = (1064.69, 30.0);
    let p = build_entry_plan(&e).unwrap();
    assert!((p.margin_usdt - p.notional_usdt / 30.0).abs() < 0.01);
    assert_eq!(p.effective_leverage, Some(rtorder::pyfmt::round_n(p.notional_usdt / 1064.69, 2)));
    assert!((10.0..20.0).contains(&p.liq_distance_pct.unwrap()) && p.margin_pct_of_equity.is_some());
    // 2026-09-25 비율 = 순자산 대비 증거금 · 🔴09-26 «넘치면 50%까지만»
    let mut e = entry("LONG", 2000.0, 2000.01, 99.0, Some(20000.0));
    (e.existing_notional, e.equity, e.leverage, e.fraction, e.fraction_of_equity, e.order_leverage, e.cap_label) = (8000.0, 2000.0, 20.0, 0.10, true, 20.0, "증거금 상한 50%");
    let p = build_entry_plan(&e).unwrap();
    assert!(p.quantity == 2.0 && p.blocked.is_none());
    e.fraction = 0.40;
    let p = build_entry_plan(&e).unwrap();
    assert!(p.quantity == 6.0 && p.blocked.is_none() && p.capped);
    assert!(p.notes.iter().any(|n| n.contains("요청 40% → 증거금 상한 50%까지 30.0%")), "{:?}", p.notes);
    (e.existing_notional, e.fraction) = (20000.0, 0.10);
    let p = build_entry_plan(&e).unwrap();
    assert!(p.quantity == 0.0 && p.blocked.as_deref() == Some("증거금 상한 50% 여유가 없습니다"));
    // equity/leverage 를 모르면 설명값은 None 이지 0 이 아니다
    let p = build_entry_plan(&entry("LONG", 2470.0, 2470.01, 1.0, None)).unwrap();
    assert!(p.effective_leverage.is_none() && p.liq_distance_pct.is_none());

    // ── 청산 ──
    let x = build_exit_plan(&ExitInput { entry_price: 2400.0, ..exit("LONG", 2.693) }).unwrap();
    assert!(x.side == "SELL" && x.position_side_field == "LONG" && x.price == Some(2470.01) && x.quantity == 2.693 && x.blocked.is_none());
    assert!(!serde_json::to_value(&x).unwrap().as_object().unwrap().contains_key("reduceOnly"), "헤지 모드에서 reduceOnly 는 -1106");
    assert!((x.exit_move_pct.unwrap() - 2.917).abs() < 0.01);
    let x = build_exit_plan(&ExitInput { entry_price: 2500.0, ..exit("SHORT", 3.025) }).unwrap();
    assert!(x.side == "BUY" && x.price == Some(2470.0) && x.exit_move_pct.unwrap() > 0.0);
    // 🔴먼지 포지션은 막지 않는다(시장가 계획 + 이유) -- 미정 분기 M7
    let x = build_exit_plan(&exit("LONG", 0.0015)).unwrap();
    assert!(x.quantity == 0.001 && x.blocked.is_none() && x.kind == "MARKET" && x.market_reason.unwrap().contains("최소") && x.price.is_none());
    assert_eq!(build_exit_plan(&exit("LONG", 0.0)).unwrap().blocked.as_deref(), Some("닫을 포지션이 없습니다"));
    // ── 변동성 연동 마감 ──
    assert_eq!(exit_deadline_sec(None).unwrap(), EXIT_DEADLINE_MAX);
    assert_eq!(exit_deadline_sec(Some(0.0)).unwrap(), EXIT_DEADLINE_MAX);
    let mut prev = 1e9;
    for v in [2.0, 5.25, 9.97, 17.74, 30.0, 100.0] {
        let t = exit_deadline_sec(Some(v)).unwrap();
        assert!(t <= prev && (EXIT_DEADLINE_MIN..=EXIT_DEADLINE_MAX).contains(&t));
        prev = t;
    }
    assert_eq!(exit_deadline_sec(Some(5.25)).unwrap(), EXIT_DEADLINE_MAX);
    assert!((exit_deadline_sec(Some(9.97)).unwrap() - 50.8).abs() < 1.0 && (exit_deadline_sec(Some(17.74)).unwrap() - 21.8).abs() < 1.0);
    assert_eq!(exit_deadline_sec(Some(30.0)).unwrap(), EXIT_DEADLINE_MIN);
    assert_eq!(realized_vol_bpm(&[100.0; 31]), Some(0.0));
    assert_eq!(realized_vol_bpm(&[100.0, 101.0]), None);
    assert_eq!(realized_vol_bpm(&[]), None);
    let noisy: Vec<f64> = (0..31).map(|i| 100.0 * 1.001f64.powi(i % 2)).collect();
    assert!(realized_vol_bpm(&noisy).unwrap() > 0.0);
    let m = build_exit_plan(&ExitInput { vol_bpm: Some(45.0), ..exit("LONG", 2.0) }).unwrap();
    assert!(m.kind == "MARKET" && m.price.is_none() && m.time_in_force.is_none() && !m.repeg && m.market_reason.is_some() && m.reference_price == 2470.01);
    let n = build_exit_plan(&ExitInput { vol_bpm: Some(12.0), ..exit("LONG", 2.0) }).unwrap();
    assert!(n.kind == "LIMIT" && n.repeg && n.market_reason.is_none() && n.vol_bpm == Some(12.0));
    assert!((EXIT_DEADLINE_MIN..EXIT_DEADLINE_MAX).contains(&n.fallback_after_sec));
    // ── 부분 청산 ──
    let full = build_exit_plan(&exit("LONG", 2.693)).unwrap();
    assert!(full.quantity == 2.693 && full.fraction == 1.0 && full.remaining_qty == 0.0);
    let half = build_exit_plan(&ExitInput { fraction: 0.5, ..exit("LONG", 2.693) }).unwrap();
    assert!(half.quantity == 1.346 && half.remaining_qty == 1.347); // 1.3465 -> 스텝 0.001 내림
    for fr in [0.1, 0.25, 0.5, 0.75, 0.99, 1.0] {
        assert!(build_exit_plan(&ExitInput { fraction: fr, ..exit("SHORT", 3.025) }).unwrap().quantity <= 3.025 + 1e-12);
    }
    for bad in [0.0, -0.5, 1.01, 2.0] {
        assert!(matches!(build_exit_plan(&ExitInput { fraction: bad, ..exit("LONG", 2.0) }), Err(PlanError::Value(_))));
    }
    let tiny = build_exit_plan(&ExitInput { fraction: 0.05, ..exit("LONG", 0.01) }).unwrap();
    assert!(tiny.blocked.unwrap().contains("최소"));
    // ── 분할 진입 ──
    const CAP: f64 = 7299.0;
    let one = build_entry_plan(&entry("LONG", 2470.0, 2470.01, 6.14, Some(CAP))).unwrap();
    assert!(one.fraction == 1.0 && (one.available_qty - one.quantity).abs() < F.step && one.available_qty < 6.14 && one.notional_usdt <= CAP + 1e-6);
    let mut held = 0.0;
    for _ in 0..3 {
        let t = build_entry_plan(&EntryInput { existing_notional: held, fraction: 1.0 / 3.0, ..entry("LONG", 2470.0, 2470.01, 6.14, Some(CAP)) }).unwrap();
        assert!(t.total_notional_usdt <= CAP + 1e-6 && t.cap_room_after_usdt.unwrap() >= 0.0);
        held = t.total_notional_usdt;
    }
    assert!(held <= CAP + 1e-6 && held > 0.0);
    let third = build_entry_plan(&EntryInput { fraction: 1.0 / 3.0, ..entry("LONG", 2470.0, 2470.01, 6.14, Some(CAP)) }).unwrap();
    assert!(third.quantity < one.quantity && third.notes.iter().any(|n| n.contains("33%") && n.contains("넣을 수 있는")));
    assert!(third.quantity < CAP / 2470.01 * 0.4);
    for bad in [0.0, -0.2, 1.5] {
        assert!(build_entry_plan(&EntryInput { fraction: bad, ..entry("LONG", 2470.0, 2470.01, 6.14, Some(CAP)) }).is_err());
    }
    // ── 청산 심볼 결정(2026-09-19) ──
    let pos: Vec<Position> = serde_json::from_value(json!([{"symbol": "ETHUSDT", "side": "LONG", "qty": 0.946}, {"symbol": "ETHUSDT", "side": "SHORT", "qty": 1.5}, {"symbol": "ETHUSDC", "side": "LONG", "qty": 2.0}])).unwrap();
    let cand = ["ETHUSDC", "ETHUSDT"];
    let (p, sym, left) = resolve_exit_position(&pos, "LONG", &cand).unwrap();
    assert!(sym == "ETHUSDC" && p.qty == Some(2.0) && left == vec![Leftover { symbol: "ETHUSDT".into(), qty: 0.946 }]);
    let (_, sym, left) = resolve_exit_position(&pos, "SHORT", &cand).unwrap();
    assert!(sym == "ETHUSDT" && left.is_empty());
    assert!(resolve_exit_position(&pos, "LONG", &["SOLUSDT"]).is_none() && resolve_exit_position(&[], "LONG", &cand).is_none());
}

#[test]
fn self_check_rec_rule() {
    let f = F;
    let c = commit_rule_l("LONG", 55.21 * 6f64.sqrt(), 0.458, COMMIT_CAP_X).unwrap();
    assert!(c.binding == "상한" && c.l == 2.0 && c.stop_pct == 0.05);
    let low = commit_rule_l("LONG", 55.21 * 6f64.sqrt(), 0.15, COMMIT_CAP_X).unwrap();
    assert!(low.binding == "노출 맞춤" && (low.l - 1.865).abs() < 1e-3);
    assert_eq!(commit_rule_l("LONG", 250.0, 1.0, HOLD_CAP_X).unwrap().l, 1.5);
    let wide = commit_rule_l("SHORT", 1800.0, 3.0, COMMIT_CAP_X).unwrap();
    assert!(wide.binding == "청산 안전" && wide.l < COMMIT_CAP_X);
    // 청산가가 −5% 손절선보다 0.5% 바깥(교차 마진 청산가 = E(1 ∓ 1/L)/(1 ∓ MMR), 평단 E = 첫 진입가 ∓0.66%)
    for sig in [5.0, 20.0, 60.0, 250.0, 600.0, 1500.0] {
        for vm in [0.2, 0.458, 1.0, 3.0] {
            for (side, sg) in [("LONG", 1.0), ("SHORT", -1.0)] {
                let k = commit_rule_l(side, sig, vm, COMMIT_CAP_X).unwrap();
                let e = 1.0 - sg * COMMIT_AVG_OFF;
                let liq_px = e * (1.0 - sg / k.l) / (1.0 - sg * COMMIT_MMR);
                let stop_px = 1.0 - sg * RULE_STOP_PCT;
                assert!(sg * (stop_px * (1.0 - sg * COMMIT_LIQ_BUF) - liq_px) >= -1e-12, "{side} {sig} {vm}");
            }
        }
    }
    // 손절 시 손실 ≤ L × 5% × 순자산 -- 물타기·다 찬 뒤 재물타기·불타기·손절선 너머
    let (eq, l0) = (2000.0, c.l);
    for path in [vec![2500.0, 2480.0, 2460.0, 2440.0, 2470.0], vec![2500.0, 2500.0, 2450.0, 2440.0], vec![2500.0, 2600.0, 2700.0], vec![2500.0, 2400.0]] {
        let sl = path[0] * (1.0 - RULE_STOP_PCT);
        let (mut q, mut cost) = (0.0f64, 0.0f64);
        for px in path {
            let used = if q != 0.0 { q * (cost / q - sl).max(0.0) } else { 0.0 };
            let rr = rec_rule_size(l0, eq, q * px, 20.0, Some(q == 0.0), used, Some((px - sl) / px)).unwrap();
            let add = rr.target_notional / px;
            (q, cost) = (q + add, cost + add * px);
        }
        assert!(q * (cost / q - sl) <= l0 * RULE_STOP_PCT * eq * (1.0 + 1e-9));
    }
    assert_eq!(rec_rule_size(l0, eq, 1000.0, 20.0, Some(false), 50.0, Some(-0.01)).unwrap().target_notional, 0.0); // 손절선 너머 물타기 = 0
    let r = rec_rule_size(c.l, 2000.0, 0.0, 20.0, None, 0.0, None).unwrap();
    assert!(r.first && (r.target_notional - REC_FIRST_SHARE * c.l * 2000.0).abs() < 1e-6);
    let p = build_entry_plan(&EntryInput { side: "LONG", best_bid: 2500.0, best_ask: 2500.01, cap_notional: Some(r.cap_notional), filters: f, equity: 2000.0,
        fraction: r.fraction, fraction_of_equity: true, order_leverage: 20.0, ..Default::default() }).unwrap();
    assert!((p.notional_usdt - REC_FIRST_SHARE * c.l * 2000.0).abs() < 2500.0 * f.step + 0.01);
    let r2 = rec_rule_size(c.l, 2000.0, c.l * 2000.0 * REC_FIRST_SHARE, 20.0, None, 0.0, None).unwrap();
    assert!(!r2.first && (r2.target_notional - c.l * 2000.0 * (1.0 - REC_FIRST_SHARE)).abs() < 1e-6); // 물타기 = 남은 2/3
    let ro = rec_rule_size(c.l, 2000.0, 300.0, 20.0, Some(true), 0.0, None).unwrap(); // 반대 다리만 있음
    assert!(ro.first && (ro.target_notional - REC_FIRST_SHARE * c.l * 2000.0).abs() < 1e-6);
    let r3 = rec_rule_size(2.0, 2000.0, 4000.0, 20.0, None, 0.0, None).unwrap();
    let p3 = build_entry_plan(&EntryInput { side: "LONG", best_bid: 2400.0, best_ask: 2400.01, cap_notional: Some(r3.cap_notional), filters: f, existing_notional: 4000.0,
        equity: 2000.0, fraction: r3.fraction, fraction_of_equity: true, order_leverage: 20.0, ..Default::default() }).unwrap();
    assert!(p3.blocked.is_some()); // 상한이 찼으면 막는다
    let r10 = rec_rule_size(2.0, 2000.0, 0.0, 2.0, None, 0.0, None).unwrap(); // 레버리지 2배면 실효 L = 1.8
    assert!((r10.l_eff - 1.8).abs() < 1e-12 && (r10.cap_notional - 3600.0).abs() < 1e-6 && (r10.target_notional - REC_FIRST_SHARE * 3600.0).abs() < 1e-6);
    let full = rec_rule_size(2.0, 2000.0, 2700.0, 2.0, None, 0.0, None).unwrap();
    assert!(full.target_notional + 2700.0 <= 0.9 * 2.0 * 2000.0 + 1e-6 && full.fraction <= 0.9);
    assert_eq!(rec_rule_size(2.0, 2000.0, 0.0, 10.0, None, 0.0, None).unwrap().l_eff, 2.0);
    let z0 = rec_rule_size(2.0, 0.0, 0.0, 10.0, None, 0.0, None).unwrap(); // 잔고 조회 실패 -- 조기 반환도 l_eff 를 싣는다
    assert!(z0.l_eff == 2.0 && z0.cap_notional == 0.0);
    for bad in [0.0, -1.0, 2.5, 10.5] {
        assert!(rec_rule_size(bad, 2000.0, 0.0, 20.0, None, 0.0, None).is_err());
    }
    let rb = rec_rule_bracket("LONG", 2500.0, 1.0, RULE_STOP_PCT, &f, None, "첫 진입가").unwrap();
    assert!(rb.sl_price == Some(2375.0) && rb.backstop_price == Some(2375.0) && rb.tp_price.is_none() && rb.rule && rb.sl_name == "첫 진입가 −5%");
    let rs = rec_rule_bracket("SHORT", 2500.0, 1.0, RULE_STOP_PCT, &f, None, "첫 진입가").unwrap();
    assert!(rs.sl_price == Some(2625.0) && rs.backstop_price == Some(2625.0));
    let kept = rec_rule_bracket("LONG", 2300.0, 1.0, 0.09, &f, Some(&PrevRule { sl_price: Some(2312.5), sl_level: Some(2312.5), backstop_price: Some(2312.5), sl_name: None, rule: true }), "첫 진입가").unwrap();
    assert!(kept.sl_price == Some(2312.5) && kept.tp_price.is_none()); // 물타기해도 손절선 그대로
    let fresh = rec_rule_bracket("LONG", 2300.0, 1.0, RULE_STOP_PCT, &f, Some(&PrevRule { sl_price: Some(2400.0), backstop_price: Some(2390.0), ..Default::default() }), "첫 진입가").unwrap();
    assert_eq!(fresh.sl_price, Some(2185.0)); // 규칙 아닌 이전 SL 은 안 물려받는다
    // 2026-10-14 자동 물타기 지정가 · 10-25 상한 2
    let f2 = Filters { tick: 0.01, step: 0.001, min_qty: 0.001, min_notional: 5.0 };
    let q1 = 4000.0 / 3.0 / 2500.0;
    let ad = rec_rule_add_order("LONG", 2375.0, 0.05, 2.0, 4000.0, 4000.0, q1, 10.0, &f2).unwrap().unwrap();
    assert!(ad.price == 2481.25 && (ad.notional - 8000.0 / 3.0).abs() < 3.0);
    assert!(q1 * (2500.0 - 2375.0) + ad.qty * (ad.price - 2375.0) <= 2.0 * 0.05 * 2000.0 * (1.0 + 1e-9));
    let sd = rec_rule_add_order("SHORT", 2625.0, 0.05, 2.0, 4000.0, 4000.0, q1, 10.0, &f2).unwrap().unwrap();
    assert!(sd.price == 2518.75 && (sd.notional - 8000.0 / 3.0).abs() < 3.0);
    assert!(rec_rule_add_order("LONG", 2375.0, 0.05, 2.0, 4000.0, 4000.0 / 3.0, q1, 10.0, &f2).unwrap().is_none()); // 다른 포지션이 여유를 다 먹음
    // 2026-10-15 «손절 해제»
    let a = arm(json!({"symbol": "ETHUSDC", "armed_at": 0.0, "sl_level": 2400.0, "sl_off": {"at": 1.0}}));
    let pos: Vec<Position> = serde_json::from_value(json!([{"symbol": "ETHUSDC", "side": "LONG", "qty": 1.0}])).unwrap();
    assert_eq!(bracket_action(&a, "LONG", &pos, true, 100.0, Some((100.0, 2390.0)), None), TickAction::Hold);
    assert_eq!(bracket_action(&Armed { sl_off: false, ..a.clone() }, "LONG", &pos, true, 100.0, Some((100.0, 2390.0)), None), TickAction::Fire);
    assert_eq!(bracket_action(&a, "LONG", &[], true, 100.0, Some((100.0, 2390.0)), None), TickAction::Clear);
    let o = |v: serde_json::Value| v.as_object().unwrap().clone();
    let m = bracket_merge(&o(json!({"k": {"armed_at": 5.0, "sl_off": {"at": 9.0}}})), &o(json!({"k": {"armed_at": 5.0, "retry_after": 1}})), &o(json!({"k": 5.0})));
    assert!(m["k"]["sl_off"] == json!({"at": 9.0}) && m["k"]["retry_after"] == json!(1));
    let m = bracket_merge(&o(json!({"k": {"armed_at": 5.0}})), &o(json!({"k": {"armed_at": 5.0, "sl_off": {"at": 9.0}, "retry_after": 1}})), &o(json!({"k": 5.0})));
    assert!(m["k"].get("sl_off").is_none() && m["k"]["retry_after"] == json!(1)); // «다시 켜기» 뒤 틱의 옛 사본이 해제를 되살리지 않는다
    // 2026-10-28 선 브래킷
    let f3 = Filters { tick: 0.01, step: 0.001, min_qty: 0.0, min_notional: 0.0 };
    let lb = line_bracket("LONG", 2500.0, 2550.0, 2480.0, 0.99951, &f3, false).unwrap();
    assert!(lb.tp_price == 2548.76 && lb.sl_level == 2480.0 && lb.backstop_price.is_none() && lb.lines);
    assert_eq!(line_bracket("SHORT", 2500.0, 2450.0, 2520.0, 0.99951, &f3, false).unwrap().tp_price, 2448.79);
    assert!(line_bracket("LONG", 2500.0, 2480.0, 2550.0, 1.0, &f3, false).is_err() && line_bracket("LONG", 2500.0, 2500.0, 2480.0, 1.0, &f3, false).is_err());
    let far = line_bracket("LONG", 2500.0, 3000.0, 2370.0, 1.0, &f3, false).unwrap(); // 2026-11-14 거리 제한 제거
    assert!(far.tp_price == 3000.0 && far.sl_level == 2370.0);
    let near = line_bracket("LONG", 2500.0, 2500.01, 2499.99, 1.0, &f3, false).unwrap(); // 2026-11-11 붙인 선도 받는다
    assert!(near.tp_price == 2500.01 && near.sl_level == 2499.99);
    assert!(!line_bracket("LONG", 2500.0, 2550.0, 2500.0, 1.0, &f3, true).unwrap().sl_up);
    let la = json!({"symbol": "ETHUSDC", "armed_at": 0.0, "lines": true, "sl_level": 2480.0, "tp_level": 2550.0, "tp_resting": true});
    let with = |extra: serde_json::Value| { let mut v = la.clone(); v.as_object_mut().unwrap().extend(o(extra)); arm(v) };
    let a = arm(la.clone());
    assert!(line_hit("LONG", &a, Some((2479.9, 2551.0))) == Some(LineHit::Sl) && line_hit("LONG", &a, Some((2481.0, 2560.0))).is_none());
    assert!(line_hit("LONG", &with(json!({"tp_resting": false})), Some((2481.0, 2550.0))) == Some(LineHit::Tp) && line_hit("LONG", &a, None).is_none());
    assert_eq!(line_hit("SHORT", &with(json!({"sl_level": 2520.0, "tp_level": 2450.0})), Some((2460.0, 2520.0))), Some(LineHit::Sl));
    assert_eq!(bracket_action(&a, "LONG", &pos, true, 100.0, None, Some((2479.0, 2500.0))), TickAction::Fire);
    assert_eq!(bracket_action(&a, "LONG", &pos, true, 100.0, Some((100.0, 2479.0)), None), TickAction::Fire); // 체결 흐름이 끊겨도 5분봉 종가 이탈
    assert_eq!(bracket_action(&a, "LONG", &pos, true, 100.0, Some((100.0, 2490.0)), Some((2485.0, 2560.0))), TickAction::Hold);
    assert_eq!(bracket_action(&with(json!({"sl_off": {"at": 1.0}})), "LONG", &pos, true, 100.0, None, Some((2470.0, 2500.0))), TickAction::Hold);
    // 2026-11-11 손절선만 지움
    let lo = |extra: serde_json::Value| { let mut v = la.clone(); v.as_object_mut().unwrap().extend(o(json!({"sl_off": {"at": 1.0}, "tp_resting": false}))); v.as_object_mut().unwrap().extend(o(extra)); arm(v) };
    assert_eq!(bracket_action(&lo(json!({})), "LONG", &pos, true, 100.0, Some((100.0, 2470.0)), Some((2470.0, 2500.0))), TickAction::Hold);
    assert_eq!(bracket_action(&lo(json!({})), "LONG", &pos, true, 100.0, None, Some((2470.0, 2551.0))), TickAction::Fire);
    assert_eq!(bracket_action(&lo(json!({"tp_resting": true})), "LONG", &pos, true, 100.0, None, Some((2470.0, 2551.0))), TickAction::Hold);
    assert_eq!(bracket_action(&lo(json!({})), "LONG", &[], true, 100.0, None, None), TickAction::Clear);
    assert_eq!(bracket_action(&lo(json!({"fired": true})), "LONG", &pos, true, 100.0, None, Some((2490.0, 2500.0))), TickAction::Hold);
    assert_eq!(bracket_action(&lo(json!({"fired": "tp"})), "LONG", &pos, true, 100.0, None, Some((2490.0, 2500.0))), TickAction::Fire);
    // 2026-11-11 분할 손절
    let sll = with(json!({"sl_legs": [{"level": 2480.0, "w": 1}, {"level": 2460.0, "w": 1}]}));
    assert_eq!(bracket_action(&sll, "LONG", &pos, true, 100.0, None, Some((2479.0, 2500.0))), TickAction::SlLeg);
    assert_eq!(bracket_action(&sll, "LONG", &pos, true, 100.0, None, Some((2459.0, 2500.0))), TickAction::Fire);
    assert_eq!(bracket_action(&sll, "LONG", &pos, true, 100.0, None, Some((2485.0, 2500.0))), TickAction::Hold);
    assert_eq!(bracket_action(&sll, "LONG", &pos, true, 100.0, Some((200.0, 2479.0)), None), TickAction::SlLeg);
    assert_eq!(sl_legs_hit("LONG", &with(json!({"sl_legs": [{"level": 2480.0, "fired": true}, {"level": 2460.0}]})), None, None), vec![0]);
    assert_eq!(bracket_action(&with(json!({"sl_legs": [{"level": 2480.0}, {"level": 2460.0}], "sl_off": {"at": 1.0}})), "LONG", &pos, true, 100.0, None, Some((2459.0, 2500.0))), TickAction::Hold);
    assert_eq!(bracket_action(&with(json!({"sl_legs": [{"level": 2520.0, "up": true}, {"level": 2460.0}]})), "LONG", &pos, true, 100.0, None, Some((2490.0, 2521.0))), TickAction::SlLeg);
    // 2026-11-11 분할 익절
    let f_ = Filters { step: 0.001, tick: 0.01, min_qty: 0.001, min_notional: 20.0 };
    let legs = |v: serde_json::Value| -> Vec<TpLegIn> { serde_json::from_value(v).unwrap() };
    let q3 = split_tp_qty(1.0, &legs(json!([{"price": 2600.0, "w": 1}, {"price": 2550.0, "w": 1}, {"price": 2700.0, "w": 2}])), &f_, "LONG").unwrap();
    assert!((q3.iter().sum::<f64>() - 1.0).abs() < 1e-9 && q3 == vec![0.25, 0.25, 0.5]);
    assert_eq!(split_tp_qty(0.012, &legs(json!([{"price": 2550.0, "w": 1}, {"price": 2600.0, "w": 1}])), &f_, "LONG").unwrap(), vec![0.0, 0.012]);
    assert_eq!(split_tp_qty(0.5, &legs(json!([{"price": 2450.0, "w": 1}, {"price": 2400.0, "w": 3}])), &f_, "SHORT").unwrap(), vec![0.125, 0.375]);
    let lg = with(json!({"tp_legs": [{"level": 2550.0, "resting": true}, {"level": 2600.0, "resting": false}]}));
    assert!(line_hit("LONG", &lg, Some((2481.0, 2560.0))).is_none() && line_hit("LONG", &lg, Some((2481.0, 2600.0))) == Some(LineHit::Tp));
    assert!(line_hit("LONG", &with(json!({"tp_legs": [{"level": 2550.0, "resting": true}, {"level": 2600.0, "resting": false, "merged": true}]})), Some((2481.0, 2650.0))).is_none());
    // 2026-11-11 익절선 지움(tp_off)
    let to = with(json!({"tp_off": {"at": 1.0}, "tp_resting": false}));
    assert!(line_hit("LONG", &to, Some((2481.0, 2600.0))).is_none() && line_hit("LONG", &to, Some((2479.0, 2600.0))) == Some(LineHit::Sl));
    assert_eq!(bracket_action(&Armed { sl_off: true, ..to }, "LONG", &pos, true, 100.0, Some((100.0, 2470.0)), Some((2470.0, 2600.0))), TickAction::Hold);
    assert_eq!(bracket_action(&with(json!({"fired": true})), "LONG", &pos, true, 100.0, None, Some((2490.0, 2500.0))), TickAction::Fire); // 익절을 지운 뒤 청산이 밀려도 닫힐 때까지
}

#[test]
fn exec_gate_reads_env_values() {
    // G3: 1·true·yes(대소문자·공백 무시)만 켠다. 그 밖·없음 = 꺼짐.
    for (v, want) in [(Some("1"), true), (Some(" TRUE "), true), (Some("yes"), true), (Some("0"), false), (Some("on"), false), (Some(""), false), (None, false)] {
        assert_eq!(exec_enabled_from(v), want, "{v:?}");
    }
}
