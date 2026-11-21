//! 감시 불변식(2026-11-20 셋째 조각) -- 가짜 바깥(ScriptIo)으로:
//! ① 닿았으면 그 틱에 청산 집행이 시작된다(잠금·스위치·재시도 대기·칸 사용 중은 원본대로 예외) ② 같은 코인 청산이 도는 칸 위에 감시가 청산을 겹쳐 시작하지 않는다
//! ③ 감시는 어떤 실패에도 죽지 않는다(한 틱 실패는 다음 틱에) ④ 저장본이 거래소와 어긋난 채 «정상»으로 남지 않는다(청산 실패 = 무장 유지 + 재시도 시각·사유).
//! 그리고 감시 루프(주기 주입 · 알림에 일찍 깸 · 예외 삼킴) · 재시작(청산 도중 죽음 → 새 프로세스가 무장을 이어 받음) · 주문 잠금 `busy_or`.

include!("common/watch_io.rs");

const T0: f64 = 1_800_000_000.0;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f64 {
        (self.next() % 1_000_000) as f64 / 1_000_000.0
    }
}

fn armed(rng: &mut Rng, sym: &str, side: &str, lines: bool) -> Value {
    let long = side == "LONG";
    let (sl, tp) = if long { (2480.0, 2550.0) } else { (2520.0, 2450.0) };
    json!({"symbol": sym, "side": side, "sl_price": sl, "sl_level": sl, "tp_price": if lines { json!(tp) } else { Value::Null }, "tp_level": if lines { json!(tp) } else { Value::Null },
           "backstop_price": if lines { Value::Null } else { json!(sl) }, "rule": !lines, "lines": lines, "sl_up": Value::Null,
           "tp_resting": lines && rng.f() < 0.5, "armed_at": T0 - 600.0, "placed": {}})
}

fn plan_for(sym: &str, side: &str) -> Value {
    json!({"symbol": sym, "side": if side == "LONG" { "SELL" } else { "BUY" }, "positionSide": side, "type": "LIMIT", "price": 2500.0, "quantity": 0.5,
           "fallback_after_sec": 60.0, "blocked": Value::Null})
}

/// 이 틱의 판정을 정답 함수(plan::bracket_action)로 따로 계산한다 -- 감시가 입력을 제대로 모았는지는 정답지 재생이 본다.
fn expected_act(b: &Value, acct: &Value, sec: &[(f64, f64)], bar: Option<(f64, f64)>) -> plan::TickAction {
    let a = plan::Armed::from_value(b);
    let since = a.armed_at.max(a.watch_from);
    let pts: Vec<f64> = sec.iter().filter(|x| x.0 > since).map(|x| x.1).collect();
    let touch = if a.lines && !pts.is_empty() { Some((pts.iter().copied().fold(f64::INFINITY, f64::min), pts.iter().copied().fold(f64::NEG_INFINITY, f64::max))) } else { None };
    let pos: Vec<plan::Position> = acct["positions"].as_array().unwrap().iter().map(|p| serde_json::from_value(p.clone()).unwrap()).collect();
    plan::bracket_action(&a, b["side"].as_str().unwrap(), &pos, acct["ok"].as_bool().unwrap(), parse_iso(acct["generated_at"].as_str().unwrap()).unwrap(), bar, touch)
}

#[test]
fn fuzz_fire_starts_exit_never_overlaps_never_dies_never_silently_ok() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut fired_ticks, mut exits, mut blocked_by_lock, mut errs) = (0, 0, 0, 0);
    for case in 0..300 {
        let dir = std::env::temp_dir().join(format!("rtorder_watch_fuzz_{}_{case}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut st = Map::new();
        let keys: Vec<(&str, &str)> = [("ETHUSDC", "LONG"), ("ETHUSDC", "SHORT"), ("SOLUSDC", "LONG")].into_iter().filter(|_| rng.f() < 0.7).collect();
        for (sym, side) in &keys {
            let lines = rng.f() < 0.6;
            st.insert(format!("{sym}:{side}"), armed(&mut rng, sym, side, lines));
        }
        bracket_save(&dir, &st).unwrap();
        let mut w = Watcher::new(&dir, "ETHUSDC");
        let mut io = ScriptIo::new(&dir);
        for tick in 0..20 {
            io.t = T0 + 2.0 * tick as f64;
            io.lock = rng.f() < 0.2;
            io.exec_on = rng.f() < 0.9;
            // 다른 주문이 칸을 쓰는 중(같은 코인 청산 · 다른 코인 진입)
            w.entry = match rng.next() % 4 { 0 => serde_json::from_value(json!({"phase": "working", "asset": "eth", "kind": "exit"})).unwrap(),
                                             1 => serde_json::from_value(json!({"phase": "working", "asset": "sol"})).unwrap(),
                                             _ => serde_json::from_value(json!({"phase": "idle"})).unwrap() };
            let held: Vec<Value> = keys.iter().filter(|_| rng.f() < 0.9).map(|(s, d)| json!({"symbol": s, "side": d, "qty": 0.5})).collect();
            io.account = json!({"ok": rng.f() < 0.9, "generated_at": iso_utc(io.t - rng.f() * 3.0), "positions": held});
            let px = |r: &mut Rng| 2440.0 + r.f() * 120.0;
            let sec: Vec<(f64, f64)> = (0..3).map(|i| (io.t - i as f64, px(&mut rng))).collect();
            let bar = (rng.f() < 0.5).then(|| (io.t, px(&mut rng)));
            io.sec = json!({"eth": sec.iter().map(|x| json!([x.0, x.1])).collect::<Vec<_>>(), "sol": sec.iter().map(|x| json!([x.0, x.1 / 13.8])).collect::<Vec<_>>()});
            io.bars = json!({"eth": bar.map_or(json!([]), |b| json!([{"time": b.0 - 300.0, "close": b.1}])), "sol": []});
            let mut q = Map::new();
            let n = 8;
            q.insert("clear_bracket".into(), json!((0..n).map(|_| if rng.f() < 0.2 { json!({"ret": {"errors": ["500 x"]}}) } else { json!({"ret": {"cancelled": 1}}) }).collect::<Vec<_>>()));
            q.insert("exit_plan".into(), json!((0..n).map(|_| match rng.next() % 10 {
                0 => json!({"err": {"error": "account_unavailable", "detail": "계좌 조회 실패"}}),
                1 => json!({"err": {"error": "no_position"}}),
                _ => json!({"plan": plan_for("ETHUSDC", "LONG")}),
            }).collect::<Vec<_>>()));
            q.insert("run_exit".into(), json!((0..n).map(|_| match rng.next() % 10 {
                0..=3 => json!({"dt": rng.f() * 60.0, "slot": {"phase": "filled_maker", "filled": 0.5}}),
                4..=6 => json!({"dt": rng.f() * 60.0, "slot": {"phase": "rejected", "error": "400 x"}}),
                7 | 8 => json!({"dt": 1.0, "slot": {"phase": "error", "error": "취소 확인 못 함"}}),
                _ => json!({"dt": 0.5, "raise": ["TypeError", "boom"]}),
            }).collect::<Vec<_>>()));
            q.insert("place_bracket".into(), json!((0..n).map(|_| json!({"ret": {"placed": true, "cleared": {}}})).collect::<Vec<_>>()));
            q.insert("place_tp_ladder".into(), json!((0..n).map(|_| json!({"ret": {"placed": true}})).collect::<Vec<_>>()));
            io.script = q;
            let before = bracket_load(&dir);
            let entry_before = w.entry.clone();
            io.calls.clear();
            let r = block_on(w.bracket_tick(&mut io));
            if r.is_err() {
                errs += 1;
            }
            let after = bracket_load(&dir);
            let calls = io.calls.clone();
            let runs: Vec<&Value> = calls.iter().filter(|c| c[1] == "run_exit").collect();
            // ② 같은 코인: 위 칸이 그 코인 청산으로 바쁘면 감시는 그 코인에 청산을 시작하지 않는다(G6) · 한 틱에 한 무장 한 번
            if busy(&entry_before) && entry_before["asset"] == "eth" {
                assert!(runs.iter().all(|c| c[3]["asset"] != "eth"), "case {case} tick {tick}: ETH 칸이 바쁜데 ETH 청산을 시작함 {runs:?}");
            }
            let mut per_key = std::collections::HashMap::new();
            for c in &runs {
                *per_key.entry(format!("{}:{}", c[3]["asset"], c[3]["side"])).or_insert(0) += 1;
            }
            assert!(per_key.values().all(|n| *n == 1), "case {case} tick {tick}: 한 틱에 같은 무장 청산 두 번 {per_key:?}");
            // G9·D9(원본 그대로): 사용자 주문이 잠금을 쥐고 있으면 감시는 청산을 시작하지 않는다(다음 틱에) -- 겹치면 이중 청산
            assert!(!io.lock || runs.is_empty(), "case {case} tick {tick}: 주문 잠금 중 청산 시작 {runs:?}");
            if !io.exec_on {
                assert!(calls.is_empty() && after == before, "case {case}: 스위치 꺼짐인데 무언가 함 {calls:?}");
                continue;
            }
            for (key, b) in &before {
                let bar_v = bar.map(|b| (b.0, b.1));
                let act = expected_act(b, &io.account, &sec_for(&io, b), bar_v.filter(|_| b["symbol"] == "ETHUSDC"));
                let retry = b.get("retry_after").and_then(Value::as_f64).unwrap_or(0.0);
                let ran = runs.iter().any(|c| c[3]["trigger"].as_str().is_some() && Some(c[3]["side"].as_str().unwrap()) == b["side"].as_str()
                    && c[3]["asset"] == if b["symbol"] == "SOLUSDC" { "sol" } else { "eth" });
                let coin_busy = busy(&entry_before) && entry_before["asset"] == if b["symbol"] == "SOLUSDC" { "sol" } else { "eth" };
                if act == plan::TickAction::Fire && r.is_ok() {
                    fired_ticks += 1;
                    if io.lock && !ran {
                        blocked_by_lock += 1;
                    }
                    // ① 막는 것이 없으면 그 틱에 청산 시도(계획 조립)가 시작된다 -- 계획이 나오면 run_exit 까지
                    let plan_calls = calls.iter().filter(|c| c[1] == "exit_plan" && c[2] == b["side"] && c[7] == b["symbol"]).count();
                    let tp_cancel_failed = after.get(key).and_then(|x| x.get("last_error")).and_then(Value::as_str).is_some_and(|s| s.starts_with("익절 지정가 취소 실패"));
                    if !io.lock && !coin_busy && retry == 0.0 && !tp_cancel_failed {
                        assert!(plan_calls >= 1, "case {case} tick {tick} {key}: 닿았는데 청산 시도 없음 calls={calls:?}");
                    }
                }
                if ran {
                    exits += 1;
                }
                // ④ 저장본이 «정상»으로 남지 않는다: 청산을 돌렸는데 다 닫히지 않았으면 무장 유지 + 재시도 시각
                if ran {
                    match after.get(key) {
                        None => assert!(calls.iter().any(|c| c[1] == "clear_bracket" && c[2] == b["symbol"]), "case {case} {key}: 정리 없이 무장이 사라짐"),
                        Some(a) => {
                            // 남았으면: 청산 실패(재시도 시각 + 사유) 또는 다 닫혔는데 정리 실패(다음 틱 clear 가 다시 -- 정리 호출이 있었다)
                            let cleanup_tried = calls.iter().filter(|c| c[1] == "clear_bracket" && c[2] == b["symbol"]).count() >= 1;
                            assert!((a.get("retry_after").and_then(Value::as_f64).is_some_and(|x| x > retry) && a.get("last_error").is_some()) || cleanup_tried,
                                    "case {case} {key}: 청산이 안 끝났는데 저장본이 «정상» {a}");
                        }
                    }
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    println!("퍼즈: 닿은 틱 {fired_ticks} · 청산 시작 {exits} · 잠금으로 미룸 {blocked_by_lock} · 틱 밖으로 샌 오류 {errs}");
    assert!(fired_ticks > 300 && exits > 100 && blocked_by_lock > 10, "퍼즈가 실제로 쏘는 길을 돌지 않았다");
}

fn sec_for(io: &ScriptIo, b: &Value) -> Vec<(f64, f64)> {
    let a = if b["symbol"] == "SOLUSDC" { "sol" } else { "eth" };
    io.sec[a].as_array().unwrap().iter().map(|x| (x[0].as_f64().unwrap(), x[1].as_f64().unwrap())).collect()
}

/// ③ 감시 루프: 주기 주입 · 알림에 일찍 깸 · 틱이 매번 실패해도(계좌 예외 · 각본 없음) 루프는 끝까지 돈다.
#[test]
fn watch_loop_survives_every_failure_and_uses_injected_period() {
    let dir = std::env::temp_dir().join(format!("rtorder_watch_loop_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut st = Map::new();
    st.insert("ETHUSDC:LONG".into(), armed(&mut Rng(7), "ETHUSDC", "LONG", false));
    bracket_save(&dir, &st).unwrap();
    let mut w = Watcher::new(&dir, "ETHUSDC");
    let mut io = ScriptIo::new(&dir);
    io.account = json!({"raise": ["ClientConnectorError", "down"]});
    io.max_naps = 50;
    io.wakes = vec![T0 + 0.2, T0 + 3.0];
    block_on(w.watch_brackets(&mut io, 0.5));
    let naps: Vec<f64> = io.calls.iter().filter(|c| c[1] == "nap").map(|c| c[0].as_f64().unwrap()).collect();
    assert_eq!(naps.len(), 51, "루프가 중간에 멈췄다");
    assert!(io.calls.iter().filter(|c| c[1] == "nap").all(|c| c[2] == json!(0.5)), "주기 주입이 안 먹음");
    let ticks: Vec<f64> = io.calls.iter().filter(|c| c[1] == "account").map(|c| c[0].as_f64().unwrap()).collect();
    assert_eq!(ticks.len(), 50);
    assert_eq!(ticks[0], T0 + 0.2, "체결 알림에 일찍 깨지 않음");
    assert_eq!(ticks[1], T0 + 0.7);
    assert_eq!(w.logs.len(), 50);
    assert!(w.logs.iter().all(|l| l == "bracket watch: ClientConnectorError: down"));
    // 각본이 비어 청산 계획마다 예외가 나도 계속
    io.account = json!({"ok": true, "generated_at": iso_utc(T0), "positions": [{"symbol": "ETHUSDC", "side": "LONG", "qty": 0.5}]});
    io.bars = json!({"eth": [{"time": T0, "close": 2000.0}]});
    io.naps = 0;
    w.logs.clear();
    block_on(w.watch_brackets(&mut io, 0.5));
    assert_eq!(io.calls.iter().filter(|c| c[1] == "account").count(), 100, "두 번째 묶음도 50틱");
    let _ = std::fs::remove_dir_all(&dir);
}

fn poll_once<F: Future>(f: F) -> bool {
    let mut f = std::pin::pin!(f);
    matches!(f.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Ready(_))
}

/// 재시작: 선 손절이 닿아 청산을 시작한 프로세스가 청산 도중 죽는다 → 새 프로세스(메모리 없음)가 재시작 정리 뒤 감시를 이어 받아 닫는다.
#[test]
fn restart_mid_exit_keeps_arming_and_next_process_finishes() {
    let dir = std::env::temp_dir().join(format!("rtorder_watch_restart_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut st = Map::new();
    let mut a = armed(&mut Rng(3), "ETHUSDC", "LONG", true);
    a["tp_resting"] = json!(true);
    st.insert("ETHUSDC:LONG".into(), a);
    bracket_save(&dir, &st).unwrap();
    let acct = json!({"ok": true, "generated_at": iso_utc(T0), "positions": [{"symbol": "ETHUSDC", "side": "LONG", "qty": 0.5}]});
    // 첫 프로세스: 닿음 → 익절 지정가 지움 → fired 저장 → 계획 → run_exit 에서 죽음
    {
        let mut w = Watcher::new(&dir, "ETHUSDC");
        let mut io = ScriptIo::new(&dir);
        io.account = acct.clone();
        io.sec = json!({"eth": [[T0, 2470.0]]});
        io.hang_run_exit = true;
        io.script = serde_json::from_value(json!({"clear_bracket": [{"ret": {"cancelled": 1}}], "exit_plan": [{"plan": plan_for("ETHUSDC", "LONG")}]})).unwrap();
        assert!(!poll_once(w.bracket_tick(&mut io)), "청산이 멈춘 채여야 한다");
        assert_eq!(io.calls.last().unwrap()[1], "run_exit");
    } // ← 프로세스 사망(메모리 상태·bracket_exiting·진행 칸 전부 사라짐, R9)
    let saved = bracket_load(&dir);
    assert_eq!(saved["ETHUSDC:LONG"]["fired"], json!(true), "발동은 청산 전에 파일에 남아야 한다(B14)");
    assert_eq!(saved["ETHUSDC:LONG"]["tp_resting"], json!(false));
    // 두 번째 프로세스: 재시작 정리(죽은 따라가기 dbpg 지정가를 지움) → 감시 틱이 fired 를 보고 익절 취소 없이 바로 청산 → 닫힘 → 정리
    let mut w = Watcher::new(&dir, "ETHUSDC");
    let mut io = ScriptIo::new(&dir);
    io.t = T0 + 30.0;
    io.account = acct;
    io.sec = json!({"eth": []});
    io.script = serde_json::from_value(json!({
        "cancel_wait_orphans": [{"ret": {"cancelled": [{"symbol": "ETHUSDC", "cid": "dbpgL1800000000001", "order_id": 5}], "errors": []}}],
        "exit_plan": [{"plan": plan_for("ETHUSDC", "LONG")}], "run_exit": [{"dt": 10.0, "slot": {"phase": "filled_maker", "filled": 0.5}}],
        "clear_bracket": [{"ret": {"cancelled": 0}}]})).unwrap();
    block_on(w.wait_cleanup(&mut io)).unwrap();
    assert_eq!(w.waits["eth"]["phase"], "restart_cancelled");
    block_on(w.bracket_tick(&mut io)).unwrap();
    let names: Vec<&str> = io.calls.iter().map(|c| c[1].as_str().unwrap()).collect();
    assert_eq!(names, ["cancel_wait_orphans", "account", "bars", "sec", "exit_plan", "run_exit", "clear_bracket"], "{:?}", io.calls);
    assert!(bracket_load(&dir).is_empty(), "다 닫히면 무장을 내린다");
    let _ = std::fs::remove_dir_all(&dir);
}

/// G8: 주문 잠금이 잡혀 있으면 즉시 None(409) · 풀려 있으면 잠근 채 돌린다 · 감시는 같은 잠금을 «잠김»으로 본다(G9).
#[test]
fn busy_or_refuses_while_locked() {
    let lock = tokio::sync::Mutex::new(());
    assert_eq!(block_on(busy_or(&lock, async { 1 })), Some(1));
    let g = lock.try_lock().unwrap();
    assert_eq!(block_on(busy_or(&lock, async { 2 })), None);
    drop(g);
    let inner = block_on(busy_or(&lock, async { lock.try_lock().is_err() }));
    assert_eq!(inner, Some(true), "핸들러가 도는 동안 잠겨 있어야 한다");
}

#[test]
fn tp_leg_filled_from_stream_queue_drops_that_leg_only() {
    // 원본 tp_leg_filled: 스트림 «우리 dbtp 체결» → 그 조각만 무장에서 뺀다 · 모르는 표식·다른 측면은 그대로
    let dir = std::env::temp_dir().join(format!("rtorder_tpfill_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut w = rtorder::watch::Watcher::new(&dir, "ETHUSDC");
    let leg = |lv: f64, cid: &str| serde_json::json!({"level": lv, "price": lv, "w": 1, "resting": true, "cid": cid, "order_id": 1, "qty": 0.2});
    let b = serde_json::json!({"symbol": "ETHUSDC", "side": "LONG", "tp_legs": [leg(2600.0, "dbtpLa"), leg(2650.0, "dbtpLb"), leg(2700.0, "dbtpLc")], "armed_at": T0});
    w.save(&serde_json::from_value(serde_json::json!({"ETHUSDC:LONG": b})).unwrap()).unwrap();
    let q = std::sync::Mutex::new(vec![("ETHUSDC".to_string(), "LONG".to_string(), "dbtpLa".to_string()),
                                       ("ETHUSDC".to_string(), "LONG".to_string(), "dbtpLzz".to_string()),
                                       ("ETHUSDC".to_string(), "SHORT".to_string(), "dbtpLb".to_string())]);
    w.drain_tp_fills(&q).unwrap();
    assert!(q.lock().unwrap().is_empty());
    let cur = w.load()["ETHUSDC:LONG"].clone();
    let cids: Vec<&str> = cur["tp_legs"].as_array().unwrap().iter().map(|x| x["cid"].as_str().unwrap()).collect();
    assert_eq!(cids, ["dbtpLb", "dbtpLc"]);
    assert_eq!(cur["tp_level"], serde_json::json!(2650.0), "가장 가까운 살아 있는 조각");
    assert_eq!(w.logs, ["bracket LONG: 분할 익절 조각 체결 dbtpLa -- 남은 조각 2"]);
    let _ = std::fs::remove_dir_all(&dir);
}
