"""rtorder 양성 대조(2026-11-20): 규칙을 하나씩 일부러 틀리게 바꿔 시험이 실패하는지 본다.
원본 소스는 건드리지 않는다 -- rtorder/target/mutwork 에 사본을 떠서 거기만 바꾸고, 끝에 원본 해시가 그대로인지 확인한다.
실행(저장소 뿌리): ( ulimit -v 4000000; timeout 1800 python3 -I rtorder/tools/mutate.py )
"""
import hashlib, os, shutil, subprocess, sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
W = SRC / "target" / "mutwork"
P, X, F = "src/plan.rs", "src/exec.rs", "src/pyfmt.rs"
S, H, B = "src/sign.rs", "src/http.rs", "src/bracket.rs"
W_ = "src/watch.rs"
PLAN = [  # (이름, 파일, 원래, 바꾼 것) -- plan_fixture 로 잡혀야 한다
    ("내림 경계 +1e-9 → -1e-9", P, "Ok(int_trunc(value / grid + 1e-9)? * grid)", "Ok(int_trunc(value / grid - 1e-9)? * grid)"),
    ("올림 경계 -1e-9 → +1e-9", P, "Ok(py_ceil(value / grid - 1e-9)? * grid)", "Ok(py_ceil(value / grid + 1e-9)? * grid)"),
    ("롱 진입가 내림 → 올림", P, 'if a.side == "LONG" { floor_to(a.best_bid, f.tick)? }', 'if a.side == "LONG" { ceil_to(a.best_bid, f.tick)? }'),
    ("상한 여유 room <= 0 → < 0", P, "if room <= 0.0 {", "if room < 0.0 {"),
    ("가용 잔고 버퍼 (1-2%) → (1+2%)", P, "(1.0 - MARGIN_BUFFER)", "(1.0 + MARGIN_BUFFER)"),
    ("비율을 상한 전 권고에 곱함", P, "qty = available_qty * a.fraction;", "qty = pyfmt::max(0.0, a.recommended_qty) * a.fraction;"),
    ("청산 수량 min → max", P, "pyfmt::min(floor_to(want, f.step)?, floor_to(held, f.step)?)", "pyfmt::max(floor_to(want, f.step)?, floor_to(held, f.step)?)"),
    ("청산 시장가 문턱 >= → >", P, "v >= EXIT_TAKER_VOL_BPM", "v > EXIT_TAKER_VOL_BPM"),
    ("clear 여유 무장+5초 → +0초", P, "account_ts > armed.armed_at + 5.0", "account_ts > armed.armed_at + 0.0"),
    ("선 손절 닿음 >= → >", P, "if if up { hi >= sl } else { lo <= sl } {", "if if up { hi > sl } else { lo < sl } {"),
    ("선 익절가 롱 올림 → 내림", P, "if long { py_ceil(tp / tick - 1e-9)? } else", "if long { py_floor(tp / tick + 1e-9)? } else"),
    ("분할 익절 carry 버림", P, "carry = raw - q;", "carry = 0.0;"),
    ("merge: 쏜 틱의 tp_resting 규칙 뒤집음", P, 'if !(py_truthy(m.get("fired")) && !py_truthy(cur.get("fired"))) {', 'if py_truthy(m.get("fired")) && !py_truthy(cur.get("fired")) {'),
    ("파이썬 max 동점 → 뒤의 것(-0.0)", F, "if b > a { b } else { a }", "if b >= a { b } else { a }"),
    ("repr 지수 경계 16 → 17", F, "if (-4..16).contains(&exp) {\n        let point", "if (-4..17).contains(&exp) {\n        let point"),
    ("round(x,8) → 7자리", F, 'format!("{:.*}", n, x).parse().unwrap()', 'format!("{:.*}", n.min(7), x).parse().unwrap()'),
    ("int(round()) 동점 짝수 → 0.5 올림", F, "x.round_ties_even()", "x.round()"),
    ("0 나누기를 inf 로 흘림", P, "if b == 0.0 { Err(PlanError::ZeroDivision) } else { Ok(a / b) }", "Ok(a / b)"),
]
EXEC = [  # exec_invariants 로 잡혀야 한다
    ("취소 확인 실패해도 계속 재호가(2026-09-13 41개)", X, "None => return (r8(done + this_filled), price, repegs, Some((Phase::Error, cancel_msg(oid)))),", "None => {}"),
    ("PARTIALLY_FILLED 를 종료로", X, 'pub const TERMINAL: [&str; 4] = ["FILLED", "CANCELED", "EXPIRED", "REJECTED"];', 'pub const TERMINAL: [&str; 5] = ["FILLED", "CANCELED", "EXPIRED", "REJECTED", "PARTIALLY_FILLED"];'),
    ("-5022 판정이 코드 앞머리만 봄(09-20)", X, 'err.contains("5022") || err.contains("could not be executed as maker")', 'err.starts_with("-5022")'),
    ("앞지름 부등호 뒤집음", X, 'if order_side == "BUY" { bid > price } else { ask < price }', 'if order_side == "BUY" { bid < price } else { ask > price }'),
    ("알림에 깨도 매번 리페그 판단(E16)", X, "if ex.now() - last_chk < POLL_SEC {", "if ex.now() - last_chk < 0.0 {"),
    ("불명 접수의 표식 조회 빼기(E6 · 네트워크 오류를 확정 거부로)", X, "if !(!e0.starts_with(|ch: char| ch.is_ascii_digit()) ||", "if !(false ||"),
    ("잔량 대신 전량 재주문", X, "let remaining = r8(total - done);\n        let cid = format!(\"{PEG_TAG}", "let remaining = total;\n        let cid = format!(\"{PEG_TAG}"),
    ("부분 체결 뒤에도 수정 적용(D4)", X, "if g.action.is_none() && done == 0.0 {", "if g.action.is_none() {"),
    ("감시 진입 maker_only 무시", X, "if plan.maker_only {", "if false {"),
    ("on_fill 을 매 조회마다(지정가 대기)", X, "            if this_filled > 0.0 && on_fill {\n                ex.on_fill().await;\n                on_fill = false;", "            if this_filled > 0.0 && on_fill {\n                ex.on_fill().await;"),
    ("오류 때 마지막 체결량 버림(E9)", X, "None => return (r8(done + this_filled), price, repegs, Some((Phase::Error, cancel_msg(oid)))),", "None => return (r8(done), price, repegs, Some((Phase::Error, cancel_msg(oid)))),"),
    ("지정가 대기 5xx 를 확정 거부로", X, "|| e0.starts_with('5') || e0.contains(\"-1007\")", "|| e0.contains(\"-1007\")"),
    ("대기 취소를 무시하고 따라가기 계속", X, "let cancel = |s: &Shared| st(s).action == Some(Action::Cancel);", "let cancel = |_s: &Shared| false;"),
    ("감시 진입: 선 옮겨도 닿음 판정 시작점 그대로", X, "since = ex.wall_ms() as f64 / 1000.0;\n            let Some(q)", "let Some(q)"),
    ("감시 진입: 닿은 뒤 maker_only 빠짐(시장가 전락)", X, "run_entry(ex, &ExecPlan { price, maker_only: true, ..plan }, s).await;", "run_entry(ex, &ExecPlan { price, maker_only: false, ..plan }, s).await;"),
]
HTTP = [  # http_replay(원본 파이썬 + aiohttp 정답지)로 잡혀야 한다
    ("quote_plus 빈칸 + → %20", S, "b' ' => o.push('+'),", "b' ' => o.push_str(\"%20\"),"),
    ("quote_plus ~ 를 안전 글자에서 뺌", S, "| b'-' | b'~' =>", "| b'-' =>"),
    ("recvWindow·timestamp 순서 바꿈", S, "    all.push((\"recvWindow\", P::I(RECV_WINDOW_MS)));\n    all.push((\"timestamp\", P::I(ts)));", "    all.push((\"timestamp\", P::I(ts)));\n    all.push((\"recvWindow\", P::I(RECV_WINDOW_MS)));"),
    ("오류 문자열에서 code 버림(09-20)", S, "Some(c) => format!(\"[{}] \", py_str(c)),", "Some(_) => String::new(),"),
    ("msg 없을 때 payload repr 대신 None", S, ".unwrap_or_else(|| py_repr(payload));", ".unwrap_or_else(|| \"None\".into());"),
    ("418 만 밴 기록(429 빠짐)", S, "if status != 418 && status != 429 {", "if status != 418 {"),
    ("GET·DELETE 끊김 한 번 더 안 보냄", H, "let mut retry = IDEMPOTENT.contains(&method);", "let mut retry = false;"),
    ("POST 끊김도 다시 보냄(주문 두 번)", H, "let mut retry = IDEMPOTENT.contains(&method);", "let mut retry = true;"),
    ("빈 본문을 None 대신 오류로", H, "return Ok(Value::Null);", "return Err(\"JSONDecodeError: empty\".into());"),
    ("주문 번호 조회에서 side·positionSide 뺌", H, "(\"side\", P::s(c.side)), (\"positionSide\", P::s(c.position_side)), (\"orderId\"", "(\"orderId\""),
    ("시계 보정 부호 뒤집음", H, "s as i64 - sign::ms(self.time())", "sign::ms(self.time()) - s as i64"),
    ("익절만 지우기인데 물타기도 지움", B, "(!only_tp && cid.starts_with(&add_pre))", "cid.starts_with(&add_pre)"),
    ("strict 무시(같은 쪽 전량 익절 둘)", B, "if strict && truthy(cleared.get(\"errors\")) {", "if false {"),
    ("사라진 조각을 조회 없이 체결로(11-12)", B, "json!(o.is_object() && o.get(\"status\").and_then(Value::as_str) == Some(\"FILLED\"))", "json!(true)"),
    ("레버리지 같아도 다시 보냄", B, "if now == Some(target) {", "if false {"),
    ("수수료 실패 캐시 5분 → 1시간", B, "if src == \"live\" { FEE_TTL_SEC } else { FEE_RETRY_SEC }", "FEE_TTL_SEC"),
    ("수수료 stale 대신 fallback", B, "hit.filter(|h| h.3 != \"fallback\")", "None::<(f64, f64, f64, &str)>"),
    ("고아 정리가 dbtp 도 지움", B, "|| cid.starts_with(crate::exec::PEG_TAG) {", "|| cid.starts_with(crate::exec::PEG_TAG) || cid.starts_with(\"dbtp\") {"),
]

WATCH = [  # watch_fixture(원본 orders.py 감시 클로저 정답지)로 잡혀야 한다
    ("스위치 꺼져도 감시 틱(G4)", W_, "if st.is_empty() || !io.exec_enabled() {", "if st.is_empty() {"),
    ("선 발동 전 잠금 무시(D9·G9)", W_, "if io.order_locked() || (t(&obj_or_empty(self.load().get(key)), \"sl_off\") && !t(b, \"sl_off\")) {", "if (t(&obj_or_empty(self.load().get(key)), \"sl_off\") && !t(b, \"sl_off\")) {"),
    ("익절 취소 실패 대기 10초 → 30초", W_, "f(io.now() + TP_CANCEL_RETRY_SEC)", "f(io.now() + RETRY_SEC)"),
    ("fired 를 청산 전에 파일에 안 씀(B14)", W_, "c.insert(\"tp_resting\".into(), json!(false));\n                    self.save(&st2)?;", "c.insert(\"tp_resting\".into(), json!(false));"),
    ("선 수정(mod.at) 비교 빼기(B15)", W_, "\n                || !py_eq(obj_or_empty(fresh.get(\"mod\")).get(\"at\"), obj_or_empty(b.get(\"mod\")).get(\"at\"))", ""),
    ("조립 뒤 손절 해제 검사 빼기(B16)", W_, "if t(&obj_or_empty(self.load().get(key)), \"sl_off\") && !(t(b, \"lines\") && t(b, \"sl_off\")) {", "if false {"),
    ("조회 실패도 «닫힘»으로 정리(B17)", W_, "Some(\"no_position\") => {", "Some(\"no_position\") || true => {"),
    ("청산 실패 뒤 재시도 시각 안 둠(B18)", W_, "b.insert(\"retry_after\".into(), f(io.now() + RETRY_SEC)); // 🔴불명", "// 🔴불명"),
    ("clear 정리 실패에도 무장 내림(B10)", W_, "b.insert(\"retry_after\".into(), f(io.now() + RETRY_SEC));\n                return Ok((true, true));", "return Ok((false, true));"),
    ("merge 대신 통째로 저장(B19)", W_, "self.save(&bracket_merge(&self.load(), &st, &seen_at))?;", "self.save(&st)?;"),
    ("같은 쪽 대기 진입 취소 빼기(W9)", W_, "ws.insert(\"action\".into(), json!(\"cancel\"));", ""),
    ("분할 손절 쏜 조각 표시 안 함(B20)", W_, "m.insert(\"fired\".into(), json!(true)); Value::Object(m)", "Value::Object(m)"),
    ("분할 손절 비율 분모 무시", W_, "let frac = num / if den != 0.0 { den } else { 1.0 };", "let frac = num;"),
    ("부분 청산 뒤 익절 strict(D7 을 몰래 고침)", W_, "io.place_bracket(&Value::Object(spec), &symbol, &pside, false)", "io.place_bracket(&Value::Object(spec), &symbol, &pside, true)"),
    ("일부 취소 실패를 서버 감시로 안 넘김(B29)", W_, "self.tp_watch_only(io, &key, &b)?;\n                return Ok(());", "return Ok(());"),
    ("고아 정리 실패여도 복구 파일 지움(W5)", W_, "if truthy(Some(orphan_errors)) {", "if false {"),
    ("6시간 넘은 복구 파일도 복구", W_, "> LIMIT_REC_MAX_AGE", "> f64::INFINITY"),
    ("같은 코인 칸이 바빠도 청산 칸 줌(G6)", W_, "if ea == json!(asset) { None } else { Some(SlotId::Side(asset.into())) }", "{ let _ = ea; Some(SlotId::Side(asset.into())) }"),
    ("미마감 5분봉도 씀(B11)", W_, "if tm + BAR_SEC <= now_s {", "if tm <= now_s {"),
    ("무장 전 1초 가격도 씀(B12)", W_, ".unwrap_or_default().into_iter().filter(|x| x.0 > since)", ".unwrap_or_default().into_iter().filter(|x| x.0 > 0.0)"),
    ("발동 로그 문구", W_, "이탈 -> {}", "이탈 => {}"),
]
WATCH_EXEC = [  # watch_exec(실제 집행 + 가짜 거래소)로 잡혀야 한다
    ("시장가 거부를 체결로 봄", X, "            (g.taker_qty, g.error) = (0.0, Some(e));\n            drop(g);\n            finish(s, Phase::TakerFailed);\n        }\n        Ok(o) => {\n            let done", "            (g.taker_qty, g.error) = (0.0, Some(e));\n            drop(g);\n            finish(s, Phase::FilledTaker);\n        }\n        Ok(o) => {\n            let done"),
]
WATCH_INV = [  # watch_invariants(퍼즈·루프·재시작)로 잡혀야 한다
    ("조립 뒤 잠금 무시(D9·G9)", W_, "if io.order_locked() || slot.is_none() {\n            return Ok((true, changed)); // D9", "if slot.is_none() {\n            return Ok((true, changed)); // D9"),
    ("감시 루프가 틱 실패에 멈춤(B8)", W_, "self.log(format!(\"bracket watch: {e}\"));", "self.log(format!(\"bracket watch: {e}\")); break;"),
    ("주기 주입 무시(2초 고정)", W_, "while io.nap(period).await {", "while io.nap(WATCH_PERIOD_SEC + 0.0 * period).await {"),
]

A_, AS, IN = "src/account.rs", "src/assemble.rs", "src/intake.rs"
DESK = [  # desk_fixture(원본 파이썬 정답지 · (가)=(나) · 스트림 계좌 규칙)로 잡혀야 한다
    ("다른 심볼 지정가의 reduce_only 무시(P4)", AS, 'if os != symbol && !truthy(o.get("reduce_only")) && quote_of(&os) == quote_of(symbol) {', 'if os != symbol && quote_of(&os) == quote_of(symbol) {'),
    ("다른 코인 대기: 걸린 주문과 큰 쪽 → 합(P5)", AS, "Some(x) => x.1 = pyfmt::max(x.1, v),", "Some(x) => x.1 += v,"),
    ("증거금 풀 가르기 없음(P3·P7)", AS, "ps.into_iter().filter(|p| quote_of(&p.get(\"symbol\").map(py_str).unwrap_or_default()) == quote_of(symbol)).collect()", "ps"),
    ("지정가 진입 롱 내림 → 올림(P22)", AS, "if side == \"LONG\" { (lu * b_ / tick + 1e-9).floor() }", "if side == \"LONG\" { (lu * b_ / tick + 1e-9).ceil() }"),
    ("시간대 없는 entry_at 을 UTC 로", AS, "let Some(t) = parse_iso(s).filter(|_| tz) else", "let Some(t) = parse_iso(s).filter(|_| tz || true) else"),
    ("표시 수수료 왕복 차이 2× → 1×", AS, "(2.0 * (mk - MAKER_BP), (tk - TAKER_BP) - (mk - MAKER_BP))", "(1.0 * (mk - MAKER_BP), (tk - TAKER_BP) - (mk - MAKER_BP))"),
    ("실주문 청산도 펀딩·수수료 조회(P28)", AS, "let (funding, fees) = if for_order { (None, Value::Null) }", "let (funding, fees) = if false { (None, Value::Null) }"),
    ("basis 실패 → 1.0 대신 다른 값(P21)", AS, "let Ok(r) = q.book(ms).await else { return 1.0 };", "let Ok(r) = q.book(ms).await else { return 1.0001 };"),
    ("받아 둔 호가 2초 → 3초", AS, "pub const MAX_AGE_BOOK: f64 = 2.0;", "pub const MAX_AGE_BOOK: f64 = 3.0;"),
    ("받아 둔 값 나이 무시(조용히 낡은 값)", AS, "Some((v, at)) if self.now - at <= max_age && self.now >= *at => Some(v),", "Some((v, _at)) => Some(v),"),
    ("진입 confirm 검사 빠짐(G2)", IN, 'if g(q, "confirm") != Some("1") {\n            return resp(400, json!({"ok": false, "error": "confirm=1 required"}));\n        }\n        let Some(asset) = query_asset(self, q) else { return bad(400, "bad_asset", "주문할 수 없는 코인입니다") };\n        let Some(frac) = query_fraction(q, "pct") else { return bad(400, "bad_pct", "진입 비율은 0 초과 100 이하여야 합니다") };\n        if let Some(r) = removed_entry(q) {', 'let Some(asset) = query_asset(self, q) else { return bad(400, "bad_asset", "주문할 수 없는 코인입니다") };\n        let Some(frac) = query_fraction(q, "pct") else { return bad(400, "bad_pct", "진입 비율은 0 초과 100 이하여야 합니다") };\n        if let Some(r) = removed_entry(q) {'),
    ("같은 선 대기 중복 허용(G11)", IN, "- limit_px.unwrap_or(0.0)).abs() < 1e-9", "- limit_px.unwrap_or(0.0)).abs() < 0.0"),
    ("도는 작업이 있는 칸 재사용(G12)", IN, 'if self.running.contains(&format!("manual_wait_{}", Watcher::wkey(&asset, n))) && !taken.contains(&n) {', 'if false {'),
    ("같은 코인 청산 칸이 바빠도 진입(G5)", IN, '|| self.w.side_exits.get(&asset).and_then(Value::as_object).is_some_and(busy)', ''),
    ("일부 체결 뒤에도 지정가 수정(W6)", IN, 'if fl(st.get("filled")) > 0.0 {\n            return bad(409, "partially_filled"', 'if false {\n            return bad(409, "partially_filled"'),
    ("포지션 조회 실패를 ok·빈 목록으로(P2 · 09-26)", A_, "let Some(risk) = risk.as_array().cloned() else {", "let Some(risk) = risk.as_array().cloned().or(Some(vec![])) else {"),
    ("뒤집기 체결 앞끝 QTY_EPS 빠짐", A_, "if QTY_EPS < closing && closing < fl.qty - QTY_EPS {", "if 0.0 < closing && closing < fl.qty {"),
    ("멀티에셋 모드 무시", A_, 'if truthy(account.get("multiAssetsMargin")) {\n        return Ok((top()?', 'if false {\n        return Ok((top()?'),
    ("재연결 대기 5분 붙으면 되돌림 → 50분(A7)", A_, "pub const RECONNECT_RESET_AFTER_S: f64 = 300.0;", "pub const RECONNECT_RESET_AFTER_S: f64 = 3000.0;"),
    ("체결 알림에 계좌 다시 읽기 빠짐(A11)", A_, '                    if kind == "fill" {\n                        io.dirty();\n                    }', ''),
    ("알림 묶음 0.3초 → 0(A11)", A_, "pub const PUSH_COALESCE_S: f64 = 0.3;", "pub const PUSH_COALESCE_S: f64 = 0.0;"),
    ("밴 중 대기 남은 시간 + 1 → + 0", A_, "if !io.sleep(b + 1.0).await {", "if !io.sleep(b).await {"),
    ("재연결 뒤 스트림 계좌를 그대로 믿음", A_, '        if self.base_epoch != us.epoch {\n            return Err("reconnected");', '        if false {\n            return Err("reconnected");'),
    ("스트림 계좌 신뢰 60초 → 600초", A_, "pub const STREAM_TRUST_SEC: f64 = 60.0;", "pub const STREAM_TRUST_SEC: f64 = 600.0;"),
    ("어긋남 뒤에도 스트림 계좌를 씀", A_, '        if self.suspect {\n            return Err("mismatch");', '        if false {\n            return Err("mismatch");'),
]
HTTP_NEW = [  # http_replay(원본 파이썬 결함 수정본 + 시장가 다리 정답지)로 잡혀야 한다
    ("M9 잔량 시장가 뒤 filled 를 응답 체결량으로(D3 원본과 다름)", X, "(g.taker_qty, g.filled, g.taker_order_id) = (remaining, total, o.order_id);", "(g.taker_qty, g.filled, g.taker_order_id) = (remaining, done + o.executed_qty, o.order_id);"),
    ("M1 시장가 다리 빠짐(잔량 버림)", X, "    taker(ex, &c, remaining, s, |f, d| r8(f + d), None).await;", "    finish(s, Phase::Cancelled);"),
    ("시장가 ACK(체결 0)를 0 으로 적음", X, "let done = if o.executed_qty != 0.0 { o.executed_qty } else { qty };", "let done = o.executed_qty;"),
    ("시장가에 reduceOnly 를 붙임", H, '("type", P::s("MARKET")), ("quantity", P::F(quantity))];', '("type", P::s("MARKET")), ("quantity", P::F(quantity)), ("reduceOnly", P::s("true"))];'),
    ("익절·물타기 불명 응답 조회 안 함(결함 되살림)", B, "    let Some(e) = error_of(&r).map(String::from) else { return r };", "    let Some(_e) = error_of(&r).map(String::from) else { return r };\n    if true { return r; }\n    let e = _e;"),
    ("200 빈 본문을 정상으로", H, "        if !(payload.is_object() || payload.is_array()) {", "        if false {"),
    ("청산 정리: 진입 따라가기(여는 쪽)도 지움", B, '.starts_with(&pre) && o.get("side").and_then(Value::as_str) == Some(side)).cloned().collect();', '.starts_with(&pre)).cloned().collect();'),
    ("청산 정리: 다시 읽지 않고 성공", B, "        if attempt == 1 {\n            let ids", "        if attempt == 9 {\n            let ids"),
]
WATCH_NEW = [  # watch_fixture(결함 수정본 정답지)로 잡혀야 한다
    ("재발동 전 정리 빠짐", W_, "        if !t(b, \"retry_after\") {\n            return Ok(true);", "        if true {\n            return Ok(true);"),
    ("정리 실패에도 재발동", W_, "            b.insert(\"last_error\".into(), json!(format!(\"앞선 청산 주문 정리 실패: {}\", es.join(\"; \"))));\n            return Ok(false);", "            b.insert(\"last_error\".into(), json!(format!(\"앞선 청산 주문 정리 실패: {}\", es.join(\"; \"))));\n            return Ok(true);"),
    ("망가진 칸이 틱을 죽임(결함 되살림)", W_, "                    self.tick_alert(io, &key, &e);\n                }", "                    return Err(e);\n                }"),
    ("알림 한 번 → 매 틱", W_, "        if !self.alerted.insert((key.to_string(), kind)) {", "        if !self.alerted.insert((key.to_string(), kind)) && false {"),
]
RN, US = "src/runner.rs", "src/userstream.rs"
SIX = [  # 여섯째 조각: desk_fixture(선 핸들러 51 · 원본 정답지) · runner_e2e · userstream_ws · account_http 로 잡혀야 한다
    ("desk_fixture", "선 조각 같은 가격 ±0.05% → 0", IN, "let near = |old: &[Value], lv: f64| old.iter().any(|x| (fv(x.get(\"level\")) / lv - 1.0).abs() < 5e-4);", "let near = |old: &[Value], lv: f64| old.iter().any(|x| (fv(x.get(\"level\")) / lv - 1.0).abs() < 0.0);"),
    ("desk_fixture", "익절 켜기: 이미 너머 검사 >= → >", IN, "if if side == \"LONG\" { r0 >= tp } else { r0 <= tp } {", "if if side == \"LONG\" { r0 > tp + 100.0 } else { r0 < tp - 100.0 } {"),
    ("desk_fixture", "익절 끄기 일부 실패에도 바꿈(서버 감시로 안 넘김)", IN, "            if truthy(r.get(\"errors\")) {\n                // 2026-11-12 검토: 조각 몇 개는", "            if false {\n                // 2026-11-12 검토: 조각 몇 개는"),
    ("desk_fixture", "그 사이 무장 바뀜에도 방금 건 조각을 안 지움", IN, "            if rl.iter().any(|x| truthy(x.get(\"resting\"))) {", "            if false {"),
    ("desk_fixture", "선 수정: 익절 레벨 그대로여도 다시 걸음(F3)", IN, "let tp_moved = !tp_off && (fv(nb.get(\"tp_level\")) - fv(b.get(\"tp_level\"))).abs() > 1e-9;", "let tp_moved = !tp_off;"),
    ("desk_fixture", "선 수정 strict 빠짐(F2)", IN, "match io.place_bracket(&json!({\"tp_price\": nb[\"tp_price\"]}), &symbol, &side, true).await", "match io.place_bracket(&json!({\"tp_price\": nb[\"tp_price\"]}), &symbol, &side, false).await"),
    ("desk_fixture", "선 추가 새 무장에 다른 쪽 never 표시 빠짐", IN, "format!(\"{other}_off\"): {\"at\": now, \"never\": true}}));", "\"x\": 0}));"),
    ("runner_e2e", "체결 뒤 무장 안 함(B3)", RN, "    if filled > 0.0 {\n        let key", "    if filled > 1e9 {\n        let key"),
    ("runner_e2e", "작업 끝나도 칸 이름 남음(G12)", RN, "            put_slot(desk, &slot, st);\n            desk.task_done(&key);\n        }\n        Task::Exit", "            put_slot(desk, &slot, st);\n        }\n        Task::Exit"),
    ("userstream_ws", "웹소켓 주소 /private 빠짐(A4)", US, "let url = url.replacen(USER_STREAM_WS, &self.ws_base, 1);", "let url = url.replacen(USER_STREAM_WS, &self.ws_base.replace(\"/private\", \"\"), 1);"),
    ("userstream_ws", "익절 조각 체결을 대기열에 안 넣음", US, "        self.hub.tp_fills.lock().unwrap().push((s.into(), ps.into(), c.into()));", ""),
    ("userstream_ws", "스트림 사건을 계좌에 반영 안 함", US, "        self.hub.stream_account.lock().unwrap().apply(msg);", ""),
    ("account_http", "동시 조회가 openOrders 를 빼먹음", "src/account.rs", "    let all_rows = rest.get_many(&oo, off).await;", "    let all_rows: Vec<Value> = oo.iter().map(|_| serde_json::json!([])).collect();"),
]
SEVEN = [  # 일곱째 조각 뒤 «sltp 만»(원본 PR 61): 새 거절 3종 · 배포 전 dbsl·dbad 정리 경로 · 집행 머리
    ("flow_fixture", "집행 머리 레버리지 빠짐(P16)", X, "match ex.prepare(&plan.symbol, leverage.then_some(plan.target_leverage)).await {", "match ex.prepare(&plan.symbol, None).await {"),
    ("desk_fixture", "선 없는 진입 통과(lines_required 빠짐)", AS, "let Some(mut ln) = a.lines else {", "let Some(mut ln) = a.lines.or(Some((0.0, 0.0))) else {"),
    ("desk_fixture", "선 없는 포지션 추가 진입 통과(add_on_unlined_position 빠짐)", AS, 'if !same.is_empty() && !pt("lines") {', "if false {"),
    ("desk_fixture", "지운 인자 wait_min 혼자면 통과", IN, '["rule", "mode", "wait_min"]', '["rule", "mode"]'),
    ("desk_fixture", "지운 청산 인자 sw_* 통과", IN, 'k.starts_with("sw_")', "false"),
    ("desk_fixture", "exec=limit 도 거절", IN, 'v.as_str() != "limit"', "true"),
    ("desk_fixture", "손절 해제: dbsl 취소 실패를 무시", IN, '.filter(|e| !e.is_empty()) {\n            let msg', '.filter(|_| false) {\n            let msg'),
    ("flow_fixture", "dbsl 정리 빠짐", B, 'if str_or_empty(&a, "clientAlgoId").starts_with(&sl_pre) && same {', "if false {"),
    ("flow_fixture", "다른 심볼 dbsl 까지 지움", B, 'let same = a.get("symbol").map_or(true, |s| s.as_str() == Some(symbol));', "let same = true;"),
    ("flow_fixture", "dbad 정리 빠짐", B, " || (!only_tp && cid.starts_with(&add_pre))", ""),
    ("http_replay", "옛 무장 backstop 칸(«안 걸었음») 빠짐", B, '    if truthy(bracket.get("backstop_price")) {\n        out.insert("backstop"', '    if false {\n        out.insert("backstop"'),
]
WIRE = [  # wire_e2e(감시 → 계좌 → 조립 → 집행)로 잡혀야 한다
    ("스트림 계좌를 쓸 수 있어도 REST 로 읽음", A_, 'Ok(v) => Ok((v, "stream")),', 'Ok(_) => Ok((book.produce(rest).await?, "stream")),'),
    ("감시 청산 조립이 for_order=True(원본은 화면 조회까지 함)", "src/wire.rs", "pside, fraction, true, asset, false, Some(only_symbol)", "pside, fraction, true, asset, true, Some(only_symbol)"),
]


def digest():
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((SRC / "src").glob("*.rs"))}


def main() -> int:
    before = digest()
    env = dict(os.environ, CARGO_TARGET_DIR=str(SRC / "target" / "mut"), PATH=f"{Path.home()}/.cargo/bin:" + os.environ["PATH"])
    caught = total = 0
    only = sys.argv[1] if len(sys.argv) > 1 else ""   # "plan" | "exec" | "http" | "watch" | "" = 전부
    for test, muts in (("plan_fixture", PLAN), ("exec_invariants", EXEC), ("http_replay", HTTP),
                       ("watch_fixture", WATCH), ("watch_exec", WATCH_EXEC), ("watch_invariants", WATCH_INV), ("desk_fixture", DESK), ("wire_e2e", WIRE), ("http_replay", HTTP_NEW), ("watch_fixture", WATCH_NEW)):
        if only and not test.startswith(only):
            continue
        for name, f, a, b in muts:
            if len(sys.argv) > 2 and sys.argv[2] not in name:
                continue
            shutil.rmtree(W, ignore_errors=True)
            shutil.copytree(SRC, W, ignore=shutil.ignore_patterns("target"))
            p = W / f
            s = p.read_text()
            assert s.count(a) >= 1, (name, a)
            p.write_text(s.replace(a, b, 1))
            r = subprocess.run(["cargo", "test", "-j", "2", "--offline", "--test", test], cwd=W, env=env, capture_output=True, text=True, timeout=600)
            out = r.stdout + r.stderr
            failed = sorted({l.split()[1] for l in out.splitlines() if l.startswith("test ") and l.endswith("FAILED")})
            ok = r.returncode != 0
            caught, total = caught + ok, total + 1
            print(f"{'잡음' if ok else '🔴못 잡음'} | {test} | {name} | {', '.join(failed)[:110]}", flush=True)
    for grp, lst in (("six", SIX), ("seven", SEVEN)):
      if not only or only == grp:
        for test, name, f, a, b in lst:
            shutil.rmtree(W, ignore_errors=True)
            shutil.copytree(SRC, W, ignore=shutil.ignore_patterns("target"))
            p = W / f
            t = p.read_text()
            assert t.count(a) >= 1, (name, a)
            p.write_text(t.replace(a, b, 1))
            r = subprocess.run(["cargo", "test", "-j", "2", "--offline", "--test", test], cwd=W, env=env, capture_output=True, text=True, timeout=900)
            out = r.stdout + r.stderr
            failed = sorted({l.split()[1] for l in out.splitlines() if l.startswith("test ") and l.endswith("FAILED")})
            ok = r.returncode != 0
            caught, total = caught + ok, total + 1
            print(f"{'잡음' if ok else '🔴못 잡음'} | {test} | {name} | {', '.join(failed)[:110]}", flush=True)
    shutil.rmtree(W, ignore_errors=True)
    assert digest() == before, "원본 소스가 바뀌었다"
    print(f"{caught}/{total} 잡음 · 원본 소스 해시 그대로")
    return 0 if caught == total else 1


if __name__ == "__main__":
    sys.exit(main())
