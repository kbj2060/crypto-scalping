#!/usr/bin/env python3
"""양성 대조: 규칙을 하나씩 틀리게 바꾼 rtcollect 사본으로 일치 시험(tests/tape_parity.rs · tests/venue_parity.rs)을 돌려
실패하는지 본다 (2026-11-20, 같은 날 수집기 셋을 더하며 파일 배치를 lib/store/binance/... 로 고쳤다).

사본 = <작업 폴더>/mut (원본 rtcollect 는 안 건드린다 -- 끝에 src·tests 바이트를 처음과 대조한다). 빌드 산출물은
<작업 폴더>/mut_target 하나를 같이 쓰되 변이마다 `cargo clean -p rtcollect` -- 🔴처음엔 안 지워서 cargo 가 직전 변이의 산출물을
그대로 써(파일 시각 비교) 결과가 전부 틀렸다. 빌드에는 메모리 상한을 안 걸고, 시험 실행만 `ulimit -v 4000000` + timeout 으로.

    python3 rtcollect/tools/mutate_check.py <작업 폴더> [이름 일부 ...]      # 예: REST · 루프
일치 시험(tape·venue·rest_parity)을 기본으로, 다섯째 칸이 있는 변이(접속 루프 상수)는 그 시험(venue_loop)만 돌린다.
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
# (이름, 파일, 찾을 글자, 바꿀 글자) -- 찾을 글자는 그 파일에 정확히 한 번
M = [
    ("무변경(대조: 통과해야 함)", "src/lib.rs", "pub mod tape;", "pub mod tape;"),
    # ── 공용(tape·store·lib) ──
    ("반올림 짝수→0에서 먼 쪽", "src/tape.rs", ".round_ties_even()", ".round()"),
    ("묶음 간격 100→20ms", "src/tape.rs", "ORDER_GAP_MS: i64 = 100", "ORDER_GAP_MS: i64 = 20"),
    ("같은 초 조건 빼기", "src/tape.rs", "&& ts_ms.div_euclid(1000) == self.last_ms.div_euclid(1000)", ""),
    ("연속 체결ID 조건 빼기", "src/tape.rs", "|| tid == Some(self.last_tid + 1)", "|| true"),
    ("주문 시각 첫→마지막 체결", "src/tape.rs", "let out = (price, self.qty, self.first_ms, sell);",
     "let out = (price, self.qty, self.last_ms, sell);"),
    ("닫힌 초 되살림(늦은 체결 안 버림)", "src/tape.rs", "if sec < self.closed_before {", "if false {"),
    ("진행 중 초도 꺼냄(< → <=)", "src/tape.rs", "everything || sec < max_sec", "everything || sec <= max_sec"),
    ("리테일 경계 < → <=", "src/tape.rs", "if notional < retail_max {", "if notional <= retail_max {"),
    ("고래 경계 >= → >", "src/tape.rs", "} else if notional >= whale_min {", "} else if notional > whale_min {"),
    ("최대 칸을 더하기로", "src/tape.rs", "c[4 + i] = py_max(c[4 + i], qty);\n    }", "c[4 + i] += qty;\n    }"),
    ("주문 단위 체결수 무시(add_agg)", "src/tape.rs", "c[2 + i] += fills as f64;", "c[2 + i] += 1.0;"),
    ("재연결 봉인 +1 빼기(seal)", "src/tape.rs", "self.max_sec + 1);", "self.max_sec);"),
    ("INSERT OR REPLACE → OR IGNORE", "src/store.rs", "INSERT OR REPLACE INTO", "INSERT OR IGNORE INTO"),
    ("공백 문턱 1500→1000ms", "src/store.rs", "to_ms - from_ms < 1500", "to_ms - from_ms < 1000"),
    ("last_ts_ms 의 +1 빼기", "src/store.rs", "Some(s) if s != 0 => (s + 1) * 1000", "Some(s) if s != 0 => s * 1000"),
    ("뷰 DDL 공백 하나", "src/store.rs", "GROUP BY 1, 2, 3, 4\";", "GROUP BY 1,2, 3, 4\";"),
    ("테이프 auto_vacuum 빼기", "src/store.rs", "        con.execute_batch(\"PRAGMA auto_vacuum = INCREMENTAL\")?;", ""),
    ("건수 칸을 REAL 로 묶기", "src/store.rs", "let n = |j: usize| c[j] as i64;", "let n = |j: usize| c[j];"),
    ("meta repr 형식(Display)", "src/lib.rs", "pub(crate) fn py_repr(x: f64) -> String {\n    py_float_repr(x)", "pub(crate) fn py_repr(x: f64) -> String {\n    format!(\"{x}\")"),
    ("문자열 진릿값 = 항상 참", "src/lib.rs", "Value::String(s) => !s.is_empty(),", "Value::String(_) => true,"),
    ("가격 문자열 공백 안 벗김", "src/lib.rs", "Value::String(s) => s.trim().parse::<f64>()", "Value::String(s) => s.parse::<f64>()"),
    # ── 바이낸스 ──
    ("바이낸스 플러시 5→4초", "src/binance.rs", "FLUSH_SECONDS: f64 = 5.0", "FLUSH_SECONDS: f64 = 4.0"),
    ("바이낸스 재연결 때 묶음 리셋 빼기", "src/binance.rs", "self.orders.reset(); // 끊김", "// 끊김"),
    ("바이낸스 수량 0 거르기 빼기", "src/binance.rs", "if !(price > 0.0 && qty > 0.0) {", "if !(price > 0.0) {"),
    ("바이낸스 startup 공백 분 시작 → 1초 전", "src/binance.rs", "ts_ms.div_euclid(60_000) * 60_000", "ts_ms - 1000"),
    ("바이낸스 끊길 때 쓰기 빼기", "src/binance.rs", "let rows = self.buffer.take_closed(false);\n        self.store.write(rows)", "0"),
    ("바이낸스 체결ID 못 읽으면 무시", "src/binance.rs", "Some(t) => Some(py_int(t)?),", "Some(t) => py_int(t).ok(),"),
    ("바이낸스 현물 체결 수 하한 1 빼기", "src/binance.rs", "+ 1).max(1);", "+ 1);"),
    # ── Bybit 청산 ──
    ("청산 플러시 2→3초", "src/bybit_liq.rs", "FLUSH_S: f64 = 2.0", "FLUSH_S: f64 = 3.0"),
    ("청산 침묵 60→55초", "src/bybit_liq.rs", "if now - self.last_msg > IDLE_S {", "if now - self.last_msg > 55.0 {"),
    ("청산 이유 160→150 글자", "src/bybit_liq.rs", "take(160)", "take(150)"),
    ("청산 공백 끝 = 끊긴 시각(접속 시각 아님)", "src/bybit_liq.rs", "&[(*from, ms(wall), why.clone())]", "&[(*from, *from + 1, why.clone())]"),
    ("청산 끊길 때 남은 행 안 씀", "src/bybit_liq.rs", "        if !self.buf.is_empty() {\n            match write", "        if false {\n            match write"),
    ("청산 data 사전을 빈 것으로", "src/bybit_liq.rs", "Some(v) if py_truthy(v) => return Err", "Some(v) if py_truthy(v) && false => return Err"),
    ("청산 주제 접두 점 빼기", "src/bybit_liq.rs", "starts_with(\"allLiquidation.\")", "starts_with(\"allLiquidation\")"),
    ("청산 recv_ms = 거래소 시각", "src/bybit_liq.rs", "(Ok(s), Ok(side)) => Ok(Some((ts, s, side, v, p, recv_ms))),", "(Ok(s), Ok(side)) => Ok(Some((ts, s, side, v, p, ts))),"),
    # ── Bybit 체결 테이프 ──
    ("Bybit 플러시 5→4초", "src/bybit_tape.rs", "FLUSH_S: f64 = 5.0", "FLUSH_S: f64 = 4.0"),
    ("Bybit 묶음 닫기 1초→2초", "src/bybit_tape.rs", "ORDER_IDLE_S: f64 = 1.0", "ORDER_IDLE_S: f64 = 2.0"),
    ("Bybit 마크 1초→0.5초", "src/bybit_tape.rs", "MARK_EVERY_MS: i64 = 1000", "MARK_EVERY_MS: i64 = 500"),
    ("Bybit 방향 뒤집기", "src/bybit_tape.rs", "side == \"Sell\", px, qty, ts));", "side == \"Buy\", px, qty, ts));"),
    ("Bybit 시각 0 거르기 빼기", "src/bybit_tape.rs", "px > 0.0 && qty > 0.0 && ts > 0 &&", "px > 0.0 && qty > 0.0 &&"),
    ("Bybit 빈 값도 합치기", "src/bybit_tape.rs", "if !(v.is_null() || v.as_str() == Some(\"\")) {", "if !v.is_null() {"),
    ("Bybit OI 안 바뀌어도 행", "src/bybit_tape.rs", "if self.last_oi.get(s) != Some(&oi) {", "if true {"),
    ("Bybit 펀딩 다음 시각 무시", "src/bybit_tape.rs", "if self.last_fr.get(s) != Some(&(fr0, nft)) {", "if self.last_fr.get(s).map(|x| x.0) != Some(fr0) {"),
    ("Bybit 마크 판정 = 합친 상태", "src/bybit_tape.rs", "if d.contains_key(\"markPrice\") &&", "if c.contains_key(\"markPrice\") &&"),
    ("Bybit 재연결 전 진행 중 초 안 씀", "src/bybit_tape.rs", "let rows = s.buf.take_closed(force);", "let rows = s.buf.take_closed(false);"),
    ("Bybit 재연결 봉인 빼기", "src/bybit_tape.rs", "            s.buf.seal();\n", ""),
    ("Bybit 접속 때 첫 체결 표시 빼기", "src/bybit_tape.rs", "s.first = true; // 연결 뒤", "// 연결 뒤"),
    ("Bybit 공백 이유 = 글 전체", "src/bybit_tape.rs", "why.split(':').next().unwrap_or(why).to_string()", "why.to_string()"),
    ("Bybit 목록 심볼도 건너뜀", "src/bybit_tape.rs", "if matches!(s, Value::Array(_) | Value::Object(_)) {", "if false {"),
    ("Bybit int(inf) 이름 ValueError", "src/bybit_tape.rs", "Reconnect(\"OverflowError\".into())", "Reconnect(\"ValueError\".into())"),
    # ── OKX ──
    ("OKX 플러시 5→4초", "src/okx.rs", "FLUSH_SECONDS: f64 = 5.0", "FLUSH_SECONDS: f64 = 4.0"),
    ("OKX ctVal 안 곱함", "src/okx.rs", "contracts * ct, side", "contracts, side"),
    ("OKX XRP ctVal 100→10", "src/okx.rs", "\"XRP-USDT-SWAP\" => 100.0,", "\"XRP-USDT-SWAP\" => 10.0,"),
    ("OKX 체결 수 하한 빼기", "src/okx.rs", "fills.max(1)))", "fills))"),
    ("OKX count 못 읽으면 0", "src/okx.rs", "Some(v) => py_int(v).unwrap_or(1),", "Some(v) => py_int(v).unwrap_or(0),"),
    ("OKX 방향 대소문자 그대로", "src/okx.rs", "map_or(String::new(), str::to_lowercase)", "map_or(String::new(), str::to_string)"),
    ("OKX 이벤트 메시지 처리", "src/okx.rs", "if event.is_some_and(py_truthy) {", "if false {"),
    ("OKX 끊길 때 쓰기 빼기", "src/okx.rs", "let rows = self.buffer.take_closed(false);\n        self.store.write(rows)", "0"),
    ("OKX meta ct_val 열쇠", "src/okx.rs", "format!(\"ct_val:{inst}\")", "format!(\"ctval:{inst}\")"),
    ("OKX startup 공백 분 시작 → 1초 전", "src/okx.rs", "ts_ms.div_euclid(60_000) * 60_000", "ts_ms - 1000"),
    # ── 바이낸스 REST 검증·복구(rest_parity) ──
    ("REST 47시간 → 48시간", "src/rest.rs", "LOOKBACK_S: i64 = 47 * 3600", "LOOKBACK_S: i64 = 48 * 3600"),
    ("REST 끝난 지 3분 → 2분", "src/rest.rs", "*m <= now - 180", "*m <= now - 120"),
    ("REST 3번 실패 → 4번", "src/rest.rs", "MAX_ATTEMPTS: i64 = 3", "MAX_ATTEMPTS: i64 = 4"),
    ("REST 맞는 분도 후보", "src/rest.rs", "(bad.contains(m) || !good.contains(m))", "true"),
    ("REST 공백 끝 분 (b-1) → b", "src/rest.rs", "(b - 1).div_euclid(60_000)", "b.div_euclid(60_000)"),
    ("REST 허용오차 1e-4 → 1e-1", "src/rest.rs", "TOL: f64 = 1e-4", "TOL: f64 = 1e-1"),
    ("REST 라이브+REST 보다 REST 먼저", "src/rest.rs", "[(&hybrid, \"라이브+REST\"), (&rows, \"REST\")]", "[(&rows, \"REST\"), (&hybrid, \"라이브+REST\")]"),
    ("REST 최대 칸 NULL 안 함", "src/rest.rs", "(r.2[4], r.2[5]) = (f64::NAN, f64::NAN);", ""),
    ("REST 간격 2→1초", "src/rest.rs", "AGG_GAP_S: f64 = 2.0", "AGG_GAP_S: f64 = 1.0"),
    ("REST 가중치 문턱 1500→1501", "src/rest.rs", "WEIGHT_YIELD: i64 = 1500", "WEIGHT_YIELD: i64 = 1501"),
    ("REST 양보 61→60초", "src/rest.rs", "self.clock.mono() + 61.0 -", "self.clock.mono() + 60.0 -"),
    ("REST 영구 거부 없음", "src/rest.rs", "(400..500).contains(&s) && s != 418 && s != 429", "false"),
    ("REST 429 도 영구 거부", "src/rest.rs", "(400..500).contains(&s) && s != 418 && s != 429", "(400..500).contains(&s) && s != 418"),
    ("REST 쪽 끝 판정(분 뒤 체결) 빼기", "src/rest.rs", "if batch.len() < 1000 || int(key(last.unwrap(), \"T\")?)? > end {", "if batch.len() < 1000 {"),
    ("REST fromId +1 빼기", "src/rest.rs", "\"a\")?)? + 1)", "\"a\")?)?)"),
    ("REST 분 끝 59_999 → 60_000", "src/rest.rs", "m * 1000 + 59_999", "m * 1000 + 60_000"),
    ("REST 대조 limit +2 → +1", "src/rest.rs", ".div_euclid(60) + 2))", ".div_euclid(60) + 1))"),
    ("REST 시도 횟수 이어 세기 빼기", "src/rest.rs", "ok as i64, prev + 1, note]", "ok as i64, 1, note]"),
    ("REST 교체 때 옛 검사 안 지움", "src/rest.rs", "DELETE FROM verify_1m WHERE symbol = ? AND ts_min = ?", "DELETE FROM verify_1m WHERE symbol = ? AND ts_min = ? AND 0"),
    ("REST 1분봉 0 이면 rel 0 빼기", "src/rest.rs", "(got, if kline_qty != 0.0 { (got - kline_qty) / kline_qty } else { 0.0 })", "(got, (got - kline_qty) / kline_qty)"),
    # ── 접속 루프(venue_loop -- 원본 상수 그대로라 하나에 약 70초) ──
    ("루프 OKX PING 20→15초", "src/net.rs", "OKX_HEARTBEAT: Duration = Duration::from_secs(20)", "OKX_HEARTBEAT: Duration = Duration::from_secs(15)", ["venue_loop"]),
    ("루프 OKX 무음 25→30초", "src/okx.rs", "RECV_TIMEOUT_S: f64 = 25.0", "RECV_TIMEOUT_S: f64 = 30.0", ["venue_loop"]),
    ("루프 Bybit 테이프 재접속 3→2초", "src/net.rs", "        self.on_error(now, wall, why);\n        3.0", "        self.on_error(now, wall, why);\n        2.0", ["venue_loop"]),
    ("루프 Bybit ping 글자 공백 없음", "src/net.rs", "BYBIT_PING: &str = r#\"{\"op\": \"ping\"}\"#", "BYBIT_PING: &str = r#\"{\"op\":\"ping\"}\"#", ["venue_loop"]),
    ("루프 Bybit 테이프 ping 20→25초", "src/bybit_tape.rs", "PING_S: f64 = 20.0", "PING_S: f64 = 25.0", ["venue_loop"]),
    ("루프 Bybit 청산 침묵 60→70초", "src/bybit_liq.rs", "IDLE_S: f64 = 60.0", "IDLE_S: f64 = 70.0", ["venue_loop"]),
    ("루프 구독 글자 json.dumps 모양 아님", "src/lib.rs", "args.join(\", \")", "args.join(\",\")", ["venue_loop"]),
    ("루프 Bybit 테이프 정지 종료코드 3→70", "src/net.rs", "EXIT_TRADE_STALL: i32 = 3", "EXIT_TRADE_STALL: i32 = 70", ["venue_loop"]),
    # ── 맥락 공용(store::open_hot) ──
    ("맥락 auto_vacuum 빼기", "src/store.rs", "    con.execute_batch(\"PRAGMA auto_vacuum = INCREMENTAL\")?;\n    con.query_row(\"PRAGMA journal_mode = WAL\", [], |_| Ok(()))?;\n    con.execute_batch(\"PRAGMA synchronous = NORMAL\")?;",
     "    con.query_row(\"PRAGMA journal_mode = WAL\", [], |_| Ok(()))?;\n    con.execute_batch(\"PRAGMA synchronous = NORMAL\")?;", ["ctx_parity"]),
    # ── OKX 맥락 ──
    ("OKX맥락 플러시 10→9초(디스크엔 안 보임 -- 끊길 때 다 쓴다)", "src/okx_ctx.rs", "FLUSH_SECONDS: f64 = 10.0", "FLUSH_SECONDS: f64 = 9.0", ["ctx_parity"]),
    ("OKX맥락 공백 문턱 1500→1000ms", "src/okx_ctx.rs", "if to - self.down_from < 1500 {", "if to - self.down_from < 1000 {", ["ctx_parity"]),
    ("OKX맥락 ctVal 모름 → 0(NULL 아님)", "src/okx_ctx.rs", "ct.map(|c| sz * c)", "Some(ct.map_or(0.0, |c| sz * c))", ["ctx_parity"]),
    ("OKX맥락 마크 0·음수 가격 허용", "src/okx_ctx.rs", "px.filter(|x| *x > 0.0)", "px", ["ctx_parity"]),
    ("OKX맥락 펀딩 ts 없음 → 벽시계 대체 빼기", "src/okx_ctx.rs", "let ts = if ts != 0 { ts } else { ms(wall) };", "let ts = if ts != 0 { ts } else { 0 };", ["ctx_parity"]),
    ("OKX맥락 청산 구독 = 모든 종목", "src/okx_ctx.rs", "        if self.inst == LIQ_INST {\n            a.push", "        if true {\n            a.push", ["ctx_parity"]),
    ("OKX맥락 DDL 들여쓰기 하나", "src/okx_ctx.rs", "okx_oi(\\n         inst", "okx_oi(\\n        inst", ["ctx_parity"]),
    ("OKX맥락 사전 아닌 원소 = 건너뜀(재연결 아님)", "src/okx_ctx.rs", "Some(Value::Array(a)) => a.iter().map(|x| x.as_object().ok_or_else(|| Reconnect(\"AttributeError: .get\".into()))).collect(),",
     "Some(Value::Array(a)) => Ok(a.iter().filter_map(|x| x.as_object()).collect()),", ["ctx_parity"]),
    ("OKX맥락 이벤트 거르기 빼기", "src/okx_ctx.rs", "        if event.is_some_and(py_truthy) {\n            return Ok(());\n        }\n        let batch", "        let batch", ["ctx_parity"]),
    ("ts 무한대·NaN 을 재연결 대신 그대로", "src/lib.rs", "Some(f) if f.is_finite() => Ok(f.trunc() as i64),", "Some(f) => Ok(f.trunc() as i64),", ["ctx_parity"]),
    # ── 바이낸스 맥락 ──
    ("바맥 마크 INSERT OR IGNORE → OR REPLACE", "src/binance_ctx.rs", "INSERT OR IGNORE INTO mark_price_1s", "INSERT OR REPLACE INTO mark_price_1s", ["ctx_parity"]),
    ("바맥 청산 수량 z 보다 q 먼저", "src/binance_ctx.rs", "first(\"z\", \"q\")?", "first(\"q\", \"z\")?", ["ctx_parity"]),
    ("바맥 롱·숏 뒤집기", "src/binance_ctx.rs", "== Some(\"SELL\") { \"long\" } else { \"short\" }", "== Some(\"SELL\") { \"short\" } else { \"long\" }", ["ctx_parity"]),
    ("바맥 느린 OI 종목을 청산 종목에서 빼기", "src/binance_ctx.rs", ".chain(slow.iter().cloned())", "", ["ctx_parity"]),
    ("바맥 거절 대기 30→20초", "src/binance_ctx.rs", "                _ => 30.0,", "                _ => 20.0,", ["ctx_parity"]),
    ("바맥 오류 뒤 1초 잠 빼기", "src/binance_ctx.rs", "                self.clock.sleep(1.0);\n", "", ["ctx_parity"]),
    ("바맥 밴 중에도 보냄", "src/binance_ctx.rs", "        if left > 0.0 {\n            // 다른", "        if false {\n            // 다른", ["ctx_parity"]),
    ("aiohttp 본문 형식 검사 빼기", "src/rest.rs", "    if !json {\n        return Err(Fail::Other(format!(\"ContentTypeError", "    if false {\n        return Err(Fail::Other(format!(\"ContentTypeError", ["ctx_parity"]),
    # ── xyz100 ──
    ("xyz 플러시 2→3초", "src/xyz.rs", "FLUSH_S: f64 = 2.0", "FLUSH_S: f64 = 3.0", ["ctx_parity"]),
    ("xyz 시한 셋→넷", "src/xyz.rs", "if self.quiet >= 3 {", "if self.quiet >= 4 {", ["ctx_parity"]),
    ("xyz 같은 열쇠 = 새 자리", "src/xyz.rs", "            Some(&i) => self.rows[i] = r,", "            Some(_) => self.rows.push(r),", ["ctx_parity"]),
    ("xyz 참거짓 ≠ 숫자 열쇠", "src/xyz.rs", "Value::Bool(b) => format!(\"n{}\", *b as i64 as f64),", "Value::Bool(b) => format!(\"b{b}\"),", ["ctx_parity"]),
    ("xyz ctx 비사전 = 빈 것(재연결 아님)", "src/xyz.rs", "        Some(_) => return Err(Reconnect(\"AttributeError: .get\".into())),\n    };\n    // float", "        Some(_) => &empty,\n    };\n    // float", ["ctx_parity"]),
    ("xyz REST 72→48시간", "src/xyz.rs", "BACKFILL_H: i64 = 72;", "BACKFILL_H: i64 = 48;", ["ctx_parity"]),
    ("xyz candle n 못 읽으면 0", "src/xyz.rs", "        Some(v) => py_int(v).ok()?,\n        None => 0,", "        Some(v) => py_int(v).unwrap_or(0),\n        None => 0,", ["ctx_parity"]),
    ("xyz 대기 리셋 빼기(디스크엔 안 보임 → 루프 시험)", "src/xyz.rs", "(Ordered::default(), Ordered::default(), now, 1.0);", "(Ordered::default(), Ordered::default(), now, self.backoff);", ["venue_loop"]),
    ("루프 OKX맥락 시한 40→30초", "src/okx_ctx.rs", "RECV_TIMEOUT_S: f64 = 40.0;", "RECV_TIMEOUT_S: f64 = 30.0;", ["venue_loop"]),
    ("루프 바맥 heartbeat 30→20초", "src/net.rs", "AIO_HEARTBEAT: Duration = Duration::from_secs(30);", "AIO_HEARTBEAT: Duration = Duration::from_secs(20);", ["venue_loop"]),
    ("루프 xyz heartbeat 25→20초", "src/net.rs", "XYZ_HEARTBEAT: Duration = Duration::from_secs(25);", "XYZ_HEARTBEAT: Duration = Duration::from_secs(20);", ["venue_loop"]),
    # ── OKX·Bybit 테이프 REST(rest_venue) ──
    ("OKX복구 쪽 넘김 tradeId(type=1)→ts(type=2)", "src/rest_venue.rs", "type=1&after={a}&limit=100", "type=2&after={a}&limit=100", ["venue_rest_parity"]),
    ("OKX복구 확정(confirm) 검사 빼기", "src/rest_venue.rs", " || py_str(Some(at(&c[0], 8)?), \"\") != \"1\"", "", ["venue_rest_parity"]),
    ("OKX복구 속도 0.15→0.1초", "src/rest_venue.rs", "OKX_MIN_GAP_S: f64 = 0.15;", "OKX_MIN_GAP_S: f64 = 0.1;", ["venue_rest_parity"]),
    ("OKX복구 429 벌점 10→0초", "src/rest_venue.rs", "OKX_429_PENALTY_S: f64 = 10.0;", "OKX_429_PENALTY_S: f64 = 0.0;", ["venue_rest_parity"]),
    ("OKX복구 쪽 사이 0.12초 빼기", "src/rest_venue.rs", "            self.clock.sleep(0.12);\n            let page", "            let page", ["venue_rest_parity"]),
    ("OKX복구 한 주기 15→10분", "src/rest_venue.rs", "crate::okx::bucket(inst), 15,", "crate::okx::bucket(inst), 10,", ["venue_rest_parity"]),
    ("OKX복구 연속 같은 (ts,side,px) 안 묶음", "src/rest_venue.rs", "if Some(key) == last {", "if false {", ["venue_rest_parity"]),
    ("OKX복구 창 7일→47시간", "src/rest_venue.rs", "15, crate::rest::VERIFY_LOOKBACK_S,", "15, crate::rest::LOOKBACK_S,", ["venue_rest_parity"]),
    ("OKX대조 형성 중 봉(confirm 0) 포함", "src/rest_venue.rs", "                if py_str(Some(at(row, 8)?), \"\") == \"1\" {", "                if true {", ["venue_rest_parity"]),
    ("OKX대조 구간 99→100분", "src/rest_venue.rs", "anchor - 99 * 60", "anchor - 100 * 60", ["venue_rest_parity"]),
    ("OKX대조 물량 volCcy→vol(계약)", "src/rest_venue.rs", "float(at(row, 6)?)?);", "float(at(row, 5)?)?);", ["venue_rest_parity"]),
    ("Bybit대조 end 끝 분 +1분", "src/rest_venue.rs", "lo * 1000, hi * 1000))?;", "lo * 1000, (hi + 60) * 1000))?;", ["venue_rest_parity"]),
    ("ctVal 상태 코드를 본다(원본은 안 본다)", "src/rest_venue.rs", "    let (_, body) = get_json(agent, url)?;", "    let (st, body) = get_json(agent, url)?;\n    if st >= 400 {\n        return Err(Fail::Other(\"상태\".into()));\n    }", ["venue_rest_parity"]),
    # ── 다섯째 조각: 의도한 차이(결함은 Rust 에서만 고침) -- 옛 동작으로 되돌리면 시험이 잡아야 한다 ──
    ("원본으로: Bybit청산 바인딩 못 하는 행 = 재연결", "src/bybit_liq.rs", "serde_json::Value::Object(d.clone()));\n                    Ok(None)", "serde_json::Value::Object(d.clone()));\n                    Err(Reconnect(\"InterfaceError\".into()))", ["venue_parity"]),
    ("원본으로: Bybit테이프 못 읽는 칸도 합침", "src/bybit_tape.rs", "if NUM_KEYS.contains(&k.as_str()) && !num_ok(k, v) {", "if false {", ["venue_parity"]),
    ("원본으로: OKX맥락 공백 시작 = 마지막 실패", "src/okx_ctx.rs", "        if self.up {\n            self.down_from = ms(wall);\n        }", "        self.down_from = ms(wall);", ["ctx_parity"]),
    ("원본으로: 바맥 깨진 JSON = 재연결", "src/binance_ctx.rs", "        if serde_json::from_str::<Value>(text).is_err() {", "        if false {", ["ctx_parity"]),
    ("원본으로: 바맥 OI seen 먼저", "src/binance_ctx.rs", "if !self.seen.contains(&ts) {", "if self.seen.insert(ts) {", ["ctx_parity"]),
    ("원본으로: 바맥 gaps 안 적음", "src/binance_ctx.rs", "            if now - d >= 1500 {", "            if false {", ["ctx_parity"]),
    ("원본으로: xyz 끊길 때 남은 행 버림", "src/xyz.rs", "        self.write_rest();\n        let b", "        (self.bars, self.ctxs) = (Ordered::default(), Ordered::default());\n        let b", ["ctx_parity"]),
    # ── HL 자산 맥락 ──
    ("HL맥락 OI 없음도 적음", "src/hl_ctx.rs", "let oi = g(\"openInterest\")?;", "let oi = g(\"openInterest\").unwrap_or(0.0);", ["ctx_parity"]),
    ("HL맥락 끊길 때 안 씀", "src/hl_ctx.rs", "    pub fn on_disconnect(&mut self) {\n        self.write();", "    pub fn on_disconnect(&mut self) {\n        self.buf.clear();", ["ctx_parity"]),
    ("HL맥락 칸 NULL → 0", "src/hl_ctx.rs", "let g = |k: &str| ctx.get(k).and_then(|v| py_float(v).ok());", "let g = |k: &str| Some(ctx.get(k).and_then(|v| py_float(v).ok()).unwrap_or(0.0));", ["ctx_parity"]),
    ("루프 HL맥락 재접속 3→1초", "src/net.rs", "        self.on_disconnect();\n        3.0", "        self.on_disconnect();\n        1.0", ["venue_loop"]),
    # ── depth_diff ──
    ("depth 스냅숏 열쇠 정렬(preserve_order 무시)", "src/lib.rs", "                for (i, (k, x)) in o.iter().enumerate() {", "                for (i, (k, x)) in { let mut v: Vec<_> = o.iter().collect(); v.sort_by(|a, b| a.0.cmp(b.0)); v }.into_iter().enumerate() {", ["depth_parity"]),
    ("depth 시 열쇠 +1시간", "src/depth.rs", "    let secs = ts_ms.div_euclid(1000);", "    let secs = ts_ms.div_euclid(1000) + 3600;", ["depth_parity"]),
    ("depth 압축 안 함", "src/depth.rs", "self.zips.push(std::thread::spawn(move || gzip_done(&p)));", "drop(p);", ["depth_parity"]),
    ("depth 구멍 판정 값 비교 → 모양 비교", "src/depth.rs", "if !py_eq(&pu, prev) {", "if pu != *prev {", ["depth_parity"]),
    ("depth 구멍 줄 got_pu 빼기", "src/depth.rs", 'json!({"_gap": true, "expected_pu": prev, "got_pu": pu})', 'json!({"_gap": true, "expected_pu": prev})', ["depth_parity"]),
    ("depth 새 파일 스냅숏 안 씀", "src/depth.rs", "            if new {\n                let s = snap();", "            if false {\n                let s = snap();", ["depth_parity"]),
    ("루프 depth 재접속 때 prev_u 유지", "src/depth.rs", "    pub fn on_connect(&mut self) {\n        self.prev_u = None;", "    pub fn on_connect(&mut self) {", ["venue_loop"]),
    ("루프 depth 깨끗한 닫힘도 3초", "src/net.rs", "        if !clean {", "        if true {", ["venue_loop"]),
]
PARITY = ["tape_parity", "venue_parity", "rest_parity", "ctx_parity", "venue_rest_parity", "depth_parity"]


def digest() -> str:
    h = hashlib.sha256()
    for p in sorted(list((SRC / "src").rglob("*")) + list((SRC / "tests").rglob("*"))):
        if p.is_file():
            h.update(str(p.relative_to(SRC)).encode() + p.read_bytes())
    return h.hexdigest()


def main() -> None:
    work = Path(sys.argv[1])
    pick = sys.argv[2:]
    w = work / "mut"
    env = dict(os.environ, CARGO_TARGET_DIR=str(work / "mut_target"), PATH=f"{Path.home()}/.cargo/bin:" + os.environ["PATH"])
    before = digest()
    caught, missed, n = 0, [], 0
    for name, f, a, b, *tests in M:
        tests = tests[0] if tests else PARITY
        targs = " ".join(f"--test {t}" for t in tests)
        if pick and name != M[0][0] and not any(x in name for x in pick):
            continue
        shutil.rmtree(w, ignore_errors=True)
        shutil.copytree(SRC, w, ignore=shutil.ignore_patterns("target"), copy_function=shutil.copy)
        p = w / f
        s = p.read_text()
        assert s.count(a) == 1, (name, s.count(a))
        p.write_text(s.replace(a, b))
        subprocess.run(["cargo", "clean", "-q", "-p", "rtcollect"], cwd=w, env=env, check=True)
        subprocess.run(["cargo", "test", "-q", "-j", "2", "--no-run", *targs.split()], cwd=w, env=env, check=True, capture_output=True)
        # 접속 루프 시험은 나란히(각 70초) · 일치 시험은 하나씩
        r = subprocess.run(["bash", "-c", f"ulimit -v 4000000; timeout 300 cargo test -q -j 2 {targs} -- "
                            + ("" if tests != PARITY else "--test-threads 1")], cwd=w, env=env, capture_output=True, text=True)
        out = r.stdout + r.stderr
        failed_tests = sorted({ln.split()[1] for ln in out.splitlines() if ln.startswith("test ") and ln.rstrip().endswith("FAILED")}
                              | {ln.strip() for ln in out.splitlines() if ln.startswith("    ") and "match_python" in ln})
        msg = next((ln.strip() for ln in out.splitlines() if "!=" in ln or "행 수" in ln or "PRAGMA" in ln or "SQL" in ln
                    or "더 많다" in ln), "")
        failed = r.returncode != 0
        if name != M[0][0]:
            n += 1
            caught += failed
            if not failed:
                missed.append(name)
        else:
            assert not failed, f"무변경 사본이 실패했다:\n{out[-2000:]}"
        print(f"{'실패(잡음)' if failed else '통과'} | {name} | {','.join(t.replace('_rows_match_python_collector', '').replace('_loop_matches_replay', '') for t in failed_tests)} | {msg[:140]}",
              flush=True)
    shutil.rmtree(w, ignore_errors=True)
    assert digest() == before, "원본 rtcollect 의 src·tests 가 바뀌었다"
    print(f"잡은 변이 {caught}/{n} · 못 잡은 것 {missed} · 원본 src·tests 바이트 그대로")


if __name__ == "__main__":
    main()
