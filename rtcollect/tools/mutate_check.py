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
    ("테이프 auto_vacuum 빼기", "src/store.rs", "con.execute_batch(\"PRAGMA auto_vacuum = INCREMENTAL\")?;", ""),
    ("건수 칸을 REAL 로 묶기", "src/store.rs", "let n = |j: usize| c[j] as i64;", "let n = |j: usize| c[j];"),
    ("meta repr 형식(Display)", "src/lib.rs", "format!(\"{x:?}\")", "format!(\"{x}\")"),
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
    ("청산 침묵 60→55초", "src/bybit_liq.rs", "if t - self.last_msg > IDLE_S {", "if t - self.last_msg > 55.0 {"),
    ("청산 이유 160→150 글자", "src/bybit_liq.rs", "take(160)", "take(150)"),
    ("청산 공백 끝 = 끊긴 시각(접속 시각 아님)", "src/bybit_liq.rs", "&[(*from, ms(t), why.clone())]", "&[(*from, *from + 1, why.clone())]"),
    ("청산 끊길 때 남은 행 안 씀", "src/bybit_liq.rs", "        if !self.buf.is_empty() {\n            match write", "        if false {\n            match write"),
    ("청산 data 사전을 빈 것으로", "src/bybit_liq.rs", "Some(v) if py_truthy(v) => return Err", "Some(v) if py_truthy(v) && false => return Err"),
    ("청산 주제 접두 점 빼기", "src/bybit_liq.rs", "starts_with(\"allLiquidation.\")", "starts_with(\"allLiquidation\")"),
    ("청산 recv_ms = 거래소 시각", "src/bybit_liq.rs", "Ok((ts, s, side, py_float(need(d, \"v\")?)?, py_float(need(d, \"p\")?)?, recv_ms))",
     "Ok((ts, s, side, py_float(need(d, \"v\")?)?, py_float(need(d, \"p\")?)?, ts))"),
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
]
PARITY = ["tape_parity", "venue_parity", "rest_parity"]


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
