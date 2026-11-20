#!/usr/bin/env python3
"""양성 대조: 규칙을 하나씩 틀리게 바꾼 rtcollect 사본으로 tests/tape_parity.rs 를 돌려 실패하는지 본다 (2026-11-20).

사본 = <작업 폴더>/mut (원본 rtcollect 는 안 건드린다). 빌드 산출물은 <작업 폴더>/mut_target 하나를 같이 쓰되
변이마다 `cargo clean -p rtcollect` -- 🔴처음엔 안 지워서 cargo 가 직전 변이의 산출물을 그대로 써(파일 시각 비교)
결과가 전부 틀렸다. 빌드에는 메모리 상한을 안 걸고, 시험 실행만 `ulimit -v 4000000` + timeout 으로 감싼다.

    python3 rtcollect/tools/mutate_check.py <작업 폴더>
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
M = [
    ("무변경(대조: 통과해야 함)", "src/lib.rs", "pub mod tape;", "pub mod tape;"),
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
    ("INSERT OR REPLACE → OR IGNORE", "src/lib.rs", "INSERT OR REPLACE INTO", "INSERT OR IGNORE INTO"),
    ("플러시 5→4초", "src/lib.rs", "FLUSH_SECONDS: f64 = 5.0", "FLUSH_SECONDS: f64 = 4.0"),
    ("공백 문턱 1500→1000ms", "src/lib.rs", "to_ms - from_ms < 1500", "to_ms - from_ms < 1000"),
    ("재연결 때 묶음 리셋 빼기", "src/lib.rs", "self.orders.reset(); // 끊김", "// 끊김"),
    ("수량 0 거르기 빼기", "src/lib.rs", "if !(price > 0.0 && qty > 0.0) {", "if !(price > 0.0) {"),
    ("startup 공백 분 시작 → 그 체결 1초 전", "src/lib.rs", "ts_ms.div_euclid(60_000) * 60_000", "ts_ms - 1000"),
    ("last_ts_ms 의 +1 빼기", "src/lib.rs", "Some(s) if s != 0 => (s + 1) * 1000", "Some(s) if s != 0 => s * 1000"),
    ("끊길 때 쓰기 빼기", "src/lib.rs", "let rows = self.buffer.take_closed(false);\n        self.store.write(rows)", "0"),
    ("체결ID 못 읽으면 무시(재연결 안 함)", "src/lib.rs", "Some(t) => Some(py_int(t)?),", "Some(t) => py_int(t).ok(),"),
    ("문자열 m 진릿값 = 항상 참", "src/lib.rs", "Value::String(s) => !s.is_empty(),", "Value::String(_) => true,"),
    ("가격 문자열 공백 안 벗김", "src/lib.rs", "Value::String(s) => s.trim().parse::<f64>()", "Value::String(s) => s.parse::<f64>()"),
    ("현물 체결 수 하한 1 빼기", "src/lib.rs", "+ 1).max(1);", "+ 1);"),
    ("뷰 DDL 공백 하나", "src/lib.rs", "GROUP BY 1, 2, 3, 4\";", "GROUP BY 1,2, 3, 4\";"),
    ("auto_vacuum 빼기", "src/lib.rs", "con.execute_batch(\"PRAGMA auto_vacuum = INCREMENTAL\")?;", ""),
    ("건수 칸을 REAL 로 묶기", "src/lib.rs", "let n = |j: usize| c[j] as i64;", "let n = |j: usize| c[j];"),
    ("meta repr 형식(Display)", "src/lib.rs", "format!(\"{x:?}\")", "format!(\"{x}\")"),
]


def main() -> None:
    work = Path(sys.argv[1])
    w = work / "mut"
    env = dict(os.environ, CARGO_TARGET_DIR=str(work / "mut_target"),
               PATH=f"{Path.home()}/.cargo/bin:" + os.environ["PATH"])
    caught = 0
    for name, f, a, b in M:
        shutil.rmtree(w, ignore_errors=True)
        shutil.copytree(SRC, w, ignore=shutil.ignore_patterns("target"), copy_function=shutil.copy)
        p = w / f
        s = p.read_text()
        assert s.count(a) == 1, (name, s.count(a))
        p.write_text(s.replace(a, b))
        subprocess.run(["cargo", "clean", "-q", "-p", "rtcollect"], cwd=w, env=env, check=True)
        subprocess.run(["cargo", "test", "-q", "-j", "2", "--no-run", "--test", "tape_parity"], cwd=w, env=env,
                       check=True, capture_output=True)
        r = subprocess.run(["bash", "-c", "ulimit -v 4000000; timeout 300 cargo test -q -j 2 --test tape_parity"],
                           cwd=w, env=env, capture_output=True, text=True)
        out = r.stdout + r.stderr
        msg = next((ln.strip() for ln in out.splitlines() if "!=" in ln or "행 수" in ln or "PRAGMA" in ln
                    or "SQL" in ln), "")
        failed = r.returncode != 0
        caught += failed and name != M[0][0]
        print(f"{'실패(잡음)' if failed else '통과'} | {name} | {msg[:160]}", flush=True)
    print(f"잡은 변이 {caught}/{len(M) - 1}")


if __name__ == "__main__":
    main()
