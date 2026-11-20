#!/usr/bin/env python3
"""재시작 복원 일치 시험의 양성 대조 (2026-11-21): 복원 규칙을 하나씩 일부러 틀리게 바꿔 tests/restore_parity.rs 가 실패하는지 본다.

mutate_footprint.py 와 같은 방식: 소스 사본을 먼저 떠 두고 변형마다 «바꿈 → 빌드(--no-run) → 실행(ulimit -v 4GB · timeout 300)
→ 사본으로 되돌림 → 바이트 비교». 되돌린 파일이 사본과 다르면 곧바로 멈춘다. 중간에 죽어도 finally 가 사본을 쓴다.
정답지가 먼저 있어야 한다(python3 rtcore/tools/gen_restore_fixture.py).
    python3 rtcore/tools/mutate_restore.py          # 전부
    python3 rtcore/tools/mutate_restore.py 0 5      # 번호만
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parents[1]
FILES = ["src/restore.rs", "src/supply.rs", "src/footprint.rs", "src/state.rs"]
RS, SU, FP, ST = FILES
M = [
    # 경계
    ("lo 하나 늦게(끊기기 직전 초 다음 초를 빼먹음)", RS, "floor.max(self.last_ms.div_euclid(1000) + 1)", "floor.max(self.last_ms.div_euclid(1000) + 2)"),
    ("lo 하나 이르게(끊기기 직전 초를 두 번 더함)", RS, "floor.max(self.last_ms.div_euclid(1000) + 1)", "floor.max(self.last_ms.div_euclid(1000))"),
    ("hi 하나 늦게(붙은 첫 초를 두 번 더함)", RS, "(lo, self.first_ms.div_euclid(1000))", "(lo, self.first_ms.div_euclid(1000) + 1)"),
    ("hi 하나 이르게(붙기 직전 초를 빼먹음)", RS, "(lo, self.first_ms.div_euclid(1000))", "(lo, self.first_ms.div_euclid(1000) - 1)"),
    ("창 바닥 143 → 144봉", RS, "let floor = bar_of(self.first_ms) - (WINDOW_BARS as i64 - 1) * BAR_SECONDS;", "let floor = bar_of(self.first_ms) - WINDOW_BARS as i64 * BAR_SECONDS;"),
    ("last_ms 를 «가장 늦은» 대신 «마지막으로 받은» 체결로", SU, "self.last_ms = self.last_ms.max(ts_ms);", "self.last_ms = ts_ms;"),
    # 더하기 대 통째 쓰기
    ("테이프 칸을 더하지 않고 덮어씀", RS, "                        c[j] += v[j];\n                    }\n                    dirty.insert(k);", "                        c[j] = v[j];\n                    }\n                    dirty.insert(k);"),
    ("테이프 칸을 두 번 더함", RS, "                        c[j] += v[j];\n                    }\n                    dirty.insert(k);", "                        c[j] += 2.0 * v[j];\n                    }\n                    dirty.insert(k);"),
    ("1초 칸이 실시간 초를 덮어씀", RS, "            if self.sec.contains_key(&s) {\n                continue;\n            }", "            if false && self.sec.contains_key(&s) {\n                continue;\n            }"),
    ("1초 칸 스냅숏이 테이프보다 먼저", RS, "                Ok(t) if s <= top => t.secs.get(&s).inspect(|_| n_tape += 1),", "                Ok(t) if s <= top && !snap_sec.contains_key(&s) => t.secs.get(&s).inspect(|_| n_tape += 1),"),
    ("1초 칸 범위 660 → 600", RS, "for s in first_s - SUPPLY_1S_SECONDS..first_s {", "for s in first_s - 600..first_s {"),
    # 칸 키 반올림(2026-10-20 SOL 동률)
    ("칸 키: round(·, 10) 없이 곧바로 나눔", RS, 'let px: f64 = format!("{:.10}", pb as f64 * tape_bucket).parse().unwrap_or(f64::NAN);', "let px: f64 = pb as f64 * tape_bucket;"),
    ("칸 키: 동률 짝수 → 0 에서 먼 쪽(SQL round)", RS, "(px / bucket).round_ties_even() as i64", "(px / bucket).round() as i64"),
    ("칸 키: 테이프 칸 그대로(정수배 안 올림)", RS, "let k = *key.entry(pb).or_insert_with(|| tape_key(pb, tape_bucket, bucket));", "let k = pb;"),
    # 판 번호·스냅숏
    ("판 번호 무시", RS, "if s.v != SNAP_V || s.symbol", "if s.symbol"),
    ("스냅숏 143봉 자르기 안 함", RS, "    s.bars.retain(|b| b.0 >= cutoff);\n", "    let _ = cutoff;\n"),
    ("스냅숏 1초 칸 안 씀", RS, "        self.rs.snap_sec = s.sec.into_iter().collect();", "        let _ = s.sec;"),
    ("스냅숏 dhl 안 씀", RS, "(b, Bar { cells: cells.into_iter().collect(), dhl })", "(b, Bar { cells: cells.into_iter().collect(), dhl: dhl.filter(|_| false) })"),
    ("메우는 중에도 스냅숏을 씀", RS, "if !self.rs.merged || self.rs.job.is_some() || self.last_ms == 0 {", "if self.last_ms == 0 {"),
    # dhl
    ("dhl 테이프 값 대신 언제나 스냅숏 값", RS, "t.dhl.get(&b).and_then(|v| dhl_ok(*v, bar)).or(mine)", "mine.or(t.dhl.get(&b).and_then(|v| dhl_ok(*v, bar)))"),
    ("dhl 테이프 값 어긋남 검사 안 함", RS, "t.dhl.get(&b).and_then(|v| dhl_ok(*v, bar)).or(mine)", "t.dhl.get(&b).copied().or(mine)"),
    ("dhl 링 가지 없음(테이프가 덜 따라온 봉)", RS, "                _ if b >= first_s - SUPPLY_1S_SECONDS => dhl_of(&self.sec, b, bar),\n", ""),
    ("dhl 최저 0 후보 뺌(min(0, lo) 아님)", RS, "r3(if lo_ < 0.0 { lo_ } else { 0.0 })", "r3(lo_)"),
    # 구멍
    ("구멍: 수집기 gaps 안 봄", RS, "                if !good.contains(&m) && x < y {", "                if false && !good.contains(&m) && x < y {"),
    ("구멍: 테이프가 덜 따라온 몫 안 남김", RS, "        if t.top < hi - 1 {\n            t.holes.push", "        if false && t.top < hi - 1 {\n            t.holes.push"),
    ("구멍: 테이프 못 읽어도 안 남김", RS, "                if lo < hi {\n                    self.fp.holes.push((lo, hi));", "                if false && lo < hi {\n                    self.fp.holes.push((lo, hi));"),
    ("구멍: 메우는 중 표시 안 함", RS, "            self.fp.pending = Some((lo, hi));", "            let _ = (lo, hi);"),
    ("구멍: 밀기가 holes 바뀜을 안 실음", FP, "        if self.dirty.is_empty() && self.dropped.is_empty() && !self.holes_dirty {", "        if self.dirty.is_empty() && self.dropped.is_empty() {"),
    ("합친 1초 칸을 밀지 않음", ST, "        if let Some(up) = up {\n            g.first", "        if let Some(up) = up.filter(|_| false) {\n            g.first"),
]
ENV = dict(os.environ, PATH=os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"])
TESTS = ["--offline", "--test", "restore_parity"]


def main() -> None:
    orig = {f: (R / f).read_bytes() for f in FILES}
    only = set(sys.argv[1:])

    def restore() -> None:
        for f, b in orig.items():
            (R / f).write_bytes(b)
        bad = [f for f, b in orig.items() if (R / f).read_bytes() != b]
        if bad:
            sys.exit(f"🔴되돌림 실패: {bad}")

    caught = 0
    try:
        for i, (name, f, old, new) in enumerate(M):
            if only and str(i) not in only:
                continue
            src = orig[f].decode()
            assert src.count(old) == 1, (name, src.count(old))
            time.sleep(1.1)       # cargo 는 mtime 으로 다시 빌드를 정한다
            (R / f).write_text(src.replace(old, new))
            b = subprocess.run(["cargo", "test", "--no-run", "-j", "2", *TESTS], cwd=R, env=ENV, capture_output=True, text=True)
            if b.returncode:
                print(f"{i:2d} {name}: 🔴변형이 빌드 안 됨(대조 무효)\n{b.stderr[-400:]}", flush=True)
                restore()
                continue
            r = subprocess.run(["bash", "-c", f"ulimit -v 4000000; timeout 300 cargo test -j 2 {' '.join(TESTS)} -- --test-threads 2"],
                               cwd=R, env=ENV, capture_output=True, text=True)
            lines = (r.stdout + r.stderr).splitlines()
            why = next((lines[k + 1].strip() for k, ln in enumerate(lines) if "panicked at" in ln and k + 1 < len(lines)), "")
            caught += bool(r.returncode)
            failed = [ln.split()[1] for ln in lines if ln.startswith("test ") and ln.endswith("FAILED")]
            print(f"{i:2d} {name}: {'잡음' if r.returncode else '못 잡음'} {failed} | {why[:150]}", flush=True)
            time.sleep(1.1)
            restore()
    finally:
        restore()
    print(f"양성 대조: {caught} 잡음 / {len(M) if not only else len(only)} · 소스는 사본과 바이트까지 같다")


if __name__ == "__main__":
    main()
