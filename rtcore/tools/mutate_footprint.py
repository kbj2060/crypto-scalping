#!/usr/bin/env python3
"""풋프린트 일치 시험의 양성 대조 (2026-11-20): 규칙을 하나씩 일부러 틀리게 바꿔 시험이 실패하는지 본다.

mutate_supply1s_venues.py 와 같은 방식: 소스 사본을 먼저 떠 두고 변형마다 «바꿈 → 빌드(--no-run) → 실행(ulimit -v 4GB · timeout 300)
→ 사본으로 되돌림 → 바이트 비교». 되돌린 파일이 사본과 다르면 곧바로 멈춘다. 중간에 죽어도 finally 가 사본을 쓴다.
    python3 rtcore/tools/mutate_footprint.py          # 전부
    python3 rtcore/tools/mutate_footprint.py 0 5      # 번호만
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parents[1]
FILES = ["src/footprint.rs", "src/supply.rs", "src/proto.rs"]
FP, SU, PR = FILES
M = [
    ("가격 칸 키 동률 짝수 → 0 에서 먼 쪽", FP, "(px / bucket).round_ties_even() as i64", "(px / bucket).round() as i64"),
    ("봉 칸 매수/매도 뒤바뀜", FP, "            cell[side] += qty;", "            cell[1 - side] += qty;"),
    ("봉 고래 경계 >= → >", FP, "            if notional >= whale_min {", "            if notional > whale_min {"),
    ("봉 리테일 경계 < → <=", FP, "            } else if notional < retail_max {", "            } else if notional <= retail_max {"),
    ("24시간 창 하나 어긋남", FP, "bar_of(now_ms) - (KEEP_BARS - 1) * BAR_SECONDS", "bar_of(now_ms) - KEEP_BARS * BAR_SECONDS"),
    ("보관 봉 수 288 → 287", FP, "pub const KEEP_BARS: i64 = 288;", "pub const KEEP_BARS: i64 = 287;"),
    ("새 봉 때 오래된 봉 안 지움", FP, "let kept = self.bars.split_off(&cutoff);", "let kept = self.bars.split_off(&i64::MIN);"),
    ("창 144 → 143", FP, "pub const WINDOW_BARS: usize = 144;", "pub const WINDOW_BARS: usize = 143;"),
    ("응답 가격 소수 자리 +1", FP, 'format!("{:.*}", dp, k as f64 * bucket)', 'format!("{:.*}", dp + 1, k as f64 * bucket)'),
    ("칸 값 반올림 없음", FP, "(px, r3(c[0]), r3(c[1])", "(px, c[0], r3(c[1])"),
    ("dhl 어긋남 허용 0.5 → 0.05", FP, "f64::max(0.5, 0.01 * cd.abs())", "f64::max(0.05, 0.01 * cd.abs())"),
    ("dhl 봉 시작 0 을 후보에서 뺌", FP, "let (mut c, mut lo, mut hi) = (0.0f64, 0.0f64, 0.0f64);", "let (mut c, mut lo, mut hi) = (0.0f64, f64::INFINITY, f64::NEG_INFINITY);"),
    ("dhl 칸 델타 부호", FP, "bar.cells.values().map(|c| c[0] - c[1]).sum()", "bar.cells.values().map(|c| c[1] - c[0]).sum()"),
    # +1 쪽은 값이 안 바뀐다(링이 아직 봉을 다 든 채 1초 일찍 얼릴 뿐) -- 틀리는 쪽(1초 늦게 = 지운 뒤 얼림)을 본다
    ("링 바닥 하나 어긋남(1초 늦게 얼림)", FP, "map(|t| t - SUPPLY_1S_SECONDS)", "map(|t| t - SUPPLY_1S_SECONDS - 1)"),
    ("스냅숏: 링이 드는 봉도 얼린 값", FP, "floor.is_some_and(|f| f <= b)", "floor.is_some_and(|f| f + BAR_SECONDS <= b)"),
    ("밀기: 링이 드는 봉 다시 재지 않음", FP, "for (&b, bar) in self.bars.range_mut(f..) {", "for (&b, bar) in self.bars.range_mut(f..).filter(|_| false) {"),
    ("밀기: drop 안 실음", FP, "self.dropped.push(b);", "let _ = b;"),
    ("밀기: 바뀐 칸 안 실음(봉만)", FP, "self.dirty.entry(bar).or_default().insert(k);", "self.dirty.entry(bar).or_default();"),
    ("dhl 안 얼림", SU, "            if s - SUPPLY_1S_SECONDS > f {", "            if false && s - SUPPLY_1S_SECONDS > f {"),
    ("얼린 뒤 늦은 체결에도 안 지움", SU, "if !order && bar < s.max(newest(&self.sec)) - SUPPLY_1S_SECONDS {", "if false && bar < s.max(newest(&self.sec)) - SUPPLY_1S_SECONDS {"),
    ("주문 칸 늦게 와도 지움", SU, "if !order && bar < s.max(newest(&self.sec)) - SUPPLY_1S_SECONDS {", "if bar < s.max(newest(&self.sec)) - SUPPLY_1S_SECONDS {"),
    ("수급 묶기: 옛 partial 안 내림", PR, "upsert(rows, vec![o], |r| r.0);", "let _ = (rows, o);"),
    ("수급 묶기: 같은 초면 앞의 것 유지", PR, "Some(i) => rows[i] = x,", "Some(_) => {}"),
]
ENV = dict(os.environ, PATH=os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"])
TESTS = ["--test", "footprint_parity", "--test", "supply1s_parity", "--test", "supply1s_venues_parity"]


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
            r = subprocess.run(["bash", "-c", f"ulimit -v 4000000; timeout 300 cargo test -j 2 {' '.join(TESTS)}"],
                               cwd=R, env=ENV, capture_output=True, text=True)
            lines = (r.stdout + r.stderr).splitlines()
            why = next((lines[k + 1].strip() for k, ln in enumerate(lines) if "panicked at" in ln and k + 1 < len(lines)), "")
            caught += bool(r.returncode)
            print(f"{i:2d} {name}: {'잡음' if r.returncode else '못 잡음'} | {why[:150]}", flush=True)
            time.sleep(1.1)
            restore()
    finally:
        restore()
    print(f"양성 대조: {caught} 잡음 / {len(M) if not only else len(only)} · 소스는 사본과 바이트까지 같다")


if __name__ == "__main__":
    main()
