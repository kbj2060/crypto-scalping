#!/usr/bin/env python3
"""1초 수급 «나머지 원천» 일치 시험의 양성 대조 (2026-11-20): 규칙을 하나씩 일부러 틀리게 바꿔 시험이 실패하는지 본다.

소스 사본을 먼저 떠 두고, 변형마다 «바꿈 → 빌드(--no-run, 상한 없음) → 실행(ulimit -v 4GB · timeout 300) → 사본으로 되돌림 →
바이트 비교»를 한다. 되돌린 파일이 사본과 다르면 곧바로 멈춘다. 중간에 죽어도 finally 가 사본을 쓴다.
    python3 rtcore/tools/mutate_supply1s_venues.py          # 전부(약 28번 빌드)
    python3 rtcore/tools/mutate_supply1s_venues.py 0 5      # 번호만
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parents[1]
FILES = ["src/feed.rs", "src/supply.rs"]
M = [
    ("force S=SELL→롱 뒤집기", "src/feed.rs", 'let long = o.get("S").and_then(Value::as_str) == Some("SELL");', 'let long = o.get("S").and_then(Value::as_str) == Some("BUY");'),
    ("force z/q 순서", "src/feed.rs", 'py_or(&[o.get("z"), o.get("q")])', 'py_or(&[o.get("q"), o.get("z")])'),
    ("force ap/p 순서", "src/feed.rs", 'py_or(&[o.get("ap"), o.get("p")])', 'py_or(&[o.get("p"), o.get("ap")])'),
    ('"0" 문자열을 거짓으로', "src/feed.rs", 'Value::String(s) => !s.is_empty(),', 'Value::String(s) => !s.is_empty() && s != "0",'),
    ("T 없을 때 수신 시각 대신 0", "src/feed.rs", 'map_or(Some(rx_us.div_euclid(1000)), py_int)', 'map_or(Some(0), py_int)'),
    ("liq 창 오른쪽 끝 포함", "src/supply.rs", '|s| bn - SUPPLY_1S_SECONDS < s && s < bn);', '|s| bn - SUPPLY_1S_SECONDS < s && s <= bn);'),
    ("USD round() 대신 내림", "src/supply.rs", 'x.round_ties_even() as i64', 'x as i64'),
    ("청산 롱/숏 칸 뒤바뀜", "src/supply.rs", 'let i = (!e.long) as usize;', 'let i = e.long as usize;'),
    ("현물 m 뒤집기", "src/feed.rs", 'let sell = t.get("m").is_some_and(py_truthy);', 'let sell = !t.get("m").is_some_and(py_truthy);'),
    ("현물·OKX 고래 경계 >= → >", "src/supply.rs", '        if notional >= whale_min {\n            c[2 + i] += qty;', '        if notional > whale_min {\n            c[2 + i] += qty;'),
    ("현물·OKX 리테일 경계 < → <=", "src/supply.rs", '        } else if notional < retail_max {\n            c[i] += qty;\n        }\n        lane', '        } else if notional <= retail_max {\n            c[i] += qty;\n        }\n        lane'),
    ("현물 가격 0 안 거름", "src/feed.rs", '    if !(px > 0.0 && qty > 0.0) {\n        SKIPPED.fetch_add(1, Relaxed);\n        return None;\n    }\n    let ts = py_ts(&t, "T", rx_us)?;', '    let ts = py_ts(&t, "T", rx_us)?;'),
    ("OKX ctVal 안 곱함", "src/feed.rs", 'let qty = sz * ct;\n                let side', 'let qty = sz;\n                let side'),
    ("OKX side 뒤집기", "src/feed.rs", 'sell: side == "sell" }', 'sell: side == "buy" }'),
    ("OKX 청산 posSide 뒤집기", "src/feed.rs", '== Some("long");\n                    out.push', '== Some("short");\n                    out.push'),
    ("OKX 청산 다른 종목 안 거름", "src/feed.rs", 'let (Some(sym), Some(ct)) = (okx_sym(inst), okx_ct_val(inst)) else { continue };', 'let (sym, ct) = ("ethusdt".to_string(), 0.1);'),
    ("okxOi 창 경계 > → >=", "src/supply.rs", 'self.okx_oi.range(oi_top - SUPPLY_1S_SECONDS + 1..)', 'self.okx_oi.range(oi_top - SUPPLY_1S_SECONDS..)'),
    ("okxOi 정리 하나 어긋남", "src/supply.rs", 'self.okx_oi = self.okx_oi.split_off(&(s - SUPPLY_1S_SECONDS));', 'self.okx_oi = self.okx_oi.split_off(&(s - SUPPLY_1S_SECONDS + 1));'),
    ("OKX oi=0 안 거름", "src/feed.rs", 'if oi > 0.0 && ts > 0 =>', 'if ts > 0 =>'),
    ("okxLiq 창을 바이낸스 초로", "src/supply.rs", '|s| ok - SUPPLY_1S_SECONDS < s && s < ok);', '|s| bn - SUPPLY_1S_SECONDS < s && s < bn);'),
    ("초 칸 정리 하나 어긋남(공통)", "src/supply.rs", '*m = m.split_off(&(s - SUPPLY_1S_SECONDS));', '*m = m.split_off(&(s - SUPPLY_1S_SECONDS + 1));'),
    ("Bybit Buy=롱 뒤집기", "src/feed.rs", 'long: side == "Buy"', 'long: side == "Sell"'),
    ("Bybit 창 끝 포함 → 제외", "src/supply.rs", '|s| b - SUPPLY_1S_SECONDS < s && s <= b);', '|s| b - SUPPLY_1S_SECONDS < s && s < b);'),
    ("Bybit 3초 지연 없음", "src/supply.rs", 'const BYBIT_LAG_S: i64 = 3;', 'const BYBIT_LAG_S: i64 = 0;'),
    ("Bybit 못 읽는 건만 빼기", "src/feed.rs", 'data.iter().map(one).collect();', 'Some(data.iter().filter_map(one).collect());'),
    ("Bybit 시각 순 대신 받은 순", "src/supply.rs", 'let i = self.bybit.partition_point(|f| f.ts_ms <= ts_ms);', 'let i = self.bybit.len();'),
    ("meta 나이 반올림 없음", "src/supply.rs", 'r1(f64::max(0.0', '(f64::max(0.0'),
    ("meta errors 안 셈", "src/supply.rs", 'lane.errors += (!up) as u64;', 'lane.errors += 0;'),
]
ENV = dict(os.environ, PATH=os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"])
TESTS = ["--test", "supply1s_venues_parity", "--test", "supply1s_parity"]


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
            time.sleep(1.1)       # cargo 는 mtime 으로 다시 빌드를 정한다 -- 같은 순간 쓰기면 옛 빌드를 쓸 수 있다
            (R / f).write_text(src.replace(old, new))
            b = subprocess.run(["cargo", "test", "--no-run", "-j", "2", *TESTS], cwd=R, env=ENV, capture_output=True, text=True)
            if b.returncode:
                print(f"{i:2d} {name}: 🔴변형이 빌드 안 됨(대조 무효)\n{b.stderr[-400:]}", flush=True)
                restore()
                continue
            r = subprocess.run(["bash", "-c", f"ulimit -v 4000000; timeout 300 cargo test -j 2 {' '.join(TESTS)}"],
                               cwd=R, env=ENV, capture_output=True, text=True)
            out = r.stdout + r.stderr
            why = next((ln.strip() for ln in out.splitlines() if ": rust " in ln or "길이" in ln or "rust 에 없다" in ln), "")
            caught += bool(r.returncode)
            print(f"{i:2d} {name}: {'잡음' if r.returncode else '못 잡음'} | {why[:160]}", flush=True)
            time.sleep(1.1)
            restore()
    finally:
        restore()
    print(f"양성 대조: {caught} 잡음 / {len(M) if not only else len(only)} · 소스는 사본과 바이트까지 같다")


if __name__ == "__main__":
    main()
