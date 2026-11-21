#!/usr/bin/env python3
"""OKX 풋프린트 봉 복원·청산 원 OKX/Bybit 몫 일치 시험의 양성 대조 (2026-11-21): mutate_footprint.py 의 틀 그대로.
    python3 rtcore/tools/gen_okx_fp_fixture.py && python3 rtcore/tools/gen_restore_venues_fixture.py   # 먼저(커밋 안 하는 정답지)
    python3 rtcore/tools/mutate_okx_fp.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutate_footprint as mf  # noqa: E402

R, S = "src/restore.rs", "src/supply.rs"
mf.FILES = [R, S]
mf.TESTS = ["--test", "okx_fp_restore", "--test", "restore_venues_parity"]
mf.M = [
    ("진행 봉 조건 안 봄", R, "            if !covered(start_bar, t0_min + 60) {", "            if false {"),
    ("진행 봉 조건 t0 분 뺌", R, "covered(start_bar, t0_min + 60)", "covered(start_bar, t0_min)"),
    ("진행 봉 테이프를 더하지 않고 교체", R, "                let t = cells.entry(*k).or_insert([0.0; 6]);\n                for j in 0..6 {\n                    t[j] += v[j];\n                }",
     "                cells.insert(*k, *v);"),
    ("앞 봉 교체 대신 더함(다시 읽을 때 두 번)", R, "            if self.okx_fr.restored.insert(b) {", "            if { self.okx_fr.restored.insert(b); true } {"),
    ("창 끝 하나 어긋남", R, "        while b >= lo && covered(b, b + 300) {", "        while b > lo && covered(b, b + 300) {"),
    ("first_bar = earliest(−300 빠짐)", R, "        self.okx_first_bar = earliest - 300;", "        self.okx_first_bar = earliest;"),
    ("청산 덮은 구간 대신 전부", R, "filter(|e| earliest * 1000 <= e.0 && e.0 < upto)", "filter(|e| e.0 < upto)"),
    ("청산 liq_upto 안 옮김", R, "        self.okx_fr.liq_upto = Some(earliest * 1000);", "        self.okx_fr.liq_upto = Some(t0);"),
    ("OI 진행 봉 처음값 안 고침", R, "                self.okx_oi5m.entry(ob).or_insert([o, c])[0] = o;", "                self.okx_oi5m.entry(ob).or_insert([o, c]);"),
    ("OI 창 앞 봉 안 넣음", R, "            if earliest <= ob && ob < start_bar {", "            if earliest < ob && ob < start_bar {"),
    ("테이프 t0 조건 안 봄", R, "        if got.tape_max < t0.div_euclid(1000) {", "        if false {"),
    ("최근 15분 길 없음", R, "m >= now_s - 900", "m >= now_s"),
    ("최근 15분 길이 공백 안 봄", R, "&& !gaps.iter().any(|&(a, b)| a < (m + 60) * 1000 && b > m * 1000)", ""),
    ("1분봉 허용 1e-4 → 1e-6", R, "if f(r, 1).map_err(err)?.abs() <= 1e-4 {", "if f(r, 1).map_err(err)?.abs() <= 1e-6 {"),
    ("칸 키 round(·,10) 빠짐", R, "(r10(pb as f64 * tape_bucket) / bucket)", "((pb as f64 * tape_bucket) / bucket)"),
    ("8회차 → 7회차", R, "pub const OKX_FP_ATTEMPTS: u32 = 8;", "pub const OKX_FP_ATTEMPTS: u32 = 7;"),
    ("라이브 봉 칸: 주문 단위 몫 안 셈", S, "                self.okx_fp.add(bar, px, qty, sell, true, whale_min, retail_max);", ""),
    ("라이브 first_bar 늘 덮어씀", S, "            if self.okx_first_bar == 0 {\n                (self.okx_first_bar, self.okx_t0_ms) = (bar, ts_ms);",
     "            if true {\n                (self.okx_first_bar, self.okx_t0_ms) = (bar, ts_ms);"),
    ("청산 원 OKX 경계 > → >=", S, "                if b > self.okx_first_bar {", "                if b >= self.okx_first_bar {"),
    ("청산 원 Bybit 시작 봉도 넣음", S, "self.bybit5m.range(f + 1..)", "self.bybit5m.range(f..)"),
    ("Bybit 롱·숏 뒤바뀜", S, "    a[if long { 0 } else { 1 }] += usd;\n    a[2] += 1.0;\n    let since", "    a[if long { 1 } else { 0 }] += usd;\n    a[2] += 1.0;\n    let since"),
]

if __name__ == "__main__":
    mf.main()
