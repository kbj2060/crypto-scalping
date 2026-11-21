#!/usr/bin/env python3
"""양성 대조: src/fpd.rs 규칙을 하나씩 틀리게 바꿔 tests/fpd_parity.rs(--release)가 실패하는지 본다 (2026-11-22).
fpd.rs 는 아직 lib 에 안 붙어 시험 실행 파일만 다시 빈다 -- 사본·별도 target 없이 그 자리에서 바꾸고 바로 되돌린다(끝에 바이트 대조).
    python3 rtserve/tools/mutate_fpd.py [이름 일부 ...]
"""
import os
import subprocess
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parents[1]
F = R / "src" / "fpd.rs"
M = [
    ("무변경(대조: 통과해야 함)", "pub const TODAY_TTL: f64 = 30.0;", "pub const TODAY_TTL: f64 = 30.0;"),
    ("np.sum 대신 차례 합", "    rtcollect::ob5m::np_sum(a)\n", "    a.iter().sum()\n"),
    ("Kahan 대신 단순 합", "        let y = v - self.c;", "        let y = v - 0.0 * self.c;"),
    ("np.round 대신 파이썬 round", "    (x * 1000.0).round_ties_even() / 1000.0", "    rtserve_round(x, 3)"),
    ("argsort 안정 → 불안정", "    o.sort_by_key(|&i| t.sec[i]); // stable", "    o.sort_unstable_by_key(|&i| t.sec[i]); // stable"),
    ("VWAP 무게 0 이면 첫 값 대신 0", "            px[idx[0]]\n", "            0.0\n"),
    ("시가 = 끝 초", "        open: vw(sec[0]),", "        open: vw(sec[n - 1]),"),
    ("고저에 체결 없는 칸 포함", "    let has: Vec<f64> = (0..n).filter(|&i| q[i] > 0.0).map(|i| px[i]).collect();", "    let has: Vec<f64> = (0..n).map(|i| px[i]).collect();"),
    ("칸 묶기 반올림", "        let b = t.pbin[o[i]].div_euclid(sp.tape_per_cell);", "        let b = (t.pbin[o[i]] as f64 / sp.tape_per_cell as f64).round() as i64;"),
    ("열 반올림 4 → 3", "    let r4 = |x: f64| V::from(py_round(x, 4));", "    let r4 = |x: f64| V::from(py_round(x, 3));"),
    ("소급본이 안 이김", "        rows.entry(d.clone()).or_insert_with(|| Row { s: Summ { oi: None, ..s.clone() }, src: \"t\" });",
     "        rows.insert(d.clone(), Row { s: Summ { oi: None, ..s.clone() }, src: \"t\" });"),
    ("오늘 OI 를 안 덮음", "            if r.s.oi.is_none() || d == today {", "            if r.s.oi.is_none() {"),
    ("청산 시작 전도 0", "        let lq = liq_day0.as_ref().filter(|d0| d.as_str() >= d0.as_str())", "        let lq = liq_day0.as_ref().filter(|_| true)"),
    ("반쪽 날 표시 빼기", "} else if liq_day0.as_deref() == Some(d.as_str()) { 2 } else { 0 }", "} else { 0 }"),
    ("오늘 TTL 30 → 0", "pub const TODAY_TTL: f64 = 30.0;", "pub const TODAY_TTL: f64 = 0.0;"),
    ("닫힌 날 캐시 안 함(오늘처럼 다시)", "        let closed: Vec<i64> = (start..today).filter(|&d| !st.recent_days.contains_key(&ds(d))).collect();",
     "        let closed: Vec<i64> = (start..today).collect();"),
    ("lake 날을 hot 이 덮음", "        let missing: Vec<String> = (start..=today).map(ds).filter(|d| !have.contains(d)).collect();",
     "        let missing: Vec<String> = (start..=today).map(ds).collect();"),
    ("hot 맥락 lake 날 다음 날부터 → 시작부터", "        let first_hot = loi.keys().filter_map(|d| parse_day(d)).map(|d| d + 1).fold(start, i64::max);", "        let first_hot = start;"),
    ("행 묶기 하한 반올림 없음", "    json!([py_round((lo * row) as f64 * sp.cell, sp.dp),", "    json!([(lo * row) as f64 * sp.cell,"),
    ("칸 날 수 400 → 10", "pub const MAX_CELL_DAYS: usize = 400;", "pub const MAX_CELL_DAYS: usize = 10;"),
    ("4시간 정착 2분 → 0", "pub const H4_SETTLE: i64 = 120;", "pub const H4_SETTLE: i64 = 0;"),
    ("4시간 빈칸 다시 보기 10분 → 1일", "pub const H4_GAP_TTL: f64 = 600.0;", "pub const H4_GAP_TTL: f64 = 86400.0;"),
    ("4시간 빈칸을 «없음»으로 안 굳힘", "                    st.h4_closed.entry(k.clone()).or_insert(None);", "                    let _ = k;"),
    ("4시간 굳은 칸 덮어씀", "                    st.h4_closed.entry(k).or_insert(Some(v));", "                    st.h4_closed.insert(k, Some(v));"),
    ("4시간 다시 세기 7일 전부터", "            let t0 = first.max(missing.iter().chain(live_keys.iter()).map(|k| sec(k)).chain([cur]).min().unwrap());", "            let t0 = first;"),
    ("4시간 청산 롱·숏 뒤바뀜", "            if sd == \"long\" { e.1 = Some(v.s) } else { e.2 = Some(v.s) }", "            if sd == \"long\" { e.2 = Some(v.s) } else { e.1 = Some(v.s) }"),
    ("4시간 OI 칸 첫 값", "            out.entry(h4_key(t.div_euclid(1000))).or_default().0 = Some(v);", "            out.entry(h4_key(t.div_euclid(1000))).or_default().0.get_or_insert(v);"),
    ("4시간 형성 칸 표시 = 직전 칸", "        let cols = build_days(None, &s, &oi, &liq, Some(1), &live_key);", "        let cols = build_days(None, &s, &oi, &liq, Some(1), &h4_key(h4_sec(&live_key).unwrap() - H4_SEC));"),
]

ENV = dict(os.environ, PATH=os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"])


def main() -> None:
    orig = F.read_bytes()
    only = sys.argv[1:]
    caught = n = 0
    try:
        for i, (name, old, new) in enumerate(M):
            if only and not any(o in name for o in only):
                continue
            src = orig.decode()
            assert src.count(old) == 1, (name, src.count(old))
            time.sleep(1.1)
            F.write_text(src.replace(old, new))
            r = subprocess.run(["bash", "-c", "ulimit -v 12000000; timeout 900 cargo test --release --offline -j 2 --test fpd_parity"], cwd=R, env=ENV, capture_output=True, text=True)
            F.write_bytes(orig)
            out = r.stdout + r.stderr
            built = "error[" not in out
            fail = r.returncode != 0
            why = next((ln.strip()[:110] for ln in out.splitlines() if "panicked at" in ln or "다름" in ln or "left:" in ln), "")
            if i == 0:
                print(f"{i:2d} {name}: {'통과(맞음)' if not fail else '🔴무변경이 실패'}", flush=True)
                continue
            n += 1
            caught += fail and built
            print(f"{i:2d} {name}: {'🔴빌드 안 됨' if not built else ('잡음' if fail else '못 잡음')} {why}", flush=True)
    finally:
        F.write_bytes(orig)
    assert F.read_bytes() == orig, "되돌림 실패"
    print(f"양성 대조: {caught} 잡음 / {n} · fpd.rs 는 처음과 바이트까지 같다")


if __name__ == "__main__":
    main()
