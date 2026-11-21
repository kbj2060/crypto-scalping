#!/usr/bin/env python3
"""rtduck 양성 대조: 변이 하나씩 넣고 시험이 실패하는지. 🔴제자리에서 고치고 되돌린다(사본은 libduckdb 를 다시 빌드한다 -- C++ 11~14분):
끝나면(예외여도) 원본 바이트로 되돌리고 src·tests 해시가 같은지 본다.   python3 rtduck/tools/mutate_check.py [이름 조각…]"""
import hashlib
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = [
    ("무변경(대조: 통과해야 함)", "src/oi.rs", "pub const PERIOD_S: f64 = 300.0;", "pub const PERIOD_S: f64 = 300.0;"),
    ("롱숏비 COALESCE 빼기(새 NULL 이 기존 값을 지움)", "src/oi.rs", "p.push(r.vals[j].or_else(|| e.and_then(|e| e.1[j])).into());", "p.push(r.vals[j].into());"),
    ("롱숏비 sources_ok 큰 쪽 → 새 값", "src/oi.rs", "p.push(r.sources_ok.max(e.and_then(|e| e.0).unwrap_or(0)).into());", "p.push(r.sources_ok.into());"),
    ("롱숏비 테이커 매수·매도 바꿈", "src/oi.rs", '&[("buySellRatio", 11), ("buyVol", 12), ("sellVol", 13)]', '&[("buySellRatio", 11), ("buyVol", 13), ("sellVol", 12)]'),
    ("롱숏비 쿼리 순서", "src/oi.rs", '"{base}{path}?symbol={}&period=5m&limit={limit}"', '"{base}{path}?symbol={}&limit={limit}&period=5m"'),
    ("롱숏비 못 읽는 칸 = 그 칸만 NULL(원본은 그 폴 전체를 버림)", "src/oi.rs", 'v[k] = Some(pyfloat(r.get(*key).ok_or_else(|| format!("KeyError: \'{key}\'"))?)?);', 'v[k] = r.get(*key).and_then(|x| pyfloat(x).ok());'),
    ("HL체결 cut 같음도 남김", "src/hl_trades.rs", "filter(|r| r.1 > self.cut)", "filter(|r| r.1 >= self.cut)"),
    ("HL체결 2일 전 파일 압축", "src/hl_trades.rs", "wall - 3.0 * 86400.0", "wall - 2.0 * 86400.0"),
    ("HL체결 20초 → 10초 플러시", "src/hl_trades.rs", "pub const FLUSH_SEC: f64 = 20.0;", "pub const FLUSH_SEC: f64 = 10.0;"),
    ("HL체결 공백 = 마지막 끊김 시각", "src/hl_trades.rs", "        if self.down.is_none() {\n            self.down", "        if true {\n            self.down"),
    ("HL포지션 429 뒤에도 간격 잠", "src/hl_positions.rs", "                    continue; // 원본: 간격 잠 없이 다음 주소", "                    env.sleep(REQ_GAP_S);\n                    continue;"),
    ("HL포지션 청산가 여유 0.5% → 0", "src/hl_positions.rs", "pub const LIQ_TOL: f64 = 0.005;", "pub const LIQ_TOL: f64 = 0.0;"),
    ("HL포지션 본 tid 다시 적음", "src/hl_positions.rs", ".filter(|r| !self.seen_tids.contains(&r.1))", ".filter(|_| true)"),
    ("HL포지션 청산 주소 대소문자 가림", "src/hl_positions.rs", ".to_lowercase() != user.to_lowercase()", " != user"),
    ("HL포지션 순위 거꾸로", "src/hl_positions.rs", "agg.sort_by(|a, b| b.1.partial_cmp(&a.1)", "agg.sort_by(|a, b| a.1.partial_cmp(&b.1)"),
    ("HL포지션 파싱 실패 주소는 응답 수에서 뺌", "src/hl_positions.rs", "                    ok += 1;\n                    let ts", "                    let ts"),
    ("시장 np.std → 표본 표준편차(n-1)", "src/market.rs", "(rtcollect::ob5m::np_sum(&sq) / a.len() as f64).sqrt()", "(rtcollect::ob5m::np_sum(&sq) / (a.len() as f64 - 1.0).max(1.0)).sqrt()"),
    ("시장 Hawkes $1만 하한 빼기", "src/market.rs", "if zp >= self.z_threshold && peak_usd >= self.min_trigger {", "if zp >= self.z_threshold {"),
    ("시장 롱·숏 청산 바꿈", "src/market.rs", "if *sell { l += usd } else { s += usd }", "if *sell { s += usd } else { l += usd }"),
    ("시장 점심시간 안 뺌", "src/market.rs", "if v.len() == 4 && at(2) <= t_min && t_min < at(3) {", "if false {"),
    ("시장 마감 분 포함 안 함(<)", "src/market.rs", "at(0) <= t_min && t_min <= at(1)", "at(0) <= t_min && t_min < at(1)"),
    ("시장 마이크로초 0 이어도 소수부", "src/market.rs", "if us == 0 { format!(\"{base}{tail}\") } else {", "if false { format!(\"{base}{tail}\") } else {"),
    ("시장 분 저장 15초 → 10초", "src/market.rs", "minute != self.last_saved_minute && second < 15", "minute != self.last_saved_minute && second < 10"),
    ("시장 깨진 메시지도 마지막 수신 시각", "src/market.rs", "        let Ok(Value::Object(m)) = serde_json::from_str::<Value>(raw) else { return };", "        self.last_msg = now;\n        let Ok(Value::Object(m)) = serde_json::from_str::<Value>(raw) else { return };"),
    ("미시 percentile 위쪽 빼기 길 없앰", "src/micro.rs", "if g >= 0.5 { y - d * (1.0 - g) } else { x + d * g }", "x + d * g"),
    ("미시 호가 앞 10단 → 12단", "src/micro.rs", "self.bids.iter().take(10)", "self.bids.iter().take(12)"),
    ("미시 체결 흐름 무효여도 taker_buy_ratio 씀", "src/micro.rs", "vt.then(|| f(\"taker_buy_ratio\"))", "Some(f(\"taker_buy_ratio\"))"),
    ("미시 부트스트랩 60분 → 59분", "src/micro.rs", "query_map([cut(3600.0)]", "query_map([cut(3540.0)]"),
    ("미시 aggTrades 대체 30초 → 60초", "src/micro.rs", "if now() - self.trade_last > 30.0 {", "if now() - self.trade_last > 60.0 {"),
    ("미시 대체 체결 같은 id 도 다시", "src/micro.rs", "if self.last_agg_id.is_some_and(|l| id <= l) {", "if self.last_agg_id.is_some_and(|l| id < l) {"),
    ("미시 squeeze 펀딩 0.0008 → 0.001", "src/micro.rs", "if eai > self.eai_threshold && f.abs() > 0.0008 {", "if eai > self.eai_threshold && f.abs() > 0.001 {"),
    ("미시 고래 이탈 문턱 = 진입 문턱", "src/micro.rs", "let exit = self.exit_th.min(entry - 1e-6);", "let exit = entry;"),
    ("미시 깊이 시각 = 호가가 읽힌 뒤에만", "src/micro.rs", "        self.depth_last = now;\n        let side", "        let side"),
    ("HL체결 주소 소문자 안 함", "src/hl_trades.rs", "py_str(&users[0]).to_lowercase()", "py_str(&users[0])"),
    # 2026-11-21 여덟째 조각 -- 다섯째 칸 = 돌릴 시험만(없으면 전부)
    ("래스터보관 -0.0 칸도 씀", "src/raster_archive.rs", "if q != 0.0 {", "if q.to_bits() != 0 {", ["raster_archive_parity"]),
    ("래스터보관 magic 안 봄", "src/raster_archive.rs", "if &raw[..4] != MAGIC {", "if false {", ["raster_archive_parity"]),
    ("래스터보관 보관본 있어도 다시 씀", "src/raster_archive.rs", "if nonempty(&dst) || nonempty(&gz) {", "if false {", ["raster_archive_parity"]),
    ("래스터보관 14일 → 13일", "src/raster_archive.rs", "now_s - (retention_days * 86_400) as f64", "now_s - ((retention_days - 1) * 86_400) as f64", ["raster_archive_parity"]),
    ("래스터보관 mid 행 = 0 아닌 칸만", "src/raster_archive.rs", "            am.append_row(params![ts, mid]).map_err(|e| e.to_string())?;", "            if nb > 0 && raw[o + 16] != 0 { am.append_row(params![ts, mid]).map_err(|e| e.to_string())?; }", ["raster_archive_parity"]),
    ("청산탭 groupby Kahan → 그냥 합", "src/liq_tab.rs", "kahan(m.entry(k).or_insert((0.0, 0.0)), v);", "m.entry(k).or_insert((0.0, 0.0)).0 += v;", ["liq_tab_parity"]),
    # «청산탭 8192칸 묶음 없앰» → rtcollect mutate_check «np.sum 8192칸 묶음 없앰»(묶음을 공유 np_sum 으로 옮김 · 2026-11-21 아홉째 조각)
    ("청산탭 convolve 차례 뒤집음", "src/liq_tab.rs", "for x in &v[j.saturating_sub(5)..(j + 6).min(m)] {", "for x in v[j.saturating_sub(5)..(j + 6).min(m)].iter().rev() {", ["liq_tab_parity"]),
    # (뺌) «청산탭 np.round → 파이썬 round»: 사실상 같은 변이 -- rint(x·10^d)/10^d 와 정확 반올림은 x·10^d 의 곱셈 반올림이 .5 경계를 넘을 때만 갈린다
    #   (값 ~1e7·d=2 면 값 하나에 ~1e-7 · share 단위 d=8 은 ~1e-11 확률). 2026-11-21 돌린 결과 «통과»(못 잡음) -- 합성으로 그 경계를 만들 길이 없다.
    ("청산탭 히트맵 컷 0.1 → 0.05", "src/liq_tab.rs", "const HEAT_CUT: f64 = 0.1;", "const HEAT_CUT: f64 = 0.05;", ["liq_tab_parity"]),
    ("청산탭 자기 봉도 닿음", "src/liq_tab.rs", "        let mut j = i + 1;\n        while j < n {", "        let mut j = i;\n        while j < n {", ["liq_tab_parity"]),
    ("청산탭 98 → 97분위", "src/liq_tab.rs", "let q = quantile(&pos, 0.98);", "let q = quantile(&pos, 0.97);", ["liq_tab_parity"]),
    ("청산탭 짝수 중앙값 = 위 값", "src/liq_tab.rs", "if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }", "s[n / 2]", ["liq_tab_parity"]),
    ("청산탭 바이비트 OI 10분 → 100분", "src/liq_tab.rs", "let since = now_ms - 600_000;", "let since = now_ms - 6_000_000;", ["liq_tab_parity"]),
    ("청산탭 띠 30일 → 29일", "src/liq_tab.rs", "days[days.len().saturating_sub(30)..]", "days[days.len().saturating_sub(29)..]", ["liq_tab_parity"]),
    ("청산탭 12시간 앞 봉 무게 안 지움", "src/liq_tab.rs", ".take(m.saturating_sub(h))", ".take(0)", ["liq_tab_parity"]),
    ("봉인 RESEAL 8일 → 0일", "src/seal.rs", "pub const RESEAL_DAYS: i64 = 8;", "pub const RESEAL_DAYS: i64 = 0;", ["seal_parity"]),
    ("봉인 적은 원천도 덮음", "src/seal.rs", "                    if n < on {", "                    if false {", ["seal_parity"]),
    ("봉인 LAG 2시간 → 0", "src/seal.rs", "pub const LAG_HOURS: i64 = 2;", "pub const LAG_HOURS: i64 = 0;", ["seal_parity"]),
    ("봉인 날짜 하루 밀림", "src/seal.rs", "cast(floor(({ts_sql}) / 86400000.0) as integer)", "cast(floor(({ts_sql}) / 86400000.0) as integer) + 1", ["seal_parity"]),
    ("HL체결 가짜 시계 = 벽시계 + 차이", "src/hl_trades.rs", "        Some((i0, t)) => t + i0.elapsed().as_secs_f64(),", "        Some((_, t)) => *t + 0.0 * rtcollect::rest::wall_s(),", ["hl_trades_loop"]),
    ("봉인 SQLite INT → DOUBLE", "src/seal.rs", "    Ok(if d.contains(\"int\") {\n        \"BIGINT\"", "    Ok(if d.contains(\"int\") {\n        \"DOUBLE\"", ["seal_full_parity"]),
    ("봉인 정리 OKX 맥락 보존 8 → 7일", "src/seal.rs", "\"ts_ms / 1000\", \"inst\", \"okx\", t.into(), Okx, 8)", "\"ts_ms / 1000\", \"inst\", \"okx\", t.into(), Okx, 7)", ["seal_full_parity"]),
    # (뺌) «봉인 SQLite REAL 에 정수 저장 = 오류»: 같은 변이 -- SQLite 는 REAL 친화 열(REAL·DOUBLE)에 넣은 정수를 REAL 로 바꿔 저장한다(정수로 되읽힐 일이 없다 · 2026-11-21 돌린 결과 «통과»).
    ("deribit butterfly 가운데 = 양끝 합", "src/deribit.rs", "eq(q[1], q[0] + q[2])", "eq(q[1], q[0])", ["deribit_parity"]),
    ("deribit 블록 아닌 베가도 vgb", "src/deribit.rs", "            if *blk {\n                b[6] += vg;", "            if true {\n                b[6] += vg;", ["deribit_parity"]),
    ("deribit 창 24 → 25시간", "src/deribit.rs", "pub const STATE_WINDOW_MS: i64 = 24 * 3600 * 1000;", "pub const STATE_WINDOW_MS: i64 = 25 * 3600 * 1000;", ["deribit_parity"]),
    ("deribit USDC 선형 헤지도 ÷가격", "src/deribit.rs", "let d = if f.instrument_name.contains(\"_USDC\") {", "let d = if false {", ["deribit_parity"]),
    ("deribit 하트비트 답 안 함", "src/main.rs", "                    if p[\"type\"] == \"test_request\" {", "                    if false {", ["deribit_loop"]),
    ("청산탭 앞쪽 메움 안 함", "src/liq_tab.rs", "        if mn > need + DAY {", "        if false {", ["liq_tab_parity"]),
]


def digest():
    h = hashlib.sha256()
    for p in sorted([*ROOT.glob("src/*.rs"), *ROOT.glob("tests/*.rs")]):
        h.update(p.read_bytes())
    return h.hexdigest()


def main():
    env = dict(os.environ, PATH=f"{Path.home()}/.cargo/bin:{os.environ.get('PATH', '')}")
    before, caught, missed = digest(), 0, []
    pick = sys.argv[1:]
    n = 0
    for name, f, a, b, *t in M:
        if pick and name != M[0][0] and not any(x in name for x in pick):
            continue
        n += 1
        targs = " ".join(f"--test {x}" for x in t[0]) if t else ""
        p = ROOT / f
        orig = p.read_bytes()
        s = orig.decode()
        assert s.count(a) == 1, (name, s.count(a))
        try:
            p.write_text(s.replace(a, b))
            r = subprocess.run(["bash", "-c", f"ulimit -v 14000000; timeout 900 cargo test -q --release -j 2 --offline {targs}"], cwd=ROOT, env=env, capture_output=True, text=True)
        finally:
            p.write_bytes(orig)
        failed = r.returncode != 0
        fails = sorted({ln.split()[1] for ln in r.stdout.splitlines() if ln.startswith("test ") and ln.rstrip().endswith("FAILED")})
        if name == M[0][0]:
            assert not failed, r.stdout[-2000:] + r.stderr[-2000:]
        else:
            caught += failed
            if not failed:
                missed.append(name)
        print(f"{'실패(잡음)' if failed else '통과'} | {name} | {','.join(fails)}", flush=True)
    assert digest() == before, "원본 src·tests 가 바뀌었다"
    print(f"잡은 변이 {caught}/{n - 1} · 못 잡은 것 {missed} · 원본 src·tests 바이트 그대로")


if __name__ == "__main__":
    main()
