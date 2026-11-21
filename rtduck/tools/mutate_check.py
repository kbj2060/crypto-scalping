#!/usr/bin/env python3
"""rtduck 양성 대조: 변이 하나씩 넣고 시험이 실패하는지. 🔴제자리에서 고치고 되돌린다(사본은 libduckdb 를 다시 빌드한다 -- C++ 11~14분):
끝나면(예외여도) 원본 바이트로 되돌리고 src·tests 해시가 같은지 본다.   python3 rtduck/tools/mutate_check.py"""
import hashlib
import os
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
    ("HL체결 주소 소문자 안 함", "src/hl_trades.rs", "py_str(&users[0]).to_lowercase()", "py_str(&users[0])"),
]


def digest():
    h = hashlib.sha256()
    for p in sorted([*ROOT.glob("src/*.rs"), *ROOT.glob("tests/*.rs")]):
        h.update(p.read_bytes())
    return h.hexdigest()


def main():
    env = dict(os.environ, PATH=f"{Path.home()}/.cargo/bin:{os.environ.get('PATH', '')}")
    before, caught, missed = digest(), 0, []
    for name, f, a, b in M:
        p = ROOT / f
        orig = p.read_bytes()
        s = orig.decode()
        assert s.count(a) == 1, (name, s.count(a))
        try:
            p.write_text(s.replace(a, b))
            r = subprocess.run(["bash", "-c", "ulimit -v 4000000; timeout 600 cargo test -q --release -j 2 --offline"], cwd=ROOT, env=env, capture_output=True, text=True)
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
    print(f"잡은 변이 {caught}/{len(M) - 1} · 못 잡은 것 {missed} · 원본 src·tests 바이트 그대로")


if __name__ == "__main__":
    main()
