#!/usr/bin/env python3
"""30분 상황 근거(situation.rs) 일치 시험의 양성 대조 (2026-11-21): mutate_footprint.py 의 «바꿈 → 빌드 → 실행 → 사본으로 되돌림 → 바이트 비교».
    python3 rtcore/tools/mutate_situation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutate_footprint as mf  # noqa: E402

S = "src/situation.rs"
mf.FILES = [S]
mf.TESTS = ["--test", "situation_parity"]
mf.M = [
    ("슈미트 진입 문턱 0.45 → 0.40", S, "pub const TREND_ENTER: f64 = 0.45;", "pub const TREND_ENTER: f64 = 0.40;"),
    ("슈미트 이탈 문턱 0.25 → 0.30", S, "pub const TREND_EXIT: f64 = 0.25;", "pub const TREND_EXIT: f64 = 0.30;"),
    ("prev_dir 안 이음(늘 0)", S, "    let d = if prev_dir == 0 {", "    let prev_dir = 0 * prev_dir;\n    let d = if prev_dir == 0 {"),
    ("반대 방향 직접 반전 막음", S, "    } else if ratio > TREND_ENTER {\n        sgn", "    } else if ratio > 9.0 {\n        sgn"),
    ("창 이동 기준 봉 하나 어긋남", S, "let base = cl(bars[bars.len() - WINDOW - 1]);", "let base = cl(bars[bars.len() - WINDOW]);"),
    ("범위 분모 mid → 종가", S, "let rng_bp = (hi - lo) / mid * 1e4;", "let rng_bp = (hi - lo) / last_c * 1e4;"),
    ("거부 봉 비율 0.5 → 0.6", S, "pub const REJECT_FRAC: f64 = 0.5;", "pub const REJECT_FRAC: f64 = 0.6;"),
    ("펀딩 중립 경계 >= → >", S, "(fr - FUNDING_NEUTRAL).abs() >= FUNDING_NEUTRAL", "(fr - FUNDING_NEUTRAL).abs() > FUNDING_NEUTRAL"),
    ("베이시스 주도 경계 >= → >", S, "bd.abs() >= bthr", "bd.abs() > bthr"),
    ("BTC 동행 문턱 0.35 → 0.3", S, "if bm > 0.35 * brg { 1 }", "if bm > 0.3 * brg { 1 }"),
    ("봉 13개 → 12개", S, "want.len().saturating_sub(2 * WINDOW + 1)", "want.len().saturating_sub(2 * WINDOW)"),
    ("형성 중 봉도 넣음", S, ".filter(|(b, _)| *b < bar_start)", ".filter(|(b, _)| *b <= bar_start)"),
    ("종가 대체: 링 무시하고 VWAP", S, "filter(|(_, v)| v[6] != 0.0).next_back()", "filter(|(_, v)| v[6] != 0.0 && false).next_back()"),
    ("종가 대체: 링 첫 초", S, "filter(|(_, v)| v[6] != 0.0).next_back()", "filter(|(_, v)| v[6] != 0.0).next()"),
    ("종가 대체: 가격 0 인 초도 씀", S, "filter(|(_, v)| v[6] != 0.0).next_back()", "filter(|(_, v)| v[6] == v[6]).next_back()"),
    ("고가 대체값 없음", S, ".or_else(|| prices().reduce(f64::max))", ""),
    ("델타 부호", S, "delta: buy - sell,", "delta: sell - buy,"),
    ("청산 롱·숏 뒤바뀜", S, "liq_long: l[0],\n            liq_short: l[1],", "liq_long: l[1],\n            liq_short: l[0],"),
    ("Δ베이시스 지평 이전 초 < 로", S, "ring.range(..=last - horizon_s)", "ring.range(..last - horizon_s)"),
    ("임계 표본 간격 30 → 1", S, "ring.keys().step_by(30)", "ring.keys().step_by(1)"),
    ("임계 표본 최소 60 → 59", S, "if ds.len() >= 60 {", "if ds.len() >= 59 {"),
    ("임계 분위 색인 +1", S, "(ds.len() as f64 * pct) as usize)", "(ds.len() as f64 * pct) as usize + 1)"),
    ("마크 링 정리 간격 10 → 1", S, "pub const MARK_TRIM_S: f64 = 10.0;", "pub const MARK_TRIM_S: f64 = 1.0;"),
    ("마크 링 경계 >= → >", S, "s as f64 >= now_s - MARK_RING_S", "s as f64 > now_s - MARK_RING_S"),
    ("cvd 반올림 동률 → 0 에서 먼 쪽", S, "x.round_ties_even() as i64", "x.round() as i64"),
    ("소수 반올림 없음(move_bp)", S, "move_bp: rn(move_bp, 1),", "move_bp: move_bp,"),
    ("BTC 기준 봉 하나 어긋남", S, "bc.get(&want[want.len() - WINDOW - 1].0)?", "bc.get(&want[want.len() - WINDOW].0)?"),
]

if __name__ == "__main__":
    mf.main()
