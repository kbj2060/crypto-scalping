#!/usr/bin/env python3
"""peg-maker 집행 섀도우 실측 vs 시뮬 예측 대조 (2026-08-22).

사용: python scripts/analyze_eth_maker_fill_shadow_vs_sim_20260822.py [--db PATH]
서버에서 축적 중인 data/live/maker_fill_shadow.duckdb를 읽어 정책별 실효 비용 분포를
시뮬 참조치(v1 raw L2 / v2 aggTrades, docs/experiments/eth_maker_fill_simulation_l2_20260822.md)와
나란히 출력한다. 하트비트로 스트림 생존도 함께 점검(조용한 사망 감지).

시뮬 참조치(peg T120): 저변동 3.09~3.26bp / 고변동 극단일 3.61~3.75bp /
역방향 조건부 3.8~4.0bp. static T120: 3.35~3.40bp(체결분 1.9~2.0bp).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SIM_REF = {
    "peg": {"calm": (3.09, 3.26), "extreme_days": (3.61, 3.75), "adverse_cond": (3.8, 4.0)},
    "static": {"calm": (3.35, 3.40)},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data/live/maker_fill_shadow.duckdb"))
    args = ap.parse_args()

    try:
        con = duckdb.connect(args.db, read_only=True)
    except Exception:
        # 워커(단일 writer)가 파일 잠금을 쥔 동안에는 사본으로 읽는다
        import shutil
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / "maker_fill_shadow_copy.duckdb"
        shutil.copy2(args.db, tmp)
        print(f"(writer lock 감지 — 사본으로 분석: {tmp})")
        con = duckdb.connect(str(tmp), read_only=True)
    hb = con.execute("""
        select max(recorded_at_utc), count(*),
               max(book_msgs), max(trade_msgs), max(legs_done)
        from maker_fill_shadow_heartbeat""").fetchone()
    print(f"heartbeat: last={hb[0]} rows={hb[1]} book_msgs={hb[2]} trade_msgs={hb[3]} legs_done={hb[4]}")
    if hb[2] in (None, 0) or hb[3] in (None, 0):
        print("⚠️ 스트림 카운터가 0 — tail_risk의 '조용한 사망' 패턴 점검 필요")

    df = con.execute("""
        select policy, timeout_s,
               count(*) n,
               avg(case when filled then 1.0 else 0.0 end) fill_rate,
               avg(cost_bp) cost_mean,
               quantile_cont(cost_bp, 0.5) cost_med,
               quantile_cont(cost_bp, 0.9) cost_p90,
               avg(case when filled then cost_bp end) filled_mean,
               avg(case when not filled and fill_mode='taker_fallback' then cost_bp end) fallback_mean,
               sum(case when fill_mode='aborted_stale' then 1 else 0 end) aborted,
               avg(repegs) repegs_mean,
               min(recorded_at_utc) first_leg, max(recorded_at_utc) last_leg
        from maker_fill_shadow_legs
        where cost_bp is not null or fill_mode='aborted_stale'
        group by 1,2 order by 1,2""").df()
    print(df.to_string(index=False))

    print("\n시뮬 참조치(bp/leg):")
    for pol, refs in SIM_REF.items():
        print(f"  {pol}: " + ", ".join(f"{k}={v[0]}~{v[1]}" for k, v in refs.items()))
    print("\n판정 가이드: 실측 peg 평균이 시뮬 밴드(3.1~4.0bp) 안이면 시뮬 검증 성공 —")
    print("차세대 OOS 판정의 권장 비용 가정(기본 3.5/스트레스 4.0/tail 7.5bp) 유지.")
    print("실측이 밴드를 크게 벗어나면 시뮬 규칙(큐/체결판정) 재보정 후 권장 가정 갱신 필요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
