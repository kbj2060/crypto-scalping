#!/usr/bin/env python3
"""장기보유 트레이드의 펀딩 캐리 회계 감사 (2026-08-24, diagnostic 전용).

배경: audit_current_rank1_baseline_safety_2026.py가 명시하듯 원장/백테스트 손익은
funding_cost_not_applied_in_accounting 상태다. 승리 트레이드가 전부 long-hold
(중앙값 336봉, 최대 4970봉)라 펀딩 누적이 창별 승패를 흔들 수 있는지 정량화한다.

- 원장: tmp/causal_regen_20260516/eth_odyssey4_single_component_win_condition_breakdown_20260818/
  enriched_trade_ledger.csv (6창, diagnostic 전용 원장 — 승격 근거 아님)
- 펀딩: data/TOTAL_ETHUSDT_fundingRate_2025_2026.csv (진짜 ETHUSDT, 8h, UTC,
  data.binance.vision monthly 아카이브 2025-01..2026-07 재구성본)
  ⚠️ 기존 data/TOTAL_ETHFIUSDT_fundingRate.csv는 ETHFI(ether.fi) 심볼로 확인됨 — 사용 금지.
- 회계 규약: 이벤트별 funding_pnl = -side * rate * notional (rate>0이면 롱이 지불),
  entry_timestamp < calc_time <= exit_timestamp 인 이벤트만 귀속.
  trade_return과 동일 단위(계좌자본 대비 분수, price_move*notional 규약)라 직접 합산 가능.
- 타임스탬프: 원장=캐노니컬 리플레이 산출(UTC-naive), 펀딩=UTC-naive — 동일 기준 가정.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_single_component_win_condition_breakdown_20260818/enriched_trade_ledger.csv"
FUNDING = ROOT / "data/TOTAL_ETHUSDT_fundingRate_2025_2026.csv"
OUT_DIR = ROOT / "tmp/funding_carry_audit_20260824"


def main() -> None:
    ledger = pd.read_csv(LEDGER, parse_dates=["entry_timestamp", "exit_timestamp"])
    funding = pd.read_csv(FUNDING, parse_dates=["calc_time"])
    funding = funding.sort_values("calc_time").reset_index(drop=True)

    f_ts = funding["calc_time"].to_numpy()
    f_rate = funding["last_funding_rate"].to_numpy()

    lo = np.searchsorted(f_ts, ledger["entry_timestamp"].to_numpy(), side="right")
    hi = np.searchsorted(f_ts, ledger["exit_timestamp"].to_numpy(), side="right")
    cum = np.concatenate([[0.0], np.cumsum(f_rate)])
    rate_sum = cum[hi] - cum[lo]
    n_events = hi - lo

    ledger["funding_events"] = n_events
    ledger["funding_rate_sum"] = rate_sum
    ledger["funding_pnl"] = -ledger["side"] * rate_sum * ledger["notional"]
    ledger["adj_return"] = ledger["trade_return"] + ledger["funding_pnl"]

    # 커버리지 확인: 보유기간 대비 기대 이벤트수(8h)와 실제 이벤트수 괴리
    hold_hours = ledger["hold_bars"] * 5 / 60.0
    expected = np.floor(hold_hours / 8.0)
    coverage_gap = int((ledger["funding_events"] < expected - 1).sum())

    per_window = (
        ledger.groupby("window")
        .agg(
            n=("trade_return", "size"),
            gross=("trade_return", "sum"),
            funding=("funding_pnl", "sum"),
            adjusted=("adj_return", "sum"),
            funding_per_trade_bp=("funding_pnl", lambda s: s.mean() * 1e4),
            max_abs_funding_bp=("funding_pnl", lambda s: s.abs().max() * 1e4),
        )
        .round(6)
    )
    per_window["sign_flip"] = np.sign(per_window["gross"]) != np.sign(per_window["adjusted"])

    per_wc = (
        ledger.groupby(["window", "component"])
        .agg(n=("trade_return", "size"), gross=("trade_return", "sum"), funding=("funding_pnl", "sum"))
        .round(6)
    )
    per_side = (
        ledger.groupby("side")
        .agg(n=("trade_return", "size"), gross=("trade_return", "sum"), funding=("funding_pnl", "sum"),
             funding_per_trade_bp=("funding_pnl", lambda s: s.mean() * 1e4))
        .round(6)
    )
    ledger["hold_tercile"] = pd.qcut(ledger["hold_bars"], 3, labels=["short", "mid", "long"])
    per_hold = (
        ledger.groupby("hold_tercile", observed=True)
        .agg(n=("trade_return", "size"), gross=("trade_return", "sum"), funding=("funding_pnl", "sum"),
             funding_per_trade_bp=("funding_pnl", lambda s: s.mean() * 1e4))
        .round(6)
    )

    print("=== 창별 (계좌자본 분수 합계) ===")
    print(per_window.to_string())
    print("\n=== 창 x 컴포넌트 ===")
    print(per_wc.to_string())
    print("\n=== 사이드별 (1=LONG, -1=SHORT) ===")
    print(per_side.to_string())
    print("\n=== 보유기간 3분위별 ===")
    print(per_hold.to_string())
    print(f"\n총 트레이드 {len(ledger)}, 펀딩 이벤트 총 {int(ledger['funding_events'].sum())}건")
    print(f"펀딩 총합 {ledger['funding_pnl'].sum():+.6f} (계좌분수) | 트레이드당 평균 {ledger['funding_pnl'].mean()*1e4:+.2f}bp")
    print(f"커버리지 경고(기대 대비 이벤트 부족 트레이드): {coverage_gap}건")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(OUT_DIR / "ledger_with_funding.csv", index=False)
    report = {
        "diagnostic_only": True,
        "funding_source": str(FUNDING.name),
        "ethfi_file_is_wrong_symbol": True,
        "convention": "funding_pnl = -side * sum(rate) * notional, entry<t<=exit, UTC",
        "total_trades": int(len(ledger)),
        "total_funding_pnl": float(ledger["funding_pnl"].sum()),
        "per_window": json.loads(per_window.reset_index().to_json(orient="records")),
        "per_side": json.loads(per_side.reset_index().to_json(orient="records")),
        "coverage_gap_trades": coverage_gap,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT_DIR}/ledger_with_funding.csv, report.json")


if __name__ == "__main__":
    main()
