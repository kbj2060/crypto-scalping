"""ETH 무기한선물 비방향 마켓메이킹(스프레드+펀딩 수확) — 수수료 산술 cheap-gate.

배경: docs/crypto_trading_strategies_literature_survey_20260823.md 종합판단 §2
(Le 2026 arXiv:2605.06405 펀딩인지 MM, "The Market Maker's Dilemma" arXiv:2502.18625).
방향예측이 필요없는 유일한 미시도 축이지만, 착수 전 수수료 구조가 산술적으로
성립하는지부터 게이트한다. 모든 입력은 이 저장소의 기존 실측치 + 공개 수수료표이며
새 데이터 수집/모델링 없음(순수 산술).

입력 출처(전부 기존 실측):
- 스프레드 0.053bp(=1틱): WS-E 파일럿 raw L2 53h(2026-07-19~21) 실측, 극단변동일
  aggTrades 재구성에서도 p90 0.053bp — docs/experiments/eth_maker_fill_simulation 계열.
- 단기 예측 gross 상한 1.3bp: 정보시간 샘플링 A/B 실측
  (eth_infotime_sampling_ab_closed_20260817). MM 호가선택이 이 예측력을 전부
  수확한다는 최대관용 가정.
- 펀딩(2026): RDE 문서 실측 분기합 VAL +0.17% / OOS +0.47% → 시간당 ≤0.04bp.
  1시간 보유 가정으로 0.04bp/RT(관용).
- 수수료: Binance USDT-M VIP0 maker 0.02%(2bp)/taker 0.05%(5bp) — 저장소 taker
  실측 5.03bp/leg과 정합. BNB 10% 할인 시 maker 1.8bp. VIP9(월 $5B+) maker 0%.
- 역선택(AS): 이 저장소 미측정(시뮬/섀도우는 체결시점까지의 shortfall만 측정,
  체결 후 드리프트는 미측정). 감도축으로만 취급: 0(최대관용)/0.5/1.1/2.0bp/leg.
  1.1bp는 peg 추격 드리프트(3.1bp 총비용 − 2bp 수수료)를 참조점으로 쓴 값.
"""
import json
from pathlib import Path

OUT_DIR = Path("tmp/eth_mm_fee_arithmetic_cheap_gate_20260823")

# 수익원 (bp, 왕복=RT 기준, 양 leg 모두 메이커 체결 가정)
SPREAD_CAPTURE_BP = 0.053     # bid 매수→ask 매도 왕복 = 전체 스프레드 1틱
ALPHA_CEILING_BP = 1.3        # 단기 예측 gross 상한(최대관용: 전부 수확 가정)
FUNDING_PER_RT_BP = 0.04      # 2026 OOS 펀딩레이트 기준 1시간 보유(관용)

REVENUE_BEST_BP = SPREAD_CAPTURE_BP + ALPHA_CEILING_BP + FUNDING_PER_RT_BP

# 비용 시나리오: (이름, maker 수수료 bp/leg)
FEE_SCENARIOS = [
    ("VIP0", 2.0),
    ("VIP0+BNB10%", 1.8),
    ("breakeven_fee", None),   # 아래서 역산
    ("zero_fee_venue", 0.0),
]
AS_GRID_BP_PER_LEG = [0.0, 0.5, 1.1, 2.0]

breakeven_fee_per_leg = REVENUE_BEST_BP / 2.0  # AS=0 최대관용에서 net=0이 되는 수수료

rows = []
for name, fee in FEE_SCENARIOS:
    fee_bp = breakeven_fee_per_leg if fee is None else fee
    for as_bp in AS_GRID_BP_PER_LEG:
        net = REVENUE_BEST_BP - 2.0 * fee_bp - 2.0 * as_bp
        rows.append({
            "scenario": name,
            "maker_fee_bp_per_leg": round(fee_bp, 4),
            "adverse_selection_bp_per_leg": as_bp,
            "net_bp_per_round_trip": round(net, 3),
        })

# VIP0에서 손익분기에 필요한 RT당 총수익(수수료만, AS=0)
required_revenue_vip0 = 2.0 * 2.0
alpha_multiple_needed = (required_revenue_vip0 - SPREAD_CAPTURE_BP - FUNDING_PER_RT_BP) / ALPHA_CEILING_BP

report = {
    "gate_id": "eth_mm_spread_capture_fee_arithmetic_cheap_gate_20260823",
    "inputs": {
        "spread_capture_bp_per_rt": SPREAD_CAPTURE_BP,
        "short_horizon_alpha_ceiling_bp_per_rt": ALPHA_CEILING_BP,
        "funding_bp_per_rt_1h_hold_2026": FUNDING_PER_RT_BP,
        "revenue_best_case_bp_per_rt": round(REVENUE_BEST_BP, 4),
        "provenance": {
            "spread": "WS-E pilot raw L2 53h 2026-07-19~21 (1 tick fixed; extreme-day p90 also 0.053bp)",
            "alpha_ceiling": "info-time sampling A/B gross (eth_infotime_sampling_ab_closed_20260817)",
            "funding": "RDE doc measured quarterly sums: VAL2026Q2 +0.17%, OOS +0.47%",
            "fees": "Binance USDT-M VIP0 0.02%/0.05% (repo-measured taker 5.03bp/leg consistent)",
            "adverse_selection": "UNMEASURED in this repo (sim/shadow stop at fill; post-fill drift not recorded) — sensitivity axis only",
        },
    },
    "results": {
        "table": rows,
        "breakeven_maker_fee_bp_per_leg_at_AS0": round(breakeven_fee_per_leg, 4),
        "breakeven_maker_fee_pct": round(breakeven_fee_per_leg / 100.0, 6),
        "required_revenue_bp_per_rt_at_VIP0": required_revenue_vip0,
        "alpha_multiple_of_measured_ceiling_needed_at_VIP0": round(alpha_multiple_needed, 2),
    },
    "verdict": "REJECTED_FEE_STRUCTURE",
    "verdict_detail": (
        "At Binance VIP0 (repo's actual fee reality, confirmed by measured taker 5.03bp/leg), "
        "the most generous case (zero adverse selection, full capture of the measured short-horizon "
        "alpha ceiling) nets about -2.6bp per round trip; fees alone (4bp/RT) exceed the sum of all "
        "measured revenue sources (~1.4bp/RT) by ~3x. Breakeven requires maker fee <= ~0.007%/leg "
        "(deep-VIP / MM-program / alternative-venue territory) AND resting-quote adverse selection "
        "<= ~0.7bp/leg (unmeasured here; live literature evidence says it is materially positive on "
        "Binance perps). Venue/fee-tier change is necessary but not sufficient."
    ),
    "no_new_data_used": True,
    "pure_arithmetic": True,
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_DIR / "report.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"revenue best case (bp/RT): {REVENUE_BEST_BP:.3f}")
print(f"breakeven maker fee (bp/leg, AS=0): {breakeven_fee_per_leg:.3f}  (= {breakeven_fee_per_leg/100:.4f}%)")
print(f"required alpha multiple at VIP0: {alpha_multiple_needed:.2f}x measured ceiling")
print()
hdr = f"{'scenario':<16}{'fee bp/leg':>11}{'AS bp/leg':>10}{'net bp/RT':>11}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['scenario']:<16}{r['maker_fee_bp_per_leg']:>11}{r['adverse_selection_bp_per_leg']:>10}{r['net_bp_per_round_trip']:>11}")
print(f"\nverdict: {report['verdict']}")
print(f"report: {OUT_DIR / 'report.json'}")
