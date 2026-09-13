"""**지금 상황의 트레이드 플랜** — 집행·크기·보유시간·분할을 한 번에 (2026-09-13).

사용자: *"현재 상황에 내가 진입 및 청산을 하려는데 어떻게 집행·크기·보유시간을 가져가는 게
좋을지 예측 및 분석하는 모델 … 분할 매매 및 분할 청산을 전략적으로"*.

새로 계산하는 건 둘뿐이고 나머지는 기존 함수를 조립한다.
  ① **보유시간 프런티어** = 지평별 로그성장 g(H,a) = L(H)·((2a−1)·b_H − 비용) − ½(L(H)·σ_H)².
     L(H) 는 생존 모델(보유시간 조건부 MAE 분위, 매 5분 갱신)의 허용 배수. b_H·σ_H 는 **과거
     데이터**(869일 5분봉, research_hold_horizon_frontier_history_20260913)에서 잰 스케일 없는
     계수 × 지금 atr_pct 다 -- 원장을 쓰지 않는다(사용자: *"원장은 정답이 아니야"*).
     방향 정확도 a 는 파라미터다: 표는 a 격자마다 최적 H 와 손익분기 정확도 a*(H)=½+비용/(2b_H) 를 준다.
     ⚠️(2a−1)·E|r| 은 정확도가 움직임 크기와 독립일 때만 맞다 -- H 사이의 **순위**용이다.
  ② **보유 예산 사다리**(분할 청산) = 지금 순자산·명목에서 «R 분 더 들려면 얼마를 닫아야
     하나»를 R 마다 낸다. 같은 부등식 N ≤ E·L(R) 을 R 에 대해 푼 것이라 단일 시점
     `exit_fraction_required` 의 일반화다. 역행으로 E 가 줄면 사다리가 자동으로 조여진다.

분할 **진입**은 시간 분할이 아니라 **조건 분할**이다: 09-06 감사에서 물타기(역행 중 추가)는
크기 매칭 후 24/24 전패, 살아남은 건 «부분 투입»뿐이었다. 그래서 첫 칸 뒤의 추가는
«평가손익 ≥ 0 이고 상한 여유가 남았을 때»만 허용한다(피라미딩 -- 근거는 약하다, OOS CI 0 포함).

집행은 이미 배포된 정책을 그대로 쓴다(peg GTX 진입 / 리페그 청산 / 변동성 연동 마감 /
극단 변동성 시장가). 여기서는 기대 체결시간(중앙 ≈ 52/vol^1.472 초, 섀도우 7,982legs 적합)만 붙인다.
"""
from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.live_eth_risk_sizing_policy_20260913 import HARD_CAP_X  # noqa: E402
from scripts.live_manual_peg_entry_20260912 import (  # noqa: E402
    EXIT_TAKER_VOL_BPM, FALLBACK_SEC, STOP_LOSS_PCT, exit_deadline_sec)

HOLD_CHOICES = (60, 120, 240, 480, 1440)
ROUND_TRIP_COST_BP = 5.88            # static 진입 + peg 청산 실측(2026-09-13)
# 🔴손절로 끝나는 거래는 청산 다리가 **시장가**라 더 비싸다: 테이커 5.0 + 슬리피지 중앙 14.0
# = 19.0bp 이고, 정상 peg 청산 2.93bp 를 대신하므로 **추가분**은 그 차액이다(2026-09-13).
# 꼬리는 훨씬 두껍다(90% 65.6 · 99% 227bp) -- 여기서 쓰는 건 기대값이라 중앙값을 쓴다.
STOP_EXTRA_COST_BP = 16.08
# 펀딩. 8시간마다 정산되므로 **보유시간에 비례**하고 롱/숏 부호가 반대다. 라이브 값을 못
# 읽으면 최근 30일 중앙값(+0.52bp/8h, 롱이 낸다)을 쓴다 -- 0 으로 두면 긴 지평이 공짜가 된다.
FUNDING_BP_8H_FALLBACK = 0.52
# 보유시간별 **손절 발동률**(869일 1분봉 봉내 기준, research_stop_hit_rate_by_horizon_20260913).
# 🔴이 표가 생기기 전에는 4시간 실측 0.043 을 전 지평에 썼다 -- 1440분에서 7.4배 과소였다.
STOP_HIT_RATE_BY_HOLD = {60: (0.0085, 0.0064), 120: (0.0230, 0.0178),
                         240: (0.0574, 0.0457), 480: (0.1229, 0.1066),
                         1440: (0.3195, 0.3007)}
# 과거 869일 5분봉의 스케일 없는 계수 (k_b = E|r_H|/atr_pct, k_s = SD(r_H)/atr_pct).
# 변동성 5분위·TRAIN→TEST 모두 ±15% 안에서 안정(연구 스크립트 assert). 원장과 무관하다.
K_HORIZON = {60: (3.518, 5.488), 120: (5.055, 7.838), 240: (7.367, 11.289),
             480: (10.862, 16.401), 1440: (20.691, 30.599)}
ACC_GRID = (0.52, 0.55, 0.58, 0.62, 0.66)
# ⭐**처방(prescribe)이 쓰는 기준 정확도.** 이건 시장 예측이 아니라 **사용자 진입의 실력**이다 --
# 모델이 낼 수 있는 값이 아니다(이 저장소의 방향 모델은 전부 0.50). 이 값이 하는 일은 하나:
# «국면 → 보유시간» 분기를 켜는 것. a>=0.60 이면 그 분기가 안정하고(0.60~0.75 에서 잔잔=길게 ·
# 험함=짧게로 일관), a<0.58 이면 답이 «아주 길게·아주 작게»로 무너진다.
# 0.60 은 사용자 실계좌 68왕복(승률 72.1% · 단위당 t=3.08)을 **크게 할인한** 값이다.
PRESCRIBE_ACC = 0.60
# ponytail: 단일 권고에 쓰는 기준 정확도는 고정 55%. 원장에서 뽑지 않는다. 화면 슬라이더로 바꾸려면 쿼리 인자 하나.
ACC_DEFAULT = 0.55
FILL_K, FILL_EXP = 52.0, 1.472       # 중앙 체결시간(초) ≈ 52 / vol^1.472
MAKER_BP, TAKER_BP = 2.0, 5.0
# ponytail: 재조정 무거래 밴드는 고정 25%. 비용(왕복 5.88bp) 대 파산감소로 최적화하면 더 낫다.
REBALANCE_BAND = 0.25


EULER_GAMMA = 0.5772156649
# 🔴**경로 뭉침 보정**(2026-09-14, 사용자 지적). 위 식은 증분이 독립이라고 본다. 실제 연대순
# 수열로 재면 낙폭이 더 깊다 -- 로그 공간 중앙 비율 1.10(80칸, research 3차).
# ⚠️정직하게: 이 보정은 **정확도를 안 올린다**(평균절대오차 0.066 → 0.065). 없애는 건 **편향**
# 뿐이다(−0.025 → +0.003). 위험 표시라 «낮게 부르는» 편향은 그대로 두면 안 되므로 적용한다.
# ⚠️효과는 지평마다 다르다(60분 1.10 · 240분 1.07 · **480분 0.95** · 1440분 1.28). 그런데
# 실제 연대순 창은 869일에서 나와 **독립 연수가 2.4개**뿐이라 지평별 표를 만들면 과적합이다.
# ⇒ 상수 하나만 쓴다. 블록 부트로는 재현이 안 된다(1일·1주·1달·3달 전부 IID 와 같았다) --
#   즉 이건 단기 변동성 뭉침이 아니라 **그 해에 실제로 일어난 순서**의 효과다.
MDD_CLUSTER_UPLIFT = 1.10


def expected_max_drawdown(L: float, move_bp: float, sd_bp: float, acc: float,
                          hold_min: int, cost_bp: float | None = None) -> float:
    """1년 경로의 **중앙 최대낙폭**. 파산이 아니라 이게 실제로 배수를 묶는 값이다.

    🔴왜 파산이 아니라 MDD 인가(2026-09-14 실측, research_ruin_term_in_objective_20260914):
      · 손절 3% 에서 단일거래 파산은 L >= 1/0.03 = 33.3배에서만 가능한데 하드캡이 25배다.
        **가용 구간 전체에서 항상 0인 죽은 항**이라 목적함수에 넣을 게 없다.
      · 경로 파산도 손절이 있으면 L=25 까지 실측 0.000 이다.
      · 그런데 L=16 에서 **파산율 0% 인데 1년 중앙 계좌가 0.013배**다(중앙 MDD 1.000).
        파산 항을 넣었다면 «안전»이라고 답했을 자리다. 죽이는 건 파산이 아니라 낙폭이다.

    식은 표류 있는 랜덤워크의 기대 최대낙폭이다
    (Magdon-Ismail·Atiya·Pratap·Abu-Mostafa 2004, *On the Maximum Drawdown of a Brownian Motion*):
        로그낙폭 ≈ (v / 2g) · (ln(2·g²·n / v) + γ)
    g=건당 로그성장 · v=건당 로그수익 분산 ≈ (L·σ)² · n=연간 거래수. **셋 다 이미 있는 값**이라
    새 표가 필요 없다. 적합은 스케일 1개(k=1.038≈1)뿐이고 **표본외 검증**(H=60·1440)에서
    상관 0.964 · 평균절대오차 0.048 이었다.

    🔴**분기가 둘이다.** 위 식은 «표류가 잡음을 이기는» 구간의 점근식이라, 실력이 낮거나
    배수가 커서 2g²n/v 가 작아지면 못 쓴다. 그 구간은 표류 없는 랜덤워크로 보는 게 맞다:
        로그낙폭 ≈ σ_g · √(π·n/8)
    처음엔 저표류 칸을 조용히 버리고 검증했다가 «전 구간 오차 0.048» 이라는 낙관적인 수를
    얻었다. 버린 칸을 되살려 **140칸 전부**로 재면 상관 0.896 · 평균절대오차 0.068 이다.
    (고표류 65칸 0.030 · 저표류 75칸 0.101 -- 저표류 쪽이 거칠다는 걸 숨기지 않는다.)

    ⚠️화면은 **구간(밴드)으로** 읽어야지 소수점을 믿으면 안 된다.
    g <= 0 이면 표류가 없으므로 저표류 분기로 간다.
    """
    if cost_bp is None:
        cost_bp = expected_cost_bp(hold_min)
    g = growth_per_trade(L, move_bp, sd_bp, acc, cost_bp)
    v = (L * sd_bp / 1e4) ** 2
    n = min(735.0, 525_600.0 / max(1, int(hold_min)))
    if v <= 0 or L <= 0:
        return 1.0
    # 전환점은 ln(arg) = 1, 즉 arg = e. 그 아래는 로그항이 음수가 되어 식이 뒤집힌다.
    arg = 2.0 * g * g * n / v if g > 0 else 0.0
    log_dd = ((v / (2.0 * g)) * (math.log(arg) + EULER_GAMMA) if arg > math.e
              else math.sqrt(v) * math.sqrt(math.pi * n / 8.0))
    return float(min(1.0, 1.0 - math.exp(-MDD_CLUSTER_UPLIFT * log_dd)))


def stop_hit_rate(hold_min: int, side: str = "LONG") -> float:
    """이 보유시간에 손절이 걸릴 확률. 표에 없는 지평이면 가장 가까운 칸을 쓴다."""
    key = min(STOP_HIT_RATE_BY_HOLD, key=lambda h: abs(h - int(hold_min)))
    return STOP_HIT_RATE_BY_HOLD[key][0 if str(side).upper() == "LONG" else 1]


def funding_cost_bp(hold_min: int, side: str, rate_bp_8h: float | None = None) -> float:
    """이 보유시간 동안 **내는** 펀딩(bp). 양수 = 비용. 롱은 rate 그대로, 숏은 부호 반대다."""
    r = FUNDING_BP_8H_FALLBACK if rate_bp_8h is None else float(rate_bp_8h)
    return (r if str(side).upper() == "LONG" else -r) * (hold_min / 480.0)


def expected_cost_bp(hold_min: int, side: str = "LONG",
                     funding_bp_8h: float | None = None) -> float:
    """이 지평 한 건의 **기대 왕복비용**. 세 조각이고 전부 보유시간에 따라 달라진다.

    ① 정상 왕복(peg 진입 + peg 청산) ② 손절로 끝날 확률 × 시장가 추가비용 ③ 펀딩.
    ⚠️손절은 **왼쪽 꼬리를 자르기도** 하는데 그 효과는 여기 없다 -- 아래 growth 의 분산항이
    정규 근사라서다. **그 한계를 2026-09-14 에 실측했다**
    (research_stop_truncation_vs_gaussian_objective_20260914, 869일 1분봉 실제 경로):
      · 🔴**최적 지평은 안 바뀐다** -- 40칸(H×L×a) 전부, 잔잔·험함 두 국면 전부 argmax 일치.
        근사를 «순위»에 쓰는 건 실측으로 정당화됐다.
      · 오차 방향은 **한쪽**이다: 40칸 중 34칸에서 공식이 **낮게** 본다. «번다»는데 실제가
        손실인 칸은 **0**. 즉 틀리는 방향이 안전하다.
      · 크기는 작지 않다: 잔잔한 국면 1440분에서 시간당 0.52 vs 실제 3.76(×1e4) -- **7배**.
        절반은 적률(꼬리 절단), 절반은 분포 모양 + 조기 종료다.
      · z = 손절폭/σ_H 하나로는 **안 무너진다**(같은 z 에서 잔잔 +1.94 vs 험함 −0.16).
        보정 함수를 하나 적합하면 10칸에 과적합이라 **안 고치고 한계로 문서화한다**.
    ⇒ 이 값은 지평 **순위**용이고, 수준은 **보수적 하한**으로 읽어야 한다."""
    p = stop_hit_rate(hold_min, side)
    return (ROUND_TRIP_COST_BP + p * STOP_EXTRA_COST_BP
            + funding_cost_bp(hold_min, side, funding_bp_8h))


def growth_per_trade(L: float, move_bp: float, sd_bp: float, acc: float,
                     cost_bp: float = ROUND_TRIP_COST_BP) -> float:
    """정확도 acc 인 사람이 이 지평을 L 배로 들 때의 건당 로그성장 근사."""
    return L * ((2 * acc - 1) * move_bp - cost_bp) / 1e4 - 0.5 * (L * sd_bp / 1e4) ** 2


def growth_per_hour(L: float, move_bp: float, sd_bp: float, acc: float, hold_min: int,
                    cost_bp: float = ROUND_TRIP_COST_BP) -> float:
    """**시간당** 로그성장. 건당으로 고르면 무조건 긴 쪽이 이긴다 -- 짧은 지평은 여러 번
    굴릴 수 있다는 사실이 건당 지표에 안 들어가기 때문이다(2026-09-13 실측: 건당 기준은
    전 국면·전 정확도에서 1440분으로 퇴화했다)."""
    return growth_per_trade(L, move_bp, sd_bp, acc, cost_bp) / (hold_min / 60.0)


def safe_mae(risk_table: dict, hold: int, side: str) -> float | None:
    cell = ((risk_table or {}).get(str(hold)) or {}).get(side.upper()) or {}
    m = cell.get("safe_mae_pct")
    return float(m) if m and m > 0 else None


def allowed_x(risk_table: dict, hold: int, side: str, cap_x: float) -> float | None:
    """이 보유시간의 허용 배수 = min(생존, 적용 상한). 모델 칸이 없으면 None."""
    m = safe_mae(risk_table, hold, side)
    return None if m is None else min(100.0 / m, cap_x, HARD_CAP_X)


def recommend_hold(risk_table: dict, side: str, cap_x: float, atr_pct: float | None,
                   acc: float = ACC_DEFAULT, funding_bp_8h: float | None = None) -> dict:
    """지평별 프런티어. 권고는 기준 정확도 acc 에서의 최대, 격자별 최적도 같이 준다."""
    if not atr_pct or atr_pct <= 0:
        return {"available": False, "reason": "atr_pct 없음"}
    rows, best_by_acc = [], {}
    for H in HOLD_CHOICES:
        L = allowed_x(risk_table, H, side, cap_x)
        if L is None:
            continue
        kb, ks = K_HORIZON[H]
        b, sd = kb * atr_pct * 1e4, ks * atr_pct * 1e4
        # 🔴비용은 지평마다 다르다(손절률·펀딩). 5.88 고정을 쓰면 긴 지평이 실제보다 싸 보인다.
        cost = expected_cost_bp(H, side, funding_bp_8h)
        row = {"hold_min": H, "leverage": round(L, 2), "move_bp": round(b, 1), "sd_bp": round(sd, 1),
               "cost_bp": round(cost, 2), "stop_hit_rate": stop_hit_rate(H, side),
               "funding_bp": round(funding_cost_bp(H, side, funding_bp_8h), 2),
               "breakeven_acc": round(0.5 + cost / (2 * b), 3),
               "growth": round(growth_per_trade(L, b, sd, acc, cost), 5),
               "growth_per_hour": round(growth_per_hour(L, b, sd, acc, H, cost), 6),
               "growth_by_acc": {str(x): round(growth_per_hour(L, b, sd, x, H, cost), 6)
                                 for x in ACC_GRID}}
        rows.append(row)
        for x in ACC_GRID:
            g = row["growth_by_acc"][str(x)]
            if x not in best_by_acc or g > best_by_acc[x][1]:
                best_by_acc[x] = (H, g)
    if not rows:
        return {"available": False, "reason": "위험모델 없음"}
    best = max(rows, key=lambda r: r["growth_per_hour"])
    # 🔴전부 음수면 «권고»가 아니다 -- «덜 나쁜 것»이다(2026-09-13). 잔잔한 국면에서는 실제로
    # 전 지평이 음수가 된다(atr 3.87bp/봉 · 55% 가정에서 −0.0038 ~ −0.0004). 그걸 «권고 1440분»
    # 으로만 적으면 «그 시간 들면 번다»로 읽힌다. 부호를 문장으로 말한다.
    none_positive = best["growth_per_hour"] <= 0
    need = min(r["breakeven_acc"] for r in rows)
    # 🔴«못 넘습니다»를 사실로 단정하지 않는다. 이 추정은 손절의 꼬리 절단이 빠져 있어
    # 실측 40칸 중 34칸에서 **낮게** 나왔다(2026-09-14). 부호가 뒤집힌 칸도 2개 있었다.
    head = (f"🔴정확도 {int(100*acc)}% 로는 **어느 보유시간도 비용을 못 넘는 것으로 나옵니다**"
            f"(최소 필요 {int(round(100*need))}%) — 덜 나쁜 쪽이 {best['hold_min']}분"
            f" · 이 추정은 손절의 꼬리 절단이 빠져 **보수적**입니다"
            if none_positive else
            f"정확도 {int(100*acc)}% 가정 시 {best['hold_min']}분이 건당 로그성장 최대"
            f"({best['growth']:+.4f})")
    return {"available": True, "recommended_min": best["hold_min"], "acc": acc, "table": rows,
            "best_by_acc": {str(x): h for x, (h, _) in best_by_acc.items()},
            "none_positive": none_positive, "breakeven_min_acc": round(need, 3),
            # 화면이 «이 숫자를 어떻게 읽어야 하나»를 말할 수 있게. 실측 근거는 위 주석.
            "estimate_is_conservative": True,
            "bias_note": ("정규 근사라 손절의 꼬리 절단이 빠져 있습니다 — 실측 40칸 중 34칸에서"
                          " 실제보다 낮게 나왔고, 잔잔한 장 1440분에서는 7배까지 낮았습니다."
                          " 최적 보유시간 자체는 40칸 전부에서 바뀌지 않았습니다"),
            "reason": (head + " · 손익분기 정확도 "
                       + " ".join(f"{r['hold_min']}분 {int(round(100*r['breakeven_acc']))}%" for r in rows))}


def expected_fill_sec(vol_bpm: float | None) -> float | None:
    return None if not vol_bpm or vol_bpm <= 0 else FILL_K / vol_bpm ** FILL_EXP


def execution_plan(vol_bpm: float | None, hold_min: int = 240, side: str = "LONG",
                   funding_bp_8h: float | None = None) -> dict:
    fill = expected_fill_sec(vol_bpm)
    market = vol_bpm is not None and vol_bpm >= EXIT_TAKER_VOL_BPM
    fund = funding_cost_bp(hold_min, side, funding_bp_8h)
    return {
        "vol_bpm": round(vol_bpm, 2) if vol_bpm is not None else None,
        # 비용을 한 줄로 합치지 않는다 -- 어느 조각이 큰지 보여야 지평 선택이 설명된다.
        "cost": {"round_trip_bp": ROUND_TRIP_COST_BP,
                 "stop_hit_rate": stop_hit_rate(hold_min, side),
                 "stop_extra_bp": round(stop_hit_rate(hold_min, side) * STOP_EXTRA_COST_BP, 2),
                 "funding_bp": round(fund, 2),
                 "funding_rate_bp_8h": round(FUNDING_BP_8H_FALLBACK if funding_bp_8h is None
                                             else float(funding_bp_8h), 3),
                 "funding_is_live": funding_bp_8h is not None,
                 "total_bp": round(expected_cost_bp(hold_min, side, funding_bp_8h), 2)},
        "entry": {"mode": "peg_gtx", "expected_fill_sec": round(fill, 1) if fill else None,
                  "fallback_sec": FALLBACK_SEC, "maker_bp": MAKER_BP, "taker_bp": TAKER_BP,
                  "note": ("저변동은 체결이 느리고 미체결이 몰립니다 — 진입 미체결은 무해하니 기다립니다"
                           if fill and fill > 10 else "빠른 장일수록 메이커가 더 잘 체결됩니다")},
        "exit": {"mode": "market" if market else "peg_repeg",
                 "deadline_sec": None if market else round(exit_deadline_sec(vol_bpm), 1),
                 "expected_fill_sec": round(fill, 1) if fill else None},
    }


def hold_budget(equity: float, notional: float, risk_table: dict, side: str,
                cap_x: float) -> dict:
    """지금 크기로 «R 분 더 들려면 얼마를 닫아야 하나». 0 이면 그 시간은 예산 안이다."""
    ladder, budget = [], 0
    for H in HOLD_CHOICES:
        L = allowed_x(risk_table, H, side, cap_x)
        if L is None or equity <= 0:
            continue
        allowed = equity * L
        f = 0.0 if notional <= allowed else min(1.0, 1.0 - allowed / notional)
        if f == 0.0:
            budget = H
        ladder.append({"hold_min": H, "allowed_notional": round(allowed, 2),
                       "required_fraction": round(f, 4)})
    return {"budget_min": budget, "ladder": ladder,
            "rebalance_band": REBALANCE_BAND,
            "note": (f"지금 크기로는 {budget}분까지가 예산입니다" if budget else
                     "지금 크기는 어떤 보유시간 예산도 넘습니다 — 사다리대로 줄이세요")}


# ── 거래소 레버리지 설정 (2026-09-13) ────────────────────────────────────────
# 🔴교차 마진에서 **설정 레버리지는 위험을 안 바꾼다**. 청산거리는 순자산/총명목이고 설정값이
# 거기 안 들어간다(2026-09-12 실측: 설정 30배인데 청산까지 13.48%, 격리 30배면 ~3%여야 한다).
# 설정이 정하는 것은 둘뿐이다:
#   ① 잠기는 증거금 = 명목 / 설정레버리지
#   ② **열 수 있는 총 명목의 하드 상한** = 가용증거금 × 설정레버리지
# ②가 이 함수의 존재 이유다. 설정을 우리 정책 상한에 맞추면 대시보드 상한이 **거래소가
# 강제하는 상한**이 된다 -- 화면을 우회해도(거래소 앱에서 직접 주문해도) 살아남는 유일한 방어다.
# ETHUSDT 실측 브래킷(2026-09-13): 1구간 30만 USDT 까지 **150배 · 유지증거금률 0.40%**.
# 우리 크기(순자산 1,089 × 8배 = 8,716)는 전부 1구간이라 거래소는 아무것도 안 막는다.
EXCHANGE_MAX_LEVERAGE = 150
MAINT_MARGIN_RATE = 0.004
LEVERAGE_STEPS = (1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 50, 75, 100, 125, 150)


def leverage_setting(*, cap_notional: float, equity: float,
                     existing_notional: float = 0.0,
                     current_notional: float = 0.0) -> dict:
    """거래소에 설정할 레버리지. **위험이 아니라 «상한을 거래소에 새기는 값»** 이다.

    돌려주는 것:
      · `setting`        실제로 걸 값(거래소가 받는 눈금으로 올림)
      · `min_feasible`   이 아래로 내리면 정책 상한만큼도 못 연다(주문 거부)
      · `max_notional`   그 설정에서 거래소가 열어 주는 총 명목 = 순자산 × 설정
      · `enforces_cap`   거래소 상한이 우리 정책 상한과 같은가(같아야 방어가 선다)
    """
    if equity <= 0 or cap_notional <= 0:
        return {"available": False, "reason": "순자산 또는 상한 없음"}
    # 정책 상한을 그대로 열려면 최소 이만큼은 있어야 한다.
    # 🔴기존 포지션을 더하지 **않는다**. 정책 상한은 기존을 **포함한 총 명목**에 걸리므로
    # 필요한 증거금은 cap/L_set 하나뿐이고, 기존 다리는 이미 그 cap 안에 들어 있다.
    # (첫 판에 (cap+existing)/equity 로 짰다가 이중 계상을 잡았다 -- 그러면 설정이 두 배로
    #  뛰어 거래소 상한이 정책의 2.5배가 되고 방어가 무의미해진다.)
    min_feasible = cap_notional / equity
    # 🔴**이미 열린 포지션이 바닥을 만든다**(2026-09-13). 레버리지를 내리면 기존 포지션의
    # 초기증거금(명목/레버리지)이 올라가고, 순자산을 넘으면 거래소가 **-2028
    # MIN_LEVERAGE_RATIO «Leverage is smaller than permitted: insufficient margin balance»**
    # 로 거부한다. (격리였다면 아예 -4161 로 막힌다 -- 이 계좌는 cross 라 해당 없다.)
    # 포지션이 정책 상한을 넘긴 상태면 이 바닥이 정책값보다 높아진다. 그건 사실이므로
    # 숨기지 않고 그대로 올린다 -- 대신 왜 그런지 화면에 적는다.
    position_floor = max(0.0, current_notional) / equity
    target = choose_leverage(min_feasible=min_feasible, cap_x=cap_notional / equity)
    target = max(target, position_floor)
    setting = next((x for x in LEVERAGE_STEPS if x >= target), EXCHANGE_MAX_LEVERAGE)
    setting = min(setting, EXCHANGE_MAX_LEVERAGE)
    forced = position_floor > choose_leverage(min_feasible=min_feasible,
                                              cap_x=cap_notional / equity)
    max_notional = equity * setting
    return {
        "available": True,
        "setting": setting,
        "min_feasible": round(min_feasible, 2),
        "cap_x": round(cap_notional / equity, 2),
        "max_notional": round(max_notional, 2),
        "margin_locked": round(cap_notional / setting, 2),
        "margin_pct_of_equity": round(100.0 * cap_notional / setting / equity, 1),
        "enforces_cap": max_notional <= cap_notional * 1.25,
        "exchange_max": EXCHANGE_MAX_LEVERAGE,
        "position_floor": round(position_floor, 2),
        "forced_by_position": forced,
        "note": ((f"지금 포지션 {current_notional:,.0f} USDT 때문에 {setting}배 아래로는 못 내립니다"
                  f" (내리면 증거금이 순자산을 넘어 거래소가 -2028 로 거부합니다)"
                  f" — 포지션을 줄이면 {choose_leverage(min_feasible=min_feasible, cap_x=cap_notional/equity):.1f}배까지 내려갑니다")
                 if forced else
                 f"거래소 레버리지를 {setting}배로 두면 열 수 있는 총 명목이 "
                 f"{max_notional:,.0f} USDT 로 묶입니다 (정책 상한 {cap_notional:,.0f})"),
    }


def choose_leverage(*, min_feasible: float, cap_x: float) -> float:
    """설정 레버리지의 **목표값**(눈금 올림 전). 여유를 얼마나 둘지가 유일한 판단이다.

    min_feasible 아래면 정책 상한만큼도 못 열어 주문이 거부된다.
    cap_x 에 딱 붙이면 거래소가 우리 상한을 강제해 주지만 수수료·헤지 반대다리 몫이 없다.
    위로 멀어질수록 주문은 편해지고 거래소 방어는 약해진다.
    """
    # **cap_x 의 1.2배**로 정한다. 근거 셋:
    #   ① 딱 붙이면(1.0배) 정책 상한만큼 열 때 초기증거금이 순자산의 100% 가 된다. 거래소는
    #      가용증거금을 **초과**하면 거부하므로 수수료·펀딩 차감분에서 마지막 주문이 막힌다.
    #      눈금이 (…5, 8, 10, 15…)로 성기어 «조금만 위»라는 선택지가 없다.
    #   ② 1.2배면 상한까지 열어도 증거금이 순자산의 83% 라 17% 가 남는다 -- 수수료·펀딩·
    #      반올림에 충분하고, 헤지 두 다리를 합쳐도 cap 안이면 그대로 성립한다.
    #   ③ 그래도 거래소 천장이 정책의 1.2배라 **방어가 산다**(enforces_cap 판정 1.25배 안).
    #      설정을 안 만지면 천장이 150배라 정책의 18배까지 열린다 -- 그게 지금 상태다.
    # 여유를 더 주고 싶으면 이 배수만 올린다. 1.25 를 넘기면 enforces_cap 이 False 가 되고
    # 화면이 «거래소가 상한을 강제하지 않습니다»라고 말한다.
    return max(min_feasible, cap_x) * 1.2


# ── 손절이 생기면 «위험»의 정의가 바뀐다 (2026-09-13) ────────────────────────
# 🔴손절 3% 가 청산선(100/L = 16.7% @6배)보다 **먼저** 오므로 청산은 구조적으로 도달 불가다
# (테이프 실측 파산율 전 구간 0%, 30배에서도 1.27%). 그러면 생존 제약이 아무것도 안 막는다.
# 대신 묶는 것은 **손절의 반복**이다: 손절당 손실 = 손절폭 × 배수 이고, 그게 쌓여 MDD 가 된다.
#   6배: 손절당 18% · 1년 중앙 계좌 9.19배 · 중앙 MDD 82.9%
#   16배: 손절당 48% · 1년 중앙 **0.00** · MDD 100%  ← 파산율은 0% 인데 계좌가 안 남는다
# ⇒ 배수를 고르는 기준이 «파산 확률»에서 **«감당 가능한 MDD»** 로 옮겨간다.
# 화면도 청산거리 대신 이 값들을 앞세운다 -- 도달 못 하는 선을 위험 지표로 쓰면 안 된다.
# 발동률 자체는 위 STOP_HIT_RATE_BY_HOLD(지평별 실측)를 쓴다.


def stop_risk(*, stop_pct: float, leverage: float, hold_min: int = 240,
              side: str = "LONG", hit_rate: float | None = None) -> dict:
    """손절이 있을 때의 위험. 청산거리가 아니라 **이 값들이 진짜 지표다**.

    `consecutive_to_half` 는 «몇 번 연속 잘리면 계좌가 반토막인가»다 -- 손절당 손실이
    커질수록 급격히 줄어든다(18%면 4회, 48%면 2회)."""
    if not (leverage > 0 and stop_pct > 0):
        return {"available": False}
    # 🔴보유시간마다 다르다. 예전에는 4시간 실측 0.043 을 전 지평에 썼다(1440분 7.4배 과소).
    hit_rate = stop_hit_rate(hold_min, side) if hit_rate is None else float(hit_rate)
    per = stop_pct * leverage                       # 손절당 계좌 손실(비용 전)
    def consec(target: float) -> int | None:
        if per >= 1.0:
            return 1
        n, w = 0, 1.0
        while w > 1 - target and n < 99:
            w *= (1 - per); n += 1
        return n
    # 실계좌 빈도(하루 2.01건)지만 **지평이 물리적 상한을 만든다** -- 1440분을 들면서
    # 하루 2건은 불가능하다. 둘의 작은 쪽을 쓴다.
    trades_per_year = min(735.0, 525_600.0 / max(1, hold_min))
    return {
        "available": True,
        "per_stop_pct": round(100 * per, 1),
        "consecutive_to_half": consec(0.5),
        "consecutive_to_80": consec(0.8),
        "hit_rate": round(hit_rate, 4),
        "hold_min": hold_min,
        "trades_per_year": int(round(trades_per_year)),
        "expected_stops_per_year": int(round(hit_rate * trades_per_year)),
        # 청산선은 이제 «도달 불가»다. 그 사실 자체를 값으로 낸다.
        "liq_unreachable": stop_pct < 1.0 / leverage,
        "liq_distance_pct": round(100.0 / leverage, 1),
    }


def prescribe(*, risk_table: dict, atr_pct: float | None, side: str, equity: float,
              cap_x: float, acc: float = PRESCRIBE_ACC,
              existing_notional: float = 0.0,
              policy_cap_x: float | None = None,
              hold_override: int | None = None,
              funding_bp_8h: float | None = None) -> dict:
    """진입 때마다 정해 주는 **세 값**: 명목배수 · 보유시간 · 분할 횟수.

    사용자: *"레버리지·보유시간·분할 횟수를 모델링해서 진입할 때마다 픽스해 줬으면"* +
    *"30배 2시간을 고정하지 말고 현재 피쳐들을 보고 모델이 판단"*.

    ## 누가 무엇을 정하나 — 사용자 지적(*"내 데이터가 사이징에 최적화되어 있다면서"*)이 옳다
      · **명목배수 L**: 100% 모델. 학습된 보유시간 조건부 MAE 분위(검증 초과율 0.098%)의
        역수를 정책 상한들과 min. 방향 가정이 **하나도 안 들어간다**. 이 저장소가 강한 축이다.
      · **분할 k**: **항상 1**. 자유 변수가 아니다 -- 869일 1분봉에서 «k 분할» 을 «같은 평균
        노출의 단일 진입» 과 크기 매칭해 붙이면 **6/6 칸 전패**였고, 생존선 위(30배)에서는
        파산조차 못 줄였다(240분 3.98% -> 4.52%로 오히려 악화: 역행 중 평단이 나빠지는데
        마지막엔 어차피 전량을 든다). 노출을 낮추려면 칸이 아니라 **L 을 낮춘다**.
        근거: research_joint_leverage_hold_tranche_20260913.
      · **보유시간 H**: 여기만 방향이 필요하다. 목적함수가 «드리프트 − 분산»인데 드리프트가
        (2a−1)·b 이고, 크기 모델은 b 만 주고 (2a−1) 은 못 준다. 다만 **a 를 정확히 알 필요는
        없다**: a>=0.60 이면 0.60~0.75 전 구간에서 «잔잔=길게 · 험함=짧게» 분기가 유지된다.
        a 는 시장이 아니라 **사용자 진입의 실력**이라 모델이 낼 수 있는 값이 아니다.

    목적함수는 **시간당** 로그성장이다. 건당으로 고르면 짧은 지평을 여러 번 굴린다는 사실이
    빠져 전 국면에서 1440분으로 퇴화한다(실측).
    """
    hold = recommend_hold(risk_table, side, cap_x, atr_pct, acc, funding_bp_8h)
    if not hold.get("available"):
        return {"available": False, "reason": hold.get("reason", "위험모델 없음")}
    # 🔴호출부가 지평을 정했으면 그걸 쓴다. 여기서 다시 고르면 «상한을 계산한 지평»과
    # «권고 지평»이 갈라져, 권고대로 들면 진입 직후 예산 사다리를 위반한다(2026-09-13 감사).
    H = int(hold_override) if hold_override else hold["recommended_min"]
    if H not in {r["hold_min"] for r in hold["table"]}:
        H = hold["recommended_min"]
    L = allowed_x(risk_table, H, side, cap_x)
    m = safe_mae(risk_table, H, side)
    total = equity * L if (equity > 0 and L) else 0.0
    row = next(r for r in hold["table"] if r["hold_min"] == H)
    # a 를 흔들었을 때 처방이 얼마나 움직이나. 화면이 «이 값이 얼마나 믿을 만한가»를 말한다.
    sens = {}
    for x in (0.58, 0.60, 0.65, 0.70):
        h2 = recommend_hold(risk_table, side, cap_x, atr_pct, x, funding_bp_8h)
        if h2.get("available"):
            sens[str(x)] = h2["recommended_min"]
    # 거래소에 걸 설정값. 기준은 **정책 상한**이지 이 지평의 배수가 아니다 -- 사용자가 보유
    # 시간을 바꾸면 배수가 달라지므로, 거래소에 새기는 천장은 전 지평의 상한이어야 한다.
    # 🔴`policy_cap_x` 는 **지평과 무관한** 정책 천장이다(원장·순자산 상한의 작은 쪽).
    # cap_x 를 쓰면 안 된다 -- 그건 위험모델의 지평별 배수까지 min 한 값이라 사용자가 보유시간
    # 선택을 바꿀 때마다 거래소 설정이 따라 움직인다. 거래소 레버리지는 한 번 걸어 두는 값이다.
    # (2026-09-13 라이브 확인에서 실제로 이 값이 4.49 를 따라가 설정이 8배로 내려가고
    #  enforces_cap 이 False 로 떴다. 로컬에서는 순자산 상한이 묶여 증상이 안 보였다.)
    lev = leverage_setting(cap_notional=equity * (policy_cap_x or cap_x), equity=equity,
                           existing_notional=existing_notional,
                           current_notional=existing_notional)
    return {
        "available": True,
        "leverage": round(L, 2), "hold_min": H, "tranches": 1,
        "exchange_leverage": lev,
        "total_notional": round(total, 2),
        "room_notional": round(max(0.0, total - max(0.0, existing_notional)), 2),
        "safe_mae_pct": m, "liq_distance_pct": round(100.0 / L, 2) if L else None,
        "acc_assumed": acc, "breakeven_acc": row["breakeven_acc"],
        "growth_per_hour": row["growth_per_hour"],
        "none_positive": hold.get("none_positive"),
        "hold_by_acc": sens,
        "stop_risk": stop_risk(stop_pct=STOP_LOSS_PCT, leverage=L or 0.0, hold_min=H, side=side),
        "cost_bp": row["cost_bp"], "funding_bp": row["funding_bp"],
        # 🔴«파산»이 아니라 이 값이 배수를 묶는다. 실측 L=16 은 파산 0% 인데 계좌가 1.3% 남았다.
        "expected_mdd": round(expected_max_drawdown(
            L or 0.0, row["move_bp"], row["sd_bp"], acc, H, row["cost_bp"]), 2),
        "size_source": "MAE 분위 모델(방향 가정 없음)",
        "tranche_reason": ("일괄 — 분할은 «같은 평균 노출의 작은 단일 진입»에 869일 6/6 칸 전패,"
                           " 생존선 위에서는 파산도 못 줄였습니다. 노출을 낮추려면 배수를 낮춥니다"),
        "hold_reason": (f"시간당 로그성장 최대. 이 국면(역행 {m}%)에서 {H}분·{round(L,2)}배 · "
                        f"손익분기 실력 {int(round(100*row['breakeven_acc']))}%"),
    }


def plan_now(*, side: str, equity: float, existing_notional: float, unrealized_pnl: float,
             risk_table: dict, vol_bpm: float | None, cap_x: float,
             atr_pct: float | None = None, hold_min: int | None = None,
             policy_cap_x: float | None = None,
             funding_bp_8h: float | None = None) -> dict:
    """한 번에 넷: 보유시간 권고 · 크기(그 보유시간의 허용 배수) · 집행 · 분할.

    🔴`hold_min` 이 오면 **처방도 그 지평을 쓴다**. 예전에는 처방이 독자적으로 다시 골라
    «상한은 240분 · 권고는 1440분» 처럼 갈라졌고, 권고대로 들면 예산 사다리를 즉시 위반했다."""
    hold = recommend_hold(risk_table, side, cap_x, atr_pct, funding_bp_8h=funding_bp_8h)
    rx = prescribe(risk_table=risk_table, atr_pct=atr_pct, side=side, equity=equity,
                   cap_x=cap_x, existing_notional=existing_notional,
                   policy_cap_x=policy_cap_x, hold_override=hold_min,
                   funding_bp_8h=funding_bp_8h)
    H = hold_min or (hold.get("recommended_min") if hold.get("available") else max(HOLD_CHOICES))
    L = allowed_x(risk_table, H, side, cap_x)
    room = max(0.0, equity * L - existing_notional) if L else 0.0
    m = safe_mae(risk_table, H, side)
    return {
        "side": side, "hold": hold, "hold_min": H, "prescription": rx,
        "size": {"leverage": round(L, 2) if L else None, "safe_mae_pct": m,
                 "total_notional": round(equity * L, 2) if L else None,
                 "room_notional": round(room, 2)},
        "execution": execution_plan(vol_bpm, hold_min=H, side=side,
                                    funding_bp_8h=funding_bp_8h),
        "entry_split": {
            # 🔴k 는 자유 변수가 아니다(2026-09-13 실측) -- 아래 prescribe 주석 참조.
            "tranches": 1, "spread_min": 0,
            "reason": ("일괄 — 분할은 «같은 평균 노출의 작은 단일 진입»에 869일 6/6 칸 전패"),
            # 🔴«역행 중 추가 금지»를 여기서 제거했다(2026-09-13). 사용자 실계좌 69왕복에서
            # 분할 자체는 건당 수익률과 **무관**했다(순위상관 −0.04 · 단일 대비 차 95%CI 0 포함)
            # 이고 순손익의 60%(+153/+253)를 분할 거래가 만들었다. 09-06 «24/24 전패»는 칩 신호
            # 위 시뮬레이션이라 이 모집단이 아니다. 진짜 결합은 **분할→크기**(상관 +0.51)이고
            # 최악 1건도 분할 탓이 아니라 명목 35,291(30배)·보유 28.7시간이었다 -- 둘 다 이미
            # 상한과 예산 사다리가 막는다. 근거: research_user_scaling_in_pnl_attribution_20260913
            "room_notional": round(room, 2),
            "rule": ("추가는 상한 여유 안에서만 하면 됩니다 — 실계좌 69왕복에서 분할 자체는"
                     " 건당 수익률과 무관했고(상관 −0.04), 손실을 만든 건 분할이 데려온"
                     " 크기(상관 +0.51)와 보유시간이었습니다")},
        "exit_ladder": hold_budget(equity, existing_notional, risk_table, side, cap_x),
    }


def _self_check() -> None:
    live = {"60": {"LONG": {"safe_mae_pct": 1.821}, "SHORT": {"safe_mae_pct": 1.592}},
            "120": {"LONG": {"safe_mae_pct": 2.365}, "SHORT": {"safe_mae_pct": 2.173}},
            "240": {"LONG": {"safe_mae_pct": 3.36}, "SHORT": {"safe_mae_pct": 3.168}},
            "480": {"LONG": {"safe_mae_pct": 6.967}, "SHORT": {"safe_mae_pct": 6.117}},
            "1440": {"LONG": {"safe_mae_pct": 23.236}, "SHORT": {"safe_mae_pct": 21.643}}}
    # 프런티어(과거 계수 × 지금 atr_pct): 손익분기 정확도는 지평에 단조 감소, 잔잔한 장에서
    # 1시간은 비용을 못 넘는다(실측 3.87bp/봉 → 1h |r| 13.6bp → a* 71.6%)
    # ── 비용 세 조각 (2026-09-13 감사 수정) ──────────────────────────────────
    # 손절률은 지평에 단조 증가하고 롱이 숏보다 높다(하락이 더 빠르다)
    rates = [stop_hit_rate(H, "LONG") for H in HOLD_CHOICES]
    assert all(a < b for a, b in zip(rates, rates[1:])), rates
    assert all(stop_hit_rate(H, "LONG") > stop_hit_rate(H, "SHORT") for H in HOLD_CHOICES)
    assert abs(stop_hit_rate(1440, "LONG") - 0.3195) < 1e-9, "실측표가 바뀌면 연구 스크립트부터"
    # 🔴예전 버그: 4시간 실측 0.043 을 전 지평에 썼다. 1440분이 7배 넘게 달라야 한다.
    assert stop_hit_rate(1440, "LONG") / stop_hit_rate(60, "LONG") > 7.0
    # 비용은 지평에 단조 증가하고, 펀딩 때문에 **롱이 숏보다 비싸다**(현 국면)
    costs = [expected_cost_bp(H, "LONG") for H in HOLD_CHOICES]
    assert all(a < b for a, b in zip(costs, costs[1:])), costs
    assert expected_cost_bp(1440, "LONG") > expected_cost_bp(1440, "SHORT") + 3.0
    assert funding_cost_bp(480, "LONG") > 0 > funding_cost_bp(480, "SHORT")
    assert funding_cost_bp(1440, "LONG") == 3 * funding_cost_bp(480, "LONG")
    # 라이브 펀딩이 0 이면 비용에서 펀딩이 빠진다(폴백을 조용히 안 쓴다)
    assert expected_cost_bp(1440, "LONG", 0.0) < expected_cost_bp(1440, "LONG")

    h = recommend_hold(live, "LONG", 8.0, 0.000387)
    assert h["available"], h
    be = [r["breakeven_acc"] for r in h["table"]]
    assert be == sorted(be, reverse=True) and abs(be[0] - 0.723) < 0.002, be
    # 🔴긴 지평이 더 이상 «거의 공짜»가 아니다: 1440분 손익분기가 0.527 -> 0.579 로 올랐다
    assert abs(h["table"][-1]["breakeven_acc"] - 0.579) < 0.002, h["table"][-1]
    assert h["table"][-1]["cost_bp"] > 2 * ROUND_TRIP_COST_BP * 0.9, "긴 지평 비용이 두 배 넘는다"
    assert abs(h["table"][0]["move_bp"] - 13.6) < 0.2, h["table"][0]
    # 🔴**시간당** 기준이면 정확도가 오를수록 최적 지평이 **짧아진다**(2026-09-13 목적함수 교체).
    # 건당 기준일 때와 방향이 반대다 -- 짧은 지평은 여러 번 굴릴 수 있기 때문이고, 그 사실이
    # 건당 지표에는 안 들어가서 전 국면 1440분으로 퇴화했었다.
    bba = h["best_by_acc"]
    assert bba["0.52"] >= bba["0.66"], bba
    # 격변기(atr_pct 3배)에는 같은 정확도에서 짧은 지평도 비용을 넘는다
    h2 = recommend_hold(live, "LONG", 8.0, 0.000387 * 3)
    assert h2["table"][0]["breakeven_acc"] < be[0]
    # 🔴전부 음수인 국면은 «권고»가 아니라 «전부 비용 미달»이라고 말해야 한다
    assert h["none_positive"] is True and "못 넘는 것으로 나옵니다" in h["reason"], h["reason"]
    # 🔴«못 넘는다»를 사실로 단정하지 않는다 -- 이 추정은 실측 40칸 중 34칸에서 낮게 나왔다
    #   (2026-09-14 절단 연구). 보수적이라는 사실이 문구와 필드 둘 다에 있어야 한다.
    assert h["estimate_is_conservative"] is True and "보수적" in h["reason"], h["reason"]
    assert "바뀌지 않았습니다" in h["bias_note"], h["bias_note"]
    assert abs(h["breakeven_min_acc"] - min(r["breakeven_acc"] for r in h["table"])) < 1e-9
    # 정확도를 손익분기 위로 올리면 양수가 되고 문장도 바뀐다
    hi = recommend_hold(live, "LONG", 8.0, 0.000387, acc=0.70)
    assert hi["none_positive"] is False and "최대" in hi["reason"], hi["reason"]
    assert recommend_hold(live, "LONG", 8.0, None)["available"] is False
    assert recommend_hold({}, "LONG", 8.0, 0.001)["available"] is False
    assert allowed_x(live, 60, "LONG", 8.0) == 8.0 and allowed_x(live, 1440, "LONG", 8.0) < 5, "상한·생존 최솟값"
    # 집행: 변동성이 오르면 기대 체결이 줄고, 극단이면 시장가
    assert expected_fill_sec(None) is None and expected_fill_sec(2.67) > expected_fill_sec(9.97)
    assert abs(expected_fill_sec(9.97) - 1.76) < 0.1, expected_fill_sec(9.97)
    assert execution_plan(45.0)["exit"]["mode"] == "market"
    assert execution_plan(12.0)["exit"]["mode"] == "peg_repeg"
    # 사다리: 명목이 커질수록 예산이 줄고 필요 비율은 시간에 단조 증가
    b = hold_budget(1000.0, 5000.0, live, "LONG", 8.0)
    assert b["budget_min"] == 480, b            # 5배: 1일(4.3배)만 예산 밖
    fr = [r["required_fraction"] for r in b["ladder"]]
    assert fr == sorted(fr) and fr[-1] > 0, fr
    assert hold_budget(1000.0, 9000.0, live, "LONG", 8.0)["budget_min"] == 0, "8배 상한 초과"
    assert hold_budget(1000.0, 500.0, live, "LONG", 8.0)["budget_min"] == 1440
    # 역행으로 순자산이 줄면 같은 명목에서 사다리가 조여진다
    assert (hold_budget(800.0, 5000.0, live, "LONG", 8.0)["ladder"][-1]["required_fraction"]
            > b["ladder"][-1]["required_fraction"])
    # 플랜: 추가 여부는 **상한 여유**로만 말한다(2026-09-13 물타기 금지 철회 -- 위 주석).
    p = plan_now(side="LONG", equity=1000.0, existing_notional=4000.0, unrealized_pnl=-30.0,
                 risk_table=live, vol_bpm=8.0, cap_x=8.0, atr_pct=0.000387)
    assert p["hold_min"] == p["hold"]["recommended_min"]
    assert "add_allowed" not in p["entry_split"], "평가손익으로 추가를 막지 않는다"
    # 평가손익 부호가 계획을 바꾸면 안 된다 -- 그게 철회한 규칙이었다
    q = plan_now(side="LONG", equity=1000.0, existing_notional=4000.0, unrealized_pnl=+30.0,
                 risk_table=live, vol_bpm=8.0, cap_x=8.0, atr_pct=0.000387)
    assert q["entry_split"] == p["entry_split"], "평가손익이 분할 계획을 바꿨다"
    assert abs(p["size"]["room_notional"] - max(0.0, 1000.0 * p["size"]["leverage"] - 4000.0)) < 10.0  # leverage 는 소수 2자리 반올림
    assert p["execution"]["exit"]["mode"] == "peg_repeg"
    p = plan_now(side="SHORT", equity=1000.0, existing_notional=0.0, unrealized_pnl=0.0,
                 risk_table=live, vol_bpm=None, cap_x=8.0, hold_min=1440)
    assert p["hold_min"] == 1440
    assert p["size"]["leverage"] < 5 and p["entry_split"]["tranches"] == 1, p["size"]
    # ── 손절 기준 위험 지표 ──────────────────────────────────────────────────
    # 🔴stop_risk 가 hold_min 을 **실제로 쓴다**(예전엔 받고도 무시해 전 지평이 같았다)
    assert (stop_risk(stop_pct=0.03, leverage=6.0, hold_min=1440)["hit_rate"]
            > 7 * stop_risk(stop_pct=0.03, leverage=6.0, hold_min=60)["hit_rate"])
    # 1440분을 들면서 하루 2건은 불가능하다 -- 빈도가 지평 상한에 묶여야 한다
    assert stop_risk(stop_pct=0.03, leverage=6.0, hold_min=1440)["trades_per_year"] == 365
    assert stop_risk(stop_pct=0.03, leverage=6.0, hold_min=60)["trades_per_year"] == 735
    # ── 기대 최대낙폭 (2026-09-14 파산항 연구) ──────────────────────────────
    # 🔴파산이 아니라 이 값이 배수를 묶는다. 배수가 오르면 낙폭도 단조 증가해야 한다.
    # 🔴**실측 대조**: research_ruin_term_in_objective_20260914 의 테이프 표본
    #    (869일 · H=240 · a=0.62 · b=92.6bp · sd=141.7bp)에서 잰 1년 중앙 MDD 다.
    #    🔴기준값은 **실제 연대순 1년 창**이다(IID 리샘플 아님) -- 뭉침이 들어간 진짜 값이라야
    #    보정 상수까지 같이 검사된다. 파라미터를 바꾸면 기준값도 바뀐다.
    TAPE_B, TAPE_SD = 92.6, 141.7
    truth = {2: 0.364, 3: 0.503, 4: 0.618, 6: 0.784, 8: 0.886}
    for L, want in truth.items():
        got = expected_max_drawdown(L, TAPE_B, TAPE_SD, 0.62, 240)
        assert abs(got - want) < 0.10, (L, got, want)
    mdds = [expected_max_drawdown(L, TAPE_B, TAPE_SD, 0.62, 240) for L in truth]
    assert all(a <= b for a, b in zip(mdds, mdds[1:])), "배수가 오르면 낙폭도 올라야 한다"
    # 🔴저표류 분기가 **1.0 으로 포화되지 않아야** 한다 -- 처음 판이 그래서 못 썼다.
    assert expected_max_drawdown(4, 38.8, 59.5, 0.62, 240) < 0.99, "저표류 분기가 죽었다"
    # 실력이 오르면 같은 배수의 낙폭이 준다
    assert (expected_max_drawdown(6, TAPE_B, TAPE_SD, 0.66, 240)
            < expected_max_drawdown(6, TAPE_B, TAPE_SD, 0.60, 240))
    # 배수가 0 이면 낙폭을 말할 대상이 없다
    assert expected_max_drawdown(0.0, TAPE_B, TAPE_SD, 0.66, 240) == 1.0

    sr = stop_risk(stop_pct=0.03, leverage=6.0)
    assert sr["per_stop_pct"] == 18.0, sr           # 3% × 6배
    assert sr["liq_unreachable"] is True, "3% < 16.7% 이므로 청산은 도달 불가"
    assert sr["consecutive_to_half"] == 4, sr       # 0.82^4 = 0.45
    # 배수를 키우면 손절당 손실이 커지고 견디는 횟수가 급격히 준다
    prev = 99
    for L in (4.0, 6.0, 10.0, 16.0, 25.0):
        c = stop_risk(stop_pct=0.03, leverage=L)["consecutive_to_half"]
        assert c <= prev, (L, c, prev); prev = c
    assert stop_risk(stop_pct=0.03, leverage=16.0)["per_stop_pct"] == 48.0
    # 🔴손절이 청산선 밖이면 보호가 사라진다 -- 그 사실을 값으로 내야 한다
    assert stop_risk(stop_pct=0.03, leverage=40.0)["liq_unreachable"] is False
    assert stop_risk(stop_pct=0.0, leverage=6.0)["available"] is False

    # ── 처방: 세 값을 한 번에 ────────────────────────────────────────────────
    rx = prescribe(risk_table=live, atr_pct=0.000387, side="LONG", equity=1000.0, cap_x=8.0)
    assert rx["available"] and rx["tranches"] == 1, "분할은 자유 변수가 아니다(869일 6/6 전패)"
    assert abs(rx["leverage"] - allowed_x(live, rx["hold_min"], "LONG", 8.0)) < 0.01, "배수는 생존·상한의 min"
    assert abs(rx["total_notional"] - 1000.0 * rx["leverage"]) < 10.0  # leverage 는 반올림값
    assert abs(rx["liq_distance_pct"] - 100.0 / rx["leverage"]) < 0.1   # 둘 다 반올림값
    # 🔴호출부가 지평을 정하면 처방이 그걸 따른다 -- 안 따르면 «상한 지평»과 «권고 지평»이
    # 갈라져 권고대로 들 때 예산 사다리를 즉시 위반한다(2026-09-13 감사에서 실제로 그랬다).
    for H in HOLD_CHOICES:
        r3 = prescribe(risk_table=live, atr_pct=0.000387, side="LONG", equity=1000.0,
                       cap_x=8.0, hold_override=H)
        assert r3["hold_min"] == H, (H, r3["hold_min"])
        assert abs(r3["leverage"] - allowed_x(live, H, "LONG", 8.0)) < 0.01, H
        assert r3["stop_risk"]["hold_min"] == H
    # 위험표에 없는 지평을 넘기면 조용히 따르지 않고 권고로 되돌린다
    assert prescribe(risk_table=live, atr_pct=0.000387, side="LONG", equity=1000.0,
                     cap_x=8.0, hold_override=999)["hold_min"] in HOLD_CHOICES
    hb = rx["hold_by_acc"]
    assert hb["0.58"] >= hb["0.7"], hb        # 시간당 목적함수: 실력이 오르면 짧아진다
    # ── 거래소 레버리지 설정 ──────────────────────────────────────────────────
    lv = rx["exchange_leverage"]
    assert lv["available"] and lv["setting"] in LEVERAGE_STEPS, lv
    # 🔴거래소 설정은 **보유시간 선택에 흔들리면 안 된다**(2026-09-13 라이브 회귀).
    # 정책 천장을 주면 지평이 달라져도 같은 값이 나와야 한다.
    setts = {prescribe(risk_table=live, atr_pct=0.000387, side="LONG", equity=1085.0,
                       cap_x=allowed_x(live, hh, "LONG", 8.0), policy_cap_x=8.0
                       )["exchange_leverage"]["setting"] for hh in HOLD_CHOICES}
    assert len(setts) == 1, f"보유시간에 따라 거래소 설정이 흔들린다: {setts}"
    # 정책 천장을 안 주면 옛 동작(지평별 배수)으로 떨어진다 -- 그게 회귀의 원인이었다
    drift = {prescribe(risk_table=live, atr_pct=0.000387, side="LONG", equity=1085.0,
                       cap_x=allowed_x(live, hh, "LONG", 8.0)
                       )["exchange_leverage"]["setting"] for hh in HOLD_CHOICES}
    assert len(drift) > 1, "대조군이 안 흔들린다 -- 이 검사가 회귀를 못 잡는다"
    # 🔴기존 포지션이 설정을 바꾸면 안 된다(이중 계상 회귀 방지). 정책 상한은 총 명목에 걸린다.
    a0 = leverage_setting(cap_notional=8000.0, equity=1000.0, existing_notional=0.0)
    a1 = leverage_setting(cap_notional=8000.0, equity=1000.0, existing_notional=4000.0)
    assert a0["setting"] == a1["setting"] == 10, (a0["setting"], a1["setting"])
    # 정책 상한만큼은 반드시 열려야 한다 -- 설정이 min_feasible 아래면 주문이 거부된다
    for cap_n, eq in ((8000.0, 1000.0), (2000.0, 1000.0), (500.0, 1000.0), (40000.0, 1000.0)):
        r = leverage_setting(cap_notional=cap_n, equity=eq)
        assert r["max_notional"] >= cap_n - 1e-6, (cap_n, eq, r)
        assert r["margin_pct_of_equity"] <= 100.0, r
        assert r["setting"] <= EXCHANGE_MAX_LEVERAGE
    # 여유는 정책의 1.2배 목표 -- 거래소 천장이 정책을 크게 넘으면 방어가 죽는다
    assert a0["max_notional"] <= 8000.0 * 1.25 and a0["enforces_cap"] is True, a0
    assert leverage_setting(cap_notional=0.0, equity=1000.0)["available"] is False
    # 🔴열린 포지션이 바닥을 만든다 -- 내리면 -2028 로 거부되므로 애초에 그 아래를 권하지 않는다.
    big = leverage_setting(cap_notional=8000.0, equity=1000.0,
                           existing_notional=11000.0, current_notional=11000.0)
    assert big["forced_by_position"] is True and big["setting"] >= 11, big
    assert big["setting"] * 1000.0 >= 11000.0, "설정으로 기존 포지션을 못 버티면 거부된다"
    assert "-2028" in big["note"] and "줄이면" in big["note"], big["note"]
    small = leverage_setting(cap_notional=8000.0, equity=1000.0,
                             existing_notional=3000.0, current_notional=3000.0)
    assert small["forced_by_position"] is False and small["setting"] == 10, small
    # 포지션이 커질수록 바닥이 단조 증가해야 한다
    prev = 0
    for n in (0.0, 3000.0, 8000.0, 11000.0, 20000.0):
        st = leverage_setting(cap_notional=8000.0, equity=1000.0,
                              existing_notional=n, current_notional=n)["setting"]
        assert st >= prev, (n, st, prev); prev = st
    assert leverage_setting(cap_notional=8000.0, equity=0.0)["available"] is False
    # 분할 횟수는 어떤 입력에서도 1 이다 -- 여기가 흔들리면 실험을 다시 돌려야 한다
    for cx in (2.0, 8.0, 25.0):
        for apx in (0.0002, 0.001, 0.003):
            assert prescribe(risk_table=live, atr_pct=apx, side="LONG", equity=1000.0,
                             cap_x=cx)["tranches"] == 1, (cx, apx)
    assert prescribe(risk_table={}, atr_pct=0.000387, side="LONG", equity=1000.0,
                     cap_x=8.0)["available"] is False
    # ⭐크기는 방향 가정과 **무관**해야 한다 -- 정확도를 바꿔도 그 보유시간의 배수는 그대로다
    for acc in (0.52, 0.60, 0.75):
        r2 = prescribe(risk_table=live, atr_pct=0.000387, side="LONG", equity=1000.0,
                       cap_x=8.0, acc=acc)
        assert abs(r2["leverage"] - allowed_x(live, r2["hold_min"], "LONG", 8.0)) < 0.01, acc

    full = plan_now(side="LONG", equity=1000.0, existing_notional=99999.0, unrealized_pnl=0.0,
                    risk_table=live, vol_bpm=8.0, cap_x=8.0, atr_pct=0.000387)
    assert full["entry_split"]["room_notional"] == 0.0, "상한을 채웠으면 여유가 0 이어야 한다"
    p = plan_now(side="LONG", equity=0.0, existing_notional=0.0, unrealized_pnl=0.0,
                 risk_table={}, vol_bpm=None, cap_x=8.0)
    assert p["size"]["leverage"] is None and p["exit_ladder"]["budget_min"] == 0
    print("통과 — 비용 세 조각(수수료·손절·펀딩) · 손절기준 위험 · 처방(배수·보유·분할·거래소설정)"
          " · 보유시간 프런티어 · 집행 · 예산 사다리 · 플랜 조립")


if __name__ == "__main__":
    _self_check()
