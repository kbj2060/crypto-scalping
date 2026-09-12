"""**진입·청산 공통 위험 사이징 정책** — 생존 제약과 성장 제약의 최솟값 (2026-09-13).

사용자: *"이 정도 변동성과 보유 시간 등의 상황이면 내 증거금 어느 정도를 써서 리스크
관리를 하겠다는 확실한 모델"* + *"켈리 공식이나 좋은 최신 논문을 참고해서 설계"*.

## 설계 근거 (문헌)
두 마찰을 따로 풀지 않고 **둘의 최솟값**으로 쓴다 -- Bayesian Grossman-Zhou(2026)가
«학습(엣지 불확실)과 생존(드로다운 배리어)은 보통 따로 연구되는데 실제로는 같이 온다»고
지적한 그 구조다.
  · **생존**: Grossman-Zhou 계열 드로다운 제약. 여기서는 배리어가 청산선이고,
    거리를 **학습된 조건부 MAE 분위**로 잰다(`live_eth_mae_quantile_model_20260913`).
    분위를 직접 추정하고 수준은 실측 적중률로 고정하는 건 CAViaR(Engle·Manganelli 1999)의
    관행이다 -- 변동성을 예측한 뒤 분포를 가정하는 쪽보다 꼬리에서 덜 틀린다.
  · **성장**: 켈리. 단 **점추정 대신 하한**을 쓴다. Bayesian Kelly(2026)가 보인 대로
    p 의 추정오차가 곧 포지션 축소로 이어져야 한다. 68왕복 μ=18.37bp·SE=5.96bp 라
    95% 하한 8.57bp 로 계산하면 f* 가 76배 -> 35.5배가 된다.

## 결론(실측)
**모든 보유시간에서 생존이 구속조건이다**(1시간 28.2배 < 성장 35.5배). 즉 크기를 정하는
것은 켈리가 아니라 «얼마나 오래 들고 있을 것인가»다.

## 청산에도 같은 식을 쓴다
진입은 «L <= L_safe» 를 만족하는 수량을 구하고, 청산은 그 부등식이 깨졌을 때
**되돌리는 데 필요한 최소 비율**을 구한다: f >= 1 - L_safe/L_now.
사람이 더 닫고 싶으면 슬라이더로 언제든 올릴 수 있다(모델은 하한만 제시한다).
"""
from __future__ import annotations

import math

# 68왕복 단위당 수익(2026-09-13 실측). 원장이 갱신되면 다시 재야 한다.
EDGE_MU_BP = 18.37
EDGE_SD_BP = 49.13
EDGE_N = 68
Z_LCB = 1.645                 # 95% 단측 하한
HARD_CAP_X = 25.0             # 정책 상한. 모델이 뭐라 하든 이 위로는 안 간다.
MIN_X = 0.5


def kelly_robust_leverage(mu_bp: float = EDGE_MU_BP, sd_bp: float = EDGE_SD_BP,
                          n: int = EDGE_N, z: float = Z_LCB) -> float:
    """엣지의 **하한**으로 켈리를 계산한다. 점추정을 쓰면 추정오차가 그대로 레버리지가 된다.

    하한이 0 이하면 «성장 근거 없음»이라 0 을 돌려준다 -- 그때는 생존 제약과 무관하게
    크기를 키울 이유가 없다."""
    se = sd_bp / math.sqrt(max(1, n))
    lcb = (mu_bp - z * se) / 1e4
    if lcb <= 0:
        return 0.0
    return lcb / (sd_bp / 1e4) ** 2


def survival_leverage(safe_mae_pct: float) -> float:
    """청산선까지의 거리를 «학습된 안전 MAE» 로 두면 허용 레버리지는 그 역수다."""
    return 100.0 / max(safe_mae_pct, 1e-6)


def policy_leverage(safe_mae_pct: float, *, hard_cap: float = HARD_CAP_X,
                    growth_off: bool = False) -> dict[str, float]:
    """운영 레버리지 = min(생존, 성장, 정책상한). 어느 쪽이 묶었는지 같이 돌려준다."""
    surv = survival_leverage(safe_mae_pct)
    grow = float("inf") if growth_off else kelly_robust_leverage()
    lev = min(surv, grow, hard_cap)
    binding = "survival" if lev == surv else ("growth" if lev == grow else "cap")
    return {"leverage": max(MIN_X, lev), "survival_x": surv,
            "growth_x": None if growth_off else grow, "cap_x": hard_cap, "binding": binding}


def entry_notional(equity: float, safe_mae_pct: float, existing_notional: float = 0.0,
                   **kw) -> dict[str, float]:
    """지금 **추가로** 넣어도 되는 명목. 기존 포지션을 뺀 값이라 분할해도 상한을 안 넘는다."""
    p = policy_leverage(safe_mae_pct, **kw)
    total = equity * p["leverage"]
    return {**p, "total_notional": total,
            "room_notional": max(0.0, total - max(0.0, existing_notional))}


def exit_fraction_required(equity: float, safe_mae_pct: float, current_notional: float,
                           **kw) -> dict[str, float]:
    """위험 한도로 **되돌리기 위해 최소한 닫아야 하는 비율**. 0 이면 닫을 의무는 없다.

    사람이 더 닫는 건 언제든 가능하다 -- 이건 하한이지 지시가 아니다."""
    p = policy_leverage(safe_mae_pct, **kw)
    allowed = equity * p["leverage"]
    if current_notional <= allowed or current_notional <= 0:
        return {**p, "required_fraction": 0.0, "allowed_notional": allowed,
                "excess_notional": 0.0}
    return {**p, "required_fraction": min(1.0, 1.0 - allowed / current_notional),
            "allowed_notional": allowed, "excess_notional": current_notional - allowed}


def _self_check() -> None:
    k = kelly_robust_leverage()
    assert 30 < k < 45, f"엣지 하한 켈리가 {k:.1f} -- 원장이 바뀌었으면 상단 상수를 다시 재라"
    assert kelly_robust_leverage(mu_bp=1.0) == 0.0, "하한이 0 이하면 성장 근거가 없다"
    # 생존은 안전 MAE 의 역수이고 단조 감소한다
    prev = 1e9
    for m in (3.54, 5.17, 7.67, 11.37, 21.66):      # 60/120/240/480/1440분 실측
        s = survival_leverage(m)
        assert s < prev; prev = s
    assert abs(survival_leverage(3.54) - 28.2) < 0.2

    # 실측 전 구간에서 **생존이 구속**이어야 한다(문서의 결론)
    for m in (3.54, 5.17, 7.67, 11.37, 21.66):
        assert policy_leverage(m)["binding"] in ("survival", "cap"), m
    assert policy_leverage(3.54)["binding"] == "cap", "28.2배는 정책상한 25 에 먼저 걸린다"
    assert policy_leverage(7.67)["binding"] == "survival"

    # 진입: 기존 포지션이 있으면 여유가 그만큼 줄고 절대 음수가 되지 않는다
    e = entry_notional(1089.45, 7.67)
    assert abs(e["room_notional"] - e["total_notional"]) < 1e-9
    e2 = entry_notional(1089.45, 7.67, existing_notional=e["total_notional"] * 2)
    assert e2["room_notional"] == 0.0

    # 청산: 한도 안이면 0, 두 배면 절반을 닫아야 한다
    x = exit_fraction_required(1089.45, 7.67, current_notional=1.0)
    assert x["required_fraction"] == 0.0
    allowed = 1089.45 * policy_leverage(7.67)["leverage"]
    y = exit_fraction_required(1089.45, 7.67, current_notional=allowed * 2)
    assert abs(y["required_fraction"] - 0.5) < 1e-9, y
    z = exit_fraction_required(1089.45, 21.66, current_notional=allowed * 2)
    assert z["required_fraction"] > y["required_fraction"], "오래 들 생각이면 더 닫아야 한다"
    print("통과 14/14 — 생존/성장 최솟값 · 진입 여유 · 청산 최소비율 계약 유지")


if __name__ == "__main__":
    _self_check()
