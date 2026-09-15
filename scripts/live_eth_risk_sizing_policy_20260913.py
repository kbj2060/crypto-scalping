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

## 🔴켈리는 지금 **한 번도 묶지 않는다** (2026-09-13 실측)
`min(생존, 성장, 정책상한)` 에서 성장 = 35.5배는 정책상한 25배보다 크고, 대시보드는 거기에
순자산 **6배**(`SIZING_CAP_EQUITY_X`)를 한 번 더 min 한다. 라이브 위험표 전 구간에서 채택은 4.3~8.0배이고 묶는 것은
**순자산 상한 아니면 생존**이다. 켈리가 실제 손잡이(8배)까지 내려오려면 μ ≤ 11.73bp 여야 하는데
지금 추정은 18.37bp·SE 5.96bp 다 -- 약 1.1 SE 거리라 **불가능하진 않고 지금은 잠들어 있다**.
그리고 μ·σ 는 **원장에서 온다**. 사용자가 «원장은 정답이 아니다»라고 한 바로 그 입력이므로,
켈리는 여기서 «크기를 정하는 식»이 아니라 **«엣지 추정이 무너지면 크기를 줄이는 안전판»** 이다.
남겨 두는 값은 그 역할뿐이다 -- 이 파일이 크기를 키우는 데 켈리를 쓴 적은 없다.

## 청산에도 같은 식을 쓴다
진입은 «L <= L_safe» 를 만족하는 수량을 구하고, 청산은 그 부등식이 깨졌을 때
**되돌리는 데 필요한 최소 비율**을 구한다: f >= 1 - L_safe/L_now.
사람이 더 닫고 싶으면 슬라이더로 언제든 올릴 수 있다(모델은 하한만 제시한다).
"""
from __future__ import annotations

import pathlib

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


# ── 드로다운 배리어 (2026-09-13, Risk-Constrained Kelly 계열) ────────────────
# 🔴지금까지 배리어는 **전손**이었다: «안전 MAE 만큼 역행하면 계좌가 0» 이 되는 배수를 썼다.
# 그런데 제약은 한 건이 아니라 **경로 전체**에 걸어야 한다(Busseti·Ryu·Boyd 2016,
# Risk-Constrained Kelly Gambling, J. Investing; MacLean·Thorp·Ziemba 의 드로다운 제약 계열).
# 실측(2026-09-13): 건당 파산 예산 0.1% 는 사용자 실제 빈도(하루 2.01건, 연 735건)에서
#   1주 1.4% · 1개월 5.9% · 3개월 16.6% · **1년 52.1%** 로 쌓인다.
# 배리어를 D(<1)로 낮추면 같은 사건이 전손이 아니라 D 만큼의 손실이 되어 경로가 살아남는다:
#   허용배수 = D × 100 / 안전MAE
# D=1.0 이 현행(전손 배리어)이고, 바꾸면 크기가 그만큼 줄어든다.
#
# 배리어별 허용 배수(2026-09-13 라이브 위험표 · 순자산 상한 8배 적용 후):
#     보유      안전MAE   D=1.0   D=0.5   D=0.3   D=0.2
#      60분      1.82%     8.00    8.00    8.00    8.00
#     240분      3.36%     8.00    8.00    8.00    5.95
#     480분      6.97%     8.00    7.18    4.31    2.87
#    1440분     23.24%     4.30    2.15    1.29    0.86
# 연 파산확률 근사(하루 2.01건): D=1.0 → 52% · D=0.5 → 27% · D=0.3 → 13% · D=0.2 → 4%
# ⭐짧은 보유는 순자산 상한이 먼저 묶어 **배리어를 낮춰도 안 변한다**. 줄어드는 건 긴 보유뿐인데,
#   거기가 2026-09-13 반사실에서 «위험의 거의 전부»로 지목된 꼬리다.
# ⚠️위 표의 «연 파산확률»은 **손절이 없던 시절** 값이다(2026-09-13 감사에서 확인).
# 지금은 손절 3% 가 청산선 안쪽이라 파산이 구조적으로 안 나고, 실제로 배수를 묶는 건
# 순자산 상한 6배다 -- 그 6배가 MDD 기준으로 골라진 값이라(§5.33) 이 배리어는 지금
# 놀고 있다. D 를 내리는 건 **긴 보유**에서만 효과가 있다. 바꾸려면 그 사실 위에서 바꾼다.
MAX_DRAWDOWN = 1.0  # TODO(human)


def survival_leverage(safe_mae_pct: float, max_drawdown: float = None) -> float:
    """«안전 MAE 만큼 역행해도 손실이 D 를 넘지 않는» 최대 배수.

    D=1.0 이면 종전과 같다(전손까지 허용). D<1 이면 그 비율만큼 작아진다."""
    d = MAX_DRAWDOWN if max_drawdown is None else max_drawdown
    return max(0.0, d) * 100.0 / max(safe_mae_pct, 1e-6)


def policy_leverage(safe_mae_pct: float, *, hard_cap: float = HARD_CAP_X,
                    growth_off: bool = False,
                    max_drawdown: float = None) -> dict[str, float]:
    """운영 레버리지 = min(생존, 성장, 정책상한). 어느 쪽이 묶었는지 같이 돌려준다.

    ⭐min() 이 맞는 이유(Risk-Constrained Kelly): g(f)=fμ−½f²σ² 는 f* 까지 **증가**하므로,
    제약 상한이 f* 보다 작으면 제약 아래에서의 최적해는 그 상한 자체다. 여기서는 생존 상한이
    항상 켈리(35.5배)보다 작아 min() 이 곧 제약 최적해다."""
    surv = survival_leverage(safe_mae_pct, max_drawdown)
    grow = float("inf") if growth_off else kelly_robust_leverage()
    lev = min(surv, grow, hard_cap)
    binding = "survival" if lev == surv else ("growth" if lev == grow else "cap")
    return {"leverage": max(MIN_X, lev), "survival_x": surv,
            "growth_x": None if growth_off else grow, "cap_x": hard_cap, "binding": binding,
            "max_drawdown": MAX_DRAWDOWN if max_drawdown is None else max_drawdown}


def evr_size_multiplier(state_path: str | None = None, max_lag_min: float = 30.0) -> dict:
    """E|r| 조건부 사이징 배수 ∈ [0,1]. 상태 파일이 없거나 낡으면 **1.0**(무효과)로 되돌린다.

    ⭐**줄이기만 한다.** 근거(실계좌 72왕복 · 09-15 · docs/homer §5.36-R): 현행 명목 ↔ E|r|백분위
    스피어만 **−0.426**(큰 E|r| 에 작게 걸고 있었다)인데 단위당 순손익은 정반대로 단조다
    (하위50% +10.34 / 50~80% +14.32 / **상위20% +53.18bp**). `현행 × E|r|백분위` 로 바꾸면
    명목 **42%** 로 손익 **90%**, 명목당 **+10.84→+23.54bp**, t **1.03→4.06**, 낙폭 **$444→$30**.
    ⚠️하한을 두면 나빠진다(×max(q,0.3) $376) — 하한이 나쁜 큰 거래를 살려둔다. **하한 없음.**
    🔴n=72 · 5주 · 단일자산이다. 배수는 **진입에만** 걸고 청산 하한에는 안 건다.
    🔴낡음 판정이 중요하다 — 워커가 죽으면 옛 배수로 계속 줄이는 게 아니라 **효과를 끈다.**"""
    import json as _json, time as _time
    # ⭐**전용 워커를 따로 띄우지 않는다.** 2026-09-15 에 다른 세션이 이미 20자산 E|r| 게이트
    #   워커를 배포했다(`live_evr_gate_worker_20260915.py` → `evr_gate_state.json`). 같은 모델을
    #   한 번 더 도는 건 요청경로 인라인 금지(2026-09-10 실장애)와 서버 부하 양쪽에 걸린다.
    #   `evr_q`(백분위)는 2026-09-15 에 그 워커에 실었다(a75cb85). 없으면 이 층은 **무효과(1.0)** 다.
    path = pathlib.Path(state_path) if state_path else (
        pathlib.Path(__file__).resolve().parents[1] / "data/live/evr_gate_state.json")
    try:
        d = _json.loads(path.read_text())
        age = (_time.time() - pathlib.Path(path).stat().st_mtime) / 60.0
        if age > max_lag_min:
            return {"evr_mult": 1.0, "evr_ok": False, "evr_why": f"상태 {age:.0f}분 낡음"}
        eth = next((a for a in d.get("assets", []) if a.get("asset") == "ETH"), None)
        if eth is None:
            return {"evr_mult": 1.0, "evr_ok": False, "evr_why": "ETH 항목 없음"}
        q = eth.get("evr_q", d.get("evr_q"))
        if q is None:
            return {"evr_mult": 1.0, "evr_ok": False, "evr_why": "백분위 미제공(워커 갱신 대기)"}
        m = float(q)
        if not (0.0 <= m <= 1.0):
            return {"evr_mult": 1.0, "evr_ok": False, "evr_why": f"배수 범위밖 {m}"}
        return {"evr_mult": round(m, 4), "evr_ok": True, "evr_q": m,
                "evr_bar": eth.get("ts"), "evr_age_min": round(age, 1)}
    except Exception as exc:
        return {"evr_mult": 1.0, "evr_ok": False, "evr_why": f"{type(exc).__name__}"}


def entry_notional(equity: float, safe_mae_pct: float, existing_notional: float = 0.0,
                   *, use_evr: bool = True, **kw) -> dict[str, float]:
    """지금 **추가로** 넣어도 되는 명목. 기존 포지션을 뺀 값이라 분할해도 상한을 안 넘는다.

    ⭐E|r| 배수는 **맨 뒤에** 곱한다 — 생존·성장·정책 상한을 먼저 통과시킨 뒤 «그 안에서 얼마나
    쓸까」를 정하는 층이다. 상한을 올리지 않으므로 기존 안전 성질이 그대로 보존된다."""
    p = policy_leverage(safe_mae_pct, **kw)
    total = equity * p["leverage"]
    room = max(0.0, total - max(0.0, existing_notional))
    e = evr_size_multiplier() if use_evr else {"evr_mult": 1.0, "evr_ok": False, "evr_why": "off"}
    return {**p, **e, "total_notional": total,
            "room_notional_precap": room,
            "room_notional": room * e["evr_mult"]}


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
    # ⚠️E|r| 배수는 «상한 뒤에」 곱하므로 상한 성질 검사는 use_evr=False 로 한다.
    e = entry_notional(1089.45, 7.67, use_evr=False)
    assert abs(e["room_notional"] - e["total_notional"]) < 1e-9
    e2 = entry_notional(1089.45, 7.67, existing_notional=e["total_notional"] * 2, use_evr=False)
    assert e2["room_notional"] == 0.0

    # ⭐E|r| 배수: [0,1] 이라 **절대 상한을 넘기지 않는다** + 낡거나 없으면 1.0(무효과)
    ev = entry_notional(1089.45, 7.67, use_evr=True)
    assert ev["room_notional"] <= ev["room_notional_precap"] + 1e-9, "배수가 상한을 키웠다"
    assert 0.0 <= ev["evr_mult"] <= 1.0, "배수가 [0,1] 밖"
    miss = evr_size_multiplier(state_path="/nonexistent/evr.json")
    assert miss["evr_mult"] == 1.0 and not miss["evr_ok"], "상태 없으면 무효과여야 한다"
    stale = evr_size_multiplier(max_lag_min=-1.0)
    assert stale["evr_mult"] == 1.0 and not stale["evr_ok"], "낡으면 무효과여야 한다"

    # 청산: 한도 안이면 0, 두 배면 절반을 닫아야 한다
    x = exit_fraction_required(1089.45, 7.67, current_notional=1.0)
    assert x["required_fraction"] == 0.0
    allowed = 1089.45 * policy_leverage(7.67)["leverage"]
    y = exit_fraction_required(1089.45, 7.67, current_notional=allowed * 2)
    assert abs(y["required_fraction"] - 0.5) < 1e-9, y
    z = exit_fraction_required(1089.45, 21.66, current_notional=allowed * 2)
    assert z["required_fraction"] > y["required_fraction"], "오래 들 생각이면 더 닫아야 한다"
    print("통과 — 생존/성장 최솟값 · 진입 여유 · 청산 최소비율")


if __name__ == "__main__":
    _self_check()
