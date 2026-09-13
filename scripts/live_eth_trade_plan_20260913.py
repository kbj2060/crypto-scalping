"""**지금 상황의 트레이드 플랜** — 집행·크기·보유시간·분할을 한 번에 (2026-09-13).

사용자: *"현재 상황에 내가 진입 및 청산을 하려는데 어떻게 집행·크기·보유시간을 가져가는 게
좋을지 예측 및 분석하는 모델 … 분할 매매 및 분할 청산을 전략적으로"*.

새로 계산하는 건 둘뿐이고 나머지는 기존 함수를 조립한다.
  ① **보유시간 권고** = 보유시간별 로그성장 g(H) = L(H)·(μ_H−비용) − ½(L(H)·σ_H)² 의 최대.
     L(H) 는 생존 모델(보유시간 조건부 MAE 분위, 매 5분 갱신)의 허용 배수이고,
     μ_H·σ_H 는 68왕복의 **방향·시점 고정 + 고정 H 청산** 반사실
     (research_hold_time_counterfactual_68trips_20260913). 실측: 60분만 t=2.29 로 유의,
     길수록 σ 가 커져 성장이 떨어지고 1일은 음수. 사용자 실제 청산(중앙 67분)은 어떤
     고정 시계보다 낫다(21.25bp, t=3.70) -- 그래서 시계는 **크기와 예산**을 정하지
     «언제 나가라»를 정하지 않는다.
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

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.live_eth_risk_sizing_policy_20260913 import (  # noqa: E402
    HARD_CAP_X, recommended_tranches)
from scripts.live_manual_peg_entry_20260912 import (  # noqa: E402
    EXIT_TAKER_VOL_BPM, FALLBACK_SEC, exit_deadline_sec)

HOLD_CHOICES = (60, 120, 240, 480, 1440)
ROUND_TRIP_COST_BP = 5.88            # static 진입 + peg 청산 실측(2026-09-13)
# 68왕복 고정-H 반사실, 비용 차감 후 (μ bp, σ bp). 원장이 갱신되면 연구 스크립트를 다시 돌려 바꾼다.
EDGE_BY_HOLD = {60: (11.35, 40.79), 120: (7.25, 55.95), 240: (8.79, 105.36),
                480: (15.55, 140.93), 1440: (-3.55, 195.62)}
EDGE_ACTUAL_BP, EDGE_ACTUAL_HOLD_MED = 21.25, 67
FILL_K, FILL_EXP = 52.0, 1.472       # 중앙 체결시간(초) ≈ 52 / vol^1.472
MAKER_BP, TAKER_BP = 2.0, 5.0
# ponytail: 재조정 무거래 밴드는 고정 25%. 비용(왕복 5.88bp) 대 파산감소로 최적화하면 더 낫다.
REBALANCE_BAND = 0.25


def growth_per_trade(L: float, mu_bp: float, sd_bp: float) -> float:
    """건당 로그성장 근사. μ 는 비용 차감 후."""
    r, s = mu_bp / 1e4, sd_bp / 1e4
    return L * r - 0.5 * (L * s) ** 2


def safe_mae(risk_table: dict, hold: int, side: str) -> float | None:
    cell = ((risk_table or {}).get(str(hold)) or {}).get(side.upper()) or {}
    m = cell.get("safe_mae_pct")
    return float(m) if m and m > 0 else None


def allowed_x(risk_table: dict, hold: int, side: str, cap_x: float) -> float | None:
    """이 보유시간의 허용 배수 = min(생존, 적용 상한). 모델 칸이 없으면 None."""
    m = safe_mae(risk_table, hold, side)
    return None if m is None else min(100.0 / m, cap_x, HARD_CAP_X)


def recommend_hold(risk_table: dict, side: str, cap_x: float) -> dict:
    """보유시간별 성장표와 최대점. 동률이면 짧은 쪽(꼬리가 위험의 대부분이다)."""
    rows, best = [], None
    for H in HOLD_CHOICES:
        L = allowed_x(risk_table, H, side, cap_x)
        if L is None:
            continue
        mu, sd = EDGE_BY_HOLD[H]
        g = growth_per_trade(L, mu, sd)
        rows.append({"hold_min": H, "leverage": round(L, 2), "edge_bp": mu, "sd_bp": sd,
                     "growth": round(g, 5)})
        if best is None or g > best["growth"]:
            best = rows[-1]
    if best is None:
        return {"available": False, "reason": "위험모델 없음"}
    return {"available": True, "recommended_min": best["hold_min"], "table": rows,
            "reason": (f"{best['hold_min']}분이 건당 로그성장 최대({best['growth']:.4f}) · "
                       f"실제 청산(중앙 {EDGE_ACTUAL_HOLD_MED}분)은 고정 시계보다 낫습니다"
                       f"({EDGE_ACTUAL_BP}bp) — 시계는 크기와 예산을 정합니다")}


def expected_fill_sec(vol_bpm: float | None) -> float | None:
    return None if not vol_bpm or vol_bpm <= 0 else FILL_K / vol_bpm ** FILL_EXP


def execution_plan(vol_bpm: float | None) -> dict:
    fill = expected_fill_sec(vol_bpm)
    market = vol_bpm is not None and vol_bpm >= EXIT_TAKER_VOL_BPM
    return {
        "vol_bpm": round(vol_bpm, 2) if vol_bpm is not None else None,
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


def plan_now(*, side: str, equity: float, existing_notional: float, unrealized_pnl: float,
             risk_table: dict, vol_bpm: float | None, cap_x: float,
             hold_min: int | None = None) -> dict:
    """한 번에 넷: 보유시간 권고 · 크기(그 보유시간의 허용 배수) · 집행 · 분할."""
    hold = recommend_hold(risk_table, side, cap_x)
    H = hold_min or (hold.get("recommended_min") if hold.get("available") else max(HOLD_CHOICES))
    L = allowed_x(risk_table, H, side, cap_x)
    room = max(0.0, equity * L - existing_notional) if L else 0.0
    m = safe_mae(risk_table, H, side)
    return {
        "side": side, "hold": hold, "hold_min": H,
        "size": {"leverage": round(L, 2) if L else None, "safe_mae_pct": m,
                 "total_notional": round(equity * L, 2) if L else None,
                 "room_notional": round(room, 2)},
        "execution": execution_plan(vol_bpm),
        "entry_split": {
            **(recommended_tranches(m, min(L, cap_x), H) if m and L else
               {"tranches": 1, "reason": "위험모델 없음 — 일괄", "spread_min": 0}),
            # 추가 칸은 «순행 중»에만. 역행 중 추가(물타기)는 크기 매칭 후 24/24 전패였다.
            "add_allowed": bool(existing_notional <= 0 or (unrealized_pnl >= 0 and room > 0)),
            "rule": "첫 칸 뒤 추가는 평가손익 ≥ 0 이고 상한 여유가 남았을 때만 (역행 중 추가 금지)"},
        "exit_ladder": hold_budget(equity, existing_notional, risk_table, side, cap_x),
    }


def _self_check() -> None:
    live = {"60": {"LONG": {"safe_mae_pct": 1.821}, "SHORT": {"safe_mae_pct": 1.592}},
            "120": {"LONG": {"safe_mae_pct": 2.365}, "SHORT": {"safe_mae_pct": 2.173}},
            "240": {"LONG": {"safe_mae_pct": 3.36}, "SHORT": {"safe_mae_pct": 3.168}},
            "480": {"LONG": {"safe_mae_pct": 6.967}, "SHORT": {"safe_mae_pct": 6.117}},
            "1440": {"LONG": {"safe_mae_pct": 23.236}, "SHORT": {"safe_mae_pct": 21.643}}}
    # 성장: 상한 8배에서는 60분이 최대(엣지 t 유일 유의), 1일은 음수
    h = recommend_hold(live, "LONG", 8.0)
    assert h["available"] and h["recommended_min"] == 60, h
    g = {r["hold_min"]: r["growth"] for r in h["table"]}
    assert g[1440] < 0 < g[60] and g[60] > g[480], g
    assert allowed_x(live, 60, "LONG", 8.0) == 8.0 and allowed_x(live, 1440, "LONG", 8.0) < 5, "상한·생존 최솟값"
    assert recommend_hold({}, "LONG", 8.0)["available"] is False
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
    # 플랜: 역행 중엔 추가 금지, 순행이고 여유 있으면 허용, 포지션 없으면 첫 칸은 허용
    p = plan_now(side="LONG", equity=1000.0, existing_notional=4000.0, unrealized_pnl=-30.0,
                 risk_table=live, vol_bpm=8.0, cap_x=8.0)
    assert p["hold_min"] == 60 and p["entry_split"]["add_allowed"] is False, p["entry_split"]
    assert p["size"]["room_notional"] == 4000.0 and p["execution"]["exit"]["mode"] == "peg_repeg"
    p = plan_now(side="LONG", equity=1000.0, existing_notional=4000.0, unrealized_pnl=+30.0,
                 risk_table=live, vol_bpm=8.0, cap_x=8.0)
    assert p["entry_split"]["add_allowed"] is True
    p = plan_now(side="SHORT", equity=1000.0, existing_notional=0.0, unrealized_pnl=0.0,
                 risk_table=live, vol_bpm=None, cap_x=8.0, hold_min=1440)
    assert p["hold_min"] == 1440 and p["entry_split"]["add_allowed"] is True
    assert p["size"]["leverage"] < 5 and p["entry_split"]["tranches"] == 1, p["size"]
    p = plan_now(side="LONG", equity=0.0, existing_notional=0.0, unrealized_pnl=0.0,
                 risk_table={}, vol_bpm=None, cap_x=8.0)
    assert p["size"]["leverage"] is None and p["exit_ladder"]["budget_min"] == 0
    print("통과 24/24 — 보유시간 성장표 · 집행 · 예산 사다리 · 플랜 조립")


if __name__ == "__main__":
    _self_check()
