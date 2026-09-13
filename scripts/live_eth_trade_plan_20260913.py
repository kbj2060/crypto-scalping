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

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.live_eth_risk_sizing_policy_20260913 import (  # noqa: E402
    HARD_CAP_X, recommended_tranches)
from scripts.live_manual_peg_entry_20260912 import (  # noqa: E402
    EXIT_TAKER_VOL_BPM, FALLBACK_SEC, exit_deadline_sec)

HOLD_CHOICES = (60, 120, 240, 480, 1440)
ROUND_TRIP_COST_BP = 5.88            # static 진입 + peg 청산 실측(2026-09-13)
# 과거 869일 5분봉의 스케일 없는 계수 (k_b = E|r_H|/atr_pct, k_s = SD(r_H)/atr_pct).
# 변동성 5분위·TRAIN→TEST 모두 ±15% 안에서 안정(연구 스크립트 assert). 원장과 무관하다.
K_HORIZON = {60: (3.518, 5.488), 120: (5.055, 7.838), 240: (7.367, 11.289),
             480: (10.862, 16.401), 1440: (20.691, 30.599)}
ACC_GRID = (0.52, 0.55, 0.58, 0.62, 0.66)
# ponytail: 단일 권고에 쓰는 기준 정확도는 고정 55%. 원장에서 뽑지 않는다. 화면 슬라이더로 바꾸려면 쿼리 인자 하나.
ACC_DEFAULT = 0.55
FILL_K, FILL_EXP = 52.0, 1.472       # 중앙 체결시간(초) ≈ 52 / vol^1.472
MAKER_BP, TAKER_BP = 2.0, 5.0
# ponytail: 재조정 무거래 밴드는 고정 25%. 비용(왕복 5.88bp) 대 파산감소로 최적화하면 더 낫다.
REBALANCE_BAND = 0.25


def growth_per_trade(L: float, move_bp: float, sd_bp: float, acc: float) -> float:
    """정확도 acc 인 사람이 이 지평을 L 배로 들 때의 건당 로그성장 근사."""
    return L * ((2 * acc - 1) * move_bp - ROUND_TRIP_COST_BP) / 1e4 - 0.5 * (L * sd_bp / 1e4) ** 2


def safe_mae(risk_table: dict, hold: int, side: str) -> float | None:
    cell = ((risk_table or {}).get(str(hold)) or {}).get(side.upper()) or {}
    m = cell.get("safe_mae_pct")
    return float(m) if m and m > 0 else None


def allowed_x(risk_table: dict, hold: int, side: str, cap_x: float) -> float | None:
    """이 보유시간의 허용 배수 = min(생존, 적용 상한). 모델 칸이 없으면 None."""
    m = safe_mae(risk_table, hold, side)
    return None if m is None else min(100.0 / m, cap_x, HARD_CAP_X)


def recommend_hold(risk_table: dict, side: str, cap_x: float, atr_pct: float | None,
                   acc: float = ACC_DEFAULT) -> dict:
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
        row = {"hold_min": H, "leverage": round(L, 2), "move_bp": round(b, 1), "sd_bp": round(sd, 1),
               "breakeven_acc": round(0.5 + ROUND_TRIP_COST_BP / (2 * b), 3),
               "growth": round(growth_per_trade(L, b, sd, acc), 5),
               "growth_by_acc": {str(x): round(growth_per_trade(L, b, sd, x), 5) for x in ACC_GRID}}
        rows.append(row)
        for x in ACC_GRID:
            g = row["growth_by_acc"][str(x)]
            if x not in best_by_acc or g > best_by_acc[x][1]:
                best_by_acc[x] = (H, g)
    if not rows:
        return {"available": False, "reason": "위험모델 없음"}
    best = max(rows, key=lambda r: r["growth"])
    return {"available": True, "recommended_min": best["hold_min"], "acc": acc, "table": rows,
            "best_by_acc": {str(x): h for x, (h, _) in best_by_acc.items()},
            "reason": (f"정확도 {int(100*acc)}% 가정 시 {best['hold_min']}분이 건당 로그성장 최대"
                       f"({best['growth']:+.4f}) · 손익분기 정확도 "
                       + " ".join(f"{r['hold_min']}분 {int(round(100*r['breakeven_acc']))}%" for r in rows))}


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
             atr_pct: float | None = None, hold_min: int | None = None) -> dict:
    """한 번에 넷: 보유시간 권고 · 크기(그 보유시간의 허용 배수) · 집행 · 분할."""
    hold = recommend_hold(risk_table, side, cap_x, atr_pct)
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
    h = recommend_hold(live, "LONG", 8.0, 0.000387)
    assert h["available"], h
    be = [r["breakeven_acc"] for r in h["table"]]
    assert be == sorted(be, reverse=True) and abs(be[0] - 0.716) < 0.01, be
    assert abs(h["table"][0]["move_bp"] - 13.6) < 0.2, h["table"][0]
    # 정확도가 오르면 최적 지평이 길어진다(비용 상각 > 분산 페널티)
    bba = h["best_by_acc"]
    assert bba["0.52"] <= bba["0.66"], bba
    # 격변기(atr_pct 3배)에는 같은 정확도에서 짧은 지평도 비용을 넘는다
    h2 = recommend_hold(live, "LONG", 8.0, 0.000387 * 3)
    assert h2["table"][0]["breakeven_acc"] < be[0]
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
    full = plan_now(side="LONG", equity=1000.0, existing_notional=99999.0, unrealized_pnl=0.0,
                    risk_table=live, vol_bpm=8.0, cap_x=8.0, atr_pct=0.000387)
    assert full["entry_split"]["room_notional"] == 0.0, "상한을 채웠으면 여유가 0 이어야 한다"
    p = plan_now(side="LONG", equity=0.0, existing_notional=0.0, unrealized_pnl=0.0,
                 risk_table={}, vol_bpm=None, cap_x=8.0)
    assert p["size"]["leverage"] is None and p["exit_ladder"]["budget_min"] == 0
    print("통과 26/26 — 보유시간 프런티어 · 집행 · 예산 사다리 · 플랜 조립")


if __name__ == "__main__":
    _self_check()
