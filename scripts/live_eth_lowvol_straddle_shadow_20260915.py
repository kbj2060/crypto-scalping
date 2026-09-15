#!/usr/bin/env python3
"""**저변동 스트래들 섀도우** — 판정 불가를 표본으로 푸는 유일한 방법 (2026-09-15).

2026-09-14 연구 결론: 익절 +10% / 손절 −3% 양측 동시진입을 **예측변동성 하위 20%** 에서만 하면
5창(2022~23·TRAIN·VAL·OOS·TEST) 전부 쌍당 손익이 양수였다. 그런데 창당 독립 쌍이 **12~76개**뿐이라
블록 t 가 **0.86** 이다 — 「번다」도 「안 번다」도 말할 수 없는 상태다.
연 60~130쌍이 쌓이면 1년에 t≈1.8 까지 온다. **이 워커는 그 표본을 사는 장치다.**

🔴**주문을 내지 않는다.** 읽기 전용이고 봇·대시보드와 분리돼 있다. 죽어도 아무 영향 없다.

## 규칙 (사전등록 — 바꾸면 그날로 표본이 리셋이다)
  진입   포지션(쌍) 없고 게이트 통과면 그 봉 종가에 롱·숏 **동시** 진입
  게이트 배포 사이징 모델의 전방 4시간 예측변동성이 **평소의 0.749배 미만**
         (= 학습창 비율 분위 20%. 서버 실측. 아티팩트가 기계마다 달라 **절대값이 아니라 비율**로
          자른다 — 2026-09-14 대시보드 칩에서 같은 함정을 밟았다)
  배리어 익절 +10% · 손절 −3% · **봉내 고저** 기준(라이브 `evaluate_exit` 컨벤션)
         같은 봉에서 양쪽 다 닿으면 **손절 우선**(라이브는 봉 안 순서를 모른다)
  비용   진입 2.95bp(지정가) · 익절 2.93bp(peg) · 손절 5.0bp(테이커) + 슬리피지 중앙 14bp · 펀딩
  동시   **한 번에 한 쌍**. 두 다리가 모두 끝나야 다음 쌍을 연다(순차 — 겹침 평균은 굴릴 수 있는
         값이 아니다: 2026-09-14 스트래들 VAL 겹침 +113.9bp → 순차 −34.7bp)

## 저장
  원장 `data/live/lowvol_straddle_shadow.jsonl` — 쌍 하나가 한 줄(append only).
  🔴duckdb 를 쓰지 않는다: 같은 날 maker_fill_shadow 를 읽으려다 워커 락에 막혔다. 쌍이 연 100개
  수준이라 JSONL 이면 충분하고, 누가 언제든 읽을 수 있다.
  상태 `data/live/lowvol_straddle_shadow_state.json` — 열린 쌍 + 마지막 점검 시각.

`--selftest` 는 연구 하네스(`research_eth_direction_barrier_label_20260914.leg`)와 **같은 봉으로
같은 숫자**가 나오는지 확인한다. 이게 「섀도우가 연구와 다른 전략을 돌고 있다」를 막는 유일한 장치다.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

SYMBOL = os.getenv("SS_SYMBOL", "ETHUSDT")
LEDGER = Path(os.getenv("SS_LEDGER", str(ROOT / "data/live/lowvol_straddle_shadow.jsonl")))
STATE = Path(os.getenv("SS_STATE", str(ROOT / "data/live/lowvol_straddle_shadow_state.json")))
SIZING_STATE = ROOT / "data/live/eth_position_sizing_state.json"
PERIOD_S = 300
KLINES = "https://fapi.binance.com/fapi/v1/klines"

UP, DOWN = 0.10, 0.03            # 익절 +10% · 손절 −3%
GATE_RATIO = 0.749               # 예측변동성 / 평소 가 이 미만일 때만 진입(학습창 20% 분위)
ENTRY_BP, PEG_EXIT_BP, TAKER_EXIT_BP = 2.95, 2.93, 5.0
STOP_SLIP_BP = 14.0
FUNDING_BP_8H = 0.52
MAX_HOLD_BARS = 17_280           # 60일. 연구와 같은 계산 상한

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("straddle-shadow")


def klines(limit: int = 1500, start_ms: int | None = None) -> list[list]:
    p = {"symbol": SYMBOL, "interval": "5m", "limit": limit}
    if start_ms is not None:
        p["startTime"] = start_ms
    r = requests.get(KLINES, params=p, timeout=20)
    r.raise_for_status()
    return r.json()


def gate_ratio() -> float | None:
    """배포 사이징 워커가 남긴 예측변동성 / 그 아티팩트 자신의 기준값. **모델을 돌리지 않는다.**"""
    try:
        d = json.loads(SIZING_STATE.read_text())
    except (OSError, ValueError):
        return None
    sm = d.get("sizing_model") or {}
    pred, ref = sm.get("pred_vol"), sm.get("ref_pred")
    if not sm.get("used") or not (isinstance(pred, (int, float)) and isinstance(ref, (int, float))):
        return None
    return float(pred) / float(ref) if ref > 0 else None


def resolve_leg(bars: list[list], entry: float, side: int) -> dict | None:
    """봉 목록(진입 **다음** 봉부터)에서 먼저 닿는 배리어. side 1=롱 2=숏. 연구 규약과 동일."""
    up_px = entry * (1 + (UP if side == 1 else DOWN))
    dn_px = entry * (1 - (DOWN if side == 1 else UP))
    win_up = side == 1                       # 롱은 위가 익절, 숏은 아래가 익절
    for k, b in enumerate(bars):
        hi, lo = float(b[2]), float(b[3])
        a, c = hi >= up_px, lo <= dn_px
        if not (a or c):
            continue
        # 같은 봉에서 양쪽 다 닿으면 **손절 우선**(라이브는 봉 안 순서를 모른다)
        win = False if (a and c) else ((a and win_up) or (c and not win_up))
        held_min = 5 * (k + 1)
        if win:
            fill, cost = UP, ENTRY_BP + PEG_EXIT_BP
        else:
            fill, cost = -(DOWN + STOP_SLIP_BP / 1e4), ENTRY_BP + TAKER_EXIT_BP
        cost += (FUNDING_BP_8H if side == 1 else -FUNDING_BP_8H) * (held_min / 480.0)
        return {"bars": k + 1, "win": bool(win), "fill": fill, "cost_bp": cost,
                "r_bp": fill * 1e4 - cost, "ts": int(b[0])}
    return None


def check(state: dict) -> dict:
    kl = klines(1500)
    last_closed = kl[-2]                     # 마지막 봉은 진행 중 — 쓰지 않는다
    close = float(last_closed[4]); ts = int(last_closed[0])
    open_pair = state.get("open")
    if open_pair is None:
        r = gate_ratio()
        if r is None:
            log.info("게이트 값 없음(사이징 워커 상태) — 대기"); return state
        if r >= GATE_RATIO:
            log.info("게이트 미통과 ratio %.3f ≥ %.3f — 대기", r, GATE_RATIO); return state
        state["open"] = {"entry_ts": ts, "entry": close, "gate_ratio": r,
                         "opened_utc": datetime.now(timezone.utc).isoformat()}
        log.info("⭐쌍 개시 ts=%s 진입가 %.2f · 게이트 %.3f", ts, close, r)
        return state
    op = open_pair
    bars = [b for b in klines(1500, start_ms=op["entry_ts"] + 1) if int(b[0]) > op["entry_ts"]]
    if not bars:
        return state
    L = resolve_leg(bars, op["entry"], 1); S = resolve_leg(bars, op["entry"], 2)
    if L is None or S is None:
        if len(bars) >= MAX_HOLD_BARS:
            log.warning("60일 미해결 — 쌍 폐기"); state["open"] = None
        log.info("보유 중 %d봉 · 롱 %s · 숏 %s", len(bars),
                 "해결" if L else "진행", "해결" if S else "진행")
        return state
    m_bp = 0.5 * (L["r_bp"] + S["r_bp"])
    row = {"entry_ts": op["entry_ts"], "entry": op["entry"], "gate_ratio": op["gate_ratio"],
           "opened_utc": op["opened_utc"], "closed_utc": datetime.now(timezone.utc).isoformat(),
           "long": L, "short": S, "pair_bp": m_bp,
           "hold_bars": max(L["bars"], S["bars"]), "hold_h": max(L["bars"], S["bars"]) * 5 / 60}
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("✅쌍 종료 %+.1fbp · %.1fh · 롱 %s / 숏 %s", m_bp, row["hold_h"],
             "익절" if L["win"] else "손절", "익절" if S["win"] else "손절")
    state["open"] = None
    return state


def selftest() -> int:
    """연구 하네스와 **같은 봉으로 같은 숫자**가 나오는지. 다르면 다른 전략을 돌고 있는 것이다."""
    import research_eth_direction_barrier_label_20260914 as B
    B.FUNDING = True
    kl_csv = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
    import pandas as pd
    d = pd.read_csv(kl_csv, usecols=["timestamp", "high", "low", "close"]).tail(60_000).reset_index(drop=True)
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    rng = np.random.default_rng(20260915)
    n = bad = 0
    for i in rng.integers(100, len(c) - 20_000, size=40):
        i = int(i)
        for side in (1, 2):
            ref = B.leg(c, hi, lo, i, side, UP, DOWN)
            bars = [[0, 0, hi[j], lo[j], c[j]] for j in range(i + 1, min(len(c), i + 1 + MAX_HOLD_BARS))]
            got = resolve_leg(bars, c[i], side)
            if ref is None or got is None:
                continue
            n += 1
            if abs(ref[0] * 1e4 - got["r_bp"]) > 1e-6 or ref[1] != got["bars"]:
                bad += 1
                log.error("불일치 i=%d side=%d · 연구 %.4fbp/%d봉 vs 섀도우 %.4f/%d",
                          i, side, ref[0] * 1e4, ref[1], got["r_bp"], got["bars"])
    assert n >= 40 and bad == 0, f"비교 {n}건 중 불일치 {bad}"
    print(f"자체점검 통과: 연구 하네스와 {n}건 전부 1e-6 이내 일치(손익·해결봉)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    log.info("저변동 스트래들 섀도우 — 익절 +%.0f%% / 손절 −%.0f%% · 게이트 ratio<%.3f · 주기 %ds "
             "· 주문 없음", UP * 100, DOWN * 100, GATE_RATIO, PERIOD_S)
    state = json.loads(STATE.read_text()) if STATE.exists() else {"open": None}
    while True:
        try:
            state = check(state)
            state["last_check_utc"] = datetime.now(timezone.utc).isoformat()
            STATE.parent.mkdir(parents=True, exist_ok=True)
            tmp = STATE.with_suffix(".tmp"); tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
            tmp.replace(STATE)
        except Exception as exc:
            log.warning("점검 실패 %s: %s", type(exc).__name__, exc)
        if a.once:
            return 0
        time.sleep(PERIOD_S - (time.time() % PERIOD_S))


if __name__ == "__main__":
    raise SystemExit(main())
