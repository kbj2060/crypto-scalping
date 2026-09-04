#!/usr/bin/env python3
"""ETH V자반등 경제라벨 **섀도우 러너** -- 가상 체결만 기록. 주문을 내지 않는다.

## 왜 섀도우가 다음 단계인가

HOLDOUT(2026-04~08)이 1회 노출로 소진됐다. 이 풀 정의로는 재노출할 수 없으므로
**남은 검증 수단은 실시간 섀도우뿐**이다. 백테스트가 못 잡는 것들을 여기서 잡는다:

  · 라이브 피쳐 계산이 백테스트와 실제로 일치하는가(패리티)
  · 신호 빈도가 실측(13~16건/일)과 맞는가
  · 봉 지연·데이터 결측 시 거동
  · 라이브 배리어 판정이 백테스트와 실제로 일치하는가

⚠️**이 스크립트는 어떤 주문도 내지 않는다.** `core.binance_client`의 주문 함수를 import조차
하지 않는다. 가격 조회(공개 API)와 가상 원장 기록만 한다. 실주문 배선은 사용자의 명시적
결정이 필요한 별도 작업이다.

## 전략 규격 (규격서 그대로)

    임계값 p>=0.8158 (5시드 앙상블) · 진입 다음 봉 시가 · 손절 5.0×ATR
    무장 1.5×ATR · 트레일 0.1×ATR · 동시보유 5

근거: `docs/model_contracts/eth_v_rebound_econ_label_autotrade_spec_20260902.md`

## ⭐배리어 판정 컨벤션 (2026-09-03 수정)

청산 판정은 **완결 봉의 고가/저가**로 한다 -- 백테스트 `sim_exit`과 같은 컨벤션이다.
그 전에는 폴링한 마크가격 한 점만 봐서, 봉 안에서 스톱을 스치고 되돌아온 wick을 놓쳤다.
그 결과 원장 9건이 **전부 양수**(평균 +69bp)로 HOLDOUT 실측 +6.09bp의 10배가 나왔는데,
이는 좋은 소식이 아니라 손실 트레이드가 사라진 **계측 아티팩트**였다.
루프 주기를 60→300초로 늘리면서 봉당 샘플이 5개→1개가 되어 격차가 더 커졌었다.

⚠️이 수정으로 **원장의 앞 9건과 이후 기록은 계측 방식이 다르다.** `exit_basis` 필드로
구분한다(`bar_high_low`가 새 방식). 대조 분석 시 섞지 말 것.

## 산출

`data/live/v_rebound_econ_shadow_state.json` -- 가상 포지션/원장. 프로세스가 죽어도 복구된다.
백테스트 대조가 가능하도록 진입 확률·ATR·손절선을 전부 남긴다.

Usage:
    python scripts/live_eth_v_rebound_econ_shadow_runner_20260902.py           # 1회
    python scripts/live_eth_v_rebound_econ_shadow_runner_20260902.py --loop    # 상시(60초)
    python scripts/live_eth_v_rebound_econ_shadow_runner_20260902.py --report  # 원장 요약
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_SIG = importlib.import_module("live_eth_v_rebound_econ_autotrade_signal_20260902")
compute_signal = _SIG.compute_signal
BRACKET, MAX_CONCURRENT, SYMBOL = _SIG.BRACKET, _SIG.MAX_CONCURRENT, _SIG.SYMBOL

STATE = ROOT / "data/live/v_rebound_econ_shadow_state.json"
MARK_URL = "https://fapi.binance.com/fapi/v1/ticker/price"
COST_BP = 10.0
MAX_HOLD_BARS = 200             # 5분봉 200개 = 1000분. `sim_exit`의 FORWARD_BARS와 같은 단위
# 2026-09-02: 60 -> 300. 5분봉 신호를 60초마다 재채점하는 건 같은 봉을 4번 중복 채점하는
# 낭비였다(GPU 사용률 97% 원인). 300초 = 봉당 1회.
# ⭐2026-09-03부터 배리어 판정이 **봉 고가/저가** 기준이라 이 주기는 성과 계측에 영향이 없다
# (예전 마크가격 폴링 방식에서는 주기가 곧 계측 해상도였다). 신호 지연에만 영향을 준다.
LOOP_SECONDS = 300


def log(m: str) -> None:
    print(f"[vreb-shadow {datetime.now(timezone.utc):%m-%d %H:%M:%S}] {m}", flush=True)


def load_state() -> dict[str, Any]:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:                                     # noqa: BLE001
            log("⚠️상태 파싱 실패 -- 새로 시작")
    return {"positions": [], "ledger": [], "consec_loss": 0, "started_utc":
            datetime.now(timezone.utc).isoformat(), "version": 1}


def save_state(s: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2, default=str))
    tmp.replace(STATE)                                        # 원자적 교체


def mark_price() -> float | None:
    try:
        r = requests.get(MARK_URL, params={"symbol": SYMBOL}, timeout=10)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:                                         # noqa: BLE001
        return None


def manage(s: dict[str, Any], bars: list[dict]) -> None:
    """⭐**봉 고가/저가**로 배리어를 판정한다 -- 백테스트 `sim_exit`과 같은 컨벤션.

    2026-09-03 수정 전에는 폴링한 마크가격 한 점(`px`)만 봤다. 그러면 봉 안에서 스톱을
    스치고 되돌아온 wick을 놓쳐 **손실 트레이드가 통째로 사라진다** -- 실제로 원장 9건이
    전부 양수(평균 +69bp)로 HOLDOUT 실측(+6.09bp)의 10배가 나왔고, 이는 좋은 소식이 아니라
    계측 아티팩트였다. 게다가 루프 주기를 60→300초로 늘리면서 봉당 샘플이 5개→1개가 되어
    격차가 더 커졌다.

    `compute_signal()`이 돌려주는 `bars`는 `_fetch_klines`가 형성 중인 봉을 버린 뒤의
    **완결 봉**이므로 lookahead가 아니다(CLAUDE.md의 h48qual/zig075 `evaluate_exit` 컨벤션과
    같은 논리: resting TP/SL 주문은 닿는 즉시 체결되고, 이미 확정된 봉만 쓴다).

    ⚠️`sim_exit`의 **순서를 그대로** 따른다 -- 봉마다 (1)불리한 쪽(롱=저가)으로 스톱 판정,
    (2)그다음 유리한 쪽으로 best 갱신, (3)무장, (4)트레일 조임. 순서를 바꾸면 같은 봉에서
    스톱을 맞았어야 할 트레이드가 살아남는다(낙관 편향).

    각 포지션은 `last_bar_utc` **이후**의 봉만 처리한다. 진입 시 `last_bar_utc`는 진입 직전의
    마지막 완결 봉이므로, 진입이 일어난 봉부터 평가된다(규격의 "다음 봉 시가" 진입과 정합).
    """
    if not bars:
        log("⚠️봉 데이터 없음 -- 포지션 갱신 건너뜀")
        return
    keep = []
    for p in s["positions"]:
        sgn = 1.0 if p["side"] == "long" else -1.0
        a = p["atr"]
        last = p.get("last_bar_utc")
        todo = [b for b in bars if last is None or b["timestamp_utc"] > last]
        closed = False
        for b in todo:
            adv = b["low"] if sgn > 0 else b["high"]          # (1) 불리한 쪽 먼저
            if (adv <= p["stop"]) if sgn > 0 else (adv >= p["stop"]):
                _close(s, p, p["stop"], "stop", b["timestamp_utc"])
                closed = True
                break
            fav = b["high"] if sgn > 0 else b["low"]          # (2) 유리한 쪽으로 best
            if sgn * (fav - p["best"]) > 0:
                p["best"] = fav
            if not p["armed"] and sgn * (p["best"] - p["entry"]) >= BRACKET["arm_atr"] * a:
                p["armed"] = True                             # (3) 무장
                log(f"  무장 {p['side']} best={p['best']:.2f} @{b['timestamp_utc'][:16]}")
            if p["armed"]:                                    # (4) 트레일(한 방향으로만)
                ns = p["best"] - sgn * BRACKET["trail_atr"] * a
                if sgn * (ns - p["stop"]) > 0:
                    p["stop"] = ns
            p["bars_held"] = int(p.get("bars_held", 0)) + 1
            p["last_bar_utc"] = b["timestamp_utc"]
            if p["bars_held"] >= MAX_HOLD_BARS:               # 시간청산은 그 봉 종가(sim_exit 동일)
                _close(s, p, b["close"], "timeout", b["timestamp_utc"])
                closed = True
                break
        if not closed:
            keep.append(p)
    s["positions"] = keep


def _close(s: dict[str, Any], p: dict[str, Any], exit_px: float, reason: str,
           bar_utc: str) -> None:
    sgn = 1.0 if p["side"] == "long" else -1.0
    pnl = sgn * (exit_px - p["entry"]) / p["entry"] * 1e4 - COST_BP
    s["ledger"].append({"entry_utc": p["entry_utc"], "exit_utc": bar_utc,
                        "recorded_utc": datetime.now(timezone.utc).isoformat(),
                        "side": p["side"], "entry": p["entry"], "exit": exit_px,
                        "atr": p["atr"], "proba": p["proba"], "pnl_bp": round(pnl, 2),
                        "bars_held": int(p.get("bars_held", 0)), "reason": reason,
                        # 2026-09-04: `reason`만으로는 "stop"이 초기 손절인지 무장 후 트레일링
                        # 익절인지 구분되지 않아, 대시보드가 이익 청산까지 "손절선 도달"로 찍고
                        # 있었다(사용자 신고 -- 당시 원장 17건 중 12건이 오표기). BRACKET이
                        # arm_atr=1.5 / trail_atr=0.1이라 무장 후 스톱은 항상 진입가 위이므로
                        # 두 경우는 원래 완전히 다른 사건이다. 이 필드가 그 구분을 확정한다
                        # (이전 행에는 없으므로 대시보드가 가격이동 부호로 보완한다).
                        "armed": bool(p.get("armed", False)),
                        "exit_basis": "bar_high_low"})
    s["consec_loss"] = 0 if pnl > 0 else s["consec_loss"] + 1
    log(f"  청산 {p['side']} {pnl:+.2f}bp ({reason}, {p.get('bars_held', 0)}봉) "
        f"연속손실 {s['consec_loss']}")


def enter(s: dict[str, Any], out: dict[str, Any], px: float) -> None:
    """가상 진입. `last_bar_utc`를 진입 직전 마지막 완결 봉으로 잡아, 진입이 일어난 봉부터
    배리어 평가가 시작되게 한다."""
    slots = MAX_CONCURRENT - len(s["positions"])
    if slots <= 0:
        return
    for call in sorted(out["calls"], key=lambda x: -x["proba"])[:slots]:
        if any(p["entry_utc"] == call["timestamp_utc"] and p["side"] == call["side"]
               for p in s["positions"]):
            continue                                          # 같은 봉 중복 방지
        if any(t["entry_utc"] == call["timestamp_utc"] and t["side"] == call["side"]
               for t in s["ledger"][-50:]):
            continue                                          # 이미 청산된 봉 재진입 방지
        sgn = 1.0 if call["side"] == "long" else -1.0
        a = float(call["atr"])
        s["positions"].append({"entry_utc": call["timestamp_utc"], "side": call["side"],
                               "entry": px, "atr": a, "stop": px - sgn * BRACKET["sl_atr"] * a,
                               "best": px, "armed": False, "bars_held": 0,
                               "last_bar_utc": out.get("last_closed_bar_utc"),
                               "proba": call["proba"],
                               "opened_utc": datetime.now(timezone.utc).isoformat()})
        log(f"  [가상진입] {call['side']} @{px:.2f} p={call['proba']:.4f} "
            f"atr={a:.2f} stop={px - sgn*BRACKET['sl_atr']*a:.2f}")


def report(s: dict[str, Any]) -> None:
    led = s["ledger"]
    if not led:
        log("원장 비어있음"); return
    pnl = np.array([t["pnl_bp"] for t in led], float)
    w = pnl > 0
    eq = np.cumsum(pnl)
    dd = (eq - np.maximum.accumulate(eq)).min()
    start = pd.Timestamp(s.get("started_utc"))
    days = max((pd.Timestamp.now(tz="UTC") - start).total_seconds() / 86400, 1e-9)
    log("=== 섀도우 원장 요약 ===")
    log(f"  가동 {days:.2f}일  체결 {len(pnl):,}건 ({len(pnl)/days:.2f}건/일)")
    log(f"  기대값 {pnl.mean():+.2f}bp  누적 {pnl.sum():+.0f}bp  승률 {w.mean()*100:.1f}%")
    if w.any() and (~w).any():
        log(f"  평균이익 {pnl[w].mean():+.1f}bp  평균손실 {pnl[~w].mean():+.1f}bp  "
            f"손익비 {pnl[w].mean()/-pnl[~w].mean():.3f}")
    log(f"  최대DD {dd:+.0f}bp  연속손실(현재) {s['consec_loss']}  포지션 {len(s['positions'])}")
    nl = sum(1 for t in led if t["side"] == "long")
    log(f"  롱 {nl}건 ({nl/len(led)*100:.1f}%) / 숏 {len(led)-nl}건")
    log("  [백테스트 대조] HOLDOUT 실측: 13.18건/일  기대값 +6.09bp  승률 78.0%  손익비 0.346")


def stamp_config(s: dict[str, Any]) -> None:
    """설정이 바뀐 채로 원장을 이어 쓰면 두 전략이 조용히 섞인다. 변경 시점을 기록한다."""
    cur = f"{_SIG.ENSEMBLE_SEEDS}seed@{_SIG.PROBA_THRESHOLD}"
    hist = s.setdefault("config_history", [])
    if not hist or hist[-1]["config"] != cur:
        hist.append({"config": cur, "since_utc": datetime.now(timezone.utc).isoformat(),
                     "ledger_len_at_change": len(s.get("ledger", []))})
        log(f"⚠️설정 변경 기록: {cur} (원장 {len(s.get('ledger', []))}건 이후부터 적용)")


def cycle(s: dict[str, Any]) -> None:
    out = compute_signal()
    px = mark_price()
    if px is None:
        log("⚠️마크가격 실패 -- 건너뜀"); return
    manage(s, out.get("bars") or [])
    if out.get("warmed_up") and not out.get("error"):
        enter(s, out, px)
    else:
        log(f"신호 이상: {out.get('error')} -- 진입 보류")
    tot = sum(t["pnl_bp"] for t in s["ledger"])
    log(f"px={px:.2f}  포지션 {len(s['positions'])}/{MAX_CONCURRENT}  "
        f"원장 {len(s['ledger'])}건 {tot:+.0f}bp  신호 {len(out.get('calls', []))}건  "
        f"최종봉 {str(out.get('last_closed_bar_utc'))[:16]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    s = load_state()
    if args.report:
        report(s); return 0
    log(f"⚠️섀도우 모드 -- 주문을 내지 않습니다. 임계값 {_SIG.PROBA_THRESHOLD} "
        f"시드 {_SIG.ENSEMBLE_SEEDS} 한도 {MAX_CONCURRENT} 주기 {LOOP_SECONDS}초")
    stamp_config(s)
    if not args.loop:
        cycle(s); save_state(s); report(s); return 0
    while True:
        try:
            cycle(s); save_state(s)
        except KeyboardInterrupt:
            save_state(s); log("중단"); return 0
        except Exception as e:                                # noqa: BLE001
            log(f"⚠️사이클 예외: {type(e).__name__}: {e}")
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
