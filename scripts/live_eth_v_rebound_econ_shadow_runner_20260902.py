#!/usr/bin/env python3
"""ETH V자반등 경제라벨 **섀도우 러너** -- 가상 체결만 기록. 주문을 내지 않는다.

## 왜 섀도우가 다음 단계인가

HOLDOUT(2026-04~08)이 1회 노출로 소진됐다. 이 풀 정의로는 재노출할 수 없으므로
**남은 검증 수단은 실시간 섀도우뿐**이다. 백테스트가 못 잡는 것들을 여기서 잡는다:

  · 라이브 피쳐 계산이 백테스트와 실제로 일치하는가(패리티)
  · 신호 빈도가 실측(13~16건/일)과 맞는가
  · 봉 지연·데이터 결측 시 거동
  · 실시간 마크가격 기준 트레일링이 백테스트 종가 기준과 얼마나 다른가

⚠️**이 스크립트는 어떤 주문도 내지 않는다.** `core.binance_client`의 주문 함수를 import조차
하지 않는다. 가격 조회(공개 API)와 가상 원장 기록만 한다. 실주문 배선은 사용자의 명시적
결정이 필요한 별도 작업이다.

## 전략 규격 (규격서 그대로)

    임계값 p>=0.8158 (5시드 앙상블) · 진입 다음 봉 시가 · 손절 5.0×ATR
    무장 1.5×ATR · 트레일 0.1×ATR · 동시보유 5

근거: `docs/model_contracts/eth_v_rebound_econ_label_autotrade_spec_20260902.md`

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
MAX_HOLD_BARS = 200
LOOP_SECONDS = 60


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


def manage(s: dict[str, Any], px: float) -> None:
    """백테스트 sim_exit과 **동일한 순서/공식**으로 가상 포지션을 갱신한다."""
    keep = []
    for p in s["positions"]:
        sgn = 1.0 if p["side"] == "long" else -1.0
        a = p["atr"]
        if sgn * (px - p["best"]) > 0:
            p["best"] = px
        if not p["armed"] and sgn * (p["best"] - p["entry"]) >= BRACKET["arm_atr"] * a:
            p["armed"] = True
            log(f"  무장 {p['side']} best={p['best']:.2f}")
        if p["armed"]:
            ns = p["best"] - sgn * BRACKET["trail_atr"] * a
            if sgn * (ns - p["stop"]) > 0:
                p["stop"] = ns
        p["ticks"] = int(p.get("ticks", 0)) + 1
        hit = (px <= p["stop"]) if sgn > 0 else (px >= p["stop"])
        # 봉 기준 보유한도: 5분봉 200개 = 1000분. 틱은 60초라 근사로 환산
        timeout = p["ticks"] >= MAX_HOLD_BARS * 5
        if hit or timeout:
            exit_px = p["stop"] if hit else px
            pnl = sgn * (exit_px - p["entry"]) / p["entry"] * 1e4 - COST_BP
            s["ledger"].append({"entry_utc": p["entry_utc"],
                                "exit_utc": datetime.now(timezone.utc).isoformat(),
                                "side": p["side"], "entry": p["entry"], "exit": exit_px,
                                "atr": a, "proba": p["proba"], "pnl_bp": round(pnl, 2),
                                "reason": "stop" if hit else "timeout"})
            s["consec_loss"] = 0 if pnl > 0 else s["consec_loss"] + 1
            log(f"  청산 {p['side']} {pnl:+.2f}bp ({'stop' if hit else 'timeout'}) "
                f"연속손실 {s['consec_loss']}")
        else:
            keep.append(p)
    s["positions"] = keep


def enter(s: dict[str, Any], out: dict[str, Any], px: float) -> None:
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
                               "best": px, "armed": False, "ticks": 0,
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


def cycle(s: dict[str, Any]) -> None:
    out = compute_signal()
    px = mark_price()
    if px is None:
        log("⚠️마크가격 실패 -- 건너뜀"); return
    manage(s, px)
    if out.get("warmed_up") and not out.get("error"):
        enter(s, out, px)
    else:
        log(f"신호 이상: {out.get('error')} -- 진입 보류")
    tot = sum(t["pnl_bp"] for t in s["ledger"])
    log(f"px={px:.2f}  포지션 {len(s['positions'])}/{MAX_CONCURRENT}  "
        f"원장 {len(s['ledger'])}건 {tot:+.0f}bp  신호 {len(out.get('calls', []))}건")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    s = load_state()
    if args.report:
        report(s); return 0
    log(f"⚠️섀도우 모드 -- 주문을 내지 않습니다. 임계값 {_SIG.PROBA_THRESHOLD} 한도 {MAX_CONCURRENT}")
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
