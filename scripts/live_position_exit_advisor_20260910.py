#!/usr/bin/env python3
"""포지션 청산 감시자 워커 (2026-09-10) -- 사용자가 연 실계좌 포지션을 5분봉마다 보고
**지금 익절·부분익절·손절·감축해야 하는가**를 판정한다. 주문은 내지 않는다.

설계: docs/eth_position_exit_advisor_research_20260910.md
- 학습 모델이 아니라 규칙 상태기계다(학습 청산 축은 3모집단에서 종결, AUC 0.60 = 시계).
- 판정에 쓰는 신호는 셋: 극점 탐지기(반대 측면 콜 = 익절, 진입 봉 극값 이탈 = 논거 무효화 손절),
  24시간 변동성 전망(청산가 거리 경고), 청산 캐스케이드 버스트(역방향 감축). 증거신호·레짐·돌파는 안 쓴다.
- 고정 bp 손절 없음(극점 진입 손절 격자 0/30). 손절은 생존·논거 무효화·시간 세 겹.
- **가격을 권고하지 않고 시점만 권고한다** -- 시장이 떠난 자리의 스톱을 체결시킨 09-07 결함 재발 방지.
- 어휘는 다른 카드(바닥/천장 발동, 강/중/약)와 겹치지 않게 **포지션 행동**(익절/부분익절/손절/감축/보유)과
  **긴급도**(즉시/권고/참고)를 쓴다(사용자 지정).

산출: data/live/position_exit_advisor_state.json (현재 판정) · ..._ledger.jsonl (매 5분 체크포인트).
ponytail: ARM/TRAIL/시간 상한은 §5-2 스윕(극점 발동 모집단, 무작위 청산 대조군) 전 초기값이다 --
  state.rules.validated=false 로 화면에 그대로 드러낸다. 스윕 후 값만 바꾼다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402
from live_binance_account_20260910 import fetch_account  # noqa: E402

load_dotenv(ROOT / ".env")

LIVE = ROOT / "data" / "live"
STATE = LIVE / "position_exit_advisor_state.json"
LEDGER = LIVE / "position_exit_advisor_ledger.jsonl"
EXTREME_STATE = LIVE / "eth_extreme_detector_state.json"
VOL_STATE = LIVE / "eth_vol_forecast_state.json"
BURST_STATE = LIVE / "liq_burst_state.json"
SYMBOL = "ETHUSDT"                 # ETH 먼저. 다른 코인은 ETH가 자리 잡은 뒤(사용자 결정 4번)
KLINES = "https://fapi.binance.com/fapi/v1/klines"
BAR_SECONDS = 300
WAKE_OFFSET_SEC = 90               # 극점 워커(60초 주기)가 새 봉을 채점한 뒤에 읽는다
RULES = {
    "time_cap_bars": 24,           # 사용자 지정(2시간). 4시간 넘긴 정보는 없다(H48 봉우리 후 붕괴)
    "invalidation_bars": 12,       # 극점 탐지기 라벨 창 그대로 -- 발동 봉 극값을 12봉 안에 깨면 빗나감
    "liq_atr_mult": 3.0,           # 청산가까지 5분 ATR 3배 안이면 생존 경고
    "arm_atr": 2.0,                # 정점 반납 규칙 무장 -- ponytail: 미검증 초기값
    "trail_atr": 1.0,              # 정점에서 이만큼 반납하면 익절 -- ponytail: 미검증 초기값
    "atr_n": 14,                   # 증거신호 모듈과 같은 5분 ATR14
    "validated": False,
}
VERDICT_TONE = {"익절": "good", "부분익절": "good", "손절": "bad", "감축": "warn", "보유": "neutral"}


def log(m: str) -> None:
    print(f"[exit-advisor {datetime.now(timezone.utc):%m-%d %H:%M:%S}Z] {m}", flush=True)


def fetch_klines(limit: int = 600) -> pd.DataFrame | None:
    """완결 봉만. 형성 중인 마지막 봉은 버린다."""
    for k in range(3):
        try:
            r = requests.get(KLINES, params={"symbol": SYMBOL, "interval": "5m", "limit": limit}, timeout=15)
            r.raise_for_status()
            kl = pd.DataFrame(r.json()).iloc[:, :7]
            kl.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
            for c in ("open", "high", "low", "close"):
                kl[c] = kl[c].astype(float)
            kl["ts"] = pd.to_datetime(kl["open_time"], unit="ms", utc=True)
            return kl[kl["close_time"] < int(time.time() * 1000)].reset_index(drop=True)
        except Exception as e:  # noqa: BLE001
            log(f"⚠️klines 실패({k + 1}/3): {type(e).__name__}: {e}")
            time.sleep(2 * (k + 1))
    return None


def atr_pct(kl: pd.DataFrame, n: int) -> float:
    prev = kl["close"].shift(1).fillna(kl["close"])
    tr = pd.concat([kl["high"] - kl["low"], (kl["high"] - prev).abs(), (kl["low"] - prev).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n, min_periods=1).mean().iloc[-1] / kl["close"].iloc[-1])


def read_json(p: Path, max_age_min: float) -> dict:
    """워커 상태 파일. 없거나 낡았으면 빈 dict -- 낡은 신호로 판정하지 않는다."""
    try:
        d = json.loads(p.read_text())
        ts = d.get("updated_utc") or d.get("updated_at")
        age = (time.time() - datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()) / 60
        return d if age <= max_age_min else {}
    except Exception:  # noqa: BLE001
        return {}


def snapshot(pos: dict, kl: pd.DataFrame, extreme: dict, vol: dict, burst: dict, mem: dict) -> dict:
    """포지션 하나의 판정 입력을 스칼라로 모은다. 가격 변동은 원시 비율(레버·명목 곱하지 않음)."""
    s = 1.0 if pos["side"] == "LONG" else -1.0
    e, m = float(pos["entry_price"]), float(pos["mark_price"])
    entry_ts = pd.Timestamp(pos["entry_at"]) if pos.get("entry_at") else None
    held = kl[kl["ts"] >= entry_ts] if entry_ts is not None else kl.iloc[0:0]
    fav = held["high"] if s > 0 else held["low"]
    adv = held["low"] if s > 0 else held["high"]
    u = s * (m - e) / e
    liq = float(pos.get("liquidation_price") or 0.0)
    snap = {
        "side": pos["side"], "s": s, "u": u,
        "mfe": max(float((s * (fav - e) / e).max()) if len(held) else 0.0, u),
        "mae": min(float((s * (adv - e) / e).min()) if len(held) else 0.0, u),
        "atr": atr_pct(kl, RULES["atr_n"]), "n": int(len(held)),
        "liq_dist": (s * (m - liq) / m) if liq > 0 else None,
        "vol_grade": vol.get("grade") if vol.get("available") else None,
        # 연율 % -> 하루 1시그마 비율
        "vol_sigma_1d": (float(vol["rv_fwd_pred"]) / 100 / 365 ** 0.5) if vol.get("rv_fwd_pred") else None,
        "burst_adverse": bool(burst.get("hawkes_active"))
                         and burst.get("crisis_type") == ("LONG_CRISIS" if s > 0 else "SHORT_CRISIS"),
        "latest_close": float(kl["close"].iloc[-1]),
        "extreme_grade": None, "extreme_opposite": False, "entry_call_px": mem.get("entry_call_px"),
    }
    # 극점 탐지기: 최신 봉 판정만 신뢰한다(등급은 최신 봉에만 있다). tone good=바닥 콜, bad=천장 콜.
    if extreme.get("available") and extreme.get("latest_ts_utc"):
        latest_ok = pd.Timestamp(extreme["latest_ts_utc"]) == kl["ts"].iloc[-1]
        call_long = extreme.get("tone") == "good"
        if latest_ok and extreme.get("grade") and not extreme.get("gated_now"):
            snap["extreme_grade"] = extreme["grade"]
            snap["extreme_opposite"] = call_long != (s > 0)
        # 진입 논거: 진입 봉 ±2봉 안 같은 측면 콜의 극값(롱=저점, 숏=고점). 한 번 찾으면 기억한다.
        if snap["entry_call_px"] is None and entry_ts is not None:
            want = "good" if s > 0 else "bad"
            for t, tone in zip(extreme.get("times") or [], extreme.get("history") or []):
                if tone == want and abs((pd.Timestamp(t) - entry_ts).total_seconds()) <= 2 * BAR_SECONDS:
                    bar = kl[kl["ts"] == pd.Timestamp(t)]
                    if len(bar):
                        snap["entry_call_px"] = float(bar["low" if s > 0 else "high"].iloc[0])
                        mem["entry_call_px"], mem["entry_call_ts"] = snap["entry_call_px"], str(t)
    return snap


def decide(x: dict) -> tuple[str, str, str]:
    """(판정, 긴급도, 사유). 위에서 아래로 첫 매치. 가격은 말하지 않는다 -- '지금'만 말한다."""
    a, u, n = x["atr"], x["u"], x["n"]
    bp = lambda v: f"{v * 1e4:+.0f}bp"  # noqa: E731
    # T0 생존
    if x["liq_dist"] is not None and x["liq_dist"] <= RULES["liq_atr_mult"] * a:
        return "감축", "즉시", f"청산가까지 {bp(x['liq_dist'])}: 5분 ATR {RULES['liq_atr_mult']:.0f}배 안"
    if x["liq_dist"] is not None and x["vol_grade"] == "위험" and x["vol_sigma_1d"] \
            and x["liq_dist"] <= x["vol_sigma_1d"]:
        return "감축", "권고", f"24h 변동성 위험 등급: 하루 1σ({bp(x['vol_sigma_1d'])}) 안에 청산가"
    # T1 논거 무효화 · 역풍 · 시간
    if x["entry_call_px"] is not None and n <= RULES["invalidation_bars"] \
            and x["s"] * (x["latest_close"] - x["entry_call_px"]) < 0:
        return "손절", "즉시", "진입 근거였던 극점 봉의 극값을 종가로 이탈: 논거 무효화"
    if x["burst_adverse"] and u < 0:
        return "감축", "권고", f"역방향 청산 캐스케이드 진행 중, 미실현 {bp(u)}"
    if n >= RULES["time_cap_bars"] and u <= 0:
        return "손절", "권고", f"{n}봉 보유에 미실현 {bp(u)}: 시간 상한, 패자는 오래 끌수록 나빠진다"
    # T2 익절
    if x["extreme_opposite"] and u > 0 and x["extreme_grade"] == "강":
        return "익절", "권고", f"반대 방향 극점 탐지(등급 강), 미실현 {bp(u)}"
    if x["extreme_opposite"] and u > 0 and x["extreme_grade"] == "중":
        return "부분익절", "권고", f"반대 방향 극점 탐지(등급 중), 미실현 {bp(u)}"
    if x["mfe"] >= RULES["arm_atr"] * a and (x["mfe"] - u) >= RULES["trail_atr"] * a:
        return "익절", "권고", f"정점 {bp(x['mfe'])}에서 {bp(x['mfe'] - u)} 반납(ATR {RULES['trail_atr']:.1f}배)"
    if n >= RULES["time_cap_bars"] and u > 0:
        return "익절", "참고", f"{n}봉 보유 시간 상한, 미실현 {bp(u)}: 청산 검토"
    return "보유", "-", f"미실현 {bp(u)} · 정점 {bp(x['mfe'])} · 최저 {bp(x['mae'])} · {n}봉"


def write_json(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, default=str))
    os.replace(tmp, p)


async def _account() -> dict:
    async with aiohttp.ClientSession() as s:
        return await fetch_account(s, [SYMBOL])


def cycle(prev: dict) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    acct = asyncio.run(_account())
    kl = fetch_klines()
    if not acct.get("ok") or kl is None or len(kl) < RULES["atr_n"] + 1:
        return {"available": False, "error": acct.get("error") or "klines_failed", "positions": [],
                "rules": RULES, "updated_utc": now}
    extreme, vol, burst = read_json(EXTREME_STATE, 15), read_json(VOL_STATE, 30), read_json(BURST_STATE, 5)
    prev_pos = {p["key"]: p for p in prev.get("positions", [])}
    out = []
    for pos in acct["positions"]:
        if pos["symbol"] != SYMBOL:
            continue
        key = f"{pos['symbol']}:{pos['side']}:{pos.get('entry_at')}"
        old = prev_pos.get(key, {})
        mem = dict(old.get("mem") or {})
        x = snapshot(pos, kl, extreme, vol, burst, mem)
        verdict, urgency, reason = decide(x)
        row = {
            "key": key, "symbol": pos["symbol"], "side": pos["side"], "leverage": pos["leverage"],
            "entry_price": pos["entry_price"], "mark_price": pos["mark_price"], "entry_at": pos.get("entry_at"),
            "entry_at_truncated": pos["symbol"] in (acct.get("trades_truncated") or []),
            "hold_bars": x["n"], "move_bp": round(x["u"] * 1e4, 1), "mfe_bp": round(x["mfe"] * 1e4, 1),
            "mae_bp": round(x["mae"] * 1e4, 1), "atr_bp": round(x["atr"] * 1e4, 1),
            "liq_dist_bp": (round(x["liq_dist"] * 1e4, 1) if x["liq_dist"] is not None else None),
            "verdict": verdict, "urgency": urgency, "reason": reason, "tone": VERDICT_TONE[verdict],
            "since_utc": old.get("since_utc", now) if old.get("verdict") == verdict else now,
            "context": {"extreme_grade": x["extreme_grade"], "extreme_opposite": x["extreme_opposite"],
                        "vol_grade": x["vol_grade"], "burst_adverse": x["burst_adverse"],
                        "entry_call_ts": mem.get("entry_call_ts")},
            "mem": mem,
        }
        out.append(row)
        with LEDGER.open("a") as f:
            f.write(json.dumps({"ts": now, "bar_utc": str(kl["ts"].iloc[-1]), **{k: v for k, v in row.items() if k != "mem"}},
                               ensure_ascii=False, default=str) + "\n")
    return {"available": True, "error": None, "updated_utc": now, "latest_bar_utc": str(kl["ts"].iloc[-1]),
            "symbol": SYMBOL, "positions": out, "rules": RULES,
            "inputs": {"extreme": bool(extreme), "vol_forecast": bool(vol), "liq_burst": bool(burst)}}


def sleep_to_next_bar() -> None:
    now = time.time()
    time.sleep(max(1.0, (int(now // BAR_SECONDS) + 1) * BAR_SECONDS + WAKE_OFFSET_SEC - now))


def selftest() -> None:
    base = {"side": "LONG", "s": 1.0, "u": 0.001, "mfe": 0.001, "mae": -0.0005, "atr": 0.002, "n": 3,
            "liq_dist": 0.05, "vol_grade": "안정", "vol_sigma_1d": 0.03, "burst_adverse": False,
            "latest_close": 100.0, "extreme_grade": None, "extreme_opposite": False, "entry_call_px": None}
    assert decide(base)[0] == "보유"
    assert decide({**base, "liq_dist": 0.005})[:2] == ("감축", "즉시")                       # T0 3ATR
    assert decide({**base, "vol_grade": "위험", "liq_dist": 0.02})[:2] == ("감축", "권고")    # T0 24h σ
    assert decide({**base, "entry_call_px": 100.5})[0] == "손절"                           # 논거 무효화(롱 저점 이탈)
    assert decide({**base, "entry_call_px": 100.5, "n": 13})[0] == "보유"                  # 12봉 지나면 안 본다
    assert decide({**base, "s": -1.0, "side": "SHORT", "entry_call_px": 99.5})[0] == "손절"  # 숏 고점 이탈
    assert decide({**base, "burst_adverse": True, "u": -0.001})[0] == "감축"
    assert decide({**base, "burst_adverse": True})[0] == "보유"                             # 이익 중엔 안 건다
    assert decide({**base, "n": 24, "u": -0.001})[:2] == ("손절", "권고")                   # 시간 손절
    assert decide({**base, "n": 24})[:2] == ("익절", "참고")                                # 시간 익절 검토
    assert decide({**base, "extreme_opposite": True, "extreme_grade": "강"})[0] == "익절"
    assert decide({**base, "extreme_opposite": True, "extreme_grade": "중"})[0] == "부분익절"
    assert decide({**base, "extreme_opposite": True, "extreme_grade": "약"})[0] == "보유"
    assert decide({**base, "extreme_opposite": True, "extreme_grade": "강", "u": -0.001})[0] == "보유"  # 손실 중 익절 없음
    assert decide({**base, "mfe": 0.005, "u": 0.002})[0] == "익절"                          # 정점 반납 2.5ATR→1.5ATR 반납
    assert decide({**base, "mfe": 0.005, "u": 0.0045})[0] == "보유"                         # 반납 0.25ATR
    # snapshot: 헤지 롱·숏 동시 보유, 진입 이후 봉만 MFE/MAE, 진입 봉 극값 기억
    ts = pd.date_range("2026-09-10 00:00", periods=30, freq="5min", tz="UTC")
    kl = pd.DataFrame({"ts": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    kl.loc[20, ["high", "low"]] = [103.0, 98.0]
    ext = {"available": True, "latest_ts_utc": str(ts[-1]), "tone": "bad", "grade": "강", "gated_now": False,
           "times": [str(t) for t in ts], "history": ["good" if i == 20 else "neutral" for i in range(30)]}
    long_ = {"side": "LONG", "entry_price": 100.0, "mark_price": 100.5, "liquidation_price": 90.0,
             "entry_at": str(ts[21])}
    short = {**long_, "side": "SHORT", "liquidation_price": 110.0}
    mem: dict = {}
    xl = snapshot(long_, kl, ext, {}, {}, mem)
    assert xl["n"] == 9 and abs(xl["mfe"] - 0.01) < 1e-9 and abs(xl["mae"] + 0.01) < 1e-9  # 20번 봉(진입 전)은 제외
    assert xl["extreme_opposite"] and xl["extreme_grade"] == "강" and xl["entry_call_px"] == 98.0
    assert mem["entry_call_px"] == 98.0
    xs = snapshot(short, kl, ext, {}, {}, {})
    assert xs["u"] < 0 and not xs["extreme_opposite"] and xs["entry_call_px"] is None
    assert abs(xs["liq_dist"] - (110 - 100.5) / 100.5) < 1e-12
    print("selftest ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    prev = read_json(STATE, 1e9)
    while True:
        try:
            prev = cycle(prev)
            write_json(STATE, prev)
            log(" · ".join(f"{p['side']} {p['verdict']}({p['urgency']}) {p['move_bp']:+.0f}bp" for p in prev["positions"])
                or (prev.get("error") or "포지션 없음"))
        except Exception as e:  # noqa: BLE001 -- 워커는 죽지 않는다
            log(f"⚠️사이클 실패: {type(e).__name__}: {e}")
        if not a.loop:
            return 0
        sleep_to_next_bar()


if __name__ == "__main__":
    raise SystemExit(main())
